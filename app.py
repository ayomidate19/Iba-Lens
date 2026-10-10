"""Iba Lens: malaria smear triage for lab scientists. Gradio app (Hugging Face Spaces entry point)."""
import json
import os
import sys
from pathlib import Path

import gradio as gr
import matplotlib

matplotlib.use("Agg")
import numpy as np
from matplotlib import colormaps
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)  # Gradio serves example/asset files relative to the working directory
sys.path.insert(0, str(ROOT / "src"))
from smearscan import SmearScan  # noqa: E402

scan = SmearScan()
METRICS = json.loads((ROOT / "results" / "metrics.json").read_text())
SLIDE_METRICS_PATH = ROOT / "results" / "slide_metrics.json"
SLIDE_METRICS = json.loads(SLIDE_METRICS_PATH.read_text()) if SLIDE_METRICS_PATH.exists() else None
SLIDES = sorted((ROOT / "assets" / "slides").glob("LAB-*.png"))

COLORS = {"likely positive": (180, 35, 42), "unclear": (214, 140, 0), "likely negative": (46, 125, 80)}
BAND_ORDER = {"likely positive": 0, "unclear": 1, "likely negative": 2}
JET = colormaps["jet"]


def annotate(field, cells):
    im = Image.fromarray(field).convert("RGB")
    dr = ImageDraw.Draw(im)
    for c in cells:
        band = scan.band(c.prob)
        if band == "likely negative":
            continue
        r0, c0, r1, c1 = c.bbox
        dr.rectangle([c0 - 3, r0 - 3, c1 + 3, r1 + 3], outline=COLORS[band], width=4)
        dr.text((c0, max(r0 - 15, 0)), f"{c.prob:.2f}", fill=COLORS[band])
    return im


def cam_panel(c):
    crop = Image.fromarray(c.crop).resize((192, 192), Image.NEAREST)
    heat = (JET(c.cam)[..., :3] * 255).astype(np.uint8)
    heat = Image.fromarray(heat).resize((192, 192), Image.BILINEAR)
    over = Image.blend(crop, heat, 0.45)
    panel = Image.new("RGB", (392, 192), "white")
    panel.paste(crop, (0, 0))
    panel.paste(over, (200, 0))
    return panel


def analyse_image(field):
    field = np.asarray(Image.fromarray(field).convert("RGB")) if isinstance(field, np.ndarray) else np.asarray(field.convert("RGB"))
    if field.shape[0] < 140 and field.shape[1] < 140:
        # a single cell crop: classify it directly
        crop = np.asarray(Image.fromarray(field).resize((96, 96), Image.BILINEAR))
        from smearscan import Cell

        c = Cell(bbox=(0, 0, 96, 96), crop=crop)
        c.prob = float(scan.predict([crop])[0])
        c.cam = scan.gradcam(crop)
        cells = [c]
        summ = {"cells": 1, "flagged_positive": int(c.prob >= scan.t_high), "flagged_unclear": int(scan.t_low <= c.prob < scan.t_high),
                "parasitemia_pct": 100.0 * (c.prob >= scan.t_high), "band": scan.band(c.prob), "single_prob": c.prob}
    else:
        cells, summ = scan.analyse(field)
    return field, cells, summ


def summary_md(sid, summ):
    color = "#%02x%02x%02x" % COLORS[summ["band"]]
    if "single_prob" in summ:
        return (
            f"### Single cell  <span style='color:{color}'>● {summ['band'].upper()}</span>\n"
            f"Probability parasitized **{summ['single_prob']:.3f}** (likely positive at ≥ {scan.t_high:.2f}, likely negative below {scan.t_low:.2f}).\n\n"
            "The heatmap shows which part of the cell drove the score. A lab scientist confirms every result."
        )
    return (
        f"### {sid}  <span style='color:{color}'>● {summ['band'].upper()}</span>\n"
        f"**{summ['cells']}** cells found · **{summ['flagged_positive']}** flagged parasitized · "
        f"**{summ['flagged_unclear']}** unclear · estimated parasitemia **{summ['parasitemia_pct']:.1f}%**\n\n"
        "Red boxes: likely parasitized. Amber: unclear, look closely. Numbers are model probabilities. "
        "This is a triage aid. A lab scientist confirms every result; Iba Lens does not diagnose, prescribe or dose."
    )


def gallery_items(cells):
    flagged = sorted([c for c in cells if c.cam is not None], key=lambda c: -c.prob)[:12]
    return [(cam_panel(c), f"p={c.prob:.2f} · {scan.band(c.prob)}") for c in flagged]


# Pre-analyse the demo queue at startup (simulated fields from held-out test patients).
QUEUE = {}
for p in SLIDES:
    field = np.asarray(Image.open(p).convert("RGB"))
    cells, summ = scan.analyse(field)
    QUEUE[p.stem] = {"field": field, "cells": cells, "summ": summ}


def queue_rows(status):
    rows = []
    for sid, q in QUEUE.items():
        s = q["summ"]
        rows.append([sid, s["band"], round(s["parasitemia_pct"], 1), s["flagged_positive"], s["flagged_unclear"], s["cells"], status.get(sid, "Awaiting review")])
    rows.sort(key=lambda r: (r[6] != "Awaiting review", BAND_ORDER[r[1]], -r[2], -r[4]))
    return rows


HEADERS = ["Slide", "AI triage", "Est. parasitemia %", "Flagged", "Unclear", "Cells", "Scientist status"]


def open_slide(status, evt: gr.SelectData):
    rows = queue_rows(status)
    sid = rows[evt.index[0]][0]
    q = QUEUE[sid]
    return sid, annotate(q["field"], q["cells"]), summary_md(sid, q["summ"]), gallery_items(q["cells"])


def set_status(sid, status, label):
    if not sid:
        return status, queue_rows(status), "Select a slide from the queue first."
    status = {**status, sid: label}
    return status, queue_rows(status), f"{sid} → **{label}** (recorded by lab scientist)"


def run_upload(img):
    if img is None:
        return None, "Upload a thin-smear field image or a single cell crop.", []
    field, cells, summ = analyse_image(img)
    return annotate(field, cells), summary_md("Uploaded image", summ), gallery_items(cells)


def metrics_md():
    t = METRICS["test"]
    a, lo, hi = t["at_0.5"], t["at_t_low"], t["at_t_high"]
    tri = METRICS["triage_on_test"]
    md = f"""
### Cell classifier, evaluated on 40 patients the model never saw ({t['n_cells']:,} cells)
MobileNetV2 (ImageNet-pretrained), fine-tuned at 96×96 px on the NIH/NLM malaria cell dataset (27,558 cells, 200 patients).
Train / validation / test split **by patient** (140 / 20 / 40), so near-identical cells from one person never leak across splits.

| | Threshold | Sensitivity | Specificity | PPV | NPV |
|---|---|---|---|---|---|
| Default | 0.50 | {a['sensitivity']:.3f} | {a['specificity']:.3f} | {a['ppv']:.3f} | {a['npv']:.3f} |
| Rule-out ("likely negative" below) | {lo['threshold']:.3f} | {lo['sensitivity']:.3f} | {lo['specificity']:.3f} | {lo['ppv']:.3f} | {lo['npv']:.3f} |
| Rule-in ("likely positive" above) | {hi['threshold']:.3f} | {hi['sensitivity']:.3f} | {hi['specificity']:.3f} | {hi['ppv']:.3f} | {hi['npv']:.3f} |

**AUC {t['auc']:.4f}**. Both triage thresholds were chosen on the validation patients (99.5% sensitivity for rule-out, 99.5% specificity for rule-in), then frozen before testing.

Triage of test cells: likely positive {tri['likely_positive']['cells']} ({tri['likely_positive']['truly_parasitized']} truly parasitized) ·
unclear {tri['unclear']['cells']} · likely negative {tri['likely_negative']['cells']} ({tri['likely_negative']['truly_parasitized']} truly parasitized, i.e. missed).
"""
    if SLIDE_METRICS:
        s = SLIDE_METRICS
        md += f"""
### Slide-level check (full image pipeline: segment → classify → count)
{s['setup']}. Slide thresholds calibrated on validation patients.

| True parasitemia | Slides | Likely positive | Unclear | Likely negative | Mean estimate % |
|---|---|---|---|---|---|
""" + "\n".join(
            f"| {k} | {v['slides']} | {v['likely_positive']} | {v['unclear']} | {v['likely_negative']} | {v['est_pct_mean']:.1f} |" for k, v in s["by_level"].items()
        ) + f"""

On 100 cells, 1% parasitemia is a single infected cell, so the slide rule refuses to call a slide "likely negative" unless almost nothing looks suspicious. Uninfected 100-cell slides land in *unclear*, never *likely positive*: the tool saves reading time on the obvious positives and never quietly rules out a borderline slide.

Infected slides not sent to the bottom of the queue: **{s['slide_sensitivity_not_negative']:.1%}** · uninfected slides not called positive: **{s['slide_specificity_0pct_not_positive']:.1%}** · mean parasitemia error **{s['parasitemia_mae_pct_points']:.2f} points**.
"""
    md += """
### Limits (read before trusting it)
- Trained on one public dataset (Giemsa thin smears, *P. falciparum*, photographed in Bangladesh). Nigerian lab images (different stains, microscopes, phone cameras) need local validation before any use.
- The queue fields are *simulated*: real cells from held-out patients laid out on a background. Real smears have touching and overlapping cells that this simple segmenter will merge.
- It does not identify species or stage, and it does not replace microscopy, RDTs or clinical judgement. Every result is confirmed by a lab scientist.
"""
    return md


CSS = """
.gradio-container {max-width: 1180px !important}
#hero h1 {margin-bottom: 0}
"""

with gr.Blocks(title="Iba Lens: malaria smear triage") as demo:
    gr.Markdown(
        "# Iba Lens\n**Malaria smear triage for busy labs.** It reads every cell on a thin-smear image, flags likely parasites with visual evidence, "
        "estimates parasitemia and sorts the queue so the lab scientist reads the most urgent slides first. The scientist confirms every result.",
        elem_id="hero",
    )
    status = gr.State({})
    current = gr.State("")
    with gr.Tab("Lab queue"):
        queue = gr.Dataframe(value=queue_rows({}), headers=HEADERS, interactive=False, wrap=True, label="Today's slides, most urgent first (click a row)")
        with gr.Row():
            with gr.Column(scale=3):
                field_img = gr.Image(label="Slide field (boxes = flagged cells)", type="pil", interactive=False)
            with gr.Column(scale=2):
                summary = gr.Markdown("Select a slide in the queue.")
                with gr.Row():
                    b_pos = gr.Button("Confirm positive", variant="stop")
                    b_neg = gr.Button("Confirm negative")
                    b_rep = gr.Button("Repeat smear")
                note = gr.Markdown()
                gallery = gr.Gallery(label="Flagged cells: cell | Grad-CAM (where the model looked)", columns=2, height=420, object_fit="contain")
        queue.select(open_slide, [status], [current, field_img, summary, gallery])
        for b, lab in [(b_pos, "Confirmed positive"), (b_neg, "Confirmed negative"), (b_rep, "Repeat smear requested")]:
            b.click(set_status, [current, status, gr.State(lab)], [status, queue, note])
    with gr.Tab("Analyse an image"):
        with gr.Row():
            with gr.Column():
                up = gr.Image(label="Thin-smear field or single cell crop", type="pil")
                go = gr.Button("Analyse", variant="primary")
                gr.Examples(
                    [str(p) for p in sorted((ROOT / "assets" / "cells").glob("*.png"))] + [str(SLIDES[0])] if SLIDES else [],
                    inputs=up,
                    label="Held-out test images",
                )
            with gr.Column():
                up_out = gr.Image(label="Result", interactive=False)
                up_sum = gr.Markdown()
                up_gal = gr.Gallery(label="Flagged cells with Grad-CAM", columns=2, height=360, object_fit="contain")
        go.click(run_upload, [up], [up_out, up_sum, up_gal])
    with gr.Tab("How well it works"):
        gr.Markdown(metrics_md())
        if (ROOT / "assets" / "metrics.png").exists():
            gr.Image(str(ROOT / "assets" / "metrics.png"), show_label=False, interactive=False)

if __name__ == "__main__":
    demo.launch(css=CSS, allowed_paths=[str(ROOT / "assets")])
