"""Iba Lens: malaria smear triage for lab scientists. Gradio app (Hugging Face Spaces entry point)."""
import base64
import io
import json
import os
import sys
from pathlib import Path

import gradio as gr
import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd
from matplotlib import colormaps
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)  # Gradio serves example/asset files relative to the working directory
sys.path.insert(0, str(ROOT / "src"))
from smearscan import Cell, SmearScan  # noqa: E402

scan = SmearScan()
METRICS = json.loads((ROOT / "results" / "metrics.json").read_text())
SLIDE_METRICS = json.loads((ROOT / "results" / "slide_metrics.json").read_text())
SLIDES = sorted((ROOT / "assets" / "slides").glob("LAB-*.png"))

# design.md palette
INK, MUTED, PAPER = "#1c1917", "#6b6460", "#faf8f5"
HEX = {"likely positive": "#b4232a", "unclear": "#b7791f", "likely negative": "#2f7d50"}
RGB = {k: tuple(int(v[i : i + 2], 16) for i in (1, 3, 5)) for k, v in HEX.items()}
LABEL = {"likely positive": "Likely positive", "unclear": "Unclear", "likely negative": "Likely negative"}
BAND_ORDER = {"likely positive": 0, "unclear": 1, "likely negative": 2}
ADVICE = {
    "likely positive": "Check the boxed cells, then confirm. Parasitemia is an estimate: count on the slide before reporting it.",
    "unclear": "Look closely at the amber cells. If you are still unsure, read more fields or request a repeat smear.",
    "likely negative": "Nothing suspicious in the cells read. Scan the field yourself before you sign it negative.",
}
JET = colormaps["jet"]
try:
    FONT = ImageFont.load_default(size=15)
except TypeError:  # Pillow < 10.1
    FONT = ImageFont.load_default()


# ---------- image rendering ----------

def annotate(field, cells):
    im = Image.fromarray(field).convert("RGB")
    dr = ImageDraw.Draw(im)
    for c in cells:
        band = scan.band(c.prob)
        if band == "likely negative":
            continue
        r0, c0, r1, c1 = c.bbox
        col = RGB[band]
        dr.rounded_rectangle([c0 - 4, r0 - 4, c1 + 4, r1 + 4], radius=8, outline=col, width=4)
        tag = f"{c.prob:.2f}"
        tw = dr.textlength(tag, font=FONT)
        dr.rounded_rectangle([c0 - 4, r0 - 22, c0 + tw + 6, r0 - 4], radius=4, fill=col)
        dr.text((c0 + 1, r0 - 21), tag, fill="white", font=FONT)
    return im


def cam_panel(c):
    crop = Image.fromarray(c.crop).resize((200, 200), Image.LANCZOS)
    heat = Image.fromarray((JET(c.cam)[..., :3] * 255).astype(np.uint8)).resize((200, 200), Image.BILINEAR)
    over = Image.blend(crop, heat, 0.45)
    panel = Image.new("RGB", (412, 200), PAPER)
    panel.paste(crop, (0, 0))
    panel.paste(over, (212, 0))
    return panel


def data_uri(im):
    buf = io.BytesIO()
    im.save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def evidence_html(cells, limit=12):
    flagged = sorted([c for c in cells if c.cam is not None], key=lambda c: -c.prob)
    if not flagged:
        return "<div class='card evidence empty'><p class='muted'>No cells were flagged, so there is no heatmap to show.</p></div>"
    tiles = "".join(
        f"<figure style='--c:{HEX[scan.band(c.prob)]}'><img src='{data_uri(cam_panel(c))}' alt='cell and heatmap'/>"
        f"<figcaption><span>{LABEL[scan.band(c.prob)]}</span><b class='mono'>p = {c.prob:.2f}</b></figcaption></figure>"
        for c in flagged[:limit]
    )
    more = f"<p class='muted'>Showing the {limit} highest-scoring of {len(flagged)} flagged cells.</p>" if len(flagged) > limit else ""
    return f"""<div class='card evidence'><h3>Evidence</h3><p class='muted'>Each flagged cell (left) and a Grad-CAM heatmap of where the model looked (right; red = strongest). Check that it points at the parasite, not the background.</p><div class='tiles'>{tiles}</div>{more}</div>"""


# ---------- HTML pieces ----------

def pill(band):
    return f"<span class='pill' style='--c:{HEX[band]}'>{LABEL[band]}</span>"


def report_html(title, summ):
    band = summ["band"]
    if "single_prob" in summ:
        return f"""
<div class='card report'>
  <div class='report-head'><span class='mono'>{title}</span>{pill(band)}</div>
  <div class='big' style='color:{HEX[band]}'>{summ['single_prob']:.2f}<span class='unit'>probability parasitized</span></div>
  <p class='muted'>Likely positive at ≥ {scan.t_high:.2f}, likely negative below {scan.t_low:.2f}. The heatmap shows which part of the cell drove the score.</p>
  <div class='safety'>Triage aid only. A lab scientist confirms every result.</div>
</div>"""
    return f"""
<div class='card report'>
  <div class='report-head'><span class='mono'>{title}</span>{pill(band)}</div>
  <div class='big' style='color:{HEX[band]}'>{summ['parasitemia_pct']:.1f}%<span class='unit'>estimated parasitemia</span></div>
  <div class='counts'>
    <div><b class='mono'>{summ['cells']}</b><span>cells read</span></div>
    <div><b class='mono' style='color:{HEX['likely positive']}'>{summ['flagged_positive']}</b><span>flagged parasitized</span></div>
    <div><b class='mono' style='color:{HEX['unclear']}'>{summ['flagged_unclear']}</b><span>unclear</span></div>
  </div>
  <p class='advice'>{ADVICE[band]}</p>
  <div class='legend'><i style='--c:{HEX['likely positive']}'></i>likely parasitized <i style='--c:{HEX['unclear']}'></i>unclear, look closely</div>
  <div class='safety'>Triage aid only. A lab scientist confirms every result; Iba Lens never diagnoses, prescribes or doses.</div>
</div>"""


EMPTY_REPORT = "<div class='card report empty'><b>No slide open</b><p class='muted'>Pick a slide from the queue. The most urgent is at the top.</p></div>"


def progress_html(status):
    bands = [q["summ"]["band"] for q in QUEUE.values()]
    done = sum(1 for s in QUEUE if s in status)
    n = len(QUEUE)
    waiting_pos = sum(1 for sid, q in QUEUE.items() if q["summ"]["band"] == "likely positive" and sid not in status)
    return f"""
<div class='strip'>
  <div><b class='mono'>{n}</b> slides today</div>
  <div>{pill('likely positive')} <b class='mono'>{bands.count('likely positive')}</b></div>
  <div>{pill('unclear')} <b class='mono'>{bands.count('unclear')}</b></div>
  <div>{pill('likely negative')} <b class='mono'>{bands.count('likely negative')}</b></div>
  <div class='grow'></div>
  <div class='prog'><span>Reviewed <b class='mono'>{done}/{n}</b>{' · ' + str(waiting_pos) + ' likely positive waiting' if waiting_pos else ''}</span>
    <div class='bar'><div style='width:{100 * done / n:.0f}%'></div></div></div>
</div>"""


t = METRICS["test"]
tri = METRICS["triage_on_test"]
pos_ppv = tri["likely_positive"]["truly_parasitized"] / tri["likely_positive"]["cells"]
neg0 = SLIDE_METRICS["by_level"]["0%"]
hi5 = sum(SLIDE_METRICS["by_level"][k]["likely_positive"] for k in ["5%", "10%", "20%"])
n5 = sum(SLIDE_METRICS["by_level"][k]["slides"] for k in ["5%", "10%", "20%"])

HERO = f"""
<div class='hero'>
  <div class='brand'>
    <div class='mark'><svg viewBox='0 0 40 40' width='40' height='40'><circle cx='20' cy='20' r='17' fill='none' stroke='{INK}' stroke-width='2.5'/><circle cx='20' cy='20' r='10' fill='#f3d9db'/><circle cx='24' cy='16' r='3.4' fill='{HEX['likely positive']}'/></svg></div>
    <div>
      <h1>Iba Lens</h1>
      <p class='tag'>AI triage for malaria blood smears. It reads every cell, shows its evidence and puts the most urgent slides at the top of the queue, so Nigeria's stretched lab scientists read them first.</p>
      <p class='safety-line'>Triage aid only. A lab scientist confirms every result. It never diagnoses, prescribes or doses.</p>
    </div>
  </div>
  <div class='stats'>
    <div class='stat'><b class='mono'>{t['auc']:.3f}</b><span>AUC on {t['n_patients']} patients the model never saw</span></div>
    <div class='stat'><b class='mono'>{pos_ppv:.1%}</b><span>of “likely positive” cells truly parasitized</span></div>
    <div class='stat'><b class='mono'>{hi5}/{n5}</b><span>slides at ≥5% parasitemia flagged; 0/{neg0['slides']} uninfected called positive</span></div>
  </div>
</div>"""


def metrics_html():
    a, lo, hi = t["at_0.5"], t["at_t_low"], t["at_t_high"]
    cm = a["confusion_matrix"]
    s = SLIDE_METRICS

    def row(name, m):
        return f"<tr><td>{name}</td><td class='mono'>{m['threshold']:.3f}</td><td class='mono'>{m['sensitivity']:.1%}</td><td class='mono'>{m['specificity']:.1%}</td><td class='mono'>{m['ppv']:.1%}</td><td class='mono'>{m['npv']:.1%}</td></tr>"

    slide_rows = "".join(
        f"<tr><td class='mono'>{k}</td><td class='mono'>{v['likely_positive']}</td><td class='mono'>{v['unclear']}</td><td class='mono'>{v['likely_negative']}</td><td class='mono'>{v['est_pct_mean']:.1f}%</td></tr>"
        for k, v in s["by_level"].items()
    )
    return f"""
<div class='pipeline'>
  <div><b>1 · Segment</b><span>find each red cell in the field</span></div>
  <div><b>2 · Classify</b><span>MobileNetV2 scores every cell</span></div>
  <div><b>3 · Explain</b><span>Grad-CAM heatmap per flagged cell</span></div>
  <div><b>4 · Triage</b><span>parasitemia + slide band, queue order</span></div>
  <div class='human'><b>5 · Scientist signs</b><span>confirm, reject or repeat</span></div>
</div>

<div class='kpis'>
  <div class='stat'><b class='mono'>{t['auc']:.3f}</b><span>AUC, cells from 40 unseen patients</span></div>
  <div class='stat'><b class='mono'>{a['sensitivity']:.1%}</b><span>sensitivity at 0.5</span></div>
  <div class='stat'><b class='mono'>{a['specificity']:.1%}</b><span>specificity at 0.5</span></div>
  <div class='stat'><b class='mono'>{s['parasitemia_mae_pct_points']:.1f} pts</b><span>mean parasitemia error per slide</span></div>
</div>

<div class='card'>
<h3>Cell classifier · {t['n_cells']:,} cells from 40 held-out patients</h3>
<p class='muted'>NIH/NLM malaria cell dataset (27,558 cells, 200 patients), split <b>by patient</b> 140 / 20 / 40, so no one's cells appear in both training and test. MobileNetV2 (ImageNet-pretrained), fine-tuned at 96×96 px. Triage thresholds were chosen on validation patients (99.5% rule-out sensitivity, 99.5% rule-in specificity) and frozen before testing.</p>
<table>
<thead><tr><th>Operating point</th><th>Threshold</th><th>Sensitivity</th><th>Specificity</th><th>PPV</th><th>NPV</th></tr></thead>
<tbody>{row('Default', a)}{row('Rule-out (below → likely negative)', lo)}{row('Rule-in (above → likely positive)', hi)}</tbody>
</table>
<p class='muted'>Confusion matrix at 0.5: TN {cm['tn']:,} · FP {cm['fp']} · FN {cm['fn']} · TP {cm['tp']:,}. Test-cell triage: {tri['likely_positive']['cells']:,} likely positive ({pos_ppv:.1%} truly parasitized) · {tri['unclear']['cells']} unclear · {tri['likely_negative']['cells']:,} likely negative ({tri['likely_negative']['truly_parasitized']} missed).</p>
</div>
<!--SPLIT-->
<div class='card'>
<h3>Slide level · full image pipeline</h3>
<p class='muted'>The public data has single cells, not labelled whole smears. Each test is a simulated field: 100 real cells from held-out patients at a known parasitemia, run through segment → classify → count, the same path a real image takes. 40 slides per level; slide thresholds calibrated on validation patients.</p>
<table>
<thead><tr><th>True parasitemia</th><th>Likely positive</th><th>Unclear</th><th>Likely negative</th><th>Mean estimate</th></tr></thead>
<tbody>{slide_rows}</tbody>
</table>
<p class='muted'>At 1% parasitemia a 100-cell field holds one infected cell. The slide rule therefore won't call a slide negative on that little evidence: uninfected slides go to <i>unclear</i> for the scientist, never to <i>likely positive</i>. It speeds up the obvious positives and never quietly clears a borderline slide.</p>
</div>

<div class='card limits'>
<h3>Limits: read before trusting it</h3>
<ul>
<li>One training dataset: Giemsa thin smears, <i>P. falciparum</i>, photographed in Bangladesh. Nigerian stains, microscopes and cameras need a local validation study before any clinical use.</li>
<li>Queue fields are simulated from real held-out cells. Real smears have touching cells that this simple segmenter would merge.</li>
<li>No species or stage, no thick films. It doesn't replace microscopy, RDTs or clinical judgement.</li>
</ul>
</div>"""


# ---------- data ----------

def analyse_image(img):
    field = np.asarray(img.convert("RGB"))
    if field.shape[0] < 140 and field.shape[1] < 140:  # a single cell crop
        crop = np.asarray(Image.fromarray(field).resize((96, 96), Image.BILINEAR))
        c = Cell(bbox=(0, 0, 96, 96), crop=crop)
        c.prob = float(scan.predict([crop])[0])
        c.cam = scan.gradcam(crop)
        summ = {"band": scan.band(c.prob), "single_prob": c.prob}
        return field, [c], summ
    cells, summ = scan.analyse(field)
    return field, cells, summ


QUEUE = {}
for p in SLIDES:  # simulated fields from held-out test patients, analysed once at startup
    f = np.asarray(Image.open(p).convert("RGB"))
    cells, summ = scan.analyse(f)
    QUEUE[p.stem] = {"field": f, "cells": cells, "summ": summ}

HEADERS = ["Slide", "AI triage", "Parasitemia", "Flagged", "Unclear", "Cells", "Scientist"]


def ordered(status):
    ids = list(QUEUE)
    return sorted(ids, key=lambda sid: (sid in status, BAND_ORDER[QUEUE[sid]["summ"]["band"]], -QUEUE[sid]["summ"]["parasitemia_pct"], -QUEUE[sid]["summ"]["flagged_unclear"]))


def queue_table(status):
    rows, bands = [], []
    for sid in ordered(status):
        s = QUEUE[sid]["summ"]
        bands.append(s["band"])
        rows.append([sid, "● " + LABEL[s["band"]], f"{s['parasitemia_pct']:.1f}%", s["flagged_positive"], s["flagged_unclear"], s["cells"], status.get(sid, "Awaiting review")])
    df = pd.DataFrame(rows, columns=HEADERS)

    def style_row(r):
        band = bands[r.name]
        done = r["Scientist"] != "Awaiting review"
        out = []
        for col in df.columns:
            css = "color:#9a938d;" if done else ""
            if col == "AI triage" and not done:
                css += f"color:{HEX[band]};font-weight:600;"
            if col == "Scientist" and done:
                css += "color:#2f7d50;font-weight:600;" if "negative" in r["Scientist"] else ("color:#b4232a;font-weight:600;" if "positive" in r["Scientist"] else "color:#b7791f;font-weight:600;")
            out.append(css)
        return out

    return df.style.apply(style_row, axis=1)


def open_slide(status, evt: gr.SelectData):
    sid = ordered(status)[evt.index[0]]
    return show(sid)


def show(sid):
    q = QUEUE[sid]
    return sid, annotate(q["field"], q["cells"]), report_html(sid, q["summ"]), evidence_html(q["cells"])


def open_first():
    return show(ordered({})[0])


def set_status(sid, status, label):
    if not sid:
        return (status, queue_table(status), progress_html(status), "<p class='note'>Open a slide from the queue first.</p>", gr.skip(), gr.skip(), gr.skip(), gr.skip())
    status = {**status, sid: label}
    msg = f"<p class='note'><span class='mono'>{sid}</span> → <b>{label}</b>, recorded by the lab scientist."
    waiting = [s for s in ordered(status) if s not in status]
    if waiting:  # move straight on to the next most urgent slide
        msg += f" Next up: <span class='mono'>{waiting[0]}</span>.</p>"
        return (status, queue_table(status), progress_html(status), msg, *show(waiting[0]))
    msg += " Queue complete.</p>"
    return (status, queue_table(status), progress_html(status), msg, gr.skip(), gr.skip(), gr.skip(), gr.skip())


def run_upload(img):
    if img is None:
        return None, "<div class='card report empty'><b>No image yet</b><p class='muted'>Upload a thin-smear field photo or a single cell crop, or pick a held-out example below.</p></div>", ""
    field, cells, summ = analyse_image(img)
    if "single_prob" in summ:  # one cell: show the cell and its heatmap large instead of a tiny boxed crop
        return cam_panel(cells[0]).resize((824, 400), Image.LANCZOS), report_html("Single cell", summ), ""
    return annotate(field, cells), report_html("Uploaded field", summ), evidence_html(cells)


# ---------- theme + layout ----------

THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.red,
    neutral_hue=gr.themes.colors.stone,
    font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("IBM Plex Mono"), "ui-monospace", "monospace"],
    radius_size=gr.themes.sizes.radius_md,
).set(
    body_background_fill=PAPER,
    body_text_color=INK,
    block_background_fill="#ffffff",
    block_border_color="#e7e2dc",
    block_label_text_color=MUTED,
    block_title_text_color=MUTED,
    button_primary_background_fill=HEX["likely positive"],
    button_primary_background_fill_hover="#951c22",
    button_primary_text_color="#ffffff",
    button_secondary_background_fill="#ffffff",
    button_secondary_border_color="#d6cfc7",
    button_secondary_text_color=INK,
    shadow_drop="none",
    block_shadow="none",
)

CSS = """
footer {opacity:.5}
.mono {font-family: 'IBM Plex Mono', ui-monospace, monospace}
.muted {color:#6b6460}
.hero {display:grid; grid-template-columns: 1.25fr 1fr; gap:28px; align-items:end; padding:18px 2px 22px; border-bottom:1px solid #e7e2dc; margin-bottom:6px}
.brand {display:flex; gap:16px; align-items:flex-start}
.hero > *, .brand > div, .strip > * {min-width:0}
.brand h1 {font-size:34px; letter-spacing:-.02em; margin:0 0 6px; font-weight:700; color:#1c1917}
.tag {font-size:16px; line-height:1.5; color:#3a3532; margin:0 0 10px; max-width:52ch}
.safety-line {font-size:13px; color:#6b6460; margin:0; padding-left:10px; border-left:3px solid #b4232a}
.stats, .kpis {display:grid; grid-template-columns: repeat(3, 1fr); gap:10px}
.kpis {grid-template-columns: repeat(4, 1fr); margin:14px 0}
.stat {background:#fff; border:1px solid #e7e2dc; border-radius:10px; padding:12px 14px}
.stat b {display:block; font-size:24px; font-weight:600; color:#1c1917; letter-spacing:-.01em}
.stat span {display:block; font-size:12.5px; color:#6b6460; line-height:1.35; margin-top:4px}
.pill {display:inline-flex; align-items:center; gap:6px; font-size:12.5px; font-weight:600; color:var(--c); background:color-mix(in srgb, var(--c) 10%, white); border:1px solid color-mix(in srgb, var(--c) 30%, white); padding:2px 10px; border-radius:999px; white-space:nowrap}
.pill:before {content:''; width:7px; height:7px; border-radius:50%; background:var(--c)}
.strip {display:flex; flex-wrap:wrap; gap:18px; align-items:center; padding:12px 16px; background:#fff; border:1px solid #e7e2dc; border-radius:10px; font-size:14px}
.strip .grow {flex:1}
.prog {min-width:220px; font-size:13px; color:#6b6460}
.bar {height:6px; background:#efeae4; border-radius:6px; margin-top:6px; overflow:hidden}
.bar div {height:100%; background:#2f7d50; border-radius:6px; transition: width .3s}
.card {background:#fff; border:1px solid #e7e2dc; border-radius:12px; padding:18px 20px; margin-bottom:12px}
.card h3 {margin:0 0 6px; font-size:17px}
.report-head {display:flex; justify-content:space-between; align-items:center; font-size:15px; font-weight:600}
.big {font-family:'IBM Plex Mono', monospace; font-size:48px; font-weight:600; letter-spacing:-.03em; margin:10px 0 4px; line-height:1}
.big .unit {display:block; font-family:Inter, sans-serif; font-size:13px; font-weight:500; letter-spacing:0; color:#6b6460; margin-top:8px}
.counts {display:grid; grid-template-columns:repeat(3,1fr); gap:8px; margin:14px 0; padding:12px 0; border-top:1px solid #efeae4; border-bottom:1px solid #efeae4}
.counts b {display:block; font-size:22px}
.counts span {font-size:12px; color:#6b6460}
.advice {font-size:14px; line-height:1.5; margin:8px 0}
.legend {font-size:12.5px; color:#6b6460; display:flex; gap:8px; align-items:center; flex-wrap:wrap}
.legend i {display:inline-block; width:14px; height:14px; border:3px solid var(--c); border-radius:4px; margin-left:6px}
.safety {margin-top:14px; font-size:12.5px; color:#6b6460; background:#faf8f5; border-radius:8px; padding:8px 10px}
.evidence .tiles {display:grid; grid-template-columns:repeat(auto-fill, minmax(min(240px, 45%), 1fr)); gap:10px; margin-top:10px}
.evidence figure {margin:0; border:1px solid #e7e2dc; border-top:3px solid var(--c); border-radius:8px; overflow:hidden; background:#faf8f5}
.evidence img {display:block; width:100%; height:auto}
.evidence figcaption {display:flex; justify-content:space-between; padding:6px 10px; font-size:12.5px; color:var(--c); font-weight:600}
.evidence figcaption b {color:#1c1917; font-weight:500}
.report.empty {text-align:center; padding:40px 20px; color:#6b6460}
.note {font-size:14px; margin:4px 2px}
.pipeline {display:grid; grid-template-columns:repeat(5,1fr); gap:8px; margin-top:6px}
.pipeline div {background:#fff; border:1px solid #e7e2dc; border-radius:10px; padding:10px 12px; font-size:13px}
.pipeline b {display:block; margin-bottom:2px}
.pipeline span {color:#6b6460}
.pipeline .human {border-color:#b4232a; background:#fdf6f6}
.card table {width:100%; border-collapse:collapse; margin:10px 0; font-size:14px}
.card th {text-align:left; font-weight:600; color:#6b6460; font-size:12.5px; border-bottom:1px solid #e7e2dc; padding:6px 8px}
.card table, .card th, .card td {border-left:none !important; border-right:none !important; border-top:none !important}
.card td {border-bottom:1px solid #f1ece6 !important; padding:7px 8px}
.card th {border-bottom:1px solid #e7e2dc !important}
.limits ul {margin:6px 0 0 18px; padding:0; line-height:1.6; font-size:14px}
#queue table {font-family: Inter, sans-serif !important; font-size:14px}
#queue td:first-child {font-family:'IBM Plex Mono', monospace !important}
@media (max-width: 820px) {
  #queue {max-width: calc(100vw - 24px); overflow-x:auto}
  .brand .mark {display:none}
  .brand h1 {font-size:28px}
  .hero {grid-template-columns:1fr}
  .stats {grid-template-columns:1fr}
  .stat {display:flex; gap:12px; align-items:baseline}
  .stat span {margin-top:0}
  .strip {gap:10px}
  .stat b {font-size:19px}
  .kpis {grid-template-columns:1fr 1fr}
  .pipeline {grid-template-columns:1fr 1fr}
  .big {font-size:40px}
}
"""

HEAD = """<style>
.gradio-container {max-width: 1200px !important; margin: 0 auto !important; background: #faf8f5 !important; box-sizing: border-box !important; min-width: 0 !important}
@media (max-width: 820px) {
  .gradio-container {width: 100% !important; max-width: 100vw !important; padding-left: 10px !important; padding-right: 10px !important}
  html, body {overflow-x: hidden}
}
</style>"""

FORCE_LIGHT = """
() => {
  const u = new URL(window.location);
  if (u.searchParams.get('__theme') !== 'light') { u.searchParams.set('__theme', 'light'); window.location.replace(u); }
}
"""

with gr.Blocks(title="Iba Lens · malaria smear triage") as demo:
    gr.HTML(HERO)
    status = gr.State({})
    current = gr.State("")
    with gr.Tab("Queue"):
        progress = gr.HTML(progress_html({}))
        queue = gr.Dataframe(value=queue_table({}), interactive=False, elem_id="queue", show_label=False,
                             column_widths=["12%", "18%", "13%", "10%", "10%", "9%", "28%"])
        gr.Markdown("<span class='muted'>Click a row to open the slide. Reviewed slides drop to the bottom.</span>")
        with gr.Row(equal_height=False):
            with gr.Column(scale=7):
                field_img = gr.Image(label="Slide field · boxes are flagged cells", type="pil", interactive=False, height=600)
                gallery = gr.HTML()
            with gr.Column(scale=5):
                report = gr.HTML(EMPTY_REPORT)
                with gr.Row():
                    b_pos = gr.Button("Confirm positive", variant="primary")
                    b_neg = gr.Button("Confirm negative")
                b_rep = gr.Button("Request repeat smear", size="sm")
                note = gr.HTML()
        queue.select(open_slide, [status], [current, field_img, report, gallery])
        for b, lab in [(b_pos, "Confirmed positive"), (b_neg, "Confirmed negative"), (b_rep, "Repeat smear requested")]:
            b.click(set_status, [current, status, gr.State(lab)], [status, queue, progress, note, current, field_img, report, gallery])
    with gr.Tab("Analyse"):
        with gr.Row(equal_height=False):
            with gr.Column(scale=5):
                up = gr.Image(label="Thin-smear field photo or single cell crop", type="pil", height=360)
                go = gr.Button("Analyse", variant="primary")
                gr.Examples(
                    [str(p) for p in sorted((ROOT / "assets" / "cells").glob("*.png"))] + ([str(SLIDES[0])] if SLIDES else []),
                    inputs=up,
                    label="Held-out test images (patients never seen in training)",
                )
            with gr.Column(scale=7):
                up_out = gr.Image(label="Result", interactive=False, height=420)
                up_rep = gr.HTML()
                up_gal = gr.HTML()
        go.click(run_upload, [up], [up_out, up_rep, up_gal])
    with gr.Tab("Accuracy"):
        top, bottom = metrics_html().split("<!--SPLIT-->")
        gr.HTML(top)
        gr.Image(str(ROOT / "assets" / "metrics.png"), show_label=False, interactive=False, buttons=[], height=400)
        gr.HTML(bottom)
    demo.load(open_first, None, [current, field_img, report, gallery])

if __name__ == "__main__":
    demo.launch(theme=THEME, css=CSS, js=FORCE_LIGHT, head=HEAD, allowed_paths=[str(ROOT / "assets")])
