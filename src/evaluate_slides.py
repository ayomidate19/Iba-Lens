"""Slide-level evaluation. Builds simulated microscope fields from real cells of
held-out patients at known parasitemia, runs the full image pipeline
(render -> segment -> classify -> aggregate), calibrates slide thresholds on
validation patients and reports results on test patients."""
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from smearscan import SmearScan, make_slide, render_field

ROOT = Path(__file__).resolve().parents[1]
LEVELS = [0.0, 0.01, 0.02, 0.05, 0.10, 0.20]
N_CELLS = 100
SLIDES_PER_LEVEL = 40


def patient(name):
    return re.match(r"(C\d+)", name).group(1)


def build_slides(X, y, names, split, which, seed):
    rng = np.random.default_rng(seed)
    pts = defaultdict(lambda: ([], []))
    for i in np.where(split == which)[0]:
        pts[patient(names[i])][int(y[i])].append(i)
    infected = [p for p, (neg, pos) in pts.items() if len(pos) >= 20]
    neg_pool = np.array([i for p, (neg, _) in pts.items() for i in neg])
    slides = []
    for level in LEVELS:
        for s in range(SLIDES_PER_LEVEL):
            if level == 0:
                idx_pos = np.array([], dtype=int)
                idx_neg = neg_pool
            else:
                pt = infected[s % len(infected)]
                idx_pos = np.array(pts[pt][1])
                own = np.array(pts[pt][0])
                need = N_CELLS - int(round(N_CELLS * level))
                # same-patient uninfected cells first, topped up from other patients in this split
                idx_neg = own if len(own) >= need else np.concatenate([own, rng.choice(neg_pool, need - len(own), replace=False)])
            idx, k = make_slide(X, y, idx_pos, idx_neg, N_CELLS, level, rng)
            slides.append({"level": level, "idx": idx, "true_pct": 100.0 * y[idx].sum() / len(idx)})
    return slides


def run(scan, X, slides):
    out = []
    for j, s in enumerate(slides):
        field = render_field([X[i] for i in s["idx"]], seed=j)
        cells, summ = scan.analyse(field, with_cam=False)
        out.append({**s, **summ})
    return out


def main():
    d = np.load(ROOT / "data" / "cells_96.npz")
    X, y, split, names = d["X"], d["y"], d["split"], d["names"]
    scan = SmearScan()

    val = run(scan, X, build_slides(X, y, names, split, "val", 1))
    # Slide thresholds on validation: "likely positive" if flagged-positive count >= k_pos,
    # chosen so >= 95% of 0% slides are not called positive; "likely negative" if
    # positive+unclear count <= k_neg, chosen so >= 95% of 1% slides are not called negative.
    neg0 = np.array([v["flagged_positive"] for v in val if v["level"] == 0])
    k_pos = int(np.quantile(neg0, 0.95)) + 1
    low1 = np.array([v["flagged_positive"] + v["flagged_unclear"] for v in val if v["level"] == 0.01])
    k_neg = max(int(np.quantile(low1, 0.05)) - 1, 0)
    k_neg = min(k_neg, k_pos - 1)
    fp_rate = float(np.mean([v["flagged_positive"] / v["cells"] for v in val if v["level"] == 0]))

    th = json.loads((ROOT / "models" / "thresholds.json").read_text())
    th.update({"slide_k_pos": k_pos, "slide_k_neg": k_neg, "cell_fp_rate": fp_rate})
    (ROOT / "models" / "thresholds.json").write_text(json.dumps(th, indent=2))
    scan = SmearScan()

    test = run(scan, X, build_slides(X, y, names, split, "test", 2))
    by_level = {}
    for level in LEVELS:
        rows = [t for t in test if t["level"] == level]
        bands = [t["band"] for t in rows]
        by_level[f"{level*100:g}%"] = {
            "slides": len(rows),
            "likely_positive": bands.count("likely positive"),
            "unclear": bands.count("unclear"),
            "likely_negative": bands.count("likely negative"),
            "cells_found_mean": float(np.mean([t["cells"] for t in rows])),
            "true_pct_mean": float(np.mean([t["true_pct"] for t in rows])),
            "est_pct_mean": float(np.mean([t["parasitemia_pct"] for t in rows])),
            "abs_err_mean": float(np.mean([abs(t["parasitemia_pct"] - t["true_pct"]) for t in rows])),
        }
    pos = [t for t in test if t["level"] > 0]
    neg = [t for t in test if t["level"] == 0]
    result = {
        "setup": f"{SLIDES_PER_LEVEL} simulated {N_CELLS}-cell fields per level from held-out patients' real cells; full image pipeline",
        "slide_thresholds": {"k_pos": k_pos, "k_neg": k_neg},
        "by_level": by_level,
        "slide_sensitivity_not_negative": float(np.mean([t["band"] != "likely negative" for t in pos])),
        "slide_specificity_0pct_not_positive": float(np.mean([t["band"] != "likely positive" for t in neg])),
        "parasitemia_mae_pct_points": float(np.mean([abs(t["parasitemia_pct"] - t["true_pct"]) for t in test])),
    }
    (ROOT / "results" / "slide_metrics.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
