"""Build the demo lab queue (simulated fields from held-out TEST patients' real cells)
and the metrics figures shown in the app."""
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.metrics import roc_curve

from smearscan import make_slide, render_field

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"

# (slide id, target parasitemia); patient chosen from test split
QUEUE = [
    ("LAB-0412", 0.0), ("LAB-0413", 0.08), ("LAB-0414", 0.0), ("LAB-0415", 0.02),
    ("LAB-0416", 0.20), ("LAB-0417", 0.0), ("LAB-0418", 0.01), ("LAB-0419", 0.04),
]


def main():
    d = np.load(ROOT / "data" / "cells_96.npz")
    X, y, split, names = d["X"], d["y"], d["split"], d["names"]
    rng = np.random.default_rng(42)
    pts = defaultdict(lambda: ([], []))
    for i in np.where(split == "test")[0]:
        pts[re.match(r"(C\d+)", names[i]).group(1)][int(y[i])].append(i)
    infected = [p for p, (neg, pos) in pts.items() if len(pos) >= 25]
    healthy = [p for p, (neg, pos) in pts.items() if len(pos) == 0 and len(neg) >= 60]
    (ASSETS / "slides").mkdir(parents=True, exist_ok=True)
    meta = []
    for j, (sid, level) in enumerate(QUEUE):
        if level == 0:
            pt = healthy[j % len(healthy)]
            idx_pos = np.array([], dtype=int)
            idx_neg = np.array(pts[pt][0])
        else:
            pt = infected[j % len(infected)]
            # same-patient uninfected cells, topped up from healthy test patients
            pool = np.array([i for h in healthy for i in pts[h][0]])
            own = np.array(pts[pt][0])
            idx_pos = np.array(pts[pt][1])
            idx_neg = np.concatenate([own, rng.choice(pool, max(0, 100 - len(own)), replace=False)])
        n = 60 if level == 0 else 100
        idx, k = make_slide(X, y, idx_pos, idx_neg, n, level, rng)
        field = render_field([X[i] for i in idx], seed=j)
        Image.fromarray(field).save(ASSETS / "slides" / f"{sid}.png", optimize=True)
        meta.append({"id": sid, "source_patient": pt, "cells": int(len(idx)), "true_parasitized": int(k)})
    (ASSETS / "slides" / "ground_truth.json").write_text(json.dumps(meta, indent=2))

    # a few single held-out cells for the upload tab examples
    (ASSETS / "cells").mkdir(exist_ok=True)
    te = np.where(split == "test")[0]
    for lab, nm in [(1, "parasitized"), (0, "uninfected")]:
        for k, i in enumerate(rng.choice(te[y[te] == lab], 2, replace=False)):
            Image.fromarray(X[i]).save(ASSETS / "cells" / f"test_{nm}_{k}.png")

    # metrics figures
    m = json.loads((ROOT / "results" / "metrics.json").read_text())
    s = np.load(ROOT / "results" / "test_scores.npz")
    fpr, tpr, _ = roc_curve(s["y"], s["p"])
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.8))
    ax[0].plot(fpr, tpr, color="#b4232a", lw=2)
    ax[0].plot([0, 1], [0, 1], ls=":", color="#888")
    ax[0].set(xlabel="False positive rate", ylabel="Sensitivity", title=f"ROC, held-out patients (AUC {m['test']['auc']:.3f})")
    cm = m["test"]["at_0.5"]["confusion_matrix"]
    mat = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
    ax[1].imshow(mat, cmap="Reds")
    for (r, c), v in np.ndenumerate(mat):
        ax[1].text(c, r, str(v), ha="center", va="center", fontsize=13, color="white" if v > mat.max() / 2 else "black")
    ax[1].set(xticks=[0, 1], yticks=[0, 1], xticklabels=["uninfected", "parasitized"], yticklabels=["uninfected", "parasitized"],
              xlabel="Predicted", ylabel="True", title="Confusion matrix (threshold 0.5)")
    fig.tight_layout()
    fig.savefig(ASSETS / "metrics.png", dpi=130)
    print("assets written", meta)


if __name__ == "__main__":
    main()
