"""Choose triage thresholds on validation patients, then freeze them and report
cell-level metrics on the held-out test patients."""
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

from train import DEV, build_model, predict

ROOT = Path(__file__).resolve().parents[1]
RULE_OUT_SENS = 0.995  # below t_low -> "likely negative"
RULE_IN_SPEC = 0.995  # at/above t_high -> "likely positive"; in between -> "unclear"


def metrics_at(y, p, t):
    pred = (p >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(t),
        "sensitivity": tp / (tp + fn),
        "specificity": tn / (tn + fp),
        "ppv": tp / max(tp + fp, 1),
        "npv": tn / max(tn + fn, 1),
        "accuracy": (tp + tn) / len(y),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def main():
    d = np.load(ROOT / "data" / "cells_96.npz")
    X, y, split = d["X"], d["y"], d["split"]
    model = build_model()
    model.load_state_dict(torch.load(ROOT / "models" / "malaria_mobilenetv2_96.pt", map_location="cpu"))
    model.to(DEV)
    yva, yte = y[split == "val"], y[split == "test"]
    pva, pte = predict(model, X[split == "val"]), predict(model, X[split == "test"])

    fpr, tpr, thr = roc_curve(yva, pva)
    t_low = float(thr[tpr >= RULE_OUT_SENS].max())
    t_high = float(thr[(1 - fpr) >= RULE_IN_SPEC].min())
    t_low, t_high = min(t_low, t_high), max(t_low, t_high)

    band = np.where(pte >= t_high, "likely_positive", np.where(pte < t_low, "likely_negative", "unclear"))
    triage = {
        b: {
            "cells": int((band == b).sum()),
            "share": float((band == b).mean()),
            "truly_parasitized": int(yte[band == b].sum()),
            "truly_uninfected": int((1 - yte[band == b]).sum()),
        }
        for b in ["likely_positive", "unclear", "likely_negative"]
    }
    hist = json.loads((ROOT / "results" / "train_history.json").read_text())
    results = {
        "model": "MobileNetV2 (ImageNet-pretrained), fine-tuned, 96x96 input",
        **hist,
        "val_auc": float(roc_auc_score(yva, pva)),
        "calibration": {"rule_out_val_sensitivity": RULE_OUT_SENS, "rule_in_val_specificity": RULE_IN_SPEC},
        "thresholds": {"t_low": t_low, "t_high": t_high},
        "test": {
            "n_cells": int(len(yte)),
            "n_patients": 40,
            "auc": float(roc_auc_score(yte, pte)),
            "at_0.5": metrics_at(yte, pte, 0.5),
            "at_t_low": metrics_at(yte, pte, t_low),
            "at_t_high": metrics_at(yte, pte, t_high),
        },
        "triage_on_test": triage,
    }
    th_path = ROOT / "models" / "thresholds.json"
    th = json.loads(th_path.read_text()) if th_path.exists() else {}
    th.update(results["thresholds"])
    th_path.write_text(json.dumps(th, indent=2))
    (ROOT / "results" / "metrics.json").write_text(json.dumps(results, indent=2, default=float))
    np.savez(ROOT / "results" / "test_scores.npz", y=yte, p=pte)
    print(json.dumps({k: results[k] for k in ["thresholds", "test", "triage_on_test"]}, indent=1, default=float))


if __name__ == "__main__":
    main()
