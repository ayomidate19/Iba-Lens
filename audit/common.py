import json, re, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import torch
from train import build_model, predict as _predict, DEV, to_tensor

def load():
    d = np.load(ROOT / "data" / "cells_96.npz")
    X, y, s, n = d["X"], d["y"], d["split"], d["names"]
    pt = np.array([re.match(r"(C\d+)", x).group(1) for x in n])
    return X, y, s, n, pt

_model = None
def model():
    global _model
    if _model is None:
        m = build_model(); m.load_state_dict(torch.load(ROOT / "models/malaria_mobilenetv2_96.pt", map_location="cpu")); m.to(DEV).eval(); _model = m
    return _model

def predict(X):
    return _predict(model(), X)

TH = json.loads((ROOT / "models/thresholds.json").read_text())
OUT = ROOT / "audit"
def save(name, obj):
    (OUT / name).write_text(json.dumps(obj, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o)))
