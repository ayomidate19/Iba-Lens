"""Export the trained classifier to ONNX for on-device inference (mobile app).
Input: float32 [N, 3, 96, 96] RGB in 0..1. Output: float32 [N] probability parasitized.
ImageNet normalisation and the sigmoid are baked into the graph."""
import json
import shutil
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn

from train import MEAN, STD, build_model

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "mobile" / "public" / "model"


class Deployable(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m
        self.register_buffer("mean", MEAN.clone())
        self.register_buffer("std", STD.clone())

    def forward(self, x):
        return torch.sigmoid(self.m((x - self.mean) / self.std)).squeeze(1)


def main():
    m = build_model()
    m.load_state_dict(torch.load(ROOT / "models" / "malaria_mobilenetv2_96.pt", map_location="cpu"))
    net = Deployable(m).eval()
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "iba_mobilenetv2_96.onnx"
    x = torch.rand(4, 3, 96, 96)
    torch.onnx.export(net, (x,), str(path), input_names=["image"], output_names=["prob"],
                      dynamic_axes={"image": {0: "n"}, "prob": {0: "n"}}, opset_version=17, dynamo=False)
    # parity check against PyTorch on real held-out cells
    d = np.load(ROOT / "data" / "cells_96.npz")
    X = d["X"][d["split"] == "test"][:256]
    xt = torch.from_numpy(X).permute(0, 3, 1, 2).float() / 255.0
    with torch.no_grad():
        ref = net(xt).numpy()
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    got = sess.run(None, {"image": xt.numpy()})[0]
    print("max |onnx - torch| =", float(np.abs(got - ref).max()), "size MB", path.stat().st_size / 1e6)
    shutil.copy(ROOT / "models" / "thresholds.json", OUT / "thresholds.json")
    metrics = json.loads((ROOT / "results" / "metrics.json").read_text())
    slide = json.loads((ROOT / "results" / "slide_metrics.json").read_text())
    (OUT / "model_card.json").write_text(json.dumps({"metrics": metrics, "slides": slide}, indent=1, default=float))


if __name__ == "__main__":
    main()
