"""Dump the Python pipeline's per-cell output on the demo slides, used to check on-device parity."""
import json
from pathlib import Path

import numpy as np
from PIL import Image

from smearscan import SmearScan

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "mobile" / "public" / "demo"


def main():
    scan = SmearScan()
    gt = {g["id"]: g for g in json.loads((ROOT / "assets" / "slides" / "ground_truth.json").read_text())}
    ref = {}
    for p in sorted((ROOT / "assets" / "slides").glob("LAB-*.png")):
        field = np.asarray(Image.open(p).convert("RGB"))
        cells, summ = scan.analyse(field, with_cam=False)
        ref[p.stem] = {
            "size": [field.shape[1], field.shape[0]],
            "truth": gt[p.stem],
            "summary": summ,
            "cells": [{"bbox": list(map(int, c.bbox)), "prob": c.prob} for c in cells],
        }
    (OUT / "reference.json").write_text(json.dumps(ref))
    for k, v in ref.items():
        print(k, v["size"], v["summary"]["cells"], v["summary"]["band"], v["summary"]["flagged_positive"])


if __name__ == "__main__":
    main()
