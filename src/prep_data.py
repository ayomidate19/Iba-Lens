"""Build a patient-grouped train/val/test split of the NIH malaria cell dataset.

Each image name starts with a slide/patient code (e.g. C100P61ThinF_...). All cells
from one patient go to exactly one split, so test metrics measure performance on
patients the model has never seen (no leakage between near-identical cells).
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image

SIZE = 96
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "cell_images"
OUT = ROOT / "data" / "cells_96.npz"
SEED = 7


def patient_of(name: str) -> str:
    m = re.match(r"(C\d+)", name)
    if not m:
        raise ValueError(name)
    return m.group(1)


def main():
    files, labels = [], []
    for label, cls in enumerate(["Uninfected", "Parasitized"]):
        for p in sorted((RAW / cls).glob("*.png")):
            files.append(p)
            labels.append(label)
    patients = sorted({patient_of(p.name) for p in files})
    rng = np.random.default_rng(SEED)
    rng.shuffle(patients)
    n = len(patients)
    split_of = {}
    for i, pt in enumerate(patients):
        split_of[pt] = "train" if i < 0.7 * n else ("val" if i < 0.8 * n else "test")

    X = np.zeros((len(files), SIZE, SIZE, 3), dtype=np.uint8)
    for i, p in enumerate(files):
        X[i] = np.asarray(Image.open(p).convert("RGB").resize((SIZE, SIZE), Image.BILINEAR))
        if i % 5000 == 0:
            print(f"{i}/{len(files)}", file=sys.stderr)
    y = np.array(labels, dtype=np.int64)
    split = np.array([split_of[patient_of(p.name)] for p in files])
    names = np.array([p.name for p in files])
    np.savez_compressed(OUT, X=X, y=y, split=split, names=names)

    summary = {
        "n_images": len(files),
        "n_patients": n,
        "image_size": SIZE,
        "splits": {
            s: {
                "patients": sum(1 for v in split_of.values() if v == s),
                "images": int((split == s).sum()),
                "parasitized": int(((split == s) & (y == 1)).sum()),
                "uninfected": int(((split == s) & (y == 0)).sum()),
            }
            for s in ["train", "val", "test"]
        },
    }
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "split_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
