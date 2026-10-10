"""Smear-level pipeline: field image -> cell crops -> per-cell probability -> Grad-CAM
-> parasitemia estimate -> triage band. Used by the app and by evaluate_slides.py."""
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from skimage import filters, measure, morphology

from train import MEAN, STD, build_model

ROOT = Path(__file__).resolve().parents[1]
SIZE = 96
BG = np.array([236, 220, 226], dtype=np.uint8)  # plasma-like slide background for rendered fields


@dataclass
class Cell:
    bbox: tuple  # (row0, col0, row1, col1) in the field image
    crop: np.ndarray  # SIZE x SIZE x 3 uint8, background blacked out like the training data
    prob: float = 0.0
    cam: np.ndarray | None = None


class SmearScan:
    def __init__(self, weights=ROOT / "models" / "malaria_mobilenetv2_96.pt", device="cpu"):
        self.device = torch.device(device)
        self.model = build_model()
        self.model.load_state_dict(torch.load(weights, map_location="cpu"))
        self.model.eval().to(self.device)
        t = json.loads((ROOT / "models" / "thresholds.json").read_text())
        self.t_low, self.t_high = t["t_low"], t["t_high"]
        self.k_pos = t.get("slide_k_pos", 1)
        self.k_neg = t.get("slide_k_neg", 0)
        self.fp_rate = t.get("cell_fp_rate", 0.0)
        # Grad-CAM on the last stride-16 block (6x6 grid at 96px input)
        self._acts = None
        self.model.features[13].register_forward_hook(self._hook)

    def _hook(self, _m, _i, out):
        self._acts = out
        if out.requires_grad:
            out.retain_grad()

    def _tensor(self, crops):
        x = torch.from_numpy(np.stack(crops)).permute(0, 3, 1, 2).float() / 255.0
        return ((x - MEAN) / STD).to(self.device)

    @torch.no_grad()
    def predict(self, crops):
        out = []
        for i in range(0, len(crops), 256):
            out.append(torch.sigmoid(self.model(self._tensor(crops[i : i + 256]))).squeeze(1).cpu())
        return torch.cat(out).numpy() if out else np.zeros(0)

    def gradcam(self, crop):
        x = self._tensor([crop]).requires_grad_(False)
        with torch.enable_grad():
            logit = self.model(x)[0, 0]
            acts = self._acts
            grads = torch.autograd.grad(logit, acts)[0]
        w = grads.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((w * acts).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=(SIZE, SIZE), mode="bilinear", align_corners=False)[0, 0]
        cam = cam.detach().cpu().numpy()
        return cam / cam.max() if cam.max() > 0 else cam

    def band(self, p):
        if p >= self.t_high:
            return "likely positive"
        if p < self.t_low:
            return "likely negative"
        return "unclear"

    def analyse(self, field: np.ndarray, with_cam=True):
        cells = segment(field)
        probs = self.predict([c.crop for c in cells])
        for c, p in zip(cells, probs):
            c.prob = float(p)
            if with_cam and self.band(c.prob) != "likely negative":
                c.cam = self.gradcam(c.crop)
        return cells, summarise(cells, self.t_low, self.t_high, self.k_pos, self.k_neg, self.fp_rate)


def summarise(cells, t_low, t_high, k_pos=1, k_neg=0, fp_rate=0.0):
    probs = np.array([c.prob for c in cells])
    n = len(probs)
    pos = int((probs >= t_high).sum())
    unc = int(((probs >= t_low) & (probs < t_high)).sum())
    # Slide band from counts calibrated on validation slides (evaluate_slides.py).
    # A lab scientist reviews every slide; the band only sets queue order.
    if pos >= k_pos:
        slide_band = "likely positive"
    elif pos + unc <= k_neg:
        slide_band = "likely negative"
    else:
        slide_band = "unclear"
    return {
        "cells": n,
        "flagged_positive": pos,
        "flagged_unclear": unc,
        # subtract the expected false-positive rate seen on uninfected validation slides
        "parasitemia_pct": max(0.0, 100.0 * (pos / n - fp_rate)) if n else 0.0,
        "raw_flagged_pct": 100.0 * pos / n if n else 0.0,
        "parasitemia_upper_pct": 100.0 * (pos + unc) / n if n else 0.0,
        "band": slide_band,
        "max_prob": float(probs.max()) if n else 0.0,
    }


def segment(field: np.ndarray, min_area=900, max_area=40000):
    """Find red blood cells in an RGB field image (cells are darker than background)."""
    img = field[..., :3].astype(np.float32)
    gray = img.mean(-1)
    # also treat pure black (cell-image padding) as background
    t = filters.threshold_otsu(gray[gray > 10]) if (gray > 10).any() else 128
    mask = (gray < t) & (gray > 10)
    mask = morphology.binary_opening(mask, morphology.disk(2))
    mask = morphology.remove_small_holes(mask, 2000)
    labels = measure.label(mask)
    cells = []
    for r in measure.regionprops(labels):
        if not (min_area <= r.area <= max_area):
            continue
        r0, c0, r1, c1 = r.bbox
        crop = field[r0:r1, c0:c1, :3].copy()
        crop[~(labels[r0:r1, c0:c1] == r.label)] = 0
        h, w = crop.shape[:2]
        s = max(h, w)
        sq = np.zeros((s, s, 3), dtype=np.uint8)
        sq[(s - h) // 2 : (s - h) // 2 + h, (s - w) // 2 : (s - w) // 2 + w] = crop
        sq = np.asarray(Image.fromarray(sq).resize((SIZE, SIZE), Image.BILINEAR))
        cells.append(Cell(bbox=(r0, c0, r1, c1), crop=sq))
    cells.sort(key=lambda c: (c.bbox[0] // 60, c.bbox[1]))
    return cells


def render_field(crops, cols=10, cell=96, gap=14, seed=0):
    """Lay real cell crops out as a microscope field: black padding -> slide background,
    small random jitter so it is not a rigid grid."""
    rng = np.random.default_rng(seed)
    rows = int(np.ceil(len(crops) / cols))
    H, W = rows * (cell + gap) + gap, cols * (cell + gap) + gap
    field = np.tile(BG, (H, W, 1))
    for i, c in enumerate(crops):
        r, k = divmod(i, cols)
        y = gap + r * (cell + gap) + int(rng.integers(-4, 5))
        x = gap + k * (cell + gap) + int(rng.integers(-4, 5))
        y, x = max(0, y), max(0, x)
        patch = field[y : y + cell, x : x + cell]
        fg = c.sum(-1) > 30
        patch[fg] = c[fg]
    return field


def make_slide(X, y, idx_pos, idx_neg, n_cells, parasitemia, rng):
    """Sample a simulated slide of n_cells from one patient's cells at a target parasitemia."""
    k = int(round(n_cells * parasitemia))
    k = min(k, len(idx_pos))
    neg = rng.choice(idx_neg, size=n_cells - k, replace=len(idx_neg) < n_cells - k)
    pos = rng.choice(idx_pos, size=k, replace=False) if k else np.array([], dtype=int)
    idx = np.concatenate([pos, neg]).astype(int)
    rng.shuffle(idx)
    return idx, k
