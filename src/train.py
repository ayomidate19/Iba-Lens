"""Fine-tune an ImageNet MobileNetV2 on malaria cells, calibrate triage thresholds
on the validation patients, and report metrics on held-out test patients."""
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from torchvision import models

ROOT = Path(__file__).resolve().parents[1]
DEV = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
EPOCHS = 6
BATCH = 128
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
torch.manual_seed(7)


def build_model():
    m = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    m.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(m.last_channel, 1))
    return m


def to_tensor(x_uint8):
    x = torch.from_numpy(x_uint8).permute(0, 3, 1, 2).float() / 255.0
    return (x - MEAN) / STD


def augment(x):
    # cells have no canonical orientation: random flips + 90-degree rotations
    if torch.rand(1) < 0.5:
        x = x.flip(3)
    if torch.rand(1) < 0.5:
        x = x.flip(2)
    x = torch.rot90(x, int(torch.randint(0, 4, (1,))), dims=(2, 3))
    # mild per-image brightness/contrast jitter (stain and microscope variation)
    b = (torch.rand(x.size(0), 1, 1, 1) - 0.5) * 0.4
    c = 1 + (torch.rand(x.size(0), 1, 1, 1) - 0.5) * 0.4
    return x * c + b


@torch.no_grad()
def predict(model, X):
    model.eval()
    out = []
    for i in range(0, len(X), 512):
        xb = to_tensor(X[i : i + 512]).to(DEV)
        out.append(torch.sigmoid(model(xb)).squeeze(1).cpu())
    return torch.cat(out).numpy()


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
    Xtr, ytr = X[split == "train"], y[split == "train"]
    Xva, yva = X[split == "val"], y[split == "val"]
    Xte, yte = X[split == "test"], y[split == "test"]

    model = build_model().to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    steps = EPOCHS * int(np.ceil(len(Xtr) / BATCH))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=steps)
    lossf = nn.BCEWithLogitsLoss()
    history = []
    best_auc, best_state = -1, None
    for ep in range(EPOCHS):
        model.train()
        t0 = time.time()
        perm = np.random.default_rng(ep).permutation(len(Xtr))
        tot = 0.0
        for i in range(0, len(perm), BATCH):
            idx = perm[i : i + BATCH]
            xb = augment(to_tensor(Xtr[idx])).to(DEV)
            yb = torch.from_numpy(ytr[idx]).float().to(DEV)
            loss = lossf(model(xb).squeeze(1), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item() * len(idx)
        pva = predict(model, Xva)
        auc = roc_auc_score(yva, pva)
        acc = ((pva >= 0.5) == yva).mean()
        history.append({"epoch": ep + 1, "train_loss": tot / len(Xtr), "val_auc": auc, "val_acc": float(acc)})
        print(f"epoch {ep+1} loss {tot/len(Xtr):.4f} val_auc {auc:.4f} val_acc {acc:.4f} ({time.time()-t0:.0f}s)", flush=True)
        if auc > best_auc:
            best_auc = auc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    pva = predict(model, Xva)
    pte = predict(model, Xte)

    # Triage thresholds, chosen on validation patients only:
    #   t_low  : highest threshold that keeps validation sensitivity >= 98% -> below = "likely negative"
    #   t_high : lowest threshold that keeps validation specificity >= 98% -> above = "likely positive"
    fpr, tpr, thr = roc_curve(yva, pva)
    ok_low = thr[tpr >= 0.98]
    t_low = float(ok_low.max()) if len(ok_low) else 0.5
    ok_high = thr[(1 - fpr) >= 0.98]
    t_high = float(ok_high.min()) if len(ok_high) else 0.5
    t_high = max(t_high, t_low)

    band = np.where(pte >= t_high, "likely_positive", np.where(pte < t_low, "likely_negative", "unclear"))
    triage = {}
    for b in ["likely_positive", "unclear", "likely_negative"]:
        m = band == b
        triage[b] = {
            "cells": int(m.sum()),
            "share": float(m.mean()),
            "truly_parasitized": int(yte[m].sum()),
            "truly_uninfected": int((1 - yte[m]).sum()),
        }

    results = {
        "model": "MobileNetV2 (ImageNet-pretrained), fine-tuned, 96x96 input",
        "epochs": EPOCHS,
        "device": str(DEV),
        "history": history,
        "val_auc": float(roc_auc_score(yva, pva)),
        "test": {
            "n_cells": int(len(yte)),
            "n_patients": 40,
            "auc": float(roc_auc_score(yte, pte)),
            "at_0.5": metrics_at(yte, pte, 0.5),
            "at_t_low": metrics_at(yte, pte, t_low),
            "at_t_high": metrics_at(yte, pte, t_high),
        },
        "thresholds": {"t_low": t_low, "t_high": t_high},
        "triage_on_test": triage,
    }
    (ROOT / "models").mkdir(exist_ok=True)
    torch.save(best_state, ROOT / "models" / "malaria_mobilenetv2_96.pt")
    (ROOT / "models" / "thresholds.json").write_text(json.dumps(results["thresholds"], indent=2))
    (ROOT / "results" / "metrics.json").write_text(json.dumps(results, indent=2, default=float))
    np.savez(ROOT / "results" / "test_scores.npz", y=yte, p=pte)
    print(json.dumps(results["test"], indent=2, default=float))
    print(json.dumps(triage, indent=2))


if __name__ == "__main__":
    main()
