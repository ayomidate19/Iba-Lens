"""Near-duplicate cells across patients: nearest train neighbour (cosine on 24x24 grayscale) for each test cell; same-patient control."""
import numpy as np, torch, torch.nn.functional as F
from common import *
X, y, s, n, pt = load()
def emb(Xs):
    t = torch.from_numpy(Xs).permute(0, 3, 1, 2).float().mean(1, keepdim=True)
    t = F.adaptive_avg_pool2d(t, 24).flatten(1); t = t - t.mean(1, keepdim=True); return F.normalize(t, dim=1)
Et, Etr = emb(X[s == "test"]), emb(X[s == "train"])
best = torch.cat([(Et[i:i + 1000] @ Etr.T).max(1).values for i in range(0, len(Et), 1000)]).numpy()
# control: within-train nearest neighbour excluding same image name prefix is costly; use within-test, different cell of same patient
print("test->train nearest cosine: mean %.3f p99 %.4f max %.4f ; frac>0.99: %.5f ; frac>0.995: %.5f" % (best.mean(), np.percentile(best, 99), best.max(), (best > .99).mean(), (best > .995).mean()))
Ev = emb(X[s == "val"]); bv = torch.cat([(Ev[i:i + 1000] @ Etr.T).max(1).values for i in range(0, len(Ev), 1000)]).numpy()
print("val->train: frac>0.99 %.5f" % (bv > .99).mean())
