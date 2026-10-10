"""3(a) strict slide-level confusion on TEST patients + why 0% slides are never 'likely negative'."""
import numpy as np, collections, json
from common import *
from evaluate_slides import build_slides, run, LEVELS
from smearscan import SmearScan
X, y, s, n, pt = load()
scan = SmearScan()
slides = build_slides(X, y, n, s, "test", 2)
res = run(scan, X, slides)
out = {"by_level": {}}
for L in LEVELS:
    rows = [r for r in res if r["level"] == L]
    c = collections.Counter(r["band"] for r in rows)
    out["by_level"][f"{L*100:g}%"] = dict(n=len(rows), **{k: c[k] for k in ["likely positive", "unclear", "likely negative"]},
        strict_pos_rate=c["likely positive"] / len(rows),
        mean_true=np.mean([r["true_pct"] for r in rows]), mean_est=np.mean([r["parasitemia_pct"] for r in rows]),
        mean_flagged_pos=np.mean([r["flagged_positive"] for r in rows]), mean_flagged_unc=np.mean([r["flagged_unclear"] for r in rows]),
        flagged_pos_dist=np.bincount([r["flagged_positive"] for r in rows]).tolist(),
        pos_plus_unc_min_med_max=[min(r["flagged_positive"] + r["flagged_unclear"] for r in rows), float(np.median([r["flagged_positive"] + r["flagged_unclear"] for r in rows])), max(r["flagged_positive"] + r["flagged_unclear"] for r in rows)])
pos = [r for r in res if r["level"] > 0]
out["strict_sensitivity_likely_positive_all_infected"] = float(np.mean([r["band"] == "likely positive" for r in pos]))
out["strict_sens_levels_le_2pct"] = float(np.mean([r["band"] == "likely positive" for r in pos if r["level"] <= .02]))
out["not_negative_sens_levels_le_2pct"] = float(np.mean([r["band"] != "likely negative" for r in pos if r["level"] <= .02]))
out["k"] = [scan.k_pos, scan.k_neg, scan.fp_rate]
out["identical_to_stored_slide_metrics"] = all(out["by_level"][k]["likely positive"] == json.load(open(ROOT / "results/slide_metrics.json"))["by_level"][k]["likely_positive"] for k in out["by_level"])
# why: cell-level unclear+pos rate for uninfected cells, split healthy-patient vs infected-patient
te = s == "test"; p = predict(X[te]); yt = y[te]; ptt = pt[te]
npos_by = {q: yt[ptt == q].sum() for q in set(ptt)}
healthy = np.array([npos_by[q] == 0 for q in ptt])
neg = yt == 0
for nm, m in [("uninfected_cells_from_healthy_patients", neg & healthy), ("uninfected_cells_from_infected_patients", neg & ~healthy)]:
    out[nm] = dict(n=int(m.sum()), frac_pos=float((p[m] >= scan.t_high).mean()), frac_unc=float(((p[m] >= scan.t_low) & (p[m] < scan.t_high)).mean()), frac_neg=float((p[m] < scan.t_low).mean()))
# simulate 0% slides of 100 cells from pooled negatives: P(pos+unc <= k_neg) analytic by sampling
rng = np.random.default_rng(0)
for nm, pool in [("pooled_all_test_neg", np.where(neg)[0]), ("healthy_patients_only", np.where(neg & healthy)[0]), ("infected_patients_neg_only", np.where(neg & ~healthy)[0])]:
    c = collections.Counter()
    for _ in range(2000):
        ix = rng.choice(pool, 100, replace=False); pp = p[ix]
        pos_ = (pp >= scan.t_high).sum(); unc = ((pp >= scan.t_low) & (pp < scan.t_high)).sum()
        c["likely positive" if pos_ >= scan.k_pos else "likely negative" if pos_ + unc <= scan.k_neg else "unclear"] += 1
    out[f"sim_0pct_100cells_{nm}"] = {k: v / 2000 for k, v in c.items()}
# 60-cell healthy-patient slides like demo
c = collections.Counter(); pool = np.where(neg & healthy)[0]
for _ in range(2000):
    ix = rng.choice(pool, 60, replace=False); pp = p[ix]
    pos_ = (pp >= scan.t_high).sum(); unc = ((pp >= scan.t_low) & (pp < scan.t_high)).sum()
    c["likely positive" if pos_ >= scan.k_pos else "likely negative" if pos_ + unc <= scan.k_neg else "unclear"] += 1
out["sim_0pct_60cells_healthy_only"] = {k: v / 2000 for k, v in c.items()}
# also: single-patient 0% slides: uninfected cells of ONE infected patient (realistic 'slide from one person')
save("exp_slide.json", out)
print(json.dumps(out, indent=1, default=float))
