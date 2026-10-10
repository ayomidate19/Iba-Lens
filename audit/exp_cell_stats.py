"""3(d) per-patient + cluster bootstrap, 3(e) calibration, 3(f) threshold stability, 3(g) baseline, 3(h) speed."""
import time, numpy as np, json
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression
from common import *

X, y, s, n, pt = load()
te, va, tr = s == "test", s == "val", s == "train"
p_all = predict(X[te | va]); 
pte = predict(X[te]); pva = predict(X[va])
yte, yva, ptt, ptv = y[te], y[va], pt[te], pt[va]
tl, th = TH["t_low"], TH["t_high"]
res = {}

# ---- (d) per-patient
pats = sorted(set(ptt))
rows = []
for q in pats:
    m = ptt == q
    pos = yte[m] == 1
    rows.append(dict(pat=q, n=int(m.sum()), npos=int(pos.sum()),
        sens=float((pte[m][pos] >= .5).mean()) if pos.sum() else None,
        fp_rate=float((pte[m][~pos] >= .5).mean()) if (~pos).sum() else None,
        pos_calls_at_thigh=int((pte[m] >= th).sum()), fp_at_thigh=int((pte[m][~pos] >= th).sum()),
        missed_below_tlow=int((pte[m][pos] < tl).sum())))
sens = np.array([r["sens"] for r in rows if r["sens"] is not None and r["npos"] >= 10])
res["per_patient"] = rows
res["per_patient_sens_summary"] = dict(n=len(sens), min=sens.min(), median=float(np.median(sens)), max=sens.max(), n_below_90=int((sens < .9).sum()), n_below_80=int((sens < .8).sum()))
worst = sorted([r for r in rows if r["sens"] is not None and r["npos"] >= 10], key=lambda r: r["sens"])[:5]
res["worst5_patients"] = worst
res["patients_with_zero_pos"] = sum(r["npos"] == 0 for r in rows)
# healthy patients: any unc/pos cells?
res["healthy_patients_fp_cells_at_0.5"] = {r["pat"]: (r["fp_rate"], r["n"]) for r in rows if r["npos"] == 0}

# cluster bootstrap
rng = np.random.default_rng(0)
idx_by = {q: np.where(ptt == q)[0] for q in pats}
B = 1000; auc = []; se = []; sp = []; ppv_h = []
for _ in range(B):
    ch = rng.choice(pats, len(pats), replace=True)
    ix = np.concatenate([idx_by[q] for q in ch])
    yy, pp = yte[ix], pte[ix]
    auc.append(roc_auc_score(yy, pp)); se.append((pp[yy == 1] >= .5).mean()); sp.append((pp[yy == 0] < .5).mean())
    k = pp >= th; ppv_h.append(yy[k].mean())
ci = lambda a: [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]
res["cluster_bootstrap_95CI"] = dict(auc=ci(auc), sens=ci(se), spec=ci(sp), ppv_at_thigh=ci(ppv_h), B=B)

# ---- (e) calibration
bins = np.linspace(0, 1, 11); ece = 0; rel = []
for i in range(10):
    m = (pte >= bins[i]) & (pte < bins[i + 1] + (1e-9 if i == 9 else 0))
    if m.sum():
        rel.append(dict(bin=f"{bins[i]:.1f}-{bins[i+1]:.1f}", n=int(m.sum()), mean_p=float(pte[m].mean()), frac_pos=float(yte[m].mean())))
        ece += m.mean() * abs(pte[m].mean() - yte[m].mean())
res["ECE"] = float(ece); res["reliability"] = rel
res["brier"] = float(((pte - yte) ** 2).mean())
# PPV of likely positive per patient
pp_rows = [(r["pat"], r["pos_calls_at_thigh"], r["fp_at_thigh"]) for r in rows if r["pos_calls_at_thigh"] > 0]
res["ppv_thigh_overall"] = float(yte[pte >= th].mean())
res["patients_with_any_fp_at_thigh"] = sum(1 for _, c, f in pp_rows if f > 0)
res["patients_with_pos_calls"] = len(pp_rows)
ppvs = [1 - f / c for _, c, f in pp_rows]
res["ppv_thigh_per_patient"] = dict(min=min(ppvs), median=float(np.median(ppvs)), n_below_90=sum(v < .9 for v in ppvs))
res["worst_ppv_patients"] = sorted([(q, c, f) for q, c, f in pp_rows], key=lambda t: 1 - t[2] / t[1])[:4]
# prevalence shift: PPV of 'likely positive' at realistic prevalences using cell sens/spec at t_high
sens_h = float((pte[yte == 1] >= th).mean()); spec_h = float((pte[yte == 0] < th).mean())
res["sens_spec_at_thigh"] = [sens_h, spec_h]
res["ppv_vs_prevalence"] = {f"{pr*100:g}%": float(sens_h * pr / (sens_h * pr + (1 - spec_h) * (1 - pr))) for pr in [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.45]}
sens_l = float((pte[yte == 1] >= tl).mean()); spec_l = float((pte[yte == 0] < tl).mean())
res["npv_vs_prevalence_at_tlow"] = {f"{pr*100:g}%": float(spec_l * (1 - pr) / (spec_l * (1 - pr) + (1 - sens_l) * pr)) for pr in [0.01, 0.05, 0.45]}
res["sens_at_tlow_test"] = sens_l

# ---- (f) threshold stability: choose thresholds on random 20-patient subsets of val+... use TEST patients (model never saw) and also val bootstrap
def pick(yy, pp):
    fpr, tpr, thr = roc_curve(yy, pp)
    a = thr[tpr >= .995].max(); b = thr[(1 - fpr) >= .995].min()
    return min(a, b), max(a, b)
stab = []
rng = np.random.default_rng(1)
for _ in range(200):
    sub = rng.choice(pats, 20, replace=False)
    m = np.isin(ptt, sub)
    if yte[m].sum() < 50 or (yte[m] == 0).sum() < 50: continue
    stab.append(pick(yte[m], pte[m]))
stab = np.array(stab)
res["thr_stability_20patient_subsets_of_test"] = dict(n=len(stab), t_low_p5_med_p95=np.percentile(stab[:, 0], [5, 50, 95]), t_high_p5_med_p95=np.percentile(stab[:, 1], [5, 50, 95]), val_chosen=[tl, th])
# val bootstrap by patient
pv = sorted(set(ptv)); iv = {q: np.where(ptv == q)[0] for q in pv}; st2 = []
for _ in range(300):
    ch = rng.choice(pv, len(pv), replace=True); ix = np.concatenate([iv[q] for q in ch]); st2.append(pick(yva[ix], pva[ix]))
st2 = np.array(st2)
res["thr_stability_val_patient_bootstrap"] = dict(t_low_p5_med_p95=np.percentile(st2[:, 0], [5, 50, 95]), t_high_p5_med_p95=np.percentile(st2[:, 1], [5, 50, 95]))
# realised sens at val-chosen t_low on test, and spec at t_high on test, per 20-patient subset
real = []
for _ in range(200):
    sub = rng.choice(pats, 20, replace=False); m = np.isin(ptt, sub)
    if yte[m].sum() < 50: continue
    real.append(((pte[m][yte[m] == 1] >= tl).mean(), (pte[m][yte[m] == 0] < th).mean()))
real = np.array(real)
res["realised_on_test_subsets"] = dict(sens_at_tlow_p5_med_p95=np.percentile(real[:, 0], [5, 50, 95]), spec_at_thigh_p5_med_p95=np.percentile(real[:, 1], [5, 50, 95]), target=0.995)

# ---- (g) baseline: colour stats + histograms
def feats(Xs):
    f = []
    for im in Xs:
        fg = im.sum(-1) > 30
        px = im[fg].astype(np.float32)
        if len(px) < 10: px = np.zeros((10, 3), np.float32)
        h = np.concatenate([np.histogram(px[:, c], bins=16, range=(0, 255))[0] / len(px) for c in range(3)])
        g = px[:, 1]; low = (g < np.percentile(g, 10))
        f.append(np.concatenate([px.mean(0), px.std(0), np.percentile(px, [5, 25, 75, 95], axis=0).ravel(), h,
                                 [fg.mean(), (px[:, 0] - px[:, 1]).mean(), (px[:, 2] / (px[:, 1] + 1)).mean(), low.mean()]]))
    return np.array(f)
t0 = time.time()
Ftr, Fte = feats(X[tr]), feats(X[te])
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier
sc = StandardScaler().fit(Ftr)
lr = LogisticRegression(max_iter=2000).fit(sc.transform(Ftr), y[tr]); pl = lr.predict_proba(sc.transform(Fte))[:, 1]
hg = HistGradientBoostingClassifier(random_state=0).fit(Ftr, y[tr]); ph = hg.predict_proba(Fte)[:, 1]
def sm(p): return dict(auc=roc_auc_score(yte, p), sens=float((p[yte == 1] >= .5).mean()), spec=float((p[yte == 0] < .5).mean()))
res["baseline_logreg_colour"] = sm(pl); res["baseline_histgb_colour"] = sm(ph); res["model"] = sm(pte)
res["baseline_time_s"] = time.time() - t0

# ---- (h) speed (onnx cpu)
import onnxruntime as ort
so = ort.SessionOptions(); 
for thr_ in [1, 4]:
    so.intra_op_num_threads = thr_
    sess = ort.InferenceSession(str(ROOT / "mobile/public/model/iba_mobilenetv2_96.onnx"), so, providers=["CPUExecutionProvider"])
    xb = (X[te][:100].transpose(0, 3, 1, 2) / 255.).astype(np.float32)
    sess.run(None, {"image": xb}); t = []
    for _ in range(10):
        t0 = time.perf_counter(); sess.run(None, {"image": xb}); t.append(time.perf_counter() - t0)
    res[f"onnx_cpu_{thr_}thr_ms_per_100cells_median"] = float(np.median(t) * 1000)
    x1 = xb[:1]; t = []
    for _ in range(50):
        t0 = time.perf_counter(); sess.run(None, {"image": x1}); t.append(time.perf_counter() - t0)
    res[f"onnx_cpu_{thr_}thr_ms_single_cell_median"] = float(np.median(t) * 1000)
res["onnx_size_MB"] = (ROOT / "mobile/public/model/iba_mobilenetv2_96.onnx").stat().st_size / 1e6
res["pt_size_MB"] = (ROOT / "models/malaria_mobilenetv2_96.pt").stat().st_size / 1e6
save("exp_cell_stats.json", res)
print(json.dumps({k: v for k, v in res.items() if k not in ("per_patient", "reliability")}, indent=1, default=float))
