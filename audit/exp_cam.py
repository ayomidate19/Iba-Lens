"""3(i) Grad-CAM and occlusion heatmaps vs a stain-intensity proxy for the parasite; pointing game on 100 parasitized test cells."""
import numpy as np, torch, json
from scipy import ndimage as ndi
from common import *
from smearscan import SmearScan
X, y, s, n, pt = load()
scan = SmearScan(device="cpu")
rng = np.random.default_rng(0)
idx = np.where((s == "test") & (y == 1))[0]
p_all = scan.predict(list(X[idx]))
# only cells the model calls positive (heatmap is only shown for those)
cand = idx[p_all >= scan.t_high]
pick = rng.choice(cand, 100, replace=False)

def proxy(im):
    """Parasite proxy: strongly purple/dark stain spots inside the cell: low green channel relative to the cell's median, top 3% darkest-green pixels within cell mask."""
    fg = im.sum(-1) > 30
    g = im[..., 1].astype(float)
    fgi = ndi.binary_erosion(fg, iterations=3)   # avoid dark cell edge
    if fgi.sum() < 50: return None
    thr = np.percentile(g[fgi], 5)
    m = fgi & (g <= thr)
    return m, fgi

def occ(crop, base_p, patch=24, stride=8):
    fgm = crop.sum(-1) > 30; fill = crop[fgm].mean(0) if fgm.any() else np.zeros(3)
    pos = [(a, b) for a in range(0, 96 - patch + 1, stride) for b in range(0, 96 - patch + 1, stride)]
    V = []
    for a, b in pos:
        v = crop.copy(); v[a:a + patch, b:b + patch] = fill; V.append(v)
    pr = scan.predict(V); mp = np.zeros((96, 96)); cnt = np.zeros((96, 96))
    for (a, b), q in zip(pos, pr):
        mp[a:a + patch, b:b + patch] += max(0, base_p - q); cnt[a:a + patch, b:b + patch] += 1
    mp = mp / np.maximum(cnt, 1); mp = ndi.uniform_filter(ndi.uniform_filter(mp, 9), 9)
    return mp / mp.max() if mp.max() > 0 else mp

res = {"gradcam": [], "occlusion": [], "random": [], "center": []}
cover = []; blank_cam = 0
for k in pick:
    im = X[k]; pr = proxy(im)
    if pr is None: continue
    m, fgi = pr; cover.append(m.sum() / fgi.sum())
    cam = scan.gradcam(im); p0 = float(scan.predict([im])[0]); oc = occ(im, p0)
    if cam.max() == 0: blank_cam += 1
    fgmask = im.sum(-1) > 30
    # dilate proxy to be tolerant (parasite blob radius ~6px)
    md = ndi.binary_dilation(m, iterations=6)
    for nm, hm in [("gradcam", cam), ("occlusion", oc)]:
        a = np.unravel_index(np.argmax(hm), hm.shape)
        res[nm].append(dict(hit=bool(md[a]), peak_in_cell=bool(fgmask[a]), pct_mass_in_cell=float(hm[fgmask].sum() / max(hm.sum(), 1e-9)),
                            overlap_top10=float(md[hm >= np.percentile(hm, 90)].mean())))
    # baselines: random in-cell point; cell centroid
    cy, cx = np.argwhere(fgmask).mean(0).astype(int); res["center"].append(dict(hit=bool(md[cy, cx])))
    rr = np.argwhere(fgmask)[rng.integers(0, fgmask.sum())]; res["random"].append(dict(hit=bool(md[tuple(rr)])))
summ = {nm: dict(n=len(v), pointing_game_hit=float(np.mean([r["hit"] for r in v])), **({k2: float(np.mean([r[k2] for r in v])) for k2 in ["peak_in_cell", "pct_mass_in_cell", "overlap_top10"]} if v and "peak_in_cell" in v[0] else {})) for nm, v in res.items()}
summ["proxy_area_fraction_of_cell_mean"] = float(np.mean(cover)); summ["dilated_proxy_note"] = "proxy=darkest 5% green-channel pixels inside eroded cell mask, dilated 6px; random in-cell point is the chance level"
summ["gradcam_blank_maps"] = blank_cam
# correlation of the two maps
cors = []
for k in pick[:40]:
    im = X[k]; cam = scan.gradcam(im); oc = occ(im, float(scan.predict([im])[0]))
    cors.append(np.corrcoef(cam.ravel(), oc.ravel())[0, 1])
summ["gradcam_vs_occlusion_pearson_mean"] = float(np.nanmean(cors))
save("exp_cam.json", summ); print(json.dumps(summ, indent=1))
