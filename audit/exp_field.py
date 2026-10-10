"""3(b) harder synthetic fields: how segmentation + parasitemia + band degrade."""
import io, collections, json, numpy as np
from PIL import Image, ImageFilter, ImageDraw
from scipy import ndimage as ndi
from common import *
from evaluate_slides import build_slides
from smearscan import SmearScan, BG
X, y, s, n, pt = load()
scan = SmearScan(device="cpu")
slides_all = build_slides(X, y, n, s, "test", 2)
NS = 12
sel = []
for L in [0.0, 0.05, 0.10]:
    sel += [sl for sl in slides_all if sl["level"] == L][:NS]

def render(idx, seed, cell=96, spacing=110, jitter=4, bg=BG, wbc=0, platelets=0):
    rng = np.random.default_rng(seed)
    cols = 10; rows = int(np.ceil(len(idx) / cols)); pad = spacing // 2
    H, W = rows * spacing + pad * 2, cols * spacing + pad * 2
    f = np.tile(np.array(bg, np.uint8), (H, W, 1))
    for i, k in enumerate(idx):
        r, c = divmod(i, cols)
        crop = X[k] if cell == 96 else np.asarray(Image.fromarray(X[k]).resize((cell, cell), Image.BILINEAR))
        yy = pad + r * spacing + int(rng.integers(-jitter, jitter + 1)); xx = pad + c * spacing + int(rng.integers(-jitter, jitter + 1))
        yy, xx = max(0, yy), max(0, xx)
        patch = f[yy:yy + cell, xx:xx + cell]; h, w = patch.shape[:2]
        fg = crop[:h, :w].sum(-1) > 30
        patch[fg] = crop[:h, :w][fg]
    extras = []
    im = Image.fromarray(f); d = ImageDraw.Draw(im)
    for _ in range(wbc):  # WBC: big purple nucleus with pale halo
        cx, cy = int(rng.integers(100, W - 100)), int(rng.integers(100, H - 100)); R = int(rng.integers(38, 52))
        d.ellipse([cx - R, cy - R, cx + R, cy + R], fill=(120, 95, 150)); d.ellipse([cx - R * .6, cy - R * .6, cx + R * .6, cy + R * .6], fill=(70, 40, 120)); extras.append((cy, cx))
    for _ in range(platelets):
        cx, cy = int(rng.integers(30, W - 30)), int(rng.integers(30, H - 30)); R = int(rng.integers(9, 14))
        d.ellipse([cx - R, cy - R, cx + R, cy + R], fill=(150, 90, 160)); extras.append((cy, cx))
    return np.asarray(im).copy(), extras

def jpeg(f, q):
    b = io.BytesIO(); Image.fromarray(f).save(b, "JPEG", quality=q); return np.asarray(Image.open(b).convert("RGB"))
def illum(f, lo=0.55):
    H, W = f.shape[:2]; yy, xx = np.mgrid[0:H, 0:W]
    g = lo + (1 - lo) * (1 - ((xx / W - .3) ** 2 + (yy / H - .3) ** 2) / .9)
    return np.clip(f * g[..., None], 0, 255).astype(np.uint8)
def noise(f, sd): return np.clip(f + np.random.default_rng(1).normal(0, sd, f.shape), 0, 255).astype(np.uint8)
def blur(f, r): return np.asarray(Image.fromarray(f).filter(ImageFilter.GaussianBlur(r)))
def stain(f, mul): return np.clip(f * np.array(mul), 0, 255).astype(np.uint8)
def rot(f, deg):
    return np.asarray(Image.fromarray(f).rotate(deg, expand=True, fillcolor=tuple(int(v) for v in BG)))

conds = {
 "baseline(render_field-like)": dict(),
 "touching(spacing100,gap4)": dict(kw=dict(spacing=100, jitter=2)),
 "overlapping(spacing86)": dict(kw=dict(spacing=86, jitter=3)),
 "overlapping(spacing76)": dict(kw=dict(spacing=76, jitter=3)),
 "uneven_illum(0.55)": dict(post=lambda f: illum(f, .55)),
 "uneven_illum(0.35)": dict(post=lambda f: illum(f, .35)),
 "noise_sd8": dict(post=lambda f: noise(f, 8)),
 "noise_sd20": dict(post=lambda f: noise(f, 20)),
 "jpeg_q30": dict(post=lambda f: jpeg(f, 30)),
 "jpeg_q10": dict(post=lambda f: jpeg(f, 10)),
 "blur_s2": dict(post=lambda f: blur(f, 2)),
 "blur_s4": dict(post=lambda f: blur(f, 4)),
 "brightness_0.75": dict(post=lambda f: stain(f, [.75] * 3)),
 "brightness_1.1(bg->saturated)": dict(post=lambda f: stain(f, [1.1] * 3)),
 "stain_purple(0.85,0.8,1.1)": dict(post=lambda f: stain(f, [.85, .8, 1.1])),
 "stain_pink-light(1.0,1.1,1.05)": dict(post=lambda f: stain(f, [1.0, 1.1, 1.05])),
 "bg_light(250,245,248)": dict(kw=dict(bg=(250, 245, 248))),
 "bg_dark(190,170,180)": dict(kw=dict(bg=(190, 170, 180))),
 "bg_white_cells_unchanged(255,255,255)": dict(kw=dict(bg=(255, 255, 255))),
 "cell_scale_40px": dict(kw=dict(cell=40, spacing=55, jitter=2)),
 "cell_scale_60px": dict(kw=dict(cell=60, spacing=75, jitter=3)),
 "cell_scale_140px": dict(kw=dict(cell=140, spacing=160, jitter=5)),
 "cell_scale_200px": dict(kw=dict(cell=200, spacing=220, jitter=5)),
 "rotated_30deg": dict(post=lambda f: rot(f, 30)),
 "rotated_90deg": dict(post=lambda f: rot(f, 90)),
 "WBC x3": dict(kw=dict(wbc=3)),
 "platelets x10": dict(kw=dict(platelets=10)),
 "WBC x3 + platelets x10": dict(kw=dict(wbc=3, platelets=10)),
 "combo: illum.55+noise8+jpeg30+blur1": dict(post=lambda f: blur(jpeg(noise(illum(f, .55), 8), 30), 1)),
}
out = {}
for name, c in conds.items():
    rows = []
    for j, sl in enumerate(sel):
        field, extras = render(sl["idx"], j, **c.get("kw", {}))
        if "post" in c: field = c["post"](field)
        cells, summ = scan.analyse(field, with_cam=False)
        # cells matching extras (WBC/platelet) flagged positive?
        extra_hit = 0
        for cell in cells:
            r0, c0, r1, c1 = cell.bbox
            if any(r0 <= ey <= r1 and c0 <= ex <= c1 for ey, ex in extras) and scan.band(cell.prob) != "likely negative": extra_hit += 1
        rows.append(dict(level=sl["level"], true_cells=len(sl["idx"]), found=summ["cells"], est=summ["parasitemia_pct"], true=sl["true_pct"], band=summ["band"], pos=summ["flagged_positive"], unc=summ["flagged_unclear"], extra_flagged=extra_hit, n_extras=len(extras)))
    o = {}
    for L in [0.0, 0.05, 0.10]:
        r = [q for q in rows if q["level"] == L]
        o[f"{L*100:g}%"] = dict(found_mean=np.mean([q["found"] for q in r]), found_min=min(q["found"] for q in r), found_max=max(q["found"] for q in r), est_mean=np.mean([q["est"] for q in r]), true_mean=np.mean([q["true"] for q in r]),
            bands=dict(collections.Counter(q["band"] for q in r)), pos_mean=np.mean([q["pos"] for q in r]), unc_mean=np.mean([q["unc"] for q in r]), extras_flagged_mean=np.mean([q["extra_flagged"] for q in r]), n_extras=r[0]["n_extras"])
    out[name] = o
    print(name, {k: (round(v["found_mean"], 1), round(v["est_mean"], 2), v["bands"], round(v["extras_flagged_mean"], 1)) for k, v in o.items()}, flush=True)
save("exp_field.json", out)
