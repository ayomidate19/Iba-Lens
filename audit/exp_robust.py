"""3(c) cell-level robustness of classifier alone on test cells."""
import io, numpy as np
from PIL import Image, ImageFilter, ImageEnhance
from sklearn.metrics import roc_auc_score
from common import *
X, y, s, n, pt = load(); te = s == "test"; Xt, yt = X[te], y[te]
tl, th = TH["t_low"], TH["t_high"]
def per(f): return np.stack([np.asarray(f(Image.fromarray(a))) for a in Xt])
def jpeg(q):
    def f(im):
        b = io.BytesIO(); im.save(b, "JPEG", quality=q); return Image.open(b).convert("RGB")
    return f
def hue(deg):
    def f(im):
        h = np.asarray(im.convert("HSV")).copy(); h[..., 0] = (h[..., 0].astype(int) + int(deg / 360 * 256)) % 256
        return Image.fromarray(h, "HSV").convert("RGB")
    return f
def keepbg(Xa, Xb):  # re-black the padding so only cell pixels change
    m = (Xa.sum(-1) > 30)[..., None]; return np.where(m, Xb, 0).astype(np.uint8)
def gamma(g): return lambda im: Image.fromarray((255 * (np.asarray(im) / 255.) ** g).astype(np.uint8))
def noise(sd):
    rng = np.random.default_rng(0)
    return lambda im: Image.fromarray(np.clip(np.asarray(im).astype(float) + rng.normal(0, sd, (96, 96, 3)), 0, 255).astype(np.uint8))
tests = {
 "clean": lambda im: im,
 "blur_r1": lambda im: im.filter(ImageFilter.GaussianBlur(1)),
 "blur_r2": lambda im: im.filter(ImageFilter.GaussianBlur(2)),
 "blur_r3": lambda im: im.filter(ImageFilter.GaussianBlur(3)),
 "brightness_0.7": lambda im: ImageEnhance.Brightness(im).enhance(0.7),
 "brightness_1.3": lambda im: ImageEnhance.Brightness(im).enhance(1.3),
 "contrast_0.6": lambda im: ImageEnhance.Contrast(im).enhance(0.6),
 "contrast_1.5": lambda im: ImageEnhance.Contrast(im).enhance(1.5),
 "gamma_0.6": gamma(0.6), "gamma_1.6": gamma(1.6),
 "jpeg_q30": jpeg(30), "jpeg_q10": jpeg(10),
 "hue_+10deg": hue(10), "hue_-10deg": hue(-10), "hue_+25deg": hue(25),
 "saturation_0.6": lambda im: ImageEnhance.Color(im).enhance(0.6),
 "saturation_1.5": lambda im: ImageEnhance.Color(im).enhance(1.5),
 "noise_sd10": noise(10), "noise_sd25": noise(25),
 "down64_up96": lambda im: im.resize((64, 64), Image.BILINEAR).resize((96, 96), Image.BILINEAR),
 "down48_up96": lambda im: im.resize((48, 48), Image.BILINEAR).resize((96, 96), Image.BILINEAR),
 "grayscale": lambda im: im.convert("L").convert("RGB"),
 "rot90": lambda im: im.rotate(90), "rot37(black fill)": lambda im: im.rotate(37),
}
rows = {}
for k, f in tests.items():
    Xp = per(f)
    if k.startswith(("blur", "jpeg", "noise", "down", "hue", "brightness", "contrast", "gamma", "saturation")) : Xp = keepbg(Xt, Xp)
    p = predict(Xp)
    rows[k] = dict(auc=roc_auc_score(yt, p), sens05=float((p[yt == 1] >= .5).mean()), spec05=float((p[yt == 0] < .5).mean()),
                   frac_likely_pos=float((p >= th).mean()), pos_cells_called_negative=float((p[yt == 1] < tl).mean()), neg_cells_called_pos=float((p[yt == 0] >= th).mean()))
    print(k, {a: round(b, 4) for a, b in rows[k].items()}, flush=True)
save("exp_robust.json", rows)
