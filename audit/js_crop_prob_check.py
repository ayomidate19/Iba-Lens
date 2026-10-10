"""Feed the crops produced by the TS segment() (node) into ONNX (python) and compare to reference.json probs/bands."""
import json, numpy as np, onnxruntime as ort, sys
from pathlib import Path
R = Path(__file__).resolve().parents[1]
ref = json.loads((R / "mobile/public/demo/reference.json").read_text()); th = json.loads((R / "models/thresholds.json").read_text())
sess = ort.InferenceSession(str(R / "mobile/public/model/iba_mobilenetv2_96.onnx"), providers=["CPUExecutionProvider"])
out = {}
for sid, r in ref.items():
    b = np.fromfile(R / f"audit/scratch/{sid}.jscrops.bin", np.uint8).reshape(-1, 96, 96, 3)
    p = sess.run(None, {"image": (b.transpose(0, 3, 1, 2) / 255.).astype(np.float32)})[0]
    pp = np.array([c["prob"] for c in r["cells"]])
    pos = int((p >= th["t_high"]).sum()); unc = int(((p >= th["t_low"]) & (p < th["t_high"])).sum())
    band = "likely positive" if pos >= th["slide_k_pos"] else "likely negative" if pos + unc <= th["slide_k_neg"] else "unclear"
    out[sid] = dict(n=len(p), max_prob_diff=float(np.abs(p - pp).max()), band_js_pipeline=band, band_ref=r["summary"]["band"], pos_js=pos, pos_ref=r["summary"]["flagged_positive"])
    print(sid, out[sid])
(R / "audit/js_crop_prob_check.json").write_text(json.dumps(out, indent=1))
