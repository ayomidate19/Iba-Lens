import hashlib
import json
from collections import defaultdict

import numpy as np
from sklearn.metrics import roc_auc_score

from conftest import ROOT, patient


def test_patient_split_has_no_overlap(data):
    pts = {s: {patient(n) for n in data["names"][data["split"] == s]} for s in ["train", "val", "test"]}
    assert pts["train"].isdisjoint(pts["val"])
    assert pts["train"].isdisjoint(pts["test"])
    assert pts["val"].isdisjoint(pts["test"])
    assert (len(pts["train"]), len(pts["val"]), len(pts["test"])) == (140, 20, 40)


def test_no_exact_duplicate_cells_across_splits(data):
    seen = defaultdict(set)
    for x, s in zip(data["X"], data["split"]):
        seen[hashlib.md5(x.tobytes()).hexdigest()].add(str(s))
    assert not [k for k, v in seen.items() if len(v) > 1]


def test_every_image_name_parses_to_a_patient(data):
    for n in data["names"]:
        patient(n)


def test_thresholds_sane(thresholds):
    assert 0 < thresholds["t_low"] < thresholds["t_high"] < 1
    assert thresholds["slide_k_neg"] < thresholds["slide_k_pos"]
    assert 0 <= thresholds["cell_fp_rate"] < 0.05


def test_stored_scores_match_fresh_inference(test_probs):
    p, y = test_probs
    s = np.load(ROOT / "results" / "test_scores.npz")
    assert np.array_equal(s["y"], y)
    assert np.abs(s["p"] - p).max() < 1e-4


def test_headline_metrics_reproduce(test_probs, metrics, thresholds):
    p, y = test_probs
    assert abs(roc_auc_score(y, p) - metrics["test"]["auc"]) < 1e-4
    pred = p >= 0.5
    sens, spec = pred[y == 1].mean(), (~pred[y == 0]).mean()
    assert abs(sens - metrics["test"]["at_0.5"]["sensitivity"]) < 1e-3
    assert abs(spec - metrics["test"]["at_0.5"]["specificity"]) < 1e-3
    assert round(sens * 100, 1) == 95.3 and round(spec * 100, 1) == 97.6
    for name, mask in [("likely_positive", p >= thresholds["t_high"]),
                       ("unclear", (p >= thresholds["t_low"]) & (p < thresholds["t_high"])),
                       ("likely_negative", p < thresholds["t_low"])]:
        t = metrics["triage_on_test"][name]
        assert t["cells"] == mask.sum() and t["truly_parasitized"] == y[mask].sum()


def test_thresholds_json_matches_metrics_json(metrics, thresholds):
    assert metrics["thresholds"]["t_low"] == thresholds["t_low"]
    assert metrics["thresholds"]["t_high"] == thresholds["t_high"]


def test_mobile_threshold_copy_in_sync(thresholds):
    t = json.loads((ROOT / "mobile" / "public" / "model" / "thresholds.json").read_text())
    assert t == thresholds


def test_onnx_matches_torch(data, scan):
    import onnxruntime as ort
    X = data["X"][data["split"] == "test"][:256]
    ref = scan.predict(list(X))
    sess = ort.InferenceSession(str(ROOT / "mobile/public/model/iba_mobilenetv2_96.onnx"), providers=["CPUExecutionProvider"])
    got = sess.run(None, {"image": (X.transpose(0, 3, 1, 2) / 255.0).astype(np.float32)})[0]
    assert np.abs(got - ref).max() < 1e-4
