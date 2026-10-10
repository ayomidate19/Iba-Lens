import json

import numpy as np
import pytest
from PIL import Image

from conftest import ROOT
from smearscan import Cell, render_field, segment, summarise

T_LOW, T_HIGH = 0.14, 0.90


def cells_with(probs):
    return [Cell(bbox=(0, 0, 1, 1), crop=np.zeros((96, 96, 3), np.uint8), prob=p) for p in probs]


def test_summarise_zero_cells():
    s = summarise([], T_LOW, T_HIGH, 3, 2, 0.005)
    assert s["cells"] == 0 and s["parasitemia_pct"] == 0.0 and s["max_prob"] == 0.0
    # BUG-CANDIDATE: an empty field (no cells segmented) yields pos+unc=0 <= k_neg -> "likely negative"
    assert s["band"] == "likely negative"


def test_summarise_all_positive():
    s = summarise(cells_with([0.99] * 50), T_LOW, T_HIGH, 3, 2, 0.005)
    assert s["band"] == "likely positive" and s["flagged_positive"] == 50
    assert s["parasitemia_pct"] == pytest.approx(99.5)


def test_summarise_all_negative():
    s = summarise(cells_with([0.01] * 50), T_LOW, T_HIGH, 3, 2, 0.005)
    assert s["band"] == "likely negative" and s["parasitemia_pct"] == 0.0


@pytest.mark.parametrize("n_pos,n_unc,band", [
    (2, 0, "likely negative"), (3, 0, "likely positive"), (0, 2, "likely negative"), (0, 3, "unclear"),
    (1, 1, "likely negative"), (1, 2, "unclear"), (0, 0, "likely negative"),
])
# NB (2,0): two cells at p=0.99 still give "likely negative" because k_neg=2 counts positives too (design quirk, see audit/findings.md)
def test_band_logic_boundaries(n_pos, n_unc, band):
    probs = [0.99] * n_pos + [0.5] * n_unc + [0.01] * 50
    assert summarise(cells_with(probs), T_LOW, T_HIGH, 3, 2, 0.0)["band"] == band


def test_threshold_edges_are_inclusive_on_high_exclusive_on_low():
    s = summarise(cells_with([T_HIGH, T_LOW]), T_LOW, T_HIGH, 1, 0, 0.0)
    assert s["flagged_positive"] == 1 and s["flagged_unclear"] == 1


def test_parasitemia_clamped_at_zero():
    s = summarise(cells_with([0.01] * 100), T_LOW, T_HIGH, 3, 2, 0.5)
    assert s["parasitemia_pct"] == 0.0


def test_segment_counts_on_rendered_fields(data):
    X, y, split = data["X"], data["y"], data["split"]
    idx = np.where(split == "test")[0]
    for n, seed in [(30, 0), (60, 1), (100, 2)]:
        pick = np.random.default_rng(seed).choice(idx, n, replace=False)
        cells = segment(render_field([X[i] for i in pick], seed=seed))
        assert len(cells) == n


def test_segment_empty_field_returns_nothing():
    assert segment(np.full((300, 300, 3), 230, np.uint8)) == []


def test_segment_pure_black_field():
    assert segment(np.zeros((300, 300, 3), np.uint8)) == []


def test_segment_ignores_tiny_specks():
    f = np.full((300, 300, 3), 230, np.uint8)
    f[100:110, 100:110] = 60
    assert segment(f) == []


def test_demo_reference_matches_python_pipeline(scan):
    ref = json.loads((ROOT / "mobile/public/demo/reference.json").read_text())
    assert len(ref) == 8
    for sid, r in ref.items():
        field = np.asarray(Image.open(ROOT / "mobile/public/demo" / f"{sid}.png").convert("RGB"))
        cells, s = scan.analyse(field, with_cam=False)
        assert len(cells) == r["summary"]["cells"], sid
        assert s["band"] == r["summary"]["band"], sid
        assert s["flagged_positive"] == r["summary"]["flagged_positive"], sid
        assert [list(c.bbox) for c in cells] == [c["bbox"] for c in r["cells"]], sid
        assert max(abs(c.prob - rc["prob"]) for c, rc in zip(cells, r["cells"])) < 1e-4, sid


def test_mobile_demo_images_identical_to_assets():
    for p in (ROOT / "assets/slides").glob("LAB-*.png"):
        a = np.asarray(Image.open(p).convert("RGB"))
        b = np.asarray(Image.open(ROOT / "mobile/public/demo" / p.name).convert("RGB"))
        assert np.array_equal(a, b), p.name


def test_demo_truth_vs_prediction(scan):
    gt = json.loads((ROOT / "assets/slides/ground_truth.json").read_text())
    for g in gt:
        field = np.asarray(Image.open(ROOT / "assets/slides" / f"{g['id']}.png").convert("RGB"))
        cells, s = scan.analyse(field, with_cam=False)
        assert s["cells"] == g["cells"]
        true_pct = 100 * g["true_parasitized"] / g["cells"]
        assert abs(s["parasitemia_pct"] - true_pct) < 4.0, (g["id"], s["parasitemia_pct"], true_pct)
        if true_pct >= 5:
            assert s["band"] == "likely positive", g["id"]
        if true_pct == 0:
            assert s["band"] != "likely positive", g["id"]


def test_demo_zero_percent_slides_never_say_likely_negative(scan):
    """Documents the k_neg weakness: truly clean demo slides are not called negative."""
    gt = {g["id"]: g for g in json.loads((ROOT / "assets/slides/ground_truth.json").read_text())}
    bands = {}
    for sid, g in gt.items():
        if g["true_parasitized"] == 0:
            field = np.asarray(Image.open(ROOT / "assets/slides" / f"{sid}.png").convert("RGB"))
            bands[sid] = scan.analyse(field, with_cam=False)[1]["band"]
    print(bands)
