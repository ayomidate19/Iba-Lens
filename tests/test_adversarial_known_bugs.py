"""Each test asserts the behaviour a safe triage tool SHOULD have. They are marked xfail(strict=True):
they pass-as-xfail today (documenting a real bug) and will start failing loudly (XPASS) once the bug is fixed."""
import numpy as np
import pytest

from smearscan import segment, summarise

THR = (0.14, 0.90, 3, 2, 0.0)


def _field(data, n=60, spacing=110, seed=0):
    """cells on a grid `spacing` px apart (96px cells => spacing < 96 overlaps them)."""
    from smearscan import BG
    idx = np.where(data["split"] == "test")[0]
    pick = np.random.default_rng(seed).choice(idx, n, replace=False)
    cols, pad = 10, 60
    rows = -(-n // cols)
    f = np.tile(BG, (rows * spacing + 2 * pad + 96, cols * spacing + 2 * pad + 96, 1))
    for i, k in enumerate(pick):
        r, c = divmod(i, cols)
        patch = f[pad + r * spacing : pad + r * spacing + 96, pad + c * spacing : pad + c * spacing + 96]
        fg = data["X"][k].sum(-1) > 30
        patch[fg] = data["X"][k][fg]
    return f


@pytest.mark.xfail(strict=True, reason="BUG: overlapping/touching cells merge into one blob > max_area and are dropped (0 cells found)")
def test_overlapping_cells_are_still_found(data):
    cells = segment(_field(data, spacing=84))
    assert len(cells) >= 48


def test_zero_cells_must_not_be_called_negative():
    assert summarise([], *THR[:2], *THR[2:])["band"] != "likely negative"


@pytest.mark.xfail(strict=True, reason="BUG: global Otsu threshold loses ~18-34% of cells under mild vignetting")
def test_uneven_illumination_keeps_cells(data):
    f = _field(data, n=100).astype(float)
    H, W = f.shape[:2]
    yy, xx = np.mgrid[0:H, 0:W]
    g = 0.55 + 0.45 * (1 - ((xx / W - .3) ** 2 + (yy / H - .3) ** 2) / .9)
    assert len(segment(np.clip(f * g[..., None], 0, 255).astype(np.uint8))) >= 95
