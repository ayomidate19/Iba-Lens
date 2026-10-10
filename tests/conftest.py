import json
import re
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


@pytest.fixture(scope="session")
def data():
    d = np.load(ROOT / "data" / "cells_96.npz")
    return {k: d[k] for k in ["X", "y", "split", "names"]}


@pytest.fixture(scope="session")
def scan():
    from smearscan import SmearScan
    return SmearScan()


@pytest.fixture(scope="session")
def thresholds():
    return json.loads((ROOT / "models" / "thresholds.json").read_text())


@pytest.fixture(scope="session")
def metrics():
    return json.loads((ROOT / "results" / "metrics.json").read_text())


@pytest.fixture(scope="session")
def test_probs(data, scan):
    m = data["split"] == "test"
    return scan.predict(list(data["X"][m])), data["y"][m]


def patient(name):
    return re.match(r"(C\d+)", name).group(1)
