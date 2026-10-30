"""run_local_change.py options added 22 Oct 2026: --box points, --period
parsing, and compare.json written as strict JSON."""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

_spec = importlib.util.spec_from_file_location("run_local_change", Path(__file__).resolve().parents[1] / "scripts" / "run_local_change.py")
rlc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rlc)


def test_box_points_stay_within_the_radius():
    la, lo = rlc.box_points(45.0, -100.0, 300.0)
    d = np.hypot((la - 45.0) * 111320.0, (lo + 100.0) * 111320.0 * np.cos(np.radians(45.0)))
    assert d.max() <= 300.0 + 1e-6 and len(la) > 1000


def test_box_and_period_arguments_are_checked():
    assert rlc.parse_box("town=45.1,-100.2,500") == ("town", 45.1, -100.2, 500.0)
    assert rlc.parse_period("20240101-20241231") == ("20240101", "20241231")
    for bad in ("town=45.1,-100.2", "town"):
        with pytest.raises(argparse.ArgumentTypeError):
            rlc.parse_box(bad)
    for bad in ("2024-2025", "20241231-20240101"):
        with pytest.raises(argparse.ArgumentTypeError):
            rlc.parse_period(bad)


def test_comparison_record_is_strict_json():
    out = {"a": float("nan"), "b": [np.float32(1.5), np.inf], "c": {"d": np.int64(3)}}
    text = json.dumps(rlc._finite(out), allow_nan=False)
    assert json.loads(text) == {"a": None, "b": [1.5, None], "c": {"d": 3}}
