"""Tide and ionosphere: automatic for large areas only."""
import dataclasses

import pytest

from insar_pipeline import corrections
from insar_pipeline.config import load_config


@pytest.fixture
def PL(tmp_path):
    """A phase-linked project over a small area (about 27 x 23 km; the worked
    example config's box until the example configs left the repository, 28 Oct 2026)."""
    p = tmp_path / "pl.yaml"
    p.write_text(f"backend: miaplpy\nproject_dir: {tmp_path / 'p'}\nbbox: [-95.89174, 29.65493, -95.60826, 29.86507]\n"
                 "start_date: '2021-10-01'\nend_date: '2026-09-30'\ntrack: 34\norbit_direction: ascending\n"
                 "stack_virtual_merge: 'False'\nmintpy_deramp: 'no'\n")
    return p


def test_small_area_skips_both(PL):
    d = corrections.decide(load_config(str(PL)))
    assert d["extent_km"][1] < 100 and not d["large_area"]
    assert d["solid_earth_tides"] is False and d["ionosphere"] is None
    assert "solidEarthTides            = no" in corrections.mintpy_lines(load_config(str(PL)))


def test_large_area_applies_both_and_explicit_settings_win(PL):
    big = dataclasses.replace(load_config(str(PL)), bbox=(-89.0, 37.8, -87.5, 38.9))
    d = corrections.decide(big)
    assert d["large_area"] and d["solid_earth_tides"] and d["ionosphere"] == "gim"
    off = dataclasses.replace(big, solid_earth_tides="no", ionosphere_correction="no")
    d = corrections.decide(off)
    assert not d["solid_earth_tides"] and d["ionosphere"] is None and "explicitly" in d["why"]
    small_forced = dataclasses.replace(load_config(str(PL)), ionosphere_correction="gim")
    assert corrections.decide(small_forced)["ionosphere"] == "gim"


def test_threshold_is_a_setting(PL):
    c = dataclasses.replace(load_config(str(PL)), large_area_km=20.0)
    assert corrections.decide(c)["large_area"]
