"""The general report's reasoning: titles, where a feature sits relative to
the mapped mines, and how a movement unfolded over time."""
import numpy as np
import pytest

from insar_pipeline import report_insights as ri


def test_titles_from_folder_names():
    assert ri._pretty("site_a_pl") == "Site A"
    assert ri._pretty("site_d") == "Site D"
    assert ri._pretty("config_site_b") == "Site B"


def test_trend_shape_reads_onset_peak_and_slowing():
    t = np.arange(0, 10, 12 / 365.25)
    years = [2016 + int(x) for x in t]
    # flat for a year, fast 2018-2020, then slowing (spoil settling)
    rate = np.interp(t, [0, 1, 2, 5, 10], [0, -5, -90, -60, -15])
    series = np.cumsum(rate * 12 / 365.25) + np.random.default_rng(0).normal(0, 1, t.size)
    d = ri.trend_shape(t, series, years, noise_mm_yr=2.0)
    assert d["shape"] == "slowing" and d["began"] == 2017 and "slowed" in d["text"]
    # F7(c), follow-up of 7 Oct 2026: with no farmland placements the yardstick
    # is approximate, so the fast years are not named (2018-2019 until then)
    assert "fastest" not in d and "fewer than 10" in d["contrasts"]["fastest"]["withheld"]


def test_trend_shape_silent_on_noise():
    t = np.arange(0, 10, 12 / 365.25)
    years = [2016 + int(x) for x in t]
    series = np.random.default_rng(1).normal(0, 3, t.size)
    assert ri.trend_shape(t, series, years, noise_mm_yr=2.0)["text"] == ""


def test_relation_to_mines():
    g = np.zeros((200, 200), bool)
    surface = g.copy(); surface[50:100, 50:100] = True
    under = g.copy(); under[150:190, 150:190] = True
    poly = {"surface": surface, "underground": under}
    on = g.copy(); on[60:80, 60:80] = True
    assert ri.relation_to_mines(on, poly, (15.0, 15.0))["key"] == "surface"
    near = g.copy(); near[120:140, 150:170] = True                 # 11 cells (165 m) above the workings
    r = ri.relation_to_mines(near, poly, (15.0, 15.0))
    assert r["key"] == "near_underground" and r["nearest_km"] == pytest.approx(0.165, abs=0.01)
    far = g.copy(); far[0:10, 180:195] = True
    # 1.36 km at 15 m cells: beyond NEAR_MINE_KM (0.5 km), so no mining attribution (decision M2, 7 Oct 2026)
    r = ri.relation_to_mines(far, poly, (15.0, 15.0))
    assert r["key"] == "none" and r["phrase"] == "1.4 km from the nearest mapped surface mine"
    assert ri.relation_to_mines(far, poly, (60.0, 60.0))["key"] == "none"           # 5.4 km at 60 m cells
    assert ri.relation_to_mines(on, {}, (15.0, 15.0))["key"] is None


def test_cause_hints_follow_context():
    assert "spoil" in ri.cause_hint("surface", -1, "slowing") and "matches" in ri.cause_hint("surface", -1, "slowing")
    assert "panel" in ri.cause_hint("underground", -1, "steady")
    assert "do not explain" in ri.cause_hint("none", -1, None)
    assert "Rising" in ri.cause_hint("surface", 1, None)
