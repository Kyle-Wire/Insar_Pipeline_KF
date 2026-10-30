"""
Phase 1, B1 (8 Oct 2026): what the relative figures are relative to -
config key reference_mode = farmland (default, unchanged) | gnss_absolute |
stable_polygon (significance.py, deliverables.py).

  farmland        every output byte-identical to the code before phase 1
                  (guard test against backups/pre_phase1_20261008).
  gnss_absolute   a datum per radar date fitted to >= 3 GNSS stations
                  projected onto the line of sight, one constant per
                  station, an optional plane (joint fit), leave-one-out
                  scoring; applied where deliverables builds the relative
                  field, so motion of the whole area is no longer cancelled.
  stable_polygon  far_field_mask returns a user-drawn stable polygon (a
                  ring outside the angle of draw) instead of the farmland.

Every test except the farmland guard fails on the code before this change
(the functions and keys did not exist, and the mode was ignored).
"""
import json
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from insar_pipeline import deliverables as dl
from insar_pipeline import significance as sg

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_fix_L7a as L7a                                       # noqa: E402  (synthetic deliverables project)

PKG = Path(__file__).resolve().parents[1]
BACKUP = PKG / "backups" / "pre_phase1_20261008"


@pytest.fixture(autouse=True)
def _clear_cache():
    dl.clear_cache()
    yield
    dl.clear_cache()


# ------------------------------------------------------------------ helpers
# a geographic grid (lat/lon, as the phase-linked products): 100 x 100 cells at 0.001 deg
GEO = {"X_FIRST": -87.60, "Y_FIRST": 38.50, "X_STEP": 0.001, "Y_STEP": -0.001, "LENGTH": 100, "WIDTH": 100}


class _Cfg:
    deramp_exclude_geojson = ""
    acceptance_far_field_buffer_m = 2000.0

    def __init__(self, **kw):
        # the AOI is the inner 80 x 80 cells
        self.bbox = (GEO["X_FIRST"] + 0.010, GEO["Y_FIRST"] - 0.090, GEO["X_FIRST"] + 0.090, GEO["Y_FIRST"] - 0.010)
        self.__dict__.update(kw)


def _box(w, s, e, n):
    return [[w, s], [e, s], [e, n], [w, n], [w, s]]


def _geojson(path: Path, *rings) -> str:
    """One polygon per ring list (outer first, holes after)."""
    feats = [{"type": "Feature", "properties": {}, "geometry": {"type": "Polygon", "coordinates": list(r)}}
             for r in rings]
    path.write_text(json.dumps({"type": "FeatureCollection", "features": feats}))
    return str(path)


def _cells(r0, r1, c0, c1):
    """lon/lat box covering grid rows r0..r1-1, cols c0..c1-1 (cell edges)."""
    return _box(GEO["X_FIRST"] + c0 * GEO["X_STEP"], GEO["Y_FIRST"] + r1 * GEO["Y_STEP"],
                GEO["X_FIRST"] + c1 * GEO["X_STEP"], GEO["Y_FIRST"] + r0 * GEO["Y_STEP"])


# ------------------------------------------------------------------ reference_mode and its settings
def test_reference_mode_defaults_to_farmland_takes_the_alias_and_refuses_anything_else():
    assert sg.reference_mode(_Cfg()) == "farmland"                     # key absent: today's behaviour
    assert sg.reference_mode(None) == "farmland"
    assert sg.reference_mode(_Cfg(reference_mode="")) == "farmland"
    assert sg.reference_mode(_Cfg(reference_mode="farmland_relative")) == "farmland"
    assert sg.reference_mode(_Cfg(reference_mode=" GNSS_Absolute ")) == "gnss_absolute"
    assert sg.reference_mode(_Cfg(reference_mode="stable_polygon")) == "stable_polygon"
    with pytest.raises(ValueError, match="not one of"):
        sg.reference_mode(_Cfg(reference_mode="regional"))
    # every key the modes read has a default that is today's behaviour
    assert sg.REFERENCE_SETTINGS["reference_mode"][0] == "farmland"
    assert set(sg.REFERENCE_SETTINGS) >= {"reference_stable_geojson", "reference_gnss_stations",
                                          "reference_gnss_plane", "reference_gnss_radius_m",
                                          "reference_gnss_match_days", "reference_gnss_cache_dir"}


def test_check_reference_settings_says_what_is_missing(tmp_path):
    assert sg.check_reference_settings(_Cfg()) == []
    assert "not one of" in sg.check_reference_settings(_Cfg(reference_mode="x"))[0]
    p = sg.check_reference_settings(_Cfg(reference_mode="stable_polygon"))
    assert p and "reference_stable_geojson is empty" in p[0]
    p = sg.check_reference_settings(_Cfg(reference_mode="stable_polygon", reference_stable_geojson=str(tmp_path / "x")))
    assert p and "does not exist" in p[0]
    p = sg.check_reference_settings(_Cfg(reference_mode="gnss_absolute", reference_gnss_stations=["A", "B"]))
    assert p and "at least 3" in p[0]
    assert sg.check_reference_settings(_Cfg(reference_mode="gnss_absolute", reference_gnss_stations=["A", "B", "C"])) == []
    p = sg.check_reference_settings(_Cfg(reference_mode="gnss_absolute", reference_gnss_stations=["crcn", "CRCN ", "LEMA"]))
    assert p and "names 2 station(s)" in p[0]                         # the same station twice is one station
    # the plane needs 4 at a date AND a fifth, or no station can be scored on a plane fitted without it
    for st in (["A", "B", "C"], ["A", "B", "C", "D"]):
        p = sg.check_reference_settings(_Cfg(reference_mode="gnss_absolute", reference_gnss_stations=st,
                                             reference_gnss_plane=True))
        assert p and "at least 5" in p[0] and "plane" in p[0]
    assert sg.check_reference_settings(_Cfg(reference_mode="gnss_absolute", reference_gnss_plane=True,
                                            reference_gnss_stations=["A", "B", "C", "D", "E"])) == []


# ------------------------------------------------------------------ stable_polygon
def test_stable_polygon_far_field_is_the_polygon_outside_the_panels_wherever_it_lies(tmp_path):
    good = np.ones((100, 100), bool)
    good[60:62, :] = False                                             # an incoherent band
    mines = _geojson(tmp_path / "mines.geojson", [_cells(40, 50, 40, 50)])
    # a ring around the mine, partly OUTSIDE the AOI (rows 0-10 are outside it)
    ring = _geojson(tmp_path / "ring.geojson", [_cells(0, 70, 20, 75), _cells(15, 65, 30, 65)[::-1]])
    cfg = _Cfg(deramp_exclude_geojson=mines, reference_mode="stable_polygon", reference_stable_geojson=ring)
    far, meta = sg.far_field_mask(cfg, good, GEO)
    poly = sg.stable_polygon_mask(cfg, good.shape, GEO)
    assert poly[5, 25] and not poly[40, 45] and poly[66, 50]           # in the ring / in its hole / in the ring
    assert np.array_equal(far, good & poly)                            # the mine is in the hole: nothing lost
    assert far[:10].any()                                              # used outside the AOI too
    wide, _ = sg.far_field_mask(cfg, good, GEO, restrict_to_aoi=False)
    assert np.array_equal(wide, far)
    # the farmland mask of the same config is something else entirely
    farm, _ = sg.far_field_mask(_Cfg(deramp_exclude_geojson=mines), good, GEO)
    assert not np.array_equal(farm, far)
    assert meta["reference_mode"] == "stable_polygon" and meta["stable_polygon"] == ring
    assert meta["n_cells"] == int(far.sum()) and meta["stable_polygon_cells"] == int((poly & good).sum())
    assert {"buffer_m", "aoi", "mine", "excluded_cells", "exclude_buffer_m", "panel_polygon"} <= set(meta)
    # a mapped panel inside the polygon is left out (not grown by the 2 km buffer)
    cfg2 = _Cfg(deramp_exclude_geojson=_geojson(tmp_path / "m2.geojson", [_cells(5, 12, 22, 30)]),
                reference_mode="stable_polygon", reference_stable_geojson=ring)
    far2, meta2 = sg.far_field_mask(cfg2, good, GEO)
    assert not far2[5:12, 22:30].any() and far2[13, 22] and meta2["stable_polygon_cells_in_panels"] == 56
    # report areas are cut out as in farmland mode
    cut = np.zeros_like(good)
    cut[0:10, 60:75] = True
    far3, meta3 = sg.far_field_mask(cfg, good, GEO, exclude=cut)
    assert not (far3 & cut).any() and meta3["excluded_cells"] > 0 and meta3["exclude_buffer_m"] == 500.0


def test_stable_polygon_refuses_a_missing_or_too_small_polygon(tmp_path):
    good = np.ones((100, 100), bool)
    with pytest.raises(ValueError, match="reference_stable_geojson is empty"):
        sg.far_field_mask(_Cfg(reference_mode="stable_polygon"), good, GEO)
    with pytest.raises(FileNotFoundError, match="stable-ground polygon"):
        sg.far_field_mask(_Cfg(reference_mode="stable_polygon", reference_stable_geojson=str(tmp_path / "no.geojson")),
                          good, GEO)
    tiny = _geojson(tmp_path / "tiny.geojson", [_cells(10, 20, 10, 20)])          # 100 cells
    with pytest.raises(ValueError, match="at least 200"):
        sg.far_field_mask(_Cfg(reference_mode="stable_polygon", reference_stable_geojson=tiny), good, GEO)


def test_stable_polygon_sees_motion_of_the_whole_area_that_farmland_cancels(tmp_path):
    """The whole area sinks 20 mm/yr LOS except a stable strip. Centred on
    the farmland (the area itself) the motion cancels; centred on the strip
    it reads -20, with the coherence-class bias still removed."""
    rng = np.random.default_rng(3)
    shape = (100, 100)
    tc = np.where(np.arange(100)[None, :] < 50, 0.5, 0.85) * np.ones(shape)
    bias = np.where(tc < 0.6, -10.0, -2.0)                            # distributed-scatterer bias by class
    strip = np.zeros(shape, bool)
    strip[85:100, :] = True                                           # 750 cells in each class
    vel = bias + np.where(strip, 0.0, -20.0) + rng.normal(0, 1.0, shape)
    good = np.ones(shape, bool)
    gj = _geojson(tmp_path / "strip.geojson", [_cells(85, 100, 0, 100)])
    area = sg.aoi_mask(_Cfg().bbox, GEO, shape) & ~strip
    farm, _ = sg.far_field_mask(_Cfg(), good, GEO)
    r_farm, _ = sg.class_matched_field(vel, tc, farm)
    far, _ = sg.far_field_mask(_Cfg(reference_mode="stable_polygon", reference_stable_geojson=gj), good, GEO)
    r_st, cen = sg.class_matched_field(vel, tc, far)
    assert abs(float(r_farm[area].mean())) < 2.0                       # cancelled (5 strip rows are farmland)
    assert float(r_st[area].mean()) == pytest.approx(-20.0, abs=0.3)  # seen
    assert cen[0] == pytest.approx(-10.0, abs=0.2) and cen[3] == pytest.approx(-2.0, abs=0.2)


def test_farmland_far_field_mask_is_the_pre_phase1_function():
    """Guard (passes on the old code by construction): the farmland branch
    is the old function, cell for cell and key for key."""
    if not (BACKUP / "significance.py").exists():
        pytest.skip("no pre-phase-1 backup on this machine")
    import importlib.util
    spec = importlib.util.spec_from_file_location("insar_pipeline._pre_p1_significance", BACKUP / "significance.py")
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    rng = np.random.default_rng(0)
    good = rng.random((100, 100)) > 0.2
    cut = np.zeros_like(good)
    cut[30:40, 30:40] = True
    for kw in ({}, {"restrict_to_aoi": False}, {"exclude": cut}, {"buffer_m": 500.0}):
        for cfg in (_Cfg(), _Cfg(reference_mode="farmland")):
            a, ma = sg.far_field_mask(cfg, good, GEO, **kw)
            b, mb = old.far_field_mask(cfg, good, GEO, **kw)
            assert np.array_equal(a, b)
            # 15 Oct 2026: the "wooded" record (landcover) is new; None with no land-cover map
            assert ma.pop("wooded") is None
            assert ma.keys() == mb.keys()
            for k in ma:
                assert np.array_equal(ma[k], mb[k]) if isinstance(ma[k], np.ndarray) else ma[k] == mb[k], k


# ------------------------------------------------------------------ the GNSS datum (significance)
T = np.arange(48) / 12.0                                              # 4 years, monthly


def test_station_constants_keep_the_datum_from_stepping_when_stations_join_and_leave():
    """CRCN/LEMA/KDL1/KDL2 end 2025-07-28, QCAP joins 2023-02-09: each
    station's GNSS origin and InSAR reference differ, so without one
    constant per station the datum steps whenever the set changes."""
    o_true = -40.0 * T + 4.0 * np.sin(2 * np.pi * T)
    b = np.array([0.0, 60.0, -35.0, 120.0])
    d = o_true[None] + b[:, None]
    d[3, :20] = np.nan                                                # joins
    d[1, 30:] = np.nan                                                # leaves
    fit = sg.gnss_datum_fit(T, d)
    assert fit["fixed"] == 0 and fit["used"].all()                    # 0 and 2 span the whole record
    off = fit["offset_mm"] - o_true
    assert np.allclose(off, off[0], atol=1e-9)                        # the datum, up to one constant
    assert np.allclose(fit["constants_mm"], b - b[0], atol=1e-9)
    assert fit["rms_residual_mm"] < 1e-9 and fit["dof"] == int(np.isfinite(d).sum()) - 48 - 3
    naive = np.nanmean(d, 0) - o_true                                 # one offset per date, no constants
    assert np.max(np.abs(np.diff(naive))) > 20.0                       # steps by tens of mm
    r = sg.datum_rate(fit, T, periodic=[1.0])
    assert r["offset_mm_yr"] == pytest.approx(-40.0, abs=1e-9) and r["epoch_share"] == 1.0


def test_dates_with_too_few_stations_and_unlinked_stations_get_no_datum():
    d = np.zeros((4, 48))
    d[2:, :] = np.nan
    d[2, 10:20] = 5.0                                                 # station 2 only where 0 and 1 are...
    d[3, 30:40] = 7.0
    d[1, 25:45] = np.nan                                              # ...and dates 25-44 have one station
    fit = sg.gnss_datum_fit(T, d)
    assert np.isnan(fit["offset_mm"][25:30]).all() and np.isnan(fit["offset_mm"][40:45]).all()
    assert np.isfinite(fit["offset_mm"][:25]).all() and np.isfinite(fit["offset_mm"][45:]).all()
    # station 3 shares 30-39 only with station 0, a lone pair: with 2 per date it IS linked
    assert fit["used"][3] and fit["constants_mm"][3] == pytest.approx(7.0)
    # a station that never shares a date with the fixed station's group is left out
    d2 = np.full((4, 48), np.nan)
    d2[0, :24] = d2[1, :24] = 1.0
    d2[2, 24:] = d2[3, 24:] = 2.0
    f2 = sg.gnss_datum_fit(T, d2)
    assert f2["used"].sum() == 2 and np.isnan(f2["constants_mm"][~f2["used"]]).all()
    assert np.isnan(f2["offset_mm"][24 if f2["used"][0] else 0])
    with pytest.raises(ValueError, match="no date has 2 stations"):
        sg.gnss_datum_fit(T, np.where(np.eye(4, 48, dtype=bool), 1.0, np.nan))


def test_the_weights_follow_sigma():
    rng = np.random.default_rng(1)
    o_true = -10.0 * T
    d = o_true[None] + np.zeros((4, 1))
    d[3] += rng.normal(0, 30.0, T.size)                               # one very noisy station
    sig = np.array([1.0, 1.0, 1.0, 30.0])
    w_fit = sg.gnss_datum_fit(T, d, sig)
    eq_fit = sg.gnss_datum_fit(T, d)
    err_w = np.std(w_fit["offset_mm"] - o_true)
    err_eq = np.std(eq_fit["offset_mm"] - o_true)
    assert err_w < 0.1 * err_eq


def test_joint_plane_is_exact_when_stations_join_and_leave_and_the_protocols_two_stage_plane_is_not():
    """protocol_B1.yaml's offset_plus_plane takes the station constants of
    the offset-only fit: each absorbs the tilt averaged over that station's
    own dates, so with stations joining or leaving the plane's rate is
    biased (0.5 read 0.28 here). The joint fit is exact; the protocol
    method is kept, and is exact when every station has every date."""
    rng = np.random.default_rng(1)
    xy = rng.uniform(-10, 10, (5, 2))
    o_true = -30.0 * T + 3.0 * np.sin(2 * np.pi * T)
    b = np.array([0.0, 40.0, -25.0, 100.0, 7.0])
    d = o_true[None] + b[:, None] + (0.5 * T)[None] * xy[:, :1] + (-0.2 * T)[None] * xy[:, 1:2]
    full = sg.gnss_datum_fit(T, d, xy=xy, plane=True, plane_method="protocol")
    r = sg.datum_rate(full, T, periodic=[1.0])
    assert r["east_mm_yr_per_km"] == pytest.approx(0.5, abs=1e-9) and r["north_mm_yr_per_km"] == pytest.approx(-0.2, abs=1e-9)
    gappy = d.copy()
    gappy[3, :16] = np.nan
    gappy[1, 32:] = np.nan
    joint = sg.gnss_datum_fit(T, gappy, xy=xy, plane=True)
    rj = sg.datum_rate(joint, T, periodic=[1.0])
    assert joint["plane"]["method"] == "joint" and joint["plane"]["epochs"] == 48
    assert rj["offset_mm_yr"] == pytest.approx(-30.0, abs=1e-6)
    assert rj["east_mm_yr_per_km"] == pytest.approx(0.5, abs=1e-9) and rj["north_mm_yr_per_km"] == pytest.approx(-0.2, abs=1e-9)
    assert joint["plane"]["east_mm_per_km"][joint["plane"]["reference_epoch"]] == 0.0
    proto = sg.gnss_datum_fit(T, gappy, xy=xy, plane=True, plane_method="protocol")
    rp = sg.datum_rate(proto, T, periodic=[1.0])
    assert abs(rp["east_mm_yr_per_km"] - 0.5) > 0.05
    # the correction at a point: the plane there
    c = sg.datum_correction(joint, (2.0, -1.0))
    assert np.allclose(c - c[0], -30.0 * T + 3.0 * np.sin(2 * np.pi * T) + 0.5 * T * 2.0 + 0.2 * T, atol=1e-6)
    # a plane needs 4 stations at a date and stations not in a line
    line = np.c_[np.arange(5.0), 2 * np.arange(5.0)]
    f = sg.gnss_datum_fit(T, d, xy=line, plane=True)
    assert f["plane"]["epochs"] == 0
    with pytest.raises(ValueError, match="positions"):
        sg.gnss_datum_fit(T, d, plane=True)
    with pytest.raises(ValueError, match="plane_method"):
        sg.gnss_datum_fit(T, d, xy=xy, plane=True, plane_method="best")


def test_leave_one_out_scores_each_station_on_a_datum_it_did_not_fit():
    """A station whose GNSS runs 12 mm/yr away from its InSAR: scored on the
    others' datum it shows the whole 12 (the others 12/3 the other way); in
    the all-station fit it would have hidden part of it in the datum."""
    rng = np.random.default_rng(2)
    insar = rng.normal(0, 0.5, (4, T.size)) + np.array([[0.0], [5.0], [-3.0], [9.0]])
    gnss = insar - 25.0 * T[None]                                     # the whole area: -25 mm/yr LOS
    gnss[2] += 12.0 * T                                               # station C disagrees
    loo = sg.gnss_absolute_loo(T, insar, gnss, ids=["A", "B", "C", "D"])
    per = loo["per_station"]
    assert all(v["scored"] for v in per.values())
    assert per["C"]["rate_bias_mm_yr"] == pytest.approx(-12.0, abs=0.3)
    for k in "ABD":
        assert per[k]["rate_bias_mm_yr"] == pytest.approx(4.0, abs=0.3)
    s = loo["summary"]
    assert s["stations_scored"] == 4 and s["rate_bias_rms_mm_yr"] == pytest.approx(np.sqrt((144 + 3 * 16) / 4), abs=0.4)
    assert len(per["C"]["residual_mm"]) == T.size and abs(np.nanmean(np.array(per["C"]["residual_mm"], float))) < 1e-6
    # in the fit that includes C, C's own residual trend is far smaller: why it must not be scored there
    fit = sg.gnss_datum_fit(T, gnss - insar)
    own = np.polyfit(T, -fit["residual_mm"][2], 1)[0]
    assert abs(own) < 0.8 * 12.0
    # the series offset removed is the protocol's plain mean over the matched dates, whatever the weights
    sig = np.where(np.arange(T.size) < 24, 1.0, 10.0)[None] * np.ones((4, 1))
    wl = sg.gnss_absolute_loo(T, insar, gnss, sig, ids=["A", "B", "C", "D"])
    fit_c = sg.gnss_datum_fit(T, np.delete(gnss - insar, 2, axis=0), sig[[0, 1, 3]])
    r_c = insar[2] + fit_c["offset_mm"] - gnss[2]
    assert wl["per_station"]["C"]["constant_mm"] == pytest.approx(float(np.mean(r_c)), abs=1e-3)
    # with two stations, one left out leaves one: nothing can be scored, and it says why
    two = sg.gnss_absolute_loo(T, insar[:2], gnss[:2])
    assert two["summary"]["stations_scored"] == 0
    assert "no datum" in two["per_station"]["0"]["reason"]


def test_joint_plane_refuses_dates_tied_together_by_fewer_than_three_stations():
    """Review 8 Oct 2026: two station groups that meet through one or two
    stations tie the offsets together but not the tilts (three unknowns per
    link). lstsq returned a minimum-norm answer without a word: with one
    shared station a true 0.5 mm/yr/km read -2.2 and a true -35 mm/yr
    offset +12. It now refuses, and so does each leave-one-out fold."""
    rng = np.random.default_rng(2)
    t = 2021.75 + np.arange(100) * 12 / 365.25
    xy = np.array([[0, 0], [10, 2], [-8, 5], [3, -9], [12, -6], [-5, -4], [6, 8], [-3, 7]], float)
    o_true = -35.0 * (t - t[0])

    def field(b):
        return (o_true[None] + b[:, None] + (0.5 * (t - t[0]))[None] * xy[:, :1]
                + (-0.3 * (t - t[0]))[None] * xy[:, 1:2])
    d = field(rng.normal(0, 10, 8))
    d[[4, 5, 6], :60] = np.nan                                      # B joins as A leaves:
    d[[0, 1, 2], 60:] = np.nan                                      # stations 3 and 7 are in both
    with pytest.raises(ValueError, match="at least 3 stations not in a line"):
        sg.gnss_datum_fit(t, d, xy=xy, plane=True)
    one = d.copy()
    one[7] = np.nan                                                 # one shared station: the case measured
    with pytest.raises(ValueError, match="at least 3 stations not in a line"):
        sg.gnss_datum_fit(t, one, xy=xy, plane=True)
    loo = sg.gnss_absolute_loo(t, np.zeros_like(d), d, xy=xy, plane=True)
    assert loo["summary"]["stations_scored"] == 0
    assert all("tie the dates" in v["reason"] for v in loo["per_station"].values())
    # the offsets alone ARE tied by one station: the offset-only fit is unaffected
    f = sg.gnss_datum_fit(t, np.where(np.isfinite(one), o_true[None] + np.arange(8.0)[:, None], np.nan))
    assert sg.datum_rate(f, t)["offset_mm_yr"] == pytest.approx(-35.0, abs=1e-9)
    # three shared stations not in a line are enough, and exact
    d2 = field(np.arange(8.0))
    d2[[4, 5, 6], :60] = np.nan
    d2[[0, 1], 60:] = np.nan                                        # 2, 3 and 7 carry on
    r2 = sg.datum_rate(sg.gnss_datum_fit(t, d2, xy=xy, plane=True), t)
    assert r2["east_mm_yr_per_km"] == pytest.approx(0.5, abs=1e-9) and r2["north_mm_yr_per_km"] == pytest.approx(-0.3, abs=1e-9)
    assert r2["offset_mm_yr"] == pytest.approx(-35.0, abs=1e-9)


def test_datum_rate_takes_time_from_mintpys_reference_date_and_its_steps():
    """Review 8 Oct 2026: with a polynomial of degree 2 MintPy's velocity is
    the slope at its reference date; the slope at year 0 of a decimal-year
    axis was returned. MintPy's own design matrix is the yardstick."""
    from mintpy.utils import time_func
    dates = [str(x) for x in (np.datetime64("2021-10-03") + 12 * np.arange(80)).astype("datetime64[D]")]
    dates = [d.replace("-", "") for d in dates]
    t = sg.years(dates)
    k = 7                                                          # MintPy's REF_DATE
    o = 3.0 - 20.0 * (t - t[k]) + 0.5 * 8.0 * (t - t[k]) ** 2 + 15.0 * (t > t[50])
    fit = {"offset_mm": o, "plane": None}
    model = {"polynomial": 2, "periodic": [], "stepDate": [dates[50]]}
    G = time_func.get_design_matrix4time_func(dates, model=model, ref_date=dates[k])
    mintpy_v = float(np.linalg.lstsq(G.astype(float), o, rcond=None)[0][1])
    ours = sg.datum_rate(fit, t, polynomial=2, steps=[t[50]], t_ref=t[k])["offset_mm_yr"]
    assert ours == pytest.approx(-20.0, abs=1e-6) and ours == pytest.approx(mintpy_v, abs=0.05)
    # on a decimal-year axis (the harness's) the answer is the same
    ty = 2021.75 + t
    assert sg.datum_rate(fit, ty, polynomial=2, steps=[ty[50]], t_ref=ty[k])["offset_mm_yr"] == pytest.approx(-20.0, abs=1e-6)
    assert abs(sg.datum_rate(fit, ty, polynomial=2, steps=[ty[50]])["offset_mm_yr"] + 20.0) > 1.0   # from the first date
    # a straight line does not care where time starts
    line = {"offset_mm": -20.0 * ty, "plane": None}
    assert sg.datum_rate(line, ty)["offset_mm_yr"] == pytest.approx(-20.0, abs=1e-9)


def test_velocity_model_reads_mintpys_dates_reference_and_steps(tmp_path):
    """Review 8 Oct 2026: an exclude-date FILE, startDate/endDate, the
    step dates and REF_DATE were ignored, so the datum's rate could be taken
    over other dates than the velocity's."""
    dates = [f"2019{m:02d}15" for m in range(1, 13)]
    (tmp_path / "exclude_date.txt").write_text("20190415\n# a comment\n20190615\n")

    class C:
        mintpy_dir = tmp_path
    attrs = {"mintpy.timeFunc.polynomial": "2", "mintpy.timeFunc.periodic": "[1.0, 0.5]",
             "mintpy.timeFunc.stepDate": "['20190715']", "mintpy.timeFunc.excludeDate": "exclude_date.txt",
             "mintpy.timeFunc.startDate": "20190215", "mintpy.timeFunc.endDate": "20191115",
             "mintpy.timeFunc.exp": "{}", "REF_DATE": "20190315"}
    vm = dl._velocity_model(attrs, dates, C())
    assert vm["polynomial"] == 2 and vm["periodic"] == [1.0, 0.5]
    assert [d for d, u in zip(dates, vm["use"]) if not u] == ["20190115", "20190415", "20190615", "20191215"]
    assert vm["t_ref"] == pytest.approx(sg.years(dates)[2])
    assert vm["steps"] == [pytest.approx(sg.years(dates)[6])]
    # nothing said: a straight line on every date from the first
    vm0 = dl._velocity_model({}, dates, C())
    assert vm0["polynomial"] == 1 and vm0["use"].all() and vm0["t_ref"] == 0.0 and vm0["steps"] == []


def test_datum_rate_uses_the_velocity_products_time_function_and_dates():
    t = T[:13]                                                        # 1 yr: an annual term leaks into a line
    o = -20.0 * t + 10.0 * np.sin(2 * np.pi * t)
    fit = {"offset_mm": o, "plane": None}
    line = sg.datum_rate(fit, t)
    seasonal = sg.datum_rate(fit, t, periodic=[1.0])
    assert abs(line["offset_mm_yr"] + 20.0) > 5.0
    assert seasonal["offset_mm_yr"] == pytest.approx(-20.0, abs=1e-9)
    use = np.ones(13, bool)
    use[5] = False
    o2 = o.copy()
    o2[5] += 500.0                                                    # a date velocity.h5 left out
    assert sg.datum_rate({"offset_mm": o2, "plane": None}, t, periodic=[1.0], use=use)["offset_mm_yr"] == \
        pytest.approx(-20.0, abs=1e-9)


# ------------------------------------------------------------------ deliverables: gnss_absolute
STATIONS = {"AAAA": (12, 12), "BBBB": (12, 45), "CCCC": (36, 15), "DDDD": (36, 48)}     # (row, col) on L7a's grid


def _fake_loader(cfg, truth, rate=-30.0, extra=None, tilt=None, only=None, stations=None, east=None, drop=None):
    """A stand-in for ground_truth.load_gnss_series: each station's GNSS
    line of sight = its InSAR (median of the reliable cells within 100 m) +
    rate * t + a constant per station (+ extra[sid] * t, + tilt * t * x_km),
    all in Up (the LOS vector is the same everywhere on this grid).
    `stations` replaces STATIONS; east[sid] mm/yr of east motion is put in
    e_mm and taken out of u_mm so the line of sight is unchanged; drop[sid]
    = the date indices with no GNSS."""
    stations = STATIONS if stations is None else stations
    attrs = truth["attrs"]
    lu = np.cos(np.radians(38.5))
    t = sg.years(L7a.DATES)
    good = dl.load_mask(cfg) & np.isfinite(truth["vel_mm"])
    pos = {sid: tuple(float(x) for x in sg.rc_to_lalo(attrs, *rc)) for sid, rc in stations.items()}
    lat0 = np.mean([p[0] for p in pos.values()])
    lon0 = np.mean([p[1] for p in pos.values()])
    calls = []

    class _Set(dict):                                                   # ground_truth.SeriesSet's shape
        missing: dict

    def load(ids, window, cache_dir):
        calls.append((list(ids), window, str(cache_dir)))
        out = _Set()
        out.missing = {}
        for k, sid in enumerate(ids):
            if only is not None and sid not in only:
                if sid == "DDDD":
                    out.missing[sid] = "download failed: HTTP 404"
                continue
            lat, lon = pos[sid]
            rr, cc = dl._station_cells(attrs, lat, lon, good.shape, 100.0)
            sel = good[rr, cc]
            los = np.nanmedian(truth["ts"][:, rr[sel], cc[sel]] * 1000.0, axis=1) + rate * t + 17.0 * k
            if extra and sid in extra:
                los = los + extra[sid] * t
            if tilt is not None:
                x, _ = sg.local_xy_km(lat, lon, lat0, lon0)
                los = los + tilt * t * float(x)
            le = -np.sin(np.radians(38.5)) * np.sin(np.radians(-100.0))    # L7a's LOS east (MintPy's formula)
            e = (east or {}).get(sid, 0.0) * t
            if drop and sid in drop:
                los = los.copy()
                los[drop[sid]] = np.nan
            df = pd.DataFrame({"e_mm": e, "n_mm": 0.0, "u_mm": (los - le * e) / lu, "se_mm": 1.0, "sn_mm": 1.0,
                               "su_mm": 2.0}, index=pd.to_datetime(L7a.DATES))
            df.attrs.update(lat=lat, lon=lon, source_url=f"test://{sid}", steps_removed=[])
            out[sid] = df
        return out
    load.calls = calls
    return load


def _abs_project(tmp_path, monkeypatch, **kw):
    cfg, truth = L7a.make_project(tmp_path)
    cfg.reference_mode = "gnss_absolute"
    cfg.reference_gnss_stations = [s.lower() for s in (kw.get("stations") or STATIONS)]   # upper-cased like NGL's
    cfg.reference_gnss_cache_dir = str(tmp_path / "gnss_cache")
    loader = _fake_loader(cfg, truth, **kw)
    monkeypatch.setattr(dl, "_load_gnss_series", loader)
    return cfg, truth, loader


def test_gnss_absolute_field_is_the_velocity_on_the_gnss_datum_so_whole_area_sinking_is_seen(tmp_path, monkeypatch):
    cfg, truth, loader = _abs_project(tmp_path, monkeypatch)
    far, rel, emp, model, null = dl._empirical_context(cfg)
    v = truth["vel_mm"]
    ok = np.isfinite(rel)
    assert ok.sum() > 1000
    assert np.allclose(rel[ok] - v[ok], -30.0, atol=1e-4)            # the whole area's -30 mm/yr is there
    rec = json.loads((cfg.project_dir / "deliverables" / dl.REFERENCE_RECORD).read_text())
    # the farmland field of the same record cancels it
    dl.clear_cache()
    cfg.reference_mode = "farmland"
    _, rel_f, _, _, null_f = dl._empirical_context(cfg)
    aoi = dl._box_mask(truth["attrs"], cfg.bbox, v.shape) & ok
    assert abs(float(np.nanmean(rel_f[aoi]))) < 3.0
    assert float(np.nanmean(rel[aoi])) < float(np.nanmean(rel_f[aoi])) - 25.0
    assert np.array_equal(np.isfinite(rel_f), ok)                      # the same cells are defined
    # the GNSS was asked for the record's dates +-3 days, into the configured cache
    ids, window, cache = loader.calls[0]
    assert ids == list(STATIONS) and window == ("2019-01-12", "2020-12-18") and cache == cfg.reference_gnss_cache_dir
    # a station listed twice is one station (scored with its twin in the datum, it would read no error)
    S = dl.gnss_station_series(cfg, stations=["aaaa", "AAAA ", "bbbb"])
    assert S["ids"] == ["AAAA", "BBBB"] and S["insar_mm"].shape[0] == 2
    assert rec["reference_mode"] == "gnss_absolute" and rec["available"] is True
    assert rec["rate"]["offset_mm_yr"] == pytest.approx(-30.0, abs=1e-6) and rec["rate"]["epoch_share"] == 1.0
    assert rec["fixed_station"] == "AAAA" and rec["variant"] == "offset_only" and rec["datum"]["plane"] is None
    assert [m["constant_mm"] for m in rec["stations"]] == pytest.approx([0.0, 17.0, 34.0, 51.0], abs=1e-6)
    assert all(m["usable"] and m["in_fit"] and m["n_cells"] >= 5 and len(m["los_vector"]) == 3 for m in rec["stations"])
    assert rec["leave_one_out"]["summary"]["stations_scored"] == 4
    assert rec["leave_one_out"]["summary"]["rate_bias_rms_mm_yr"] == pytest.approx(0.0, abs=1e-6)
    assert len(rec["datum"]["offset_mm"]) == len(L7a.DATES) and "how_to_apply" in rec["datum"]


def test_gnss_absolute_noise_carries_the_datums_leave_one_out_rate_error(tmp_path, monkeypatch):
    cfg, truth, _ = _abs_project(tmp_path, monkeypatch, extra={"CCCC": 8.0})
    _, rel, _, _, null = dl._empirical_context(cfg)
    rec = json.loads((cfg.project_dir / "deliverables" / dl.REFERENCE_RECORD).read_text())
    s = rec["datum_rate_sigma_mm_yr"]
    per = rec["leave_one_out"]["per_station"]
    assert per["CCCC"]["rate_bias_mm_yr"] == pytest.approx(-8.0, abs=0.05)
    assert s == pytest.approx(np.sqrt((64 + 3 * (8 / 3) ** 2) / 4), abs=0.05)
    dl.clear_cache()
    cfg.reference_mode = "farmland"
    _, _, _, _, null_f = dl._empirical_context(cfg)
    ok = np.isfinite(null_f)
    assert np.allclose(null[ok], np.sqrt(null_f[ok].astype(float) ** 2 + s ** 2), atol=1e-4)


FIVE = dict(STATIONS, EEEE=(24, 30))                                  # the plane: 4 at a date + 1 to leave out


def test_gnss_absolute_plane_tilts_the_field_across_the_area(tmp_path, monkeypatch):
    cfg, truth, _ = _abs_project(tmp_path, monkeypatch, tilt=3.0, stations=FIVE)
    cfg.reference_gnss_plane = True
    _, rel, _, _, null = dl._empirical_context(cfg)
    v = truth["vel_mm"]
    rec = json.loads((cfg.project_dir / "deliverables" / dl.REFERENCE_RECORD).read_text())
    assert rec["variant"] == "offset_plus_plane (joint)" and rec["datum"]["plane"]["dates_with_plane"] == 24
    assert rec["rate"]["east_mm_yr_per_km"] == pytest.approx(3.0, abs=1e-6)
    lat, lon = dl._grid_latlon(truth["attrs"], v.shape)
    gx, _ = sg.local_xy_km(lat, lon, *rec["datum"]["plane"]["origin_latlon"])
    ok = np.isfinite(rel)
    assert np.allclose(rel[ok] - v[ok], -30.0 + 3.0 * gx[ok], atol=2e-3)
    assert np.ptp(rel[ok] - v[ok]) > 5.0                               # ~2.6 km across: ~8 mm/yr of tilt
    assert rec["leave_one_out"]["summary"]["stations_scored"] == 5    # each on a plane fitted to the other 4
    assert rec["datum_rate_sigma_mm_yr"] == pytest.approx(0.0, abs=1e-6)


def test_gnss_absolute_refuses_when_the_datums_uncertainty_cannot_be_measured(tmp_path, monkeypatch):
    """Review 8 Oct 2026. With the plane and 4 stations, or stations whose
    records only meet through a third, no station can be scored on a datum
    fitted without it: the leave-one-out rate error was None, nothing was
    added to the cell noise and the absolute flags claimed more than is
    known. The mode now refuses and records why."""
    cfg, _, _ = _abs_project(tmp_path, monkeypatch, tilt=3.0)
    cfg.reference_gnss_plane = True
    with pytest.raises(ValueError, match="at least 5 stations"):
        dl._empirical_context(cfg)
    rec = json.loads((cfg.project_dir / "deliverables" / dl.REFERENCE_RECORD).read_text())
    assert rec["available"] is False
    # offset only: AAAA has the first year, CCCC the second, BBBB both - a datum at every date,
    # but leaving any one out leaves nothing to score it on
    dl.clear_cache()
    three = {k: STATIONS[k] for k in ("AAAA", "BBBB", "CCCC")}
    cfg, _, _ = _abs_project(tmp_path / "b", monkeypatch, stations=three,
                             drop={"AAAA": slice(12, None), "CCCC": slice(0, 12)})
    with pytest.raises(ValueError, match="uncertainty cannot be measured"):
        dl._empirical_context(cfg)
    rec = json.loads((cfg.project_dir / "deliverables" / dl.REFERENCE_RECORD).read_text())
    assert rec["available"] is False and "BBBB" in rec["reason"]


def test_gnss_absolute_record_says_the_frame_carries_the_plates_motion(tmp_path, monkeypatch):
    """Review 8 Oct 2026: absolute rates are in IGS20, so the plate's
    horizontal motion is on the line of sight (Site A ~ +9 mm/yr on an
    ascending track). The record gives its size per station; the README says
    stable ground does not read zero. East motion that leaves the line of
    sight unchanged leaves the datum unchanged."""
    cfg, _, _ = _abs_project(tmp_path, monkeypatch, east={k: 10.0 for k in STATIONS})
    D = cfg.project_dir / "deliverables"
    dl._empirical_context(cfg)
    rec = json.loads((D / dl.REFERENCE_RECORD).read_text())
    le = -np.sin(np.radians(38.5)) * np.sin(np.radians(-100.0))
    assert rec["rate"]["offset_mm_yr"] == pytest.approx(-30.0, abs=1e-6)
    fr = rec["frame"]
    assert fr["name"].startswith("IGS20") and "plate" in fr["note"]
    assert set(fr["station_horizontal_los_rate_mm_yr"]) == set(STATIONS)
    assert all(v == pytest.approx(10.0 * le, abs=0.01) for v in fr["station_horizontal_los_rate_mm_yr"].values())
    assert fr["station_horizontal_los_rate_median_mm_yr"] == pytest.approx(10.0 * le, abs=0.01)
    dl.export_measurement_points(cfg, D)
    readme = (D / "points_README.txt").read_text()
    assert "IGS20" in readme and "stable ground does not read zero" in readme


def test_gnss_absolute_with_too_few_stations_refuses_and_records_why(tmp_path, monkeypatch):
    """Site A has two NGL stations in its grid (FTBR, LAMC) and they
    never overlap in time: the mode must refuse, never fall back to the
    farmland field under an absolute label."""
    cfg, truth, _ = _abs_project(tmp_path, monkeypatch, only={"AAAA", "BBBB"})
    with pytest.raises(ValueError, match="at least 3 stations"):
        dl._empirical_context(cfg)
    rec = json.loads((cfg.project_dir / "deliverables" / dl.REFERENCE_RECORD).read_text())
    assert rec["available"] is False and "this record has 2" in rec["reason"]
    assert "CCCC: no GNSS series" in rec["reason"] and "DDDD: download failed: HTTP 404" in rec["reason"]
    D = cfg.project_dir / "deliverables"
    assert dl.export_signal_to_noise(cfg, D) is None and not (D / "rate_signal_to_noise.tif").exists()
    s = dl._cell_significance(cfg, np.ones((L7a.NY, L7a.NX), bool))
    assert s["available"] is False and "at least 3 stations" in s["note"]
    with pytest.raises(ValueError, match="not one of"):
        cfg.reference_mode = "absolute"
        dl.clear_cache()
        dl._empirical_context(cfg)


def test_gnss_absolute_products_say_they_are_absolute_and_the_report_does_not_rewrite_them(tmp_path, monkeypatch):
    import rasterio
    cfg, truth, _ = _abs_project(tmp_path, monkeypatch, extra={"DDDD": 4.0})
    D = cfg.project_dir / "deliverables"
    dl.export_signal_to_noise(cfg, D)
    dl.export_cell_noise(cfg, D)
    dl.export_measurement_points(cfg, D)
    with rasterio.open(D / "rate_signal_to_noise.tif") as r:
        tags = r.tags()
    assert tags["RELATIVE_TO"].startswith("absolute: the GNSS datum") and "REFERENCE_MODE" in tags
    assert "FARMLAND" not in tags                                       # no 'first pass, the report rewrites it'
    cn = next(D.glob("los_velocity_cell_noise_1sigma_*.tif"))
    with rasterio.open(cn) as r:
        assert "leave-one-out" in r.tags()["QUANTITY"] and r.tags()["RELATIVE_TO"].startswith("absolute")
    st = json.loads((D / "signal_to_noise.json").read_text())
    assert st["reference_mode"] == "gnss_absolute" and "GNSS datum" in st["definition"]
    readme = (D / "points_README.txt").read_text()
    assert "mean_velocity_relative = mean_velocity on the GNSS datum: ABSOLUTE" in readme
    assert "FIRST pass" not in readme and "removes it by coherence class" not in readme
    summ = dl.summary_stats(cfg)
    spc = summ["significance_per_cell"]
    assert spc["reference_mode"] == "gnss_absolute" and "farmland_pass" not in spc
    # all four stations set the datum: DDDD's +4 mm/yr moves it by 4/4
    assert spc["gnss_datum"]["stations"] == 4 and spc["gnss_datum"]["rate_added_mm_yr"] == pytest.approx(-29.0, abs=0.01)
    assert summ["deforming_area"]["relative_to"].startswith("absolute") and "GNSS datum" in summ["deforming_area"]["threshold"]
    before = (D / "points_README.txt").read_text()
    out = dl.points_on_report_farmland(cfg, {"dates": [L7a.DATES[0], L7a.DATES[-1]]}, ctx=object())
    assert out["applied"] is False and "gnss_absolute" in out["reason"]
    assert (D / "points_README.txt").read_text() == before


def test_a_farmland_run_removes_the_record_an_earlier_mode_left(tmp_path, monkeypatch):
    """Review 8 Oct 2026: a gnss_absolute run's reference_datum.json stayed
    in deliverables/ after the project went back to farmland, telling a
    reader (and the harness) the products were absolute."""
    cfg, _, _ = _abs_project(tmp_path, monkeypatch)
    dl._empirical_context(cfg)
    rec = cfg.project_dir / "deliverables" / dl.REFERENCE_RECORD
    assert json.loads(rec.read_text())["reference_mode"] == "gnss_absolute"
    dl.clear_cache()
    cfg.reference_mode = "farmland"
    dl._empirical_context(cfg)
    assert not rec.exists()


# ------------------------------------------------------------------ deliverables: stable_polygon
def test_stable_polygon_products_are_centred_on_the_polygon(tmp_path):
    cfg, truth = L7a.make_project(tmp_path)
    a = truth["attrs"]
    # a strip along the bottom of the grid, over both coherence halves
    w, e = a["X_FIRST"], a["X_FIRST"] + L7a.NX * a["X_STEP"]
    n, s = a["Y_FIRST"] + 34 * a["Y_STEP"], a["Y_FIRST"] + L7a.NY * a["Y_STEP"]
    cfg.reference_mode = "stable_polygon"
    cfg.reference_stable_geojson = _geojson(tmp_path / "stable.geojson", [_box(w, s, e, n)])
    far, rel, _, model, _ = dl._empirical_context(cfg)
    assert far[34:].sum() == far.sum() > 0 and not far[:34].any()
    tc = truth["tc"]
    k = sg.coherence_class(tc)
    for i in (0, 3):                                                   # each class is centred on the strip
        assert abs(float(np.nanmean(rel[far & (k == i)]))) < 1e-6
    rec = json.loads((cfg.project_dir / "deliverables" / dl.REFERENCE_RECORD).read_text())
    assert rec["reference_mode"] == "stable_polygon" and rec["cells_used"] == int(far.sum())
    assert set(rec["class_centres_mm_yr_los"]) == {"0.4-0.6", "0.8-0.9"}
    um = json.loads((cfg.project_dir / "deliverables" / "uncertainty_model.json").read_text())
    assert um["comparison_ground"]["definition"].startswith("the stable ground")
    assert um["basis"].startswith("pixels of the stable ground")
    D = cfg.project_dir / "deliverables"
    dl.export_measurement_points(cfg, D)
    readme = (D / "points_README.txt").read_text()
    assert "minus the mean of the stable ground" in readme and "stable.geojson" in readme
    s = dl.summary_stats(cfg)
    assert s["significance_per_cell"]["reference_mode"] == "stable_polygon"
    assert "stable ground" in s["deforming_area"]["threshold"]


# ------------------------------------------------------------------ sampling helpers
def test_gnss_for_the_datum_is_screened_about_a_local_line(monkeypatch, tmp_path):
    """Review 8 Oct 2026: ground_truth's default screening is one line over
    the record, and each day's sigma is at least the scatter about it. Where
    the ground does not move in a straight line that scatter is tens of mm,
    so the stations moving most counted for almost nothing in the datum."""
    from datetime import date
    from insar_pipeline import ground_truth as gt
    seen = {}

    def fake(ids, **kw):
        seen.update(kw)
        return gt.SeriesSet()
    monkeypatch.setattr(gt, "load_gnss_series", fake)
    dl._load_gnss_series(["CRCN"], ("2021-10-01", "2026-09-30"), tmp_path)
    assert seen["model"] == "local" and seen["cache_dir"] == tmp_path
    # the size of it: CRCN-like water-year rates -262, -295, +43, -64 mm/yr, a 15 mm annual term, 5 mm noise
    rng = np.random.default_rng(0)
    o = np.arange(date(2021, 10, 1).toordinal(), date(2025, 7, 28).toordinal())
    t = (o - o[0]) / 365.25
    u = sum(r * np.clip(t - i, 0, 1) for i, r in enumerate((-262, -295, 43, -64)))
    u = u + 15 * np.sin(2 * np.pi * t) + rng.normal(0, 5, t.size)
    X = np.column_stack([rng.normal(0, 1.5, t.size), rng.normal(0, 1.5, t.size), u])
    _, _, lin = gt._screen(o, X, "linear", [], gt.OUTLIER_SIGMA, gt.LOCAL_HALF_DAYS)
    _, _, loc = gt._screen(o, X, "local", [], gt.OUTLIER_SIGMA, gt.LOCAL_HALF_DAYS)
    assert lin[2] > 50.0 and loc[2] < 7.0


def test_station_cells_are_within_the_radius_on_geographic_and_projected_grids():
    lat, lon = sg.rc_to_lalo(GEO, 50, 50)
    rr, cc = dl._station_cells(GEO, float(lat), float(lon), (100, 100), 250.0)
    clat, clon = sg.rc_to_lalo(GEO, rr, cc)
    dist = np.hypot((clat - lat) * 111_320.0, (clon - lon) * 111_320.0 * np.cos(np.radians(lat)))
    assert rr.size > 10 and (dist <= 250.0).all()
    # nothing within 250 m was missed
    allr, allc = np.mgrid[40:61, 40:61]
    al, ao = sg.rc_to_lalo(GEO, allr, allc)
    d_all = np.hypot((al - lat) * 111_320.0, (ao - lon) * 111_320.0 * np.cos(np.radians(lat)))
    assert int((d_all <= 250.0).sum()) == rr.size
    assert dl._station_cells(GEO, 10.0, 10.0, (100, 100), 250.0)[0].size == 0          # off the grid
    # a UTM 11N grid (Corcoran) with its EPSG: 30 m cells, a 100 m radius -> the 37 cells of a disc
    from pyproj import Transformer
    utm = {"X_FIRST": 760000.0, "Y_FIRST": 4020000.0, "X_STEP": 30.0, "Y_STEP": -30.0, "EPSG": 32611,
           "LENGTH": 200, "WIDTH": 200}
    x, y = 760000.0 + 100.5 * 30.0, 4020000.0 - 80.5 * 30.0                             # centre of cell (80, 100)
    lon_s, lat_s = Transformer.from_crs("EPSG:32611", "EPSG:4326", always_xy=True).transform(x, y)
    rr, cc = dl._station_cells(utm, lat_s, lon_s, (200, 200), 100.0)
    assert rr.size == 37 and (np.hypot(rr - 80, cc - 100) * 30.0 <= 100.0 + 1e-6).all()
    # the grid's own lat/lon (the plane's coordinates): with the EPSG, or placed by a hint without it
    la, lo = dl._grid_latlon(utm, (200, 200))
    assert la[80, 100] == pytest.approx(lat_s, abs=1e-9) and lo[80, 100] == pytest.approx(lon_s, abs=1e-9)
    bare = {k: v for k, v in utm.items() if k != "EPSG"}
    la2, lo2 = dl._grid_latlon(bare, (200, 200), lonlat=(lon_s, lat_s))
    assert np.allclose(la2, la) and np.allclose(lo2, lo)


def test_gnss_at_dates_takes_the_mean_within_the_window_and_never_fills():
    idx = pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-04", "2020-02-20"])
    df = pd.DataFrame({"e_mm": [1.0, 2.0, 3.0, 9.0], "n_mm": 0.0, "u_mm": [10.0, np.nan, 30.0, 90.0],
                       "se_mm": 1.0, "sn_mm": 1.0, "su_mm": [2.0, 4.0, 6.0, 8.0]}, index=idx)
    g = dl._gnss_at_dates(df, ["20200102", "20200201", "20200221"], 3)
    assert g["e_mm"][0] == pytest.approx(2.0) and g["u_mm"][0] == pytest.approx(20.0)
    assert g["su_mm"][0] == pytest.approx(4.0)                         # averaged, not divided by sqrt(n)
    assert np.isnan(g["u_mm"][1]) and g["n_days"][1] == 0             # nothing within 3 days: no value
    assert g["u_mm"][2] == pytest.approx(90.0)


# ------------------------------------------------------------------ farmland: unchanged (guard)
_DIGEST = textwrap.dedent('''
    import hashlib, json, re, sys
    from pathlib import Path
    import numpy as np
    sys.path.insert(0, sys.argv[2])
    import test_fix_L7a as L7a
    import insar_pipeline
    from insar_pipeline import deliverables as dl
    # the build identity: the version (bumped per release since 16 Oct 2026) and the code hash
    CODE = re.compile(r"insar_pipeline [0-9][0-9.]* [(]code [0-9a-f]{12}|code [0-9a-f]{12}")
    out = Path(sys.argv[1])
    cfg, truth = L7a.make_project(out)
    D = cfg.project_dir / "deliverables"
    far, rel, emp, model, null = dl._empirical_context(cfg)
    h = lambda a: hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
    dig = {"far": h(far), "rel": h(rel), "emp": h(emp), "null": h(null)}
    dl.export_signal_to_noise(cfg, D); dl.export_cell_noise(cfg, D); dl.export_measurement_points(cfg, D)
    import rasterio, pyogrio
    for p in sorted(D.glob("*.tif")):
        with rasterio.open(p) as r:
            dig[p.name] = [h(r.read(1)), CODE.sub("", json.dumps(r.tags(), sort_keys=True))]
    for p in sorted(D.glob("*.json")):
        # 'software': the build label a record carries since 16 Oct 2026 (readiness review OUT-01)
        dig[p.name] = re.sub(r'"(computed|written|software)": "[^"]*"|"mtime": [0-9.e+]+', "", p.read_text())
    dig["readme"] = CODE.sub("", (D / "points_README.txt").read_text())
    dig["points"] = hashlib.sha256(pyogrio.read_dataframe(D / "points_attributes.gpkg", layer="points",
                                                          read_geometry=False).to_csv().encode()).hexdigest()
    dig["summary"] = CODE.sub("", json.dumps(dl.summary_stats(cfg), sort_keys=True, default=str))
    dig["files"] = sorted(p.name for p in D.iterdir())
    dig["package"] = str(Path(insar_pipeline.__file__).resolve().parent)
    print(json.dumps(dig))
''')


def test_farmland_products_are_byte_identical_to_the_pre_phase1_code(tmp_path):
    """Guard (passes on the old code by construction): with reference_mode
    unset, the relative field, noise, rasters (pixels and tags), JSON
    records, point table, README and summary of the synthetic project are
    what the code before phase 1 writes. Paths and the build identity in the
    source line (version and code hash) and in a record's 'software' key are
    the only differences allowed."""
    if not (BACKUP / "deliverables.py").exists():
        pytest.skip("no pre-phase-1 backup on this machine")
    script = tmp_path / "digest.py"
    script.write_text(_DIGEST)
    old_root = tmp_path / "old_pkg"
    old_root.mkdir()
    (old_root / "insar_pipeline").symlink_to(BACKUP)
    res = {}
    run_dir = tmp_path / "proj"                                            # the same path for both runs
    for name, root in (("old", old_root), ("new", PKG.parent)):
        env = dict(os.environ)
        env["PYTHONPATH"] = str(root)
        r = subprocess.run([sys.executable, str(script), str(run_dir), str(PKG / "tests")], capture_output=True,
                           text=True, env=env, cwd=str(root), timeout=600)
        assert r.returncode == 0, r.stderr[-2000:]
        res[name] = json.loads(r.stdout.strip().splitlines()[-1])
        shutil.rmtree(run_dir)
    assert res["old"].pop("package") == str(BACKUP.resolve())
    assert res["new"].pop("package") == str(PKG.resolve())
    assert res["old"].keys() == res["new"].keys()
    for k in res["old"]:
        assert res["old"][k] == res["new"][k], k
