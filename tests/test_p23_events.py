"""PLAN phase 2, F5 (review 8 Oct 2026, group 'events'): event products
(events.py) and, from M1, the too-fast flag's wavelength (deliverables.py).

What is checked, on a synthetic longwall event (a Gaussian trough that sinks
300 mm in the line of sight between 2018-05-12 and 2019-04-03 and then stops)
on a lat/lon grid and on a UTM 56S grid (Appin's), with a block of
incoherent cells and a 36-day gap in the dates:

  - cumulative_between: the change between two dates is the difference of
    the delivered series at the SAR dates used (outward by default, nearest,
    or interpolated within 24 days), NaN where the cell is not reliable,
    with the dates asked for and used in its attributes; a window beyond the
    record is clamped and says so; a window between two SAR dates is refused;
  - panel_max: depth (5th percentile, a robust minimum), date (the first SAR
    date after the trough stopped), half-depth date and place of the
    deepest subsidence; the polygon's own dates; flag shares; a polygon on
    incoherent ground has no value and says why;
  - profile: cells within 100 m of the line in 50 m bins (counted
    independently), the trough in the middle bins, the incoherent block
    reported, not filled;
  - values_at: the harness's own sampling (validation.sample_points) at SAR
    dates, linear interpolation between them, no value across a gap > 24 days
    or outside the record, masked on incoherent ground; a truth CSV's own
    dates are used;
  - deliverables/events/ is written when event_products is set (GeoTIFF with
    its dates in the tags, panel_max.csv/json, profiles.csv, events.json),
    export_all calls it, and with the key unset nothing is written (an
    earlier write is moved aside);
  - config: the three keys load, default off, and bad values are refused;
  - M1: the too-fast flag takes the wavelength from the product, refuses
    C-band for a NISAR product that states none, and says it is a lower bound.

events.py did not exist before, and the M1 tests fail on the old
deliverables.py (it wrote the flag with C-band for a NISAR product and had no
wavelength tag). No network, nothing under ~/insar_projects.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest

from insar_pipeline import deliverables as dl
from insar_pipeline import events as E
from insar_pipeline.config import PipelineConfig

LAT0, LON0 = -34.15, 150.60
DLAT, DLON = -0.00015, 0.00018           # ~16.7 m x 16.6 m
NY = NX = 120
A_MM = 300.0                              # LOS sinking at the trough's centre
SIG_M = 250.0
T0, T1 = "2018-05-12", "2019-04-03"
CR, CC = 60, 60                           # the trough's centre cell
HOLE = (slice(15, 40), slice(80, 115))    # incoherent block (larger than a 100 m circle)
UTM_X0, UTM_Y0, UTM_STEP = 290000.0, 6219000.0, 16.5


def _days(dates):
    return np.array([E._day(d) for d in dates])


def make_dates():
    d0 = E._day("2018-01-04")
    days = [d0 + 12 * k for k in range(70)]
    days = days[:50] + days[52:]          # a 36-day gap in Aug-Sep 2019
    return [E._ymd(d) for d in days]


def _s(day):
    a, b = E._day(T0), E._day(T1)
    return np.clip((np.asarray(day, float) - a) / (b - a), 0.0, 1.0)


def make_project(tmp_path, projected=False, wavelength=True, drift_mm_yr=0.0):
    """A geocoded MintPy-like product under tmp_path/proj/mintpy/geo.
    drift_mm_yr: every cell but the reference corner (rows/cols 0-9) drifts
    at this rate, as farmland does against a reference pixel on other ground."""
    root = tmp_path / "proj"
    geo = root / "mintpy" / "geo"
    geo.mkdir(parents=True)
    (root / "deliverables").mkdir()
    dates = make_dates()
    days = _days(dates)
    rr, cc = np.mgrid[0:NY, 0:NX]
    if projected:
        x = UTM_X0 + (cc + 0.5) * UTM_STEP
        y = UTM_Y0 - (rr + 0.5) * UTM_STEP
        dist = np.hypot(x - x[CR, CC], y - y[CR, CC])
        attrs = {"X_FIRST": str(UTM_X0), "Y_FIRST": str(UTM_Y0), "X_STEP": str(UTM_STEP),
                 "Y_STEP": str(-UTM_STEP), "EPSG": "32756"}
    else:
        lat = LAT0 + (rr + 0.5) * DLAT
        lon = LON0 + (cc + 0.5) * DLON
        dy = (lat - lat[CR, CC]) * 111_320.0
        dx = (lon - lon[CR, CC]) * 111_320.0 * math.cos(math.radians(lat[CR, CC]))
        dist = np.hypot(dx, dy)
        attrs = {"X_FIRST": str(LON0), "Y_FIRST": str(LAT0), "X_STEP": str(DLON), "Y_STEP": str(DLAT)}
    shape_ = np.exp(-dist ** 2 / (2 * SIG_M ** 2))
    los = -A_MM * _s(days)[:, None, None] * shape_[None]               # mm
    drift = np.full((NY, NX), drift_mm_yr)
    drift[:10, :10] = 0.0
    los = los + drift[None] * ((days - days[0]) / 365.25)[:, None, None]
    ts = (los - los[:, 5, 5][:, None, None]) / 1000.0                     # reference pixel (5, 5), metres
    tc = np.full((NY, NX), 0.9, np.float32)
    tc[HOLE] = 0.2
    attrs.update({"WIDTH": str(NX), "LENGTH": str(NY), "REF_Y": "5", "REF_X": "5", "REF_DATE": dates[0],
                  "START_DATE": dates[0], "END_DATE": dates[-1]})
    if wavelength:
        attrs["WAVELENGTH"] = "0.05546576"
    with h5py.File(geo / "geo_timeseries_ERA5_demErr.h5", "w") as f:
        f.create_dataset("timeseries", data=ts.astype(np.float32), chunks=(8, 40, 40))
        f.create_dataset("date", data=np.array([d.encode() for d in dates]))
        f.attrs.update({**attrs, "UNIT": "m"})
    with h5py.File(geo / "geo_velocity.h5", "w") as f:
        f.create_dataset("velocity", data=((ts[-1] - ts[0]) / 2.6).astype(np.float32))
        f.attrs.update({**attrs, "UNIT": "m/year"})
    with h5py.File(geo / "geo_temporalCoherence.h5", "w") as f:
        f.create_dataset("temporalCoherence", data=tc)
        f.attrs.update(attrs)
    with h5py.File(geo / "geo_geometryRadar.h5", "w") as f:
        f.create_dataset("incidenceAngle", data=np.full((NY, NX), 39.0, np.float32))
        f.create_dataset("azimuthAngle", data=np.full((NY, NX), 100.0, np.float32))
        f.create_dataset("height", data=np.zeros((NY, NX), np.float32))
        f.attrs.update(attrs)
    cfg = PipelineConfig(project_dir=root, bbox=[150.6, -34.17, 150.62, -34.15], start_date="2018-01-01",
                         end_date="2020-04-30", backend="miaplpy")
    return cfg, {"ts_mm": ts * 1000.0, "dates": dates, "days": days, "tc": tc, "attrs": attrs, "dist": dist}


@pytest.fixture(autouse=True)
def _clear_cache():
    dl.clear_cache()
    yield
    dl.clear_cache()


@pytest.fixture
def proj(tmp_path):
    return make_project(tmp_path)


def _lonlat(truth, r, c):
    lon, lat = dl._rc_to_lonlat(truth["attrs"], r, c)
    return float(lon), float(lat)


def _square(truth, r0, r1, c0, c1, props=None):
    """A lon/lat polygon around cells r0..r1, c0..c1 (edges between cells)."""
    pts = [_lonlat(truth, r - 0.5, c - 0.5) for r, c in ((r0, c0), (r0, c1 + 1), (r1 + 1, c1 + 1), (r1 + 1, c0))]
    pts.append(pts[0])
    return {"type": "Feature", "properties": props or {}, "geometry": {"type": "Polygon", "coordinates": [pts]}}


def _fc(*feats):
    return {"type": "FeatureCollection", "features": list(feats)}


def _write(path: Path, obj) -> Path:
    path.write_text(json.dumps(obj))
    return path


# ------------------------------------------------------------------ windows and dates
def test_parse_windows_forms_and_refusals():
    w = E.parse_windows(["2018-05-12:2019-04-03", ["20191101", "20210407"],
                         {"name": "LW904", "start": "2021-05-20", "end": "2022-08-09"}, "record"])
    assert [x["name"] for x in w] == ["2018-05-12_2019-04-03", "2019-11-01_2021-04-07", "LW904", "record"]
    assert w[1]["start"] == "2019-11-01" and w[3]["start"] is None
    assert E.parse_windows(True) == [{"name": "record", "start": None, "end": None}]
    assert E.parse_windows([]) == [] and E.parse_windows(None) == [] and E.parse_windows(False) == []
    assert E.parse_windows(E.parse_windows([True, "2018-05-12:2019-04-03"])) == E.parse_windows(
        [True, "2018-05-12:2019-04-03"])                                      # read twice: the same
    from datetime import date
    assert E.parse_windows([[date(2018, 5, 12), date(2019, 4, 3)]])[0]["end"] == "2019-04-03"   # YAML dates
    for bad in ("2019-04-03:2018-05-12", "05/02/2022:06/02/2022", "2018-05-12", [["2018-05-12"]],
                {"start": "2018-05-12"}, ["2018-13-01:2019-01-01"]):
        with pytest.raises(ValueError):
            E.parse_windows(bad)


def test_cumulative_between_is_the_difference_at_the_sar_dates_used(proj):
    cfg, T = proj
    arr, attrs = E.cumulative_between(cfg, T0, T1)
    days = T["days"]
    i0 = int(np.searchsorted(days, E._day(T0), side="right")) - 1        # last on or before
    i1 = int(np.searchsorted(days, E._day(T1), side="left"))             # first on or after
    expect = T["ts_mm"][i1] - T["ts_mm"][i0]
    ok = T["tc"] >= 0.4
    assert np.allclose(arr[ok], expect[ok], atol=1e-3)
    assert np.isnan(arr[~ok]).all()
    # the whole event: the trough's centre sank the full 300 mm (relative to the reference pixel)
    assert arr[CR, CC] == pytest.approx(-A_MM, abs=0.5)
    assert attrs["EVENT_EPOCH_START"] == E._iso(days[i0]) and attrs["EVENT_EPOCH_END"] == E._iso(days[i1])
    assert attrs["START_DATE"] == T["dates"][i0] and attrs["END_DATE"] == T["dates"][i1]
    assert attrs["EVENT_REQUESTED_START"] == T0 and attrs["EVENT_CLAMPED"] == "no"
    assert attrs["EVENT_RELIABLE_CELLS"] == str(int(ok.sum())) and "reference pixel" in attrs["EVENT_DATUM"]


def test_cumulative_between_snaps_clamps_interpolates_and_refuses(proj):
    cfg, T = proj
    days, ts = T["days"], T["ts_mm"]
    # nearest
    arr, a = E.cumulative_between(cfg, "2018-05-13", "2019-04-04", snap="nearest")
    j0 = int(np.argmin(np.abs(days - E._day("2018-05-13"))))
    j1 = int(np.argmin(np.abs(days - E._day("2019-04-04"))))
    assert arr[CR, CC] == pytest.approx(ts[j1, CR, CC] - ts[j0, CR, CC], abs=1e-3)
    # interpolated, exactly at the asked dates
    arr, a = E.cumulative_between(cfg, T0, T1, snap="interpolate")

    def at(d):
        x = E._day(d)
        return np.array([np.interp(x, days, ts[:, CR, CC])])[0]
    assert arr[CR, CC] == pytest.approx(at(T1) - at(T0), abs=1e-3)
    assert "interpolated" in a["EVENT_EPOCH_START"]
    # across the 36-day gap: no interpolated value
    gap_mid = E._iso((days[49] + days[50]) // 2)
    with pytest.raises(ValueError, match="more than 24 days apart"):
        E.cumulative_between(cfg, T0, gap_mid, snap="interpolate")
    # beyond the record: clamped and said so
    arr, a = E.cumulative_between(cfg, "2017-01-01", "2030-01-01")
    assert a["EVENT_CLAMPED"] == "yes" and a["EVENT_EPOCH_START"] == E._iso(days[0])
    assert a["EVENT_EPOCH_END"] == E._iso(days[-1])
    # a window wholly after the record is refused (both ends clamp to the last date)
    with pytest.raises(ValueError, match="no interval"):
        E.cumulative_between(cfg, "2030-01-01", "2031-01-01")
    with pytest.raises(ValueError, match="ends on or before"):
        E.cumulative_between(cfg, T1, T0)
    # a floor above every cell's coherence: nothing reliable, nothing filled
    arr, a = E.cumulative_between(cfg, T0, T1, floor=0.95)
    assert np.isnan(arr).all() and a["EVENT_RELIABLE_CELLS"] == "0"


# ------------------------------------------------------------------ panels
def test_panel_max_depth_timing_and_place(proj, tmp_path):
    cfg, T = proj
    gj = _write(tmp_path / "panels.geojson", _fc(
        _square(T, CR - 18, CR + 18, CC - 18, CC + 18, {"name": "LW902", "mined_start": T0, "mined_end": T1}),
        _square(T, CR - 18, CR + 18, CC - 18, CC + 18, {"name": "LW902 record"}),
        _square(T, 21, 28, 88, 105, {"name": "on incoherent ground"})))
    rows = E.panel_max(cfg, gj)
    assert [r["panel"] for r in rows] == ["LW902", "LW902 record", "on incoherent ground"]
    own, rec, hole = rows
    days = T["days"]
    first_after = T["dates"][int(np.searchsorted(days, E._day(T1)))]
    # depth: the 5th percentile of the panel's change (robust minimum), computed independently
    ok = T["tc"] >= 0.4
    sub = (slice(CR - 18, CR + 19), slice(CC - 18, CC + 19))
    i0 = int(np.searchsorted(days, E._day(T0), side="right")) - 1
    i1 = int(np.searchsorted(days, E._day(T1)))
    expect = np.percentile((T["ts_mm"][i1] - T["ts_mm"][i0])[sub][ok[sub]], 5)
    assert own["max_mm"] == pytest.approx(expect, abs=0.02)
    assert -A_MM <= own["max_mm"] < -0.9 * A_MM
    assert own["window"]["source"] == "the polygon's own dates" and own["window"]["requested_start"] == T0
    # timing: the trough stops at T1, so the deepest date is the first SAR date on or after it
    assert own["date_of_max"] == E._iso(E._day(first_after))
    assert own["max_at_window_end"]                   # its own window ends on the date the trough stopped
    assert rec["date_of_max"] == own["date_of_max"] and rec["window"]["source"] == "the whole record"
    assert not rec["max_at_window_end"]                # over the record it visibly stopped
    assert rec["max_mm"] == pytest.approx(own["max_mm"], abs=0.05)
    # half the depth was reached about half way through the event (linear sinking)
    half = E._day(own["date_half"])
    assert abs(half - (E._day(T0) + E._day(T1)) / 2) <= 13
    # place: the trough's centre cell
    lon_c, lat_c = _lonlat(T, CR, CC)
    assert abs(own["lat"] - lat_c) < 1.5 * abs(DLAT) and abs(own["lon"] - lon_c) < 1.5 * DLON
    assert own["vertical_assumed_mm"] == pytest.approx(own["max_mm"] / math.cos(math.radians(39.0)), abs=0.05)
    assert own["n_cells_in_polygon"] == 37 * 37 and own["coverage_share"] == 1.0
    assert own["series"][0]["date"] == E._iso(days[i0 + 1]) and own["deepest_cell"]["mm"] <= own["max_mm"]
    assert own["window"]["epoch_start"] == E._iso(days[i0]) and own["window"]["epoch_end"] == E._iso(days[i1])
    # incoherent ground: no value, and the reason
    assert hole["max_mm"] is None and "reliable cells" in hole["reason"]
    # each event window, for a polygon without dates
    rows = E.panel_max(cfg, _fc(_square(T, CR - 18, CR + 18, CC - 18, CC + 18, {"panel": "P"})),
                       windows=["2018-01-01:2018-05-01", f"{T0}:{T1}"])
    assert [r["window"]["source"] for r in rows] == ["event window", "event window"]
    assert abs(rows[0]["max_mm"]) < 1.0 and rows[1]["max_mm"] < -0.9 * A_MM


def test_panel_max_window_end_and_flag_shares(proj, tmp_path):
    cfg, T = proj
    # a window that ends while the trough is still sinking: the deepest value is on its last date
    rows = E.panel_max(cfg, _fc(_square(T, CR - 10, CR + 10, CC - 10, CC + 10, {"name": "P", "start": T0,
                                                                                   "end": "2018-11-01"})))
    assert rows[0]["max_at_window_end"] and rows[0]["date_of_max"] >= "2018-11-01"
    # the F2 aliasing layer (contract C-alias: 0 none, 1 risk, 2 likely aliased) and the old flag
    alias = np.zeros((NY, NX), np.uint8)
    alias[CR - 5:CR + 6, CC - 5:CC + 6] = 2
    fast = np.zeros((NY, NX), np.uint8)
    D = cfg.project_dir / "deliverables"
    dl._write_geotiff(D / "quality_aliasing_risk.tif", alias, T["attrs"], nodata=255, dtype="uint8")
    dl._write_geotiff(D / "quality_too_fast_to_measure.tif", fast, T["attrs"], nodata=255, dtype="uint8")
    r = E.panel_max(cfg, _fc(_square(T, CR - 10, CR + 10, CC - 10, CC + 10, {"name": "P"})))[0]
    fl = r["flags"]
    assert fl["quality_aliasing_risk.tif"]["share_likely_aliased"] == pytest.approx(121 / 441, abs=1e-3)
    assert fl["quality_too_fast_to_measure.tif"]["share_flagged"] == 0.0


def test_panel_max_ring_takes_off_what_the_ground_around_did(tmp_path):
    cfg, T = make_project(tmp_path, drift_mm_yr=-40.0)
    gj = _fc(_square(T, CR - 18, CR + 18, CC - 18, CC + 18, {"name": "P", "start": T0, "end": T1}),
             _square(T, 100, 110, 100, 110, {"name": "other panel"}))
    r = E.panel_max(cfg, gj, ring_m=(300.0, 600.0))[0]
    days = T["days"]
    i0 = int(np.searchsorted(days, E._day(T0), side="right")) - 1
    i1 = int(np.searchsorted(days, E._day(T1)))
    drift = -40.0 * (days[i1] - days[i0]) / 365.25
    ch = T["ts_mm"][i1] - T["ts_mm"][i0]
    sub = (slice(CR - 18, CR + 19), slice(CC - 18, CC + 19))
    trough_only = np.percentile(ch[sub] - drift, 5)
    # on the product's datum the panel reads the trough plus the field's drift ...
    assert r["max_mm"] == pytest.approx(np.percentile(ch[sub], 5), abs=0.02)
    assert r["max_mm"] == pytest.approx(trough_only + drift, abs=0.02)
    # ... the ring (300-600 m outside, the other panel + 300 m left out) is the drift plus the trough's tail,
    # computed independently on the cells: distance to the square, flat metres
    rr, cc = np.mgrid[0:NY, 0:NX]
    h = 18.5 * 16.6
    dy = np.maximum(np.abs(rr - CR) * 16.7 - h, 0)
    dx = np.maximum(np.abs(cc - CC) * 16.6 - h, 0)
    d = np.hypot(dx, dy)
    d_other = np.hypot(np.maximum(np.abs(rr - 105) * 16.7 - 5.5 * 16.6, 0),
                       np.maximum(np.abs(cc - 105) * 16.6 - 5.5 * 16.6, 0))
    ring = (d > 300) & (d <= 600) & (d_other > 300) & (T["tc"] >= 0.4)
    assert r["ring"]["median_mm"] == pytest.approx(np.median(ch[ring]), abs=1.0)
    assert r["ring"]["robust_mm"] == pytest.approx(np.percentile(ch[ring], 5), abs=1.0)
    assert abs(r["ring"]["cells"] - int(ring.sum())) <= 0.03 * ring.sum()
    # like for like: the panel's 5th percentile less the ring's 5th percentile, median less median
    assert r["max_mm_relative_to_ring"] == pytest.approx(r["max_mm"] - r["ring"]["robust_mm"], abs=0.011)
    assert r["median_mm_relative_to_ring"] == pytest.approx(r["median_mm_at_max"] - r["ring"]["median_mm"], abs=0.011)
    # the drift is gone; what is left over is the trough's own tail in a ring this close (300-600 m against a
    # 250 m trough), which makes the relative figure a little shallower, never deeper
    assert trough_only - 1.0 <= r["max_mm_relative_to_ring"] <= trough_only + 12.0
    assert r["max_mm"] < r["max_mm_relative_to_ring"] - 25
    # off the grid: no ring cells, no relative figure, said so
    r = E.panel_max(cfg, gj)[0]
    assert r["ring"]["cells"] == 0 and r["ring"]["median_mm"] is None and "max_mm_relative_to_ring" not in r
    assert "median_mm_relative_to_ring" not in r
    assert "ring" not in E.panel_max(cfg, gj, ring_m=None)[0]


def test_panel_max_ring_reads_zero_where_the_panel_moves_like_its_surroundings(tmp_path):
    """Reviewer, 8 Oct 2026. Ground with no trough, a field-wide drift and a
    spread of rates from cell to cell (as on farmland): the panel moves like
    the ground around it, so the ring figures must read about 0. The first
    version took the ring's MEDIAN from the panel's 5th percentile, which
    reads the spread (-1.645 sigma, here ~-90 mm) as subsidence; on the
    Site A record it put every Area A1 panel 22-173 mm below its
    ring, where the median of the panel and of the ring agree
    (review_2026-10-08/p23/events_review/ring_check.log)."""
    cfg, T = make_project(tmp_path)
    days = T["days"]
    t = (days - days[0]) / 365.25
    rng = np.random.default_rng(20261008)
    rate = -25.0 + 25.0 * rng.standard_normal((NY, NX))                     # mm/yr, cell by cell
    los = rate[None] * t[:, None, None]
    ts = (los - los[:, 5, 5][:, None, None]) / 1000.0
    ts_path = cfg.project_dir / "mintpy" / "geo" / "geo_timeseries_ERA5_demErr.h5"
    with h5py.File(ts_path, "r+") as f:
        f["timeseries"][...] = ts.astype(np.float32)
    dl.clear_cache()
    gj = _fc(_square(T, 45, 75, 45, 75, {"name": "quiet panel"}))
    r = E.panel_max(cfg, gj, ring_m=(150.0, 450.0))[0]
    ch = (ts[-1] - ts[0]) * 1000.0
    spread = 1.4826 * np.median(np.abs(ch - np.median(ch)))
    assert r["max_mm"] < -1.2 * spread                                    # the 5th percentile sits in the spread
    assert r["max_mm"] - r["ring"]["median_mm"] < -1.2 * spread           # what the first version reported
    assert abs(r["max_mm_relative_to_ring"]) < 0.15 * spread              # like for like: about 0
    assert abs(r["median_mm_relative_to_ring"]) < 0.15 * spread


# ------------------------------------------------------------------ profiles
def test_profile_bins_cells_within_the_buffer(proj, tmp_path):
    cfg, T = proj
    # west-east through the trough's centre, from column 20 to column 100
    a, b = _lonlat(T, CR, 20), _lonlat(T, CR, 100)
    line = {"type": "Feature", "properties": {"name": "XL1"}, "geometry": {"type": "LineString",
                                                                          "coordinates": [list(a), list(b)]}}
    # north-south through the incoherent block
    c, d = _lonlat(T, 10, 95), _lonlat(T, 40, 95)
    line2 = {"type": "Feature", "properties": {"name": "through the hole"},
             "geometry": {"type": "LineString", "coordinates": [list(c), list(d)]}}
    df = E.profile(cfg, _fc(line, line2), [T0, T1])
    x = df[df["line"] == "XL1"]
    from pyproj import Geod
    length = Geod(ellps="WGS84").inv(a[0], a[1], b[0], b[1])[2]
    assert len(x) == math.ceil(length / 50.0)
    assert (x["status"] == "ok").all()
    # the deepest bin is the middle one, and it holds the trough's centre value (median of its cells)
    k = int(x["change_mm"].idxmin())
    assert abs(x.loc[k, "distance_m"] - length / 2) <= 50
    assert x.loc[k, "change_mm"] < -0.9 * A_MM
    # cell counts: every cell within 100 m alongside the line, counted independently of events.py
    # (geodesic distances north-south from the line and east-west along it)
    rr, cc = np.mgrid[0:NY, 0:NX]
    lat = LAT0 + (rr + 0.5) * DLAT
    lon = LON0 + (cc + 0.5) * DLON
    g = Geod(ellps="WGS84")
    _, _, py = g.inv(np.full(lat.shape, a[0]), np.full(lat.shape, a[1]), np.full(lat.shape, a[0]), lat)
    _, _, along = g.inv(np.full(lat.shape, a[0]), lat, lon, lat)
    along = np.where(lon >= a[0], along, -along)
    near = (py <= 100.0) & (along >= 0) & (along <= length)
    first_bin = near & (along < 50.0)
    assert int(x["n_cells"].iloc[0]) == int(first_bin.sum())
    i0 = int(np.searchsorted(T["days"], E._day(T0), side="right")) - 1
    i1 = int(np.searchsorted(T["days"], E._day(T1)))
    change = T["ts_mm"][i1] - T["ts_mm"][i0]
    assert x["change_mm"].iloc[0] == pytest.approx(np.median(change[first_bin]), abs=0.01)   # the trough's tail
    assert x["n_independent"].iloc[0] == x["n_cells"].iloc[0]          # no radar grid: each cell its own pixel
    assert x["vertical_assumed_mm"].iloc[k] == pytest.approx(x["change_mm"].iloc[k] / math.cos(math.radians(39)),
                                                             abs=0.05)
    assert x["epoch"].iloc[0] == E._iso(E._day(T["dates"][int(np.searchsorted(T["days"], E._day(T1)))]))
    h = df[df["line"] == "through the hole"]
    holed = h[h["status"] != "ok"]
    assert len(holed) and holed["status"].str.contains("too few reliable cells").all()
    assert holed["change_mm"].isna().all()
    assert df.attrs["buffer_m"] == 100.0 and df.attrs["bin_m"] == 50.0
    # the last bin is shorter than 50 m: its middle is on the line, not beyond its end
    last = x.iloc[-1]
    assert last["distance_m"] == pytest.approx(((len(x) - 1) * 50.0 + length) / 2, abs=0.11)
    assert last["distance_m"] <= length


# ------------------------------------------------------------------ points
def test_values_at_is_the_harness_sampling_interpolated(proj):
    cfg, T = proj
    from insar_pipeline import validation as V
    lon_c, lat_c = _lonlat(T, CR, CC)
    lon_h, lat_h = _lonlat(T, 27, 97)
    pts = {"C": {"lat": lat_c, "lon": lon_c}, "H": {"lat": lat_h, "lon": lon_h}}
    days, dates = T["days"], T["dates"]
    mid = E._iso((days[10] + days[11]) // 2)
    gap_mid = E._iso((days[49] + days[50]) // 2)
    df = E.values_at(cfg, pts, [dates[10], mid, gap_mid, "2017-06-01", dates[30]], 100.0)
    prod = V.InsarProduct(cfg)
    s = V.sample_points(prod, pts, 100.0, [0.4], 5, 2)["C"][0.4]["series"]
    c = df[df["id"] == "C"].set_index("epoch")
    assert c.loc[E._iso(E._day(dates[10])), "value_mm"] == pytest.approx(s[10], abs=0.01)
    assert c.loc[E._iso(E._day(dates[30])), "value_mm"] == pytest.approx(s[30], abs=0.01)
    w = (E._day(mid) - days[10]) / (days[11] - days[10])
    assert c.loc[mid, "value_mm"] == pytest.approx((1 - w) * s[10] + w * s[11], abs=0.01)
    assert c.loc[mid, "sar_before"] == E._iso(days[10]) and c.loc[mid, "weight_after"] == pytest.approx(w, abs=1e-4)
    assert c.loc[gap_mid, "status"] == "no value" and "24 days" in c.loc[gap_mid, "why"]
    assert c.loc["2017-06-01", "status"] == "no value" and "outside" in c.loc["2017-06-01", "why"]
    first = c.loc[E._iso(E._day(dates[10])), "value_mm"]
    assert c.loc[E._iso(E._day(dates[30])), "change_mm"] == pytest.approx(s[30] - first, abs=0.02)
    assert c["los_up"].iloc[0] == pytest.approx(math.cos(math.radians(39.0)), abs=1e-4)
    h = df[df["id"] == "H"]
    assert (h.loc[h["epoch"] == E._iso(E._day(dates[10])), "status"] == "masked").all()
    assert int(h["reliable_cells"].iloc[0]) == 0 and int(h["cells_within_radius"].iloc[0]) > 0


def test_values_at_takes_a_truth_csvs_own_dates(proj, tmp_path):
    cfg, T = proj
    lon_c, lat_c = _lonlat(T, CR, CC)
    csv = tmp_path / "tahmoor_w_units.csv"
    csv.write_text("id,lat,lon,date,up_mm,sigma_mm,digitising_sigma_mm,provenance\n"
                   f"TW01,{lat_c},{lon_c},2018-05-12,0,8,5,Report 8 p12 fig 3\n"
                   f"TW01,{lat_c},{lon_c},2018-09-10,-120,8,5,Report 8 p12 fig 3\n"
                   f"TW01,{lat_c},{lon_c},2019-04-03,-240,8,5,Report 8 p12 fig 3\n")
    df = E.values_at(cfg, str(csv), None, 100.0)
    assert list(df["epoch"]) == ["2018-05-12", "2018-09-10", "2019-04-03"] and (df["status"] == "ok").all()
    assert df["change_mm"].iloc[0] == 0.0 and df["change_mm"].iloc[-1] < -0.9 * A_MM
    df2 = E.values_at(cfg, [(lat_c, lon_c)], ["2018-09-10"])
    assert df2["value_mm"].iloc[0] == pytest.approx(df["value_mm"].iloc[1], abs=1e-6)
    # GeoJSON points, here in a projected file that names its CRS (UTM 56S)
    from pyproj import Transformer
    x, y = Transformer.from_crs("EPSG:4326", "EPSG:32756", always_xy=True).transform(lon_c, lat_c)
    gj = {"type": "FeatureCollection", "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:EPSG::32756"}},
          "features": [{"type": "Feature", "properties": {"name": "TW01"},
                        "geometry": {"type": "Point", "coordinates": [x, y]}}]}
    df3 = E.values_at(cfg, gj, ["2018-09-10"])
    assert df3["id"].iloc[0] == "TW01" and df3["value_mm"].iloc[0] == pytest.approx(df["value_mm"].iloc[1], abs=1e-6)
    del gj["crs"]
    with pytest.raises(ValueError, match="not longitude/latitude"):
        E.values_at(cfg, gj, ["2018-09-10"])


# ------------------------------------------------------------------ a projected southern grid
def test_utm_56s_grid_gives_the_same_answers(tmp_path):
    cfg, T = make_project(tmp_path, projected=True)
    from pyproj import Transformer
    to_ll = Transformer.from_crs("EPSG:32756", "EPSG:4326", always_xy=True).transform

    def ll(r, c):
        return list(to_ll(UTM_X0 + (c + 0.5) * UTM_STEP, UTM_Y0 - (r + 0.5) * UTM_STEP))

    sq = [ll(CR - 18.5, CC - 18.5), ll(CR - 18.5, CC + 18.5), ll(CR + 18.5, CC + 18.5), ll(CR + 18.5, CC - 18.5)]
    gj = _fc({"type": "Feature", "properties": {"name": "P", "start": T0, "end": T1},
              "geometry": {"type": "Polygon", "coordinates": [sq + [sq[0]]]}})
    r = E.panel_max(cfg, gj)[0]
    lon_c, lat_c = ll(CR, CC)
    assert r["max_mm"] < -0.9 * A_MM
    assert abs(r["lat"] - lat_c) < 3e-4 and abs(r["lon"] - lon_c) < 3e-4      # within ~1.5 cells
    assert r["n_cells_in_polygon"] == 37 * 37
    line = _fc({"type": "Feature", "properties": {"name": "L"},
                "geometry": {"type": "LineString", "coordinates": [ll(CR, 20), ll(CR, 100)]}})
    df = E.profile(cfg, line, [T0, T1])
    assert len(df) == math.ceil(80 * UTM_STEP / 50.0)
    k = int(df["change_mm"].idxmin())
    assert abs(df.loc[k, "distance_m"] - 40 * UTM_STEP) <= 50 and df.loc[k, "change_mm"] < -0.9 * A_MM
    v = E.values_at(cfg, {"C": {"lat": lat_c, "lon": lon_c}}, [T1])
    arr, _ = E.cumulative_between(cfg, T["dates"][0], T["dates"][-1])
    assert v["status"].iloc[0] == "ok" and arr[CR, CC] == pytest.approx(-A_MM, abs=0.5)


# ------------------------------------------------------------------ the products folder
def _set(cfg, **kw):
    for k, v in kw.items():
        setattr(cfg, k, v)
    return cfg


def test_events_folder_is_written_when_event_products_is_set(proj, tmp_path):
    cfg, T = proj
    import rasterio
    panels = _write(tmp_path / "panels.geojson", _fc(
        _square(T, CR - 18, CR + 18, CC - 18, CC + 18, {"name": "LW902", "mined_start": T0, "mined_end": T1}),
        _square(T, 95, 115, 0, 20, {"name": "quiet"})))
    lines = _write(tmp_path / "lines.geojson", _fc({"type": "Feature", "properties": {"name": "XL1"},
                                                     "geometry": {"type": "LineString",
                                                                  "coordinates": [list(_lonlat(T, CR, 20)),
                                                                                  list(_lonlat(T, CR, 100))]}}))
    _set(cfg, event_products=[f"{T0}:{T1}", {"name": "after", "start": "2019-06-01", "end": "2020-01-01"}],
         event_polygons=str(panels), event_lines=str(lines))
    D = cfg.project_dir / "deliverables"
    rec = dl.export_event_products(cfg, D)
    assert rec["status"] == "written" and rec["maps"] == 2 and rec["panels"] == 3 and rec["profile_lines"] == 1
    ev = D / "events"
    man = json.loads((ev / "events.json").read_text())
    assert man["status"] == "written" and not man["errors"]
    tifs = sorted(ev.glob("los_change_mm_*.tif"))
    assert len(tifs) == 2 and all(t.with_suffix(".png").exists() for t in tifs)
    first = next(t for t in tifs if "after" not in t.name)
    with rasterio.open(first) as src:
        tags, a = src.tags(), src.read(1)
        assert src.crs.to_epsg() == 4326 and src.units[0] == "mm"
    assert tags["EVENT_REQUESTED_START"] == T0 and tags["EVENT_EPOCH_START"].replace("-", "") in first.name
    assert "line-of-sight change" in tags["QUANTITY"] and "Negative" in tags["SIGN_CONVENTION"]
    assert a[CR, CC] == pytest.approx(-A_MM, abs=0.5) and np.isnan(a[HOLE]).all()
    pm = pd.read_csv(ev / "panel_max.csv")
    # LW902 once (its own dates), "quiet" once per window
    assert list(pm["panel"]) == ["LW902", "quiet", "quiet"]
    assert pm.loc[0, "max_mm"] < -0.9 * A_MM and abs(pm.loc[1, "max_mm"]) < 1.0
    assert json.loads((ev / "panel_max.json").read_text())["panels"][0]["series"]
    prof = pd.read_csv(ev / "profiles.csv")
    assert set(prof["window"]) == {f"{T0}_{T1}", "after"} and (ev / "profile_XL1.png").exists()
    assert (ev / "README.txt").exists() and (ev / "panels.png").exists()
    # written again: the earlier write is moved aside, not mixed in; the whole record as the window
    rec2 = dl.export_event_products(_set(cfg, event_products=True, event_lines=""), D)
    assert rec2["status"] == "written" and rec2["maps"] == 1 and len(list(ev.glob("los_change_mm_*.tif"))) == 1
    pm = pd.read_csv(ev / "panel_max.csv")
    assert list(pm["window_source"]) == ["the polygon's own dates", "the whole record"]
    assert list((D / "superseded").glob("events_*/panel_max.csv"))
    # a bad polygon file: the rest is still written and the error is recorded
    bad = _write(tmp_path / "bad.geojson", {"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {}, "geometry": {"type": "Polygon", "coordinates": [
            [[500000, 6200000], [500100, 6200000], [500100, 6200100], [500000, 6200000]]]}}]})
    rec3 = dl.export_event_products(_set(cfg, event_polygons=str(bad)), D)
    assert rec3["status"] == "written with errors" and rec3["maps"] == 1
    assert "not longitude/latitude" in rec3["errors"][0]


def test_without_event_products_nothing_is_written_or_kept(proj):
    cfg, T = proj
    D = cfg.project_dir / "deliverables"
    assert cfg.event_products == [] and cfg.event_polygons == "" and cfg.event_lines == ""
    assert dl.export_event_products(cfg, D) is None and not (D / "events").exists()
    # an earlier run's events folder is moved aside when the key is off
    (D / "events").mkdir()
    (D / "events" / "panel_max.csv").write_text("old")
    assert dl.export_event_products(cfg, D) is None
    assert not any((D / "events").iterdir())
    assert list((D / "superseded").glob("events_*/panel_max.csv"))


def test_a_failure_is_recorded_and_earlier_files_moved_aside(proj):
    cfg, T = proj
    D = cfg.project_dir / "deliverables"
    (D / "events").mkdir()
    (D / "events" / "panel_max.csv").write_text("old")
    rec = dl.export_event_products(_set(cfg, event_products=["2019-04-03:2018-05-12"]), D)
    assert rec["status"] == "failed" and "ends on or before" in rec["error"]
    assert sorted(p.name for p in (D / "events").iterdir()) == ["events.json"]
    assert json.loads((D / "events" / "events.json").read_text())["status"] == "failed"
    assert list((D / "superseded").glob("events_*/panel_max.csv"))


def test_export_all_writes_the_event_products(proj, monkeypatch):
    """The products stage (export_all) calls the event products and records
    them in summary.json; every other exporter is stubbed here."""
    cfg, T = proj
    for name in ("supersede_results", "export_velocity", "export_velocity_uncertainty", "export_empirical_uncertainty",
                 "export_cell_noise", "export_signal_to_noise", "export_vertical_velocity", "export_quality",
                 "export_extra_layers", "export_cumulative_displacement", "plot_displacement_series",
                 "plot_selected_points", "export_measurement_points", "export_coverage_curve",
                 "_apply_qa_and_gnss"):
        monkeypatch.setattr(dl, name, lambda *a, **k: None)
    monkeypatch.setattr(dl, "summary_stats", lambda cfg: {})
    monkeypatch.setattr(dl.provenance, "write", lambda *a, **k: None)
    _set(cfg, event_products=[f"{T0}:{T1}"], gnss_validation=False)
    dl.export_all(cfg)
    S = json.loads((cfg.project_dir / "deliverables" / "summary.json").read_text())
    assert S["event_products"]["status"] == "written" and S["event_products"]["maps"] == 1
    assert list((cfg.project_dir / "deliverables" / "events").glob("los_change_mm_*.tif"))
    _set(cfg, event_products=[])
    dl.export_all(cfg)
    assert "event_products" not in json.loads((cfg.project_dir / "deliverables" / "summary.json").read_text())


# ------------------------------------------------------------------ config keys
def _yaml(tmp_path, extra=""):
    y = tmp_path / "c.yaml"
    y.write_text(f"project_dir: {tmp_path / 'p'}\nbbox: [150.55, -34.30, 150.78, -34.14]\n"
                 "start_date: '2018-01-01'\nend_date: '2019-12-31'\nbackend: hyp3\n" + extra)
    return y


def test_config_keys_load_default_off_and_refuse_bad_values(tmp_path):
    from insar_pipeline.config import ConfigError, load_config
    cfg = load_config(str(_yaml(tmp_path)))
    assert cfg.event_products == [] and cfg.event_polygons == "" and cfg.event_lines == ""
    T = {"attrs": {"X_FIRST": str(LON0), "Y_FIRST": str(LAT0), "X_STEP": str(DLON), "Y_STEP": str(DLAT)}}
    gj = _write(tmp_path / "panels.geojson", _fc(_square(T, 10, 20, 10, 20, {"name": "LW902", "start": T0,
                                                                              "end": T1})))
    cfg = load_config(str(_yaml(tmp_path, "event_products:\n  - 2018-05-12:2019-04-03\n  - [2019-06-01, 2019-12-01]\n"
                                          f"event_polygons: {gj}\n")))
    assert len(E.parse_windows(cfg.event_products)) == 2 and cfg.event_polygons == str(gj)
    for extra, words in (("event_products: [2019-04-03:2018-05-12]\n", "ends on or before"),
                         ("event_products: ['05/02/2019:06/02/2019']\n", "write it as YYYY-MM-DD"),
                         ("event_products: [['2019-02-30', '2019-06-01']]\n", "not a date"),
                         (f"event_polygons: {gj}\n", "event_products is empty"),
                         (f"event_products: true\nevent_lines: {tmp_path / 'missing.geojson'}\n", "does not exist"),
                         (f"event_products: true\nevent_lines: {gj}\n", "holds no line feature")):
        with pytest.raises(ConfigError, match=words):
            load_config(str(_yaml(tmp_path, extra)))
    bad = _write(tmp_path / "bad_dates.geojson", _fc(_square(T, 10, 20, 10, 20, {"name": "X", "start": T0})))
    with pytest.raises(ConfigError, match="only one of its dates"):
        load_config(str(_yaml(tmp_path, f"event_products: true\nevent_polygons: {bad}\n")))


def test_command_line_writes_into_out(proj, tmp_path, capsys):
    cfg, T = proj
    y = tmp_path / "exp.yaml"
    y.write_text(f"project_dir: {cfg.project_dir}\nbbox: [150.6, -34.17, 150.62, -34.15]\n"
                 "start_date: '2018-01-01'\nend_date: '2020-04-30'\nbackend: hyp3\n")
    gj = _write(tmp_path / "p.geojson", _fc(_square(T, CR - 10, CR + 10, CC - 10, CC + 10, {"name": "P"})))
    out = tmp_path / "ev"
    assert E.main([str(y), "--between", T0, T1, "--polygons", str(gj), "--out", str(out)]) == 0
    assert (out / "panel_max.csv").exists() and len(list(out.glob("los_change_mm_*.tif"))) == 1
    assert not (cfg.project_dir / "deliverables" / "events").exists()
    assert E.main([str(y), "--out", str(out)]) == 2                    # nothing asked for


# ------------------------------------------------------------------ M1: the too-fast flag's wavelength
def test_product_wavelength_from_the_product_not_a_c_band_constant():
    cfg = PipelineConfig(project_dir=Path("/tmp/x"), bbox=[0, 0, 1, 1], start_date="2020-01-01",
                         end_date="2020-12-31", backend="nisar")
    w, src = dl.product_wavelength_m(cfg, {"WAVELENGTH": "0.2384"})
    assert w == pytest.approx(0.2384) and "WAVELENGTH" in src
    with pytest.raises(ValueError, match="not Sentinel-1"):
        dl.product_wavelength_m(cfg, {})
    with pytest.raises(ValueError):
        dl.product_wavelength_m(PipelineConfig(project_dir=Path("/tmp/x"), bbox=[0, 0, 1, 1],
                                               start_date="2020-01-01", end_date="2020-12-31", backend="hyp3"),
                                {"PLATFORM": "NISAR"})
    cfg.backend = "miaplpy"
    w, src = dl.product_wavelength_m(cfg, {"PLATFORM": "sen"})
    assert w == dl.S1_WAVELENGTH_M and "Sentinel-1" in src


def test_too_fast_flag_uses_the_product_wavelength(tmp_path, caplog):
    import sys
    import rasterio
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import test_fix_L7a as L7a                                      # the deliverables project builder
    cfg, _ = L7a.make_project(tmp_path)
    vp = cfg.mintpy_dir / "geo" / "geo_velocity.h5"
    with h5py.File(vp, "r+") as f:                                  # one neighbour step of 0.4 m/yr
        v = f["velocity"][:]
        v[20, 30] = v[20, 29] + 0.4
        f["velocity"][...] = v
        f.attrs["WAVELENGTH"] = "0.05546576"
    D = cfg.project_dir / "deliverables"
    dl.clear_cache()
    dl.export_extra_layers(cfg, D)
    with rasterio.open(D / "quality_too_fast_to_measure.tif") as src:
        c_flag, tags = src.read(1), src.tags()
        desc = src.descriptions[0]
    assert c_flag[20, 30] == 1                                      # monthly C-band: ~163 mm/yr limit
    c_limit = float(tags["GRADIENT_LIMIT_MM_YR"])
    assert tags["WAVELENGTH_SOURCE"] == "the product's WAVELENGTH attribute"
    assert float(tags["QUARTER_WAVELENGTH_MM"]) == pytest.approx(13.87, abs=0.01)
    assert "lower bound" in desc.lower() and tags["LOWER_BOUND"].startswith("yes")
    # L-band (NISAR, ~0.24 m): a quarter wavelength 4.3x larger, the same step is measurable
    with h5py.File(vp, "r+") as f:
        f.attrs["WAVELENGTH"] = "0.2384"
    dl.clear_cache()
    dl.export_extra_layers(cfg, D)
    with rasterio.open(D / "quality_too_fast_to_measure.tif") as src:
        assert src.read(1)[20, 30] == 0 and float(src.tags()["GRADIENT_LIMIT_MM_YR"]) > 4 * c_limit
    # a NISAR product that states no wavelength: no flag at all, never a C-band one
    with h5py.File(vp, "r+") as f:
        del f.attrs["WAVELENGTH"]
    cfg.backend = "nisar"
    dl.clear_cache()
    with caplog.at_level("WARNING"):
        dl.export_extra_layers(cfg, D)
    assert not (D / "quality_too_fast_to_measure.tif").exists()
    assert "states no radar wavelength" in caplog.text


# ------------------------------------------------------------------ flags: stale layers, the interval's own flag
def test_a_flag_layer_older_than_the_time_series_is_not_used(proj):
    """A layer left from an earlier inversion (older than the time series)
    would report another run's flags beside this run's figures."""
    import os
    cfg, T = proj
    alias = np.full((NY, NX), 2, np.uint8)
    D = cfg.project_dir / "deliverables"
    p = D / "quality_aliasing_risk.tif"
    dl._write_geotiff(p, alias, T["attrs"], nodata=255, dtype="uint8")
    ts = cfg.project_dir / "mintpy" / "geo" / "geo_timeseries_ERA5_demErr.h5"
    t = ts.stat().st_mtime
    os.utime(p, (t - 3600, t - 3600))                               # an hour older than the series
    sq = _fc(_square(T, CR - 5, CR + 5, CC - 5, CC + 5, {"name": "P", "start": T0, "end": T1}))
    prod = E._Product(cfg)
    r = E.panel_max(cfg, sq, _prod=prod)[0]
    assert "quality_aliasing_risk.tif" not in (r.get("flags") or {})
    assert any("older than" in n for n in prod.flags_note)
    os.utime(p, (t + 60, t + 60))                                   # made after the series: used
    r = E.panel_max(cfg, sq)[0]
    assert r["flags"]["quality_aliasing_risk.tif"]["share_likely_aliased"] == 1.0


def test_panels_and_maps_carry_the_aliasing_flag_of_their_own_interval(proj, tmp_path, monkeypatch):
    """aliasing.risk_between (F2 per interval, protocol A's flag definition)
    is asked for the panel's figure's own dates, and for each map's."""
    from insar_pipeline import aliasing
    cfg, T = proj
    asked = []

    def fake_risk(cfg_, d0, d1, index=None):
        asked.append((d0, d1))
        a = np.zeros((NY, NX), np.uint8)
        a[CR - 5:CR + 6, CC - 5:CC + 6] = 2                         # 121 cells likely aliased
        a[CR - 10:CR - 5, CC - 10:CC + 11] = 1                      # 105 cells at risk
        a[:3] = 255                                                 # outside the radar crop
        return a
    monkeypatch.setattr(aliasing, "risk_between", fake_risk)
    monkeypatch.setattr(aliasing, "enabled", lambda c: True)
    sq = _fc(_square(T, CR - 10, CR + 10, CC - 10, CC + 10, {"name": "P", "start": T0, "end": T1}))
    r = E.panel_max(cfg, sq)[0]
    iv = r["flags"]["aliasing_risk_interval"]
    days = T["days"]
    i0 = int(np.searchsorted(days, E._day(T0), side="right")) - 1
    assert asked[-1] == (T["dates"][i0], r["date_of_max"].replace("-", ""))      # the figure's own interval
    assert iv["cells"] == 441 and iv["share_likely_aliased"] == pytest.approx(121 / 441, abs=1e-4)
    assert iv["share_flagged"] == pytest.approx(226 / 441, abs=1e-4)
    # the products folder: the panel CSV and each map's record carry it
    gj = _write(tmp_path / "p.geojson", sq)
    _set(cfg, event_products=[f"{T0}:{T1}"], event_polygons=str(gj))
    D = cfg.project_dir / "deliverables"
    dl.export_event_products(cfg, D)
    pm = pd.read_csv(D / "events" / "panel_max.csv")
    assert pm.loc[0, "likely_aliased_share_interval"] == pytest.approx(121 / 441, abs=1e-4)
    man = json.loads((D / "events" / "events.json").read_text())
    mi = man["maps"][0]["flags"]["aliasing_risk_interval"]
    assert mi["cells"] == int((T["tc"][3:] >= 0.4).sum())          # reliable cells inside the crop
    # the diagnostic off for this config: not asked, not reported
    monkeypatch.setattr(aliasing, "enabled", lambda c: False)
    n = len(asked)
    r = E.panel_max(cfg, sq)[0]
    assert len(asked) == n and "aliasing_risk_interval" not in (r.get("flags") or {})


def test_out_folder_keeps_files_that_are_not_event_products(proj, tmp_path):
    """--out may name a folder that holds other things: only the event
    products of an earlier write are moved aside."""
    cfg, T = proj
    out = tmp_path / "shared"
    out.mkdir()
    (out / "validation.json").write_text("{}")
    (out / "notes.txt").write_text("keep me")
    E.write_event_products(cfg, out, windows=[f"{T0}:{T1}"])
    E.write_event_products(cfg, out, windows=[f"{T0}:{T1}"])         # again: the first write moves aside
    assert (out / "validation.json").exists() and (out / "notes.txt").read_text() == "keep me"
    moved = list((tmp_path / "superseded").glob("shared_*"))
    assert len(moved) == 1 and (moved[0] / "events.json").exists()
    assert not (moved[0] / "validation.json").exists() and not (moved[0] / "notes.txt").exists()


def test_export_all_writes_the_aliasing_layer_before_the_event_products(proj, monkeypatch):
    """The products stage writes quality_aliasing_risk.tif (contract C-alias,
    aliasing.export_products) after the extra layers and before the event
    products, so a panel's flagged share is this run's; a client config
    (diagnostic off) gets neither call nor key."""
    from insar_pipeline import aliasing
    cfg, T = proj
    order = []
    for name in ("supersede_results", "export_velocity", "export_velocity_uncertainty", "export_empirical_uncertainty",
                 "export_cell_noise", "export_signal_to_noise", "export_vertical_velocity", "export_quality",
                 "export_cumulative_displacement", "plot_displacement_series",
                 "plot_selected_points", "export_measurement_points", "export_coverage_curve",
                 "_apply_qa_and_gnss"):
        monkeypatch.setattr(dl, name, lambda *a, **k: None)
    monkeypatch.setattr(dl, "export_extra_layers", lambda *a, **k: order.append("extra_layers"))
    monkeypatch.setattr(dl, "summary_stats", lambda cfg: {})
    monkeypatch.setattr(dl.provenance, "write", lambda *a, **k: None)

    def fake_export(cfg_, out_dir=None):
        order.append("aliasing")
        a = np.zeros((NY, NX), np.uint8)
        a[CR - 5:CR + 6, CC - 5:CC + 6] = 2
        dl._write_geotiff(Path(out_dir) / "quality_aliasing_risk.tif", a, T["attrs"], nodata=255, dtype="uint8")
        return {"tif": str(Path(out_dir) / "quality_aliasing_risk.tif"), "pairs": 3, "flagged_share": 0.0084}
    monkeypatch.setattr(aliasing, "export_products", fake_export)
    monkeypatch.setattr(aliasing, "enabled", lambda c: True)
    monkeypatch.setattr(aliasing, "risk_between", lambda *a, **k: None)
    gj = _write(cfg.project_dir / "p.geojson",
                _fc(_square(T, CR - 10, CR + 10, CC - 10, CC + 10, {"name": "P", "start": T0, "end": T1})))
    _set(cfg, event_products=[f"{T0}:{T1}"], event_polygons=str(gj), gnss_validation=False)
    dl.export_all(cfg)
    assert order == ["extra_layers", "aliasing"]
    D = cfg.project_dir / "deliverables"
    S = json.loads((D / "summary.json").read_text())
    assert S["aliasing_risk"]["pairs"] == 3
    pm = pd.read_csv(D / "events" / "panel_max.csv")
    assert pm.loc[0, "likely_aliased_share"] == pytest.approx(121 / 441, abs=1e-4)
    # a client project: the diagnostic is off, nothing is called and summary.json has no key
    monkeypatch.setattr(aliasing, "enabled", lambda c: False)
    order.clear()
    _set(cfg, event_products=[], event_polygons="")
    dl.export_all(cfg)
    assert order == ["extra_layers"]
    S = json.loads((D / "summary.json").read_text())
    assert "aliasing_risk" not in S and "event_products" not in S


def test_diagnostic_off_moves_an_earlier_aliasing_flag_aside(proj, monkeypatch):
    """Reviewer, 8 Oct 2026. With the F2 diagnostic off, export_aliasing_products
    returned before aliasing.export_products, whose own set-aside never ran: a
    quality_aliasing_risk.tif from an earlier run with it on stayed in
    deliverables/, to be packaged and read (by the harness) as this run's. It
    is now moved to superseded/; a project without one is left untouched."""
    from insar_pipeline import aliasing
    cfg, T = proj
    monkeypatch.setattr(aliasing, "enabled", lambda c: False)
    called = []
    monkeypatch.setattr(aliasing, "export_products", lambda *a, **k: called.append(1))
    D = cfg.project_dir / "deliverables"
    assert dl.export_aliasing_products(cfg, D) is None                    # nothing there: nothing made or moved
    assert not (D / "superseded").exists() and not called
    dl._write_geotiff(D / "quality_aliasing_risk.tif", np.zeros((NY, NX), np.uint8), T["attrs"], nodata=255,
                      dtype="uint8")
    (D / "aliasing_risk.json").write_text("{}")
    assert dl.export_aliasing_products(cfg, D) is None
    assert not (D / "quality_aliasing_risk.tif").exists() and not (D / "aliasing_risk.json").exists()
    assert len(list((D / "superseded").glob("aliasing_*/quality_aliasing_risk.tif"))) == 1
    assert len(list((D / "superseded").glob("aliasing_*/aliasing_risk.json"))) == 1 and not called
