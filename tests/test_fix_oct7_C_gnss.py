"""7 Oct 2026 fixes, group C_gnss (gnss_tie.py). Each test here failed on the
code in backups/pre_fixes_20261007/ unless its docstring says otherwise.

#8  a station inside the area with no usable radar cells crashed the tie
    (and with it the client report stage);
#9  a tie with no regional floor was anchored with its formal error only.
"""
import json
import types
from datetime import datetime, timedelta

import numpy as np
import pytest

import test_fix_L1 as L1T                                        # the L1 tie fixture


@pytest.fixture
def project(tmp_path, monkeypatch):
    return L1T.project.__wrapped__(tmp_path, monkeypatch)


def _dates(n=30, start="20160105", step_days=120):
    d0 = datetime.strptime(start, "%Y%m%d")
    return [(d0 + timedelta(days=step_days * i)).strftime("%Y%m%d") for i in range(n)]


def _write_ts(path, dates, shape=(40, 40)):
    import h5py
    rng = np.random.default_rng(1)
    with h5py.File(path, "w") as f:
        f["date"] = np.array(dates, dtype="S8")
        f["timeseries"] = rng.normal(0, 0.002, (len(dates),) + shape).astype(np.float32)


# ---------------------------------------------------------------- #8 _series_check with nothing to compare
def _gnss_series():
    t = np.arange(2015.5, 2026.5, 1 / 365.25)
    return {"t": t, "u": -1.0 * (t - t[0]) + np.random.default_rng(2).normal(0, 1.0, t.size)}


@pytest.fixture
def series(tmp_path, monkeypatch):
    from insar_pipeline import deliverables as dl
    dates = _dates()
    ts = tmp_path / "geo_timeseries_test.h5"
    _write_ts(ts, dates)
    monkeypatch.setattr(dl, "timeseries_file", lambda c: ts)
    cfg = types.SimpleNamespace(project_dir=tmp_path / "p")
    return cfg, dates


def test_series_check_with_good_cells_still_compares(series):
    """Control (passes on the old code too): the path the guards sit on."""
    from insar_pipeline import gnss_tie as gt
    cfg, dates = series
    out = gt._series_check(cfg, dates, _gnss_series(), 20, 20, np.full((40, 40), 0.78), np.ones((40, 40), bool))
    assert out["overlap_sar_dates"] == 30 and "series_check" not in out


def test_series_check_station_in_a_masked_window_does_not_raise(series):
    from insar_pipeline import gnss_tie as gt
    cfg, dates = series
    good = np.ones((40, 40), bool)
    good[17:24, 17:24] = False                                     # the station's 7x7 window: trees / water
    out = gt._series_check(cfg, dates, _gnss_series(), 20, 20, np.full((40, 40), 0.78), good)
    assert set(out) == {"series_check"} and "no usable radar" in out["series_check"]


def test_series_check_without_geometry_at_the_station_does_not_raise(series):
    from insar_pipeline import gnss_tie as gt
    cfg, dates = series
    up = np.full((40, 40), 0.78)
    up[20, 20] = np.nan
    out = gt._series_check(cfg, dates, _gnss_series(), 20, 20, up, np.ones((40, 40), bool))
    assert set(out) == {"series_check"} and "no usable radar" in out["series_check"]


def test_series_check_needs_20_dates_with_both_series(series):
    """A GNSS gap leaves 11 of the 30 SAR dates with both series: below the
    20-date rule the overlap test already states."""
    from insar_pipeline import gnss_tie as gt
    cfg, dates = series
    g = _gnss_series()
    t_sar = np.array([gt._decyear(d) for d in dates])
    gap = (g["t"] > t_sar[9] + 7 / 365) & (g["t"] < t_sar[-1] - 7 / 365)   # no GNSS for SAR dates 10-28
    g = {"t": g["t"][~gap], "u": g["u"][~gap]}
    out = gt._series_check(cfg, dates, g, 20, 20, np.full((40, 40), 0.78), np.ones((40, 40), bool))
    assert set(out) == {"series_check"} and "fewer than 20" in out["series_check"]


def test_tie_survives_a_station_inside_the_area_with_no_usable_cells(project, monkeypatch):
    """The finding's scenario end to end: tie() raised TypeError from polyfit,
    run() wrote an untied file and re-raised, and the report stage failed."""
    import h5py
    from insar_pipeline import gnss_tie as gt
    monkeypatch.setitem(L1T.SITES, "HHHH", (39.0, -87.30, 2010.5, 2026.6, -1.0))   # the box centre: row 20, col 20
    monkeypatch.setattr(gt, "discover", lambda cfg, root, radius=None: ["AAAA", "HHHH"])
    dates = _dates()
    _write_ts(project.geo / "geo_timeseries_test.h5", dates)
    with h5py.File(project.geo / "geo_velocity.h5", "r+") as f:
        v = f["velocity"][:]
        v[17:24, 17:24] = np.nan                                   # no radar cell around the station
        f["velocity"][...] = v
    out = gt.run(project.cfg)
    chk = [c for c in out["inside_checks"] if c["site"] == "HHHH"]
    assert len(chk) == 1 and chk[0]["cells"] == 0
    assert "no usable radar" in chk[0]["series_check"]
    assert out["anchored"] is True and out["record_window"] == [dates[0], dates[-1]]


# ---------------------------------------------------------------- #9 no regional floor, no tie
def test_a_null_floor_is_stale():
    from insar_pipeline import gnss_tie as gt
    tie = {"uncertainty_floor_mm_yr": None,
           "background": {"vertical_mm_yr": -1.0, "vertical_uncertainty_mm_yr": 0.29, "uncertainty_basis": "formal error"}}
    why = gt.stale_reason(tie)
    assert why and "too narrow" in why and "re-run the GNSS tie" in why
    assert gt.stale_reason(dict(tie, background=dict(tie["background"], uncertainty_basis=None)))
    # the floored tie and an unanchored record stay usable
    assert gt.stale_reason(dict(tie, uncertainty_floor_mm_yr=0.91)) is None
    assert gt.stale_reason({"uncertainty_floor_mm_yr": None, "background": None}) is None


def test_check_refuses_an_anchored_tie_with_a_null_floor(tmp_path):
    from insar_pipeline import gnss_tie as gt
    d = tmp_path / "p" / "deliverables"; d.mkdir(parents=True)
    (d / "gnss_tie.json").write_text(json.dumps({
        "record_window": L1T.DATES, "uncertainty_floor_mm_yr": None, "anchored": True,
        "background": {"vertical_mm_yr": -1.0, "vertical_uncertainty_mm_yr": 0.29, "uncertainty_basis": "formal error"}}))
    cfg = types.SimpleNamespace(project_dir=tmp_path / "p", mintpy_dir=tmp_path / "p" / "mintpy")
    tie, why = gt.check(cfg, L1T.DATES)
    assert tie is None and "re-run the GNSS tie" in why


def test_too_few_regional_stations_leaves_the_record_unanchored(project):
    """Only AAAA (and BBBB, too short) in the catalogue within 100 km: the
    floor cannot be measured, so the single-station tie is not anchored at
    +/- its formal error (Site D INLN 0.29 against a 0.91 floor)."""
    from insar_pipeline import gnss_tie as gt
    midas = project.cache / "unr_midas_igs14.txt"
    midas.write_text("".join(l + "\n" for l in midas.read_text().splitlines() if l.split()[0] in ("AAAA", "BBBB")))
    out = gt.run(project.cfg)
    assert out["uncertainty_floor_mm_yr"] is None and out["regional_scatter"]["catalogue_cached"] is True
    assert out["anchored"] is False and out["background"] is None and out["leave_one_out"] is None
    assert out["reason"].startswith("Fewer than 3 GNSS stations within 100 km")
    assert out["reason"].endswith(gt.UNTIED_SUFFIX)
    assert out["stations_used"] == ["AAAA"]                         # the station's own rate is kept
    on_disk = gt.load_valid(project.cfg, L1T.DATES)
    assert on_disk["background"] is None and on_disk["reason"] == out["reason"]
    assert gt.stale_reason(on_disk) is None


def test_no_catalogue_leaves_the_record_unanchored_and_says_why(project):
    from insar_pipeline import gnss_tie as gt
    (project.cache / "unr_midas_igs14.txt").unlink()
    (project.cache / "unr_llh.out").unlink()
    out = gt.run(project.cfg)
    assert out["anchored"] is False and out["background"] is None
    assert out["reason"].startswith("The regional list of GNSS stations could not be downloaded")
    assert "relative to the surrounding farmland" in out["reason"]
