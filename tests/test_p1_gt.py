"""P1 V2 (review 8 Oct 2026, group 'gt'): ground_truth.py, the truth readers
the validation harness uses, and the reader additions in gnss_tie.py.

What is checked (PLAN.md section 3 V2 and 5.1; protocol_*.yaml
gnss_screening, epoch_matching, station roles):
  - NGL IGS20 series: equipment steps removed as offsets estimated LOCALLY
    (a non-linear signal does not leak into a step), earthquake steps only
    when asked, antenna-height slips dropped and real changes treated as
    steps, outliers beyond 5 robust sigma removed as whole days, gaps never
    filled, sigma = max(robust, formal), window, datum, provenance attrs;
  - the screening models (linear, piecewise with water-year hinges, local
    line that keeps a fast event at the edge of a gap);
  - downloads only into the cache passed, refresh rules, 404 and offline
    behaviour, the steps file required;
  - station_rate (linear with lag-1 autocorrelation, MIDAS), the
    displacement comparison under 1 year, the +-3 day epoch matching;
  - user/digitised CSV truth (wide and long), the GA CEMP workbook,
    stations_in_box (live inventory, cached, frozen fallback), protocol
    roles (the anchor/validation split);
  - gnss_tie.read_tenv3's default output unchanged.
ground_truth.py did not exist before P1 and read_tenv3 had no `full`, so every
test fails on the old tree. No network (urlopen is replaced where a download
is exercised); nothing under ~/insar_projects is touched.
"""
from __future__ import annotations

import io
import math
import os
import time
import urllib.error
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from insar_pipeline import gnss_tie
from insar_pipeline import ground_truth as gt

PKG = Path(__file__).resolve().parents[1]
REVIEW = PKG / "review_2026-10-08"
MON = ("JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC")
HEADER = ("site YYMMMDD yyyy.yyyy __MJD week d reflon _e0(m) __east(m) ____n0(m) _north(m) u0(m) ____up(m) "
          "_ant(m) sig_e(m) sig_n(m) sig_u(m) __corr_en __corr_eu __corr_nu _latitude(deg) _longitude(deg) "
          "__height(m)")


def ngl(d: date) -> str:
    return f"{d.year % 100:02d}{MON[d.month - 1]}{d.day:02d}"


def write_tenv3(path: Path, site: str, dates, e, n, u, ant=None, lat=36.3, lon=-119.6, h=60.0,
                sig=(1.0, 1.2, 4.0)) -> Path:
    """A synthetic NGL tenv3 (values in mm in, metres out, integer + fraction)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [HEADER]
    for i, d in enumerate(dates):
        dec = 2000 + (d - date(2000, 1, 1)).days / 365.25
        mjd = (d - date(1858, 11, 17)).days
        parts = []
        for v, base in ((e[i], 0), (n[i], 0), (u[i], 60000)):
            m = (v + base) / 1000.0
            i0 = math.floor(m)
            parts += [str(i0), f"{m - i0:.6f}"]
        a = 0.0 if ant is None else ant[i]
        lines.append(f"{site} {ngl(d)} {dec:.4f} {mjd} 2000 1 {lon:.1f} {' '.join(parts)} {a:.4f} "
                     f"{sig[0] / 1000:.6f} {sig[1] / 1000:.6f} {sig[2] / 1000:.6f} 0.0 0.0 0.0 "
                     f"{lat:.10f} {lon:.10f} {h:.5f}")
    path.write_text("\n".join(lines) + "\n")
    return path


def write_steps(cache: Path, entries=()) -> Path:
    """NGL steps file: entries (site, date, code[, text]); padded with other
    sites so it looks like the real (large) file."""
    lines = []
    for site, d, code, *rest in entries:
        if code == 1:
            lines.append(f"{site}  {ngl(d)}  1  {rest[0] if rest else 'Antenna_Type_Changed'}")
        else:
            lines.append(f"{site}  {ngl(d)}  2   257.040   202.595  6.4 ci38443183")
    lines += [f"Z{i:03d}  10JAN01  1  Antenna_Type_Changed" for i in range(80)]
    p = cache / gnss_tie.STEPS_FILE
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(lines) + "\n")
    return p


def days(d0: date, d1: date, skip=()):
    out, d = [], d0
    while d <= d1:
        if not any(a <= d <= b for a, b in skip):
            out.append(d)
        d += timedelta(days=1)
    return out


def yrs(dates, d0):
    return np.array([(d - d0).days / 365.25 for d in dates])


def series(dates, rate_u=-50.0, rate_e=10.0, annual=5.0, noise=(1.0, 1.0, 3.0), seed=1):
    rng = np.random.default_rng(seed)
    t = yrs(dates, dates[0])
    e = rate_e * t + rng.normal(0, noise[0], t.size)
    n = 2.0 * t + rng.normal(0, noise[1], t.size)
    u = rate_u * t + annual * np.sin(2 * np.pi * t) + rng.normal(0, noise[2], t.size)
    return t, e, n, u


@pytest.fixture
def cache(tmp_path):
    return tmp_path / "cache"


@pytest.fixture
def no_net(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("no download expected in this test")
    monkeypatch.setattr(gt.urllib.request, "urlopen", boom)


# ----------------------------------------------------------------- gnss_tie readers
def test_read_tenv3_default_unchanged_full_adds_sigmas_and_dates(tmp_path):
    ds = days(date(2022, 1, 1), date(2022, 1, 10))
    _, e, n, u = series(ds)
    p = write_tenv3(tmp_path / "AAAA.tenv3", "AAAA", ds, e, n, u, sig=(1.5, 2.0, 5.0))
    g = gnss_tie.read_tenv3(p)
    assert set(g) == {"site", "t", "e", "n", "u", "ant", "lat", "lon"}
    full = gnss_tie.read_tenv3(p, full=True)
    assert set(full) - set(g) == {"date", "se", "sn", "su", "h"}
    assert full["date"][0] == date(2022, 1, 1) and full["date"][-1] == date(2022, 1, 10)
    assert np.allclose(full["su"], 5.0) and np.allclose(full["se"], 1.5)
    assert np.allclose(full["u"], g["u"]) and full["h"] == pytest.approx(60.0)
    assert gnss_tie.ngl_date("21AUG27") == date(2021, 8, 27)
    assert gnss_tie._ngl_decyear("21AUG27") == 2000 + (date(2021, 8, 27) - date(2000, 1, 1)).days / 365.25


# ----------------------------------------------------------------- steps
def test_equipment_step_removed_as_offset(cache, no_net):
    ds = days(date(2021, 10, 1), date(2024, 9, 30))
    t, e, n, u = series(ds)
    s = date(2023, 3, 15)
    after = np.array([d >= s for d in ds])
    e, u = e - 12.0 * after, u + 30.0 * after
    write_tenv3(cache / "STEP" / "STEP.tenv3", "STEP", ds, e, n, u)
    write_steps(cache, [("STEP", s, 1)])
    df = gt.load_gnss_series(["STEP"], ("2021-10-01", "2024-09-30"), cache, refresh=False)["STEP"]
    (st,) = df.attrs["steps_removed"]
    assert st["date"] == "2023-03-15" and st["events"] == ["Antenna_Type_Changed"]
    assert st["offset_mm"]["u"] == pytest.approx(30.0, abs=2.5)
    assert st["offset_mm"]["e"] == pytest.approx(-12.0, abs=1.0)
    r = gt.station_rate(df, None, "u", "linear")
    assert r["rate_mm_yr"] == pytest.approx(-50.0, abs=1.5)
    assert gt.station_rate(df, None, "e", "linear")["rate_mm_yr"] == pytest.approx(10.0, abs=0.5)


def test_step_offset_is_local_so_a_nonlinear_signal_does_not_leak_into_it(cache, no_net):
    """A logfile change (true offset 0) at the turn from drought to a wet
    year (rates -250 -> +80 -> -320 -> -20 mm/yr, turns over ~2 months, plus
    a 15 mm seasonal term): a step fitted with a whole-record model takes
    ~55 mm of real motion; the local line + step takes ~1 mm."""
    ds = days(date(2021, 10, 1), date(2024, 9, 30))
    rng = np.random.default_rng(3)
    t = yrs(ds, ds[0])
    sg = lambda z: 1 / (1 + np.exp(-z))  # noqa: E731
    rate = -250 + 330 * sg((t - 1.0) / 0.06) - 400 * sg((t - 1.7) / 0.06) + 300 * sg((t - 2.4) / 0.06)
    u = np.cumsum(rate) / 365.25 + 15 * np.sin(2 * np.pi * t) + rng.normal(0, 3.0, t.size)
    e, n = rng.normal(0, 1, t.size), rng.normal(0, 1, t.size)
    s = date(2022, 10, 1)
    write_tenv3(cache / "WETS" / "WETS.tenv3", "WETS", ds, e, n, u)
    write_steps(cache, [("WETS", s, 1, "Logfile_Changed")])
    df = gt.load_gnss_series("WETS", None, cache, refresh=False)["WETS"]
    (st,) = df.attrs["steps_removed"]
    assert st["events"] == ["Logfile_Changed"] and abs(st["offset_mm"]["u"]) < 4.0
    # the whole-record alternative is badly wrong here (what the local fit avoids)
    A = np.column_stack([np.ones_like(t), t, np.sin(2 * np.pi * t), np.cos(2 * np.pi * t),
                         np.array([d >= s for d in ds], dtype=float)])
    glob = np.linalg.lstsq(A, u, rcond=None)[0][-1]
    assert abs(glob) > 20.0


def test_earthquake_step_only_when_flagged(cache, no_net):
    ds = days(date(2022, 1, 1), date(2023, 12, 31))
    _, e, n, u = series(ds)
    s = date(2022, 12, 1)
    u = u - 20.0 * np.array([d >= s for d in ds])
    write_tenv3(cache / "QUAK" / "QUAK.tenv3", "QUAK", ds, e, n, u)
    write_steps(cache, [("QUAK", s, 2)])
    df = gt.load_gnss_series("QUAK", None, cache, refresh=False)["QUAK"]
    assert df.attrs["steps_removed"] == [] and df.attrs["earthquake_steps_removed"] is False
    (nr,) = df.attrs["steps_not_removed"]
    assert nr["date"] == "2022-12-01" and "earthquake" in nr["why"]
    jump = df.loc["2022-12-01":"2022-12-20", "u_mm"].mean() - df.loc["2022-11-11":"2022-11-30", "u_mm"].mean()
    assert jump < -15.0                                    # still in the series
    df2 = gt.load_gnss_series("QUAK", None, cache, refresh=False, earthquake_steps=True)["QUAK"]
    (st,) = df2.attrs["steps_removed"]
    assert st["offset_mm"]["u"] == pytest.approx(-20.0, abs=2.5)


def test_antenna_slip_days_dropped_and_real_height_change_is_a_step(cache, no_net):
    ds = days(date(2021, 1, 1), date(2024, 6, 30))
    _, e, n, u = series(ds, annual=0.0)
    ant = np.zeros(len(ds))
    slip = np.array([date(2021, 6, 1) <= d < date(2021, 7, 21) for d in ds])     # 50 days at 0.035 m: a slip
    ant[slip] = 0.035
    u = u + 5.0 * slip                                    # small: only the antenna rule can drop these days
    change = date(2022, 9, 1)
    after = np.array([d >= change for d in ds])
    ant[after] = 0.1                                      # held ~670 days: a real change
    u = u + 15.0 * after
    write_tenv3(cache / "ANTH" / "ANTH.tenv3", "ANTH", ds, e, n, u, ant=ant)
    write_steps(cache, [])
    df = gt.load_gnss_series("ANTH", None, cache, refresh=False)["ANTH"]
    assert df.attrs["antenna_slip_days_dropped"] == 50
    assert df.attrs["antenna_heights_dropped_m"] == [0.035]
    assert not df.loc["2021-06-01":"2021-07-20"].shape[0] and df.attrs["outliers_removed"] < 5
    (st,) = df.attrs["steps_removed"]
    assert st["date"] == "2022-09-01" and "antenna height changed in the series" in st["events"]
    assert st["offset_mm"]["u"] == pytest.approx(15.0, abs=2.5)


def test_step_without_days_on_one_side_is_not_estimated_and_said(cache, no_net):
    ds = days(date(2022, 1, 1), date(2022, 12, 31))
    _, e, n, u = series(ds)
    s = date(2022, 12, 30)                                 # two days after it
    write_tenv3(cache / "EDGE" / "EDGE.tenv3", "EDGE", ds, e, n, u)
    write_steps(cache, [("EDGE", s, 1)])
    df = gt.load_gnss_series("EDGE", None, cache, refresh=False)["EDGE"]
    assert df.attrs["steps_removed"] == []
    (nr,) = df.attrs["steps_not_removed"]
    assert "not estimated" in nr["why"] and nr["days_after"] == 2


# ----------------------------------------------------------------- screening
def test_outliers_beyond_5_sigma_removed_as_days_and_gaps_never_filled(cache, no_net):
    gap = (date(2022, 6, 1), date(2022, 6, 30))
    ds = days(date(2022, 1, 1), date(2023, 6, 30), skip=[gap])
    _, e, n, u = series(ds, noise=(1.0, 1.0, 3.0), seed=7)
    spikes = {date(2022, 3, 3): (0, 0, 60.0), date(2022, 10, 10): (25.0, 0, 0), date(2023, 2, 2): (0, -20.0, 0)}
    for d, (de, dn, du) in spikes.items():
        i = ds.index(d)
        e[i] += de; n[i] += dn; u[i] += du
    write_tenv3(cache / "SPIK" / "SPIK.tenv3", "SPIK", ds, e, n, u, sig=(0.5, 0.5, 1.0))
    write_steps(cache, [])
    df = gt.load_gnss_series("SPIK", None, cache, refresh=False)["SPIK"]
    assert sorted(df.attrs["outlier_days"]) == sorted(d.isoformat() for d in spikes)
    assert len(df) == len(ds) - 3 and df.attrs["gaps_filled"] is False
    assert df.loc["2022-06-01":"2022-06-30"].empty
    assert not df.isna().any().any()
    rs = df.attrs["robust_sigma_mm"]
    assert rs["u"] == pytest.approx(3.0, rel=0.2) and rs["e"] == pytest.approx(1.0, rel=0.2)
    assert np.allclose(df["su_mm"], rs["u"], atol=1e-3) and np.allclose(df["se_mm"], rs["e"], atol=1e-3)  # robust > formal
    assert list(df.columns) == list(gt.SERIES_COLUMNS) and df.index.name == "date"


def test_formal_sigma_wins_when_larger(cache, no_net):
    ds = days(date(2022, 1, 1), date(2023, 6, 30))
    _, e, n, u = series(ds, noise=(1.0, 1.0, 3.0))
    write_tenv3(cache / "FORM" / "FORM.tenv3", "FORM", ds, e, n, u, sig=(5.0, 5.0, 12.0))
    write_steps(cache, [])
    df = gt.load_gnss_series("FORM", None, cache, refresh=False)["FORM"]
    assert np.allclose(df["su_mm"], 12.0) and np.allclose(df["se_mm"], 5.0)


def test_window_datum_and_provenance(cache, no_net):
    ds = days(date(2021, 1, 1), date(2025, 7, 28))
    _, e, n, u = series(ds)
    write_tenv3(cache / "ENDS" / "ENDS.tenv3", "ENDS", ds, e, n, u, lat=36.114, lon=-119.568)
    write_steps(cache, [])
    df = gt.load_gnss_series("ENDS", ("2021-10-01", "2026-09-30"), cache, refresh=False)["ENDS"]
    a = df.attrs
    assert df.index[0] == pd.Timestamp("2021-10-01") and df.index[-1] == pd.Timestamp("2025-07-28")
    assert a["ends_before_window_end"] is True and a["last_epoch_in_file"] == "2025-07-28"
    assert a["window"] == ["2021-10-01", "2026-09-30"]
    assert abs(df["u_mm"].iloc[:10].mean()) < 5.0          # zero near the first day kept
    assert a["source_url"].endswith("/IGS20/ENDS.tenv3") and len(a["sha256"]) == 64
    assert a["lat"] == pytest.approx(36.114) and a["lon"] == pytest.approx(-119.568)
    assert a["steps_file"]["sha256"] and a["frame"].startswith("IGS20")


def test_eastern_longitude_normalised(cache, no_net):
    ds = days(date(2022, 1, 1), date(2022, 3, 1))
    _, e, n, u = series(ds)
    write_tenv3(cache / "CA19" / "CA19.tenv3", "CA19", ds, e, n, u, lat=-34.1788786, lon=-209.3087585)
    write_steps(cache, [])
    df = gt.load_gnss_series("CA19", None, cache, refresh=False)["CA19"]
    assert df.attrs["lon"] == pytest.approx(150.6912415)


def test_piecewise_model_with_water_year_hinges(cache, no_net):
    W = ("2021-10-01", "2026-09-30")
    assert gt.water_year_hinges(W) == [date(2022, 10, 1), date(2023, 10, 1), date(2024, 10, 1), date(2025, 10, 1)]
    ds = days(date(2021, 10, 1), date(2026, 9, 30))
    rng = np.random.default_rng(5)
    t = yrs(ds, ds[0])
    rates = [-290.0, 43.0, -64.0, -78.0, -60.0]           # per water year
    u = np.cumsum([rates[min(int(x), 4)] / 365.25 for x in t]) + 20 * np.sin(2 * np.pi * t) + rng.normal(0, 4, t.size)
    write_tenv3(cache / "CRCX" / "CRCX.tenv3", "CRCX", ds, rng.normal(0, 1, t.size), rng.normal(0, 1, t.size), u)
    write_steps(cache, [])
    lin = gt.load_gnss_series("CRCX", W, cache, refresh=False)["CRCX"]
    pw = gt.load_gnss_series("CRCX", W, cache, refresh=False, model="piecewise",
                             hinges=gt.water_year_hinges(W))["CRCX"]
    assert pw.attrs["robust_sigma_mm"]["u"] == pytest.approx(4.0, rel=0.25)
    assert lin.attrs["robust_sigma_mm"]["u"] > 2 * pw.attrs["robust_sigma_mm"]["u"]
    assert pw.attrs["model"]["hinges"] == ["2022-10-01", "2023-10-01", "2024-10-01", "2025-10-01"]
    with pytest.raises(ValueError, match="needs hinges"):
        gt.load_gnss_series("CRCX", W, cache, refresh=False, model="piecewise")
    with pytest.raises(ValueError, match="screening model"):
        gt.load_gnss_series("CRCX", W, cache, refresh=False, model="median")


def test_local_model_keeps_a_fast_event_at_the_edges_of_a_gap(cache, no_net):
    """CA19-like: -3 mm/day for 200 days with a 91-day outage inside it. A
    running median lags the motion at the gap edges and would flag real days."""
    gap = (date(2018, 7, 20), date(2018, 10, 17))
    ds = days(date(2018, 1, 1), date(2019, 6, 30), skip=[gap])
    rng = np.random.default_rng(11)
    d0, d1 = date(2018, 5, 12), date(2018, 11, 28)
    u = np.array([-3.0 * min(max((d - d0).days, 0), (d1 - d0).days) for d in ds]) + rng.normal(0, 3, len(ds))
    e = np.array([0.6 * min(max((d - d0).days, 0), (d1 - d0).days) for d in ds]) + rng.normal(0, 1, len(ds))
    n = rng.normal(0, 1, len(ds))
    write_tenv3(cache / "EVNT" / "EVNT.tenv3", "EVNT", ds, e, n, u)
    write_steps(cache, [])
    df = gt.load_gnss_series("EVNT", None, cache, refresh=False, model="local")["EVNT"]
    edge = {"2018-07-17", "2018-07-18", "2018-07-19", "2018-10-18", "2018-10-19", "2018-10-20"}
    assert not edge & set(df.attrs["outlier_days"])
    assert df.attrs["outliers_removed"] <= 2
    assert df.attrs["robust_sigma_mm"]["u"] == pytest.approx(3.0, rel=0.25)


# ----------------------------------------------------------------- downloads
class _Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_downloads_go_to_cache_404_is_reported_and_steps_are_required(cache, monkeypatch, tmp_path):
    ds = days(date(2022, 1, 1), date(2023, 1, 1))
    _, e, n, u = series(ds)
    src = write_tenv3(tmp_path / "src" / "GOOD.tenv3", "GOOD", ds, e, n, u)
    steps_src = write_steps(tmp_path / "src")
    calls = []

    def fake(url, timeout=None):
        calls.append(url)
        if url.endswith("GOOD.tenv3"):
            return _Resp(src.read_bytes())
        if url.endswith("steps.txt"):
            return _Resp(steps_src.read_bytes())
        raise urllib.error.HTTPError(url, 404, "Not Found", None, None)
    monkeypatch.setattr(gt.urllib.request, "urlopen", fake)
    out = gt.load_gnss_series(["good", "NONE"], None, cache)
    assert list(out) == ["GOOD"] and "HTTP 404" in out.missing["NONE"]
    assert (cache / "GOOD" / "GOOD.tenv3").exists() and (cache / gnss_tie.STEPS_FILE).exists()
    assert out["GOOD"].attrs["fetch_note"] == "downloaded"
    assert {p.name for p in cache.iterdir()} == {"GOOD", gnss_tie.STEPS_FILE}     # nothing else written
    with pytest.raises(ValueError, match="cache_dir is required"):
        gt.load_gnss_series(["GOOD"], None, None)
    with pytest.raises(ValueError, match="only NGL's IGS20"):
        gt.load_gnss_series(["GOOD"], None, cache, frame="IGS14")
    # no steps file and no download: refuse rather than load unscreened series
    (cache / gnss_tie.STEPS_FILE).unlink()
    with pytest.raises(RuntimeError, match="equipment changes cannot be screened"):
        gt.load_gnss_series(["GOOD"], None, cache, refresh=False)


def test_refresh_auto_updates_a_stale_series_that_ends_before_the_window(cache, monkeypatch, tmp_path):
    ds = days(date(2022, 1, 1), date(2023, 1, 1))
    _, e, n, u = series(ds)
    old = write_tenv3(cache / "LIVE" / "LIVE.tenv3", "LIVE", ds[:200], e[:200], n[:200], u[:200])
    write_steps(cache, [])
    new = write_tenv3(tmp_path / "src" / "LIVE.tenv3", "LIVE", ds, e, n, u)
    calls = []

    def fake(url, timeout=None):
        calls.append(url)
        return _Resp(new.read_bytes())
    monkeypatch.setattr(gt.urllib.request, "urlopen", fake)
    w = ("2022-01-01", "2022-12-31")
    df = gt.load_gnss_series("LIVE", w, cache)["LIVE"]               # fresh cache: used as is
    assert calls == [] and len(df) == 200
    ten_days = time.time() - 10 * 86400
    os.utime(old, (ten_days, ten_days))
    os.utime(cache / gnss_tie.STEPS_FILE, (ten_days, ten_days))
    df = gt.load_gnss_series("LIVE", w, cache)["LIVE"]
    assert any(c.endswith("LIVE.tenv3") for c in calls) and len(df) == len(ds) - 1
    assert df.attrs["fetch_note"] == "downloaded"
    # no window end: a stale copy is refreshed too; a fresh one is not
    series_calls = lambda: sum(c.endswith("LIVE.tenv3") for c in calls)  # noqa: E731
    n_calls = series_calls()
    gt.load_gnss_series("LIVE", None, cache)
    assert series_calls() == n_calls
    os.utime(old, (ten_days, ten_days))
    gt.load_gnss_series("LIVE", None, cache)
    assert series_calls() == n_calls + 1
    # offline with a stale cache: the cached copy is used and that is said
    os.utime(old, (ten_days, ten_days))
    write_tenv3(old, "LIVE", ds[:200], e[:200], n[:200], u[:200])
    os.utime(old, (ten_days, ten_days))

    def down(url, timeout=None):
        raise urllib.error.URLError("network unreachable")
    monkeypatch.setattr(gt.urllib.request, "urlopen", down)
    df = gt.load_gnss_series("LIVE", w, cache)["LIVE"]
    assert len(df) == 200 and df.attrs["fetch_note"].startswith("refresh failed")


# ----------------------------------------------------------------- rates, changes, matching
def test_station_rate_linear_midas_and_short_records(cache, no_net):
    ds = days(date(2020, 1, 1), date(2023, 12, 31))
    rng = np.random.default_rng(2)
    t = yrs(ds, ds[0])
    ar = np.zeros(t.size)
    for i in range(1, t.size):                             # coloured noise: AR(1), r1 = 0.8
        ar[i] = 0.8 * ar[i - 1] + rng.normal(0, 2.0)
    u = -12.0 * t + 3 * np.sin(2 * np.pi * t) + ar
    write_tenv3(cache / "RATE" / "RATE.tenv3", "RATE", ds, rng.normal(0, 1, t.size), rng.normal(0, 1, t.size), u)
    write_steps(cache, [])
    df = gt.load_gnss_series("RATE", None, cache, refresh=False)["RATE"]
    lin = gt.station_rate(df, None, "u", "linear")
    assert lin["rate_mm_yr"] == pytest.approx(-12.0, abs=1.0) and lin["comparison"] == "rate"
    assert set(lin) >= {"rate_mm_yr", "sigma_mm_yr", "n", "span_yr"}
    assert 0.7 < lin["r1"] < 0.9 and lin["n_eff"] < 0.2 * lin["n"]
    # the plain least-squares error, without the autocorrelation, is ~3x smaller
    A = gt._design(gt._decyear_ord([d.toordinal() for d in df.index.date]), np.array([]), True)
    r = df["u_mm"].to_numpy() - A @ np.linalg.lstsq(A, df["u_mm"].to_numpy(), rcond=None)[0]
    plain = math.sqrt(np.linalg.inv(A.T @ A)[1, 1] * (r @ r) / (len(r) - A.shape[1]))
    assert lin["sigma_mm_yr"] > 2.5 * plain
    mid = gt.station_rate(df, None, "u", "midas")
    assert mid["rate_mm_yr"] == pytest.approx(-12.0, abs=1.5) and mid["n_pairs"] > 500
    sub = gt.station_rate(df, ("2022-01-01", "2022-07-31"), "u", "linear")
    assert sub["rate_mm_yr"] is None and sub["comparison"] == "displacement" and "under 1 yr" in sub["reason"]
    m12 = gt.station_rate(df, ("2022-01-01", "2023-03-01"), "u", "midas")
    assert m12["rate_mm_yr"] is None and m12["comparison"] == "displacement"
    assert gt.station_rate(df, ("2022-01-01", "2022-07-31"), "u", "linear", min_span_yr=0.5)["rate_mm_yr"] is not None
    with pytest.raises(ValueError):
        gt.station_rate(df, None, "los")


def test_match_epochs_and_displacement_use_plus_minus_3_days_and_never_fill(cache, no_net):
    ds = days(date(2022, 1, 1), date(2022, 3, 31), skip=[(date(2022, 2, 1), date(2022, 2, 20))])
    u = np.array([float((d - ds[0]).days) for d in ds])     # 1 mm/day, no noise
    z = np.zeros(len(ds))
    write_tenv3(cache / "MTCH" / "MTCH.tenv3", "MTCH", ds, z, z, u, sig=(1.0, 1.0, 3.0))
    write_steps(cache, [])
    df = gt.load_gnss_series("MTCH", None, cache, refresh=False, outlier_sigma=1e9)["MTCH"]
    m = gt.match_epochs(df, ["2022-01-10", "2022-02-10", "2022-01-31"])
    assert m.loc["2022-01-10", "u_mm"] == pytest.approx(df.loc["2022-01-10", "u_mm"], abs=1e-6)
    assert m.loc["2022-01-10", "n_days"] == 7
    assert np.isnan(m.loc["2022-02-10", "u_mm"]) and m.loc["2022-02-10", "n_days"] == 0
    assert m.loc["2022-01-31", "n_days"] == 4               # 28..31 Jan; the gap starts 1 Feb
    d = gt.station_displacement(df, "2022-01-10", "2022-03-10")
    assert d["change_mm"] == pytest.approx(59.0, abs=1e-3)
    assert d["sigma_mm"] == pytest.approx(math.hypot(3.0, 3.0), rel=1e-3)
    miss = gt.station_displacement(df, "2022-01-10", "2022-02-10")
    assert miss["change_mm"] is None and "2022-02-10" in miss["reason"]


# ----------------------------------------------------------------- user CSV truth
def test_points_csv_wide_and_long_with_roles(tmp_path):
    wide = tmp_path / "tw.csv"
    wide.write_text(
        "id,lat,lon,date,up_mm,east_mm,sigma_mm,source_report,page,role\n"
        "TW01,-34.20,150.60,2019-11-15,0,0,5,Report 1,12,\n"
        "TW01,-34.20,150.60,2020-05-15,-120,,5,Report 3,14,\n"
        "TW02,-34.21,150.61,20191115,0,1.5,6,Report 1,12,null\n"
        "TW02,-34.21,150.61,20200515,,2.5,6,Report 3,14,null\n")
    pts = gt.load_points_csv(wide, roles={"TW01": "validation"})
    (p1, d1), (p2, d2) = pts
    assert p1 == {"id": "TW01", "kind": "survey", "lat": -34.2, "lon": 150.6, "h_m": None,
                  "source": "Report 1, page 12", "role": "validation"}
    assert p2["role"] == "null"
    assert list(d1.columns) == ["time", "value_mm", "sigma_mm", "component"]
    assert d1[d1.component == "u"]["value_mm"].tolist() == [0.0, -120.0]
    assert d1[d1.component == "e"]["value_mm"].tolist() == [0.0]          # empty cell dropped, not filled
    assert d2[d2.component == "u"]["time"].tolist() == [pd.Timestamp("2019-11-15")]
    long = tmp_path / "long.csv"
    long.write_text("id,lat,lon,date,component,value_mm,sigma_mm,kind,h_m\n"
                    "BM1,36.1,-119.6,2022-01-01,Up,0,2,survey,61.5\n"
                    "BM1,36.1,-119.6,2023-01-01,vertical,-150,2,survey,61.5\n"
                    "BM1,36.1,-119.6,2023-01-01,los,-120,3,survey,61.5\n")
    ((p, d),) = gt.load_points_csv(long, default_role="validation")
    assert p["h_m"] == 61.5 and p["kind"] == "survey"
    assert sorted(set(d.component)) == ["los", "u"] and len(d) == 3
    bad = tmp_path / "bad.csv"
    bad.write_text("id,lat,lon,date,up_mm\nX,1,2,2022-01-01,5\n")
    with pytest.raises(ValueError, match="sigma"):
        gt.load_points_csv(bad)
    bad.write_text("id,lat,lon,date,up_mm,sigma_mm\nX,1,2,2022-01-01,5,1\nX,1.5,2,2022-02-01,6,1\n")
    with pytest.raises(ValueError, match="more than one position"):
        gt.load_points_csv(bad)
    bad.write_text("id,lat,lon,date,component,value_mm,sigma_mm\nX,1,2,2022-01-01,z,5,1\n")
    with pytest.raises(ValueError, match="unknown component"):
        gt.load_points_csv(bad)
    bad.write_text("id,lat,lon,date,up_mm,sigma_mm,role\nX,1,2,2022-01-01,5,1,anchor\n")
    with pytest.raises(ValueError, match="role"):
        gt.load_points_csv(bad)


# ----------------------------------------------------------------- GA CEMP workbook
def write_ga(path: Path, sheets: dict) -> Path:
    import openpyxl
    wb = openpyxl.Workbook()
    wb.remove(wb.active)
    head = ["Date [yyyymmdd]", "East Displacement [mm]", "North Displacement [mm]", "Up Displacement[mm]",
            "East Displacement Uncertainty [mm]", "North Displacement Uncertainty [mm]",
            "Up Displacement Uncertainty [mm]"]
    for name, rows in sheets.items():
        ws = wb.create_sheet(name)
        ws.append(head)
        for r in rows:
            ws.append(list(r))
    path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(path)
    return path


def test_ga_xlsx_stated_sigma_if_larger_local_screen_coords_and_frame(tmp_path):
    rng = np.random.default_rng(4)
    ds = days(date(2018, 1, 1), date(2019, 6, 30), skip=[(date(2018, 7, 20), date(2018, 10, 17))])
    d0, d1 = date(2018, 5, 12), date(2019, 4, 3)
    up = np.array([-326.0 * min(max((d - d0).days, 0), (d1 - d0).days) / (d1 - d0).days for d in ds])
    up += rng.normal(0, 3, len(ds))
    spike = ds.index(date(2019, 5, 29))
    rows19 = [(int(d.strftime("%Y%m%d")), round(rng.normal(0, 1), 2), round(rng.normal(0, 1), 2),
               round(up[i] + (60 if i == spike else 0), 2), 8.6, 8.6, 17.2) for i, d in enumerate(ds)]
    rows19[0] = (rows19[0][0], 0, 0, 0, 0, 0, 0)          # GA's datum epoch
    rows07 = [(int(d.strftime("%Y%m%d")), round(rng.normal(0, 1), 2), round(rng.normal(0, 1), 2),
               round(rng.normal(0, 3), 2), 0.2, 0.2, 0.5) for d in ds]
    p = write_ga(tmp_path / "ga.xlsx", {"CA19": rows19, "CA07": rows07})
    out = gt.load_ga_xlsx(p, ["CA19", "CA07", "CA99"], coords={"CA19": (-34.178878, 150.691241)})
    assert set(out) == {"CA19", "CA07"} and "no sheet CA99" in out.missing["CA99"]
    c19, c07 = out["CA19"], out["CA07"]
    assert c19.attrs["outlier_days"] == ["2019-05-29"]
    assert np.allclose(c19["su_mm"].iloc[1:], 17.2)                        # stated > robust
    assert c07["su_mm"].iloc[0] == pytest.approx(c07.attrs["robust_sigma_mm"]["u"], abs=1e-3)   # robust > stated 0.5
    assert c07["su_mm"].iloc[0] > 2.0
    assert c19.attrs["lat"] == -34.178878 and c07.attrs["lat"] is None
    assert "not IGS20" in c19.attrs["frame"] and c19.attrs["steps_removed"] == []
    ch = gt.station_displacement(c19, "2018-05-12", "2019-04-03")
    assert ch["change_mm"] == pytest.approx(-326.0, abs=6.0)
    w = gt.load_ga_xlsx(p, "CA19", window=("2018-05-01", "2018-12-31"))["CA19"]
    assert w.index[0] >= pd.Timestamp("2018-05-01") and w.index[-1] <= pd.Timestamp("2018-12-31")
    assert w.loc["2018-07-20":"2018-10-17"].empty
    with pytest.raises(ValueError, match="no coordinates"):
        gt.point_from_series("CA07", c07)
    pt = gt.point_from_series("CA19", c19, role="validation")
    assert pt["kind"] == "gnss" and pt["lon"] == 150.691241


# ----------------------------------------------------------------- inventory
HOLD_HEAD = "Sta  Lat(deg)   Long(deg) Hgt(m)  X(m)           Y(m)         Z(m)          Dtbeg      Dtend      Dtmod      NumSol StaOrigName"


def test_stations_in_box_live_cached_and_frozen_fallback(tmp_path, monkeypatch):
    cat = tmp_path / "cat"
    cat.mkdir()
    lines = [HOLD_HEAD,
             "CAHA  36.3234   240.3772   46.902 0 0 0 2016-01-12 2026-09-20 2026-10-03   3193",
             "CRCN  36.1140   240.4320   50.000 0 0 0 2010-09-27 2025-07-28 2026-10-03   5000",
             "HANF  36.2152   240.3409   50.000 0 0 0 2025-03-06 2026-09-12 2026-10-03    500",
             "FARX  37.0000   240.3000   50.000 0 0 0 2010-01-01 2026-09-12 2026-10-03   5000",
             "CA19 -34.1789   150.6912  182.785 0 0 0 2016-07-04 2020-09-29 2026-10-03    714"]
    lines += [f"Z{i:03d}  10.0  10.0  0.0 0 0 0 2010-01-01 2020-01-01 2026-10-03 100" for i in range(40)]
    (cat / gt.HOLDINGS_FILE).write_text("\n".join(lines) + "\n")
    calls = []
    monkeypatch.setattr(gt.urllib.request, "urlopen", lambda *a, **k: calls.append(a) or (_ for _ in ()).throw(
        urllib.error.URLError("offline")))
    box = (-119.82, 36.08, -119.52, 36.35)
    pts = gt.stations_in_box(box, cat, 2.0, window=("2021-10-01", "2026-09-30"), roles={"CAHA": "calibration"})
    assert calls == []                                     # fresh cache: no download
    assert [p["id"] for p in pts] == ["CAHA", "CRCN"]      # HANF 1.5 yr in the window; FARX outside the box
    caha = pts[0]
    assert caha["lon"] == pytest.approx(-119.6228) and caha["role"] == "calibration" and pts[1]["role"] == "validation"
    assert caha["years_in_window"] == pytest.approx(4.97, abs=0.01) and caha["inventory"].startswith("NGL DataHoldings")
    assert [p["id"] for p in gt.stations_in_box(box, cat, 0.0)] == ["CAHA", "CRCN", "HANF"]
    assert [p["id"] for p in gt.stations_in_box((150.55, -34.30, 150.78, -34.14), cat)] == ["CA19"]
    assert [p["id"] for p in gt.stations_in_box((170.0, -40.0, -170.0, 40.0), cat)] == []   # across 180 deg
    # a stale cache: refresh tried, failure said, cached copy used
    old = time.time() - 3 * 86400
    os.utime(cat / gt.HOLDINGS_FILE, (old, old))
    pts = gt.stations_in_box(box, cat, 2.0, window=("2021-10-01", "2026-09-30"))
    assert len(calls) == 1 and "refresh failed" in pts[0]["inventory"]
    # no inventory at all: the pipeline's frozen station list, said so
    (cat / gt.HOLDINGS_FILE).unlink()
    (cat / "unr_llh.out").write_text("CAHA 36.3234 -119.6228 46.9\nFARX 37.0 -119.7 50.0\n")
    (cat / "unr_midas_igs14.txt").write_text(
        "CAHA MIDAS5 2016.0301 2024.7400 8.7 3000 2990 2500 0 0 -0.1597 0 0 0.001 0 0 0\n")
    pts = gt.stations_in_box(box, cat, 0.0, refresh=False)
    assert [p["id"] for p in pts] == ["CAHA"] and "FROZEN" in pts[0]["inventory"] and pts[0]["end"] == "2024-09-28"
    for f in ("unr_llh.out", "unr_midas_igs14.txt"):
        (cat / f).unlink()
    with pytest.raises(FileNotFoundError):
        gt.stations_in_box(box, cat, 0.0, refresh=False)


# ----------------------------------------------------------------- protocol roles
def test_protocol_roles_give_the_anchor_validation_split():
    a = {p["id"]: p for p in gt.protocol_points(REVIEW / "protocol_A.yaml")}
    assert a["CA19"]["role"] == "validation" and a["CA07"]["role"] == "reference"
    assert {a[s]["role"] for s in ("PCTN", "MENA", "CRDX")} == {"null"}
    assert a["CA10"]["role"] is None and a["CMDN"]["role"] is None
    assert a["CA19"]["lat"] == -34.178878 and a["CA19"]["kind"] == "gnss"
    split = gt.split_roles(a.values())
    assert split["reference"] == ["CA07"] and split["validation"] == ["CA19"]
    b1 = {p["id"]: p for p in gt.protocol_points(REVIEW / "protocol_B1.yaml")}
    assert {b1[s]["role"] for s in ("CRCN", "LEMA", "CAHA", "KDL1", "QCAP", "KDL2", "HANF")} == {"calibration"}
    assert "calibration_pool" in b1["CAHA"]["protocol_roles"] and "rate" not in b1["HANF"]["protocol_roles"]
    assert b1["CAMD"]["protocol_roles"] == ["nisar"] and b1["CAMD"]["sub_area"] == "el_nido"
    assert "CHOW" not in b1
    b2 = {p["id"]: p for p in gt.protocol_points(REVIEW / "protocol_B2.yaml")}
    assert b2["UHKS"]["role"] is None and b2["TXKY"]["role"] is None and b2["MRHK"]["role"] == "calibration"
    with pytest.raises(ValueError, match="two roles"):
        gt.split_roles([gt.make_point("X", "gnss", 0, 0, role="reference"),
                        gt.make_point("X", "gnss", 0, 0, role="validation")])
    with pytest.raises(ValueError, match="kind"):
        gt.make_point("X", "levelling", 0, 0)
    with pytest.raises(ValueError, match="role"):
        gt.make_point("X", "gnss", 0, 0, role="anchor")


# ----------------------------------------------------------------- independent review, 8 Oct 2026
def test_hinges_apply_only_to_piecewise_in_a_mixed_model_dict(cache, no_net):
    """{id: model} with hinges: the 'linear' station must stay linear (it took
    the water-year hinges, and its attrs said 'linear' with hinges)."""
    W = ("2021-10-01", "2026-09-30")
    H = gt.water_year_hinges(W)
    ds = days(date(2021, 10, 1), date(2026, 9, 30))
    rng = np.random.default_rng(5)
    t = yrs(ds, ds[0])
    rates = [-290.0, 43.0, -64.0, -78.0, -60.0]
    u = np.cumsum([rates[min(int(x), 4)] / 365.25 for x in t]) + rng.normal(0, 4, t.size)
    for sid in ("PWAA", "LINB"):
        write_tenv3(cache / sid / f"{sid}.tenv3", sid, ds, rng.normal(0, 1, t.size), rng.normal(0, 1, t.size), u)
    write_steps(cache, [])
    out = gt.load_gnss_series(["PWAA", "LINB"], W, cache, refresh=False,
                              model={"PWAA": "piecewise", "LINB": "linear"}, hinges=H)
    pw, lin = out["PWAA"], out["LINB"]
    assert lin.attrs["model"] == {"name": "linear", "hinges": [], "local_half_days": None}
    assert pw.attrs["model"]["hinges"] == [h.isoformat() for h in H]
    assert lin.attrs["robust_sigma_mm"]["u"] > 2 * pw.attrs["robust_sigma_mm"]["u"]


def test_step_just_inside_the_window_is_estimated_from_the_whole_record(cache, no_net):
    """P307: antenna change 2025-11-12, 11 days into the NISAR window. A step 2
    days into a window was left in the series ('not estimated'); the days
    before the window estimate it."""
    ds = days(date(2024, 6, 1), date(2026, 9, 30))
    _, e, n, u = series(ds, rate_u=-5.0, annual=3.0)
    s = date(2025, 11, 3)
    after = np.array([d >= s for d in ds])
    u = u + 25.0 * after
    write_tenv3(cache / "P3X7" / "P3X7.tenv3", "P3X7", ds, e, n, u)
    write_steps(cache, [("P3X7", s, 1, "Antenna_Type_Changed")])
    df = gt.load_gnss_series("P3X7", ("2025-11-01", "2026-09-30"), cache, refresh=False)["P3X7"]
    assert df.attrs["steps_not_removed"] == []
    (st,) = df.attrs["steps_removed"]
    assert st["date"] == "2025-11-03" and st["days_before"] >= 10
    assert st["offset_mm"]["u"] == pytest.approx(25.0, abs=3.0)
    jump = df.loc["2025-11-03":"2025-11-10", "u_mm"].mean() - df.loc["2025-11-01":"2025-11-02", "u_mm"].mean()
    assert abs(jump) < 8.0


def test_a_step_left_in_the_series_bounds_offset_fits_and_midas_pairs(cache, no_net):
    """An earthquake offset not removed (earthquake_steps=False) 30 days before
    an equipment change: the equipment offset is not fitted across it, and
    MIDAS pairs no day across it."""
    ds = days(date(2020, 1, 1), date(2023, 12, 31))
    _, e, n, u = series(ds, rate_u=-5.0, annual=0.0, noise=(1.0, 1.0, 2.0), seed=9)
    quake, equip = date(2022, 1, 10), date(2022, 2, 9)
    u = u - 40.0 * np.array([d >= quake for d in ds]) + 15.0 * np.array([d >= equip for d in ds])
    write_tenv3(cache / "EQEQ" / "EQEQ.tenv3", "EQEQ", ds, e, n, u)
    write_steps(cache, [("EQEQ", quake, 2), ("EQEQ", equip, 1)])
    df = gt.load_gnss_series("EQEQ", None, cache, refresh=False)["EQEQ"]
    (st,) = df.attrs["steps_removed"]
    assert st["date"] == "2022-02-09" and st["offset_mm"]["u"] == pytest.approx(15.0, abs=2.5)
    assert [r["date"] for r in df.attrs["steps_not_removed"]] == ["2022-01-10"]
    mid = gt.station_rate(df, None, "u", "midas")
    assert mid["rate_mm_yr"] == pytest.approx(-5.0, abs=1.5)
    # 2.5 yr with the earthquake 1.25 yr in: two thirds of the one-year pairs cross it, so
    # MIDAS's trimming cannot hide them (-45 mm/yr if they are kept)
    ds = days(date(2020, 1, 1), date(2022, 6, 30))
    _, e, n, u = series(ds, rate_u=-5.0, annual=0.0, noise=(1.0, 1.0, 2.0), seed=10)
    quake = date(2021, 4, 1)
    u = u - 40.0 * np.array([d >= quake for d in ds])
    write_tenv3(cache / "EQMI" / "EQMI.tenv3", "EQMI", ds, e, n, u)
    write_steps(cache, [("EQMI", quake, 2)])
    df = gt.load_gnss_series("EQMI", None, cache, refresh=False)["EQMI"]
    assert df.attrs["steps_removed"] == [] and len(df.attrs["steps_not_removed"]) == 1
    mid = gt.station_rate(df, None, "u", "midas")
    assert mid["rate_mm_yr"] == pytest.approx(-5.0, abs=1.5) and mid["n_pairs"] >= 30


def test_points_csv_refuses_guessed_dates_missing_date_or_position_and_bad_sigma(tmp_path):
    p = tmp_path / "t.csv"
    p.write_text("id,lat,lon,date,up_mm,sigma_mm\nTW01,-34.2,150.6,05/02/2022,-10,5\n")   # 5 Feb in Australia
    with pytest.raises(ValueError, match="YYYY-MM-DD"):
        gt.load_points_csv(p)
    p.write_text("id,lat,lon,date,up_mm,sigma_mm\nTW01,-34.2,150.6,,-10,5\nTW01,-34.2,150.6,2022-02-05,-12,5\n")
    with pytest.raises(ValueError, match="without a date"):
        gt.load_points_csv(p)
    p.write_text("id,lat,lon,date,up_mm,sigma_mm\nTW01,,150.6,2022-02-05,-10,5\n")
    with pytest.raises(ValueError, match="no lat"):
        gt.load_points_csv(p)
    p.write_text("id,lat,lon,date,up_mm,sigma_mm\nTW01,-34.2,150.6,2022-02-05,-10,0\n")
    with pytest.raises(ValueError, match="above zero"):
        gt.load_points_csv(p)
    p.write_text("id,lat,lon,date,up_mm,sigma_mm\nTW01,-34.2,150.6,20220205,-10,5\nTW01,-34.2,150.6,2022-03-05,-11,5\n")
    ((pt, d),) = gt.load_points_csv(p)
    assert d["time"].tolist() == [pd.Timestamp("2022-02-05"), pd.Timestamp("2022-03-05")]
