"""7 Oct 2026 trust review, group gnss (gnss_tie.py): decisions E15 and E16.
Each test here fails on the code in backups/pre_trust_20261007/ unless its
docstring says otherwise.

E15  GNSS rates as MIDAS (Blewitt et al. 2016): pairs closest to one year,
     2-sigma trimming, sigma x 3; NGL steps-file events and antenna-height
     changes screened before estimating; a robust background (neighbour
     outlier test before the weighted mean); radius_sensitivity only from
     stations actually downloaded, else "not evaluated".
E16  the check at a station inside the area compares the same dates on
     both sides (radar over the overlap, station minus the regional
     stations over the overlap); too coarse a tolerance reads "not
     testable", never "agree".
"""
import json
import math
import types
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

import test_fix_L1 as L1T                                        # the synthetic tie project


@pytest.fixture
def project(tmp_path, monkeypatch):
    return L1T.project.__wrapped__(tmp_path, monkeypatch)


def _daily(t0=2016.0, years=6.0):
    """Daily epochs in NGL's convention (days since 2000-01-01 / 365.25)."""
    k0 = round((t0 - 2000.0) * 365.25)
    return 2000.0 + (k0 + np.arange(int(years * 365.25))) / 365.25


# ---------------------------------------------------------------- E15 the estimator
def test_pairs_are_one_year_apart_so_a_seasonal_sawtooth_cancels():
    """A slow seasonal rise and a quick fall (groundwater-like sawtooth, 20 mm
    a year): one-year pairs see none of it. The old pairing (the first day at
    least 0.97 yr later, median gap 0.972 yr) read the rise over the missing
    0.028 yr as trend: -1.58 for a true -1.00 mm/yr."""
    from insar_pipeline import gnss_tie as gt
    t = _daily()
    x = -1.0 * (t - t[0]) + 20.0 * np.mod(t, 1.0)
    m = gt.midas_rate(t, x)
    assert m["rate_mm_yr"] == pytest.approx(-1.0, abs=0.05)


def test_sigma_carries_midas_factor_of_three():
    """Blewitt et al. 2016: sigma = 3 x 1.2533 x sd / sqrt(n/4). The old
    sigma had no factor 3 (2-2.6 x smaller than NGL's on the same spans)."""
    from insar_pipeline import gnss_tie as gt
    t = _daily()
    x = -1.0 * (t - t[0]) + np.random.default_rng(3).normal(0, 2.0, t.size)
    m = gt.midas_rate(t, x)
    i = np.arange(t.size)
    j = np.clip(np.searchsorted(t, t + 1.0) - 1, 0, t.size - 1)     # the day closest to one year later
    ok = np.abs(t[j] - t - 1.0) <= 0.01
    s = (x[j[ok]] - x[i[ok]]) / (t[j[ok]] - t[i[ok]])
    sd = 1.4826 * np.median(np.abs(s - np.median(s)))
    plain = 1.2533 * sd / math.sqrt(s.size / 4)             # MIDAS without the factor 3 or the trim
    assert 2.5 < m["sigma_mm_yr"] / plain < 3.1


def test_slopes_beyond_two_sigma_are_trimmed():
    """120 days sitting 40 mm high (a metadata slip without its antenna
    column): the pairs from them are trimmed, so the rate moves by 0.03 mm/yr
    instead of the 0.24 the untrimmed median moved."""
    from insar_pipeline import gnss_tie as gt
    t = _daily()
    x = -1.0 * (t - t[0]) + np.random.default_rng(3).normal(0, 2.0, t.size)
    bad = (t >= 2016.3) & (t < 2016.3 + 120 / 365.25)
    clean, dirty = gt.midas_rate(t, x), gt.midas_rate(t, x + np.where(bad, 40.0, 0.0))
    assert abs(dirty["rate_mm_yr"] - clean["rate_mm_yr"]) < 0.08
    assert dirty["n_trimmed"] >= 120


def test_no_pair_straddles_a_step():
    """A -6 mm step (INVI's 2024 antenna change put one into every pair across
    it): with the step epoch screened the rate is the step-free one."""
    from insar_pipeline import gnss_tie as gt
    t = _daily()
    x = -1.0 * (t - t[0]) + np.random.default_rng(3).normal(0, 2.0, t.size)
    clean = gt.midas_rate(t, x)["rate_mm_yr"]
    stepped = x + np.where(t >= 2019.0, -6.0, 0.0)
    assert gt.midas_rate(t, stepped)["rate_mm_yr"] < clean - 0.5          # unscreened: biased
    assert gt.midas_rate(t, stepped, steps=(2019.0,))["rate_mm_yr"] == pytest.approx(clean, abs=0.05)


def test_ngl_steps_file_is_read_in_the_series_own_time_convention(tmp_path):
    """INEV's tenv3 dates 21APR19 2021.2977; the steps file's 21AUG27 lands
    on its own day's sample. Code 1 and 2 lines are read, others skipped."""
    from insar_pipeline import gnss_tie as gt
    assert gt._ngl_decyear("21APR19") == pytest.approx(2021.2977, abs=1e-4)
    p = tmp_path / gt.STEPS_FILE
    p.write_text("INEV  21AUG27  1  Elevation_Cutoff_Changed\n"
                 "INVI  24JUN06  1  Antenna_and_Radome_Type_Changed\n"
                 "XXXX  19JUL06  2  0.2156  110.15  6.4  ci38443183\n"
                 "garbage line\n")
    st = gt.read_steps(p)
    assert [e["t"] for e in st["INEV"]] == [pytest.approx(2021.6537, abs=1e-4)]
    assert st["INVI"][0]["event"] == "Antenna_and_Radome_Type_Changed"
    assert st["XXXX"][0]["event"] == "possible earthquake offset"


def test_antenna_height_slip_days_are_dropped_and_a_real_change_is_a_step():
    """INEV: 0 m instead of 0.035 m on 192 scattered days, +35 mm high - a
    metadata slip, dropped. A height held for years is a real change: kept,
    with no pair across it."""
    from insar_pipeline import gnss_tie as gt
    t = _daily(years=8.0)
    rng = np.random.default_rng(4)
    x = -1.0 * (t - t[0]) + rng.normal(0, 2.0, t.size)
    ant = np.full(t.size, 0.035)
    slip = np.zeros(t.size, bool)
    slip[rng.choice(t.size, 120, replace=False)] = True
    slip[(t > 2021.3) & (t < 2021.5)] = True                 # and a block
    ant[slip] = 0.0
    sc = gt.screen_series({"t": t, "u": x + np.where(slip, 35.0, 0.0), "ant": ant})
    assert sc["antenna_epochs_dropped"] == int(slip.sum()) and sc["antenna_heights_dropped_m"] == [0.0]
    assert sc["steps"] == []
    assert gt.midas_rate(sc["t"], sc["u"])["rate_mm_yr"] == pytest.approx(
        gt.midas_rate(t[~slip], x[~slip])["rate_mm_yr"], abs=1e-9)
    # a real antenna change (new height for the last four years, positions continuous)
    ant2 = np.where(t >= 2020.0, 0.1, 0.035)
    sc2 = gt.screen_series({"t": t, "u": x, "ant": ant2}, steps=(2018.5,))
    assert sc2["antenna_epochs_dropped"] == 0
    assert sc2["steps"] == [2018.5, pytest.approx(float(t[t >= 2020.0][0]))]


def test_fetching_a_series_also_fetches_the_steps_file(tmp_path, monkeypatch):
    """The steps file travels with the series (a newer series may carry a
    step an older file does not name); a failed steps download is not fatal."""
    import urllib.request
    from insar_pipeline import gnss_tie as gt
    got = []

    class R:
        def __init__(self, data): self.data = data
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return self.data

    def urlopen(url, timeout=None):
        got.append(url)
        if url == gt.NGL_STEPS:
            if fail:
                raise OSError("offline")
            return R(b"INVI  24JUN06  1  Antenna_and_Radome_Type_Changed\n" * 40)
        return R(b"x" * 2000)

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    fail = True
    gt.fetch_tenv3("AAAA", tmp_path)                         # steps fail: logged, the series still comes
    assert (tmp_path / "AAAA" / "AAAA.tenv3").exists() and not (tmp_path / gt.STEPS_FILE).exists()
    fail = False
    gt.fetch_tenv3("AAAA", tmp_path)                         # cached series, missing steps file: fetched
    assert (tmp_path / gt.STEPS_FILE).exists() and got.count(gt.NGL_STEPS) == 2
    gt.fetch_tenv3("AAAA", tmp_path)                         # both cached and fresh: nothing fetched
    assert len(got) == 3


def test_a_steps_file_that_names_no_step_is_not_screening(tmp_path, monkeypatch):
    """An error page saved as the steps file parsed to no events and the tie
    recorded the series as screened. Now it reads 'not screened', and a
    download that is not NGL's list never replaces the cached one."""
    import urllib.request
    from insar_pipeline import gnss_tie as gt
    p = tmp_path / gt.STEPS_FILE
    p.write_text("<html>" + "x" * 2000 + "</html>\n")
    steps, rec = gt._load_steps(tmp_path, None)
    assert steps == {} and rec["screened"] is False and "not screened" in rec["note"]
    good = "INVI  24JUN06  1  Antenna_and_Radome_Type_Changed\n" * 40
    p.write_text(good)
    assert gt._load_steps(tmp_path, None)[1]["screened"] is True

    class R:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return b"<html>" + b"x" * 2000 + b"</html>"
    monkeypatch.setattr(urllib.request, "urlopen", lambda url, timeout=None: R())
    with pytest.raises(ValueError):
        gt.fetch_steps(tmp_path)
    assert p.read_text() == good and not p.with_suffix(".tmp").exists()


# ---------------------------------------------------------------- E15 the robust background
def _site(name, lat, lon, rate, local=0.0, seed=0, common=None, years=10.0):
    t = _daily(2016.0, years)
    rng = np.random.default_rng(seed)
    u = rate * (t - t[0]) + local * (t - t[0]) + rng.normal(0, 1.0, t.size)
    if common is not None:
        u = u + common[: t.size]
    return ({"site": name, "lat": lat, "lon": lon, "rate_mm_yr": gt_rate(t, u), "sigma_mm_yr": 0.9},
            {"t": t, "u": u, "steps": []})


def gt_rate(t, u):
    from insar_pipeline import gnss_tie as gt
    return gt.midas_rate(t, u)["rate_mm_yr"]


def test_a_local_mover_next_to_a_neighbour_is_left_out():
    """VINU's case made clear: two stations 7 km apart share the regional
    field; one has a local -3 mm/yr on top. Their differenced series shows it
    far beyond its sigma, so the one farther from the third site leaves the
    background."""
    from insar_pipeline import gnss_tie as gt
    common = np.random.default_rng(9).normal(0, 3.0, 4000)    # regional common-mode noise
    a, sa = _site("NEAR", 38.70, -87.50, -1.0, seed=1, common=common)
    b, sb = _site("MOVE", 38.76, -87.50, -1.0, local=-3.0, seed=2, common=common)
    c, sc = _site("FARS", 38.20, -87.55, -1.2, seed=3)
    kept, rec = gt.outlier_test([a, b, c], {"NEAR": sa, "MOVE": sb, "FARS": sc}, 2016.0, 2026.0)
    assert [u["site"] for u in kept] == ["NEAR", "FARS"] and rec["left_out"] == ["MOVE"]
    p = rec["pairs"][0]
    assert p["tested"] and p["disagree"] and p["left_out"] == "MOVE" and rec["z"] == pytest.approx(1.96, abs=0.01)


def test_a_difference_within_its_uncertainty_keeps_both_and_two_sites_alone_keep_both():
    from insar_pipeline import gnss_tie as gt
    common = np.random.default_rng(9).normal(0, 3.0, 4000)
    a, sa = _site("NEAR", 38.70, -87.50, -1.0, seed=1, common=common)
    b, sb = _site("SAME", 38.76, -87.50, -1.0, local=-0.05, seed=2, common=common)
    c, sc = _site("FARS", 38.20, -87.55, -1.2, seed=3)
    kept, rec = gt.outlier_test([a, b, c], {"NEAR": sa, "SAME": sb, "FARS": sc}, 2016.0, 2026.0)
    assert len(kept) == 3 and rec["left_out"] == [] and rec["pairs"][0]["disagree"] is False
    # a real disagreement but no third site to say which one moves: both stay
    m, sm = _site("MOVE", 38.76, -87.50, -1.0, local=-3.0, seed=2, common=common)
    kept, rec = gt.outlier_test([a, m], {"NEAR": sa, "MOVE": sm}, 2016.0, 2026.0)
    assert len(kept) == 2 and rec["pairs"][0]["disagree"] and rec["pairs"][0]["left_out"] is None
    # sites farther apart than NEIGHBOUR_KM are not tested against each other
    kept, rec = gt.outlier_test([a, c], {"NEAR": sa, "FARS": sc}, 2016.0, 2026.0)
    assert rec["pairs"] == [] and rec["z"] is None and len(kept) == 2


def test_tie_leaves_a_local_mover_out_of_the_background(project, monkeypatch):
    """End to end on the synthetic project: NNNN, 4.4 km from AAAA, sinks
    3 mm/yr faster; GGGG (20 km) arbitrates. The background is AAAA+GGGG."""
    from insar_pipeline import gnss_tie as gt
    monkeypatch.setitem(L1T.SITES, "GGGG", (39.18, -87.30, 2010.5, 2026.6, -1.4))
    monkeypatch.setitem(L1T.SITES, "NNNN", (39.14, -87.30, 2010.5, 2026.6, -4.0))
    monkeypatch.setattr(gt, "discover", lambda cfg, root, radius=None: ["AAAA", "GGGG", "NNNN"])
    out = gt.run(project.cfg)
    assert out["stations_used"] == ["AAAA", "GGGG"] and out["outlier_test"]["left_out"] == ["NNNN"]
    nn = next(s for s in out["stations"] if s["site"] == "NNNN")
    assert nn["used_for_background"] is False and "neighbour" in nn["left_out"]
    assert out["background"]["left_out"] == ["NNNN"]
    assert out["background"]["vertical_mm_yr"] == pytest.approx(-1.2, abs=0.25)
    assert out["steps"]["screened"] is False and "not available" in out["steps"]["note"]


def test_radius_sensitivity_is_not_evaluated_where_a_qualifying_station_was_not_downloaded(project):
    """Site B's 50 km read -1.85 'unchanged' while INDA and INJS (44-45 km,
    8.7 yr of overlap) were never downloaded. Here CCCC (41.7 km, long record)
    is in the catalogue but not downloaded: 50 km is 'not evaluated'."""
    from insar_pipeline import gnss_tie as gt
    rs = gt.run(project.cfg)["radius_sensitivity"]
    assert rs["50"]["vertical_mm_yr"] is None and rs["50"]["status"] == "not evaluated"
    assert rs["50"]["missing_sites"] == ["CCCC"] and rs["50"]["evaluated"] is False
    assert rs["30"]["evaluated"] is True and rs["30"]["vertical_mm_yr"] is not None
    assert "CCCC" in rs["not_cached"]


def test_a_missing_station_that_could_not_qualify_does_not_block_the_radius(project):
    """CCCC's catalogue span overlapping the record by under 3 years would
    not have entered the background, so 50 km is evaluated without it."""
    from insar_pipeline import gnss_tie as gt
    midas = project.cache / "unr_midas_igs14.txt"
    midas.write_text("".join((l.replace(" 2010.5 2026.6 ", " 2024.5 2026.6 ") if l.startswith("CCCC") else l) + "\n"
                             for l in midas.read_text().splitlines()))
    rs = gt.run(project.cfg)["radius_sensitivity"]
    assert rs["50"]["evaluated"] is True and rs["50"]["vertical_mm_yr"] is not None


def test_a_missing_station_still_recording_at_the_catalogue_date_is_judged_to_the_record_end(project):
    """Review: the cached IGS14 catalogue stops at its own date (2024.74).
    CCCC began in 2022 and records to that date: 2.7 years by the catalogue,
    4.5 by its daily series - what the tie would measure. 50 km read
    'evaluated' without it; it is 'not evaluated'. A station that stopped
    before the catalogue's date is still judged by its span."""
    from insar_pipeline import gnss_tie as gt
    midas = project.cache / "unr_midas_igs14.txt"
    lines = [l.replace(" 2026.6 ", " 2024.7 ").replace(" 2026.0 ", " 2024.7 ")      # the file's own date
             for l in midas.read_text().splitlines()]
    midas.write_text("".join((l.replace(" 2010.5 2024.7 ", " 2022.0 2024.7 ") if l.startswith("CCCC") else l) + "\n"
                             for l in lines))
    rs = gt.run(project.cfg)["radius_sensitivity"]
    assert rs["50"]["evaluated"] is False and rs["50"]["missing_sites"] == ["CCCC"]
    midas.write_text("".join((l.replace(" 2010.5 2024.7 ", " 2020.0 2022.5 ") if l.startswith("CCCC") else l) + "\n"
                             for l in lines))                                     # stopped in 2022: 2.5 yr
    assert gt.run(project.cfg)["radius_sensitivity"]["50"]["evaluated"] is True


# ---------------------------------------------------------------- E16 the check at a station inside the area
N = 40


def _cube_dates(n=30, start="20160105", step_days=27):
    d0 = datetime.strptime(start, "%Y%m%d")
    return [(d0 + timedelta(days=step_days * i)).strftime("%Y%m%d") for i in range(n)]


@pytest.fixture
def inside(tmp_path, monkeypatch):
    """A 40x40 record whose time series the test writes, and a station at
    row 20, col 20 recording through the whole record."""
    from insar_pipeline import deliverables as dl
    ts = tmp_path / "geo_timeseries_inside.h5"
    monkeypatch.setattr(dl, "timeseries_file", lambda c: ts)
    cfg = types.SimpleNamespace(project_dir=tmp_path / "p")

    def write(dates, cube_mm):
        import h5py
        with h5py.File(ts, "w") as f:
            f["date"] = np.array(dates, dtype="S8")
            f["timeseries"] = (cube_mm / 1000.0).astype(np.float32)
    return cfg, write


def _gnss(rate, t0=2015.5, t1=2026.5, seed=5, local=None):
    t = _daily(t0, t1 - t0)
    u = rate * (t - t0) + np.random.default_rng(seed).normal(0, 1.0, t.size)
    if local is not None:
        u = u + local(t)
    return {"t": t, "u": u, "steps": []}


def _field(dates, sd, seed=0, block=1):
    """Per-date noise, constant over block x block cells (atmosphere-like)."""
    rng = np.random.default_rng(seed)
    nb = -(-N // block)
    e = rng.normal(0, sd, (len(dates), nb, nb))
    return np.repeat(np.repeat(e, block, axis=1), block, axis=2)[:, :N, :N]


def test_an_overlap_too_short_to_resolve_reads_not_testable_never_agree(inside):
    """LAMC's case: the radar over the overlap is too noisy to resolve 0.25
    in/yr. The verdict is 'not testable' and consistent None - the old
    check said 'the two agree'."""
    from insar_pipeline import gnss_tie as gt
    cfg, write = inside
    dates = _cube_dates()
    write(dates, _field(dates, 25.0, block=8))
    out = gt._series_check(cfg, dates, _gnss(-1.0), 20, 20, np.full((N, N), 0.78), np.ones((N, N), bool),
                           regional=[dict(_gnss(-1.0, seed=6), site="REG1")])
    assert out["verdict"] == "not testable" and out["consistent"] is None
    assert out["tolerance_95_vertical_mm_yr"] > gt.INSIDE_CHECK_USEFUL_MM_YR
    assert out["verdict_reason"].startswith("Not testable:") and "agree" not in out["verdict_reason"]
    assert "could only see a difference of about" in out["verdict_reason"]


def test_same_window_check_agrees_when_station_and_ground_move_with_the_region(inside):
    """The station moves with the regional stations (-3 mm/yr both) and the
    ground at it with the farmland: station minus region ~ 0, radar excess
    ~ 0, a tolerance under 0.25 in/yr - 'agree'. Its regional movement, which
    the relative radar cannot see, cancels."""
    from insar_pipeline import gnss_tie as gt
    cfg, write = inside
    dates = _cube_dates()
    write(dates, _field(dates, 1.0))
    out = gt._series_check(cfg, dates, _gnss(-3.0), 20, 20, np.full((N, N), 0.78), np.ones((N, N), bool),
                           regional=[dict(_gnss(-3.0, seed=6), site="REG1"), dict(_gnss(-3.0, seed=7), site="REG2")])
    assert out["verdict"] == "agree" and out["consistent"] is True
    assert out["regional_sites_over_overlap"] == ["REG1", "REG2"]
    assert abs(out["station_minus_regional_vertical_mm_yr"]) < 1.0
    assert out["tolerance_95_vertical_mm_yr"] <= gt.INSIDE_CHECK_USEFUL_MM_YR
    assert "the two agree within" in out["verdict_reason"]


def test_a_station_that_moves_on_its_own_differs(inside):
    """The station sinks 8 mm/yr against the regional stations while the
    radar reads its ground with the farmland: 'differ', consistent False."""
    from insar_pipeline import gnss_tie as gt
    cfg, write = inside
    dates = _cube_dates()
    write(dates, _field(dates, 1.0))
    out = gt._series_check(cfg, dates, _gnss(-9.0), 20, 20, np.full((N, N), 0.78), np.ones((N, N), bool),
                           regional=[dict(_gnss(-1.0, seed=6), site="REG1")])
    assert out["verdict"] == "differ" and out["consistent"] is False
    assert out["station_minus_regional_vertical_mm_yr"] == pytest.approx(-8.0, abs=1.0)
    assert "differ by more than this check can explain" in out["verdict_reason"]


def test_only_the_overlap_dates_are_compared(inside):
    """The ground at the station starts sinking 30 mm/yr (LOS) after the
    station stopped recording. The old page set the whole-record radar rate
    against the station's short record; here only the shared dates count,
    so the check agrees."""
    from insar_pipeline import gnss_tie as gt
    cfg, write = inside
    dates = _cube_dates(n=60)                                # 4.4 years; the station records the first 2.2
    t = np.array([gt._decyear(d) for d in dates])
    cube = _field(dates, 1.0)
    late = np.clip(t - 2018.3, 0, None) * -30.0
    cube[:, 17:24, 17:24] += late[:, None, None]
    write(dates, cube)
    g = _gnss(-1.0, t0=2015.5, t1=2018.25)
    out = gt._series_check(cfg, dates, g, 20, 20, np.full((N, N), 0.78), np.ones((N, N), bool),
                           regional=[dict(_gnss(-1.0, seed=6), site="REG1")])
    assert out["overlap"][1] <= 2018.25 and out["overlap_sar_dates"] < 60
    assert abs(out["insar_relative_over_overlap_vertical_mm_yr"]) < 2.0
    assert out["verdict"] == "agree"


def test_a_difference_beyond_even_a_coarse_tolerance_is_found(inside):
    """Too coarse to confirm agreement is not too coarse to find a large
    difference: a station sinking 60 mm/yr against the regional stations,
    on ground the radar reads with the farmland, 'differs' (it read 'not
    testable' when the coarse rule came first)."""
    from insar_pipeline import gnss_tie as gt
    cfg, write = inside
    dates = _cube_dates()
    write(dates, _field(dates, 25.0, block=8))
    out = gt._series_check(cfg, dates, _gnss(-61.0), 20, 20, np.full((N, N), 0.78), np.ones((N, N), bool),
                           regional=[dict(_gnss(-1.0, seed=6), site="REG1")])
    assert out["tolerance_95_vertical_mm_yr"] > gt.INSIDE_CHECK_USEFUL_MM_YR
    assert abs(out["difference_vertical_mm_yr"]) > out["tolerance_95_vertical_mm_yr"]
    assert out["verdict"] == "differ" and out["consistent"] is False


def test_an_overlap_too_short_for_a_gnss_rate_is_not_testable(inside):
    """Thirty SAR dates 12 days apart: under a year of overlap, no MIDAS rate
    or sigma for station minus region. The tolerance used to fall back to
    the radar threshold alone and called GNSS noise a difference ('differ'
    for a station moving exactly with the region)."""
    from insar_pipeline import gnss_tie as gt
    cfg, write = inside
    dates = _cube_dates(n=30, step_days=12)
    write(dates, _field(dates, 1.0))
    out = gt._series_check(cfg, dates, _gnss(-1.0), 20, 20, np.full((N, N), 0.78), np.ones((N, N), bool),
                           regional=[dict(_gnss(-1.0, seed=6), site="REG1")])
    assert out["overlap_years"] < 1.5
    assert out["verdict"] == "not testable" and out["consistent"] is None
    assert "at least 1.5 years" in out["verdict_reason"] and "agree" not in out["verdict_reason"]


def test_without_a_regional_station_over_the_overlap_it_is_not_testable(inside):
    """The station's own movement cannot be told from the region's when no
    background station recorded over the same years (ILWS and INEV began in
    2021; LAMC stopped in 2018)."""
    from insar_pipeline import gnss_tie as gt
    cfg, write = inside
    dates = _cube_dates()
    write(dates, _field(dates, 1.0))
    late = dict(_gnss(-1.0, t0=2021.2, t1=2026.5, seed=6), site="ILWS")
    out = gt._series_check(cfg, dates, _gnss(-1.0), 20, 20, np.full((N, N), 0.78), np.ones((N, N), bool),
                           regional=[late])
    assert out["verdict"] == "not testable" and out["consistent"] is None
    assert out["regional_sites_over_overlap"] == [] and "region" in out["verdict_reason"]


def test_tie_inside_check_never_sets_the_whole_record_figure_against_the_station(project, monkeypatch):
    """End to end: the inside station's entry carries the same-window verdict
    and `consistent`; the whole-record radar figure is kept only under
    full_record, labelled as not compared (the old entry put it top level,
    where the page set it against the station's short record as 'agree')."""
    import h5py
    from insar_pipeline import gnss_tie as gt
    monkeypatch.setitem(L1T.SITES, "HHHH", (39.0, -87.30, 2015.5, 2018.3, -1.0))   # the box centre, 2.8 yr
    monkeypatch.setattr(gt, "discover", lambda cfg, root, radius=None: ["AAAA", "HHHH"])
    dates = _cube_dates(n=60)
    with h5py.File(project.geo / "geo_timeseries_test.h5", "w") as f:
        f["date"] = np.array(dates, dtype="S8")
        f["timeseries"] = (_field(dates, 25.0, block=8) / 1000.0).astype(np.float32)   # atmosphere-like noise
    out = gt.run(project.cfg)
    chk = next(c for c in out["inside_checks"] if c["site"] == "HHHH")
    assert "insar_relative_class_matched_mm_yr" not in chk and "detrended_correlation" not in chk
    assert chk["full_record"]["insar_relative_class_matched_mm_yr"] is not None
    assert "NOT compared" in chk["full_record"]["note"]
    assert chk["verdict"] == "not testable" and chk["consistent"] is None
    assert chk["regional_sites_over_overlap"] == ["AAAA"]
    assert "HHHH" not in out["stations_used"]                        # 2.8 yr: too short for the background


def test_tie_inside_check_with_nothing_to_compare_is_not_testable(project, monkeypatch):
    """A station with too few SAR dates in its record gets a verdict too."""
    from insar_pipeline import gnss_tie as gt
    monkeypatch.setitem(L1T.SITES, "HHHH", (39.0, -87.30, 2010.5, 2026.6, -1.0))
    monkeypatch.setattr(gt, "discover", lambda cfg, root, radius=None: ["AAAA", "HHHH"])
    chk = next(c for c in gt.run(project.cfg)["inside_checks"] if c["site"] == "HHHH")   # the fixture has 2 dates
    assert chk["verdict"] == "not testable" and chk["consistent"] is None
    assert "fewer than 20 radar images" in chk["verdict_reason"]


def test_every_not_testable_reason_is_plain_english(inside):
    """Review: verdict_reason is the sentence the client page prints, so the
    reasons a check could not run in say 'radar images', not 'SAR dates'."""
    from insar_pipeline import gnss_tie as gt
    cfg, write = inside
    dates = _cube_dates()
    write(dates, _field(dates, 1.0))
    late = _gnss(-1.0, t0=2021.2, t1=2026.5)                     # recorded after the record's dates
    for out in (gt._series_check(cfg, dates, late, 20, 20, np.full((N, N), 0.78), np.ones((N, N), bool)),
                gt._series_check(cfg, dates, _gnss(-1.0), 20, 20, np.full((N, N), 0.78), np.zeros((N, N), bool))):
        assert "SAR" not in out["series_check"] and "overlap" not in out["series_check"]
    out = gt._series_check(cfg, dates, _gnss(-1.0), 20, 20, np.full((N, N), 0.78), np.ones((N, N), bool),
                           regional=[dict(_gnss(-1.0, t0=2021.2, t1=2026.5, seed=6), site="ILWS")])
    assert "SAR" not in out["verdict_reason"] and "background" not in out["verdict_reason"]


def test_inside_check_never_sets_the_station_against_a_monument_at_its_own_site(project, monkeypatch):
    """Review: a short-record station inside the area (not in the background)
    was set against every background site, a monument 0.6 km away at the
    same site included - the difference cancels the station's own movement
    and reads as agreement. Only the other sites are its region."""
    import h5py
    from insar_pipeline import gnss_tie as gt
    monkeypatch.setitem(L1T.SITES, "HHHH", (39.0, -87.30, 2015.5, 2018.3, -1.0))     # 2.8 yr: not in the background
    monkeypatch.setitem(L1T.SITES, "HHH2", (39.005, -87.30, 2010.5, 2026.6, -1.0))   # same site, long record
    monkeypatch.setattr(gt, "discover", lambda cfg, root, radius=None: ["AAAA", "HHH2", "HHHH"])
    dates = _cube_dates(n=60)
    with h5py.File(project.geo / "geo_timeseries_test.h5", "w") as f:
        f["date"] = np.array(dates, dtype="S8")
        f["timeseries"] = (_field(dates, 1.0) / 1000.0).astype(np.float32)
    out = gt.run(project.cfg)
    assert out["stations_used"] == ["AAAA", "HHH2"]
    chk = next(c for c in out["inside_checks"] if c["site"] == "HHHH")
    assert chk["regional_sites_over_overlap"] == ["AAAA"]


# ---------------------------------------------------------------- the real stations (read-only cache)
CACHE = Path("/home/kylew/insar_projects/gnss")
real = pytest.mark.skipif(not (CACHE / "INEV" / "INEV.tenv3").exists(), reason="no cached NGL series")


@real
def test_inev_antenna_slip_days_and_invi_antenna_step_on_the_real_series():
    """The audit's numbers: INEV has 192 days processed at 0 m antenna height
    (dropped); INVI's 6 Jun 2024 antenna change (steps file) moves its rate
    over the Site A record from -1.51 to -1.09 mm/yr."""
    from insar_pipeline import gnss_tie as gt
    t0, t1 = gt._decyear("20160105"), gt._decyear("20260730")
    inev = gt.screen_series(gt.read_tenv3(CACHE / "INEV" / "INEV.tenv3"))
    assert inev["antenna_epochs_dropped"] == 192 and inev["antenna_heights_dropped_m"] == [0.0]
    g = gt.read_tenv3(CACHE / "INVI" / "INVI.tenv3")
    w = (g["t"] >= t0) & (g["t"] <= t1)
    plain = gt.midas_rate(g["t"][w], g["u"][w])
    screened = gt.midas_rate(g["t"][w], g["u"][w], steps=(gt._ngl_decyear("15JUL02"), gt._ngl_decyear("24JUN06")))
    assert plain["rate_mm_yr"] == pytest.approx(-1.51, abs=0.02)
    assert screened["rate_mm_yr"] == pytest.approx(-1.09, abs=0.02)
    assert 0.8 < screened["sigma_mm_yr"] < 1.0


@real
def test_vinu_against_invi_is_not_an_outlier_with_realistic_sigmas():
    """Site B: VINU - INVI = -1.83 +/- 1.15 mm/yr over their common days -
    1.6 sigma, inside the 1.96 sigma limit - so VINU stays in the background
    (its disagreement widens the uncertainty instead)."""
    from insar_pipeline import gnss_tie as gt
    t0, t1 = gt._decyear("20160105"), gt._decyear("20260617")
    use, series = [], {}
    for s, steps in (("VINU", ("14MAY16",)), ("INVI", ("15JUL02", "24JUN06"))):
        g = gt.read_tenv3(CACHE / s / f"{s}.tenv3")
        sc = gt.screen_series(g, [gt._ngl_decyear(x) for x in steps])
        w = (sc["t"] >= t0) & (sc["t"] <= t1)
        m = gt.midas_rate(sc["t"][w], sc["u"][w], sc["steps"])
        use.append({"site": s, "lat": g["lat"], "lon": g["lon"], "rate_mm_yr": m["rate_mm_yr"],
                    "sigma_mm_yr": m["sigma_mm_yr"]})
        series[s] = sc
    kept, rec = gt.outlier_test(use, series, t0, t1)
    p = rec["pairs"][0]
    assert p["tested"] and p["difference_mm_yr"] == pytest.approx(-1.83, abs=0.03)
    assert p["sigma_mm_yr"] == pytest.approx(1.15, abs=0.03) and p["disagree"] is False and len(kept) == 2


def test_tie_json_stays_serialisable(project):
    """Control (passes on the old code too): the file round-trips."""
    from insar_pipeline import gnss_tie as gt
    out = gt.run(project.cfg)
    assert json.loads((project.cfg.project_dir / "deliverables" / "gnss_tie.json").read_text())["radius_km"] == \
        out["radius_km"]
