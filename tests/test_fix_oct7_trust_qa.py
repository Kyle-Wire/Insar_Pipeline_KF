"""
Trust review, 7 Oct 2026 - group qa (quality_assessment.py, stack_qc.py):
decisions E7 (the multi-scale stand-out calibrated for trying three
scales), E21 (one noise-floor rule for every project), E22 (geolocation:
a half counts only beyond one posting cell AND beyond the halves' own
scatter), E23 (image accounting counts every catalogue date the run
recorded) and E24 (a check that did not run reads "not run", never false).
Each test fails on the code before this round
(backups/pre_trust_20261007/) and passes on the new code.

Run: conda run -n insar python -m pytest tests/test_fix_oct7_trust_qa.py -q
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from insar_pipeline import quality_assessment as qa            # noqa: E402
from insar_pipeline import stack_qc                             # noqa: E402
import test_fix_L8 as L8                                         # noqa: E402


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("the QA tried to fetch OSM water in a test")
    monkeypatch.setattr(stack_qc, "fetch_osm_water", boom, raising=False)


# ======================================================================
# E21: one noise-floor population and one estimator for every project
# ======================================================================
def _noise_cells(n_pure, n_mixed, seed=0, tail=0.3):
    """Linked coherence of decorrelated cells: n_pure with MintPy TC < 0.15
    and n_mixed with 0.15-0.25; noise at 0.371 +/- 0.0075, a share `tail`
    of partly coherent cells (0.40-0.60), as Site B's population."""
    rng = np.random.default_rng(seed)
    n = n_pure + n_mixed
    linked = rng.normal(0.371, 0.0075, n)
    k = rng.random(n) < tail
    linked[k] = rng.uniform(0.40, 0.60, k.sum())
    um = np.r_[rng.uniform(0.0, 0.149, n_pure), rng.uniform(0.15, 0.249, n_mixed)]
    # coherent ground the population must leave out
    linked = np.r_[linked, rng.normal(0.6, 0.08, 20000)]
    um = np.r_[um, rng.uniform(0.5, 0.95, 20000)]
    return linked, um


def test_noise_floor_is_the_mode_plus_three_sigma_on_one_population():
    # Site B-like: 156 cells below 0.15, so the old rule moved to < 0.2 and
    # took the median, which the partly coherent tail drags up: 0.436 there
    linked, um = _noise_cells(156, 2640)
    r = stack_qc.noise_level(linked, um)
    assert r["measured"] is True
    assert r["population"].endswith("< 0.25") and r["n_cells"] == 2796
    assert abs(r["mode"] - 0.371) < 0.004
    assert 0.385 < r["floor"] < 0.40, r                     # 0.371 + 3 x ~0.0075; the old rule read > 0.41
    assert r["population_median"] > r["mode"]               # the tail moves the median, not the mode
    assert "median" not in r                                # the median is no longer the noise level


def test_noise_floor_does_not_move_with_the_population_purity():
    # Site A-like (a big pure lake) and Site B-like (mostly mixed
    # cells): the same noise must give the same floor (old: 0.397 vs 0.436)
    pure = stack_qc.noise_level(*_noise_cells(5000, 7600, seed=1, tail=0.05))
    mixed = stack_qc.noise_level(*_noise_cells(156, 2640, seed=2, tail=0.3))
    assert abs(pure["floor"] - mixed["floor"]) < 0.006, (pure, mixed)


def test_noise_population_share_on_mapped_water_is_recorded():
    linked, um = _noise_cells(156, 2640)
    wet = np.zeros(linked.size, bool)
    wet[:1000] = True
    r = stack_qc.noise_level(linked, um, on_water=wet)
    assert r["share_on_mapped_water"] == pytest.approx(1000 / 2796, abs=0.001)
    few = stack_qc.noise_level(linked[:200], um[:200])
    assert few["measured"] is False and few["floor"] is None


def test_an_unmeasured_floor_is_not_replaced_by_the_runs_lenient_level(tmp_path):
    # fewer than 300 decorrelated cells: the old code put the time-series
    # step's mode + 1.645 sigma (a more lenient point) in as the floor
    tc = np.full((L8.NY, L8.NX), 0.6)
    um = np.full((L8.NY, L8.NX), 0.8)
    um[:5, :20] = 0.1
    tc[:5, :20] = 0.37
    cfg = L8.build(tmp_path, tc=tc, um=um)
    (cfg.mintpy_dir / "run_products.json").write_text(json.dumps({"linked_noise_level": 0.384}))
    vel, attrs = qa._read2d(cfg.mintpy_dir / "geo" / "geo_velocity.h5", "velocity")
    rg = qa._RadarLayers(cfg, attrs, vel.shape)
    r = qa._noise_level(cfg, rg, tc, um, "test")
    assert r["measured"] is False and r["floor"] is None
    assert r["linked_noise_level_from_run"] == 0.384          # kept for the record only
    # the time-series step writes "nan" when it measured nothing: no level,
    # never a number a caption could quote ("scores about nan")
    (cfg.mintpy_dir / "run_products.json").write_text(json.dumps({"linked_noise_level": "nan"}))
    r = qa._noise_level(cfg, rg, tc, um, "test")
    assert "linked_noise_level_from_run" not in r and r["floor"] is None


def _square_lonlat(c0, c1, r0, r1):
    """a lon/lat polygon over grid columns c0..c1-1 and rows r0..r1-1 of the L8 grid"""
    x0, x1 = L8.X0 + c0 * L8.DX, L8.X0 + c1 * L8.DX
    y0, y1 = L8.Y0 + r0 * L8.DY, L8.Y0 + r1 * L8.DY
    return {"type": "Polygon", "coordinates": [[[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]]}


def test_an_invalid_water_polygon_never_stops_the_qa(tmp_path):
    # review 7 Oct 2026: the share on water is a record only, but a hand-made
    # context/water.geojson with a self-intersecting polygon made the union
    # raise inside the coverage check and the whole QA failed
    rng = np.random.default_rng(1)
    tc = np.full((L8.NY, L8.NX), 0.55)
    um = np.full((L8.NY, L8.NX), 0.85)
    noise = np.zeros((L8.NY, L8.NX), bool)
    noise[60:80, 60:85] = True                                    # 500 decorrelated cells
    tc[noise] = rng.normal(0.37, 0.008, noise.sum())
    um[noise] = rng.uniform(0.0, 0.2, noise.sum())
    cfg = L8.build(tmp_path, tc=tc, um=um)
    lake = _square_lonlat(60, 73, 60, 80)                         # 13 of the block's 25 columns
    sq = lake["coordinates"][0]
    bow = {"type": "Polygon", "coordinates": [[sq[0], sq[2], sq[1], sq[3], sq[0]]]}   # a bow-tie over it
    ctx = cfg.project_dir / "context"
    ctx.mkdir()
    (ctx / "water.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {}, "geometry": g} for g in (lake, bow)]}))
    nl = qa.evaluate(cfg)["checks"]["coverage"]["noise_level"]
    assert nl["measured"] is True
    assert nl["share_on_mapped_water"] == pytest.approx(13 / 25, abs=0.01)   # the repaired polygon counts once
    # an unreadable map: no share recorded, and still no failure
    (ctx / "water.geojson").write_text("{not json")
    nl = qa.evaluate(cfg)["checks"]["coverage"]["noise_level"]
    assert nl["measured"] is True and "share_on_mapped_water" not in nl


# ======================================================================
# E22: geolocation halves - one posting cell AND beyond their own scatter
# ======================================================================
SITE_A_GEO = {"available": True, "n_water_bodies": 53, "step_m": 2.5, "cell_m": 15.0, "search_m": 75.0,
                  "dE_m": 2.5, "dN_m": 0.0, "offset_m": 2.5, "score": 0.316,
                  "halves": {"all": {"dE_m": 2.5, "dN_m": 0.0, "offset_m": 2.5},
                             "west": {"dE_m": 0.0, "dN_m": 2.5, "offset_m": 2.5, "n_windows": 41},
                             "east": {"dE_m": 5.01, "dN_m": -10.02, "offset_m": 11.2, "n_windows": 12}},
                  "west_east_disagreement_m": 13.49, "split_test": {"p_value": 0.008, "n_permutations": 500}}
SITE_B_GEO = {"available": True, "n_water_bodies": 46, "step_m": 2.5, "cell_m": 15.0, "search_m": 75.0,
                 "dE_m": 5.01, "dN_m": 0.0, "offset_m": 5.01, "score": 0.347,
                 "halves": {"all": {"dE_m": 5.01, "dN_m": 0.0, "offset_m": 5.01},
                            "west": {"dE_m": 12.52, "dN_m": -17.53, "offset_m": 21.54, "n_windows": 5},
                            "east": {"dE_m": 5.01, "dN_m": 2.5, "offset_m": 5.6, "n_windows": 41}},
                 "west_east_disagreement_m": 21.39, "split_test": {"p_value": 0.30, "n_permutations": 500}}


def test_a_half_within_one_cell_is_a_note_not_a_hold():
    g = qa.judge_geolocation(SITE_A_GEO)
    assert g["flag"] is False and not g.get("hold") and g["within_cell"] is True
    assert "within about 3 m overall" in g["text"] and "east half reads about 11 m" in g["text"]
    assert "less than one 15 m map cell" in g["text"]
    assert qa.judge_geolocation(g) == g                                   # idempotent
    sr = qa.send_rule({"checks": {}, "geolocation": g})
    assert not any("geolocation" in h for h in sr["hold_reasons"])
    assert any(n.startswith("geolocation within a cell") for n in sr["notes"])


def test_a_half_resting_on_scatter_is_judged_on_the_whole_area():
    g = qa.judge_geolocation(SITE_B_GEO)                 # 5 lakes in the west half, p 0.3
    assert g["flag"] is False and not g.get("hold") and "within_cell" not in g
    assert g["text"].startswith("Map positions agree with mapped water edges to within about 5 m")
    assert not any("geolocation" in h for h in qa.send_rule({"checks": {}, "geolocation": g})["hold_reasons"])


def test_a_half_over_a_cell_beyond_scatter_is_flagged():
    rec = dict(SITE_A_GEO, offset_m=6.0,
               halves={"west": {"offset_m": 4.0, "n_windows": 30}, "east": {"offset_m": 20.0, "n_windows": 20}},
               west_east_disagreement_m=17.0, split_test={"p_value": 0.01})
    g = qa.judge_geolocation(rec)
    assert g["flag"] is True and "20 m off" in g["text"] and "east half" in g["text"]
    assert "more than one 15 m map cell" in g["text"]
    sr = qa.send_rule({"checks": {}, "geolocation": g})
    assert sr["value"] == "hold for specialist" and any("20.0 m off" in h for h in sr["hold_reasons"])


def test_split_test_and_window_counts_from_the_water_check():
    lakes, coh = L8._lakes_and_field(0.0, 0.0)
    r = stack_qc.geolocation_vs_water(coh, L8._attrs(), lakes, max_shift_m=150.0, n_permutations=60)
    assert r["available"] and r["cell_m"] == pytest.approx(132, abs=2)
    nw, ne = r["halves"]["west"]["n_windows"], r["halves"]["east"]["n_windows"]
    assert nw + ne == r["n_water_bodies"]
    st = r["split_test"]
    assert st["n_permutations"] > 0 and 0 < st["p_value"] <= 1
    # no shift anywhere: the halves agree within their scatter
    assert st["p_value"] > 0.05


# ======================================================================
# E23: image accounting counts the run's own record once slc/ is emptied
# ======================================================================
def test_date_accounting_counts_short_dates_from_the_run_record(tmp_path):
    cfg = L8.build(tmp_path)
    processed = list(L8.DATES)
    (cfg.project_dir / "run_results.json").write_text(json.dumps(
        {"dropped_short_dates": {"modal_burst_count": 3, "dropped": ["20180617", "20190110"]}}))
    r = qa.date_accounting(cfg, processed)
    assert r["available"] == len(processed) + 2 and r["processed"] == len(processed)
    assert r["dropped"]["20180617"] == qa._SEARCH_SHORT
    assert r["text"] == f"{len(processed)} of {len(processed) + 2} images used; 2 left out (see dropped)"
    assert r["download_record"].startswith("run_results.json")
    # slc/ on disk wins over the run record
    (cfg.project_dir / "slc").mkdir()
    (cfg.project_dir / "slc" / "dropped_short_dates.json").write_text(json.dumps({"dropped": ["20180617"]}))
    assert qa.date_accounting(cfg, processed)["available"] == len(processed) + 1


def test_each_download_record_falls_back_on_its_own(tmp_path):
    # slc/ keeps only the short-date list; the catalogue the download searched
    # survives only in run_results.json: both count, and the record says where
    # each came from (the old code read slc/ only and never the run record)
    cfg = L8.build(tmp_path)
    processed = list(L8.DATES)
    (cfg.project_dir / "slc").mkdir()
    (cfg.project_dir / "slc" / "dropped_short_dates.json").write_text(json.dumps({"dropped": ["20180617"]}))
    (cfg.project_dir / "run_results.json").write_text(json.dumps(
        {"assembly_status": {"searched": processed + ["20180617", "20190110"], "dropped": {}}}))
    r = qa.date_accounting(cfg, processed)
    assert r["available"] == len(processed) + 2
    assert "20190110" in r["dropped"] and r["unexplained"] == ["20190110"]
    src = r["download_record"]
    assert src.startswith("run_results.json assembly_status") and src.endswith(", slc/dropped_short_dates.json")
    assert "slc/assembly_status.json" not in src


# ======================================================================
# E24: the removed-tilt check that did not run reads "not run"
# ======================================================================
def test_removed_gradient_without_a_record_reads_not_measured(tmp_path):
    r = qa.evaluate(L8.build(tmp_path, mintpy_deramp="linear"))
    rs = r["ramp_sensitivity"]
    assert rs["removed_gradient_flag"] is None and rs["removed_gradient_status"].startswith("not measured")
    assert any("tilt the deramp removed was not checked" in n for n in r["send_rule"]["notes"])
    assert r["send_rule"]["value"] != "send"


def test_removed_gradient_is_measured_from_the_series_on_disk(tmp_path):
    # the series before and after the deramp are on disk (geocoded copies):
    # the removed tilt is a plane of 40 mm/yr per 10 km eastward
    import h5py
    cfg = L8.build(tmp_path, mintpy_deramp="linear")
    a = L8._attrs()
    t = np.arange(len(L8.DATES)) * 24 / 365.25
    lon, _ = L8._lonlat()
    east_km = (lon - lon.mean()) * 111.32 * np.cos(np.radians(38.41))
    plane = 4.0 * east_km / 1000.0                                  # m/yr: 40 mm/yr per 10 km
    after = np.zeros((len(L8.DATES), L8.NY, L8.NX), np.float32)
    before = (after + plane[None] * t[:, None, None]).astype(np.float32)
    for name, cube in (("timeseries_ERA5.h5", before), ("timeseries_ERA5_ramp.h5", after)):
        L8._h5(cfg.mintpy_dir / name, {"timeseries": cube, "date": np.array([d.encode() for d in L8.DATES])},
               dict(a, FILE_TYPE="timeseries"))
    r = qa.evaluate(cfg)
    rs = r["ramp_sensitivity"]
    assert rs["before_deramp"].startswith("measured")
    assert rs["removed_ramp"]["gradient_mm_yr_per_10km"] == pytest.approx(40.0, rel=0.02)
    assert rs["removed_ramp"]["source"].startswith("measured by the QA")
    assert rs["removed_gradient_flag"] in (True, False)
    assert rs["removed_gradient_status"].startswith(("measured", "not tested"))
    with h5py.File(cfg.mintpy_dir / "timeseries_ERA5.h5", "r") as f:
        assert f["timeseries"].shape[0] == len(L8.DATES)


# ======================================================================
# E7: the multi-scale stand-out, calibrated for trying three scales
# ======================================================================
CELL = 100.0                       # synthetic grid: 100 m cells, 1/2/4 km windows of 10/20/40 cells


def _grid_attrs(ny, nx):
    dy = CELL / 110_950.0
    dx = CELL / (111_320.0 * np.cos(np.radians(38.4)))
    return {"X_FIRST": "-87.8", "Y_FIRST": "38.5", "X_STEP": str(dx), "Y_STEP": str(-dy),
            "LENGTH": str(ny), "WIDTH": str(nx)}


def _noise_field(ny, nx, seed, corr_m=300.0, df=None):
    """stationary correlated noise (smoothed on a larger periodic grid and
    cropped, so no edge effect) with a Gaussian or Student-t (df) marginal"""
    from scipy.ndimage import gaussian_filter
    from scipy.stats import norm, t as student
    rng = np.random.default_rng(seed)
    g = gaussian_filter(rng.standard_normal((ny + 200, nx + 200)), corr_m / CELL, mode="wrap")[100:-100, 100:-100]
    g = (g - g.mean()) / g.std()
    if df is None:
        return 3.0 * g
    u = norm.cdf(g)
    return 3.0 * student.ppf(np.clip(u, 1e-9, 1 - 1e-9), df)


def _search(field, n_copies=39, seed=0, bowl=None):
    ny, nx = field.shape
    attrs = _grid_attrs(ny, nx)
    inside = np.zeros((ny, nx), bool)
    inside[20:-20, 20:-20] = True
    mine = np.zeros((ny, nx), bool)
    mine[60:70, 60:70] = True
    from insar_pipeline.significance import euclidean_buffer
    mine_buf = euclidean_buffer(mine, attrs, 500.0)
    good = np.isfinite(field)
    far = good & inside & ~mine_buf
    return qa.multi_scale_standout(field, good & inside, inside, mine_buf, far, attrs, mine, n_copies=n_copies)


def test_old_gap_rule_fires_on_heavy_tailed_noise_and_the_calibrated_rule_does_not():
    # Student-t (3) farmland, no feature anywhere: the uncalibrated rule
    # (gap > 2 at any scale) fired in a third of these reports; the
    # calibrated one stays near its level (39 copies: about 7%)
    old = new = 0
    n = 40
    for s in range(n):
        ms, cal = _search(_noise_field(220, 220, 2000 + s, df=3), n_copies=39)
        assert cal["tested"] is True
        old += any(m["gap_in_spreads"] > 2.0 for m in ms)
        new += any(m["stands_out"] for m in ms)
    assert old >= 8, old
    assert new <= 0.15 * n and 2 * new < old, (old, new)


def test_calibrated_p_is_uniform_on_gaussian_farmland():
    # pure Gaussian farmland: the smallest calibrated p over the scales is a
    # p-value, so about half the reports read p <= 0.5 and few p <= 0.05
    ps = []
    for s in range(40):
        ms, _ = _search(_noise_field(220, 220, 3000 + s), n_copies=39)
        ps.append(min(m["p_calibrated"] for m in ms))
    ps = np.asarray(ps)
    assert 0.3 <= np.mean(ps <= 0.5) <= 0.7, ps
    assert np.mean(ps <= 0.05) <= 0.15, ps


def test_a_real_bowl_stands_out_and_every_scale_keeps_its_location():
    f = _noise_field(220, 220, 7)
    yy, xx = np.indices(f.shape)
    f -= 14.0 * np.exp(-(((yy - 150) ** 2 + (xx - 60) ** 2) / (2 * 12.0 ** 2)))      # ~2.5 km wide, 14 mm/yr deep
    ms, cal = _search(f)
    out = [m for m in ms if m["stands_out"]]
    assert out, ms
    assert all(m["gap_in_spreads"] > cal["gap_threshold"] and m["p_calibrated"] <= 0.05 for m in out)
    assert all({"lat", "lon", "scale_km"} <= set(m) for m in ms)
    assert cal["copies_match"] and all(v["ratio"] is not None for v in cal["copies_match"].values())


def test_calibration_fails_closed_without_farmland():
    f = _noise_field(220, 220, 3)
    ny, nx = f.shape
    attrs = _grid_attrs(ny, nx)
    inside = np.zeros((ny, nx), bool)
    inside[20:-20, 20:-20] = True
    none = np.zeros((ny, nx), bool)
    ms, cal = qa.multi_scale_standout(f, inside, inside, none, none, attrs, none, n_copies=9)
    assert cal["tested"] is False and "could not be calibrated" in cal["reason"]
    assert ms and all(m["stands_out"] is None and m["gap_threshold"] is None for m in ms)


def test_qa_records_the_calibration_and_all_standouts(tmp_path):
    rng = np.random.default_rng(5)
    v = rng.normal(0, 1.0, (L8.NY, L8.NX))
    yy, xx = np.indices(v.shape)
    v -= 9.0 * np.exp(-(((yy - 70) ** 2 + (xx - 25) ** 2) / (2 * 5.0 ** 2)))         # a bowl away from the mine
    r = qa.evaluate(L8.build(tmp_path, vel_mm=v))
    mp = r["findings"]["monitoring_point"]
    cal = mp["standout_calibration"]
    assert cal["tested"] is True and cal["n_copies"] == qa.STANDOUT_COPIES
    assert len(cal["copies_largest_gap"]) == qa.STANDOUT_COPIES
    assert all(m["gap_threshold"] == cal["gap_threshold"] for m in mp["multi_scale"])
    assert mp["standouts"] == [m for m in sorted(mp["multi_scale"], key=lambda x: -x["gap_in_spreads"])
                               if m["stands_out"]]
    assert mp["standout"] == (mp["standouts"][0] if mp["standouts"] else None)


def test_calibrated_standouts_ignores_an_older_uncalibrated_record():
    old = {"findings": {"monitoring_point": {"standout": {"lat": 38.24, "lon": -87.65, "scale_km": 4.0,
                                                          "gap_in_spreads": 2.2, "stands_out": True}}}}
    assert qa.calibrated_standouts(old) == []                  # gap > 2, never calibrated
    so = {"lat": 38.24, "lon": -87.65, "scale_km": 4.0, "gap_in_spreads": 2.2, "gap_threshold": 1.9,
          "p_calibrated": 0.03, "stands_out": True}
    new = {"findings": {"monitoring_point": {"standout": so, "standouts": [so],
                                             "standout_calibration": {"tested": True, "gap_threshold": 1.9}}}}
    assert qa.calibrated_standouts(new) == [so]
    new["findings"]["monitoring_point"]["standout_calibration"]["tested"] = False
    assert qa.calibrated_standouts(new) == []


def test_a_standout_on_unmatched_copies_is_noted():
    so = {"lat": 38.24, "lon": -87.65, "scale_km": 4.0, "gap_in_spreads": 2.2, "gap_threshold": 2.02,
          "p_calibrated": 0.03, "stands_out": True}
    cal = {"tested": True, "gap_threshold": 2.02, "copies_match_within_band": False,
           "copies_match": {"1": {"ratio": 0.88}, "2": {"ratio": 0.85}, "4": {"ratio": 0.72}}}
    q = {"checks": {}, "geolocation": {"flag": False},
         "findings": {"monitoring_point": {"standout": so, "standouts": [so], "standout_calibration": cal}}}
    sr = qa.send_rule(q)
    assert sr["value"] == "send with notes" and any("4 km 0.72" in n for n in sr["notes"]), sr
    assert not any("1 km" in n for n in sr["notes"])
    cal["copies_match_within_band"] = True
    assert qa.send_rule(q)["value"] == "send"
