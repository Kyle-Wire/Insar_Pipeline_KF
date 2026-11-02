"""
Lane L8 (5 Oct 2026 review): the quality assessment's new records, each
tested on synthetic data so that the old code (which had none of them)
fails and the new code passes. No network: the OSM fetch is replaced by a
function that raises if it is ever called.

Run: conda run -n insar python -m pytest tests/test_fix_L8.py -q
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from insar_pipeline import quality_assessment as qa            # noqa: E402
from insar_pipeline import stack_qc                             # noqa: E402
from insar_pipeline.config import PipelineConfig               # noqa: E402

NY, NX = 90, 110
X0, Y0 = -87.80, 38.47
DX, DY = 0.0015, -0.0012            # ~131 m x 133 m cells: a small grid with room for a far field
DATES = [f"2018{m:02d}{d:02d}" for m, d in
         [(6, 11), (7, 5), (7, 29), (8, 22), (9, 15), (10, 9), (11, 2), (11, 26), (12, 20)]] + \
        [f"2019{m:02d}{d:02d}" for m, d in [(1, 13), (2, 6), (3, 2), (3, 26), (4, 19), (5, 13), (6, 6)]]


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("the QA tried to fetch OSM water in a test")
    monkeypatch.setattr(stack_qc, "fetch_osm_water", boom, raising=False)


def _attrs(**extra):
    a = {"X_FIRST": str(X0), "Y_FIRST": str(Y0), "X_STEP": str(DX), "Y_STEP": str(DY),
         "LENGTH": str(NY), "WIDTH": str(NX), "REF_Y": "5", "REF_X": "5",
         "REF_LAT": str(Y0 + 5.5 * DY), "REF_LON": str(X0 + 5.5 * DX), "UNIT": "m/year",
         "WAVELENGTH": "0.05546576", "CENTER_INCIDENCE_ANGLE": "39.0"}
    a.update(extra)
    return a


def _lonlat():
    lon = X0 + (np.arange(NX)[None, :] + 0.5) * DX + 0 * np.arange(NY)[:, None]
    lat = Y0 + (np.arange(NY)[:, None] + 0.5) * DY + 0 * np.arange(NX)[None, :]
    return lon, lat


def _square(w, s, e, n):
    return [[w, s], [e, s], [e, n], [w, n], [w, s]]


def _fc(*rings):
    return {"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {}, "geometry": {"type": "Polygon", "coordinates": [r]}} for r in rings]}


def _h5(path, datasets, attrs):
    import h5py
    with h5py.File(path, "w") as f:
        for k, v in datasets.items():
            f.create_dataset(k, data=v)
        for k, v in attrs.items():
            f.attrs[k] = v


def build(tmp_path, vel_mm=None, tc=None, um=None, mask=None, thr=0.4, report_areas=(), **over):
    """A geocoded MintPy-schema project: mine square in the middle, steady
    rates (vel_mm, LOS mm/yr) written as a series, optional unwrapping-
    consistency layer `um` (then tc is labelled MiaplPy's)."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    lon, lat = _lonlat()
    mine_ring = _square(-87.70, 38.40, -87.66, 38.43)
    (tmp_path / "mine.geojson").write_text(json.dumps(_fc(mine_ring)))
    kw = dict(project_dir=tmp_path / "proj", bbox=[-87.80, 38.362, -87.635, 38.47],
              start_date="2018-06-01", end_date="2019-06-30", backend="miaplpy",
              deramp_exclude_geojson=str(tmp_path / "mine.geojson"),
              reference_lalo=f"{Y0 + 5.5 * DY:.6f},{X0 + 5.5 * DX:.6f}",
              acceptance_far_field_buffer_m=500.0, min_temporal_coherence=thr,
              report_areas=tuple(report_areas) or ("Full Satellite Imaging:full",))
    kw.update(over)
    cfg = PipelineConfig(**kw)
    cfg.mintpy_dir.mkdir(parents=True, exist_ok=True)
    geo = cfg.mintpy_dir / "geo"
    geo.mkdir(exist_ok=True)
    rng = np.random.default_rng(3)
    if vel_mm is None:
        vel_mm = rng.normal(0, 1.0, (NY, NX))
    if tc is None:
        tc = np.full((NY, NX), 0.75)
    t = np.arange(len(DATES)) * 24 / 365.25
    ts = (np.asarray(vel_mm)[None] / 1000.0 * t[:, None, None]
          + rng.normal(0, 0.0005, (len(DATES), NY, NX))).astype(np.float32)
    vel = np.polyfit(t, ts.reshape(len(DATES), -1), 1)[0].reshape(NY, NX).astype(np.float32)
    if mask is None:
        mask = tc >= thr
    # like MintPy: the velocity exists everywhere, the mask is separate
    a = _attrs()
    _h5(geo / "geo_velocity.h5", {"velocity": vel, "velocityStd": np.full((NY, NX), 0.0005, np.float32)},
        dict(a, FILE_TYPE="velocity"))
    _h5(geo / "geo_timeseries_ERA5_ramp_demErr.h5",
        {"timeseries": ts, "date": np.array([d.encode() for d in DATES])}, dict(a, FILE_TYPE="timeseries"))
    _h5(geo / "geo_maskTempCoh.h5", {"mask": mask.astype(bool)}, a)
    tca = dict(a)
    if um is not None:
        tca["temporalCoherenceSource"] = "MiaplPy phase linking, average (mini-stack) temporal coherence"
        _h5(geo / "geo_temporalCoherence_mintpy.h5", {"temporalCoherence": um.astype(np.float32)}, a)
    _h5(geo / "geo_temporalCoherence.h5", {"temporalCoherence": tc.astype(np.float32)}, tca)
    _h5(geo / "geo_geometryRadar.h5", {"incidenceAngle": np.full((NY, NX), 39.0, np.float32),
                                       "azimuthAngle": np.full((NY, NX), -100.0, np.float32),
                                       "height": np.zeros((NY, NX), np.float32)}, a)
    return cfg


# ======================================================================
# empirical-14: dates available / processed / dropped, with the reason
# ======================================================================
def test_date_accounting_names_every_left_out_date(tmp_path):
    cfg = build(tmp_path, stack_reference_date="20180729")
    processed = [d for d in DATES if d not in ("20180729", "20190413")]
    (cfg.mintpy_dir / "network_connectivity.json").write_text(json.dumps(
        {"dates": len(DATES), "kept_pairs": 40, "orphaned_dates": [], "components": [len(processed)],
         "excluded_dates": ["20180729"]}))
    (cfg.mintpy_dir / "reference_excluded_pairs.json").write_text(json.dumps(
        {"excluded_for_reference": ["20180705_20180729", "20180729_20180822"]}))
    slc = cfg.project_dir / "slc"
    slc.mkdir()
    (slc / "dropped_short_dates.json").write_text(json.dumps({"modal_burst_count": 3, "dropped": ["20180617"]}))
    coreg = cfg.stack_dir / "coreg_secondarys"
    coreg.mkdir(parents=True)
    for d in processed + ["20190413"]:            # 20190413 coregistered, never in the series, no reason
        (coreg / f"{d}.cleaned").write_text("")
    r = qa.date_accounting(cfg, processed)
    assert r["processed"] == len(processed)
    assert r["available"] == len(processed) + 3
    assert "reference-pixel check" in r["dropped"]["20180729"]
    # a legacy dropped_short_dates.json proves only that the search came back
    # short (outcome check, 6 Oct 2026: 4 of Site B's 5 were rendering gaps)
    assert "archive search returned it incomplete" in r["dropped"]["20180617"]
    assert r["unexplained"] == ["20190413"]
    assert r["reference_date_excluded"] is True
    assert r["hold"] is True and r["text"].startswith(f"{len(processed)} of {len(processed) + 3} images used")
    s = qa.send_rule({"checks": {}, "dates": r})
    assert s["value"] == "hold for specialist" and any("20190413" in x for x in s["hold_reasons"])
    # explained drops only: noted, not held
    r2 = qa.date_accounting(cfg, processed + ["20190413"])
    assert r2["unexplained"] == [] and r2["hold"] is False
    assert qa.send_rule({"checks": {}, "dates": r2, "geolocation": {"flag": False}})["value"] == "send with notes"


def test_network_check_carries_excluded_dates(tmp_path):
    import h5py
    cfg = build(tmp_path, stack_reference_date="20180729")
    pairs = [(a, b) for a, b in zip(DATES, DATES[1:])] + [(a, b) for a, b in zip(DATES, DATES[2:])]
    (cfg.mintpy_dir / "inputs").mkdir()
    with h5py.File(cfg.mintpy_dir / "inputs" / "ifgramStack.h5", "w") as f:
        f.create_dataset("date", data=np.array([[a.encode(), b.encode()] for a, b in pairs]))
        f.create_dataset("dropIfgram", data=np.ones(len(pairs), bool))
    (cfg.mintpy_dir / "network_connectivity.json").write_text(json.dumps({"dates": 17, "excluded_dates": ["20180729"]}))
    r = qa.evaluate(cfg)
    net = r["checks"]["network"]
    assert "20180729" in net["excluded_dates"] and net["reference_date_excluded"] is True
    assert r["dates"]["processed"] == len(DATES)


# ======================================================================
# methodology-3: noise floor, both gates, the second gate's effect
# ======================================================================
def test_noise_level_floor_from_decorrelated_ground():
    rng = np.random.default_rng(0)
    linked = np.r_[rng.normal(0.37, 0.008, 4000), rng.normal(0.55, 0.05, 20000)]
    um = np.r_[rng.uniform(0.0, 0.12, 4000), rng.uniform(0.5, 0.95, 20000)]
    r = stack_qc.noise_level(linked, um)
    assert r["measured"] and abs(r["mode"] - 0.37) < 0.003            # E21: the mode, 7 Oct 2026
    assert 0.39 < r["floor"] < 0.40                       # 0.37 + 3 x 0.008
    assert stack_qc.noise_level(linked[4000:], um[4000:])["measured"] is False   # no decorrelated ground


def test_coverage_graded_on_both_gates_and_noise_floor(tmp_path):
    rng = np.random.default_rng(1)
    tc = np.full((NY, NX), 0.55); um = np.full((NY, NX), 0.85)
    tc[:, :44] = 0.42; um[:, :44] = 0.45                  # 40% of the grid: passes linked, fails 0.7
    water = np.zeros((NY, NX), bool); water[60:80, 60:85] = True
    tc[water] = rng.normal(0.37, 0.008, water.sum()); um[water] = rng.uniform(0.0, 0.12, water.sum())
    cfg = build(tmp_path, tc=tc, um=um, thr=0.4)
    c = qa.evaluate(cfg)["checks"]["coverage"]
    land = ~water
    # the old check: share at the threshold, graded alone (it passed at 0.4 on noise)
    assert c["aoi_percent_at_threshold"] > 70
    assert c["noise_level"]["measured"] and abs(c["noise_level"]["mode"] - 0.37) < 0.005   # E21
    assert c["threshold_above_noise"] is True and c["pass"] is True
    exp_rel = 100 * (land & (um >= 0.7)).sum() / (tc.size)
    assert abs(c["reliable_percent"] - exp_rel) < 3.0      # imaged AOI ~ whole grid
    assert c["second_gate"]["coverage_percent"]["0.7"] == c["reliable_percent"]
    assert c["second_gate"]["by_linked_band"][0]["share_reliable"] == 0.0
    assert c["second_gate"]["effect_on_figures"]["available"] is True
    # a threshold inside the noise floor is a flag, not a pass
    cfg2 = build(tmp_path / "b", tc=tc, um=um, thr=0.38)
    c2 = qa.evaluate(cfg2)["checks"]["coverage"]
    assert c2["threshold_above_noise"] is False and c2["pass"] is False
    assert qa.send_rule({"checks": {"coverage": c2}})["value"] == "hold for specialist"


def test_second_gate_applied_only_when_configured(tmp_path):
    tc = np.full((NY, NX), 0.55); um = np.full((NY, NX), 0.85)
    tc[:, :44] = 0.42; um[:, :44] = 0.45
    cfg = build(tmp_path, tc=tc, um=um)
    cfg.min_unwrap_consistency = 0.5                       # set as an attribute (new config field)
    c = qa.evaluate(cfg)["checks"]["coverage"]
    assert c["second_gate"]["applied"] is True
    assert abs(c["aoi_percent_at_threshold"] - 60.0) < 3.0 and c["aoi_percent_first_gate"] > 95
    assert c["pass"] is False                              # 60% < 70% once the gate is used


# ======================================================================
# gap-surface_mine_site-8: plausibility from rate gradients
# ======================================================================
def test_gradient_limit_and_flags_not_absolute_rate():
    dates = [f"2020{m:02d}{d:02d}" for m in range(1, 13) for d in (1, 13, 25)]
    lim = stack_qc.gradient_limit_mm_yr(0.05546576, dates)
    assert lim["median_gap_days"] == 12.0 and 415 < lim["limit_mm_yr"] < 430
    yy, xx = np.mgrid[0:60, 0:60]
    bowl = -600.0 * np.exp(-((yy - 30) ** 2 + (xx - 30) ** 2) / (2 * 15.0 ** 2))   # 600 mm/yr, smooth
    assert stack_qc.rate_gradient_flags(bowl, (1, 1), lim["limit_mm_yr"]).sum() == 0
    step = bowl.copy(); step[20:25, 40] -= 700.0                                   # a 700 mm/yr cliff
    bad = stack_qc.rate_gradient_flags(step, (1, 1), lim["limit_mm_yr"])
    assert bad[20:25, 40].all() and bad.sum() <= 5 * 3 + 2


def test_plausibility_recorded_in_qa(tmp_path):
    v = np.random.default_rng(2).normal(0, 1.0, (NY, NX))
    v[40:44, 70] = -900.0                                   # aliased cells inside the mine square
    cfg = build(tmp_path, vel_mm=v)
    p = qa.evaluate(cfg)["plausibility"]
    assert p["available"] and p["cells_over_gradient_limit"] >= 4
    assert p["flag"] is True
    assert any(a["name"] == "mapped mine polygons" and a["cells"] >= 4 for a in p["by_area"])


# ======================================================================
# methodology-13: matching sensitivity on a second axis
# ======================================================================
def _ctx_for_matching(tmp_path, landcover=False):
    rng = np.random.default_rng(4)
    tc = np.full((NY, NX), 0.5)                              # one coherence class everywhere
    um = rng.uniform(0.3, 0.95, (NY, NX))
    # unwrap-inconsistent ground (most of this farmland) reads -12 mm/yr: a
    # majority, so robust (median) class centres carry it too
    vel = rng.normal(0, 0.5, (NY, NX)) - 12.0 * (um < 0.75)
    region = np.zeros((NY, NX), bool); region[10:30, 10:30] = True
    um[region] = rng.uniform(0.8, 0.95, region.sum())       # a consistent patch, unlike most farmland
    vel[region] = rng.normal(0, 0.5, region.sum())          # and not moving
    far = ~region
    from insar_pipeline import significance as sg
    r_cm, _ = sg.class_matched_field(vel, tc, far)
    cfg = build(tmp_path)
    if landcover:
        import rasterio
        from rasterio.transform import from_origin
        lc = np.where(um < 0.75, 82, 71).astype(np.uint8)    # land cover tracks the bias here
        lc[region] = 71
        p = tmp_path / "nlcd.tif"
        with rasterio.open(p, "w", driver="GTiff", height=NY, width=NX, count=1, dtype="uint8",
                           crs="EPSG:4326", transform=from_origin(X0, Y0, DX, -DY)) as ds:
            ds.write(lc, 1)
        cfg.landcover_raster = str(p)
    rg = qa._RadarLayers(cfg, _attrs(), (NY, NX))
    return {"cfg": cfg, "vel": vel, "tc": tc, "good": np.ones((NY, NX), bool), "far": far, "um": um,
            "um_src": "test", "r_cm": r_cm, "rg": rg, "attrs": _attrs(),
            "regions": {"patch": (region, "area")}, "sigmas": {"patch": 1.0}}


def test_matching_sensitivity_flags_a_second_axis_bias(tmp_path):
    ctx = _ctx_for_matching(tmp_path)
    r = qa._matching_sensitivity(ctx)
    row = r["areas"][0]
    assert row["reported_mm_yr"] > 2.5                       # coherence-only matching: biased by the farmland mix
    assert abs(row["unwrap_consistency_tercile_mm_yr"]) < 1.0  # matched on the second axis: ~0
    assert row["max_change_in_sigma"] > 1.0 and r["flag"] is True
    assert qa.send_rule({"checks": {}, "matching_sensitivity": r})["value"] == "hold for specialist"


def test_matching_sensitivity_uses_landcover_raster(tmp_path):
    ctx = _ctx_for_matching(tmp_path, landcover=True)
    ctx["um"] = None                                         # land cover alone
    r = qa._matching_sensitivity(ctx)
    assert "land_cover" in r["axes"]
    assert abs(r["areas"][0]["land_cover_mm_yr"]) < 1.0 and r["flag"] is True


# ======================================================================
# methodology-14: ramp sensitivity, configurable buffer, before-deramp
# ======================================================================
def test_surface_fit_is_exact_on_a_plane_and_linear():
    yy, xx = np.mgrid[0:50, 0:70].astype(float)
    plane = 3.0 + 0.2 * yy - 0.1 * xx
    quad = 0.01 * (xx - 35) ** 2
    m = np.ones(plane.shape, bool); m[20:30, 30:40] = False
    np.testing.assert_allclose(stack_qc.surface_fit(plane, m, 1), plane, atol=1e-9)
    # fitting a quadratic to (quad - its plane) returns the same as quad - plane: linearity
    p_ = stack_qc.surface_fit(quad, m, 1)
    np.testing.assert_allclose(stack_qc.surface_fit(quad - p_, m, 2), quad - p_, atol=1e-8)


def test_ramp_sensitivity_measures_each_choice(tmp_path):
    lon, lat = _lonlat()
    # residual curvature across the scene that a quadratic deramp would remove
    v = 150.0 * ((lon - lon.mean()) / (lon.max() - lon.min())) ** 2
    cfg = build(tmp_path, vel_mm=v, mintpy_deramp="linear",
                report_areas=("Full Satellite Imaging:full", "east box:38.38,-87.70,38.44,-87.64"))
    cfg.deramp_buffer_m = 1000.0
    # the series before and after the deramp step (here on the product grid):
    # the step removed a 5 mm/yr west-east plane
    t = np.arange(len(DATES)) * 24 / 365.25
    ramp = 5.0 * (lon - lon.mean()) / (lon.max() - lon.min())
    after = np.zeros((len(DATES), NY, NX), np.float32)
    before = (ramp[None] / 1000.0 * t[:, None, None]).astype(np.float32)
    for name, cube in (("timeseries_ERA5.h5", before), ("timeseries_ERA5_ramp.h5", after)):
        _h5(cfg.mintpy_dir / name, {"timeseries": cube, "date": np.array([d.encode() for d in DATES])},
            _attrs(FILE_TYPE="timeseries"))
    r = qa.evaluate(cfg)
    rs = r["ramp_sensitivity"]
    assert rs["buffer_m"] == 1000.0 and rs["alternative_buffer_m"] == 4000.0
    assert 50 < rs["fit_area_percent"] < 100
    row = next(a for a in rs["areas"] if a["name"] == "east box")
    assert row["quadratic_ramp_mm_yr"] is not None and "buffer_4km_mm_yr" in row
    assert abs(row["quadratic_ramp_mm_yr"] - row["reported_mm_yr"]) > 1.0
    # adding back the removed plane moves the east box by its ramp minus the farmland's
    assert row["before_deramp_mm_yr"] - row["reported_mm_yr"] > 0.5
    assert rs["before_deramp"].startswith("measured")


def test_ramp_sensitivity_without_preramp_series_says_so(tmp_path):
    cfg = build(tmp_path, mintpy_deramp="linear")
    rs = qa.evaluate(cfg)["ramp_sensitivity"]
    assert "not on disk" in rs["before_deramp"]
    assert all("before_deramp_mm_yr" not in a for a in rs["areas"])


# ======================================================================
# gap-geolocation-4: DEM-error position shift near mine edges
# ======================================================================
def test_dem_error_shift_flagged_near_mine_edges(tmp_path):
    from insar_pipeline import significance as sg
    cfg = build(tmp_path, surface_mines_geojson="")
    a = _attrs()
    mine = sg.panel_mask(str(tmp_path / "mine.geojson"), (NY, NX), a)
    edge = sg.euclidean_buffer(mine, a, 300.0) & sg.euclidean_buffer(~mine, a, 300.0)
    rng = np.random.default_rng(5)
    dh = rng.normal(0, 3.0, (NY, NX))                         # farmland: a few metres of DEM-error noise
    dh[edge] = -150.0                                         # 185 m of shift: over one (coarse) test cell
    _h5(cfg.mintpy_dir / "geo" / "geo_demErr.h5", {"dem": dh.astype(np.float32)}, a)
    d = qa.evaluate(cfg)["dem_error"]
    assert d["available"] and d["flag"] is True
    z = next(z for z in d["zones"] if z["name"].startswith("mine edges"))
    assert z["excess_over_farmland_points"] > 50
    # noise alone does not flag
    _h5(cfg.mintpy_dir / "geo" / "geo_demErr.h5", {"dem": rng.normal(0, 3.0, (NY, NX)).astype(np.float32)}, a)
    assert qa.evaluate(cfg)["dem_error"]["flag"] is False


# ======================================================================
# gap-geolocation-7: geolocation against mapped water (synthetic map)
# ======================================================================
def _lakes_and_field(shift_e_m, shift_n_m, seed=0):
    from shapely.affinity import translate
    from shapely.geometry import Point, mapping
    rng = np.random.default_rng(seed)
    mx = 111_320.0 * np.cos(np.radians(38.41)); my = 111_320.0
    lakes, shifted = [], []
    for i in range(10):
        c = Point(X0 + DX * rng.uniform(12, NX - 12), Y0 + DY * rng.uniform(12, NY - 12)).buffer(DX * rng.uniform(2.5, 4))
        lakes.append(mapping(c))
        shifted.append(mapping(translate(c, xoff=shift_e_m / mx, yoff=shift_n_m / my)))
    f = 8
    frac = stack_qc.rasterise_lonlat(shifted, X0, Y0, DX / f, DY / f, (NY * f, NX * f)).reshape(NY, f, NX, f).mean(axis=(1, 3))
    coh = 0.75 - 0.5 * frac + rng.normal(0, 0.03, (NY, NX))
    return lakes, coh


def test_water_edge_search_recovers_a_known_shift():
    lakes, coh = _lakes_and_field(60.0, -40.0)              # cells are ~131 m: sub-cell shifts
    r = stack_qc.geolocation_vs_water(coh, _attrs(), lakes, max_shift_m=150.0)
    assert r["available"]
    assert abs(r["dE_m"] - 60.0) <= 22 and abs(r["dN_m"] + 40.0) <= 22


def test_geolocation_check_in_qa_flags_and_never_fetches(tmp_path):
    lakes, coh = _lakes_and_field(0.0, 0.0)
    cfg = build(tmp_path)
    _h5(cfg.mintpy_dir / "geo" / "geo_avgSpatialCoh.h5", {"coherence": coh.astype(np.float32)}, _attrs())
    g = qa.evaluate(cfg)["geolocation"]                     # no water map, no fetch
    assert g["available"] is False and g["flag"] is None
    ctx = cfg.project_dir / "context"
    ctx.mkdir()
    (ctx / "water.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {}, "geometry": g_} for g_ in lakes]}))
    g = qa.evaluate(cfg)["geolocation"]
    assert g["available"] and g["offset_m"] <= qa.GEOLOCATION_FLAG_M and g["flag"] is False
    lakes2, coh2 = _lakes_and_field(-60.0, 0.0)
    _h5(cfg.mintpy_dir / "geo" / "geo_avgSpatialCoh.h5", {"coherence": coh2.astype(np.float32)}, _attrs())
    g = qa.evaluate(cfg)["geolocation"]
    assert g["dE_m"] < -30 and g["flag"] is True
    assert qa.send_rule({"checks": {}, "geolocation": g})["value"] == "hold for specialist"


# ======================================================================
# usability-6: on_farmland_expected and the send rule
# ======================================================================
def _check(p, exp=None):
    c = {"pass": p}
    if exp is not None:
        c["on_farmland_expected"] = exp
    return c


def test_send_rule_keys_on_which_checks_flag():
    base = {"geolocation": {"flag": False}}
    expected_only = {"checks": {"reference_stability": _check(False), "rate_precision": _check(False),
                                "distributed_scatterer_bias": _check(False), "coverage": _check(True),
                                "network": _check(True), "single_epoch_noise": _check(True)}, **base}
    assert qa.send_rule(expected_only)["value"] == "send with notes"
    clean = {"checks": {k: _check(True) for k in expected_only["checks"]}, **base}
    assert qa.send_rule(clean)["value"] == "send"
    for bad in ("coverage", "network", "single_epoch_noise"):
        d = json.loads(json.dumps(clean)); d["checks"][bad]["pass"] = False
        assert qa.send_rule(d)["value"] == "hold for specialist", bad
    d = json.loads(json.dumps(clean)); d["checks"]["distributed_scatterer_bias"]["pass"] = None
    assert qa.send_rule(d)["value"] == "hold for specialist"   # untested
    # the GNSS tie (it runs after the QA) re-evaluated by the report stage
    assert qa.send_rule(clean, gnss={"background": None, "reason": "no station"})["value"] == "hold for specialist"
    assert qa.send_rule(clean, gnss={"background": {"vertical_mm_yr": -1.0}})["value"] == "send"


def test_every_check_carries_on_farmland_expected_and_verdict_kept(tmp_path):
    r = qa.evaluate(build(tmp_path))
    assert all("on_farmland_expected" in c and c["plain"] for c in r["checks"].values())
    assert r["checks"]["rate_precision"]["on_farmland_expected"] is True
    assert r["checks"]["coverage"]["on_farmland_expected"] is False
    assert r["send_rule"]["value"] in qa.SEND_VALUES and r["send_rule"]["gnss_considered"] is False
    assert r["verdict"] in ("MEETS PRACTICE", "USABLE WITH LIMITATIONS", "REVIEW")
    assert "send rule:" in qa.format_report(r)


# ======================================================================
# contract: far-field sufficiency, DISP stitch check
# ======================================================================
def test_far_field_thin_when_a_report_area_fills_the_box(tmp_path):
    big = "big:38.375,-87.795,38.465,-87.64"                  # nearly the whole grid
    r = qa.evaluate(build(tmp_path, report_areas=("Full Satellite Imaging:full", f"{big}")))
    ff = r["far_field"]
    assert ff["thin"] is True and ff["ratio_wide_to_largest_report_area"] < qa.FAR_MIN_RATIO
    assert r["verdict"] == "REVIEW" and r["send_rule"]["value"] == "hold for specialist"
    small = "small:38.44,-87.79,38.46,-87.77"
    ff2 = qa.evaluate(build(tmp_path / "b", report_areas=(f"{small}",)))["far_field"]
    assert ff2["thin"] is False and ff2["cells"] > qa.FAR_MIN_CELLS


def test_stitch_check_fails_on_each_stitching_fault():
    ok = {"stitch": {"epochs": 100, "skipped_granules": ["20200101"], "boundary_datum": {"flagged": []},
                     "boundary_steps": {"boundary_rate": 0.001, "control_rate": 0.001, "excess": 0.0},
                     "corrections_applied": {"solid_earth_tide": 100}, "epochs_fully_corrected": 100,
                     "unwrap_spikes": {"fraction": 0.001}}}
    assert qa.stitch_check(ok)["pass"] is True
    for mut in ({"lost_anchors": ["20200101"]}, {"epochs_fully_corrected": 60},
                {"boundary_datum": {"flagged": [{"date": "20210101"}]}},
                {"boundary_steps": {"boundary_rate": 0.02, "control_rate": 0.002, "excess": 0.018}},
                {"unwrap_spikes": {"fraction": 0.05}}):
        bad = json.loads(json.dumps(ok)); bad["stitch"].update(mut)
        assert qa.stitch_check(bad)["pass"] is False, mut
    rep = json.loads(json.dumps(ok)); rep["stitch"]["boundary_datum"] = {"flagged": [{"d": 1}], "repaired": True}
    assert qa.stitch_check(rep)["pass"] is True
    assert qa.stitch_check({})["pass"] is None


def test_stitch_check_in_qa_for_a_disp_project(tmp_path):
    cfg = build(tmp_path)
    (cfg.mintpy_dir / "disp_chain.json").write_text(json.dumps(
        {"stitch": {"epochs": 16, "skipped_granules": [], "lost_anchors": ["20180822"]}}))
    r = qa.evaluate(cfg)
    assert r["checks"]["stitch"]["pass"] is False and r["checks"]["stitch"]["on_farmland_expected"] is False
    assert r["send_rule"]["value"] == "hold for specialist"


# ======================================================================
# radar grid -> product grid
# ======================================================================
def test_radar_layers_are_mapped_by_nearest_neighbour(tmp_path):
    cfg = build(tmp_path)
    # a coarse radar grid (half resolution, slightly rotated) with its own lat/lon
    ry, rx = NY // 2, NX // 2
    rr, cc = np.mgrid[0:ry, 0:rx].astype(float)
    lat = Y0 + (2 * rr + 1.5) * DY + 1e-6 * cc            # the centre of product cell (2r+1, 2c+1)
    lon = X0 + (2 * cc + 1.5) * DX
    (cfg.mintpy_dir / "inputs").mkdir()
    _h5(cfg.mintpy_dir / "inputs" / "geometryRadar.h5", {"latitude": lat.astype(np.float32),
                                                         "longitude": lon.astype(np.float32)}, {"FILE_TYPE": "geometry"})
    val = (rr * 1000 + cc).astype(np.float32)
    _h5(cfg.mintpy_dir / "temporalCoherence_mintpy.h5", {"temporalCoherence": val}, {"FILE_TYPE": "temporalCoherence"})
    rg = qa._RadarLayers(cfg, _attrs(), (NY, NX))
    m, src = rg.layer("temporalCoherence_mintpy.h5", "temporalCoherence")
    assert m is not None and "nearest neighbour" in src
    # geo cell (2r+1, 2c+1) sits on radar pixel (r, c)
    for r_, c_ in ((3, 4), (20, 30), (40, 50)):
        assert m[2 * r_ + 1, 2 * c_ + 1] == r_ * 1000 + c_
