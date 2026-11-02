"""Lane L9c of the 5 Oct 2026 review: technical figures and panel_history.

clientout-14   captions route-aware and computed from the data
clientout-15   mining-year history from mined_year_geojson for any site
generality-10  no Gibson file name or MINES dict; names from the layer
generality-7   figures without a mine layer and with display_units mm
clientout-17   _rings('') no longer raises
tests-7        summary table None-safe (reference-stability fit missing)
plus the _Context contract: second gate (min_unwrap_consistency) and
significance.placement_field for the far_wide placements.

Every project here is synthetic (a ~60 x 70 geocoded grid) and runs in a
temporary folder.
"""
import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from insar_pipeline.config import PipelineConfig

NY, NX = 60, 70
X0, Y0 = -87.80, 38.47
DX, DY = 0.28 / NX, -0.27 / NY
BBOX = [-87.8, 38.2, -87.52, 38.47]
MINE = (-87.70, 38.30, -87.60, 38.40)
BAD = ("Site A", "Gibson", "contractor", "12-day", "snaphu can only unwrap")


def _square(w, s, e, n):
    return [[w, s], [e, s], [e, n], [w, n], [w, s]]


def _fc(*feats):
    return {"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": pr, "geometry": {"type": "Polygon", "coordinates": [_square(*box)]}}
        for box, pr in feats]}


def _dates(start="2018-01-05", n=46, step=24):
    import datetime as dt
    d0 = dt.date.fromisoformat(start)
    return [(d0 + dt.timedelta(days=step * i)).strftime("%Y%m%d") for i in range(n)]


def _grid():
    lon = X0 + (np.arange(NX)[None, :] + 0.5) * DX + 0 * np.arange(NY)[:, None]
    lat = Y0 + (np.arange(NY)[:, None] + 0.5) * DY + 0 * np.arange(NX)[None, :]
    return lat, lon


def _attrs():
    return {"X_FIRST": str(X0), "Y_FIRST": str(Y0), "X_STEP": str(DX), "Y_STEP": str(DY),
            "LENGTH": str(NY), "WIDTH": str(NX), "REF_Y": "3", "REF_X": "3",
            "REF_LAT": str(Y0 + 3.5 * DY), "REF_LON": str(X0 + 3.5 * DX), "UNIT": "m/year"}


def _h5(path, name, data, attrs, dates=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f.create_dataset(name, data=data)
        if dates is not None:
            f.create_dataset("date", data=np.array([d.encode() for d in dates]))
        for k, v in attrs.items():
            f.attrs[k] = v


def _tif(path, arr):
    import rasterio
    from rasterio.transform import from_origin
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", driver="GTiff", height=NY, width=NX, count=1, dtype="float32",
                       crs="EPSG:4326", transform=from_origin(X0, Y0, DX, -DY), nodata=np.nan) as dst:
        dst.write(arr.astype(np.float32), 1)


def make_project(tmp_path, *, steps=None, rate=None, outside_offset=0.0, tc=None, **over) -> PipelineConfig:
    """A geocoded MintPy product, QA and deliverables for a 60 x 70 grid.
    `steps`: [(mask, YYYYMMDD, mm)] planted steps; `rate`: m/yr field."""
    kw = dict(project_dir=tmp_path / "proj", bbox=BBOX, start_date="2018-01-01", end_date="2020-12-31",
              backend="miaplpy", acceptance_far_field_buffer_m=500.0, graph_n_points=30)
    kw.update(over)
    cfg = PipelineConfig(**kw)
    geo = cfg.mintpy_dir / "geo"
    dates = _dates()
    t = np.array([(np.datetime64(f"{d[:4]}-{d[4:6]}-{d[6:]}") - np.datetime64("2018-01-05")).astype(int) / 365.25 for d in dates])
    rng = np.random.default_rng(3)
    rate = np.zeros((NY, NX)) if rate is None else rate
    lat, lon = _grid()
    aoi = (lon >= cfg.bbox[0]) & (lon <= cfg.bbox[2]) & (lat >= cfg.bbox[1]) & (lat <= cfg.bbox[3])
    rate = rate + np.where(aoi, 0.0, outside_offset / 1000.0)
    ts = np.zeros((len(dates), NY, NX), np.float32)
    for i, d in enumerate(dates):
        ts[i] = rate * t[i] + rng.normal(0, 0.0015, (NY, NX))
        for m, when, mm in steps or ():
            if d >= when:
                ts[i][m] += mm / 1000.0
    vel = np.zeros((NY, NX), np.float32)
    for r in range(NY):
        vel[r] = np.polyfit(t, ts[:, r, :], 1)[0]
    A = _attrs()
    _h5(geo / "geo_velocity.h5", "velocity", vel, A)
    _h5(geo / "geo_timeseries_ERA5_ramp_demErr.h5", "timeseries", ts, A, dates)
    _h5(geo / "geo_maskTempCoh.h5", "mask", np.ones((NY, NX), bool), A)
    tc = np.full((NY, NX), 0.95, np.float32) if tc is None else tc
    _h5(geo / "geo_temporalCoherence.h5", "temporalCoherence", tc, A)
    D = cfg.project_dir / "deliverables"
    unit = "in" if str(cfg.display_units).startswith("in") else "mm"
    div = 25.4 if unit == "in" else 1.0
    _tif(D / f"los_velocity_{unit}_per_year.tif", vel * 1000.0 / div)
    _tif(D / "quality_temporal_coherence.tif", tc)
    _tif(D / f"los_cumulative_displacement_{unit}_{dates[0]}_{dates[-1]}.tif", (ts[-1] - ts[0]) * 1000.0 / div)
    (D / "figures").mkdir(parents=True, exist_ok=True)
    (D / "summary.json").write_text(json.dumps({
        "date_range": f"{dates[0]} to {dates[-1]}", "n_acquisitions": len(dates), "span_days": 1080,
        "total_measured_points": NY * NX, "track": 92, "orbit_direction": "ASCENDING", "wavelength_m": 0.05546576}))
    (cfg.project_dir / "quality_assessment.json").write_text(json.dumps({
        "far_field": {"class_mean_rate_mm_yr": {"0.9-1.0": 0.0}, "endpoint_total_mm": 0.0},
        "reference_pixel": {"lat": Y0 + 3.5 * DY, "lon": X0 + 3.5 * DX},
        "checks": {"coverage": {"threshold": 0.4, "aoi_percent_at_threshold": 100.0, "aoi_percent_at_0_7": 100.0,
                                "coherence_source": "MiaplPy phase linking, average (mini-stack) temporal coherence - not comparable with MintPy's"},
                   "single_epoch_noise": {"robust_mm": 5.0, "rms_mm": 6.0, "pass": True},
                   "reference_stability": {"value_mm_yr": 0.1, "pass": True},
                   "network": {"pairs_formed": 10, "pairs_inverted": 10, "mean_connections_per_date": 3.0, "connected": True, "pass": True}},
        "findings": {}, "passed": 3, "flagged": 0, "verdict": "MEETS PRACTICE"}))
    return cfg


def _mine_mask():
    lat, lon = _grid()
    w, s, e, n = MINE
    return (lon > w) & (lon < e) & (lat > s) & (lat < n)


def _write(path: Path, obj) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj))
    return str(path)


# ------------------------------------------------------------------ _rings (clientout-17, generality-7)

def test_rings_without_a_configured_layer_is_empty(tmp_path):
    from insar_pipeline import figures
    # the old reader raised IsADirectoryError on Path('')
    assert figures._rings("") == []
    assert figures._rings(None) == []
    gj = _write(tmp_path / "m.geojson", _fc((MINE, {})))
    assert len(figures._rings(gj)) == 1
    with pytest.raises(FileNotFoundError):         # a configured file that is missing still fails loudly
        figures._rings(str(tmp_path / "missing.geojson"))


def test_make_figures_with_no_mine_layer_and_mm_units(tmp_path, monkeypatch):
    """A screening project with no mine outlines and display_units mm: every
    map draws, and the figures read the mm rasters, not a stale inch one."""
    from insar_pipeline import figures
    import rasterio
    cfg = make_project(tmp_path, display_units="mm", deramp_exclude_geojson="")
    D = cfg.project_dir / "deliverables"
    # a stale inch raster from before the unit switch, with other values
    _tif(D / "los_velocity_in_per_year.tif", np.full((NY, NX), 9.0))
    _tif(D / "los_cumulative_displacement_in_20170101_20171231.tif", np.full((NY, NX), 9.0))
    opened = []
    real_open = rasterio.open
    monkeypatch.setattr(rasterio, "open", lambda p, *a, **k: (opened.append(Path(p).name), real_open(p, *a, **k))[1])
    figures.make_figures(cfg)
    man = {m["file"] for m in json.loads((D / "figures" / "figures.json").read_text())}
    assert {"velocity_relative.png", "cumulative_relative.png", "temporal_coherence.png", "summary_table.png"} <= man
    assert all((D / "figures" / f).exists() for f in man)
    assert "los_velocity_mm_per_year.tif" in opened
    assert not any("_in_" in n for n in opened), opened


def test_missing_rate_raster_in_configured_unit_says_so(tmp_path):
    from insar_pipeline import figures
    cfg = make_project(tmp_path)                    # inch rasters
    cfg.display_units = "mm"
    with pytest.raises(FileNotFoundError, match="display_units"):
        figures.make_figures(cfg)


# ------------------------------------------------------------------ summary table (tests-7)

@pytest.mark.parametrize("checks", [
    {"reference_stability": {"value_mm_yr": None, "pass": None}},      # the fit was not possible
    {},                                                                  # no checks at all
    {"reference_stability": {"value_mm_yr": None}, "rate_precision": {}, "single_epoch_noise": {"robust_mm": None}},
])
def test_summary_table_survives_unmeasured_quantities(tmp_path, checks):
    from insar_pipeline import figures
    cfg = make_project(tmp_path)
    out = cfg.project_dir / "deliverables" / "figures"
    Q = {"checks": checks, "findings": {"workings_relative_to_far_field": {"rate_mm_yr": None, "significance": {"testable": False}}}}
    entry = figures.summary_table(cfg, Q, out)          # f"{None:+.2f}" raised TypeError before
    assert entry["file"] == "summary_table.png" and (out / "summary_table.png").exists()
    assert "p < 0.05" not in entry["caption"] and "41 m" not in entry["caption"]


def test_num_formatter_reads_missing_as_dash():
    from insar_pipeline import figures
    assert figures._num(None, "+.2f", " mm/yr") == "—"
    assert figures._num(float("nan"), ".1f") == "—"
    assert figures._num(-3.114, "+.2f", " mm/yr") == "-3.11 mm/yr"


# ------------------------------------------------------------------ captions (clientout-14)

def _stack(cfg, pairs, L=NY, W=NX):
    """A tiny merged/interferograms stack on the product grid."""
    S = cfg.stack_dir
    G = S / "merged" / "geom_reference"; G.mkdir(parents=True, exist_ok=True)
    lat, lon = _grid()
    np.asarray(lat, np.float64).tofile(G / "lat.rdr"); np.asarray(lon, np.float64).tofile(G / "lon.rdr")
    (S / "crop_window.json").write_text(json.dumps({"r0": 0, "r1": L, "c0": 0, "c1": W, "length": L, "width": W}))
    rng = np.random.default_rng(0)
    hist = {}
    for k, p in enumerate(pairs):
        d = S / "merged" / "interferograms" / p; d.mkdir(parents=True, exist_ok=True)
        (np.exp(1j * rng.uniform(-3, 3, (L, W)))).astype(np.complex64).tofile(d / "filt_fine.int")
        np.full((L, W), 0.6, np.float32).tofile(d / "filt_fine.cor")
        np.zeros((L, 2, W), np.float32).tofile(d / "filt_fine.unw")
        cc = np.ones((L, W), np.uint8); cc[:, : W // 4] = 0
        cc.tofile(d / "filt_fine.unw.conncomp")
        hist[p] = 1.0 + k
    (S / "unwrap_history.json").write_text(json.dumps(hist))
    # the network: ifgramStack + timeseries for dates/bperp
    dates = sorted({x for p in pairs for x in p.split("_")})
    _h5(cfg.mintpy_dir / "inputs" / "ifgramStack.h5", "dropIfgram", np.ones(len(pairs), bool), {})
    with h5py.File(cfg.mintpy_dir / "inputs" / "ifgramStack.h5", "a") as f:
        f.create_dataset("date", data=np.array([[a.encode(), b.encode()] for a, b in (p.split("_") for p in pairs)]))
    with h5py.File(cfg.mintpy_dir / "timeseries.h5", "w") as f:
        f.create_dataset("date", data=np.array([d.encode() for d in dates]))
        f.create_dataset("bperp", data=np.zeros(len(dates), np.float32))


def test_captions_follow_the_route_and_the_data(tmp_path):
    from insar_pipeline import figures
    cfg = make_project(tmp_path, deramp_exclude_geojson=_write(tmp_path / "mine.geojson", _fc((MINE, {}))),
                       stack_reference_date="20180105")
    # Site B's case: the "summer" pair is 48 days long, not 12
    _stack(cfg, ["20180105_20180129", "20180609_20180727", "20180129_20180222"])
    figures.make_figures(cfg)
    man = json.loads((cfg.project_dir / "deliverables" / "figures" / "figures.json").read_text())
    by = {m["file"]: m for m in man}
    text = json.dumps(man)
    for word in BAD:
        assert word not in text, word
    summer = by["summer_pair.png"]
    assert "48 days" in summer["title"] and "2018-06-09 → 2018-07-27" in summer["title"]
    assert "24 days" in by["winter_pair.png"]["title"]
    # the unwrapped share is measured: all but the first quarter of columns has a component
    assert f"{1 - (NX // 4) / NX:.0%} of the imaged crop" in summer["caption"]
    # phase-linked route: the layer is MiaplPy's, not MintPy's network coherence
    tcap = by["temporal_coherence.png"]["caption"]
    assert "MiaplPy" in tcap and "MintPy's temporal coherence for every cell" not in tcap
    assert "phase-linked interferogram" in by["winter_pair.png"]["caption"]
    # posting from the raster (~350-400 m on this toy grid), not a fixed 15 m
    assert "Each 15 m cell" not in by["velocity_relative.png"]["caption"]
    # network: measured connections, not "three nearest neighbours"
    assert "three nearest neighbours" not in by["network.png"]["caption"]
    assert "kept pairs" in by["network.png"]["caption"]


def test_classic_route_coherence_caption_is_mintpys(tmp_path):
    from insar_pipeline import figures
    cfg = make_project(tmp_path, backend="stack")
    q = json.loads((cfg.project_dir / "quality_assessment.json").read_text())
    q["checks"]["coverage"].pop("coherence_source")
    (cfg.project_dir / "quality_assessment.json").write_text(json.dumps(q))
    figures.make_figures(cfg)
    man = {m["file"]: m for m in json.loads((cfg.project_dir / "deliverables" / "figures" / "figures.json").read_text())}
    assert "MintPy" in man["temporal_coherence.png"]["caption"]
    assert "MiaplPy" not in man["temporal_coherence.png"]["caption"]


def test_season_and_days_from_pair():
    from insar_pipeline import figures
    assert figures._pair_days("20160609_20160727") == 48
    assert figures._season("20160609_20160727") == "summer"
    assert figures._season("20210219_20210303") == "winter"


# ------------------------------------------------------------------ mining-year history (clientout-15, generality-10)

def _dated_project(tmp_path, **over):
    """Two dated pits in custom fields; pit A (mined 2019) settles 10 mm in
    July 2019, pit B (mined 2018) does not move."""
    a_box = (-87.70, 38.30, -87.66, 38.40)
    b_box = (-87.64, 38.30, -87.60, 38.40)
    lat, lon = _grid()
    a = (lon > a_box[0]) & (lon < a_box[2]) & (lat > a_box[1]) & (lat < a_box[3])
    dated = _write(tmp_path / "pits.geojson", _fc((a_box, {"PIT": "North pit", "YR_MINED": "2019-03-01"}),
                                                    (b_box, {"PIT": "South pit", "YR_MINED": 2018})))
    mine = _write(tmp_path / "mine.geojson", _fc((MINE, {})))
    cfg = make_project(tmp_path, steps=[(a, "20190701", -10.0)], deramp_exclude_geojson=mine, **over)
    cfg.mined_year_geojson = dated                # set as attributes (L4a adds the fields)
    cfg.mined_year_field = "YR_MINED"
    cfg.mined_name_field = "PIT"
    return cfg


def test_panel_history_runs_for_any_site_from_the_configured_layer(tmp_path):
    from insar_pipeline import panel_history as ph
    cfg = _dated_project(tmp_path)
    res = ph.run(cfg)
    assert set(res["panels"]) == {"North pit 2019", "South pit 2018"}
    assert set(res["mines"]) == {"North pit", "South pit"}
    north = res["panels"]["North pit 2019"]
    # the planted 10 mm drop between the half-years either side of 2019
    assert -13 < north["step_across_mining_year_mm"] < -7
    assert abs(res["panels"]["South pit 2018"].get("step_across_mining_year_mm") or 0) < 3
    # change by years since mining: North pit's 2019 annual mean drops
    k0 = [a for a in north["annual_change_mm"] if a["years_since_mining"] == 0]
    assert k0 and k0[0]["change_mm"] < -2
    assert {r["years_since_mining"] for r in res["by_years_since_mining"]} >= {0, 1}
    out = cfg.project_dir / "deliverables"
    assert (out / "figures" / "panel_history.png").exists()
    saved = json.loads((out / "panel_history.json").read_text())
    assert saved["layer"]["year_field"] == "YR_MINED"
    man = {m["file"]: m for m in json.loads((out / "figures" / "figures.json").read_text())}
    assert "CMIS" not in man["panel_history.png"]["caption"] and "pits.geojson" in man["panel_history.png"]["caption"]


def test_no_hard_coded_gibson_names_or_file():
    import insar_pipeline.panel_history as ph
    src = Path(ph.__file__).read_text()
    assert not hasattr(ph, "MINES")
    assert "cmis_gibson_panels" not in src and "801777" not in src


def test_year_and_name_parsing():
    from insar_pipeline import panel_history as ph
    assert ph._year_of(2016) == 2016 and ph._year_of(2016.0) == 2016
    assert ph._year_of("2016-05-01") == 2016 and ph._year_of("mined 2009") == 2009
    assert ph._year_of(1462060800000) == 2016          # ArcGIS epoch milliseconds
    assert ph._year_of(None) is None and ph._year_of("n/a") is None and ph._year_of(12345) is None
    assert ph._name_of(802033) == "Mine 802033" and ph._name_of("Site D") == "Site D"


def test_skipped_history_retires_the_previous_one(tmp_path):
    """A project without a dated layer must not keep drawing summary-table
    rows from an earlier panel_history.json."""
    from insar_pipeline import panel_history as ph
    cfg = _dated_project(tmp_path)
    ph.run(cfg)
    cfg.mined_year_geojson = ""
    assert ph.run(cfg) == {}
    out = cfg.project_dir / "deliverables"
    saved = json.loads((out / "panel_history.json").read_text())
    assert saved["skipped"] is True and saved["mines"] == {} and "mined_year_geojson" in saved["reason"]
    assert not (out / "figures" / "panel_history.png").exists()
    man = json.loads((out / "figures" / "figures.json").read_text()) if (out / "figures" / "figures.json").exists() else []
    assert "panel_history.png" not in {m["file"] for m in man}


def test_wrong_year_field_is_reported_with_the_available_properties(tmp_path):
    from insar_pipeline import panel_history as ph
    cfg = _dated_project(tmp_path)
    cfg.mined_year_field = "YEAR_RECLAIMED"
    feats, reason = ph._load_dated(cfg)
    assert feats == [] and "YR_MINED" in reason and "PIT" in reason


def test_site_caption_uses_the_configured_cluster_size(tmp_path):
    from insar_pipeline import panel_history as ph
    cfg = _dated_project(tmp_path, focus_sites_lalo=("Client parcel: 38.33,-87.68,38.37,-87.66",), graph_n_points=30)
    ph.run(cfg)
    man = {m["file"]: m for m in json.loads((cfg.project_dir / "deliverables" / "figures" / "figures.json").read_text())}
    cap = man["site_history.png"]["caption"]
    assert "30-point cluster" in cap and "Client parcel" in cap
    assert "126" not in cap and "contractor" not in cap
    sh = json.loads((cfg.project_dir / "deliverables" / "site_history.json").read_text())
    assert set(sh["mines"]) == {"North pit", "South pit"}
    assert sh["sites"]["Client parcel"]["overlapping_panels"][0]["mine"] == "North pit"


# ------------------------------------------------------------------ _Context: second gate

def test_second_gate_off_by_default_and_applied_when_set(tmp_path):
    from insar_pipeline import panel_history as ph
    cfg = make_project(tmp_path)
    ctx = ph._Context(cfg)
    assert ctx.second_gate["applied"] is False and ctx.good.all()
    um = np.full((NY, NX), 0.9, np.float32); um[:, : NX // 2] = 0.3
    _h5(cfg.mintpy_dir / "geo" / "geo_temporalCoherence_mintpy.h5", "temporalCoherence", um, _attrs())
    cfg.min_unwrap_consistency = 0.5
    ctx = ph._Context(cfg)
    assert ctx.second_gate["applied"] is True
    assert not ctx.good[:, : NX // 2].any() and ctx.good[:, NX // 2:].all()
    assert not ctx.far[:, : NX // 2].any()              # the comparison ground is gated too
    assert ctx.second_gate["removed_percent"] == pytest.approx(50.0, abs=1)


def test_second_gate_requested_without_the_layer_raises(tmp_path):
    from insar_pipeline import panel_history as ph
    cfg = make_project(tmp_path)
    cfg.min_unwrap_consistency = 0.5
    with pytest.raises(FileNotFoundError, match="min_unwrap_consistency"):
        ph._Context(cfg)


# ------------------------------------------------------------------ _Context: placements and totals

def test_null_is_placed_on_significance_placement_field(tmp_path, monkeypatch):
    """_Context hands assess the field placement_field returns (stats-4 wiring)."""
    from insar_pipeline import panel_history as ph
    from insar_pipeline import significance as sg
    cfg = make_project(tmp_path, bbox=[-87.76, 38.24, -87.56, 38.43])
    marker = {}

    def fake(vel, tc, far, far_wide, aoi):
        r, _ = sg.class_matched_field(vel, tc, far)
        r = r.copy(); marker["aoi_cells"] = int(aoi.sum()); marker["field"] = r
        return r
    monkeypatch.setattr(sg, "placement_field", fake, raising=False)
    seen = {}
    real = sg.assess
    monkeypatch.setattr(sg, "assess", lambda field, far, region, *a, **k: (seen.setdefault("field", field), real(field, far, region, *a, **k))[1])
    ctx = ph._Context(cfg)
    assert ctx.r_null is marker["field"] and 0 < marker["aoi_cells"] < NY * NX
    m = np.zeros((NY, NX), bool); m[20:30, 20:30] = True
    series, far, _ = ctx.series({"x": m})
    ctx.summarise(series["x"], m, far)
    assert seen["field"] is marker["field"]


def test_outside_ring_offset_does_not_move_the_null(tmp_path):
    """+2 mm/yr on every far cell outside the AOI box leaves the null of an
    in-box region where it was (needs L1's placement_field)."""
    from insar_pipeline import panel_history as ph
    from insar_pipeline import significance as sg
    if not hasattr(sg, "placement_field"):
        pytest.skip("significance.placement_field (lane L1) not present yet")
    box = [-87.76, 38.24, -87.56, 38.43]
    m = np.zeros((NY, NX), bool); m[25:31, 25:31] = True
    meds, sds = [], []
    for off in (0.0, 2.0):
        cfg = make_project(tmp_path / f"o{off}", bbox=box, outside_offset=off)
        ctx = ph._Context(cfg)
        series, far, _ = ctx.series({"x": m})
        sig = ctx.summarise(series["x"], m, far)["significance"]
        meds.append(sig.get("null_median_mm_yr")); sds.append(sig.get("farmland_spread_1sigma_mm_yr"))
    # measured before the fix: median 0.00 -> 0.36, spread 0.04 -> 0.89 mm/yr
    assert meds[0] is not None and abs(meds[1] - meds[0]) < 0.1
    assert sds[1] < 1.5 * sds[0] + 0.02


def test_totals_are_the_measured_change_with_rate_x_span_kept(tmp_path, monkeypatch):
    from insar_pipeline import panel_history as ph
    from insar_pipeline import significance as sg
    cfg = make_project(tmp_path)
    ctx = ph._Context(cfg)
    m = np.zeros((NY, NX), bool); m[20:30, 20:30] = True
    series, far, _ = ctx.series({"x": m})
    monkeypatch.setattr(sg, "record_change", lambda t, y, min_years=2.0: {
        "change_mm": -42.0, "linear_change_mm": -40.0, "nonlinear": True, "sigma_mm": 1.0, "method": "piecewise_linear"},
        raising=False)
    s = ctx.summarise(series["x"], m, far, significance=False)
    assert s["total_mm"] == -42.0 and s["total_method"] == "piecewise_linear" and s["nonlinear"] is True
    fit = sg.fit_rate(ctx.t, np.array([np.nan if v is None else v for v in s["series_mm"]]))
    assert s["total_at_average_rate_mm"] == pytest.approx(round(sg.trend_total(fit), 1), abs=0.11)
    # record_change has no answer (too short / too few epochs): rate x span
    monkeypatch.setattr(sg, "record_change", lambda t, y, min_years=2.0: None)
    s2 = ctx.summarise(series["x"], m, far, significance=False)
    assert s2["total_method"] == "rate_x_span" and s2["total_mm"] == s2["total_at_average_rate_mm"]
