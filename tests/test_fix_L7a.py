"""Lane L7a (deliverables.py, gis_products.py) - fixes from the 5 Oct 2026 review.

Every test builds a small synthetic MintPy project (geographic grid, a
radar grid geocoded by nearest neighbour, two coherence classes, a farmland
bias) and checks the OUTCOME on disk: what a GIS, pandas or rasterio reads
back, not what the writer logged.
"""
import json
import re
import zipfile
from pathlib import Path

import h5py
import numpy as np
import pytest

from insar_pipeline.config import PipelineConfig

# geocoded grid: 48 x 60 cells at 0.0005 deg; radar grid 16 x 20, each radar
# pixel copied into a 3 x 3 block of map cells (MintPy's nearest neighbour)
NY, NX, K = 48, 60, 3
X0, Y0, DX, DY = -87.60, 38.47, 0.0005, -0.0005
DATES = [f"{2019 + m // 12}{m % 12 + 1:02d}15" for m in range(0, 24)]     # 2 years, monthly


def _w(path, data: dict, attrs: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        for k, v in data.items():
            f.create_dataset(k, data=v)
        for k, v in attrs.items():
            f.attrs[k] = v


def make_project(tmp_path, display_units="in", **over):
    """A synthetic phase-linked project; returns (cfg, truth dict)."""
    rng = np.random.default_rng(7)
    # bbox covers the inner part of the grid, so the margin is outside the AOI
    bbox = [X0 + 6 * DX, Y0 + (NY - 6) * DY, X0 + (NX - 6) * DX, Y0 + 6 * DY]
    kw = dict(project_dir=tmp_path / "proj", bbox=bbox, start_date="2019-01-01",
              end_date="2020-12-31", backend="miaplpy", display_units=display_units,
              deforming_threshold=0.5)
    kw.update(over)
    cfg = PipelineConfig(**kw)
    cfg.mintpy_dir.mkdir(parents=True, exist_ok=True)
    (cfg.project_dir / "deliverables").mkdir(parents=True, exist_ok=True)

    ry, rx = NY // K, NX // K
    rr, rc = np.indices((ry, rx))
    # radar pixel lat/lon = centre of its 3x3 block (float32, as MintPy stores it)
    rlat = (Y0 + (rr * K + 1.5) * DY).astype(np.float32)
    rlon = (X0 + (rc * K + 1.5) * DX).astype(np.float32)
    # coherence: left half 0.5 (farmland), right half 0.85; one radar column below 0.4
    rtc = np.where(rc < rx // 2, 0.5, 0.85).astype(np.float32)
    rtc[:, 0] = 0.3
    # absolute rate: a class-dependent farmland bias (-15 / -2 mm/yr) + noise,
    # and a sinking bowl of -40 mm/yr relative to its class
    bias = np.where(rtc < 0.6, -15.0, -2.0)
    rvel = bias + rng.normal(0, 1.5, (ry, rx))
    bowl = (rr >= 6) & (rr < 9) & (rc >= 12) & (rc < 15)
    rvel[bowl] -= 40.0
    rvel_m = (rvel / 1000.0).astype(np.float32)

    def to_geo(a):
        return np.repeat(np.repeat(a, K, axis=0), K, axis=1)

    vel, tc = to_geo(rvel_m), to_geo(rtc)
    t = (np.arange(len(DATES)) * 30.4 / 365.25).astype(np.float32)
    ts = (t[:, None, None] * vel[None]
          + rng.normal(0, 0.002, (len(DATES), NY, NX)).astype(np.float32)).astype(np.float32)
    ts[:, 0, 0] = np.nan                                     # one cell with no data at all
    ts -= ts[0][None]                                        # reference date = first epoch
    ts[:, 0, 0] = np.nan
    attrs = {"X_FIRST": X0, "Y_FIRST": Y0, "X_STEP": DX, "Y_STEP": DY, "LENGTH": NY, "WIDTH": NX,
             "X_UNIT": "degrees", "Y_UNIT": "degrees", "REF_Y": 20, "REF_X": 30,
             "REF_LAT": 38.46, "REF_LON": -87.585, "REF_DATE": DATES[0],
             "START_DATE": DATES[0], "END_DATE": DATES[-1], "RANGE_PIXEL_SIZE": 25.63,
             "AZIMUTH_PIXEL_SIZE": 41.7, "ORBIT_DIRECTION": "ASCENDING", "UNIT": "m/year"}
    geo = cfg.mintpy_dir / "geo"
    _w(geo / "geo_velocity.h5", {"velocity": vel, "velocityStd": np.full_like(vel, 0.0005),
                                 "residue": np.full_like(vel, 0.01)}, attrs)
    _w(geo / "geo_temporalCoherence.h5", {"temporalCoherence": tc}, attrs)
    _w(geo / "geo_timeseries_ERA5_ramp_demErr.h5",
       {"timeseries": ts, "date": np.array([d.encode() for d in DATES])}, attrs)
    _w(geo / "geo_geometryRadar.h5",
       {"latitude": to_geo(rlat), "longitude": to_geo(rlon),
        "incidenceAngle": np.full((NY, NX), 38.5, np.float32),
        "azimuthAngle": np.full((NY, NX), -100.0, np.float32),
        "height": np.full((NY, NX), 120.0, np.float32)}, attrs)
    _w(cfg.mintpy_dir / "inputs" / "geometryRadar.h5", {"latitude": rlat, "longitude": rlon},
       {"LENGTH": ry, "WIDTH": rx})
    truth = {"vel_mm": vel * 1000.0, "tc": tc, "ts": ts, "rvel": rvel, "radar_shape": (ry, rx),
             "attrs": attrs, "bowl_geo": to_geo(bowl)}
    return cfg, truth


@pytest.fixture(autouse=True)
def _clear_cache():
    from insar_pipeline import deliverables as dl
    dl.clear_cache()
    yield
    dl.clear_cache()


# ------------------------------------------------- techout-1 / gap-geolocation-2

def test_gpkg_geometry_is_lonlat_inside_the_box(tmp_path):
    """The point geometry is WGS84 lon/lat (it was UTM metres tagged 4326)."""
    import geopandas as gpd
    from pyproj import Transformer
    from insar_pipeline import deliverables as dl
    cfg, _ = make_project(tmp_path)
    out = cfg.project_dir / "deliverables"
    p = dl.export_measurement_points(cfg, out)
    assert p == out / "points_attributes.gpkg" and p.exists()
    gdf = gpd.read_file(p)
    assert gdf.crs.to_epsg() == 4326
    x0, y0, x1, y1 = gdf.total_bounds
    assert X0 <= x0 <= x1 <= X0 + NX * DX and Y0 + NY * DY <= y0 <= y1 <= Y0
    assert np.allclose(gdf.geometry.x, gdf.longitude) and np.allclose(gdf.geometry.y, gdf.latitude)
    # easting/northing keep their metre meaning, in the zone the header names
    assert gdf.easting.between(400_000, 600_000).all()
    hdr = (out / "points_README.txt").read_text()
    epsg = int(re.search(r"EPSG:(326\d\d)", hdr).group(1))
    lon, lat = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True).transform(
        gdf.easting.values, gdf.northing.values)
    assert np.max(np.abs(lat - gdf.latitude.values)) * 111_000 < 1.0
    assert np.max(np.abs(lon - gdf.longitude.values)) * 87_000 < 1.0


def test_point_coords_projected_product():
    """A UTM product keeps its own easting/northing and still gets lon/lat geometry."""
    from insar_pipeline import deliverables as dl
    attrs = {"X_FIRST": 450000.0, "Y_FIRST": 4260000.0, "X_STEP": 30.0, "Y_STEP": -30.0, "EPSG": 32616}
    lon, lat, e, n, epsg = dl._point_coords(attrs, np.array([0, 10]), np.array([0, 5]))
    assert epsg == 32616
    assert np.allclose(e, [450015.0, 450165.0]) and np.allclose(n, [4259985.0, 4259685.0])
    assert -88.2 < lon.min() < lon.max() < -87.0 and 38.0 < lat.min() < lat.max() < 38.6


def test_gpkg_that_fails_its_readback_is_not_delivered(tmp_path):
    """Verify outcomes: UTM metres as geometry are caught and renamed .INVALID."""
    from insar_pipeline import deliverables as dl
    n = 4
    meta = {"lon": np.full(n, 453862.95), "lat": np.full(n, 4262588.43)}   # the old defect
    data = {"pid": np.arange(1, n + 1), "longitude": np.full(n, -87.5), "latitude": np.full(n, 38.5)}
    p = dl._write_points_gpkg(tmp_path / "points_attributes.gpkg", data, ["pid", "longitude", "latitude"],
                              meta, ["# test"], (-88.0, 38.0, -87.0, 39.0))
    assert p is None
    assert not (tmp_path / "points_attributes.gpkg").exists()
    assert (tmp_path / "points_attributes.gpkg.INVALID").exists()


# ------------------------------------------------------------------- techout-7

def test_point_products_are_compact_by_default(tmp_path):
    import geopandas as gpd
    from insar_pipeline import deliverables as dl
    cfg, truth = make_project(tmp_path)
    out = cfg.project_dir / "deliverables"
    dl.export_measurement_points(cfg, out)
    assert not (out / "measurement_points_egms.csv").exists()            # opt-in only
    gdf = gpd.read_file(out / "points_attributes.gpkg")
    assert len(gdf.columns) <= 25 and not any(c.isdigit() for c in gdf.columns)
    # in_aoi follows the cell CENTRE against the bbox
    w, s, e, n = cfg.bbox
    inside = gdf.longitude.between(w, e) & gdf.latitude.between(s, n)
    assert (gdf.in_aoi.astype(bool) == inside).all() and 0 < gdf.in_aoi.sum() < len(gdf)
    # the time series as GeoParquet, joined by pid, mm from each point's first epoch
    tsg = gpd.read_parquet(out / "points_timeseries.parquet")
    # GeoParquet without a "crs" member is OGC:CRS84 = WGS84 lon/lat
    assert tsg.crs is not None and tsg.crs.equals("OGC:CRS84")
    assert len(tsg) == len(gdf) and list(tsg.columns[:1]) == ["pid"]
    assert DATES[0] in tsg.columns and DATES[-1] in tsg.columns
    row = gdf.iloc[100]
    r, c = dl._lalo_to_rc(truth["attrs"], row.latitude, row.longitude)
    ser = truth["ts"][:, r, c]
    got = tsg.loc[tsg.pid == row.pid, DATES[-1]].iloc[0]
    assert got == pytest.approx((ser[-1] - ser[0]) * 1000.0, abs=1e-3)
    assert np.allclose(tsg.geometry.x, gdf.geometry.x)


def test_full_csv_and_kmz_only_when_asked(tmp_path, monkeypatch):
    from insar_pipeline import deliverables as dl
    cfg, _ = make_project(tmp_path)
    out = cfg.project_dir / "deliverables"
    cfg.points_full_csv = True
    dl.export_measurement_points(cfg, out)
    csv = out / "measurement_points_egms.csv"
    assert csv.exists()
    import pandas as pd
    df = pd.read_csv(csv, comment="#")
    assert DATES[-1] in df.columns and "radar_pixel_id" in df.columns
    # rounded, not 17-digit float64 noise
    line = next(l for l in csv.read_text().splitlines() if not l.startswith("#") and l[:1].isdigit())
    assert max(len(tok.split(".")[-1]) for tok in line.split(",") if "." in tok) <= 7
    # the MintPy KMZ is opt-in in export_all
    called = []
    monkeypatch.setattr(dl, "export_kmz_points", lambda *a, **k: called.append(1))
    for name in ("export_velocity", "export_velocity_uncertainty", "export_empirical_uncertainty",
                 "export_cell_noise", "export_signal_to_noise", "export_vertical_velocity",
                 "export_quality", "export_cumulative_displacement", "plot_displacement_series",
                 "plot_selected_points", "export_measurement_points", "export_coverage_curve"):
        monkeypatch.setattr(dl, name, lambda *a, **k: None)
    monkeypatch.setattr(dl, "summary_stats", lambda cfg: {})
    monkeypatch.setattr(dl.provenance, "write", lambda *a, **k: None)
    cfg.gnss_validation = False
    dl.export_all(cfg)
    assert called == []
    cfg.full_timeseries_kmz = True
    dl.export_all(cfg)
    assert called == [1]


# ------------------------------------------------------------------- techout-8

def test_egms_columns_in_mm_whatever_the_display_units(tmp_path):
    import geopandas as gpd
    import pyogrio
    from insar_pipeline import deliverables as dl
    cfg, truth = make_project(tmp_path, display_units="in")
    out = cfg.project_dir / "deliverables"
    dl.export_measurement_points(cfg, out)
    gdf = gpd.read_file(out / "points_attributes.gpkg")
    r, c = dl._lalo_to_rc(truth["attrs"], gdf.latitude.iloc[50], gdf.longitude.iloc[50])
    assert gdf.mean_velocity.iloc[50] == pytest.approx(truth["vel_mm"][r, c], abs=1e-3)     # mm/yr, not in/yr
    assert (gdf.mp_type == "DS").all()
    assert gdf.incidence_angle.notna().all() and gdf.incidence_angle.iloc[0] == pytest.approx(38.5)
    units = dict(dl.egms_field_units(cfg))
    assert units["mean_velocity"].startswith("mm/yr") and units["rmse"].startswith("mm")
    assert "product CRS" not in units["easting"]
    # units and sign travel inside the GeoPackage, not only in a CSV comment
    info = pyogrio.read_info(out / "points_attributes.gpkg", layer="points")
    meta = info.get("layer_metadata") or {}
    assert "mm" in meta.get("UNITS", "") and "satellite" in meta.get("SIGN_CONVENTION", "")
    readme = (out / "points_README.txt").read_text()
    assert "NOT an EGMS product" in readme and "Copernicus Sentinel" in readme


# ----------------------------------------------------- techout-9 / gap-licensing-4

def test_geotiff_carries_tags_units_compression_overviews(tmp_path):
    import rasterio
    from insar_pipeline import deliverables as dl
    attrs = {"X_FIRST": X0, "Y_FIRST": Y0, "X_STEP": 0.0001, "Y_STEP": -0.0001}
    arr = np.random.default_rng(1).normal(size=(600, 600)).astype(np.float32)
    arr[:10] = np.nan
    p = tmp_path / "x.tif"
    dl._write_geotiff(p, arr, attrs, tags={"QUANTITY": "test", "UNITS": "mm/yr", "SIGN_CONVENTION": "neg"},
                      description="a test band")
    with rasterio.open(p) as r:
        assert r.tags()["QUANTITY"] == "test" and r.tags()["TIFFTAG_IMAGEDESCRIPTION"] == "a test band"
        assert r.descriptions[0] == "a test band" and r.units[0] == "mm/yr"
        assert r.compression is not None and r.compression.name.lower() == "deflate"
        assert r.overviews(1) and r.profile.get("tiled")
        assert np.allclose(r.read(1)[10:], arr[10:])


def test_delivered_rasters_say_what_they_are(tmp_path):
    import rasterio
    from insar_pipeline import deliverables as dl
    cfg, truth = make_project(tmp_path)
    out = cfg.project_dir / "deliverables"
    p = dl.export_velocity(cfg, out)
    with rasterio.open(p) as r:
        t = r.tags()
        assert t["UNITS"] == "in/yr" and r.units[0] == "in/yr"
        assert "toward the satellite" in t["SIGN_CONVENTION"]
        assert "official" in t["ROUTE"] and "Copernicus Sentinel" in t["SOURCE"]
        assert "cell centre" in t["REFERENCE_PIXEL"] and t["DATE_RANGE"].startswith(DATES[0])
        assert t["TEMPORAL_COHERENCE_MIN"] == str(cfg.min_temporal_coherence)
    q = dl.export_quality(cfg, out)
    with rasterio.open(q) as r:
        a = r.read(1)
        assert np.nanmin(a) < cfg.min_temporal_coherence            # deliberately unmasked ...
        assert "UNMASKED" in r.descriptions[0] and "none" in r.tags()["TEMPORAL_COHERENCE_MIN"]


# ------------------------------------------------------------------ techout-13

def test_fallback_kmz_is_change_in_display_units(tmp_path, monkeypatch):
    from insar_pipeline import deliverables as dl
    cfg, truth = make_project(tmp_path, display_units="in", kmz_point_step=1)
    # the reference date is NOT the first epoch, so ts[-1] alone is a position
    p = cfg.mintpy_dir / "geo" / "geo_timeseries_ERA5_ramp_demErr.h5"
    with h5py.File(p, "r+") as f:
        ts = f["timeseries"][:]
        f["timeseries"][...] = ts - ts[5][None]
    monkeypatch.setattr(dl.subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(OSError("no mintpy")))
    out = cfg.project_dir / "deliverables"
    k = dl.export_kmz_points(cfg, out)
    kml = zipfile.ZipFile(k).read("doc.kml").decode()
    assert "mm/yr" not in kml and "in/yr" in kml
    pm = re.search(r"Cumulative LOS displacement:</b> (-?[\d.]+) in<br/><b>Location:</b> ([\d.-]+), ([\d.-]+)", kml)
    cum, lat, lon = float(pm.group(1)), float(pm.group(2)), float(pm.group(3))
    r, c = dl._lalo_to_rc(truth["attrs"], lat, lon)
    ser = truth["ts"][:, r, c]
    assert cum == pytest.approx((ser[-1] - ser[0]) * 1000.0 / 25.4, abs=0.01)
    # the point sits on the cell centre
    lon_c, lat_c = dl._rc_to_lonlat(truth["attrs"], r, c)
    assert abs(lat - lat_c) < 1e-5 and abs(lon - lon_c) < 1e-5


def test_gis_products_keep_zero_and_sign(tmp_path):
    import rasterio
    from rasterio.transform import from_origin
    from insar_pipeline import gis_products as gp
    merged = tmp_path / "merged"
    merged.mkdir()
    amp = np.ones((4, 4), np.float32); amp[0, :] = 0.0                 # outside the footprint
    ph = np.zeros((4, 4), np.float32); ph[3, 3] = 2 * np.pi           # one fringe of range increase
    with rasterio.open(merged / "filt_topophase.unw.geo", "w", driver="GTiff", height=4, width=4, count=2,
                       dtype="float32", crs="EPSG:4326", transform=from_origin(-87.6, 38.5, 0.001, 0.001)) as d:
        d.write(amp, 1); d.write(ph, 2)
    out = gp.make_los_displacement(merged)
    with rasterio.open(out) as r:
        a = r.read(1)
        assert np.isnan(r.nodata)
        assert np.isnan(a[0]).all()                     # no data stays hidden
        assert a[1, 1] == 0.0                           # a genuine zero is a measurement
        assert a[3, 3] < 0                              # range increase = away = negative
    assert "TOWARD" in gp.make_los_displacement.__doc__


# ------------------------------------------------------------------ techout-11

def test_header_posting_comes_from_the_product(tmp_path):
    from insar_pipeline import deliverables as dl
    cfg, truth = make_project(tmp_path)
    out = cfg.project_dir / "deliverables"
    dl.export_measurement_points(cfg, out, include_timeseries=False)
    readme = (out / "points_README.txt").read_text()
    assert "Posting 15 m; independent resolution cell ~41 m" not in readme
    assert "15 m map cell" not in readme
    px = dl._pixel_metres(truth["attrs"])                          # ~50 m on this synthetic grid
    info = dl.resolution_info(cfg, truth["attrs"])
    assert info["text"] in readme and f"{px:.0f} m" in info["text"]
    assert info["posting_m"] == pytest.approx(px, abs=1.0)
    # a 30 m product says 30 m
    info30 = dl.resolution_info(cfg, {"X_FIRST": 450000.0, "Y_FIRST": 4260000.0, "X_STEP": 30.0,
                                      "Y_STEP": -30.0, "EPSG": 32616})
    assert "30 m" in info30["text"]


# -------------------------------------------------------------------- tests-10

def test_los_up_or_raise_never_invents_a_value(tmp_path):
    from insar_pipeline import deliverables as dl
    cfg, _ = make_project(tmp_path)
    up = dl.los_up_or_raise(cfg, (NY, NX))
    assert up.shape == (NY, NX) and np.allclose(up, np.cos(np.radians(38.5)))
    with pytest.raises(ValueError):
        dl.los_up_or_raise(cfg, (NY + 1, NX))
    (cfg.mintpy_dir / "geo" / "geo_geometryRadar.h5").unlink()
    with pytest.raises(FileNotFoundError):
        dl.los_up_or_raise(cfg)
    assert dl.export_vertical_velocity(cfg, cfg.project_dir / "deliverables") is None


def test_los_up_fill_uses_this_grid(tmp_path, monkeypatch):
    from insar_pipeline import deliverables as dl
    cfg, _ = make_project(tmp_path)
    m = np.full((4, 4), 0.70, np.float32); m[0, 0] = np.nan
    monkeypatch.setattr(dl, "geometry_field", lambda cfg, name: m)
    assert np.isnan(dl.los_up_or_raise(cfg)[0, 0])
    assert dl.los_up_or_raise(cfg, fill=True)[0, 0] == pytest.approx(0.70)      # not 0.78


# ------------------------------------------------------------------ empirical-3

def test_one_cell_noise_everywhere(tmp_path):
    import rasterio
    from insar_pipeline import deliverables as dl
    cfg, _ = make_project(tmp_path)
    out = cfg.project_dir / "deliverables"
    dl.export_cell_noise(cfg, out)
    dl.export_signal_to_noise(cfg, out)
    model = json.loads((out / "uncertainty_model.json").read_text())
    assert "key" in model and model["by_class"]
    for r in model["by_class"]:
        assert r["cell_null_mm_yr"] == pytest.approx(max(r["empirical_mm_yr"],
                                                         r["farmland_cell_spread_mm_yr"] or 0), abs=1e-3)
    far, rel, emp, _, null = dl._empirical_context(cfg)
    with rasterio.open(out / "los_velocity_cell_noise_1sigma_in_per_yr.tif") as r:
        noise_mm = r.read(1) * 25.4
    with rasterio.open(out / "rate_signal_to_noise.tif") as r:
        snr = r.read(1)
    ok = np.isfinite(snr) & np.isfinite(noise_mm)
    assert ok.sum() > 1000
    # |relative| / the delivered noise raster IS the delivered SNR raster
    assert np.allclose(np.abs(rel[ok]) / noise_mm[ok], snr[ok], rtol=1e-4)
    # and the per-class table in the json is what the raster holds
    from insar_pipeline.significance import coherence_class
    tc = h5py.File(cfg.mintpy_dir / "geo" / "geo_temporalCoherence.h5", "r")["temporalCoherence"][:]
    k = coherence_class(tc)
    for key, v in model["cell_null_by_class"].items():
        sel = ok & (k == int(key))
        if sel.any():
            assert np.allclose(noise_mm[sel], v, atol=1e-3)


def test_points_significance_matches_the_snr_raster(tmp_path):
    import geopandas as gpd
    import rasterio
    from insar_pipeline import deliverables as dl
    cfg, truth = make_project(tmp_path)
    out = cfg.project_dir / "deliverables"
    dl.export_signal_to_noise(cfg, out)
    dl.export_measurement_points(cfg, out, include_timeseries=False)
    gdf = gpd.read_file(out / "points_attributes.gpkg")
    with rasterio.open(out / "rate_signal_to_noise.tif") as r:
        snr = r.read(1)
    rc = [dl._lalo_to_rc(truth["attrs"], la, lo) for la, lo in zip(gdf.latitude, gdf.longitude)]
    s = np.array([snr[r, c] for r, c in rc])
    fin = np.isfinite(s)
    assert ((s[fin] > 1.96) == gdf.significant_95.values[fin].astype(bool)).all()
    assert np.allclose(np.abs(gdf.mean_velocity_relative.values[fin]) / gdf.cell_noise_1sigma.values[fin],
                       s[fin], rtol=1e-4)


# ------------------------------------------------------ empirical-4 / techout-10

def test_summary_deforming_area_is_relative_and_quality_is_the_qa_verdict(tmp_path):
    import os
    import time
    from insar_pipeline import deliverables as dl
    cfg, truth = make_project(tmp_path)
    S = dl.summary_stats(cfg)
    raw, rel = S["raw_los_rate_below_threshold"], S["deforming_area"]
    # the -15 mm/yr farmland bias crosses -0.5 in/yr (-12.7 mm/yr) in absolute
    # terms; relative to farmland of its class only the bowl does
    assert raw["n_pixels"] > 5 * rel["n_pixels"]
    bowl_in_aoi = int((truth["bowl_geo"] & dl._box_mask(truth["attrs"], cfg.bbox, (NY, NX))).sum())
    assert abs(rel["n_pixels"] - bowl_in_aoi) <= 0.1 * bowl_in_aoi
    assert rel["far_field_share_beyond_threshold_percent"] <= rel["share_of_measured_aoi_percent"]
    assert "velocity_median" not in S and "raw_los_velocity_vs_reference_median" in S
    assert S["coverage_grade"] in ("Good", "Fair", "Poor") and S["data_quality"] is None
    assert S["independent_radar_measurements"] < S["total_measured_points"] / 5
    # QA written after the product: refresh_summary puts its verdict in data_quality
    out = cfg.project_dir / "deliverables"
    (out / "summary.json").write_text(json.dumps(S))
    qa = cfg.project_dir / "quality_assessment.json"
    qa.write_text(json.dumps({"verdict": "REVIEW", "flagged": 3, "checks": {"a": {}, "b": {}}}))
    later = time.time() + 5
    os.utime(qa, (later, later))
    assert dl.refresh_summary(cfg)["data_quality"] == "REVIEW"
    assert json.loads((out / "summary.json").read_text())["data_quality"] == "REVIEW"
    # a QA file older than the product belongs to another run
    old = time.time() - 3600
    os.utime(qa, (old, old))
    assert dl.refresh_summary(cfg)["data_quality"] is None


def test_old_summary_files_still_read(tmp_path):
    from insar_pipeline import deliverables as dl
    old = {"data_quality": "Good", "velocity_median": -0.483, "velocity_min": -3.0, "velocity_max": 1.0,
           "deforming_area": {"n_pixels": 2142698, "threshold": "faster than -0.5 in/yr",
                              "median_velocity": -0.706}}
    (tmp_path / "summary.json").write_text(json.dumps(old))
    S = dl.read_summary(tmp_path)
    assert S["raw_los_velocity_vs_reference_median"] == -0.483
    assert S["coverage_grade"] == "Good" and S["data_quality"] is None
    assert S["raw_los_rate_below_threshold"]["n_pixels"] == 2142698 and "deforming_area" not in S
    assert dl.read_summary(tmp_path / "nothing_here") == {}


# ---------------------------------------------------------- gap-geolocation-3

def test_radar_pixel_id_traces_the_geocoding_exactly(tmp_path):
    from insar_pipeline import deliverables as dl
    cfg, truth = make_project(tmp_path)
    rows, cols = np.indices((NY, NX))
    rows, cols = rows.ravel(), cols.ravel()
    ids, basis = dl.radar_pixel_ids(cfg, rows, cols, (NY, NX))
    ry, rx = truth["radar_shape"]
    assert (ids == (rows // K) * rx + cols // K).all()
    assert np.unique(ids).size == ry * rx and "lookup" in basis
    # every geocoded value equals its radar pixel's
    rv = truth["rvel"].ravel()
    assert np.allclose(truth["vel_mm"].ravel(), rv[ids] , atol=1e-3)
    # a product made natively in map coordinates: each cell is its own measurement
    (cfg.mintpy_dir / "inputs" / "geometryRadar.h5").unlink()
    ids2, basis2 = dl.radar_pixel_ids(cfg, rows, cols, (NY, NX))
    assert np.unique(ids2).size == NY * NX and "product-grid" in basis2


# ---------------------------------------------------------- gap-geolocation-9

def test_no_corner_coordinates_left_in_deliverables():
    src = (Path(__file__).resolve().parents[1] / "deliverables.py").read_text()
    for pat in (r'X_FIRST"\]\) \+ np\.arange', r'x0 \+ c \* dx', r'y0 \+ r \* dy',
                r'lat = y0 \+ r \* dy', r'np\.arange\(nx\) \* dx'):
        assert not re.search(pat, src), pat


def test_reference_label_is_the_cell_centre(tmp_path):
    from insar_pipeline import deliverables as dl
    cfg, truth = make_project(tmp_path)
    lab = dl.reference_label(cfg)
    lat_c = Y0 + (20 + 0.5) * DY
    lon_c = X0 + (30 + 0.5) * DX
    assert lab.startswith(f"{lat_c:.6f},{lon_c:.6f}") and "radar position 38.460000,-87.585000" in lab
    # in-the-box tests use centres: the first column's centre is half a cell east of X_FIRST
    m = dl._box_mask(truth["attrs"], [X0 + 0.25 * DX, Y0 + NY * DY, X0 + NX * DX, Y0], (NY, NX))
    assert m[:, 0].all()
    m = dl._box_mask(truth["attrs"], [X0 + 0.75 * DX, Y0 + NY * DY, X0 + NX * DX, Y0], (NY, NX))
    assert not m[:, 0].any()


# ------------------------------------------------------- timeseries_file

def test_timeseries_file_follows_run_products_and_checks_it(tmp_path):
    from insar_pipeline import deliverables as dl
    cfg, truth = make_project(tmp_path)
    geo = cfg.mintpy_dir / "geo"
    good = geo / "geo_timeseries_ERA5_ramp_demErr.h5"
    # a more-corrected-looking name that run_products.json does NOT name
    other = geo / "geo_timeseries_SET_ERA5_ramp_demErr.h5"
    a = dict(truth["attrs"], REF_Y=3)                            # referenced elsewhere
    _w(other, {"timeseries": truth["ts"], "date": np.array([d.encode() for d in DATES])}, a)
    rp = cfg.mintpy_dir / "run_products.json"
    rp.write_text(json.dumps({"timeseries_geo": str(good)}))
    assert dl.timeseries_file(cfg) == good                      # the run record wins over tokens
    # the record names a series that does not match the velocity: fall back, then check
    rp.write_text(json.dumps({"timeseries_geo": str(other)}))
    with pytest.raises(ValueError):                              # tokens pick `other`, which mismatches
        dl.timeseries_file(cfg)
    other.unlink()
    assert dl.timeseries_file(cfg) == good
    rp.unlink()
    assert dl.timeseries_file(cfg) == good                      # legacy runs still work
