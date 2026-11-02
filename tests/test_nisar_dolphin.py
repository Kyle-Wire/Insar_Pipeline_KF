"""The phase-linked NISAR route's dolphin driver (nisar_dolphin.py,
PLAN_nisar_phase_linked Phase 2, 28 Oct 2026): GSLC crops become dated
GeoTIFFs (the empty date left out), and the Gate 2 measures and window rule
work as written before any result. Synthetic data; dolphin itself is not run."""
import json

import numpy as np
import pytest

from insar_pipeline import nisar_dolphin as nd

EPSG = 32613
X0, Y0, PX = 510000.0, 4310000.0, 15.0


def _tif(path, a, dtype="float32"):
    import rasterio
    from rasterio.transform import from_origin
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", driver="GTiff", height=a.shape[0], width=a.shape[1], count=1, dtype=dtype,
                       crs=f"EPSG:{EPSG}", transform=from_origin(X0, Y0, PX, PX)) as d:
        d.write(a.astype(dtype), 1)


def test_crops_become_dated_geotiffs_without_the_empty_date(tmp_path):
    import h5py
    import rasterio
    x, y = 500000.0 + 5 * np.arange(20), 4300000.0 - 5 * np.arange(16)
    reads = []
    for d in ("20251030", "20251111", "20260907"):
        p = tmp_path / "g" / "crops" / f"{d}.h5"
        p.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(p, "w") as f:
            f["slc"] = np.full((16, 20), 1 + 2j, np.complex64)
            f["mask"] = np.ones((16, 20), np.uint8)
            f["x"], f["y"] = x, y
            f.attrs.update({"epsg": EPSG, "centerFrequency": 1.239e9})
        reads.append({"date": d, "file": str(p)})
    (tmp_path / "g" / "gslc_stack.json").write_text(json.dumps({"reads": reads, "empty": [{"date": "20260907"}]}))
    inp = nd.write_inputs(tmp_path / "g", tmp_path / "w")
    assert [p.split("/")[-1] for p in inp["files"]] == ["20251030.slc.tif", "20251111.slc.tif"]
    assert inp["left_out"] == ["20260907"] and inp["wavelength_m"] == pytest.approx(0.24196, abs=1e-5)
    with rasterio.open(inp["files"][0]) as s:
        assert s.dtypes[0] == "complex64" and s.crs.to_epsg() == EPSG
        assert s.transform.c == pytest.approx(500000.0 - 2.5) and s.read(1)[0, 0] == 1 + 2j


def _work(tmp_path, n=100, rate_mm=5.0, bad_ifg=None):
    """A dolphin-like work folder: 6 dates, a uniform LOS rate, coherence 0.9
    (0.2 in a band), nearest-3 unwrapped interferograms; bad_ifg gets one
    whole cycle added over half the scene."""
    w = tmp_path / "w"
    dates = ["20251030", "20251111", "20251123", "20251205", "20251217", "20251229"]
    from insar_pipeline.significance import years
    t = np.array(years(dates)) - years(dates)[0]
    tc = np.full((n, n), 0.9)
    tc[:, :3] = 0.2
    _tif(w / "interferograms" / "temporal_coherence_average_x.tif", tc)
    rng = np.random.default_rng(0)
    field = rng.normal(0, 0.5, (n, n))                                   # mm/yr cell-to-cell spread
    for i, d in enumerate(dates[1:], 1):
        _tif(w / "timeseries" / f"{dates[0]}_{d}.tif", (rate_mm + field) * t[i] / 1000.0)
    lam = 0.2419
    for i in range(len(dates)):
        for j in range(i + 1, min(i + 4, len(dates))):
            ph = -4 * np.pi / lam * ((rate_mm + field) * (t[j] - t[i]) / 1000.0)
            if bad_ifg == (i, j):
                ph = ph.copy()
                ph[:, n // 2:] += 2 * np.pi
            _tif(w / "unwrapped" / f"{dates[i]}_{dates[j]}.unw.tif", ph)
    (w / "nisar_pl_run.json").write_text(json.dumps({"window": 11, "strides": [3, 3],
                                                     "reference_rowcol_output": [5, 5], "seconds": 1}))
    # the region: a 5 x 5 cell square in the middle, as lon/lat
    from pyproj import Transformer
    tr = Transformer.from_crs(EPSG, 4326, always_xy=True)
    xs = [X0 + 48 * PX, X0 + 53 * PX]                                   # 1.5 km scene: ground beyond 500 m
    ys = [Y0 - 48 * PX, Y0 - 53 * PX]
    ring = [list(tr.transform(xs[a], ys[b])) for a, b in ((0, 0), (1, 0), (1, 1), (0, 1), (0, 0))]
    g = tmp_path / "region.geojson"
    g.write_text(json.dumps({"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {}, "geometry": {"type": "Polygon", "coordinates": [ring]}}]}))
    return w, g


def test_gate2_measures(tmp_path):
    w, g = _work(tmp_path)
    r = nd.gate2(w, g)
    assert r["footprint_cells"] == 25 and r["posting_m"] == PX
    assert r["temporal_coherence_median_ground"] == pytest.approx(0.9) and r["pass_temporal_coherence"]
    assert r["closure_share"] == pytest.approx(1.0) and r["pass_closure"] and r["triplets"] == 4
    assert r["rate_ground_robust_sd_mm_yr"] == pytest.approx(0.5, rel=0.15)        # the planted cell spread
    assert r["reliable_ground_cells"] < r["comparison_ground_cells"]              # the 0.2 band is left out
    assert json.loads((w / "gate2.json").read_text())["closure_share"] == r["closure_share"]


def test_gate2_counts_an_unwrapping_error(tmp_path):
    w, g = _work(tmp_path, bad_ifg=(1, 2))                    # in two of the four triplets (as i+1 and as i)
    r = nd.gate2(w, g)
    assert r["closure_share"] < 0.8 and not r["pass_closure"]


def test_window_rule():
    g = lambda s, c=0.95: {"spread_1sigma_mm_yr": s, "closure_share": c}            # noqa: E731
    assert nd.window_rule(g(5.0), g(4.5))["chosen_window"] == 11                   # 1.11 <= 1.25
    assert nd.window_rule(g(6.0), g(4.5))["chosen_window"] == 21                   # 1.33 > 1.25
    assert nd.window_rule(g(4.0, c=0.7), g(4.5))["chosen_window"] == 21            # closure below 80%
