"""The phase-linked NISAR route (nisar_pl.py, PLAN_nisar_phase_linked Phase 3,
28 Oct 2026): each kept GUNW crop's unwrapped phase, coherence and components
become dolphin's for the same pair (JPL's kept as *GUNW), idempotently; a
swapped crop is never reused by the plain GUNW route; the config checks; the
resolution sentence. Synthetic data; dolphin itself is not run."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from insar_pipeline import nisar_backend as nb, nisar_pl as npl

U = npl.U + "HH/"


def _crop(path, ref, sec, lat, lon):
    import h5py
    with h5py.File(path, "w") as h:
        h[U + "unwrappedPhase"] = np.full((lat.size, lon.size), 9.0, np.float32)
        h[U + "coherenceMagnitude"] = np.full((lat.size, lon.size), 0.3, np.float32)
        h[U + "connectedComponents"] = np.ones((lat.size, lon.size), np.uint16)
        h[U + "ionospherePhaseScreen"] = np.full((lat.size, lon.size), 0.5, np.float32)
        h[U + "xCoordinates"], h[U + "yCoordinates"] = lon, lat
        h.attrs.update({"reference": ref, "secondary": sec, "lattice": "L", "granule": "G", "crop_version": nb.CROP_VERSION})


def _tif(path, a, lon, lat, dtype="float32"):
    import rasterio
    from rasterio.transform import from_origin
    path.parent.mkdir(parents=True, exist_ok=True)
    dx, dy = lon[1] - lon[0], lat[0] - lat[1]
    with rasterio.open(path, "w", driver="GTiff", height=lat.size, width=lon.size, count=1, dtype=dtype,
                       crs="EPSG:4326", transform=from_origin(lon[0] - dx / 2, lat[0] + dy / 2, dx, dy)) as d:
        d.write(a.astype(dtype), 1)


@pytest.fixture
def stack(tmp_path):
    lon = -104.9 + 0.000173 * np.arange(30)
    lat = 38.98 - 0.000135 * np.arange(20)
    work = tmp_path / "dolphin"
    ph = np.add.outer(np.zeros(20), np.linspace(0, 3, 30))
    cc = np.ones((20, 30))
    cc[:, :2] = 0                                                     # snaphu left two columns out
    _tif(work / "unwrapped" / "20251030_20251111.unw.tif", ph, lon, lat)
    _tif(work / "unwrapped" / "20251030_20251111.unw.conncomp.tif", cc, lon, lat, "uint16")
    _tif(work / "interferograms" / "20251030_20251111.int.cor.tif", np.full((20, 30), 0.8), lon, lat)
    _tif(work / "interferograms" / "temporal_coherence_average_20251030_20251111.tif", np.full((20, 30), 0.95), lon, lat)
    (work / "nisar_pl_run.json").write_text(json.dumps({"returncode": 0, "window": 11, "strides": [3, 3]}))
    crop = tmp_path / "crop.h5"
    _crop(crop, "20251030", "20251111", lat, lon)
    return crop, work, ph


def test_swap_puts_dolphins_phase_in_and_keeps_jpls(stack, tmp_path):
    import h5py
    crop, work, ph = stack
    r = npl.swap_into_crops([crop], work)
    assert r[0]["swapped"] is True and r[0]["pair"] == "20251030_20251111"
    with h5py.File(crop) as h:
        new = h[U + "unwrappedPhase"][()]
        assert np.allclose(new[:, 2:], ph[:, 2:], atol=1e-5)
        assert np.isnan(new[:, :2]).all()                             # connected component 0: no phase
        assert np.allclose(h[U + "coherenceMagnitude"][()], 0.8)
        assert (h[U + "unwrappedPhaseGUNW"][()] == 9.0).all()          # JPL's kept
        assert np.allclose(h[U + "ionospherePhaseScreen"][()], 0.5)    # JPL's ionosphere untouched
        assert h.attrs["phase_source"] == "dolphin"
    assert npl.swap_into_crops([crop], work)[0]["swapped"] == "already"
    out = npl.write_temporal_coherence(work, crop, tmp_path / npl.TC_FILE)
    import rasterio
    with rasterio.open(out) as s:
        assert np.allclose(s.read(1), 0.95)


def test_a_reversed_pair_takes_the_negated_interferogram(stack, tmp_path):
    import h5py
    crop, work, ph = stack
    for f in (work / "unwrapped").glob("20251030_20251111*"):
        f.rename(f.with_name(f.name.replace("20251030_20251111", "20251111_20251030")))
    (work / "interferograms" / "20251030_20251111.int.cor.tif").rename(
        work / "interferograms" / "20251111_20251030.int.cor.tif")
    npl.swap_into_crops([crop], work)
    with h5py.File(crop) as h:
        assert np.allclose(h[U + "unwrappedPhase"][()][:, 2:], -ph[:, 2:], atol=1e-5)


def test_a_swapped_crop_is_never_the_gunw_routes(stack):
    crop, work, _ = stack
    lat = SimpleNamespace(signature=lambda: "L")
    assert nb._reusable(crop, "G", lat)
    npl.swap_into_crops([crop], work)
    assert not nb._reusable(crop, "G", lat)                           # the GUNW route reads the granule again
    assert nb._reusable(crop, "G", lat, phase_linked=True)


def test_current_run_and_stride(tmp_path):
    w = tmp_path / "d"
    (w / "slcs").mkdir(parents=True)
    (w / "unwrapped").mkdir()
    for d in ("20251030", "20251111"):
        (w / "slcs" / f"{d}.slc.tif").write_bytes(b"")
    (w / "nisar_pl_run.json").write_text(json.dumps({"returncode": 0, "window": 11, "strides": [3, 3]}))
    assert npl.current_run(w, 11, 3, ["20251030", "20251111"])
    assert not npl.current_run(w, 21, 3, ["20251030", "20251111"])
    assert not npl.current_run(w, 11, 3, ["20251030", "20251111", "20251123"])
    assert npl.stride_for(SimpleNamespace(nisar_posting_m=15.0)) == 3
    with pytest.raises(ValueError, match="multiple"):
        npl.stride_for(SimpleNamespace(nisar_posting_m=12.0))


def test_config_checks_and_resolution_sentence(tmp_path):
    from insar_pipeline.config import ConfigError, load_config
    from insar_pipeline import resolution
    base = (f"project_dir: {tmp_path / 'p'}\nbackend: nisar\nbbox: [-104.9, 38.87, -104.78, 38.96]\n"
            "start_date: '2025-10-01'\nend_date: '2026-10-26'\ntrack: 19\norbit_direction: ascending\n"
            "nisar_phase_linked: true\n")
    p = tmp_path / "c.yaml"
    p.write_text(base + "nisar_pl_window: 10\n")
    with pytest.raises(ConfigError, match="odd|reference_lalo"):
        load_config(str(p))
    p.write_text(base + "reference_lalo: '38.9477,-104.7948'\n")
    cfg = load_config(str(p))
    info = resolution.effective_resolution(cfg, {})
    assert info["window"] == [11, 11] and "55 m" in info["text"] and "5 m pixel" in info["text"]
