"""12 Oct 2026: on the NISAR route MintPy's temporal coherence is 1.0 by
construction (a 12-day chain has no redundant pairs), so the products'
reliability masks screened nothing. apply_nisar_pair_coherence writes the
mean pair coherence into temporalCoherence.h5 and the mask."""
from types import SimpleNamespace

import h5py
import numpy as np
import rasterio
from rasterio.transform import from_origin

from insar_pipeline import time_series as ts


def _project(tmp_path):
    P = tmp_path / "proj"
    (P / "mintpy").mkdir(parents=True)
    (P / "nisar_gunw").mkdir()
    attrs = dict(X_FIRST="150.0", Y_FIRST="-34.0", X_STEP="0.001", Y_STEP="-0.001", LENGTH="10", WIDTH="12")
    with h5py.File(P / "mintpy" / "temporalCoherence.h5", "w") as f:
        f["temporalCoherence"] = np.ones((10, 12), np.float32)
        f.attrs.update(attrs)
    with h5py.File(P / "mintpy" / "maskTempCoh.h5", "w") as f:
        f["mask"] = np.ones((10, 12), bool)
        f.attrs.update(attrs)
    gate = np.full((10, 12), 0.7, np.float32)
    gate[:, :6] = 0.2                                   # left half decorrelated
    with rasterio.open(P / "nisar_gunw" / "coherence_gate.tif", "w", driver="GTiff", height=10, width=12,
                       count=2, dtype="float32", crs="EPSG:4326",
                       transform=from_origin(150.0, -34.0, 0.001, 0.001)) as dst:
        dst.write(gate, 1)
        dst.write((gate >= 0.4).astype("float32"), 2)
    return SimpleNamespace(project_dir=P, mintpy_dir=P / "mintpy", min_temporal_coherence=0.4, backend="nisar")


def test_pair_coherence_replaces_the_trivial_chain_coherence(tmp_path):
    cfg = _project(tmp_path)
    assert ts.apply_nisar_pair_coherence(cfg)
    with h5py.File(cfg.mintpy_dir / "temporalCoherence.h5") as f:
        tc = f["temporalCoherence"][:]
        assert f.attrs["temporalCoherenceSource"] == ts.NISAR_TC_SOURCE
    assert np.allclose(tc[:, :6], 0.2) and np.allclose(tc[:, 6:], 0.7)
    with h5py.File(cfg.mintpy_dir / "maskTempCoh.h5") as f:
        m = f["mask"][:]
    assert not m[:, :6].any() and m[:, 6:].all()
    with h5py.File(cfg.mintpy_dir / "temporalCoherence_mintpy.h5") as f:
        assert np.all(f["temporalCoherence"][:] == 1.0)       # MintPy's own kept aside


def test_idempotent_and_safe_without_the_gate(tmp_path):
    cfg = _project(tmp_path)
    assert ts.apply_nisar_pair_coherence(cfg)
    assert ts.apply_nisar_pair_coherence(cfg)                 # second call: no change, no error
    with h5py.File(cfg.mintpy_dir / "temporalCoherence_mintpy.h5") as f:
        assert np.all(f["temporalCoherence"][:] == 1.0)       # not overwritten by the adopted layer
    (cfg.project_dir / "nisar_gunw" / "coherence_gate.tif").unlink()
    cfg2 = _project(tmp_path / "b")
    (cfg2.project_dir / "nisar_gunw" / "coherence_gate.tif").unlink()
    assert ts.apply_nisar_pair_coherence(cfg2) is False
