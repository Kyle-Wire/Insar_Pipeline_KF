"""
26 Oct 2026: the NISAR route gets the full product set (Kyle: Site E's
NISAR comparison needs "all of the files and maps that we would normally get").

- The products stage ran the QA, figures and client report only on the
  Sentinel-1 routes; it runs them on nisar too (pipeline.py).
- Every route said "Sentinel-1" and carried the Copernicus Sentinel notice:
  the NISAR run's GeoTIFFs and points_README credited Sentinel data it does
  not contain. The satellite now comes from config.sensor_name and the credit
  is NISAR's (NASA-ISRO, via ASF DAAC).
- On routes already on the map grid (HyP3, NISAR) the extra layers were
  "used where they are", but every reader opens geo/geo_<name>: none reached
  the products. They are copied to their geo_ names; on NISAR MintPy's
  temporal coherence (1.0 on a chain) is not, since it tests nothing.
"""
import sys
import types
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from insar_pipeline import config, credits, deliverables, pipeline   # noqa: E402
from insar_pipeline import time_series as ts                         # noqa: E402

GEO = dict(FILE_TYPE="temporalCoherence", LENGTH="6", WIDTH="5", Y_FIRST="40.0", X_FIRST="-100.0",
           Y_STEP="-0.0007", X_STEP="0.0009")


def _cfg(tmp_path, backend):
    P = tmp_path / backend
    (P / "mintpy").mkdir(parents=True)
    return types.SimpleNamespace(backend=backend, project_dir=P, mintpy_dir=P / "mintpy", dem_dir=P / "dem",
                                 dem_path=None, start_date="2025-10-01", end_date="2026-10-26",
                                 mintpy_tropo_method="raider", weather_model="HRRR", gnss_validation=False,
                                 gnss_tie_radius_km=0)


def test_the_satellite_is_named_from_the_route():
    assert config.sensor_name("nisar") == "NISAR"
    assert config.sensor_name(types.SimpleNamespace(backend="miaplpy")) == "Sentinel-1"
    assert config.sensor_name("disp") == "Sentinel-1" and config.sensor_name("") == "Sentinel-1"


def test_nisar_is_credited_as_nisar_not_copernicus_sentinel(tmp_path):
    n = _cfg(tmp_path, "nisar")
    c = credits.credits(n)
    assert c[0]["source"] == "NISAR" and "NASA-ISRO" in c[0]["text"] and "PROVISIONAL" in c[0]["text"]
    assert not any(x["source"] in ("Copernicus Sentinel-1", "ASF DAAC") for x in c)
    assert any(x["source"] == "NOAA HRRR" for x in c)
    line = deliverables.source_line(n, {"START_DATE": "20251030", "END_DATE": "20260826"})
    assert line.startswith("NISAR data") and "Copernicus Sentinel" not in line
    s1 = deliverables.source_line(_cfg(tmp_path, "miaplpy"), {"START_DATE": "20160801", "END_DATE": "20261009"})
    assert "Copernicus Sentinel" in s1                               # unchanged on Sentinel-1


def test_extra_layers_on_a_geocoded_route_reach_their_geo_names(tmp_path, monkeypatch):
    monkeypatch.setattr(ts, "write_disturbed_layer", lambda cfg: None)
    for backend, want in (("nisar", ["geo_demErr.h5"]),
                          ("hyp3", ["geo_demErr.h5", "geo_temporalCoherence_mintpy.h5"])):
        cfg = _cfg(tmp_path, backend)
        M = cfg.mintpy_dir
        for name, ds, v in (("temporalCoherence.h5", "temporalCoherence", 0.6),
                            ("temporalCoherence_mintpy.h5", "temporalCoherence", 1.0), ("demErr.h5", "dem", 6.3)):
            with h5py.File(M / name, "w") as f:
                f[ds] = np.full((6, 5), v, np.float32)
                f.attrs.update(GEO)
        out = ts.geocode_extra_layers(cfg, missing_only=True)
        assert sorted(p.name for p in out) == want, backend
        with h5py.File(M / "geo" / "geo_demErr.h5", "r") as f:
            assert float(f["dem"][0, 0]) == np.float32(6.3) and "description" in f.attrs
    assert ts.extra_layers_for(types.SimpleNamespace(backend="nisar")) == ("demErr.h5", "disturbed.h5")
    assert ts.extra_layers_for(types.SimpleNamespace(backend="miaplpy")) == ts.EXTRA_LAYERS


def test_the_products_stage_lists_only_the_layers_the_route_makes(tmp_path, monkeypatch):
    """No 'still not on the product grid' note for NISAR's meaningless
    temporal-coherence layer."""
    cfg = _cfg(tmp_path, "nisar")
    G = cfg.mintpy_dir / "geo"
    G.mkdir()
    for n in ("geo_demErr.h5", "geo_disturbed.h5"):
        (G / n).write_bytes(b"")
    rec = pipeline.geocode_missing_layers(cfg)
    assert rec["present"] == ["geo_demErr.h5", "geo_disturbed.h5"] and rec["missing_after"] == []
    assert pipeline.extra_layers_note(cfg, rec) is None
