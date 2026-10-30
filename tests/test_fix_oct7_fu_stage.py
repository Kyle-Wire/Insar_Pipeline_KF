"""Trust review of 7 Oct 2026, follow-up F5 (stage group): the products
stage geocodes the extra layers that are missing, before the deliverables.

A project whose MintPy run finished before time_series.geocode_extra_layers
existed (Site A, 27 Sep 2026) has no geo/geo_temporalCoherence_mintpy.h5,
geo_demErr.h5 or geo_disturbed.h5, so its point table had no second-gate
columns and the rate-corrected "reshaped since the elevation model" test (E1)
could not run. 'Products only' now makes the missing ones with the existing
helper; a layer already there is never geocoded again; what was done is
logged; a failure is logged and the products go on without them.

Each test fails on the code before this round (backups/pre_followup_20261007:
no geocoding step in run_from, and geocode_extra_layers had no missing_only).
Synthetic projects only; MintPy's geocode.py is replaced by a fake that writes
what it would.
"""
import logging
import os
import time
import types
from pathlib import Path

import h5py
import numpy as np
import pytest

from insar_pipeline import notify, pipeline, procs, progress, runs
from insar_pipeline import time_series as ts

RSHAPE = (12, 10)                     # radar grid
OLD = time.time() - 86400             # "geocoded yesterday"
GEO_NAMES = ["geo_temporalCoherence_mintpy.h5", "geo_demErr.h5", "geo_disturbed.h5"]


@pytest.fixture(autouse=True)
def _quiet(monkeypatch, tmp_path):
    monkeypatch.setenv("INSAR_NOTIFY", "off")
    monkeypatch.setattr(notify, "LOGFILE", tmp_path / "notifications.log")
    monkeypatch.setattr(runs, "REGISTRY", tmp_path / "reg.json")
    yield
    procs.clear_stopping()


def _w(path, datasets: dict, attrs: dict | None = None, mtime=None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        for k, v in datasets.items():
            f.create_dataset(k, data=v)
        for k, v in (attrs or {}).items():
            f.attrs[k] = v
    if mtime is not None:
        os.utime(path, (mtime, mtime))
    return path


def _project(tmp_path, backend="miaplpy"):
    """A finished phase-linked project as Site A's: the radar-grid layers
    in mintpy/, the products in geo/, none of the three extra layers there."""
    from insar_pipeline.config import load_config
    f = tmp_path / "c.yaml"
    f.write_text(f"project_dir: {tmp_path / 'proj'}\nbbox: [-87.7, 38.2, -87.6, 38.3]\n"
                 "start_date: '2020-01-01'\nend_date: '2021-01-01'\n")
    cfg = load_config(str(f))
    cfg.backend = backend
    cfg.project_dir.mkdir(parents=True, exist_ok=True)
    M = cfg.mintpy_dir
    tc = np.full(RSHAPE, 0.8, np.float32)
    _w(M / "temporalCoherence.h5", {"temporalCoherence": tc}, {"FILE_TYPE": "temporalCoherence"}, OLD)
    _w(M / "temporalCoherence_mintpy.h5", {"temporalCoherence": tc}, {"FILE_TYPE": "temporalCoherence"}, OLD)
    _w(M / "demErr.h5", {"dem": np.ones(RSHAPE, np.float32)}, {"FILE_TYPE": "dem"}, OLD)
    _w(M / "inputs" / "geometryRadar.h5", {"latitude": np.zeros(RSHAPE)}, {}, OLD)
    _w(M / "geo" / "geo_velocity.h5", {"velocity": np.zeros(RSHAPE, np.float32)}, {}, OLD)
    return cfg


def _fakes(monkeypatch, cfg, calls, fail=None):
    """MintPy's geocode.py (geo_<name>, float) and the disturbed layer's
    writer, both recording what they were asked to do."""
    M = cfg.mintpy_dir

    def fake_disturbed(cfg_):
        calls.append(("disturbed",))
        return _w(M / "disturbed.h5", {"disturbed": np.full(RSHAPE, 3.0, np.float32)}, {"usable": "no"})

    def fake_geocode(cfg_, files):
        calls.append(("geocode", [Path(p).name for p in files]))
        if fail:
            raise fail
        for src in files:
            with h5py.File(src, "r") as s, h5py.File(M / "geo" / f"geo_{Path(src).name}", "w") as g:
                for k in s:
                    g.create_dataset(k, data=s[k][:].astype(np.float32))
                g.attrs.update(dict(s.attrs))
        return types.SimpleNamespace(returncode=0)
    monkeypatch.setattr(ts, "write_disturbed_layer", fake_disturbed)
    monkeypatch.setattr(ts, "_run_geocode", fake_geocode)


def _stub_products(monkeypatch, cfg, seen):
    """The deliverables record which extra layers were on disk when they ran."""
    def export(cfg_):
        seen.append(sorted(p.name for p in (cfg.mintpy_dir / "geo").glob("geo_*.h5") if p.name in GEO_NAMES))
        return cfg.project_dir / "deliverables"

    def ok(name):
        def f(cfg_):
            progress.emit(cfg_.project_dir, name, "done")
            return True
        return f
    monkeypatch.setattr(pipeline, "export_all", export)
    for name in ("quality_assessment", "figures", "client_report"):
        monkeypatch.setattr(pipeline, f"run_{name}", ok(name))


# ------------------------------------------------------------------ the products run
def test_products_run_geocodes_the_missing_layers_before_the_deliverables(tmp_path, monkeypatch, caplog):
    """Site A's case: none of the three on disk. 'Products only' makes
    them before export_all, so the deliverables (point table, extra GeoTIFFs)
    and then the report see them, and says so in the log."""
    cfg = _project(tmp_path)
    calls, seen = [], []
    _fakes(monkeypatch, cfg, calls)
    _stub_products(monkeypatch, cfg, seen)
    with caplog.at_level(logging.INFO):
        rc = pipeline.run_from(cfg, "products", no_acceptance=True, watchdog=False, notify=False)
    assert rc == 0
    assert seen == [sorted(GEO_NAMES)]                     # on disk when the deliverables ran
    assert calls == [("disturbed",), ("geocode", ["temporalCoherence_mintpy.h5", "demErr.h5", "disturbed.h5"])]
    msg = "\n".join(r.getMessage() for r in caplog.records)
    assert "geocoded now geo_temporalCoherence_mintpy.h5, geo_demErr.h5, geo_disturbed.h5" in msg
    with h5py.File(cfg.mintpy_dir / "geo" / "geo_temporalCoherence_mintpy.h5", "r") as f:
        assert "MintPy network-inversion" in f.attrs["temporalCoherenceSource"]   # tagged as the helper does


def test_layers_already_on_the_grid_are_never_geocoded_again(tmp_path, monkeypatch, caplog):
    """geo_demErr.h5 there (e.g. from the MintPy run), the other two missing:
    only those two are made; the present file is not rewritten, re-tagged or
    touched. With all three there nothing runs at all - not even the
    disturbed layer's recomputation."""
    cfg = _project(tmp_path)
    G = cfg.mintpy_dir / "geo"
    present = _w(G / "geo_demErr.h5", {"dem": np.full(RSHAPE, 7.0, np.float32)}, {"mark": "kept"}, OLD)
    before = present.read_bytes()
    calls = []
    _fakes(monkeypatch, cfg, calls)
    with caplog.at_level(logging.INFO):
        rec = pipeline.geocode_missing_layers(cfg)
    assert calls == [("disturbed",), ("geocode", ["temporalCoherence_mintpy.h5", "disturbed.h5"])]
    assert rec["present"] == ["geo_demErr.h5"]
    assert rec["made"] == ["geo_temporalCoherence_mintpy.h5", "geo_disturbed.h5"] and rec["missing_after"] == []
    assert present.read_bytes() == before and os.path.getmtime(present) == pytest.approx(OLD)
    msg = "\n".join(r.getMessage() for r in caplog.records)
    assert "geo_demErr.h5 already there, left as they are" in msg

    calls.clear()
    stamps = {n: os.path.getmtime(G / n) for n in GEO_NAMES}
    rec = pipeline.geocode_missing_layers(cfg)
    assert calls == [] and rec["made"] == [] and rec["present"] == GEO_NAMES
    assert {n: os.path.getmtime(G / n) for n in GEO_NAMES} == stamps


def test_the_helper_leaves_present_layers_alone_only_when_asked(tmp_path, monkeypatch):
    """geocode_extra_layers(missing_only=True) skips a layer whose geo_ copy
    exists (and the disturbed layer's recomputation when geo_disturbed.h5 is
    there); the MintPy run's default call still makes all three."""
    cfg = _project(tmp_path)
    G = cfg.mintpy_dir / "geo"
    _w(G / "geo_disturbed.h5", {"disturbed": np.zeros(RSHAPE, np.int16)}, {}, OLD)
    calls = []
    _fakes(monkeypatch, cfg, calls)
    out = ts.geocode_extra_layers(cfg, missing_only=True)
    assert calls == [("geocode", ["temporalCoherence_mintpy.h5", "demErr.h5"])]       # no disturbed rewrite
    assert sorted(p.name for p in out) == ["geo_demErr.h5", "geo_temporalCoherence_mintpy.h5"]
    calls.clear()
    out = ts.geocode_extra_layers(cfg)
    assert calls == [("disturbed",), ("geocode", ["temporalCoherence_mintpy.h5", "demErr.h5", "disturbed.h5"])]
    assert sorted(p.name for p in out) == sorted(GEO_NAMES)


@pytest.mark.parametrize("fail", [RuntimeError("geocode.py: pyresample not importable"),
                                  OSError(28, "No space left on device")])
def test_a_failed_geocode_is_logged_and_the_products_go_on(tmp_path, monkeypatch, caplog, fail):
    """As before the step existed, the deliverables run without the layers;
    the run is not failed for a diagnostic layer, but the log says what
    happened."""
    cfg = _project(tmp_path)
    calls, seen = [], []
    _fakes(monkeypatch, cfg, calls, fail=fail)
    _stub_products(monkeypatch, cfg, seen)
    with caplog.at_level(logging.INFO):
        rc = pipeline.run_from(cfg, "products", no_acceptance=True, watchdog=False, notify=False)
    assert rc == 0 and seen == [[]]
    assert ("geocode", ["temporalCoherence_mintpy.h5", "demErr.h5", "disturbed.h5"]) in calls
    errs = [r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR]
    assert any("Geocoding the missing extra layers failed" in m and "the products go on without them" in m
               for m in errs), errs
    assert any("still not on disk geo_temporalCoherence_mintpy.h5, geo_demErr.h5, geo_disturbed.h5" in r.getMessage()
               for r in caplog.records)


def test_a_stop_during_the_step_still_stops_the_run(tmp_path, monkeypatch):
    """Only an Exception is swallowed: a Stop (KeyboardInterrupt / SystemExit
    from the teardown) ends the run as a stop, not as 'went on without them'."""
    cfg = _project(tmp_path)
    calls, seen = [], []
    _fakes(monkeypatch, cfg, calls, fail=KeyboardInterrupt())
    _stub_products(monkeypatch, cfg, seen)
    rc = pipeline.run_from(cfg, "products", no_acceptance=True, watchdog=False, notify=False)
    assert rc == 130 and seen == []
    assert progress.latest(cfg.project_dir)["deliverables"]["status"] == "failed"


@pytest.mark.parametrize("backend,marker", [("disp", False), ("local", False), ("miaplpy", True)])
def test_routes_without_the_layers_and_an_unfinished_run_are_left_alone(tmp_path, monkeypatch, caplog,
                                                                          backend, marker):
    """The DISP chain writes its own geo files and the local route's MintPy
    run never made these layers: nothing is geocoded there. Nor while
    mintpy/run_incomplete.json says the MintPy run did not finish (the
    deliverables refuse that run; its layers must not be added to it)."""
    cfg = _project(tmp_path, backend=backend)
    if marker:
        (cfg.mintpy_dir / ts.RUN_INCOMPLETE).write_text("{}")
    calls = []
    _fakes(monkeypatch, cfg, calls)
    with caplog.at_level(logging.INFO):
        rec = pipeline.geocode_missing_layers(cfg)
    assert calls == [] and rec["made"] == [] and rec["skipped"]
    assert rec["missing_after"] == GEO_NAMES
    assert any("not geocoded" in r.getMessage() for r in caplog.records)


# ------------------------------------------------------------------ review of F5
def _partial_geocode(monkeypatch, cfg, calls, rc=0, exc=None, skip_existing=False):
    """geocode.py cut off part-way: it creates each geo_ file before it writes
    the data, so the file is there and all zeros; then it exits rc or the run
    is stopped (exc). skip_existing: --update leaves a current output alone."""
    M = cfg.mintpy_dir

    def fake_geocode(cfg_, files):
        calls.append(("geocode", [Path(p).name for p in files]))
        for src in files:
            g = M / "geo" / f"geo_{Path(src).name}"
            if skip_existing and g.exists():
                continue
            with h5py.File(src, "r") as s, h5py.File(g, "w") as out:
                for k in s:
                    out.create_dataset(k, shape=s[k].shape, dtype=np.float32)     # created, never written
        if exc is not None:
            raise exc
        return types.SimpleNamespace(returncode=rc)
    monkeypatch.setattr(ts, "_run_geocode", fake_geocode)


@pytest.mark.parametrize("rc,exc", [(-9, None), (1, None), (0, KeyboardInterrupt()),
                                    (0, SystemExit(143)), (0, RuntimeError("pyresample"))])
def test_an_unfinished_geocoding_leaves_no_layer_behind(tmp_path, monkeypatch, caplog, rc, exc):
    """The products stage keeps a layer that is on disk; a geocoding that
    timed out, failed or was stopped part-way would otherwise leave a
    zero-filled geo_temporalCoherence_mintpy.h5 (every point failing the
    second gate) or geo_demErr.h5 (no ground ever reshaped) that the
    deliverables, QA and report read on every later run. Its files are
    removed, and the next products run makes them."""
    cfg = _project(tmp_path)
    G = cfg.mintpy_dir / "geo"
    calls = []
    _fakes(monkeypatch, cfg, calls)
    _partial_geocode(monkeypatch, cfg, calls, rc=rc, exc=exc)
    with caplog.at_level(logging.INFO):
        if exc is None:
            assert ts.geocode_extra_layers(cfg, missing_only=True) == []
        else:
            with pytest.raises(type(exc)):
                ts.geocode_extra_layers(cfg, missing_only=True)
    assert not [n for n in GEO_NAMES if (G / n).exists()]
    assert any("Removed geo_temporalCoherence_mintpy.h5, geo_demErr.h5, geo_disturbed.h5" in r.getMessage()
               for r in caplog.records if r.levelno >= logging.ERROR)
    _fakes(monkeypatch, cfg, calls)                      # the next products run: geocode.py works
    rec = pipeline.geocode_missing_layers(cfg)
    assert rec["made"] == GEO_NAMES and rec["missing_after"] == []


def test_an_unfinished_geocoding_keeps_what_it_did_not_write(tmp_path, monkeypatch):
    """Only the files the failed call wrote go: a current layer geocode.py
    left alone (--update) stays, as does any layer not asked for."""
    cfg = _project(tmp_path)
    G = cfg.mintpy_dir / "geo"
    kept = _w(G / "geo_demErr.h5", {"dem": np.full(RSHAPE, 7.0, np.float32)}, {}, time.time() + 60)
    calls = []
    _fakes(monkeypatch, cfg, calls)
    _partial_geocode(monkeypatch, cfg, calls, rc=1, skip_existing=True)
    out = ts.geocode_extra_layers(cfg)                    # the MintPy run's call: all three asked for
    assert calls[-1] == ("geocode", ["temporalCoherence_mintpy.h5", "demErr.h5", "disturbed.h5"])
    assert kept.exists() and [p.name for p in out] == ["geo_demErr.h5"]
    assert not (G / "geo_temporalCoherence_mintpy.h5").exists() and not (G / "geo_disturbed.h5").exists()


def test_a_failing_disturbed_layer_does_not_stop_the_other_two(tmp_path, monkeypatch, caplog):
    """The second gate's layer and the DEM error - what the products step is
    for - do not need the disturbed layer; until this review its failure
    ended the call and neither was ever made."""
    cfg = _project(tmp_path)
    calls = []
    _fakes(monkeypatch, cfg, calls)

    def boom(cfg_):
        raise MemoryError("stack too large")
    monkeypatch.setattr(ts, "write_disturbed_layer", boom)
    with caplog.at_level(logging.INFO):
        rec = pipeline.geocode_missing_layers(cfg)
    assert calls == [("geocode", ["temporalCoherence_mintpy.h5", "demErr.h5"])]
    assert rec["made"] == ["geo_temporalCoherence_mintpy.h5", "geo_demErr.h5"]
    assert rec["missing_after"] == ["geo_disturbed.h5"] and rec["error"] is None
    assert any("Disturbed-ground layer not written (MemoryError" in r.getMessage()
               for r in caplog.records if r.levelno >= logging.ERROR)
