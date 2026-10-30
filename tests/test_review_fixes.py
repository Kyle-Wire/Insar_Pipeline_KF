"""Regression tests for the defects found by the review of 26-27 Sep 2026."""
import json
from pathlib import Path

import h5py
import numpy as np

from insar_pipeline.config import PipelineConfig


def make_cfg(tmp_path, **over) -> PipelineConfig:
    kw = dict(project_dir=tmp_path / "proj", bbox=[-87.8, 38.2, -87.52, 38.47],
              start_date="2018-06-01", end_date="2019-06-30", backend="stack")
    kw.update(over)
    cfg = PipelineConfig(**kw)
    (cfg.stack_dir / "merged").mkdir(parents=True, exist_ok=True)
    cfg.mintpy_dir.mkdir(parents=True, exist_ok=True)
    return cfg


def test_polygon_mask_tests_cell_centres(tmp_path):
    from insar_pipeline.stack_backend import _polygon_mask
    # 10 x 10 grid of 0.001-degree cells from (0, 0.01) (a step >= 0.01 is read as metres);
    # a square covering x 0.0024-0.0046, y 0.0054-0.0076
    attrs = {"X_FIRST": 0.0, "Y_FIRST": 0.01, "X_STEP": 0.001, "Y_STEP": -0.001}
    k = 0.001
    gj = tmp_path / "sq.geojson"
    gj.write_text(json.dumps({"type": "FeatureCollection", "features": [{"type": "Feature", "properties": {},
        "geometry": {"type": "Polygon", "coordinates": [[[2.4 * k, 5.4 * k], [4.6 * k, 5.4 * k], [4.6 * k, 7.6 * k],
                                                          [2.4 * k, 7.6 * k], [2.4 * k, 5.4 * k]]]}}]}))
    m = _polygon_mask(gj, (10, 10), attrs)
    # centres inside: x 2.5, 3.5 (cols 2, 3) and 4.5 (col 4); y 7.5, 6.5, 5.5 (rows 2, 3, 4)
    assert m.sum() == 9 and m[2:5, 2:5].all()


def test_polygon_mask_hole_is_excluded(tmp_path):
    from insar_pipeline.stack_backend import _polygon_mask
    attrs = {"X_FIRST": 0.0, "Y_FIRST": 0.01, "X_STEP": 0.001, "Y_STEP": -0.001}
    k = 0.001
    gj = tmp_path / "ring.geojson"
    gj.write_text(json.dumps({"type": "Polygon", "coordinates": [
        [[0, 0], [10 * k, 0], [10 * k, 10 * k], [0, 10 * k], [0, 0]],
        [[4 * k, 4 * k], [6 * k, 4 * k], [6 * k, 6 * k], [4 * k, 6 * k], [4 * k, 4 * k]]]}))
    m = _polygon_mask(gj, (10, 10), attrs)
    assert m.sum() == 96 and not m[4:6, 4:6].any()


def _write_npz(path, avg, full, marked):
    extra = {"layer_order": np.array("miaplpy temporalCoherence index 0 = average (mini-stack), 1 = full (whole stack)")}
    if marked:
        np.savez_compressed(path, average=avg, full=full, **extra)
    else:                                   # the old writer: index 0 stored under 'full'
        np.savez_compressed(path, full=avg, average=full)


def test_linked_coherence_legacy_file_is_read_the_right_way_round(tmp_path):
    from insar_pipeline import miaplpy_backend as mb
    cfg = make_cfg(tmp_path, backend="miaplpy")
    avg, full = np.full((3, 3), 0.52, np.float32), np.full((3, 3), 0.03, np.float32)
    _write_npz(mb.linked_temporal_coherence_path(cfg), avg, full, marked=False)
    assert np.allclose(mb.load_linked_temporal_coherence(cfg), 0.52)          # default layer = average
    assert np.allclose(mb.load_linked_temporal_coherence(cfg, "full"), 0.03)
    _write_npz(mb.linked_temporal_coherence_path(cfg), avg, full, marked=True)
    assert np.allclose(mb.load_linked_temporal_coherence(cfg), 0.52)
    assert np.allclose(mb.load_linked_temporal_coherence(cfg, "full"), 0.03)


def test_reference_pin_also_clears_lalo(tmp_path, monkeypatch):
    from insar_pipeline import time_series as ts
    cfg = make_cfg(tmp_path, reference_lalo="38.3,-87.6")
    mask = np.ones((20, 20), bool); mask[10, 10] = False
    with h5py.File(cfg.mintpy_dir / "maskConnComp.h5", "w") as f:
        f["mask"] = mask
    monkeypatch.setattr(ts, "mintpy_reference_yx", lambda cfg, la, lo: (10, 10))
    tmpl = cfg.mintpy_dir / "smallbaselineApp.cfg"
    tmpl.write_text("mintpy.reference.yx            = auto\nmintpy.reference.lalo          = 38.3,-87.6\n")
    info = ts.pin_reference_into_conncomp_mask(cfg, tmpl)
    text = tmpl.read_text()
    assert "pinned_yx" in info
    assert "mintpy.reference.lalo = auto" in text and "38.3,-87.6" not in text
    y, x = info["pinned_yx"]
    assert f"mintpy.reference.yx = {y},{x}" in text


def test_egms_units_are_mm_whatever_the_display_units(tmp_path):
    # techout-8 / Kyle's decision 4: EGMS-style point columns are always mm and mm/yr
    # (the EGMS convention); inches stay in the report, KMZ and client rasters.
    from insar_pipeline.deliverables import egms_field_units
    for units in ("in", "mm"):
        u = dict(egms_field_units(make_cfg(tmp_path, display_units=units)))
        assert u["mean_velocity"].startswith("mm/yr") and u["rmse"].startswith("mm")


def test_iter_epochs_reads_every_epoch_once(tmp_path):
    from insar_pipeline.significance import iter_epochs
    p = tmp_path / "ts.h5"
    data = np.arange(19 * 4 * 5, dtype=np.float32).reshape(19, 4, 5)
    with h5py.File(p, "w") as f:
        f.create_dataset("timeseries", data=data, chunks=(8, 4, 5))
    with h5py.File(p, "r") as f:
        got = list(iter_epochs(f["timeseries"]))
    assert [i for i, _ in got] == list(range(19))
    assert all(np.array_equal(e, data[i]) for i, e in got)


def test_run_figures_draws_the_histories_before_the_summary_table(monkeypatch, tmp_path):
    from types import SimpleNamespace

    from insar_pipeline import figures, panel_history, pipeline, progress
    order = []
    monkeypatch.setattr(panel_history, "run", lambda cfg: order.append("history"))
    monkeypatch.setattr(figures, "make_figures", lambda cfg: order.append("figures") or Path("."))
    pipeline.run_figures(SimpleNamespace(project_dir=tmp_path))
    assert order == ["history", "figures"]
    assert progress.latest(tmp_path)["figures"]["status"] == "done"


def test_marker_goes_stale_when_the_config_changes(tmp_path):
    import os
    import time
    from insar_pipeline import stack_backend as sb
    conf = tmp_path / "config_igram_unw_20220101_20220113"
    conf.write_text("defomax : 4.0\n")
    cmd = f"python -m insar_pipeline.unwrap_crop --config {conf} --window w.json"
    marker = tmp_path / "m.done"
    rec = {"returncode": 0, "config_sha1": sb._config_sha1(cmd)}
    marker.write_text(json.dumps(rec))
    assert sb._marker_stale(marker, rec, cmd) is None
    conf.write_text("defomax : 4.0\nengine : snaphu2\n")
    assert sb._marker_stale(marker, rec, cmd)
    # a marker from before the fingerprint: judged by modification time
    old = {"returncode": 0}
    marker.write_text(json.dumps(old))
    past = time.time() - 100
    os.utime(conf, (past, past))
    assert sb._marker_stale(marker, old, cmd) is None
    os.utime(conf, None)
    os.utime(marker, (past, past))
    assert sb._marker_stale(marker, old, cmd)


def test_engine_keys_leave_default_configs_untouched(tmp_path):
    from insar_pipeline.unwrap_crop import apply_engine
    text = "[Function-1]\nunwrap : \nnomcf : True\ndefomax : 4.0\n"
    cfg = make_cfg(tmp_path)
    assert apply_engine(text, cfg) == text                      # isce: byte-identical
    cfg2 = make_cfg(tmp_path, stack_unw_engine="snaphu2", stack_unw_cost_mode="smooth",
                    stack_unw_snaphu_bin="/bin/true")
    out = apply_engine(text, cfg2)
    assert "engine : snaphu2" in out and "cost : SMOOTH" in out and "maxcomp : 64" in out
    assert apply_engine(out, cfg2) == out                       # idempotent
    assert apply_engine(out, cfg) == text.rstrip("\n") + "\n" or "engine" not in apply_engine(out, cfg)


def test_stale_ifgram_stack_is_set_aside(tmp_path):
    import os
    import time
    from insar_pipeline import time_series as ts
    cfg = make_cfg(tmp_path)
    (cfg.mintpy_dir / "inputs").mkdir(parents=True, exist_ok=True)
    stack = cfg.mintpy_dir / "inputs" / "ifgramStack.h5"
    stack.write_bytes(b"old")
    unw = tmp_path / "ifg" / "20220101_20220113"; unw.mkdir(parents=True)
    f = unw / "filt_fine.unw"; f.write_bytes(b"x")
    lines = f"mintpy.load.unwFile = {tmp_path}/ifg/*/filt_fine.unw\n"
    past = time.time() - 100
    os.utime(f, (past, past))
    assert ts.retire_stale_ifgram_stack(cfg, lines) is None and stack.exists()
    os.utime(f, None)
    os.utime(stack, (past, past))
    aside = ts.retire_stale_ifgram_stack(cfg, lines)
    assert aside is not None and aside.exists() and not stack.exists()
    # the very next step clears stale products: the set-aside stack must survive it
    (cfg.mintpy_dir / "inputs" / "geometryRadar.h5").write_bytes(b"g")
    (cfg.mintpy_dir / "avgSpatialCoh.h5").write_bytes(b"a")
    gone = ts.clear_stale_mintpy_products(cfg)
    assert aside.exists() and len(gone) == 2


def test_closure_integer_correction_fixes_an_injected_cycle_error(tmp_path):
    from insar_pipeline import time_series as ts
    cfg = make_cfg(tmp_path)
    (cfg.mintpy_dir / "inputs").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(4)
    dates = [f"2022{m:02d}01" for m in range(1, 9)]
    pairs = [(i, j) for i in range(8) for j in range(i + 1, min(i + 4, 8))]      # 3-connection
    truth = np.cumsum(rng.normal(0, 1.0, (8, 20, 30)), axis=0).astype(np.float32)
    unw = np.stack([truth[j] - truth[i] for i, j in pairs]) + rng.normal(0, 0.05, (len(pairs), 20, 30)).astype(np.float32)
    k = pairs.index((3, 4))
    bad = unw.copy(); bad[k, 5:9, 10:15] += 2 * np.pi; bad[k, 12, 3] -= 4 * np.pi
    with h5py.File(cfg.mintpy_dir / "inputs" / "ifgramStack.h5", "w") as f:
        f["date"] = np.array([[dates[i].encode(), dates[j].encode()] for i, j in pairs])
        f["dropIfgram"] = np.ones(len(pairs), bool)
        f.create_dataset("unwrapPhase_bridging", data=bad, chunks=(len(pairs), 10, 30))
        f.attrs["REF_Y"], f.attrs["REF_X"] = "0", "0"
    rec = ts.closure_integer_correction(cfg, "unwrapPhase_bridging", block_rows=7)
    with h5py.File(cfg.mintpy_dir / "inputs" / "ifgramStack.h5", "r") as f:
        out = f["unwrapPhase_bridging_phaseClosure"][:]
    assert np.allclose(out, unw, atol=1e-4)                       # every injected cycle removed, nothing else touched
    assert rec["percent_loops_in_error_after"] == 0 and rec["ifg_pixels_corrected"] == 21


def test_closure_diagnostic_follows_the_inverted_dataset(tmp_path):
    """A reused stack can hold another variant's dataset; the diagnostic and
    the QA must describe the one the inversion used (27 Sep 2026)."""
    from types import SimpleNamespace
    from insar_pipeline.time_series import inverted_dataset
    cfg = SimpleNamespace(mintpy_dir=tmp_path)
    names = ["unwrapPhase", "unwrapPhase_bridging", "unwrapPhase_bridging_phaseClosure"]
    assert inverted_dataset(cfg, names) == "unwrapPhase_bridging_phaseClosure"      # no record: most corrected
    (tmp_path / "unwrap_error_applied.json").write_text(json.dumps({"dataset": "unwrapPhase_bridging"}))
    assert inverted_dataset(cfg, names) == "unwrapPhase_bridging"
