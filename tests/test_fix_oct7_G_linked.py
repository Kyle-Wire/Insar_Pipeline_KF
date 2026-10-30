"""Group G_linked of the 7 Oct 2026 fixes: the time-series run and the
phase-linked route (findings 31-34).

Every test here fails on the code before the fix (backups/pre_fixes_20261007)
and passes after it. Synthetic data only; MintPy, snaphu and MiaplPy are never
run - the time-series tests reuse the fake MintPy of test_fix_L2.py.

Run: conda run -n insar python -m pytest tests/test_fix_oct7_G_linked.py -q -p no:cacheprovider
"""
import importlib.util
import json
import os
import subprocess
import sys
import time
import types
from pathlib import Path

import h5py
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_fix_L2 as L2                                   # noqa: E402  (fixtures only)
from insar_pipeline import miaplpy_backend as mb          # noqa: E402
from insar_pipeline import stack_backend as sb            # noqa: E402
from insar_pipeline import time_series as ts              # noqa: E402
from insar_pipeline import unwrap_crop as uc              # noqa: E402
from insar_pipeline.config import PipelineConfig          # noqa: E402


# ------------------------------------------------- 31: the run_incomplete.json marker
class _StoppedInPlotTail(L2.FakeMintPy):
    """A dashboard Stop (or an out-of-memory kill) during MintPy's KMZ/plot tail."""

    def __call__(self, cfg, config_path, start, end, log_name, append=False):
        if start == "google_earth":
            raise KeyboardInterrupt("stopped")
        return super().__call__(cfg, config_path, start, end, log_name, append)


@pytest.mark.parametrize("how", ["end_state", "second_gate", "stop_in_plot_tail"])
def test_a_run_that_fails_after_geocoding_leaves_the_incomplete_marker(tmp_path, monkeypatch, how):
    """Before: retire moved run_products.json aside, geocoding wrote current
    products, then the deramp-mask end check / the second gate raised (or a
    Stop hit the plot tail): geo/ looked finished and nothing said otherwise,
    so 'Resume' offered 'Products only'. Now run_incomplete.json stays until
    run_products.json is written, and a clean re-run removes it."""
    cfg = L2.make_cfg(tmp_path, mintpy_tropo_method="no")
    L2._tiny_stack(cfg)
    L2._previous_run(cfg)
    M = cfg.mintpy_dir
    fake = (_StoppedInPlotTail if how == "stop_in_plot_tail" else L2.FakeMintPy)(cfg)
    L2._stub_preamble(monkeypatch, fake)
    real_end_state = ts._report_mintpy_end_state
    if how == "end_state":
        def boom(cfg_):
            raise RuntimeError("THE DERAMP DID NOT USE THE EXCLUSION MASK")
        monkeypatch.setattr(ts, "_report_mintpy_end_state", boom)
    elif how == "second_gate":
        L2._write(M / "temporalCoherence_mintpy.h5", {"temporalCoherence": np.ones(L2.SHAPE, np.float32)}, {})
        cfg.min_unwrap_consistency = 0.5           # asked for, but no layer was geocoded
    why = {"end_state": "DERAMP DID NOT USE", "second_gate": "no unwrapping-consistency layer",
           "stop_in_plot_tail": "stopped"}[how]
    with pytest.raises((RuntimeError, KeyboardInterrupt), match=why):    # failed where meant to, after geocoding
        ts.run_mintpy_with_loader(cfg, L2.LOADER)
    assert (M / "geo" / "geo_velocity.h5").exists()               # current-looking products...
    assert not (M / "run_products.json").exists()                 # ...with no record of a finished run
    mark = json.loads((M / "run_incomplete.json").read_text())
    assert mark["stamp"] and mark["started"]

    # the next clean run removes it
    time.sleep(1.1)                                               # its own run_<stamp> folder
    monkeypatch.setattr(ts, "_report_mintpy_end_state", real_end_state)
    cfg.min_unwrap_consistency = 0.0
    L2._stub_preamble(monkeypatch, L2.FakeMintPy(cfg))
    ts.run_mintpy_with_loader(cfg, L2.LOADER)
    assert (M / "run_products.json").exists() and not (M / "run_incomplete.json").exists()


def test_the_marker_is_down_before_anything_is_moved(tmp_path, monkeypatch):
    """A failure while moving the previous run aside leaves a mixed mintpy/:
    it must already be marked."""
    cfg = L2.make_cfg(tmp_path, mintpy_tropo_method="no")
    L2._tiny_stack(cfg)
    L2._previous_run(cfg)
    L2._stub_preamble(monkeypatch, L2.FakeMintPy(cfg))

    def half_moved(cfg_, stamp):
        assert (cfg_.mintpy_dir / "run_incomplete.json").exists()
        raise RuntimeError("Could not move the previous run's products aside")
    monkeypatch.setattr(ts, "retire_downstream_products", half_moved)
    with pytest.raises(RuntimeError, match="Could not move"):
        ts.run_mintpy_with_loader(cfg, L2.LOADER)
    assert (cfg.mintpy_dir / "run_incomplete.json").exists()


def test_a_successful_run_leaves_no_marker(tmp_path, monkeypatch):
    cfg = L2.make_cfg(tmp_path, mintpy_tropo_method="no")
    L2._tiny_stack(cfg)
    (cfg.mintpy_dir / "run_incomplete.json").write_text("{}")     # left by an earlier failed run
    L2._stub_preamble(monkeypatch, L2.FakeMintPy(cfg))
    ts.run_mintpy_with_loader(cfg, L2.LOADER)
    assert not (cfg.mintpy_dir / "run_incomplete.json").exists()


def test_the_topsapp_route_marks_and_clears_too(tmp_path, monkeypatch):
    """The older single-call MintPy route (run_mintpy) moves the previous
    products aside as well: marked while it runs, cleared on success."""
    cfg = L2.make_cfg(tmp_path)
    M = cfg.mintpy_dir
    (M / "smallbaselineApp.cfg").write_text("")
    monkeypatch.setattr(ts, "write_mintpy_config", lambda cfg_, dem: M / "smallbaselineApp.cfg")
    monkeypatch.setattr(ts, "patch_stack_bperp", lambda cfg_: None)
    state = {"write": False}

    def run(argv, **kw):
        if "modify_network" not in argv:                      # load_data
            return types.SimpleNamespace(returncode=0, stdout="", stderr="")
        if state["write"]:
            (M / "velocity.h5").write_bytes(b"v")
        return types.SimpleNamespace(returncode=0 if state["write"] else 1, stdout="", stderr="")
    monkeypatch.setattr(ts, "subprocess", types.SimpleNamespace(run=run))
    with pytest.raises(RuntimeError):
        ts.run_mintpy(cfg, tmp_path / "dem")
    assert (M / "run_incomplete.json").exists()
    time.sleep(1.1)
    state["write"] = True
    ts.run_mintpy(cfg, tmp_path / "dem")
    assert not (M / "run_incomplete.json").exists()


# ------------------------------------------------- 33: a reload keeps the previous radar products
def test_reloading_the_stack_moves_the_previous_radar_products_aside(tmp_path):
    """Before: with no ifgramStack.h5 every top-level *.h5 was deleted,
    the previous run's velocity.h5 and corrected series included."""
    cfg = L2.make_cfg(tmp_path)
    M = cfg.mintpy_dir
    for n in ("velocity.h5", "timeseries_ramp_demErr.h5", "demErr.h5", "waterMask.h5", "avgSpatialCoh.h5",
              "timeseries.h5", "inputs/geometryRadar.h5"):
        (M / n).write_bytes(b"x")
    gone = ts.clear_stale_mintpy_products(cfg, "20261007_010203")
    dest = M / "superseded" / "run_20261007_010203"
    for n in ("velocity.h5", "timeseries_ramp_demErr.h5", "demErr.h5"):
        assert (dest / n).exists() and not (M / n).exists()
    for n in ("waterMask.h5", "avgSpatialCoh.h5", "timeseries.h5", "inputs/geometryRadar.h5"):
        assert not (M / n).exists() and not (dest / n).exists()       # derived inputs: still deleted
    assert len(gone) == 7
    # the rest of the same run goes into the same folder, listed in one RETIRED.json
    (M / "geo").mkdir()
    (M / "geo" / "geo_velocity.h5").write_bytes(b"g")
    (M / "run_products.json").write_text("{}")
    assert ts.retire_downstream_products(cfg, "20261007_010203") == dest
    files = json.loads((dest / "RETIRED.json").read_text())["files"]
    assert {"velocity.h5", "timeseries_ramp_demErr.h5", "geo/geo_velocity.h5", "run_products.json"} <= set(files)
    # nothing more to move, but this run's folder exists: it is still named
    assert ts.retire_downstream_products(cfg, "20261007_010203") == dest


class _Reloads(L2.FakeMintPy):
    """load_data builds the stack (the previous one was set aside)."""

    def __call__(self, cfg, config_path, start, end, log_name, append=False):
        if start == "load_data":
            L2._tiny_stack(cfg)
        return super().__call__(cfg, config_path, start, end, log_name, append)


def _load_reunwrap_script():
    spec = importlib.util.spec_from_file_location(
        "reunwrap_linked_g", Path(__file__).resolve().parents[1] / "scripts" / "reunwrap_linked.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_reunwrap_note_points_at_a_folder_that_holds_the_previous_velocity_and_series(tmp_path, monkeypatch):
    """A re-unwrap reloads the stack. Its note says the previous velocity and
    time series are in superseded/run_<stamp>/; before, the radar ones had
    been deleted on the way."""
    cfg = L2.make_cfg(tmp_path, mintpy_tropo_method="no")
    L2._previous_run(cfg)                                     # no ifgramStack.h5: it was set aside
    M = cfg.mintpy_dir
    keep = _load_reunwrap_script().keep_previous(cfg)
    note = (keep / "README.txt").read_text()
    assert "run_<stamp>" in note and "radar" in note
    L2._stub_preamble(monkeypatch, _Reloads(cfg))
    ts.run_mintpy_with_loader(cfg, L2.LOADER)
    runs = list((M / "superseded").glob("run_*"))
    assert len(runs) == 1
    for n in ("velocity.h5", "timeseries_ERA5_ramp_demErr.h5", "geo/geo_velocity.h5",
              "geo/geo_timeseries_ERA5_ramp_demErr.h5", "run_products.json"):
        assert (runs[0] / n).exists(), n
    with h5py.File(runs[0] / "velocity.h5", "r") as f:              # the PREVIOUS run's (ones), not this one's
        assert float(f["velocity"][0, 0]) == 1.0
    assert json.loads((M / "run_products.json").read_text())["superseded_dir"] == str(runs[0].absolute())


# ------------------------------------------------- 32: sample pair coherence re-unwraps its pairs
PAIR = "20230104_20230116"


def _linked_cfg(tmp_path):
    cfg = PipelineConfig(project_dir=tmp_path / "proj", bbox=[-87.8, 38.2, -87.52, 38.47], start_date="2023-01-01",
                         end_date="2023-08-31", backend="miaplpy", stack_reference_date="20230116")
    cfg.stack_azimuth_looks, cfg.stack_range_looks = 2, 2
    s = cfg.stack_dir
    (s / "merged").mkdir(parents=True, exist_ok=True)
    (s / "crop_window.json").write_text(json.dumps({"r0": 0, "r1": 4, "c0": 0, "c1": 5, "length": 4, "width": 5}))
    slc = mb.work_dir(cfg) / "inputs" / "slcStack.h5"
    slc.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1)
    with h5py.File(slc, "w") as f:
        f["date"] = np.array([b"20230104", b"20230116"])
        f.create_dataset("slc", data=(rng.normal(size=(2, 8, 10)) + 1j * rng.normal(size=(2, 8, 10))
                                      ).astype(np.complex64), chunks=(1, 8, 10))
    d = s / "merged" / "interferograms" / PAIR
    d.mkdir(parents=True)
    (d / "filt_fine.int").write_bytes(b"\0" * 8 * 20)
    np.full((4, 5), 0.77, np.float32).tofile(d / "filt_fine.cor")      # the phasor figure
    (d / "filt_fine.unw").write_bytes(b"\0" * 8 * 20)
    (d / "filt_fine.unw.conncomp").write_bytes(b"\1" * 20)
    return cfg


def _fake_isce_io(monkeypatch):
    img = types.SimpleNamespace(length=4, width=5, bands=1, dataType="FLOAT", scheme="BIL")
    monkeypatch.setattr(uc, "_load_image", lambda p: img)
    monkeypatch.setattr(uc, "_read_raw", lambda p, i: np.fromfile(p, np.float32).reshape(4, 5))
    monkeypatch.setattr(uc, "_write_like", lambda tpl, out, arr: np.asarray(arr, np.float32).tofile(out))


def test_switching_to_sample_coherence_unwraps_the_pair_again(tmp_path, monkeypatch):
    """Before: the pair's unwrap marker survived the coherence rewrite (its
    config has no coherence kind), so 'Re-unwrap' skipped every pair and
    MintPy kept the phasor coherence while the records said 'sample'."""
    cfg = _linked_cfg(tmp_path)
    _fake_isce_io(monkeypatch)
    ran = []

    def fake_execute(cfg_, rf, fail_fast):
        for cmd in sb.parse_run_file(rf):
            m = sb._status_path(cfg_, rf.name, cmd)
            if m.exists() and json.loads(m.read_text()).get("returncode") == 0:
                continue                                    # what _run_one does with a valid marker
            ran.append(cmd)
            m.parent.mkdir(parents=True, exist_ok=True)
            m.write_text(json.dumps({"cmd": cmd, "returncode": 0, "config_sha1": sb._config_sha1(cmd)}))
        return {"run_file": rf.name}
    monkeypatch.setattr(sb, "execute_run_file", fake_execute)
    mb.unwrap_exported(cfg, [PAIR])                          # the phasor unwrap
    assert len(ran) == 1
    mb.unwrap_exported(cfg, [PAIR])                          # nothing changed: skipped
    assert len(ran) == 1

    cfg.miaplpy_pair_coherence = "sample"
    rec = mb.write_pair_coherence(cfg, [PAIR])
    d = cfg.stack_dir / "merged" / "interferograms" / PAIR
    assert rec["kind"] == "sample" and (d / "filt_fine.phasor.cor").exists()
    assert not np.allclose(np.fromfile(d / "filt_fine.cor", np.float32), 0.77)
    assert not list((cfg.stack_dir / ".status" / "run_16_unwrap").glob("*.done"))
    mb.unwrap_exported(cfg, [PAIR])
    assert len(ran) == 2 and f"config_igram_unw_{PAIR}" in ran[-1]
    mb.write_pair_coherence(cfg, [PAIR])                     # already sample: the new marker stays
    assert len(list((cfg.stack_dir / ".status" / "run_16_unwrap").glob("*.done"))) == 1


def test_a_stack_older_than_its_coherence_is_reloaded(tmp_path):
    """Backstop: filt_fine.cor rewritten after the stack was loaded, with the
    unwrapped phase unchanged. Before, only unwFile was compared."""
    cfg = L2.make_cfg(tmp_path)
    stack = cfg.mintpy_dir / "inputs" / "ifgramStack.h5"
    stack.write_bytes(b"old")
    d = tmp_path / "ifg" / PAIR
    d.mkdir(parents=True)
    unw, cor = d / "filt_fine.unw", d / "filt_fine.cor"
    unw.write_bytes(b"u"); cor.write_bytes(b"c")
    past = time.time() - 100
    os.utime(unw, (past - 50, past - 50))
    os.utime(stack, (past, past))
    lines = (f"mintpy.load.unwFile        = {tmp_path}/ifg/*/filt_fine.unw\n"
             f"mintpy.load.corFile        = {tmp_path}/ifg/*/filt_fine.cor\n")
    aside = ts.retire_stale_ifgram_stack(cfg, lines)
    assert aside is not None and aside.exists() and not stack.exists()
    # and a stack newer than both is kept
    stack.write_bytes(b"new")
    os.utime(cor, (past - 50, past - 50))
    assert ts.retire_stale_ifgram_stack(cfg, lines) is None and stack.exists()


# ------------------------------------------------- 34: the patch-check advice points the right way
def _patch_check_says(monkeypatch, lag, ratio):
    out = json.dumps({"patched": False, "ratio": ratio, "ministack": 10, "time_lag": lag,
                      "miaplpy_version": "0.2.1+overlapdatum"})
    monkeypatch.setattr(mb.subprocess, "run", lambda argv, **kw: subprocess.CompletedProcess(argv, 1, out + "\n", ""))


def test_a_too_large_window_overlap_is_told_to_come_down(monkeypatch):
    """Measured: the boundary error grows with the overlap (2.11x at 7 with
    mini-stack 10). Before, the advice was 'Raise miaplpy_time_lag'."""
    _patch_check_says(monkeypatch, 7, 2.11)
    with pytest.raises(RuntimeError) as e:
        mb.verify_patched_miaplpy("miaplpy", ministack=10, time_lag=7)
    assert "Lower miaplpy_time_lag" in str(e.value) and "Raise" not in str(e.value)
    assert "3 is the tested default" in str(e.value)


def test_no_window_overlap_is_still_told_to_go_up(monkeypatch):
    _patch_check_says(monkeypatch, 0, 21.95)
    with pytest.raises(RuntimeError) as e:
        mb.verify_patched_miaplpy("miaplpy", ministack=10, time_lag=0)
    assert "Raise miaplpy_time_lag to at least 1" in str(e.value) and "Lower" not in str(e.value)
