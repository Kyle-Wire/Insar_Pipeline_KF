"""
Lane L6b of the 5 Oct 2026 review: the official phase-linked route.

Every test here failed on the code before the fix and passes after it:
completion markers instead of file existence (an interrupted concatenation
or load is never skipped, and the 12-hour PATCHES are never deleted on a
probe), the linking fingerprint (a changed setting stops a resume with plain
steps; the datum gate reads the method the series was linked with), the
datum gate on coherent cells only, phase-linking progress and stall
detection, the reference far field measured to polygon EDGES, failed
unwraps quarantined, the patch check run with the configured overlap, and
the re-unwrap launcher holding the lock with teardown installed.

Synthetic data only; no network, no MiaplPy (one test uses the miaplpy env
when it is installed and skips otherwise).

Run: conda run -n insar python -m pytest tests/test_fix_L6b.py -q -p no:cacheprovider
"""
import importlib.util
import json
import os
import subprocess
import sys
import time
import types
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from insar_pipeline import miaplpy_backend as mb          # noqa: E402
from insar_pipeline import procs                          # noqa: E402
from insar_pipeline import progress                       # noqa: E402
from insar_pipeline import stack_backend as sb            # noqa: E402
from insar_pipeline import unwrap_crop as uc              # noqa: E402
from insar_pipeline.config import PipelineConfig          # noqa: E402

h5py = pytest.importorskip("h5py")
DATES = ["20230104", "20230116", "20230128", "20230209", "20230221"]


# ---------------------------------------------------------------- fixtures
def _cfg(tmp_path, **over):
    """A phase-linked project whose aligned subset is the whole 400 x 400 SLC frame
    (crop 200 x 200 multilooked at 2 x 2), so MiaplPy's patch grid is 2 x 2."""
    cfg = PipelineConfig(project_dir=tmp_path / "proj", bbox=[-87.8, 38.2, -87.52, 38.47], start_date="2023-01-01",
                         end_date="2023-08-31", backend="miaplpy", stack_reference_date="20230221")
    cfg.stack_azimuth_looks, cfg.stack_range_looks = 2, 2
    for k, v in over.items():
        setattr(cfg, k, v)
    s = cfg.stack_dir
    (s / "merged" / "SLC" / "20230221").mkdir(parents=True, exist_ok=True)
    (s / "crop_window.json").write_text(json.dumps({"r0": 0, "r1": 200, "c0": 0, "c1": 200,
                                                    "length": 200, "width": 200}))
    (s / "merged" / "SLC" / "20230221" / "20230221.slc.full.xml").write_text(
        '<property name="width"><value>400</value></property><property name="length"><value>400</value></property>')
    (mb.work_dir(cfg) / "inverted").mkdir(parents=True, exist_ok=True)
    return cfg


def _patches(cfg, n=4, flagged=4):
    inv = mb.work_dir(cfg) / "inverted"
    for k in range(n):
        d = inv / "PATCHES" / f"PATCH_{k:04d}"; d.mkdir(parents=True, exist_ok=True)
        if k < flagged:
            np.save(d / "flag.npy", [1])
    return inv / "PATCHES"


def _phase_series(cfg, rows_written=None, n=5, L=400, W=400, value=0.3):
    """phase_series.h5 the way MiaplPy's initiate_output leaves it (phase created
    with no fill value, temporalCoherence = -1), then rows [0:rows_written)
    written as unpatch() writes blocks (phase first, temporalCoherence last,
    clamped >= 0). rows_written=None writes everything."""
    ps = mb.work_dir(cfg) / "inverted" / "phase_series.h5"
    ps.unlink(missing_ok=True)
    with h5py.File(ps, "w") as f:
        f.create_dataset("phase", shape=(n, L, W), maxshape=(None, L, W), chunks=True, dtype=np.float32)
        tc = f.create_dataset("temporalCoherence", shape=(2, L, W), maxshape=(2, L, W), chunks=True, dtype=np.float32)
        tc[:, :, :] = -1
        f.create_dataset("date", data=np.array(DATES[:n], dtype=np.bytes_))
        f.attrs.update({"SUBSET_YMIN": 0, "SUBSET_YMAX": L, "SUBSET_XMIN": 0, "SUBSET_XMAX": W})
    r = L if rows_written is None else rows_written
    with h5py.File(ps, "a") as f:
        f["phase"][:, :r, :] = value
        f["temporalCoherence"][:, :r, :] = 0.6
    return ps


def _slc_stack(cfg, n=5, L=300, W=300, partial_last=False):
    f = mb.work_dir(cfg) / "inputs" / "slcStack.h5"
    f.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1)
    with h5py.File(f, "w") as h:
        ds = h.create_dataset("slc", shape=(n, L, W), chunks=(1, 64, 64), dtype=np.complex64)
        h["date"] = np.array([d.encode() for d in DATES[:n]])
        for i in range(n):
            a = (rng.normal(size=(L, W)) + 1j).astype(np.complex64)
            if partial_last and i == n - 1:
                ds[i, :L - 40, :] = a[:L - 40]     # killed inside the last date: centre written, last rows not
            else:
                ds[i] = a
    return f


# ------------------------------------------------- phaselinked-1 / ops-14: completion
def test_interrupted_concatenation_is_not_done_and_patches_kept(tmp_path):
    cfg = _cfg(tmp_path)
    patches = _patches(cfg)
    _phase_series(cfg, rows_written=200)                 # half the blocks written: centre probe sees data
    slc = _slc_stack(cfg)
    assert mb.step_done(cfg, "concatenate_patches") is False
    assert mb.step_done(cfg, "phase_linking") is True    # every patch flagged: only concatenation re-runs
    with pytest.raises(RuntimeError, match="not a complete concatenation"):
        mb.delete_patches_after_concatenate(cfg)
    assert patches.exists()
    with pytest.raises(RuntimeError):
        mb.delete_slc_stack_after_concatenate(cfg)
    assert slc.exists()
    scan = mb.scan_phase_series(mb.work_dir(cfg))
    assert not scan["ok"] and scan["unwritten_fraction"] == pytest.approx(0.5)


def test_incomplete_series_does_not_mask_a_missing_slc_stack(tmp_path):
    cfg = _cfg(tmp_path)
    _phase_series(cfg, rows_written=100)
    assert mb.step_done(cfg, "load_data") is False       # used to be True on the file's existence


def test_concatenation_marker_after_a_clean_step_then_cleanup_allowed(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    patches = _patches(cfg)
    _slc_stack(cfg)
    tpl = mb.work_dir(cfg) / "tpl.cfg"; tpl.write_text("x = 1\n")

    def fake_run(cfg_, args, log_name):
        _phase_series(cfg_)                              # MiaplPy writes every block
        (mb.work_dir(cfg_) / log_name).write_text("Concatenate block 3/4\nSuccessfully concatenated\n")
        return 0
    monkeypatch.setattr(mb, "_run", fake_run)
    mb.run_step(cfg, tpl, "concatenate_patches")
    inv = mb.work_dir(cfg) / "inverted"
    marker = json.loads((inv / mb.CONCAT_MARKER).read_text())
    assert marker["unwritten_cells"] == 0 and marker["dates"] == 5 and "full scan" in marker["source"]
    assert (mb.work_dir(cfg) / ".status" / "concatenate_patches.done.json").exists()
    fp = json.loads(mb.fingerprint_path(cfg).read_text())
    assert fp["status"] == "complete" and fp["dates"] == DATES and fp["time_lag"] == cfg.miaplpy_time_lag
    assert mb.step_done(cfg, "concatenate_patches") is True
    assert mb.delete_patches_after_concatenate(cfg) > 0 and not patches.exists()
    assert "verified complete" in (inv / "PATCHES.cleaned").read_text()


def test_stale_marker_is_removed_when_concatenation_restarts_and_fails(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    patches = _patches(cfg)
    _phase_series(cfg)
    inv = mb.work_dir(cfg) / "inverted"
    assert mb.phase_series_complete(mb.work_dir(cfg))   # writes the marker
    assert (inv / mb.CONCAT_MARKER).exists()
    time.sleep(0.01)
    _phase_series(cfg, rows_written=0)                   # MiaplPy restarted the concatenation, then was killed
    assert mb.step_done(cfg, "concatenate_patches") is False   # the old marker does not vouch for the new file
    tpl = mb.work_dir(cfg) / "tpl.cfg"; tpl.write_text("x = 1\n")
    (inv / mb.CONCAT_MARKER).write_text(json.dumps({"bytes": -1}))      # a marker left from earlier

    def killed(cfg_, args, log_name):
        assert not (inv / mb.CONCAT_MARKER).exists()     # gone BEFORE MiaplPy starts
        _phase_series(cfg_, rows_written=50)
        (mb.work_dir(cfg_) / log_name).write_text("Concatenate block 0/4\n")
        return 137
    monkeypatch.setattr(mb, "_run", killed)
    with pytest.raises(RuntimeError, match="rc=137"):
        mb.run_step(cfg, tpl, "concatenate_patches")
    assert not (inv / mb.CONCAT_MARKER).exists() and patches.exists()


def test_rc0_concatenation_with_unwritten_blocks_fails_and_keeps_patches(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    patches = _patches(cfg)
    tpl = mb.work_dir(cfg) / "tpl.cfg"; tpl.write_text("x = 1\n")
    monkeypatch.setattr(mb, "_run", lambda c, a, n: (_phase_series(c, rows_written=300),
                                                     (mb.work_dir(c) / n).write_text("ok\n"), 0)[-1])
    with pytest.raises(RuntimeError, match="never written"):
        mb.run_step(cfg, tpl, "concatenate_patches")
    assert not (mb.work_dir(cfg) / "inverted" / mb.CONCAT_MARKER).exists() and patches.exists()


def test_legacy_complete_series_gets_its_marker_from_a_scan(tmp_path):
    cfg = _cfg(tmp_path)
    _phase_series(cfg)
    assert mb.step_done(cfg, "concatenate_patches") is True
    m = json.loads((mb.work_dir(cfg) / "inverted" / mb.CONCAT_MARKER).read_text())
    assert "resume" in m["source"] and m["unwritten_cells"] == 0


def test_interrupted_load_is_not_skipped_and_says_what_to_delete(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    _slc_stack(cfg, partial_last=True)
    assert mb.step_done(cfg, "load_data") is False       # used to be True: shape >= 2
    ran = []
    monkeypatch.setattr(mb, "_run", lambda *a: ran.append(a) or 0)
    tpl = mb.work_dir(cfg) / "tpl.cfg"; tpl.write_text("x = 1\n")
    with pytest.raises(RuntimeError, match="Delete that file and restart 'Phase linking onward'"):
        mb.run_step(cfg, tpl, "load_data")
    assert not ran                                       # MiaplPy would not refill the hole: not run
    with h5py.File(mb.work_dir(cfg) / "inputs" / "slcStack.h5", "r") as f:
        assert mb._empty_slc_dates(f) == [DATES[4]]


def test_clean_load_writes_its_marker_and_legacy_stack_is_checked_once(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    tpl = mb.work_dir(cfg) / "tpl.cfg"; tpl.write_text("x = 1\n")

    def load(cfg_, args, log_name):
        _slc_stack(cfg_)
        (mb.work_dir(cfg_) / "inputs" / "geometryRadar.h5").write_bytes(b"x" * 10)
        (mb.work_dir(cfg_) / log_name).write_text("loaded\n")
        return 0
    monkeypatch.setattr(mb, "_run", load)
    mb.run_step(cfg, tpl, "load_data")
    rec = json.loads((mb.work_dir(cfg) / ".status" / "load_data.done.json").read_text())
    assert rec["slc_stack_bytes"] == (mb.work_dir(cfg) / "inputs" / "slcStack.h5").stat().st_size
    assert rec["template_sha1"] and mb.step_done(cfg, "load_data")
    # a run from before the markers: a complete stack is accepted once checked
    (mb.work_dir(cfg) / ".status" / "load_data.done.json").unlink()
    assert mb.step_done(cfg, "load_data")
    assert "resume" in json.loads((mb.work_dir(cfg) / ".status" / "load_data.done.json").read_text())["source"]


def test_delete_merged_slcs_error_names_the_remedy(tmp_path):
    cfg = _cfg(tmp_path)
    f = _slc_stack(cfg)
    with h5py.File(f, "r+") as h:
        h["slc"][1] = np.nan
    with pytest.raises(RuntimeError, match="Delete that file and restart 'Phase linking onward'"):
        mb.delete_merged_slcs_after_load(cfg)
    f.unlink()
    assert mb.delete_merged_slcs_after_load(cfg) == 0.0  # deleted after concatenation: nothing to do, no crash


def test_resume_guard_stops_when_series_is_incomplete_and_patches_are_gone(tmp_path):
    cfg = _cfg(tmp_path)
    _phase_series(cfg, rows_written=200)
    (mb.work_dir(cfg) / "inverted" / "PATCHES.cleaned").write_text("old code\n")
    with pytest.raises(RuntimeError, match="Nothing was deleted"):
        mb._resume_guard(cfg)
    assert (mb.work_dir(cfg) / "inverted" / "phase_series.h5").exists()


# ------------------------------------------------- phaselinked-2: linking fingerprint
def _complete_with_fingerprint(cfg):
    _patches(cfg)
    _phase_series(cfg)
    mb.write_linking_fingerprint(cfg, "complete", "test", dates=DATES)


@pytest.mark.parametrize("field,value", [("miaplpy_phase_linking_method", "EMI"), ("miaplpy_ministack_size", 13),
                                         ("miaplpy_time_lag", 5), ("miaplpy_range_window", 21)])
def test_linking_settings_change_after_phase_series_is_refused(tmp_path, field, value):
    cfg = _cfg(tmp_path)
    _complete_with_fingerprint(cfg)
    assert mb.step_done(cfg, "phase_linking") is True    # unchanged config: resumes
    setattr(cfg, field, value)
    for step in ("phase_linking", "concatenate_patches"):
        with pytest.raises(RuntimeError, match=r"re-link: delete the folder .*inverted"):
            mb.step_done(cfg, step)


def test_dates_added_after_linking_are_refused(tmp_path):
    cfg = _cfg(tmp_path)
    _complete_with_fingerprint(cfg)
    d = cfg.stack_dir / "merged" / "SLC" / "20230305"; d.mkdir(parents=True)
    (d / "20230305.slc.full").write_bytes(b"x")
    with pytest.raises(RuntimeError, match=r"dates: 1 date\(s\) on disk were not in the linking"):
        mb.step_done(cfg, "concatenate_patches")


def test_legacy_series_fingerprint_comes_from_miaplpys_log_not_the_config(tmp_path):
    """A series linked before the fingerprint, with the config edited since: the
    log says time_lag 3, the config now 5 - the change must be caught, not
    recorded as if the new value had made the series."""
    cfg = _cfg(tmp_path, miaplpy_time_lag=5)
    _patches(cfg)
    _phase_series(cfg)
    (mb.work_dir(cfg) / "miaplpy_phase_linking.log").write_text(
        "x * phase_linking.py --work_dir /w --range_window 15 --azimuth_window 15 --patch_size 200 "
        "--method sequential_EMI --test ks --num_worker 20 --mini_stack_size 10 --time_lag 3 --ps_num_shp 10\n")
    with pytest.raises(RuntimeError, match="time_lag: linked with 3, config now 5"):
        mb.step_done(cfg, "concatenate_patches")
    fp = json.loads(mb.fingerprint_path(cfg).read_text())
    assert fp["time_lag"] == 3 and fp["subset_yx"] == [0, 400, 0, 400] and "measured" in fp["source"]


def test_partly_linked_patches_with_changed_settings_are_refused(tmp_path):
    cfg = _cfg(tmp_path)
    _patches(cfg, flagged=2)
    mb._before_step(cfg, "phase_linking")                # fingerprint written as linking starts
    assert json.loads(mb.fingerprint_path(cfg).read_text())["status"] == "linking"
    cfg.miaplpy_time_lag = 5
    with pytest.raises(RuntimeError, match="re-link"):
        mb.step_done(cfg, "phase_linking")


def test_patches_linked_before_the_fingerprint_are_judged_by_miaplpys_log(tmp_path):
    """Half the patches linked by the old code with time_lag 3, the config now
    5: the fingerprint must come from what ran, and the resume must stop."""
    cfg = _cfg(tmp_path, miaplpy_time_lag=5)
    _patches(cfg, flagged=2)
    _slc_stack(cfg)
    (mb.work_dir(cfg) / "miaplpy_phase_linking.log").write_text(
        "x * phase_linking.py --work_dir /w --range_window 15 --azimuth_window 15 --patch_size 200 "
        "--method sequential_EMI --mini_stack_size 10 --time_lag 3 --index 0\n")
    with pytest.raises(RuntimeError, match="time_lag: linked with 3, config now 5"):
        mb.step_done(cfg, "phase_linking")
    fp = json.loads(mb.fingerprint_path(cfg).read_text())
    assert fp["status"] == "linking" and fp["time_lag"] == 3 and fp["dates"] == DATES


def test_datum_gate_reads_the_linked_method_not_the_config(tmp_path):
    rng = np.random.default_rng(3)
    n, L, W, size = 30, 30, 66, 10
    stack = tmp_path / "stack"; (stack / "merged").mkdir(parents=True)
    bad = _series(n, L, W, size, rng, datum_noise=True)
    dates = [f"2021{1 + k // 28:02d}{1 + k % 28:02d}" for k in range(n)]
    cfg = types.SimpleNamespace(stack_dir=stack, stack_azimuth_looks=3, stack_range_looks=11,
                                miaplpy_ministack_size=13, miaplpy_phase_linking_method="EMI", miaplpy_time_lag=3)
    with pytest.raises(RuntimeError, match="own datum") as e:
        mb.check_datum_consistency(cfg, bad, dates, linked={"method": "sequential_EMI", "ministack_size": 10})
    msg = str(e.value)
    assert "miaplpy_time_lag" in msg and "check_miaplpy_patch.py" in msg and "Do not switch" in msg
    assert "Re-run phase linking with miaplpy_phase_linking_method: EMI" not in msg
    rec = json.loads((stack / "merged" / "linked_datum_check.json").read_text())
    assert rec["method"] == "sequential_EMI" and rec["config_method"] == "EMI" and rec["mini_stack_size"] == 10


def _export_fixture(tmp_path, monkeypatch):
    stack = tmp_path / "stack"; (stack / "merged" / "SLC" / "20230221").mkdir(parents=True)
    (stack / "crop_window.json").write_text(json.dumps({"r0": 1, "r1": 3, "c0": 2, "c1": 4, "length": 5, "width": 8}))
    (stack / "merged" / "SLC" / "20230221" / "20230221.slc.full.xml").write_text(
        '<property name="width"><value>16</value></property><property name="length"><value>10</value></property>')
    cfg = types.SimpleNamespace(project_dir=tmp_path, stack_dir=stack, stack_azimuth_looks=2, stack_range_looks=2,
                                stack_reference_date="20230221", min_temporal_coherence=0.4,
                                miaplpy_network_connections=0, miaplpy_ministack_size=10, miaplpy_time_lag=3,
                                miaplpy_phase_linking_method="EMI", miaplpy_range_window=15, miaplpy_azimuth_window=15)
    work = tmp_path / "miaplpy" / "inverted"; work.mkdir(parents=True)
    dates = ["20230104", "20230221", "20230305"]
    with h5py.File(work / "phase_series.h5", "w") as f:
        f["date"] = np.array([d.encode() for d in dates]); f["phase"] = np.zeros((3, 4, 4), np.float32)
        f["temporalCoherence"] = np.full((2, 4, 4), 0.6, np.float32)
    written = []
    import insar_pipeline.isceenv as ie
    monkeypatch.setattr(ie, "import_isce", lambda: (None, None))
    monkeypatch.setattr(mb, "_write_isce", lambda isceobj, path, arr, dtype: (written.append(path.parent.name + "/" + path.name),
                                                                               path.write_bytes(b"x" * 8)))
    pdir = stack / "merged" / "interferograms" / "20230104_20230221"; pdir.mkdir(parents=True)
    for nm in ("filt_fine.int", "filt_fine.cor", "filt_fine.unw", "filt_fine.unw.conncomp", "unwrap_crop.json"):
        (pdir / nm).write_bytes(b"old")
    return cfg, pdir, written


def test_export_rewrites_and_drops_stale_unwraps_when_the_series_changed(tmp_path, monkeypatch):
    cfg, pdir, written = _export_fixture(tmp_path, monkeypatch)
    out = cfg.stack_dir / "merged" / "interferograms"
    (out / mb.EXPORT_MANIFEST).write_text(json.dumps({"fingerprint_sha": "0123456789abcdef"}))
    mb.export_phase_linked_ifgrams(cfg)
    assert not (pdir / "filt_fine.unw").exists() and not (pdir / "filt_fine.unw.conncomp").exists()
    assert not (pdir / "unwrap_crop.json").exists()
    assert "20230104_20230221/filt_fine.int" in written                 # rewritten, not reused
    assert json.loads((out / mb.EXPORT_MANIFEST).read_text())["fingerprint_sha"] != "0123456789abcdef"


def test_export_after_a_relink_with_identical_settings_is_rewritten(tmp_path, monkeypatch):
    """Same settings and dates, new series (inverted/ deleted and linked again):
    the content probe, not the settings, ties the pairs to the series."""
    cfg, pdir, written = _export_fixture(tmp_path, monkeypatch)
    mb.export_phase_linked_ifgrams(cfg)                       # adopts the legacy pairs, writes the manifest
    assert (pdir / "filt_fine.unw").exists()
    with h5py.File(tmp_path / "miaplpy" / "inverted" / "phase_series.h5", "r+") as f:
        f["phase"][0] = 0.7                                   # the re-linked series differs
    written.clear()
    mb.export_phase_linked_ifgrams(cfg)
    assert not (pdir / "filt_fine.unw").exists() and "20230104_20230221/filt_fine.int" in written
    written.clear()
    mb.export_phase_linked_ifgrams(cfg)                       # unchanged series: nothing rewritten
    assert "20230104_20230221/filt_fine.int" not in written


def test_export_without_manifest_adopts_the_existing_pairs(tmp_path, monkeypatch):
    cfg, pdir, written = _export_fixture(tmp_path, monkeypatch)
    mb.export_phase_linked_ifgrams(cfg)
    assert (pdir / "filt_fine.unw").read_bytes() == b"old"                # a legacy export is not re-unwrapped
    assert "20230104_20230221/filt_fine.int" not in written
    assert (cfg.stack_dir / "merged" / "interferograms" / mb.EXPORT_MANIFEST).exists()


# ------------------------------------------------- phaselinked-5: datum gate on coherent cells
def _series(n, L, W, size, rng, datum_noise=False, decorrelated=None):
    """Smooth true phase per date; optionally a random per-pixel datum per
    mini-stack (the upstream defect); optionally cells whose phase is random
    every date (ground that keeps no coherence: a mine, crops)."""
    yy, xx = np.mgrid[0:L, 0:W]
    ph = np.zeros((n, L, W), np.float32)
    offs = {}
    for k in range(n):
        ph[k] = 0.002 * k * xx + 0.001 * k * yy + rng.normal(scale=0.05, size=(L, W))
        if datum_noise:
            ms = min(k // size, max(1, n // size) - 1)
            if ms not in offs:
                offs[ms] = rng.uniform(-np.pi, np.pi, size=(L, W)).astype(np.float32)
            ph[k] += offs[ms]
        if decorrelated is not None:
            ph[k][decorrelated] = rng.uniform(-np.pi, np.pi, size=int(decorrelated.sum()))
    return ph


def _gate_cfg(stack, method="sequential_EMI", size=10):
    return types.SimpleNamespace(stack_dir=stack, stack_azimuth_looks=3, stack_range_looks=11,
                                 miaplpy_ministack_size=size, miaplpy_phase_linking_method=method)


def test_gate_passes_a_healthy_series_on_60_percent_decorrelated_ground(tmp_path):
    """The old gate (median over every cell of the central block, floor 0.5)
    failed this series, which has no datum error at all."""
    rng = np.random.default_rng(5)
    n, L, W, size = 30, 60, 220, 10
    stack = tmp_path / "stack"; (stack / "merged").mkdir(parents=True)
    dec = np.zeros((L, W), bool); dec[:, : int(0.6 * W) // 11 * 11] = True      # 60% of the cells, look-aligned
    ph = _series(n, L, W, size, rng, decorrelated=dec)
    dates = [f"2021{1 + k // 28:02d}{1 + k % 28:02d}" for k in range(n)]
    info = mb.check_datum_consistency(_gate_cfg(stack), ph, dates)
    assert info["passed"] and info["testable"]
    assert info["median_block_coherence_across"] < 0.5          # every cell: what the old gate tripped on
    assert info["coherent_cells"]["median_ratio"] > 0.9 and info["coherent_cells"]["boundaries_judged"] == 2


def test_gate_still_fails_the_upstream_defect_with_decorrelated_ground(tmp_path):
    rng = np.random.default_rng(6)
    n, L, W, size = 30, 60, 220, 10
    stack = tmp_path / "stack"; (stack / "merged").mkdir(parents=True)
    dec = np.zeros((L, W), bool); dec[:, :110] = True
    ph = _series(n, L, W, size, rng, datum_noise=True, decorrelated=dec)
    dates = [f"2021{1 + k // 28:02d}{1 + k % 28:02d}" for k in range(n)]
    with pytest.raises(RuntimeError, match="own datum"):
        mb.check_datum_consistency(_gate_cfg(stack), ph, dates)
    rec = json.loads((stack / "merged" / "linked_datum_check.json").read_text())
    assert rec["coherent_cells"]["median_ratio"] < 0.4 and not rec["passed"]


def test_gate_samples_the_whole_crop_not_a_central_block(tmp_path):
    rng = np.random.default_rng(7)
    n, L, W = 22, 600, 2200
    stack = tmp_path / "stack"; (stack / "merged").mkdir(parents=True)
    tiles = mb._gate_tiles(L, W, 3, 11)
    assert len(tiles) == 16 and min(t[0] for t in tiles) == 0 and max(t[1] for t in tiles) == L
    assert min(t[2] for t in tiles) == 0 and max(t[3] for t in tiles) == W
    assert all((t[1] - t[0]) % 3 == 0 and (t[3] - t[2]) % 11 == 0 for t in tiles)
    ph = np.zeros((n, L, W), np.float32) + rng.normal(scale=0.05, size=(1, L, W)).astype(np.float32)
    info = mb.check_datum_consistency(_gate_cfg(stack), ph, [f"2021{1 + k // 28:02d}{1 + k % 28:02d}" for k in range(n)])
    assert info["cells_sampled"] == 16 * 50 * 50 and info["passed"]


# ------------------------------------------------- phaselinked-8: progress and stall detection
def test_patch_count_matches_miaplpys_grid_on_site_b():
    assert mb.patch_count(2604, 9240, 15, 15, 200) == 611     # what Site B's concatenation wrote
    assert mb.patch_count(400, 400, 15, 15, 200) == 4


def _stub(monkeypatch, code):
    monkeypatch.setattr(mb, "_command_argv", lambda cfg, args: [sys.executable, "-c", code])
    monkeypatch.setattr(procs, "install_teardown", lambda: None)
    procs.clear_stopping()


def test_phase_linking_reports_patch_progress(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path, miaplpy_num_processors=2)
    monkeypatch.setattr(mb, "POLL_S", 0.05)
    inv = mb.work_dir(cfg) / "inverted"
    _stub(monkeypatch, "import os, time, sys\n"
                       f"for k in range(4):\n"
                       f"    d = os.path.join({str(inv)!r}, 'PATCHES', 'PATCH_%04d' % k); os.makedirs(d)\n"
                       "    time.sleep(0.15); open(os.path.join(d, 'flag.npy'), 'wb').write(b'1')\n"
                       "    print('patch', k, flush=True)\n")
    rc = mb._run(cfg, ["tpl", "--dir", "w", "--dostep", "phase_linking"], "miaplpy_phase_linking.log")
    assert rc == 0
    ev = [e for e in progress.read(cfg.project_dir) if e["stage"] == "miaplpy_phase_linking"]
    assert ev and all(e["total"] == 4 for e in ev) and max(e["done"] for e in ev) >= 2
    assert "miaplpy_phase_linking.log" in ev[-1]["msg"]


def test_a_stalled_step_is_stopped_with_a_plain_reason(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(mb, "POLL_S", 0.05)
    monkeypatch.setattr(mb, "STALL_FLOOR_S", 0.4)
    pidf = tmp_path / "pid"
    _stub(monkeypatch, f"import os, time\nopen({str(pidf)!r}, 'w').write(str(os.getpid()))\n"
                       "print('Reading SLC data ...', flush=True)\ntime.sleep(60)\n")
    t0 = time.time()
    with pytest.raises(RuntimeError, match="made no progress .* stopped") as e:
        mb._run(cfg, ["tpl", "--dir", "w", "--dostep", "phase_linking"], "miaplpy_phase_linking.log")
    assert time.time() - t0 < 10
    assert "lower miaplpy_num_processors" in str(e.value) and "Reading SLC data" in str(e.value)
    pid = int(pidf.read_text())
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)                                   # the hung child is gone, not orphaned


# ------------------------------------------------- tests-2 / generality-8 / gap-surface_mine_site-4
LAT0, LON0, CELL_M = 38.40, -87.30, 50.0
DLAT = CELL_M / 1000 / 110.57
DLON = CELL_M / 1000 / (111.32 * np.cos(np.radians(LAT0)))


def _grid_latlon(L=400, W=400):
    rows, cols = np.mgrid[0:L, 0:W]
    return LAT0 - rows * DLAT, LON0 + cols * DLON


def _rect(r0, r1, c0, c1):
    """A builder-style rectangle: 5 vertices (closed ring), edges of several km."""
    la = lambda r: float(LAT0 - r * DLAT)       # noqa: E731
    lo = lambda c: float(LON0 + c * DLON)       # noqa: E731
    return [[lo(c0), la(r0)], [lo(c1), la(r0)], [lo(c1), la(r1)], [lo(c0), la(r1)], [lo(c0), la(r0)]]


def _reference_fixture(tmp_path, monkeypatch, rings, bright=(slice(203, 208), slice(198, 203)), **over):
    stack = tmp_path / "stack"; G = stack / "merged" / "geom_reference"; G.mkdir(parents=True)
    L = W = 400
    lat, lon = _grid_latlon(L, W)
    lat.astype(np.float64).tofile(G / "lat.rdr"); lon.astype(np.float64).tofile(G / "lon.rdr")
    (stack / "crop_window.json").write_text(json.dumps({"r0": 0, "r1": L, "c0": 0, "c1": W, "length": L, "width": W}))
    pdir = stack / "merged" / "interferograms" / "20230104_20230116"; pdir.mkdir(parents=True)
    np.ones((L, W), np.uint8).tofile(pdir / "filt_fine.unw.conncomp")
    gj = tmp_path / "mines.geojson"
    gj.write_text(json.dumps({"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {}, "geometry": {"type": "Polygon", "coordinates": [r]}} for r in rings]}))
    tc = np.full((L, W), 0.5, np.float32); tc[bright] = 0.99
    monkeypatch.setattr(mb, "load_linked_temporal_coherence", lambda cfg, layer=None: tc)
    cfg = types.SimpleNamespace(stack_dir=stack, reference_lalo="", bbox=(lon.min(), lat.min(), lon.max(), lat.max()),
                                deramp_exclude_geojson=str(gj), acceptance_far_field_buffer_m=2000.0)
    for k, v in over.items():
        setattr(cfg, k, v)
    return cfg, gj


def _edge_km(gj, lat, lon):
    """Independent of the code under test: geodesic (WGS84) distance from the
    point to its nearest mine edge, the edges sampled every ~5 m. The code
    measures in a local plane scaled at the crop's mean latitude, good to a
    few metres over a 20 km crop - hence the 10 m tolerance (FAR_KM) below."""
    from pyproj import Geod
    g = Geod(ellps="WGS84")
    best = np.inf
    for f in json.loads(Path(gj).read_text())["features"]:
        ring = np.asarray(f["geometry"]["coordinates"][0], float)
        for (x0, y0), (x1, y1) in zip(ring[:-1], ring[1:]):
            n = max(2, int(g.inv(x0, y0, x1, y1)[2] / 5) + 1)
            xs, ys = np.linspace(x0, x1, n), np.linspace(y0, y1, n)
            best = min(best, float(np.min(g.inv(np.full(n, lon), np.full(n, lat), xs, ys)[2])))
    return best / 1000


FAR_KM = 2.0 - 0.01


def test_auto_reference_respects_the_edge_buffer(tmp_path, monkeypatch):
    cfg, gj = _reference_fixture(tmp_path, monkeypatch, [_rect(100, 200, 100, 300)])
    info = mb.choose_reference(cfg, ["20230104_20230116"])
    ch = info["chosen"]
    assert _edge_km(gj, ch["lat"], ch["lon"]) >= FAR_KM             # the old vertex rule chose 0.25 km
    assert ch["distance_to_nearest_mapped_mine_km"] >= 2.0 and ch["far_field_buffer_m"] == 2000.0
    assert "> 2 km from the edge" in ch["rule"]
    rec = json.loads((cfg.stack_dir / "reference_choice.json").read_text())
    assert rec["chosen"]["distance_to_nearest_mapped_mine_km"] == ch["distance_to_nearest_mapped_mine_km"]


def test_a_mine_just_outside_the_crop_still_excludes_nearby_pixels(tmp_path, monkeypatch):
    # a tall strip 0.5 km west of the grid: its vertices are 10+ km from the bright block beside it
    cfg, gj = _reference_fixture(tmp_path, monkeypatch, [_rect(-200, 600, -40, -10)],
                                 bright=(slice(198, 203), slice(5, 10)))
    ch = mb.choose_reference(cfg, ["20230104_20230116"])["chosen"]
    assert _edge_km(gj, ch["lat"], ch["lon"]) >= FAR_KM            # the old rule chose 1.4 km


def test_far_field_buffer_comes_from_the_config(tmp_path, monkeypatch):
    cfg, gj = _reference_fixture(tmp_path, monkeypatch, [_rect(100, 200, 100, 300)],
                                 bright=(slice(250, 255), slice(198, 203)), acceptance_far_field_buffer_m=4000.0)
    ch = mb.choose_reference(cfg, ["20230104_20230116"])["chosen"]
    assert _edge_km(gj, ch["lat"], ch["lon"]) >= 4.0 - 0.01 and ch["far_field_buffer_m"] == 4000.0


def test_a_kept_automatic_reference_inside_the_buffer_is_chosen_again(tmp_path, monkeypatch):
    cfg, gj = _reference_fixture(tmp_path, monkeypatch, [_rect(100, 200, 100, 300)])
    lat, lon = _grid_latlon()
    near = f"{lat[205, 200]:.6f},{lon[205, 200]:.6f}"          # 0.25 km from the long edge
    cfg.reference_lalo, cfg.reference_auto = near, True
    info = mb.choose_reference(cfg, ["20230104_20230116"])
    assert info["kept_configured"] is False and info["configured"]["distance_to_nearest_mapped_mine_km"] < 0.5
    assert info["chosen"]["distance_to_nearest_mapped_mine_km"] >= 2.0 and cfg.reference_lalo != near
    # the same point set in the YAML is kept (deliberate), with its measured distance on record
    cfg.reference_lalo, cfg.reference_auto = near, False
    info = mb.choose_reference(cfg, ["20230104_20230116"])
    assert info["kept_configured"] is True and info["chosen"]["distance_to_nearest_mapped_mine_km"] < 0.5


# ------------------------------------------------- stackrunner-8: failed unwraps
def test_a_failed_reunwrap_leaves_no_previous_unwrap_behind(tmp_path, monkeypatch):
    pdir = tmp_path / "20230104_20230116"; pdir.mkdir()
    L, W = 10, 12
    for nm in ("filt_fine.int", "filt_fine.cor"):
        (pdir / nm).write_bytes(b"\0" * 8)
    for nm in ("filt_fine.unw", "filt_fine.unw.xml", "filt_fine.unw.conncomp", "filt_fine.unw.conncomp.xml",
               "unwrap_crop.json"):
        (pdir / nm).write_text("previous settings")
    conf = tmp_path / "config_igram_unw_20230104_20230116"
    conf.write_text(f"[Function-1]\nunwrap : \nifg : {pdir / 'filt_fine.int'}\nunw : {pdir / 'filt_fine.unw'}\n"
                    f"coh : {pdir / 'filt_fine.cor'}\nengine : snaphu2\ncost : SMOOTH\n")
    win = tmp_path / "crop_window.json"
    win.write_text(json.dumps({"r0": 2, "r1": 8, "c0": 3, "c1": 9, "length": L, "width": W}))
    img = lambda p: types.SimpleNamespace(length=L, width=W, bands=1, dataType="CFLOAT", scheme="BIL",   # noqa: E731
                                          filename=str(p))
    monkeypatch.setattr(uc, "_load_image", img)
    monkeypatch.setattr(uc, "_read_raw", lambda p, i: np.ones((L, W), np.complex64))
    monkeypatch.setattr(uc, "_write_like", lambda tpl, out, arr: Path(out).write_bytes(b"crop"))

    def boom(*a, **k):
        raise MemoryError("snaphu 2 ran out of memory")
    monkeypatch.setattr(uc, "_run_snaphu2", boom)
    with pytest.raises(MemoryError):
        uc.run_pair(conf, win)
    left = sorted(p.name for p in pdir.iterdir())
    assert not any(n.startswith("filt_fine.unw") for n in left) and "unwrap_crop.json" not in left


def test_unwrap_exported_quarantines_a_pair_whose_rerun_failed(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    ifg = cfg.stack_dir / "merged" / "interferograms"
    pairs = ["20230104_20230221", "20230116_20230221"]
    N = 6
    for p in pairs:
        d = ifg / p; d.mkdir(parents=True)
        (d / "filt_fine.int").write_bytes(b"\0" * 8 * N)
        (d / "filt_fine.cor").write_bytes(b"\0" * 4 * N)
        (d / "filt_fine.unw").write_bytes(b"\0" * 8 * N)          # an unwrap from the PREVIOUS settings
        (d / "filt_fine.unw.conncomp").write_bytes(b"\0" * N)

    def fake_execute(cfg_, rf, fail_fast):
        cmds = sb.parse_run_file(rf)
        good = [c for c in cmds if pairs[0] in c][0]
        m = sb._status_path(cfg_, rf.name, good); m.parent.mkdir(parents=True, exist_ok=True)
        m.write_text(json.dumps({"cmd": good, "returncode": 0, "config_sha1": sb._config_sha1(good)}))
        return {"run_file": rf.name, "n": len(cmds), "failed": [c for c in cmds if c != good]}
    monkeypatch.setattr(sb, "execute_run_file", fake_execute)
    with pytest.raises(RuntimeError, match=r"1 of 2 phase-linked pairs did not unwrap: \['20230116_20230221'\]"):
        mb.unwrap_exported(cfg, pairs)
    q = cfg.stack_dir / "quarantine_unwrap_timeout" / pairs[1]
    assert (q / "filt_fine.unw").exists() and not (ifg / pairs[1] / "filt_fine.unw").exists()
    assert (ifg / pairs[0] / "filt_fine.unw").exists()           # the good pair stays


# ------------------------------------------------- patch check with the configured window overlap
def test_patch_check_is_run_with_the_configured_ministack_and_overlap(monkeypatch):
    seen = {}

    def fake(argv, **kw):
        seen["cmd"] = argv[-1]
        out = json.dumps({"patched": False, "ratio": 21.95, "ministack": 10, "time_lag": 0,
                          "miaplpy_version": "0.2.1+overlapdatum"})
        return subprocess.CompletedProcess(argv, 1, out + "\n", "")
    monkeypatch.setattr(mb.subprocess, "run", fake)
    with pytest.raises(RuntimeError, match="does NOT tie the windows together .*miaplpy_time_lag 0"):
        mb.verify_patched_miaplpy("miaplpy", ministack=10, time_lag=0)
    assert seen["cmd"].endswith("check_miaplpy_patch.py --ministack 10 --time-lag 0")


def test_patch_check_measures_the_configured_overlap_in_the_miaplpy_env():
    from insar_pipeline.settings import conda_sh
    env = Path(conda_sh()).parents[2] / "envs" / "miaplpy"
    if not env.exists():
        pytest.skip("no miaplpy env on this machine")
    out = {}
    for lag in (0, 3):
        r = subprocess.run(["bash", "-c", f"source {conda_sh()} && conda activate miaplpy && cd / && "
                            f"python {mb.PATCH_CHECK} --ministack 10 --time-lag {lag}"],
                           capture_output=True, text=True, timeout=120)
        out[lag] = json.loads([ln for ln in r.stdout.splitlines() if ln.startswith("{")][-1])
    assert out[3]["patched"] and out[3]["time_lag"] == 3
    assert not out[0]["patched"] and out[0]["ratio"] > 2.0       # no shared images: the default check never saw this


# ------------------------------------------------- reunwrap_linked.py: teardown + lock
def _load_script():
    spec = importlib.util.spec_from_file_location(
        "reunwrap_linked_under_test", Path(__file__).resolve().parents[1] / "scripts" / "reunwrap_linked.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


def test_reunwrap_installs_teardown_first_and_runs_under_the_shared_launcher(tmp_path, monkeypatch):
    import insar_pipeline.config as config_mod
    import insar_pipeline.pipeline as pipeline_mod
    calls = []
    monkeypatch.setattr(procs, "install_teardown", lambda: calls.append("teardown"))
    monkeypatch.setattr(config_mod, "load_config",
                        lambda p: calls.append("config") or types.SimpleNamespace(backend="miaplpy", project_dir=tmp_path))

    def launch(path, kind, before=None, **kw):
        calls.append(("launch", kind, getattr(before, "__name__", None)))
        return 0
    monkeypatch.setattr(pipeline_mod, "launch", launch)
    mod = _load_script()
    assert mod.main(["x.yaml"]) == 0
    assert calls[0] == "teardown" and ("launch", "reunwrap", "keep_previous") in calls


def test_reunwrap_has_no_chain_of_its_own_and_copies_no_series(tmp_path):
    # INT1: pipeline.launch exists in every build now, so the script's own copy
    # of the stage chain (and its lock fallback) is gone; the backup copies only
    # what the re-run overwrites in place - never the series, the velocity or
    # geo/, which the MintPy step moves aside itself (no fixed series name).
    mod = _load_script()
    assert not hasattr(mod, "reunwrap") and not hasattr(mod, "_guard")
    assert not any("timeseries" in r or r.startswith("geo/") or r.startswith("velocity") for r in mod.SUPERSEDED)
    cfg = types.SimpleNamespace(mintpy_dir=tmp_path / "mintpy", stack_dir=tmp_path / "stack")
    (cfg.mintpy_dir / "geo").mkdir(parents=True)
    for f in ("temporalCoherence.h5", "maskTempCoh.h5", "timeseries_ramp_demErr.h5", "velocity.h5",
              "geo/geo_velocity.h5", "unwrap_error_applied.json"):
        (cfg.mintpy_dir / f).write_bytes(b"x")
    keep = mod.keep_previous(cfg)
    got = sorted(p.relative_to(keep).as_posix() for p in keep.rglob("*") if p.is_file())
    assert got == ["README.txt", "maskTempCoh.h5", "temporalCoherence.h5", "unwrap_error_applied.json"]
