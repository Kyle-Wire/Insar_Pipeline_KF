"""Phase linking on the GPU (11 Nov 2026; pl_gpu.py, review_2026-11-10/PLAN.md
Phase 2). The kernel ports MiaplPy's setup exactly (KS threshold, patch grid,
mini-stack windows, C's cabsf), the pipeline uses it only when miaplpy_gpu
asks and a self-check against MiaplPy on the CPU passes, and in 'auto' any
failure leaves the patches to the CPU. The last test links one synthetic
patch with both kernels; it runs only where both envs and a GPU exist."""
import ctypes
import ctypes.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from insar_pipeline import miaplpy_backend as mb          # noqa: E402
from insar_pipeline import pl_gpu                          # noqa: E402
from insar_pipeline.config import PipelineConfig, validate_config  # noqa: E402

import test_fix_L6b as L6b                                 # noqa: E402  (a synthetic phase-linked project)


# ---------------------------------------------------------------- MiaplPy's setup, ported

def _ks_lut_reference(n, alpha=0.01):
    """ks_lut_cy in plain double precision (the version used for Phase 0/1)."""
    N = n * n / (2 * n)
    for d in np.arange(0.01, 1, 0.001, dtype=np.float32):
        v = d * (math.sqrt(N) + 0.12 + 0.11 / math.sqrt(N))
        p = 2 * sum(((-1) ** (t - 1)) * math.exp(-2 * v * v * t * t) for t in range(1, 101))
        if min(max(p, 0), 1) <= alpha:
            return float(d)


def test_ks_threshold_matches_miaplpy_lut():
    for n in (10, 18, 60, 279, 292, 445, 600):
        assert pl_gpu.ks_threshold(n) == _ks_lut_reference(n)
    assert pl_gpu.ks_threshold(445) == pytest.approx(0.109, abs=1e-6)
    # the integer ECDF gap allowed: float32(k / n) <= threshold, as MiaplPy stores it
    assert pl_gpu.ks_max_count(445, pl_gpu.ks_threshold(445)) == 48


def test_patch_grid_is_miaplpys():
    boxes = pl_gpu.patch_boxes(2124, 8096, 200, 15, 15)            # study_areav2
    assert len(boxes) == 451 == mb.patch_count(2124, 8096, 15, 15, 200)
    assert boxes[0] == (0, 0, 200, 200, 0) and boxes[-1] == (8000, 2000, 8096, 2124, 450)
    for L, W in [(2604, 9240), (400, 400), (230, 1000)]:
        assert len(pl_gpu.patch_boxes(L, W, 200, 15, 15)) == mb.patch_count(L, W, 15, 15, 200)


def test_windows_overlap_and_take_the_rest():
    w = pl_gpu.windows(445, "sequential_EMI", 10, 3)
    assert len(w) == 44 and w[0] == (0, 0, 10) and w[1] == (7, 10, 20) and w[-1] == (427, 430, 445)
    kept = [t for a, f, l in w for t in range(f, l)]
    assert kept == list(range(445))                                 # every date once
    assert pl_gpu.windows(9, "EMI", 10, 3) == [(0, 0, 9)]


def test_amplitude_is_cs_cabsf():
    """numpy's complex64 abs differs from glibc's cabsf/hypotf by 1 ulp on ~1/3
    of values - enough to move a KS gap on the threshold (one pixel did, on
    the first real comparison)."""
    lib = ctypes.CDLL(ctypes.util.find_library("m") or "libm.so.6")
    lib.hypotf.restype, lib.hypotf.argtypes = ctypes.c_float, [ctypes.c_float, ctypes.c_float]
    rng = np.random.default_rng(3)
    z = ((rng.standard_normal(4000) + 1j * rng.standard_normal(4000)) * 50).astype(np.complex64)
    ref = np.array([lib.hypotf(float(v.real), float(v.imag)) for v in z], np.float32)
    assert np.array_equal(pl_gpu.cabsf(z), ref)


def test_write_patch_flags_last_and_names_the_kernel(tmp_path):
    res = {"phase_ref": np.ones((3, 2, 2), np.complex64), "shp": np.ones((2, 2), np.int32),
           "tempCoh": np.zeros((2, 2, 2), np.float32), "mask_ps": np.zeros((2, 2), bool),
           "ps_products": np.zeros((4, 2, 2), np.float32)}
    d = pl_gpu.write_patch(tmp_path, 7, res, {"kernel": pl_gpu.kernel_label()})
    assert d.name == "PATCH_0007"
    names = {p.name for p in d.iterdir()}
    assert names == {f"{k}.npy" for k in pl_gpu.PATCH_FILES} | {"flag.npy", "kernel.json"}   # no temporaries left
    assert all(d.joinpath(f"{k}.npy").stat().st_mtime_ns <= d.joinpath("flag.npy").stat().st_mtime_ns
               for k in pl_gpu.PATCH_FILES)
    assert json.loads((d / "kernel.json").read_text())["kernel"].startswith("pl_gpu-")


def test_selfcheck_verdict_uses_the_gate_limits():
    good = {"shp_identical": 1.0, "ps_mask_mismatch_frac": 0.0, "phase_median_rad": 1e-6,
            "phase_p99_rad": 2e-4, "nan_mismatch": 0}
    assert pl_gpu.selfcheck_verdict(good)[0]
    for k, v in [("shp_identical", 0.99), ("phase_p99_rad", 0.06), ("phase_median_rad", 0.02), ("nan_mismatch", 3)]:
        ok, why = pl_gpu.selfcheck_verdict({**good, k: v})
        assert not ok and why


# ---------------------------------------------------------------- when the pipeline uses it

def test_config_accepts_only_the_three_modes():
    cfg = PipelineConfig(project_dir=Path("/tmp/x"), bbox=[-87.8, 38.2, -87.5, 38.4], start_date="2023-01-01",
                         end_date="2023-06-01", backend="miaplpy")
    assert cfg.miaplpy_gpu == "auto"                               # Kyle 11 Nov 2026: the GPU when it works
    for mode in ("auto", "on", "off"):
        cfg.miaplpy_gpu = mode
        assert not [p for p in validate_config(cfg) if "miaplpy_gpu" in p]
    cfg.miaplpy_gpu = "yes"
    assert [p for p in validate_config(cfg) if "miaplpy_gpu" in p]


def test_off_never_probes(tmp_path, monkeypatch):
    cfg = L6b._cfg(tmp_path, miaplpy_gpu="off")
    monkeypatch.setattr(mb, "gpu_probe", lambda c: pytest.fail("probed with miaplpy_gpu off"))
    assert mb.run_gpu_phase_linking(cfg) == {"used": False, "mode": "off", "reason": "miaplpy_gpu is off"}


@pytest.mark.parametrize("method", ["sequential_PTA", "SBW", "EVD"])
def test_unsupported_method_stays_on_the_cpu(tmp_path, monkeypatch, method):
    cfg = L6b._cfg(tmp_path, miaplpy_gpu="auto", miaplpy_phase_linking_method=method)
    monkeypatch.setattr(mb, "gpu_probe", lambda c: pytest.fail("probed for a method the GPU cannot link"))
    plan = mb.gpu_linking_plan(cfg)
    assert not plan["use"] and "not implemented on the GPU" in plan["reason"]
    cfg.miaplpy_gpu = "on"
    with pytest.raises(RuntimeError, match="miaplpy_gpu is 'on'"):
        mb.gpu_linking_plan(cfg)


def test_no_gpu_auto_falls_back_on_requires(tmp_path, monkeypatch):
    cfg = L6b._cfg(tmp_path, miaplpy_gpu="auto")
    monkeypatch.setattr(mb, "gpu_probe", lambda c: {"gpu": False, "reason": "JAX sees no GPU"})
    plan = mb.gpu_linking_plan(cfg)
    assert not plan["use"] and "JAX sees no GPU" in plan["reason"]
    cfg.miaplpy_gpu = "on"
    with pytest.raises(RuntimeError, match="no usable GPU"):
        mb.gpu_linking_plan(cfg)


def _usable(monkeypatch, ok=True):
    monkeypatch.setattr(mb, "gpu_probe", lambda c: {"gpu": True, "device": "cuda:0", "kind": "test"})
    monkeypatch.setattr(mb, "verify_gpu_kernel", lambda c: {"ok": ok, "why": "test" if ok else "phase p99 0.2 rad",
                                                            "metrics": {"shp_identical": 1.0, "phase_p99_rad": 1e-4}})


def test_failed_selfcheck_links_nothing_on_the_gpu(tmp_path, monkeypatch):
    cfg = L6b._cfg(tmp_path, miaplpy_gpu="auto")
    _usable(monkeypatch, ok=False)
    monkeypatch.setattr(mb, "_gpu_command", lambda c, a: pytest.fail("ran the GPU kernel after a failed self-check"))
    rec = mb.run_gpu_phase_linking(cfg)
    assert not rec["used"] and "self-check" in rec["reason"]
    cfg.miaplpy_gpu = "on"
    with pytest.raises(RuntimeError, match="self-check"):
        mb.run_gpu_phase_linking(cfg)


def test_gpu_failure_auto_leaves_the_rest_to_the_cpu(tmp_path, monkeypatch):
    """A kernel that links two patches and then dies: 'auto' records it and
    returns (MiaplPy links the rest), 'on' stops the run; finished patches stay."""
    cfg = L6b._cfg(tmp_path, miaplpy_gpu="auto")
    _usable(monkeypatch)
    inv = mb.work_dir(cfg) / "inverted" / "PATCHES"
    script = (f"for i in 0 1; do d={inv}/PATCH_000$i; mkdir -p $d; echo '{{}}' > $d/kernel.json; "
              f"touch $d/flag.npy; done; echo 'CUDA_ERROR_OUT_OF_MEMORY' >&2; exit 3")
    monkeypatch.setattr(mb, "_gpu_command", lambda c, a: ["bash", "-c", script])
    monkeypatch.setattr(mb, "POLL_S", 0.2)
    rec = mb.run_gpu_phase_linking(cfg)
    assert rec["used"] and rec["patches_linked"] == 2 and rec["rc"] == 3 and "rc=3" in rec["error"]
    assert mb.gpu_patch_count(mb.work_dir(cfg)) == 2
    for i in (0, 1):
        (inv / f"PATCH_000{i}" / "kernel.json").unlink()
    cfg.miaplpy_gpu = "on"
    with pytest.raises(RuntimeError, match="Finished patches are kept"):
        mb.run_gpu_phase_linking(cfg)


def test_gpu_args_follow_the_config_and_the_skip_mask(tmp_path):
    cfg = L6b._cfg(tmp_path, miaplpy_gpu="auto", miaplpy_range_window=11, miaplpy_time_lag=2)
    args = [str(a) for a in mb._gpu_args(cfg)]
    pairs = dict(zip(args[1::2], args[2::2]))
    assert pairs["--range-window"] == "11" and pairs["--azimuth-window"] == "15" and pairs["--lag"] == "2"
    assert pairs["--method"] == "sequential_EMI" and pairs["--ps-shp"] == "10" and pairs["--patch-size"] == "200"
    assert "--mask" not in pairs                                    # MiaplPy links every pixel without the file
    mb.skip_mask_path(cfg).parent.mkdir(parents=True, exist_ok=True)
    mb.skip_mask_path(cfg).write_bytes(b"")
    assert str(mb.skip_mask_path(cfg)) in [str(a) for a in mb._gpu_args(cfg)]


def test_phase_linking_step_runs_the_gpu_first(tmp_path, monkeypatch):
    cfg = L6b._cfg(tmp_path)
    calls = []
    monkeypatch.setattr(mb, "step_done", lambda c, s: False)
    monkeypatch.setattr(mb, "_before_step", lambda c, s: None)
    monkeypatch.setattr(mb, "run_gpu_phase_linking", lambda c: calls.append("gpu") or {"used": False})
    def fake_run(c, a, n):
        (mb.work_dir(c) / n).write_text("")
        calls.append("cpu")
        return 1
    monkeypatch.setattr(mb, "_run", fake_run)
    with pytest.raises(RuntimeError):
        mb.run_step(cfg, tmp_path / "t.cfg", "phase_linking")
    assert calls == ["gpu", "cpu"]
    calls.clear()
    with pytest.raises(RuntimeError):
        mb.run_step(cfg, tmp_path / "t.cfg", "load_data")
    assert calls == ["cpu"]


def test_release_ships_the_kernel():
    pkg = Path(__file__).resolve().parents[1]
    assert (pkg / "pl_gpu.py").exists()
    text = (pkg / "install" / "make_release.sh").read_text()
    assert '"*.py"' in text                                         # pl_gpu.py sits in the package root
    env = (pkg / "install" / "envs" / "dolphin.yml").read_text()
    assert "jaxlib=" in env and "cuda-version=" in env


# ---------------------------------------------------------------- both kernels on one synthetic patch

def _env_python(name):
    try:
        from insar_pipeline import settings
        p = settings.conda_base() / "envs" / name / "bin" / "python"
        return p if p.exists() else None
    except Exception:                                               # noqa: BLE001
        return None


@pytest.mark.skipif(not (_env_python("miaplpy") and _env_python("dolphin")), reason="needs the miaplpy and dolphin envs")
def test_gpu_kernel_matches_miaplpy_on_a_synthetic_patch(tmp_path):
    cfg = L6b._cfg(tmp_path, miaplpy_gpu="auto")
    probe = mb.gpu_probe(cfg)
    if not probe.get("gpu"):
        pytest.skip(f"no usable GPU: {probe.get('reason')}")
    rec = mb.verify_gpu_kernel(cfg)
    assert rec["ok"], rec
    m = rec["metrics"]
    assert m["shp_identical"] == 1.0 and m["ps_mask_mismatch_frac"] == 0.0 and m["phase_p99_rad"] < 0.005
    assert json.loads((mb.work_dir(cfg) / mb.GPU_CHECK).read_text())["ok"]
    assert not (mb.work_dir(cfg) / ".gpu_check").exists()           # its scratch is removed


# ---------------------------------------------------------------- the NISAR route's dolphin switch (Phase 3)

def _fake_dolphin_config(calls):
    import subprocess
    import yaml

    def fake(cmd, **kw):
        calls.append(cmd)
        out = Path(cmd[cmd.index("--outfile") + 1])
        out.write_text(yaml.safe_dump({"unwrap_options": {"snaphu_options": {}}, "timeseries_options": {}}))
        return subprocess.CompletedProcess(cmd, 0, "", "")
    return fake


@pytest.mark.parametrize("gpu, flag", [(True, "--worker-settings.gpu-enabled"), (False, "--worker-settings.no-gpu-enabled")])
def test_dolphin_config_states_the_gpu_switch(tmp_path, monkeypatch, gpu, flag):
    """dolphin defaults gpu_enabled to False and then hides the GPU from JAX:
    the config always says which, explicitly."""
    from insar_pipeline import nisar_dolphin as nd
    calls = []
    monkeypatch.setattr(nd.subprocess, "run", _fake_dolphin_config(calls))
    nd.write_config(tmp_path, {"files": ["a.tif", "b.tif"]}, 11, (30, 60), gpu=gpu)
    assert flag in calls[0] and sum("gpu-enabled" in c for c in calls[0]) == 1


def test_dolphin_gpu_modes(monkeypatch):
    from insar_pipeline import nisar_dolphin as nd
    monkeypatch.setattr(nd, "gpu_probe", lambda: pytest.fail("probed with the GPU off"))
    assert nd.resolve_gpu("off")["enabled"] is False
    monkeypatch.setattr(nd, "gpu_probe", lambda: {"gpu": True, "device": "cuda:0"})
    assert nd.resolve_gpu("auto") == {"enabled": True, "mode": "auto", "device": "cuda:0", "kind": None}
    monkeypatch.setattr(nd, "gpu_probe", lambda: {"gpu": False, "reason": "JAX sees no GPU"})
    assert nd.resolve_gpu("auto")["enabled"] is False
    with pytest.raises(RuntimeError, match="no usable GPU"):
        nd.resolve_gpu("on")


def test_out_of_memory_is_recognised():
    class XlaRuntimeError(RuntimeError):
        pass
    assert pl_gpu.is_out_of_memory(XlaRuntimeError("RESOURCE_EXHAUSTED: Out of memory while trying to allocate 3.2GiB"))
    assert pl_gpu.is_out_of_memory(RuntimeError("CUDA_ERROR_OUT_OF_MEMORY: out of memory"))
    assert not pl_gpu.is_out_of_memory(ValueError("shapes do not match"))
