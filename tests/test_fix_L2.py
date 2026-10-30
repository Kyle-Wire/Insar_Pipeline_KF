"""Lane L2 (time series): the fixes from the review of 5 Oct 2026.

Every test here fails on the code before the fix (backups/pre_fixes_20261005)
and passes after it. Synthetic data only; MintPy itself is never run - its
calls are replaced by a fake that writes (or fails to write) what the real
step would, so the ORCHESTRATION is what is tested: which outcomes count as
success, what is moved aside, what is recorded.
"""
import json
import logging
import os
import time
import types
from pathlib import Path

import h5py
import numpy as np
import pytest

from insar_pipeline import corrections
from insar_pipeline import time_series as ts
from insar_pipeline.config import PipelineConfig

DATES = ["20200101", "20200113", "20200125", "20200206", "20200218", "20200301"]
SHAPE = (12, 10)            # radar grid
GSHAPE = (14, 13)           # product (geo) grid
OLD = time.time() - 86400   # "the previous run", a day ago


def make_cfg(tmp_path, **over) -> PipelineConfig:
    kw = dict(project_dir=tmp_path / "proj", bbox=[-87.8, 38.2, -87.52, 38.47],
              start_date="2020-01-01", end_date="2020-03-01", backend="stack")
    kw.update(over)
    cfg = PipelineConfig(**kw)
    (cfg.stack_dir / "merged").mkdir(parents=True, exist_ok=True)
    (cfg.mintpy_dir / "inputs").mkdir(parents=True, exist_ok=True)
    return cfg


def _attrs(**kw):
    a = {"LENGTH": str(SHAPE[0]), "WIDTH": str(SHAPE[1]), "REF_Y": "3", "REF_X": "4", "REF_DATE": DATES[0],
         "START_DATE": DATES[0], "END_DATE": DATES[-1], "PLATFORM": "Sen"}
    a.update(kw)
    return a


def _geo_attrs(**kw):
    a = _attrs(LENGTH=str(GSHAPE[0]), WIDTH=str(GSHAPE[1]), REF_Y="5", REF_X="6", REF_LAT="38.33",
               REF_LON="-87.35", Y_FIRST="38.47", X_FIRST="-87.8", Y_STEP="-0.02", X_STEP="0.02")
    a.update(kw)
    return a


def _write(path, datasets: dict, attrs: dict, mtime=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        for k, v in datasets.items():
            f.create_dataset(k, data=v)
        for k, v in attrs.items():
            f.attrs[k] = v
    if mtime is not None:
        os.utime(path, (mtime, mtime))
    return path


def _ts_file(path, data=None, shape=SHAPE, attrs=None, mtime=None):
    n = len(DATES)
    if data is None:
        data = np.zeros((n,) + shape, np.float32)
    return _write(path, {"date": np.array([d.encode() for d in DATES]), "timeseries": data,
                         "bperp": np.zeros(n, np.float32)},
                  {"FILE_TYPE": "timeseries", **(attrs or _attrs())}, mtime)


def _tiny_stack(cfg, coherence=None):
    pairs = [(DATES[i], DATES[i + 1]) for i in range(len(DATES) - 1)]
    ds = {"date": np.array([[a.encode(), b.encode()] for a, b in pairs]),
          "dropIfgram": np.ones(len(pairs), bool),
          "bperp": np.zeros(len(pairs), np.float32),
          "unwrapPhase": np.zeros((len(pairs),) + SHAPE, np.float32),
          "unwrapPhase_bridging": np.zeros((len(pairs),) + SHAPE, np.float32)}
    if coherence is not None:
        ds["coherence"] = coherence
    p = _write(cfg.mintpy_dir / "inputs" / "ifgramStack.h5", ds,
               {"FILE_TYPE": "ifgramStack", "REF_Y": "3", "REF_X": "4", **_attrs()})
    with h5py.File(p, "r+") as f:
        f["unwrapPhase"].attrs["MODIFICATION_TIME"] = "1.0"
        f["unwrapPhase_bridging"].attrs["MODIFICATION_TIME"] = "2.0"
    return p


def _plane_series(gx=0.002, gy=-0.001):
    """A series whose deramp removed a plane growing linearly in time:
    returns (before, after) with before - after = plane * t."""
    n = len(DATES)
    t = ts._epoch_years(DATES)
    yy, xx = np.mgrid[0:SHAPE[0], 0:SHAPE[1]]
    plane = (gx * xx + gy * yy).astype(np.float64)              # metres per year
    after = np.random.default_rng(0).normal(0, 1e-4, (n,) + SHAPE)
    before = after + t[:, None, None] * plane[None]
    return before.astype(np.float32), after.astype(np.float32)


class FakeMintPy:
    """Stands in for time_series._mintpy: writes what each smallbaselineApp
    call would write, or fails like it, and records the calls."""

    def __init__(self, cfg, tokens=("ramp", "demErr"), fail_at=None, fail_rc=1, plot_rc=0,
                 write_geo_velocity=True, era5_dates=None):
        self.cfg, self.tokens, self.fail_at, self.fail_rc = cfg, tokens, fail_at, fail_rc
        self.plot_rc, self.write_geo_velocity, self.era5_dates = plot_rc, write_geo_velocity, era5_dates
        self.calls, self.plot_off_during_science = [], None

    def __call__(self, cfg, config_path, start, end, log_name, append=False):
        M = cfg.mintpy_dir
        self.calls.append(start)
        with open(M / log_name, "a" if append else "w") as fh:
            fh.write(f"******************** step - {start} ********************\n")
        if start == self.fail_at:
            with open(M / log_name, "a") as fh:
                fh.write(f"Traceback ...\nERROR in {start}\n")
            return types.SimpleNamespace(returncode=self.fail_rc)
        if start == "correct_unwrap_error":
            with h5py.File(M / "inputs" / "ifgramStack.h5", "r+") as f:
                f["unwrapPhase_bridging"].attrs["MODIFICATION_TIME"] = str(time.time())
        elif start == "invert_network":
            _ts_file(M / "timeseries.h5")
            _write(M / "temporalCoherence.h5", {"temporalCoherence": np.full(SHAPE, 0.9, np.float32)},
                   {"FILE_TYPE": "temporalCoherence", **_attrs()})
            _write(M / "numInvIfgram.h5", {"mask": np.full(SHAPE, 5, np.float32)}, _attrs())
            _write(M / "maskTempCoh.h5", {"mask": np.ones(SHAPE, bool)}, {"FILE_TYPE": "mask", **_attrs()})
        elif start == "correct_LOD":                  # ..correct_troposphere
            self.plot_off_during_science = "mintpy.plot = no" in config_path.read_text()
            if "ERA5" in self.tokens:
                _ts_file(M / "timeseries_ERA5.h5", _plane_series()[0])
                if self.era5_dates is not None:
                    _write(M / "inputs" / "ERA5.h5", {"date": np.array([d.encode() for d in self.era5_dates])}, {})
        elif start == "deramp":                       # ..geocode
            self.plot_off_during_science &= "mintpy.plot = no" in config_path.read_text()
            self._science(M)
        elif start == "google_earth":
            return types.SimpleNamespace(returncode=self.plot_rc)
        return types.SimpleNamespace(returncode=0)

    def _science(self, M):
        before, after = _plane_series()
        name = "timeseries"
        for i, tok in enumerate(self.tokens):
            prev = name
            name = f"{name}_{tok}"
            data = after if tok in ("ramp", "demErr") else before
            _ts_file(M / f"{name}.h5", data)
            if tok == "ramp":                  # the deramp's input carries the plane
                _ts_file(M / f"{prev}.h5", before)
        _write(M / "velocity.h5", {"velocity": np.zeros(SHAPE, np.float32)}, {"FILE_TYPE": "velocity", **_attrs()})
        _write(M / "demErr.h5", {"dem": np.zeros(SHAPE, np.float32)}, _attrs())
        G = M / "geo"
        gattrs = _geo_attrs(**{"mintpy.deramp": "linear", "mintpy.deramp.maskFile": str(M / "maskDeramp.h5")})
        if self.write_geo_velocity:
            _write(G / "geo_velocity.h5", {"velocity": np.zeros(GSHAPE, np.float32)}, {"FILE_TYPE": "velocity", **gattrs})
        _ts_file(G / f"geo_{name}.h5", shape=GSHAPE, attrs=gattrs)
        _write(G / "geo_temporalCoherence.h5", {"temporalCoherence": np.full(GSHAPE, 0.9, np.float32)}, gattrs)
        _write(G / "geo_maskTempCoh.h5", {"mask": np.ones(GSHAPE, bool)}, gattrs)


LOADER = ("mintpy.load.processor = isce\nmintpy.unwrapError.method = bridging\n"
          "mintpy.deramp = linear\n")


def _previous_run(cfg):
    """What a finished earlier run (with ERA5) leaves in mintpy/, a day old."""
    M = cfg.mintpy_dir
    gattrs = _geo_attrs(REF_DATE="20200113")
    _ts_file(M / "geo" / "geo_timeseries_ERA5_ramp_demErr.h5", shape=GSHAPE, attrs=gattrs, mtime=OLD)
    _write(M / "geo" / "geo_velocity.h5", {"velocity": np.ones(GSHAPE, np.float32)},
           {"FILE_TYPE": "velocity", **gattrs}, OLD)
    _write(M / "geo" / "geo_temporalCoherence.h5", {"temporalCoherence": np.ones(GSHAPE, np.float32)}, gattrs, OLD)
    _write(M / "geo" / "geo_maskTempCoh.h5", {"mask": np.ones(GSHAPE, bool)}, gattrs, OLD)
    _ts_file(M / "timeseries_ERA5_ramp_demErr.h5", mtime=OLD)
    _write(M / "velocity.h5", {"velocity": np.ones(SHAPE, np.float32)}, _attrs(), OLD)
    _write(M / "maskTempCoh.h5", {"mask": np.ones(SHAPE, bool)}, _attrs(), OLD)
    (M / "run_products.json").write_text(json.dumps({"timeseries_geo": "old"}))
    os.utime(M / "run_products.json", (OLD, OLD))


def _stub_preamble(monkeypatch, fake):
    """Replace the MintPy calls and the steps that need a real stack."""
    monkeypatch.setattr(ts, "_mintpy", fake)
    monkeypatch.setattr(ts, "reference_unreliable_pairs", lambda cfg: ([], {"n_kept": 5}))
    monkeypatch.setattr(ts, "ensure_connected_network", lambda cfg, p: {})
    monkeypatch.setattr(ts, "pin_reference_into_conncomp_mask", lambda cfg, p: {})
    monkeypatch.setattr(ts, "deramp_mask_step", lambda cfg: None)
    monkeypatch.setattr(ts, "phase_closure_diagnostic", lambda cfg: None)
    monkeypatch.setattr(ts, "geocode_extra_layers", lambda cfg: [], raising=False)      # tested on its own below


# ---------------------------------------------------------------- ops-1, timeseries-1
def test_failed_corrections_do_not_reuse_old_geo_products(tmp_path, monkeypatch):
    """correct_LOD..geocode fails on a re-run: the run must raise, and the
    previous run's geo products must be moved aside, not left to be
    delivered as this run's (before: 'velocity.h5 was written - a later step
    (usually plotting) failed', and a normal return)."""
    cfg = make_cfg(tmp_path, mintpy_tropo_method="no")
    _tiny_stack(cfg)
    _previous_run(cfg)
    fake = FakeMintPy(cfg, fail_at="correct_LOD")
    _stub_preamble(monkeypatch, fake)
    with pytest.raises(RuntimeError) as e:
        ts.run_mintpy_with_loader(cfg, LOADER)
    assert "correct_LOD" in str(e.value) and "ERROR in correct_LOD" in str(e.value)
    M = cfg.mintpy_dir
    assert not [p for p in (M / "geo").glob("*") if p.is_file()]          # nothing old left to deliver
    sup = list((M / "superseded").glob("run_*"))
    assert len(sup) == 1
    assert (sup[0] / "geo" / "geo_velocity.h5").exists()
    assert (sup[0] / "geo" / "geo_timeseries_ERA5_ramp_demErr.h5").exists()
    assert (sup[0] / "timeseries_ERA5_ramp_demErr.h5").exists() and (sup[0] / "velocity.h5").exists()
    assert not (M / "run_products.json").exists()
    assert json.loads((sup[0] / "RETIRED.json").read_text())["files"]
    assert "google_earth" not in fake.calls                                  # stopped before the tail


def test_inversion_killed_on_a_rerun_raises_although_an_old_mask_exists(tmp_path, monkeypatch):
    """invert_network killed (rc -9) with the previous run's maskTempCoh.h5
    still there: before, the existence check passed and the corrections ran
    on the old timeseries.h5."""
    cfg = make_cfg(tmp_path, mintpy_tropo_method="no")
    _tiny_stack(cfg)
    _previous_run(cfg)
    fake = FakeMintPy(cfg, fail_at="invert_network", fail_rc=-9)
    _stub_preamble(monkeypatch, fake)
    with pytest.raises(RuntimeError) as e:
        ts.run_mintpy_with_loader(cfg, LOADER)
    assert "invert_network failed" in str(e.value) and "-9" in str(e.value)
    assert "correct_LOD" not in fake.calls


def test_science_step_exiting_zero_without_geocoding_is_a_failure(tmp_path, monkeypatch):
    """rc 0 but no geo_velocity.h5 written: the old file must not stand in."""
    cfg = make_cfg(tmp_path, mintpy_tropo_method="no")
    _tiny_stack(cfg)
    _previous_run(cfg)
    _stub_preamble(monkeypatch, FakeMintPy(cfg, write_geo_velocity=False))
    with pytest.raises(RuntimeError) as e:
        ts.run_mintpy_with_loader(cfg, LOADER)
    assert "geo_velocity.h5" in str(e.value)


def test_require_fresh_names_missing_and_old_files(tmp_path):
    new = tmp_path / "new.h5"; new.write_bytes(b"x")
    old = tmp_path / "old.h5"; old.write_bytes(b"x"); os.utime(old, (OLD, OLD))
    ts._require_fresh([new], time.time() - 5, "x", "log")
    with pytest.raises(RuntimeError) as e:
        ts._require_fresh([new, old, tmp_path / "gone.h5"], time.time() - 5, "step", "mintpy_error.log")
    assert "gone.h5" in str(e.value) and "old.h5" in str(e.value) and "new.h5" not in str(e.value)


# ------------------------------------------- timeseries-1/-3, timeseries-6 (positive path)
def test_successful_rerun_records_its_own_chain_and_plots_may_fail(tmp_path, monkeypatch):
    """A re-run that dropped ERA5 (tropo 'no') after an ERA5 run: the plots
    may fail, the run is accepted, the old ERA5 geo series is moved aside
    (before, it outranked the new series by token count and the report said
    weather delays were corrected), and run_products.json names this run's
    files and corrections, read from the products."""
    cfg = make_cfg(tmp_path, mintpy_tropo_method="no")
    _tiny_stack(cfg)
    _previous_run(cfg)
    fake = FakeMintPy(cfg, plot_rc=1)
    _stub_preamble(monkeypatch, fake)
    ts.run_mintpy_with_loader(cfg, LOADER)
    M = cfg.mintpy_dir
    assert fake.plot_off_during_science is True                  # plotting cannot fail the science call
    assert "mintpy.plot" not in (M / "smallbaselineApp.cfg").read_text()   # template restored
    assert sorted(p.name for p in (M / "geo").glob("geo_timeseries*.h5")) == ["geo_timeseries_ramp_demErr.h5"]
    rp = json.loads((M / "run_products.json").read_text())
    assert rp["timeseries_geo"] == str((M / "geo" / "geo_timeseries_ramp_demErr.h5").absolute())
    assert rp["timeseries_radar"] == str((M / "timeseries_ramp_demErr.h5").absolute())
    assert rp["velocity_geo"] == str((M / "geo" / "geo_velocity.h5").absolute())
    assert rp["corrections_applied"] == ["ramp", "demErr"] and rp["era5_missing_dates"] == []
    assert rp["tide_applied"] is False and rp["iono_applied"] is False
    assert (rp["ref_y"], rp["ref_x"], rp["ref_date"]) == (5, 6, DATES[0])
    assert rp["ref_lat"] == pytest.approx(38.33) and rp["started"] and rp["finished"]
    assert rp["unwrap_error"]["dataset"] == "unwrapPhase_bridging" and rp["inversion"] == "inverted"
    assert rp["removed_ramp"]["applied"] is True and rp["plots_exit_code"] == 1
    assert rp["superseded_dir"] and Path(rp["superseded_dir"]).exists()


def test_rerun_with_partial_era5_stops_and_names_the_dates(tmp_path, monkeypatch):
    """timeseries-2 wired into the run: ERA5.h5 lacks the last date."""
    cfg = make_cfg(tmp_path, mintpy_tropo_method="pyaps")
    _tiny_stack(cfg)
    fake = FakeMintPy(cfg, tokens=("ERA5", "ramp", "demErr"), era5_dates=DATES[:-1])
    _stub_preamble(monkeypatch, fake)
    with pytest.raises(RuntimeError) as e:
        ts.run_mintpy_with_loader(cfg, LOADER)
    assert DATES[-1] in str(e.value) and "1 of 6" in str(e.value)
    assert "deramp" not in fake.calls and "google_earth" not in fake.calls      # stopped before deramp/geocode
    assert not list((cfg.mintpy_dir / "geo").glob("geo_*")) if (cfg.mintpy_dir / "geo").exists() else True
    rec = json.loads((cfg.mintpy_dir / "corrections.json").read_text())["troposphere"]
    assert rec["missing_dates"] == [DATES[-1]] and rec["dates_corrected"] == 5


def test_retire_moves_downstream_products_and_keeps_this_runs_inversion(tmp_path):
    cfg = make_cfg(tmp_path)
    M = cfg.mintpy_dir
    keep = ["timeseries.h5", "temporalCoherence.h5", "temporalCoherence_mintpy.h5", "maskTempCoh.h5",
            "numInvIfgram.h5", "maskDeramp.h5", "numTriNonzeroIntAmbiguity.h5", "inputs/ERA5.h5"]
    move = ["timeseries_ERA5.h5", "timeseries_ERA5_ramp_demErr.h5", "timeseriesResidual.h5",
            "timeseriesResidual_ramp.h5", "velocity.h5", "velocityERA5.h5", "demErr.h5", "reference_date.txt",
            "rms_timeseriesResidual_ramp.txt", "run_products.json", "geo/geo_velocity.h5",
            "geo/geo_timeseries_ERA5_ramp_demErr.kmz"]
    for n in keep + move:
        (M / n).parent.mkdir(parents=True, exist_ok=True)
        (M / n).write_bytes(b"x")
    dest = ts.retire_downstream_products(cfg, "20261005_120000")
    assert dest == M / "superseded" / "run_20261005_120000"
    assert all((M / n).exists() for n in keep)
    assert not any((M / n).exists() for n in move)
    assert all((dest / n).exists() for n in move)
    assert ts.retire_downstream_products(cfg, "20261005_120001") is None     # nothing left to move


def test_check_inversion_accepts_an_up_to_date_skip_but_not_old_products(tmp_path):
    cfg = make_cfg(tmp_path)
    M = cfg.mintpy_dir
    _tiny_stack(cfg)
    with h5py.File(M / "inputs" / "ifgramStack.h5", "r+") as f:
        f["unwrapPhase_bridging"].attrs["MODIFICATION_TIME"] = str(OLD - 100)
    for n in ("timeseries.h5", "temporalCoherence.h5", "numInvIfgram.h5", "maskTempCoh.h5"):
        _write(M / n, {"x": np.zeros(2)}, {"REF_Y": "3", "REF_X": "4"}, OLD)
    ok = types.SimpleNamespace(returncode=0)
    assert ts.check_inversion(cfg, ok, "unwrapPhase_bridging", time.time()) == "up to date"
    with h5py.File(M / "inputs" / "ifgramStack.h5", "r+") as f:               # the input moved on since
        f["unwrapPhase_bridging"].attrs["MODIFICATION_TIME"] = str(time.time())
    with pytest.raises(RuntimeError):
        ts.check_inversion(cfg, ok, "unwrapPhase_bridging", time.time())
    with pytest.raises(RuntimeError):
        ts.check_inversion(cfg, types.SimpleNamespace(returncode=-9), "unwrapPhase_bridging", 0.0)


# ------------------------------------------------------------------ timeseries-3
def test_chain_files_uses_mintpys_own_naming(tmp_path):
    """The series the velocity is fitted to, from MintPy's
    get_timeseries_filename on the resolved template - not from token
    counting. tropo 'no' -> no ERA5; height_correlation -> tropHgt."""
    cfg = make_cfg(tmp_path)
    M = cfg.mintpy_dir
    _ts_file(M / "timeseries.h5")
    p = M / "smallbaselineApp.cfg"
    p.write_text("mintpy.troposphericDelay.method = no\nmintpy.deramp = linear\nmintpy.solidEarthTides = no\n")
    ch = ts.chain_files(cfg, p)
    assert ch["final"].name == "timeseries_ramp_demErr.h5" and ch["tokens"] == ["ramp", "demErr"]
    assert ch["deramp_in"].name == "timeseries.h5" and ch["deramp_out"].name == "timeseries_ramp.h5"
    p.write_text("mintpy.troposphericDelay.method = height_correlation\nmintpy.deramp = no\n"
                 "mintpy.solidEarthTides = yes\n")
    ch = ts.chain_files(cfg, p)
    assert ch["final"].name == "timeseries_SET_tropHgt_demErr.h5"
    assert ch["deramp_in"] == ch["deramp_out"]


# ------------------------------------------------------------------ timeseries-4
def test_stale_closure_dataset_is_not_inverted(tmp_path):
    """Site A, 27 Sep: unwrapPhase_bridging_phaseClosure (stamped before
    the bridging it should follow) sits in the stack; MintPy's closure step
    found nothing and left it. Before: presence was enough and the rejected
    variant was inverted."""
    cfg = make_cfg(tmp_path)
    p = _tiny_stack(cfg)
    with h5py.File(p, "r+") as f:
        f.create_dataset("unwrapPhase_bridging_phaseClosure", data=np.zeros((5,) + SHAPE, np.float32))
        f["unwrapPhase_bridging"].attrs["MODIFICATION_TIME"] = "1790543325.15"
        f["unwrapPhase_bridging_phaseClosure"].attrs["MODIFICATION_TIME"] = "1790543324.15"
    cfgp = cfg.mintpy_dir / "smallbaselineApp.cfg"
    cfgp.write_text("mintpy.unwrapError.method = bridging+phase_closure\n")
    rec = ts.unwrap_error_applied(cfg, cfgp)
    assert rec["dataset"] == "unwrapPhase_bridging" and rec["applied"] == "bridging"
    assert rec["stale_not_used"] == ["unwrapPhase_bridging_phaseClosure"]
    assert "mintpy.unwrapError.method = bridging" in cfgp.read_text()


def test_pipeline_closure_dataset_removed_when_closure_is_off(tmp_path):
    """The per-pixel closure dataset written AFTER bridging looks current to
    MintPy, which then skips its own closure step: with the correction off it
    must go before correct_unwrap_error (marker stamp, or the legacy record)."""
    cfg = make_cfg(tmp_path)                       # mintpy_closure_correction defaults to False
    p = _tiny_stack(cfg)
    rec = ts.closure_integer_correction(cfg, "unwrapPhase_bridging", block_rows=4)
    with h5py.File(p, "r") as f:
        assert ts.PIPELINE_CLOSURE_ATTR in f[rec["dataset_out"]].attrs
    assert ts.remove_rejected_closure_dataset(cfg) == ["unwrapPhase_bridging_phaseClosure"]
    with h5py.File(p, "r") as f:
        assert "unwrapPhase_bridging_phaseClosure" not in f
    # an older stack: no stamp, named by closure_correction.json
    with h5py.File(p, "r+") as f:
        f.create_dataset("unwrapPhase_bridging_phaseClosure", data=np.zeros((5,) + SHAPE, np.float32))
    assert ts.remove_rejected_closure_dataset(cfg) == ["unwrapPhase_bridging_phaseClosure"]
    # with the correction on it stays (the pipeline recomputes or reuses it)
    with h5py.File(p, "r+") as f:
        f.create_dataset("unwrapPhase_bridging_phaseClosure", data=np.zeros((5,) + SHAPE, np.float32))
    cfg.mintpy_closure_correction = True
    assert ts.remove_rejected_closure_dataset(cfg) == []


def test_closure_that_found_nothing_after_a_bump_is_not_a_recompute_failure(tmp_path):
    """Bridging recomputed, closure (later in the chain) left stale: before,
    'did not recompute unwrapPhase_bridging_phaseClosure' was raised."""
    cfg = make_cfg(tmp_path)
    p = _tiny_stack(cfg)
    cfgp = cfg.mintpy_dir / "smallbaselineApp.cfg"
    cfgp.write_text("mintpy.unwrapError.method = bridging+phase_closure\n")
    with h5py.File(p, "r+") as f:
        f.create_dataset("unwrapPhase_bridging_phaseClosure", data=np.zeros((5,) + SHAPE, np.float32))
        f["unwrapPhase_bridging_phaseClosure"].attrs["MODIFICATION_TIME"] = "5.0"
    bump = 100.0
    with h5py.File(p, "r+") as f:
        f["unwrapPhase_bridging"].attrs["MODIFICATION_TIME"] = "200.0"
    ts.record_unwrap_error_state(cfg, bump, cfgp)                  # no raise
    with h5py.File(p, "r+") as f:                                   # bridging itself not recomputed: raise
        f["unwrapPhase_bridging"].attrs["MODIFICATION_TIME"] = "50.0"
    with pytest.raises(RuntimeError):
        ts.record_unwrap_error_state(cfg, bump, cfgp)
    cfgp.write_text("mintpy.unwrapError.method = no\n")              # nothing asked for: nothing to recompute
    ts.record_unwrap_error_state(cfg, bump, cfgp)


# ------------------------------------------------------------------ timeseries-8
def _cc_stack(cfg, cc):
    n = cc.shape[0]
    dates = [f"2020{m:02d}01" for m in range(1, n + 2)]
    return _write(cfg.mintpy_dir / "inputs" / "ifgramStack.h5",
                  {"date": np.array([[dates[i].encode(), dates[i + 1].encode()] for i in range(n)]),
                   "dropIfgram": np.ones(n, bool), "connectComponent": cc},
                  {"FILE_TYPE": "ifgramStack", "LENGTH": "20", "WIDTH": "20"})


def test_reference_pin_sees_the_current_kept_pairs_on_a_fresh_run(tmp_path, monkeypatch):
    """No maskConnComp.h5 yet (MintPy writes it only inside reference_point):
    before, the pin returned {} and did nothing on every fresh run."""
    cfg = make_cfg(tmp_path, reference_lalo="38.3,-87.6")
    cc = np.ones((3, 20, 20), np.int16)
    cc[1, 10, 10] = 0                                     # reference in component 0 in one kept pair
    _cc_stack(cfg, cc)
    monkeypatch.setattr(ts, "mintpy_reference_yx", lambda cfg, la, lo: (10, 10))
    tmpl = cfg.mintpy_dir / "smallbaselineApp.cfg"
    tmpl.write_text("mintpy.reference.lalo = 38.3,-87.6\n")
    assert not (cfg.mintpy_dir / "maskConnComp.h5").exists()
    info = ts.pin_reference_into_conncomp_mask(cfg, tmpl)
    assert info["in_mask"] is False and "pinned_yx" in info
    assert (cfg.mintpy_dir / "reference_pin.json").exists()
    with h5py.File(cfg.mintpy_dir / "maskConnComp.h5", "r") as f:
        assert f.attrs["FILE_TYPE"] == "mask" and not f["mask"][10, 10] and f["mask"][0, 0]


def test_reference_pin_rebuilds_a_mask_older_than_the_stack(tmp_path, monkeypatch):
    """Re-run: the previous run's mask says the reference is fine, but the
    current kept set leaves it in component 0."""
    cfg = make_cfg(tmp_path, reference_lalo="38.3,-87.6")
    _write(cfg.mintpy_dir / "maskConnComp.h5", {"mask": np.ones((20, 20), bool)}, {"FILE_TYPE": "mask"}, OLD)
    cc = np.ones((2, 20, 20), np.int16)
    cc[0, 10, 10] = 0
    _cc_stack(cfg, cc)
    monkeypatch.setattr(ts, "mintpy_reference_yx", lambda cfg, la, lo: (10, 10))
    tmpl = cfg.mintpy_dir / "smallbaselineApp.cfg"
    tmpl.write_text("mintpy.reference.lalo = 38.3,-87.6\n")
    info = ts.pin_reference_into_conncomp_mask(cfg, tmpl)
    assert info["in_mask"] is False and "pinned_yx" in info


def test_conncomp_mask_ignores_dropped_pairs(tmp_path):
    cfg = make_cfg(tmp_path)
    cc = np.ones((2, 20, 20), np.int16)
    cc[1, 2, 2] = 0
    p = _cc_stack(cfg, cc)
    with h5py.File(p, "r+") as f:
        f["dropIfgram"][1] = False
    ts.write_conncomp_mask(cfg)
    with h5py.File(cfg.mintpy_dir / "maskConnComp.h5", "r") as f:
        assert f["mask"][:].all()


# --------------------------------------- gap-surface_mine_site-1, timeseries-5, phaselinked-6
def test_linked_noise_level_reads_the_noise_mode_not_the_tail():
    rng = np.random.default_rng(1)
    n = 40000
    linked = rng.uniform(0.45, 0.75, n)                  # coherent ground
    mtc = rng.uniform(0.5, 1.0, n)
    noise = slice(0, 3000)                               # water: linked ~0.373 +- 0.008, random unwrapping
    linked[noise] = rng.normal(0.373, 0.008, 3000)
    mtc[noise] = rng.uniform(0.05, 0.24, 3000)
    tail = slice(3000, 4000)                             # coherent ground with unwrapping errors
    linked[tail] = rng.uniform(0.45, 0.7, 1000)
    mtc[tail] = rng.uniform(0.05, 0.24, 1000)
    rec = ts.linked_noise_level(linked, mtc)
    assert 0.380 <= rec["level"] <= 0.395, rec              # mode + 1.645 sigma of the noise alone
    assert rec["noise_cells"] == 4000
    assert ts.linked_noise_level(linked, np.ones(n))["level"] is None    # a star: nothing to measure on


def test_adopting_linked_coherence_records_noise_level_and_sources(tmp_path, caplog):
    from insar_pipeline import miaplpy_backend as mb
    cfg = make_cfg(tmp_path, backend="miaplpy")
    M = cfg.mintpy_dir
    rng = np.random.default_rng(2)
    shape = (60, 60)
    lk = rng.uniform(0.45, 0.7, shape).astype(np.float32)
    mtc = rng.uniform(0.5, 1.0, shape).astype(np.float32)
    lk[:20] = rng.normal(0.373, 0.008, (20, 60))           # 1200 water cells
    mtc[:20] = 0.1
    np.savez_compressed(mb.linked_temporal_coherence_path(cfg), average=lk, full=np.full(shape, 0.03, np.float32),
                        layer_order=np.array("miaplpy temporalCoherence index 0 = average (mini-stack), 1 = full"))
    _write(M / "temporalCoherence.h5", {"temporalCoherence": mtc}, {"FILE_TYPE": "temporalCoherence"})
    _write(M / "maskTempCoh.h5", {"mask": np.ones(shape, bool)}, {"FILE_TYPE": "mask"})
    with caplog.at_level(logging.WARNING, logger="insar_pipeline.time_series"):
        ts.apply_linked_temporal_coherence(cfg)
    for n in ("maskTempCoh.h5", "temporalCoherence.h5"):
        with h5py.File(M / n, "r") as f:
            lvl = float(f.attrs["linked_noise_level"])
            assert 0.38 <= lvl <= 0.40 and float(f.attrs["threshold_minus_noise"]) == pytest.approx(0.4 - lvl, abs=1e-3)
    with h5py.File(M / "temporalCoherence_mintpy.h5", "r") as f:          # which coherence it is
        assert "MintPy network-inversion" in f.attrs["temporalCoherenceSource"]
        assert np.allclose(f["temporalCoherence"][:], mtc)
    assert any("only" in r.message and "above the measured noise level" in r.message for r in caplog.records)


def _disturbed_fixture(cfg):
    """60x40: A water (noise throughout, 800 cells), B coherent but decorrelated
    for 12 consecutive nearest pairs, C coherent throughout. 31 dates, 3 connections."""
    M = cfg.mintpy_dir
    rng = np.random.default_rng(3)
    nd, L, W = 31, 60, 40
    dates = [f"{2020 + (12 * i) // 365}{1 + ((12 * i) % 365) // 31:02d}{1 + (12 * i) % 28:02d}" for i in range(nd)]
    dates = sorted(set(dates))[:nd]
    pairs = [(i, j) for i in range(nd) for j in range(i + 1, min(i + 4, nd))]
    coh = np.empty((len(pairs), L, W), np.float32)
    for k, (i, j) in enumerate(pairs):
        coh[k] = rng.uniform(0.97, 1.0, (L, W))                    # coherent: above the noise level
        coh[k, :20] = rng.uniform(0.0, 1.0, (20, W))               # A: water
        if j - i == 1 and 8 <= i < 20:                             # B: 12 nearest pairs decorrelated
            coh[k, 20:30] = rng.uniform(0.0, 1.0, (10, W)) * 0.8
    _write(M / "inputs" / "ifgramStack.h5",
           {"date": np.array([[dates[i].encode(), dates[j].encode()] for i, j in pairs]),
            "dropIfgram": np.ones(len(pairs), bool), "coherence": coh}, {"FILE_TYPE": "ifgramStack"})
    lk = np.full((L, W), 0.55, np.float32)
    lk[:20] = rng.normal(0.373, 0.008, (20, W))
    mtc = np.full((L, W), 0.9, np.float32)
    mtc[:20] = 0.1
    _write(M / "temporalCoherence.h5", {"temporalCoherence": lk},
           {"FILE_TYPE": "temporalCoherence", "LENGTH": str(L), "WIDTH": str(W), "temporalCoherenceSource": "linked"})
    _write(M / "temporalCoherence_mintpy.h5", {"temporalCoherence": mtc}, {"FILE_TYPE": "temporalCoherence"})
    return L, W


def test_disturbed_layer_flags_a_stretch_of_noise_like_pairs(tmp_path):
    cfg = make_cfg(tmp_path, backend="miaplpy")
    _disturbed_fixture(cfg)
    p = ts.write_disturbed_layer(cfg)
    with h5py.File(p, "r") as f:
        run = f["disturbed"][:]
        frac = f["noise_like_fraction"][:]
        lvl = float(f.attrs["pair_noise_level"])
        assert f.attrs["pair_coherence_kind"] == "phasor" and int(f.attrs["nearest_pairs_used"]) == 30
    assert 0.9 < lvl < 1.0                               # 95th percentile of uniform(0, 1) noise
    assert np.median(run[:20]) >= 15                     # water: 95% of its pairs below, long runs
    assert run[20:30].min() >= 12                        # B: the 12-pair stretch
    assert np.median(run[30:]) <= 2 and frac[30:].max() < 0.2
    with h5py.File(p, "r") as f:
        assert f.attrs["usable"] == "yes"                # reliable ground (rows 20-60) mostly unflagged


def test_disturbed_layer_that_flags_reliable_ground_is_marked_unusable(tmp_path):
    """6 Oct 2026: on Site B the phasor pair coherence put the noise level at
    0.926 and flagged 93.8% of cells; a layer that flags ground MintPy grades
    reliable separates nothing and must not be delivered."""
    cfg = make_cfg(tmp_path, backend="miaplpy")
    _disturbed_fixture(cfg)
    M = cfg.mintpy_dir
    with h5py.File(M / "inputs" / "ifgramStack.h5", "r+") as f:      # reliable ground now as noisy as water
        c = f["coherence"][:]
        c[:, 20:] = np.random.default_rng(5).uniform(0.0, 1.0, c[:, 20:].shape)
        f["coherence"][:] = c
    p = ts.write_disturbed_layer(cfg)
    with h5py.File(p, "r") as f:
        assert f.attrs["usable"] == "no" and "cannot separate" in f.attrs["usable_reason"]


def test_geocode_extra_layers_adds_the_three_layers_without_touching_products(tmp_path, monkeypatch):
    """temporalCoherence_mintpy (second gate), demErr (reshaped ground) and the
    disturbed layer on the product grid; callable on a finished project."""
    cfg = make_cfg(tmp_path, backend="miaplpy")
    M = cfg.mintpy_dir
    _disturbed_fixture(cfg)
    _write(M / "demErr.h5", {"dem": np.zeros((60, 40), np.float32)}, {"FILE_TYPE": "dem"})
    _write(M / "inputs" / "geometryRadar.h5", {"latitude": np.zeros((60, 40))}, {})
    _write(M / "velocity.h5", {"velocity": np.zeros((60, 40), np.float32)}, {}, OLD)
    calls = []

    def fake_geocode(cfg, files):                         # MintPy's geocode.py: geo_<name>, float, NaN fill
        calls.append([Path(p).name for p in files])
        for src in files:
            with h5py.File(src, "r") as s, h5py.File(M / "geo" / f"geo_{Path(src).name}", "w") as g:
                for k in s:
                    a = s[k][:].astype(np.float32)
                    a[0, 0] = np.nan
                    g.create_dataset(k, data=a)
                g.attrs.update(dict(s.attrs))
        return types.SimpleNamespace(returncode=0)

    monkeypatch.setattr(ts, "_run_geocode", fake_geocode)
    out = ts.geocode_extra_layers(cfg)
    assert sorted(p.name for p in out) == ["geo_demErr.h5", "geo_disturbed.h5", "geo_temporalCoherence_mintpy.h5"]
    assert calls == [["temporalCoherence_mintpy.h5", "demErr.h5", "disturbed.h5"]]
    with h5py.File(M / "geo" / "geo_disturbed.h5", "r") as f:
        d = f["disturbed"]
        assert d.dtype == np.int16 and d[0, 0] == -1 and d.attrs["NO_DATA_VALUE"] == -1
    with h5py.File(M / "geo" / "geo_temporalCoherence_mintpy.h5", "r") as f:     # says which coherence it is
        assert "MintPy network-inversion" in f.attrs["temporalCoherenceSource"]
    assert os.path.getmtime(M / "velocity.h5") == pytest.approx(OLD)


def test_second_gate_asked_for_without_its_layer_raises(tmp_path):
    cfg = make_cfg(tmp_path, backend="miaplpy")
    M = cfg.mintpy_dir
    cfg.min_unwrap_consistency = 0.5
    prod = {"mask_geo": M / "geo" / "geo_maskTempCoh.h5"}
    with pytest.raises(RuntimeError):
        ts.second_gate_record(cfg, prod)
    _write(M / "geo" / "geo_maskTempCoh.h5", {"mask": np.ones((4, 5), bool)}, {})
    um = np.full((4, 5), 0.9, np.float32)
    um[0] = 0.3
    _write(M / "geo" / "geo_temporalCoherence_mintpy.h5", {"temporalCoherence": um}, {})
    rec = ts.second_gate_record(cfg, prod)
    assert rec["applied"] and rec["kept_percent_of_first_gate"] == 75.0     # one row of four below 0.5
    cfg.min_unwrap_consistency = 0.0
    assert ts.second_gate_record(cfg, prod)["applied"] is False           # default: off, nothing raised


# ------------------------------------------------------------------ timeseries-6
def test_deramp_mask_that_excludes_nothing_or_leaves_a_strip_raises(tmp_path):
    cfg = make_cfg(tmp_path)
    M = cfg.mintpy_dir
    _write(M / "maskTempCoh.h5", {"mask": np.ones((10, 10), bool)}, {})
    _write(M / "maskDeramp.h5", {"mask": np.ones((10, 10), bool)}, {})       # polygon off this grid
    with pytest.raises(RuntimeError) as e:
        ts.check_deramp_mask(cfg, M / "maskDeramp.h5", built=True, buffer_m=2000.0)
    assert "excludes NO coherent pixel" in str(e.value)
    strip = np.zeros((10, 10), bool)
    strip[0, :] = True                                                        # 10% kept
    _write(M / "maskDeramp.h5", {"mask": strip}, {})
    with pytest.raises(RuntimeError) as e:
        ts.check_deramp_mask(cfg, M / "maskDeramp.h5", built=True, buffer_m=2000.0)
    assert "10%" in str(e.value) and "deramp_buffer_m" in str(e.value)
    rec = ts.check_deramp_mask(cfg, M / "maskDeramp.h5", built=False)          # the user's own mask: warned
    assert rec["kept_fraction"] == 0.1
    assert json.loads((M / "deramp_mask.json").read_text())["excluded_pixels"] == 90


def test_deramp_mask_step_uses_the_configured_buffer(tmp_path, monkeypatch):
    from insar_pipeline import stack_backend as sb
    cfg = make_cfg(tmp_path)
    M = cfg.mintpy_dir
    cfg.deramp_buffer_m = 750.0
    (M / "smallbaselineApp.cfg").write_text("mintpy.deramp = linear\n")
    _write(M / "maskTempCoh.h5", {"mask": np.ones((10, 10), bool)}, {})
    seen = {}

    def fake_build(cfg, buffer_m=2000.0, **kw):
        seen["buffer_m"] = buffer_m
        m = np.ones((10, 10), bool)
        m[:3] = False
        return _write(cfg.mintpy_dir / "maskDeramp.h5", {"mask": m}, {})

    monkeypatch.setattr(sb, "build_deramp_mask", fake_build)
    ts.deramp_mask_step(cfg)
    assert seen["buffer_m"] == 750.0
    assert json.loads((M / "deramp_mask.json").read_text())["kept_fraction"] == 0.7


def test_removed_ramp_is_measured(tmp_path):
    cfg = make_cfg(tmp_path)
    M = cfg.mintpy_dir
    before, after = _plane_series(gx=0.002, gy=0.0)            # 2 mm/yr per pixel east
    _ts_file(M / "timeseries_ERA5.h5", before)
    _ts_file(M / "timeseries_ERA5_ramp.h5", after)
    _write(M / "maskTempCoh.h5", {"mask": np.ones(SHAPE, bool)}, {})
    lat = np.full(SHAPE, 38.3)
    lon = -87.6 + np.arange(SHAPE[1])[None, :] * (1.0 / (111.32 * np.cos(np.radians(38.3))))   # 1 km per pixel
    _write(M / "inputs" / "geometryRadar.h5", {"latitude": lat, "longitude": lon * np.ones(SHAPE)}, {})
    rec = ts.measure_removed_ramp(cfg, {"deramp_in": M / "timeseries_ERA5.h5", "deramp_out": M / "timeseries_ERA5_ramp.h5"})
    assert rec["applied"] is True
    assert rec["gradient_mm_yr_per_10km"] == pytest.approx(20.0, rel=0.02)      # 2 mm/yr per km
    assert rec["removed_rate_range_mm_yr"] == pytest.approx(0.96 * 18, rel=0.1)  # p2..p98 of 0..18 mm/yr
    assert json.loads((M / "deramp_effect.json").read_text())["applied"] is True


def test_deramp_that_ignored_its_mask_stops_the_run(tmp_path):
    cfg = make_cfg(tmp_path)
    _write(cfg.mintpy_dir / "geo" / "geo_velocity.h5", {"velocity": np.zeros((2, 2))},
           {"mintpy.deramp": "linear", "mintpy.deramp.maskFile": "maskTempCoh.h5"})
    with pytest.raises(RuntimeError) as e:
        ts._report_mintpy_end_state(cfg)
    assert "DID NOT USE THE EXCLUSION MASK" in str(e.value)


# ------------------------------------------------------------------ timeseries-2
def test_era5_coverage_missing_date_raises_unless_allowed(tmp_path):
    cfg = make_cfg(tmp_path)
    M = cfg.mintpy_dir
    _ts_file(M / "timeseries.h5")
    w = _write(M / "inputs" / "ERA5.h5", {"date": np.array([d.encode() for d in DATES[:-1]])}, {})
    corrections.record(cfg)
    with pytest.raises(RuntimeError) as e:
        corrections.verify_era5_coverage(cfg, M / "timeseries.h5", w)
    assert DATES[-1] in str(e.value) and "era5_allow_partial" in str(e.value)
    cfg.era5_allow_partial = True
    rec = corrections.verify_era5_coverage(cfg, M / "timeseries.h5", w)
    assert rec["dates_corrected"] == 5 and rec["dates_total"] == 6 and rec["missing_dates"] == [DATES[-1]]
    saved = json.loads((M / "corrections.json").read_text())
    assert saved["troposphere"]["missing_dates"] == [DATES[-1]] and "solid_earth_tides" in saved   # amended, not replaced
    _write(M / "inputs" / "ERA5.h5", {"date": np.array([d.encode() for d in DATES])}, {})
    cfg.era5_allow_partial = False
    assert corrections.verify_era5_coverage(cfg, M / "timeseries.h5", w)["missing_dates"] == []
    with pytest.raises(RuntimeError):
        corrections.verify_era5_coverage(cfg, M / "timeseries.h5", M / "inputs" / "MERRA.h5")


def test_weather_watch_reports_downloads_as_they_land(tmp_path):
    from insar_pipeline import progress
    cfg = make_cfg(tmp_path)
    w = ts._WeatherWatch(cfg, "mintpy_correct_LOD_to_geocode", 6, interval=0.01)
    assert w.poll() is True and w.poll() is False                 # 0 files, then unchanged
    d = cfg.mintpy_dir / "ERA5"
    d.mkdir()
    for i in range(3):
        (d / f"ERA5_N38_W88_2020010{i}_00.grb").write_bytes(b"g")
    assert w.poll() is True
    ev = progress.latest(cfg.project_dir)["mintpy_correct_LOD_to_geocode"]
    assert ev["done"] == 3 and ev["total"] == 6 and ev["status"] == "running"
    with ts._WeatherWatch(cfg, "s", 6, interval=0.01):
        time.sleep(0.05)                                           # the thread starts and stops cleanly


def test_plot_switch_restores_the_template_line(tmp_path):
    p = tmp_path / "smallbaselineApp.cfg"
    p.write_text("mintpy.plot           = auto  #[yes / no], auto for yes\nmintpy.plot.dpi = 150\n")
    with pytest.raises(ValueError):
        with ts._template_key(p, "mintpy.plot", "no"):
            assert "mintpy.plot = no" in p.read_text() and "mintpy.plot.dpi = 150" in p.read_text()
            raise ValueError("a failing science call")
    text = p.read_text()
    assert "mintpy.plot = no" not in text and "mintpy.plot           = auto" in text and "mintpy.plot.dpi = 150" in text
