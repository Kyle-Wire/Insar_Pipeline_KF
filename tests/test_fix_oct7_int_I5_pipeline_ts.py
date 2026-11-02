"""Integration group I5 of the 7 Oct 2026 fixes: pipeline.py and
time_series.py.

(a) #42: the screening set-aside catches every client KMZ name
    (<T>_[SCREENING_]ground_movement[_DRAFT].kmz) and nothing else. The
    wider glob itself went in with I_disp #51 (test_fix_oct7_I_disp.py); this
    checks the technical KMZs and the set-aside folders stay where they are.
(b) mintpy/run_incomplete.json goes down before run_mintpy_with_loader sets
    aside or moves anything. It used to go down only just before
    retire_downstream_products, so a run that reloaded the stack (moving the
    previous radar velocity and series aside) and then died in load_data or
    the inversion left a mixed mintpy/ with no marker.

Synthetic data only; MintPy is never run (the fake MintPy of test_fix_L2.py).

Run: conda run -n insar python -m pytest tests/test_fix_oct7_int_I5_pipeline_ts.py -q -p no:cacheprovider
"""
import json
import sys
import time
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_fix_L2 as L2                                   # noqa: E402  (fixtures only)
from insar_pipeline import pipeline                        # noqa: E402
from insar_pipeline import time_series as ts               # noqa: E402


# ------------------------------------------------- (a) #42 the set-aside glob
def test_screening_set_aside_takes_every_client_kmz_and_no_technical_one(tmp_path):
    D = tmp_path / "deliverables"
    (D / "superseded" / "kmz_20261001_120000").mkdir(parents=True)
    client = ["Bear_ground_movement.kmz", "Bear_ground_movement_DRAFT.kmz",
              "Bear_SCREENING_ground_movement_DRAFT.kmz", "Bear_SCREENING_ground_movement.kmz"]
    technical = ["geo_timeseries_ERA5_ramp_demErr.kmz", "los_displacement_points.kmz", "client_points.kmz"]
    for n in client + technical:
        (D / n).write_text(n)
    older = D / "superseded" / "kmz_20261001_120000" / "Bear_ground_movement.kmz"   # already set aside
    older.write_text("older")
    moved = pipeline.set_aside_client_files(types.SimpleNamespace(project_dir=tmp_path))
    assert sorted(p.name for p in moved) == sorted(client)
    assert all((D / n).exists() for n in technical)
    assert older.exists()


# ------------------------------------------------- (b) the marker goes down first
def test_a_reload_that_dies_in_load_data_leaves_the_marker(tmp_path, monkeypatch):
    """No stack: clear_stale_mintpy_products moves the previous radar velocity
    and series aside, then load_data fails. Before: no marker, so geo/ and
    run_products.json of the previous run sat beside a mintpy/ missing its
    radar products and nothing said the run had not finished."""
    cfg = L2.make_cfg(tmp_path, mintpy_tropo_method="no")
    L2._previous_run(cfg)                                    # no inputs/ifgramStack.h5: a reload
    M = cfg.mintpy_dir
    L2._stub_preamble(monkeypatch, L2.FakeMintPy(cfg, fail_at="load_data"))
    with pytest.raises(RuntimeError, match="load step failed"):
        ts.run_mintpy_with_loader(cfg, L2.LOADER)
    assert not (M / "velocity.h5").exists()                  # moved aside...
    mark = json.loads((M / ts.RUN_INCOMPLETE).read_text())   # ...and marked
    assert (M / "superseded" / f"run_{mark['stamp']}" / "velocity.h5").exists()
    assert mark["started"]

    # a clean run afterwards removes it
    time.sleep(1.1)
    L2._tiny_stack(cfg)
    L2._stub_preamble(monkeypatch, L2.FakeMintPy(cfg))
    ts.run_mintpy_with_loader(cfg, L2.LOADER)
    assert (M / "run_products.json").exists() and not (M / ts.RUN_INCOMPLETE).exists()


def test_an_inversion_that_dies_after_a_reload_leaves_the_marker(tmp_path, monkeypatch):
    cfg = L2.make_cfg(tmp_path, mintpy_tropo_method="no")
    L2._previous_run(cfg)
    M = cfg.mintpy_dir

    class _LoadsThenDies(L2.FakeMintPy):
        def __call__(self, cfg_, config_path, start, end, log_name, append=False):
            if start == "load_data":
                L2._tiny_stack(cfg_)
            return super().__call__(cfg_, config_path, start, end, log_name, append)
    fake = _LoadsThenDies(cfg, fail_at="invert_network")
    L2._stub_preamble(monkeypatch, fake)
    with pytest.raises(RuntimeError):
        ts.run_mintpy_with_loader(cfg, L2.LOADER)
    assert "invert_network" in fake.calls and "correct_LOD" not in fake.calls    # died in the inversion
    assert not (M / "velocity.h5").exists()
    assert (M / ts.RUN_INCOMPLETE).exists()


def test_the_marker_is_down_before_the_stack_is_set_aside(tmp_path, monkeypatch):
    cfg = L2.make_cfg(tmp_path, mintpy_tropo_method="no")
    L2._tiny_stack(cfg)
    L2._previous_run(cfg)
    L2._stub_preamble(monkeypatch, L2.FakeMintPy(cfg))
    seen = {}

    def set_aside(cfg_, lines):
        seen["marked"] = (cfg_.mintpy_dir / ts.RUN_INCOMPLETE).exists()
        raise RuntimeError("could not rename the stack")
    monkeypatch.setattr(ts, "retire_stale_ifgram_stack", set_aside)
    with pytest.raises(RuntimeError, match="could not rename"):
        ts.run_mintpy_with_loader(cfg, L2.LOADER)
    assert seen["marked"] is True
    assert (cfg.mintpy_dir / ts.RUN_INCOMPLETE).exists()


def test_a_failure_before_mintpy_is_touched_leaves_the_finished_run_unmarked(tmp_path, monkeypatch):
    """Nothing of the previous run was changed, so it stays usable."""
    cfg = L2.make_cfg(tmp_path, mintpy_tropo_method="no")
    L2._tiny_stack(cfg)
    L2._previous_run(cfg)
    L2._stub_preamble(monkeypatch, L2.FakeMintPy(cfg))

    def bad_settings(cfg_):
        raise ValueError("bad tide setting")
    monkeypatch.setattr(ts.corrections, "mintpy_lines", bad_settings)
    with pytest.raises(ValueError, match="bad tide"):
        ts.run_mintpy_with_loader(cfg, L2.LOADER)
    assert not (cfg.mintpy_dir / ts.RUN_INCOMPLETE).exists()
    assert (cfg.mintpy_dir / "run_products.json").exists() and (cfg.mintpy_dir / "velocity.h5").exists()
