"""One burst per sub-swath has no burst overlaps: ESD cannot run, so the
stack is coregistered by geometry and the records say so (21 Oct 2026)."""
import json
from types import SimpleNamespace

from insar_pipeline import stack_backend as sb


def _cfg(tmp_path, do_esd=True, swaths="2", bursts=None):
    slc, stack = tmp_path / "slc", tmp_path / "stack"
    slc.mkdir(parents=True); stack.mkdir(parents=True)
    if bursts is not None:
        (slc / "assembly_status.json").write_text(json.dumps({"modal_burst_count": bursts}))
    return SimpleNamespace(do_esd=do_esd, stack_swaths=swaths, slc_dir=slc, stack_dir=stack)


def test_one_burst_falls_back_to_geometry(tmp_path):
    cfg = _cfg(tmp_path, bursts=1)
    ok, why = sb.esd_possible(cfg)
    assert not ok and "no burst overlaps" in why
    assert sb.coregistration_mode(cfg) == "geometry"


def test_several_bursts_or_no_record_keep_esd(tmp_path):
    assert sb.coregistration_mode(_cfg(tmp_path / "a", bursts=3)) == "NESD"
    assert sb.coregistration_mode(_cfg(tmp_path / "b")) == "NESD"            # no record: as before
    assert sb.coregistration_mode(_cfg(tmp_path / "c", bursts=2, swaths="1 2")) == "geometry"
    assert sb.coregistration_mode(_cfg(tmp_path / "d", do_esd=False, bursts=9)) == "geometry"


def test_used_reads_what_the_stack_was_generated_with(tmp_path):
    cfg = _cfg(tmp_path, bursts=9)
    assert sb.coregistration_used(cfg) == "NESD"
    (cfg.stack_dir / "generation.json").write_text(json.dumps({"argv": ["stackSentinel.py", "-C", "geometry"]}))
    assert sb.coregistration_used(cfg) == "geometry"


def test_run_refused_before_download_for_one_burst():
    import pytest
    from insar_pipeline import pipeline
    cfg = SimpleNamespace(stack_swaths="2")
    plan = SimpleNamespace(bursts_per_date=1, bursts_source="burst catalogue, 3 dates from 2024-09-01")
    with pytest.raises(SystemExit, match="single-burst"):
        pipeline.burst_gate(cfg, plan, SimpleNamespace(force=False))
    pipeline.burst_gate(cfg, plan, SimpleNamespace(force=True))                         # --force goes on
    pipeline.burst_gate(cfg, SimpleNamespace(bursts_per_date=2, bursts_source="x"), SimpleNamespace(force=False))
    pipeline.burst_gate(cfg, SimpleNamespace(bursts_per_date=1, bursts_source="assumed 3 per sub-swath"),
                        SimpleNamespace(force=False))                                    # not measured: no refusal
