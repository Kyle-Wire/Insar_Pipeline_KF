"""
21 Oct 2026: a requested unwrap-error correction step that made nothing is
said where the reviewer reads.

Three phase-linked projects asked MintPy for bridging+phase_closure. Phase
closure found no region unwrapped in every kept pair ("number of common
regions: 0") and wrote nothing, so the inversion used the bridging dataset.
mintpy/unwrap_error_applied.json and provenance.json recorded the downgrade,
and one log WARNING said it. The QA record, its send rule and run_result.json
did not. Now:

- the QA carries an 'unwrap_correction' block (time_series.
  unwrap_error_shortfall) that is flagged when a requested step was skipped,
  with MintPy's reason read back from mintpy_unwerr.log;
- the send rule names the skip in a note (a recorded limitation, not a hold,
  because the closure correction was reviewed on 27 Sep 2026 and can hurt
  farmland), so the release gate's send-rule line shows it to the reviewer;
- format_report prints the block;
- a products run puts the same sentence in run_result.json's notes.

The client page is unchanged. Each test fails on the code at 0e28698.
Synthetic projects in tmp_path only.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_fix_L8 as L8                                         # noqa: E402  (the QA project builder)
import test_fix_oct16_runs as R16                               # noqa: E402  (run_result stubs)

from insar_pipeline import pipeline, runs                       # noqa: E402
from insar_pipeline import quality_assessment as qa             # noqa: E402
from insar_pipeline import stack_qc                              # noqa: E402
from insar_pipeline import time_series as ts                    # noqa: E402

SKIPPED = {"requested": "bridging+phase_closure", "applied": "bridging", "dataset": "unwrapPhase_bridging",
           "datasets_present": ["unwrapPhase", "unwrapPhase_bridging"]}
# what MintPy's unwrap_error_phase_closure.py prints when it corrects nothing
LOG = """unwrap_error_phase_closure.py inputs/ifgramStack.h5 --update --cc-mask maskConnComp.h5 -i unwrapPhase_bridging
calculating the integer ambiguity for the common regions defined in maskConnComp.h5
number of interferograms: 120
number of triplets: 110
read common mask from maskConnComp.h5
set minimum region size from 2500.0 to 1046.01 (image_size * 0.3%)
remove regions with area < 1046 pixels
number of common regions: 0
WARNING: NO common region found! Try a smaller value for the mintpy.unwrapError.connCompMinArea (2500.0) option.
skip phase closure correction ...
"""
CLEAN = {"checks": {"coverage": {"pass": True}}, "geolocation": {"flag": False}}


@pytest.fixture(autouse=True)
def no_osm(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("the QA tried to fetch OSM water in a test")
    monkeypatch.setattr(stack_qc, "fetch_osm_water", boom, raising=False)


def _record(cfg, rec=SKIPPED, log=LOG):
    cfg.mintpy_dir.mkdir(parents=True, exist_ok=True)
    (cfg.mintpy_dir / "unwrap_error_applied.json").write_text(json.dumps(rec))
    if log is not None:
        (cfg.mintpy_dir / "mintpy_unwerr.log").write_text(log)


# ------------------------------------------------------------------ the record
def test_a_skipped_phase_closure_is_flagged_with_mintpys_reason(tmp_path):
    cfg = R16._cfg(tmp_path)
    assert ts.unwrap_error_shortfall(cfg) is None                       # no record, nothing said
    _record(cfg)
    r = ts.unwrap_error_shortfall(cfg)
    assert r["flag"] is True and r["skipped"] == ["phase_closure"]
    assert (r["requested"], r["applied"], r["dataset"]) == ("bridging+phase_closure", "bridging",
                                                            "unwrapPhase_bridging")
    assert r["source"] == "mintpy/unwrap_error_applied.json"
    assert r["reason"] == ("MintPy found no region of at least 1046 pixels unwrapped in every kept pair "
                           "(mintpy_unwerr.log: number of common regions: 0)")
    text = ts.unwrap_error_skip_text(r)
    assert text.startswith("the requested phase-closure unwrapping-error correction was skipped (asked for "
                           "bridging+phase_closure, applied bridging; the inversion used unwrapPhase_bridging)")
    # without the log the skip is still said, with a pointer to the log
    (cfg.mintpy_dir / "mintpy_unwerr.log").unlink()
    assert "see mintpy_unwerr.log" in ts.unwrap_error_shortfall(cfg)["reason"]


def test_a_dataset_left_by_an_earlier_run_is_part_of_the_reason(tmp_path):
    cfg = R16._cfg(tmp_path)
    _record(cfg, dict(SKIPPED, stale_not_used=["unwrapPhase_bridging_phaseClosure"]), log=None)
    r = ts.unwrap_error_shortfall(cfg)
    assert r["flag"] and r["stale_not_used"] == ["unwrapPhase_bridging_phaseClosure"]
    assert "unwrapPhase_bridging_phaseClosure was left by an earlier run and not used" in r["reason"]


@pytest.mark.parametrize("rec", [
    # the pipeline's own per-pixel closure correction ran (time_series.run_mintpy_with_loader)
    {"requested": "bridging+phase_closure", "applied": "bridging+phase_closure",
     "dataset": "unwrapPhase_bridging_phaseClosure", "closure_correction": "per-pixel (pipeline)"},
    # nothing asked for: MintPy reads 'auto' as 'no'
    {"requested": "auto", "applied": "no", "dataset": "unwrapPhase"},
    {"requested": "no", "applied": "no", "dataset": "unwrapPhase"},
    # a route that took phase closure out before the run (no loops) asked only for bridging
    {"requested": "bridging", "applied": "bridging", "dataset": "unwrapPhase_bridging"},
])
def test_nothing_skipped_is_not_flagged(tmp_path, rec):
    cfg = R16._cfg(tmp_path)
    _record(cfg, rec)
    r = ts.unwrap_error_shortfall(cfg)
    assert r["flag"] is False and r["skipped"] == [] and "reason" not in r
    assert ts.unwrap_error_skip_text(r) is None


def test_run_products_copy_is_read_when_the_record_is_gone(tmp_path):
    cfg = R16._cfg(tmp_path)
    cfg.mintpy_dir.mkdir(parents=True, exist_ok=True)
    (cfg.mintpy_dir / "run_products.json").write_text(json.dumps({"unwrap_error": {
        k: SKIPPED[k] for k in ("requested", "applied", "dataset")}}))
    r = ts.unwrap_error_shortfall(cfg)
    assert r["flag"] and r["source"] == "mintpy/run_products.json"


# ------------------------------------------------------------------ the QA and its send rule
def test_send_rule_names_the_skip_in_a_note_not_a_hold():
    rec = dict(SKIPPED, skipped=["phase_closure"], flag=True, source="mintpy/unwrap_error_applied.json",
               reason="MintPy found no region of at least 1046 pixels unwrapped in every kept pair")
    assert qa.send_rule(CLEAN)["value"] == "send"
    sr = qa.send_rule(dict(CLEAN, unwrap_correction=rec))
    assert sr["value"] == "send with notes" and sr["hold_reasons"] == []
    assert any(n.startswith("the requested phase-closure unwrapping-error correction was skipped")
               and "1046 pixels" in n and n.endswith("(see unwrap_correction)") for n in sr["notes"]), sr
    ok = dict(rec, skipped=[], flag=False, applied="bridging+phase_closure")
    assert qa.send_rule(dict(CLEAN, unwrap_correction=ok))["value"] == "send"
    crashed = {"available": False, "reason": "could not run: ValueError('bad json')", "flag": None}
    sr = qa.send_rule(dict(CLEAN, unwrap_correction=crashed))
    assert sr["value"] == "send with notes" and any(n.startswith("unwrap_correction could not run")
                                                    for n in sr["notes"])


def test_evaluate_records_the_skip_and_the_report_prints_it(tmp_path):
    cfg = L8.build(tmp_path)
    _record(cfg)
    r = qa.evaluate(cfg)
    uc = r["unwrap_correction"]
    assert uc["flag"] is True and uc["skipped"] == ["phase_closure"]
    assert any("phase-closure unwrapping-error correction was skipped" in n for n in r["send_rule"]["notes"])
    assert r["send_rule"]["value"] != "send"
    # the verdict is the checks' count: the skip does not move it
    assert r["verdict"] == qa.verdict(qa.check_counts(r), thin=bool(r["far_field"].get("thin")))
    text = qa.format_report(r)
    assert ("[FLAG] unwrap_correction: requested=bridging+phase_closure, applied=bridging, "
            "dataset=unwrapPhase_bridging") in text
    assert "no region of at least 1046 pixels unwrapped in every kept pair" in text


def test_evaluate_without_a_record_has_no_block(tmp_path):
    r = qa.evaluate(L8.build(tmp_path, vel_mm=np.zeros((L8.NY, L8.NX))))
    assert "unwrap_correction" not in r
    assert not any("unwrapping-error correction" in n for n in r["send_rule"]["notes"])


# ------------------------------------------------------------------ the run's end record
def test_a_products_run_notes_the_skip_in_run_result(tmp_path, monkeypatch):
    cfg = R16._cfg(tmp_path)
    R16._stubs(monkeypatch)
    _record(cfg)
    assert pipeline.run_from(cfg, "products", watchdog=False, notify=False) == 0
    r = runs.result(cfg.project_dir)
    assert r["status"] == "done"
    assert any(n.startswith("The requested phase-closure unwrapping-error correction was skipped (asked for "
                            "bridging+phase_closure, applied bridging") and n.endswith(".") for n in r["notes"])
    assert "phase-closure unwrapping-error correction was skipped" in r["plain_cause"]
    # a run with everything it asked for applied has nothing to say
    _record(cfg, {"requested": "bridging", "applied": "bridging", "dataset": "unwrapPhase_bridging"})
    assert pipeline.unwrap_correction_note(cfg) is None


def test_an_unreadable_record_never_stops_the_run(tmp_path, monkeypatch):
    cfg = R16._cfg(tmp_path)
    R16._stubs(monkeypatch)

    def boom(cfg):
        raise PermissionError("mintpy/ not readable")
    monkeypatch.setattr(ts, "unwrap_error_shortfall", boom)
    assert pipeline.unwrap_correction_note(cfg) is None
    assert pipeline.run_from(cfg, "products", watchdog=False, notify=False) == 0
