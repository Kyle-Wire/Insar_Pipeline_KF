"""7 Oct 2026 integration round, group I2_release (release.py, provenance.py,
results.py). Each test failed on the code before this round's I2 changes
unless its docstring says it is a guard. Synthetic projects in tmp_path only;
no real project and no network.

  (a) the release gate reads cross_check.json only when it is this run's
      (crosscheck.record_is_current: the option on, written no earlier than
      the velocity product)
  (b) provenance records the send rule the report and the release list use
      (re-evaluated with the GNSS tie), and says where it came from
  (c) a Results export carries the report's numbers_hash, so the package
      keeps it through the sign-off rebuild (#46, producer side)
"""
import json
import os
import time
from datetime import datetime
from pathlib import Path

import pytest

import test_fix_L7b as L7b                                     # the gate's synthetic project
from insar_pipeline import package, provenance, release


@pytest.fixture(autouse=True)
def _identity_set(monkeypatch):
    from insar_pipeline import settings
    monkeypatch.setattr(settings, "identity", lambda: {"firm": "Test Firm", "preparer": "A. Engineer", "contact": ""})


def _age(p: Path, seconds: float):
    t = time.time() - seconds
    os.utime(p, (t, t))


def _iso(seconds_ago: float) -> str:
    return datetime.fromtimestamp(time.time() - seconds_ago).isoformat(timespec="seconds")


# ------------------------------------------------------------------ (a) the gate and cross_check.json
def _gate_project(tmp_path, **cfg_kw):
    """A project whose velocity product was written an hour ago."""
    cfg = L7b._cfg(tmp_path, **cfg_kw)
    rep = L7b._rep()
    L7b._project(tmp_path, rep, L7b._qa())
    vel = cfg.mintpy_dir / "geo" / "geo_velocity.h5"
    vel.parent.mkdir(parents=True, exist_ok=True)
    vel.write_bytes(b"not read, only dated")
    _age(vel, 3600)
    return cfg, rep


def _disagreeing_cross_check(tmp_path, generated: str | None):
    T = tmp_path / "deliverables" / "technical"
    T.mkdir(parents=True, exist_ok=True)
    rec = {"status": "complete", "available": True, "indicative_pass": False,
           "areas": [{"name": "Area A1", "agree": False}]}
    if generated is not None:
        rec["generated"] = generated
    (T / "cross_check.json").write_text(json.dumps(rec))


def _codes(cfg, rep):
    return {b["code"] for b in release.blockers(cfg, rep, L7b._qa())}


def test_a_cross_check_older_than_the_velocity_product_is_not_read(tmp_path):
    """It checked a previous MintPy run: its disagreement is not this record's."""
    cfg, rep = _gate_project(tmp_path, cross_check_disp=True)
    _disagreeing_cross_check(tmp_path, _iso(7200))              # before the velocity product
    assert "cross_check_disagrees" not in _codes(cfg, rep)
    _disagreeing_cross_check(tmp_path, _iso(60))                # this run's
    assert "cross_check_disagrees" in _codes(cfg, rep)


def test_a_cross_check_left_after_the_option_was_turned_off_is_not_read(tmp_path):
    cfg, rep = _gate_project(tmp_path, cross_check_disp=False)
    _disagreeing_cross_check(tmp_path, _iso(60))
    assert "cross_check_disagrees" not in _codes(cfg, rep)


def test_a_cross_check_without_a_date_is_not_read(tmp_path):
    """A record that cannot say when it was made cannot be placed after the
    velocity product (one from before the stage stamped 'generated')."""
    cfg, rep = _gate_project(tmp_path, cross_check_disp=True)
    _disagreeing_cross_check(tmp_path, None)
    assert "cross_check_disagrees" not in _codes(cfg, rep)


def test_the_cross_check_never_moves_final_and_a_bare_folder_leaves_it_out(tmp_path):
    """The first half is a guard (it passed on the old code too): the
    cross-check line is soft, so a record never changes the hard list. The
    dashboard's projects table passes only the folder, which carries no
    option: the gate still answers, without the cross-check line."""
    cfg, rep = _gate_project(tmp_path, cross_check_disp=True)
    L7b._sign_all(tmp_path, rep)
    _disagreeing_cross_check(tmp_path, _iso(60))
    hard_with = {b["code"] for b in release.hard_blockers(cfg, rep, L7b._qa())}
    (tmp_path / "deliverables" / "technical" / "cross_check.json").unlink()
    hard_without = {b["code"] for b in release.hard_blockers(cfg, rep, L7b._qa())}
    assert hard_with == hard_without
    _disagreeing_cross_check(tmp_path, _iso(60))
    codes = {b["code"] for b in release.blockers(tmp_path)}
    assert "cross_check_disagrees" not in codes and "no_report" not in codes


# ------------------------------------------------------------------ (b) provenance's send rule
SR_QA = {"value": "send with notes", "reasons": ["rate precision flagged"], "gnss_considered": False}
SR_TIE = {"value": "hold for specialist", "reasons": ["no GNSS anchor: no station within 50 km"],
          "gnss_considered": True}


def _sr_project(tmp_path, rep_after_qa: bool | None, summary: dict | None, summary_after_qa: bool = True):
    cfg = L7b._cfg(tmp_path)
    D = tmp_path / "deliverables"
    D.mkdir(parents=True, exist_ok=True)
    qa_f = tmp_path / "quality_assessment.json"
    qa_f.write_text(json.dumps({**L7b._qa(), "send_rule": SR_QA}))
    _age(qa_f, 1800)
    if rep_after_qa is not None:
        f = D / "client_report.json"
        f.write_text(json.dumps(L7b._rep(send_rule=SR_TIE)))
        _age(f, 60 if rep_after_qa else 3600)
    if summary is not None:
        f = D / "summary.json"
        f.write_text(json.dumps(summary))
        _age(f, 60 if summary_after_qa else 3600)
    return cfg


def _qa_stage(cfg):
    return provenance.stages(cfg)["quality_assessment"]


def test_provenance_records_the_reports_send_rule_not_the_qa_time_one(tmp_path):
    """The QA runs before the GNSS tie: its stored rule said 'send with notes'
    where the report (and the release list) hold it for want of an anchor."""
    q = _qa_stage(_sr_project(tmp_path, rep_after_qa=True, summary=None))
    assert q["send_rule"]["value"] == "hold for specialist" and q["send_rule"]["gnss_considered"] is True
    assert q["send_rule_source"] == "client_report.json"


def test_provenance_takes_summary_json_when_the_report_predates_the_qa(tmp_path):
    summ = {"send_rule": "hold for specialist", "send_rule_gnss_considered": True}
    q = _qa_stage(_sr_project(tmp_path, rep_after_qa=False, summary=summ))
    assert q["send_rule"] == {"value": "hold for specialist", "gnss_considered": True}
    assert q["send_rule_source"] == "summary.json"


def test_provenance_falls_back_to_the_qa_rule(tmp_path):
    """An old report and an old summary, or a summary whose rule never saw a
    tie: the QA's own rule, said to be the QA's."""
    summ = {"send_rule": "hold for specialist", "send_rule_gnss_considered": True}
    q = _qa_stage(_sr_project(tmp_path, rep_after_qa=False, summary=summ, summary_after_qa=False))
    assert q["send_rule"] == SR_QA and q["send_rule_source"] == "quality_assessment.json"
    T2 = tmp_path / "second"
    T2.mkdir()
    q = _qa_stage(_sr_project(T2, rep_after_qa=None,
                              summary={"send_rule": "send with notes", "send_rule_gnss_considered": False}))
    assert q["send_rule"] == SR_QA and q["send_rule_source"] == "quality_assessment.json"


def test_provenance_says_whether_an_optional_stage_record_is_used(tmp_path):
    cfg, rep = _gate_project(tmp_path, cross_check_disp=True)
    _disagreeing_cross_check(tmp_path, _iso(60))
    (tmp_path / "deliverables" / "technical" / "injection_recovery.json").write_text(
        json.dumps({"status": "complete", "seed": 1, "generated": _iso(60)}))     # injection_test is off
    st = provenance.stages(cfg)
    assert st["cross_check"]["current"] is True and st["injection_test"]["current"] is False
    _disagreeing_cross_check(tmp_path, _iso(7200))
    assert provenance.stages(cfg)["cross_check"]["current"] is False


# ------------------------------------------------------------------ (c) #46: the export carries the report's hash
@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    """A built report, one report area analysed and exported to
    deliverables/results/ (where the package looks)."""
    import test_fix_L9b as L9b
    from insar_pipeline import client_report as cr
    from insar_pipeline import results as rs
    from insar_pipeline import settings
    mp = pytest.MonkeyPatch()
    mp.setattr(settings, "identity", lambda: {"firm": "Test Firm", "preparer": "A. Engineer", "contact": ""})
    root = tmp_path_factory.mktemp("i2")
    cfg = L9b.make_project(root)
    cr.build(cfg)
    D = cfg.project_dir / "deliverables"
    rep = json.loads((D / "client_report.json").read_text())
    res = rs.analyse(cfg, "Test Field")
    out = rs.export(cfg, res)
    mp.undo()
    return cfg, rep, res, out, D


def _plan_results(cfg, rep):
    copies, not_copied, _ = package._plan(cfg, rep, draft=True)
    taken = [d for _, d, _, _ in copies if d.endswith("points_yearly_rates_and_statistics.xlsx")]
    left = [why for s, _, why in not_copied if s.suffix == ".xlsx"]
    return taken, left


def test_results_export_writes_the_reports_numbers_hash(exported):
    cfg, rep, res, out, D = exported
    assert out.parent == D / "results"
    summ = json.loads((out / "summary.json").read_text())
    assert summ["numbers_hash"] == release.numbers_hash(rep) == res.numbers_hash


def test_an_export_survives_the_sign_off_rebuild_and_not_a_changed_number(exported):
    """End to end with package._results_problem: the sign-off rebuild rewrites
    client_report.json with the same numbers (a later 'generated', a release
    block), after the export; the workbook stays in client/. A changed
    number leaves it out."""
    cfg, rep, res, out, D = exported
    rep_f = D / "client_report.json"
    original = rep_f.read_text()
    try:
        for p in out.iterdir():
            _age(p, 3600)                                       # exported an hour before signing
        rebuilt = json.loads(original)
        rebuilt["generated"] = _iso(0)[:16]
        rebuilt["release"] = {"final": True, "blockers": [], "reviewed_by": "K. Reviewer"}
        rep_f.write_text(json.dumps(rebuilt))
        assert release.numbers_hash(rebuilt) == release.numbers_hash(rep)
        taken, left = _plan_results(cfg, rebuilt)
        assert taken and not left, left
        changed = json.loads(original)
        changed["areas"][1]["vertical_mm_yr"] = (changed["areas"][1].get("vertical_mm_yr") or 0) + 1.0
        rep_f.write_text(json.dumps(changed))
        taken, left = _plan_results(cfg, changed)
        assert not taken and any("earlier version of the report's numbers" in w for w in left), left
    finally:
        rep_f.write_text(original)


def test_the_hash_is_the_analysed_reports_not_one_rebuilt_before_export(exported, tmp_path):
    """The figures came from the report analyse read: a report whose numbers
    changed between analysing and exporting must not be stamped on them."""
    from insar_pipeline import results as rs
    cfg, rep, res, out, D = exported
    rep_f = D / "client_report.json"
    original = rep_f.read_text()
    try:
        changed = json.loads(original)
        changed["areas"][1]["vertical_mm_yr"] = (changed["areas"][1].get("vertical_mm_yr") or 0) + 1.0
        rep_f.write_text(json.dumps(changed))
        o2 = rs.export(cfg, res, out_dir=tmp_path / "x")
        assert json.loads((o2 / "summary.json").read_text())["numbers_hash"] == release.numbers_hash(rep)
        # a Result that analyse did not make takes the report on disk; no report, no hash
        import dataclasses
        bare = dataclasses.replace(res, numbers_hash=None)
        o3 = rs.export(cfg, bare, out_dir=tmp_path / "y")
        assert json.loads((o3 / "summary.json").read_text())["numbers_hash"] == release.numbers_hash(changed)
        o4 = rs.export(cfg, dataclasses.replace(res, numbers_hash=""), out_dir=tmp_path / "z")
        assert json.loads((o4 / "summary.json").read_text())["numbers_hash"] is None
    finally:
        rep_f.write_text(original)
