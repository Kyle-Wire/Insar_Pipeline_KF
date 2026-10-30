"""7 Oct 2026 decisions D7 (the release gate's half) and D15, in release.py;
D7's provenance half confirmed.

D7: the report's send rule is authoritative only when client_report.json is
at least as new as quality_assessment.json; a QA re-run after the report is
a hard blocker ('report_older_than_qa'), and the gate's own send-rule line
then reads the QA's rule, marked as such. D15: a report area the report
calls "Not testable" (no verdict) is a reviewer item, not a hard blocker.

Each test failed on the code in backups/pre_fixes_20261007/ unless its
docstring says otherwise. Synthetic projects in tmp_path only; no real
project and no network."""
import json
import os

import pytest

import test_fix_L7b as L7b                                     # the gate's synthetic project
from insar_pipeline import provenance, release

FARMLAND_REASON = ("Too little ordinary farmland around this area to compare it with: an area this large fits "
                   "only about 3 separate times in the farmland measured (5 needed).")


@pytest.fixture(autouse=True)
def _identity_set(monkeypatch):
    from insar_pipeline import settings
    monkeypatch.setattr(settings, "identity", lambda: {"firm": "Test Firm", "preparer": "A. Engineer", "contact": ""})


def _signed(tmp_path, rep=None, qa=None):
    cfg = L7b._cfg(tmp_path)
    rep = rep or L7b._rep()
    D = L7b._project(tmp_path, rep, qa or L7b._qa())
    # a report held for a specialist is signed with one (16 Oct 2026, QUA-03)
    held = release._send_rule(tmp_path, rep, release._qa(tmp_path, None)).get("value") == release.HOLD
    release.sign_off(tmp_path, "K. Reviewer", "imagery checked", [f["n"] for f in release.features(rep)], rep,
                     specialist="K. Specialist" if held else "")
    return cfg, rep, D


def _qa_written(tmp_path, seconds_after_report: float, qa: dict | None = None):
    """quality_assessment.json (re)written this many seconds after the report."""
    f = tmp_path / "quality_assessment.json"
    if qa is not None:
        f.write_text(json.dumps(qa))
    t = (tmp_path / "deliverables" / "client_report.json").stat().st_mtime + seconds_after_report
    os.utime(f, (t, t))


def _codes(b, hard=None):
    return [x["code"] for x in b if hard is None or x["hard"] is hard]


# ------------------------------------------------------------------ D7: a QA re-run after the report
def test_a_qa_re_run_after_the_report_is_a_hard_blocker(tmp_path):
    cfg, rep, D = _signed(tmp_path)
    _qa_written(tmp_path, -600)
    assert release.is_final(cfg)                                            # the report follows its QA
    _qa_written(tmp_path, 600)                                              # the QA re-run on its own
    b = release.blockers(cfg)
    assert _codes(b, hard=True) == ["report_older_than_qa"], b
    txt = next(x["text"] for x in b if x["code"] == "report_older_than_qa")
    assert "predates the quality assessment" in txt and txt.endswith("rebuild the report.")
    assert release.is_final(cfg) is False and release.status(cfg)["final"] is False
    # the dashboard asks with the project folder alone
    assert "report_older_than_qa" in _codes(release.blockers(tmp_path))
    # rebuilt: the report is newer again
    (D / "client_report.json").write_text(json.dumps(rep))
    _qa_written(tmp_path, -60)
    assert release.is_final(cfg)


def test_a_qa_written_with_the_report_or_before_it_is_not_a_re_run(tmp_path):
    """The guard against false alarms (the old code had no such blocker to
    raise, nor the grace this reads). Files written in one breath (a script,
    a copy, a test fixture) are not a re-run; a QA re-run takes 20-30 s on
    Site B."""
    cfg, rep, D = _signed(tmp_path)
    for dt in (-3600, -1, 0.001, release.QA_AFTER_REPORT_GRACE_S - 0.5):
        _qa_written(tmp_path, dt)
        assert "report_older_than_qa" not in _codes(release.blockers(cfg)), dt
        assert release.is_final(cfg), dt
    _qa_written(tmp_path, release.QA_AFTER_REPORT_GRACE_S + 1)
    assert "report_older_than_qa" in _codes(release.blockers(cfg))


def test_a_report_being_built_is_not_judged_against_the_report_on_disk(tmp_path):
    """client_report.compute asks the gate for its release block before it
    writes client_report.json: the old file on disk predates the QA, but
    the report being built has just read the QA as it is. A guard against a
    false alarm (the old code had no such blocker)."""
    cfg, rep, D = _signed(tmp_path)
    _qa_written(tmp_path, 600)
    building = {**rep, "generated": "2026-10-07T12:00", "checks_flagged": 1}
    assert "report_older_than_qa" not in _codes(release.blockers(cfg, building, L7b._qa()))
    assert release.report_predates_qa(tmp_path, building) is None
    # the report on disk, passed in (the package, the KMZ) is judged
    on_disk = json.loads((D / "client_report.json").read_text())
    assert "report_older_than_qa" in _codes(release.blockers(cfg, on_disk, L7b._qa()))
    assert release.report_predates_qa(tmp_path) and release.report_predates_qa(tmp_path, on_disk)


def test_the_reports_own_qa_staleness_line_is_not_repeated_after_a_qa_re_run(tmp_path):
    """The report found its QA older than the velocity product when it was
    built; the QA has since been re-run. What is left is the rebuild, said
    once (report_older_than_qa), not also 're-run it' for a QA already
    re-run. The report's other stale inputs stay."""
    qa_line = "the quality assessment (quality_assessment.json) is older than the velocity product it describes; re-run it"
    summ_line = ("the product summary (point counts, per-cell shares) (summary.json) is older than the velocity "
                 "product it describes; re-run it")
    cfg, rep, D = _signed(tmp_path, rep=L7b._rep(stale_inputs=[qa_line, summ_line]))
    _qa_written(tmp_path, -600)
    stale = [x["text"] for x in release.blockers(cfg) if x["code"] == "stale_inputs"]
    assert any(t.startswith("The quality assessment (quality_assessment.json)") for t in stale)    # unchanged
    _qa_written(tmp_path, 600)
    b = release.blockers(cfg)
    stale = [x["text"] for x in b if x["code"] == "stale_inputs"]
    assert not any("quality assessment" in t for t in stale), stale
    assert any(t.startswith("The product summary") for t in stale)
    assert _codes(b).count("report_older_than_qa") == 1


def test_the_gates_send_rule_line_is_the_qas_when_the_report_is_out_of_date(tmp_path):
    """The report's rule (the QA's re-evaluated with the GNSS tie) speaks
    for the QA only while the report is at least as new; after a QA re-run
    the line said 'The quality assessment says ...' with the previous QA's
    verdict."""
    old = {"value": "send with notes", "reasons": ["rate precision flagged"], "gnss_considered": True}
    new = {"value": "hold for specialist", "reasons": ["the project area could not be compared with farmland"],
           "gnss_considered": False}
    cfg, rep, D = _signed(tmp_path, rep=L7b._rep(send_rule=old), qa={**L7b._qa(), "send_rule": old})
    _qa_written(tmp_path, -600)
    line = next(x for x in release.blockers(cfg) if x["code"] == "qa_send_rule")
    assert "'send with notes'" in line["text"] and "out of date" not in line["text"] and line["hard"] is False
    _qa_written(tmp_path, 600, qa={**L7b._qa(), "send_rule": new})
    line = next(x for x in release.blockers(cfg) if x["code"] == "qa_send_rule")
    assert line["text"].startswith("The quality assessment says 'hold for specialist' (report out of date): ")
    # hard since 16 Oct 2026 (readiness review QUA-03): 'hold for specialist' needs a named specialist
    assert "could not be compared with farmland" in line["text"] and line["hard"] is True
    # a re-run QA that says 'send' gives no rule line, but the report still has to be rebuilt
    _qa_written(tmp_path, 600, qa={**L7b._qa(), "send_rule": {"value": "send", "reasons": []}})
    b = release.blockers(cfg)
    assert "qa_send_rule" not in _codes(b) and "report_older_than_qa" in _codes(b, hard=True)


def test_the_gates_send_rule_line_follows_the_dashboard_and_provenance_after_a_qa_re_run(tmp_path):
    """Review of this group (7 Oct 2026): after a QA re-run the gate's line
    read the QA's own rule, which never saw the GNSS tie, while the dashboard
    (send_rule_shown) and provenance read summary.json's, which
    quality_assessment.write re-evaluates with the tie straight after the
    QA. With an unusable tie the gate said 'send with notes' beside their
    'hold for specialist'. Failed on the group's first version."""
    own = {"value": "send with notes", "reasons": ["rate precision flagged"], "gnss_considered": False}
    cfg, rep, D = _signed(tmp_path, rep=L7b._rep(send_rule=own), qa={**L7b._qa(), "send_rule": own})
    _qa_written(tmp_path, 600, qa={**L7b._qa(), "send_rule": own})
    S = D / "summary.json"
    S.write_text(json.dumps({"send_rule": "hold for specialist", "send_rule_gnss_considered": True}))
    t = (tmp_path / "quality_assessment.json").stat().st_mtime + 1
    os.utime(S, (t, t))
    line = next(x for x in release.blockers(cfg) if x["code"] == "qa_send_rule")
    assert line["text"].startswith("The quality assessment says 'hold for specialist' (report out of date): ")
    assert "GNSS tie" in line["text"] and "rate precision" not in line["text"]
    assert provenance.stages(cfg)["quality_assessment"]["send_rule"]["value"] == "hold for specialist"
    # the same value: the QA's own reasons stay
    S.write_text(json.dumps({"send_rule": "send with notes", "send_rule_gnss_considered": True}))
    os.utime(S, (t, t))
    line = next(x for x in release.blockers(cfg) if x["code"] == "qa_send_rule")
    assert line["text"] == "The quality assessment says 'send with notes' (report out of date): rate precision flagged."
    # a summary from before this QA, or one not re-evaluated with the tie, is not read
    S.write_text(json.dumps({"send_rule": "hold for specialist", "send_rule_gnss_considered": True}))
    os.utime(S, (t - 1200, t - 1200))
    assert "'send with notes'" in next(x for x in release.blockers(cfg) if x["code"] == "qa_send_rule")["text"]
    S.write_text(json.dumps({"send_rule": "hold for specialist", "send_rule_gnss_considered": False}))
    os.utime(S, (t, t))
    assert "'send with notes'" in next(x for x in release.blockers(cfg) if x["code"] == "qa_send_rule")["text"]
    # the report at least as new as the QA: its own rule, unmarked, whatever summary.json says
    (D / "client_report.json").write_text(json.dumps(rep))
    _qa_written(tmp_path, -60)
    line = next(x for x in release.blockers(cfg) if x["code"] == "qa_send_rule")
    assert line["text"] == "The quality assessment says 'send with notes': rate precision flagged."


# ------------------------------------------------------------------ D7: provenance already follows it
def test_provenance_takes_the_reports_rule_only_when_the_report_is_at_least_as_new(tmp_path):
    """Confirms D7's provenance half; passes on the old code too."""
    rule_rep = {"value": "hold for specialist", "reasons": ["no GNSS anchor"], "gnss_considered": True}
    rule_qa = {"value": "send with notes", "reasons": ["rate precision flagged"], "gnss_considered": False}
    cfg, rep, D = _signed(tmp_path, rep=L7b._rep(send_rule=rule_rep), qa={**L7b._qa(), "send_rule": rule_qa})
    _qa_written(tmp_path, -1)
    q = provenance.stages(cfg)["quality_assessment"]
    assert q["send_rule"] == rule_rep and q["send_rule_source"] == "client_report.json"
    _qa_written(tmp_path, 600)
    q = provenance.stages(cfg)["quality_assessment"]
    assert q["send_rule"] == rule_qa and q["send_rule_source"] == "quality_assessment.json"


# ------------------------------------------------------------------ D15: a "Not testable" area
def _not_testable_rep(measured=97.9, reason=FARMLAND_REASON):
    rep = L7b._rep(area_name="Site B project area", measured=measured, testable=False)
    a = rep["areas"][1]
    a["not_testable_reason"] = reason
    for k in ("detected", "z", "range95_mm_yr"):
        a.pop(k, None)
    a["reading"] = "Not testable"
    return rep


def test_a_not_testable_area_is_a_reviewer_item_not_a_hard_blocker(tmp_path):
    cfg, rep, D = _signed(tmp_path, rep=_not_testable_rep())
    _qa_written(tmp_path, -600)
    b = release.blockers(cfg)
    item = next(x for x in b if x["code"] == "area_not_testable")
    assert item["hard"] is False
    t = item["text"]
    assert t.startswith("'Site B project area' is reported as not testable, with no verdict")
    assert "fits only about 3 separate times in the farmland measured (5 needed). Read that" in t
    assert t.endswith("accept it before signing off.")
    # the honest report can be issued once it is signed off
    assert not _codes(b, hard=True), b
    assert release.is_final(cfg) and release.status(cfg)["final"] is True
    # and is listed under [read] by the command line
    assert "[read] 'Site B project area' is reported as not testable" in "\n".join(
        ("  [must] " if x["hard"] else "  [read] ") + x["text"] for x in b)


def test_a_not_testable_reason_reads_as_a_sentence(tmp_path):
    """client_report.tested_fields' fallback reason starts in lower case;
    the item read 'Reason given: no comparison ...'. Failed on the group's
    first version."""
    cfg, rep, D = _signed(tmp_path, rep=_not_testable_rep(reason="no comparison with ordinary farmland could be made"))
    t = next(x["text"] for x in release.blockers(cfg) if x["code"] == "area_not_testable")
    assert "Reason given: No comparison with ordinary farmland could be made. Read that" in t
    rep2 = _not_testable_rep(reason=None)
    (D / "client_report.json").write_text(json.dumps(rep2))
    t = next(x["text"] for x in release.blockers(cfg, rep2) if x["code"] == "area_not_testable")
    assert t == ("'Site B project area' is reported as not testable, with no verdict: it could not be compared "
                 "with ordinary ground. Read that in the report and accept it before signing off.")


def test_an_under_measured_area_stays_a_hard_blocker(tmp_path):
    cfg, rep, D = _signed(tmp_path, rep=_not_testable_rep(measured=31.0, reason="only 31% of the area was measured"))
    b = release.blockers(cfg)
    assert "area_under_measured" in _codes(b, hard=True)
    assert "area_not_testable" in _codes(b, hard=False) and "area_not_testable" not in _codes(b, hard=True)
    assert release.is_final(cfg) is False


def test_a_legacy_area_with_no_comparison_stays_a_hard_blocker(tmp_path):
    """A report from before 5 Oct 2026 carries no 'testable' and, here, no
    farmland comparison: it does not say the area is not testable, it
    prints a figure with nothing to read it by. Still hard (was
    'area_not_testable'; now 'area_not_compared', so the soft D15 item
    is only ever the report's own "Not testable")."""
    rep = L7b._rep()
    for k in ("testable", "detected", "z"):
        rep["areas"][1].pop(k)
    cfg, rep, D = _signed(tmp_path, rep=rep)
    b = release.blockers(cfg)
    assert "area_not_compared" in _codes(b, hard=True) and "area_not_testable" not in _codes(b)
    assert release.is_final(cfg) is False
