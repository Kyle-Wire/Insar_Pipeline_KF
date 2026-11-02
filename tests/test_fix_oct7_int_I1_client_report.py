"""7 Oct 2026 integration fixes, group I1_client_report (client_report.py):
findings 11, 22, 43 and 48 of the 5 Oct review and two round-1 hand-offs.

  11  gnss_tie.check's own reason ('gnss_tie.json is older than ...') reached
      the client page; the page now says it plainly, the technical reason
      stays in gnss.tie_not_used and the send rule's reasons
  22  the send rule judged a report area by the QA's first-pass testability
      while the area table used the report's own
  43  the area PDF is written aside and moved into place; a failure leaves no
      PDF rather than the last build's or a truncated one
  48  cross_check.json / injection_recovery.json are used only when they are
      this run's (crosscheck.record_is_current), and the cross-check's
      per-feature sentence is redone for this report's features
  --  _plural_reason covers the date-accounting reasons that open with 'it'
  --  a tie gnss_tie.check refused holds the send rule, as summary.json does

Each test checks the outcome and failed on the code before these fixes.

Run: python -m pytest insar_pipeline/tests/test_fix_oct7_int_I1_client_report.py -q -p no:cacheprovider
"""
import copy
import json
import os
import re
import sys
import time
import types
from datetime import datetime
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_fix_L5 as L5                                          # noqa: E402  (the client-report project builder)

from insar_pipeline import client_report as cr                   # noqa: E402
from insar_pipeline import crosscheck as xc                      # noqa: E402
from insar_pipeline import quality_assessment as qa              # noqa: E402

# gnss_tie.check's own words; the DRAFT box's release list may still say
# 're-run the GNSS tie (it does not match the current velocity product)'
TECH_WORDS = ("gnss_tie.json", "it would anchor", "too narrow to use", "not this record's")


def _stamp(t: float) -> str:
    return datetime.fromtimestamp(t).isoformat(timespec="seconds")


def _velocity(cfg) -> Path:
    return xc.product(cfg, "velocity.h5")


# ------------------------------------------------------------------ one project for 11, 22, 48 and the send rule
@pytest.fixture(scope="module")
def stale(tmp_path_factory):
    """A project whose GNSS tie is older than the velocity product (gnss_tie.
    check refuses it), whose QA judged the 70%-unmeasured 'Wet Pit' testable
    and 'Test Field' not, with a current cross-check and an injection record
    older than the velocity product."""
    cfg = L5.make_project(tmp_path_factory.mktemp("i1"), gnss=True)
    cfg.cross_check_disp = True
    cfg.injection_test = True
    P = cfg.project_dir
    qf = P / "quality_assessment.json"
    Q = json.loads(qf.read_text())
    Q["send_rule"] = {"value": "send", "reasons": [], "hold_reasons": [], "notes": []}
    Q["area_testability"] = {"Wet Pit": {"kind": "area", "testable": True, "not_testable_reason": None},
                             "Test Field": {"kind": "area", "testable": False,
                                            "not_testable_reason": "the QA's first pass said so"}}
    qf.write_text(json.dumps(Q))
    vt = _velocity(cfg).stat().st_mtime
    tie = P / "deliverables" / "gnss_tie.json"
    os.utime(tie, (vt - 3600, vt - 3600))                    # written before the product it would anchor
    T = P / "deliverables" / "technical"
    T.mkdir(parents=True, exist_ok=True)
    (T / "injection_recovery.json").write_text(json.dumps(
        {"status": "complete", "sentence": "An earlier run's detection table.", "generated": _stamp(vt - 86400)}))
    (T / "cross_check.json").write_text(json.dumps(
        {"status": "complete", "available": True, "generated": _stamp(time.time() + 5),
         "common_window": ["20160105", "20250320"], "one_km_means": {"correlation": 0.8},
         "areas": [{"name": "Test Field", "agree": True, "combined_95_mm_yr": 3.0}],
         # the PREVIOUS report's three features
         "features": [{"index": i, "confirmed": True} for i in range(3)],
         "sentence": "Old sentence; 3 of 3 reported features also stand out in it."}))
    seen = {}

    def recheck(cfg_, rec, feats):
        seen["feats"] = feats
        out = dict(rec, features=[{"index": i, "confirmed": i == 0, "cells": 50} for i in range(len(feats))])
        out["sentence"] = xc.sentence(out)
        return out

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(xc, "recheck_features", recheck)
        rep, ctx, masks, outlines = cr.compute(cfg)
    return types.SimpleNamespace(cfg=cfg, rep=rep, ctx=ctx, masks=masks, outlines=outlines, seen=seen, Q=Q)


def _page(built):
    return cr.page(built.cfg, built.rep, L5._fake_figs(built))


# ------------------------------------------------------------------ 11: plain GNSS reason on the page
def test_a_refused_tie_reaches_the_page_in_plain_words(stale):
    rep = stale.rep
    assert rep["anchored"] is False
    assert rep["gnss_reason"] == cr.GNSS_TIE_OUT_OF_DATE and rep["gnss"]["reason"] == cr.GNSS_TIE_OUT_OF_DATE
    # the technical reason is kept for the reviewer
    assert "older than the velocity product" in rep["gnss"]["tie_not_used"]
    html = _page(stale)
    assert "No GNSS station could anchor these figures (" + cr.GNSS_TIE_OUT_OF_DATE + ")" in html
    assert not any(w in html for w in TECH_WORDS), [w for w in TECH_WORDS if w in html]
    notes = " ".join(rep.get("quality_notes") or [])
    assert "out of date" in notes and "gnss_tie.json" not in notes


def test_client_gnss_reason_maps_every_check_refusal_and_keeps_a_ties_own_reason():
    f = cr.client_gnss_reason
    assert f({}, "no gnss_tie.json (the GNSS tie has not run)") == cr.GNSS_TIE_NOT_RUN
    assert f({}, "no GNSS tie on disk") == cr.GNSS_TIE_NOT_RUN
    assert f({}, "gnss_tie.json is not readable (Expecting value)") == cr.GNSS_TIE_UNREADABLE
    for why in ("gnss_tie.json covers ['20160105', '20240101'], not this record's ['20160105', '20250320']",
                "gnss_tie.json is older than the velocity product it would anchor",
                "the GNSS tie on disk was made before its uncertainty allowed for how much GNSS stations across the "
                "region differ, so its range is too narrow to use; re-run the GNSS tie"):
        assert f({}, why) == cr.GNSS_TIE_OUT_OF_DATE
    # this record's own unanchored tie already says why in plain words
    own = "No continuous GNSS station lies within 40 km of the area, so every figure is relative to the farmland."
    assert f({"background": None, "reason": own}, own) == own
    assert f({"background": {"vertical_mm_yr": -1.0}}, "") == ""


def test_a_missing_tie_reads_not_run_on_the_page(tmp_path):
    cfg = L5.make_project(tmp_path, gnss=False)
    rep, ctx, masks, outlines = cr.compute(cfg)
    built = types.SimpleNamespace(cfg=cfg, rep=rep)
    assert "gnss_tie.json" not in rep["gnss_reason"] and "gnss_tie.json" not in _page(built)
    assert rep["gnss_reason"] == cr.GNSS_TIE_NOT_RUN and "gnss_tie.json" in rep["gnss"]["tie_not_used"]


# ------------------------------------------------------------------ send rule: a refused tie holds, as summary.json does
def test_a_refused_tie_holds_the_send_rule_with_the_technical_reason(stale):
    sr = stale.rep["send_rule"]
    assert sr["gnss_considered"] is True and sr["value"] == "hold for specialist"
    gn = [r for r in sr["hold_reasons"] if r.startswith("no GNSS anchor")]
    assert gn and "older than the velocity product" in gn[0]


def test_a_pre_floor_tie_holds_the_report_and_summary_alike(tmp_path):
    """deliverables._apply_qa_and_gnss holds summary.json on a pre-floor tie;
    the report's rule now says the same."""
    from insar_pipeline import deliverables as dl
    cfg = L5.make_project(tmp_path, gnss=True)
    tie_f = cfg.project_dir / "deliverables" / "gnss_tie.json"
    tie = json.loads(tie_f.read_text())
    tie.pop("uncertainty_floor_mm_yr")
    tie["background"].pop("uncertainty_basis")
    tie_f.write_text(json.dumps(tie))
    qa_path = cfg.project_dir / "quality_assessment.json"
    Q = json.loads(qa_path.read_text())
    Q["send_rule"] = {"value": "send", "reasons": []}
    qa_path.write_text(json.dumps(Q))
    report = cr.compute(cfg)[0]["send_rule"]
    summary = dl._apply_qa_and_gnss(cfg, {})
    assert report["value"] == summary["send_rule"] == "hold for specialist"
    assert any(r.startswith("no GNSS anchor") and "too narrow" in r for r in report["hold_reasons"])


def test_no_tie_file_leaves_the_qas_rule_as_summary_json_does(tmp_path):
    cfg = types.SimpleNamespace(project_dir=tmp_path)
    assert cr._tie_for_send_rule(cfg, {}, "no gnss_tie.json (the GNSS tie has not run)") is None
    own = {"background": None, "reason": "no station"}
    assert cr._tie_for_send_rule(cfg, own, "no station") is own


# ------------------------------------------------------------------ 22: the send rule judges areas as the table does
def test_send_rule_judges_report_areas_by_the_reports_own_testability(stale):
    rep = stale.rep
    pit = next(a for a in rep["areas"] if a["name"] == "Wet Pit")
    field = next(a for a in rep["areas"] if a["name"] == "Test Field")
    assert pit["testable"] is False and field["testable"] is True
    holds = rep["send_rule"]["hold_reasons"]
    pit_hold = [r for r in holds if "'Wet Pit'" in r]
    assert pit_hold and pit["not_testable_reason"].rstrip(".")[:40] in pit_hold[0]
    assert not any("'Test Field'" in r for r in holds)
    # the QA on disk is not changed
    assert json.loads((stale.cfg.project_dir / "quality_assessment.json").read_text()) == stale.Q


def test_send_rule_override_keeps_the_qas_kind_and_leaves_q_alone():
    Q = {"checks": {}, "findings": {}, "send_rule": {"value": "send", "reasons": []},
         "area_testability": {"A": {"kind": "area", "testable": True, "not_testable_reason": None},
                              "M": {"kind": "panel", "testable": True, "not_testable_reason": None}}}
    before = copy.deepcopy(Q)
    areas = [{"name": "Whole", "kind": "full", "testable": None},
             {"name": "A", "kind": "area", "testable": False, "not_testable_reason": "only 30% measured"},
             {"name": "M", "kind": "area", "testable": False, "not_testable_reason": "x"},
             {"name": "Z", "kind": "area", "testable": False, "not_testable_reason": "not in the QA"}]
    got = cr._send_rule(Q, None, areas)
    assert Q == before
    assert got["value"] == "hold for specialist"
    assert got["hold_reasons"] == ["report area 'A' cannot be compared with farmland: only 30% measured"]


# ------------------------------------------------------------------ 48: records of this run only; features re-measured
def test_report_uses_only_current_records_and_this_reports_features(stale):
    rep = stale.rep
    inj, ic = rep["injection_test"], rep["independent_check"]
    assert inj["status"] == "out of date" and inj["sentence"] is None
    assert "older" not in (inj["sentence"] or "") and "earlier run" in inj["reason"]
    # recheck_features got exactly the features compute() built
    feats = stale.seen.get("feats")
    assert feats is not None and len(feats) == len(rep["features"]) >= 1
    assert [f["lat"] for f in feats] == [f["lat"] for f in rep["features"]]
    n = len(rep["features"])
    assert f"1 of {n} reported features" in ic["sentence"] and "3 of 3" not in ic["sentence"]
    assert len(ic["features"]) == n and ic["features"][0]["confirmed"] is True
    html = _page(stale)
    assert "An earlier run's detection table." not in html and f"1 of {n} reported features" in html


def _cfg(tmp_path, cc=True, inj=True):
    cfg = types.SimpleNamespace(project_dir=tmp_path / "proj", mintpy_dir=tmp_path / "proj" / "mintpy",
                                reference_auto=False, cross_check_disp=cc, injection_test=inj)
    v = cfg.mintpy_dir / "velocity.h5"
    v.parent.mkdir(parents=True)
    v.write_bytes(b"")
    T = cfg.project_dir / "deliverables" / "technical"
    T.mkdir(parents=True)
    return cfg, v.stat().st_mtime, T


CC = {"status": "complete", "available": True, "common_window": ["20200101", "20221231"],
      "one_km_means": {"correlation": 0.8}, "features": [{"index": i, "confirmed": True} for i in range(3)],
      "sentence": "Old; 3 of 3 reported features also stand out in it."}


@pytest.mark.parametrize("option_on", [True, False])
def test_stale_or_switched_off_records_are_not_this_records(tmp_path, option_on):
    cfg, vt, T = _cfg(tmp_path, cc=option_on, inj=option_on)
    old = _stamp(vt - 3600) if option_on else _stamp(vt + 60)
    (T / "injection_recovery.json").write_text(json.dumps({"status": "complete", "sentence": "Old.", "generated": old}))
    (T / "cross_check.json").write_text(json.dumps(dict(CC, generated=old)))
    b = cr._record_blocks(cfg, {}, {}, {}, [])
    assert b["injection_test"]["sentence"] is None and b["injection_test"]["status"] == "out of date"
    assert b["independent_check"]["available"] is False and b["independent_check"]["sentence"] is None
    assert b["independent_check"]["status"] == "out of date"
    assert ("switched off" in b["independent_check"]["reason"]) is (not option_on)


def test_current_records_are_used_and_features_redone(tmp_path, monkeypatch):
    cfg, vt, T = _cfg(tmp_path)
    now = _stamp(vt + 60)
    (T / "injection_recovery.json").write_text(json.dumps({"status": "complete", "sentence": "New.", "generated": now}))
    (T / "cross_check.json").write_text(json.dumps(dict(CC, generated=now)))
    calls = []

    def recheck(cfg_, rec, feats):
        calls.append(feats)
        out = dict(rec, features=[{"index": 0, "confirmed": False, "cells": 9}, {"index": 1, "confirmed": True}])
        out["sentence"] = xc.sentence(out)
        return out
    monkeypatch.setattr(xc, "recheck_features", recheck)
    feats = [{"lat": 38.3, "lon": -87.5}, {"lat": 38.31, "lon": -87.51}]
    b = cr._record_blocks(cfg, {}, {}, {}, feats)
    assert calls == [feats]
    # E4 / C5 (trust review 7 Oct 2026): technical only - no client sentence
    assert b["injection_test"]["sentence"] is None and b["injection_test"]["status"] == "complete"
    ic = b["independent_check"]
    assert "1 of 2 reported features" in ic["sentence"] and [f["index"] for f in ic["features"]] == [0, 1]


def test_no_features_or_a_failed_recheck_drops_the_features_clause(tmp_path, monkeypatch):
    cfg, vt, T = _cfg(tmp_path)
    (T / "cross_check.json").write_text(json.dumps(dict(CC, generated=_stamp(vt + 60))))

    def boom(*a):
        raise RuntimeError("DISP grid not readable")
    monkeypatch.setattr(xc, "recheck_features", boom)
    for feats in ([], [{"lat": 38.3, "lon": -87.5}]):            # none: not even attempted; one: the recheck fails
        ic = cr._record_blocks(cfg, {}, {}, {}, feats)["independent_check"]
        assert ic["available"] is True and ic["sentence"] and "reported features" not in ic["sentence"]
        assert ic["features"] == [] and "matches the record's pattern" in ic["sentence"]


def test_a_recheck_that_returns_the_record_unchanged_drops_the_old_clause(tmp_path, monkeypatch):
    cfg, vt, T = _cfg(tmp_path)
    (T / "cross_check.json").write_text(json.dumps(dict(CC, generated=_stamp(vt + 60))))
    monkeypatch.setattr(xc, "recheck_features", lambda cfg_, rec, feats: rec)
    ic = cr._record_blocks(cfg, {}, {}, {}, [{"lat": 38.3, "lon": -87.5}])["independent_check"]
    assert "3 of 3" not in ic["sentence"] and ic["features"] == []


# ------------------------------------------------------------------ 43: the PDF is written aside and moved into place
def _fake_build(monkeypatch, pdf_writer):
    monkeypatch.setattr(cr, "compute", lambda cfg: ({"title": "T"}, None, {}, {}))
    monkeypatch.setattr(cr, "figures", lambda *a, **k: {})
    monkeypatch.setattr(cr, "write_rasters", lambda *a, **k: None)
    monkeypatch.setattr(cr, "page", lambda *a, **k: "<html></html>")
    monkeypatch.setattr(cr, "area_pdf", pdf_writer)


def test_a_failed_area_pdf_leaves_no_pdf_at_all(tmp_path, monkeypatch):
    cfg = types.SimpleNamespace(project_dir=tmp_path)
    D = tmp_path / "deliverables"
    D.mkdir()
    (D / cr.PDF_NAME).write_bytes(b"%PDF the last build's")

    def truncated(cfg_, rep, ctx, outlines, path, figs=None):
        Path(path).write_bytes(b"%PDF-1.4 half")
        raise RuntimeError("matplotlib fell over")
    _fake_build(monkeypatch, truncated)
    cr.build(cfg)
    assert (D / "client_report.json").exists()
    assert not (D / cr.PDF_NAME).exists() and not (D / (cr.PDF_NAME + ".partial")).exists()


def test_a_written_area_pdf_is_moved_into_place(tmp_path, monkeypatch):
    cfg = types.SimpleNamespace(project_dir=tmp_path)
    D = tmp_path / "deliverables"
    D.mkdir()
    (D / cr.PDF_NAME).write_bytes(b"%PDF the last build's")
    seen = []

    def ok(cfg_, rep, ctx, outlines, path, figs=None):
        seen.append(Path(path).name)
        Path(path).write_bytes(b"%PDF-1.4 whole")
        return path
    _fake_build(monkeypatch, ok)
    cr.build(cfg)
    assert seen == [cr.PDF_NAME + ".partial"]
    assert (D / cr.PDF_NAME).read_bytes() == b"%PDF-1.4 whole" and not (D / (cr.PDF_NAME + ".partial")).exists()


# ------------------------------------------------------------------ date reasons said of several dates
@pytest.mark.parametrize("one, many", [
    ("it falls outside the configured date range, so it was not processed",
     "they fall outside the configured date range, so they were not processed"),
    ("it is in the archive catalogue but was not downloaded, so it was left out",
     "they are in the archive catalogue but were not downloaded, so they were left out"),
    ("it is in the archive catalogue but was never assembled and no reason was recorded - unexplained",
     "they are in the archive catalogue but were never assembled and no reason was recorded - unexplained"),
    (qa._DROP_REASON["not_in_plan"],
     "they were found in the archive only after this project's processing had started, so they could not be "
     "added to the processing plan"),
    (qa._NEVER_ENTERED, "they were assembled or coregistered but never entered the stack - unexplained"),
    ("it is not in the processing plan", "they are not in the processing plan"),
    (qa._EXCLUDED_REASON.split(" (")[0], "no interferogram of these dates was kept after the network filter"),
    (qa._DISP_NO_BASELINE, "there are no orbit-stack perpendicular baselines for these DISP-S1 epochs"),
    ("the archive search returned it incomplete", "the archive search returned them incomplete"),
    (qa._PAIRS_DROPPED, "all of their interferograms were dropped before the time series was formed"),
    (qa._NO_WEATHER.split(" (")[0].format("HRRR"),
     "no HRRR weather data was available for the times of these passes, so their weather delays could not be "
     "removed"),
    (qa._CATALOGUE_DEFECT.split(" (")[0],
     "the satellite archive serves these dates in a form that cannot be assembled"),
    (qa._EXCLUDED_FROM_STACK,
     "they could not be aligned with the other dates, so they were kept out of the processing on purpose"),
])
def test_plural_reason_covers_the_date_accounting_reasons(one, many):
    assert cr._plural_reason(one) == many


# Every reason the date accounting can give (30 Oct 2026). A clause opens
# with its subject and has a verb: "1 left out because in the interferogram
# stack but not in the time series" had neither.
_REASONS = [qa._EXCLUDED_REASON, qa._EXCLUDED_REF_REASON, qa._EXCLUDED_FROM_STACK, qa._PAIRS_DROPPED,
            qa._NO_WEATHER.format("HRRR", "NOAA's HRRR archive has no file for the hours around the pass"),
            qa._CATALOGUE_DEFECT.format("burst annotations empty (catalogue returned the date short)"),
            qa._failed_reason("exit 1: login timed out"), qa._NEVER_ENTERED, qa._DISP_SKIPPED,
            qa._DISP_NO_BASELINE, qa._SEARCH_SHORT, *qa._DROP_REASON.values(),
            "it falls outside the configured date range, so it was not processed",
            "it is in the archive catalogue but was not downloaded, so it was left out",
            "it is in the archive catalogue but was never assembled and no reason was recorded - unexplained"]
_SUBJECT = ("it ", "they ", "the ", "no ", "every ", "all of ", "there ", "burst assembly ")
_VERB = re.compile(r"\b(is|are|was|were|has|holds|serves|returned|failed|falls|fall|could|cannot)\b")


@pytest.mark.parametrize("why", _REASONS)
@pytest.mark.parametrize("n", [1, 2])
def test_every_left_out_reason_reads_as_a_clause_after_because(stale, why, n):
    rep = json.loads(json.dumps(stale.rep))
    dates = ["20150101", "20150113"][:n]
    rep["record_dates"] = {"available": 282, "processed": 282 - n,
                           "dropped": {d: why + " (accepted as missing by allow_missing_dates)" for d in dates}}
    text = " ".join(t for _, t in cr.methods_stages(stale.cfg, rep))
    m = re.search(rf"{n} left out because (.+?)\. ", text)
    assert m, text
    clause = m.group(1)
    assert clause.startswith(_SUBJECT) and _VERB.search(clause), clause
    if n > 1:
        assert not re.search(r"\b(it|its|this date|this pass)\b", clause), clause


def test_methods_sentence_names_a_date_without_weather_data(stale):
    """The first city site: "1 left out because in the interferogram stack
    but not in the time series"."""
    rep = json.loads(json.dumps(stale.rep))
    why = qa._NO_WEATHER.format("HRRR", "NOAA's HRRR archive has no file for the hours around the pass")
    rep["record_dates"] = {"available": 301, "processed": 300, "dropped": {"20170129": why}}
    text = " ".join(t for _, t in cr.methods_stages(stale.cfg, rep))
    assert ("1 left out because no HRRR weather data was available for the time of this pass, so its weather "
            "delay could not be removed.") in text


def test_methods_sentence_says_they_for_several_dates(stale):
    rep = json.loads(json.dumps(stale.rep))
    why = "it falls outside the configured date range, so it was not processed"
    rep["record_dates"] = {"available": 282, "processed": 280, "dropped": {"20150101": why, "20150113": why}}
    text = " ".join(t for _, t in cr.methods_stages(stale.cfg, rep))
    assert "2 left out because they fall outside the configured date range, so they were not processed" in text
