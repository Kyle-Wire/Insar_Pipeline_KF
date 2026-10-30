"""Follow-ups to the 7 Oct 2026 decisions (decisions.md D1-D18).

D2 (a check that cannot apply is not counted) wherever a count of the
quality checks is printed: summary.json's data_quality_source, the client
report's "Quality checks" line, the CLI status line, the run-summary page and
the provenance record all count with quality_assessment.check_counts - "N of
M applicable checks flagged", naming any check that does not apply - and a
QA file written before D2 (no applicable / not_applicable fields) still reads.

D18 (Site A's mine_layer_sources): the untyped mine layer's source reads
as plain words ("dated mined areas (Indiana Department of Natural
Resources, ...)", not "dated: Indiana ..."), and a date the file was
retrieved or downloaded is never read as the year the outlines are mapped to.

Every test builds what it needs in tmp_path or reads the package's own
configs; none touches a real project.
"""
import json
import os
import time
from pathlib import Path

import pytest
from client_files import need                                    # client data: skips on a fresh clone

from insar_pipeline import client_report, provenance
from insar_pipeline import report_insights as ri
from insar_pipeline.config import PipelineConfig, load_config
from insar_pipeline.quality_assessment import check_counts

PKG = Path(__file__).resolve().parents[1]

def _load_cfg(name):
    """The package config; the DISP one points at files of the site_a_pl
    project (its grid, geometry, baselines), so it is skipped when that
    project is not on this machine (deleted or archived) - 13 Oct 2026."""
    from insar_pipeline.config import ConfigError
    try:
        return load_config(str(need(PKG / "configs" / name)))
    except ConfigError as exc:
        if "_disp" in name and "does not exist" in str(exc):
            pytest.skip(f"{name} needs the site_a_pl project on disk: {str(exc).splitlines()[-1].strip()}")
        raise


def _cfg(tmp_path: Path, **kw) -> PipelineConfig:
    return PipelineConfig(project_dir=tmp_path, bbox=(-87.6, 38.2, -87.3, 38.5), start_date="2016-01-01",
                          end_date="2026-06-30", backend="miaplpy", track=19, orbit_direction="ascending", **kw)


def _qa_new() -> dict:
    """A QA record written after D2: the DS-bias check does not apply (one
    far-field coherence class), one check could not be run."""
    checks = {"reference_stability": {"pass": False}, "rate_precision": {"pass": False},
              "single_epoch_noise": {"pass": True}, "coverage": {"pass": True}, "network": {"pass": None},
              "distributed_scatterer_bias": {"pass": None, "not_applicable": "the far field holds one coherence class"}}
    c = check_counts({"checks": checks})
    return {"checks": checks, "verdict": "USABLE WITH LIMITATIONS", "send_rule": {"value": "send with notes"},
            **{k: c[k] for k in ("passed", "flagged", "untested", "applicable", "not_applicable")},
            "checks_text": c["text"]}


def _qa_old() -> dict:
    """A QA record from before D2: passed / flagged / untested only."""
    checks = {"reference_stability": {"pass": False}, "rate_precision": {"pass": False},
              "single_epoch_noise": {"pass": False}, "coverage": {"pass": False}, "network": {"pass": True},
              "distributed_scatterer_bias": {"pass": True}}
    return {"checks": checks, "verdict": "REVIEW", "passed": 2, "flagged": 4, "untested": 0,
            "send_rule": {"value": "hold for specialist"}}


NEW_TEXT = "2 of 5 applicable checks flagged, 1 could not be run; 1 not applicable (distributed scatterer bias)"
OLD_TEXT = "4 of 6 applicable checks flagged"


def test_check_counts_reads_both_records():
    assert check_counts(_qa_new())["text"] == NEW_TEXT
    assert check_counts(_qa_old())["text"] == OLD_TEXT


# ------------------------------------------------------------------ D2: summary.json
@pytest.mark.parametrize("qa,want", [(_qa_new(), NEW_TEXT), (_qa_old(), OLD_TEXT)])
def test_summary_data_quality_source_counts_applicable_checks(tmp_path, qa, want):
    from insar_pipeline import deliverables
    cfg = _cfg(tmp_path)
    vel = cfg.mintpy_dir / "geo" / "geo_velocity.h5"
    vel.parent.mkdir(parents=True)
    vel.write_bytes(b"")
    os.utime(vel, (time.time() - 60, time.time() - 60))
    (tmp_path / "quality_assessment.json").write_text(json.dumps(qa))
    stats = deliverables._apply_qa_and_gnss(cfg, {})
    assert stats["data_quality"] == qa["verdict"]
    # until 7 Oct 2026 it read "(2 of 6 checks flagged)": the check that cannot apply was in the denominator
    assert stats["data_quality_source"] == f"quality_assessment.json verdict ({want})"


# ------------------------------------------------------------------ D2: the client report
def test_client_quality_checks_line_counts_applicable_checks_and_names_the_rest():
    c = check_counts(_qa_new())
    rep = {"checks_passed": c["passed"], "checks_flagged": c["flagged"], "checks_untested": c["untested"],
           "checks_applicable": c["applicable"], "checks_not_applicable": c["not_applicable"],
           "control_ground_label": "farmland"}
    t = client_report.quality_checks_text(rep)
    assert t.startswith("Checked against field practice: 2 of 5 applicable checks flagged, 1 could not be run, "
                        "and 2 met.")
    assert "1 more check does not apply to this record and is not counted" in t
    assert "ordinary farmland used for comparison here is all of one measurement quality" in t
    assert "Reading the numbers" in t and "_" not in t
    # a report from before D2: met + flagged is the count, as it printed then
    old = client_report.quality_checks_text({"checks_passed": 2, "checks_flagged": 4})
    assert old.startswith("Checked against field practice: 4 of 6 applicable checks flagged, and 2 met.")
    assert "does not apply" not in old


def test_client_report_json_carries_the_counts_from_check_counts():
    src = (PKG / "client_report.py").read_text()
    body = src[src.index("def compute(cfg)"):]
    body = body[:body.index("\ndef ")]
    assert "check_counts(Q)" in body
    for k in ("checks_passed", "checks_flagged", "checks_untested", "checks_applicable", "checks_not_applicable"):
        assert f'"{k}": cnt[' in body, k
    assert 'int(Q.get("passed")' not in body


# ------------------------------------------------------------------ D2: CLI, summary page, provenance
@pytest.mark.parametrize("qa,want", [(_qa_new(), NEW_TEXT), (_qa_old(), OLD_TEXT)])
def test_cli_status_line_counts_applicable_checks(tmp_path, capsys, monkeypatch, qa, want):
    from insar_pipeline import cli, runs
    monkeypatch.setattr(runs, "REGISTRY", tmp_path / "reg.json")
    cfg = _cfg(tmp_path)
    (tmp_path / "quality_assessment.json").write_text(json.dumps(qa))
    cli.act_status(cfg, None)
    out = capsys.readouterr().out
    assert f"verdict: {qa['verdict']}  ({want})" in out
    assert "checks meet practice" not in out


@pytest.mark.parametrize("qa,flagged,applicable,passed,na", [(_qa_new(), 2, 5, 2, "distributed scatterer bias"),
                                                             (_qa_old(), 4, 6, 2, None)])
def test_run_summary_page_counts_applicable_checks(tmp_path, qa, flagged, applicable, passed, na):
    from insar_pipeline.scripts import run_summary_page as rsp
    h = rsp.build({"project": str(tmp_path), "quality": qa})
    line = h[h.index('<span class="word">'):]
    line = line[:line.index("</div>")]
    assert (f"{flagged} of {applicable} applicable quality checks flagged as limitations to carry into any "
            f"interpretation; {passed} meet practice") in line
    if na:
        assert "1 could not be run" in line
        assert f"not counted, because it does not apply here: {na}" in line
        # the DS-bias row says so, not "Not tested"
        row = h[h.index("Distributed-scatterer bias</th>"):]
        assert row[:row.index("</td>")].endswith("Not applicable</span>")
    else:
        assert "not apply" not in line and "could not be run" not in line


def test_provenance_names_the_checks_that_do_not_apply(tmp_path):
    cfg = _cfg(tmp_path)
    (tmp_path / "quality_assessment.json").write_text(json.dumps(_qa_new()))
    q = provenance.stages(cfg)["quality_assessment"]
    assert q["not_applicable"] == ["distributed_scatterer_bias"] and q["checks_text"] == NEW_TEXT
    (tmp_path / "quality_assessment.json").write_text(json.dumps(_qa_old()))
    q = provenance.stages(cfg)["quality_assessment"]
    assert q["not_applicable"] == [] and q["checks_text"] == OLD_TEXT


@pytest.mark.parametrize("qa_after_s", [1.0, 5.0])
def test_provenance_judges_report_older_than_qa_as_the_gate_does(tmp_path, qa_after_s):
    """D7: provenance, the release gate and the dashboard take the report's
    send rule under one test of "the report predates the QA" (the gate's,
    with QA_AFTER_REPORT_GRACE_S). Until the review of 7 Oct, provenance
    compared the times exactly, so a QA written 1 s after the report gave
    provenance the QA's rule while the gate and the dashboard gave the
    report's."""
    from insar_pipeline import release
    from insar_pipeline.dashboard import send_rule_shown
    cfg = _cfg(tmp_path)
    D = tmp_path / "deliverables"
    D.mkdir()
    rep_rule = {"value": "hold for specialist", "reasons": ["no GNSS anchor"], "gnss_considered": True}
    (D / "client_report.json").write_text(json.dumps({"send_rule": rep_rule}))
    (tmp_path / "quality_assessment.json").write_text(json.dumps(_qa_new()))
    t = time.time() - 600
    os.utime(D / "client_report.json", (t, t))
    os.utime(tmp_path / "quality_assessment.json", (t + qa_after_s, t + qa_after_s))
    stale = release.report_predates_qa(tmp_path) is not None
    assert stale == (qa_after_s > release.QA_AFTER_REPORT_GRACE_S)
    q = provenance.stages(cfg)["quality_assessment"]
    assert (q["send_rule_source"] == "client_report.json") == (not stale)
    assert send_rule_shown(tmp_path) == (("send with notes", True) if stale else ("hold for specialist", False))

# ------------------------------------------------------------------ D18: the mine layer's source and vintage
@pytest.mark.parametrize("text", ["retrieved 21 Sep 2026", "Retrieved 20-21 Sep 2026", "retrieved 2026-09-21",
                                  "Retrieved on September 21, 2026", "downloaded 9/21/2026", "accessed: 2026",
                                  "file dated 2026-09-21", "file dated 21 Sep 2026", "obtained 21-Sep-2026",
                                  "retrieved in 2026", "downloaded in September 2026"])
def test_a_date_the_file_was_obtained_is_never_the_vintage(text):
    assert ri.layer_vintage(text) is None
    assert ri.layer_vintage(f"State mine maps, 2016, {text}") == 2016
    assert ri.layer_vintage(f"annual polygons 2000-2025, {text}") == 2025


def test_a_mapped_year_is_still_read():
    assert ri.layer_vintage("Indiana Geological & Water Survey, Coal_Mines_Underground_Dec2016") == 2016
    assert ri.layer_vintage("retrieved from the 2016 survey") == 2016
    assert ri.layer_vintage("source and date not recorded") is None


def test_typed_layers_and_older_reports_keep_their_label():
    info = [{"key": "underground", "label": "underground workings",
             "source": "Indiana Geological & Water Survey, Dec 2016", "vintage": 2016},
            {"key": "surface", "label": "surface mines", "source": None, "vintage": None}]
    line = client_report.mine_outlines_line({"mine_layer_info": info})
    # escaped once: the page puts this line into HTML as it is
    assert ("underground workings - Indiana Geological &amp; Water Survey, Dec 2016; "
            "surface mines - source and date not recorded.") in line and "&amp;amp;" not in line
    # a report built before 7 Oct 2026 joined the internal keys: its label stays in front
    old = [{"key": "mapped", "label": "mine areas", "source": "dated: Indiana DNR, 2000-2025", "vintage": 2025}]
    assert "mine areas - dated: Indiana DNR" in client_report.mine_outlines_line({"mine_layer_info": old})
    assert client_report.mine_outlines_line({}).startswith("No mine outlines were supplied")


def test_two_recorded_sources_each_keep_their_plain_label():
    cfg = _cfg(Path("/nonexistent"), mine_layer_sources=(
        "Underground mine workings: Indiana Geological & Water Survey, mines worked to 2015 (Dec 2016 layer), "
        "retrieved 20 Sep 2026",
        "Dated mined areas: Indiana DNR CMIS, annual polygons 2000-2025, retrieved 21 Sep 2026"))
    src = ri.layer_sources(cfg)
    assert src["mapped"] == ("underground mine workings (Indiana Geological & Water Survey, mines worked to 2015 "
                             "(Dec 2016 layer), retrieved 20 Sep 2026); dated mined areas (Indiana DNR CMIS, "
                             "annual polygons 2000-2025, retrieved 21 Sep 2026)")
    assert ri.layer_vintage(src["underground"]) == 2016 and ri.layer_vintage(src["mapped"]) == 2025


