"""16 Oct 2026: the quick fixes from the technical readiness review
(Copy of Code_Check_List.xlsx, filled in insar_projects/).

ERR-01  the release gate fails closed: a check that raises adds a line that
        says so, as hard as the check it stands for (it used to vanish).
QUA-03  'hold for specialist' is a hard blocker until the sign-off bound to
        the numbers names the specialist who read those hold reasons.
SEC-05  names reach CSV/KMZ exports on one line; a report-area name with a
        line break or a leading = + - @ is refused.
DAT-07  provenance (which goes to the client in the package) records the
        named preparer, not the OS login and computer name, and writes the
        home folder as '~'.
BLD-04  results_viewer.html ships in a release: in
        test_fix_oct7_dec_D14D18_cfg.test_release_txt_says_the_release_is_internal.

Each test failed on the code in backups/pre_checklist_20261016/ except the
guards that must pass on both (ordinary area names, 'send with notes' soft).
Synthetic projects in tmp_path only; no real project and no network."""
import json
import os
from pathlib import Path

import pytest

import test_fix_L7b as L7b                                     # the gate's synthetic project
from insar_pipeline import areas, provenance, release

HOLD = {"value": "hold for specialist",
        "reasons": ["Too little ordinary farmland around this area to compare it with."]}


@pytest.fixture(autouse=True)
def _identity_set(monkeypatch):
    from insar_pipeline import settings
    monkeypatch.setattr(settings, "identity", lambda: {"firm": "Test Firm", "preparer": "A. Engineer", "contact": ""})


def _final_project(tmp_path, rep=None, sign=True, **sign_kw):
    """A signed, FINAL synthetic project (the QA older than the report)."""
    cfg = L7b._cfg(tmp_path)
    rep = rep or L7b._rep()
    D = L7b._project(tmp_path, rep, L7b._qa())
    q = tmp_path / "quality_assessment.json"
    t = (D / "client_report.json").stat().st_mtime - 600
    os.utime(q, (t, t))
    if sign:
        release.sign_off(tmp_path, "K. Reviewer", "imagery checked", [f["n"] for f in release.features(rep)], rep,
                         **sign_kw)
    return cfg, rep, D


def _line(cfg, code):
    return next((x for x in release.blockers(cfg) if x["code"] == code), None)


# ------------------------------------------------------------------ ERR-01: the gate fails closed
def test_a_reissue_check_that_raises_is_a_hard_blocker(tmp_path, monkeypatch):
    cfg, rep, D = _final_project(tmp_path)
    assert release.is_final(cfg)

    def boom(*a, **k):
        raise RuntimeError("manifest unreadable")
    monkeypatch.setattr(release, "_reissue_problem", boom)
    line = _line(cfg, "reissue_without_note")
    assert line and line["hard"] and "could not run (RuntimeError: manifest unreadable)" in line["text"]
    assert not release.is_final(cfg)


def test_a_gnss_tie_check_that_raises_is_a_hard_stale_input(tmp_path, monkeypatch):
    from insar_pipeline import gnss_tie
    cfg, rep, D = _final_project(tmp_path)
    assert release.is_final(cfg)

    def boom(*a, **k):
        raise ValueError("bad tie")
    monkeypatch.setattr(gnss_tie, "stale_reason", boom)
    hard = [x for x in release.blockers(cfg) if x["hard"]]
    assert [x["code"] for x in hard] == ["stale_inputs"], hard
    assert "GNSS tie is out of date could not run (ValueError: bad tie)" in hard[0]["text"]
    assert hard[0]["text"].endswith("check the release again.")


def test_soft_checks_that_raise_say_so_and_stay_soft(tmp_path, monkeypatch):
    cfg, rep, D = _final_project(tmp_path)
    monkeypatch.setattr(release, "_diagnostic_items", lambda rep, add: 1 / 0)
    line = _line(cfg, "diagnostics_not_read")
    assert line and line["hard"] is False and "ZeroDivisionError" in line["text"]
    # a cross-check record whose date check raises: said, soft, and not read as a result
    (D / "technical").mkdir(exist_ok=True)
    (D / "technical" / "cross_check.json").write_text(json.dumps({"status": "complete", "agree": False}))
    from insar_pipeline import crosscheck
    def no_stamp(*a):
        raise OSError("no stamp")
    monkeypatch.setattr(crosscheck, "record_is_current", no_stamp)
    b = release.blockers(cfg)
    assert any(x["code"] == "cross_check_unread" and not x["hard"] for x in b)
    assert not any(x["code"] == "cross_check_disagrees" for x in b)
    assert release.is_final(cfg)                                 # soft lines do not change FINAL


# ------------------------------------------------------------------ QUA-03: hold for specialist
def test_hold_for_specialist_is_hard_until_a_specialist_releases_it(tmp_path):
    rep = L7b._rep(send_rule=HOLD)
    cfg, rep, D = _final_project(tmp_path, rep=rep, sign=False)
    feats = [f["n"] for f in release.features(rep)]
    line = _line(cfg, "qa_send_rule")
    assert line["hard"] and "Too little ordinary farmland" in line["text"] and "--specialist" in line["text"]
    # the reviewer alone cannot sign a held report off
    with pytest.raises(ValueError, match="holds this report for a specialist"):
        release.sign_off(tmp_path, "K. Reviewer", "", feats, rep)
    assert not (D / release.REVIEW_FILE).exists()
    release.sign_off(tmp_path, "K. Reviewer", "", feats, rep, specialist="K. Wire")
    rec = release.review_record(tmp_path)
    assert rec["specialist"] == {"name": "K. Wire", "send_rule": "hold for specialist",
                                 "reasons": HOLD["reasons"]}
    line = _line(cfg, "qa_send_rule")
    assert line["hard"] is False and "released by the specialist K. Wire" in line["text"]
    assert release.is_final(cfg)


def test_a_specialist_release_for_other_reasons_does_not_count(tmp_path):
    rep = L7b._rep(send_rule=HOLD)
    cfg, rep, D = _final_project(tmp_path, rep=rep, specialist="K. Wire")
    assert release.is_final(cfg)
    f = D / release.REVIEW_FILE
    rec = json.loads(f.read_text())
    rec["specialist"]["reasons"] = ["an earlier, different hold"]
    f.write_text(json.dumps(rec))
    assert _line(cfg, "qa_send_rule")["hard"] and not release.is_final(cfg)
    rec.pop("specialist")
    f.write_text(json.dumps(rec))
    assert _line(cfg, "qa_send_rule")["hard"]


def test_send_with_notes_stays_a_reading_item(tmp_path):
    rep = L7b._rep(send_rule={"value": "send with notes", "reasons": ["rate precision flagged"]})
    cfg, rep, D = _final_project(tmp_path, rep=rep)                 # no specialist needed
    line = _line(cfg, "qa_send_rule")
    assert line["hard"] is False and line["text"] == "The quality assessment says 'send with notes': rate precision flagged."
    assert "specialist" not in release.review_record(tmp_path) and release.is_final(cfg)


def test_cli_passes_the_specialist(tmp_path, monkeypatch):
    rep = L7b._rep(send_rule=HOLD)
    cfg, rep, D = _final_project(tmp_path, rep=rep, sign=False)
    monkeypatch.setattr("insar_pipeline.config.load_config", lambda p: cfg)
    assert release.main(["x.yaml", "--sign", "K. Reviewer", "--features", "1", "--specialist", "K. Wire"]) == 0
    assert release.review_record(tmp_path)["specialist"]["name"] == "K. Wire"


@pytest.fixture
def home(tmp_path, monkeypatch):
    """test_fix_L9a's home fixture: temporary INSAR_PROJECTS / INSAR_SETTINGS,
    registry, no disk probe of the real machine."""
    import test_fix_L9a as L9a
    from insar_pipeline import dashboard as dash
    root = tmp_path / "projects"
    root.mkdir()
    monkeypatch.setenv("INSAR_PROJECTS", str(root))
    monkeypatch.setenv("INSAR_SETTINGS", str(tmp_path / "settings.json"))
    monkeypatch.setattr(L9a.runs, "REGISTRY", tmp_path / "reg.json")
    monkeypatch.setattr(L9a.settings, "disk_budget_gb",
                        lambda path: {"budget": 400.0, "source": "test", "fs_free": 400.0, "c_free": None})
    monkeypatch.setattr(L9a.runs, "_oom_killed", lambda pids, since: False)
    getattr(dash, "_RELEASE_CACHE", {}).clear()
    return root


def test_dashboard_sign_off_asks_for_the_specialist_on_a_held_report(home, monkeypatch):
    import test_fix_L9a as L9a
    from insar_pipeline import dashboard as dash
    root = home
    rep = {"generated": "2026-09-29 15:08", "verdict": "REVIEW", "anchored": True, "dates": ["20170101", "20200101"],
           "span_years": 3.0, "areas": [L9a.FULL, L9a.SITE_A_AOI], "send_rule": HOLD,
           "features": [{"lat": 38.3, "lon": -87.6, "sign": -1, "area_km2": 0.4}]}
    f = L9a._project(root, rep=rep)
    got = []
    monkeypatch.setattr(dash.RunScreen, "_sign", lambda self, data: got.append(data))

    async def body(app, pilot):
        await L9a._settle(app, pilot)
        await pilot.press("r")
        await pilot.pause()
        so = app.screen
        assert isinstance(so, dash.SignOffScreen)
        so.query_one("#reviewer", dash.Input).value = "K. Reviewer"
        so.query_one("#feat1", dash.Checkbox).value = True
        so.query_one("#sign", dash.Button).press()
        await pilot.pause()
        assert "type the specialist's name" in str(so.query_one("#sign-error", dash.Static).render()) and got == []
        so.query_one("#specialist", dash.Input).value = "K. Wire"
        so.query_one("#sign", dash.Button).press()
        await pilot.pause()
        assert got == [{"reviewer": "K. Reviewer", "notes": "", "features": [1], "specialist": "K. Wire"}]

    L9a._pilot(dash.InsarDashboard(str(f)), body)


# ------------------------------------------------------------------ SEC-05: names on one line
def test_one_line():
    assert areas.one_line("North\r\n=cmd|' /C calc'!A0\tx\u2028y") == "North =cmd|' /C calc'!A0 x y"
    assert areas.one_line(None) == "" and areas.one_line("  Area  A1 ") == "Area A1"


@pytest.mark.parametrize("bad", ["Area A1\n=cmd|' /C calc'!A0", "Area\tA1", "=HYPERLINK(\"x\")", "+1 field",
                                 "-North", "@SUM(1)"])
def test_report_area_names_that_spreadsheets_would_run_are_refused(bad):
    assert areas.client_name_problem(bad)


@pytest.mark.parametrize("ok", ["Area A1", "Area A2", "North field 2", "1st St parcel"])
def test_ordinary_report_area_names_still_pass(ok):
    assert areas.client_name_problem(ok) is None


def test_viewer_csv_export_keeps_a_hostile_name_on_its_header_line(monkeypatch):
    """The exploit the review ran on Site C: line 2 of the file began
    '=cmd|' /C calc'!A0'. Through the real writer, with a one-point table."""
    import numpy as np
    from insar_pipeline import field_points as fp
    from insar_pipeline import results_viewer as rv
    hostile = "North\n=cmd|' /C calc'!A0\r@SUM(1+1)"
    one = lambda v: np.array([v])                               # noqa: E731
    fd = {"meta": {"title": "Test\nsite", "dates": ["20170101", "20200101"], "numbers_hash": "abc", "spacing_m": 100},
          "lat": one(38.3), "lon": one(-87.6), "group": np.zeros(1, "u1"), "fast": np.zeros(1, bool),
          "rate": one(-2.0), "los_rel": one(-1.0), "change_los": one(-3.0), "snr": one(1.2), "tc": one(0.8),
          "unwrap": one(0.9)}
    monkeypatch.setattr(rv, "_table", lambda config: (None, fd, None))
    monkeypatch.setattr(rv, "_inside", lambda fd, rings: np.array([0]))
    monkeypatch.setattr(rv, "_poly_stats", lambda fd, idx: {"series_vertical_in": []})
    monkeypatch.setattr(fp, "series_vertical_in_rows", lambda fd, idx: np.full((1, 2), 0.1))
    out = rv.api_export_csv({"config": "x", "rings": [[]], "name": hostile})
    lines = out["text"].splitlines()
    assert lines[0].startswith("# Test site - North =cmd|' /C calc'!A0 @SUM(1+1): 1 measurement points")
    # 8 '#' lines since the build label joined them (16 Oct 2026, readiness review OUT-01)
    # nine '#' lines since 19 Oct 2026 (+ the signal-to-threshold column's)
    assert all(ln.startswith("#") for ln in lines[:9]) and lines[9].startswith("latitude,")
    assert not any(ln.startswith(areas.FORMULA_START) for ln in lines)
    assert out["filename"] == "North_cmd_C_calc_A0_SUM_1_1_points.csv"


# ------------------------------------------------------------------ DAT-07: provenance names the preparer
def test_provenance_names_the_preparer_not_the_login_or_computer(tmp_path, monkeypatch):
    cfg = L7b._phase_linked_project(tmp_path)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path.parent))
    monkeypatch.setattr("sys.argv", [str(tmp_path.parent / "bin" / "insar"), "--config", str(tmp_path / "x.yaml")])
    p = provenance.write(cfg, tmp_path / "deliverables")
    rec = json.loads(p.read_text())
    assert rec["run"]["operator"] == "A. Engineer" and "host" not in rec["run"]
    assert rec["run"]["command"] == ["~/bin/insar", "--config", f"~/{tmp_path.name}/x.yaml"]
    assert rec["project_dir"] == f"~/{tmp_path.name}"
    assert str(tmp_path.parent) + "/" not in p.read_text()
    from insar_pipeline import settings
    monkeypatch.setattr(settings, "identity", lambda: {})
    assert provenance._run_identity(cfg)["operator"].startswith("not set")


def test_portable_leaves_other_paths_and_values_alone(monkeypatch):
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: Path("/home/someone")))
    got = provenance.portable({"a": "/home/someone/x", "b": ["/home/someone", "/home/someoneelse/y", 3, None],
                               "c": Path("/home/someone/z"), "d": "/data/home/someone/q"})
    assert got == {"a": "~/x", "b": ["~", "/home/someoneelse/y", 3, None], "c": "~/z", "d": "/data/home/someone/q"}


# ------------------------------------------------------------------ ERR-05: tests leave the real log alone
def test_notifications_in_tests_go_to_a_temporary_log(tmp_path_factory):
    import os
    from insar_pipeline import notify
    real = Path.home() / ".insar_pipeline_notifications.log"
    assert notify.LOGFILE != real and os.environ["INSAR_NOTIFY"] == "off"
    assert Path(os.environ["INSAR_NOTIFY_LOG"]).parent == tmp_path_factory.getbasetemp()
    # a subprocess (the pipeline run by a test) reads the same place
    import subprocess
    import sys
    out = subprocess.run([sys.executable, "-c", "from insar_pipeline import notify; print(notify.LOGFILE)"],
                         capture_output=True, text=True, cwd=str(Path(notify.__file__).parents[1]), timeout=60)
    assert out.stdout.strip() == os.environ["INSAR_NOTIFY_LOG"], out.stderr


# ------------------------------------------------------------------ the DRAFT page keeps to plain words
def test_the_draft_page_gets_plain_wording_for_the_hold_and_for_a_failed_check(tmp_path, monkeypatch):
    """The QA's hold reasons and an exception's text are for the reviewer;
    the DRAFT report page lists the hard lines, so they carry page_text
    (test_fix_oct7_int_I1: 'gnss_tie.json is older than the velocity product
    it would anchor' reached the page once the hold became hard)."""
    rep = L7b._rep(send_rule={"value": "hold for specialist",
                              "reasons": ["no GNSS anchor: gnss_tie.json is older than the velocity product"]})
    cfg, rep, D = _final_project(tmp_path, rep=rep, sign=False)
    line = _line(cfg, "qa_send_rule")
    assert "gnss_tie.json" in line["text"] and line["page_text"] == release.PAGE_HOLD
    monkeypatch.setattr(release, "_reissue_problem", lambda *a: 1 / 0)
    line = _line(cfg, "reissue_without_note")
    assert "ZeroDivisionError" in line["text"] and line["page_text"] == release.PAGE_CHECK_FAILED
    # the report page showing it: test_fix_oct7_int_I1_client_report::
    # test_a_refused_tie_reaches_the_page_in_plain_words renders the real page


# ------------------------------------------------------------------ RUN-01: the doctor's suite limit
def test_the_doctor_gives_the_suite_room_on_a_slower_pc():
    """The suite took ~510 s on the developer's PC (16 Oct 2026) against the
    doctor's 600 s limit, so an install on a slower PC could end 'not ready'."""
    import inspect
    from insar_pipeline import doctor
    assert inspect.signature(doctor.suite_check).parameters["timeout"].default >= 1800


# ------------------------------------------------------------------ a tool's error text off the client's page
def test_a_dropped_dates_tool_error_stays_off_the_client_page():
    """Site B's DRAFT page read '1 left out because burst assembly failed:
    exit 1: ERROR conda.cli.main_run:execute(148): `conda run python -c ...'
    (the reason as the quality assessment records it). The page keeps the
    plain words; the record keeps the detail."""
    from insar_pipeline import client_report as cr
    raw = ("burst assembly failed: exit 1: ERROR conda.cli.main_run:execute(148): `conda run python -c import "
           "json, sys")
    assert cr._plain_drop_reason(raw) == "burst assembly failed"
    # plain reasons, with or without a colon, are left as they were
    plain = "the satellite archive holds only part of the area's bursts on this date, so it could not be assembled"
    assert cr._plain_drop_reason(plain) == plain
    assert cr._plain_drop_reason("left out: too few pairs") == "left out: too few pairs"
