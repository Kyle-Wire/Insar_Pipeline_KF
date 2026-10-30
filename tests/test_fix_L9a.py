"""The dashboard (lane L9a, review 5 Oct 2026): area figures as the client
report reads them, every processing start through the preflight with its
kind, the run view's timing / earlier-run labels / plain cause, sign-off and
package, the guide, the new-area writer, the files screen's big-file guard,
and Textual pilot tests on a fake finished project under a temporary
INSAR_PROJECTS. No network, no real runs."""
import asyncio
import json
import os
import time
from pathlib import Path

import pytest

from insar_pipeline import areas, dashboard as dash, files_screen as fs, release, runs, settings

SITE_A_AOI = {"name": "Area A2", "spec": "38.35114,-87.66331,38.35349,-87.65863", "kind": "area",
                  "cells": 486, "area_km2": 0.11, "los_up": 0.784, "relative_vertical_mm_yr": -4.58,
                  "vertical_mm_yr": -5.67, "vertical_uncertainty_1sigma_mm_yr": 7.53, "total_mm": -59.9,
                  "anchored_to_gnss": True, "detection_limit_mm_yr": 15.0, "detected": False, "z": -0.61,
                  "p_two_sided": 0.523, "signal_to_noise": 0.6}
FULL = {"name": "Full Satellite Imaging", "spec": "full", "kind": "full", "cells": 3157158, "area_km2": 712.6,
        "vertical_mm_yr": -1.09, "vertical_uncertainty_1sigma_mm_yr": 0.73, "total_mm": -11.5,
        "anchored_to_gnss": True}
DEAD_PID = 4_000_000            # above pid_max: never a live process


# ---------------------------------------------------------------- fixtures
def _project(root: Path, name="demo", backend="miaplpy", rep=None, run_result=True, qa=True) -> Path:
    """A finished project: config, client report, QA, run record, progress."""
    P = root / name
    (P / "deliverables").mkdir(parents=True)
    (P / "logs").mkdir()
    (P / "mintpy").mkdir()
    (P / "mintpy" / "geo").mkdir()
    vel = P / "mintpy" / "velocity.h5"
    vel.write_bytes(b"")
    (P / "mintpy" / "geo" / "geo_velocity.h5").write_bytes(b"")
    (P / "mintpy" / "run_products.json").write_text("{}")
    # MintPy finished before the report: the velocity, its geocoded copy and
    # the run record (run_products.json) share the velocity's time
    t_vel = time.time() - 3600
    for f in (vel, P / "mintpy" / "geo" / "geo_velocity.h5", P / "mintpy" / "run_products.json"):
        os.utime(f, (t_vel, t_vel))
    cfgs = root / "configs"
    cfgs.mkdir(exist_ok=True)
    f = cfgs / f"{name}.yaml"
    f.write_text(f"project_dir: {P}\nbackend: {backend}\nbbox: [-87.7, 38.3, -87.6, 38.4]\n"
                 "start_date: '2017-01-01'\nend_date: '2020-01-01'\ntrack: 165\n"
                 "stack_virtual_merge: 'False'\nmintpy_deramp: 'no'\nmintpy_tropo_method: 'no'\n"
                 "report_areas: ['Full Satellite Imaging:full', "
                 "'Area A2:38.35114,-87.66331,38.35349,-87.65863']\n")
    rep = rep or {"generated": "2026-09-29 15:08", "verdict": "REVIEW", "anchored": True,
                  "dates": ["20170101", "20200101"], "span_years": 3.0, "areas": [FULL, SITE_A_AOI],
                  "features": [{"lat": 38.3, "lon": -87.6, "sign": -1, "area_km2": 0.4}]}
    (P / "deliverables" / "client_report.json").write_text(json.dumps(rep))
    (P / "deliverables" / "client_report.html").write_text("<html>report</html>")
    if qa:
        (P / "quality_assessment.json").write_text(json.dumps(
            {"checks": {}, "send_rule": {"value": "send with notes", "reasons": ["thin null"]}}))
    t0 = time.time() - 7200
    log = P / "logs" / "full_x.log"
    log.write_text("line one\nline two\n")
    (P / "run.json").write_text(json.dumps({"pid": DEAD_PID, "start_ticks": 1, "kind": "full",
                                            "config": str(f), "log": str(log), "started": t0}))
    ev = [{"t": t0 + 1, "stage": "pipeline", "status": "start"},
          {"t": t0 + 10, "stage": "client_report", "status": "start"},
          {"t": t0 + 600, "stage": "client_report", "status": "done"},
          {"t": t0 + 601, "stage": "pipeline", "status": "done"}]
    (P / "progress.jsonl").write_text("\n".join(json.dumps(e) for e in ev) + "\n")
    if run_result:
        (P / "run_result.json").write_text(json.dumps(
            {"status": "done", "kind": "full", "stage": "done", "plain_cause": "Finished in 10 min.",
             "next_action": f"Results are in {P / 'deliverables'}.", "finished": t0 + 602, "pid": DEAD_PID}))
    return f


@pytest.fixture
def home(tmp_path, monkeypatch):
    """A temporary INSAR_PROJECTS / INSAR_SETTINGS, the run registry in it,
    and no disk-budget probe of the real machine (reg.exe, the vhdx)."""
    root = tmp_path / "projects"
    root.mkdir()
    monkeypatch.setenv("INSAR_PROJECTS", str(root))
    monkeypatch.setenv("INSAR_SETTINGS", str(tmp_path / "settings.json"))
    monkeypatch.setattr(runs, "REGISTRY", tmp_path / "reg.json")
    monkeypatch.setattr(settings, "disk_budget_gb",
                        lambda path: {"budget": 400.0, "source": "test", "fs_free": 400.0, "c_free": None})
    monkeypatch.setattr(runs, "_oom_killed", lambda pids, since: False)
    getattr(dash, "_RELEASE_CACHE", {}).clear()
    return root


def _pilot(app, body, size=(180, 55)):
    async def go():
        async with app.run_test(size=size) as pilot:
            await pilot.pause()
            await body(app, pilot)
    asyncio.run(go())


async def _settle(app, pilot, n=3):
    for _ in range(n):
        await app.workers.wait_for_complete()
        await pilot.pause()


# ---------------------------------------------------------------- usability-7
def test_area_figures_carry_the_reports_range_and_reading_not_a_bare_sigma():
    """Site A's Area A2: the dashboard showed '-0.223 in/yr ±0.296'
    while the report says 'Within normal ground variation', -0.80 to +0.36."""
    v = dash.area_view(SITE_A_AOI, True)
    assert v["rate"] == "-0.223 in/yr"
    assert v["range"] == "(-0.80 to +0.36 in/yr)"
    assert v["reading"] == "Within normal ground variation"
    # the legacy range is the report's own arithmetic (client_report Z95 x 1 sigma), not a dashboard one
    from insar_pipeline.client_report import Z95
    lo, hi = v["range_mm"]
    assert lo == pytest.approx(-5.67 - Z95 * 7.53) and hi == pytest.approx(-5.67 + Z95 * 7.53)
    line = dash.area_line(SITE_A_AOI, True).plain
    assert "±" not in line and "within normal ground variation" in line
    assert dash.area_view({**SITE_A_AOI, "detected": True}, True)["reading"] == "Movement detected"
    # a report that stores its range and reading (5 Oct contract) is shown verbatim, never recomputed
    new = {**SITE_A_AOI, "range95_mm_yr": [-30.0, -10.0], "reading": "Movement detected"}
    v2 = dash.area_view(new, True)
    assert v2["range"] == f"({-30 / 25.4:+.2f} to {-10 / 25.4:+.2f} in/yr)" and v2["reading"] == "Movement detected"
    # not testable: said so, no range
    nt = dash.area_view({**SITE_A_AOI, "testable": False, "not_testable_reason": "too few placements"}, True)
    assert nt["range"] == "" and "Not testable" in nt["reading"] and "too few placements" in nt["reading"]


def test_projects_row_reads_like_the_report_and_says_when_not_anchored(home):
    f = _project(home, rep={"verdict": "REVIEW", "anchored": False, "areas": [
        {**FULL, "anchored_to_gnss": False}, {**SITE_A_AOI, "anchored_to_gnss": False}]})
    p = runs.project_summary(f)
    cells = dash.project_row(p)
    txt = cells["results"].plain
    assert "Area A2: -0.223 in/yr (-0.80 to +0.36 in/yr) · within normal ground variation" in txt
    assert "(not GNSS-anchored)" in txt and "±" not in txt
    assert "relative to its own farmland" in txt          # the full area, unanchored
    # verdict: the QA send rule, with the release gate's DRAFT
    assert cells["verdict"].plain.startswith("send with notes") and "DRAFT" in cells["verdict"].plain


# ---------------------------------------------------------------- usability-1
def test_s_then_whole_pipeline_opens_the_preflight_with_its_kind(home, monkeypatch):
    f = _project(home)
    started = []
    monkeypatch.setattr(runs, "start", lambda cfg, P, kind="full", extra=None: started.append(kind) or
                        {"pid": 1, "kind": kind})
    verdict = {"res": [("fail", "ERA5: licence not accepted")]}
    monkeypatch.setattr(dash, "preflight_checks", lambda cfg, config, kind="full", say=None: list(verdict["res"]),
                        raising=False)

    async def body(app, pilot):
        await _settle(app, pilot)
        await pilot.press("s")
        await pilot.pause()
        ch = app.screen
        assert isinstance(ch, dash.ChooseRun)
        ol = ch.query_one(dash.OptionList)
        # the recommendation first (a finished project: Products only), Whole pipeline under Advanced
        assert ol.get_option_at_index(0).id == "products" and "Recommended" in str(ol.get_option_at_index(0).prompt)
        ol.highlighted = ol.get_option_index("full")
        await pilot.press("enter")
        await _settle(app, pilot)
        pf = app.screen
        assert isinstance(pf, dash.PreflightScreen) and pf.kind == "full"
        assert pf.query_one("#go", dash.Button).disabled is True
        assert started == []
        await pilot.press("escape")
        await pilot.pause()
        # phase linking onward keeps its kind through the checks, and Start returns to the same run view
        verdict["res"] = [("ok", "all good")]
        await pilot.press("s")
        await pilot.pause()
        ol = app.screen.query_one(dash.OptionList)
        ol.highlighted = ol.get_option_index("miaplpy")
        await pilot.press("enter")
        await _settle(app, pilot)
        pf = app.screen
        assert isinstance(pf, dash.PreflightScreen) and pf.kind == "miaplpy"
        assert pf.query_one("#go", dash.Button).disabled is False
        pf.query_one("#go", dash.Button).press()
        await _settle(app, pilot)
        assert started == ["miaplpy"]
        assert isinstance(app.screen, dash.RunScreen)
        assert sum(isinstance(s, dash.RunScreen) for s in app.screen_stack) == 1
        # p and e exist on the run view
        keys = {b[0] if isinstance(b, tuple) else b.key for b in dash.RunScreen.BINDINGS}
        assert {"p", "e", "r", "k"} <= keys

    _pilot(dash.InsarDashboard(str(f)), body)


def test_whole_pipeline_after_cleaned_dates_asks_first(home, monkeypatch):
    f = _project(home)
    from insar_pipeline.config import load_config
    cs = Path(load_config(str(f)).stack_dir) / "coreg_secondarys"
    cs.mkdir(parents=True)
    for d in ("20170105", "20170117"):
        (cs / f"{d}.cleaned").write_text("{}")
    monkeypatch.setattr(dash, "preflight_checks", lambda *a, **k: [("ok", "fine")], raising=False)

    async def body(app, pilot):
        await _settle(app, pilot)
        await pilot.press("s")
        await pilot.pause()
        assert "2 dates" in str(app.screen.query_one("#cleaned-note", dash.Static).render())
        ol = app.screen.query_one(dash.OptionList)
        ol.highlighted = ol.get_option_index("full")
        await pilot.press("enter")
        await pilot.pause()
        assert isinstance(app.screen, dash.Confirm)
        assert "2 dates" in app.screen.question

    _pilot(dash.InsarDashboard(str(f)), body)


def test_preflight_measures_the_budget_planner_peak_ram_and_blocks_only_on_failures(home, monkeypatch):
    from insar_pipeline import data_acquisition as da, miaplpy_backend as mb, pipeline, stack_backend as sb
    from insar_pipeline.config import load_config
    f = _project(home)
    cfg = load_config(str(f))
    cfg.mintpy_tropo_method = "pyaps"
    cfg.miaplpy_ministack_size, cfg.miaplpy_time_lag = 12, 4
    budgets = []
    monkeypatch.setattr(settings, "disk_budget_gb", lambda p: budgets.append(p) or
                        {"budget": 100.0, "source": "WSL: vhdx headroom + C: free"})
    plan = sb.StackPlan(est_peak_gb=300.0, est_ram_gb=21.0, mem_gb=23.0)
    monkeypatch.setattr(sb, "plan_stack", lambda c: plan)
    monkeypatch.setattr(pipeline, "project_gb", lambda P: 0.0)
    monkeypatch.setattr(da, "check_earthdata_login", lambda: (False, "ASF rejected the login"))
    monkeypatch.setattr(sb, "check_weather_credentials", lambda c: [sb.WeatherProblem("warn", "CDS unreachable")])
    monkeypatch.setattr(sb, "era5_lag_note", lambda c, today=None: "ERA5: the record ends inside the lag")
    monkeypatch.setattr(sb, "era5_prefetch_plan", lambda c: {"ok": True, "present": 5, "needed": 80, "dir": "/x"})
    seen = {}
    monkeypatch.setattr(mb, "verify_patched_miaplpy",
                        lambda env, timeout=300, ministack=None, time_lag=None:
                        seen.update(ministack=ministack, time_lag=time_lag) or {"ratio": 1.0})
    said = []
    res = dash.preflight_checks(cfg, str(f), "full", say=said.append)
    fails = [t for lv, t in res if lv == "fail"]
    warns = [t for lv, t in res if lv == "warn"]
    assert any("projected peak 300 GB" in t and "budget 100 GB" in t for t in fails)       # planner vs real budget
    assert any("memory" in t and "21 GB" in t for t in fails)                              # 21 of 23 GB
    assert any("ASF rejected" in t for t in fails)                                          # verified login
    assert any("CDS unreachable" in t for t in warns) and not any("CDS" in t for t in fails)
    assert any("lag" in t for t in warns)
    assert any("5 of 80 weather files" in t for lv, t in res)
    assert seen == {"ministack": 12, "time_lag": 4}
    assert budgets and any("topsStack run plan" in s for s in said)
    # everything in order: nothing blocks
    plan.est_peak_gb, plan.est_ram_gb = 40.0, 4.0
    monkeypatch.setattr(da, "check_earthdata_login", lambda: (True, "accepted by Earthdata and ASF"))
    res = dash.preflight_checks(cfg, str(f), "full")
    assert not [t for lv, t in res if lv == "fail"], res
    # a MintPy restart downloads nothing: no login check, no planner; the watchdog limits apply
    monkeypatch.setattr(sb, "plan_stack", lambda c: pytest.fail("no planner on a restart"))
    monkeypatch.setattr(settings, "disk_budget_gb", lambda p: {"budget": 10.0, "source": "t"})
    res = dash.preflight_checks(cfg, str(f), "mintpy")
    assert not any("Earthdata" in t for _, t in res)
    assert any(lv == "fail" and "stops itself below 20 GB" in t for lv, t in res)


# ---------------------------------------------------------------- usability-9
def test_run_view_times_labels_and_explains_from_the_records(home):
    t0 = 1_000_000.0
    rec = {"pid": 5, "started": t0, "kind": "miaplpy"}
    L = {"pipeline": {"t": t0 - 5000, "stage": "pipeline", "status": "failed", "msg": "IndexError('x')",
                      "started": t0 - 6000},
         "run_07_pairs_misreg": {"t": t0 + 100, "stage": "run_07_pairs_misreg", "status": "failed", "done": 804,
                                 "total": 804, "msg": "2 failed", "started": t0 + 10},
         "run_08_timeseries_misreg": {"t": t0 + 200, "stage": "run_08_timeseries_misreg", "status": "done",
                                      "started": t0 + 150},
         "run_16_unwrap": {"t": t0 + 400, "stage": "run_16_unwrap", "status": "running", "done": 610, "total": 804,
                           "skipped": 600, "started": t0 + 300, "work_started": t0 + 302}}
    now = t0 + 146 * 3600                       # a week later
    # finished (not running): elapsed ends at the run's last event, not now
    assert dash.run_end(rec, "done", None, L, now) == t0 + 400
    res = {"finished": t0 + 500, "pid": 5, "status": "done"}
    assert dash.run_end(rec, "done", res, L, now) == t0 + 500
    assert dash.run_end({**rec, "stopped": t0 + 50}, "stopped", res, L, now) == t0 + 50
    assert dash.run_end(rec, "running", res, L, now) == now
    # an older run's result is not this run's
    assert dash.result_for_run(rec, {"finished": t0 - 10, "pid": 3}) is None
    rows = {r["key"]: r for r in dash.stage_rows(L, rec, "done", now)}
    assert "earlier run" in rows["pipeline"]["note"]
    assert rows["run_07_pairs_misreg"]["icon"] == "!" and "tolerated" in rows["run_07_pairs_misreg"]["note"]
    assert "600 verified from the last run" in rows["run_16_unwrap"]["note"]
    assert rows["run_16_unwrap"]["remaining"] == ""                 # no ETA once not running
    assert rows["run_16_unwrap"]["elapsed"] == dash.fmt_dur(100)   # frozen at its last event
    live = {r["key"]: r for r in dash.stage_rows(L, rec, "running", t0 + 410)}
    assert live["run_16_unwrap"]["remaining"] != "" and live["run_16_unwrap"]["elapsed"] == dash.fmt_dur(110)


def test_run_view_shows_plain_cause_next_action_and_an_interruption(home, monkeypatch):
    f = _project(home, run_result=False)
    P = home / "demo"
    rec = json.loads((P / "run.json").read_text())
    (P / "run_result.json").write_text(json.dumps(
        {"status": "failed", "stage": "miaplpy", "finished": rec["started"] + 50, "pid": DEAD_PID,
         "plain_cause": "MiaplPy is not the patched build, so phase linking cannot start.",
         "next_action": "Run patches/miaplpy/build_miaplpy.sh, then 'Phase linking onward'.",
         "log": str(P / "logs" / "full_x.log")}))

    async def body(app, pilot):
        await _settle(app, pilot)
        title = str(app.screen.query_one("#runtitle", dash.Static).render())
        assert "MiaplPy is not the patched build" in title
        assert "What now:" in title and "Phase linking onward" in title
        assert "took" in title and "146h" not in title
        # the run vanished without its final event: interrupted, with where and the restart
        (P / "run_result.json").unlink()
        (P / "progress.jsonl").write_text("\n".join(json.dumps(e) for e in (
            {"t": rec["started"] + 1, "stage": "pipeline", "status": "start"},
            {"t": rec["started"] + 60, "stage": "download", "status": "running", "done": 3, "total": 90})) + "\n")
        app.screen.refresh_view()
        await pilot.pause()
        title = str(app.screen.query_one("#runtitle", dash.Static).render())
        assert "interrupted" in title and "Stopped without reporting during download" in title

    _pilot(dash.InsarDashboard(str(f)), body)


# ---------------------------------------------------------------- release
def test_sign_off_rebuilds_the_report_and_packages_only_when_final(home, monkeypatch):
    from insar_pipeline import package
    from insar_pipeline.config import load_config
    f = _project(home)
    cfg = load_config(str(f))
    rebuilt = []

    def fake_rebuild(config, P):
        rebuilt.append(str(config))
        return True, str(P / "logs" / "report.log")
    # a partial acknowledgement is refused, nothing recorded
    with pytest.raises(ValueError, match="every feature must be checked"):
        dash.sign_off_and_release(cfg, str(f), {"reviewer": "K. Wire", "features": []}, rebuild=fake_rebuild)
    assert rebuilt == [] and not (home / "demo" / "deliverables" / "review.json").exists()
    # signed, rebuilt, but other hard blockers remain: DRAFT, no package
    monkeypatch.setattr(package, "build", lambda c, force_draft=False: pytest.fail("not final: no package"))
    out = dash.sign_off_and_release(cfg, str(f), {"reviewer": "K. Wire", "features": [1]}, rebuild=fake_rebuild)
    assert rebuilt == [str(f)] and out["rebuilt"] is True
    assert release.review_record(cfg.project_dir)["reviewer"] == "K. Wire"
    assert out["final"] is False and out["package"] is None and "still a DRAFT" in out["message"]
    # final: the package is built and checked by its MANIFEST
    monkeypatch.setattr(release, "status", lambda c, rep=None, qa=None: {"final": True, "blockers": []})
    rel = home / "demo" / "deliverables" / "releases" / "20261005_1200"

    def build(c, force_draft=False):
        rel.mkdir(parents=True, exist_ok=True)
        (rel / "MANIFEST.json").write_text("{}")
        return rel
    monkeypatch.setattr(package, "build", build)
    out = dash.sign_off_and_release(cfg, str(f), {"reviewer": "K. Wire", "features": [1]}, rebuild=fake_rebuild)
    assert out["package"] == str(rel)
    # a build that leaves no manifest is not reported as a package
    (rel / "MANIFEST.json").unlink()
    monkeypatch.setattr(package, "build", lambda c, force_draft=False: rel)
    path, msg = dash.build_package(cfg)
    assert path is None and "MANIFEST" in msg
    # a failed rebuild says so and goes no further
    out = dash.sign_off_and_release(cfg, str(f), {"reviewer": "K. Wire", "features": [1]},
                                    rebuild=lambda c, P: (False, "/x/report.log"))
    assert out["rebuilt"] is False and "/x/report.log" in out["message"]


def test_rebuild_report_holds_the_project_lock(home, monkeypatch):
    f = _project(home)
    P = home / "demo"
    seen = {}

    def run(argv, **kw):
        seen["holder"] = runs.lock_holder(P)
        seen["argv"] = argv

        class R:
            returncode = 0
        return R()
    monkeypatch.setattr(dash.subprocess, "run", run)
    ok, log = dash.rebuild_report(str(f), P)
    assert ok and seen["argv"][-1] == str(f) and "run_client_report" in seen["argv"][2]
    assert seen["holder"] is not None and seen["holder"].get("kind") == "report"
    assert runs.lock_holder(P) is None


def test_open_report_asks_first_when_it_is_a_draft(home, monkeypatch):
    f = _project(home)
    opened = []
    monkeypatch.setattr(dash, "open_path", lambda p: opened.append(p) or str(p))

    async def body(app, pilot):
        await _settle(app, pilot)
        await pilot.press("o")
        await pilot.pause()
        assert isinstance(app.screen, dash.Confirm) and opened == []
        assert "DRAFT" in app.screen.question and "Not yet reviewed and signed off" in app.screen.question
        app.screen.query_one("#yes", dash.Button).press()
        await pilot.pause()
        assert opened and opened[0].name == "client_report.html"

    _pilot(dash.InsarDashboard(str(f)), body)


def test_sign_off_screen_needs_a_name_and_every_feature(home, monkeypatch):
    f = _project(home)
    got = []
    monkeypatch.setattr(dash.RunScreen, "_sign", lambda self, data: got.append(data))

    async def body(app, pilot):
        await _settle(app, pilot)
        await pilot.press("r")
        await pilot.pause()
        so = app.screen
        assert isinstance(so, dash.SignOffScreen)
        so.query_one("#reviewer", dash.Input).value = "K. Wire"
        so.query_one("#sign", dash.Button).press()
        await pilot.pause()
        assert "0 of 1" in str(so.query_one("#sign-error", dash.Static).render()) and got == []
        so.query_one("#feat1", dash.Checkbox).value = True
        so.query_one("#sign", dash.Button).press()
        await pilot.pause()
        assert got == [{"reviewer": "K. Wire", "notes": "", "features": [1]}]

    _pilot(dash.InsarDashboard(str(f)), body)


# ---------------------------------------------------------------- guide, subcommands
def test_question_mark_opens_the_guide_or_says_it_is_missing(home, monkeypatch, tmp_path):
    monkeypatch.setattr(runs, "known_configs", lambda: [])
    monkeypatch.setattr(dash, "guide_path", lambda: None)

    async def body(app, pilot):
        await pilot.press("question_mark")
        await pilot.pause()
        assert isinstance(app.screen, dash.HelpScreen)
        assert "User guide not found" in str(app.screen.query_one("#guide-missing", dash.Static).render())
        await pilot.press("escape")
        await pilot.pause()
        g = tmp_path / "USER_GUIDE.md"
        g.write_text("# Using the dashboard\n\nPress s to start.\n")
        monkeypatch.setattr(dash, "guide_path", lambda: g)
        await pilot.press("question_mark")
        await pilot.pause()
        assert app.screen.query("#guide") and not app.screen.query("#guide-missing")

    _pilot(dash.InsarDashboard(), body)


def test_insar_subcommands_route_the_optional_checks_and_version(monkeypatch, capsys):
    from insar_pipeline import cli, version
    calls = []
    monkeypatch.setattr(dash.InsarDashboard, "run", lambda self: pytest.fail("the subcommand opened the dashboard"))
    monkeypatch.setattr(cli, "tool_main", lambda name, argv: calls.append((name, argv)) or 3)
    monkeypatch.setattr(dash.sys, "argv", ["insar", "cross-check", "x.yaml", "--force"])
    with pytest.raises(SystemExit) as e:
        dash.main()
    assert e.value.code == 3 and calls == [("cross-check", ["x.yaml", "--force"])]
    monkeypatch.setattr(dash.sys, "argv", ["insar", "injection-test", "x.yaml"])
    with pytest.raises(SystemExit):
        dash.main()
    assert calls[-1] == ("injection-test", ["x.yaml"])
    monkeypatch.setattr(version, "label", lambda: "insar_pipeline 9.9 (code abc)")
    monkeypatch.setattr(dash.sys, "argv", ["insar", "--version"])
    dash.main()
    assert "insar_pipeline 9.9 (code abc)" in capsys.readouterr().out
    assert "--force" in dash.USAGE and "injection-test" in dash.USAGE


# ---------------------------------------------------------------- new area
def test_archive_text_names_the_stations_the_tie_will_use_and_the_tracks_frame():
    from insar_pipeline.config import GNSS_TIE_RADIUS_KM
    g = {"radius_km": GNSS_TIE_RADIUS_KM, "warning": "Only 1 station(s) can anchor the report",
         "stations": [{"site": "INEV", "distance_km": 12.0, "will_use": True},
                      {"site": "ILMT", "distance_km": 35.0, "will_use": False}]}
    t = dash.gnss_text(g).plain
    assert "used for the anchor (within 40 km" in t and "INEV 12 km" in t
    assert "nearby, not used: ILMT 35 km" in t and "Only 1 station" in t
    assert "50 km" not in t and "anchored to them" not in t
    detail = [{"frame": "32248", "track": 121, "direction": "ascending", "granules": 300, "overlap_fraction": 1.0},
              {"frame": "04862", "track": 19, "direction": "ascending", "granules": 134, "overlap_fraction": 0.8}]
    row = areas.frame_for_track(detail, 19, "ASCENDING")
    ft = dash.frame_text(row, 19, "ASCENDING").plain
    assert "F04862" in ft and "only 80%" in ft
    assert "cannot run on this track" in dash.frame_text(None, 7, "DESCENDING").plain


def test_new_area_write_confirms_overwrite_and_looks_up_swaths_for_the_chosen_track(home, monkeypatch):
    calls = []
    out_cfg = home / "configs" / "site_d_disp_screening.yaml"

    def write_project(name, area, route, start, end, **kw):
        calls.append({"name": name, "route": route, **kw})
        out_cfg.parent.mkdir(parents=True, exist_ok=True)
        out_cfg.write_text("x: 1\n")
        return out_cfg
    monkeypatch.setattr(areas, "write_project", write_project)
    monkeypatch.setattr(areas, "existing_project", lambda name, route: {
        "config": str(out_cfg), "project_dir": str(home / "site_d_disp_screening"), "backend": "disp",
        "holds": ["a DISP-S1 time series"]})
    monkeypatch.setattr(dash, "preflight_checks", lambda *a, **k: [("ok", "fine")], raising=False)
    area = areas.parse_coordinates("38.24, -87.65", radius_km=2)
    archive = {"tracks": [{"direction": "ASCENDING", "track": 121, "dates": 200, "by_year": {}},
                          {"direction": "ASCENDING", "track": 19, "dates": 180, "by_year": {}}],
               "disp_frames": {"32248": 300, "04862": 134},
               "disp_frame_detail": [{"frame": "32248", "track": 121, "direction": "ascending", "granules": 300,
                                      "overlap_fraction": 1.0},
                                     {"frame": "04862", "track": 19, "direction": "ascending", "granules": 134,
                                      "overlap_fraction": 1.0}]}

    async def body(app, pilot):
        scr = dash.NewAreaScreen()
        await app.push_screen(scr)
        await pilot.pause()
        scr._set_area(area)
        scr._show_archive(archive)
        scr.query_one("#name", dash.Input).value = "Site D"
        scr.query_one("#route", dash.Select).value = "disp"
        scr.query_one("#track", dash.Select).value = "ASCENDING:19"
        await pilot.pause()
        assert "site_d_disp_screening" in str(scr.query_one("#trackinfo", dash.Static).render())
        scr.write()
        await pilot.pause()
        assert isinstance(app.screen, dash.Confirm) and "already exists" in app.screen.question
        app.screen.query_one("#yes", dash.Button).press()
        await _settle(app, pilot)
        assert len(calls) == 1
        c = calls[0]
        assert c["overwrite"] is True and c["swaths"] is None and c["track"] == 19
        assert c["disp_frame"] == "04862"                 # the chosen track's frame, not the most granules
        assert c["report_areas"][1][0] == "Site D" and c["report_areas"][1][1] is area
        assert isinstance(app.screen, dash.PreflightScreen)

    monkeypatch.setattr(runs, "known_configs", lambda: [])
    _pilot(dash.InsarDashboard(), body)


# ---------------------------------------------------------------- tests-15: the app on a finished project
def test_projects_open_run_view_and_results_on_a_finished_project(home, monkeypatch):
    f = _project(home)
    monkeypatch.setattr(runs, "known_configs", lambda: [f])

    async def body(app, pilot):
        await _settle(app, pilot)
        t = app.screen.query_one(dash.DataTable)
        assert t.row_count == 1
        row = t.get_row_at(0)
        assert str(row[2]) == "done"
        assert "within normal ground variation" in str(row[4]) and "±" not in str(row[4])
        assert str(row[3]).startswith("send with notes")
        await pilot.press("enter")
        await _settle(app, pilot)
        rs = app.screen
        assert isinstance(rs, dash.RunScreen)
        assert " done" in str(rs.query_one("#runtitle", dash.Static).render())
        res = str(rs.query_one("#results", dash.Static).render())
        assert "likely range -0.80 to +0.36 in/yr" in res and "DRAFT" in res
        await pilot.press("v")
        await _settle(app, pilot)
        from insar_pipeline.results_screen import ResultsScreen
        assert isinstance(app.screen, ResultsScreen)
        await pilot.press("escape")
        await pilot.press("escape")
        await pilot.pause()
        assert isinstance(app.screen, dash.ProjectsScreen)

    _pilot(dash.InsarDashboard(), body)


# ---------------------------------------------------------------- files screen (usability-16, disp-1)
def test_big_files_are_shown_in_their_folder_not_opened(home, monkeypatch, tmp_path):
    big = tmp_path / "measurement_points_egms.csv"
    with open(big, "wb") as fh:
        fh.truncate(int(fs.BIG_FILE_BYTES) + 1)          # sparse: no real disk used
    small = tmp_path / "small.csv"
    small.write_text("a,b\n1,2\n")
    why = fs.open_advice(big)
    assert why and "1,048,576 rows" in why and "Results (v)" in why
    assert fs.open_advice(small) is None and fs.open_advice(tmp_path) is None
    opened, shown = [], []
    monkeypatch.setattr(fs, "open_default", lambda p: opened.append(p) or str(p))
    monkeypatch.setattr(fs, "reveal", lambda p: shown.append(p) or str(p))
    monkeypatch.setattr(fs, "windows_path", lambda p: None)
    monkeypatch.setattr(runs, "known_configs", lambda: [])

    async def body(app, pilot):
        scr = fs.FilesScreen(tmp_path)
        await app.push_screen(scr)
        await pilot.pause()
        scr.current = big
        scr.action_open()
        scr.current = small
        scr.action_open()
        assert shown == [big] and opened == [small]
        assert "o shows it in its folder" in fs.describe(big).plain

    _pilot(dash.InsarDashboard(), body)


def test_files_preview_says_what_a_layer_is_and_why_a_screening_has_no_report(tmp_path, monkeypatch):
    monkeypatch.setattr(fs, "windows_path", lambda p: None)
    P = tmp_path / "site_d_disp_screening"
    (P / "deliverables").mkdir(parents=True)
    (P / "mintpy").mkdir()
    (P / "mintpy" / "disp_chain.json").write_text("{}")
    note = fs.describe(P / "deliverables").plain
    assert "screening: no client report (see deliverables/SCREENING_NOT_FOR_CLIENTS.txt)" in note
    (P / "deliverables" / "client_report.json").write_text("{}")
    assert "screening: no client report" not in fs.describe(P).plain
    for name, words in (("quality_unwrapping_consistency.tif", "second quality check"),
                        ("quality_too_fast_to_measure.tif", "too fast"),
                        ("quality_position_uncertainty_m.tif", "elevation-model error"),
                        ("disturbed_ground_runs.tif", "not a map of disturbed ground"),
                        ("dem_difference_m.tif", "minus the elevation model")):
        f = P / "deliverables" / name
        f.write_bytes(b"not a tiff")
        assert words in fs.describe(f).plain, name
    assert os.environ.get("INSAR_PROJECTS") != str(P)
