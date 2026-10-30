"""Decisions of 7 Oct 2026, group D4D7_dash (runs.py, dashboard.py).

D4: Resume after an interrupted, failed or stopped Re-unwrap suggests
'Re-unwrap' (read from run.json / run_result.json). What the disk alone
suggested is not always safe there: 'MintPy onward' (after an earlier
unfinished MintPy run) loads the half re-unwrapped stack, mixing old and
new unwraps, and 'Products only' (killed before an unwrap was rewritten)
drops the re-unwrap unannounced. ('Phase linking onward' re-runs the same
unwrap step, whose markers carry each pair's config fingerprint, so it
would redo the remaining pairs too.) Not counted: a re-unwrap refused by
its checks before it touched an unwrap, and one whose MintPy run finished.

D7 (dashboard half): the report's send rule is shown only when
client_report.json is at least as new as quality_assessment.json; when the
QA is newer the dashboard shows the QA's rule marked '(report out of date)'.

Synthetic projects only: no network, no real runs, nothing under
/home/kylew/insar_projects."""
import asyncio
import fcntl
import json
import os
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from insar_pipeline import dashboard as dash, runs, settings

T0 = time.time() - 3 * 86400            # the finished project (after runs.RUN_RECORD_SINCE)
T1 = T0 + 3600                          # the re-unwrap starts
DEAD_PID = 4_000_000                    # above pid_max: never a live process
PAIRS = [f"2017{m:02d}01_2017{m + 1:02d}01" for m in range(1, 9)]


@pytest.fixture(autouse=True)
def _quiet(monkeypatch, tmp_path):
    monkeypatch.setenv("INSAR_NOTIFY", "off")
    monkeypatch.setenv("INSAR_PROJECTS", str(tmp_path / "projects"))
    monkeypatch.setenv("INSAR_SETTINGS", str(tmp_path / "settings.json"))
    monkeypatch.setattr(runs, "REGISTRY", tmp_path / "reg.json")
    monkeypatch.delenv(runs.LOCK_FD_ENV, raising=False)
    dash._RELEASE_CACHE.clear()
    yield
    dash._RELEASE_CACHE.clear()


def _touch(f: Path, t: float, data: str = "") -> Path:
    f.parent.mkdir(parents=True, exist_ok=True)
    if not f.exists():
        f.write_text(data)
    os.utime(f, (t, t))
    return f


def _finished(P: Path, backend="miaplpy"):
    """A finished phase-linked project: every pair exported, configured and
    unwrapped at T0, MintPy finished (its record) at T0+400, report at T0+500."""
    S, M = P / "stack", P / "mintpy"
    for p in PAIRS:
        d = S / "merged" / "interferograms" / p
        for n in ("filt_fine.int", "filt_fine.unw", "filt_fine.unw.conncomp"):
            _touch(d / n, T0)
        _touch(S / "configs" / f"config_igram_unw_{p}", T0)
    _touch(P / "miaplpy" / "inverted" / "phase_series.h5", T0 - 100)
    _touch(M / "inputs" / "ifgramStack.h5", T0 + 100)
    _touch(M / "velocity.h5", T0 + 200)
    _touch(M / "geo" / "geo_velocity.h5", T0 + 300)
    _touch(M / "run_products.json", T0 + 400, "{}")
    _touch(P / "deliverables" / "client_report.json", T0 + 500, "{}")
    return SimpleNamespace(backend=backend, stack_dir=S, mintpy_dir=M)


def _reunwrap_reached(P: Path, n: int, t: float = T1):
    """The re-unwrap's unwrap step rewrote every pair's config at t+5 and
    re-unwrapped the first n pairs by t+60."""
    S = P / "stack"
    for p in PAIRS:
        _touch(S / "configs" / f"config_igram_unw_{p}", t + 5)
    for p in PAIRS[:n]:
        d = S / "merged" / "interferograms" / p
        _touch(d / "filt_fine.unw", t + 60)
        _touch(d / "filt_fine.unw.conncomp", t + 60)


def _runjson(P: Path, kind="reunwrap", started=T1, pid=DEAD_PID, **extra) -> dict:
    rec = {"pid": pid, "start_ticks": 1, "kind": kind, "config": None, "started": started, **extra}
    (P / runs.RUNFILE).write_text(json.dumps(rec))
    return rec


def _result(P: Path, status: str, stage: str, pid=DEAD_PID, finished=T1 + 100, **extra):
    (P / runs.RESULTFILE).write_text(json.dumps({"status": status, "stage": stage, "pid": pid,
                                                 "finished": finished, **extra}))


# ---------------------------------------------------------------- D4: the restart
def test_an_interrupted_reunwrap_resumes_as_reunwrap_not_phase_linking_onward(tmp_path):
    P = tmp_path / "p"
    c = _finished(P)
    assert runs.suggest_kind(P, c)[0] == "products"
    _reunwrap_reached(P, 3)
    _runjson(P)
    # the disk alone says 'phase linking is done'; 'Re-unwrap' is the restart
    # that says it carries on the re-unwrap of the 5 pairs it never reached
    k, why = runs.suggest_kind(P, c)
    assert k == "reunwrap"
    assert why.startswith("the last re-unwrap was interrupted before it finished")
    assert "no pair is left with its old unwrap" in why
    # Resume runs it (pipeline.launch -> KIND_START) and the interruption note names it
    assert runs.KIND_START[k] == "unwrap"
    i = runs.interruption(P, c)
    assert i is None or i["restart_kind"] == "reunwrap"
    (P / "progress.jsonl").write_text(json.dumps({"t": T1 + 1, "stage": "pipeline", "status": "start"}) + "\n"
                                      + json.dumps({"t": T1 + 60, "stage": "run_16_unwrap", "status": "running",
                                                    "done": 3, "total": 8}) + "\n")
    i = runs.interruption(P, c)
    assert i["restart_kind"] == "reunwrap" and "Restart with 'Re-unwrap the phase-linked export" in i["text"]


def test_after_an_earlier_failed_mintpy_it_is_still_reunwrap_not_mintpy_onward(tmp_path):
    """run_incomplete.json from an earlier failed MintPy run used to send the
    half re-unwrapped stack straight into MintPy."""
    P = tmp_path / "p"
    c = _finished(P)
    _touch(P / "mintpy" / "run_incomplete.json", T0 + 450, "{}")
    _reunwrap_reached(P, 3)
    _runjson(P)
    k, why = runs.suggest_kind(P, c)
    assert k == "reunwrap" and "before it finished" in why


def test_killed_before_it_rewrote_an_unwrap_it_is_not_products_only(tmp_path):
    """The unwrap step had rewritten every pair's config (the new settings)
    but no unwrap yet: the disk alone read 'everything is finished', and
    'Products only' dropped the re-unwrap without saying so."""
    P = tmp_path / "p"
    c = _finished(P)
    _reunwrap_reached(P, 0)
    _runjson(P)
    k, why = runs.suggest_kind(P, c)
    assert k == "reunwrap" and why.startswith("the last re-unwrap was interrupted before it finished")


def test_one_that_died_in_mintpy_is_reunwrap_too(tmp_path):
    P = tmp_path / "p"
    c = _finished(P)
    _reunwrap_reached(P, len(PAIRS))
    _touch(P / "mintpy" / "inputs" / "ifgramStack.h5", T1 + 150)
    _touch(P / "mintpy" / "run_incomplete.json", T1 + 160, "{}")
    _runjson(P)
    k, why = runs.suggest_kind(P, c)
    assert (k, why.split(";")[0]) == ("reunwrap", "the last re-unwrap was interrupted during MintPy")


def test_failed_and_stopped_reunwraps_say_how_they_ended(tmp_path):
    P = tmp_path / "p"
    c = _finished(P)
    _reunwrap_reached(P, 3)
    _runjson(P)
    _result(P, "failed", "unwrap", kind="reunwrap", start="unwrap")
    assert runs.suggest_kind(P, c)[1].startswith("the last re-unwrap failed before it finished")
    assert runs.suggest_kind(P, c)[0] == "reunwrap"
    _result(P, "stopped", "unwrap", kind="reunwrap", start="unwrap")
    assert runs.suggest_kind(P, c)[1].startswith("the last re-unwrap was stopped")
    (P / runs.RESULTFILE).unlink()
    _runjson(P, stopped=T1 + 90)                     # runs.stop marks the record
    assert runs.suggest_kind(P, c)[:1] == ("reunwrap",)
    assert runs.suggest_kind(P, c)[1].startswith("the last re-unwrap was stopped")
    # a re-unwrap started from a terminal at the unwrap start is recorded as 'unwrap'
    _runjson(P, kind="unwrap")
    assert runs.suggest_kind(P, c)[0] == "reunwrap"


def test_a_refused_or_a_finished_reunwrap_is_left_to_the_disk(tmp_path):
    # refused by its checks: it touched nothing, and its refusal names the restart
    P = tmp_path / "a"
    c = _finished(P)
    _runjson(P)
    _result(P, "failed", "checks", kind="reunwrap", start="unwrap")
    assert runs.suggest_kind(P, c)[0] == "products"
    # its MintPy finished, then the figures failed: only the products need building
    Q = tmp_path / "b"
    c = _finished(Q)
    _reunwrap_reached(Q, len(PAIRS))
    for rel, t in (("inputs/ifgramStack.h5", T1 + 150), ("velocity.h5", T1 + 250),
                   ("geo/geo_velocity.h5", T1 + 260), ("run_products.json", T1 + 300)):
        _touch(Q / "mintpy" / rel, t)
    _runjson(Q)
    _result(Q, "failed", "figures", kind="reunwrap", start="unwrap", finished=T1 + 900)
    assert runs.suggest_kind(Q, c)[0] == "products"
    # finished: products
    _result(Q, "done", "done", kind="reunwrap", start="unwrap", finished=T1 + 900)
    assert runs.suggest_kind(Q, c)[0] == "products"


def test_only_the_phase_linked_route_and_only_a_reunwrap(tmp_path):
    P = tmp_path / "p"
    c = _finished(P)
    _reunwrap_reached(P, 3)
    _runjson(P, kind="miaplpy")                      # phase linking onward, interrupted in its unwrap step
    assert runs.suggest_kind(P, c)[0] == "miaplpy"
    _runjson(P)
    assert runs.suggest_kind(P, SimpleNamespace(**{**vars(c), "backend": "stack"}))[0] != "reunwrap"


def test_a_resume_that_ran_as_the_reunwrap_counts_as_one(tmp_path):
    P = tmp_path / "p"
    c = _finished(P)
    _reunwrap_reached(P, 3)
    prev = {"pid": DEAD_PID + 1, "start_ticks": 1, "kind": "reunwrap", "started": T1}
    # (a) a Resume that failed: its end record says what it resumed as
    _runjson(P, kind="resume", started=T1 + 1000, previous=prev)
    _result(P, "failed", "unwrap", finished=T1 + 1100, kind="resume", start="unwrap", resumed_as="reunwrap")
    assert runs.suggest_kind(P, c)[:1] == ("reunwrap",)
    assert runs.suggest_kind(P, c)[1].startswith("the last re-unwrap failed")
    # (b) a Resume killed without a record of what it ran: what Resume chose
    # then, i.e. the re-unwrap the run before it left unfinished
    (P / runs.RESULTFILE).unlink()
    assert runs.suggest_kind(P, c)[1].startswith("the last re-unwrap was interrupted")
    # ... but not after a run that was not a re-unwrap
    _runjson(P, kind="resume", started=T1 + 1000, previous={**prev, "kind": "full"})
    assert runs.suggest_kind(P, c)[0] != "reunwrap"
    # (c) killed twice in a row: runs.start notes what each Resume resolved to
    _reunwrap_reached(P, 5, t=T1 + 1000)
    _runjson(P, kind="resume", started=T1 + 2000, resumed_as="reunwrap",
             previous={"pid": DEAD_PID + 2, "start_ticks": 1, "kind": "resume", "started": T1 + 1000,
                       "resumed_as": "reunwrap"})
    assert runs.suggest_kind(P, c)[0] == "reunwrap"
    # a sizing run on top is not a run
    rec = json.loads((P / runs.RUNFILE).read_text())
    _runjson(P, kind="plan", started=T1 + 3000, previous=rec)
    assert runs.suggest_kind(P, c)[0] == "reunwrap"


def test_runs_start_notes_what_a_resume_resolves_to(tmp_path, monkeypatch):
    from insar_pipeline import config as config_mod
    P = tmp_path / "p"
    c = _finished(P)
    _reunwrap_reached(P, 3)
    _runjson(P)
    cfgp = tmp_path / "x.yaml"
    cfgp.write_text("")
    monkeypatch.setattr(config_mod, "load_config", lambda path: c)
    monkeypatch.setitem(runs.KINDS, "resume", ("test", ["-c", "pass"]))
    rec = runs.start(str(cfgp), P, "resume")
    t = time.time()
    while runs.alive(rec) and time.time() - t < 20:
        time.sleep(0.05)
    assert rec["resumed_as"] == "reunwrap" and runs.record(P)["resumed_as"] == "reunwrap"
    assert runs.record(P)["previous"]["kind"] == "reunwrap"
    # that Resume died too: still the re-unwrap
    assert runs.suggest_kind(P, c)[0] == "reunwrap"


def test_two_resumes_killed_before_their_unwrap_step_still_carry_on_the_reunwrap(tmp_path, monkeypatch):
    """run.json keeps only one earlier record: after a Resume that never
    reached the unwrap step was followed by another killed as early, the
    re-unwrap's own record (and its start) was gone, and Resume fell back to
    the disk - here 'MintPy onward' after an earlier unfinished MintPy run,
    which loads the half re-unwrapped stack. runs.start notes when the
    re-unwrap began (reunwrap_started)."""
    from insar_pipeline import config as config_mod
    P = tmp_path / "p"
    c = _finished(P)
    _touch(P / "mintpy" / "run_incomplete.json", T0 + 450, "{}")
    _reunwrap_reached(P, 3)
    _runjson(P)
    cfgp = tmp_path / "x.yaml"
    cfgp.write_text("")
    monkeypatch.setattr(config_mod, "load_config", lambda path: c)
    monkeypatch.setitem(runs.KINDS, "resume", ("test", ["-c", "pass"]))
    for _ in range(2):                               # each Resume ends at once, touching nothing
        rec = runs.start(str(cfgp), P, "resume")
        t = time.time()
        while runs.alive(rec) and time.time() - t < 20:
            time.sleep(0.05)
        assert rec["resumed_as"] == "reunwrap" and rec["reunwrap_started"] == T1
    assert runs.record(P)["previous"]["kind"] == "resume"      # the re-unwrap's own record is gone
    k, why = runs.suggest_kind(P, c)
    assert k == "reunwrap" and why.startswith("the last re-unwrap was interrupted before it finished")
    assert runs.unfinished_reunwrap(P, c)["started"] == T1


def test_the_run_itself_names_reunwrap_even_before_its_record_is_written(tmp_path, monkeypatch):
    """pipeline.launch's Resume reads the disk before run.json and the lock
    note of a run started by runs.start may be written: the lock it inherited
    is its own, not 'a run elsewhere'."""
    P = tmp_path / "p"
    c = _finished(P)
    _reunwrap_reached(P, 3)
    _runjson(P)
    fd = os.open(P / runs.LOCKFILE, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        os.pwrite(fd, json.dumps({"pid": DEAD_PID + 7}).encode(), 0)       # the previous holder's note
        monkeypatch.setenv(runs.LOCK_FD_ENV, str(fd))
        assert runs.suggest_kind(P, c)[0] == "reunwrap"
        monkeypatch.delenv(runs.LOCK_FD_ENV)
        # held by some other process: it is going, so nothing to carry on
        assert runs.unfinished_reunwrap(P, c) is None
    finally:
        os.close(fd)
    # the run's own failure path (run.json is this process, no end record yet)
    _runjson(P, pid=os.getpid(), start_ticks=runs._start_ticks(os.getpid()))
    k, why = runs.suggest_kind(P, c)
    assert k == "reunwrap" and why.startswith("the last re-unwrap ended early;")
    # once it has written its end record it is judged by it
    _result(P, "failed", "unwrap", pid=os.getpid())
    assert runs.suggest_kind(P, c)[1].startswith("the last re-unwrap failed")


def test_a_reunwrap_still_going_elsewhere_is_not_carried_on(tmp_path):
    P = tmp_path / "p"
    c = _finished(P)
    _reunwrap_reached(P, 3)
    p = subprocess.Popen(["sleep", "30"])
    try:
        ticks = None
        for _ in range(50):
            ticks = runs._start_ticks(p.pid)
            if ticks is not None:
                break
            time.sleep(0.02)
        _runjson(P, pid=p.pid, start_ticks=ticks)
        assert runs.unfinished_reunwrap(P, c) is None
    finally:
        p.kill()
        p.wait()


def test_the_end_record_of_a_failing_reunwrap_names_reunwrap(tmp_path, monkeypatch):
    """run_from's failure path writes the restart it detects into
    run_result.json ('What now' in the dashboard)."""
    from insar_pipeline import miaplpy_backend as mb, pipeline
    from insar_pipeline.config import load_config
    f = tmp_path / "c.yaml"
    f.write_text(f"project_dir: {tmp_path / 'proj'}\nbbox: [-87.7, 38.2, -87.6, 38.3]\n"
                 "start_date: '2020-01-01'\nend_date: '2021-01-01'\n")
    cfg = load_config(str(f))
    cfg.backend, cfg.mintpy_tropo_method = "miaplpy", "no"
    P = Path(cfg.project_dir)
    _finished(P)
    monkeypatch.setattr(pipeline, "check_preconditions", lambda cfg, start, force=False: {"pair_names": PAIRS})
    monkeypatch.setattr(mb, "write_pair_coherence", lambda cfg, pairs: {})

    def unwrap(cfg, pairs):
        now = time.time()
        for p in pairs:
            _touch(cfg.stack_dir / "configs" / f"config_igram_unw_{p}", now)
        for p in pairs[:2]:
            _touch(cfg.stack_dir / "merged" / "interferograms" / p / "filt_fine.unw", now)
        raise RuntimeError("snaphu ran out of memory")
    monkeypatch.setattr(mb, "unwrap_exported", unwrap)
    assert pipeline.run_from(cfg, "unwrap", kind="reunwrap", watchdog=False, notify=False) == 1
    res = json.loads((P / runs.RESULTFILE).read_text())
    assert res["status"] == "failed" and res["stage"] == "unwrap"
    assert res["detected_restart"] == "reunwrap"
    assert "re-unwrap" in res["detected_restart_reason"]
    assert runs.suggest_kind(P, cfg)[0] == "reunwrap"


# ---------------------------------------------------------------- D4: the dashboard offers it
def _dash_project(root: Path) -> Path:
    P = root / "demo"
    _finished(P)
    (P / "logs").mkdir(parents=True)
    cfgs = root / "configs"
    cfgs.mkdir(parents=True, exist_ok=True)
    f = cfgs / "demo.yaml"
    f.write_text(f"project_dir: {P}\nbackend: miaplpy\nbbox: [-87.7, 38.3, -87.6, 38.4]\n"
                 "start_date: '2017-01-01'\nend_date: '2020-01-01'\ntrack: 165\n"
                 "stack_virtual_merge: 'False'\nmintpy_deramp: 'no'\nmintpy_tropo_method: 'no'\n"
                 "report_areas: ['Full Satellite Imaging:full']\n")
    _reunwrap_reached(P, 3)
    log = P / "logs" / "reunwrap_x.log"
    log.write_text("unwrapping\n")
    _runjson(P, config=str(f), log=str(log))
    (P / "progress.jsonl").write_text(json.dumps({"t": T1 + 1, "stage": "pipeline", "status": "start"}) + "\n"
                                      + json.dumps({"t": T1 + 60, "stage": "run_16_unwrap", "status": "running",
                                                    "done": 3, "total": 8}) + "\n")
    return f


@pytest.fixture
def home(tmp_path, monkeypatch):
    root = tmp_path / "projects"
    root.mkdir()
    monkeypatch.setattr(settings, "disk_budget_gb",
                        lambda path: {"budget": 400.0, "source": "test", "fs_free": 400.0, "c_free": None})
    monkeypatch.setattr(runs, "_oom_killed", lambda pids, since: False)
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


def test_the_dashboard_recommends_reunwrap_and_starts_it_through_the_preflight(home, monkeypatch):
    f = _dash_project(home)
    started = []
    monkeypatch.setattr(runs, "start", lambda cfg, P, kind="full", extra=None: started.append(kind) or
                        {"pid": 1, "kind": kind})
    monkeypatch.setattr(dash, "preflight_checks", lambda cfg, config, kind="full", say=None: [("ok", "fine")])

    async def body(app, pilot):
        await _settle(app, pilot)
        title = str(app.screen.query_one("#runtitle", dash.Static).render())
        assert "interrupted" in title and "Re-unwrap the phase-linked export" in title
        await pilot.press("s")
        await pilot.pause()
        ch = app.screen
        assert isinstance(ch, dash.ChooseRun)
        ol = ch.query_one(dash.OptionList)
        first = ol.get_option_at_index(0)
        assert first.id == "reunwrap"
        assert "Recommended" in str(first.prompt) and "the last re-unwrap was interrupted" in str(first.prompt)
        ol.highlighted = 0
        await pilot.press("enter")
        await _settle(app, pilot)
        pf = app.screen
        assert isinstance(pf, dash.PreflightScreen) and pf.kind == "reunwrap"
        pf.query_one("#go", dash.Button).press()
        await _settle(app, pilot)
        assert started == ["reunwrap"]

    _pilot(dash.InsarDashboard(str(f)), body)


def test_the_preflight_of_resume_says_it_will_reunwrap(home):
    from insar_pipeline.config import load_config
    f = _dash_project(home)
    cfg = load_config(str(f))
    res = dash.preflight_checks(cfg, str(f), "resume")
    first = res[0]
    assert first[0] == "info"
    assert first[1].startswith("Resume will run 'Re-unwrap the phase-linked export, then MintPy onward': "
                               "the last re-unwrap was interrupted before it finished")
    assert not any(level == "fail" for level, _ in res)


# ---------------------------------------------------------------- D7: which send rule is shown
@pytest.fixture
def proj(tmp_path):
    P = tmp_path / "demo"
    (P / "deliverables").mkdir(parents=True)
    return P


def _write(f: Path, d: dict, t: float) -> None:
    f.write_text(json.dumps(d))
    os.utime(f, (t, t))


def _rep(P, value, t):
    _write(P / "deliverables" / "client_report.json",
           {"verdict": "REVIEW", "areas": [], "send_rule": {"value": value, "reasons": []}}, t)


def _qa(P, value, t):
    _write(P / "quality_assessment.json", {"checks": {}, "send_rule": {"value": value, "reasons": []}}, t)


def test_the_reports_rule_only_when_the_report_is_at_least_as_new_as_the_qa(proj):
    now = time.time()
    _qa(proj, "send", now - 600)
    _rep(proj, "hold for specialist", now - 300)
    assert dash.send_rule_shown(proj) == ("hold for specialist", False)
    assert dash.qa_send_rule(proj) == "hold for specialist"
    _rep(proj, "hold for specialist", now - 600)                 # same time: the report's
    assert dash.qa_send_rule(proj) == "hold for specialist"
    # the QA re-run after the report: its rule, marked; the report must be rebuilt
    _qa(proj, "send with notes", now - 100)
    assert dash.send_rule_shown(proj) == ("send with notes", True)
    assert dash.qa_send_rule(proj) == "send with notes (report out of date)"
    # no report at all: nothing is out of date
    (proj / "deliverables" / "client_report.json").unlink()
    assert dash.qa_send_rule(proj) == "send with notes"


def test_out_of_date_is_the_release_gates_own_test(proj):
    """The mark and the release gate's hard blocker (FINAL / DRAFT beside it)
    use one test: a QA written within the gate's grace of the report (one
    script, one copy) was written with it, not re-run after it."""
    from insar_pipeline import release
    now = time.time()
    _rep(proj, "hold for specialist", now - 100)
    for dt in (0.0, 1.0, release.QA_AFTER_REPORT_GRACE_S + 1.0, 30.0):
        _qa(proj, "send", now - 100 + dt)
        stale = dash.send_rule_shown(proj)[1]
        assert stale == (release.report_predates_qa(proj) is not None), dt
        assert stale == (dt > release.QA_AFTER_REPORT_GRACE_S), dt
    assert dash.qa_send_rule(proj) == "send (report out of date)"


def test_summary_counts_only_when_written_after_the_qa(proj):
    """The provenance order: summary.json's rule (re-evaluated with the GNSS
    tie) only when written after this QA."""
    now = time.time()
    _rep(proj, "send", now - 900)
    _qa(proj, "send", now - 600)
    S = proj / "deliverables" / "summary.json"
    _write(S, {"send_rule": "hold for specialist", "send_rule_gnss_considered": True}, now - 300)
    assert dash.qa_send_rule(proj) == "hold for specialist (report out of date)"
    _write(S, {"send_rule": "hold for specialist", "send_rule_gnss_considered": True}, now - 700)
    assert dash.qa_send_rule(proj) == "send (report out of date)"
    # a current report with no rule of its own: an older summary is not read either
    _rep(proj, None, now)
    assert dash.qa_send_rule(proj) == "send"


def test_the_projects_table_and_the_run_view_mark_an_out_of_date_report(proj, monkeypatch):
    now = time.time()
    _rep(proj, "hold for specialist", now - 600)
    _qa(proj, "send", now - 100)
    monkeypatch.setattr(dash, "release_status", lambda *a, **k: None)
    row = dash.project_row(SimpleNamespace(project_dir=str(proj), verdict="REVIEW", error=None))
    assert row["verdict"].plain == "send (report out of date)"
    _rep(proj, "hold for specialist", now)
    row = dash.project_row(SimpleNamespace(project_dir=str(proj), verdict="REVIEW", error=None))
    assert row["verdict"].plain == "hold for specialist"


def test_the_run_views_results_line_says_rebuild(home, monkeypatch):
    f = _dash_project(home)
    P = home / "demo"
    (P / runs.RUNFILE).unlink()
    now = time.time()
    _write(P / "deliverables" / "client_report.json",
           {"verdict": "REVIEW", "areas": [], "send_rule": {"value": "hold for specialist"}}, now - 600)
    _qa(P, "send with notes", now - 60)
    monkeypatch.setattr(dash, "release_status", lambda *a, **k: None)

    async def body(app, pilot):
        await _settle(app, pilot)
        app.screen._results(P)
        await pilot.pause()
        txt = str(app.screen.query_one("#results", dash.Static).render())
        assert "(send with notes) (report out of date): rebuild the report" in txt
        assert "hold for specialist" not in txt

    _pilot(dash.InsarDashboard(str(f)), body)


def test_the_open_draft_dialog_says_the_report_is_out_of_date_once(home, monkeypatch):
    """The open-draft dialog lists the gate's hard blockers. When the gate has
    its own report_older_than_qa line, that line is said once - first, so it
    is never cut off below the first eight - and the dialog does not add its
    own sentence for the same thing; without the gate's line it still does."""
    f = _dash_project(home)
    P = home / "demo"
    (P / runs.RUNFILE).unlink()
    now = time.time()
    _write(P / "deliverables" / "client_report.json",
           {"verdict": "REVIEW", "areas": [], "send_rule": {"value": "send with notes"}}, now - 600)
    (P / "deliverables" / "client_report.html").write_text("<html></html>")
    _qa(P, "send with notes", now - 60)
    gate = ("The report predates the quality assessment (re-run 07 Oct 10:00, after the report was built "
            "07 Oct 09:50), so its send rule and quality notes are the previous assessment's; rebuild the report.")
    others = [{"code": f"other_{i}", "hard": True, "text": f"Blocker {i}."} for i in range(9)]
    status = {"final": False, "blockers": others + [{"code": "report_older_than_qa", "hard": True, "text": gate}]}
    monkeypatch.setattr(dash, "release_status", lambda *a, **k: status)
    monkeypatch.setattr(dash, "open_path", lambda path: str(path))
    own = "The report is older than the quality assessment (report out of date)"

    async def body(app, pilot):
        await _settle(app, pilot)
        run = app.screen
        run.action_report()
        await pilot.pause()
        q = app.screen.question
        assert isinstance(app.screen, dash.Confirm)
        assert q.count("predates the quality assessment") == 1 and own not in q
        assert q.split("\n\n")[1].startswith(f"• {gate}")
        assert "…and 2 more" in q
        await pilot.press("escape")
        await pilot.pause()
        # no gate line (release.py not importable): the dialog says it itself, once
        status["blockers"] = others[:2]
        run.action_report()
        await pilot.pause()
        q = app.screen.question
        assert q.count(own) == 1 and "predates the quality assessment" not in q

    _pilot(dash.InsarDashboard(str(f)), body)
