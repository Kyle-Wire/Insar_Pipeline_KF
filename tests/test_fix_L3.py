"""Run control (lane L3a, review 5 Oct 2026): teardown, project lock,
interrupted state, restart suggestion, one stage chain with run_result.json,
advice and notification, disk gate and watchdog, DISP client-report gate,
route warning, skipped-aware ETA. Synthetic projects only, no network."""
import errno
import json
import logging
import os
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

from insar_pipeline import advice, notify, pipeline, procs, progress, runs

PARENT = str(Path(__file__).resolve().parents[2])


@pytest.fixture(autouse=True)
def _quiet(monkeypatch, tmp_path):
    monkeypatch.setenv("INSAR_NOTIFY", "off")
    monkeypatch.setattr(notify, "LOGFILE", tmp_path / "notifications.log")
    monkeypatch.setattr(runs, "REGISTRY", tmp_path / "reg.json")
    yield
    procs.clear_stopping()


def _cfg(tmp_path, backend="miaplpy", **attrs):
    from insar_pipeline.config import load_config
    f = tmp_path / "c.yaml"
    f.write_text(f"project_dir: {tmp_path / 'proj'}\nbbox: [-87.7, 38.2, -87.6, 38.3]\n"
                 "start_date: '2020-01-01'\nend_date: '2021-01-01'\n")
    cfg = load_config(str(f))
    cfg.backend = backend
    for k, v in attrs.items():
        setattr(cfg, k, v)
    cfg.project_dir.mkdir(parents=True, exist_ok=True)
    return cfg


def _stub_products(monkeypatch, calls, fail=()):
    def mk(name, ok=True):
        def f(cfg):
            calls.append(name)
            progress.emit(cfg.project_dir, name, "done" if name not in fail else "failed")
            return name not in fail
        return f

    def export(cfg):
        calls.append("deliverables")
        if "deliverables" in fail:
            raise FileNotFoundError("velocity.h5 not found")
        return cfg.project_dir / "deliverables"
    monkeypatch.setattr(pipeline, "export_all", export)
    monkeypatch.setattr(pipeline, "run_quality_assessment", mk("quality_assessment"))
    monkeypatch.setattr(pipeline, "run_figures", mk("figures"))
    monkeypatch.setattr(pipeline, "run_client_report", mk("client_report"))


def _run_py(code: str, **kw) -> subprocess.Popen:
    env = {**os.environ, "PYTHONPATH": PARENT, "PYTHONUNBUFFERED": "1"}
    return subprocess.Popen([sys.executable, "-c", textwrap.dedent(code)], stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True, env=env, start_new_session=True, **kw)


def _gone(pid, timeout=5.0) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout:
        if procs.start_ticks(pid) is None:
            return True
        time.sleep(0.05)
    return False


# ---------------------------------------------------------------- stackrunner-3 / ops-11
def test_signals_installed_after_a_worker_thread_called_first():
    """Old code: the worker-thread call set the one flag, so SIGTERM stayed SIG_DFL."""
    p = _run_py("""
        import signal
        from concurrent.futures import ThreadPoolExecutor
        from insar_pipeline import procs
        with ThreadPoolExecutor(2) as ex:
            list(ex.map(lambda _: procs.install_teardown(), range(2)))
        assert not procs.signals_installed()
        procs.install_teardown()
        print(signal.getsignal(signal.SIGTERM) not in (signal.SIG_DFL, signal.SIG_IGN), procs.signals_installed())
    """)
    out, err = p.communicate(timeout=30)
    assert out.split() == ["True", "True"], err


def test_stop_kills_a_child_started_from_a_worker_thread():
    """End to end, the way runs.stop signals: the runner exits 143 and its child dies."""
    p = _run_py("""
        import threading, time
        from insar_pipeline import procs
        box = []
        def w():
            procs.install_teardown()
            box.append(procs.popen_group(["sleep", "60"]).pid)
        t = threading.Thread(target=w); t.start(); t.join()
        procs.install_teardown()          # stack_backend.execute's main-thread call
        print(box[0], flush=True)
        time.sleep(60)
    """)
    child = int(p.stdout.readline())
    try:
        os.killpg(p.pid, signal.SIGTERM)
        p.wait(timeout=15)
        assert p.returncode == 143
        assert _gone(child), "child survived the stop"
    finally:
        try:
            os.killpg(child, signal.SIGKILL)
        except ProcessLookupError:
            pass


def test_run_from_installs_teardown_and_records_children(tmp_path):
    """The restart kinds (phase linking onward, re-unwrap) installed nothing:
    run_from must, in the main thread, before any stage."""
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text(f"project_dir: {tmp_path / 'proj'}\nbbox: [-87.7, 38.2, -87.6, 38.3]\n"
                        "start_date: '2020-01-01'\nend_date: '2021-01-01'\n")
    p = _run_py(f"""
        import os, signal, json
        os.environ["INSAR_NOTIFY"] = "off"
        from insar_pipeline import pipeline, procs
        from insar_pipeline.config import load_config
        seen = {{}}
        def export(cfg):
            seen["sig"] = signal.getsignal(signal.SIGTERM) not in (signal.SIG_DFL, signal.SIG_IGN)
            c = procs.popen_group(["sleep", "0.2"]); c.wait(); procs.unregister(c.pid)
            seen["file"] = os.path.exists(cfg.project_dir / ".run_children.json")
            return cfg.project_dir
        pipeline.export_all = export
        cfg = load_config({str(cfg_path)!r}); cfg.ensure_dirs()
        rc = pipeline.run_from(cfg, "products", no_acceptance=True, watchdog=False, notify=False)
        print(json.dumps({{**seen, "rc": rc}}))
    """)
    out, err = p.communicate(timeout=60)
    rec = json.loads(out.strip().splitlines()[-1])
    assert rec == {"sig": True, "file": True, "rc": 0}, err[-2000:]


def test_orphans_of_a_killed_runner_are_stopped(tmp_path):
    """A runner killed outright (no teardown) left snaphu running; its recorded
    groups are now stopped by runs.stop, identified by start ticks."""
    P = tmp_path / "proj"; P.mkdir()
    p = _run_py(f"""
        import time
        from insar_pipeline import procs
        procs.track_children_in({str(P / runs.CHILDFILE)!r})
        c = procs.popen_group(["sleep", "60"])
        print(c.pid, flush=True)
        time.sleep(60)
    """)
    child = int(p.stdout.readline())
    try:
        os.kill(p.pid, signal.SIGKILL)               # OOM-kill style: no handler runs
        p.wait(timeout=5)
        assert procs.start_ticks(child) is not None  # the orphan the old code left behind
        how = runs.stop(P, grace_s=2)
        assert "stopped 1 leftover" in how
        assert _gone(child)
    finally:
        try:
            os.killpg(child, signal.SIGKILL)
        except ProcessLookupError:
            pass


def test_orphans_check_start_ticks(tmp_path):
    """A recorded pgid now belonging to another process is never signalled."""
    f = tmp_path / runs.CHILDFILE
    f.write_text(json.dumps({"runner_pid": 999999999, "runner_start_ticks": 1,
                             "groups": {str(os.getpid()): 1}}))      # our pid, wrong ticks
    assert procs.recorded_groups(f) == {}
    assert runs.stop_orphans(tmp_path) == {"stopped": [], "survivors": []}


# ---------------------------------------------------------------- ops-5 / usability-11 / stackrunner-14
def test_project_lock_refuses_a_second_run_and_shows_running(tmp_path):
    P = tmp_path / "proj"; P.mkdir()
    holder = _run_py(f"""
        import time
        from insar_pipeline import runs
        with runs.project_lock({str(P)!r}, kind="full"):
            print("held", flush=True)
            time.sleep(30)
    """)
    try:
        assert holder.stdout.readline().strip() == "held"
        info = runs.lock_holder(P)
        assert info and info["pid"] == holder.pid and info["kind"] == "full"
        assert runs.state(P) == "running"               # no run.json: started outside the dashboard
        with pytest.raises(RuntimeError, match=rf"already running \(a full run, pid {holder.pid}"):
            with runs.project_lock(P):
                pass
        cfgp = tmp_path / "x.yaml"; cfgp.write_text("")
        with pytest.raises(RuntimeError, match="already running"):
            runs.start(str(cfgp), P, "products")
        assert not (P / runs.RUNFILE).exists()
    finally:
        os.killpg(holder.pid, signal.SIGKILL)
        holder.wait()
    assert runs.lock_holder(P) is None                    # the kernel dropped it with the process


def test_dashboard_start_hands_its_lock_to_the_run(tmp_path, monkeypatch):
    P = tmp_path / "proj"
    cfgp = tmp_path / "x.yaml"; cfgp.write_text("")
    monkeypatch.setitem(runs.KINDS, "sleep", ("test", ["-c", "import time; time.sleep(30)"]))
    rec = runs.start(str(cfgp), P, "sleep")
    try:
        held = runs.lock_holder(P)
        assert held and held["pid"] == rec["pid"]
        with pytest.raises(RuntimeError, match="already running"):
            with runs.project_lock(P):
                pass
    finally:
        runs.stop(P, grace_s=5)
    assert runs.lock_holder(P) is None and runs.state(P) == "stopped"


def test_project_lock_is_reentrant_and_launch_refuses_when_held(tmp_path):
    cfg = _cfg(tmp_path)
    with runs.project_lock(cfg.project_dir, kind="a"):
        with runs.project_lock(cfg.project_dir, kind="b"):
            pass
        assert runs.lock_holder(cfg.project_dir)["kind"] == "a"
    holder = _run_py(f"""
        import time
        from insar_pipeline import runs
        with runs.project_lock({str(cfg.project_dir)!r}, kind="full"):
            print("held", flush=True); time.sleep(30)
    """)
    try:
        holder.stdout.readline()
        assert pipeline.launch(tmp_path / "c.yaml", "products", notify=False, watchdog=False) == 3
        assert not (cfg.project_dir / runs.RESULTFILE).exists()
        assert not progress.path_for(cfg.project_dir).exists()
    finally:
        os.killpg(holder.pid, signal.SIGKILL)
        holder.wait()


def test_cli_run_writes_its_own_run_record(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    calls = []
    _stub_products(monkeypatch, calls)
    assert pipeline.run_from(cfg, "products", kind="products", no_acceptance=True, watchdog=False) == 0
    rec = runs.record(cfg.project_dir)
    assert rec["pid"] == os.getpid() and rec["kind"] == "products" and rec["launched_by"] == "self"


# ---------------------------------------------------------------- ops-4 / ops-12
def test_failure_writes_run_result_with_plain_cause_and_notifies(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    sent = []
    monkeypatch.setattr(notify, "send", lambda title, text: sent.append((title, text)) or {})
    calls = []
    _stub_products(monkeypatch, calls)
    from insar_pipeline import miaplpy_backend as mb

    def boom(cfg):
        raise ValueError("could not convert string to float: ''")
    monkeypatch.setattr(mb, "run", boom)
    monkeypatch.setattr(pipeline, "check_preconditions", lambda *a, **k: {})
    rc = pipeline.run_from(cfg, "miaplpy", kind="miaplpy", watchdog=False)
    assert rc == 1 and calls == []
    r = json.loads((cfg.project_dir / runs.RESULTFILE).read_text())
    assert r["status"] == "failed" and r["stage"] == "miaplpy" and r["kind"] == "miaplpy"
    assert "could not convert string to float" in r["plain_cause"] and r["next_action"]
    assert isinstance(r["log_tail"], list) and r["finished"] >= r["started"]
    assert progress.latest(cfg.project_dir)["pipeline"]["status"] == "failed"
    assert runs.state(cfg.project_dir) == "failed"
    assert sent and "FAILED" in sent[0][0]


@pytest.mark.parametrize("start,fail,code,events", [
    ("products", (), 0, ["deliverables", "quality_assessment", "figures", "client_report"]),
    ("products", ("figures",), 1, ["deliverables", "quality_assessment", "figures", "client_report"]),
    ("products", ("deliverables",), 1, ["deliverables"]),
    ("mintpy", (), 0, ["mintpy", "deliverables", "quality_assessment", "figures", "client_report"]),
])
def test_one_chain_events_and_exit_codes(tmp_path, monkeypatch, start, fail, code, events):
    """Every start point: a progress event for every stage, a final pipeline
    event, and a non-zero exit when ANY stage failed (main used to exit 0
    after 'deliverables failed' and say 'pipeline done' after a failed report)."""
    cfg = _cfg(tmp_path, backend="stack")
    calls = []
    _stub_products(monkeypatch, calls, fail=fail)
    from insar_pipeline import stack_backend as sb, time_series
    monkeypatch.setattr(sb, "mintpy_loader_lines", lambda cfg: "")
    monkeypatch.setattr(time_series, "run_mintpy_with_loader", lambda cfg, lines: calls.append("mintpy"))
    rc = pipeline.run_from(cfg, start, watchdog=False, notify=False)
    assert rc == code and calls == events
    L = progress.latest(cfg.project_dir)
    assert L["pipeline"]["status"] == ("done" if code == 0 else "failed")
    if "deliverables" in fail:
        assert L["deliverables"]["status"] == "failed"
    r = json.loads((cfg.project_dir / runs.RESULTFILE).read_text())
    assert r["exit_code"] == code and r["status"] == ("done" if code == 0 else "failed")
    if fail:
        assert r["stage"] == fail[0] and fail[0] in r["failed_stages"]
    assert runs.state(cfg.project_dir) == r["status"]


def test_star_network_override_applies_to_every_start(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path, miaplpy_network_connections=0, mintpy_unwrap_error_method="phase_closure")
    _stub_products(monkeypatch, [])
    monkeypatch.setattr(pipeline, "check_preconditions", lambda *a, **k: {})
    seen = {}
    from insar_pipeline import stack_backend as sb, time_series
    monkeypatch.setattr(sb, "mintpy_loader_lines", lambda cfg: "")
    monkeypatch.setattr(time_series, "run_mintpy_with_loader",
                        lambda cfg, lines: seen.setdefault("m", cfg.mintpy_unwrap_error_method))
    assert pipeline.run_from(cfg, "mintpy", watchdog=False, notify=False, no_acceptance=True) == 0
    assert seen["m"] == "bridging"


def test_advice_names_causes_in_plain_words(tmp_path):
    cases = {
        "disk": OSError(errno.ENOSPC, "No space left on device"),
        "login": RuntimeError("no urs.earthdata.nasa.gov entry in ~/.netrc"),
        "weather": RuntimeError("ERA5 download: 403 Client Error: required licences not accepted"),
        "memory": MemoryError(),
        "network": ConnectionError("Max retries exceeded with url"),
        "stopped": KeyboardInterrupt(),
        "refused": SystemExit("backend miaplpy needs stack_virtual_merge: \"False\""),
    }
    for cat, exc in cases.items():
        a = advice.explain(exc, "coregistration", restart="miaplpy")
        assert a["category"] == cat, (cat, a)
        assert a["plain_cause"] and a["next_action"]
    a = advice.explain(RuntimeError("wrapped"), "x")
    assert a["category"] == "unknown" and "wrapped" in a["plain_cause"]
    try:                                                 # a cause chain still says "disk"
        try:
            raise OSError(errno.ENOSPC, "No space left")
        except OSError as inner:
            raise RuntimeError("h5py write failed") from inner
    except RuntimeError as outer:
        assert advice.explain(outer, "mintpy")["category"] == "disk"
    m = tmp_path / "mintpy"; m.mkdir()
    (m / "mintpy_error.log").write_text("step 1\nok\nValueError: reference point is masked out\n")
    from types import SimpleNamespace
    a = advice.explain(RuntimeError("MintPy processing failed - see mintpy_error.log"), "mintpy",
                       cfg=SimpleNamespace(project_dir=tmp_path, mintpy_dir=m))
    assert a["category"] == "mintpy" and "reference point is masked out" in a["plain_cause"]


def test_notify_never_raises_and_escapes(monkeypatch, tmp_path):
    monkeypatch.delenv("INSAR_NOTIFY")
    monkeypatch.setattr(notify, "powershell", lambda: None)
    out = notify.send("t", "x")
    assert out["logged"] and out["toast"] is False
    assert "x" in notify.LOGFILE.read_text()
    monkeypatch.setattr(notify, "powershell", lambda: "/nonexistent/powershell.exe")
    assert notify.send("t", "y")["toast"] is False             # OSError swallowed
    s = notify.toast_script("it's <b>", "a'); Remove-Item C:\\ -Recurse; ('")
    assert "it''s &lt;b&gt;" in s and "a''); Remove-Item" in s   # stays inside one quoted string


# ---------------------------------------------------------------- ops-9
def test_vanished_run_is_interrupted_not_failed(tmp_path, monkeypatch):
    P = tmp_path / "proj"; P.mkdir()
    t0 = time.time()
    (P / runs.RUNFILE).write_text(json.dumps({"pid": 999999999, "start_ticks": 5, "kind": "miaplpy",
                                             "started": t0 - 100}))
    progress.emit(P, "run", "start")
    progress.emit(P, "pipeline", "start")
    progress.emit(P, "run_16_unwrap", "running", done=400, total=804)
    assert runs.state(P) == "interrupted"                 # was "failed", with a clock still running
    monkeypatch.setattr(runs, "_boot_time", lambda: time.time() + 1)
    i = runs.interruption(P)
    assert i["stage"] == "run_16_unwrap" and "restarted" in i["likely"] and i["restart_kind"]
    monkeypatch.setattr(runs, "_boot_time", lambda: 0.0)
    monkeypatch.setattr(runs, "_oom_killed", lambda pids, since: True)
    assert "out of memory" in runs.interruption(P)["likely"]
    ev = progress.latest(P)["run_16_unwrap"]
    e = progress.eta_seconds(dict(ev, started=ev["t"] - 400))
    assert progress.eta_seconds(dict(ev, started=ev["t"] - 400), now=ev["t"] + 100) == pytest.approx(e - 100)


def test_a_launch_that_dies_before_any_stage_is_still_failed(tmp_path):
    P = tmp_path / "proj"; P.mkdir()
    (P / runs.RUNFILE).write_text(json.dumps({"pid": 999999999, "kind": "full", "started": time.time()}))
    progress.emit(P, "run", "start")
    assert runs.state(P) == "failed"


# ---------------------------------------------------------------- ops-10
def test_plan_only_does_not_change_a_finished_project(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path, backend="stack")
    P = cfg.project_dir
    (P / runs.RUNFILE).write_text(json.dumps({"pid": 999999999, "kind": "full", "started": time.time() - 50}))
    progress.emit(P, "pipeline", "done")
    assert runs.state(P) == "done"
    monkeypatch.setitem(pipeline.RUNNERS, "stack", lambda cfg, args: None)
    assert pipeline.run_from(cfg, "full", plan_only=True) == 0
    assert progress.latest(P)["pipeline"]["status"] == "done"        # no "pipeline start" written
    assert not (P / runs.RESULTFILE).exists() and runs.state(P) == "done"
    # the dashboard's "Size the run only" kind writes run.json; the run before it decides the state
    prev = runs.record(P)
    (P / runs.RUNFILE).write_text(json.dumps({"pid": 999999998, "kind": "plan", "started": time.time(),
                                             "previous": prev}))
    progress.emit(P, "run", "start", msg="plan")
    assert runs.state(P) == "done"


def test_to_step_ends_with_a_final_event(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path, backend="stack")
    monkeypatch.setitem(pipeline.RUNNERS, "stack", lambda cfg, args: None)
    assert pipeline.run_from(cfg, "full", to_step="filter", watchdog=False, notify=False) == 0
    ev = progress.latest(cfg.project_dir)["pipeline"]
    assert ev["status"] == "done" and "filter" in ev["msg"]
    assert runs.state(cfg.project_dir) == "done"


# ---------------------------------------------------------------- ops-15
def test_eta_ignores_work_verified_from_the_last_run(tmp_path):
    """600 of 804 verified in 2 s, then 10 real unwraps at 12 s each: the old
    estimate said 39 s left against ~41 min."""
    t = 1_000_000.0
    lines = [{"t": t, "stage": "u", "status": "start", "done": 0, "total": 804, "skipped": 0}]
    for i in range(1, 601):
        lines.append({"t": t + 2 * i / 600, "stage": "u", "status": "running", "done": i, "total": 804, "skipped": i})
    for k in range(1, 11):
        lines.append({"t": t + 2 + 12 * k, "stage": "u", "status": "running", "done": 600 + k, "total": 804,
                      "skipped": 600})
    progress.path_for(tmp_path).write_text("".join(json.dumps(x) + "\n" for x in lines))
    ev = progress.latest(tmp_path)["u"]
    truth = 194 * 12.0
    assert abs(progress.eta_seconds(ev) - truth) / truth < 0.2
    progress.emit(tmp_path, "v", "running", done=5, total=10, skipped=0)   # emit carries the count
    assert progress.latest(tmp_path)["v"]["skipped"] == 0


# ---------------------------------------------------------------- ops-6
def test_phase_linked_route_has_a_disk_gate(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path, stack_virtual_merge="False", stack_cleanup_coreg_after_filter=False)
    from types import SimpleNamespace
    from insar_pipeline import stack_backend as sb
    monkeypatch.setattr(sb, "plan_stack", lambda cfg: SimpleNamespace(est_peak_gb=311.0, free_gb=900.0))
    monkeypatch.setattr(sb, "format_plan", lambda plan: "plan")
    monkeypatch.setattr(sb, "stage_inputs", lambda *a, **k: pytest.fail("ran past the gate"))
    monkeypatch.setattr(pipeline, "disk_budget", lambda path: (100.0, "stub"))
    args = SimpleNamespace(plan_only=True, force=False, skip_download=False)
    with pytest.raises(SystemExit, match="disk budget is 100 GB"):
        pipeline.run_miaplpy(cfg, args)
    # a resume needs only what the project does not hold yet
    monkeypatch.setattr(pipeline, "project_gb", lambda p: 250.0)
    pipeline.run_miaplpy(cfg, args)                            # 61 GB more < 100 GB: plan returns
    monkeypatch.setattr(pipeline, "project_gb", lambda p: 0.0)
    pipeline.run_miaplpy(cfg, SimpleNamespace(plan_only=True, force=True, skip_download=False))


def test_disk_watchdog_stops_the_run_cleanly(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path, backend="stack")
    _stub_products(monkeypatch, [])
    monkeypatch.setattr(pipeline, "DISK_SAMPLE_S", 0.05)
    monkeypatch.setattr(pipeline, "disk_budget", lambda path: (10.0, "stub"))
    killed = []
    monkeypatch.setattr(procs, "kill_children", lambda *a, **k: killed.append(1) or 0)

    def slow(cfg, args):
        for _ in range(100):
            time.sleep(0.05)
    monkeypatch.setitem(pipeline.RUNNERS, "stack", slow)
    rc = pipeline.run_from(cfg, "full", notify=False)
    assert rc == 1 and killed
    r = json.loads((cfg.project_dir / runs.RESULTFILE).read_text())
    assert r["status"] == "failed" and r["category"] == "disk" and r["stage"] == "processing"
    assert progress.latest(cfg.project_dir)["disk"]["status"] == "failed"


# ---------------------------------------------------------------- disp-1 (pipeline part)
def test_screening_route_builds_no_client_report_by_default(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path, backend="disp", disp_record_chain=True)
    D = cfg.project_dir / "deliverables"; D.mkdir()
    (D / "client_report.html").write_text("old official-looking report")
    (D / "Site A_ground_movement.kmz").write_text("kmz")
    calls = []
    _stub_products(monkeypatch, calls)
    assert pipeline.run_from(cfg, "products", watchdog=False, notify=False) == 0
    assert "client_report" not in calls and calls[:3] == ["deliverables", "quality_assessment", "figures"]
    assert progress.latest(cfg.project_dir)["client_report"]["status"] == "skipped"
    assert not (D / "client_report.html").exists()
    assert list((D / "screening_not_for_clients").glob("*/client_report.html"))
    assert "NOT FOR CLIENTS" in (D / pipeline.SCREENING_NOTE).read_text()
    r = json.loads((cfg.project_dir / runs.RESULTFILE).read_text())
    assert r["route"] == "screening" and "screening" in r["plain_cause"]
    calls.clear()
    cfg.disp_client_report = True
    assert pipeline.run_from(cfg, "products", watchdog=False, notify=False) == 0
    assert "client_report" in calls


# ---------------------------------------------------------------- tests-12
def test_non_official_backend_logs_a_warning(tmp_path, monkeypatch, caplog):
    _stub_products(monkeypatch, [])
    for backend, warned in (("stack", True), ("miaplpy", False)):
        cfg = _cfg(tmp_path, backend=backend)
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger="insar_pipeline.pipeline"):
            pipeline.run_from(cfg, "products", no_acceptance=True, watchdog=False, notify=False)
        hit = [r for r in caplog.records if "Official results come from" in r.getMessage()]
        assert bool(hit) == warned, backend
        assert json.loads((cfg.project_dir / runs.RESULTFILE).read_text())["route"] == \
            ("comparison" if warned else "official")


# ---------------------------------------------------------------- phaselinked-3
def _site_b_like(P: Path, dates, merged=None, slcstack=False, phase_series=False, unwrapped=None):
    S = P / "stack"
    (S / "configs").mkdir(parents=True)
    (S / "coreg_secondarys").mkdir()
    st = S / ".status" / "run_12_merge_reference_secondary_slc"; st.mkdir(parents=True)
    for d in dates:
        (S / "configs" / f"config_merge_{d}").write_text("")
        (S / "coreg_secondarys" / f"{d}.cleaned").write_text("")
    for i, d in enumerate(merged if merged is not None else dates):
        (st / f"{i:016x}.done").write_text(json.dumps(
            {"cmd": f"SentinelWrapper.py -c {S}/configs/config_merge_{d}", "returncode": 0}))
    if slcstack:
        (P / "miaplpy" / "inputs").mkdir(parents=True, exist_ok=True)
        (P / "miaplpy" / "inputs" / "slcStack.h5").write_text("")
    if phase_series:
        (P / "miaplpy" / "inverted").mkdir(parents=True, exist_ok=True)
        (P / "miaplpy" / "inverted" / "phase_series.h5").write_text("")
    for pair, done in (unwrapped or {}).items():
        d = S / "merged" / "interferograms" / pair; d.mkdir(parents=True)
        (d / "filt_fine.int").write_text("")
        if done:
            (d / "filt_fine.unw").write_text(""); (d / "filt_fine.unw.conncomp").write_text("")


def test_suggest_kind_reads_the_disk(tmp_path):
    from types import SimpleNamespace
    dates = [f"2016{m:02d}01" for m in range(1, 13)]

    def case(name, **kw):
        P = tmp_path / name
        _site_b_like(P, dates, **kw)
        return runs.suggest_kind(P, SimpleNamespace(backend="miaplpy", stack_dir=P / "stack",
                                                    mintpy_dir=P / "mintpy"))
    # Site B after the merge: every date .cleaned, slcStack present, no phase series -> NOT 'full'
    assert case("loaded", slcstack=True)[0] == "miaplpy"
    assert case("merged")[0] == "miaplpy"
    k, why = case("half", merged=dates[:6])
    assert k == "full" and "6 of 12" in why
    k, why = case("unw", phase_series=True, unwrapped={"20160101_20160201": True, "20160201_20160301": False})
    assert k == "miaplpy" and "1 of 2" in why
    P = tmp_path / "mp"
    _site_b_like(P, dates, phase_series=True, unwrapped={"20160101_20160201": True})
    (P / "mintpy" / "inputs").mkdir(parents=True)
    (P / "mintpy" / "inputs" / "ifgramStack.h5").write_text("")
    c = SimpleNamespace(backend="miaplpy", stack_dir=P / "stack", mintpy_dir=P / "mintpy")
    assert runs.suggest_kind(P, c)[0] == "mintpy"
    time.sleep(0.02)
    # a bare radar velocity.h5 written after run records began (5 Oct 2026):
    # MintPy died after it (no geocode, no run_products.json) -> not finished
    vel = P / "mintpy" / "velocity.h5"
    vel.write_text("")
    assert vel.stat().st_mtime > runs.RUN_RECORD_SINCE
    assert runs.suggest_kind(P, c)[0] == "mintpy"
    # geocoded and recorded no earlier than the velocity: finished
    (P / "mintpy" / "geo").mkdir()
    (P / "mintpy" / "geo" / "geo_velocity.h5").write_text("")
    (P / "mintpy" / "run_products.json").write_text("{}")
    t = vel.stat().st_mtime
    for f in (P / "mintpy" / "geo" / "geo_velocity.h5", P / "mintpy" / "run_products.json"):
        os.utime(f, (t, t))
    assert runs.suggest_kind(P, c)[0] == "products"
    assert list(runs.KINDS)[0] == "resume" and runs.KIND_START["reunwrap"] == "unwrap"


def test_phase_linking_onward_refuses_a_half_merged_stack(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    dates = ["20160101", "20160113", "20160125"]
    _site_b_like(cfg.project_dir, dates, merged=dates[:2])
    from insar_pipeline import miaplpy_backend as mb
    monkeypatch.setattr(mb, "run", lambda cfg: pytest.fail("phase linking started on a short stack"))
    rc = pipeline.run_from(cfg, "miaplpy", watchdog=False, notify=False)
    r = json.loads((cfg.project_dir / runs.RESULTFILE).read_text())
    assert rc == 1 and r["stage"] == "checks" and "20160125" in r["plain_cause"]


def test_reunwrap_refuses_an_incomplete_export(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path, miaplpy_network_connections=2, stack_reference_date="20160101")
    dates = ["20160101", "20160113", "20160125", "20160206"]
    from insar_pipeline import miaplpy_backend as mb
    exp = [f"{a}_{b}" for a, b, _, _ in mb.network_pairs(dates, 2, "20160101")]
    _site_b_like(cfg.project_dir, dates, unwrapped={p: True for p in exp if p != "20160113_20160206"})
    monkeypatch.setattr(pipeline, "_phase_series_dates", lambda cfg: dates)
    monkeypatch.setattr(mb, "unwrap_exported", lambda *a: pytest.fail("unwrapped a short export"))
    with pytest.raises(SystemExit, match="20160113_20160206"):
        pipeline.check_preconditions(cfg, "unwrap")
    assert pipeline.run_from(cfg, "unwrap", watchdog=False, notify=False) == 1
    pipeline.check_preconditions(cfg, "unwrap", force=True)        # --force goes ahead, with a warning


def test_resume_kind_starts_where_the_disk_says(tmp_path, monkeypatch):
    _cfg(tmp_path)
    seen = {}
    monkeypatch.setattr(runs, "suggest_kind", lambda P, cfg=None: ("products", "MintPy finished"))
    monkeypatch.setattr(pipeline, "run_from", lambda cfg, start, **o: seen.update(start=start, **o) or 0)
    assert pipeline.launch(tmp_path / "c.yaml", "resume") == 0
    assert seen["start"] == "products" and seen["kind"] == "resume" and seen["resumed_as"] == "products"


def test_other_live_runs_are_listed(tmp_path, monkeypatch):
    """stackrunner-14: a second project's run competes for the same machine."""
    monkeypatch.setenv("INSAR_PROJECTS", str(tmp_path / "projects"))
    a, b = tmp_path / "projects" / "a", tmp_path / "projects" / "b"
    a.mkdir(parents=True); b.mkdir()
    holder = _run_py(f"""
        import time
        from insar_pipeline import runs
        with runs.project_lock({str(b)!r}, kind="full"):
            print("held", flush=True); time.sleep(30)
    """)
    try:
        holder.stdout.readline()
        live = runs.live_runs(exclude=a)
        assert [(Path(x["project"]).name, x["pid"]) for x in live] == [("b", holder.pid)]
        assert runs.live_runs(exclude=b) == []
    finally:
        os.killpg(holder.pid, signal.SIGKILL)
        holder.wait()


def test_dashboard_run_end_to_end_adopts_the_lock_and_reports(tmp_path):
    """runs.start -> `pipeline --start products` in its own session: the run
    adopts the lock runs.start handed it (no 'already running' against
    itself), keeps runs.start's run.json, and an empty project ends as a
    reported failure with a plain cause - not a bare traceback."""
    cfg = _cfg(tmp_path)
    P = cfg.project_dir
    rec = runs.start(str(tmp_path / "c.yaml"), P, "products")
    t0 = time.time()
    while time.time() - t0 < 60 and runs.state(P) == "running":
        time.sleep(0.2)
    r = runs.result(P)
    assert r is not None, runs.tail(rec["log"], 40)
    assert r["pid"] == rec["pid"] and r["status"] == "failed" and r["stage"] == "deliverables"
    assert r["plain_cause"] and r["next_action"] and r["log"] == rec["log"] and r["log_tail"]
    assert runs.record(P)["pid"] == rec["pid"] and runs.state(P) == "failed"
    assert runs.lock_holder(P) is None


def test_a_config_that_does_not_load_is_reported(tmp_path):
    P = tmp_path / "proj"; P.mkdir()
    bad = tmp_path / "bad.yaml"
    bad.write_text(f"project_dir: {P}\nbbox: [-87.7, 38.2, -87.6, 38.3]\nstart_date: '2020-01-01'\n"
                   "end_date: '2021-01-01'\nbackend: miaplpy\n")
    with pytest.raises(Exception):
        pipeline.launch(bad, "full")
    r = runs.result(P)
    assert r and r["stage"] == "config" and "config file has problems" in r["plain_cause"]
    assert progress.latest(P)["pipeline"]["status"] == "failed"


# ================================================================ lane L3b: launchers, cleanup, planner
def _yaml(tmp_path, backend="miaplpy", name="cfg.yaml") -> Path:
    f = tmp_path / name
    f.write_text(f"project_dir: {tmp_path / 'proj'}\nbackend: {backend}\ntrack: 165\nmintpy_deramp: 'no'\n"
                 "stack_virtual_merge: 'False'\nbbox: [-87.7, 38.2, -87.6, 38.3]\n"
                 "start_date: '2020-01-01'\nend_date: '2021-01-01'\n")
    return f


def _touch(p: Path, data: bytes = b"x" * 64, t: float | None = None) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)
    if t is not None:
        os.utime(p, (t, t))
    return p


def _finished_project(tmp_path, chain=("ramp", "demErr"), backend="miaplpy"):
    """A phase-linked project after a good run: MintPy chain without ERA5
    (`chain`), products newer than the velocity, logs of four runs, inputs
    of every level, a packaged release."""
    from insar_pipeline.config import load_config
    cf = _yaml(tmp_path, backend)
    cfg = load_config(str(cf))
    P, M, D = cfg.project_dir, cfg.mintpy_dir, cfg.project_dir / "deliverables"
    t = time.time() - 10_000
    names = ["timeseries"] + ["timeseries_" + "_".join(chain[:i + 1]) for i in range(len(chain))]
    for i, n in enumerate(names):
        _touch(M / f"{n}.h5", t=t + i)
    _touch(M / "timeseriesResidual.h5", t=t + 10)
    _touch(M / "geo" / f"geo_{names[-1]}.h5", t=t + 20)
    _touch(M / "velocity.h5", t=t + 20)
    _touch(M / "geo" / "geo_velocity.h5", t=t + 21)
    _touch(M / "pic" / "a.png")
    _touch(P / "quality_assessment.json", b"{}", t=t + 100)
    _touch(D / "client_report.json", b"{}", t=t + 200)
    for i, n in enumerate(("full_20260928_125211", "full_20260928_131131", "miaplpy_20260929_083016",
                           "miaplpy_20260929_101426")):
        _touch(P / "logs" / f"{n}.log", t=t + i)
    _touch(P / "slc" / "S1A_X.SAFE" / "manifest.safe")
    _touch(P / "slc" / "dropped_short_dates.json", b'{"20180601": "short"}')
    _touch(P / "slc" / "frame_footprint.json", b"{}")
    _touch(P / "miaplpy" / "inputs" / "slcStack.h5")
    _touch(P / "miaplpy" / "inverted" / "phase_series.h5")
    _touch(P / "miaplpy" / "inverted" / "linking_fingerprint.json", b"{}")
    _touch(P / "stack" / "merged" / "interferograms" / "20200101_20200113" / "filt_fine.unw")
    _touch(P / "stack" / "SAFE_files.txt", b"S1A_X.SAFE\n")
    _touch(P / "orbits" / "o.EOF")
    _touch(D / "measurement_points_egms.csv")
    _touch(D / "releases" / "20261005_1200" / "client" / "01_report.html")
    _touch(D / "releases" / "20261005_1200" / "data" / "measurement_points_egms.csv")
    _touch(D / "screening_not_for_clients" / "20261001_000000" / "client_report.json")
    return cfg, cf


def _cleanup(*argv):
    from insar_pipeline.scripts import cleanup
    return cleanup.main([str(a) for a in argv])


# ---------------------------------------------------------------- stackrunner-6 / usability-12
def test_cleanup_refuses_while_a_run_is_going_even_with_force(tmp_path, capsys):
    """Old code: no runs.state check - rmtree under a live run."""
    cfg, cf = _finished_project(tmp_path)
    P = cfg.project_dir
    (P / runs.RUNFILE).write_text(json.dumps({"pid": os.getpid(), "start_ticks": procs.start_ticks(os.getpid()),
                                              "kind": "miaplpy", "started": time.time()}))
    assert runs.state(P) == "running"
    assert _cleanup(cf, "--level", "2", "--delete", "--force", "--yes") == 2
    assert "REFUSED" in capsys.readouterr().out
    assert (P / "miaplpy" / "inputs" / "slcStack.h5").exists() and (P / "slc").exists()
    assert (cfg.mintpy_dir / "timeseries.h5").exists()           # not even level 1


def test_cleanup_refuses_when_another_launcher_holds_the_lock(tmp_path):
    cfg, cf = _finished_project(tmp_path)
    p = _run_py(f"""
        import time
        from insar_pipeline import runs
        with runs.project_lock({str(cfg.project_dir)!r}, kind="full"):
            print("held", flush=True); time.sleep(30)
    """)
    try:
        assert p.stdout.readline().strip() == "held"
        assert _cleanup(cf, "--level", "1", "--delete") == 2
        assert (cfg.mintpy_dir / "timeseries.h5").exists()
    finally:
        os.killpg(p.pid, signal.SIGKILL); p.wait()


@pytest.mark.parametrize("damage", ["failed_run", "no_report", "stale_qa", "no_velocity"])
def test_cleanup_level2_needs_a_finished_project(tmp_path, capsys, damage):
    """Old code: level 2 deleted the SLC stack of a failed/unfinished run."""
    cfg, cf = _finished_project(tmp_path)
    P = cfg.project_dir
    if damage == "failed_run":
        (P / runs.RESULTFILE).write_text(json.dumps({"status": "failed", "stage": "mintpy",
                                                     "finished": time.time(), "pid": 1}))
    elif damage == "no_report":
        (P / "deliverables" / "client_report.json").unlink()
    elif damage == "stale_qa":
        os.utime(P / "quality_assessment.json", (time.time() - 50_000,) * 2)
    else:
        for f in (cfg.mintpy_dir / "velocity.h5", cfg.mintpy_dir / "geo" / "geo_velocity.h5"):
            f.unlink()
    assert _cleanup(cf, "--level", "2", "--delete", "--yes") == 2
    out = capsys.readouterr().out
    assert "not finished" in out and "REFUSED" in out
    assert (P / "miaplpy" / "inputs" / "slcStack.h5").exists() and (P / "slc" / "S1A_X.SAFE").exists()
    # --force is the documented override for this check (not for a live run)
    assert _cleanup(cf, "--level", "2", "--delete", "--yes", "--force") == 0
    assert not (P / "miaplpy" / "inputs").exists()


def test_cleanup_asks_for_yes_at_a_terminal(tmp_path, monkeypatch):
    cfg, cf = _finished_project(tmp_path)

    class Tty:
        def isatty(self):
            return True
    monkeypatch.setattr(sys, "stdin", Tty())
    monkeypatch.setattr("builtins.input", lambda *_: "y")      # anything but a typed 'yes'
    assert _cleanup(cf, "--level", "2", "--delete") == 1
    assert (cfg.project_dir / "slc").exists()


def test_cleanup_level2_on_a_finished_project_keeps_records_and_holds_the_lock(tmp_path, monkeypatch):
    from insar_pipeline.scripts import cleanup
    cfg, cf = _finished_project(tmp_path)
    P = cfg.project_dir
    seen = []
    real = cleanup.delete

    def spy(c, items):
        seen.append(runs.lock_holder(P))
        return real(c, items)
    monkeypatch.setattr(cleanup, "delete", spy)
    assert _cleanup(cf, "--level", "2", "--delete", "--yes") == 0
    assert seen and seen[0] and seen[0].get("kind") == "cleanup"     # no run could start meanwhile
    for d in ("miaplpy/inputs", "miaplpy/inverted", "stack/merged", "slc", "orbits"):
        assert not (P / d).exists(), d
    rec = P / "logs" / "records"
    assert json.loads((rec / "slc" / "dropped_short_dates.json").read_text()) == {"20180601": "short"}
    assert (rec / "slc" / "frame_footprint.json").exists()
    assert (rec / "miaplpy__inverted" / "linking_fingerprint.json").exists()
    assert (rec / "stack__SAFE_files.txt").read_text() == "S1A_X.SAFE\n"
    assert runs.lock_holder(P) is None


def test_cleanup_keeps_a_folder_whose_records_cannot_be_copied(tmp_path, monkeypatch):
    from insar_pipeline.scripts import cleanup
    cfg, cf = _finished_project(tmp_path)
    monkeypatch.setattr(cleanup, "save_records", lambda c, d: (d.name != "slc", []))
    assert _cleanup(cf, "--level", "2", "--delete", "--yes") == 1        # reported, not 'done'
    assert (cfg.project_dir / "slc" / "dropped_short_dates.json").exists()
    assert not (cfg.project_dir / "orbits").exists()


# ---------------------------------------------------------------- tests-9 / stackrunner-7
@pytest.mark.parametrize("chain", [("ramp", "demErr"), ("ERA5", "demErr"), ("ERA5", "ramp", "demErr")])
def test_cleanup_keeps_the_final_series_of_the_chain_that_ran(tmp_path, chain):
    """Old code kept only timeseries_ERA5_ramp_demErr.h5 and listed any other
    chain's final series as 'intermediate time series'."""
    from insar_pipeline.scripts import cleanup
    cfg, _ = _finished_project(tmp_path, chain=chain)
    final = "timeseries_" + "_".join(chain) + ".h5"
    tg = {p.name for _, p, _ in cleanup.targets(cfg, 1)}
    assert final not in tg
    assert "timeseriesResidual.h5" in tg                          # real intermediates still go
    # timeseries.h5 too - unless it is the deramp's input (chain ramp, demErr):
    # that pair is kept until the QA has measured the removed tilt (INT1,
    # methodology-14: cleanup.ramp_pair_needed)
    assert ("timeseries.h5" in tg) == (chain[0] != "ramp")


def test_cleanup_keeps_the_newest_chain_over_a_stale_longer_name(tmp_path):
    """An older run's timeseries_ERA5_ramp_demErr.h5 must not make the current
    final timeseries_ramp_demErr.h5 look intermediate (run_products.json and
    the newest file name it)."""
    from insar_pipeline.scripts import cleanup
    cfg, _ = _finished_project(tmp_path)
    M = cfg.mintpy_dir
    _touch(M / "timeseries_ERA5_ramp_demErr.h5", t=time.time() - 90_000)
    tg = {p.name for _, p, _ in cleanup.targets(cfg, 1)}
    assert "timeseries_ramp_demErr.h5" not in tg
    (M / "run_products.json").write_text(json.dumps({"timeseries_radar": str(M / "timeseries_ramp_demErr.h5")}))
    assert "timeseries_ramp_demErr.h5" not in {p.name for _, p, _ in cleanup.targets(cfg, 1)}


# ---------------------------------------------------------------- techout-12 / stackrunner-7
def test_cleanup_never_touches_logs_releases_or_set_aside_client_files(tmp_path):
    """Old code deleted every log but the newest at level 1 (Site B: both
    coregistration logs) - and archive_project said logs are never deleted."""
    from insar_pipeline.scripts import cleanup
    cfg, cf = _finished_project(tmp_path)
    P, D = cfg.project_dir, cfg.project_dir / "deliverables"
    tg = [p for _, p, _ in cleanup.targets(cfg, 3)]
    assert not [p for p in tg if "logs" in p.parts or "releases" in p.parts]
    assert cleanup.protected(cfg, D / "releases" / "20261005_1200" / "data" / "measurement_points_egms.csv")
    assert cleanup.protected(cfg, P / "logs" / "full_20260928_131131.log")
    assert cleanup.protected(cfg, Path("/etc/passwd")) and cleanup.protected(cfg, P)
    assert _cleanup(cf, "--level", "3", "--delete", "--yes") == 0
    assert len(list((P / "logs").glob("*.log"))) == 4
    assert (D / "releases" / "20261005_1200" / "client" / "01_report.html").exists()
    assert (D / "releases" / "20261005_1200" / "data" / "measurement_points_egms.csv").exists()
    assert (D / "screening_not_for_clients" / "20261001_000000" / "client_report.json").exists()
    assert not (D / "measurement_points_egms.csv").exists()
    assert (D / "client_report.json").exists() and (P / "quality_assessment.json").exists()


# ---------------------------------------------------------------- stackrunner-16: archive_project, plan_run
def test_archive_project_is_an_alias_of_cleanup(tmp_path, monkeypatch, capsys):
    """Old archive_project found nothing on a burst2safe/phase-linked project
    (it globbed slc/*.zip and the retired pairs_dir)."""
    from insar_pipeline import archive_project
    from insar_pipeline.scripts import cleanup
    assert archive_project.translate(["c.yaml", "--level", "slc", "--apply"]) == ["c.yaml", "--level", "2", "--delete"]
    assert archive_project.translate(["c.yaml", "--level=scratch"]) == ["c.yaml", "--level", "1"]
    cfg, cf = _finished_project(tmp_path)
    assert archive_project.main([str(cf), "--level", "slc"]) == 0             # dry run
    out = capsys.readouterr().out
    assert "S1A" not in out and str(cfg.project_dir / "slc") in out and "dry run" in out
    assert (cfg.project_dir / "slc").exists()
    called = []
    monkeypatch.setattr(cleanup, "main", lambda a: called.append(a) or 0)
    archive_project.main([str(cf), "--level", "all", "--apply"])
    assert called == [[str(cf), "--level", "2", "--delete"]]


def test_plan_run_sizes_a_miaplpy_config_with_the_pipeline_plan(tmp_path, monkeypatch, capsys):
    """Old plan_run told a miaplpy config 'stack -> supported, re-run with backend: stack'."""
    from insar_pipeline import plan_run
    cf = _yaml(tmp_path, "miaplpy")
    seen = []
    monkeypatch.setattr(pipeline, "run_from", lambda cfg, start, **o: seen.append((cfg.backend, start, o)) or 0)
    assert plan_run.main([str(cf)]) == 0
    assert seen and seen[0][0] == "miaplpy" and seen[0][1] == "full" and seen[0][2]["plan_only"] is True
    assert "backend: stack" not in capsys.readouterr().out
    monkeypatch.setattr(pipeline, "run_from", lambda cfg, start, **o: 1)     # the disk gate refused
    assert plan_run.main([str(cf)]) == 1
    assert plan_run.main([]) == 2


# ---------------------------------------------------------------- generality-17 / tests-8 / ops-12: thin launchers
SCRIPTS = {"run_pipeline": None, "run_mintpy_only": "mintpy", "run_miaplpy_only": "miaplpy",
           "rebuild_products": "products"}


@pytest.mark.parametrize("name,start", SCRIPTS.items())
def test_runner_scripts_require_a_config_and_go_through_the_pipeline(name, start, monkeypatch, capsys):
    """Old scripts defaulted to config_site_a_p2.yaml (gone) and copied the
    stage chain: bare export_all, no events, exit 0 when a stage failed."""
    import importlib
    mod = importlib.import_module(f"insar_pipeline.scripts.{name}")
    seen = []

    def fake_main(argv):
        seen.append(list(argv))
    monkeypatch.setattr(pipeline, "main", fake_main)
    monkeypatch.setenv("EARTHDATA_USERNAME", "u")
    monkeypatch.setenv("EARTHDATA_PASSWORD", "p")
    assert mod.main([]) == 2 and not seen
    assert mod.main(["--force"]) == 2 and not seen
    assert mod.main(["x.yaml", "--no-notify"]) == 0
    assert seen[-1] == (["x.yaml", "--no-notify"] if start is None else ["x.yaml", "--start", start, "--no-notify"])
    if name == "run_miaplpy_only":
        mod.main(["x.yaml", "--from-products"])
        assert seen[-1] == ["x.yaml", "--start", "products"]

    def failing(argv):
        raise SystemExit(3)
    monkeypatch.setattr(pipeline, "main", failing)
    assert mod.main(["x.yaml"]) == 3


def test_no_launcher_copies_the_stage_chain():
    """One function = one order: the scripts and the menu call the pipeline,
    never the stage modules directly (cli ran figures before panel_history)."""
    import re
    root = Path(pipeline.__file__).resolve().parent
    files = [root / "cli.py"] + [root / "scripts" / f"{n}.py" for n in SCRIPTS]
    bad = re.compile(r"export_all|make_figures|panel_history\b|insar_pipeline\.figures|run_mintpy_with_loader|"
                     r"miaplpy_backend|run_client_report\(|quality_assessment\.evaluate|config_site_a")
    for f in files:
        hits = [ln for ln in f.read_text().splitlines() if bad.search(ln) and not ln.lstrip().startswith("#")
                and "panel_history.json" not in ln and "panel and site histories" not in ln]
        assert not hits, (f.name, hits)


def test_rebuild_products_exits_nonzero_when_a_stage_fails(tmp_path, monkeypatch):
    """End to end through the real launch(): a failed figures stage is exit 1
    and named in run_result.json (the old script exited 0)."""
    from insar_pipeline.scripts import rebuild_products
    cf = _yaml(tmp_path, "stack")
    calls = []
    _stub_products(monkeypatch, calls, fail=("figures",))
    real = pipeline.launch
    monkeypatch.setattr(pipeline, "launch", lambda p, k, **o: real(p, k, watchdog=False, **o))
    code = rebuild_products.main([str(cf), "--no-notify"])
    assert code == 1
    assert calls == ["deliverables", "quality_assessment", "figures", "client_report"]
    res = runs.result(tmp_path / "proj")
    assert res["status"] == "failed" and res["failed_stages"] == ["figures"] and res["kind"] == "products"



# ---------------------------------------------------------------- tests-8 / usability-11: the old menu
def test_menu_figures_go_through_pipeline_run_figures_under_the_lock(tmp_path, monkeypatch):
    from insar_pipeline import cli
    cf = _yaml(tmp_path, "miaplpy")
    P = tmp_path / "proj"
    P.mkdir()
    seen = []
    monkeypatch.setattr(pipeline, "run_figures", lambda cfg: seen.append(runs.lock_holder(P)) or False)
    assert cli.stage_main("figures", str(cf)) == 1                  # a failed stage is not exit 0
    assert seen and seen[0] and seen[0]["kind"] == "figures"
    monkeypatch.setattr(pipeline, "run_quality_assessment", lambda cfg: True)
    assert cli.stage_main("quality_assessment", str(cf)) == 0
    with runs.project_lock(P, kind="full"):
        p = subprocess.run([sys.executable, "-m", "insar_pipeline.cli", "--stage", "figures", str(cf)],
                           env={**os.environ, "PYTHONPATH": PARENT}, capture_output=True, text=True, timeout=60)
    assert p.returncode == 3 and "already running" in p.stdout


def test_menu_starts_and_stops_through_runs(tmp_path, monkeypatch):
    """Old menu: its own PID job list in ~/.insar_pipeline_cli.json, the
    scripts/ runners, os.killpg by bare PID."""
    from insar_pipeline import cli
    from insar_pipeline.config import load_config
    cf = _yaml(tmp_path, "miaplpy")
    cfg = load_config(str(cf))
    monkeypatch.setattr(cli, "STATE", tmp_path / "cli.json")
    (tmp_path / "cli.json").write_text(json.dumps({"config": str(cf), "jobs": [{"pid": 1}]}))
    started, stopped = [], []
    monkeypatch.setattr(runs, "start", lambda c, p, k, extra=None: started.append((c, Path(p), k)) or
                        {"pid": 42, "log": "x.log"})
    monkeypatch.setattr(runs, "stop", lambda p, grace_s=20.0: stopped.append(p) or "stopped")
    monkeypatch.setattr("builtins.input", lambda *_: "y")
    for act, kind in ((cli.act_full, "full"), (cli.act_mintpy, "mintpy"), (cli.act_miaplpy, "miaplpy")):
        act(cfg, str(cf))
        assert started[-1] == (str(cf.resolve()), cfg.project_dir, kind)
    monkeypatch.setattr(runs, "suggest_kind", lambda P, c=None: ("products", "velocity is current"))
    cli.act_resume(cfg, str(cf))
    assert started[-1][2] == "resume"
    cli.save_state(cli.load_state())
    assert "jobs" not in json.loads((tmp_path / "cli.json").read_text())
    monkeypatch.setattr(runs, "state", lambda P: "running")
    monkeypatch.setattr(runs, "record", lambda P: {"pid": 42, "kind": "full", "start_ticks": 1})
    monkeypatch.setattr(runs, "alive", lambda r: True)
    cli.act_stop(cfg, str(cf))
    assert stopped == [cfg.project_dir]
    src = Path(cli.__file__).read_text()
    assert "killpg" not in src and "download config.yaml and project_zone.geojson" not in src
