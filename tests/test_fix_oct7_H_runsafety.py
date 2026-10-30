"""Run safety (review 7 Oct 2026, group H): the restart a half-finished
MintPy run gets, an interrupted re-unwrap, quarantined stack pairs, sizing
twice, Stop while the dashboard rebuilds the report, and the client package
while a run is going. Synthetic projects only, no network, no real runs."""
import json
import os
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from insar_pipeline import dashboard as dash, progress, release, runs

PARENT = str(Path(__file__).resolve().parents[2])
T0 = time.time() - 3 * 86400            # after runs.RUN_RECORD_SINCE (5 Oct 2026)


@pytest.fixture(autouse=True)
def _quiet(monkeypatch, tmp_path):
    monkeypatch.setenv("INSAR_NOTIFY", "off")
    monkeypatch.setattr(runs, "REGISTRY", tmp_path / "reg.json")


def _touch(f: Path, t: float, data: str = "") -> Path:
    f.parent.mkdir(parents=True, exist_ok=True)
    if not f.exists():
        f.write_text(data)
    os.utime(f, (t, t))
    return f


def _project(P: Path, pairs, backend="miaplpy", t_unw=T0, t_ifg=T0 + 100, t_vel=T0 + 200,
             t_geo=None, t_record=None, t_report=None, retired=False):
    """A stack / phase-linked project at the MintPy end: every pair exported
    and unwrapped at t_unw, ifgramStack at t_ifg, radar velocity at t_vel; the
    geo velocity, run_products.json and client report only when given."""
    S, M = P / "stack", P / "mintpy"
    for p in pairs:
        d = S / "merged" / "interferograms" / p
        _touch(d / "filt_fine.int", t_unw)
        _touch(d / "filt_fine.unw", t_unw)
        _touch(d / "filt_fine.unw.conncomp", t_unw)
    if backend == "miaplpy":
        _touch(P / "miaplpy" / "inverted" / "phase_series.h5", t_unw - 100)
    _touch(M / "inputs" / "ifgramStack.h5", t_ifg)
    if t_vel:
        _touch(M / "velocity.h5", t_vel)
    if t_geo:
        _touch(M / "geo" / "geo_velocity.h5", t_geo)
    if t_record:
        _touch(M / "run_products.json", t_record, "{}")
    if t_report:
        _touch(P / "deliverables" / "client_report.json", t_report, "{}")
    if retired:
        _touch(M / "superseded" / "run_20261003_101010" / "RETIRED.json", t_vel - 50, "{}")
    return SimpleNamespace(backend=backend, stack_dir=S, mintpy_dir=M)


PAIRS = [f"2017{m:02d}01_2017{m + 1:02d}01" for m in range(1, 9)]


# ---------------------------------------------------------------- 35
def test_mintpy_that_died_after_the_velocity_is_not_finished(tmp_path):
    # (a) velocity.h5 written, then the run died before geocoding: no geo/, no record
    c = _project(tmp_path / "a", PAIRS)
    k, why = runs.suggest_kind(tmp_path / "a", c)
    assert k == "mintpy" and "MintPy did not finish" in why
    # (b) geocoded too, but the run raised in its own checks: the earlier run's
    # products were retired and no record was written
    c = _project(tmp_path / "b", PAIRS, t_geo=T0 + 300, retired=True)
    assert runs.suggest_kind(tmp_path / "b", c)[0] == "mintpy"
    # (c) the record written after the products: finished
    c = _project(tmp_path / "c", PAIRS, t_geo=T0 + 300, t_record=T0 + 400, retired=True)
    assert runs.suggest_kind(tmp_path / "c", c) == ("products", "MintPy finished; the products and report need "
                                                                 "(re)building")
    _touch(tmp_path / "c" / "deliverables" / "client_report.json", T0 + 500)
    assert runs.suggest_kind(tmp_path / "c", c)[0] == "products"
    # (d) a run from before the records (Site B, site_a_pl): velocity before
    # 5 Oct 2026, geo after it, no record, nothing retired
    old = runs.RUN_RECORD_SINCE - 7 * 86400
    c = _project(tmp_path / "d", PAIRS, t_unw=old - 300, t_ifg=old - 200, t_vel=old, t_geo=old + 14)
    assert runs.suggest_kind(tmp_path / "d", c)[0] == "products"
    # ... but not when its geocode never ran
    c = _project(tmp_path / "e", PAIRS, t_unw=old - 300, t_ifg=old - 200, t_vel=old)
    assert runs.suggest_kind(tmp_path / "e", c)[0] == "mintpy"
    # the stack route reads the same way
    c = _project(tmp_path / "f", PAIRS, backend="stack")
    assert runs.suggest_kind(tmp_path / "f", c)[0] == "mintpy"


def test_the_run_incomplete_marker_means_mintpy(tmp_path):
    """Shared contract 1: time_series marks a MintPy run unfinished
    (mintpy/run_incomplete.json) until its record is written."""
    P = tmp_path / "p"
    c = _project(P, PAIRS, t_geo=T0 + 300, t_record=T0 + 400, t_report=T0 + 500)
    assert runs.suggest_kind(P, c)[0] == "products"
    _touch(P / "mintpy" / "run_incomplete.json", T0 + 600, json.dumps({"stamp": "x", "started": T0 + 600}))
    k, why = runs.suggest_kind(P, c)
    assert k == "mintpy" and "did not finish" in why
    # pairs still to unwrap: the unwrapping restart, never 'Products only'
    (P / "stack" / "merged" / "interferograms" / PAIRS[0] / "filt_fine.unw.conncomp").unlink()
    assert runs.suggest_kind(P, c)[0] == "miaplpy"
    # a route that has no 'MintPy onward' restarts whole
    Q = tmp_path / "q"
    _touch(Q / "mintpy" / "velocity.h5", T0)
    _touch(Q / "mintpy" / "run_incomplete.json", T0 + 10, "{}")
    c = SimpleNamespace(backend="hyp3", stack_dir=Q / "stack", mintpy_dir=Q / "mintpy")
    assert runs.suggest_kind(Q, c)[0] == "full"


# ---------------------------------------------------------------- 36
def test_an_interrupted_reunwrap_is_not_everything_finished(tmp_path):
    pairs = [f"{2016 + i // 120}{(i // 10) % 12 + 1:02d}{i % 10 + 10}_x{i:03d}" for i in range(240)]
    pairs = sorted(pairs)
    P = tmp_path / "p"
    c = _project(P, pairs, t_geo=T0 + 300, t_record=T0 + 400, t_report=T0 + 500)
    assert runs.suggest_kind(P, c)[0] == "products"
    now = time.time()
    for p in pairs[:60]:                 # the re-unwrap redid the first 60 by name, then stopped
        os.utime(P / "stack" / "merged" / "interferograms" / p / "filt_fine.unw", (now, now))
    assert runs.suggest_kind(P, c)[0] != "products"


# ---------------------------------------------------------------- 40
def test_quarantined_stack_pairs_do_not_hold_a_finished_project_back(tmp_path):
    P = tmp_path / "p"
    c = _project(P, PAIRS[:5], backend="stack", t_geo=T0 + 300, t_record=T0 + 400, t_report=T0 + 500)
    bad = PAIRS[5]
    _touch(P / "stack" / "merged" / "interferograms" / bad / "filt_fine.int", T0)
    _touch(P / "stack" / "quarantine_unwrap_timeout" / bad / "REASON.txt", T0 + 10, "timeout")
    assert runs.suggest_kind(P, c)[0] == "products"
    # quarantined after the last MintPy run: that run did not see it, so not finished
    _touch(P / "stack" / "quarantine_unwrap_timeout" / bad / "REASON.txt", T0 + 900)
    assert runs.suggest_kind(P, c)[0] == "full"
    # the phase-linked route never tolerates a failed pair
    Q = tmp_path / "q"
    c = _project(Q, PAIRS[:5], t_geo=T0 + 300, t_record=T0 + 400, t_report=T0 + 500)
    _touch(Q / "stack" / "merged" / "interferograms" / bad / "filt_fine.int", T0)
    _touch(Q / "stack" / "quarantine_unwrap_timeout" / bad / "REASON.txt", T0 + 10, "timeout")
    assert runs.suggest_kind(Q, c)[0] == "miaplpy"


# ---------------------------------------------------------------- 39
def _plan(P, cfgp):
    rec = runs.start(str(cfgp), P, "plan")
    t = time.time()
    while runs.alive(rec) and time.time() - t < 20:
        time.sleep(0.05)
    assert not runs.alive(rec)
    return rec


def test_sizing_twice_keeps_a_finished_project_done(tmp_path, monkeypatch):
    monkeypatch.setitem(runs.KINDS, "plan", ("test", ["-c", "pass"]))
    cfgp = tmp_path / "x.yaml"; cfgp.write_text("")
    P = tmp_path / "proj"; P.mkdir()
    (P / runs.RUNFILE).write_text(json.dumps({"pid": 999999999, "kind": "full", "started": time.time() - 50}))
    # the run ended a while before the sizing
    progress.path_for(P).write_text(json.dumps({"t": time.time() - 30, "stage": "pipeline", "status": "done"}) + "\n")
    assert runs.state(P) == "done"
    _plan(P, cfgp)
    assert runs.state(P) == "done"
    _plan(P, cfgp)
    assert runs.state(P) == "done"
    assert runs.record(P)["previous"]["kind"] == "full"
    # a run started from a terminal after the sizing keeps the real run, not the plan
    rec = runs.claim_runfile(P, "products")
    assert rec["previous"]["kind"] == "full"


def test_sizing_a_new_project_twice_is_still_idle(tmp_path, monkeypatch):
    monkeypatch.setitem(runs.KINDS, "plan", ("test", ["-c", "pass"]))
    cfgp = tmp_path / "x.yaml"; cfgp.write_text("")
    P = tmp_path / "proj"; P.mkdir()
    _plan(P, cfgp)
    assert runs.state(P) == "idle"
    _plan(P, cfgp)
    assert runs.state(P) == "idle"


def test_old_nested_plan_records_still_read_by_the_real_run(tmp_path):
    """A run.json written by the old start(): plan -> plan, the real run lost;
    the run's own result decides."""
    P = tmp_path / "proj"; P.mkdir()
    t = time.time()
    (P / runs.RESULTFILE).write_text(json.dumps({"status": "done", "finished": t - 100, "pid": 999999997}))
    (P / runs.RUNFILE).write_text(json.dumps({"pid": 999999998, "kind": "plan", "started": t - 10,
                                             "previous": {"pid": 999999996, "kind": "plan", "started": t - 50}}))
    progress.emit(P, "run", "start", msg="plan")
    assert runs.state(P) == "done"


# ---------------------------------------------------------------- 38
def _run_py(code: str) -> subprocess.CompletedProcess:
    env = {**os.environ, "PYTHONPATH": PARENT + os.pathsep + os.environ.get("PYTHONPATH", "")}
    return subprocess.run([sys.executable, "-c", textwrap.dedent(code)], capture_output=True, text=True,
                          env=env, timeout=60, start_new_session=True)


def test_stop_does_not_signal_the_process_that_asks(tmp_path):
    P = tmp_path / "proj"; P.mkdir()
    (P / runs.RUNFILE).write_text(json.dumps({"pid": 999999999, "kind": "full", "started": time.time() - 50}))
    r = _run_py(f"""
        from insar_pipeline import runs
        with runs.project_lock({str(P)!r}, kind="report"):
            print("state:", runs.state({str(P)!r}), flush=True)
            print("stop:", runs.stop({str(P)!r}, grace_s=1), flush=True)
        print("survived", flush=True)
    """)
    assert r.returncode == 0, (r.returncode, r.stdout, r.stderr)
    assert "state: running" in r.stdout and "nothing to stop" in r.stdout and "survived" in r.stdout


def test_dashboard_stop_key_says_the_dashboard_is_busy(tmp_path, monkeypatch):
    seen = {}
    fake = SimpleNamespace(cfg=SimpleNamespace(project_dir=tmp_path), busy="building the client package…",
                           notify=lambda msg, **kw: seen.setdefault("msg", msg),
                           app=SimpleNamespace(push_screen=lambda *a, **k: pytest.fail("asked to stop itself")))
    monkeypatch.setattr(runs, "state", lambda P: "running")
    monkeypatch.setattr(runs, "lock_holder", lambda P: {"pid": os.getpid(), "kind": "package"})
    dash.RunScreen.action_stop(fake)
    assert "No run to stop" in seen["msg"] and "building the client package" in seen["msg"]


# ---------------------------------------------------------------- 37
def _cfg(tmp_path):
    from insar_pipeline.config import load_config
    f = tmp_path / "c.yaml"
    f.write_text(f"project_dir: {tmp_path / 'proj'}\nbbox: [-87.7, 38.2, -87.6, 38.3]\n"
                 "start_date: '2020-01-01'\nend_date: '2021-01-01'\n")
    cfg = load_config(str(f))
    cfg.project_dir.mkdir(parents=True, exist_ok=True)
    return cfg


def test_no_client_package_while_a_run_is_going(tmp_path, monkeypatch):
    from insar_pipeline import package
    cfg = _cfg(tmp_path)
    P = cfg.project_dir
    monkeypatch.setattr(release, "status", lambda c, rep=None, qa=None: {"final": True, "blockers": []})
    monkeypatch.setattr(package, "build", lambda c, force_draft=False: pytest.fail("packaged during a run"))
    env = {**os.environ, "PYTHONPATH": PARENT + os.pathsep + os.environ.get("PYTHONPATH", "")}
    holder = subprocess.Popen([sys.executable, "-c", textwrap.dedent(f"""
        import time
        from insar_pipeline import runs
        with runs.project_lock({str(P)!r}, kind="products"):
            print("held", flush=True)
            time.sleep(30)
    """)], stdout=subprocess.PIPE, text=True, env=env, start_new_session=True)
    try:
        assert holder.stdout.readline().strip() == "held"
        path, msg = dash.build_package(cfg)
        assert path is None and "run is going" in msg
        assert not (P / "deliverables" / "releases").exists()
        # the key refuses up front, before any worker starts
        seen = {}
        fake = SimpleNamespace(cfg=cfg, notify=lambda m, **kw: seen.setdefault("msg", m),
                               _package=lambda: pytest.fail("package worker started during a run"))
        dash.RunScreen.action_package(fake)
        assert "Wait for the run to finish" in seen["msg"]
    finally:
        os.killpg(holder.pid, signal.SIGKILL)
        holder.wait()
