"""
22 Oct 2026: the ERA5 prefetch kept a finished run's process alive.

A phase-linked run stopped with --to-step merge_reference_secondary_slc
logged "Stopped after merge_reference_secondary_slc as requested" and wrote
run_result.json "done", then stayed alive for ten minutes until it was
stopped by hand: the prefetch's worker threads (a ThreadPoolExecutor, which
Python joins at exit) were waiting on CDS requests queued for hours. Its
era5_prefetch.json, written only at the end, was never written, so the 265
failures of 270 had no recorded reason. Now:

- the fetch workers are daemon threads, so a pending fetch cannot hold the
  process (a subprocess must exit with a fetch that never returns);
- the end of every run cancels its project's prefetch, which records
  'cancelled' with what was done and what was left;
- a --to-step run does not start the prefetch (the run that carries on to
  MintPy starts its own);
- era5_prefetch.json is rewritten as each fetch starts and ends: the
  failures with their reasons (without the per-request URLs and ids), a
  count per reason, and the fetches in flight since when.

Each test fails on the code at 7b97c4f. No network: the fetch function is
injected; synthetic projects in tmp_path only.
"""
import json
import subprocess
import sys
import textwrap
import threading
import time
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from insar_pipeline import pipeline, progress                  # noqa: E402
from insar_pipeline import stack_backend as sb                 # noqa: E402

# what the CDS answered 265 times on 22 Oct 2026 (job ids made up)
QUEUE_FULL = ("FAILED 400 Client Error: Bad Request for url: https://cds.climate.copernicus.eu/api/retrieve/"
              "v1/jobs/14ba0f6d-4161-4f84-94fa-0b97747f64ac/results\nThe job has been rejected\nMaximum number "
              "queued requests for this dataset has been reached. Please configure your scripts to request "
              "data sequentially")
# a failure that is not the queue (made up, in the CDS's shape)
SERVER_ERR = ("FAILED 500 Server Error: Internal Server Error for url: https://cds.climate.copernicus.eu/api/"
              "retrieve/v1/jobs/14ba0f6d-4161-4f84-94fa-0b97747f64ac/results\nThe job failed\nAn internal "
              "error occurred while processing the request")
DATES = ["20200101", "20200113", "20200125", "20200206"]


def fake_plan(P: Path, dates=DATES) -> dict:
    d = P / "mintpy" / "ERA5"
    files = {dt: str(d / sb.era5_grib_name((30, 50, -90, -80), dt, "23")) for dt in dates}
    return {"ok": True, "reason": "", "dir": str(d), "snwe": [30, 50, -90, -80], "hour": "23",
            "dates": list(dates), "needed": len(dates), "present": 0, "missing": list(dates), "files": files}


def fake_cfg(tmp_path) -> types.SimpleNamespace:
    P = tmp_path / "proj"
    P.mkdir(parents=True, exist_ok=True)
    return types.SimpleNamespace(project_dir=P, mintpy_dir=P / "mintpy", mintpy_tropo_method="pyaps",
                                 backend="miaplpy")


@pytest.fixture
def offline(monkeypatch):
    """No credential check, a fixed plan; the fetches the test injects."""
    monkeypatch.setattr(sb, "check_weather_credentials", lambda c: [])
    monkeypatch.setattr(sb, "era5_prefetch_plan", lambda c: fake_plan(Path(c.project_dir)))


@pytest.fixture
def gate():
    """A fetch that never returns until the test ends (a CDS request queued
    for hours); released at teardown so no thread outlives the test."""
    ev = threading.Event()
    yield ev
    ev.set()


def status_file(cfg) -> dict:
    try:
        return json.loads((cfg.mintpy_dir / sb.ERA5_PREFETCH_STATUS).read_text())
    except (OSError, ValueError):
        return {}


def wait_for(pred, timeout=10.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        v = pred()
        if v:
            return v
        time.sleep(0.05)
    return pred()


def era5_events(cfg) -> list:
    return [e for e in progress.read(cfg.project_dir) if e["stage"] == "era5_prefetch"]


# ------------------------------------------------------------------ the process exits

def test_the_process_exits_while_fetches_are_pending(tmp_path):
    """The run's main work is over and nothing cancels the prefetch: the
    process must still exit, though every fetch is still waiting."""
    root = Path(pipeline.__file__).parent.parent         # the package this test imported
    P = tmp_path / "proj"
    child = textwrap.dedent(f"""
        import json, sys, threading, time, types
        from pathlib import Path
        sys.path.insert(0, {str(root)!r})
        from insar_pipeline import pipeline, stack_backend as sb
        P = Path({str(P)!r}); P.mkdir(parents=True, exist_ok=True)
        cfg = types.SimpleNamespace(project_dir=P, mintpy_dir=P / "mintpy", mintpy_tropo_method="pyaps",
                                    backend="miaplpy")
        d = P / "mintpy" / "ERA5"
        dates = {DATES!r}
        sb.check_weather_credentials = lambda c: []
        sb.era5_prefetch_plan = lambda c: {{"ok": True, "reason": "", "dir": str(d), "snwe": [30, 50, -90, -80],
            "hour": "23", "dates": dates, "needed": len(dates), "present": 0, "missing": dates,
            "files": {{x: str(d / ("ERA5_" + x + ".grb")) for x in dates}}}}
        started = []
        def never(date, path, snwe, hour):
            started.append(date)
            threading.Event().wait()            # a CDS queue that never moves
        sb.fetch_era5_grib = never
        pipeline.start_era5_prefetch(cfg)
        t0 = time.time()
        while len(started) < 4 and time.time() - t0 < 10:
            time.sleep(0.05)
        print("pending", len(started), flush=True)
    """)
    t0 = time.time()
    try:
        out = subprocess.run([sys.executable, "-c", child], capture_output=True, text=True, timeout=30)
    except subprocess.TimeoutExpired:
        pytest.fail("the process stayed alive for 30 s after its main work ended (the prefetch held it)")
    assert out.returncode == 0, out.stderr
    assert "pending 4" in out.stdout                     # every worker was inside a fetch
    assert time.time() - t0 < 25


# ------------------------------------------------------------------ cancel

def test_cancel_ends_the_prefetch_and_records_what_was_left(tmp_path, offline, gate):
    cfg = fake_cfg(tmp_path)
    seen = []

    def slow(date, path, snwe, hour):
        seen.append(date)
        if date == "20200206":                          # newest first: the first one fails at once
            return SERVER_ERR
        gate.wait()
        return "ok"
    cancel = threading.Event()
    box = {}
    th = threading.Thread(target=lambda: box.update(st=sb.start_era5_prefetch(cfg, workers=2, fetch=slow,
                                                                               cancel=cancel)), daemon=True)
    th.start()
    assert wait_for(lambda: len(status_file(cfg).get("in_flight", {})) == 2)
    cancel.set()
    th.join(5)
    assert not th.is_alive(), "start_era5_prefetch waited for the pending fetches after it was cancelled"
    st = box["st"]
    assert st["status"] == "cancelled" and "ended before" in st["reason"]
    assert sorted(st["in_flight"]) == ["20200113", "20200125"] and st["waiting"] == 1
    assert list(st["failed"]) == ["20200206"] and st["fetched"] == 0
    assert status_file(cfg)["status"] == "cancelled"
    assert era5_events(cfg)[-1]["status"] == "failed"     # one failure: the dashboard shows it
    gate.set()                                          # the late answers change nothing on record
    time.sleep(0.3)
    assert status_file(cfg)["status"] == "cancelled" and box["st"]["fetched"] == 0
    assert seen.count("20200101") == 0                  # never requested


def test_the_end_of_a_run_cancels_its_prefetch(tmp_path, offline, gate, monkeypatch):
    """A run that ends (here: phase linking fails) while fetches are still
    pending cancels the prefetch it started."""
    from insar_pipeline import miaplpy_backend as mb, notify
    from insar_pipeline.config import load_config
    f = tmp_path / "c.yaml"
    f.write_text(f"project_dir: {tmp_path / 'proj'}\nbbox: [-87.7, 38.2, -87.6, 38.3]\n"
                 "start_date: '2020-01-01'\nend_date: '2021-01-01'\n")
    cfg = load_config(str(f))
    cfg.backend, cfg.mintpy_tropo_method = "miaplpy", "pyaps"
    cfg.project_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(pipeline, "check_preconditions", lambda *a, **k: {})
    monkeypatch.setattr(notify, "send", lambda *a, **k: {})

    def never(date, path, snwe, hour):
        gate.wait()
        return "ok"
    monkeypatch.setattr(sb, "fetch_era5_grib", never)

    def phase_linking(cfg):
        assert wait_for(lambda: len(status_file(cfg).get("in_flight", {})) == 4)
        raise RuntimeError("stop here")
    monkeypatch.setattr(mb, "run", phase_linking)
    t0 = time.time()
    assert pipeline.run_from(cfg, "miaplpy", kind="miaplpy", watchdog=False, notify=False) == 1
    assert time.time() - t0 < 15
    rec = status_file(cfg)
    assert rec["status"] == "cancelled" and len(rec["in_flight"]) == 4
    assert era5_events(cfg)[-1]["status"] == "skipped"     # nothing failed: left for MintPy
    assert not [t for t in threading.enumerate() if t.name == "era5-prefetch" and t.is_alive()]


# ------------------------------------------------------------------ --to-step

def _stub_coregistration(monkeypatch, calls, stub_prefetch=True):
    monkeypatch.setattr(sb, "plan_stack", lambda cfg: "plan")
    monkeypatch.setattr(sb, "format_plan", lambda plan: "plan")
    for name in ("disk_gate", "burst_gate"):
        monkeypatch.setattr(pipeline, name, lambda *a, **k: None)
    monkeypatch.setattr(pipeline, "refuse_screening_folder", lambda cfg: None)
    monkeypatch.setattr(sb, "stage_inputs", lambda cfg, skip_download=False: "dem")
    monkeypatch.setattr(sb, "generate_run_files", lambda cfg, dem: None)
    monkeypatch.setattr(sb, "patch_configs", lambda cfg: None)
    monkeypatch.setattr(sb, "execute", lambda cfg, from_step=None, to_step=None: calls.append(("execute", to_step)))
    if stub_prefetch:
        monkeypatch.setattr(pipeline, "start_era5_prefetch", lambda cfg: calls.append(("prefetch",)))


@pytest.mark.parametrize("backend", ["miaplpy", "stack"])
def test_a_to_step_run_does_not_start_the_prefetch(tmp_path, monkeypatch, backend):
    calls = []
    _stub_coregistration(monkeypatch, calls)
    cfg = types.SimpleNamespace(project_dir=tmp_path, backend=backend, stack_virtual_merge="False",
                                stack_cleanup_coreg_after_filter=False, mintpy_tropo_method="pyaps")
    args = types.SimpleNamespace(to_step="merge_reference_secondary_slc", from_step=None, plan_only=False,
                                 skip_download=True)
    pipeline.RUNNERS[backend](cfg, args)
    assert ("prefetch",) not in calls and calls[-1][0] == "execute"
    # the run that carries on to MintPy still starts it
    calls.clear()

    def stop(*a, **k):
        raise RuntimeError("past coregistration")
    monkeypatch.setattr(sb, "execute", stop)
    with pytest.raises(RuntimeError):
        pipeline.RUNNERS[backend](cfg, types.SimpleNamespace(**{**vars(args), "to_step": None}))
    assert calls == [("prefetch",)]


def test_a_to_step_run_leaves_no_prefetch_running(tmp_path, offline, gate, monkeypatch):
    """The whole 22 Oct path: run_from with --to-step on the phase-linked
    route ends 'done', and no prefetch thread or fetch is left."""
    from insar_pipeline import notify
    from insar_pipeline.config import load_config
    f = tmp_path / "c.yaml"
    f.write_text(f"project_dir: {tmp_path / 'proj'}\nbbox: [-87.7, 38.2, -87.6, 38.3]\n"
                 "start_date: '2020-01-01'\nend_date: '2021-01-01'\n")
    cfg = load_config(str(f))
    cfg.backend, cfg.mintpy_tropo_method = "miaplpy", "pyaps"
    cfg.stack_virtual_merge, cfg.stack_cleanup_coreg_after_filter = "False", False
    cfg.project_dir.mkdir(parents=True, exist_ok=True)
    calls = []
    _stub_coregistration(monkeypatch, calls, stub_prefetch=False)
    monkeypatch.setattr(pipeline, "check_preconditions", lambda *a, **k: {})
    monkeypatch.setattr(notify, "send", lambda *a, **k: {})
    fetched = []

    def never(date, path, snwe, hour):
        fetched.append(date)
        gate.wait()
        return "ok"
    monkeypatch.setattr(sb, "fetch_era5_grib", never)
    assert pipeline.run_from(cfg, "full", kind="full", to_step="merge_reference_secondary_slc",
                             watchdog=False, notify=False) == 0
    assert json.loads((cfg.project_dir / "run_result.json").read_text())["status"] == "done"
    time.sleep(0.3)
    assert fetched == [] and not era5_events(cfg)
    assert not [t for t in threading.enumerate() if t.name.startswith("era5-") and t.is_alive()]



# ------------------------------------------------------------------ reasons as they happen

def test_failure_reasons_are_recorded_as_they_happen(tmp_path, offline, gate, caplog):
    """While the prefetch is still going, era5_prefetch.json already says
    which dates failed and why, the count per reason, and what is in flight
    since when; the first failure of each reason is a log WARNING."""
    cfg = fake_cfg(tmp_path)

    def fetch(date, path, snwe, hour):
        if date in ("20200206", "20200125"):
            return SERVER_ERR.replace("14ba0f6d", "aaaaaaaa" if date == "20200125" else "14ba0f6d")
        if date == "20200113":
            return "FAILED HTTPSConnectionPool(host='cds.climate.copernicus.eu', port=443): Read timed out."
        gate.wait()
        return "ok"
    cancel = threading.Event()
    caplog.set_level("INFO", logger=sb.log.name)
    th = threading.Thread(target=sb.start_era5_prefetch, args=(cfg,),
                          kwargs=dict(workers=1, fetch=fetch, cancel=cancel), daemon=True)
    th.start()
    rec = wait_for(lambda: (lambda r: r if "20200101" in r.get("in_flight", {}) else None)(status_file(cfg)))
    assert rec, "era5_prefetch.json was not written while the prefetch was going"
    assert rec["status"] == "running" and rec["fetched"] == 0 and rec["waiting"] == 0
    assert set(rec["failed"]) == {"20200206", "20200125", "20200113"}
    q = rec["failed"]["20200206"]
    assert "An internal error occurred while processing the request" in q
    assert "http" not in q and "14ba0f6d" not in q            # the same cause reads the same
    assert rec["reasons"] == {q: 2, rec["failed"]["20200113"]: 1}
    assert rec["in_flight"]["20200101"][:4] == time.strftime("%Y")
    warned = [r.getMessage() for r in caplog.records if r.levelname == "WARNING" and "ERA5" in r.getMessage()]
    assert len(warned) == 2 and any("An internal error occurred" in w for w in warned)
    msgs = [e["msg"] for e in era5_events(cfg) if e["status"] == "running"]
    assert msgs[0].startswith("1 failed; last: 500 Server Error") and "An internal error" in msgs[0]
    assert msgs[-1] == f"3 failed; last: {rec['failed']['20200113']}"
    cancel.set()
    th.join(5)
    assert status_file(cfg)["status"] == "cancelled"


def test_a_finished_prefetch_still_writes_its_record(tmp_path, offline):
    """The end record is unchanged in substance: done, nothing in flight."""
    cfg = fake_cfg(tmp_path)

    def ok(date, path, snwe, hour):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(b"G")
        return "ok"
    st = sb.start_era5_prefetch(cfg, fetch=ok)
    rec = status_file(cfg)
    assert st["status"] == rec["status"] == "done" and rec["fetched"] == 4
    assert rec["in_flight"] == {} and rec["waiting"] == 0 and rec["failed"] == {}
    assert not [t for t in threading.enumerate() if t.name.startswith("era5-fetch") and t.is_alive()]


# ------------------------------------------------------------------ the CDS queue is full (23 Oct)

def test_a_full_cds_queue_is_waited_out_not_counted_as_a_failure(tmp_path, offline, monkeypatch):
    """23 Oct 2026: the CDS turned away every request for a minute, then
    accepted four; each refused date had been dropped as 'failed'. Now the
    same date is asked for again after a wait, and the refusals are counted
    apart from the failures."""
    monkeypatch.setattr(sb, "ERA5_BUSY_WAIT_S", (0.01, 0.02))
    cfg = fake_cfg(tmp_path)
    tries = {}

    def busy_then_ok(date, path, snwe, hour):
        tries[date] = tries.get(date, 0) + 1
        if tries[date] <= 2:
            return QUEUE_FULL
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(b"G")
        return "ok"
    st = sb.start_era5_prefetch(cfg, workers=2, fetch=busy_then_ok)
    assert st["status"] == "done" and st["fetched"] == 4 and st["failed"] == {}
    assert st["queue_full"] == 8 and tries == {d: 3 for d in DATES}
    assert status_file(cfg)["queue_full"] == 8
    assert any(e["msg"].startswith("CDS queue full") for e in era5_events(cfg))


def test_a_cancel_during_the_wait_leaves_the_date_for_mintpy(tmp_path, offline, monkeypatch):
    monkeypatch.setattr(sb, "ERA5_BUSY_WAIT_S", (30.0, 60.0))
    cfg = fake_cfg(tmp_path)
    cancel = threading.Event()
    box = {}
    th = threading.Thread(target=lambda: box.update(st=sb.start_era5_prefetch(
        cfg, workers=1, fetch=lambda *a: QUEUE_FULL, cancel=cancel)), daemon=True)
    th.start()
    assert wait_for(lambda: status_file(cfg).get("queue_full") == 1)
    t0 = time.time()
    cancel.set()
    th.join(5)
    assert not th.is_alive() and time.time() - t0 < 3       # the 30 s wait did not hold it
    st = box["st"]
    assert st["status"] == "cancelled" and st["failed"] == {} and st["in_flight"] == {}
    assert st["waiting"] == 4 or "4 not requested" in st["reason"]


def test_other_failures_still_fail_at_once():
    assert sb._ERA5_BUSY_RE.search(QUEUE_FULL)
    assert not sb._ERA5_BUSY_RE.search(SERVER_ERR)
    assert not sb._ERA5_BUSY_RE.search("FAILED 403 Client Error: required licences not accepted")
