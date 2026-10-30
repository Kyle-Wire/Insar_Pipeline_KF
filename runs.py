"""Start, find and stop detached pipeline runs, and index the projects.

A run is recorded in <project>/run.json when it starts:

    {"pid": 1234, "start_ticks": 987654, "kind": "full", "config": ".../x.yaml",
     "cmd": [...], "log": ".../logs/full_20260928_0100.log", "started": 1759...}

and is identified afterwards by that PID AND the kernel's start time for it
(/proc/<pid>/stat field 22). PIDs are reused; a PID alone can point at an
unrelated process a week later. Nothing here ever matches processes by name
or command-line pattern, and stop() signals only the recorded process group.

The run executes in its own session (setsid), so it survives the dashboard,
the terminal and an SSH drop; the dashboard reattaches by reading run.json.

Every launcher - the dashboard, `python -m insar_pipeline.pipeline`, the
scripts, the old menu - also holds an exclusive flock on <project>/.run.lock
for the life of the run (project_lock / guarded). run.json is written only by
runs.start, so a run started any other way used to be invisible here and a
second one could be started on the same folder (two runners on one stack/,
cleanups deleting what the other reads). The kernel drops the lock on exit,
SIGKILL, OOM kill or reboot, so a held lock is also a reliable "alive" test.
A run's end is written to <project>/run_result.json by pipeline.run_from.
"""
from __future__ import annotations

import contextlib
import fcntl
import json
import logging
import os
import re
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from . import procs, progress

log_ = logging.getLogger(__name__)

PKG = Path(__file__).resolve().parent
PARENT = PKG.parent
PY = sys.executable
RUNFILE = "run.json"
LOCKFILE = ".run.lock"
RESULTFILE = "run_result.json"
# Every earlier run_result.json, one JSON per line, oldest first (16 Oct
# 2026, readiness review ERR-04: each run overwrote the last one's, so a
# failure record was gone once the next run ended).
HISTORYFILE = "run_result_history.jsonl"
# One line per finished run of every project, under the projects root
# (OUT-05): which runs, of which project, did build X make.
RUN_REGISTRY = "run_history.jsonl"
RUN_REGISTRY_ENV = "INSAR_RUN_HISTORY"   # another file (the tests' isolation)
CHILDFILE = ".run_children.json"
LOCK_FD_ENV = "INSAR_RUN_LOCK_FD"     # runs.start hands its lock to the run it launches
REGISTRY = Path.home() / ".insar_pipeline_cli.json"

# What can be started, as (label, argv after the interpreter). {cfg} is the
# config path. Every one of these ends in the client report.
KINDS = {
    # First, so the dashboard lists it first: the restart that cannot be the
    # wrong one. It reads the disk (suggest_kind) - a "Whole pipeline" pick
    # after a phase-linked crash re-assembled every cleaned date (3 h, ~111
    # GB on Site B), and "Phase linking onward" after a half-finished
    # merge linked a shortened stack for good.
    "resume": ("Resume: pick up where the last run stopped (worked out from the files on disk)",
               ["-m", "insar_pipeline.pipeline", "{cfg}", "--resume"]),
    "full": ("Whole pipeline: data, processing, products, report",
             ["-m", "insar_pipeline.pipeline", "{cfg}"]),
    # One stage chain (pipeline.run_from) for every kind; the scripts/ runners
    # are thin calls to the same function for use from a terminal.
    "products": ("Products only: deliverables, QA, figures, GNSS tie, client report",
                 ["-m", "insar_pipeline.pipeline", "{cfg}", "--start", "products"]),
    "mintpy": ("MintPy onward on a finished stack",
               ["-m", "insar_pipeline.pipeline", "{cfg}", "--start", "mintpy"]),
    "miaplpy": ("Phase linking onward on a finished coregistration",
                ["-m", "insar_pipeline.pipeline", "{cfg}", "--start", "miaplpy"]),
    "reunwrap": ("Re-unwrap the phase-linked export, then MintPy onward",
                 [str(PKG / "scripts" / "reunwrap_linked.py"), "{cfg}"]),
    "plan": ("Size the run only (downloads nothing)",
             ["-m", "insar_pipeline.pipeline", "{cfg}", "--plan-only"]),
}
# The stage chain each kind starts from (pipeline.run_from).
KIND_START = {"full": "full", "products": "products", "mintpy": "mintpy", "miaplpy": "miaplpy",
              "reunwrap": "unwrap", "plan": "full"}


def _start_ticks(pid: int) -> int | None:
    """Kernel start time of a live process in clock ticks; None if it is gone
    or a zombie (exited, not yet reaped - which is what a run started from
    the dashboard looks like after it finishes, until the dashboard reaps it)."""
    try:
        os.waitpid(pid, os.WNOHANG)          # reap it if it is our child and done
    except (ChildProcessError, OSError):
        pass
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return None
    # the command name (field 2) may contain spaces and parentheses
    rest = stat[stat.rindex(")") + 2:].split()
    if rest[0] in ("Z", "X"):
        return None
    return int(rest[19])         # field 22 overall; 20th after "pid (comm)"


def alive(rec: dict) -> bool:
    pid = rec.get("pid")
    return bool(pid) and _start_ticks(pid) is not None and _start_ticks(pid) == rec.get("start_ticks")


def record(project_dir) -> dict | None:
    p = Path(project_dir) / RUNFILE
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return None


# ---------------------------------------------------------------- the project lock
_HELD: dict = {}                   # resolved project dir -> {"fd", "n", "info"} held by THIS process
_HELD_LOCK = threading.Lock()


def _lock_path(project_dir) -> Path:
    return Path(project_dir) / LOCKFILE


def _read_lock_info(fd: int) -> dict:
    try:
        raw = os.pread(fd, 65536, 0)
        return json.loads(raw.decode() or "{}")
    except (OSError, ValueError):
        return {}


def _write_lock_info(fd: int, info: dict) -> None:
    try:
        data = json.dumps(info, default=str).encode()
        os.ftruncate(fd, 0)
        os.pwrite(fd, data, 0)
    except OSError:
        pass


def _busy_text(project_dir, info: dict | None) -> str:
    info = info or {}
    when = ""
    if info.get("started"):
        when = time.strftime(", started %d %b %H:%M", time.localtime(info["started"]))
    what = f"a {info['kind']} run" if info.get("kind") else "a run"
    pid = info.get("pid", "?")
    return f"already running ({what}, pid {pid}{when}) in {project_dir}"


def lock_holder(project_dir) -> dict | None:
    """The run holding <project>/.run.lock ({pid, start_ticks, kind, started,
    argv}, {} if it wrote nothing), or None if no process holds it. Never
    takes the lock itself for longer than the test."""
    f = _lock_path(project_dir)
    try:
        fd = os.open(f, os.O_RDONLY)
    except OSError:
        return None
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError:
            return _read_lock_info(fd)
        except OSError:
            return None                  # a filesystem without flock: nothing to learn here
        fcntl.flock(fd, fcntl.LOCK_UN)
        return None
    finally:
        os.close(fd)


def _lock_info(kind: str | None, config: str | None = None) -> dict:
    return {"pid": os.getpid(), "start_ticks": _start_ticks(os.getpid()), "kind": kind,
            "config": config, "argv": sys.argv, "started": time.time()}


def _adopt_inherited(project_dir) -> int | None:
    """The lock fd runs.start passed to this process, if it is this project's."""
    val = os.environ.get(LOCK_FD_ENV)
    if not val:
        return None
    try:
        fd = int(val)
        a, b = os.fstat(fd), os.stat(_lock_path(project_dir))
        if (a.st_dev, a.st_ino) != (b.st_dev, b.st_ino):
            return None
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)     # same open file: already ours, a no-op
    except (ValueError, OSError):
        return None
    os.environ.pop(LOCK_FD_ENV, None)    # it is not passed further down
    return fd


@contextlib.contextmanager
def project_lock(project_dir, kind: str | None = None, config: str | None = None):
    """Hold the exclusive per-project run lock for the body; raise
    RuntimeError("already running (pid N, started ...)") when another process
    holds it. Re-entrant within one process (launch -> run_from). The lock
    file itself is never deleted (deleting a lock file is its own race)."""
    P = Path(project_dir)
    P.mkdir(parents=True, exist_ok=True)
    key = str(P.resolve())
    with _HELD_LOCK:
        if key in _HELD:
            _HELD[key]["n"] += 1
            nested = True
        else:
            nested = False
    if nested:
        try:
            yield _HELD[key]["info"]
        finally:
            with _HELD_LOCK:
                _HELD[key]["n"] -= 1
        return
    fd = _adopt_inherited(P)
    if fd is None:
        fd = os.open(_lock_path(P), os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            info = _read_lock_info(fd)
            os.close(fd)
            raise RuntimeError(_busy_text(P, info)) from None
        except OSError as exc:
            # flock unsupported (a project on /mnt/c): run.json is then the only guard
            log_.warning("Could not lock %s (%s); only run.json guards against a second run",
                         _lock_path(P), exc)
    info = _lock_info(kind, config)
    _write_lock_info(fd, info)
    with _HELD_LOCK:
        _HELD[key] = {"fd": fd, "n": 1, "info": info}
    try:
        yield info
    finally:
        with _HELD_LOCK:
            _HELD.pop(key, None)
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        except OSError:
            pass
        os.close(fd)


@contextlib.contextmanager
def guarded(project_dir, kind: str | None = None, config: str | None = None):
    """What every launcher does first: child teardown on SIGTERM/SIGINT/SIGHUP
    (procs.install_teardown, from the main thread) and the project lock."""
    procs.install_teardown()
    with project_lock(project_dir, kind=kind, config=config) as info:
        yield info


def _keep_as_previous(cur: dict | None) -> dict | None:
    """The record a new run.json keeps under "previous". Sizing is not a run:
    when the current record is a plan, the run before it is kept instead, so
    sizing twice does not lose the last real run (state() then judged the
    first plan, which never reports an end, and showed a finished project
    'failed')."""
    if cur and cur.get("kind") == "plan" and not cur.get("stopped") and cur.get("previous"):
        cur = cur["previous"]
    return {k: v for k, v in cur.items() if k != "previous"} if cur else None


def claim_runfile(project_dir, kind: str, config: str | None = None) -> dict:
    """Make run.json describe THIS process. A run started by runs.start is
    already recorded (its pid is ours) and is left alone; a run started from a
    terminal, a script or the old menu writes its own record, so the
    dashboard can show, follow and stop it like any other. The record it
    replaces is kept under "previous"."""
    P = Path(project_dir)
    cur = record(P)
    if cur and cur.get("pid") == os.getpid() and alive(cur):
        res = result(P) or {}
        if not (res.get("pid") == os.getpid() and res.get("finished", 0) >= cur.get("started", 0)):
            return cur                   # launched by runs.start: already recorded
        # else: an earlier run_from in this same process ended; this is a new run
    log_path = None
    try:
        lp = os.readlink("/proc/self/fd/1")
        if lp.startswith("/") and Path(lp).is_file():
            log_path = lp
    except OSError:
        pass
    rec = {"pid": os.getpid(), "start_ticks": _start_ticks(os.getpid()), "kind": kind,
           "config": str(config) if config else None, "cmd": [PY] + sys.argv, "log": log_path,
           "started": time.time(), "launched_by": "self",
           # SIGTERM to the group only works when the run leads its own group
           "own_group": os.getpgid(0) == os.getpid()}
    if cur:
        rec["previous"] = _keep_as_previous(cur)
    try:
        (P / RUNFILE).write_text(json.dumps(rec, indent=1))
    except OSError:
        pass
    return rec


# ---------------------------------------------------------------- start / stop
def stop_orphans(project_dir, grace_s: float = 3.0) -> dict:
    """Stop command groups a DEAD runner left behind (SIGKILL, OOM kill, a
    runner whose teardown never ran): they are listed in .run_children.json
    with their start ticks, so only those exact processes are signalled. A
    live runner's children are never touched. {"stopped": [...], "survivors": [...]}."""
    P = Path(project_dir)
    out = {"stopped": [], "survivors": []}
    f = P / CHILDFILE
    if not f.exists() or lock_holder(P) is not None:
        return out
    try:
        rec = json.loads(f.read_text())
    except (OSError, ValueError):
        return out
    rp = rec.get("runner_pid")
    if rp and rec.get("runner_start_ticks") is not None and procs.start_ticks(rp) == rec["runner_start_ticks"]:
        return out                       # its runner is alive (and lockless): not orphans
    groups = procs.recorded_groups(f)
    if groups:
        survivors = procs.kill_groups(groups, grace_s=grace_s)
        out["survivors"] = survivors
        out["stopped"] = [g for g in groups if g not in survivors]
        log_.warning("Stopped %d command group(s) left running by a run that died: %s",
                     len(out["stopped"]), out["stopped"])
    return out


def start(cfg_path: str, project_dir, kind: str = "full", extra: list | None = None) -> dict:
    """Launch a run detached; refuse if one is already running here - by
    run.json, or by ANY launcher's project lock. The lock is taken here and
    handed to the run (pass_fds), so two starts racing cannot both win."""
    if kind not in KINDS:
        raise ValueError(f"unknown run kind {kind!r}; one of {list(KINDS)}")
    project_dir = Path(project_dir)
    project_dir.mkdir(parents=True, exist_ok=True)
    cur = record(project_dir)
    if cur and alive(cur):
        raise RuntimeError(f"a {cur.get('kind')} run is already going in {project_dir} (pid {cur['pid']})")
    held = lock_holder(project_dir)
    if held is not None:
        raise RuntimeError(_busy_text(project_dir, held))
    orphans = stop_orphans(project_dir)
    if orphans["survivors"]:
        raise RuntimeError(f"commands left by an earlier run could not be stopped (process groups "
                           f"{orphans['survivors']}); stop them before starting again")
    resumed_as = reunwrap_started = None
    if kind == "resume":
        # what Resume resolves to, worked out (before the lock is taken: a
        # held lock reads as another run) from the same disk the run reads a
        # moment later in pipeline.launch; kept in run.json so a Resume that
        # is itself killed still says what it was doing (unfinished_reunwrap)
        try:
            from .config import load_config
            c = load_config(str(cfg_path))
            resumed_as = suggest_kind(project_dir, c)[0]
        except Exception:                                    # noqa: BLE001 - a note, not a gate
            c = resumed_as = None
        if resumed_as == "reunwrap":
            # and when the re-unwrap it carries on began: run.json keeps one
            # earlier record, so after two Resumes killed before their unwrap
            # step the first re-unwrap's start would be lost
            try:
                reunwrap_started = (unfinished_reunwrap(project_dir, c) or {}).get("started")
            except Exception:                                # noqa: BLE001 - a note, not a gate
                reunwrap_started = None
    fd = os.open(_lock_path(project_dir), os.O_RDWR | os.O_CREAT, 0o644)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(_busy_text(project_dir, _read_lock_info(fd))) from None
        except OSError:
            pass
        logs = project_dir / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        log = logs / f"{kind}_{time.strftime('%Y%m%d_%H%M%S')}.log"
        argv = [PY] + [a.replace("{cfg}", str(cfg_path)) for a in KINDS[kind][1]] + list(extra or [])
        env = {**os.environ, "PYTHONPATH": str(PARENT) + os.pathsep + os.environ.get("PYTHONPATH", ""),
               "PYTHONUNBUFFERED": "1", LOCK_FD_ENV: str(fd)}
        with open(log, "a") as fh:
            write_log_label(fh)          # the log's first line names the code (ERR-04, 16 Oct 2026)
            p = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=fh, stderr=subprocess.STDOUT,
                                 start_new_session=True, cwd=str(PARENT), env=env, pass_fds=(fd,))
        ticks = None
        for _ in range(50):                      # /proc entry appears immediately; be patient anyway
            ticks = _start_ticks(p.pid)
            if ticks is not None:
                break
            time.sleep(0.02)
        rec = {"pid": p.pid, "start_ticks": ticks, "kind": kind, "config": str(cfg_path), "cmd": argv,
               "log": str(log), "started": time.time()}
        if resumed_as:
            rec["resumed_as"] = resumed_as
        if reunwrap_started:
            rec["reunwrap_started"] = reunwrap_started
        if cur:
            rec["previous"] = _keep_as_previous(cur)
        if orphans["stopped"]:
            rec["orphans_stopped"] = orphans["stopped"]
        _write_lock_info(fd, {"pid": p.pid, "start_ticks": ticks, "kind": kind, "config": str(cfg_path),
                              "argv": argv, "started": rec["started"]})
    finally:
        os.close(fd)                     # the run holds the same lock through its inherited copy
    (project_dir / RUNFILE).write_text(json.dumps(rec, indent=1))
    progress.emit(project_dir, "run", "start", msg=f"{kind} (pid {p.pid})")
    _remember(cfg_path)
    return rec


def _signal_run(rec: dict, sig: int) -> None:
    pid = rec["pid"]
    try:
        if os.getpgid(pid) == pid:
            os.killpg(pid, sig)
        else:
            os.kill(pid, sig)            # a terminal-started run: its handler stops its children
    except ProcessLookupError:
        pass


def stop(project_dir, grace_s: float = 20.0) -> str:
    """SIGTERM the run's process group (the runners tear down their children
    and leave resumable state), then SIGKILL if it is still there; then stop
    any command group the run recorded that is still alive, and say so if
    one survives - "stopped" is reported only once nothing is left."""
    P = Path(project_dir)
    rec = record(P)
    if not rec or not alive(rec):
        held = lock_holder(P)
        if held and held.get("pid") == os.getpid():
            # the dashboard itself holds the project (rebuilding the report or
            # building the package): signalling the holder would kill it
            return "nothing to stop: this dashboard is rebuilding the report or package; wait for it to finish"
        if held and held.get("pid") and held.get("start_ticks") is not None \
                and _start_ticks(held["pid"]) == held["start_ticks"]:
            rec = {**held, "launched_by": "elsewhere"}      # started outside the dashboard
        else:
            left = stop_orphans(P)
            if left["stopped"] or left["survivors"]:
                return (f"nothing running; stopped {len(left['stopped'])} leftover command group(s)"
                        + (f", {len(left['survivors'])} could not be stopped: {left['survivors']}"
                           if left["survivors"] else ""))
            return "nothing running"
    try:
        pgid = os.getpgid(rec["pid"])
    except ProcessLookupError:
        return "nothing running"
    if pgid != rec["pid"] and rec.get("launched_by") not in ("self", "elsewhere"):
        # runs.start uses start_new_session, so its run leads its own group;
        # anything else means this is not the process we launched
        return f"pid {rec['pid']} is not a group leader - not signalling"
    _signal_run(rec, signal.SIGTERM)
    t0 = time.time()
    while time.time() - t0 < grace_s and alive(rec):
        time.sleep(0.25)
    how = "stopped"
    if alive(rec):
        _signal_run(rec, signal.SIGKILL)
        how = "killed after the grace period"
        t0 = time.time()
        while time.time() - t0 < 5 and alive(rec):
            time.sleep(0.1)
    left = stop_orphans(P)
    if left["stopped"]:
        how += f"; {len(left['stopped'])} command group(s) it left running were stopped too"
    if left["survivors"]:
        how += f"; WARNING {len(left['survivors'])} command group(s) are still running: {left['survivors']}"
    cur = record(P) or rec
    if cur.get("pid") == rec["pid"]:
        cur["stopped"] = time.time()
        (P / RUNFILE).write_text(json.dumps(cur, indent=1))
    progress.emit(P, "run", "failed", msg=f"{how} by the user")
    return how


# ---------------------------------------------------------------- state
def result(project_dir) -> dict | None:
    """<project>/run_result.json written by pipeline.run_from at the end of a run."""
    try:
        return json.loads((Path(project_dir) / RESULTFILE).read_text())
    except (OSError, ValueError):
        return None


def _append_line(path: Path, obj) -> None:
    """One JSON line appended under an exclusive lock (two projects' runs
    can end at the same moment and share the registry)."""
    line = json.dumps(obj, default=str, separators=(",", ":")) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as fh:
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        except OSError:
            pass
        fh.write(line)


def keep_previous_result(project_dir, new: dict | None = None) -> bool:
    """Append the run_result.json on disk to <project>/run_result_history.jsonl
    before a run replaces it (16 Oct 2026, readiness review ERR-04). Not
    when it is `new`'s own run (same pid and start) - a run that writes its
    end twice keeps one entry - and an unreadable one is kept as its text.
    Never raises; returns whether a line was added."""
    P = Path(project_dir)
    f = P / RESULTFILE
    try:
        raw = f.read_text()
    except OSError:
        return False
    try:
        prev = json.loads(raw)
    except ValueError:
        prev = {"unreadable_run_result": raw[:4000]}
    new = new or {}
    if (isinstance(prev, dict) and prev.get("started") is not None and prev.get("started") == new.get("started")
            and prev.get("pid") == new.get("pid")):
        return False
    try:
        _append_line(P / HISTORYFILE, prev)
        return True
    except OSError as exc:
        log_.warning("The previous run_result.json was not kept in %s: %s", P / HISTORYFILE, exc)
        return False


def result_history(project_dir) -> list[dict]:
    """The earlier run_result.json records, oldest first (keep_previous_result)."""
    out = []
    try:
        with open(Path(project_dir) / HISTORYFILE) as fh:
            for line in fh:
                try:
                    out.append(json.loads(line))
                except ValueError:
                    continue
    except OSError:
        pass
    return out


def run_registry_path() -> Path:
    """<projects root>/run_history.jsonl ($INSAR_RUN_HISTORY when set)."""
    env = os.environ.get(RUN_REGISTRY_ENV)
    if env:
        return Path(os.path.expanduser(env))
    from . import settings
    return settings.projects_root() / RUN_REGISTRY


def _iso(t) -> str | None:
    try:
        return datetime.fromtimestamp(float(t)).astimezone().isoformat(timespec="seconds")
    except (TypeError, ValueError, OverflowError, OSError):
        return None


def register_run(project_dir, rec: dict) -> Path | None:
    """One line for a finished run in the run registry (run_registry_path;
    16 Oct 2026, readiness review OUT-05): {project, kind, started,
    finished, status, stage, exit_code, version, code_hash}. The project is
    its folder's name, never a path (a path names the user; DAT-07 took
    user and host names out of the records). Append-only; never raises.
    A run killed outright (SIGKILL, out of memory) writes no end record
    and so no line."""
    line = {"project": Path(project_dir).name, "kind": rec.get("kind"), "started": _iso(rec.get("started")),
            "finished": _iso(rec.get("finished")), "status": rec.get("status"), "stage": rec.get("stage"),
            "exit_code": rec.get("exit_code"), "version": rec.get("version"), "code_hash": rec.get("code_hash")}
    try:
        path = run_registry_path()
        _append_line(path, line)
        return path
    except Exception as exc:                                 # noqa: BLE001 - the run's own end comes first
        log_.warning("The run was not added to the run registry: %s", exc)
        return None


def run_registry(code_hash: str | None = None, project: str | None = None) -> list[dict]:
    """The registry's lines, oldest first; only those of one build or one
    project folder when given ('which outputs did build X make')."""
    out = []
    try:
        with open(run_registry_path()) as fh:
            for line in fh:
                try:
                    d = json.loads(line)
                except ValueError:
                    continue
                if code_hash is not None and d.get("code_hash") != code_hash:
                    continue
                if project is None or d.get("project") == project:
                    out.append(d)
    except OSError:
        pass
    return out


def build_label() -> str:
    """'insar_pipeline <version> (code <hash>)' of the code on disk NOW: the
    dashboard lives for days, and version.label() is the hash it computed
    when it started, so it would name an older build than the run's."""
    try:
        from . import version
        return f"insar_pipeline {version._read_version()} (code {version._hash_tree(version.PACKAGE_DIR)})"
    except Exception as exc:                                 # noqa: BLE001 - a log still starts
        return f"insar_pipeline (version not read: {exc})"


def write_log_label(fh) -> None:
    """The build label as a log's first line (16 Oct 2026, readiness review
    ERR-04): a log on its own says which code made it."""
    try:
        fh.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {build_label()}\n")
        fh.flush()
    except (OSError, ValueError):
        pass


def state(project_dir) -> str:
    """running / done / failed / stopped / interrupted / idle - from the
    process, the project lock, run_result.json and the progress feed, never
    from log text. "interrupted": the process is gone without a final event
    after it had started work - every failure inside a run is caught and
    reported (pipeline.run_from), so this is a kill it could not report: WSL
    shutdown, a reboot, an out-of-memory kill (interruption() says which)."""
    P = Path(project_dir)
    rec = record(P)
    res = result(P)
    if rec and alive(rec):
        # a host that called run_from in-process (tests, a notebook) outlives
        # its run: the run's own result says it is over
        if not (res and res.get("pid") == rec.get("pid") and res.get("finished", 0) >= rec.get("started", 0)
                and res.get("status") in ("done", "failed", "stopped") and lock_holder(P) is None):
            return "running"
    elif lock_holder(P) is not None:
        return "running"                 # started outside the dashboard; the lock says it is alive
    while rec and rec.get("kind") == "plan" and not rec.get("stopped"):
        rec = rec.get("previous")        # sizing is not a run: judge by the run before it
    if rec and rec.get("stopped"):
        return "stopped"
    L = progress.latest(P)
    if rec:
        since = rec["started"] - 1
        if res and res.get("finished", 0) >= since and res.get("pid") in (None, rec.get("pid")) \
                and res.get("status") in ("done", "failed", "stopped"):
            return res["status"]
        # the run's own final event; a process that vanished without one died
        mine = [e for e in L.values() if e["t"] >= since]
        if any(e["status"] == "failed" for e in mine):
            return "failed"
        # the newest finishing event of THIS run: a partial restart (phase
        # linking onward, products only) emits no "pipeline" event, so an
        # earlier run's "pipeline failed" must not decide it
        tops = [L[k] for k in ("pipeline", "client_report") if k in L and L[k]["t"] >= since]
        top = max(tops, key=lambda e: e["t"]) if tops else None
        if top and top["status"] == "done":
            return "done"
        if any(k != "run" for k, e in L.items() if e["t"] >= since):
            return "interrupted"
        # nothing but the launch: it died before its first stage (a config or
        # import error at start-up), which the log shows
        return "failed" if mine else "idle"
    if res and res.get("status") in ("done", "failed", "stopped"):
        return res["status"]
    if (P / "deliverables" / "client_report.json").exists():
        return "done"
    return "idle"


def _boot_time() -> float | None:
    try:
        for line in Path("/proc/stat").read_text().splitlines():
            if line.startswith("btime "):
                return float(line.split()[1])
    except (OSError, ValueError):
        pass
    return None


def _oom_killed(pids, since: float) -> bool | None:
    """True if the kernel log shows an out-of-memory kill of one of `pids`
    since `since`; None when the log cannot be read (no permission, no journal)."""
    pids = {str(p) for p in pids if p}
    if not pids:
        return None
    for argv in (["journalctl", "-k", "-q", "--no-pager", "--since", f"@{int(since)}"], ["dmesg"]):
        try:
            p = subprocess.run(argv, capture_output=True, text=True, timeout=5)
        except (OSError, subprocess.SubprocessError):
            continue
        if p.returncode != 0:
            continue
        for m in re.finditer(r"Killed process (\d+)", p.stdout):
            if m.group(1) in pids:
                return True
        return False
    return None


def interruption(project_dir, cfg=None) -> dict | None:
    """For an "interrupted" run: where it was, the likely reason and the
    restart to use. {"stage", "t", "likely", "restart_kind", "restart_reason",
    "text"}; None when the run is not interrupted. Reads the kernel's boot
    time (WSL/Windows restarted after the run's last report) and the kernel
    log (an OOM kill of the runner or one of its commands)."""
    P = Path(project_dir)
    if state(P) != "interrupted":
        return None
    rec = record(P) or {}
    while rec.get("kind") == "plan" and not rec.get("stopped"):
        rec = rec.get("previous") or {}
    since = rec.get("started", 0) - 1
    L = progress.latest(P)
    evs = [(k, e) for k, e in L.items() if k not in ("run", "pipeline", "disk") and e["t"] >= since] \
        or [(k, e) for k, e in L.items() if k == "pipeline" and e["t"] >= since]
    stage, last = max(evs, key=lambda kv: kv[1]["t"]) if evs else ("the start", {"t": rec.get("started", 0)})
    t = last["t"]
    boot = _boot_time()
    pids = [rec.get("pid")]
    try:
        pids += list(json.loads((P / CHILDFILE).read_text()).get("groups", {}))
    except (OSError, ValueError):
        pass
    if boot and boot > t:
        likely = "WSL or Windows restarted while it was running (Windows Update, a shutdown or `wsl --shutdown`)"
    elif _oom_killed(pids, since):
        likely = "the computer ran out of memory and Linux killed it"
    else:
        likely = "it was killed without a chance to report (WSL shutdown, out of memory, or a kill from outside)"
    kind, why = suggest_kind(P, cfg)
    when = time.strftime("%d %b %H:%M", time.localtime(t))
    return {"stage": stage, "t": t, "likely": likely, "restart_kind": kind, "restart_reason": why,
            "text": f"Stopped without reporting during {stage} at {when}: {likely}. "
                    f"Restart with '{KINDS.get(kind, (kind,))[0].split(':')[0]}' ({why})."}


def live_runs(exclude=None) -> list:
    """Runs going right now in OTHER projects under the projects root (by
    run.json or by the project lock): two runs share 24 cores, ~23 GB of RAM
    and one disk budget, and each sizes its pools as if it had the machine to
    itself (2 x 24 snaphu + MiaplPy is past the OOM line). [{project, pid, kind}]."""
    try:
        from .settings import projects_root
        root = projects_root()
    except Exception:                                        # noqa: BLE001
        return []
    ex = Path(exclude).resolve() if exclude else None
    out = []
    try:
        dirs = [d for d in root.iterdir() if d.is_dir()]
    except OSError:
        return []
    for d in dirs:
        if ex is not None and d.resolve() == ex:
            continue
        if not ((d / RUNFILE).exists() or (d / LOCKFILE).exists()):
            continue
        rec = record(d)
        if rec and alive(rec):
            out.append({"project": str(d), "pid": rec.get("pid"), "kind": rec.get("kind")})
            continue
        held = lock_holder(d)
        if held is not None:
            out.append({"project": str(d), "pid": held.get("pid"), "kind": held.get("kind")})
    return out


# ---------------------------------------------------------------- what to restart with
def _mtime(p: Path) -> float:
    try:
        return p.stat().st_mtime
    except OSError:
        return 0.0


def coregistration_status(stack_dir) -> dict:
    """Merge completeness of a topsStack coregistration, from disk.

    expected = every config_merge_<date> stackSentinel wrote; merged = dates
    whose merge command has a success marker (stack/.status/run_*_merge_
    reference_secondary_slc, kept even after the merged SLCs are deleted
    post-load) or whose merged/SLC/<date>/<date>.slc.full exists.
    {"expected": [...], "merged": [...], "missing": [...]}."""
    S = Path(stack_dir)
    expected = sorted(m.group(1) for f in (S / "configs").glob("config_merge_*")
                      for m in [re.fullmatch(r"config_merge_(\d{8})", f.name)] if m)
    merged = set()
    for d in (S / ".status").glob("run_*_merge_reference_secondary_slc"):
        for mk in d.glob("*.done"):
            try:
                r = json.loads(mk.read_text())
            except (OSError, ValueError):
                continue
            m = re.search(r"config_merge_(\d{8})(?:\s|$)", r.get("cmd", ""))
            if m and r.get("returncode") == 0:
                merged.add(m.group(1))
    for d in expected:
        if d not in merged and (S / "merged" / "SLC" / d / f"{d}.slc.full").exists():
            merged.add(d)
    return {"expected": expected, "merged": sorted(merged & set(expected)) if expected else sorted(merged),
            "missing": [d for d in expected if d not in merged]}


def unwrap_status(stack_dir) -> dict:
    """Exported (filt_fine.int) and unwrapped (filt_fine.unw.conncomp) pairs
    under stack/merged/interferograms."""
    root = Path(stack_dir) / "merged" / "interferograms"
    exported, unwrapped = [], []
    if root.exists():
        for d in sorted(root.iterdir()):
            if (d / "filt_fine.int").exists():
                exported.append(d.name)
                if (d / "filt_fine.unw.conncomp").exists():
                    unwrapped.append(d.name)
    return {"exported": exported, "unwrapped": unwrapped,
            "missing": sorted(set(exported) - set(unwrapped))}


# MintPy runs on the stack and phase-linked routes write mintpy/run_products.json
# (time_series.write_run_products) since 5 Oct 2026; a velocity older than this
# with no record and nothing retired is a finished run from before then
# (Site B, site_a_pl).
RUN_RECORD_SINCE = time.mktime((2026, 10, 5, 0, 0, 0, 0, 0, -1))


def _mintpy_finished(M: Path) -> bool:
    """Whether the last MintPy run in M finished (stack and phase-linked
    routes): its run_products.json record (written only once every product
    is verified) is newer than the velocities - or, with no record, a run
    from before records existed, geocoded after its radar velocity. A run
    that died between velocity.h5 and the record (disk, memory, a failed
    geocode or product check) leaves a new radar velocity, an empty or
    failed geo/ and no record; 'Products only' would then build from it or
    fail on it every time."""
    vel = M / "velocity.h5"
    geo_vel = M / "geo" / "geo_velocity.h5"
    newest = max(_mtime(vel), _mtime(geo_vel))
    rp = M / "run_products.json"
    if rp.exists():
        return _mtime(rp) >= newest
    geocoded = not vel.exists() or _mtime(geo_vel) >= _mtime(vel)
    retired = any((M / "superseded").glob("run_*/RETIRED.json"))
    return geocoded and not retired and newest < RUN_RECORD_SINCE


def _lock_held_here(project_dir) -> bool:
    """Whether THIS process holds the project's lock: taken here
    (project_lock), or inherited from runs.start and not yet adopted - a run
    started by runs.start reads the disk (pipeline.launch's Resume) before
    its run.json and lock note may have been written."""
    P = Path(project_dir)
    try:
        key = str(P.resolve())
    except OSError:
        return False
    with _HELD_LOCK:
        if key in _HELD:
            return True
    val = os.environ.get(LOCK_FD_ENV)
    if not val:
        return False
    try:
        a, b = os.fstat(int(val)), os.stat(_lock_path(P))
    except (ValueError, OSError):
        return False
    return (a.st_dev, a.st_ino) == (b.st_dev, b.st_ino)


def _owns(res: dict | None, r: dict | None) -> bool:
    """Whether run_result.json `res` is the end record of run record `r`."""
    return bool(res) and bool(r) and res.get("pid") in (None, r.get("pid")) \
        and res.get("finished", 0) >= r.get("started", 0) - 1 and res.get("status") in ("done", "failed", "stopped")


def _ran_reunwrap(r: dict, own: dict | None) -> bool | None:
    """Whether run record `r` ran the Re-unwrap chain: its kind ('unwrap' is
    run_from's own name for a direct start there), or for a Resume what it
    resolved to (its end record's resumed_as / start, or the resumed_as
    runs.start noted). None: a Resume that left neither."""
    if r.get("kind") in ("reunwrap", "unwrap"):
        return True
    if r.get("kind") != "resume":
        return False
    if own:
        return own.get("resumed_as") == "reunwrap" or own.get("start") == "unwrap"
    if r.get("resumed_as"):
        return r["resumed_as"] == "reunwrap"
    return None


def unfinished_reunwrap(project_dir, cfg=None) -> dict | None:
    """The last run was a Re-unwrap that was interrupted, failed or was
    stopped after it had started on the unwraps and before its MintPy run
    finished: {"how", "started", "in_mintpy"}; else None.

    Such a project holds new unwraps for the pairs the re-unwrap reached and
    the old ones for the rest (decision D4, 7 Oct 2026: Resume then suggests
    'Re-unwrap'). What the disk alone suggests is not always safe there:
    'MintPy onward' (after an earlier unfinished MintPy run left
    run_incomplete.json) loads the half re-unwrapped stack as it is, and
    'Products only' (killed before any pair's unwrap was rewritten) drops the
    re-unwrap without saying so. ('Phase linking onward' re-runs the same
    unwrap step, whose done-markers carry each pair's unwrap-config
    fingerprint, so it would redo the remaining pairs too; 'Re-unwrap' is
    the restart that says so.) Read from run.json and run_result.json (kind
    and state), never from log text. Not counted: a re-unwrap refused by its
    checks before it touched an unwrap (its refusal names the restart;
    suggesting it again would loop), and one whose MintPy run finished
    (run_products.json since it began): every pair then carries the new
    unwrap and only the products need building. A Resume that ran as the
    re-unwrap continues the one before it, so it counts from that one's start
    (or from the start runs.start noted, reunwrap_started, which survives a
    chain of Resumes). Called by the run itself (run_from's failure path,
    launch's Resume), the record is this process: it is ending, or about to
    resume."""
    P = Path(project_dir)
    rec = record(P)
    while rec and rec.get("kind") == "plan" and not rec.get("stopped"):
        rec = rec.get("previous")              # sizing is not a run
    if not rec or not rec.get("started"):
        return None
    res = result(P)
    own = res if _owns(res, rec) else None
    # this process's own run, not yet ended (a host that ran run_from in
    # process and outlived it has the run's end record: judged by that)
    me = rec.get("pid") == os.getpid() and own is None
    if rec.get("pid") != os.getpid():
        if alive(rec):
            return None                        # still going
        held = lock_holder(P)
        if held is not None and held.get("pid") != os.getpid() and not _lock_held_here(P):
            return None                        # a run started elsewhere holds the project
    if own and own.get("status") == "done":
        return None
    ran = _ran_reunwrap(rec, own)
    if ran is False:
        return None
    first = rec
    prev = rec.get("previous")
    if rec.get("kind") == "resume" and prev and prev.get("started"):
        prev_own = res if _owns(res, prev) else None
        if not (prev_own and prev_own.get("status") == "done") and _ran_reunwrap(prev, prev_own):
            first = prev                       # Resume runs the re-unwrap only to carry on the one before it
        elif ran is None:
            return None
    elif ran is None:
        return None
    # a Resume that ran as the re-unwrap, started by runs.start, noted when
    # the re-unwrap it carries on began (the run before the one it follows
    # is not kept in run.json)
    began = min([float(first["started"])] + [float(r["reunwrap_started"]) for r in (rec, first)
                                             if r.get("kind") == "resume" and r.get("reunwrap_started")])
    since = began - 1
    S = Path(getattr(cfg, "stack_dir", P / "stack"))
    M = Path(getattr(cfg, "mintpy_dir", P / "mintpy"))
    if _mtime(M / "run_products.json") >= since:
        return None                            # its MintPy finished: every pair carries the new unwrap
    # the unwrap step rewrites every pair's unwrap config before it unwraps
    # one; the sample-coherence step records each block it rewrites
    touched = (_mtime(S / "merged" / "pair_coherence.json") >= since
               or any(_mtime(c) >= since for c in (S / "configs").glob("config_igram_unw_*"))
               or any(_mtime(u) >= since for u in (S / "merged" / "interferograms").glob("*/filt_fine.unw")))
    if not touched:
        return None
    if me:
        how = "ended early"
    elif rec.get("stopped") or (own and own.get("status") == "stopped"):
        how = "was stopped"
    elif own:
        how = "failed"
    else:
        how = "was interrupted"
    return {"how": how, "started": began, "in_mintpy": _mtime(M / "run_incomplete.json") >= since}


def _guess_backend(P: Path) -> str:
    rec = record(P) or {}
    if rec.get("config"):
        try:
            from .config import load_config
            return load_config(rec["config"]).backend
        except Exception:                                    # noqa: BLE001
            pass
    if (P / "miaplpy").exists():
        return "miaplpy"
    if (P / "stack").exists():
        return "stack"
    return "disp" if (P / "disp").exists() else "stack"


def suggest_kind(project_dir, cfg=None) -> tuple[str, str]:
    """The restart that fits what is on disk: (kind, plain reason).

    Most advanced finished stage first. Every kind it returns is safe to run
    on that state: the stage chains skip work verified on disk. On the
    phase-linked route a Re-unwrap that was interrupted, failed or was
    stopped part-way (unfinished_reunwrap: run.json / run_result.json) is
    restarted as 'Re-unwrap', whatever the files alone would suggest."""
    P = Path(project_dir)
    backend = getattr(cfg, "backend", None) or _guess_backend(P)
    S = Path(getattr(cfg, "stack_dir", P / "stack"))
    M = Path(getattr(cfg, "mintpy_dir", P / "mintpy"))
    vel = max((M / "velocity.h5", M / "geo" / "geo_velocity.h5"), key=_mtime)
    ifg = M / "inputs" / "ifgramStack.h5"
    if backend == "disp":
        if vel.exists():
            return "products", "the time series is finished; only the products need building"
        return "full", "the DISP-S1 time series is not finished; downloads already on disk are kept"
    if backend == "miaplpy":
        # before anything the disk says (decision D4): a half-done re-unwrap
        # looks like 'phase linking is done', or, after an earlier failed
        # MintPy run, 'MintPy onward' (which would load the old unwraps of the
        # pairs it had not reached), or, killed before it rewrote an unwrap,
        # 'everything is finished' (which would drop it unannounced)
        try:
            ru = unfinished_reunwrap(P, cfg)
        except Exception:                                    # noqa: BLE001 - the disk reading still holds
            ru = None
        if ru:
            where = (" during MintPy" if ru["in_mintpy"] else
                     "" if ru["how"] == "ended early" else " before it finished")
            return "reunwrap", (f"the last re-unwrap {ru['how']}{where}; 'Re-unwrap' carries on, keeping the "
                                "pairs it already re-unwrapped, so no pair is left with its old unwrap")
    unw = unwrap_status(S)
    missing = unw["missing"]
    if backend == "stack":
        # the stack route carries on past a pair whose unwrap failed (its
        # output is quarantined, filt_fine.int stays): a pair quarantined
        # before the last MintPy run is not unfinished work
        qdir = S / "quarantine_unwrap_timeout"
        q = {d.name for d in qdir.iterdir() if d.is_dir()} if qdir.exists() else set()
        missing = [p for p in missing if p not in q or _mtime(qdir / p / "REASON.txt") > _mtime(vel)]
    # every pair, not a sample: a re-unwrap rewrites pairs in name order, so
    # the newest are often not the last names
    newest_unw = max((_mtime(S / "merged" / "interferograms" / p / "filt_fine.unw") for p in unw["unwrapped"]),
                     default=0.0)
    # On these routes a MintPy run moves the previous products aside, marks
    # itself unfinished (run_incomplete.json) and writes run_products.json
    # once every product is verified. velocity.h5 is written before geocoding
    # and the checks, so a run that died after it is NOT finished.
    record_route = backend in ("stack", "miaplpy")
    if (M / "run_incomplete.json").exists() and not missing:
        if record_route:
            return "mintpy", "MintPy started but did not finish; its partial products are not used"
        return "full", "MintPy started but did not finish; its partial products are not used"
    if vel.exists() and not missing and _mtime(vel) >= max(_mtime(ifg), newest_unw) \
            and (not record_route or _mintpy_finished(M)):
        if _mtime(P / "deliverables" / "client_report.json") >= _mtime(vel):
            return "products", "everything is finished; 'Products only' rebuilds the report and maps"
        return "products", "MintPy finished; the products and report need (re)building"
    if ifg.exists() and unw["exported"] and not missing and _mtime(ifg) >= newest_unw:
        return "mintpy", "the interferograms are unwrapped and loaded; MintPy did not finish"
    if backend == "miaplpy":
        W = P / "miaplpy"
        if (W / "inverted" / "phase_series.h5").exists():
            if unw["missing"]:
                return "miaplpy", (f"phase linking is done; {len(unw['missing'])} of {len(unw['exported'])} "
                                   "exported pairs still need unwrapping (finished ones are kept)")
            return "miaplpy", "phase linking is done; export, unwrapping and MintPy pick up from there"
        cs = coregistration_status(S)
        if (W / "inputs" / "slcStack.h5").exists() and not cs["missing"]:
            return "miaplpy", "the SLC stack is loaded; phase linking did not finish"
        if cs["expected"] and not cs["missing"]:
            return "miaplpy", f"coregistration is complete ({len(cs['merged'])} dates merged); phase linking is next"
        if cs["expected"]:
            note = ""
            if cfg is not None:
                try:
                    from . import stack_backend as sb
                    if sb.coreg_started(cfg):
                        # data_acquisition skips the dates already coregistered
                        # (stack_backend.coregistered_dates, .cleaned included) and
                        # refuses archive dates the existing plan does not hold
                        note = (f"; {len(sb.coregistered_dates(cfg, include_reference=False))} dates are "
                                "coregistered and are not downloaded again. If the archive now holds dates "
                                "outside the plan the run stops and names them (accept them with "
                                "allow_missing_dates, or start a new project)")
                except Exception:                            # noqa: BLE001 - the plain reason is still right
                    note = ""
            return "full", (f"coregistration is not finished ({len(cs['merged'])} of {len(cs['expected'])} "
                            "dates merged); finished dates are kept" + note)
        return "full", "nothing processed yet"
    if ifg.exists() or unw["unwrapped"]:
        return "full", "unwrapping is not finished; finished pairs are skipped"
    if (S / "configs").exists():
        return "full", "the stack is part-way; finished commands are skipped"
    return "full", "nothing processed yet"


# ---------------------------------------------------------------- projects
@dataclass
class Project:
    config: str
    name: str
    backend: str = "?"
    project_dir: str = ""
    state: str = "idle"
    verdict: str = ""
    areas: list = field(default_factory=list)
    updated: float = 0.0
    error: str = ""


def _remember(cfg_path) -> None:
    try:
        st = json.loads(REGISTRY.read_text()) if REGISTRY.exists() else {}
    except ValueError:
        st = {}
    lst = [c for c in st.get("configs", []) if c != str(cfg_path)]
    st["configs"] = [str(cfg_path)] + lst[:49]
    st["config"] = str(cfg_path)
    REGISTRY.write_text(json.dumps(st, indent=1))


def forget(cfg_path) -> bool:
    """Take a config off the remembered list (storage.delete_project, after
    its project folder is deleted). The yaml itself is not touched: a config
    in the configs folder still shows on the projects screen, ready to run
    again. True when it was on the list."""
    try:
        st = json.loads(REGISTRY.read_text()) if REGISTRY.exists() else {}
    except (OSError, ValueError):
        return False
    want = {str(cfg_path), str(Path(cfg_path).resolve())}
    lst = st.get("configs", [])
    keep = [c for c in lst if c not in want and str(Path(c).resolve()) not in want]
    hit = len(keep) != len(lst) or st.get("config") in want
    if not hit:
        return False
    st["configs"] = keep
    if st.get("config") in want:
        st["config"] = keep[0] if keep else ""
    REGISTRY.write_text(json.dumps(st, indent=1))
    return True


def known_configs() -> list[Path]:
    seen, out = set(), []
    try:
        reg = json.loads(REGISTRY.read_text()).get("configs", []) if REGISTRY.exists() else []
    except ValueError:
        reg = []
    from .settings import configs_dir
    cands = ([Path(c) for c in reg] + sorted(PKG.glob("config_*.yaml")) + sorted((PKG / "configs").glob("*.yaml"))
             + sorted(configs_dir().glob("*.yaml")))
    for c in cands:
        c = c.resolve()
        if c in seen or not c.exists() or c.name.endswith("_example.yaml"):   # templates, not projects
            continue
        seen.add(c); out.append(c)
    return out


def project_summary(cfg_path) -> Project:
    from .config import load_config
    from . import combine
    p = Project(config=str(cfg_path), name=Path(cfg_path).stem.replace("config_", ""))
    if combine.is_combined_config(cfg_path):
        return _combined_summary(p, cfg_path)
    try:
        cfg = load_config(str(cfg_path))
    except Exception as exc:                                 # noqa: BLE001
        p.error = str(exc).splitlines()[0][:120]
        p.state = "invalid"
        return p
    P = Path(cfg.project_dir)
    p.backend, p.project_dir = cfg.backend, str(P)
    p.state = state(P)
    cr = P / "deliverables" / "client_report.json"
    if cr.exists():
        try:
            d = json.loads(cr.read_text())
            p.verdict = d.get("verdict", "")
            p.areas = [(a["name"], a.get("vertical_mm_yr"), a.get("vertical_uncertainty_1sigma_mm_yr"))
                       for a in d.get("areas", [])]
        except (ValueError, KeyError):
            pass
    stamps = [f.stat().st_mtime for f in (P / progress.FILENAME, cr, P / RUNFILE) if f.exists()]
    p.updated = max(stamps) if stamps else 0.0
    return p


def _combined_summary(p: Project, cfg_path) -> Project:
    """A combined project (combine.py: ascending + descending of the same
    ground) in the project list: backend "combined", state done when its
    combined_report.json exists, the verdict saying whether it is current."""
    from . import combine
    try:
        cc = combine.load(cfg_path)
    except combine.CombinedConfigError as exc:
        p.error, p.state = str(exc).splitlines()[0][:120], "invalid"
        return p
    p.backend, p.project_dir = "combined", str(cc.project_dir)
    rp = cc.deliverables / combine.REPORT
    if not rp.exists():
        p.state = "idle"
        return p
    p.state = "done"
    try:
        d = json.loads(rp.read_text())
        why = combine.stale_reason(cfg_path)
        p.verdict = "COMBINED" if why is None else f"COMBINED, OUT OF DATE: {why}"
        p.areas = [(a["name"], (a.get("vertical") or {}).get("vertical_mm_yr"), None) for a in d.get("areas", [])]
    except (ValueError, KeyError, OSError):
        pass
    p.updated = rp.stat().st_mtime
    return p


def tail(path, n: int = 200) -> list[str]:
    """The last n lines of a log without reading a multi-GB file whole."""
    try:
        with open(path, "rb") as fh:
            fh.seek(0, os.SEEK_END)
            size = fh.tell()
            block = min(size, max(8192, n * 240))
            fh.seek(size - block)
            data = fh.read().decode(errors="replace")
    except OSError:
        return []
    return data.splitlines()[-n:]
