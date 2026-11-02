"""
Child-process teardown shared by every module that launches long subprocesses.

Every command runs in its own session so it can be killed as a GROUP.

WHY: the runner starts /usr/bin/time -> bash -> SentinelWrapper.py -> snaphu,
or smallbaselineApp.py -> its own subprocess.run children. Killing the parent
Python process does NOT reach any of those - they are reparented to PID 1 and
keep running. That happened on the pilot: 26 orphaned unwraps from runs
killed up to 100 minutes earlier were still going, load 33 on 24 cores,
reading input files that had since been deleted and poised to write 8x2
output into directories that by then held an 11x3 stack.

The MintPy launcher was missed by the first fix, which is why HANDOFF.md had
to give a manual process-tree walk for stopping it. It registers here now.
"""
import atexit
import json
import logging
import os
import re
import shlex
import signal
import subprocess
import threading
import time

log = logging.getLogger(__name__)

_CHILD_PGIDS: dict = {}           # pgid -> kernel start ticks (None if it could not be read)
# Re-entrant: kill_children runs inside a signal handler on the main thread,
# which may be interrupted while that same thread is inside register().
_CHILD_LOCK = threading.RLock()
# Two flags, not one. The single _INSTALLED flag was set BEFORE the
# main-thread check, so when the first caller was a worker thread (on every
# run that downloads anything, data_acquisition._assemble inside a
# ThreadPoolExecutor) the atexit hook went in, the signal handlers did not,
# and every later main-thread call returned early. Measured 5 Oct 2026: after
# two pool-thread calls and one main-thread call SIGTERM was still SIG_DFL, a
# Stop killed the runner outright (rc -15, no atexit) and its snaphu /
# SentinelWrapper / MintPy children, each in its own session, kept running.
_ATEXIT_INSTALLED = False
_SIGNALS_INSTALLED = False
_INSTALL_LOCK = threading.Lock()
_STOPPING = threading.Event()      # set by the teardown; launchers must check it
# Optional <project>/.run_children.json mirror of the registered groups, so a
# runner that dies without its teardown (SIGKILL, OOM kill, WSL shutdown)
# leaves a record runs.stop / runs.start can use to stop its orphans.
_CHILD_FILE = None


def stopping() -> bool:
    return _STOPPING.is_set()


def clear_stopping() -> None:
    """For tests and for a launcher that deliberately survives a teardown."""
    _STOPPING.clear()


def start_ticks(pid: int) -> int | None:
    """Kernel start time of a live, non-zombie process in clock ticks
    (/proc/<pid>/stat field 22), or None. A PID alone is reused within days;
    PID + start ticks identifies one process."""
    try:
        stat = open(f"/proc/{int(pid)}/stat").read()
    except (OSError, ValueError):
        return None
    rest = stat[stat.rindex(")") + 2:].split()
    if rest[0] in ("Z", "X"):
        return None
    return int(rest[19])


def track_children_in(path) -> None:
    """Mirror the registered groups into `path` (JSON) on every change; None stops it."""
    global _CHILD_FILE
    with _CHILD_LOCK:
        _CHILD_FILE = None if path is None else str(path)
        _write_child_file()


def _write_child_file() -> None:
    """Caller holds _CHILD_LOCK. Atomic replace; a failure never stops a run."""
    if _CHILD_FILE is None:
        return
    rec = {"runner_pid": os.getpid(), "runner_start_ticks": start_ticks(os.getpid()),
           "groups": {str(pg): t for pg, t in _CHILD_PGIDS.items()}}
    tmp = f"{_CHILD_FILE}.{os.getpid()}.tmp"
    try:
        with open(tmp, "w") as fh:
            json.dump(rec, fh)
        os.replace(tmp, _CHILD_FILE)
    except OSError:
        pass


def register(pid: int) -> None:
    """Record a process started with start_new_session=True (pgid == pid)."""
    with _CHILD_LOCK:
        _CHILD_PGIDS[pid] = start_ticks(pid)
        _write_child_file()


def unregister(pid: int) -> None:
    with _CHILD_LOCK:
        _CHILD_PGIDS.pop(pid, None)
        _write_child_file()


def registered() -> set:
    with _CHILD_LOCK:
        return set(_CHILD_PGIDS)


def recorded_groups(path) -> dict:
    """{pgid: start_ticks} from a .run_children.json, keeping only groups whose
    leader is still the SAME process (start ticks match). {} if none."""
    try:
        rec = json.loads(open(path).read())
    except (OSError, ValueError):
        return {}
    out = {}
    for pg, t in (rec.get("groups") or {}).items():
        try:
            pg = int(pg)
        except ValueError:
            continue
        now = start_ticks(pg)
        if now is not None and t is not None and now == t:
            out[pg] = now
    return out


def group_alive(pgid: int) -> bool:
    """Any LIVE process left in the group. Zombies (dead, waiting for their
    parent to reap them) do not count: they hold no CPU, memory or files, and
    signal 0 would report them alive until reaped."""
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    try:
        names = os.listdir("/proc")
    except OSError:
        return True
    for n in names:
        if not n.isdigit():
            continue
        try:
            stat = open(f"/proc/{n}/stat").read()
        except OSError:
            continue
        rest = stat[stat.rindex(")") + 2:].split()
        if len(rest) > 2 and rest[2] == str(pgid) and rest[0] not in ("Z", "X"):
            return True
    return False


def kill_groups(pgids, grace_s: float = 3.0) -> list:
    """SIGTERM, wait up to grace_s, SIGKILL the given process groups. Returns
    the groups that STILL have a process afterwards (checked, not assumed).
    Callers verify each group's identity (start ticks) before passing it."""
    pgids = list(pgids)
    for pg in pgids:
        try:
            os.killpg(pg, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass
    t0 = time.time()
    while pgids and time.time() - t0 < grace_s and any(group_alive(pg) for pg in pgids):
        time.sleep(0.1)
    for pg in pgids:
        if group_alive(pg):
            try:
                os.killpg(pg, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
    t0 = time.time()
    while pgids and time.time() - t0 < 2.0 and any(group_alive(pg) for pg in pgids):
        time.sleep(0.05)        # SIGKILL is asynchronous; give the kernel a moment
    return [pg for pg in pgids if group_alive(pg)]


def kill_children(*_, grace_s: float = 3.0) -> int:
    """SIGTERM every registered group, wait, SIGKILL. Returns groups signalled.

    Also raises the stopping flag FIRST: a ThreadPoolExecutor being torn down
    still runs its queued futures, and each of those would launch a fresh
    child after this has already fired (six SentinelWrapper jobs survived a
    stop that way on the P2 run). Launchers refuse while the flag is set."""
    _STOPPING.set()
    with _CHILD_LOCK:
        pgids = list(_CHILD_PGIDS)
    if pgids:
        # waits only while a group is still alive (it used to sleep the full
        # grace, twice: once in the signal handler and again at exit)
        survivors = kill_groups(pgids, grace_s=grace_s)
        with _CHILD_LOCK:
            for pg in pgids:
                if pg not in survivors:
                    _CHILD_PGIDS.pop(pg, None)
            _write_child_file()
        log.warning("Stopped %d running command(s) and their children.", len(pgids) - len(survivors))
        if survivors:
            log.error("Could not stop command group(s) %s", survivors)
    return len(pgids)


def install_teardown() -> None:
    """atexit (any thread) + SIGTERM/SIGINT/SIGHUP handlers (main thread
    only). Idempotent. A call from a worker thread installs the atexit hook
    and leaves the signal handlers for the first MAIN-thread call - the two
    are tracked separately (see _SIGNALS_INSTALLED). Every launcher calls this
    first (pipeline.run_from / runs.guarded), so the handlers are in place
    before any worker thread starts a child."""
    global _ATEXIT_INSTALLED, _SIGNALS_INSTALLED
    with _INSTALL_LOCK:
        if not _ATEXIT_INSTALLED:
            _ATEXIT_INSTALLED = True
            atexit.register(kill_children)
        if _SIGNALS_INSTALLED:
            return
        if threading.current_thread() is not threading.main_thread():
            log.debug("install_teardown from a worker thread: signal handlers wait for the main thread")
            return
        _SIGNALS_INSTALLED = True
        for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
            prev = signal.getsignal(sig)

            def handler(signum, frame, _prev=prev):
                kill_children()
                # Re-raise into the previous handler so KeyboardInterrupt and any
                # host-installed handler still fire; otherwise exit by signal
                # convention. (The old code raised SystemExit after calling a
                # default handler that itself raises, so that line never ran.)
                if callable(_prev) and _prev not in (signal.SIG_DFL, signal.SIG_IGN):
                    _prev(signum, frame)
                raise SystemExit(128 + signum)
            signal.signal(sig, handler)


def signals_installed() -> bool:
    return _SIGNALS_INSTALLED


# A conda environment's name goes into the `bash -c` command lines that
# activate it (miaplpy_backend: miaplpy_env; stack_backend: stack_conda_env),
# so a value from a config cannot be allowed to carry shell syntax. A name is
# what `conda create -n` makes in practice (letters, digits, '_', '.', '-',
# not starting with '-'); an absolute path is taken as the environment's
# folder (version._env_prefix reads one too) and quoted. Anything else is
# refused before a command is built (16 Oct 2026, readiness review SEC-02 /
# RUN-04: the names were pasted into the command unquoted).
_CONDA_ENV_NAME = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]*")


def conda_env_arg(env, setting: str = "the conda environment") -> str:
    """`env` ready for a shell command line: a valid name as it is, an
    absolute folder path quoted. Raises ValueError, in words, otherwise."""
    s = "" if env is None else str(env)
    if _CONDA_ENV_NAME.fullmatch(s):
        return s
    if s.startswith("/") and "\n" not in s and "\0" not in s:
        return shlex.quote(s)
    raise ValueError(f"{setting} = {s!r} is not a conda environment name: use only letters, digits, '_', '.' "
                     "and '-' (the name `conda env list` shows), or the environment's full folder path")


def popen_group(argv, **kwargs) -> subprocess.Popen:
    """Popen in a new session and register it for teardown."""
    if _STOPPING.is_set():
        raise RuntimeError("shutting down - refusing to launch a new child")
    kwargs.setdefault("start_new_session", True)
    proc = subprocess.Popen(argv, **kwargs)
    register(proc.pid)
    return proc


def run_group(argv, timeout: float | None = None, **kwargs) -> subprocess.CompletedProcess:
    """
    subprocess.run equivalent whose whole process tree is killed on teardown
    or timeout. Returns CompletedProcess; returncode -9 on timeout.
    """
    kwargs.setdefault("stdout", subprocess.PIPE)
    kwargs.setdefault("stderr", subprocess.PIPE)
    kwargs.setdefault("text", True)
    proc = popen_group(argv, **kwargs)
    try:
        try:
            out, err = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            out, err = proc.communicate()
            proc.returncode = -9
    finally:
        unregister(proc.pid)
    return subprocess.CompletedProcess(argv, proc.returncode, out, err)
