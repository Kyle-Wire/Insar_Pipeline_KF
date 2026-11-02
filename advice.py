"""Plain-language cause and next step for a run that stopped.

A run takes a day and nobody watches it overnight. When it fails, the
dashboard used to show `repr(exc)[:200]` cut to 60 characters (Site B,
29 Sep: "ValueError: could not convert string to float: ''") and the reader
had to know the code to decide what to do. explain() turns the exception and
the stage it happened in into one sentence a coworker can act on, plus the
restart to use, for <project>/run_result.json, the dashboard header and the
desktop notification.

Rules are matched in order, first hit wins; each comes from a failure this
pipeline has actually had, or one its launchers can raise. The raw message is
always kept beside the plain one (run_result.json "technical").
"""
from __future__ import annotations

import errno
import shutil
from pathlib import Path

# The restart kinds as the dashboard words them (runs.KINDS labels, shortened).
KIND_WORDS = {
    "resume": "'Resume'",
    "full": "'Whole pipeline'",
    "miaplpy": "'Phase linking onward'",
    "reunwrap": "'Re-unwrap'",
    "mintpy": "'MintPy onward'",
    "products": "'Products only'",
}


def _chain(exc: BaseException):
    """The exception and its causes, outermost first (a wrapped OSError
    still says 'no space left')."""
    seen = []
    while exc is not None and exc not in seen and len(seen) < 6:
        seen.append(exc)
        exc = exc.__cause__ or exc.__context__
    return seen


def _text(exc: BaseException) -> str:
    return " | ".join(f"{type(e).__name__}: {e}" for e in _chain(exc))


def _first_line(exc: BaseException, n: int = 300) -> str:
    s = str(exc).strip().splitlines()
    return (s[0] if s else type(exc).__name__)[:n]


def _free_gb(path) -> float | None:
    """The real disk budget when settings provides it (WSL: C: free + what
    the vhdx holds on C: that ext4 does not use), else the filesystem's free
    space."""
    if path is None:
        return None
    try:
        from . import settings
        fn = getattr(settings, "disk_budget_gb", None)
        if fn is not None:
            return float(fn(path)["budget"])
    except Exception:                                        # noqa: BLE001
        pass
    try:
        return shutil.disk_usage(path).free / 1e9
    except OSError:
        return None


def mintpy_error_tail(mintpy_dir, n: int = 6) -> str:
    """The last error lines of MintPy's own log (Site B's ran to 4,663
    lines; the reason is at the end)."""
    f = Path(mintpy_dir) / "mintpy_error.log" if mintpy_dir else None
    if f is None or not f.exists():
        return ""
    try:
        from .runs import tail
        lines = [ln for ln in tail(f, 400) if ln.strip()]
    except Exception:                                        # noqa: BLE001
        return ""
    hits = [ln for ln in lines if any(k in ln for k in ("Error", "error", "Exception", "Traceback"))]
    keep = (hits or lines)[-n:]
    return "\n".join(ln.strip()[:240] for ln in keep)


def explain(exc: BaseException, stage: str | None = None, cfg=None, restart: str | None = None) -> dict:
    """{"plain_cause", "next_action", "category", "restart_kind", "technical"}.

    `restart` is the kind runs.suggest_kind() picked from what is on disk; it
    is quoted in next_action where restarting is the remedy."""
    stage = stage or "the run"
    txt = _text(exc)
    low = txt.lower()
    proj = getattr(cfg, "project_dir", None)
    again = KIND_WORDS.get(restart or "resume", KIND_WORDS["resume"])
    out = {"category": "unknown", "restart_kind": restart or "resume", "technical": txt[:2000]}

    def done(category, cause, action):
        out.update(category=category, plain_cause=cause, next_action=action)
        return out

    # ---- stopped on purpose
    if isinstance(exc, KeyboardInterrupt) or (isinstance(exc, SystemExit) and exc.code in (130, 143, 129)):
        return done("stopped", f"The run was stopped during {stage} (Stop button, Ctrl-C or a shutdown signal). "
                               "Its running commands were stopped with it and finished work is kept.",
                    f"Restart with {again} when ready.")
    # ---- another run holds the project
    if "already running" in low:
        return done("busy", f"Another run is already working on this project ({_first_line(exc)}).",
                    "Wait for it to finish, or stop it from the dashboard, then start again.")
    # ---- weather delays missing for some dates (corrections.verify_era5_coverage)
    from .corrections import WeatherCoverageError
    weather = next((e for e in _chain(exc) if isinstance(e, WeatherCoverageError)), None)
    if weather is not None:
        return done("weather", weather.cause, weather.action[:1].upper() + weather.action[1:])
    # ---- disk
    if any(getattr(e, "errno", None) == errno.ENOSPC for e in _chain(exc)) or "no space left" in low \
            or "out of disk" in low:
        free = _free_gb(proj)
        return done("disk", f"The disk filled up during {stage}"
                            + (f" (about {free:.0f} GB free now)" if free is not None else "") + ".",
                    "Free space (Files screen: trash, finished projects' intermediates via `insar cleanup`, "
                    f"or Windows C:), then restart with {again}. Finished work is kept.")
    if "projected peak" in low or "disk budget" in low:
        return done("disk", _first_line(exc, 400),
                    "Free disk space or shorten the date range, then start again. Pass --force only if "
                    "you are batching downloads and deleting SAFEs as you go.")
    # ---- memory
    if any(isinstance(e, MemoryError) for e in _chain(exc)) or "cannot allocate memory" in low \
            or "out of memory" in low or "oom-kill" in low:
        return done("memory", f"The computer ran out of memory during {stage}.",
                    "Close other programs or wait for another project's run to finish, then restart with "
                    f"{again}. If it happens again at the same step, lower the parallel jobs for that step "
                    "in the config (stack_num_proc*, miaplpy_num_processors).")
    # ---- logins and licences
    if "netrc" in low or "earthdata" in low or "urs.earthdata" in low or ("401" in low and "unauthor" in low):
        return done("login", "The NASA Earthdata login is missing or was refused, so the radar data could not be downloaded.",
                    "Put a working urs.earthdata.nasa.gov login in ~/.netrc (`insar doctor` checks it), "
                    f"then restart with {again}.")
    if "cdsapirc" in low or ("cdsapi" in low or "copernicus" in low or "era5" in low) \
            and any(k in low for k in ("403", "licen", "forbidden", "credential", "401")):
        return done("weather", "The ERA5 weather data could not be fetched: the Copernicus CDS key in ~/.cdsapirc "
                               "is missing or wrong, or the ERA5 licence has not been accepted on the CDS website.",
                    "Fix the key, accept the licence (`insar doctor` names which), then restart with "
                    f"{again}.")
    # ---- network
    net_types = ("ConnectionError", "Timeout", "TimeoutError", "gaierror", "RemoteDisconnected",
                 "ChunkedEncodingError", "IncompleteRead", "ProtocolError", "MaxRetryError", "SSLError")
    if any(any(t in type(e).__name__ for t in net_types) for e in _chain(exc)) \
            or any(k in low for k in ("max retries exceeded", "temporary failure in name resolution",
                                      "connection reset", "connection aborted", "timed out", "503 server error",
                                      "502 server error", "504 server error", "network is unreachable")):
        return done("network", f"The internet connection or a NASA/ESA data server dropped during {stage}.",
                    f"Restart with {again}; downloads already finished are kept. If it keeps happening, "
                    "the data server may be down: try again in a few hours.")
    # ---- the pipeline's own known failure messages
    if "disp-s1 stitching is not usable" in low:
        # disp_backend.check_stitch: partial corrections, a lost anchor granule
        # or a coverage ratchet - each named in the message with its remedy
        return done("disp_stitch", "The OPERA DISP-S1 granules could not be stitched into one record: "
                                   + _first_line(exc, 400).split("\n")[0].rstrip(":") + ". "
                                   "The measured record is in mintpy/disp_chain.json (status 'stitch refused').",
                    f"Restart with {again}: granule reads are cached, and a granule that failed to read is "
                    "often read the next time. If the same date fails again: set disp_apply_corrections: false "
                    "(when the problem is partly applied corrections) or list the date in allow_missing_dates "
                    "(a lost anchor granule; its offset is then accepted), as the message says.")
    if "refusing to write disp-s1 products into" in low:
        return done("refused", "This project folder holds stack or phase-linked products, so the DISP-S1 "
                               "screening was not written into it. Nothing was changed.",
                    "Use a separate project for the DISP screening (the builder names it "
                    "<name>_disp_screening), then start it there.")
    if "network is not connected" in low or "not connected - it falls" in low:
        return done("science", "The interferogram network fell apart into separate pieces after unwrapping, "
                               "so the time series cannot be tied together.",
                    "This needs Kyle: send him the log. The message names the pieces and the dates on the small side.")
    if "all" in low and "commands failed" in low:
        return done("state", f"Every command in {stage} failed: usually inputs that were cleaned away while "
                             "the record of finished work survived.",
                    "Send Kyle the log; do not restart repeatedly, it will fail the same way.")
    if "miaplpy" in low and ("not the patched" in low or "patch" in low and "verif" in low):
        return done("install", "The phase-linking software (MiaplPy) is not the patched build this pipeline needs.",
                    "Run patches/miaplpy/build_miaplpy.sh (see install/README.md), then restart with "
                    f"{again}.")
    if "mintpy" in low and ("failed" in low or "mintpy_error.log" in low):
        tail = mintpy_error_tail(getattr(cfg, "mintpy_dir", None))
        return done("mintpy", "The time-series step (MintPy) failed."
                              + (f" Its log ends with:\n{tail}" if tail else ""),
                    f"If the log names a missing file or a network error, restart with {again}; "
                    "otherwise send Kyle the log and mintpy/mintpy_error.log.")
    if "miaplpy step" in low:
        return done("miaplpy", f"A phase-linking step failed: {_first_line(exc)}",
                    f"Restart with {again}; if it fails at the same step again, send Kyle the log.")
    if "coregistration" in low and "missing" in low or "merged" in low and "missing" in low:
        return done("precondition", _first_line(exc, 500),
                    f"Restart with {again}; it finishes the coregistration first.")
    if isinstance(exc, SystemExit) and isinstance(exc.code, str):
        return done("refused", exc.code.strip().splitlines()[0][:500],
                    "Change what the message asks for, then start again.")
    if isinstance(exc, FileNotFoundError):
        what = (getattr(exc, "filename", "") or _first_line(exc)).rstrip(".")
        return done("missing", f"A file the run needs is missing during {stage}: {what}.",
                    f"Restart with {again} (it rebuilds what is missing from the step before). If the same "
                    "file is reported again, send Kyle the log.")
    return done("unknown", f"The run stopped with an error during {stage}: {_first_line(exc).rstrip('.')}.",
                f"Look at the end of the log. Restart with {again} once the cause is fixed; if it fails "
                "the same way twice, send Kyle the log.")
