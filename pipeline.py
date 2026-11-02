"""Main entry point: wires all stages together.

Usage:
    python -m insar_pipeline.pipeline config.yaml [--resume | --start STAGE]

Backends (config `backend`), with what their numbers are (config.ROUTE_TAG):
    miaplpy  phase-linked distributed scatterers on the topsStack
             coregistration (miaplpy_backend.py). OFFICIAL results.
    stack    topsStack: one coregistration for the whole archive, snaphu with
             its cost optimisation actually running (stack_backend.py).
             Comparison only.
    disp     OPERA DISP-S1 ready-made time series. Screening only: no client
             report or KMZ unless disp_client_report: true.
    local    topsApp per pair. Testing only.
    hyp3     ASF HyP3 interferograms. Comparison only.
    nisar    Pre-made NISAR L2 GUNW products. Comparison only.

Purpose (config `purpose`, V3 8 Oct 2026): "client" (default) ends the
products stage with the GNSS tie, the client report and the release status;
"experiment" (a validation run) ends it with the validation harness
(validation.py, when validation_protocol is set) and never builds a client
report, KMZ, release status or package.

Run control (review 5 Oct 2026). Every launcher - this module's main, the
dashboard's run kinds and the scripts/ runners - goes through launch() /
run_from(), which own, in one place:
  * the child teardown (procs.install_teardown from the main thread, so a
    Stop reaches snaphu / SentinelWrapper / MintPy in their own sessions)
    and the per-project lock (runs.project_lock) with run.json;
  * one stage chain per start point (full / miaplpy / unwrap / mintpy /
    products) with its preconditions, a progress event for every stage, the
    route overrides, and a non-zero exit when any stage failed;
  * <project>/run_result.json {status, kind, stage, plain_cause, next_action,
    log, log_tail, finished} in plain words (advice.explain) and a desktop
    notification (notify.send) when an unattended run ends;
  * the disk gate before a stack/phase-linked run and a disk watchdog during it.
Five launchers used to copy the chain and had drifted apart in failure
reporting, events, exit codes and settings (ops-12).

The local backend's modules (isce_processing, tropo_correction, pair_selection)
are imported inside its branch only: importing ISCE at module load made every
`python -m insar_pipeline.pipeline` under the wrong interpreter fail before
the config was even read.
"""
import _thread
import argparse
import errno
import logging
import os
import shutil
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

from .config import BACKENDS, load_config
from .deliverables import export_all

def _log_level() -> int:
    """INSAR_LOG_LEVEL (DEBUG, INFO, WARNING, ERROR; default INFO): the run log's level (ERR-03, 2 Nov 2026)."""
    from .runs import log_level
    return log_level()


logging.basicConfig(level=_log_level(), format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# Stage chains run_from can start at. Aliases: "download" is the whole chain
# (downloads skip what is on disk), "coreg" the whole chain without the
# download step, "reunwrap" the re-unwrap chain.
STARTS = ("full", "miaplpy", "unwrap", "mintpy", "products")
START_ALIASES = {"download": "full", "coreg": "full", "reunwrap": "unwrap", "phase_linking": "miaplpy"}

# Disk watchdog during a run: warn below DISK_WARN_GB of real budget, stop the
# run cleanly below DISK_STOP_GB (two samples in a row). When C: fills, the
# vhdx cannot grow and ext4 throws I/O errors mid-write, which can corrupt
# in-flight HDF5 files of EVERY project on the WSL disk, not just this run.
DISK_WARN_GB = 60.0
DISK_STOP_GB = 20.0
DISK_SAMPLE_S = 60.0

def route_tag(cfg) -> str:
    """'official' | 'screening' | 'comparison' (config.ROUTE_TAG)."""
    from .config import route_tag as _rt
    return _rt(cfg)


def is_screening(cfg) -> bool:
    from .config import is_screening as _scr
    return bool(_scr(cfg))


def is_experiment(cfg) -> bool:
    """A validation run (config purpose: experiment): config.is_experiment."""
    from .config import is_experiment as _exp
    return bool(_exp(cfg))


def refuse_screening_folder(cfg) -> None:
    """The stack and phase-linked runners refuse a project folder that holds a
    DISP-S1 screening (mintpy/disp_chain.json without an ifgramStack): run
    after the screening, they would leave its layers and records beside the
    official products. disp_backend.refuse_foreign_mintpy is the mirror guard
    for a screening written into an official folder (disp-3)."""
    m = Path(cfg.mintpy_dir)
    if (m / "disp_chain.json").exists() and not (m / "inputs" / "ifgramStack.h5").exists():
        raise SystemExit(f"Refusing to run the {cfg.backend} route in {cfg.project_dir}: the folder holds a "
                         f"DISP-S1 screening ({m / 'disp_chain.json'}). Nothing was changed. Use a fresh "
                         "project_dir for the official run (the builder names screenings <name>_disp_screening).")


def warn_route(cfg) -> str:
    """Log a WARNING for any route other than the official phase-linked one,
    at the start of the run where the log reader sees it first. The other
    backends are still reachable from the CLI with no other sign that their
    numbers are not the official ones."""
    tag = route_tag(cfg)
    if tag != "official":
        log.warning("Backend %r is the %s route: its results are for %s only. Official results come from "
                    "the phase-linked route (backend: miaplpy).", cfg.backend, tag.upper(),
                    "screening" if tag == "screening" else "comparison")
    return tag


# ---------------------------------------------------------------- disk
def disk_budget(path) -> tuple:
    """(GB a run under `path` can really write, where the figure comes from).
    settings.disk_budget_gb counts WSL's real budget: C: free + what the vhdx
    already holds on C: that ext4 does not use. ext4's own free space
    overstated it by ~320 GB on 5 Oct 2026, and the vhdx's length (the
    file is sparse) by ~98 GB on 7 Oct 2026. The gate (disk_gate) and the
    watchdog (DiskWatchdog) both read this figure."""
    try:
        from . import settings
        fn = getattr(settings, "disk_budget_gb", None)
        if fn is not None:
            d = fn(path)
            return float(d["budget"]), str(d.get("source", "settings.disk_budget_gb"))
    except Exception as exc:                                 # noqa: BLE001
        log.debug("disk_budget_gb unavailable: %s", exc)
    p = Path(path)
    while not p.exists() and p != p.parent:
        p = p.parent
    return shutil.disk_usage(p).free / 1e9, "filesystem free space"


def project_gb(project_dir) -> float:
    """Bytes already allocated under the project, in GB (st_blocks, so sparse
    files count what they hold). 0.2 s for Site B's 205 GB / 32k files."""
    tot, stack = 0, [str(project_dir)]
    while stack:
        d = stack.pop()
        try:
            it = os.scandir(d)
        except OSError:
            continue
        with it:
            for e in it:
                try:
                    if e.is_dir(follow_symlinks=False):
                        stack.append(e.path)
                    else:
                        tot += e.stat(follow_symlinks=False).st_blocks * 512
                except OSError:
                    pass
    return tot / 1e9


def burst_gate(cfg, plan, args) -> None:
    """Refuse a stack / phase-linked run whose area spans one burst per
    sub-swath (21 Oct 2026). ISCE's mergeBursts skips a sub-swath holding a
    single burst, so every merge command fails ("min() arg is an empty
    sequence"), and ESD has no burst overlaps to measure: a 10 x 8 km
    experiment crop failed twice that way. Checked before anything is
    downloaded, also with --plan-only; --force goes on regardless. A burst
    count the planner had to assume (no record, no catalogue answer) is not
    refused."""
    n = int(getattr(plan, "bursts_per_date", 0) or 0)
    n_sw = max(1, len(str(getattr(cfg, "stack_swaths", "2") or "2").split()))
    src = str(getattr(plan, "bursts_source", ""))
    if n <= 0 or n > n_sw or src.startswith("assumed"):
        return
    msg = (f"The area spans about {n} burst(s) per date over {n_sw} sub-swath(s) ({src}). topsStack cannot "
           "merge a single-burst sub-swath, and ESD needs burst overlaps. Enlarge the bbox along the track "
           "(a Sentinel-1 burst is about 20 km north-south) so each sub-swath spans two bursts; the analysis "
           "can still be limited to the smaller area afterwards.")
    if getattr(args, "force", False):
        log.warning("--force: proceeding although %s", msg)
        return
    raise SystemExit(msg)


def disk_gate(cfg, plan, args) -> dict:
    """Refuse a stack / phase-linked run whose projected peak does not fit.

    The planner's est_peak_gb is the project's whole peak; what the project
    already holds is part of it, so a resume needs only the difference (a
    resumed Site B holds 205 of its ~311 GB). The budget is the real one
    (disk_budget). Raises SystemExit unless --force; also before --plan-only
    returns, so the dashboard's planner check turns red."""
    budget, source = disk_budget(cfg.project_dir.parent)
    have = project_gb(cfg.project_dir) if Path(cfg.project_dir).exists() else 0.0
    need = max(0.0, float(plan.est_peak_gb) - have)
    out = {"budget_gb": round(budget, 1), "source": source, "project_gb": round(have, 1),
           "peak_gb": plan.est_peak_gb, "need_gb": round(need, 1)}
    log.info("Disk: projected peak %.0f GB, %.0f GB already in the project, so %.0f GB more needed; "
             "budget %.0f GB (%s)", plan.est_peak_gb, have, need, budget, source)
    if need > budget and not getattr(args, "force", False):
        raise SystemExit(
            f"Projected peak {plan.est_peak_gb} GB: with {have:.0f} GB already in the project, "
            f"{need:.0f} GB more is needed, but the disk budget is {budget:.0f} GB ({source}). "
            "Free space, shorten the date range, or pass --force if you are batching "
            "downloads and deleting SAFEs as you go.")
    return out


class DiskWatchdog:
    """Samples the disk budget every DISK_SAMPLE_S while a run goes. Below
    DISK_WARN_GB: one warning event and log line. Below DISK_STOP_GB on two
    samples in a row: stop the run's commands (procs.kill_children) and
    interrupt the main thread, so the run ends as a reported failure ("out
    of disk") with its resumable state intact, instead of ext4 I/O errors
    halfway through an HDF5 write."""

    def __init__(self, cfg, sample_s: float = None, warn_gb: float = None, stop_gb: float = None,
                 budget_fn=None):
        self.cfg = cfg
        self.sample_s = DISK_SAMPLE_S if sample_s is None else sample_s
        self.warn_gb = DISK_WARN_GB if warn_gb is None else warn_gb
        self.stop_gb = DISK_STOP_GB if stop_gb is None else stop_gb
        self.budget_fn = budget_fn or (lambda: disk_budget(cfg.project_dir.parent))
        self.tripped = None
        self.warned = False
        self._halt = threading.Event()
        self._low = 0
        self._th = None

    def start(self):
        self._th = threading.Thread(target=self._loop, name="disk-watchdog", daemon=True)
        self._th.start()
        return self

    def stop(self) -> None:
        self._halt.set()

    def check(self) -> None:
        from . import procs, progress
        try:
            gb, source = self.budget_fn()
        except Exception:                                    # noqa: BLE001
            return
        if gb < self.stop_gb:
            self._low += 1
        else:
            self._low = 0
        if gb < self.warn_gb and not self.warned:
            self.warned = True
            log.warning("Disk is running low: %.0f GB left (%s). The run stops itself below %.0f GB.",
                        gb, source, self.stop_gb)
            progress.emit(self.cfg.project_dir, "disk", "running", msg=f"low: {gb:.0f} GB left")
        if self._low >= 2 and self.tripped is None:
            self.tripped = (f"out of disk: {gb:.0f} GB left ({source}), below the {self.stop_gb:.0f} GB "
                            "the run keeps in reserve")
            log.error("Stopping the run: %s", self.tripped)
            progress.emit(self.cfg.project_dir, "disk", "failed", msg=self.tripped)
            procs.kill_children()
            _thread.interrupt_main()

    def _loop(self) -> None:
        while not self._halt.wait(self.sample_s):
            self.check()
            if self.tripped:
                return


# This process's ERA5 prefetches: (project folder, thread, cancel event).
_PREFETCHES: list = []


def start_era5_prefetch(cfg):
    """stack_backend.start_era5_prefetch(cfg), when it exists, in a background
    thread during coregistration: the ERA5 downloads (hours, network-bound)
    then overlap hours of CPU-bound coregistration instead of following it.
    A failure only logs; MintPy fetches whatever is missing itself. The run
    cancels it when it ends (stop_era5_prefetch)."""
    try:
        from . import stack_backend as sb
    except ImportError:
        return None
    fn = getattr(sb, "start_era5_prefetch", None)
    if fn is None:
        return None
    cancel = threading.Event()

    def _go():
        try:
            fn(cfg, cancel=cancel)
        except Exception as exc:                             # noqa: BLE001
            log.warning("ERA5 prefetch did not complete (MintPy fetches what is missing): %s", exc)
    th = threading.Thread(target=_go, name="era5-prefetch", daemon=True)
    _PREFETCHES.append((Path(cfg.project_dir), th, cancel))
    th.start()
    return th


def stop_era5_prefetch(cfg, wait_s: float = 5.0) -> None:
    """At the end of a run, however it ends: cancel the project's prefetch
    and give it wait_s to record what it did and what it left
    (era5_prefetch.json 'cancelled'). A fetch still waiting on the CDS is
    not waited for: its daemon thread ends with the process. On 22 Oct 2026
    three such fetches kept a finished run alive until it was stopped by
    hand, and the stop recorded "run failed" after a successful run."""
    P = Path(cfg.project_dir)
    mine = [x for x in _PREFETCHES if x[0] == P]
    for _p, _th, cancel in mine:
        cancel.set()
    t_end = time.time() + wait_s
    for _p, th, _cancel in mine:
        th.join(max(0.0, t_end - time.time()))
    _PREFETCHES[:] = [x for x in _PREFETCHES if x[0] != P]


def skip_era5_prefetch(args) -> bool:
    """True for a --to-step run: it stops before MintPy, so it starts no ERA5
    prefetch. The run that carries on ('full' or 'Phase linking onward')
    starts its own."""
    if getattr(args, "to_step", None):
        log.info("No ERA5 prefetch on a run that stops at %s; the run that carries on fetches them.",
                 args.to_step)
        return True
    return False


def run_hyp3(cfg, args) -> None:
    """Sentinel-1 via HyP3: ordering replaces stages 2-4."""
    from . import hyp3_backend
    from .time_series import run_mintpy_with_loader

    products = hyp3_backend.run(cfg, looks=cfg.hyp3_looks)
    loader = hyp3_backend.mintpy_config_lines(cfg, products)
    run_mintpy_with_loader(cfg, loader)


def run_disp(cfg, args) -> None:
    """OPERA DISP-S1: the time series already exists. With
    disp_record_chain it goes through the stack record's MintPy corrections
    (disp_chain.py); otherwise a rate is fitted to it directly."""
    if cfg.disp_record_chain:
        from . import disp_chain
        disp_chain.run(cfg)
        return
    from . import disp_backend
    disp_backend.run(cfg)


def run_nisar(cfg, args) -> None:
    """NISAR: products are pre-made, so only download and staging apply."""
    from . import nisar_backend
    from .data_acquisition import get_credentials
    from .time_series import run_mintpy_with_loader

    username, password = get_credentials()
    staged, loader = nisar_backend.run(cfg, username, password)
    run_mintpy_with_loader(cfg, loader)


def run_stack(cfg, args) -> None:
    """
    topsStack: coregister the whole archive against one reference, then form
    every interferogram off that stack.

    This is the backend with control over the unwrapper, which is the reason
    it exists - see stack_backend's module docstring. Unlike "local", pairs are
    not independent: the coregistration chain must complete before any
    interferogram is meaningful, so the runner fails fast through that section
    and only tolerates per-pair failures afterwards.

    The deramp mask is built inside run_mintpy_with_loader (deramp_mask_step),
    between the inversion and the deramp; it is no longer a hook the caller
    can forget.
    """
    from . import stack_backend as sb
    from .time_series import run_mintpy_with_loader

    refuse_screening_folder(cfg)
    plan = sb.plan_stack(cfg)
    log.info("\n%s", sb.format_plan(plan))
    disk_gate(cfg, plan, args)
    burst_gate(cfg, plan, args)
    if getattr(args, "plan_only", False):
        return

    dem_path = sb.stage_inputs(cfg, skip_download=args.skip_download)
    sb.generate_run_files(cfg, dem_path)
    sb.patch_configs(cfg)
    if not skip_era5_prefetch(args):
        start_era5_prefetch(cfg)
    sb.execute(cfg, from_step=getattr(args, "from_step", None),
               to_step=getattr(args, "to_step", None))
    if getattr(args, "to_step", None):
        log.info("Stopped at %s as requested; not running MintPy.", args.to_step)
        return
    run_mintpy_with_loader(cfg, sb.mintpy_loader_lines(cfg))


def run_miaplpy(cfg, args) -> None:
    """
    Distributed-scatterer route: topsStack coregistration (run_01..run_12,
    merged full-resolution SLCs kept), then MiaplPy phase linking, its
    interferograms, unwrapping and MintPy inversion, then the same
    deliverables, assessment and figures as the stack route, read from
    MiaplPy's network directory.
    """
    from . import miaplpy_backend, stack_backend as sb

    if str(cfg.stack_virtual_merge).lower() != "false":
        raise SystemExit("backend miaplpy needs stack_virtual_merge: \"False\" (MiaplPy reads real merged SLC files)")
    if cfg.stack_cleanup_coreg_after_filter:
        raise SystemExit("backend miaplpy needs stack_cleanup_coreg_after_filter: false (phase linking reads the coregistered SLCs)")
    refuse_screening_folder(cfg)
    plan = sb.plan_stack(cfg)
    log.info("\n%s", sb.format_plan(plan))
    # The official route had no gate at all: run_stack refused a peak above
    # free space, this only logged it (review 5 Oct 2026, ops-6).
    disk_gate(cfg, plan, args)
    burst_gate(cfg, plan, args)
    if getattr(args, "plan_only", False):
        return
    dem_path = sb.stage_inputs(cfg, skip_download=args.skip_download)
    sb.generate_run_files(cfg, dem_path)
    sb.patch_configs(cfg)
    if not skip_era5_prefetch(args):
        start_era5_prefetch(cfg)
    sb.execute(cfg, from_step=getattr(args, "from_step", None), to_step="merge_reference_secondary_slc")
    if getattr(args, "to_step", None):
        log.info("Stopped after coregistration as requested; not running MiaplPy.")
        return
    from .time_series import run_mintpy_with_loader
    miaplpy_backend.run(cfg)
    route_overrides(cfg)
    run_mintpy_with_loader(cfg, sb.mintpy_loader_lines(cfg))


def route_overrides(cfg) -> None:
    """Settings a route forces whatever the config says - in ONE place, so
    every start point applies them (run_mintpy_only.py lacked this one)."""
    if cfg.backend == "miaplpy" and int(cfg.miaplpy_network_connections) <= 0:
        cfg.mintpy_unwrap_error_method = "bridging"   # a star has no closed loops


# The routes whose MintPy run puts the extra layers on the product grid
# (time_series.run_mintpy_with_loader -> geocode_extra_layers). The DISP chain
# writes its own geo files and the local route's MintPy run predates the
# layers, so the products stage leaves those two as they are.
EXTRA_LAYER_ROUTES = ("miaplpy", "stack", "hyp3", "nisar")


def geocode_missing_layers(cfg) -> dict:
    """
    In the products stage, before the deliverables (F5, 7 Oct 2026): the
    extra layers the MintPy run puts on the product grid
    (time_series.geocode_extra_layers: unwrapping consistency, DEM error,
    disturbed ground) that are NOT on disk are made, so every project's
    point table carries the second-gate columns and the "ground reshaped
    since the elevation model" test (E1, rate-corrected) can run. A project
    whose MintPy run finished before the helper existed (Site A, 27 Sep
    2026) had none of them. A layer already on disk is never geocoded again.
    What was done is logged; a failure is logged and the stage goes on
    without the layers, as before. Never raises an Exception (a stop still
    stops the run).

    Returns {"present": [...], "made": [...], "missing_after": [...],
    "skipped": why or None, "error": text or None}, names as geo/geo_<name>.
    """
    from . import time_series as ts
    geo = Path(cfg.mintpy_dir) / "geo"
    names = [f"geo_{n}" for n in ts.extra_layers_for(cfg)]
    rec = {"present": [n for n in names if (geo / n).exists()], "made": [], "missing_after": [],
           "skipped": None, "error": None}
    missing = [n for n in names if n not in rec["present"]]
    if not missing:
        log.info("Extra layers: all on the product grid already (%s); none geocoded", ", ".join(rec["present"]))
        return rec
    if cfg.backend not in EXTRA_LAYER_ROUTES:
        rec["skipped"] = f"the {cfg.backend} route does not make these layers"
        log.info("Extra layers not on disk (%s): %s, so they are not geocoded here", ", ".join(missing),
                 rec["skipped"])
        rec["missing_after"] = missing
        return rec
    if (Path(cfg.mintpy_dir) / ts.RUN_INCOMPLETE).exists():
        rec["skipped"] = f"mintpy/{ts.RUN_INCOMPLETE} is there (the MintPy run did not finish)"
        log.warning("Extra layers not geocoded: %s", rec["skipped"])
        rec["missing_after"] = missing
        return rec
    log.info("Extra layers not on the product grid: %s - geocoding them now (%s already there, left as they "
             "are)", ", ".join(missing), ", ".join(rec["present"]) or "none")
    try:
        made = ts.geocode_extra_layers(cfg, missing_only=True)
        rec["made"] = [Path(p).name for p in made]
    except Exception as exc:                                  # noqa: BLE001 - logged, the products go on
        rec["error"] = f"{type(exc).__name__}: {exc}"
        log.error("Geocoding the missing extra layers failed (%s); the products go on without them",
                  rec["error"], exc_info=True)
    rec["missing_after"] = [n for n in missing if not (geo / n).exists()]
    log.info("Extra layers: geocoded now %s%s", ", ".join(rec["made"]) or "none",
             f"; still not on disk {', '.join(rec['missing_after'])} (the products go on without them)"
             if rec["missing_after"] else "")
    return rec


def extra_layers_note(cfg, rec: dict | None) -> str | None:
    """The run_result.json note for geocode_missing_layers' record, or None
    when nothing is missing that the route should have. Until 16 Oct 2026
    the record was discarded: a failed geocode was logged at ERROR and the
    run still read 'done, notes: []' (readiness review ERR-01), so the point
    table's second-gate columns and the 'reshaped since the elevation model'
    test could be missing with nothing in the run's end record."""
    rec = rec or {}
    left = [str(n) for n in rec.get("missing_after") or []]
    if not left or cfg.backend not in EXTRA_LAYER_ROUTES:
        return None
    what = ", ".join(left)
    if rec.get("error"):
        return (f"Geocoding the extra quality layers failed ({str(rec['error'])[:200]}); the products were made "
                f"without {what} (the log has the traceback; 'Products only' tries again).")
    if rec.get("skipped"):
        return f"The extra quality layers were not geocoded ({rec['skipped']}); the products were made without {what}."
    return f"The extra quality layers {what} are still not on the product grid; the products were made without them."


def unwrap_correction_note(cfg) -> str | None:
    """The run_result.json note for an unwrap-error correction step the run
    asked MintPy for that made nothing (time_series.unwrap_error_shortfall),
    else None. Until 21 Oct 2026 the run read 'done, notes: []' when phase
    closure had been skipped, and only provenance.json and one log WARNING
    said so. Never raises: the note is a record, not a step."""
    try:
        from .time_series import unwrap_error_shortfall, unwrap_error_skip_text
        text = unwrap_error_skip_text(unwrap_error_shortfall(cfg))
    except Exception as exc:                                 # noqa: BLE001
        log.warning("The unwrap-error correction record could not be read: %s", exc)
        return None
    return (text[0].upper() + text[1:] + ".") if text else None


def run_quality_assessment(cfg) -> bool:
    """Quality assessment against field practice, after the deliverables, on
    the stack backend. Writes <project>/quality_assessment.json; never raises
    into the pipeline. Returns whether it ran (run_from counts a False as a
    failed stage: the run exits non-zero and says which)."""
    from . import progress
    progress.emit(cfg.project_dir, "quality_assessment", "start")
    try:
        from . import deliverables, quality_assessment as qa
        # The deliverables stage caches the full time series (6 GB at 251
        # dates); the assessment loads its own. Both resident at once is
        # what silently killed the process on the production run.
        deliverables.clear_cache()
        # fetch_water: the geolocation check compares the product with mapped
        # water edges, fetched once from OpenStreetMap into
        # context/osm_water.geojson (a context/water.geojson overrides it).
        # Library and test calls leave it off, so only a run touches the network.
        r = qa.evaluate(cfg, fetch_water=True)
        out = qa.write(cfg, r)
        log.info("Quality assessment: %s -> %s", r["verdict"], out)
        log.info("\n%s", qa.format_report(r))
        # summary.json was written before the QA existed; attach its verdict now
        _refresh_summary(cfg)
        sr = (r.get("send_rule") or {}).get("value") if isinstance(r.get("send_rule"), dict) else None
        progress.emit(cfg.project_dir, "quality_assessment", "done",
                      msg=r["verdict"] + (f"; {sr}" if sr else ""))
        return True
    except Exception as exc:                                  # noqa: BLE001
        progress.emit(cfg.project_dir, "quality_assessment", "failed", msg=repr(exc)[:200])
        log.error("Quality assessment could not run: %s", exc, exc_info=True)
        return False


def run_figures(cfg) -> bool:
    """Deliverable figures and the summary table into deliverables/figures/,
    after the quality assessment they read. Never raises into the pipeline."""
    from . import progress
    progress.emit(cfg.project_dir, "figures", "start")
    try:
        from . import figures, panel_history
        # panel_history FIRST: the summary table in make_figures reads the
        # panel and site histories it writes (the other order drew the table
        # from the previous run's figures)
        panel_history.run(cfg)
        out = figures.make_figures(cfg)
        log.info("Figures in %s", out)
        progress.emit(cfg.project_dir, "figures", "done")
        return True
    except Exception as exc:                                  # noqa: BLE001
        progress.emit(cfg.project_dir, "figures", "failed", msg=repr(exc)[:200])
        log.error("Figures could not be drawn: %s", exc, exc_info=True)
        return False


def run_client_report(cfg) -> bool:
    """GNSS anchor and the plain-language client report, after the figures they
    read (gnss_tie.py, client_report.py). Never raises into the pipeline.
    Callers decide whether the route may have one (client_report_allowed).
    An experiment run (purpose: experiment) never has one, whoever calls:
    the dashboard's report rebuild comes here directly, so the refusal is
    here and not only in run_from. It is logged, recorded as a skipped stage
    and returns False (nothing was built)."""
    from . import progress
    if is_experiment(cfg):
        log.warning("%s.", EXPERIMENT_NO_REPORT)
        progress.emit(cfg.project_dir, "client_report", "skipped", msg="experiment run: no client report")
        return False
    progress.emit(cfg.project_dir, "client_report", "start")
    try:
        from . import client_report, gnss_tie
        try:
            gnss_tie.run(cfg)               # always writes gnss_tie.json; re-raises a failed tie
        finally:
            _refresh_summary(cfg)           # the tie's outcome (or its reason) into summary.json
        log.info("Client report: %s", client_report.build(cfg))
        from . import client_kmz
        log.info("Client KMZ: %s", client_kmz.build(cfg))
        progress.emit(cfg.project_dir, "client_report", "done")
    except Exception as exc:                                  # noqa: BLE001
        progress.emit(cfg.project_dir, "client_report", "failed", msg=repr(exc)[:200])
        log.error("Client report could not be built: %s", exc, exc_info=True)
        return False
    return record_report_stages(cfg)


def _refresh_summary(cfg) -> None:
    """deliverables.refresh_summary, logged rather than raised: summary.json is
    written at export, before the QA verdict and the GNSS tie it reports."""
    try:
        from . import deliverables
        deliverables.refresh_summary(cfg)
    except Exception as exc:                                  # noqa: BLE001
        log.warning("summary.json not refreshed: %s", exc)


def record_report_stages(cfg) -> bool:
    """After the client report: add the QA, GNSS-tie, report and review records
    to provenance.json (written at export, before any of them existed), and
    log what still stands between this report and the client. The package
    itself is built only on request once the report is final (python -m
    insar_pipeline.package <config>, or the dashboard): a release is a frozen,
    numbered copy, not something every rebuild should add. Returns whether
    the record was written (a False fails the stage)."""
    from . import progress
    progress.emit(cfg.project_dir, "provenance", "start")
    try:
        from . import provenance, release
        out = provenance.record_stages(cfg)
        st = release.status(cfg)
        hard = [b["text"] for b in st.get("blockers", []) if b.get("hard")]
        if st.get("final"):
            log.info("Release: FINAL (reviewed by %s on %s). Build the client package with "
                     "`python -m insar_pipeline.package <config>`.", st.get("reviewed_by"), st.get("review_date"))
        else:
            log.warning("Release: DRAFT - %d item(s) to settle before this report can go to a client:\n  %s",
                        len(hard), "\n  ".join(hard) or "(see release.status)")
        progress.emit(cfg.project_dir, "provenance", "done",
                      msg=("final" if st.get("final") else f"draft: {len(hard)} to settle"))
        log.info("Provenance stages recorded in %s", out)
        return True
    except Exception as exc:                                  # noqa: BLE001
        progress.emit(cfg.project_dir, "provenance", "failed", msg=repr(exc)[:200])
        log.error("Provenance stages could not be recorded: %s", exc, exc_info=True)
        return False


def _optional_stage(cfg, stage: str, module: str, what: str, call=None) -> bool:
    """A stage the config switched on, run by <module>.run(cfg) (written by
    another part of the package), or by call(module, cfg) when given. Never
    raises into the pipeline; a module that is not there is a failure of a
    stage that was asked for."""
    from . import progress
    progress.emit(cfg.project_dir, stage, "start")
    try:
        import importlib
        from . import deliverables
        # these run straight after the deliverables stage, whose cached time
        # series (6 GB at 251 dates) must not stay resident beside them
        deliverables.clear_cache()
        mod = importlib.import_module(f".{module}", __package__)
        out = call(mod, cfg) if call is not None else mod.run(cfg)
        log.info("%s: %s", what, out)
        progress.emit(cfg.project_dir, stage, "done")
        return True
    except Exception as exc:                                  # noqa: BLE001
        progress.emit(cfg.project_dir, stage, "failed", msg=repr(exc)[:200])
        log.error("%s could not run: %s", what, exc, exc_info=True)
        return False


def run_crosscheck(cfg) -> bool:
    """DISP-S1 cross-check of the official result (crosscheck.py), when
    cross_check_disp: true. Off by default (Kyle, 5 Oct 2026)."""
    return _optional_stage(cfg, "crosscheck", "crosscheck", "DISP-S1 cross-check")


def run_injection_test(cfg) -> bool:
    """Injection-recovery test (injection_test.py), when injection_test: true:
    no positive-control site exists, so a known signal is planted and
    recovered instead. Off by default."""
    return _optional_stage(cfg, "injection_test", "injection_test", "Injection-recovery test")


def run_validation(cfg) -> bool:
    """The validation harness (validation.py, review_2026-10-08/PLAN.md V1)
    against the fixed protocol in validation_protocol, on an experiment run
    only, after the quality assessment and the figures. Never raises into the
    pipeline; a missing harness is a failure of a stage that was asked for.
    How the harness is called: _call_harness."""
    return _optional_stage(cfg, "validation", "validation", "Validation against ground truth", call=_call_harness)


def _call_harness(mod, cfg):
    """validation.run(cfg) when the harness has one; otherwise its command
    line, validation.main([<config file>, "--protocol", <protocol>], cfg=cfg),
    as `insar validate` runs it. The config file's name is how the harness
    finds the protocol's run (B1, A3, ...), and it writes under
    <project>/validation/. A non-zero exit raises, so the stage fails: a
    validation that was asked for and not made never reads as done."""
    if callable(getattr(mod, "run", None)):
        return mod.run(cfg)
    main = getattr(mod, "main", None)
    if not callable(main):
        raise RuntimeError("validation.py has neither run(cfg) nor main(argv, cfg): the harness cannot be called")
    src = getattr(cfg, "_source_path", None)
    if not src:
        raise RuntimeError("the config was not read from a file (load_config), so the harness cannot tell "
                           "which of the protocol's runs this is")
    code = main([str(src), "--protocol", str(cfg.validation_protocol)], cfg=cfg)
    if code:
        raise RuntimeError(f"the validation harness stopped with exit code {code}; its reason is in the log above")
    return f"written under {Path(cfg.project_dir) / 'validation'}"


def record_experiment_stages(cfg) -> bool:
    """An experiment's provenance: the QA (and any optional stage's) records
    into deliverables/provenance.json, as record_report_stages does after a
    client report, but with no release status - an experiment is never
    released. Returns whether the record was written."""
    from . import progress
    progress.emit(cfg.project_dir, "provenance", "start")
    try:
        from . import provenance
        out = provenance.record_stages(cfg)
        progress.emit(cfg.project_dir, "provenance", "done", msg="experiment run: no release status")
        log.info("Provenance stages recorded in %s (experiment run: no release status)", out)
        return True
    except Exception as exc:                                  # noqa: BLE001
        progress.emit(cfg.project_dir, "provenance", "failed", msg=repr(exc)[:200])
        log.error("Provenance stages could not be recorded: %s", exc, exc_info=True)
        return False


SCREENING_NOTE = "SCREENING_NOT_FOR_CLIENTS.txt"
EXPERIMENT_NOTE = "EXPERIMENT_NOT_FOR_CLIENTS.txt"
EXPERIMENT_NO_REPORT = ("This is an experiment run (purpose: experiment): no client report, client KMZ, release "
                        "status or client package is built. Its result is the validation against ground truth")


def client_report_allowed(cfg) -> tuple:
    """(allowed, why not). The DISP-S1 route is screening only (config.ROUTE_TAG):
    its client report, KMZ and GeoTIFFs looked exactly like an official
    delivery once they left the dashboard (review 5 Oct 2026, disp-1). It
    builds them only when the config says disp_client_report: true. An
    experiment run (purpose: experiment) never builds them."""
    if is_experiment(cfg):
        return False, EXPERIMENT_NO_REPORT
    if not is_screening(cfg):
        return True, ""
    if cfg.disp_client_report:
        return True, ""
    return False, ("DISP-S1 is the screening route: no client report or KMZ is built "
                   "(official results come from the phase-linked route; disp_client_report: true "
                   "builds one anyway, marked as screening)")


def set_aside_client_files(cfg, why: str = "screening") -> list:
    """Move client-facing files an earlier run left in deliverables/
    (client_report.html/.json, the area PDF, every *ground_movement*.kmz -
    DRAFT and SCREENING ones too) into
    deliverables/<why>_not_for_clients/<stamp>/ - moved, never deleted -
    and write a plain note saying what this folder holds: a DISP-S1
    screening (why "screening") or an experiment run (why "experiment").
    Returns the moved paths."""
    if why not in ("screening", "experiment"):
        raise ValueError(f"set_aside_client_files: why {why!r} is not 'screening' or 'experiment'")
    D = Path(cfg.project_dir) / "deliverables"
    if not D.exists():
        return []
    if why == "experiment":
        return _set_aside(D, "experiment_not_for_clients", EXPERIMENT_NOTE, (
            "EXPERIMENT RUN - NOT FOR CLIENTS\n\n"
            f"Project {Path(cfg.project_dir).name} is a validation experiment (config purpose: experiment).\n"
            "Its products are checked against ground truth by the validation harness; they are not a client\n"
            "delivery. No client report, client KMZ, release status or client package is built for it.\n\n"
            "Technical outputs here: quality_assessment.json (in the project folder), summary.json,\n"
            "the GeoTIFFs, the figures/ folder and the validation records. Any client file an earlier run\n"
            "left here was moved to experiment_not_for_clients/.\n"))
    return _set_aside(D, "screening_not_for_clients", SCREENING_NOTE, (
        "SCREENING ONLY - NOT FOR CLIENTS\n\n"
        f"Project {Path(cfg.project_dir).name} was processed on the OPERA DISP-S1 route (backend: disp).\n"
        "Its figures are a screening: they have not had the full processing and may differ from it.\n"
        "Official results come from the phase-linked route (backend: miaplpy).\n\n"
        "Technical outputs here: quality_assessment.json (in the project folder), summary.json,\n"
        "the GeoTIFFs and the figures/ folder. No client report or KMZ is built for this route\n"
        "unless the config says disp_client_report: true.\n"))


def _set_aside(D: Path, folder: str, note_name: str, note: str) -> list:
    """set_aside_client_files' move into D/<folder>/<stamp>/ and its note."""
    # client_kmz.kmz_name: <T>_[SCREENING_]ground_movement[_DRAFT].kmz;
    # client_report.PDF_NAME is the area PDF
    victims = [f for f in [D / "client_report.html", D / "client_report.json", D / "client_area_summaries.pdf",
                           *D.glob("*ground_movement*.kmz")]
               if f.is_file()]
    moved = []
    if victims:
        dest = D / folder / time.strftime("%Y%m%d_%H%M%S")
        dest.mkdir(parents=True, exist_ok=True)
        for f in victims:
            shutil.move(str(f), str(dest / f.name))
            moved.append(dest / f.name)
        log.warning("Moved %d client-facing file(s) from an earlier run into %s", len(moved), dest)
    try:
        (D / note_name).write_text(note)
    except OSError:
        pass
    return moved


run_acceptance = run_quality_assessment          # old name, same step


def run_local(cfg, args) -> None:
    """topsApp per pair, the original testing path."""
    from .data_acquisition import download_scenes, get_credentials, get_dem, search_scenes
    from .isce_processing import _pair_dir, run_all_pairs
    from .pair_selection import build_sequential_network
    from .time_series import run_mintpy
    from .tropo_correction import correct_pair

    log.info("=== Stage 1: data acquisition ===")
    results = search_scenes(cfg)  # metadata needed for pairing either way
    if not args.skip_download:
        username, password = get_credentials()  # prompts once, reused below
        download_scenes(cfg, results, username, password)
        # Orbit files are NOT pre-downloaded here - topsApp.py fetches them
        # itself at process time as long as it has internet access.
    dem_path = get_dem(cfg)

    if args.local_only:
        on_disk = [r for r in results
                   if (cfg.slc_dir / r.properties["fileName"]).exists()]
        log.info("Restricting to %d scenes already on disk (of %d found)",
                 len(on_disk), len(results))
        results = on_disk

    log.info("=== Stage 2: pair selection ===")
    pairs = build_sequential_network(results,
                                     max_temporal_days=cfg.max_temp_baseline,
                                     n_neighbors=cfg.n_neighbors)

    log.info("=== Stage 3: interferogram formation (ISCE2) ===")
    succeeded, failed = run_all_pairs(cfg, pairs, dem_path)
    if failed:
        log.warning("%d pairs failed and will be excluded from the time series", len(failed))

    log.info("=== Stage 4: tropospheric correction (RAiDER) ===")
    # RAiDER needs a plain georeferenced raster, not ISCE's dem.wgs84 -
    # get_dem() keeps the intermediate GeoTIFF for exactly this.
    dem_tif = cfg.dem_dir / "dem_raw.tif"
    if not dem_tif.exists():
        log.warning("No %s found; skipping tropospheric correction", dem_tif)
    else:
        for ref, sec in succeeded:
            pair_dir = _pair_dir(cfg, ref, sec)
            try:
                correct_pair(cfg, ref, sec, pair_dir, dem_tif)
            except Exception as exc:                          # noqa: BLE001
                log.error("Tropo correction failed for %s: %s", pair_dir.name, exc)

    log.info("=== Stage 5: time series inversion (MintPy) ===")
    run_mintpy(cfg, dem_path)


RUNNERS = {"hyp3": run_hyp3, "nisar": run_nisar, "disp": run_disp,
           "stack": run_stack, "local": run_local, "miaplpy": run_miaplpy}


# ---------------------------------------------------------------- preconditions
def _phase_series_dates(cfg) -> list:
    f = Path(cfg.project_dir) / "miaplpy" / "inverted" / "phase_series.h5"
    if not f.exists():
        return []
    import h5py
    with h5py.File(f, "r") as h:
        return [d.decode() if isinstance(d, bytes) else str(d) for d in h["date"][:]]


def expected_pairs(cfg) -> tuple:
    """(expected pair names, how they were worked out) for the phase-linked
    export: miaplpy_backend.network_pairs over phase_series.h5's dates; when
    the phase series is gone (cleanup), over the dates the exported pairs
    span (a pair missing between known dates is still caught)."""
    from . import miaplpy_backend as mb, runs
    dates = _phase_series_dates(cfg)
    how = "the phase series' dates"
    if not dates:
        ex = runs.unwrap_status(cfg.stack_dir)["exported"]
        dates = sorted({d for p in ex for d in p.split("_")})
        how = "the dates of the exported pairs (phase_series.h5 is gone)"
    if not dates:
        return [], "no phase series and no exported pairs"
    ref = mb.stack_reference(cfg)
    if ref not in dates:
        ref = dates[0]
    return [f"{a}_{b}" for a, b, _, _ in mb.network_pairs(dates, int(cfg.miaplpy_network_connections), ref)], how


def check_preconditions(cfg, start: str, force: bool = False) -> dict:
    """Refuse a start point whose inputs are not complete on disk, with the
    restart that fixes it. A wrong restart used to run anyway and silently
    shorten the record for good: 'Phase linking onward' after a merge that
    stopped at 150 of 270 dates loads 150 dates and skips load_data on every
    later resume; 'Re-unwrap' after an interrupted export unwraps only the
    pairs that happen to exist. --force overrides with a warning."""
    from . import runs
    out = {}
    problems = []
    if start in ("miaplpy", "unwrap") and cfg.backend != "miaplpy":
        raise SystemExit(f"'{start}' is a phase-linked (backend miaplpy) restart; this project's backend is {cfg.backend}")
    if start == "mintpy" and cfg.backend not in ("stack", "miaplpy"):
        raise SystemExit(f"'MintPy onward' runs on the stack or phase-linked routes, not {cfg.backend}")
    W = Path(cfg.project_dir) / "miaplpy"
    if start == "miaplpy" and not (W / "inverted" / "phase_series.h5").exists():
        cs = runs.coregistration_status(cfg.stack_dir)
        out["coregistration"] = {k: len(v) for k, v in cs.items()}
        stack_h5 = W / "inputs" / "slcStack.h5"
        if stack_h5.exists() and cs["expected"]:
            import h5py
            with h5py.File(stack_h5, "r") as h:
                loaded = {d.decode() if isinstance(d, bytes) else str(d) for d in h["date"][:]} if "date" in h else set()
            short = [d for d in cs["expected"] if d not in loaded]
            if loaded and short:
                problems.append(
                    f"MiaplPy's SLC stack holds {len(loaded)} dates but the coregistration has "
                    f"{len(cs['expected'])} ({', '.join(short[:6])}{' ...' if len(short) > 6 else ''} missing). "
                    "Phase linking would run on the shorter record for good: finish the coregistration "
                    "('Resume' or 'Whole pipeline'), delete miaplpy/inputs/slcStack.h5, then restart.")
        elif cs["missing"]:
            problems.append(
                f"Coregistration is not finished: {len(cs['missing'])} of {len(cs['expected'])} dates have no "
                f"merged SLC ({', '.join(cs['missing'][:6])}{' ...' if len(cs['missing']) > 6 else ''}). "
                "Phase linking would load the shorter stack for good. Restart with 'Resume' or "
                "'Whole pipeline' (finished dates are kept, nothing is re-downloaded that is coregistered).")
    if start in ("unwrap", "mintpy") and cfg.backend == "miaplpy":
        exp, how = expected_pairs(cfg)
        st = runs.unwrap_status(cfg.stack_dir)
        out["pairs"] = {"expected": len(exp), "exported": len(st["exported"]), "unwrapped": len(st["unwrapped"]),
                        "from": how}
        never = sorted(set(exp) - set(st["exported"]))
        if not exp:
            problems.append("There is no phase-linked export to work on (no phase_series.h5, no exported pairs). "
                            "Restart with 'Resume' or 'Phase linking onward'.")
        elif never:
            problems.append(
                f"{len(never)} of {len(exp)} expected interferograms were never exported "
                f"({', '.join(never[:6])}{' ...' if len(never) > 6 else ''}; expected from {how}). "
                "The record would come out shorter without saying so. Restart with 'Phase linking "
                "onward', which finishes the export.")
        elif start == "mintpy":
            unw = sorted(set(exp) - set(st["unwrapped"]))
            if unw:
                problems.append(
                    f"{len(unw)} of {len(exp)} interferograms are not unwrapped "
                    f"({', '.join(unw[:6])}{' ...' if len(unw) > 6 else ''}). Restart with 'Phase linking "
                    "onward' (it resumes the unwrapping) instead of 'MintPy onward'.")
        out["pair_names"] = exp
    if problems:
        if force:
            for p in problems:
                log.warning("--force: proceeding although %s", p)
        else:
            raise SystemExit(" ".join(problems))
    return out


# ---------------------------------------------------------------- the stage chain
def _args_from(opts: dict) -> SimpleNamespace:
    return SimpleNamespace(
        skip_download=bool(opts.get("skip_download", False)), local_only=bool(opts.get("local_only", False)),
        plan_only=bool(opts.get("plan_only", False)), force=bool(opts.get("force", False)),
        from_step=opts.get("from_step"), to_step=opts.get("to_step"),
        no_acceptance=bool(opts.get("no_acceptance", False)))


def _log_file(project_dir) -> str | None:
    """This process's log: run.json's when it is ours, else where stdout goes."""
    from . import runs
    rec = runs.record(project_dir) or {}
    if rec.get("pid") == os.getpid() and rec.get("log"):
        return rec["log"]
    try:
        lp = os.readlink("/proc/self/fd/1")
        if lp.startswith("/") and Path(lp).is_file():
            return lp
    except OSError:
        pass
    return None


def _is_stop(exc: BaseException) -> bool:
    return isinstance(exc, KeyboardInterrupt) or (isinstance(exc, SystemExit) and exc.code in (129, 130, 143))


class WarningCounter(logging.Handler):
    """The WARNING-and-above log records of a run, counted and grouped by
    logger, level and message template (16 Oct 2026, readiness review
    ERR-02). They were neither counted nor shown: the archived products run
    logged 10 WARNINGs and its run_result.json read 'done, notes: []', and
    one full log held 860 WARNING lines, 846 of them one message. Only this
    process's logging records are counted - the lines a child program
    (MiaplPy, MintPy, snaphu) prints into the log are not."""

    MAX_DISTINCT = 500                    # a message built with an f-string is a new template every time

    def __init__(self):
        super().__init__(level=logging.WARNING)
        self.count = self.errors = self.uncounted = 0
        self.groups: dict = {}            # (logger, level, template) -> [first message, n]

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.count += 1
            self.errors += record.levelno >= logging.ERROR
            key = (record.name, record.levelno, str(record.msg))
            got = self.groups.get(key)
            if got is not None:
                got[1] += 1
            elif len(self.groups) < self.MAX_DISTINCT:
                try:
                    msg = record.getMessage()
                except Exception:                            # noqa: BLE001 - a bad format string is still a warning
                    msg = str(record.msg)
                first = (msg.strip().splitlines() or [""])[0]
                self.groups[key] = [f"{record.levelname} {record.name}: {first}"[:300], 1]
            else:
                self.uncounted += 1
        except Exception:                                    # noqa: BLE001 - counting never breaks logging
            pass

    def summary(self) -> dict:
        """{count, errors, distinct, top: [[message, n], ...] (at most 10, most frequent first)}; the
        message is the first of its kind, with its level and logger."""
        top = sorted(self.groups.values(), key=lambda g: -g[1])[:10]
        out = {"count": self.count, "errors": self.errors, "distinct": len(self.groups),
               "top": [[m, n] for m, n in top]}
        if self.uncounted:
            out["distinct_capped"] = True                     # more templates than MAX_DISTINCT
        return out


# The counter of the run going on in this process (run_from), read by write_run_result.
_RUN_WARNINGS: WarningCounter | None = None


def write_run_result(cfg, rec: dict) -> Path:
    """<project>/run_result.json, atomically (the dashboard polls it).

    Since 16 Oct 2026 (readiness review ERR-02, ERR-04, OUT-05) it also
    carries the run's WARNING count ('warnings', WarningCounter) and the
    build that ran it ('version', 'code_hash'); the record it replaces is
    appended to <project>/run_result_history.jsonl first
    (runs.keep_previous_result), and the run gets one line in the run
    registry under the projects root (runs.register_run)."""
    import json
    from . import runs
    P = Path(cfg.project_dir)
    for h in logging.getLogger().handlers:
        try:
            h.flush()
        except Exception:                                    # noqa: BLE001
            pass
    try:
        sys.stdout.flush(); sys.stderr.flush()
    except Exception:                                        # noqa: BLE001
        pass
    lg = rec.get("log") or _log_file(P)
    rec = {**rec, "log": lg, "log_tail": runs.tail(lg, 40) if lg else [], "finished": time.time(),
           "pid": os.getpid()}
    if _RUN_WARNINGS is not None and "warnings" not in rec:
        rec["warnings"] = _RUN_WARNINGS.summary()
    try:
        from . import version
        rec.setdefault("version", version.__version__)
        rec.setdefault("code_hash", version.code_hash())
    except Exception as exc:                                 # noqa: BLE001 - the end record is written regardless
        log.warning("run_result.json: the code version not read (%s)", exc)
    f = P / runs.RESULTFILE
    runs.keep_previous_result(P, rec)
    tmp = f.with_suffix(".json.tmp")
    try:
        tmp.write_text(json.dumps(rec, indent=1, default=str))
        os.replace(tmp, f)
    except OSError as exc:                 # a full disk must not hide the run's own failure
        log.error("Could not write %s: %s", f, exc)
    runs.register_run(P, rec)
    return f


def run_from(cfg, start: str = "full", **opts) -> int:
    """_run_from (the stage chain, below), then - once its project lock is
    released - the automatic 'keep results only' of a finished experiment
    run: its result is the validation, written by the products stage
    (config auto_keep_results; 13 Oct 2026). Client projects are trimmed when
    their FINAL package is built instead (package.build).

    The run's WARNING-and-above log records are counted for its
    run_result.json while _run_from runs (WarningCounter; ERR-02), and its
    log is capped at runs.LOG_MAX_BYTES, rolling over to .1-.3 (ERR-03)."""
    global _RUN_WARNINGS
    from . import runs
    counter, outer = WarningCounter(), _RUN_WARNINGS
    root = logging.getLogger()
    root.addHandler(counter)
    _RUN_WARNINGS = counter
    # the run log's size cap (runs.LogRoller; ERR-03): only when stdout is a run log
    lp = runs.run_log_path()
    roller = runs.LogRoller(lp).start() if lp is not None else None
    try:
        code = _run_from(cfg, start, **opts)
    finally:
        if roller is not None:
            roller.stop()
        root.removeHandler(counter)
        _RUN_WARNINGS = outer
    if (code == 0 and is_experiment(cfg) and not opts.get("plan_only")
            and opts.get("kind") != "plan" and str(start) not in ("plan", "size")):
        from .storage import auto_keep_results
        auto_keep_results(cfg, "the experiment run finished and its validation is written")
    return code


def _run_from(cfg, start: str = "full", **opts) -> int:
    """Run the stage chain from `start` and report how it ended. Returns the
    exit code (0 done; 1 a stage failed; 128+signal when stopped).

    start: full (download -> coregistration -> [phase linking] -> unwrap ->
    MintPy), miaplpy (phase linking onward), unwrap (re-unwrap the
    phase-linked export), mintpy, products (deliverables onward); aliases in
    START_ALIASES. Products always run: the missing extra layers
    (geocode_missing_layers) -> deliverables -> DISP cross-check /
    injection test (when the config asks) -> quality assessment (which
    copies their records) -> figures -> GNSS tie + client report (not on the
    screening route unless disp_client_report). An experiment run (purpose:
    experiment) ends instead with the validation harness (when
    validation_protocol is set) and the provenance record: no GNSS tie,
    client report, KMZ or release status, and older client files are moved
    to deliverables/experiment_not_for_clients/. A screening config is
    refused into a folder holding official products whatever the start.
    opts: kind (the run kind named in run_result.json),
    skip_download, local_only, force, from_step, to_step, plan_only,
    no_acceptance, config_path, before (callable(cfg) run under the lock
    before the chain - reunwrap_linked.py's backup), notify (default True),
    watchdog (default True), resumed_as.

    Every exception is caught here and turned into a failed (or stopped)
    stage event, a "pipeline" final event, run_result.json and a
    notification; a process that vanishes without them was killed
    (runs.state "interrupted")."""
    from . import advice, procs, progress, runs
    alias = start
    start = START_ALIASES.get(start, start)
    if start not in STARTS:
        raise ValueError(f"unknown start {alias!r}; one of {STARTS} or {list(START_ALIASES)}")
    if alias == "coreg":
        opts.setdefault("skip_download", True)
    args = _args_from(opts)
    kind = opts.get("kind") or alias
    procs.install_teardown()
    if cfg.backend not in RUNNERS:
        raise SystemExit(f"backend {cfg.backend!r} is not one of {list(BACKENDS)}")

    if args.plan_only:
        # Sizing is not a run: no lock, no events, no run_result. It used to
        # emit "pipeline start" and return, which turned a finished project
        # "failed" in the dashboard (ops-10).
        warn_route(cfg)
        if cfg.backend in ("stack", "miaplpy"):
            RUNNERS[cfg.backend](cfg, args)
        else:
            log.info("Nothing to size on the %s route.", cfg.backend)
        return 0

    P = Path(cfg.project_dir)
    with runs.project_lock(P, kind=kind, config=opts.get("config_path")):
        runs.claim_runfile(P, kind, opts.get("config_path"))
        procs.track_children_in(P / runs.CHILDFILE)
        tag = warn_route(cfg)
        log.info("=== Backend: %s (%s route), %s run from %s ===", cfg.backend, tag, kind, start)
        others = runs.live_runs(exclude=P)
        if others:
            log.warning("%d other run(s) are going on this machine (%s): they share the cores, memory and "
                        "disk, and each sizes its pools as if it were alone - expect it slower, and an "
                        "out-of-memory kill is possible.", len(others),
                        "; ".join(f"{Path(o['project']).name} {o.get('kind') or ''} pid {o.get('pid')}" for o in others))
        progress.emit(P, "pipeline", "start", msg=f"{cfg.backend} route, {kind}, pid {os.getpid()}")
        wd = DiskWatchdog(cfg).start() if opts.get("watchdog", True) else None
        stage = "pipeline"
        failed_stages = []
        notes = []
        t0 = time.time()
        res = {"kind": kind, "start": start, "backend": cfg.backend, "route": tag,
               "config": opts.get("config_path"), "started": t0}
        if is_experiment(cfg):
            res["purpose"] = "experiment"
        if opts.get("resumed_as"):
            res["resumed_as"] = opts["resumed_as"]
        try:
            stage = "checks"
            pre = check_preconditions(cfg, start, force=args.force)
            if is_screening(cfg):
                # the DISP runners refuse an official folder, but 'Products
                # only' never calls them: export, QA and the screening
                # set-aside would rebuild and move the official client files
                from . import disp_backend
                disp_backend.refuse_foreign_mintpy(cfg)
            if callable(opts.get("before")):
                stage = "prepare"
                opts["before"](cfg)
            route_overrides(cfg)
            if start == "full":
                stage = "processing"
                RUNNERS[cfg.backend](cfg, args)
                if args.to_step:
                    msg = f"stopped after {args.to_step} as requested"
                    log.info("%s; no products built.", msg[0].upper() + msg[1:])
                    progress.emit(P, "pipeline", "done", msg=msg)
                    write_run_result(cfg, {**res, "status": "done", "stage": str(args.to_step),
                                           "plain_cause": f"The run {msg}.",
                                           "next_action": "Start the next part when ready ('Resume' picks it up).",
                                           "exit_code": 0})
                    return 0
            elif start in ("miaplpy", "unwrap", "mintpy"):
                from . import miaplpy_backend as mb, stack_backend as sb
                from .time_series import run_mintpy_with_loader
                if start in ("miaplpy", "unwrap") and getattr(cfg, "mintpy_tropo_method", "no") == "pyaps":
                    # ops-13: the resumed 'Phase linking onward' / 're-unwrap'
                    # run fetched ERA5 serially at the end; now the downloads
                    # overlap the ~12 h of phase linking or unwrapping
                    start_era5_prefetch(cfg)
                if start == "miaplpy":
                    stage = "miaplpy"
                    mb.run(cfg)
                elif start == "unwrap":
                    pairs = pre.get("pair_names") or runs.unwrap_status(cfg.stack_dir)["exported"]
                    log.info("%d exported pairs; engine %s, cost %s, pair coherence %s", len(pairs),
                             cfg.stack_unw_engine, cfg.stack_unw_cost_mode, cfg.miaplpy_pair_coherence)
                    stage = "pair_coherence"
                    mb.write_pair_coherence(cfg, pairs)
                    stage = "unwrap"
                    mb.unwrap_exported(cfg, pairs)
                    stage = "reference"
                    mb.choose_reference(cfg, pairs)
                route_overrides(cfg)
                stage = "mintpy"
                run_mintpy_with_loader(cfg, sb.mintpy_loader_lines(cfg))

            stage = "deliverables"
            log.info("=== Deliverables ===")
            progress.emit(P, "deliverables", "start")
            try:
                # F5: never raises an Exception; what it could not make goes
                # into run_result.json's notes (ERR-01, 16 Oct 2026)
                layers_note = extra_layers_note(cfg, geocode_missing_layers(cfg))
                if layers_note:
                    notes.append(layers_note)
                # a requested unwrap-error step that made nothing (21 Oct 2026)
                unwerr_note = unwrap_correction_note(cfg)
                if unwerr_note:
                    notes.append(unwerr_note)
                from .time_series import config_exclude_dates
                dropped = config_exclude_dates(cfg)
                if dropped:
                    notes.append(f"{len(dropped)} date(s) were left out of the series by the config "
                                 f"(mintpy_exclude_dates): {', '.join(dropped)}.")
                from .weather_raider import model_name, registered_unavailable, unavailable_path
                no_weather = {d: r for d, r in registered_unavailable(cfg).items() if d not in set(dropped)}
                if no_weather:
                    notes.append(f"{len(no_weather)} date(s) had no {model_name(cfg)} weather data and were left out "
                                 "of the series automatically: " + "; ".join(f"{d} ({r})" for d, r in no_weather.items())
                                 + f" ({unavailable_path(cfg).relative_to(cfg.project_dir)}).")
                if str(cfg.backend) == "nisar":        # finished runs too: idempotent (12 Oct 2026)
                    from .time_series import apply_nisar_pair_coherence
                    apply_nisar_pair_coherence(cfg)
                    from .nisar_backend import posting_note
                    grid_note = posting_note(cfg)      # crops on an older grid (28 Oct 2026)
                    if grid_note:
                        notes.append(grid_note)
                from .landcover import ensure as ensure_landcover
                ensure_landcover(cfg)                  # never raises: no map = no tree-cover labels (15 Oct 2026)
                out_dir = export_all(cfg)
            except BaseException as exc:
                progress.emit(P, "deliverables", "failed", msg=repr(exc)[:200])
                log.info("Time series outputs are still in %s", cfg.mintpy_dir)
                raise
            log.info("Deliverables in %s", out_dir)
            progress.emit(P, "deliverables", "done")
            experiment = is_experiment(cfg)
            # nisar since 26 Oct 2026 (Kyle: the comparison run needs the same files and maps;
            # the report and KMZ say it is a NISAR comparison, config.sensor_name)
            if (cfg.backend in ("stack", "miaplpy", "nisar") or (cfg.backend == "disp" and cfg.disp_record_chain)) \
                    and not args.no_acceptance:
                # the cross-check and the injection test BEFORE the quality
                # assessment, which copies their records into
                # quality_assessment.json (after it, the QA carried the
                # previous run's records)
                if cfg.cross_check_disp:
                    stage = "crosscheck"
                    if not run_crosscheck(cfg):
                        failed_stages.append("crosscheck")
                if cfg.injection_test:
                    stage = "injection_test"
                    if not run_injection_test(cfg):
                        failed_stages.append("injection_test")
                stage = "quality_assessment"
                log.info("=== Quality assessment ===")
                if not run_quality_assessment(cfg):
                    failed_stages.append("quality_assessment")
                stage = "figures"
                log.info("=== Figures ===")
                if not run_figures(cfg):
                    failed_stages.append("figures")
                if not experiment:
                    allowed, why = client_report_allowed(cfg)
                    stage = "client_report"
                    if allowed:
                        # a project switched from purpose: experiment builds its report now: the
                        # experiment's "not for clients" note no longer describes it (26 Oct 2026)
                        stale = Path(cfg.project_dir) / "deliverables" / EXPERIMENT_NOTE
                        if stale.exists():
                            stale.unlink()
                            log.info("Removed %s: this run is purpose '%s', not an experiment", stale.name,
                                     getattr(cfg, "purpose", "client"))
                        if not run_client_report(cfg):
                            failed_stages.append("client_report")
                    else:
                        log.warning("%s.", why)
                        moved = set_aside_client_files(cfg)
                        progress.emit(P, "client_report", "skipped", msg="screening route: no client report")
                        notes.append(why + "." + (f" {len(moved)} older client file(s) were moved to "
                                                  "deliverables/screening_not_for_clients/." if moved else ""))
            if experiment:
                # V3: a validation run ends with the harness, never with a
                # client report, release status or package (any route: the
                # harness reads the products, whether or not the QA ran)
                stage = "validation"
                if args.no_acceptance:
                    progress.emit(P, "validation", "skipped", msg="--no-acceptance: no validation")
                elif str(cfg.validation_protocol or "").strip():
                    log.info("=== Validation against ground truth ===")
                    if not run_validation(cfg):
                        failed_stages.append("validation")
                else:
                    progress.emit(P, "validation", "skipped", msg="no validation_protocol in the config")
                    notes.append("No validation was run: the config names no validation_protocol.")
                stage = "client_report"
                log.warning("%s.", EXPERIMENT_NO_REPORT)
                moved = set_aside_client_files(cfg, "experiment")
                progress.emit(P, "client_report", "skipped", msg="experiment run: no client report")
                notes.append(EXPERIMENT_NO_REPORT + "." + (f" {len(moved)} older client file(s) were moved to "
                                                           "deliverables/experiment_not_for_clients/." if moved else ""))
                if not args.no_acceptance:
                    stage = "provenance"
                    if not record_experiment_stages(cfg):
                        failed_stages.append("provenance")
        except BaseException as exc:                         # noqa: BLE001 - reported, then returned
            stopped = _is_stop(exc) and not (wd and wd.tripped)
            cause = exc
            if wd and wd.tripped:
                cause = OSError(errno.ENOSPC, wd.tripped)
            try:
                kind_next, why_next = runs.suggest_kind(P, cfg)
            except Exception:                                # noqa: BLE001
                kind_next, why_next = "resume", ""
            adv = advice.explain(cause, stage, cfg=cfg, restart="resume")
            if why_next and adv.get("category") not in ("busy", "refused"):
                adv["next_action"] += f" (On disk now: {why_next}.)"
            if not stopped:
                log.error("Run failed during %s: %s", stage, exc, exc_info=not isinstance(exc, SystemExit))
            status = "stopped" if stopped else "failed"
            code = exc.code if stopped and isinstance(exc, SystemExit) else (130 if stopped else 1)
            progress.emit(P, "pipeline", "failed", msg=(f"stopped during {stage}" if stopped else
                                                       f"{stage}: {adv['plain_cause']}")[:300])
            write_run_result(cfg, {**res, "status": status, "stage": stage, "exit_code": code,
                                   "failed_stages": failed_stages + [stage], "notes": notes,
                                   "detected_restart": kind_next, "detected_restart_reason": why_next,
                                   **{k: adv.get(k) for k in ("plain_cause", "next_action", "category",
                                                               "restart_kind", "technical")}})
            if not stopped and opts.get("notify", True):
                _notify(cfg, "FAILED", f"{adv['plain_cause']} {adv['next_action']}")
            return code
        finally:
            if wd:
                wd.stop()
            stop_era5_prefetch(cfg)
            procs.track_children_in(None)

        mins = (time.time() - t0) / 60
        if failed_stages:
            names = ", ".join(failed_stages)
            progress.emit(P, "pipeline", "failed", msg=f"{names} failed")
            cause = f"The run finished, but these steps failed: {names}."
            nxt = ("Look in the log for the step's error; 'Products only' re-runs them without the "
                   "processing. If it fails the same way twice, send Kyle the log.")
            write_run_result(cfg, {**res, "status": "failed", "stage": failed_stages[0], "exit_code": 1,
                                   "failed_stages": failed_stages, "plain_cause": cause, "next_action": nxt,
                                   "category": "stage", "restart_kind": "products", "notes": notes})
            if opts.get("notify", True):
                _notify(cfg, "finished with problems", f"{cause} {nxt}")
            return 1
        progress.emit(P, "pipeline", "done", msg=f"{kind}, {mins:.0f} min")
        cause = f"Finished in {mins:.0f} min." + (" " + " ".join(notes) if notes else "")
        nxt = f"Results are in {P / 'deliverables'}."
        write_run_result(cfg, {**res, "status": "done", "stage": "done", "exit_code": 0, "failed_stages": [],
                               "plain_cause": cause, "next_action": nxt, "category": "done", "notes": notes})
        if opts.get("notify", True):
            _notify(cfg, "finished", f"{cause} {nxt}")
        return 0


def _notify(cfg, what: str, text: str) -> None:
    try:
        from . import notify
        notify.send(f"InSAR {Path(cfg.project_dir).name}: {what}", text)
    except Exception as exc:                                 # noqa: BLE001
        log.info("No notification: %s", exc)


def _report_config_error(cfg_path, exc, kind) -> None:
    """A config that does not load is the commonest start-up failure, and it
    used to leave only a traceback in the log. When the config still names a
    project folder that exists, say so in its run_result.json and progress
    feed - unless another run holds the project."""
    try:
        import yaml
        from . import progress, runs
        raw = yaml.safe_load(Path(cfg_path).read_text()) or {}
        P = Path(os.path.expanduser(os.path.expandvars(str(raw.get("project_dir", "")))))
        if not raw.get("project_dir") or not P.is_dir():
            return
        held = runs.lock_holder(P)
        if held is not None and held.get("pid") != os.getpid():
            return
        lines = str(exc).strip().splitlines()
        cause = "The config file has problems: " + "; ".join(x.strip() for x in lines[1:] or lines)
        progress.emit(P, "pipeline", "failed", msg=cause[:300])
        write_run_result(SimpleNamespace(project_dir=P), {
            "status": "failed", "kind": kind, "stage": "config", "category": "config", "exit_code": 1,
            "plain_cause": cause[:1500], "config": str(cfg_path),
            "next_action": "Fix the config (the project builder writes a valid one), then start again.",
            "technical": f"{type(exc).__name__}: {exc}"[:2000], "failed_stages": ["config"]})
    except Exception:                                        # noqa: BLE001
        pass


def launch(cfg_path, kind: str = "full", **opts) -> int:
    """What every launcher calls: teardown + project lock + run_from, for a
    run kind (runs.KINDS: resume / full / products / mintpy / miaplpy /
    reunwrap / plan). Returns the exit code; a run already going on the
    project is refused (exit 3) without touching its run.json, progress feed
    or run_result.json."""
    from . import procs, runs
    procs.install_teardown()
    try:
        cfg = load_config(str(cfg_path))
    except Exception as exc:
        _report_config_error(cfg_path, exc, kind)
        raise
    cfg.ensure_dirs()
    opts.setdefault("config_path", str(cfg_path))
    if kind == "plan":
        return run_from(cfg, "full", kind="plan", plan_only=True, **opts)
    if kind == "resume":
        k, why = runs.suggest_kind(cfg.project_dir, cfg)
        log.info("Resume: %s -> %s", why, runs.KINDS.get(k, (k,))[0])
        opts.setdefault("resumed_as", k)
        start = runs.KIND_START.get(k, "full")
    else:
        start = runs.KIND_START.get(kind, kind)
    try:
        with runs.project_lock(cfg.project_dir, kind=kind, config=str(cfg_path)):
            return run_from(cfg, start, kind=kind, **opts)
    except RuntimeError as exc:
        if "already running" not in str(exc):
            raise
        log.error("Not started: %s", exc)
        return 3


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Run the InSAR pipeline for one config")
    parser.add_argument("config", help="Path to pipeline YAML config")
    parser.add_argument("--resume", action="store_true",
                        help="work out from the files on disk where to pick up (runs.suggest_kind)")
    parser.add_argument("--start", default=None, choices=list(STARTS) + list(START_ALIASES),
                        help="start the chain here instead of at the beginning")
    parser.add_argument("--kind", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--skip-download", action="store_true", help="Reuse SLCs already on disk")
    parser.add_argument("--local-only", action="store_true",
                        help="Process only scenes already downloaded, ignoring the rest of the search")
    parser.add_argument("--plan-only", action="store_true",
                        help="stack/miaplpy backends: size the run and stop, downloading nothing")
    parser.add_argument("--force", action="store_true",
                        help="proceed even if the projected peak disk exceeds the budget, or a restart's "
                             "inputs are incomplete")
    parser.add_argument("--from-step", default=None, dest="from_step",
                        help="stack backend: first run file, by index or name substring")
    parser.add_argument("--to-step", default=None, dest="to_step",
                        help="stack backend: last run file, by index or name substring")
    parser.add_argument("--no-acceptance", action="store_true",
                        help="stack backend: skip the quality assessment")
    parser.add_argument("--no-notify", action="store_true", help="no desktop notification at the end")
    args = parser.parse_args(argv)
    from . import runs
    if not os.environ.get(runs.LOCK_FD_ENV):
        # the build label first (ERR-04, 16 Oct 2026); a run runs.start
        # launched has it as its log's first line already
        log.info("%s", runs.build_label())
    opts = dict(skip_download=args.skip_download, local_only=args.local_only, force=args.force,
                from_step=args.from_step, to_step=args.to_step, no_acceptance=args.no_acceptance,
                notify=not args.no_notify)
    if args.plan_only:
        kind = "plan"
    elif args.resume:
        kind = "resume"
    elif args.start:
        kind = args.kind or {"unwrap": "reunwrap", "reunwrap": "reunwrap"}.get(args.start, args.start)
        if kind in ("download", "coreg"):
            kind = "full"
            opts["skip_download"] = opts["skip_download"] or args.start == "coreg"
    else:
        kind = args.kind or "full"
    code = launch(args.config, kind, **opts)
    if code:
        raise SystemExit(code)


def script_main(argv, start: str | None = None, doc: str = "") -> int:
    """The scripts/ runners (run_pipeline, run_mintpy_only, run_miaplpy_only,
    rebuild_products): `<config.yaml> [pipeline flags]`, with the start
    point fixed by the script, through main() -> launch() -> run_from like
    every other launcher (lock, teardown, run.json, events, run_result.json,
    exit code). They used to copy the stage chain - bare export_all, no
    events, exit 0 when a stage failed, no star -> bridging rule - and two of
    them defaulted to config_site_a_p2.yaml, which no longer exists: the
    config is required. Returns the exit code (0 done, 1 failed, 2 usage,
    3 another run holds the project, 130/143 stopped)."""
    argv = [str(a) for a in argv]
    if not argv or argv[0].startswith("-"):
        print((doc or "usage: <config.yaml> [pipeline flags]").strip(), file=sys.stderr)
        return 2
    logging.basicConfig(level=_log_level(), format="%(asctime)s %(levelname)s %(message)s", force=True)
    if start:
        argv = [argv[0], "--start", start, *argv[1:]]
    try:
        main(argv)
    except SystemExit as exc:
        if exc.code is None or isinstance(exc.code, int):
            return exc.code or 0
        print(exc.code, file=sys.stderr)              # a plain refusal message
        return 1
    return 0


if __name__ == "__main__":
    main()
