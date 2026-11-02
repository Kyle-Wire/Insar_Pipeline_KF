"""
topsStack backend: a controlled Sentinel-1 interferometric chain.

WHY THIS EXISTS
---------------
The "local" backend runs one topsApp per pair. Every pair re-coregisters from
scratch, and the generated XML sets no multilook, no filter strength and no
unwrapper cost mode. For a 250-epoch stack that is both slow and, more
importantly, uncontrollable - which matters because the measured defect in
the product this replaces is a PHASE UNWRAPPING failure.

topsStack coregisters every acquisition once against a single reference, then
forms interferograms off that stack. It exposes multilooking, Goldstein filter
strength, network connectivity and - after the config patch below - the snaphu
cost mode.

THE ONE THING THAT IS NOT ON THE COMMAND LINE
---------------------------------------------
topsStack/Stack.py:11-12 hardcodes:

    noMCF  = 'False'
    defoMax = '2'

and unwrap.py:236 maps noMCF=False to

    runUnwrap(costMode='SMOOTH', initMethod='MCF', initOnly=True)

so SNAPHU'S STATISTICAL COST OPTIMISATION NEVER RUNS. It initialises with
minimum-cost-flow and stops. That is the same weak unwrapper topsApp's
"snaphu_mcf" uses, so migrating without changing it would reproduce the
problem at greater expense.

There is no CLI flag. patch_configs() rewrites the generated
configs/config_igram_unw_* files, which is the supported escape hatch:
stackSentinel's whole design is that the run_files/ and configs/ it emits are
an inspectable, editable plan rather than an opaque execution.

WHY WE EXECUTE THE RUN FILES OURSELVES
--------------------------------------
stackSentinel is always invoked with --num_proc 1. At higher values
Stack.write_wrapper_config2run_file appends " &" to commands and emits bare
"wait" lines - a shell-level batch structure that cannot be resumed into and
inside which a single failure cannot be isolated. At --num_proc 1 every line
is exactly one independent command, so we can run them in a bounded pool,
record per-command status, and restart a 300-pair unwrap at pair 250 instead
of at zero.
"""
import hashlib
import json
import logging
import os
import random
import sys
import re
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from .config import PipelineConfig

log = logging.getLogger(__name__)

# Installed topsStack scripts. Run files invoke these by bare name, and the
# directory is NOT on PATH in the conda environment, so every subprocess needs
# it injected or the first run file fails with "command not found".
_ISCE_SHARE = Path(os.environ.get(
    "ISCE_STACK", str(Path(sys.prefix) / "share" / "isce2")))      # the running env's ISCE2
ISCE_STACK = _ISCE_SHARE / "topsStack"

# Run files that are memory-hungry enough to want their own worker count.
_HEAVY_TOPO = ("topo", "geo2rdr", "resample", "overlap")
_HEAVY_UNWRAP = ("unwrap",)


@dataclass
class StackPlan:
    """What a run would do, computed before anything is downloaded."""
    dates: list = field(default_factory=list)
    n_dates: int = 0
    n_scenes: int = 0
    n_pairs: int = 0
    swaths: str = ""
    bbox_snwe: str = ""
    range_looks: int = 0
    azimuth_looks: int = 0
    est_slc_gb: float = 0.0
    est_coreg_gb: float = 0.0
    est_merged_gb: float = 0.0
    est_peak_gb: float = 0.0
    # disk BUDGET (GB = 1e9) the run can really write: settings.disk_budget_gb,
    # which on WSL is C: free + what the vhdx holds on C: that ext4 does not
    # use (not the vhdx's length: it is sparse), not ext4's virtual free space
    free_gb: float = 0.0
    budget_source: str = ""
    project_gb: float = 0.0           # already allocated under the project (a resume needs peak - this)
    bursts_per_date: int = 0
    bursts_source: str = ""
    crop_fraction: float | None = None   # phase-linking crop / processed frame (miaplpy route)
    crop_source: str = ""
    geo_cells: int = 0                # geocoded grid cells the deliverables read
    est_ram_gb: float = 0.0           # deliverables-stage peak (the whole geocoded cube in memory)
    mem_gb: float = 0.0               # this machine's RAM (settings.machine_resources)
    notes: list = field(default_factory=list)


# ----------------------------------------------------------------------
# environment
# ----------------------------------------------------------------------
def _env() -> dict:
    """
    Subprocess environment with topsStack reachable.

    Both PATH (run files call `SentinelWrapper.py` etc. by bare name) and
    PYTHONPATH (those scripts import each other, e.g. s1a_isce_utils) are
    needed. Missing either produces a failure in the first seconds of the
    first run file, which is exactly what P0 is for.
    """
    env = dict(os.environ)
    # PATH: run files invoke SentinelWrapper.py and friends by bare name.
    env["PATH"] = f"{ISCE_STACK}:{env.get('PATH', '')}"
    # PYTHONPATH needs BOTH directories, because the scripts use two import
    # styles against the same files. stackSentinel.py does
    #     from topsStack.Stack import config
    # which resolves topsStack as an implicit namespace package and therefore
    # needs its PARENT on the path, while other scripts do
    #     from Stack import config
    # which needs the directory itself. There is no __init__.py, so neither
    # style works from the other's path alone. Omitting the parent is what
    # made P0 fail with "No module named 'topsStack'".
    # The unwrap wrapper (`python -m insar_pipeline.unwrap_crop`) runs as a
    # run-file command, so this package's parent has to be importable in the
    # child too.
    pkg_parent = str(Path(__file__).resolve().parent.parent)
    env["PYTHONPATH"] = ":".join(
        [str(_ISCE_SHARE), str(ISCE_STACK), pkg_parent]
        + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    return env


def _wrap_env(cmd: str, cfg: PipelineConfig) -> list:
    """
    Build the argv for one run-file command.

    /usr/bin/time -v gives us peak RSS per command, which is how the RAM
    ceiling stops being a guess. cfg.stack_conda_env lets the heavy geometry
    steps run against a CUDA build of ISCE without disturbing the environment
    everything else uses.

    The run-file line itself is a shell command by design (ISCE writes it);
    the env name around it is a config value and is checked first (16 Oct
    2026, readiness review SEC-02 / RUN-04).
    """
    inner = cmd
    if cfg.stack_conda_env:
        env = _procs.conda_env_arg(cfg.stack_conda_env, "stack_conda_env")
        # a name as before; a folder (quoted, so it may start with ') needs -p
        flag = "-p" if env.startswith(("/", "'")) else "-n"
        inner = f"conda run {flag} {env} --no-capture-output {cmd}"
    return ["/usr/bin/time", "-v", "bash", "-lc", inner]


_RSS_RE = re.compile(r"Maximum resident set size \(kbytes\):\s*(\d+)")


# ----------------------------------------------------------------------
# child process teardown - see procs.py for why every command runs in its
# own session and is killed as a group. Kept as aliases for older callers.
# ----------------------------------------------------------------------
import signal as _signal  # noqa: E402
import threading as _threading  # noqa: E402

from . import procs as _procs  # noqa: E402

_kill_children = _procs.kill_children
_install_teardown = _procs.install_teardown


# ----------------------------------------------------------------------
# run-file parsing and execution
# ----------------------------------------------------------------------
def parse_run_file(path: Path) -> list:
    """
    One command per returned element.

    Tolerates the batch syntax even though we generate with --num_proc 1, so
    that a run file hand-edited by someone (or generated by a future
    stackSentinel) does not silently execute a trailing "&" as part of an
    argument.
    """
    cmds = []
    for raw in Path(path).read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line == "wait":
            continue
        if line.endswith("&"):
            line = line[:-1].strip()
        if line:
            cmds.append(line)
    return cmds


def _status_path(cfg: PipelineConfig, run_name: str, cmd: str) -> Path:
    """
    Where a completed command records itself.

    Deliberately OUTSIDE run_files/. stackSentinel refuses to write into an
    existing run_files/, so any regeneration has to move that directory aside -
    and if the markers lived inside it, every regeneration would silently throw
    away the record of what had already run. Keyed on the command text, so a
    command whose parameters changed gets a new hash and re-runs, while an
    unchanged one is skipped.
    """
    h = hashlib.sha1(cmd.encode()).hexdigest()[:16]
    return cfg.stack_dir / ".status" / run_name / f"{h}.done"


def _config_paths(cmd: str) -> list:
    """Config files a command reads its parameters from (`-c <f>` / `--config <f>`)."""
    toks = cmd.split()
    return [Path(toks[i + 1]) for i, t in enumerate(toks[:-1]) if t in ("-c", "--config") and Path(toks[i + 1]).is_file()]


def _config_sha1(cmd: str) -> str | None:
    """
    Fingerprint of the config files a command points at. The marker is keyed
    on the command TEXT, and for SentinelWrapper / unwrap_crop that text is
    only a config path, so a changed defomax, filter strength or unwrapping
    engine used to leave every pair "done" (review 27 Sep 2026).
    """
    paths = _config_paths(cmd)
    if not paths:
        return None
    h = hashlib.sha1()
    for p in paths:
        h.update(p.read_bytes())
    return h.hexdigest()[:16]


def _marker_stale(marker: Path, rec: dict, cmd: str) -> str | None:
    """Why a 'done' marker no longer describes the command's parameters, or None."""
    now = _config_sha1(cmd)
    if now is None:
        return None
    if rec.get("config_sha1"):
        return None if rec["config_sha1"] == now else "its config file changed since it ran"
    # markers written before the fingerprint existed: judge by modification time
    try:
        m = marker.stat().st_mtime
        if any(p.stat().st_mtime > m + 1 for p in _config_paths(cmd)):
            return "its config file was modified after it ran"
    except OSError:
        return None
    return None


# Peak RSS of one command, for the memory clamp in _workers_for (GB):
# merge_reference_secondary_slc 1.27 measured (site_b run_12 .done
# records), snaphu 0.53 at 11 x 3 looks; anything else is not measured and
# gets 1 GB.
_CMD_PEAK_GB = {"merge": 1.3, "unwrap": 0.6}
_CMD_PEAK_DEFAULT_GB = 1.0
_CLAMP_LOGGED: set = set()


def _workers_for(cfg: PipelineConfig, run_name: str) -> int:
    """The pool size for a run file: the config's, capped at what this
    machine's AVAILABLE memory holds right now (80% of MemAvailable over the
    command's measured peak). A config copied from a bigger machine (or a
    second run on this one) otherwise starts more workers than fit, and the
    kernel's OOM killer takes them one by one with returncode -9."""
    low = run_name.lower()
    if any(k in low for k in _HEAVY_UNWRAP):
        n = max(1, cfg.stack_num_proc_unwrap)
        # snaphu is single-threaded: more workers than cores only queue on
        # the CPU (generality-13: a YAML copied from the 24-core machine
        # asked for 24 on an 8-core laptop)
        cores = os.cpu_count() or 1
        if n > cores:
            if (run_name, n, -cores) not in _CLAMP_LOGGED:
                _CLAMP_LOGGED.add((run_name, n, -cores))
                log.warning("%s: %d unwrap workers configured, but this machine has %d cores and snaphu is "
                            "single-threaded - running %d at a time", run_name, n, cores, cores)
            n = cores
    elif any(k in low for k in _HEAVY_TOPO):
        n = max(1, cfg.stack_num_proc_topo)
    else:
        n = max(1, cfg.stack_num_proc)
    peak = next((g for k, g in _CMD_PEAK_GB.items() if k in low), _CMD_PEAK_DEFAULT_GB)
    try:
        from .settings import machine_resources
        avail = float(machine_resources().get("mem_available_gb") or 0.0)
    except Exception:                                         # noqa: BLE001 - no clamp without a reading
        avail = 0.0
    if avail > 0:
        fit = max(1, int(0.8 * avail / peak))
        if fit < n:
            if (run_name, n, fit) not in _CLAMP_LOGGED:
                _CLAMP_LOGGED.add((run_name, n, fit))
                log.warning("%s: %d workers configured, but %.1f GB of memory is available now and each "
                            "command peaks near %.1f GB - running %d at a time", run_name, n, avail, peak, fit)
            n = fit
    return n


def _threads_for(cfg: PipelineConfig, workers: int) -> int:
    """
    OMP threads per command so a pool fills the machine without
    oversubscribing it.

    MEASURED: with nothing set, topo runs 24 threads (2368% CPU) and geo2rdr
    ~8, so 12 geometry workers would be ~96 threads on 24 cores. snaphu is
    single-threaded whatever this says.
    """
    cores = int(getattr(cfg, "stack_cores", 0) or 0) or (os.cpu_count() or 1)
    return max(1, cores // max(1, workers))


# ----------------------------------------------------------------------
# outcome verification
# ----------------------------------------------------------------------
_CONFIG_RE = re.compile(r"SentinelWrapper\.py\s+-c\s+(\S+)")


def _read_config_kv(path: Path) -> dict:
    d = {}
    if not path.exists():
        return d
    for line in path.read_text().splitlines():
        if ":" in line and not line.startswith(("#", "[")):
            k, v = line.split(":", 1)
            d.setdefault(k.strip(), v.strip())
    return d


def expected_outputs(cfg: PipelineConfig, run_name: str, cmd: str) -> list:
    """
    What must exist on disk for this command to count as done, as a list of
    (path, kind) with kind in {"file", "dir", "merged"}.

    A return code is what the command CLAIMED; this is what is TRUE. Without
    it, a SentinelWrapper that exits 0 after writing a zero-length product
    is recorded as done and skipped on every later resume - the class of
    bug that has cost the most time in this project.
    """
    m = _CONFIG_RE.search(cmd)
    conf_path = None
    if m:
        conf_path = Path(m.group(1))
    elif "unwrap_crop" in cmd:
        mm = re.search(r"--config\s+(\S+)", cmd)
        conf_path = Path(mm.group(1)) if mm else None
    if conf_path is None:
        # Three coregistration commands take no config and used to be marked
        # done on their exit code alone (review 5 Oct 2026: mapping every
        # Site B command through this function gave NONE for run_04,
        # run_08 and run_11). Their products are known from the topsStack
        # scripts themselves.
        return _configless_outputs(cfg, cmd)
    c = _read_config_kv(conf_path)
    out = []
    # Ordered by the config FAMILY's final product, because several
    # families share keys: the misreg config runs generateIgram (key
    # `interferogram`, a coarse overlap product) and then the two misreg
    # estimators whose text files are the actual output. Matching on
    # `interferogram` first failed all 87 misreg commands on the v2 run.
    if "out_azimuth" in c or "out_range" in c:                 # config_misreg_*
        out += [(Path(c[k]), "file") for k in ("out_azimuth", "out_range") if k in c]
    elif "baseline_file" in c:                                  # config_baseline_*
        out += [(Path(c["baseline_file"]), "file")]
    elif "unw" in c and ("unwrap" in c or "unwrap" in run_name):  # config_igram_unw_*
        out += [(Path(c["unw"]), "file"), (Path(c["unw"] + ".conncomp"), "file")]
    elif "filt" in c and "coh" in c:                            # config_igram_filt_coh_*
        out += [(Path(c["filt"]), "file"), (Path(c["coh"]), "file")]
    elif "outfile" in c:                                        # config_merge_*
        if str(c.get("use_virtual_files", "True")) == "False" and \
                str(c.get("multilook", "True")) == "False":
            # A REAL full-resolution merge (stack_virtual_merge False, the
            # phase-linked route): mergeBursts writes <outfile>.full and a
            # .full.vrt beside it, so the old "the .full.vrt exists" clause
            # passed a merged SLC truncated by a filling disk.
            out += [(Path(c["outfile"] + ".full"), "merged_real")]
        else:
            out += [(Path(c["outfile"]), "merged")]
    elif "interferogram" in c:                                  # config_generate_igram_*
        # Per-burst interferograms are deleted by cleanup_bursts once merged,
        # so "done" is either the burst directory or the merged product.
        out += [(Path(c["interferogram"]), "bursts_or_merged")]
    elif "coregSLCdir" in c or "coregdir" in c:                 # geo2rdr / resample
        # The overlap stage (run_05/06) and the full-burst stage (run_09/10)
        # write into the SAME date directory - overlap/IW*/ vs IW*/ - so a
        # bare directory check passes run_09 on run_05's output. Look for the
        # stage's own files: range/azimuth offsets for geo2rdr, burst SLCs
        # for resample; under overlap/ when `overlap : True`.
        d = Path(c.get("coregSLCdir") or c["coregdir"])
        sub = "overlap/IW*" if str(c.get("overlap", "False")) == "True" else "IW*"
        pat = "*.off" if "coregSLCdir" in c else "*.slc"
        out += [((d, f"{sub}/{pat}"), "glob")]
    elif "outdir" in c:                                         # Sentinel1_TOPS unpack
        out += [(Path(c["outdir"]), "dir")]
        if "geom_referenceDir" in c:                            # run_01 also runs topo
            # The per-burst geometry is deleted by cleanup_burst_geometry once
            # merged; its marker stands in for it. Without this clause a
            # project whose geometry had been removed by hand passed run_01 on
            # reference/ alone and topo was never redone.
            out += [(Path(c["geom_referenceDir"]), "dir_or_cleaned")]
    return out


def _stack_swaths(cfg: PipelineConfig) -> list:
    """Sub-swath numbers the stack actually carries: the reference's IW<n>.xml
    once run_01 has unpacked it, else the configured stack_swaths."""
    ref = cfg.stack_dir / "reference"
    have = sorted(p.stem[2:] for p in ref.glob("IW[1-3].xml")) if ref.exists() else []
    return have or str(getattr(cfg, "stack_swaths", "2")).split()


def _configless_outputs(cfg: PipelineConfig, cmd: str) -> list:
    """Expected products of the topsStack commands that take no config file."""
    out = []
    swaths = _stack_swaths(cfg)
    m = re.search(r"(?:^|\s)(?:-m|--reference)\s+(\S+)", cmd)
    if "subsetReference.py" in cmd and m:                      # run_04
        # subsetReference.py:252 saves reference/overlap/IW<n>_top.xml and
        # _bottom.xml once the overlap bursts are cut - the ESD inputs.
        ref = Path(m.group(1))
        for s in swaths:
            out += [(ref / "overlap" / f"IW{s}_top.xml", "file"),
                    (ref / "overlap" / f"IW{s}_bottom.xml", "file")]
    elif "extractCommonValidRegion.py" in cmd and m:           # run_11
        # extractCommonValidRegion.py:105/133 writes the stack's common
        # valid region to <dirname(reference)>/stack/IW<n>.xml - what every
        # merge reads.
        stack = Path(m.group(1)).parent / "stack"
        out += [(stack / f"IW{s}.xml", "file") for s in swaths]
    elif "invertMisreg.py" in cmd:                             # run_08
        # One <date>.txt per date of the misregistration network. topsStack
        # geo2rdr.py:245-249 silently uses a zero azimuth correction when a
        # date's file is missing, so an isolated date was coregistered on
        # geometry alone with no flag. Every planned date must have one -
        # except the dates left out of the stack (their pairs never ran).
        from .data_acquisition import excluded_dates
        mo = re.search(r"(?:-o|--output)\s+(\S+)", cmd)
        if mo:
            dates = sorted({d for f in (cfg.stack_dir / "configs").glob("config_misreg_*")
                            for d in re.findall(r"\d{8}", f.name)} - excluded_dates(cfg))
            out += [(Path(mo.group(1)) / f"{d}.txt", "file") for d in dates]
    return out


# bytes per pixel of the ISCE data types (isceobj/Image: BYTE ... CDOUBLE)
_ISCE_BYTES = {"BYTE": 1, "CHAR": 1, "SHORT": 2, "INT": 4, "LONG": 8, "FLOAT": 4,
               "DOUBLE": 8, "CFLOAT": 8, "CDOUBLE": 16}


def _xml_int(text: str, name: str):
    m = re.search(rf'<property name="{name}">\s*<value>\s*(-?\d+)', text, flags=re.IGNORECASE)
    return int(m.group(1)) if m else None


def isce_size_problem(path: Path) -> str:
    """Why an ISCE raster is shorter than its .xml says, or "" (also "" when
    the xml cannot be read - nothing to judge against). Measured: merged
    hgt.rdr at Site B is 1396 x 2241 DOUBLE = 25,027,488 bytes exactly."""
    xml = Path(str(path) + ".xml")
    try:
        t = xml.read_text(errors="ignore")
        have = Path(path).stat().st_size
    except OSError:
        return ""
    w, ln = _xml_int(t, "width"), _xml_int(t, "length")
    nb = _xml_int(t, "number_bands") or 1
    m = re.search(r'<property name="data_type">\s*<value>\s*(\w+)', t, flags=re.IGNORECASE)
    bpp = _ISCE_BYTES.get(m.group(1).upper()) if m else (8 if ".slc" in Path(path).name else None)
    if not (w and ln and bpp):
        return ""
    want = w * ln * nb * bpp
    if have < want:
        return f"{Path(path).name} is {have} bytes, expected {want} ({w} x {ln})"
    return ""


_BURSTS_CACHE: dict = {}


def _bursts_in_xml(x: Path):
    """numberOfBursts of an ISCE swath product xml (cached on mtime: the
    SAFE cleanup asks once per date after every resample)."""
    try:
        key = (str(x), x.stat().st_mtime_ns)
    except OSError:
        return None
    if key not in _BURSTS_CACHE:
        _BURSTS_CACHE[key] = _xml_int(x.read_text(errors="ignore"), "numberofbursts")
    return _BURSTS_CACHE[key]


def resample_incomplete(date_dir: Path, swaths=None) -> str:
    """
    Why a date's FULL-BURST resample is not complete, or "".

    resamp_withCarrier.py writes burst_NN.slc one burst at a time and saves
    the swath product coreg_secondarys/<date>/IW<n>.xml LAST (line 293),
    listing how many bursts it wrote. The old test - any one non-empty
    burst - was true after the first burst while the others were still
    being read through VRTs into the date's SAFE, which SAFE cleanup then
    deleted (review 5 Oct 2026: phaselinked-9, stackrunner-9).
    """
    date_dir = Path(date_dir)
    xmls = [date_dir / f"IW{s}.xml" for s in swaths] if swaths else \
        sorted(date_dir.glob("IW[1-3].xml"))
    if not xmls:
        return f"no IW*.xml in {date_dir} (resample writes it after the last burst)"
    for x in xmls:
        if not x.exists():
            return f"{x} missing (resample writes it after the last burst)"
        need = _bursts_in_xml(x) or 1
        have = 0
        for p in (date_dir / x.stem).glob("burst_*.slc"):
            try:
                have += p.stat().st_size > 0
            except OSError:
                pass
        if have < need:
            return f"{date_dir.name}/{x.stem}: {have} of {need} burst SLCs written"
    return ""


def verify_outputs(cfg: PipelineConfig, run_name: str, cmd: str) -> str:
    """Empty string when every expected output is present; else the reason."""
    for path, kind in expected_outputs(cfg, run_name, cmd):
        if kind == "file":
            if not path.exists() or path.stat().st_size == 0:
                return f"missing or empty output {path}"
        elif kind == "dir":
            if not path.is_dir() or not any(path.rglob("*")):
                return f"missing or empty output directory {path}"
        elif kind == "dir_or_cleaned":
            # Consumed: the marker, or (projects cleaned before the marker
            # existed) the merged geometry that cleanup_burst_geometry
            # requires before it deletes anything.
            merged = path.parent / "merged" / path.name
            if not (path.is_dir() and any(path.rglob("*"))) and \
                    not path.with_name(path.name + ".cleaned").exists() and \
                    not (merged / "hgt.rdr").exists():
                return f"missing or empty output directory {path} (and no {path.name}.cleaned)"
        elif kind == "glob":
            d, pat = path
            d = Path(d)
            hits = [p for p in d.glob(pat) if p.stat().st_size > 0]
            cleaned = _consumed_and_cleaned(d, pat)
            if not hits and not cleaned:
                return f"no non-empty {pat} under {d}"
            if pat == "IW*/*.slc" and not cleaned:
                # full-burst resample: every burst, not the first one
                why = resample_incomplete(d)
                if why:
                    return why
        elif kind == "merged_real":
            if path.exists():
                why = isce_size_problem(path)
                if why:
                    return f"merged product truncated: {why}"
            elif not (path.parent / "DELETED_AFTER_MIAPLPY_LOAD").exists():
                # the phase-linked route deletes merged SLCs only after
                # slcStack.h5 is verified, and leaves that marker
                return f"real merge produced no {path}"
        elif kind == "bursts_or_merged":
            has_bursts = path.is_dir() and any(path.glob("IW*/*.int"))
            merged = cfg.stack_dir / "merged" / "interferograms" / path.name
            has_merged = any((merged / n).exists() and (merged / n).stat().st_size > 0
                             for n in ("fine.int", "filt_fine.int"))
            if not (has_bursts or has_merged):
                return f"neither burst interferograms in {path} nor a merged product"
        elif kind == "merged":
            # A virtual merge without multilook writes only <out>.full.vrt;
            # the unfiltered interferogram may have been cleaned after
            # filtering, in which case the filtered product is the evidence.
            ok = (path.exists() and path.stat().st_size > 0) or \
                Path(str(path) + ".full.vrt").exists() or \
                (path.parent / "fine.cleaned").exists() or \
                ((path.parent / "filt_fine.int").exists()
                 and (path.parent / "filt_fine.int").stat().st_size > 0)
            if not ok:
                return f"merge produced neither {path} nor {path}.full.vrt"
            if path.exists() and path.stat().st_size > 0:
                why = isce_size_problem(path)
                if why:
                    return f"merged product truncated: {why}"
    # Unwrapped phase is 2 floats per pixel, the interferogram 1 complex:
    # equal byte counts. A truncated unwrap is caught here, not in MintPy.
    if "unwrap" in run_name.lower():
        m = _CONFIG_RE.search(cmd) or re.search(r"--config\s+(\S+)", cmd)
        c = _read_config_kv(Path(m.group(1))) if m else {}
        if c.get("ifg") and c.get("unw"):
            ifg, unw = Path(c["ifg"]), Path(c["unw"])
            if ifg.exists() and unw.exists():
                if unw.stat().st_size != ifg.stat().st_size:
                    return (f"{unw.name} is {unw.stat().st_size} bytes, expected "
                            f"{ifg.stat().st_size} (same as {ifg.name})")
                cc = Path(str(unw) + ".conncomp")
                if cc.exists() and cc.stat().st_size * 8 != ifg.stat().st_size:
                    return f"{cc.name} is {cc.stat().st_size} bytes, expected " \
                           f"{ifg.stat().st_size // 8}"
    return ""


def _consumed_and_cleaned(d: Path, pat: str) -> bool:
    """
    True when the products a glob check asks for were deliberately removed
    by a cleanup that left its marker - the stage ran and its outputs were
    consumed. Which markers count depends on what was asked for:

      overlap/IW*/*.off   overlap_offsets.cleaned, overlap.cleaned, <date>.cleaned
      overlap/IW*/*.slc   overlap.cleaned, <date>.cleaned
      IW*/*.off           offsets.cleaned, <date>.cleaned
      IW*/*.slc           <date>.cleaned
    """
    date_marker = d.with_name(d.name + ".cleaned")
    if date_marker.exists():
        return True
    if pat.startswith("overlap/"):
        if (d / "overlap.cleaned").exists():
            return True
        return pat.endswith(".off") and (d / "overlap_offsets.cleaned").exists()
    return pat.endswith(".off") and (d / "offsets.cleaned").exists()


def pair_of(cmd: str):
    m = re.search(r"(\d{8})_(\d{8})", cmd)
    return m.group(0) if m else None


def quarantine_unwrap(cfg: PipelineConfig, cmd: str, reason: str) -> Path | None:
    """
    Move a killed unwrap's partial output out of merged/interferograms.

    snaphu writes the connected components BEFORE integrating the phase, and
    unwrap.py masks in a third pass, so a kill leaves files that exist, have
    the right size, and are wrong. MintPy's loader globs *.unw and would take
    them. Only the unwrap products move; the filtered interferogram stays so
    the pair can be retried with a longer limit.
    """
    m = _CONFIG_RE.search(cmd) or re.search(r"--config\s+(\S+)", cmd)
    c = _read_config_kv(Path(m.group(1))) if m else {}
    if not c.get("unw"):
        return None
    unw = Path(c["unw"])
    pair = unw.parent.name
    qdir = cfg.stack_dir / "quarantine_unwrap_timeout" / pair
    moved = []
    for p in sorted(unw.parent.glob(unw.name + "*")) + \
            sorted(unw.parent.glob("filt_fine_crop.*")):
        qdir.mkdir(parents=True, exist_ok=True)
        shutil.move(str(p), str(qdir / p.name))
        moved.append(p.name)
    # the record of WHY is written even when nothing was left to move (a pair
    # that failed before writing any output); it used to raise here
    qdir.mkdir(parents=True, exist_ok=True)
    (qdir / "REASON.txt").write_text(
        f"{time.strftime('%Y-%m-%dT%H:%M:%S')}\n{reason}\n{cmd}\n"
        f"moved: {moved}\n")
    log.warning("Quarantined %s -> %s (%s)", pair, qdir, reason.splitlines()[0])
    return qdir


def quarantined_pairs(cfg: PipelineConfig) -> list:
    q = cfg.stack_dir / "quarantine_unwrap_timeout"
    return sorted(d.name for d in q.iterdir() if d.is_dir()) if q.exists() else []


def check_network_connectivity(dates: list, pairs: list):
    """
    (connected, orphaned_dates, components) for a pair network.

    Pure graph arithmetic. Doing it by hand is feasible when three pairs are
    cut; at 750 pairs with a timeout nobody will.
    A disconnected network is not "some dates missing" - MintPy inverts each
    piece against its own datum and the result looks fine.
    """
    dates = sorted(set(dates))
    adj = {d: set() for d in dates}
    for p in pairs:
        a, b = p.split("_")
        if a in adj and b in adj:
            adj[a].add(b)
            adj[b].add(a)
    orphaned = [d for d in dates if not adj[d]]
    seen, components = set(), []
    for d in dates:
        if d in seen or d in orphaned:
            continue
        stack, comp = [d], []
        seen.add(d)
        while stack:
            x = stack.pop()
            comp.append(x)
            for y in adj[x]:
                if y not in seen:
                    seen.add(y)
                    stack.append(y)
        components.append(sorted(comp))
    components.sort(key=len, reverse=True)
    connected = len(components) == 1 and not orphaned
    return connected, orphaned, components


def _misreg_pair_estimated(cmd: str) -> bool:
    """True when a misreg pair's azimuth estimate exists with a finite median.
    invertMisreg.py replaces a NaN median by 0.0 (design_matrix), so a pair
    that 'succeeded' with no coherent overlap pixels is no evidence at all."""
    m = _CONFIG_RE.search(cmd)
    c = _read_config_kv(Path(m.group(1))) if m else {}
    if not c.get("out_azimuth"):
        return True                      # cannot judge from here: trust the exit code
    p = Path(c["out_azimuth"])
    try:
        txt = p.read_text(errors="ignore")
    except OSError:
        return False
    if not txt.strip():
        return False
    for line in txt.splitlines():
        if line.split(":")[0].strip().lower() == "median":
            try:
                v = float(line.split(":", 1)[1])
            except ValueError:
                return False
            return v == v                                    # False for NaN
    return True


def check_misreg_network(cfg: PipelineConfig, run_name: str, results: list) -> dict:
    """
    Is every date still tied into the ESD misregistration network?

    run_07 failures were tolerated up to 25% on the claim that "each date
    keeps its other connections", which nothing checked. A date whose every
    pair failed (a snow date, all 3-6 pairs) gets no <date>.txt from
    invertMisreg, and topsStack geo2rdr then uses a zero azimuth correction
    for it without a word - burst-seam phase jumps in every interferogram
    with that date. A network that falls into pieces is worse: invertMisreg
    solves it with a pseudo-inverse and the pieces float against each other.
    Writes stack/misreg_network.json; raises when not connected.
    """
    every = {d for r in results for p in [pair_of(r.get("cmd", ""))] if p for d in p.split("_")}
    ok = [pair_of(r["cmd"]) for r in results
          if r.get("returncode") == 0 and pair_of(r.get("cmd", ""))
          and _misreg_pair_estimated(r["cmd"])]
    nan_pairs = sorted(pair_of(r["cmd"]) for r in results
                       if r.get("returncode") == 0 and pair_of(r.get("cmd", ""))
                       and pair_of(r["cmd"]) not in ok)
    connected, orphaned, components = check_network_connectivity(sorted(every), ok)
    rec = {"run_file": run_name, "dates": len(every), "pairs": len(results),
           "pairs_estimated": len(ok), "pairs_without_estimate": nan_pairs,
           "connected": connected, "orphaned_dates": orphaned,
           "components": [[c[0], c[-1], len(c)] for c in components],
           "checked": time.strftime("%Y-%m-%dT%H:%M:%S")}
    try:
        (cfg.stack_dir / "misreg_network.json").write_text(json.dumps(rec, indent=1))
    except OSError:
        pass
    if not every:
        return {"misreg_network_connected": None}
    if not connected:
        pieces = ", ".join(f"{c[0]}..{c[-1]} ({len(c)})" for c in components)
        raise RuntimeError(
            f"{run_name}: the burst-overlap (ESD) estimates no longer connect every "
            f"date. {len(orphaned)} date(s) have no estimated pair"
            + (f" ({', '.join(orphaned[:10])}{'...' if len(orphaned) > 10 else ''})" if orphaned else "")
            + (f"; the rest fall into {len(components)} separate groups ({pieces})" if len(components) > 1 else "")
            + ". Those dates would be aligned on orbits alone, which leaves phase "
            "jumps at burst edges in every interferogram they take part in. Lower "
            f"esd_coherence_threshold (now {cfg.esd_coherence_threshold}) or set "
            "do_esd: false to align every date the same way, then re-run; finished "
            "commands are skipped. Details: stack/misreg_network.json.")
    log.info("%s: misregistration network checked - %d of %d pairs estimated, "
             "every one of %d dates connected.", run_name, len(ok), len(results), len(every))
    return {"misreg_network_connected": True}


def _longest_first(cfg: PipelineConfig, run_name: str, order: list) -> list:
    """
    Submit the slow pairs first.

    The pool is work-conserving, so the wall clock is max(longest job,
    total/workers); what ordering controls is how much idle time collects at
    the tail. Measured on a one-year stack: FIFO happened to be near-optimal only
    because it STARTED in June and the summer pairs sat first in the
    run file. Over a decade they are interleaved. Key: this pair's duration
    from a previous run when one exists, otherwise the month of the first
    date (May-September first, MEASURED at 10-50x the winter unwrap time).
    """
    history = _unwrap_history(cfg)

    def key(item):
        _, cmd = item
        p = pair_of(cmd) or ""
        if p in history:
            return (0, -float(history[p]), p)
        month = int(p[4:6]) if len(p) >= 6 else 0
        summer = 1 if 5 <= month <= 9 else 0
        return (1, -summer, p)
    return sorted(order, key=key)


def _unwrap_history(cfg: PipelineConfig) -> dict:
    """pair -> seconds from every unwrap this stack has ever run. Survives
    invalidation (status markers do not), so a re-run orders on measurements."""
    p = cfg.stack_dir / "unwrap_history.json"
    if p.exists():
        try:
            return json.loads(p.read_text())
        except json.JSONDecodeError:
            pass
    # First run under this code: harvest whatever status markers exist.
    hist = {}
    for d in (cfg.stack_dir / ".status").glob("run_*unwrap*"):
        for m in d.glob("*.done"):
            try:
                rec = json.loads(m.read_text())
            except (json.JSONDecodeError, OSError):
                continue
            pr = pair_of(rec.get("cmd", ""))
            if pr and rec.get("seconds"):
                hist[pr] = max(float(rec["seconds"]), hist.get(pr, 0.0))
    return hist


def _record_unwrap_history(cfg: PipelineConfig, results: list) -> None:
    hist = _unwrap_history(cfg)
    for r in results:
        p = pair_of(r.get("cmd", ""))
        if p and not r.get("skipped") and r.get("seconds"):
            hist[p] = float(r["seconds"])      # latest measurement, not max:
                                               # a re-run after a fix runs FASTER
    (cfg.stack_dir / "unwrap_history.json").write_text(json.dumps(hist, indent=1, sort_keys=True))


def _run_one(cfg: PipelineConfig, run_name: str, idx: int, cmd: str,
             workers: int = 1) -> dict:
    """Execute one command, recording status so a rerun skips it."""
    if _procs.stopping():
        return {"cmd": cmd, "returncode": -15, "seconds": 0.0, "peak_rss_gb": None,
                "skipped": False, "reason": "run was stopped before this command started"}
    marker = _status_path(cfg, run_name, cmd)
    if cfg.skip_existing and marker.exists():
        try:
            rec = json.loads(marker.read_text())
            if rec.get("returncode") == 0:
                # The marker says done; the parameters and the disk have to agree.
                why = _marker_stale(marker, rec, cmd) or verify_outputs(cfg, run_name, cmd)
                if not why:
                    return {**rec, "skipped": True}
                log.warning("%s: status marker says done but %s - re-running",
                            run_name, why)
                marker.unlink()
        except (json.JSONDecodeError, OSError):
            pass

    log_dir = cfg.stack_dir / "logs" / run_name
    log_dir.mkdir(parents=True, exist_ok=True)
    short = re.sub(r"[^A-Za-z0-9_.-]+", "_", cmd.split()[0])[:40]
    log_file = log_dir / f"{idx:04d}_{short}.log"

    env = _env()
    threads = _threads_for(cfg, workers)
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        env[var] = str(threads)

    t0 = time.time()
    proc = _procs.popen_group(_wrap_env(cmd, cfg), cwd=cfg.stack_dir, env=env,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              text=True)
    limit = 0.0
    if "unwrap" in run_name.lower():
        limit = float(getattr(cfg, "stack_unwrap_timeout_min", 0) or 0) * 60
    timed_out = False
    try:
        try:
            out, err = proc.communicate(timeout=limit or None)
        except subprocess.TimeoutExpired:
            timed_out = True
            try:
                os.killpg(proc.pid, _signal.SIGKILL)
            except ProcessLookupError:
                pass
            out, err = proc.communicate()
            err = (err or "") + (f"\nKILLED: exceeded stack_unwrap_timeout_min "
                                 f"({limit/60:.0f} min). Low-coherence pairs can "
                                 f"run unbounded under full-DEFO snaphu.")
            proc.returncode = -9
    finally:
        _procs.unregister(proc.pid)
    proc = subprocess.CompletedProcess(proc.args, proc.returncode, out, err)
    elapsed = time.time() - t0

    log_file.write_text(f"$ {cmd}\n\n--- stdout ---\n{proc.stdout}\n"
                        f"--- stderr ---\n{proc.stderr}\n")
    m = _RSS_RE.search(proc.stderr or "")
    rec = {"cmd": cmd, "returncode": proc.returncode, "config_sha1": _config_sha1(cmd),
           "seconds": round(elapsed, 1),
           "peak_rss_gb": round(int(m.group(1)) / 1024 / 1024, 2) if m else None,
           "log": str(log_file), "skipped": False, "omp_threads": threads}
    if proc.returncode == 0:
        why = verify_outputs(cfg, run_name, cmd)
        if why:
            rec["returncode"] = -1
            rec["reason"] = why
            log.error("EXITED 0 BUT %s: %s\n  see %s", why, cmd[:140], log_file)
        else:
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_text(json.dumps(rec))
    else:
        if proc.returncode == -9 and not timed_out:
            # SIGKILL from outside the pipeline: on WSL that is the kernel's
            # OOM killer (journalctl -k shows "Killed process <pid>")
            rec["reason"] = "killed (most likely out of memory)"
        log.error("FAILED (%s%s) %s\n  see %s", proc.returncode,
                  f", {rec['reason']}" if rec.get("reason") else "", cmd[:140], log_file)
    if "unwrap" in run_name.lower() and rec["returncode"] != 0:
        # snaphu writes the components before the phase and unwrap.py masks in
        # a third pass, so ANY failed unwrap (not only a timeout) can leave
        # files that exist, have the right size and are wrong - or an older
        # unwrap from other settings. MintPy's loader would take either.
        why = (f"timed out after {elapsed/60:.0f} min" if timed_out
               else rec.get("reason") or f"exit code {rec['returncode']}")
        q = quarantine_unwrap(cfg, cmd, why)
        rec["quarantined"] = str(q) if q else None
    return rec


def _progress_counter(cfg: PipelineConfig, stage: str, total: int):
    """A thread-safe tick() that reports commands finished in a stage to
    <project>/progress.jsonl (progress.py), for the dashboard."""
    import threading
    from . import progress
    lock = threading.Lock()
    state = {"done": 0, "failed": 0, "skipped": 0}
    progress.emit(cfg.project_dir, stage, "start", done=0, total=total, skipped=0)

    def tick(res: dict) -> None:
        with lock:
            state["done"] += 1
            state["failed"] += int(res.get("returncode") != 0)
            # Commands only VERIFIED on disk (a resume) finish in milliseconds.
            # Counted in `done` they made the ETA (elapsed / done) collapse:
            # 600 of 804 skipped in 2 s, then 10 real pairs in 120 s, read
            # "39 s left" against a true ~40 min. progress.eta_seconds takes
            # the rate from done - skipped.
            state["skipped"] += int(bool(res.get("skipped")))
            fin = state["done"] >= total
            note = (f"{state['skipped']} verified from the last run" if state["skipped"] else "")
            progress.emit(cfg.project_dir, stage,
                          ("failed" if state["failed"] else "done") if fin else "running",
                          done=state["done"], total=total, skipped=state["skipped"],
                          msg="; ".join(x for x in (f"{state['failed']} failed" if state["failed"] else "",
                                                    note) if x))
    return tick


def plan_commands(cfg: PipelineConfig, path: Path) -> list:
    """parse_run_file without the commands of the dates left out of the stack
    (stack_exclude_dates). A plan made with -x has none; a plan made before
    the exclusion keeps them in its run files, and they are skipped here."""
    from .data_acquisition import excluded_dates
    cmds = parse_run_file(path)
    ex = excluded_dates(cfg)
    if not ex:
        return cmds
    pat = re.compile(r"(?<!\d)(" + "|".join(sorted(ex)) + r")(?!\d)")
    keep = [c for c in cmds if not pat.search(c)]
    if len(keep) < len(cmds):
        log.info("%s: skipping %d command(s) of the date(s) left out of the stack (stack_exclude_dates: %s)",
                 Path(path).name, len(cmds) - len(keep), ", ".join(sorted(ex)))
    return keep


def set_aside_excluded(cfg: PipelineConfig) -> list:
    """
    Move the excluded dates' part-made coregistered images (and merged SLCs)
    to stack/excluded_dates/. extractCommonValidRegion globs every
    coreg_secondarys/2* directory and MiaplPy every merged/SLC date, so a
    date whose commands are skipped must not stay where they look. Moved,
    not deleted: taking the date out of stack_exclude_dates and moving the
    folders back restores it. Refuses the stack's reference date.
    """
    from .data_acquisition import excluded_dates
    ex = excluded_dates(cfg)
    if not ex:
        return []
    ref = ex & _reference_date_set(cfg)
    if ref:
        raise RuntimeError(
            f"stack_exclude_dates lists {', '.join(sorted(ref))}, this stack's reference date: every other "
            "date is aligned to it, so it cannot be left out. Take it out of stack_exclude_dates.")
    moved = []
    for sub in ("coreg_secondarys", "merged/SLC"):
        for d in sorted(ex):
            for name in (d, f"{d}.cleaned"):
                src = cfg.stack_dir / sub / name
                if not (src.exists() or src.is_symlink()):
                    continue
                dst = cfg.stack_dir / "excluded_dates" / sub / name
                if dst.exists():
                    dst = dst.with_name(f"{name}.{time.strftime('%Y%m%d_%H%M%S')}")
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(src), str(dst))
                moved.append(f"{sub}/{name}")
    if moved:
        log.warning("Left out of the stack (stack_exclude_dates): moved %s to %s",
                    ", ".join(moved), cfg.stack_dir / "excluded_dates")
    return moved


def execute_run_file(cfg: PipelineConfig, path: Path, fail_fast: bool, after=None) -> dict:
    """
    Run every command in one run file, in a bounded pool.

    `after(cmd, result)` is called in the worker right after a command
    returns 0 (or was verified-skipped): the hook for per-command cleanups
    that must not wait for the whole stage, such as deleting a date's
    coregistered bursts once its merged SLC is on disk.

    Commands WITHIN a run file are independent; run files are not, so the
    caller never overlaps them. That barrier is the whole ordering guarantee -
    run_03 genuinely needs run_02 to have finished.
    """
    run_name = path.name
    cmds = plan_commands(cfg, path)
    if not cmds:
        log.info("%s: nothing to do", run_name)
        return {"run_file": run_name, "n": 0, "failed": [], "seconds": 0.0}

    workers = _workers_for(cfg, run_name)
    order = list(enumerate(cmds))
    if "unwrap" in run_name.lower():
        order = _longest_first(cfg, run_name, order)
    log.info("%s: %d commands, %d workers x %d threads", run_name, len(cmds),
             workers, _threads_for(cfg, workers))
    t0 = time.time()
    tick = _progress_counter(cfg, run_name, len(cmds))
    pool = ThreadPoolExecutor(max_workers=workers)
    try:
        def _one(i, c):
            res = _run_one(cfg, run_name, i, c, workers)
            tick(res)
            if after is not None and res.get("returncode") == 0:
                try:
                    after(c, res)
                except Exception:                                 # noqa: BLE001
                    log.exception("%s: post-command hook failed for command %d", run_name, i)
            return res
        futs = {i: pool.submit(_one, i, c) for i, c in order}
        results = []
        for i in range(len(cmds)):
            try:
                results.append(futs[i].result())
            except Exception as exc:                          # noqa: BLE001
                # One command's bookkeeping blowing up must not discard the
                # record of the others, which have run to completion anyway.
                log.exception("%s: command %d raised", run_name, i)
                results.append({"cmd": cmds[i], "returncode": -2, "seconds": 0.0,
                                "peak_rss_gb": None, "skipped": False,
                                "reason": repr(exc)})
    finally:
        # A SIGTERM arrives here as SystemExit; queued commands must not
        # start during the drain.
        pool.shutdown(wait=True, cancel_futures=True)

    return _finish_run_file(cfg, run_name, cmds, results, t0, fail_fast)


def _finish_run_file(cfg: PipelineConfig, run_name: str, cmds: list, results: list,
                     t0: float, fail_fast: bool) -> dict:
    """Summarise one run file's results and apply its failure policy.
    Shared by execute_run_file and execute_fused."""
    failed = [r for r in results if r["returncode"] != 0]
    ran = [r for r in results if not r["skipped"]]
    peaks = [r["peak_rss_gb"] for r in ran if r["peak_rss_gb"]]
    summary = {
        "run_file": run_name,
        "n": len(cmds),
        "n_skipped": len(results) - len(ran),
        "failed": [r["cmd"] for r in failed],
        "seconds": round(time.time() - t0, 1),
        "peak_rss_gb": max(peaks) if peaks else None,
        "median_seconds_per_cmd": (
            round(sorted(r["seconds"] for r in ran)[len(ran) // 2], 1) if ran else None),
        "max_seconds_per_cmd": (round(max(r["seconds"] for r in ran), 1) if ran else None),
        "sum_seconds": round(sum(r["seconds"] for r in ran), 1),
        "quarantined": sorted(p for r in results if r.get("quarantined") for p in [pair_of(r["cmd"])] if p),
    }
    if "timeseries_misreg" in run_name:
        # The inversion itself (run_08) is two commands, not a redundant
        # network: if it did not write a correction for every date, the dates
        # without one are resampled with a zero correction (geo2rdr.py:245).
        if failed:
            raise RuntimeError(
                f"{run_name}: the misregistration inversion did not produce a "
                f"correction for every date ({failed[0].get('reason') or 'exit code ' + str(failed[0]['returncode'])}). "
                "See stack/misreg_network.json and the run_07 logs.")
        return summary
    if "misreg" in run_name:
        # Checked, not claimed: the per-date inversion is determined only if
        # the pairs that WERE estimated still connect every date.
        summary.update(check_misreg_network(cfg, run_name, results))
    if failed and "misreg" in run_name:
        # Redundant, but not unboundedly so: if a large share fails the ESD
        # threshold is wrong for this terrain rather than a few pairs being
        # unlucky, and every per-date correction downstream is then suspect.
        frac = len(failed) / len(cmds)
        dates = sorted({d for r in failed for p in [pair_of(r["cmd"])] if p
                        for d in p.split("_")})
        log.warning("%s: %d of %d pairs (%.0f%%) could not be estimated - "
                    "burst overlaps below the ESD coherence threshold. Dates "
                    "affected: %s. Checked: the estimated pairs still connect "
                    "every date, so the per-date inversion remains determined.",
                    run_name, len(failed), len(cmds), 100 * frac,
                    ", ".join(dates[:10]) + ("..." if len(dates) > 10 else ""))
        if frac > 0.25:
            raise RuntimeError(
                f"{run_name}: {len(failed)} of {len(cmds)} pairs failed "
                f"({frac:.0%}). Above 25% this is the ESD coherence threshold "
                f"being wrong for this terrain, not bad luck. Lower "
                f"esd_coherence_threshold (currently "
                f"{cfg.esd_coherence_threshold}) or set do_esd: false to "
                f"coregister on geometry alone.")

    if "unwrap" in run_name:
        _record_unwrap_history(cfg, results)
        pairs_ok = [pair_of(r["cmd"]) for r in results if r["returncode"] == 0]
        every = {d for r in results for p in [pair_of(r["cmd"])] if p for d in p.split("_")}
        connected, orphaned, components = check_network_connectivity(
            sorted(every), [p for p in pairs_ok if p])
        summary["network_connected"] = connected
        summary["orphaned_dates"] = orphaned
        if orphaned:
            log.warning("%d date(s) have NO successfully unwrapped pair and will "
                        "drop out of the time series: %s", len(orphaned),
                        ", ".join(orphaned))
        if not connected:
            raise RuntimeError(
                f"{run_name}: the unwrapped network is NOT connected - it falls "
                f"into {len(components)} pieces "
                f"({', '.join(f'{c[0]}..{c[-1]} ({len(c)})' for c in components)}). "
                "MintPy would invert each piece against its own datum. Recover "
                "the quarantined pairs with a longer stack_unwrap_timeout_min or "
                "exclude the dates on the small side.")

    if failed and len(failed) == len(cmds) and len(cmds) > 2:
        # Every command failing is a broken state, not unlucky data, and
        # "continuing - these pairs will be excluded" is actively misleading
        # when the exclusion is the entire stack. Stop and say so.
        raise RuntimeError(
            f"{run_name}: ALL {len(cmds)} commands failed. This is a pipeline "
            f"state problem rather than bad data - most often inputs that were "
            f"cleaned away while their status markers survived. Check "
            f"{cfg.stack_dir / 'logs' / run_name} and consider clearing the "
            f"markers for the stage that produces this stage's inputs.")

    if failed:
        msg = f"{run_name}: {len(failed)} of {len(cmds)} commands failed"
        if fail_fast:
            raise RuntimeError(
                msg + " - this run file is in the coregistration chain, where a "
                      "missing product poisons everything downstream. Fix and rerun; "
                      "completed commands will be skipped.")
        q = sorted(p for p in summary["quarantined"] if p)
        if q:
            log.warning("%s - continuing. Their outputs were quarantined to %s; these %d pair(s) are missing "
                        "from the time series: %s", msg, cfg.stack_dir / "quarantine_unwrap_timeout", len(q),
                        ", ".join(q[:10]) + (" ..." if len(q) > 10 else ""))
        else:
            log.warning("%s - continuing without them (logs in %s)", msg, cfg.stack_dir / "logs" / run_name)
    return summary


# ----------------------------------------------------------------------
# fused geo2rdr -> resample, per date
# ----------------------------------------------------------------------
_FUSED = (("overlap_geo2rdr", "overlap_resample"),
          ("fullBurst_geo2rdr", "fullBurst_resample"))
_housekeeping_lock = _threading.Lock()      # one cleanup at a time


def _date_of(cmd: str):
    m = re.search(r"_(\d{8})\s*$", cmd.strip())
    return m.group(1) if m else None


def _fused_partner(files: list, idx: int):
    """Index of the resample run file that pairs with files[idx], or None."""
    name = files[idx].name
    for geo, res in _FUSED:
        if geo in name and idx + 1 < len(files) and res in files[idx + 1].name:
            return idx + 1
    return None


def execute_fused(cfg: PipelineConfig, f_geo: Path, f_res: Path, fail_fast: bool) -> list:
    """
    Run geo2rdr and resample as ONE per-date chain instead of two stages.

    Stage-major order means every date's range/azimuth offsets (MEASURED
    862 MB per date, the same size as the coregistered SLCs they produce)
    exist at once before the first resample starts - 236 GB at 274 dates,
    for files that are dead the moment their own date's resample finishes.
    Per date, the chain is geo2rdr -> resample -> delete the offsets ->
    (full-burst stage only) delete the SAFE the resample just consumed, so
    the transient never accumulates. Same commands, same per-command status
    markers, same failure policy; only the interleaving changes.
    """
    cmds_g, cmds_r = plan_commands(cfg, f_geo), plan_commands(cfg, f_res)
    by_g = {_date_of(c): (i, c) for i, c in enumerate(cmds_g)}
    by_r = {_date_of(c): (i, c) for i, c in enumerate(cmds_r)}
    if None in by_g or None in by_r or set(by_g) != set(by_r):
        log.warning("%s / %s do not pair up by date - running them as separate "
                    "stages", f_geo.name, f_res.name)
        s1 = execute_run_file(cfg, f_geo, fail_fast)
        s2 = execute_run_file(cfg, f_res, fail_fast)
        return [s1, s2]
    workers = _workers_for(cfg, f_geo.name)
    overlap = "overlap" in f_geo.name
    log.info("%s + %s: %d dates as fused chains, %d workers x %d threads",
             f_geo.name, f_res.name, len(by_g), workers, _threads_for(cfg, workers))
    t0 = time.time()

    def chain(date):
        ig, cg = by_g[date]
        rg = _run_one(cfg, f_geo.name, ig, cg, workers)
        if rg["returncode"] != 0:
            ir, cr = by_r[date]
            return rg, {"cmd": cr, "returncode": -3, "seconds": 0.0, "peak_rss_gb": None,
                        "skipped": False, "reason": "geo2rdr failed for this date"}
        ir, cr = by_r[date]
        rr = _run_one(cfg, f_res.name, ir, cr, workers)
        if rr["returncode"] == 0:
            # Housekeeping runs under a lock (cleanup_safes scans the whole
            # SLC directory, and two chains finishing together raced on a
            # SAFE one of them was deleting) and can never fail the chain:
            # both commands have already verified their outputs.
            try:
                with _housekeeping_lock:
                    _after_resample(cfg, date, overlap)
            except Exception as exc:                      # noqa: BLE001
                log.warning("post-resample housekeeping for %s failed (%s) - "
                            "products are intact; cleanup will retry later", date, exc)
        return rg, rr

    res_g = [None] * len(cmds_g)
    res_r = [None] * len(cmds_r)
    pool = ThreadPoolExecutor(max_workers=workers)
    try:
        futs = {d: pool.submit(chain, d) for d in sorted(by_g)}
        for d, fut in futs.items():
            try:
                rg, rr = fut.result()
            except Exception as exc:                          # noqa: BLE001
                log.exception("fused chain for %s raised", d)
                rg = {"cmd": by_g[d][1], "returncode": -2, "seconds": 0.0,
                      "peak_rss_gb": None, "skipped": False, "reason": repr(exc)}
                rr = {"cmd": by_r[d][1], "returncode": -2, "seconds": 0.0,
                      "peak_rss_gb": None, "skipped": False, "reason": repr(exc)}
            res_g[by_g[d][0]] = rg
            res_r[by_r[d][0]] = rr
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
    s1 = _finish_run_file(cfg, f_geo.name, cmds_g, res_g, t0, fail_fast)
    s2 = _finish_run_file(cfg, f_res.name, cmds_r, res_r, t0, fail_fast)
    return [s1, s2]


def _after_resample(cfg: PipelineConfig, date: str, overlap: bool) -> None:
    """Per-date housekeeping once a date's resample has verified outputs."""
    if getattr(cfg, "stack_cleanup_offsets", False):
        cleanup_offsets(cfg, date, overlap)
    if not overlap and cfg.stack_delete_safes_after_coreg:
        cleanup_safes(cfg)


_FUSED_PAIRS = ("generate_burst_igram", "merge_burst_igram", "filter_coherence")


def _fused_pair_stage(files: list, idx: int):
    """Indices of the three per-pair run files starting at files[idx], or None."""
    if _FUSED_PAIRS[0] not in files[idx].name:
        return None
    want = list(range(idx, idx + 3))
    if want[-1] >= len(files):
        return None
    if all(k in files[i].name for i, k in zip(want, _FUSED_PAIRS)):
        return want
    return None


def execute_fused_pairs(cfg: PipelineConfig, f_igram: Path, f_merge: Path, f_filt: Path) -> list:
    """
    Run burst interferogram -> merge -> filter as ONE per-pair chain.

    Stage-major order keeps every date's coregistered SLCs (MEASURED 0.86 GB
    per date) alive until the very last pair is filtered, because
    FilterAndCoherence estimates coherence through merged/SLC VRTs that
    point INTO them. Chained per pair, in chronological order, a date's
    last pair is filtered a few pairs after its first, and its SLCs (and
    each pair's burst products and unfiltered interferogram) can go at once
    - the disk high-water mark at production scale drops from ~365 GB to
    the ~240 GB floor set by coregistration itself. Same commands, same
    markers, same failure policy (per-pair failures are tolerated).
    """
    cmds = [plan_commands(cfg, f) for f in (f_igram, f_merge, f_filt)]
    by = [{pair_of(c): (i, c) for i, c in enumerate(cs)} for cs in cmds]
    keys = [set(b) for b in by]
    if None in keys[0] or not (keys[0] == keys[1] == keys[2]):
        log.warning("%s / %s / %s do not pair up - running as separate stages",
                    f_igram.name, f_merge.name, f_filt.name)
        return [execute_run_file(cfg, f, fail_fast=False) for f in (f_igram, f_merge, f_filt)]
    names = (f_igram.name, f_merge.name, f_filt.name)
    workers = _workers_for(cfg, f_igram.name)
    pairs = sorted(keys[0])
    log.info("%s -> %s -> %s: %d pairs as fused chains, %d workers x %d threads",
             *names, len(pairs), workers, _threads_for(cfg, workers))
    t0 = time.time()
    filtered: set = set()

    def chain(pair):
        out = []
        for stage, (b, name) in enumerate(zip(by, names)):
            i, c = b[pair]
            r = _run_one(cfg, name, i, c, workers)
            out.append(r)
            if r["returncode"] != 0:
                for b2, n2 in list(zip(by, names))[stage + 1:]:
                    out.append({"cmd": b2[pair][1], "returncode": -3, "seconds": 0.0,
                                "peak_rss_gb": None, "skipped": False,
                                "reason": f"{name} failed for this pair"})
                return out
        with _housekeeping_lock:
            filtered.add(pair)
            try:
                _after_filter(cfg, pair, filtered)
            except Exception as exc:                      # noqa: BLE001
                log.warning("post-filter housekeeping for %s failed (%s) - products "
                            "are intact; cleanup will retry at the end of the stage",
                            pair, exc)
        return out

    results = [[None] * len(cs) for cs in cmds]
    pool = ThreadPoolExecutor(max_workers=workers)
    try:
        futs = {pr: pool.submit(chain, pr) for pr in pairs}
        for pr, fut in futs.items():
            try:
                rs = fut.result()
            except Exception as exc:                          # noqa: BLE001
                log.exception("fused chain for %s raised", pr)
                rs = [{"cmd": b[pr][1], "returncode": -2, "seconds": 0.0, "peak_rss_gb": None,
                       "skipped": False, "reason": repr(exc)} for b in by]
            for stage, r in enumerate(rs):
                results[stage][by[stage][pr][0]] = r
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
    return [_finish_run_file(cfg, n, cs, rs, t0, fail_fast=False)
            for n, cs, rs in zip(names, cmds, results)]


def _after_filter(cfg: PipelineConfig, pair: str, filtered: set) -> None:
    """Per-pair housekeeping once a pair's filtered products are verified."""
    if cfg.stack_cleanup_burst_igrams:
        cleanup_bursts(cfg, pairs=[pair])
    if getattr(cfg, "stack_cleanup_unfiltered_igrams", False):
        cleanup_unfiltered(cfg, pairs=[pair])
    if getattr(cfg, "stack_cleanup_coreg_after_filter", False):
        cleanup_coreg_slcs(cfg, dates=pair.split("_"))


# ----------------------------------------------------------------------
# cleanups with markers
# ----------------------------------------------------------------------
# Every cleanup leaves a .cleaned marker where the product was, naming what
# was removed and why it was safe. The runner's outcome check accepts the
# marker in place of the product, so a resumed run does not re-run a stage
# whose outputs were consumed and then deliberately deleted - the trap the
# handoff describes for run_13.
def _mark_cleaned(marker: Path, what: str, evidence) -> None:
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps({"cleaned": what, "evidence": evidence,
                                  "when": time.strftime("%Y-%m-%dT%H:%M:%S")}, indent=1))


def _rm_files(paths) -> int:
    freed = 0
    for p in paths:
        try:
            freed += p.stat().st_size
            p.unlink()
        except FileNotFoundError:
            pass
    return freed


def _tree_size(path: Path) -> int:
    total = 0
    for f in path.rglob("*"):
        try:
            if f.is_file():
                total += f.stat().st_size
        except FileNotFoundError:
            pass
    return total


def cleanup_offsets(cfg: PipelineConfig, date: str, overlap: bool) -> float:
    """
    Delete a date's geo2rdr offsets once its resample has produced the SLCs.

    MEASURED: 862 MB of .off per date at full-burst, as much as the
    coregistered SLCs themselves; consumed only by that date's resample.
    """
    d = cfg.stack_dir / "coreg_secondarys" / date
    sub = d / "overlap" if overlap else d
    slcs = [p for p in sub.glob("IW*/*.slc") if p.stat().st_size > 0]
    if not slcs:
        return 0.0                                   # resample output not there
    offs = list(sub.glob("IW*/*.off")) + list(sub.glob("IW*/*.off.vrt")) + \
        list(sub.glob("IW*/*.off.xml"))
    freed = _rm_files(offs)
    if freed:
        _mark_cleaned(d / ("overlap_offsets.cleaned" if overlap else "offsets.cleaned"),
                      "geo2rdr offsets", {"slcs_present": len(slcs), "files": len(offs)})
        log.info("%s: removed %.2f GB of %s offsets (%d files) after resample", date,
                 freed / 1024 ** 3, "overlap" if overlap else "full-burst", len(offs))
    return freed / 1024 ** 3


def cleanup_misreg_inputs(cfg: PipelineConfig) -> float:
    """
    After run_08 has inverted the per-pair misregistration into per-date
    corrections, the ESD products, coarse interferograms and overlap SLCs
    that fed it are dead. MEASURED: 158 MB per pair (ESD 69 + coarse 89)
    plus 177 MB per date of overlap SLCs - 130 GB + 48 GB at production
    scale, sitting on disk for the whole run.
    """
    if not getattr(cfg, "stack_cleanup_misreg_inputs", False):
        return 0.0
    s = cfg.stack_dir
    dates_az = {p.stem for p in (s / "misreg" / "azimuth" / "dates").glob("*.txt")
                if p.stat().st_size > 0}
    dates_rg = {p.stem for p in (s / "misreg" / "range" / "dates").glob("*.txt")
                if p.stat().st_size > 0}
    coreg = s / "coreg_secondarys"
    dates = {d.name for d in coreg.iterdir() if d.is_dir()} if coreg.exists() else set()
    if not dates or not (dates <= dates_az and dates <= dates_rg):
        log.info("misreg inversion outputs not complete (%d/%d dates) - keeping the "
                 "ESD inputs", len(dates & dates_az & dates_rg), len(dates))
        return 0.0
    freed = 0
    for sub in ("ESD", "coarse_interferograms"):
        path = s / sub
        if path.exists():
            freed += sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
            shutil.rmtree(path, ignore_errors=True)
            _mark_cleaned(s / f"{sub}.cleaned", sub, {"dates_inverted": len(dates)})
    for d in sorted(dates):
        ov = coreg / d / "overlap"
        if ov.exists():
            freed += sum(f.stat().st_size for f in ov.rglob("*") if f.is_file())
            shutil.rmtree(ov, ignore_errors=True)
            _mark_cleaned(coreg / d / "overlap.cleaned", "overlap SLCs",
                          {"misreg_dates_txt": True})
    gb = freed / 1024 ** 3
    if gb:
        log.info("Reclaimed %.1f GB: ESD, coarse interferograms and overlap SLCs, "
                 "consumed by the misregistration inversion (run_08).", gb)
    return gb


def _planned_pairs_for_date(cfg: PipelineConfig, date: str) -> list:
    return sorted(f.name.split("config_igram_filt_coh_")[-1]
                  for f in (cfg.stack_dir / "configs").glob(f"config_igram_filt_coh_*{date}*"))


def cleanup_coreg_slcs(cfg: PipelineConfig, dates=None) -> float:
    """
    Delete a date's coregistered SLCs once every interferogram it takes part
    in has been filtered. MEASURED: 862 MB per date; 236 GB at 274 dates,
    the largest thing on disk for the whole second half of a run.

    Consumers: generateIgram (run_13) via the burst SLCs and
    FilterAndCoherence (run_15) via merged/SLC/<date>.slc.full.vrt, which is
    a VRT INTO these files - so the gate is the filtered products, not the
    merge. THE TRADE: adding dates or changing looks afterwards means
    re-coregistering (and re-downloading, since the SAFEs are gone too);
    invalidate_after_looks_change refuses when it sees the markers.
    """
    if not getattr(cfg, "stack_cleanup_coreg_after_filter", False):
        return 0.0
    coreg = cfg.stack_dir / "coreg_secondarys"
    merged = cfg.stack_dir / "merged" / "interferograms"
    if not coreg.exists():
        return 0.0
    freed = 0
    for d in sorted(x for x in coreg.iterdir() if x.is_dir()):
        if dates is not None and d.name not in dates:
            continue
        pairs = _planned_pairs_for_date(cfg, d.name)
        if not pairs:
            continue
        done = [p for p in pairs
                if all((merged / p / n).exists() and (merged / p / n).stat().st_size > 0
                       for n in ("filt_fine.int", "filt_fine.cor"))]
        if len(done) != len(pairs):
            continue
        freed += _tree_size(d)
        shutil.rmtree(d, ignore_errors=True)
        _mark_cleaned(coreg / f"{d.name}.cleaned", "coregistered SLCs",
                      {"filtered_pairs": done})
    gb = freed / 1024 ** 3
    if freed > 10 * 1024 ** 2:
        log.info("Reclaimed %.1f GB of coregistered SLCs whose every interferogram "
                 "is filtered. Adding dates now needs a fresh coregistration.", gb)
    return gb


def cleanup_unfiltered(cfg: PipelineConfig, pairs=None) -> float:
    """
    Delete the unfiltered merged interferogram and complex coherence once
    the filtered products exist. MEASURED: 50 MB of 111 MB per pair. MintPy
    loads filt_fine.unw / filt_fine.cor / .conncomp only.
    """
    if not getattr(cfg, "stack_cleanup_unfiltered_igrams", False):
        return 0.0
    merged = cfg.stack_dir / "merged" / "interferograms"
    if not merged.exists():
        return 0.0
    freed = 0
    for d in sorted(x for x in merged.iterdir() if x.is_dir()):
        if pairs is not None and d.name not in pairs:
            continue
        ok = all((d / n).exists() and (d / n).stat().st_size > 0
                 for n in ("filt_fine.int", "filt_fine.cor"))
        if not ok or (d / "fine.cleaned").exists():
            continue
        victims = [p for p in d.iterdir()
                   if p.name.startswith(("fine.int", "fine.cor"))]
        if not victims:
            continue
        freed += _rm_files(victims)
        _mark_cleaned(d / "fine.cleaned", "unfiltered fine.int/fine.cor",
                      {"filtered_present": True})
    gb = freed / 1024 ** 3
    if freed > 10 * 1024 ** 2:                 # sidecar xml/vrt alone is not news
        log.info("Reclaimed %.1f GB of unfiltered interferograms.", gb)
    return gb


def _record_timing(cfg: PipelineConfig, summary: dict) -> None:
    """
    Append to run_files/timing.json.

    This is what replaces my wall-clock estimates with measurements, and it is
    the only honest basis for sizing the production run.
    """
    p = cfg.stack_dir / "timing.json"
    hist = []
    if p.exists():
        try:
            hist = json.loads(p.read_text())
        except json.JSONDecodeError:
            pass
    hist = [h for h in hist if h.get("run_file") != summary["run_file"]]
    hist.append(summary)
    hist.sort(key=lambda h: h["run_file"])
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(hist, indent=2))


def run_files(cfg: PipelineConfig) -> list:
    """Generated run files in execution order (run_01_..., run_02_..., ...)."""
    d = cfg.stack_dir / "run_files"
    if not d.exists():
        return []
    # Exactly stackSentinel's naming: run_NN_name, no extension. Anything
    # else in the directory (backups, notes) is not a run file.
    return sorted(p for p in d.iterdir()
                  if p.is_file() and re.fullmatch(r"run_\d+_[A-Za-z0-9_]+", p.name))


def _merge_cleanup_hook(cfg: PipelineConfig, low: str):
    """On the phase-linked route's merge (real merged SLCs): a per-command
    hook dropping each date's coregistered bursts as soon as its merged copy
    is verified. None for every other run file."""
    if not ("merge_reference" in low and cfg.backend == "miaplpy"):
        return None
    from . import miaplpy_backend as _mb

    def after(cmd, res):
        d = _mb.merge_date_from_command(cmd)
        if d:
            gb = _mb.cleanup_coreg_date(cfg, d)
            if gb:
                log.info("merged %s verified; %.2f GB of its coregistered bursts removed", d, gb)
    return after


def _aliasing_hook(cfg: PipelineConfig, run_file: Path, when: str, pairs=None):
    """F2 / F3 hooks around the classic route's unwrap run file (8 Oct 2026):
    before, the pre-unwrap aliasing diagnostic on the filtered pairs (off
    unless the config asks; never raises); after, the fast zones (nothing
    unless fast_zone_geojson is or was set). Returns the run file's pairs."""
    from . import aliasing
    if pairs is None:
        pairs = sorted({p for p in (pair_of(c) for c in plan_commands(cfg, run_file)) if p})
    if when == "before":
        aliasing.pre_unwrap(cfg, pairs, route="stack")
    else:
        aliasing.apply_fast_zones(cfg, pairs)
    return pairs


def execute(cfg: PipelineConfig, from_step=None, to_step=None) -> list:
    """
    Execute the generated run files in order, with a barrier between each.

    from_step/to_step accept either an index (1-based, matching the run file
    numbering) or a substring of the name, so a human can say "redo unwrap"
    after changing defomax without re-reading the whole chain.
    """
    _install_teardown()
    files = run_files(cfg)
    if not files:
        raise FileNotFoundError(
            f"no run files in {cfg.stack_dir / 'run_files'} - "
            "call generate_run_files() first")
    set_aside_excluded(cfg)

    def _match(sel, files):
        if sel is None:
            return None
        if isinstance(sel, int) or str(sel).isdigit():
            return int(sel) - 1
        for i, f in enumerate(files):
            if str(sel) in f.name:
                return i
        raise ValueError(f"no run file matching {sel!r}")

    i0 = _match(from_step, files) or 0
    i1 = _match(to_step, files)
    i1 = len(files) - 1 if i1 is None else i1

    summaries = []
    idx = i0
    while idx <= i1:
        f = files[idx]
        low = f.name.lower()
        partner = _fused_partner(files, idx)
        if partner is not None and partner <= i1:
            for s in execute_fused(cfg, f, files[partner], fail_fast=True):
                _record_timing(cfg, s)
                summaries.append(s)
            idx = partner + 1
            continue
        trio = _fused_pair_stage(files, idx)
        if trio is not None and trio[-1] <= i1:
            for s in execute_fused_pairs(cfg, *[files[i] for i in trio]):
                _record_timing(cfg, s)
                summaries.append(s)
            # anything the per-pair hooks did not reach (e.g. a resumed run)
            cleanup_unfiltered(cfg)
            cleanup_coreg_slcs(cfg)
            idx = trio[-1] + 1
            continue
        if "unwrap" in low and getattr(cfg, "stack_crop_unwrap", False):
            # The window is normally computed right after the geometry merge;
            # a run resumed at the unwrap step still needs one.
            from .unwrap_crop import WINDOW_FILE, compute_crop_window
            if not (cfg.stack_dir / WINDOW_FILE).exists():
                compute_crop_window(cfg)
        # Everything up to and including the common valid region is the
        # coregistration chain, where one missing product poisons the rest.
        # (the merge counts as coregistration too: MiaplPy reads every merged
        # SLC, so a date that failed to merge stops the run, never a gap)
        fail_fast = not _passed_coreg(f.name)
        unwrap_pairs = _aliasing_hook(cfg, f, "before") if ("unwrap" in low and "ion" not in low) else None
        s = execute_run_file(cfg, f, fail_fast=fail_fast, after=_merge_cleanup_hook(cfg, low))
        _record_timing(cfg, s)
        summaries.append(s)
        if unwrap_pairs is not None:
            _aliasing_hook(cfg, f, "after", unwrap_pairs)
        if "unpack_topo_reference" in low:
            # DEM and AOI-in-frame check on the per-burst geometry BEFORE any
            # secondary is touched: found after the merge it used to cost a
            # full re-download (the SAFEs and the geometry were gone by then).
            _check_burst_dem(cfg)
        # Coregistration is done once the stack's valid region has been
        # extracted; from that point the SAFEs and the per-burst geometry are
        # only needed to ADD dates, not to process the ones already here.
        # The per-burst geometry goes only after the merged geometry has
        # passed its own check.
        if "extract_stack_valid_region" in low:
            cleanup_safes(cfg)
        if "merge_reference" in low:
            _check_merged_dem(cfg)
            cleanup_safes(cfg)
            cleanup_burst_geometry(cfg)
            if getattr(cfg, "stack_crop_unwrap", False):
                from .unwrap_crop import compute_crop_window
                compute_crop_window(cfg)
        if "timeseries_misreg" in low:
            cleanup_misreg_inputs(cfg)
        if cfg.stack_cleanup_burst_igrams and "merge_burst_igram" in low:
            cleanup_bursts(cfg)
        if "filter_coherence" in low:
            cleanup_unfiltered(cfg)
            cleanup_coreg_slcs(cfg)
        idx += 1
    return summaries


def _check_merged_dem(cfg: PipelineConfig) -> float:
    """
    Fraction of the merged frame with a DEM height, checked as soon as the
    geometry is merged rather than discovered in the time series.

    A run once completed with 86% of the frame at height -500: the
    DEM had been stitched to the AOI bbox, topo filled the rest with nodata,
    and nothing between run_01 and MintPy noticed. Coverage inside the AOI
    is what matters for the answer; coverage of the frame is what matters
    for the unwrapper, which solves the whole thing.
    """
    from .unwrap_crop import _read_gdal
    geom = cfg.stack_dir / "merged" / "geom_reference"
    try:
        hgt = _read_gdal(geom / "hgt.rdr")
        lat = _read_gdal(geom / "lat.rdr")
        lon = _read_gdal(geom / "lon.rdr")
    except FileNotFoundError:
        log.warning("merged geometry not present after the merge step - cannot "
                    "check DEM coverage")
        return float("nan")
    return _judge_geometry(cfg, [(lat, lon, hgt)], "Merged")


# AOI cells the coverage test works in: ~0.005 deg is 440-555 m at these
# latitudes, against merged pixels of ~25 x 42 m (11 x 3 looks) and decimated
# burst samples of ~46 x 70 m, so a cell inside the frame always holds samples.
AOI_CELL_DEG = 0.005
MIN_AOI_IN_FRAME = 0.99


def _aoi_cells(bbox, cell_deg: float = AOI_CELL_DEG):
    import numpy as np
    w, s, e, n = bbox
    nx = max(1, int(np.ceil((e - w) / cell_deg)))
    ny = max(1, int(np.ceil((n - s) / cell_deg)))
    return nx, ny


def _judge_geometry(cfg: PipelineConfig, chunks, where: str) -> float:
    """
    DEM and AOI checks on radar geometry, given as (lat, lon, hgt) arrays
    (the merged grid, or each burst of stack/geom_reference, decimated).

    Two separate questions. (1) Does the AOI ground that IS in the frame have
    a height? -500 there means topography was never removed. (2) Is the AOI
    in the frame at all? The old check counted only pixels that exist, so an
    AOI whose edge lies in a sub-swath that is not processed passed, and its
    report areas were silently part-measured. (2) bins frame samples into
    ~500 m AOI cells and needs MIN_AOI_IN_FRAME of them hit.
    """
    import numpy as np
    from .unwrap_crop import NODATA_HEIGHT
    w, s, e, n = cfg.bbox
    n_all = n_valid = aoi_n = aoi_missing = 0
    shape = None
    step = 0.0
    pts_lat, pts_lon = [], []
    for lat, lon, hgt in chunks:
        shape = shape or hgt.shape
        valid = hgt > NODATA_HEIGHT
        n_all += valid.size
        n_valid += int(valid.sum())
        aoi = (lat >= s) & (lat <= n) & (lon >= w) & (lon <= e)
        aoi_n += int(aoi.sum())
        aoi_missing += int((aoi & ~valid).sum())
        pts_lat.append(lat[aoi])
        pts_lon.append(lon[aoi])
        # sample spacing, so a coarse grid cannot leave cells empty inside the frame
        for a in (lat, lon):
            for ax in (0, 1):
                if a.shape[ax] > 1:
                    step = max(step, float(np.nanmedian(np.abs(np.diff(a, axis=ax)))))
    if not n_all:
        return float("nan")
    cell = max(AOI_CELL_DEG, 2.5 * step)
    nx, ny = _aoi_cells(cfg.bbox, cell)
    hit = np.zeros((ny, nx), bool)
    la, lo = np.concatenate(pts_lat), np.concatenate(pts_lon)
    ix = np.clip(((lo - w) / (e - w) * nx).astype(int), 0, nx - 1)
    iy = np.clip(((la - s) / (n - s) * ny).astype(int), 0, ny - 1)
    hit[iy, ix] = True
    frac = n_valid / n_all
    in_frame = float(hit.mean())
    log.info("%s DEM covers %.1f%% of the frame%s; %d AOI pixels lack height; "
             "%.1f%% of the AOI lies inside the processed frame", where, 100 * frac,
             f" ({shape[1]}x{shape[0]})" if shape and where == "Merged" else "",
             aoi_missing, 100 * in_frame)
    if aoi_missing > 0.001 * max(1, aoi_n):
        raise RuntimeError(
            f"{aoi_missing} pixels INSIDE the AOI have no DEM height (-500). The "
            "DEM does not cover the area of interest; nothing downstream can be "
            "trusted. Delete dem/ and re-stage it (get_dem now stitches the "
            "frame footprint).")
    if in_frame < MIN_AOI_IN_FRAME:
        raise RuntimeError(
            f"Only {100 * in_frame:.0f}% of the area of interest lies inside the radar "
            f"frame being processed (sub-swath(s) {cfg.stack_swaths}). The rest is in a "
            "sub-swath that is not processed, so any report area there would be "
            "partly unmeasured without saying so. Add the neighbouring sub-swath to "
            "stack_swaths (for example \"2 3\") or move the area of interest.")
    if frac < 0.9:
        log.warning("Only %.0f%% of the frame has a DEM. Topographic phase is "
                    "not removed elsewhere and those pixels unwrap as noise. "
                    "Re-stage the DEM over the frame footprint before "
                    "production; stack_crop_unwrap limits the damage to speed.",
                    100 * frac)
    return frac


def _read_gdal_decimated(path: Path, step_x: int, step_y: int):
    """Every step-th sample of an ISCE raster, read by GDAL without loading
    the full-resolution burst (a burst's lat/lon are ~300 MB each)."""
    from osgeo import gdal
    gdal.UseExceptions()
    for cand in (str(path) + ".vrt", str(path)):
        if os.path.exists(cand):
            ds = gdal.Open(cand)
            if ds is not None:
                bx = max(1, ds.RasterXSize // step_x)
                by = max(1, ds.RasterYSize // step_y)
                return ds.GetRasterBand(1).ReadAsArray(buf_xsize=bx, buf_ysize=by)
    raise FileNotFoundError(path)


def _check_burst_dem(cfg: PipelineConfig, step_x: int = 20, step_y: int = 5) -> float:
    """
    The DEM/AOI check on the per-burst geometry, right after run_01 (topo).

    The merged check runs after run_12, by which time SAFE cleanup and
    cleanup_burst_geometry have removed what redoing topo and
    coregistration needs - a failure there meant a full re-download.
    Here it costs the topo run only. Decimated 20 x 5 (~46 m x 70 m).
    """
    geom = cfg.stack_dir / "geom_reference"
    hgts = sorted(geom.glob("IW*/hgt_*.rdr"))
    if not hgts:
        log.info("no per-burst geometry under %s - DEM check deferred to the merge", geom)
        return float("nan")

    def chunks():
        for h in hgts:
            k = h.name[len("hgt"):]                  # "_01.rdr"
            try:
                yield (_read_gdal_decimated(h.with_name("lat" + k), step_x, step_y),
                       _read_gdal_decimated(h.with_name("lon" + k), step_x, step_y),
                       _read_gdal_decimated(h, step_x, step_y))
            except FileNotFoundError:
                log.warning("burst geometry %s incomplete - skipped in the DEM check", h.parent / k)
    return _judge_geometry(cfg, chunks(), "Burst")


def _passed_coreg(name: str) -> bool:
    """
    True once we are past the coregistration chain.

    Run files are numbered, and everything up to and including
    extract_stack_valid_region builds the coregistered stack. After that each
    command produces one independent interferogram, so one failure is a
    dropped pair rather than a corrupt stack.
    """
    m = re.match(r"run_(\d+)_(.+)", name)
    if not m:
        return False
    name = m.group(2)
    # ESD misregistration is estimated PER PAIR and then inverted into a
    # per-date correction, so the pair network is redundant: losing a few
    # long-baseline pairs whose burst overlaps have decorrelated still leaves
    # every date determined. Treating this as fail-fast stopped a 31-date
    # stack over 5 pairs of 87. The tolerance is bounded in execute_run_file.
    if "misreg" in name:
        return True
    return name in (
        "generate_burst_igram", "merge_burst_igram", "filter_coherence",
        "unwrap", "generateIgram", "mergeBurstsIon",
    ) or "igram" in name or "unwrap" in name


def reference_dates(ref_dir: Path) -> set[str]:
    """The acquisition date(s) the stack's reference/ was built from, read
    from the SLC SAFE names its xml and VRT files point at. Only SAFE names
    count: the orbit file named in the same xml carries its own production
    date (S1A_OPER_AUX_POEORB_OPOD_20210310T052706_V2016...), which a bare
    date pattern picked up as the reference on Site B (28 Sep 2026)."""
    import re as _re
    pat = _re.compile(r"S1[A-D]_IW_SLC__\w{4}_(\d{8})T\d{6}_")
    hits: set[str] = set()
    for x in list(ref_dir.rglob("*.xml")) + list(ref_dir.rglob("*.vrt")):
        hits |= set(pat.findall(x.read_text(errors="ignore")))
    return hits


# ----------------------------------------------------------------------
# which dates the stack holds - ONE predicate for download skip, SAFE
# cleanup and the plan check
# ----------------------------------------------------------------------
_DATE_RE = re.compile(r"^\d{8}$")
SAFE_DATE_RE = re.compile(r"_(\d{8})T\d{6}_")


def _reference_date_set(cfg: PipelineConfig) -> set[str]:
    """The stack reference date, once run_01 has unpacked reference/."""
    ref_dir = cfg.stack_dir / "reference"
    if not any(ref_dir.glob("IW[1-3].xml")):
        return set()
    if cfg.stack_reference_date:
        return {cfg.stack_reference_date}
    return reference_dates(ref_dir)


def _cleaned_date_verified(cfg: PipelineConfig, marker: Path) -> bool:
    """
    A coreg_secondarys/<date>.cleaned marker counts only when what replaced
    the bursts is accounted for: the stack route's marker
    (cleanup_coreg_slcs) is written after every pair of the date is
    filtered; the phase-linked route's (miaplpy_backend.cleanup_coreg_date)
    after the real merged SLC is verified - which must still verify now, or
    have been consumed into slcStack.h5 (DELETED_AFTER_MIAPLPY_LOAD).
    A bare marker whose merged SLC was removed by hand does not count, so
    the date's archive is fetched again rather than silently skipped.
    """
    try:
        rec = json.loads(marker.read_text())
        if isinstance(rec, dict) and rec.get("cleaned") == "coregistered SLCs":
            return True
    except (OSError, ValueError):
        pass
    d = marker.stem
    mdir = cfg.stack_dir / "merged" / "SLC" / d
    if (mdir / "DELETED_AFTER_MIAPLPY_LOAD").exists():
        return True
    full = mdir / f"{d}.slc.full"
    return full.exists() and Path(str(full) + ".xml").exists() and not isce_size_problem(full)


def coregistered_dates(cfg: PipelineConfig, include_reference: bool = True) -> set[str]:
    """
    Dates whose coregistration is finished and accounted for:
      * coreg_secondarys/<date>/ whose full-burst resample is complete
        (resample_incomplete: the swath xml written last, every burst SLC);
      * coreg_secondarys/<date>.cleaned whose replacement is verified
        (_cleaned_date_verified);
      * the reference date once reference/ is unpacked (include_reference).

    The single predicate for the download skip (data_acquisition) and SAFE
    cleanup. The two used to disagree: the download counted directories
    only, so on a phase-linked project - where the merge replaces every
    date directory with <date>.cleaned - a 'Whole pipeline' restart queued
    ~263 dates (~110 GB, ~3 h) again (Site B: 269 markers, 0 directories,
    review 5 Oct 2026, ops-2 / stackrunner-4).
    """
    coreg = cfg.stack_dir / "coreg_secondarys"
    swaths = _stack_swaths(cfg)
    done: set[str] = set()
    if coreg.exists():
        for p in coreg.iterdir():
            if p.is_dir() and _DATE_RE.match(p.name):
                if not resample_incomplete(p, swaths):
                    done.add(p.name)
            elif p.suffix == ".cleaned" and _DATE_RE.match(p.stem) and p.is_file():
                if _cleaned_date_verified(cfg, p):
                    done.add(p.stem)
    if include_reference:
        done |= _reference_date_set(cfg)
    return done


def coreg_started(cfg: PipelineConfig) -> bool:
    """True once any date has entered coreg_secondarys/ (directory or marker).
    From then on stackSentinel will not plan a fresh stack: it globs date
    directories only (stackSentinel.py:915) and, after the phase-linked
    merge has replaced them with markers, would plan a NEW stack from the
    handful of SAFEs left on disk."""
    coreg = cfg.stack_dir / "coreg_secondarys"
    if not coreg.exists():
        return False
    return any(_DATE_RE.match(p.name) or (p.suffix == ".cleaned" and _DATE_RE.match(p.stem))
               for p in coreg.iterdir())


def _cleaned_coreg_markers(cfg: PipelineConfig) -> list:
    coreg = cfg.stack_dir / "coreg_secondarys"
    return sorted(p for p in coreg.glob("*.cleaned") if _DATE_RE.match(p.stem)) \
        if coreg.exists() else []


def _window(cfg: PipelineConfig):
    """stackSentinel's date window as YYYYMMDD: an acquisition is used when
    start <= date < stop (it compares the sensing time with MIDNIGHT of
    --stop_date, stackSentinel.py:301, so the stop date itself is out)."""
    s = str(cfg.start_date or "").replace("-", "") or "00000000"
    e = str(cfg.end_date or "").replace("-", "") or "99999999"
    return s, e


def available_input_dates(cfg: PipelineConfig) -> set[str]:
    """Acquisition dates of the SAFEs/zips in slc/ that stackSentinel would
    use (inside the date window). Partial assemblies live in hidden
    subdirectories and quarantine/, which stackSentinel never globs."""
    s, e = _window(cfg)
    out = set()
    if cfg.slc_dir.exists():
        for p in list(cfg.slc_dir.glob("S1*_IW_SLC*.SAFE")) + list(cfg.slc_dir.glob("S1*_IW_SLC*.zip")):
            m = SAFE_DATE_RE.search(p.name)
            if m and s <= m.group(1) < e:
                out.add(m.group(1))
    return out


def _generation_record(cfg: PipelineConfig) -> dict:
    try:
        rec = json.loads((cfg.stack_dir / "generation.json").read_text())
        return rec if isinstance(rec, dict) else {}
    except (OSError, ValueError):
        return {}


def _safe_files_dates(cfg: PipelineConfig) -> set[str] | None:
    """Dates stackSentinel listed in stack/SAFE_files.txt (it writes the SAFEs
    it used there) - the plan's input for records made before input_dates."""
    p = cfg.stack_dir / "SAFE_files.txt"
    try:
        txt = p.read_text()
    except OSError:
        return None
    return {m.group(1) for line in txt.splitlines()
            for m in [SAFE_DATE_RE.search(Path(line.strip()).name)] if m}


def planned_dates(cfg: PipelineConfig, rec: dict | None = None) -> set[str] | None:
    """
    The acquisition dates the current plan (run_files/) covers: generation.json
    'input_dates' (written since 5 Oct 2026), else stack/SAFE_files.txt, plus
    every date already coregistered, less the dates left out of the stack
    (stack_exclude_dates). None when no plan exists or nothing records what
    it was made from - unknown, never assumed complete.
    """
    if not (cfg.stack_dir / "run_files").exists():
        return None
    rec = _generation_record(cfg) if rec is None else rec
    if rec.get("input_dates"):
        dates = set(rec["input_dates"])
    else:
        dates = _safe_files_dates(cfg)
        if not dates:
            return None
    from .data_acquisition import excluded_dates
    return (dates | coregistered_dates(cfg)) - excluded_dates(cfg)


def dates_missing_from_plan(cfg: PipelineConfig, rec: dict | None = None) -> list:
    """
    Dates on disk (slc/) that the existing plan does not cover and that
    nobody accepted as missing (allow_missing_dates). One-directional on
    purpose: SAFE cleanup deletes the archives of planned dates, so planned
    but not on disk is the normal state of a resumed run.
    """
    plan = planned_dates(cfg, rec)
    if plan is None:
        return []
    from .data_acquisition import accepted_missing_dates
    accepted, accept_all = accepted_missing_dates(cfg)
    if accept_all:
        return []
    return sorted(available_input_dates(cfg) - plan - accepted)


def cleanup_safes(cfg: PipelineConfig) -> float:
    """
    Delete SAFE zips whose date is already coregistered.

    Measured on the P0 run: 32 GB of SAFEs against 12 GB for everything the
    processing actually produced. Nothing downstream of coregistration opens
    them - the burst SLCs live in coreg_secondarys/ and reference/ - so after
    run_02 they are the largest dead weight on disk by a wide margin.

    A SAFE is removed only when that date's coregistered output exists, so an
    interrupted run never loses input it still needs. The most recent
    stack_keep_recent_safes dates are kept regardless, because stackSentinel's
    incremental path re-reads them to extend a stack.
    """
    import re as _re
    if not cfg.stack_delete_safes_after_coreg:
        return 0.0
    ref_dir = cfg.stack_dir / "reference"
    # "Coregistered" means the full-burst resample is COMPLETE (every burst
    # and the swath xml written last), or its products were cleaned after
    # verified use. The overlap stage (run_05/06) creates the same directory
    # first, and resample reads the secondary through VRTs into the SAFE
    # burst by burst - a directory test, or "one burst exists", deletes
    # archives still being read (with more topo workers than
    # stack_keep_recent_safes, or keep = 0, which config documents).
    # Same predicate as the download skip: coregistered_dates.
    done = coregistered_dates(cfg, include_reference=False)
    if not done:
        return 0.0

    # THE REFERENCE DATE'S ARCHIVE IS NEVER DELETABLE. Its burst SLCs under
    # reference/IW*/ are VRTs whose relative paths point back INTO the SAFE -
    # they are references, not extracted copies. Deleting it leaves every
    # interferogram that involves the reference date reading a missing tiff,
    # which on this stack was 6 of 87 pairs and cost a full interferogram pass.
    # (An earlier version of this function added the reference to the deletable
    # set on the reasoning that it had been "unpacked". It had not.)
    protected = {cfg.stack_reference_date} if cfg.stack_reference_date else set()
    if not protected and ref_dir.exists():
        # No explicit reference: infer it from what reference/ points at, and
        # refuse to delete anything rather than guess wrong.
        hits = reference_dates(ref_dir)
        if hits:
            protected = hits
        else:
            log.warning("Cannot determine the reference date, whose archive must "
                        "not be deleted - skipping SAFE cleanup entirely.")
            return 0.0

    keep = set(sorted(done, reverse=True)[:max(0, cfg.stack_keep_recent_safes)])
    keep |= protected
    freed = 0
    removed = []
    candidates = sorted(cfg.slc_dir.glob("S1*.zip")) + sorted(cfg.slc_dir.glob("S1*.SAFE"))
    for safe in candidates:
        m = _re.search(r"_(\d{8})T\d{6}_", safe.name)
        if not m:
            continue
        date = m.group(1)
        if date in done and date not in keep:
            try:
                if safe.is_dir():
                    freed += sum(f.stat().st_size for f in safe.rglob("*") if f.is_file())
                    shutil.rmtree(safe, ignore_errors=True)
                else:
                    freed += safe.stat().st_size
                    safe.unlink()
            except FileNotFoundError:
                continue                       # already gone - nothing to reclaim
            removed.append(safe.name)
    gb = freed / 1024 ** 3
    if removed:
        log.info("Protected the reference date(s) %s from cleanup.",
                 ", ".join(sorted(protected)))
        log.info("Reclaimed %.1f GB: deleted %d SAFE zip(s) whose dates are "
                 "coregistered (keeping the %d most recent for incremental "
                 "extension).", gb, len(removed), len(keep))
    return gb


def cleanup_burst_geometry(cfg: PipelineConfig) -> float:
    """
    Drop the per-burst reference geometry once every date is coregistered.

    stack/geom_reference holds full-resolution lat/lon/hgt/los per burst and
    is about 5.8 GB regardless of how many dates the stack has. It is consumed
    by geo2rdr and resample, so it is only needed to coregister NEW dates.
    merged/geom_reference - the one MintPy loads - is a separate, far smaller
    product and is never touched here.
    """
    if not cfg.stack_cleanup_burst_geometry:
        return 0.0
    geom = cfg.stack_dir / "geom_reference"
    merged_geom = cfg.stack_dir / "merged" / "geom_reference"
    if not geom.exists():
        return 0.0
    if not merged_geom.exists() or not any(merged_geom.iterdir()):
        log.info("merged/geom_reference is not populated yet - keeping the "
                 "per-burst geometry, which is what produces it.")
        return 0.0
    freed = sum(f.stat().st_size for f in geom.rglob("*") if f.is_file())
    shutil.rmtree(geom, ignore_errors=True)
    # The marker is what run_01's outcome check accepts in place of the
    # geometry, and what tells _generate_run_files that new dates cannot be
    # coregistered in place (geo2rdr reads this geometry).
    _mark_cleaned(cfg.stack_dir / "geom_reference.cleaned", "per-burst reference geometry",
                  {"merged_geom_reference": str(merged_geom), "bytes": freed})
    gb = freed / 1024 ** 3
    log.info("Reclaimed %.1f GB: removed the per-burst reference geometry. "
             "Adding dates later will need it rebuilt (run_01).", gb)
    return gb


def cleanup_bursts(cfg: PipelineConfig, pairs=None) -> float:
    """
    Delete per-burst interferogram directories once the merge has produced the
    full-swath product.

    These are the largest transient on disk - roughly 790 MB per pair against
    220 MB for the merged product they feed. topsStack keeps burst-level output
    in <stack>/interferograms/<pair>/IW*/ and the merged result in
    <stack>/merged/interferograms/<pair>/, so the merged file is checked before
    anything is removed: deleting bursts whose merge never ran would force the
    pair to be reprocessed from coregistration.
    """
    freed = 0
    base = cfg.stack_dir / "interferograms"
    merged_base = cfg.stack_dir / "merged" / "interferograms"
    if not base.exists():
        return 0.0
    for pair in sorted(p for p in base.iterdir() if p.is_dir()):
        if pairs is not None and pair.name not in pairs:
            continue
        merged = merged_base / pair.name
        # Existence is not enough: a zero-length merge counts as "merged" and
        # the 790 MB of inputs it needs are then gone for good.
        products = [merged / "filt_fine.int", merged / "fine.int"]
        if not any(p.exists() and p.stat().st_size > 0 for p in products):
            continue                      # merge has not run; the bursts are live
        for sub in pair.glob("IW*"):
            if sub.is_dir():
                freed += sum(f.stat().st_size for f in sub.rglob("*") if f.is_file())
                shutil.rmtree(sub, ignore_errors=True)
    gb = freed / 1024 ** 3
    if gb > 0.1:
        log.info("Cleaned %.1f GB of per-burst interferograms", gb)
    return gb


# ----------------------------------------------------------------------
# generation
# ----------------------------------------------------------------------
def _bbox_snwe(cfg: PipelineConfig) -> str:
    """cfg.bbox is (W, S, E, N); stackSentinel -b wants 'S N W E'."""
    w, s, e, n = cfg.bbox
    m = cfg.stack_bbox_margin_deg
    return f"{s - m} {n + m} {w - m} {e + m}"


def esd_possible(cfg: PipelineConfig) -> tuple:
    """(possible, why) for ESD (NESD) coregistration. ESD measures the
    azimuth misregistration in the strips where neighbouring bursts overlap;
    a crop holding one burst per sub-swath has no overlaps, and every
    overlap_geo2rdr command then writes nothing (21 Oct 2026: a 10 x 8 km
    experiment crop failed run_05 on all 49 dates). Read from the download's
    own burst record only (no catalogue query here); without one, ESD is
    assumed possible, as before."""
    n_sw = max(1, len(str(getattr(cfg, "stack_swaths", "2") or "2").split()))
    for name in ("assembly_status.json", "dropped_short_dates.json"):
        try:
            n = int(json.loads((cfg.slc_dir / name).read_text()).get("modal_burst_count") or 0)
        except (OSError, ValueError, TypeError, AttributeError):
            continue
        if n > 0:
            if n <= n_sw:
                return False, (f"{n} burst(s) per date over {n_sw} sub-swath(s) (slc/{name}): no burst "
                               "overlaps to measure")
            return True, f"{n} bursts per date (slc/{name})"
    return True, "burst count not recorded"


def coregistration_mode(cfg: PipelineConfig) -> str:
    """'NESD' or 'geometry': do_esd, unless the crop has no burst overlaps
    (esd_possible), when geometry coregistration is the only one that can run.
    With a single burst there are also no burst seams for ESD to correct."""
    if not getattr(cfg, "do_esd", True):
        return "geometry"
    ok, why = esd_possible(cfg)
    if not ok:
        log.warning("do_esd is on but %s; coregistering by geometry instead (no burst seams to correct "
                    "with one burst per sub-swath).", why)
        return "geometry"
    return "NESD"


def coregistration_used(cfg: PipelineConfig) -> str:
    """The coregistration the stack was actually generated with (stack/
    generation.json's argv), else what this config would use now."""
    try:
        argv = json.loads((cfg.stack_dir / "generation.json").read_text()).get("argv") or []
        if "-C" in argv:
            return str(argv[argv.index("-C") + 1])
    except (OSError, ValueError, AttributeError, IndexError):
        pass
    return coregistration_mode(cfg)


def _stack_argv(cfg: PipelineConfig, dem_path: Path) -> list:
    """
    Build the stackSentinel.py command line.

    --num_proc is pinned to 1 on purpose. At higher values the generated run
    files use shell backgrounding and `wait`, which destroys per-command
    resumability and failure isolation. Our own pool (stack_num_proc*) provides
    the parallelism instead.
    """
    aux = cfg.stack_dir / "aux"
    aux.mkdir(parents=True, exist_ok=True)   # required flag; only IPF 002.36
                                             # (2015 data) actually reads it
    argv = [
        "stackSentinel.py",
        "-s", str(cfg.slc_dir),
        "-o", str(cfg.orbit_dir),
        "-a", str(aux),
        "-w", str(cfg.stack_dir),
        "-d", str(dem_path),
        "-p", cfg.stack_polarization,
        "-W", "interferogram",
        "-n", cfg.stack_swaths,
        "-c", str(cfg.stack_num_connections),
        "-O", str(cfg.stack_num_overlap_connections),
        "-z", str(cfg.stack_azimuth_looks),
        "-r", str(cfg.stack_range_looks),
        "-f", str(cfg.stack_filter_strength),
        "-u", cfg.stack_unw_method,
        "-C", coregistration_mode(cfg),
        "-e", str(cfg.esd_coherence_threshold),
        "-V", cfg.stack_virtual_merge,
        "--num_proc", "1",
    ]
    if cfg.stack_bbox_margin_deg >= 0 and cfg.bbox:
        argv += ["-b", _bbox_snwe(cfg)]
    if cfg.stack_reference_date:
        argv += ["-m", cfg.stack_reference_date]
    if cfg.start_date:
        argv += ["--start_date", cfg.start_date]
    if cfg.end_date:
        argv += ["--stop_date", cfg.end_date]
    from .data_acquisition import excluded_dates
    if excluded_dates(cfg):
        argv += ["-x", ",".join(sorted(excluded_dates(cfg)))]
    # exclude_pairs is NOT passed here. stackSentinel's -x takes DATES
    # ("20141007,20141031") and drops those acquisitions from the whole stack;
    # a pair name matches no date, and changing it would also alter the argv
    # and force a regeneration. Pair exclusion belongs to MintPy, which gets it
    # as mintpy.network.excludeDate12 via time_series._screened_out_pairs.
    if cfg.stack_use_gpu:
        argv += ["-useGPU"]
    if cfg.stack_ionosphere:
        argv += ["--param_ion", str(ISCE_STACK / "ion_param.txt"),
                 "--num_connections_ion", str(cfg.stack_ion_num_connections)]
    return argv


def _restore_superseded(cfg: PipelineConfig):
    """
    Put back the most recently archived plan.

    Generation has to move run_files/ aside before stackSentinel will run, so
    a generation that then turns out to be a no-op would otherwise leave the
    project with no plan at all.
    """
    archives = sorted(cfg.stack_dir.glob("run_files.superseded_*"))
    if not archives:
        return None
    newest = archives[-1]
    stamp = newest.name.split("superseded_")[-1]
    for d in ("run_files", "configs"):
        src = cfg.stack_dir / f"{d}.superseded_{stamp}"
        dst = cfg.stack_dir / d
        if src.exists() and not dst.exists():
            src.rename(dst)
    return newest


def _apply_unwrap_mode(cfg: PipelineConfig) -> None:
    """Make run_16 match stack_crop_unwrap, whichever way the plan arrived."""
    from .unwrap_crop import restore_unwrap_run_file, rewrite_unwrap_run_file
    if getattr(cfg, "stack_crop_unwrap", False):
        rewrite_unwrap_run_file(cfg)
    else:
        n = restore_unwrap_run_file(cfg)
        if n:
            log.info("stack_crop_unwrap is off - restored %d unwrap run file(s) "
                     "to SentinelWrapper", n)


def _extension_problem(cfg: PipelineConfig, new: list, rec: dict) -> str:
    """
    Why acquisitions missing from the plan cannot be added to a stack whose
    coregistration has started, or "" when stackSentinel's update mode can.

    Update mode pairs the new dates with the LATEST coregistered ones only
    (stackSentinel.py:952) and finds the existing stack by globbing date
    DIRECTORIES (line 915). So it works only for dates after the end of the
    plan, on an intact stack, with the latest SAFEs still on disk.
    """
    plan = planned_dates(cfg, rec) or set()
    if _cleaned_coreg_markers(cfg):
        return ("the coregistered images of the planned dates were removed after use "
                "(the phase-linked merge, or stack_cleanup_coreg_after_filter), so "
                "nothing is left to add new dates to")
    if (cfg.stack_dir / "geom_reference.cleaned").exists() or (
            not (cfg.stack_dir / "geom_reference").exists()
            and (cfg.stack_dir / "merged" / "geom_reference").exists()):
        return "the per-burst reference geometry that coregistering new dates needs was removed"
    if cfg.backend == "miaplpy":
        return "phase linking would have to be redone over the whole record"
    if plan and min(new) <= max(plan):
        return (f"{min(new)} falls inside the planned record (which ends {max(plan)}); "
                "stackSentinel can only append dates after the last one")
    k = (2 * int(cfg.stack_num_overlap_connections) if coregistration_used(cfg) == "NESD"
         else int(cfg.stack_num_connections))
    latest = sorted(coregistered_dates(cfg, include_reference=False))[-k:]
    gone = sorted(set(latest) - available_input_dates(cfg))
    if gone:
        return (f"extending needs the archives of the latest {k} coregistered dates and "
                f"{len(gone)} have been cleaned up ({', '.join(gone[:6])})")
    return ""


def _refuse_missing(cfg: PipelineConfig, missing: list, why: str) -> None:
    """Stop before any compute, naming the dates and the ways out."""
    rec = {"dates_not_in_plan": missing, "reason": why,
           "written": time.strftime("%Y-%m-%dT%H:%M:%S")}
    try:
        (cfg.stack_dir / "dates_not_in_plan.json").write_text(json.dumps(rec, indent=1))
    except OSError:
        pass
    try:
        from . import progress
        progress.emit(cfg.project_dir, "stage_inputs", "failed",
                      msg=f"{len(missing)} downloaded date(s) are not in the stack plan")
    except Exception:                                            # noqa: BLE001
        pass
    raise RuntimeError(
        f"{len(missing)} acquisition(s) are downloaded but not in this project's "
        f"processing plan: {', '.join(missing[:12])}{'...' if len(missing) > 12 else ''}. "
        "The plan was made before they arrived, and coregistration has already "
        f"started, so they cannot be added in place: {why}. Reusing the plan would "
        "leave them out of every product without saying so. Either list them under "
        "allow_missing_dates in the config to carry on without them (they are then "
        "recorded as deliberately excluded), or start a new project to process the "
        "full record. Details: stack/dates_not_in_plan.json.")


def _argv_changes(previous: list, argv: list) -> list:
    """'-f 0.4 -> 0.5' style differences between two stackSentinel argvs."""
    def opts(a):
        d, key = {}, None
        for t in a[1:]:
            if t.startswith("-") and not re.fullmatch(r"-?\d+(\.\d+)?", t):
                key = t
                d[key] = ""
            elif key is not None:
                d[key] = (d[key] + " " + t).strip()
        return d
    p, n = opts(previous or []), opts(argv)
    return [f"{k} {p.get(k, '(absent)')} -> {n.get(k, '(absent)')}"
            for k in sorted(set(p) | set(n)) if p.get(k) != n.get(k)]


def _without_exclude(argv: list) -> list:
    """A stackSentinel argv without its -x DATES."""
    a = list(argv or [])
    if "-x" in a:
        i = a.index("-x")
        del a[i:i + 2]
    return a


def _refuse_replan(cfg: PipelineConfig, previous: list, argv: list) -> None:
    """
    Changed arguments on a stack whose coregistered images were removed after
    use. stackSentinel finds no date directories (it does not see the
    <date>.cleaned markers), would plan a NEW stack from the few SAFEs left
    in slc/ (Site B: 7 of 270) - or fail on 'original SAFE files ... is
    needed' after run_files/ had been archived. Refuse before touching
    anything.
    """
    changes = _argv_changes(previous, argv)
    raise RuntimeError(
        "This project's coregistered images were removed after use (coreg_secondarys "
        "holds only <date>.cleaned markers), so its processing plan cannot be remade "
        "with different settings in place. Changed: "
        + ("; ".join(changes) if changes else "(the arguments differ)") + ". "
        "Put the settings back to continue this project; for a late failure use "
        "'Phase linking onward' or 'MintPy onward', which do not re-plan. A longer "
        "record or new stack settings need a new project (or delete "
        f"{cfg.stack_dir} and run from scratch).")


def generate_run_files(cfg: PipelineConfig, dem_path: Path, dry_run: bool = False):
    """
    Invoke stackSentinel.py to produce run_files/ and configs/, then apply
    the unwrap mode (stack_crop_unwrap) to the unwrap run file.

    Generation is cheap and side-effect-free apart from those two directories,
    so it is also how plan_stack() learns the real date and burst counts
    without downloading anything.
    """
    argv, files = _generate_run_files(cfg, dem_path, dry_run)
    if files is not None:
        _apply_unwrap_mode(cfg)
    return argv, files


def _generate_run_files(cfg: PipelineConfig, dem_path: Path, dry_run: bool = False):
    cfg.stack_dir.mkdir(parents=True, exist_ok=True)
    argv = _stack_argv(cfg, dem_path)
    log.info("stackSentinel: %s", " ".join(argv))
    if dry_run:
        return argv, None

    # stackSentinel refuses to write into an existing run_files/ and exits 1
    # with "Please remove or rename this folder". Decide what that means here
    # rather than making every rerun a manual cleanup:
    #   same arguments  -> the existing plan is the plan; reuse it, and the
    #                      per-command status markers make the rerun a no-op
    #                      for work already done.
    #   different        -> the plan is stale; archive it and regenerate. The
    #                      markers survive (they live outside run_files/), and
    #                      commands whose text changed get new hashes and
    #                      correctly re-run.
    #   either, when acquisitions on disk are missing from the plan (a date
    #                      that failed to download and arrived on a resume,
    #                      20161206 at Site A): the argv carries only
    #                      `-s slc/`, so equality said nothing about the
    #                      dates. Before coregistration -> regenerate; after
    #                      -> refuse, naming the dates (see _refuse_missing).
    marker = cfg.stack_dir / "generation.json"
    rf = cfg.stack_dir / "run_files"
    started = coreg_started(cfg)
    rec: dict = {}
    missing: list = []
    if rf.exists():
        previous, adopted = None, False
        if marker.exists():
            try:
                rec = json.loads(marker.read_text())
                previous = rec.get("argv")
                adopted = "adopted" in rec
            except json.JSONDecodeError:
                rec = {}
        missing = dates_missing_from_plan(cfg, rec)
        if missing and started:
            why = _extension_problem(cfg, missing, rec)
            if why:
                _refuse_missing(cfg, missing, why)
        if (previous is not None and previous != argv and started and not missing
                and _without_exclude(previous) == _without_exclude(argv)):
            # Only stack_exclude_dates changed, on a stack under way:
            # stackSentinel would find the coregistered dates and plan nothing
            # new, so the exclusion is applied to the existing plan instead
            # (plan_commands, set_aside_excluded).
            files = run_files(cfg)
            rec.update({"argv": argv, "exclude_dates_applied_to_plan": time.strftime("%Y-%m-%dT%H:%M:%S")})
            marker.write_text(json.dumps(rec, indent=2))
            log.info("Only the excluded dates changed (%s) - reusing the existing plan (%d run files); the "
                     "excluded dates' commands are skipped.",
                     "; ".join(_argv_changes(previous, argv)), len(files))
            return argv, files
        if previous is not None and previous != argv and _cleaned_coreg_markers(cfg):
            _refuse_replan(cfg, previous, argv)
        if missing:
            log.warning("%d acquisition(s) on disk are not in the existing plan (%s). %s",
                        len(missing), ", ".join(missing[:10]) + ("..." if len(missing) > 10 else ""),
                        "They all follow the last planned date and the coregistered "
                        "stack is intact, so stackSentinel's update mode adds them."
                        if started else
                        "Coregistration has not started, so the plan is archived and "
                        "regenerated to include them.")
        elif planned_dates(cfg, rec) is None:
            log.warning("No record of which acquisitions the existing plan was made "
                        "from (no input_dates, no stack/SAFE_files.txt) - cannot check "
                        "that every downloaded date is in it.")
        if previous is None and not missing:
            # A plan exists but nothing recorded how it was made - the usual
            # case is a run that predates this marker. Reuse it rather than
            # archiving: destroying a working plan on the strength of missing
            # evidence is the worse failure, and if it really is stale the
            # commands differ and their hashes force a re-run anyway.
            files = run_files(cfg)
            known = _safe_files_dates(cfg)
            marker.write_text(json.dumps(
                {"argv": argv, "adopted": time.strftime("%Y-%m-%dT%H:%M:%S"),
                 "note": "adopted a pre-existing plan; argv assumed to match",
                 "n_run_files": len(files),
                 **({"input_dates": sorted(known), "input_dates_source": "SAFE_files.txt"}
                    if known else {})}, indent=2))
            log.warning("run_files/ exists but no generation record - adopting "
                        "the existing plan (%d run files) and recording these "
                        "arguments. Delete run_files/ to force a clean "
                        "regeneration.", len(files))
            return argv, files
        if previous == argv and not missing:
            files = run_files(cfg)
            if not rec.get("input_dates"):
                # Records made before 5 Oct 2026: backfill the date set from
                # what stackSentinel wrote, so the next check has it.
                known = _safe_files_dates(cfg)
                if known:
                    rec.update({"input_dates": sorted(known),
                                "input_dates_source": "SAFE_files.txt"})
                    marker.write_text(json.dumps(rec, indent=2))
            if adopted:
                # The record was written by us on adoption, not by a
                # generation; equality proves nothing about the plan itself.
                log.info("run_files/ was ADOPTED under these arguments (not "
                         "generated with them) - reusing the existing plan (%d "
                         "run files). patch_configs() brings looks, nomcf, "
                         "defomax and the ESD threshold up to date; anything "
                         "else that differs is not applied.", len(files))
            else:
                log.info("run_files/ was generated with these arguments - reusing "
                         "the existing plan (%d run files). Completed commands "
                         "will be skipped.", len(files))
            return argv, files
        stamp = time.strftime("%Y%m%d_%H%M%S")
        for d in ("run_files", "configs"):
            src = cfg.stack_dir / d
            if src.exists():
                src.rename(cfg.stack_dir / f"{d}.superseded_{stamp}")
        log.warning("%s, so the existing plan is stale. Archived it as "
                    "*.superseded_%s and regenerating. Commands whose text is "
                    "unchanged stay skipped; the rest will re-run.%s",
                    "Acquisitions were added" if missing else
                    "Arguments changed since the last generation", stamp,
                    "" if previous else
                    " (No record of the previous arguments was found, so the "
                    "difference could not be shown.)")
    elif _cleaned_coreg_markers(cfg):
        # run_files/ is gone (deleted by hand) on a stack whose coregistered
        # images were removed after use: stackSentinel would see no date
        # directories and plan a NEW stack from the few SAFEs still in slc/.
        raise RuntimeError(
            f"{cfg.stack_dir / 'run_files'} is missing, and this project's coregistered "
            "images were removed after use (coreg_secondarys holds only <date>.cleaned "
            "markers). A new plan made now would cover only the archives still in slc/, "
            f"not the record. Restore run_files/ (look for {cfg.stack_dir}/run_files.superseded_*), "
            "use 'Phase linking onward' / 'MintPy onward', or start a new project.")
    proc = subprocess.run(argv, cwd=cfg.stack_dir, env=_env(),
                          capture_output=True, text=True)
    (cfg.stack_dir / "stackSentinel.log").write_text(
        f"$ {' '.join(argv)}\n\n{proc.stdout}\n--- stderr ---\n{proc.stderr}\n")
    if proc.returncode != 0:
        # stackSentinel exits 1 for a genuine no-op as well as for errors.
        # "No new acquisition found to update the stack" means every date is
        # already coregistered - the correct outcome of a rerun, not a failure.
        benign = ("no new acquisition found" in (proc.stdout or "").lower()
                  or "the stack already exists" in (proc.stdout or "").lower())
        if benign:
            restored = _restore_superseded(cfg)
            files = run_files(cfg)
            if missing:
                raise RuntimeError(
                    f"stackSentinel added none of the {len(missing)} acquisition(s) "
                    f"missing from the plan ({', '.join(missing[:10])}); the previous plan "
                    "was restored. They would be left out of every product - list them in "
                    "allow_missing_dates to carry on without them, or start a new project.")
            log.info("stackSentinel reports no new acquisitions - the stack is "
                     "already complete for these dates.%s",
                     f" Restored the previously generated plan ({len(files)} "
                     f"run files) from {restored.name}." if restored else "")
            if restored:
                log.warning(
                    "THE RESTORED PLAN WAS GENERATED WITH THE PREVIOUS "
                    "ARGUMENTS. stackSentinel will not regenerate over an "
                    "existing coregistered stack, so parameters that live in "
                    "the generated configs are applied by patching instead - "
                    "currently the unwrap cost mode, defomax and the ESD "
                    "coherence threshold. Anything else that changed (looks, "
                    "swaths, network connectivity) is NOT applied and needs "
                    "%s removed to take effect.", cfg.stack_dir / "coreg_secondarys")
            if files:
                return argv, files
            raise RuntimeError(
                "stackSentinel found nothing to do but no run files are "
                f"present in {cfg.stack_dir / 'run_files'}. Delete "
                f"{cfg.stack_dir / 'coreg_secondarys'} to force a rebuild.")
        if "original safe files" in ((proc.stdout or "") + (proc.stderr or "")).lower():
            # stackSentinel's update mode needs the SAFEs of the latest
            # 2 x num_overlap_connections coregistered dates. Never leave the
            # project without a plan because of it.
            restored = _restore_superseded(cfg)
            raise RuntimeError(
                "stackSentinel could not extend the stack: it needs the original "
                "archives of the most recent coregistered dates, and SAFE cleanup "
                "has removed them. "
                + (f"The previous plan was restored from {restored.name}. " if restored else "")
                + f"See {cfg.stack_dir / 'stackSentinel.log'}. Start a new project to "
                "process the longer record.")
        raise RuntimeError(
            f"stackSentinel.py failed ({proc.returncode}); see "
            f"{cfg.stack_dir / 'stackSentinel.log'}")
    files = run_files(cfg)
    # The plan's input dates, so the next run can tell whether anything on
    # disk is missing from it. An update-mode generation plans only the new
    # dates, so the previous plan's dates carry over.
    used = _safe_files_dates(cfg) or available_input_dates(cfg)
    if started:
        used |= set(rec.get("input_dates") or []) | (_safe_files_dates(cfg) or set()) \
            | coregistered_dates(cfg)
    from .data_acquisition import accepted_missing_dates
    accepted, accept_all = accepted_missing_dates(cfg)
    marker.write_text(json.dumps(
        {"argv": argv, "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
         "n_run_files": len(files), "input_dates": sorted(used),
         "accepted_missing_dates": ["all"] if accept_all else sorted(accepted)}, indent=2))
    log.info("Generated %d run files", len(files))
    return argv, files


def invalidate_after_looks_change(cfg: PipelineConfig) -> int:
    """
    Clear everything built at the previous multilook factor.

    Looks are applied at MERGE time, so every product from run_12 onward - the
    merged geometry, the merged interferograms, filtered coherence and
    unwrapped phase - is on the old grid. Their status markers would otherwise
    make the next run skip them, leaving geometry at one size and
    interferograms at another, and MintPy refuses to stack layers whose shapes
    disagree. That failure surfaces late and reads as a corrupt product rather
    than a stale one, so this is done automatically rather than left to be
    remembered.

    Coregistration (run_01-run_11) and the per-burst interferograms (run_13)
    are computed at full resolution and are NOT affected, which is what keeps
    a looks change cheap: about ten minutes of rebuild rather than a fresh
    coregistration.
    """
    # run_13 MUST be included even though burst interferograms are computed at
    # full resolution and are not themselves looks-dependent. cleanup_bursts
    # deletes them once they have been merged, so their "done" markers outlive
    # the data. Invalidating the merged products without invalidating run_13
    # leaves the stage with no inputs AND no reason to run - the pipeline
    # cannot rebuild, and every downstream command fails instantly.
    #
    # The general rule this encodes: invalidating a stage must cascade to any
    # earlier stage whose outputs were consumed and then cleaned.
    cleaned = sorted((cfg.stack_dir / "coreg_secondarys").glob("*.cleaned")) \
        if (cfg.stack_dir / "coreg_secondarys").exists() else []
    if cleaned:
        raise RuntimeError(
            f"Looks changed, but the coregistered SLCs of {len(cleaned)} date(s) were "
            "deleted after their interferograms were filtered "
            "(stack_cleanup_coreg_after_filter). The merge cannot be redone from "
            "what is on disk: this needs a fresh coregistration - delete "
            f"{cfg.stack_dir / 'coreg_secondarys'} and {cfg.stack_dir / 'secondarys'} "
            "and re-run (the SAFEs will be downloaded again).")
    cleared = 0
    for run in ("run_12_merge_reference_secondary_slc", "run_13_generate_burst_igram",
                "run_14_merge_burst_igram",
                "run_15_filter_coherence", "run_16_unwrap"):
        d = cfg.stack_dir / ".status" / run
        if d.exists():
            n = len(list(d.glob("*.done")))
            shutil.rmtree(d, ignore_errors=True)
            cleared += n
    freed = 0
    # Pair directories left empty by cleanup_bursts confuse nothing, but they
    # make "87 interferograms exist" read as true when none do.
    igdir = cfg.stack_dir / "interferograms"
    if igdir.exists():
        for d in igdir.iterdir():
            if d.is_dir() and not any(d.iterdir()):
                d.rmdir()
    for sub in ("merged/interferograms", "merged/geom_reference", "merged/SLC"):
        path = cfg.stack_dir / sub
        if path.exists():
            freed += sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
            shutil.rmtree(path, ignore_errors=True)
    # The crop window is in pixels of the old grid, and MintPy's loaded stack
    # is at the old shape - both would be reused silently otherwise.
    from .unwrap_crop import WINDOW_FILE
    win = cfg.stack_dir / WINDOW_FILE
    if win.exists():
        win.unlink()
    inputs = cfg.mintpy_dir / "inputs"
    if inputs.exists():
        freed += sum(f.stat().st_size for f in inputs.rglob("*") if f.is_file())
        shutil.rmtree(inputs, ignore_errors=True)
        # mark mintpy/ unfinished first: geo/ and run_products.json stay, and
        # without the marker 'Products only' would build from the old looks'
        # geo products until MintPy runs again (time_series.mark_run_incomplete)
        from .time_series import mark_run_incomplete
        mark_run_incomplete(cfg, time.strftime("%Y%m%d_%H%M%S"), time.time())
        for h5 in cfg.mintpy_dir.glob("*.h5"):
            h5.unlink()
        log.warning("Removed mintpy/inputs and mintpy/*.h5: they were loaded at "
                    "the previous looks.")
    log.warning("Cleared %d status markers and %.1f GB of products built at the "
                "previous looks. run_13 onward will rebuild (burst "
                "interferograms included, because cleanup removed them after "
                "merging); coregistration - run_01 to run_11, the expensive "
                "part - is unaffected.",
                cleared, freed / 1024 ** 3)
    return cleared


def patch_configs(cfg: PipelineConfig) -> int:
    """
    Rewrite the unwrap configs so snaphu actually optimises.

    THIS IS THE POINT OF THE WHOLE BACKEND. Stack.py hardcodes noMCF='False'
    and defoMax='2' with no CLI exposure, which routes unwrap.py to
    costMode='SMOOTH', initMethod='MCF', initOnly=True - minimum-cost-flow
    initialisation and then stop. Setting nomcf True selects costMode='DEFO',
    initMethod='MST', maxComponents=20, which is snaphu's actual statistical
    cost optimisation.

    defomax=2 cycles is 55 mm of LOS. A longwall face moves several times that
    between 12-day acquisitions, so the default guarantees the fast ground is
    clipped. Returns the number of files changed.
    """
    cfg_dir = cfg.stack_dir / "configs"
    if not cfg_dir.exists():
        log.warning("no configs/ directory - nothing to patch")
        return 0
    # Multilooking. stackSentinel bakes -r/-z into five config families at
    # generation time, and it will not regenerate over an existing coregistered
    # stack - so without this, changing looks would mean discarding the whole
    # coregistration. ALL of them must move together: the geometry merges
    # (hgt/lat/lon/los/shadowMask/incLocal) produce the grid MintPy reads, and
    # if they disagree with the interferograms by even one row the stack will
    # not load.
    want_r, want_a = str(cfg.stack_range_looks), str(cfg.stack_azimuth_looks)
    looks_changed, looks_files = 0, 0
    for f in sorted(cfg_dir.glob("config_*")):
        txt = f.read_text()
        if not re.search(r"^(range_looks|rlks)\s*:", txt, flags=re.MULTILINE):
            continue
        looks_files += 1
        new_txt = re.sub(r"^(range_looks\s*:\s*).*$", rf"\g<1>{want_r}", txt,
                         flags=re.MULTILINE)
        new_txt = re.sub(r"^(azimuth_looks\s*:\s*).*$", rf"\g<1>{want_a}", new_txt,
                         flags=re.MULTILINE)
        new_txt = re.sub(r"^(rlks\s*:\s*).*$", rf"\g<1>{want_r}", new_txt,
                         flags=re.MULTILINE)
        new_txt = re.sub(r"^(alks\s*:\s*).*$", rf"\g<1>{want_a}", new_txt,
                         flags=re.MULTILINE)
        if new_txt != txt:
            f.write_text(new_txt)
            looks_changed += 1
    if looks_changed:
        log.warning("Patched %d of %d looks-bearing configs to %sx%s.",
                    looks_changed, looks_files, want_r, want_a)
        invalidate_after_looks_change(cfg)

    # ESD threshold. This is nominally set by stackSentinel's -e flag, so it
    # would not need patching - except that a plan RESTORED after a no-op
    # regeneration still carries the value it was generated with. stackSentinel
    # refuses to regenerate over an existing coregistered stack, so without
    # this the only way to change the threshold would be to discard the
    # coregistration and start again.
    want_esd = f"{cfg.esd_coherence_threshold}"
    esd_changed = 0
    for f in sorted(cfg_dir.glob("config_misreg_*")):
        txt = f.read_text()
        new_txt = re.sub(r"^(coh_threshold\s*:\s*).*$", rf"\g<1>{want_esd}",
                         txt, flags=re.MULTILINE)
        if new_txt != txt:
            f.write_text(new_txt)
            esd_changed += 1
    if esd_changed:
        log.info("Patched %d misregistration configs to coh_threshold=%s "
                 "(ESD needs coherence in the small burst-overlap strips, which "
                 "decorrelate faster than the full scene).",
                 esd_changed, want_esd)

    want_nomcf = "True" if cfg.stack_unw_full_cost else "False"
    want_defomax = str(cfg.stack_unw_defomax)
    changed = 0
    for f in sorted(cfg_dir.glob("config_igram_unw*")):
        txt = f.read_text()
        new = re.sub(r"^nomcf\s*:.*$", f"nomcf : {want_nomcf}",
                     txt, flags=re.MULTILINE)
        new = re.sub(r"^defomax\s*:.*$", f"defomax : {want_defomax}",
                     new, flags=re.MULTILINE)
        from .unwrap_crop import apply_engine
        new = apply_engine(new, cfg)          # engine keys (the marker fingerprint sees them)
        if new != txt:
            f.write_text(new)
            changed += 1
    log.info("Patched %d unwrap configs: nomcf=%s (snaphu %s), defomax=%s cycles "
             "(%.0f mm LOS)", changed, want_nomcf,
             "full DEFO cost" if cfg.stack_unw_full_cost else "MCF init only",
             want_defomax, float(want_defomax) * 27.73)
    # Counting edits cannot distinguish "already correct" from "never applied",
    # and those need opposite responses. Verify the end state instead - for
    # every key this function owns, across every config that carries it.
    problems = verify_configs(cfg)
    unw = sorted(cfg_dir.glob("config_igram_unw*"))
    if not unw:
        log.warning("No unwrap configs exist in %s. Nothing has been generated "
                    "yet, or the workflow does not include unwrapping.", cfg_dir)
    elif problems:
        for p in problems[:5]:
            log.error("config verification: %s", p)
        raise RuntimeError(
            f"{len(problems)} generated config(s) do not carry the requested "
            f"settings after patching (first: {problems[0]}). Refusing to run a "
            "stack whose unwrapper or grid would differ from the config.")
    else:
        log.info("Verified all %d unwrap configs carry nomcf=%s, defomax=%s and "
                 "all %d looks-bearing configs carry %sx%s.",
                 len(unw), want_nomcf, want_defomax, looks_files, want_r, want_a)
    return changed


def verify_configs(cfg: PipelineConfig) -> list:
    """
    End-state check of the generated configs against cfg. Returns problems.

    Looks are the dangerous one: a geometry merge left at the old looks
    while the interferograms move is exactly the shape mismatch MintPy
    refuses to load, hours later.
    """
    cfg_dir = cfg.stack_dir / "configs"
    if not cfg_dir.exists():
        return []
    want_r, want_a = str(cfg.stack_range_looks), str(cfg.stack_azimuth_looks)
    want_nomcf = "True" if cfg.stack_unw_full_cost else "False"
    want_defomax = str(cfg.stack_unw_defomax)
    want_esd = f"{cfg.esd_coherence_threshold}"
    problems = []
    for f in sorted(cfg_dir.glob("config_*")):
        kv = _read_config_kv(f)
        name = f.name
        r = kv.get("range_looks", kv.get("rlks"))
        a = kv.get("azimuth_looks", kv.get("alks"))
        if r is not None and r != want_r:
            problems.append(f"{name}: range looks {r} != {want_r}")
        if a is not None and a != want_a:
            problems.append(f"{name}: azimuth looks {a} != {want_a}")
        if name.startswith("config_igram_unw"):
            if kv.get("nomcf") != want_nomcf:
                problems.append(f"{name}: nomcf {kv.get('nomcf')} != {want_nomcf}")
            if kv.get("defomax") != want_defomax:
                problems.append(f"{name}: defomax {kv.get('defomax')} != {want_defomax}")
        if name.startswith("config_misreg") and "coh_threshold" in kv:
            try:
                if float(kv["coh_threshold"]) != float(want_esd):
                    problems.append(f"{name}: coh_threshold {kv['coh_threshold']} != {want_esd}")
            except ValueError:
                problems.append(f"{name}: coh_threshold {kv['coh_threshold']!r} unparsable")
    return problems


# ----------------------------------------------------------------------
# input staging
# ----------------------------------------------------------------------
def stage_inputs(cfg: PipelineConfig, username=None, password=None,
                 skip_download=False) -> Path:
    """
    SLCs, precise orbits and the DEM.

    Orbits are pre-staged deliberately. stackSentinel resolves them during
    GENERATION by shelling out to fetchOrbit.py once per date against the
    Copernicus Data Space OData API - a long serial, credentialed network loop
    inside a step that should be instant. download_orbits() has existed in this
    package since the beginning and was never called; here it is required.
    """
    from . import data_acquisition as da

    if str(getattr(cfg, "miaplpy_relink_from", "") or "").strip():
        # F3 hook (8 Oct 2026): nothing to download or coregister here
        from . import miaplpy_backend as _mb
        raise SystemExit(_mb.relink_start_message(cfg))
    cfg.ensure_dirs()
    # Credentials are needed for the orbit download on EVERY path, not only
    # the full-archive one. Environment first (the launchers export ~/.netrc
    # into it), then a prompt - never silently None.
    if username is None and not skip_download:
        username, password = da.get_credentials()

    paths = []
    if getattr(cfg, "stack_use_burst2safe", False):
        # Assemble from bursts: same data, a fraction of the transfer. The
        # burst search is the catalogue query on this path; a separate SLC
        # search would only repeat it against a different product type.
        paths = da.download_bursts_as_safes(cfg) if not skip_download \
            else sorted(cfg.slc_dir.glob("*.SAFE"))
    elif not skip_download:
        results = da.search_scenes(cfg)
        log.info("%d SLC scenes match the AOI and date range", len(results))
        if not results:
            raise RuntimeError("no scenes found - check bbox, track and dates")
        paths = da.download_scenes(cfg, results, username, password)
    else:
        paths = sorted(cfg.slc_dir.glob("*.zip"))

    try:
        da.download_orbits(cfg, paths, username, password)
    except Exception as exc:                      # noqa: BLE001
        log.warning("Orbit pre-staging failed (%s). stackSentinel will fall back "
                    "to fetching them itself during generation, which is slow "
                    "and needs Copernicus Data Space credentials.", exc)

    # After the SAFEs, so the DEM can be sized from their footprints.
    return da.get_dem(cfg)


# ----------------------------------------------------------------------
# planning
# ----------------------------------------------------------------------
class WeatherProblem(dict):
    """One finding of check_weather_credentials: {"level": "fail"|"warn", "text": ...}.

    A dict (the contract: doctor and the dashboard read level/text), which
    still behaves like the plain string every earlier caller printed or
    sliced - str(p) and f"{p}" give the text, p[:160] slices it - so a
    caller not yet updated keeps working instead of crashing on a dict."""

    def __init__(self, level: str, text: str):
        super().__init__(level=level, text=text)

    @property
    def level(self) -> str:
        return self["level"]

    @property
    def text(self) -> str:
        return self["text"]

    def __getitem__(self, key):
        if isinstance(key, (int, slice)):
            return dict.__getitem__(self, "text")[key]
        return dict.__getitem__(self, key)

    def __str__(self) -> str:
        return self["text"]

    __format__ = lambda self, spec: format(self["text"], spec)       # noqa: E731


# The dataset PyAPS3 / MintPy's tropo_pyaps3 actually requests.
ERA5_DATASET = "reanalysis-era5-pressure-levels"
# Licence ids that have granted ERA5 at some point. ONLY a fallback for when
# the catalogue cannot be read: the CDS moved ERA5 from the Copernicus
# products licence to CC-BY ('cc-by', rev 1; catalogue read 5 Oct 2026), and
# the hard-coded old id reported "NOT accepted ... 403" on an account that
# downloaded 270 Site B GRIBs on 29 Sep - then blocked every preflight.
ERA5_KNOWN_LICENCES = ("cc-by", "licence-to-use-copernicus-products")


def _int_or_none(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def era5_required_licences(base: str, dataset: str = ERA5_DATASET, timeout: float = 20.0):
    """The licences the CDS catalogue says `dataset` needs, as {id: (rev, title)},
    from {base}/catalogue/v1/collections/<dataset> (public, no token): the
    links with rel == "license". Live on 5 Oct 2026 for ERA5 pressure levels:
    [{'rel': 'license', 'id': 'cc-by', 'rev': 1, 'title': 'CC-BY licence',
    'href': 'https://spdx.org/licenses/CC-BY-4.0'}]. None when the catalogue
    cannot be read or lists no licence: the requirement is then UNKNOWN, and
    nothing is invented in its place."""
    import json
    import urllib.request
    try:
        with urllib.request.urlopen(f"{base}/catalogue/v1/collections/{dataset}", timeout=timeout) as r:
            doc = json.loads(r.read())
    except Exception as exc:                                  # noqa: BLE001
        log.info("CDS catalogue for %s not readable (%s)", dataset, exc)
        return None
    out = {}
    for link in (doc.get("links") or []) if isinstance(doc, dict) else []:
        if isinstance(link, dict) and link.get("rel") == "license" and link.get("id"):
            out[str(link["id"])] = (_int_or_none(link.get("rev")), str(link.get("title") or link["id"]))
    return out or None


def check_weather_credentials(cfg: PipelineConfig) -> list:
    """
    Verify ERA5 access BEFORE a long run starts.

    MintPy reaches its tropospheric step near the END of the chain, after
    coregistration, interferogram formation, unwrapping and inversion. On a
    full stack that is many hours in. A missing token or an unaccepted licence
    then fails the run at the most expensive possible moment, and the CDS
    error mentions neither.

    Returns a list of WeatherProblem {"level": "fail"|"warn", "text": ...};
    empty means ready. "fail" = ERA5 will not work as configured (no or
    unparseable ~/.cdsapirc, legacy uid:key token, token rejected, a licence
    the dataset requires not accepted). "warn" = could not be verified (CDS
    unreachable, or its catalogue unreadable and the account holds none of
    the known ERA5 licences): a preflight shows it but must not block on it.

    The licences required are READ from the CDS catalogue for the dataset
    PyAPS3 requests and compared with the account's accepted (id, revision)
    pairs - CDS acceptance is per revision, so a licence bumped to a new
    revision has to be accepted again.
    """
    if getattr(cfg, "mintpy_tropo_method", "no") != "pyaps":
        return []
    import json
    import os
    import urllib.error
    import urllib.request

    fail = lambda t: WeatherProblem("fail", t)                # noqa: E731
    warn = lambda t: WeatherProblem("warn", t)                # noqa: E731
    problems = []
    rc = Path(os.path.expanduser("~/.cdsapirc"))
    if not rc.exists():
        return [fail(f"mintpy_tropo_method is 'pyaps' but {rc} does not exist. ERA5 "
                     "download will fail. Create it, or set mintpy_tropo_method: no.")]
    try:
        from cdsapi.api import read_config
        conf = read_config(str(rc))
    except Exception as exc:                                  # noqa: BLE001
        return [fail(f"{rc} could not be parsed ({exc}).")]
    if not conf.get("key") or not conf.get("url"):
        return [fail(f"{rc} is missing 'url' or 'key'.")]
    if ":" in conf["key"]:
        problems.append(fail(
            f"{rc} holds a legacy 'uid:key' token. The CDS migrated in "
            "September 2024; the key is now the Personal Access Token alone. "
            "cdsapi splits on the first colon, so this authenticates as the "
            "UID and fails confusingly."))

    base = conf["url"].rstrip("/")
    page = f"https://cds.climate.copernicus.eu/datasets/{ERA5_DATASET}?tab=download"
    try:
        req = urllib.request.Request(base + "/profiles/v1/account/licences",
                                     headers={"PRIVATE-TOKEN": conf["key"]})
        with urllib.request.urlopen(req, timeout=20) as r:
            rows = json.loads(r.read()).get("licences", [])
    except urllib.error.HTTPError as e:
        if e.code in (401, 403):
            problems.append(fail(f"CDS rejected the token in {rc} (HTTP {e.code} {e.reason}). "
                                 "Copy the Personal Access Token again from your CDS profile page."))
        else:
            problems.append(warn(f"Could not verify ERA5 access: the CDS answered HTTP {e.code} "
                                 f"{e.reason}. The run can start; MintPy will fail at its "
                                 "tropospheric step if access really is missing."))
        return problems
    except Exception as exc:                                  # noqa: BLE001
        problems.append(warn(f"Could not reach the CDS to verify ERA5 access ({exc}). The run can "
                             "start; MintPy will fail at its tropospheric step if access really "
                             "is missing."))
        return problems
    revs: dict = {}
    for row in rows if isinstance(rows, list) else []:
        if isinstance(row, dict) and row.get("id"):
            revs.setdefault(str(row["id"]), []).append(
                _int_or_none(row.get("revision", row.get("rev"))))
    # id -> highest accepted revision (None when the profile gives none)
    accepted = {k: max((r for r in v if r is not None), default=None) for k, v in revs.items()}

    required = era5_required_licences(base)
    if required is None:
        # The requirement is unknown. Any licence that has ever granted ERA5
        # is taken as a pass; with none of them the account may well lack
        # access, but saying "every request will fail" would be a guess.
        if not any(k in accepted for k in ERA5_KNOWN_LICENCES):
            problems.append(warn(
                "Could not read from the CDS catalogue which licence ERA5 needs, and this "
                f"account has accepted none of the known ones ({', '.join(ERA5_KNOWN_LICENCES)}). "
                f"If ERA5 downloads are refused, accept the licence at {page} -> Terms of use "
                "(a legal agreement: the account holder has to do it)."))
        return problems
    for lid, (rev, title) in sorted(required.items()):
        have = accepted.get(lid, "absent")
        if have == "absent":
            problems.append(fail(
                f"The CDS token works but the ERA5 licence '{title}' ({lid}) has not been "
                f"accepted on this account, so ERA5 requests will be refused. Accept it at {page} "
                "-> Terms of use. (Accepting a licence is a legal agreement and has to be done by "
                "the account holder.)"))
        elif rev is not None and have is not None and have < rev:
            problems.append(fail(
                f"The ERA5 licence '{title}' ({lid}) was accepted at revision {have}, but the "
                f"CDS now requires revision {rev}, so ERA5 requests will be refused. Accept the "
                f"new revision at {page} -> Terms of use."))
    return problems


# ----------------------------------------------------------------------
# ERA5 prefetch
# ----------------------------------------------------------------------
# MintPy's tropospheric step (tropo_pyaps3 -> pyaps3.ECMWFdload) downloads one
# GRIB per date, SERIALLY, at the very end of the run: Site B 29 Sep 2026,
# 270 files from 10:21 to 13:39 (~3.3 h) after ~12 h of phase linking - and a
# CDS outage or queue backlog is discovered only then. The dates and the
# acquisition hour are known as soon as the inputs are staged, so the files
# are fetched during coregistration instead, under EXACTLY the names MintPy
# will look for (it skips any file already present):
#     <weather dir>/ERA5/ERA5_N30_N50_W90_W80_<YYYYMMDD>_<HH>.grb
# A few at a time: the CDS queues requests per user. It also refuses new
# requests while its queue for the dataset is full: on 22 Oct 2026 it refused
# 46 of 48 and 265 of 270 ("Maximum number queued requests for this dataset
# has been reached"), from the first request on, with nothing else of this
# account queued; the few it accepted waited an hour or more before running.
ERA5_PREFETCH_WORKERS = 4
ERA5_PRESSURE_LEVELS = ['1', '2', '3', '5', '7', '10', '20', '30', '50', '70', '100', '125', '150',
                        '175', '200', '225', '250', '300', '350', '400', '450', '500', '550', '600',
                        '650', '700', '750', '775', '800', '825', '850', '875', '900', '925', '950',
                        '975', '1000']
ERA5_PREFETCH_STATUS = "era5_prefetch.json"
# Extent tolerance: the processed area's lat/lon extent is only estimated
# before the geometry exists. When moving it by this much changes MintPy's
# tile, the tile is ambiguous and nothing is fetched (MintPy fetches what it
# needs itself) rather than hours of files under a name it will not use.
ERA5_EXTENT_TOL_DEG = 0.1
_SAFE_TIMES_RE = re.compile(r"_(\d{8})T(\d{2})(\d{2})(\d{2})_(\d{8})T(\d{2})(\d{2})(\d{2})_")
_ERA5_AUTH_RE = re.compile(r"\b(401|403)\b|unauthori[sz]ed|forbidden|licen[cs]e|not accepted", re.I)
# The CDS turning a request away because its queue is full: not a failure of
# that date. The prefetch waits and asks again (1 min, doubling to 15 min).
_ERA5_BUSY_RE = re.compile(r"Maximum number (?:of )?queued requests|\b429\b|too many requests", re.I)
ERA5_BUSY_WAIT_S = (60.0, 900.0)


def era5_snwe(lat0: float, lat1: float, lon0: float, lon1: float,
              min_buffer: int = 2, step: int = 10) -> tuple:
    """MintPy's ERA5 download area for a lat/lon extent: tropo_pyaps3.get_snwe
    (MintPy 1.6): 2 degrees of buffer, then out to multiples of 10."""
    import math
    S = math.floor(min(lat0, lat1) - min_buffer)
    N = math.ceil(max(lat0, lat1) + min_buffer)
    W = math.floor(min(lon0, lon1) - min_buffer)
    E = math.ceil(max(lon0, lon1) + min_buffer)
    if step > 1:
        S, W = S - S % step, W - W % step
        N = N if N % step == 0 else N + (step - N % step)
        E = E if E % step == 0 else E + (step - E % step)
    return int(S), int(N), int(W), int(E)


def era5_area_str(snwe) -> str:
    """'_N30_N50_W90_W80' - pyaps3.utils.snwe2str / tropo_pyaps3.get_grib_filenames."""
    s, n, w, e = snwe
    return (f"_S{abs(s)}" if s < 0 else f"_N{abs(s)}") + (f"_S{abs(n)}" if n < 0 else f"_N{abs(n)}") \
        + (f"_W{abs(w)}" if w < 0 else f"_E{abs(w)}") + (f"_W{abs(e)}" if e < 0 else f"_E{abs(e)}")


def era5_grib_name(snwe, date: str, hour: str) -> str:
    return f"ERA5{era5_area_str(snwe)}_{date}_{hour}.grb"


def era5_hour(seconds_utc: float) -> tuple:
    """(hour 'HH', ambiguous) as tropo_pyaps3.closest_weather_model_hour picks
    it from CENTER_LINE_UTC: the nearest of ERA5's hours 0..23 (no wrap to the
    next day). Ambiguous when the two nearest are within 2 minutes of a tie,
    since the prefetch knows the time only from the SAFE names."""
    t = float(seconds_utc) / 3600.0
    d = sorted((abs(h - t), h) for h in range(24))
    return f"{d[0][1]:02d}", (d[1][0] - d[0][0]) < 2.0 / 60.0


def era5_request(date: str, snwe, hour: str, humidity: str = "specific_humidity") -> dict:
    """The request pyaps3.ECMWFdload sends for one date (pyaps3 0.3.7), so a
    prefetched file is the same data MintPy would have downloaded."""
    s, n, w, e = snwe
    return {"product_type": ["reanalysis"],
            "variable": ["geopotential", "temperature", humidity],
            "year": [date[0:4]], "month": [date[4:6]], "day": [date[6:8]],
            "time": [f"{hour}:00"], "pressure_level": list(ERA5_PRESSURE_LEVELS),
            "data_format": "grib", "area": [n, w, s, e]}


def fetch_era5_grib(date: str, path, snwe, hour: str, client=None) -> str:
    """Download one GRIB to <path>.part and rename it into place, so MintPy
    never sees a partial file. 'ok' / 'cached' / 'cached-meanwhile' (MintPy
    or another prefetch got there first) / 'FAILED <reason>'."""
    path = str(path)
    if os.path.exists(path):
        return "cached"
    part = path + ".part"
    try:
        if client is None:
            import cdsapi
            client = cdsapi.Client(quiet=True)
        client.retrieve(ERA5_DATASET, era5_request(date, snwe, hour), part)
        if not os.path.exists(part) or os.path.getsize(part) == 0:
            raise RuntimeError("the CDS returned no data")
        if os.path.exists(path):
            os.unlink(part)
            return "cached-meanwhile"
        os.replace(part, path)
        return "ok"
    except Exception as exc:                                  # noqa: BLE001
        try:
            os.unlink(part)
        except FileNotFoundError:
            pass
        return f"FAILED {exc}"


def era5_dir(cfg: PipelineConfig) -> Path:
    """Where MintPy will look: the phase-linked route sets weatherDir to
    <project>/mintpy (miaplpy_backend.write_template); the stack route leaves
    it 'auto', i.e. $WEATHER_DIR, else MintPy's working directory."""
    if cfg.backend == "miaplpy":
        return cfg.project_dir / "mintpy" / "ERA5"
    if os.environ.get("WEATHER_DIR"):
        return Path(os.path.expandvars(os.path.expanduser(os.environ["WEATHER_DIR"]))) / "ERA5"
    return cfg.mintpy_dir / "ERA5"


def _acquisition_times(cfg: PipelineConfig) -> dict:
    """{date: seconds of day UTC at mid-acquisition} from the SAFE names in
    slc/ and in stack/SAFE_files.txt (which outlives the SAFE cleanup).
    Site B: 23:47:41-23:47:46 -> 85663.5 s; MintPy's CENTER_LINE_UTC
    85662.98 s."""
    names = [q.name for q in cfg.slc_dir.glob("*.SAFE")] if cfg.slc_dir.exists() else []
    sf = cfg.stack_dir / "SAFE_files.txt"
    if sf.exists():
        names += [Path(x.strip()).name for x in sf.read_text().replace(",", "\n").splitlines() if x.strip()]
    out = {}
    for nm in names:
        m = _SAFE_TIMES_RE.search(nm)
        if not m:
            continue
        t0 = int(m.group(2)) * 3600 + int(m.group(3)) * 60 + int(m.group(4))
        t1 = int(m.group(6)) * 3600 + int(m.group(7)) * 60 + int(m.group(8))
        if t1 < t0:                                           # crosses midnight
            t1 += 86400
        out[m.group(1)] = ((t0 + t1) / 2.0) % 86400
    return out


def _processed_extent(cfg: PipelineConfig) -> tuple:
    """(W, S, E, N, what) of the ground MintPy's geometry will cover: the
    phase-linking crop (bbox + stack_bbox_margin_deg) on the phase-linked
    route; the processed burst frame on the stack route."""
    m = float(getattr(cfg, "stack_bbox_margin_deg", 0.05) or 0.0)
    cw = cfg.stack_dir / "crop_window.json"
    if cfg.backend == "miaplpy" or cw.exists():
        try:
            rec = json.loads(cw.read_text())
            w, s_, e, n = (float(v) for v in rec["bbox_wsen"])
            m = float(rec.get("margin_deg", m))
        except (OSError, ValueError, KeyError, TypeError):
            w, s_, e, n = (float(v) for v in cfg.bbox)
        return w - m, s_ - m, e + m, n + m, "the area box + margin (the phase-linking crop)"
    from . import data_acquisition as da
    w, s_, e, n = da.frame_bbox(cfg)
    return w, s_, e, n, "the processed burst frame"


def era5_prefetch_plan(cfg: PipelineConfig) -> dict:
    """What MintPy's ERA5 step will need and how much of it is on disk:
    {ok, reason, dir, snwe, hour, dates, needed, present, missing, files}.
    ok False (with a plain reason) when the tile or hour cannot be settled
    before the geometry exists - then nothing should be fetched. Also what a
    preflight can show: whether the cache already covers this project."""
    out = {"ok": False, "reason": "", "dir": str(era5_dir(cfg)), "snwe": None, "hour": None,
           "dates": [], "needed": 0, "present": 0, "missing": [], "files": {}}
    # the plan's dates (generation.json / SAFE_files.txt) or the SAFEs that
    # will be planned; SAFE names only as the last resort, inside the window
    times = _acquisition_times(cfg)
    s0, e0 = _window(cfg)
    dates = sorted(planned_dates(cfg) or available_input_dates(cfg)
                   or {d for d in times if s0 <= d < e0})
    if not dates:
        out["reason"] = "no acquisition dates known yet (nothing downloaded or planned)"
        return out
    if not times:
        out["reason"] = "no SAFE names to read the acquisition time from"
        return out
    import statistics
    secs = sorted(times.values())
    if secs[-1] - secs[0] > 3600:
        out["reason"] = (f"acquisition times spread over {(secs[-1] - secs[0]) / 3600:.1f} h "
                         "(mixed tracks, or across midnight): the ERA5 hour is not settled")
        return out
    hour, ambiguous = era5_hour(statistics.median(secs))
    if ambiguous:
        out["reason"] = "the acquisitions fall within 2 minutes of a half hour: the ERA5 hour is not settled"
        return out
    w, s_, e, n, what = _processed_extent(cfg)
    snwe = era5_snwe(s_, n, w, e)
    t = ERA5_EXTENT_TOL_DEG
    if era5_snwe(s_ - t, n + t, w - t, e + t) != snwe or era5_snwe(s_ + t, n - t, w + t, e - t) != snwe:
        out["reason"] = (f"{what} ({w:.2f}..{e:.2f} E, {s_:.2f}..{n:.2f} N) lies within "
                         f"{t} deg of an ERA5 tile edge: MintPy's tile is not settled until its geometry exists")
        return out
    d = Path(out["dir"])
    files = {dt: d / era5_grib_name(snwe, dt, hour) for dt in dates}
    present = [dt for dt, f in files.items() if f.exists() and f.stat().st_size > 0]
    out.update({"ok": True, "snwe": list(snwe), "hour": hour, "dates": dates, "needed": len(dates),
                "present": len(present), "missing": [dt for dt in dates if dt not in set(present)],
                "files": {dt: str(f) for dt, f in files.items()}, "extent": what})
    return out


def _era5_reason(text: str) -> str:
    """A fetch's failure on one line, without the per-request URL and job id
    the CDS puts in every answer, so the same cause reads (and counts) the
    same: '400 Client Error: Bad Request; The job has been rejected; Maximum
    number queued requests for this dataset has been reached. ...'."""
    t = re.sub(r"\s*for url:\s*\S+", "", str(text))
    t = re.sub(r"https?://\S+", "<url>", t)
    t = re.sub(r"\b[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}\b", "<id>", t, flags=re.I)
    return "; ".join(s.strip() for s in t.splitlines() if s.strip())[:300]


def start_era5_prefetch(cfg: PipelineConfig, workers: int | None = None, fetch=None, cancel=None) -> dict:
    """
    Fetch MintPy's ERA5 GRIBs now, while coregistration runs (pipeline.py
    calls this in a background thread; it BLOCKS until done or until
    `cancel`, a threading.Event, is set). Returns and writes
    <mintpy>/era5_prefetch.json: {status, reason, snwe, hour, needed,
    present_before, fetched, failed{date: why}, reasons{why: count},
    in_flight{date: since}, waiting, queue_full, dir}.

    "The CDS queue is full" (it refuses new requests while its ERA5 queue is
    full) is not a failure: the date is asked for again after a wait (1 min,
    doubling to 15), and queue_full counts the refusals. On 22-23 Oct 2026
    the CDS refused most requests for minutes or hours at a time, and every
    refused date was dropped: 265 of 270 'failed'.

    The record is rewritten as each fetch starts and ends (status
    'running'), so a prefetch that stalls can be read while it stalls: on
    22 Oct 2026 one ran an hour, 265 of 270 requests refused, and its record
    was never written because it never finished.

    Refused access (no token, token rejected, licence not accepted - the
    'fail' level of check_weather_credentials, or a 401/403/licence answer
    to the first requests) stops it at once with the plain reason, as a
    'failed' era5_prefetch progress event the dashboard shows hours before
    MintPy would meet it. Anything it cannot settle (tile, hour) is
    'skipped': MintPy then fetches what it needs itself, as before.

    The fetches run in daemon threads: a CDS request can wait in the queue
    for hours, and the ThreadPoolExecutor this used to run on is joined at
    interpreter exit, which held a finished run's process alive (22 Oct
    2026). Cancelled, it stops starting fetches, records 'cancelled' with
    what is still in flight and returns without waiting for them; a request
    already queued at the CDS stays queued there.
    """
    import threading
    from . import progress

    status = {"status": "skipped", "reason": "", "needed": 0, "present_before": 0, "fetched": 0,
              "failed": {}, "reasons": {}, "queue_full": 0, "in_flight": {}, "waiting": 0,
              "started": time.strftime("%Y-%m-%dT%H:%M:%S")}
    lock = threading.Lock()
    closed = threading.Event()             # finished: late answers are not recorded

    def write_status(quiet: bool = True) -> None:
        out = cfg.mintpy_dir / ERA5_PREFETCH_STATUS
        try:
            cfg.mintpy_dir.mkdir(parents=True, exist_ok=True)
            tmp = out.with_name(out.name + ".tmp")
            tmp.write_text(json.dumps(status, indent=2))
            os.replace(tmp, out)
        except OSError as exc:
            (log.debug if quiet else log.warning)("could not write %s: %s", ERA5_PREFETCH_STATUS, exc)

    def finish(state: str, reason: str = "") -> dict:
        with lock:
            closed.set()
            status.update(status=state, reason=reason, finished=time.strftime("%Y-%m-%dT%H:%M:%S"))
            write_status(quiet=False)
        ev = {"skipped": "skipped", "done": "done"}.get(state, "failed")
        if state == "cancelled" and not status["failed"]:
            ev = "skipped"
        progress.emit(cfg.project_dir, "era5_prefetch", ev, done=status["present_before"] + status["fetched"],
                      total=status["needed"] or None, msg=reason)
        (log.warning if ev == "failed" else log.info)("ERA5 prefetch %s%s", state, f": {reason}" if reason else "")
        return status

    if getattr(cfg, "mintpy_tropo_method", "no") != "pyaps":
        return finish("skipped", f"mintpy_tropo_method is {getattr(cfg, 'mintpy_tropo_method', 'no')!r}, not 'pyaps'")
    probs = check_weather_credentials(cfg)
    hard = [p for p in probs if (p.get("level") if isinstance(p, dict) else "fail") == "fail"]
    if hard:
        return finish("failed", "ERA5 access refused before any download: " + str(hard[0]))
    plan = era5_prefetch_plan(cfg)
    status.update({k: plan.get(k) for k in ("snwe", "hour", "dir")})
    status["needed"], status["present_before"] = plan["needed"], plan["present"]
    if not plan["ok"]:
        return finish("skipped", plan["reason"] + "; MintPy will fetch the ERA5 files itself")
    todo = list(reversed(plan["missing"]))      # newest first: meets MintPy's oldest-first loop in the middle
    if not todo:
        return finish("done", f"all {plan['needed']} ERA5 files already present")
    Path(plan["dir"]).mkdir(parents=True, exist_ok=True)
    fetch = fetch or fetch_era5_grib
    snwe, hour = tuple(plan["snwe"]), plan["hour"]
    cancel = cancel or threading.Event()
    stop = threading.Event()
    queue = list(todo)
    status["status"], status["waiting"] = "running", len(queue)
    write_status()
    progress.emit(cfg.project_dir, "era5_prefetch", "start", done=plan["present"], total=plan["needed"],
                  msg=f"{len(todo)} ERA5 files to fetch ({era5_area_str(snwe)[1:]}, {hour}:00 UTC)")

    def worker() -> None:
        while not (stop.is_set() or cancel.is_set()):
            with lock:
                if closed.is_set() or not queue:
                    return
                date = queue.pop(0)
                status["in_flight"][date] = time.strftime("%Y-%m-%dT%H:%M:%S")
                status["waiting"] = len(queue)
                write_status()
            wait_s = ERA5_BUSY_WAIT_S[0]
            while True:
                res = fetch(date, plan["files"][date], snwe, hour)
                if not (res.startswith("FAILED") and _ERA5_BUSY_RE.search(res)):
                    break
                with lock:
                    if closed.is_set():
                        return
                    status["queue_full"] += 1
                    if status["queue_full"] == 1:
                        log.warning("ERA5 prefetch: the CDS queue is full (%s); waiting and asking again, "
                                    "not counting it as a failure", _era5_reason(res[7:]))
                    msg = (f"CDS queue full: {status['queue_full']} requests turned away so far; "
                           f"asking again for {date} in {wait_s / 60:.0f} min")
                    write_status()
                    progress.emit(cfg.project_dir, "era5_prefetch", "running",
                                  done=plan["present"] + status["fetched"], total=plan["needed"], msg=msg)
                if cancel.wait(wait_s * random.uniform(0.8, 1.2)):
                    with lock:                     # back on the list: 'not requested' in the record
                        status["in_flight"].pop(date, None)
                        queue.insert(0, date)
                    return
                wait_s = min(2 * wait_s, ERA5_BUSY_WAIT_S[1])
            with lock:
                if closed.is_set():
                    return
                del status["in_flight"][date]
                msg = ""
                if res.startswith("FAILED"):
                    why = _era5_reason(res[7:])
                    status["failed"][date] = why
                    n = status["reasons"].get(why, 0)
                    status["reasons"][why] = n + 1
                    if n == 0:
                        log.warning("ERA5 prefetch: %s failed: %s (repeats are counted in %s)",
                                    date, why, ERA5_PREFETCH_STATUS)
                    if _ERA5_AUTH_RE.search(why):
                        stop.set()
                    msg = f"{len(status['failed'])} failed; last: {why[:160]}"
                else:
                    status["fetched"] += 1
                    if status["failed"]:
                        msg = f"{len(status['failed'])} failed"
                write_status()
                progress.emit(cfg.project_dir, "era5_prefetch", "running",
                              done=plan["present"] + status["fetched"], total=plan["needed"], msg=msg)

    n_workers = max(1, int(workers or ERA5_PREFETCH_WORKERS))
    threads = [threading.Thread(target=worker, name=f"era5-fetch-{i + 1}", daemon=True) for i in range(n_workers)]
    for t in threads:
        t.start()
    for t in threads:
        while t.is_alive() and not cancel.is_set():
            t.join(0.5)
    with lock:
        left_in_flight, left_waiting = len(status["in_flight"]), len(queue)
        if cancel.is_set() and (left_in_flight or left_waiting):
            closed.set()                   # the counts below are the record's
    if closed.is_set():
        return finish("cancelled", f"the run ended before the prefetch finished: {status['fetched']} fetched, "
                                   f"{len(status['failed'])} failed, {left_in_flight} still in progress, "
                                   f"{left_waiting} not requested; MintPy fetches what is missing itself")
    if stop.is_set():
        first = next(v for v in status["failed"].values() if _ERA5_AUTH_RE.search(v))
        return finish("failed", f"the CDS refused ERA5 access ({first[:200]}). Check the token and the "
                                "licence (insar doctor); MintPy's tropospheric step will fail the same way")
    if status["failed"]:
        return finish("failed", f"{len(status['failed'])} of {len(todo)} ERA5 files failed (first: "
                                f"{next(iter(status['failed'].items()))}); MintPy retries them itself")
    return finish("done", f"{status['fetched']} ERA5 files fetched, {plan['present']} already present")


# Ground area of one IW burst: Site B's merged 3-burst IW2 frame is
# 3,128,436 multilooked pixels of ~40.7 m x 41.7 m (11 x 3 looks) = 5,310
# km2, i.e. ~1,770 km2 (about 20.5 km along track x 86 km across) per burst.
BURST_GROUND_KM2 = 1770.0
# Sentinel-1 heading relative to north at mid latitudes (resolution.py): the
# radar crop window is the radar-aligned rectangle ENCLOSING the bbox, which
# is larger than the bbox (Site B: 1,237 km2 of crop for a 913 km2 box).
S1_HEADING_DEG = 13.0
# Products-stage memory. The deliverables no longer hold the geocoded
# time-series cube (deliverables._Cube streams it: epochs in chunk-aligned
# blocks, cells in row bands of at most 256 MB), so the peak grows with the
# map's CELLS, not cells x dates. MEASURED 5 Oct 2026 on a synthetic 0.7 M-cell,
# 100-date project (deliverables' cube readers + the point products): 547 MB
# above the imported libraries streamed, 785 MB with the cube in memory, i.e.
# ~780 bytes per cell (the point table and its GeoPackage are most of it).
# Plus ~95 bytes per cell for significance.assess (measured 0.30 / 0.63 / 1.15
# GB at 2.2 / 6.2 / 12.2 M cells), one row band, the parquet writer's
# 100,000-point block and ~1.5 GB of interpreter and libraries.
STREAMED_BYTES_PER_CELL = 800.0
ASSESS_BYTES_PER_CELL = 95.0
BAND_GB = 0.26
POINT_BLOCK = 100_000
DELIVERABLES_BASE_GB = 1.5


def products_ram_gb(cells: int, n_dates: int) -> float:
    """The products stage's peak memory (GB) for a map of `cells` cells and
    `n_dates` dates (the rule above)."""
    return (DELIVERABLES_BASE_GB + cells * (STREAMED_BYTES_PER_CELL + ASSESS_BYTES_PER_CELL) / 1e9
            + BAND_GB + 2 * POINT_BLOCK * max(0, int(n_dates)) * 4 / 1e9)


def _box_km(w: float, s: float, e: float, n: float) -> tuple:
    """(east-west, north-south) size of a lon/lat box in km."""
    import math
    lat = math.radians((s + n) / 2.0)
    return abs(e - w) * 111.32 * math.cos(lat), abs(n - s) * 110.9


def _bursts_per_date(cfg: PipelineConfig, n_sw: int) -> tuple:
    """(bursts per date, where the figure comes from). The download's own
    record first (slc/assembly_status.json, then dropped_short_dates.json:
    the modal burst count over every catalogue date); then ONE burst query
    over the first ~5 weeks of the record (the full search pages through
    every date and retries, minutes the planner should not spend); then 3
    per sub-swath, the Site A/Site B frame, said as an assumption."""
    import statistics
    from collections import Counter
    for name in ("assembly_status.json", "dropped_short_dates.json"):
        rec = cfg.slc_dir / name
        try:
            n = int(json.loads(rec.read_text()).get("modal_burst_count") or 0)
        except (OSError, ValueError, TypeError, AttributeError):
            continue
        if n > 0:
            return n, f"measured, slc/{name}"
    try:
        import datetime as _dt
        import asf_search as asf
        swaths = {f"IW{x}" for x in str(getattr(cfg, "stack_swaths", "2")).split()}
        pol = str(getattr(cfg, "stack_polarization", "vv")).upper()
        want = str(getattr(cfg, "orbit_direction", "ascending")).upper()[:3]
        t0 = _dt.date.fromisoformat(str(cfg.start_date)[:10])
        results = asf.search(dataset=asf.DATASET.SLC_BURST, intersectsWith=cfg.aoi_wkt,
                             start=str(t0), end=str(t0 + _dt.timedelta(days=36)),
                             relativeOrbit=cfg.track, polarization=[pol])
        per_date = Counter()
        for r in results:
            pr = r.properties
            if pr.get("flightDirection", "").upper()[:3] != want or pr["burst"]["subswath"] not in swaths:
                continue
            per_date[pr["startTime"][:10]] += 1
        if per_date:
            counts = sorted(per_date.values())
            modes = statistics.multimode(counts)
            return int(max(modes)), f"burst catalogue, {len(per_date)} dates from {t0}"
    except Exception as exc:                                  # noqa: BLE001
        log.info("Burst count query failed (%s); assuming 3 per sub-swath", exc)
    return 3 * n_sw, "assumed 3 per sub-swath (no catalogue record or query)"


def _crop_fraction(cfg: PipelineConfig, bursts_per_date: int) -> tuple:
    """(phase-linking crop as a fraction of the processed frame, source).
    MEASURED once the crop exists (stack/crop_window.json, written after the
    geometry: Site B 0.2331); before that, ESTIMATED as the radar-aligned
    rectangle enclosing bbox + stack_bbox_margin_deg over bursts x ~1,770
    km2 (Site B: 0.24 estimated, 0.233 measured)."""
    import math
    rec = cfg.stack_dir / "crop_window.json"
    try:
        f = float(json.loads(rec.read_text())["crop_fraction"])
        if 0 < f <= 1:
            return f, "measured, stack/crop_window.json"
    except (OSError, ValueError, KeyError, TypeError):
        pass
    m = float(getattr(cfg, "stack_bbox_margin_deg", 0.05) or 0.0)
    w, s_, e, n = (float(v) for v in cfg.bbox)
    ew, ns = _box_km(w - m, s_ - m, e + m, n + m)
    th = math.radians(S1_HEADING_DEG)
    crop_km2 = (ew * math.cos(th) + ns * math.sin(th)) * (ns * math.cos(th) + ew * math.sin(th))
    frame_km2 = max(1, bursts_per_date) * BURST_GROUND_KM2
    f = min(1.0, max(0.02, crop_km2 / frame_km2))
    return round(f, 4), f"estimated, ~{crop_km2:.0f} of ~{frame_km2:.0f} km2"


def _disk_budget(path) -> tuple:
    """(GB, source): settings.disk_budget_gb (on WSL C: free + what the vhdx
    holds on C: that ext4 does not use),
    else the filesystem's own free space."""
    try:
        from . import settings
        d = settings.disk_budget_gb(path)
        return round(float(d["budget"]), 1), str(d.get("source") or "settings.disk_budget_gb")
    except Exception as exc:                                  # noqa: BLE001
        log.debug("settings.disk_budget_gb unavailable (%s)", exc)
    p = Path(path)
    while not p.exists() and p != p.parent:
        p = p.parent
    return round(shutil.disk_usage(p).free / 1e9, 1), "filesystem free space"


def _allocated_gb(path) -> float:
    """GB (1e9) allocated under `path` (st_blocks, so sparse files count what
    they hold); 0 when it does not exist."""
    tot, todo = 0, [str(path)]
    while todo:
        d = todo.pop()
        try:
            it = os.scandir(d)
        except OSError:
            continue
        with it:
            for e in it:
                try:
                    if e.is_dir(follow_symlinks=False):
                        todo.append(e.path)
                    else:
                        tot += e.stat(follow_symlinks=False).st_blocks * 512
                except OSError:
                    pass
    return tot / 1e9


def _geo_grid(cfg: PipelineConfig) -> tuple:
    """(cells, extent text) of the geocoded grid the deliverables read: bbox
    + stack_bbox_margin_deg when the products are cropped (phase-linked
    route, subset_to_aoi), else the processed frame; posting from
    mintpy_geocode_lalo_step (Site B: 0.000135 x 0.000172 deg = 15 m,
    1882 x 1854 cells over a crop 0.254 x 0.319 deg)."""
    m = float(getattr(cfg, "stack_bbox_margin_deg", 0.05) or 0.0)
    if cfg.backend == "miaplpy" or getattr(cfg, "subset_to_aoi", False):
        w, s_, e, n = (float(v) for v in cfg.bbox)
        w, s_, e, n, what = w - m, s_ - m, e + m, n + m, "the area box"
    else:
        try:
            from . import data_acquisition as da
            w, s_, e, n = da.frame_bbox(cfg)
            what = "the processed frame"
        except Exception:                                     # noqa: BLE001
            w, s_, e, n = (float(v) for v in cfg.bbox)
            w, s_, e, n, what = w - 0.6, s_ - 0.6, e + 0.6, n + 0.6, "the area box + 0.6 deg"
    try:
        dlat, dlon = (abs(float(v)) for v in str(cfg.mintpy_geocode_lalo_step).split(","))
    except (ValueError, AttributeError):
        dlat, dlon = 0.000135, 0.000172                      # MintPy 'auto' on this route: ~15 m
    return int(abs(n - s_) / dlat * abs(e - w) / dlon), what


# per date and per unit of crop fraction, per 3 bursts (Site B, see plan_stack)
PL_SLCSTACK_GB, PL_PATCHES_GB, PL_PHASE_SERIES_GB = 0.78, 0.84, 0.79


def check_phase_linking_disk(cfg: PipelineConfig, raise_on_short: bool = True) -> dict:
    """
    The phase-linking part of the disk plan again, once the crop is MEASURED
    (stack/crop_window.json) and before MiaplPy's load_data: the planner had
    only the bbox estimate when the run started, and a county-sized area that
    fills the disk does so ~20 h in, during phase linking or concatenation.
    Needs no network: dates from the plan (or the merged SLC folders),
    bursts from the download's record. Returns {need_gb, budget_gb,
    budget_source, crop_fraction, n_dates, bursts_per_date, ok, text};
    raises RuntimeError when short and raise_on_short. For
    miaplpy_backend.run, right before load_data.
    """
    n_sw = len(str(getattr(cfg, "stack_swaths", "2")).split())
    merged = cfg.stack_dir / "merged" / "SLC"
    dates = planned_dates(cfg) or ({q.name for q in merged.iterdir() if q.is_dir()} if merged.exists() else set())
    n = len(dates)
    bursts, _ = _bursts_per_date(cfg, n_sw) if (cfg.slc_dir / "assembly_status.json").exists() or \
        (cfg.slc_dir / "dropped_short_dates.json").exists() else (3 * n_sw, "assumed")
    f, fsrc = _crop_fraction(cfg, bursts)
    scale = bursts / 3.0
    slcstack = PL_SLCSTACK_GB * f * scale * n
    patches = PL_PATCHES_GB * f * scale * n
    series = PL_PHASE_SERIES_GB * f * scale * n
    # what is still to be written from here: the SLC stack, then the patches
    # and the series beside it (the merged SLCs are already on disk)
    have = 0.0
    work = cfg.project_dir / "miaplpy"
    if work.exists():
        have = _allocated_gb(work)
    # The merged secondary SLCs are deleted right after load_data when
    # miaplpy_delete_merged_slc_after_load (miaplpy_backend.run: load_data,
    # then the deletion, then phase_linking), so the peak is the larger of
    # (a) the SLC stack beside the merged SLCs and (b) everything written
    # once they are gone. Until 13 Oct 2026 the deletion was not credited and
    # the gate asked for slcstack + patches + series (Site C: 289 GB asked,
    # ~96 GB the real peak, 265 GB free - refused).
    freed = 0.0
    if getattr(cfg, "miaplpy_delete_merged_slc_after_load", False) and merged.exists() and n > 1:
        freed = _allocated_gb(merged) * (n - 1) / n          # the reference date's files stay
    total = slcstack + patches + series
    need = max(0.0, max(slcstack, total - freed) + 3.0 - have)
    budget, src = _disk_budget(cfg.project_dir.parent)
    ok = need <= budget
    text = (f"Phase linking still needs ~{need:.0f} GB at its peak (SLC stack ~{slcstack:.0f}, patches "
            f"~{patches:.0f}, phase series ~{series:.0f} GB for {n} dates at crop {f:.0%} [{fsrc}]"
            + (f"; ~{freed:.0f} GB of merged SLCs are deleted after loading" if freed >= 1 else "")
            + (f", {have:.0f} GB already written" if have >= 1 else "")
            + f"); the disk budget is {budget:.0f} GB ({src}).")
    out = {"need_gb": round(need, 1), "budget_gb": budget, "budget_source": src, "crop_fraction": f,
           "merged_freed_gb": round(freed, 1),
           "n_dates": n, "bursts_per_date": bursts, "ok": ok, "text": text}
    log.info("%s", text)
    if not ok and raise_on_short:
        raise RuntimeError(text + " Free space (or move the project) before phase linking: a disk that "
                                  "fills during concatenation leaves an interrupted series.")
    return out


def _plan_memory(cfg: PipelineConfig, plan: StackPlan) -> None:
    """RAM the deliverables stage will want, against this machine's RAM.
    Site B's geocoded cube is 3.79 GB (270 dates x 3.49 M cells x 4 B);
    held whole (before 5 Oct 2026) a box ~5x larger needed ~18 GB for ONE copy
    and the stage died of memory after 1-2 days of processing. Streamed, the
    peak follows the cell count (products_ram_gb): ~4.6 GB for Site B."""
    try:
        plan.geo_cells, what = _geo_grid(cfg)
    except Exception as exc:                                  # noqa: BLE001
        log.info("Memory estimate skipped (%s)", exc)
        return
    cube = plan.n_dates * plan.geo_cells * 4 / 1e9
    plan.est_ram_gb = round(products_ram_gb(plan.geo_cells, plan.n_dates), 1)
    try:
        from . import settings
        plan.mem_gb = float(settings.machine_resources()["mem_gb"])
    except Exception:                                         # noqa: BLE001
        plan.mem_gb = 0.0
    if not plan.mem_gb:
        return
    head = (f"MEMORY: the products stage streams the {cube:.1f} GB time-series cube "
            f"({plan.n_dates} dates x {plan.geo_cells / 1e6:.1f} M cells over {what}) and needs "
            f"~{plan.est_ram_gb:.0f} GB of this machine's {plan.mem_gb:.0f} GB RAM for its per-cell layers")
    if plan.est_ram_gb > 0.9 * plan.mem_gb:
        plan.notes.append(head + ". That WILL NOT FIT: the products stage would be killed after the "
                          "whole run. Shrink the area, coarsen mintpy_geocode_lalo_step, or shorten "
                          "the date range.")
    elif plan.est_ram_gb > 0.5 * plan.mem_gb:
        plan.notes.append(head + ", more than half of it. Run nothing else alongside it, or shrink "
                          "the area or the date range.")


def plan_stack(cfg: PipelineConfig) -> StackPlan:
    """
    Size a run before committing to it.

    Everything here is ARITHMETIC over the scene search and per-unit sizes
    measured at Site A; nothing is generated or downloaded. bbox
    subsetting selects whole bursts and crops in AZIMUTH ONLY - a burst is
    always the full sub-swath in range, so a 30 km AOI still yields a stack
    ~90 km wide - which is why the merged-product size is per pair rather
    than per area, and why unwrapping is cropped (stack_crop_unwrap).
    """
    from . import data_acquisition as da

    if str(getattr(cfg, "miaplpy_relink_from", "") or "").strip():
        # F3 hook (8 Oct 2026): a project re-linking another project's stack
        # downloads and coregisters nothing (miaplpy_backend.relink_plan)
        from . import miaplpy_backend as _mb
        return _mb.relink_plan(cfg)
    plan = StackPlan(swaths=cfg.stack_swaths, bbox_snwe=_bbox_snwe(cfg),
                     range_looks=cfg.stack_range_looks,
                     azimuth_looks=cfg.stack_azimuth_looks)
    results = da.search_scenes(cfg)
    dates = sorted({r.properties["startTime"][:10].replace("-", "")
                    for r in results})
    plan.dates = dates
    plan.n_dates = len(dates)
    # Download volume follows SCENES, not dates. A track normally needs two or
    # more SAFEs per acquisition to span the AOI along-track, so sizing off the
    # date count understates the download by exactly that factor - measured at
    # 2x on Site A track 19 (61 scenes for 31 dates).
    plan.n_scenes = len(results)
    n_sw = len(cfg.stack_swaths.split())
    # Bursts per date from the project's own catalogue record or one narrow
    # burst query; 3 per sub-swath only when neither is available. Every
    # per-unit size below was measured on a 3-burst, one-sub-swath frame, so
    # it scales by bursts / 3: an AOI 4 bursts long was sized 33% short, one
    # straddling two sub-swaths 2x short when stack_swaths was right.
    plan.bursts_per_date, plan.bursts_source = _bursts_per_date(cfg, n_sw)
    bursts_per_date = plan.bursts_per_date
    if getattr(cfg, "do_esd", True) and bursts_per_date <= n_sw:
        plan.notes.append(
            f"About {bursts_per_date} burst(s) per date over {n_sw} sub-swath(s): no burst overlaps, so ESD "
            "coregistration cannot run; the stack will be coregistered by geometry (coregistration_mode).")
    units = bursts_per_date / 3.0
    # stackSentinel pairs by time only: each date joins the next N.
    plan.n_pairs = max(0, cfg.stack_num_connections * plan.n_dates
                       - cfg.stack_num_connections * (cfg.stack_num_connections + 1) // 2)

    # Per-unit sizes MEASURED at Site A (21 Sep 2026, 1 sub-swath, 3 bursts,
    # 11x3 looks), per swath so a 3-swath run scales:
    #   coregistered date  1.90 GB = 0.86 SLC + 0.86 geo2rdr offsets + 0.18 overlap
    #   misreg pair        0.158 GB (ESD 0.069 + coarse igram 0.089)
    #   merged pair        0.111 GB, of which 0.050 unfiltered fine.int/.cor
    #   burst igram        0.79 GB transient until merged
    slc_gb, burst_gb = 4.5, 0.79 * units
    coreg_slc_gb, coreg_off_gb, coreg_ovl_gb = 0.86 * units, 0.86 * units, 0.18 * units
    misreg_gb, merged_gb, unfilt_gb = 0.158 * units, 0.111 * units, 0.050 * units
    looks_scale = (11 * 3) / max(1, cfg.stack_range_looks * cfg.stack_azimuth_looks)
    merged_gb *= looks_scale
    unfilt_gb *= looks_scale
    ovl_cleaned = getattr(cfg, "stack_cleanup_misreg_inputs", False)
    coreg_gb = coreg_slc_gb + (0.0 if ovl_cleaned else coreg_ovl_gb) + \
        (0.0 if getattr(cfg, "stack_cleanup_offsets", False) else coreg_off_gb)

    # Measured on Site A track 19: ~150 MB per burst.
    if getattr(cfg, "stack_use_burst2safe", False):
        # Sizing off DATES here, not scenes: assembling from bursts collapses
        # the several archives that covered one acquisition into a single SAFE.
        plan.est_slc_gb = round(0.15 * bursts_per_date * plan.n_dates, 1)
        plan.notes.append(
            f"Assembling from bursts: ~{bursts_per_date} bursts per date at "
            f"~150 MB, against ~{slc_gb:.1f} GB per full archive. That is "
            f"{plan.est_slc_gb:.1f} GB instead of "
            f"{slc_gb * max(plan.n_scenes, plan.n_dates):.0f} GB.")
    else:
        plan.est_slc_gb = round(slc_gb * max(plan.n_scenes, plan.n_dates), 1)
    n_sec = max(0, plan.n_dates - 1)
    plan.est_coreg_gb = round(coreg_gb * n_sec + 3.0, 1)
    plan.est_merged_gb = round(merged_gb * plan.n_pairs, 1)
    transient = burst_gb * min(plan.n_pairs, _workers_for(cfg, "merge") * 4)
    misreg_total = misreg_gb * plan.n_pairs
    keep_misreg = 0.0 if getattr(cfg, "stack_cleanup_misreg_inputs", False) else misreg_total
    # Three moments can be the peak. End of coregistration: every SAFE not yet
    # deleted, every date's coregistered products, the ESD inputs unless
    # cleaned. End of the merge: that plus any real merged SLCs. End of
    # filtering: coregistered SLCs (unless cleaned), real merged SLCs, every
    # merged pair, the burst-interferogram transient.
    safes_at_coreg_end = plan.est_slc_gb * (
        (cfg.stack_keep_recent_safes / max(1, plan.n_dates))
        if cfg.stack_delete_safes_after_coreg else 1.0)
    peak_coreg = safes_at_coreg_end + plan.est_coreg_gb + keep_misreg
    # A REAL merge on the stack route (stack_virtual_merge False) writes a
    # full-resolution SLC per date (0.84 GB per 3 bursts, measured at Sites B
    # and D) BESIDE its coregistered bursts: nothing swaps them on this route
    # (the per-date swap is the miaplpy merge hook). Left out, the Site D
    # full-record plan printed 303 GB for a ~475 GB merge (22 Oct 2026).
    real_merge_gb = 0.84 * units * plan.n_dates if (
        cfg.backend == "stack" and str(cfg.stack_virtual_merge).lower() == "false") else 0.0
    peak_merge = peak_coreg + real_merge_gb
    coreg_after_filter = plan.est_coreg_gb if not getattr(
        cfg, "stack_cleanup_coreg_after_filter", False) else 3.0
    # igram -> merge -> filter run as per-pair chains, so with coreg cleanup
    # on only a rolling window of ~(connections + workers) dates' SLCs is
    # alive during filtering; otherwise all of them are.
    coreg_during_filter = plan.est_coreg_gb if not getattr(
        cfg, "stack_cleanup_coreg_after_filter", False) else \
        coreg_gb * min(n_sec, cfg.stack_num_connections + _workers_for(cfg, "merge")) + 3.0
    peak_filter = (plan.est_slc_gb if not cfg.stack_delete_safes_after_coreg else 0.0) \
        + coreg_during_filter + plan.est_merged_gb + transient + keep_misreg + real_merge_gb
    steady = coreg_after_filter + keep_misreg + real_merge_gb + plan.n_pairs * (
        merged_gb - (unfilt_gb if getattr(cfg, "stack_cleanup_unfiltered_igrams", False) else 0.0))
    plan.est_peak_gb = round(max(peak_coreg, peak_merge, peak_filter), 1)
    plan.notes.append(
        f"Disk by moment: end of coregistration ~{peak_coreg:.0f} GB, "
        + (f"end of the merge ~{peak_merge:.0f} GB (real merged SLCs ~{real_merge_gb:.0f} GB beside the "
           f"coregistered bursts: stack_virtual_merge False), " if real_merge_gb else "")
        + f"end of filtering ~{peak_filter:.0f} GB, after cleanups ~{steady:.0f} GB. "
        f"Cleanups on: offsets={getattr(cfg, 'stack_cleanup_offsets', False)}, "
        f"misreg inputs={getattr(cfg, 'stack_cleanup_misreg_inputs', False)}, "
        f"coreg SLCs after filter={getattr(cfg, 'stack_cleanup_coreg_after_filter', False)}, "
        f"unfiltered igrams={getattr(cfg, 'stack_cleanup_unfiltered_igrams', False)}, "
        f"SAFEs after coreg={cfg.stack_delete_safes_after_coreg}.")
    if cfg.backend == "miaplpy":
        # MEASURED on Site B (28 Sep - 1 Oct 2026; 270 dates, 3 bursts, one
        # sub-swath, phase-linking crop 23.31% of the frame): merged real SLCs
        # 223.1 GB (0.83 GB per date, the whole frame); slcStack.h5 49 GB,
        # PATCHES 52.7 GB, phase_series 50 GB. The last three hold only the
        # CROP, so per date and per unit of crop fraction f they are 0.78 /
        # 0.84 / 0.79 GB. They were constants measured at a 34% crop and
        # scaled by bursts alone, which overstated Site B 1.6x and would
        # understate any AOI whose crop exceeds ~39% of the frame (f = 0.6 at
        # 275 dates: ~400 GB against a planned 259).
        scale = units                             # bursts / 3 (the measured frame had 3)
        plan.crop_fraction, plan.crop_source = _crop_fraction(cfg, bursts_per_date)
        f = plan.crop_fraction
        merged_slc = 0.84 * scale * plan.n_dates           # real merged SLC per date (whole frame)
        slcstack = PL_SLCSTACK_GB * f * scale * plan.n_dates         # MiaplPy slcStack.h5 (lzf gains nothing on SLC noise)
        patches = PL_PATCHES_GB * f * scale * plan.n_dates          # inverted/PATCHES
        phase_series = PL_PHASE_SERIES_GB * f * scale * plan.n_dates
        # exported + unwrapped single-reference pairs: ~0.05 GB per date at
        # the 34% crop of the 2023 test (not re-measured), so scaled from there
        star = 0.05 * (f / 0.34) * scale * plan.n_dates
        during_load = merged_slc + slcstack + 3.0                       # both exist until the stack is verified
        after_load = (0.0 if cfg.miaplpy_delete_merged_slc_after_load else merged_slc) + slcstack + 3.0
        during_pl = after_load + patches + phase_series                 # concatenation writes the series beside the patches
        after_concat = after_load + phase_series + star \
            + (0.0 if cfg.miaplpy_delete_patches_after_concatenate else patches) \
            - (slcstack if cfg.miaplpy_delete_slc_stack_after_concatenate else 0.0)
        # the coregistration peak stands: the merge swaps each date's bursts for its merged copy
        pl_peak = max(peak_coreg + 3.0, during_load, during_pl)
        plan.est_merged_gb = round(merged_slc, 1)
        plan.est_peak_gb = round(pl_peak, 1)
        plan.notes.append(
            f"Phase-linked route (crop {f:.0%} of the frame, {plan.crop_source}): merged real SLCs ~{merged_slc:.0f} GB are written while "
            f"the bursts are removed date by date (~{peak_coreg:.0f} GB); loading MiaplPy's SLC stack (~{slcstack:.0f} GB) "
            f"beside them is the peak, ~{during_load:.0f} GB; merged SLCs {'deleted' if cfg.miaplpy_delete_merged_slc_after_load else 'KEPT'} "
            f"once it is verified; phase linking + concatenation ~{during_pl:.0f} GB (patches ~{patches:.0f} GB, "
            f"{'deleted' if cfg.miaplpy_delete_patches_after_concatenate else 'KEPT'} afterwards; SLC stack "
            f"{'deleted' if cfg.miaplpy_delete_slc_stack_after_concatenate else 'KEPT'}); "
            f"after concatenation ~{after_concat:.0f} GB. Projected peak ~{pl_peak:.0f} GB. No per-pair merged products.")
    # The real budget, in GB (1e9) like every other figure on screen. ext4's
    # free space (in GiB) was 692 against a real ~424 GB on 5 Oct 2026: the
    # rest of a WSL vhdx can only grow into what C: still has free.
    plan.free_gb, plan.budget_source = _disk_budget(cfg.project_dir.parent)
    plan.project_gb = round(_allocated_gb(cfg.project_dir), 1)

    if cfg.stack_delete_safes_after_coreg:
        steady = plan.est_peak_gb - plan.est_slc_gb + (
            cfg.stack_keep_recent_safes * 4.5)
        plan.notes.append(
            f"SAFE cleanup is on: steady-state settles near {steady:.0f} GB "
            f"once coregistration completes. The {plan.est_peak_gb:.0f} GB peak "
            "still occurs, because every scene is downloaded before processing "
            "begins - interleaving the two is what a longer run will need.")

    # What is already in the project is part of the peak (a resumed
    # Site B holds ~205 of its ~311 GB), so a resume needs only the rest.
    need = max(0.0, plan.est_peak_gb - plan.project_gb)
    if need > plan.free_gb:
        plan.notes.append(
            f"PROJECTED PEAK {plan.est_peak_gb} GB EXCEEDS THE DISK BUDGET: "
            f"{need:.0f} GB more is needed"
            + (f" ({plan.project_gb:.0f} GB is already in the project)" if plan.project_gb >= 1 else "")
            + f", {plan.free_gb} GB can be written ({plan.budget_source}). "
            "Free space, shorten the date range or the area, or download SLCs in "
            "batches and delete each SAFE once its date is coregistered, keeping a "
            f"rolling window of about {2 * cfg.stack_num_overlap_connections} for NESD.")
    _plan_memory(cfg, plan)
    if plan.n_dates and plan.n_dates < 10:
        plan.notes.append("Fewer than 10 dates: fine for a smoke test, too few "
                          "for a velocity.")
    if n_sw == 1:
        plan.notes.append(
            "Single sub-swath. Each IW swath is ~80 km wide; confirm the AOI "
            "does not straddle a boundary, or half of it is silently dropped.")
    # A reference point outside the processed extent is silent: MintPy picks
    # its own pixel instead, every displacement is then relative to something
    # nobody argued was stable, and nothing in the logs says so.
    if cfg.reference_lalo.strip() and cfg.subset_to_aoi:
        try:
            rlat, rlon = (float(v) for v in cfg.reference_lalo.split(","))
            lo, la, LO, LA = cfg.bbox
            if not (la <= rlat <= LA and lo <= rlon <= LO):
                plan.notes.append(
                    f"REFERENCE POINT {rlat},{rlon} FALLS OUTSIDE THE AOI "
                    f"({la}..{LA} lat, {lo}..{LO} lon) AND subset_to_aoi IS ON, "
                    "so it will be cropped out and MintPy will choose its own "
                    "reference instead - silently. Move the point inside, widen "
                    "the bbox, or set subset_to_aoi false.")
        except ValueError:
            plan.notes.append(f"reference_lalo {cfg.reference_lalo!r} is not "
                              "'lat,lon' and will be ignored by MintPy.")

    lag = era5_lag_note(cfg)
    if lag:
        plan.notes.append(lag)
    for problem in check_weather_credentials(cfg):
        level = problem.get("level") if isinstance(problem, dict) else "fail"
        plan.notes.append(("ERA5 (could not verify, not blocking): " if level == "warn"
                           else "ERA5: ") + str(problem))

    if cfg.stack_unw_full_cost:
        plan.notes.append(
            "Full-cost snaphu is enabled (nomcf True). Expect 5-10x the "
            "unwrapping time of the ISCE default, which is the trade this "
            "backend exists to make.")
    # F1 hook (8 Oct 2026): can these intervals follow the expected tilt?
    # Only when the config asks (aliasing.plan_note returns None otherwise).
    from . import aliasing as _al
    note = _al.plan_note(cfg, plan.dates)
    if note:
        plan.notes.append(note)
    return plan


# ERA5 is published about 5 days behind real time (CDS ERA5 "latency"); one
# day of margin for the CDS queue.
ERA5_LAG_DAYS = 6


def era5_lag_note(cfg: PipelineConfig, today=None) -> str | None:
    """A plan note when the record ends inside ERA5's publication lag: the
    run would stop at the troposphere step with those dates uncorrected
    (corrections.verify_era5_coverage), a day or two into the run instead of
    here. None when ERA5 is not used or the end date is old enough."""
    import datetime as _dt
    if str(cfg.mintpy_tropo_method).lower() != "pyaps" or cfg.era5_allow_partial:
        return None
    try:
        end = _dt.date.fromisoformat(str(cfg.end_date)[:10])
    except ValueError:
        return None
    today = today or _dt.date.today()
    newest = today - _dt.timedelta(days=ERA5_LAG_DAYS)
    if end <= newest:
        return None
    return (f"ERA5: the record ends {end.isoformat()}, inside ERA5's ~5-day publication lag; images after "
            f"about {newest.isoformat()} will have no weather-model delay yet and the run will STOP at the "
            f"troposphere step naming them. Set end_date to {newest.isoformat()} or earlier, wait a few days "
            "before the MintPy stage, or set era5_allow_partial: true to deliver them uncorrected.")


def format_plan(plan: StackPlan) -> str:
    lines = [
        "topsStack run plan",
        f"  dates            {plan.n_dates}"
        + (f"   ({plan.dates[0]} to {plan.dates[-1]})" if plan.dates else ""),
        f"  SLC scenes       {plan.n_scenes}"
        + (f"   ({plan.n_scenes / plan.n_dates:.1f} per date)" if plan.n_dates else ""),
        f"  pairs            {plan.n_pairs}",
        f"  sub-swaths       {plan.swaths}",
        f"  bbox (S N W E)   {plan.bbox_snwe}",
        f"  looks            {plan.range_looks} range x {plan.azimuth_looks} azimuth",
        "",
        f"  SLC downloads    {plan.est_slc_gb:>8.1f} GB",
        f"  coregistered     {plan.est_coreg_gb:>8.1f} GB",
        f"  merged products  {plan.est_merged_gb:>8.1f} GB",
        f"  projected peak   {plan.est_peak_gb:>8.1f} GB"
        + (f"   ({plan.bursts_per_date} bursts per date: {plan.bursts_source})" if plan.bursts_per_date else ""),
        f"  disk budget      {plan.free_gb:>8.1f} GB"
        + (f"   ({plan.budget_source})" if plan.budget_source else ""),
    ]
    if plan.project_gb >= 1:
        lines.append(f"  already in proj. {plan.project_gb:>8.1f} GB")
    if plan.est_ram_gb:
        lines.append(f"  products RAM     {plan.est_ram_gb:>8.1f} GB"
                     + (f"   of {plan.mem_gb:.0f} GB RAM" if plan.mem_gb else ""))
    for n in plan.notes:
        lines += ["", f"  ! {n}"]
    return "\n".join(lines)


# ----------------------------------------------------------------------
# MintPy handoff
# ----------------------------------------------------------------------
def _unwrap_error_method(cfg: PipelineConfig) -> str:
    """
    Drop phase_closure when the network has no closed loops.

    Phase-closure unwrap-error correction works on TRIPLETS: it sums phase
    around a closed loop of three interferograms and attributes any non-zero
    residue to an integer cycle miscount. With num_connections = 1 the network
    is a simple chain, there are no triplets, and MintPy does not detect this -
    get_common_region_int_ambiguity returns None and the run dies on
    "'NoneType' object has no attribute 'astype'", which says nothing about
    the real cause.

    Bridging needs no loops (it reconnects components across gaps using the
    connected-component labels), so it still applies. Two connections give the
    first triplets; production uses three.
    """
    method = cfg.mintpy_unwrap_error_method
    if cfg.stack_num_connections < 2 and "phase_closure" in method:
        reduced = "+".join(m for m in method.split("+") if m != "phase_closure")
        reduced = reduced or "bridging"
        log.warning("num_connections = %d gives a chain with no closed loops, so "
                    "phase closure has no triplets to work on. Falling back to "
                    "'%s'. Use num_connections >= 2 for phase-closure correction.",
                    cfg.stack_num_connections, reduced)
        return reduced
    return method


def _mintpy_compute_lines(cfg: PipelineConfig) -> list:
    """MintPy's dask settings, only when asked for (mintpy_num_workers > 0),
    clamped at run time to what THIS machine holds
    (settings.sized_parallelism): a YAML copied from the 23 GB / 24-core
    machine carries 16 workers and 16 GB, which a 16 GB laptop's WSL (~7.8 GB)
    cannot give (generality-13). The clamp is logged; it changes speed only."""
    n = int(getattr(cfg, "mintpy_num_workers", 0) or 0)
    if n <= 0:
        return []
    mem = float(getattr(cfg, "mintpy_max_memory_gb", 16.0) or 16.0)
    try:
        from .settings import machine_resources, sized_parallelism
        res = machine_resources()
        cap = sized_parallelism(res["mem_gb"], res["cores"])
        n2, mem2 = min(n, int(cap["mintpy_num_workers"])), min(mem, float(cap["mintpy_max_memory_gb"]))
        if (n2, mem2) != (n, mem):
            log.warning("MintPy: %d workers / %.0f GB configured, but this machine (%.1f GB, %d cores) holds %d / "
                        "%.0f GB - using those", n, mem, res["mem_gb"], res["cores"], n2, mem2)
        n, mem = n2, mem2
    except Exception as exc:                                  # noqa: BLE001 - no reading, no clamp
        log.info("MintPy workers not clamped to this machine (%s)", exc)
    return [
        "mintpy.compute.cluster                = local",
        f"mintpy.compute.numWorker              = {n}",
        f"mintpy.compute.maxMemory              = {mem}",
    ]


def mintpy_loader_lines(cfg: PipelineConfig) -> str:
    """
    The topsStack loader block for MintPy.

    Unlike the topsApp path, nothing here needs rescuing: topsStack writes
    baselines/<pair>/<pair>.txt in exactly the format MintPy's
    isce_utils.read_tops_baseline parses, reference/IW*.xml is the intended
    metaFile, and geom_reference/ supplies the lookup tables. The whole
    bperp-scraping layer in time_series.py is unnecessary on this path.

    connCompFile is the one that matters most: it is what makes MintPy's
    bridging unwrap-error correction possible at all, and the topsApp path
    never produced it.
    """
    from . import corrections as _corrections
    s = cfg.stack_dir
    geom = s / "merged" / "geom_reference"
    ifg = s / "merged" / "interferograms"
    # F3 hook (8 Oct 2026): the pairs MintPy is about to load carry the fast
    # zones the config sets now ('MintPy onward' runs neither unwrap hook);
    # nothing happens without fast_zone_geojson or earlier zones
    from . import aliasing as _al
    _al.ensure_fast_zones(cfg)
    return "\n".join([
        "mintpy.load.processor      = isce",
        f"mintpy.load.metaFile       = {s}/reference/IW*.xml",
        f"mintpy.load.baselineDir    = {s}/baselines",
        f"mintpy.load.unwFile        = {ifg}/*/filt_fine.unw",
        f"mintpy.load.corFile        = {ifg}/*/filt_fine.cor",
        f"mintpy.load.connCompFile   = {ifg}/*/filt_fine.unw.conncomp",
        f"mintpy.load.demFile        = {geom}/hgt.rdr",
        f"mintpy.load.lookupYFile    = {geom}/lat.rdr",
        f"mintpy.load.lookupXFile    = {geom}/lon.rdr",
        f"mintpy.load.incAngleFile   = {geom}/los.rdr",
        f"mintpy.load.azAngleFile    = {geom}/los.rdr",
        f"mintpy.load.shadowMaskFile = {geom}/shadowMask.rdr",
        "",
        # NO hard per-pair coherence mask in the inversion. MEASURED at
        # Site A: with maskDataset = coherence at 0.4, 91.8% of pixels were
        # inverted from ZERO pairs - summer interferograms are 86-93% below
        # 0.4, masking them per pixel disconnects the per-pixel network, and
        # MintPy then skips the pixel outright. Coverage collapsed to 8.2% of
        # the AOI and 2.6% of the mine.
        #
        # weightFunc = var already down-weights low-coherence observations
        # smoothly, without cutting the network. Quality is then judged AFTER
        # inversion by temporal coherence - MintPy's own default approach.
        "mintpy.networkInversion.maskDataset   = no",
        "mintpy.networkInversion.weightFunc    = var",
        # A phase-linked stack is a single-reference star by construction:
        # every pair touches the reference date, so a temporal-baseline cap
        # would simply delete the dates far from it (MEASURED on the 2023 test:
        # 120 d dropped the four summer dates; on a 10-year stack it would
        # drop nearly all of them). No baseline limits on that route.
        f"mintpy.network.perpBaseMax            = {'no' if cfg.backend == 'miaplpy' else cfg.max_perp_baseline}",
        f"mintpy.network.tempBaseMax            = {'no' if cfg.backend == 'miaplpy' else cfg.max_temp_baseline}",
        "mintpy.network.keepMinSpanTree        = yes",
        f"mintpy.unwrapError.method             = {_unwrap_error_method(cfg)}",
    ] + _corrections.mintpy_tropo_lines(cfg, 38) + [
        f"mintpy.deramp                         = {cfg.mintpy_deramp}",
        # mintpy.deramp.maskFile is NOT written here. It is appended by
        # time_series.run_mintpy_with_loader between the inversion and the
        # deramp, once the mask can be built from maskTempCoh.h5 (or taken
        # from cfg.mintpy_deramp_mask). One writer, one line.
    ] + _mintpy_compute_lines(cfg) + [
        f"mintpy.geocode.laloStep               = {cfg.mintpy_geocode_lalo_step}",
        "",          # MUST end with a newline: the caller concatenates more
                     # settings directly onto this block, and without it the
                     # last line here and the first line there merge into one
                     # unparseable key. That silently broke geocoding.
    ])


# ----------------------------------------------------------------------
# deramp mask
# ----------------------------------------------------------------------
# The fit ground must SURROUND the area: kept pixels in at least this many of
# the four quadrants around the AOI centre, each holding at least this share
# of them. A strip on one side passes any count-based test, and the plane
# fitted on it is then extrapolated across the mine and subtracted there.
DERAMP_MIN_QUADRANTS = 3
DERAMP_QUADRANT_MIN_SHARE = 0.05


def _geojson_overlaps_bbox(path: Path, bbox) -> bool:
    """Whether a GeoJSON's coordinates (lon, lat) reach into bbox (W, S, E, N);
    False for an unreadable file or one with no coordinates."""
    try:
        gj = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return False
    xs, ys = [], []

    def walk(o):
        if isinstance(o, dict):
            for k, v in o.items():
                if k == "coordinates":
                    flat(v)
                elif isinstance(v, (dict, list)):
                    walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)

    def flat(c):
        if c and isinstance(c[0], (int, float)):
            xs.append(float(c[0])); ys.append(float(c[1]))
        else:
            for v in c:
                flat(v)
    walk(gj)
    if not xs:
        return False
    w, s_, e, n = (float(v) for v in bbox)
    return not (max(xs) < w or min(xs) > e or max(ys) < s_ or min(ys) > n)


def build_deramp_mask(cfg: PipelineConfig, exclude_geojson=None,
                      buffer_m: float | None = None, source_mask: str = "maskTempCoh.h5",
                      out_name: str = "maskDeramp.h5") -> Path | None:
    """
    A ramp-fitting mask with the deforming area cut out.

    WHY THIS IS NOT OPTIONAL. MintPy's mintpy.deramp.maskFile defaults to
    maskTempCoh.h5 - every coherent pixel, INCLUDING the subsidence bowl. A
    plane fitted through real deformation is pulled toward it, and the biased
    plane is then subtracted from the bowl, shrinking the signal by an amount
    nobody can see afterwards because the residual still looks flat.

    The run therefore refuses to deramp without an exclusion polygon (this
    function raises), and provenance records how many pixels the fit used so
    the exclusion can be checked in the product.

    The buffer is a true (Euclidean) distance, buffer_m or else
    cfg.deramp_buffer_m - the far field's definition (significance.
    euclidean_buffer). It was a binary_dilation with the default cross
    element: an L1 diamond reaching only 71% of the width on diagonals
    (measured on Site B: kept pixels 1,432 m from the mapped mines, 4% of
    the fit within the stated 2 km).

    It also REFUSES (RuntimeError) when the kept pixels do not surround the
    area of interest (DERAMP_MIN_QUADRANTS of the four quadrants around the
    bbox centre, each with DERAMP_QUADRANT_MIN_SHARE of the fit), because a
    plane fitted on one side and subtracted across the mine tilts every rate
    inside it - and that used to leave only a warning in a multi-hour log.

    Returns the written path, or None when the source mask does not exist yet
    - which is the normal case before MintPy's inversion has run, and is why
    this is called BETWEEN MintPy phases rather than up front.
    """
    import h5py
    import numpy as np

    if buffer_m is None:
        buffer_m = cfg.deramp_buffer_m
        buffer_m = 2000.0 if buffer_m is None else float(buffer_m)

    src = cfg.mintpy_dir / source_mask
    if not src.exists():
        alt = cfg.mintpy_dir / "geo" / f"geo_{source_mask}"
        if not alt.exists():
            log.info("%s not present yet - deramp mask deferred until after "
                     "the network inversion", source_mask)
            return None
        src = alt

    with h5py.File(src, "r") as f:
        key = "mask" if "mask" in f else sorted(f.keys())[0]
        mask = f[key][:].astype(bool)
        attrs = {k: v for k, v in f.attrs.items()}

    if exclude_geojson is None and getattr(cfg, "deramp_exclude_geojson", ""):
        exclude_geojson = Path(cfg.deramp_exclude_geojson)
        if not exclude_geojson.exists():
            raise FileNotFoundError(
                f"deramp_exclude_geojson points at {exclude_geojson}, which does "
                "not exist. Fix the path rather than letting the ramp be fitted "
                "through the deforming area.")
    if exclude_geojson is None:
        # Project first, then the polygons shipped with the package - a new
        # project directory has no deliverables/context/ yet, and that absence
        # silently disabled the exclusion on every fresh run.
        # A fallback is used only when it actually lies on this site: the
        # package's Gibson polygon at another site excluded nothing, and the
        # ramp was fitted through that site's own deformation.
        off_site = []
        for ctx in (cfg.project_dir / "deliverables" / "context",
                    Path(__file__).parent / "context"):
            cands = sorted(ctx.glob("*active*.geojson")) if ctx.exists() else []
            for c in cands:
                if _geojson_overlaps_bbox(c, cfg.bbox):
                    exclude_geojson = c
                    log.info("Using %s to exclude the deforming area from the ramp "
                             "fit", exclude_geojson)
                    break
                off_site.append(c)
            if exclude_geojson is not None:
                break
        if exclude_geojson is None and off_site:
            raise FileNotFoundError(
                "NO EXCLUSION POLYGON for the deramp on this site: the only candidates ("
                + ", ".join(str(c) for c in off_site[:3]) + ") lie outside the processing box "
                f"{list(cfg.bbox)}, so they would exclude nothing. Set deramp_exclude_geojson to this "
                "site's mine / deforming-area polygon, or set mintpy_deramp: no.")
    if exclude_geojson is None:
        # A wrong path raises above; no path at all must not be gentler. The
        # only sign of a ramp fitted through the mine is a warning in a
        # multi-hour log, and the shrunken result looks fine.
        raise FileNotFoundError(
            "NO EXCLUSION POLYGON for the deramp. Searched deramp_exclude_geojson, "
            "<project>/deliverables/context/ and the package context/ and found "
            "none. Without it the ramp is fitted through any deformation in the "
            "scene and subtracted from it, shrinking the signal invisibly. Set "
            "deramp_exclude_geojson to the deforming-area polygon, or set "
            "mintpy_deramp: no.")
    else:
        latlon = None
        if "X_FIRST" not in attrs:
            geom = cfg.mintpy_dir / "inputs" / "geometryRadar.h5"
            with h5py.File(geom, "r") as g:
                latlon = (g["latitude"][:], g["longitude"][:])
        excluded = _polygon_mask(Path(exclude_geojson), mask.shape, attrs, latlon)
        if buffer_m > 0 and excluded.any():
            if latlon is not None:
                # Radar grid: not square, ~13 degrees off north and slightly
                # skewed on the ground, so an axis-sampled distance transform
                # still came up ~30 m short (Site B: 1,971 m). The ground
                # distance from each coherent pixel to the nearest excluded
                # one, measured on the pixels' own lat/lon, is exact.
                excluded = excluded | _within_ground_distance(excluded, mask, *latlon, buffer_m)
            else:
                from scipy.ndimage import distance_transform_edt
                dy_m, dx_m = _grid_spacing_m(attrs)
                excluded = distance_transform_edt(~excluded, sampling=(dy_m, dx_m)) <= buffer_m

    out = mask & ~excluded
    kept = out.sum()
    if kept < 0.2 * max(mask.sum(), 1):
        log.warning("The exclusion leaves only %.0f%% of coherent pixels to fit "
                    "the ramp on. That is few enough that the plane is poorly "
                    "constrained; consider a smaller buffer.",
                    100 * kept / max(mask.sum(), 1))
    surround = _fit_surrounds_aoi(cfg, out, attrs, latlon)
    if not surround["surrounds"]:
        shares = ", ".join(f"{q} {100 * v:.0f}%" for q, v in surround["share"].items())
        raise RuntimeError(
            f"The deramp fit ground does not surround the area: of the {int(kept)} pixels left "
            f"after excluding {Path(exclude_geojson).name} + {buffer_m:.0f} m, only "
            f"{surround['quadrants']} of the 4 quadrants around the area's centre hold at least "
            f"{100 * DERAMP_QUADRANT_MIN_SHARE:.0f}% of them ({shares}). A plane fitted on one side and "
            "subtracted across the area tilts every rate inside it. Reduce deramp_buffer_m "
            f"(now {buffer_m:.0f} m), widen the bbox so farmland surrounds the mines, or set "
            "mintpy_deramp: no (the QA then measures the tilt it leaves in).")

    dst = cfg.mintpy_dir / out_name
    with h5py.File(dst, "w") as f:
        d = f.create_dataset("mask", data=out.astype(np.bool_))
        d.attrs["MODIFICATION_TIME"] = str(time.time())
        for k, v in attrs.items():
            f.attrs[k] = v
        f.attrs["FILE_TYPE"] = "mask"
        # what the fit was built from, for provenance / QA to read back
        f.attrs["DERAMP_BUFFER_M"] = float(buffer_m)
        f.attrs["DERAMP_BUFFER_KIND"] = "euclidean"
        f.attrs["DERAMP_QUADRANTS"] = int(surround["quadrants"])
        f.attrs["DERAMP_QUADRANT_SHARE"] = json.dumps({k: round(v, 4) for k, v in surround["share"].items()})
    n_excl = int((mask & excluded).sum())
    log.info("Deramp mask: %d of %d coherent pixels kept (%.0f%%); %s -> %s",
             kept, mask.sum(), 100 * kept / max(mask.sum(), 1),
             (f"{n_excl} excluded by {Path(exclude_geojson).name} + {buffer_m:.0f} m buffer"
              if exclude_geojson is not None else
              "NOTHING EXCLUDED - no polygon was applied"), dst)
    return dst


def _within_ground_distance(src, cand, lat, lon, metres: float):
    """Pixels of `cand` (outside `src`) within `metres` on the ground of any
    `src` pixel, from per-pixel lat/lon (local equirectangular metres over a
    project-sized area; a KD-tree, so memory stays ~a few arrays of the grid)."""
    import numpy as np
    from scipy.spatial import cKDTree
    lat, lon = np.asarray(lat, dtype=float), np.asarray(lon, dtype=float)
    ok = np.isfinite(lat) & np.isfinite(lon) & (lat != 0) & (lon != 0)
    c = np.cos(np.radians(float(np.median(lat[ok])))) if ok.any() else 1.0

    def xy(m):
        return np.c_[lon[m] * 111_320.0 * c, lat[m] * 110_900.0]
    out = np.zeros(src.shape, bool)
    q = cand & ~src & ok
    if not (src & ok).any() or not q.any():
        return out
    d, _ = cKDTree(xy(src & ok)).query(xy(q), distance_upper_bound=metres)
    out[q] = d <= metres
    return out


def _grid_spacing_m(attrs) -> tuple:
    """(row, col) pixel size in metres of a geocoded or projected grid - the
    far field's own definition (significance.pixel_metres)."""
    try:
        from .significance import pixel_metres
        return tuple(float(v) for v in pixel_metres(attrs))
    except (ImportError, KeyError, TypeError, ValueError):
        px = _pixel_metres(attrs)
        return px, px


def _fit_surrounds_aoi(cfg: PipelineConfig, fit, attrs, latlon=None) -> dict:
    """Whether the deramp fit pixels surround the area of interest: the share
    of them in each quadrant around the bbox centre, and how many quadrants
    hold at least DERAMP_QUADRANT_MIN_SHARE. {surrounds, quadrants, share}."""
    import numpy as np
    w, s_, e, n = (float(v) for v in cfg.bbox)
    clon, clat = (w + e) / 2.0, (s_ + n) / 2.0
    rows, cols = np.nonzero(fit)
    total = rows.size
    if latlon is not None:
        la, lo = latlon[0][rows, cols], latlon[1][rows, cols]
    elif "X_FIRST" in attrs and abs(float(attrs["X_STEP"])) < 0.01:
        # cell centres of a geographic grid (as _polygon_mask places them)
        lo = float(attrs["X_FIRST"]) + (cols + 0.5) * float(attrs["X_STEP"])
        la = float(attrs["Y_FIRST"]) + (rows + 0.5) * float(attrs["Y_STEP"])
    elif "X_FIRST" in attrs:
        x = float(attrs["X_FIRST"]) + (cols + 0.5) * float(attrs["X_STEP"])
        y = float(attrs["Y_FIRST"]) + (rows + 0.5) * float(attrs["Y_STEP"])
        from pyproj import Transformer
        from .stack_qc import product_epsg
        # the product's EPSG, else its zone/hemisphere placed by the AOI
        # centre - never a default zone (16N, Indiana, until 8 Oct 2026)
        epsg = product_epsg(attrs, lonlat=(clon, clat), shape=np.shape(fit))
        lo, la = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True).transform(x, y)
    else:
        return {"surrounds": True, "quadrants": 4, "share": {}, "note": "no coordinates to judge from"}
    la, lo = np.asarray(la), np.asarray(lo)
    north, east = la >= clat, lo >= clon
    counts = {"NE": int((north & east).sum()), "NW": int((north & ~east).sum()),
              "SE": int((~north & east).sum()), "SW": int((~north & ~east).sum())}
    share = {q: c / max(total, 1) for q, c in counts.items()}
    quads = sum(v >= DERAMP_QUADRANT_MIN_SHARE for v in share.values())
    return {"surrounds": bool(total) and quads >= DERAMP_MIN_QUADRANTS, "quadrants": int(quads),
            "share": share}


def _pixel_metres(attrs) -> float:
    """Ground pixel size in metres; X_STEP is 30.0 for UTM, ~0.0003 for degrees."""
    import math
    try:
        dx = abs(float(attrs["X_STEP"]))
        dy = abs(float(attrs.get("Y_STEP", dx)))
    except (KeyError, TypeError, ValueError):
        return 30.0
    if dx < 0.01:
        lat = float(attrs.get("Y_FIRST", 0.0) or 0.0)
        return float((dx * 111_320.0 * math.cos(math.radians(lat))
                      + dy * 110_900.0) / 2.0)
    return float((dx + dy) / 2.0)


def _polygon_mask(geojson: Path, shape, attrs, latlon=None):
    """
    Rasterise GeoJSON polygons (lon/lat) onto the product grid.

    Handles both coordinate systems, because a UTM product needs the grid
    projected to lon/lat before the point-in-polygon test and a geographic
    one does not.
    """
    import numpy as np

    ny, nx = shape
    if latlon is not None:
        # RADAR coordinates: there is no regular grid, so X_FIRST does not
        # exist. The topsStack path stays in radar geometry until MintPy's final
        # geocode step - which is exactly when the deramp mask is needed - so
        # per-pixel lat/lon from geometryRadar.h5 is the only way to place the
        # polygon. Assuming a geocoded grid here crashed the mask builder, and
        # the deramp silently fell back to fitting through the mine.
        lat, lon = latlon
        return _points_in_polygons(geojson, lon, lat, shape)
    x0, y0 = float(attrs["X_FIRST"]), float(attrs["Y_FIRST"])
    dx, dy = float(attrs["X_STEP"]), float(attrs["Y_STEP"])
    # cell CENTRES: X_FIRST/Y_FIRST are the outer corner of the first cell
    # (MintPy yx2lalo adds 0.5); testing corners shifted every mask half a
    # cell north-west
    yy, xx = np.mgrid[0:ny, 0:nx]
    X, Y = x0 + (xx + 0.5) * dx, y0 + (yy + 0.5) * dy
    if abs(dx) >= 0.01:                                   # metres -> lon/lat
        from pyproj import Transformer
        from .stack_qc import epsg_attr, product_epsg
        epsg = epsg_attr(attrs)
        if epsg is None:
            # No EPSG in the metadata: the zone and hemisphere come from
            # UTM_ZONE and the polygons being placed, checked against the
            # grid (stack_qc.product_epsg). Until 8 Oct 2026 this fell back
            # to 16N (Indiana), and a latitude-band "11S" (Corcoran, 36 N)
            # was read as the southern hemisphere.
            hint = _geojson_lonlat_centre(geojson)
            if hint is None:                  # no polygon: nothing can be inside
                return np.zeros((ny, nx), bool)
            epsg = product_epsg(attrs, lonlat=hint, shape=shape)
        lon, lat = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326",
                                        always_xy=True).transform(X, Y)
    else:
        lon, lat = X, Y

    return _points_in_polygons(geojson, lon, lat, shape)


def _geojson_lonlat_centre(geojson):
    """(lon, lat) mean of the exterior-ring vertices of every polygon in a
    lon/lat GeoJSON (the features _points_in_polygons tests), or None when
    it holds no polygon. Only used to place a grid whose metadata states no
    EPSG, so the mean of vertices is precise enough."""
    import json
    import numpy as np
    try:
        data = json.loads(Path(geojson).read_text())
    except (OSError, ValueError):
        return None
    feats = data.get("features", [data] if data.get("type") in ("Polygon", "MultiPolygon") else [])
    pts = []
    for feat in feats:
        g = (feat or {}).get("geometry") or feat or {}
        if g.get("type") not in ("Polygon", "MultiPolygon", None):      # as _points_in_polygons reads them
            continue
        polygons = g.get("coordinates", []) if g.get("type") == "MultiPolygon" else [g.get("coordinates", [])]
        for polygon in polygons:
            if polygon and polygon[0]:
                pts.extend((float(p[0]), float(p[1])) for p in polygon[0])
    if not pts:
        return None
    a = np.asarray(pts)
    return float(a[:, 0].mean()), float(a[:, 1].mean())


def _points_in_polygons(geojson, lon, lat, shape):
    import json
    import numpy as np
    from matplotlib.path import Path as MPath
    ny, nx = shape
    data = json.loads(Path(geojson).read_text())
    lonr, latr = np.asarray(lon).ravel(), np.asarray(lat).ravel()
    out = np.zeros(ny * nx, bool)

    def contains(ring):
        """point-in-ring for the cells inside the ring's bounding box only
        (a panel is ~0.1% of the grid; testing all 6 M cells per ring made
        the 53-panel history take minutes)"""
        r = np.array(ring)[:, :2]
        cand = np.flatnonzero((lonr >= r[:, 0].min()) & (lonr <= r[:, 0].max())
                              & (latr >= r[:, 1].min()) & (latr <= r[:, 1].max()))
        hit = np.zeros(ny * nx, bool)
        if cand.size:
            hit[cand] = MPath(r).contains_points(np.c_[lonr[cand], latr[cand]])
        return hit
    feats = data.get("features", [data] if data.get("type") in ("Polygon", "MultiPolygon") else [])
    for feat in feats:
        g = feat.get("geometry") or feat
        polygons = (g.get("coordinates", []) if g.get("type") == "MultiPolygon"
                    else [g.get("coordinates", [])])
        for polygon in polygons:
            if not polygon or not polygon[0]:
                continue
            # ring 0 is the exterior; every further ring is a HOLE. The mine
            # polygon has nine of them (3.3% of its area), and treating them
            # as filled put non-mine ground into every mine statistic.
            inside = contains(polygon[0])
            for hole in polygon[1:]:
                if hole:
                    inside &= ~contains(hole)
            out |= inside
    return out.reshape(ny, nx)
