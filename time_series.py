"""Stage 5: MintPy time series inversion (SBAS)."""
import json
import logging
import os
import re
import subprocess
import threading
import time

import numpy as np
from pathlib import Path

from . import corrections
from .config import PipelineConfig

log = logging.getLogger(__name__)

_BPERP_RE = re.compile(
    r"baseline\.IW-2 Bperp at midrange for (first|last) common burst = ([-\d.]+)"
)


def write_baselines(cfg: PipelineConfig) -> Path | None:
    """
    Build the baseline files MintPy expects, from ISCE's own logs.

    topsApp computes perpendicular baselines during runComputeBaseline and
    writes them to each pair's isce.log, but it doesn't emit the
    baselines/<pair>/<pair>.txt layout that MintPy's isce reader looks for
    (that's a topsStack convention). Without them, load_data fails on a
    missing P_BASELINE_TOP_HDR key after successfully loading every
    interferogram.

    Uses IW-2, the middle sub-swath, as representative of the scene.
    """
    baseline_root = cfg.project_dir / "baselines"
    baseline_root.mkdir(parents=True, exist_ok=True)
    written = 0

    for pair_dir in sorted(cfg.pairs_dir.iterdir()):
        log_file = pair_dir / "isce.log"
        if not log_file.is_dir() and log_file.exists():
            text = log_file.read_text(errors="ignore")
        else:
            continue

        found = dict(_BPERP_RE.findall(text))
        if "first" not in found or "last" not in found:
            log.warning("No IW-2 baseline entries in %s", log_file)
            continue

        out_dir = baseline_root / pair_dir.name
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"{pair_dir.name}.txt").write_text(
            f"PERP_BASELINE_TOP {found['first']}\n"
            f"PERP_BASELINE_BOTTOM {found['last']}\n"
        )
        written += 1

    if not written:
        log.warning("Wrote no baseline files — MintPy may fail on missing "
                    "P_BASELINE_TOP_HDR. Check that isce.log survives in each "
                    "pair directory.")
        return None

    log.info("Wrote baseline files for %d pairs", written)
    return baseline_root


def patch_baseline_metadata(cfg: PipelineConfig) -> int:
    """
    Write baseline values into each interferogram's ISCE .xml.

    MintPy's prep_isce step normally injects these, but it fails silently
    on topsApp output and load_data then sees NaN baselines. Patching the
    .xml (not the .rsc) is what sticks: MintPy regenerates the .rsc from
    the .xml during load, so anything written to the .rsc is overwritten.

    Any stale .rsc is removed so it is rebuilt from the patched .xml.
    """
    import xml.etree.ElementTree as ET

    patched = 0
    for pair_dir in sorted(cfg.pairs_dir.iterdir()):
        log_file = pair_dir / "isce.log"
        if not log_file.exists():
            continue
        found = dict(_BPERP_RE.findall(log_file.read_text(errors="ignore")))
        if "first" not in found or "last" not in found:
            log.warning("No IW-2 baselines in %s — its bperp will stay NaN",
                        log_file)
            continue

        for name in ("filt_topophase.tropocorrected.unw.geo",
                     "filt_topophase.unw.geo"):
            target = pair_dir / "merged" / name
            xml_path = Path(str(target) + ".xml")
            if not xml_path.exists():
                continue

            tree = ET.parse(xml_path)
            root = tree.getroot()
            for key, val in (("P_BASELINE_TOP_HDR", found["first"]),
                             ("P_BASELINE_BOTTOM_HDR", found["last"])):
                for prop in root.findall(f".//property[@name='{key}']"):
                    root.remove(prop)
                prop = ET.SubElement(root, "property", {"name": key})
                ET.SubElement(prop, "value").text = str(val)
            tree.write(xml_path)

            # Keep the .rsc and update it in place. Deleting it loses
            # DATE12, which MintPy requires at load time and does not
            # regenerate from the xml.
            rsc = Path(str(target) + ".rsc")
            keys = {}
            if rsc.exists():
                for line in rsc.read_text().splitlines():
                    parts = line.split(None, 1)
                    if len(parts) == 2:
                        keys[parts[0]] = parts[1].strip()

            ref, sec = pair_dir.name.split("_")
            keys.setdefault("DATE12", f"{ref[2:]}-{sec[2:]}")
            keys["P_BASELINE_TOP_HDR"] = str(found["first"])
            keys["P_BASELINE_BOTTOM_HDR"] = str(found["last"])
            rsc.write_text("".join(f"{k:<25}{v}\n" for k, v in sorted(keys.items())))
            patched += 1

    log.info("Patched baseline metadata into %d interferogram xml files", patched)
    return patched


def _screened_out_pairs(cfg: PipelineConfig) -> list[str]:
    """
    Pairs to exclude because their tropospheric delay varies too much.

    Uses the per-pair statistics recorded during Stage 4. The gate is on
    the spatial standard deviation of the differential delay, not its
    mean, since a uniform offset cancels against the reference point.

    Two caveats worth keeping in mind:
      * ERA5's native grid is coarse (tens of km), so it captures
        large-scale water vapour patterns but misses small-scale
        turbulent structure. A pair can pass this gate and still be noisy.
      * SBAS needs a connected network. Excluding too many pairs can
        fragment the graph and leave dates unconstrained, which is worse
        than including a mediocre pair. The logged count is worth reading
        before trusting the result.
    """
    manual = list(cfg.exclude_pairs or [])
    if manual:
        log.info("Excluding %d manually listed pairs: %s", len(manual), ", ".join(manual))

    if not cfg.max_tropo_std_cm:
        return manual

    stats_path = cfg.project_dir / "tropo_screening.json"
    if not stats_path.exists():
        log.info("No tropo screening data — keeping all pairs")
        return manual

    stats = json.loads(stats_path.read_text())
    excluded = [name for name, s in stats.items()
                if s.get("delay_std_cm", 0) > cfg.max_tropo_std_cm]
    for name in excluded:
        log.info("Screening out %s (delay std %.2f cm > %.2f cm limit)",
                 name, stats[name]["delay_std_cm"], cfg.max_tropo_std_cm)
    if excluded:
        log.warning("Excluding %d of %d pairs on tropospheric delay. Check the "
                    "network stays connected — MintPy's network plots in "
                    "mintpy/pic/ will show this.", len(excluded), len(stats))
    return sorted(set(manual) | set(excluded))


def _resolve_unw_pattern(cfg: PipelineConfig) -> tuple[str, str]:
    """
    Decide which unwrapped files to feed MintPy.

    Prefers the RAiDER-corrected files, but falls back to the raw geocoded
    output if Stage 4 didn't produce any — otherwise a tropo failure
    silently guarantees a Stage 5 failure with a confusing "no data files
    found" error. Returns (glob pattern, tropo method for MintPy).
    """
    corrected = list(cfg.pairs_dir.glob("*/merged/filt_topophase.tropocorrected.unw.geo"))
    if corrected:
        log.info("Using %d RAiDER-corrected interferograms", len(corrected))
        return (f"{cfg.pairs_dir}/*/merged/filt_topophase.tropocorrected.unw.geo", "no")

    raw = list(cfg.pairs_dir.glob("*/merged/filt_topophase.unw.geo"))
    log.warning(
        "No RAiDER-corrected interferograms found — falling back to %d "
        "uncorrected ones. Tropospheric delay will NOT be removed; treat "
        "the velocities accordingly.", len(raw)
    )
    return (f"{cfg.pairs_dir}/*/merged/filt_topophase.unw.geo", "no")


# mintpy.troposphericDelay.method is "no" because the correction is handled
# in Stage 4 via RAiDER (or deliberately skipped) — leaving MintPy's own
# tropo step on would double-correct.
#
# mintpy.topographicResidual is a SEPARATE, unrelated correction: it removes
# residual phase caused by an imperfect DEM (error that scales with
# perpendicular baseline), not atmospheric delay. RAiDER doesn't touch this —
# it's estimated and removed here, during the time series inversion.
_MINTPY_CONFIG_TEMPLATE = """\
mintpy.load.processor              = isce
mintpy.load.metaFile               = {meta_file}
mintpy.load.baselineDir            = {baseline_dir}
mintpy.load.unwFile                = {unw_pattern}
mintpy.load.corFile                = {pairs_dir}/*/merged/phsig.cor.geo
mintpy.load.demFile                = {dem_path}
mintpy.load.incAngleFile           = {pairs_dir}/*/merged/los.rdr.geo
mintpy.reference.lalo              = {reference_lalo}
mintpy.networkInversion.minTempCoh = {min_temp_coh}
mintpy.network.minCoherence        = {min_spatial_coh}
mintpy.reference.minCoherence      = {ref_min_coh}
{subset_line}
mintpy.troposphericDelay.method    = {tropo_method}
mintpy.topographicResidual         = yes
mintpy.topographicResidual.polyOrder = 2
mintpy.networkInversion.weightFunc = var
{exclude_line}"""


def write_mintpy_config(cfg: PipelineConfig, dem_path: Path) -> Path:
    cfg.mintpy_dir.mkdir(parents=True, exist_ok=True)
    unw_pattern, tropo_method = _resolve_unw_pattern(cfg)
    # MintPy reads platform metadata (satellite altitude, etc.) from one
    # pair's reference IW xml; without it, load_data fails on a missing
    # HEIGHT key partway through building the geometry file.
    meta_candidates = sorted(cfg.pairs_dir.glob("*/reference/IW*.xml"))
    meta_file = str(meta_candidates[0]) if meta_candidates else "auto"
    if meta_file == "auto":
        log.warning("No reference/IW*.xml found under %s — MintPy will likely "
                    "fail on missing platform metadata. Regenerate it by "
                    "running topsApp.py --start=startup --end=preprocess in "
                    "one pair directory.", cfg.pairs_dir)

    # An explicit reference point matters for anything leaving the
    # building: every displacement is relative to it, so an automatically
    # chosen pixel makes the numbers un-defensible.
    reference_lalo = cfg.reference_lalo.strip() if cfg.reference_lalo else "auto"
    if reference_lalo == "auto":
        log.warning("reference_lalo is unset - MintPy will choose the "
                    "reference point. Fine for testing, not for a deliverable.")

    baseline_dir = write_baselines(cfg)
    baseline_dir = str(baseline_dir) if baseline_dir else "auto"
    patch_baseline_metadata(cfg)

    excluded = _screened_out_pairs(cfg)
    exclude_line = (f"mintpy.network.excludeDate12           = {','.join(excluded)}\n"
                    if excluded else "")
    config_text = _MINTPY_CONFIG_TEMPLATE.format(
        pairs_dir=cfg.pairs_dir, dem_path=dem_path,
        unw_pattern=unw_pattern, tropo_method=tropo_method,
        exclude_line=exclude_line, meta_file=meta_file,
        baseline_dir=baseline_dir, reference_lalo=reference_lalo,
        min_temp_coh=cfg.min_temporal_coherence,
        min_spatial_coh=cfg.min_spatial_coherence,
        ref_min_coh=cfg.reference_min_coherence,
        subset_line=(
            f"mintpy.subset.lalo                 = "
            f"{cfg.bbox[1]}:{cfg.bbox[3]},{cfg.bbox[0]}:{cfg.bbox[2]}"
            if cfg.subset_to_aoi else ""
        ),
    )
    config_path = cfg.mintpy_dir / "smallbaselineApp.cfg"
    config_path.write_text(config_text)
    return config_path


def patch_stack_bperp(cfg: PipelineConfig) -> None:
    """
    Write perpendicular baselines straight into ifgramStack.h5.

    MintPy derives bperp by converting the baselines directory into a
    date-indexed time series, and that conversion leaves the first pair as
    NaN on topsApp output no matter how the metadata is staged. plot_network
    then refuses to run. Writing the values directly into the loaded stack
    sidesteps the conversion entirely - the numbers come from ISCE's own
    runComputeBaseline output either way.
    """
    import h5py

    stack = cfg.mintpy_dir / "inputs" / "ifgramStack.h5"
    if not stack.exists():
        log.warning("No ifgramStack.h5 to patch at %s", stack)
        return

    bperp_by_pair = {}
    for pair_dir in sorted(cfg.pairs_dir.iterdir()):
        log_file = pair_dir / "isce.log"
        if not log_file.exists():
            continue
        found = dict(_BPERP_RE.findall(log_file.read_text(errors="ignore")))
        if "first" in found and "last" in found:
            bperp_by_pair[pair_dir.name] = (
                float(found["first"]) + float(found["last"])) / 2

    with h5py.File(stack, "r+") as f:
        dates = f["date"][:]
        bperp = f["bperp"][:]
        for i, (ref, sec) in enumerate(dates):
            name = f"{ref.decode()}_{sec.decode()}"
            if name in bperp_by_pair:
                bperp[i] = bperp_by_pair[name]
            elif not np.isfinite(bperp[i]):
                log.warning("No baseline for %s and its bperp is NaN — "
                            "setting 0 so the inversion can proceed", name)
                bperp[i] = 0.0
        f["bperp"][:] = bperp
        log.info("Patched bperp in ifgramStack.h5: %s",
                 np.array2string(bperp, precision=1))

        # MintPy's variance-based weighting needs the multilook factors,
        # which topsApp doesn't record in the geocoded product. We never
        # set looks in topsApp.xml, and the product's pixel sizes are
        # native Sentinel-1 IW (~2.3 m range, ~15.6 m azimuth), so 1/1 is
        # correct here. These only affect the inversion's weighting, not
        # the displacements themselves.
        for key, val in (("ALOOKS", "1"), ("RLOOKS", "1")):
            if key not in f.attrs:
                f.attrs[key] = val
                log.info("Set %s=%s on ifgramStack.h5", key, val)


def run_mintpy(cfg: PipelineConfig, dem_path: Path) -> None:
    """
    Run MintPy in two phases, patching baselines in between.

    MintPy derives bperp by converting the baselines directory into a
    date-indexed time series, and on topsApp output that conversion leaves
    the first pair NaN no matter how the metadata is staged - plot_network
    then refuses to continue. Loading first, writing the baselines straight
    into ifgramStack.h5, then running the rest sidesteps the conversion.
    The numbers come from ISCE's own runComputeBaseline either way.
    """
    config_path = write_mintpy_config(cfg, dem_path)

    log.info("Running MintPy load_data")
    load = subprocess.run(
        ["smallbaselineApp.py", str(config_path),
         "--start", "load_data", "--end", "load_data"],
        cwd=cfg.mintpy_dir, capture_output=True, text=True,
    )
    (cfg.mintpy_dir / "mintpy_load.log").write_text(load.stdout + load.stderr)
    if load.returncode != 0:
        log.error("MintPy load_data failed - see mintpy_load.log")
        raise RuntimeError("MintPy load_data failed - see mintpy_load.log")

    patch_stack_bperp(cfg)

    # The previous run's corrected series and velocities go aside first, so
    # velocity.h5 below can only be this run's (retire_downstream_products);
    # until this run succeeds, run_incomplete.json says so (mark_run_incomplete).
    t_retire = time.time()
    stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(t_retire))
    mark_run_incomplete(cfg, stamp, t_retire)
    retire_downstream_products(cfg, stamp)
    log.info("Running MintPy remaining steps")
    t = time.time()
    result = subprocess.run(
        ["smallbaselineApp.py", str(config_path), "--start", "modify_network"],
        cwd=cfg.mintpy_dir, capture_output=True, text=True,
    )
    (cfg.mintpy_dir / "mintpy_error.log").write_text(result.stdout + result.stderr)

    # MintPy's final plotting step can fail on a one-pixel size mismatch
    # between the DEM and the interferogram grid while every science
    # product is already written. Judge success on velocity.h5 written BY
    # THIS RUN, not on the exit code (nor on an old one still being there).
    velocity = cfg.mintpy_dir / "velocity.h5"
    if velocity.exists() and velocity.stat().st_mtime >= t - 1.0:
        if result.returncode != 0:
            log.warning("MintPy exited non-zero but velocity.h5 was written - "
                        "the inversion finished and only a later step (usually "
                        "plotting) failed. See mintpy_error.log.")
        clear_run_incomplete(cfg)
        log.info("MintPy time series complete. Outputs in %s", cfg.mintpy_dir)
        return

    log.error("MintPy failed - see mintpy_error.log")
    raise RuntimeError("MintPy processing failed - see mintpy_error.log:\n"
                       + _log_tail(cfg.mintpy_dir / "mintpy_error.log"))


def _mintpy(cfg: PipelineConfig, config_path: Path, start: str, end: str | None,
            log_name: str, append: bool = False) -> subprocess.CompletedProcess:
    """One smallbaselineApp.py invocation, in its own process group. With
    append, the output is added to log_name instead of replacing it (two
    calls that make up one phase share one log)."""
    from . import procs
    procs.install_teardown()
    from . import progress
    argv = ["smallbaselineApp.py", str(config_path), "--start", start]
    if end:
        argv += ["--end", end]
    stage = f"mintpy_{start}" + (f"_to_{end}" if end and end != start else "")
    progress.emit(cfg.project_dir, stage, "start")
    result = procs.run_group(argv, cwd=cfg.mintpy_dir)
    with open(cfg.mintpy_dir / log_name, "a" if append else "w") as fh:
        fh.write((result.stdout or "") + (result.stderr or ""))
    progress.emit(cfg.project_dir, stage, "done" if result.returncode == 0 else "failed",
                  msg="" if result.returncode == 0 else f"rc={result.returncode}, see {log_name}")
    return result


def _log_tail(path: Path, n: int = 12) -> str:
    """The last `n` lines of a log, for an error message a coworker can act on."""
    try:
        lines = Path(path).read_text(errors="replace").splitlines()
    except OSError:
        return f"({path} could not be read)"
    return "\n".join(lines[-n:])


def _require_fresh(paths, since: float, what: str, log_name: str) -> None:
    """
    Every path exists AND was written at or after `since`, else RuntimeError
    naming each missing or old one.

    Judged on the files, not on MintPy's exit code or on a file merely
    existing: until 5 Oct 2026 a re-run whose ERA5 step failed returned
    normally because the PREVIOUS run's geo/geo_velocity.h5 was still there,
    and the deliverables were rebuilt, dated today, from the old numbers.
    One second of slack covers filesystem timestamp granularity.
    """
    paths = [Path(p) for p in paths]
    missing = [p for p in paths if not p.exists()]
    old = [p for p in paths if p.exists() and p.stat().st_mtime < since - 1.0]
    if not missing and not old:
        return
    parts = []
    if missing:
        parts.append("not written: " + ", ".join(p.name for p in missing))
    if old:
        parts.append("older than this step (left from an earlier run): " + ", ".join(
            f"{p.name} ({time.strftime('%d %b %H:%M', time.localtime(p.stat().st_mtime))})" for p in old))
    raise RuntimeError(f"MintPy {what} did not produce its products - " + "; ".join(parts)
                       + f". See {log_name}.")


def _mod_time(ds) -> float | None:
    """An HDF5 dataset's MODIFICATION_TIME stamp (MintPy writes it as a str,
    the pipeline sometimes as a float), or None when it carries none."""
    v = ds.attrs.get("MODIFICATION_TIME")
    if v is None:
        return None
    try:
        return float(v.decode() if isinstance(v, bytes) else v)
    except (TypeError, ValueError):
        return None


def mintpy_template(config_path: Path) -> dict:
    """
    The template smallbaselineApp actually runs with: MintPy's defaults with
    this config laid over them and every 'auto' resolved, built the way
    TimeSeriesAnalysis._read_template builds it (loose yes/no spellings
    normalised, deramp and troposphere methods lower-cased). Used to name the
    products MintPy will write, rather than guessing from file names.
    """
    import mintpy
    from mintpy.utils import readfile, utils as ut
    default = Path(mintpy.__file__).parent / "defaults" / "smallbaselineApp.cfg"
    t = readfile.read_template(str(default))
    std = {"def": "auto", "default": "auto", "y": "yes", "on": "yes", "true": "yes",
           "n": "no", "off": "no", "false": "no"}
    custom = {k: std.get(v, v) for k, v in readfile.read_template(str(config_path)).items()}
    for k in ("mintpy.deramp", "mintpy.troposphericDelay.method"):
        if k in custom:
            custom[k] = custom[k].lower().replace("-", "_")
    t.update(custom)
    return ut.check_template_auto_value(t)


def chain_files(cfg: PipelineConfig, config_path: Path) -> dict:
    """
    MintPy's own names for this run's correction chain:
    TimeSeriesAnalysis.get_timeseries_filename on the resolved template - the
    function smallbaselineApp itself uses to decide what each step reads and
    writes. "final" is the series the velocity is fitted to and the one the
    geocode step geocodes; the deramp step's input and output are what the
    removed tilt is measured from. Needs mintpy/timeseries.h5 (MintPy reads
    its PLATFORM to decide on the LOD step).
    """
    from mintpy.smallbaselineApp import TimeSeriesAnalysis
    t = mintpy_template(config_path)
    steps = TimeSeriesAnalysis.get_timeseries_filename(t, str(cfg.mintpy_dir))
    final = Path(steps["velocity"]["input"])
    return {"final": final,
            "tokens": [tok for tok in final.stem.split("_")[1:] if tok],
            "deramp_in": Path(steps["deramp"]["input"]), "deramp_out": Path(steps["deramp"]["output"]),
            "geocode": bool(t.get("mintpy.geocode", True)),
            "tropo_method": t.get("mintpy.troposphericDelay.method") or "no",
            # MintPy writes the delays to inputs/<MODEL upper-cased>.h5
            "weather_model": str(t.get("mintpy.troposphericDelay.weatherModel") or "ERA5").upper(),
            "template": t}


class _WeatherWatch:
    """
    Progress lines while MintPy's ERA5 step waits on the Copernicus queue.

    Measured on Site B (29 Sep 2026): correct_LOD ran from 10:21 to 13:30,
    almost all of it ERA5 requests queued at the CDS, with nothing in the
    progress record - the run looked hung for three hours. This counts the
    weather files as they land (mintpy/ERA5/, and $WEATHER_DIR/ERA5 when set)
    and reports the count against the number of dates, once a minute and only
    when it changes. Never raises: it is a display, not a check.
    """

    def __init__(self, cfg, stage: str, total: int, interval: float = 60.0):
        self.cfg, self.stage, self.total, self.interval = cfg, stage, int(total), float(interval)
        self.dirs = [Path(cfg.mintpy_dir) / "ERA5"]
        if os.environ.get("WEATHER_DIR"):
            self.dirs.append(Path(os.environ["WEATHER_DIR"]) / "ERA5")
        self.last = None
        self._stop = threading.Event()
        self._thread = None

    def count(self) -> int:
        n = 0
        for d in self.dirs:
            try:
                n += sum(1 for _ in d.glob("*.grb"))
            except OSError:
                pass
        return n

    def poll(self) -> bool:
        """Report the count if it changed; True when a line was written."""
        from . import progress
        try:
            n = self.count()
        except Exception:                                      # noqa: BLE001
            return False
        if n == self.last:
            return False
        self.last = n
        progress.emit(self.cfg.project_dir, self.stage, "running", done=min(n, self.total) if self.total else n,
                      total=self.total or None,
                      msg="ERA5 weather files downloaded (requests can wait hours in the Copernicus queue)")
        return True

    def _loop(self):
        while not self._stop.wait(self.interval):
            self.poll()

    def __enter__(self):
        self._thread = threading.Thread(target=self._loop, name="era5-watch", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        return False


def mintpy_reference_yx(cfg: PipelineConfig, lat: float, lon: float) -> tuple[int, int]:
    """The radar pixel MintPy itself maps a lat/lon to (its coordinate utility on
    the loaded geometry), so every check here looks at the pixel MintPy uses.
    MEASURED 26 Sep 2026: a nearest-lat/lon argmin landed one pixel away from
    MintPy's answer and let four pairs with the reference in component 0 through."""
    from mintpy.utils import readfile, utils as ut
    stack = cfg.mintpy_dir / "inputs" / "ifgramStack.h5"
    atr = readfile.read_attribute(str(stack))
    coord = ut.coordinate(atr, lookup_file=str(cfg.mintpy_dir / "inputs" / "geometryRadar.h5"))
    y, x = coord.geo2radar(np.array([lat]), np.array([lon]))[0:2]
    return int(np.ravel(y)[0]), int(np.ravel(x)[0])


def nearest_unmasked(mask: np.ndarray, y: int, x: int, radius: int = 4):
    """Closest (y, x) with mask True within `radius` pixels of (y, x), or None."""
    best = None
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            yy, xx = y + dy, x + dx
            if 0 <= yy < mask.shape[0] and 0 <= xx < mask.shape[1] and mask[yy, xx]:
                d = dy * dy + dx * dx
                if best is None or d < best[0]:
                    best = (d, yy, xx)
    return None if best is None else (best[1], best[2])


def _set_config_key(config_path: Path, key: str, value: str) -> None:
    lines = [l for l in config_path.read_text().splitlines() if not l.strip().startswith(key + " ") and not l.strip().startswith(key + "=")]
    lines.append(f"{key} = {value}")
    config_path.write_text("\n".join(lines) + "\n")


def write_conncomp_mask(cfg: PipelineConfig) -> Path | None:
    """
    mintpy/maskConnComp.h5 from the CURRENT kept pair set: the pixels whose
    connected component (unwrapped phase when the stack carries no
    components) is non-zero and finite in every pair dropIfgram keeps - what
    MintPy's generate_mask.py --nonzero writes inside its reference_point
    step (ifgramStack.nonzero_mask), computed here so the reference pin can
    look at it BEFORE that step. MintPy's own step then finds the file newer
    than the stack and keeps it (--update), so both see the same mask.
    Streams one pair at a time. None when the stack is missing.
    """
    import h5py
    stack = cfg.mintpy_dir / "inputs" / "ifgramStack.h5"
    if not stack.exists():
        return None
    out = cfg.mintpy_dir / "maskConnComp.h5"
    with h5py.File(stack, "r") as f:
        name = "connectComponent" if "connectComponent" in f else "unwrapPhase"
        ds = f[name]
        keep = f["dropIfgram"][:].astype(bool) if "dropIfgram" in f else np.ones(ds.shape[0], bool)
        mask = np.ones(ds.shape[1:], bool)
        for i in np.flatnonzero(keep):
            a = ds[i, :, :]
            mask &= (a != 0) & np.isfinite(a)
        attrs = {k: (v.decode() if isinstance(v, bytes) else v) for k, v in f.attrs.items()}
    attrs["FILE_TYPE"] = "mask"
    tmp = out.with_name(out.stem + "_tmp.h5")
    with h5py.File(tmp, "w") as g:
        g.create_dataset("mask", data=mask)
        for k, v in attrs.items():
            g.attrs[k] = v
    os.replace(tmp, out)
    log.info("maskConnComp.h5 rebuilt from the %d kept pairs: %.1f%% of the grid unwrapped in every one",
             int(keep.sum()), 100 * float(mask.mean()))
    return out


def pin_reference_into_conncomp_mask(cfg: PipelineConfig, config_path: Path, radius: int = 4) -> dict:
    """
    MintPy's reference_point refuses a pixel that maskConnComp.h5 masks out,
    and that mask is the intersection of connected components over every kept
    pair, so a single kept pair with the reference in component 0 stops the
    run. After the unreliable pairs are excluded, look at the pixel MintPy
    will use; if it is still masked, move the reference to the nearest
    unmasked pixel within `radius` (15 m posting: a few tens of metres) by
    writing mintpy.reference.yx AND setting mintpy.reference.lalo to auto:
    MintPy 1.6 converts a set lalo to y/x and overwrites the yx with it
    (reference_point.read_reference_input), so a pin written beside the
    configured lalo never took effect. Records the
    move in mintpy/reference_pin.json.

    The mask is rebuilt from the current kept set first whenever the stack is
    newer than it or it is missing (write_conncomp_mask). MintPy only writes
    it inside reference_point, AFTER this check, and the run deletes it
    whenever pairs are excluded - so until 5 Oct 2026 the pin found no mask on
    every fresh run and did nothing (no reference_pin.json in either
    delivered project), and on a re-run read the previous run's mask.
    """
    import h5py
    mask_file = cfg.mintpy_dir / "maskConnComp.h5"
    if not (cfg.reference_lalo or "").strip():
        return {}
    stack = cfg.mintpy_dir / "inputs" / "ifgramStack.h5"
    if stack.exists() and (not mask_file.exists() or mask_file.stat().st_mtime < stack.stat().st_mtime):
        write_conncomp_mask(cfg)
    if not mask_file.exists():
        return {}
    lat0, lon0 = (float(v) for v in cfg.reference_lalo.split(","))
    y, x = mintpy_reference_yx(cfg, lat0, lon0)
    with h5py.File(mask_file, "r") as f:
        mask = f["mask"][:].astype(bool)
    info = {"lalo": [lat0, lon0], "mintpy_yx": [y, x], "in_mask": bool(mask[y, x]), "mask_fraction": float(mask.mean())}
    if mask[y, x]:
        (cfg.mintpy_dir / "reference_pin.json").write_text(json.dumps(info, indent=2))
        return info
    hit = nearest_unmasked(mask, y, x, radius)
    if hit is None:
        raise RuntimeError(
            f"The reference {cfg.reference_lalo} (MintPy pixel {y},{x}) is outside maskConnComp.h5 and no pixel within "
            f"{radius} px is unwrapped in every kept pair (mask keeps {100 * mask.mean():.1f}% of the grid). Move reference_lalo.")
    ny, nx = hit
    _set_config_key(config_path, "mintpy.reference.yx", f"{ny},{nx}")
    _set_config_key(config_path, "mintpy.reference.lalo", "auto")    # or MintPy overrides the yx
    info.update({"pinned_yx": [ny, nx], "shift_px": [ny - y, nx - x]})
    (cfg.mintpy_dir / "reference_pin.json").write_text(json.dumps(info, indent=2))
    log.warning("Reference %s is MintPy pixel %d,%d, which one or more kept pairs leave in component 0; pinned to the "
                "nearest pixel unwrapped in every kept pair, %d,%d (%d,%d px away, ~%d m) via mintpy.reference.yx",
                cfg.reference_lalo, y, x, ny, nx, ny - y, nx - x, int(np.hypot((ny - y) * 42, (nx - x) * 41)))
    return info


def reference_unreliable_pairs(cfg: PipelineConfig, cc_min_area: float = 2.5e3) -> tuple[list, dict]:
    """
    Pairs in which the reference pixel sits in connected component 0.

    MintPy's maskConnComp is an INTERSECTION over every kept pair, so one
    pair in which snaphu could not reliably unwrap the reference pixel
    masks it out of the whole run and reference_point dies with "input
    reference point is in masked OUT area". On v1 that was two pairs,
    found by hand and written into exclude_pairs; on v2 a third appeared
    (one of the pairs v1 never finished). At 820 pairs nobody will do this
    by hand, so it is read from ifgramStack.h5 after the load.

    Returns (pairs, info). Pairs is empty when the reference is reliable
    everywhere or when reference_lalo is unset.
    """
    import h5py
    lalo = (cfg.reference_lalo or "").strip()
    stack = cfg.mintpy_dir / "inputs" / "ifgramStack.h5"
    geom = cfg.mintpy_dir / "inputs" / "geometryRadar.h5"
    if not lalo or not stack.exists():
        return [], {}
    lat0, lon0 = (float(v) for v in lalo.split(","))
    with h5py.File(stack, "r") as f:
        if "connectComponent" not in f:
            return [], {"note": "no connectComponent dataset"}
        dates = f["date"][:]
        d12 = [f"{a.decode()}_{b.decode()}" for a, b in dates]
        keep = f["dropIfgram"][:].astype(bool) if "dropIfgram" in f else np.ones(len(d12), bool)
        attrs = dict(f.attrs)
        if geom.exists() and "Y_FIRST" not in attrs:
            with h5py.File(geom, "r") as g:
                la, lo = g["latitude"][:], g["longitude"][:]
            try:
                y, x = mintpy_reference_yx(cfg, lat0, lon0)
            except Exception:                                  # noqa: BLE001
                y, x = np.unravel_index(np.argmin((la - lat0) ** 2 + (lo - lon0) ** 2), la.shape)
        else:
            y = int(round((lat0 - float(attrs["Y_FIRST"])) / float(attrs["Y_STEP"])))
            x = int(round((lon0 - float(attrs["X_FIRST"])) / float(attrs["X_STEP"])))
        cc = f["connectComponent"][:, int(y), int(x)]
        coh = f["coherence"][:, int(y), int(x)] if "coherence" in f else np.full(len(d12), np.nan)
        bad = [p for p, c, k in zip(d12, cc, keep) if k and c == 0]
        # MintPy's bridging does not test the raw component: it labels each
        # kept pair's components (skimage, connectivity 1), DROPS regions
        # smaller than connCompMinArea (2500 px, capped at 0.3% of the image)
        # and regions that vanish under a 5x5 erosion, and only then asks
        # whether the reference pixel is labelled. P2 passed the raw test on
        # 672 pairs and died in bridging on one whose reference component
        # was 2402 px. So the same labelling is applied here, with MintPy's
        # own class, to every kept pair that passed the raw test.
        small = []
        try:
            from mintpy.objects.conncomp import connectComponent
            meta = {"REF_Y": str(int(y)), "REF_X": str(int(x))}
            for i, (p, k) in enumerate(zip(d12, keep)):
                if not k or p in bad:
                    continue
                obj = connectComponent(conncomp=f["connectComponent"][i, :, :], metadata=meta)
                try:
                    obj.label(min_area=cc_min_area)
                except ValueError:
                    small.append(p)
        except ImportError:
            log.warning("mintpy not importable - reference check is the raw "
                        "component test only")
    info = {"ref_yx": [int(y), int(x)], "n_kept": int(keep.sum()),
            "ref_median_coherence": round(float(np.nanmedian(coh[keep])), 3),
            "pairs_ref_in_cc0": bad, "pairs_ref_component_too_small": small,
            "cc_min_area": cc_min_area}
    return bad + small, info


def _set_exclude_line(config_path: Path, pairs: list) -> None:
    lines = [l for l in config_path.read_text().splitlines()
             if not l.strip().startswith("mintpy.network.excludeDate12")]
    if pairs:
        lines.append(f"mintpy.network.excludeDate12 = {','.join(sorted(set(pairs)))}")
    config_path.write_text("\n".join(lines) + "\n")


def config_exclude_dates(cfg) -> list[str]:
    """The dates the config leaves out of the series (mintpy_exclude_dates), sorted."""
    return sorted({str(d).strip() for d in (getattr(cfg, "mintpy_exclude_dates", None) or [])})


def series_exclude_dates(cfg) -> list[str]:
    """Every date left out of the series: the config's (mintpy_exclude_dates)
    and the ones the weather check found no weather data for
    (weather_raider.registered_unavailable), sorted."""
    from .weather_raider import registered_unavailable
    return sorted(set(config_exclude_dates(cfg)) | set(registered_unavailable(cfg)))


def exclude_weather_unavailable(cfg: PipelineConfig, config_path: Path) -> list[str]:
    """Before the inversion, on the RAiDER route: fetch every stack date's
    weather (weather_raider.check_dates: cached cubes reused, a failure tried
    again from a clean download) and leave the dates still without any out of
    the series, re-running modify_network when that set changed. Until 27 Oct
    2026 a missing date stopped the run after the inversion and the user had
    to add it to mintpy_exclude_dates (Site E descending: three dates, none of
    them really unavailable - cut-off cached downloads). Returns the dates
    left out for want of weather."""
    if str(getattr(cfg, "mintpy_tropo_method", "")) != "raider":
        return []
    from . import weather_raider as wr
    before = set(wr.registered_unavailable(cfg))        # what the template was written with
    dates, _, _ = kept_network(cfg)
    now = wr.check_dates(cfg, [d for d in dates if d not in set(config_exclude_dates(cfg))])
    if set(now) != before:
        _set_exclude_date_line(config_path, series_exclude_dates(cfg))
        (cfg.mintpy_dir / "maskConnComp.h5").unlink(missing_ok=True)
        redo = _mintpy(cfg, config_path, "modify_network", "modify_network", "mintpy_network.log")
        if redo.returncode != 0:
            raise RuntimeError("MintPy modify_network failed after leaving out the dates without weather "
                               "data - see mintpy_network.log")
    return sorted(now)


def _set_exclude_date_line(config_path: Path, dates: list) -> None:
    lines = [l for l in config_path.read_text().splitlines()
             if not l.strip().startswith("mintpy.network.excludeDate ")
             and not l.strip().startswith("mintpy.network.excludeDate=")]
    if dates:
        lines.append(f"mintpy.network.excludeDate = {','.join(sorted(set(dates)))}")
    config_path.write_text("\n".join(lines) + "\n")


def kept_network(cfg: PipelineConfig) -> tuple[list, list, list]:
    """(dates, kept pairs, all formed pairs with (dt_days, bperp_m)) from
    ifgramStack.h5 after modify_network."""
    import h5py
    from datetime import datetime as _dt
    stack = cfg.mintpy_dir / "inputs" / "ifgramStack.h5"
    with h5py.File(stack, "r") as f:
        d = f["date"][:].astype(str)
        keep = f["dropIfgram"][:].astype(bool) if "dropIfgram" in f else np.ones(len(d), bool)
        bperp = f["bperp"][:] if "bperp" in f else np.zeros(len(d))
    formed = []
    for (a, b), k, bp in zip(d, keep, bperp):
        dt = (_dt.strptime(b, "%Y%m%d") - _dt.strptime(a, "%Y%m%d")).days
        formed.append((f"{a}_{b}", bool(k), int(dt), float(bp)))
    kept = [p for p, k, _, _ in formed if k]
    return sorted(set(d.ravel())), kept, formed


def ensure_connected_network(cfg: PipelineConfig, config_path: Path) -> dict:
    """
    Refuse to invert a network that is not one connected piece.

    MintPy only WARNS ("the network is NOT fully connected ... use SVD to
    resolve the offset between different subsets") and then inverts each
    piece against its own datum, which makes every rate that spans a break
    arbitrary. Measured on the production record: two acquisition gaps
    (Mar-May 2018, May-Jul 2024) were bridged only by pairs longer than the
    48-day tempBaseMax or wider than the 200 m perpBaseMax, every bridge was
    pruned, and 251 dates were inverted as three pieces of 35, 175 and 41.

    A date left with NO kept pair (typically one whose every pair failed the
    reference check) is excluded outright and modify_network re-run. Any
    remaining split is raised, naming the formed pairs that would bridge it
    and the threshold that removed them - that is a config decision, not
    something to relax silently.
    """
    from .stack_backend import check_network_connectivity
    # dates the config leaves out have no kept pair by design: not orphans
    by_config = set(series_exclude_dates(cfg))
    dates, kept, formed = kept_network(cfg)
    dates = [d for d in dates if d not in by_config]
    connected, orphans, comps = check_network_connectivity(dates, kept)
    info = {"dates": len(dates), "kept_pairs": len(kept), "orphaned_dates": orphans,
            "components": sorted((len(c) for c in comps), reverse=True), "excluded_dates": [],
            "excluded_by_config": sorted(by_config)}
    if orphans:
        log.warning("Date(s) with no kept interferogram: %s - excluding them and "
                    "rebuilding the network", ", ".join(orphans))
        _set_exclude_date_line(config_path, orphans + sorted(by_config))
        (cfg.mintpy_dir / "maskConnComp.h5").unlink(missing_ok=True)
        redo = _mintpy(cfg, config_path, "modify_network", "modify_network", "mintpy_network.log")
        if redo.returncode != 0:
            raise RuntimeError("MintPy modify_network failed after excluding dates - see "
                               "mintpy_network.log")
        info["excluded_dates"] = orphans
        dates, kept, formed = kept_network(cfg)
        dates = [d for d in dates if d not in orphans and d not in by_config]
        kept = [p for p in kept if not any(o in p for o in orphans)]
        connected, orphans, comps = check_network_connectivity(dates, kept)
        info.update(kept_pairs=len(kept), orphaned_dates=orphans,
                    components=sorted((len(c) for c in comps), reverse=True))
    (cfg.mintpy_dir / "network_connectivity.json").write_text(json.dumps(info, indent=2))
    if connected:
        log.info("Kept network: %d pairs over %d dates, one connected piece", len(kept), len(dates))
        return info
    cid = {d: i for i, c in enumerate(comps) for d in c}
    bridges = [(p, dt, bp) for p, k, dt, bp in formed
               if not k and cid.get(p[:8]) is not None and cid.get(p[9:]) is not None
               and cid[p[:8]] != cid[p[9:]]]
    lines = [f"  {p}  {dt} d  {bp:+.0f} m" for p, dt, bp in sorted(bridges)]
    raise RuntimeError(
        f"The kept network is {len(comps)} pieces ({info['components']} dates) and MintPy "
        "would invert each against its own datum. Formed pairs that bridge the pieces, all "
        f"dropped (tempBaseMax {cfg.max_temp_baseline} d, perpBaseMax {cfg.max_perp_baseline} m):\n"
        + "\n".join(lines) + "\nRaise max_temp_baseline / max_perp_baseline in the config to keep "
        "the bridges (weightFunc = var down-weights the low-coherence ones), or exclude the "
        "dates on the far side of a gap.")


def _unwrap_error_inputs(cfg: PipelineConfig) -> dict:
    """What MintPy's unwrap-error correction depends on: the reference pixel
    (bridging anchors every pair's components to the reference component)
    and the kept pair set (dropped pairs are copied through unbridged)."""
    import hashlib
    import h5py
    stack = cfg.mintpy_dir / "inputs" / "ifgramStack.h5"
    with h5py.File(stack, "r") as f:
        d = f["date"][:].astype(str)
        keep = f["dropIfgram"][:].astype(bool) if "dropIfgram" in f else np.ones(len(d), bool)
        ref = [str(f.attrs.get("REF_Y", "")), str(f.attrs.get("REF_X", ""))]
    kept = sorted(f"{a}_{b}" for (a, b), k in zip(d, keep) if k)
    return {"ref_yx": ref, "n_kept": len(kept),
            "kept_sha1": hashlib.sha1(",".join(kept).encode()).hexdigest()}


def refresh_unwrap_error_if_stale(cfg: PipelineConfig) -> float | None:
    """
    Make MintPy recompute the unwrap-error correction when its inputs changed.

    MintPy's run_or_skip for correct_unwrap_error compares dataset
    MODIFICATION_TIME stamps and the bridging parameters - not the reference
    pixel and not which pairs are kept. Measured on the production record: a
    re-run with a new reference pixel and 56 more kept pairs skipped the step
    in 1.3 s and inverted a bridging dataset computed for the old reference,
    with the newly kept pairs carried through unbridged. Bumping the input
    dataset's stamp is the one lever the skip test honours.

    Returns the bump time when a recompute was forced, else None.
    """
    import h5py
    state_file = cfg.mintpy_dir / "unwrap_error_state.json"
    now = _unwrap_error_inputs(cfg)
    old = json.loads(state_file.read_text()) if state_file.exists() else None
    stack = cfg.mintpy_dir / "inputs" / "ifgramStack.h5"
    with h5py.File(stack, "r") as f:
        have = [k for k in f.keys() if k.startswith("unwrapPhase_")]
    if old == now and have:
        log.info("Unwrap-error correction inputs unchanged (reference %s, %d kept pairs)",
                 now["ref_yx"], now["n_kept"])
        return None
    t = time.time()
    with h5py.File(stack, "r+") as f:
        f["unwrapPhase"].attrs["MODIFICATION_TIME"] = t
    log.info("Unwrap-error correction will be recomputed: %s",
             "no previous record" if old is None else
             f"reference {old['ref_yx']} -> {now['ref_yx']}, kept pairs {old['n_kept']} -> {now['n_kept']}")
    return t


def _requested_unwrap_method(config_path: Path) -> str:
    """The template's mintpy.unwrapError.method (the last line wins, as in
    MintPy's own reader); "" when none is set."""
    requested = ""
    for line in Path(config_path).read_text().splitlines():
        if line.strip().startswith("mintpy.unwrapError.method"):
            requested = line.split("=", 1)[1].split("#")[0].strip()
    return requested


def record_unwrap_error_state(cfg: PipelineConfig, bumped_at: float | None,
                              config_path: Path | None = None) -> None:
    """Verify the recompute happened (outcome, not the log) and record the
    inputs it used.

    After a forced recompute the FIRST step of the requested chain (bridging)
    must carry a stamp at or after the bump: that step always writes. A
    LATER step that is still older is not an error - MintPy's phase closure
    re-runs when its input is newer, finds no region unwrapped in every pair
    on this farmland, and returns without touching the dataset left by an
    earlier run (MintPy unwrap_error_phase_closure.py) - so it is reported
    and left to unwrap_error_applied, which never uses a dataset older than
    its input. Until 5 Oct 2026 that case raised 'did not recompute
    unwrapPhase_bridging_phaseClosure', blaming the wrong step."""
    import h5py
    stack = cfg.mintpy_dir / "inputs" / "ifgramStack.h5"
    if bumped_at is not None:
        with h5py.File(stack, "r") as f:
            outs = [k for k in f.keys() if k.startswith("unwrapPhase_")]
            stale = [k for k in outs if (_mod_time(f[k]) or 0.0) < bumped_at]
        if config_path is None:                       # the chain is unknown: every dataset must be new
            bad = not outs or bool(stale)
        else:
            parts = [m for m in _requested_unwrap_method(config_path).split("+") if m and m != "no"]
            # nothing requested ('no'): nothing to recompute, the inversion uses unwrapPhase
            bad = bool(parts) and (unwrap_error_dataset(parts[0]) not in outs
                                   or unwrap_error_dataset(parts[0]) in stale)
        if bad:
            raise RuntimeError(
                f"Unwrap-error correction did not recompute {stale or 'any dataset'} after its "
                "inputs changed - see mintpy_unwerr.log")
        if stale:
            log.warning("Unwrap-error step(s) not recomputed after the inputs changed: %s. MintPy found "
                        "nothing to correct there and left the earlier run's dataset in place; it will "
                        "not be used.", ", ".join(stale))
    (cfg.mintpy_dir / "unwrap_error_state.json").write_text(
        json.dumps(_unwrap_error_inputs(cfg), indent=2))


_UNWERR_SUFFIX = {"bridging": "bridging", "phase_closure": "phaseClosure"}

# Attribute the pipeline's own per-pixel loop-closure correction stamps on
# the dataset it writes (closure_integer_correction), so a later run can tell
# it from MintPy's phase_closure output of the same name.
PIPELINE_CLOSURE_ATTR = "PIPELINE_CLOSURE_CORRECTION"


def unwrap_error_dataset(method: str) -> str:
    """MintPy's dataset name for an unwrapError.method string."""
    parts = [m for m in method.split("+") if m and m != "no"]
    return "unwrapPhase" + "".join("_" + _UNWERR_SUFFIX.get(m, m) for m in parts)


def remove_rejected_closure_dataset(cfg: PipelineConfig) -> list:
    """
    Before correct_unwrap_error: when the pipeline's per-pixel loop-closure
    correction is OFF (mintpy_closure_correction false), remove a dataset it
    wrote in an earlier run.

    It carries MintPy's name for bridging+phase_closure, so MintPy's own
    closure step takes it for its output: when it is newer than the bridging
    dataset (the pipeline writes it straight after bridging) MintPy skips the
    step as up to date, and the inversion then used the variant Kyle rejected
    on 26-27 Sep, recorded as MintPy's 'bridging+phase_closure'. The
    delivered Site A stack still holds one. Recognised by the stamp
    closure_integer_correction now writes, or for older stacks by
    mintpy/closure_correction.json naming it. Removing it costs nothing that
    cannot be recomputed: MintPy's closure re-runs (and on this farmland finds
    nothing to do), and the pipeline's own takes seconds when switched back on.
    """
    import h5py
    if getattr(cfg, "mintpy_closure_correction", False):
        return []
    stack = cfg.mintpy_dir / "inputs" / "ifgramStack.h5"
    if not stack.exists():
        return []
    legacy = None
    rec_f = cfg.mintpy_dir / "closure_correction.json"
    if rec_f.exists():
        try:
            legacy = json.loads(rec_f.read_text()).get("dataset_out")
        except ValueError:
            legacy = None
    removed = []
    with h5py.File(stack, "r+") as f:
        for k in [k for k in f.keys() if k.startswith("unwrapPhase_")]:
            if f[k].attrs.get(PIPELINE_CLOSURE_ATTR) is not None or k == legacy:
                del f[k]
                removed.append(k)
    if removed:
        log.warning("Removed %s from ifgramStack.h5: written by the pipeline's per-pixel loop-closure "
                    "correction, which this config has off (mintpy_closure_correction: false) - MintPy "
                    "would otherwise take it for its own phase-closure output and invert it.",
                    ", ".join(removed))
    return removed


def unwrap_error_applied(cfg: PipelineConfig, config_path: Path) -> dict:
    """
    Which unwrap-error corrections actually produced a dataset, and make the
    template ask the inversion for THAT one.

    MintPy names outputs by chaining the methods (bridging ->
    unwrapPhase_bridging, then phase_closure -> unwrapPhase_bridging_
    phaseClosure). A step that finds nothing to do prints a warning and
    writes no dataset; the next step fails on the missing name. Here the
    longest prefix of the requested chain that exists on disk wins, the
    template's unwrapError.method is rewritten to match, and the outcome is
    written to mintpy/unwrap_error_applied.json.

    Present is not enough: a step's dataset counts only when its
    MODIFICATION_TIME is at or after its input's (the previous dataset in the
    chain, or unwrapPhase) - MintPy's own up-to-date rule. A dataset left by
    an earlier run that this run's inputs have overtaken (phase closure that
    found nothing to do this time, 27 Sep 2026 on Site A: closure stamped
    one second before the bridging it was meant to follow) is listed as stale
    and not used.
    """
    import h5py
    requested = _requested_unwrap_method(config_path)
    parts = [m for m in requested.split("+") if m and m != "no"]
    stack = cfg.mintpy_dir / "inputs" / "ifgramStack.h5"
    with h5py.File(stack, "r") as f:
        present = {k for k in f.keys() if k.startswith("unwrapPhase")}
        stamps = {k: _mod_time(f[k]) for k in present}
    applied, stale = [], []
    prev = "unwrapPhase"
    for m in parts:
        name = unwrap_error_dataset("+".join(applied + [m]))
        if name not in present:
            break
        t_in, t_out = stamps.get(prev), stamps.get(name)
        if t_in is not None and t_out is not None and t_out < t_in:
            stale.append(name)
            break
        applied.append(m)
        prev = name
    ds = unwrap_error_dataset("+".join(applied)) if applied else "unwrapPhase"
    method = "+".join(applied) if applied else "no"
    rec = {"requested": requested, "applied": method, "dataset": ds,
           "datasets_present": sorted(present)}
    if stale:
        rec["stale_not_used"] = stale
        log.warning("%s is older than the dataset it should follow (left by an earlier run; this run's "
                    "step wrote nothing) - not used.", ", ".join(stale))
    if method != requested:
        log.warning("Unwrap-error correction '%s' was requested but only '%s' produced a "
                    "dataset (present: %s). %s The inversion will use %s.", requested,
                    method, ", ".join(sorted(present)),
                    ("Phase closure needs a region unwrapped reliably in EVERY kept pair "
                     "(>= connCompMinArea px); none exists across this many pairs - a "
                     "property of the summer decorrelation, not a failure of the run."
                     if "phase_closure" in parts and "phase_closure" not in applied else ""),
                    ds)
        lines = [l for l in config_path.read_text().splitlines()
                 if not l.strip().startswith("mintpy.unwrapError.method")]
        lines.append(f"mintpy.unwrapError.method = {method}")
        config_path.write_text("\n".join(lines) + "\n")
    (cfg.mintpy_dir / "unwrap_error_applied.json").write_text(json.dumps(rec, indent=2))
    return rec


# The lines MintPy's phase-closure step prints about its common regions
# (unwrap_error_phase_closure.py), read back for the reason a step was skipped.
_CLOSURE_SKIP_LINES = ("remove regions with area <", "number of common regions:")


def unwrap_error_shortfall(cfg: PipelineConfig) -> dict | None:
    """
    The unwrap-error correction as the reviewer needs it: what the run asked
    MintPy for, what made the dataset the inversion used, and the steps that
    were asked for and made nothing ('skipped'), with MintPy's own reason
    from mintpy_unwerr.log. None when the run left no record.

    Read from mintpy/unwrap_error_applied.json (unwrap_error_applied), else
    from run_products.json's copy. 'requested' is the template's method,
    after the route has taken phase closure out of a network with no loops
    (pipeline.route_overrides, stack_backend._unwrap_error_method). That is
    decided before the run and logged. This record is about a step the run
    tried that wrote nothing.

    Until 21 Oct 2026 a skip was said only in provenance.json and one log
    WARNING. Three phase-linked projects asked for bridging+phase_closure.
    MintPy found no region unwrapped in every kept pair and inverted the
    bridging dataset. The QA, its send rule and run_result.json did not say
    so.
    """
    M = cfg.mintpy_dir
    rec, source = None, None
    for name, get in (("unwrap_error_applied.json", lambda d: d),
                      ("run_products.json", lambda d: d.get("unwrap_error"))):
        try:
            d = json.loads((M / name).read_text())
        except (OSError, ValueError):
            continue
        got = get(d) if isinstance(d, dict) else None
        if isinstance(got, dict) and got.get("requested") is not None:
            rec, source = got, f"mintpy/{name}"
            break
    if rec is None:
        return None
    requested = str(rec.get("requested") or "").strip()
    applied = str(rec.get("applied") or "no").strip()
    # MintPy reads 'auto' as 'no' (its default template)
    asked = [m for m in requested.split("+") if m and m not in ("no", "auto")]
    done = applied.split("+")
    skipped = [m for m in asked if m not in done]
    out = {"requested": requested, "applied": applied, "dataset": rec.get("dataset"),
           "skipped": skipped, "flag": bool(skipped), "source": source}
    for k in ("stale_not_used", "closure_correction"):
        if rec.get(k):
            out[k] = rec[k]
    if not skipped:
        return out
    why = []
    if "phase_closure" in skipped:
        try:
            lines = (M / "mintpy_unwerr.log").read_text(errors="replace").splitlines()
        except OSError:
            lines = []
        said = [l.strip() for l in lines if l.strip().startswith(_CLOSURE_SKIP_LINES)]
        area = next((m.group(1) for l in reversed(said) for m in [re.search(r"area < (\d+)", l)] if m), None)
        if said and said[-1].endswith("common regions: 0"):
            why.append("MintPy found no region" + (f" of at least {area} pixels" if area else "")
                       + " unwrapped in every kept pair (mintpy_unwerr.log: number of common regions: 0)")
        elif said:
            why.append("MintPy: " + "; ".join(said[-2:]) + " (mintpy_unwerr.log)")
    if rec.get("stale_not_used"):
        why.append(f"{', '.join(rec['stale_not_used'])} was left by an earlier run and not used")
    out["reason"] = " ".join(why) or "the step wrote no dataset this run (see mintpy_unwerr.log)"
    return out


def unwrap_error_skip_text(rec: dict | None) -> str | None:
    """One sentence for a skipped unwrap-error step (unwrap_error_shortfall),
    for the QA's send-rule notes and run_result.json's notes; None when
    nothing that was asked for was skipped."""
    if not rec or not rec.get("skipped"):
        return None
    steps = " and ".join(s.replace("_", "-") for s in rec["skipped"])
    return (f"the requested {steps} unwrapping-error correction was skipped (asked for {rec.get('requested')}, "
            f"applied {rec.get('applied')}; the inversion used {rec.get('dataset')}): {rec.get('reason')}")


def closure_integer_correction(cfg: PipelineConfig, dataset: str, block_rows: int = 48, max_iter: int = 4) -> dict:
    """
    Per-pixel whole-cycle correction of unwrapping errors from loop closure.

    Around a closed loop of three interferograms (a-b, b-c, a-c) the phases
    must sum to zero; after unwrapping, a non-zero whole number of cycles at
    a pixel means one of the three is off by that many cycles there. MintPy's
    phase_closure method solves this per pixel with an L1 integer program
    but, for speed, only on samples of regions unwrapped in EVERY pair, and
    applies one correction per region - on farmland that decorrelates every
    summer no such region exists, so it corrects nothing.

    This does it at every pixel with the simplest rule that is safe: an
    interferogram in two or more loops, ALL of which fail at the pixel and
    ALL of which are closed by the same correction, is corrected by that
    many cycles. Repeated until nothing changes. Anything ambiguous is left
    alone. Phases are compared after referencing to the reference pixel, as
    MintPy does; the correction is an exact multiple of 2 pi, so it moves no
    phase by less than a cycle. Written as the dataset MintPy's own
    bridging+phase_closure chain would produce, so the inversion reads it;
    the record is mintpy/closure_correction.json.
    """
    import h5py
    stack = cfg.mintpy_dir / "inputs" / "ifgramStack.h5"
    out_name = "unwrapPhase_bridging_phaseClosure" if dataset.endswith("bridging") else "unwrapPhase_phaseClosure"
    with h5py.File(stack, "r") as f:
        d12 = [f"{a.decode()}_{b.decode()}" for a, b in f["date"][:]]
        keep = f["dropIfgram"][:].astype(bool)
        ry, rx = int(f.attrs["REF_Y"]), int(f.attrs["REF_X"])
        n, L, W = f[dataset].shape
    kept = [i for i in range(n) if keep[i]]
    pos = {d12[i]: i for i in kept}
    dts = sorted({d for i in kept for d in d12[i].split("_")})
    tri = [(pos[f"{a}_{b}"], pos[f"{b}_{c}"], pos[f"{a}_{c}"])
           for i, a in enumerate(dts) for j, b in enumerate(dts[i + 1:], i + 1) for c in dts[j + 1:]
           if f"{a}_{b}" in pos and f"{b}_{c}" in pos and f"{a}_{c}" in pos]
    memb: dict = {}
    for t, (a, b, c) in enumerate(tri):
        memb.setdefault(a, []).append((t, 1)); memb.setdefault(b, []).append((t, 1)); memb.setdefault(c, []).append((t, -1))
    memb = {k: v for k, v in memb.items() if len(v) >= 2}
    two_pi = np.float32(2 * np.pi)
    stats = {"loops": len(tri), "interferograms_checked": len(memb), "pixels": 0,
             "loop_errors_before": 0, "loop_errors_after": 0, "cycles_corrected": 0, "ifg_pixels_corrected": 0}
    t0 = time.time()
    with h5py.File(stack, "r+") as f:
        src = f[dataset]
        if out_name in f:
            del f[out_name]
        dst = f.create_dataset(out_name, shape=src.shape, dtype=np.float32, chunks=src.chunks)
        dst.attrs.update(dict(src.attrs))
        ref = np.array(src[:, ry, rx], dtype=np.float32)
        for r0 in range(0, L, block_rows):
            blk = np.array(src[:, r0:r0 + block_rows, :], dtype=np.float32)        # (n, rows, W)
            U = blk.reshape(n, -1) - ref[:, None]
            valid = np.isfinite(U).all(axis=0) & (U != -ref[:, None]).any(axis=0)
            K = np.zeros((len(tri), U.shape[1]), np.int16)
            for t, (a, b, c) in enumerate(tri):
                K[t] = np.rint((U[a] + U[b] - U[c]) / two_pi).astype(np.int16)
            K[:, ~valid] = 0
            stats["loop_errors_before"] += int((K != 0).sum()); stats["pixels"] += int(valid.sum())
            corr = np.zeros((n, U.shape[1]), np.int16)
            for _ in range(max_iter):
                changed = 0
                for k, mt in memb.items():
                    us = np.stack([-K[t] * s for t, s in mt])
                    ok = (us[0] != 0) & (us == us[0]).all(axis=0)
                    if ok.any():
                        u = np.where(ok, us[0], 0).astype(np.int16)
                        corr[k] += u
                        for t, s in mt:
                            K[t] += (s * u).astype(np.int16)
                        changed += int(ok.sum())
                if not changed:
                    break
            stats["loop_errors_after"] += int((K != 0).sum())
            stats["cycles_corrected"] += int(np.abs(corr).sum()); stats["ifg_pixels_corrected"] += int((corr != 0).sum())
            dst[:, r0:r0 + block_rows, :] = (blk.reshape(n, -1) + two_pi * corr).reshape(blk.shape)
        dst.attrs["MODIFICATION_TIME"] = str(time.time())
        # tells this dataset from MintPy's phase_closure output of the same
        # name (remove_rejected_closure_dataset)
        dst.attrs[PIPELINE_CLOSURE_ATTR] = "per-pixel greedy loop-closure integer correction (time_series.py)"
    tot = max(1, stats["pixels"] * len(tri))
    stats.update({"dataset_in": dataset, "dataset_out": out_name, "seconds": round(time.time() - t0, 1),
                  "percent_loops_in_error_before": round(100 * stats["loop_errors_before"] / tot, 2),
                  "percent_loops_in_error_after": round(100 * stats["loop_errors_after"] / tot, 2),
                  "percent_ifg_pixels_corrected": round(100 * stats["ifg_pixels_corrected"] / max(1, stats["pixels"] * len(kept)), 2),
                  "method": "per-pixel greedy loop-closure integer correction (pipeline; see time_series.closure_integer_correction)"})
    (cfg.mintpy_dir / "closure_correction.json").write_text(json.dumps(stats, indent=2))
    log.info("Loop-closure correction: loops in error %.1f%% -> %.1f%%, %.2f%% of interferogram-pixels corrected, %s -> %s (%.0f s)",
             stats["percent_loops_in_error_before"], stats["percent_loops_in_error_after"],
             stats["percent_ifg_pixels_corrected"], dataset, out_name, stats["seconds"])
    return stats


# MintPy's network temporal coherence below which a cell's unwrapped network
# is random: on the 3-connection linked network the interferograms close by
# construction, so this residual measures unwrapping consistency, and pure
# noise (open water) scores below it. Site A's Gibson Lake: MintPy TC
# < 0.25 and linked TC p5/p50/p95 0.360/0.373/0.388 (finding, 5 Oct 2026).
UNWRAP_NOISE_TC = 0.25
NOISE_MIN_CELLS = 500
MINTPY_TC_SOURCE = ("MintPy network-inversion temporal coherence (unwrapping consistency: on the "
                    "phase-linked network the interferograms close by construction, so its residual "
                    "measures unwrapping errors)")


def linked_noise_level(linked: np.ndarray, mintpy_tc: np.ndarray, min_cells: int = NOISE_MIN_CELLS) -> dict:
    """
    The linked (mini-stack) temporal coherence that pure noise reaches, for
    THIS run - not a constant, because it depends on the mini-stack size and
    the looks.

    Noise cells are those whose unwrapped network is random (MintPy TC below
    UNWRAP_NOISE_TC). Among them the linked coherence has a sharp mode at the
    noise level plus an upper tail of coherent ground with unwrapping errors,
    so the level is read from the mode and the LOWER half only (mirrored):
    level = mode + 1.645 x sigma, sigma the 68th percentile of the distances
    below the mode - the one-sided 95% point of the noise population, blind
    to the tail. Measured (5 Oct 2026): Site A 0.387 (Gibson Lake's own
    95th percentile 0.386), Site B 0.384 (water-like cells: 0.388). The
    working threshold of 0.4 is therefore only ~0.015 above noise.

    {"level": None, "reason": ...} with fewer than `min_cells` noise cells
    (e.g. a single-reference star, where MintPy TC is 1 everywhere).
    """
    lk = np.asarray(linked, dtype=np.float64)
    mt = np.asarray(mintpy_tc, dtype=np.float64)
    cand = np.isfinite(lk) & (lk > 0) & np.isfinite(mt) & (mt < UNWRAP_NOISE_TC)
    n = int(cand.sum())
    out = {"level": None, "noise_cells": n, "mintpy_tc_below": UNWRAP_NOISE_TC,
           "method": ("mode + 1.645 sigma of the linked temporal coherence of cells whose MintPy "
                      f"(unwrapping-consistency) temporal coherence is below {UNWRAP_NOISE_TC}; sigma from "
                      "the half below the mode")}
    if n < min_cells:
        out["reason"] = (f"only {n} cells with MintPy temporal coherence below {UNWRAP_NOISE_TC} "
                         f"(need {min_cells}) - no noise population to measure on this stack")
        return out
    v = lk[cand]
    h, e = np.histogram(v, bins=500, range=(0.0, 1.0))
    hs = np.convolve(h, np.ones(5) / 5.0, mode="same")
    mode = float(e[int(np.argmax(hs))] + 0.001)
    below = mode - v[v < mode]
    if below.size < min_cells // 2:
        out["reason"] = f"only {below.size} noise cells below the mode {mode:.3f} - the lower half is too thin"
        return out
    sigma = float(np.percentile(below, 68.27))
    out.update(level=round(mode + 1.645 * sigma, 4), mode=round(mode, 4), sigma=round(sigma, 4))
    return out


def apply_linked_temporal_coherence(cfg: PipelineConfig) -> None:
    """
    Replace MintPy's temporal coherence and its mask with MiaplPy's phase-
    linking temporal coherence on the phase-linked route.

    Written for the single-reference star, where MintPy's inversion is
    exactly determined and temporalCoherence.h5 is 1 at every pixel. The
    export is now a 3-connection network, and there MintPy's temporal
    coherence is meaningful: the linked interferograms close by construction,
    so its inversion residual measures UNWRAPPING consistency, which the
    linked (mini-stack) coherence cannot see. It is therefore kept, as
    temporalCoherence_mintpy.h5 beside the adopted one (MiaplPy's own chain
    does the same), before the linked coherence is written over it. Written
    before the deramp mask, so every later step, the geocoding, the
    deliverables and the assessment see the adopted layer.
    """
    import h5py
    from .miaplpy_backend import LINKED_COHERENCE_LAYER, linked_temporal_coherence_path, load_linked_temporal_coherence
    src = linked_temporal_coherence_path(cfg)
    if not src.exists():
        raise RuntimeError(f"{src} missing - miaplpy_backend.export_linked_temporal_coherence did not run")
    tc_file = cfg.mintpy_dir / "temporalCoherence.h5"; mask_file = cfg.mintpy_dir / "maskTempCoh.h5"
    with h5py.File(tc_file, "r") as f:
        a = dict(f.attrs); shape = f["temporalCoherence"].shape
        adopted_before = "temporalCoherenceSource" in f.attrs
    keep = cfg.mintpy_dir / "temporalCoherence_mintpy.h5"
    if not adopted_before:                     # still MintPy's own: keep it before overwriting
        import shutil
        shutil.copy2(tc_file, keep)
        log.info("Kept MintPy's own temporal coherence (unwrapping consistency on this network) as %s", keep)
    y0 = int(a.get("SUBSET_YMIN", 0)); x0 = int(a.get("SUBSET_XMIN", 0))
    tc = load_linked_temporal_coherence(cfg)[y0:y0 + shape[0], x0:x0 + shape[1]]
    if tc.shape != tuple(shape):
        raise RuntimeError(f"linked temporal coherence {tc.shape} does not cover MintPy's grid {shape} at offset {y0},{x0}")
    tc = np.nan_to_num(tc, nan=0.0).astype(np.float32)
    mask = tc >= float(cfg.min_temporal_coherence)
    # Say which coherence each file holds, so a GIS user (and the GeoTIFF
    # export) never sets the linked layer beside MintPy's as like for like.
    noise = {"level": None, "reason": f"{keep.name} missing"}
    if keep.exists():
        with h5py.File(keep, "r+") as f:
            f.attrs["temporalCoherenceSource"] = MINTPY_TC_SOURCE
            mtc = f["temporalCoherence"][:]
        if mtc.shape == tc.shape:
            noise = linked_noise_level(tc, mtc)
    thr = float(cfg.min_temporal_coherence)
    level = noise.get("level")
    # MintPy keeps attributes as strings; "nan" = not measurable on this stack
    noise_attrs = {"linked_noise_level": "nan" if level is None else str(level),
                   "linked_noise_method": noise.get("reason") or noise.get("method", ""),
                   "threshold_minus_noise": "nan" if level is None else str(round(thr - level, 4))}
    with h5py.File(tc_file, "r+") as f:
        f["temporalCoherence"][:] = tc
        f.attrs["temporalCoherenceSource"] = (f"MiaplPy phase linking, {LINKED_COHERENCE_LAYER} (mini-stack) temporal "
                                              "coherence, block-averaged over the looks - not comparable with MintPy's")
        for k, v in noise_attrs.items():
            f.attrs[k] = v
    with h5py.File(mask_file, "r+") as f:
        f["mask"][:] = mask
        f.attrs["temporalCoherenceSource"] = f"MiaplPy phase linking, {LINKED_COHERENCE_LAYER} (mini-stack) temporal coherence"
        f.attrs["threshold"] = str(cfg.min_temporal_coherence)
        for k, v in noise_attrs.items():
            f.attrs[k] = v
    log.info("Adopted MiaplPy's linked temporal coherence: median %.3f, mask keeps %.1f%% of the grid at >= %.2f",
             float(np.median(tc[tc > 0])) if np.any(tc > 0) else float("nan"),
             100 * float(mask.mean()), cfg.min_temporal_coherence)
    if level is None:
        log.warning("Linked-coherence noise level not measurable: %s", noise.get("reason"))
    elif thr - level < 0.05:
        log.warning("The reliability threshold %.2f is only %.3f above the measured noise level of the linked "
                    "coherence (%.3f, from %d cells whose unwrapping is random): ground decorrelated for part "
                    "of the record can pass it. See geo_disturbed.h5 and geo_temporalCoherence_mintpy.h5.",
                    thr, thr - level, level, noise["noise_cells"])
    else:
        log.info("Linked-coherence noise level %.3f; threshold %.2f sits %.3f above it", level, thr, thr - level)


NISAR_TC_SOURCE = ("mean interferometric coherence over the NISAR GUNW pairs used (nisar_gunw/coherence_gate.tif "
                   "band 1): a 12-day chain has no redundant pairs, so MintPy's temporal coherence is 1.0 by "
                   "construction and tests nothing")


NISAR_PL_TC_SOURCE = ("dolphin's phase-linking temporal coherence over the NISAR L2 GSLC stack "
                      "(nisar_gunw/phase_linked_temporal_coherence.tif; the phase-linked NISAR route, nisar_pl.py)")


def apply_nisar_pair_coherence(cfg: PipelineConfig) -> bool:
    """
    Replace MintPy's temporal coherence and its mask with the mean pair
    coherence on the NISAR route. A GUNW chain is exactly determined, so
    MintPy's temporal coherence is 1.0 at every pixel: until 12 Oct 2026 every
    reliability mask, uncertainty class, QA check and point mask downstream
    read that 1.0 and screened nothing (A5a, Appin). The validation harness
    already used the pair coherence; this makes the products agree with it.

    MintPy's own layer is kept as temporalCoherence_mintpy.h5, as on the
    phase-linked route. Idempotent: a layer already carrying NISAR_TC_SOURCE
    is left alone, so the products stage can call it on finished runs.
    Returns True when the layer now holds the pair coherence.
    """
    import shutil
    import h5py
    import rasterio
    from . import lband
    gate = Path(cfg.project_dir) / "nisar_gunw" / "coherence_gate.tif"
    source = NISAR_TC_SOURCE
    if bool(getattr(cfg, "nisar_phase_linked", False)):
        # the phase-linked route: dolphin's temporal coherence (nisar_pl)
        gate, source = Path(cfg.project_dir) / "nisar_gunw" / "phase_linked_temporal_coherence.tif", NISAR_PL_TC_SOURCE
    tc_file = cfg.mintpy_dir / "temporalCoherence.h5"
    mask_file = cfg.mintpy_dir / "maskTempCoh.h5"
    if not tc_file.exists():
        return False
    with h5py.File(tc_file, "r") as f:
        a = {k: (v.decode() if isinstance(v, bytes) else v) for k, v in f.attrs.items()}
        shape = f["temporalCoherence"].shape
    if str(a.get("temporalCoherenceSource", "")) == source:
        return True
    if not gate.exists():
        log.warning("NISAR run without %s: temporalCoherence.h5 stays MintPy's (1.0 on a chain), so the "
                    "products' reliability masks screen nothing", gate)
        return False
    with rasterio.open(gate) as src:
        mean = src.read(1).astype(float)
        t = src.transform
    src_grid = lband.Grid(float(t.c), float(t.f), float(t.a), float(t.e), mean.shape[1], mean.shape[0])
    tc = lband.resample_to(mean, src_grid, lband.Grid.from_attrs(a, shape))
    keep = cfg.mintpy_dir / "temporalCoherence_mintpy.h5"
    if not keep.exists():
        shutil.copy2(tc_file, keep)
        with h5py.File(keep, "r+") as f:
            f.attrs["temporalCoherenceSource"] = ("MintPy network-inversion temporal coherence: 1.0 by "
                                                  "construction on a NISAR 12-day chain (no redundancy)")
    tc = np.nan_to_num(tc, nan=0.0).astype(np.float32)
    mask = tc >= float(cfg.min_temporal_coherence)
    with h5py.File(tc_file, "r+") as f:
        f["temporalCoherence"][:] = tc
        f.attrs["temporalCoherenceSource"] = source
    if mask_file.exists():
        with h5py.File(mask_file, "r+") as f:
            f["mask"][:] = mask
            f.attrs["temporalCoherenceSource"] = source
            f.attrs["threshold"] = str(cfg.min_temporal_coherence)
    log.info("Adopted the NISAR mean pair coherence as temporal coherence: median %.3f, mask keeps %.1f%% "
             "of the grid at >= %.2f", float(np.median(tc[tc > 0])) if np.any(tc > 0) else float("nan"),
             100 * float(mask.mean()), cfg.min_temporal_coherence)
    return True


def deramp_mask_step(cfg: PipelineConfig) -> Path | None:
    """
    Build the deramp mask and point MintPy at it. Called between
    invert_network and deramp - the only window in which maskTempCoh.h5
    exists and the deramp has not yet read its mask.

    Not optional when deramp is on. Without it MintPy fits the ramp through
    every coherent pixel INCLUDING the deforming area, then subtracts that
    biased plane from the deformation. This used to live in the launcher
    scripts, in two copies, one of them without the deramp guard.
    """
    from . import stack_backend as sb
    if getattr(cfg, "mintpy_deramp", "no") == "no":
        return None
    buffer_m = float(cfg.deramp_buffer_m)
    if getattr(cfg, "mintpy_deramp_mask", ""):
        path = Path(cfg.mintpy_deramp_mask)
        if not path.exists():
            raise FileNotFoundError(f"mintpy_deramp_mask {path} does not exist")
        built = False
    else:
        path = sb.build_deramp_mask(cfg, buffer_m=buffer_m)
        if path is None:
            raise RuntimeError("deramp mask could not be built: maskTempCoh.h5 "
                               "is missing after the inversion")
        built = True
    rec = check_deramp_mask(cfg, path, built=built, buffer_m=buffer_m)
    log.info("Deramp fit uses %d of %d coherent pixels (%.0f%%); %d excluded", rec["kept_pixels"],
             rec["coherent_pixels"], 100 * rec["kept_fraction"], rec["excluded_pixels"])
    cfgp = cfg.mintpy_dir / "smallbaselineApp.cfg"
    lines = [l for l in cfgp.read_text().splitlines()
             if not l.strip().startswith("mintpy.deramp.maskFile")]
    lines.append(f"mintpy.deramp.maskFile = {path}")
    cfgp.write_text("\n".join(lines) + "\n")
    log.info("Deramp will use %s", path)
    return path


# Below this share of the coherent pixels the plane is fitted on a strip and
# extrapolated across the excluded ground (stack_backend.build_deramp_mask
# warned at the same 20% and carried on).
DERAMP_MIN_KEPT_FRACTION = 0.2


def check_deramp_mask(cfg: PipelineConfig, path: Path, built: bool = True, buffer_m: float | None = None) -> dict:
    """
    What the deramp fit will actually use, measured on the mask (outcome),
    written to mintpy/deramp_mask.json, and refused when it cannot do its job.

    A BUILT mask must exclude something and leave enough to fit on:
      - 0 coherent pixels excluded means the polygon does not reach this
        grid - e.g. a hand-written config with no deramp_exclude_geojson
        picking up the package's Gibson polygon at another site - and the
        ramp would be fitted straight through any mine in the scene;
      - fewer than DERAMP_MIN_KEPT_FRACTION of the coherent pixels kept
        means a plane fitted on a strip and extrapolated over the excluded
        ground.
    Both used to be a log line in a multi-hour log. A mask the config names
    (mintpy_deramp_mask) is the user's own decision: recorded and warned
    about, not refused.
    """
    import h5py
    with h5py.File(cfg.mintpy_dir / "maskTempCoh.h5", "r") as f:
        coh = f["mask"][:].astype(bool)
    with h5py.File(path, "r") as f:
        key = "mask" if "mask" in f else sorted(f.keys())[0]
        dm = f[key][:].astype(bool)
    if dm.shape != coh.shape:
        raise RuntimeError(f"deramp mask {path} is {dm.shape}, maskTempCoh.h5 is {coh.shape}")
    n_coh = int(coh.sum())
    kept = int((dm & coh).sum())
    excl = n_coh - kept
    frac = kept / max(n_coh, 1)
    rec = {"mask": str(path), "built_by_pipeline": bool(built),
           "exclude_geojson": str(getattr(cfg, "deramp_exclude_geojson", "") or ""),
           "buffer_m": buffer_m, "coherent_pixels": n_coh, "kept_pixels": kept,
           "excluded_pixels": excl, "kept_fraction": round(frac, 4)}
    (cfg.mintpy_dir / "deramp_mask.json").write_text(json.dumps(rec, indent=2))
    problem = None
    if excl == 0:
        problem = ("excludes NO coherent pixel - the exclusion polygon (+ buffer) does not reach this grid, so "
                   "the ramp would be fitted through any deformation in the scene and subtracted from it. Point "
                   "deramp_exclude_geojson at the mines/deforming area of THIS site, or set mintpy_deramp: no. "
                   "(If the mapped area really is all below the coherence threshold, name a mask explicitly "
                   "with mintpy_deramp_mask)")
    elif frac < DERAMP_MIN_KEPT_FRACTION:
        problem = (f"leaves only {100 * frac:.0f}% of the coherent pixels ({kept} of {n_coh}) to fit the ramp "
                   f"on (minimum {100 * DERAMP_MIN_KEPT_FRACTION:.0f}%): the plane would be fitted on a strip "
                   "and extrapolated across the excluded ground. Reduce deramp_buffer_m "
                   f"(now {buffer_m} m), or set mintpy_deramp: no and judge the tilt from deramp_effect.json")
    if problem:
        if built:
            raise RuntimeError(f"The deramp mask {Path(path).name} {problem}.")
        log.warning("The configured deramp mask %s %s.", path, problem)
    return rec


def inverted_dataset(cfg: PipelineConfig, names: list[str]) -> str:
    """The unwrapped-phase dataset the inversion actually used: from
    unwrap_error_applied.json, else the product's obsDatasetName attribute,
    else the most-corrected name present.

    NOT simply the longest name in the stack: a stack reused across runs can
    carry a dataset from an earlier variant that this run did not invert
    (27 Sep 2026: the rejected loop-closure variant's
    unwrapPhase_bridging_phaseClosure stayed in the stack of the delivered
    bridging-only run, and the closure diagnostic reported its statistics)."""
    rec = cfg.mintpy_dir / "unwrap_error_applied.json"
    if rec.exists():
        try:
            ds = json.loads(rec.read_text()).get("dataset")
            if ds in names:
                return ds
        except ValueError:
            pass
    try:
        import h5py
        for f in (cfg.mintpy_dir / "geo" / "geo_velocity.h5", cfg.mintpy_dir / "velocity.h5"):
            if f.exists():
                with h5py.File(f, "r") as h:
                    ds = h.attrs.get("mintpy.networkInversion.obsDatasetName")
                ds = ds.decode() if isinstance(ds, bytes) else ds
                if ds in names:
                    return ds
    except Exception:                                         # noqa: BLE001
        pass
    return sorted(names, key=len)[-1]


def phase_closure_diagnostic(cfg: PipelineConfig) -> Path | None:
    """
    Count triplets with non-zero integer ambiguity on the CORRECTED phase.

    MintPy's quick_overview step writes this, but on the uncorrected
    unwrapPhase and with a histogram that crashes when the count is zero
    everywhere - the best case. So the step is skipped and the calculation
    done here directly, on the dataset the inversion actually used. Output:
    numTriNonzeroIntAmbiguity4<dataset>.h5 (MintPy's own naming); the
    uncorrected one is written too so the correction can be judged.
    """
    from . import procs
    stack = cfg.mintpy_dir / "inputs" / "ifgramStack.h5"
    if not stack.exists():
        return None
    try:
        import h5py
        with h5py.File(stack, "r") as f:
            names = [k for k in f.keys() if k.startswith("unwrapPhase")]
    except Exception as exc:                                  # noqa: BLE001
        log.warning("Could not open %s for the closure diagnostic: %s", stack, exc)
        return None
    corrected = inverted_dataset(cfg, names)
    written = None
    for ds in {"unwrapPhase", corrected}:
        argv = ["unwrap_error_phase_closure.py", str(stack), "--action", "calculate",
                "-i", ds, "--cc-mask", str(cfg.mintpy_dir / "maskConnComp.h5")]
        r = procs.run_group(argv, cwd=cfg.mintpy_dir, timeout=1800)
        out = cfg.mintpy_dir / ("numTriNonzeroIntAmbiguity.h5" if ds == "unwrapPhase"
                                else f"numTriNonzeroIntAmbiguity4{ds}.h5")
        if out.exists():
            log.info("Phase-closure diagnostic for %s: %s", ds, out.name)
            if ds == corrected:
                written = out
        else:
            log.warning("Phase-closure diagnostic for %s not written (rc %s): %s",
                        ds, r.returncode, (r.stderr or r.stdout or "")[-400:])
    return written


def clear_stale_mintpy_products(cfg: PipelineConfig, stamp: str | None = None) -> list:
    """
    With no inputs/ifgramStack.h5 the stack is being rebuilt, so derived
    products at the previous extent must go. MintPy writes some into the
    work-directory root (waterMask.h5, avgSpatialCoh.h5...), and a failed
    earlier load can leave inputs/geometryRadar.h5 with no stack beside it;
    MintPy then skips loading geometry and pairs old geometry with a new
    stack. Only the top level of mintpy/ and mintpy/inputs/ is cleared -
    never superseded/ or any other subdirectory.

    The previous run's own products among them (radar velocity, corrected
    series, DEM error: _DOWNSTREAM_GLOBS) are MOVED, not deleted, into
    mintpy/superseded/run_<stamp>/ - the folder retire_downstream_products
    fills with the rest of that run later on (pass the run's stamp). Until
    7 Oct 2026 a re-unwrap deleted them here, so the before/after comparison
    had no radar velocity or series although the re-unwrap's note said they
    were set aside. Returns every file cleared from the top level.
    """
    import fnmatch
    stack = cfg.mintpy_dir / "inputs" / "ifgramStack.h5"
    if stack.exists():
        return []
    top = sorted(cfg.mintpy_dir.glob("*.h5"))
    keep = [f for f in top if f.is_file() and any(fnmatch.fnmatch(f.name, g) for g in _DOWNSTREAM_GLOBS)]
    if keep:
        dest = _retire_files(cfg, keep, stamp or time.strftime("%Y%m%d_%H%M%S"))
        log.warning("Reloading the stack: moved the previous run's %d radar products (%s) aside to %s",
                    len(keep), ", ".join(p.name for p in keep), dest)
    stale = [f for f in top if f not in keep] + [f for f in (cfg.mintpy_dir / "inputs").glob("*.h5")]
    for f in stale:
        f.unlink(missing_ok=True)
    if stale:
        log.info("Cleared %d stale MintPy products before reloading", len(stale))
    return keep + stale


def retire_stale_ifgram_stack(cfg: PipelineConfig, loader_lines: str) -> Path | None:
    """
    MintPy's loader skips loading when inputs/ifgramStack.h5 looks complete -
    it compares the file list and sizes, not the contents - so after a
    re-unwrap it would silently invert the OLD unwrapped phase. If any file
    the loader's unwFile or corFile pattern matches is newer than the stack,
    the stack is renamed aside (kept, not deleted) and MintPy loads afresh.
    The coherence counts too (7 Oct 2026): rewritten in place (phasor ->
    sample pair coherence) it left the stack's old 'coherence' dataset in use.
    """
    import glob
    stack = cfg.mintpy_dir / "inputs" / "ifgramStack.h5"
    m = re.search(r"^mintpy\.load\.unwFile\s*=\s*(\S+)", loader_lines, flags=re.MULTILINE)
    if not stack.exists() or not m:
        return None
    files = glob.glob(m.group(1))
    if not files:
        return None
    c = re.search(r"^mintpy\.load\.corFile\s*=\s*(\S+)", loader_lines, flags=re.MULTILINE)
    files += glob.glob(c.group(1)) if c else []
    newest = max(Path(f).stat().st_mtime for f in files)
    if newest <= stack.stat().st_mtime:
        return None
    # OUTSIDE mintpy/ and mintpy/inputs/: with ifgramStack.h5 gone,
    # clear_stale_mintpy_products() clears every *.h5 in both (27 Sep 2026:
    # a set-aside stack kept in inputs/ was deleted that way seconds later)
    keep_dir = cfg.mintpy_dir / "superseded"
    keep_dir.mkdir(exist_ok=True)
    aside = keep_dir / f"ifgramStack_{time.strftime('%Y%m%d_%H%M')}.h5"
    stack.rename(aside)
    log.warning("Unwrapped interferograms (or their coherence) are newer than %s: set it aside as %s so MintPy "
                "reloads them (delete it once the new run is accepted)", stack.name, aside.name)
    return aside


# What MintPy's correct_LOD onward (re)writes into the work directory, plus
# the pipeline's own records of a finished run. Everything in geo/ goes too.
# NOT: timeseries.h5, temporalCoherence*.h5, maskTempCoh.h5, numInvIfgram.h5
# (the inversion of THIS run wrote them), maskDeramp.h5 and the closure
# diagnostics (written between the inversion and correct_LOD), inputs/
# (ERA5.h5 is the weather cache MintPy reuses - whatever dates it holds, so
# retire_partial_weather_file moves one that misses a date).
_DOWNSTREAM_GLOBS = ("timeseries_*.h5", "timeseriesResidual*.h5", "velocity*.h5", "demErr.h5",
                     "rms_timeseriesResidual*.txt", "reference_date.txt", "*.he5",
                     "run_products.json", "deramp_effect.json", "disturbed.h5")


def retire_partial_weather_file(cfg: PipelineConfig, weather_file: Path, stamp: str) -> Path | None:
    """
    Before MintPy's troposphere step: move inputs/<MODEL>.h5 into
    superseded/run_<stamp>/ when it does not hold every date of
    timeseries.h5. MintPy applies whatever such file it finds on the
    series' grid ("Use existed tropospheric delay file"), whatever its
    dates. One built with ERA5 unreachable held NO dates, and every re-run
    after it died in diff.py ("IndexError: list index out of range", 23-24
    Oct 2026); a partial one corrects only its dates. Moved aside, MintPy
    (or weather_raider) builds it again from the weather files on disk.
    Returns the new place, or None when the file was kept or absent.
    """
    ts = cfg.mintpy_dir / "timeseries.h5"
    if not weather_file.exists() or not ts.exists():
        return None
    try:
        need = set(corrections._dates(ts))
        have = set(corrections._dates(weather_file))
    except (OSError, KeyError) as exc:
        have, need = set(), {"?"}
        log.warning("Could not read the dates of %s (%s)", weather_file.name, exc)
    if need <= have:
        return None
    dest = _retire_files(cfg, [weather_file], stamp)
    log.warning("%s held %d of the series' %d dates; moved it to %s so it is built again rather than applied "
                "as it is", weather_file.name, len(need & have), len(need), dest)
    return dest / weather_file.relative_to(cfg.mintpy_dir)


def retire_downstream_products(cfg: PipelineConfig, stamp: str) -> Path | None:
    """
    Before correct_LOD onward: RENAME the previous run's corrected series,
    velocities, DEM error and the whole of geo/ into
    mintpy/superseded/run_<stamp>/ (same relative layout; never deleted).

    MintPy's update mode and the deliverables both took an existing file for
    a current one, so a re-run that failed after the inversion (ERA5 queue,
    disk, memory) was delivered from the PREVIOUS run's geo products, and a
    re-run with a correction switched off lost to last week's file with more
    tokens in its name. Once these are moved, every product this run is
    judged on can only have come from this run; a failure leaves geo/ empty
    and "Products only" fails loudly instead of rebuilding old numbers. Same
    filesystem, so a rename: no copy, no extra disk - but the folder keeps
    several GB per re-run (geo_timeseries is 3.8-6.9 GB): delete it once the
    new run is accepted, or move it back by hand to undo.
    """
    M = cfg.mintpy_dir
    moves = sorted({p for g in _DOWNSTREAM_GLOBS for p in M.glob(g) if p.is_file()})
    geo = M / "geo"
    if geo.is_dir():
        moves += sorted(p for p in geo.rglob("*") if p.is_file())
    dest = M / "superseded" / f"run_{stamp}"
    if not moves:
        # the radar products may already be there (clear_stale_mintpy_products)
        return dest if (dest / "RETIRED.json").exists() else None
    _retire_files(cfg, moves, stamp)
    gb = sum((dest / p.relative_to(M)).stat().st_size for p in moves) / 1e9
    log.warning("Moved the previous run's %d MintPy products (%.1f GB, incl. geo/) aside to %s, so nothing "
                "from them can pass for this run's. Delete that folder once this run is accepted.",
                len(moves), gb, dest)
    return dest


def _retire_files(cfg: PipelineConfig, moves: list, stamp: str) -> Path:
    """Rename files under mintpy/ into mintpy/superseded/run_<stamp>/ (same
    relative layout) and list them in its RETIRED.json, adding to what an
    earlier step of the same run already put there."""
    M = cfg.mintpy_dir
    dest = M / "superseded" / f"run_{stamp}"
    for p in moves:
        tgt = dest / p.relative_to(M)
        tgt.parent.mkdir(parents=True, exist_ok=True)
        os.rename(p, tgt)
    left = [p for p in moves if p.exists()]
    if left:
        raise RuntimeError(f"Could not move the previous run's products aside: {', '.join(map(str, left))}")
    try:
        files = list(json.loads((dest / "RETIRED.json").read_text()).get("files") or [])
    except (OSError, ValueError):
        files = []
    files += [str(p.relative_to(M)) for p in moves if str(p.relative_to(M)) not in files]
    gb = sum((dest / f).stat().st_size for f in files if (dest / f).is_file()) / 1e9
    (dest / "RETIRED.json").write_text(json.dumps({
        "stamp": stamp, "moved_from": str(M), "files": files,
        "gb": round(gb, 2),
        "why": "superseded by the MintPy run started at this stamp (time_series.retire_downstream_products); "
               "delete once that run is accepted, or move back to undo"}, indent=2))
    return dest


def verify_science_products(cfg: PipelineConfig, config_path: Path, since: float) -> dict:
    """
    After correct_LOD..geocode: every product the deliverables read exists and
    was written by this step - the chain's final series (MintPy's own name for
    it, chain_files), velocity.h5 and, when MintPy geocodes, geo_velocity,
    geo_<final series>, geo_temporalCoherence and geo_maskTempCoh. Exactly
    one geocoded series may exist. Raises otherwise; returns the paths.
    """
    import h5py
    M = cfg.mintpy_dir
    chain = chain_files(cfg, config_path)
    final = chain["final"]
    with h5py.File(M / "timeseries.h5", "r") as f:
        geocoded_input = "Y_FIRST" in f.attrs
    need = [final, M / "velocity.h5"]
    out = {"chain": chain, "timeseries_radar": None if geocoded_input else final,
           "velocity_radar": None if geocoded_input else M / "velocity.h5",
           "geocoded_input": geocoded_input}
    if geocoded_input or not chain["geocode"]:
        out.update(timeseries_geo=final if geocoded_input else None,
                   velocity_geo=M / "velocity.h5" if geocoded_input else None)
    else:
        G = M / "geo"
        geo = {"timeseries_geo": G / f"geo_{final.name}", "velocity_geo": G / "geo_velocity.h5",
               "tc_geo": G / "geo_temporalCoherence.h5", "mask_geo": G / "geo_maskTempCoh.h5"}
        need += list(geo.values())
        out.update(geo)
    _require_fresh(need, since, "correct_LOD..geocode", "mintpy_error.log")
    series = sorted((M / "geo").glob("geo_timeseries*.h5")) if (M / "geo").is_dir() else []
    if len(series) > 1:
        raise RuntimeError(f"More than one geocoded series after this run ({', '.join(p.name for p in series)}): "
                           f"only geo_{final.name} belongs to it")
    return out


def _grid_km(cfg: PipelineConfig, attrs: dict, shape) -> tuple | None:
    """(east_km, north_km) of every pixel, about the grid centre: from
    geometryRadar.h5 on a radar grid, from X_FIRST/Y_FIRST on a geocoded one."""
    import h5py
    if "Y_FIRST" in attrs:
        y0, x0 = float(attrs["Y_FIRST"]), float(attrs["X_FIRST"])
        dy, dx = float(attrs["Y_STEP"]), float(attrs["X_STEP"])
        la = (y0 + (np.arange(shape[0]) + 0.5) * dy)[:, None] * np.ones((1, shape[1]))
        lo = np.ones((shape[0], 1)) * (x0 + (np.arange(shape[1]) + 0.5) * dx)[None, :]
    else:
        geom = cfg.mintpy_dir / "inputs" / "geometryRadar.h5"
        if not geom.exists():
            return None
        with h5py.File(geom, "r") as g:
            la, lo = g["latitude"][:].astype(np.float64), g["longitude"][:].astype(np.float64)
        if la.shape != tuple(shape):
            return None
    ok = np.isfinite(la) & np.isfinite(lo)
    if not ok.any():
        return None
    lat0, lon0 = float(np.nanmean(la[ok])), float(np.nanmean(lo[ok]))
    return ((lo - lon0) * 111.32 * np.cos(np.radians(lat0)), (la - lat0) * 110.95)


def _plane(rate: np.ndarray, use: np.ndarray, km) -> dict:
    """Least-squares plane through `rate` (mm/yr) over `use`: gradient per 10 km."""
    e, n = km
    ok = use & np.isfinite(rate) & np.isfinite(e) & np.isfinite(n)
    if ok.sum() < 3:
        return {}
    A = np.c_[np.ones(int(ok.sum())), e[ok], n[ok]]
    c, *_ = np.linalg.lstsq(A, rate[ok].astype(np.float64), rcond=None)
    return {"gradient_mm_yr_per_10km": round(float(np.hypot(c[1], c[2]) * 10.0), 3),
            "east_mm_yr_per_10km": round(float(c[1] * 10.0), 3),
            "north_mm_yr_per_10km": round(float(c[2] * 10.0), 3),
            "extent_km": [round(float(np.ptp(e[ok])), 1), round(float(np.ptp(n[ok])), 1)]}


def _epoch_years(dates) -> np.ndarray:
    from datetime import datetime as _dt
    ds = [d.decode() if isinstance(d, bytes) else str(d) for d in dates]
    t0 = _dt.strptime(ds[0], "%Y%m%d")
    return np.array([(_dt.strptime(d, "%Y%m%d") - t0).days / 365.25 for d in ds])


def measure_removed_ramp(cfg: PipelineConfig, chain: dict, block_rows: int = 32) -> dict:
    """
    How much tilt the deramp step took out, measured: the per-pixel rate of
    (series before the deramp step - series after it), streamed in row
    blocks, summarised over the coherent cells (2nd/98th percentile, range)
    and as a plane gradient in mm/yr per 10 km. Written to
    mintpy/deramp_effect.json and returned for run_products.json.

    Nothing recorded this before 5 Oct 2026. Measured on Site B: the
    linear deramp removed -4.65..+2.09 mm/yr LOS (p2-p98) across the 28 km
    box, a 2.5 mm/yr per 10 km plane, against a client map band of ~10 mm/yr -
    and a broad regional movement larger than the box is absorbed the same
    way. With the deramp off, the same plane is fitted to the velocity over
    every coherent cell as a diagnostic (it includes any real deformation,
    so it is what a linear deramp WOULD have removed, not a correction).
    """
    import h5py
    M = cfg.mintpy_dir
    with h5py.File(M / "maskTempCoh.h5", "r") as f:
        coh = f["mask"][:].astype(bool)
    mask_rec = {}
    if (M / "deramp_mask.json").exists():
        try:
            mask_rec = json.loads((M / "deramp_mask.json").read_text())
        except ValueError:
            mask_rec = {}
    a, b = Path(chain["deramp_in"]), Path(chain["deramp_out"])
    if a == b:
        vel_f = M / "velocity.h5"
        rec = {"applied": False, "method": getattr(cfg, "mintpy_deramp", "no")}
        if vel_f.exists():
            with h5py.File(vel_f, "r") as f:
                vel = f["velocity"][:].astype(np.float64) * 1000.0
                attrs = dict(f.attrs)
            km = _grid_km(cfg, attrs, vel.shape) if vel.shape == coh.shape else None
            if km is not None:
                rec["diagnostic_plane"] = dict(_plane(vel, coh, km), note=(
                    "plane fitted to the velocity over every coherent cell, deformation included: what a "
                    "linear deramp would have removed; nothing was removed"))
    else:
        with h5py.File(a, "r") as fa, h5py.File(b, "r") as fb:
            t = _epoch_years(fa["date"][:])
            tc = t - t.mean()
            den = float((tc ** 2).sum())
            n, L, W = fa["timeseries"].shape
            attrs = dict(fa.attrs)
            rate = np.zeros((L, W), np.float32)
            for r0 in range(0, L, block_rows):
                d = fa["timeseries"][:, r0:r0 + block_rows, :] - fb["timeseries"][:, r0:r0 + block_rows, :]
                rate[r0:r0 + block_rows] = np.tensordot(tc, d, axes=(0, 0)) / den * 1000.0
        use = coh & np.isfinite(rate) if coh.shape == rate.shape else np.isfinite(rate)
        v = rate[use]
        rec = {"applied": True, "method": getattr(cfg, "mintpy_deramp", "linear"),
               "series_before": a.name, "series_after": b.name,
               "removed_rate_p2_mm_yr": round(float(np.percentile(v, 2)), 3) if v.size else None,
               "removed_rate_p98_mm_yr": round(float(np.percentile(v, 98)), 3) if v.size else None,
               "removed_rate_range_mm_yr": round(float(np.percentile(v, 98) - np.percentile(v, 2)), 3) if v.size else None,
               "definition": "per-pixel least-squares rate of (series before - series after the deramp step), LOS, "
                             "over the coherent cells"}
        km = _grid_km(cfg, attrs, rate.shape)
        if km is not None:
            rec.update(_plane(rate, use, km))
    for k in ("coherent_pixels", "kept_pixels", "excluded_pixels", "kept_fraction", "exclude_geojson", "buffer_m"):
        if k in mask_rec:
            rec[f"fit_{k}"] = mask_rec[k]
    (M / "deramp_effect.json").write_text(json.dumps(rec, indent=2))
    if rec.get("applied"):
        log.info("Deramp removed %.2f..%.2f mm/yr LOS (p2-p98) - a %.2f mm/yr per 10 km tilt",
                 rec["removed_rate_p2_mm_yr"] or 0, rec["removed_rate_p98_mm_yr"] or 0,
                 rec.get("gradient_mm_yr_per_10km", float("nan")))
    return rec


def _nearest_pairs(d12: list, keep) -> tuple[list, int]:
    """Stack indices of the nearest-neighbour pairs (consecutive dates of the
    kept network, in date order) and how many of those pairs the network
    lacks (not formed or dropped)."""
    pos = {p: i for i, (p, k) in enumerate(zip(d12, keep)) if k}
    dates = sorted({d for p in pos for d in p.split("_")})
    nn = [pos.get(f"{a}_{b}") for a, b in zip(dates[:-1], dates[1:])]
    return [i for i in nn if i is not None], sum(i is None for i in nn)


def _noise_reference(cfg: PipelineConfig) -> tuple:
    """
    Cells that behave as pure noise, to measure the pair-coherence noise level
    on: unwrapping random (MintPy TC below UNWRAP_NOISE_TC) and, on the
    phase-linked route, linked coherence at the measured noise level (the
    mode linked_noise_level reads) - open water and fully decorrelated ground.
    Returns (mask or None, info).
    """
    import h5py
    M = cfg.mintpy_dir
    with h5py.File(M / "temporalCoherence.h5", "r") as f:
        tc = f["temporalCoherence"][:].astype(np.float64)
    keep_f = M / "temporalCoherence_mintpy.h5"
    if keep_f.exists():
        with h5py.File(keep_f, "r") as f:
            mtc = f["temporalCoherence"][:].astype(np.float64)
        if mtc.shape != tc.shape:
            return None, {"reason": f"{keep_f.name} is {mtc.shape}, temporalCoherence.h5 {tc.shape}"}
        lvl = linked_noise_level(tc, mtc)
        if lvl.get("level") is None:
            return None, {"reason": lvl.get("reason")}
        ref = np.isfinite(tc) & (tc > 0) & (tc <= lvl["level"]) & np.isfinite(mtc) & (mtc < UNWRAP_NOISE_TC)
        info = {"linked_noise_level": lvl["level"],
                "definition": (f"MintPy temporal coherence < {UNWRAP_NOISE_TC} and linked temporal coherence "
                               f"<= its noise level {lvl['level']}")}
    else:
        ref = np.isfinite(tc) & (tc > 0) & (tc < UNWRAP_NOISE_TC)
        info = {"definition": f"MintPy temporal coherence < {UNWRAP_NOISE_TC}"}
    info["cells"] = int(ref.sum())
    if info["cells"] < NOISE_MIN_CELLS:
        return None, {**info, "reason": f"only {info['cells']} noise cells ({info['definition']}); need "
                                        f"{NOISE_MIN_CELLS}"}
    return ref, info


# the disturbed-ground layer is only meaningful when noise pairs sit well
# below real ground: above these limits it flags nearly everything (6 Oct 2026)
DISTURBED_MIN_RUN = 10
DISTURBED_MAX_SHARE = 0.5


def write_disturbed_layer(cfg: PipelineConfig, block_mb: float = 150.0, max_ref_cells: int = 20000) -> Path | None:
    """
    mintpy/disturbed.h5 on the MintPy grid: for every cell, `disturbed` = the
    longest run of consecutive nearest-neighbour pairs whose coherence is
    below the measured noise level, and `noise_like_fraction` = the share of
    those pairs below it.

    The noise level is MEASURED for this run: the 95th percentile of the
    nearest-pair coherence over cells that are pure noise (_noise_reference),
    so a pair below it is indistinguishable from noise. A cell dug, filled or
    flooded during part of the record is noise-like for that stretch while
    its mini-stack coherence, averaged over the whole record, can still pass
    the 0.4 mask (finding gap-surface_mine_site-1). The layer is delivered
    and its effect recorded in the QA; it gates nothing (Kyle, 5 Oct 2026:
    measure, then decide).

    MEASURED (5 Oct 2026) - read before using it as a gate: on the phase-
    linked route with phasor pair coherence (miaplpy_pair_coherence: phasor)
    the linked phase is smooth over the look window even on open water, so
    noise pairs span 0.18-0.97 (median 0.60, 95th percentile 0.92-0.93) and
    moderately coherent ground also falls below the level: runs of 30+ pairs
    in 30% (Site A) and 39% (Site B) of the kept cells, 20-22% of the
    cells MintPy grades > 0.7. The layer separates far better with sample
    coherence; Kyle decides on the measured effect.

    None (and a log line) when the stack does not allow it: no coherence
    dataset, fewer than 10 nearest pairs (a single-reference star), or too
    few noise cells to measure the level on. Streams the stack in row blocks
    (~block_mb each); a few seconds on Site B and Site A.
    """
    import h5py
    M = cfg.mintpy_dir
    stack = M / "inputs" / "ifgramStack.h5"
    out = M / "disturbed.h5"
    if not stack.exists() or not (M / "temporalCoherence.h5").exists():
        log.info("Disturbed-ground layer: no stack or temporal coherence in %s - not written", M)
        return None
    ref, info = _noise_reference(cfg)
    if ref is None:
        log.info("Disturbed-ground layer not written: %s", info.get("reason"))
        return None
    with h5py.File(stack, "r") as f:
        if "coherence" not in f:
            log.info("Disturbed-ground layer not written: the stack has no coherence dataset")
            return None
        d12 = [f"{a.decode() if isinstance(a, bytes) else a}_{b.decode() if isinstance(b, bytes) else b}"
               for a, b in f["date"][:]]
        keep = f["dropIfgram"][:].astype(bool) if "dropIfgram" in f else np.ones(len(d12), bool)
        nn, n_missing = _nearest_pairs(d12, keep)
        coh = f["coherence"]
        _, L, W = coh.shape
        if len(nn) < 10:
            log.info("Disturbed-ground layer not written: %d nearest-neighbour pairs in the kept network "
                     "(a single-reference star has none)", len(nn))
            return None
        if ref.shape != (L, W):
            log.info("Disturbed-ground layer not written: stack grid %s, coherence grid %s", (L, W), ref.shape)
            return None
        rows = max(4, int(block_mb * 1e6 / (len(nn) * W * 4)))
        # pass 1: the noise level, on at most max_ref_cells noise cells
        flat = np.flatnonzero(ref.ravel())
        pick = np.zeros(L * W, bool)
        pick[flat[::max(1, flat.size // max_ref_cells)]] = True
        pick = pick.reshape(L, W)
        vals = []
        for r0 in range(0, L, rows):
            s = pick[r0:r0 + rows]
            if s.any():
                blk = np.stack([coh[i, r0:r0 + rows, :] for i in nn]).astype(np.float32)
                vals.append(blk[:, s].ravel())
        v = np.concatenate(vals)
        v = v[np.isfinite(v)]
        level = float(np.percentile(v, 95))
        # pass 2: the longest run below it, every cell
        run = np.full((L, W), np.nan, np.float32)
        frac = np.full((L, W), np.nan, np.float32)
        for r0 in range(0, L, rows):
            blk = np.stack([coh[i, r0:r0 + rows, :] for i in nn]).astype(np.float32)
            below = ~(blk >= level)                              # NaN counts as noise-like
            cur = np.zeros(blk.shape[1:], np.int32)
            best = np.zeros(blk.shape[1:], np.int32)
            for k in range(below.shape[0]):
                cur = np.where(below[k], cur + 1, 0)
                np.maximum(best, cur, out=best)
            valid = (np.nan_to_num(blk) > 0).any(axis=0)         # outside the data: coherence 0 in every pair
            run[r0:r0 + rows] = np.where(valid, best, np.nan)
            frac[r0:r0 + rows] = np.where(valid, below.mean(axis=0), np.nan)
    # Is the layer able to separate anything? Measured 6 Oct 2026 on Site B
    # (phase-linked, phasor pair coherence): noise level 0.926 and 93.8% of the
    # cells with a noise-like run of >= 10 pairs - ground MintPy grades reliable
    # is flagged as often as water, so the layer says nothing. The test: the
    # share of cells MintPy grades >= 0.7 (unwrapping consistency) that the
    # layer flags. Such a layer is kept for the record but marked unusable:
    # never exported, never used by the QA.
    okc = np.isfinite(run)
    ref_cells = okc.copy()
    mtc_f = M / "temporalCoherence_mintpy.h5"
    if mtc_f.exists():
        with h5py.File(mtc_f, "r") as f:
            mtc = f["temporalCoherence"][:]
        if mtc.shape == run.shape:
            ref_cells = okc & (mtc >= 0.7)
    share10 = float((run[ref_cells] >= DISTURBED_MIN_RUN).mean()) if ref_cells.any() else 1.0
    usable = share10 <= DISTURBED_MAX_SHARE
    why = ("" if usable else
           f"cannot separate disturbed ground: {100 * share10:.0f}% of the cells MintPy grades reliable "
           f"(unwrapping consistency >= 0.7) have a noise-like run of >= {DISTURBED_MIN_RUN} pairs (limit "
           f"{100 * DISTURBED_MAX_SHARE:.0f}%; pair noise level {level:.2f}). The phasor pair coherence of "
           "phase-linked interferograms is high on noise too; sample pair coherence would separate it")
    with h5py.File(M / "temporalCoherence.h5", "r") as f:
        attrs = {k: (v.decode() if isinstance(v, bytes) else v) for k, v in f.attrs.items()
                 if k not in ("temporalCoherenceSource", "linked_noise_level", "linked_noise_method",
                              "threshold_minus_noise")}
    attrs.update({"usable": "yes" if usable else "no", "usable_reason": why})
    attrs.update({"FILE_TYPE": "disturbed", "UNIT": "1",
                  "description": ("disturbed: longest run of consecutive nearest-neighbour pairs (kept network, "
                                  "date order) whose coherence is below pair_noise_level; noise_like_fraction: "
                                  "share of those pairs below it. NaN / -1 outside the data."),
                  "pair_noise_level": str(round(level, 4)),
                  "pair_noise_method": (f"95th percentile of nearest-pair coherence over {int(pick.sum())} noise "
                                        f"cells ({info['definition']})"),
                  "pair_noise_cells": str(info["cells"]),
                  "nearest_pairs_used": str(len(nn)), "nearest_pairs_missing": str(n_missing),
                  "linked_noise_level": str(info.get("linked_noise_level", "nan")),
                  # which coherence the pairs carry: on the phase-linked route
                  # the phasor figure is block smoothness of the linked phase
                  # (see the docstring for what that does to this layer)
                  "pair_coherence_kind": (str(getattr(cfg, "miaplpy_pair_coherence", "phasor"))
                                          if str(getattr(cfg, "backend", "")) == "miaplpy" else "sample")})
    tmp = out.with_name(out.stem + "_tmp.h5")
    with h5py.File(tmp, "w") as g:
        g.create_dataset("disturbed", data=run)
        g.create_dataset("noise_like_fraction", data=frac)
        for k, val in attrs.items():
            g.attrs[k] = val
    os.replace(tmp, out)
    ok = np.isfinite(run)
    if not usable:
        log.warning("Disturbed-ground layer marked unusable (not exported, not used by the QA): %s", why)
    log.info("Disturbed-ground layer: pair noise level %.3f (%d noise cells); %d nearest pairs (%d missing); "
             "%.1f%% of cells have a noise-like run of >= 10 pairs -> %s", level, info["cells"], len(nn),
             n_missing, 100 * float((run[ok] >= 10).mean()) if ok.any() else 0.0, out)
    return out


def _disturbed_to_int16(path: Path) -> None:
    """The geocoded `disturbed` as int16, -1 off the data (MintPy's geocode
    resamples to float with NaN fill)."""
    import h5py
    with h5py.File(path, "r+") as f:
        if f["disturbed"].dtype == np.int16:
            return
        x = f["disturbed"][:]
        d = np.where(np.isfinite(x), np.rint(np.nan_to_num(x)), -1).astype(np.int16)
        del f["disturbed"]
        ds = f.create_dataset("disturbed", data=d)
        ds.attrs["NO_DATA_VALUE"] = -1
        ds.attrs["UNIT"] = "pairs"
        f.attrs["disturbed_no_data_value"] = "-1"


def _run_geocode(cfg: PipelineConfig, files: list) -> subprocess.CompletedProcess:
    """MintPy's geocode.py on `files`, exactly as its geocode step calls it
    (lookup inputs/geometryRadar.h5, the run's template for the posting,
    --outdir geo, --update)."""
    from . import procs
    M = cfg.mintpy_dir
    argv = ["geocode.py", *[str(p) for p in files], "-l", str(M / "inputs" / "geometryRadar.h5"),
            "-t", str(M / "smallbaselineApp.cfg"), "--outdir", str(M / "geo"), "--update"]
    r = procs.run_group(argv, cwd=M, timeout=3600)
    (M / "mintpy_geocode_extra.log").write_text((r.stdout or "") + (r.stderr or ""))
    return r


def _drop_unfinished_geo(targets: list, before: dict, why: str) -> list:
    """
    Remove the geo_ files a geocode.py call that did not finish wrote or
    changed (targets: the geo/geo_<name> paths it was asked for; before:
    {path: st_mtime_ns} of those on disk before the call). geocode.py creates
    each output file before it writes the data, so one cut off part-way is
    all zeros - and the products stage keeps a layer that is on disk as it
    is (geocode_extra_layers(missing_only=True), F5 7 Oct 2026), so a broken
    one would be read by the deliverables, the QA and the report on every
    later run. Removed, the next products run makes it again. Returns the
    names removed.
    """
    gone = []
    for g in targets:
        try:
            if g.exists() and before.get(g) != g.stat().st_mtime_ns:
                g.unlink()
                gone.append(g.name)
        except OSError as exc:
            log.error("Could not remove %s, written by a geocoding that did not finish (%s): delete it before "
                      "the next products run", g, exc)
    if gone:
        log.error("Removed %s (%s): an unfinished geocoding's output is not a layer; the next products run "
                  "makes it again", ", ".join(gone), why)
    return gone


# The layers geocode_extra_layers puts on the product grid, by their mintpy/
# name (each becomes geo/geo_<name>), in the order it geocodes them.
EXTRA_LAYERS = ("temporalCoherence_mintpy.h5", "demErr.h5", "disturbed.h5")


def extra_layers_for(cfg) -> tuple:
    """EXTRA_LAYERS this route can make. NISAR's network is a chain of 12-day
    pairs: MintPy's temporal coherence is 1.0 by construction (no loops), not
    an unwrapping-consistency test, so it never becomes that layer (Site E
    NISAR, 26 Oct 2026: median 1.0); the route's reliability is the pair
    coherence (apply_nisar_pair_coherence)."""
    if str(getattr(cfg, "backend", "")) == "nisar":
        return tuple(n for n in EXTRA_LAYERS if n != "temporalCoherence_mintpy.h5")
    return EXTRA_LAYERS


def geocode_extra_layers(cfg: PipelineConfig, missing_only: bool = False) -> list:
    """
    The diagnostic layers MintPy's own geocode step leaves in radar
    coordinates, onto the product grid, beside the products:

      temporalCoherence_mintpy.h5 -> geo/geo_temporalCoherence_mintpy.h5
          MintPy's network temporal coherence = unwrapping consistency on the
          phase-linked route (the second quality gate's layer,
          min_unwrap_consistency; the first gate cannot see unwrapping errors)
      demErr.h5 -> geo/geo_demErr.h5
          the DEM error MintPy estimated: ground that differs from the
          2011-2015 elevation model (15-25 m on Site B's 'spoil?' patch
          against ~5 m on farmland) - the best available sign of reshaped land
      disturbed.h5 -> geo/geo_disturbed.h5 (int16, -1 off the data)
          written first (write_disturbed_layer) when the stack allows it

    Callable on its own, so finished projects gain the layers without
    re-running MintPy: it only adds files (mintpy/disturbed.h5 and the geo_
    files), never changes an existing product. Returns the geo files this
    call verified (each present and newer than its source).

    missing_only (the products stage, F5 7 Oct 2026): a layer whose
    geo/geo_<name> is already on disk is left exactly as it is - not
    geocoded again, not re-tagged, and the disturbed layer not recomputed -
    and only the others are made. The MintPy run uses the default (all of
    them: it has just retired geo/).

    Because a present layer is kept, a geocoding that is stopped, raises or
    exits non-zero leaves none of the files it wrote (_drop_unfinished_geo),
    and a disturbed layer that fails to compute no longer stops the other
    two (review of F5, 7 Oct 2026).
    """
    import h5py
    import shutil
    M = cfg.mintpy_dir
    G = M / "geo"
    tcf = M / "temporalCoherence.h5"
    if not tcf.exists():
        log.info("No %s - no extra layers to geocode", tcf)
        return []
    present = [n for n in EXTRA_LAYERS if missing_only and (G / f"geo_{n}").exists()]
    if present:
        log.info("Already on the product grid, left as they are (not geocoded again): %s",
                 ", ".join(f"geo_{n}" for n in present))
    with h5py.File(tcf, "r") as f:
        geocoded_grid = "Y_FIRST" in f.attrs
    dist = None
    if "disturbed.h5" not in present:
        try:
            dist = write_disturbed_layer(cfg)
        except Exception as exc:                              # noqa: BLE001 - the other two do not need it
            # until 7 Oct 2026 this ended the call: the second gate's layer and
            # the DEM error (what F5's products step is for) were never made
            log.error("Disturbed-ground layer not written (%s: %s); the other extra layers are geocoded "
                      "without it", type(exc).__name__, exc, exc_info=True)
    if geocoded_grid:
        # the products are on the map grid already (HyP3, NISAR): each layer
        # is copied to its geo_ name. Until 26 Oct 2026 only the disturbed
        # layer was ("the layers are used where they are"), but every reader -
        # deliverables, QA, report, provenance - opens geo/geo_<name>, so on
        # these routes the DEM error and the unwrapping consistency never
        # reached the products (Site E NISAR).
        G.mkdir(exist_ok=True)
        done = []
        for s in (M / "temporalCoherence_mintpy.h5", M / "demErr.h5"):
            if s.name in present or not s.exists() or s.name not in extra_layers_for(cfg):
                continue
            dst = G / f"geo_{s.name}"
            shutil.copy2(s, dst)
            with h5py.File(dst, "r+") as f:
                if s.name == "temporalCoherence_mintpy.h5":
                    f.attrs["temporalCoherenceSource"] = MINTPY_TC_SOURCE
                elif "description" not in f.attrs:
                    f.attrs["description"] = ("MintPy DEM error (correct_topography): ground height minus the "
                                              "elevation model, m - ground reshaped since the DEM epoch shows here")
            done.append(dst)
        if dist is not None:
            dst = G / "geo_disturbed.h5"
            shutil.copy2(dist, dst)
            _disturbed_to_int16(dst)
            done.append(dst)
        log.info("Extra layers on the product grid (copied, already geocoded): %s",
                 ", ".join(p.name for p in done) or "none")
        return done
    srcs = [p for p in (M / "temporalCoherence_mintpy.h5", M / "demErr.h5", dist)
            if p is not None and p.exists() and p.name not in present]
    if not srcs:
        return []
    if not (M / "inputs" / "geometryRadar.h5").exists():
        log.error("Cannot geocode %s: no inputs/geometryRadar.h5", ", ".join(p.name for p in srcs))
        return []
    G.mkdir(exist_ok=True)
    targets = [G / f"geo_{s.name}" for s in srcs]
    before = {g: g.stat().st_mtime_ns for g in targets if g.exists()}
    try:
        r = _run_geocode(cfg, srcs)
    except BaseException:
        _drop_unfinished_geo(targets, before, "the geocoding was stopped or failed part-way")
        raise
    if r.returncode != 0:
        _drop_unfinished_geo(targets, before, f"geocode.py exited {r.returncode}")
    done = []
    for s in srcs:
        g = G / f"geo_{s.name}"
        if g.exists() and g.stat().st_mtime >= s.stat().st_mtime - 1.0:
            if s.name == "disturbed.h5":
                _disturbed_to_int16(g)
            else:
                # say what the layer is (finished projects' radar files predate the tag)
                with h5py.File(g, "r+") as f:
                    if s.name == "temporalCoherence_mintpy.h5":
                        f.attrs["temporalCoherenceSource"] = MINTPY_TC_SOURCE
                    elif s.name == "demErr.h5" and "description" not in f.attrs:
                        f.attrs["description"] = ("MintPy DEM error (correct_topography): ground height minus the "
                                                  "elevation model, m - ground reshaped since the DEM epoch shows here")
            done.append(g)
        else:
            log.error("Geocoding %s did not produce a current %s (rc %s) - see mintpy_geocode_extra.log",
                      s.name, g.name, r.returncode)
    log.info("Extra layers on the product grid: %s", ", ".join(p.name for p in done) or "none")
    return done


def write_run_products(cfg: PipelineConfig, rec: dict) -> Path:
    """mintpy/run_products.json, written atomically (the deliverables read it
    to pick the series and to say which corrections were applied)."""
    out = cfg.mintpy_dir / "run_products.json"
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(json.dumps(rec, indent=2, default=str))
    os.replace(tmp, out)
    return out


RUN_INCOMPLETE = "run_incomplete.json"


def mark_run_incomplete(cfg: PipelineConfig, stamp: str, started: float) -> Path:
    """
    mintpy/run_incomplete.json, written before a MintPy run sets aside or
    moves anything of the previous run's (in run_mintpy_with_loader, before
    it touches mintpy/ at all) and removed only once its run_products.json
    is written (clear_run_incomplete). While it is there, mintpy/ and geo/
    hold a run that did not pass its own end checks, or was stopped part-way:
    'Resume' runs MintPy again (runs.suggest_kind), the deliverables refuse
    (deliverables.export_all) and the release is blocked (release.blockers).
    Until 7 Oct 2026 a run that failed its deramp-mask or second-gate check
    after geocoding left current-looking geo products with no record, and
    'Products only' built a report from them.
    """
    out = cfg.mintpy_dir / RUN_INCOMPLETE
    out.write_text(json.dumps({
        "stamp": stamp, "started": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(started)),
        "why": "a MintPy run started at this time and has not finished its checks; its products must not be "
               "used. Run MintPy again (this file is removed when a run finishes)."}, indent=2))
    return out


def clear_run_incomplete(cfg: PipelineConfig) -> None:
    """The run finished and its record is written (see mark_run_incomplete)."""
    (cfg.mintpy_dir / RUN_INCOMPLETE).unlink(missing_ok=True)


def run_mintpy_with_loader(cfg: PipelineConfig, loader_lines: str,
                           mid_hook=None) -> None:
    """
    Run MintPy against a backend-supplied loader block.

    The HyP3 and NISAR backends produce products MintPy reads natively, so
    they skip the metadata staging the topsApp path needs and only differ
    in how the stack is located. Everything after loading - reference
    point, thresholds, inversion, deliverables - is identical across
    backends, which is what keeps results comparable between them.

    Runs in phases: load..reference_point, correct_unwrap_error..
    invert_network, then correct_LOD..geocode, then MintPy's KMZ/plot tail.
    Between the inversion and correct_LOD the deramp mask is built
    (deramp_mask_step) and the phase-closure diagnostic computed. A failure in
    the mask step FAILS THE RUN: the alternative is a ramp fitted through the
    mine with only a log line to show for it.

    Success is judged on the products, never on a file merely existing
    (5 Oct 2026; until then a re-run that failed after the inversion was
    delivered from the previous run's geo products): the previous run's
    downstream products are moved to mintpy/superseded/run_<stamp>/ before
    correct_LOD (retire_downstream_products); every science step must exit 0
    and leave current products (check_inversion, verify_science_products);
    only the plotting tail may fail; partial ERA5 coverage stops the run
    (corrections.verify_era5_coverage). Then the extra layers are geocoded
    (geocode_extra_layers), the removed tilt measured (measure_removed_ramp)
    and mintpy/run_products.json written - the record the deliverables use
    to pick the series and to say which corrections were applied. From
    before the first thing in mintpy/ is rewritten, set aside or moved until
    that record is written, mintpy/run_incomplete.json says the products are
    not a finished run (mark_run_incomplete).

    mid_hook is retained for callers that need something else done in that
    window; it runs after the deramp mask.
    """
    t_run = time.time()
    stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(t_run))
    cfg.mintpy_dir.mkdir(parents=True, exist_ok=True)
    reference_lalo = cfg.reference_lalo.strip() if cfg.reference_lalo else "auto"
    if reference_lalo == "auto":
        log.warning("reference_lalo is unset - MintPy will choose the "
                    "reference point. Fine for testing, not for a deliverable.")

    excluded = _screened_out_pairs(cfg)
    exclude_line = (f"mintpy.network.excludeDate12           = {','.join(excluded)}\n"
                    if excluded else "")
    drop_dates = series_exclude_dates(cfg)
    if drop_dates:
        exclude_line += f"mintpy.network.excludeDate             = {','.join(drop_dates)}\n"
        log.info("Leaving %d date(s) out of the series (mintpy_exclude_dates, and no weather data): %s", len(drop_dates),
                 ", ".join(drop_dates))

    # See PipelineConfig.subset_to_aoi: HyP3 multi-burst products vary in
    # size between dates, and MintPy will not stack layers whose shapes
    # disagree. Cropping everything to the AOI gives them a common grid.
    min_lon, min_lat, max_lon, max_lat = cfg.bbox
    subset_line = (
        f"mintpy.subset.lalo                 = "
        f"{min_lat}:{max_lat},{min_lon}:{max_lon}\n"
        if cfg.subset_to_aoi else ""
    )
    if cfg.subset_to_aoi:
        log.info("Cropping all layers to the AOI (%.4f-%.4f lat, %.4f-%.4f lon) "
                 "so they share one grid", min_lat, max_lat, min_lon, max_lon)

    # A loader block that does not end in a newline merges its last setting
    # with the first one appended below, producing a key MintPy cannot parse
    # and a step that fails for reasons the traceback does not explain.
    if loader_lines and not loader_lines.endswith("\n"):
        loader_lines += "\n"

    config_text = (
        loader_lines
        + f"mintpy.reference.lalo              = {reference_lalo}\n"
        + f"mintpy.networkInversion.minTempCoh = {cfg.min_temporal_coherence}\n"
        + f"mintpy.network.minCoherence        = {cfg.min_spatial_coherence}\n"
        + f"mintpy.reference.minCoherence      = {cfg.reference_min_coherence}\n"
        + "".join(line + "\n" for line in corrections.mintpy_tropo_lines(cfg))
        + corrections.mintpy_lines(cfg)
        + "mintpy.topographicResidual         = yes\n"
        + "mintpy.networkInversion.weightFunc = var\n"
        + subset_line
        + exclude_line
    )
    # From here the run changes mintpy/: the template, the stack set aside
    # (retire_stale_ifgram_stack), the previous radar products moved on a
    # reload (clear_stale_mintpy_products), then the rest before correct_LOD
    # (retire_downstream_products). The marker goes down before all of them:
    # until 7 Oct 2026 it came only at the last, so a death in load_data or
    # the inversion after a reload left a mixed mintpy/ with no marker.
    mark_run_incomplete(cfg, stamp, t_run)
    config_path = cfg.mintpy_dir / "smallbaselineApp.cfg"
    config_path.write_text(config_text)
    retire_stale_ifgram_stack(cfg, loader_lines)

    # Run in two chunks, skipping quick_overview. That step plots a
    # histogram of triplets with non-zero integer ambiguity, and crashes
    # on `bins must be positive` when the count is zero everywhere - i.e.
    # precisely when phase closure finds NO unwrapping errors, which is
    # the best case. Its output is diagnostic only; correct_unwrap_error
    # and everything after are unaffected.
    # MintPy writes derived products (waterMask.h5, geometryGeo.h5 copies,
    # avgSpatialCoh.h5 and similar) into the work directory root rather
    # than inputs/. Deleting inputs/ to force a reload therefore leaves
    # them behind at the previous extent, and the inversion later dies on
    # "water mask file has different size from ifgramStack file". Clear
    # them whenever the stack is being rebuilt.
    stack = cfg.mintpy_dir / "inputs" / "ifgramStack.h5"
    clear_stale_mintpy_products(cfg, stamp)

    t0 = time.time()
    log.info("Running MintPy (load_data through modify_network)")
    first = _mintpy(cfg, config_path, "load_data", "modify_network", "mintpy_load.log")
    if first.returncode != 0 or not stack.exists():
        log.error("MintPy failed loading the stack - see mintpy_load.log")
        raise RuntimeError("MintPy load step failed - see mintpy_load.log")

    # the weather first: a date it cannot serve leaves the series here, before
    # anything is inverted, not as a stop after the inversion
    exclude_weather_unavailable(cfg, config_path)

    # Before reference_point: is the reference pixel reliably unwrapped in
    # every kept pair? If not, drop those pairs (they are the ones with the
    # least to contribute) and re-run modify_network so maskConnComp is
    # rebuilt without them. Bounded: many failures mean the reference is
    # badly placed, not that the pairs are.
    bad, info = reference_unreliable_pairs(cfg)
    if bad:
        if len(bad) > max(3, 0.2 * info.get("n_kept", len(bad))):
            raise RuntimeError(
                f"The reference pixel {cfg.reference_lalo} is in connected component 0 "
                f"in {len(bad)} of {info.get('n_kept')} pairs (median coherence "
                f"{info.get('ref_median_coherence')}). That is a badly placed reference, "
                "not a few unlucky pairs - move reference_lalo.")
        log.warning("Reference pixel is unreliable in %d pair(s) - component 0 in %d, "
                    "component below MintPy's %d-px minimum (or eroded away) in %d: %s. "
                    "Excluding them (reference median coherence %.3f over %d pairs) and "
                    "rebuilding the network.", len(bad), len(info["pairs_ref_in_cc0"]),
                    int(info["cc_min_area"]), len(info["pairs_ref_component_too_small"]),
                    ", ".join(bad), info["ref_median_coherence"], info["n_kept"])
        _set_exclude_line(config_path, excluded + bad)
        (cfg.mintpy_dir / "reference_excluded_pairs.json").write_text(json.dumps(
            {"excluded_for_reference": bad, "manual_or_tropo": excluded, **info}, indent=2))
        # maskConnComp.h5 is the intersection over the OLD kept set; make
        # reference_point rebuild it rather than trust --update's mtime test.
        (cfg.mintpy_dir / "maskConnComp.h5").unlink(missing_ok=True)
        redo = _mintpy(cfg, config_path, "modify_network", "modify_network", "mintpy_network.log")
        if redo.returncode != 0:
            raise RuntimeError("MintPy modify_network failed after excluding pairs - see "
                               "mintpy_network.log")
    else:
        log.info("Reference pixel is reliably unwrapped in all %s kept pairs (median "
                 "coherence %s)", info.get("n_kept", "?"), info.get("ref_median_coherence", "?"))

    ensure_connected_network(cfg, config_path)
    pin_reference_into_conncomp_mask(cfg, config_path)

    log.info("Running MintPy (reference_point)")
    ref = _mintpy(cfg, config_path, "reference_point", "reference_point", "mintpy_reference.log")
    if ref.returncode != 0:
        log.error("MintPy reference_point failed - see mintpy_reference.log")
        raise RuntimeError("MintPy reference step failed - see mintpy_reference.log")
    t1 = time.time()

    # Unwrap-error correction on its own, then CHECK what it wrote before the
    # inversion asks for it. Phase closure needs a region reliably unwrapped
    # in every kept pair (>= connCompMinArea); over 670 pairs and 9.5 years
    # of summer decorrelation there was none on P2, it wrote nothing, and
    # the inversion died asking for unwrapPhase_bridging_phaseClosure. The
    # corrections that did land are used; the downgrade is recorded. Its exit
    # code is NOT required to be 0 (that downgrade can end non-zero); what it
    # wrote is checked instead (record_unwrap_error_state, unwrap_error_applied).
    remove_rejected_closure_dataset(cfg)
    bumped_at = refresh_unwrap_error_if_stale(cfg)
    log.info("Running MintPy (correct_unwrap_error)")
    unw = _mintpy(cfg, config_path, "correct_unwrap_error", "correct_unwrap_error", "mintpy_unwerr.log")
    if unw.returncode != 0:
        log.warning("MintPy correct_unwrap_error exited %s; judging it on the datasets it wrote. Last lines "
                    "of mintpy_unwerr.log:\n%s", unw.returncode, _log_tail(cfg.mintpy_dir / "mintpy_unwerr.log"))
    record_unwrap_error_state(cfg, bumped_at, config_path)
    applied = unwrap_error_applied(cfg, config_path)
    if getattr(cfg, "mintpy_closure_correction", False) and applied["dataset"] in ("unwrapPhase", "unwrapPhase_bridging"):
        rec = closure_integer_correction(cfg, applied["dataset"])
        method = "bridging+phase_closure" if applied["dataset"].endswith("bridging") else "phase_closure"
        lines = [l for l in config_path.read_text().splitlines() if not l.strip().startswith("mintpy.unwrapError.method")]
        config_path.write_text("\n".join(lines + [f"mintpy.unwrapError.method = {method}"]) + "\n")
        applied.update({"applied": method, "dataset": rec["dataset_out"], "closure_correction": "per-pixel (pipeline)"})
        (cfg.mintpy_dir / "unwrap_error_applied.json").write_text(json.dumps(applied, indent=2))
    log.info("Unwrap-error correction: requested %s, applied %s, inversion will use %s",
             applied["requested"], applied["applied"] or "none", applied["dataset"])

    # Stop after the inversion so the deramp mask can see maskTempCoh.h5,
    # then resume at the step after it. rc 0 AND current products: an
    # inversion killed for memory on a re-run used to pass on the previous
    # run's maskTempCoh.h5 and correct the previous run's timeseries.h5.
    log.info("Running MintPy (invert_network)")
    t_inv = time.time()
    inv = _mintpy(cfg, config_path, "invert_network", "invert_network", "mintpy_invert.log")
    inversion = check_inversion(cfg, inv, applied["dataset"], t_inv)
    t2 = time.time()

    if cfg.backend == "miaplpy":
        apply_linked_temporal_coherence(cfg)   # before anything masks on it
    elif cfg.backend == "nisar":
        apply_nisar_pair_coherence(cfg)        # a chain's temporal coherence is 1.0: use the pairs'
    deramp_mask_step(cfg)          # raises rather than fit through the mine
    phase_closure_diagnostic(cfg)
    # large areas: the ionosphere, from global maps, before MintPy's own
    # corrections (the tide is MintPy's correct_SET step, set in the template)
    decision = corrections.decide(cfg)
    iono_rec = None
    if decision["ionosphere"] == "gim":
        iono_rec = corrections.apply_gim_ionosphere(
            cfg, cfg.mintpy_dir / "timeseries.h5", cfg.mintpy_dir / "inputs" / "geometryRadar.h5")
    corrections.record(cfg, {"ionosphere_result": iono_rec})
    log.info("Tide / ionosphere: %s", decision["why"])
    if mid_hook is not None:
        mid_hook(cfg)

    # Nothing from an earlier run may stand in for this one's products (see
    # retire_downstream_products); after this, a product exists only if this
    # run wrote it. run_incomplete.json has been down since the run began, so
    # a failure in the moves or anywhere after (end checks, a Stop in the plot
    # tail) leaves it there.
    retired = retire_downstream_products(cfg, stamp)

    # The science steps must succeed - exit 0 and every product current. Only
    # the KMZ / HDF-EOS / plotting tail may fail, so it runs as its own call:
    # MintPy plots at the end of any multi-step call (mintpy.plot), so the
    # plot is switched off for the science calls and the template restored.
    # The science runs in two calls split after the troposphere step, so the
    # ERA5 coverage is checked BEFORE anything is deramped or geocoded: a
    # partly corrected series never reaches geo/, where "Products only" would
    # have delivered it under the ERA5 name.
    try:
        tmpl = mintpy_template(config_path)
        tropo = str(tmpl.get("mintpy.troposphericDelay.method") or "no")
        model = str(tmpl.get("mintpy.troposphericDelay.weatherModel") or "ERA5").upper()
    except Exception as exc:                                  # noqa: BLE001
        log.warning("Could not resolve the MintPy template (%s); taking the troposphere method from the config", exc)
        tropo, model = corrections.mintpy_tropo_method(cfg), corrections.weather_model(cfg)
    try:
        n_dates = _n_dates(cfg.mintpy_dir / "timeseries.h5")
    except Exception:                                         # noqa: BLE001  (display only)
        n_dates = 0
    t3 = time.time()
    era5 = None
    if tropo == "pyaps":
        weather_file = cfg.mintpy_dir / "inputs" / f"{model}.h5"
        retire_partial_weather_file(cfg, weather_file, stamp)
        if str(getattr(cfg, "mintpy_tropo_method", "")) == "raider":
            from . import weather_raider
            weather_raider.build_delay_file(cfg, cfg.mintpy_dir / "timeseries.h5",
                                            cfg.mintpy_dir / "inputs" / "geometryRadar.h5"
                                            if (cfg.mintpy_dir / "inputs" / "geometryRadar.h5").exists()
                                            else cfg.mintpy_dir / "inputs" / "geometryGeo.h5", weather_file)
    with _template_key(config_path, "mintpy.plot", "no"):
        log.info("Running MintPy (correct_LOD through correct_troposphere)")
        if tropo == "pyaps" and model == "ERA5":
            with _WeatherWatch(cfg, "mintpy_correct_LOD_to_correct_troposphere", n_dates):
                result = _mintpy(cfg, config_path, "correct_LOD", "correct_troposphere", "mintpy_error.log")
        else:
            result = _mintpy(cfg, config_path, "correct_LOD", "correct_troposphere", "mintpy_error.log")
        _science_call_ok(cfg, result, "correct_LOD..correct_troposphere", retired)
        if tropo == "pyaps":
            era5 = corrections.verify_era5_coverage(
                cfg, cfg.mintpy_dir / "timeseries.h5", cfg.mintpy_dir / "inputs" / f"{model}.h5", model)
        log.info("Running MintPy (deramp through geocode)")
        result = _mintpy(cfg, config_path, "deramp", "geocode", "mintpy_error.log", append=True)
        _science_call_ok(cfg, result, "deramp..geocode", retired)
    prod = verify_science_products(cfg, config_path, t3)
    chain = prod["chain"]
    t4 = time.time()

    log.info("Running MintPy (google_earth onward: KMZ and plots)")
    plots = _mintpy(cfg, config_path, "google_earth", None, "mintpy_plot.log")
    if plots.returncode != 0:
        log.warning("MintPy's KMZ/plotting tail exited %s (see mintpy_plot.log); every science product was "
                    "written by this run and checked, so the run continues.", plots.returncode)
    # The diagnostic layers and the tilt measurement must not throw away a
    # verified run if they hit a bug; what failed is recorded (and a gate that
    # NEEDS its layer still raises in second_gate_record).
    try:
        extra = geocode_extra_layers(cfg)
    except Exception as exc:                                  # noqa: BLE001
        log.error("Extra layers (unwrapping consistency, DEM error, disturbed ground) failed: %s", exc)
        extra = []
    gate = second_gate_record(cfg, prod)
    try:
        ramp = measure_removed_ramp(cfg, chain)
    except Exception as exc:                                  # noqa: BLE001
        log.error("Could not measure the removed ramp: %s", exc)
        ramp = {"error": str(exc)}
    end_state = _report_mintpy_end_state(cfg)          # raises if the deramp ignored its mask
    t5 = time.time()
    log.info("MintPy timing: load %.0f s, unwrap-error+inversion %.0f s, corrections through geocode %.0f s, "
             "plots and extra layers %.0f s, total %.1f min",
             t1 - t0, t2 - t1, t4 - t3, t5 - t4, (t5 - t0) / 60)
    rec = run_products_record(cfg, prod, started=t_run, stamp=stamp, retired=retired, era5=era5, ramp=ramp,
                              extra=extra, gate=gate, applied=applied, inversion=inversion,
                              iono_rec=iono_rec, plots_rc=plots.returncode, end_state=end_state)
    write_run_products(cfg, rec)
    clear_run_incomplete(cfg)
    log.info("MintPy time series complete and verified (run %s). Outputs in %s; record in run_products.json",
             stamp, cfg.mintpy_dir)


def _n_dates(ts_file: Path) -> int:
    import h5py
    with h5py.File(ts_file, "r") as f:
        return int(f["date"].shape[0])


def _science_call_ok(cfg: PipelineConfig, result, what: str, retired) -> None:
    """A science call of smallbaselineApp must exit 0; say where the previous
    run's products went (nothing from them is used)."""
    if result.returncode == 0:
        return
    raise RuntimeError(
        f"MintPy failed in {what} (exit {result.returncode}). Nothing from an earlier run is used: its "
        f"products are in {retired or 'mintpy/superseded/'} and geo/ holds only what this run wrote. Last "
        "lines of mintpy_error.log:\n" + _log_tail(cfg.mintpy_dir / "mintpy_error.log"))


class _template_key:
    """Set one template key for a block and put back the line(s) it replaced
    (MintPy rewrites the template on every call, so the restore removes
    whatever line it left and re-appends the originals)."""

    def __init__(self, config_path: Path, key: str, value: str):
        self.path, self.key, self.value = Path(config_path), key, value
        self.old = []

    def _match(self, line: str) -> bool:
        s = line.strip()
        return s.startswith(self.key + " ") or s.startswith(self.key + "=")

    def __enter__(self):
        self.old = [l for l in self.path.read_text().splitlines() if self._match(l)]
        _set_config_key(self.path, self.key, self.value)
        return self

    def __exit__(self, *exc):
        lines = [l for l in self.path.read_text().splitlines() if not self._match(l)]
        self.path.write_text("\n".join(lines + self.old) + "\n")
        return False


def check_inversion(cfg: PipelineConfig, result, dataset: str, since: float) -> str:
    """
    invert_network must exit 0 and leave current products: timeseries.h5,
    temporalCoherence.h5, numInvIfgram.h5 and maskTempCoh.h5 either written by
    this call, or - MintPy's update mode skips an inversion whose products are
    newer than the dataset it inverts and carry the same reference pixel - left
    by an earlier inversion of exactly this input. Anything else raises.
    Returns "inverted" or "up to date".
    """
    import h5py
    M = cfg.mintpy_dir
    if result.returncode != 0:
        raise RuntimeError(f"MintPy invert_network failed (exit {result.returncode}; -9 is usually memory). "
                           "Last lines of mintpy_invert.log:\n" + _log_tail(M / "mintpy_invert.log"))
    outs = [M / "timeseries.h5", M / "temporalCoherence.h5", M / "numInvIfgram.h5"]
    missing = [p.name for p in outs + [M / "maskTempCoh.h5"] if not p.exists()]
    if missing:
        raise RuntimeError(f"MintPy invert_network exited 0 but did not write {', '.join(missing)} - see "
                           "mintpy_invert.log")
    oldest = min(p.stat().st_mtime for p in outs)
    if oldest >= since - 1.0:
        return "inverted"
    stack = M / "inputs" / "ifgramStack.h5"
    with h5py.File(stack, "r") as f:
        t_in = _mod_time(f[dataset]) if dataset in f else None
        ref_s = [str(f.attrs.get(k, "")) for k in ("REF_Y", "REF_X")]
    with h5py.File(M / "timeseries.h5", "r") as f:
        ref_t = [str(f.attrs.get(k, "")) for k in ("REF_Y", "REF_X")]
    t_in = t_in if t_in is not None else stack.stat().st_mtime
    if oldest >= t_in and ref_s == ref_t:
        log.info("MintPy found the inversion up to date (its products are newer than %s and share its "
                 "reference pixel) and kept them", dataset)
        return "up to date"
    raise RuntimeError(f"The inversion products are older than the {dataset} they should invert (or carry "
                       f"another reference pixel: {ref_t} vs {ref_s}) and invert_network did not rewrite them - "
                       "see mintpy_invert.log")


def second_gate_record(cfg: PipelineConfig, prod: dict) -> dict:
    """
    The unwrapping-consistency layer the second quality gate reads, and - when
    min_unwrap_consistency > 0 - its measured effect on the delivered mask.

    The gate itself is applied by the figures and the QA (panel_history,
    quality_assessment); the delivered masks stay the first gate, so the QA
    can measure the second against them. Default 0.0 = off (Kyle, 5 Oct 2026:
    measure, then decide). A gate asked for whose layer this run could not
    produce raises here, at the end of MintPy, rather than in the report.
    """
    import h5py
    M = cfg.mintpy_dir
    thr = float(cfg.min_unwrap_consistency or 0.0)
    on_geo = prod.get("mask_geo") is not None             # products geocoded by MintPy (radar route)
    if (M / "temporalCoherence_mintpy.h5").exists() or str(getattr(cfg, "backend", "")) == "miaplpy":
        # phase-linked route: MintPy's network coherence was set aside as *_mintpy
        layer = M / "geo" / "geo_temporalCoherence_mintpy.h5" if on_geo else M / "temporalCoherence_mintpy.h5"
    else:
        # elsewhere MintPy's own temporal coherence IS the unwrapping-consistency layer
        layer = prod.get("tc_geo") or (M / "temporalCoherence.h5")
    rec = {"min_unwrap_consistency": thr, "applied": thr > 0, "layer": str(layer) if Path(layer).exists() else None}
    if thr <= 0:
        return rec
    if not Path(layer).exists():
        raise RuntimeError(f"min_unwrap_consistency is {thr:g} but this run produced no unwrapping-consistency "
                           f"layer ({Path(layer).relative_to(M)}) - the second quality gate cannot be applied. Set "
                           "min_unwrap_consistency: 0 or fix the geocoding (mintpy_geocode_extra.log).")
    mask_f = prod.get("mask_geo") if on_geo else M / "maskTempCoh.h5"
    try:
        with h5py.File(layer, "r") as f:
            um = f["temporalCoherence"][:]
        with h5py.File(mask_f, "r") as f:
            m = f["mask"][:].astype(bool)
        if um.shape == m.shape:
            kept = m & np.isfinite(um) & (um >= thr)
            rec.update(first_gate_cells=int(m.sum()), both_gates_cells=int(kept.sum()),
                       kept_percent_of_first_gate=round(100.0 * kept.sum() / max(int(m.sum()), 1), 2))
            log.info("Second quality gate (MintPy unwrapping consistency >= %g) keeps %.1f%% of the cells the "
                     "first gate keeps", thr, rec["kept_percent_of_first_gate"])
    except (OSError, KeyError, TypeError) as exc:
        rec["note"] = f"effect not measured: {exc}"
    return rec


def run_products_record(cfg: PipelineConfig, prod: dict, *, started: float, stamp: str, retired, era5, ramp,
                        extra, gate, applied, inversion, iono_rec, plots_rc, end_state) -> dict:
    """mintpy/run_products.json: which files THIS run produced and what was
    applied, read back from the products (absolute paths)."""
    import h5py
    M = cfg.mintpy_dir

    def ab(p):
        return str(Path(p).absolute()) if p else None

    def s(v):
        return v.decode() if isinstance(v, bytes) else str(v)

    chain = prod["chain"]
    vel = prod.get("velocity_geo") or prod.get("velocity_radar")
    with h5py.File(vel, "r") as f:
        va = {k: s(v) for k, v in f.attrs.items()}
    with h5py.File(chain["final"], "r") as f:
        fa = {k: s(v) for k, v in f.attrs.items()}
        dates = [s(d) for d in f["date"][:]]
    tokens = list(chain["tokens"])
    iono = bool(fa.get("IONO_GIM_APPLIED")) or "ion" in tokens
    if fa.get("IONO_GIM_APPLIED"):
        tokens.append("GIM")
    noise = None
    with h5py.File(M / "maskTempCoh.h5", "r") as f:
        v = f.attrs.get("linked_noise_level")
        try:
            noise = float(s(v)) if v is not None else None
            noise = None if noise is None or not np.isfinite(noise) else noise
        except ValueError:
            noise = None
    pair_noise = None
    if (M / "disturbed.h5").exists():
        with h5py.File(M / "disturbed.h5", "r") as f:
            try:
                pair_noise = float(s(f.attrs.get("pair_noise_level", "nan")))
            except ValueError:
                pair_noise = None
    layers = {p.name: p for p in extra}

    def num(k, conv):
        try:
            return conv(va[k]) if k in va else None
        except ValueError:
            return None

    return {
        "timeseries_radar": ab(prod.get("timeseries_radar")),
        "timeseries_geo": ab(prod.get("timeseries_geo")),
        "velocity_geo": ab(prod.get("velocity_geo")),
        "velocity_radar": ab(prod.get("velocity_radar")),
        "ref_y": num("REF_Y", lambda x: int(float(x))), "ref_x": num("REF_X", lambda x: int(float(x))),
        "ref_lat": num("REF_LAT", float), "ref_lon": num("REF_LON", float),
        "ref_date": va.get("REF_DATE"),
        "start_date": dates[0] if dates else None, "end_date": dates[-1] if dates else None,
        "n_dates": len(dates),
        "started": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(started)),
        "finished": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "run_stamp": stamp,
        "superseded_dir": ab(retired),
        "corrections_applied": tokens,
        "era5_missing_dates": list((era5 or {}).get("missing_dates") or []),
        "era5": era5,
        "tide_applied": "SET" in tokens,
        "iono_applied": iono,
        "iono_method": ("IGS global ionosphere maps" if fa.get("IONO_GIM_APPLIED")
                        else "split spectrum" if "ion" in tokens else None),
        "iono_result": iono_rec,
        "removed_ramp": ramp,
        "unwrap_error": {k: applied.get(k) for k in ("requested", "applied", "dataset", "stale_not_used",
                                                     "closure_correction") if k in applied},
        "inversion": inversion,
        "deramp_mask_file": (end_state or {}).get("mintpy.deramp.maskFile"),
        "linked_noise_level": noise,
        "min_temporal_coherence": float(cfg.min_temporal_coherence),
        "pair_noise_level": pair_noise,
        "extra_layers": {"temporal_coherence_mintpy": ab(layers.get("geo_temporalCoherence_mintpy.h5")),
                         "dem_error": ab(layers.get("geo_demErr.h5")),
                         "disturbed": ab(layers.get("geo_disturbed.h5"))},
        "second_gate": gate,
        "plots_exit_code": plots_rc,
    }


def _report_mintpy_end_state(cfg: PipelineConfig) -> dict:
    """
    Read back what MintPy actually did from the product attributes, and say
    so. The log is what was attempted; geo_velocity.h5 is what is true.
    A deramp that did not use the exclusion mask RAISES: the ramp was fitted
    through the deforming area and subtracted from it, and a log line in a
    multi-hour log was all that ever showed it.
    """
    import h5py
    for cand in (cfg.mintpy_dir / "geo" / "geo_velocity.h5", cfg.mintpy_dir / "velocity.h5"):
        if cand.exists():
            break
    else:
        return {}
    keys = ("mintpy.networkInversion.obsDatasetName", "mintpy.networkInversion.numIfgram",
            "mintpy.deramp", "mintpy.deramp.maskFile", "mintpy.troposphericDelay.method",
            "mintpy.topographicResidual", "REF_DATE")
    with h5py.File(cand, "r") as f:
        got = {k: (f.attrs[k].decode() if isinstance(f.attrs[k], bytes) else str(f.attrs[k]))
               for k in keys if k in f.attrs}
    for k in keys:
        log.info("  %-46s %s", k, got.get(k, "(not recorded)"))
    mask = got.get("mintpy.deramp.maskFile", "")
    if got.get("mintpy.deramp", "no") != "no" and "maskDeramp" not in mask and \
            not getattr(cfg, "mintpy_deramp_mask", ""):
        raise RuntimeError(f"THE DERAMP DID NOT USE THE EXCLUSION MASK (maskFile = {mask!r}): the ramp was "
                           "fitted through the deforming area and subtracted from it. Check "
                           "mintpy.deramp.maskFile in mintpy/smallbaselineApp.cfg and re-run from MintPy.")
    return got
