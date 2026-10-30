"""Tropospheric correction for the OPERA DISP-S1 time series.

DISP-S1 applies no tropospheric correction and ships no layer for one -
`ceos_atmospheric_phase_correction` reads `None` in every granule. At
these length scales troposphere is the dominant error source, and it is
the same one that made the Philadelphia stack uninterpretable. This module
estimates it with RAiDER and removes it.

WHAT THIS CORRECTS, AND WHAT IT DOES NOT
----------------------------------------
The delay has two parts that behave very differently for a small AOI:

  * A near-uniform offset across the scene. This is large - a couple of
    metres of zenith hydrostatic delay - and almost entirely irrelevant,
    because it cancels against the reference pixel. Removing it changes
    nothing.

  * Spatial structure across the AOI. This is what corrupts a
    displacement map, and it is the part worth having.

ERA5's grid is ~30 km. Over a 7 km AOI it therefore resolves almost no
structure, and the correction is mostly the first kind - which the
reference point already removed. **Expect it to do very little at
Site A, and expect it to matter at a site with topography or a larger
footprint.** The module reports the spatial standard deviation of what it
removed so this can be judged rather than assumed. A correction whose
spatial std is 1 mm did nothing, however impressive the mean looks.

The elevation-dependent ("stratified") component is where ERA5 earns its
keep, and Site A is flat, which is the other reason to expect little
here.

SIGN
----
RAiDER returns a one-way path delay in metres, positive meaning a longer
path. A longer path makes the target appear further away, which in this
pipeline's convention (positive = motion toward the satellite) is a more
negative displacement. So the correction ADDS the differential delay:

    d_corrected(t) = d_observed(t) + [delay(t) - delay(t_0)]

That is derived, not assumed, and it is then CHECKED: a correct
tropospheric correction reduces the scatter of a time series about its
own trend. If it increases it, something is wrong - the sign, the units,
or the weather model - and the module says so rather than returning a
worse answer confidently.

COST
----
Cost depends on the model. ERA5 is a queued request to the Copernicus
CDS and a decade at 12-day cadence (254 requests) can take far longer
than the InSAR streaming itself. HRRR comes from AWS Open Data via
Herbie - no queue, no account - and is 3 km against ERA5's ~31 km, which
over a small AOI is the difference between resolving structure and not.
HRRR is CONUS only and starts 2016-07-15. Delays are cached per date under
`<project>/tropo_corrections/`, so a re-run is cheap and an interrupted
run resumes. This is why `disp_tropo_correction` defaults to False:
turning it on should be a decision, not a surprise.

Requires CDS credentials in ~/.cdsapirc and the ERA5 licence accepted on
the Copernicus site, which is a separate step from registering.
"""
import logging
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np

from .config import PipelineConfig

log = logging.getLogger(__name__)


# First date each weather model can serve. RAiDER knows these internally
# and raises partway through a run when asked for something earlier; a
# stack of 254 epochs is a bad place to discover that, so the range is
# checked before any download starts.
#
# HRRR is the one that matters here: 3 km against ERA5's ~31 km, no CDS
# queue (it comes from AWS Open Data via Herbie), CONUS only - and valid
# from 2016-07-15, which is three days before the Site A stack's
# first epoch. That is uncomfortably close to the edge.
MODEL_START = {
    "HRRR": datetime(2016, 7, 15),
    "HRRRAK": datetime(2018, 7, 13),
    "ERA5": datetime(1950, 1, 1),
    "ERA5T": datetime(1950, 1, 1),
    "GMAO": datetime(2014, 2, 20),
    "MERRA2": datetime(1980, 1, 1),
}

# Models that only cover the continental US.
CONUS_ONLY = {"HRRR"}


def check_model_range(cfg: PipelineConfig, dates: list[str]) -> tuple[bool, str]:
    """
    Can this weather model serve every epoch, before anything is downloaded?

    Returns (ok, message). A partial correction is worse than none - it
    puts a step the size of the correction into the series - so the
    caller should abandon rather than proceed on a subset.
    """
    model = (cfg.weather_model or "").upper()
    start = MODEL_START.get(model)
    if start is None:
        return True, f"{model}: no start date recorded here; RAiDER will decide"

    too_early = [d for d in dates
                 if datetime.strptime(d, "%Y%m%d") < start]
    if too_early:
        return False, (
            f"{model} has no data before {start:%Y-%m-%d}, but "
            f"{len(too_early)} epoch(s) precede it, the earliest being "
            f"{too_early[0]}. Correcting only the later epochs would put a "
            f"step into the series that the ground never had. Either use a "
            f"model with a longer archive (ERA5 reaches back to 1950) or "
            f"start the stack after {start:%Y-%m-%d}.")

    if model in CONUS_ONLY:
        min_lon, min_lat, max_lon, max_lat = cfg.bbox
        if not (-130 < min_lon and max_lon < -60 and 20 < min_lat and max_lat < 55):
            return False, (f"{model} covers the continental US only, and this "
                           f"AOI ({min_lon:.2f}, {min_lat:.2f}) to "
                           f"({max_lon:.2f}, {max_lat:.2f}) falls outside it.")

    margin = (datetime.strptime(min(dates), "%Y%m%d") - start).days
    note = f"{model} covers all {len(dates)} epochs"
    if margin < 30:
        note += (f" - but only by {margin} day(s). The first epoch sits "
                 f"almost on the edge of the archive, so a single missing "
                 f"file there will abandon the whole correction.")
    return True, note


def _config_text(cfg: PipelineConfig, dt: datetime, out_dir: Path) -> str:
    """
    A RAiDER run-config for one acquisition, in zenith mode.

    No `los_file` is given, so RAiDER returns the ZENITH delay and this
    module projects it to the line of sight itself using the per-pixel
    incidence angle from DISP-S1-STATIC. That is deliberate: RAiDER's
    slant modes want either an ISCE LOS raster or a Sentinel-1 .EOF
    statevector file, neither of which exists on the DISP path, and
    dividing by cos(incidence) with geometry we already have per-pixel is
    both simpler and easier to check.

    The trade is RAiDER's raytracing, which integrates along the true ray
    path through the 3D model and is more accurate in high relief. For a
    flat AOI the projection is a good approximation; over mountains it is
    not, and the .EOF route would be worth the trouble there.
    """
    min_lon, min_lat, max_lon, max_lat = cfg.bbox
    # A margin so the interpolation at the AOI edge is not extrapolating.
    pad = 0.15
    # NISAR looks left, Sentinel-1 right (only read when RAiDER ray-traces)
    look = "left" if str(getattr(cfg, "backend", "")) == "nisar" else "right"
    return f"""\
look_dir: {look}
weather_model: {cfg.weather_model}

date_group:
  date_start: {dt.strftime('%Y%m%d')}

time_group:
  time: "{dt.strftime('%H:%M:%S')}"
  interpolate_time: 'center_time'

aoi_group:
  bounding_box: {min_lat - pad} {max_lat + pad} {min_lon - pad} {max_lon + pad}

height_group:
  use_dem_latlon: False

los_group:
  ray_trace: False

runtime_group:
  verbose: True
  file_format: GTiff
  output_directory: {out_dir}
"""


def delay_cube_for_date(cfg: PipelineConfig, dt: datetime) -> Path | None:
    """Run RAiDER for one acquisition datetime, or reuse a cached cube."""
    # Cache per MODEL as well as per date. Keyed on the date alone,
    # switching weather_model on a project that already has delays
    # would silently reuse the other model's cubes and label the
    # result with the new model's name - an ERA5 correction
    # reported as HRRR, with nothing anywhere to reveal it.
    model = (cfg.weather_model or "unknown").upper()
    out_dir = cfg.tropo_dir / model / dt.strftime("%Y%m%d")
    existing = sorted(out_dir.glob("*.nc")) if out_dir.exists() else []
    if existing:
        return existing[0]

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "raider_config.yaml").write_text(_config_text(cfg, dt, out_dir))

    cmd = ["raider.py", str(out_dir / "raider_config.yaml")]
    if cfg.raider_env:
        # RAiDER cannot share an environment with ISCE2 - they pin
        # conflicting numpy and xarray versions, and installing both
        # silently resolves to a RAiDER old enough to crash on np.float.
        cmd = ["conda", "run", "-n", cfg.raider_env, "--no-capture-output"] + cmd

    # in its own process group, registered: a stopped run takes RAiDER down too
    from . import procs
    result = procs.run_group(cmd, cwd=out_dir)
    text = result.stdout + result.stderr
    cache = out_dir / "weather_files"
    if result.returncode != 0 and cache.exists():
        # Once more from a clean cache. A run stopped part-way leaves a cut-off
        # model file in weather_files/, and Herbie reuses it ("IDX @ local")
        # on every later attempt: Site E descending (27 Oct 2026) failed on
        # 20180305, 20211027 and 20240226 with "cannot rename 'gh'" and
        # "'str' object has no attribute '_coord_names'"; all three gave a
        # cube from a fresh download.
        shutil.rmtree(cache, ignore_errors=True)
        log.info("RAiDER failed for %s; retrying with its download cache cleared", dt.date())
        result = procs.run_group(cmd, cwd=out_dir)
        text += "\n===== retry with the download cache cleared =====\n" + result.stdout + result.stderr
    (out_dir / "raider_run.log").write_text(text)
    if result.returncode != 0:
        log.warning("RAiDER failed for %s - see %s", dt.date(),
                    out_dir / "raider_run.log")
        return None

    produced = sorted(out_dir.glob("*.nc"))
    if not produced:
        log.warning("RAiDER exited cleanly for %s but wrote no delay cube",
                    dt.date())
        return None
    return produced[0]


# Physical bounds on a RAiDER cube's zenith delays (metres). The cube spans
# fixed heights from just below sea level, so its largest hydrostatic delay is
# the sea-level one, ~2.3 m anywhere on Earth (2.36-2.45 m over the 297 dates
# of the first city site); the wet delay is 0-0.5 m. The first city site's
# 2018-08-04 HRRR cube read -5,787 m hydrostatic and a negative wet delay,
# RAiDER exiting cleanly: subtracted, it put a 58 m step into that date, and
# MintPy's DEM-error fit spread it into every other (30 Oct 2026).
HYDRO_MAX_RANGE_M = (1.5, 3.5)
WET_MAX_M = 1.0
NEGATIVE_TOLERANCE_M = -0.01


def cube_problem(cube_path: Path) -> str:
    """Why a delay cube cannot be physically right (its numbers), or "" when it can."""
    import xarray as xr
    try:
        with xr.open_dataset(cube_path) as ds:
            if "wet" not in ds or "hydro" not in ds:
                return f"no wet/hydro variables (found {list(ds.data_vars)})"
            h, w = ds["hydro"].values, ds["wet"].values
    except Exception as exc:                                     # noqa: BLE001
        return f"could not be read ({exc})"
    if not (np.isfinite(h).any() and np.isfinite(w).any()):
        return "holds no finite delays"
    hmax, hmin, wmax, wmin = (float(f(a)) for f, a in ((np.nanmax, h), (np.nanmin, h), (np.nanmax, w), (np.nanmin, w)))
    lo, hi = HYDRO_MAX_RANGE_M
    if not lo <= hmax <= hi or hmin < NEGATIVE_TOLERANCE_M:
        return f"hydrostatic delay {hmin:.4g} to {hmax:.4g} m, where sea level is ~2.3 m and nothing is negative"
    if wmax > WET_MAX_M or wmin < NEGATIVE_TOLERANCE_M:
        return f"wet delay {wmin:.4g} to {wmax:.4g} m, outside 0 to {WET_MAX_M:g} m"
    return ""


def _sample_cube(cube_path: Path, lon: np.ndarray, lat: np.ndarray,
                 height: np.ndarray) -> np.ndarray | None:
    """
    Zenith delay in metres at each pixel, from RAiDER's 3D cube.

    The cube spans height levels from below sea level to the top of the
    atmosphere, so producing a 2D map means sampling it at each pixel's
    own terrain elevation - which is the entire point of the stratified
    correction and the reason a DEM is needed here.
    """
    import xarray as xr
    from scipy.interpolate import RegularGridInterpolator

    why = cube_problem(cube_path)
    if why:
        log.warning("%s refused: its delays are not physical (%s)", cube_path.name, why)
        return None
    try:
        ds = xr.open_dataset(cube_path)
    except Exception as exc:
        log.warning("Could not open %s (%s)", cube_path.name, exc)
        return None

    try:
        if "wet" not in ds or "hydro" not in ds:
            log.warning("%s has no wet/hydro variables (found %s)",
                        cube_path.name, list(ds.data_vars))
            return None
        total = (ds["wet"] + ds["hydro"]).values      # (z, y, x), metres
        z = ds["z"].values
        y = ds["y"].values
        x = ds["x"].values
        # RegularGridInterpolator needs ascending coordinates; RAiDER
        # writes latitude north-to-south.
        if y[0] > y[-1]:
            y, total = y[::-1], total[:, ::-1, :]
        if x[0] > x[-1]:
            x, total = x[::-1], total[:, :, ::-1]
        interp = RegularGridInterpolator((z, y, x), total,
                                         bounds_error=False, fill_value=None)
        pts = np.column_stack([height.ravel(), lat.ravel(), lon.ravel()])
        return interp(pts).reshape(height.shape).astype("float32")
    finally:
        ds.close()


def differential_los_delay(cfg: PipelineConfig, dates: list[str],
                           acq_time: str, grid: dict,
                           incidence_deg: np.ndarray,
                           height: np.ndarray) -> tuple[np.ndarray | None, dict]:
    """
    LOS delay at every epoch relative to the first, in metres.

    Returns (cube, info). `cube[i]` is what must be ADDED to epoch i to
    remove the troposphere - see the module docstring on sign. Epochs
    RAiDER could not produce are NaN in the cube, and the caller must
    leave those epochs alone rather than treating a missing correction as
    zero: a partly-corrected stack has a step in it that the data never
    had, which is worse than an uncorrected one.
    """
    from pyproj import Transformer

    tr = Transformer.from_crs(f"EPSG:{grid['epsg']}", "EPSG:4326",
                              always_xy=True)
    r, c = np.mgrid[0:grid["ny"], 0:grid["nx"]]
    xs = grid["x0"] + c * grid["dx"]
    ys = grid["y0"] + r * grid["dy"]
    lon, lat = tr.transform(xs, ys)

    cos_inc = np.cos(np.radians(incidence_deg))
    cos_inc = np.where(np.abs(cos_inc) > 0.2, cos_inc, np.nan)

    zenith0 = None
    cube = np.full((len(dates), grid["ny"], grid["nx"]), np.nan, dtype="float32")
    ok, failed = 0, []
    for i, d in enumerate(dates):
        dt = datetime.strptime(f"{d} {acq_time}", "%Y%m%d %H:%M:%S")
        path = delay_cube_for_date(cfg, dt)
        if path is None:
            failed.append(d)
            continue
        z = _sample_cube(path, lon, lat, height)
        if z is None:
            failed.append(d)
            continue
        if zenith0 is None:
            if i != 0:
                log.warning("The first epoch (%s) has no delay, so the "
                            "correction is referenced to %s instead. That is "
                            "a constant offset, not an error.", dates[0], d)
            zenith0 = z
        # Zenith -> slant, then differenced against the reference epoch.
        cube[i] = ((z - zenith0) / cos_inc).astype("float32")
        ok += 1
        if ok % 20 == 0:
            log.info("  tropospheric delay: %d / %d epochs", ok, len(dates))

    if ok == 0:
        return None, {"epochs_corrected": 0, "epochs_failed": len(dates),
                      "failed_dates": failed}

    per_epoch_std = [float(np.nanstd(cube[i])) for i in range(len(dates))
                     if np.isfinite(cube[i]).any()]
    info = {
        "epochs_corrected": ok,
        "epochs_failed": len(failed),
        "failed_dates": failed[:12],
        "spatial_std_mm_median": float(np.median(per_epoch_std)) * 1000,
        "spatial_std_mm_max": float(np.max(per_epoch_std)) * 1000,
        "magnitude_mm_p95": float(np.nanpercentile(np.abs(cube), 95)) * 1000,
    }
    return cube, info


def scatter_about_trend(ts: np.ndarray, t: np.ndarray) -> float:
    """
    Median per-pixel RMS about a straight line - the thing a correction
    should reduce.

    This is the check that makes the correction defensible. A
    tropospheric correction removes a signal that is uncorrelated with
    time, so it must tighten the series about its trend. If it loosens
    them, the correction is wrong in sign, units or model, and applying
    it makes the product worse while looking exactly as convincing.
    """
    flat = ts.reshape(ts.shape[0], -1)
    finite = np.isfinite(flat)
    if ts.shape[0] < 5:
        return float("nan")   # too short to fit a trend and judge the residual
    keep = np.where(finite.sum(axis=0) >= max(4, ts.shape[0] // 2))[0]
    if keep.size == 0:
        return float("nan")
    # Sample for speed; the median over thousands of pixels is stable.
    if keep.size > 4000:
        keep = keep[:: keep.size // 4000]
    rms = []
    for j in keep:
        m = finite[:, j]
        A = np.vstack([t[m], np.ones(m.sum())]).T
        sol, *_ = np.linalg.lstsq(A, flat[m, j], rcond=None)
        rms.append(np.sqrt(np.mean((flat[m, j] - A @ sol) ** 2)))
    return float(np.median(rms))
