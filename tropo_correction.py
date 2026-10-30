"""Stage 4: tropospheric delay estimation (RAiDER) and correction.

RAiDER is driven by a YAML run-config, not command-line flags. This module
writes one config per acquisition, runs raider.py, then applies the
differential delay to the unwrapped interferogram.

Requires CDS credentials in ~/.cdsapirc for ERA5, and the ERA5 licence
accepted on the Copernicus site (a separate step from registering).
"""
import logging
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling

from .config import PipelineConfig

log = logging.getLogger(__name__)


def _acq_datetime(product) -> datetime:
    """Parse the acquisition start time from an ASF product's metadata."""
    return datetime.strptime(product.properties["startTime"][:19], "%Y-%m-%dT%H:%M:%S")


def write_raider_config(cfg: PipelineConfig, product, dem_tif: Path,
                        los_file: Path, out_dir: Path) -> Path:
    """
    Write a RAiDER run-config for one acquisition.

    Uses the projected slant-delay mode (ray_trace: False): the zenith
    delay is divided by cos(incidence angle) using the LOS raster ISCE2
    produces per pair.

    Note this gives up RAiDER's raytracing, which integrates through the
    3D weather model along the true ray path and is the more accurate
    option in high relief. Raytracing needs `orbit_file` pointing at a
    Sentinel-1 .EOF statevector file, NOT the LOS raster — RAiDER tries to
    parse whatever `los_file` holds as statevectors when ray_trace is on.
    Staging .EOF files per acquisition would restore it; for low-relief
    AOIs the projected mode is a reasonable approximation.

    The dem_tif argument is accepted but deliberately not written into the
    config: RAiDER 0.4.x crashes in its own rio_profile() when given a DEM
    path as a string from YAML. Omitting it makes RAiDER fetch Copernicus
    GLO-30 itself, which is the same DEM dem_stitcher supplies anyway.
    """
    dt = _acq_datetime(product)
    min_lon, min_lat, max_lon, max_lat = cfg.bbox
    out_dir.mkdir(parents=True, exist_ok=True)

    # RAiDER wants the bounding box as: South North West East
    config_text = f"""\
look_dir: right
weather_model: {cfg.weather_model}

date_group:
  date_start: {dt.strftime('%Y%m%d')}

time_group:
  time: "{dt.strftime('%H:%M:%S')}"
  interpolate_time: 'center_time'

aoi_group:
  bounding_box: {min_lat} {max_lat} {min_lon} {max_lon}

height_group:
  use_dem_latlon: False

los_group:
  ray_trace: False
  los_file: {los_file}
  los_convention: isce

runtime_group:
  verbose: True
  file_format: GTiff
  output_directory: {out_dir}
"""
    config_path = out_dir / "raider_config.yaml"
    config_path.write_text(config_text)
    return config_path


def run_raider(cfg: PipelineConfig, product, dem_tif: Path, los_file: Path):
    """Run RAiDER for one acquisition; returns the delay raster it produced."""
    dt = _acq_datetime(product)
    # Cache per MODEL as well as per date. Keyed on the date alone,
    # switching weather_model on a project that already has delays
    # would silently reuse the other model's cubes and label the
    # result with the new model's name - an ERA5 correction
    # reported as HRRR, with nothing anywhere to reveal it.
    model = (cfg.weather_model or "unknown").upper()
    out_dir = cfg.tropo_dir / model / dt.strftime("%Y%m%d")

    existing = sorted(out_dir.glob("*_std.nc")) if out_dir.exists() else []
    if existing and cfg.skip_existing:
        log.info("Delay raster already present for %s", dt.date())
        return existing[0]

    config_path = write_raider_config(cfg, product, dem_tif, los_file, out_dir)
    log.info("Running RAiDER for %s", dt.date())
    # RAiDER often can't share an environment with ISCE2 (they pin
    # conflicting numpy/xarray versions), so allow running it in a
    # separate conda env via `conda run`.
    if cfg.raider_env:
        cmd = ["conda", "run", "-n", cfg.raider_env, "--no-capture-output",
               "raider.py", str(config_path)]
    else:
        cmd = ["raider.py", str(config_path)]
    result = subprocess.run(cmd, cwd=out_dir, capture_output=True, text=True)
    (out_dir / "raider_run.log").write_text(result.stdout + result.stderr)

    if result.returncode != 0:
        log.error("RAiDER failed for %s - see %s", dt.date(), out_dir / "raider_run.log")
        return None

    # RAiDER's output filename varies by model and settings, so find it
    # rather than predicting it. Weather model files live in a subfolder,
    # so a top-level glob avoids picking those up.
    # RAiDER writes a netCDF cube (not a GeoTIFF) when given a bounding
    # box, regardless of the file_format setting.
    produced = sorted(out_dir.glob("*_std.nc"))
    if not produced:
        log.error("RAiDER exited cleanly for %s but wrote no delay cube", dt.date())
        return None
    return produced[0]


def _pixel_lonlat(profile: dict):
    """Longitude/latitude of every pixel centre on the target grid."""
    transform = profile["transform"]
    rows = np.arange(profile["height"])
    cols = np.arange(profile["width"])
    col_grid, row_grid = np.meshgrid(cols, rows)
    lon = transform.c + (col_grid + 0.5) * transform.a + (row_grid + 0.5) * transform.b
    lat = transform.f + (col_grid + 0.5) * transform.d + (row_grid + 0.5) * transform.e
    return lon, lat


def _dem_on_grid(dem_tif: Path, target_profile: dict) -> np.ndarray:
    """Resample the DEM onto the interferogram grid, in metres."""
    out = np.zeros((target_profile["height"], target_profile["width"]), dtype="float32")
    with rasterio.open(dem_tif) as src:
        reproject(
            source=src.read(1), destination=out,
            src_transform=src.transform, src_crs=src.crs,
            dst_transform=target_profile["transform"], dst_crs=target_profile["crs"],
            resampling=Resampling.bilinear,
        )
    return out


def sample_delay_cube(cube_path: Path, dem_tif: Path, target_profile: dict) -> np.ndarray:
    """
    Collapse a RAiDER 3D delay cube into a 2D delay map on the
    interferogram grid.

    RAiDER writes a netCDF cube of (z, y, x) with separate `wet` and
    `hydro` components in metres; total delay is their sum. The cube spans
    height levels from below sea level to the top of the atmosphere, so a
    2D delay map means sampling it at each pixel's terrain elevation —
    which is why the DEM is needed here even though RAiDER fetched its own
    internally.
    """
    import xarray as xr
    from scipy.interpolate import RegularGridInterpolator

    ds = xr.open_dataset(cube_path)
    total = (ds["wet"] + ds["hydro"]).values      # (z, y, x), metres
    z = ds["z"].values
    y = ds["y"].values
    x = ds["x"].values

    # RegularGridInterpolator needs strictly ascending coordinates, and
    # RAiDER writes latitude descending (north to south).
    if y[0] > y[-1]:
        y = y[::-1]
        total = total[:, ::-1, :]
    if x[0] > x[-1]:
        x = x[::-1]
        total = total[:, :, ::-1]

    interp = RegularGridInterpolator(
        (z, y, x), total, bounds_error=False, fill_value=None
    )

    lon, lat = _pixel_lonlat(target_profile)
    heights = _dem_on_grid(dem_tif, target_profile)
    pts = np.column_stack([heights.ravel(), lat.ravel(), lon.ravel()])
    delay = interp(pts).reshape(heights.shape)
    ds.close()
    return delay.astype("float32")


def correct_pair(cfg: PipelineConfig, ref, sec, pair_dir: Path, dem_tif: Path):
    """Estimate and remove the differential tropospheric delay for one pair."""
    merged = pair_dir / "merged"
    los_file = merged / "los.rdr.geo"
    unw_path = merged / "filt_topophase.unw.geo"

    if not unw_path.exists():
        log.warning("No unwrapped interferogram in %s - skipping tropo", pair_dir.name)
        return None

    delay_ref_path = run_raider(cfg, ref, dem_tif, los_file)
    delay_sec_path = run_raider(cfg, sec, dem_tif, los_file)
    if delay_ref_path is None or delay_sec_path is None:
        log.warning("Skipping tropo correction for %s - missing delay raster", pair_dir.name)
        return None

    with rasterio.open(unw_path) as src:
        phase = src.read(2)          # band 1 is amplitude, band 2 the phase
        target_profile = src.profile

    d_ref = sample_delay_cube(delay_ref_path, dem_tif, target_profile)
    d_sec = sample_delay_cube(delay_sec_path, dem_tif, target_profile)

    # RAiDER delays are in metres; convert to radians to match the phase.
    wavelength = 0.0555
    diff_rad = (d_sec - d_ref) * (4 * np.pi / wavelength)
    corrected = phase - diff_rad

    # Name it so ".unw" is the recognised extension. MintPy infers file
    # type from the suffix: for .unw it reads band 2 (phase), but for an
    # unrecognised type it reads band 1 - which here is amplitude, and
    # silently produces velocities orders of magnitude too large.
    out_path = merged / "filt_topophase.tropocorrected.unw.geo"

    # Write in ISCE format, not GeoTIFF: MintPy's prep_isce requires an
    # .xml metadata sidecar next to the data, which the ISCE GDAL driver
    # writes and the GTiff driver does not. Keep the original's two-band
    # layout (amplitude, phase) so MintPy reads it exactly as it reads the
    # uncorrected product.
    with rasterio.open(unw_path) as src:
        amplitude = src.read(1)

    profile = dict(target_profile)
    profile.update(driver="ISCE", count=2, dtype="float32")
    for key in ("blockxsize", "blockysize", "tiled", "compress", "interleave"):
        profile.pop(key, None)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(amplitude.astype("float32"), 1)
        dst.write(corrected.astype("float32"), 2)

    # The sidecar the ISCE driver writes carries only the file's basename.
    # Rewrite it with the absolute path so MintPy resolves it regardless of
    # what directory it runs from.
    try:
        import importlib
        importlib.import_module("isce")       # registers ISCE's import paths (the import IS the point)
        import isceobj
        img = isceobj.createImage()
        img.load(str(out_path) + ".xml")
        img.filename = str(out_path)
        img.setAccessMode("READ")
        img.renderVRT()
        img.renderHdr()
    except Exception as exc:
        log.warning("Could not normalise the ISCE sidecar for %s: %s",
                    out_path.name, exc)

    # Record how large the differential delay was. The spatial variation
    # (std), not the mean, is what corrupts an interferogram: a uniform
    # offset cancels out against the reference point, while structure does
    # not. This feeds the screening gate in screening.py.
    diff = diff_rad[np.isfinite(diff_rad)]
    stats = {
        "pair": pair_dir.name,
        "delay_std_rad": float(np.std(diff)),
        "delay_mean_rad": float(np.mean(diff)),
        "delay_std_cm": float(np.std(diff) * wavelength / (4 * np.pi) * 100),
    }
    stats_path = cfg.project_dir / "tropo_screening.json"
    try:
        import json
        existing = json.loads(stats_path.read_text()) if stats_path.exists() else {}
        existing[pair_dir.name] = stats
        stats_path.write_text(json.dumps(existing, indent=2))
    except Exception as exc:
        log.warning("Could not record tropo stats: %s", exc)

    log.info("Tropo correction for %s: differential delay std %.2f rad (%.2f cm)",
             pair_dir.name, stats["delay_std_rad"], stats["delay_std_cm"])
    return out_path
