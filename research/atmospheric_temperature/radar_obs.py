"""Read-only radar-observation exporter: observed, modelled and residual
delay as SEPARATE, labelled products.

Phase 1 of research/atmospheric_temperature/ASSESSMENT.md (section 2.2 item 8).
It reads a finished MintPy run of the pipeline and the RAiDER cubes cached
beside it, and writes into an output folder of its own. It never writes into
the project.

WHAT IT WRITES (radar_obs.h5, radar_obs.json in out_dir)
-------------------------------------------------------
  model_hydro   kind=model     RAiDER hydrostatic delay, sampled at every
                               pixel's height, zenith / cos(incidence)
  model_wet     kind=model     the same for the wet delay
  residual      kind=residual  observation - model_hydro - model_wet
  date          the series' dates

All three are in the OBSERVATION's convention, so they subtract directly:
metres of line-of-sight displacement (positive toward the satellite; a longer
path is negative), referenced like the observation - minus the reference
date's field, minus the reference pixel. That is exactly what MintPy's
diff.py does with inputs/<MODEL>.h5 (weather_raider.py docstring), and
export() CHECKS it: model_hydro + model_wet must reproduce
observation - timeseries_<...>_<MODEL>.h5 (the correction MintPy actually
applied), and the measured difference is recorded.

The observation itself is not copied: it is the series MintPy's weather step
read (timeseries.h5 on Sentinel-1; timeseries_ion.h5 on NISAR, i.e. after the
split-spectrum ionosphere), named with its size, mtime and sha256 in the
record. The record also lists what had touched it (step_history): the
reference pixel and date, the corrections before the weather step, and the
steps AFTER it (deramp, DEM error), which do not touch these products but do
touch MintPy's own timeseriesResidual.h5.

WHAT THE RESIDUAL STILL CONTAINS
--------------------------------
residual = (true delay - model delay), differenced against the reference
date and pixel, PLUS ground deformation, DEM-error phase (proportional to the
perpendicular baseline), unwrapping errors, decorrelation noise and, on the
phase-linked routes, the phase-linking bias. It is not an atmospheric
measurement until those are dealt with; observability.py works on
short-interval differences and records which it used.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

PRODUCTS = {
    "model_hydro": ("model", "RAiDER hydrostatic delay, LOS, observation convention"),
    "model_wet": ("model", "RAiDER wet delay, LOS, observation convention"),
    "residual": ("residual", "observation - model_hydro - model_wet; still holds deformation, DEM error, "
                             "unwrapping errors, noise and phase-linking bias"),
}
CFG_KEYS = ("mintpy.troposphericDelay.method", "mintpy.troposphericDelay.weatherModel", "mintpy.deramp",
            "mintpy.topographicResidual", "mintpy.topographicResidual.polyOrder",
            "mintpy.reference.lalo", "mintpy.reference.yx", "mintpy.reference.date",
            "mintpy.ionosphericDelay.method")


# ---------------------------------------------------------------- files
def sha256(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def file_record(path: Path, digest: bool = True) -> dict:
    st = path.stat()
    rec = {"path": str(path), "bytes": st.st_size, "mtime": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(st.st_mtime))}
    if digest:
        rec["sha256"] = sha256(path)
    return rec


def _attrs(h) -> dict:
    return {k: (v.decode() if isinstance(v, bytes) else v) for k, v in h.attrs.items()}


def _cfg_values(cfg_path: Path) -> dict:
    out = {}
    if not cfg_path.exists():
        return out
    for line in cfg_path.read_text().splitlines():
        m = re.match(r"\s*([\w.]+)\s*=\s*([^#]*)", line)
        if m and m.group(1) in CFG_KEYS:
            out[m.group(1)] = m.group(2).strip()
    return out


def find_series(mintpy_dir: Path, model: str = "HRRR") -> dict:
    """The weather-corrected series (timeseries[_x]_<MODEL>.h5, without later
    suffixes) and the observation it was made from (the same name without
    _<MODEL>). Raises when either is missing."""
    mintpy_dir = Path(mintpy_dir)
    cands = sorted(p for p in mintpy_dir.glob(f"timeseries*_{model}.h5"))
    if not cands:
        raise FileNotFoundError(f"no timeseries*_{model}.h5 in {mintpy_dir}")
    if len(cands) > 1:
        raise RuntimeError(f"more than one weather-corrected series in {mintpy_dir}: {[c.name for c in cands]}")
    corrected = cands[0]
    observed = corrected.with_name(corrected.name.replace(f"_{model}.h5", ".h5"))
    if not observed.exists():
        raise FileNotFoundError(f"{observed.name} (the input of the weather step) is missing")
    before = observed.stem.split("_")[1:]          # e.g. ['ion'] on NISAR
    return {"observed": observed, "corrected": corrected, "corrections_before_weather": before}


def step_history(mintpy_dir: Path, series: dict) -> dict:
    """What touched the observation, read from the run's own records."""
    import h5py
    mintpy_dir = Path(mintpy_dir)
    cfg = mintpy_dir / "smallbaselineApp.cfg"
    if not cfg.exists():
        cfg = mintpy_dir / "inputs" / "smallbaselineApp.cfg"
    with h5py.File(series["observed"], "r") as f:
        a = _attrs(f)
    hist = {"config_file": str(cfg) if cfg.exists() else None, "config": _cfg_values(cfg),
            "observation_file": series["observed"].name,
            "corrections_before_weather": series["corrections_before_weather"],
            "reference": {k: a.get(k) for k in ("REF_DATE", "REF_Y", "REF_X", "REF_LAT", "REF_LON")},
            "grid": {k: a.get(k) for k in ("LENGTH", "WIDTH", "X_FIRST", "Y_FIRST", "X_STEP", "Y_STEP", "EPSG",
                                          "RLOOKS", "ALOOKS", "RANGE_PIXEL_SIZE", "AZIMUTH_PIXEL_SIZE")},
            "sensor": {k: a.get(k) for k in ("PLATFORM", "WAVELENGTH", "ORBIT_DIRECTION", "CENTER_LINE_UTC",
                                             "PROCESSOR", "UNIT")}}
    for name in ("run_products.json", "weather_raider.json", "deramp_effect.json"):
        p = mintpy_dir / name
        if p.exists():
            try:
                d = json.loads(p.read_text())
            except ValueError:
                continue
            if name == "run_products.json":
                d = {k: d.get(k) for k in ("corrections_applied", "ref_date", "n_dates", "finished", "removed_ramp",
                                           "iono_applied", "iono_method")}
            hist[name] = d
    hist["after_weather_step_not_in_these_products"] = [
        s for s, on in (("deramp", hist["config"].get("mintpy.deramp", "no").split()[0] not in ("no", "auto")),
                        ("topographicResidual (DEM error + polynomial deformation)",
                         hist["config"].get("mintpy.topographicResidual", "").startswith("yes"))) if on]
    return hist


# ---------------------------------------------------------------- geometry
def geometry(mintpy_dir: Path) -> dict:
    """lat, lon, height (m), incidence (deg) per pixel, from the run's
    geometry file. Geocoded grids: pixel CENTRES from X_FIRST/Y_FIRST (the
    outer corner, lband.Grid) - EPSG 4326 only."""
    import h5py
    inputs = Path(mintpy_dir) / "inputs"
    radar, geo = inputs / "geometryRadar.h5", inputs / "geometryGeo.h5"
    path = radar if radar.exists() else geo
    if not path.exists():
        raise FileNotFoundError(f"no geometryRadar.h5 or geometryGeo.h5 in {inputs}")
    with h5py.File(path, "r") as g:
        a = _attrs(g)
        height = g["height"][()].astype("float64")
        inc = g["incidenceAngle"][()].astype("float64")
        if "latitude" in g and "longitude" in g:
            lat, lon = g["latitude"][()].astype("float64"), g["longitude"][()].astype("float64")
        else:
            if str(a.get("EPSG", "4326")) != "4326":
                raise ValueError(f"{path.name}: EPSG {a.get('EPSG')}; only lat/lon grids are read")
            ny, nx = height.shape
            x = float(a["X_FIRST"]) + (np.arange(nx) + 0.5) * float(a["X_STEP"])
            y = float(a["Y_FIRST"]) + (np.arange(ny) + 0.5) * float(a["Y_STEP"])
            lon, lat = np.meshgrid(x, y)
    valid = np.isfinite(height) & np.isfinite(inc) & (inc > 0) & np.isfinite(lat) & np.isfinite(lon)
    return {"file": path, "lat": lat, "lon": lon, "height": height, "incidence": inc, "valid": valid}


# ---------------------------------------------------------------- the model
def sample_cube_components(cube_path: Path, lon, lat, height) -> tuple[np.ndarray, np.ndarray] | None:
    """(hydro, wet) zenith delay in metres at each pixel, from RAiDER's 3D
    cube, interpolated linearly in (height, lat, lon) exactly as
    tropo_disp._sample_cube does - which returns their SUM. Kept apart here
    because the research needs the components (ASSESSMENT.md C4)."""
    import xarray as xr
    from scipy.interpolate import RegularGridInterpolator
    with xr.open_dataset(cube_path) as ds:
        if "wet" not in ds or "hydro" not in ds:
            log.warning("%s has no wet/hydro variables", Path(cube_path).name)
            return None
        z, y, x = ds["z"].values, ds["y"].values, ds["x"].values
        out = []
        for name in ("hydro", "wet"):
            v = ds[name].values
            yy, xx = y, x
            if yy[0] > yy[-1]:
                yy, v = yy[::-1], v[:, ::-1, :]
            if xx[0] > xx[-1]:
                xx, v = xx[::-1], v[:, :, ::-1]
            itp = RegularGridInterpolator((z, yy, xx), v, bounds_error=False, fill_value=None)
            pts = np.column_stack([np.ravel(height), np.ravel(lat), np.ravel(lon)])
            out.append(itp(pts).reshape(np.shape(height)))
    return out[0], out[1]


def cube_for_date(tropo_dir: Path, model: str, date: str) -> Path | None:
    d = Path(tropo_dir) / model.upper() / date
    found = sorted(d.glob("*.nc")) if d.exists() else []
    return found[0] if found else None


def _reference(stack: np.ndarray, ref_index: int, ref_y: int, ref_x: int) -> np.ndarray:
    """MintPy's referencing: minus the reference date's field, then minus
    the reference pixel of each date."""
    out = stack - stack[ref_index]
    return out - out[:, ref_y:ref_y + 1, ref_x:ref_x + 1]


# ---------------------------------------------------------------- export
def export(project_dir: Path, out_dir: Path, model: str = "HRRR", digest: bool = True) -> dict:
    """Write radar_obs.h5 + radar_obs.json for one project (module docstring).
    Refuses to write inside the project."""
    import h5py
    project_dir, out_dir = Path(project_dir).resolve(), Path(out_dir).resolve()
    if out_dir == project_dir or project_dir in out_dir.parents:
        raise ValueError(f"out_dir {out_dir} is inside the project {project_dir}: this exporter is read-only")
    mintpy_dir = project_dir / "mintpy"
    series = find_series(mintpy_dir, model)
    hist = step_history(mintpy_dir, series)
    geom = geometry(mintpy_dir)
    with h5py.File(series["observed"], "r") as f:
        a = _attrs(f)
        dates = [d.decode() if isinstance(d, bytes) else str(d) for d in f["date"][()]]
        obs = f["timeseries"][()].astype("float64")
    ref_date, ref_y, ref_x = str(a["REF_DATE"]), int(a["REF_Y"]), int(a["REF_X"])
    if obs.shape[1:] != geom["height"].shape:
        raise RuntimeError(f"geometry {geom['height'].shape} and series {obs.shape[1:]} are not the same grid")
    cos_inc = np.cos(np.radians(np.where(geom["valid"], geom["incidence"], 0.0)))
    lat = np.where(geom["valid"], geom["lat"], 0.0)
    lon = np.where(geom["valid"], geom["lon"], 0.0)
    hgt = np.where(geom["valid"], geom["height"], 0.0)

    hydro = np.full(obs.shape, np.nan)
    wet = np.full(obs.shape, np.nan)
    cubes, missing = {}, []
    for i, d in enumerate(dates):
        c = cube_for_date(project_dir / "tropo_corrections", model, d)
        comp = sample_cube_components(c, lon, lat, hgt) if c else None
        if comp is None:
            missing.append(d)
            continue
        cubes[d] = c.name
        # pyaps3 / weather_raider sign: the delay file holds -zenith/cos(inc)
        hydro[i] = np.where(geom["valid"], -comp[0] / cos_inc, np.nan)
        wet[i] = np.where(geom["valid"], -comp[1] / cos_inc, np.nan)
    if ref_date not in cubes:
        raise RuntimeError(f"no {model} cube for the reference date {ref_date}: cannot reference the model")
    ri = dates.index(ref_date)
    hydro, wet = _reference(hydro, ri, ref_y, ref_x), _reference(wet, ri, ref_y, ref_x)
    residual = obs - hydro - wet

    # the check: the model must reproduce what MintPy subtracted
    with h5py.File(series["corrected"], "r") as f:
        applied = obs - f["timeseries"][()].astype("float64")
    ok = np.isfinite(applied) & np.isfinite(hydro) & (applied != 0)
    diff = (applied - hydro - wet)[ok]
    over = np.abs(diff) > 1e-4
    check = {"against": series["corrected"].name, "pixels_dates": int(ok.sum()),
             "rms_mm": float(np.sqrt(np.mean(diff ** 2)) * 1e3) if diff.size else None,
             "max_abs_mm": float(np.max(np.abs(diff)) * 1e3) if diff.size else None,
             "pixel_dates_over_0.1mm": int(over.sum()),
             "dates_with_any_over_0.1mm": sorted({dates[i] for i in np.where(ok)[0][over]})}
    # passed: the model reproduces the applied correction everywhere but a negligible
    # share of pixel-dates (isolated odd values in the applied file are listed, not hidden)
    check["passed"] = (check["rms_mm"] is not None and check["rms_mm"] < 0.1
                       and check["pixel_dates_over_0.1mm"] <= 1e-4 * check["pixels_dates"])

    out_dir.mkdir(parents=True, exist_ok=True)
    h5_path, tmp = out_dir / "radar_obs.h5", out_dir / "radar_obs.h5.part"
    with h5py.File(tmp, "w") as f:
        f.attrs.update({"UNIT": "m", "FILE_TYPE": "timeseries", "REF_DATE": ref_date, "REF_Y": ref_y, "REF_X": ref_x,
                        "CONVENTION": "LOS displacement, positive toward the satellite; referenced to REF_DATE and "
                                      "the reference pixel, as the observation",
                        "MODEL": model, "OBSERVATION": str(series["observed"])})
        f["date"] = np.array([d.encode() for d in dates], dtype="S8")
        for name, arr in (("model_hydro", hydro), ("model_wet", wet), ("residual", residual)):
            ds = f.create_dataset(name, data=arr.astype("float32"), compression="gzip", compression_opts=4,
                                  chunks=(1,) + arr.shape[1:])
            ds.attrs["kind"], ds.attrs["description"] = PRODUCTS[name]
            ds.attrs["UNIT"] = "m"
    tmp.replace(h5_path)

    rec = {"written": time.strftime("%Y-%m-%dT%H:%M:%S"), "module": __name__,
           "products": {k: {"kind": v[0], "description": v[1]} for k, v in PRODUCTS.items()},
           "file": str(h5_path), "project": str(project_dir), "model": model,
           "observation": file_record(series["observed"], digest),
           "geometry": file_record(geom["file"], digest),
           "dates": len(dates), "model_dates": len(cubes), "missing_model_dates": missing,
           "cube_projection": "zenith / cos(incidence), no ray tracing (as the pipeline: tropo_disp._config_text)",
           "step_history": hist, "check_against_applied_correction": check,
           "crs": "radar grid with per-pixel lat/lon from the geometry file" if "geometryRadar" in geom["file"].name
                  else "EPSG:4326 pixel centres from X_FIRST/Y_FIRST"}
    (out_dir / "radar_obs.json").write_text(json.dumps(rec, indent=2, default=str))
    log.info("radar_obs: %d dates, check vs %s rms %.4f mm (%s)", len(dates), series["corrected"].name,
             check["rms_mm"] or float("nan"), "passed" if check["passed"] else "FAILED")
    return rec
