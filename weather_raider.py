"""Weather delays from RAiDER (NOAA HRRR by default) for the MintPy routes.

mintpy_tropo_method: raider with weather_model: HRRR (24 Oct 2026). The
ERA5 route asks the Copernicus CDS for one file per date, and from 22 Oct
2026 the CDS refused most requests ("Number of available slots for this
dataset has been reached ... Running: 100 - Queued: 2397") and made the
ones it took wait 1-4.6 h. HRRR comes from NOAA's public archive on AWS
through RAiDER: no account, no queue, about 2 minutes per date, and 3 km
against ERA5's ~31 km. It covers the continental US from 15 Jul 2016.

HOW IT REACHES MINTPY
---------------------
MintPy's pyaps branch applies inputs/<weatherModel>.h5 whenever that file
exists on the series' grid ("Use existed tropospheric delay file" ->
diff.py). So the run writes inputs/HRRR.h5 in exactly the layout
tropo_pyaps3.calc_delay_timeseries writes inputs/ERA5.h5, and the template
says method = pyaps, weatherModel = HRRR (corrections.mintpy_tropo_lines).
Everything after that is MintPy's own step and the pipeline's own checks:
the series is named timeseries_HRRR*, and corrections.verify_era5_coverage
stops the run if a date is missing from the file.

  timeseries  (date, length, width) float32, metres: the ABSOLUTE one-way
              line-of-sight delay of each date, sign reversed as pyaps3 does
              (get_delay: "pha *= -1"), so diff.py subtracting it removes it;
  date        the dates; attributes: the series' own, FILE_TYPE timeseries,
              UNIT m, without REF_* (diff.py references the delays to the
              series' reference pixel and date itself).

Each date: RAiDER's zenith delay cube (tropo_disp.delay_cube_for_date,
cached under <project>/tropo_corrections/<MODEL>/<date>/), sampled at every
pixel's latitude, longitude and height from MintPy's geometry file, then
divided by cos(incidence). Pixels without geometry are 0, as pyaps3 leaves
them.
"""
from __future__ import annotations

import json
import logging
import shutil
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

# RAiDER runs at once: each is one HRRR download of two hours of model
# (the hours either side of the pass) and a few seconds of computing.
RAIDER_WORKERS = 4
RECORD = "weather_raider.json"
# Herbie's line in raider_run.log when no archive it searches (AWS, Google,
# Azure, NOMADS, Pando) holds a file RAiDER needs.
ARCHIVE_GAP = "Did not find"


def model_name(cfg) -> str:
    return str(getattr(cfg, "weather_model", "") or "HRRR").upper()


def acquisition_time(cfg) -> str:
    """'HH:MM:SS' UTC of mid-acquisition, the median over the SAFE names
    (stack_backend._acquisition_times), or on the NISAR route over the GUNW
    crops' identification (nisar_backend.acquisition_times). RAiDER
    interpolates the model to it between the two hours either side."""
    if str(getattr(cfg, "backend", "")) == "nisar":
        from . import nisar_backend as nb
        secs, where = sorted(nb.acquisition_times(cfg).values()), "NISAR crops (nisar_gunw/crops/)"
    else:
        from . import stack_backend as sb
        secs, where = sorted(sb._acquisition_times(cfg).values()), "SAFE names (slc/ or stack/SAFE_files.txt)"
    if not secs:
        raise RuntimeError(f"no {where} to read the acquisition time from")
    if secs[-1] - secs[0] > 3600:
        raise RuntimeError(f"acquisition times spread over {(secs[-1] - secs[0]) / 3600:.1f} h (mixed tracks?)")
    s = int(round(statistics.median(secs))) % 86400
    return f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"


def _one_cube(cfg, date: str, acq_time: str) -> Path | None:
    """One date's cube; the raw model files RAiDER downloaded (~7 MB a
    date) are deleted once the cube (~0.6 MB) exists."""
    from . import tropo_disp as td
    dt = datetime.strptime(f"{date} {acq_time}", "%Y%m%d %H:%M:%S")
    path = td.delay_cube_for_date(cfg, dt)
    if path is not None:
        shutil.rmtree(Path(path).parent / "weather_files", ignore_errors=True)
    return path


def fetch_cubes(cfg, dates, acq_time: str | None = None, workers: int | None = None) -> dict:
    """RAiDER cubes for `dates` (already cached ones are reused). Dates the
    model cannot serve (before its archive starts) are not asked for.
    Returns {date: Path or None}, None = no cube."""
    from . import progress
    from .tropo_disp import MODEL_START
    acq_time = acq_time or acquisition_time(cfg)
    model = model_name(cfg)
    start = MODEL_START.get(model)
    dates = sorted(set(dates))
    usable = [d for d in dates if start is None or datetime.strptime(d, "%Y%m%d") >= start]
    out = {d: None for d in dates}
    if len(usable) < len(dates):
        log.warning("%s has no data before %s: %d of %d dates cannot be corrected with it (%s ...)", model,
                    f"{start:%Y-%m-%d}", len(dates) - len(usable), len(dates), ", ".join(sorted(set(dates) - set(usable))[:5]))
    ok, t0 = 0, time.time()
    progress.emit(cfg.project_dir, "weather_raider", "start", done=0, total=len(usable),
                  msg=f"{model} weather delays for {len(usable)} dates ({acq_time} UTC), via RAiDER")
    n = max(1, int(workers or RAIDER_WORKERS))
    with ThreadPoolExecutor(max_workers=n) as pool:
        for d, p in zip(usable, pool.map(lambda x: _one_cube(cfg, x, acq_time), usable)):
            out[d] = p
            ok += p is not None
            progress.emit(cfg.project_dir, "weather_raider", "running", done=ok, total=len(usable),
                          msg=f"{len(usable) - ok} not yet" if ok < len(usable) else "")
    failed = [d for d in usable if out[d] is None]
    if failed:
        log.warning("RAiDER gave no %s delay for %d dates (%s ...); see tropo_corrections/%s/<date>/raider_run.log",
                    model, len(failed), ", ".join(failed[:5]), model)
        gaps = archive_gaps(cfg, failed)
        if gaps:
            log.warning("NOAA's %s archive has no file for %d of them (%s ...): retrying will not help",
                        model, len(gaps), ", ".join(gaps[:5]))
    progress.emit(cfg.project_dir, "weather_raider", "failed" if failed else "done", done=ok, total=len(usable),
                  msg=f"{ok} of {len(dates)} dates in {(time.time() - t0) / 60:.0f} min"
                      + (f"; {len(failed)} failed" if failed else ""))
    return out


def archive_gaps(cfg, dates) -> list[str]:
    """The dates among `dates` (ones RAiDER gave no cube for) whose RAiDER
    log says the model archive has no file for an hour it needed - a hole in
    the archive, not a download that may work next time. A date missing only
    one of its two hours is not failed: RAiDER uses the other hour alone
    (Site B: 20160901, 20161031, 20170204).

    Site B 2016-08-08 (pass 23:48 UTC): no HRRR native-level analysis for
    23Z, 00Z or 01Z on AWS or Google. The 22Z cycle's 1 h and 2 h forecasts,
    valid at those hours, were there, but they are not substituted: that
    evening the 21Z cycle's 1 h forecast against the 22Z analysis differed
    by 4.1 mm spatial std (and 23 mm in the mean) over a field of 2.0 mm,
    and on 2016-07-27 1 h forecasts against the analyses at the pass by
    10.1 mm over 5.5 mm - worse than no correction. (On five dates in
    2018-2025 the same comparison gave 16-79% of the field.)"""
    root = Path(cfg.tropo_dir) / model_name(cfg)
    gaps = []
    for d in dates:
        try:
            if ARCHIVE_GAP in (root / d / "raider_run.log").read_text(errors="replace"):
                gaps.append(d)
        except OSError:
            pass
    return gaps


UNAVAILABLE = "unavailable_dates.json"   # in tropo_corrections/<MODEL>/: the last check's dates without weather


def unavailable_path(cfg) -> Path:
    return Path(cfg.tropo_dir) / model_name(cfg) / UNAVAILABLE


def registered_unavailable(cfg) -> dict:
    """{date: reason} the last weather check found no weather data for
    (check_dates' record); {} off the RAiDER route or before any check."""
    if str(getattr(cfg, "mintpy_tropo_method", "")) != "raider":
        return {}
    try:
        return dict(json.loads(unavailable_path(cfg).read_text()).get("dates") or {})
    except (OSError, ValueError, AttributeError):
        return {}


def refetch_implausible(cfg, cubes: dict) -> dict:
    """Cubes in `cubes` whose delays are not physical (tropo_disp.cube_problem)
    are set aside as <cube>.implausible and fetched once more from a clean
    download; a date still not physical gets None in `cubes`. Returns
    {date: why} for those dates. RAiDER exits cleanly on such a cube (the
    first city site, 2018-08-04 HRRR: -5,787 m hydrostatic), so nothing else
    catches it before MintPy subtracts it."""
    from .tropo_disp import cube_problem
    still = {}
    for d, p in sorted(cubes.items()):
        why = cube_problem(Path(p)) if p and Path(p).exists() else ""
        if not why:
            continue
        log.warning("%s: the %s delay cube is not physical (%s) - fetching it again from a clean download",
                    d, model_name(cfg), why)
        Path(p).replace(Path(str(p) + ".implausible"))
        shutil.rmtree(Path(p).parent / "weather_files", ignore_errors=True)
        again = _one_cube(cfg, d, acquisition_time(cfg))
        why2 = cube_problem(Path(again)) if again else "no cube from the clean download"
        if why2:
            if again:
                Path(again).replace(Path(str(again) + ".implausible"))
            cubes[d] = None
            still[d] = why2
        else:
            cubes[d] = again
            log.info("%s: the clean download gave a physical cube", d)
    return still


def check_dates(cfg, dates, workers: int | None = None) -> dict:
    """Every date's weather before the network is inverted (26 Oct 2026,
    Kyle: dates the weather model cannot serve are registered and left out
    automatically, never an edit for the user): the cubes are fetched (or
    reused), a failed date is tried again from a clean download
    (tropo_disp.delay_cube_for_date), and the dates still without one are
    recorded with the reason in tropo_corrections/<MODEL>/unavailable_dates.json.
    The record is rewritten by every check, so a date that becomes available
    is taken back. Returns {date: reason}."""
    from .tropo_disp import MODEL_START
    model = model_name(cfg)
    dates = sorted(set(dates))
    cubes = fetch_cubes(cfg, dates, workers=workers)
    implausible = refetch_implausible(cfg, cubes)
    missing = [d for d in dates if not cubes.get(d)]
    gaps = set(archive_gaps(cfg, missing))
    start = MODEL_START.get(model)
    reasons = {}
    for d in missing:
        if start is not None and datetime.strptime(d, "%Y%m%d") < start:
            reasons[d] = f"before {model}'s first day ({start:%Y-%m-%d})"
        elif d in implausible:
            reasons[d] = f"{model}'s delays for this pass are not physical, also from a clean download ({implausible[d]})"
        elif d in gaps:
            reasons[d] = f"NOAA's {model} archive has no file for the hours around the pass"
        else:
            reasons[d] = (f"RAiDER gave no {model} delay, also from a clean download "
                          f"(tropo_corrections/{model}/{d}/raider_run.log)")
    path = unavailable_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"model": model, "checked": len(dates), "dates": reasons,
                                "written": time.strftime("%Y-%m-%dT%H:%M:%S")}, indent=1))
    if reasons:
        log.warning("%s weather unavailable for %d of %d dates; they are left out of the series: %s", model,
                    len(reasons), len(dates), "; ".join(f"{d} ({r})" for d, r in list(reasons.items())[:8]))
    else:
        log.info("%s weather available for all %d dates", model, len(dates))
    return reasons


def _read_dates(ts_file: Path) -> list[str]:
    import h5py
    with h5py.File(ts_file, "r") as f:
        return [d.decode() if isinstance(d, bytes) else str(d) for d in f["date"][:]]


def _geometry(geom_file: Path, attrs: dict):
    """lat, lon, height, incidence (degrees) on the series' grid."""
    import h5py
    with h5py.File(geom_file, "r") as f:
        height = f["height"][:].astype("float64")
        inc = f["incidenceAngle"][:].astype("float64")
        if "latitude" in f and "longitude" in f:
            lat, lon = f["latitude"][:].astype("float64"), f["longitude"][:].astype("float64")
        else:
            a = {k: (v.decode() if isinstance(v, bytes) else v) for k, v in f.attrs.items()} or attrs
            ny, nx = height.shape
            lat = float(a["Y_FIRST"]) + float(a["Y_STEP"]) * (np.arange(ny) + 0.5)
            lon = float(a["X_FIRST"]) + float(a["X_STEP"]) * (np.arange(nx) + 0.5)
            lon, lat = np.meshgrid(lon, lat)
    return lat, lon, height, inc


def write_delay_file(cfg, ts_file: Path, geom_file: Path, out_file: Path, cubes: dict) -> dict:
    """inputs/<MODEL>.h5 from the cubes (module docstring: layout and sign).
    Dates without a cube are left out of the file, so the coverage check
    names them. Returns the record also written to mintpy/weather_raider.json."""
    import h5py
    from .tropo_disp import _sample_cube
    dates = _read_dates(ts_file)
    with h5py.File(ts_file, "r") as f:
        attrs = {k: (v.decode() if isinstance(v, bytes) else v) for k, v in f.attrs.items()}
        shape = f["timeseries"].shape[1:]
    lat, lon, height, inc = _geometry(geom_file, attrs)
    if lat.shape != tuple(shape):
        raise RuntimeError(f"{geom_file.name} is {lat.shape}, the series {tuple(shape)}: not the same grid")
    valid = np.isfinite(inc) & (inc != 0) & np.isfinite(height) & np.isfinite(lat) & np.isfinite(lon)
    cos_inc = np.cos(np.radians(np.where(valid, inc, 0.0)))
    have = [d for d in dates if cubes.get(d)]
    layers, spatial = [], []
    for d in have:
        z = _sample_cube(Path(cubes[d]), np.where(valid, lon, 0.0), np.where(valid, lat, 0.0),
                         np.where(valid, height, 0.0))
        if z is None:
            continue
        slant = np.where(valid, -z.astype("float64") / cos_inc, 0.0).astype("float32")
        layers.append((d, slant))
        spatial.append(float(np.std(slant[valid])) * 1000 if valid.any() else float("nan"))
    keep = [d for d, _ in layers]
    out_file.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_file.with_name(out_file.name + ".part")
    with h5py.File(tmp, "w") as f:
        for k, v in attrs.items():
            if k not in ("REF_DATE", "REF_X", "REF_Y", "REF_LAT", "REF_LON"):
                f.attrs[k] = v
        f.attrs["FILE_TYPE"], f.attrs["UNIT"] = "timeseries", "m"
        f.attrs["WEATHER_SOURCE"] = f"RAiDER {model_name(cfg)} (pipeline weather_raider.py)"
        f["date"] = np.array([d.encode() for d in keep], dtype="S8")
        ts = f.create_dataset("timeseries", (len(keep),) + tuple(shape), dtype="float32")
        for i, (_d, slant) in enumerate(layers):
            ts[i] = slant
    tmp.replace(out_file)
    missing = [d for d in dates if d not in keep]
    rec = {"model": model_name(cfg), "source": "RAiDER", "file": str(out_file), "dates_total": len(dates),
           "dates_written": len(keep), "missing_dates": missing,
           "spatial_std_mm_median": round(float(np.nanmedian(spatial)), 2) if spatial else None,
           "spatial_std_mm_max": round(float(np.nanmax(spatial)), 2) if spatial else None,
           "written": time.strftime("%Y-%m-%dT%H:%M:%S")}
    (Path(cfg.mintpy_dir) / RECORD).write_text(json.dumps(rec, indent=2))
    log.info("%s delays written for %d of %d dates to %s (spatial std per date: median %s mm, max %s mm)",
             rec["model"], len(keep), len(dates), out_file, rec["spatial_std_mm_median"], rec["spatial_std_mm_max"])
    return rec


def build_delay_file(cfg, ts_file: Path, geom_file: Path, out_file: Path, workers: int | None = None) -> dict:
    """What the run calls before MintPy's troposphere step: keep an
    out_file that already holds every date of the series, otherwise fetch
    (or reuse cached) cubes and write it."""
    dates = _read_dates(ts_file)
    if out_file.exists():
        try:
            if set(dates) <= set(_read_dates(out_file)):
                log.info("%s already holds all %d dates; reusing it", out_file.name, len(dates))
                return {"model": model_name(cfg), "file": str(out_file), "reused": True, "dates_total": len(dates)}
        except (OSError, KeyError):
            pass
    cubes = fetch_cubes(cfg, dates, workers=workers)
    rec = write_delay_file(cfg, ts_file, geom_file, out_file, cubes)
    _refuse_without_reference_date(ts_file, rec)
    return rec


def _refuse_without_reference_date(ts_file: Path, rec: dict) -> None:
    """MintPy's diff.py references the delays to the series' reference date
    (REF_DATE, its first date after the inversion) and fails with a bare
    'ValueError: ... is not in list' when the delay file lacks it - even
    with era5_allow_partial. Say why, before MintPy does."""
    import h5py
    with h5py.File(ts_file, "r") as f:
        ref = f.attrs.get("REF_DATE")
    ref = ref.decode() if isinstance(ref, bytes) else (str(ref) if ref is not None else None)
    missing = list(rec.get("missing_dates") or [])
    if ref and ref in missing:
        from .tropo_disp import MODEL_START
        start = MODEL_START.get(str(rec.get("model")))
        why = (f"{rec['model']} has no data before {start:%Y-%m-%d}" if start and datetime.strptime(ref, "%Y%m%d") < start
               else f"RAiDER gave no {rec['model']} delay for it")
        raise RuntimeError(
            f"The time series' reference date {ref} has no {rec['model']} delay ({why}); {len(missing)} date(s) "
            f"lack one: {', '.join(missing[:8])}{' ...' if len(missing) > 8 else ''}. MintPy cannot apply weather "
            "delays without the reference date. Leave those dates out of the series, or use mintpy_tropo_method: "
            "pyaps (ERA5) for this project.")
