"""Solid-earth tide and ionosphere: switched on automatically for large areas.

Both effects vary over hundreds of kilometres. Across a small area they are
almost a uniform shift plus a tilt, which the reference point and the ramp
removal take out. Measured on the Site A record (30 x 24 km, 28 Sep 2026):
per-image differences across the area of 0.46 mm (tide) and 2.8 mm
(ionosphere), leaving at most 0.003 mm/yr in any rate. At ~100 km and beyond
they stop being planar, so there they are applied:

  solid_earth_tides      "auto" | "yes" | "no"   MintPy's own step (pysolid)
  ionosphere_correction  "auto" | "gim" | "no"   IGS global ionosphere maps
                                                 (iono_tec.py), the same kind of
                                                 source as OPERA's DISP-S1 layer
  large_area_km          the "auto" threshold on the processed box's longer side

The OPERA DISP-S1 route always subtracts the product's own tide and ionosphere
layers (they cost nothing), so these settings affect the stack routes.

Why IGS and not JPL maps: MintPy switches to the IGS long file names at GPS
week 2238 (27 Nov 2022). The IGS combined product switched then; JPL kept the
short names for months, so its files 404 in between (seen 28 Sep 2026).
"""
from __future__ import annotations

import json
import logging
import math
import os
import subprocess
from pathlib import Path

log = logging.getLogger(__name__)

GIM_CENTRE = "igs"


def extent_km(cfg) -> tuple[float, float]:
    """(east-west, north-south) size of the processed box in km."""
    w, s, e, n = cfg.bbox
    lat = math.radians(0.5 * (s + n))
    return (abs(e - w) * 111.32 * math.cos(lat), abs(n - s) * 110.95)


def decide(cfg) -> dict:
    """Which of the two corrections this run applies, and why."""
    ew, ns = extent_km(cfg)
    size = max(ew, ns)
    big = size >= float(getattr(cfg, "large_area_km", 100.0))
    tide_opt = str(getattr(cfg, "solid_earth_tides", "auto")).lower()
    iono_opt = str(getattr(cfg, "ionosphere_correction", "auto")).lower()
    tide = big if tide_opt == "auto" else tide_opt == "yes"
    iono = ("gim" if big else None) if iono_opt == "auto" else ("gim" if iono_opt == "gim" else None)
    if getattr(cfg, "backend", "") == "disp":
        # What is ASKED for. Whether every epoch's layers were actually read is
        # measured at stitching (disp_backend: stitch.corrections) and written
        # over these keys by disp_chain; a partial set stops the run.
        if not bool(getattr(cfg, "disp_apply_corrections", True)):
            return {"extent_km": [round(ew, 1), round(ns, 1)], "large_area": big, "solid_earth_tides": False,
                    "ionosphere": None, "why": "OPERA DISP-S1: the product's tide and ionosphere layers are NOT "
                                               "subtracted (disp_apply_corrections: false)"}
        why = ("OPERA DISP-S1: the product's own tide and ionosphere layers are to be subtracted from every "
               "epoch; what was actually applied is measured at stitching (disp_chain.json stitch.corrections)")
        return {"extent_km": [round(ew, 1), round(ns, 1)], "large_area": big, "solid_earth_tides": True,
                "ionosphere": "product layer", "requested_only": True, "why": why}
    thr = getattr(cfg, "large_area_km", 100.0)
    why = (f"processed box {ew:.0f} x {ns:.0f} km; {'at or above' if big else 'below'} the {thr:g} km "
           f"threshold, so 'auto' {'applies' if big else 'skips'} them"
           + ("" if big else " (at this size both are planar and cancel: measured < 0.003 mm/yr)"))
    if tide_opt != "auto" or iono_opt != "auto":
        why += f"; set explicitly: tide {tide_opt}, ionosphere {iono_opt}"
    return {"extent_km": [round(ew, 1), round(ns, 1)], "large_area": big, "solid_earth_tides": bool(tide),
            "ionosphere": iono, "why": why}


# Weather-model names as MintPy puts them in a series' name (timeseries_ERA5,
# timeseries_HRRR ...): the troposphere token of a correction chain.
WEATHER_TOKENS = ("ERA5T", "ERA5", "ERAI", "MERRA2", "MERRA", "NARR", "HRRRAK", "HRRR", "GMAO")


def mintpy_tropo_method(cfg) -> str:
    """MintPy's troposphericDelay.method for the config. 'raider' is the
    pipeline's own (weather_raider.py writes inputs/<MODEL>.h5, which
    MintPy's pyaps branch then applies), so MintPy is told 'pyaps': a method
    word MintPy does not know is skipped without a word, which would have
    run the series uncorrected."""
    m = str(getattr(cfg, "mintpy_tropo_method", "pyaps") or "no")
    return "pyaps" if m == "raider" else m


def weather_model(cfg) -> str:
    """MintPy's troposphericDelay.weatherModel: ERA5 for pyaps, the
    config's weather_model (HRRR ...) for raider."""
    if str(getattr(cfg, "mintpy_tropo_method", "")) == "raider":
        return str(getattr(cfg, "weather_model", "") or "HRRR").upper()
    return "ERA5"


def mintpy_tropo_lines(cfg, width: int = 34) -> list[str]:
    """The two template lines, `key = value` with the key padded to width."""
    return [f"{'mintpy.troposphericDelay.method':<{width}} = {mintpy_tropo_method(cfg)}",
            f"{'mintpy.troposphericDelay.weatherModel':<{width}} = {weather_model(cfg)}"]


def mintpy_lines(cfg) -> str:
    """smallbaselineApp settings for the tide (the ionosphere is applied by
    apply_gim_ionosphere, between MintPy phases)."""
    d = decide(cfg)
    return f"mintpy.solidEarthTides            = {'yes' if d['solid_earth_tides'] else 'no'}\n"


def apply_gim_ionosphere(cfg, ts_file: Path, geom_file: Path, tec_dir: Path | None = None) -> dict:
    """Subtract the IGS global-ionosphere-map delay from ts_file, in place,
    once. Idempotent: the file carries IONO_GIM_APPLIED afterwards, and a
    re-run of MintPy's inversion writes a fresh file without it."""
    import h5py
    ts_file, geom_file = Path(ts_file), Path(geom_file)
    with h5py.File(ts_file, "r") as f:
        if f.attrs.get("IONO_GIM_APPLIED"):
            return {"applied": True, "skipped": "already applied", "file": str(ts_file)}
    tec_dir = Path(tec_dir or (cfg.mintpy_dir / "IONEX"))
    tec_dir.mkdir(parents=True, exist_ok=True)
    iono = ts_file.parent / "inputs" / f"ionosphere_gim_{GIM_CENTRE}.h5"
    iono.parent.mkdir(exist_ok=True)
    env = {**os.environ}
    r = subprocess.run(["iono_tec.py", str(ts_file), "-g", str(geom_file), "-s", GIM_CENTRE,
                        "--tec-dir", str(tec_dir), "--iono-file", str(iono)],
                       capture_output=True, text=True, cwd=str(ts_file.parent), env=env)
    (ts_file.parent / "iono_gim.log").write_text(r.stdout + r.stderr)
    if r.returncode != 0 or not iono.exists():
        raise RuntimeError(f"iono_tec.py failed ({r.returncode}); see {ts_file.parent / 'iono_gim.log'}")
    tmp = ts_file.with_name(ts_file.stem + "_gimtmp.h5")
    r = subprocess.run(["diff.py", str(ts_file), str(iono), "-o", str(tmp), "--force"],
                       capture_output=True, text=True, cwd=str(ts_file.parent))
    if r.returncode != 0 or not tmp.exists():
        raise RuntimeError(f"diff.py failed applying the ionosphere: {(r.stderr or r.stdout)[-400:]}")
    with h5py.File(tmp, "a") as f:
        f.attrs["IONO_GIM_APPLIED"] = f"{GIM_CENTRE} global ionosphere maps (iono_tec.py)"
    os.replace(tmp, ts_file)
    rec = {"applied": True, "centre": GIM_CENTRE, "iono_file": str(iono), "file": str(ts_file)}
    log.info("Ionosphere (IGS global maps) subtracted from %s", ts_file.name)
    return rec


def record(cfg, extra: dict | None = None) -> Path:
    """mintpy/corrections.json: what was decided and done. Starts the file
    afresh for the run (nothing from an earlier run's record survives)."""
    out = Path(cfg.mintpy_dir) / "corrections.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    base = decide(cfg)
    if extra and "disp_corrections" in extra:
        base.pop("requested_only", None)          # the stitch measured what was applied
    out.write_text(json.dumps({**base, **(extra or {})}, indent=1))
    return out


def amend(cfg, extra: dict) -> Path:
    """Add outcomes to this run's corrections.json (written by record())
    without dropping what is already there."""
    out = Path(cfg.mintpy_dir) / "corrections.json"
    try:
        cur = json.loads(out.read_text()) if out.exists() else decide(cfg)
    except ValueError:
        cur = decide(cfg)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({**cur, **extra}, indent=1, default=str))
    return out


class WeatherCoverageError(RuntimeError):
    """Weather delays missing for some dates; `cause` and `action` are the two
    halves advice.explain shows (the message is both)."""

    def __init__(self, cause: str, action: str):
        super().__init__(f"{cause} {action}")
        self.cause, self.action = cause, action


def _dates(h5file: Path) -> list[str]:
    import h5py
    with h5py.File(h5file, "r") as f:
        return [d.decode() if isinstance(d, bytes) else str(d) for d in f["date"][:]]


def era5_coverage(ts_file: Path, weather_file: Path) -> dict:
    """Which dates of the time series the weather-model delay file covers."""
    ts = _dates(ts_file)
    have = set(_dates(weather_file))
    missing = [d for d in ts if d not in have]
    return {"dates_total": len(ts), "dates_corrected": len(ts) - len(missing), "missing_dates": missing}


def verify_era5_coverage(cfg, ts_file: Path, weather_file: Path, model: str = "ERA5") -> dict:
    """
    After MintPy's troposphere step: every date of the time series must have a
    weather-model delay, or the run stops naming the dates.

    MintPy 1.6.4 tries each download three times, then "stop trying and
    continue" (tropo_pyaps3.dload_grib_files) and builds inputs/ERA5.h5 from
    the files that exist; diff.py --force then corrects the shared dates only.
    The missing dates kept their troposphere, and the product was still named
    timeseries_ERA5* and described to the client as "weather delays (ERA5)
    corrected". Most likely trigger: an end date within ~5 days of today (ERA5
    lags real time) or a Copernicus outage mid-queue. Measured cost on
    Site B when it does happen: after the linear deramp the uncorrected
    part is the non-planar ERA5 field, median 0.23 mm std per epoch (max
    1.2 mm) - small, but the claim must be true.

    With era5_allow_partial: true the run goes on and the uncorrected dates are
    recorded (here, in corrections.json, and in run_products.json) so the
    report can say "for N of M dates". Returns the coverage record.
    """
    ts_file, weather_file = Path(ts_file), Path(weather_file)
    if not weather_file.exists():
        raise RuntimeError(
            f"The troposphere method is pyaps but {weather_file} does not exist, so which dates were corrected "
            "cannot be checked - see mintpy_error.log (correct_troposphere).")
    rec = {"model": model, "file": str(weather_file), **era5_coverage(ts_file, weather_file)}
    allowed = bool(cfg.era5_allow_partial)
    rec["allowed_partial"] = allowed
    amend(cfg, {"troposphere": rec})
    miss = rec["missing_dates"]
    if not miss:
        log.info("%s delay found for all %d dates", model, rec["dates_total"])
        return rec
    shown = ", ".join(miss[:12]) + (f" and {len(miss) - 12} more" if len(miss) > 12 else "")
    if not allowed:
        cause = (f"{model} weather delays are missing for {len(miss)} of {rec['dates_total']} dates ({shown}), so "
                 f"those dates were NOT corrected although the product is named {model}.")
        partial = "- or set era5_allow_partial: true to deliver with those dates uncorrected and say so."
        if model.upper() in ("ERA5", "ERA5T"):
            raise WeatherCoverageError(cause, (
                f"{model} is published about 5 days after real time and the Copernicus queue can fail part-way. "
                "Wait and re-run from MintPy (the weather files already fetched are kept), or set an earlier "
                f"end_date {partial}"))
        # RAiDER (HRRR ...): weather_raider.py fetched them and logged each date
        from .weather_raider import ARCHIVE_GAP, archive_gaps
        gaps = archive_gaps(cfg, miss)
        others = [d for d in miss if d not in gaps]
        logs = f"tropo_corrections/{model}/<date>/raider_run.log"
        if gaps:
            cause += (f" NOAA's {model} archive has no file for {', '.join(gaps[:12])} (RAiDER's log: "
                      f"'{ARCHIVE_GAP}'), so waiting will not bring {'it' if len(gaps) == 1 else 'them'}.")
            action = ("Resume: the weather check before the inversion leaves such dates out of the series by "
                      "itself (time_series.exclude_weather_unavailable; no config edit), and the other dates' "
                      "weather is kept")
            if others:
                action += (f"; RAiDER failed for {', '.join(others[:12])} for another reason ({logs}), "
                           "and the same check retries those from a clean download")
            raise WeatherCoverageError(cause, f"{action} {partial}")
        raise WeatherCoverageError(cause, (
            f"RAiDER could not build them ({logs}). Resume: the weather check before the inversion retries them "
            "from a clean download and leaves out any still without data, by itself (no config edit) "
            f"{partial}"))
    log.warning("%s delays missing for %d of %d dates (%s): delivered uncorrected for those dates "
                "(era5_allow_partial)", model, len(miss), rec["dates_total"], shown)
    return rec
