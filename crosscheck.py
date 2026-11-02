"""Independent cross-check of the official record against OPERA DISP-S1.

WHY THIS EXISTS. DISP-S1 is JPL's own processing of the same Sentinel-1
track: different software, coregistration, phase linking and unwrapping. It
is the cheapest independent product there is, and until 5 Oct 2026 it was
compared with the record only by hand (scripts/compare_products.py, once, at
Site A: per-cell r 0.52, 0.81 at 1 km, all three report areas reading
the same) - Site B, whose one detection is unverified, never was. Nothing
routine would confirm or contradict a finding at the next site.

WHAT IT DOES (`run(cfg)`, opt-in with `cross_check_disp: true`):
  1. finds the DISP-S1 frame on the record's own track and direction that
     covers the processing box (disp_backend.choose_frame); none - outside
     North America, before mid-2016, another track only - is recorded as
     "no independent satellite product was available", not as a failure;
  2. writes a sibling project `<project>_disp_check` whose config points the
     DISP chain at the official products (grid = the official
     geo_velocity.h5 coarsened to 30 m, its geometry, baselines, ERA5 files
     and reference point) and the official date window, and runs
     disp_chain.run on it - the record's own MintPy correction chain on
     DISP-S1 (an hour or so of download the first time; cached after);
  3. compares the two class-matched relative rate fields on the DISP grid,
     over the dates BOTH cover (the record is refitted over the common
     window when the DISP archive ends earlier - it lags by months - since
     comparing a decade with nine years reads seasonal farmland as
     disagreement): per cell, per coherence class, after 1 km averaging, per
     report area (agree within the combined 95% range of the two farmland
     nulls) and per reported feature, both products over the feature's
     outline (same sign beyond the DISP null; recheck_features redoes this
     for the features of the report being built);
  4. writes deliverables/technical/cross_check.json with one plain sentence.

It is a technical check, not a client number, and it does NOT gate the
verdict: `indicative_pass` (1 km r >= 0.6 and every testable area agrees)
is reported with its thresholds, which have not been calibrated on more
than one site (Site A). Treat a fail as a reason to look, not a result.

Comparison core (`grid`, `onto`, `block`, `robust_sd`, `agree`, `window`) is
shared with scripts/compare_products.py.
"""
from __future__ import annotations

import dataclasses
import json
import logging
import time
from datetime import datetime
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

# 1 km correlation and per-area agreement that would count as a pass. Not a
# gate: Site A (the only site compared) gave r 0.81 at 1 km.
PASS_R_1KM = 0.6
# The common window is used when either end of the two records differs by
# more than this many days (one 12-day revisit either side is not a
# difference worth a refit).
WINDOW_TOLERANCE_DAYS = 45


# ---------------------------------------------------------------- comparison core
def grid(attrs) -> dict:
    return {k: float(attrs[k]) for k in ("Y_FIRST", "X_FIRST", "Y_STEP", "X_STEP")} | \
           {"LENGTH": int(attrs["LENGTH"]), "WIDTH": int(attrs["WIDTH"])}


def onto(field: np.ndarray, src: dict, dst: dict) -> np.ndarray:
    """field on src's grid -> dst's grid: block nan-mean when nested, else nearest."""
    f = dst["Y_STEP"] / src["Y_STEP"]
    fx = dst["X_STEP"] / src["X_STEP"]
    nested = (abs(f - round(f)) < 1e-6 and abs(fx - round(fx)) < 1e-6 and round(f) == round(fx)
              and abs(src["Y_FIRST"] - dst["Y_FIRST"]) < 1e-9 and abs(src["X_FIRST"] - dst["X_FIRST"]) < 1e-9)
    if nested:
        k = int(round(f))
        a = field[: dst["LENGTH"] * k, : dst["WIDTH"] * k].astype("float64")
        a = a.reshape(dst["LENGTH"], k, dst["WIDTH"], k)
        n = np.isfinite(a).sum(axis=(1, 3))
        with np.errstate(invalid="ignore"):
            m = np.nansum(a, axis=(1, 3)) / np.where(n > 0, n, np.nan)
        m[n < max(1, (k * k) // 2)] = np.nan
        return m
    lat = dst["Y_FIRST"] + (np.arange(dst["LENGTH"]) + 0.5) * dst["Y_STEP"]
    lon = dst["X_FIRST"] + (np.arange(dst["WIDTH"]) + 0.5) * dst["X_STEP"]
    r = np.floor((lat - src["Y_FIRST"]) / src["Y_STEP"]).astype(int)
    c = np.floor((lon - src["X_FIRST"]) / src["X_STEP"]).astype(int)
    ok_r = (r >= 0) & (r < src["LENGTH"]); ok_c = (c >= 0) & (c < src["WIDTH"])
    out = np.full((dst["LENGTH"], dst["WIDTH"]), np.nan)
    out[np.ix_(ok_r, ok_c)] = field[np.ix_(r[ok_r], c[ok_c])]
    return out


def block(field: np.ndarray, k: int) -> np.ndarray:
    """Non-overlapping k x k nan-means, at least half the block valid."""
    ny, nx = (s // k for s in field.shape)
    a = field[: ny * k, : nx * k].reshape(ny, k, nx, k)
    n = np.isfinite(a).sum(axis=(1, 3))
    with np.errstate(invalid="ignore"):
        m = np.nansum(a, axis=(1, 3)) / np.where(n > 0, n, np.nan)
    m[n < (k * k) // 2] = np.nan
    return m


def robust_sd(x) -> float:
    x = x[np.isfinite(x)]
    return float(1.4826 * np.median(np.abs(x - np.median(x)))) if x.size else float("nan")


def agree(a: np.ndarray, b: np.ndarray) -> dict:
    s = np.isfinite(a) & np.isfinite(b)
    if s.sum() < 50:
        return {"n": int(s.sum())}
    d = a[s] - b[s]
    return {"n": int(s.sum()), "median_difference_mm_yr": round(float(np.median(d)), 2),
            "difference_robust_sd_mm_yr": round(robust_sd(d), 2),
            "per_product_equivalent_mm_yr": round(robust_sd(d) / np.sqrt(2), 2),
            "correlation": round(float(np.corrcoef(a[s], b[s])[0, 1]), 3)}


def window(field, g, lat, lon, km) -> tuple[float, int]:
    dlat = km / 110.95 / 2
    dlon = km / (111.32 * np.cos(np.radians(lat))) / 2
    r0 = int((lat + dlat - g["Y_FIRST"]) / g["Y_STEP"]); r1 = int((lat - dlat - g["Y_FIRST"]) / g["Y_STEP"])
    c0 = int((lon - dlon - g["X_FIRST"]) / g["X_STEP"]); c1 = int((lon + dlon - g["X_FIRST"]) / g["X_STEP"])
    w = field[max(r0, 0):r1 + 1, max(c0, 0):c1 + 1]
    return float(np.nanmean(w)), int(np.isfinite(w).sum())


# ---------------------------------------------------------------- products
def _read2d(path: Path, name: str | None = None):
    import h5py
    with h5py.File(path, "r") as f:
        if name is None or name not in f:
            name = next(k for k in f.keys() if f[k].ndim == 2)
        arr = f[name][:]
        attrs = {k: (v.decode() if isinstance(v, bytes) else v) for k, v in f.attrs.items()}
    return arr, attrs


def product(cfg, name: str) -> Path:
    """The geocoded product if there is one (deliverables._product's rule)."""
    geo = Path(cfg.mintpy_dir) / "geo" / f"geo_{name}"
    return geo if geo.exists() else Path(cfg.mintpy_dir) / name


def record_is_current(cfg, rec: dict | None, option: str) -> bool:
    """True when `rec` (a cross_check.json or injection_recovery.json record)
    belongs to this record: its option (`cross_check_disp` or
    `injection_test`) is on, and it was generated no earlier than the
    velocity product it checked. An older record checked a previous MintPy
    run, and one left behind after the option was turned off is not this
    run's; the report, the QA and the release gate all skip such records."""
    if not rec or not getattr(cfg, option, False):
        return False
    try:
        made = datetime.fromisoformat(str(rec.get("generated"))).timestamp()
    except (TypeError, ValueError):
        return False
    vel = product(cfg, "velocity.h5")
    return not vel.exists() or made >= vel.stat().st_mtime - 1


def timeseries_file(cfg) -> Path:
    """The time series the velocity was fitted to: deliverables' choice when
    it can make one, else the most corrected geo_timeseries*.h5.

    deliverables refuses (ValueError) a series that does not match the
    velocity product; that refusal is raised, not replaced by a pick by name -
    the refit and the injection test would otherwise run on dates and
    baselines the velocity was not fitted to."""
    try:
        from . import deliverables as dl
        f = Path(dl.timeseries_file(cfg))
        if f.exists():
            return f
    except (ImportError, FileNotFoundError) as exc:   # nothing to choose from there; fall back, logged
        log.info("deliverables.timeseries_file unavailable (%s); choosing by name", exc)
    cands = sorted((Path(cfg.mintpy_dir) / "geo").glob("geo_timeseries*.h5"),
                   key=lambda p: (p.stem.count("_"), p.stat().st_mtime))
    if not cands:
        cands = sorted(Path(cfg.mintpy_dir).glob("timeseries*.h5"),
                       key=lambda p: (p.stem.count("_"), p.stat().st_mtime))
    cands = [c for c in cands if "Residual" not in c.name]
    if not cands:
        raise FileNotFoundError(f"no time series under {cfg.mintpy_dir}")
    return cands[-1]


def read_dates(ts_file: Path) -> tuple[list[str], np.ndarray | None]:
    import h5py
    with h5py.File(ts_file, "r") as f:
        dates = [d.decode() if isinstance(d, bytes) else str(d) for d in f["date"][:]]
        bperp = f["bperp"][:].astype("float64") if "bperp" in f else None
    return dates, bperp


def load_fields(cfg, velocity: np.ndarray | None = None) -> dict:
    """
    The fields the report reads, loaded the way panel_history._Context loads
    them: velocity in mm/yr, temporal coherence, the reliable cells (with the
    second gate when min_unwrap_consistency > 0), the AOI far field and the
    wide far field - both without the report areas other than the whole
    study area (+ 500 m), which the report keeps out of its comparison
    ground (E28; F7(d), 7 Oct 2026: until then each area was compared with
    farmland that could hold it) - and the class-matched relative field
    (placement_field for placements over the wide far field, when
    significance has it). `velocity` (m/yr, on the product grid) replaces
    the product's own - the common-window refit uses it.
    """
    from . import significance as sg
    vel, attrs = _read2d(product(cfg, "velocity.h5"), "velocity")
    if velocity is not None:
        vel = velocity
    vel = vel.astype("float64") * 1000.0
    tc, _ = _read2d(product(cfg, "temporalCoherence.h5"), "temporalCoherence")
    good, _ = _read2d(product(cfg, "maskTempCoh.h5"), "mask")
    good = good.astype(bool) & np.isfinite(vel)
    thr = float(getattr(cfg, "min_unwrap_consistency", 0.0) or 0.0)
    gate = None
    if thr > 0:
        f2 = Path(cfg.mintpy_dir) / "geo" / "geo_temporalCoherence_mintpy.h5"
        if f2.exists():
            tcm, _ = _read2d(f2, "temporalCoherence")
            if tcm.shape == good.shape:
                good &= np.nan_to_num(tcm) >= thr
                gate = thr
    from . import landcover
    good, _ = landcover.drop_wooded(cfg, good, attrs)
    # the report areas other than the whole study area are not comparison
    # ground (E28, as the client report; F7(d)): both products' class
    # centres and nulls are drawn without them
    from .quality_assessment import report_farmland_masks
    far, far_wide, _, meta = report_farmland_masks(cfg, good, attrs)
    aoi = sg.aoi_mask(cfg.bbox, attrs, good.shape)
    r_cm, means = sg.class_matched_field(vel, tc, far)
    place = r_cm
    if hasattr(sg, "placement_field"):
        try:
            place = sg.placement_field(vel, tc, far, far_wide, aoi)
        except Exception as exc:                 # noqa: BLE001
            log.warning("placement_field failed (%s); using the class-matched field", exc)
    return {"attrs": attrs, "vel": vel, "tc": tc, "good": good, "far": far, "far_wide": far_wide,
            "aoi": aoi, "r_cm": r_cm, "placement": place, "class_means": means,
            "kcls": sg.coherence_class(tc), "second_gate": gate, "mine": meta.get("mine"),
            "comparison_ground": meta["comparison_ground"]}


def los_up_median(cfg, good: np.ndarray | None = None) -> float | None:
    """Median cos(incidence) of the product's own geometry (what a LOS rate is
    divided by for vertical); None when the product has no geometry."""
    for name in ("geometryRadar.h5", "geometryGeo.h5"):
        f = product(cfg, name)
        if not f.exists():
            f = Path(cfg.mintpy_dir) / "inputs" / name
        if not f.exists():
            continue
        try:
            inc, _ = _read2d(f, "incidenceAngle")
        except (KeyError, StopIteration, OSError):
            continue
        inc = inc.astype("float64")
        inc[inc == 0] = np.nan
        sel = np.isfinite(inc)
        if good is not None and good.shape == inc.shape:
            sel &= good
        if sel.any():
            return float(np.median(np.cos(np.radians(inc[sel]))))
    return None


def rate_over_window(ts_file: Path, d0: str, d1: str, shape=None) -> np.ndarray:
    """
    Per-cell linear rate (m/yr) over the epochs between d0 and d1 inclusive,
    streamed epoch by epoch (chunk-aligned blocks, significance.iter_epochs)
    into least-squares sums - the cube is never held, only five float64
    accumulators per cell (~140 MB for the 3.5 M-cell Site B grid).
    """
    import h5py
    from .significance import iter_epochs
    with h5py.File(ts_file, "r") as f:
        dates = [d.decode() if isinstance(d, bytes) else str(d) for d in f["date"][:]]
        use = {i for i, d in enumerate(dates) if d0 <= d <= d1}
        if len(use) < 3:
            raise ValueError(f"only {len(use)} epochs between {d0} and {d1}")
        base = datetime.strptime(dates[min(use)], "%Y%m%d")
        tt = {i: (datetime.strptime(dates[i], "%Y%m%d") - base).days / 365.25 for i in use}
        ds = f["timeseries"]
        sh = ds.shape[1:]
        n = np.zeros(sh); st = np.zeros(sh); stt = np.zeros(sh); sy = np.zeros(sh); sty = np.zeros(sh)
        for i, ep in iter_epochs(ds):
            if i not in use:
                continue
            y = ep.astype("float64")
            ok = np.isfinite(y)
            y = np.where(ok, y, 0.0)
            t = tt[i]
            n += ok; st += ok * t; stt += ok * t * t; sy += y; sty += y * t
    with np.errstate(invalid="ignore", divide="ignore"):
        den = n * stt - st * st
        rate = (n * sty - st * sy) / den
    rate[(n < 3) | ~np.isfinite(rate)] = np.nan
    return rate


# ---------------------------------------------------------------- areas
def area_masks(cfg, good: np.ndarray, attrs: dict) -> list[tuple[str, str, np.ndarray]]:
    """[(name, spec, mask)] for the report areas, as client_report draws them."""
    from . import significance as sg
    try:
        from .client_report import parse_areas
        specs = parse_areas(cfg)
    except Exception:                            # noqa: BLE001 - same rule, inline
        specs = [tuple(s.split(":", 1)) for s in (getattr(cfg, "report_areas", ()) or ())] or \
            [("Full Satellite Imaging", "full")]
    out = []
    aoi = sg.aoi_mask(cfg.bbox, attrs, good.shape)
    for name, spec in specs:
        name, spec = str(name).strip(), str(spec).strip()
        if spec.lower() == "full":
            m = good & aoi
        elif spec.lower().endswith(".geojson"):
            from .stack_backend import _polygon_mask
            if not Path(spec).exists():
                continue
            m = _polygon_mask(Path(spec), good.shape, attrs) & good
        else:
            try:
                s_, w, n, e = (float(v) for v in spec.split(","))
            except ValueError:
                continue
            s_, n, w, e = min(s_, n), max(s_, n), min(w, e), max(w, e)
            m = sg.aoi_mask((w, s_, e, n), attrs, good.shape) & good
        out.append((name, spec, m))
    return out


def _bbox_mask(bbox, attrs, shape) -> np.ndarray:
    from . import significance as sg
    w, s_, e, n = bbox
    return sg.aoi_mask((w, s_, e, n), attrs, shape)


# ---------------------------------------------------------------- the sibling project
def sibling_config(cfg, official: dict, window: tuple[str, str]):
    """The DISP-S1 config that runs the record's chain on the record's grid.

    Settings follow configs/config_site_a_disp.yaml, the one comparison
    made so far: conncomp 0 kept (as the record keeps low-quality pixels),
    OPERA's recommended mask off, ERA5 through the MintPy chain rather than
    RAiDER, geometry from the record."""
    P = Path(cfg.project_dir)
    sib = P.parent / f"{P.name}_disp_check"
    geo_dir = Path(cfg.mintpy_dir) / "geo"
    geom = geo_dir / "geo_geometryRadar.h5"
    baselines = Path(cfg.project_dir) / "stack" / "baselines"
    weather = Path(cfg.mintpy_dir)
    ref = official.get("ref_lalo") or getattr(cfg, "reference_lalo", "")
    d0, d1 = window
    changes = dict(
        backend="disp", project_dir=sib, time_series_dir="",
        start_date=f"{d0[:4]}-{d0[4:6]}-{d0[6:]}", end_date=f"{d1[:4]}-{d1[4:6]}-{d1[6:]}",
        disp_frame_id="", disp_record_chain=True,
        disp_target_grid_file=str(official["velocity_file"]), disp_target_grid_factor=2,
        disp_geometry_file=str(geom) if geom.exists() else "",
        disp_use_static_layers=not geom.exists(),
        disp_baseline_dir=str(baselines) if baselines.exists() else "",
        disp_weather_dir=str(weather) if (weather / "ERA5").exists() else "",
        disp_apply_corrections=True, disp_mask_connected_components=False,
        disp_flag_unwrap_jumps=True, disp_min_valid_fraction=0.5, disp_use_recommended_mask=False,
        disp_tropo_correction=False, disp_repair_boundary_steps=False, disp_repair_boundary_datum=False,
        reference_lalo=ref, disp_max_per_year=0,
    )
    for k in ("disp_client_report", "cross_check_disp", "injection_test"):
        changes[k] = False
    fields = {f.name for f in dataclasses.fields(cfg)} if dataclasses.is_dataclass(cfg) else set(vars(cfg))
    changes = {k: v for k, v in changes.items() if k in fields or not dataclasses.is_dataclass(cfg)}
    if dataclasses.is_dataclass(cfg):
        return dataclasses.replace(cfg, **changes)
    import copy
    c = copy.copy(cfg)
    for k, v in changes.items():
        setattr(c, k, v)
    return c


def _write_sibling_yaml(sib_cfg) -> Path | None:
    """The sibling's settings, beside its products, so the DISP run can be
    repeated or inspected by hand (python -m insar_pipeline.pipeline ...)."""
    import yaml
    try:
        raw = dataclasses.asdict(sib_cfg)
    except TypeError:
        return None
    clean = {}
    for k, v in raw.items():
        if isinstance(v, Path):
            v = str(v)
        elif isinstance(v, tuple):
            v = list(v)
        clean[k] = v
    out = Path(sib_cfg.project_dir) / "config_disp_check.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("# Written by crosscheck.py: OPERA DISP-S1 through the record's chain, on the record's grid.\n"
                   + yaml.safe_dump(clean, sort_keys=False))
    return out


def _search_disp(sib_cfg):
    """DISP-S1 granules for the sibling (patched in tests: no network)."""
    from . import disp_backend as db
    return db.search(sib_cfg)


def _run_disp_chain(sib_cfg):
    """disp_chain.run on the sibling (patched in tests: no download)."""
    from . import disp_chain
    return disp_chain.run(sib_cfg)


def _disk_ok(sib_cfg, n_granules: int) -> tuple[bool, str]:
    """Rough disk need: granule and corrections caches over the padded box at
    30 m (about 40 bytes a cell a granule, compressed) plus the chain's three
    time-series copies on the target grid. Checked against
    settings.disk_budget_gb when it exists (the WSL-aware budget)."""
    min_lon, min_lat, max_lon, max_lat = sib_cfg.bbox
    cells = ((max_lon - min_lon) * 111_320 * np.cos(np.radians(0.5 * (min_lat + max_lat))) / 30.0) * \
            ((max_lat - min_lat) * 110_950 / 30.0)
    need_gb = (n_granules * cells * 40 + 3 * n_granules * cells * 4) / 1e9
    try:
        from . import settings
        budget = settings.disk_budget_gb(Path(sib_cfg.project_dir).parent)
        free = float(budget.get("budget"))
    except Exception:                            # noqa: BLE001 - plain free space
        import shutil
        free = shutil.disk_usage(Path(sib_cfg.project_dir).parent).free / 1e9
    ok = free >= 1.5 * need_gb
    return ok, f"needs about {need_gb:.1f} GB (x1.5 margin), {free:.0f} GB available"


# ---------------------------------------------------------------- run
def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _write(path: Path, rec: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.part")
    tmp.write_text(json.dumps(rec, indent=1, default=str))
    tmp.replace(path)
    return path


def run(cfg, out_dir: str | Path | None = None, force: bool = False) -> Path:
    """
    Cross-check the official record against OPERA DISP-S1 and write
    deliverables/technical/cross_check.json (or `out_dir`/cross_check.json).

    "No DISP-S1 here" is a result, written and returned. A failure while
    running or comparing is written (`status: failed`, with the cause) and
    then raised, so an opted-in check never ends as a silent "done".
    """
    t0 = time.time()
    out = Path(out_dir) if out_dir else Path(cfg.project_dir) / "deliverables" / "technical"
    path = out / "cross_check.json"
    rec: dict = {"check": "OPERA DISP-S1 cross-check", "generated": _now(),
                 "project": str(cfg.project_dir), "official_backend": getattr(cfg, "backend", ""),
                 "gates_verdict": False,
                 "note": ("Technical check only, never a client number. Thresholds for indicative_pass are "
                          "not calibrated beyond one site; a fail is a reason to look, not a result.")}
    if str(getattr(cfg, "backend", "")).lower() == "disp":
        rec.update({"available": False, "status": "not applicable",
                    "reason": "this project is the DISP-S1 screening itself"})
        return _write(path, rec)
    try:
        vel_file = product(cfg, "velocity.h5")
        if not vel_file.exists():
            raise FileNotFoundError(f"official velocity not found ({vel_file})")
        ts_file = timeseries_file(cfg)
        dates, _ = read_dates(ts_file)
        _, attrs = _read2d(vel_file, "velocity")
        ref = (f"{float(attrs['REF_LAT']):.6f},{float(attrs['REF_LON']):.6f}"
               if "REF_LAT" in attrs and "REF_LON" in attrs else getattr(cfg, "reference_lalo", ""))
        official = {"velocity_file": vel_file, "timeseries_file": ts_file, "dates": [dates[0], dates[-1]],
                    "ref_lalo": ref}
        rec["official"] = {"velocity": str(vel_file), "timeseries": str(ts_file), "first": dates[0],
                           "last": dates[-1], "epochs": len(dates)}
        sib = sibling_config(cfg, official, (dates[0], dates[-1]))
        rec["sibling_project"] = str(sib.project_dir)

        from . import disp_backend as db
        try:
            results = _search_disp(sib)
        except db.NoFrameOnTrack as exc:         # no frame on this track/direction: a result
            # any other ValueError (mixed major versions, a granule off the
            # track) is a failure of the check, written and raised below
            results, why = [], str(exc)
        else:
            why = ("no OPERA DISP-S1 granules cover this area and period (published scope: North America "
                   "from mid-2016; a few frames elsewhere, e.g. F39360 over Sydney)")
        if not results:
            rec.update({"available": False, "status": "no independent product",
                        "reason": why,
                        "sentence": "No independent satellite product (OPERA DISP-S1) was available for this "
                                    "area and period, so the record was not cross-checked."})
            log.info("Cross-check: %s", why)
            return _write(path, rec)
        names = [p.properties["fileName"] for p in results]
        rec["disp"] = {"frame": db._frame_of(names[0]), "granules": len(names),
                       "first": db._dates(names[0])[1], "last": db._dates(names[-1])[1],
                       "versions": sorted({db._version(n) for n in names})}
        done = Path(sib.mintpy_dir) / "disp_chain.json"
        complete = False
        if done.exists() and not force:
            try:
                prev = json.loads(done.read_text())
                complete = (prev.get("status") == "complete"
                            and (prev.get("grid") or {}).get("source") == str(vel_file))
            except ValueError:
                complete = False
        if not complete:
            ok, why_disk = _disk_ok(sib, len(names))
            rec["disk"] = why_disk
            if not ok:
                rec.update({"available": False, "status": "not run",
                            "reason": f"not enough disk for the DISP-S1 download: {why_disk}"})
                log.warning("Cross-check not run: %s", rec["reason"])
                return _write(path, rec)
            _write_sibling_yaml(sib)
            log.info("Cross-check: running OPERA DISP-S1 F%s (%d granules) through the record's chain "
                     "into %s", rec["disp"]["frame"], len(names), sib.project_dir)
            _run_disp_chain(sib)
        rec["disp_chain"] = json.loads(done.read_text()) if done.exists() else None
        rec.update(compare(cfg, sib))
        rec.update({"available": True, "status": "complete", "wall_s": round(time.time() - t0)})
    except Exception as exc:
        rec.update({"status": "failed", "reason": f"{type(exc).__name__}: {exc}"})
        _write(path, rec)
        raise
    log.info("Cross-check: %s", rec.get("sentence"))
    return _write(path, rec)


def _assess(field, far_wide, region) -> dict:
    from . import significance as sg
    try:
        return sg.assess(field, far_wide, region) or {}
    except Exception as exc:                     # noqa: BLE001 - one area must not sink the check
        return {"testable": False, "not_testable_reason": f"assess failed: {exc}"}


def relative_fields(cfg, sib) -> dict:
    """Both products' class-matched relative rates on the DISP grid (the
    coarser one), inside the AOI, over the dates both cover - what every
    comparison below reads. The record whose dates reach beyond the common
    window is refitted over it."""
    O = load_fields(cfg)
    Dd = load_fields(sib)
    Go, Gd = grid(O["attrs"]), grid(Dd["attrs"])
    dts_o, _ = read_dates(timeseries_file(cfg))
    dts_d, _ = read_dates(timeseries_file(sib))
    d0, d1 = max(dts_o[0], dts_d[0]), min(dts_o[-1], dts_d[-1])
    out: dict = {"common_window": [d0, d1]}

    def _days(a, b):
        return abs((datetime.strptime(a, "%Y%m%d") - datetime.strptime(b, "%Y%m%d")).days)
    for name, dts, F, c in (("official", dts_o, O, cfg), ("disp", dts_d, Dd, sib)):
        if _days(dts[0], d0) > WINDOW_TOLERANCE_DAYS or _days(dts[-1], d1) > WINDOW_TOLERANCE_DAYS:
            log.info("Cross-check: refitting the %s rate over the common window %s-%s (its record is "
                     "%s-%s)", name, d0, d1, dts[0], dts[-1])
            v = rate_over_window(timeseries_file(c), d0, d1)
            F.update(load_fields(c, velocity=v))
            out[f"{name}_refitted_to_common_window"] = True
        else:
            out[f"{name}_refitted_to_common_window"] = False

    o_rel = np.where(O["good"], O["r_cm"], np.nan)
    o_rel = o_rel if Go == Gd else onto(o_rel, Go, Gd)
    d_rel = np.where(Dd["good"], Dd["r_cm"], np.nan)
    from . import significance as sg
    aoi_d = sg.aoi_mask(cfg.bbox, Dd["attrs"], d_rel.shape)
    o_rel = np.where(aoi_d, o_rel, np.nan)
    d_rel = np.where(aoi_d, d_rel, np.nan)
    up_o = los_up_median(cfg, O["good"]) or None
    up_d = los_up_median(sib, Dd["good"]) or up_o
    return {"record": out, "O": O, "Dd": Dd, "Go": Go, "Gd": Gd, "o_rel": o_rel, "d_rel": d_rel,
            "up_o": up_o, "up_d": up_d}


def compare(cfg, sib) -> dict:
    """The two products' class-matched relative rates on the DISP grid, over
    the common window, cell by cell, by area and by feature."""
    R = relative_fields(cfg, sib)
    out = R["record"]
    O, Dd, Go, Gd, o_rel, d_rel = R["O"], R["Dd"], R["Go"], R["Gd"], R["o_rel"], R["d_rel"]
    from . import significance as sg
    px_km = abs(Gd["Y_STEP"]) * 110.95
    k1 = max(1, int(round(1.0 / px_km)))
    cells = agree(o_rel, d_rel)
    km1 = agree(block(o_rel, k1), block(d_rel, k1))
    by_class = []
    for k, lab in enumerate(sg.class_labels()):
        sel = Dd["kcls"] == k
        r = agree(np.where(sel, o_rel, np.nan), np.where(sel, d_rel, np.nan))
        if r.get("n", 0) >= 200:
            by_class.append({"tc": lab, **r})
    out.update({"grid": Gd, "all_cells": cells, "one_km_means": km1, "by_disp_coherence_class": by_class})
    out["comparison_ground"] = {
        "official": O.get("comparison_ground"), "disp": Dd.get("comparison_ground"),
        "note": ("both products are relative to farmland outside the mapped mines plus buffer and outside the report "
                 "areas other than the whole study area plus 500 m, as the client report's comparison ground "
                 "(not cut only when they cover nearly all the farmland: see 'reason')")}

    up_o, up_d = R["up_o"], R["up_d"]
    out["los_up"] = {"official": up_o, "disp": up_d}

    # the official placement field on the DISP grid, for the official null
    o_place = O["placement"] if Go == Gd else onto(np.where(O["good"], O["placement"], np.nan), Go, Gd)
    o_far_wide = O["far_wide"] if Go == Gd else (onto(O["far_wide"].astype(float), Go, Gd) >= 0.5)
    areas = []
    for name, spec, m in area_masks(sib, Dd["good"], Dd["attrs"]):
        m = m & np.isfinite(o_rel) & np.isfinite(d_rel)
        if m.sum() < 5:
            areas.append({"name": name, "spec": spec, "cells": int(m.sum()), "note": "too few cells measured by both"})
            continue
        so = _assess(o_place, o_far_wide & np.isfinite(o_place), m)
        sd = _assess(Dd["placement"], Dd["far_wide"], m)
        vo, vd = float(np.nanmean(o_rel[m])), float(np.nanmean(d_rel[m]))
        spo = so.get("spread_for_ranges_mm_yr", so.get("farmland_spread_1sigma_mm_yr"))
        spd = sd.get("spread_for_ranges_mm_yr", sd.get("farmland_spread_1sigma_mm_yr"))
        a = {"name": name, "spec": spec, "cells": int(m.sum()),
             "official_relative_los_mm_yr": round(vo, 2), "disp_relative_los_mm_yr": round(vd, 2),
             "difference_mm_yr": round(vo - vd, 2),
             "official_reading": so.get("verdict"), "disp_reading": sd.get("verdict"),
             "official_distinguishable": so.get("distinguishable_from_farmland"),
             "disp_distinguishable": sd.get("distinguishable_from_farmland")}
        if up_o:
            a["official_relative_vertical_mm_yr"] = round(vo / up_o, 2)
            a["disp_relative_vertical_mm_yr"] = round(vd / (up_d or up_o), 2)
        if spo and spd and so.get("testable", True) and sd.get("testable", True):
            comb = 1.96 * float(np.hypot(spo, spd))
            a.update({"combined_95_mm_yr": round(comb, 2), "agree": bool(abs(vo - vd) <= comb)})
        else:
            a.update({"agree": None, "note": so.get("not_testable_reason") or sd.get("not_testable_reason")
                      or "no farmland null for an area this size"})
        areas.append(a)
    out["areas"] = areas

    # the features of the report on disk when the check ran: in the pipeline
    # the cross-check runs before the client report, so these are the
    # PREVIOUS report's (none on a first build); the client report redoes
    # them for its own features with recheck_features
    feats = []
    rep_f = Path(cfg.project_dir) / "deliverables" / "client_report.json"
    if rep_f.exists():
        try:
            feats = json.loads(rep_f.read_text()).get("features") or []
        except ValueError:
            feats = []
        out["features_from"] = {"report": str(rep_f), "modified": datetime.fromtimestamp(
            rep_f.stat().st_mtime).isoformat(timespec="seconds")}
    out["features"] = feature_rows(feats, o_rel, d_rel, Dd, up_o, up_d)

    testable = [a for a in areas if a.get("agree") is not None]
    all_agree = all(a["agree"] for a in testable) if testable else None
    r1 = km1.get("correlation")
    out["indicative_pass"] = (bool(r1 is not None and r1 >= PASS_R_1KM and all_agree is not False)
                              if r1 is not None else None)
    out["thresholds"] = {"one_km_correlation_at_least": PASS_R_1KM,
                         "areas": "agree within 1.96 x the combined farmland spread",
                         "calibrated_on": ["site_a (r 0.81 at 1 km, 28 Sep 2026)"]}
    out["sentence"] = sentence(out)
    return out


def _feature_mask(f: dict, attrs: dict, shape) -> tuple[np.ndarray | None, str, list | None]:
    """(cells, which, box) of one reported feature on a grid: its outline
    (client_report.json 'ring') when it has one, else its box, else 500 m
    around its centre."""
    bb = f.get("bbox")
    if not bb and f.get("lat") is not None:
        dl_ = 0.25 / 110.95; dn = 0.25 / (111.32 * np.cos(np.radians(f["lat"])))
        bb = [f["lon"] - dn, f["lat"] - dl_, f["lon"] + dn, f["lat"] + dl_]
    if f.get("ring"):
        from .results import ring_mask
        m = ring_mask(f["ring"], attrs, shape)
        if m.any():
            return m, "outline", bb
    if not bb:
        return None, "", None
    return _bbox_mask(bb, attrs, shape), "box", bb


def feature_rows(feats: list, o_rel: np.ndarray, d_rel: np.ndarray, Dd: dict, up_o, up_d) -> list[dict]:
    """Each reported feature measured by BOTH products over the same cells -
    its outline, else its box - on the DISP grid. The DISP mean over the box
    used to be set beside the report's value over the outline, so identical
    products 'disagreed' by about 40% (Site B, -37.3 against -22.9 mm/yr)
    and a weaker real feature could be diluted below the DISP null. The
    report's own value is kept as report_vertical_relative_mm_yr; `same_sign`
    is against it (the feature as reported), else against the official
    value over the same cells."""
    rows = []
    for i, f in enumerate(feats or []):
        if not isinstance(f, dict):
            continue
        m, which, bb = _feature_mask(f, Dd["attrs"], d_rel.shape)
        if m is None:
            continue
        m = m & np.isfinite(d_rel) & np.isfinite(o_rel)
        rv = f.get("vertical_relative_mm_yr", f.get("mean_rel_vertical_mm_yr"))
        row = {"index": i, "lat": f.get("lat"), "lon": f.get("lon"), "bbox": bb, "cells_from": which,
               "report_vertical_relative_mm_yr": rv, "cells": int(m.sum())}
        if m.sum() >= 5:
            s = _assess(Dd["placement"], Dd["far_wide"], m)
            vo, vd = float(np.nanmean(o_rel[m])), float(np.nanmean(d_rel[m]))
            ref = rv if rv else vo
            row.update({"official_relative_los_mm_yr": round(vo, 2),
                        "official_relative_vertical_mm_yr": round(vo / up_o, 2) if up_o else None,
                        "disp_relative_los_mm_yr": round(vd, 2),
                        "disp_relative_vertical_mm_yr": round(vd / up_d, 2) if up_d else None,
                        "disp_distinguishable": s.get("distinguishable_from_farmland"),
                        "same_sign": (bool(np.sign(vd) == np.sign(ref)) if ref else None)})
            row["confirmed"] = bool(row["same_sign"] and row["disp_distinguishable"])
        else:
            row.update({"confirmed": None, "note": "too few cells measured by both products here"})
        rows.append(row)
    return rows


def recheck_features(cfg, rec: dict, feats: list) -> dict:
    """`rec` (a cross_check.json record) with its feature rows and sentence
    redone for `feats` - the features of the report being built.

    The cross-check runs before the quality assessment and the client report
    (the QA copies its record), so the features it could read were the
    previous report's, or none on a first build, and the page printed
    'k of 3 reported features' beside a list of 2. client_report calls this
    with the features it has just found. A record that is not a completed
    comparison is returned unchanged."""
    if not isinstance(rec, dict) or rec.get("status") != "complete" or not rec.get("available") \
            or len(rec.get("common_window") or ()) != 2:
        return rec
    d0, d1 = (str(d) for d in rec["common_window"])
    sib = sibling_config(cfg, {"velocity_file": product(cfg, "velocity.h5")}, (d0, d1))
    R = relative_fields(cfg, sib)
    out = dict(rec)
    out["features"] = feature_rows(feats, R["o_rel"], R["d_rel"], R["Dd"], R["up_o"], R["up_d"])
    out["features_from"] = {"report": "the report being built", "modified": _now()}
    out["sentence"] = sentence(out)
    return out


def sentence(c: dict) -> str:
    """One plain sentence for the technical page."""
    r1 = (c.get("one_km_means") or {}).get("correlation")
    areas = [a for a in c.get("areas") or [] if a.get("agree") is not None]
    agree_n = sum(1 for a in areas if a["agree"])
    feats = c.get("features") or []
    conf = sum(1 for f in feats if f.get("confirmed"))
    d0, d1 = c.get("common_window") or ("", "")
    s = (f"An independent satellite product (OPERA DISP-S1, processed by NASA JPL) over the same dates "
         f"({d0[:4]}-{d1[:4]}) ")
    if r1 is not None and r1 >= PASS_R_1KM:
        s += f"matches the record's pattern of movement with a correlation of {r1:.2f} at 1 km"
    elif r1 is not None and r1 > 0:
        s += (f"agrees only weakly with the record's pattern of movement (correlation {r1:.2f} at 1 km; "
              f"{PASS_R_1KM} or more is expected)")
    elif r1 is not None:                         # zero or negative: no agreement at all
        s += (f"does not follow the record's pattern of movement (correlation {r1:.2f} at 1 km; "
              f"{PASS_R_1KM} or more is expected)")
    else:
        s += "could not be compared cell by cell"
    if areas:
        s += f"; {agree_n} of {len(areas)} report areas read the same within the combined uncertainty"
    if feats:
        s += f"; {conf} of {len(feats)} reported features also stand out in it"
    return s + "."


def main(argv=None) -> int:
    import argparse
    from .config import load_config
    ap = argparse.ArgumentParser(description="Cross-check a finished project against OPERA DISP-S1.")
    ap.add_argument("config")
    ap.add_argument("--out-dir", default=None, help="write cross_check.json here instead of deliverables/technical")
    ap.add_argument("--force", action="store_true", help="re-run the DISP chain even if it is complete")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = run(load_config(a.config), out_dir=a.out_dir, force=a.force)
    print(p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
