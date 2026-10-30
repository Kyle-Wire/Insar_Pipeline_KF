"""OPERA DISP-S1 through the record's correction chain.

WHY THIS EXISTS. DISP-S1 is JPL's operational phase-linked product: dolphin
phase linking over the full-resolution CSLC stack, snaphu on a nearest-3
interferogram network, an L1 time-series inversion. That is an independent
processing of the same Sentinel-1 track as the stack record - different
software, different coregistration, different unwrapper settings - and so the
cheapest available check on anything the record finds. But a check is only
fair if both products receive the same corrections and are read the same way.
disp_backend.build_stack fits a rate directly to the stitched product with its
own model; this module instead puts DISP-S1 through the steps the record's
MintPy run applied, in MintPy's order, on the record's lat/lon grid:

  stitch     disp_backend.assemble_stack: solid-earth tide and ionosphere
             layers subtracted (OPERA supplies but does not apply them),
             snaphu connected component 0 dropped per epoch, reference-epoch
             carry, whole-cycle steps into each ANCHOR epoch (the value the
             carry copies) counted per pixel, whole-cycle spikes that revert
             rejected. Network misclosure (the L1 inversion residual) is
             counted per pixel as the DISP-S1 counterpart of the record's
             triplet-closure diagnostic. The run STOPS (disp_backend.
             check_stitch) on a partly corrected stack, a lost anchor granule
             or a coverage ratchet - each used to be a log line before a full
             report - and records every measured problem in disp_chain.json.
  resample   nearest neighbour from DISP-S1's 30 m UTM grid onto the record's
             lat/lon grid coarsened to 30 m (MintPy geocodes with nearest
             too), so masks, far field, panels and report areas are the same
             cells in both products.
  gap-fill   epochs a pixel lost to the conncomp or spike masks are linearly
             interpolated in time for pixels valid in at least
             disp_min_valid_fraction of epochs (MintPy's steps and its rate
             fit do not take gaps); the effect on the rate is measured and
             recorded, and pixels below the fraction are left out - as are
             pixels whose first/last valid epoch leaves more than 10% of the
             record to be held flat (END_HOLD_MAX_FRACTION).
  reference  the record's reference pixel (reference_lalo).
  troposphere  tropo_pyaps3.py, ERA5, the record's grib files.
  deramp     remove_ramp.py linear, fitted outside the mine panels + 2 km.
  DEM error  dem_error.py, polynomial order 2 (MintPy default, as the record),
             with baselines from the record's orbit stack (DISP-S1 v1.0 ships
             an all-zero perpendicular_baseline layer).
  reference date   minimum residual RMS date (timeseries_rms.py), as MintPy.
  rate       timeseries2velocity.py, linear, residue-based uncertainty.
  coherence  DISP-S1's temporal_coherence (stack mean), graded at the same
             min_temporal_coherence as the record.

The products land in mintpy/geo/ under the record's file names, so
deliverables, QA, significance, panel history, figures, the GNSS tie and the
client report run on them unchanged. The per-cell DISP-S1 quality layers
(valid-epoch share, recommended mask, network misclosure, anchor suspects,
anchor conncomp-0 share) go to deliverables/technical/disp_quality/*.tif;
corrections.json records the tide/ionosphere epochs actually corrected, and
disp_chain.json the ERA5 scatter before/after. The chain refuses to run in a
project whose mintpy/ holds another route's products (refuse_foreign_mintpy).

What is NOT the same, and cannot be: DISP-S1 distributes unwrapped
displacement only, so the record's unwrapping (snaphu 2 with the DEFO cost,
then MintPy bridging) cannot be re-run on it. The unwrapping study on the
record found snaphu settings do not change the error rate on this ground
(HANDOFF "Unwrapping"), which is why that difference is acceptable; the
post-unwrapping checks above are the parity measures that are possible.
"""
from __future__ import annotations

import json
import logging
import shutil
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np

from . import disp_backend as db
from . import progress
from .config import PipelineConfig

log = logging.getLogger(__name__)

STAGES = ("disp_search", "disp_prefetch", "disp_stitch", "disp_resample", "mintpy_troposphere",
          "mintpy_deramp", "mintpy_dem_error", "mintpy_reference_date", "mintpy_velocity",
          "disp_products")


# ---------------------------------------------------------------- grid
def target_grid(cfg: PipelineConfig) -> dict:
    """Lat/lon grid the product is resampled onto."""
    f = int(cfg.disp_target_grid_factor or 1)
    if cfg.disp_target_grid_file:
        import h5py
        with h5py.File(cfg.disp_target_grid_file, "r") as h:
            a = dict(h.attrs)
        g = {"Y_FIRST": float(a["Y_FIRST"]), "X_FIRST": float(a["X_FIRST"]),
             "Y_STEP": float(a["Y_STEP"]) * f, "X_STEP": float(a["X_STEP"]) * f,
             "LENGTH": int(a["LENGTH"]) // f, "WIDTH": int(a["WIDTH"]) // f,
             "factor": f, "source": str(cfg.disp_target_grid_file)}
    else:
        min_lon, min_lat, max_lon, max_lat = cfg.bbox
        dlat = 30.0 / 111_320.0
        dlon = 30.0 / (111_320.0 * np.cos(np.radians(0.5 * (min_lat + max_lat))))
        g = {"Y_FIRST": float(max_lat), "X_FIRST": float(min_lon), "Y_STEP": -dlat, "X_STEP": dlon,
             "LENGTH": int(np.ceil((max_lat - min_lat) / dlat)),
             "WIDTH": int(np.ceil((max_lon - min_lon) / dlon)), "factor": 1, "source": "bbox"}
    return g


def cell_centres(g: dict) -> tuple[np.ndarray, np.ndarray]:
    lat = g["Y_FIRST"] + (np.arange(g["LENGTH"]) + 0.5) * g["Y_STEP"]
    lon = g["X_FIRST"] + (np.arange(g["WIDTH"]) + 0.5) * g["X_STEP"]
    return lat, lon


def source_coords(cfg: PipelineConfig, results, shape) -> tuple[np.ndarray, np.ndarray]:
    """Pixel-centre x/y of the cached subset.

    The window corners are AOI corners rather than pixel edges, so the exact
    grid comes from the coordinates the granule read stored in the cache
    (db.grid_coords). Only a cache written before those were stored needs one
    remote granule, and the answer is then remembered - a fully cached re-run
    no longer needs Earthdata."""
    win = None
    for p in results:
        c = db._granule_cache_path(cfg, p.properties["fileName"])
        if c.exists():
            with np.load(c, allow_pickle=False) as z:
                win = db._window_from_cache(z["__window"])
            break
    if win is None:
        raise RuntimeError("no cached granule to take the window from")
    xy = db.grid_coords(cfg, results, win, shape)
    if xy is None:
        raise RuntimeError(f"the DISP-S1 subset's pixel coordinates could not be read (stack {shape})")
    return xy


def resample_index(g: dict, xs: np.ndarray, ys: np.ndarray, epsg: int):
    """Nearest source row/col for every target cell centre, and whether it
    falls inside the source grid (within half a pixel of its edge)."""
    from pyproj import Transformer
    lat, lon = cell_centres(g)
    LON, LAT = np.meshgrid(lon, lat)
    tr = Transformer.from_crs("EPSG:4326", f"EPSG:{int(epsg)}", always_xy=True)
    X, Y = tr.transform(LON, LAT)
    dx, dy = xs[1] - xs[0], ys[1] - ys[0]
    col = np.rint((X - xs[0]) / dx).astype(np.int64)
    row = np.rint((Y - ys[0]) / dy).astype(np.int64)
    ok = (col >= 0) & (col < len(xs)) & (row >= 0) & (row < len(ys))
    return np.clip(row, 0, len(ys) - 1), np.clip(col, 0, len(xs) - 1), ok


def take(arr2d: np.ndarray, idx) -> np.ndarray:
    row, col, ok = idx
    out = arr2d[row, col].astype("float32")
    out[~ok] = np.nan
    return out


# ---------------------------------------------------------------- inputs
def geometry(cfg: PipelineConfig, g: dict, out: Path, A: dict, idx) -> Path:
    """MintPy geometry on the target grid: block mean of the record's geo
    geometry, or DISP-S1-STATIC resampled when there is no record."""
    import h5py
    lat, lon = cell_centres(g)
    data = {}
    if cfg.disp_geometry_file:
        f = g["factor"]
        with h5py.File(cfg.disp_geometry_file, "r") as h:
            for k in ("height", "incidenceAngle", "azimuthAngle", "slantRangeDistance"):
                if k in h:
                    a = h[k][: g["LENGTH"] * f, : g["WIDTH"] * f].astype("float64")
                    if k != "height":           # 0 is MintPy's no-data for angles and range
                        a[a == 0] = np.nan
                    with np.errstate(invalid="ignore"), warnings.catch_warnings():
                        warnings.simplefilter("ignore", RuntimeWarning)     # all-NaN edge blocks
                        data[k] = np.nanmean(a.reshape(g["LENGTH"], f, g["WIDTH"], f), axis=(1, 3)).astype("float32")
            if "shadowMask" in h:
                s = h["shadowMask"][: g["LENGTH"] * f, : g["WIDTH"] * f]
                data["shadowMask"] = s.reshape(g["LENGTH"], f, g["WIDTH"], f).any(axis=(1, 3))
        src = f"block mean ({f}x{f}) of {cfg.disp_geometry_file}"
    else:
        from . import static_layers
        grid = {"x0": A["x0"], "y0": A["y0"], "dx": A["dx"], "dy": A["dy"],
                "ny": A["shape"][0], "nx": A["shape"][1], "epsg": A["epsg"]}
        geo = static_layers.load(cfg, grid, frame=A["frame_id"], want_dem=True)
        inc = take(geo["incidence_deg"], idx)
        data["incidenceAngle"] = inc
        if "height" in geo:
            data["height"] = take(geo["height"], idx)
        if "los_east" in geo and "los_north" in geo:
            # MintPy azimuthAngle: LOS azimuth from north, anti-clockwise
            # positive, pointing ground -> satellite
            e, n = take(geo["los_east"], idx), take(geo["los_north"], idx)
            data["azimuthAngle"] = np.degrees(np.arctan2(-e, n)).astype("float32")
        # slant range from the incidence angle and a nominal orbit altitude;
        # it only scales the DEM-error coefficient, and 1% there is 1% of a
        # millimetre-level correction.
        H, R = 693_000.0, 6_371_000.0
        th = np.radians(inc.astype("float64"))
        look = np.arcsin(np.clip(R * np.sin(th) / (R + H), -1, 1))
        data["slantRangeDistance"] = ((R + H) * np.sin(th - look) / np.sin(th)).astype("float32")
        src = "DISP-S1-STATIC resampled"
    data["latitude"] = np.repeat(lat[:, None], g["WIDTH"], 1).astype("float32")
    data["longitude"] = np.repeat(lon[None, :], g["LENGTH"], 0).astype("float32")
    with h5py.File(out, "w") as h:
        for k, v in data.items():
            h.create_dataset(k, data=v)
        for k, v in {**_grid_attrs(g), "FILE_TYPE": "geometry", "GEOMETRY_SOURCE": src}.items():
            h.attrs[k] = v
    return out


def baselines(cfg: PipelineConfig, dates: list[str]) -> dict[str, float]:
    """Perpendicular baseline per date from a topsStack baselines/ folder,
    relative to the first date that has one."""
    if not cfg.disp_baseline_dir:
        return {}
    from mintpy.utils import isce_utils
    bl = isce_utils.read_baseline_timeseries(str(cfg.disp_baseline_dir), processor="tops") or {}
    out = {d: float(np.mean(bl[d])) for d in dates if d in bl}
    if out:
        b0 = out[sorted(out)[0]]
        out = {d: v - b0 for d, v in out.items()}
    return out


# Largest share of the record span a pixel may have held flat at its two ends
# (before its first / after its last valid epoch) and still be gap-filled.
# Holding an end flat pulls a linear rate toward zero; measured on a true
# -10 mm/yr series with the production gap_fill, the bias depends only on the
# held share (the same at 3 and 10 years): 5% -> 0.7%, 10% -> 2.8%, 15% ->
# 6.3%, 20% -> 10.6%, 45% -> 43% (all at the start; split between the ends it
# is about half). 10% keeps the bias under 3%. Pixels that stop being
# measured part-way through - active pits and spoil - are the ones that move.
END_HOLD_MAX_FRACTION = 0.10


def gap_fill(ts: np.ndarray, t: np.ndarray, min_count: int, block: int = 200_000,
             max_end_hold: float | None = END_HOLD_MAX_FRACTION, stats: dict | None = None) -> np.ndarray:
    """Linear interpolation in time over NaN epochs, in place, for pixels
    with at least min_count valid epochs; the rest become all-NaN. Before the
    first / after the last valid epoch the nearest valid value is held - but
    only for pixels whose held ends add up to at most `max_end_hold` of the
    record span (None: no limit); the others become all-NaN too and are
    counted in stats ("dropped_end_hold", and the per-pixel
    "dropped_end_hold_mask"). Returns the per-pixel valid count before any
    of this (what validEpochFraction reports)."""
    n = ts.shape[0]
    flat = ts.reshape(n, -1)
    counts = np.isfinite(flat).sum(axis=0)
    keep = counts >= min_count
    over = np.zeros(counts.shape, dtype=bool)
    if max_end_hold is not None and n > 1 and keep.any():
        fin = np.isfinite(flat)
        first = np.argmax(fin, axis=0)
        last = n - 1 - np.argmax(fin[::-1], axis=0)
        span = float(t[-1] - t[0]) or 1.0
        held = (t[first] - t[0]) + (t[-1] - t[last])
        over = keep & (held > max_end_hold * span + 1e-9)
        keep &= ~over
        del fin
    if stats is not None:
        stats["dropped_end_hold"] = int(over.sum())
        stats["dropped_end_hold_mask"] = over.reshape(ts.shape[1:])
        stats["max_end_hold_fraction"] = max_end_hold
    flat[:, ~keep] = np.nan
    cols = np.where(keep & (counts < n))[0]
    ix = np.arange(n)[:, None]
    for s in range(0, cols.size, block):
        c = cols[s:s + block]
        y = flat[:, c]
        ok = np.isfinite(y)
        prev = np.maximum.accumulate(np.where(ok, ix, -1), axis=0)
        nxt = np.minimum.accumulate(np.where(ok, ix, n)[::-1], axis=0)[::-1]
        p = np.clip(prev, 0, n - 1)
        q = np.clip(nxt, 0, n - 1)
        yp = np.take_along_axis(y, p, 0)
        yq = np.take_along_axis(y, q, 0)
        tp, tq = t[p], t[q]
        w = np.where(q > p, (t[:, None] - tp) / np.where(q > p, tq - tp, 1.0), 0.0)
        fill = np.where(prev < 0, yq, np.where(nxt >= n, yp, yp + w * (yq - yp)))
        flat[:, c] = np.where(ok, y, fill)
    return counts.reshape(ts.shape[1:])


def _grid_attrs(g: dict) -> dict:
    return {"LENGTH": str(g["LENGTH"]), "WIDTH": str(g["WIDTH"]),
            "Y_FIRST": repr(g["Y_FIRST"]), "X_FIRST": repr(g["X_FIRST"]),
            "Y_STEP": repr(g["Y_STEP"]), "X_STEP": repr(g["X_STEP"]),
            "Y_UNIT": "degrees", "X_UNIT": "degrees"}


def seconds_of_day(acq) -> int | None:
    """UTC seconds of day from "HH:MM:SS[.f]" or an ISO date-time; None if
    it cannot be read."""
    if acq is None or str(acq).strip() == "":
        return None
    s = str(acq).strip().replace("Z", "")
    for fmt in ("%H:%M:%S", "%H:%M:%S.%f"):
        try:
            tt = datetime.strptime(s, fmt)
            return tt.hour * 3600 + tt.minute * 60 + tt.second
        except ValueError:
            pass
    try:
        tt = datetime.fromisoformat(s)
        return tt.hour * 3600 + tt.minute * 60 + tt.second
    except ValueError:
        return None


def base_attrs(cfg: PipelineConfig, g: dict, A: dict) -> dict:
    """MintPy metadata for the product: the record's own where there is one
    (same track and orbit, so centre time, altitude, heading and wavelength
    are the same), otherwise the minimum MintPy needs."""
    import h5py
    keep = ("ANTENNA_SIDE", "CENTER_INCIDENCE_ANGLE", "CENTER_LINE_UTC", "EARTH_RADIUS", "HEADING", "HEIGHT",
            "ORBIT_DIRECTION", "PLATFORM", "POLARIZATION", "PRF", "RANGE_PIXEL_SIZE", "AZIMUTH_PIXEL_SIZE",
            "STARTING_RANGE", "WAVELENGTH", "trackNumber", "relative_orbit", "beam_mode", "beam_swath")
    a = {}
    if cfg.disp_target_grid_file:
        with h5py.File(cfg.disp_target_grid_file, "r") as h:
            a = {k: str(h.attrs[k]) for k in keep if k in h.attrs}
    # The ERA5 hour: tropo_pyaps3 samples the weather model at the hour
    # nearest CENTER_LINE_UTC. This used to parse the "HH:MM:SS" time with
    # datetime.fromisoformat, which raises on a bare time, and fall back to
    # 85680 (23:48) - right only by coincidence for an ascending Midwest
    # pass; at a descending site (~12 UTC) the chain subtracted midnight
    # delays from noon acquisitions. Now: the granules' own time, and an
    # error rather than a default when there is none.
    sec = A.get("acq_seconds")
    if sec is None:
        sec = seconds_of_day(A.get("acq_time"))
    if "CENTER_LINE_UTC" in a and sec is not None:
        rec_sec = float(a["CENTER_LINE_UTC"])
        gap = abs(rec_sec - sec) % 86400
        if min(gap, 86400 - gap) > 1800:
            raise RuntimeError(
                f"the target grid's record was acquired at {rec_sec:.0f} s UTC but these DISP-S1 "
                f"granules at {sec} s: a different pass, so its track, heading and centre time do "
                "not apply. Use a record on the same track and direction.")
    if "CENTER_LINE_UTC" not in a:
        if sec is None:
            raise RuntimeError("no acquisition time for the DISP-S1 granules (names or "
                               "/identification), so the weather model hour is unknown")
        a["CENTER_LINE_UTC"] = str(int(sec))
    a.setdefault("WAVELENGTH", "0.05546576")
    a.setdefault("PLATFORM", "sen")
    a.setdefault("ORBIT_DIRECTION", cfg.orbit_direction.upper())
    a.setdefault("HEIGHT", "693000.0")
    a.setdefault("EARTH_RADIUS", "6371000.0")
    a.setdefault("trackNumber", str(cfg.track))
    a.update(_grid_attrs(g))
    a.update({"PROCESSOR": "opera_disp", "UNIT": "m", "DATA_TYPE": "float32",
              "FRAME_ID": str(A.get("frame_id")), "SOURCE_PRODUCT": db.COLLECTION})
    return a


# ---------------------------------------------------------------- MintPy
def _mintpy(argv: list[str], cwd: Path, name: str, timeout: float = 6 * 3600) -> None:
    from . import procs
    t0 = time.time()
    logf = cwd / f"disp_chain_{name}.log"
    with open(logf, "w") as fh:
        r = procs.run_group(argv, cwd=cwd, timeout=timeout, stdout=fh, stderr=fh)
    if r.returncode != 0:
        tail = "\n".join(logf.read_text(errors="replace").splitlines()[-25:])
        raise RuntimeError(f"{argv[0]} failed ({r.returncode}); see {logf}\n{tail}")
    log.info("%s done in %.0f s", argv[0], time.time() - t0)


def _stage(cfg, stage, status="done", **kw):
    progress.emit(cfg.project_dir, stage, status, **kw)


def _reference_cell(cfg, g, counts, n, tc) -> tuple[int, int, str]:
    """The record's reference pixel on this grid; if DISP-S1 did not keep it
    through 90% of epochs, the nearest cell within 150 m that did. With no
    reference_lalo (a new area): the most coherent cell observed in every
    epoch, outside the deforming area and its buffer - MintPy's own rule
    (highest coherence) with the deformation kept out of it."""
    from .significance import euclidean_buffer, lalo_to_rc, panel_mask
    if not cfg.reference_lalo:
        attrs = _grid_attrs(g)
        cand = (counts == n) & np.isfinite(tc)
        dm = panel_mask(getattr(cfg, "deramp_exclude_geojson", "") or "", counts.shape, attrs)
        if dm.any():
            cand &= ~euclidean_buffer(dm, attrs, float(cfg.acceptance_far_field_buffer_m or 2000))
        if not cand.any():
            raise RuntimeError("no cell is observed in every DISP-S1 epoch; set reference_lalo")
        # coherence averaged over ~150 m, so one lucky cell cannot win, and
        # nothing within ~1 km of the grid edge
        from scipy.ndimage import uniform_filter
        v = np.nan_to_num(tc).astype("float64")
        w = np.isfinite(tc).astype("float64")
        with np.errstate(invalid="ignore", divide="ignore"):
            sm = uniform_filter(v, 5) / uniform_filter(w, 5)
        edge = max(1, int(round(1000.0 / (abs(g["Y_STEP"]) * 110_950.0))))
        cand[:edge] = cand[-edge:] = False
        cand[:, :edge] = cand[:, -edge:] = False
        if not cand.any():
            raise RuntimeError("no fully observed cell away from the grid edge; set reference_lalo")
        score = np.where(cand, np.nan_to_num(sm), -1.0)
        r, c = np.unravel_index(int(np.argmax(score)), score.shape)
        return int(r), int(c), (f"auto: most coherent fully observed cell away from the edge "
                                f"(temporal coherence {tc[r, c]:.2f}, {sm[r, c]:.2f} over ~150 m)")
    lat, lon = (float(v) for v in cfg.reference_lalo.split(","))
    r0, c0 = lalo_to_rc(_grid_attrs(g), lat, lon)
    need = 0.9 * n
    if counts[r0, c0] >= need:
        return r0, c0, "record reference pixel"
    best = None
    for rad in range(1, 6):
        rr = slice(max(0, r0 - rad), r0 + rad + 1); cc = slice(max(0, c0 - rad), c0 + rad + 1)
        ys, xs = np.where(counts[rr, cc] >= need)
        if ys.size:
            d = (ys + rr.start - r0) ** 2 + (xs + cc.start - c0) ** 2
            k = int(np.argmin(d)); best = (int(ys[k] + rr.start), int(xs[k] + cc.start))
            break
    if best is None:
        raise RuntimeError(f"no cell within 150 m of reference_lalo {cfg.reference_lalo} is valid in 90% "
                           f"of DISP-S1 epochs (the pixel itself: {int(counts[r0, c0])}/{n})")
    log.warning("DISP-S1 keeps the record's reference pixel in only %d/%d epochs; using the nearest "
                "cell that is valid in 90%%, (%d, %d)", int(counts[r0, c0]), n, *best)
    return best[0], best[1], f"nearest 90%-valid cell to the record reference ({int(counts[r0, c0])}/{n} there)"


def _json_safe(o):
    if isinstance(o, dict):
        return {str(k): _json_safe(v) for k, v in o.items() if not isinstance(v, np.ndarray) or v.size < 64}
    if isinstance(o, (list, tuple)):
        return [_json_safe(v) for v in o]
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return o


def run(cfg: PipelineConfig) -> Path:
    import h5py
    t_start = time.time()
    M = cfg.mintpy_dir
    # Before anything is written: this writes timeseries.h5, temporal
    # coherence and masks at the mintpy root and replaces geo/geo_*.h5, which
    # in a stack or phase-linked project are the official products.
    db.refuse_foreign_mintpy(cfg)
    M.mkdir(parents=True, exist_ok=True)
    (M / "inputs").mkdir(exist_ok=True)
    (M / "geo").mkdir(exist_ok=True)
    rec: dict = {"product": db.COLLECTION, "frame": cfg.disp_frame_id, "steps": []}

    _stage(cfg, "disp_search", "start")
    results = db.subsample(db.search(cfg), cfg.disp_max_per_year)
    if not results:
        raise RuntimeError("No DISP-S1 coverage for this AOI and date range")
    rec["frame"] = db._frame_of(results[0].properties["fileName"]) or cfg.disp_frame_id
    rec["versions"] = sorted({db._version(p.properties["fileName"]) for p in results})
    _stage(cfg, "disp_search", "done", done=len(results), total=len(results))

    rec["prefetch"] = db.prefetch(cfg, results)

    _stage(cfg, "disp_stitch", "start", total=len(results))
    A = db.assemble_stack(cfg, results)
    _stage(cfg, "disp_stitch", "done", done=len(A["dates"]), total=len(results))
    dates = list(A["dates"])
    rec["stitch"] = {
        "epochs": len(dates), "first": dates[0], "last": dates[-1],
        "skipped_granules": A["skipped"], "lost_anchor_granules": A["lost_anchors"],
        "reference_epoch_boundaries": len(A["boundary_idx"]), "anchor_epochs": len(A["anchor_idx"]),
        "corrections_applied": A["corrections_applied"], "epochs_fully_corrected": A["corrected_epochs"],
        "corrections": A["corrections"],
        "conncomp0_dropped_mean_percent": round(100 * float(np.mean(A["conncomp_dropped"])), 2) if A["conncomp_dropped"] else 0.0,
        "conncomp0_dropped_worst_percent": round(100 * float(np.max(A["conncomp_dropped"])), 2) if A["conncomp_dropped"] else 0.0,
        "conncomp0_masked": bool(cfg.disp_mask_connected_components),
        "conncomp0_share_median_percent": round(100 * float(np.median(A["conncomp0_share"])), 1) if A["conncomp0_share"] else None,
        "conncomp0_share_max_percent": round(100 * float(np.max(A["conncomp0_share"])), 1) if A["conncomp0_share"] else None,
        "anchor_conncomp0_share_median_percent": (
            round(100 * float(np.median(A["anchor_conncomp0_share"])), 1)
            if A.get("anchor_conncomp0_share") is not None else None),
        "boundary_steps": _json_safe(A["boundary_steps"]), "boundary_datum": _json_safe(A["boundary_datum"]),
        "coverage_ratchet": A["coverage_ratchet"],
        "unwrap_spikes": _json_safe(A["jumps"]),
        "acquisition_time_utc": A.get("acq_time"), "grid_origin": A.get("grid_origin"),
        "problems": A["stitch_problems"],
    }
    # Refuse a stitch known to be wrong (partial corrections, lost anchor,
    # coverage ratchet) before an hour of MintPy turns it into a report.
    # The measured record is written first so the refusal can be read.
    (M / "disp_chain.json").write_text(json.dumps(_json_safe({**rec, "status": "stitch refused"
                                                               if any(p["hard"] for p in A["stitch_problems"])
                                                               else "stitched"}),
                                                   indent=1, default=str))
    db.check_stitch(A, cfg)

    # ---------------- resample onto the target grid
    _stage(cfg, "disp_resample", "start")
    g = target_grid(cfg)
    xs, ys = source_coords(cfg, results, A["shape"])
    idx = resample_index(g, xs, ys, A["epsg"])
    bperp = baselines(cfg, dates)
    if cfg.disp_baseline_dir:
        keep = [i for i, d in enumerate(dates) if d in bperp]
        dropped = [d for d in dates if d not in bperp]
        if dropped:
            log.warning("No orbit-stack baseline for %d DISP-S1 epoch(s) %s - dropped after stitching "
                        "(the carry across them is already applied)", len(dropped), ", ".join(dropped))
        rec["epochs_dropped_no_baseline"] = dropped
    else:
        keep = list(range(len(dates)))
        log.warning("disp_baseline_dir is empty: the DEM-error correction will be SKIPPED")
    dates = [dates[i] for i in keep]
    n = len(dates)
    ts = np.empty((n, g["LENGTH"], g["WIDTH"]), dtype="float32")
    for k, i in enumerate(keep):
        ts[k] = take(A["ts"][i], idx)
    t = np.array([(datetime.strptime(d, "%Y%m%d") - datetime.strptime(dates[0], "%Y%m%d")).days / 365.25
                  for d in dates])
    tc = take(db.mean_layer(A, db.COHERENCE_LAYER) if db.COHERENCE_LAYER in A["sums"]
              else db.mean_layer(A, db.QUALITY_LAYER), idx)
    water = db.mean_layer(A, db.WATER_LAYER)
    recm = db.mean_layer(A, db.MASK_LAYER)
    misclosure = None
    if A["residual_obs"] is not None:
        with np.errstate(invalid="ignore", divide="ignore"):
            frac = A["residual_bad"] / np.where(A["residual_obs"] > 0, A["residual_obs"], np.nan)
        misclosure = take(frac.astype("float32"), idx)
    anchor_suspect = take(A["anchor_suspect"].astype("float32"), idx)
    anchor_cc0 = (take(A["anchor_conncomp0_share"], idx)
                  if A.get("anchor_conncomp0_share") is not None else None)
    del A["ts"], A["flat"], A["finite"]

    # gap-fill, with a measured effect on the rate
    min_count = max(3, int(round(cfg.disp_min_valid_fraction * n)))
    raw_counts = np.isfinite(ts).sum(axis=0)
    probe = np.where((raw_counts >= min_count) & (raw_counts < n))
    rng = np.random.default_rng(0)
    pick = rng.choice(probe[0].size, size=min(5000, probe[0].size), replace=False) if probe[0].size else []
    pr, pc = probe[0][pick], probe[1][pick]
    y_gappy = ts[:, pr, pc].copy()
    gstats: dict = {}
    counts = gap_fill(ts, t, min_count, stats=gstats)
    end_dropped = gstats.pop("dropped_end_hold_mask")
    y_filled = ts[:, pr, pc]
    gap_effect = None
    if len(pick):
        def slopes(Y):
            out = np.full(Y.shape[1], np.nan)
            for j in range(Y.shape[1]):
                m = np.isfinite(Y[:, j])
                if m.sum() >= 3:
                    out[j] = np.polyfit(t[m], Y[m, j], 1)[0]
            return out
        dv = (slopes(y_filled) - slopes(y_gappy)) * 1000
        dv = dv[np.isfinite(dv)]           # pixels the end-hold limit dropped have no filled rate
        if dv.size:
            gap_effect = {"pixels_sampled": int(dv.size),
                          "median_abs_rate_change_mm_yr": round(float(np.median(np.abs(dv))), 3),
                          "p95_abs_rate_change_mm_yr": round(float(np.percentile(np.abs(dv), 95)), 3)}
            log.info("Gap-fill changes the linear rate by %.3f mm/yr median, %.3f at the 95th percentile "
                     "(%d gappy pixels sampled)", gap_effect["median_abs_rate_change_mm_yr"],
                     gap_effect["p95_abs_rate_change_mm_yr"], dv.size)
    valid = (counts >= min_count) & ~end_dropped
    frac_valid = counts / n
    if gstats.get("dropped_end_hold"):
        log.warning("%d pixels left out: their first or last valid epoch is more than %.0f%% of the "
                    "record from its end, and holding the end flat would shrink their rate",
                    gstats["dropped_end_hold"], 100 * END_HOLD_MAX_FRACTION)
    rec["gap_fill"] = {"min_valid_epochs": min_count, "pixels_complete": int((counts == n).sum()),
                       "pixels_gap_filled": int((valid & (counts < n)).sum()),
                       "pixels_dropped": int((~valid & np.isfinite(tc)).sum()),
                       "pixels_dropped_end_hold": int(gstats.get("dropped_end_hold", 0)),
                       "max_end_hold_share_of_record": END_HOLD_MAX_FRACTION,
                       "mean_filled_share_of_epochs": round(float(1 - frac_valid[valid & (counts < n)].mean()), 4)
                       if (valid & (counts < n)).any() else 0.0,
                       "rate_effect": gap_effect}

    # spatial reference
    ry, rx, how = _reference_cell(cfg, g, raw_counts, n, tc)
    ts -= ts[:, ry, rx][:, None, None]
    lat, lon = cell_centres(g)
    rec["reference"] = {"row": ry, "col": rx, "lat": round(float(lat[ry]), 6), "lon": round(float(lon[rx]), 6), "how": how}

    attrs = base_attrs(cfg, g, A)
    attrs.update({"FILE_TYPE": "timeseries", "REF_DATE": dates[0], "REF_Y": str(ry), "REF_X": str(rx),
                  "REF_LAT": str(round(float(lat[ry]), 6)), "REF_LON": str(round(float(lon[rx]), 6)),
                  "START_DATE": dates[0], "END_DATE": dates[-1]})
    geom_f = geometry(cfg, g, M / "inputs" / "geometryGeo.h5", A, idx)
    bp = np.array([bperp.get(d, 0.0) for d in dates], dtype="float32")
    with h5py.File(M / "timeseries.h5", "w") as h:
        h.create_dataset("timeseries", data=ts, chunks=(n, 128, 128) if min(ts.shape[1:]) >= 128 else None)
        h.create_dataset("date", data=np.array(dates, dtype="S8"))
        h.create_dataset("bperp", data=bp)
        for k, v in attrs.items():
            h.attrs[k] = v
    del ts

    # quality layers and masks on the target grid
    observable = np.isfinite(tc)
    if water is not None:
        observable &= ~(take(water, idx) < 0.5)
    good = valid & observable & (np.nan_to_num(tc) >= cfg.min_temporal_coherence)

    def write2d(path, name, data, ftype, extra=None):
        with h5py.File(path, "w") as h:
            h.create_dataset(name, data=data)
            for k, v in {**attrs, "FILE_TYPE": ftype, **(extra or {})}.items():
                h.attrs[k] = v
    tc_out = np.where(valid, tc, np.nan).astype("float32")
    write2d(M / "temporalCoherence.h5", "temporalCoherence", tc_out, "temporalCoherence",
            {"temporalCoherenceSource": "OPERA DISP-S1 temporal_coherence (dolphin phase linking), mean over "
                                        "every 8th granule"})
    write2d(M / "maskTempCoh.h5", "mask", good, "mask")
    write2d(M / "maskObservable.h5", "mask", observable, "mask")
    layers = {"validEpochFraction": (frac_valid.astype("float32"),
                                     "share of epochs in which the cell was measured")}
    write2d(M / "validEpochFraction.h5", "validEpochFraction", layers["validEpochFraction"][0], "mask")
    if recm is not None:
        layers["recommendedMask"] = ((take(recm, idx) > 0.5).astype("float32"),
                                     "OPERA recommended_mask, majority over the sampled granules (1 = keep)")
        write2d(M / "recommendedMask.h5", "mask", take(recm, idx) > 0.5, "mask")
    if misclosure is not None:
        layers["networkMisclosureFraction"] = (misclosure, "share of epochs whose L1 inversion residual is at "
                                                           "least pi (a whole-cycle inconsistency)")
        write2d(M / "networkMisclosureFraction.h5", "networkMisclosureFraction", misclosure, "mask",
                {"NOTE": "share of epochs whose L1 inversion residual is at least pi (a whole-cycle inconsistency)"})
    # The anchor check per cell: how many anchors put a suspected whole-cycle
    # step into every later epoch, and how often the carried value came from
    # snaphu's unreliable component 0 - the cells to grey out on a screening
    # map, and what a reviewer looks at when a patch seems suspicious.
    layers["anchorSuspect"] = (anchor_suspect, "number of anchor epochs at which the cell carries a suspected "
                                               "whole-cycle (27.7 mm) error into every later epoch")
    write2d(M / "anchorSuspect.h5", "anchorSuspect", anchor_suspect, "mask",
            {"NOTE": layers["anchorSuspect"][1]})
    if anchor_cc0 is not None:
        layers["anchorConncomp0Share"] = (anchor_cc0, "share of anchor epochs whose carried value was in "
                                                      "snaphu connected component 0 (not reliably unwrapped)")
        write2d(M / "anchorConncomp0Share.h5", "anchorConncomp0Share", anchor_cc0, "mask",
                {"NOTE": layers["anchorConncomp0Share"][1]})
    _stage(cfg, "disp_resample", "done", msg=f"{n} epochs on a {g['LENGTH']}x{g['WIDTH']} grid")

    # ---------------- the record's MintPy chain
    weather = Path(cfg.disp_weather_dir) if cfg.disp_weather_dir else M
    steps = []
    cur = "timeseries.h5"
    if cfg.mintpy_tropo_method == "pyaps":
        _stage(cfg, "mintpy_troposphere", "start")
        _mintpy(["tropo_pyaps3.py", "-f", cur, "-g", str(geom_f), "-m", "ERA5", "-w", str(weather),
                 "-o", "timeseries_ERA5.h5"], M, "tropo")
        # Did it help? A tropospheric delay is uncorrelated with time, so a
        # right correction tightens each series about its own trend; a wrong
        # hour, sign or model loosens it (disp_backend._apply_tropo applies
        # the same test). Measured and recorded; the record's chain applies
        # ERA5 unconditionally, so it is not reverted, but it is flagged.
        rec["troposphere"] = _tropo_effect(M / cur, M / "timeseries_ERA5.h5", t,
                                           attrs.get("CENTER_LINE_UTC"))
        cur = "timeseries_ERA5.h5"; steps.append("ERA5 troposphere (pyaps3)")
        _stage(cfg, "mintpy_troposphere")
    else:
        _stage(cfg, "mintpy_troposphere", "skipped", msg=f"mintpy_tropo_method={cfg.mintpy_tropo_method}")
    dm = None
    if cfg.mintpy_deramp and cfg.mintpy_deramp != "no":
        _stage(cfg, "mintpy_deramp", "start")
        from .stack_backend import build_deramp_mask
        buf = float(cfg.deramp_buffer_m or cfg.acceptance_far_field_buffer_m or 2000)
        dm = build_deramp_mask(cfg, buffer_m=buf)
        out = cur.replace(".h5", "_ramp.h5")
        _mintpy(["remove_ramp.py", cur, "-s", cfg.mintpy_deramp, "-m", str(dm), "-o", out], M, "deramp")
        with h5py.File(dm, "r") as h:
            rec["deramp_mask_cells"] = int(h["mask"][:].sum())
        cur = out; steps.append(f"{cfg.mintpy_deramp} deramp outside the mine panels + buffer")
        _stage(cfg, "mintpy_deramp")
    dem_ran = False
    t_dem = time.time()
    if bperp:
        _stage(cfg, "mintpy_dem_error", "start")
        out = cur.replace(".h5", "_demErr.h5")
        _mintpy(["dem_error.py", cur, "-g", str(geom_f), "-p", "2", "-o", out,
                 "--num-worker", str(cfg.mintpy_num_workers), "--ram", str(cfg.mintpy_max_memory_gb)], M, "dem_error")
        cur = out; steps.append("DEM-error correction (order-2 polynomial, record baselines)")
        dem_ran = True
        _stage(cfg, "mintpy_dem_error")
    else:
        _stage(cfg, "mintpy_dem_error", "skipped", msg="no baselines")
    # minimum-RMS reference date, as smallbaselineApp does - from the
    # residual THIS run's dem_error wrote. A timeseriesResidual.h5 left by
    # another run (or another dataset) must not choose the date.
    _stage(cfg, "mintpy_reference_date", "start")
    ref_date, from_rms = dates[0], False
    resid = M / "timeseriesResidual.h5"
    if dem_ran and resid.exists() and resid.stat().st_mtime >= t_dem - 1:
        t_rms = time.time()
        _mintpy(["timeseries_rms.py", str(resid), "-m", str(M / "maskTempCoh.h5"), "-r", "quadratic"], M, "rms")
        rms_txt = sorted((p for p in M.glob("rms_timeseriesResidual*.txt") if p.stat().st_mtime >= t_rms - 1),
                         key=lambda p: p.stat().st_mtime)
        if rms_txt:
            rows = [l.split() for l in rms_txt[-1].read_text().splitlines() if l.strip() and not l.startswith("#")]
            rows = [r for r in rows if r and r[0] in dates]
            if rows:
                ref_date, from_rms = min(rows, key=lambda r: float(r[1]))[0], True
    rec["reference_date_rule"] = ("minimum residual RMS (this run's DEM-error residual)" if from_rms
                                  else "first epoch (no DEM-error residual from this run)")
    _mintpy(["reference_date.py", cur, "-r", ref_date], M, "reference_date")
    rec["reference_date"] = ref_date
    _stage(cfg, "mintpy_reference_date", msg=ref_date)
    _stage(cfg, "mintpy_velocity", "start")
    _mintpy(["timeseries2velocity.py", cur, "-o", "velocity.h5"], M, "velocity")
    steps.append("linear rate (timeseries2velocity, residue uncertainty)")
    _stage(cfg, "mintpy_velocity")

    # ---------------- into the record's layout
    _stage(cfg, "disp_products", "start")
    G = M / "geo"
    for f in G.glob("geo_*.h5"):
        f.unlink()
    final = f"geo_{cur}"          # named for the corrections actually applied
    shutil.move(str(M / cur), str(G / final))
    for src, dst in (("velocity.h5", "geo_velocity.h5"), ("temporalCoherence.h5", "geo_temporalCoherence.h5"),
                     ("maskTempCoh.h5", "geo_maskTempCoh.h5")):
        shutil.copy2(M / src, G / dst)
    shutil.copy2(geom_f, G / "geo_geometryRadar.h5")
    vel = None
    with h5py.File(G / "geo_velocity.h5", "a") as h:
        for k in ("velocity", "velocityStd"):
            if k in h:
                v = h[k][:]; v[~valid] = np.nan; h[k][...] = v
                if k == "velocity":
                    vel = v
        h.attrs["DISP_CHAIN"] = "; ".join(steps)
    with h5py.File(G / final, "a") as h:
        d = h["timeseries"]
        for i in range(d.shape[0]):
            a = d[i]; a[~valid] = np.nan; d[i] = a
    corr = A["corrections"]
    stitch_text = ({"all": f"SET + iono removed from all {corr['epochs']} epochs",
                    "none": "SET + iono NOT removed (no epoch could be corrected)",
                    "off": "SET + iono not removed (disp_apply_corrections false)"}
                   .get(corr["status"], f"SET + iono removed from {corr['epochs_corrected']} of {corr['epochs']} epochs"))
    rec.update({"steps": ["stitch (" + stitch_text + ", " + ("conncomp 0 dropped, " if cfg.disp_mask_connected_components
                                                             else "conncomp 0 kept as the record does, ")
                          + "reference-epoch carry, whole-cycle spikes rejected)",
                          f"resample to {g['LENGTH']}x{g['WIDTH']} lat/lon grid ({g['source']}, factor {g['factor']})",
                          "gap-fill in time", "reference pixel"] + steps,
                "grid": {k: g[k] for k in ("Y_FIRST", "X_FIRST", "Y_STEP", "X_STEP", "LENGTH", "WIDTH", "factor", "source")},
                "coverage_percent_at_threshold": round(100 * float(good.sum() / max(1, observable.sum())), 1),
                "network_misclosure": _misclosure_summary(misclosure, good, tc_out),
                "misclosure_vs_rate": _misclosure_vs_rate(misclosure, vel, tc_out, good, dm),
                "anchor_suspect_cells": int((np.nan_to_num(anchor_suspect) > 0).sum()),
                "status": "complete",
                "wall_s": round(time.time() - t_start)})
    rec["quality_layers"] = deliver_quality_layers(cfg, layers, attrs)
    (M / "disp_chain.json").write_text(json.dumps(_json_safe(rec), indent=1, default=str))
    from . import corrections
    # What was MEASURED, not what the route would normally do: decide()
    # says "always subtracted" for every DISP run, and the report used to
    # repeat that whatever the stitch managed.
    full = corr["status"] == "all"
    corrections.record(cfg, extra={
        "solid_earth_tides": full, "ionosphere": "product layer" if full else None,
        "disp_corrections": corr,
        "why": ({"all": f"OPERA DISP-S1 tide and ionosphere layers subtracted from all {corr['epochs']} epochs",
                 "none": "OPERA DISP-S1 tide and ionosphere layers could not be read for any epoch; not removed",
                 "off": "OPERA DISP-S1 tide and ionosphere layers not applied (disp_apply_corrections false)"}
                .get(corr["status"], f"OPERA DISP-S1 tide and ionosphere layers subtracted from "
                                     f"{corr['epochs_corrected']} of {corr['epochs']} epochs"))})
    _stage(cfg, "disp_products", msg=f"{len(steps)} MintPy steps; products in {G}")
    log.info("DISP-S1 chain complete in %.0f min: %s", (time.time() - t_start) / 60, "; ".join(steps))
    return M


def _tropo_effect(before: Path, after: Path, t: np.ndarray, center_line_utc) -> dict:
    """Median per-pixel scatter about a straight line, before and after the
    ERA5 correction, on every ~40th row (a few tens of MB however large the
    grid; tropo_disp.scatter_about_trend samples 4000 pixels of it)."""
    import h5py
    from .tropo_disp import scatter_about_trend
    out = {"method": "ERA5 (pyaps3)", "center_line_utc_s": center_line_utc}
    try:
        vals = []
        for f in (before, after):
            with h5py.File(f, "r") as h:
                d = h["timeseries"]
                step = max(1, d.shape[1] // 40)
                vals.append(scatter_about_trend(d[:, ::step, :], t))
        b, a = vals
    except Exception as exc:
        out["error"] = str(exc)
        return out
    out.update({"scatter_before_mm": round(1000 * b, 2), "scatter_after_mm": round(1000 * a, 2)})
    if np.isfinite(b) and np.isfinite(a) and b > 0:
        out["change_percent"] = round(100 * (a - b) / b, 1)
        out["helped"] = bool(a < b)
        if a >= b:
            log.error("The ERA5 correction made the DISP-S1 series WORSE: scatter about trend %.2f -> "
                      "%.2f mm. Check CENTER_LINE_UTC (%s s) and the weather files.", 1000 * b,
                      1000 * a, center_line_utc)
        else:
            log.info("ERA5 correction: scatter about trend %.2f -> %.2f mm (%+.1f%%)", 1000 * b,
                     1000 * a, out["change_percent"])
    return out


def _misclosure_vs_rate(mis, vel, tc, good, deramp_mask_file) -> dict | None:
    """
    Does misclosure predict a worse rate? Over far-field cells (the deramp
    fit cells: coherent, outside the mines + buffer), within each coherence
    class, the robust spread of the rate about the class median for cells
    above vs below the class's median misclosure. Equal spreads mean the
    misclosure layer says nothing about rate error; a clearly larger spread
    in high-misclosure cells is the case for masking on it.
    """
    if mis is None or vel is None:
        return None
    far = good.copy()
    if deramp_mask_file is not None:
        try:
            import h5py
            with h5py.File(deramp_mask_file, "r") as h:
                m = h["mask"][:].astype(bool)
            if m.shape == far.shape:
                far &= m
        except Exception:
            pass
    from .significance import class_labels, coherence_class
    k = coherence_class(tc)
    v = vel * 1000.0
    rows = []

    def rsd(x):
        return float(1.4826 * np.median(np.abs(x - np.median(x)))) if x.size else float("nan")
    for i, lab in enumerate(class_labels()):
        sel = far & (k == i) & np.isfinite(v) & np.isfinite(mis)
        if sel.sum() < 400:
            continue
        cut = float(np.median(mis[sel]))
        lo, hi = sel & (mis <= cut), sel & (mis > cut)
        if lo.sum() < 100 or hi.sum() < 100:
            continue
        c = float(np.median(v[sel]))
        rows.append({"tc": lab, "misclosure_split": round(cut, 4), "n_low": int(lo.sum()), "n_high": int(hi.sum()),
                     "spread_low_mm_yr": round(rsd(v[lo] - c), 2), "spread_high_mm_yr": round(rsd(v[hi] - c), 2),
                     "median_low_mm_yr": round(float(np.median(v[lo]) - c), 2),
                     "median_high_mm_yr": round(float(np.median(v[hi]) - c), 2)})
    if not rows:
        return {"by_coherence": [], "note": "too few far-field cells with misclosure to compare"}
    ratio = float(np.median([r["spread_high_mm_yr"] / r["spread_low_mm_yr"] for r in rows
                             if r["spread_low_mm_yr"] > 0])) if rows else None
    return {"by_coherence": rows, "spread_ratio_high_to_low": round(ratio, 2) if ratio else None,
            "note": ("far-field cells (deramp fit cells) split at each coherence class's median network "
                     "misclosure; ratio > 1 means high-misclosure cells read less consistent rates")}


def deliver_quality_layers(cfg, layers: dict, attrs: dict) -> dict:
    """The per-cell DISP-S1 quality layers as GeoTIFFs under
    deliverables/technical/disp_quality/, so a reviewer can open them next
    to the rate map (until 5 Oct 2026 they were written to mintpy/ and read
    by nothing). Returns {layer: path}."""
    out_dir = Path(cfg.project_dir) / "deliverables" / "technical" / "disp_quality"
    out_dir.mkdir(parents=True, exist_ok=True)
    names = {"validEpochFraction": "valid_epoch_fraction", "recommendedMask": "recommended_mask",
             "networkMisclosureFraction": "network_misclosure_fraction",
             "anchorSuspect": "anchor_suspect_count", "anchorConncomp0Share": "anchor_conncomp0_share"}
    written = {}
    for key, (arr, desc) in layers.items():
        path = out_dir / f"{names.get(key, key)}.tif"
        try:
            _write_tif(path, np.asarray(arr, dtype="float32"), attrs, desc)
            written[key] = str(path)
        except Exception as exc:
            log.warning("Could not write %s (%s)", path.name, exc)
    return written


def _write_tif(path: Path, arr: np.ndarray, attrs: dict, description: str) -> None:
    tags = {"SOURCE": "OPERA DISP-S1 (screening route)", "LAYER": path.stem, "DESCRIPTION": description}
    try:
        from . import deliverables as dl
        dl._write_geotiff(path, arr, attrs, tags=tags, description=description)
        if path.exists():
            return
    except Exception as exc:        # noqa: BLE001 - the writer without tags, or not importable
        log.debug("deliverables._write_geotiff not used for %s (%s)", path.name, exc)
    import rasterio
    from rasterio.transform import from_origin
    x0, y0 = float(attrs["X_FIRST"]), float(attrs["Y_FIRST"])
    dx, dy = float(attrs["X_STEP"]), float(attrs["Y_STEP"])
    crs = f"EPSG:{int(float(attrs['EPSG']))}" if attrs.get("EPSG") else "EPSG:4326"
    with rasterio.open(path, "w", driver="GTiff", height=arr.shape[0], width=arr.shape[1], count=1,
                       dtype="float32", crs=crs, transform=from_origin(x0, y0, dx, abs(dy)),
                       nodata=np.nan, compress="deflate") as dst:
        dst.write(arr, 1)
        dst.update_tags(**tags)
        dst.set_band_description(1, description)


def _misclosure_summary(mis, good, tc) -> dict | None:
    if mis is None:
        return None
    sel = good & np.isfinite(mis)
    if not sel.any():
        return None
    out = {"median_share_of_epochs": round(float(np.median(mis[sel])), 4),
           "mean_share_of_epochs": round(float(np.mean(mis[sel])), 4), "by_coherence": []}
    for lo, hi in ((0.4, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.01)):
        s = sel & (tc >= lo) & (tc < hi)
        if s.sum() > 200:
            out["by_coherence"].append({"tc": f"{lo}-{min(hi, 1.0)}", "n": int(s.sum()),
                                        "mean_share": round(float(mis[s].mean()), 4)})
    out["note"] = ("share of epochs where dolphin's L1 inversion residual for that date is at least pi - the "
                   "unwrapped nearest-3 interferograms disagree by a whole cycle. Counterpart of the record's "
                   "triplet-closure diagnostic; not masked, because the L1 inversion already out-votes the "
                   "inconsistent pair.")
    return out
