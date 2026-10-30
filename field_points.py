"""Measurement points over the WHOLE study area - one table the client KMZ
and the results viewer both read, so the two cannot disagree.

Until 8 Oct 2026 the KMZ placed ~400 points per report area on a coarse grid
and ~150 per feature, so a feature looked dense and the field around it
looked empty (a coworker's review of the Site B KMZ). A Sentinel-1 value
already combines ground within about 90 m x 240 m, so a grid finer than
about 100 m only repeats values: Site B's study area holds ~243,000
distinct radar measurements behind 1.8 million 15 m map cells, ~41,000 points
at 100 m; Site A ~71,000.

    grid_cells(mask, px_m, spacing_m)   the cell nearest the centre of each
                                        spacing x spacing block, inside mask
                                        (blocks aligned to the product grid,
                                        so a rebuild picks the same cells)
    gather(cfg, rep, ctx, cells, ...)   each cell's own figures (the rule of
                                        rate_signal_to_noise.tif and the
                                        client rasters) and the 3 x 3 cells'
                                        series against farmland of each
                                        cell's own coherence class, in one
                                        pass over the cube
    relative_change(ctx)                the line-of-sight change from the first
                                        to the last image per cell, relative to
                                        farmland of the same class - the
                                        client raster line_of_sight_cumulative_
                                        displacement_* and every balloon read it
    save(fd, path) / load(path)         the viewer's copy:
                                        deliverables/viewer/field_points.npz

Table version 2 (19 Oct 2026) adds what the results viewer's date windows
are computed from (viewer_bundle.window_figures): each point's OWN cell's
series against the same farmland frame (cell_los - the 3 x 3 average above
differs from the stored rate by up to 12.5 mm/yr; the own cell's slope IS
it), its coherence class and whether it is the report's farmland, a sample
of farmland cells per class (_far_sample) for an untested window's noise,
the class frames, and a check that the own-cell series give the stored rate
and change back over the whole record.

Every figure is the report's: vertical = relative LOS rate / los_up (+ the
GNSS regional rate when the report is anchored); signal-to-noise =
|relative LOS rate| / that cell's farmland null (rep cell_null_by_class),
1.96 or more stands out. The series are LOS mm, farmland-relative, NOT zeroed:
a chart zeroes them on their own trend (series_vertical_in).

    python -m insar_pipeline.field_points <config.yaml>    the viewer copy only
"""
from __future__ import annotations

import json
import logging
import math
import warnings
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

SPACING_M = 100.0          # ~ one point per independent radar measurement (90 m x 240 m)
MAX_POINTS = 150_000       # a larger area takes a wider spacing rather than an unopenable KMZ
MM_IN = 25.4
Z95 = 1.96
VIEWER_FILE = "field_points.npz"
# 2 (19 Oct 2026): + each point's own-cell series and coherence class, the
# report's farmland flag, a farmland sample and the class frames - what the
# results viewer's date windows are computed from
TABLE_VERSION = 2
FAR_SAMPLE_PER_CLASS = 2000    # farmland cells per class for an untested window's noise (all when fewer)
MIN_SPREAD_CELLS = 200         # significance.farmland_cell_spread: a class's spread needs this many cells


# ------------------------------------------------------------------ cells
def grid_cells(mask: np.ndarray, px_m: float, spacing_m: float = SPACING_M,
               max_points: int | None = MAX_POINTS) -> tuple[np.ndarray, np.ndarray, float]:
    """(rows, cols, spacing used): per k x k block of the product grid (k =
    spacing / cell size, at least 1), the mask cell nearest the block centre.
    The spacing widens until the count is at most max_points."""
    rows, cols = np.nonzero(mask)
    if not rows.size:
        return rows, cols, spacing_m
    k = max(1, int(round(spacing_m / px_m)))
    while True:
        br, bc = rows // k, cols // k
        nbc = int(bc.max()) + 1
        block = br.astype(np.int64) * nbc + bc
        d2 = (rows - (br * k + (k - 1) / 2)) ** 2 + (cols - (bc * k + (k - 1) / 2)) ** 2
        order = np.lexsort((d2, block))
        first = np.ones(order.size, bool)
        first[1:] = block[order][1:] != block[order][:-1]
        pick = order[first]
        if max_points is None or pick.size <= max_points or k > 200:
            break
        k = int(math.ceil(k * math.sqrt(pick.size / max_points)))
    return rows[pick], cols[pick], k * px_m


def _neighbours(rows: np.ndarray, cols: np.ndarray, good: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(flat index (n, 9), valid (n, 9)): the 3 x 3 good cells around each
    point; a point with none uses its own cell (as client_kmz did)."""
    ny, nx = good.shape
    dr, dc = np.meshgrid([-1, 0, 1], [-1, 0, 1], indexing="ij")
    rr = rows[:, None] + dr.ravel()[None, :]
    cc = cols[:, None] + dc.ravel()[None, :]
    inside = (rr >= 0) & (rr < ny) & (cc >= 0) & (cc < nx)
    rr, cc = np.clip(rr, 0, ny - 1), np.clip(cc, 0, nx - 1)
    valid = inside & good[rr, cc]
    none = ~valid.any(axis=1)
    valid[none, 4] = True                                   # the centre cell
    return rr * nx + cc, valid


# ------------------------------------------------------------------ farmland frame
def _class_setup(ctx):
    """(measured classes, their farmland cells, tilt, lender per class, slot
    count). A class with too few farmland cells to measure BORROWS the frame
    of the nearest measured class (ties to the lower), as its rate borrows
    that class's centre (significance.class_means); with no class measured
    every class takes the pooled farmland, as class_means pools. Until
    19 Oct 2026 a borrowing class had no frame at all, so its cells had no
    change and no series: one project's line-of-sight cumulative raster missed
    132 cells of coherence 0.90-0.98 that its velocity raster had."""
    from . import significance as sg
    n_cls = len(sg.CLASS_EDGES) - 1
    classes = sorted(int(k) for k in ctx.class_means if isinstance(k, (int, np.integer)))
    far_idx = {k: np.flatnonzero((ctx.far & (ctx.kcls == k)).ravel()) for k in classes}
    tilt = getattr(ctx, "class_tilt", {}) or {}
    if classes:
        lender = {i: min(classes, key=lambda j: (abs(j - i), j)) for i in range(n_cls)}
    else:
        far_idx = {-2: np.flatnonzero((ctx.far & (ctx.kcls >= 0)).ravel())}      # the pooled farmland
        lender = {i: -2 for i in range(n_cls)}
    return classes, far_idx, tilt, lender, n_cls


def _slot(kcls: np.ndarray, setup) -> np.ndarray:
    """Each cell's index into _frame's array: its coherence class, or the
    last (always NaN) slot below the first class, as class_matched_field
    leaves those cells without a relative rate."""
    n_cls = setup[4]
    k = np.asarray(kcls)
    return np.where((k >= 0) & (k < n_cls), k, n_cls)


def _frame(v: np.ndarray, setup, dt: float) -> np.ndarray:
    """Farmland's value of each coherence class at one epoch (mm): the mean
    of the class's (or its lender's) farmland, + the lender's tilt x the
    time since the first image (class_tilt re-centres each class series on
    its centre, as the report's zero does). One slot per class, then a NaN
    slot for cells below the first class (_slot)."""
    classes, far_idx, tilt, lender, n_cls = setup
    own = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for k, ix in far_idx.items():
            if ix.size:
                own[k] = np.nanmean(v[ix]) + float(tilt.get(k, 0.0)) * dt
    fe = np.full(n_cls + 1, np.nan)
    for i in range(n_cls):
        fe[i] = own.get(lender[i], np.nan)
    return fe


def relative_change(ctx) -> np.ndarray:
    """Per cell, the line-of-sight change (mm) from the first to the last
    image, each image relative to farmland of the cell's coherence class
    (the series' own frame); NaN off ctx.good. A single-date difference: it
    carries each date's noise, which a chart of the series shows."""
    import h5py
    setup = _class_setup(ctx)
    kc = _slot(ctx.kcls.ravel(), setup)
    with h5py.File(ctx.ts_file, "r") as f:
        ds = f["timeseries"]
        a, b = ds[0].ravel() * 1000.0, ds[ds.shape[0] - 1].ravel() * 1000.0
    t = ctx.t
    ra = a - _frame(a, setup, 0.0)[kc]
    rb = b - _frame(b, setup, float(t[-1] - t[0]))[kc]
    out = (rb - ra).reshape(ctx.good.shape)
    out[~ctx.good] = np.nan
    return out


def report_null(cfg, rep, ctx) -> np.ndarray:
    """The per-cell farmland null (LOS mm/yr) the report tests a cell against."""
    from . import client_report as cr
    try:
        model = json.loads((Path(cfg.project_dir) / "deliverables" / "uncertainty_model.json").read_text())
    except (OSError, ValueError):
        model = {}
    return cr.cell_sigma_map(rep.get("cell_null_by_class"), ctx.kcls, model)


# ------------------------------------------------------------------ gather
def _far_sample(ctx, field_own: np.ndarray, kc: np.ndarray, n_cls: int, per_class: int = FAR_SAMPLE_PER_CLASS,
                seed: int = 0) -> np.ndarray:
    """Flat indices of the farmland cells an untested date window measures
    its noise on, beyond the field points that are farmland themselves: per
    coherence class, enough of the report's farmland cells (a fixed-seed
    choice, sorted) that the class has `per_class` with the field's own -
    or all of them when it has fewer. kc: each cell's _slot."""
    far = ctx.far.ravel()
    have = np.zeros(n_cls + 1, np.int64)
    np.add.at(have, kc[field_own[far[field_own]]], 1)
    taken = np.zeros(far.size, bool)
    taken[field_own] = True
    rng = np.random.default_rng(seed)
    out = []
    for j in range(n_cls):
        need = per_class - int(have[j])
        if need <= 0:
            continue
        cand = np.flatnonzero(far & (kc == j) & ~taken)
        if cand.size > need:
            cand = np.sort(rng.choice(cand, need, replace=False))
        out.append(cand)
    return np.concatenate(out).astype(np.int64) if out else np.zeros(0, np.int64)


def slopes(t: np.ndarray, Y: np.ndarray, min_n: int = 6) -> np.ndarray:
    """Each row's least-squares slope over its finite values, NaN with fewer
    than min_n (results._fit_rates' rule, MIN_DATES), in float64. A NaN
    in t leaves that image out (fit_t: a date MintPy left out of the fit)."""
    t = np.asarray(t, float)
    ok = np.isfinite(Y) & np.isfinite(t)[None, :]
    n = ok.sum(axis=1).astype(float)
    T = np.where(ok, t[None, :], 0.0)
    Yz = np.where(ok, Y, 0.0).astype(float)
    st, sy, stt, sty = T.sum(1), Yz.sum(1), (T * T).sum(1), (T * Yz).sum(1)
    den = n * stt - st * st
    with np.errstate(invalid="ignore", divide="ignore"):
        b = (n * sty - st * sy) / den
    b[(n < min_n) | (den <= 0)] = np.nan
    return b


def mintpy_years(dates) -> np.ndarray:
    """MintPy's decimal years (mintpy.utils.ptime.yyyymmdd2years: year +
    (day of year - 1) / 365.25), the axis its velocity is fitted on."""
    from datetime import datetime
    out = []
    for d in dates:
        x = datetime.strptime(str(d)[:8], "%Y%m%d")
        out.append(x.year + (x.timetuple().tm_yday - 1) / 365.25)
    return np.asarray(out, float)


def fit_axis(cfg, dates) -> tuple[np.ndarray, list]:
    """(time axis from the first date, dates left out): what MintPy fitted
    the delivered velocity on - its own decimal years, without the dates it
    judged too noisy (mintpy/exclude_date.txt, its timeFunc.excludeDate).

    The whole-record check (does a table's own-cell series give the
    delivered rate back?) fits this way. Until 26 Oct 2026 it fitted on
    significance.years over every date: Site E (20250510 left out by
    MintPy) missed by up to 0.054 mm/yr and Site E NISAR (10 months:
    the two axes differ by 0.1% of a rate) by 0.13, and the viewer turned
    date windows off. A window's own figures stay results.analyse's (t,
    every date), which the viewer is held to point by point."""
    t = mintpy_years(dates)
    t = t - t[0] if t.size else t
    try:
        listed = {x.strip()[:8] for x in (Path(cfg.mintpy_dir) / "exclude_date.txt").read_text().split() if x.strip()}
    except OSError:
        listed = set()
    return t, [d for d in dates if d in listed]


def fit_t(meta: dict) -> np.ndarray:
    """A table's rate-fit axis: t_fit with NaN at its fit_exclude dates; a
    table from before 26 Oct 2026, its t over every date."""
    t = np.asarray(meta.get("t_fit") or meta["t"], float).copy()
    ex = set(meta.get("fit_exclude") or [])
    if ex:
        t[[i for i, d in enumerate(meta["dates"]) if d in ex]] = np.nan
    return t


def add_fit_axis(cfg, fd: dict) -> dict:
    """Give a table built before 26 Oct 2026 its fit axis (fit_axis) and
    redo its whole-record check on it, in place; a newer table is left as
    it is. The results viewer calls it on load, so such a table needs no
    rebuild."""
    m = fd["meta"]
    if "t_fit" in m or "cell_los" not in fd or not m.get("dates"):
        return fd
    tf, ex = fit_axis(cfg, list(m["dates"]))
    m["t_fit"], m["fit_exclude"] = [float(x) for x in tf], ex
    rc, cc = whole_record_check(fit_t(m), np.asarray(fd["cell_los"], float), np.asarray(fd["los_rel"], float),
                                np.asarray(fd["change_los"], float))
    m["whole_record_check_mm_yr"], m["whole_record_change_check_mm"] = rc, cc
    return fd


def whole_record_check(t, C, los_rel, change_los) -> tuple[float | None, float | None]:
    """(largest |slope of the own-cell series - stored rate| in mm/yr,
    largest |last - first image of it - stored change| in mm) over the points
    where both are finite; None when no point has both."""
    t = np.asarray(t, float)
    b = slopes(t, C) if C.size else np.zeros(0)
    d = C[:, -1].astype(float) - C[:, 0].astype(float) if C.size else np.zeros(0)
    with np.errstate(invalid="ignore"):
        e1 = np.abs(b - los_rel)
        e2 = np.abs(d - change_los)
    e1, e2 = e1[np.isfinite(e1)], e2[np.isfinite(e2)]
    return (float(e1.max()) if e1.size else None), (float(e2.max()) if e2.size else None)


def _read_flag(paths, shape):
    import rasterio
    for p in paths:
        if Path(p).exists():
            try:
                with rasterio.open(p) as r:
                    a = r.read(1)
                if a.shape == shape:
                    return a == 1
            except Exception:                                 # noqa: BLE001
                pass
    return None


def cell_maps(cfg, rep: dict, ctx, *, up_map=None, null_map=None, change=None) -> dict:
    """The per-cell maps a cell's whole-record figures are read from (gather,
    and the results viewer's 15 m cells): the LOS up component, the report's
    per-cell null, the relative change, the unwrapping consistency and the
    too-fast flag, with the regional GNSS rate when the report is anchored."""
    from . import client_report as cr
    from . import deliverables as dl
    D = Path(cfg.project_dir) / "deliverables"
    anchored = bool(rep.get("anchored"))
    bg = ((rep.get("gnss") or {}).get("background") or {}) if anchored else {}
    um, _ = dl.unwrap_consistency_grid(cfg, ctx.good.shape)
    return {"anchored": anchored, "bg_v": float(bg.get("vertical_mm_yr") or 0.0) if anchored else 0.0,
            "bg_u": bg.get("vertical_uncertainty_mm_yr") if anchored else None,
            "up": cr._los_up(cfg, ctx.vel.shape) if up_map is None else up_map,
            "null": report_null(cfg, rep, ctx) if null_map is None else null_map,
            "change": relative_change(ctx) if change is None else change, "um": um,
            "fast": _read_flag([D / "quality_too_fast_to_measure.tif",
                                D / "technical" / "quality_too_fast_to_measure.tif"], ctx.good.shape)}


def cell_figures(ctx, maps: dict, rows, cols) -> dict:
    """Each cell's whole-record figures from cell_maps, by the rule of
    rate_signal_to_noise.tif and the client rasters: what a point reads in
    the client KMZ, the point table and the viewer."""
    rows, cols = np.asarray(rows, int), np.asarray(cols, int)
    a = ctx.attrs
    y0, x0, dy, dx = (float(a[k]) for k in ("Y_FIRST", "X_FIRST", "Y_STEP", "X_STEP"))
    up = maps["up"][rows, cols].astype(float)
    los_rel = ctx.r_cm[rows, cols].astype(float)
    null = maps["null"][rows, cols].astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        rel_v = los_rel / up
        sd_v = np.where(np.isfinite(null) & (null > 0), null / up, np.nan)
        snr = np.where(np.isfinite(null) & (null > 0) & np.isfinite(los_rel), np.abs(los_rel) / null, np.nan)
    um, fast = maps["um"], maps["fast"]
    return {"row": rows.astype(np.int32), "col": cols.astype(np.int32),
            "lat": y0 + (rows + 0.5) * dy, "lon": x0 + (cols + 0.5) * dx,
            "up": up, "los_rel": los_rel, "rel_v": rel_v, "rate": rel_v + maps["bg_v"], "sd_v": sd_v, "snr": snr,
            "tc": ctx.tc[rows, cols].astype(float),
            "unwrap": um[rows, cols].astype(float) if um is not None else np.full(rows.size, np.nan),
            "fast": fast[rows, cols] if fast is not None else np.zeros(rows.size, bool),
            "change_los": maps["change"][rows, cols].astype(float)}


def cells_from_frames(ctx, maps: dict, rows, cols, frame_los, meta: dict) -> dict:
    """gather's table for a few cells without its pass over the whole cube
    (the results viewer's click on a 15 m cell): the figures from cell_maps,
    and each cell's own series and its 3 x 3 cells' from the cube at those
    cells only, less the farmland frame of each cell's class at each image -
    the frames gather measured over the whole cube and stored in the table
    (frame_los). The same numbers as gather's (tested); no farmland sample,
    so no date windows."""
    import h5py
    rows, cols = np.asarray(rows, int), np.asarray(cols, int)
    fig = cell_figures(ctx, maps, rows, cols)
    setup = _class_setup(ctx)
    FR = np.vstack([np.asarray(frame_los, float), np.full((1, np.shape(frame_los)[1]), np.nan)])   # + the NaN slot
    kc = _slot(ctx.kcls, setup)
    idx, valid = _neighbours(rows, cols, ctx.good)
    ny, nx = ctx.good.shape
    T = FR.shape[1]
    P = np.full((rows.size, T), np.nan, dtype=np.float32)
    C = np.full((rows.size, T), np.nan, dtype=np.float32)
    with h5py.File(ctx.ts_file, "r") as f:
        ds = f["timeseries"]
        for j, (r, c) in enumerate(zip(rows, cols)):
            r0, r1, c0, c1 = max(r - 1, 0), min(r + 2, ny), max(c - 1, 0), min(c + 2, nx)
            blk = ds[:, r0:r1, c0:c1].astype(float) * 1000.0          # (T, <=3, <=3)
            nb_r, nb_c = idx[j] // nx, idx[j] % nx
            vals = blk[:, nb_r - r0, nb_c - c0] - FR[kc[nb_r, nb_c]].T  # (T, 9)
            ok = valid[j][None, :] & np.isfinite(vals)
            with np.errstate(invalid="ignore", divide="ignore"):
                P[j] = np.where(ok.any(1), np.where(ok, vals, 0.0).sum(1) / ok.sum(1), np.nan)
            C[j] = blk[:, r - r0, c - c0] - FR[kc[r, c]]
    return {"group": np.zeros(rows.size, np.int16), **fig, "series_los": P, "cell_los": C,
            "kcls": ctx.kcls[rows, cols].astype(np.int8), "meta": dict(meta)}


def gather(cfg, rep: dict, ctx, cells: list[tuple[np.ndarray, np.ndarray]], *, up_map=None, null_map=None,
           change=None) -> dict:
    """Every figure for the given cells, in one pass over the cube.

    cells: a list of (rows, cols) groups (the field grid, each feature's
    points); `group` in the result says which. up_map / null_map / change
    are reused when the caller has them (client_kmz)."""
    import h5py
    from . import deliverables as dl
    D = Path(cfg.project_dir) / "deliverables"
    rows = np.concatenate([np.asarray(r, int) for r, _ in cells]) if cells else np.zeros(0, int)
    cols = np.concatenate([np.asarray(c, int) for _, c in cells]) if cells else np.zeros(0, int)
    group = np.concatenate([np.full(len(r), i, np.int16) for i, (r, _) in enumerate(cells)]) if cells \
        else np.zeros(0, np.int16)
    maps = cell_maps(cfg, rep, ctx, up_map=up_map, null_map=null_map, change=change)
    anchored, bg_v, bg_u = maps["anchored"], maps["bg_v"], maps["bg_u"]
    fig = cell_figures(ctx, maps, rows, cols)
    los_rel = fig["los_rel"]
    # ---- one pass over the cube: each point's own cell and its 3 x 3 cells,
    # and the farmland sample, each against farmland of the cell's own class
    idx, valid = _neighbours(rows, cols, ctx.good)
    setup = _class_setup(ctx)
    n_cls = setup[4]
    kc = _slot(ctx.kcls.ravel(), setup)
    cls = kc[idx]
    ny, nx = ctx.good.shape
    own = rows.astype(np.int64) * nx + cols
    k_own = kc[own]
    far_all = np.flatnonzero(ctx.far.ravel())
    fs_idx = _far_sample(ctx, own[group == 0], kc, n_cls)
    k_fs = kc[fs_idx]
    n_ep = len(ctx.dates)
    P = np.full((rows.size, n_ep), np.nan, dtype=np.float32)
    C = np.full((rows.size, n_ep), np.nan, dtype=np.float32)
    FS = np.full((fs_idx.size, n_ep), np.nan, dtype=np.float32)
    FR = np.full((n_cls, n_ep), np.nan, dtype=np.float32)
    far_nan = np.zeros(far_all.size, bool)
    t = ctx.t
    if rows.size:
        with h5py.File(ctx.ts_file, "r") as f:
            for i, ep in ctx.sg.iter_epochs(f["timeseries"]):
                v = ep.ravel() * 1000.0
                fe = _frame(v, setup, float(t[i] - t[0]))
                vals = np.where(valid, v[idx] - fe[cls], 0.0)
                ok = valid & np.isfinite(vals)
                with np.errstate(invalid="ignore", divide="ignore"):
                    P[:, i] = np.where(ok.any(axis=1), np.where(ok, vals, 0.0).sum(axis=1) / ok.sum(axis=1), np.nan)
                C[:, i] = v[own] - fe[k_own]
                FS[:, i] = v[fs_idx] - fe[k_fs]
                FR[:, i] = fe[:n_cls]
                far_nan |= ~np.isfinite(v[far_all])
    try:
        from . import release
        nh = release.numbers_hash(rep)
    except Exception:                                         # noqa: BLE001
        nh = ""
    sig = rep.get("signal_to_noise") or {}
    try:
        snr_json = json.loads((D / "signal_to_noise.json").read_text())
    except (OSError, ValueError):
        snr_json = {}
    kcls_pt = ctx.kcls.ravel()[own]
    far_pt = ctx.far.ravel()[own]
    fcl = ctx.kcls.ravel()
    far_by_class = {str(j): int(((fcl == j) & ctx.far.ravel()).sum()) for j in range(n_cls)}
    try:
        model = json.loads((D / "uncertainty_model.json").read_text())
    except (OSError, ValueError):
        model = {}
    emp = {str(int(r["class_index"])): float(r["empirical_mm_yr"]) for r in model.get("by_class", [])
           if r.get("empirical_mm_yr") is not None}
    # the identity the viewer's date windows stand on (19 Oct 2026): over the
    # whole record each point's own-cell series has the stored rate as its
    # slope and the stored change as its last minus first image. The viewer
    # checks it again on load and turns window figures off if it fails
    t_fit, fit_exclude = fit_axis(cfg, list(ctx.dates))
    chk_rate, chk_change = whole_record_check(fit_t({"t": t, "t_fit": list(t_fit), "fit_exclude": fit_exclude,
                                                     "dates": list(ctx.dates)}),
                                              C, los_rel, fig["change_los"])
    return {
        "group": group, **fig,
        "series_los": P,
        "cell_los": C, "kcls": kcls_pt.astype(np.int8), "far": far_pt.astype(bool),
        "far_row": (fs_idx // nx).astype(np.int32), "far_col": (fs_idx % nx).astype(np.int32),
        "far_cls": ctx.kcls.ravel()[fs_idx].astype(np.int8), "far_los": FS, "frame_los": FR,
        "meta": {"table_version": TABLE_VERSION,
                 "class_measured": [int(k) for k in setup[0]],
                 "class_lender": {str(i): int(j) for i, j in setup[3].items()},
                 "class_tilt": {str(int(k)): float(v) for k, v in (setup[2] or {}).items()
                                if isinstance(k, (int, np.integer))},
                 "emp_by_class": emp, "rep_cell_null_by_class": rep.get("cell_null_by_class"),
                 "far_cells_by_class": far_by_class, "far_partial_nan_cells": int(far_nan.sum()),
                 "far_sample_per_class": FAR_SAMPLE_PER_CLASS, "min_spread_cells": MIN_SPREAD_CELLS,
                 "whole_record_check_mm_yr": chk_rate, "whole_record_change_check_mm": chk_change,
                 "dates": list(ctx.dates), "t": [float(x) for x in t], "span": float(ctx.span),
                 "t_fit": [float(x) for x in t_fit], "fit_exclude": fit_exclude,
                 "anchored": anchored, "bg_v": bg_v, "bg_u": None if bg_u is None else float(bg_u),
                 "numbers_hash": nh, "gate": float(dl.second_gate_level(cfg)),
                 "resolution": rep.get("resolution"), "ground": rep.get("control_ground_label") or "farmland",
                 "snr_chance_pct": snr_json.get("percent_of_farmland_standing_out_at_1_96_by_chance",
                                                sig.get("percent_of_farmland_standing_out_at_1_96_by_chance")),
                 "title": rep.get("title") or Path(cfg.project_dir).name, "project_dir": str(cfg.project_dir)},
    }


def series_vertical_in(fd: dict, j: int) -> np.ndarray:
    """Point j's chart line (in): its 3 x 3 relative series / los_up, zeroed
    on its own trend (not on one noisy image), + the regional GNSS motion
    when the report is anchored - what client_kmz drew per point."""
    t = np.asarray(fd["meta"]["t"])
    rel = fd["series_los"][j].astype(float) / fd["up"][j]
    ok = np.isfinite(rel)
    if ok.sum() > 2:
        rel = rel - np.polyfit(t[ok], rel[ok], 1)[1]
    return (rel + fd["meta"]["bg_v"] * t) / MM_IN


def series_los_in(fd: dict, j: int) -> np.ndarray:
    """Point j's line-of-sight series (in), relative, zeroed on its own trend."""
    t = np.asarray(fd["meta"]["t"])
    s = fd["series_los"][j].astype(float)
    ok = np.isfinite(s)
    if ok.sum() > 2:
        s = s - np.polyfit(t[ok], s[ok], 1)[1]
    return s / MM_IN


def _detrend_intercepts(t: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Each row's least-squares intercept over its finite values (NaN with
    fewer than 3): np.polyfit(t, y, 1)[1] row by row, vectorised."""
    ok = np.isfinite(Y)
    n = ok.sum(axis=1).astype(float)
    T = np.where(ok, t[None, :], 0.0)
    Yz = np.where(ok, Y, 0.0)
    st, sy, stt, sty = T.sum(1), Yz.sum(1), (T * T).sum(1), (T * Yz).sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        den = n * stt - st * st
        slope = (n * sty - st * sy) / den
        b = (sy - slope * st) / n
    b[n < 3] = 0.0                                            # as the single-point charts: not zeroed
    return b


def series_vertical_in_rows(fd: dict, idx: np.ndarray) -> np.ndarray:
    """series_vertical_in for many points at once (rows of idx)."""
    t = np.asarray(fd["meta"]["t"])
    rel = fd["series_los"][idx].astype(float) / fd["up"][idx][:, None]
    rel = rel - _detrend_intercepts(t, rel)[:, None]
    return (rel + fd["meta"]["bg_v"] * t[None, :]) / MM_IN


def series_los_in_rows(fd: dict, idx: np.ndarray) -> np.ndarray:
    t = np.asarray(fd["meta"]["t"])
    s = fd["series_los"][idx].astype(float)
    return (s - _detrend_intercepts(t, s)[:, None]) / MM_IN


# ------------------------------------------------------------------ the viewer's copy
ARRAYS = ("row", "col", "group", "lat", "lon", "up", "los_rel", "rel_v", "rate", "sd_v", "snr", "tc", "unwrap",
          "fast", "change_los", "series_los",
          # table version 2 (19 Oct 2026); a version-1 table loads without them
          "cell_los", "kcls", "far", "far_row", "far_col", "far_cls", "far_los", "frame_los")


def save(fd: dict, path: Path, extra: dict | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = dict(fd["meta"], **(extra or {}))
    part = path.with_name(path.name + ".partial.npz")
    np.savez_compressed(part, meta=np.array(json.dumps(meta, default=str)),
                        **{k: fd[k] for k in ARRAYS})
    part.replace(path)
    return path


# a combined project's table (combine_viewer) carries east-west beside the vertical
COMBINED_ARRAYS = ("east", "east_change", "series_east")


def load(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        fd = {k: z[k] for k in ARRAYS + COMBINED_ARRAYS if k in z.files}
        fd["meta"] = json.loads(str(z["meta"]))
    return fd


def viewer_path(cfg, out_dir: Path | None = None) -> Path:
    return Path(out_dir or (Path(cfg.project_dir) / "deliverables")) / "viewer" / VIEWER_FILE


def main(argv=None) -> int:
    """The viewer copy without rebuilding the KMZ (the KMZ build writes it too)."""
    import argparse
    import sys
    from .config import load_config
    from . import client_kmz
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(prog="python -m insar_pipeline.field_points")
    ap.add_argument("config")
    a = ap.parse_args(argv if argv is not None else sys.argv[1:])
    print(client_kmz.build(load_config(a.config), viewer_only=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
