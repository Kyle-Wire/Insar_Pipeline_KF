"""C-band against L-band cycle consistency (F4, review_2026-10-08/PLAN.md phase 3).

WHY
---
Where a C-band interferogram cannot follow the ground - a longwall panel
moving decimetres between two passes, faster than a quarter wavelength per
look cell - the unwrapper still returns a smooth answer, but one short by a
whole number of phase cycles. Each cycle is half a wavelength of line of
sight: 27.7 mm for Sentinel-1. L-band's cycle is ~122 mm (NISAR), so over the
same months it follows ground that C-band cannot. Comparing the two over
matching intervals, differences that cluster at k x 27.7 mm (k = +-1, +-2 ...)
are the fingerprint of C-band aliasing; differences near 0 are agreement;
anything else is noise, geometry or an L-band problem. L-band arbitrates
because aliasing is about 4.4 times less likely there - it is not truth, and
its own ionosphere residual is larger.

WHAT IT DOES
------------
For each interval between two L-band epochs that have C-band epochs within
`tol_days` of both ends (and for the whole common window):

  1. the C-band change and the L-band change over the interval, each from its
     own delivered time series (deliverables.timeseries_file);
  2. C-band put on the L-band grid (median of the C cells in each L cell);
  3. the L-band change PREDICTED in C-band's line of sight: with one L-band
     geometry, assuming vertical motion (d_C = d_L x up_C / up_L, per cell -
     the two sensors look from different sides, so a horizontal component is
     NOT cancelled and shows as non-integer differences); with decomposed L-band
     Up and East (D1 on two NISAR geometries), d_C = east_C x E + up_C x U;
  4. the difference, less its median over the common cells (the two series
     have unrelated reference pixels), optionally less a plane. The C epochs
     may sit up to tol_days from the L epochs at each end; motion in those
     days is a real difference, so each interval records end_offsets_days;
  5. every cell classed:
       agree        |d| < agree_mm (15 mm: PLAN 5.3's NISAR row);
       whole cycle  not agree, |k| >= 1 and |d - k x cycle| <= tol_mm
                    (5 mm: protocol_B1's "within 5 mm of a non-zero
                    multiple of 27.7 mm");
       other        the rest.
     The chance level matters: a difference spread at random over cycles
     lands within +-5 mm of a whole cycle 36% of the time, so the record
     reports the excess over that and a binomial z (cells are spatially
     correlated, so z overstates the evidence - read it as a ranking, not a
     p-value). Checked in review (8 Oct 2026): for zero-mean symmetric noise
     with no aliasing (Gaussian, Laplace or Student-t3, sd 5-100 mm) the
     whole-cycle share among disagreeing cells is 0.00-0.37, never more
     than ~0.01 above 0.36, so an excess below ~0.01 is not evidence of
     aliasing however large z is (noise_null_ceiling, in every record).

Two frames, both reported (protocol amendment 2026-10-08/p23-5, A, B1 and
B2, 8 Oct 2026, before any result):

  - the CYCLE-CONSISTENCY (aliasing) test is made in C-BAND LINE OF SIGHT: a
    missed cycle is 27.7 mm of C-band LOS (27.7 / up_C, ~35 mm at 38 deg
    incidence, in Up), so the L-band change is projected into C-band's LOS
    under the stated vertical-motion assumption (or from D1's L-band Up and
    East when given), and the whole-cycle classes are read there;
  - the ACCURACY comparison (share of cells agreeing within 15 mm, the
    spread of the difference) is made in VERTICAL: each sensor's own LOS
    change divided by cos of its own incidence (los_up), both labelled
    'assumes vertical motion'. With different look sides and incidences a
    horizontal component does not cancel, so it shows in this row too.

protocol_B1 comparisons.nisar.against_sentinel1 worded its statistic "in Up"
with "a non-zero multiple of 27.7 mm", which mixes the two frames; the
amendment separates them as above.

The aliasing map (lband_whole_cycle_count.tif) counts, per L-band cell, the
consecutive intervals that showed a whole-cycle difference (band 2: the
intervals compared); the whole-window interval is in the JSON only.

USE
---
    python -m insar_pipeline.lband CFG_C CFG_L --out DIR [--tol-days 6]
        [--agree-mm 15] [--tol-mm 5] [--plane] [--l-up FILE --l-east FILE]

CFG_C is a Sentinel-1 run, CFG_L a NISAR run over the same ground; both must
have finished MintPy. Writes DIR/lband_cycle_consistency.json and the map.
Nothing in a client project reads this module.
"""
from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

AGREE_MM = 15.0          # PLAN 5.3, NISAR: "differences either below 15 mm or explained by whole C-band cycles"
# protocol_B1.yaml comparisons.nisar.against_sentinel1 (fixed 8 Oct 2026):
# "share whose difference lies within 5 mm of a non-zero multiple of 27.7 mm"
TOL_MM = 5.0
TOL_DAYS = 6             # half a 12-day repeat: the C epoch nearest each L epoch


# ---------------------------------------------------------------- grids
@dataclass(frozen=True)
class Grid:
    """A MintPy geographic grid: X_FIRST/Y_FIRST are the outer corner of cell (0, 0)."""
    x_first: float
    y_first: float
    dx: float
    dy: float
    nx: int
    ny: int

    @classmethod
    def from_attrs(cls, attrs, shape=None):
        ny, nx = shape if shape is not None else (int(attrs["LENGTH"]), int(attrs["WIDTH"]))
        g = cls(float(attrs["X_FIRST"]), float(attrs["Y_FIRST"]), float(attrs["X_STEP"]),
                float(attrs["Y_STEP"]), int(nx), int(ny))
        if abs(g.dx) > 0.1 or abs(g.dy) > 0.1:
            raise ValueError("lband compares geographic (lat/lon) grids; this one is projected "
                             f"(X_STEP {g.dx}): the NISAR route writes lat/lon since 8 Oct 2026")
        return g

    def centres(self):
        return (self.x_first + (np.arange(self.nx) + 0.5) * self.dx,
                self.y_first + (np.arange(self.ny) + 0.5) * self.dy)


def resample_to(src, src_grid: Grid, dst_grid: Grid, how: str = "median"):
    """src (on src_grid) on dst_grid: the median (or mean) of the src cells
    whose centres fall in each dst cell. For C-band at 15 m onto L-band at
    80 m that is the ~25 cells inside each L cell. A dst cell that holds no
    finite src cell is NaN. Coarser src than dst: nearest src cell."""
    from scipy import ndimage
    src = np.asarray(src, dtype="float64")
    xs, ys = src_grid.centres()
    col = np.floor((xs - dst_grid.x_first) / dst_grid.dx).astype(np.int64)
    row = np.floor((ys - dst_grid.y_first) / dst_grid.dy).astype(np.int64)
    out = np.full((dst_grid.ny, dst_grid.nx), np.nan)
    if abs(src_grid.dx) >= abs(dst_grid.dx) and abs(src_grid.dy) >= abs(dst_grid.dy):
        xd, yd = dst_grid.centres()
        c = np.floor((xd - src_grid.x_first) / src_grid.dx).astype(np.int64)
        r = np.floor((yd - src_grid.y_first) / src_grid.dy).astype(np.int64)
        okc, okr = (c >= 0) & (c < src_grid.nx), (r >= 0) & (r < src_grid.ny)
        sub = src[np.ix_(np.where(okr, r, 0), np.where(okc, c, 0))]
        sub[~np.outer(okr, okc)] = np.nan
        return sub
    R, C = np.meshgrid(row, col, indexing="ij")
    ok = (R >= 0) & (R < dst_grid.ny) & (C >= 0) & (C < dst_grid.nx) & np.isfinite(src)
    if not ok.any():
        return out
    lab = (R[ok] * dst_grid.nx + C[ok]) + 1
    vals = src[ok]
    idx = np.unique(lab)
    agg = ndimage.median(vals, labels=lab, index=idx) if how == "median" else ndimage.mean(vals, labels=lab, index=idx)
    out.ravel()[idx - 1] = agg
    return out


# ---------------------------------------------------------------- the test
def classify_differences(diff_mm, cycle_mm: float, agree_mm: float = AGREE_MM, tol_mm: float = TOL_MM) -> dict:
    """Class of every cell: 1 agree, 2 whole cycle, 3 other, 0 no data; with
    the shares, k counts, the chance level and the excess over it.

    agree: |d| < agree_mm. whole cycle: not agree, |k| >= 1 and |d - k x
    cycle| <= tol_mm (protocol_B1: within 5 mm of a non-zero multiple of the
    C-band cycle). Chance level: a difference spread at random over many
    cycles lands within +-tol_mm of a whole cycle 2 x tol_mm / cycle of the
    time (36% for 5 mm at 27.7 mm), so the share among DISAGREEING cells is
    read against that, with a binomial z (cells are spatially correlated, so
    z overstates the evidence: read it as a ranking, not a p-value)."""
    d = np.asarray(diff_mm, dtype="float64")
    fin = np.isfinite(d)
    k = np.where(fin, np.rint(d / cycle_mm), 0)
    r = np.where(fin, d - k * cycle_mm, np.nan)
    agree = fin & (np.abs(d) < agree_mm)
    whole = fin & ~agree & (np.abs(k) >= 1) & (np.abs(r) <= tol_mm)
    other = fin & ~agree & ~whole
    cls = np.zeros(d.shape, dtype=np.uint8)
    cls[agree], cls[whole], cls[other] = 1, 2, 3
    chance = min(1.0, 2.0 * tol_mm / cycle_mm)
    n = int(fin.sum())
    n_dis = int((whole | other).sum())
    share_whole_of_dis = (whole.sum() / n_dis) if n_dis else float("nan")
    z = ((whole.sum() - n_dis * chance) / math.sqrt(n_dis * chance * (1 - chance))) \
        if n_dis and 0 < chance < 1 else float("nan")
    ks = {}
    for kk in np.unique(k[whole]).astype(int).tolist():
        ks[str(kk)] = int(((k == kk) & whole).sum())
    return {
        "classes": cls, "k": k.astype(np.int16),
        "cells": n,
        "share_agree": round(float(agree.sum()) / n, 4) if n else None,
        "share_whole_cycle": round(float(whole.sum()) / n, 4) if n else None,
        "share_other": round(float(other.sum()) / n, 4) if n else None,
        "disagreeing_cells": n_dis,
        "whole_cycle_share_of_disagreeing": round(float(share_whole_of_dis), 4) if n_dis else None,
        "chance_share": round(chance, 3),
        "excess_over_chance": round(float(share_whole_of_dis) - chance, 4) if n_dis else None,
        "binomial_z": round(float(z), 2) if n_dis and np.isfinite(z) else None,
        "k_counts": ks,
        "cycle_mm": round(float(cycle_mm), 3), "agree_mm": agree_mm, "tol_mm": tol_mm,
    }


def noise_null_ceiling(cycle_mm: float, agree_mm: float = AGREE_MM, tol_mm: float = TOL_MM) -> float:
    """The highest whole-cycle share among DISAGREEING cells that zero-mean
    symmetric noise gives with no aliasing at all: the maximum over Gaussian,
    Laplace and Student-t (3 dof) differences with standard deviations of
    5-100 mm (review, 8 Oct 2026). With the defaults it is 0.370 against the
    0.361 chance level, so an excess over chance below the gap is noise
    however large the binomial z."""
    from scipy import stats
    ks = np.arange(1, int(np.ceil(1000.0 / cycle_mm)) + 1)
    best = 0.0
    for sd in np.arange(5.0, 101.0, 1.0):
        for dist in (stats.norm(0, sd), stats.laplace(0, sd / math.sqrt(2)), stats.t(3, 0, sd / math.sqrt(3))):
            p_dis = 2 * dist.sf(agree_mm)
            if p_dis <= 0:
                continue
            lo = np.maximum(ks * cycle_mm - tol_mm, agree_mm)
            hi = ks * cycle_mm + tol_mm
            p_whole = float(np.sum(2 * np.clip(dist.cdf(hi) - dist.cdf(lo), 0, None)))
            best = max(best, p_whole / p_dis)
    return round(best, 3)


def match_intervals(dates_c, dates_l, tol_days: int = TOL_DAYS) -> list[dict]:
    """Intervals between consecutive L-band epochs (and the whole common
    window) whose two ends each have a C-band epoch within tol_days. Each:
    {'l': (i0, i1), 'c': (j0, j1), 'dates_l': ..., 'dates_c': ..., 'kind',
    'end_offsets_days'}."""
    def od(d):
        s = d.decode() if isinstance(d, bytes) else str(d)
        return datetime.strptime(s[:8], "%Y%m%d").toordinal()
    tc = np.array([od(d) for d in dates_c])
    tl = np.array([od(d) for d in dates_l])

    def nearest(t):
        j = int(np.argmin(np.abs(tc - t)))
        return j if abs(int(tc[j]) - int(t)) <= tol_days else None

    near = [nearest(t) for t in tl]
    out = []
    for i in range(len(tl) - 1):
        j0, j1 = near[i], near[i + 1]
        if j0 is not None and j1 is not None and j1 > j0:
            out.append({"kind": "consecutive", "l": (i, i + 1), "c": (j0, j1)})
    ok = [i for i, j in enumerate(near) if j is not None]
    if len(ok) >= 2 and near[ok[-1]] > near[ok[0]] and (ok[-1] - ok[0]) > 1:
        out.append({"kind": "whole window", "l": (ok[0], ok[-1]), "c": (near[ok[0]], near[ok[-1]])})
    for x in out:
        x["dates_l"] = [_s(dates_l[x["l"][0]]), _s(dates_l[x["l"][1]])]
        x["dates_c"] = [_s(dates_c[x["c"][0]]), _s(dates_c[x["c"][1]])]
        # C epoch minus L epoch at each end: ground moving in those days is a
        # REAL difference (a longwall face can move centimetres in a week)
        x["end_offsets_days"] = [int(tc[x["c"][0]] - tl[x["l"][0]]), int(tc[x["c"][1]] - tl[x["l"][1]])]
    return out


def _s(d) -> str:
    return (d.decode() if isinstance(d, bytes) else str(d))[:8]


def predict_c_los(l_los_mm=None, l_up=None, c_up=None, c_east=None, l_up_mm=None, l_east_mm=None):
    """The L-band change in C-band's line of sight. With decomposed L-band
    Up/East (mm): c_east x East + c_up x Up (north not estimated, as D1).
    Otherwise vertical motion assumed: l_los x c_up / l_up (per cell)."""
    if l_up_mm is not None:
        e = np.zeros_like(l_up_mm) if l_east_mm is None else l_east_mm
        return np.asarray(c_east) * e + np.asarray(c_up) * l_up_mm
    return np.asarray(l_los_mm) * np.asarray(c_up) / np.asarray(l_up)


def remove_datum(diff, plane: bool = False):
    """diff less its median (the two series share no reference pixel), or less
    a least-squares plane over the finite cells; returns (diff, what)."""
    d = np.asarray(diff, dtype="float64")
    fin = np.isfinite(d)
    if fin.sum() < 3:
        return d, {"removed": "nothing (too few cells)"}
    if not plane:
        m = float(np.median(d[fin]))
        return d - m, {"removed": "median", "offset_mm": round(m, 2)}
    rr, cc = np.nonzero(fin)
    A = np.column_stack([np.ones(rr.size), rr, cc]).astype("float64")
    coef, *_ = np.linalg.lstsq(A, d[fin], rcond=None)
    R, C = np.indices(d.shape)
    return d - (coef[0] + coef[1] * R + coef[2] * C), {"removed": "plane",
                                                       "coefficients": [round(float(c), 5) for c in coef]}


def cycle_consistency(c: dict, l: dict, tol_days: int = TOL_DAYS, agree_mm: float = AGREE_MM,
                      tol_mm: float = TOL_MM, plane: bool = False) -> dict:
    """The test on arrays. c and l: {'dates': [...], 'epoch': callable(i) ->
    2-D mm on its grid (or 'cube': (t, ny, nx) mm), 'grid': Grid,
    'wavelength_m', 'los_up' and 'los_east' (2-D or scalar, on the own grid),
    optional 'mask' (2-D bool, reliable)}. l may carry 'up_mm'/'east_mm'
    epoch callables from a decomposition instead of its LOS. Returns the
    record and the per-L-cell whole-cycle count."""
    from . import stack_qc
    cycle = stack_qc.half_wavelength_mm({"WAVELENGTH": c["wavelength_m"]})
    ep_c = c.get("epoch") or (lambda i: c["cube"][i])
    ep_l = l.get("epoch") or (lambda i: l["cube"][i])
    gl = l["grid"]

    def on_l(a):
        return resample_to(a, c["grid"], gl) if c["grid"] != gl else np.asarray(a, dtype="float64")

    c_up = on_l(np.broadcast_to(np.asarray(c["los_up"], dtype="float64"), (c["grid"].ny, c["grid"].nx)))
    c_east = on_l(np.broadcast_to(np.asarray(c.get("los_east", 0.0), dtype="float64"), (c["grid"].ny, c["grid"].nx)))
    mask = np.ones((gl.ny, gl.nx), bool)
    if c.get("mask") is not None:
        mask &= on_l(np.asarray(c["mask"], dtype="float64")) >= 0.5
    if l.get("mask") is not None:
        mask &= np.asarray(l["mask"], bool)
    intervals = match_intervals(c["dates"], l["dates"], tol_days)
    count = np.zeros((gl.ny, gl.nx), dtype=np.int16)
    seen = np.zeros((gl.ny, gl.nx), dtype=np.int16)
    rows = []
    for iv in intervals:
        (i0, i1), (j0, j1) = iv["l"], iv["c"]
        dc = on_l(ep_c(j1) - ep_c(j0))
        if l.get("up_mm") is not None:
            pred = predict_c_los(c_up=c_up, c_east=c_east, l_up_mm=l["up_mm"](i1) - l["up_mm"](i0),
                                 l_east_mm=(l["east_mm"](i1) - l["east_mm"](i0)) if l.get("east_mm") else None)
            how = "decomposed L-band Up/East projected on C-band's line of sight"
        else:
            pred = predict_c_los(ep_l(i1) - ep_l(i0), l_up=l["los_up"], c_up=c_up)
            how = "L-band LOS scaled to C-band's LOS assuming vertical motion"
        d = np.where(mask, dc - pred, np.nan)
        d, datum = remove_datum(d, plane=plane)
        res = classify_differences(d, cycle, agree_mm, tol_mm)
        cls = res.pop("classes")
        res.pop("k")
        if iv["kind"] == "consecutive":          # the whole window is reported on its own, not counted twice
            count += (cls == 2)
            seen += (cls > 0)
        # accuracy in vertical (amendment 2026-10-08/p23-5): each sensor's LOS / cos(its own incidence)
        acc = vertical_accuracy(dc, ep_l(i1) - ep_l(i0), c_up, l["los_up"], mask, agree_mm, plane)
        rows.append({**{k: v for k, v in iv.items() if k in ("kind", "dates_l", "dates_c", "end_offsets_days")},
                     "frame": "C-band line of sight (the cycle-consistency test)",
                     "comparison": how, "datum": datum, **res, "accuracy_vertical": acc})
    window = [r for r in rows if r["kind"] == "whole window"]
    cons = [r for r in rows if r["kind"] == "consecutive"]
    ceiling = noise_null_ceiling(cycle, agree_mm, tol_mm)
    summary = {
        "intervals": len(cons),
        "frames": {"cycle_consistency": "C-band LOS: the L-band change projected into C-band's LOS (vertical motion "
                                        "assumed, or D1's L-band Up/East)",
                   "accuracy": "vertical: each sensor's LOS / cos(its own incidence), both assuming vertical motion",
                   "amendment": "2026-10-08/p23-5"},
        "accuracy_vertical": {
            "median_share_agree": _med([(r.get("accuracy_vertical") or {}).get("share_agree") for r in cons]),
            "median_abs_difference_mm": _med([(r.get("accuracy_vertical") or {}).get("median_abs_mm") for r in cons]),
            "whole_window": (window[0].get("accuracy_vertical") if window else None),
            "label": VERTICAL_LABEL},
        "median_share_agree": _med([r["share_agree"] for r in cons]),
        "median_share_whole_cycle": _med([r["share_whole_cycle"] for r in cons]),
        "median_excess_over_chance": _med([r["excess_over_chance"] for r in cons]),
        "cells_with_any_whole_cycle_interval": int((count > 0).sum()),
        "cells_compared": int((seen > 0).sum()),
        "whole_window": window[0] if window else None,
        "reading": ("A whole-cycle share well above chance (excess > 0 with many disagreeing cells) where the "
                    "C-band record shows a smooth, shallow trough marks C-band aliasing there; an excess near 0 "
                    "means the disagreements are noise, geometry (horizontal motion breaks the vertical-motion "
                    "scaling) or L-band error, not missed cycles. Noise alone, with no aliasing, can give a "
                    f"whole-cycle share among disagreeing cells up to {ceiling}, so an excess over chance below "
                    f"{max(0.0, ceiling - round(min(1.0, 2.0 * tol_mm / cycle), 3)):.3f} is not evidence, however "
                    "large binomial_z is"),
        "noise_null_ceiling": ceiling,
    }
    return {"cycle_mm": round(cycle, 3), "tol_days": tol_days, "agree_mm": agree_mm, "tol_mm": tol_mm,
            "chance_share": round(min(1.0, 2.0 * tol_mm / cycle), 3), "summary": summary, "by_interval": rows,
            "count_map": count, "seen_map": seen}


def _med(v):
    v = [x for x in v if x is not None and np.isfinite(x)]
    return round(float(np.median(v)), 4) if v else None


VERTICAL_LABEL = "assumes vertical motion: each sensor's LOS change / cos(its own incidence)"


def vertical_accuracy(c_los_mm, l_los_mm, c_up, l_up, mask=None, agree_mm: float = AGREE_MM,
                      plane: bool = False) -> dict:
    """The accuracy comparison in vertical (amendment 2026-10-08/p23-5):
    C-band LOS / up_C against L-band LOS / up_L (up = cos(incidence), each
    sensor's own), on the L-band grid, less the median (or a plane: the two
    series have unrelated reference pixels). The share agreeing within
    agree_mm (PLAN 5.3's 15 mm), the median |difference| and its robust
    spread. Labelled 'assumes vertical motion'."""
    cu = np.asarray(c_up, dtype="float64")
    lu = np.asarray(l_up, dtype="float64")
    with np.errstate(invalid="ignore", divide="ignore"):
        vc = np.where(np.abs(cu) > 0.1, np.asarray(c_los_mm, dtype="float64") / cu, np.nan)
        vl = np.where(np.abs(lu) > 0.1, np.asarray(l_los_mm, dtype="float64") / lu, np.nan)
    d = vc - vl
    if mask is not None:
        d = np.where(np.asarray(mask, bool), d, np.nan)
    d, datum = remove_datum(d, plane=plane)
    fin = np.isfinite(d)
    n = int(fin.sum())
    if not n:
        return {"cells": 0, "label": VERTICAL_LABEL, "datum": datum}
    a = np.abs(d[fin])
    med = float(np.median(d[fin]))
    return {"cells": n, "share_agree": round(float((a < agree_mm).mean()), 4), "agree_mm": agree_mm,
            "median_abs_mm": round(float(np.median(a)), 2),
            "robust_sd_mm": round(float(1.4826 * np.median(np.abs(d[fin] - med))), 2),
            "rms_mm": round(float(np.sqrt(np.mean(d[fin] ** 2))), 2),
            "label": VERTICAL_LABEL, "datum": datum}


# ---------------------------------------------------------------- products
def _series(cfg) -> dict:
    """A finished run's delivered time series as lband reads it (mm epochs on
    demand), its grid, wavelength, LOS vectors and coherence mask."""
    import h5py
    from . import deliverables, stack_qc
    ts = Path(deliverables.timeseries_file(cfg))
    with h5py.File(ts, "r") as f:
        attrs = {k: (v.decode() if isinstance(v, bytes) else v) for k, v in f.attrs.items()}
        dates = [d.decode() if isinstance(d, bytes) else str(d) for d in f["date"][()]]
        shape = f["timeseries"].shape[1:]
    grid = Grid.from_attrs(attrs, shape)

    def epoch(i):
        with h5py.File(ts, "r") as f:
            return f["timeseries"][i].astype("float64") * 1000.0

    up, east = _los(cfg, shape)
    return {"dates": dates, "epoch": epoch, "grid": grid, "file": str(ts),
            "wavelength_m": stack_qc.product_wavelength_m(attrs), "look_side": stack_qc.product_look_side(attrs),
            "los_up": up, "los_east": east, "mask": _reliable(cfg, shape, grid)}


def _los(cfg, shape):
    """Per-pixel (up, east) LOS components: deliverables.geometry_field, else
    MintPy's incidenceAngle/azimuthAngle in mintpy/inputs/geometryGeo.h5 (what
    prep_nisar writes; geometry_field does not read it)."""
    import h5py
    from . import deliverables
    up = deliverables.geometry_field(cfg, "los_up")
    east = deliverables.geometry_field(cfg, "los_east")
    if up is not None and up.shape == tuple(shape):
        return up.astype("float64"), (east.astype("float64") if east is not None else np.zeros(shape))
    for p in (Path(cfg.mintpy_dir) / "inputs" / "geometryGeo.h5", Path(cfg.mintpy_dir) / "geometryGeo.h5"):
        if p.exists():
            with h5py.File(p, "r") as f:
                if "incidenceAngle" in f and "azimuthAngle" in f and f["incidenceAngle"].shape == tuple(shape):
                    inc = np.radians(f["incidenceAngle"][()].astype("float64"))
                    az = np.radians(f["azimuthAngle"][()].astype("float64"))
                    inc[inc == 0] = np.nan
                    return np.cos(inc), -np.sin(inc) * np.sin(az)
    raise FileNotFoundError(f"no per-pixel geometry for {cfg.project_dir} on the series grid {shape}")


def _reliable(cfg, shape, grid: Grid | None = None):
    """Reliable cells: NISAR's coherence gate when present (nisar_backend:
    mean pair coherence >= the floor, protocol_B1's NISAR rule; the gate's
    lattice carries a margin around the AOI, so it is put on the series grid
    by its own transform), else MintPy's temporal-coherence mask, else
    everything. A chain network's temporal coherence is 1.0 by construction,
    so for NISAR only the gate means anything."""
    import h5py
    gate = Path(cfg.project_dir) / "nisar_gunw" / "coherence_gate.tif"
    if gate.exists():
        import rasterio
        with rasterio.open(gate) as src:
            g = src.read(2).astype("float64")
            t = src.transform
        if g.shape == tuple(shape) and grid is None:
            return g > 0.5
        if grid is not None:
            src_grid = Grid(float(t.c), float(t.f), float(t.a), float(t.e), g.shape[1], g.shape[0])
            on = resample_to(g, src_grid, grid)
            return np.nan_to_num(on, nan=0.0) > 0.5
    for p in (Path(cfg.mintpy_dir) / "geo" / "geo_maskTempCoh.h5", Path(cfg.mintpy_dir) / "maskTempCoh.h5"):
        if p.exists():
            with h5py.File(p, "r") as f:
                k = [x for x in f.keys() if f[x].shape == tuple(shape)]
                if k:
                    return f[k[0]][()].astype(bool)
    return None


def run(cfg_c, cfg_l, out_dir, tol_days=TOL_DAYS, agree_mm=AGREE_MM, tol_mm=TOL_MM, plane=False,
        l_up=None, l_east=None) -> dict:
    """Read both finished runs, run the test, write the JSON and the map."""
    import rasterio
    from rasterio.transform import from_origin
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    c, l = _series(cfg_c), _series(cfg_l)
    from . import stack_qc
    if stack_qc.sensor_band(c["wavelength_m"]) != "C":
        log.warning("The first run is not C-band (wavelength %.4f m): the whole-cycle test uses ITS cycle",
                    c["wavelength_m"])
    if l_up:
        l["up_mm"], l["east_mm"] = _decomposed(l_up, l["grid"]), (_decomposed(l_east, l["grid"]) if l_east else None)
    rec = cycle_consistency(c, l, tol_days, agree_mm, tol_mm, plane)
    count, seen = rec.pop("count_map"), rec.pop("seen_map")
    g = l["grid"]
    tif = out_dir / "lband_whole_cycle_count.tif"
    with rasterio.open(tif, "w", driver="GTiff", height=g.ny, width=g.nx, count=2, dtype="int16",
                       crs="EPSG:4326", transform=from_origin(g.x_first, g.y_first, g.dx, -g.dy)) as dst:
        dst.write(count, 1)
        dst.write(seen, 2)
        dst.set_band_description(1, "consecutive intervals with a whole C-band cycle difference")
        dst.set_band_description(2, "consecutive intervals compared")
    rec.update({"c_band": {"project": str(cfg_c.project_dir), "series": c["file"], "wavelength_m": c["wavelength_m"],
                           "look_side": c["look_side"]},
                "l_band": {"project": str(cfg_l.project_dir), "series": l["file"], "wavelength_m": l["wavelength_m"],
                           "look_side": l["look_side"], "decomposed": bool(l_up)},
                "map": str(tif), "written": datetime.now().strftime("%Y-%m-%d %H:%M:%S")})
    (out_dir / "lband_cycle_consistency.json").write_text(json.dumps(rec, indent=1, default=str))
    return rec


def _decomposed(path, grid: Grid):
    """Epoch reader for a D1 Up/East series (HDF5 'timeseries' in metres or
    'displacement_mm'), resampled to the L-band grid."""
    import h5py
    path = Path(path)
    with h5py.File(path, "r") as f:
        attrs = {k: (v.decode() if isinstance(v, bytes) else v) for k, v in f.attrs.items()}
        key = "timeseries" if "timeseries" in f else [k for k in f.keys() if f[k].ndim == 3][0]
        shape = f[key].shape[1:]
    src = Grid.from_attrs(attrs, shape)
    scale = 1.0 if "mm" in key else 1000.0

    def epoch(i):
        with h5py.File(path, "r") as f:
            return resample_to(f[key][i].astype("float64") * scale, src, grid)
    return epoch


def main(argv=None) -> int:
    import argparse
    from .config import load_config
    ap = argparse.ArgumentParser(description="C-band vs L-band cycle consistency (F4)")
    ap.add_argument("cfg_c", help="the Sentinel-1 (C-band) run's config")
    ap.add_argument("cfg_l", help="the NISAR (L-band) run's config")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tol-days", type=int, default=TOL_DAYS)
    ap.add_argument("--agree-mm", type=float, default=AGREE_MM)
    ap.add_argument("--tol-mm", type=float, default=TOL_MM, help="whole-cycle tolerance (protocol_B1: 5 mm)")
    ap.add_argument("--plane", action="store_true", help="remove a plane from each difference, not just its median")
    ap.add_argument("--l-up", help="decomposed L-band Up series (D1)")
    ap.add_argument("--l-east", help="decomposed L-band East series (D1)")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    rec = run(load_config(a.cfg_c), load_config(a.cfg_l), a.out, a.tol_days, a.agree_mm, a.tol_mm, a.plane,
              a.l_up, a.l_east)
    s = rec["summary"]
    print(f"{s['intervals']} intervals. Cycle consistency in C-band LOS: median shares agree {s['median_share_agree']}, "
          f"whole cycle {s['median_share_whole_cycle']} (excess over chance {s['median_excess_over_chance']}); "
          f"{s['cells_with_any_whole_cycle_interval']} of {s['cells_compared']} cells showed a whole C-band cycle. "
          f"Accuracy in vertical ({VERTICAL_LABEL}): median share within {a.agree_mm:g} mm "
          f"{s['accuracy_vertical']['median_share_agree']}, median |difference| "
          f"{s['accuracy_vertical']['median_abs_difference_mm']} mm")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
