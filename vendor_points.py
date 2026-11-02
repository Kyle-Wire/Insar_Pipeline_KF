"""
Like-for-like comparison with a vendor's point exports (21 Oct 2026).

A vendor point export (a "cumulative_displacement_Line_of_Sight" CSV) holds, per point,
point_id, longitude, latitude, local_X, local_Y, rate_in_per_year,
displacement_angle, displacement_in and one column per acquisition date
(inches LOS, negative = away from the satellite; blank = no value).

What the exports were measured to do (Site D, 1376 points, 21 Oct 2026):
  * the series is a running sum of single-interval increments, zeroed at the
    first column of the window the user picked (that column is blank);
  * an interval (previous column -> this column) adds its increment only when
    the point passes the vendor's quality test for it; a blank cell adds nothing and
    the level is carried flat, so movement during snow or crop seasons is
    never counted;
  * displacement_in is the last non-blank value;
  * rate_in_per_year = displacement_in / accepted time, where accepted time
    is the sum, over non-blank columns, of the days since the previous
    column, in years of 365.25 days (1373 of 1376 points to 0.0015 in/yr).

So the vendor's rate is a velocity WHILE OBSERVED, and its plotted series is a
gated sum. Neither compares with our continuous fit. This module puts our
series through the same bookkeeping (same points, same accepted intervals),
so the numbers can be set side by side.
"""
from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

IN_PER_M = 1.0 / 0.0254
YEAR_DAYS = 365.25
META_COLUMNS = ("point_id", "longitude", "latitude", "local_X", "local_Y", "rate_in_per_year",
                "displacement_angle", "displacement_in")


def _d(s: str) -> date:
    return date(int(s[:4]), int(s[4:6]), int(s[6:8]))


@dataclass
class Export:
    path: Path
    ids: list[str]
    lon: np.ndarray
    lat: np.ndarray
    rate: np.ndarray          # listed rate_in_per_year
    disp: np.ndarray          # listed displacement_in
    dates: list[str]          # YYYYMMDD, export order
    values: np.ndarray        # (points, dates) inches, NaN = blank

    @property
    def valid(self) -> np.ndarray:
        return np.isfinite(self.values)


def read_export(path) -> Export:
    """One vendor point CSV."""
    path = Path(path)
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.reader(f))
    head = rows[0]
    missing = [c for c in ("point_id", "longitude", "latitude", "rate_in_per_year", "displacement_in")
               if c not in head]
    if missing:
        raise ValueError(f"{path.name}: not a vendor point export (missing {', '.join(missing)})")
    dcols = [i for i, c in enumerate(head) if len(c) == 8 and c.isdigit()]
    if len(dcols) < 2:
        raise ValueError(f"{path.name}: fewer than two date columns")
    ix = {c: head.index(c) for c in head}

    def num(s):
        return float(s) if s.strip() not in ("", "nan", "NaN") else np.nan

    body = [r for r in rows[1:] if r and any(x.strip() for x in r)]
    return Export(path=path,
                  ids=[r[ix["point_id"]] for r in body],
                  lon=np.array([num(r[ix["longitude"]]) for r in body]),
                  lat=np.array([num(r[ix["latitude"]]) for r in body]),
                  rate=np.array([num(r[ix["rate_in_per_year"]]) for r in body]),
                  disp=np.array([num(r[ix["displacement_in"]]) for r in body]),
                  dates=[head[i] for i in dcols],
                  values=np.array([[num(r[i]) for i in dcols] for r in body], dtype=float))


def column_steps_days(dates: list[str]) -> np.ndarray:
    """Days from the previous column to each column (0 for the first)."""
    d = [_d(s) for s in dates]
    return np.array([0.0] + [(b - a).days for a, b in zip(d[:-1], d[1:])])


def accepted_years(ex: Export) -> np.ndarray:
    """Per point: time covered by accepted intervals, in years."""
    return (ex.valid * column_steps_days(ex.dates)).sum(axis=1) / YEAR_DAYS


def rate_identity(ex: Export) -> dict:
    """How well rate_in_per_year = displacement_in / accepted time holds."""
    t = accepted_years(ex)
    with np.errstate(divide="ignore", invalid="ignore"):
        err = np.abs(ex.disp / t - ex.rate)
    ok = np.isfinite(err)
    return {"points": int(len(err)), "match_0p0015": int((err[ok] <= 0.0015).sum()),
            "median_abs_error_in_yr": round(float(np.median(err[ok])), 5) if ok.any() else None}


def carried(ex: Export) -> np.ndarray:
    """Their level at every column: the last non-blank value at or before it, 0 before the first."""
    v = ex.values.copy()
    v[:, 0] = np.where(np.isfinite(v[:, 0]), v[:, 0], 0.0)
    for j in range(1, v.shape[1]):
        v[:, j] = np.where(np.isfinite(v[:, j]), v[:, j], v[:, j - 1])
    return v


def common_intervals(ex: Export, our_dates: list[str]) -> list[tuple[int, int]]:
    """Intervals (p, q) of export column indices, p < q, between consecutive
    export columns that both products have, up to our last date. An export
    column we lack is merged into the interval that spans it."""
    have = set(our_dates)
    last = max(our_dates)
    cols = [j for j, s in enumerate(ex.dates) if s in have and s <= last]
    return list(zip(cols[:-1], cols[1:]))


def gated(ex: Export, ours_in: np.ndarray, our_dates: list[str]) -> dict:
    """Both products through the vendor's bookkeeping over the intervals both can
    reproduce. ours_in: (points, len(our_dates)) our LOS series in inches at
    each point (NaN where we have no value).

    An interval counts for a point only when the vendor accepted every export column
    inside it and we have values at both ends. Returns per-point arrays:
    their_d / our_d (gated sums, inches), years (accepted time), and
    our_fit (our straight-line rate over the common window, in/yr), plus
    per-interval increments for cluster-level comparison."""
    pos = {s: i for i, s in enumerate(our_dates)}
    iv = common_intervals(ex, our_dates)
    if not iv:
        raise ValueError(f"{ex.path.name}: no dates in common with the product")
    C = carried(ex)
    steps = column_steps_days(ex.dates)
    n = len(ex.ids)
    acc = np.zeros((n, len(iv)), bool)
    inc_t = np.full((n, len(iv)), np.nan)
    inc_o = np.full((n, len(iv)), np.nan)
    days = np.zeros(len(iv))
    for k, (p, q) in enumerate(iv):
        a = ex.valid[:, p + 1:q + 1].all(axis=1)
        op, oq = ours_in[:, pos[ex.dates[p]]], ours_in[:, pos[ex.dates[q]]]
        a &= np.isfinite(op) & np.isfinite(oq)
        acc[:, k] = a
        inc_t[a, k] = (C[:, q] - C[:, p])[a]
        inc_o[a, k] = (oq - op)[a]
        days[k] = steps[p + 1:q + 1].sum()
    years = (acc * days).sum(axis=1) / YEAR_DAYS
    # our continuous straight line over the same span
    j0, j1 = pos[ex.dates[iv[0][0]]], pos[ex.dates[iv[-1][1]]]
    t = np.array([(_d(s) - _d(our_dates[j0])).days for s in our_dates[j0:j1 + 1]]) / YEAR_DAYS
    y = ours_in[:, j0:j1 + 1]
    fit = np.full(n, np.nan)
    for i in range(n):
        m = np.isfinite(y[i])
        if m.sum() >= 3:
            fit[i] = np.polyfit(t[m], y[i, m], 1)[0]
    return {"intervals": iv, "interval_days": days, "accepted": acc,
            "their_inc": inc_t, "our_inc": inc_o,
            "their_d": np.nansum(inc_t, axis=1), "our_d": np.nansum(inc_o, axis=1),
            "years": years, "our_fit": fit,
            "span": (ex.dates[iv[0][0]], ex.dates[iv[-1][1]])}


def summarize(ex: Export, g: dict) -> dict:
    """Cluster medians of the like-for-like figures (inches, in/yr)."""
    yrs = g["years"]
    with np.errstate(divide="ignore", invalid="ignore"):
        tr = np.where(yrs > 0, g["their_d"] / yrs, np.nan)
        orr = np.where(yrs > 0, g["our_d"] / yrs, np.nan)
    # cluster-median increments per interval (points accepted on it), then their agreement
    ti = np.array([np.nanmedian(c) if np.isfinite(c).sum() >= 10 else np.nan for c in g["their_inc"].T])
    oi = np.array([np.nanmedian(c) if np.isfinite(c).sum() >= 10 else np.nan for c in g["our_inc"].T])
    m = np.isfinite(ti) & np.isfinite(oi)
    full_t = np.arange(ex.values.shape[1])
    tfit = np.full(len(ex.ids), np.nan)
    tt = np.array([(_d(s) - _d(ex.dates[0])).days for s in ex.dates]) / YEAR_DAYS
    for i in range(len(ex.ids)):
        k = ex.valid[i]
        if k.sum() >= 3:
            tfit[i] = np.polyfit(tt[full_t[k]], ex.values[i, k], 1)[0]

    def med(a):
        a = np.asarray(a, float)
        return round(float(np.nanmedian(a)), 3) if np.isfinite(a).any() else None

    return {
        "points": len(ex.ids),
        "export_window": [ex.dates[0], ex.dates[-1]],
        "common_span": list(g["span"]),
        "their_listed_rate_in_yr": med(ex.rate),
        "their_straight_line_in_yr": med(tfit),
        "their_accepted_share": round(float(ex.valid[:, 1:].mean()), 3),
        "rate_identity": rate_identity(ex),
        "common": {
            "accepted_share": round(float(g["accepted"].mean()), 3),
            "their_gated_in": med(g["their_d"]), "our_gated_in": med(g["our_d"]),
            "their_rate_observed_in_yr": med(tr), "our_rate_observed_in_yr": med(orr),
            "our_straight_line_in_yr": med(g["our_fit"]),
            "interval_median_corr": round(float(np.corrcoef(ti[m], oi[m])[0, 1]), 2) if m.sum() >= 5 else None,
            "interval_median_sd_ratio_ours_over_theirs": (round(float(np.std(oi[m]) / np.std(ti[m])), 2)
                                                          if m.sum() >= 5 and np.std(ti[m]) > 0 else None),
            "intervals": int(m.sum()),
        },
    }


def ours_at_points(ts_file, lat: np.ndarray, lon: np.ndarray) -> tuple[np.ndarray, list[str], np.ndarray, np.ndarray]:
    """Our LOS series (inches, first date = 0 as stored) at the nearest
    product cell of each point, reading only the box around the points.
    Returns (series (points, dates), dates, rows, cols)."""
    import h5py
    with h5py.File(ts_file, "r") as f:
        a = dict(f.attrs)
        dates = [x.decode() if isinstance(x, bytes) else str(x) for x in f["date"][:]]
        x0, y0 = float(a["X_FIRST"]), float(a["Y_FIRST"])
        dx, dy = float(a["X_STEP"]), float(a["Y_STEP"])
        ny, nx = f["timeseries"].shape[1:]
        r = np.round((lat - y0) / dy - 0.5).astype(int)
        c = np.round((lon - x0) / dx - 0.5).astype(int)
        inside = (r >= 0) & (r < ny) & (c >= 0) & (c < nx)
        out = np.full((len(lat), len(dates)), np.nan)
        if inside.any():
            r0, r1, c0, c1 = r[inside].min(), r[inside].max() + 1, c[inside].min(), c[inside].max() + 1
            blk = f["timeseries"][:, r0:r1, c0:c1]
            out[inside] = (blk[:, r[inside] - r0, c[inside] - c0].T * IN_PER_M)
    return out, dates, r, c


def area_series(ts_file, box, keep=None) -> tuple[np.ndarray, int]:
    """Median of our LOS series (inches) over the cells in box = (south,
    north, west, east), optionally only where keep (a boolean grid on the
    product, e.g. built-up and coherent) is True. Subtracting it re-measures
    every point against that area instead of the reference pixel. Returns
    (series over dates, cells used)."""
    import h5py
    s, n, w, e = box
    with h5py.File(ts_file, "r") as f:
        a = dict(f.attrs)
        x0, y0 = float(a["X_FIRST"]), float(a["Y_FIRST"])
        dx, dy = float(a["X_STEP"]), float(a["Y_STEP"])
        r0, r1 = sorted((int((n - y0) / dy), int((s - y0) / dy) + 1))
        c0, c1 = sorted((int((w - x0) / dx), int((e - x0) / dx) + 1))
        blk = f["timeseries"][:, r0:r1, c0:c1] * IN_PER_M
    m = np.ones(blk.shape[1:], bool) if keep is None else np.asarray(keep)[r0:r1, c0:c1]
    if not m.any():
        raise ValueError("no usable cells in the reference box")
    return np.nanmedian(blk[:, m], axis=1), int(m.sum())
