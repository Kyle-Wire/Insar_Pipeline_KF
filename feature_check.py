"""
Long-pair feature check (22 Oct 2026): how a small feature moves against the
ground around it, year by year, from image pairs about 6 months and 1 year apart.

Why long pairs: the drift test (Site D, 22 Oct 2026) showed that farmland's
apparent sinking is a short-baseline "fading" signal. Cropland minus built-up
reads -17.9 mm/yr from 12-day pairs, about -1.4 from ~6-month pairs and
+0.2 +- 0.6 from ~1-year pairs. Pairs that long carry little or none of it, so
the phase of a feature against its surroundings needs no partner search, no
per-step quality gate and no weather model: the feature and a ring around it
share the atmosphere to within a few mm at ~1 km.

Per pair (dates i, j about `days` apart, matched within a tolerance):
  * both merged full-resolution SLCs over a window holding the feature and its
    ring; the rlks x 1 interferogram as topsStack forms it (mean of a x conj(b));
  * optionally each pixel's DEM-error residual removed (demErr x dBperp /
    (R sin inc), as the local change layer does);
  * the complex sum over the feature and over the ring; the pair's relative
    change is arg(S_feature x conj(S_ring)) in mm LOS (+ towards the
    satellite). The sums' coherence |S| / sqrt(P1 P2) must clear the noise level
    for their pixel count, or the pair is dropped for that area.
Then a velocity per calendar year (piecewise constant) by least squares over the
pairs of one separation: a pair's change = sum over years of v_year x (days of
the pair in that year) / 365.25. A year the usable pairs cover for less than
half a year, or cannot tell apart from another year, is unknown (NaN), never 0.
Uncertainty by delete-a-block jackknife over pairs grouped by start date (40
blocks, ~3 months each over 10 years; n_groups, recorded as jackknife_groups).
On pure noise with every pair kept (synthetic, review fc-2 and two re-checks,
23 Oct 2026, 4 seeds x 400 trials each) |v/se| >= 2 then passes in 4.3-4.9% of
years on ~6-month pairs, about the nominal 4.6%; the 22 Oct code with 10 blocks
of ~1 year gave 6.5-7.4%. That holds only while nearly every pair is kept. With
pairs dropped at random it passes in about 7% of years with half kept, 8-10%
with a quarter, about 11% with a tenth and 13% with a twentieth; keeping only
pairs that start in Nov-Feb (a third) gives 11% (23 Oct 2026, 2 seeds,
exploratory). The pit edge at Site D kept 14 of 257 pairs (5%). So where many
pairs drop, read the 6-month error as a lower bound too; pairs_kept, pairs_used
and groups_per_year in the record say how many pairs and blocks a year rests on.
On ~1-year pairs it passes in 10.2-11.4% with every pair kept (8-12% with
pairs dropped), about a point MORE than the 22 Oct code (9.4-10.0%).
Back-to-back 1-year pairs share a date (one ends where the next starts) but
fall in different blocks, so the blocks are not independent.
Blocks of dates instead of start dates over-correct (~2% on the same noise, and
~1% on 6-month pairs). Read the 1-year error as a lower bound.
A year with a value but no error (a refit without one block cannot solve it) is
weakly determined: its value can be far off, so don't judge it.
The 22 Oct 2026 errors can be reproduced only with that day's code: since
review fc-1 an error is given only where every refit solves the year, so
n_groups=10 now leaves some years without one (its verdicts did not use the
errors).

A pair holds +-lambda/4 (13.9 mm) without ambiguity, i.e. +-28 mm/yr on ~6-month
pairs and +-14 mm/yr on ~1-year pairs. Faster relative motion aliases in the
1-year set, which is why both separations are computed and compared.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .local_change import (RAD2MM, YEAR, bperp_by_date, looks_mean, raster_info, topo_factor,
                           topo_step_mm)

EFF_RATIO = 0.65          # independent share of the rlks x pixels samples (oversampled range)


def _ordinal(d: str) -> float:
    return float(np.datetime64(f"{d[:4]}-{d[4:6]}-{d[6:]}").astype("datetime64[D]").astype(int))


def pairs_by_days(dates: list[str], days: float, tol: float = 18.0) -> list[tuple[int, int]]:
    """For each date, the date closest to `days` later (within +-tol days)."""
    t = np.array([_ordinal(d) for d in dates])
    out = []
    for i in range(len(dates)):
        k = np.argmin(np.abs(t - (t[i] + days)))
        if k > i and abs(t[k] - t[i] - days) <= tol:
            out.append((i, int(k)))
    return out


def year_design(dates: list[str], pairs: list[tuple[int, int]]) -> tuple[np.ndarray, list[int]]:
    """(A, years): A[p, y] = days of pair p inside calendar year y / 365.25."""
    if not dates:                                         # --start/--end left no dates
        return np.zeros((len(pairs), 0)), []
    t = np.array([_ordinal(d) for d in dates])
    y0, y1 = int(dates[0][:4]), int(dates[-1][:4])
    years = list(range(y0, y1 + 1))
    starts = np.array([_ordinal(f"{y}0101") for y in years] + [_ordinal(f"{y1 + 1}0101")])
    A = np.zeros((len(pairs), len(years)))
    for p, (i, j) in enumerate(pairs):
        a, b = t[i], t[j]
        A[p] = np.clip(np.minimum(b, starts[1:]) - np.maximum(a, starts[:-1]), 0, None) / YEAR
    return A, years


def _solve(M: np.ndarray, d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Least squares M x = d, and which unknowns the rows actually determine: an
    unknown the null space of M touches can take any value (lstsq then returns
    the minimum-norm split, e.g. one pair shared out over two years)."""
    n = M.shape[1]
    if M.shape[0] == 0:
        return np.zeros(n), np.zeros(n, bool)
    u, s, vt = np.linalg.svd(M, full_matrices=False)
    r = int((s > s.max() * max(M.shape) * np.finfo(float).eps).sum())      # lstsq's own cut-off
    x = vt[:r].T @ ((u[:, :r].T @ d) / s[:r])                              # = lstsq's minimum-norm solution
    return x, 1.0 - (vt[:r] ** 2).sum(0) < 1e-10                           # unknown lies in the row space


def yearly_velocity(A: np.ndarray, d_mm: np.ndarray, groups: np.ndarray, min_years_cover: float = 0.5) -> dict:
    """Least-squares velocity per year (mm/yr) from pair changes d_mm, with a
    delete-a-group jackknife standard error.

    A year is solved only if the pairs used for it cover at least
    min_years_cover years of it. A pair reaching into a year short of cover
    is dropped, because it would carry that year's movement unmodelled. That can
    leave a neighbouring year short in turn, so the check repeats until nothing
    changes. A year the remaining pairs cannot separate from another (one pair
    split over two years) is not solved either; its pairs stay in the solve,
    which leaves the years they do determine unchanged. A year not solved is
    NaN, never 0. Its standard error is given only where every delete-a-group
    refit still solves it (review fc-1, 22 Oct 2026: years with no usable pair
    read 0.0). cover_years is each year's cover by all the pairs given;
    cover_used_years is its cover by the pairs used, so a year with cover_years
    >= min_years_cover can still read NaN when its pairs were dropped."""
    cover = A.sum(0)
    v = np.full(A.shape[1], np.nan)
    se = np.full(A.shape[1], np.nan)
    keep = cover >= min_years_cover
    while True:                                           # keep only shrinks, so this ends
        rows = (A[:, keep].sum(1) > 0) & (A[:, ~keep].sum(1) == 0)
        now = keep & (A[rows].sum(0) >= min_years_cover)
        if (now == keep).all():
            break
        keep = now
    groups_per_year = np.zeros(A.shape[1], int)
    out = {"v": v, "se": se, "cover_years": cover, "cover_used_years": A[rows].sum(0),
           "pairs_used": int(rows.sum()), "groups_per_year": groups_per_year}
    if not keep.any() or not rows.any():
        return out
    cols = np.flatnonzero(keep)
    x, ok = _solve(A[rows][:, keep], d_mm[rows])
    v[cols[ok]] = x[ok]
    for y in cols[ok]:
        groups_per_year[y] = len(np.unique(groups[rows & (A[:, y] > 0)]))
    g = np.unique(groups[rows])
    if len(g) >= 3:
        est = []
        for k in g:
            m = rows & (groups != k)
            xk, okk = _solve(A[m][:, keep], d_mm[m])
            est.append(np.where(okk, xk, np.nan))
        est = np.array(est)
        n = len(est)
        every = ok & np.isfinite(est).all(0)
        jk = np.sqrt((n - 1) / n * ((est - est.mean(0)) ** 2).sum(0))
        se[cols[every]] = jk[every]
    return out


def noise_coherence(n_pixels: int, rlks: int, eff_ratio: float = EFF_RATIO) -> float:
    """Coherence a sum of pure noise over n_pixels reaches (about 3 / sqrt(N_eff))."""
    return float(min(1.0, 3.0 / np.sqrt(max(1.0, eff_ratio * rlks * n_pixels))))


class Stack:
    """Merged SLCs and geometry of a project, read over small windows."""

    def __init__(self, stack_dir: Path, rlks: int):
        M = Path(stack_dir) / "merged"
        self.slc_dir = M / "SLC"
        self.dates = sorted(d.name for d in self.slc_dir.iterdir() if (d / f"{d.name}.slc.full").exists())
        self.shape = raster_info(self.slc_dir / self.dates[0] / f"{self.dates[0]}.slc.full")[:2]
        self.rlks = int(rlks)
        lat = M / "geom_reference" / "lat.rdr"
        h, w, dt = raster_info(lat)
        if (h, w) != (self.shape[0], self.shape[1] // self.rlks):
            raise RuntimeError(f"merged geometry {(h, w)} is not on the {self.rlks} x 1 grid of the SLCs {self.shape}")
        self.lat = np.memmap(lat, dtype=dt, mode="r", shape=(h, w))
        self.lon = np.memmap(M / "geom_reference" / "lon.rdr", dtype=dt, mode="r", shape=(h, w))

    def window(self, lat0: float, lon0: float, radius_m: float, step: int = 8) -> tuple[int, int, int, int]:
        """Rows/cols (on the rlks x 1 grid) of a box holding every pixel within radius_m."""
        sub_lat = np.asarray(self.lat[::step, ::step]); sub_lon = np.asarray(self.lon[::step, ::step])
        c = np.cos(np.radians(lat0))
        d = np.hypot((sub_lat - lat0) * 111320.0, (sub_lon - lon0) * 111320.0 * c)
        near = np.argwhere(d <= radius_m + 3 * step * 16)
        if not len(near):
            raise RuntimeError(f"({lat0}, {lon0}) is outside the merged frame")
        r0, c0 = near.min(0) * step; r1, c1 = (near.max(0) + 1) * step
        return max(0, r0 - step), min(self.lat.shape[0], r1 + step), max(0, c0 - step), min(self.lat.shape[1], c1 + step)

    def slc(self, date: str, r0, r1, c0, c1) -> np.ndarray:
        m = np.memmap(self.slc_dir / date / f"{date}.slc.full", dtype=np.complex64, mode="r", shape=self.shape)
        return np.asarray(m[r0:r1, self.rlks * c0:self.rlks * c1])


def masks_for(lat, lon, area: dict, landcover=None) -> tuple[np.ndarray, np.ndarray]:
    """(feature, ring) boolean masks on a window. The feature is the pixels within
    area['radius_m'] of the centre, or within area['buffer_m'] of any of
    area['points'] (lat, lon pairs); the ring is ring_m = (inner, outer) from the
    centre, minus the feature, minus water when a land-cover class array is given."""
    c = np.cos(np.radians(area["lat"]))
    d = np.hypot((lat - area["lat"]) * 111320.0, (lon - area["lon"]) * 111320.0 * c)
    if area.get("points") is not None:
        from scipy.spatial import cKDTree
        pts = np.asarray(area["points"], float)
        tree = cKDTree(np.c_[pts[:, 0] * 111320.0, pts[:, 1] * 111320.0 * c])
        dist, _ = tree.query(np.c_[lat.ravel() * 111320.0, lon.ravel() * 111320.0 * c])
        feat = (dist <= float(area.get("buffer_m", 20.0))).reshape(lat.shape)
    else:
        feat = d <= float(area["radius_m"])
    r_in, r_out = area["ring_m"]
    ring = (d >= r_in) & (d <= r_out) & ~feat
    if landcover is not None:
        ring &= landcover != 80                                # WorldCover permanent water
        if area.get("ring_classes"):
            ring &= np.isin(landcover, list(area["ring_classes"]))
    return feat, ring


def worldcover_at(path: Path, lat, lon) -> np.ndarray:
    import rasterio
    with rasterio.open(path) as r:
        wc = r.read(1); t = r.transform
    row = np.floor((lat - t.f) / t.e).astype(int); col = np.floor((lon - t.c) / t.a).astype(int)
    ok = (row >= 0) & (row < wc.shape[0]) & (col >= 0) & (col < wc.shape[1])
    out = np.zeros(lat.shape, np.uint8); out[ok] = wc[row[ok], col[ok]]
    return out


def check_area(stack: Stack, area: dict, separations=(182.0, 365.0), tol: float = 18.0,
               dem_error: Path | None = None, baselines: Path | None = None, landcover: Path | None = None,
               start: str | None = None, end: str | None = None, n_groups: int = 40) -> dict:
    """The yearly relative velocity of one area against its ring, per separation.
    n_groups: the jackknife's blocks of start dates (~3 months each over 10 years;
    10 blocks of ~1 year left each year's error resting on 1-2 blocks, review fc-2),
    recorded as jackknife_groups. n_groups=10 does not reproduce the 22 Oct 2026
    errors (see the module docstring)."""
    r0, r1, c0, c1 = stack.window(area["lat"], area["lon"], float(area["ring_m"][1]))
    lat = np.asarray(stack.lat[r0:r1, c0:c1], np.float64); lon = np.asarray(stack.lon[r0:r1, c0:c1], np.float64)
    lc = worldcover_at(landcover, lat, lon) if landcover else None
    feat, ring = masks_for(lat, lon, area, lc)
    if feat.sum() < 3 or ring.sum() < 10:
        raise RuntimeError(f"{area['name']}: {int(feat.sum())} feature and {int(ring.sum())} ring pixels in the frame")
    s, e = (start or "00000000").replace("-", ""), (end or "99999999").replace("-", "")
    dates = [d for d in stack.dates if s <= d <= e]
    K = bp = None
    if dem_error is not None:
        K = topo_factor(dem_error, lat, lon)
        bp = bperp_by_date(baselines)
    cache = {d: stack.slc(d, r0, r1, c0, c1) for d in dates}
    floor_f = noise_coherence(int(feat.sum()), stack.rlks); floor_r = noise_coherence(int(ring.sum()), stack.rlks)
    out = {"name": area["name"], "centre": [area["lat"], area["lon"]], "feature_pixels": int(feat.sum()),
           "ring_pixels": int(ring.sum()), "ring_m": list(area["ring_m"]),
           "dem_error": bool(K is not None and np.any(K[feat] != 0)), "jackknife_groups": int(n_groups),
           "separations": {}}
    for days in separations:
        pairs = pairs_by_days(dates, days, tol)
        d_mm, coh_f, coh_r, kept = [], [], [], []
        for i, j in pairs:
            a, b = cache[dates[i]], cache[dates[j]]
            z = looks_mean(a * np.conj(b), stack.rlks)
            p1 = looks_mean((a.real ** 2 + a.imag ** 2).astype(np.float32), stack.rlks)
            p2 = looks_mean((b.real ** 2 + b.imag ** 2).astype(np.float32), stack.rlks)
            if K is not None and dates[i] in bp and dates[j] in bp:
                z = z * np.exp(-1j * topo_step_mm(K, bp[dates[i]], bp[dates[j]]) / RAD2MM)
            sf, sr = z[feat].sum(), z[ring].sum()
            cf = abs(sf) / np.sqrt(max(p1[feat].sum() * p2[feat].sum(), 1e-30))
            cr = abs(sr) / np.sqrt(max(p1[ring].sum() * p2[ring].sum(), 1e-30))
            d_mm.append(float(np.angle(sf * np.conj(sr)) * RAD2MM)); coh_f.append(cf); coh_r.append(cr)
            kept.append(cf >= floor_f and cr >= floor_r)
        kept = np.array(kept, bool); d_mm = np.array(d_mm)
        use = [p for p, k in zip(pairs, kept) if k]
        A, years = year_design(dates, use)
        order = np.argsort([i for i, _ in use]) if use else np.array([], int)
        groups = np.empty(len(use), int)
        groups[order] = np.arange(len(use)) * n_groups // max(1, len(use))
        yv = yearly_velocity(A, d_mm[kept], groups) if use else None
        t = np.array([_ordinal(d) for d in dates])
        all_rate = float(np.linalg.lstsq(((t[[j for _, j in use]] - t[[i for i, _ in use]]) / YEAR)[:, None],
                                         d_mm[kept], rcond=None)[0][0]) if use else None
        out["separations"][str(int(days))] = {
            "pairs": len(pairs), "pairs_kept": int(kept.sum()),
            "coherence_feature_median": float(np.median(coh_f)) if coh_f else None,
            "coherence_ring_median": float(np.median(coh_r)) if coh_r else None,
            "noise_floor": [floor_f, floor_r],
            "years": years,
            "velocity_mm_yr": None if yv is None else [None if not np.isfinite(x) else float(x) for x in yv["v"]],
            "se_mm_yr": None if yv is None else [None if not np.isfinite(x) else float(x) for x in yv["se"]],
            "cover_years": None if yv is None else [float(x) for x in yv["cover_years"]],
            "cover_used_years": None if yv is None else [float(x) for x in yv["cover_used_years"]],
            "pairs_used": 0 if yv is None else yv["pairs_used"],
            "groups_per_year": None if yv is None else [int(x) for x in yv["groups_per_year"]],
            "whole_record_rate_mm_yr": all_rate,
            "pair_changes": [[dates[i], dates[j], float(x), bool(k)] for (i, j), x, k in zip(pairs, d_mm, kept)],
        }
    return out


def write(path: Path, rec: dict) -> Path:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(rec, indent=1))
    tmp.replace(path)
    return path
