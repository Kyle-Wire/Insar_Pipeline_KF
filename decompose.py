"""
Up and East from two or more look directions (PLAN D1, experiment grade).

    python -m insar_pipeline.decompose <cfg_asc> <cfg_desc> --out DIR

One satellite geometry measures one projection of a three-component motion,
along its line of sight (LOS). An ascending and a descending pass look at the
same ground from opposite sides, so together they separate vertical motion
from east-west motion. North-south motion stays hidden: both passes fly close
to north-south, so neither is sensitive to it. North is NOT estimated, and
every output says so. Whatever north-south motion there is leaks into Up and
East; the leak factor is written beside them (north_leak_into_*.tif: mm of
error in Up or East per mm of northward motion, about -0.16 for Up on
Sentinel-1 at 30-46 degrees incidence).

What it does, for each run (one config per geometry):

1. Reads the run's delivered products through the pipeline's own helpers:
   the time series (deliverables.timeseries_file), the per-cell LOS unit
   vector (deliverables.geometry_field: geo/geo_geometryRadar.h5 or the
   DISP-S1 geometryGeo.h5) and temporal coherence. A cell is reliable when
   its value is finite and its temporal coherence is at or above the floor
   (the config's min_temporal_coherence unless --min-coherence is given).
2. Puts every run on one latitude/longitude grid: the area all runs cover,
   at the coarser run's posting (or --posting-m). Values are averaged over
   the source cells inside each grid cell (reliable cells only; a grid cell
   needs at least half of its source cells reliable); the LOS unit vectors
   are averaged as vectors and renormalised.
3. Uses only a common window: the dates all runs cover (narrowed further by
   --window). Runs that started or ended on different dates are never mixed.
   - Rates: each run's straight-line rate is fitted to its own dates inside
     the common window (the same formal error as MintPy's velocityStd), then
     decomposed.
   - Series: each run is interpolated linearly to common epochs (default:
     the 1st of each month) from its two SAR dates either side, only when
     both are at most --max-gap-days (12) away. Different-day acquisitions
     are never treated as simultaneous. The series is the change since the
     first epoch every run has.
4. References every run to the same ground (default: the first run's own
   reference point, if every run has reliable cells there; or --ref LAT,LON):
   the median of reliable cells within 100 m is subtracted per run and epoch.
5. Solves, cell by cell, the least-squares system
       LOS_g = e_g * East + u_g * Up        (one row per run g)
   with each run's own per-cell LOS vector (positive toward the satellite,
   MintPy's convention: uplift positive). For rates the rows are weighted by
   the runs' 1-sigma rate errors and the errors are propagated. This is the
   system MintPy's asc_desc2horz_vert.py solves (horizontal azimuth -90 deg,
   i.e. east), cell by cell rather than per 20x20 window.

Outputs (DIR): GeoTIFFs in mm and mm/yr, MintPy-style HDF5 series in metres,
decomposition_summary.json and a plain-English decomposition_README.txt.

Nothing here changes a run's own products or any client number: it reads the
runs and writes only into DIR.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

M_PER_DEG_LAT = 111_320.0

NORTH_NOTE = ("North is not estimated. Both look directions fly close to north-south, so neither "
              "is sensitive to north-south motion; any north-south motion is folded into Up and "
              "East. north_leak_into_vertical.tif and north_leak_into_east_west.tif give the size "
              "of that leak: mm of error per mm of northward motion.")
UP_SIGN = "Positive = up (uplift). Negative = down (sinking)."
EAST_SIGN = "Positive = moving east. Negative = moving west."
SERIES_SIGMA_NOTE = ("not stated: the time series carry no per-epoch uncertainty. "
                     "amplification_vertical / amplification_east (in the layers file) give the "
                     "Up and East 1-sigma per 1 mm of LOS noise in every run.")
RATE_MODELS = ("linear", "linear_annual")


# ------------------------------------------------------------------ grids

@dataclass(frozen=True)
class Grid:
    """A north-up raster grid. (x0, y0) is the outer corner of the first
    cell, as MintPy's X_FIRST / Y_FIRST; dy is negative."""
    x0: float
    dx: float
    y0: float
    dy: float
    nx: int
    ny: int
    crs: str = "EPSG:4326"

    @classmethod
    def from_attrs(cls, attrs: dict, shape) -> "Grid":
        from .deliverables import _crs, _geotransform
        x0, dx, y0, dy = _geotransform(attrs)
        if dy >= 0 or dx <= 0:
            raise ValueError(f"The product grid is not north-up (X_STEP {dx}, Y_STEP {dy}); "
                             "only north-up geocoded products can be decomposed.")
        return cls(x0, dx, y0, dy, int(shape[1]), int(shape[0]), _crs(attrs))

    @property
    def geographic(self) -> bool:
        return self.crs.upper() == "EPSG:4326"

    @property
    def shape(self) -> tuple:
        return (self.ny, self.nx)

    def transform(self):
        from affine import Affine
        return Affine(self.dx, 0.0, self.x0, 0.0, self.dy, self.y0)

    def bounds(self) -> tuple:
        """(W, S, E, N) in the grid's own coordinates."""
        return (self.x0, self.y0 + self.ny * self.dy, self.x0 + self.nx * self.dx, self.y0)

    def lonlat_bounds(self) -> tuple:
        if self.geographic:
            return self.bounds()
        from rasterio.warp import transform_bounds
        return tuple(transform_bounds(self.crs, "EPSG:4326", *self.bounds(), densify_pts=21))

    def sub(self, r0: int, r1: int, c0: int, c1: int) -> "Grid":
        return Grid(self.x0 + c0 * self.dx, self.dx, self.y0 + r0 * self.dy, self.dy,
                    c1 - c0, r1 - r0, self.crs)

    def centres(self) -> tuple:
        """(x of each column centre, y of each row centre)."""
        return (self.x0 + (np.arange(self.nx) + 0.5) * self.dx,
                self.y0 + (np.arange(self.ny) + 0.5) * self.dy)

    def attrs(self) -> dict:
        """MintPy-style grid attributes (deliverables._write_geotiff reads these)."""
        if not self.geographic:
            raise ValueError("the common grid is always latitude/longitude")
        return {"X_FIRST": repr(self.x0), "Y_FIRST": repr(self.y0), "X_STEP": repr(self.dx),
                "Y_STEP": repr(self.dy), "WIDTH": str(self.nx), "LENGTH": str(self.ny),
                "X_UNIT": "degrees", "Y_UNIT": "degrees", "EPSG": "4326"}

    def posting_m(self) -> tuple:
        """Approximate (east-west, north-south) cell size in metres."""
        if not self.geographic:
            return abs(self.dx), abs(self.dy)
        lat = self.y0 + self.ny * self.dy / 2
        return (abs(self.dx) * M_PER_DEG_LAT * math.cos(math.radians(lat)),
                abs(self.dy) * M_PER_DEG_LAT)


def _deg_steps(g: Grid, lat_c: float) -> tuple:
    if g.geographic:
        return abs(g.dx), abs(g.dy)
    return (abs(g.dx) / (M_PER_DEG_LAT * math.cos(math.radians(lat_c))),
            abs(g.dy) / M_PER_DEG_LAT)


def common_grid(grids, bbox=None, posting_m: float | None = None) -> Grid:
    """
    The latitude/longitude grid every run is put on: the area all of them
    cover (cut to bbox = (W, S, E, N) when given), at the coarsest posting of
    the runs in each direction unless posting_m is given. When a run is
    already on a lat/lon grid with that posting, the common grid is aligned
    to its cells, so that run is used cell for cell without resampling.
    """
    bnds = [g.lonlat_bounds() for g in grids]
    W = max(b[0] for b in bnds)
    S = max(b[1] for b in bnds)
    E = min(b[2] for b in bnds)
    N = min(b[3] for b in bnds)
    if bbox is not None:
        W, S, E, N = max(W, bbox[0]), max(S, bbox[1]), min(E, bbox[2]), min(N, bbox[3])
    if not (E > W and N > S):
        raise ValueError("The runs do not cover any common ground (their grids do not overlap"
                         + (" inside the requested box" if bbox is not None else "") + ").")
    lat_c = (S + N) / 2
    snap_x = snap_y = None
    if posting_m:
        dx = posting_m / (M_PER_DEG_LAT * math.cos(math.radians(lat_c)))
        dy = posting_m / M_PER_DEG_LAT
    else:
        steps = [_deg_steps(g, lat_c) for g in grids]
        dx = max(s[0] for s in steps)
        dy = max(s[1] for s in steps)
        for g in grids:
            if g.geographic and snap_x is None and math.isclose(abs(g.dx), dx, rel_tol=1e-9):
                snap_x = g.x0
            if g.geographic and snap_y is None and math.isclose(abs(g.dy), dy, rel_tol=1e-9):
                snap_y = g.y0
    x0 = W if snap_x is None else snap_x + math.ceil((W - snap_x) / dx - 1e-6) * dx
    y0 = N if snap_y is None else snap_y - math.ceil((snap_y - N) / dy - 1e-6) * dy
    nx = int(math.floor((E - x0) / dx + 1e-6))
    ny = int(math.floor((y0 - S) / dy + 1e-6))
    if nx < 1 or ny < 1:
        raise ValueError("The common area is smaller than one grid cell.")
    return Grid(x0, dx, y0, -dy, nx, ny, "EPSG:4326")


class _Resampler:
    """Moves arrays from a window of a run's grid onto the common grid:
    cell for cell when the grids line up, otherwise an average of the source
    cells inside each common cell (NaN cells left out), with the common cell
    blanked when fewer than `min_valid_fraction` of its source cells hold a
    value."""

    def __init__(self, src: Grid, dst: Grid, min_valid_fraction: float = 0.5):
        self.dst, self.min_valid_fraction = dst, float(min_valid_fraction)
        self.window = self._window(src, dst)
        r0, r1, c0, c1 = self.window
        self.src = src.sub(r0, r1, c0, c1)
        self.offset = None
        if self.src.geographic and math.isclose(self.src.dx, dst.dx, rel_tol=1e-9) \
                and math.isclose(self.src.dy, dst.dy, rel_tol=1e-9):
            co = (dst.x0 - self.src.x0) / dst.dx
            ro = (dst.y0 - self.src.y0) / dst.dy
            if abs(co - round(co)) < 1e-4 and abs(ro - round(ro)) < 1e-4:
                ro, co = int(round(ro)), int(round(co))
                if ro >= 0 and co >= 0 and ro + dst.ny <= self.src.ny and co + dst.nx <= self.src.nx:
                    self.offset = (ro, co)

    @staticmethod
    def _window(src: Grid, dst: Grid) -> tuple:
        W, S, E, N = dst.bounds()
        if not src.geographic:
            from rasterio.warp import transform_bounds
            W, S, E, N = transform_bounds("EPSG:4326", src.crs, W, S, E, N, densify_pts=21)
        m = 2
        c0 = int(math.floor((W - src.x0) / src.dx)) - m
        c1 = int(math.ceil((E - src.x0) / src.dx)) + m
        r0 = int(math.floor((src.y0 - N) / -src.dy)) - m
        r1 = int(math.ceil((src.y0 - S) / -src.dy)) + m
        r0, c0 = max(r0, 0), max(c0, 0)
        r1, c1 = min(r1, src.ny), min(c1, src.nx)
        if r1 <= r0 or c1 <= c0:
            raise ValueError("A run's grid does not reach the common area.")
        return r0, r1, c0, c1

    @property
    def identity(self) -> bool:
        return self.offset is not None

    def __call__(self, arr: np.ndarray) -> np.ndarray:
        """arr is on the source WINDOW (self.src)."""
        arr = np.asarray(arr, dtype=np.float32)
        if arr.shape != self.src.shape:
            raise ValueError(f"array {arr.shape} is not on the source window {self.src.shape}")
        if self.offset is not None:
            ro, co = self.offset
            return np.array(arr[ro:ro + self.dst.ny, co:co + self.dst.nx], dtype=np.float32)
        from rasterio.warp import Resampling, reproject
        out = np.full(self.dst.shape, np.nan, np.float32)
        reproject(arr, out, src_transform=self.src.transform(), src_crs=self.src.crs,
                  dst_transform=self.dst.transform(), dst_crs=self.dst.crs,
                  resampling=Resampling.average, src_nodata=np.nan, dst_nodata=np.nan)
        frac = np.full(self.dst.shape, np.nan, np.float32)
        reproject(np.isfinite(arr).astype(np.float32), frac, src_transform=self.src.transform(),
                  src_crs=self.src.crs, dst_transform=self.dst.transform(), dst_crs=self.dst.crs,
                  resampling=Resampling.average, src_nodata=-1.0, dst_nodata=np.nan)
        out[~(frac >= self.min_valid_fraction)] = np.nan
        return out

    def vectors(self, e, n, u) -> tuple:
        """LOS unit vectors: averaged as vectors, then renormalised."""
        e, n, u = self(e), self(n), self(u)
        norm = np.sqrt(e * e + n * n + u * u)
        with np.errstate(invalid="ignore", divide="ignore"):
            return e / norm, n / norm, u / norm


# ------------------------------------------------------------------ the solve

def solve_up_east(los, east_c, up_c, north_c=None, sigma=None, max_condition: float = 10.0) -> dict:
    """
    Per-cell least squares for East and Up from N >= 2 LOS observations.

    los, east_c, up_c, north_c, sigma: arrays of shape (N, ...), one row per
    run. east_c / up_c / north_c are the east, up and north components of the
    LOS unit vector toward the satellite. Rows are weighted by 1/sigma^2 when
    sigma is given and finite (the estimate is the same either way when
    N = 2); a cell is solved when at least two runs have a value there and
    the condition number of its unweighted design matrix is at most
    max_condition.

    Returns arrays of shape (...): up, east; sigma_up, sigma_east (propagated
    from sigma, NaN without it); amplification_up / _east (1-sigma per 1 of
    LOS noise in every run); condition; leak_up / leak_east (error per unit of
    northward motion, which is not estimated; NaN without north_c); n_used.
    """
    los = np.asarray(los, dtype=np.float64)
    e = np.asarray(east_c, dtype=np.float64)
    u = np.asarray(up_c, dtype=np.float64)
    if los.shape != e.shape or los.shape != u.shape or los.ndim < 1 or los.shape[0] < 2:
        raise ValueError("solve_up_east needs matching (N, ...) arrays with N >= 2")
    ok = np.isfinite(los) & np.isfinite(e) & np.isfinite(u)
    if sigma is not None:
        s = np.asarray(sigma, dtype=np.float64)
        s_ok = np.isfinite(s) & (s > 0)
        sigma_ok = np.all(s_ok | ~ok, axis=0)
        w = np.where(ok & s_ok, 1.0 / np.where(s_ok, s, 1.0) ** 2, 0.0)
        w = np.where(sigma_ok[None], w, ok.astype(np.float64))
    else:
        sigma_ok = np.zeros(los.shape[1:], bool)
        w = ok.astype(np.float64)
    w1 = ok.astype(np.float64)
    d = np.where(ok, los, 0.0)
    e0, u0 = np.where(ok, e, 0.0), np.where(ok, u, 0.0)

    def normal(wt):
        return (np.sum(wt * e0 * e0, 0), np.sum(wt * e0 * u0, 0), np.sum(wt * u0 * u0, 0))

    A, B, C = normal(w)
    A1, B1, C1 = normal(w1)
    n_used = ok.sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        det, det1 = A * C - B * B, A1 * C1 - B1 * B1
        tr = (A1 + C1) / 2
        disc = np.sqrt(np.maximum(((A1 - C1) / 2) ** 2 + B1 * B1, 0.0))
        lmin, lmax = tr - disc, tr + disc
        cond = np.where(lmin > 0, np.sqrt(lmax / np.where(lmin > 0, lmin, 1.0)), np.inf)
        good = (n_used >= 2) & (cond <= max_condition) & (det > 0) & (det1 > 0)
        rE, rU = np.sum(w * e0 * d, 0), np.sum(w * u0 * d, 0)
        east = (C * rE - B * rU) / det
        up = (A * rU - B * rE) / det
        sig_e = np.sqrt(C / det)
        sig_u = np.sqrt(A / det)
        amp_e = np.sqrt(C1 / det1)
        amp_u = np.sqrt(A1 / det1)
        if north_c is not None:
            nn = np.where(ok, np.asarray(north_c, dtype=np.float64), 0.0)
            nE, nU = np.sum(w * e0 * nn, 0), np.sum(w * u0 * nn, 0)
            leak_e = (C * nE - B * nU) / det
            leak_u = (A * nU - B * nE) / det
        else:
            leak_e = leak_u = np.full(los.shape[1:], np.nan)

    def m(a, extra=None):
        keep = good if extra is None else good & extra
        return np.where(keep, a, np.nan).astype(np.float32)

    return {"up": m(up), "east": m(east), "sigma_up": m(sig_u, sigma_ok),
            "sigma_east": m(sig_e, sigma_ok), "amplification_up": m(amp_u),
            "amplification_east": m(amp_e), "condition": np.where(n_used >= 2, cond, np.nan).astype(np.float32),
            "leak_up": m(leak_u), "leak_east": m(leak_e), "n_used": n_used.astype(np.int16),
            "solved": good}


# ------------------------------------------------------------------ runs

@dataclass
class Geometry:
    """One run (one look direction) as the decomposition sees it."""
    label: str                      # short, file-safe: "asc_t9"
    name: str                       # "Sentinel-1 ascending track 9"
    ts_path: Path
    dates: list                     # YYYYMMDD strings
    grid: Grid
    reliable: np.ndarray            # bool, the product grid
    los_e: np.ndarray               # LOS unit vector toward the satellite, product grid
    los_n: np.ndarray
    los_u: np.ndarray
    floor: float | None = None
    ref_lonlat: tuple | None = None
    velocity_path: Path | None = None
    source: dict = field(default_factory=dict)


def _decode_dates(raw) -> list:
    return [d.decode() if isinstance(d, bytes) else str(d) for d in raw]


def _series_head(ts_path: Path) -> tuple:
    import h5py
    with h5py.File(ts_path, "r") as f:
        attrs = dict(f.attrs)
        dates = _decode_dates(f["date"][:])
        shape = f["timeseries"].shape[1:]
    attrs = {k: (v.decode() if isinstance(v, bytes) else v) for k, v in attrs.items()}
    return dates, attrs, shape


def _ref_lonlat(attrs: dict, grid: Grid):
    """Centre of the run's reference cell (REF_Y / REF_X), as (lon, lat)."""
    try:
        ry, rx = int(float(attrs["REF_Y"])), int(float(attrs["REF_X"]))
    except (KeyError, TypeError, ValueError):
        return None
    from .deliverables import _rc_to_lonlat
    lon, lat = _rc_to_lonlat(grid_attrs_any(grid), ry, rx)
    return float(lon), float(lat)


def grid_attrs_any(grid: Grid) -> dict:
    a = {"X_FIRST": repr(grid.x0), "Y_FIRST": repr(grid.y0), "X_STEP": repr(grid.dx),
         "Y_STEP": repr(grid.dy), "WIDTH": str(grid.nx), "LENGTH": str(grid.ny)}
    a["EPSG"] = grid.crs.split(":")[-1]
    return a


def _reliable(coh: np.ndarray | None, floor: float | None, *vecs) -> np.ndarray:
    ok = np.ones(vecs[0].shape, bool)
    for v in vecs:
        ok &= np.isfinite(v)
    if coh is not None and floor is not None:
        with np.errstate(invalid="ignore"):
            ok &= np.asarray(coh) >= floor
    return ok


def geometry_from_files(ts_path, geometry_path, coherence_path=None, *, label: str,
                        name: str | None = None, min_coherence: float | None = 0.4,
                        velocity_path=None) -> Geometry:
    """
    A run from its files: a MintPy-style time series (timeseries, date), a
    geometry file holding either incidenceAngle + azimuthAngle (MintPy's
    geocoded geometryRadar) or los_east/los_north/los_up (the DISP-S1
    geometryGeo.h5), and a temporal coherence file.
    """
    import h5py
    ts_path = Path(ts_path)
    dates, attrs, shape = _series_head(ts_path)
    grid = Grid.from_attrs(attrs, shape)
    with h5py.File(geometry_path, "r") as f:
        if "los_up" in f:
            e, n, u = (f[k][:].astype(np.float32) for k in ("los_east", "los_north", "los_up"))
        else:
            inc = np.radians(f["incidenceAngle"][:].astype(np.float32))
            az = np.radians(f["azimuthAngle"][:].astype(np.float32))
            inc[inc == 0] = np.nan
            # MintPy: azimuthAngle from north, anticlockwise, of the ground-to-satellite
            # direction; the same conversion as deliverables.geometry_field
            e, n, u = -np.sin(inc) * np.sin(az), np.sin(inc) * np.cos(az), np.cos(inc)
    if tuple(u.shape) != tuple(shape):
        raise ValueError(f"The geometry {geometry_path} is {u.shape} but the series is {tuple(shape)}.")
    coh = None
    if coherence_path is not None:
        with h5py.File(coherence_path, "r") as f:
            coh = f["temporalCoherence"][:]
    return Geometry(label=label, name=name or label, ts_path=ts_path, dates=dates, grid=grid,
                    reliable=_reliable(coh, min_coherence if coherence_path is not None else None, e, n, u),
                    los_e=e, los_n=n, los_u=u, floor=min_coherence if coherence_path is not None else None,
                    ref_lonlat=_ref_lonlat(attrs, grid),
                    velocity_path=Path(velocity_path) if velocity_path else None,
                    source={"series": str(ts_path), "geometry": str(geometry_path),
                            "coherence": str(coherence_path) if coherence_path else None})


def _label_for(cfg) -> tuple:
    direction = str(getattr(cfg, "orbit_direction", "") or "").lower()
    track = getattr(cfg, "track", None)
    short = {"ascending": "asc", "descending": "desc"}.get(direction, direction[:4] or "run")
    backend = str(getattr(cfg, "backend", "") or "")
    sensor = "NISAR" if backend == "nisar" else "Sentinel-1"
    label = f"{short}_t{track}" if track not in (None, "") else short
    name = f"{sensor} {direction or 'unknown direction'}" + (f" track {track}" if track not in (None, "") else "")
    return label, name


def geometry_from_config(cfg, min_coherence: float | None = None) -> Geometry:
    """A run from its pipeline config, through the pipeline's own product helpers."""
    from . import deliverables as dl
    ts_path = dl.timeseries_file(cfg)
    dates, attrs, shape = _series_head(ts_path)
    grid = Grid.from_attrs(attrs, shape)
    vecs = [dl.geometry_field(cfg, k) for k in ("los_east", "los_north", "los_up")]
    if any(v is None for v in vecs):
        raise FileNotFoundError(f"No per-cell LOS geometry in {cfg.mintpy_dir} (geo/geo_geometryRadar.h5 "
                                "or geometryGeo.h5); a decomposition needs each run's own LOS vectors.")
    e, n, u = (np.asarray(v, dtype=np.float32) for v in vecs)
    if tuple(u.shape) != tuple(shape):
        raise ValueError(f"The LOS geometry is {u.shape} but the series {ts_path.name} is {tuple(shape)}: "
                         "they come from different geocoding runs.")
    floor = float(cfg.min_temporal_coherence if min_coherence is None else min_coherence)
    coh_path = dl._product(cfg, "temporalCoherence.h5")
    import h5py
    with h5py.File(coh_path, "r") as f:
        coh = f["temporalCoherence"][:]
    if coh.shape != u.shape:
        raise ValueError(f"temporal coherence is {coh.shape} but the series is {tuple(shape)}")
    label, name = _label_for(cfg)
    vel = dl._product(cfg, "velocity.h5")
    src = getattr(cfg, "_source_path", None)
    return Geometry(label=label, name=name, ts_path=Path(ts_path), dates=dates, grid=grid,
                    reliable=_reliable(coh, floor, e, n, u), los_e=e, los_n=n, los_u=u, floor=floor,
                    ref_lonlat=_ref_lonlat(attrs, grid), velocity_path=vel if vel.exists() else None,
                    source={"config": str(src) if src else None, "project_dir": str(cfg.project_dir),
                            "series": str(ts_path), "coherence": str(coh_path),
                            "backend": str(getattr(cfg, "backend", "")),
                            "orbit_direction": str(getattr(cfg, "orbit_direction", "")),
                            "track": getattr(cfg, "track", None)})


# ------------------------------------------------------------------ dates

def _d(s) -> date:
    if isinstance(s, datetime):
        return s.date()
    if isinstance(s, date):
        return s
    s = str(s).strip()
    return datetime.strptime(s.replace("-", ""), "%Y%m%d").date()


def common_window(geoms, window=None) -> tuple:
    """(first, last) date every run covers, narrowed to `window` when given."""
    d0 = max(_d(g.dates[0]) for g in geoms)
    d1 = min(_d(g.dates[-1]) for g in geoms)
    if window is not None:
        d0, d1 = max(d0, _d(window[0])), min(d1, _d(window[1]))
    if d1 <= d0:
        raise ValueError("The runs share no common window of dates"
                         + (" inside the requested window" if window is not None else "")
                         + "; Up and East cannot be separated from runs that do not overlap in time.")
    return d0, d1


def monthly_epochs(d0: date, d1: date) -> list:
    """The 1st of every month from the first one on or after d0 to d1."""
    y, m = d0.year, d0.month
    if d0.day != 1:
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    out = []
    while date(y, m, 1) <= d1:
        out.append(date(y, m, 1))
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


def interpolation_plan(dates, epoch: date, max_gap_days: int):
    """[(index, weight), ...] that interpolates a run's series linearly to
    `epoch` from its SAR dates either side, both at most max_gap_days away
    (an exact date is used as it is); None when the run has no such pair."""
    ds = [_d(x) for x in dates]
    import bisect
    i = bisect.bisect_left(ds, epoch)
    if i < len(ds) and ds[i] == epoch:
        return [(i, 1.0)]
    if i == 0 or i == len(ds):
        return None
    lo, hi = ds[i - 1], ds[i]
    if (epoch - lo).days > max_gap_days or (hi - epoch).days > max_gap_days:
        return None
    w_hi = (epoch - lo).days / (hi - lo).days
    return [(i - 1, 1.0 - w_hi), (i, w_hi)]


# ------------------------------------------------------------------ reading

class _RateFit:
    """Straight line (optionally plus an annual term) fitted to every cell
    by accumulating the normal equations one date at a time, so the series
    is read once. Only cells finite on every date of the window are fitted.
    The rate error is MintPy's residue-based one: sqrt(RSS/(n-p) * (G'G)^-1)."""

    def __init__(self, dates, d0: date, d1: date, model: str, shape):
        if model not in RATE_MODELS:
            raise ValueError(f"rate model {model!r} is not one of {RATE_MODELS}")
        self.idx = [i for i, s in enumerate(dates) if d0 <= _d(s) <= d1]
        self.n = len(self.idx)
        t = np.array([(_d(dates[i]) - d0).days / 365.25 for i in self.idx])
        tc = t - (t.mean() if t.size else 0.0)
        cols = [np.ones_like(tc), tc]
        if model == "linear_annual":
            cols += [np.sin(2 * np.pi * t), np.cos(2 * np.pi * t)]
        self.G = np.stack(cols, axis=1)
        self.p = self.G.shape[1]
        self.row = {i: k for k, i in enumerate(self.idx)}
        self.Gty = np.zeros((self.p,) + tuple(shape), np.float64)
        self.yy = np.zeros(shape, np.float64)
        self.count = np.zeros(shape, np.int32)
        self.span_yr = float(t[-1] - t[0]) if t.size else 0.0

    def add(self, i: int, y: np.ndarray) -> None:
        k = self.row.get(i)
        if k is None:
            return
        fin = np.isfinite(y)
        yv = np.where(fin, y, 0.0).astype(np.float64)
        for j in range(self.p):
            self.Gty[j] += self.G[k, j] * yv
        self.yy += yv * yv
        self.count += fin

    def solve(self) -> tuple:
        """(rate, rate 1-sigma) in the series' units per year; NaN where a
        cell missed any date of the window."""
        if self.n <= self.p:
            raise ValueError(f"only {self.n} dates in the common window; a rate needs more than {self.p}")
        Ninv = np.linalg.inv(self.G.T @ self.G)
        m = np.einsum("ij,j...->i...", Ninv, self.Gty)
        rss = np.maximum(self.yy - np.sum(m * self.Gty, axis=0), 0.0)
        s2 = rss / (self.n - self.p)
        full = self.count == self.n
        rate = np.where(full, m[1], np.nan).astype(np.float32)
        std = np.where(full, np.sqrt(s2 * Ninv[1, 1]), np.nan).astype(np.float32)
        return rate, std


class _BlockReader:
    """Reads one run's series in chunk-aligned blocks of dates, in time order,
    over the window of rows/columns the common grid needs (significance.
    iter_epochs: reading ds[i] alone decodes a whole chunk-deep block). Every
    block read is passed to the rate fit, so the file is read once."""

    def __init__(self, path: Path, window: tuple, needed: list, rate: _RateFit):
        import h5py
        self.path, self.window, self.rate = path, window, rate
        with h5py.File(path, "r") as f:
            ds = f["timeseries"]
            self.n = ds.shape[0]
            self.depth = max(1, int(ds.chunks[0])) if ds.chunks else 1
        needed = sorted(set(needed) | set(rate.idx))
        self.first_block = needed[0] // self.depth if needed else 0
        self.last_block = needed[-1] // self.depth if needed else -1
        self.cache: dict = {}
        self.loaded = self.first_block - 1

    def _load_next(self) -> None:
        import h5py
        b = self.loaded + 1
        r0, r1, c0, c1 = self.window
        i0, i1 = b * self.depth, min(self.n, (b + 1) * self.depth)
        with h5py.File(self.path, "r") as f:
            blk = f["timeseries"][i0:i1, r0:r1, c0:c1]
        for j in range(blk.shape[0]):
            self.rate.add(i0 + j, blk[j])
        self.cache[b] = (i0, blk)
        for old in [k for k in self.cache if k < b - 1]:
            del self.cache[old]
        self.loaded = b

    def get(self, i: int) -> np.ndarray:
        b = i // self.depth
        while self.loaded < b:
            self._load_next()
        if b not in self.cache:
            raise RuntimeError(f"date {i} of {self.path.name} was requested after the reader moved past it")
        i0, blk = self.cache[b]
        return blk[i - i0]

    def finish(self) -> None:
        while self.loaded < self.last_block:
            self._load_next()
        self.cache.clear()


# ------------------------------------------------------------------ reference

def _ref_cells(dst: Grid, lon: float, lat: float, radius_m: float) -> np.ndarray:
    xs, ys = dst.centres()
    dxm = (xs - lon) * M_PER_DEG_LAT * math.cos(math.radians(lat))
    dym = (ys - lat) * M_PER_DEG_LAT
    return (dym[:, None] ** 2 + dxm[None, :] ** 2) <= radius_m ** 2


def _ref_value(arr: np.ndarray, cells: np.ndarray, min_cells: int):
    v = arr[cells]
    v = v[np.isfinite(v)]
    return (float(np.median(v)), int(v.size)) if v.size >= min_cells else (None, int(v.size))


# ------------------------------------------------------------------ writing

def _tags(quantity: str, units: str, sign: str, extra: dict) -> dict:
    t = {"QUANTITY": quantity, "UNITS": units, "SIGN": sign, "NORTH": NORTH_NOTE,
         "METHOD": ("PLAN D1: per-cell least squares for Up and East from ascending and descending "
                    "line-of-sight data with each run's own per-cell LOS vector (experiment grade)")}
    t.update({k: str(v) for k, v in extra.items()})
    return t


def _write_tif(path: Path, data, grid: Grid, tags: dict, description: str, units: str) -> Path:
    from .deliverables import _write_geotiff
    _write_geotiff(path, np.asarray(data, np.float32), grid.attrs(), tags=tags,
                   description=description, units=units)
    return path


def _series_file(path: Path, grid: Grid, epochs: list, attrs: dict):
    """An open MintPy-style timeseries HDF5 (metres) to be filled epoch by epoch."""
    import h5py
    f = h5py.File(path, "w")
    k = len(epochs)
    f.create_dataset("date", data=np.array([e.strftime("%Y%m%d").encode() for e in epochs], dtype="S8"))
    f.create_dataset("bperp", data=np.zeros(k, np.float32))
    f.create_dataset("timeseries", shape=(k, grid.ny, grid.nx), dtype=np.float32,
                     chunks=(1, min(grid.ny, 256), min(grid.nx, 256)), compression="lzf",
                     fillvalue=np.nan)
    a = dict(grid.attrs())
    a.update({"FILE_TYPE": "timeseries", "UNIT": "m", "LENGTH": str(grid.ny), "WIDTH": str(grid.nx),
              "START_DATE": epochs[0].strftime("%Y%m%d"), "END_DATE": epochs[-1].strftime("%Y%m%d"),
              "REF_DATE": epochs[0].strftime("%Y%m%d"), "DECOMPOSITION_NORTH": NORTH_NOTE})
    a.update({k2: str(v) for k2, v in attrs.items()})
    for k2, v in a.items():
        f.attrs[k2] = v
    return f


# ------------------------------------------------------------------ the run

def decompose(runs, out_dir, *, window=None, epochs="monthly", bbox=None, posting_m=None,
              min_coherence: float | None = None, max_gap_days: int = 12, ref="auto",
              ref_radius_m: float = 100.0, min_ref_cells: int = 5, rate_model: str = "linear",
              min_valid_fraction: float = 0.5, max_condition: float = 10.0,
              with_los_series: bool = False) -> dict:
    """
    Decompose two or more runs (PipelineConfig objects, config paths or
    Geometry objects) into Up and East; write everything into out_dir and
    return the summary (also decomposition_summary.json).

    window      (d0, d1) to narrow the common window of dates.
    epochs      "monthly" (1st of each month), "none" (rates only), or a list
                of dates; the first epoch every run can supply is the base.
    ref         "auto" (the first run's reference point when every run has
                reliable cells within ref_radius_m of it, else none), "none",
                or (lat, lon).
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    geoms = [_as_geometry(r, min_coherence) for r in runs]
    if len(geoms) < 2:
        raise ValueError("A decomposition needs at least two runs (one ascending and one descending).")
    labels = [g.label for g in geoms]
    if len(set(labels)) != len(labels):
        raise ValueError(f"The same geometry is given twice ({', '.join(labels)}); a decomposition "
                         "needs different look directions.")
    d0, d1 = common_window(geoms, window)
    dst = common_grid([g.grid for g in geoms], bbox=bbox, posting_m=posting_m)
    log.info("Common grid: %d x %d cells, about %.0f x %.0f m, %.5f..%.5f E, %.5f..%.5f N",
             dst.nx, dst.ny, *dst.posting_m(), dst.bounds()[0], dst.bounds()[2],
             dst.bounds()[1], dst.bounds()[3])
    log.info("Common window: %s to %s", d0, d1)
    res = [_Resampler(g.grid, dst, min_valid_fraction) for g in geoms]

    def win(g, r, a):
        r0, r1, c0, c1 = r.window
        return a[r0:r1, c0:c1]

    vec = []
    rel = []
    for g, r in zip(geoms, res):
        e, n, u = r.vectors(win(g, r, g.los_e), win(g, r, g.los_n), win(g, r, g.los_u))
        vec.append((e, n, u))
        rel.append(r(np.where(win(g, r, g.reliable), 1.0, np.nan)) >= 0)   # finite = enough reliable cells
    E_ = np.stack([v[0] for v in vec])
    N_ = np.stack([v[1] for v in vec])
    U_ = np.stack([v[2] for v in vec])
    geo_check = solve_up_east(np.zeros_like(E_), E_, U_, N_, max_condition=max_condition)
    cond = geo_check["condition"]
    med_cond = float(np.nanmedian(cond)) if np.isfinite(cond).any() else float("nan")
    if not np.isfinite(med_cond) or med_cond > max_condition:
        raise ValueError(f"These runs look at the ground from directions too alike to separate Up from East "
                         f"(median condition number {med_cond:.1f}, limit {max_condition:g}). Use one "
                         "ascending and one descending run.")

    # ---- reference
    ref_info = _choose_reference(ref, geoms, dst, rel, ref_radius_m, min_ref_cells)
    cells = ref_info.pop("_cells", None)

    # ---- epochs and the read plan
    outside = []
    if isinstance(epochs, str) and epochs.lower() == "none":
        ep = []
    elif isinstance(epochs, str) and epochs.lower() == "monthly":
        ep = monthly_epochs(d0, d1)
    else:
        # explicit epochs are kept to the common window too: with --window
        # narrowing it, an epoch outside would otherwise still be decomposed
        req = sorted({_d(x) for x in epochs})
        ep = [e for e in req if d0 <= e <= d1]
        outside = [e.isoformat() for e in req if not d0 <= e <= d1]
        if outside:
            log.warning("%d requested epoch(s) fall outside the common window %s to %s and are left out: %s",
                        len(outside), d0, d1, ", ".join(outside))
    plans = [[interpolation_plan(g.dates, e, max_gap_days) for e in ep] for g in geoms]
    base = next((k for k in range(len(ep)) if all(p[k] is not None for p in plans)), None)
    if ep and base is None:
        log.warning("No epoch has a SAR date within %d days on both sides in every run; "
                    "only rates are written.", max_gap_days)
    rates = [_RateFit(g.dates, d0, d1, rate_model, r.src.shape) for g, r in zip(geoms, res)]
    for g, rf in zip(geoms, rates):
        if rf.n < 6:
            raise ValueError(f"{g.name} has only {rf.n} dates between {d0} and {d1}; a rate needs at least 6.")
        if rf.span_yr < 1.0:
            log.warning("%s spans %.2f years inside the common window: its rate is extrapolated to a "
                        "year and seasonal motion is folded into it.", g.name, rf.span_yr)
    used_ep = list(range(base, len(ep))) if base is not None else []
    before_base = [e.isoformat() for e in ep[:base if base is not None else len(ep)]]
    readers = []
    for g, r, rf, p in zip(geoms, res, rates, plans):
        need = [i for k in used_ep if p[k] for i, _ in p[k]]
        readers.append(_BlockReader(g.ts_path, r.window, need, rf))

    tagx = {"WINDOW": f"{d0} to {d1}", "RUNS": "; ".join(g.name for g in geoms),
            "REFERENCE": ref_info["text"]}
    series_info = None
    if used_ep:
        series_info = _write_series(out, geoms, res, readers, plans, [ep[k] for k in used_ep], used_ep,
                                    vec, cells, min_ref_cells, max_condition, tagx, with_los_series)
    for rd in readers:
        rd.finish()

    rate_info = _write_rates(out, geoms, res, rates, vec, cells, min_ref_cells, max_condition,
                             d0, d1, rate_model, tagx, geo_check)
    summary = _summary(geoms, dst, d0, d1, window, ref_info, rate_info, series_info, max_gap_days,
                       min_valid_fraction, max_condition, res, rate_model)
    # requested epochs that are not in the series, and why (none are dropped silently)
    summary["series"]["epochs_outside_window"] = outside
    summary["series"]["epochs_before_base_without_value"] = before_base
    (out / "decomposition_summary.json").write_text(json.dumps(summary, indent=2, default=_json_default))
    (out / "decomposition_README.txt").write_text(_readme(summary))
    log.info("Decomposition written to %s", out)
    return summary


def _as_geometry(r, min_coherence) -> Geometry:
    if isinstance(r, Geometry):
        return r
    if isinstance(r, (str, Path)):
        from .config import load_config
        r = load_config(str(r))
    return geometry_from_config(r, min_coherence)


def _choose_reference(ref, geoms, dst, rel, radius_m, min_cells) -> dict:
    if isinstance(ref, str) and ref.strip().lower() not in ("auto", "none"):
        ref = _parse_ref(ref.strip())          # "LAT,LON" through the API as on the command line
    if ref is None or (isinstance(ref, str) and ref.lower() == "none"):
        return {"mode": "none", "text": _NO_REF}
    if isinstance(ref, str) and ref.lower() == "auto":
        if geoms[0].ref_lonlat is None:
            log.warning("The first run records no reference point; no common reference is applied.")
            return {"mode": "none", "text": _NO_REF}
        lon, lat = geoms[0].ref_lonlat
        explicit = False
    else:
        lat, lon = (float(v) for v in ref)
        explicit = True
    cells = _ref_cells(dst, lon, lat, radius_m)
    counts = {g.label: int((cells & r).sum()) for g, r in zip(geoms, rel)}
    if min(counts.values()) < min_cells:
        msg = (f"Too few reliable cells within {radius_m:g} m of {lat:.5f} N, {lon:.5f} E "
               f"({', '.join(f'{k}: {v}' for k, v in counts.items())}; at least {min_cells} needed in every run)")
        if explicit:
            raise ValueError(msg + ". Choose another reference point.")
        log.warning("%s; no common reference is applied.", msg)
        return {"mode": "none", "text": _NO_REF, "tried": {"lat": lat, "lon": lon, "cells": counts}}
    how = "the point given" if explicit else f"the reference point of {geoms[0].name}"
    text = (f"every run referenced to the same ground ({how}, {lat:.5f} N {lon:.5f} E): the median of its "
            f"reliable cells within {radius_m:g} m is subtracted from each run before decomposing, per epoch "
            "for the series. Up and East are then relative to that point; where a cell's look direction "
            "differs from the point's, the point's own motion is seen slightly differently (about 1.3% of it "
            "per degree of incidence difference), so a stable reference point is best")
    return {"mode": "point", "lat": lat, "lon": lon, "radius_m": radius_m, "min_cells": min_cells,
            "cells": counts, "source": "given" if explicit else "first run's reference point",
            "text": text, "_cells": cells}


_NO_REF = ("none: each run is still relative to its own reference point, so Up and East carry an "
           "unknown constant (rates) or per-epoch (series) offset that a datum must remove")


def _referenced(arr, cells, min_cells):
    """arr minus its reference median; (array, value, n) - all NaN when the
    reference point has too few reliable cells."""
    if cells is None:
        return arr, None, None
    v, n = _ref_value(arr, cells, min_cells)
    if v is None:
        return np.full_like(arr, np.nan), None, n
    return arr - np.float32(v), v, n


def _write_series(out, geoms, res, readers, plans, epochs, ks, vec, cells, min_cells, max_condition,
                  tagx, with_los):
    E_ = np.stack([v[0] for v in vec])
    N_ = np.stack([v[1] for v in vec])
    U_ = np.stack([v[2] for v in vec])
    dst = res[0].dst
    extra = {"DECOMPOSITION_REFERENCE": tagx["REFERENCE"], "DECOMPOSITION_RUNS": tagx["RUNS"],
             "DECOMPOSITION_SIGN_UP": UP_SIGN, "DECOMPOSITION_SIGN_EAST": EAST_SIGN,
             "DECOMPOSITION_UNCERTAINTY": SERIES_SIGMA_NOTE}
    paths = {"vertical": out / "vertical_timeseries_decomposed.h5",
             "east_west": out / "east_west_timeseries.h5"}
    tmp = {k: p.with_name(p.name + ".partial") for k, p in paths.items()}
    files = {"vertical": _series_file(tmp["vertical"], dst, epochs,
                                      {**extra, "DECOMPOSITION_QUANTITY": "vertical displacement (Up)"}),
             "east_west": _series_file(tmp["east_west"], dst, epochs,
                                       {**extra, "DECOMPOSITION_QUANTITY": "east-west displacement (East)"})}
    los_files = {}
    if with_los:
        for g in geoms:
            p = out / f"los_change_{g.label}_common_grid.h5"
            tmp[g.label] = p.with_name(p.name + ".partial")
            paths[g.label] = p
            los_files[g.label] = _series_file(tmp[g.label], dst, epochs, {
                "DECOMPOSITION_QUANTITY": f"LOS change of {g.name}, positive toward the satellite, "
                                          "interpolated to the common epochs on the common grid",
                "DECOMPOSITION_REFERENCE": tagx["REFERENCE"]})
    per_epoch, base_vals = [], None
    last_up = last_east = None
    last_epoch = None
    done = False
    try:
        for j, (k, ep) in enumerate(zip(ks, epochs)):
            vals = []
            for g, r, rd, p in zip(geoms, res, readers, plans):
                pl = p[k]
                if pl is None:
                    vals.append(None)
                    continue
                v = sum(np.float32(w) * rd.get(i).astype(np.float32) for i, w in pl)
                r0, r1, c0, c1 = r.window
                v = np.where(g.reliable[r0:r1, c0:c1], v, np.nan)
                vals.append(r(v) * np.float32(1000.0))
            if j == 0:
                base_vals = vals
            row = {"epoch": ep.isoformat(), "runs_with_value": [], "reference_cells": {}}
            los = []
            for g, v, b in zip(geoms, vals, base_vals):
                if v is None:
                    los.append(np.full(dst.shape, np.nan, np.float32))
                    continue
                row["runs_with_value"].append(g.label)
                ch, rv, n = _referenced(v - b, cells, min_cells)
                if cells is not None:
                    row["reference_cells"][g.label] = n
                los.append(ch)
            L = np.stack(los)
            sol = solve_up_east(L, E_, U_, N_, max_condition=max_condition)
            files["vertical"]["timeseries"][j] = sol["up"] / 1000.0
            files["east_west"]["timeseries"][j] = sol["east"] / 1000.0
            for g, arr in zip(geoms, los):
                if g.label in los_files:
                    los_files[g.label]["timeseries"][j] = arr / 1000.0
            row["cells_solved"] = int(np.isfinite(sol["up"]).sum())
            per_epoch.append(row)
            if row["cells_solved"]:
                last_up, last_east, last_epoch = sol["up"], sol["east"], ep
            log.info("Epoch %s: %d cells solved", ep, row["cells_solved"])
        done = True
    finally:
        for f in list(files.values()) + list(los_files.values()):
            f.close()
        if not done:                      # never leave a half-written series behind
            for t in tmp.values():
                t.unlink(missing_ok=True)
    for key, p in paths.items():
        os.replace(tmp[key], p)
    out_files = {k: str(p) for k, p in paths.items()}
    if last_up is not None and last_epoch != epochs[0]:
        a, b = epochs[0].strftime("%Y%m%d"), last_epoch.strftime("%Y%m%d")
        t = {"WINDOW": f"{epochs[0]} to {last_epoch}", "RUNS": tagx["RUNS"], "REFERENCE": tagx["REFERENCE"],
             "UNCERTAINTY": SERIES_SIGMA_NOTE}
        p1 = _write_tif(out / f"vertical_change_mm_{a}_{b}_decomposed.tif", last_up, dst,
                        _tags("vertical change (Up), decomposed", "mm", UP_SIGN, t),
                        f"Vertical change {epochs[0]} to {last_epoch}, mm, from ascending + descending; "
                        "north not estimated", "mm")
        p2 = _write_tif(out / f"east_west_change_mm_{a}_{b}.tif", last_east, dst,
                        _tags("east-west change (East), decomposed", "mm", EAST_SIGN, t),
                        f"East-west change {epochs[0]} to {last_epoch}, mm, positive east; "
                        "north not estimated", "mm")
        out_files["vertical_change"], out_files["east_west_change"] = str(p1), str(p2)
    missing = [r["epoch"] for r in per_epoch if r["cells_solved"] == 0]
    return {"epochs": [e.isoformat() for e in epochs], "base_epoch": epochs[0].isoformat(),
            "epochs_without_value": missing, "per_epoch": per_epoch, "files": out_files,
            "units": "metres in the HDF5 files (MintPy convention); mm in the GeoTIFFs",
            "uncertainty": SERIES_SIGMA_NOTE}


def _write_rates(out, geoms, res, rates, vec, cells, min_cells, max_condition, d0, d1, model, tagx,
                 geo_check) -> dict:
    dst = res[0].dst
    los, sig, per_run = [], [], {}
    for gi, (g, r, rf) in enumerate(zip(geoms, res, rates)):
        rate, std = rf.solve()
        r0, r1, c0, c1 = r.window
        relw = g.reliable[r0:r1, c0:c1]
        rate = np.where(relw, rate, np.nan) * np.float32(1000.0)
        std = np.where(relw & np.isfinite(rate), std, np.nan) * np.float32(1000.0)
        check = _velocity_check(g, r, rate, _d(g.dates[rf.idx[0]]), _d(g.dates[rf.idx[-1]]))
        rate_c, std_c = r(rate), r(std)
        std_c[~np.isfinite(rate_c)] = np.nan
        rate_c, rv, n = _referenced(rate_c, cells, min_cells)
        if cells is not None and rv is None:
            raise ValueError(f"The reference point has too few reliable rate cells in {g.name} ({n}).")
        los.append(rate_c)
        sig.append(std_c)
        per_run[g.label] = {"name": g.name, "dates_in_window": rf.n,
                            "first_date": _d(g.dates[rf.idx[0]]).isoformat(),
                            "last_date": _d(g.dates[rf.idx[-1]]).isoformat(),
                            "span_years": round(rf.span_yr, 3), "reference_rate_removed_mm_yr": rv,
                            "reference_cells": n, "check_against_run_velocity": check,
                            "median_los_vector_enu": [_med(v) for v in vec[gi]],
                            "reliable_cells_on_common_grid": int(np.isfinite(rate_c).sum())}
    E_ = np.stack([v[0] for v in vec])
    N_ = np.stack([v[1] for v in vec])
    U_ = np.stack([v[2] for v in vec])
    sol = solve_up_east(np.stack(los), E_, U_, N_, sigma=np.stack(sig), max_condition=max_condition)
    a, b = d0.strftime("%Y%m%d"), d1.strftime("%Y%m%d")
    t = {"WINDOW": f"{d0} to {d1}", "RUNS": tagx["RUNS"], "REFERENCE": tagx["REFERENCE"],
         "RATE_MODEL": _model_text(model)}
    files = {}
    files["vertical_velocity"] = _write_tif(
        out / f"vertical_velocity_mm_per_year_{a}_{b}_decomposed.tif", sol["up"], dst,
        _tags("vertical velocity (Up), decomposed", "mm/yr", UP_SIGN, t),
        "Vertical velocity, mm/yr, from ascending + descending; north not estimated", "mm/yr")
    files["vertical_velocity_uncertainty"] = _write_tif(
        out / f"vertical_velocity_uncertainty_1sigma_mm_per_year_{a}_{b}_decomposed.tif", sol["sigma_up"], dst,
        _tags("FORMAL 1-sigma of the decomposed vertical velocity (each run's residue-based rate error "
              "propagated; assumes independent dates - a floor, not the uncertainty)", "mm/yr",
              "unsigned (a 1-sigma uncertainty)", t),
        "Formal 1-sigma of the decomposed vertical velocity, mm/yr: a floor", "mm/yr")
    files["east_west_velocity"] = _write_tif(
        out / f"east_west_velocity_mm_per_year_{a}_{b}.tif", sol["east"], dst,
        _tags("east-west velocity (East), decomposed", "mm/yr", EAST_SIGN, t),
        "East-west velocity, mm/yr, positive east, from ascending + descending; north not estimated", "mm/yr")
    files["east_west_velocity_uncertainty"] = _write_tif(
        out / f"east_west_velocity_uncertainty_1sigma_mm_per_year_{a}_{b}.tif", sol["sigma_east"], dst,
        _tags("FORMAL 1-sigma of the decomposed east-west velocity (a floor)", "mm/yr",
              "unsigned (a 1-sigma uncertainty)", t),
        "Formal 1-sigma of the decomposed east-west velocity, mm/yr: a floor", "mm/yr")
    files["condition_number"] = _write_tif(
        out / "decomposition_condition_number.tif", geo_check["condition"], dst,
        _tags("condition number of the per-cell design matrix (1 = ideal; large = the look directions "
              "are too alike to separate Up from East)", "1", "not applicable", t),
        "Condition number of the Up/East solve per cell", "1")
    files["north_leak_into_vertical"] = _write_tif(
        out / "north_leak_into_vertical.tif", geo_check["leak_up"], dst,
        _tags("error in Up per unit of northward motion (north is not estimated)", "mm per mm",
              "a factor: Up error = factor x northward motion", t),
        "North leak into Up: mm of Up error per mm of northward motion", "mm per mm")
    files["north_leak_into_east_west"] = _write_tif(
        out / "north_leak_into_east_west.tif", geo_check["leak_east"], dst,
        _tags("error in East per unit of northward motion (north is not estimated)", "mm per mm",
              "a factor: East error = factor x northward motion", t),
        "North leak into East: mm of East error per mm of northward motion", "mm per mm")
    layers = out / "decomposition_layers.h5"
    tmp = layers.with_name(layers.name + ".partial")
    try:
        _write_layers(tmp, dst, d0, d1, tagx, model, sol, geo_check, geoms, los, sig, vec)
    except BaseException:
        tmp.unlink(missing_ok=True)       # never leave a half-written layers file behind
        raise
    os.replace(tmp, layers)
    files["layers"] = layers
    solved = np.isfinite(sol["up"])
    return {"model": _model_text(model), "files": {k: str(v) for k, v in files.items()},
            "cells_solved": int(solved.sum()), "share_of_grid_solved": round(float(solved.mean()), 4),
            "median_vertical_mm_yr": _med(sol["up"]), "median_east_mm_yr": _med(sol["east"]),
            "median_sigma_vertical_mm_yr": _med(sol["sigma_up"]),
            "median_sigma_east_mm_yr": _med(sol["sigma_east"]),
            "median_condition_number": _med(geo_check["condition"]),
            "median_north_leak_into_vertical": _med(geo_check["leak_up"]),
            "median_north_leak_into_east": _med(geo_check["leak_east"]),
            "median_amplification_vertical": _med(geo_check["amplification_up"]),
            "median_amplification_east": _med(geo_check["amplification_east"]),
            "runs": per_run,
            "uncertainty": ("formal: each run's residue-based rate error (as MintPy's velocityStd; "
                            "assumes independent dates, so a floor), averaged (not reduced) where source "
                            "cells are combined on the common grid, then propagated through the solve; "
                            "the reference point's own error is not included")}


def _write_layers(path, dst, d0, d1, tagx, model, sol, geo_check, geoms, los, sig, vec) -> None:
    import h5py
    with h5py.File(path, "w") as f:
        for k2, v in dst.attrs().items():
            f.attrs[k2] = v
        f.attrs["NORTH"] = NORTH_NOTE
        f.attrs["UNITS"] = "mm/yr for rates; unit vectors and factors are dimensionless"
        f.attrs["WINDOW"] = f"{d0} to {d1}"
        f.attrs["REFERENCE"] = tagx["REFERENCE"]
        f.attrs["RATE_MODEL"] = _model_text(model)
        for k2, src in (("vertical_velocity", "up"), ("vertical_velocity_std", "sigma_up"),
                        ("east_velocity", "east"), ("east_velocity_std", "sigma_east")):
            f.create_dataset(k2, data=sol[src], compression="lzf")
        for k2, src in (("condition_number", "condition"), ("north_leak_vertical", "leak_up"),
                        ("north_leak_east", "leak_east"), ("amplification_vertical", "amplification_up"),
                        ("amplification_east", "amplification_east")):
            f.create_dataset(k2, data=geo_check[src], compression="lzf")
        for g, v, s, (e, n, u) in zip(geoms, los, sig, vec):
            f.create_dataset(f"los_velocity_{g.label}", data=v, compression="lzf")
            f.create_dataset(f"los_velocity_std_{g.label}", data=s, compression="lzf")
            for comp, arr in (("east", e), ("north", n), ("up", u)):
                f.create_dataset(f"los_{comp}_{g.label}", data=arr.astype(np.float32), compression="lzf")


def _velocity_check(g: Geometry, r: _Resampler, rate_mm, d0, d1):
    """Median |refitted rate - the run's own velocity| over reliable cells,
    when the run's velocity was fitted from the same first and last date as
    the refit (d0, d1: the run's own first and last date inside the common
    window). A check that the refit reads the same series, not a correction."""
    if g.velocity_path is None or not Path(g.velocity_path).exists():
        return None
    import h5py
    try:
        with h5py.File(g.velocity_path, "r") as f:
            a = {k: (v.decode() if isinstance(v, bytes) else v) for k, v in f.attrs.items()}
            if _d(a.get("START_DATE", "19000101")) != d0 or _d(a.get("END_DATE", "19000101")) != d1:
                return {"compared": False, "why": "the run's velocity covers "
                        f"{a.get('START_DATE')} to {a.get('END_DATE')}, the refit {d0} to {d1}"}
            r0, r1, c0, c1 = r.window
            v = f["velocity"][r0:r1, c0:c1].astype(np.float32) * 1000.0
    except (OSError, KeyError, ValueError) as exc:
        return {"compared": False, "why": f"could not read it ({exc})"}
    if v.shape != rate_mm.shape:
        return {"compared": False, "why": "the run's velocity is on another grid"}
    dlt = np.abs(rate_mm - v)
    dlt = dlt[np.isfinite(dlt)]
    if not dlt.size:
        return {"compared": False, "why": "no common reliable cells"}
    return {"compared": True, "median_abs_difference_mm_yr": round(float(np.median(dlt)), 4),
            "p99_abs_difference_mm_yr": round(float(np.percentile(dlt, 99)), 4)}


def _model_text(model: str) -> str:
    return {"linear": "straight line (offset + rate) over the common window, each run on its own dates",
            "linear_annual": ("straight line plus an annual sine and cosine over the common window, each "
                              "run on its own dates")}[model]


def _med(a):
    a = np.asarray(a, dtype=np.float64)
    a = a[np.isfinite(a)]
    return round(float(np.median(a)), 4) if a.size else None


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (date, datetime)):
        return o.isoformat()
    if isinstance(o, Path):
        return str(o)
    return str(o)


def _summary(geoms, dst, d0, d1, window, ref_info, rate_info, series_info, max_gap_days,
             min_valid_fraction, max_condition, res, model) -> dict:
    try:
        from .version import __version__, code_hash
        software = {"insar_pipeline": __version__, "code_hash": code_hash()}
    except Exception as exc:                       # provenance must not stop the result  # noqa: BLE001
        software = {"error": str(exc)}
    runs = []
    for g, r in zip(geoms, res):
        runs.append({"label": g.label, "name": g.name, **g.source,
                     "coherence_floor": g.floor, "dates": len(g.dates),
                     "first_date": _d(g.dates[0]).isoformat(), "last_date": _d(g.dates[-1]).isoformat(),
                     "run_reference_point_lon_lat": g.ref_lonlat,
                     "grid_crs": g.grid.crs, "posting_m": [round(x, 2) for x in g.grid.posting_m()],
                     "resampling": ("none: cell for cell (the common grid is aligned with this run)"
                                    if r.identity else
                                    f"average of the reliable source cells in each common cell "
                                    f"(at least {min_valid_fraction:.0%} of them reliable)")})
    W, S, E, N = dst.bounds()
    return {
        "what": ("Up and East from ascending + descending line-of-sight data (PLAN D1, experiment grade): "
                 "per-cell least squares with each run's own per-cell LOS vector on one lat/lon grid"),
        "north": NORTH_NOTE,
        "sign": {"up": UP_SIGN, "east": EAST_SIGN,
                 "los_input": "LOS positive toward the satellite (MintPy): uplift positive"},
        "runs": runs,
        "common_window": [d0.isoformat(), d1.isoformat()],
        "window_rule": ("the dates every run covers" + (f", narrowed to {window[0]} .. {window[1]}"
                                                        if window else "")),
        "grid": {"crs": "EPSG:4326", "west": W, "south": S, "east": E, "north": N,
                 "lon_step_deg": dst.dx, "lat_step_deg": dst.dy, "width": dst.nx, "length": dst.ny,
                 "posting_m_approx": [round(x, 2) for x in dst.posting_m()]},
        "reliability": ("a cell is used when its value is finite and its temporal coherence is at or above "
                        "the run's floor (coherence_floor); on the common grid at least "
                        f"{min_valid_fraction:.0%} of the source cells must be reliable; a cell missing any "
                        "date of the common window gets no rate"),
        "max_condition_number": max_condition,
        "reference": {k: v for k, v in ref_info.items() if not k.startswith("_")},
        "rates": rate_info,
        "series": series_info if series_info else {"written": False,
                                                   "why": "no epochs requested or none with data in every run"},
        "series_rule": (f"each run interpolated linearly to the epochs from its SAR dates either side, both at "
                        f"most {max_gap_days} days away; change since the first epoch every run has; "
                        "different-day acquisitions are never treated as simultaneous"),
        "not_done": [
            "North is not estimated (see 'north').",
            "No per-epoch uncertainty for the series (see series.uncertainty).",
            "The protocol's 'at least 2 independent radar pixels' rule at the reference point is not applied "
            "here; the reference uses the median of at least the stated number of reliable grid cells.",
            "No GNSS datum: Up and East are relative to the reference point (or to each run's own reference "
            "point with reference 'none'); the validation harness applies the protocol's datum.",
        ] + ([
            "With more than two runs the rate solve weights each run by its rate error, while the north leak, "
            "condition number and amplification written are for the unweighted solve (as the series uses); "
            "for the rates they are close but not exact."] if len(geoms) > 2 else []),
        "software": software,
        "written": datetime.now().isoformat(timespec="seconds"),
    }


def _readme(s: dict) -> str:
    r = s["rates"]
    lines = [
        "Up and East from ascending + descending runs (experiment grade)",
        "",
        "What this folder holds",
        "  Vertical (Up) and east-west (East) motion worked out from two look directions at the same ground.",
        "  " + s["north"],
        "",
        "Runs used",
    ]
    for run in s["runs"]:
        lines.append(f"  - {run['name']} ({run['label']}): {run['dates']} dates, {run['first_date']} to "
                     f"{run['last_date']}; coherence floor {run['coherence_floor']}; {run['resampling']}.")
    lines += [
        "",
        f"Common window: {s['common_window'][0]} to {s['common_window'][1]} ({s['window_rule']}).",
        f"Grid: latitude/longitude, about {s['grid']['posting_m_approx'][0]:.0f} m x "
        f"{s['grid']['posting_m_approx'][1]:.0f} m cells, {s['grid']['width']} x {s['grid']['length']}.",
        f"Reference: {s['reference']['text']}.",
        "",
        "Rates",
        f"  Model: {r['model']}.",
        f"  Cells solved: {r['cells_solved']} ({r['share_of_grid_solved']:.0%} of the grid).",
        f"  Median condition number {r['median_condition_number']} (1 is ideal); median north leak into Up "
        f"{r['median_north_leak_into_vertical']} mm per mm of northward motion.",
        f"  Uncertainty: {r['uncertainty']}.",
        "",
        "Time series",
    ]
    ser = s["series"]
    if ser.get("epochs"):
        lines += [f"  {len(ser['epochs'])} epochs from {ser['base_epoch']} (the base, zero) to {ser['epochs'][-1]}; "
                  f"{len(ser['epochs_without_value'])} without a value.",
                  f"  {s['series_rule']}.",
                  f"  Uncertainty: {ser['uncertainty']}",
                  "  The HDF5 series are in metres (MintPy layout: datasets timeseries and date)."]
    else:
        lines.append(f"  Not written: {ser.get('why')}.")
    if ser.get("epochs_outside_window"):
        lines.append(f"  Left out, outside the common window: {', '.join(ser['epochs_outside_window'])}.")
    if ser.get("epochs_before_base_without_value"):
        lines.append("  Left out, before the first epoch every run has a value for: "
                     f"{', '.join(ser['epochs_before_base_without_value'])}.")
    lines += ["", "Signs",
              f"  Up: {s['sign']['up']}", f"  East: {s['sign']['east']}",
              "", "Files"]
    for k, v in sorted(r["files"].items()):
        lines.append(f"  {Path(v).name}")
    for k, v in sorted((ser.get("files") or {}).items()):
        lines.append(f"  {Path(v).name}")
    lines += ["  decomposition_summary.json (everything above, with the numbers)", ""]
    return "\n".join(lines)


# ------------------------------------------------------------------ CLI

def _parse_epochs(v: str):
    v = v.strip()
    if v.lower() in ("monthly", "none"):
        return v.lower()
    if v.startswith("@"):
        txt = Path(v[1:]).read_text()
        lines = (ln.split("#", 1)[0] for ln in txt.splitlines())       # '#' starts a comment
        return [x.strip() for ln in lines for x in ln.split(",") if x.strip()]
    return [x.strip() for x in v.split(",") if x.strip()]


def _parse_ref(v: str):
    if v.lower() in ("auto", "none"):
        return v.lower()
    try:
        lat, lon = (float(x) for x in v.split(","))
    except ValueError:
        raise argparse.ArgumentTypeError("--ref takes auto, none or LAT,LON") from None
    return (lat, lon)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m insar_pipeline.decompose",
        description=("Work out vertical (Up) and east-west (East) motion from an ascending and a "
                     "descending run of the same ground. North-south motion is not estimated."))
    ap.add_argument("configs", nargs="+", help="two or more run configs: one ascending, one descending")
    ap.add_argument("--out", required=True, help="folder to write into")
    ap.add_argument("--window", nargs=2, metavar=("FIRST", "LAST"),
                    help="narrow the common window of dates (YYYY-MM-DD YYYY-MM-DD)")
    ap.add_argument("--epochs", default="monthly", type=_parse_epochs,
                    help="monthly (default, the 1st of each month), none (rates only), a comma list of "
                         "dates, or @file with one date per line")
    ap.add_argument("--ref", default="auto", type=_parse_ref,
                    help="auto (the first run's reference point, default), none, or LAT,LON")
    ap.add_argument("--ref-radius-m", type=float, default=100.0)
    ap.add_argument("--min-ref-cells", type=int, default=5)
    ap.add_argument("--bbox", nargs=4, type=float, metavar=("W", "S", "E", "N"),
                    help="cut the common grid to this box (degrees)")
    ap.add_argument("--posting-m", type=float, default=None,
                    help="common grid cell size in metres (default: the coarser run's posting)")
    ap.add_argument("--min-coherence", type=float, default=None,
                    help="temporal coherence floor (default: each config's min_temporal_coherence)")
    ap.add_argument("--max-gap-days", type=int, default=12)
    ap.add_argument("--rate-model", choices=RATE_MODELS, default="linear")
    ap.add_argument("--min-valid-fraction", type=float, default=0.5)
    ap.add_argument("--max-condition", type=float, default=10.0)
    ap.add_argument("--with-los-series", action="store_true",
                    help="also write each run's LOS change on the common grid and epochs")
    a = ap.parse_args(argv)
    if len(a.configs) < 2:
        ap.error("give at least two configs (one ascending, one descending)")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    s = decompose(a.configs, a.out, window=a.window, epochs=a.epochs, bbox=a.bbox, posting_m=a.posting_m,
                  min_coherence=a.min_coherence, max_gap_days=a.max_gap_days, ref=a.ref,
                  ref_radius_m=a.ref_radius_m, min_ref_cells=a.min_ref_cells, rate_model=a.rate_model,
                  min_valid_fraction=a.min_valid_fraction, max_condition=a.max_condition,
                  with_los_series=a.with_los_series)
    r = s["rates"]
    print(f"Up and East written to {a.out}")
    print(f"  window {s['common_window'][0]} to {s['common_window'][1]}; {r['cells_solved']} cells solved; "
          f"median condition number {r['median_condition_number']}")
    print(f"  {NORTH_NOTE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
