"""
Slope geometry layers (step S1 of review_2026-11-07/PLAN_oil_gas_slopes.md):
where this satellite can and cannot see slope movement.

From the kept geometry file (mintpy/geo/geo_geometryRadar.h5, or DISP-S1's
geometryGeo.h5) on the product grid:

  * slope and aspect from the DEM, and the downslope unit vector;
  * |LOS . downslope|, the share of straight-downslope movement the line of
    sight sees;
  * layover and shadow recomputed from the DEM and the per-cell look geometry
    (the kept shadowMask is used as well, and cross-checked);
  * a visibility class per cell.

Conventions
-----------
  * Rows run north to south (Y_STEP < 0, as every MintPy geocoded product),
    columns west to east. Positions are metres on a local tangent plane.
  * LOS unit vector (east, north, up) points from the ground TOWARD the
    satellite, the convention of deliverables.geometry_field and
    decompose: east = -sin(inc) sin(az), north = sin(inc) cos(az),
    up = cos(inc), with MintPy's azimuthAngle.
  * Aspect is the azimuth of the DOWNHILL direction, degrees clockwise from
    north. The downslope unit vector is (cos s sin A, cos s cos A, -sin s).

Layover and shadow (the discrete definition, which the S1 gate's
independent recomputation reproduces)
-----------------------------------------------------------------------
  * The DEM is the bilinear surface through the cell centres. For each cell
    P the look direction g is the horizontal projection of -LOS at P (away
    from the satellite) and the step is delta = LOOK_STEP_CELLS times the
    smaller grid spacing.
  * LAYOVER (active): the slant range decreases going away from the radar,
    over P - delta g .. P + delta g: 2 delta sin(inc) - cos(inc) (h+ - h-) < 0,
    i.e. the terrain slope toward the radar along g exceeds the incidence.
    Passive layover (the ground elsewhere in the same slant-range cell, which
    receives the overlaid return) is NOT flagged.
  * SHADOW: the ray from P toward the satellite (rising cot(inc) per metre of
    ground) passes below the terrain at some P - k delta g, k = 1, 2, ...
    This covers both the back slope steeper than 90 deg - incidence and the
    ground hidden behind a ridge.

Approximations and limits
-------------------------
  * Parallel rays per cell and flat earth: each cell's ray uses that cell's
    own incidence and azimuth along its whole length (the change over a few
    km of ray is < 0.5 deg); Earth curvature over the ray (< 0.1 m at 1 km)
    is ignored. Azimuth and incidence are therefore handled per cell, not per
    row.
  * Geographic grids: east spacing is computed per row from the WGS84 prime
    vertical radius at that row's latitude; north spacing from the meridian
    radius at the scene centre. Projected (UTM) grids take the steps as
    metres, and grid north is treated as true north (up to ~2 deg off near a
    zone edge, which shifts aspect by the same amount).
  * Heights are ellipsoidal (GLO-30 as kept): fine for local slopes. The DEM
    resolution limits everything: slopes are the 2-cell central difference
    (np.gradient), so cliffs narrower than two cells are smoothed.
  * The kept shadowMask (ISCE, in radar geometry, then geocoded) and the
    recomputed shadow will not match cell for cell; their agreement is
    reported, and either one marks a cell not visible.
"""
from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np

log = logging.getLogger(__name__)

# --- constants (pre-registered with Gate S1; do not tune on results) -------
FLAT_ASPECT_DEG = 0.5        # below this slope the aspect is undefined (NaN)
FLAT_SLOPE_DEG = 5.0         # below this slope motion is not assessed (class 4)
POOR_MAX = 0.3               # |LOS . downslope| < 0.3 -> poor
FAIR_MAX = 0.6               # 0.3 <= s <= 0.6 -> fair, > 0.6 -> good
LOOK_STEP_CELLS = 1.0        # layover/shadow step along the look, in cells

CLASS_NOT_VISIBLE = 0
CLASS_POOR = 1
CLASS_FAIR = 2
CLASS_GOOD = 3
CLASS_FLAT = 4
CLASS_NODATA = 255
CLASS_NAMES = {
    CLASS_NOT_VISIBLE: "not visible (layover or shadow)",
    CLASS_POOR: "poor (|LOS.downslope| < 0.3)",
    CLASS_FAIR: "fair (0.3-0.6)",
    CLASS_GOOD: "good (> 0.6)",
    CLASS_FLAT: "flat (slope < 5 deg, slope motion not assessed)",
}

WGS84_A = 6378137.0
WGS84_E2 = 6.69437999014e-3

SUMMARY_NAME = "slope_visibility_summary.json"


# --- geometry helpers -------------------------------------------------------
def los_from_angles(incidence_deg, azimuth_deg):
    """(east, north, up) toward the satellite from MintPy incidenceAngle and
    azimuthAngle, the same conversion as deliverables.geometry_field (an
    incidence of 0 is MintPy's fill value outside the swath -> NaN)."""
    inc = np.radians(np.asarray(incidence_deg, dtype=np.float64))
    az = np.radians(np.asarray(azimuth_deg, dtype=np.float64))
    inc = np.where(inc == 0, np.nan, inc)
    return -np.sin(inc) * np.sin(az), np.sin(inc) * np.cos(az), np.cos(inc)


def grid_spacing_m(attrs: dict, shape):
    """(dx_m, dy_m): east spacing (scalar, or (rows, 1) per-row array on a
    geographic grid) and north spacing (scalar), both positive metres."""
    xs, ys = float(attrs["X_STEP"]), float(attrs["Y_STEP"])
    if ys > 0:
        raise ValueError("slope_geometry expects rows running north to south (Y_STEP < 0).")
    if abs(xs) >= 0.01:                                    # projected, metres
        return abs(xs), abs(ys)
    rows = shape[0]
    lat = float(attrs["Y_FIRST"]) + (np.arange(rows) + 0.5) * ys
    phi = np.radians(lat)
    s2 = np.sin(phi) ** 2
    n_rad = WGS84_A / np.sqrt(1 - WGS84_E2 * s2)
    dx = (np.radians(abs(xs)) * n_rad * np.cos(phi))[:, None]
    phic = math.radians(float(np.mean(lat)))
    m_rad = WGS84_A * (1 - WGS84_E2) / (1 - WGS84_E2 * math.sin(phic) ** 2) ** 1.5
    return dx, float(math.radians(abs(ys)) * m_rad)


def _bilinear(h, rf, cf):
    """Bilinear sample of h at fractional (row, col); NaN outside the grid
    (beyond the outermost cell centres) or where a corner is NaN."""
    nr, nc = h.shape
    inside = (rf >= 0) & (rf <= nr - 1) & (cf >= 0) & (cf <= nc - 1)
    rfc = np.where(inside, rf, 0.0)
    cfc = np.where(inside, cf, 0.0)
    r0 = np.clip(np.floor(rfc).astype(np.int64), 0, max(nr - 2, 0))
    c0 = np.clip(np.floor(cfc).astype(np.int64), 0, max(nc - 2, 0))
    fr, fc = rfc - r0, cfc - c0
    r1, c1 = np.minimum(r0 + 1, nr - 1), np.minimum(c0 + 1, nc - 1)
    v = (h[r0, c0] * (1 - fr) * (1 - fc) + h[r1, c0] * fr * (1 - fc)
         + h[r0, c1] * (1 - fr) * fc + h[r1, c1] * fr * fc)
    return np.where(inside, v, np.nan)


# --- the layers -------------------------------------------------------------
def slope_aspect(height, dx_m, dy_m):
    """
    (slope_deg, aspect_deg, (down_e, down_n, down_u)).

    Slope from central differences; aspect = azimuth of the downhill
    direction, deg clockwise from north (NaN where slope < FLAT_ASPECT_DEG);
    downslope unit vector (cos s sin A, cos s cos A, -sin s), whose
    horizontal part is NaN where the aspect is.
    """
    h = np.asarray(height, dtype=np.float64)
    d_rows, d_cols = np.gradient(h)
    dzdx = d_cols / dx_m                    # rise per metre east
    dzdy = -d_rows / dy_m                   # rise per metre north (rows run south)
    grad = np.hypot(dzdx, dzdy)
    slope = np.degrees(np.arctan(grad))
    flat = ~(slope >= FLAT_ASPECT_DEG)
    with np.errstate(invalid="ignore", divide="ignore"):
        aspect = np.mod(np.degrees(np.arctan2(-dzdx, -dzdy)), 360.0)
        he, hn = -dzdx / grad, -dzdy / grad
    aspect[flat] = np.nan
    he[flat] = np.nan
    hn[flat] = np.nan
    s = np.radians(slope)
    return slope, aspect, (np.cos(s) * he, np.cos(s) * hn, -np.sin(s))


def los_downslope_sensitivity(los_enu, downslope_enu):
    """|LOS . downslope| per cell (both unit vectors)."""
    le, ln, lu = (np.asarray(a, dtype=np.float64) for a in los_enu)
    de, dn, du = (np.asarray(a, dtype=np.float64) for a in downslope_enu)
    return np.abs(le * de + ln * dn + lu * du)


def layover_shadow(height, los_enu, dx_m, dy_m):
    """
    (layover, shadow) boolean masks from the DEM and the per-cell LOS unit
    vector toward the satellite (see the module docstring for the exact
    definition and its limits). `los_enu` may be broadcastable scalars.
    """
    h = np.asarray(height, dtype=np.float64)
    shape = h.shape
    e, n, u = (np.broadcast_to(np.asarray(a, dtype=np.float64), shape) for a in los_enu)
    inc = np.arccos(np.clip(u, -1.0, 1.0))
    hz = np.hypot(e, n)
    with np.errstate(invalid="ignore", divide="ignore"):
        ge, gn = -e / hz, -n / hz                    # ground look direction (away from radar)
        cot = np.cos(inc) / np.sin(inc)
    dx = np.broadcast_to(np.asarray(dx_m, dtype=np.float64), (shape[0], 1))
    step = LOOK_STEP_CELLS * min(float(np.nanmin(dx)), float(dy_m))
    rows, cols = np.indices(shape, dtype=np.float64)
    dcol = step * ge / dx                            # cells per +step along g
    drow = -step * gn / dy_m

    hp = _bilinear(h, rows + drow, cols + dcol)
    hm = _bilinear(h, rows - drow, cols - dcol)
    with np.errstate(invalid="ignore"):
        dslant = 2 * step * np.sin(inc) - np.cos(inc) * (hp - hm)
        layover = dslant < 0

    shadow = np.zeros(shape, dtype=bool)
    ok = np.isfinite(h) & np.isfinite(cot) & (cot > 0)
    if ok.any():
        relief = float(np.nanmax(h[ok]) - np.nanmin(h[ok]))
        kmax = int(math.ceil(relief / (step * float(np.min(cot[ok]))))) + 1
        hmax = float(np.nanmax(h))
        cand = np.flatnonzero(ok)
        hc, cotc = h.ravel()[cand], cot.ravel()[cand]
        rc, cc = rows.ravel()[cand], cols.ravel()[cand]
        drc, dcc = drow.ravel()[cand], dcol.ravel()[cand]
        hit = np.zeros(cand.size, dtype=bool)
        for k in range(1, kmax + 1):
            ray = hc + k * step * cotc
            live = (~hit) & (ray <= hmax)
            if not live.any():
                break
            q = _bilinear(h, rc[live] - k * drc[live], cc[live] - k * dcc[live])
            with np.errstate(invalid="ignore"):
                hit[np.flatnonzero(live)[q > ray[live]]] = True
        shadow.flat[cand[hit]] = True
    return layover, shadow


def visibility_class(slope_deg, sensitivity, layover, shadow, kept_shadow=None, valid=None):
    """
    uint8 class per cell: 0 not visible (layover, recomputed shadow or the
    kept shadowMask; takes precedence), 4 flat (slope < 5 deg, slope motion
    not assessed), else 1 poor (< 0.3), 2 fair (0.3-0.6), 3 good (> 0.6).
    255 outside `valid` (default: finite slope) or where the sensitivity is
    undefined.
    """
    slope = np.asarray(slope_deg, dtype=np.float64)
    sens = np.asarray(sensitivity, dtype=np.float64)
    hidden = np.asarray(layover, bool) | np.asarray(shadow, bool)
    if kept_shadow is not None:
        hidden = hidden | np.asarray(kept_shadow, bool)
    if valid is None:
        valid = np.isfinite(slope)
    cls = np.full(slope.shape, CLASS_NODATA, dtype=np.uint8)
    with np.errstate(invalid="ignore"):
        flat = slope < FLAT_SLOPE_DEG
        cls[valid & ~flat & (sens > FAIR_MAX)] = CLASS_GOOD
        cls[valid & ~flat & (sens >= POOR_MAX) & (sens <= FAIR_MAX)] = CLASS_FAIR
        cls[valid & ~flat & (sens < POOR_MAX)] = CLASS_POOR
    cls[valid & flat] = CLASS_FLAT
    cls[valid & hidden] = CLASS_NOT_VISIBLE
    return cls


# --- loading and writing ----------------------------------------------------
def _as_cfg(cfg_or_path):
    if hasattr(cfg_or_path, "mintpy_dir"):
        return cfg_or_path
    p = Path(cfg_or_path)
    if p.is_dir():
        mintpy = p / "mintpy" if (p / "mintpy").is_dir() else p
    else:
        mintpy = p.parent.parent if p.parent.name == "geo" else p.parent
    return SimpleNamespace(mintpy_dir=mintpy, backend="", project_dir=mintpy.parent)


def load_geometry(cfg_or_path) -> dict:
    """
    The geometry on the product grid: height, incidence (deg), LOS (e, n, u)
    toward the satellite (from deliverables.geometry_field), the kept
    shadowMask (None if absent), attrs, dx_m/dy_m, valid (imaged cells) and
    the file read. Accepts a PipelineConfig, a project or mintpy directory,
    or the geometry .h5 path.
    """
    import h5py
    from . import deliverables as D
    cfg = _as_cfg(cfg_or_path)
    path = None
    for name in ("geometryGeo.h5", "geometryRadar.h5"):
        cand = D._product(cfg, name)
        if cand.exists():
            path = cand
            break
    if path is None:
        raise FileNotFoundError(f"No geometry file under {cfg.mintpy_dir}")
    with h5py.File(path, "r") as f:
        attrs = dict(f.attrs)
        kept = f["shadowMask"][:].astype(bool) if "shadowMask" in f else None
    height = D.geometry_field(cfg, "height")
    inc = D.geometry_field(cfg, "incidence")
    los = tuple(D.geometry_field(cfg, k) for k in ("los_east", "los_north", "los_up"))
    if height is None or inc is None or any(a is None for a in los):
        raise ValueError(f"{path} lacks height, incidence or the LOS geometry.")
    height = height.astype(np.float64)
    valid = np.isfinite(inc) & np.isfinite(height) & np.all([np.isfinite(a) for a in los], axis=0)
    height[~valid] = np.nan
    dx_m, dy_m = grid_spacing_m(attrs, height.shape)
    return {"height": height, "incidence": inc.astype(np.float64),
            "los": tuple(a.astype(np.float64) for a in los), "shadow_mask": kept,
            "attrs": attrs, "dx_m": dx_m, "dy_m": dy_m, "valid": valid, "path": path}


def compute_layers(geom: dict) -> dict:
    """Every S1 layer and the summary from a load_geometry() result."""
    slope, aspect, down = slope_aspect(geom["height"], geom["dx_m"], geom["dy_m"])
    sens = los_downslope_sensitivity(geom["los"], down)
    layover, shadow = layover_shadow(geom["height"], geom["los"], geom["dx_m"], geom["dy_m"])
    valid = geom["valid"] & np.isfinite(slope)
    kept = geom.get("shadow_mask")
    cls = visibility_class(slope, sens, layover, shadow, kept, valid)
    n = int(valid.sum())
    shares = {CLASS_NAMES[c]: (float((cls[valid] == c).sum()) / n if n else None)
              for c in CLASS_NAMES}
    summary = {
        "imaged_cells": n,
        "class_shares": shares,
        "layover_share": float(layover[valid].mean()) if n else None,
        "shadow_recomputed_share": float(shadow[valid].mean()) if n else None,
    }
    if kept is not None and n:
        k, s = kept[valid], shadow[valid]
        summary["shadow_kept_share"] = float(k.mean())
        summary["shadow_agreement_share"] = float((k == s).mean())
        summary["kept_shadow_recomputed_share"] = (float(s[k].mean()) if k.any() else None)
        summary["recomputed_shadow_kept_share"] = (float(k[s].mean()) if s.any() else None)
    return {"slope_deg": slope, "aspect_deg": aspect, "sensitivity": sens,
            "layover": layover, "shadow": shadow, "visibility_class": cls,
            "valid": valid, "summary": summary}


def write_layers(cfg, out_dir) -> dict:
    """GeoTIFFs slope_deg, aspect_deg, los_downslope_sensitivity,
    visibility_class and slope_visibility_summary.json in out_dir; returns
    {name: path} plus "summary"."""
    from . import deliverables as D
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    geom = load_geometry(cfg)
    lay = compute_layers(geom)
    attrs = geom["attrs"]
    cfg_t = _as_cfg(cfg)

    def tags(q, units, sign):
        try:
            return D.product_tags(cfg_t, attrs, q, units, masked=False, sign=sign)
        except Exception:                                     # noqa: BLE001
            return {"QUANTITY": q, "UNITS": units, "SIGN_CONVENTION": sign}

    valid = lay["valid"]
    out = {}
    for name, arr, q, units, sign in (
            ("slope_deg", lay["slope_deg"], "terrain slope from the elevation model", "deg", "unsigned"),
            ("aspect_deg", lay["aspect_deg"], "downhill direction, clockwise from north", "deg",
             "0 = north, 90 = east; empty where the slope is under 0.5 deg"),
            ("los_downslope_sensitivity", lay["sensitivity"],
             "share of straight-downslope movement seen along the line of sight |LOS.downslope|",
             "1", "unsigned (0-1)")):
        a = np.where(valid, arr, np.nan).astype(np.float32)
        p = out_dir / f"{name}.tif"
        D._write_geotiff(p, a, attrs, tags=tags(q, units, sign), description=q, units=units)
        out[name] = p
    p = out_dir / "visibility_class.tif"
    legend = "; ".join(f"{k} = {v}" for k, v in CLASS_NAMES.items())
    D._write_geotiff(p, lay["visibility_class"], attrs, nodata=CLASS_NODATA, dtype="uint8",
                     tags=tags("where this satellite can and cannot see slope movement",
                               "class", legend),
                     description="slope visibility class: " + legend, units="class")
    out["visibility_class"] = p
    summary = dict(lay["summary"], geometry_file=str(geom["path"]),
                   thresholds={"flat_slope_deg": FLAT_SLOPE_DEG, "poor_max": POOR_MAX,
                               "fair_max": FAIR_MAX, "flat_aspect_deg": FLAT_ASPECT_DEG,
                               "look_step_cells": LOOK_STEP_CELLS})
    sp = out_dir / SUMMARY_NAME
    sp.write_text(json.dumps(summary, indent=2))
    out["summary_json"] = sp
    out["summary"] = summary
    return out
