"""Gate S1 (review_2026-11-07/PLAN_oil_gas_slopes.md): slope geometry layers.

Pre-registered, implemented exactly as the plan states; thresholds are the
plan's and were not tuned on results.

Fixtures (written here, no run needed):
  * a facet mosaic: planar tiles at 10, 25, 40 deg facing E, W, N, S;
  * a ridge with 65 deg faces, its axis 10 deg off north, high enough for
    layover on the radar-facing face and shadow on the far side;
  * an ascending-like look (toward the satellite at bearing ~258 deg) and a
    descending-like one (~102 deg), incidence varying 33-43 deg across range
    and the bearing varying along azimuth, so the geometry is per cell.

Gate:
  (a) slope and aspect on facet interiors within 0.1 deg of truth;
  (b) layover and shadow masks IDENTICAL to an independent recomputation
      written below in plain Python: an explicit satellite position per cell
      (layover: compare 3-D distances to the satellite either side of the
      cell along the look) and per-cell ray casting by elevation angle
      (shadow); its own interpolation, its own LOS from the bearing; no
      module function is used;
  (c) |LOS . downslope| within 0.01 of the analytic value on facet interiors.
Also: visibility_class boundaries, N/S facets read poor for a near-polar
look (at 10 deg), and the loader/writer on a small geometry file.
"""
import json
import math
from types import SimpleNamespace

import numpy as np
import pytest

from insar_pipeline import slope_geometry as sg

DX = DY = 15.0
TILE = 24
MARGIN = 2
SLOPES = (10.0, 25.0, 40.0)
ASPECTS = (90.0, 270.0, 0.0, 180.0)          # E, W, N, S (downhill direction)


# --- geometry, written independently of the module -------------------------
def look(kind, shape):
    """(bearing_deg, incidence_deg) per cell: bearing of the ground-to-
    satellite direction, clockwise from north. Ascending: radar to the WSW,
    incidence grows eastward; descending: radar to the ESE, grows westward."""
    nr, nc = shape
    col = np.arange(nc)[None, :] / max(nc - 1, 1)
    row = np.arange(nr)[:, None] / max(nr - 1, 1)
    if kind == "asc":
        inc = 33.0 + 10.0 * col
        bearing = 258.0 + 1.5 * (row - 0.5)
    else:
        inc = 43.0 - 10.0 * col
        bearing = 102.0 - 1.5 * (row - 0.5)
    return np.broadcast_to(bearing, shape).copy(), np.broadcast_to(inc, shape).copy()


def los_ref(bearing, inc):
    b, t = np.radians(bearing), np.radians(inc)
    return np.sin(t) * np.sin(b), np.sin(t) * np.cos(b), np.cos(t)


def facet_mosaic():
    """DEM, true slope, true aspect, interior mask."""
    nr, nc = TILE * len(SLOPES), TILE * len(ASPECTS)
    h = np.zeros((nr, nc))
    slope = np.full((nr, nc), np.nan)
    aspect = np.full((nr, nc), np.nan)
    interior = np.zeros((nr, nc), bool)
    for i, a in enumerate(SLOPES):
        for j, A in enumerate(ASPECTS):
            rc, cc = i * TILE + (TILE - 1) / 2, j * TILE + (TILE - 1) / 2
            for r in range(i * TILE, (i + 1) * TILE):
                for c in range(j * TILE, (j + 1) * TILE):
                    x, y = (c - cc) * DX, -(r - rc) * DY
                    # height falls along the downhill azimuth A
                    h[r, c] = -math.tan(math.radians(a)) * (
                        x * math.sin(math.radians(A)) + y * math.cos(math.radians(A)))
                    slope[r, c], aspect[r, c] = a, A
                    interior[r, c] = (MARGIN <= r - i * TILE < TILE - MARGIN
                                      and MARGIN <= c - j * TILE < TILE - MARGIN)
    return h, slope, aspect, interior


RIDGE_H, RIDGE_FACE, RIDGE_ROT = 300.0, 65.0, 10.0


def ridge():
    nr, nc = 80, 140
    rows, cols = np.indices((nr, nc), dtype=float)
    x, y = (cols - (nc - 1) / 2) * DX, -(rows - (nr - 1) / 2) * DY
    rot = math.radians(RIDGE_ROT)
    dist = x * math.cos(rot) - y * math.sin(rot)       # signed, across the axis
    h = np.maximum(0.0, RIDGE_H - np.abs(dist) * math.tan(math.radians(RIDGE_FACE)))
    return h, dist


# --- the independent recomputation ------------------------------------------
def _interp(h, r, c):
    nr, nc = h.shape
    if not (0 <= r <= nr - 1 and 0 <= c <= nc - 1):
        return math.nan
    i, j = min(int(math.floor(r)), nr - 2), min(int(math.floor(c)), nc - 2)
    tr, tc = r - i, c - j
    left = h[i, j] + tr * (h[i + 1, j] - h[i, j])           # along rows first
    right = h[i, j + 1] + tr * (h[i + 1, j + 1] - h[i, j + 1])
    return left + tc * (right - left)


def reference_masks(h, bearing, inc):
    """Plain loops, 3-D points in metres (x east, y north, z up)."""
    nr, nc = h.shape
    step = min(DX, DY)
    R = 1.0e8                                               # satellite distance, m
    hmax = float(np.max(h))
    lay = np.zeros((nr, nc), bool)
    sha = np.zeros((nr, nc), bool)
    for r in range(nr):
        for c in range(nc):
            b, t = math.radians(bearing[r, c]), math.radians(inc[r, c])
            to_sat = (math.sin(t) * math.sin(b), math.sin(t) * math.cos(b), math.cos(t))
            hor = math.hypot(to_sat[0], to_sat[1])
            gx, gy = -to_sat[0] / hor, -to_sat[1] / hor       # away from the radar
            px, py, pz = c * DX, -r * DY, h[r, c]
            sat = (px + R * to_sat[0], py + R * to_sat[1], pz + R * to_sat[2])

            def ground(sx, sy):
                return sx, sy, _interp(h, -sy / DY, sx / DX)

            far = ground(px + step * gx, py + step * gy)
            near = ground(px - step * gx, py - step * gy)
            if not (math.isnan(far[2]) or math.isnan(near[2])):
                d_far = math.dist(sat, far)
                d_near = math.dist(sat, near)
                lay[r, c] = d_far < d_near                  # slant range runs backwards

            elev_sat = math.atan2(to_sat[2], hor)           # 90 - incidence
            k = 1
            while True:
                qx, qy, qz = ground(px - k * step * gx, py - k * step * gy)
                if math.isnan(qz):
                    break
                if math.atan2(qz - pz, k * step) > elev_sat:
                    sha[r, c] = True
                    break
                if pz + k * step * math.tan(elev_sat) > hmax:
                    break
                k += 1
    return lay, sha


# --- gate -------------------------------------------------------------------
@pytest.mark.parametrize("kind", ["asc", "desc"])
def test_gate_s1_facets(kind):
    h, slope_t, aspect_t, interior = facet_mosaic()
    bearing, inc = look(kind, h.shape)
    los = los_ref(bearing, inc)
    slope, aspect, down = sg.slope_aspect(h, DX, DY)
    # (a)
    ds = np.abs(slope - slope_t)[interior]
    da = np.abs((aspect - aspect_t + 180.0) % 360.0 - 180.0)[interior]
    print(f"{kind} (a) max |slope err| {ds.max():.2e} deg, max |aspect err| {da.max():.2e} deg")
    assert ds.max() < 0.1 and da.max() < 0.1
    # (c)
    a, A = np.radians(slope_t), np.radians(aspect_t)
    d_true = (np.cos(a) * np.sin(A), np.cos(a) * np.cos(A), -np.sin(a))
    s_true = np.abs(sum(l * d for l, d in zip(los, d_true)))
    sens = sg.los_downslope_sensitivity(los, down)
    err = np.abs(sens - s_true)[interior]
    print(f"{kind} (c) max |sensitivity err| {err.max():.2e}")
    assert err.max() < 0.01
    # (b) on the mosaic too (tile cliffs give extra layover and shadow)
    lay, sha = sg.layover_shadow(h, los, DX, DY)
    rl, rs = reference_masks(h, bearing, inc)
    print(f"{kind} (b) mosaic: layover {lay.sum()} vs {rl.sum()}, shadow {sha.sum()} vs {rs.sum()}, "
          f"mismatches {int((lay != rl).sum())}/{int((sha != rs).sum())}")
    assert np.array_equal(lay, rl) and np.array_equal(sha, rs)


@pytest.mark.parametrize("kind", ["asc", "desc"])
def test_gate_s1_ridge(kind):
    h, dist = ridge()
    bearing, inc = look(kind, h.shape)
    los = los_ref(bearing, inc)
    lay, sha = sg.layover_shadow(h, los, DX, DY)
    rl, rs = reference_masks(h, bearing, inc)
    print(f"{kind} (b) ridge: layover {lay.sum()} vs {rl.sum()}, shadow {sha.sum()} vs {rs.sum()}")
    assert lay.sum() > 0 and sha.sum() > 0
    assert np.array_equal(lay, rl) and np.array_equal(sha, rs)
    # physical sanity on face interiors: the face toward the radar lays over,
    # the far face and the ground just behind it are in shadow
    rot = math.radians(RIDGE_ROT)
    normal_e, normal_n = math.cos(rot), -math.sin(rot)      # direction of +dist
    to_sat_e, to_sat_n, _ = los_ref(bearing, inc)
    radar_side = np.sign(normal_e * to_sat_e + normal_n * to_sat_n)   # +1: +dist faces radar
    half = RIDGE_H / math.tan(math.radians(RIDGE_FACE))
    inside = np.abs(dist) < half - 2 * DX
    away_from_crest = np.abs(dist) > 2 * DX
    facing = (np.sign(dist) == radar_side)
    rr, cc = np.indices(h.shape)
    off_edge = (rr >= 2) & (rr < h.shape[0] - 2) & (cc >= 2) & (cc < h.shape[1] - 2)
    inside &= off_edge                    # samples beyond the grid edge read as unknown
    assert lay[inside & away_from_crest & facing].all()
    assert sha[inside & away_from_crest & ~facing].all()
    # analytic shadow reach from the axis: H tan(inc) cos(look-to-normal angle,
    # <= 25 deg here); 0.9 of it leaves room for the 15 m sampling of the crest.
    # Rows within 6 of the edge are left out: their rays leave the grid.
    reach = 0.9 * RIDGE_H * np.tan(np.radians(inc)) * math.cos(math.radians(25))
    behind = ((rr >= 6) & (rr < h.shape[0] - 6) & (~facing)
              & (np.abs(dist) > half + 2 * DX) & (np.abs(dist) < reach))
    assert behind.any() and sha[behind].all()
    front_flat = facing & (np.abs(dist) > half + 2 * DX)
    assert not lay[front_flat].any() and not sha[front_flat].any()


def test_los_convention_matches_mintpy_angles():
    """los_from_angles (the deliverables.geometry_field conversion) gives the
    bearing-based vector with MintPy azimuthAngle = -bearing."""
    bearing, inc = look("asc", (5, 7))
    got = sg.los_from_angles(inc, -bearing)
    for g, w in zip(got, los_ref(bearing, inc)):
        assert np.allclose(g, w, atol=1e-12)


@pytest.mark.parametrize("kind", ["asc", "desc"])
def test_north_south_facets_read_poor(kind):
    h, slope_t, aspect_t, interior = facet_mosaic()
    bearing, inc = look(kind, h.shape)
    los = los_ref(bearing, inc)
    slope, aspect, down = sg.slope_aspect(h, DX, DY)
    sens = sg.los_downslope_sensitivity(los, down)
    lay, sha = sg.layover_shadow(h, los, DX, DY)
    cls = sg.visibility_class(slope, sens, lay, sha)
    ns10 = interior & (slope_t == 10.0) & np.isin(aspect_t, (0.0, 180.0))
    assert (sens[ns10] < sg.POOR_MAX).all()
    assert np.isin(cls[ns10], (sg.CLASS_POOR, sg.CLASS_NOT_VISIBLE)).all()
    assert (cls[ns10] == sg.CLASS_POOR).any()
    for a in SLOPES:
        for A, name in zip(ASPECTS, "EWNS"):
            m = interior & (slope_t == a) & (aspect_t == A)
            print(f"{kind} {a:.0f} deg {name}: |LOS.d| {sens[m].min():.3f}-{sens[m].max():.3f}")


def test_visibility_class_boundaries():
    slope = np.array([4.999, 5.0, 10, 10, 10, 10, 10, 3.0, 10, 10, np.nan])
    sens = np.array([0.9, 0.9, 0.2999, 0.3, 0.6, 0.6001, 0.9, 0.9, 0.9, 0.9, 0.9])
    lay = np.zeros(11, bool)
    sha = np.zeros(11, bool)
    kept = np.zeros(11, bool)
    lay[6] = True            # layover on a steep cell
    sha[7] = True            # shadow on a flat cell: not visible wins
    kept[8] = True           # only the kept shadowMask
    cls = sg.visibility_class(slope, sens, lay, sha, kept)
    assert cls.tolist() == [4, 3, 1, 2, 2, 3, 0, 0, 0, 3, 255]
    assert cls.dtype == np.uint8


def test_flat_aspect_is_nan():
    h = np.tile(np.arange(10.0) * DX * math.tan(math.radians(0.3)), (10, 1))
    slope, aspect, (e, n, u) = sg.slope_aspect(h, DX, DY)
    assert np.isnan(aspect).all() and np.isnan(e).all()
    assert np.allclose(u, -np.sin(np.radians(0.3)))


def test_load_and_write_layers(tmp_path):
    h5py = pytest.importorskip("h5py")
    pytest.importorskip("rasterio")
    h, _ = ridge()
    bearing, inc = look("asc", h.shape)
    geo = tmp_path / "proj" / "mintpy" / "geo"
    geo.mkdir(parents=True)
    lat0 = 38.98
    xs = 15.0 / (111_320.0 * math.cos(math.radians(lat0)))
    ys = -15.0 / 110_950.0
    with h5py.File(geo / "geo_geometryRadar.h5", "w") as f:
        f["height"] = h.astype(np.float32)
        f["incidenceAngle"] = inc.astype(np.float32)
        f["azimuthAngle"] = (-bearing).astype(np.float32)
        f["shadowMask"] = np.zeros(h.shape, bool)
        f.attrs.update({"X_FIRST": -104.9, "X_STEP": xs, "Y_FIRST": lat0, "Y_STEP": ys,
                        "LENGTH": h.shape[0], "WIDTH": h.shape[1]})
    cfg = SimpleNamespace(mintpy_dir=tmp_path / "proj" / "mintpy", backend="",
                          project_dir=tmp_path / "proj")
    g = sg.load_geometry(cfg)
    assert np.allclose(g["dx_m"], 15.0, atol=0.1) and abs(g["dy_m"] - 15.0) < 0.1
    for a, w in zip(g["los"], los_ref(bearing, inc)):
        assert np.allclose(a, w, atol=1e-6)
    out = sg.write_layers(cfg, tmp_path / "out")
    for k in ("slope_deg", "aspect_deg", "los_downslope_sensitivity", "visibility_class"):
        assert out[k].exists()
    s = json.loads(out["summary_json"].read_text())
    assert abs(sum(v for v in s["class_shares"].values()) - 1.0) < 1e-9
    assert 0 <= s["shadow_agreement_share"] <= 1
    assert s["class_shares"][sg.CLASS_NAMES[sg.CLASS_NOT_VISIBLE]] > 0
