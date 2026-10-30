"""
Georeference a mine drawing from its own coordinate grid and read panel
outlines from its vector paths (T1 digitising, 8 Oct 2026).

The MSEC and South32 drawings are CAD exports: the MGA grid lines and the
panel outlines are vector paths in the PDF. The grid labels are drawn as
glyph outlines, not text, so the value of one grid line per axis is read by
eye from the rendered drawing and recorded in the drawing's spec (with the
grid step); every other line's value follows from the step. The fit is
least squares, so its residual at the grid lines is the georeference error
of the drawing itself (CAD-exact drawings give centimetres).

Units: page points (pt, origin top left, y down) in, metres (MGA zone 56)
out. pyproj converts MGA94 (EPSG:28356) or MGA2020 (EPSG:7856) to WGS 84
longitude/latitude.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

import pdfvec

TAN_DRAW = float(np.tan(np.radians(26.5)))      # NSW Southern Coalfield angle of draw (protocol A)


def segments(paths, colour=None, width=None, tol=0.02):
    """Every straight segment of the stroked paths: array (n, 4) x0 y0 x1 y1."""
    out = []
    for p in paths:
        if p.stroke is None:
            continue
        if colour is not None and pdfvec.colour_name(p.stroke) != colour:
            continue
        if width is not None and abs(p.width - width) > tol:
            continue
        for s in p.subpaths:
            s = np.asarray(s)
            for i in range(len(s) - 1):
                out.append((s[i, 0], s[i, 1], s[i + 1, 0], s[i + 1, 1]))
    return np.array(out) if out else np.zeros((0, 4))


def long_lines(paths, min_len, colour=None, width=None):
    """Axis-parallel segments at least min_len pt long: (xs of vertical lines,
    ys of horizontal lines), each sorted and de-duplicated to 0.01 pt."""
    seg = segments(paths, colour, width)
    if not len(seg):
        return np.zeros(0), np.zeros(0)
    dx, dy = seg[:, 2] - seg[:, 0], seg[:, 3] - seg[:, 1]
    ver = seg[(np.abs(dx) < 0.01) & (np.abs(dy) >= min_len)]
    hor = seg[(np.abs(dy) < 0.01) & (np.abs(dx) >= min_len)]
    xs = np.unique(np.round(ver[:, 0], 2))
    ys = np.unique(np.round(hor[:, 1], 2))
    return xs, ys


def regular(vals, step_guess=None, tol=0.05):
    """The longest run of evenly spaced values (a grid) in vals."""
    v = np.sort(np.asarray(vals, float))
    if v.size < 3:
        return v
    d = np.diff(v)
    step = step_guess or float(np.median(d))
    best, cur = [v[0]], [v[0]]
    for a in v[1:]:
        k = (a - cur[-1]) / step
        if abs(k - round(k)) < tol and round(k) == 1:
            cur.append(a)
        else:
            if len(cur) > len(best):
                best = cur
            cur = [a]
    if len(cur) > len(best):
        best = cur
    return np.array(best)


@dataclass
class AxisFit:
    """map = a + b * page along one page axis."""
    a: float
    b: float
    rms_m: float
    n: int
    lines: list = field(default_factory=list)       # [(page, value)]

    def __call__(self, p):
        return self.a + self.b * np.asarray(p, float)


def fit_axis(page_vals, first_page, first_value, step_value):
    """Give every grid line its value from the one read by eye (the line
    nearest first_page has first_value; values change by step_value per line
    in increasing page coordinate) and fit value = a + b * page."""
    pv = np.sort(np.asarray(page_vals, float))
    i0 = int(np.argmin(np.abs(pv - first_page)))
    if abs(pv[i0] - first_page) > 3:
        raise ValueError(f"no grid line within 3 pt of {first_page} (nearest {pv[i0]:.2f})")
    vals = first_value + step_value * (np.arange(pv.size) - i0)
    b, a = np.polyfit(pv, vals, 1)
    res = vals - (a + b * pv)
    return AxisFit(float(a), float(b), float(np.sqrt(np.mean(res ** 2))), int(pv.size),
                   [(float(p), float(v)) for p, v in zip(pv, vals)])


@dataclass
class GridGeoref:
    """A drawing whose MGA grid is parallel to the page edges. `east_axis` is
    'x' (normal) or 'y' (page turned a quarter turn)."""
    east: AxisFit
    north: AxisFit
    east_axis: str = "x"

    def __call__(self, pts):
        p = np.asarray(pts, float).reshape(-1, 2)
        if self.east_axis == "x":
            return np.c_[self.east(p[:, 0]), self.north(p[:, 1])]
        return np.c_[self.east(p[:, 1]), self.north(p[:, 0])]

    @property
    def scale_m_per_pt(self):
        return float(np.sqrt(abs(self.east.b * self.north.b)))

    def record(self) -> dict:
        return {"kind": "grid parallel to the page", "east_from_page": self.east_axis,
                "east": {"a": self.east.a, "b_m_per_pt": self.east.b, "rms_m": round(self.east.rms_m, 3),
                         "lines": self.east.n, "first_last": [self.east.lines[0], self.east.lines[-1]]},
                "north": {"a": self.north.a, "b_m_per_pt": self.north.b, "rms_m": round(self.north.rms_m, 3),
                          "lines": self.north.n, "first_last": [self.north.lines[0], self.north.lines[-1]]},
                "scale_m_per_pt": round(self.scale_m_per_pt, 4)}


@dataclass
class AffineGeoref:
    """map = A @ [x, y, 1] fitted to control points (any grid rotation)."""
    M: np.ndarray
    rms_m: float
    n: int
    residuals: list

    def __call__(self, pts):
        p = np.asarray(pts, float).reshape(-1, 2)
        return np.c_[p, np.ones(len(p))] @ self.M.T

    def record(self) -> dict:
        return {"kind": "affine from control points", "matrix": self.M.tolist(), "rms_m": round(self.rms_m, 3),
                "points": self.n, "residuals_m": self.residuals}


def fit_affine(page_xy, map_xy, labels=None) -> AffineGeoref:
    P = np.c_[np.asarray(page_xy, float), np.ones(len(page_xy))]
    Q = np.asarray(map_xy, float)
    M, *_ = np.linalg.lstsq(P, Q, rcond=None)
    res = Q - P @ M
    r = np.hypot(res[:, 0], res[:, 1])
    labels = labels or [str(i) for i in range(len(r))]
    return AffineGeoref(M.T, float(np.sqrt(np.mean(r ** 2))), len(r),
                        [(lab, round(float(x), 2)) for lab, x in zip(labels, r)])


def closed_polygons(paths, stroke=None, fill=None, width=None, min_pts=4, tol=0.02):
    """Closed subpaths (or open ones that end where they start) of the given
    style, as page-coordinate arrays without the repeated last vertex."""
    out = []
    for p in paths:
        if stroke is not None and pdfvec.colour_name(p.stroke) != stroke:
            continue
        if fill is not None and pdfvec.colour_name(p.fill) != fill:
            continue
        if width is not None and abs(p.width - width) > tol:
            continue
        for s, c in zip(p.subpaths, p.closed):
            s = np.asarray(s, float)
            if len(s) >= 2 and np.hypot(*(s[0] - s[-1])) < 0.05:
                s, c = s[:-1], True
            if c and len(s) >= min_pts:
                out.append(s)
    return out


def dedupe(polys, tol=0.3):
    keep = []
    for q in polys:
        if not any(len(q) == len(k) and np.max(np.abs(np.sort(q, axis=0) - np.sort(k, axis=0))) < tol for k in keep):
            keep.append(q)
    return keep


def rect_metrics(poly) -> dict:
    """Side lengths, long-axis bearing (deg from grid north, 0-180) and area of
    a quadrilateral in metres."""
    q = np.asarray(poly, float)
    sides = np.hypot(*np.diff(np.vstack([q, q[:1]]), axis=0).T)
    x, y = q[:, 0], q[:, 1]
    area = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    i = int(np.argmax(sides))
    d = q[(i + 1) % len(q)] - q[i]
    bearing = float(np.degrees(np.arctan2(d[0], d[1])) % 180.0)
    return {"sides_m": [round(float(s), 1) for s in sides], "length_m": round(float(sides.max()), 1),
            "width_m": round(float(sides.min()), 1), "bearing_deg": round(bearing, 2),
            "area_ha": round(float(area) / 1e4, 2)}


_TRANSFORMERS = {}


def mga_to_lonlat(pts, epsg: int):
    """MGA zone 56 (28356 = GDA94, 7856 = GDA2020) to WGS 84 lon/lat."""
    from pyproj import Transformer
    if epsg not in _TRANSFORMERS:
        _TRANSFORMERS[epsg] = Transformer.from_crs(epsg, 4326, always_xy=True)
    p = np.asarray(pts, float).reshape(-1, 2)
    lon, lat = _TRANSFORMERS[epsg].transform(p[:, 0], p[:, 1])
    return np.c_[lon, lat]


def lonlat_to_mga(pts, epsg: int):
    from pyproj import Transformer
    key = ("inv", epsg)
    if key not in _TRANSFORMERS:
        _TRANSFORMERS[key] = Transformer.from_crs(4326, epsg, always_xy=True)
    p = np.asarray(pts, float).reshape(-1, 2)
    e, n = _TRANSFORMERS[key].transform(p[:, 0], p[:, 1])
    return np.c_[e, n]
