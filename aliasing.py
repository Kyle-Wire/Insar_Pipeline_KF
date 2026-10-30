"""
Fast motion: where a value may be wrong because the ground moved too fast
between two images, found BEFORE unwrapping (PLAN phase 2, F2, F1, F3;
8 Oct 2026). Nothing here changes a delivered number.

WHY THIS EXISTS
---------------
Unwrapping can follow a change between neighbouring look cells of at most
half a cycle (a quarter wavelength of line-of-sight motion: 13.9 mm at
Sentinel-1's C-band). A longwall panel can sink faster than that across the
ground in one revisit. Then the phase "aliases": it wraps back and reads as a
smaller change, and nothing downstream can see that it happened. The old
flag, quality_too_fast_to_measure.tif, is computed from the UNWRAPPED
velocity (deliverables.export_extra_layers). An aliased cell already reads
too slow there, so the flag misses exactly the cells it exists for. It is a
LOWER BOUND, and export_products() relabels it so.

This module looks at the wrapped interferograms instead, before snaphu runs:

F2  pre-unwrap diagnostic (pre_unwrap, export_products, risk_between)
    Per exported pair and per look cell:
    - the wrapped phase gradient in cycles per look cell. On the phase-linked
      route it is measured on the FULL-RESOLUTION linked phase, from the unit
      neighbour products summed over the 3 x 3 look cells centred on each
      cell (GRADIENT_BLOCK: over one cell alone, the phase-linking window's
      smooth noise read as a gradient on 1-2% of still farmland per pair), so
      it can show gradients above half a cycle per look cell. A multilooked
      interferogram cannot: its own wrapped gradient never exceeds half a
      cycle (the classic stack route reports it, it never flags on it);
    - a coherence hole, RELATIVE to the ground around it (protocol A
      amendment 2026-10-08/p23-2): the cell's pair coherence, averaged over
      the 3 x 3 look cells centred on it, is at most 2/pi of the median pair
      coherence of the reliable cells 1-3 km around it (at least 10 of them,
      median >= 0.4), i.e. decorrelation local to moving ground; a
      scene-wide low, e.g. a summer pair, is not a hole. The rule as first
      fixed (pair coherence below 0.4 while half the ring reaches 0.4) is
      kept per pair as level_v1 for the protocol's version_1 report;
    - the residue density of the wrapped phase (share of 2x2 loops with a
      residue), inside the deforming polygons, in a 1-3 km ring around them
      and over the crop. Reported, never a flag.
    Levels: 0 = no sign of aliasing found, 1 = risk (coherence hole),
    2 = likely aliased (gradient above half a cycle per look cell). This is the
    flag protocol_A.yaml fixed on 8 Oct 2026 (comparisons.unflagged_failure_rate
    .flag_definition), with the gradient test as level 2 and the hole as 1,
    and the hole as amended (2026-10-08/p23-2). The rule as first fixed,
    MEASURED on Site B farmland (2019, 81 pairs, phasor pair coherence,
    read only; A1 and A3 also carry phasor coherence), flagged a median 19%
    (3-36%) of the crop per pair, about half per interval between dates and
    93% over the year, so on such ground it flagged most points; the amended
    rule's figures on the same pairs are in review_2026-10-08/protdec/. The
    harness also checks that the flag is selective (its share on stable
    ground, amendment 2026-10-08/p23-2) before the honest-or-flagged row can
    pass.
    LOWER-BOUND SEMANTICS: a flagged cell has evidence of aliasing or of
    ground that cannot be measured; an unflagged cell has no evidence, which
    is not proof it is right (a gradient beyond half a cycle per FULL-
    resolution pixel wraps again, and the linked phase is itself smoothed).
    Outputs: <project>/aliasing/ (per-pair levels on the radar crop, kept by
    cleanup level 2) and, on the product grid,
    deliverables/quality_aliasing_risk.tif (uint8; 255 = outside the radar
    crop) and deliverables/aliasing_risk.json (method, thresholds, per-pair
    and per-interval summary).

F1  detectability preflight (detectability, plan_note)
    lambda/4 per look cell against the expected tilt per interval, from
    expected_max_rate_mm_yr and the trough half-width (expected_trough_
    halfwidth_m, or half the narrow side of each zone polygon). Shown in the
    run plan (stack_backend.plan_stack -> python -m insar_pipeline.plan_run).

F3  fast-network zones (apply_fast_zones)
    Inside fast_zone_geojson, only each date's fast_zone_connections nearest
    pairs are used: the longer pairs, which alias first, are set to MintPy's
    no-data there (unwrapped phase 0 and connected component 0, which MintPy's
    inversion reads as missing) after unwrapping and before MintPy loads them.
    The originals are kept beside each pair (fast_zone_original.npz) and put
    back when the setting changes or is removed. Smaller phase-linking
    windows for the same question: miaplpy_range_window/miaplpy_azimuth_window
    in a project that re-links another project's stack (miaplpy_relink_from,
    miaplpy_backend.prepare_relink).

The wavelength always comes from a product (attributes, the stack's
reference/IW*.xml, the SLC or interferogram stack, a NISAR granule). Only
the preflight, before any product exists, falls back to the sensor's nominal
value, and says so.

Settings (config.py): aliasing_diagnostic ("auto" = on for purpose
experiment or site_profile mine_longwall, "on", "off"), aliasing_zone_geojson
(empty = deramp_exclude_geojson), expected_max_rate_mm_yr,
expected_trough_halfwidth_m, fast_zone_geojson, fast_zone_connections.
"""
import json
import logging
import math
import os
import re
import time
from datetime import datetime
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

VERSION = "F2/F1/F3 v3 (8 Oct 2026: coherence hole relative to its ring, amendment 2026-10-08/p23-2)"

# ---- the flag (fixed 8 Oct 2026, before any experiment data: protocol_A.yaml
# comparisons.unflagged_failure_rate.flag_definition)
GRADIENT_LIMIT_CYCLES = 0.5          # per look cell: beyond it neighbouring look cells differ by > half a cycle
# A gradient counts only where it is consistent: |sum of the unit neighbour
# products| / their number within the look cell. Pure noise over the 33
# products of an 11 x 3 look cell reads ~0.15 (0.46 at the 99.9th percentile);
# a clean fringe pattern reads ~1. Without it, noise would read as steep fringes.
GRADIENT_MIN_CONSISTENCY = 0.7
# The gradient is measured over 3 x 3 look cells centred on each cell (~120 m
# at 11 x 3 looks), not over the cell alone. MEASURED on the Site B
# phase-linked record (8 Oct 2026, review_2026-10-08/p23/alias_smoke/
# explore_gradient.log): ground with no motion had 1.0-1.7% of cells per
# summer pair above half a cycle with a consistent single-cell gradient - the
# phase-linking window (15 x 15 pixels) makes noise spatially smooth, so it
# looks like a gradient over one look cell - and 0.000-0.003% over 3 x 3.
# A longwall flank (200-500 m) holds its gradient over 120 m.
GRADIENT_BLOCK = 3
NOISE_FALSE_ALARM = 1e-3             # with few looks the floor rises to pure noise's 99.9th percentile


def min_consistency(products: int) -> float:
    """The consistency a gradient needs with `products` unit neighbour
    products per look cell: GRADIENT_MIN_CONSISTENCY, or more where pure
    noise reaches it more often than NOISE_FALSE_ALARM (|mean of N random
    unit phasors| is Rayleigh: P(> r) = exp(-N r^2)). 0.7 at 11 x 3 or
    5 x 3 looks; above 1 (never flags) below about 7 products."""
    n = max(1, int(products))
    return max(GRADIENT_MIN_CONSISTENCY, math.sqrt(math.log(1.0 / NOISE_FALSE_ALARM) / n))
HOLE_FLOOR = 0.4                     # pair coherence below this ...
HOLE_RING_M = (1000.0, 3000.0)       # ... while half the reliable cells 1-3 km around it reach it
HOLE_MIN_RING_CELLS = 10             # fewer reliable cells in the ring: no judgement (not a hole)
# The coherence hole as amended (protocol A amendment 2026-10-08/p23-2, 8 Oct
# 2026, before any experiment result): RELATIVE to the ring, not an absolute
# low. A cell is in a hole when its pair coherence, averaged over the 3 x 3
# look cells centred on it (the support of the gradient test, GRADIENT_BLOCK),
# is at most HOLE_RATIO x the median pair coherence of the reliable cells 1-3
# km around it in the same pair, and that median is at least HOLE_FLOOR (the
# protocol's 0.4: a ring that is itself barely coherent cannot show a hole -
# the protocol's "a scene-wide low is not a flag"). HOLE_RATIO = 2/pi = sinc(1/2):
# a phase ramp of half a cycle across a look cell - the protocol's aliasing
# limit, the same 0.5 cycle as the gradient test - multiplies the look cell's
# coherence (|mean unit phasor|, or the sample coherence over the estimation
# window) by |sinc(g)| = 0.64 at g = 1/2. So a cell that kept at most 64% of its
# ring's coherence lost at least what that gradient costs: the margin is 36%
# of the ring's median. The rule as first fixed (any cell below 0.4 while half
# the ring reaches 0.4) flagged 93% of Site B's crop over 2019; it is kept
# as coherence_holes_as_first_fixed and recorded beside the amended level
# (level_v1), so the harness can report the protocol as first fixed.
HOLE_RATIO = 2.0 / math.pi
HOLE_BLOCK = 3                       # = GRADIENT_BLOCK
HOLE_RING_GRID_M = 200.0             # the ring median is taken on blocks no larger than this (1/5 of the inner radius)
HOLE_RING_BINS = 100                 # coherence resolution of the ring median: 0.01
HOLE_RULE = {"kind": "relative to the ring", "ratio": round(2.0 / math.pi, 4), "cell": "mean over 3 x 3 look cells",
             "ring_m": [1000.0, 3000.0], "ring_statistic": "median of the reliable cells", "ring_min_cells": 10,
             "ring_median_at_least": 0.4, "amendment": "2026-10-08/p23-2"}
ZONE_RING_M = (1000.0, 3000.0)       # the ring around the deforming polygons, for the reported statistics
LEVEL_NONE, LEVEL_RISK, LEVEL_LIKELY = 0, 1, 2
NODATA = 255

LEVEL_TEXT = {0: "no sign of aliasing found (not proof the value is right)",
              1: "risk: coherence hole (the cell kept at most 2/pi of the coherence of the ground 1-3 km around it "
                 "in a pair: decorrelation local to this ground; aliasing cannot be seen there)",
              2: "likely aliased: wrapped phase gradient above half a cycle per look cell in a pair"}

# ---- the preflight's verdict bands (ratio of the cell-to-cell change per interval to lambda/4)
MEASURABLE_BELOW = 0.5
MARGINAL_BELOW = 1.0

# ---- nominal sensor values, ONLY for the preflight before any product exists
C_M_S = 299_792_458.0
NOMINAL_WAVELENGTH = {
    "sentinel-1": (C_M_S / 5.405e9, "nominal Sentinel-1 C-band (5.405 GHz centre frequency, ESA): no product read yet"),
    "nisar-l": (0.243932, "nominal NISAR L-band (1.229 GHz, as nisar_mining): no product read yet"),
}
NOMINAL_REVISIT_DAYS = {"sentinel-1": 12.0, "nisar-l": 12.0}
NOMINAL_INCIDENCE_DEG = 39.0         # resolution.S1_INCIDENCE_DEG; NISAR's swath is similar

SETTING_DEFAULTS = {
    "aliasing_diagnostic": "auto",
    "aliasing_zone_geojson": "",
    "expected_max_rate_mm_yr": 0.0,
    "expected_trough_halfwidth_m": 0.0,
    "fast_zone_geojson": "",
    "fast_zone_connections": 1,
}
DIAGNOSTIC_VALUES = ("auto", "on", "off")
FAST_SIDECAR = "fast_zone_original.npz"      # stack/merged/interferograms/<pair>/
FAST_ZONE_UNWRAP_METHODS = ("no", "auto", "bridging", "bridging+phase_closure")
PAIR_RE = re.compile(r"^(\d{8})_(\d{8})$")


# ======================================================================
# settings
# ======================================================================
def setting(cfg, name: str):
    v = getattr(cfg, name, None)
    return SETTING_DEFAULTS[name] if v is None else v


def _diagnostic_value(cfg) -> str:
    v = setting(cfg, "aliasing_diagnostic")
    if isinstance(v, bool):                  # YAML reads an unquoted on/off as a boolean
        return "on" if v else "off"
    return str(v).strip().lower()


def enabled(cfg) -> bool:
    """Whether the F2 diagnostic runs for this config. "auto" (default):
    on for validation runs (purpose: experiment), longwall sites and any
    config with fast zones; off for client projects, whose results this
    module must not touch."""
    v = _diagnostic_value(cfg)
    if v == "on":
        return True
    if v == "off":
        return False
    from .config import is_experiment, site_profile
    return is_experiment(cfg) or site_profile(cfg) == "mine_longwall" or bool(str(setting(cfg, "fast_zone_geojson")).strip())


def zones_geojson(cfg) -> Path | None:
    """The deforming polygons the statistics and the preflight use:
    aliasing_zone_geojson, else the deramp exclusion polygon."""
    for name in ("aliasing_zone_geojson", "deramp_exclude_geojson"):
        v = str(getattr(cfg, name, "") or "").strip()
        if v and Path(v).exists():
            return Path(v)
    return None


def check_settings(cfg) -> list:
    """Config problems in this module's settings (config.validate_config).
    Static only: no file is required to exist here (a finished project must
    still load after its inputs are cleaned up)."""
    out = []
    v = _diagnostic_value(cfg)
    if v not in DIAGNOSTIC_VALUES:
        out.append(f"aliasing_diagnostic = {setting(cfg, 'aliasing_diagnostic')!r}; expected one of {list(DIAGNOSTIC_VALUES)}")
    for name in ("expected_max_rate_mm_yr", "expected_trough_halfwidth_m"):
        try:
            if float(setting(cfg, name)) < 0:
                out.append(f"{name} = {setting(cfg, name)!r} must be >= 0 (0 = not set)")
        except (TypeError, ValueError):
            out.append(f"{name} = {setting(cfg, name)!r} is not a number")
    try:
        k = int(setting(cfg, "fast_zone_connections"))
        if k < 1:
            out.append(f"fast_zone_connections = {k} must be >= 1 (1 = only each date's nearest pair inside the zones)")
    except (TypeError, ValueError):
        out.append(f"fast_zone_connections = {setting(cfg, 'fast_zone_connections')!r} is not a whole number")
        k = 1
    if str(setting(cfg, "fast_zone_geojson") or "").strip():
        backend = str(getattr(cfg, "backend", "")).lower()
        if backend not in ("miaplpy", "stack"):
            out.append(f"fast_zone_geojson works on the topsStack routes (miaplpy, stack), not {backend!r}")
        if backend == "miaplpy" and int(getattr(cfg, "miaplpy_network_connections", 3) or 0) <= 0:
            out.append("fast_zone_geojson needs a nearest-pair network (miaplpy_network_connections >= 1); "
                       "a single-reference star has no nearest pairs to keep")
        method = str(getattr(cfg, "mintpy_unwrap_error_method", "") or "").strip().lower()
        if method and method not in FAST_ZONE_UNWRAP_METHODS:
            out.append(f"fast_zone_geojson with mintpy_unwrap_error_method {method!r}: MintPy's phase-closure "
                       "step on its own re-references every cell, so the no-data value the fast zones write "
                       "would come back as a number. Use one of "
                       f"{list(FAST_ZONE_UNWRAP_METHODS)}")
        if bool(getattr(cfg, "mintpy_closure_correction", False)):
            out.append("fast_zone_geojson with mintpy_closure_correction: true is not supported (the closure "
                       "correction was never checked against the zones' no-data cells); set it false")
    rel = str(getattr(cfg, "miaplpy_relink_from", "") or "").strip()
    if rel:
        if str(getattr(cfg, "backend", "")).lower() != "miaplpy":
            out.append("miaplpy_relink_from re-links another project's SLC stack: backend must be miaplpy")
        try:
            if Path(rel).expanduser().resolve() == Path(getattr(cfg, "project_dir", "")).expanduser().resolve():
                out.append("miaplpy_relink_from names this project itself; it must be the project whose stack is re-linked")
        except (OSError, RuntimeError):
            pass
        if not Path(rel).is_absolute():
            out.append(f"miaplpy_relink_from {rel} is a relative path; give it in full")
    return out


# ======================================================================
# where things are
# ======================================================================
def _project(cfg) -> Path:
    return Path(cfg.project_dir)


def _stack(cfg) -> Path:
    return _project(cfg) / "stack"


def record_dir(cfg) -> Path:
    """<project>/aliasing: the radar-grid record (kept by cleanup level 2,
    which deletes stack/merged)."""
    return _project(cfg) / "aliasing"


def _pair_file(cfg, pair: str) -> Path:
    return record_dir(cfg) / "pairs" / f"{pair}.npz"


def _fid(path: Path) -> list:
    st = Path(path).stat()
    return [int(st.st_size), int(st.st_mtime_ns)]


def _write_json(path: Path, obj) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=_json_default))
    os.replace(tmp, path)
    return path


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    return str(o)


def _r(x, nd: int = 4):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return round(x, nd) if math.isfinite(x) else None


# ======================================================================
# wavelength: from a product, never assumed
# ======================================================================
def _pos(v):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) and v > 0 else None


def _sensor(cfg) -> str:
    return "nisar-l" if str(getattr(cfg, "backend", "")).lower() == "nisar" else "sentinel-1"


def wavelength(cfg, attrs: dict | None = None, allow_nominal: bool = True) -> tuple[float | None, str]:
    """(wavelength in m, where it came from). In order: the given product
    attributes; the stack's reference/IW*.xml (radarwavelength); MintPy's
    loaded stack or products; MiaplPy's SLC stack; a NISAR granule's centre
    frequency. Only then, when allow_nominal, the sensor's nominal value,
    labelled as such (the preflight, before any product exists)."""
    v = _pos((attrs or {}).get("WAVELENGTH"))
    if v and 0.005 < v < 1.0:                     # X-band 3 cm .. P-band 70 cm (as deliverables)
        return v, "product attributes (WAVELENGTH)"
    P = _project(cfg)
    refs = [P / "stack" / "reference"]
    rel = str(getattr(cfg, "miaplpy_relink_from", "") or "").strip()
    if rel:                                       # a re-link reads that project's stack (before its links exist)
        refs.append(Path(rel).expanduser() / "stack" / "reference")
    for ref in refs:
        for x in sorted(ref.glob("IW*.xml")) if ref.is_dir() else []:
            try:
                m = re.search(r'<property name="radarwavelength">\s*<value>([0-9.eE+-]+)', x.read_text())
            except OSError:
                m = None
            if m and _pos(m.group(1)):
                where = "stack reference" if ref == refs[0] else f"re-linked project {ref.parent.parent.name}'s reference"
                return float(m.group(1)), f"{where} {x.name} (radarwavelength)"
    M = Path(getattr(cfg, "mintpy_dir", P / "mintpy"))
    for h5p in (M / "inputs" / "ifgramStack.h5", M / "geo" / "geo_velocity.h5", M / "velocity.h5",
                P / "miaplpy" / "inputs" / "slcStack.h5"):
        if h5p.exists():
            try:
                import h5py
                with h5py.File(h5p, "r") as f:
                    v = _pos(f.attrs.get("WAVELENGTH"))
            except (OSError, ImportError):
                v = None
            if v:
                return v, f"{h5p.name} attributes (WAVELENGTH)"
    gunw = sorted((P / "nisar_gunw").glob("*.h5")) if (P / "nisar_gunw").is_dir() else []
    if gunw:
        try:
            import h5py
            from .nisar_mining import GRID, META, C_M_S as C
            with h5py.File(gunw[0], "r") as h:
                for p in (GRID + "centerFrequency", META + "centerFrequency"):
                    if p in h:
                        return C / float(np.asarray(h[p][()]).item()), f"{gunw[0].name} (centerFrequency)"
        except Exception:                                     # noqa: BLE001 - fall through to nominal
            pass
    if not allow_nominal:
        return None, "no product carries a wavelength"
    lam, why = NOMINAL_WAVELENGTH[_sensor(cfg)]
    return lam, why


# ======================================================================
# the radar crop: geometry, zones, reliability
# ======================================================================
def _crop_window(cfg) -> dict:
    return json.loads((_stack(cfg) / "crop_window.json").read_text())


def _local_xy(lat, lon, lat0: float):
    """Metres on a local plane (enough for spacings and ring distances)."""
    return (np.asarray(lon, float) * 111_320.0 * math.cos(math.radians(lat0)),
            np.asarray(lat, float) * 110_574.0)


def spacing_m(lat: np.ndarray, lon: np.ndarray) -> tuple[float, float]:
    """(row, column) spacing in metres of a radar grid from its lat/lon,
    medians over the central block."""
    L, W = lat.shape
    r0, r1 = max(0, L // 2 - 32), min(L, L // 2 + 32)
    c0, c1 = max(0, W // 2 - 32), min(W, W // 2 + 32)
    la, lo = lat[r0:r1, c0:c1], lon[r0:r1, c0:c1]
    lat0 = float(np.nanmedian(la))
    x, y = _local_xy(la, lo, lat0)
    dy = np.hypot(np.diff(x, axis=0), np.diff(y, axis=0))
    dx = np.hypot(np.diff(x, axis=1), np.diff(y, axis=1))
    return float(np.nanmedian(dy)), float(np.nanmedian(dx))


def radar_grid(cfg, save: bool = True) -> dict:
    """The unwrap crop of the multilooked frame: window, lat/lon of each
    cell, spacing. Read from stack/merged/geom_reference; a copy goes to
    <project>/aliasing/radar_grid.npz so the product-grid export still works
    after cleanup removes stack/merged."""
    rec = record_dir(cfg) / "radar_grid.npz"
    G = _stack(cfg) / "merged" / "geom_reference"
    if (G / "lat.rdr").exists() and ((_stack(cfg) / "crop_window.json").exists() or (G / "lat.rdr.xml").exists()):
        if (_stack(cfg) / "crop_window.json").exists():
            cw = _crop_window(cfg)
            Lm, Wm = int(cw["length"]), int(cw["width"])
            r0, r1, c0, c1 = (int(cw[k]) for k in ("r0", "r1", "c0", "c1"))
        else:                                     # no unwrap crop (stack_crop_unwrap off): the whole frame
            xt = (G / "lat.rdr.xml").read_text()
            Wm = int(re.search(r'<property name="width">\s*<value>(\d+)', xt).group(1))
            Lm = int(re.search(r'<property name="length">\s*<value>(\d+)', xt).group(1))
            r0, r1, c0, c1 = 0, Lm, 0, Wm
        lat = np.array(np.memmap(G / "lat.rdr", dtype=np.float64, mode="r", shape=(Lm, Wm))[r0:r1, c0:c1])
        lon = np.array(np.memmap(G / "lon.rdr", dtype=np.float64, mode="r", shape=(Lm, Wm))[r0:r1, c0:c1])
        dy, dx = spacing_m(lat, lon)
        g = {"window": [r0, r1, c0, c1], "frame": [Lm, Wm], "lat": lat, "lon": lon, "dy_m": dy, "dx_m": dx}
        if save:
            rec.parent.mkdir(parents=True, exist_ok=True)
            tmp = rec.with_name("radar_grid_tmp.npz")
            np.savez_compressed(tmp, window=np.array(g["window"]), frame=np.array(g["frame"]), lat=lat, lon=lon,
                                spacing=np.array([dy, dx]))
            os.replace(tmp, rec)
        return g
    if rec.exists():
        with np.load(rec) as z:
            return {"window": [int(v) for v in z["window"]], "frame": [int(v) for v in z["frame"]],
                    "lat": z["lat"], "lon": z["lon"], "dy_m": float(z["spacing"][0]), "dx_m": float(z["spacing"][1])}
    raise FileNotFoundError(f"no radar geometry: neither {G / 'lat.rdr'} (with stack/crop_window.json) nor {rec}")


def _polygons(geojson) -> list:
    """[(name, shapely geometry, properties)] of the polygons in a GeoJSON (lon/lat)."""
    from shapely.geometry import shape
    out = []
    feats = json.loads(Path(geojson).read_text()).get("features", [])
    for i, feat in enumerate(feats):
        g = (feat or {}).get("geometry") or {}
        if g.get("type") not in ("Polygon", "MultiPolygon"):
            continue
        s = shape(g)
        s = s if s.is_valid else s.buffer(0)
        if s.is_empty:
            continue
        props = (feat or {}).get("properties") or {}
        name = str(props.get("name") or props.get("Name") or props.get("panel") or props.get("id") or f"polygon {i + 1}")
        out.append((name, s, props))
    return out


def polygon_mask(geojson, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Cells whose centre lies inside any polygon of the GeoJSON."""
    import shapely
    polys = _polygons(geojson)
    if not polys:
        return np.zeros(lat.shape, bool)
    u = shapely.union_all([p for _, p, _ in polys])
    shapely.prepare(u)
    ok = np.isfinite(lat) & np.isfinite(lon)
    m = np.zeros(lat.shape, bool)
    m[ok] = shapely.contains_xy(u, lon[ok], lat[ok])
    return m


def ring_around(mask: np.ndarray, dy: float, dx: float, r0: float, r1: float) -> np.ndarray:
    """Cells between r0 and r1 metres outside `mask`."""
    if not mask.any():
        return np.zeros(mask.shape, bool)
    from scipy.ndimage import distance_transform_edt
    d = distance_transform_edt(~mask, sampling=(dy, dx))
    return (~mask) & (d >= r0) & (d <= r1)


def reliable_mask(cfg, grid: dict) -> tuple[np.ndarray | None, str]:
    """The cells the product will call reliable, on the crop, when that is
    known before unwrapping: on the phase-linked route MiaplPy's linked
    temporal coherence >= min_temporal_coherence (the mask the products
    adopt). None on the classic route (its mask comes from the inversion)."""
    p = _stack(cfg) / "merged" / "linked_temporal_coherence.npz"
    if not p.exists():
        return None, "not known before unwrapping on this route: every cell with data counts"
    try:
        from .miaplpy_backend import load_linked_temporal_coherence
        tc = load_linked_temporal_coherence(cfg)
    except Exception as exc:                                  # noqa: BLE001
        return None, f"linked temporal coherence not readable ({exc}): every cell with data counts"
    r0, r1, c0, c1 = grid["window"]
    tc = tc[r0:r1, c0:c1]
    thr = float(getattr(cfg, "min_temporal_coherence", 0.4))
    return np.nan_to_num(tc, nan=0.0) >= thr, f"linked temporal coherence >= {thr:g} (the products' reliability mask)"


# ======================================================================
# F2: the measures (pure functions on arrays)
# ======================================================================
def _unit(z: np.ndarray) -> np.ndarray:
    a = np.abs(z)
    with np.errstate(invalid="ignore", divide="ignore"):
        u = np.where(a > 0, z / np.where(a > 0, a, 1), 0)
    return np.nan_to_num(u).astype(np.complex64)


def full_res_sums(dphi: np.ndarray, al: int, rl: int, next_row: np.ndarray | None = None) -> dict:
    """
    Per look cell (h, w) of FULL-resolution phase: the sum of the unit
    neighbour products along range (S_rg) and azimuth (S_az), and how many
    there were (N_rg, N_az). dphi: (al*h, rl*w) phase difference of the pair
    (radians, NaN = no data); next_row: the full-resolution row below the
    block (for the azimuth products across its last row), or None.
    """
    al, rl = int(al), int(rl)
    H, Wf = dphi.shape
    h, w = H // al, Wf // rl
    d = dphi[:h * al, :w * rl].astype(np.float32)
    ok = np.isfinite(d)
    z = np.where(ok, np.cos(d), 0).astype(np.float32) + 1j * np.where(ok, np.sin(d), 0).astype(np.float32)
    p = np.zeros_like(z)
    p[:, :-1] = z[:, 1:] * np.conj(z[:, :-1])
    if next_row is not None:
        nr = np.asarray(next_row, dtype=np.float32)[:w * rl]
        okn = np.isfinite(nr)
        zn = (np.where(okn, np.cos(nr), 0) + 1j * np.where(okn, np.sin(nr), 0)).astype(np.complex64)
        zz = np.vstack([z, zn[None, :]])
    else:
        zz = np.vstack([z, np.zeros((1, z.shape[1]), np.complex64)])
    q = zz[1:] * np.conj(zz[:-1])
    out = {}
    for name, prod in (("rg", p), ("az", q)):
        out[f"S_{name}"] = prod.reshape(h, al, w, rl).sum(axis=(1, 3)).astype(np.complex64)
        out[f"N_{name}"] = (np.abs(prod) > 0.5).reshape(h, al, w, rl).sum(axis=(1, 3)).astype(np.float32)
    return out


def gradient_from_sums(sums: dict, al: int, rl: int, block: int = 1) -> dict:
    """
    Cycles per look cell along range (cyc_rg) and azimuth (cyc_az) - the
    angle of the summed unit neighbour products times the looks / 2 pi,
    which can exceed half a cycle per look cell (up to half a cycle per
    full-resolution pixel) - and their consistency (cons_*: |sum| / count,
    1 = one clean gradient, ~0 = noise), NaN where nothing was measured.
    block > 1 sums over block x block look cells centred on each cell first
    (GRADIENT_BLOCK, why there).
    """
    out = {}
    for name, looks in (("rg", int(rl)), ("az", int(al))):
        S, N = sums[f"S_{name}"], sums[f"N_{name}"]
        if block > 1:
            from scipy.ndimage import uniform_filter
            k = int(block)
            S = (uniform_filter(S.real.astype(np.float64), k, mode="constant")
                 + 1j * uniform_filter(S.imag.astype(np.float64), k, mode="constant")) * k * k
            N = uniform_filter(N.astype(np.float64), k, mode="constant") * k * k
        with np.errstate(invalid="ignore", divide="ignore"):
            ok = N > 0.5
            cons = np.where(ok, np.abs(S) / np.maximum(N, 1), np.nan)
            cyc = np.where(ok, np.abs(np.angle(S)) * looks / (2 * np.pi), np.nan)
        out[f"cyc_{name}"] = cyc.astype(np.float32)
        out[f"cons_{name}"] = cons.astype(np.float32)
    return out


def full_res_gradient(dphi: np.ndarray, al: int, rl: int, next_row: np.ndarray | None = None,
                      block: int = 1) -> dict:
    """Wrapped phase gradient per look cell from FULL-resolution phase
    (full_res_sums, then gradient_from_sums)."""
    return gradient_from_sums(full_res_sums(dphi, al, rl, next_row), al, rl, block)


def looked_gradient(ifg: np.ndarray, win: int = 3) -> dict:
    """Wrapped phase gradient per look cell from a MULTILOOKED interferogram
    (neighbour products averaged over win x win cells). By construction it
    cannot exceed half a cycle per look cell: reported on the classic route,
    never a flag."""
    from scipy.ndimage import uniform_filter
    z = _unit(np.asarray(ifg))
    out = {}
    for name, axis in (("rg", 1), ("az", 0)):
        p = np.zeros_like(z)
        if axis == 1:
            p[:, :-1] = z[:, 1:] * np.conj(z[:, :-1])
        else:
            p[:-1, :] = z[1:, :] * np.conj(z[:-1, :])
        S = uniform_filter(p.real, win) + 1j * uniform_filter(p.imag, win)
        N = uniform_filter((np.abs(p) > 0.5).astype(np.float32), win)
        with np.errstate(invalid="ignore", divide="ignore"):
            out[f"cons_{name}"] = np.where(N > 0, np.abs(S) / np.maximum(N, 1e-6), np.nan).astype(np.float32)
            out[f"cyc_{name}"] = np.where(N > 0, np.abs(np.angle(S)) / (2 * np.pi), np.nan).astype(np.float32)
    return out


def gradient_exceeds(grad: dict, min_cons: float = GRADIENT_MIN_CONSISTENCY) -> np.ndarray:
    """Cells whose consistent wrapped gradient exceeds GRADIENT_LIMIT_CYCLES
    per look cell along range or azimuth."""
    out = None
    for ax in ("rg", "az"):
        c, k = grad[f"cyc_{ax}"], grad[f"cons_{ax}"]
        with np.errstate(invalid="ignore"):
            e = np.nan_to_num(c) > GRADIENT_LIMIT_CYCLES
            e &= np.nan_to_num(k) >= min_cons
        out = e if out is None else (out | e)
    return out


def residues(phase: np.ndarray) -> np.ndarray:
    """Residue charge (-1, 0, +1) of each 2x2 loop of a wrapped phase field,
    shape (L-1, W-1); 0 where any corner is missing."""
    def wrap(x):
        return (x + np.pi) % (2 * np.pi) - np.pi
    p = np.asarray(phase, np.float64)
    a, b, c, d = p[:-1, :-1], p[:-1, 1:], p[1:, 1:], p[1:, :-1]
    s = wrap(b - a) + wrap(c - b) + wrap(d - c) + wrap(a - d)
    q = np.rint(s / (2 * np.pi))
    q[~np.isfinite(q)] = 0
    return q.astype(np.int8)


def ring_kernel(dy: float, dx: float, r0: float, r1: float) -> np.ndarray:
    """Annulus r0..r1 metres on a grid of (dy, dx) metre cells."""
    ny, nx = int(math.ceil(r1 / dy)), int(math.ceil(r1 / dx))
    yy, xx = np.mgrid[-ny:ny + 1, -nx:nx + 1]
    d = np.hypot(yy * dy, xx * dx)
    return ((d >= r0) & (d <= r1)).astype(np.float32)


def ring_median(coh: np.ndarray, reliable: np.ndarray | None, dy: float, dx: float, ring=HOLE_RING_M,
                grid_m: float = HOLE_RING_GRID_M, bins: int = HOLE_RING_BINS) -> tuple[np.ndarray, np.ndarray]:
    """
    (median, count): the median pair coherence of the reliable cells `ring`
    metres around each cell, and how many there are. The ring is taken
    between blocks of b x b cells (b x spacing <= grid_m, 200 m: a fifth of
    the 1 km inner radius) and the coherence in `bins` steps (0.01): a
    histogram per block, summed over the ring of blocks (FFT), read at its
    middle. Each cell takes its block's value. NaN where the ring is empty.
    """
    from scipy.signal import fftconvolve
    c = np.asarray(coh, np.float32)
    H, W = c.shape
    valid = np.isfinite(c) & (c > 0)
    rel = valid if reliable is None else (valid & np.asarray(reliable, bool))
    b = max(1, int(grid_m // max(float(dy), float(dx), 1e-6)))
    by, bx = -(-H // b), -(-W // b)
    rr, cc = np.nonzero(rel)
    q = np.clip((np.nan_to_num(c[rr, cc]) * bins).astype(np.int64), 0, bins - 1)
    flat = q * (by * bx) + (rr // b) * bx + (cc // b)
    hist = np.bincount(flat, minlength=bins * by * bx).reshape(bins, by, bx).astype(np.float64)
    K = ring_kernel(dy * b, dx * b, *ring)
    cum = np.zeros((by, bx))
    med = np.full((by, bx), np.nan)
    tot = np.rint(fftconvolve(hist.sum(axis=0), K, mode="same"))
    half = tot / 2.0
    step = 20                                                # bins per FFT batch: bounded memory
    for k0 in range(0, bins, step):
        part = np.rint(fftconvolve(hist[k0:k0 + step], K[None], mode="same", axes=(1, 2)))
        for k in range(part.shape[0]):
            cum += part[k]
            hit = np.isnan(med) & (tot > 0) & (cum >= half)
            med[hit] = (k0 + k + 0.5) / bins
    up = (np.arange(H) // b)[:, None], (np.arange(W) // b)[None, :]
    return med[up].astype(np.float32), tot[up]


def coherence_holes(coh: np.ndarray, reliable: np.ndarray | None, dy: float, dx: float,
                    ratio: float = HOLE_RATIO, floor: float = HOLE_FLOOR, ring=HOLE_RING_M,
                    min_cells: int = HOLE_MIN_RING_CELLS, block: int = HOLE_BLOCK) -> np.ndarray:
    """
    Cells in a coherence hole as amended (protocol A 2026-10-08/p23-2):
    the cell's pair coherence, averaged over block x block look cells
    centred on it, is at most `ratio` (2/pi) x the median coherence of the
    reliable cells `ring` metres around it in the same pair, where that ring
    holds at least `min_cells` reliable cells and its median reaches `floor`
    (a scene-wide low is not a hole). Relative, not an absolute low.
    """
    from scipy.ndimage import uniform_filter
    c = np.asarray(coh, np.float32)
    valid = np.isfinite(c) & (c > 0)
    if block > 1:
        s = uniform_filter(np.where(valid, c, 0).astype(np.float64), int(block), mode="constant")
        n = uniform_filter(valid.astype(np.float64), int(block), mode="constant")
        with np.errstate(invalid="ignore", divide="ignore"):
            cell = np.where(n > 0, s / np.where(n > 0, n, 1), np.nan)
    else:
        cell = np.where(valid, c, np.nan)
    med, cnt = ring_median(c, reliable, dy, dx, ring)
    with np.errstate(invalid="ignore"):
        ok = (cnt >= min_cells) & np.isfinite(med) & (med >= floor)
        return valid & ok & (np.nan_to_num(cell, nan=np.inf) <= ratio * np.nan_to_num(med))


def coherence_holes_as_first_fixed(coh: np.ndarray, reliable: np.ndarray | None, dy: float, dx: float,
                                   floor: float = HOLE_FLOOR, ring=HOLE_RING_M,
                                   min_cells: int = HOLE_MIN_RING_CELLS) -> np.ndarray:
    """
    The hole as protocol A first fixed it (kept for its version_1 report):
    pair coherence below `floor` while at least half of the reliable cells
    `ring` metres around them reach `floor` (the median test). A scene-wide
    low is not a hole.
    """
    from scipy.signal import fftconvolve
    c = np.asarray(coh, np.float32)
    valid = np.isfinite(c) & (c > 0)
    rel = valid if reliable is None else (valid & reliable)
    hi = rel & (c >= floor)
    K = ring_kernel(dy, dx, *ring)
    n_rel = np.rint(fftconvolve(rel.astype(np.float32), K, mode="same"))
    n_hi = np.rint(fftconvolve(hi.astype(np.float32), K, mode="same"))
    ring_ok = (n_rel >= min_cells) & (2 * n_hi >= n_rel)
    return valid & (c < floor) & ring_ok


def classify(coh: np.ndarray, grad: dict | None, reliable: np.ndarray | None, dy: float, dx: float,
             grad_can_flag: bool, min_cons: float = GRADIENT_MIN_CONSISTENCY) -> tuple[np.ndarray, dict]:
    """Levels 0/1/2 for one pair on the crop (the hole as amended), and what
    decided them; why['level_v1'] is the same with the hole as first fixed."""
    holes = coherence_holes(coh, reliable, dy, dx)
    holes_v1 = coherence_holes_as_first_fixed(coh, reliable, dy, dx)
    level = np.zeros(coh.shape, np.uint8)
    level[holes] = LEVEL_RISK
    level_v1 = np.zeros(coh.shape, np.uint8)
    level_v1[holes_v1] = LEVEL_RISK
    exceeds = np.zeros(coh.shape, bool)
    if grad is not None and grad_can_flag:
        e = gradient_exceeds(grad, min_cons)
        h, w = e.shape
        exceeds[:h, :w] = e
        level[exceeds] = LEVEL_LIKELY
        level_v1[exceeds] = LEVEL_LIKELY
    return level, {"holes": holes, "holes_v1": holes_v1, "exceeds": exceeds, "level_v1": level_v1}


# ======================================================================
# F2: reading a pair
# ======================================================================
def _read_crop(path: Path, dtype, frame, window, bands: int = 1, band: int = 0) -> np.ndarray:
    Lm, Wm = frame
    r0, r1, c0, c1 = window
    need = Lm * Wm * bands * np.dtype(dtype).itemsize
    size = Path(path).stat().st_size
    if size != need:
        raise ValueError(f"{path} is {size} bytes; a {Lm} x {Wm} x {bands} {np.dtype(dtype).name} frame is {need}")
    if bands == 1:
        mm = np.memmap(path, dtype=dtype, mode="r", shape=(Lm, Wm))
        return np.array(mm[r0:r1, c0:c1])
    mm = np.memmap(path, dtype=dtype, mode="r", shape=(Lm, bands, Wm))
    return np.array(mm[r0:r1, band, c0:c1])


def pair_dates(pairs) -> tuple[list, dict]:
    """(sorted dates, {pair: connection rank k = index(B) - index(A)})."""
    ds = sorted({d for p in pairs for d in p.split("_")})
    idx = {d: i for i, d in enumerate(ds)}
    return ds, {p: idx[p.split("_")[1]] - idx[p.split("_")[0]] for p in pairs}


def _days(pair: str) -> int:
    a, b = pair.split("_")
    return (datetime.strptime(b, "%Y%m%d") - datetime.strptime(a, "%Y%m%d")).days


def _sha(path) -> str:
    import hashlib
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]
    except OSError:
        return ""


class _LinkedPhase:
    """Full-resolution linked phase of the phase-linked route
    (miaplpy/inverted/phase_series.h5), a few dates cached, on the crop's
    look grid (aligned_subset: linked look cell (r, c) = crop cell (r, c))."""

    def __init__(self, cfg, keep: int):
        import h5py
        from .miaplpy_backend import aligned_subset
        self.path = _project(cfg) / "miaplpy" / "inverted" / "phase_series.h5"
        self.f = h5py.File(self.path, "r")
        self.ph = self.f["phase"]
        self.dates = [d.decode() if isinstance(d, bytes) else str(d) for d in self.f["date"][:]]
        self.al, self.rl = int(cfg.stack_azimuth_looks), int(cfg.stack_range_looks)
        n, L, W = self.ph.shape
        self.h, self.w = L // self.al, W // self.rl
        y0, y1, x0, x1 = aligned_subset(cfg)
        cw = _crop_window(cfg)
        # the linked crop's first look cell, in crop-window cells (0, 0 by construction)
        self.off = (y0 // self.al - int(cw["r0"]), x0 // self.rl - int(cw["c0"]))
        # MiaplPy chunks the series over a few dates (5 at Site B): one
        # date's read costs the whole chunk, so a chunk's dates are read together
        c = int((self.ph.chunks or (1,))[0])
        self.block = c if 1 <= c <= 8 else 1              # (a deep time chunk: one date at a time, memory first)
        self.cache, self.keep = {}, max(2, int(keep) + 1) + self.block

    def date(self, d: str) -> np.ndarray:
        if d not in self.cache:
            k = self.dates.index(d)
            k0 = (k // self.block) * self.block
            k1 = min(len(self.dates), k0 + self.block)
            arr = self.ph[k0:k1, :self.h * self.al, :self.w * self.rl]
            for i in range(k1 - k0):
                self.cache[self.dates[k0 + i]] = arr[i].astype(np.float32)
            while len(self.cache) > self.keep:
                self.cache.pop(min(x for x in self.cache if x != d))
        return self.cache[d]

    def gradient(self, a: str, b: str, rows: int = 64, block: int = GRADIENT_BLOCK) -> dict:
        """Per look cell: the sums in strips of `rows` look rows (bounded
        memory), then the gradient over block x block look cells."""
        pa, pb = self.date(a), self.date(b)
        al, rl = self.al, self.rl
        parts = {k: [] for k in ("S_rg", "N_rg", "S_az", "N_az")}
        for r in range(0, self.h, rows):
            r1 = min(self.h, r + rows)
            blk = pa[r * al:r1 * al] - pb[r * al:r1 * al]
            nxt = (pa[r1 * al] - pb[r1 * al]) if r1 * al < pa.shape[0] else None
            g = full_res_sums(blk, al, rl, next_row=nxt)
            for k in parts:
                parts[k].append(g[k])
        return gradient_from_sums({k: np.vstack(v) for k, v in parts.items()}, al, rl, block)

    def close(self):
        self.f.close()


def _place(arr: np.ndarray, shape, off) -> np.ndarray:
    """A linked-grid layer on the crop grid (NaN where the linked crop does not reach)."""
    out = np.full(shape, np.nan, np.float32)
    r, c = off
    h = min(arr.shape[0], shape[0] - r)
    w = min(arr.shape[1], shape[1] - c)
    if h > 0 and w > 0:
        out[r:r + h, c:c + w] = arr[:h, :w]
    return out


def pair_coherence_kind(cfg, route: str) -> str:
    """Which coherence filt_fine.cor holds, i.e. what the hole test reads.
    The protocol's 'pair coherence' means different things on the two
    phase-linked settings: the phasor figure runs high (median ~0.77 on the
    Indiana record) and drops where the linked phase is rough or has a steep
    gradient inside a look cell; sample coherence is the standard estimate
    from the two dates' SLCs."""
    if route == "miaplpy":
        kind = "phasor"
        try:
            rec = json.loads((_stack(cfg) / "merged" / "pair_coherence.json").read_text())
            kind = str(rec.get("kind") or kind)
        except (OSError, ValueError, AttributeError):
            kind = str(getattr(cfg, "miaplpy_pair_coherence", "phasor") or "phasor")
        if kind == "sample":
            return ("sample coherence of the two dates' SLCs over each look cell "
                    "(miaplpy_pair_coherence: sample)")
        return ("phasor coherence of the LINKED phase over each look cell (miaplpy_pair_coherence: phasor; "
                "|mean unit phasor|, not the two dates' decorrelation; runs high)")
    return "topsStack's coherence of the filtered pair (filt_fine.cor)"


def _median(a, m):
    v = np.asarray(a)[m] if m is not None else np.asarray(a).ravel()
    v = v[np.isfinite(v)]
    return _r(np.median(v), 3) if v.size else None


def _share(flag, m):
    n = int(np.count_nonzero(m))
    return _r(np.count_nonzero(flag & m) / n, 5) if n else None


def pre_unwrap(cfg, pairs: list, route: str | None = None) -> dict:
    """
    F2 before unwrapping: every exported pair's levels on the radar crop,
    into <project>/aliasing/pairs/<pair>.npz, then the summary
    (<project>/aliasing/summary.json). A pair whose record matches its
    files and settings is not measured again. Never raises: a failure is
    logged and recorded, and the run goes on (the diagnostic changes no
    number; validation then falls back to the old, non-grading flag).
    """
    if not enabled(cfg):
        return {"skipped": "aliasing_diagnostic is off for this config"}
    t0 = time.time()
    try:
        return _pre_unwrap(cfg, list(pairs), route, t0)
    except Exception as exc:                                  # noqa: BLE001 - a diagnostic never stops the run
        log.error("Aliasing diagnostic (F2) failed before unwrapping: %s: %s - the run goes on without it; "
                  "quality_aliasing_risk.tif will not be written", type(exc).__name__, exc, exc_info=True)
        try:
            _write_json(record_dir(cfg) / "summary.json",
                        {"version": VERSION, "error": f"{type(exc).__name__}: {exc}",
                         "written": time.strftime("%Y-%m-%dT%H:%M:%S")})
        except OSError:
            pass
        return {"error": f"{type(exc).__name__}: {exc}"}


def _pre_unwrap(cfg, pairs: list, route, t0: float) -> dict:
    route = route or str(getattr(cfg, "backend", "")).lower()
    grid = radar_grid(cfg)
    shape = grid["lat"].shape
    dy, dx = grid["dy_m"], grid["dx_m"]
    zgj = zones_geojson(cfg)
    zone = polygon_mask(zgj, grid["lat"], grid["lon"]) if zgj else np.zeros(shape, bool)
    ring = ring_around(zone, dy, dx, *ZONE_RING_M)
    rel, rel_how = reliable_mask(cfg, grid)
    fz_gj = str(setting(cfg, "fast_zone_geojson") or "").strip()
    fz = polygon_mask(fz_gj, grid["lat"], grid["lon"]) if fz_gj and Path(fz_gj).exists() else None
    K = int(setting(cfg, "fast_zone_connections"))
    dates, rank = pair_dates(pairs)
    ifg_root = _stack(cfg) / "merged" / "interferograms"
    linked = None
    ps = _project(cfg) / "miaplpy" / "inverted" / "phase_series.h5"
    if route == "miaplpy" and ps.exists():
        conn = max(rank.values()) if rank else 1
        linked = _LinkedPhase(cfg, conn)
        grad_how = "full-resolution linked phase (miaplpy/inverted/phase_series.h5), summed over each look cell"
        cons_min = min_consistency(GRADIENT_BLOCK ** 2 * linked.al * linked.rl)
    else:
        cons_min = GRADIENT_MIN_CONSISTENCY
        grad_how = ("multilooked interferogram (filt_fine.int): a wrapped gradient read there cannot exceed half a "
                    "cycle per look cell, so on this route only coherence holes flag")
    settings_key = json.dumps({"v": VERSION, "zones": _sha(zgj) if zgj else "", "fast": _sha(fz_gj) if fz is not None else "",
                               "K": K, "reliable": rel_how, "grad": "linked" if linked else "looked",
                               "cons_min": round(cons_min, 4), "block": GRADIENT_BLOCK if linked else None,
                               "linked_tc": _sha(_stack(cfg) / "merged" / "linked_temporal_coherence.npz"),
                               "series": _fid(ps) if linked else None}, sort_keys=True)
    done, measured = [], 0
    try:
        for p in sorted(pairs, key=lambda q: (q.split("_")[0], q.split("_")[1])):
            pdir = ifg_root / p
            fi, fc = pdir / "filt_fine.int", pdir / "filt_fine.cor"
            if not (fi.exists() and fc.exists()):
                log.warning("Aliasing diagnostic: %s has no exported interferogram - skipped", p)
                continue
            key = json.dumps({"settings": settings_key, "int": _fid(fi), "cor": _fid(fc)}, sort_keys=True)
            out = _pair_file(cfg, p)
            if out.exists():
                try:
                    with np.load(out) as z:
                        if str(z["key"]) == key:
                            done.append(p)
                            continue
                except (OSError, KeyError, ValueError):
                    pass
            frame, window = grid["frame"], grid["window"]
            coh = _read_crop(fc, np.float32, frame, window)
            ifg = _read_crop(fi, np.complex64, frame, window)
            if linked is not None:
                a, b = p.split("_")
                g = linked.gradient(a, b)
                grad = {k: _place(v, shape, linked.off) for k, v in g.items()}
                can_flag = True
            else:
                grad = looked_gradient(ifg)
                can_flag = False
            level, why = classify(coh, grad, rel, dy, dx, grad_can_flag=can_flag, min_cons=cons_min)
            valid = np.isfinite(coh) & (coh > 0)
            unused = None
            if fz is not None and rank.get(p, 1) > K:
                unused = fz.copy()
                level[unused] = LEVEL_NONE                    # not used there (F3): it cannot alias anything there
                why["level_v1"][unused] = LEVEL_NONE
            res = residues(np.angle(ifg) if np.iscomplexobj(ifg) else ifg)
            res_any = np.zeros(shape, bool)
            res_any[:-1, :-1] = res != 0
            loops_ok = np.zeros(shape, bool)
            loops_ok[:-1, :-1] = valid[:-1, :-1] & valid[1:, 1:] & valid[:-1, 1:] & valid[1:, :-1]
            stats = _pair_stats(p, rank.get(p), level, why, grad, coh, valid, zone, ring, res_any, loops_ok,
                                unused, cons_min)
            out.parent.mkdir(parents=True, exist_ok=True)
            tmp = out.with_name(out.stem + "_tmp.npz")
            np.savez_compressed(tmp, level=level, level_v1=why["level_v1"], key=np.array(key),
                                hole_rule=np.array(json.dumps(HOLE_RULE)),
                                stats=np.array(json.dumps(stats, default=_json_default)))
            os.replace(tmp, out)
            done.append(p)
            measured += 1
    finally:
        if linked is not None:
            linked.close()
    summary = summarise(cfg, done, extra={"gradient_measured_from": grad_how, "reliable_cells": rel_how,
                                          "zones_geojson": str(zgj) if zgj else None,
                                          "fast_zone_geojson": fz_gj or None,
                                          "fast_zone_connections": K if fz is not None else None,
                                          "route": route, "spacing_m": [_r(dy, 1), _r(dx, 1)],
                                          "pair_coherence": pair_coherence_kind(cfg, route),
                                          "gradient_min_consistency_used": _r(cons_min, 3),
                                          "gradient_block_look_cells": GRADIENT_BLOCK if linked else None})
    log.info("Aliasing diagnostic (F2) before unwrapping: %d pairs (%d measured now, %d unchanged) in %.0f s; "
             "%s of the crop flagged in at least one pair (%s likely aliased)", len(done), measured,
             len(done) - measured, time.time() - t0, _pct(summary.get("overall", {}).get("flagged_share")),
             _pct(summary.get("overall", {}).get("likely_share")))
    return summary


def _pct(x) -> str:
    return "n/a" if x is None else f"{100 * float(x):.2f}%"


def _pair_stats(pair, k, level, why, grad, coh, valid, zone, ring, res_any, loops_ok, unused,
                cons_min: float = GRADIENT_MIN_CONSISTENCY) -> dict:
    flagged, likely, hole = level > 0, level == LEVEL_LIKELY, why["holes"]
    hole_v1 = why.get("holes_v1", hole)
    zv, rv = zone & valid, ring & valid
    # report only (never a flag): holes of the rule as first fixed whose 3 x 3
    # neighbourhood is low as well - a patch, not a single speckled cell (the
    # data behind amendment 2026-10-08/p23-2; aliasing_risk.json "hole_rule_note")
    from scipy.ndimage import median_filter
    patch = hole_v1 & (median_filter(np.where(valid, coh, 0).astype(np.float32), size=3) < HOLE_FLOOR)
    lv1 = why.get("level_v1")
    out = {"pair": pair, "days": _days(pair), "connection": k, "cells_with_data": int(valid.sum()),
           "flagged_share": _share(flagged, valid), "likely_share": _share(likely, valid),
           "hole_share": _share(hole, valid), "hole_share_as_first_fixed": _share(hole_v1, valid),
           "flagged_share_as_first_fixed": _share(lv1 > 0, valid) if lv1 is not None else None,
           "hole_share_in_low_3x3_patch": _share(patch, valid),
           "coherence_median": {"crop": _median(coh, valid), "zones": _median(coh, zv), "ring": _median(coh, rv)},
           "residue_density": {"crop": _share(res_any, loops_ok), "zones": _share(res_any, loops_ok & zone),
                               "ring": _share(res_any, loops_ok & ring)}}
    if zone.any():
        out["zones"] = {"cells_with_data": int(zv.sum()), "flagged_share": _share(flagged, zv),
                        "likely_share": _share(likely, zv), "hole_share": _share(hole, zv)}
    if grad is not None:
        g = {}
        for ax in ("rg", "az"):
            c, s = grad[f"cyc_{ax}"], grad[f"cons_{ax}"]
            m = np.isfinite(c) & np.isfinite(s) & (s >= cons_min)
            # share_consistent: the cells whose gradient could be judged at all
            # (elsewhere level 2 is impossible, so a 0 there means "not judged")
            g[ax] = {"p99_cycles_per_look_cell_crop": _r(np.percentile(c[m], 99), 3) if m.any() else None,
                     "max_cycles_per_look_cell_zones": _r(np.max(c[m & zone]), 3) if (m & zone).any() else None,
                     "share_consistent": _share(m, valid),
                     "share_consistent_zones": _share(m, zv) if zone.any() else None}
        out["gradient"] = g
    if unused is not None:
        out["fast_zone_unused_cells"] = int(unused.sum())
    return out


# ======================================================================
# F2: summary on the radar crop
# ======================================================================
LEVEL_VERSIONS = ("amended", "as_first_fixed")


def level_key(z, version: str = "amended") -> str:
    """The array of a per-pair record that holds the flag under `version`:
    'level' (the hole as amended, 2026-10-08/p23-2), or 'level_v1' (the hole
    as protocol A first fixed it). A record written before the amendment has
    only 'level', made with the rule as first fixed: it answers
    'as_first_fixed' and, for 'amended', is reported by record_hole_rule."""
    if version not in LEVEL_VERSIONS:
        raise ValueError(f"level version {version!r}: use one of {LEVEL_VERSIONS}")
    return "level_v1" if version == "as_first_fixed" and "level_v1" in z.files else "level"


def record_hole_rule(z) -> dict | None:
    """The hole rule a per-pair record's 'level' was made with (None: written
    before amendment 2026-10-08/p23-2, i.e. the rule as first fixed)."""
    if "hole_rule" not in z.files:
        return None
    try:
        return json.loads(str(z["hole_rule"]))
    except ValueError:
        return None


def load_levels(cfg, pairs: list | None = None, version: str = "amended") -> tuple[list, np.ndarray | None, list]:
    """(pairs, levels (n, H, W) uint8, stats) from <project>/aliasing/pairs/;
    `version` 'as_first_fixed' reads the level made with the hole rule as
    protocol A first fixed it (level_key)."""
    d = record_dir(cfg) / "pairs"
    names = sorted(pairs) if pairs is not None else sorted(f.stem for f in d.glob("*.npz")
                                                          if PAIR_RE.match(f.stem))
    got, arrs, stats = [], [], []
    for p in names:
        f = d / f"{p}.npz"
        if not f.exists():
            continue
        with np.load(f) as z:
            arrs.append(z[level_key(z, version)])
            stats.append(json.loads(str(z["stats"])))
        got.append(p)
    return got, (np.stack(arrs) if arrs else None), stats


_LEVELS: dict = {}


def _summary_pairs(cfg) -> list | None:
    """The pairs of the last pre-unwrap call (summary.json), or None when
    there is no readable summary. A pair file left in aliasing/pairs/ by an
    earlier network or a partial measurement is not one of them."""
    s = record_dir(cfg) / "summary.json"
    try:
        names = [p["pair"] for p in json.loads(s.read_text()).get("pairs") or [] if p.get("pair")]
    except (OSError, ValueError, TypeError, KeyError):
        return None
    return names or None


def _levels_cached(cfg) -> tuple[list, np.ndarray | None]:
    """load_levels for the harness's repeated risk_between calls: read once
    per state of the record (summary.json is rewritten after every change).
    The same pairs as quality_aliasing_risk.tif (the summary's), so one
    interval never flags more than the whole-record layer does."""
    s = record_dir(cfg) / "summary.json"
    key = (str(record_dir(cfg)), tuple(_fid(s)) if s.exists() else None)
    if key not in _LEVELS:
        _LEVELS.clear()
        names, L, _ = load_levels(cfg, _summary_pairs(cfg))
        _LEVELS[key] = (names, L)
    return _LEVELS[key]


def _spanning(pairs: list, d0: str, d1: str) -> list:
    """Indices of pairs spanning part of (d0, d1): A < d1 and B > d0."""
    return [i for i, p in enumerate(pairs) if p.split("_")[0] < d1 and p.split("_")[1] > d0]


def summarise(cfg, pairs: list | None = None, extra: dict | None = None) -> dict:
    """<project>/aliasing/summary.json and max_level.npz from the per-pair
    records: per pair, per interval between consecutive dates (the union of
    the pairs spanning it), and over the whole record."""
    names, L, stats = load_levels(cfg, pairs)
    rec = {"version": VERSION, "written": time.strftime("%Y-%m-%dT%H:%M:%S"), **(extra or {})}
    old = record_dir(cfg) / "summary.json"
    if not extra and old.exists():
        try:
            prev = json.loads(old.read_text())
            rec.update({k: prev[k] for k in ("gradient_measured_from", "reliable_cells", "zones_geojson",
                                             "fast_zone_geojson", "fast_zone_connections", "route", "spacing_m",
                                             "pair_coherence",
                                             "gradient_min_consistency_used", "gradient_block_look_cells")
                        if k in prev})
        except (OSError, ValueError):
            pass
    if L is None:
        rec.update({"pairs": [], "intervals": [], "overall": {}, "note": "no pair measured"})
        _write_json(old, rec)
        return rec
    grid = radar_grid(cfg, save=False)
    zgj = zones_geojson(cfg)
    zone = polygon_mask(zgj, grid["lat"], grid["lon"]) if zgj else np.zeros(L.shape[1:], bool)
    mx = L.max(axis=0)
    likely_any = (L == LEVEL_LIKELY).any(axis=0)
    tmp = record_dir(cfg) / "max_level_tmp.npz"
    np.savez_compressed(tmp, level=mx, pairs=np.array(names))
    os.replace(tmp, record_dir(cfg) / "max_level.npz")
    have = np.isfinite(grid["lat"])
    dates = sorted({d for p in names for d in p.split("_")})
    intervals, run = [], np.zeros(mx.shape, bool)
    for a, b in zip(dates[:-1], dates[1:]):
        idx = _spanning(names, a, b)
        u = (L[idx] > 0).any(axis=0) if idx else np.zeros(mx.shape, bool)
        run |= u
        intervals.append({"from": a, "to": b, "pairs": len(idx), "flagged_share": _share(u, have),
                          "flagged_share_zones": _share(u, have & zone) if zone.any() else None,
                          "flagged_share_since_first_date": _share(run, have)})
    # per epoch: the pairs that start or end on that date
    epochs = []
    for d in dates:
        idx = [i for i, p in enumerate(names) if d in p.split("_")]
        u = (L[idx] > 0).any(axis=0)
        epochs.append({"date": d, "pairs": len(idx), "flagged_share": _share(u, have),
                       "likely_share": _share((L[idx] == LEVEL_LIKELY).any(axis=0), have),
                       "flagged_share_zones": _share(u, have & zone) if zone.any() else None})
    # the same, with the hole as protocol A first fixed it (reported beside, never the flag)
    _, L1, _ = load_levels(cfg, names, "as_first_fixed")
    mx1 = L1.max(axis=0) if L1 is not None else None
    iv1 = []
    if L1 is not None:
        for a, b in zip(dates[:-1], dates[1:]):
            idx = _spanning(names, a, b)
            iv1.append(_share((L1[idx] > 0).any(axis=0), have) if idx else 0.0)
    rec.update({
        "hole_rule": HOLE_RULE,
        "pairs": stats,
        "intervals": intervals,
        "epochs": epochs,
        "overall": {"pairs": len(names), "dates": len(dates), "first": dates[0], "last": dates[-1],
                    "flagged_share": _share(mx > 0, have), "likely_share": _share(likely_any, have),
                    "flagged_share_zones": _share(mx > 0, have & zone) if zone.any() else None,
                    "likely_share_zones": _share(likely_any, have & zone) if zone.any() else None,
                    "interval_flagged_share_median": _r(np.median([i["flagged_share"] for i in intervals
                                                                   if i["flagged_share"] is not None]), 5)
                    if intervals else None},
        "as_first_fixed": ({"what": "the same record with the coherence hole as protocol A first fixed it (any cell "
                                    "below 0.4 while half the ring reaches 0.4): reported beside, never the flag",
                            "flagged_share": _share(mx1 > 0, have),
                            "interval_flagged_share_median": _r(np.median([x for x in iv1 if x is not None]), 5)
                            if iv1 else None}
                           if mx1 is not None else None),
    })
    _write_json(old, rec)
    return rec


# ======================================================================
# F2: onto the product grid
# ======================================================================
def _product_grid(cfg):
    from .deliverables import _product, _read
    vel, attrs = _read(_product(cfg, "velocity.h5"), "velocity")
    return vel.shape, attrs


def geo_index(cfg, attrs: dict, shape, grid: dict | None = None) -> np.ndarray:
    """For each product-grid cell, the flat index of the nearest radar-crop
    cell centre, or -1 where none lies within 0.75 x the cell diagonal
    (nearest-neighbour, as MintPy geocodes)."""
    from scipy.spatial import cKDTree
    from .deliverables import _rc_to_lonlat
    grid = grid or radar_grid(cfg, save=False)
    la, lo = grid["lat"], grid["lon"]
    ok = np.isfinite(la) & np.isfinite(lo)
    lat0 = float(np.nanmedian(la))
    x, y = _local_xy(la[ok], lo[ok], lat0)
    tree = cKDTree(np.column_stack([x, y]))
    flat = np.flatnonzero(ok.ravel())
    ny, nx = shape
    out = np.full(ny * nx, -1, np.int64)
    maxd = 0.75 * math.hypot(grid["dy_m"], grid["dx_m"])
    step = max(1, 2_000_000 // max(nx, 1))
    for r0 in range(0, ny, step):
        rr, cc = np.mgrid[r0:min(ny, r0 + step), 0:nx]
        glon, glat = _rc_to_lonlat(attrs, rr.ravel(), cc.ravel())
        gx, gy = _local_xy(glat, glon, lat0)
        dist, i = tree.query(np.column_stack([gx, gy]), distance_upper_bound=maxd)
        hit = np.isfinite(dist)
        seg = np.full(rr.size, -1, np.int64)
        seg[hit] = flat[i[hit]]
        out[r0 * nx:r0 * nx + rr.size] = seg
    return out.reshape(shape)


def to_product_grid(level: np.ndarray, index: np.ndarray) -> np.ndarray:
    out = np.full(index.shape, NODATA, np.uint8)
    hit = index >= 0
    out[hit] = level.ravel()[index[hit]]
    return out


def risk_between(cfg, d0: str, d1: str, index: np.ndarray | None = None) -> np.ndarray | None:
    """
    The flag for one comparison interval, on the product grid: the highest
    level over the exported pairs spanning part of (d0, d1) (protocol A:
    "flagged for a comparison interval when, in any exported interferogram
    spanning part of that interval ..."). d0, d1 as YYYYMMDD or YYYY-MM-DD.
    255 outside the radar crop. None without a record.
    """
    d0, d1 = sorted((str(d0).replace("-", "")[:8], str(d1).replace("-", "")[:8]))    # either order
    names, L = _levels_cached(cfg)
    if L is None:
        return None
    idx = _spanning(names, d0, d1)
    lv = L[idx].max(axis=0) if idx else np.zeros(L.shape[1:], np.uint8)
    if index is None:
        index = _index_cached(cfg)
    return to_product_grid(lv, index)


_INDEX: dict = {}


def _index_cached(cfg) -> np.ndarray:
    """geo_index on the product grid, once per product and radar record."""
    from .deliverables import _product
    vp = _product(cfg, "velocity.h5")
    rg = record_dir(cfg) / "radar_grid.npz"
    key = (str(vp), tuple(_fid(vp)), tuple(_fid(rg)) if rg.exists() else None)
    if key not in _INDEX:
        _INDEX.clear()
        shape, attrs = _product_grid(cfg)
        _INDEX[key] = geo_index(cfg, attrs, shape)
    return _INDEX[key]


def export_products(cfg, out_dir=None) -> dict:
    """
    deliverables/quality_aliasing_risk.tif and aliasing_risk.json from the
    pre-unwrap record, on the product grid, and the old too-fast flag
    relabelled a lower bound. Needs the MintPy products (after the time
    series). Returns what was written, or why nothing was. Never raises.
    """
    out_dir = Path(out_dir) if out_dir else _project(cfg) / "deliverables"
    if not enabled(cfg):
        r = {"skipped": "aliasing_diagnostic is off for this config"}
    else:
        try:
            r = _export(cfg, out_dir)
        except Exception as exc:                              # noqa: BLE001
            log.error("quality_aliasing_risk.tif not written: %s: %s", type(exc).__name__, exc, exc_info=True)
            r = {"error": f"{type(exc).__name__}: {exc}"}
    if "tif" not in r:
        # an earlier run's flag must not be read (by the harness or events.py) as this run's
        try:
            from .deliverables import _set_aside
            moved = _set_aside(out_dir, list(EXPORT_NAMES), "aliasing",
                               "this run wrote no aliasing flag: " + str(r.get("skipped") or r.get("error")))
            if moved:
                r["set_aside"] = [str(p) for p in moved]
        except Exception as exc:                              # noqa: BLE001
            log.warning("An earlier quality_aliasing_risk.tif could not be moved aside: %s", exc)
    return r


EXPORT_NAMES = ("quality_aliasing_risk.tif", "aliasing_risk.json")


def record_check(cfg, names: list) -> dict:
    """Does the pre-unwrap record still describe the interferograms MintPy
    read? Pairs unwrapped now but never measured, and measured pairs whose
    interferogram was rewritten since (re-exported). Not checkable
    ("checked": False) once cleanup has removed stack/merged."""
    root = _stack(cfg) / "merged" / "interferograms"
    if not root.is_dir():
        return {"checked": False, "why": "stack/merged/interferograms is gone (cleanup): not checkable"}
    from . import runs
    now = set(runs.unwrap_status(_stack(cfg))["unwrapped"])
    rewritten = []
    for p in names:
        fi = root / p / "filt_fine.int"
        try:
            with np.load(_pair_file(cfg, p)) as z:
                k = json.loads(str(z["key"]))
        except (OSError, KeyError, ValueError):
            continue
        if fi.exists() and k.get("int") != _fid(fi):
            rewritten.append(p)
    missing = sorted(now - set(names))
    was, cur = _fast_key_measured(cfg), _fast_key_now(cfg)
    fast_ok = was is None or _same_fast(was, cur)
    return {"checked": True, "complete": not (missing or rewritten) and fast_ok, "unwrapped_not_measured": missing,
            "rewritten_since_measured": sorted(rewritten), "fast_zones_as_measured": fast_ok}


def _export(cfg, out_dir: Path) -> dict:
    from .deliverables import _write_geotiff, product_tags
    srec = record_dir(cfg) / "summary.json"
    if not srec.exists():
        return {"skipped": f"no pre-unwrap record ({srec}): the diagnostic runs before unwrapping, so this "
                           "project's interferograms were unwrapped without it"}
    S = json.loads(srec.read_text())
    if S.get("error"):
        return {"skipped": f"the pre-unwrap diagnostic failed: {S['error']}"}
    names, L, _ = load_levels(cfg, [p["pair"] for p in S.get("pairs", [])] or None)
    if L is None:
        return {"skipped": "the pre-unwrap record holds no pair"}
    chk = record_check(cfg, names)
    if chk.get("checked") and not chk["complete"]:
        log.warning("The aliasing record does not cover every pair MintPy read: %d unwrapped pair(s) never "
                    "measured, %d re-exported since, fast zones %s (aliasing_risk.json 'record_check'); their "
                    "cells may be flagged too little", len(chk["unwrapped_not_measured"]),
                    len(chk["rewritten_since_measured"]),
                    "as measured" if chk.get("fast_zones_as_measured", True) else "CHANGED since measured")
    shape, attrs = _product_grid(cfg)
    index = geo_index(cfg, attrs, shape)
    flag = to_product_grid(L.max(axis=0), index)
    out_dir.mkdir(parents=True, exist_ok=True)
    lam, lam_src = wavelength(cfg, attrs)
    q_mm = 1000.0 * lam / 4.0
    cmin = float(S.get("gradient_min_consistency_used") or GRADIENT_MIN_CONSISTENCY)
    tif = out_dir / "quality_aliasing_risk.tif"
    desc = ("Aliasing risk before unwrapping (F2): 0 = no sign of aliasing found (NOT proof the value is right), "
            "1 = risk: coherence hole (pair coherence over 3 x 3 look cells at most 2/pi of the median of the "
            "reliable ground 1-3 km around, that median >= 0.4; protocol A amendment 2026-10-08/p23-2), "
            "2 = likely aliased: wrapped phase gradient above half a cycle per look cell in a pair. The highest "
            "level over every exported pair; aliasing.risk_between(cfg, d0, d1) gives one interval. "
            "255 = outside the processed radar crop. A lower bound on the aliased ground.")
    _write_geotiff(tif, flag, attrs, nodata=NODATA, dtype="uint8",
                   tags=product_tags(cfg, attrs, "aliasing risk before unwrapping (F2)",
                                     "flag (0 none found, 1 risk, 2 likely aliased)", "not applicable",
                                     masked=False, sign="not applicable (a flag)",
                                     lower_bound="yes: 0 means no evidence found, not proof",
                                     gradient_limit_cycles_per_look_cell=GRADIENT_LIMIT_CYCLES,
                                     gradient_min_consistency=cmin,
                                     hole_floor=HOLE_FLOOR, hole_ring_m=f"{HOLE_RING_M[0]:.0f}-{HOLE_RING_M[1]:.0f}",
                                     hole_ratio=f"{HOLE_RATIO:.4f}", hole_cell=f"mean over {HOLE_BLOCK} x {HOLE_BLOCK} look cells",
                                     quarter_wavelength_mm=f"{q_mm:.2f}", wavelength_m=f"{lam:.8f}",
                                     wavelength_source=lam_src, pairs=len(names), method_version=VERSION),
                   description=desc, units="flag")
    have = flag != NODATA
    old = _relabel_old_flag(out_dir)
    try:
        from .resolution import effective_resolution
        look = effective_resolution(cfg, attrs).get("look_cell_m")
    except Exception:                                         # noqa: BLE001
        look = None
    js = {
        "file": tif.name, "written": time.strftime("%Y-%m-%dT%H:%M:%S"), "version": VERSION,
        "values": {str(k): v for k, v in LEVEL_TEXT.items()} | {str(NODATA): "outside the processed radar crop"},
        "lower_bound": ("The flagged area is a LOWER BOUND on the aliased ground. A wrapped gradient beyond half a "
                        "cycle per full-resolution pixel wraps again, the linked phase is smoothed over the "
                        "phase-linking window, and the classic route cannot see gradients above half a cycle per "
                        "look cell at all. 0 means no evidence of aliasing was found, not that the value is right."),
        "method": {
            "when": "before unwrapping, on every exported interferogram (miaplpy_backend.unwrap_exported / "
                    "stack_backend.execute)",
            "gradient": S.get("gradient_measured_from"),
            "gradient_block_look_cells": S.get("gradient_block_look_cells"),
            "gradient_flag": f"> {GRADIENT_LIMIT_CYCLES} cycle per look cell along range or azimuth, where the "
                             f"gradient is consistent (|mean unit neighbour product| >= {cmin:g}: "
                             f"{GRADIENT_MIN_CONSISTENCY}, or pure noise's 99.9th percentile with few looks)",
            "coherence_hole": f"pair coherence averaged over {HOLE_BLOCK} x {HOLE_BLOCK} look cells at most "
                              f"{HOLE_RATIO:.4f} (2/pi) x the median pair coherence of the reliable cells "
                              f"{HOLE_RING_M[0]:.0f}-{HOLE_RING_M[1]:.0f} m around (at least {HOLE_MIN_RING_CELLS} "
                              f"of them, median >= {HOLE_FLOOR}; a scene-wide low is not a hole): relative to its "
                              "ring, protocol A amendment 2026-10-08/p23-2. 2/pi = sinc(1/2), what a phase ramp of "
                              "half a cycle across a look cell does to its coherence",
            "coherence_hole_as_first_fixed": f"pair coherence < {HOLE_FLOOR} while at least half of the reliable "
                                             f"cells {HOLE_RING_M[0]:.0f}-{HOLE_RING_M[1]:.0f} m around reach "
                                             f"{HOLE_FLOOR}: per pair in the record as level_v1, reported in "
                                             "hole_rule_note, never the flag",
            "pair_coherence": S.get("pair_coherence"),
            "gradient_judged_where": ("only where the gradient is consistent (pairs[].gradient.*.share_consistent, "
                                      "share_consistent_zones): with per-pixel phase noise above ~0.6 rad no "
                                      "gradient can reach level 2, so 0 there means not judged"),
            "reliable_cells": S.get("reliable_cells"),
            "residue_density": "share of 2x2 loops of the wrapped multilooked phase with a residue; reported "
                               "per pair inside the zones, in a 1-3 km ring and over the crop, never a flag",
            "flag_definition_source": "review_2026-10-08/protocol_A.yaml comparisons.unflagged_failure_rate."
                                      "flag_definition (fixed 8 Oct 2026, before any data)",
            "to_product_grid": "nearest radar-crop cell centre within 0.75 x the look-cell diagonal",
            "per_interval": "aliasing.risk_between(cfg, d0, d1): highest level over the pairs spanning part of "
                            "the interval. This file is the whole record (every pair), so it flags more than "
                            "any one interval",
        },
        "thresholds": {"gradient_limit_cycles_per_look_cell": GRADIENT_LIMIT_CYCLES,
                       "gradient_min_consistency": cmin, "hole_floor": HOLE_FLOOR,
                       "hole_ratio": _r(HOLE_RATIO, 4), "hole_block_look_cells": HOLE_BLOCK,
                       "hole_ring_m": list(HOLE_RING_M), "hole_min_ring_cells": HOLE_MIN_RING_CELLS,
                       "zone_ring_m": list(ZONE_RING_M)},
        "hole_rule": S.get("hole_rule"),
        "wavelength_m": lam, "wavelength_source": lam_src, "quarter_wavelength_mm": _r(q_mm, 2),
        "look_cell_m": look,
        "tilt_limit_mm_per_m_per_interval": (_r(q_mm / max(look), 3) if look else None),
        "route": S.get("route"), "zones_geojson": S.get("zones_geojson"),
        "fast_zones": ({"geojson": S.get("fast_zone_geojson"), "connections": S.get("fast_zone_connections"),
                        "note": ("inside the fast zones the longer pairs were not used, so they cannot flag there "
                                 "(protocol A's flag counts 'any exported interferogram': a graded run with fast "
                                 "zones would need that read as 'any interferogram used there'); with one "
                                 "connection MintPy's temporal coherence there is 1 by construction, so it cannot "
                                 "grade those cells")}
                       if S.get("fast_zone_geojson") else None),
        "product_grid": {"cells": int(flag.size), "covered": int(have.sum()),
                         "flagged_share": _share(flag > 0, have) if have.any() else None,
                         "likely_share": _share(flag == LEVEL_LIKELY, have) if have.any() else None},
        "overall_radar_crop": S.get("overall"),
        "hole_rule_note": {**_hole_note(S.get("pairs") or [], js_share(flag, have), S.get("as_first_fixed")),
                           "pair_coherence_read": S.get("pair_coherence")},
        "record_check": chk,
        "fast_zones_stale": S.get("fast_zones_stale"),
        "pairs": S.get("pairs"),
        "intervals": S.get("intervals"),
        "epochs": S.get("epochs"),
        "old_flag": old,
        "harness": {"flag_file": tif.name, "flagged_values": [LEVEL_RISK, LEVEL_LIKELY],
                    "per_interval_function": "insar_pipeline.aliasing.risk_between(cfg, d0, d1)",
                    "hole_rule": S.get("hole_rule"),
                    "as_first_fixed": "the per-pair record's level_v1 (aliasing.load_levels(..., 'as_first_fixed'))"},
    }
    _write_json(out_dir / "aliasing_risk.json", js)
    log.info("Wrote %s and aliasing_risk.json: %s of the covered cells flagged in at least one pair (%s likely "
             "aliased); a lower bound", tif.name, _pct(js["product_grid"]["flagged_share"]),
             _pct(js["product_grid"]["likely_share"]))
    return {"tif": str(tif), "json": str(out_dir / "aliasing_risk.json"), "pairs": len(names),
            "flagged_share": js["product_grid"]["flagged_share"], "old_flag": old}


def js_share(flag: np.ndarray, have: np.ndarray):
    return _share(flag > 0, have) if have.any() else None


def _hole_note(pairs: list, whole_record_share, as_first_fixed: dict | None = None) -> dict:
    """How much the coherence-hole rule flags here, per pair and over the
    record: the amended rule (the flag) beside the rule as first fixed and a
    report-only reading of it (a low 3 x 3 patch). The rule as first fixed,
    MEASURED on Site B farmland (2019, 81 pairs, phasor pair coherence;
    review_2026-10-08/p23/alias_smoke): 3-36% of cells per pair, 93% of the
    crop over the year - a flag that covers most points makes protocol A's
    unflagged-failure row uninformative, hence amendment 2026-10-08/p23-2
    (relative hole) and the harness's selectivity check."""
    h = [p.get("hole_share") for p in pairs if p.get("hole_share") is not None]
    h1 = [p.get("hole_share_as_first_fixed") for p in pairs if p.get("hole_share_as_first_fixed") is not None]
    q = [p.get("hole_share_in_low_3x3_patch") for p in pairs if p.get("hole_share_in_low_3x3_patch") is not None]
    out = {"rule": "amendment 2026-10-08/p23-2: relative to the ring (aliasing_risk.json method.coherence_hole)",
           "hole_share_per_pair_median": _r(np.median(h), 4) if h else None,
           "hole_share_per_pair_max": _r(np.max(h), 4) if h else None,
           "as_first_fixed_hole_share_per_pair_median": _r(np.median(h1), 4) if h1 else None,
           "as_first_fixed_hole_share_per_pair_max": _r(np.max(h1), 4) if h1 else None,
           "as_first_fixed_whole_record_flagged_share": (as_first_fixed or {}).get("flagged_share"),
           "report_only_low_3x3_patch_share_per_pair_median": _r(np.median(q), 4) if q else None,
           "whole_record_flagged_share": whole_record_share}
    if whole_record_share is not None and whole_record_share > 0.5:
        out["warning"] = ("the flag covers most of the ground over the whole record: per comparison interval use "
                          "aliasing.risk_between (the pairs spanning that interval only), and read the flagged share "
                          "beside the unflagged-failure rate (protocol A reported_beside_it)")
    return out


OLD_FLAG = "quality_too_fast_to_measure.tif"
OLD_FLAG_LABEL = ("LOWER BOUND: computed from the UNWRAPPED velocity, so a cell that aliased (and therefore reads "
                  "too slow) is missed; quality_aliasing_risk.tif is the pre-unwrap check")


def _relabel_old_flag(out_dir: Path) -> dict:
    """Mark quality_too_fast_to_measure.tif a lower bound (its tags and band
    description; its values are untouched)."""
    hits = [p for p in (out_dir / OLD_FLAG, out_dir / "technical" / OLD_FLAG) if p.exists()]
    if not hits:
        return {"file": OLD_FLAG, "present": False, "label": "lower bound (unwrapped velocity)"}
    import rasterio
    done, already = [], []
    for p in hits:
        try:
            with rasterio.open(p) as ds:
                tags = {k.upper(): v for k, v in ds.tags().items()}
                d = ds.descriptions[0] or ""
            if "lower bound" in d.lower() and (tags.get("LOWER_BOUND") or "").strip():
                already.append(str(p))           # deliverables.export_extra_layers labels it since 8 Oct 2026
                continue
            with rasterio.open(p, "r+") as ds:
                if "lower bound" not in d.lower():
                    ds.set_band_description(1, f"LOWER BOUND - {d}" if d else OLD_FLAG_LABEL)
                ds.update_tags(LOWER_BOUND=OLD_FLAG_LABEL, SEE_ALSO="quality_aliasing_risk.tif")
            done.append(str(p))
        except Exception as exc:                              # noqa: BLE001
            log.warning("%s not relabelled a lower bound (%s)", p, exc)
    return {"file": OLD_FLAG, "present": True, "relabelled": done, "already_labelled": already,
            "label": OLD_FLAG_LABEL}


# ======================================================================
# F1: detectability preflight
# ======================================================================
def _gaps(dates: list) -> list:
    d = sorted({str(x).replace("-", "")[:8] for x in dates})
    t = [datetime.strptime(x, "%Y%m%d") for x in d]
    return [(b - a).days for a, b in zip(t[:-1], t[1:])]


def _spans(dates: list, k: int) -> list:
    d = sorted({str(x).replace("-", "")[:8] for x in dates})
    t = [datetime.strptime(x, "%Y%m%d") for x in d]
    return [(t[i + k] - t[i]).days for i in range(len(t) - k)]


def _connections(cfg) -> int:
    b = str(getattr(cfg, "backend", "")).lower()
    if b == "miaplpy":
        return max(1, int(getattr(cfg, "miaplpy_network_connections", 3) or 1))
    if b == "stack":
        return max(1, int(getattr(cfg, "stack_num_connections", 3) or 1))
    if b == "hyp3":
        return max(1, int(getattr(cfg, "n_neighbors", 1) or 1))
    return 1                                          # NISAR GUNW chains: nearest only (DISP-S1: see intervals_note)


def _halfwidth(poly) -> float:
    """Half the narrow side of the polygon's minimum rotated rectangle, in metres."""
    import shapely
    c = poly.centroid
    xs, ys = poly.exterior.coords.xy if poly.geom_type == "Polygon" else max(poly.geoms, key=lambda g: g.area).exterior.coords.xy
    x, y = _local_xy(np.array(ys), np.array(xs), float(c.y))
    from shapely.geometry import Polygon
    r = shapely.minimum_rotated_rectangle(Polygon(np.column_stack([x, y])))
    q = np.asarray(r.exterior.coords)
    sides = [math.hypot(*(q[i + 1] - q[i])) for i in range(min(4, len(q) - 1))]
    return 0.5 * min(sides) if sides else float("nan")


def detectability(cfg, dates: list | None = None, attrs: dict | None = None) -> dict:
    """
    F1. Can this sensor and network follow the expected tilt, before any run?
    lambda/4 between neighbouring look cells per interval, against the
    expected change across one look cell per interval: rate x interval x
    cos(incidence) x look cell / half-width (a bowl going from zero at the
    edge to its peak rate at the centre). Per zone (polygon properties
    max_rate_mm_yr / halfwidth_m override the config) and interval kind:
    nearest pairs (median, longest) and the network's longest connection.
    """
    from .resolution import effective_resolution
    lam, lam_src = wavelength(cfg, attrs)
    q_mm = 1000.0 * lam / 4.0
    res = effective_resolution(cfg, attrs)
    look = [float(v) for v in (res.get("look_cell_m") or [41.0, 42.0])]
    support = [float(v) for v in (res.get("support_m") or look)]
    cell = max(look)
    a = attrs or {}
    inc = _pos(a.get("CENTER_INCIDENCE_ANGLE"))
    inc_src = "product attributes" if inc else "nominal (no product yet)"
    inc = inc or NOMINAL_INCIDENCE_DEG
    los = math.cos(math.radians(inc))
    N = _connections(cfg)
    if dates and len(dates) > 1:
        g = _gaps(dates)
        spans = _spans(dates, N) if len(set(dates)) > N else g
        intervals = {"nearest pairs (median)": float(np.median(g)), "nearest pairs (longest)": float(max(g))}
        if N > 1:
            intervals[f"{N}-connection pairs (median)"] = float(np.median(spans))
            intervals[f"{N}-connection pairs (longest)"] = float(max(spans))
        int_src = f"{len(set(dates))} dates"
    else:
        rv = NOMINAL_REVISIT_DAYS[_sensor(cfg)]
        intervals = {"nearest pairs (nominal revisit)": rv}
        if N > 1:
            intervals[f"{N}-connection pairs (nominal)"] = rv * N
        int_src = "nominal revisit (no dates given)"
    rate = float(setting(cfg, "expected_max_rate_mm_yr") or 0.0)
    hw_cfg = float(setting(cfg, "expected_trough_halfwidth_m") or 0.0)
    zones = []
    zgj = zones_geojson(cfg)
    for name, poly, props in (_polygons(zgj) if zgj else []):
        r = _pos(props.get("max_rate_mm_yr")) or rate
        hw = _pos(props.get("halfwidth_m")) or hw_cfg or _halfwidth(poly)
        how = ("polygon property" if _pos(props.get("halfwidth_m")) else
               "expected_trough_halfwidth_m" if hw_cfg else "half the polygon's narrow side")
        if r:
            zones.append({"zone": name, "rate_mm_yr": r, "halfwidth_m": hw, "halfwidth_from": how})
    if not zones and rate:
        zones.append({"zone": "expected", "rate_mm_yr": rate, "halfwidth_m": hw_cfg or None,
                      "halfwidth_from": "expected_trough_halfwidth_m" if hw_cfg else "not set"})
    for z in zones:
        z["intervals"] = {}
        hw = z.get("halfwidth_m")
        if not hw or not math.isfinite(hw):
            z["note"] = "no trough half-width: set expected_trough_halfwidth_m or give the polygons"
            continue
        for kind, days in intervals.items():
            dd = z["rate_mm_yr"] * days / 365.25 * los * min(1.0, cell / hw)
            ratio = dd / q_mm
            z["intervals"][kind] = {"days": days, "cell_to_cell_mm": _r(dd, 2), "ratio_to_limit": _r(ratio, 3),
                                    "verdict": ("measurable" if ratio < MEASURABLE_BELOW else
                                                "marginal" if ratio < MARGINAL_BELOW else "too fast: aliased")}
        near = next(iter(intervals.values()))
        # (a trough narrower than one look cell changes by its whole depth
        # between neighbouring cells: the same cap as the ratio above)
        z["fastest_followable_mm_yr_nearest"] = _r(q_mm * max(hw, cell) / (cell * near / 365.25 * los), 0)
        if str(getattr(cfg, "backend", "")).lower() == "miaplpy":
            sup = max(support)
            z["support_change_mm_nearest"] = _r(z["rate_mm_yr"] * near / 365.25 * los * min(1.0, sup / hw), 1)
    note = None
    if str(getattr(cfg, "backend", "")).lower() == "disp":
        # OPERA, not this pipeline, unwraps DISP-S1: each product is the change
        # since its mini-stack reference date (disp_backend: reset about every
        # six months), so the interval it had to follow may be far longer
        note = ("DISP-S1: OPERA unwraps these, and each product is the change since its mini-stack reference "
                "date (reset about every six months). The intervals above are the best case; if OPERA unwrapped "
                "from that reference, the interval is up to the reference span (UNCONFIRMED).")
    return {"wavelength_m": lam, "wavelength_source": lam_src, "quarter_wavelength_mm": _r(q_mm, 2),
            "look_cell_m": look, "support_m": support, "incidence_deg": inc, "incidence_source": inc_src,
            "los_factor_vertical": _r(los, 3), "tilt_limit_mm_per_m": _r(q_mm / cell, 3),
            "intervals": intervals, "intervals_source": int_src, "connections": N, "zones": zones,
            "expected_max_rate_mm_yr": rate or None,
            "tilt_basis": "mean tilt, peak / half-width (the steepest flank of a real trough is about 2x)",
            "intervals_note": note}


def detectability_lines(rec: dict) -> list:
    """Plain lines for the run plan."""
    look = rec["look_cell_m"]
    out = [f"Fast motion (F1): unwrapping follows at most a quarter wavelength, {rec['quarter_wavelength_mm']:.1f} mm "
           f"(wavelength {rec['wavelength_m']:.4f} m, {rec['wavelength_source']}), between look cells "
           f"{look[0]:.0f} x {look[1]:.0f} m apart: {rec['tilt_limit_mm_per_m']:.2f} mm/m of tilt per interval."]
    iv = ", ".join(f"{k} {v:.0f} d" for k, v in rec["intervals"].items())
    out.append(f"Intervals ({rec['intervals_source']}): {iv}.")
    if rec.get("intervals_note"):
        out.append(rec["intervals_note"])
    if not rec["zones"]:
        out.append("No expected rate is set (expected_max_rate_mm_yr), so no site verdict; a bowl aliases where "
                   "its steepest tilt per interval (about twice the peak change / half-width) exceeds the limit.")
        return out
    for z in rec["zones"]:
        if not z.get("intervals"):
            out.append(f"{z['zone']}: {z.get('note', 'not judged')}.")
            continue
        parts = [f"{k} {v['cell_to_cell_mm']:.1f} mm = {v['ratio_to_limit']:.2f} x the limit ({v['verdict']})"
                 for k, v in z["intervals"].items()]
        out.append(f"{z['zone']}: {z['rate_mm_yr']:.0f} mm/yr over a {z['halfwidth_m']:.0f} m half-width "
                   f"({z['halfwidth_from']}), vertical at {rec['incidence_deg']:.0f} deg incidence "
                   f"({rec['incidence_source']}): " + "; ".join(parts)
                   + f". Fastest followable on nearest pairs: about {z['fastest_followable_mm_yr_nearest']:.0f} mm/yr.")
        if z.get("support_change_mm_nearest") is not None:
            out.append(f"  Across the phase-linking window (~{max(rec['support_m']):.0f} m) the ground changes "
                       f"~{z['support_change_mm_nearest']:.0f} mm per nearest interval; above half a wavelength "
                       f"({2 * rec['quarter_wavelength_mm']:.0f} mm) the window mixes phases and loses coherence "
                       "(smaller windows: F3, miaplpy_relink_from).")
    out.append("These use the MEAN tilt (peak / half-width). A real trough's steepest flank is about twice that "
               "(Knothe profile: peak / r, r about half the edge-to-centre distance), so 'measurable' (< 0.5 x) "
               "keeps the steepest flank under the limit, 'marginal' (0.5-1 x) means the steepest flank probably "
               "aliases, and 'too fast' (>= 1 x) means even the mean tilt is over it. Horizontal motion on a flank "
               "adds to this. The run's own check is quality_aliasing_risk.tif (F2).")
    return out


def plan_note(cfg, dates: list | None = None) -> str | None:
    """The run plan's F1 note (stack_backend.plan_stack), or None when the
    config asks for neither the diagnostic nor an expected rate."""
    if not (enabled(cfg) or float(setting(cfg, "expected_max_rate_mm_yr") or 0) > 0):
        return None
    try:
        return "\n    ".join(detectability_lines(detectability(cfg, dates)))
    except Exception as exc:                                  # noqa: BLE001 - a plan never fails on this
        log.warning("Detectability preflight not computed: %s", exc)
        return None


# ======================================================================
# F3: nearest pairs only inside the fast zones
# ======================================================================
def _zone_cells(cfg, gj) -> tuple[np.ndarray, np.ndarray, list]:
    """Full-frame (rows, cols) of the multilooked cells inside the polygons, and the frame."""
    grid = radar_grid(cfg)
    m = polygon_mask(gj, grid["lat"], grid["lon"])
    r0, _, c0, _ = grid["window"]
    rr, cc = np.nonzero(m)
    return (rr + r0).astype(np.int32), (cc + c0).astype(np.int32), grid["frame"]


def _unw_maps(pdir: Path, frame, mode: str):
    Lm, Wm = frame
    unw = np.memmap(pdir / "filt_fine.unw", dtype=np.float32, mode=mode, shape=(Lm, 2, Wm))
    cc = np.memmap(pdir / "filt_fine.unw.conncomp", dtype=np.uint8, mode=mode, shape=(Lm, Wm))
    return unw, cc


def _restore(pdir: Path, frame, z) -> None:
    unw, cc = _unw_maps(pdir, frame, "r+")
    r, c = z["rows"], z["cols"]
    unw[r, 1, c] = z["phase"]
    cc[r, c] = z["cc"]
    unw.flush(); cc.flush()
    del unw, cc


def _fast_one(pdir: Path, frame, rows, cols, want: bool, key: str) -> str:
    """Apply, keep, change or remove one pair's fast-zone no-data. Returns what happened."""
    unw_f, cc_f = pdir / "filt_fine.unw", pdir / "filt_fine.unw.conncomp"
    side = pdir / FAST_SIDECAR
    if not (unw_f.exists() and cc_f.exists()):
        return "not unwrapped"
    Lm, Wm = frame
    if unw_f.stat().st_size != Lm * Wm * 8 or cc_f.stat().st_size != Lm * Wm:
        raise ValueError(f"{pdir.name}: unwrapped files are not the {Lm} x {Wm} frame")
    cur = [_fid(unw_f), _fid(cc_f)]
    if side.exists():
        with np.load(side) as z:
            meta = json.loads(str(z["meta"]))
            status = meta.get("status")
            if status == "applied" and meta.get("after") == cur:
                if want and meta.get("key") == key:
                    return "kept"
                _restore(pdir, frame, z)                     # the setting changed or was removed
                action = "restored"
            elif status == "pending" and meta.get("before") == cur:
                action = "restarted (stopped before writing)"   # the files are still the originals
            elif status == "pending":
                unw, cc = _unw_maps(pdir, frame, "r")
                ours = bool(np.all(unw[z["rows"], 1, z["cols"]] == 0) or np.all(cc[z["rows"], z["cols"]] == 0))
                del unw, cc
                if ours:                                      # stopped part-way through our own write
                    _restore(pdir, frame, z)
                action = "restored" if ours else "dropped (files rewritten since)"
            else:
                action = "dropped (files rewritten since)"   # a re-unwrap replaced them: they are originals
        side.unlink()
        if not want:
            return action
    elif not want:
        return "none"
    unw, cc = _unw_maps(pdir, frame, "r")
    rec = {"rows": rows, "cols": cols, "phase": np.array(unw[rows, 1, cols]), "cc": np.array(cc[rows, cols])}
    del unw, cc
    meta = {"status": "pending", "before": cur, "key": key, "written": time.strftime("%Y-%m-%dT%H:%M:%S")}
    tmp = pdir / "fast_zone_original_tmp.npz"
    np.savez_compressed(tmp, meta=np.array(json.dumps(meta)), **rec)
    os.replace(tmp, side)
    unw, cc = _unw_maps(pdir, frame, "r+")
    unw[rows, 1, cols] = 0.0
    cc[rows, cols] = 0
    unw.flush(); cc.flush()
    del unw, cc
    meta.update(status="applied", after=[_fid(unw_f), _fid(cc_f)])
    np.savez_compressed(tmp, meta=np.array(json.dumps(meta)), **rec)
    os.replace(tmp, side)
    return "applied"


def apply_fast_zones(cfg, pairs: list) -> dict:
    """
    F3 after unwrapping, before MintPy: inside fast_zone_geojson, every pair
    longer than each date's fast_zone_connections nearest becomes no-data
    (unwrapped phase 0 and connected component 0: MintPy reads a zero phase
    as missing and inverts each such cell from the pairs it has). The
    originals go to <pair>/fast_zone_original.npz first and are put back when
    the zone or the count changes, or the setting is removed. Recorded in
    <project>/aliasing/fast_zones.json. Raises on a failure: half-applied
    zones must not reach MintPy.
    """
    gj = str(setting(cfg, "fast_zone_geojson") or "").strip()
    K = int(setting(cfg, "fast_zone_connections"))
    root = _stack(cfg) / "merged" / "interferograms"
    sides = [p for p in pairs if (root / p / FAST_SIDECAR).exists()]
    if not gj and not sides:
        return {"applied": False}
    dates, rank = pair_dates(pairs)
    if gj:
        rows, cols, frame = _zone_cells(cfg, gj)
        key = json.dumps({"zone": _sha(gj), "K": K, "cells": int(rows.size)}, sort_keys=True)
        if not rows.size:
            log.warning("Fast zones (F3): no look cell of the crop lies inside %s - nothing set to no-data", gj)
    else:
        rows = cols = np.zeros(0, np.int32)
        frame, key = radar_grid(cfg)["frame"], ""
    counts: dict = {}
    for p in pairs:
        want = bool(gj) and rank.get(p, 1) > K and rows.size > 0
        what = _fast_one(root / p, frame, rows, cols, want, key)
        counts[what] = counts.get(what, 0) + 1
    rec = {"fast_zone_geojson": gj or None, "fast_zone_connections": K, "zone_cells": int(rows.size),
           "pairs": len(pairs), "outcome": counts, "written": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "why": "inside the zones only each date's nearest pairs are used: the longer pairs alias first",
           "caveat": ("with one connection inside the zones MintPy's temporal coherence there is 1 by "
                      "construction (no loops), so the unwrapping-consistency layer cannot grade those cells; "
                      "one missing nearest pair splits their record")}
    _write_json(record_dir(cfg) / "fast_zones.json", rec)
    if gj:
        log.info("Fast zones (F3): %d cells inside %s; pairs longer than %d connection(s) set to no-data there: %s",
                 rows.size, Path(gj).name, K, ", ".join(f"{v} {k}" for k, v in sorted(counts.items())))
    else:
        log.info("Fast zones removed from the config: originals put back (%s)",
                 ", ".join(f"{v} {k}" for k, v in sorted(counts.items())))
    return rec


def ensure_fast_zones(cfg) -> dict:
    """
    Right before MintPy loads the pairs (stack_backend.mintpy_loader_lines):
    the fast zones as the config says now, whatever restart got here ('MintPy
    onward' runs neither unwrap hook). A cheap no-op when fast_zone_geojson
    is empty and no pair holds originals; otherwise apply_fast_zones on
    every exported pair (idempotent: 'kept' where nothing changed). Raises
    on a failure, as apply_fast_zones does. Then the F2 record is brought
    to the same zones (refresh_for_fast_zones; never raises).
    """
    root = _stack(cfg) / "merged" / "interferograms"
    gj = str(setting(cfg, "fast_zone_geojson") or "").strip()
    if not gj and not (root.is_dir() and any(root.glob(f"*/{FAST_SIDECAR}"))):
        # (a record measured with zones whose originals are gone: still re-measured)
        f2 = refresh_for_fast_zones(cfg) if enabled(cfg) else {}
        return {"applied": False} if not f2 or f2.get("unchanged") else {"applied": False, "f2_record": f2}
    from . import runs
    pairs = [p for p in runs.unwrap_status(_stack(cfg))["exported"] if PAIR_RE.match(p)]
    rec = apply_fast_zones(cfg, pairs)
    rec["f2_record"] = refresh_for_fast_zones(cfg, pairs)
    return rec


def _fast_key_now(cfg) -> tuple:
    """(fast-zone GeoJSON sha, fast_zone_connections) as pre_unwrap keys them."""
    gj = str(setting(cfg, "fast_zone_geojson") or "").strip()
    return (_sha(gj) if gj and Path(gj).exists() else "", int(setting(cfg, "fast_zone_connections")))


def _fast_key_measured(cfg) -> tuple | None:
    """(fast-zone sha, K) the F2 record was measured with (its first readable
    pair), or None without a record."""
    for p in _summary_pairs(cfg) or []:
        try:
            with np.load(_pair_file(cfg, p)) as z:
                s = json.loads(json.loads(str(z["key"]))["settings"])
            return (s.get("fast") or "", int(s.get("K")))
        except (OSError, KeyError, ValueError, TypeError):
            continue
    return None


def _same_fast(a, b) -> bool:
    return a[0] == b[0] and (a[0] == "" or a[1] == b[1])


def refresh_for_fast_zones(cfg, pairs: list | None = None) -> dict:
    """
    F2 sets a pair's level to 0 inside the fast zones where the zones leave
    that pair unused. A record measured under other fast-zone settings than
    the pairs now carry ('MintPy onward' after the zones were added, changed
    or removed) would flag cells by pairs MintPy no longer reads, or miss
    cells of pairs it reads again. So it is measured again here, before
    MintPy loads the pairs; where it cannot be measured the same way (the
    phase-linked route without phase_series.h5) the summary is marked stale
    and the export says so. Nothing when the diagnostic is off or the
    settings match. Never raises.
    """
    if not enabled(cfg):
        return {"skipped": "aliasing_diagnostic is off for this config"}
    try:
        was = _fast_key_measured(cfg)
        now = _fast_key_now(cfg)
        if was is None or _same_fast(was, now):
            return {"unchanged": True}
        route = str(getattr(cfg, "backend", "")).lower()
        if pairs is None:
            from . import runs
            pairs = [p for p in runs.unwrap_status(_stack(cfg))["exported"] if PAIR_RE.match(p)]
        if route == "miaplpy" and not (_project(cfg) / "miaplpy" / "inverted" / "phase_series.h5").exists():
            why = ("the fast zones changed since the aliasing record was measured, and phase_series.h5 is gone, "
                   "so it cannot be measured again the same way: the record describes the earlier zones")
            log.warning("Aliasing diagnostic: %s", why)
            s = record_dir(cfg) / "summary.json"
            try:
                S = json.loads(s.read_text())
                S["fast_zones_stale"] = {"measured_with": list(was), "now": list(now), "why": why}
                _write_json(s, S)
            except (OSError, ValueError):
                pass
            return {"stale": why}
        log.info("Fast zones changed since the aliasing record was measured: measuring the %d pairs again",
                 len(pairs))
        return pre_unwrap(cfg, pairs, route=route)
    except Exception as exc:                                  # noqa: BLE001 - a diagnostic never stops the run
        log.error("Aliasing record not refreshed for the fast zones: %s: %s", type(exc).__name__, exc)
        return {"error": f"{type(exc).__name__}: {exc}"}


# ======================================================================
# command line
# ======================================================================
def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="python -m insar_pipeline.aliasing",
                                 description="Aliasing diagnostic (F2), detectability preflight (F1).")
    ap.add_argument("config")
    ap.add_argument("--export", action="store_true",
                    help="write deliverables/quality_aliasing_risk.tif and aliasing_risk.json from the record")
    ap.add_argument("--out", help="folder for --export (default <project>/deliverables)")
    ap.add_argument("--detectability", action="store_true", help="print the F1 preflight")
    ap.add_argument("--measure", action="store_true",
                    help="measure every exported pair now (writes <project>/aliasing/; reads only the stack)")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    from .config import load_config
    cfg = load_config(a.config)
    rc = 0
    if a.detectability or not (a.export or a.measure):
        dates = None
        try:
            from .stack_backend import planned_dates
            dates = sorted(planned_dates(cfg) or []) or None
        except Exception:                                     # noqa: BLE001
            pass
        print("\n".join(detectability_lines(detectability(cfg, dates))))
    if a.measure:
        from . import runs
        pairs = runs.unwrap_status(_stack(cfg))["exported"]
        r = pre_unwrap(cfg, pairs)
        rc |= 1 if r.get("error") else 0
    if a.export:
        r = export_products(cfg, a.out)
        print(json.dumps({k: v for k, v in r.items() if k != "old_flag"}, indent=1))
        rc |= 1 if (r.get("error") or r.get("skipped")) else 0
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
