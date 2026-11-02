"""
What one value on the map stands for on the ground: the POSTING (how far
apart the points are) and the RESOLUTION (how much ground each value
combines). One source for every place that states them - the point CSV
header, the technical table, the client report, provenance - instead of the
"posting 15 m; independent resolution cell ~41 m" that was typed into each.

WHY THIS EXISTS (review of 5 Oct 2026)
--------------------------------------
"~41 m" is the 11 x 3 look cell of the classic stack. On the OFFICIAL route
(phase-linked, MiaplPy) each full-resolution pixel's phase history is first
estimated from the statistically similar pixels in a 15 x 15 full-resolution
window (miaplpy_range_window x miaplpy_azimuth_window) and only then
averaged over the 11 x 3 look cell, so a value draws on up to
(window + looks - 1) pixels each way: 25 x 17 pixels, about 90 m across
track by 240 m along it at 3.7 m x 14.1 m. Measured on Site B: median
164 of the 225 window pixels pass as similar (5/25/75/95%: 19/95/212/224),
and the high-passed velocity is still correlated 0.35 about 190 m apart
along track - so "41 m" understated the along-track smoothing about five
times.

Directions: Sentinel-1 flies within ~13 degrees of north-south (heading
-13 ascending, -167 descending at these latitudes), so across track (range)
is called east-west and along track (azimuth) north-south; both are "about".

Pixel sizes come from the product's own attributes when given (MintPy:
RANGE_PIXEL_SIZE / RLOOKS slant range, ground range = slant / sin(incidence);
AZIMUTH_PIXEL_SIZE / ALOOKS at orbit height for ISCE, scaled to the ground
by R / (R + HEIGHT) as mintpy.utils.azimuth_ground_resolution does), else
from the Sentinel-1 IW defaults below.
"""
import logging
import math
from pathlib import Path

log = logging.getLogger(__name__)

# Sentinel-1 IW SLC (config.py looks table): 2.33 m slant range; ~14.1 m
# azimuth on the ground; incidence ~39 deg over Indiana.
S1_SLANT_RANGE_M = 2.329562
S1_AZIMUTH_GROUND_M = 14.07
S1_INCIDENCE_DEG = 39.0
EARTH_RADIUS_M = 6_371_000.0
S1_HEIGHT_M = 700_000.0
# OPERA DISP-S1 (ATBD JPL D-108765 Rev A, Oct 2025, sec. 2.5 and 2.8): 30 m
# posting from 5 m x 10 m CSLCs; distributed scatterers are phase-linked over
# similar pixels found within a search window of ~200 m.
DISP_POSTING_M = 30.0
DISP_SUPPORT_M = 200.0
# NISAR GUNW unwrapped phase posting (nisar_backend).
NISAR_POSTING_M = 80.0
# NISAR L2 GSLC frequency A pixel at 40 MHz (the phase-linked route, nisar_pl)
NISAR_GSLC_PIXEL_M = 5.0
# HyP3 INSAR_GAMMA: 20x4 looks -> 80 m pixels, 10x2 -> 40 m (HyP3 docs).
HYP3_PIXEL_M = {"20x4": 80.0, "10x2": 40.0}


def _f(attrs, key):
    try:
        v = float(attrs[key])
        return v if math.isfinite(v) else None
    except (KeyError, TypeError, ValueError):
        return None


def _posting_from_attrs(attrs) -> float | None:
    if attrs and _f(attrs, "X_STEP") and _f(attrs, "Y_STEP"):
        from .significance import pixel_metres
        try:
            dy, dx = pixel_metres(attrs)
            return float((abs(dy) + abs(dx)) / 2)
        except (KeyError, TypeError, ValueError):
            return None
    return None


def _posting_from_cfg(cfg) -> float | None:
    """mintpy_geocode_lalo_step "-0.000135,0.000172" -> metres at the bbox centre."""
    step = str(getattr(cfg, "mintpy_geocode_lalo_step", "") or "")
    try:
        dlat, dlon = (abs(float(v)) for v in step.split(","))
    except ValueError:
        return None
    bbox = getattr(cfg, "bbox", None)
    lat = (bbox[1] + bbox[3]) / 2 if bbox else 38.5
    return float((dlat * 111_320.0 + dlon * 111_320.0 * math.cos(math.radians(lat))) / 2)


def _full_res_pixel(attrs) -> tuple[float, float, str]:
    """(across-track, along-track) ground size of one FULL-resolution pixel, and where it came from."""
    rng, rl = _f(attrs or {}, "RANGE_PIXEL_SIZE"), _f(attrs or {}, "RLOOKS")
    az, al = _f(attrs or {}, "AZIMUTH_PIXEL_SIZE"), _f(attrs or {}, "ALOOKS")
    inc = _f(attrs or {}, "CENTER_INCIDENCE_ANGLE")
    if rng and rl and az and al and inc and 10 < inc < 60:
        h = _f(attrs, "HEIGHT")
        h = h if h and 3e5 < h < 1.2e6 else S1_HEIGHT_M
        proc = str((attrs or {}).get("PROCESSOR", "isce")).lower()
        az_g = az / al * (EARTH_RADIUS_M / (EARTH_RADIUS_M + h) if proc in ("isce", "roipac") else 1.0)
        return rng / rl / math.sin(math.radians(inc)), az_g, "product attributes"
    return (S1_SLANT_RANGE_M / math.sin(math.radians(S1_INCIDENCE_DEG)), S1_AZIMUTH_GROUND_M,
            "Sentinel-1 IW defaults (product attributes not given)")


def _round_m(x: float) -> int:
    """Whole metres below 50 m (the classic cell stays "41 by 42"), 5 m steps
    below 100 m, 10 m above: the precision the statement deserves."""
    if x < 50:
        return int(round(x))
    return int(5 * round(x / 5)) if x < 100 else int(10 * round(x / 10))


def effective_resolution(cfg, attrs: dict | None = None, measure_shp: bool = False) -> dict:
    """
    {route, route_tag, posting_m, look_cell_m: [ew, ns], support_m: [ew, ns],
    pixel_m, looks, window, source, text}. support is the ground one value
    draws on: the look cell on the classic stack; (window + looks - 1) pixels
    on the phase-linked route; OPERA's search window on DISP-S1. `text` is one
    plain sentence for any reader. attrs: the product's attributes (e.g.
    geo_velocity.h5); without them the config and Sentinel-1 defaults are
    used and `source` says so. measure_shp adds the SHP-count summary of the
    phase-linked run (shp_summary) when phase_series.h5 is still on disk.
    """
    route = str(getattr(cfg, "backend", "") or "").lower()
    try:
        from .config import route_tag
        tag = route_tag(cfg)
    except ImportError:                                   # pragma: no cover
        tag = None
    attrs = dict(attrs or {})
    posting = _posting_from_attrs(attrs)
    src = ["posting from the product grid" if posting else ""]
    out = {"route": route, "route_tag": tag, "window": None, "looks": None, "pixel_m": None}
    if route == "disp":
        posting = posting or DISP_POSTING_M * max(1, int(getattr(cfg, "disp_decimate", 1) or 1))
        look = [posting, posting]
        support = [max(DISP_SUPPORT_M, posting)] * 2
        src.append("OPERA DISP-S1 ATBD (Rev A): 30 m posting, similar pixels searched within ~200 m")
        text = (f"Points are {_round_m(posting)} m apart; each value combines similar ground found within up to "
                f"about {_round_m(support[0])} m (OPERA's processing).")
    elif route == "nisar" and bool(getattr(cfg, "nisar_phase_linked", False)):
        # the phase-linked NISAR route (nisar_pl): the 5 m GSLC pixels phase-
        # linked over a window, output on the project's grid
        w = int(getattr(cfg, "nisar_pl_window", 11) or 11)
        posting = posting or 15.0
        support = [w * NISAR_GSLC_PIXEL_M] * 2
        look = [posting, posting]
        out["window"] = [w, w]
        out["pixel_m"] = [NISAR_GSLC_PIXEL_M, NISAR_GSLC_PIXEL_M]
        src.append(f"NISAR L2 GSLC 5 m pixels, dolphin phase linking over {w} x {w} pixels")
        text = (f"Points are {_round_m(posting)} m apart; each value combines similar ground within about "
                f"{_round_m(support[0])} m (NISAR's 5 m images, phase-linked); a strong reflector such as a "
                "building keeps its own 5 m pixel.")
    elif route == "nisar":
        # the product grid may be finer than the GUNW's own 80 m posting (the
        # crops are put on the Sentinel-1 products' 15 m grid): each value is
        # still the GUNW's ~80 m average, whatever the grid
        posting = posting or NISAR_POSTING_M
        native = max(NISAR_POSTING_M, posting)
        look = support = [native, native]
        src.append("NISAR GUNW posting (80 m)")
        text = (f"Points are {_round_m(posting)} m apart; each value is an average over about {_round_m(native)} m"
                + (" (NISAR's product, put on the same grid as the Sentinel-1 products)." if posting < native else "."))
    elif route == "hyp3":
        px = HYP3_PIXEL_M.get(str(getattr(cfg, "hyp3_looks", "20x4")), 80.0)
        posting = posting or px
        look = support = [px, px]
        src.append(f"HyP3 INSAR_GAMMA {getattr(cfg, 'hyp3_looks', '20x4')} looks")
        text = (f"Points are {_round_m(posting)} m apart; each value is an average over about "
                f"{_round_m(px)} m by {_round_m(px)} m.")
    else:
        pr, pa, psrc = _full_res_pixel(attrs)
        rl = int(_f(attrs, "RLOOKS") or getattr(cfg, "stack_range_looks", 11) or 11)
        al = int(_f(attrs, "ALOOKS") or getattr(cfg, "stack_azimuth_looks", 3) or 3)
        posting = posting or _posting_from_cfg(cfg) or 15.0
        look = [rl * pr, al * pa]
        out.update({"looks": [rl, al], "pixel_m": [round(pr, 2), round(pa, 2)]})
        src.append(f"pixel size from {psrc}")
        if route == "miaplpy":
            wr = int(getattr(cfg, "miaplpy_range_window", 15) or 15)
            wa = int(getattr(cfg, "miaplpy_azimuth_window", 15) or 15)
            support = [(wr + rl - 1) * pr, (wa + al - 1) * pa]
            out["window"] = [wr, wa]
            text = (f"Points are {_round_m(posting)} m apart; each value combines similar ground within about "
                    f"{_round_m(support[0])} m east-west and {_round_m(support[1])} m north-south.")
        else:
            support = list(look)
            text = (f"Points are {_round_m(posting)} m apart; each value is an average over about "
                    f"{_round_m(support[0])} m east-west by {_round_m(support[1])} m north-south.")
    out.update({"posting_m": round(float(posting), 1),
                "look_cell_m": [round(float(v), 1) for v in look],
                "support_m": [round(float(v), 1) for v in support],
                "source": "; ".join(s for s in src if s), "text": text})
    if measure_shp and route == "miaplpy":
        out["shp"] = shp_summary(cfg)
    return out


def shp_summary(cfg, stride: int = 4) -> dict | None:
    """Similar-pixel counts of the phase-linked run (miaplpy/inverted/
    phase_series.h5 'shp', every `stride`-th row: 24 MB of the 96 MB layer):
    median and 5/25/75/95 percentiles, against the window's pixel count.
    None when the file is gone (MiaplPy cleanup) or unreadable."""
    p = Path(cfg.project_dir) / "miaplpy" / "inverted" / "phase_series.h5"
    if not p.exists():
        return None
    try:
        import h5py
        import numpy as np
        with h5py.File(p, "r") as f:
            if "shp" not in f:
                return None
            x = f["shp"][::max(1, int(stride))].ravel()
        x = x[x > 0]
        if not x.size:
            return None
        q = np.percentile(x, [5, 25, 50, 75, 95])
        w = int(getattr(cfg, "miaplpy_range_window", 15)) * int(getattr(cfg, "miaplpy_azimuth_window", 15))
        return {"median": int(q[2]), "p5_p25_p75_p95": [int(v) for v in q[[0, 1, 3, 4]]],
                "window_pixels": w, "rows_sampled_every": int(stride)}
    except Exception as exc:                              # noqa: BLE001 - informational
        log.warning("SHP summary not readable: %s", exc)
        return None


def extent_m(mask, attrs: dict) -> tuple[float, float]:
    """(east-west, north-south) extent in metres of a mask's bounding box on a geocoded grid."""
    import numpy as np
    from .significance import pixel_metres
    rows, cols = np.nonzero(mask)
    if not rows.size:
        return 0.0, 0.0
    dy, dx = pixel_metres(attrs)
    return float((cols.max() - cols.min() + 1) * dx), float((rows.max() - rows.min() + 1) * dy)


def area_caveat(res: dict, ew_m: float, ns_m: float, factor: float = 2.0) -> str | None:
    """A plain warning when an area is less than `factor` x the ground each
    value combines, in either direction (its figure then blends in the
    ground around it), else None."""
    sew, sns = (res.get("support_m") or [0, 0])[:2]
    short, ref = [], []
    if sew and ew_m < factor * sew:
        short.append(f"{_round_m(ew_m)} m east-west"); ref.append(f"{_round_m(sew)} m east-west")
    if sns and ns_m < factor * sns:
        short.append(f"{_round_m(ns_m)} m north-south"); ref.append(f"{_round_m(sns)} m north-south")
    if not short:
        return None
    return (f"This area spans only about {' and '.join(short)}, less than twice the ground each value combines "
            f"(about {' and '.join(ref)}), so its figure also takes in some of the ground around it.")
