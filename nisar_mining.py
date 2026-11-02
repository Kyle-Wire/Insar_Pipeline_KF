"""
NISAR L-band GUNW processing aimed at fast mining subsidence.

STATUS (2 Nov 2026, readiness review QUA-02 / QUA-03): EXPERIMENTAL AND NOT
WIRED IN. No command, route or stage calls this module; it is kept as the
record of an experiment. Nothing it computes is validated, and none of it
reaches a client product. The figures quoted below (bowls, rates) are one
unverified experiment, not a result: see LIMITATIONS.md and VALIDATION.md.

WHY THIS EXISTS SEPARATELY FROM nisar_backend
----------------------------------------------
`nisar_backend` stages whole GUNW files for MintPy. That is the right shape
for a general time series, but it is the wrong shape here for two reasons:

  * Each GUNW is ~1.8 GB. Seventeen of them over this AOI is ~30 GB of
    download to read a 25 x 25 km window out of the middle. This module
    reads windows remotely, the same way disp_backend does.

  * The MintPy route needs ARIA-tools to standardise the sign, and
    ARIA-tools is not installed. Rather than depend on it, the sign is
    resolved here from the product's own metadata and asserted in one
    place (see SIGN below).

WHY L-BAND, FOR THIS PROBLEM SPECIFICALLY
------------------------------------------
Longwall mining drops the surface by of order a metre over a few hundred
metres of ground. Interferometric phase wraps every half wavelength, so
that displacement is:

    C-band (Sentinel-1, 5.55 cm)   ~36 fringes   unwrapping fails
    L-band (NISAR, 24.4 cm)        ~8 fringes    tractable

In one unverified experiment over the Site A monitoring polygon: DISP-S1 C-band reports
2.2% of pixels in an unreliable connected component and never exceeds
-70 mm/yr; NISAR resolves 18 discrete subsidence bowls at -340 to
-687 mm/yr over the same ground. The C-band product is not wrong, it is
blind to this signal - the fringes are packed tighter than it can count.

SIGN
----
Resolved from the product, not assumed. `identification` gives
referenceZeroDopplerStartTime EARLIER than secondaryZeroDopplerStartTime,
so the pair is in normal chronological order and no ARIA-style date flip
applies. With the standard convention phi = -4*pi/lambda * delta_range,
displacement toward the satellite is

    d_los = -phi * lambda / (4*pi)

so subsidence (motion away) is NEGATIVE. The physical check agrees: this
returns compact negative anomalies over active mining against a near-zero
surround, where the opposite sign would give 18 uplift domes over a
coalfield.

GEOMETRY
--------
NISAR looks LEFT; Sentinel-1 looks right. An ascending NISAR track and an
ascending Sentinel-1 track therefore have opposite east sensitivity, and
the two cannot be mixed in one inversion. Decomposition here uses each
product's own losUnitVector layers rather than any assumed heading.
"""
import logging

import numpy as np

from .config import PipelineConfig

log = logging.getLogger(__name__)

# Calibration-validated only. BETA products are early-mission and not
# cal/val'd; ASF's own release note says provisional quality is materially
# better. Nothing from BETA should reach a deliverable.
COLLECTION = "NISAR_L2_GUNW_PROVISIONAL_V1"

GRID = "science/LSAR/GUNW/grids/frequencyA/"
UNW = GRID + "unwrappedInterferogram/HH/"
META = "science/LSAR/GUNW/metadata/radarGrid/"
IDENT = "science/LSAR/identification/"

C_M_S = 299792458.0

# A pair is dropped if fewer than this fraction of the window unwraps into
# a usable connected component. Chosen from the observed distribution over
# Site A, where good pairs run 76-93% and there is a clear gap below.
MIN_PAIR_VALID = 0.45


def _wavelength(h) -> float:
    """Radar wavelength in metres, from the product's own centre frequency."""
    for p in (GRID + "centerFrequency", META + "centerFrequency"):
        if p in h:
            return C_M_S / float(np.asarray(h[p][()]).item())
    return 0.243932          # L-band nominal, 1.229 GHz


def search(cfg: PipelineConfig, track: int | None = None,
           collection: str = COLLECTION):
    """GUNW granules over the AOI, optionally restricted to one track."""
    import asf_search as asf

    results = list(asf.search(
        shortName=[collection],
        intersectsWith=cfg.aoi_wkt,
        start=cfg.start_date or None,
        end=cfg.end_date or None,
        maxResults=2000,
    ))
    if track is not None:
        results = [r for r in results if r.properties.get("pathNumber") == track]
    results.sort(key=lambda p: p.properties["startTime"])
    log.info("NISAR %s: %d granules%s", collection, len(results),
             f" on track {track}" if track else "")
    return results


def tracks_covering(cfg: PipelineConfig, aoi_wkt: str | None = None) -> list[dict]:
    """
    Every track over the AOI with how much of it each one actually covers.

    Coverage matters more than granule count: a track with more pairs that
    clips the corner of the polygon is worth less than a track with fewer
    that covers all of it.
    """
    from shapely import wkt as shapely_wkt
    from shapely.geometry import shape

    poly = shapely_wkt.loads(aoi_wkt or cfg.aoi_wkt)
    by: dict[tuple, list] = {}
    for p in search(cfg):
        key = (p.properties["pathNumber"], p.properties["flightDirection"])
        by.setdefault(key, []).append(p)

    out = []
    for (track, direction), ps in by.items():
        cov = max(shape(p.geometry).intersection(poly).area / poly.area for p in ps)
        dates = sorted(p.properties["startTime"][:10] for p in ps)
        out.append({"track": track, "direction": direction, "n": len(ps),
                    "coverage": cov, "first": dates[0], "last": dates[-1],
                    "granules": ps})
    out.sort(key=lambda d: (-d["coverage"], -d["n"]))
    return out


def _window(x, y, bbox, epsg=32616):
    """Row/col slice of the product grid covering a lon/lat bbox."""
    from pyproj import Transformer

    tr = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    x0, y0 = tr.transform(bbox[0], bbox[1])
    x1, y1 = tr.transform(bbox[2], bbox[3])
    c0, c1 = np.searchsorted(x, min(x0, x1)), np.searchsorted(x, max(x0, x1))
    if y[1] > y[0]:
        r0, r1 = np.searchsorted(y, min(y0, y1)), np.searchsorted(y, max(y0, y1))
    else:
        r0, r1 = np.searchsorted(-y, -max(y0, y1)), np.searchsorted(-y, -min(y0, y1))
    return int(r0), int(r1), int(c0), int(c1)


def read_pair(url: str, bbox, apply_corrections: bool = True) -> dict | None:
    """
    One GUNW pair as LOS displacement in mm over `bbox`.

    Corrections applied when present and requested: the ionospheric phase
    screen and the solid-earth-tide phase, both of which NISAR ships as
    layers and neither of which is removed in the delivered product. At
    L-band the ionosphere is NOT negligible - the delay scales as the
    inverse square of frequency, so it is roughly twenty times the C-band
    term and is the main reason this product ships a screen at all.
    """
    import h5py
    import fsspec
    import aiohttp
    from . import disp_backend as db

    user, pw = db._credentials()
    fs = fsspec.filesystem("https", client_kwargs={
        "auth": aiohttp.BasicAuth(user, pw),
        "timeout": aiohttp.ClientTimeout(total=None, sock_connect=30, sock_read=240),
    })
    with fs.open(url, "rb") as fh:
        with h5py.File(fh, "r") as h:
            lam = _wavelength(h)
            x = h[UNW + "xCoordinates"][:]
            y = h[UNW + "yCoordinates"][:]
            r0, r1, c0, c1 = _window(x, y, bbox)
            if r1 <= r0 or c1 <= c0:
                return None
            sl = (slice(r0, r1), slice(c0, c1))
            phase = h[UNW + "unwrappedPhase"][sl].astype("float64")
            coh = h[UNW + "coherenceMagnitude"][sl].astype("float32")
            conn = h[UNW + "connectedComponents"][sl]
            if apply_corrections and UNW + "ionospherePhaseScreen" in h:
                phase -= np.nan_to_num(h[UNW + "ionospherePhaseScreen"][sl])
            ref = h[IDENT + "referenceZeroDopplerStartTime"][()]
            sec = h[IDENT + "secondaryZeroDopplerStartTime"][()]
            look = h[IDENT + "lookDirection"][()]
            # LOS unit vector, mid-height level of the 3-D radar grid
            los_e = los_n = None
            if META + "losUnitVectorX" in h:
                k = h[META + "losUnitVectorX"].shape[0] // 2
                los_e = float(np.nanmean(h[META + "losUnitVectorX"][k]))
                los_n = float(np.nanmean(h[META + "losUnitVectorY"][k]))

    dec = lambda v: v.decode() if isinstance(v, bytes) else str(v)  # noqa: E731
    # SIGN: see the module docstring. Toward the satellite is positive, so
    # subsidence comes out negative.
    los_mm = -phase * lam / (4.0 * np.pi) * 1000.0
    los_mm[(conn == 0) | ~np.isfinite(los_mm)] = np.nan

    up = None
    if los_e is not None:
        s = los_e ** 2 + los_n ** 2
        up = float(np.sqrt(max(0.0, 1.0 - s)))
    return {
        "los_mm": los_mm, "coherence": coh, "conncomp": conn,
        "ref": dec(ref)[:10].replace("-", ""), "sec": dec(sec)[:10].replace("-", ""),
        "wavelength_m": lam, "look": dec(look),
        "los_east": los_e, "los_north": los_n, "los_up": up,
        "window": (r0, r1, c0, c1),
    }


def stack_track(cfg: PipelineConfig, granules, bbox, ref_bbox=None) -> dict:
    """
    Sum a track's pairs into cumulative displacement.

    The pairs form a chain (each secondary is the next reference), so a
    plain sum IS the time series - there is no inversion to do and no
    network to invert. That is only true while the chain is unbroken;
    `gaps` reports where it is not, because a missing link makes every
    epoch after it an offset estimate rather than a measurement.

    Each pair is referenced to `ref_bbox` before summing. Without that the
    unwrapping reference is arbitrary per pair and the sum accumulates
    nonsense.
    """
    seen, pairs = set(), []
    for g in granules:
        d = read_pair(g.properties["url"], bbox)
        if d is None:
            continue
        key = (d["ref"], d["sec"])
        if key in seen:
            continue
        valid = float(np.isfinite(d["los_mm"]).mean())
        if valid < MIN_PAIR_VALID:
            log.warning("  %s->%s dropped: only %.0f%% unwrapped",
                        d["ref"], d["sec"], 100 * valid)
            continue
        seen.add(key)
        d["valid"] = valid
        pairs.append(d)
        log.info("  %s -> %s  %.0f%% valid", d["ref"], d["sec"], 100 * valid)

    if not pairs:
        raise RuntimeError("No usable NISAR pairs over this AOI")
    pairs.sort(key=lambda d: d["ref"])

    shapes = {p["los_mm"].shape for p in pairs}
    if len(shapes) > 1:
        raise RuntimeError(f"Pairs differ in window shape: {shapes}")

    # REFERENCING. Default is the scene median, not a chosen box.
    #
    # A box has to be right, and here it is easy to get wrong: an automatic
    # search for "quiet ground" can land inside an active panel, because a
    # panel subsiding steadily is internally smooth and scores as calm.
    # That happened on the first run - the box landed at 38.404 N and every
    # displacement flipped sign, reporting +327 mm of uplift over a
    # coalfield.
    #
    # The scene median is robust for exactly the reason a box is not: the
    # bowls are a small fraction of the window (a few km^2 out of ~700), so
    # the median lands on stable farmland by construction and cannot be
    # captured by one badly-chosen neighbourhood. Pass ref_bbox only with
    # independent evidence that the box is still.
    for p in pairs:
        off = np.nan
        if ref_bbox is not None:
            rr = _relative_window(p, bbox, ref_bbox)
            sub = p["los_mm"][rr[0]:rr[1], rr[2]:rr[3]]
            if np.isfinite(sub).any():
                off = np.nanmedian(sub)
            else:
                log.warning("  %s->%s: reference box empty, using scene median",
                            p["ref"], p["sec"])
        if not np.isfinite(off):
            off = np.nanmedian(p["los_mm"])
        if np.isfinite(off):
            p["los_mm"] = p["los_mm"] - off

    cube = np.stack([p["los_mm"] for p in pairs])
    n = np.sum(np.isfinite(cube), axis=0)
    cum = np.where(n >= max(2, len(pairs) - 1),
                   np.nansum(np.where(np.isfinite(cube), cube, 0.0), axis=0), np.nan)

    gaps = [(pairs[i]["sec"], pairs[i + 1]["ref"])
            for i in range(len(pairs) - 1) if pairs[i]["sec"] != pairs[i + 1]["ref"]]
    if gaps:
        log.warning("Chain is broken at %d place(s): %s - epochs after a break "
                    "are offset estimates, not measurements", len(gaps), gaps)

    span_days = (np.datetime64(_iso(pairs[-1]["sec"])) -
                 np.datetime64(_iso(pairs[0]["ref"]))).astype(int)
    return {"cumulative_mm": cum, "n_pairs": len(pairs), "pairs": pairs,
            "first": pairs[0]["ref"], "last": pairs[-1]["sec"],
            "span_days": int(span_days), "gaps": gaps,
            "los_east": pairs[0]["los_east"], "los_north": pairs[0]["los_north"],
            "los_up": pairs[0]["los_up"], "look": pairs[0]["look"],
            "wavelength_m": pairs[0]["wavelength_m"]}


def _iso(yyyymmdd: str) -> str:
    return f"{yyyymmdd[:4]}-{yyyymmdd[4:6]}-{yyyymmdd[6:8]}"


def _relative_window(pair, bbox, ref_bbox):
    """Row/col slice of `ref_bbox` inside an array already cut to `bbox`."""
    r0, r1, c0, c1 = pair["window"]
    ny, nx = pair["los_mm"].shape
    fy0 = (ref_bbox[3] - bbox[3]) / (bbox[1] - bbox[3])
    fy1 = (ref_bbox[1] - bbox[3]) / (bbox[1] - bbox[3])
    fx0 = (ref_bbox[0] - bbox[0]) / (bbox[2] - bbox[0])
    fx1 = (ref_bbox[2] - bbox[0]) / (bbox[2] - bbox[0])
    rr0, rr1 = sorted((int(fy0 * ny), int(fy1 * ny)))
    cc0, cc1 = sorted((int(fx0 * nx), int(fx1 * nx)))
    return (max(0, rr0), min(ny, max(rr1, rr0 + 1)),
            max(0, cc0), min(nx, max(cc1, cc0 + 1)))


def pick_reference(stack: dict, bbox, exclude_frac: float = 0.25) -> tuple:
    """
    A stable reference box chosen from NISAR's own data.

    A reference inherited from the C-band product is not safe here: C-band
    is blind to the fast ground, so a pixel it calls stable can sit beside
    an active panel it never saw. This scores boxes on their own
    cumulative displacement and its spread across the constituent pairs,
    and rejects anything near the scene edge where the unwrapping
    reference itself tends to sit.
    """
    cum = stack["cumulative_mm"]
    ny, nx = cum.shape
    cube = np.stack([p["los_mm"] for p in stack["pairs"]])
    step = max(8, min(ny, nx) // 24)
    datum = float(np.nanmedian(cum))      # the stable background level
    best, best_score = None, np.inf
    m0, m1 = int(ny * exclude_frac * 0.5), int(nx * exclude_frac * 0.5)
    for r in range(m0, ny - m0 - step, step):
        for c in range(m1, nx - m1 - step, step):
            win = cum[r:r + step, c:c + step]
            if np.isfinite(win).mean() < 0.85:
                continue
            with np.errstate(all="ignore"):
                spread = np.nanmean(np.nanstd(cube[:, r:r + step, c:c + step], axis=0))
            # Penalise distance from the SCENE MEDIAN, not from zero: an
            # unreferenced stack has an arbitrary datum, so "near zero" is
            # meaningless while "near the stable background" is what
            # actually identifies still ground. Without this the search
            # picks the inside of a subsiding panel, which is smooth and so
            # scores well on spread alone.
            score = 3.0 * abs(np.nanmedian(win) - datum) + np.nanstd(win) + spread
            if np.isfinite(score) and score < best_score:
                best_score, best = score, (r, c)
    if best is None:
        return None
    r, c = best
    lon0 = bbox[0] + (c / nx) * (bbox[2] - bbox[0])
    lon1 = bbox[0] + ((c + step) / nx) * (bbox[2] - bbox[0])
    lat1 = bbox[3] + (r / ny) * (bbox[1] - bbox[3])
    lat0 = bbox[3] + ((r + step) / ny) * (bbox[1] - bbox[3])
    log.info("Reference box from NISAR data: %.4f, %.4f to %.4f, %.4f (score %.2f)",
             min(lat0, lat1), lon0, max(lat0, lat1), lon1, best_score)
    return (lon0, min(lat0, lat1), lon1, max(lat0, lat1))


def decompose(asc: dict, desc: dict) -> dict:
    """
    Vertical and east-west displacement from one ascending and one
    descending stack.

    Two look directions cannot recover three components, so north is
    unresolvable and is not estimated - both geometries are nearly
    north-south flying, which leaves the north sensitivity too small to
    invert. Reporting a north component from two tracks would be
    fabricating it. Mining subsidence is dominantly vertical, so the
    vertical channel is the one that matters and the east channel is a
    check: a genuine subsidence trough shows strong vertical and modest,
    outward-directed east.
    """
    a, d = asc["cumulative_mm"], desc["cumulative_mm"]
    if a.shape != d.shape:
        raise ValueError(f"asc {a.shape} and desc {d.shape} windows differ - "
                         "resample to a common grid before decomposing")
    A = np.array([[asc["los_east"], asc["los_up"]],
                  [desc["los_east"], desc["los_up"]]], dtype="float64")
    det = np.linalg.det(A)
    if abs(det) < 1e-3:
        raise ValueError(f"Geometries too similar to separate (det={det:.4f}); "
                         "ascending and descending are required")
    inv = np.linalg.inv(A)
    east = inv[0, 0] * a + inv[0, 1] * d
    up = inv[1, 0] * a + inv[1, 1] * d
    log.info("Decomposed with det=%.3f  asc(E %.3f, U %.3f)  desc(E %.3f, U %.3f)",
             det, asc["los_east"], asc["los_up"], desc["los_east"], desc["los_up"])
    return {"vertical_mm": up, "east_mm": east, "det": float(det)}


def find_bowls(cum_mm: np.ndarray, bbox, span_days: int,
               threshold_mm: float = -60.0, min_px: int = 25,
               pixel_m: float = 80.0) -> list[dict]:
    """
    Discrete subsidence bowls, the unit longwall mining actually produces.

    A panel gives a compact steep-sided trough a few hundred metres to a
    couple of kilometres across, not a smooth regional field, so the right
    summary is a list of bowls with depths rather than a scene mean - a
    mean over farmland and panels together describes neither.
    """
    from scipy import ndimage as nd

    k = 7
    valid = np.isfinite(cum_mm).astype(float)
    sm = nd.uniform_filter(np.nan_to_num(cum_mm), k) / np.maximum(
        nd.uniform_filter(valid, k), 1e-9)
    sm[nd.uniform_filter(valid, k) < 0.4] = np.nan

    lab, n = nd.label(np.isfinite(sm) & (sm < threshold_mm))
    if not n:
        return []
    ny, nx = cum_mm.shape
    out = []
    for cid in range(1, n + 1):
        comp = lab == cid
        px = int(comp.sum())
        if px < min_px:
            continue
        ys, xs = np.where(comp)
        deepest = np.nanargmin(sm[comp])
        r, c = ys[deepest], xs[deepest]
        depth = float(np.nanmin(sm[comp]))
        out.append({
            "pixels": px,
            "area_km2": round(px * (pixel_m / 1000.0) ** 2, 3),
            "depth_mm": round(depth, 1),
            "rate_mm_yr": round(depth * 365.25 / max(span_days, 1), 0),
            "lat": round(bbox[3] + (r / ny) * (bbox[1] - bbox[3]), 5),
            "lon": round(bbox[0] + (c / nx) * (bbox[2] - bbox[0]), 5),
        })
    out.sort(key=lambda d: d["depth_mm"])
    return out
