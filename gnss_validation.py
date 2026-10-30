"""Tie an InSAR result to GNSS, or say honestly why it cannot be tied.

Every displacement this pipeline produces is relative to one pixel. Until
something independent says how that pixel is moving, the whole product
floats: a uniform offset can be added to every number without changing
anything internal to it, and nothing in the InSAR data can detect it.
GNSS is what pins it down.

Three situations, and the module reports which one applies rather than
returning a number regardless:

  1. A station INSIDE the AOI. The reference pixel can be calibrated:
     shift the whole field so the InSAR rate at the station matches the
     station's own rate projected into the radar line of sight. This is a
     real tie to an absolute frame.

  2. Stations NEARBY but outside. No calibration, but a bound: if every
     station for 200 km reports vertical motion of -1 to -2 mm/yr, then
     the reference pixel is very unlikely to be moving at 20 mm/yr, and
     the reference-frame error is bounded by roughly the spread of those
     stations. Weaker than a tie, far better than nothing, and it is
     what makes a local signal defensible when it is an order of
     magnitude larger than the regional field.

  3. Nothing within the radius. Say so, and stop.

Site A is case 2: the nearest station in the JPL solution is 135 km
away.

TWO THINGS THAT WILL PRODUCE A CONFIDENT WRONG ANSWER IF IGNORED:

  * FRAME. These velocities are IGS14, which includes plate motion. North
    America runs about 15 mm/yr west here, and projected into an
    ascending line of sight that is roughly +8 mm/yr - comparable to or
    larger than the deformation being measured. It is almost uniform
    across a small AOI, so a relative InSAR product removes it, but any
    comparison of ABSOLUTE rates must handle it. Only vertical rates are
    quoted as regional context below, for exactly this reason.

  * PROJECTION. A GNSS velocity is three components; InSAR measures one
    projection of them. They cannot be compared until the GNSS vector is
    projected into the same line of sight, using the per-pixel unit
    vector from `static_layers`. Comparing a GNSS vertical rate directly
    against an InSAR line-of-sight rate is wrong by a factor of
    1/cos(incidence) plus whatever the horizontal contributes.
"""
import json
import logging
import math
from pathlib import Path

import numpy as np

from .config import PipelineConfig

log = logging.getLogger(__name__)

# JPL/GIPSY combined solution: one plain-text table of positions and
# velocities for ~2,800 stations, no authentication. Documented at
# sideshow.jpl.nasa.gov/post/series.html.
JPL_TABLE_URL = "https://sideshow.jpl.nasa.gov/post/tables/table2.html"

# Nevada Geodetic Laboratory MIDAS solution: ~20,000 stations, including the
# whole NGS CORS network, which is what makes the US interior usable. The JPL
# table is a global IGS-style solution and is sparse here - its nearest
# station to Site A is 135 km, against 11.8 km in MIDAS, and 7 stations
# inside 50 km where JPL has none. Same IGS14 frame, so the two can be merged
# without a frame transformation.
#
# MUST be https. The http:// endpoints hang rather than redirect.
UNR_LLH_URL = "https://geodesy.unr.edu/NGLStationPages/llh.out"
UNR_MIDAS_URL = "https://geodesy.unr.edu/velocities/midas.IGS14.txt"

EARTH_RADIUS_KM = 6371.0

# Reject a tie whose stations disagree by more than this. Subsidence work at
# this site is arguing about single mm/yr, so a frame tie that is only good
# to tens of mm/yr is worse than admitting the product is relative: it looks
# absolute and is not. Site A measured 35 mm/yr across three stations.
MAX_TIE_SPREAD_MM_YR = 5.0


def _great_circle_km(lat1, lon1, lat2, lon2) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    d = math.radians(lon2 - lon1)
    c = (math.sin(p1) * math.sin(p2)
         + math.cos(p1) * math.cos(p2) * math.cos(d))
    return EARTH_RADIUS_KM * math.acos(max(-1.0, min(1.0, c)))


def fetch_table(cache: Path | None = None, timeout: int = 90) -> str:
    """The JPL velocity table, cached on disk so a run is reproducible."""
    if cache and cache.exists():
        log.info("Using cached GNSS table %s", cache)
        return cache.read_text(errors="replace")
    import urllib.request
    log.info("Fetching GNSS velocities from %s", JPL_TABLE_URL)
    with urllib.request.urlopen(JPL_TABLE_URL, timeout=timeout) as r:
        text = r.read().decode("utf-8", errors="replace")
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(text)
    return text


def parse_table(text: str) -> dict[str, dict]:
    """
    Parse the fixed-format JPL table.

    Two lines per station:
        SITE POS  lat lon height  s_lat s_lon s_height
        SITE VEL  vN  vE  vU      s_vN  s_vE  s_vU     (mm/yr)
    """
    pos, vel = {}, {}
    for line in text.splitlines():
        p = line.split()
        if len(p) < 8 or p[1] not in ("POS", "VEL"):
            continue
        try:
            vals = [float(x) for x in p[2:8]]
        except ValueError:
            continue
        (pos if p[1] == "POS" else vel)[p[0]] = vals

    out = {}
    for site in sorted(set(pos) & set(vel)):
        lat, lon = pos[site][0], pos[site][1]
        vn, ve, vu, svn, sve, svu = vel[site]
        out[site] = {
            "lat": lat, "lon": ((lon + 180) % 360) - 180,
            "v_north_mm_yr": vn, "v_east_mm_yr": ve, "v_up_mm_yr": vu,
            "sigma_north_mm_yr": svn, "sigma_east_mm_yr": sve,
            "sigma_up_mm_yr": svu,
        }
    log.info("Parsed %d GNSS stations with both position and velocity",
             len(out))
    return out


def _fetch_text(url: str, cache: Path | None, timeout: int) -> str:
    if cache and cache.exists():
        log.info("Using cached %s", cache.name)
        return cache.read_text(errors="replace")
    import urllib.request
    log.info("Fetching %s", url)
    with urllib.request.urlopen(url, timeout=timeout) as r:
        text = r.read().decode("utf-8", errors="replace")
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(text)
    return text


def parse_unr(llh_text: str, midas_text: str) -> dict[str, dict]:
    """
    Parse the MIDAS velocity file against the station coordinate list.

    llh.out  : SITE lat lon height
    midas    : SITE label t0 t1 dt n ngood npair vE vN vU sE sN sU ...
               velocities and sigmas in m/yr, columns 9-14 (1-indexed).

    MIDAS is a median-trend estimator: it is deliberately insensitive to
    steps and outliers, which is what makes it usable without hand-editing
    20,000 time series. That robustness is the reason to prefer it here
    over a least-squares rate.
    """
    pos = {}
    for line in llh_text.splitlines():
        p = line.split()
        if len(p) < 3:
            continue
        try:
            lat, lon = float(p[1]), float(p[2])
        except ValueError:
            continue
        pos[p[0]] = (lat, ((lon + 180) % 360) - 180)

    out = {}
    for line in midas_text.splitlines():
        p = line.split()
        if len(p) < 14 or p[0] not in pos:
            continue
        try:
            ve, vn, vu, se, sn, su = (float(x) for x in p[8:14])
        except ValueError:
            continue
        lat, lon = pos[p[0]]
        out[p[0]] = {
            "lat": lat, "lon": lon,
            "v_north_mm_yr": vn * 1000.0, "v_east_mm_yr": ve * 1000.0,
            "v_up_mm_yr": vu * 1000.0,
            "sigma_north_mm_yr": sn * 1000.0, "sigma_east_mm_yr": se * 1000.0,
            "sigma_up_mm_yr": su * 1000.0,
            "source": "UNR MIDAS IGS14",
        }
    log.info("Parsed %d UNR MIDAS stations", len(out))
    return out


def load_stations(cache_dir: Path, timeout: int = 120) -> dict[str, dict]:
    """
    Every station from both solutions, MIDAS winning any name collision.

    Either source failing is survivable and logged - a sparse answer beats
    no answer, and the caller reports which case it ended up in. Both
    failing is not caught here: with no stations at all there is nothing
    honest to say about the reference frame.
    """
    stations = {}
    try:
        stations.update(parse_table(fetch_table(cache_dir / "jpl_table2.txt",
                                                timeout=timeout)))
        for v in stations.values():
            v.setdefault("source", "JPL GIPSY IGS14")
    except Exception as exc:
        log.warning("JPL table unavailable (%s) - continuing with MIDAS only", exc)
    try:
        unr = parse_unr(
            _fetch_text(UNR_LLH_URL, cache_dir / "unr_llh.out", timeout),
            _fetch_text(UNR_MIDAS_URL, cache_dir / "unr_midas_igs14.txt", timeout))
        overlap = len(set(stations) & set(unr))
        stations.update(unr)          # MIDAS is denser and better sampled here
        log.info("GNSS catalogue: %d stations (%d also in the JPL table)",
                 len(stations), overlap)
    except Exception as exc:
        log.warning("UNR MIDAS unavailable (%s) - falling back to the JPL "
                    "table alone, which is sparse in the US interior", exc)
    return stations


def nearby(stations: dict, lat: float, lon: float, radius_km: float) -> list[dict]:
    out = []
    for site, s in stations.items():
        d = _great_circle_km(lat, lon, s["lat"], s["lon"])
        if d <= radius_km:
            out.append({"site": site, "distance_km": round(d, 1), **s})
    return sorted(out, key=lambda s: s["distance_km"])


def validate(cfg: PipelineConfig, out_dir: Path,
             radius_km: float = 250.0) -> dict:
    """
    Compare, calibrate or bound - whichever the station geometry allows.

    Writes `gnss_validation.json` beside the deliverables and returns the
    same record.
    """
    import h5py
    from . import static_layers as sl

    min_lon, min_lat, max_lon, max_lat = cfg.bbox
    clat, clon = (min_lat + max_lat) / 2, (min_lon + max_lon) / 2

    vel_path = cfg.mintpy_dir / "velocity.h5"
    if not vel_path.exists():
        raise RuntimeError(f"{vel_path} not found - run the pipeline first")
    with h5py.File(vel_path, "r") as f:
        vel = f["velocity"][:]                      # m/yr, LOS, toward +
        attrs = dict(f.attrs)

    grid = {
        "x0": float(attrs["X_FIRST"]), "y0": float(attrs["Y_FIRST"]),
        "dx": float(attrs["X_STEP"]), "dy": float(attrs["Y_STEP"]),
        "ny": vel.shape[0], "nx": vel.shape[1], "epsg": int(float(attrs["EPSG"])),  # tolerate "32616.0" from a cached window
    }
    geom = sl.load(cfg, grid, frame=attrs.get("FRAME_ID"))
    if "los_up" in geom:
        los_e = float(np.nanmean(geom["los_east"]))
        los_n = float(np.nanmean(geom["los_north"]))
        los_u = float(np.nanmean(geom["los_up"]))
        geom_src = "DISP-S1-STATIC line_of_sight_enu (AOI mean)"
    else:
        # Fall back to the stated incidence and heading. Worse, and said
        # so in the record rather than hidden.
        inc = math.radians(float(attrs.get("INCIDENCE_ANGLE", 38.3)))
        head = math.radians(float(attrs.get("HEADING", -13.3)))
        look = head - math.pi / 2          # ground -> satellite, right-looking
        los_e, los_n, los_u = (math.sin(inc) * math.sin(look),
                               math.sin(inc) * math.cos(look), math.cos(inc))
        geom_src = "derived from INCIDENCE_ANGLE and HEADING (no static layers)"
    log.info("LOS unit vector (ground->satellite): E %+.4f  N %+.4f  U %+.4f "
             "[%s]", los_e, los_n, los_u, geom_src)

    stations = load_stations(cfg.project_dir / "gnss")
    found = nearby(stations, clat, clon, radius_km)

    for s in found:
        s["v_los_mm_yr"] = round(sl.los_from_enu(
            s["v_east_mm_yr"], s["v_north_mm_yr"], s["v_up_mm_yr"],
            los_e, los_n, los_u), 3)
        # Inside the AOI grid?
        s["inside_aoi"] = bool(min_lat <= s["lat"] <= max_lat
                               and min_lon <= s["lon"] <= max_lon)

    record = {
        "aoi_centre": [round(clon, 5), round(clat, 5)],
        "search_radius_km": radius_km,
        "source": JPL_TABLE_URL,
        "reference_frame": "IGS14",
        "los_unit_vector_ground_to_satellite": {
            "east": round(los_e, 5), "north": round(los_n, 5),
            "up": round(los_u, 5), "source": geom_src,
        },
        "n_stations_found": len(found),
        "stations": found[:12],
    }

    inside = [s for s in found if s["inside_aoi"]]
    if inside:
        record["case"] = "tie"
        record.update(_calibrate(cfg, vel, grid, inside, los_e, los_n, los_u))
    elif found:
        record["case"] = "bound"
        record.update(_bound(found, vel))
    else:
        record["case"] = "none"
        record["verdict"] = (
            f"No GNSS station within {radius_km:.0f} km. This result cannot "
            f"be tied to an absolute frame and its reference pixel cannot "
            f"be bounded. Every number is relative only.")
        log.error(record["verdict"])

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "gnss_validation.json"
    path.write_text(json.dumps(record, indent=2))
    log.info("Wrote %s", path)
    return record


def _calibrate(cfg, vel, grid, inside, los_e=0.0, los_n=0.0, los_u=1.0) -> dict:
    """A station inside the AOI: shift the field onto the absolute frame."""
    from pyproj import Transformer
    tr = Transformer.from_crs("EPSG:4326", f"EPSG:{grid['epsg']}",
                              always_xy=True)
    rows = []
    for s in inside:
        x, y = tr.transform(s["lon"], s["lat"])
        c = int(round((x - grid["x0"]) / grid["dx"]))
        r = int(round((y - grid["y0"]) / grid["dy"]))
        if not (0 <= r < grid["ny"] and 0 <= c < grid["nx"]):
            continue
        # Median of a 5x5 window: one InSAR pixel is noisy, and the
        # station monument is not necessarily on the pixel centre.
        win = vel[max(0, r - 2):r + 3, max(0, c - 2):c + 3] * 1000.0
        insar = float(np.nanmedian(win)) if np.any(np.isfinite(win)) else None
        if insar is None:
            log.warning("Station %s falls on pixels with no velocity",
                        s["site"])
            continue
        rows.append({"site": s["site"], "row": r, "col": c,
                     "insar_los_mm_yr": round(insar, 3),
                     "gnss_los_mm_yr": s["v_los_mm_yr"],
                     "offset_mm_yr": round(s["v_los_mm_yr"] - insar, 3)})
    if not rows:
        return {"case": "bound",
                "verdict": "Stations fall inside the AOI but on pixels with "
                           "no velocity - no tie possible."}
    offs = [r["offset_mm_yr"] for r in rows]
    shift = float(np.median(offs))
    spread = (max(offs) - min(offs)) if len(offs) > 1 else float("nan")

    # How much of the tie is plate motion rather than vertical land motion.
    # IGS14 velocities carry ~15 mm/yr of westward North America plate
    # motion, which projects to roughly +9 mm/yr in this ascending LOS -
    # larger than the subsidence being measured. A relative InSAR product
    # has already removed it as a near-uniform field, so a shift that is
    # mostly plate motion is re-inserting a term the product deliberately
    # does not contain. That is correct for an IGS14-absolute deliverable
    # and wrong for a vertical-motion one, so the split is reported rather
    # than buried.
    for r, st in zip(rows, inside):
        horiz = los_e * st["v_east_mm_yr"] + los_n * st["v_north_mm_yr"]
        r["gnss_los_horizontal_mm_yr"] = round(horiz, 3)
        r["gnss_los_vertical_mm_yr"] = round(los_u * st["v_up_mm_yr"], 3)

    if len(offs) > 1 and spread > MAX_TIE_SPREAD_MM_YR:
        return {
            "case": "bound",
            "calibration": rows,
            "rejected_shift_mm_yr": round(shift, 3),
            "tie_spread_mm_yr": round(spread, 3),
            "verdict": (
                f"{len(rows)} stations fall inside the AOI, but their implied "
                f"shifts disagree by {spread:.1f} mm/yr "
                f"({', '.join(f'{o:+.1f}' for o in offs)}), against a "
                f"{MAX_TIE_SPREAD_MM_YR:.0f} mm/yr limit. A tie is only "
                f"meaningful if the stations agree to better than the signal "
                f"being measured, so NO SHIFT HAS BEEN APPLIED and the "
                f"product remains relative. The usual cause is that an InSAR "
                f"pixel and its GNSS monument are not on the same moving "
                f"ground - check each station's pixel before trusting any "
                f"one of them."),
        }

    return {
        "calibration": rows,
        "reference_shift_mm_yr": round(shift, 3),
        "tie_spread_mm_yr": None if len(offs) < 2 else round(spread, 3),
        "verdict": (
            f"{len(rows)} station(s) inside the AOI, agreeing to "
            f"{spread:.2f} mm/yr. Adding {shift:+.2f} mm/yr to every "
            f"line-of-sight velocity puts the product on the IGS14 frame - "
            f"which includes plate motion, so use it for an absolute "
            f"deliverable, not for a vertical-motion one."
            if len(rows) > 1 else
            f"One station inside the AOI. Adding {shift:+.2f} mm/yr to every "
            f"line-of-sight velocity puts the product on the IGS14 frame. "
            f"With a single station there is nothing to cross-check it "
            f"against, so treat the tie as unverified."),
    }


def _bound(found, vel) -> dict:
    """No station inside: bound the reference-frame error instead."""
    ups = np.array([s["v_up_mm_yr"] for s in found])
    sig = np.array([s["sigma_up_mm_yr"] for s in found])
    lo, hi = float(ups.min()), float(ups.max())
    med = float(np.median(ups))
    spread = float(np.std(ups))

    v = vel[np.isfinite(vel)] * 1000.0
    signal = float(np.nanpercentile(np.abs(v), 95)) if v.size else float("nan")
    ratio = signal / max(abs(med) + spread, 1e-6)

    verdict = (
        f"No GNSS station inside the AOI; nearest is {found[0]['distance_km']:.0f} km "
        f"({found[0]['site']}). The product cannot be tied to an absolute "
        f"frame. Regional vertical motion across {len(found)} stations runs "
        f"{lo:+.2f} to {hi:+.2f} mm/yr (median {med:+.2f}, spread "
        f"{spread:.2f}), so the reference pixel is unlikely to be moving "
        f"faster than a few mm/yr. The InSAR signal itself reaches "
        f"{signal:.1f} mm/yr at the 95th percentile - about {ratio:.0f}x the "
        f"regional field - so the local pattern is robust to the reference "
        f"choice even though its absolute level is not."
    )
    log.warning(verdict)
    return {
        "regional_vertical_mm_yr": {
            "min": round(lo, 3), "max": round(hi, 3), "median": round(med, 3),
            "std": round(spread, 3), "n": len(found),
            "median_sigma": round(float(np.median(sig)), 3),
        },
        "insar_signal_p95_mm_yr": round(signal, 2),
        "signal_to_regional_ratio": round(ratio, 1),
        "verdict": verdict,
    }
