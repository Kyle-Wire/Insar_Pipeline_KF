"""
Backend-independent quality measurements for an InSAR time series.

WHY THIS MODULE EXISTS
----------------------
A displacement product's noise must not grow with distance from its own
reference point. Epoch-to-epoch step RMS measured as a function of that
distance is the check: a datum that propagates gives a flat curve, one
that does not gives a curve that climbs. It is also where reference pixel
candidates come from (select_reference_candidates): high temporal coherence,
far from every mapped panel, in agreement with the surrounding stable
ground, chosen from the product itself.

This module reads any MintPy-schema time series - topsStack, HyP3, NISAR -
and knows nothing about which backend produced it.

WHAT THE METRIC MEANS
---------------------
Ground does not move and then unmove. Real deformation is smooth in time:
consecutive 12-day epochs differ by the true rate times 12 days, which for
even fast longwall subsidence is a few millimetres. A large step that
reverses at the next epoch is therefore not ground - it is an unwrapping
error, an atmospheric screen, or a referencing artefact.

Measuring those steps AS A FUNCTION OF DISTANCE separates the causes:

  flat with distance        -> a scene-wide problem (orbit, ionosphere)
  rising with distance      -> the datum is not propagating; unwrapping
                               or long-wavelength atmosphere
  rising only in some bins  -> localised decorrelation

WHAT IT IS NOT
--------------
Not a precision figure for the velocity. A step RMS of 20 mm does not mean
the rate is wrong by 20 mm/yr - averaging hundreds of epochs suppresses
uncorrelated noise. It bounds the SINGLE-EPOCH reading, and it diagnoses
whether the errors are random (they average down) or systematic (they do
not). Reversing jumps of one phase cycle are the systematic kind.
"""
import logging
import re
from datetime import datetime
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

# C-band Sentinel-1 half wavelength: the LOS displacement equivalent of one
# 2-pi phase cycle. An unwrapping error moves a pixel by a whole multiple of
# this, which is what makes it distinguishable from atmosphere.
C_BAND_HALF_WAVELENGTH_MM = 27.73

# A step this large in 12 days is 840 mm/yr sustained. Real ground somewhere
# does that, but not Indiana farmland, so steps above it are the population
# we are trying to characterise rather than outliers to discard.
# (Sentinel-1. Another sensor scales it with its own cycle:
# default_step_threshold_mm.)
DEFAULT_STEP_THRESHOLD_MM = 20.0

# ---------------------------------------------------------------- sensor
# Wavelength, look side and revisit come from the product, not from
# Sentinel-1 constants (M1, review_2026-10-08/PLAN.md phase 3, 8 Oct 2026).
# MintPy writes WAVELENGTH (metres) and PLATFORM into every product it
# loads; prep_nisar takes WAVELENGTH from the GUNW's own centre frequency.
# A Sentinel-1 product keeps the exact constants used so far: its WAVELENGTH
# attribute (0.05546576 m) is read as SENTINEL1_WAVELENGTH_M and its cycle as
# C_BAND_HALF_WAVELENGTH_MM, so no Indiana number moves by a rounding.
SENTINEL1_WAVELENGTH_M = 0.05546576          # C band, 5.405 GHz
# Nominal NISAR L-SAR wavelength (centre 1.229 GHz, as nisar_mining._wavelength
# falls back to). Used ONLY to recognise the sensor when a product carries
# PLATFORM but no WAVELENGTH; every NISAR GUNW states its centre frequency
# (1.2215-1.2575 GHz depending on mode, i.e. 0.238-0.245 m).
NISAR_L_WAVELENGTH_M = 0.243932
# 12 days: one Sentinel-1 satellite's repeat (the record's spacing on a
# single-satellite track) and NISAR's repeat on each pass direction. Used only
# when a product has fewer than two dates; otherwise the dates decide.
NOMINAL_REVISIT_DAYS = {"sentinel-1": 12.0, "nisar": 12.0}
# Look side by platform: Sentinel-1 looks right; NISAR looks left (its
# identification/lookDirection says "left"). MintPy's own ANTENNA_SIDE
# attribute (-1 right, +1 left) wins when present.
LOOK_SIDE = {"sentinel-1": "right", "nisar": "left"}
# A WAVELENGTH within this fraction of Sentinel-1's is Sentinel-1's.
_S1_MATCH = 0.005


def _platform(attrs) -> str:
    """'sentinel-1', 'nisar' or '' from MintPy's PLATFORM / MISSION attribute."""
    p = str((attrs or {}).get("PLATFORM") or (attrs or {}).get("MISSION") or "").lower()
    if p.startswith(("sen", "s1")):
        return "sentinel-1"
    if "nisar" in p or p.startswith("nis"):
        return "nisar"
    return ""


def product_wavelength_m(attrs, default=SENTINEL1_WAVELENGTH_M) -> float:
    """Radar wavelength in metres from the product's own WAVELENGTH attribute.

    Order: WAVELENGTH when it is a plausible radar wavelength (1-50 cm);
    else the platform's nominal value (PLATFORM); else `default` (Sentinel-1,
    which is what every product without the attribute has been so far). A
    value within 0.5% of Sentinel-1's is returned as SENTINEL1_WAVELENGTH_M
    exactly, so a Sentinel-1 record's numbers are unchanged to the last digit.
    """
    v = (attrs or {}).get("WAVELENGTH")
    if isinstance(v, bytes):
        v = v.decode()
    try:
        wl = float(v)
    except (TypeError, ValueError):
        wl = float("nan")
    if not (np.isfinite(wl) and 0.01 <= wl <= 0.5):
        plat = _platform(attrs)
        wl = (NISAR_L_WAVELENGTH_M if plat == "nisar" else
              SENTINEL1_WAVELENGTH_M if plat == "sentinel-1" else float(default))
    if abs(wl - SENTINEL1_WAVELENGTH_M) <= _S1_MATCH * SENTINEL1_WAVELENGTH_M:
        return SENTINEL1_WAVELENGTH_M
    return wl


def is_sentinel1_wavelength(wavelength_m) -> bool:
    return abs(float(wavelength_m) - SENTINEL1_WAVELENGTH_M) <= _S1_MATCH * SENTINEL1_WAVELENGTH_M


def sensor_band(wavelength_m) -> str:
    """'X', 'C', 'S' or 'L' for a wavelength in metres (IEEE letter bands)."""
    wl = float(wavelength_m)
    if wl < 0.0375:
        return "X"
    if wl < 0.075:
        return "C"
    if wl < 0.15:
        return "S"
    return "L"


def half_wavelength_mm(attrs) -> float:
    """LOS displacement of one 2-pi phase cycle, in mm, for this product:
    27.73 for Sentinel-1 (the constant used since Sep 2026), WAVELENGTH/2 for
    anything else (about 122 mm for NISAR L-band)."""
    wl = product_wavelength_m(attrs)
    if is_sentinel1_wavelength(wl):
        return C_BAND_HALF_WAVELENGTH_MM
    return 1000.0 * wl / 2.0


def default_step_threshold_mm(attrs) -> float:
    """The 'big step' threshold of step_rms_vs_distance for this product: 20 mm
    for Sentinel-1 (unchanged), the same fraction of a cycle (0.72) for any
    other sensor - 88 mm at L-band, where a 20 mm step is a sixth of a cycle
    and the whole-cycle test would be counting atmosphere."""
    half = half_wavelength_mm(attrs)
    if half == C_BAND_HALF_WAVELENGTH_MM:
        return DEFAULT_STEP_THRESHOLD_MM
    return round(DEFAULT_STEP_THRESHOLD_MM * half / C_BAND_HALF_WAVELENGTH_MM, 2)


def product_look_side(attrs) -> str | None:
    """'right' or 'left': MintPy's ANTENNA_SIDE (-1 right, +1 left) when the
    product states it, else LOOK_DIRECTION, else by PLATFORM; None when
    nothing says (the caller must not assume Sentinel-1's right look)."""
    a = attrs or {}
    side = a.get("ANTENNA_SIDE")
    try:
        s = int(float(side.decode() if isinstance(side, bytes) else side))
        if s in (-1, 1):
            return "right" if s == -1 else "left"
    except (TypeError, ValueError):
        pass
    look = a.get("LOOK_DIRECTION") or a.get("lookDirection")
    if look is not None:
        look = (look.decode() if isinstance(look, bytes) else str(look)).strip().lower()
        if look in ("right", "left"):
            return look
    return LOOK_SIDE.get(_platform(a))


def revisit_days(dates, attrs=None) -> float:
    """Typical interval between the product's dates (median, days); with
    fewer than two dates, the platform's nominal repeat (12 days)."""
    days = []
    for d in dates or []:
        s = d.decode() if isinstance(d, bytes) else str(d)
        try:
            days.append(datetime.strptime(s[:8], "%Y%m%d").toordinal())
        except ValueError:
            continue
    gaps = np.diff(sorted(days))
    if gaps.size:
        return float(np.median(gaps))
    return float(NOMINAL_REVISIT_DAYS.get(_platform(attrs), 12.0))


def _open_attrs(h5file):
    import h5py
    with h5py.File(h5file, "r") as f:
        return {k: (v.decode() if isinstance(v, bytes) else v)
                for k, v in f.attrs.items()}


def _grid_metres(attrs, shape):
    """
    Per-pixel distance grid helper: returns (x, y) in metres and a flag for
    whether the product was geographic.

    MintPy products come in two coordinate systems and the difference is not
    cosmetic: X_STEP is 30.0 for a UTM product and 0.00027 for a geographic
    one. Treating degrees as metres puts every distance off by 10^5, so the
    units are decided from the step size rather than assumed.
    """
    ny, nx = shape
    y0 = float(attrs["Y_FIRST"])
    dx, dy = float(attrs["X_STEP"]), float(attrs["Y_STEP"])
    yy, xx = np.mgrid[0:ny, 0:nx]
    if abs(dx) < 0.01:                      # degrees
        lat = y0 + yy * dy
        mlat = 110_900.0
        mlon = 111_320.0 * np.cos(np.radians(np.nanmean(lat)))
        return (xx * dx * mlon, yy * dy * mlat), True
    return (xx * dx, yy * dy), False        # already metres


def utm_zone_number(lon: float) -> int:
    """Standard 6-degree UTM zone (1-60) of a longitude. The Norway and
    Svalbard exceptions are not applied: a product there states its EPSG."""
    return int(np.floor((float(lon) + 180.0) / 6.0)) % 60 + 1


def epsg_attr(attrs):
    """The product's own EPSG code (int), or None when the metadata does not
    state one. MintPy writes it as '32616', h5 round trips as '32616.0'."""
    v = attrs.get("EPSG", attrs.get("epsg"))
    if isinstance(v, bytes):
        v = v.decode()
    try:
        return int(float(v)) if v not in (None, "", "None") else None
    except (TypeError, ValueError):
        return None


def product_epsg(attrs, lonlat=None, shape=None) -> int:
    """
    EPSG code of a projected (metre) MintPy grid. There is no default zone.

      1. The product's EPSG attribute, when present, is the answer.
      2. The zone NUMBER comes from UTM_ZONE ('56S', '11N', '16'), else from
         the longitude of `lonlat` - the point the caller is placing (the AOI
         centre, the reference point, the polygon). Without UTM_ZONE the
         neighbouring zones are tried as well, since a frame-wide product
         near a zone edge can be projected in the next zone.
      3. The HEMISPHERE: with `lonlat` and the grid origin, whichever
         candidate puts that point on (or nearest) the grid. A northern and
         a southern UTM projection of one point differ by exactly 10,000 km
         of northing, so this cannot be confused, and it does not depend on
         how a letter was written: MintPy's 'S' means south, but a latitude-
         band 'S' is 32-40 N (Corcoran is in band 11S). Without `lonlat`,
         UTM_ZONE's letter as MintPy writes it ('N' north, 'S' south; other
         band letters, C-M south and P-X north).

    `shape` (rows, cols), or LENGTH/WIDTH in the metadata, gives the grid's
    extent for step 3; without either only the origin corner is used.

    Raises ValueError when neither the metadata nor `lonlat` place the grid.
    Until 8 Oct 2026 the fallback here was UTM 16N (Indiana) with the
    hemisphere dropped, which put a southern grid (Appin, 56S) 10,000 km
    off without an error.
    """
    epsg = epsg_attr(attrs)
    if epsg is not None:
        return epsg
    hint = None
    if lonlat is not None:
        lon, lat = (float(v) for v in lonlat)
        if np.isfinite(lon) and np.isfinite(lat):
            hint = (lon, lat)
    label = attrs.get("UTM_ZONE")
    label = (label.decode() if isinstance(label, bytes) else str(label or "")).strip().upper()
    # a zone is 1-60 with an optional letter; anything else ('None', 'NAN',
    # a mangled value) states no zone - its last letter is not a hemisphere
    m = re.fullmatch(r"(\d{1,2})\s*([A-Z]?)", label)
    if m and 1 <= int(m.group(1)) <= 60:
        digits, letter = m.group(1), m.group(2)
    else:
        if label:
            log.warning("UTM_ZONE %r is not a UTM zone; ignoring it", label)
        digits = letter = ""
    if digits:
        zones = [int(digits)]
    elif hint is not None:
        z = utm_zone_number(hint[0])
        zones = [z, (z - 2) % 60 + 1, z % 60 + 1]          # the hint's zone first, then its neighbours
    else:
        raise ValueError("projected product states no EPSG or UTM_ZONE and no location was given "
                         "to derive one from - there is no default zone")
    if letter in ("N", "S"):
        south_label = letter == "S"
    elif letter:
        south_label = letter < "N"                           # latitude band C-M is south
    else:
        south_label = None
    origin = None
    try:
        origin = (float(attrs["X_FIRST"]), float(attrs["Y_FIRST"]),
                  float(attrs["X_STEP"]), float(attrs["Y_STEP"]))
    except (KeyError, TypeError, ValueError):
        pass
    if hint is None or origin is None:
        if len(zones) > 1:
            zones = zones[:1]
        south = south_label if south_label is not None else (hint[1] < 0 if hint else None)
        if south is None:
            raise ValueError(f"UTM_ZONE {label!r} names no hemisphere and no location was given "
                             "to decide it - there is no default")
        return (32700 if south else 32600) + zones[0]

    from pyproj import Transformer
    x0, y0, dx, dy = origin
    if shape is not None:
        ny, nx = int(shape[0]), int(shape[1])
    else:
        try:
            ny, nx = int(float(attrs["LENGTH"])), int(float(attrs["WIDTH"]))
        except (KeyError, TypeError, ValueError):
            ny = nx = 0
    xs = sorted((x0, x0 + nx * dx))
    ys = sorted((y0, y0 + ny * dy))
    best = None
    for z in zones:
        for south in (False, True):
            code = (32700 if south else 32600) + z
            x, y = Transformer.from_crs("EPSG:4326", f"EPSG:{code}",
                                        always_xy=True).transform(*hint)
            d = float(np.hypot(max(0.0, xs[0] - x, x - xs[1]), max(0.0, ys[0] - y, y - ys[1])))
            if not np.isfinite(d):
                continue
            if best is None or d < best[0] - 1e-6:
                best = (d, code)
    if best is None:
        raise ValueError(f"could not place the grid of a product with UTM_ZONE {label!r} "
                         f"around {hint} - there is no default zone")
    code = best[1]
    if south_label is not None and (code >= 32700) != south_label:
        log.warning("UTM_ZONE %r reads as the %s hemisphere, but the grid lies where EPSG:%d puts "
                    "the point (%.4f, %.4f); using EPSG:%d", label,
                    "southern" if south_label else "northern", code, hint[0], hint[1], code)
    if not digits:
        log.info("No EPSG or UTM_ZONE in the product metadata; EPSG:%d from the location "
                 "(%.4f, %.4f) and the grid origin", code, hint[0], hint[1])
    return code


def _reference_rc(attrs, shape, ref_lalo=None):
    """
    Locate the reference pixel.

    Prefers REF_Y/REF_X, which is what MintPy actually referenced to, over a
    lat/lon the caller believes was used - those disagree more often than is
    comfortable, and every number in the product is relative to the former.
    """
    if attrs.get("REF_Y") not in (None, "", "None"):
        try:
            return int(float(attrs["REF_Y"])), int(float(attrs["REF_X"])), "REF_Y/REF_X"
        except (TypeError, ValueError):
            pass
    if ref_lalo:
        lat, lon = [float(v) for v in str(ref_lalo).split(",")]
        x0, y0 = float(attrs["X_FIRST"]), float(attrs["Y_FIRST"])
        dx, dy = float(attrs["X_STEP"]), float(attrs["Y_STEP"])
        if abs(dx) < 0.01:
            r, c = int(round((lat - y0) / dy)), int(round((lon - x0) / dx))
        else:
            from pyproj import Transformer
            # the product's EPSG, else its UTM zone and hemisphere placed by
            # this very point (no Indiana 16N default, no dropped 'S')
            epsg = product_epsg(attrs, lonlat=(lon, lat), shape=shape)
            x, y = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}",
                                        always_xy=True).transform(lon, lat)
            r, c = int(round((y - y0) / dy)), int(round((x - x0) / dx))
        ny, nx = shape
        if 0 <= r < ny and 0 <= c < nx:
            return r, c, "ref_lalo"
        log.warning("ref_lalo %s falls outside the grid - using the scene centre",
                    ref_lalo)
    ny, nx = shape
    return ny // 2, nx // 2, "scene centre (NO REFERENCE FOUND)"


def step_rms_vs_distance(ts_file, ref_lalo=None, bins_km=(0.5, 2, 5, 10, 20, 50),
                         mask_file=None, block_rows=64,
                         step_threshold_mm=None):
    """
    Epoch-to-epoch displacement step statistics, binned by distance from the
    reference pixel.

    Streams the cube in row blocks rather than loading it: a 254-epoch
    1678x1678 float32 series is 2.9 GB, and this has to run on the same
    machine that is processing a stack.

    Returns a dict with one record per distance bin holding the plain RMS,
    a robust RMS (1.4826 x MAD, which ignores the tail a handful of gross
    unwrapping failures would otherwise dominate), the fraction of steps
    over the threshold, and how many of those sit within 0.15 cycles of a
    whole multiple of the half wavelength.

    THAT LAST NUMBER NEEDS A BASELINE TO MEAN ANYTHING. A step drawn at
    random lands within 0.15 cycles of an integer 30% of the time, so 30%
    is the null result, not zero. Only a clear excess over that is evidence
    of cycle slips, and the function reports the excess rather than letting
    a raw percentage look damning on its own.

    The cycle is the PRODUCT's half wavelength (half_wavelength_mm: 27.73 mm
    for Sentinel-1, ~122 mm for NISAR L-band), and step_threshold_mm defaults
    to the same fraction of it as Sentinel-1's 20 mm (M1, 8 Oct 2026). A
    non-Sentinel-1 record also carries 'half_wavelength_mm' and
    'wavelength_m'; a Sentinel-1 record is unchanged.
    """
    import h5py

    ts_file = Path(ts_file)
    attrs = _open_attrs(ts_file)
    cycle_mm = half_wavelength_mm(attrs)
    if step_threshold_mm is None:
        step_threshold_mm = default_step_threshold_mm(attrs)

    with h5py.File(ts_file, "r") as f:
        key = "timeseries" if "timeseries" in f else None
        if key is None:
            cands = [k for k in f.keys() if f[k].ndim == 3]
            if not cands:
                raise ValueError(f"no 3-D dataset in {ts_file}")
            key = cands[0]
        nt, ny, nx = f[key].shape

    (gx, gy), was_geographic = _grid_metres(attrs, (ny, nx))
    rr, cc, ref_src = _reference_rc(attrs, (ny, nx), ref_lalo)
    dist_km = np.sqrt((gx - gx[rr, cc]) ** 2 + (gy - gy[rr, cc]) ** 2) / 1000.0

    mask = None
    if mask_file and Path(mask_file).exists():
        with h5py.File(mask_file, "r") as f:
            mk = [k for k in f.keys() if f[k].shape == (ny, nx)]
            if mk:
                mask = f[mk[0]][:].astype(bool)

    edges = [0.0] + list(bins_km)
    acc = [{"n": 0, "sumsq": 0.0, "abs": [], "big": [], "lo": a, "hi": b}
           for a, b in zip(edges, edges[1:])]

    with h5py.File(ts_file, "r") as f:
        ds = f[key]
        for r0 in range(0, ny, block_rows):
            r1 = min(r0 + block_rows, ny)
            blk = ds[:, r0:r1, :].astype(np.float32) * 1000.0      # m -> mm
            if mask is not None:
                blk[:, ~mask[r0:r1, :]] = np.nan
            steps = np.diff(blk, axis=0)
            del blk
            d = dist_km[r0:r1, :]
            for a in acc:
                sel = (d >= a["lo"]) & (d < a["hi"])
                if not sel.any():
                    continue
                v = steps[:, sel]
                v = v[np.isfinite(v)]
                if v.size == 0:
                    continue
                a["n"] += v.size
                a["sumsq"] += float(np.sum(v.astype(np.float64) ** 2))
                # Subsample for the robust statistics: a full copy of every
                # step in the scene is gigabytes, and a median does not need it.
                if v.size > 20000:
                    v = np.random.default_rng(0).choice(v, 20000, replace=False)
                a["abs"].append(np.abs(v))
                big = v[np.abs(v) > step_threshold_mm]
                if big.size:
                    a["big"].append(big)
            del steps

    # How far the scene actually reaches from the reference: a "20-50 km"
    # bin on a 31 km scene is really 20-31 km, and a comparison with a
    # 50 km product has to say so.
    reach = dist_km if mask is None else np.where(mask, dist_km, np.nan)
    out = {"file": str(ts_file), "n_epochs": nt, "reference_rc": [rr, cc],
           "reference_source": ref_src, "geographic_grid": was_geographic,
           "step_threshold_mm": step_threshold_mm,
           "max_distance_km": round(float(np.nanmax(reach)), 1),
           "mask_file": str(mask_file) if mask is not None else None,
           "bins": []}
    if cycle_mm != C_BAND_HALF_WAVELENGTH_MM:
        out["half_wavelength_mm"] = round(cycle_mm, 2)
        out["wavelength_m"] = product_wavelength_m(attrs)

    for a in acc:
        if a["n"] == 0:
            continue
        absv = np.concatenate(a["abs"]) if a["abs"] else np.array([])
        rms = float(np.sqrt(a["sumsq"] / a["n"]))
        robust = float(1.4826 * np.median(absv)) if absv.size else float("nan")
        rec = {"lo_km": a["lo"], "hi_km": a["hi"], "n_steps": int(a["n"]),
               "step_rms_mm": round(rms, 2),
               "step_rms_robust_mm": round(robust, 2)}
        if a["big"]:
            big = np.concatenate(a["big"])
            frac_int = float(np.mean(
                np.abs(np.round(big / cycle_mm) - big / cycle_mm) < 0.15))
            rec.update({
                "frac_steps_over_threshold": round(float(big.size) / a["n"], 4),
                "frac_near_whole_cycle": round(frac_int, 3),
                # 0.30 is what random steps give. Report the excess so nobody
                # reads 44% as damning when the null is 30%.
                "cycle_excess_over_chance": round(frac_int - 0.30, 3),
            })
        out["bins"].append(rec)

    if len(out["bins"]) >= 2:
        near, far = out["bins"][0]["step_rms_mm"], out["bins"][-1]["step_rms_mm"]
        out["far_over_near_ratio"] = round(far / near, 2) if near > 0 else None
    return out


def format_report(res) -> str:
    """Human-readable table. The ratio at the bottom is the acceptance number."""
    lines = [
        f"Step RMS vs distance from reference  ({Path(res['file']).name})",
        f"  {res['n_epochs']} epochs, reference at row/col {res['reference_rc']} "
        f"from {res['reference_source']}",
        "",
        f"  {'range (km)':>14}  {'n steps':>11}  {'RMS':>8}  {'robust':>8}  "
        f"{'>thr':>7}  {'cycle excess':>13}",
    ]
    for b in res["bins"]:
        cyc = b.get("cycle_excess_over_chance")
        cyc_s = f"{cyc:+.0%}" if cyc is not None else "-"
        frac = b.get("frac_steps_over_threshold")
        lines.append(
            f"  {b['lo_km']:5.1f} - {b['hi_km']:5.1f}  {b['n_steps']:11,d}  "
            f"{b['step_rms_mm']:7.1f}  {b['step_rms_robust_mm']:7.1f}  "
            f"{(f'{frac:6.2%}' if frac is not None else '     -')}  {cyc_s:>13}"
        )
    if res.get("far_over_near_ratio"):
        lines += ["", f"  far/near ratio: {res['far_over_near_ratio']}x   "
                      f"(practice: under 1.5x)"]
    return "\n".join(lines)


def main():
    import argparse, json
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("timeseries", help="MintPy-schema timeseries HDF5")
    p.add_argument("--ref-lalo", default=None, help="fallback reference 'lat,lon'")
    p.add_argument("--mask", default=None, help="mask HDF5, e.g. maskTempCoh.h5")
    p.add_argument("--bins", default="0.5,2,5,10,20,50")
    p.add_argument("--json", default=None, help="also write the record here")
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    res = step_rms_vs_distance(
        a.timeseries, ref_lalo=a.ref_lalo, mask_file=a.mask,
        bins_km=tuple(float(x) for x in a.bins.split(",")))
    print(format_report(res))
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=2))
        print(f"\nwrote {a.json}")


if __name__ == "__main__":
    main()


# ----------------------------------------------------------------------
# reference point selection
# ----------------------------------------------------------------------
def select_reference_candidates(cfg, n=5, min_tc=0.9, exclude_buffer_m=3000.0,
                                edge_margin_deg=0.015, min_sep_m=3000.0, grid_step=4):
    """
    Rank stable far-field pixels to reference a time series to.

    Practice in the field is to reference to ground argued to be stable
    and to CHECK that it is - a reference inherited from another product
    turned out to rise ~0.5 in/yr against its own surroundings over 9.5
    years, offsetting every absolute rate. This scores every coherent
    pixel on the geocoded grid and returns the best `n`, spatially
    separated, with the evidence for each:

      keep       temporal coherence >= min_tc, outside every deforming
                 polygon + exclude_buffer_m, inside the AOI by edge_margin
      consistent |velocity - far-field median| small (agrees with the
                 surrounding ground, whatever the current reference does)
      smooth     5x5 velocity std small (not an isolated bright pixel)
      steady     the pixel's own series minus the far-field mean series
                 has a small trend and scatter over the record

    Returns a list of dicts sorted best first; also writes
    <project>/reference_candidates.json.
    """
    import json
    import h5py
    from scipy.ndimage import binary_dilation, uniform_filter
    from insar_pipeline import deliverables as dl
    from insar_pipeline.stack_backend import _polygon_mask, _pixel_metres

    vel, attrs = dl._read(dl._product(cfg, "velocity.h5"), "velocity")
    tc, _ = dl._read(dl._product(cfg, "temporalCoherence.h5"), "temporalCoherence")
    vel = vel * 1000.0                                   # mm/yr
    ny, nx = vel.shape
    good = np.isfinite(vel) & (tc >= min_tc)
    px_m = _pixel_metres(attrs)
    geojson = getattr(cfg, "deramp_exclude_geojson", "")
    if geojson and Path(geojson).exists():
        deform = _polygon_mask(Path(geojson), vel.shape, attrs)
        deform = binary_dilation(deform, iterations=max(1, int(exclude_buffer_m / px_m)))
    else:
        deform = np.zeros_like(good)
    x0, y0 = float(attrs["X_FIRST"]), float(attrs["Y_FIRST"])
    dx, dy = float(attrs["X_STEP"]), float(attrs["Y_STEP"])
    lon = x0 + np.arange(nx) * dx
    lat = y0 + np.arange(ny) * dy
    w, s, e, nn = cfg.bbox
    m = edge_margin_deg
    inside = np.outer((lat >= s + m) & (lat <= nn - m), (lon >= w + m) & (lon <= e - m))
    far = good & ~deform & inside
    far_med = float(np.nanmedian(vel[far]))
    # local smoothness on the coherent field
    fin = np.isfinite(vel) & (tc >= 0.4)
    z = np.where(fin, vel, 0.0)
    cnt = uniform_filter(fin.astype(float), 5, mode="nearest") * 25
    m1 = uniform_filter(z, 5, mode="nearest") * 25 / np.maximum(cnt, 1)
    m2 = uniform_filter(z * z, 5, mode="nearest") * 25 / np.maximum(cnt, 1)
    lstd = np.sqrt(np.clip(m2 - m1 * m1, 0, None))
    ltc = uniform_filter(np.where(np.isfinite(tc), tc, 0.0), 5, mode="nearest")
    score = np.where(far & (cnt >= 20),
                     np.abs(vel - far_med) + 2.0 * lstd + 10.0 * (1.0 - ltc), np.inf)
    # coarse grid keeps the candidate search cheap and spatially spread
    cand = []
    ys, xs = np.mgrid[0:ny:grid_step, 0:nx:grid_step]
    sc = score[::grid_step, ::grid_step]
    order = np.argsort(sc, axis=None)
    sep_px = min_sep_m / px_m
    for k in order:
        if not np.isfinite(sc.flat[k]):
            break
        yy, xx = int(ys.flat[k]), int(xs.flat[k])
        if any((yy - c["y"]) ** 2 + (xx - c["x"]) ** 2 < sep_px ** 2 for c in cand):
            continue
        cand.append({"y": yy, "x": xx})
        if len(cand) >= n * 3:
            break
    # steadiness from the time series: candidate minus far-field mean
    ts_file = dl.timeseries_file(cfg)
    with h5py.File(ts_file, "r") as f:
        dates = [d.decode() if isinstance(d, bytes) else str(d) for d in f["date"][:]]
        ds = f["timeseries"]
        t = np.array([(datetime.strptime(d, "%Y%m%d") - datetime(2000, 1, 1)).days
                      for d in dates]) / 365.25
        from insar_pipeline.significance import iter_epochs     # chunk-aligned: ds[i] re-reads 8 dates per date
        far_series = np.array([np.nanmean(ep[far]) for _, ep in iter_epochs(ds)]) * 1000.0
        for c in cand:
            s_ = ds[:, c["y"], c["x"]] * 1000.0 - far_series
            ok = np.isfinite(s_)
            A = np.c_[t[ok], np.ones(ok.sum())]
            coef, res, _, _ = np.linalg.lstsq(A, s_[ok], rcond=None)
            resid = s_[ok] - A @ coef
            c.update({"lat": round(float(lat[c["y"]]), 6), "lon": round(float(lon[c["x"]]), 6),
                      "temporal_coherence": round(float(tc[c["y"], c["x"]]), 3),
                      "local_tc_5x5": round(float(ltc[c["y"], c["x"]]), 3),
                      "velocity_minus_far_field_mm_yr": round(float(vel[c["y"], c["x"]] - far_med), 2),
                      "local_velocity_std_mm_yr": round(float(lstd[c["y"], c["x"]]), 2),
                      "trend_vs_far_field_mm_yr": round(float(coef[0]), 2),
                      "scatter_vs_far_field_mm": round(float(np.std(resid)), 2),
                      "n_epochs": int(ok.sum())})
    for c in cand:
        c["rank_score"] = round(abs(c["trend_vs_far_field_mm_yr"]) + 0.2 * c["scatter_vs_far_field_mm"]
                                + 0.5 * c["local_velocity_std_mm_yr"], 3)
    cand.sort(key=lambda c: c["rank_score"])
    cand = cand[:n]
    out = {"far_field_median_velocity_mm_yr": round(far_med, 2),
           "far_field_pixels": int(far.sum()), "min_tc": min_tc,
           "exclude_buffer_m": exclude_buffer_m, "candidates": cand,
           "current_reference": attrs.get("REF_LAT", "") and
           {"lat": float(attrs["REF_LAT"]), "lon": float(attrs["REF_LON"])}}
    (cfg.project_dir / "reference_candidates.json").write_text(json.dumps(out, indent=2))
    return cand, out


# ----------------------------------------------------------------------
# radar grid -> product grid
# ----------------------------------------------------------------------
def radar_to_geo_index(lat_r, lon_r, attrs, shape, max_dist_m=None, block_rows=256):
    """
    For every cell of the geocoded product grid, the flat index of the
    nearest radar-grid pixel (from geometryRadar.h5 latitude/longitude), or
    -1 where the nearest one is farther than `max_dist_m`.

    WHY. Several diagnostics MintPy computes stay in radar coordinates and
    are never geocoded on the official route (temporalCoherence_mintpy.h5,
    demErr.h5, numTriNonzeroIntAmbiguity*.h5, the pre-deramp series). The QA
    needs them on the product grid to say anything about a report area, and
    it must work on finished projects without re-running MintPy's geocode,
    so it maps them itself. Nearest neighbour is the honest resampling for
    a diagnostic: it never invents a value between a water pixel and a field.
    Measured on Site B (586x595 radar, 1882x1854 product at 15 m): the
    radar velocity mapped this way matches geo_velocity.h5 to the posting.
    """
    from scipy.spatial import cKDTree
    lat_r = np.asarray(lat_r, dtype=np.float64)
    lon_r = np.asarray(lon_r, dtype=np.float64)
    ok = np.isfinite(lat_r) & np.isfinite(lon_r) & ((lat_r != 0) | (lon_r != 0))
    flat = np.flatnonzero(ok.ravel())
    ny, nx = shape
    out = np.full(ny * nx, -1, dtype=np.int32)                    # radar grids are far below 2**31 cells
    if flat.size == 0:
        return out.reshape(shape)
    lat0 = float(np.nanmedian(lat_r[ok]))
    mx = 111_320.0 * np.cos(np.radians(lat0)); my = 110_900.0
    tree = cKDTree(np.c_[lon_r.ravel()[flat] * mx, lat_r.ravel()[flat] * my])
    if max_dist_m is None:
        # 1.5 radar pixels: from the spacing actually present in the arrays
        dy_m = float(np.nanmedian(np.abs(np.diff(lat_r, axis=0)))) * my
        dx_m = float(np.nanmedian(np.abs(np.diff(lon_r, axis=1)))) * mx
        max_dist_m = 1.5 * max(dy_m, dx_m, 1.0)
    x0, y0 = float(attrs["X_FIRST"]), float(attrs["Y_FIRST"])
    dx, dy = float(attrs["X_STEP"]), float(attrs["Y_STEP"])
    lon_g = (x0 + (np.arange(nx) + 0.5) * dx) * mx
    for r0 in range(0, ny, block_rows):
        r1 = min(r0 + block_rows, ny)
        lat_g = (y0 + (np.arange(r0, r1) + 0.5) * dy) * my
        q = np.c_[np.repeat(lon_g[None, :], r1 - r0, 0).ravel(), np.repeat(lat_g[:, None], nx, 1).ravel()]
        d, i = tree.query(q, distance_upper_bound=max_dist_m)
        hit = np.isfinite(d)
        blk = np.full(q.shape[0], -1, dtype=np.int32)
        blk[hit] = flat[i[hit]]
        out[r0 * nx:r1 * nx] = blk
    return out.reshape(shape)


def map_to_geo(radar_arr, index):
    """A radar-grid array on the product grid through radar_to_geo_index (NaN off the swath)."""
    a = np.asarray(radar_arr, dtype=np.float32).ravel()
    out = np.full(index.shape, np.nan, dtype=np.float32)          # a diagnostic layer: float32 halves the RAM
    hit = index >= 0
    out[hit] = a[index[hit]]
    return out


# ----------------------------------------------------------------------
# the coherence noise floor
# ----------------------------------------------------------------------
# ONE population for every project (decision E21, 7 Oct 2026): the cells
# MintPy's own network temporal coherence calls decorrelated, below
# time_series.UNWRAP_NOISE_TC (0.25) - the population the time-series step's
# linked_noise_level reads. Until 7 Oct the first of 0.15 / 0.2 / 0.25 with
# 300 cells was taken and the floor was median + 3 lower-half sigma, so the
# floor moved with the population: Site A got <0.15 (0.397, passes at
# 0.4) and Site B <0.2 (0.436, fails), while on one population both read
# alike. The median is dragged up by the population's upper tail (partly
# coherent cells with unwrapping errors, water included); the MODE is not:
# mode + 3 sigma reads 0.393-0.394 (Site B) and 0.397-0.398 (Site A)
# on <0.2, <0.25 and water-only cells alike (audit_chain_verify/noise_mode).
NOISE_POPULATION = 0.25            # = time_series.UNWRAP_NOISE_TC
NOISE_MIN_CELLS = 300
NOISE_SIGMAS = 3.0                 # the reviewer's criterion: typical noise + 3 sigma


def noise_level(linked_tc, unwrap_tc, population=NOISE_POPULATION, min_cells=NOISE_MIN_CELLS,
                on_water=None) -> dict:
    """
    Where pure noise sits on the linked (phase-linking) coherence scale.

    MiaplPy's quality figure is not zero on noise: an EMI fit to pure noise
    over 10-13 images and 40-200 similar pixels scores 0.34-0.42 (the
    reviewer's simulation), and the open water of Gibson Lake reads 0.37. A
    working threshold is only a reliability statement when it sits clear of
    that level. Measured on cells the independent unwrapping-consistency
    layer (MintPy's network temporal coherence) calls decorrelated, so the
    measurement does not depend on the threshold under test.

    The rule, the same on every project (E21, 7 Oct 2026):
      population  linked coherence > 0 on cells whose MintPy temporal
                  coherence is below `population` (0.25), at least
                  `min_cells` of them;
      mode        the peak of a 500-bin histogram over 0-1, smoothed over
                  5 bins (time_series.linked_noise_level's estimator);
      sigma       the 68.27th percentile of the distances BELOW the mode
                  (the upper tail is coherent ground with unwrapping errors,
                  which would otherwise widen the band and excuse a
                  threshold inside it);
      floor       mode + 3 sigma.
    `on_water` (optional boolean array, same shape) marks mapped water: the
    population's share on it is recorded, so a caption says "open water"
    only when most of the noise IS water. `population_median` is kept for
    the record; it is not the noise level.
    """
    lt = np.asarray(linked_tc, dtype=np.float64)
    ut = np.asarray(unwrap_tc, dtype=np.float64)
    ok = np.isfinite(lt) & np.isfinite(ut) & (lt > 0)
    sel = ok & (ut < population)
    n = int(sel.sum())
    label = f"cells with unwrapping consistency (MintPy temporal coherence) < {population}"
    method = (f"mode + {NOISE_SIGMAS:g} sigma of the linked coherence over {label}: mode of a 500-bin histogram "
              "(5-bin smoothing), sigma the 68.27th percentile of the distances below the mode; one population "
              "and one estimator for every project")
    out = {"mode": None, "sigma": None, "floor": None, "n_cells": n, "measured": False,
           "population": label, "method": method}
    if n < min_cells:
        out["population"] = (f"fewer than {min_cells} {label}: no decorrelated ground to measure the noise on")
        return out
    v = lt[sel]
    h, e = np.histogram(v, bins=500, range=(0.0, 1.0))
    hs = np.convolve(h, np.ones(5) / 5.0, mode="same")
    mode = float(e[int(np.argmax(hs))] + 0.001)                   # the bin centre
    below = mode - v[v < mode]
    if below.size < min_cells // 2:
        out["population"] = (f"{label}: only {below.size} of them lie below the mode {mode:.3f}, too few for "
                             "the lower-half spread")
        return out
    sig = float(np.percentile(below, 68.27))
    out.update({"mode": round(mode, 4), "sigma": round(sig, 4), "floor": round(mode + NOISE_SIGMAS * sig, 4),
                "population_median": round(float(np.median(v)), 4),
                "p95": round(float(np.percentile(v, 95)), 4), "measured": True})
    if on_water is not None and np.shape(on_water) == lt.shape:
        out["share_on_mapped_water"] = round(float(np.asarray(on_water, bool)[sel].mean()), 3)
    return out


# ----------------------------------------------------------------------
# polynomial surfaces (deramp sensitivity)
# ----------------------------------------------------------------------
def surface_fit(field, fit_mask, order=1, max_cells=400_000, seed=0):
    """
    The least-squares plane (order 1) or quadratic (order 2) through `field`
    over `fit_mask`, evaluated on the whole grid, in normalised row/col
    coordinates (MintPy's deramp is the same model in its own grid).

    Fitting is linear in the data, so a ramp fitted to a rate field is the
    rate of the per-epoch ramps whenever the fit area is the same on every
    epoch: the deramp sensitivity can be computed on the velocity alone,
    without re-reading a cube of tens of GB. A random subset of at most
    `max_cells` fit cells keeps the solve cheap; the plane through 400 k of
    3.5 M cells differs from the full fit by far less than 0.01 mm/yr.
    """
    f = np.asarray(field, dtype=np.float64)
    ny, nx = f.shape
    sel = np.flatnonzero((fit_mask & np.isfinite(f)).ravel())
    if sel.size < 10 * (3 if order == 1 else 6):
        return None
    if sel.size > max_cells:
        sel = np.sort(np.random.default_rng(seed).choice(sel, max_cells, replace=False))

    def design(rows, cols):
        y = rows / max(ny - 1, 1) - 0.5; x = cols / max(nx - 1, 1) - 0.5
        cols_ = [np.ones_like(x), x, y]
        if order >= 2:
            cols_ += [x * x, y * y, x * y]
        return np.stack(cols_, axis=-1)
    r, c = np.divmod(sel, nx)
    coef = np.linalg.lstsq(design(r.astype(float), c.astype(float)), f.ravel()[sel], rcond=None)[0]
    # evaluated in row blocks: the full-grid design matrix is 6 x 28 MB
    out = np.empty((ny, nx))
    cols = np.arange(nx, dtype=float)
    for r0 in range(0, ny, 256):
        rows = np.arange(r0, min(r0 + 256, ny), dtype=float)
        out[r0:r0 + rows.size] = design(rows[:, None] + 0 * cols[None, :], cols[None, :] + 0 * rows[:, None]) @ coef
    return out


def stream_rate(ts_file, minus_file=None, dataset="timeseries"):
    """
    Per-pixel straight-line rate (mm/yr) of a MintPy-schema cube, or of the
    difference of two cubes on the same grid and dates (the ramp a deramp
    step removed: timeseries_ERA5 minus timeseries_ERA5_ramp). Streams in
    chunk-aligned epoch blocks, holding five accumulators of the 2-D grid.
    """
    import h5py
    from insar_pipeline.significance import iter_epochs, years
    with h5py.File(ts_file, "r") as f:
        dates = [d.decode() if isinstance(d, bytes) else str(d) for d in f["date"][:]]
        t = years(dates)
        g = h5py.File(minus_file, "r") if minus_file else None
        try:
            if g is not None:
                gd = [d.decode() if isinstance(d, bytes) else str(d) for d in g["date"][:]]
                if gd != dates or g[dataset].shape != f[dataset].shape:
                    raise ValueError(f"{Path(minus_file).name} does not share the dates/grid of {Path(ts_file).name}")
            shape = f[dataset].shape[1:]
            n = np.zeros(shape); st = np.zeros(shape); stt = np.zeros(shape); sy = np.zeros(shape); sty = np.zeros(shape)
            it_b = iter_epochs(g[dataset]) if g is not None else None
            for i, ep in iter_epochs(f[dataset]):
                y = ep.astype(np.float64) * 1000.0
                if it_b is not None:
                    y = y - next(it_b)[1].astype(np.float64) * 1000.0
                ok = np.isfinite(y)
                ti = float(t[i])
                n += ok; st += ok * ti; stt += ok * ti * ti
                y0 = np.where(ok, y, 0.0); sy += y0; sty += y0 * ti
        finally:
            if g is not None:
                g.close()
    den = n * stt - st * st
    with np.errstate(invalid="ignore", divide="ignore"):
        rate = np.where((n >= 6) & (den > 0), (n * sty - st * sy) / den, np.nan)
    return rate, dates


# ----------------------------------------------------------------------
# rate-gradient plausibility
# ----------------------------------------------------------------------
def gradient_limit_mm_yr(wavelength_m, dates) -> dict:
    """
    The largest rate DIFFERENCE between adjacent looked cells that unwrapping
    can follow: a quarter wavelength (13.9 mm LOS at C-band) per typical
    nearest-date interval. Unwrapping is spatial, so a broad smooth bowl can
    sink faster than this in absolute terms and be measured correctly; what
    cannot be measured is a neighbour-to-neighbour difference above it, which
    aliases into a whole-cycle error. At 12-day revisit that is ~422 mm/yr
    between 41 m cells (Site B: median gap 12 d, 88% of steps).
    """
    from datetime import datetime as _dt
    d = sorted(dates)
    gaps = np.diff([(_dt.strptime(x, "%Y%m%d") - _dt(2000, 1, 1)).days for x in d]) if len(d) > 1 else np.array([12])
    gap = float(np.median(gaps)) if gaps.size else 12.0
    q = 1000.0 * float(wavelength_m) / 4.0
    return {"quarter_wavelength_mm": round(q, 2), "median_gap_days": gap,
            "p90_gap_days": float(np.percentile(gaps, 90)) if gaps.size else gap,
            "limit_mm_yr": round(q * 365.25 / max(gap, 1.0), 1)}


def rate_gradient_flags(vel, step_rc=(1, 1), limit=422.0, valid=None):
    """Cells whose rate differs from a neighbour `step_rc` cells away (N, S,
    E or W) by more than `limit` - more than unwrapping can follow."""
    v = np.asarray(vel, dtype=np.float64)
    ok = np.isfinite(v) if valid is None else (valid & np.isfinite(v))
    vv = np.where(ok, v, np.nan)
    bad = np.zeros(v.shape, bool)
    sr, sc = (max(1, int(s)) for s in step_rc)
    with np.errstate(invalid="ignore"):
        d = np.abs(vv[sr:, :] - vv[:-sr, :]) > limit
        bad[sr:, :] |= d; bad[:-sr, :] |= d
        d = np.abs(vv[:, sc:] - vv[:, :-sc]) > limit
        bad[:, sc:] |= d; bad[:, :-sc] |= d
    return bad & ok


# ----------------------------------------------------------------------
# geolocation against water edges
# ----------------------------------------------------------------------
def water_edge_offset(field, water_fine, factor, cell_m, max_shift_m=75.0, valid=None) -> dict:
    """
    Where the product's own low-coherence water sits against an independent
    map of water, for one window, by a shift search.

    `field` is a coherence layer on the product grid (water is dark);
    `water_fine` the independent water mask rasterised `factor` times finer
    on the same origin and padded by S = max_shift_m / (cell_m / factor)
    fine cells on every side, i.e. of shape (ny*factor + 2S, nx*factor + 2S).
    For each trial shift of the map in 1/factor-cell steps the fraction of
    each product cell covered by water is formed and correlated, negatively,
    with the coherence. The best shift is reported as where the RASTER sits
    relative to the map: dE > 0 means the product is east of the water map.
    Offsets are None when the best correlation is under 0.2 (no usable edge).
    """
    f = np.asarray(field, dtype=np.float64)
    ny, nx = f.shape
    fine_m = cell_m / factor
    S = int(round(max_shift_m / fine_m))
    if np.shape(water_fine) != (ny * factor + 2 * S, nx * factor + 2 * S):
        raise ValueError(f"water_fine must be {(ny * factor + 2 * S, nx * factor + 2 * S)} for a "
                         f"{ny}x{nx} window at factor {factor} and {max_shift_m} m search; got {np.shape(water_fine)}")
    ok = np.isfinite(f) if valid is None else (valid & np.isfinite(f))
    sc = _window_scores(f, water_fine, factor, cell_m, max_shift_m, ok)
    if sc is None:
        return {"dE_m": None, "dN_m": None, "offset_m": None, "score": None, "score_at_zero": None,
                "n_cells": int(ok.sum())}
    k = np.unravel_index(np.argmax(sc), sc.shape)
    sy, sx = k[0] - S, k[1] - S
    usable = sc[k] > 0.2
    # sampling the map sx fine cells EAST puts the map's water under the
    # raster cell: the raster shows that water -sx fine cells from where it is
    return {"dE_m": round(-sx * fine_m, 2) if usable else None,
            "dN_m": round(sy * fine_m, 2) if usable else None,
            "offset_m": round(float(np.hypot(sx, sy)) * fine_m, 2) if usable else None,
            "score": round(float(sc[k]), 3), "score_at_zero": round(float(sc[S, S]), 3),
            "n_cells": int(ok.sum()), "step_m": round(fine_m, 2)}


def rasterise_lonlat(features, x0, y0, dx, dy, shape):
    """Polygon features (GeoJSON geometry dicts, lon/lat) burned onto a
    north-up grid whose outer corner is (x0, y0) and cell (dx, dy) degrees."""
    from rasterio import features as rfeat
    from rasterio.transform import Affine
    geoms = [g for g in features if g]
    if not geoms:
        return np.zeros(shape, bool)
    return rfeat.rasterize(((g, 1) for g in geoms), out_shape=shape,
                           transform=Affine(dx, 0, x0, 0, dy, y0), fill=0, dtype="uint8").astype(bool)


def geolocation_vs_water(field, attrs, water_geoms, factor=6, max_shift_m=75.0, min_cells=40,
                         valid=None, n_permutations=500, seed=0) -> dict:
    """
    The product's geolocation against independently mapped water bodies:
    each water body's window is searched for the shift of best agreement and
    all windows are scored together, then the west and east halves on their
    own (they should agree to a few metres; a disagreement is a rotation or
    a bad window, not a shift). `water_geoms` are GeoJSON geometries in
    lon/lat (polygons of at least ~2 ha).

    Each half carries its window count (n_windows), the result the posting
    cell (cell_m), and split_test whether the halves disagree beyond their
    own scatter (_split_test: random same-size splits of the scored windows,
    p_value), so a disagreement resting on five lakes is not read as a
    shift (decision E22, 7 Oct 2026).
    """
    from insar_pipeline.significance import pixel_metres
    from shapely.geometry import shape as shp
    f = np.asarray(field, dtype=np.float64)
    ny, nx = f.shape
    x0, y0 = float(attrs["X_FIRST"]), float(attrs["Y_FIRST"])
    dx, dy = float(attrs["X_STEP"]), float(attrs["Y_STEP"])
    pm = pixel_metres(attrs)
    cell_m = float(np.sqrt(pm[0] * pm[1]))
    S = int(round(max_shift_m / (cell_m / factor)))
    margin = int(np.ceil(max_shift_m / cell_m)) + 3
    wins = []
    for g in water_geoms:
        try:
            b = shp(g).bounds
        except Exception:                                         # noqa: BLE001
            continue
        c0 = int(np.floor((b[0] - x0) / dx)) - margin; c1 = int(np.ceil((b[2] - x0) / dx)) + margin
        r0 = int(np.floor((b[3] - y0) / dy)) - margin; r1 = int(np.ceil((b[1] - y0) / dy)) + margin
        r0, c0 = max(r0, 0), max(c0, 0); r1, c1 = min(r1, ny), min(c1, nx)
        if r1 - r0 < 5 or c1 - c0 < 5:
            continue
        wins.append((r0, r1, c0, c1))
    if not wins:
        return {"available": False, "reason": "no mapped water body inside the product grid", "n_water_bodies": 0}
    fields, fines, oks, centres = [], [], [], []
    for r0, r1, c0, c1 in wins:
        sub = f[r0:r1, c0:c1]
        ok = np.isfinite(sub) if valid is None else (valid[r0:r1, c0:c1] & np.isfinite(sub))
        if ok.sum() < min_cells:
            continue
        fdx, fdy = dx / factor, dy / factor
        wf = rasterise_lonlat(water_geoms, x0 + c0 * dx - S * fdx, y0 + r0 * dy - S * fdy, fdx, fdy,
                              ((r1 - r0) * factor + 2 * S, (c1 - c0) * factor + 2 * S))
        frac0 = wf[S:S + (r1 - r0) * factor, S:S + (c1 - c0) * factor].reshape(
            r1 - r0, factor, c1 - c0, factor).mean(axis=(1, 3))
        # a window needs both water and land inside it to have an edge
        if not (0.05 < frac0[ok].mean() < 0.95):
            continue
        # score only the band within the search distance of the mapped
        # shore: a cell deeper inside the lake or the field reads the same
        # water fraction at every trial shift and only dilutes the score
        # (it also made a 300-cell lake window take half a minute)
        from scipy.ndimage import distance_transform_edt
        wet = frac0 > 0.5
        band = ((distance_transform_edt(~wet, sampling=pm) <= max_shift_m + cell_m)
                & (distance_transform_edt(wet, sampling=pm) <= max_shift_m + cell_m))
        ok &= band
        if ok.sum() < min_cells:
            continue
        fields.append(sub); fines.append(wf); oks.append(ok); centres.append((c0 + c1) / 2)
    if not fields:
        return {"available": False, "reason": "no water body with both water and coherent land in its window",
                "n_water_bodies": 0}

    scored = [_window_scores(fields[i], fines[i], factor, cell_m, max_shift_m, oks[i]) for i in range(len(fields))]

    def joint(idx):
        """one shift search over several windows: the windows' correlation surfaces averaged"""
        sc = [scored[i] for i in idx if scored[i] is not None]
        if not sc:
            return None
        tot = np.sum(sc, axis=0)
        k = np.unravel_index(np.argmax(tot), tot.shape)
        sy, sx = k[0] - S, k[1] - S
        fine_m = cell_m / factor
        score = float(tot[k] / len(sc))
        if score < 0.2:                                           # no edge the coherence follows
            return {"dE_m": None, "dN_m": None, "offset_m": None, "score": round(score, 3)}
        return {"dE_m": round(-sx * fine_m, 2), "dN_m": round(sy * fine_m, 2),
                "offset_m": round(float(np.hypot(sx, sy)) * fine_m, 2), "score": round(score, 3)}
    allw = joint(range(len(fields)))
    mid = nx / 2
    scored_idx = [i for i in range(len(fields)) if scored[i] is not None]
    west_idx = [i for i in scored_idx if centres[i] < mid]
    east_idx = [i for i in scored_idx if centres[i] >= mid]
    west = joint(west_idx)
    east = joint(east_idx)
    halves = None
    if west and east and west["dE_m"] is not None and east["dE_m"] is not None:
        halves = round(float(np.hypot(west["dE_m"] - east["dE_m"], west["dN_m"] - east["dN_m"])), 2)
    per = {"all": allw, "west": west, "east": east}
    for k, idx in (("west", west_idx), ("east", east_idx)):
        if per[k] is not None:
            per[k] = dict(per[k], n_windows=len(idx))
    split = (_split_test(joint, scored_idx, len(west_idx), halves, n_permutations, seed)
             if halves is not None else None)
    return {"available": bool(allw and allw["offset_m"] is not None), "n_water_bodies": len(fields),
            "step_m": round(cell_m / factor, 2), "cell_m": round(cell_m, 2),
            "search_m": max_shift_m, **(allw or {}), "halves": per, "west_east_disagreement_m": halves,
            "split_test": split}


def _split_test(joint, idx, n_first, observed, n_permutations=500, seed=0) -> dict:
    """
    Do the west and east halves disagree by more than their own scatter?
    The already-scored windows are split at random into groups of the same
    sizes `n_permutations` times; p = (1 + splits disagreeing at least as
    much as the real halves) / (1 + usable splits). A split where either
    group has no edge the coherence follows (joint score < 0.2) is not
    usable. Deterministic (fixed seed). Measured: Site A's 41/12 split
    13.5 m apart, p about 0.007 (a real local offset); Site B's 5/41
    split 21 m apart, p about 0.3 (five lakes: noise).
    """
    rng = np.random.default_rng(seed)
    idx = np.asarray(idx)
    ge = used = 0
    for _ in range(int(n_permutations)):
        p = rng.permutation(idx)
        a, b = joint(list(p[:n_first])), joint(list(p[n_first:]))
        if not a or not b or a["dE_m"] is None or b["dE_m"] is None:
            continue
        used += 1
        if np.hypot(a["dE_m"] - b["dE_m"], a["dN_m"] - b["dN_m"]) >= float(observed) - 1e-6:
            ge += 1
    return {"p_value": round((1 + ge) / (1 + used), 4) if used else None, "n_permutations": used,
            "method": (f"{used} random splits of the {idx.size} scored windows into groups of {n_first} and "
                       f"{idx.size - n_first}; p = (1 + splits at least {float(observed):.1f} m apart) / (1 + splits)")}


def _window_scores(sub, wf, factor, cell_m, max_shift_m, ok):
    """correlation of -(water fraction) with coherence over the `ok` cells at
    every trial shift: a (2S+1) x (2S+1) surface, S search steps each way"""
    S = int(round(max_shift_m / (cell_m / factor)))
    fv = sub[ok]
    if fv.size < 20 or fv.std() == 0:
        return None
    fz = (fv - fv.mean()) / fv.std()
    W = np.asarray(wf, dtype=np.float64)
    # integral image: a cell's water fraction at any fine offset is four lookups
    I = np.zeros((W.shape[0] + 1, W.shape[1] + 1)); I[1:, 1:] = W.cumsum(0).cumsum(1)
    rows, cols = np.nonzero(ok)
    rr, cc = rows * factor + S, cols * factor + S
    out = np.zeros((2 * S + 1, 2 * S + 1))
    for sy in range(-S, S + 1):
        r0 = rr + sy
        for sx in range(-S, S + 1):
            c0 = cc + sx
            w = (I[r0 + factor, c0 + factor] - I[r0, c0 + factor] - I[r0 + factor, c0] + I[r0, c0]) / factor ** 2
            sd = w.std()
            out[sy + S, sx + S] = float(-(fz * (w - w.mean())).mean() / sd) if sd > 0 else 0.0
    return out


OVERPASS_URLS = ("https://overpass-api.de/api/interpreter",
                 "https://overpass.kumi.systems/api/interpreter",
                 "https://overpass.private.coffee/api/interpreter")


def fetch_osm_water(bbox, dest, min_area_ha=2.0, timeout=90):
    """
    OpenStreetMap water polygons (natural=water, and water=* / landuse=
    reservoir) inside bbox = (W, S, E, N) from the Overpass API, at least
    `min_area_ha`, written to `dest` as GeoJSON. Returns the path, or None
    when the service cannot be reached (never raises: the geolocation check
    is then recorded as not run, and the run goes on).
    """
    import json
    import urllib.parse
    import urllib.request
    from shapely.geometry import LineString, Polygon, mapping
    from shapely.ops import polygonize, unary_union
    w, s, e, n = bbox
    q = (f"[out:json][timeout:{int(timeout) - 10}];("
         f'way["natural"="water"]({s},{w},{n},{e});relation["natural"="water"]({s},{w},{n},{e});'
         f'way["landuse"="reservoir"]({s},{w},{n},{e});'
         ");out geom;")
    data, err = None, None
    # the main instance answers 504 when busy (5 Oct 2026, three tries in a
    # row); the public mirrors serve the same database
    for url in OVERPASS_URLS:
        try:
            req = urllib.request.Request(url, data=urllib.parse.urlencode({"data": q}).encode(),
                                         headers={"User-Agent": "insar_pipeline QA (geolocation check)"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                data = json.loads(r.read().decode())
            break
        except Exception as exc:                                  # noqa: BLE001
            err = f"{url}: {exc}"
    if data is None:
        log.warning("OSM water could not be fetched for the geolocation check: %s", err)
        return None
    polys = []
    for el in data.get("elements", []):
        if el.get("type") == "way" and el.get("geometry"):
            pts = [(p["lon"], p["lat"]) for p in el["geometry"]]
            if len(pts) >= 4 and pts[0] == pts[-1]:
                polys.append(Polygon(pts))
        elif el.get("type") == "relation":
            outer = [LineString([(p["lon"], p["lat"]) for p in m["geometry"]])
                     for m in el.get("members", []) if m.get("role") == "outer" and m.get("geometry")]
            inner = [LineString([(p["lon"], p["lat"]) for p in m["geometry"]])
                     for m in el.get("members", []) if m.get("role") == "inner" and m.get("geometry")]
            o = list(polygonize(unary_union(outer))) if outer else []
            i_ = list(polygonize(unary_union(inner))) if inner else []
            for p in o:
                for h in i_:
                    if p.contains(h):
                        p = p.difference(h)
                polys.append(p)
    lat0 = (s + n) / 2
    ha = 111_320.0 * np.cos(np.radians(lat0)) * 110_900.0 / 1e4
    feats = [{"type": "Feature", "properties": {"source": "OpenStreetMap contributors (ODbL)"},
              "geometry": mapping(p)} for p in polys if p.is_valid and p.area * ha >= min_area_ha]
    Path(dest).parent.mkdir(parents=True, exist_ok=True)
    Path(dest).write_text(json.dumps({"type": "FeatureCollection", "features": feats}))
    log.info("OSM water: %d polygons of at least %.0f ha -> %s", len(feats), min_area_ha, dest)
    return Path(dest)
