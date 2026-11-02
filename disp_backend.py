"""OPERA DISP-S1 backend: use JPL's ready-made displacement time series.

The lightest backend by far. OPERA already ran phase linking on the
full-resolution CSLC stack, unwrapped it, inverted the time series and
applied corrections - so there is no ordering step, no credits, no
interferogram formation and no MintPy inversion. The work here is
searching, subsetting, stitching reference epochs, and writing the result
in MintPy's format so the existing deliverables stage can consume it.

Coverage: OPERA's published scope is North America, mid-2016 onward, 30 m
posting - but frames exist elsewhere: F39360 (descending track 147) covers
Sydney and Appin, NSW, with F31577 (Melbourne), F39355 (Brisbane) and F43655
(Perth) beside it (review_2026-10-08/sensors/notes.md). Nothing here assumes
a region or a hemisphere: the frame is chosen from the archive's own track,
direction and footprint (choose_frame), the CRS is the granule's own
spatial_epsg (_granule_epsg; there is no fallback zone), and the AOI window
and reference point are projected into it (checked on a synthetic
southern-hemisphere EPSG:32756 frame, tests/test_p0_proj.py). The CONUS-only
parts are elsewhere and say so: HRRR in tropo_disp.check_model_range.

NOTE ON A BETTER PATH: OPERA publishes `opera-utils`, whose
`opera_utils.disp.rebase_reference` performs the reference-epoch
stitching below, and whose `mintpy` export writes a MintPy timeseries
directly. OPERA's own Cal/Val repo (OPERA-Cal-Val/calval-DISP) has
`run2_prep_mintpy_opera.py` for the same purpose. Both are maintained by
the product's authors and are preferable to this hand-rolled reader for
anything going to a client - this module exists because it slots into the
rest of the pipeline unchanged, not because it is more correct.

Three things this module handles that will silently corrupt a result if
ignored:

  * SIGN. The `displacement` layer carries the attribute "Positive values
    indicate apparent motion towards the platform", which is MintPy's
    convention for timeseries.h5, so values pass through unchanged and
    subsidence reads NEGATIVE in both. This was verified by reading the
    layer attribute directly out of a granule, not from documentation.
    An earlier version negated on read, which inverted the result.

    A previous investigation concluded the opposite - that a known
    subsiding site read positive without the negation. That conclusion
    was an artefact of the masking cascade described below: the only
    pixels surviving the mask sat in one small non-subsiding corner of
    the AOI, so the reported velocity range was positive while the
    unmasked velocity field was correctly negative over the subsidence
    bowl. Fix the mask before questioning the sign.

  * REFERENCE EPOCHS, and the carry that stitches them. DISP-S1 resets
    its reference date roughly every six months. Each product measures
    displacement since its own reference, so segments must be accumulated
    across the boundaries or the series is a sawtooth.

    The accumulation is the single most fragile step in this module. The
    offset carried across a boundary is an ARRAY, and any pixel that is
    NaN in that array is NaN for the rest of the record - NaN + x = NaN.
    Masking per epoch and then carrying the masked frame therefore
    ratchets coverage down at every boundary: with 16 boundaries over a
    decade, an innocuous 15% per-granule mask compounds to roughly 1%
    survival. That is not a hypothetical - it is what produced the 0.8%
    coverage this module used to report. The carry below is built from
    the last epoch at which each pixel was individually valid, so a pixel
    only dies when it is bad everywhere.

  * UNWRAPPING ERRORS. Granules contain patches SNAPHU could not unwrap
    reliably, appearing as isolated jumps of several 2*pi cycles that
    revert at the next epoch. `connected_component_labels` flags these
    with label 0. They are dropped per epoch - but see the carry note
    above for why that dropping must not feed the accumulation.

Full frames run to several GB, so granules are opened remotely and
subset to the AOI rather than downloaded whole.
"""
import contextlib
import logging
import signal
import threading
import time
import warnings
import re
from datetime import datetime
from pathlib import Path

import numpy as np

from .config import PipelineConfig

log = logging.getLogger(__name__)

COLLECTION = "OPERA_L3_DISP-S1_V1"

# Layers worth carrying through. `displacement` is the unfiltered LOS
# series; `short_wavelength_displacement` has signals above ~30 km removed
# for visualisation, which also removes real large-scale subsidence, so it
# is not the layer to measure from.
DISPLACEMENT_LAYER = "displacement"

# DISP-S1 ships a real `temporal_coherence` layer. Earlier versions of this
# module used `estimated_phase_quality` as a stand-in for it, which meant
# `min_temporal_coherence` in the config did not mean the same thing on
# this backend as it does on the MintPy backends - the same threshold was
# being applied to two different quantities, which is part of why the two
# backends disagreed. Prefer the real layer; fall back only if absent.
COHERENCE_LAYER = "temporal_coherence"
QUALITY_LAYER = "estimated_phase_quality"
MASK_LAYER = "recommended_mask"

# `water_mask` is the honest denominator for a coverage figure: land that
# could in principle be measured, as opposed to the whole AOI rectangle.
# It is a KEEP mask - 1 means land, 0 means water.
WATER_LAYER = "water_mask"

# OPERA's own identification of point-like targets. Worth carrying because
# it is the closest open analogue to a commercial PS vendor's point set -
# but do not expect it to close a point-density gap over farmland, where
# it selects well under 1% of the grid. It earns its keep in built-up
# areas and around infrastructure.
PS_LAYER = "persistent_scatterer_mask"

# SNAPHU connected components. Label 0 means "masked / not reliably
# unwrapped", so those pixels are dropped PER EPOCH - a granule can be
# fine everywhere except a patch where unwrapping failed, and that patch
# is what produces isolated multi-cycle jumps that revert the next epoch.
# One 2*pi cycle is about 2.8 cm in C-band line of sight, so a spike of
# several inches is a handful of cycles.
#
# These drops must never reach the reference-epoch carry - see the module
# docstring.
CONNCOMP_LAYER = "connected_component_labels"

# dolphin's L1 inversion of the nearest-3 network leaves a per-date residual
# (radians). See assemble_stack for how it is used.
RESIDUAL_LAYER = "timeseries_inversion_residuals"

# Corrections DISP-S1 supplies but does NOT apply. Both layers say so in
# their own description: "Subtract from /displacement to apply as a
# correction", and the product metadata records
# `ceos_atmospheric_phase_correction = None` and
# `ceos_ionospheric_phase_correction = None`. Reading `displacement` alone
# therefore leaves solid-earth tide and ionospheric delay in the result.
#
# They live in the /corrections group, which is not visible in the root
# dataset - which is how they came to be overlooked.
#
# NOTE what is NOT in this list, because DISP-S1 does not provide it:
# TROPOSPHERE. That is the dominant error source at these scales, and the
# same one that made the Philadelphia stack uninterpretable. A DISP-S1
# time series is uncorrected for it.
CORRECTION_LAYERS = ("solid_earth_tide", "ionospheric_delay")

# Quality layers averaged over the stack rather than taken from whichever
# granule happened to be read first. A single 12-day pair at the start of
# a decade is not representative of the decade.
_MEAN_LAYERS = (COHERENCE_LAYER, QUALITY_LAYER, MASK_LAYER, WATER_LAYER, PS_LAYER)

# Read the quality layers from every Nth granule. water_mask and
# persistent_scatterer_mask barely move over a frame's lifetime, and a
# mean coherence over ~30 samples spread across a decade is indis-
# tinguishable from one over every epoch - at a fraction of the transfer.
_QUALITY_STRIDE = 8


def _dates(fname: str) -> tuple[str, str]:
    m = re.findall(r"_(\d{8})T\d{6}Z", fname)
    return (m[0], m[1]) if len(m) >= 2 else ("", "")


def _stamps(fname: str) -> list[str]:
    """Every YYYYmmddTHHMMSS stamp in a granule name: reference, secondary
    and (third) the production time - OPERA_L3_DISP-S1_IW_F04861_VV_
    20250704T234804Z_20251231T234758Z_v1.0_20260310T150018Z.nc."""
    return re.findall(r"_(\d{8}T\d{6})Z", fname)


def _version(fname: str) -> str:
    m = re.search(r"_v(\d+)\.(\d+)_", fname)
    return f"{m.group(1)}.{m.group(2)}" if m else ""


def _frame_of(fname: str) -> str:
    m = re.search(r"_F(\d+)_", fname)
    return m.group(1) if m else ""


def acq_seconds(fname: str) -> int | None:
    """Secondary acquisition time of day (UTC seconds) from the granule name.

    The name carries it for every granule, so the ERA5 hour no longer depends
    on reading /identification over the network - which a fully cached re-run
    does not otherwise need, and which, when it failed, used to leave the
    chain's CENTER_LINE_UTC on a hard-coded 23:48 (right only for ascending
    passes over the Midwest)."""
    s = _stamps(fname)
    if len(s) < 2:
        return None
    hh, mm, ss = int(s[1][9:11]), int(s[1][11:13]), int(s[1][13:15])
    return hh * 3600 + mm * 60 + ss


def dedupe(results) -> list:
    """
    One granule per (frame, reference date, secondary date), the latest
    production of the newest minor version; refuses to mix major versions.

    WHY. ASF lists every production of a pair. All 254 F04861 granules are
    v1.0 today with no duplicate pair, but the day OPERA publishes a v1.1
    reprocessing beside v1.0, two granules share a secondary date and
    timeseries.h5 gets that epoch twice - MintPy rejects it or double-weights
    it. A major version may rename or redefine layers (misclosure and
    coherence would then silently switch off), so mixing majors is refused.
    """
    from collections import Counter
    if not results:
        return results
    versions = Counter(_version(p.properties["fileName"]) for p in results)
    majors = {v.split(".")[0] for v in versions if v}
    if len(majors) > 1:
        raise ValueError(
            "DISP-S1 granules over this AOI mix major product versions "
            f"({', '.join(f'v{v}: {n}' for v, n in sorted(versions.items()))}). "
            "Layers can change meaning between major versions, so they are not "
            "stitched together; restrict the dates or wait for the reprocessing "
            "to complete.")
    best: dict[tuple, object] = {}
    for p in results:
        f = p.properties["fileName"]
        ref, sec = _dates(f)
        key = (_frame_of(f), ref, sec)
        st = _stamps(f)
        rank = (tuple(int(x) for x in _version(f).split(".")) if _version(f) else (0, 0),
                st[2] if len(st) >= 3 else "")
        if key not in best or rank > best[key][0]:
            best[key] = (rank, p)
    kept = [v[1] for v in best.values()]
    if len(versions) > 1 or len(kept) < len(results):
        log.warning("DISP-S1 versions over this AOI: %s; %d duplicate production(s) of the "
                    "same (frame, reference, secondary) dropped, keeping the newest",
                    ", ".join(f"v{v}: {n}" for v, n in sorted(versions.items())),
                    len(results) - len(kept))
    return kept


def _bbox_overlap(geometry: dict | None, bbox) -> float | None:
    """Share of the processing box (W, S, E, N) inside a granule footprint."""
    if not geometry:
        return None
    try:
        from shapely.geometry import box, shape
        b = box(*bbox)
        return float(shape(geometry).intersection(b).area / b.area) if b.area else None
    except Exception:  # noqa: BLE001
        return None


def frame_summary(results, bbox) -> list[dict]:
    """
    Per frame: track, flight direction, granule count, date span and the share
    of the processing box (padded, as processed) its footprint covers.

    Choosing a frame by granule count alone - what the dashboard and builder
    did - can pick the other track's frame (on a Site D-sized box F04862,
    track 19 asc, has 224 granules and F32248, track 121 asc, 134) and then
    label the result with the configured track, or pick a frame that covers
    only part of the box. This is what a choice should be made from.
    """
    out: dict[str, dict] = {}
    for p in results:
        pr = p.properties
        f = _frame_of(pr["fileName"])
        d = out.setdefault(f, {"frame": f, "granules": 0, "tracks": set(), "directions": set(),
                               "first": None, "last": None, "overlap_fraction": None})
        d["granules"] += 1
        if pr.get("pathNumber") is not None:
            d["tracks"].add(int(pr["pathNumber"]))
        if pr.get("flightDirection"):
            d["directions"].add(str(pr["flightDirection"]).lower())
        sec = _dates(pr["fileName"])[1]
        d["first"] = min(d["first"] or sec, sec)
        d["last"] = max(d["last"] or sec, sec)
        if d["overlap_fraction"] is None:
            d["overlap_fraction"] = _bbox_overlap(getattr(p, "geometry", None), bbox)
    rows = []
    for d in out.values():
        tracks = sorted(d.pop("tracks"))
        d["track"] = tracks[0] if tracks else None
        if len(tracks) > 1:
            d["tracks"] = tracks
        dirs = sorted(d.pop("directions"))
        d["direction"] = dirs[0] if len(dirs) == 1 else ("/".join(dirs) or None)
        if d["overlap_fraction"] is not None:
            d["overlap_fraction"] = round(d["overlap_fraction"], 4)
        rows.append(d)
    return sorted(rows, key=lambda r: (-(r["overlap_fraction"] or 0.0), -r["granules"]))


# A frame must cover at least this share of the processing box, or part of
# the AOI comes back empty with nothing to say why.
MIN_FRAME_OVERLAP = 0.95


class NoFrameOnTrack(ValueError):
    """No DISP-S1 frame over the AOI on the configured track and direction:
    there is no product here, which the cross-check records as a result. Any
    other ValueError from the search (mixed major versions, a granule off the
    track) is a failure and is not this."""


def choose_frame(results, cfg: PipelineConfig) -> tuple[str, dict, list[dict]]:
    """
    The frame to process: the configured one (checked against the configured
    track and direction), else the one on the configured track and direction
    that covers most of the processing box. Raises when nothing matches,
    rather than processing one track and labelling it with another.
    """
    rows = frame_summary(results, cfg.bbox)
    if not rows:
        raise NoFrameOnTrack("no DISP-S1 frame in the search results")
    want_dir = str(getattr(cfg, "orbit_direction", "") or "").lower()
    want_trk = getattr(cfg, "track", None)

    def matches(r) -> bool:
        ok_t = want_trk is None or r["track"] is None or int(r["track"]) == int(want_trk)
        ok_d = not want_dir or r["direction"] is None or r["direction"] == want_dir
        return ok_t and ok_d

    def describe(r) -> str:
        return (f"F{r['frame']} (track {r['track']}, {r['direction']}, {r['granules']} granules, "
                f"{'?' if r['overlap_fraction'] is None else round(100 * r['overlap_fraction'])}% of the box)")

    if getattr(cfg, "disp_frame_id", ""):
        want = cfg.disp_frame_id.strip().lstrip("Ff").zfill(5)
        r = next((x for x in rows if x["frame"] == want), None)
        if r is None:
            raise ValueError(
                f"disp_frame_id {cfg.disp_frame_id!r} matches no granule over "
                f"this AOI. Frames available: {'; '.join(describe(x) for x in rows)}.")
        if not matches(r):
            raise ValueError(
                f"disp_frame_id F{want} is track {r['track']} {r['direction']}, but the config says "
                f"track {want_trk} {want_dir}. The products would be labelled with the wrong track "
                f"and the weather model sampled at the wrong hour. Fix track/orbit_direction or "
                f"disp_frame_id. Frames here: {'; '.join(describe(x) for x in rows)}.")
        chosen = r
    else:
        cand = [r for r in rows if matches(r)]
        if not cand:
            raise NoFrameOnTrack(
                f"no DISP-S1 frame over this AOI is on track {want_trk} {want_dir}. Frames here: "
                f"{'; '.join(describe(x) for x in rows)}. Set track/orbit_direction (or "
                f"disp_frame_id) to one of them.")
        chosen = cand[0]                 # rows are sorted by overlap, then count
        if len(rows) > 1:
            log.info("DISP-S1 frames over this AOI: %s. Using F%s: on the configured track and "
                     "direction, covering the most of the box.", "; ".join(describe(x) for x in rows),
                     chosen["frame"])
    ov = chosen["overlap_fraction"]
    if ov is not None and ov < MIN_FRAME_OVERLAP:
        log.warning("Frame F%s covers only %.0f%% of the processing box; the rest of the AOI "
                    "will have no DISP-S1 data.", chosen["frame"], 100 * ov)
    return chosen["frame"], chosen, rows


def search(cfg: PipelineConfig):
    """Find DISP-S1 granules covering the AOI and date range: one frame,
    on the configured track and direction, one production per pair."""
    import asf_search as asf

    results = list(asf.search(
        shortName=[COLLECTION],
        intersectsWith=cfg.aoi_wkt,
        start=cfg.start_date,
        end=cfg.end_date,
        maxResults=5000,
    ))
    if not results:
        log.warning("No DISP-S1 granules found over this AOI and date range. OPERA's "
                    "published scope is North America from mid-2016, with some frames "
                    "elsewhere (e.g. F39360 over Sydney, to 2024-11-15); check the AOI "
                    "and dates.")
        return []
    return _select(cfg, results)


def _select(cfg: PipelineConfig, results) -> list:
    """search() after the archive query: frame choice, track check, de-dup.

    One frame only, always. build_stack() locks onto the grid of whichever
    granule it reads first and skips the others as shape mismatches, so a
    second frame was never stitched - it was silently discarded, and which
    frame won depended on date order rather than on coverage."""
    frame, info, rows = choose_frame(results, cfg)
    kept = [p for p in results if _frame_of(p.properties["fileName"]) == frame]
    if len(rows) > 1:
        log.info("Frame %s: keeping %d of %d granules (%s discarded)", frame, len(kept),
                 len(results), ", ".join(f"F{r['frame']}" for r in rows if r["frame"] != frame))
    # A granule of the chosen frame on another track would be a different
    # geometry under the same frame id - refuse rather than stitch it.
    want_trk = getattr(cfg, "track", None)
    if want_trk is not None:
        off = [p.properties["fileName"] for p in kept
               if p.properties.get("pathNumber") is not None
               and int(p.properties["pathNumber"]) != int(want_trk)]
        if off:
            raise ValueError(f"{len(off)} granule(s) of F{frame} are not on track {want_trk}: "
                             f"{', '.join(o[:60] for o in off[:3])}")
    results = dedupe(kept)
    results.sort(key=lambda p: _dates(p.properties["fileName"])[1])
    refs = sorted({_dates(p.properties["fileName"])[0] for p in results})
    log.info("%d DISP-S1 granules, %d reference epochs, %s to %s",
             len(results), len(refs),
             _dates(results[0].properties["fileName"])[1],
             _dates(results[-1].properties["fileName"])[1])
    return results


def subsample(results, max_per_year: int):
    """
    Thin the granule list while preserving stitching accuracy.

    Reference-epoch boundaries are kept unconditionally. DISP-S1 restarts
    each segment from zero, and the offset carried across a boundary is
    read from the epoch just before it - so dropping that epoch shifts
    everything after it by however much the ground moved in the gap.
    """
    if not max_per_year or max_per_year <= 0:
        return results

    # The last granule of each reference epoch anchors the next segment.
    anchors = set()
    prev_ref, prev_name = None, None
    for p in results:
        ref = _dates(p.properties["fileName"])[0]
        if prev_ref is not None and ref != prev_ref:
            anchors.add(prev_name)
        prev_ref, prev_name = ref, p.properties["fileName"]
    anchors.add(results[0].properties["fileName"])
    anchors.add(results[-1].properties["fileName"])

    by_year: dict[str, list] = {}
    for p in results:
        by_year.setdefault(_dates(p.properties["fileName"])[1][:4], []).append(p)

    kept = []
    for year in sorted(by_year):
        items = by_year[year]
        forced = [p for p in items if p.properties["fileName"] in anchors]
        room = max(0, max_per_year - len(forced))
        rest = [p for p in items if p.properties["fileName"] not in anchors]
        if room and rest:
            idx = np.linspace(0, len(rest) - 1, room).round().astype(int)
            forced += [rest[i] for i in sorted(set(idx))]
        kept += forced

    kept.sort(key=lambda p: _dates(p.properties["fileName"])[1])
    log.info("Subsampled %d granules to %d (max %d per year, %d boundary "
             "anchors kept)", len(results), len(kept), max_per_year,
             len(anchors))
    return kept


def _data_url(product) -> str:
    """
    The URL that actually serves this granule.

    Granules list several locations. The cloud path on
    cumulus.asf.earthdatacloud.nasa.gov is preferred when present. Older
    granules list only datapool.asf.alaska.edu, which does serve the file
    with basic auth from ~/.netrc - an earlier note here said it always
    401s, which is not the case, and treating it as unusable would skip
    most of the archive.
    """
    candidates = [e.get("URL", "") for e in product.umm.get("RelatedUrls", [])
                  if e.get("Type") == "GET DATA" and e.get("URL", "").endswith(".nc")]
    for url in candidates:
        if "earthdatacloud" in url:
            return url
    for url in candidates:
        if url.startswith("http"):
            return url
    return ""


_SESSION = {"auth": None}


def _credentials() -> tuple[str, str]:
    """Earthdata username and password, from ~/.netrc or a prompt."""
    if _SESSION["auth"]:
        return _SESSION["auth"]
    import netrc
    try:
        user, _, pw = netrc.netrc().authenticators("urs.earthdata.nasa.gov")
        log.info("Using Earthdata credentials from ~/.netrc")
    except Exception:  # noqa: BLE001
        from .data_acquisition import get_credentials
        user, pw = get_credentials()
    _SESSION["auth"] = (user, pw)
    return user, pw


def _open_remote(url: str, group: str | None = None):
    """
    Open a granule over HTTP without downloading it whole.

    `group` opens one of the granule's netCDF groups instead of the root -
    `identification` and `metadata` carry the geometry and the processing
    parameters, and neither is visible in the default root view.

    Uses fsspec with basic auth rather than earthaccess: earthaccess
    resolves these granules to the legacy datapool host, which 401s.
    Reading a window out of a ~440 MB file beats downloading all of it.
    """
    import fsspec
    import aiohttp
    import xarray as xr

    user, pw = _credentials()
    # WHY THE EXPLICIT TIMEOUT. fsspec passes client_kwargs straight to
    # aiohttp.ClientSession and adds nothing of its own, and aiohttp's
    # default leaves sock_read=None - no limit on how long a connection may
    # sit idle mid-response. CloudFront half-closes an idle keepalive
    # connection under load, aiohttp hands that dead socket to the next
    # request, and nothing ever times out: one run stalled 6 hours at
    # granule 192/254 with four sockets in CLOSE-WAIT and the process at
    # 0.03% CPU. sock_read is the one that catches that case.
    fs = fsspec.filesystem(
        "https",
        client_kwargs={
            "auth": aiohttp.BasicAuth(user, pw),
            "timeout": aiohttp.ClientTimeout(
                total=None,        # a 440 MB window read can legitimately be long
                sock_connect=30,   # but connecting must not be
                sock_read=120,     # and a live transfer must keep producing bytes
            ),
        },
    )
    fh = fs.open(url, "rb")
    return xr.open_dataset(fh, engine="h5netcdf", group=group)


def _decimate(sub, cfg):
    """
    Keep every Nth pixel, for a wide coarse run.

    Subsampling, not averaging - deliberately. The only use for this is
    estimating a long-wavelength ramp against GNSS, which needs width
    rather than detail, and averaging would cost read time for nothing.
    It is the wrong tool for deformation: skipped pixels leave
    single-pixel noise intact while undersampling the signal.
    """
    d = int(getattr(cfg, "disp_decimate", 1) or 1)
    if d > 1:
        sub = sub.isel(x=slice(None, None, d), y=slice(None, None, d))
    return sub


def _granule_epsg(ds) -> int:
    """The granule's projected CRS. There used to be a fallback to EPSG 32616
    - Site A's UTM zone - which anywhere outside zone 16 would have put
    the AOI window on the wrong ground without an error. A granule that does
    not state its CRS is refused instead."""
    sr = ds["spatial_ref"].attrs if "spatial_ref" in getattr(ds, "variables", {}) else {}
    for src in (sr, ds.attrs):
        if src.get("spatial_epsg") is not None:
            return int(src["spatial_epsg"])
    for src in (sr, ds.attrs):
        wkt = src.get("crs_wkt") or src.get("spatial_ref")
        if wkt:
            from pyproj import CRS
            code = CRS.from_wkt(str(wkt)).to_epsg()
            if code:
                return int(code)
    raise RuntimeError("DISP-S1 granule states no spatial_epsg or crs_wkt; cannot place the AOI window")


def _subset_window(ds, cfg: PipelineConfig):
    """Pixel window covering the AOI, in the granule's projected CRS."""
    from pyproj import Transformer

    epsg = _granule_epsg(ds)
    tr = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    min_lon, min_lat, max_lon, max_lat = cfg.bbox
    xs, ys = [], []
    for lon in (min_lon, max_lon):
        for lat in (min_lat, max_lat):
            x, y = tr.transform(lon, lat)
            xs.append(x); ys.append(y)
    return min(xs), max(xs), min(ys), max(ys), epsg


# Sentinel-1 IW spans roughly 29-46 degrees of incidence; the middle of
# the swath is near 37.5. Used only when the granule does not state the
# angle. It is a stated assumption rather than a null because a null
# silently turns any LOS-to-vertical projection downstream into a no-op.
# Cross-check: the co-located HyP3 run on this same track computes 37.46
# from its own orbit geometry.
NOMINAL_IW_INCIDENCE_DEG = 37.5


def _geometry_from_granule(url: str) -> tuple[float | None, str | None]:
    """
    Mid-swath incidence angle and acquisition time, from /identification.

    DISP-S1 states `near_range_incidence_angle` and
    `far_range_incidence_angle` there - 30.7 and 45.9 degrees on the
    Site A frame - along with the zero-Doppler acquisition times.
    None of it is visible in the root dataset, which is why this looked
    absent and a nominal value was being assumed.

    The midpoint is a scene constant, so it is right in the middle of the
    swath and wrong by up to ~7 degrees at the edges. That is a ~10% error
    on any line-of-sight to vertical projection. For per-pixel geometry,
    use the DISP-S1-STATIC product, whose URL every granule gives in
    `identification/static_layers_data_access`.

    The acquisition time matters for the tropospheric correction: the
    weather model must be sampled at the moment the radar looked, and
    Sentinel-1 is sun-synchronous, so one frame is always acquired within
    a few seconds of the same UTC time.
    """
    try:
        ident = _open_remote(url, group="identification")
    except Exception as exc:  # noqa: BLE001
        log.warning("Could not read /identification (%s)", exc)
        return None, None
    inc = acq = None
    try:
        near = float(np.asarray(ident["near_range_incidence_angle"].values).ravel()[0])
        far = float(np.asarray(ident["far_range_incidence_angle"].values).ravel()[0])
        if 10.0 < near < 80.0 and 10.0 < far < 80.0:
            inc = (near + far) / 2
            log.info("Incidence angle from the granule: %.1f deg near range, "
                     "%.1f far range - using the midpoint %.1f",
                     near, far, inc)
    except (KeyError, TypeError, ValueError, IndexError):
        pass
    try:
        raw = np.asarray(ident["secondary_zero_doppler_start_time"].values).ravel()[0]
        if isinstance(raw, bytes):
            raw = raw.decode()
        acq = str(raw)[11:19]
        datetime.strptime(acq, "%H:%M:%S")
        log.info("Acquisition time of day for this frame: %s UTC", acq)
    except (KeyError, TypeError, ValueError, IndexError):
        acq = None
    finally:
        ident.close()
    return inc, acq


def _whole_cycle_step(ts: np.ndarray, i: int, med: np.ndarray, scatter: np.ndarray,
                      cycle: float, tolerance: float, n_sigma: float):
    """Is the step INTO epoch i (ts[i] - ts[i-1]) a whole number of cycles on
    top of the pixel's usual step, and large against its own scatter?
    Returns (hit, k cycles, finite)."""
    anomaly = (ts[i] - ts[i - 1]) - med
    c = anomaly / cycle
    k = np.round(c)
    fin = np.isfinite(c)
    hit = ((np.abs(k) >= 1)
           & (np.abs(c - k) <= tolerance)
           & (np.abs(anomaly) >= n_sigma * np.maximum(scatter, 1e-12))
           & fin)
    return hit, k, fin


def _step_scatter(ts: np.ndarray):
    """Per-pixel robust median and scatter of the epoch-to-epoch steps."""
    d = np.diff(ts, axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        med = np.nanmedian(d, axis=0)
        mad = np.nanmedian(np.abs(d - med), axis=0)
    return med, 1.4826 * mad


def _anchor_hits(ts: np.ndarray, a: int, med, scatter, cycle, tolerance, n_sigma):
    """Whole-cycle steps INTO anchor a that the carry made permanent.

    Two look-alikes are left out, because in both the error reverts and
    nothing wrong is carried: a spike in the epoch BEFORE the anchor (the
    step into the anchor is its reversal), and a step into the anchor that
    is undone at the very next epoch (an isolated spike, which
    _flag_unwrap_jumps rejects; subtracting cycles from it onward would
    bake it in)."""
    hit, k, fin = _whole_cycle_step(ts, a, med, scatter, cycle, tolerance, n_sigma)
    rev = np.zeros_like(hit)
    if a >= 2:
        hp, kp, _ = _whole_cycle_step(ts, a - 1, med, scatter, cycle, tolerance, n_sigma)
        rev |= hit & hp & (np.abs(k + kp) <= 0.5)
    if a + 1 < ts.shape[0]:
        hn, kn, _ = _whole_cycle_step(ts, a + 1, med, scatter, cycle, tolerance, n_sigma)
        rev |= hit & hn & (np.abs(k + kn) <= 0.5)
    return hit & ~rev, k, fin, rev


def _repair_boundary_steps(ts: np.ndarray, anchor_idx: list[int],
                           wavelength_m: float, tolerance: float = 0.25,
                           n_sigma: float = 3.0) -> dict:
    """
    Remove whole-cycle steps that the reference-epoch carry baked in.

    `anchor_idx` are the ANCHOR epochs - the last epoch of each segment,
    whose value the carry copies into every later epoch (assemble_stack
    returns them). Only worth doing when `_check_boundary_steps` shows a clear
    EXCESS at anchors over ordinary epochs - that excess is the evidence the
    steps come from the stitching rather than from noise.

    For each qualifying pixel the integer number of cycles is subtracted
    from the anchor onward, because the carry propagated the error to every
    later epoch.

    THE INDEX MATTERS. Until 5 Oct 2026 this tested and repaired at the first
    epoch of the NEW segment (anchor + 1). A carry error is never there, and a
    reverting unwrapping spike in that epoch - an ordinary, harmless error -
    was "repaired" into a permanent one: driven through the production
    assemble_stack, 85 pixels with a +1 cycle spike there ended a whole cycle
    off after the repair. Testing the step into the anchor cannot do that.

    WHEN THIS IS WRONG. If the ground genuinely moved a whole cycle
    between two consecutive epochs, this erases real displacement. One
    C-band cycle is 27.7 mm of line-of-sight motion; on a 12-day stack
    that is a rate of 840 mm/yr. Site A runs about 25 mm/yr, so the
    two are nowhere near each other and the repair is safe. Over active
    longwall mining, which reaches metres per year, it would not be -
    which is why this is opt-in rather than automatic.
    """
    cycle = wavelength_m / 2.0
    n = ts.shape[0]
    aset = sorted({i for i in anchor_idx if 0 < i < n})
    out = {"repaired_pixels": 0, "boundaries": 0, "cycles_removed": 0, "index": "anchor"}
    if not aset or n < 4:
        return out

    med, scatter = _step_scatter(ts)
    for a in aset:
        hit, k, _, _ = _anchor_hits(ts, a, med, scatter, cycle, tolerance, n_sigma)
        if not hit.any():
            continue
        ts[a:] -= np.where(hit, k * cycle, 0.0)[None, :, :]
        out["repaired_pixels"] += int(hit.sum())
        out["cycles_removed"] += int(np.abs(k[hit]).sum())
        out["boundaries"] += 1
    return out


def _check_boundary_datum(ts: np.ndarray, boundary_idx: list[int],
                          dates: list[str], repair: bool = False,
                          anchor_idx: list[int] | None = None) -> dict:
    """
    Find and optionally remove a DATUM SHIFT at a reference-epoch boundary.

    This is a different failure from `_check_boundary_steps`, which hunts
    whole-cycle unwrapping errors. The one caught here is smaller than a
    cycle and therefore invisible to that test, but it moves every epoch
    after it and so corrupts the rate.

    HOW IT HAPPENS. DISP-S1 rebases periodically, and the carry that ties
    the new reference epoch to the old one is estimated from the granules
    around the boundary. When those granules are sparse - a hard winter,
    snow, saturated fields - the carry is estimated from very few pixels
    and lands wrong. Site A, 2025-01-17: the carry was +39.0 mm, the
    largest of sixteen boundaries, computed across epochs where 6-16% of
    pixels reported. Everything afterwards sat about 25 mm low, which
    turned stable farmland 15 km from any deformation into an apparent
    -32.7 mm/yr subsidence and inflated the real bowl by ~3 mm/yr.

    WHAT IS REMOVED, AND WHY NOT THE WHOLE STEP. An earlier version of
    this subtracted the scene-median step outright, on the argument that a
    spatially uniform offset is unmeasurable by InSAR anyway. That was
    wrong in a way worth recording: the step between two CONSECUTIVE
    epochs also contains whatever the scene genuinely did between those
    two dates - seasonal swing, regional motion, atmosphere. Removing it
    at ten boundaries accumulated a spurious +19 mm/yr scene-wide trend
    and pushed the whole map into apparent uplift.

    So only the ANOMALY is removed: the step at the boundary minus the
    step the scene was taking either side of it. That isolates the part
    attributable to a bad carry and leaves the real evolution alone.

    WHICH STEP. The detection and the opt-in repair stay on the step into
    the new segment's first epoch (`boundary_idx`), where the Site A
    2025-01-17 shift was found and measured. A datum error in the ANCHOR
    granule itself would instead sit in the step into the anchor; that step
    is measured too (`anchor_shift_mm` per boundary, `flagged_at_anchor`)
    and reported, but not repaired: the Site A DISP project that
    validated the repair is no longer on disk to check an anchor-side
    repair against, and an unvalidated correction that quietly rewrites a
    decade is the failure this function was written to avoid.

    Even then this is opt-in. The failure it repairs is real, but so is
    the failure mode of the repair, and a detected-and-reported boundary
    that a human looks at beats an automatic correction that quietly
    rewrites a decade.

    Reported per boundary so the magnitudes can be judged rather than
    trusted, and `repair` is separate from detection for the same reason.
    """
    n = ts.shape[0]
    bset = sorted(i for i in boundary_idx if 0 < i < n)
    aset = {i for i in (anchor_idx or ()) if 0 < i < n}
    out = {"boundaries": [], "repaired": False, "max_shift_mm": 0.0,
           "flagged": [], "flagged_at_anchor": []}
    if not bset or n < 6:
        return out

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        steps = np.array([np.nanmedian(ts[i] - ts[i - 1]) for i in range(1, n)])
        typical = 1.4826 * np.nanmedian(np.abs(steps - np.nanmedian(steps)))
        cover = np.array([np.isfinite(ts[i]).mean() for i in range(n)])

    def _anomaly(i: int) -> float:
        """Step into epoch i, minus what the scene was doing either side."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            step = float(np.nanmedian(ts[i] - ts[i - 1]))
            lo = steps[max(0, i - 6):max(1, i - 1)]
            hi = steps[min(n - 2, i):min(n - 1, i + 5)]
            local = np.nanmedian(np.concatenate([lo, hi])) if (
                lo.size + hi.size) else 0.0
        return step - (local if np.isfinite(local) else 0.0)

    for i in bset:
        shift = _anomaly(i)
        rec = {
            "index": int(i),
            "date": dates[i] if i < len(dates) else "",
            "shift_mm": round(shift * 1000.0, 2),
            "sigma": round(abs(shift) / typical, 1) if typical else float("nan"),
            "coverage_before": round(float(cover[i - 1]), 3),
            "coverage_after": round(float(cover[i]), 3),
        }
        a = i - 1
        if a in aset:
            ash = _anomaly(a)
            rec.update({"anchor_index": int(a), "anchor_date": dates[a] if a < len(dates) else "",
                        "anchor_shift_mm": round(ash * 1000.0, 2),
                        "anchor_sigma": round(abs(ash) / typical, 1) if typical else float("nan"),
                        "anchor_coverage": round(float(cover[a]), 3)})
        out["boundaries"].append(rec)
        out["max_shift_mm"] = max(out["max_shift_mm"], abs(rec["shift_mm"]),
                                  abs(rec.get("anchor_shift_mm", 0.0)))

    # Strict AND, not OR. A low-coverage boundary whose step is ordinary is
    # not evidence of anything, and flagging on coverage alone caught ten of
    # sixteen boundaries here - far too many to be credible.
    flagged = [b for b in out["boundaries"]
               if abs(b["shift_mm"]) > 1000 * 4.0 * typical
               and b["coverage_before"] < 0.30]
    out["flagged"] = flagged
    # The same rule on the step into a sparse anchor.
    out["flagged_at_anchor"] = [b for b in out["boundaries"]
                                if "anchor_shift_mm" in b
                                and abs(b["anchor_shift_mm"]) > 1000 * 4.0 * typical
                                and b["anchor_coverage"] < 0.30]
    for b in flagged:
        log.error("Reference-epoch boundary %s carries a %+.1f mm datum shift "
                  "(%.1f x the typical epoch step) with only %.0f%% of pixels "
                  "valid in the epoch before it. Every epoch after this one is "
                  "offset, which biases the rate over the whole record.",
                  b["date"], b["shift_mm"], b["sigma"], 100 * b["coverage_before"])
    for b in out["flagged_at_anchor"]:
        log.error("Anchor epoch %s (carried into every later epoch) steps %+.1f mm "
                  "(%.1f x the typical epoch step) with only %.0f%% of pixels valid in it. "
                  "Reported, not repaired.", b["anchor_date"], b["anchor_shift_mm"],
                  b["anchor_sigma"], 100 * b["anchor_coverage"])
    if repair and flagged:
        for b in flagged:
            i = b["index"]
            shift = _anomaly(i)
            if np.isfinite(shift):
                ts[i:] -= shift
        out["repaired"] = True
        log.info("Removed the scene-median datum shift at %d boundary(ies). "
                 "A uniform offset is not measurable by InSAR in the first "
                 "place, so this can only remove artefact.", len(flagged))
    return out


def _check_boundary_steps(ts: np.ndarray, anchor_idx: list[int],
                          wavelength_m: float, tolerance: float = 0.25,
                          n_sigma: float = 3.0) -> dict:
    """
    Look for stitching errors disguised as ground movement.

    `_flag_unwrap_jumps` deliberately ignores a step that does NOT revert,
    because a permanent step is how real ground movement looks. But there
    is one case where a permanent step is almost certainly not real: a
    jump of a whole number of half-wavelength cycles INTO AN ANCHOR epoch.

    THE INDEX. OPERA starts each new segment at the last secondary date of
    the one before - the ANCHOR, epoch a - and the carry copies ts[a] into
    every later epoch. An unwrapping error in the anchor granule therefore
    sits in the step into the anchor, ts[a] - ts[a-1], and never reverts.
    `anchor_idx` are those epochs (assemble_stack returns them). Until 5 Oct
    2026 this tested ts[b] - ts[b-1] with b = a + 1, the new segment's own
    first increment, which cannot contain a carry error: driven through the
    production assemble_stack with +1 cycle in 5% of anchor pixels it found
    0 suspects (excess -0.38%, the anchor steps sitting in the control)
    while those pixels ended 4 cycles off, +57.5 mm/yr against a true -10.
    The same data checked at the anchors: 333 suspects, excess +5.2%.

    THE CONTROL. Counting qualifying steps at anchors alone does not say
    whether the stitching makes errors MORE likely, because noise produces
    whole-cycle-looking steps everywhere - the first version of this check
    reported 7% of the grid per boundary and that was almost entirely noise.
    So the same count is taken at every NON-anchor epoch and the two rates
    are compared (`excess`). But a hit at an anchor is a permanent error
    whatever the control says: at an ordinary epoch the error reverts at the
    next epoch, at an anchor the carry keeps it. So the per-pixel count of
    anchor hits is returned as well (`suspect_count`, not JSON - callers pop
    it), and `suspect_pixels` is reported as permanent steps, not as noise.

    Steps must also be large against the pixel's own robust scatter, for
    the reason set out in `_flag_unwrap_jumps`; and a whole-cycle step into
    the anchor that is just the reversal of a spike in the epoch before it
    is not counted (`reversal_excluded`) - the anchor value is right there.

    Reported, not corrected. Removing an integer number of cycles from a
    pixel that genuinely moved that far would destroy a real signal, and
    which case applies is a judgement about the site.
    """
    cycle = wavelength_m / 2.0
    n = ts.shape[0]
    aset = {i for i in anchor_idx if 0 < i < n}
    out = {"checked_at": "anchor epochs (the value the carry copies)", "anchor_idx": sorted(aset),
           "boundaries_checked": 0, "suspect_pixels": 0, "per_boundary": [],
           "boundary_rate": 0.0, "control_rate": 0.0, "excess": 0.0,
           "reversal_excluded": 0, "pixels_with_any": 0,
           "suspect_count": np.zeros(ts.shape[1:], dtype="int16")}
    if not aset or n < 4:
        return out

    med, scatter = _step_scatter(ts)
    b_hit = b_tot = c_hit = c_tot = 0
    for i in range(1, n):
        if i in aset:
            hit, _, fin, rev = _anchor_hits(ts, i, med, scatter, cycle, tolerance, n_sigma)
            h = int(hit.sum())
            out["boundaries_checked"] += 1
            out["suspect_pixels"] += h
            out["reversal_excluded"] += int(rev.sum())
            out["suspect_count"] += hit.astype("int16")
            if h:
                out["per_boundary"].append({"epoch_index": i, "n_pixels": h})
            b_hit += h; b_tot += int(fin.sum())
        else:
            hit, _, fin = _whole_cycle_step(ts, i, med, scatter, cycle, tolerance, n_sigma)
            c_hit += int(hit.sum()); c_tot += int(fin.sum())

    out["boundary_rate"] = b_hit / b_tot if b_tot else 0.0
    out["control_rate"] = c_hit / c_tot if c_tot else 0.0
    out["excess"] = out["boundary_rate"] - out["control_rate"]
    out["pixels_with_any"] = int((out["suspect_count"] > 0).sum())
    return out


def _flag_unwrap_jumps(ts: np.ndarray, wavelength_m: float,
                       tolerance: float = 0.3,
                       n_sigma: float = 3.0) -> tuple[np.ndarray, dict]:
    """
    Find residual unwrapping errors that `connected_component_labels` missed.

    An unwrapping error moves a pixel by a whole number of half-wavelength
    cycles - 2.77 cm in C-band line of sight - and, being an error in one
    epoch's integer ambiguity rather than real ground motion, it usually
    reverts at the next epoch. Real ground does not move 2.8 cm in twelve
    days and exactly back again.

    THE TRAP, and why `n_sigma` exists. Successive differences of a noisy
    series are anti-correlated by construction: d_i and d_{i+1} share the
    term -x_i, so a large positive difference tends to be followed by a
    negative one. "Jumped then reverted" is therefore the NORMAL behaviour
    of noise, not evidence of anything. On a sparsely sampled stack whose
    epoch-to-epoch scatter approaches a cycle, a detector built only on
    "near-integer and reverts" flags several percent of all samples, and
    the discarded data is mostly good.

    So a candidate must also be large against that pixel's OWN scatter:
    at least `n_sigma` robust standard deviations of its epoch-to-epoch
    differences. A pixel whose normal step is 15 mm has no business
    calling a 28 mm step an unwrapping error; a pixel whose normal step is
    3 mm does.

    Both thresholds were set by measurement, not taste. Against the real
    Site A stack and synthetic cases at 1, 2 and 4 cycles:

        tolerance  n_sigma   noise false +   spikes caught   real stack
          0.30       none         --              all          3.44%
          0.30       3.0        0.03%             all          0.035%
          0.20       3.0        0.01%            none          0.003%

    `n_sigma` is what does the work - it cut the real-stack rejection rate
    a hundredfold. Tightening `tolerance` instead merely stopped the
    detector finding anything.

    Returns a boolean cube marking epochs to reject, and a summary.
    """
    cycle = wavelength_m / 2.0
    n = ts.shape[0]
    bad = np.zeros(ts.shape, dtype=bool)
    info = {"cycle_m": cycle, "flagged": 0, "checked": int(np.isfinite(ts).sum()),
            "fraction": 0.0, "median_scatter_mm": float("nan")}
    if n < 3:
        return bad, info

    d = np.diff(ts, axis=0)                     # (n-1, ny, nx)

    # Robust per-pixel drift and scatter of the differences. MAD rather
    # than std so the spikes being hunted do not inflate the threshold
    # meant to catch them.
    with warnings.catch_warnings():
        # Pixels that are NaN throughout are expected, not exceptional.
        warnings.simplefilter("ignore", RuntimeWarning)
        med = np.nanmedian(d, axis=0)           # the pixel's usual step
        mad = np.nanmedian(np.abs(d - med), axis=0)
    scatter = 1.4826 * mad
    info["median_scatter_mm"] = float(np.nanmedian(scatter) * 1000)

    # An unwrapping error is a whole-cycle jump ON TOP OF whatever the
    # pixel was already doing. Measuring the raw difference against a
    # cycle instead charges the pixel's own drift against the tolerance,
    # so a subsiding pixel fails the near-integer test for a jump that is
    # plainly a cycle.
    anomaly = d - med
    cycles = anomaly / cycle

    near_int = np.abs(cycles - np.round(cycles))
    big_enough = np.abs(anomaly) >= n_sigma * np.maximum(scatter, 1e-12)
    whole = (np.abs(np.round(cycles)) >= 1) & (near_int <= tolerance) & big_enough

    # A spike is a qualifying jump followed by its near-integer reversal.
    reverts = np.abs(np.round(cycles[:-1]) + np.round(cycles[1:])) <= 0.5
    spike = (whole[:-1] & whole[1:] & reverts
             & np.isfinite(cycles[:-1]) & np.isfinite(cycles[1:]))
    # The offending epoch is the one between the two jumps.
    bad[1:-1] = spike

    info["flagged"] = int(bad.sum())
    info["fraction"] = info["flagged"] / info["checked"] if info["checked"] else 0.0
    return bad, info


# Column order of the design matrix. Fixed, because every caller reads
# coefficients by index and a silent reordering would relabel a rate as an
# intercept without any error. Optional terms are appended, never inserted.
TERM_RATE = 0
TERM_INTERCEPT = 1


def _design(t: np.ndarray, seasonal: bool, acceleration: bool = False) -> np.ndarray:
    """
    Design matrix for the rate fit.

    Column 0 is always the rate and column 1 the intercept, so every
    caller can read the slope and its variance from a fixed index
    regardless of what else is in the model.

    With `seasonal`, an annual sine and cosine are added. On farmland that
    matters: soil moisture and crop cycles put a real annual oscillation
    into the phase, and fitting a straight line through it leaves that
    oscillation in the residual, inflating the apparent rate uncertainty
    and - where the record does not span whole years evenly - biasing the
    rate itself.

    With `acceleration`, a quadratic term is added. EGMS reports one, so
    the field exists in the schema either way, but it TRADES AGAINST THE
    RATE: over a short record a steady trend and a gentle curve are nearly
    indistinguishable, and fitting both inflates the uncertainty on each.
    Off by default for that reason.
    """
    cols = [t, np.ones_like(t)]
    if seasonal:
        cols += [np.sin(2 * np.pi * t), np.cos(2 * np.pi * t)]
    if acceleration:
        cols += [t ** 2]
    return np.vstack(cols).T


def _term_index(seasonal: bool, acceleration: bool) -> dict[str, int | None]:
    """Where each named term sits in a design matrix built as above."""
    idx: dict[str, int | None] = {"rate": TERM_RATE, "intercept": TERM_INTERCEPT}
    nxt = 2
    if seasonal:
        idx["sin"], idx["cos"] = nxt, nxt + 1
        nxt += 2
    else:
        idx["sin"] = idx["cos"] = None
    idx["accel"] = nxt if acceleration else None
    return idx


def _fit(A: np.ndarray, y: np.ndarray, inflation: float = 1.0):
    """
    Least-squares fit returning EVERY coefficient, its standard error, and
    the residual RMS - for one pixel or many.

    `y` is (n_epochs,) or (n_epochs, n_pixels).
    Returns (coefficients, sigmas, rmse), each with a leading axis of
    length p for the first two.

    This used to return the slope and its sigma alone and discard the
    rest, which meant recomputing the same least-squares elsewhere to get
    the seasonal amplitude, and throwing away the RMS entirely. EGMS
    reports acceleration, seasonality and RMSE per point, so they are all
    read off the one solve here.

    The standard errors are the textbook ones:

        sigma_j^2 = chi2/(n - p) * (A^T A)^-1 [j, j]

    which assume the residuals are independent. InSAR residuals are not:
    atmospheric delay is correlated between nearby epochs, so this
    UNDERSTATES the true uncertainty. `inflation` scales it by the
    effective-sample-size factor from the residual lag-1 autocorrelation
    (see `_noise_inflation`).

    Even inflated it remains a FLOOR, and the reason matters for how it is
    used downstream: it is a variance statistic, so it says nothing about
    BIAS. A phase-unwrapping error gives a smooth series, an excellent
    fit, small residuals - and a rate wrong by tens of mm/yr. Never select
    points on this alone; gate on unwrapping validity and report this as
    an attribute.
    """
    n, p = A.shape
    multi = y.ndim > 1
    if n <= p:
        shape = (p, y.shape[1]) if multi else (p,)
        rshape = (y.shape[1],) if multi else ()
        return (np.full(shape, np.nan), np.full(shape, np.nan),
                np.full(rshape, np.nan))

    sol, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ sol
    dof = n - p
    chi2 = np.sum(resid ** 2, axis=0)
    rmse = np.sqrt(np.maximum(chi2 / n, 0.0))
    try:
        cov_diag = np.diag(np.linalg.inv(A.T @ A))
    except np.linalg.LinAlgError:
        cov_diag = np.full(p, np.nan)

    var = np.maximum(chi2 / dof, 0.0)
    if multi:
        sigmas = np.sqrt(cov_diag[:, None] * var[None, :]) * inflation
    else:
        sigmas = np.sqrt(cov_diag * var) * inflation
    return sol, sigmas, rmse


def _rate_and_sigma(A: np.ndarray, y: np.ndarray, inflation: float = 1.0):
    """Rate and its standard error - a thin wrapper over `_fit`."""
    sol, sigmas, _ = _fit(A, y, inflation)
    return sol[TERM_RATE], sigmas[TERM_RATE]


def _noise_inflation(resid: np.ndarray) -> float:
    """
    How much to widen a rate error bar for temporally correlated noise.

    Residual lag-1 autocorrelation r gives an effective sample size of
    roughly n*(1-r)/(1+r), so the standard error grows by
    sqrt((1+r)/(1-r)). Atmospheric delay makes r positive, so this is
    always >= 1. Reported, not hidden, because it is an assumption.
    """
    if resid.shape[0] < 4:
        return 1.0
    a, b = resid[:-1], resid[1:]
    a = a - a.mean(axis=0)
    b = b - b.mean(axis=0)
    denom = np.sqrt((a ** 2).sum(axis=0) * (b ** 2).sum(axis=0))
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.where(denom > 0, (a * b).sum(axis=0) / denom, 0.0)
    r = float(np.nanmedian(r))
    r = min(max(r, 0.0), 0.95)
    return float(np.sqrt((1 + r) / (1 - r)))


def _apply_tropo(cfg: PipelineConfig, ts: np.ndarray, dates: list[str],
                 t: np.ndarray, acq_time: str | None, grid: dict,
                 geom: dict, ref_rc: tuple[int, int] | None) -> dict:
    """
    Remove the tropospheric delay in place, but only if it helps.

    DISP-S1 ships no tropospheric correction and no layer for one, so this
    is estimated with RAiDER from ERA5 - see `tropo_disp`.

    Two guards, both of which exist because the failure modes are silent:

      * ALL OR NOTHING. An epoch RAiDER could not produce is left alone,
        and if that happens to any epoch the whole correction is
        abandoned. Correcting some epochs and not others puts a step into
        the series the ground never had, and the step is the size of the
        correction - which is exactly the lesson from the solid-earth-tide
        wiring.

      * IT MUST ACTUALLY HELP. A tropospheric signal is uncorrelated with
        time, so removing it must tighten each pixel's series about its
        own trend. If the scatter goes UP, the correction is wrong -
        sign, units, model or geometry - and the uncorrected series is
        kept. Nothing about the output would otherwise reveal this: a
        wrong correction produces a complete, plausible, worse answer.
    """
    from . import tropo_disp

    if acq_time is None:
        log.warning("No acquisition time could be read from the granules, so "
                    "the weather model cannot be sampled at the right moment. "
                    "Skipping the tropospheric correction.")
        return {"applied": False, "reason": "no acquisition time"}
    if "incidence_deg" not in geom or "height" not in geom:
        log.warning("Tropospheric correction needs per-pixel incidence and "
                    "terrain height from DISP-S1-STATIC, and %s missing. "
                    "Skipping.",
                    "both are" if "incidence_deg" not in geom
                    and "height" not in geom else "one is")
        return {"applied": False, "reason": "no static layers"}

    ok, note = tropo_disp.check_model_range(cfg, dates)
    if not ok:
        log.error("Tropospheric correction skipped. %s", note)
        return {"applied": False, "reason": "weather model range"}
    log.info("%s", note)

    # Where the data comes from decides what this costs. ERA5 is queued
    # requests to the Copernicus CDS and can take hours; HRRR is AWS Open
    # Data via Herbie, which is a straight download with no queue and no
    # account. Saying "CDS request" for both was simply wrong.
    source = {"ERA5": "a queued Copernicus CDS request",
              "ERA5T": "a queued Copernicus CDS request",
              "HRRR": "a direct AWS Open Data download (no queue)",
              "HRRRAK": "a direct AWS Open Data download (no queue)",
              }.get((cfg.weather_model or "").upper(), "a separate request")
    log.info("Estimating tropospheric delay with RAiDER/%s for %d epochs. "
             "Each epoch is %s. Delays are cached per date, so an "
             "interrupted run resumes.",
             cfg.weather_model, len(dates), source)
    cube, info = tropo_disp.differential_los_delay(
        cfg, dates, acq_time, grid, geom["incidence_deg"], geom["height"])

    if cube is None:
        log.error("No tropospheric delay could be estimated for any epoch. "
                  "The series is uncorrected for troposphere.")
        return {"applied": False, "reason": "raider produced nothing", **info}

    if info["epochs_failed"]:
        log.error("RAiDER produced delays for %d of %d epochs. Correcting "
                  "only some would put a step into the series that the "
                  "ground never had, so the correction is ABANDONED. Missing "
                  "dates: %s. Re-run to retry - delays are cached per date, "
                  "so completed epochs are not recomputed.",
                  info["epochs_corrected"], len(dates),
                  ", ".join(info["failed_dates"]))
        return {"applied": False, "reason": "incomplete", **info}

    log.info("Tropospheric delay: %.1f mm at the 95th percentile, spatial "
             "spread across the AOI %.2f mm median (%.2f mm worst epoch)",
             info["magnitude_mm_p95"], info["spatial_std_mm_median"],
             info["spatial_std_mm_max"])
    if info["spatial_std_mm_median"] < 1.0:
        log.warning("The correction is almost spatially uniform (%.2f mm "
                    "across the AOI). A uniform delay cancels against the "
                    "reference point, so this will change almost nothing - "
                    "expected for a %s grid over an AOI this small. It is "
                    "not evidence the troposphere is quiet.",
                    info["spatial_std_mm_median"], cfg.weather_model)

    # THE CORRECTION MUST BE REFERENCED THE SAME WAY THE DATA IS.
    #
    # `ts` has already had a reference pixel's series subtracted, which
    # removes whatever part of the atmospheric delay is common to the
    # whole AOI - and over a small AOI that is nearly all of it. The delay
    # cube is absolute. Applying it unreferenced therefore injects the
    # entire uniform component, which the observed series never contained.
    #
    # Measured when this was wrong: the cube reached 176 mm at the 95th
    # percentile while its spatial spread across the AOI was 2.6 mm, and
    # adding it took the scatter about trend from 17.5 mm to 70.0 mm - the
    # correction was 300% worse than doing nothing. Referencing the cube
    # leaves only the part that is actually in the data.
    if ref_rc is not None:
        ref_series = cube[:, ref_rc[0], ref_rc[1]].astype("float64").copy()
        gap = ~np.isfinite(ref_series)
        if gap.any():
            tt = np.arange(cube.shape[0], dtype="float64")
            if (~gap).sum() >= 2:
                ref_series[gap] = np.interp(tt[gap], tt[~gap], ref_series[~gap])
            else:
                ref_series[gap] = 0.0
        cube = cube - ref_series[:, None, None].astype("float32")
        log.info("Delay cube referenced to the same pixel as the time series "
                 "(row %d col %d); residual magnitude %.2f mm at the 95th "
                 "percentile", ref_rc[0], ref_rc[1],
                 float(np.nanpercentile(np.abs(cube), 95)) * 1000)
    else:
        log.warning("No reference pixel, so the delay cube cannot be "
                    "referenced to match the time series. The correction "
                    "will carry the AOI-wide uniform delay, which the data "
                    "does not contain.")

    before = tropo_disp.scatter_about_trend(ts, t)
    corrected = ts + cube          # see tropo_disp on why this is a plus
    after = tropo_disp.scatter_about_trend(corrected, t)
    info.update({"scatter_before_mm": before * 1000,
                 "scatter_after_mm": after * 1000})

    if not np.isfinite(before) or not np.isfinite(after):
        # Too few epochs to fit a trend and measure scatter about it. The
        # correction is physically derived, so it is applied rather than
        # silently dropped - but the usual check did not run, and that
        # goes in the metadata so nobody assumes it did.
        ts[...] = corrected
        log.warning("The stack is too short to test whether the correction "
                    "helped (scatter about a trend needs more epochs than "
                    "this). Applying it on physical grounds, UNVERIFIED. A "
                    "short stack is the case where troposphere matters most "
                    "and is least checkable - treat the result as "
                    "indicative.")
        return {"applied": True, "verified": False,
                "reason": "stack too short to verify", **info}

    change = (after - before) / before * 100
    if after >= before:
        log.error("The tropospheric correction made the time series WORSE: "
                  "scatter about trend went from %.2f mm to %.2f mm (%+.1f%%). "
                  "A correct correction tightens a series, so something is "
                  "wrong - sign, units, weather model or geometry. The "
                  "UNCORRECTED series is kept.",
                  before * 1000, after * 1000, change)
        return {"applied": False, "reason": "made it worse", **info}

    ts[...] = corrected
    log.info("Tropospheric correction applied and verified: scatter about "
             "trend %.2f -> %.2f mm (%+.1f%%)",
             before * 1000, after * 1000, change)
    return {"applied": True, "verified": True, **info}


# Transient server errors are normal on a run this long. A 254-granule
# stack is ~3 hours of continuous HTTP reads against ASF, and a single
# 502 killed one such run at 2.5 hours with nothing kept. Two defences:
# retry the read, and cache each granule's extracted arrays so a re-run
# resumes instead of starting over.
# 403 and 429 belong here. Earthdata/CloudFront throttles sustained reads by
# refusing freshly-minted presigned URLs, which surfaces as a 403 Forbidden on
# a URL that is nowhere near its Expires time and that succeeds on a retry
# seconds later. Treating those as permanent is what let a whole epoch lose
# its /corrections without the retry loop ever firing a second attempt.
_TRANSIENT = ("403", "forbidden", "429", "too many requests", "slow down",
              "502", "503", "504", "timeout", "timed out", "bad gateway",
              "connection reset", "connection aborted", "server disconnected",
              "incomplete read", "temporarily unavailable")


class _ReadTimeout(Exception):
    """A granule read stopped making progress and was abandoned."""


@contextlib.contextmanager
def _deadline(seconds: int):
    """
    Abandon a read that has stopped making progress entirely.

    The aiohttp timeouts in _open_remote cover a stalled socket, but not
    every way this can wedge. fsspec's sync() bridge waits on its loop
    thread with timeout=None (fsspec/asyn.py), so if that thread stops
    servicing the coroutine the main thread sits in `while True:
    event.wait(1)` forever and no socket timeout ever gets the chance to
    fire. SIGALRM interrupts that, because it is an ordinary Python-level
    loop and the signal is checked between polls.

    Only the main thread can take a signal, so this is a no-op anywhere
    else - the aiohttp timeouts still apply there.
    """
    if seconds <= 0 or threading.current_thread() is not threading.main_thread():
        yield
        return

    def _fire(signum, frame):
        raise _ReadTimeout(f"no progress for {seconds}s - abandoning attempt")

    previous = signal.signal(signal.SIGALRM, _fire)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


def _reset_http_cache() -> None:
    """
    Throw away fsspec's cached filesystem, and with it the wedged session.

    fsspec caches HTTPFileSystem instances by their constructor arguments,
    so a plain retry reuses the same aiohttp session - including the dead
    keepalive connection that caused the stall. Clearing the cache forces a
    fresh session and loop on the next attempt.
    """
    try:
        import fsspec
        fsspec.AbstractFileSystem.clear_instance_cache()
    except Exception:  # noqa: BLE001
        pass


def _is_transient(exc: Exception) -> bool:
    if isinstance(exc, _ReadTimeout):
        return True
    msg = str(exc).lower()
    return any(t in msg for t in _TRANSIENT)


def _window_from_cache(arr) -> tuple:
    """
    Rebuild a window tuple from its cached form, EPSG code intact.

    The window is stored as one float64 array, which silently turns the
    EPSG code in the last slot into a float: 32616 comes back as 32616.0.
    rasterio and pyproj both tolerate that, so the rasters are unaffected,
    but anything doing int() on its string form does not - the GNSS step
    reads attrs["EPSG"] and int("32616.0") is a ValueError. That only ever
    fires on a cache hit, which is why it stayed hidden until a resumed run
    first reached the validation stage.
    """
    win = tuple(float(v) for v in arr)
    return win[:-1] + (int(round(win[-1])),)


def _atomic_savez(path: Path, **arrays) -> None:
    """Write an .npz under a temporary name and rename it into place, so a
    run killed mid-write never leaves a truncated cache entry behind."""
    import os
    tmp = path.with_name(path.name + ".part.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


def _granule_cache_path(cfg: PipelineConfig, fname: str) -> Path:
    return cfg.project_dir / "granule_cache" / (fname.replace(".nc", "") + ".npz")


def _corrections_cache_path(cfg: PipelineConfig, fname: str) -> Path:
    return cfg.project_dir / "corrections_cache" / (fname.replace(".nc", "") + ".npz")


def _read_corrections(cfg: PipelineConfig, url: str, fname: str, win,
                      expected_shape, attempts: int = 4):
    """
    Every correction layer for one granule, from disk if we already have it.

    Returns {layer: array}. An empty dict means this epoch could not be
    corrected; the caller turns that into the all-or-nothing report.

    WHY THIS IS CACHED ALL-OR-NOTHING. The displacement arrays get cached
    per granule, but /corrections used to be re-fetched on every run - about
    50 s per granule, so a resumed run paid ~3.5 hours to re-download data
    it had already read. Caching it removes that.

    The invariant that makes caching safe here is that a cache entry is
    written ONLY when every layer in CORRECTION_LAYERS was read cleanly at
    the expected shape. A partial read is never persisted, so a granule that
    failed is retried on the next run instead of having its failure frozen
    in. That matters because a partly-corrected stack is worse than an
    uncorrected one - it puts an ~8 mm step between the corrected and
    uncorrected epochs - and a cache that could memoise a partial result
    would defeat the check that exists to catch exactly that.

    The window is stored and verified: corrections are cropped to the AOI
    before caching, so an entry from a different AOI must not be reused
    even though the granule filename is the same.
    """
    cache = _corrections_cache_path(cfg, fname)
    want = set(CORRECTION_LAYERS)

    if cache.exists():
        try:
            with np.load(cache, allow_pickle=False) as z:
                cached_win = (_window_from_cache(z["__window"])
                              if "__window" in z.files else None)
                arrays = {k: z[k] for k in z.files if k != "__window"}
            if (set(arrays) == want
                    and cached_win == tuple(win)
                    and all(a.shape == expected_shape for a in arrays.values())):
                return arrays
            log.debug("Corrections cache for %s does not match this window or "
                      "layer set - refetching", fname[:60])
        except Exception as exc:  # noqa: BLE001
            log.warning("Cached corrections %s unreadable (%s) - refetching",
                        cache.name, exc)
            cache.unlink(missing_ok=True)

    xmin, xmax, ymin, ymax, _ = win
    last_exc = None
    for attempt in range(attempts):
        corr = None
        try:
            with _deadline(cfg.disp_read_timeout_s):
                corr = _open_remote(url, group="corrections")
                csub = _decimate(corr.sel(x=slice(xmin, xmax), y=slice(ymax, ymin)), cfg)
                arrays = {}
                for name in CORRECTION_LAYERS:
                    if name not in csub:
                        log.warning("%s absent from /corrections in %s",
                                    name, fname[:60])
                        continue
                    c = np.asarray(csub[name].values, dtype="float32")
                    if c.shape != expected_shape:
                        log.warning("%s has shape %s, expected %s in %s",
                                    name, c.shape, expected_shape, fname[:60])
                        continue
                    arrays[name] = c
            if set(arrays) == want:
                cache.parent.mkdir(parents=True, exist_ok=True)
                _atomic_savez(cache, __window=np.array(win, dtype="float64"), **arrays)
            return arrays
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if not _is_transient(exc) or attempt == attempts - 1:
                break
            _reset_http_cache()
            time.sleep(5 * (2 ** attempt))
        finally:
            if corr is not None:
                corr.close()

    log.warning("Could not read /corrections for %s (%s)", fname[:60], last_exc)
    return {}


def _read_granule(cfg: PipelineConfig, url: str, fname: str, window,
                  layers: tuple, attempts: int = 4):
    """
    Extract one granule's arrays, with retries and an on-disk cache.

    Returns (arrays, window, incidence, acq_time) or None. `window` is
    returned because it is computed from the first granule opened.

    WHY THE RETRY IS AROUND THE WHOLE READ, not just the open. xarray
    opens lazily: `open_dataset` succeeds while the data is still on the
    server, and the HTTP traffic happens when `.values` is touched. A
    guard around the open alone therefore catches almost nothing - which
    is exactly how a 502 mid-read escaped one and ended a three-hour run.

    The cache is keyed on the granule filename, which includes the
    reference and secondary dates and the processing timestamp, so it
    cannot collide across reprocessings.
    """
    cache = _granule_cache_path(cfg, fname)
    if cache.exists():
        try:
            with np.load(cache, allow_pickle=False) as z:
                # "__" entries are bookkeeping (window, pixel-centre x/y),
                # not layers
                arrays = {k: z[k] for k in z.files if not k.startswith("__")}
                win = (_window_from_cache(z["__window"])
                       if "__window" in z.files else window)
            return arrays, win, None, None
        except Exception as exc:  # noqa: BLE001
            log.warning("Cached %s is unreadable (%s) - refetching",
                        cache.name, exc)
            cache.unlink(missing_ok=True)

    last = None
    for attempt in range(attempts):
        ds = None
        try:
            with _deadline(cfg.disp_read_timeout_s):
                ds = _open_remote(url)
                win = window if window is not None else _subset_window(ds, cfg)
                xmin, xmax, ymin, ymax, _ = win
                sub = _decimate(ds.sel(x=slice(xmin, xmax), y=slice(ymax, ymin)), cfg)
                if sub.sizes.get("x", 0) == 0 or sub.sizes.get("y", 0) == 0:
                    return None, win, None, None  # AOI miss: not retryable
                arrays = {}
                for name in layers:
                    if name in sub:
                        arrays[name] = np.asarray(sub[name].values, dtype="float32")
                # The pixel centres actually selected. The window holds the
                # AOI corners, not pixel edges, so without these the grid
                # origin was the AOI corner - up to one 30 m pixel off - and
                # the chain had to reopen a remote granule on every run just
                # to recover them.
                xs = np.asarray(sub["x"].values, dtype="float64")
                ys = np.asarray(sub["y"].values, dtype="float64")
            if DISPLACEMENT_LAYER not in arrays:
                return None, win, None, None
            cache.parent.mkdir(parents=True, exist_ok=True)
            _atomic_savez(cache, __window=np.array(win, dtype="float64"),
                          __x=xs, __y=ys, **arrays)
            return arrays, win, None, None
        except Exception as exc:  # noqa: BLE001
            last = exc
            if "credential" in str(exc).lower() or "auth" in str(exc).lower():
                raise RuntimeError(f"Earthdata authentication failed: {exc}")
            if not _is_transient(exc) or attempt == attempts - 1:
                break
            _reset_http_cache()
            wait = 5 * (2 ** attempt)
            log.warning("Transient error on %s (attempt %d/%d): %s - retrying "
                        "in %ds", fname[:50], attempt + 1, attempts, exc, wait)
            time.sleep(wait)
        finally:
            if ds is not None:
                try:
                    ds.close()
                except Exception:  # noqa: BLE001
                    pass
    log.error("Could not read %s after %d attempts: %s", fname[:60], attempts, last)
    return None, window, None, None


def granule_layers() -> tuple:
    """Every layer assemble_stack reads from a granule, in one pass."""
    return (DISPLACEMENT_LAYER, CONNCOMP_LAYER, RESIDUAL_LAYER) + _MEAN_LAYERS


def _sidecar(cfg, name: str) -> Path | None:
    pd = getattr(cfg, "project_dir", None)
    return Path(pd) / "granule_cache" / name if pd else None


def identification(cfg, url: str, fname: str) -> tuple[float | None, str | None]:
    """_geometry_from_granule, remembered per frame under granule_cache/ so a
    fully cached re-run does not need Earthdata to be up."""
    import json
    side = _sidecar(cfg, "_identification.json")
    frame = _frame_of(fname)
    try:
        rec = json.loads(side.read_text()) if side is not None and side.exists() else {}
    except (OSError, ValueError):
        rec = {}
    if frame in rec:
        return rec[frame].get("incidence_deg"), rec[frame].get("acq_time")
    inc, acq = _geometry_from_granule(url)
    if side is not None and inc is not None:
        try:
            side.parent.mkdir(parents=True, exist_ok=True)
            rec[frame] = {"incidence_deg": inc, "acq_time": acq, "from": fname}
            side.write_text(json.dumps(rec, indent=1))
        except OSError:
            pass
    return inc, acq


def grid_coords(cfg, results, window, shape) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Pixel-centre x/y of the cached AOI subset, or None if they cannot be had.

    From a cached granule that stored them (every granule read since 5 Oct
    2026), else from the sidecar an earlier lookup wrote, else by repeating
    the selection on one remote granule (older caches hold arrays and the
    window only) and remembering the answer.
    """
    want = tuple(shape) if shape is not None else None

    def fits(xs, ys):
        return xs.size > 1 and ys.size > 1 and (want is None or (ys.size, xs.size) == want)

    if getattr(cfg, "project_dir", None) is not None:
        for p in results:
            c = _granule_cache_path(cfg, p.properties["fileName"])
            if not c.exists():
                continue
            try:
                with np.load(c, allow_pickle=False) as z:
                    if "__x" not in z.files or "__y" not in z.files:
                        continue
                    if window is not None and "__window" in z.files \
                            and _window_from_cache(z["__window"]) != tuple(window):
                        continue
                    xs, ys = np.asarray(z["__x"], float), np.asarray(z["__y"], float)
                if fits(xs, ys):
                    return xs, ys
            except Exception:  # noqa: BLE001
                continue
    side = _sidecar(cfg, "_grid_coords.npz")
    if side is not None and side.exists():
        try:
            with np.load(side, allow_pickle=False) as z:
                if window is None or _window_from_cache(z["__window"]) == tuple(window):
                    xs, ys = np.asarray(z["__x"], float), np.asarray(z["__y"], float)
                    if fits(xs, ys):
                        return xs, ys
        except Exception:  # noqa: BLE001
            pass
    if window is None:
        return None
    url = next((u for u in (_data_url(p) for p in results) if u), "")
    if not url:
        return None
    try:
        ds = _open_remote(url)
        try:
            xmin, xmax, ymin, ymax, _ = window
            sub = _decimate(ds.sel(x=slice(xmin, xmax), y=slice(ymax, ymin)), cfg)
            xs = np.asarray(sub["x"].values, float)
            ys = np.asarray(sub["y"].values, float)
        finally:
            ds.close()
    except Exception as exc:  # noqa: BLE001
        log.warning("Could not read the subset's pixel coordinates (%s)", exc)
        return None
    if not fits(xs, ys):
        log.warning("Granule subset is %dx%d, the stack is %s - coordinates not used",
                    ys.size, xs.size, want)
        return None
    if side is not None:
        try:
            side.parent.mkdir(parents=True, exist_ok=True)
            _atomic_savez(side, __window=np.array(window, dtype="float64"), __x=xs, __y=ys)
        except OSError:
            pass
    return xs, ys


def _prefetch_one(job):
    cfg, url, fname, window = job[:4]
    if len(job) > 4 and job[4]:
        time.sleep(job[4])          # stagger the first wave of Earthdata logins
    arrays, win, _, _ = _read_granule(cfg, url, fname, window, granule_layers())
    if arrays is None:
        return fname, False, False
    corr_ok = True
    if cfg.disp_apply_corrections:
        corr = _read_corrections(cfg, url, fname, win,
                                 arrays[DISPLACEMENT_LAYER].shape)
        corr_ok = set(corr) == set(CORRECTION_LAYERS)
    return fname, True, corr_ok


def prefetch(cfg: PipelineConfig, results, workers: int | None = None) -> dict:
    """
    Fill the granule and corrections caches concurrently, so assemble_stack
    then reads everything from disk.

    The first granule is read here in the main process: it fixes the AOI
    window every other read must use. The rest go to separate PROCESSES,
    not threads - _deadline's SIGALRM watchdog only works in a main thread,
    and a stalled fsspec read is the failure it exists for. 'spawn' rather
    than fork, because by then this process has fsspec's event-loop thread
    running and forking a process with live threads can deadlock.

    Returns {"read": n, "failed": [...], "uncorrected": [...]}; a failed
    granule is simply retried by assemble_stack's own sequential read.
    """
    import multiprocessing as mp

    from . import progress

    workers = max(1, int(workers or cfg.disp_prefetch_workers or 1))
    jobs = [(p.properties["fileName"], _data_url(p)) for p in results]
    jobs = [(f, u) for f, u in jobs if u]
    todo = [(f, u) for f, u in jobs
            if not _granule_cache_path(cfg, f).exists()
            or (cfg.disp_apply_corrections and not _corrections_cache_path(cfg, f).exists())]
    total = len(jobs)
    progress.emit(cfg.project_dir, "disp_prefetch", "start", done=total - len(todo),
                  total=total, msg=f"{len(todo)} granules to fetch, {workers} at a time")
    if not todo:
        progress.emit(cfg.project_dir, "disp_prefetch", "done", done=total, total=total)
        return {"read": total, "failed": [], "uncorrected": []}

    # the window: from any cached granule, else by reading the first one here
    window = None
    for f, _ in jobs:
        c = _granule_cache_path(cfg, f)
        if c.exists():
            with np.load(c, allow_pickle=False) as z:
                if "__window" in z.files:
                    window = _window_from_cache(z["__window"])
                    break
    if window is None:
        f0, u0 = todo.pop(0)
        _, ok0, _ = _prefetch_one((cfg, u0, f0, None))
        if not ok0:
            raise RuntimeError(f"first DISP-S1 granule {f0} could not be read")
        with np.load(_granule_cache_path(cfg, f0), allow_pickle=False) as z:
            window = _window_from_cache(z["__window"])
    failed, uncorrected = [], []
    done = total - len(todo)
    ctx = mp.get_context("spawn")
    with ctx.Pool(workers) as pool:
        for fname, ok, corr_ok in pool.imap_unordered(
                _prefetch_one, [(cfg, u, f, window, 2.0 * i if i < workers else 0)
                                for i, (f, u) in enumerate(todo)]):
            done += 1
            if not ok:
                failed.append(fname)
            elif not corr_ok:
                uncorrected.append(fname)
            progress.emit(cfg.project_dir, "disp_prefetch", "running", done=done,
                          total=total, msg=fname[:80])
            if done % 25 == 0:
                log.info("  prefetched %d / %d granules", done, total)
    if failed or uncorrected:
        # one sequential pass over the failures, away from the crowd
        urls = dict(todo)
        for fname in list(failed) + list(uncorrected):
            _, ok, corr_ok = _prefetch_one((cfg, urls[fname], fname, window))
            if ok and fname in failed:
                failed.remove(fname)
            if ok and corr_ok and fname in uncorrected:
                uncorrected.remove(fname)
            elif ok and not corr_ok and fname not in uncorrected:
                uncorrected.append(fname)
    log.info("Prefetch: %d granules cached, %d failed, %d without corrections",
             total - len(failed), len(failed), len(uncorrected))
    progress.emit(cfg.project_dir, "disp_prefetch", "done", done=done, total=total,
                  msg=f"{len(failed)} failed, {len(uncorrected)} without corrections")
    return {"read": total - len(failed), "failed": failed, "uncorrected": uncorrected}


def assemble_stack(cfg: PipelineConfig, results) -> dict:
    """
    Stitch the granules into one continuous, AOI-subset LOS series on the
    DISP-S1 grid: supplied corrections removed, connected-component 0
    dropped per epoch, reference epochs carried, anchor steps checked and
    residual whole-cycle spikes rejected. No spatial reference, no fit.

    build_stack() fits a rate to this directly; disp_chain.py resamples it
    and runs the record's MintPy correction chain instead. Both then call
    check_stitch(), which refuses a stack whose stitching is known to be
    wrong - this function measures and records, it does not decide.
    """
    dates: list[str] = []
    frames_data: list[np.ndarray] = []
    dropped_cc: list[float] = []
    cc0_share: list[float] = []       # component 0 per epoch, masked or not
    applied_corr: dict[str, int] = {}
    corrected_epochs: list[bool] = []
    sums: dict[str, np.ndarray] = {}
    counts_layer: dict[str, np.ndarray] = {}
    x0 = y0 = dx = dy = epsg = None
    xmin = ymax = None
    shape = None
    incidence_deg = None
    acq_ident = None
    frame_id = None

    # The reference-epoch carry. `last_abs` holds, per pixel, the most
    # recent absolute displacement at which that pixel was valid, and
    # `last_idx` which epoch that came from. Carrying `last_abs` rather
    # than the previous frame is what stops a single bad granule at a
    # boundary from killing a pixel for the rest of the decade.
    carry = None
    last_abs = None
    last_idx = None
    last_ref = None
    n_boundaries = 0
    # boundary_idx: the first epoch of each new segment (datum and coverage
    # reporting). anchor_idx: the epoch the new segment is referenced to -
    # the last epoch of the segment before, whose value the carry copies
    # into every later epoch. They differ by one, and checking the wrong one
    # is how an anchor error went undetected until 5 Oct 2026.
    boundary_idx: list[int] = []
    anchor_idx: list[int] = []
    lost_anchors: list[str] = []
    # per pixel: at how many anchors the carried value came from a
    # component snaphu marked unreliable (0)
    anchor_cc0 = None
    n_anchor_cc0 = 0
    prev_cc0 = None

    # Every layer this run needs, read in one pass per granule so a
    # retry or a cache hit covers all of them together.
    want = granule_layers()
    resid_bad = resid_obs = None
    window = None
    skipped: list[str] = []
    read_names: list[str] = []

    for i, p in enumerate(results):
        fname = p.properties["fileName"]
        ref, sec = _dates(fname)
        url = _data_url(p)
        if not url:
            log.warning("No usable data URL for %s - skipping", fname[:60])
            skipped.append(sec)
            continue

        arrays, window, inc_g, acq_g = _read_granule(
            cfg, url, fname, window, want)
        if arrays is None:
            skipped.append(sec)
            continue
        if epsg is None and window is not None:
            xmin, xmax, ymin, ymax, epsg = window

        # NO sign flip. The `displacement` layer states "Positive values
        # indicate apparent motion towards the platform", which is
        # MintPy's convention for timeseries.h5 / velocity.h5 - the two
        # already agree and negating would invert the result. Subsidence
        # (motion away from the satellite) is negative in both.
        disp = arrays[DISPLACEMENT_LAYER].copy()

        if shape is None:
            shape = disp.shape
        elif disp.shape != shape:
            # Granules from one frame share a grid; a mismatch means the
            # AOI is being served by more than one frame, and stacking
            # would silently misalign pixels.
            log.warning("%s has shape %s, expected %s - skipping. An AOI "
                        "spanning two frames must be processed one frame "
                        "at a time.", fname[:60], disp.shape, shape)
            skipped.append(sec)
            continue

        # Apply the corrections DISP-S1 ships but does not apply itself.
        #
        # These must be applied to EVERY epoch or to none. Correcting some
        # epochs and not others puts a step of the correction's own size -
        # measured at ~8 mm over this AOI, comparable to the signal -
        # between the corrected and uncorrected epochs, which is worse
        # than leaving the correction in throughout. So failures are
        # retried, recorded per epoch, and refused by check_stitch.
        if cfg.disp_apply_corrections:
            corrections = _read_corrections(cfg, url, fname, window, disp.shape)
            for name, c in corrections.items():
                # NaN in a correction means "unknown", not "zero
                # correction" - but blanking the pixel would be
                # the carry bug again, so treat it as zero here.
                disp = disp - np.nan_to_num(c)
            got = set(corrections)
            corrected_epochs.append(got == set(CORRECTION_LAYERS))
            for name in got:
                applied_corr[name] = applied_corr.get(name, 0) + 1

        # Network misclosure: dolphin inverts a nearest-3 network with an
        # L1 norm, and the summed residual for this date is where the
        # unwrapped interferograms disagree with each other. A whole-cycle
        # unwrapping error in one pair leaves ~2 pi here, so |r| >= pi is
        # the DISP-S1 counterpart of MintPy's non-zero integer ambiguity
        # around a triplet. Counted for QA, not masked: L1 has already
        # out-voted the inconsistent pair.
        if RESIDUAL_LAYER in arrays and arrays[RESIDUAL_LAYER].shape == disp.shape:
            r = arrays[RESIDUAL_LAYER]
            if resid_bad is None:
                resid_bad = np.zeros(disp.shape, dtype="int32")
                resid_obs = np.zeros(disp.shape, dtype="int32")
            ok_r = np.isfinite(r) & np.isfinite(disp)
            resid_obs += ok_r
            resid_bad += ok_r & (np.abs(np.nan_to_num(r)) >= np.pi)

        # Drop pixels this granule could not unwrap reliably. These drops
        # affect the reported epoch only - never the carry.
        cur_cc0 = None
        if CONNCOMP_LAYER in arrays:
            bad = arrays[CONNCOMP_LAYER] == 0
            cur_cc0 = bad
            cc0_share.append(float(bad.mean()))
            if cfg.disp_mask_connected_components and bad.any():
                disp = np.where(bad, np.nan, disp)
                dropped_cc.append(float(bad.mean()))

        if x0 is None:
            incidence_deg, acq_ident = identification(cfg, url, fname)
            m = re.search(r"_F(\d+)_", fname)
            frame_id = int(m.group(1)) if m else None
            # DISP-S1 posts at 30 m, but disp_decimate keeps every Nth
            # pixel, so the EFFECTIVE pixel size is 30 * N. This must scale
            # or everything downstream is wrong by that factor: the
            # reference-point lookup divides by dx and lands off-grid, and
            # X_STEP written into the product georeferences the raster at
            # the wrong scale. A decimated run silently produced a raster
            # ten times too small on the ground before this was caught.
            step = 30.0 * max(1, int(getattr(cfg, "disp_decimate", 1) or 1))
            dx, dy = step, -step
            # Provisional origin: the AOI corner taken as the outer edge of
            # the first pixel. Replaced below by the real first pixel centre
            # (grid_coords) - the corner is up to one pixel off it.
            x0, y0 = xmin + dx / 2, ymax + dy / 2

        # Quality layers accumulate over the stack rather than being taken
        # from whichever granule was read first - a 12-day pair at the
        # start of a decade does not describe the decade. They are sampled
        # rather than read every epoch: a mean over ~30 granules spread
        # across the record is as good as one over 254, and each extra
        # layer is another remote read.
        for name in (_MEAN_LAYERS if i % _QUALITY_STRIDE == 0 else ()):
            if name not in arrays:
                continue
            arr = arrays[name]
            ok = np.isfinite(arr)
            if name not in sums:
                sums[name] = np.zeros(shape, dtype="float64")
                counts_layer[name] = np.zeros(shape, dtype="int32")
            sums[name] += np.where(ok, arr, 0.0)
            counts_layer[name] += ok

        # Stitch across a reference-epoch change. The new segment restarts
        # from zero at its own reference date, which - for a complete
        # granule list - is exactly the last secondary date of the segment
        # before it, so `last_abs` is the displacement at that date.
        #
        # Where a pixel was NaN at that date, `last_abs` still holds its
        # value from the most recent epoch where it was valid. That is a
        # small error (whatever the ground did in the gap, typically days)
        # in exchange for keeping the pixel. The alternative - carrying the
        # NaN - loses the pixel permanently, and compounds over every
        # later boundary.
        if last_ref is not None and ref != last_ref:
            n_boundaries += 1
            boundary_idx.append(len(frames_data))
            a = len(frames_data) - 1
            # The carry is only right if the new reference date IS the
            # epoch just stitched. Verified on every boundary of F04861 and
            # F04862 today, but a reprocessing or an overlapping segment
            # could break it, and then every later epoch would be offset by
            # the motion between the two dates with nothing to show for it.
            if ref == dates[a]:
                anchor_idx.append(a)
                if prev_cc0 is not None:
                    if anchor_cc0 is None:
                        anchor_cc0 = np.zeros(shape, dtype="int16")
                    anchor_cc0 += prev_cc0
                    n_anchor_cc0 += 1
            elif ref in skipped:
                # The anchor granule itself could not be read: the carry
                # comes from the epoch before it, so every later epoch is
                # off by whatever the ground did in that gap.
                lost_anchors.append(ref)
            elif ref in dates:
                raise RuntimeError(
                    f"DISP-S1 segment starting {sec} is referenced to {ref}, an earlier secondary date "
                    f"than the last epoch stitched ({dates[a]}). The segments overlap, so carrying the "
                    "last value would offset every later epoch by the motion in between. This stack "
                    "cannot be stitched here; check the granule list (a reprocessing mixed in?).")
            else:
                raise RuntimeError(
                    f"DISP-S1 segment starting {sec} is referenced to {ref}, which is not the secondary "
                    "date of any earlier granule in this stack. The reference-epoch carry needs that "
                    "date's displacement; check the date range and the granule list.")
            carry = last_abs.copy()
            stale = (last_idx < len(frames_data) - 1) & np.isfinite(carry)
            log.info("Reference epoch change at %s (%s -> %s): carrying "
                     "%+.1f mm mean over %d pixels, %d of them from an "
                     "earlier epoch",
                     sec, last_ref, ref, float(np.nanmean(carry) * 1000),
                     int(np.isfinite(carry).sum()), int(stale.sum()))
        if carry is not None:
            disp = disp + carry

        if last_abs is None:
            last_abs = np.full(shape, np.nan, dtype="float32")
            last_idx = np.full(shape, -1, dtype="int32")
        good = np.isfinite(disp)
        last_abs[good] = disp[good]
        last_idx[good] = len(frames_data)

        dates.append(sec)
        frames_data.append(disp)
        read_names.append(fname)
        last_ref = ref
        prev_cc0 = cur_cc0

        if (i + 1) % 25 == 0:
            log.info("  read %d / %d granules", i + 1, len(results))

    if not frames_data:
        raise RuntimeError("No DISP-S1 granules could be read for this AOI")

    if skipped:
        # A skipped ordinary epoch costs one sample. A skipped ANCHOR
        # granule costs the offset that anchors every later epoch, because
        # the carry is read at that date - so the two are not equally bad
        # and the log should not treat them the same. (The anchor is the
        # new segment's REFERENCE date; this used to look up the new
        # segment's first secondary date instead, one epoch late.)
        log.warning("%d granule(s) could not be read and were skipped: %s%s",
                    len(skipped), ", ".join(skipped[:8]),
                    " ..." if len(skipped) > 8 else "")
        if lost_anchors:
            log.error("%d of those are reference-epoch anchor granules "
                      "(%s). The carry is read at those dates, so every "
                      "later epoch is offset by whatever the ground did in "
                      "the gap. Re-run to pick them up - granules already "
                      "read are cached under granule_cache/ and will not be "
                      "refetched.", len(lost_anchors), ", ".join(lost_anchors))

    ts = np.stack(frames_data)                # metres, LOS, toward-positive
    # np.stack copies, so the per-epoch list is now a second full copy of
    # the stack. At a 7 km AOI that is 68 MB and nobody notices; at 50 km
    # it is 2.8 GB, and holding both halves the size of AOI this machine
    # can process. Drop it here.
    frames_data.clear()
    del last_abs, last_idx, carry

    # The real pixel-centre grid. The window holds AOI corners, and the
    # first selected pixel centre lies up to one pixel inside them - so the
    # corner-based origin put the legacy route's rasters up to ~15 m off
    # (about +7 E / -8 N m at Site A, -10.5 / -11.3 m at Site B).
    grid_origin = "aoi corner (approximate: the subset's pixel coordinates could not be read)"
    xy = grid_coords(cfg, results, window, shape)
    if xy is not None:
        xs_c, ys_c = xy
        x0, y0 = float(xs_c[0]), float(ys_c[0])
        dx, dy = float(xs_c[1] - xs_c[0]), float(ys_c[1] - ys_c[0])
        grid_origin = "first pixel centre read from the granule"
    else:
        log.warning("Pixel coordinates of the subset are unavailable; the grid origin is "
                    "the AOI corner, which can be up to one %.0f m pixel off.", abs(dx))

    # Acquisition time of day from the granule names (every granule states
    # it), the median over those read; /identification only cross-checks.
    secs = [s for s in (acq_seconds(f) for f in read_names) if s is not None]
    acq_sec = int(np.median(secs)) if secs else None
    acq_time = (f"{acq_sec // 3600:02d}:{acq_sec % 3600 // 60:02d}:{acq_sec % 60:02d}"
                if acq_sec is not None else acq_ident)
    if acq_sec is not None and acq_ident:
        try:
            hh, mm, ss = (int(v) for v in acq_ident.split(":"))
            if abs(hh * 3600 + mm * 60 + ss - acq_sec) > 120:
                log.warning("Acquisition time from the granule names (%s) and /identification "
                            "(%s) differ by more than 2 minutes; using the names.", acq_time, acq_ident)
        except ValueError:
            pass

    attrs_wavelength = 0.05546576             # C-band, from the granules
    log.info("Stack: %d epochs, %d x %d pixels at %.0f m, %d reference-epoch "
             "boundaries stitched",
             ts.shape[0], ts.shape[2], ts.shape[1], abs(dx), n_boundaries)
    n_ok = int(sum(corrected_epochs)) if corrected_epochs else 0
    corr = {"requested": bool(cfg.disp_apply_corrections), "epochs": int(ts.shape[0]),
            "epochs_corrected": n_ok, "layers": list(CORRECTION_LAYERS),
            "uncorrected_dates": [d for d, ok in zip(dates, corrected_epochs) if not ok]}
    if cfg.disp_apply_corrections:
        if n_ok == len(corrected_epochs) and n_ok:
            corr["status"] = "all"
            log.info("Corrections removed from every one of the %d epochs: %s",
                     n_ok, ", ".join(sorted(applied_corr)))
        elif n_ok == 0:
            corr["status"] = "none"
            log.error("disp_apply_corrections is set but NO epoch could be "
                      "corrected. Solid earth tide and ionospheric delay "
                      "remain in every epoch of this result.")
        else:
            corr["status"] = "partial"
            missing = corr["uncorrected_dates"]
            log.error("Corrections were applied to %d of %d epochs. The "
                      "uncorrected epochs (%s) now sit about 8 mm away from "
                      "the rest for no physical reason - a step this stack "
                      "did not have before. Re-run, or set "
                      "disp_apply_corrections: false so every epoch is "
                      "treated the same.",
                      n_ok, len(corrected_epochs),
                      ", ".join(missing[:8]) + (" ..." if len(missing) > 8 else ""))
    else:
        corr["status"] = "off"
        log.warning("disp_apply_corrections is false: solid earth tide and "
                    "ionospheric delay are left in the displacement. They are "
                    "near-uniform across a small AOI, so most of their effect "
                    "cancels against the reference point - but note it.")
    log.warning("DISP-S1 applies NO tropospheric correction and supplies no "
                "layer for it. Troposphere is the dominant error source at "
                "these scales - treat short-span rates from this backend as "
                "indicative only.")
    if dropped_cc:
        log.info("connected_component_labels==0 removed %.1f%% of pixels per "
                 "epoch on average (worst granule %.1f%%)",
                 100 * float(np.mean(dropped_cc)), 100 * float(np.max(dropped_cc)))

    # A stitching error hides as a permanent step, so the spike detector
    # below cannot see it. Check the anchors specifically.
    bstep = _check_boundary_steps(ts, anchor_idx, attrs_wavelength)
    suspect = bstep.pop("suspect_count")
    bdat = _check_boundary_datum(ts, boundary_idx, dates,
                                 repair=cfg.disp_repair_boundary_datum,
                                 anchor_idx=anchor_idx)
    if bstep["boundaries_checked"]:
        n_px = int(np.prod(ts.shape[1:]))
        log.info("Whole-cycle steps into anchor epochs: %.2f%% of samples, "
                 "against %.2f%% at ordinary epochs (control)",
                 100 * bstep["boundary_rate"], 100 * bstep["control_rate"])
        if bstep["suspect_pixels"]:
            # Whatever the control says: an ordinary-epoch error reverts at
            # the next epoch, an anchor error is carried for good.
            log.warning("%d pixels (%.2f%% of the grid) carry a suspected whole-cycle "
                        "error into every later epoch (%d pixel-anchors over %d anchors; "
                        "layer anchorSuspect). Not corrected automatically - removing "
                        "whole cycles from ground that genuinely moved that far would "
                        "destroy the signal.", bstep["pixels_with_any"],
                        100 * bstep["pixels_with_any"] / max(1, n_px), bstep["suspect_pixels"],
                        bstep["boundaries_checked"])
        if bstep["excess"] > 0.005 and bstep["boundary_rate"] > 1.5 * max(
                bstep["control_rate"], 1e-9):
            log.warning("Anchors show %.2f%% more whole-cycle steps than "
                        "ordinary epochs. OPERA's reference schedule has "
                        "nothing to do with this site, so that excess points "
                        "at the carry inheriting unwrapping errors.", 100 * bstep["excess"])

        if cfg.disp_repair_boundary_steps:
            if bstep["excess"] > 0.005:
                rep = _repair_boundary_steps(ts, anchor_idx, attrs_wavelength)
                log.warning("Repaired %d pixel-anchors across %d "
                            "anchors, removing %d whole cycles. This "
                            "erases real motion wherever the ground actually "
                            "moved 27.7 mm between consecutive epochs - at "
                            "this site's rates it does not, but check that "
                            "before trusting it elsewhere.",
                            rep["repaired_pixels"], rep["boundaries"],
                            rep["cycles_removed"])
                after = _check_boundary_steps(ts, anchor_idx, attrs_wavelength)
                after.pop("suspect_count")
                bstep["after_repair"] = {k: after[k] for k in ("suspect_pixels", "boundary_rate",
                                                               "control_rate", "excess")}
                bstep["repair"] = rep
                log.info("After repair: anchor rate %.2f%% against control "
                         "%.2f%% (excess %+.2f%%)",
                         100 * after["boundary_rate"],
                         100 * after["control_rate"], 100 * after["excess"])
            else:
                log.info("disp_repair_boundary_steps is on but there is no "
                         "excess to repair - leaving the stack alone.")

    # Residual unwrapping errors that survived the per-granule conncomp
    # mask, caught temporally: a whole-cycle jump that reverts next epoch.
    jinfo = None
    if cfg.disp_flag_unwrap_jumps:
        jumps, jinfo = _flag_unwrap_jumps(ts, float(attrs_wavelength))
        if jinfo["flagged"]:
            log.info("Rejected %d epoch-pixels (%.3f%% of valid samples) as "
                     "residual unwrapping errors: near-integer jumps of "
                     "%.1f mm that revert the next epoch",
                     jinfo["flagged"], 100 * jinfo["fraction"],
                     jinfo["cycle_m"] * 1000)
            if jinfo["fraction"] > 0.02:
                log.warning("More than 2%% of samples look like unwrapping "
                            "spikes. That is high enough to suspect the "
                            "unwrapping itself rather than isolated errors.")
            ts = np.where(jumps, np.nan, ts)
        else:
            log.info("No residual unwrapping spikes found beyond conncomp")

    flat = ts.reshape(ts.shape[0], -1)
    finite = np.isfinite(flat)
    counts = finite.sum(axis=0)
    valid_fraction = (counts / ts.shape[0]).reshape(ts.shape[1:])

    # A coverage ratchet is the failure mode this module is most prone to,
    # and it is invisible in the final product - so check for it directly
    # rather than waiting for someone to notice 0.8% coverage. A healthy
    # stack has a roughly flat per-epoch valid fraction; a ratchet decays
    # monotonically towards zero. The decision uses the share of pixels
    # valid in ANY of the first / last tenth of the epochs, not the per-epoch
    # share: a ratchet kills pixels for good, while a snowy final winter
    # only dips the per-epoch share and must not be called a ratchet.
    per_epoch = finite.mean(axis=1)
    first, last = float(per_epoch[:5].mean()), float(per_epoch[-5:].mean())
    log.info("Per-epoch valid fraction: %.1f%% at the start, %.1f%% at the "
             "end, %.1f%% mean", first * 100, last * 100,
             float(per_epoch.mean()) * 100)
    k = max(1, min(max(10, ts.shape[0] // 10), ts.shape[0] // 2))
    alive_first = float(finite[:k].any(axis=0).mean())
    alive_last = float(finite[-k:].any(axis=0).mean())
    ratchet = {"per_epoch_first5": round(first, 4), "per_epoch_last5": round(last, 4),
               "epochs_compared": k, "alive_first": round(alive_first, 4),
               "alive_last": round(alive_last, 4),
               "fired": bool(alive_first > 0.05 and alive_last < 0.5 * alive_first)}
    if ratchet["fired"]:
        log.error("Pixels observed at all fell from %.1f%% (first %d epochs) to %.1f%% (last %d) "
                  "across the stack. That is the reference-epoch carry losing pixels, "
                  "not the ground decorrelating.", alive_first * 100, k, alive_last * 100, k)

    A = {"ts": ts, "dates": dates, "results": results, "flat": flat, "finite": finite,
         "counts": counts, "valid_fraction": valid_fraction, "per_epoch_valid": per_epoch,
         "x0": x0, "y0": y0, "dx": dx, "dy": dy, "epsg": epsg, "shape": shape,
         "grid_origin": grid_origin,
         "sums": sums, "counts_layer": counts_layer, "incidence_deg": incidence_deg,
         "acq_time": acq_time, "acq_seconds": acq_sec, "frame_id": frame_id,
         "boundary_idx": boundary_idx, "anchor_idx": anchor_idx, "lost_anchors": lost_anchors,
         "anchor_suspect": suspect,
         "anchor_conncomp0_share": (anchor_cc0 / max(1, n_anchor_cc0)).astype("float32")
         if anchor_cc0 is not None else None,
         "boundary_steps": bstep, "boundary_datum": bdat, "coverage_ratchet": ratchet,
         "jumps": jinfo,
         "conncomp_dropped": dropped_cc, "conncomp0_share": cc0_share, "skipped": skipped,
         "residual_bad": resid_bad, "residual_obs": resid_obs,
         "corrections_applied": dict(applied_corr), "corrections": corr,
         "corrected_epochs": n_ok}
    A["stitch_problems"] = stitch_problems(A, cfg)
    return A


def stitch_problems(A: dict, cfg) -> list[dict]:
    """
    What the stitching measured that should stop or qualify a result, as
    [{code, text, hard}] in plain words. check_stitch() refuses on the hard
    ones; the rest go into disp_chain.json for the QA and the reviewer.

    Hard: an epoch subset left uncorrected (an ~8 mm step the ground never
    had), an anchor granule lost (a permanent offset unless the date is
    accepted in allow_missing_dates), and a coverage ratchet. Until 5 Oct
    2026 all three were log lines and the run went on to a full report.
    """
    out = []
    c = A.get("corrections") or {}
    n = c.get("epochs") or len(A.get("dates") or ())
    if c.get("status") == "partial":
        out.append({"code": "corrections_partial", "hard": True,
                    "text": (f"tide and ionosphere layers were removed from {c['epochs_corrected']} of {n} "
                             f"epochs only; the rest sit about 8 mm away for no physical reason "
                             f"({', '.join(c['uncorrected_dates'][:6])}"
                             f"{' ...' if len(c['uncorrected_dates']) > 6 else ''}). Re-run (reads "
                             "are cached), or set disp_apply_corrections: false.")})
    elif c.get("status") == "none":
        out.append({"code": "corrections_none", "hard": False,
                    "text": "disp_apply_corrections is set but no epoch could be corrected; tides and "
                            "ionosphere remain in every epoch (consistently)."})
    lost = list(A.get("lost_anchors") or ())
    if lost:
        allowed = {str(d) for d in (cfg.allow_missing_dates or ())}
        unaccepted = [d for d in lost if "all" not in allowed and d not in allowed]
        out.append({"code": "anchor_lost", "hard": bool(unaccepted),
                    "text": (f"{len(lost)} reference-epoch anchor granule(s) could not be read "
                             f"({', '.join(lost)}): every later epoch is offset by the motion in the gap. "
                             + ("Re-run to fetch them, or list the date(s) in allow_missing_dates "
                                "to accept the offset." if unaccepted else
                                "Accepted in allow_missing_dates."))})
    other = sorted(set(A.get("skipped") or ()) - set(lost))
    if other:
        out.append({"code": "skipped_granules", "hard": False,
                    "text": f"{len(other)} ordinary granule(s) could not be read and are missing: "
                            f"{', '.join(other[:8])}{' ...' if len(other) > 8 else ''}."})
    r = A.get("coverage_ratchet") or {}
    if r.get("fired"):
        out.append({"code": "coverage_ratchet", "hard": True,
                    "text": (f"pixels observed at all fell from {100 * r['alive_first']:.0f}% to "
                             f"{100 * r['alive_last']:.0f}% across the stack - the carry is losing "
                             "pixels. Do not use this result.")})
    b = A.get("boundary_steps") or {}
    if b.get("suspect_pixels"):
        n_px = int(np.prod(A["shape"])) if A.get("shape") else 0
        out.append({"code": "anchor_whole_cycle", "hard": False,
                    "text": (f"{b['pixels_with_any']} pixels ({100 * b['pixels_with_any'] / max(1, n_px):.2f}% "
                             f"of the grid) carry a suspected whole-cycle (27.7 mm) error from an anchor "
                             f"epoch into every later epoch (anchors {100 * b['boundary_rate']:.2f}% of "
                             f"samples vs {100 * b['control_rate']:.2f}% at ordinary epochs); see the "
                             "anchorSuspect layer.")})
    d = A.get("boundary_datum") or {}
    for key, where in (("flagged", "boundary"), ("flagged_at_anchor", "anchor")):
        for f in d.get(key) or ():
            if key == "flagged" and d.get("repaired"):
                continue
            mm = f["shift_mm"] if key == "flagged" else f["anchor_shift_mm"]
            date = f["date"] if key == "flagged" else f["anchor_date"]
            out.append({"code": "boundary_datum", "hard": False,
                        "text": f"datum shift of {mm:+.1f} mm at the {where} epoch {date}, carried into "
                                "every later epoch (not repaired)."})
    j = A.get("jumps") or {}
    if j.get("fraction", 0) > 0.02:
        out.append({"code": "unwrap_spikes_high", "hard": False,
                    "text": f"{100 * j['fraction']:.1f}% of samples were rejected as unwrapping spikes; "
                            "suspect the unwrapping itself."})
    return out


def check_stitch(A: dict, cfg) -> list[dict]:
    """Refuse a stack whose stitching is known to be wrong; return the
    problems otherwise (soft ones are recorded, not raised)."""
    probs = A["stitch_problems"] = stitch_problems(A, cfg)      # with this cfg's acceptances
    hard = [p for p in probs if p["hard"]]
    if hard:
        raise RuntimeError("DISP-S1 stitching is not usable:\n  - "
                           + "\n  - ".join(p["text"] for p in hard))
    for p in probs:
        log.warning("Stitching: %s", p["text"])
    return probs


def refuse_foreign_mintpy(cfg) -> None:
    """
    Refuse to write DISP-S1 products into a mintpy/ folder that holds
    another route's products.

    Both DISP routes write timeseries.h5, temporalCoherence.h5 and
    maskTempCoh.h5 at the mintpy root ("w"), and the chain deletes
    geo/geo_*.h5 before moving its own in. In a stack or phase-linked
    project those are the official products: a DISP screening made from the
    dashboard under the same project name would delete them and leave a
    client report built from screening data in their place. Only this
    project's own mintpy_dir is checked - disp_target_grid_file and
    disp_weather_dir legitimately point into other projects.
    """
    import h5py
    M = Path(cfg.mintpy_dir)
    why = None
    if (M / "inputs" / "ifgramStack.h5").exists():
        why = "it holds an interferogram stack (inputs/ifgramStack.h5)"
    elif (M / "run_products.json").exists():
        why = "it holds a MintPy run record (run_products.json) from the stack or phase-linked route"
    else:
        gv = M / "geo" / "geo_velocity.h5"
        if gv.exists():
            try:
                with h5py.File(gv, "r") as h:
                    proc = h.attrs.get("PROCESSOR", "")
                proc = proc.decode() if isinstance(proc, bytes) else str(proc)
            except OSError:
                proc = "unreadable"
            if proc != "opera_disp":
                why = f"its geo/geo_velocity.h5 was made by {proc or 'an unnamed processor'}, not OPERA DISP-S1"
    if why:
        raise RuntimeError(f"Refusing to write DISP-S1 products into {M}: {why}. DISP screening must use "
                           "its own project_dir (e.g. <name>_disp_screening); nothing was changed.")


def mean_layer(A: dict, name: str):
    """Stack mean of one quality layer from assemble_stack's sampled sums."""
    if name not in A["sums"]:
        return None
    n = A["counts_layer"][name]
    out = np.full(A["shape"], np.nan, dtype="float32")
    np.divide(A["sums"][name], np.maximum(n, 1), out=out, where=n > 0)
    return out


def build_stack(cfg: PipelineConfig, results) -> Path:
    """
    Assemble the granules into one continuous, AOI-subset time series and
    fit a rate to it directly.

    Returns the path to a MintPy-format directory so the existing
    deliverables stage works unchanged.
    """
    import h5py

    out_dir = cfg.mintpy_dir
    refuse_foreign_mintpy(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    A = assemble_stack(cfg, results)
    check_stitch(A, cfg)          # refuses a partly corrected / ratcheted / unanchored stack
    ts, dates, flat, finite, counts = A["ts"], A["dates"], A["flat"], A["finite"], A["counts"]
    valid_fraction, shape = A["valid_fraction"], A["shape"]
    x0, y0, dx, dy, epsg = A["x0"], A["y0"], A["dx"], A["dy"], A["epsg"]
    sums, counts_layer = A["sums"], A["counts_layer"]
    incidence_deg, acq_time, frame_id = A["incidence_deg"], A["acq_time"], A["frame_id"]

    # --- reference point -------------------------------------------------
    # Every displacement is relative to one pixel. MintPy does this for the
    # other backends; DISP-S1 arrives referenced to its own frame-wide
    # solution, so without this step the two backends are not measuring
    # from the same place and will not agree.
    ref_rc = None
    if cfg.reference_lalo:
        from pyproj import Transformer
        lat, lon = [float(v) for v in cfg.reference_lalo.split(",")]
        tr = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
        rx, ry = tr.transform(lon, lat)
        c = int(round((rx - x0) / dx)); r = int(round((ry - y0) / dy))
        if not (0 <= r < ts.shape[1] and 0 <= c < ts.shape[2]):
            raise RuntimeError(
                f"reference_lalo {cfg.reference_lalo} falls outside the AOI grid")
        n_ok = int(np.isfinite(ts[:, r, c]).sum())
        if n_ok < 0.9 * ts.shape[0]:
            raise RuntimeError(
                f"reference_lalo {cfg.reference_lalo} is valid in only "
                f"{n_ok}/{ts.shape[0]} epochs - pick a pixel that survives "
                f"the stack, or the reference itself carries the noise")
        ref_rc = (r, c)
        if n_ok < ts.shape[0]:
            log.warning("Reference pixel is missing in %d epoch(s); its "
                        "series is interpolated there. Subtracting a NaN "
                        "would blank the whole epoch for every pixel.",
                        ts.shape[0] - n_ok)
        log.info("Reference point %s -> row %d col %d, valid in %d/%d epochs",
                 cfg.reference_lalo, r, c, n_ok, ts.shape[0])
    else:
        # Fall back to a well-observed pixel that is both quiet AND not
        # visibly moving.
        #
        # Two mistakes have been made here, in order:
        #
        #   1. Requiring a pixel valid in EVERY epoch. On a real stack -
        #      40 epochs over farmland, conncomp rejecting ~58% of the
        #      grid per granule - none qualifies, and the run silently
        #      produced no reference at all.
        #
        #   2. Ranking candidates by residual scatter about a straight
        #      line ALONE. That does not measure stability, it measures
        #      smoothness: a pixel subsiding at a steady 10 mm/yr has
        #      almost no residual scatter and wins. Subtracting its series
        #      then shifts EVERY velocity in the product by +10 mm/yr,
        #      uniformly and invisibly - correlation with another backend
        #      is unaffected, so nothing internal to the product reveals
        #      it, and the whole field can change sign.
        #
        # So: keep the quiet half (a noisy reference injects its noise
        # everywhere), then among those take the pixel closest to zero
        # rate. That is still an assumption - it assumes SOMETHING in the
        # AOI is stable - and it is the assumption GNSS exists to replace.
        best_count = int(counts.max()) if counts.size else 0
        floor = max(int(0.9 * ts.shape[0]), int(0.9 * best_count), 3)
        cand = np.where(counts >= floor)[0]
        if cand.size:
            tt = np.array([(datetime.strptime(d, "%Y%m%d")
                            - datetime.strptime(dates[0], "%Y%m%d")).days / 365.25
                           for d in dates])
            scatter = np.full(cand.size, np.inf)
            rate = np.full(cand.size, np.inf)
            for k, j in enumerate(cand):
                m = finite[:, j]
                A = np.vstack([tt[m], np.ones(m.sum())]).T
                sol, *_ = np.linalg.lstsq(A, flat[m, j], rcond=None)
                rate[k] = sol[0]
                scatter[k] = np.std(flat[m, j] - A @ sol)
            quiet = scatter <= np.median(scatter)
            pool = np.where(quiet)[0]
            best = int(pool[np.argmin(np.abs(rate[pool]))])
            ref_rc = tuple(int(v) for v in
                           np.unravel_index(cand[best], ts.shape[1:]))
            spread = float(np.percentile(rate[pool], 84)
                           - np.percentile(rate[pool], 16)) / 2
            log.warning("No reference_lalo set. Auto-selected row %d col %d "
                        "from %d pixels valid in at least %d of %d epochs: "
                        "the one nearest zero rate (%.2f mm/yr) among the "
                        "quieter half (residual scatter %.1f mm).",
                        ref_rc[0], ref_rc[1], cand.size, floor, ts.shape[0],
                        float(rate[best]) * 1000, float(scatter[best]) * 1000)
            log.warning("Candidate rates spread +/-%.2f mm/yr, so the choice "
                        "of reference moves every velocity in this product by "
                        "about that much. Nothing has argued any of these "
                        "pixels is actually stable - that is what a GNSS tie "
                        "would settle.", spread * 1000)
        else:
            log.error("No pixel is valid in even %d of %d epochs, so no "
                      "reference point could be chosen. The series is "
                      "relative to OPERA's own frame reference, which is not "
                      "a defensible basis for a deliverable.",
                      floor, ts.shape[0])

    if ref_rc is not None:
        ref_series = ts[:, ref_rc[0], ref_rc[1]].astype("float64").copy()
        # A NaN in the reference series would subtract into every pixel of
        # that epoch and blank it - the same class of mistake as the carry
        # bug above, so fill the gaps rather than propagate them.
        gap = ~np.isfinite(ref_series)
        if gap.any():
            tt = np.arange(ts.shape[0], dtype="float64")
            ref_series[gap] = np.interp(tt[gap], tt[~gap], ref_series[~gap])
        ts = ts - ref_series[:, None, None].astype("float32")
        flat = ts.reshape(ts.shape[0], -1)
        finite = np.isfinite(flat)
        counts = finite.sum(axis=0)

    # Years since the first epoch. Needed by the tropospheric check below
    # as well as by the rate fit, so it is computed once here rather than
    # inside the fit.
    t = np.array([(datetime.strptime(d, "%Y%m%d")
                   - datetime.strptime(dates[0], "%Y%m%d")).days / 365.25
                  for d in dates])

    # --- per-pixel geometry ----------------------------------------------
    # The displacement granule states only a near- and far-range incidence
    # angle for the whole frame. Per-pixel line-of-sight vectors live in
    # the separate DISP-S1-STATIC product, and they are what makes a
    # vertical projection or a GNSS comparison meaningful.
    grid = {"x0": x0, "y0": y0, "dx": dx, "dy": dy,
            "ny": ts.shape[1], "nx": ts.shape[2], "epsg": epsg}
    geom = {}
    if cfg.disp_use_static_layers:
        from . import static_layers
        try:
            # The DEM is needed for the tropospheric correction and for
            # the `height` field of the point product, so fetch it either way.
            geom = static_layers.load(cfg, grid, frame=frame_id,
                                      want_dem=True)
        except Exception as exc:  # noqa: BLE001
            log.warning("Could not load DISP-S1-STATIC layers (%s) - falling "
                        "back to the frame-average incidence angle", exc)
    else:
        log.warning("disp_use_static_layers is false - no per-pixel geometry, "
                    "so any vertical projection uses one angle for the whole "
                    "AOI.")


    tropo_info = {"applied": False}
    if cfg.disp_tropo_correction:
        tropo_info = _apply_tropo(cfg, ts, dates, t, acq_time, grid, geom,
                                  ref_rc)

    # --- velocity --------------------------------------------------------

    # One threshold, used for both the velocity fit and the quality layer.
    # Previously these were 25% and 90%, so the velocity field extended
    # well beyond the pixels the quality layer admitted, and the summary
    # statistics described a different set of pixels from the raster.
    min_frac = cfg.disp_min_valid_fraction
    min_epochs = max(3, int(round(min_frac * ts.shape[0])))
    usable = counts >= min_epochs

    seasonal = cfg.disp_seasonal_model
    accel = cfg.disp_acceleration_model

    # Drop optional terms the stack cannot support rather than fitting an
    # underdetermined model. With four parameters and three epochs the
    # least-squares is singular, every coefficient comes back NaN, and the
    # run still reports "762 pixels complete" before writing an entirely
    # empty product - the failure is invisible until someone opens the
    # raster. Degrade, and say so.
    n_ep = ts.shape[0]
    if accel and n_ep < 6:
        log.warning("Acceleration needs more than %d epochs to be estimable "
                    "alongside a rate and a seasonal term - dropping it.", n_ep)
        accel = False
    if seasonal and n_ep < 5:
        log.warning("A seasonal term needs more than %d epochs - dropping it. "
                    "The rate is still fitted.", n_ep)
        seasonal = False
    A_full = _design(t, seasonal, accel)
    if A_full.shape[0] <= A_full.shape[1]:
        raise RuntimeError(
            f"{n_ep} epochs cannot support even a linear fit "
            f"({A_full.shape[1]} parameters). Widen the date range or lower "
            f"disp_max_per_year.")
    IDX = _term_index(seasonal, accel)
    span = float(t[-1] - t[0])
    if seasonal and span < 2.0:
        log.warning("Seasonal terms requested but the stack spans only %.1f "
                    "years - an annual term is barely identifiable and will "
                    "trade against the rate. Consider disp_seasonal_model: "
                    "false.", span)
    if accel and span < 3.0:
        log.warning("Acceleration requested over only %.1f years. A steady "
                    "trend and a gentle curve are nearly indistinguishable "
                    "over a short record, so both error bars inflate and "
                    "neither coefficient means much.", span)

    npix = flat.shape[1]
    def _blank():
        return np.full(npix, np.nan, dtype="float32")

    # The EGMS per-point field set, all read off one solve per pixel.
    vel, vel_std = _blank(), _blank()
    seas_amp, seas_std = _blank(), _blank()
    acc, acc_std = _blank(), _blank()
    rmse = _blank()

    def _unpack(sol, sigmas, res, sel):
        vel[sel] = sol[IDX["rate"]]
        vel_std[sel] = sigmas[IDX["rate"]]
        rmse[sel] = res
        if IDX["sin"] is not None:
            si, ci = IDX["sin"], IDX["cos"]
            amp = np.hypot(sol[si], sol[ci])
            seas_amp[sel] = amp
            # Amplitude is sqrt(a^2+b^2); propagate both coefficients'
            # errors through it rather than quoting one of them.
            with np.errstate(invalid="ignore", divide="ignore"):
                seas_std[sel] = np.where(
                    amp > 0,
                    np.sqrt((sol[si] * sigmas[si]) ** 2
                            + (sol[ci] * sigmas[ci]) ** 2) / np.maximum(amp, 1e-12),
                    np.nan)
        if IDX["accel"] is not None:
            # A quadratic coefficient c in c*t^2 is half the acceleration.
            acc[sel] = 2.0 * sol[IDX["accel"]]
            acc_std[sel] = 2.0 * sigmas[IDX["accel"]]

    complete = usable & (counts == ts.shape[0])
    inflation = 1.0
    if complete.any():
        y = flat[:, complete]
        sol0, *_ = np.linalg.lstsq(A_full, y, rcond=None)
        inflation = _noise_inflation(y - A_full @ sol0)
        log.info("Residual lag-1 autocorrelation implies a %.2fx widening of "
                 "every rate error bar for temporally correlated noise",
                 inflation)
        if inflation > 2.5:
            log.warning("That inflation is large. Atmospheric delay "
                        "decorrelates within days, so strongly correlated "
                        "residuals on a stack sampled this coarsely usually "
                        "mean unmodelled STRUCTURE - a residual seasonal "
                        "term, a noisy reference pixel, or steps left by the "
                        "stitching - rather than genuine noise colour. The "
                        "error bars are widened either way, which is the "
                        "safe direction, but the cause is worth finding.")
        _unpack(*_fit(A_full, y, inflation), complete)

    # Pixels with gaps need their own fit, since the design matrix differs
    # per pixel.
    partial = np.where(usable & ~complete)[0]
    for j in partial:
        m = finite[:, j]
        Aj = _design(t[m], seasonal, accel)
        if Aj.shape[0] <= Aj.shape[1]:
            continue
        _unpack(*_fit(Aj, flat[m, j], inflation), j)

    model = "linear" + (" + annual" if seasonal else "") + \
            (" + acceleration" if accel else "")
    log.info("Velocity fit (%s) at >=%.0f%% valid epochs (%d of %d): %d "
             "pixels complete, %d with gaps, %d dropped",
             model, min_frac * 100, min_epochs, ts.shape[0],
             int(complete.sum()), len(partial), int((~usable).sum()))

    # Report the uncertainty over the pixels that were FIT, before any
    # quality threshold is applied. Quoting it after filtering on a
    # quality metric that correlates with it would be selection on the
    # outcome - it would describe the pixels whose noise happened to be
    # small and call that the product's precision.
    finite_sig = vel_std[np.isfinite(vel_std)]
    if finite_sig.size:
        log.info("Rate uncertainty (1 sigma) over all %d fitted pixels, "
                 "BEFORE any quality gate: median %.2f mm/yr, 90th "
                 "percentile %.2f mm/yr. Formal error only - it excludes the "
                 "reference point, the troposphere, and unwrapping bias, and "
                 "is a floor rather than the uncertainty.",
                 finite_sig.size,
                 float(np.median(finite_sig)) * 1000,
                 float(np.percentile(finite_sig, 90)) * 1000)
        sig95 = np.abs(vel) > 1.96 * vel_std
        log.info("Rates distinguishable from zero at 95%%: %d of %d fitted "
                 "pixels (%.0f%%)", int(np.nansum(sig95)), finite_sig.size,
                 100 * float(np.nansum(sig95)) / max(finite_sig.size, 1))
    if seasonal:
        fa = seas_amp[np.isfinite(seas_amp)]
        if fa.size:
            log.info("Fitted annual amplitude: median %.1f mm, 90th "
                     "percentile %.1f mm",
                     float(np.median(fa)) * 1000,
                     float(np.percentile(fa, 90)) * 1000)
    if accel:
        fc = acc[np.isfinite(acc)]
        if fc.size:
            log.info("Fitted acceleration: median %+.2f mm/yr^2, 90th "
                     "percentile magnitude %.2f",
                     float(np.median(fc)) * 1000,
                     float(np.percentile(np.abs(fc), 90)) * 1000)

    shape2d = ts.shape[1:]
    vel = vel.reshape(shape2d)
    vel_std = vel_std.reshape(shape2d)
    seas_amp = seas_amp.reshape(shape2d)
    seas_std = seas_std.reshape(shape2d)
    acc = acc.reshape(shape2d)
    acc_std = acc_std.reshape(shape2d)
    rmse = rmse.reshape(shape2d)

    # --- quality ---------------------------------------------------------
    def _mean_layer(name):
        if name not in sums:
            return None
        n = counts_layer[name]
        out = np.full(shape, np.nan, dtype="float32")
        np.divide(sums[name], np.maximum(n, 1), out=out, where=n > 0)
        return out

    coherence = _mean_layer(COHERENCE_LAYER)
    if coherence is None:
        coherence = _mean_layer(QUALITY_LAYER)
        log.warning("No %s layer in these granules - falling back to %s, "
                    "which is not the same quantity as MintPy's temporal "
                    "coherence. Say so in any cross-backend comparison.",
                    COHERENCE_LAYER, QUALITY_LAYER)
    if coherence is None:
        coherence = np.where(np.isfinite(vel), 1.0, np.nan).astype("float32")
    else:
        log.info("Stack-mean %s: median %.2f over %d finite pixels",
                 COHERENCE_LAYER if COHERENCE_LAYER in sums else QUALITY_LAYER,
                 float(np.nanmedian(coherence)), int(np.isfinite(coherence).sum()))

    quality = np.where(usable.reshape(ts.shape[1:]), coherence, np.nan)

    water = _mean_layer(WATER_LAYER)
    ps = _mean_layer(PS_LAYER)
    recommended = _mean_layer(MASK_LAYER)

    # "Observable" means ground this sensor actually saw at least once -
    # land, not in layover or shadow, and measured in at least one epoch.
    # It deliberately does NOT include the quality threshold, because the
    # point of the second coverage figure is to be independent of it.
    # Defining it as isfinite(velocity), as this once did, made the two
    # figures identical by construction and the quality figure meaningless.
    observable = np.isfinite(coherence) & (valid_fraction > 0)
    # Layover and shadow are not low quality, they are unmeasurable, so
    # they come out of the coverage denominator rather than counting as
    # failures.
    if "layover_shadow" in geom:
        observable &= ~(geom["layover_shadow"] > 0)
    if water is not None:
        # OPERA's water_mask is 1 where the pixel is LAND, 0 where it is
        # water - it is a keep-mask, not a reject-mask. Verified against a
        # wholly inland AOI, where it reads 1.0 across the entire grid.
        # Inverting this silently masks everything.
        observable &= ~(water < 0.5)    # NaN is unknown, so keep it
        log.info("water_mask excludes %.1f%% of the AOI grid as water",
                 100 * float(np.nanmean(water < 0.5)))

    if recommended is not None:
        if cfg.disp_use_recommended_mask:
            kept = float(np.nanmean(recommended > 0.5))
            log.info("OPERA recommended_mask (majority over the stack) keeps "
                     "%.1f%% of the grid", kept * 100)
            quality = np.where(recommended > 0.5, quality, np.nan)
        else:
            log.warning("recommended_mask NOT applied (disp_use_recommended_mask "
                        "is false) - pixels OPERA flagged as unreliable are "
                        "retained. Note that in any report.")

    if ps is not None:
        log.info("persistent_scatterer_mask: %d pixels are point-like in more "
                 "than half the stack (%.1f%% of the grid) - this is the "
                 "layer to quote against a PS vendor's point count",
                 int(np.nansum(ps > 0.5)), 100 * float(np.nanmean(ps > 0.5)))

    # --- write -----------------------------------------------------------
    attrs = {
        "FILE_TYPE": "timeseries", "PROCESSOR": "opera_disp",
        # MintPy's X_FIRST/Y_FIRST are the OUTER corner of the first pixel;
        # x0/y0 are its centre (the convention static_layers and tropo_disp
        # read the grid in).
        "X_FIRST": str(x0 - dx / 2), "Y_FIRST": str(y0 - dy / 2),
        "X_STEP": str(dx), "Y_STEP": str(dy),
        "X_UNIT": "meters", "Y_UNIT": "meters",
        "EPSG": str(epsg), "WAVELENGTH": "0.05546576",
        "ORBIT_DIRECTION": cfg.orbit_direction.upper(),
        "UNIT": "m", "REF_DATE": dates[0],
    }
    track = None
    for p in results:
        track = p.properties.get("pathNumber") or p.properties.get("relativeOrbit")
        if track:
            break
    if track is None:
        track = cfg.track
    if track is not None:
        attrs["trackNumber"] = str(track)
    if incidence_deg is None:
        incidence_deg = NOMINAL_IW_INCIDENCE_DEG
        attrs["INCIDENCE_ANGLE_SOURCE"] = (
            "nominal Sentinel-1 IW mid-swath - the granule does not state it, "
            "and exact per-pixel geometry lives in the CSLC-S1-STATIC product")
        log.warning("DISP-S1 granules carry no incidence angle; assuming "
                    "%.1f deg (nominal IW mid-swath). Load CSLC-S1-STATIC "
                    "line-of-sight layers before quoting vertical motion.",
                    NOMINAL_IW_INCIDENCE_DEG)
    else:
        attrs["INCIDENCE_ANGLE_SOURCE"] = "granule metadata"
    if "incidence_deg" in geom:
        incidence_deg = float(np.nanmean(geom["incidence_deg"]))
        attrs["INCIDENCE_ANGLE_SOURCE"] = (
            "DISP-S1-STATIC line_of_sight_enu, AOI mean of the per-pixel field")
    attrs["INCIDENCE_ANGLE"] = str(round(float(incidence_deg), 2))
    if frame_id is not None:
        attrs["FRAME_ID"] = str(frame_id)
    attrs["RATE_MODEL"] = model
    if tropo_info.get("applied"):
        attrs["TROPO_CORRECTION"] = (
            f"{cfg.weather_model} via RAiDER"
            + ("" if tropo_info.get("verified") else " (UNVERIFIED)"))
    else:
        attrs["TROPO_CORRECTION"] = (
            f"none ({tropo_info.get('reason', 'not requested')})")
    attrs["RATE_SIGMA_INFLATION"] = str(round(inflation, 3))
    if ref_rc is not None:
        attrs["REF_Y"], attrs["REF_X"] = str(ref_rc[0]), str(ref_rc[1])
        attrs["REF_LALO"] = (cfg.reference_lalo
                             or "auto (quietest well-observed pixel)")

    def _write(path, name, data, extra=None):
        with h5py.File(path, "w") as f:
            f.create_dataset(name, data=data)
            for k, v in {**attrs, **(extra or {})}.items():
                f.attrs[k] = v

    _write(out_dir / "timeseries.h5", "timeseries", ts,
           {"FILE_TYPE": "timeseries"})
    with h5py.File(out_dir / "timeseries.h5", "a") as f:
        f.create_dataset("date", data=np.array(dates, dtype="S8"))
    with h5py.File(out_dir / "velocity.h5", "w") as f:
        # Names follow the EGMS per-point schema wherever one exists, so
        # the product is directly comparable to a public operational
        # service rather than to a vendor convention nobody can audit.
        # MintPy's own names are kept alongside for the existing readers.
        f.create_dataset("velocity", data=vel)                  # m/yr
        # The error bar travels with the number it belongs to. A rate
        # quoted without one cannot be reviewed.
        f.create_dataset("velocityStd", data=vel_std)
        f.create_dataset("rmse", data=rmse)                     # m
        if seasonal:
            f.create_dataset("annualAmplitude", data=seas_amp)
            f.create_dataset("annualAmplitudeStd", data=seas_std)
        if accel:
            f.create_dataset("acceleration", data=acc)          # m/yr^2
            f.create_dataset("accelerationStd", data=acc_std)
        for k, v in {**attrs, "FILE_TYPE": "velocity"}.items():
            f.attrs[k] = v
    _write(out_dir / "temporalCoherence.h5", "temporalCoherence", quality,
           {"FILE_TYPE": "temporalCoherence"})
    _write(out_dir / "maskTempCoh.h5", "mask",
           (np.nan_to_num(quality) >= cfg.min_temporal_coherence),
           {"FILE_TYPE": "mask"})

    # Diagnostics. `validEpochFraction` is what makes a coverage ratchet
    # visible after the fact; `maskObservable` is the denominator the
    # coverage figure should use - land that could be measured, rather
    # than the whole AOI rectangle.
    _write(out_dir / "validEpochFraction.h5", "validEpochFraction",
           valid_fraction.astype("float32"), {"FILE_TYPE": "mask"})
    # "Observable" means ground this sensor actually saw at least once -
    # land, and measured in at least one epoch. It deliberately does NOT
    # include the quality threshold, because the point of the second
    # coverage figure is to be independent of it. Defining it as
    # isfinite(velocity), as this once did, made the two figures identical
    # by construction and the quality figure meaningless.
    _write(out_dir / "maskObservable.h5", "mask", observable,
           {"FILE_TYPE": "mask"})
    log.info("Observable (land, non-layover, seen at least once): %d of %d "
             "cells (%.1f%%)", int(observable.sum()), observable.size,
             100 * float(observable.mean()))
    if ps is not None:
        _write(out_dir / "persistentScatterer.h5", "persistentScatterer",
               ps.astype("float32"), {"FILE_TYPE": "mask"})

    if "los_up" in geom:
        with h5py.File(out_dir / "geometryGeo.h5", "w") as f:
            # `height` is in the EGMS schema and is otherwise fetched and
            # thrown away - it is only loaded when the tropospheric
            # correction asks for the DEM.
            for name in ("los_east", "los_north", "los_up", "incidence_deg",
                         "layover_shadow", "height"):
                if name in geom:
                    f.create_dataset(name, data=geom[name].astype("float32"))
            for k, v in {**attrs, "FILE_TYPE": "geometry"}.items():
                f.attrs[k] = v
        log.info("Wrote %s (per-pixel line-of-sight geometry)",
                 out_dir / "geometryGeo.h5")

    log.info("Wrote MintPy-format products to %s", out_dir)
    return out_dir


def run(cfg: PipelineConfig) -> Path:
    refuse_foreign_mintpy(cfg)          # before the search: nothing to fetch for a refused run
    results = search(cfg)
    if not results:
        raise RuntimeError("No DISP-S1 coverage for this AOI and date range")
    results = subsample(results, cfg.disp_max_per_year)
    return build_stack(cfg, results)
