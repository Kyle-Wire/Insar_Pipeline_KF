"""NISAR backend: L2 GUNW interferograms from ASF, read as windows, for MintPy.

Rewritten 8 Oct 2026 (M2 of review_2026-10-08/PLAN.md; defects N1-N12 in
review_2026-10-08/gaps/notes.md section 2). The route as first wired would
not have run: it searched BETA beside PROVISIONAL, grouped granules by frame
instead of track, downloaded every 2.0-2.5 GB granule whole, staged no DEM,
never applied the ionosphere screen, and left the products on a UTM grid that
significance.py reads as degrees.

WHAT IT DOES NOW, IN ORDER
--------------------------
1. Search PROVISIONAL only (NISAR_L2_GUNW_PROVISIONAL_V1). BETA ("not fully
   calibrated", Oct 2025 - Jan 2026) is never mixed in; ASF: "exercise caution
   if you analyze BETA products together with PROVISIONAL products".
2. Keep ONE geometry: the config's `track` and `orbit_direction`. The track is
   the 6th name field (NISAR_L2_PR_GUNW_<cycle>_<TRACK>_<A|D>_<frame>_...);
   the old code keyed on (direction, frame) and merged T076A with T148A at
   Site A.
3. Drop the calibration-mode frames ASF lists as unusable before August 2026
   acquisitions (161/174, 161/175, 169/090, 169/091).
4. One granule per date pair. Four kinds of duplicate exist (dup_check,
   sensors/notes.md 1.2; sensors_check/dup_order_check_out.txt):
     * full vs partial frame (Site A T148A: every pair twice);
     * the same pair reprocessed (product counter 001 vs 002);
     * the same pair in two range-bandwidth modes (20 vs 40 MHz, both partial,
       where the acquisition mode changes: Appin/Tahmoor T167A F160);
     * the same pair in adjacent frames (Katy T062A F016/F017, x20).
   Candidates are ranked: coverage of the AOI (footprint), the stack's
   bandwidth mode, the stack's dominant frame, full over partial frame,
   highest counter, latest processing. The ranked list is kept: when the best
   candidate's DATA cover the AOI badly (incomplete L0B, a footprint that
   promised more than the raster holds), the next one is read instead.
5. Read only the AOI. Each kept granule is opened over HTTPS and only the
   chunks covering the AOI (+margin) are fetched (windowed HDF5 reads, as
   disp_backend and nisar_mining do). Nothing 2.4 GB is ever stored. The
   optional `download` fetch (nisar_fetch: download) crops on arrival and
   deletes the whole file before the next one.
6. Resample to a geographic grid on arrival. Every layer is put on one
   lat/lon lattice over the AOI at the product posting (80 m), by nearest
   neighbour (unwrapped phase must not be averaged across connected
   components), so every pair - whatever its frame or UTM zone - lands on the
   same grid, and significance.py, client_report and the harness read degrees
   as they expect. The radar-grid metadata cube (incidence, LOS vector, slant
   range, baselines, troposphere and tide phase) is re-gridded the same way
   (linear in the source projection, only the height levels the AOI's DEM
   needs). Each crop is written as a small GUNW-layout HDF5 that MintPy's own
   prep_nisar reads unchanged, with its provenance in the file attributes.
7. Screen each pair on arrival (its valid share of the AOI against the group
   median; a pair below half of it is read again from its next-ranked
   granule, else dropped), keep the longest unbroken chain of 12-day pairs,
   and write the chain QA record. Each pair's median coherence over the AOI
   is recorded, not used to drop it.
8. Stage a DEM (Copernicus GLO-30, ellipsoidal heights - the radar-grid cube
   is in heights above the ellipsoid) over the lattice.
9. Hand MintPy a loader block: processor nisar on the crops, the DEM, the
   product's polarisation, and the ionosphere screen applied by MintPy's
   split-spectrum step.

CONVENTIONS (sign and date order)
---------------------------------
NISAR GUNW pairs are in chronological order: identification/
referenceZeroDopplerStartTime is EARLIER than secondaryZeroDopplerStartTime
(checked on every crop; the old docstring's "opposite date order to ARIA" was
about ARIA's Sentinel-1 GUNW, whose reference is the later date). The phase is
reference x conj(secondary) with ISCE3's exp(-j 4 pi R / lambda), so a positive
unwrapped phase is a range INCREASE (motion away from the satellite).
prep_nisar writes the pair as reference_secondary and MintPy converts phase
with -lambda/(4 pi): subsidence comes out NEGATIVE, toward the satellite
positive - MintPy's convention for every route. No flip is applied. A granule
whose reference date is later than its secondary date (none seen) is put in
chronological order by negating every differential layer (unwrapped phase,
ionosphere, troposphere, tide, baselines) and swapping the dates; the crop
says so in its attributes and in nisar_stack.json.

IONOSPHERE
----------
The GUNW ships an ionospherePhaseScreen that is "not applied to the data
layers by default" (nisar-docs gunw). prep_nisar writes it to
inputs/ionStack.h5 and `mintpy.ionosphericDelay.method = split_spectrum`
makes MintPy invert it to a series with the same phase-to-range conversion as
the data and SUBTRACT it (iono_split_spectrum.py: diff.py timeseries ion).
That is right only if the screen is the ionospheric part OF the unwrapped
phase, with the same sign. The producer's code says it is (ISCE3, read 8 Oct
2026, develop branch; copies in review_2026-10-08/p23/nisar/isce3_sign/):
split_band_estimation.estimate_iono_low_high models each sub-band phase as
(f/f0) phi_nondispersive + (f0/f) phi_dispersive, so at the main band
phi_main = phi_nondispersive + phi_dispersive, and nisar/workflows/
ionosphere.py writes that `dispersive` array (filtered) to
ionospherePhaseScreen with no change of sign. Subtracting it is the
correction. The PROVISIONAL products use a different estimator from that
split-band one: the verified ASC-34 granule's metadata/processingInformation/
algorithms/ionosphereEstimation/ionosphereAlgorithm reads "Range split-spectrum
with sub-band interferograms from the main band and the difference of the
main and side band" (main_diff_ms_band). Its solver,
isce3/atmosphere/main_band_estimation.estimate_iono_main_diff, fits
phi_main = phi_dispersive + phi_nondispersive (first row of its coefficient
matrix is [1, 1]), and so does estimate_iono_main_side: all three ISCE3
estimators define the screen as the dispersive part OF the main-band phase,
same sign (independent review, 8 Oct 2026; copy and sha256 in
review_2026-10-08/p23/nisar/isce3_sign_review/). Each crop records the
product's ionosphereAlgorithm. prep_nisar copies both layers unchanged (real
ASC-34 crop: inputs/ifgramStack.h5 unwrapPhase and inputs/ionStack.h5 equal
the crop's unwrappedPhase and ionospherePhaseScreen exactly, 126,082 cells).
Every crop also measures it as a guard: over the AOI's reliable
cells, subtracting a screen that is part of the phase lowers the phase
variance and the two correlate positively. nisar_stack.json reports the share
of pairs where it does; only strong, repeated evidence of the opposite stops
the run (summarise_ionosphere_sign), and nothing ever flips a sign silently.
The pipeline's own GIM ionosphere (corrections.py, `ionosphere_correction`)
is refused on this route: it would correct the ionosphere twice.

LINE OF SIGHT
-------------
The GUNW's losUnitVectorX/Y are the east and north components of the
target-to-sensor vector. MintPy 1.6.4's prep_nisar turns them into an
azimuthAngle 90 degrees off MintPy's own convention (found on the first real
crop; los_encoding_for_prep_nisar). Each crop therefore stores them encoded
so that the installed prep_nisar writes the right azimuth, found by calling
prep_nisar itself on test vectors; the product's own components are kept in
the crop as losUnitVectorEast/North.

WAVELENGTH
----------
MintPy uses one WAVELENGTH for a stack (prep_nisar reads the first file). A
pair taken in the other bandwidth mode (a different centre frequency) has its
phase layers rescaled to the stack's wavelength so its range change is exact
(harmonise_wavelength); nisar_stack.json lists them.

TROPOSPHERE AND TIDE
--------------------
Troposphere: ERA5 through MintPy/PyAPS, as every other route
(`mintpy_tropo_method`). The GUNW's own troposphere screens reach
inputs/tropoStack.h5 but MintPy 1.6.4 does not use them. Solid-earth tide:
MintPy's correct_SET step (`solid_earth_tides`); the GUNW's tide layer reaches
inputs/setStack.h5 and is likewise unused.

CHAIN NETWORK
-------------
"Only nearest-neighbor (12-day) GUNW products are generated" (nisar-docs).
A chain has no loops: MintPy's temporal coherence is 1.0 by construction (no
test) and phase closure is not applicable (no triangles), so the closure step
is never requested (`mintpy_unwrap_error_method` is passed without
phase_closure). Reliability is gated on interferometric coherence instead:
nisar_gunw/coherence_gate.tif holds the mean coherence over the pairs used and
the 0/1 gate at the config's coherence floor (min_temporal_coherence, 0.4 -
protocol B1's NISAR reliability rule). One missing pair splits a chain;
time_series refuses a split network, so the longest unbroken segment is kept
and the rest reported (nisar_chain_gap: fail stops instead).

WIRED ELSEWHERE (phase 2/3 finishing pass, 8 Oct 2026)
------------------------------------------------------
* `insar cleanup` (scripts/cleanup.py) offers cleanup_targets() below on a
  nisar project (level 2 only once mintpy/inputs/ifgramStack.h5 exists);
  `python -m insar_pipeline.nisar_backend cleanup CFG --level N [--delete]`
  does the same on its own (dry run unless --delete).
* config.py declares the nisar_* settings (SETTINGS below holds their
  defaults; a test keeps the two equal) and checks their values.
* deliverables.geometry_field reads mintpy/inputs/geometryGeo.h5 on this
  route, and the validation harness takes the mean pair coherence of
  coherence_gate.tif as its reliability quantity (protocol_B1).

QA, FIGURES AND REPORT (26 Oct 2026)
------------------------------------
The products stage runs the QA, figures and client report on this route
too (pipeline.py). Every client page names NISAR (config.sensor_name), flags
the run as a comparison from PROVISIONAL, unvalidated products, and credits
NASA-ISRO via ASF rather than the Copernicus Sentinel notice (credits.py).
"""
from __future__ import annotations

import contextlib
import functools
import json
import logging
import math
import re
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from .config import PipelineConfig

log = logging.getLogger(__name__)

# ---------------------------------------------------------------- collections
COLLECTION_PROVISIONAL = "NISAR_L2_GUNW_PROVISIONAL_V1"
COLLECTION_BETA = "NISAR_L2_GUNW_BETA_V1"
# PROVISIONAL only. Until 8 Oct 2026 this list also held BETA (and a plain
# NISAR_L2_GUNW_V1 that does not exist in CMR), and run() searched it.
GUNW_COLLECTIONS = [COLLECTION_PROVISIONAL]
GUNW_COLLECTIONS_CALIBRATED = [COLLECTION_PROVISIONAL]

# ASF PROVISIONAL known issues (30 Jul 2026): calibration-mode frames are "not
# usable" until August 2026 acquisitions.
CALIBRATION_MODE_FRAMES = {(161, 174), (161, 175), (169, 90), (169, 91)}
CALIBRATION_MODE_USABLE_FROM = "20260801"

# Optional settings: config.py declares them with these defaults (read with getattr, so a bare
# namespace works too).
SETTINGS = {
    "nisar_allow_beta": False,            # True: search BETA too (never mixed into a deliverable)
    "nisar_polarization": "",             # "" = the product's own co-pol (HH, else VV)
    "nisar_ionosphere": "split_spectrum",  # or "no"
    "nisar_chain_gap": "longest_segment",  # or "fail"
    "nisar_fetch": "window",              # or "download" (crop on arrival, then delete)
    # the grid the crops are put on: the 15 m posting every Sentinel-1 product
    # has (areas.POSTING_M), filled nearest-neighbour as MintPy's geocode fills
    # theirs, so a NISAR run's maps, KMZ, figures and viewer cells match the
    # Sentinel-1 run's cell for cell (26 Oct 2026, Kyle). The product's own
    # posting stays 80 m (resolution.NISAR_POSTING_M): each value still
    # averages ~80 m, which the products state.
    "nisar_posting_m": 15.0,
    "nisar_crop_margin_deg": 0.02,        # around the AOI, on every side
    "nisar_phase_linked": False,          # each crop's phase from dolphin on the 5 m GSLCs (nisar_pl.py)
    "nisar_pl_window": 11,                # its phase-linking window (the first test site, Gate 2)
    "nisar_min_valid_ratio": 0.5,         # a pair below this share of the group's median valid area is dropped
}

GRIDS = "science/LSAR/GUNW/grids/frequencyA/"
UNW = GRIDS + "unwrappedInterferogram/"
RADARGRID = "science/LSAR/GUNW/metadata/radarGrid/"
IDENT = "science/LSAR/identification/"
PARAMS = "science/LSAR/GUNW/metadata/processingInformation/parameters/unwrappedInterferogram/frequencyA/"
# Which ISCE3 estimator made ionospherePhaseScreen (recorded per crop; module
# docstring, IONOSPHERE: every ISCE3 estimator gives the same sign)
IONO_ALGORITHM = "science/LSAR/GUNW/metadata/processingInformation/algorithms/ionosphereEstimation/ionosphereAlgorithm"
# Layers on the 80 m grid that are differential phase (negated if a pair is reversed)
PHASE_LAYERS = ("unwrappedPhase", "ionospherePhaseScreen")
GRID_LAYERS = ("unwrappedPhase", "coherenceMagnitude", "connectedComponents",
               "ionospherePhaseScreen", "ionospherePhaseScreenUncertainty")
# Radar-grid cube fields prep_nisar reads, plus the secondary slant range.
CUBE_FIELDS = ("referenceSlantRange", "incidenceAngle", "losUnitVectorX", "losUnitVectorY",
               "wetTroposphericPhaseScreen", "hydrostaticTroposphericPhaseScreen",
               "slantRangeSolidEarthTidesPhase", "perpendicularBaseline", "parallelBaseline")
CUBE_DIFFERENTIAL = ("wetTroposphericPhaseScreen", "hydrostaticTroposphericPhaseScreen",
                     "slantRangeSolidEarthTidesPhase", "perpendicularBaseline", "parallelBaseline")

C_M_S = 299792458.0
M_PER_DEG_LAT = 111_132.954
# Crop file format. 2 (8 Oct 2026): LOS vector re-encoded for prep_nisar
# (los_encoding_for_prep_nisar) and wavelength harmonisation attributes. A
# crop of another version is read again, never reused.
CROP_VERSION = 2


def setting(cfg, key):
    return getattr(cfg, key, SETTINGS[key]) if cfg is not None else SETTINGS[key]


# ---------------------------------------------------------------- granule names
_NAME_RE = re.compile(
    r"NISAR_L2_(?P<config>[A-Z]{2})_GUNW_(?P<ref_cycle>\d{3})_(?P<track>\d{3})_(?P<direction>[AD])_"
    r"(?P<frame>\d{3})_(?P<sec_cycle>\d{3})_(?P<mode>\d{4})_(?P<polcode>[A-Z]{2})_"
    r"(?P<ref_start>\d{8}T\d{6})_(?P<ref_end>\d{8}T\d{6})_(?P<sec_start>\d{8}T\d{6})_(?P<sec_end>\d{8}T\d{6})_"
    r"(?P<crid>[A-Z]\d{5})_(?P<flag1>[A-Z])_(?P<coverage>[FP])_(?P<flag2>[A-Z])_(?P<counter>\d{3})")


def parse_name(name: str) -> dict | None:
    """The fields of a GUNW granule name, or None when it is not one.

    NISAR_L2_PR_GUNW_030_034_A_020_031_4000_SH_<ref start>_<ref end>_<sec start>_
    <sec end>_P05023_N_F_J_001: track 034, ascending, frame 020, range
    bandwidth 40 MHz ('4000'), full frame ('F'), product counter 001. The
    letter after the CRID ('N'/'F') does not follow ORBIT_TYPE and its meaning
    is UNCONFIRMED (sensors/notes.md 1.2); it is recorded, never used.
    """
    m = _NAME_RE.search(str(name or ""))
    if not m:
        return None
    g = m.groupdict()
    return {**g, "track": int(g["track"]), "frame": int(g["frame"]), "counter": int(g["counter"]),
            "bandwidth_mhz": int(g["mode"][:2]), "ref_date": g["ref_start"][:8], "sec_date": g["sec_start"][:8],
            "full_frame": g["coverage"] == "F"}


# ---------------------------------------------------------------- search
def allowed_collections(cfg) -> list[str]:
    if bool(setting(cfg, "nisar_allow_beta")):
        log.warning("nisar_allow_beta is set: BETA granules (not fully calibrated) are searched as well. "
                    "They are never mixed with PROVISIONAL in one stack (the group is one maturity).")
        return [COLLECTION_PROVISIONAL, COLLECTION_BETA]
    return list(GUNW_COLLECTIONS)


def records_from_results(results, aoi_wkt: str, start: str | None = None, end: str | None = None) -> list[dict]:
    """asf_search results -> plain records with the name fields, the footprint's
    share of the AOI and the window filter (reference date >= start, secondary
    date <= end, both inclusive). Results whose name does not parse are
    dropped with a warning."""
    from shapely import wkt as shapely_wkt
    from shapely.geometry import shape
    aoi = shapely_wkt.loads(aoi_wkt)
    s0 = str(start or "").replace("-", "")[:8] or None
    e0 = str(end or "").replace("-", "")[:8] or None
    out = []
    for r in results:
        p = (r.get("properties") if isinstance(r, dict) else getattr(r, "properties", None)) or {}
        name = p.get("fileName") or p.get("sceneName") or ""
        f = parse_name(name)
        if f is None:
            log.warning("Not a GUNW name, skipped: %s", name)
            continue
        if (s0 and f["ref_date"] < s0) or (e0 and f["sec_date"] > e0):
            continue
        geom = r.get("geometry") if isinstance(r, dict) else getattr(r, "geometry", None)
        try:
            cov = float(shape(geom).intersection(aoi).area / aoi.area) if geom else float("nan")
        except Exception:                                    # noqa: BLE001 - a bad footprint is not fatal
            cov = float("nan")
        maturity = "BETA" if "BETA" in str(p.get("collectionName") or p.get("url") or "").upper() else "PROVISIONAL"
        out.append({"name": Path(name).stem if name.endswith(".h5") else name, "url": p.get("url"),
                    "coverage_aoi": cov, "processing_date": str(p.get("processingDate") or ""),
                    "maturity": maturity, **f})
    return out


def search_gunw(cfg: PipelineConfig, collections: list[str] | None = None) -> list[dict]:
    """GUNW granules over the AOI and window, as records (records_from_results).
    PROVISIONAL only unless nisar_allow_beta."""
    import asf_search as asf
    cols = collections or allowed_collections(cfg)
    results = list(asf.search(shortName=cols, intersectsWith=cfg.aoi_wkt, start=cfg.start_date or None,
                              end=cfg.end_date or None, maxResults=5000))
    recs = records_from_results(results, cfg.aoi_wkt, cfg.start_date, cfg.end_date)
    log.info("NISAR GUNW (%s): %d granules over the AOI, %d inside the window", ", ".join(cols),
             len(results), len(recs))
    return recs


def groups(records) -> dict:
    """{(track, 'A'|'D'): [records]} - one geometry each."""
    out: dict = {}
    for r in records:
        out.setdefault((int(r["track"]), r["direction"]), []).append(r)
    return out


def describe_groups(records) -> list[dict]:
    rows = []
    for (trk, d), rs in sorted(groups(records).items()):
        pairs = {(r["ref_date"], r["sec_date"]) for r in rs}
        rows.append({"track": trk, "direction": "ascending" if d == "A" else "descending",
                     "frames": sorted({r["frame"] for r in rs}), "granules": len(rs), "unique_pairs": len(pairs),
                     "first": min(p[0] for p in pairs), "last": max(p[1] for p in pairs),
                     "maturity": sorted({r["maturity"] for r in rs})})
    return rows


def select_group(cfg, records) -> list[dict]:
    """The records of the config's geometry (track + orbit_direction). Both are
    required: ascending and descending, or two ascending tracks, measure
    different projections of the motion and cannot share an inversion."""
    rows = describe_groups(records)
    listing = "; ".join(f"track {r['track']} {r['direction']} (frames {r['frames']}, {r['unique_pairs']} pairs "
                        f"{r['first']}..{r['last']})" for r in rows) or "none"
    trk = getattr(cfg, "track", None)
    direction = str(getattr(cfg, "orbit_direction", "") or "").lower()
    if trk is None or str(trk).strip() == "" or direction not in ("ascending", "descending"):
        raise ValueError("The NISAR route needs `track` and `orbit_direction` in the config: one geometry per "
                         f"project. Over this AOI and window: {listing}")
    d = "A" if direction == "ascending" else "D"
    chosen = groups(records).get((int(trk), d), [])
    if not chosen:
        raise ValueError(f"No PROVISIONAL GUNW on track {int(trk)} {direction} over this AOI and window. "
                         f"Available: {listing}")
    mats = {r["maturity"] for r in chosen}
    if len(mats) > 1:
        # one maturity per stack: keep PROVISIONAL
        chosen = [r for r in chosen if r["maturity"] == "PROVISIONAL"]
        log.warning("BETA and PROVISIONAL both on this track: only PROVISIONAL is kept (ASF: do not mix)")
    return chosen


def exclude_calibration_frames(records) -> tuple[list, list]:
    kept, dropped = [], []
    for r in records:
        if (r["track"], r["frame"]) in CALIBRATION_MODE_FRAMES and r["ref_date"] < CALIBRATION_MODE_USABLE_FROM:
            dropped.append({**_brief(r), "why": "calibration-mode frame, unusable before August 2026 "
                                                "acquisitions (ASF PROVISIONAL known issues)"})
        else:
            kept.append(r)
    return kept, dropped


def _brief(r) -> dict:
    return {"name": r["name"], "frame": r["frame"], "bandwidth_mhz": r["bandwidth_mhz"],
            "coverage": r["coverage"], "counter": r["counter"],
            "coverage_aoi": None if not np.isfinite(r.get("coverage_aoi", np.nan)) else round(r["coverage_aoi"], 3)}


def _duplicate_kind(kept, other) -> str:
    if other["frame"] != kept["frame"]:
        return "same pair in an adjacent frame"
    if other["bandwidth_mhz"] != kept["bandwidth_mhz"]:
        return "same pair in another range-bandwidth mode"
    if other["coverage"] != kept["coverage"]:
        return "full vs partial frame"
    if other["counter"] != kept["counter"]:
        return "same pair reprocessed (product counter)"
    return "same pair, same frame and mode (other copy)"


COVERS = 0.99        # a footprint covering this share of the AOI counts as covering it


def deduplicate(records) -> dict:
    """One granule per (reference, secondary) date pair, with the ranked
    alternatives kept for a data-level fallback. Returns {"pairs": [{"pair":
    (ref, sec), "candidates": [records, best first]}], "stack_mode_mhz",
    "dominant_frame", "decisions": [...]}.

    Ranking (first wins): footprint coverage of the AOI (to 1%), in the stack's
    bandwidth mode, in the stack's dominant frame, full over partial frame,
    higher product counter, later processing date, then the name (so the
    choice never depends on the order ASF returns results in - the old code
    kept whichever started first, so 001 could beat 002 and a 20 MHz partial
    beat a fuller 40 MHz one).

    Stack mode: the bandwidth with a covering granule in most pairs (ties:
    the wider band). A pair offered only in the other mode is kept (a chain
    with a pair missing is split) and listed under 'mode_mixed'.
    """
    by_pair: dict = {}
    for r in records:
        by_pair.setdefault((r["ref_date"], r["sec_date"]), []).append(r)

    def cov(r):
        c = r.get("coverage_aoi")
        return float(c) if c is not None and np.isfinite(c) else 0.0

    mode_votes: dict = {}
    for rs in by_pair.values():
        for m in {r["bandwidth_mhz"] for r in rs if cov(r) >= COVERS} or {max(rs, key=cov)["bandwidth_mhz"]}:
            mode_votes[m] = mode_votes.get(m, 0) + 1
    stack_mode = max(mode_votes, key=lambda m: (mode_votes[m], m)) if mode_votes else None
    frame_votes: dict = {}
    for rs in by_pair.values():
        good = [r for r in rs if r["bandwidth_mhz"] == stack_mode] or rs
        best = max(cov(r) for r in good)
        for f in {r["frame"] for r in good if cov(r) >= best - 0.01}:
            frame_votes[f] = frame_votes.get(f, 0) + 1
    dominant = max(frame_votes, key=lambda f: (frame_votes[f], -f)) if frame_votes else None

    def key(r):
        return (-round(cov(r), 2), r["bandwidth_mhz"] != stack_mode, r["frame"] != dominant,
                not r["full_frame"], -r["counter"], _neg_text(r.get("processing_date", "")), r["name"])

    pairs, decisions, mixed = [], [], []
    for pair in sorted(by_pair):
        cands = sorted(by_pair[pair], key=key)
        best = cands[0]
        if best["bandwidth_mhz"] != stack_mode:
            mixed.append({"pair": f"{pair[0]}_{pair[1]}", "bandwidth_mhz": best["bandwidth_mhz"]})
        pairs.append({"pair": pair, "candidates": cands})
        if len(cands) > 1:
            decisions.append({"pair": f"{pair[0]}_{pair[1]}", "kept": _brief(best),
                              "dropped": [{**_brief(o), "kind": _duplicate_kind(best, o)} for o in cands[1:]]})
    return {"pairs": pairs, "stack_mode_mhz": stack_mode, "dominant_frame": dominant,
            "decisions": decisions, "mode_mixed": mixed}


def _neg_text(s: str) -> tuple:
    """Sort key putting a LATER ISO date first."""
    return tuple(-ord(c) for c in str(s))


# ---------------------------------------------------------------- chain
def chain_segments(pairs) -> list[list[tuple]]:
    """Connected pieces of a pair network (dates are nodes), each a list of
    (ref, sec) pairs sorted by date; the longest (most dates) first, ties to
    the later one."""
    parent: dict = {}

    def find(a):
        parent.setdefault(a, a)
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a, b in pairs:
        parent[find(a)] = find(b)
    comps: dict = {}
    for p in pairs:
        comps.setdefault(find(p[0]), []).append(tuple(p))
    segs = [sorted(v) for v in comps.values()]
    segs.sort(key=lambda s: (-len({d for p in s for d in p}), _neg_text(max(p[1] for p in s))))
    return segs


def chain_report(pairs) -> dict:
    """Gaps, long pairs and pieces of a chain of pairs."""
    pairs = sorted(tuple(p) for p in pairs)
    dt = {p: (_ord(p[1]) - _ord(p[0])) for p in pairs}
    segs = chain_segments(pairs)
    gaps = []
    for s0, s1 in zip(sorted(segs, key=lambda s: s[0]), sorted(segs, key=lambda s: s[0])[1:]):
        gaps.append({"from": max(p[1] for p in s0), "to": min(p[0] for p in s1)})
    return {"pairs": len(pairs), "dates": len({d for p in pairs for d in p}),
            "pieces": [len({d for p in s for d in p}) for s in segs],
            "gaps": gaps, "non_12_day_pairs": [f"{a}_{b} ({dt[(a, b)]} d)" for a, b in pairs if dt[(a, b)] != 12],
            "loops": max(0, len(pairs) - (len({d for p in pairs for d in p}) - len(segs)))}


def _ord(yyyymmdd: str) -> int:
    return datetime.strptime(str(yyyymmdd)[:8], "%Y%m%d").toordinal()


# ---------------------------------------------------------------- lattice
@dataclass(frozen=True)
class Lattice:
    """A north-up lat/lon grid: cell (0, 0)'s north-west corner at (west, north)."""
    west: float
    north: float
    dlon: float
    dlat: float
    nx: int
    ny: int

    def lon(self):
        return self.west + (np.arange(self.nx) + 0.5) * self.dlon

    def lat(self):
        return self.north - (np.arange(self.ny) + 0.5) * self.dlat

    def bbox(self):
        return (self.west, self.north - self.ny * self.dlat, self.west + self.nx * self.dlon, self.north)

    def signature(self) -> str:
        return f"{self.west:.9f},{self.north:.9f},{self.dlon:.10f},{self.dlat:.10f},{self.nx},{self.ny}"

    def cells_in(self, bbox_wsen):
        """Boolean (ny, nx) mask of the cells whose centre lies inside bbox."""
        w, s, e, n = bbox_wsen
        lo, la = self.lon(), self.lat()
        return np.outer((la >= s) & (la <= n), (lo >= w) & (lo <= e))


def make_lattice(bbox_wsen, posting_m: float = 80.0, margin_deg: float = 0.02) -> Lattice:
    """The lattice every pair is put on: the AOI plus margin at the product's
    posting, in degrees at the AOI's mid-latitude (80 m is 0.00072 deg of
    latitude; longitude scaled by cos(lat))."""
    w, s, e, n = (float(v) for v in bbox_wsen)
    lat0 = math.radians((s + n) / 2)
    dlat = round(posting_m / M_PER_DEG_LAT, 10)
    dlon = round(posting_m / (111_319.49 * math.cos(lat0)), 10)
    west, north = round(w - margin_deg, 9), round(n + margin_deg, 9)
    nx = int(math.ceil((e + margin_deg - west) / dlon))
    ny = int(math.ceil((north - (s - margin_deg)) / dlat))
    return Lattice(west, north, dlon, dlat, nx, ny)


# ---------------------------------------------------------------- reading
class _CountingFile:
    """File-like wrapper counting the bytes h5py reads (the windowed-read proof)."""

    def __init__(self, fh):
        self.fh, self.bytes_read, self.reads = fh, 0, 0

    def read(self, n=-1):
        b = self.fh.read(n)
        self.bytes_read += len(b)
        self.reads += 1
        return b

    def readinto(self, buf):
        b = self.read(len(buf))
        buf[:len(b)] = b
        return len(b)

    def seek(self, *a):
        return self.fh.seek(*a)

    def tell(self):
        return self.fh.tell()

    def seekable(self):
        return True

    def readable(self):
        return True


@contextlib.contextmanager
def open_gunw(src, auth=None):
    """(h5py.File, counter) for a local path or an https URL. Over HTTPS only
    the HDF5 metadata and the chunks a read touches are fetched (1 MB blocks,
    cached), never the whole 2.0-2.5 GB file."""
    import h5py
    s = str(src)
    if s.startswith(("http://", "https://")):
        import aiohttp
        import fsspec
        if auth is None:
            from . import disp_backend as db
            auth = db._credentials()
        fs = fsspec.filesystem("https", client_kwargs={
            "auth": aiohttp.BasicAuth(*auth),
            "timeout": aiohttp.ClientTimeout(total=None, sock_connect=30, sock_read=240)})
        with fs.open(s, "rb", block_size=1 << 20, cache_type="blockcache") as raw:
            fh = _CountingFile(raw)
            with h5py.File(fh, "r") as h:
                yield h, fh
    else:
        with open(s, "rb") as raw:
            fh = _CountingFile(raw)
            with h5py.File(fh, "r") as h:
                yield h, fh


def _text(v) -> str:
    if isinstance(v, np.ndarray):
        v = v.ravel()[0] if v.size else ""
    return v.decode() if isinstance(v, (bytes, np.bytes_)) else str(v)


def _zd_time(v) -> datetime:
    return datetime.strptime(_text(v)[:19], "%Y-%m-%dT%H:%M:%S")


def product_polarization(h, wanted: str = "") -> str:
    """The co-pol layer to read: `wanted` when given (it must exist), else HH,
    else VV (GUNW are co-pol only; nisar-docs gunw)."""
    have = [_text(p) for p in np.atleast_1d(h[GRIDS + "listOfPolarizations"][()])] \
        if GRIDS + "listOfPolarizations" in h else []
    have = [p for p in have if UNW + p + "/unwrappedPhase" in h] or \
        [p for p in ("HH", "VV") if UNW + p + "/unwrappedPhase" in h]
    if wanted:
        if wanted.upper() not in have:
            raise ValueError(f"polarisation {wanted} is not in this GUNW (it has {have})")
        return wanted.upper()
    for p in ("HH", "VV"):
        if p in have:
            return p
    raise ValueError(f"no co-polarised unwrapped layer in this GUNW (listOfPolarizations {have})")


def _source_window(x, y, lattice: Lattice, epsg: int, pad: int = 2):
    """(r0, r1, c0, c1) of the source grid covering the lattice, or None."""
    from pyproj import Transformer
    tr = Transformer.from_crs("EPSG:4326", f"EPSG:{int(epsg)}", always_xy=True)
    w, s, e, n = lattice.bbox()
    xmin, ymin, xmax, ymax = tr.transform_bounds(w, s, e, n, densify_pts=21)
    dx, dy = float(x[1] - x[0]), float(y[1] - y[0])
    cs = sorted(((xmin - x[0]) / dx, (xmax - x[0]) / dx))
    rs = sorted(((ymin - y[0]) / dy, (ymax - y[0]) / dy))
    c0, c1 = max(0, int(math.floor(cs[0])) - pad), min(len(x), int(math.ceil(cs[1])) + pad + 1)
    r0, r1 = max(0, int(math.floor(rs[0])) - pad), min(len(y), int(math.ceil(rs[1])) + pad + 1)
    if c1 <= c0 or r1 <= r0:
        return None
    return r0, r1, c0, c1


def _nearest_index(xw, yw, lattice: Lattice, epsg: int):
    """Row/col into the source window of every lattice cell centre, and the
    'inside' mask (nearest neighbour)."""
    from pyproj import Transformer
    tr = Transformer.from_crs("EPSG:4326", f"EPSG:{int(epsg)}", always_xy=True)
    LON, LAT = np.meshgrid(lattice.lon(), lattice.lat())
    X, Y = tr.transform(LON, LAT)
    dx, dy = float(xw[1] - xw[0]), float(yw[1] - yw[0])
    col = np.rint((X - xw[0]) / dx).astype(np.int64)
    row = np.rint((Y - yw[0]) / dy).astype(np.int64)
    inside = (row >= 0) & (row < len(yw)) & (col >= 0) & (col < len(xw))
    return np.where(inside, row, 0), np.where(inside, col, 0), inside


def ionosphere_sign_stats(unw, ion, valid) -> dict:
    """Is the ionosphere screen part of the unwrapped phase with the same sign?
    Over the reliable cells (means removed): the correlation of the two, and
    the phase variance after subtracting vs after adding the screen. If the
    screen is the ionospheric part OF the phase, subtracting it lowers the
    variance (correlation > 0). Deformation and troposphere are uncorrelated
    noise to this test, so one pair proves little; the run reads the share
    over all pairs (summarise_ionosphere_sign)."""
    m = valid & np.isfinite(unw) & np.isfinite(ion)
    n = int(m.sum())
    if n < 50:
        return {"cells": n, "informative": False, "why": "too few reliable cells"}
    a = unw[m].astype("float64")
    b = ion[m].astype("float64")
    a -= a.mean()
    b -= b.mean()
    sa, sb = float(a.std()), float(b.std())
    if sb < 1e-6 or sa < 1e-9:
        return {"cells": n, "informative": False, "why": "flat screen or phase", "ion_std_rad": round(sb, 4)}
    corr = float((a * b).mean() / (sa * sb))
    vm, vp, v0 = float((a - b).var()), float((a + b).var()), float(a.var())
    return {"cells": n, "corr": round(corr, 3), "ion_std_rad": round(sb, 3), "phase_std_rad": round(sa, 3),
            "var_ratio_subtract": round(vm / v0, 3), "var_ratio_add": round(vp / v0, 3),
            "subtract_better": vm < vp, "informative": abs(corr) >= 0.3}


ION_STOP_MIN_PAIRS = 5        # informative pairs needed before the data may stop a run
ION_STOP_SHARE = 0.75         # ... and the share of them that must say "add"


def summarise_ionosphere_sign(stats: list[dict]) -> dict:
    """The data's verdict on the screen's sign, over all pairs.

    The sign is settled by the producer's code (module docstring,
    IONOSPHERE): the screen is the dispersive part OF the main-band phase, so
    subtracting it is right. The data test is a guard against a product
    whose convention changed, not the source of the sign, and it is weak:
    smooth troposphere and deformation fields correlate with a smooth screen
    by chance either way over a 20-30 km box (the first real pair, ASC-34 over
    Corcoran, gave corr -0.15: subtracting the screen RAISED that pair's phase
    variance by 29%, adding it by 5% - against the code's sign, but below the
    0.3 bar; a 0.2 rad screen on an early-morning pass is mostly estimation
    noise, and a smooth troposphere correlates with it at |0.3-0.5| either
    way. Watch this share on the first full stacks). So the run stops only on strong, repeated
    evidence: at least ION_STOP_MIN_PAIRS informative pairs (|corr| >= 0.3)
    of which at least ION_STOP_SHARE say adding fits better. A weaker
    majority is recorded as 'doubtful' and the screen is applied."""
    inf = [s for s in stats if s.get("informative")]
    sub = sum(1 for s in inf if s.get("subtract_better"))
    add = len(inf) - sub
    out = {"pairs": len(stats), "informative_pairs": len(inf), "subtract_better": sub, "add_better": add,
           "rule": (f"informative = |corr(phase, screen)| >= 0.3 over the AOI's reliable cells; the run stops when "
                    f"at least {ION_STOP_MIN_PAIRS} informative pairs exist and at least {ION_STOP_SHARE:.0%} of "
                    f"them say adding the screen fits better")}
    if len(inf) >= ION_STOP_MIN_PAIRS and add >= ION_STOP_SHARE * len(inf):
        out["verdict"] = "opposite sign"
    elif len(inf) >= 3 and add > len(inf) / 2:
        out["verdict"] = ("doubtful: most informative pairs fit better with the screen added, below the stop rule; "
                          "applied as the producer's code defines it (subtracted)")
    elif len(inf) >= 3:
        out["verdict"] = "subtract (MintPy split_spectrum) agrees with the data"
    else:
        out["verdict"] = ("not testable from the data (fewer than 3 informative pairs): subtracted, as the "
                          "producer's code defines it")
    return out


# The (x, y) a crop stores in losUnitVectorX/Y, as functions of the product's
# true (east, north) LOS components, tried in this order (identity first).
_LOS_ENCODINGS = (
    ("east_north", lambda e, n: (e, n)),
    ("minus_north_east", lambda e, n: (-n, e)),
    ("north_minus_east", lambda e, n: (n, -e)),
    ("minus_east_minus_north", lambda e, n: (-e, -n)),
    ("north_east", lambda e, n: (n, e)),
    ("minus_north_minus_east", lambda e, n: (-n, -e)),
    ("east_minus_north", lambda e, n: (e, -n)),
    ("minus_east_north", lambda e, n: (-e, n)),
)
_LOS_PROBE = ((0.641, 0.182), (-0.612, 0.205), (0.301, -0.552))   # (east, north) of target-to-sensor vectors


def mintpy_azimuth_deg(east, north):
    """MintPy's azimuthAngle (degrees, from north, anti-clockwise positive) of
    the target-to-sensor LOS vector with these east/north components:
    MintPy's enu2los unit vector is east = -sin(inc) sin(az), north =
    sin(inc) cos(az) (utils0.get_unit_vector4component_of_interest, and
    deliverables.geometry_field), so az = atan2(-east, north)."""
    return np.degrees(np.arctan2(-np.asarray(east, dtype="float64"), np.asarray(north, dtype="float64")))


def los_encoding_for_prep_nisar() -> tuple[str, object]:
    """(name, function) turning the product's LOS (east, north) into what a
    crop must store as losUnitVectorX/Y so that the INSTALLED prep_nisar
    writes azimuthAngle in MintPy's own convention.

    Why (found 8 Oct 2026 on the verification crop, ASC-34 over Corcoran):
    the GUNW's losUnitVectorX/Y are the east and north components of the
    target-to-sensor vector (their own descriptions say so). MintPy 1.6.4's
    prep_nisar.interpolate_geometry computes az = atan2(-losy, -losx), which
    is 90 degrees off MintPy's convention atan2(-east, north). On that pair
    it wrote -164.1 deg where the product's vector (east 0.641, north 0.182)
    means -74.1 deg, so every consumer of geometryGeo.h5 - deliverables.
    geometry_field (the harness's GNSS-to-LOS projection), decompose.py,
    MintPy's solid-earth-tide step - would have read the line of sight as
    east 0.18, north -0.64.

    The formula is found by BEHAVIOUR, not assumed: prep_nisar's own
    interpolate_geometry is called on three synthetic vectors and the first
    encoding that gives MintPy's azimuth for all three wins. A MintPy that
    fixes prep_nisar is then met with the identity; a MintPy whose formula
    is none of the eight signed swaps stops the run here."""
    return _los_encoding_probe()


@functools.lru_cache(maxsize=1)
def _los_encoding_probe():
    from mintpy import prep_nisar as pn
    want = [float(mintpy_azimuth_deg(e, n)) for e, n in _LOS_PROBE]
    one = np.array([[0.5]])
    for name, enc in _LOS_ENCODINGS:
        ok = True
        for (e, n), w in zip(_LOS_PROBE, want):
            x, y = enc(e, n)
            rdr = {"xcoord_radar_grid": np.array([0.0, 1.0]), "ycoord_radar_grid": np.array([0.0, 1.0]),
                   "height_radar_grid": np.array([0.0, 1.0]), "slant_range": np.full((2, 2, 2), 9e5),
                   "incidence_angle": np.full((2, 2, 2), 40.0), "los_x": np.full((2, 2, 2), x),
                   "los_y": np.full((2, 2, 2), y)}
            _, _, az = pn.interpolate_geometry(one, one, one, rdr, np.array([[True]]))
            d = (float(az[0, 0]) - w + 180.0) % 360.0 - 180.0
            if not abs(d) < 1e-3:
                ok = False
                break
        if ok:
            return name, enc
    raise RuntimeError("MintPy's prep_nisar writes azimuthAngle by a formula none of the known LOS encodings "
                       "matches: the line of sight it would record is unknown. Check prep_nisar."
                       "interpolate_geometry against MintPy's convention (az = atan2(-east, north)) before "
                       "running the NISAR route.")


def crop_gunw(src, lattice: Lattice, out_path, auth=None, pol: str = "", height_range=None,
              aoi_bbox=None, meta: dict | None = None) -> dict:
    """Read one GUNW over the lattice (windowed when src is a URL), put every
    layer on the lattice, write a GUNW-layout HDF5 prep_nisar reads, and
    return the pair's record (dates, window, bytes read, valid share, coherence,
    ionosphere sign test). Returns {"covered": False, ...} and writes nothing
    when the granule's grid does not reach the lattice."""
    import h5py
    t0 = time.time()
    out_path = Path(out_path)
    with open_gunw(src, auth) as (h, fh):
        pol = product_polarization(h, pol)
        G = UNW + pol + "/"
        epsg = int(np.asarray(h[G + "projection"][()]).item())
        x = h[G + "xCoordinates"][()]
        y = h[G + "yCoordinates"][()]
        win = _source_window(x, y, lattice, epsg)
        ref_t = _zd_time(h[IDENT + "referenceZeroDopplerStartTime"][()])
        sec_t = _zd_time(h[IDENT + "secondaryZeroDopplerStartTime"][()])
        rec = {"source": str(src), "epsg_source": epsg, "polarization": pol,
               "reference": ref_t.strftime("%Y%m%d"), "secondary": sec_t.strftime("%Y%m%d"),
               "posting_source_m": [float(x[1] - x[0]), float(y[1] - y[0])]}
        if win is None:
            return {**rec, "covered": False, "why": "the granule's grid does not reach the AOI"}
        r0, r1, c0, c1 = win
        xw, yw = x[c0:c1], y[r0:r1]
        row, col, inside = _nearest_index(xw, yw, lattice, epsg)
        layers, attrs = {}, {}
        for name in GRID_LAYERS + ("mask",):
            path = (UNW + "mask") if name == "mask" else (G + name)
            if path not in h:
                continue
            d = h[path]
            a = d[r0:r1, c0:c1]
            attrs[name] = {k: d.attrs[k] for k in d.attrs.keys() if k in ("_FillValue", "units", "description")}
            fill = d.attrs.get("_FillValue", None)
            if np.issubdtype(a.dtype, np.floating):
                a = a.astype("float32")
                if fill is not None and np.isfinite(fill):
                    a[a == fill] = np.nan
                o = np.full((lattice.ny, lattice.nx), np.nan, dtype="float32")
            else:
                fv = int(fill) if fill is not None else 0
                o = np.full((lattice.ny, lattice.nx), fv if name == "mask" else 0, dtype=a.dtype)
                if name == "connectedComponents" and fill is not None:
                    a = np.where(a == fill, 0, a)          # no data -> component 0 (unreliable)
            o[inside] = a[row[inside], col[inside]]
            layers[name] = o
        if "unwrappedPhase" not in layers:
            raise ValueError(f"{src}: no unwrappedPhase layer for {pol}")
        # date order: chronological is the convention (module docstring)
        reversed_pair = ref_t > sec_t
        if reversed_pair:
            for name in PHASE_LAYERS:
                if name in layers:
                    layers[name] = -layers[name]
            rec["reference"], rec["secondary"] = rec["secondary"], rec["reference"]
        # the metadata cube, re-gridded on a coarse lat/lon lattice
        cube = _read_cube(h, lattice, epsg, height_range, negate=reversed_pair)
        # small copies: identification, centre frequency, looks
        ident = {k: h[IDENT + k][()] for k in h[IDENT].keys() if isinstance(h[IDENT + k], h5py.Dataset)
                 and h[IDENT + k].size <= 64}
        params = {k: h[PARAMS + k][()] for k in (h[PARAMS].keys() if PARAMS in h else [])
                  if isinstance(h[PARAMS + k], h5py.Dataset) and h[PARAMS + k].size <= 64}
        cfreq = float(np.asarray(h[GRIDS + "centerFrequency"][()]).item())
        look = _text(h[IDENT + "lookDirection"][()]) if IDENT + "lookDirection" in h else ""
        pass_dir = _text(h[IDENT + "orbitPassDirection"][()]) if IDENT + "orbitPassDirection" in h else ""
        iono_alg = _text(h[IONO_ALGORITHM][()]) if IONO_ALGORITHM in h else ""
        rec["bytes_read"] = int(fh.bytes_read)
    if reversed_pair:
        for a, b in (("referenceZeroDopplerStartTime", "secondaryZeroDopplerStartTime"),
                     ("referenceZeroDopplerEndTime", "secondaryZeroDopplerEndTime")):
            if a in ident and b in ident:
                ident[a], ident[b] = ident[b], ident[a]

    # reliability over the AOI (not the margin) and the ionosphere sign test
    aoi = lattice.cells_in(aoi_bbox) if aoi_bbox is not None else np.ones((lattice.ny, lattice.nx), bool)
    valid = valid_cells(layers, attrs.get("mask", {}).get("_FillValue"))
    coh = layers.get("coherenceMagnitude")
    rec.update({
        "covered": True, "window_rows_cols": [int(r0), int(r1), int(c0), int(c1)],
        "window_source_cells": int((r1 - r0) * (c1 - c0)),
        "valid_share_aoi": round(float((valid & aoi).sum()) / max(1, int(aoi.sum())), 4),
        "inside_share_aoi": round(float((inside & aoi).sum()) / max(1, int(aoi.sum())), 4),
        "coherence_median_aoi": (round(float(np.nanmedian(coh[valid & aoi])), 3)
                                 if coh is not None and (valid & aoi).any() else None),
        "reversed_by_pipeline": bool(reversed_pair), "wavelength_m": C_M_S / cfreq,
        "wavelength_native_m": C_M_S / cfreq, "phase_scale": 1.0,
        "look_direction": look.lower(), "orbit_pass_direction": pass_dir,
        "ionosphere_algorithm": iono_alg or None,
        "seconds": round(time.time() - t0, 1)})
    if "ionospherePhaseScreen" in layers:
        rec["ionosphere_sign"] = ionosphere_sign_stats(layers["unwrappedPhase"], layers["ionospherePhaseScreen"],
                                                       valid & aoi)
    _write_crop(out_path, lattice, pol, layers, attrs, cube, ident, params, cfreq, {**(meta or {}), **rec})
    return _py(rec)


def _py(v):
    """Plain Python values (numpy scalars and arrays converted) for JSON."""
    if isinstance(v, dict):
        return {k: _py(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_py(x) for x in v]
    if isinstance(v, np.ndarray):
        return [_py(x) for x in v.tolist()]
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, bytes):
        return v.decode()
    return v


def _read_cube(h, lattice: Lattice, epsg: int, height_range=None, negate=False) -> dict:
    """The radar-grid cube fields on a lat/lon lattice at the cube's own
    spacing (~500 m), covering the data lattice + 2 cube cells, for the height
    levels bracketing height_range (all levels when None). Linear
    interpolation in the source projection, level by level."""
    from pyproj import Transformer
    from scipy.interpolate import RegularGridInterpolator
    cx = h[RADARGRID + "xCoordinates"][()]
    cy = h[RADARGRID + "yCoordinates"][()]
    hgt = h[RADARGRID + "heightAboveEllipsoid"][()]
    k0, k1 = 0, len(hgt)
    if height_range is not None:
        lo, hi = float(height_range[0]), float(height_range[1])
        k0 = max(0, int(np.searchsorted(hgt, lo, side="right")) - 2)
        k1 = min(len(hgt), int(np.searchsorted(hgt, hi, side="left")) + 2)
        if k1 - k0 < 2:
            k0, k1 = max(0, k0 - 1), min(len(hgt), k0 + 2)
    # coarse lat/lon lattice at the cube spacing, 2 cube cells beyond the data
    dxm = abs(float(cx[1] - cx[0]))
    w, s, e, n = lattice.bbox()
    lat0 = math.radians((s + n) / 2)
    dlat_c = dxm / M_PER_DEG_LAT
    dlon_c = dxm / (111_319.49 * math.cos(lat0))
    lon_c = np.arange(w - 2 * dlon_c, e + 2 * dlon_c + 1e-12, dlon_c)
    lat_c = np.arange(n + 2 * dlat_c, s - 2 * dlat_c - 1e-12, -dlat_c)
    LON, LAT = np.meshgrid(lon_c, lat_c)
    X, Y = Transformer.from_crs("EPSG:4326", f"EPSG:{int(epsg)}", always_xy=True).transform(LON, LAT)
    # source sub-window of the cube
    dx, dy = float(cx[1] - cx[0]), float(cy[1] - cy[0])
    ci = np.clip(np.rint((np.array([X.min(), X.max()]) - cx[0]) / dx).astype(int), 0, len(cx) - 1)
    ri = np.clip(np.rint((np.array([Y.min(), Y.max()]) - cy[0]) / dy).astype(int), 0, len(cy) - 1)
    c0, c1 = max(0, ci.min() - 2), min(len(cx), ci.max() + 3)
    r0, r1 = max(0, ri.min() - 2), min(len(cy), ri.max() + 3)
    xs, ys = cx[c0:c1], cy[r0:r1]
    out = {"xCoordinates": lon_c, "yCoordinates": lat_c, "heightAboveEllipsoid": hgt[k0:k1],
           "_window": [int(k0), int(k1), int(r0), int(r1), int(c0), int(c1)]}
    yflip = ys[0] > ys[-1]
    yy = ys[::-1] if yflip else ys
    pts = np.column_stack([Y.ravel(), X.ravel()])
    for name in CUBE_FIELDS:
        if RADARGRID + name not in h:
            continue
        sub = h[RADARGRID + name][k0:k1, r0:r1, c0:c1].astype("float64")
        res = np.empty((sub.shape[0],) + LON.shape, dtype="float64")
        for k in range(sub.shape[0]):
            f = sub[k][::-1] if yflip else sub[k]
            itp = RegularGridInterpolator((yy, xs), f, method="linear", bounds_error=False, fill_value=np.nan)
            res[k] = itp(pts).reshape(LON.shape)
        if negate and name in CUBE_DIFFERENTIAL:
            res = -res
        out[name] = res.astype("float32" if h[RADARGRID + name].dtype == np.float32 else "float64")
    return out


def _write_crop(path: Path, lattice: Lattice, pol: str, layers: dict, attrs: dict, cube: dict,
                ident: dict, params: dict, cfreq: float, meta: dict) -> None:
    """A GUNW-layout HDF5 on the lat/lon lattice: the datasets MintPy's
    prep_nisar reads (grids/frequencyA/unwrappedInterferogram/<pol>/...,
    the mask, centerFrequency, listOfPolarizations, the radar-grid cube, the
    identification and the looks), EPSG 4326."""
    import h5py
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".h5.part")
    with h5py.File(tmp, "w") as f:
        G = UNW + pol + "/"
        f.create_dataset(G + "xCoordinates", data=lattice.lon())
        f.create_dataset(G + "yCoordinates", data=lattice.lat())
        f.create_dataset(G + "xCoordinateSpacing", data=float(lattice.dlon))
        f.create_dataset(G + "yCoordinateSpacing", data=-float(lattice.dlat))
        for g in (G, UNW, RADARGRID):
            d = f.create_dataset(g + "projection", data=np.uint32(4326))
            d.attrs["epsg_code"] = 4326
            d.attrs["source_epsg"] = int(meta.get("epsg_source", 0) or 0)
        for name, arr in layers.items():
            p = (UNW + "mask") if name == "mask" else (G + name)
            d = f.create_dataset(p, data=arr, compression="gzip", compression_opts=4, chunks=True)
            for k, v in attrs.get(name, {}).items():
                if k == "_FillValue" and name == "connectedComponents":
                    continue                                   # fill already mapped to component 0
                d.attrs[k] = v
        f.create_dataset(GRIDS + "centerFrequency", data=float(cfreq))
        f.create_dataset(GRIDS + "listOfPolarizations", data=np.array([pol.encode()], dtype="S2"))
        cube = dict(cube)
        los_note = "no LOS vector in the product"
        if "losUnitVectorX" in cube and "losUnitVectorY" in cube:
            # see los_encoding_for_prep_nisar: what prep_nisar must read to
            # write MintPy's azimuth; the product's own east/north are kept
            # beside it under names prep_nisar does not read
            enc_name, enc = los_encoding_for_prep_nisar()
            east, north = cube["losUnitVectorX"], cube["losUnitVectorY"]
            x, y = enc(east, north)
            cube["losUnitVectorX"], cube["losUnitVectorY"] = np.asarray(x), np.asarray(y)
            cube["losUnitVectorEast"], cube["losUnitVectorNorth"] = east, north
            los_note = enc_name
        for k, v in cube.items():
            if k.startswith("_"):
                continue
            d = f.create_dataset(RADARGRID + k, data=v, compression="gzip" if np.ndim(v) == 3 else None)
            if k in ("losUnitVectorEast", "losUnitVectorNorth"):
                d.attrs["description"] = (f"{'East' if k.endswith('East') else 'North'} component of the target-to-"
                                          "sensor LOS unit vector as the GUNW states it (losUnitVectorX/Y of this "
                                          "crop are re-encoded for MintPy's prep_nisar: attribute los_vector_encoding)")
        f.attrs["los_vector_encoding"] = los_note
        f.attrs["crop_version"] = CROP_VERSION
        for k, v in ident.items():
            f.create_dataset(IDENT + k, data=v)
        for k, v in params.items():
            f.create_dataset(PARAMS + k, data=v)
        f.attrs["pipeline_crop"] = ("insar_pipeline.nisar_backend crop: windowed read of a NISAR L2 GUNW, every "
                                    "layer resampled to EPSG:4326 by nearest neighbour; the radar-grid cube "
                                    "re-gridded linearly")
        f.attrs["lattice"] = lattice.signature()
        f.attrs["cube_window_levels_rows_cols"] = list(cube.get("_window", []))
        for k, v in meta.items():
            if v is None:
                continue
            f.attrs[k] = (v if isinstance(v, (int, float, str, bool, np.integer, np.floating, np.bool_))
                          else json.dumps(_py(v)))
    tmp.replace(path)


def read_crop_record(path) -> dict:
    """The pair record a crop carries in its attributes (crop_gunw's return)."""
    import h5py
    with h5py.File(path, "r") as f:
        out = {}
        for k, v in f.attrs.items():
            if k in ("pipeline_crop", "lattice", "cube_window_levels_rows_cols", "los_vector_encoding",
                     "crop_version"):
                continue
            v = _py(v)
            if isinstance(v, str) and v[:1] in "[{":
                try:
                    v = json.loads(v)
                except ValueError:
                    pass
            out[k] = v
    return out


# ---------------------------------------------------------------- staging
def stage_dem(cfg, lattice: Lattice) -> Path:
    """A DEM over the lattice (+0.05 deg) for prep_nisar: cfg.dem_path when set,
    else Copernicus GLO-30 converted to ELLIPSOIDAL heights (the GUNW radar-grid
    cube is in heights above the WGS84 ellipsoid), as a GeoTIFF in EPSG:4326.
    data_acquisition.get_dem is not used: it sizes the DEM from Sentinel-1
    burst footprints and converts it for ISCE2."""
    if getattr(cfg, "dem_path", None) and Path(cfg.dem_path).exists():
        return Path(cfg.dem_path)
    out = Path(cfg.dem_dir) / "nisar_dem_glo30_ellipsoid.tif"
    rec = out.with_suffix(".json")
    w, s, e, n = lattice.bbox()
    want = [w - 0.05, s - 0.05, e + 0.05, n + 0.05]
    if out.exists() and rec.exists():
        try:
            have = json.loads(rec.read_text())["bbox_wsen"]
            if have[0] <= want[0] and have[1] <= want[1] and have[2] >= want[2] and have[3] >= want[3]:
                return out
        except (ValueError, KeyError, TypeError):
            pass
    import rasterio
    from dem_stitcher import stitch_dem
    log.info("Stitching Copernicus GLO-30 (ellipsoidal heights) over W %.3f S %.3f E %.3f N %.3f", *want)
    X, profile = stitch_dem(want, dem_name="glo_30", dst_ellipsoidal_height=True, dst_area_or_point="Point")
    profile.update(driver="GTiff", count=1, dtype=X.dtype, height=X.shape[0], width=X.shape[1],
                   compress="deflate")
    out.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out, "w", **profile) as dst:
        dst.write(X, 1)
        dst.update_tags(AREA_OR_POINT="Point")
    with rasterio.open(out) as src:
        got = list(src.bounds)
    rec.write_text(json.dumps({"bbox_wsen": got, "requested_wsen": want, "source": "glo_30",
                               "heights": "ellipsoidal (dem_stitcher dst_ellipsoidal_height)"}, indent=1))
    return out


def dem_height_range(dem_path, lattice: Lattice) -> tuple[float, float] | None:
    """(min, max) height of the DEM over the lattice; None when unreadable."""
    try:
        import rasterio
        from rasterio.windows import from_bounds
        with rasterio.open(dem_path) as src:
            a = src.read(1, window=from_bounds(*lattice.bbox(), transform=src.transform), masked=True)
        a = np.asarray(a.filled(np.nan), dtype="float64")
        a = a[np.isfinite(a) & (a > -1000)]
        return (float(a.min()), float(a.max())) if a.size else None
    except Exception as exc:                                 # noqa: BLE001 - then every level is read
        log.warning("DEM height range unreadable (%s): reading every height level of the metadata cube", exc)
        return None


def coherence_gate(crop_files, lattice: Lattice, out_tif: Path, floor: float) -> dict:
    """Mean interferometric coherence over the pairs used, and the 0/1 gate at
    `floor` - the reliability rule for a chain (temporal coherence is 1.0 by
    construction). Two-band GeoTIFF (EPSG:4326): band 1 mean coherence, band 2
    gate (1 = reliable)."""
    import h5py
    import rasterio
    from rasterio.transform import from_origin
    acc = np.zeros((lattice.ny, lattice.nx))
    cnt = np.zeros((lattice.ny, lattice.nx))
    for p in crop_files:
        with h5py.File(p, "r") as f:
            pols = [k for k in f[UNW].keys() if isinstance(f[UNW + k], h5py.Group)]
            c = f[UNW + pols[0] + "/coherenceMagnitude"][()].astype("float64")
        m = np.isfinite(c)
        acc[m] += c[m]
        cnt[m] += 1
    mean = np.where(cnt > 0, acc / np.maximum(cnt, 1), np.nan).astype("float32")
    gate = (np.nan_to_num(mean, nan=0.0) >= float(floor)).astype("float32")
    out_tif = Path(out_tif)
    out_tif.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_tif, "w", driver="GTiff", height=lattice.ny, width=lattice.nx, count=2,
                       dtype="float32", crs="EPSG:4326", nodata=np.nan,
                       transform=from_origin(lattice.west, lattice.north, lattice.dlon, lattice.dlat)) as dst:
        dst.write(mean, 1)
        dst.write(gate, 2)
        dst.set_band_description(1, "mean coherence over the pairs used")
        dst.set_band_description(2, f"reliable: mean coherence >= {floor}")
    have = np.isfinite(mean)
    return {"file": str(out_tif), "floor": float(floor), "pairs": len(crop_files),
            "share_reliable": round(float(gate[have].mean()), 4) if have.any() else None,
            "rule": ("chain network: MintPy's temporal coherence is 1.0 by construction and is not a test; "
                     "a cell is reliable when its mean interferometric coherence over the pairs used is at least "
                     "the floor (min_temporal_coherence)")}


# Layers in radians of phase at the product's own centre frequency; the rest
# (coherence, components, angles, slant range, baselines in metres) do not
# depend on the wavelength.
SCALED_GRID_LAYERS = ("unwrappedPhase", "ionospherePhaseScreen", "ionospherePhaseScreenUncertainty")
SCALED_CUBE_FIELDS = ("wetTroposphericPhaseScreen", "hydrostaticTroposphericPhaseScreen",
                      "slantRangeSolidEarthTidesPhase")


def stack_wavelength(recs) -> float | None:
    """The wavelength most kept pairs have (ties: the shorter, i.e. the
    higher centre frequency of the 40 MHz mode)."""
    votes: dict = {}
    exact: dict = {}
    for r in recs:
        w = r.get("wavelength_native_m", r.get("wavelength_m"))
        if w:
            k = round(float(w), 9)                  # the key only: the value returned is the product's own
            votes[k] = votes.get(k, 0) + 1
            exact.setdefault(k, float(w))
    return exact[min(votes, key=lambda w: (-votes[w], w))] if votes else None


def harmonise_wavelength(crop_files, target_wavelength_m: float) -> list[dict]:
    """Put every crop's phase on ONE wavelength, in place.

    MintPy converts the whole stack with a single WAVELENGTH (prep_nisar
    takes it from the first file). The 20 MHz and 40 MHz modes have different
    centre frequencies, so a pair the stack had to take in the other mode
    (deduplicate's 'mode_mixed': a chain with that pair missing would split)
    would be converted with the wrong wavelength, a scale error of a few per
    cent on that pair. Each phase layer of such a crop is multiplied by
    lambda_pair / lambda_target, which keeps its range-equivalent value
    (range = -lambda/(4 pi) x phase) exactly; this holds for the dispersive
    ionosphere screen as well, because MintPy converts and subtracts it in the
    same units. centerFrequency is set to the target's. Idempotent: the scale
    applied so far is kept in the attribute phase_scale. Returns one record per
    crop changed."""
    import h5py
    out = []
    tw = float(target_wavelength_m)
    for p in crop_files:
        with h5py.File(p, "a") as f:
            native = float(f.attrs.get("wavelength_native_m", f.attrs.get("wavelength_m", tw)))
            have = float(f.attrs.get("phase_scale", 1.0))
            need = native / tw
            if abs(need - have) <= 1e-9 * max(1.0, abs(need)):
                continue
            factor = need / have
            pols = [k for k in f[UNW].keys() if isinstance(f[UNW + k], h5py.Group)]
            for pol in pols:
                for name in SCALED_GRID_LAYERS:
                    path = UNW + pol + "/" + name
                    if path in f:
                        f[path][...] = f[path][()] * factor
            for name in SCALED_CUBE_FIELDS:
                if RADARGRID + name in f:
                    f[RADARGRID + name][...] = f[RADARGRID + name][()] * factor
            f[GRIDS + "centerFrequency"][()] = C_M_S / tw
            f.attrs["phase_scale"] = need
            f.attrs["wavelength_m"] = tw
            f.attrs["wavelength_note"] = (f"phase layers scaled by {need:.6f} (native wavelength {native:.6f} m) so "
                                          f"MintPy's single WAVELENGTH {tw:.6f} m gives this pair's range change")
            out.append({"file": str(p), "native_wavelength_m": native, "target_wavelength_m": tw,
                        "phase_scale": round(need, 9)})
    return out


def unwrap_error_method(cfg) -> tuple[str, str]:
    """The config's mintpy_unwrap_error_method without phase closure, which
    needs triangles a chain does not have."""
    req = str(getattr(cfg, "mintpy_unwrap_error_method", "no") or "no").lower()
    parts = [p for p in req.split("+") if p and p not in ("phase_closure", "auto", "no")]
    used = "+".join(parts) or "no"
    why = "" if used == req else f"requested {req}; phase closure is not applicable to a chain of 12-day pairs"
    return used, why


def mintpy_loader_lines(cfg, stack_dir: Path, dem: Path, ionosphere: bool) -> str:
    """MintPy settings for the cropped GUNW: prep_nisar on stack_dir/*.h5 with
    the staged DEM, MintPy's split-spectrum step on the product's ionosphere
    screen, no coherence-based pair dropping (one dropped pair splits a
    chain), and an unwrap-error method without phase closure."""
    method, _ = unwrap_error_method(cfg)
    return (
        "mintpy.load.processor              = nisar\n"
        f"mintpy.load.unwFile                = {stack_dir}/*.h5\n"
        f"mintpy.load.demFile                = {dem}\n"
        f"mintpy.ionosphericDelay.method     = {'split_spectrum' if ionosphere else 'no'}\n"
        "mintpy.network.coherenceBased      = no\n"
        f"mintpy.unwrapError.method          = {method}\n")


def prep_other_polarization(cfg, stack_dir: Path, dem: Path, pol: str) -> None:
    """MintPy 1.6.4's prep_nisar reads HH and has no option for VV
    (getattr(inps, 'polarization', 'HH')). For a VV stack it is called here,
    through its own function with the polarisation set, into mintpy/inputs/
    before smallbaselineApp; load_data's own call then fails on the missing
    HH layer, warns "Assuming its result exists" and goes on with these files
    (load_data.py 671-677). HH stacks never come here."""
    from argparse import Namespace

    from mintpy.prep_nisar import load_nisar
    w, s, e, n = cfg.bbox
    sub = bool(getattr(cfg, "subset_to_aoi", True))
    inps = Namespace(input_glob=f"{stack_dir}/*.h5", dem_file=str(dem), mask_file=None, frequency="A",
                     out_dir=str(cfg.mintpy_dir), update_mode=False, polarization=pol,
                     subset_lat=[s, n] if sub else None, subset_lon=[w, e] if sub else None)
    Path(cfg.mintpy_dir, "inputs").mkdir(parents=True, exist_ok=True)
    log.warning("Polarisation %s: running prep_nisar here (MintPy reads HH only); MintPy's own load step will "
                "warn that prep_nisar failed and use these inputs", pol)
    load_nisar(inps)


# ---------------------------------------------------------------- cleanup
def cleanup_targets(cfg) -> list[tuple[int, Path, str]]:
    """What `insar cleanup` should offer on a NISAR project, as
    (level, path, why) like scripts/cleanup.targets. Level 1: whole granules
    (left by an interrupted 'download' fetch or by the pre-8-Oct route) and
    crops no longer used; level 2: the crops MintPy loaded (mintpy/inputs/
    holds what they gave; a re-run fetches them again)."""
    P = Path(cfg.project_dir) / "nisar_gunw"
    out = []
    if not P.is_dir():
        return out
    whole = sorted(P.glob("*.h5")) + sorted((P / "whole").glob("*.h5")) + sorted((P / "whole").glob("*.part"))
    for f in whole:
        out.append((1, f, "whole NISAR GUNW granule (the route reads windows; not used)"))
    crops, stack = P / "crops", P / "stack"
    if crops.is_dir():
        for f in sorted(crops.glob("*_alt.h5")) + sorted(crops.glob("*.h5.part")):
            out.append((1, f, "left by an interrupted read of an alternative granule"))
        used = {q.name for q in stack.glob("*.h5")} if stack.is_dir() else None
        if used:
            for f in sorted(crops.glob("NISAR_GUNW_CROP_*.h5")):
                if f.name not in used and not f.name.endswith("_alt.h5"):
                    out.append((1, f, "crop of a pair dropped from the stack (screened out or off the chain)"))
    for d in (crops, stack):
        if d.is_dir():
            out.append((2, d, "cropped NISAR GUNW windows (MintPy's inputs hold them; a re-run fetches again)"))
    return out


# ---------------------------------------------------------------- the route
def product_mask_keep(mask, fill=None) -> np.ndarray:
    """The GUNW `mask` decoded as MintPy 1.6.4's prep_nisar decodes it
    (_read_is_land_and_valid_mask): bits 0-7 are three digits - water flag,
    reference subswath, secondary subswath - and a subswath of 0 means that
    RSLC has no valid sample there. Keep = both RSLCs valid and not water."""
    m = np.asarray(mask)
    ok = m != fill if fill is not None else np.ones(m.shape, bool)
    low = np.where(ok, m.astype(np.int64) & 0xFF, 0)
    return ok & ((low // 10) % 10 > 0) & (low % 10 > 0) & (low // 100 != 1)


def valid_cells(layers: dict, mask_fill=None) -> np.ndarray:
    """Cells a pair actually measures: finite unwrapped phase in a connected
    component, kept by the product's own mask (product_mask_keep).

    The mask is part of the test because prep_nisar keeps only cells kept
    in EVERY pair. Site E T019A (26 Oct 2026): the 2026-09-07
    acquisition's subswath digit was 0 over the whole box in both pairs
    touching it (coherence NaN, flat phase), while the phase was finite with
    one component. The old test passed both pairs at 100% valid, and
    prep_nisar then masked every cell of the stack - MintPy stopped at
    plot_network with NaN coherence in all 28 pairs."""
    cc = layers.get("connectedComponents")
    valid = np.isfinite(layers["unwrappedPhase"]) & (cc > 0 if cc is not None else True)
    if layers.get("mask") is not None:
        valid &= product_mask_keep(layers["mask"], mask_fill)
    return valid


def crop_validity(path, lattice: Lattice, aoi_bbox) -> dict:
    """valid_share_aoi and coherence_median_aoi of a crop on disk, by
    valid_cells - so a crop reused from before 26 Oct 2026 is screened by
    today's test without being read again."""
    import h5py
    with h5py.File(path, "r") as f:
        pol = _text(f[GRIDS + "listOfPolarizations"][()][0])
        G = UNW + pol + "/"
        layers = {k: f[G + k][()] for k in ("unwrappedPhase", "connectedComponents", "coherenceMagnitude")
                  if G + k in f}
        fill = None
        if UNW + "mask" in f:
            layers["mask"] = f[UNW + "mask"][()]
            fill = f[UNW + "mask"].attrs.get("_FillValue")
    aoi = lattice.cells_in(aoi_bbox) if aoi_bbox is not None else np.ones((lattice.ny, lattice.nx), bool)
    valid = valid_cells(layers, fill)
    coh = layers.get("coherenceMagnitude")
    return {"valid_share_aoi": round(float((valid & aoi).sum()) / max(1, int(aoi.sum())), 4),
            "coherence_median_aoi": (round(float(np.nanmedian(coh[valid & aoi])), 3)
                                     if coh is not None and (valid & aoi).any() else None)}


def acquisition_times(cfg) -> dict:
    """{date: seconds of day UTC at mid-acquisition} from the crops'
    identification (each side's zero-Doppler start and end), the NISAR
    counterpart of stack_backend._acquisition_times for weather_raider (HRRR
    through RAiDER needs the time of the pass). Site E T019A:
    12:11:15-12:11:44 on 2025-10-30."""
    import h5py
    out = {}
    for p in sorted((Path(cfg.project_dir) / "nisar_gunw" / "crops").glob("NISAR_GUNW_CROP_*.h5")):
        if p.stem.endswith("_alt"):
            continue
        try:
            with h5py.File(p, "r") as h:
                for side in ("reference", "secondary"):
                    t0 = _zd_time(h[f"{IDENT}{side}ZeroDopplerStartTime"][()])
                    end = f"{IDENT}{side}ZeroDopplerEndTime"
                    t1 = _zd_time(h[end][()]) if end in h else t0
                    s0 = t0.hour * 3600 + t0.minute * 60 + t0.second
                    out[t0.strftime("%Y%m%d")] = (s0 + (t1 - t0).total_seconds() / 2.0) % 86400
        except (OSError, KeyError, ValueError) as exc:
            log.warning("No acquisition time from %s (%s)", p.name, exc)
    return out


def _crop_name(pair, trk: int, d: str) -> str:
    return f"NISAR_GUNW_CROP_T{trk:03d}{d}_{pair[0]}_{pair[1]}.h5"


def _reusable(path: Path, src_name: str, lattice: Lattice, phase_linked: bool = False) -> bool:
    import h5py
    try:
        with h5py.File(path, "r") as f:
            # a crop holding the phase-linked phase (nisar_pl) is never the
            # GUNW route's: that route reads the granule again
            swapped = f.attrs.get("phase_source") == "dolphin"
            return (f.attrs.get("lattice") == lattice.signature() and f.attrs.get("granule") == src_name
                    and int(f.attrs.get("crop_version", 0)) == CROP_VERSION and (phase_linked or not swapped))
    except OSError:
        return False


def _fetch_one(cfg, cand: dict, out: Path, lattice: Lattice, auth, pol: str, height_range) -> dict:
    meta = {"granule": cand["name"], "url": cand.get("url"), "frame": cand["frame"],
            "bandwidth_mhz": cand["bandwidth_mhz"], "full_frame": cand["full_frame"], "counter": cand["counter"]}
    if out.exists() and bool(getattr(cfg, "skip_existing", True)) and \
            _reusable(out, cand["name"], lattice, bool(setting(cfg, "nisar_phase_linked"))):
        rec = read_crop_record(out)
        rec.update(crop_validity(out, lattice, tuple(cfg.bbox)))
        rec["reused"] = True
        rec["bytes_read"] = 0
        return rec
    mode = str(setting(cfg, "nisar_fetch"))
    for attempt in range(3):
        try:
            if mode == "download":
                return _download_crop(cfg, cand, out, lattice, auth, pol, height_range, meta)
            return crop_gunw(cand["url"], lattice, out, auth=auth, pol=pol, height_range=height_range,
                             aoi_bbox=tuple(cfg.bbox), meta=meta)
        except (OSError, TimeoutError) as exc:
            if attempt == 2:
                raise
            log.warning("  %s: read failed (%s); retrying in %d s", cand["name"], exc, 20 * (attempt + 1))
            time.sleep(20 * (attempt + 1))
    raise RuntimeError("unreachable")


def _download_crop(cfg, cand, out, lattice, auth, pol, height_range, meta) -> dict:
    """Whole-file fallback: download one granule, crop it, delete it."""
    import asf_search as asf
    tmp_dir = Path(cfg.project_dir) / "nisar_gunw" / "whole"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp = tmp_dir / (cand["name"] + ".h5")
    session = asf.ASFSession().auth_with_creds(*auth) if auth else asf.ASFSession()
    with session.get(cand["url"], stream=True, timeout=600) as r:
        r.raise_for_status()
        with open(tmp, "wb") as fh:
            for chunk in r.iter_content(1 << 22):
                fh.write(chunk)
    try:
        rec = crop_gunw(tmp, lattice, out, pol=pol, height_range=height_range, aoi_bbox=tuple(cfg.bbox),
                        meta={**meta, "fetched": "whole file, cropped on arrival and deleted"})
    finally:
        tmp.unlink(missing_ok=True)
    return rec


def run(cfg: PipelineConfig, username: str | None = None, password: str | None = None,
        use_aria: bool = False) -> tuple[Path, str]:
    """Search, choose, crop, screen and stage; returns (the crop folder MintPy
    reads, the MintPy loader block). use_aria is ignored: ARIA-tools is not
    installed and expects whole files (kept for pipeline.run_nisar's call)."""
    P = Path(cfg.project_dir) / "nisar_gunw"
    P.mkdir(parents=True, exist_ok=True)
    from . import corrections
    if corrections.decide(cfg).get("ionosphere") == "gim":
        raise ValueError("ionosphere_correction would apply the IGS global maps on the NISAR route, on top of the "
                         "product's own ionosphere screen (MintPy split_spectrum): set ionosphere_correction: \"no\"")
    auth = (username, password) if username and password else None
    record: dict = {"route": "nisar GUNW (windowed reads, lat/lon lattice, prep_nisar)", "written": None,
                    "settings": {k: setting(cfg, k) for k in SETTINGS},
                    "window": [cfg.start_date, cfg.end_date], "aoi_bbox": list(cfg.bbox)}

    records = search_gunw(cfg)
    if not records:
        raise RuntimeError("No PROVISIONAL NISAR GUNW over this AOI and window (PROVISIONAL covers acquisitions "
                           "from 17 Jun 2026, earlier only at cal/val frames)")
    record["groups_found"] = describe_groups(records)
    chosen = select_group(cfg, records)
    chosen, cal = exclude_calibration_frames(chosen)
    record["calibration_frames_dropped"] = cal
    dd = deduplicate(chosen)
    record["duplicates"] = {"stack_mode_mhz": dd["stack_mode_mhz"], "dominant_frame": dd["dominant_frame"],
                            "decisions": dd["decisions"], "mode_mixed": dd["mode_mixed"]}
    trk, d = int(cfg.track), ("A" if str(cfg.orbit_direction).lower() == "ascending" else "D")
    log.info("NISAR track %d%s: %d granules -> %d pairs (stack mode %s MHz, frame %s; %d with duplicates)",
             trk, d, len(chosen), len(dd["pairs"]), dd["stack_mode_mhz"], dd["dominant_frame"], len(dd["decisions"]))

    lattice = make_lattice(cfg.bbox, float(setting(cfg, "nisar_posting_m")), float(setting(cfg, "nisar_crop_margin_deg")))
    record["lattice"] = {"bbox_wsen": list(lattice.bbox()), "dlon": lattice.dlon, "dlat": lattice.dlat,
                         "shape": [lattice.ny, lattice.nx], "crs": "EPSG:4326",
                         "resampling": "nearest neighbour (layers); linear (metadata cube)"}
    dem = stage_dem(cfg, lattice)
    hr = dem_height_range(dem, lattice)
    record["dem"] = {"file": str(dem), "height_range_m": hr}

    crops_dir = P / "crops"
    pol = str(setting(cfg, "nisar_polarization") or "")
    pairs_out, total_bytes = [], 0

    def fetch(cand, out):
        nonlocal total_bytes
        rec = _fetch_one(cfg, cand, out, lattice, auth, pol, hr)
        total_bytes += int(rec.get("bytes_read", 0) or 0)
        return rec, {"granule": cand["name"], "covered": rec.get("covered"),
                     "valid_share_aoi": rec.get("valid_share_aoi")}

    for item in dd["pairs"]:
        pair = item["pair"]
        out = crops_dir / _crop_name(pair, trk, d)
        tried = []
        for i, cand in enumerate(item["candidates"]):
            rec, t = fetch(cand, out)
            tried.append(t)
            if rec.get("covered") and float(rec.get("valid_share_aoi") or 0) > 0.05:
                pol = pol or rec.get("polarization", "")
                pairs_out.append({"pair": pair, "file": out, "rec": rec, "tried": tried,
                                  "alternatives": item["candidates"][i + 1:]})
                log.info("  %s_%s  %s  valid %.0f%%  coherence %s  %.1f MB read", pair[0], pair[1],
                         cand["name"][-40:], 100 * float(rec["valid_share_aoi"]), rec.get("coherence_median_aoi"),
                         float(rec.get("bytes_read", 0)) / 1e6)
                break
        else:
            log.warning("  %s_%s: no granule holds usable data over the AOI - footprint, or valid share by "
                        "phase, components and the product mask (%s)", pair[0], pair[1], tried)
            record.setdefault("pairs_without_data", []).append({"pair": f"{pair[0]}_{pair[1]}", "tried": tried})

    # screen against the group: incomplete-L0B products, and footprints that
    # promised more of the AOI than the raster holds. A pair below the bar is
    # read again from its next-ranked granule (into a side file: a worse
    # alternative never replaces a better crop), and dropped if none does.
    ratio = float(setting(cfg, "nisar_min_valid_ratio"))
    med = float(np.median([float(p["rec"]["valid_share_aoi"]) for p in pairs_out])) if pairs_out else 0.0
    kept, screened = [], []
    for p in pairs_out:
        v = float(p["rec"]["valid_share_aoi"])
        while v < ratio * med and p["alternatives"]:
            cand = p["alternatives"].pop(0)
            alt = p["file"].with_name(p["file"].stem + "_alt.h5")
            rec, t = fetch(cand, alt)
            p["tried"].append(t)
            if rec.get("covered") and float(rec.get("valid_share_aoi") or 0) > v:
                alt.replace(p["file"])
                p["rec"], v = rec, float(rec["valid_share_aoi"])
            else:
                alt.unlink(missing_ok=True)
        if v < ratio * med:
            screened.append({"pair": f"{p['pair'][0]}_{p['pair'][1]}", "valid_share_aoi": v,
                             "group_median": round(med, 4), "why": f"valid share below {ratio} x the group median "
                                                                 "(incomplete L0B or decorrelated: ASF known issues)"})
        else:
            kept.append(p)
    record["screened_out"] = screened

    # the chain: keep the longest unbroken piece
    rep = chain_report([p["pair"] for p in kept])
    record["chain_before"] = rep
    segs = chain_segments([p["pair"] for p in kept])
    if len(segs) > 1:
        if str(setting(cfg, "nisar_chain_gap")) == "fail":
            raise RuntimeError(f"The NISAR chain is broken at {rep['gaps']} (pieces of {rep['pieces']} dates); "
                               "nisar_chain_gap is 'fail'")
        keep = set(segs[0])
        record["chain_dropped_pairs"] = [f"{p['pair'][0]}_{p['pair'][1]}" for p in kept if p["pair"] not in keep]
        log.warning("The NISAR chain is broken at %s: keeping the longest piece (%d dates) and dropping %s",
                    rep["gaps"], len({x for q in keep for x in q}), record["chain_dropped_pairs"])
        kept = [p for p in kept if p["pair"] in keep]
    if len(kept) < 2:
        raise RuntimeError(f"Only {len(kept)} usable NISAR pair(s) in one piece: no time series")
    record["chain"] = chain_report([p["pair"] for p in kept])
    record["chain"]["temporal_coherence"] = "1.0 by construction (no loops): not a quality test on this route"
    record["chain"]["phase_closure"] = "not applicable (no triangles)"

    # the phase-linked route (nisar_pl): dolphin's phase in every kept crop
    if bool(setting(cfg, "nisar_phase_linked")):
        from . import nisar_pl
        prep = nisar_pl.prepare(cfg, P)
        swaps = nisar_pl.swap_into_crops([p["file"] for p in kept], prep["work"])
        tcf = nisar_pl.write_temporal_coherence(prep["work"], kept[0]["file"], P / nisar_pl.TC_FILE)
        run = prep["run"]
        record["route"] = "nisar phase-linked: GUNW crops with dolphin's phase (L2 GSLC, 5 m) - nisar_pl.py"
        record["phase_linked"] = {"window": run.get("window"), "strides": run.get("strides"),
                                  "gslc_dates": len(prep["dates"]), "gslc_dates_left_out": prep["empty"],
                                  "dolphin_seconds": run.get("seconds"), "peak_rss_gb": run.get("peak_rss_gb"),
                                  "work": str(prep["work"]), "temporal_coherence": str(tcf), "swaps": swaps}
        record["chain"]["temporal_coherence"] = ("dolphin's phase-linking temporal coherence "
                                                 f"({nisar_pl.TC_FILE}), adopted by the products")

    # the stack MintPy reads: links to the kept crops only
    stack_dir = P / "stack"
    stack_dir.mkdir(parents=True, exist_ok=True)
    for old in stack_dir.glob("*.h5"):
        old.unlink()
    for p in kept:
        (stack_dir / p["file"].name).symlink_to(p["file"].resolve())

    # ionosphere: the product screen, applied by MintPy, sign checked on the data
    ion_stats = [p["rec"].get("ionosphere_sign") or {} for p in kept]
    ion_sum = summarise_ionosphere_sign(ion_stats)
    use_ion = str(setting(cfg, "nisar_ionosphere")) == "split_spectrum" and any(
        p["rec"].get("ionosphere_sign") is not None for p in kept)
    record["ionosphere"] = {"method": "split_spectrum (product screen, MintPy iono_split_spectrum: subtracted)"
                            if use_ion else "none", "sign_check": ion_sum}
    if use_ion and ion_sum.get("verdict") == "opposite sign":
        _write_record(P, record, kept)
        raise RuntimeError("The NISAR ionosphere screen correlates NEGATIVELY with the unwrapped phase in most "
                           f"informative pairs ({ion_sum}): subtracting it would add the ionosphere twice. "
                           "Investigate (nisar_gunw/nisar_stack.json: per pair 'ionosphere_sign'). To run without "
                           "the screen, set nisar_ionosphere: no in the config")
    if use_ion and str(ion_sum.get("verdict", "")).startswith("doubtful"):
        log.warning("NISAR ionosphere screen: %s (%d of %d informative pairs fit better with it added); see "
                    "nisar_gunw/nisar_stack.json", ion_sum["verdict"], ion_sum["add_better"],
                    ion_sum["informative_pairs"])
    record["troposphere"] =(f"{getattr(cfg, 'mintpy_tropo_method', 'pyaps')} (MintPy); the GUNW troposphere "
                             "screens (inputs/tropoStack.h5) are written by prep_nisar but not used")
    record["solid_earth_tide"] = "MintPy correct_SET when solid_earth_tides applies; the GUNW tide layer is not used"
    record["sign_convention"] = ("chronological pairs (reference earlier); positive unwrapped phase = range increase; "
                                 "MintPy -lambda/4pi: subsidence negative. Reversed pairs: " +
                                 str(sum(1 for p in kept if p["rec"].get("reversed_by_pipeline"))))
    # one wavelength for the stack (pairs taken in the other bandwidth mode)
    target_wl = stack_wavelength([p["rec"] for p in kept])
    scaled = harmonise_wavelength([p["file"] for p in kept], target_wl) if target_wl else []
    for p in kept:
        p["rec"]["wavelength_m"] = target_wl or p["rec"]["wavelength_m"]
        for s in scaled:
            if s["file"] == str(p["file"]):
                p["rec"]["phase_scale"] = s["phase_scale"]
    native = sorted({round(float(p["rec"].get("wavelength_native_m", p["rec"]["wavelength_m"])), 6) for p in kept})
    record["sensor"] = {"wavelength_m": target_wl, "half_wavelength_mm": round(1000 * target_wl / 2, 2) if target_wl
                        else None, "native_wavelengths_m": native, "pairs_rescaled_to_stack_wavelength": scaled,
                        "look_direction": sorted({p["rec"].get("look_direction", "") for p in kept}),
                        "polarization": pol or sorted({p["rec"].get("polarization") for p in kept}),
                        "los_vector_encoding": los_encoding_for_prep_nisar()[0]}
    gate = coherence_gate([p["file"] for p in kept], lattice, P / "coherence_gate.tif",
                          float(getattr(cfg, "min_temporal_coherence", 0.4)))
    record["coherence_gate"] = gate
    method, why = unwrap_error_method(cfg)
    record["unwrap_error"] = {"method": method, "note": why}
    record["fetch"] = {"bytes_read_total": total_bytes, "pairs": len(kept),
                       "whole_files_avoided_gb": round(sum(2.4 for _ in kept), 1),
                       "mode": str(setting(cfg, "nisar_fetch"))}
    _write_record(P, record, kept)
    loader = mintpy_loader_lines(cfg, stack_dir, dem, use_ion)
    pols = {p["rec"].get("polarization") for p in kept}
    if pols - {"HH"}:
        if len(pols) > 1:
            raise RuntimeError(f"The stack mixes polarisations {sorted(pols)}")
        prep_other_polarization(cfg, stack_dir, dem, pols.pop())
    log.info("NISAR: %d pairs %s..%s staged in %s (%.0f MB read instead of ~%.0f GB of whole granules)",
             len(kept), kept[0]["pair"][0], kept[-1]["pair"][1], stack_dir, total_bytes / 1e6,
             record["fetch"]["whole_files_avoided_gb"])
    return stack_dir, loader


def posting_note(cfg) -> str | None:
    """A run note when this project's crops are on another grid than the
    config asks for (28 Oct 2026, Kyle: a Products-only rerun after the 15 m
    default still showed 80 m blocks, and nothing said why). The grid is set
    when the crops are made, so only Whole pipeline (which re-crops) changes
    it. None when they agree, or when there is no stack record yet."""
    try:
        rec = json.loads((Path(cfg.project_dir) / "nisar_gunw" / "nisar_stack.json").read_text())
        made = float((rec.get("settings") or {})["nisar_posting_m"])
    except (OSError, ValueError, KeyError, TypeError):
        return None
    want = float(setting(cfg, "nisar_posting_m"))
    if math.isclose(made, want, rel_tol=1e-6):
        return None
    return (f"These products are on the {made:g} m grid the NISAR data were cropped to on "
            f"{str(rec.get('written') or '?')[:10]}; the config asks for {want:g} m. Only Whole pipeline "
            f"re-crops the data onto the {want:g} m grid.")


def _write_record(P: Path, record: dict, kept: list) -> Path:
    record["written"] = time.strftime("%Y-%m-%d %H:%M:%S")
    record["pairs"] = [{"pair": f"{p['pair'][0]}_{p['pair'][1]}", "file": str(p["file"]),
                        **{k: v for k, v in p["rec"].items() if k not in ("source",)},
                        "tried": p["tried"]} for p in kept]
    out = P / "nisar_stack.json"
    out.write_text(json.dumps(record, indent=1, default=str))
    return out


def _size(p: Path) -> int:
    if p.is_symlink() or p.is_file():
        return p.lstat().st_size if p.is_symlink() else p.stat().st_size
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file() and not f.is_symlink())


def cleanup(cfg, level: int, delete: bool = False, confirm=input) -> dict:
    """The NISAR cleanup entry (until scripts/cleanup.py calls
    cleanup_targets): list, and with delete=True remove, what
    cleanup_targets offers up to `level`. Level 2 is refused until MintPy has
    loaded the stack (mintpy/inputs/ifgramStack.h5) and no run is marked
    incomplete (mintpy/run_incomplete.json): before then the crops are the
    only copy of the pairs. Deleting asks for a typed 'yes', as insar cleanup
    does. Nothing outside <project>/nisar_gunw is ever touched."""
    import os
    import shutil
    P = Path(cfg.project_dir) / "nisar_gunw"
    items = [(lv, p, why) for lv, p, why in cleanup_targets(cfg) if lv <= int(level)]
    M = Path(cfg.mintpy_dir)
    refused = None
    if any(lv == 2 for lv, _, _ in items):
        if not (M / "inputs" / "ifgramStack.h5").exists():
            refused = "MintPy has not loaded the stack (no mintpy/inputs/ifgramStack.h5): the crops are its only input"
        elif (M / "run_incomplete.json").exists():
            refused = "mintpy/run_incomplete.json: the last MintPy run did not finish"
        if refused:
            items = [x for x in items if x[0] < 2]
    dirs = [p for lv, p, _ in items if lv == 2]
    items = [x for x in items if not any(d in Path(x[1]).parents for d in dirs)]
    rows =[{"level": lv, "path": str(p), "why": why, "bytes": _size(p)} for lv, p, why in items]
    out = {"items": rows, "total_bytes": sum(r["bytes"] for r in rows), "deleted": [], "level_2_refused": refused}
    if not delete or not rows:
        return out
    if str(confirm(f"Delete {len(rows)} item(s), {out['total_bytes'] / 1e9:.2f} GB, under {P}? Type yes: ")).strip() != "yes":
        out["cancelled"] = True
        return out
    base = Path(os.path.abspath(Path(cfg.project_dir) / "nisar_gunw"))
    for lv, p, _ in items:
        q = Path(os.path.abspath(p))            # the link itself, never what it points to
        if base not in q.parents:
            raise RuntimeError(f"refusing to delete {q}: not under {base}")
        if q.is_dir() and not q.is_symlink():
            shutil.rmtree(q)
        else:
            q.unlink(missing_ok=True)
        out["deleted"].append(str(q))
    return out


def main(argv=None) -> int:
    """python -m insar_pipeline.nisar_backend cleanup CFG --level N [--delete]"""
    import argparse
    from .config import load_config
    ap = argparse.ArgumentParser(description="NISAR route utilities")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("cleanup", help="list (dry run) or delete the NISAR route's files: level 1 whole granules and "
                                       "unused crops, level 2 the crops MintPy has loaded")
    c.add_argument("config")
    c.add_argument("--level", type=int, default=1, choices=(1, 2))
    c.add_argument("--delete", action="store_true")
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    rec = cleanup(cfg, a.level, a.delete)
    for r in rec["items"]:
        print(f"  level {r['level']}  {r['bytes'] / 1e6:10.1f} MB  {r['path']}  ({r['why']})")
    print(f"{len(rec['items'])} item(s), {rec['total_bytes'] / 1e9:.2f} GB"
          + ("" if a.delete else " - dry run: add --delete to remove them"))
    if rec["level_2_refused"]:
        print(f"level 2 not offered: {rec['level_2_refused']}")
    if rec.get("cancelled"):
        print("cancelled")
    return 0


def physical_sign_check(los_mm, inside, outside) -> dict:
    """A route's sign on a known subsidence (gaps M2: "subsidence must come out
    negative"): the median LOS change inside known sinking ground minus the
    median outside. Negative is the expected sign for subsidence on either
    look direction (LOS up-component > 0)."""
    a = np.asarray(los_mm, dtype="float64")
    i = a[np.asarray(inside, bool) & np.isfinite(a)]
    o = a[np.asarray(outside, bool) & np.isfinite(a)]
    if i.size < 5 or o.size < 5:
        return {"testable": False, "cells_inside": int(i.size), "cells_outside": int(o.size)}
    diff = float(np.median(i) - np.median(o))
    return {"testable": True, "inside_minus_outside_mm": round(diff, 1), "sign_ok": diff < 0,
            "cells_inside": int(i.size), "cells_outside": int(o.size)}


if __name__ == "__main__":
    raise SystemExit(main())
