"""
Quality assessment of a delivered InSAR time series against the practice of
the field, plus the site measurements the client asked for.

WHAT THIS IS
------------
Not a comparison with another product. The record is judged on its own
terms: is it referenced to stable ground and does that choice matter; how
precise is a rate and a single epoch; how much of the site is measured;
is the network sound; and does the product carry the known systematic of
distributed scatterers over vegetated ground. Each check states the value
it uses and where that value comes from.

THE YARDSTICKS
--------------
  Reference stability   A reference is chosen on stable ground and CHECKED:
                        its own series against the mean of the other stable
                        (high-coherence) ground should show |trend| <= 1 mm/yr
                        (Hooper et al. 2007 and every PSI/SBAS practice
                        guide). The choice must not move the headline
                        figures - reported under every candidate.
  Rate precision        Multi-year Sentinel-1 products quote ~1 mm/yr (1 sigma)
                        on persistent scatterers (EGMS product specification:
                        0.7 mm/yr at coherence > 0.7) and 1-3 mm/yr on
                        distributed scatterers, figures validated against
                        GNSS - i.e. EMPIRICAL errors. Graded on the empirical
                        repeatability of a cell's rate (significance.
                        empirical_rate_model): median <= 2 mm/yr. MintPy's
                        formal sigma assumes independent epochs and is shown
                        beside it, never graded on (until 27 Sep 2026 it was,
                        and passed at 0.36 mm/yr against an empirical ~5).
  Single-epoch noise    C-band single-interferogram displacement precision is
                        a few mm over coherent ground and degrades to ~10 mm
                        over vegetation (Hanssen 2001; Zebker & Villasenor
                        1992 for decorrelation). Robust epoch-step RMS far
                        from the reference <= 10 mm.
  Coverage              Operational services and vendors report 70%+ of the
                        AOI measurable over agricultural terrain; MintPy's
                        default reliability threshold is temporal coherence
                        0.7 (Yunjun et al. 2019), lowered deliberately here.
                        Reported at the working threshold AND at 0.7.
  Network               Redundancy >= 3 connections per date (Yunjun 2019),
                        one connected component, and the fraction of pairs
                        inverted.
  DS phase bias         Distributed scatterers in short-temporal-baseline
                        networks carry a systematic phase bias that reads as
                        a negative rate over vegetated ground (Ansari, De Zan
                        & Parizzi 2021; Zheng et al. 2022). Detected as a
                        monotonic velocity difference between coherence
                        classes far from any deformation. A difference over
                        3 mm/yr FLAGS the product: absolute rates on
                        low-coherence ground are biased low; use relative
                        rates or apply a closure-phase bias correction.

Site measurements (findings, not pass/fail): the workings relative to the
surrounding farmland; the named panel's step across the event window; the
feature of interest; the extent of ground moving faster than the
deforming-area threshold - all RELATIVE to the far field and MATCHED BY
COHERENCE CLASS, which cancels the reference exactly and the DS bias to the
extent it is a function of coherence. Each regional figure carries its
farmland null (significance.assess): the spread of the same measurement
over ordinary farmland of the same footprint. That spread, not a fitted
line's standard error, is the uncertainty to quote.

SITE PROFILE (config site_profile, 8 Oct 2026)
"Farmland" above is the default profile, the Indiana one. Which flags the
ground explains, and whether the send rule may treat them as notes, comes
from SITE_PROFILES (farmland, urban, mixed, mine_longwall,
groundwater_basin; annotate_site). On any profile the record's prose says
the comparison ground's words instead of "farmland": the config's
control_ground_label when set, else its measured land cover
(landcover.comparison_ground, 28 Oct 2026 - "farmland" where cropland
dominates). On any profile but farmland an expected flag still holds the
report until the figures' way of coping with it has been checked on that
ground.
"""
import json
import logging
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

MM_PER_IN = 25.4
# Every check the send rule reads, in the order the QA writes them.
CHECKS = ("reference_stability", "rate_precision", "single_epoch_noise", "coverage", "network",
          "distributed_scatterer_bias", "stitch")
# Which flags the GROUND explains, per kind of ground (config site_profile,
# V3 8 Oct 2026; it replaces the farmland-only table EXPECTED_ON_FARMLAND).
#   where     how a sentence says where the run is ("expected on farmland")
#   expected  {check: why the ground explains a flag on it, in plain words}
#   handled   whether the figures' way of coping (class matching, the null,
#             relative rates) has been CHECKED on this ground. Only then is an
#             expected flag a note in the send rule; otherwise it still holds
#             the report, with the reason said, until the validation
#             experiments (review_2026-10-08/PLAN.md) have checked it.
#   whole_area_moves  the far field itself moves with the signal, so figures
#             relative to it cancel the shared motion (send_rule holds).
# farmland: measured on every record so far (Site A classic and
# phase-linked, Site B): the reference trend, rate precision and DS bias are
# flagged on row-crop ground by its seasonal decorrelation, and every reported
# figure is built to cope with exactly that. A flag on coverage, the network,
# single-epoch noise or the DISP stitching is not explained by any ground here
# and needs a specialist before anything is sent.
SITE_PROFILES = {
    "farmland": {
        "where": "on farmland",
        "expected": {
            "reference_stability": "row-crop ground decorrelates with the seasons, so the reference wanders against "
                                   "other high-coherence ground",
            "rate_precision": "seasonal decorrelation makes a single cell's rate repeat poorly",
            "distributed_scatterer_bias": "low-coherence crop ground carries the short-baseline phase bias of "
                                          "distributed scatterers",
        },
        "handled": True, "whole_area_moves": False,
    },
    "urban": {
        "where": "on built-up ground",
        # buildings and paving hold their phase (persistent scatterers): none
        # of the three is explained by the ground, so each points at the processing
        "expected": {},
        "handled": False, "whole_area_moves": False,
    },
    "mixed": {
        "where": "on mixed built-up and vegetated ground",
        # built-up ground gives the reference stable neighbours, so a reference
        # drift is not explained by the ground
        "expected": {
            "rate_precision": "the vegetated part decorrelates, so a single cell's rate there repeats poorly",
            "distributed_scatterer_bias": "the vegetated part carries the short-baseline phase bias of distributed "
                                          "scatterers",
        },
        "handled": False, "whole_area_moves": False,
    },
    "mine_longwall": {
        "where": "at a longwall mining site",
        # the reference must sit on stable ground outside the angle of draw: a
        # drift there is exactly what such a site has to catch, so it holds
        "expected": {
            "rate_precision": "the ground around the panels is mostly forest or pasture, which decorrelates at "
                              "C-band, so a single cell's rate repeats poorly",
            "distributed_scatterer_bias": "forest and pasture carry the short-baseline phase bias of distributed "
                                          "scatterers",
        },
        "handled": False, "whole_area_moves": False,
    },
    "groundwater_basin": {
        "where": "in a subsiding groundwater basin",
        "expected": {
            "reference_stability": "no ground in the basin is stable, and the reference and the other "
                                   "high-coherence ground sink at different rates",
            # the check is the spread over far-field cells of (second-half
            # rate - first-half rate) about its median: a change shared by the
            # whole far field cancels, one that differs from place to place does not
            "rate_precision": "the far field itself sinks, by amounts that differ from place to place and change "
                              "between dry and wet years, so a cell's rates from the two halves of the record "
                              "differ for real",
            "distributed_scatterer_bias": "land use sets both coherence and how fast the ground sinks, so a "
                                          "difference between coherence classes can be real subsidence that the "
                                          "check cannot tell from the bias",
        },
        "handled": False, "whole_area_moves": True,
    },
}
# What the figures are measured against, in every profile but farmland, when
# the config gives no control_ground_label
NEUTRAL_GROUND = "comparison ground"
# The send-rule reason for a profile whose far field moves with the signal
WHOLE_AREA_HOLD = ("the whole area moves {where}: figures relative to the far field cancel the motion the far field "
                   "shares, so they understate it; absolute rates need a datum tied to GNSS")


def expected_flags(profile: str = "farmland") -> dict:
    """{check: True when the ground of this site profile explains a flag on it}."""
    exp = SITE_PROFILES[profile]["expected"]
    return {c: c in exp for c in CHECKS}


# The farmland table, kept under its old name for the readers that list the
# checks (the user-guide test); the send rule reads SITE_PROFILES.
EXPECTED_ON_FARMLAND = expected_flags("farmland")
# What a flag on each check means, in plain words (release.blockers and the
# coworker guide print these; `note` stays the specialist's explanation)
PLAIN = {
    "reference_stability": ("the reference point drifts against other stable ground by more than 1 mm/yr; absolute "
                            "rates carry that drift, the figures relative to farmland do not"),
    "rate_precision": ("a single cell's rate repeats to worse than 2 mm/yr; area figures average many cells and "
                       "quote the farmland spread as their uncertainty"),
    "single_epoch_noise": "single-date readings far from the reference are noisier than 10 mm",
    "coverage": ("less than 70% of the area is measured, or the coherence threshold sits inside the noise level, "
                 "so 'measured' does not mean reliable"),
    "network": "the interferogram network is broken into pieces or too thin to check itself",
    "distributed_scatterer_bias": ("low-coherence farmland reads more negative than high-coherence ground; every "
                                   "figure compares ground with farmland of the same coherence, which removes it"),
    "stitch": "the OPERA DISP-S1 granules were not stitched cleanly (see problems)",
}
SEND_VALUES = ("send", "send with notes", "hold for specialist")
# What holds the report, the first half of send_rule's 'rule' (the farmland
# wording; other site profiles say their comparison ground instead)
RULE_HOLD = ("hold for specialist: any unexpected flag, an untested check, thin farmland, a report area "
             "that cannot be compared with farmland, an unexplained missing date, a geolocation offset "
             "over 10 m (or one half more than one map cell off when the halves disagree beyond their own "
             "scatter), a figure moving more than 1 null "
             "sigma with the deramp or matching, >5% too-fast cells in a report area, one of those "
             "diagnostics failing to run, or a failed GNSS tie")
# send_rule's 'experiment' entry on a record from an experiment run
EXPERIMENT_SEND_NOTE = ("an experiment run (config purpose: experiment): no client report is built from it, so "
                        "nothing is sent; this value is kept only to compare with client projects, and the "
                        "experiment's result is the validation against ground truth")
# Second-gate levels whose effect on coverage and figures is measured
# (Kyle, 5 Oct 2026: measure, then decide; nothing is gated by default).
SECOND_GATE_LEVELS = (0.4, 0.5, 0.7)
# runs of noise-like consecutive pairs (geo_disturbed.h5) whose effect is
# recorded: measured, never applied
DISTURBED_RUNS = (10, 20, 30, 60)
# DISP-S1: share of cells carrying a whole-cycle error from a reference
# anchor above which the stitch check fails (disp_chain anchor_suspect_cells;
# below it the count is a note). Provisional, as the review recommended.
ANCHOR_SUSPECT_SHARE = 0.01
# MintPy's default reliability threshold (Yunjun et al. 2019) on its own
# network temporal coherence: the "passes both gates" share is graded on it.
RELIABLE_UNWRAP_CONSISTENCY = 0.7
# The farmland control is too thin to test a report area when it holds
# fewer cells than this or the widest farmland fits the largest report area
# fewer than FAR_MIN_RATIO times over (significance.assess calls an area with
# fewer than 5 independent placements not testable, under 10 thin).
FAR_MIN_CELLS = 2000
FAR_MIN_RATIO = 5.0
# The report areas are cut from the comparison ground (report_farmland_masks)
# unless that leaves fewer farmland cells than one coherence class needs for
# its own centre (significance.class_means' min_n): nothing could be measured.
CUT_MIN_CELLS = 200
# DEM-error position shift: a report area or the mine edges carrying this
# many percentage points MORE cells displaced by over one posting cell than
# the farmland does is worth a warning (verify note on gap-geolocation-4: "a
# few percent"). Against farmland, because demErr is noisy: on Site B 11%
# of ALL coherent cells exceed 15 m (farmland median |demErr| 5 m), the mine
# polygons and their edges 10-11%, i.e. nothing beyond the noise there.
DEM_SHIFT_WARN_PERCENT = 3.0
# Geolocation against mapped water: flag above this offset (the reviewer
# measured <= 7.5 m on both records; half a 15 m posting cell).
GEOLOCATION_FLAG_M = 10.0
# The multi-scale stand-out (multi_scale_standout, decision E7 7 Oct 2026):
# the deepest window at each scale against the deepest ordinary window more
# than STANDOUT_AWAY_M away, in robust spreads; it stands out when that gap
# beats the STANDOUT_LEVEL percentile of the LARGEST gap over the three
# scales on STANDOUT_COPIES simulated farmland copies (standout_null).
# Until 7 Oct the rule was gap > 2 at any one scale, never calibrated: on
# Gaussian fields with the measured farmland covariance it fired in 6.6% of
# Site B reports (0.8% Site A), on rolled real farmland in 19-24%.
STANDOUT_SCALES_KM = (1.0, 2.0, 4.0)
STANDOUT_AWAY_M = 5000.0
STANDOUT_COPIES = 199              # the 95th percentile lies between the 11th and 10th largest
STANDOUT_LEVEL = 95.0
STANDOUT_LAG_M = 8000.0            # measured farmland covariance kept to twice the largest window
STANDOUT_TAPER_M = 2000.0
STANDOUT_SEED = 0
STANDOUT_MATCH = (0.8, 1.25)       # copies' window spread against the farmland's: said, not gated on
# A half of the area is held against the positions only when it is more than
# one posting cell off AND the halves disagree beyond their own scatter
# (decision E22, 7 Oct 2026; judge_geolocation). The cell when a record does
# not carry its own cell_m: the 15 m product grid of every record so far.
GEOLOCATION_CELL_M = 15.0
GEOLOCATION_SPLIT_ALPHA = 0.05
YARDSTICK = {
    "reference_trend_mm_yr": 1.0,
    "rate_sigma_median_mm_yr": 2.0,
    "rate_sigma_high_coh_mm_yr": 1.0,
    "epoch_step_robust_mm": 10.0,
    "coverage_percent": 70.0,
    "redundancy": 3.0,
    "ds_bias_mm_yr": 3.0,
}


def _read2d(path, name=None):
    import h5py
    with h5py.File(path, "r") as f:
        if name is None:
            keys = [k for k in f.keys() if f[k].ndim == 2]
            name = keys[0]
        arr = f[name][:]
        attrs = {k: (v.decode() if isinstance(v, bytes) else v) for k, v in f.attrs.items()}
    return arr, attrs


def _dates(f):
    return [d.decode() if isinstance(d, bytes) else str(d) for d in f["date"][:]]


def _years(dates):
    return np.array([(datetime.strptime(d, "%Y%m%d") - datetime(2000, 1, 1)).days for d in dates]) / 365.25


def _fit(t, y):
    ok = np.isfinite(y)
    if ok.sum() < 4:
        return None
    A = np.c_[t[ok], np.ones(ok.sum())]
    coef, res, _, _ = np.linalg.lstsq(A, y[ok], rcond=None)
    resid = y[ok] - A @ coef
    dof = max(1, ok.sum() - 2)
    cov = (resid @ resid / dof) * np.linalg.inv(A.T @ A)
    return {"slope": float(coef[0]), "stderr": float(np.sqrt(cov[0, 0])),
            "scatter": float(np.std(resid)), "n": int(ok.sum())}


def _rc(attrs, lat, lon):
    """MintPy's geo lat/lon -> row/col (floor, not round); see significance.lalo_to_rc."""
    from insar_pipeline.significance import lalo_to_rc
    return lalo_to_rc(attrs, lat, lon)


def _polygon(shape, attrs, geojson):
    from insar_pipeline.stack_backend import _polygon_mask
    return _polygon_mask(Path(geojson), shape, attrs)


def _buffer(mask, attrs, metres):
    """True-distance buffer (the old cross-element dilation was an L1 diamond)."""
    from insar_pipeline.significance import euclidean_buffer
    return euclidean_buffer(mask, attrs, metres)


def evaluate(cfg, fetch_water: bool = False) -> dict:
    """
    The assessment. `fetch_water` allows one Overpass request for OSM water
    (cached in <project>/context/osm_water.geojson) for the geolocation
    check. Off by default so library and test calls never touch the
    network; the pipeline stage and the CLI turn it on. Without it the
    check runs only on a water map already on disk (context/water.geojson
    or context/osm_water.geojson).
    """
    import h5py
    from insar_pipeline import deliverables as dl
    from insar_pipeline import significance as sg
    from insar_pipeline import stack_qc

    out = {"yardsticks": YARDSTICK, "checks": {}, "findings": {}}
    ts_file = dl.timeseries_file(cfg)
    vel_f = dl._product(cfg, "velocity.h5")
    mask_f = dl._product(cfg, "maskTempCoh.h5")
    tc_f = dl._product(cfg, "temporalCoherence.h5")

    vel_m, attrs = _read2d(vel_f, "velocity")
    vel = vel_m * 1000.0                                          # mm/yr
    tc, tc_attrs = _read2d(tc_f, "temporalCoherence")
    good = np.isfinite(vel)
    if mask_f.exists():
        m, _ = _read2d(mask_f, "mask")
        good &= m.astype(bool)
    ny, nx = vel.shape
    sig = None
    with h5py.File(vel_f, "r") as f:
        if "velocityStd" in f:
            sig = f["velocityStd"][:] * 1000.0
    measured = good.copy()                                         # first gate only (the delivered mask)
    # the second gate (unwrapping consistency, MintPy's own network temporal
    # coherence) - mapped onto the product grid when MintPy left it in radar
    # coordinates; applied to every figure here only when the config asks
    # (min_unwrap_consistency > 0), so the QA grades what the report uses
    rg = _RadarLayers(cfg, attrs, vel.shape)
    um, um_src = _unwrap_consistency(cfg, rg, tc, tc_attrs)
    gate = float(cfg.min_unwrap_consistency or 0.0)
    gate_applied = False
    if gate > 0:
        if um is not None:
            good &= np.nan_to_num(um, nan=-1.0) >= gate
            gate_applied = True
        else:
            log.warning("min_unwrap_consistency = %.2f but no unwrapping-consistency layer was found; "
                        "the QA grades the first gate only", gate)

    # ---------------- regions: one far-field definition for every figure
    panel_geojson = getattr(cfg, "acceptance_panel_geojson", "") or None
    # the report areas other than the whole study area are not comparison
    # ground (E28, as the client report; F7(d)): every far field below,
    # the null's included, is drawn without them
    # far: the control, AOI farmland; far_wide: where null placements may go
    far, far_wide, cut, fmeta = report_farmland_masks(cfg, good, attrs)
    ground = fmeta["comparison_ground"]
    # the comparison ground in words, from its measured land cover
    # (landcover.comparison_ground, 28 Oct 2026; the report measures its own
    # the same way): annotate_site words the record with it, the figures
    # read it from here
    from insar_pipeline import landcover
    out["comparison_ground"] = landcover.comparison_ground(cfg, attrs, far_wide if far_wide is not None else far)
    mine, inside = fmeta["mine"], fmeta["aoi"]
    buf_m = fmeta["buffer_m"]
    stable = far & (tc >= 0.85)                                    # the high-coherence population
    r_cm, cmeans = sg.class_matched_field(vel, tc, far)
    # Null placements range over far_wide, which reaches outside the box onto
    # ground with its own class centres (Site B: +1.86 mm/yr). The
    # placement field re-centres that ring on its own classes; inside the AOI
    # it IS r_cm, so every reported value is unchanged (stats-4).
    r_place = sg.placement_field(vel, tc, far, far_wide, inside)
    # coherence classes with too few far-field cells to centre on their own
    # borrow the nearest measured class; a report area resting on one is
    # compared with farmland of ANOTHER coherence (stats-3)
    kcls = sg.coherence_class(tc)
    borrowed = sorted(int(k) for k in (sg.class_means(vel, tc, far).get("_borrowed") or [])
                      if (good & inside & (kcls == k)).any())          # classes the AOI actually holds
    labels = sg.class_labels()
    ref = sg.reference_pixel(attrs, cfg)
    out["reference_pixel"] = ref
    cut_buf = ground.get("buffer_m") or sg.EXCLUDE_BUFFER_M
    out["regions"] = {"mine_px": int((mine & good).sum()), "far_field_px": int(far.sum()),
                      "far_field_wide_px": int(far_wide.sum()), "stable_px": int(stable.sum()), "buffer_m": buf_m,
                      "far_field_definition": (f"coherent cells inside the AOI, outside every mapped panel + {buf_m:.0f} m "
                                               "(Euclidean distance)"
                                               + (f", and outside the report areas other than the whole study area "
                                                  f"+ {cut_buf:.0f} m (as the client report)" if ground["applied"] else "")),
                      "report_areas_cut": ground["report_areas_cut"],
                      "report_area_cells_removed": ground["cells_removed"],
                      "mine_polygon": fmeta["panel_polygon"], "panel_polygon": panel_geojson}
    if ground.get("reason"):
        out["regions"]["report_areas_not_cut"] = ground.get("report_areas_not_cut")
        out["regions"]["report_areas_not_cut_reason"] = ground["reason"]
    out["far_field"] = {"n_cells": int(far.sum()),
                        "mean_rate_mm_yr": round(float(vel[far].mean()), 2),
                        "median_rate_mm_yr": round(float(np.median(vel[far])), 2),
                        "class_mean_rate_mm_yr": {labels[k]: round(v, 3) for k, v in cmeans.items()},
                        "borrowed_classes": [labels[k] for k in borrowed]}

    kcls = sg.coherence_class(tc)
    log.info("QA: regions ready (far field %d px, workings %d px); reading the series", int(far.sum()), int((mine & good).sum()))
    with h5py.File(ts_file, "r") as f:
        dates = _dates(f)
        key = "timeseries" if "timeseries" in f else [k for k in f if f[k].ndim == 3][0]
        cube = f[key]
        t = sg.years(dates)
        mine_good = mine & good
        far_class_masks = {k: far & (kcls == k) for k in cmeans}
        far_series = np.empty(len(dates)); stable_series = np.empty(len(dates))
        mine_series = np.empty(len(dates)) if mine_good.any() else None
        far_class_series = {k: np.empty(len(dates)) for k in far_class_masks}
        roi = roi_series = None
        if panel_geojson and Path(panel_geojson).exists():
            roi = _polygon(vel.shape, attrs, panel_geojson) & good
            roi_series = np.empty(len(dates))
        for i, ep in sg.iter_epochs(cube):
            sl = ep * 1000.0
            far_series[i] = np.nanmean(sl[far])
            stable_series[i] = np.nanmean(sl[stable]) if stable.sum() > 50 else far_series[i]
            for k, fm in far_class_masks.items():
                far_class_series[k][i] = np.nanmean(sl[fm])
            if mine_series is not None:
                mine_series[i] = np.nanmean(sl[mine_good])
            if roi is not None:
                roi_series[i] = np.nanmean(sl[roi])
        # the reference and its candidates: the pixel's own series. The
        # reference is the pixel MintPy referenced (REF_Y/REF_X), not the
        # configured lat/lon rounded onto the grid - on the record those are
        # two rows apart and the old check measured the wrong pixel.
        cands = []
        if ref:
            cands.append(("reference", ref["lat"], ref["lon"], ref["row"], ref["col"]))
        for i, s in enumerate(getattr(cfg, "reference_candidates_lalo", ()) or ()):
            la, lo = (float(v) for v in s.split(","))
            cands.append((f"candidate {i + 1}", la, lo, *sg.lalo_to_rc(attrs, la, lo)))
        cand_series = {name: (la, lo, cube[:, r, c] * 1000.0) for name, la, lo, r, c in cands
                       if 0 <= r < ny and 0 <= c < nx}

    def class_matched_series(region_series, region_mask):
        """region series minus the far-field class series weighted by the region's own coherence mix"""
        w = {k: int((region_mask & (kcls == k)).sum()) for k in far_class_series}
        tot = sum(w.values())
        if not tot:
            return region_series - far_series
        return region_series - sum(far_class_series[k] * (n / tot) for k, n in w.items())

    log.info("QA: series read; reference stability")
    # ---------------- 1. reference stability
    ref_fit = sg.fit_rate(t, cand_series["reference"][2] - stable_series) if "reference" in cand_series else None
    ref_resid = None
    if ref_fit:
        e = (cand_series["reference"][2] - stable_series) - (ref_fit["slope"] * t + ref_fit["intercept"])
        ref_resid = float(np.nanstd(e))
    sens = []
    for name, (la, lo, s) in cand_series.items():
        fr = sg.fit_rate(t, s - stable_series)
        sens.append({"name": name, "lat": la, "lon": lo,
                     "trend_vs_stable_ground_mm_yr": round(fr["slope"], 2) if fr else None,
                     "trend_stderr_ar1_mm_yr": round(fr["stderr_ar1"], 2) if fr else None,
                     "scatter_vs_stable_ground_mm": (round(float(np.nanstd(s - stable_series - (fr["slope"] * t + fr["intercept"]))), 2)
                                                     if fr else None)})
    # The figures that DO depend on a choice: how the far field is drawn.
    # Relative figures cannot depend on the reference pixel (subtracting any
    # pixel's series from both region and far field cancels exactly), so
    # re-referencing is not a robustness test of them.
    fsens = []
    if mine_good.any():
        def cm_value(far_variant):
            rv, _ = sg.class_matched_field(vel, tc, far_variant)
            return round(float(np.nanmean(rv[mine_good])), 2)
        for label, fv in (("AOI farmland, 2 km buffer (headline)", far),
                          ("all farmland in the product grid", far_wide)):
            fsens.append({"far_field": label, "workings_class_matched_mm_yr": cm_value(fv)})
        for b in (1000.0, 4000.0):
            fv, _ = sg.far_field_mask(cfg, good, attrs, buffer_m=b, exclude=cut)
            fsens.append({"far_field": f"AOI farmland, {b / 1000:.0f} km buffer", "workings_class_matched_mm_yr": cm_value(fv)})
        fsens.append({"far_field": "AOI farmland, plain mean (no class matching)",
                      "workings_class_matched_mm_yr": round(float(vel[mine_good].mean() - vel[far].mean()), 2)})
        fsens.append({"far_field": "AOI farmland, plain median (no class matching)",
                      "workings_class_matched_mm_yr": round(float(np.median(vel[mine_good]) - np.median(vel[far])), 2)})
    vals = [x["workings_class_matched_mm_yr"] for x in fsens]
    out["checks"]["reference_stability"] = {
        "value_mm_yr": round(ref_fit["slope"], 2) if ref_fit else None,
        "stderr_ar1_mm_yr": round(ref_fit["stderr_ar1"], 2) if ref_fit else None,
        "scatter_mm": round(ref_resid, 2) if ref_resid is not None else None,
        "yardstick_mm_yr": YARDSTICK["reference_trend_mm_yr"],
        "pass": (abs(ref_fit["slope"]) <= YARDSTICK["reference_trend_mm_yr"]) if ref_fit else None,
        "reference_pixel": ref,
        "stable_ground_definition": f"coherent pixels with temporal coherence >= 0.85 in the far field ({int(stable.sum())} px)",
        "sensitivity": sens,
        "relative_figures_depend_on_reference": False,
        "far_field_sensitivity": fsens,
        "far_field_sensitivity_spread_mm_yr": (round(max(vals) - min(vals), 2) if vals else None),
        "note": ("the referenced pixel's own series against the mean of the other stable ground; |trend| <= 1 mm/yr "
                 "is the practice. The candidate table is for choosing a reference: it affects ABSOLUTE rates only. "
                 "Relative figures are independent of the reference by construction, so re-referencing tests nothing "
                 "about them; the far-field table shows what they do depend on."),
    }

    log.info("QA: rate precision (empirical model)")
    # ---------------- 2. rate precision: formal, AR(1) and empirical
    model = sg.empirical_rate_model(ts_file, tc, far, cache=sg.uncertainty_cache_path(cfg))
    emp = sg.empirical_sigma_map(model, tc)
    rep = good & inside & np.isfinite(emp)
    rp = {"empirical_median_mm_yr": round(float(np.median(emp[rep])), 2) if rep.any() else None,
          "empirical_median_high_coherence_mm_yr": (round(float(np.median(emp[rep & (tc >= 0.7)])), 2)
                                                    if (rep & (tc >= 0.7)).any() else None),
          "ar1_median_mm_yr": model["all"]["ar1_median_mm_yr"],
          "residual_lag1_autocorrelation_median": model["all"]["lag1_autocorrelation_median"],
          "split_half_robust_sd_mm_yr": model["all"]["split_half_robust_sd_mm_yr"],
          "by_class": model["by_class"], "halves": model["halves"],
          "yardstick_mm_yr": YARDSTICK["rate_sigma_median_mm_yr"],
          "yardstick_high_coherence_mm_yr": YARDSTICK["rate_sigma_high_coh_mm_yr"]}
    if sig is not None:
        s_all = sig[good & np.isfinite(sig)]
        s_hi = sig[good & np.isfinite(sig) & (tc >= 0.7)]
        rp.update({"median_sigma_mm_yr": round(float(np.median(s_all)), 2),
                   "p90_sigma_mm_yr": round(float(np.percentile(s_all, 90)), 2),
                   "median_sigma_high_coherence_mm_yr": round(float(np.median(s_hi)), 2) if s_hi.size else None})
    rp["pass"] = (bool(rp["empirical_median_mm_yr"] <= YARDSTICK["rate_sigma_median_mm_yr"])
                  if rp["empirical_median_mm_yr"] is not None else None)
    rp["note"] = ("graded on the EMPIRICAL repeatability of a single cell's rate: over the far field, the rate fitted to "
                  "each half of the record is compared and the robust spread of the difference, by coherence class, "
                  "is halved (full-record error under the random-walk-like noise the residual autocorrelation "
                  "indicates). The formal 1-sigma (MintPy's residue-based velocityStd, which assumes independent "
                  "epochs) and its AR(1) inflation are shown beside it; the formal figure understates the real "
                  "error several-fold and is not a precision.")
    out["checks"]["rate_precision"] = rp

    log.info("QA: single-epoch noise")
    # ---------------- 3. single-epoch noise
    qc = stack_qc.step_rms_vs_distance(ts_file, ref_lalo=cfg.reference_lalo,
                                       mask_file=mask_f if mask_f.exists() else None,
                                       bins_km=(0.5, 2, 5, 10, 20, 50))
    far_bin = qc["bins"][-1]
    out["step_rms"] = qc
    out["checks"]["single_epoch_noise"] = {
        "robust_mm": far_bin.get("step_rms_robust_mm"), "rms_mm": far_bin.get("step_rms_mm"),
        "bin_km": [far_bin.get("lo_km"), min(far_bin.get("hi_km", 0), qc.get("max_distance_km", 0))],
        "far_over_near_ratio": qc.get("far_over_near_ratio"),
        "yardstick_robust_mm": YARDSTICK["epoch_step_robust_mm"],
        "pass": bool(far_bin.get("step_rms_robust_mm", 99) <= YARDSTICK["epoch_step_robust_mm"]),
        "note": "epoch-to-epoch displacement step RMS in the farthest distance bin from the reference; robust = 1.4826 x MAD",
    }

    # ---------------- 4. coverage
    # Graded on the share the figures use, and only counted as a pass when
    # the working threshold sits clear of the coherence noise floor: on the
    # phase-linked route 0.4 is close to what pure noise scores (open water
    # 0.37), so "99% coverage" at 0.4 was a pass that tested nothing. The
    # share that ALSO passes the unwrapping-consistency gate is reported
    # beside it (reliable_percent) whatever the config gates on.
    imaged = inside & np.isfinite(vel)
    cov_work = float((good & inside).sum() / max(1, imaged.sum()))
    cov_first = float((measured & inside).sum() / max(1, imaged.sum()))
    cov_07 = float((np.isfinite(vel) & (tc >= 0.7) & inside).sum() / max(1, imaged.sum()))
    src = str(tc_attrs.get("temporalCoherenceSource", "") or "MintPy network-inversion temporal coherence")
    thr = float(cfg.min_temporal_coherence)
    noise = _noise_level(cfg, rg, tc, um, um_src, water=_water_on_disk(cfg))
    floor = noise.get("floor")
    above_noise = (bool(thr >= floor) if floor is not None else None)
    noise["threshold_minus_floor"] = round(thr - floor, 4) if floor is not None else None
    second = {"layer": um_src, "available": um is not None,
              "applied_threshold": gate if gate_applied else 0.0,
              "requested_threshold": gate, "applied": gate_applied}
    reliable = None
    if um is not None:
        umf = np.nan_to_num(um, nan=-1.0)
        second["coverage_percent"] = {f"{g:.1f}": round(100 * float((measured & inside & (umf >= g)).sum()
                                                                   / max(1, imaged.sum())), 1)
                                      for g in SECOND_GATE_LEVELS}
        # the delivered points (first gate, AOI) each level would remove
        second["points_failing"] = {f"{g:.1f}": int((measured & inside & (umf < g)).sum())
                                    for g in SECOND_GATE_LEVELS}
        reliable = round(100 * float((measured & inside & (umf >= RELIABLE_UNWRAP_CONSISTENCY)).sum()
                                     / max(1, imaged.sum())), 1)
        # what the linked threshold admits, band by band: the 0.40-0.45 band
        # on Site B is as unwrap-inconsistent as noise (MintPy tc median
        # 0.46, none of it >= 0.7)
        bands = []
        for lo, hi in ((thr, thr + 0.05), (thr + 0.05, thr + 0.1), (thr + 0.1, 0.6), (0.6, 1.01)):
            sel = measured & inside & (tc >= lo) & (tc < hi) & np.isfinite(um)
            if sel.sum() >= 50 and hi > lo:
                bands.append({"linked_tc": f"{lo:.2f}-{min(hi, 1.0):.2f}", "cells": int(sel.sum()),
                              "unwrap_consistency_median": round(float(np.median(um[sel])), 3),
                              "share_reliable": round(float(np.mean(um[sel] >= RELIABLE_UNWRAP_CONSISTENCY)), 3)})
        if um is not tc:                                           # one gate on the classic route
            second["by_linked_band"] = bands
    out["checks"]["coverage"] = {
        "threshold": cfg.min_temporal_coherence,
        "aoi_percent_at_threshold": round(100 * cov_work, 1),
        "aoi_percent_first_gate": round(100 * cov_first, 1),
        "aoi_percent_at_0_7": round(100 * cov_07, 1),
        "aoi_imaged_cells": int(imaged.sum()),
        "coherence_source": src,
        "yardstick_percent": YARDSTICK["coverage_percent"],
        "noise_level": noise,
        "threshold_above_noise": above_noise,
        "reliable_percent": reliable,
        "second_gate": second,
        "pass": bool(100 * cov_work >= YARDSTICK["coverage_percent"] and above_noise is not False),
        "note": ("aoi_percent_at_threshold: share of imaged AOI cells the figures use (the coherence threshold, and "
                 "the unwrapping-consistency gate when the config applies one). It passes only when that share "
                 "reaches the yardstick AND the threshold sits at or above the noise floor (noise_level.floor = "
                 "mode + 3 sigma of the linked coherence on ground MintPy's own network coherence calls "
                 "decorrelated, below 0.25 - one population and one estimator for every project, "
                 "stack_qc.noise_level); below the floor the threshold admits noise and the share says nothing about "
                 "reliability. reliable_percent: the share that also passes unwrapping consistency >= 0.7 "
                 "(MintPy's default reliability threshold, Yunjun et al. 2019) - the honest 'measured reliably' "
                 "figure; second_gate shows coverage at 0.4/0.5/0.7 and, in second_gate.effect_on_figures, what "
                 "gating would do to each reported figure. Coverage figures are comparable between products only "
                 "when they grade on the same kind of coherence (coherence_source)."),
    }

    # ---------------- 5. network
    stack = cfg.mintpy_dir / "inputs" / "ifgramStack.h5"
    if stack.exists():
        with h5py.File(stack, "r") as f:
            d12 = [f"{a.decode()}_{b.decode()}" for a, b in f["date"][:]]
            keep = f["dropIfgram"][:].astype(bool)
        kept = [p for p, k in zip(d12, keep) if k]
        dts = sorted({d for p in kept for d in p.split("_")})
        from insar_pipeline.stack_backend import check_network_connectivity
        connected, orphaned, comps = check_network_connectivity(dts, kept)
        redundancy = 2 * len(kept) / max(1, len(dts))
        from collections import Counter
        hub = Counter(d for p in kept for d in p.split("_")).most_common(1)
        star = bool(kept) and len(kept) == len(dts) - 1 and hub and hub[0][1] == len(kept)
        closure = _closure(cfg)
        # dates MintPy left out of the network (no kept interferogram after
        # the network filter, time_series.py): with the reason, so "271
        # images" can be read against the 273 processed
        nc = _load_json(cfg.mintpy_dir / "network_connectivity.json") or {}
        excl = _excluded_reasons(cfg)
        ref_date = str(getattr(cfg, "stack_reference_date", "") or "")
        out["checks"]["network"] = {
            "pairs_formed": len(d12), "pairs_inverted": len(kept), "dates": len(dts),
            "mean_connections_per_date": round(redundancy, 2),
            "connected": connected, "orphaned_dates": orphaned,
            "excluded_dates": excl,
            "reference_date_excluded": bool(ref_date and ref_date in excl),
            "dates_processed_by_network": nc.get("dates"),
            "components": sorted((len(c) for c in comps), reverse=True),
            "yardstick_redundancy": YARDSTICK["redundancy"],
            "pass": bool(connected and (redundancy >= YARDSTICK["redundancy"] or star)),
            "network_type": ("single-reference (one pair per date, all sharing one date): redundancy is not a yardstick "
                             "for a phase-linked stack, whose phases are already mutually consistent" if star else "small-baseline"),
            **closure,
        }

    elif (cfg.mintpy_dir / "disp_chain.json").exists():
        # OPERA DISP-S1: the interferograms are OPERA's, not ours. dolphin
        # unwraps a nearest-3 network per ministack and inverts it with an
        # L1 norm; the per-date inversion residual is the misclosure check.
        dc = json.loads((cfg.mintpy_dir / "disp_chain.json").read_text())
        mis = dc.get("network_misclosure") or {}
        out["checks"]["network"] = {
            "network_type": "OPERA DISP-S1: nearest-3 interferograms, L1 time-series inversion (dolphin); not formed by this pipeline",
            "dates": dc.get("stitch", {}).get("epochs"),
            "reference_epoch_boundaries": dc.get("stitch", {}).get("reference_epoch_boundaries"),
            "closure_median_fraction_open": mis.get("median_share_of_epochs"),
            "closure_mean_fraction_open": mis.get("mean_share_of_epochs"),
            "closure_by_coherence": mis.get("by_coherence"),
            "closure_note": mis.get("note", "no misclosure layer read"),
            "pass": None,
        }
    loops = _open_loops(cfg, rg, measured & inside, (out["checks"].get("network") or {}).get("closure_triplets"))
    if loops:
        out["checks"]["coverage"]["second_gate"]["open_loops"] = loops
    dc_f = cfg.mintpy_dir / "disp_chain.json"
    if dc_f.exists():
        # the stitching of OPERA's granules is the step most able to bias a
        # DISP-S1 rate without any other check seeing it (disp-4)
        out["checks"]["stitch"] = stitch_check(_load_json(dc_f) or {}, cfg)

    # ---------------- 6. distributed-scatterer bias
    by_class = []
    for k, lab in enumerate(labels):
        sel = far & (kcls == k)
        if sel.sum() > 200:
            by_class.append({"tc": lab, "n": int(sel.sum()), "median_mm_yr": round(float(np.median(vel[sel])), 2),
                             "mean_mm_yr": round(float(vel[sel].mean()), 2)})
    # the lowest and the highest class the far field holds (by_class runs low
    # to high): at a working threshold of 0.6 or more there is no 0.4-0.6
    # class, and requiring one left the check untested on every such run.
    # One class alone has nothing to compare with: not applicable, not a pass
    # (pass None, and send_rule notes it rather than holding). The verdict
    # does not count it at all (check_counts, D2 7 Oct 2026): no difference
    # between classes exists to bias the class-matched figures.
    two = len(by_class) >= 2
    lowc = by_class[0]["median_mm_yr"] if two else None
    highc = by_class[-1]["median_mm_yr"] if two else None
    bias = (lowc - highc) if (lowc is not None and highc is not None) else None
    # MintPy's closure-phase susceptibility mask (closure_phase_bias.py -a mask):
    # 1 where the average closure phase is consistent with zero. A DIAGNOSTIC
    # of where the bias lives; the ESTIMATE needs long-connection pairs.
    cp_f = dl._product(cfg, "maskClosurePhase.h5")
    cp_free = None
    if cp_f.exists():
        cp_m, _ = _read2d(cp_f, "mask")
        if cp_m.shape == vel.shape:
            cp_free = cp_m.astype(bool)
    closure_mask = None
    if cp_free is not None:
        closure_mask = {
            "file": str(cp_f),
            "bias_free_percent_of_reported_aoi": round(100 * float((cp_free & good & inside).sum() / max(1, (good & inside).sum())), 1),
            "bias_free_percent_of_far_field": round(100 * float((cp_free & far).sum() / max(1, far.sum())), 1),
            "bias_free_percent_of_workings": round(100 * float((cp_free & mine_good).sum() / max(1, mine_good.sum())), 1),
            "far_field_median_rate_bias_free_mm_yr": round(float(np.median(vel[far & cp_free])), 2) if (far & cp_free).sum() > 200 else None,
            "far_field_median_rate_susceptible_mm_yr": round(float(np.median(vel[far & ~cp_free])), 2) if (far & ~cp_free).sum() > 200 else None,
            "note": ("closure_phase_bias.py -a mask at the stack's own connection level (3); 1 = average "
                     "closure phase consistent with zero. The bias time-series estimate (Zheng et al. 2022) "
                     "needs interferograms at a connection level well above the bandwidth (nl > bw) that "
                     "this 3-connection stack does not contain; over this farmland such 60-120 day pairs "
                     "decorrelate, so the estimate is not available here."),
        }
    out["checks"]["distributed_scatterer_bias"] = {
        "velocity_by_coherence_class": by_class,
        "closure_phase_mask": closure_mask,
        "low_minus_high_coherence_mm_yr": round(bias, 2) if bias is not None else None,
        "low_class": by_class[0]["tc"] if two else None,
        "high_class": by_class[-1]["tc"] if two else None,
        "yardstick_mm_yr": YARDSTICK["ds_bias_mm_yr"],
        "pass": (abs(bias) <= YARDSTICK["ds_bias_mm_yr"]) if bias is not None else None,
        **({} if len(by_class) != 1 else {"not_applicable": (
            f"the far field holds one coherence class ({by_class[0]['tc']}) with more than 200 cells: there is "
            "no other class to compare it with, and the class-matched figures compare like with like")}),
        "note": ("median rate far from any deformation, by temporal-coherence class; a monotonic drift toward "
                 "negative rates at low coherence is the short-baseline phase bias of distributed scatterers "
                 "(Ansari et al. 2021; Zheng et al. 2022), with a contribution from unwrapping errors, which "
                 "are also more frequent on low-coherence ground and read systematically negative here. Absolute "
                 "rates on low-coherence ground are biased low; site figures are matched by coherence class "
                 "against the far field."),
    }

    log.info("QA: site findings")
    # ---------------- site findings: class-matched, against the farmland null
    fd = out["findings"]
    if mine_series is not None:
        cm_series = class_matched_series(mine_series, mine_good)
        f_cm = sg.fit_rate(t, cm_series)
        f_pl = sg.fit_rate(t, mine_series - far_series)
        # the total is the measured change over the record (record_change,
        # which may bend), as panel_history's totals; rate x span kept beside it
        avg = sg.trend_total(f_cm)
        ch = sg.record_change(t, cm_series)
        tot = float(ch["change_mm"]) if ch and ch.get("change_mm") is not None else avg
        fd["workings_relative_to_far_field"] = {
            "rate_mm_yr": round(f_cm["slope"], 2), "rate_in_yr": round(f_cm["slope"] / MM_PER_IN, 3),
            "rate_plain_mm_yr": round(f_pl["slope"], 2),
            "stderr_white_mm_yr": round(f_cm["stderr_white"], 2), "stderr_ar1_mm_yr": round(f_cm["stderr_ar1"], 2),
            "total_mm": round(tot, 1), "total_in": round(tot / MM_PER_IN, 2),
            "total_at_average_rate_mm": round(avg, 1),
            "total_method": (ch or {}).get("method") or "rate_x_span",
            "n_epochs": f_cm["n"], "n_pixels": int(mine_good.sum()),
            "significance": sg.assess(r_place, far_wide, mine_good),
            "note": ("mean series over every mapped panel minus the far-field series matched to the panels' coherence "
                     "mix; positive = toward the satellite. The uncertainty to quote is significance."
                     "farmland_spread_1sigma_mm_yr (how much ordinary farmland of the same footprint differs), not "
                     "the temporal stderr"),
            "per_pixel": _relative_rates(vel, tc, mine, far, good, cp_free),
        }
    if roi is not None and roi.any():
        ev_s, ev_e = getattr(cfg, "acceptance_event_start", ""), getattr(cfg, "acceptance_event_end", "")
        span = int(getattr(cfg, "acceptance_window_days", 30))
        from datetime import timedelta

        def shift(d, k):
            return (datetime.strptime(d, "%Y%m%d") + timedelta(days=k)).strftime("%Y%m%d")
        pre = [i for i, d in enumerate(dates) if ev_s and shift(ev_s, -span) <= d <= ev_s]
        pst = [i for i, d in enumerate(dates) if ev_e and ev_e <= d <= shift(ev_e, span)]
        rel = class_matched_series(roi_series, roi)
        step = float(np.nanmean(rel[pst]) - np.nanmean(rel[pre])) if pre and pst else None
        f_roi = sg.fit_rate(t, rel)
        fd["named_panel"] = {
            "polygon": panel_geojson, "n_pixels": int(roi.sum()),
            "event_window": [ev_s, ev_e], "window_days": span,
            "pre_epochs": [dates[i] for i in pre], "post_epochs": [dates[i] for i in pst],
            "step_across_window_mm": round(step, 2) if step is not None else None,
            "step_across_window_in": round(step / MM_PER_IN, 3) if step is not None else None,
            "rate_relative_to_far_field_mm_yr": round(f_roi["slope"], 2) if f_roi else None,
            "significance": sg.assess(r_place, far_wide, roi),
            "note": "panel mean minus the coherence-matched far-field mean; negative = subsidence",
            "per_pixel": _relative_rates(vel, tc, roi, far, good, cp_free),
        }
    out["series_mm"] = {"dates": dates,
                        "panel": ([round(float(v), 2) for v in roi_series] if roi is not None else None),
                        "far_field": [round(float(v), 2) for v in far_series],
                        "workings": ([round(float(v), 2) for v in mine_series] if mine_series is not None else None)}
    out["far_field"]["endpoint_total_mm"] = round(float(far_series[-1] - far_series[0]), 2)
    f_far = sg.fit_rate(t, far_series)
    out["far_field"]["series_rate_mm_yr"] = round(f_far["slope"], 2) if f_far else None
    out["far_field"]["note"] = ("series are MintPy's, referenced to REF_DATE; a total over the record is "
                                "last minus first date, never the last value alone")

    # ---------------- the feature of interest
    # Pinned by config if graph_center_lalo is set; otherwise FOUND IN THE
    # DATA as the deepest point of the class-matched relative rate field
    # averaged over 1 km inside the AOI. Found that way it is a SELECTED
    # EXTREME: over a few hundred independent 1 km patches of farmland the
    # deepest is expected to be several sigma out by chance alone, so its
    # value is set against that expectation and against the next-deepest
    # independent feature, rather than against zero.
    from scipy.ndimage import uniform_filter
    from scipy.stats import norm
    py, px = sg.pixel_metres(attrs)
    k1 = max(3, int(round(1000.0 / max(np.sqrt(py * px), 1e-6))))
    sel_in = good & inside & np.isfinite(r_cm)
    num = uniform_filter(np.where(sel_in, r_cm, 0.0), k1)
    den = uniform_filter(sel_in.astype(float), k1)
    smooth = np.where(den > 0.3, num / np.maximum(den, 1e-9), np.nan)
    selw = far_wide & np.isfinite(r_cm)
    numw = uniform_filter(np.where(selw, r_cm, 0.0), k1)
    denw = uniform_filter(selw.astype(float), k1)
    null_1km = (numw / np.maximum(denw, 1e-9))[denw > 0.5]
    point = (getattr(cfg, "graph_center_lalo", "") or "").strip()
    how = "pinned by config (graph_center_lalo)"
    r0_ = c0_ = None
    if not point and np.isfinite(smooth).any():
        r0_, c0_ = np.unravel_index(np.nanargmin(smooth), smooth.shape)
        la, lo = sg.rc_to_lalo(attrs, r0_, c0_)
        point = f"{float(la):.5f},{float(lo):.5f}"
        how = ("deepest point of the class-matched relative rate field averaged over 1 km inside the AOI "
               "(a selected extreme - see 'extreme')")
    if point:
        la, lo = (float(v) for v in point.split(","))
        r, c = (r0_, c0_) if r0_ is not None else sg.lalo_to_rc(attrs, la, lo)
        if 0 <= r < ny and 0 <= c < nx:
            win = np.zeros_like(good); win[max(0, r - 6):r + 7, max(0, c - 6):c + 7] = True
            win &= good
            v_abs = vel[win]; v_rel = r_cm[win & np.isfinite(r_cm)]
            n_indep = max(1, int(round(sel_in.sum() / (k1 * k1))))
            sd1 = float(null_1km.std()) if null_1km.size else None
            expected = float(sd1 * norm.ppf(1.0 / (n_indep + 1))) if sd1 else None
            second = None
            if np.isfinite(smooth[r, c]):
                yy, xx = np.ogrid[:ny, :nx]
                away = ((yy - r) * py) ** 2 + ((xx - c) * px) ** 2 > 3000.0 ** 2
                if np.isfinite(smooth[away]).any():
                    second = float(np.nanmin(np.where(away, smooth, np.nan)))
            val1 = float(smooth[r, c]) if np.isfinite(smooth[r, c]) else None
            p_sel = None
            if val1 is not None and null_1km.size:
                F = float(np.mean(null_1km <= val1))
                p_sel = round(float(1.0 - (1.0 - F) ** n_indep), 3)
            fd["monitoring_point"] = {
                "lat": round(la, 5), "lon": round(lo, 5), "how_chosen": how, "n_pixels_13x13": int(win.sum()),
                "rate_mm_yr": round(float(np.median(v_abs)), 2) if v_abs.size else None,
                "rate_relative_to_far_field_mm_yr": round(float(np.median(v_rel)), 2) if v_rel.size else None,
                "rate_relative_1km_mm_yr": round(val1, 2) if val1 is not None else None,
                "extreme": {
                    "independent_1km_patches_in_aoi": n_indep,
                    "farmland_1km_spread_1sigma_mm_yr": round(sd1, 2) if sd1 else None,
                    "expected_deepest_by_chance_mm_yr": round(expected, 1) if expected is not None else None,
                    "next_deepest_feature_3km_away_mm_yr": round(second, 1) if second is not None else None,
                    "p_selection_corrected": p_sel,
                    "stands_out": (bool(p_sel < 0.05) if p_sel is not None else None),
                    "note": ("found by searching the whole AOI for its minimum, so it is compared with the minimum "
                             "that ordinary farmland produces over as many independent 1 km patches: "
                             "p_selection_corrected = 1 - (1 - F)^N, F the share of farmland 1 km patches at least "
                             "this deep, N the independent patches searched. stands_out = p < 0.05. The Gaussian "
                             "expectation and the next-deepest feature are given for scale."),
                },
                "note": ("13x13-cell (~200 m) medians: absolute, and class-matched relative to the far field; the "
                         "1 km figure is the one the search used"),
                "per_pixel": _relative_rates(vel, tc, win, far, good, cp_free),
            }
    # ---------------- the same search at 1, 2 and 4 km
    # A feature can be invisible at 1 km (1 km patches are noisy) and stand
    # out at a larger scale. At each scale the deepest window in the AOI is
    # compared with windows built the same way on ordinary ground (outside
    # every panel + buffer and > 5 km from the candidate).
    # The deepest of N windows is always deeper than the other N-1, so no
    # count of how many windows it beats can show it is unusual (an earlier
    # version counted overlapping sliding windows as independent and called
    # everything significant). What does show an outlier is the GAP to the
    # deepest other window: under noise the two deepest are close together
    # (1 km here: -16.7 against -16.3). Until 7 Oct the rule was gap > 2
    # spreads at any one scale, never calibrated (on simulated and rolled
    # farmland it fired in 1-24% of reports); now the gap must beat the 95th percentile of
    # the largest gap across the three scales on simulated farmland copies
    # (decision E7, multi_scale_standout). A Gaussian-tail p (selection- and
    # scale-corrected) is given as an indication only.
    if "monitoring_point" in fd:
        log.info("QA: multi-scale stand-out, calibrated on %d farmland copies", STANDOUT_COPIES)
        mine_buf = sg.euclidean_buffer(mine, attrs, buf_m)
        ms, cal = multi_scale_standout(r_cm, good & inside & np.isfinite(r_cm), inside, mine_buf,
                                       far & np.isfinite(r_cm), attrs, mine)
        outl = sorted((m for m in ms if m["stands_out"]), key=lambda x: -x["gap_in_spreads"])
        fd["monitoring_point"]["multi_scale"] = ms
        fd["monitoring_point"]["standout_calibration"] = cal
        # every scale that stands out, with its own location (a 2 km stand-out
        # can sit on another feature than the 1 km one); `standout` is the one
        # with the largest gap, as before
        fd["monitoring_point"]["standouts"] = outl
        fd["monitoring_point"]["standout"] = outl[0] if outl else None

    # ---------------- ground faster than the threshold, relative and class-matched
    thr = float(getattr(cfg, "deforming_threshold", 0.5))
    thr_mm = thr * (MM_PER_IN if str(getattr(cfg, "display_units", "in")).lower().startswith("in") else 1.0)
    moving = good & inside & (r_cm < -thr_mm)
    far_share = float((far & (r_cm < -thr_mm)).sum() / max(1, far.sum()))
    px_m2 = py * px
    unit = "in/yr" if thr_mm != thr else "mm/yr"
    fd["deforming_area"] = {
        "threshold": f"faster than -{thr} {unit} relative to the far field, matched by coherence class",
        "n_pixels": int(moving.sum()), "area_km2": round(moving.sum() * px_m2 / 1e6, 2),
        "inside_mapped_panels_percent": round(100 * float((moving & mine).sum() / max(1, moving.sum())), 1),
        "median_rate_mm_yr": round(float(np.median(r_cm[moving])), 2) if moving.any() else None,
        "fastest_mm_yr": round(float(np.min(r_cm[moving])), 2) if moving.any() else None,
        "far_field_share_beyond_threshold_percent": round(100 * far_share, 1),
        "note": ("single cells are noisy (empirical rate error several mm/yr on low-coherence ground), so some "
                 "farmland crosses any threshold by chance: far_field_share_beyond_threshold_percent is that "
                 "chance rate, measured where nothing should be moving"),
    }

    # ---------------- what the figures depend on (measured, not gated)
    # release the feature search's full-grid temporaries first: at 3.5 M
    # cells each is 28 MB and a dozen of them were still held here (peak
    # 1.3 GB on Site B before this; the machine runs many jobs at once)
    num = den = smooth = numw = denw = selw = sel_in = null_1km = None       # noqa: F841
    yy = xx = mine_buf = emp = rep = sig = None                                # noqa: F841
    log.info("QA: dates, far field, sensitivities, geolocation")
    out["dates"] = date_accounting(cfg, dates)
    regions = _sensitivity_regions(cfg, attrs, vel.shape, mine, roi, panel_geojson)
    sigmas, assessed = {}, {}
    for name, (reg, kind) in regions.items():
        if kind == "full":
            continue
        if name == "mapped mine polygons" and "workings_relative_to_far_field" in fd:
            a = fd["workings_relative_to_far_field"].get("significance") or {}
        elif name == "named panel" and "named_panel" in fd:
            a = fd["named_panel"].get("significance") or {}
        else:
            # assess never returns {} (too few cells -> testable False with a reason)
            a = sg.assess(r_place, far_wide, reg & good)
        assessed[name] = a
        sigmas[name] = (a or {}).get("farmland_spread_1sigma_mm_yr")
    r_place = None
    out["far_field"].update(_far_field_sufficiency(far, far_wide, good, regions))
    if ground.get("reason"):
        # the report areas cover nearly all the farmland (report_farmland_masks):
        # nothing is left to compare them with, whatever the uncut far field holds
        out["far_field"].update(thin=True, thin_reason=ground["reason"])
    out["area_testability"] = _area_testability(regions, assessed, good, kcls, borrowed, labels)
    ctx = {"cfg": cfg, "vel": vel, "tc": tc, "good": good, "measured": measured, "inside": inside, "far": far,
           "mine": mine, "attrs": attrs, "r_cm": r_cm, "regions": regions, "sigmas": sigmas, "rg": rg,
           "um": um, "um_src": um_src, "dates": dates, "assessed": assessed, "ref": ref}
    for key, fn in (("second_gate_effect", _second_gate_effect), ("disturbed_effect", _disturbed_effect),
                    ("ramp_sensitivity", _ramp_sensitivity),
                    ("matching_sensitivity", _matching_sensitivity), ("plausibility", _plausibility),
                    ("dem_error", _dem_error_diagnostic), ("resolution_check", _resolution_check),
                    ("reference_distance", _reference_distance)):
        try:
            res = fn(ctx)
        except Exception as exc:                                  # noqa: BLE001
            # a diagnostic that cannot run is recorded as not run, never as a pass
            log.warning("QA: %s could not run: %s", key, exc, exc_info=True)
            res = {"available": False, "reason": f"could not run: {exc!r}"[:300], "flag": None}
        if key == "second_gate_effect":
            out["checks"]["coverage"]["second_gate"]["effect_on_figures"] = res
        elif key == "disturbed_effect":
            out["checks"]["coverage"]["disturbed"] = res
        else:
            out[key] = res
    out["phase_linking"] = _phase_linking_record(cfg)
    uc = _unwrap_correction_record(cfg)
    if uc is not None:
        out["unwrap_correction"] = uc
    for key, fn in (("injection_recovery", _injection_record), ("independent_check", _independent_check)):
        res = fn(cfg)
        if res is not None:
            out[key] = res
    try:
        out["geolocation"] = _geolocation(cfg, attrs, tc, imaged, fetch_water)
    except Exception as exc:                                      # noqa: BLE001
        log.warning("QA: geolocation check could not run: %s", exc, exc_info=True)
        out["geolocation"] = {"available": False, "reason": f"could not run: {exc!r}"[:300], "flag": None}
    nl = out["checks"]["coverage"]["noise_level"]
    if nl.get("measured") and "share_on_mapped_water" not in nl and _water_on_disk(cfg) is not None:
        # the water map the geolocation check has just fetched: how much of the
        # noise population is open water (a caption says "open water" only then)
        try:
            again = _noise_level(cfg, rg, tc, um, um_src, water=_water_on_disk(cfg))
            for k in ("share_on_mapped_water", "water_map"):
                if k in again:
                    nl[k] = again[k]
        except Exception as exc:                                  # noqa: BLE001 - a record, never a failure
            log.warning("QA: noise population's share on water not measured: %s", exc)

    out = annotate_site(cfg, out)
    # counted out of the checks that apply (decision D2, 7 Oct 2026): one
    # that cannot apply is named, never counted as a pass or as untested
    cnt = check_counts(out)
    out.update({k: cnt[k] for k in ("passed", "flagged", "untested", "applicable", "not_applicable")})
    out["checks_text"] = cnt["text"]
    out["verdict"] = verdict(cnt, thin=bool(out["far_field"].get("thin")))
    out["send_rule"] = send_rule(out)
    return out


def _relative_rates(vel, tc, region, far, good, cp_free=None) -> dict:
    """
    Region-minus-far-field rate three ways, from per-pixel velocities:
      plain          median(region) - median(far)
      class_matched  the same difference within each temporal-coherence class,
                     weighted by the region's class counts - cancels the
                     distributed-scatterer bias even when the region and the
                     far field have different coherence mixes
      bias_free      the same difference on closure-phase-clean pixels only
    """
    classes = [(0.4, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.01)]
    r = region & good; f = far & good
    if r.sum() < 20:
        return {}
    out = {"plain_mm_yr": round(float(np.median(vel[r]) - np.median(vel[f])), 2)}
    num = den = 0.0; mix = []
    for lo, hi in classes:
        rc = r & (tc >= lo) & (tc < hi); fc = f & (tc >= lo) & (tc < hi)
        if rc.sum() >= 20 and fc.sum() >= 200:
            d = float(np.median(vel[rc]) - np.median(vel[fc]))
            num += d * rc.sum(); den += rc.sum()
            mix.append({"tc": f"{lo:.1f}-{min(hi, 1.0):.1f}", "n_region": int(rc.sum()), "diff_mm_yr": round(d, 2)})
    if den:
        out["class_matched_mm_yr"] = round(num / den, 2)
        out["class_matched_share_of_region"] = round(float(den / r.sum()), 3)
        out["by_class"] = mix
    if cp_free is not None and (r & cp_free).sum() >= 20 and (f & cp_free).sum() >= 200:
        out["bias_free_mm_yr"] = round(float(np.median(vel[r & cp_free]) - np.median(vel[f & cp_free])), 2)
        out["bias_free_n_region"] = int((r & cp_free).sum())
    return out


def _scale_windows(field, sel, inside, excl, pm, scales=None, dens=None) -> list:
    """
    The multi-scale search on one field: at each scale (km) the deepest
    window mean over `sel` cells inside the AOI, the deepest window on
    ordinary ground (centre not in `excl`, more than STANDOUT_AWAY_M from
    it) and the gap between them in robust spreads of those windows.
    [record or None per scale]; rr/cc are on `field`'s grid. Windows are
    zero-padded at the edge of the array (mode 'constant'), so a crop with
    half a window of margin gives the same numbers as the whole grid.
    `dens` are the scales' window counts of `sel`, when already formed.
    """
    from scipy.ndimage import uniform_filter
    py, px = pm
    scales = STANDOUT_SCALES_KM if scales is None else scales
    ny, nx = np.shape(field)
    yy, xx = np.ogrid[:ny, :nx]
    out = []
    for i, km in enumerate(scales):
        kk = max(3, int(round(km * 1000.0 / np.sqrt(py * px))))
        den = dens[i] if dens is not None else uniform_filter(sel.astype(float), kk, mode="constant")
        num = uniform_filter(np.where(sel, field, 0.0), kk, mode="constant")
        smk = np.where((den > 0.5) & inside, num / np.maximum(den, 1e-9), np.nan)
        num = None
        if not np.isfinite(smk).any():
            out.append(None)
            continue
        rr, cc = np.unravel_index(np.nanargmin(smk), smk.shape)
        v = float(smk[rr, cc])
        away = ((yy - rr) * py) ** 2 + ((xx - cc) * px) ** 2 > STANDOUT_AWAY_M ** 2
        null = smk[np.isfinite(smk) & ~excl & away]
        if null.size < 50:
            out.append(None)
            continue
        med = float(np.median(null))
        sd = float(1.4826 * np.median(np.abs(null - med)))
        nxt = float(null.min())
        out.append({"scale_km": km, "rr": int(rr), "cc": int(cc), "value": v, "spread": sd, "next": nxt,
                    "median": med, "gap": (nxt - v) / sd if sd > 0 else 0.0,
                    "n_ind": max(1, int(round(float(sel.sum()) / (kk * kk))))})
    return out


def standout_null(field, sel, inside, excl, farmland, pm, real=None, n_copies=None, seed=STANDOUT_SEED,
                  scales=None) -> dict:
    """
    The multi-scale gap rule calibrated for having tried every scale
    (decision E7, 7 Oct 2026): what ordinary farmland alone produces.

      1. the farmland: `farmland` cells of `field` (coherent AOI ground
         outside every mapped mine and its buffer);
      2. its spatial autocorrelation, measured on normal scores of the gappy
         field (masked FFT) and kept as measured out to STANDOUT_LAG_M
         (twice the largest window), then tapered to zero over
         STANDOUT_TAPER_M - the copies reproduce the measured covariance
         beyond the largest window (as E8 asks of the feature null);
      3. n_copies Gaussian fields with that autocorrelation on the searched
         cells, each rank-mapped onto the farmland's own values (amplitude-
         adjusted Fourier surrogates, Theiler et al. 1992: the copies carry
         farmland's skew and tails cell by cell);
      4. on each copy the same search (_scale_windows: same cells, same
         mines, same 5 km rule) and its LARGEST gap over the scales.

    gap_threshold = the STANDOUT_LEVEL percentile of those largest gaps: a
    stand-out at any scale is then a 5% false alarm over all three together.
    p_calibrated of a gap = (1 + copies whose largest gap is at least as
    large) / (1 + n_copies). Deterministic (fixed seed).

    copies_match records, per scale, the robust spread of the farmland's
    window means against the copies' (median): the check that the copies
    reproduce farmland at the scale the rule reads (a ratio outside
    STANDOUT_MATCH is said here, not gated on). Measured 7 Oct 2026 (the
    QA's own records): Site B ratios 0.91/0.97/1.05 at 1/2/4 km;
    Site A 0.88/0.85/0.72. The final verification of 7 Oct 2026 traced
    Site A's 4 km ratio to the candidate itself, not to heavy farmland
    tails (its 4 km windows have sd / robust spread 1.02): the copies are
    built from farmland that still holds the sinking patch (Feature 1).
    With it and 500 m around left out the ratios were 1.05/1.12/1.13 and the
    stand-out held (p 0.035/0.025/0.02); against 200 rolled and flipped
    real fields it read about p 0.15. Leaving the candidate out here is not
    done yet. Until it is, a ratio outside the band means the report says
    no more than 'worth a look' of a feature holding this stand-out
    (report_insights.qa_standout_doubt, decision H1).
    Fails closed (`tested` False with a reason): no stand-out is then
    calibrated, and none is reported.
    """
    from scipy import fft as sfft
    from scipy.ndimage import uniform_filter
    from scipy.stats import norm
    py, px = pm
    scales = STANDOUT_SCALES_KM if scales is None else scales
    n_copies = STANDOUT_COPIES if n_copies is None else int(n_copies)
    out = {"tested": False, "n_copies": n_copies, "seed": int(seed), "level_percent": STANDOUT_LEVEL,
           "scales_km": list(scales), "max_lag_m": STANDOUT_LAG_M, "taper_m": STANDOUT_TAPER_M,
           "method": ("amplitude-adjusted Fourier surrogates of the AOI farmland (measured normal-score "
                      f"autocorrelation kept to {STANDOUT_LAG_M / 1000:.0f} km, tapered over "
                      f"{STANDOUT_TAPER_M / 1000:.0f} km; values the farmland's own), searched exactly like the "
                      "record; threshold = the 95th percentile of each copy's largest gap over the scales")}
    nfarm = int(np.sum(farmland))
    out["farmland_cells"] = nfarm
    if nfarm < FAR_MIN_CELLS or not np.any(sel):
        out["reason"] = (f"only {nfarm} farmland cells to simulate from ({FAR_MIN_CELLS} needed): the stand-out "
                         "rule could not be calibrated, so no stand-out is reported")
        return out
    ny, nx = np.shape(field)
    L = int(round((STANDOUT_LAG_M + STANDOUT_TAPER_M) / np.sqrt(py * px)))
    sh = (sfft.next_fast_len(ny + L), sfft.next_fast_len(nx + L))
    # 2. masked autocorrelation on normal scores
    v = np.asarray(field)[farmland]
    order = np.argsort(v, kind="stable")
    rank = np.empty(v.size)
    rank[order] = np.arange(v.size)
    z = np.zeros((ny, nx))
    z[farmland] = norm.ppf((rank + 0.5) / v.size)
    order = rank = None
    P = _copy_spectrum(z, farmland, sh, pm)
    z = None
    out["negative_power_share"] = round(float(-np.minimum(P, 0).sum() / max(np.abs(P).sum(), 1e-12)), 4)
    amp = np.sqrt(np.maximum(P, 0.0)).astype(np.float32)
    P = None
    # 3. the values the copies take: the farmland's own, one per searched cell
    q = np.sort(v)
    nsel = int(np.sum(sel))
    qv = q[np.minimum(((np.arange(nsel) + 0.5) / nsel * q.size).astype(int), q.size - 1)]
    q = v = None
    dens = [uniform_filter(np.asarray(sel, dtype=float), max(3, int(round(km * 1000.0 / np.sqrt(py * px)))),
                           mode="constant") for km in scales]
    rng = np.random.default_rng(seed)
    largest, spreads = [], {km: [] for km in scales}
    cp = np.zeros((ny, nx))
    for _ in range(n_copies):
        g = sfft.irfft2(amp * sfft.rfft2(rng.standard_normal(sh, dtype=np.float32)), sh)[:ny, :nx]
        o = np.argsort(g[sel], kind="stable")
        g = None
        vals = np.empty(nsel)
        vals[o] = qv
        cp[sel] = vals
        o = vals = None
        res = [m for m in _scale_windows(cp, sel, inside, excl, pm, scales, dens) if m]
        largest.append(max((m["gap"] for m in res), default=0.0))
        for m in res:
            spreads[m["scale_km"]].append(m["spread"])
    amp = cp = dens = None
    largest = np.sort(np.asarray(largest))
    thr = float(np.percentile(largest, STANDOUT_LEVEL))
    match, ok = {}, True
    for m in (real or []):
        if m and spreads.get(m["scale_km"]):
            cs = float(np.median(spreads[m["scale_km"]]))
            r = m["spread"] / cs if cs > 0 else None
            match[f"{m['scale_km']:g}"] = {"farmland_spread_mm_yr": round(m["spread"], 2),
                                           "copies_spread_median_mm_yr": round(cs, 2),
                                           "ratio": round(r, 2) if r is not None else None}
            ok &= r is not None and STANDOUT_MATCH[0] <= r <= STANDOUT_MATCH[1]
    out.update({"tested": True, "gap_threshold": round(thr, 2),
                "copies_largest_gap": [round(float(x), 2) for x in largest],
                "copies_largest_gap_median": round(float(np.median(largest)), 2),
                "copies_match": match, "copies_match_within_band": bool(ok) if match else None,
                "note": ("a scale stands out when its gap is larger than gap_threshold; copies_match compares the "
                         "robust spread of window means on the farmland with the copies' (a ratio below 1: the "
                         "farmland's windows vary less in their core than the copies'; it does not test their "
                         "tails). The copies are built from farmland that still holds the stand-out's own ground, "
                         "which can by itself push a ratio out of the band. "
                         f"Band {STANDOUT_MATCH[0]}-{STANDOUT_MATCH[1]}: said here, and a ratio outside it keeps "
                         "the client report's label of a feature holding the stand-out at 'worth a look'.")})
    return out


def _copy_spectrum(z, farmland, sh, pm, how=None):
    """power spectrum (rfft2 layout on grid `sh`) of the copies: the
    farmland's normal-score autocorrelation, measured on the gappy field
    (masked FFT: lag by lag over the cell pairs behind it), averaged in
    rings of one cell (farmland noise is close to isotropic, and the ring
    average takes out the lag-by-lag sampling noise that otherwise turns
    into spurious power), kept to STANDOUT_LAG_M and tapered to zero over
    STANDOUT_TAPER_M"""
    from scipy import fft as sfft
    how = how or _SPECTRUM_HOW
    py, px = pm
    Fx = sfft.rfft2(z, sh)
    if how == "periodogram":
        return (Fx * np.conj(Fx)).real / max(float(np.sum(farmland)), 1.0)
    Cx = sfft.irfft2(Fx * np.conj(Fx), sh)
    Fx = None
    Fm = sfft.rfft2(np.asarray(farmland, dtype=np.float64), sh)
    Cm = sfft.irfft2(Fm * np.conj(Fm), sh)
    Fm = None
    ok = Cm > 1000.0
    C = np.where(ok, Cx / np.maximum(Cm, 1.0), 0.0)
    Cx = Cm = None
    dy = np.minimum(np.arange(sh[0]), sh[0] - np.arange(sh[0]))[:, None] * py
    dx = np.minimum(np.arange(sh[1]), sh[1] - np.arange(sh[1]))[None, :] * px
    R = np.hypot(dy, dx)
    if how == "isotropic":
        step = float(np.sqrt(py * px))
        ring = np.rint(R / step).astype(np.int64)
        lim = int(np.ceil((STANDOUT_LAG_M + STANDOUT_TAPER_M) / step)) + 1
        use = ok & (ring <= lim)
        n = np.bincount(ring[use], minlength=lim + 1)
        prof = np.bincount(ring[use], weights=C[use], minlength=lim + 1) / np.maximum(n, 1)
        prof[n == 0] = 0.0
        C = np.where(ring <= lim, prof[np.minimum(ring, lim)], 0.0)
        ring = use = None
    C *= np.where(R <= STANDOUT_LAG_M, 1.0,
                  np.where(R <= STANDOUT_LAG_M + STANDOUT_TAPER_M,
                           0.5 * (1 + np.cos(np.pi * (R - STANDOUT_LAG_M) / STANDOUT_TAPER_M)), 0.0))
    R = None
    return sfft.rfft2(C).real


_SPECTRUM_HOW = "isotropic"


def multi_scale_standout(r_cm, sel_all, inside, mine_buf, farmland, attrs, mine, n_copies=None) -> tuple:
    """
    (multi_scale records, calibration): the deepest window at 1, 2 and 4 km
    and whether it stands out from ordinary ground beyond what farmland
    alone produces when all three scales are tried (standout_null). Run on
    the AOI's bounding box plus half the largest window, so the copies are
    simulated on that box only. stands_out is None (not calibrated) when the
    calibration could not run - never True on an untested rule.
    """
    from scipy.stats import norm
    pm = _pm(attrs)
    rows, cols = np.flatnonzero(np.any(inside, 1)), np.flatnonzero(np.any(inside, 0))
    if not rows.size:
        return [], {"tested": False, "reason": "no AOI cell in the product grid"}
    pad = int(np.ceil(max(STANDOUT_SCALES_KM) * 1000.0 / np.sqrt(pm[0] * pm[1]) / 2)) + 2
    r0, r1 = max(0, rows[0] - pad), min(inside.shape[0], rows[-1] + 1 + pad)
    c0, c1 = max(0, cols[0] - pad), min(inside.shape[1], cols[-1] + 1 + pad)
    S = (slice(r0, r1), slice(c0, c1))
    sel, ins, excl, fld = sel_all[S], inside[S], mine_buf[S], np.nan_to_num(r_cm[S], nan=0.0)
    real = _scale_windows(fld, sel, ins, excl, pm)
    if not any(real):
        return [], {"tested": False, "reason": "no scale had enough ordinary ground to compare with"}
    try:
        cal = standout_null(fld, sel, ins, excl, farmland[S], pm, real=real, n_copies=n_copies)
    except Exception as exc:                                      # noqa: BLE001 - fails closed, said
        log.warning("QA: the stand-out calibration could not run: %s", exc, exc_info=True)
        cal = {"tested": False, "reason": f"could not run: {exc!r}"[:300]}
    big = np.asarray(cal.get("copies_largest_gap") or [])
    ms = []
    from insar_pipeline.significance import rc_to_lalo
    for m in real:
        if not m:
            continue
        rr, cc = m["rr"] + r0, m["cc"] + c0
        sd = m["spread"]
        z = (m["value"] - m["median"]) / sd if sd > 0 else 0.0
        p_g = 1 - (1 - norm.cdf(z)) ** m["n_ind"]
        la_, lo_ = rc_to_lalo(attrs, rr, cc)
        rec = {"scale_km": m["scale_km"], "lat": round(float(la_), 4), "lon": round(float(lo_), 4),
               "rate_relative_mm_yr": round(m["value"], 2), "farmland_spread_1sigma_mm_yr": round(sd, 2),
               "deepest_other_farmland_mm_yr": round(m["next"], 2), "gap_in_spreads": round(m["gap"], 2),
               "independent_windows": m["n_ind"],
               "p_gaussian_indicative": round(float(min(1.0, len(STANDOUT_SCALES_KM) * p_g)), 4)}
        if cal.get("tested"):
            rec.update({"gap_threshold": cal["gap_threshold"],
                        "p_calibrated": round(float((1 + np.sum(big >= m["gap"])) / (1 + big.size)), 4),
                        "stands_out": bool(m["gap"] > cal["gap_threshold"])})
        else:
            rec.update({"gap_threshold": None, "p_calibrated": None, "stands_out": None})
        rec.update({"inside_mapped_panels": bool(mine[rr, cc]), "within_panel_buffer": bool(mine_buf[rr, cc])})
        ms.append(rec)
    return ms, cal


def calibrated_standouts(qa: dict | None) -> list:
    """The QA's multi-scale stand-outs that passed the calibrated rule
    (decision E7), largest gap first, each with its own location (lat, lon,
    scale_km, gap_in_spreads, gap_threshold, p_calibrated). [] when the
    calibration did not run, and for a QA written before 7 Oct 2026: its
    'standout' came from the uncalibrated gap > 2 rule and supports no
    'stands out clearly'."""
    mp = ((qa or {}).get("findings") or {}).get("monitoring_point") or {}
    if not (mp.get("standout_calibration") or {}).get("tested"):
        return []
    return [m for m in (mp.get("standouts") or [])
            if isinstance(m, dict) and m.get("stands_out") is True and m.get("p_calibrated") is not None]


def _pm(attrs):
    from insar_pipeline.significance import pixel_metres
    return pixel_metres(attrs)


def _closure_file(cfg):
    """The phase-closure diagnostic of the dataset the inversion USED, never
    simply the last file: a reused stack can hold another variant's dataset
    and its file. Returns (path or None, note when None)."""
    import h5py
    files = sorted(cfg.mintpy_dir.glob("numTriNonzeroIntAmbiguity4*.h5")) or \
        ([cfg.mintpy_dir / "numTriNonzeroIntAmbiguity.h5"] if (cfg.mintpy_dir / "numTriNonzeroIntAmbiguity.h5").exists() else [])
    files = [f for f in files if not f.stem.endswith("_phaseClosure")] or files
    if not files:
        return None, "no phase-closure diagnostic on disk"
    stack_f = cfg.mintpy_dir / "inputs" / "ifgramStack.h5"
    if stack_f.exists():
        from insar_pipeline.time_series import inverted_dataset
        with h5py.File(stack_f, "r") as f:
            names = [k for k in f.keys() if k.startswith("unwrapPhase")]
        used = inverted_dataset(cfg, names)
        want = cfg.mintpy_dir / ("numTriNonzeroIntAmbiguity.h5" if used == "unwrapPhase"
                                 else f"numTriNonzeroIntAmbiguity4{used}.h5")
        if not want.exists():
            return None, (f"no phase-closure diagnostic on disk for {used}, the dataset the inversion used "
                          f"({', '.join(p.name for p in files)} are for other datasets)")
        return want, None
    return files[-1], None


def _closure(cfg) -> dict:
    """Median fraction of closed loops with nonzero integer ambiguity, far field."""
    import h5py
    f_used, why = _closure_file(cfg)
    if f_used is None:
        return {"closure_note": why}
    files = [f_used]
    a, a_attrs = _read2d(files[-1])
    a = a.astype(float)
    sel = np.isfinite(a)
    rmask = cfg.mintpy_dir / "maskTempCoh.h5"
    if rmask.exists():
        mk, _ = _read2d(rmask, "mask")
        if mk.shape == a.shape:
            sel &= mk.astype(bool)
    stack = cfg.mintpy_dir / "inputs" / "ifgramStack.h5"
    n_tri = None
    if stack.exists():
        with h5py.File(stack, "r") as f:
            d12 = [f"{x.decode()}_{y.decode()}" for x, y in f["date"][:]]
            keep = f["dropIfgram"][:].astype(bool)
        pairs = {p for p, k in zip(d12, keep) if k}
        dts = sorted({d for p in pairs for d in p.split("_")})
        n_tri = sum(1 for i, x in enumerate(dts) for y in dts[i + 1:] if f"{x}_{y}" in pairs
                    for z in dts if z > y and f"{x}_{z}" in pairs and f"{y}_{z}" in pairs)
    med = float(np.median(a[sel])) if sel.any() else None
    return {"closure_dataset": files[-1].name, "closure_triplets": n_tri,
            "closure_median_open_triplets": med,
            "closure_median_fraction_open": (round(med / n_tri, 3) if (n_tri and med is not None) else None),
            "closure_note": "per-pixel count of closed loops whose phase does not sum to a multiple of 2 pi, on the dataset the inversion used; the open loops at this site are the growing-season ones"}


# ======================================================================
# second layers, dates, sensitivities and the send rule (5 Oct 2026)
# ======================================================================
_EXCLUDED_REASON = ("no interferogram of this date was kept after the network filter (orphaned), so MintPy left it "
                    "out of the inversion")
_EXCLUDED_REF_REASON = ("every interferogram of this date failed the reference-pixel check (the reference fell in "
                        "another unwrapped component; reference_excluded_pairs.json), so it had no kept pair and "
                        "MintPy left it out of the inversion")


def _load_json(path):
    try:
        return json.loads(Path(path).read_text()) if Path(path).exists() else None
    except (OSError, ValueError):
        return None


class _RadarLayers:
    """
    MintPy layers on the product grid: the geocoded copy when one exists
    (geo/geo_<name>, written by MintPy or time_series.geocode_extra_layers),
    else the radar-grid file mapped by nearest neighbour through
    inputs/geometryRadar.h5 (stack_qc.radar_to_geo_index), so the checks
    run on finished projects without re-running MintPy's geocode.
    """

    def __init__(self, cfg, attrs, shape):
        self.cfg, self.attrs, self.shape = cfg, attrs, tuple(shape)
        self._index = None
        self._tried = False
        self.radar_shape = None
        self._cache = {}

    @property
    def index(self):
        if not self._tried:
            self._tried = True
            geom = self.cfg.mintpy_dir / "inputs" / "geometryRadar.h5"
            if geom.exists() and "X_FIRST" in self.attrs:
                import h5py
                from insar_pipeline import stack_qc
                with h5py.File(geom, "r") as g:
                    if "latitude" in g and "longitude" in g:
                        lat, lon = g["latitude"][:], g["longitude"][:]
                        self.radar_shape = lat.shape
                        self._index = stack_qc.radar_to_geo_index(lat, lon, self.attrs, self.shape)
        return self._index

    def radar(self, arr):
        """a radar-grid array on the product grid, or None"""
        from insar_pipeline import stack_qc
        if self.index is None or np.shape(arr) != self.radar_shape:
            return None
        return stack_qc.map_to_geo(arr, self.index)

    def layer(self, name, dataset=None):
        """(array on the product grid, source text) or (None, reason)"""
        key = (name, dataset)
        if key in self._cache:
            return self._cache[key]
        res = (None, f"{name} not on disk")
        geo = self.cfg.mintpy_dir / "geo" / f"geo_{name}"
        rad = self.cfg.mintpy_dir / name
        try:
            if geo.exists():
                arr, _ = _read2d(geo, dataset)
                if arr.shape == self.shape:
                    res = (arr.astype(np.float32), f"geo/geo_{name}")
            if res[0] is None and rad.exists():
                arr, a = _read2d(rad, dataset)
                if "X_FIRST" in a and arr.shape == self.shape:
                    res = (arr.astype(np.float32), name)
                elif "X_FIRST" not in a:
                    m = self.radar(arr)
                    res = ((m, f"{name} (radar grid, mapped to the product grid by nearest neighbour)")
                           if m is not None else (None, f"{name} is on a radar grid that inputs/geometryRadar.h5 "
                                                        "does not describe"))
        except (OSError, KeyError, IndexError) as exc:
            res = (None, f"{name} unreadable: {exc}")
        self._cache[key] = res
        return res


def _unwrap_consistency(cfg, rg, tc, tc_attrs):
    """
    The second gate's layer: MintPy's own network temporal coherence, i.e.
    how consistently the unwrapped interferograms agree with one history.
    On the phase-linked route the working coherence is MiaplPy's (how well
    the phases fit), and temporalCoherence_mintpy.h5 is kept beside it; on
    the classic route the working coherence IS MintPy's, so there is one gate.
    """
    um, src = rg.layer("temporalCoherence_mintpy.h5", "temporalCoherence")
    if um is not None:
        return um, f"MintPy network temporal coherence (unwrapping consistency): {src}"
    work_src = str(tc_attrs.get("temporalCoherenceSource", "") or "")
    disp = (cfg.mintpy_dir / "disp_chain.json").exists() or str(getattr(cfg, "backend", "")) == "disp"
    if not work_src and not disp:
        return tc, "the working coherence is MintPy's own network temporal coherence (one gate)"
    return None, ("no unwrapping-consistency layer for this route"
                  + (" (OPERA DISP-S1 supplies none)" if disp else f" ({src})"))


def _water_on_disk(cfg):
    """the water map the geolocation check uses, when it is on disk (no fetch)"""
    ctx_dir = Path(cfg.project_dir) / "context"
    return next((p for p in (ctx_dir / "water.geojson", ctx_dir / "osm_water.geojson") if p.exists()), None)


def _on_water(path, lat, lon, sel):
    """True where a `sel` cell's (lat, lon) lies in a polygon of the water
    map; None when the map cannot be read or tested. The share on water is a
    record only, so it must never stop the QA: a hand-made water.geojson
    with a self-intersecting polygon made the union raise (GEOS
    TopologyException) in the middle of the coverage check (review 7 Oct
    2026); invalid polygons are now repaired, and anything else leaves the
    share unrecorded."""
    if path is None:
        return None
    out = np.zeros(np.shape(sel), bool)
    if not np.any(sel):
        return out
    rec = _load_json(path)
    if not isinstance(rec, dict):
        return None
    try:
        import shapely
        from shapely.geometry import shape
        feats = rec.get("features") or []
        geoms = []
        for f in feats:
            try:
                g = shape(f["geometry"])
                geoms.append(g if g.is_valid else shapely.make_valid(g))
            except Exception:                                     # noqa: BLE001 - a broken polygon is skipped
                continue
        if feats and not geoms:
            return None
        if geoms:
            out[sel] = shapely.contains_xy(shapely.union_all(geoms), np.asarray(lon)[sel], np.asarray(lat)[sel])
    except Exception as exc:                                      # noqa: BLE001 - a record, never a failure
        log.warning("QA: noise population's share on mapped water not measured (%s): %s", path, exc)
        return None
    return out


def _noise_level(cfg, rg, tc, um, um_src, water=None) -> dict:
    """noise floor of the working coherence (stack_qc.noise_level: one
    population and one estimator for every project, E21), on the radar grid
    where both layers are native, else on the product grid. With a water map
    (`water`, a path) the population's share on mapped water is recorded."""
    from insar_pipeline import stack_qc
    if um is None:
        return {"measured": False, "floor": None, "population": um_src}
    if um is tc:
        return {"measured": False, "floor": None,
                "population": ("the working coherence is MintPy's own: its reliability threshold is the "
                               "literature's (0.7, Yunjun et al. 2019), not a noise floor measured here")}
    res = None
    r_tc, r_um = cfg.mintpy_dir / "temporalCoherence.h5", cfg.mintpy_dir / "temporalCoherence_mintpy.h5"
    from_run = None
    pop = stack_qc.NOISE_POPULATION
    if r_tc.exists() and r_um.exists():
        try:
            a, aa = _read2d(r_tc, "temporalCoherence")
            b, ba = _read2d(r_um, "temporalCoherence")
            from_run = aa.get("linked_noise_level")
            if a.shape == b.shape and "X_FIRST" not in aa:
                wet = None
                geom = cfg.mintpy_dir / "inputs" / "geometryRadar.h5"
                if water is not None and geom.exists():
                    import h5py
                    with h5py.File(geom, "r") as g:
                        if "latitude" in g and g["latitude"].shape == a.shape:
                            wet = _on_water(water, g["latitude"][:], g["longitude"][:],
                                            np.isfinite(a) & np.isfinite(b) & (a > 0) & (b < pop))
                res = stack_qc.noise_level(a, b, on_water=wet)
                res["grid"] = "radar (native)"
        except (OSError, KeyError):
            res = None
    if res is None:
        wet = None
        if water is not None and "Y_FIRST" in rg.attrs:
            ny, nx = np.shape(tc)
            lat = float(rg.attrs["Y_FIRST"]) + (np.arange(ny)[:, None] + 0.5) * float(rg.attrs["Y_STEP"])
            lon = float(rg.attrs["X_FIRST"]) + (np.arange(nx)[None, :] + 0.5) * float(rg.attrs["X_STEP"])
            lat, lon = np.broadcast_to(lat, (ny, nx)), np.broadcast_to(lon, (ny, nx))
            wet = _on_water(water, lat, lon, np.isfinite(tc) & np.isfinite(um) & (tc > 0) & (um < pop))
        res = stack_qc.noise_level(tc, um, on_water=wet)
        res["grid"] = "product"
    if water is not None:
        res["water_map"] = str(water)
    rp = _load_json(cfg.mintpy_dir / "run_products.json") or {}
    from_run = rp.get("linked_noise_level", from_run)
    if from_run not in (None, ""):
        try:
            # for the record only: the time-series step's mode + 1.645 sigma is
            # a different (more lenient) point on the same population, and E21
            # allows one rule - an unmeasured floor stays unmeasured. The
            # step writes the attribute "nan" when it measured nothing (fewer
            # than 300 noise cells): that is no level, and a caption must not
            # quote it ("scores about nan", review 7 Oct 2026)
            v = float(from_run)
            if np.isfinite(v):
                res["linked_noise_level_from_run"] = round(v, 4)
        except (TypeError, ValueError):
            pass
    return res


def _open_loops(cfg, rg, sel, n_tri):
    """open (non-closing) triplets per cell over the measured AOI on the
    product grid: the median count and its fraction of the triplets formed"""
    f, why = _closure_file(cfg)
    if f is None:
        return None
    a, src = rg.layer(f.name)
    if a is None:
        return {"available": False, "reason": src}
    ok = sel & np.isfinite(a)
    if ok.sum() < 50:
        return None
    med = float(np.median(a[ok]))
    out = {"available": True, "layer": src, "triplets": n_tri, "median_open": round(med, 1),
           "median_fraction_open": round(med / n_tri, 3) if n_tri else None}
    return out


def _parse_areas(cfg):
    """[(name, spec)] - the client report's report_areas (client_report.parse_areas)"""
    specs = list(getattr(cfg, "report_areas", ()) or ())
    if not specs:
        specs = ["Full Satellite Imaging:full"] + list(getattr(cfg, "focus_sites_lalo", ()) or ())
    out = []
    for s_ in specs:
        if ":" in str(s_):
            name, val = str(s_).split(":", 1)
            out.append((name.strip(), val.strip()))
    return out


def _area_outline(cfg, spec, attrs, shape):
    """an area's outline on the grid (every cell, coherent or not), or None"""
    from insar_pipeline import significance as sg
    if spec.lower() == "full":
        return sg.aoi_mask(cfg.bbox, attrs, shape)
    if spec.lower().endswith(".geojson"):
        spec = os.path.expandvars(os.path.expanduser(spec))
        return sg.panel_mask(spec, shape, attrs) if Path(spec).exists() else None
    try:
        s_, w, n, e = (float(v) for v in spec.split(","))
    except ValueError:
        return None
    return sg.aoi_mask((min(w, e), min(s_, n), max(w, e), max(s_, n)), attrs, shape)


def _sensitivity_regions(cfg, attrs, shape, mine, roi, panel_geojson) -> dict:
    """{name: (outline mask, kind)}: the report areas, the mapped mine
    polygons and the named panel - every region a figure is reported for"""
    out = {}
    for name, spec in _parse_areas(cfg):
        m = _area_outline(cfg, spec, attrs, shape)
        if m is not None and m.any():
            out[name] = (m, "full" if spec.lower() == "full" else "area")
    if mine is not None and mine.any():
        out["mapped mine polygons"] = (mine, "mine")
    if roi is not None and panel_geojson:
        from insar_pipeline import significance as sg
        out["named panel"] = (sg.panel_mask(panel_geojson, shape, attrs), "panel")
    return out


def report_area_cut(cfg, attrs, shape) -> tuple[np.ndarray | None, list[str]]:
    """
    The report areas other than the whole study area, as one mask of their
    whole outlines (every cell inside, measured or not - client_report._mask's
    footprint), and their names: what the client report takes out of its
    comparison ground (E28: an area must not be compared with itself), for
    significance.far_field_mask(exclude=...) to take out here too, grown by
    its 500 m. Used by the QA, the DISP cross-check (crosscheck.load_fields),
    the products' per-cell null (deliverables._empirical_context) and the
    GNSS inside-station check (gnss_tie.tie) - decision F7(d), 7 Oct 2026 -
    so their technical records read against the report's farmland: until
    then they kept Site B's project area in its own farmland (87,019
    cells). (None, []) when there is no such area; an area whose outline
    cannot be drawn (a missing geojson) is logged and left in, as
    _sensitivity_regions leaves it out of the QA's regions.
    """
    cut, names = None, []
    for name, spec in _parse_areas(cfg):
        if spec.strip().lower() == "full":
            continue
        m = _area_outline(cfg, spec, attrs, shape)
        if m is None:
            log.warning("Report area '%s' (%s) could not be drawn: it stays in the comparison ground", name, spec)
            continue
        m = np.asarray(m, bool)
        cut = m.copy() if cut is None else (cut | m)
        names.append(name)
    return cut, names


def report_farmland_masks(cfg, good, attrs, wide: bool = True) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, dict]:
    """
    (far, far_wide or None, the cut applied or None, meta): the AOI far
    field and the wide far field by significance.far_field_mask with
    exclude= the report areas other than the whole study area
    (report_area_cut) - the client report's comparison ground before its
    feature pass (E28; F7(d), 7 Oct 2026). One exception: when the cut
    itself takes the AOI farmland under CUT_MIN_CELLS cells, the report
    areas cover (nearly) all the farmland and not even a class centre is
    left to measure; the masks are then drawn without the cut so the
    records can still be made (the per-cell model needs farmland cells),
    `applied` is False with the reason, and the QA marks the far field
    thin, which holds the report. (A cut that leaves more, but fewer than
    FAR_MIN_CELLS, is applied: the QA's own rule then reads the far field
    thin.) meta is far_field_mask's (mine, aoi, buffer_m, ...) with
    meta["comparison_ground"] = {report_areas_cut, applied, far_cells,
    cells_removed: {far_field, far_field_wide}, buffer_m, reason}.
    """
    from insar_pipeline import significance as sg
    cut, names = report_area_cut(cfg, attrs, good.shape)
    far, meta = sg.far_field_mask(cfg, good, attrs, exclude=cut)
    rec = {"report_areas_cut": names, "applied": bool(names), "buffer_m": meta.get("exclude_buffer_m"),
           "cells_removed": {"far_field": int(meta.get("excluded_cells") or 0)}}
    # only when it is the cut that empties the farmland: farmland already
    # under CUT_MIN_CELLS without it is the QA's own thin far field, not this
    if names and int(meta.get("excluded_cells") or 0) > 0 and int(far.sum()) < CUT_MIN_CELLS:
        rec.update(applied=False, report_areas_cut=[], report_areas_not_cut=names,
                   reason=(f"the report areas ({', '.join(names)}) cover nearly all the farmland: "
                           f"{int(far.sum())} farmland cells would be left without them ({FAR_MIN_CELLS} are "
                           "needed to test an area), so they were not cut and the far field is too thin to "
                           "test them against"))
        log.warning("Comparison ground: %s", rec["reason"])
        cut = None
        far, meta = sg.far_field_mask(cfg, good, attrs)
        rec["cells_removed"]["far_field"] = 0
        rec["buffer_m"] = None
    far_wide = None
    if wide:
        far_wide, wmeta = sg.far_field_mask(cfg, good, attrs, restrict_to_aoi=False, exclude=cut)
        rec["cells_removed"]["far_field_wide"] = int(wmeta.get("excluded_cells") or 0)
    rec["far_cells"] = int(far.sum())
    meta["comparison_ground"] = rec
    return far, far_wide, cut, meta


def _far_field_sufficiency(far, far_wide, good, regions) -> dict:
    """
    Is there enough ordinary ground to test the report areas against? The
    null of significance.assess places the area's footprint over the
    widest farmland; fewer than ~10 independent placements make its spread
    barely measured (null_thin), and a far field of a few hundred cells
    makes the class means themselves noise.
    """
    # the client's report areas (boxes, polygons; not "full") - the mine
    # layers and the named panel are tested by their own nulls (assess)
    sizes = {n: int((m & good).sum()) for n, (m, k) in regions.items() if k == "area"}
    largest = max(sizes.values()) if sizes else 0
    ratio = (far.sum() / largest) if largest else None
    ratio_w = (far_wide.sum() / largest) if largest else None
    thin = bool(far.sum() < FAR_MIN_CELLS or (ratio_w is not None and ratio_w < FAR_MIN_RATIO))
    return {"cells": int(far.sum()), "cells_wide": int(far_wide.sum()),
            "largest_report_area": max(sizes, key=sizes.get) if sizes else None,
            "largest_report_area_cells": largest,
            "ratio_to_largest_report_area": round(float(ratio), 2) if ratio is not None else None,
            "ratio_wide_to_largest_report_area": round(float(ratio_w), 2) if ratio_w is not None else None,
            "thin": thin,
            "sufficiency_rule": (f"thin when the AOI far field has fewer than {FAR_MIN_CELLS} cells or the widest "
                                 f"farmland holds the largest report area fewer than {FAR_MIN_RATIO:.0f} times")}


def _area_testability(regions, assessed, good, kcls, borrowed, labels) -> dict:
    """
    Per reported region: can it be compared with farmland at all
    (significance.assess 'testable'), on how many independent placements,
    and does it rest on a coherence class whose centre was BORROWED from a
    neighbouring class (too few far-field cells of its own)? A not-testable
    report area holds the report (send_rule); a borrowed class is a note.
    """
    out = {}
    for name, (reg, kind) in regions.items():
        if kind == "full" or name not in assessed:
            continue
        a = assessed[name] or {}
        sel = reg & good
        n = int(sel.sum())
        b_cells = {labels[k]: int((sel & (kcls == k)).sum()) for k in borrowed}
        b_cells = {k: v for k, v in b_cells.items() if v}
        share = round(100 * sum(b_cells.values()) / max(1, n), 1)
        out[name] = {"kind": kind, "cells": n, "testable": a.get("testable"),
                     "not_testable_reason": a.get("not_testable_reason"),
                     "approx_independent_placements": a.get("approx_independent_placements"),
                     "null_thin": a.get("null_thin"),
                     "borrowed_classes": sorted(b_cells), "cells_in_borrowed_classes_percent": share,
                     # a report area (not the mine layers, which have their own
                     # nulls) that cannot be judged against farmland
                     "flag": bool(a.get("testable") is False or share >= 10.0)}
    return out


def _disturbed_effect(ctx) -> dict:
    """
    What the disturbed-ground layer (time_series.write_disturbed_layer:
    the longest run of consecutive nearest-neighbour pairs whose coherence
    is below the measured pair noise level) would do if it were a gate, at
    runs of 10, 20, 30 and 60 pairs: the share of measured AOI cells it
    removes and the move of each reported figure in null sigmas. Measured,
    never applied (Kyle, 5 Oct 2026: measure, then decide).
    """
    from insar_pipeline import significance as sg
    cfg, regions, good, inside = ctx["cfg"], ctx["regions"], ctx["good"], ctx["inside"]
    f = cfg.mintpy_dir / "geo" / "geo_disturbed.h5"
    if not f.exists():
        return {"available": False, "flag": None,
                "reason": "geo/geo_disturbed.h5 not on disk (time_series.geocode_extra_layers adds it)"}
    import h5py
    with h5py.File(f, "r") as h:
        if "disturbed" not in h:
            return {"available": False, "flag": None, "reason": f"{f.name} has no 'disturbed' dataset"}
        d = h["disturbed"][:]
        fa = {k: (v.decode() if isinstance(v, bytes) else v) for k, v in h.attrs.items()}
    if str(fa.get("usable", "yes")) == "no":
        return {"available": False, "flag": None, "usable": False, "reason": fa.get("usable_reason", "")}
    if d.shape != good.shape:
        return {"available": False, "flag": None, "reason": f"{f.name} is {d.shape}, the product {good.shape}"}
    vel, tc, far = ctx["vel"], ctx["tc"], ctx["far"]
    known = d >= 0                                       # -1 = no data
    base = _values(ctx["r_cm"], regions, good)
    variants, removed = {}, {}
    for n in DISTURBED_RUNS:
        keep = good & ~(known & (d >= n))
        removed[str(n)] = round(100 * float((good & inside & ~keep).sum() / max(1, (good & inside).sum())), 1)
        rv, _ = sg.class_matched_field(vel, tc, far & keep)
        variants[f"runs_ge_{n}_mm_yr"] = _values(rv, regions, keep)
    rows, worst = _compare(base, variants, ctx["sigmas"], regions)

    def num(k):
        try:
            return round(float(fa[k]), 4)
        except (KeyError, TypeError, ValueError):
            return None
    return {"available": True, "layer": str(f), "pair_coherence_kind": fa.get("pair_coherence_kind"),
            "pair_noise_level": num("pair_noise_level"), "linked_noise_level": num("linked_noise_level"),
            "no_data_percent": round(100 * float((good & inside & ~known).sum() / max(1, (good & inside).sum())), 1),
            "removed_percent_of_measured_aoi": removed, "areas": rows,
            "worst_change_in_sigma": round(worst, 2), "flag": bool(worst > 1.0),
            "note": ("cells whose longest run of noise-like consecutive pairs reaches N would be removed; the "
                     "figures are recomputed with those cells out of both the area and the farmland. Not applied "
                     "to any delivered figure. With pair_coherence_kind 'phasor' the layer also flags moderately "
                     "coherent ground (Site B/Site A: >= 10 pairs on over 90% of kept cells).")}


def _resolution_check(ctx) -> dict:
    """
    How far apart two cells must be before their errors stop agreeing,
    measured on the product: the autocorrelation of the high-passed rate
    (1 km box removed) on coherent ground (linked coherence >= 0.4) east-west
    and north-south, at lag 1 and where it falls below 0.5. Set against the
    support resolution.py states for the route: a measured length well
    beyond it means the values are smoother than the report says.
    """
    from scipy.ndimage import uniform_filter
    from insar_pipeline import significance as sg
    cfg, vel, tc, good, attrs = ctx["cfg"], ctx["vel"], ctx["tc"], ctx["good"], ctx["attrs"]
    py, px = sg.pixel_metres(attrs)
    sel = good & (tc >= 0.4) & np.isfinite(vel)
    if sel.sum() < 2000:
        return {"available": False, "flag": None, "reason": f"only {int(sel.sum())} coherent cells"}
    k = max(3, int(round(1000.0 / np.sqrt(py * px))) | 1)
    v = np.where(sel, vel, 0.0).astype(np.float32)
    num = uniform_filter(v, k); den = uniform_filter(sel.astype(np.float32), k)
    hp = np.where(sel & (den > 0.2), v - num / np.maximum(den, 1e-6), np.nan).astype(np.float32)
    num = den = v = None
    hp -= np.nanmean(hp)
    var = float(np.nanmean(hp * hp))
    if not np.isfinite(var) or var <= 0:
        return {"available": False, "flag": None, "reason": "no high-passed variance"}

    def acf(axis, step_m, max_m=600.0):
        rows = []
        for lag in range(1, max(2, int(max_m / step_m)) + 1):
            a = hp[:, :-lag] if axis == 1 else hp[:-lag, :]
            b = hp[:, lag:] if axis == 1 else hp[lag:, :]
            prod = a * b
            ok = np.isfinite(prod)
            if ok.sum() < 500:
                break
            rows.append((lag, float(np.mean(prod[ok]) / var)))
        half = next((lag * step_m for lag, r in rows if r < 0.5), None)
        # still at or above 0.5 at the last lag measured: errors shared over
        # at least that distance (the worst case, never "no result")
        if half is None and rows:
            return rows, rows[-1][0] * step_m, True
        return rows, half, False
    ew, half_ew, low_ew = acf(1, px)
    ns, half_ns, low_ns = acf(0, py)
    hp = None
    try:
        from insar_pipeline.resolution import effective_resolution
        res = effective_resolution(cfg, attrs)
    except Exception as exc:                                      # noqa: BLE001
        res = {"text": f"not available ({exc})"}
    sup = res.get("support_m") or [None, None]

    def over(half, lower, s):
        """True beyond twice the support; False within it; None when a lower
        bound does not reach it (the real distance is not known)"""
        if half is None or not s:
            return None
        return True if half > 2 * s else (None if lower else False)
    axes = [over(half_ew, low_ew, sup[0]), over(half_ns, low_ns, sup[1])]
    flag = True if True in axes else (False if axes == [False, False] else None)
    return {"available": True, "posting_m": [round(px, 1), round(py, 1)],
            "lag1_correlation": {"east_west": round(ew[0][1], 3) if ew else None,
                                 "north_south": round(ns[0][1], 3) if ns else None},
            "half_correlation_distance_m": {"east_west": round(half_ew) if half_ew else None,
                                            "north_south": round(half_ns) if half_ns else None},
            "half_correlation_is_lower_bound": {"east_west": low_ew, "north_south": low_ns},
            "stated_support_m": sup, "stated": res.get("text"), "flag": flag,
            "note": ("autocorrelation of the rate after a 1 km box mean is removed, on cells at linked coherence "
                     ">= 0.4; the half-correlation distance is the lag where it first drops below 0.5. When it "
                     "never drops below 0.5 within 600 m, the last lag measured is given as a lower bound "
                     "(half_correlation_is_lower_bound) and flags when it alone is more than twice the support. "
                     "Neighbouring values within it share their errors, so they are not independent "
                     "measurements. flag: more than twice the support resolution.py states.")}


def _reference_distance(ctx) -> dict:
    """
    How far the reference pixel the product is tied to lies from the
    nearest mapped mine EDGE (miaplpy_backend.distance_to_mines_km),
    against acceptance_far_field_buffer_m. A YAML reference inside the
    buffer is kept on purpose by choose_reference, with only a log
    warning; this puts it in the record (tests-2).
    """
    cfg, ref = ctx["cfg"], ctx.get("ref")
    if not ref:
        return {"available": False, "flag": None, "reason": "no reference pixel in the product"}
    mines = str(cfg.deramp_exclude_geojson or "")
    mines = os.path.expandvars(os.path.expanduser(mines)) if mines else ""
    if not mines or not Path(mines).exists():
        return {"available": False, "flag": None, "reason": "no mapped-mine layer (deramp_exclude_geojson)"}
    from insar_pipeline.miaplpy_backend import distance_to_mines_km
    d = distance_to_mines_km(mines, float(ref["lat"]), float(ref["lon"]))
    if d is None:
        return {"available": False, "flag": None, "reason": "the mine layer holds no polygon"}
    buf_km = float(cfg.acceptance_far_field_buffer_m or 0.0) / 1000.0
    rc = _load_json(cfg.stack_dir / "reference_choice.json") or {}
    return {"available": True, "lat": ref["lat"], "lon": ref["lon"],
            "distance_to_nearest_mapped_mine_km": round(float(d), 3), "far_field_buffer_km": buf_km,
            "source": ("set in the YAML" if str(cfg.reference_lalo or "").strip()
                       and not getattr(cfg, "reference_auto", False) else "chosen by the run"),
            "recorded_by_run": ((rc.get("chosen") or {}).get("distance_to_nearest_mapped_mine_km")
                                if isinstance(rc.get("chosen"), dict) else None),
            "flag": bool(d <= buf_km),
            "note": ("distance from the product's reference pixel (REF_LAT/REF_LON) to the nearest mapped mine "
                     "edge (0 inside a polygon); flag when it is not beyond acceptance_far_field_buffer_m. "
                     "Relative-to-farmland figures do not depend on the reference; the absolute ones do.")}


def _unwrap_correction_record(cfg) -> dict | None:
    """What the unwrap-error correction asked for and what it applied
    (time_series.unwrap_error_shortfall). A step that was asked for and made
    nothing is flagged here, and the send rule names it in a note (21 Oct
    2026). None when the run left no record."""
    from insar_pipeline.time_series import unwrap_error_shortfall
    try:
        return unwrap_error_shortfall(cfg)
    except Exception as exc:                                  # noqa: BLE001 - a record, never a failure
        log.warning("QA: unwrap-error correction record could not be read: %s", exc, exc_info=True)
        return {"available": False, "reason": f"could not run: {exc!r}"[:300], "flag": None}


def _phase_linking_record(cfg) -> dict | None:
    """The phase-linked route's own checks, as the reviewer needs them: the
    linked datum gate per mini-stack boundary (miaplpy_backend.
    check_datum_consistency) and what the linking fingerprint recorded."""
    if cfg.backend != "miaplpy":
        return None
    merged = cfg.stack_dir / "merged"
    dc = _load_json(merged / "linked_datum_check.json")
    fp = _load_json(cfg.project_dir / "miaplpy" / "inverted" / "linking_fingerprint.json")
    out = {"datum_check": "absent (linked_datum_check.json not on disk)"}
    if isinstance(dc, dict) and not isinstance(dc.get("coherent_cells"), dict):
        # a record written before 5 Oct 2026: the old central-block gate, which
        # raised when it failed, so a record on disk is one that passed it
        out["datum_check"] = {"legacy_record": True, "passed_old_gate": True, "method": dc.get("method"),
                              "ratio_across_over_within": dc.get("ratio_across_over_within"),
                              "median_block_coherence_within": dc.get("median_block_coherence_within"),
                              "median_block_coherence_across": dc.get("median_block_coherence_across"),
                              "note": "central-block check before 5 Oct 2026; the per-boundary coherent-cell "
                                      "check runs on the next phase-linking run"}
    elif isinstance(dc, dict):
        cc = dc.get("coherent_cells") if isinstance(dc.get("coherent_cells"), dict) else {}
        out["datum_check"] = {"passed": dc.get("passed"), "testable": dc.get("testable"),
                              "median_ratio": cc.get("median_ratio"),
                              "median_across": cc.get("median_across"),
                              "median_neighbours": cc.get("median_neighbours"),
                              "boundaries_judged": cc.get("boundaries_judged"),
                              "boundaries_not_judged": cc.get("boundaries_not_judged"),
                              "weakest_boundary_ratio": (min((b.get("ratio") for b in cc.get("per_boundary") or []
                                                              if isinstance(b, dict) and b.get("ratio") is not None),
                                                             default=None)),
                              "rule": dc.get("rule")}
    if isinstance(fp, dict):
        out["linking"] = {k: fp.get(k) for k in ("method", "ministack_size", "time_lag", "range_window",
                                                  "azimuth_window", "n_dates", "status", "source",
                                                  "miaplpy_version", "patch_check_ratio")}
    return out


def _injection_record(cfg) -> dict | None:
    """injection_test.py's detection table (deliverables/technical/
    injection_recovery.json), when the stage ran: informational, never a gate.
    A record left from an earlier velocity product, or with injection_test
    now off, is not this run's and reads as not run
    (crosscheck.record_is_current)."""
    from insar_pipeline.crosscheck import record_is_current
    rec = _load_json(cfg.project_dir / "deliverables" / "technical" / "injection_recovery.json")
    if not isinstance(rec, dict):
        return None
    if not record_is_current(cfg, rec, "injection_test"):
        log.info("QA: injection_recovery.json is not this run's (injection_test off, or older than the "
                 "velocity product) - read as not run")
        return None
    # E4 (7 Oct 2026): technical only. The record has no client sentence
    # ('sentence' is None on a current record); it carries the decision rule
    # behind 'found' and the technical summary (detections with their 95%
    # ranges beside the zero-rate control).
    return {"status": rec.get("status"), "sentence": rec.get("sentence"), "summary": rec.get("summary"),
            "decision_rule": rec.get("decision_rule"), "technical_summary": rec.get("technical_summary"),
            "not_covered": rec.get("not_covered"), "seed": rec.get("seed"),
            "chain_attenuation": rec.get("temporal_gain"), "reason": rec.get("reason"),
            "gates_verdict": False,
            "note": ("technical record, not for the client page: known signals planted into the official velocity "
                     "through the same deramp, DEM-error and rate operators; 'found' is the report-area rule over "
                     "an area drawn on the planted patch, sinking only, and the map's feature search is not "
                     "tested. It does not cover phase linking, unwrapping or the coherence mask, so detection "
                     "rates are an upper bound")}


def _independent_check(cfg) -> dict | None:
    """crosscheck.py's comparison with OPERA DISP-S1 (deliverables/technical/
    cross_check.json), when the stage ran: informational, never a gate - its
    pass thresholds are calibrated on one site only. A record left from an
    earlier velocity product, or with cross_check_disp now off, is not this
    run's and reads as not run (crosscheck.record_is_current)."""
    from insar_pipeline.crosscheck import record_is_current
    rec = _load_json(cfg.project_dir / "deliverables" / "technical" / "cross_check.json")
    if not isinstance(rec, dict):
        return None
    if not record_is_current(cfg, rec, "cross_check_disp"):
        log.info("QA: cross_check.json is not this run's (cross_check_disp off, or older than the "
                 "velocity product) - read as not run")
        return None
    one = rec.get("one_km_means") if isinstance(rec.get("one_km_means"), dict) else {}
    return {"status": rec.get("status"), "available": rec.get("available"), "sentence": rec.get("sentence"),
            "reason": rec.get("reason"), "one_km_correlation": one.get("correlation"),
            "areas_agree": {a.get("name"): a.get("agree") for a in rec.get("areas") or [] if isinstance(a, dict)},
            "features_confirmed": [f.get("index") for f in rec.get("features") or []
                                   if isinstance(f, dict) and f.get("confirmed")],
            "indicative_pass": rec.get("indicative_pass"), "gates_verdict": False}


def _values(field, regions, good, kinds=("area", "mine", "panel", "full")) -> dict:
    out = {}
    for name, (m, k) in regions.items():
        sel = m & good & np.isfinite(field)
        if k in kinds and sel.sum() >= 5:
            out[name] = float(field[sel].mean())
    return out


def _compare(base, variants, sigmas, regions, hold_sigma=1.0, note_sigma=None):
    """per-region table of a figure under alternatives, in null sigmas"""
    rows, worst = [], 0.0
    for name, v0 in base.items():
        kind = regions[name][1]
        sd = sigmas.get(name)
        row = {"name": name, "kind": kind, "reported_mm_yr": round(v0, 2),
               "null_sigma_mm_yr": round(sd, 2) if sd else None}
        ch = []
        for vname, vals in variants.items():
            v = vals.get(name)
            row[vname] = round(v, 2) if v is not None else None
            if v is not None:
                ch.append(v - v0)
        if ch:
            mx = max(ch, key=abs)
            row["max_change_mm_yr"] = round(mx, 2)
            if sd and kind != "full":
                row["max_change_in_sigma"] = round(abs(mx) / sd, 2)
                worst = max(worst, abs(mx) / sd)
        rows.append(row)
    return rows, worst


def _second_gate_effect(ctx) -> dict:
    """what gating on unwrapping consistency at 0.5 / 0.7 would do to each
    reported figure: value, measured share and the move in null sigmas"""
    from insar_pipeline import significance as sg
    um, regions, measured = ctx["um"], ctx["regions"], ctx["measured"]
    if um is None:
        return {"available": False, "reason": ctx["um_src"], "flag": None}
    vel, tc, far = ctx["vel"], ctx["tc"], ctx["far"]
    umf = np.nan_to_num(um, nan=-1.0)
    base_f, _ = sg.class_matched_field(vel, tc, far & measured)
    base = _values(base_f, regions, measured)
    variants, shares = {}, {}
    for g in (0.5, RELIABLE_UNWRAP_CONSISTENCY):
        gm = measured & (umf >= g)
        rv, _ = sg.class_matched_field(vel, tc, far & gm)
        variants[f"gate_{g:.1f}_mm_yr"] = _values(rv, regions, gm)
        shares[f"{g:.1f}"] = {n: round(float((m & gm).sum() / max(1, (m & measured).sum())), 3)
                              for n, (m, k) in regions.items()}
    rows, worst = _compare(base, variants, ctx["sigmas"], regions)
    for r in rows:
        r["measured_share"] = {g: v.get(r["name"]) for g, v in shares.items()}
    return {"available": True, "areas": rows, "worst_change_in_sigma": round(worst, 2),
            "flag": bool(worst > 1.0),
            "note": ("each figure (class-matched, relative to farmland, LOS mm/yr) recomputed with the cells and the "
                     "farmland both restricted to unwrapping consistency >= 0.5 and >= 0.7; measured_share is the "
                     "share of the area's measured cells that survive. Report figures are not gated unless "
                     "min_unwrap_consistency > 0. flag: a figure moves by more than its farmland null sigma.")}


def _ramp_sensitivity(ctx) -> dict:
    """
    How much the deramp decided each reported figure. A plane fitted to the
    deramp-masked field of a ramp-removed rate is, by linearity, the change
    the other choice would make: fitting a quadratic to the linear-deramped
    rate gives (quadratic ramp - linear ramp); fitting a plane over a 4 km
    exclusion gives (4 km plane - 2 km plane). So both variants come from
    the delivered velocity alone. "before deramp" needs the series before
    the step (timeseries_ERA5.h5 and its _ramp copy), which cleanup level 2
    removes; then it is reported as not available.
    """
    from insar_pipeline import significance as sg
    from insar_pipeline import stack_qc
    cfg, vel, tc, good, far, mine, attrs = (ctx[k] for k in ("cfg", "vel", "tc", "good", "far", "mine", "attrs"))
    regions = ctx["regions"]
    deramp = str(getattr(cfg, "mintpy_deramp", "no") or "no").lower()
    buf = float(cfg.deramp_buffer_m or 2000.0)
    alt = 4000.0 if abs(buf - 4000.0) > 1.0 else 2000.0
    fit = good & ~sg.euclidean_buffer(mine, attrs, buf)
    fit_alt = good & ~sg.euclidean_buffer(mine, attrs, alt)
    out = {"available": True, "deramp": deramp, "buffer_m": buf, "alternative_buffer_m": alt,
           "fit_area_percent": round(100 * float(fit.sum() / max(1, good.sum())), 1)}
    rmask = cfg.mintpy_dir / "maskDeramp.h5"
    # how the fit area was built and how evenly it surrounds the area
    # (stack_backend.build_deramp_mask attrs; time_series' deramp_mask.json)
    dmj = _load_json(cfg.mintpy_dir / "deramp_mask.json") or {}
    if dmj:
        out["deramp_mask_record"] = {k: dmj.get(k) for k in ("mask", "built_by_pipeline", "buffer_m", "kept_fraction",
                                                             "coherent_pixels", "kept_pixels") if k in dmj}
    if rmask.exists():
        try:
            dm, dma = _read2d(rmask, "mask")
            for k in ("DERAMP_BUFFER_M", "DERAMP_BUFFER_KIND", "DERAMP_QUADRANTS", "DERAMP_QUADRANT_SHARE"):
                if k in dma:
                    v = dma[k]
                    v = v.decode() if isinstance(v, bytes) else v
                    if k == "DERAMP_QUADRANT_SHARE":
                        try:
                            v = json.loads(v)
                        except (TypeError, ValueError):
                            pass
                    out.setdefault("deramp_mask_geometry", {})[k[7:].lower()] = v.item() if hasattr(v, "item") else v
            src_m = cfg.mintpy_dir / "maskTempCoh.h5"
            tm = _read2d(src_m, "mask")[0].astype(bool) if src_m.exists() else np.ones(dm.shape, bool)
            if tm.shape == dm.shape:
                out["fit_area_percent_mintpy_mask"] = round(100 * float((dm.astype(bool) & tm).sum() / max(1, tm.sum())), 1)
        except (OSError, KeyError):
            pass
    if mine is not None and mine.any():
        # how much of the ground around the mines the plane is actually fitted
        # on: with mines filling the box the plane is pinned on a fringe
        near = sg.euclidean_buffer(mine, attrs, 10_000.0) & good
        out["fit_area_percent_within_10km_of_mines"] = round(100 * float((near & fit).sum() / max(1, near.sum())), 1)
        near = None
    base = _values(ctx["r_cm"], regions, good)
    vals = {}

    def variant(name, field):
        """one alternative field -> its class-matched region values (field freed after)"""
        rv, _ = sg.class_matched_field(field, tc, far)
        vals[name] = _values(rv, regions, good)
    if deramp in ("linear", "no"):
        q = stack_qc.surface_fit(vel, fit, 2)
        if q is not None:
            variant("quadratic_ramp_mm_yr", vel - q)
    if deramp == "no":
        q = stack_qc.surface_fit(vel, fit, 1)
        if q is not None:
            variant("linear_ramp_mm_yr", vel - q)
    if deramp in ("linear", "quadratic"):
        q = stack_qc.surface_fit(vel, fit_alt, 1 if deramp == "linear" else 2)
        if q is not None:
            variant(f"buffer_{alt / 1000:.0f}km_mm_yr", vel - q)
    q = None
    before, why = _ramp_removed(cfg, ctx["rg"]) if deramp != "no" else (None, "no deramp applied")
    if before is not None:
        variant("before_deramp_mm_yr", vel + before)
        rr = before[good & np.isfinite(before)]
        if rr.size:
            out["ramp_rate_spread_p5_p95_mm_yr"] = [round(float(np.percentile(rr, 5)), 2),
                                                    round(float(np.percentile(rr, 95)), 2)]
        out["before_deramp"] = f"measured: {why}"
    else:
        out["before_deramp"] = why
    rows, worst = _compare(base, vals, ctx["sigmas"], regions)
    # The tilt the deramp actually removed (time_series.measure_removed_ramp,
    # mintpy/deramp_effect.json = run_products.json removed_ramp), against the
    # smallest 80%-power detection limit of the reported areas: a removed
    # gradient of more than half that limit per 10 km is a tilt the size of
    # what the report could detect, taken out by a modelling choice.
    removed = _removed_ramp_record(cfg)
    if removed is None and before is not None:
        # no record from the time-series step, but the series before the
        # deramp is on disk: measure the removed tilt from it, the way
        # time_series.measure_removed_ramp does (a least-squares plane)
        removed = _removed_ramp_from(cfg, attrs, before, good, deramp, why)
    before = None
    # a check that did not run reads "not run", never as a pass (E24, 7 Oct 2026)
    gflag, gstatus = None, None
    if removed is not None:
        out["removed_ramp"] = removed
    if deramp == "no":
        gflag, gstatus = False, "no deramp applied: nothing was removed"
    elif removed is None:
        gstatus = ("not measured: no deramp_effect.json or run_products.json removed_ramp, and the series before "
                   "the deramp is not on disk")
    else:
        lims = [a.get("detectable_80_mm_yr") for a in (ctx.get("assessed") or {}).values()
                if isinstance(a, dict) and a.get("detectable_80_mm_yr")]
        g = removed.get("gradient_mm_yr_per_10km")
        if removed.get("applied") is False:
            gflag, gstatus = False, "the record says no ramp was removed"
        elif g is None:
            gstatus = "not measured: the removed-ramp record has no gradient"
        elif not lims:
            gstatus = "not tested: no report area has a detection limit to compare the tilt with"
        else:
            lim = float(min(lims))
            out["removed_gradient_vs_detection_limit"] = round(float(g) / lim, 2)
            gflag = bool(float(g) > 0.5 * lim)
            gstatus = (f"measured: {float(g):.2f} mm/yr per 10 km against half the smallest detection limit "
                       f"({0.5 * lim:.2f})")
    out.update({"areas": rows, "worst_change_in_sigma": round(worst, 2),
                "removed_gradient_flag": gflag, "removed_gradient_status": gstatus,
                "flag": bool(worst > 0.5 or bool(gflag)), "hold": bool(worst > 1.0),
                "note": ("each reported figure (class-matched, relative to farmland, LOS mm/yr) under the other "
                         "deramp choices; flag above 0.5 null sigma, hold above 1. A dewatering cone wider than "
                         "the exclusion buffer is partly removed by the ramp and partly falls in the far field, "
                         "which this table shows as a buffer-dependent figure.")})
    return out


def _removed_ramp_record(cfg) -> dict | None:
    """the removed-ramp record the time-series step wrote (deramp_effect.json,
    or run_products.json removed_ramp); None on older projects"""
    rec = _load_json(cfg.mintpy_dir / "deramp_effect.json")
    if not isinstance(rec, dict):
        rec = (_load_json(cfg.mintpy_dir / "run_products.json") or {}).get("removed_ramp")
    if not isinstance(rec, dict):
        return None
    keep = ("applied", "method", "removed_rate_p2_mm_yr", "removed_rate_p98_mm_yr", "removed_rate_range_mm_yr",
            "gradient_mm_yr_per_10km", "east_mm_yr_per_10km", "north_mm_yr_per_10km", "extent_km",
            "fit_kept_fraction", "fit_buffer_m", "diagnostic_plane")
    return {k: rec[k] for k in keep if k in rec}


def _removed_ramp_from(cfg, attrs, before, good, deramp, why) -> dict | None:
    """the removed-ramp record measured from the removed rate on the product
    grid (`before`, _ramp_removed): percentiles and the plane gradient of
    time_series.measure_removed_ramp, over the coherent cells"""
    from insar_pipeline.time_series import _grid_km, _plane
    use = good & np.isfinite(before)
    if use.sum() < 3:
        return None
    km = _grid_km(cfg, attrs, before.shape)
    pl = _plane(before, use, km) if km is not None else {}
    if not pl:
        return None
    v = before[use]
    return {"applied": True, "method": deramp,
            "removed_rate_p2_mm_yr": round(float(np.percentile(v, 2)), 3),
            "removed_rate_p98_mm_yr": round(float(np.percentile(v, 98)), 3),
            "removed_rate_range_mm_yr": round(float(np.percentile(v, 98) - np.percentile(v, 2)), 3),
            **pl, "source": f"measured by the QA from {why} (no record from the time-series step)"}


def _ramp_removed(cfg, rg):
    """the rate of the ramp the deramp step removed, on the product grid"""
    from insar_pipeline import stack_qc
    M = cfg.mintpy_dir
    pairs = []
    for rp in sorted(M.glob("timeseries*_ramp.h5")):
        if "Residual" in rp.name:
            continue
        base = rp.with_name(rp.name[:-len("_ramp.h5")] + ".h5")
        if base.exists():
            pairs.append((base, rp))
    if not pairs:
        return None, ("the series before the deramp (timeseries*_ramp.h5 and its input) is not on disk - "
                      "cleanup removes it; re-run the QA before cleanup to measure it")
    import h5py
    # the pair the delivered series came through (geo_timeseries_ERA5_ramp_demErr
    # -> timeseries_ERA5_ramp), not whichever sorts last
    try:
        from insar_pipeline import deliverables as dl
        delivered = dl.timeseries_file(cfg).name.replace("geo_", "", 1)
        pairs.sort(key=lambda bp: delivered.startswith(bp[1].stem))
    except Exception:                                             # noqa: BLE001
        pass
    base, rp = pairs[-1]
    rate, _ = stack_qc.stream_rate(base, rp)
    with h5py.File(base, "r") as f:
        is_geo = "X_FIRST" in f.attrs
    if is_geo:
        return (rate if rate.shape == rg.shape else None), f"{base.name} minus {rp.name}"
    m = rg.radar(rate)
    return m, (f"{base.name} minus {rp.name}" if m is not None else "radar grid not mappable")


def _second_axis_fields(ctx) -> dict:
    """{label: integer class per cell} for the alternative matchings"""
    cfg, far, rg = ctx["cfg"], ctx["far"], ctx["rg"]
    axes, notes = {}, {}

    def terciles(a, label):
        ok = far & np.isfinite(a)
        if ok.sum() < 600:
            notes[label] = "too few far-field cells"
            return
        e1, e2 = np.percentile(a[ok], [100 / 3, 200 / 3])
        k = np.full(a.shape, -1, int)
        fin = np.isfinite(a)
        k[fin] = np.digitize(a[fin], [e1, e2])
        axes[label] = k
    f, why = _closure_file(cfg)
    if f is not None:
        a, src = rg.layer(f.name)
        if a is not None:
            terciles(a, "open_loop_tercile")
        else:
            notes["open_loop_tercile"] = src
    else:
        notes["open_loop_tercile"] = why
    if ctx["um"] is not None and ctx["um"] is not ctx["tc"]:
        terciles(ctx["um"], "unwrap_consistency_tercile")
    else:
        notes["unwrap_consistency_tercile"] = "no independent unwrapping-consistency layer"
    from insar_pipeline import landcover
    lc = landcover.raster_path(cfg)          # landcover_raster, or the fetched WorldCover map
    if lc:
        try:
            axes["land_cover"] = _landcover_on_grid(lc, ctx["attrs"], ctx["vel"].shape)
        except Exception as exc:                                  # noqa: BLE001
            notes["land_cover"] = f"landcover_raster could not be read: {exc}"
    return axes, notes


def _landcover_on_grid(path, attrs, shape):
    """an NLCD-like class raster sampled (nearest) at every cell centre"""
    from insar_pipeline import landcover
    return landcover.sample_on_grid(path, attrs, shape)


def _matched_field(vel, tc, far, axis, min_n=200):
    """vel minus the far-field mean of the cell's (coherence class, second
    class); cells whose pair is thin in the far field fall back to their
    coherence class alone (significance.class_matched_field)"""
    from insar_pipeline import significance as sg
    base, means = sg.class_matched_field(vel, tc, far)
    k = sg.coherence_class(tc)
    key = np.where((k >= 0) & (axis >= 0), k * 100_000 + axis, -1)
    out = base.copy()
    sel_far = far & np.isfinite(vel) & (key >= 0)
    ks, inv, cnt = np.unique(key[sel_far], return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=vel[sel_far])
    for kk, n, sm in zip(ks, cnt, sums):
        if n >= min_n:
            sel = key == kk
            out[sel] = vel[sel] - sm / n
    return out


def _matching_sensitivity(ctx) -> dict:
    """
    Class matching uses one variable, the linked coherence. Within a class
    the far-field rate still varies with unwrapping consistency (Site A
    0.6-0.7 class: lowest vs highest open-loop tercile -3.4 vs -10.1 mm/yr;
    the dominant 0.4-0.6 class only ~1 mm/yr), so ground unlike farmland of
    its class can carry a residual of several mm/yr. Each figure is
    recomputed with a second matching axis (open-loop tercile, unwrapping-
    consistency tercile, land cover when landcover_raster is set); a change
    above 1 farmland null sigma flags it.
    """
    axes, notes = _second_axis_fields(ctx)
    if not axes:
        return {"available": False, "reason": "; ".join(f"{k}: {v}" for k, v in notes.items()), "flag": None}
    regions, good = ctx["regions"], ctx["good"]
    base = _values(ctx["r_cm"], regions, good)
    vals = {}
    for label, ax in axes.items():
        vals[f"{label}_mm_yr"] = _values(_matched_field(ctx["vel"], ctx["tc"], ctx["far"], ax), regions, good)
    rows, worst = _compare(base, vals, ctx["sigmas"], regions)
    return {"available": True, "axes": sorted(axes), "not_used": notes, "areas": rows,
            "worst_change_in_sigma": round(worst, 2), "flag": bool(worst > 1.0),
            "note": ("each figure recomputed matching the farmland by coherence class AND a second variable (the "
                     "far field's terciles, or land-cover class); flag when any figure moves by more than its "
                     "farmland null sigma: the figure then depends on what the comparison ground is taken to be.")}


def _look_cell_m(cfg, attrs):
    """ground size (ew, ns) in metres of one multilooked radar cell"""
    try:
        from insar_pipeline import resolution
        r = resolution.effective_resolution(cfg, attrs)
        lc = r.get("look_cell_m")
        if lc:
            return float(lc[0]), float(lc[1])
    except Exception:                                             # noqa: BLE001
        pass
    try:
        inc = np.radians(float(attrs.get("CENTER_INCIDENCE_ANGLE", 39.0)))
        ew = float(attrs["RANGE_PIXEL_SIZE"]) / np.sin(inc)
        # AZIMUTH_PIXEL_SIZE is at the satellite's altitude; the ground
        # spacing is smaller by R/(R+H), 6371/(6371+693) for Sentinel-1
        ns = float(attrs["AZIMUTH_PIXEL_SIZE"]) * 6371.0 / (6371.0 + 693.0)
        return ew, ns
    except (KeyError, TypeError, ValueError):
        return 41.0, 42.0                                         # S-1 IW at 11 x 3 looks


def _plausibility(ctx) -> dict:
    """
    Cells the network cannot have measured: a rate that differs from the
    adjacent looked cell by more than a quarter wavelength per typical
    revisit (~422 mm/yr at 12 days). Unwrapping is spatial, so a broad bowl
    may sink faster than that and be right; a neighbour-to-neighbour
    difference that large is aliased (a whole-cycle error), whatever the
    absolute rate. Counted, not masked: the count and where it falls go to
    the report; the max_plausible_velocity_mm_yr absolute check is shown
    beside it for continuity.
    """
    from insar_pipeline import significance as sg
    from insar_pipeline import stack_qc
    cfg, vel, good, inside, attrs, regions = (ctx[k] for k in ("cfg", "vel", "good", "inside", "attrs", "regions"))
    wl = float(attrs.get("WAVELENGTH", 0.05546576) or 0.05546576)
    lim = stack_qc.gradient_limit_mm_yr(wl, ctx["dates"])
    py, px = sg.pixel_metres(attrs)
    ew, ns = _look_cell_m(cfg, attrs)
    step = (max(1, int(round(ns / py))), max(1, int(round(ew / px))))
    bad = stack_qc.rate_gradient_flags(vel, step, lim["limit_mm_yr"], valid=good)
    n_aoi = int((good & inside).sum())
    by_area = []
    worst_area = 0.0
    for name, (m, k) in regions.items():
        n = int((m & good).sum())
        if n:
            pct = 100 * float((bad & m).sum() / n)
            by_area.append({"name": name, "kind": k, "cells": int((bad & m).sum()), "percent": round(pct, 3)})
            if k != "full":
                worst_area = max(worst_area, pct)
    cap = float(getattr(cfg, "max_plausible_velocity_mm_yr", 500.0) or 500.0)
    pct_aoi = 100 * float((bad & inside).sum() / max(1, n_aoi))
    return {"available": True, **lim, "look_cell_m": [round(ew, 1), round(ns, 1)], "neighbour_step_cells": list(step),
            "cells_over_gradient_limit": int((bad & inside).sum()), "percent_of_measured_aoi": round(pct_aoi, 3),
            "by_area": by_area,
            "max_plausible_velocity_mm_yr": cap,
            "cells_over_max_plausible": int((good & inside & (np.abs(vel) > cap)).sum()),
            "flag": bool(pct_aoi > 0.1 or worst_area > 1.0), "hold": bool(worst_area > 5.0),
            "note": ("cells whose rate differs from a cell one look away by more than a quarter wavelength per "
                     "median revisit: too fast a change across the ground for unwrapping to follow, so the value "
                     "is not a measurement ('not measurable - too fast for this satellite'). flag above 0.1% of "
                     "the measured AOI or 1% of a report area; hold above 5% of a report area.")}


def _dem_error_diagnostic(ctx) -> dict:
    """
    Position shift implied by the DEM height error MintPy estimated: a
    surface dh off the DEM moves a cell by |dh| / tan(incidence) in ground
    range (x1.23 at 39 deg). The DEM (Copernicus GLO-30, 2011-15) predates
    pits and spoil, so near active mining cells can sit 2-4 postings from
    where they are drawn - and fall on the wrong side of a mine outline.
    Part of demErr is deformation that correlates with baseline, so this is
    an indication, not a correction.
    """
    from insar_pipeline import deliverables as dl
    from insar_pipeline import significance as sg
    cfg, good, mine, attrs, regions, rg = (ctx[k] for k in ("cfg", "good", "mine", "attrs", "regions", "rg"))
    dh, src = rg.layer("demErr.h5", "dem")
    if dh is None:
        return {"available": False, "reason": src, "flag": None}
    inc = dl.geometry_field(cfg, "incidence")
    if inc is None or np.shape(inc) != dh.shape:
        inc = np.full(dh.shape, float(attrs.get("CENTER_INCIDENCE_ANGLE", 39.0) or 39.0))
    shift = np.abs(dh) / np.tan(np.radians(inc))
    py, px = sg.pixel_metres(attrs)
    cell = float(min(py, px))
    ok = good & np.isfinite(shift)
    zones = {n: (m, k) for n, (m, k) in regions.items()}
    # ~100 m either side of an outline (verify note), never less than one
    # cell either side on a coarse grid
    w = max(100.0, 1.5 * float(max(py, px)))
    outlines = [(f"mine edges (within {w:.0f} m)", mine)]
    sm = str(getattr(cfg, "surface_mines_geojson", "") or "")
    if sm and Path(sm).exists():
        # pits and spoil are where the 2011-15 DEM is most out of date
        outlines.append((f"surface-mine edges (within {w:.0f} m)", sg.panel_mask(sm, good.shape, attrs)))
    for label, poly in outlines:
        if poly is not None and poly.any():
            zones[label] = (sg.euclidean_buffer(poly, attrs, w) & sg.euclidean_buffer(~poly, attrs, w), "edge")
    far = ctx["far"] & np.isfinite(shift)
    base = 100 * float(np.mean(shift[far] > cell)) if far.sum() >= 20 else None
    rows, worst = [], 0.0
    for name, (m, k) in zones.items():
        sel = ok & m
        if sel.sum() < 20:
            continue
        pct = 100 * float(np.mean(shift[sel] > cell))
        rows.append({"name": name, "kind": k, "cells": int(sel.sum()), "percent_over_one_cell": round(pct, 2),
                     "percent_over_two_cells": round(100 * float(np.mean(shift[sel] > 2 * cell)), 2),
                     "excess_over_farmland_points": round(pct - base, 2) if base is not None else None,
                     "median_shift_m": round(float(np.median(shift[sel])), 1),
                     "p99_shift_m": round(float(np.percentile(shift[sel], 99)), 1)})
        if k != "full" and base is not None:
            worst = max(worst, pct - base)
    return {"available": True, "layer": src, "posting_m": round(cell, 1),
            "farmland_median_abs_dem_error_m": round(float(np.median(np.abs(dh[far]))), 2) if far.any() else None,
            "farmland_percent_over_one_cell": round(base, 2) if base is not None else None,
            "percent_over_one_cell_measured": round(100 * float(np.mean(shift[ok] > cell)), 2) if ok.any() else None,
            "zones": rows, "worst_excess_points": round(worst, 2), "flag": bool(worst > DEM_SHIFT_WARN_PERCENT),
            "note": (f"share of coherent cells whose estimated DEM height error would shift them by more than one "
                     f"posting cell ({cell:.0f} m) in ground range (|demErr| / tan(incidence)), set against the "
                     f"farmland's share (demErr is noisy); flag when a report area or the mine edges exceed the "
                     f"farmland by more than {DEM_SHIFT_WARN_PERCENT:.0f} percentage points - there cells near an "
                     "outline may be drawn on the wrong side of it. Indicative: part of demErr is deformation "
                     "that correlates with baseline.")}


def _geolocation(cfg, attrs, tc, imaged, fetch_water) -> dict:
    """
    The product's position against independently mapped water (OSM, cached
    in <project>/context/osm_water.geojson; context/water.geojson is used
    first when a coworker supplies one). Nothing else in the chain would
    catch a geoid/ellipsoid DEM mix-up (~40 m) or a corner/centre slip.
    """
    from insar_pipeline import stack_qc
    ctx_dir = cfg.project_dir / "context"
    water = next((p for p in (ctx_dir / "water.geojson", ctx_dir / "osm_water.geojson") if p.exists()), None)
    if water is None and fetch_water:
        water = stack_qc.fetch_osm_water(tuple(cfg.bbox), ctx_dir / "osm_water.geojson")
    if water is None:
        return {"available": False, "flag": None,
                "reason": ("no water map: context/osm_water.geojson is not on disk"
                           + ("" if fetch_water else " and the fetch was not allowed")
                           + " - the check did not run")}
    feats = (_load_json(water) or {}).get("features", [])
    geoms = [f.get("geometry") for f in feats if f.get("geometry")]
    field, src = tc, "temporal coherence"
    sc = cfg.mintpy_dir / "geo" / "geo_avgSpatialCoh.h5"
    if sc.exists():
        a, _ = _read2d(sc, "coherence")
        if a.shape == tc.shape:
            field, src = a, "geo_avgSpatialCoh.h5 (average spatial coherence)"
    res = stack_qc.geolocation_vs_water(field, attrs, geoms, valid=np.isfinite(field))
    res.update({"water_map": str(water), "field": src})
    return judge_geolocation(res)


def judge_geolocation(res: dict | None) -> dict | None:
    """
    The verdict and the sentence of a geolocation record (stack_qc.
    geolocation_vs_water), so the QA and a report built from an older QA
    say the same thing. Idempotent.

    A half of the area counts against the positions only when its offset is
    more than one posting cell (cell_m, 15 m) AND the halves disagree beyond
    their own scatter (split_test p < GEOLOCATION_SPLIT_ALPHA) - decision
    E22, 7 Oct 2026. Before it, one half over 10 m held the report: an 11 m
    east half at Site A (12 lakes, under one cell) and a 22 m west half
    at Site B resting on 5 lakes, which random 5/41 splits of the same
    windows reach in about a third of cases.

    flag True   the joint offset is over GEOLOCATION_FLAG_M (whatever the
                halves say: a split never softens it), or one half is more
                than one cell off and the halves disagree beyond their own
                scatter;
    flag None   inconclusive: a record without the split test (an older
                QA) in which one half is more than one cell off and the
                halves disagree by more than max(joint, GEOLOCATION_FLAG_M).
                `hold` is True: the test that could clear it did not run;
    flag False  otherwise - the "agree ... to within about N m" sentence,
                on the whole-area figure. When a half reads noticeably more
                but within one cell, the sentence gives it too and
                `within_cell` is True (send_rule notes it).
    worst_offset_m is the largest of the joint and the half offsets.
    """
    if not isinstance(res, dict) or not res.get("available", True):
        return res
    res = dict(res)
    off, dis = res.get("offset_m"), res.get("west_east_disagreement_m")
    for k in ("status", "hold", "worst_offset_m", "within_cell"):
        res.pop(k, None)
    if off is None:
        res["flag"] = None
        res.pop("text", None)
        return res
    off = float(off)
    cell = float(res.get("cell_m") or GEOLOCATION_CELL_M)
    halves = res.get("halves") or {}
    sides = {k: float(h["offset_m"]) for k, h in halves.items()
             if k in ("west", "east") and isinstance(h, dict) and h.get("offset_m") is not None}
    side = max(sides, key=sides.get) if sides else None
    half_over_cell = side is not None and sides[side] > cell
    st = res.get("split_test") if isinstance(res.get("split_test"), dict) else {}
    p = st.get("p_value")
    tested = p is not None
    beyond_scatter = tested and float(p) < GEOLOCATION_SPLIT_ALPHA
    split = dis is not None and float(dis) > max(off, GEOLOCATION_FLAG_M)
    halves_txt = (f"west half {sides['west']:.0f} m, east half {sides['east']:.0f} m"
                  if "west" in sides and "east" in sides else "")
    p_txt = f", split test p = {float(p):.3f}" if tested else ""
    worst = max([off, *sides.values()])
    res["worst_offset_m"] = round(worst, 2)
    within = max(off, float(res.get("step_m") or 0))
    within_txt = int(within + 0.5)                     # "within about": an upper figure, so 2.5 m reads 3
    # the joint offset first: a split between the halves is a reason to doubt
    # a small offset, never to excuse a large one
    if off > GEOLOCATION_FLAG_M or (half_over_cell and beyond_scatter):
        in_half = half_over_cell and beyond_scatter and sides[side] > off
        res["flag"] = True
        res["status"] = "flagged" + (f" ({halves_txt}{p_txt})" if halves_txt else "")
        res["text"] = (f"Map positions are about {worst:.0f} m off mapped water edges"
                       + (f" in the {side} half, more than one {cell:.0f} m map cell, and the two halves of the "
                          "area disagree by more than their own scatter" if in_half else
                          f", more than the {GEOLOCATION_FLAG_M:.0f} m the check allows")
                       + "; a specialist checks the positioning before the map is used at field scale.")
    elif not tested and half_over_cell and split:
        res["flag"] = None
        res["hold"] = True
        res["status"] = (f"inconclusive (the west and east halves disagree by {float(dis):.0f} m"
                         + (f": {halves_txt}" if halves_txt else "")
                         + "; not tested against their own scatter)")
        res["text"] = ("The check of map positions against mapped water edges was inconclusive: the two halves of "
                       f"the area disagree by about {float(dis):.0f} m, so positions are not confirmed.")
    else:
        res["flag"] = False
        n = res.get("n_water_bodies")
        if side is not None and sides[side] > max(within, GEOLOCATION_FLAG_M) and (beyond_scatter or split):
            # a half reads more than the check allows the whole area, but
            # within one cell (or not beyond the halves' own scatter)
            res["within_cell"] = sides[side] <= cell
            how = ("the halves differ beyond their own scatter" if beyond_scatter else
                   "the halves differ by no more than their own scatter" if tested else
                   "the halves' difference was not tested against their scatter")
            res["status"] = ((f"within a cell ({halves_txt}; {how}{p_txt}; the {side} half is within one "
                              f"{cell:.0f} m map cell)") if res["within_cell"] else
                             f"agrees overall ({halves_txt}; {how}{p_txt})")
            res["text"] = (f"Map positions agree with mapped water edges to within about {within_txt} m overall "
                           f"({n} water bodies)"
                           + (f"; the {side} half reads about {sides[side]:.0f} m, less than one {cell:.0f} m map "
                              "cell." if res["within_cell"] else "."))
            if not res["within_cell"]:
                res.pop("within_cell")
        else:
            res["text"] = (f"Map positions agree with mapped water edges to within about {within_txt} m "
                           f"({n} water bodies).")
    return res


def stitch_check(dc: dict, cfg=None) -> dict:
    """
    The DISP-S1 stitching, from disp_chain.json: lost reference-epoch
    anchors, unrepaired datum shifts, whole-cycle steps in excess at the
    boundaries, partly applied corrections and the share of samples
    rejected as unwrapping spikes. Each one biases a rate in a way no other
    check sees; any of them fails the check.
    """
    st = (dc or {}).get("stitch") or {}
    n = st.get("epochs")
    skipped = list(st.get("skipped_granules") or [])
    lost = list(st.get("lost_anchors") or st.get("lost_anchor_granules") or dc.get("lost_anchors") or [])
    datum = st.get("boundary_datum") or {}
    flagged = list(datum.get("flagged") or [])
    repaired = bool(datum.get("repaired"))
    bs = st.get("boundary_steps") or {}
    br, cr, ex = bs.get("boundary_rate"), bs.get("control_rate"), bs.get("excess")
    step_excess = bool(ex is not None and br is not None and ex > 0.005 and br > 1.5 * max(cr or 0.0, 1e-9))
    n_corr = st.get("epochs_fully_corrected")
    want_corr = bool(getattr(cfg, "disp_apply_corrections", bool(st.get("corrections_applied")))) if cfg else \
        bool(st.get("corrections_applied"))
    spikes = (st.get("unwrap_spikes") or {}).get("fraction")
    problems = []
    if lost:
        problems.append(f"{len(lost)} reference-epoch boundary granule(s) could not be read ({', '.join(lost[:4])}): "
                        "every later epoch is offset")
    if flagged and not repaired:
        problems.append(f"{len(flagged)} datum shift(s) at reference-epoch boundaries, not repaired")
    if step_excess:
        problems.append(f"whole-cycle steps at the boundaries exceed ordinary epochs by {100 * ex:.2f}%")
    if want_corr and n and n_corr is not None and n_corr < n:
        problems.append(f"corrections applied to {n_corr} of {n} epochs: the rest sit ~8 mm apart for no "
                        "physical reason" if n_corr else "corrections were requested but no epoch was corrected")
    # provisional: more than 1 sample in 100 rejected as a reverting whole-
    # cycle spike is an unwrapping problem the per-pixel fit cannot absorb
    if spikes is not None and spikes > 0.01:
        problems.append(f"{100 * spikes:.1f}% of samples rejected as whole-cycle unwrapping spikes")
    # disp_backend.stitch_problems (hard ones already stopped the chain; a
    # record that carries one was made before that guard) and the measured
    # records disp_chain adds: coverage ratchet, the ERA5 before/after, the
    # end-hold drop and pixels carrying a whole-cycle error from an anchor
    notes = []
    for pr in st.get("problems") or []:
        if not isinstance(pr, dict):
            continue
        if pr.get("hard"):
            if not any(pr.get("text", "")[:40] in p for p in problems):
                problems.append(str(pr.get("text")))
        else:
            notes.append(str(pr.get("text")))
    if (st.get("coverage_ratchet") or {}).get("fired") and not any("ratchet" in p for p in problems):
        problems.append("coverage ratchet: pixels observed early in the record stop being observed later")
    cells = None
    grid = (dc or {}).get("grid") or {}
    gf = (dc or {}).get("gap_fill") or {}
    if gf.get("pixels_complete") is not None:
        cells = int(gf.get("pixels_complete", 0)) + int(gf.get("pixels_gap_filled", 0))
    elif grid.get("LENGTH") and grid.get("WIDTH"):
        cells = int(grid["LENGTH"]) * int(grid["WIDTH"])
    suspect = (dc or {}).get("anchor_suspect_cells")
    suspect_share = (suspect / cells) if (suspect is not None and cells) else None
    if suspect_share is not None and suspect_share > ANCHOR_SUSPECT_SHARE:
        problems.append(f"{100 * suspect_share:.1f}% of cells carry a suspected whole-cycle error from a reference "
                        "anchor into every later epoch (layer anchor_suspect_count)")
    elif suspect:
        notes.append(f"{suspect} cells ({100 * (suspect_share or 0):.2f}%) carry a suspected whole-cycle anchor error")
    tropo = (dc or {}).get("troposphere") or {}
    if tropo.get("helped") is False:
        notes.append(f"the ERA5 correction made the scatter about trend worse ({tropo.get('scatter_before_mm')} -> "
                     f"{tropo.get('scatter_after_mm')} mm)")
    if gf.get("pixels_dropped_end_hold"):
        notes.append(f"{gf['pixels_dropped_end_hold']} cells left out: most of their record would be held flat at "
                     "the ends")
    return {"epochs": n, "skipped_granules": skipped, "lost_anchors": lost,
            "corrections_status": (st.get("corrections") or {}).get("status"),
            "anchor_suspect_cells": suspect,
            "anchor_suspect_share": round(suspect_share, 5) if suspect_share is not None else None,
            "coverage_ratchet_fired": (st.get("coverage_ratchet") or {}).get("fired"),
            "troposphere_helped": tropo.get("helped"),
            "pixels_dropped_end_hold": gf.get("pixels_dropped_end_hold"),
            "misclosure_vs_rate": (dc or {}).get("misclosure_vs_rate"),
            "notes": notes,
            "datum_shifts_flagged": len(flagged), "datum_shifts_repaired": repaired,
            "boundary_step_excess": ex, "epochs_fully_corrected": n_corr, "spike_fraction": spikes,
            "problems": problems, "pass": (not problems) if st else None,
            "note": ("OPERA DISP-S1 stitching (disp_chain.json): skipped ordinary granules cost one sample each; "
                     "a lost boundary granule, an unrepaired datum shift, a boundary-step excess, partly applied "
                     "corrections or more than 1% spike samples fail the check")}


def _excluded_reasons(cfg) -> dict:
    """{date: reason} for the dates time_series.ensure_connected_network
    excluded (network_connectivity.json): a date with no kept interferogram.
    On the Site A record both (20210219 - the configured stack reference
    date - and 20250105) lost every pair to the reference-pixel check, which
    reference_excluded_pairs.json lists; say so when that is the cause."""
    nc = _load_json(cfg.mintpy_dir / "network_connectivity.json") or {}
    rx = _load_json(cfg.mintpy_dir / "reference_excluded_pairs.json") or {}
    ref_pairs = set(rx.get("excluded_for_reference") or [])
    formed = None
    stack = cfg.mintpy_dir / "inputs" / "ifgramStack.h5"
    if stack.exists() and ref_pairs:
        try:
            import h5py
            with h5py.File(stack, "r") as f:
                formed = ["_".join(x.decode() if isinstance(x, bytes) else str(x) for x in pr) for pr in f["date"][:]]
        except (OSError, KeyError):
            formed = None
    out = {}
    for d in nc.get("excluded_dates") or []:
        d = str(d)
        mine = [p for p in (formed if formed is not None else ref_pairs) if d in p.split("_")]
        # the reference check is the cause only if it took EVERY pair the date had
        out[d] = _EXCLUDED_REF_REASON if (mine and all(p in ref_pairs for p in mine)) else _EXCLUDED_REASON
    return out


# slc/assembly_status.json 'dropped' reason codes (data_acquisition.classify_short_dates)
# stack_exclude_dates (29 Oct 2026): said in the client report too, so no setting names.
_EXCLUDED_FROM_STACK = "it could not be aligned with the other dates, so it was kept out of the processing on purpose"
# Every date_accounting reason reads as a clause after "because": the
# client's methods sentence is "K left out because <reason>"
# (client_report.methods_stages; 30 Oct 2026, the first city site read
# "1 left out because in the interferogram stack but not in the time series").
_PAIRS_DROPPED = "all of its interferograms were dropped before the time series was formed"
# tropo_corrections/<MODEL>/unavailable_dates.json (weather_raider.check_dates):
# the date's pairs are dropped before the inversion, so without this it read
# as _PAIRS_DROPPED. {model}, then the check's own reason in brackets.
_NO_WEATHER = ("no {} weather data was available for the time of this pass, so its weather delay could not be "
               "removed ({})")
# a date the archive serves in a form no attempt could assemble
# (data_acquisition.CATALOGUE_DEFECTS); the defect's name in brackets
_CATALOGUE_DEFECT = ("the satellite archive serves this date in a form that cannot be assembled (a known catalogue "
                     "defect: {})")
_NEVER_ENTERED = "it was assembled or coregistered but never entered the stack - unexplained"
_DISP_SKIPPED = "the DISP-S1 granule could not be read and was skipped at stitching"
_DISP_NO_BASELINE = "there is no orbit-stack perpendicular baseline for this DISP-S1 epoch"
_DROP_REASON = {
    "catalogue_short": "the satellite archive holds only part of the area's bursts on this date, so it could not "
                       "be assembled",
    "catalogue_unreachable": "the archive's burst catalogue could not be reached to check this date, so it was "
                             "left out",
    # a catalogue date the started stack's plan cannot take (accepted with
    # allow_missing_dates, or refused): not short, so never _SEARCH_SHORT.
    # short_date_reasons (the 'dropped' code) and date_accounting (the
    # 'accepted_not_in_plan' list) both quote this entry, so they agree.
    "not_in_plan": "it was found in the archive only after this project's processing had started, so it could "
                   "not be added to the processing plan",
}
# A date the burst SEARCH returned short, with nothing on disk proving the
# archive itself is short (a legacy dropped_short_dates.json, or a code this
# version does not know). On Site B 4 of the 5 such dates (20250517,
# 20251113, 20260605, 20260624) are search-rendering gaps - the plain-JSON
# catalogue holds every burst - so "the burst catalogue held fewer bursts"
# was wrong for them (review outcome check, 6 Oct 2026). Only assembly_status
# 'catalogue_short', set after the plain-JSON check, says the archive is short.
_SEARCH_SHORT = ("the archive search returned it incomplete (fewer bursts than the other dates), so it was not "
                 "assembled")
# the wording the report used before 6 Oct 2026, recognised in an older QA
_LEGACY_SHORT = "the burst catalogue held fewer bursts than the other dates"


def assembly_records(cfg) -> tuple[dict, dict, str]:
    """(assembly_status, dropped_short_dates, where they came from): the
    download's records of every catalogue date on the track and window.
    slc/ first; when cleanup has emptied slc/ (Site A), the copies the
    run's collect_results kept in run_results.json ('assembly_status',
    'dropped_short_dates'), so the image accounting still counts every
    catalogue date (decision E23, 7 Oct 2026: Site A read "271 of 273"
    while its own run record lists 5 more dates the archive search returned
    incomplete - 271 of 278). Each record is taken from slc/ when it is
    there, else from run_results.json; the third value names the files the
    records came from."""
    slc = Path(cfg.project_dir) / "slc"
    rr = None
    recs, src = [], []
    for name, key in (("assembly_status.json", "assembly_status"), ("dropped_short_dates.json", "dropped_short_dates")):
        rec = _load_json(slc / name)
        if isinstance(rec, dict):
            src.append(f"slc/{name}")
        else:
            rr = (_load_json(Path(cfg.project_dir) / "run_results.json") or {}) if rr is None else rr
            rec = rr.get(key) if isinstance(rr.get(key), dict) else None
            if rec is not None:
                src.append(f"run_results.json {key} (the copy collect_results kept; slc/ has no {name})")
        recs.append(rec if isinstance(rec, dict) else {})
    return recs[0], recs[1], (", ".join(src) if src else "no download record on disk")


def short_date_reasons(cfg) -> dict:
    """{date: reason} for every date left out at assembly for being short
    (slc/dropped_short_dates.json, slc/assembly_status.json, or their copies
    in run_results.json once cleanup has emptied slc/ - assembly_records):
    the archive is said to be short only where assembly_status records the
    plain-JSON check's verdict ('catalogue_short'); otherwise the search
    returned it incomplete (_SEARCH_SHORT)."""
    asm, ds, _ = assembly_records(cfg)
    drop = asm.get("dropped") or {}
    codes = dict(drop.items()) if isinstance(drop, dict) else {str(x): "" for x in drop}
    out = {}
    for d, code in codes.items():
        out[str(d)] = _DROP_REASON.get(str(code), _SEARCH_SHORT)
    for d in asm.get("dropped_short") or []:
        out.setdefault(str(d), _SEARCH_SHORT)
    for d in ds.get("dropped") or []:
        out.setdefault(str(d), _SEARCH_SHORT)
    return out


def refresh_drop_reasons(cfg, rec: dict | None) -> dict | None:
    """A QA 'dates' record with the short-date reasons in the current wording
    (short_date_reasons), so a report rebuilt from an older QA does not keep
    'the burst catalogue held fewer bursts' for search-rendering gaps. Any
    '(accepted as missing ...)' suffix is kept; other reasons are untouched."""
    if not isinstance(rec, dict) or not rec.get("dropped"):
        return rec
    now = short_date_reasons(cfg)
    out = dict(rec)
    dropped = {}
    for d, why in (rec.get("dropped") or {}).items():
        why = str(why)
        if d in now and (why.startswith(_LEGACY_SHORT) or why.startswith(_SEARCH_SHORT[:30])
                         or why.startswith(_DROP_REASON["catalogue_short"][:30])):
            tail = why[why.index(" (accepted as missing"):] if " (accepted as missing" in why else ""
            why = now[d] + tail
        dropped[d] = why
    out["dropped"] = dropped
    return out


def _failed_reason(why) -> str:
    """An assembly_status 'failed' entry as a date_accounting reason: a known
    catalogue defect in plain words, any other failure with its detail."""
    from .data_acquisition import DEFECT_PREFIX
    why = str(why or "")
    if why.startswith(DEFECT_PREFIX):
        return _CATALOGUE_DEFECT.format(why[len(DEFECT_PREFIX):][:160])
    return f"burst assembly failed{': ' + why[:160] if why else ''}"


def date_accounting(cfg, processed) -> dict:
    """
    Every acquisition the run knew about and what became of it: processed
    (in the delivered series) or dropped, with the reason. A date that was
    assembled or coregistered and never reached the series is UNEXPLAINED
    - the 20161206 defect - and holds the report.
    """
    import re
    proc = sorted({str(d) for d in processed})
    dropped, accepted = {}, set()

    def add(d, why):
        d = str(d)
        if d not in proc:
            dropped.setdefault(d, why)
    M = cfg.mintpy_dir
    # first: with no weather data the date is taken out of the series before
    # the network is checked, so this is the cause whatever else is said of it
    from . import weather_raider as wr
    for d, why in wr.registered_unavailable(cfg).items():
        add(d, _NO_WEATHER.format(wr.model_name(cfg), why))
    for d, why in _excluded_reasons(cfg).items():
        add(d, why)
    for d in sorted({str(v).strip().replace("-", "") for v in getattr(cfg, "stack_exclude_dates", None) or ()}):
        add(d, _EXCLUDED_FROM_STACK)
    stack = M / "inputs" / "ifgramStack.h5"
    if stack.exists():
        try:
            import h5py
            with h5py.File(stack, "r") as f:
                sd = {x.decode() if isinstance(x, bytes) else str(x) for pr in f["date"][:] for x in pr}
            for d in sd:
                add(d, _PAIRS_DROPPED)
        except (OSError, KeyError):
            pass
    dc = _load_json(M / "disp_chain.json") or {}
    for d in (dc.get("stitch") or {}).get("skipped_granules") or []:
        add(str(d)[:8], _DISP_SKIPPED)
    for d in dc.get("epochs_dropped_no_baseline") or []:
        add(str(d)[:8], _DISP_NO_BASELINE)
    slc = cfg.project_dir / "slc"
    asm, _, rec_src = assembly_records(cfg)
    # A download record written before 30 Oct 2026 relabelled an accepted
    # catalogue-defect date 'not_in_plan' on the run after it failed; the
    # config's auto-accept note still says why (the first city site).
    from .data_acquisition import auto_accepted_in_config
    codes = asm.get("dropped") if isinstance(asm.get("dropped"), dict) else {}
    late = {str(d) for d, c in codes.items() if c == "not_in_plan"} | {
        str(x) for x in asm.get("accepted_not_in_plan") or []}
    for d, name in auto_accepted_in_config(cfg).items():
        if d in late:
            add(d, _CATALOGUE_DEFECT.format(name))
    # data_acquisition.download_bursts_as_safes writes assembly_status
    # 'dropped' {date: reason code}; 'dropped_short' is read too for any record
    # written by hand; dropped_short_dates.json is the legacy record. The
    # archive is said to be short only where assembly_status proves it.
    for d, why in short_date_reasons(cfg).items():
        add(d, why)
    failed = asm.get("failed") or {}
    for d in (failed.items() if isinstance(failed, dict) else ((x, "") for x in failed)):
        add(d[0], _failed_reason(d[1]))
    for key in ("accepted_missing", "allowed_missing", "allow_missing_dates", "accepted_failures"):
        accepted |= {str(x) for x in (asm.get(key) or []) if str(x) != "all"}
    # dates beyond the processing plan the config accepts are taken out
    # before download (data_acquisition 'accepted_not_in_plan'), so no other
    # record names them; the config's own list counts too
    beyond_plan = {str(x) for x in asm.get("accepted_not_in_plan") or []}
    amd = getattr(cfg, "allow_missing_dates", ()) or ()
    if isinstance(amd, str):
        amd = amd.replace(",", " ").split()
    if not isinstance(amd, (list, tuple, set)):
        amd = ()                                    # True ("all") names no date
    accepted |= beyond_plan | {x for x in (str(v).strip().replace("-", "") for v in amd)
                               if x.isdigit() and len(x) == 8}
    available = set(proc) | set(dropped)
    catalogue = set()
    for key in ("searched", "available", "catalogue_dates", "catalogue"):
        if isinstance(asm.get(key), list):
            catalogue |= {str(x) for x in asm[key]}
    available |= catalogue
    s0 = str(getattr(cfg, "start_date", "") or "").replace("-", "") or "00000000"
    s1 = str(getattr(cfg, "end_date", "") or "").replace("-", "") or "99999999"
    # on disk but never in the series and not accounted for above
    unexplained = []
    on_disk = set()
    if slc.exists():
        for p in slc.iterdir():
            m = re.search(r"_(\d{8})T\d{6}_", p.name)
            if m and p.name.endswith((".SAFE", ".zip")):
                on_disk.add(m.group(1))
    coreg = cfg.stack_dir / "coreg_secondarys" if hasattr(cfg, "stack_dir") else None
    if coreg is not None and coreg.exists():
        for p in coreg.iterdir():
            m = re.fullmatch(r"(\d{8})(\.cleaned)?", p.name)
            if m:
                on_disk.add(m.group(1))
    for d in sorted(on_disk):
        if s0 <= d <= s1 and d not in proc and d not in dropped:
            add(d, _NEVER_ENTERED)
            unexplained.append(d)
            available.add(d)
    # in the archive catalogue but neither processed, left out with a reason
    # nor found on disk: named, so "N of M images used; K left out" adds up,
    # and held unless the config accepted the date as missing
    for d in sorted(catalogue - set(proc) - set(dropped)):
        if not s0 <= d <= s1:
            add(d, "it falls outside the configured date range, so it was not processed")
        elif d in beyond_plan:
            add(d, _DROP_REASON["not_in_plan"])
        elif d in accepted:
            add(d, "it is in the archive catalogue but was not downloaded, so it was left out")
        else:
            add(d, "it is in the archive catalogue but was never assembled and no reason was recorded - unexplained")
            unexplained.append(d)
    for d in accepted & set(dropped):
        if dropped[d] != _EXCLUDED_FROM_STACK:
            dropped[d] += " (accepted as missing by allow_missing_dates)"
    ref = str(getattr(cfg, "stack_reference_date", "") or "")
    ref_excl = bool(ref and ref in dropped)
    return {"available": len(available), "processed": len(proc), "dropped": dict(sorted(dropped.items())),
            "unexplained": sorted(unexplained), "reference_date_excluded": ref_excl,
            "download_record": rec_src,
            "text": f"{len(proc)} of {len(available)} images used" + (f"; {len(dropped)} left out (see dropped)" if dropped else ""),
            "flag": bool(unexplained or ref_excl), "hold": bool(unexplained),
            "note": ("available = every acquisition the run recorded (processed, dropped with a reason, in the "
                     "archive catalogue, or found on disk); an acquisition assembled or coregistered, or in the "
                     "catalogue and not accepted as missing by allow_missing_dates, but absent from the series "
                     "without a recorded reason is unexplained and holds the report. reference_date_excluded: the "
                     "configured stack_reference_date itself was left out of the inversion.")}


def _crashed(b) -> str | None:
    """The error of a diagnostic evaluate() recorded as raised ("could not
    run: ..."), else None - a 'not applicable' (no layer, no axes) is not one."""
    r = str((b or {}).get("reason") or "") if isinstance(b, dict) else ""
    if not r.startswith("could not run"):
        return None
    return r[len("could not run"):].lstrip(": ") or "no reason recorded"


# ------------------------------------------------------------------ site profile (V3, 8 Oct 2026)
# whole word, and not a path part ("/data/farmland/x", "farmland.geojson"
# inside a sentence): a file named in a reason must still name that file
_FARMLAND_WORD = re.compile(r"(?<![/\\.])\bfarmland\b(?![/\\]|\.\w)")
_FARMLAND_WORD_CAP = re.compile(r"(?<![/\\.])\bFarmland\b(?![/\\]|\.\w)")
# a string with no space that holds a slash is a path or a URL, never prose
_PATH_LIKE = re.compile(r"^\S*[/\\]\S*$")


def reword_ground(text, ground: str):
    """`text` with the comparison ground called `ground` where the farmland
    wording says "farmland" (whole word: an identifier such as
    farmland_spread_1sigma_mm_yr is left alone). Unchanged when the ground is
    itself some kind of farmland, when `text` is not a string, and for paths."""
    if not isinstance(text, str) or not ground or "farmland" in ground.lower() or _PATH_LIKE.match(text):
        return text
    # a function, not a replacement string: a backslash in the label is text
    text = _FARMLAND_WORD.sub(lambda m: ground, text)
    cap = ground[:1].upper() + ground[1:]
    return _FARMLAND_WORD_CAP.sub(lambda m: cap, text)


def _reword_record(obj, ground: str):
    """Every string VALUE in a QA record reworded for the ground (keys, which
    other modules read, are never changed)."""
    if isinstance(obj, dict):
        return {k: _reword_record(v, ground) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_reword_record(v, ground) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_reword_record(v, ground) for v in obj)
    return reword_ground(obj, ground)


def site_profile_of(cfg, measured: dict | None = None) -> tuple:
    """(profile name, comparison-ground words) for a config: the name from
    config.site_profile ("farmland" when unset). The words: the config's
    control_ground_label when someone set it; else the measured land cover
    (`measured`, landcover.comparison_ground: "farmland" where cropland
    dominates, "ground (mostly built-up and grassland)" elsewhere, "ground"
    without a land-cover map; 28 Oct 2026 - until then the farmland profile
    said "farmland" whatever the ground was); without `measured`,
    "farmland" on the farmland profile and NEUTRAL_GROUND on the others."""
    from insar_pipeline.config import site_profile
    from insar_pipeline.landcover import configured_ground_label
    name = site_profile(cfg)
    if name not in SITE_PROFILES:
        raise ValueError(f"site_profile {name!r} is not one of {list(SITE_PROFILES)}")
    own = configured_ground_label(cfg)
    if own:
        return name, own
    if (measured or {}).get("label"):
        return name, str(measured["label"])
    return name, "farmland" if name == "farmland" else NEUTRAL_GROUND


def _noun(ground: str) -> str:
    """The words without their land-cover note, for rewording prose
    (report_insights.ground_noun)."""
    from insar_pipeline.report_insights import ground_noun
    return ground_noun(ground)


def _profile_of_record(qa: dict) -> tuple:
    """(name, profile table entry, ground word) from a QA record. A record
    without 'site_profile' is farmland: every record before 8 Oct 2026, and
    every farmland record since (the farmland record is unchanged)."""
    rec = (qa or {}).get("site_profile")
    if not isinstance(rec, dict) or not rec.get("name") or rec.get("name") == "farmland":
        # the farmland profile's words: what annotate_site recorded (the
        # measured ground, 28 Oct 2026), "farmland" on an older record
        cg = (qa or {}).get("comparison_ground") or {}
        return "farmland", SITE_PROFILES["farmland"], str(cg.get("words") or "farmland")
    name = str(rec["name"])
    prof = SITE_PROFILES.get(name) or {"where": rec.get("where") or f"at this site ({name})", "expected": {},
                                       "handled": False, "whole_area_moves": bool(rec.get("whole_area_moves"))}
    return name, prof, str(rec.get("ground") or NEUTRAL_GROUND)


def annotate_site(cfg, out: dict) -> dict:
    """Read the checks for the site's kind of ground (config site_profile)
    and record what the run is for (config purpose). Returns the record.
    farmland (the default): each check gets on_farmland_expected and its
    plain words exactly as before 8 Oct 2026, and the record carries no
    'site_profile' (absent means farmland). Any other profile:
      - every prose value in the record says the config's comparison
        ground (control_ground_label) instead of "farmland";
      - each check gets expected_on_site (the ground explains a flag on
        it), expected_handled (the figures' way of coping has been checked
        on this ground; False on every profile but farmland) and, when
        expected, expected_why - and NO on_farmland_expected, which would
        say farmland where there is none;
      - out['site_profile'] records the profile for send_rule, which reads
        it again when the report stage re-runs the rule.
    purpose experiment adds out['purpose'] = 'experiment' (absent = a
    client project)."""
    from insar_pipeline.config import is_experiment
    from insar_pipeline.landcover import configured_ground_label
    name, ground = site_profile_of(cfg, out.get("comparison_ground"))
    prof = SITE_PROFILES[name]
    if isinstance(out.get("comparison_ground"), dict):
        # the words every prose value below uses (send_rule reads them again)
        out["comparison_ground"]["words"] = ground
    if name == "farmland":
        for check, c in out["checks"].items():
            c["on_farmland_expected"] = EXPECTED_ON_FARMLAND.get(check, False)
            c["plain"] = PLAIN.get(check, "")
        # the farmland profile on ground that is not farmland (the first combined site, 28
        # Oct 2026): the expectations stay the profile's, the prose says the
        # ground it was measured on
        out = _reword_record(out, _noun(ground))
    else:
        out = _reword_record(out, _noun(ground))
        for check, c in out["checks"].items():
            why = prof["expected"].get(check)
            c["expected_on_site"] = why is not None
            c["expected_handled"] = bool(why is not None and prof["handled"])
            if why is not None:
                c["expected_why"] = why
            c["plain"] = reword_ground(PLAIN.get(check, ""), _noun(ground))
        note = (f"the checks are read for a run {prof['where']} (config site_profile: {name}). "
                + ("A flag the ground explains is named as such, but it still holds the send rule: the figures' "
                   "way of coping with it (class matching, the null, relative rates) has only been checked on "
                   "farmland. " if prof["expected"] and not prof["handled"] else "")
                + ("No flag is explained by this ground: each one points at the processing. "
                   if not prof["expected"] else "")
                + (WHOLE_AREA_HOLD.format(where=prof["where"]) + ". " if prof["whole_area_moves"] else "")
                + f"Figures are measured against {ground} ("
                + ("its measured land cover" if (out.get("comparison_ground") or {}).get("source") == "measured"
                   and not configured_ground_label(cfg) else "config control_ground_label") + ").")
        out["site_profile"] = {"name": name, "where": prof["where"], "ground": ground,
                               "expected": [c for c in CHECKS if c in prof["expected"]],
                               "handled": bool(prof["handled"]), "whole_area_moves": bool(prof["whole_area_moves"]),
                               "note": note}
    if is_experiment(cfg):
        out["purpose"] = "experiment"
    return out


def check_counts(qa: dict | None) -> dict:
    """
    The checks of a QA record counted as the verdict counts them (decision
    D2, 7 Oct 2026): a check that cannot apply here (`not_applicable` with
    no pass, e.g. the distributed-scatterer bias when the far field holds
    one coherence class) is named and never counted - not as a pass, not
    as flagged, not as untested. A check that applies but was not run
    (pass None, no `not_applicable`) stays untested.
      passed, flagged, untested   counts among the applicable checks
      applicable                  passed + flagged + untested
      not_applicable              the names of the checks that do not apply
      text                        the sentence every reader quotes, e.g.
                                  "2 of 5 applicable checks flagged; 1 not
                                  applicable (distributed scatterer bias)"
      head                        its first part, up to the ';' (tables)
    Read from `checks`, so an older record is counted the same way.
    """
    checks = {n: c for n, c in ((qa or {}).get("checks") or {}).items() if isinstance(c, dict)}
    na = [n for n, c in checks.items() if c.get("pass") is None and c.get("not_applicable")]
    # numpy booleans count as booleans; anything else is not a result
    res = [c.get("pass") for n, c in checks.items() if n not in na]
    res = [bool(p) if isinstance(p, (bool, np.bool_)) else None for p in res]
    passed, flagged = res.count(True), res.count(False)
    untested = res.count(None)
    head = f"{flagged} of {len(res)} applicable check{'' if len(res) == 1 else 's'} flagged" \
        + (f", {untested} could not be run" if untested else "")
    tail = f"; {len(na)} not applicable ({', '.join(n.replace('_', ' ') for n in na)})" if na else ""
    return {"passed": passed, "flagged": flagged, "untested": untested, "applicable": len(res),
            "not_applicable": na, "text": head + tail, "head": head}


def verdict(counts: dict, thin: bool = False) -> str:
    """The overall verdict from check_counts: MEETS PRACTICE when every
    applicable check passed, USABLE WITH LIMITATIONS with at most two
    flagged otherwise (an untested check is a limitation), REVIEW with three
    or more. A check that does not apply is not counted, so MEETS PRACTICE
    is reachable without it; with no applicable check at all nothing was
    checked (REVIEW). Thin farmland is REVIEW whatever the checks say:
    nothing they passed makes the figures testable (stats-2)."""
    if thin or counts["applicable"] == 0:
        return "REVIEW"
    if counts["flagged"] == 0 and counts["untested"] == 0:
        return "MEETS PRACTICE"
    return "USABLE WITH LIMITATIONS" if counts["flagged"] <= 2 else "REVIEW"


def send_rule(qa: dict, gnss: dict | None = None) -> dict:
    """
    Can the client report be sent? Keyed on WHICH checks are flagged, not
    on how many: on farmland the reference trend, rate precision and DS
    bias flags are expected and handled (class matching, the farmland
    null), so a product carrying only those is "send with notes" - the
    Site A record. Anything else - an unexpected flag, an untested
    check, too little farmland to test against, an unexplained missing
    date, a geolocation offset (or one half more than one map cell off
    beyond the halves' own scatter, E22), a figure that moves more than its
    null sigma with the
    deramp or the matching, a diagnostic that could show that but raised
    instead of running, or (when `gnss` is given, after gnss_tie has run) a
    GNSS tie that failed - is "hold for specialist". A check that does not
    apply (`not_applicable`), a note-only diagnostic that raised, and an
    unwrap-error correction step that was asked for and made nothing
    (`unwrap_correction`) are notes.
    `gnss` is the gnss_tie.json record; the QA runs before the tie, so the
    value stored in quality_assessment.json has gnss_considered False and
    the report stage re-evaluates it with the tie.
    Which flags count as expected comes from the record's site profile
    (annotate_site; none recorded = farmland, the rule above). On any other
    profile a flag the ground explains is said as such but still holds,
    because the figures' way of coping has only been checked on farmland,
    and a profile whose whole area moves holds whatever the checks say.
    A record from an experiment run (purpose 'experiment') carries
    'experiment': nothing is sent from it, and the value is kept only to
    compare with client projects.
    """
    hold, notes = [], []
    prof_name, prof, ground = _profile_of_record(qa)
    farm = prof_name == "farmland"

    def g(text):
        return reword_ground(text, _noun(ground))
    for name, c in (qa.get("checks") or {}).items():
        p = c.get("pass")
        label = name.replace("_", " ")
        if p is None and c.get("not_applicable"):
            notes.append(f"the {label} check does not apply: {c['not_applicable']}")
        elif p is None:
            hold.append(f"the {label} check could not be run")
        elif p is False and farm:
            exp = c.get("on_farmland_expected", EXPECTED_ON_FARMLAND.get(name, False))
            (notes if exp else hold).append(
                f"{label} flagged" + (" (expected on farmland; the figures are built to cope with it)" if exp else ""))
        elif p is False:
            why = c.get("expected_why") or prof["expected"].get(name)
            exp = bool(c.get("expected_on_site", name in prof["expected"]))
            if exp and c.get("expected_handled", bool(prof["handled"])):
                notes.append(f"{label} flagged (expected {prof['where']}; the figures are built to cope with it)")
            elif exp:
                hold.append(f"{label} flagged (expected {prof['where']}" + (f": {why}" if why else "")
                            + "; it still holds the report because the figures' way of coping with it has only "
                            "been checked on farmland)")
            else:
                hold.append(f"{label} flagged")
    if not farm and prof.get("whole_area_moves"):
        hold.append(WHOLE_AREA_HOLD.format(where=prof["where"]))
    cov = (qa.get("checks") or {}).get("coverage") or {}
    nl = cov.get("noise_level") or {}
    if cov.get("threshold_above_noise") is False:
        hold.append(f"the coherence threshold ({cov.get('threshold')}) is inside the noise floor "
                    f"({nl.get('floor')}): coverage does not mean reliability")
    elif (cov.get("threshold_above_noise") is None and nl.get("measured") is False and nl.get("floor") is None
          and "phase linking" in str(cov.get("coherence_source", "")).lower()):
        # a phase-linking coherence does not fall to zero on noise: an
        # unmeasured floor leaves the coverage pass untested, not passed
        notes.append(f"the coherence threshold ({cov.get('threshold')}) was not checked against a measured noise "
                     f"floor ({nl.get('population') or 'no noise population'}): coverage may count noise as "
                     "measured ground")
    rel = cov.get("reliable_percent")
    if rel is not None and rel < YARDSTICK["coverage_percent"]:
        notes.append(f"only {rel:.0f}% of the area passes both quality checks (the rest is shown but less certain)")
    sg_eff = (cov.get("second_gate") or {}).get("effect_on_figures") or {}
    if sg_eff.get("flag"):
        notes.append(f"a figure moves by {sg_eff.get('worst_change_in_sigma')} null sigma when gated on unwrapping "
                     "consistency")
    ff = qa.get("far_field") or {}
    if ff.get("thin"):
        hold.append(g("too little ordinary farmland to test the report areas against"))
    dt = qa.get("dates") or {}
    if dt.get("unexplained"):
        hold.append(f"{len(dt['unexplained'])} acquisition(s) never reached the stack without a reason: "
                    + ", ".join(dt["unexplained"][:6]))
    if dt.get("reference_date_excluded"):
        notes.append("the configured stack reference date was left out of the inversion")
    if dt.get("dropped"):
        notes.append(dt.get("text", ""))
    uc = qa.get("unwrap_correction") or {}
    if uc.get("flag"):
        # a recorded limitation, not a hold: the closure correction was
        # reviewed on 27 Sep 2026 and helps coherent ground but can hurt
        # farmland (21 Oct 2026)
        from insar_pipeline.time_series import unwrap_error_skip_text
        notes.append(unwrap_error_skip_text(uc) + " (see unwrap_correction)")
    geo = qa.get("geolocation") or {}
    if geo.get("flag"):
        hold.append(f"map positions are {geo.get('worst_offset_m') or geo.get('offset_m')} m off mapped water "
                    "edges")
    elif geo.get("hold"):
        hold.append(f"geolocation inconclusive with one half {geo.get('worst_offset_m')} m off mapped water edges, "
                    f"more than one {float(geo.get('cell_m') or GEOLOCATION_CELL_M):.0f} m map cell, and the halves "
                    f"not tested against their own scatter: {geo.get('status')}")
    elif geo.get("flag") is None:
        notes.append("geolocation was not checked against mapped water" if not geo.get("status")
                     else f"geolocation check {geo['status']}")
    elif geo.get("within_cell"):
        notes.append(f"geolocation {geo.get('status')}")
    for key, label in (("ramp_sensitivity", "the deramp choice"), ("matching_sensitivity", g("the farmland matching")),
                       ("plausibility", "too-fast cells"), ("dem_error", "DEM height errors near mine edges")):
        b = qa.get(key) or {}
        why = _crashed(b)
        if why is not None:
            # evaluate records a diagnostic that raised as not run: one that
            # can hold the report holds it until it has run
            hold.append(f"the {key.replace('_', ' ')} check could not run ({why})")
        elif b.get("hold") or (key == "matching_sensitivity" and b.get("flag")):
            hold.append(f"{label}: " + g("a reported figure depends on it beyond its farmland null")
                        if key != "plausibility" else "more than 5% of a report area changes too fast to measure")
        elif b.get("flag"):
            notes.append(f"{label} flagged (see {key})")
    for name, a in (qa.get("area_testability") or {}).items():
        if a.get("kind") == "area" and a.get("testable") is False:
            hold.append(f"report area '{name}' " + g("cannot be compared with farmland") + ": "
                        f"{a.get('not_testable_reason') or 'too little ordinary ground'}")
        elif a.get("borrowed_classes"):
            notes.append(f"'{name}': {a.get('cells_in_borrowed_classes_percent')}% of its cells are in coherence "
                         + g("classes compared with a neighbouring class's farmland")
                         + f" ({', '.join(a['borrowed_classes'])})")
    st = (qa.get("checks") or {}).get("stitch") or {}
    notes.extend(x for x in st.get("notes") or [] if x)
    for key, label in (("resolution_check", "neighbouring values share errors over a longer distance than stated"),
                       ("reference_distance", "the reference pixel is within the far-field buffer of a mapped mine")):
        if (qa.get(key) or {}).get("flag"):
            notes.append(f"{label} (see {key})")
    dist = ((qa.get("checks") or {}).get("coverage") or {}).get("disturbed") or {}
    if dist.get("flag"):
        notes.append(f"a figure moves by {dist.get('worst_change_in_sigma')} null sigma when disturbed-ground cells "
                     "are left out (see coverage.disturbed)")
    # the note-only diagnostics that raised: a note, never silence
    for key, b in (("resolution_check", qa.get("resolution_check")),
                   ("reference_distance", qa.get("reference_distance")),
                   ("coverage.second_gate.effect_on_figures", sg_eff), ("coverage.disturbed", dist),
                   ("unwrap_correction", uc)):
        why = _crashed(b)
        if why is not None:
            notes.append(f"{key} could not run ({why})")
    rs = qa.get("ramp_sensitivity") or {}
    if rs.get("removed_gradient_flag"):
        notes.append("the tilt the deramp removed is more than half the smallest detection limit per 10 km "
                     "(see ramp_sensitivity.removed_ramp)")
    elif rs.get("removed_gradient_flag", False) is None and _crashed(rs) is None and rs.get("available", True):
        notes.append("the tilt the deramp removed was not checked against the detection limits ("
                     + str(rs.get("removed_gradient_status") or "not measured") + ")")
    mp = (qa.get("findings") or {}).get("monitoring_point") or {}
    cal = mp.get("standout_calibration") or {}
    if calibrated_standouts(qa) and cal.get("copies_match_within_band") is False:
        # a stand-out read against copies that do not vary like the farmland at
        # the scale the rule reads: said, never gated on (E7)
        off = ", ".join(f"{k} km {v.get('ratio')}" for k, v in (cal.get("copies_match") or {}).items()
                        if v.get("ratio") is None or not STANDOUT_MATCH[0] <= v["ratio"] <= STANDOUT_MATCH[1])
        notes.append(g("the multi-scale stand-out's simulated farmland does not vary like the real farmland at "
                       f"every scale (window-spread ratio {off}; band {STANDOUT_MATCH[0]}-{STANDOUT_MATCH[1]}): "
                       "read the stand-out with that in mind (see monitoring_point.standout_calibration); the "
                       "copies are built from farmland that still holds the stand-out's own ground, and the client "
                       "report calls a feature holding it no more than 'worth a look'"))
    if gnss is not None:
        if (gnss.get("background") is None) and gnss.get("reason"):
            hold.append(f"no GNSS anchor: {gnss.get('reason')}")
    if farm and "farmland" not in ground.lower() and (qa.get("comparison_ground") or {}).get("source") == "measured":
        # measured land cover against the profile (28 Oct 2026): a note, not a hold
        notes.append(f"the site profile is farmland but the comparison ground is {ground}: the flags it treats as "
                     "expected were judged on farmland (config site_profile)")
    value = SEND_VALUES[2] if hold else SEND_VALUES[1] if notes else SEND_VALUES[0]
    if farm:
        rule = RULE_HOLD + "; send with notes: only farmland-expected flags and recorded limitations; send: nothing flagged"
    else:
        also = ((["a flag the ground explains, until the figures' way of coping with it has been checked here"]
                 if prof["expected"] and not prof["handled"] else [])
                + (["the whole area moving with the signal"] if prof.get("whole_area_moves") else []))
        rule = (g(RULE_HOLD) + (f"; on this site profile ({prof_name}), also " + " and ".join(also) if also else "")
                + "; send with notes: " + (f"only flags expected {prof['where']} and recorded limitations"
                                           if prof["expected"] and prof["handled"] else "only recorded limitations")
                + "; send: nothing flagged")
    out = {"value": value, "reasons": hold + notes, "hold_reasons": hold, "notes": notes,
           "gnss_considered": gnss is not None, "rule": rule}
    if not farm:
        out["site_profile"] = prof_name
    if qa.get("purpose") == "experiment":
        out["experiment"] = EXPERIMENT_SEND_NOTE
    return out


def format_report(r) -> str:
    lines = ["Quality assessment against field practice", ""]
    prof_name, prof, ground = _profile_of_record(r)
    head = []
    if r.get("purpose") == "experiment":
        head.append("  EXPERIMENT RUN: no client report is built from this record; the experiment's result is the "
                    "validation against ground truth, not this verdict")
    if prof_name != "farmland":
        head.append(f"  site profile: {prof_name} ({prof['where']}); figures measured against {ground}")
    if head:
        lines[1:1] = head
    cnt = check_counts(r)
    for name, c in r["checks"].items():
        # N/A only for a check that cannot apply; one that applies but was
        # not run is said as such (D2, 7 Oct 2026)
        mark = ("N/A " if name in cnt["not_applicable"] else
                {True: "OK  ", False: "FLAG"}.get(c.get("pass"), "NOT RUN"))
        keys = [k for k in c if k not in ("pass", "note", "plain", "sensitivity", "velocity_by_coherence_class",
                                          "stable_ground_definition", "orphaned_dates", "expected_why")
                and not isinstance(c[k], (dict, list))]
        lines.append(f"  [{mark}] {name}: " + ", ".join(f"{k}={c[k]}" for k in keys))
    lines.append("")
    for name, f in r["findings"].items():
        lines.append(f"  {name}: " + ", ".join(f"{k}={v}" for k, v in f.items()
                                               if k not in ("note", "pre_epochs", "post_epochs", "standout_calibration")
                                               and not isinstance(v, list)))
    cal = ((r.get("findings") or {}).get("monitoring_point") or {}).get("standout_calibration")
    if isinstance(cal, dict):
        lines.append("  multi-scale stand-out: " + (
            reword_ground(f"gap threshold {cal.get('gap_threshold')} (95th percentile of the largest gap over the "
                          f"scales on {cal.get('n_copies')} farmland copies); copies' window spread against the "
                          "farmland's: ", ground)
            + ", ".join(f"{k} km {v.get('ratio')}" for k, v in (cal.get("copies_match") or {}).items())
            if cal.get("tested") else f"not calibrated ({cal.get('reason')})"))
    lines += ["", f"  {cnt['passed']} of {cnt['applicable']} applicable checks meet practice, {cnt['flagged']} flagged, "
                  f"{cnt['untested']} untested"
              + (f"; not applicable: {', '.join(n.replace('_', ' ') for n in cnt['not_applicable'])}"
                 if cnt["not_applicable"] else "") + f" -> {r['verdict']}"]
    for key in ("far_field", "dates", "ramp_sensitivity", "matching_sensitivity", "plausibility", "dem_error",
                "geolocation", "unwrap_correction"):
        b = r.get(key) or {}
        if not b:
            continue
        mark = {True: "FLAG", False: "OK  ", None: "N/A "}[b.get("flag", b.get("thin"))]
        keys = [k for k in b if not isinstance(b[k], (dict, list)) and k not in ("note", "rule", "reason", "flag")]
        lines.append(f"  [{mark}] {key}: " + ", ".join(f"{k}={b[k]}" for k in keys[:10])
                     + (f" ({b['reason']})" if b.get("reason") else ""))
    sr = r.get("send_rule")
    if sr:
        lines += ["", f"  send rule: {sr['value'].upper()}"] + [f"    - {x}" for x in sr["reasons"]]
    return "\n".join(lines)


def write(cfg, r) -> Path:
    """quality_assessment.json, then the verdict, send rule and coverage
    judgement into deliverables/summary.json (deliverables.refresh_summary) -
    from every caller, the CLI included, so summary.json never keeps a
    stand-alone 'Good' beside a flagged coverage check. A summary that cannot
    be refreshed is logged, never raised."""
    from insar_pipeline import deliverables as dl
    out = cfg.project_dir / "quality_assessment.json"
    # 'software': the code that made the assessment (16 Oct 2026, readiness
    # review OUT-01); the caller's dict is left as it was
    out.write_text(json.dumps({**r, "software": dl._software()}, indent=2, default=str))
    try:
        dl.refresh_summary(cfg)
    except Exception as exc:                                  # noqa: BLE001 - the QA itself is written
        log.warning("summary.json not refreshed after the QA: %s", exc)
    return out


def main():
    from insar_pipeline.config import load_config
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config(sys.argv[1])
    # --no-fetch: no Overpass request for the geolocation water map
    r = evaluate(cfg, fetch_water="--no-fetch" not in sys.argv[2:])
    print(format_report(r))
    print(f"\nwrote {write(cfg, r)}")


if __name__ == "__main__":
    main()
