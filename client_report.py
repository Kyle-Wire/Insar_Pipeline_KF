"""
The client report: ground movement for the areas the client asked about, in
plain language.

    python -m insar_pipeline.client_report <config.yaml> [--out report.html] [--out-dir preview/]

Reads the delivered products (panel_history._Context for the class-matched
fields and series, quality_assessment.json, summary.json, gnss_tie.json,
uncertainty_model.json) and writes deliverables/client_report.{json,html}
plus three figures. This module chooses what a non-specialist needs and says
it plainly:

  * VERTICAL movement (line-of-sight / its up-component; subsidence is read
    as vertical), in inches first and millimetres second;
  * anchored to GNSS when THIS record's tie has a background
    (gnss_tie.check): the area's movement relative to the surrounding
    farmland plus the regional motion the stations measure over the same
    dates. Without one every figure is relative and the study area as a
    whole has no figure at all (`relative_only`): its movement relative to
    its own farmland is ~0 by construction;
  * one definition for every area, the whole study area included: rate =
    relative + background, total = the MEASURED change over the record
    (significance.record_change: the bending curve only when its change of
    pace beats farmland copies of the same footprint, else rate x span -
    measured_change, E10; rate x span is kept as `total_at_average_rate_mm`),
    the series drawn zeroed on the fitted curve so it ends at the total;
    every total and every untested figure with its 95% range (E9);
  * the comparison ground: ordinary farmland outside the mapped mines, with
    the report areas other than the whole study area (E28) and the features
    that beat chance (the second pass) cut out of it;
  * the farmland test (significance.assess, one rule): detected <=> the
    excess over the farmland null passes its empirical 95% threshold, and
    signal_to_noise = |excess| / threshold (>= 1 <=> detected). The
    detection limit is the 80%-power limit for sinking (E5): "sustained
    sinking of X or more across this area would be detected 4 times in 5" -
    the 2 sigma that was printed until 5 Oct 2026 is caught only about half
    the time. A tested
    area's 95% range is on the same frame (excess +/- threshold, + GNSS when
    anchored; tested_range, 7 Oct 2026), so a relative-only area's range
    excludes 0 exactly when it is detected, and a direction word follows;
  * an area is TESTABLE only when the null exists and rests on at least 5
    independent placements AND at least half the outline was measured with
    no side left unmeasured. Otherwise it reads "Not testable" with the
    reason, no range and no detection limit - never "within normal ground
    variation", never a zero-width range;
  * features found by searching the whole area are kept only when they beat
    what simulated ordinary farmland of the same area produces under the
    same search (report_insights.feature_null / selection_p); fail closed;
    the first pass's strongest candidates are also tested with themselves
    left out of the farmland, and one that passes is cut (F1,
    _leave_candidate_out); "stands out clearly" only when the QA's
    calibrated stand-out agrees and the copies are as strong as the real
    farmland (E7, E8);
  * the release block (release.status): a DRAFT until a reviewer has signed
    off these exact numbers.
"""
import base64
import contextlib
import contextvars
import functools
import html
import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path

import numpy as np

from .version import png_metadata          # the build, in every PNG (OUT-01)

log = logging.getLogger(__name__)


def sensor_name(backend) -> str:
    """'NISAR' or 'Sentinel-1' (config.sensor_name)."""
    from .config import sensor_name as _sensor
    return _sensor(str(backend or ""))


MM_IN = 25.4
# The per-cell average-rate raster (average rate x years). Its name says so:
# "vertical_movement_total_*" (until 6 Oct 2026) read as the measured total.
AVERAGE_RATE_RASTER = "vertical_change_at_average_rate"
OLD_TOTAL_RASTER = "vertical_movement_total"
Z95 = 1.96
# An area figure needs at least this share of its outline measured, with no
# quarter of it left almost blank (release.MIN_MEASURED_PERCENT is the same
# number: under it the report cannot be released).
MIN_MEASURED_PERCENT = 50.0
# The recent rate: the last two years of the record (two full seasonal
# cycles, so the season cancels), given only when the record is at least
# three years long - otherwise "recent" is most of the record.
RECENT_YEARS = 2.0
MIN_RECENT_SPAN = 3.0
# The second pass of the farmland (stats-3): features that beat chance are
# taken out of it grown by this much (the map's smoothing: a feature's
# smoothed edge is out with it).
FARMLAND_SECOND_PASS_M = 500.0
# Farmland placements of each feature's own footprint for its year-by-year
# null (report_insights.trend_shape).
YEARLY_NULL_PLACEMENTS = 40
# The whole study area's relative term: the scatter of its 3 x 3 blocks'
# means (blocks ~5-9 km on the two projects), as the standard error of the
# area mean - the satellite part of the whole-area figure, beside GNSS.
FULL_BLOCKS = 3
# The chart band's per-date noise (series_noise): placements of each report
# area's footprint on the comparison ground, and how many must fit before
# they are used instead of the area's own scatter.
SERIES_NOISE_PLACEMENTS = 30
SERIES_NOISE_MIN = 10
# A feature is "ground reshaped during the record" when this share of it has
# a DEM height error beyond the comparison ground's 99th percentile (_reshaped).
RESHAPED_SHARE = 0.5
# "Ground reshaped since the elevation model" (gap-surface_mine_site-6): the
# DEM height error averaged over 500 m (a single cell's demErr is noisy and
# partly deformation correlated with baseline) differs from the comparison
# ground's by more than 3 of its robust sd. Site B 6 Oct 2026: farmland's
# 500 m mean demErr has a robust sd of 4.2 m (threshold 12.5 m); Feature 1
# (the spoil patch) reads -14.9 m, Feature 2 -5.4 m. A flag for the words
# only: the rate stays (RESHAPED_SHARE above withholds it, for metres of
# change during the record).
DEM_CHANGE_SMOOTH_M = 500.0
DEM_CHANGE_SIGMAS = 3.0
# A total "at the average rate" is printed beside the measured one when they
# differ by this much (INT1: Site A Area A2 -84.1 measured against
# -76.7 at the average rate, a 9% gap the page should show; under 5 mm or 5%
# the second figure is noise to a reader).
TOTALS_DIFFER_MM = 5.0
TOTALS_DIFFER_SHARE = 0.05
# The whole study area's comparison-ground term (E13, trust review 7 Oct
# 2026): its relative figure is measured again with everything cut from the
# comparison ground (report areas, features that beat chance) grown by each of
# these widths; the scatter of the results is added to its uncertainty
# (Site A: -0.55 / -0.835 / -0.97 mm/yr, sd about 0.2).
COMPARISON_GROUND_BUFFERS_M = (0.0, FARMLAND_SECOND_PASS_M, 1000.0)
# The rising detection limit is printed beside the sinking one only when it
# differs by more than this share of it (E5).
RISING_LIMIT_DIFFERS = 0.10
# A coloured patch is listed under a report area only when at least this
# share of it lies inside (E12); otherwise it is listed as mostly outside.
PATCH_INSIDE_SHARE = 0.5
# 'Movement by years since mining': rows resting on fewer dated areas are not
# printed for the client (E3 / E20, finding 51a: rows -8 and 26 rested on 1).
YEARS_SINCE_MIN_AREAS = 3
# The presenter output: one page per report area (area_pdf).
PDF_NAME = "client_area_summaries.pdf"
# Every area page of the PDF and the page's legend carry these (E26: the PDF
# travels without the page's Limitations).
PDF_CAVEAT = ("Vertical assumes all movement is up or down: the radar measures along its line of sight from one "
              "satellite track, so sideways movement (at the edges of a sinking area) can read as vertical. Each "
              "value combines the ground around it{support}. {frame}")
# The technical record is the preparer's working file (package.py's README:
# technical/ is not sent to the client); 'available on request' offered it
# (final check of 7 Oct 2026).
TECHNICAL_RECORD_NOTE = ("A technical record (quality assessment, GNSS tie, provenance) is kept with the project; the "
                         "preparer can answer questions about any figure from it.")
# Which words a person wrote and which the software worked out (2 Nov 2026,
# readiness review HIL-02): the readings, verdicts, feature descriptions and
# possible causes are calculated by the program; the named reviewer checks
# them at sign-off. In every report footer, and beside the observations.
GENERATED_NOTE = ("The readings, verdicts, feature descriptions and possible causes in this report are calculated by "
                  "the software from the measurements. They become a professional opinion only through the review "
                  "and sign-off named here.")
# The standard limitation of use, in every report footer and one-pager. Its
# last sentence says which figures carry a range (final check of 7 Oct 2026:
# 'every figure ... with the uncertainty stated beside it' while the map's
# peak and the patch table carry none - they are measured values read
# against ordinary ground's range, not estimates of an area's movement).
DISCLAIMER = ("This report describes ground movement measured by satellite radar over the dates stated. It is a "
              "screening of ground movement, not a survey: it does not replace levelling, GNSS monitoring or "
              "inspection on the ground, and should not be used on its own for design, valuation or legal "
              "decisions. Every figure is an estimate: area, feature and mine-ground figures and every total carry a "
              "95% range; map values are set against ordinary ground.")


def parse_areas(cfg) -> list[tuple[str, str]]:
    """[(name, spec)] from cfg.report_areas; default: the study area + every focus site."""
    specs = list(getattr(cfg, "report_areas", ()) or ())
    if not specs:
        specs = ["Full Satellite Imaging:full"] + [s for s in (getattr(cfg, "focus_sites_lalo", ()) or ())]
    out = []
    for s in specs:
        name, val = s.split(":", 1)
        out.append((name.strip(), val.strip()))
    return out


def _mask(ctx, spec: str) -> tuple[np.ndarray, list, np.ndarray]:
    """The area's coherent cells, its outline rings (lon/lat) for the map,
    and its whole footprint on the grid - every cell inside the outline,
    measured or not, the denominator of measured_percent."""
    from .stack_backend import _polygon_mask
    sg, good, attrs = ctx.sg, ctx.good, ctx.attrs
    aoi = sg.aoi_mask(ctx.cfg.bbox, attrs, good.shape)
    if spec.lower() == "full":
        w, s_, e, n = ctx.cfg.bbox
        return good & aoi, [np.array([[w, s_], [e, s_], [e, n], [w, n], [w, s_]])], aoi
    if spec.lower().endswith(".geojson"):
        gj = Path(spec)
        rings = []
        for f in json.loads(gj.read_text())["features"]:
            g = f["geometry"]
            for poly in (g["coordinates"] if g["type"] == "MultiPolygon" else [g["coordinates"]]):
                rings.append(np.array(poly[0])[:, :2])
        fp = _polygon_mask(gj, good.shape, attrs).astype(bool)
        return fp & good, rings, fp
    s_, w, n, e = (float(v) for v in spec.split(","))
    s_, n, w, e = min(s_, n), max(s_, n), min(w, e), max(w, e)
    box = sg.aoi_mask((w, s_, e, n), attrs, good.shape)
    return box & good, [np.array([[w, s_], [e, s_], [e, n], [w, n], [w, s_]])], box


def coverage(m: np.ndarray, fp: np.ndarray, imaged: np.ndarray | None, px_km2: float) -> dict:
    """
    How much of an area's outline was measured (methodology-4,
    gap-surface_mine_site-3). Until 5 Oct 2026 an area's "km²" was its
    MEASURED cells only, with no denominator, so an outline 30% measured
    still got one number and "within normal ground variation" - and on a
    surface mine the cells that drop out are the pits, haul roads and fresh
    spoil, the ground most likely to be moving.

    footprint_km2 the outline on the grid; area_km2 / cells what was
    measured; measured_percent = measured / outline; outside_frame_percent
    the part of the outline the satellite never imaged (reported apart from
    decorrelation); coverage_balanced False when a quarter of the outline's
    box holding >= 20% of its imaged cells is < 10% measured (the measured
    cells all on one side).
    """
    m = m & fp
    fp_n, meas = int(fp.sum()), int(m.sum())
    img = fp & imaged if imaged is not None and imaged.shape == fp.shape else fp
    img_n = int(img.sum())
    out = {"cells": meas, "area_km2": round(meas * px_km2, 2), "footprint_km2": round(fp_n * px_km2, 2),
           "measured_percent": round(100.0 * meas / fp_n, 1) if fp_n else 0.0,
           "outside_frame_percent": round(100.0 * (fp_n - img_n) / fp_n, 1) if fp_n else 100.0,
           "coverage_balanced": True}
    if img_n:
        ys, xs = np.nonzero(img)
        q = (ys > 0.5 * (ys.min() + ys.max())).astype(int) * 2 + (xs > 0.5 * (xs.min() + xs.max())).astype(int)
        n_q = np.bincount(q, minlength=4)
        m_q = np.bincount(q, weights=m[ys, xs].astype(float), minlength=4)
        out["coverage_balanced"] = not bool(np.any((n_q >= 0.2 * img_n) & (m_q < 0.1 * np.maximum(n_q, 1))))
    return out


def _gnss(cfg, dates) -> tuple[dict, str]:
    """(this record's GNSS tie or {}, why there is no background). Only a tie
    whose record_window is these dates and that is not older than the
    velocity product is used (gnss_tie.check); a tie left by an earlier run
    is never read as this one's."""
    P = Path(cfg.project_dir)
    try:
        from . import gnss_tie
        tie, why = gnss_tie.check(cfg, dates)
        tie = tie or {}
    except (ImportError, AttributeError):
        f = P / "deliverables" / "gnss_tie.json"
        tie, why = (json.loads(f.read_text()) if f.exists() else {}), ""
        if not tie:
            why = "no GNSS tie on disk"
    if (tie.get("background") or {}).get("vertical_mm_yr") is None and not why:
        why = tie.get("reason") or "no GNSS station overlapped the record long enough to measure the regional motion"
    return tie, why


# What the client reads when gnss_tie.check refused the tie file (client_gnss_reason)
GNSS_TIE_NOT_RUN = "the comparison with the regional GNSS stations has not been made for this record"
GNSS_TIE_UNREADABLE = "the comparison with the regional GNSS stations on file could not be read"
# (no ', so it was not used': the page, the notes and the KMZ go on '..., so
# they are relative to ...', and 'No GNSS station could anchor' already says it)
GNSS_TIE_OUT_OF_DATE = "the comparison with the regional GNSS stations on file is out of date for these measurements"


def client_gnss_reason(tie: dict | None, why: str | None) -> str | None:
    """_gnss's reason in the client's words. A tie of this record that could
    not anchor (background null) already says why in plain words and is kept;
    a tie FILE that gnss_tie.check refused or that is missing gets a plain
    sentence - its own reason names files and stages ('gnss_tie.json is older
    than the velocity product it would anchor' reached the page until 7 Oct
    2026). The technical reason stays in gnss.tie_not_used and in the send
    rule's reasons, the reviewer's list."""
    if tie or not why:
        return why
    s = str(why)
    if s.startswith(("no gnss_tie.json", "no GNSS tie on disk")):
        return GNSS_TIE_NOT_RUN
    if "not readable" in s:
        return GNSS_TIE_UNREADABLE
    return GNSS_TIE_OUT_OF_DATE


def _tie_for_send_rule(cfg, tie: dict, why: str | None) -> dict | None:
    """The tie the send rule judges: this record's; or, when gnss_tie.check
    refused the file on disk (made before the regional floor, for another
    record window, older than the velocity product, unreadable), that
    refusal as a tie with no background - the figures went relative-only,
    so the rule holds ('no GNSS anchor: <the technical reason>'), as
    summary.json does for a pre-floor tie (deliverables._apply_qa_and_gnss).
    Until 7 Oct 2026 a refusal gave None and the QA's rule unchanged. No file
    at all: None, the QA's own rule (the tie has not run; summary.json says
    the same)."""
    if tie:
        return tie
    if why and (Path(cfg.project_dir) / "deliverables" / "gnss_tie.json").exists():
        return {"background": None, "reason": why}
    return None


def _los_up(cfg, shape) -> np.ndarray:
    """The LOS -> vertical factor map; raises rather than assume another
    site's incidence (deliverables.los_up_or_raise; 0.78 was Site A's)."""
    from . import deliverables as dl
    f = getattr(dl, "los_up_or_raise", None)
    if f is not None:
        return f(cfg, shape, fill=True)
    up = dl.geometry_field(cfg, "los_up")
    if up is None:
        raise FileNotFoundError("No LOS geometry on the product grid: vertical figures need it")
    med = float(np.nanmedian(up))
    return np.where(np.isfinite(up), up, med)


def _imaged(cfg, shape) -> np.ndarray | None:
    try:
        from . import deliverables as dl
        m, _ = dl.observable_mask(cfg, shape)
        return m
    except Exception as exc:                                  # noqa: BLE001 - coverage then counts the outline
        log.info("Imaged footprint not read (%s); measured share is against the whole outline", exc)
        return None


def _stream(ctx, groups: dict, t_from: float | None) -> tuple[dict, np.ndarray | None]:
    """
    One pass over the geocoded cube: the mean series (mm) of each group of
    flat cell indices, and - for the epochs at or after t_from - each cell's
    least-squares rate (mm/yr) over that window. Index groups, not grid
    masks, so 40 farmland placements per feature cost kilobytes, not a grid
    each. Reads only from the first needed epoch when there are no groups.
    """
    import h5py
    n_ep = len(ctx.dates)
    out = {k: np.full(n_ep, np.nan) for k in groups}
    size = ctx.good.size
    win = None
    if t_from is not None:
        win = ctx.t >= t_from
        tw = ctx.t - t_from
        # counts and time sums in float32 (t <= 2 yr, <= ~70 dates); the
        # displacement sums in float64
        n, St, Stt = (np.zeros(size, np.float32) for _ in range(3))
        Sy, Sty = np.zeros(size), np.zeros(size)
    with h5py.File(ctx.ts_file, "r") as f:
        ds = f["timeseries"]
        b = max(1, int(ds.chunks[0])) if ds.chunks else 1
        i_start = 0 if groups or win is None else (int(np.argmax(win)) // b) * b
        for i0 in range(i_start, n_ep, b):
            blk = ds[i0:i0 + b]
            for j in range(blk.shape[0]):
                i = i0 + j
                sl = blk[j].ravel()
                for k, ix in groups.items():
                    v = sl[ix]
                    fin = np.isfinite(v)
                    if fin.any():
                        out[k][i] = float(v[fin].mean()) * 1000.0
                if win is not None and win[i]:
                    ok = np.isfinite(sl)
                    vz = np.where(ok, sl * np.float32(1000.0), np.float32(0.0))
                    tj = np.float32(tw[i])
                    n += ok
                    St[ok] += tj
                    Stt[ok] += tj * tj
                    Sy += vz
                    Sty += vz * tj
    rate = None
    if win is not None:
        nn = n.astype(np.float64)
        St, Stt = St.astype(np.float64), Stt.astype(np.float64)
        den = nn * Stt - St * St
        with np.errstate(invalid="ignore", divide="ignore"):
            rate = np.where((nn >= 6) & (den > 0), (nn * Sty - St * Sy) / den, np.nan).reshape(ctx.good.shape)
        rate[~ctx.good] = np.nan
    return out, rate


def _placements(mask: np.ndarray, domain: np.ndarray, n_max: int, seed: int = 0,
                min_valid: float = 0.5) -> list[np.ndarray]:
    """Up to n_max non-overlapping copies of `mask`'s footprint over `domain`
    (each at least min_valid on it), as flat index arrays of the domain cells
    each copy covers. Positions on a grid stepped by the footprint's own box,
    so the copies are about independent; chosen with a fixed seed."""
    ys, xs = np.nonzero(mask)
    if not ys.size:
        return []
    r0, c0 = ys.min(), xs.min()
    K = mask[r0:ys.max() + 1, c0:xs.max() + 1]
    h, w = K.shape
    nk = float(K.sum())
    ny, nx = domain.shape
    cands = []
    for r in range(0, ny - h + 1, h):
        for c in range(0, nx - w + 1, w):
            win = domain[r:r + h, c:c + w] & K
            if win.sum() >= min_valid * nk:
                rr, cc = np.nonzero(win)
                cands.append((rr + r) * nx + (cc + c))
    if len(cands) > n_max:
        pick = np.random.default_rng(seed).choice(len(cands), n_max, replace=False)
        cands = [cands[i] for i in sorted(pick)]
    return cands


def _placement_series(ctx, groups: dict, pl_series: dict, prefix: str, far: dict, up: float) -> list:
    """The relative series (vertical mm, class-matched like the region's own)
    of every streamed placement whose group key starts with `prefix`."""
    R = []
    for key in sorted((k for k in groups if k.startswith(prefix)), key=lambda k: int(k.rsplit("_", 1)[1])):
        if key not in pl_series:
            continue
        tmp = np.zeros(ctx.good.size, bool)
        tmp[groups[key]] = True
        R.append(ctx.matched(pl_series[key], tmp.reshape(ctx.good.shape), far) / up)
    return R


def _scatter_about_trend(y: np.ndarray, t: np.ndarray) -> float | None:
    """RMS of a series about its own straight-line fit (mm) - what one image
    adds on top of the trend: season, weather, unwrapping noise."""
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(y)
    if ok.sum() < 12:
        return None
    r = y[ok] - np.polyval(np.polyfit(t[ok], y[ok], 1), t[ok])
    return float(np.sqrt(np.mean(r * r)))


def series_noise(R: list, summ: dict | None, up: float, t: np.ndarray, ground: str = "farmland") -> dict:
    """
    The per-date noise of a region's chart band (clientout-6, stats-8): how far
    one image of ordinary ground of the same footprint strays from its own
    trend, vertical mm, 1 sigma. Until 5 Oct 2026 the band was a rate cone,
    zero wide at the first date, so Site B's project zone (judged within
    normal variation) left it on 14 of 270 dates, 7 of them in 2016, under a
    caption that called leaving the band a detection.

    Measured on the comparison-ground placements of the region's own
    footprint (the RMS over placements of each one's scatter about its trend)
    when at least SERIES_NOISE_MIN placements fit; else on the region's own
    series (its scatter about its own trend: the same quantity for ground
    that is not moving, larger - so a wider band - for ground that is).
    """
    rms = [s for s in (_scatter_about_trend(r, t) for r in R) if s is not None]
    if len(rms) >= SERIES_NOISE_MIN:
        return {"series_noise_mm": round(float(np.sqrt(np.mean(np.square(rms)))), 2),
                "series_noise_basis": f"{len(rms)} {_ri().ground_noun(ground)} patches of the same footprint"}
    own = None
    if summ and summ.get("series_mm"):
        own = _scatter_about_trend(np.array([np.nan if v is None else v for v in summ["series_mm"]], float) / up, t)
    return {"series_noise_mm": round(own, 2) if own is not None else None,
            "series_noise_basis": ("the area's own scatter about its trend (too few comparison patches of its size)"
                                   if own is not None else None)}


def series_band(t: np.ndarray, centre_rate: float, cone_rate: float | None, noise_mm: float | None) -> dict:
    """
    The two bands of a movement-over-time chart (mm, per date):

      inner  centre_rate x t +/- cone_rate x t - how far the AVERAGE RATE of
             ordinary ground of this footprint reaches (the area's decision
             threshold). The chart's straight line (the area's rate x t, from
             zero) ends outside it exactly when |rate - centre_rate| exceeds
             the threshold - the table's reading, by construction.
      outer  centre_rate x t +/- sqrt((cone_rate x t)^2 + (1.96 noise)^2) -
             where ordinary ground's single images wander (season, weather):
             a quiet area's thin line stays inside it on ~19 dates in 20.

    cone_rate None (no comparison) gives no bands at all - never a zero-width
    one. noise None gives the inner band only.
    """
    t = np.asarray(t, dtype=float)
    out = {"centre": centre_rate * t, "inner": None, "outer": None}
    if cone_rate is None or not np.isfinite(cone_rate) or cone_rate <= 0:
        return out
    c = out["centre"]
    out["inner"] = (c - cone_rate * t, c + cone_rate * t)
    if noise_mm is not None and np.isfinite(noise_mm) and noise_mm > 0:
        h = np.sqrt((cone_rate * t) ** 2 + (Z95 * noise_mm) ** 2)
        out["outer"] = (c - h, c + h)
    return out


def _dem_error_grid(cfg, shape) -> np.ndarray | None:
    """MintPy's DEM height error on the product grid (geo/geo_demErr.h5, L2
    geocodes it), or None. Read once; only the feature gate below uses it."""
    p = Path(getattr(cfg, "mintpy_dir", Path(cfg.project_dir) / "mintpy")) / "geo" / "geo_demErr.h5"
    if not p.exists():
        return None
    try:
        import h5py
        with h5py.File(p, "r") as f:
            k = next((k for k in ("dem", "demErr") if k in f), None)
            if k is None or tuple(f[k].shape) != tuple(shape):
                return None
            return f[k][:].astype(np.float32)
    except OSError as exc:
        log.info("DEM error layer not read (%s)", exc)
        return None


def _reshaped(dh: np.ndarray | None, m: np.ndarray, far: np.ndarray) -> bool | None:
    """
    Was this feature's ground dug out or built up by metres? Its DEM height
    error (the 2011-15 elevation model against the ground the radar saw) is
    beyond the comparison ground's own 99th percentile of |error| on at least
    RESHAPED_SHARE of its cells - 1% of ordinary ground passes that by
    construction, so half of a patch passing it is not ordinary ground. Part
    of demErr is deformation correlated with baseline (QA dem_error note),
    hence a share, not a cell. None when the layer is absent: no claim either
    way (verify note on gap-surface_mine_site-7: gate on the measured change,
    not on every rising feature).
    """
    if dh is None or m.sum() < 20:
        return None
    a = np.abs(dh)
    ref = a[far & np.isfinite(a)]
    if ref.size < 1000:
        return None
    lim = float(np.percentile(ref, 99))
    v = a[m & np.isfinite(a)]
    return bool(v.size and float(np.mean(v > lim)) >= RESHAPED_SHARE)


def _dem_smoothed(dh: np.ndarray | None, good: np.ndarray, attrs: dict,
                  smooth_m: float = DEM_CHANGE_SMOOTH_M) -> np.ndarray | None:
    """The DEM height error averaged over a smooth_m box of measured cells
    (NaN-aware; NaN where under a fifth of the box is measured)."""
    if dh is None:
        return None
    from scipy import ndimage
    from .significance import pixel_metres
    py, px = pixel_metres(attrs)
    k = max(1, int(round(smooth_m / float(np.sqrt(py * px)))))
    ok = good & np.isfinite(dh)
    w = ndimage.uniform_filter(ok.astype(np.float32), k, mode="constant")
    v = ndimage.uniform_filter(np.where(ok, dh, 0.0).astype(np.float32), k, mode="constant")
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(w > 0.2, v / np.maximum(w, 1e-6), np.nan).astype(np.float32)
    del w, v, ok
    return out


def _dem_change(sm: np.ndarray | None, m: np.ndarray, far: np.ndarray) -> dict | None:
    """{flag, feature_median_m, farmland_median_m, farmland_robust_sd_m,
    threshold_m, smoothing_m} - the raw gate: the feature's 500 m mean DEM
    error against DEM_CHANGE_SIGMAS robust sd of the comparison ground's;
    None without the layer. No longer what the report uses (E1, 7 Oct 2026:
    MintPy's DEM error follows the rate, so Site B's fast-sinking Feature
    1 passed it): compute calls report_insights.dem_change, which tests what
    the movement does not explain and keeps this gate as its raw_flag."""
    if sm is None or m.sum() < 20:
        return None
    ref = sm[far & np.isfinite(sm)]
    v = sm[m & np.isfinite(sm)]
    if ref.size < 1000 or v.size < 20:
        return None
    c = float(np.median(ref))
    rsd = 1.4826 * float(np.median(np.abs(ref - c)))
    med = float(np.median(v))
    thr = DEM_CHANGE_SIGMAS * rsd
    return {"flag": bool(rsd > 0 and abs(med - c) > thr), "feature_median_m": round(med, 1),
            "farmland_median_m": round(c, 1), "farmland_robust_sd_m": round(rsd, 2), "threshold_m": round(thr, 1),
            "smoothing_m": DEM_CHANGE_SMOOTH_M,
            "basis": (f"MintPy DEM height error averaged over {DEM_CHANGE_SMOOTH_M:.0f} m; flagged when the feature's "
                      f"median differs from the comparison ground's by more than {DEM_CHANGE_SIGMAS:g} of its "
                      "robust sd")}


def _block_sd(field: np.ndarray, m: np.ndarray, nb: int = FULL_BLOCKS) -> float | None:
    """Standard error of the mean of `field` over m, from the scatter of the
    means of an nb x nb split of m's bounding box (blocks with < 1000 cells
    skipped) - the satellite part of the whole study area's uncertainty."""
    ys, xs = np.nonzero(m & np.isfinite(field))
    if ys.size < 1000:
        return None
    ry = np.linspace(ys.min(), ys.max() + 1, nb + 1)
    rx = np.linspace(xs.min(), xs.max() + 1, nb + 1)
    by = np.clip(np.searchsorted(ry, ys, side="right") - 1, 0, nb - 1)
    bx = np.clip(np.searchsorted(rx, xs, side="right") - 1, 0, nb - 1)
    b = by * nb + bx
    v = field[ys, xs]
    cnt = np.bincount(b, minlength=nb * nb)
    s = np.bincount(b, weights=v, minlength=nb * nb)
    means = s[cnt >= 1000] / cnt[cnt >= 1000]
    if means.size < 3:
        return None
    return float(np.std(means, ddof=1) / np.sqrt(means.size))


def _outline(mask: np.ndarray, attrs: dict, tol_cells: float = 2.0) -> list | None:
    """The largest polygon of a feature's cells, simplified to ~tol_cells
    pixels, as [[lon, lat], ...] - for the KMZ and the dashboard map."""
    try:
        from affine import Affine
        from rasterio import features as rf
        from shapely.geometry import shape
    except ImportError:
        return None
    ys, xs = np.nonzero(mask)
    if not ys.size:
        return None
    r0, c0 = ys.min(), xs.min()
    sub = mask[r0:ys.max() + 1, c0:xs.max() + 1].astype(np.uint8)
    x0, y0 = float(attrs["X_FIRST"]), float(attrs["Y_FIRST"])
    dx, dy = float(attrs["X_STEP"]), float(attrs["Y_STEP"])
    tr = Affine(dx, 0.0, x0 + c0 * dx, 0.0, dy, y0 + r0 * dy)
    geoms = [shape(g) for g, v in rf.shapes(sub, mask=sub.astype(bool), transform=tr) if v == 1]
    if not geoms:
        return None
    g = max(geoms, key=lambda x: x.area).simplify(tol_cells * abs(dx), preserve_topology=True)
    return [[round(float(x), 5), round(float(y), 5)] for x, y in g.exterior.coords]


# ------------------------------------------------------------------ the figures of one region
def tested_fields(sig: dict | None, up: float, cov: dict | None = None) -> dict:
    """
    One region's farmland test in the report's terms (vertical, per year).

    testable   significance says the null exists (>= 5 independent
               placements) AND, when `cov` is given, at least
               MIN_MEASURED_PERCENT of the outline was measured with no
               quarter of it left almost blank. Otherwise detected, the
               limits and the S/N are None and not_testable_reason says why.
    detection_limit_mm_yr     80%-power limit for SINKING (significance
                              detection_limit_sink_mm_yr; E5, trust review 7 Oct
                              2026): the client's question. The worse sign
                              (detectable_80) it replaces was set by rising
                              ground on the skewed farmland null, so sinking of
                              the stated limit was caught about 9 times in 10,
                              not 4 in 5 (Site A Area A2 19.7 -> 18.3
                              mm/yr). detection_limit_rise_mm_yr beside it, and
                              the worse sign as detection_limit_either_direction
                              _mm_yr; detection_limit_basis says which the first
                              is ('either direction' for a significance dict
                              from before 7 Oct 2026, which has no per-sign limit)
    null_local / null_coherence_matched  significance's diagnostic nulls (E6,
                              contract C1): None or {median_mm_yr, spread_mm_yr,
                              n_independent, disagrees, why, ...}, line of sight
    decision_threshold_mm_yr  the empirical 95% threshold of |null - median|
    signal_to_noise           |excess| / threshold (>= 1 <=> detected)
    spread_for_ranges_mm_yr   the null sd, widened to a prediction interval
                              when the null is thin - what the recent-rate
                              range uses (the area's own 95% range is on the
                              test's frame: tested_range)
    A significance dict from before 5 Oct 2026 (no `testable`) is read with
    1.96 / 2.8 sd for the threshold and the limit.
    """
    sig = sig or {}
    testable = sig.get("testable")
    if testable is None:
        testable = (sig.get("farmland_spread_1sigma_mm_yr") is not None
                    and not str(sig.get("null", "")).startswith("unavailable"))
    reason = sig.get("not_testable_reason") or (None if testable else
                                                "no comparison with ordinary ground could be made")
    if testable and cov is not None:
        if cov["measured_percent"] < MIN_MEASURED_PERCENT:
            testable, reason = False, (
                f"Only {cov['measured_percent']:.0f}% of this area could be measured; the rest (water, woodland, "
                "active workings or fresh spoil) changes too much between images to measure and may be moving")
        elif not cov.get("coverage_balanced", True):
            testable, reason = False, ("The measurements cover only part of this area: one side of it is almost "
                                       "unmeasured, so one figure cannot stand for the whole")
    out = {"testable": bool(testable), "not_testable_reason": None if testable else reason,
           "null_method": sig.get("method"), "null_shape": sig.get("null_shape"),
           "approx_independent_placements": sig.get("approx_independent_placements"),
           "null_thin": sig.get("null_thin"), "n_placements": sig.get("n_placements")}
    if not testable:
        out.update({"detected": None, "detection_limit_mm_yr": None, "detection_limit_rise_mm_yr": None,
                    "detection_limit_either_direction_mm_yr": None, "detection_limit_basis": None,
                    "decision_threshold_mm_yr": None,
                    "signal_to_noise": None, "z": None, "p_two_sided": None, "farmland_spread_1sigma_mm_yr": None,
                    "spread_for_ranges_mm_yr": None, "null_local": None, "null_coherence_matched": None})
        return out
    sd = float(sig["farmland_spread_1sigma_mm_yr"])
    spread = float(sig.get("spread_for_ranges_mm_yr") or sd)
    excess, thr = _test_frame(sig)
    det80 = sig.get("detectable_80_mm_yr")
    det80 = float(det80) if det80 is not None else 2.8 * sd
    sink, rise = sig.get("detection_limit_sink_mm_yr"), sig.get("detection_limit_rise_mm_yr")
    s2t = sig.get("signal_to_threshold")
    s2t = float(s2t) if s2t is not None else (abs(excess) / thr if thr > 0 else None)
    det = sig.get("distinguishable_from_farmland")
    out.update({"detected": bool(det) if det is not None else bool(s2t is not None and s2t >= 1),
                "detection_limit_mm_yr": round((float(sink) if sink is not None else det80) / up, 1),
                "detection_limit_rise_mm_yr": round(float(rise) / up, 1) if rise is not None else None,
                "detection_limit_either_direction_mm_yr": round(det80 / up, 1),
                "detection_limit_basis": "sinking" if sink is not None else "either direction",
                "null_local": _null_diag(sig.get("local_null")),
                "null_coherence_matched": _null_diag(sig.get("coherence_null")),
                "null_checks_note": sig.get("null_checks_note"),
                "decision_threshold_mm_yr": round(thr / up, 1),
                "signal_to_noise": round(s2t, 2) if s2t is not None else None,
                "z": sig.get("z"), "p_two_sided": sig.get("p_two_sided"),
                "excess_vertical_mm_yr": round(excess / up, 2),
                "null_median_vertical_mm_yr": (round(float(sig["null_median_mm_yr"]) / up, 2)
                                               if sig.get("null_median_mm_yr") is not None else None),
                "farmland_spread_1sigma_mm_yr": round(sd / up, 2),
                "spread_for_ranges_mm_yr": round(spread / up, 2)})
    return out


def _null_diag(d: dict | None) -> dict | None:
    """significance's local / coherence-matched null (E6) as the report
    stores it: the same dict, its figures said to be line of sight (assess is
    unit-agnostic; the page's figures are vertical)."""
    if not isinstance(d, dict):
        return None
    return {**d, "units": "mm/yr along the line of sight"}


def _test_frame(sig: dict) -> tuple[float, float]:
    """(excess over the farmland null median, the decision threshold), LOS
    mm/yr, as significance.assess gives them; a significance dict from before
    5 Oct 2026 is read with the area's value and 1.96 sd."""
    thr = sig.get("threshold_95_mm_yr")
    thr = float(thr) if thr is not None else Z95 * float(sig["farmland_spread_1sigma_mm_yr"])
    excess = sig.get("excess_mm_yr")
    excess = float(excess if excess is not None else sig.get("value_mm_yr") or 0.0)
    return excess, thr


# how far rounding alone can put a range's inner end on the wrong side of 0
# (excess and threshold are stored to 0.01 LOS, the ends to 0.01 vertical)
_ROUNDING_MM_YR = 0.02


def tested_range(tf: dict, sig: dict | None, up: float, bg_v: float | None = None,
                 bg_u: float | None = None) -> tuple[list | None, float | None]:
    """
    A tested report area's 95% range and its 1 sigma (vertical mm/yr), on
    the test's own frame (D1, 7 Oct 2026): the relative part is the excess
    over the farmland null median +/- the decision threshold - the same two
    numbers the reading compares - so for an area that is not anchored "the
    range excludes 0" <=> "Movement detected", always (it was the printed
    rate +/- 1.96 x the spread for ranges, so near the threshold "Movement
    detected" could come with a range that included 0, or "Within normal
    ground variation" with one that excluded it). Anchored (bg_v given): the
    GNSS background is added to both ends and the half-width is
    hypot(threshold, 1.96 x bg_u). The 1 sigma is the half-width / 1.96, so
    the two never disagree. The printed rate is not moved: it stays the
    measured mean (the map's mean over the area); the range is centred on it
    less the null median.

    (None, None) when the area was not tested, or is anchored with no GNSS
    uncertainty. The ends are rounded to 0.01; an inner end that rounding
    alone (within _ROUNDING_MM_YR of 0) puts on the other side of 0 from the
    test is set to the test's side, so rounding cannot contradict the reading.
    """
    if not tf.get("testable") or not sig:
        return None, None
    anchored = bg_v is not None
    if anchored and bg_u is None:
        return None, None
    excess, thr = _test_frame(sig)
    half = float(np.hypot(thr / up, Z95 * float(bg_u) if anchored else 0.0))
    if not np.isfinite(half) or half <= 0:
        return None, None
    centre = excess / up + (float(bg_v) if anchored else 0.0)
    lo, hi = round(centre - half, 2), round(centre + half, 2)
    det = tf.get("detected")
    if not anchored and det is not None:
        if det and lo <= 0 <= hi:
            if excess >= 0 and -_ROUNDING_MM_YR <= lo:
                lo = 0.01
            elif excess < 0 and hi <= _ROUNDING_MM_YR:
                hi = -0.01
        elif not det and lo > 0 and lo <= _ROUNDING_MM_YR:
            lo = 0.0
        elif not det and hi < 0 and hi >= -_ROUNDING_MM_YR:
            hi = 0.0
    return [lo, hi], half / Z95


def moves_with_region(a: dict) -> bool:
    """A GNSS-anchored report area the farmland test did not detect, whose
    absolute 95% range still excludes 0: the regional movement does that, so
    it moves with the surrounding region and nothing marks it as moving on
    its own (D1, 7 Oct 2026)."""
    rng = a.get("range95_mm_yr")
    return bool(a.get("kind") == "area" and a.get("anchored_to_gnss") and a.get("testable")
                and a.get("detected") is False and rng and rng[0] is not None and rng[1] is not None
                and (rng[0] > 0 or rng[1] < 0))


def _wooded_note(a: dict) -> str:
    """'; trees cover 12%' for an area's size line (landcover, 15 Oct 2026);
    '' without a land-cover map or under 5%."""
    lc = a.get("land_cover") or {}
    w = lc.get("wooded_percent")
    if w is None or float(w) < 5:
        return ""
    if lc.get("wooded_masked"):
        return f"; trees cover {float(w):.0f}%, left out (C-band sees the canopy, not the ground)"
    return f"; trees cover {float(w):.0f}% (C-band sees the canopy there)"


def area_word(a: dict) -> str:
    """'sinking' / 'rising' for a report area, read from the side of 0 its
    95% range lies on and only when that range excludes 0 (stats-9). A tested
    area that is not anchored gets one only when the test detected it (its
    range is on the test's frame, so the two agree; D1, 7 Oct 2026). An
    anchored area the test did not detect takes the word of the regional
    movement: the callers say it moves with the surrounding region
    (moves_with_region)."""
    if (a.get("kind") == "area" and a.get("testable") and not a.get("anchored_to_gnss")
            and not a.get("detected")):
        return ""
    rng = a.get("range95_mm_yr")
    if a.get("vertical_mm_yr") is None or not rng or rng[0] is None or rng[1] is None:
        return ""
    return _dir(0.5 * (float(rng[0]) + float(rng[1])), rng)


REGION_NOTE = "moves with the surrounding region"


def _zeroed(summ: dict, up: float, t: np.ndarray) -> np.ndarray:
    """The region's relative series in vertical mm, zeroed on the fitted curve
    at the first date (record_change's start, season removed) so the line
    ends at the measured total; on the straight line's intercept for a
    summary without it. One image carries ~7 mm of noise: zeroing on it
    would shift the whole line."""
    rel = np.array([np.nan if v is None else v for v in summ["series_mm"]], dtype=float) / up
    start = summ.get("start_mm")
    ok = np.isfinite(rel)
    if start is not None:
        rel = rel - float(start) / up
    elif ok.sum() > 2:
        rel = rel - np.polyfit(t[ok], rel[ok], 1)[1]
    return rel


def _series_list(x: np.ndarray) -> list:
    return [None if not np.isfinite(v) else round(float(v), 2) for v in x]


def _range(v: float | None, u: float | None) -> list | None:
    """[lo, hi] of v +/- 1.96 u; None (never a zero-width range) when either is missing or u is 0."""
    if v is None or u is None or not np.isfinite(u) or u <= 0:
        return None
    return [round(v - Z95 * u, 2), round(v + Z95 * u, 2)]


def measured_change(t: np.ndarray, summ: dict, up: float, copies) -> dict:
    """
    The region's change over the record, re-measured with the farmland null
    of the pace change (E10, trust review 7 Oct 2026; contract C3).
    panel_history's summarise measures it with significance.record_change's
    old F-test, which called a change of pace on every farmland patch tried
    (60/60, 40/40, 12/12) and 'nonlinear' on half of them, so every total was
    the hinge model's. Here the hinge total is used only when the region's
    pace change beats the 95th percentile of the same statistic on farmland
    copies of its own footprint (`copies`: their series, vertical mm, as
    _placement_series gives them; the region's LOS series is measured in
    that frame and its total and start put back in LOS); else the total is
    rate x span, and `nonlinear` follows
    the same test. Fewer than significance.PACE_NULL_MIN copies (or [] - the
    whole study area, of which no copy fits) cannot test it: rate x span.

    Returns summ with total_mm, total_method, nonlinear and start_mm (LOS,
    as summarise gives them) replaced, `pace_change` (vertical mm) and, when
    the hinge total is used, `copies_piecewise_vertical_mm` - the copies'
    own hinge totals, the farmland null of that total (total_range).
    summ unchanged when no change can be measured.
    """
    from . import significance as sg
    y = np.array([np.nan if v is None else v for v in summ.get("series_mm") or []], dtype=float)
    if y.size != np.size(t) or np.isfinite(y).sum() < 6:
        return summ
    copies = [np.asarray(r, dtype=float) for r in (copies or [])] if copies is not None else []
    # measured in the copies' frame (vertical), so the test's own words say
    # vertical mm; the statistic scales with the frame, the outcome does not
    ch = sg.record_change(t, y / up, pace_null=copies)
    if not ch:
        return summ
    out = dict(summ)
    out.update({"total_mm": round(float(ch["change_mm"]) * up, 1), "total_method": ch.get("method") or "rate_x_span",
                "nonlinear": bool(ch.get("nonlinear")),
                "start_mm": round(float(ch["start_mm"]) * up, 1) if ch.get("start_mm") is not None else None})
    out["pace_change"] = {"tested": bool(ch.get("pace_change_tested")), "pace_change_mm": ch.get("pace_change_mm"),
                          "farmland_p95_mm": ch.get("pace_null_p95_mm"), "farmland_copies": ch.get("pace_null_n"),
                          "method": ch.get("method"), "reason": ch.get("reason"), "frame": "vertical mm"}
    if out["total_method"] != "rate_x_span" and copies:
        pw = [sg.record_change(t, r) for r in copies]
        out["copies_piecewise_vertical_mm"] = [float(c["piecewise_change_mm"]) for c in pw
                                               if c and c.get("piecewise_change_mm") is not None]
    return out


def fitted_path(t: np.ndarray, y: np.ndarray, min_years: float = 2.0) -> np.ndarray | None:
    """The pace-change curve a 'piecewise_linear_yearly' total is read on,
    from zero at the first date, season removed: significance.record_change's
    own model (a hinge each whole year, the two end segments refined by one
    half-way hinge when p < 0.01), so it ends at that total. For the
    movement-over-time chart (final check of 7 Oct 2026: a feature's chart
    had no line its stated total is read on, and its thin line, zeroed on
    the fitted start, did not start at zero). None when no hinge model fits."""
    from . import significance as sg
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(y)
    if ok.sum() < 6:
        return None
    tt, yy = t[ok] - t[ok][0], y[ok]
    span = float(tt[-1])
    knots = sg._yearly_knots(tt) if span >= min_years else []
    if not knots:
        return None
    for lo, hi in ((0.0, knots[0]), (knots[-1], span)):          # as significance._pace_model
        mid = 0.5 * (lo + hi)
        if np.sum((tt >= lo) & (tt < mid)) < 6 or np.sum(tt >= mid) - np.sum(tt > hi) < 6:
            continue
        _, rss_a, _ = sg._ls_fit(sg._change_design(tt, knots), yy)
        trial = sorted(knots + [mid])
        A = sg._change_design(tt, trial)
        _, rss_b, rho_b = sg._ls_fit(A, yy)
        if sg._nested_p(rss_a, rss_b, 1, tt.size - A.shape[1], rho_b) < 0.01:
            knots = trial
    b, *_ = sg._ls_fit(sg._change_design(tt, knots), yy)
    ta = t - t[ok][0]
    return b[1] * ta + sum(b[4 + j] * np.maximum(ta - k, 0.0) for j, k in enumerate(knots))


def total_range(rel_tot: float, method: str | None, tf: dict | None, sig: dict | None, up: float, span: float,
                copies_piecewise=None, bg_v: float | None = None, bg_u: float | None = None) -> tuple[list | None, str | None]:
    """
    The 95% range of a measured total (vertical mm; E9, trust review 7 Oct
    2026: no total was printed with any uncertainty - Site A Specific
    AOI's '-3.09 in measured' has a 95% range of roughly -8 to +1.8 in).
    Relative part, from the farmland null of the same quantity:
      rate x span  the area's own test frame times the years - (total - the
                   null median x span) +/- the decision threshold x span, so
                   it is the rate's range x span (tested_range)
      hinge        (pace change beat farmland) the farmland copies' own hinge
                   totals (measured_change): total - their median +/- the
                   95th percentile of their |deviation|
    Anchored: + the background x span, half-width hypot(relative, 1.96 x
    GNSS uncertainty x span). (None, None) when the relative part cannot be
    stated (not tested: untested_uncertainty gives what can be) or an
    anchored area has no GNSS uncertainty.
    """
    from . import significance as sg
    anchored = bg_v is not None
    if anchored and bg_u is None:
        return None, None
    P = np.asarray([x for x in (copies_piecewise or []) if x is not None and np.isfinite(x)], dtype=float)
    if method and method != "rate_x_span" and P.size >= getattr(sg, "PACE_NULL_MIN", 20):
        med = float(np.median(P))
        centre, half = rel_tot - med, float(np.quantile(np.abs(P - med), 0.95))
        basis = f"the same measured change on {P.size} farmland copies of this footprint"
    elif tf and tf.get("testable") and sig and sig.get("threshold_95_mm_yr") is not None:
        _, thr = _test_frame(sig)
        med = float(sig.get("null_median_mm_yr") or 0.0) / up
        centre, half = rel_tot - med * span, thr / up * span
        basis = "the area's comparison with ordinary farmland of its size, times the years"
    else:
        return None, None
    if anchored:
        centre += float(bg_v) * span
        half = float(np.hypot(half, Z95 * float(bg_u) * span))
        basis += ", with the regional GNSS uncertainty times the years"
    if not np.isfinite(half) or half <= 0:
        return None, None
    return [round(centre - half, 1), round(centre + half, 1)], basis


def untested_uncertainty(sig: dict | None, up: float, bg_u: float | None = None) -> tuple[float | None, str | None]:
    """
    The 1 sigma (vertical mm/yr) that can be stated for an area that could
    not be compared with farmland (E9, trust review 7 Oct 2026: Site B's
    project area printed '0.000 in/yr' - +1.85 relative plus -1.85 GNSS, an
    exact cancellation - with no uncertainty at all). Its parts:
      the regional GNSS rate's uncertainty, when anchored;
      the comparison-ground part: the null's spread when one exists (the
      area failed only on coverage), else the spread of the few farmland
      patches of its size that fit, widened to a prediction interval for
      that few (significance.prediction_inflation; 3 patches: x2.5).
    (None, None) when neither can be stated. The words say which.
    """
    from . import significance as sg
    sig = sig or {}
    parts, words = [], []
    if bg_u:
        parts.append(float(bg_u))
        words.append(f"the regional GNSS rate's uncertainty (±{Z95 * float(bg_u):.1f} mm/yr)")
    n = sig.get("approx_independent_placements")
    sd = None
    if sig.get("farmland_spread_1sigma_mm_yr") is not None:
        sd = float(sig.get("spread_for_ranges_mm_yr") or sig["farmland_spread_1sigma_mm_yr"]) / up
        what, approx = "ordinary ground of its size", ""
    elif sig.get("thin_null_spread_1sigma_mm_yr") is not None and n and int(n) >= 2:
        sd = float(sig["thin_null_spread_1sigma_mm_yr"]) * sg.prediction_inflation(int(n)) / up
        what, approx = f"the {int(n)} patches of ordinary ground of its size that fit", ", approximate"
    if sd is not None and np.isfinite(sd) and sd > 0:
        parts.append(sd)
        words.append(f"the spread of {what} (±{Z95 * sd:.1f} mm/yr{approx})")
    if not parts:
        return None, None
    return float(np.hypot.reduce(parts)) if len(parts) > 1 else parts[0], " and ".join(words)


def mine_ground_range(tf: dict, sig: dict | None, up: float, v: float) -> tuple[list | None, str | None]:
    """(95% range, basis) of the mapped mine ground's average rate v
    (vertical mm/yr, relative): on its farmland test's frame, as a report
    area's (tested_range, M4); for mine ground that could not be compared
    with farmland, the range that can be stated (untested_uncertainty: the
    spread of ordinary ground of its size, or of the few patches that fit),
    with the basis in words, as an untested report area's (E9; final check
    of 7 Oct 2026 - the footer promises every mine-ground figure a range,
    and untested mine ground printed none). (None, None) when neither."""
    rng, _ = tested_range(tf, sig, up)
    if rng is not None:
        return rng, None
    sd, basis = untested_uncertainty(sig, up)
    rng = _range(v, sd)
    return (rng, basis) if rng else (None, None)


def _recent_range(rec: dict, up: float, bg_v: float | None, bg_u: float | None, full: bool) -> list | None:
    """The last two years' 95% range (vertical mm/yr; E11, trust review 7 Oct
    2026): for a report area or a feature on its farmland test's own frame,
    as the record's range (D1) - the excess over the farmland null median +/-
    the decision threshold, + the GNSS background with half-width
    hypot(threshold, 1.96 x its uncertainty) when anchored; it was the value
    +/- 1.96 x the null's spread, not centred on the null. The whole study
    area (no farmland test): its value +/- 1.96 x hypot(block scatter, GNSS)."""
    anchored = bg_v is not None
    if anchored and bg_u is None:
        return None
    g = Z95 * float(bg_u) if anchored else 0.0
    if not full and rec.get("excess_mm_yr") is not None and rec.get("threshold_mm_yr") is not None:
        c = float(rec["excess_mm_yr"]) / up + (float(bg_v) if anchored else 0.0)
        h = float(np.hypot(float(rec["threshold_mm_yr"]) / up, g))
        return [round(c - h, 2), round(c + h, 2)] if np.isfinite(h) and h > 0 else None
    if rec.get("spread_mm_yr") is None:
        return None
    rv = float(rec["value_mm_yr"]) / up + (float(bg_v) if anchored else 0.0)
    return _range(rv, float(np.hypot(float(rec["spread_mm_yr"]) / up, float(bg_u) if anchored else 0.0)))


def area_entry(name: str, spec: str, kind: str, cov: dict, summ: dict | None, sig: dict | None, up: float,
               bg_v: float | None, bg_u: float | None, span: float, t: np.ndarray,
               recent: dict | None = None, full_rel_sd: float | None = None) -> dict:
    """
    One report area, every figure defined once and the same way for the
    whole study area as for any other (clientout-2: the whole area's rate
    was the GNSS background alone while its series was satellite + GNSS, so
    the report, KMZ chart, workbook and rasters gave three numbers):

      vertical_mm_yr  relative to farmland + the GNSS background (anchored),
                      the relative figure alone (not anchored); None for the
                      whole study area when not anchored (`relative_only`)
      total_mm        the measured change (significance.record_change, with
                      the farmland null of the pace change: measured_change,
                      E10) + background x span; total_at_average_rate_mm =
                      rate x span; total_range95_mm its 95% range
                      (total_range, E9)
      range95_mm_yr   a tested area: on the test's own frame (tested_range,
                      D1 7 Oct 2026) - excess over the farmland null median
                      +/- the decision threshold, + the GNSS background with
                      half-width hypot(threshold, 1.96 x GNSS uncertainty)
                      when anchored; the whole area: vertical +/- 1.96 x
                      hypot(GNSS, the satellite's relative uncertainty). None
                      when it cannot be measured - never a zero-width range
      untested_range95_mm_yr / untested_total_range95_mm  an area that could
                      not be compared with farmland: its figures +/- the
                      uncertainty that can be stated (untested_uncertainty,
                      E9), labelled by untested_uncertainty_basis; range95
                      stays None (no direction word is read from it)
      vertical_uncertainty_1sigma_mm_yr   the range's half-width / 1.96
      reading         the one label every product prints; an anchored area
                      the test did not detect whose range excludes 0 adds
                      "; moves with the surrounding region"
    recent: {"value_mm_yr" (LOS, relative), "window": [d0, d1], and for a
    report area its last-two-years test (_recent_of: excess_mm_yr,
    threshold_mm_yr), for the whole area "spread_mm_yr" (its block scatter)}
    - the last RECENT_YEARS of the record (range: _recent_range, E11).
    """
    anchored = bg_v is not None
    full = kind == "full"
    a = {"name": name, "spec": spec, "kind": kind, **cov, "los_up": round(float(up), 3),
         "anchored_to_gnss": anchored, "relative_only": not anchored}
    if summ is None:
        why = ("The satellite did not image this area" if cov.get("outside_frame_percent", 0) >= 99.9 else
               "Fewer than 5 points in this area could be measured (water, woodland, or ground that changed "
               "between images)")
        a.update({"testable": False if not full else None, "not_testable_reason": why,
                  "reading": "Not measurable: " + why[0].lower() + why[1:],
                  "note": "no reliable measurements", "vertical_mm_yr": None, "relative_vertical_mm_yr": None,
                  "total_mm": None, "total_at_average_rate_mm": None, "total_range95_mm": None,
                  "nonlinear": None, "range95_mm_yr": None,
                  "vertical_uncertainty_1sigma_mm_yr": None, "detected": None, "detection_limit_mm_yr": None,
                  "decision_threshold_mm_yr": None, "signal_to_noise": None, "series_vertical_mm": None,
                  "recent_rate_mm_yr": None, "recent_range95_mm_yr": None})
        return a
    rel_v = float(summ["rate_mm_yr"]) / up
    rel_tot = float(summ["total_mm"] if summ.get("total_mm") is not None else summ["rate_mm_yr"] * span) / up
    rel_series = _zeroed(summ, up, t)
    method = summ.get("total_method") or "rate_x_span"
    a.update({"relative_vertical_mm_yr": round(rel_v, 2), "relative_total_mm": round(rel_tot, 1),
              "total_method": method, "nonlinear": bool(summ.get("nonlinear", False)),
              "pace_change": summ.get("pace_change")})
    if full:
        a.update({"testable": None, "not_testable_reason": None,
                  "regional_gnss_vertical_mm_yr": round(bg_v, 2) if anchored else None,
                  "regional_gnss_uncertainty_mm_yr": round(bg_u, 2) if anchored and bg_u is not None else None,
                  "relative_uncertainty_1sigma_mm_yr": round(full_rel_sd, 2) if full_rel_sd is not None else None})
        if not anchored:
            a.update({"vertical_mm_yr": None, "total_mm": None, "total_at_average_rate_mm": None,
                      "total_range95_mm": None,
                      "vertical_uncertainty_1sigma_mm_yr": None, "range95_mm_yr": None, "signal_to_noise": None,
                      # no "GNSS" in it: the reading of an unanchored whole area
                      # used to be "Region moves with the GNSS network"
                      "reading": "Relative only: no regional anchor, so the movement of the area as a whole is not measured",
                      "series_vertical_mm": _series_list(rel_series)})
        else:
            v = rel_v + bg_v
            unc = float(np.hypot(bg_u, full_rel_sd or 0.0)) if bg_u is not None else None
            rng = _range(v, unc)
            tot = rel_tot + bg_v * span
            a.update({"vertical_mm_yr": round(v, 2), "total_mm": round(tot, 1),
                      "total_at_average_rate_mm": round(v * span, 1),
                      # the whole area's total is rate x span (no copy of it fits:
                      # measured_change), so its range is the rate's x the years
                      "total_range95_mm": ([round(tot - Z95 * unc * span, 1), round(tot + Z95 * unc * span, 1)]
                                           if unc else None),
                      "total_range_basis": ("the whole area's 95% range times the years" if unc else None),
                      "vertical_uncertainty_1sigma_mm_yr": round(unc, 2) if unc else None, "range95_mm_yr": rng,
                      # >= 1 when the 95% range excludes zero: the same scale as every other S/N
                      "signal_to_noise": round(abs(v) / (Z95 * unc), 2) if unc else None,
                      "reading": ("No measurable movement of the region as a whole" if rng is None or rng[0] <= 0 <= rng[1]
                                  else f"The region as a whole is {'sinking' if v < 0 else 'rising'} (GNSS-anchored)"),
                      "series_vertical_mm": _series_list(rel_series + bg_v * t)})
    else:
        tf = tested_fields(sig, up, cov)
        a.update(tf)
        v = rel_v + (bg_v if anchored else 0.0)
        tot = rel_tot + (bg_v * span if anchored else 0.0)
        # the range on the test's own frame (D1, 7 Oct 2026; tested_range)
        rng, unc = tested_range(tf, sig, up, bg_v if anchored else None, bg_u)
        trng, tbasis = total_range(rel_tot, method, tf, sig, up, span, summ.get("copies_piecewise_vertical_mm"),
                                   bg_v if anchored else None, bg_u)
        a.update({"vertical_mm_yr": round(v, 2), "total_mm": round(tot, 1),
                  "total_at_average_rate_mm": round(v * span, 1),
                  "total_range95_mm": trng, "total_range_basis": tbasis,
                  "vertical_uncertainty_1sigma_mm_yr": round(unc, 2) if unc else None,
                  "range95_mm_yr": rng,
                  "series_vertical_mm": _series_list(rel_series + (bg_v * t if anchored else 0.0))})
        if not tf["testable"]:
            # what can be said of an untested figure (E9): never a bare value
            u, words = untested_uncertainty(sig, up, bg_u if anchored else None)
            a.update({"untested_range95_mm_yr": _range(round(v, 2), u),
                      "untested_total_range95_mm": ([round(tot - Z95 * u * span, 1), round(tot + Z95 * u * span, 1)]
                                                    if u else None),
                      "untested_uncertainty_1sigma_mm_yr": round(u, 2) if u else None,
                      "untested_uncertainty_basis": words})
        a["reading"] = (("Movement detected" if tf["detected"] else
                         "Within normal ground variation" + (f"; {REGION_NOTE}" if moves_with_region(a) else ""))
                        if tf["testable"] else "Not testable: " + (tf["not_testable_reason"] or "").rstrip("."))
    rec = recent or {}
    if rec.get("value_mm_yr") is not None and (anchored or not full):
        rv = float(rec["value_mm_yr"]) / up + (bg_v if anchored else 0.0)
        a.update({"recent_rate_mm_yr": round(rv, 2), "recent_range95_mm_yr": _recent_range(rec, up, bg_v, bg_u, full),
                  "recent_window": rec.get("window"),
                  "recent_range_frame": ("the last two years' comparison with farmland (excess over its median +/- "
                                         "its 95% threshold)" if not full and rec.get("excess_mm_yr") is not None
                                         else None)})
    else:
        a.update({"recent_rate_mm_yr": None, "recent_range95_mm_yr": None, "recent_window": rec.get("window")})
    return a


def feature_entry(f: dict, summ: dict, sig: dict | None, up: float, span: float, t: np.ndarray, years: list,
                  null_yearly: dict | None = None, recent: dict | None = None) -> dict:
    """
    A searched-for feature's figures, relative to the surrounding farmland:
    total_relative_mm is the MEASURED change (record_change with the
    farmland null of the pace change, measured_change - E10: the hinge total
    only when its change of pace beats farmland's; Site B Feature 1 read
    415 mm at the average rate against ~347 in its own series, which slowed
    after 2020), total_at_average_rate_mm = rate x span, total_range95_mm
    its 95% range (total_range on the feature's footprint test, E9); the
    series is zeroed on the fitted curve so the chart ends where the title
    says; trend_shape is tested against farmland patches of the same
    footprint (null_yearly). `detected` is the footprint test (the feature's
    own shape shifted over farmland) - informational, since the shape was
    chosen from the data; the decision is f["chance"] (selection-corrected).
    The last two years' range is on that footprint test's frame (E11,
    _recent_range).
    """
    f = {k: v for k, v in f.items() if k != "_mask"}
    sig = sig or {}
    rate = float(summ["rate_mm_yr"]) / up
    rel = _zeroed(summ, up, t)
    tot = float(summ["total_mm"] if summ.get("total_mm") is not None else summ["rate_mm_yr"] * span) / up
    spread = float(sig.get("farmland_spread_1sigma_mm_yr") or 0.0) / up
    method = summ.get("total_method") or "rate_x_span"
    testable = bool(sig.get("testable", sig.get("threshold_95_mm_yr") is not None))
    trng, tbasis = total_range(tot, method, {"testable": testable},
                               sig, up, span, summ.get("copies_piecewise_vertical_mm"))
    # M4 (final check of 7 Oct 2026): the rate's own 95% range, on the same
    # footprint test's frame as its total (so for a rate x span total the
    # total's range is this range x the years); 1.96 x the footprint spread
    # when the test has no threshold
    framed = testable and (sig.get("threshold_95_mm_yr") is not None
                           or sig.get("farmland_spread_1sigma_mm_yr") is not None)
    rrng, _ = tested_range({"testable": True, "detected": None}, sig, up) if framed else (None, None)
    if rrng is None and spread > 0:
        rrng = _range(rate, spread)
    path = fitted_path(t, rel) if method == "piecewise_linear_yearly" else None
    f.update({"vertical_relative_mm_yr": round(rate, 1),
              "rate_range95_mm_yr": rrng,
              "rate_range_basis": ("the feature's own outline moved across ordinary farmland, as its total" if rrng
                                   else None),
              # the curve a pace-change total is read on, from zero (the chart's thick line)
              "fit_vertical_mm": _series_list(path) if path is not None else None,
              # what the radar measured, beside the vertical reading of it
              # (methodology-15: one track cannot separate the two)
              "los_relative_mm_yr": round(float(summ["rate_mm_yr"]), 1),
              "total_relative_mm": round(tot, 0),
              "total_at_average_rate_mm": round(rate * span, 0),
              # its range: the rate's 95% range times the years (E9: every
              # total with its range; the page printed this one bare)
              "total_at_average_rate_range95_mm": ([round(rrng[0] * span, 0), round(rrng[1] * span, 0)]
                                                   if rrng else None),
              "total_range95_mm": [round(trng[0], 0), round(trng[1], 0)] if trng else None,
              "total_range_basis": tbasis,
              "total_method": method,
              "nonlinear": bool(summ.get("nonlinear", False)),
              "pace_change": summ.get("pace_change"),
              "farmland_spread_1sigma_mm_yr": round(spread, 2),
              "detected": sig.get("distinguishable_from_farmland"), "z": sig.get("z"),
              "series_vertical_mm": _series_list(rel)})
    from . import report_insights as ri
    f["trend"] = ri.trend_shape(t, rel, years, spread, null_yearly=null_yearly)
    rec = recent or {}
    if rec.get("value_mm_yr") is not None:
        rv = float(rec["value_mm_yr"]) / up
        f.update({"recent_rate_mm_yr": round(rv, 1), "recent_range95_mm_yr": _recent_range(rec, up, None, None, False),
                  "recent_window": rec.get("window")})
    return f


def _recent_of(r2_cm, r2_null, far_wide, m, window, sg) -> dict | None:
    """The last-two-years rate of region m relative to farmland (LOS mm/yr)
    and its farmland test over that window: the excess over the null median
    and the decision threshold the range is built on (E11, _recent_range),
    and the spread for ranges (kept for the record)."""
    if r2_cm is None or m.sum() < 5:
        return None
    s2 = sg.assess(r2_null, far_wide, m)
    val = s2.get("value_mm_yr")
    if val is None:
        x = r2_cm[m]
        x = x[np.isfinite(x)]
        val = float(x.mean()) if x.size else None
    ok = s2.get("testable", True)
    spread = s2.get("spread_for_ranges_mm_yr", s2.get("farmland_spread_1sigma_mm_yr")) if ok else None
    return {"value_mm_yr": val, "spread_mm_yr": spread, "window": window,
            "excess_mm_yr": s2.get("excess_mm_yr") if ok else None,
            "threshold_mm_yr": s2.get("threshold_95_mm_yr") if ok else None,
            "null_median_mm_yr": s2.get("null_median_mm_yr") if ok else None}


# ------------------------------------------------------------------ compute
def _with_omitted_list(fn):
    """fn run with a fresh 'omitted' list collecting _optional's failures
    (_collecting; 16 Oct 2026, readiness review ERR-01)."""
    @functools.wraps(fn)
    def run(*a, **k):
        with _collecting([]):
            return fn(*a, **k)
    return run


@_with_omitted_list
def compute(cfg) -> dict:
    """(record, context, masks, outlines) of the report. The record's
    'omitted' list names every optional block that failed - the list
    object itself, so a build adds what its figures and page leave out
    (build)."""
    from .panel_history import _Context
    from . import report_insights as ri
    ctx = _Context(cfg)
    sg = ctx.sg
    P = cfg.project_dir
    Q = json.loads((P / "quality_assessment.json").read_text())
    if isinstance(Q.get("geolocation"), dict):
        # the current verdict on the QA's measurement (an older QA said
        # "within about 5 m" with one half 21 m off - quality_assessment.
        # judge_geolocation); the send rule below reads it too
        from .quality_assessment import judge_geolocation
        Q["geolocation"] = judge_geolocation(Q["geolocation"])
    # the checks counted as the verdict counts them (D2, 7 Oct 2026): out of
    # the applicable ones, a check that cannot apply named and not counted;
    # read from 'checks', so a QA record from before D2 is counted the same way
    from .quality_assessment import check_counts
    cnt = check_counts(Q)
    S = json.loads((P / "deliverables" / "summary.json").read_text())
    tie, gnss_why = _gnss(cfg, ctx.dates)
    gnss_reason = client_gnss_reason(tie, gnss_why)          # the page's words; gnss_why the reviewer's
    bg = tie.get("background") or {}
    bg_v, bg_u = bg.get("vertical_mm_yr"), bg.get("vertical_uncertainty_mm_yr")
    anchored = bg_v is not None
    if anchored and not bg_u:
        bg_u = None                       # unknown, not zero: ranges are then withheld
    los_up_map = _los_up(cfg, ctx.good.shape)
    ident = ri.identity(cfg)
    layers = ri.mine_layers(cfg)
    poly = ri.polygon_masks(ctx, layers)
    aoi = sg.aoi_mask(cfg.bbox, ctx.attrs, ctx.good.shape)
    imaged = _imaged(cfg, ctx.good.shape)
    px_km2 = float(np.prod(sg.pixel_metres(ctx.attrs))) / 1e6
    t, span = ctx.t, ctx.span
    years = [int(d[:4]) for d in ctx.dates]
    up_med = float(np.nanmedian(los_up_map[ctx.good]))
    dh = _dem_error_grid(cfg, ctx.good.shape)
    dsm = _dem_smoothed(dh, ctx.good, ctx.attrs)          # 'reshaped since the elevation model' (features only)

    # ---- the report areas first: an area other than the whole study area
    # is not comparison ground (E28, contract C2: Site B's project area,
    # not in the mine layer, had 11.5% of its cells in its own farmland and
    # read +1.85 mm/yr against itself, +1.99 kept out). Its outline (every
    # cell inside it, measured or not) is cut from the farmland grown by the
    # second pass's width, before the feature search, so the class centres,
    # the nulls, the feature search's farmland and the recent windows all
    # leave it out; on Site A no report area touches the farmland.
    areas_spec = parse_areas(cfg)
    masks, outlines, fps = {}, {}, {}
    for name, spec in areas_spec:
        m, rings, fp = _mask(ctx, spec)
        masks[name], outlines[name], fps[name] = m, rings, fp
    far_first = ctx.far.copy()                                # E13's variants are cut from this
    state0 = _farmland_state(ctx)
    area_cut = np.zeros(ctx.good.shape, bool)
    for name, spec in areas_spec:
        if spec.lower() != "full":
            area_cut |= fps[name]
    out_areas = _exclude_report_areas(ctx, area_cut, [n for n, s_ in areas_spec if s_.lower() != "full"])
    cut = area_cut if out_areas.get("applied") else np.zeros(ctx.good.shape, bool)
    del area_cut

    # ---- the map's field and the calibrated feature search, twice when the
    # first finds real movement in the farmland itself (stats-3): it is taken
    # out of the farmland, every class centre, the null and the map band are
    # measured again, and the search re-run on the corrected field. A feature
    # is listed, and cut, by the calibrated chance test alone (E7, contract
    # C4); the quality assessment's stand-outs (every scale that passed its
    # calibrated rule) only mark the features that hold one.
    so_q = _calibrated_standouts(Q)
    sm, cands, null, features_assessed = _feature_search(ctx, los_up_map, poly, aoi, so_q)
    # F1: the leading candidates re-tested with themselves left out of the
    # farmland; one that passes is cut like a first-pass stand-out
    leave_out = _leave_candidate_out(ctx, sm, cands, null, los_up_map, poly, aoi, so_q)
    found = [f["_mask"] for f in cands if f["stands_out"] or (f.get("leave_out") or {}).get("passed")]
    loo_cut = [(f["leave_out"], f["_mask"]) for f in cands
               if not f["stands_out"] and (f.get("leave_out") or {}).get("passed")]
    # the first pass's own test listed these: its chance is the one printed
    # (ri.first_pass_chance, final check of 7 Oct 2026)
    first_cut = [(f["sign"], f.get("chance"), f["_mask"]) for f in cands if f["stands_out"]]
    features_after = None
    if found:
        del sm
        _exclude_from_farmland(ctx, found, FARMLAND_SECOND_PASS_M)
        for m in found:
            cut |= m
        for f in cands:
            f.pop("_mask", None)
        sm, cands, null, features_assessed = _feature_search(ctx, los_up_map, poly, aoi, so_q)
        features_after = sum(1 for f in cands if f["stands_out"])
        # whether each candidate cut on its leave-out test is listed by the
        # second pass's own test (the listing rule is unchanged); one that is
        # carries the re-test's chance corrected for the number re-tested
        # (G2: ri.leave_out_chance - Site A Feature 1 "about 3 in 200",
        # not the second pass's "less than 1 in 200")
        for r, m in loo_cut:
            g = ri.match_candidate({"sign": r["sign"], "_mask": m}, cands)
            r["second_pass"] = ({"stands_out": g.get("stands_out"), "p_text": (g.get("chance") or {}).get("p_text"),
                                 "strength": (g.get("chance") or {}).get("cluster_mass"),
                                 "p": (g.get("chance") or {}).get("p")} if g else None)
            if g is not None and g.get("stands_out"):
                g["chance"] = ri.leave_out_chance(g.get("chance"), r.get("left_out"))
                r["second_pass"]["printed_p_text"] = g["chance"].get("p_text")
        for sign, ch1, m in first_cut:
            g = ri.match_candidate({"sign": sign, "_mask": m}, cands)
            if g is not None and g.get("stands_out") and (g.get("chance") or {}).get("listed_on") is None:
                g["chance"] = ri.first_pass_chance(g.get("chance"), ch1)
    del loo_cut, first_cut
    # one record of everything cut from the first-pass farmland (the report
    # areas and the features that beat chance, both grown by the same width):
    # the exact cells cut (before the buffer), so the KMZ and the Results
    # workbook rebuild this farmland to the cell (results.report_farmland:
    # 'recorded cut') instead of from the second search's simplified outlines
    # (Site A: 1,792,126 farmland cells that way against the report's
    # 1,819,691, nulls 0.1 mm/yr off), and the class centres' shift from the
    # first pass
    second = None
    if found or out_areas.get("applied"):
        second = {"features_removed": len(found),
                  # of them, cut on the leave-candidate-out test alone (F1)
                  "features_removed_on_leave_out": sum(1 for r in leave_out.get("candidates") or []
                                                       if r.get("left_out") and r["passed"]),
                  "report_areas_removed": out_areas.get("areas_cut") or [],
                  "buffer_m": float(FARMLAND_SECOND_PASS_M), **_pass_record(state0, _farmland_state(ctx)),
                  "cut_cells": mask_runs(cut)}
        if features_after is not None:
            second["features_after"] = features_after
    del found
    ctx.report_field = sm                                     # the map draws exactly this
    # the comparison ground in words, measured on this, the report's own
    # farmland after both passes (landcover.comparison_ground, 28 Oct 2026):
    # "farmland" only where cropland dominates; a control_ground_label
    # someone set still wins. Every printer reads the record's label.
    from . import landcover
    cground = landcover.comparison_ground(cfg, ctx.attrs, ctx.far_wide)
    ground = cground["label"]
    # shown: every candidate that beats chance and, failing closed, every one
    # that could not be tested
    shown = [f for f in cands if f["stands_out"] or f["stands_out"] is None]
    listed, beyond_list = shown[:ri.MAX_FEATURES], shown[ri.MAX_FEATURES:]
    within = [f for f in cands if f["stands_out"] is False]

    # ---- the whole study area's comparison-ground term (E13): its relative
    # figure with the cut grown by 0 / 500 / 1000 m
    m_full = next((masks[n] for n, s_ in areas_spec if s_.lower() == "full"), None)
    ground_terms = (_comparison_ground_variants(ctx, far_first, cut, m_full, los_up_map)
                    if m_full is not None and m_full.any() else None)
    del far_first, cut

    # ---- one pass for the areas, mine layers and listed features
    extra = {f"__mine_{L['key']}": poly[L["key"]] & ctx.good & aoi for L in layers}
    extra.update({f"__feature_{i}": f["_mask"] & ctx.good for i, f in enumerate(listed)})
    series, far, _ = ctx.series({k: v for k, v in {**masks, **extra}.items() if v.sum() >= 5})

    # ---- a second pass: farmland placements of each listed feature (its
    # year-by-year null and its pace-change null) and of each report area
    # (the per-date noise of its chart's band and its pace-change null), the
    # dated mined areas (the years-since-mining table, on this farmland), and
    # the last RECENT_YEARS' rate of every cell
    groups = {}
    cand_all = np.zeros(ctx.good.shape, bool)
    for f in cands:
        cand_all |= f["_mask"]
    domain = (ctx.far_wide & ~sg.euclidean_buffer(cand_all, ctx.attrs, ri.NULL_EXCLUDE_M)) if cand_all.any() \
        else ctx.far_wide.copy()
    del cand_all
    for i, f in enumerate(listed):
        for j, ix in enumerate(_placements(f["_mask"] & ctx.good, domain, YEARLY_NULL_PLACEMENTS, seed=i)):
            groups[f"__pl_{i}_{j}"] = ix
    for ai, (name, spec) in enumerate(areas_spec):
        if spec.lower() != "full" and masks[name].sum() >= 5:
            for j, ix in enumerate(_placements(masks[name], domain, SERIES_NOISE_PLACEMENTS, seed=100 + ai)):
                groups[f"__ap_{ai}_{j}"] = ix
    del domain
    dated = _optional(lambda: _dated_area_groups(cfg, ctx, los_up_map), "dated_mined_areas") or {}
    for key, d in dated.items():
        groups[key] = d["ix"]
    for f in cands:                                           # a grid per candidate; only the listed need one now
        if not any(f is g for g in listed):
            f.pop("_mask", None)
    t_from = float(t[-1] - RECENT_YEARS) if span >= MIN_RECENT_SPAN else None
    window = None
    if t_from is not None:
        w0 = int(np.argmax(t >= t_from))
        window = [ctx.dates[w0], ctx.dates[-1]]
    pl_series, r2 = _stream(ctx, groups, t_from) if (groups or t_from is not None) else ({}, None)
    r2_cm = r2_null = None
    if r2 is not None:
        r2_cm, _ = sg.class_matched_field(r2, ctx.tc, ctx.far)
        pf = getattr(sg, "placement_field", None)
        r2_null = pf(r2, ctx.tc, ctx.far, ctx.far_wide, getattr(ctx, "aoi", aoi)) if pf is not None else r2_cm
        del r2
    years_rows = _optional(lambda: _years_since_mining(ctx, dated, pl_series, far), "years_since_mining") \
        if dated else None

    # ---- the per-point yardstick on this farmland: the cells standing out
    # (E14) and the single-point uncertainty the rasters carry (E18)
    cell_null = _cell_null(cfg, ctx)
    points = _optional(lambda: _point_figures(cfg, ctx, cell_null, m_full, los_up_map, bg_u if anchored else None),
                       "point_figures") if m_full is not None else None

    # ---- the report areas
    res = _optional(lambda: _module("resolution").effective_resolution(cfg, ctx.attrs), "resolution")
    areas = []
    for name, spec in areas_spec:
        m, full = masks[name], spec.lower() == "full"
        cov = coverage(m, fps[name], imaged, px_km2)
        # an area narrower than ~2x what each value averages over cannot be
        # resolved by it (resolution.area_caveat; generality-2)
        cov["resolution_caveat"] = (None if full or not res else _optional(
            lambda: _module("resolution").area_caveat(res, *_module("resolution").extent_m(fps[name], ctx.attrs)),
            "resolution_caveat"))
        up = float(np.nanmedian(los_up_map[m])) if m.any() else up_med
        if name not in series:
            areas.append(area_entry(name, spec, "full" if full else "area", cov, None, None, up, bg_v, bg_u, span, t))
            continue
        summ = ctx.summarise(series[name], m, far, significance=False)
        if not full:
            # the farmland test with its diagnostic nulls (E6) and per-sign
            # limits (E5): summarise's own call does not pass the coherence
            # and the grid, so the local / coherence-matched nulls need this one
            summ["significance"] = sg.assess(ctx.r_null, ctx.far_wide, m, tc=ctx.tc, attrs=ctx.attrs)
        ai = [n_ for n_, _ in areas_spec].index(name)
        R = [] if full else _placement_series(ctx, groups, pl_series, f"__ap_{ai}_", far, up)
        # the total over the record with the farmland null of its change of
        # pace (E10): the whole area has no copy of itself, so rate x span
        summ = measured_change(t, summ, up, R)
        rec = None
        if r2_cm is not None:
            if full:
                x = r2_cm[m]
                x = x[np.isfinite(x)]
                rec = {"value_mm_yr": float(x.mean()) if x.size else None, "window": window,
                       "spread_mm_yr": _block_sd(r2_cm, m)}
            elif tested_fields(summ.get("significance"), up, cov)["testable"]:
                rec = _recent_of(r2_cm, r2_null, ctx.far_wide, m, window, sg)
        block = _block_sd(ctx.r_cm / los_up_map, m) if full else None
        cg_sd = (ground_terms or {}).get("sd_mm_yr") if full else None
        full_sd = (float(np.hypot(block, cg_sd or 0.0)) if block is not None else None) if full else None
        a = area_entry(name, spec, "full" if full else "area", cov, summ, summ.get("significance"), up,
                       bg_v, bg_u, span, t, recent=rec, full_rel_sd=full_sd)
        a["land_cover"] = ri._land_cover(ctx, m, ctx.r_cm)       # wooded share (landcover, 15 Oct 2026)
        if not full:
            a.update(series_noise(R, summ, up, t, ground))
        if full:
            a["relative_block_sd_mm_yr"] = round(block, 2) if block is not None else None
            a["comparison_ground"] = ground_terms
            pt = Q["findings"].get("monitoring_point") or {}
            ex = pt.get("extreme") or {}
            # the quality assessment's calibrated stand-out with the largest
            # gap (E7); a QA from before 7 Oct 2026 (the uncalibrated gap > 2
            # rule) gives none
            so = so_q[0] if so_q else None
            a["standout"] = ({"lat": so["lat"], "lon": so["lon"], "scale_km": so["scale_km"],
                              "vertical_relative_mm_yr": round(so["rate_relative_mm_yr"] / up, 1),
                              "vertical_mm_yr": round(so["rate_relative_mm_yr"] / up + (bg_v or 0.0), 1),
                              "next_deepest_vertical_mm_yr": round(so["deepest_other_farmland_mm_yr"] / up, 1),
                              "gap_in_spreads": so["gap_in_spreads"], "gap_threshold": so.get("gap_threshold"),
                              "p_calibrated": so.get("p_calibrated"),
                              # H1: the QA's own calibration flagged (None: sound)
                              "calibration_doubt": so.get("calibration_doubt"),
                              "in_mapped_mine": bool(so.get("inside_mapped_panels") or so.get("within_panel_buffer")),
                              # the two flags apart, so the KMZ can say inside / next to /
                              # outside every mapped mine and its buffer (gap-geolocation-1)
                              "inside_mapped_panels": bool(so.get("inside_mapped_panels")),
                              "within_panel_buffer": bool(so.get("within_panel_buffer")),
                              "buffer_m": (Q.get("regions") or {}).get("buffer_m"),
                              # the listed feature whose patch holds THIS stand-out (1 =
                              # Feature 1), or None - not the first feature holding any
                              # stand-out (Site B: the 1 km one is in Feature 2, the
                              # 2 km one in Feature 1)
                              "feature": _listed_feature_holding(ctx, listed, so),
                              "signal_to_noise": round(abs(so["rate_relative_mm_yr"]) / so["farmland_spread_1sigma_mm_yr"], 1)
                              if so.get("farmland_spread_1sigma_mm_yr") else None}
                             if so and all(so.get(k) is not None for k in ("rate_relative_mm_yr",
                                                                           "deepest_other_farmland_mm_yr"))
                             else None)
            cells = (points or {}).get("cells") or {}
            rel1 = pt.get("rate_relative_1km_mm_yr")
            chance1 = ex.get("expected_deepest_by_chance_mm_yr")
            a.update({"typical_relative_vertical_mm_yr": round(float(np.nanmedian(ctx.r_cm[m])) / up, 2),
                      # the single cells standing out, on this farmland (E14)
                      "cells_standing_out_percent": cells.get("cells_standing_out_percent"),
                      "cells_standing_out_by_chance_percent": cells.get("cells_standing_out_by_chance_percent"),
                      "cells_standing_out_rule": cells.get("rule"),
                      # F7(a): 'the same as chance' only inside this range
                      "cells_standing_out_chance_range95_percent": cells.get("chance_range95_percent"),
                      "deepest_patch": {"lat": pt.get("lat"), "lon": pt.get("lon"),
                                        "vertical_mm_yr": (round(rel1 / up + (bg_v or 0), 1) if rel1 is not None else None),
                                        "chance_vertical_mm_yr": (round(chance1 / up + (bg_v or 0), 1)
                                                                  if chance1 is not None else None),
                                        # the page compares the two relative to farmland
                                        "relative_vertical_mm_yr": round(rel1 / up, 1) if rel1 is not None else None,
                                        "chance_relative_vertical_mm_yr": (round(chance1 / up, 1)
                                                                           if chance1 is not None else None),
                                        "stands_out": ex.get("stands_out"), "p": ex.get("p_selection_corrected")}})
        areas.append(a)

    # ---- ground on the mapped mine layers
    mine_ground = []
    for L in layers:
        k = f"__mine_{L['key']}"
        m = extra[k]
        cov = coverage(m, poly[L["key"]] & aoi, imaged, px_km2)
        base = {"key": L["key"], "label": L["label"], **cov,
                "land_cover": ri._land_cover(ctx, poly[L["key"]] & aoi, ctx.r_cm)}
        if k not in series or m.sum() < 30:
            mine_ground.append({**base, "note": "too few measured points", "testable": False,
                                "not_testable_reason": "too few measured points", "detected": None,
                                "relative_vertical_mm_yr": None}); continue
        summ = ctx.summarise(series[k], m, far, significance=False)
        # its verdict is printed (Observations): the same test as an area's,
        # with its diagnostic nulls (E6; release reads them here too)
        summ["significance"] = sg.assess(ctx.r_null, ctx.far_wide, m, tc=ctx.tc, attrs=ctx.attrs)
        tf = tested_fields(summ.get("significance"), up_med, cov)
        # its 95% range on the test's frame, as a report area's (M4, final
        # check of 7 Oct 2026: the page printed this average with none)
        mrng, mbasis = mine_ground_range(tf, summ.get("significance"), up_med, float(summ["rate_mm_yr"]) / up_med)
        mine_ground.append({**base, "relative_vertical_mm_yr": round(summ["rate_mm_yr"] / up_med, 2), **tf,
                            "range95_mm_yr": mrng, "range_untested_basis": mbasis})

    # ---- the features
    features = []
    for i, f in enumerate(listed):
        k = f"__feature_{i}"
        if k not in series:
            continue
        summ = ctx.summarise(series[k], extra[k], far, significance=True)
        R = _placement_series(ctx, groups, pl_series, f"__pl_{i}_", far, up_med)
        summ = measured_change(t, summ, up_med, R)                # E10
        null_yearly = None
        if R:
            yr = ri.yearly_rates(t, np.vstack(R), years)
            if yr:
                ys_ = sorted(yr)
                null_yearly = {"years": ys_, "rates": np.column_stack([yr[y] for y in ys_])}
        rec = _recent_of(r2_cm, r2_null, ctx.far_wide, extra[k], window, sg) if r2_cm is not None else None
        fe = feature_entry(f, summ, summ.get("significance"), up_med, span, t, years, null_yearly, rec)
        fe["yearly_null_placements"] = len(R)
        fe.update(series_noise(R, summ, up_med, t, ground))
        fe["control_ground_label"] = ground
        fe["ground_reshaped"] = _reshaped(dh, f["_mask"] & ctx.good, ctx.far)
        # E1: the DEM error less what the feature's movement explains
        # (report_insights.dem_change on the report's own field)
        fe["dem_change"] = ri.dem_change(dsm, sm.get("field"), f["_mask"] & ctx.good, ctx.far, DEM_CHANGE_SMOOTH_M)
        fe["ground_reshaped_since_dem"] = ri.reshaped_since_dem(fe)
        fe["ring"] = _outline(f["_mask"], ctx.attrs)
        ch = fe.get("chance") or {}
        fe["selection_p"] = ch.get("p")
        fe["chance_level_mm_yr"] = ch.get("chance_level_mm_yr")
        if fe.get("stands_out") is None:
            fe["cause_hint"] = ""                    # untested: no cause, no story
        else:
            fe["cause_hint"] = ri.cause_hint((fe["mines"].get("key") or "none"), fe["sign"],
                                             (fe.get("trend") or {}).get("shape"),
                                             reshaped_since_dem=fe["ground_reshaped_since_dem"])
        features.append(fe)
    for n_, fe in enumerate(features, 1):
        fe["text"] = ri.feature_text(n_, fe)
    brief = lambda f: {k: f.get(k) for k in ("sign", "area_km2", "mean_rel_vertical_mm_yr", "beyond_farmland_mm_yr",  # noqa: E731
                                             "lat", "lon", "bbox", "quality_standout")} | {"selection_p": (f.get("chance") or {}).get("p")}

    ref_choice = {}
    try:
        ref_choice = json.loads((cfg.stack_dir / "reference_choice.json").read_text())
    except (OSError, ValueError, AttributeError):
        pass
    rp = Q["checks"].get("rate_precision") or {}
    used = [s["site"] for s in tie.get("stations", []) if s.get("used_for_background")]
    # the home folder as '~': the record goes to the client in the package's technical/
    # folder, and the full path carried the operator's login name (2 Nov 2026, DAT-07)
    from .provenance import portable
    out = {"project": portable(str(P)), "generated": datetime.now().isoformat(timespec="minutes"),
           "title": ident["title"], "location": ident["location"],
           "route": _route(cfg), "backend": cfg.backend,
           "mine_layers": [L["label"] for L in layers], "mine_ground": mine_ground,
           "features": features, "features_assessed": features_assessed,
           "features_reason": None if features_assessed else sm.get("reason"),
           "features_within_chance": [brief(f) for f in within],
           "features_beyond_listed": [brief(f) for f in beyond_list],
           "feature_search": {k: v for k, v in null.items()},
           # F1: the first pass's leading candidates, each re-tested with
           # itself left out of the farmland (technical record)
           "feature_leave_out": leave_out,
           "map_band_mm_yr": round(float(sm["band_mm_yr"]), 1) if sm.get("band_ok") else None,
           "map_band": ({"centre_mm_yr": round(sm["centre_mm_yr"], 2), "low_mm_yr": round(sm["band_lo_mm_yr"], 2),
                         "high_mm_yr": round(sm["band_hi_mm_yr"], 2),
                         "farmland_beyond_low_percent": sm.get("farmland_beyond_low_percent"),
                         "farmland_beyond_high_percent": sm.get("farmland_beyond_high_percent"),
                         "basis": sm.get("band_basis")} if sm.get("band_ok") else None),
           "map_smoothing_m": sm["smoothing_m"],
           "corrections": ri.corrections_applied(cfg, getattr(ctx, "ts_file", None)),
           "reference_auto": bool(getattr(cfg, "reference_auto", False) or (ref_choice.get("configured") is None and ref_choice.get("chosen"))),
           "checks_passed": cnt["passed"], "checks_flagged": cnt["flagged"], "checks_untested": cnt["untested"],
           "checks_applicable": cnt["applicable"], "checks_not_applicable": cnt["not_applicable"],
           "anchored": anchored, "gnss_stations_used": used, "gnss_reason": None if anchored else gnss_reason,
           "dates": [ctx.dates[0], ctx.dates[-1]], "n_dates": len(ctx.dates), "span_years": round(span, 2),
           "recent_window": window,
           "track": S.get("track"), "orbit": S.get("orbit_direction"), "look_angle_deg": S.get("look_angle_deg"),
           "points": S.get("total_measured_points"), "coverage_percent": (Q["checks"].get("coverage") or {}).get("aoi_percent_at_threshold"),
           "cell_repeatability_mm_yr": rp.get("empirical_median_mm_yr"), "verdict": Q.get("verdict"),
           "cell_null_by_class": cell_null, "los_up_median": round(up_med, 4),
           # the one per-point uncertainty (E18): the uncertainty raster's own
           # value over the study area, not the split-half repeatability
           "single_point_uncertainty_1sigma_mm_yr": (points or {}).get("single_point_uncertainty_1sigma_mm_yr"),
           "single_point_uncertainty_basis": (points or {}).get("single_point_uncertainty_basis"),
           "gnss": {**tie, "anchored": anchored, "reason": None if anchored else gnss_reason, "stations_used": used,
                    # why gnss_tie.check refused the file (technical, for the reviewer)
                    **({"tie_not_used": gnss_why} if not tie and gnss_why else {})},
           "areas": areas, "dates_list": ctx.dates, "n_areas": len(areas_spec),
           # woods left out of every figure (landcover.drop_wooded, 21 Oct 2026)
           "wooded_masked": getattr(ctx, "wooded_mask", None)}
    out["farmland_second_pass"] = second
    # the report areas kept out of the comparison ground (E28)
    out["report_areas_out_of_farmland"] = out_areas
    # whether the simulated copies are as strong as the real farmland (E8,
    # contract C4): release lists 'feature_null_misspecified' when not
    out["feature_null_check"] = (null.get("feature_null_check") if null.get("tested")
                                 else _optional(lambda: ri.feature_null_check(null), "feature_null_check"))
    out.update(_record_blocks(cfg, Q, S, ref_choice, features))
    if years_rows is not None:
        # on the report's own farmland and vertical (E3), not panel_history's
        # first-pass line-of-sight table
        out.update(years_rows)
    if (points or {}).get("radar_cells") is not None:
        # E25: counted over the study area, not the whole geocoded rectangle
        out["independent_radar_measurements"] = points["radar_cells"]
        out["independent_radar_measurements_basis"] = points["radar_cells_basis"]
    out["send_rule"] = _send_rule(Q, _tie_for_send_rule(cfg, tie, gnss_why), areas)
    out["stale_inputs"] = _stale_inputs(cfg)
    out["control_ground_label"] = ground
    out["comparison_ground"] = cground
    mark_inside_checks(out)
    out["resolution"] = res
    out["version"] = _optional(lambda: _module("version").describe(cfg), "version")
    out.update(_presentation_blocks(cfg, ctx, out, Q, layers, masks, fps, aoi, px_km2, up_med))
    _record_chance_range(out)
    out["quality_notes"] = ri.quality_notes(
        _with_point_uncertainty(Q, out.get("single_point_uncertainty_1sigma_mm_yr"), up_med),
        {"anchored": anchored, "gnss_reason": gnss_reason, "control_ground_label": ground,
         "detection_limits_mm_yr": [a.get("detection_limit_mm_yr") for a in areas if a.get("kind") == "area"]},
        up_med)
    out["observations"] = ri.observations(out)
    out["credits"] = _optional(lambda: _module("credits").credits(cfg, out), "credits")
    # M3: a reissue says so, and what changed (the config's report_revision)
    out["report_revision"] = report_revision(cfg)
    # every sentence in the record as it will be printed: no '-0.00'
    # (report_insights.no_negative_zero) - the KMZ and the workbook quote
    # these strings verbatim - and "farmland" only where the comparison
    # ground is farmland (ri.say_ground)
    out = ri.say_ground(ri.tidy_text(out), ground)
    # before the release block, which reads it (release.omitted_items)
    out["omitted"] = _OMITTED.get()
    out["release"] = ri.say_ground(ri.tidy_text(release_block(cfg, out, Q)), ground)
    return out, ctx, masks, outlines


def _record_chance_range(rep: dict) -> None:
    """F7(a): the whole area's cells_standing_out_chance_range95_percent on
    the radar-cell count the page words its sentence with
    (independent_radar_measurements). _point_figures sets it from its own
    count (E25's radar cells); when that count could not be made, the page
    uses summary.json's count or the estimate, and the record carried no
    range - it now carries that count's range, with its basis."""
    from . import report_insights as ri
    n = rep.get("independent_radar_measurements")
    if not n or not hasattr(ri, "chance_share_range"):
        return
    for a in rep.get("areas") or []:
        chance = a.get("cells_standing_out_by_chance_percent")
        if a.get("kind") != "full" or chance is None or a.get("cells_standing_out_chance_range95_percent"):
            continue
        a["cells_standing_out_chance_range95_percent"] = ri.chance_share_range(chance, n)
        a["cells_standing_out_chance_range_basis"] = (
            f"95% binomial range of the chance share over {int(n):,} radar resolution cells ("
            + str(rep.get("independent_radar_measurements_basis") or "the count the page gives") + ")")


def _median_gap_days(dates: list) -> float | None:
    if len(dates) < 2:
        return None
    d = [datetime.strptime(x, "%Y%m%d") for x in sorted(dates)]
    return float(np.median([(b - a).days for a, b in zip(d[:-1], d[1:])]))


def _presentation_blocks(cfg, ctx, rep: dict, Q: dict, layers: list, masks: dict, fps: dict, aoi: np.ndarray,
                         px_km2: float, up: float) -> dict:
    """
    What the page, the figures and the one-pagers state about the record
    beyond the area figures - every one measured or read from the run, not
    typed for one site (generality-16, tests-6: '15 m', '40 m', '2 km', '12
    days', 'N'/'W' and 'Full Satellite Imaging' were fixed strings):

      full_area_name / display names  the names a client reads
      revisit_days            median spacing of the record's dates
      gradient_limit          stack_qc.gradient_limit_mm_yr: the fastest
                              difference between neighbouring looks the
                              radar can follow
      los_east_over_up        |east / up| of the LOS: how much east-west
                              movement reads as vertical (stats-16)
      independent_radar_measurements  summary.json's count, else estimated
                              from the cells and the look cell (labelled so)
      phase_linking           the windows the sequential estimator used
      mine_layer_info         each mine layer's source and vintage
      map_scale, map_patches  the map's colour bins and every coloured patch
                              inside a report area (report_insights)
      technical_methods       one paragraph for the technical package
    """
    from . import report_insights as ri
    out = {}
    names = parse_areas(cfg)
    full = next((n for n, s in names if s.lower() == "full"), None)
    out["full_area_name"] = full or "Whole study area"
    for a in rep["areas"]:
        a["display_name"] = ri.display_name(a["name"], a.get("spec", "")) if a.get("kind") != "full" else a["name"]
    out["revisit_days"] = _median_gap_days(list(ctx.dates))
    out["gradient_limit"] = _optional(lambda: _module("stack_qc").gradient_limit_mm_yr(
        float(ctx.attrs.get("WAVELENGTH", 0.05546576) or 0.05546576), list(ctx.dates)), "gradient_limit")
    out["los_east_over_up"] = _optional(lambda: _los_east_over_up(cfg, ctx), "los_east_over_up")
    S_n = rep.get("independent_radar_measurements")
    res = rep.get("resolution") or {}
    fa = next((a for a in rep["areas"] if a.get("kind") == "full"), None)
    if S_n is None and fa and fa.get("cells") and res.get("look_cell_m") and res.get("posting_m"):
        py, px = ctx.sg.pixel_metres(ctx.attrs)
        lc = float(np.prod(res["look_cell_m"]))
        if lc > 0:
            out["independent_radar_measurements"] = int(round(fa["cells"] * py * px / max(lc, py * px)))
            out["independent_radar_measurements_basis"] = (
                f"estimated: study-area cells x the {py:.0f} x {px:.0f} m map cell / the "
                f"{res['look_cell_m'][0]:.0f} x {res['look_cell_m'][1]:.0f} m look cell")
    method = str(getattr(cfg, "miaplpy_phase_linking_method", "") or "")
    if str(getattr(cfg, "backend", "")).lower() == "miaplpy":
        size, lag = int(getattr(cfg, "miaplpy_ministack_size", 10) or 10), int(getattr(cfg, "miaplpy_time_lag", 3) or 3)
        gap = out["revisit_days"] or 12.0
        seq = method.lower().startswith("sequential")
        out["phase_linking"] = {"method": method, "sequential": seq, "ministack_size": size, "time_lag": lag,
                                "window_images": size + lag if seq else None,
                                # the images a window spans, end to end, in months
                                "window_months": round((size + lag - 1) * gap / 30.44, 1) if seq else None}
    out["mine_layer_info"] = [{"key": L["key"], "label": L["label"], "source": L.get("source"),
                               "vintage": L.get("vintage")} for L in layers]
    sm = getattr(ctx, "report_field", None) or {}
    if sm.get("band_ok") and sm.get("field") is not None:
        fld = sm["field"]
        lo_b, hi_b = float(sm["band_lo_mm_yr"]), float(sm["band_hi_mm_yr"])
        ms = ri.map_scale(fld, aoi, lo_b, hi_b)
        if ms.get("peak_rc"):
            r, c = ms["peak_rc"]
            a = ctx.attrs
            ms["peak_lat"] = round(float(a["Y_FIRST"]) + (r + 0.5) * float(a["Y_STEP"]), 4)
            ms["peak_lon"] = round(float(a["X_FIRST"]) + (c + 0.5) * float(a["X_STEP"]), 4)
        ms["map_km2"] = round(float((aoi & np.isfinite(fld)).sum()) * px_km2, 1)
        bl, bh = sm.get("farmland_beyond_low_percent"), sm.get("farmland_beyond_high_percent")
        # the colour chance alone puts on this map: the share of ordinary
        # ground beyond each edge x the map's measured area (clientout-5:
        # "some coloured specks" was ~20 km2 of colour on Site B)
        ms["chance_km2"] = [round(ms["map_km2"] * float(bl or 0) / 100.0, 1),
                            round(ms["map_km2"] * float(bh or 0) / 100.0, 1)]
        ms["peak_in"] = _peak_patch(ms, rep)
        out["map_scale"] = ms
        regions = {a["name"]: fps[a["name"]] for a in rep["areas"] if a.get("kind") == "area" and a["name"] in fps}
        out["map_patches"] = ri.map_patches(fld, lo_b, hi_b, regions, rep.get("features") or [], px_km2, ctx.attrs)
        for pa in out["map_patches"]:
            # E12: listed under an area only when at least half of it is inside
            share = float(pa.get("inside_area_km2") or 0.0) / float(pa["area_km2"]) if pa.get("area_km2") else 0.0
            pa["inside_share"] = round(share, 2)
            pa["inside"] = bool(share >= PATCH_INSIDE_SHARE)
    else:
        out["map_scale"], out["map_patches"] = None, []
    out["technical_methods"] = technical_methods(cfg, rep, out)
    return out


def _peak_patch(ms: dict, rep: dict) -> dict | None:
    """Which patch the map's strongest cell lies in (finding 51i: Site A's
    'strongest movement on the map' sat in a patch listed only as within
    chance, and the text never said so): a listed feature {"feature": n},
    a patch the search found within chance {"within_chance": True}, or None
    (by the patches' boxes)."""
    lat, lon = ms.get("peak_lat"), ms.get("peak_lon")
    if lat is None or lon is None:
        return None

    def inside(b):
        return bool(b) and b[0] <= lon <= b[2] and b[1] <= lat <= b[3]
    for i, f in enumerate(rep.get("features") or [], 1):
        if inside(f.get("bbox")):
            return {"feature": i}
    if any(inside(f.get("bbox")) for f in rep.get("features_within_chance") or []):
        return {"within_chance": True}
    return None


def _los_east_over_up(cfg, ctx) -> float | None:
    """Median |los_east / los_up| over the measured cells: east-west ground
    movement of 1 reads as this much vertical (Site B 0.81)."""
    from . import deliverables as dl
    e = dl.geometry_field(cfg, "los_east")
    u = dl.geometry_field(cfg, "los_up")
    if e is None or u is None or np.shape(e) != ctx.good.shape or np.shape(u) != ctx.good.shape:
        return None
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.abs(e / u)[ctx.good]
    r = r[np.isfinite(r)]
    return round(float(np.median(r)), 3) if r.size else None


def technical_methods(cfg, rep: dict, blocks: dict) -> str:
    """One paragraph for the technical package and README (methodology-9 step
    5), not the client page: the estimator by name and version, its windows,
    the network exported, the unwrapping, and what the run recorded."""
    route = str(getattr(cfg, "backend", "") or "")
    parts = []
    ver = rep.get("version") or {}
    if route == "miaplpy":
        pl = blocks.get("phase_linking") or {}
        mv = ((ver.get("miaplpy") or {}).get("version") if isinstance(ver.get("miaplpy"), dict) else None)
        parts.append(f"Phase linking: MiaplPy{(' ' + mv) if mv else ''}, {pl.get('method')}"
                     + (f", mini-stacks of {pl.get('ministack_size')} images solved with the previous "
                        f"{pl.get('time_lag')} (window overlap; the patched library's datum fix keeps one datum "
                        "across mini-stacks)" if pl.get("sequential") else "")
                     + f", similar-pixel window {getattr(cfg, 'miaplpy_range_window', '?')} x "
                       f"{getattr(cfg, 'miaplpy_azimuth_window', '?')}; network exported with "
                       f"{getattr(cfg, 'miaplpy_network_connections', '?')} nearest connections per date.")
    elif route == "disp":
        parts.append("OPERA DISP-S1 displacement (screening route), re-referenced and put through the record's chain.")
    elif route == "nisar":
        parts.append("NISAR L2 GUNW interferograms (PROVISIONAL), one geometry, a chain of 12-day pairs read as "
                     "windows over the area and resampled to one 80 m latitude/longitude grid; the product's own "
                     "split-spectrum ionosphere screen subtracted (MintPy); no phase closure (a chain has no "
                     "triangles); reliability from the mean pair coherence (nisar_backend.py).")
    else:
        parts.append(f"Interferogram network ({route}) with {getattr(cfg, 'stack_range_looks', '?')} x "
                     f"{getattr(cfg, 'stack_azimuth_looks', '?')} looks.")
    res = rep.get("resolution") or {}
    if res.get("text"):
        parts.append(res["text"] + (f" ({res.get('source')})" if res.get("source") else ""))
    if rep.get("corrections"):
        parts.append("Corrections applied (as recorded by the run): " + "; ".join(rep["corrections"]) + ".")
    parts.append("Areas are compared with the comparison ground of the same coherence class (class-matched), "
                 "against empirical placement nulls; features are tested against amplitude-adjusted surrogates of "
                 "that ground under the same search.")
    return " ".join(parts)


def _read_json(p: Path) -> dict:
    try:
        d = json.loads(Path(p).read_text())
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


def _send_rule(Q: dict, tie: dict | None, areas: list | None = None) -> dict:
    """quality_assessment.send_rule re-evaluated WITH the GNSS tie (the QA runs
    before the tie, so the rule it stored never saw it). Only for a QA that
    stored a send rule itself: one written before the rule existed lacks the
    checks it reads (dates, geolocation, area testability), and re-running it
    there gave Site B "send with notes" where the current QA holds it for
    a specialist.

    `areas`: the report's areas. Each report area the QA judged
    (area_testability) is judged by the report's own second-pass testability
    and reason, so the rule and the area table agree (7 Oct 2026: Site B's
    rule said 'about 4 separate times' where its area and the release list
    said 'about 3'). The QA's record is not changed."""
    if not isinstance(Q.get("send_rule"), dict):
        return {"value": None, "reasons": ["the quality assessment predates the send rule; re-run it"]}
    at = Q.get("area_testability")
    if areas and isinstance(at, dict):
        import copy
        Q = copy.deepcopy(Q)
        for a in areas:
            if (not isinstance(a, dict) or a.get("kind") == "full"
                    or not isinstance(Q["area_testability"].get(a.get("name")), dict)):
                continue
            # 'kind' stays the QA's: send_rule holds only on kind 'area'
            Q["area_testability"][a["name"]].update({"testable": a.get("testable"),
                                                     "not_testable_reason": a.get("not_testable_reason")})
    got = _optional(lambda: _module("quality_assessment").send_rule(Q, gnss=tie or None), "send_rule")
    return got if isinstance(got, dict) else Q["send_rule"]


def _stale_inputs(cfg) -> list[str]:
    """Inputs the report takes numbers from that are OLDER than the velocity
    product they describe (tests-4): a quality assessment, summary or
    uncertainty model left by an earlier run would be printed beside new
    rates. Plain sentences; release.blockers makes the QA one a hard
    blocker."""
    try:
        from . import deliverables as dl
        vel = dl._product(cfg, "velocity.h5")
        vt = vel.stat().st_mtime
    except Exception as exc:                                  # noqa: BLE001 - no product to compare is not stale
        log.debug("Input freshness not checked: %s", exc)
        return []
    P = Path(cfg.project_dir)
    out = []
    for f, what in ((P / "quality_assessment.json", "the quality assessment"),
                    (P / "deliverables" / "summary.json", "the product summary (point counts, per-cell shares)"),
                    (P / "deliverables" / "uncertainty_model.json", "the per-cell uncertainty model")):
        if f.exists() and f.stat().st_mtime < vt:
            out.append(f"{what} ({f.name}) is older than the velocity product it describes; re-run it")
    for s_ in out:
        log.warning("Stale input: %s", s_)
    return out


def _current_record(cfg, rec: dict, option: str) -> bool:
    """crosscheck.record_is_current (option on, not older than the velocity
    product); a record that cannot be judged is not used."""
    try:
        return bool(_module("crosscheck").record_is_current(cfg, rec, option))
    except Exception as exc:                                  # noqa: BLE001 - not judged is not this run's
        log.warning("The %s record could not be checked against the velocity product (%s); not used", option, exc)
        return False


def _cross_check_for(cfg, cc: dict, features: list | None) -> dict:
    """A completed cross-check with its per-feature rows and sentence for
    THIS report's features (crosscheck.recheck_features): the stage runs
    before the report, so its record measured the previous report's
    features, and the page printed 'k of 3 reported features' beside a list
    of 2. No features: the clause is dropped without re-measuring. A recheck
    that fails drops the clause too - never the previous report's."""
    if features is None or cc.get("status") != "complete" or not cc.get("available"):
        return cc
    xc = _module("crosscheck")
    if features:
        try:
            got = xc.recheck_features(cfg, cc, features)
            if got is not cc:                                 # returned unchanged: not redone (no common window)
                return got
            log.warning("Cross-check not re-measured on this report's features; its features clause is left out")
        except Exception as exc:                              # noqa: BLE001 - the page keeps the rest of the check
            log.warning("Cross-check not re-measured on this report's features (%s); its features clause is "
                        "left out", exc)
    out = dict(cc, features=[])
    try:
        out["sentence"] = xc.sentence(out)
    except Exception as exc:                                  # noqa: BLE001 - no sentence rather than a wrong one
        log.warning("Cross-check sentence not rebuilt (%s)", exc)
        out["sentence"] = None
    return out


_CC_FEATURE_KEYS = ("index", "confirmed", "same_sign", "disp_distinguishable", "cells_from", "cells",
                    "official_relative_vertical_mm_yr", "disp_relative_vertical_mm_yr", "note")


def _not_current_reason(cfg, option: str, what: str) -> str:
    """Why a cross_check.json / injection_recovery.json on file was not used."""
    if not getattr(cfg, option, False):
        return (f"the {what} is switched off for this project ({option}: false); the record left on file is from "
                "an earlier run and was not used")
    return (f"the {what} record on file is not dated after the velocity product it would check (an earlier "
            "run's), so it was not used; re-run it")


def _combined_view(cfg) -> dict | None:
    """The combined project this one belongs to (combine.pointer_for: an
    ascending + descending view of the same ground, 28 Oct 2026), for the one
    line under Independent tests; None when there is none. Never fails the
    report."""
    try:
        return _module("combine").pointer_for(cfg)
    except Exception as exc:                                  # noqa: BLE001 - a pointer is not worth a failed report
        log.warning("Combined view not looked up (%s)", exc)
        return None


def _record_blocks(cfg, Q: dict, S: dict, ref_choice: dict, features: list | None = None) -> dict:
    """What the other stages measured about this record, carried into
    client_report.json so the page, the KMZ, the dashboard and the package
    print the same thing (the wording is the page's):

      record_dates     quality_assessment dates: available / processed /
                       dropped {date: reason} and its sentence
      geolocation      the QA's check against mapped water edges
      qa_flags         dem_error / plausibility flags (surface-mine sites)
      reference        the reference point: auto or configured, and its
                       MEASURED distance to the nearest mapped mine edge
                       (None: no mines file, or the configured one was kept)
      injection_test   deliverables/technical/injection_recovery.json, for the
                       technical record only (E4: no client sentence)
      independent_check deliverables/technical/cross_check.json (DISP-S1),
                       its feature rows and sentence redone for `features`
                       (this report's; _cross_check_for)
                       - each only when crosscheck.record_is_current; a
                       record that is not is 'out of date', never printed
      by_years_since_mining  None here: compute rebuilds the table on the
                       report's farmland, vertical (E3, _years_since_mining)
      independent_radar_measurements  summary.json (cells vs looks)
    """
    P = Path(cfg.project_dir)
    T = P / "deliverables" / "technical"
    geo = Q.get("geolocation") or {}
    chosen = ref_choice.get("chosen") or {}
    conf = ref_choice.get("configured")
    dist = chosen.get("distance_to_nearest_mapped_mine_km", (conf or {}).get("distance_to_nearest_mapped_mine_km"))
    buf = ref_choice.get("far_field_buffer_m", chosen.get("far_field_buffer_m"))
    # a record counts only when it is this run's (crosscheck.record_is_current:
    # option on, not older than the velocity product): an older one checked a
    # previous MintPy run, one left after the option was turned off is not this
    # run's (7 Oct 2026: both were printed as this record's)
    inj = _read_json(T / "injection_recovery.json")
    inj_old = bool(inj) and not _current_record(cfg, inj, "injection_test")
    cc = next((d for d in (_read_json(T / "cross_check.json"), _read_json(P / "deliverables" / "crosscheck.json")) if d), {})
    cc_old = bool(cc) and not _current_record(cfg, cc, "cross_check_disp")
    inj, cc = ({} if inj_old else inj), ({} if cc_old else _cross_check_for(cfg, cc, features))
    return {
        # the short-date reasons in the current wording (an older QA said
        # "the burst catalogue held fewer bursts" for search-rendering gaps)
        "record_dates": _module("quality_assessment").refresh_drop_reasons(cfg, Q.get("dates")),
        "geolocation": ({k: geo.get(k) for k in ("text", "flag", "offset_m", "status", "available")} if geo else None),
        "qa_flags": {k: bool((Q.get(k) or {}).get("flag")) for k in ("dem_error", "plausibility") if Q.get(k)},
        "reference": {"auto": bool(getattr(cfg, "reference_auto", False) or (conf is None and chosen)),
                      "lat": chosen.get("lat", (conf or {}).get("lat")), "lon": chosen.get("lon", (conf or {}).get("lon")),
                      "distance_to_nearest_mapped_mine_km": dist, "far_field_buffer_m": buf,
                      # a distance may be claimed only when it was measured and clears the buffer
                      "clears_buffer": (bool(dist * 1000.0 >= buf) if dist is not None and buf else None),
                      "rule": chosen.get("rule")} if ref_choice else None,
        # technical only (E4, contract C5): no client sentence - the page does
        # not print it; the record keeps what the reviewer needs
        "injection_test": ({"status": inj.get("status"), "sentence": None, "reason": inj.get("reason"),
                            "decision_rule": inj.get("decision_rule"),
                            "technical_summary": inj.get("technical_summary")} if inj else
                           {"status": "out of date", "sentence": None,
                            "reason": _not_current_reason(cfg, "injection_test", "injection-recovery test")} if inj_old else
                           {"status": "not run", "sentence": None, "reason": "the injection-recovery test was not run for this record"}),
        "independent_check": ({"available": bool(cc.get("available", cc.get("status") == "complete")),
                               "status": cc.get("status"), "sentence": cc.get("sentence"), "reason": cc.get("reason"),
                               "areas": [{k: a.get(k) for k in ("name", "agree", "combined_95_mm_yr", "note")}
                                         for a in cc.get("areas") or [] if isinstance(a, dict)],
                               # this report's features (Feature n = index + 1), as the sentence counts them
                               "features": [{k: f.get(k) for k in _CC_FEATURE_KEYS}
                                            for f in cc.get("features") or [] if isinstance(f, dict)]} if cc else
                              {"available": False, "status": "out of date", "sentence": None,
                               "reason": _not_current_reason(cfg, "cross_check_disp", "cross-check against OPERA DISP-S1")}
                              if cc_old else
                              {"available": False, "status": "not run", "sentence": None,
                               "reason": "no independent cross-check was run (cross_check_disp: false)"}),
        # panel_history's table is line of sight on the first farmland pass:
        # never the client's (E3) - compute rebuilds it on the report's frame
        "by_years_since_mining": None, "by_years_since_mining_note": None,
        # the combined ascending + descending project of this ground, if any
        # (combine.pointer_for): one line on the page, nothing more
        "combined_view": _combined_view(cfg),
        "independent_radar_measurements": S.get("independent_radar_measurements"),
        "independent_radar_measurements_basis": S.get("independent_radar_measurements_basis"),
    }


def _feature_search(ctx, los_up_map, poly, aoi, standout) -> tuple[dict, list, dict, bool]:
    """(the map's smoothed field, every candidate the search finds - each
    with its selection-corrected `chance` and `stands_out` - the farmland
    null, whether the search could be made at all).

    `stands_out` is the calibrated chance test alone (E7, trust review 7 Oct
    2026; contract C4): until then a patch holding the quality assessment's
    stand-out was listed - and cut from the farmland - whatever its chance
    (a Gaussian null field gave a 1.6 km2 'stands out clearly' patch the
    chance test scored p = 1.0). `standout`: the QA's calibrated stand-outs
    (_calibrated_standouts), which set each candidate's quality_standout;
    selection_p sets stands_out_clearly from both tests and the null check."""
    from . import report_insights as ri
    sm = ri.smoothed_field(ctx, los_up_map)
    cands = ri.find_features(ctx, sm, poly, aoi, standout, limit=None)
    assessed = cands is not None
    cands = cands or []
    null = (ri.feature_null(ctx, sm, cands, aoi) if assessed
            else {"tested": False, "reason": sm.get("reason")})
    sm.pop("null_field", None)                                # the null is measured; the map needs only the field
    for f in cands:
        f["chance"] = ri.selection_p(f, sm, null)
        f["stands_out"] = f["chance"]["stands_out"]
    return sm, cands, null, assessed


# what panel_history.exclude_from_farmland / measure_farmland replace on the
# context (each is re-assigned, never changed in place, so holding the old
# objects is the snapshot)
_FARMLAND_STATE = ("far", "far_wide", "r_cm", "r_null", "r_place", "class_means", "class_tilt", "farmland_pass")


def _leave_candidate_out(ctx, sm: dict, cands: list, null: dict, los_up_map, poly, aoi, standout) -> dict:
    """
    F1 (follow-up of 7 Oct 2026; E28's principle - nothing is compared with
    itself): the first pass's leading candidates (report_insights.
    leading_candidates: the LEAVE_OUT_LEADING strongest by cluster mass, at
    the map's one search scale) are each re-tested against farmland with
    THAT candidate (+ FARMLAND_SECOND_PASS_M) taken out - class centres,
    band, field and the simulated copies all measured again on it
    (_feature_search) - under the unchanged selection-corrected rule (its
    strength against the copies' maxima). A candidate already beating
    chance on the first pass is not re-run (it is cut anyway); its
    first-pass figures are recorded. Nor is one that does not reach the
    farmland even with its buffer (_touches_farmland): left out, the
    farmland and so the test are the first pass's exactly, and its
    first-pass figures are its left-out result. The context's farmland is put back
    after each test. Sets f["leave_out"] on every leading candidate;
    returns the record client_report.json keeps as feature_leave_out:
    {tested, leading, buffer_m, smoothing_m, rule, candidates: [{sign, lat,
    lon, area_km2, first_pass: {strength, copies_as_strong, copies,
    copies_percentile, p}, left_out: leave_out_result | None, passed,
    basis}], passed, retested}.

    G2 (calibration of 7 Oct 2026): `retested` L is the number of leading
    candidates given a leave-out test (those that did not beat chance on
    the first pass and could be tested on it, the ones outside the farmland
    included; at most LEAVE_OUT_LEADING), counted before any is run; each
    passes only when min(1, L x p) <= SELECTION_ALPHA / 2 (leave_out_result
    with retested=L, which records p, L and p_adjusted). Uncorrected, these
    re-tests listed a feature in 5.6% of pure-noise fields at 540 m
    correlation, 1.9% without them.

    Site A (trust-check copy, 199 copies): Feature 1 (38.241 N 87.647 W)
    scored 88.6 on the first pass with 24 of 199 copies as strong - tested
    against farmland that still held it; left out, 115.4 against copies up
    to 96.3 (p 0.005; 3 re-tested, adjusted 0.015, "about 3 in 200"): it
    passes, is cut, and the second pass lists it (23.7 km²). Site B's
    2.1 km² patch, re-tested with the 6.0 km² feature still in the
    farmland, stays within chance (p 0.14; 2 re-tested, adjusted 0.28); the
    second pass lists it by its own test (G1: 4 of 199 copies, p 0.025).
    """
    from . import report_insights as ri
    rec = {"tested": False, "leading": ri.LEAVE_OUT_LEADING, "buffer_m": float(FARMLAND_SECOND_PASS_M),
           "smoothing_m": sm.get("smoothing_m"), "candidates": [], "passed": 0,
           "rule": ("each of the first pass's strongest candidates (cluster mass) that does not beat chance on the "
                    "first pass is re-tested on farmland with that candidate taken out, the simulated copies built "
                    "from that farmland; with L candidates re-tested, one passes when min(1, L x p) <= "
                    "SELECTION_ALPHA / 2 a side, p = (1 + copies as strong) / (1 + copies), compared in whole "
                    "numbers (G1, G2)")}
    if not cands or not null.get("tested"):
        rec["reason"] = ("no candidate" if not cands else
                         null.get("reason") or "the first pass's chance test could not be made")
        return rec
    rec["tested"] = True
    leading = ri.leading_candidates(cands, sm)
    # G2: L, the re-tests to be made, fixed before any is run
    L = sum(1 for f in leading if f.get("stands_out") is False)
    rec["retested"] = L
    for f in leading:
        ch = f.get("chance") or {}
        side = (null.get("sides") or {}).get("sinking" if f["sign"] < 0 else "rising") or {}
        r = {"sign": f["sign"], "lat": f["lat"], "lon": f["lon"], "area_km2": f["area_km2"],
             "first_pass": {"strength": ch.get("cluster_mass"), "copies_as_strong": ch.get("copies_as_strong"),
                            "copies": ch.get("copies"),
                            "copies_percentile": ri.copies_percentile(ch.get("cluster_mass"), side.get("max_mass")),
                            "p": ch.get("p")},
             "left_out": None}
        if f.get("stands_out"):
            r.update(passed=True, basis="beats chance on the first pass, with itself in the farmland")
        elif f.get("stands_out") is None:
            r.update(passed=False, basis="not tested on the first pass: " + str(ch.get("reason") or "no null"))
        elif not _touches_farmland(ctx, f["_mask"], FARMLAND_SECOND_PASS_M):
            # nothing of it, buffer included, is farmland (on a mapped mine
            # or a report area): taking it out leaves the farmland - so the
            # class centres, band and copies (fixed seed) - exactly the first
            # pass's, and the re-test would repeat it; its figures are the
            # result (Site A's third candidate: one null, ~2 minutes)
            got = ri.leave_out_result(f, sm, null, retested=L)
            r.update(left_out=got, passed=bool(got.get("passed")),
                     basis="outside the farmland, buffer included: left out, the test is the first pass's")
        else:
            saved = {k: getattr(ctx, k) for k in _FARMLAND_STATE if hasattr(ctx, k)}
            try:
                _exclude_from_farmland(ctx, [f["_mask"]], FARMLAND_SECOND_PASS_M)
                sm2, cands2, null2, _ = _feature_search(ctx, los_up_map, poly, aoi, standout)
                got = ri.leave_out_result(ri.match_candidate(f, cands2), sm2, null2, retested=L)
                del sm2, cands2, null2
            finally:
                # put back exactly what was there: an attribute the context
                # did not have (farmland_pass on a light context) is removed
                for k in _FARMLAND_STATE:
                    if k in saved:
                        setattr(ctx, k, saved[k])
                    elif hasattr(ctx, k):
                        delattr(ctx, k)
                del saved
            r.update(left_out=got, passed=bool(got.get("passed")),
                     basis="re-tested with itself (+ {:.0f} m) left out of the farmland".format(FARMLAND_SECOND_PASS_M))
            log.info("Leave-candidate-out (F1): %.1f km2 near %.3f, %.3f - strength %s against copies up to %s, "
                     "p %s, x %d re-tested = %s: %s", f["area_km2"], f["lat"], f["lon"], got.get("strength"),
                     got.get("copies_max"), got.get("p"), L, got.get("p_adjusted"),
                     "passes" if r["passed"] else "within chance")
        f["leave_out"] = r
        rec["candidates"].append(r)
    rec["passed"] = sum(1 for r in rec["candidates"] if r["passed"])
    return rec


def _touches_farmland(ctx, mask: np.ndarray, buffer_m: float) -> bool:
    """Whether mask grown by buffer_m (the same growth panel_history.
    exclude_from_farmland cuts with) reaches the in-box or the wide
    farmland: if not, cutting it changes neither."""
    grown = ctx.sg.euclidean_buffer(mask, ctx.attrs, buffer_m)
    return bool((grown & ctx.far).any() or (grown & ctx.far_wide).any())


def mask_runs(mask: np.ndarray) -> dict:
    """A boolean grid as row runs, {"shape": [rows, cols], "runs": [[row,
    first col, last col + 1], ...]}: exact, and a few hundred numbers for a
    feature-sized cut (JSON-safe, so client_report.json can carry it)."""
    m = np.asarray(mask, bool)
    runs = []
    for r in np.flatnonzero(m.any(axis=1)):
        d = np.diff(np.concatenate(([0], m[r].view(np.int8), [0])))
        runs.extend([int(r), int(a), int(b)] for a, b in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))
    return {"shape": [int(m.shape[0]), int(m.shape[1])], "runs": runs}


def runs_mask(rec: dict | None, shape) -> np.ndarray | None:
    """mask_runs back to a grid; None when absent or for another grid."""
    if not isinstance(rec, dict) or list(rec.get("shape") or []) != [int(shape[0]), int(shape[1])]:
        return None
    m = np.zeros(tuple(shape), bool)
    for r, a, b in rec.get("runs") or []:
        m[int(r), int(a):int(b)] = True
    return m


def _exclude_from_farmland(ctx, masks: list, buffer_m: float) -> dict:
    """
    The second pass of the farmland definition (stats-3): movement FOUND in
    the farmland (features that beat chance) is taken out of it, grown by
    buffer_m, and everything measured against farmland is measured again.
    Only features that beat chance are cut: removing chance-level patches
    too would trim ordinary farmland of its own extremes and narrow every
    null. The one definition is panel_history.exclude_from_farmland (the
    same code _Context.__init__ measures the first pass with); returns what
    changed, for client_report.json.
    """
    from .panel_history import exclude_from_farmland
    return exclude_from_farmland(ctx, masks, buffer_m)


def _calibrated_standouts(Q: dict) -> list:
    """The quality assessment's stand-outs that passed its calibrated
    multi-scale rule, every scale with its own location (E7, contract C4:
    quality_assessment.calibrated_standouts - Site B's 2 km stand-out lies
    in a different feature from its 1 km one). [] for a QA written before 7
    Oct 2026, whose single 'standout' came from the uncalibrated gap > 2 rule
    and supports no 'stands out clearly'.

    Each carries `calibration_doubt` (H1, final check of 7 Oct 2026;
    report_insights.qa_standout_doubt): why the QA's own calibration cannot
    confirm a feature - its copies flagged as not varying like the real
    farmland, or built from farmland still holding the candidate - or None.
    A feature holding only doubted stand-outs is at most "worth a look"."""
    from . import report_insights as ri
    got = _optional(lambda: _module("quality_assessment").calibrated_standouts(Q), "standouts")
    cal = (((Q or {}).get("findings") or {}).get("monitoring_point") or {}).get("standout_calibration")
    doubt = ri.qa_standout_doubt(cal)
    return [{**s, "calibration_doubt": doubt} for s in got or []]


def _listed_feature_holding(ctx, listed: list, so: dict) -> int | None:
    """The number (1 = Feature 1) of the listed feature whose patch holds the
    stand-out's cell - the same cell report_insights.find_features matches
    it by - or None."""
    try:
        rc = ctx.sg.lalo_to_rc(ctx.attrs, float(so["lat"]), float(so["lon"]))
    except (KeyError, TypeError, ValueError):
        return None
    for i, f in enumerate(listed, 1):
        mk = f.get("_mask")
        if (mk is not None and 0 <= rc[0] < mk.shape[0] and 0 <= rc[1] < mk.shape[1]
                and bool(mk[rc[0], rc[1]])):
            return i
    return None


def _farmland_state(ctx) -> dict:
    """The comparison ground's size and class centres now (for the record of
    what the cuts changed)."""
    kcls = getattr(ctx, "kcls", None)
    kcls = ctx.sg.coherence_class(ctx.tc) if kcls is None else kcls
    by = {int(k): int(n) for k, n in zip(*np.unique(kcls[ctx.far], return_counts=True)) if k >= 0}
    return {"far": int(ctx.far.sum()), "far_wide": int(ctx.far_wide.sum()), "by_class": by,
            "centres": {int(k): float(v) for k, v in ctx.class_means.items() if isinstance(k, (int, np.integer))}}


def _pass_record(s0: dict, s1: dict) -> dict:
    """What the cuts changed from the first pass, in panel_history.
    exclude_from_farmland's terms (results.report_farmland applies
    class_centre_shift_mm_yr on top of a fresh first pass)."""
    c0, c1 = s0["by_class"], s1["by_class"]
    return {"far_cells": [s0["far"], s1["far"]], "far_wide_cells": [s0["far_wide"], s1["far_wide"]],
            "far_cells_by_class": {str(k): [c0.get(k, 0), c1.get(k, 0)] for k in sorted(set(c0) | set(c1))},
            "classes_measured": [sorted(s0["centres"]), sorted(s1["centres"])],
            "class_centre_shift_mm_yr": {str(k): round(v - s0["centres"][k], 3) for k, v in s1["centres"].items()
                                         if k in s0["centres"]}}


def _exclude_report_areas(ctx, area_cut: np.ndarray, names: list) -> dict:
    """
    E28 (trust review 7 Oct 2026; contract C2): the report areas other than
    the whole study area (`area_cut`, the union of their outlines) are taken
    out of the comparison ground - class centres, nulls, the feature search's
    farmland, the recent windows - grown by FARMLAND_SECOND_PASS_M (the width
    significance.far_field_mask(exclude=...) grows them by), so an area is
    never compared with itself. Done only when they touch it; returns
    {areas, buffer_m, far_cells_removed, far_wide_cells_removed, applied,
    areas_cut, class_centre_shift_mm_yr (when applied)}.

    One exception, the technical records' own (F7(d), follow-up of 7 Oct
    2026: quality_assessment.report_farmland_masks): when the cut would
    leave the in-box farmland under quality_assessment.CUT_MIN_CELLS cells,
    the report areas cover (nearly) all of it and not even a class centre
    is left; the areas are not cut (report_areas_not_cut, with the reason),
    so the report and the QA, DISP cross-check, per-cell model and GNSS
    check read the same farmland, and the QA's thin far field holds the
    report. Until then the report cut them anyway and compute stopped on
    the empty farmland (KeyError '__far__' in panel_history's series).
    """
    sg = ctx.sg
    out = {"areas": list(names), "buffer_m": float(FARMLAND_SECOND_PASS_M), "applied": False, "areas_cut": [],
           "far_cells_removed": 0, "far_wide_cells_removed": 0}
    if not area_cut.any():
        return out
    grown = sg.euclidean_buffer(area_cut, ctx.attrs, FARMLAND_SECOND_PASS_M)
    hit, hit_w = int((grown & ctx.far).sum()), int((grown & ctx.far_wide).sum())
    left = int((ctx.far & ~grown).sum())
    del grown
    out.update({"far_cells_removed": hit, "far_wide_cells_removed": hit_w})
    if not (hit or hit_w):
        out["note"] = "no report area touches the comparison ground"
        return out
    min_cells = int(getattr(_module("quality_assessment"), "CUT_MIN_CELLS", 0) or 0)
    if hit and left < min_cells:
        out.update({"far_cells_removed": 0, "far_wide_cells_removed": 0, "report_areas_not_cut": list(names),
                    "reason": (f"the report areas ({', '.join(names)}) cover nearly all the farmland: {left} "
                               f"farmland cells would be left without them, so they were not cut (as the quality "
                               "assessment) and the far field is too thin to test them against")})
        log.warning("Report areas NOT kept out of the comparison ground: %s", out["reason"])
        return out
    rec = _exclude_from_farmland(ctx, [area_cut], FARMLAND_SECOND_PASS_M)
    out.update({"applied": True, "areas_cut": list(names),
                "class_centre_shift_mm_yr": rec.get("class_centre_shift_mm_yr")})
    log.info("Report areas kept out of the comparison ground (E28): %d farmland cells, %d wide",
             hit, hit_w)
    return out


def _comparison_ground_variants(ctx, far_first: np.ndarray, cut: np.ndarray, m: np.ndarray,
                                up_map: np.ndarray) -> dict | None:
    """
    The whole study area's relative figure with the comparison ground cut
    at each of COMPARISON_GROUND_BUFFERS_M (E13, trust review 7 Oct 2026):
    everything cut from the first-pass farmland (`cut`: the report areas and
    the features that beat chance) grown by 0, 500 and 1000 m, the class
    centres measured again each time (significance.class_means), and the
    area's mean of the class-matched rate (vertical, the field the whole
    area's relative figure averages). Their sd is the comparison-ground
    term added to the whole area's relative uncertainty: the zero it is
    measured against depends on how wide the cut is (Site A: -0.55 /
    -0.835 / -0.97 mm/yr). Nothing cut: sd 0 - there is no cut to vary.
    """
    sg = ctx.sg
    kcls = getattr(ctx, "kcls", None)
    kcls = sg.coherence_class(ctx.tc) if kcls is None else kcls
    sel = m & np.isfinite(ctx.vel) & (kcls >= 0) & np.isfinite(up_map) & (up_map > 0)
    n = int(sel.sum())
    if n < 1000:
        return None
    k = kcls[sel]
    w = 1.0 / up_map[sel]
    s1 = np.bincount(k, weights=ctx.vel[sel] * w)
    s2 = np.bincount(k, weights=w)
    del k, w
    if not cut.any():
        return {"buffers_m": [], "relative_vertical_mm_yr": [], "sd_mm_yr": 0.0,
                "basis": "nothing was cut from the comparison ground, so its width cannot move the zero"}
    vals = []
    for b in COMPARISON_GROUND_BUFFERS_M:
        far_b = far_first & ~(cut if b <= 0 else sg.euclidean_buffer(cut, ctx.attrs, float(b)))
        cen = sg.class_means(ctx.vel, ctx.tc, far_b)
        del far_b
        tot = sum(s1[i] - float(cen[i]) * s2[i] for i in range(s1.size) if s2[i] > 0 and i in cen)
        vals.append(float(tot) / n)
    sd = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    return {"buffers_m": [float(b) for b in COMPARISON_GROUND_BUFFERS_M],
            "relative_vertical_mm_yr": [round(v, 3) for v in vals], "sd_mm_yr": round(sd, 3),
            "basis": ("the whole area's relative figure with the report areas and the features that beat chance "
                      "cut from the comparison ground by " + " / ".join(f"{b:g}" for b in COMPARISON_GROUND_BUFFERS_M)
                      + " m; their sd is added to its relative uncertainty")}


def _dated_area_groups(cfg, ctx, up_map: np.ndarray) -> dict:
    """The dated mined areas panel_history tabulates by years since mining
    (one region per mine and year, at least 30 measured cells), as flat cell
    indices for _stream, with each one's median LOS-to-vertical factor (E3)."""
    import tempfile
    from . import panel_history as ph
    feats = ph._panel_features(cfg)
    if not feats:
        return {}
    by = {}
    for ft in feats:
        pr = ft["properties"]
        by.setdefault((pr["mine"], pr["year"]), []).append(ft)
    out = {}
    with tempfile.TemporaryDirectory() as td:
        for j, ((mine, y), fts) in enumerate(sorted(by.items())):
            m = ph._feature_mask(fts, ctx.good.shape, ctx.attrs, td) & ctx.good
            if m.sum() >= 30:
                out[f"__yr_{j}"] = {"name": f"{mine} {y}", "year": int(y), "ix": np.flatnonzero(m),
                                    "up": float(np.nanmedian(up_map[m]))}
    return out


def _years_since_mining(ctx, dated: dict, pl_series: dict, far: dict) -> dict:
    """
    'Movement by years since mining' on the report's own terms (E3, trust
    review 7 Oct 2026): panel_history's table was line of sight, against the
    FIRST farmland pass (Site A's 23-years row read -0.56 in where the
    report's frame gives -0.75), under a legend saying 'vertical'. Here every
    dated area's series is measured against this report's farmland (second
    pass, report areas out) by the same class matching as every area
    (ctx.matched), made vertical with the area's own factor, and tabulated by
    panel_history's own rules (_annual_levels, _by_years_since). Rows on
    fewer than YEARS_SINCE_MIN_AREAS areas are kept only in the record.
    """
    from . import panel_history as ph
    from . import report_insights as ri
    ground = ri.ground_word(getattr(ctx, "cfg", None) or {})
    panels = {}
    for key, d in dated.items():
        if key not in pl_series:
            continue
        tmp = np.zeros(ctx.good.size, bool)
        tmp[d["ix"]] = True
        rel = ctx.matched(pl_series[key], tmp.reshape(ctx.good.shape), far)
        del tmp
        fin = np.isfinite(rel)
        if not fin.any():
            continue
        rel = (rel - rel[fin][0]) / d["up"]
        lv = ph._annual_levels(ctx.dates, rel)
        panels[d["name"]] = {"annual_change_mm": [{"year": Y, "years_since_mining": Y - d["year"],
                                                   "change_mm": round(lv[Y] - lv[Y - 1], 1)}
                                                  for Y in sorted(lv) if Y - 1 in lv]}
    rows = ph._by_years_since(panels)
    return {"by_years_since_mining": [r for r in rows if r["n_areas"] >= YEARS_SINCE_MIN_AREAS],
            "by_years_since_mining_all_rows": rows,
            "by_years_since_mining_frame": "vertical",
            "by_years_since_mining_note": (
                f"Each dated mined area's change in average level from one calendar year to the next, vertical, "
                f"relative to the same surrounding {ground} as every other figure; the median over the areas and the "
                f"middle half of them. Rows resting on fewer than {YEARS_SINCE_MIN_AREAS} areas are left out. The "
                "spread is between areas: the table is not compared with chance.")}


def _point_figures(cfg, ctx, cell_null: dict | None, m: np.ndarray, up_map: np.ndarray,
                   bg_u: float | None) -> dict:
    """
    What the report says about single points, on its own frame:
      cells       the share of the study area's cells standing out and the
                  share of the comparison ground's that do by chance, on the
                  report's farmland (E14; report_insights.cells_standing_out)
      single_point_uncertainty_1sigma_mm_yr  the uncertainty raster's own
                  value (E18): the median over the study area of each cell's
                  farmland cell null (its coherence class, cell_sigma_map),
                  vertical, with the GNSS uncertainty when anchored - the
                  split-half repeatability quoted until 7 Oct 2026 was 20-40%
                  smaller than the rasters' and the KMZ's ranges
      radar_cells the radar cells behind the study area's points (E25:
                  distinct radar pixels over the study area's cells, not the
                  whole geocoded rectangle - 1.4x too many)
    """
    from . import report_insights as ri
    try:
        model = json.loads((Path(cfg.project_dir) / "deliverables" / "uncertainty_model.json").read_text())
    except (OSError, ValueError):
        model = {}
    null = cell_sigma_map(cell_null, ctx.kcls, model)
    out = {}
    fn = getattr(ri, "cells_standing_out", None)
    if fn is not None:
        out["cells"] = fn(ctx.r_cm, null, m & ctx.good, ctx.far)
    v = null[m & ctx.good] / up_map[m & ctx.good]
    v = v[np.isfinite(v) & (v > 0)]
    if v.size:
        sp = np.hypot(v, float(bg_u)) if bg_u else v
        out["single_point_uncertainty_1sigma_mm_yr"] = round(float(np.median(sp)), 2)
        out["single_point_uncertainty_basis"] = (
            "median over the study area of the per-point uncertainty raster: the farmland cell null of each point's "
            "coherence class, vertical" + (", with the regional GNSS uncertainty" if bg_u else ""))
    del null, v
    got = _optional(lambda: _module("deliverables").radar_pixel_ids(cfg, *np.nonzero(m & ctx.good), ctx.good.shape),
                    "radar_cells")
    if got is not None:
        ids, basis = got
        ids = np.asarray(ids)
        out["radar_cells"] = int(np.unique(ids[ids >= 0]).size)
        out["radar_cells_basis"] = f"distinct radar cells behind the study area's points ({basis})"
    cells = out.get("cells")
    if isinstance(cells, dict) and out.get("radar_cells") and hasattr(ri, "chance_share_range"):
        # F7(a): the share chance alone gives over this many radar cells, 95%
        cells["chance_range95_percent"] = ri.chance_share_range(cells.get("cells_standing_out_by_chance_percent"),
                                                                out["radar_cells"])
        cells["chance_range_basis"] = (f"95% binomial range of the chance share over the study area's "
                                       f"{out['radar_cells']:,} radar resolution cells")
    return out


def _with_point_uncertainty(Q: dict, sp_v: float | None, up: float) -> dict:
    """The QA record as report_insights.quality_notes reads it, its
    single-point figure replaced by the report's one per-point uncertainty
    (E18; quality_notes prints rate_precision.empirical_median_mm_yr / up).
    The QA's own record is not changed."""
    rp = (Q.get("checks") or {}).get("rate_precision")
    if not sp_v or not isinstance(rp, dict) or rp.get("empirical_median_mm_yr") is None:
        return Q
    q = dict(Q)
    q["checks"] = dict(Q["checks"])
    q["checks"]["rate_precision"] = dict(rp, empirical_median_mm_yr=float(sp_v) * float(up))
    return q


def _module(name: str):
    import importlib
    return importlib.import_module(f".{name}", __package__)


# What the build in progress left out (16 Oct 2026, readiness review
# ERR-01). A block _optional could not make used to leave one WARNING in the
# log while the report was still built and the run read 'done': the version
# line, the send rule re-evaluated with the GNSS tie or the QA's stand-outs
# could vanish with nothing on the page or in the record saying so. Each
# failure is now an entry of client_report.json's "omitted" list
# ({block, what, error}), named on the page under "How confident are these
# numbers?" and read by the release gate (release.omitted_items: one soft
# reading item, hard for the blocks whose absence makes a figure or the
# gate's own input wrong). A ContextVar: a build collects only its own.
_OMITTED: contextvars.ContextVar = contextvars.ContextVar("client_report_omitted", default=None)

# Each _optional block's key -> what the page and the release gate call it.
# The last four are made while the figures and the page are drawn, after
# compute's release block: they must stay soft and out of the numbers hash
# (release.OMITTED_HARD, release._OMITTED_NOT_HASHED).
OPTIONAL_BLOCKS = {
    "dated_mined_areas": "the dated mined areas behind the table of movement by years since mining",
    "years_since_mining": "the table of movement by years since mining",
    "point_figures": "the single-point uncertainty and the share of points standing out",
    "radar_cells": "the count of independent radar measurements (an estimate is given instead)",
    "resolution": "the effective resolution of the measurements",
    "resolution_caveat": "the note on a report area too narrow for the resolution",
    "feature_null_check": "the check that the feature search's simulated copies are as strong as the real "
                          "comparison ground",
    "standouts": "the quality assessment's calibrated stand-outs",
    "send_rule": "the send rule re-evaluated with the GNSS tie",
    "version": "the software version line",
    "credits": "the data credits",
    "gradient_limit": "the fastest movement between neighbouring points the radar can follow",
    "los_east_over_up": "how much east-west movement reads as vertical",
    "figure_credits": "the data credits printed on the figures",
    "footer_version": "the software version in the page footer",
    "footer_identity": "who prepared the report, in the page footer",
    "credits_section": "the data sources and credits section",
}


def _upper_first(s: str) -> str:
    return s[:1].upper() + s[1:]


def _note_omitted(block: str, exc: BaseException) -> None:
    """One entry in the omitted list of the build in progress (none outside
    one), once per block."""
    rec = _OMITTED.get()
    if rec is None or any(o.get("block") == block for o in rec):
        return
    rec.append({"block": block, "what": OPTIONAL_BLOCKS.get(block, f"the report block '{block}'"),
                "error": f"{type(exc).__name__}: {str(exc)[:160]}"})


@contextlib.contextmanager
def _collecting(omitted: list):
    """Record _optional failures into `omitted` (a report's 'omitted' list)
    for the body."""
    tok = _OMITTED.set(omitted)
    try:
        yield omitted
    finally:
        _OMITTED.reset(tok)


def _optional(fn, block: str = ""):
    """The value of a contract function another module provides, or None
    when it raises - logged, and recorded in the build's 'omitted' list
    (_note_omitted). An ImportError or AttributeError used to be 'not there
    yet' and logged at INFO; every contract module exists now, so it is a
    failure like any other."""
    try:
        return fn()
    except Exception as exc:                                  # noqa: BLE001 - never stop the report for a side block
        log.warning("Optional report block '%s' failed and is left out of the report: %s: %s", block or "?",
                    type(exc).__name__, exc)
        _note_omitted(block or "unnamed", exc)
        return None


def _route(cfg) -> str:
    try:
        from .config import route_tag
        return route_tag(cfg)
    except (ImportError, AttributeError):
        return {"miaplpy": "official", "disp": "screening"}.get(str(getattr(cfg, "backend", "")).lower(), "comparison")


def _cell_null(cfg, ctx) -> dict | None:
    """{coherence class index (str): per-cell farmland null, mm/yr LOS} =
    max(farmland cell spread, the cell's own repeatability) - the one
    per-cell yardstick (significance.cell_null_by_class), read from
    uncertainty_model.json when deliverables wrote it there, else measured
    here on the same class-matched field."""
    sg = ctx.sg
    try:
        model = json.loads((Path(cfg.project_dir) / "deliverables" / "uncertainty_model.json").read_text())
    except (OSError, ValueError):
        model = {}
    got = model.get("cell_null_by_class")
    if isinstance(got, dict) and got:
        return {str(k): float(v) for k, v in got.items() if str(k).lstrip("-").isdigit() and v is not None}
    fn = getattr(sg, "cell_null_by_class", None)
    if fn is None:
        return None
    _, rows = sg.farmland_cell_spread(ctx.r_cm, ctx.tc, ctx.far)
    return {str(k): float(v) for k, v in fn(model, rows).items() if isinstance(k, int)}


def release_block(cfg, rep: dict, qa: dict | None) -> dict:
    """{final, blockers, reviewed_by, review_date} from release.status (Kyle,
    5 Oct 2026: a DRAFT until a reviewer signs off these exact numbers). A
    missing gate is a hard blocker, never a pass."""
    try:
        from . import release
        st = release.status(cfg, rep, qa) if hasattr(release, "status") else {
            "final": release.is_final(cfg, rep, qa), "blockers": release.blockers(cfg, rep, qa)}
        return {"final": bool(st.get("final")), "blockers": list(st.get("blockers") or []),
                "reviewed_by": st.get("reviewed_by"), "review_date": st.get("review_date")}
    except (ImportError, AttributeError) as exc:
        return {"final": False, "blockers": [{"code": "release_gate_unavailable", "hard": True,
                                              "text": f"The release gate could not be read ({exc})."}]}


# ------------------------------------------------------------------ figures
def stamp_lines(cfg, rep, extra: str = "") -> list[str]:
    """The notices and identity every client image carries, because images
    travel without the page (clientout-9, gap-licensing-1): the data credits
    (credits.short_line) and `extra` (the basemap credit when one was drawn)
    on one line; the report, its date and the version on the next. One
    wording for the report's PNGs and every download of the results viewer
    (19 Oct 2026), which shows these lines and never composes its own."""
    parts = [x for x in (_optional(lambda: _module("credits").short_line(cfg, rep), "figure_credits"), extra) if x]
    ver = rep.get("version") or {}
    vl = f"insar_pipeline {ver.get('version')} (code {ver.get('code_hash')})" if ver.get("version") else ""
    gen = str(rep.get("generated") or "")[:10]
    line2 = " · ".join(x for x in (f"{rep.get('title') or ''} ground movement report", gen, vl) if x)
    return [x for x in (" · ".join(parts), line2) if x]


def _stamp(fig, cfg, rep, final: bool, extra: str = "", y: float = 0.002) -> None:
    """stamp_lines under the figure, and a DRAFT watermark until the report
    is final."""
    import textwrap
    width = max(60, int(fig.get_figwidth() * 17))           # characters of 7 pt across the figure
    fig.text(0.005, y, "\n".join(textwrap.fill(x, width) for x in stamp_lines(cfg, rep, extra)), fontsize=7,
             color="#58636e", ha="left", va="top" if y < 0 else "bottom")
    if not final:
        fig.text(0.5, 0.5, "DRAFT", fontsize=96, color="#a8432f", alpha=0.13, rotation=28, ha="center",
                 va="center", weight="bold", zorder=100)


def _lat_fmt(v, _pos=None):
    return f"{abs(v):.2f}°{'N' if v >= 0 else 'S'}"


def _lon_fmt(v, _pos=None):
    return f"{abs(v):.2f}°{'W' if v < 0 else 'E'}"


def _nice_km(width_km: float) -> float:
    """A scale-bar length near a fifth of the map's width."""
    target = width_km / 5.0
    return max((x for x in (0.5, 1, 2, 5, 10, 20, 50, 100) if x <= max(target, 0.5)), default=0.5)


def _scale_bar_and_north(ax, w, s_, e, n, outside: bool = False) -> None:
    """A km and mile scale bar and a north arrow (upper right), in axes
    coordinates (clientout-12). The bars sit inside the lower right, or -
    outside=True, the client map - under the map's right edge, where they
    cover nothing (on Site B they covered Feature 2's box)."""
    lat = 0.5 * (s_ + n)
    km_per_deg = 111.32 * np.cos(np.radians(lat))
    width_km = (e - w) * km_per_deg
    L = _nice_km(width_km)
    mi = max((x for x in (0.5, 1, 2, 5, 10, 20, 50) if x * 1.609344 <= L * 1.2), default=0.5)
    x1 = 1.0 if outside else 0.97
    ys = (-0.075, -0.115) if outside else (0.045, 0.085)
    for k, (length_km, label, y) in enumerate(((L, f"{L:g} km", ys[0]), (mi * 1.609344, f"{mi:g} mi", ys[1]))):
        frac = length_km / width_km
        ax.plot([x1 - frac, x1], [y, y], transform=ax.transAxes, color="#1c2530", lw=3, solid_capstyle="butt",
                zorder=20, clip_on=False)
        ax.text(x1 - frac / 2, y + 0.008, label, transform=ax.transAxes, ha="center", va="bottom", fontsize=8.5,
                color="#1c2530", zorder=20, bbox={"boxstyle": "round,pad=0.15", "fc": "white", "ec": "none", "alpha": 0.7})
    ax.annotate("N", xy=(0.95, 0.96), xytext=(0.95, 0.87), xycoords="axes fraction", textcoords="axes fraction",
                ha="center", va="center", fontsize=12, weight="bold", color="#1c2530", zorder=20,
                arrowprops={"arrowstyle": "-|>", "color": "#1c2530", "lw": 1.6})


def _map_cmap(edges: list, n_below: int):
    """Reds for the n_below bins under the band (darkest furthest out), grey
    for the band, blues above; the open ends (extend arrows) the darkest of
    each side."""
    import matplotlib
    from matplotlib.colors import BoundaryNorm, ListedColormap
    reds, blues = matplotlib.colormaps["Reds"], matplotlib.colormaps["Blues"]
    n_above = len(edges) - 2 - n_below
    neg = [reds(x) for x in np.linspace(0.95, 0.4, n_below)] if n_below else []
    pos = [blues(x) for x in np.linspace(0.4, 0.95, n_above)] if n_above > 0 else []
    cmap = ListedColormap(neg + ["#dcdcdc"] + pos).with_extremes(under=reds(1.0), over=blues(1.0), bad=(0, 0, 0, 0))
    return cmap, BoundaryNorm(edges, cmap.N)


def figures(cfg, rep, ctx, masks, outlines, out_dir: Path | None = None, cache_dir: Path | None = None) -> dict:
    """
    The three client PNGs. The map (clientout-5, generality-4, clientout-12,
    methodology-4): the smoothed field relative to the comparison ground,
    grey inside the band ordinary ground stays within, coloured beyond it in
    bins from the data (report_insights.map_scale) with extend arrows;
    unmeasured ground left uncoloured (hatched inside report areas); a
    cached public-domain basemap when one can be had; scale bar, north arrow,
    full latitude labels. The series (clientout-6, stats-8): each area's
    image-by-image line, its straight average-rate line, and the two bands
    of series_band. Every PNG carries the credits, version and date, and a
    DRAFT watermark until the report is final.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.ticker import FuncFormatter
    from . import report_insights as ri
    out_dir = out_dir or (cfg.project_dir / "deliverables" / "figures"); out_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "font.family": "DejaVu Sans", "hatch.color": "#6b7580", "hatch.linewidth": 0.6})
    final = bool((rep.get("release") or {}).get("final"))
    anchored = bool(rep.get("anchored"))
    bg_v = (((rep.get("gnss") or {}).get("background") or {}).get("vertical_mm_yr") or 0.0) if anchored else 0.0
    ground = ri.ground_word(rep)
    names = {a["name"]: a.get("display_name") or a["name"] for a in rep["areas"]}
    rf = ctx.report_field
    sm_rel = rf["field"]
    ms = rep.get("map_scale")
    if rf.get("band_ok", True) and np.isfinite(rf.get("band_mm_yr", np.nan)):
        lo_b = rf.get("band_lo_mm_yr", -rf["band_mm_yr"])
        hi_b = rf.get("band_hi_mm_yr", rf["band_mm_yr"])
        quiet = (sm_rel > lo_b) & (sm_rel < hi_b)
    else:
        lo_b = hi_b = None
        quiet = np.zeros(sm_rel.shape, bool)
    a = ctx.attrs
    x0, y0, dx, dy = (float(a[kk]) for kk in ("X_FIRST", "Y_FIRST", "X_STEP", "Y_STEP"))
    w, s_, e, n = cfg.bbox
    r0, c0 = int((n - y0) / dy), int((w - x0) / dx); r1, c1 = int((s_ - y0) / dy), int((e - x0) / dx)
    sl = (slice(max(0, r0), r1), slice(max(0, c0), c1))
    sub, q = sm_rel[sl], quiet[sl]
    ext = (x0 + max(0, c0) * dx, x0 + c1 * dx, y0 + r1 * dy, y0 + max(0, r0) * dy)
    fig, ax = plt.subplots(figsize=(10.2, 8.6))
    # basemap: read from the project's cache, fetched into the writable one
    bm = None
    if str(getattr(cfg, "report_basemap", "usgs") or "").lower() not in ("", "none", "off", "false", "no"):
        caches = [Path(cfg.project_dir) / "context" / "basemap"] + ([cache_dir] if cache_dir else [])
        bm = ri.basemap((w, s_, e, n), caches, cache_dir or caches[0])
    if bm is not None:
        g = bm["image"].mean(axis=2)                       # greyscale, lightened: the data stay the colour
        ax.imshow(0.4 + 0.6 * g, extent=bm["extent"], cmap="gray", vmin=0, vmax=1, interpolation="bilinear",
                  zorder=0)
    if ms and ms.get("edges_mm_yr"):
        cmap, norm = _map_cmap(ms["edges_mm_yr"], int(ms.get("n_below") or 0))
        im = ax.imshow(np.where(q, np.nan, sub), extent=ext, cmap=cmap, norm=norm, interpolation="nearest",
                       alpha=0.9 if bm is not None else 1.0, zorder=1)
    else:                                                  # no band: one plain diverging scale, labelled so
        from matplotlib.colors import TwoSlopeNorm
        lim = max(10.0, float(np.nanpercentile(np.abs(sub), 99)) if np.isfinite(sub).any() else 10.0)
        im = ax.imshow(sub, extent=ext, cmap="RdBu", norm=TwoSlopeNorm(0, -lim, lim), interpolation="nearest",
                       alpha=0.9 if bm is not None else 1.0, zorder=1)
    ax.imshow(np.where(q & np.isfinite(sub), 1.0, np.nan), extent=ext, cmap="Greys", vmin=0, vmax=4,
              interpolation="nearest", alpha=0.55 if bm is not None else 1.0, zorder=1)
    # not measured INSIDE a report area: hatched, so a mostly blank outline is
    # seen as blank, not as quiet (methodology-4)
    unmeas = np.zeros(sub.shape, bool)
    for ar in rep["areas"]:
        if ar.get("kind") == "area" and ar["name"] in masks:
            fp = _footprint_of(ctx, ar)
            if fp is not None:
                unmeas |= (fp & ~ctx.good)[sl]
    handles = []
    if unmeas.any():
        yy = np.linspace(ext[3], ext[2], sub.shape[0])
        xx = np.linspace(ext[0], ext[1], sub.shape[1])
        ax.contourf(xx, yy, unmeas.astype(float), levels=[0.5, 1.5], colors="none", hatches=["////"], zorder=2)
        handles.append(Patch(facecolor="white", edgecolor="#6b7580", hatch="////",
                             label="not measured, inside a report area"))
    colours = ["#1c2530", "#0f7a4a", "#6b3fa0"]
    for i, ar in enumerate(rep["areas"]):
        if ar.get("kind") == "full":
            continue
        col = colours[i % len(colours)]
        for ring in outlines.get(ar["name"], []):
            ax.plot(ring[:, 0], ring[:, 1], color=col, lw=1.6 if len(outlines[ar["name"]]) < 3 else 0.9, zorder=5)
        allr = np.vstack(outlines.get(ar["name"], [np.zeros((1, 2))]))
        if (allr[:, 0].max() - allr[:, 0].min()) < 0.02:          # too small to find: ring it
            cx, cy = allr[:, 0].mean(), allr[:, 1].mean()
            ax.plot(cx, cy, marker="o", ms=22, mfc="none", mec=col, mew=2.2, ls="none", zorder=5)
            ax.annotate(names[ar["name"]], (cx, cy), xytext=(14, 12), textcoords="offset points", fontsize=9.5,
                        color=col, weight="bold", zorder=6)
        handles.append(Line2D([], [], color=col, lw=1.8, label=names[ar["name"]]))
    for i, f in enumerate(rep.get("features") or [], 1):
        w0, s0, e0, n0 = f["bbox"]
        pad = 0.004
        ax.add_patch(plt.Rectangle((w0 - pad, s0 - pad), e0 - w0 + 2 * pad, n0 - s0 + 2 * pad, fill=False,
                                   ec="#1c2530", lw=1.6, zorder=7))
        ax.annotate(str(i), (w0 - pad, n0 + pad), xytext=(2, 2), textcoords="offset points", fontsize=11,
                    weight="bold", color="#1c2530", va="bottom", zorder=8,
                    bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "#1c2530", "lw": 0.8})
    if rep.get("features"):
        handles.append(Line2D([], [], color="#1c2530", lw=1.6, label="numbered features (see Observations)"))
    mine_cols = {"underground": "#7a2a1a", "surface": "#8a3aa0", "other": "#b06a00", "mapped": "#3a3a3a"}
    for L in ri.mine_layers(cfg):
        for ft in json.loads(L["path"].read_text())["features"]:
            g_ = ft.get("geometry") or {}
            for poly in (g_.get("coordinates") or []) if g_.get("type") == "MultiPolygon" else [g_.get("coordinates") or []]:
                if poly:
                    r_ = np.array(poly[0])[:, :2]
                    ax.plot(r_[:, 0], r_[:, 1], color=mine_cols.get(L["key"], "#3a3a3a"), lw=0.6, alpha=0.8, zorder=4)
        handles.append(Line2D([], [], color=mine_cols.get(L["key"], "#3a3a3a"), lw=1.0,
                              label=f"mapped {L['label']}" + (f" (to {L['vintage']})" if L.get("vintage") else "")))
    seen_gnss = False
    for st in (rep.get("gnss") or {}).get("stations", []):
        if st.get("inside_processed_area"):
            ax.plot(st["lon"], st["lat"], marker="^", ms=11, mfc="#ffd23f", mec="k", ls="none", zorder=9)
            ax.annotate(f"GNSS {st['site']}", (st["lon"], st["lat"]), xytext=(6, 6), textcoords="offset points",
                        fontsize=9, zorder=9)
            if not seen_gnss:
                handles.append(Line2D([], [], marker="^", ms=9, mfc="#ffd23f", mec="k", ls="none", label="GNSS station"))
                seen_gnss = True
    ax.set_xlim(w, e); ax.set_ylim(s_, n); ax.set_aspect(1 / np.cos(np.radians((s_ + n) / 2)))
    ax.xaxis.set_major_formatter(FuncFormatter(_lon_fmt)); ax.yaxis.set_major_formatter(FuncFormatter(_lat_fmt))
    ax.tick_params(labelsize=8.5)
    _scale_bar_and_north(ax, w, s_, e, n, outside=True)
    if ms and ms.get("edges_mm_yr"):
        edges = ms["edges_mm_yr"]
        cb = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02, extend="both", ticks=edges, spacing="uniform")
        cb.ax.set_yticklabels([f"{v_ / MM_IN:+.2f} in ({v_:+.0f} mm)" for v_ in edges], fontsize=8)
        cb.set_label(f"vertical movement per year relative to the surrounding {ground}\n"
                     f"(red = sinking, blue = rising; arrows: beyond {ms['top_in_per_yr']:g} in/yr)", fontsize=9)
        handles.append(Patch(facecolor="#dcdcdc", edgecolor="none",
                             label=f"within normal ground variation ({lo_b / MM_IN:+.2f} to {hi_b / MM_IN:+.2f} in/yr)"))
    else:
        cb = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02, extend="both")
        cb.set_label(f"vertical movement per year relative to the surrounding {ground}, mm/yr", fontsize=9)
        handles.append(Patch(facecolor="none", edgecolor="none",
                             label=f"no grey band: too little {ground} to measure its normal range"))
    handles.append(Patch(facecolor="white", edgecolor="#8a95a1",
                         label=("no colour or grey: not measured" + (" (the base map shows through)" if bm is not None else ""))))
    # below the map, not over it (it covered a report area on the synthetic test site)
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, -0.13), ncol=2, fontsize=8.5, frameon=False)
    d0, d1 = rep["dates"]
    ax.set_title(f"{rep.get('full_area_name') or 'Study area'}: vertical ground movement per year, {d0[:4]}–{d1[:4]}",
                 fontsize=12)
    fig.canvas.draw()
    leg_bottom = min(ax.get_legend().get_window_extent().y0 / fig.bbox.height, 0.1)
    _stamp(fig, cfg, rep, final, bm["credit"] if bm is not None else "", y=leg_bottom - 0.06)
    ri.tidy_figure(fig, rep.get("control_ground_label"))
    fig.savefig(out_dir / "client_map.png", dpi=120, bbox_inches="tight", pad_inches=0.25, metadata=png_metadata())
    plt.close(fig)

    # series: each area's line, its straight average-rate line and the bands of series_band
    t = ctx.t
    dt_ = [datetime.strptime(d, "%Y%m%d") for d in rep["dates_list"]]
    items = series_items(rep)
    fig, axs = plt.subplots(max(1, len(items)), 1, figsize=(10, 3.5 * max(1, len(items))), sharex=True, squeeze=False)
    if not items:
        axs[0, 0].text(0.5, 0.5, "No report area could be measured, so there is no series to show.",
                       ha="center", va="center", transform=axs[0, 0].transAxes)
        axs[0, 0].set_axis_off()
    for axx, it in zip(axs[:, 0], items):
        _series_axes(axx, dt_, t, it, bg_v if not it["relative"] else 0.0)
    fig.suptitle("Vertical movement over the record (negative = sinking; every thick line starts at zero)", fontsize=12)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    _stamp(fig, cfg, rep, final)
    ri.tidy_figure(fig, rep.get("control_ground_label"))
    fig.savefig(out_dir / "client_series.png", dpi=120, bbox_inches="tight", pad_inches=0.2, metadata=png_metadata())
    plt.close(fig)
    summary_table(rep, out_dir / "client_summary.png", cfg=cfg)
    return {"map": out_dir / "client_map.png", "series": out_dir / "client_series.png",
            "summary": out_dir / "client_summary.png", "basemap_credit": bm["credit"] if bm is not None else None}


def _footprint_of(ctx, ar: dict) -> np.ndarray | None:
    try:
        return _mask(ctx, ar["spec"])[2]
    except Exception as exc:                                  # noqa: BLE001 - only the hatching needs it
        log.info("Footprint of %s not redrawn (%s)", ar.get("name"), exc)
        return None


def series_items(rep: dict, max_features: int | None = 3) -> list[dict]:
    """What each chart draws, in the report's own numbers: the line (zeroed
    on its fitted start), the rate of its straight line, and the bands:
    centre = the rate ordinary ground of this footprint would have (the
    area's rate minus its tested excess, so the straight line is exactly
    `excess` away from it), cone = the decision threshold, noise = the
    per-date noise. Areas that could not be compared get no bands. Features
    are relative to the comparison ground and tested by the search (selection
    p), so they get only the outer band of ordinary ground of their footprint
    (cone 1.96 x its footprint spread), no decision band. The report's
    chart page shows the first max_features features; the KMZ (None) all.
    `key` is the area's name as client_report.json stores it."""
    anchored = bool(rep.get("anchored"))
    bg_v = (((rep.get("gnss") or {}).get("background") or {}).get("vertical_mm_yr") or 0.0) if anchored else 0.0
    out = []
    for ar in rep["areas"]:
        if ar.get("kind") != "area" or not ar.get("series_vertical_mm") or ar.get("vertical_mm_yr") is None:
            continue
        v = float(ar["vertical_mm_yr"])
        testable = ar.get("testable") is not False and ar.get("decision_threshold_mm_yr")
        ex = ar.get("excess_vertical_mm_yr")
        centre = (v - float(ex)) if (testable and ex is not None) else (
            float(ar.get("null_median_vertical_mm_yr") or 0.0) + (bg_v if anchored else 0.0))
        out.append({"name": ar.get("display_name") or ar["name"], "key": ar["name"],
                    "series_vertical_mm": ar["series_vertical_mm"],
                    "relative": not anchored, "rate": v, "total_mm": ar.get("total_mm"),
                    "centre_rate": centre, "cone_rate": float(ar["decision_threshold_mm_yr"]) if testable else None,
                    "noise_mm": ar.get("series_noise_mm") if testable else None, "decision": bool(testable),
                    "reading": ar.get("reading") or "", "detected": ar.get("detected")})
    for i, f in enumerate((rep.get("features") or [])[:max_features], 1):
        if f.get("series_vertical_mm") and not f.get("ground_reshaped"):
            sd = f.get("farmland_spread_1sigma_mm_yr")
            out.append({"name": f"Feature {i}", "key": f"Feature {i}", "series_vertical_mm": f["series_vertical_mm"],
                        "relative": True,
                        "rate": float(f["vertical_relative_mm_yr"]), "total_mm": f.get("total_relative_mm"),
                        "centre_rate": 0.0, "cone_rate": Z95 * float(sd) if sd else None, "feature": True,
                        "noise_mm": f.get("series_noise_mm"), "decision": False, "reading": "", "detected": None,
                        # the line its stated total is read on (final check of 7 Oct 2026)
                        "total_method": f.get("total_method") or "rate_x_span",
                        "fit_vertical_mm": f.get("fit_vertical_mm")})
    return out


def feature_line(it: dict, t: np.ndarray) -> tuple[np.ndarray, str] | None:
    """(the line a feature's stated total is read on, vertical mm from zero,
    its legend label) - None for a report area or a feature with no rate.
    A rate x span total: the average rate x the years; a pace-change total:
    the fitted curve it was read from (feature_entry's fit_vertical_mm),
    else the average rate (a record from before 7 Oct 2026)."""
    if not it.get("feature") or it.get("rate") is None:
        return None
    fv = it.get("fit_vertical_mm")
    if it.get("total_method") not in (None, "rate_x_span") and fv and len(fv) == np.size(t):
        y = np.array([np.nan if v is None else v for v in fv], dtype=float)
        if np.isfinite(y).sum() > 1:
            return y, "fitted curve (its measured change)"
    return float(it["rate"]) * np.asarray(t, dtype=float), "its average rate"


def _series_axes(axx, dt_, t, it: dict, bg_line: float, compact: bool = False) -> None:
    y = np.array([np.nan if v is None else v for v in it["series_vertical_mm"]], dtype=float) / MM_IN
    band = series_band(t, it["centre_rate"], it["cone_rate"], it["noise_mm"])
    if band["outer"] is not None:
        axx.fill_between(dt_, band["outer"][0] / MM_IN, band["outer"][1] / MM_IN, color="#dfe5ea", lw=0,
                         label="where single images of ordinary ground wander (95%)")
    # features: the lighter band only (series_items, the page's caption) - the
    # darker one is an area's decision band and a feature's test is the search's
    if band["inner"] is not None and not it.get("feature"):
        axx.fill_between(dt_, band["inner"][0] / MM_IN, band["inner"][1] / MM_IN, color="#b9c5cf", lw=0,
                         label=("how far ordinary ground's average rate reaches (95%)" if it["decision"] else
                                "ordinary ground's average rate (95%)"))
    if bg_line:
        axx.plot(dt_, bg_line * t / MM_IN, color="#5b6672", lw=1, ls="--", label="regional ground motion (GNSS)")
    col = "#a8432f" if it["rate"] < 0 else "#2d5f7c"
    axx.plot(dt_, y, color=col, lw=1.6 if it.get("feature") else 0.9, alpha=1.0 if it.get("feature") else 0.75,
             label=f"{it['name']}, every image")
    fit = feature_line(it, t)
    if fit is not None:
        # a feature: the line its stated total is read on, from zero (final
        # check of 7 Oct 2026: none was drawn, and the thin line - zeroed on
        # the fitted start - began at +0.93 in on Site B Feature 1); the
        # average rate when the total is rate x span, else the fitted curve
        axx.plot(dt_, fit[0] / MM_IN, color=col, lw=2.4, label=fit[1])
    elif it["decision"] or not it.get("feature"):
        # the straight line the decision is read on (areas)
        axx.plot(dt_, it["rate"] * t / MM_IN, color=col, lw=2.4, label="its average rate")
    axx.axhline(0, color="#8a95a1", lw=0.6)
    axx.set_ylabel("inches"); axx.grid(alpha=0.25)
    axx.secondary_yaxis("right", functions=(lambda v: v * MM_IN, lambda v: v / MM_IN)).set_ylabel("mm")
    if compact:                       # the one-pager states the numbers above the chart
        axx.set_title("Movement over the record (every thick line starts at zero)", fontsize=10, loc="left")
        axx.legend(fontsize=7.5, loc="best", ncol=2, framealpha=0.9)
        return
    # the rate only: the total over the record is in the table with its 95%
    # range (E9 - a total is never printed without it)
    axx.set_title(f"{it['name']}: {_s(it['rate'] / MM_IN)} in/yr ({it['rate']:+.1f} mm/yr)"
                  + ("  (relative to the surrounding ground)" if it.get("relative") else ""), fontsize=10.5, loc="left")
    tag = ("Movement detected" if it.get("detected") else "Within normal ground variation" if it["decision"]
           else "Not tested against ordinary ground" if not it.get("feature") else "")
    if tag:
        axx.set_title(tag, fontsize=9.5, loc="right",
                      color="#a8432f" if it.get("detected") else "#3d7a4c" if it["decision"] else "#6b7580")
    axx.legend(fontsize=8, loc="best", ncol=2, framealpha=0.9)


def summary_rows(rep) -> list[list[str]]:
    """The client summary table's rows, from the report's own fields: an area
    with no figure reads 'relative only' / '—' (stats-13: an unmeasurable
    area crashed this table), a range is never derived here (range95_mm_yr,
    None -> '—'), and the reading is the area's `reading` - the whole study
    area no longer claims 'moves with the GNSS network' without a tie.
    Rates to 0.01 in/yr (E9: '0.000 in/yr' claimed 50 times the precision
    the data have); every total with its 95% range; an area that could not
    be compared with farmland shows the uncertainty that can be stated."""
    ground = _ri().ground_word(rep)
    rows = []
    for ar in rep["areas"]:
        v = ar.get("vertical_mm_yr")
        rng = ar.get("range95_mm_yr")
        if "range95_mm_yr" not in ar and v is not None and ar.get("vertical_uncertainty_1sigma_mm_yr"):
            sd = ar["vertical_uncertainty_1sigma_mm_yr"]
            rng = [v - Z95 * sd, v + Z95 * sd]           # a report built before 5 Oct 2026
        det = ar.get("detection_limit_mm_yr")
        tot = ar.get("total_mm")
        trng = ar.get("total_range95_mm") or ar.get("untested_total_range95_mm")
        reading = ar.get("reading") or (
            ("Movement detected" if ar.get("detected") else "Within normal ground variation")
            if ar.get("kind") == "area" else "Whole study area")
        size = (f"{ar['area_km2']:,.1f} of {ar['footprint_km2']:,.1f} km² measured" if ar.get("footprint_km2")
                else f"{ar.get('area_km2') or 0:,.1f} km²")
        urng = None if rng else ar.get("untested_range95_mm_yr")
        rows.append([f"{ar.get('display_name') or ar['name']}\n{size}  ·  {ar.get('cells') or 0:,} points",
                     (f"{_s(v / MM_IN)} in/yr\n{v:+.1f} mm/yr" if v is not None else
                      ("relative only" if ar.get("relative_only") else "—")),
                     (f"{_s(tot / MM_IN)} in\n{tot:+.0f} mm" + (f"\n95%: {_s(trng[0] / MM_IN)} to {_s(trng[1] / MM_IN)} in"
                                                                 if trng else "")
                      if tot is not None else "—"),
                     (f"{_s(rng[0] / MM_IN)} to {_s(rng[1] / MM_IN)} in/yr" if rng else
                      f"{_s(urng[0] / MM_IN)} to {_s(urng[1] / MM_IN)} in/yr\nnot compared with {ground}" if urng
                      else "—"),
                     (f"{det / MM_IN:.2f} in/yr\n{det:.0f} mm/yr" if det else "—"),
                     (f"{ar['signal_to_noise']:.1f}" if ar.get("signal_to_noise") is not None else "—"),
                     reading])
    return rows


def _reading_colour(row: list[str], area: dict) -> str:
    if area.get("detected") is True:
        return "#c23434"
    if area.get("testable") is False or area.get("vertical_mm_yr") is None:
        return "#6b7580"
    return "#16803a" if area.get("kind") == "area" else "#1c2530"


def summary_table(rep, path, cfg=None) -> None:
    """The report areas as one graphic, in the client report's own numbers
    (vertical, GNSS-anchored when tied) - the table to put in front of a
    client. The technical summary_table.png (line of sight, relative, every
    mine) is not. Stamped like the other PNGs (credits, version, date; DRAFT
    until final)."""
    import matplotlib.pyplot as plt
    from . import report_insights as ri
    ground = ri.ground_word(rep)
    import textwrap
    rows = summary_rows(rep)
    head = ["Area", "Vertical,\nper year", f"Measured over\n{rep['span_years']:.1f} years", "95% range",
            "Reliably detects\nsinking (4 in 5)", "Signal ÷\nthreshold", "Reading"]
    widths = [0.22, 0.11, 0.1, 0.13, 0.11, 0.075, 0.255]
    # cells wrap at their column's width (11 pt at 16 in: ~1.55 characters
    # per 0.01 of width): the reason an area is not testable used to run off
    # the table, and a long area line was cut at its column edge
    wrap = [max(10, int(w * 155)) for w in widths]
    rows = [["\n".join(textwrap.fill(part, wrap[c]) for part in str(v).split("\n")) for c, v in enumerate(r)]
            for r in rows]
    lines = [max(2, max(str(v).count("\n") + 1 for v in r)) for r in rows]
    row_h = [0.22 * k + 0.3 for k in [2] + lines]                # inches; the header first
    # heights in inches: the footnote grows with the features listed under it,
    # and the stamp (credits, version, date) sits under the footnote
    n_note = 4 + min(3, len(rep.get("features") or [])) + (1 if rep.get("features") else 0)
    stamp_h = 0.45
    note_h, title_h, table_h = 0.2 * n_note + 0.25 + stamp_h, 0.55, sum(row_h)
    H = note_h + title_h + table_h
    fig = plt.figure(figsize=(16, H))
    ax = fig.add_axes([0.01, note_h / H, 0.98, table_h / H])
    ax.axis("off")
    tb = ax.table(cellText=rows, colLabels=head, cellLoc="left", colLoc="left", bbox=[0, 0, 1, 1],
                  colWidths=widths)
    tb.auto_set_font_size(False); tb.set_fontsize(11)
    for (r, c), cell in tb.get_celld().items():
        cell.set_edgecolor("#d9dfe5")
        cell.set_height(row_h[r] / table_h)
        if r == 0:
            cell.set_facecolor("#17202a"); cell.get_text().set_color("white"); cell.get_text().set_weight("bold")
        elif c == 6:
            cell.get_text().set_color(_reading_colour(rows[r - 1], rep["areas"][r - 1]))
    d0, d1 = rep["dates"]
    note = (f"{sensor_name(rep.get('backend'))} radar, {rep['n_dates']} images, {d0[:4]}-{d0[4:6]}-{d0[6:]} to {d1[:4]}-{d1[4:6]}-{d1[6:]}. "
            "Vertical movement, negative = sinking" + (", anchored to the regional GNSS network. " if rep.get("anchored") else f", relative to the area's {ground} (no GNSS anchor). ")
            + "95% range = this record's uncertainty for an area of that size and ground type. Reliably detects = "
            "sustained sinking of that size across the area would be detected 4 times in 5.\n"
            f"Signal ÷ threshold = the area's movement beyond ordinary {ground} divided by the 95% threshold measured "
            f"on {ground} patches of its size: 1 or more stands out (for the whole area: its rate, satellite plus "
            "GNSS, ÷ its 95% uncertainty; 1 or more means its range excludes zero). 'Not testable' areas have no "
            "comparison: their range is only the uncertainty that can be stated without one.")
    feats3 = (rep.get("features") or [])[:3]
    for i, f in enumerate(feats3, 1):
        v = f.get("vertical_relative_mm_yr", f["mean_rel_vertical_mm_yr"])
        what = ri.feature_kind(f)
        rr = ri.rate_range_words(v, f.get("rate_range95_mm_yr"))
        rate = ("" if f.get("ground_reshaped") else
                f", about {abs(v) / MM_IN:.2f} in/yr{' (' + rr + ')' if rr else ''} {'down' if v < 0 else 'up'} "
                f"relative to {ground}")
        note += (f"\nFeature {i}: {what}, about {f['area_km2']:.1f} km² at {ri.lalo_txt(f['lat'], f['lon'])}"
                 f"{rate}; {(f.get('mines') or {}).get('phrase', '')}.")
    if any(not f.get("ground_reshaped") for f in feats3):
        # M4: each outline is drawn from the same data
        note += ("\nEach feature's outline was drawn from these same measurements, so its rate and range are for that "
                 "outline and read somewhat stronger than the ground there truly moved.")
    title = rep.get("title") or "Ground movement"
    fig.text(0.01, 1 - 0.12 / H, f"{title} ground movement: the report areas", fontsize=14, weight="bold", va="top")
    fig.text(0.01, (stamp_h + 0.05) / H, "\n".join(textwrap.fill(x, 190) for x in note.split("\n")),
             fontsize=10, color="#55616e", va="bottom")
    if cfg is not None:
        _stamp(fig, cfg, rep, bool((rep.get("release") or {}).get("final")), y=0.05 / H)
    _ri().tidy_figure(fig, rep.get("control_ground_label"))
    fig.savefig(path, dpi=130, bbox_inches="tight", metadata=png_metadata()); plt.close(fig)


# ------------------------------------------------------------------ page
def _in(mm, nd=2):
    return "—" if mm is None else f"{abs(mm) / MM_IN:.{nd}f} in"


def _sin(mm, nd=2):
    """Signed inches: a figure without a direction word."""
    return "—" if mm is None else f"{_s(mm / MM_IN, nd)} in"


def _s(x, nd=2) -> str:
    """Signed, never '-0.00' (a value that rounds to zero is '0.00')."""
    from . import report_insights as ri
    return ri.signed(x, nd)


def _dir(mm, rng):
    """'sinking' / 'rising' only when the 95% range excludes zero (stats-9)."""
    from . import report_insights as ri
    return ri.direction(mm, rng)


def _ri():
    from . import report_insights as ri
    return ri


def _b64(p):
    return base64.b64encode(Path(p).read_bytes()).decode()


# System fonts only: the page is self-contained and makes no request when
# opened (gap-licensing-11: two Google Fonts links made every client's browser
# contact Google, though README.txt calls the report self-contained).
CSS = """
:root{--paper:#f4f5f2;--card:#ffffff;--ink:#1c2530;--muted:#58636e;--rule:#d8ddd6;--accent:#2d5f7c;
--down:#a8432f;--up:#2d5f7c;--ok:#3d7a4c;--band:#e8ece6;
--sans:"Segoe UI",system-ui,-apple-system,"Helvetica Neue",Arial,sans-serif;--mono:ui-monospace,Menlo,Consolas,monospace}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){color-scheme:dark;--paper:#14181c;--card:#1b2026;--ink:#e6eaee;
--muted:#9aa4ae;--rule:#2c333b;--accent:#78a9c8;--down:#df8a73;--up:#86b6d8;--ok:#7fb98d;--band:#20262d}}
:root[data-theme="dark"]{color-scheme:dark;--paper:#14181c;--card:#1b2026;--ink:#e6eaee;--muted:#9aa4ae;--rule:#2c333b;
--accent:#78a9c8;--down:#df8a73;--up:#86b6d8;--ok:#7fb98d;--band:#20262d}
body{background:var(--paper);color:var(--ink);font-family:var(--sans);font-size:17px;line-height:1.6;margin:0;
padding-block:0 72px;padding-inline:16px}
main{max-width:920px;margin:0 auto}
.prose{max-width:68ch}
header{padding-block:44px 20px;border-bottom:2px solid var(--ink);margin-bottom:28px}
.kicker{font-family:var(--mono);font-size:12.5px;letter-spacing:.08em;text-transform:uppercase;color:var(--muted)}
.kicker .flag{color:var(--down);font-weight:700}
h1{font-size:clamp(30px,5vw,44px);line-height:1.08;margin:.25em 0 .35em;font-weight:700;letter-spacing:-.01em;text-wrap:balance}
h2{font-size:23px;margin:48px 0 10px;font-weight:700;text-wrap:balance}
h3{font-size:15px;margin:22px 0 4px;font-weight:600;letter-spacing:.04em;text-transform:uppercase;color:var(--muted)}
p{margin:.55em 0}.muted{color:var(--muted)}.small{font-size:15px}
.answer{background:var(--card);border:1px solid var(--rule);border-left:5px solid var(--accent);padding:18px 22px;margin:24px 0;font-size:18.5px}
.answer.hold{border-left-color:var(--down)}
.tablewrap{overflow-x:auto;margin:14px 0}
table{border-collapse:collapse;width:100%;background:var(--card);border:1px solid var(--rule);font-size:15.5px}
th,td{text-align:left;padding:12px 14px;border-top:1px solid var(--rule);vertical-align:top}
thead th{border-top:0;font-size:12.5px;letter-spacing:.06em;text-transform:uppercase;color:var(--muted);font-weight:600}
th[scope=row]{font-weight:700;min-width:9em}
.num{font-family:var(--mono);font-variant-numeric:tabular-nums;white-space:nowrap}
.big{font-size:19px;font-weight:600}
.down{color:var(--down)}.up{color:var(--up)}
.tag{display:inline-block;font-size:13px;font-weight:600;padding:2px 10px;border-radius:3px;border:1px solid currentColor;white-space:nowrap}
.tag.ok{color:var(--ok)}.tag.hit{color:var(--down)}.tag.gnss{color:var(--accent)}
figure{margin:22px 0;background:#fff;border:1px solid var(--rule);padding:10px}
figure img{display:block;width:100%;height:auto}
figcaption{font-size:14.5px;color:#58636e;padding:10px 4px 2px;line-height:1.5}
dl dt{font-weight:700;margin-top:14px}dl dd{margin:3px 0 0 0;color:var(--muted)}
ul{padding-left:1.2em}li{margin:.35em 0}
footer{margin-top:56px;padding-top:14px;border-top:1px solid var(--rule);font-size:13.5px;color:var(--muted)}
footer p{margin:.4em 0}
@media print{body{background:#fff;font-size:11pt;padding:0}.answer,table,figure{break-inside:avoid}
h2{break-after:avoid}figure{border:0}}
"""
# Only a page that is not final carries this (package.build refuses a final
# package whose report contains the word): a watermark on every screen and
# printed page.
DRAFT_CSS = """
body::before{content:"DRAFT";position:fixed;top:38%;left:50%;transform:translate(-50%,-50%) rotate(-28deg);
font-size:min(26vw,220px);font-weight:800;letter-spacing:.08em;color:rgba(168,67,47,.11);pointer-events:none;z-index:50}
"""


def rising_limit_differs(a: dict) -> bool:
    """The rising detection limit differs from the sinking one by more than
    RISING_LIMIT_DIFFERS of it (E5): only then is it printed beside it."""
    sink, rise = a.get("detection_limit_mm_yr"), a.get("detection_limit_rise_mm_yr")
    return bool(sink and rise and a.get("detection_limit_basis") == "sinking"
                and abs(float(rise) - float(sink)) > RISING_LIMIT_DIFFERS * float(sink))


def detection_sentence(a: dict) -> str:
    """Kyle's wording (5 Oct 2026) for an area's 80%-power detection limit,
    for SINKING since 7 Oct 2026 (E5: the limit printed until then was the
    worse sign's - rising, on the skewed farmland null - so sinking of that
    size was caught about 9 times in 10, not 4 in 5); the rising limit is
    added only when it differs by more than a tenth. A record from before 7
    Oct (no per-sign limit) keeps the old wording."""
    lim = a.get("detection_limit_mm_yr")
    if not lim:
        return ""
    if a.get("detection_limit_basis") != "sinking":
        return (f"Sustained movement of {_in(lim, 2)}/yr ({lim:.0f} mm/yr) or more across this area would be detected "
                "4 times in 5.")
    out = (f"Sustained sinking of {_in(lim, 2)}/yr ({lim:.0f} mm/yr) or more across this area would be detected "
           "4 times in 5")
    if rising_limit_differs(a):
        rise = float(a["detection_limit_rise_mm_yr"])
        out += f"; rising ground needs {_in(rise, 2)}/yr ({rise:.0f} mm/yr) for the same"
    return out + "."


def _about_m(x, default: float = 500.0) -> int:
    """A smoothing distance as the page says it: to the nearest 50 m (the
    map's 496 m read 'about 496 m', a precision the averaging does not
    have; final check of 7 Oct 2026)."""
    try:
        x = float(x)
    except (TypeError, ValueError):
        x = default
    return int(50 * round(x / 50)) if x >= 75 else int(round(x))


def patch_rate_cells(p: dict, feats: list, smoothing_m=None) -> tuple[str, str]:
    """(rate, what it is) for a row of the coloured-patches table. A patch
    that IS a numbered feature carries the feature's own figure - the one
    its Observations paragraph and the KMZ give - and says the map's value
    is the smoothed one: Site B Feature 1 read -1.38 in/yr here (the
    500 m map patch) and 1.47 in/yr in Observations, with nothing saying
    why (review outcome check, 6 Oct 2026)."""
    sm = f"{_about_m(smoothing_m)} m" if smoothing_m else "smoothed"
    mv = p.get("mean_rel_vertical_mm_yr")
    map_v = f"{_s(mv / MM_IN)} in/yr" if mv is not None else "—"
    n = p.get("feature")
    if not n:
        return map_v, "within what ordinary ground produces by chance"
    f = feats[n - 1] if isinstance(n, int) and 0 < n <= len(feats) else None
    if f is not None and f.get("ground_reshaped"):
        return "no rate", f"Feature {n} (Observations): ground reshaped during the record, so no rate is given"
    fv = None if f is None else f.get("vertical_relative_mm_yr", f.get("mean_rel_vertical_mm_yr"))
    if fv is None:
        return f"{map_v} ({sm} map value)", f"see Feature {n} (Observations)"
    # the feature's figure with its 95% range, as Observations gives it (M4;
    # review of the final check, 7 Oct 2026: the row printed it with none)
    try:
        lo, hi = sorted(float(x) for x in f.get("rate_range95_mm_yr"))
        rr = f"; its 95% range {_s(lo / MM_IN)} to {_s(hi / MM_IN)} in/yr"
    except (TypeError, ValueError):
        rr = ""
    return (f"{_s(fv / MM_IN)} in/yr (Feature {n}'s figure)",
            f"Feature {n} (Observations){rr}; the {sm} map averages {map_v} over this patch")


def thin_null_sentence(a: dict, ground: str = "farmland", what: str = "this range and detection limit") -> str:
    """How much the farmland comparison behind an area's range rests on
    (methodology-2): said only when it is thin or not the area's own shape.
    `what` names the figure it qualifies (the mine-ground observation, which
    prints no range, says "this comparison")."""
    n = a.get("approx_independent_placements")
    if a.get("null_thin") and n:
        return (f" Only about {n} comparison areas of this size fit in the surrounding {ground}, so {what} "
                f"{'are' if ' and ' in what else 'is'} approximate (widened to allow for it).")
    if a.get("null_shape") == "compact_square":
        return (f" Compared with square {ground} patches of the same area (its own outline does not fit in the "
                f"{ground}).")
    return ""


def totals_differ(tot, avg) -> bool:
    """The measured total and the average-rate total differ enough to print
    both (INT1: not only when the bend tested significant)."""
    if tot is None or avg is None:
        return False
    return abs(float(avg) - float(tot)) >= max(TOTALS_DIFFER_MM, TOTALS_DIFFER_SHARE * abs(float(tot)))


def recent_sentence(a: dict) -> str:
    """The last two full years' rate of an area with its own range
    (clientout-16: 'is it still moving now?')."""
    rv, rw = a.get("recent_rate_mm_yr"), a.get("recent_window") or []
    if rv is None or len(rw) != 2:
        return ""
    rr = a.get("recent_range95_mm_yr")
    return (f"Over {rw[0][:4]}–{rw[1][:4]}: {_s(rv / MM_IN)} in/yr ({rv:+.1f} mm/yr)"
            + (f", likely range {_s(rr[0] / MM_IN)} to {_s(rr[1] / MM_IN)} in/yr" if rr else "") + ".")


_BASIS_WORDS = {"formal error": "the stations' own measurement error",
                "scatter between the sites used": "the scatter between the stations used",
                "regional floor": "the scatter between GNSS stations across the wider region"}


def regional_gnss(rep: dict) -> dict:
    """
    The regional movement, said as what it is (methodology-5, clientout-10,
    stats-9, empirical-5, generality-14): the GNSS stations' rate - not the
    satellite's, which measures movement relative to the comparison ground -
    with its uncertainty, how that uncertainty was set and from how many
    stations, a direction word only when its 95% range excludes zero, and a
    warning when leaving one station out moves it by more than its
    uncertainty. 'as the regional GNSS stations also show' restated the GNSS
    number as a second, independent finding.
    """
    from . import report_insights as ri
    tie = rep.get("gnss") or {}
    bg = tie.get("background") or {}
    g, u = bg.get("vertical_mm_yr"), bg.get("vertical_uncertainty_mm_yr")
    if not rep.get("anchored") or g is None:
        return {"anchored": False, "sentence": ""}
    g, u = float(g), (float(u) if u else None)
    rng = [g - Z95 * u, g + Z95 * u] if u else None
    word = ri.direction(g, rng)
    n = bg.get("n_sites") or len(bg.get("sites") or []) or len(tie.get("stations_used") or [])
    basis = _BASIS_WORDS.get(str(bg.get("uncertainty_basis") or ""))
    # counted as the background counts them: two stations on one monument
    # group are one (finding 51c: '3 GNSS stations used' beside four names)
    merged = [str(x.get("site")) for x in bg.get("sites") or [] if isinstance(x, dict) and "+" in str(x.get("site"))]
    one = "; ".join(" and ".join(m.split("+")) + " count as one" for m in merged)
    who = (f"the {n} GNSS station{'s' if n != 1 else ''} used{(' (' + one + ')') if one else ''} give" if n
           else "the GNSS stations give")
    s = (f"{who} {_s(g / MM_IN)} in/yr ({g:+.1f} ± {u:.1f} mm/yr vertical, 1 sigma" if u else
         f"{who} {_s(g / MM_IN)} in/yr ({g:+.1f} mm/yr vertical")
    s += (f"; the uncertainty is {basis}" if basis else "") + ")"
    s += (f": the region is {word}" if word else
          (", which on its own cannot be told apart from no movement" if rng else ""))
    loo = tie.get("leave_one_out") or {}
    warn = ""
    if u and loo.get("max_change_mm_yr") is not None and float(loo["max_change_mm_yr"]) > u:
        site, val = max(((k, v) for k, v in (loo.get("without") or {}).items()), key=lambda kv: abs(kv[1] - g),
                        default=(None, None))
        if site:
            warn = (f" It rests heavily on one station: without {site} it would be {val:+.1f} mm/yr.")
    return {"anchored": True, "value": g, "u": u, "range": rng, "word": word, "n": n,
            "sentence": s[0].upper() + s[1:] + "." + warn}


def station_check(chk: dict, rep: dict) -> dict | None:
    """
    The check at a GNSS station inside the study area, with what it could
    have found (methodology-5: 'agrees whenever |rel| < 1.96 x 6.37 mm/yr'
    was sold as validation; it could fail only above ~0.6 in/yr, and the
    two series' shapes did not match). One rule for agreement: the tie's
    own threshold/distinguishable when it carries them, else 1.96 x the
    footprint spread. When the station's rate over its overlap is known, the
    satellite's relative rate is set against the station's rate minus the
    regional background. Returns {agrees, html} or None. A tie from 7 Oct
    2026 on (gnss_tie's same-window check, E16) carries its own verdict and
    sentence: agrees is its `consistent` (None: not testable) and the page
    prints its verdict_reason.
    """
    from . import report_insights as ri
    if chk and "verdict" in chk:
        # the tie's own same-window check (E16, gnss_tie from 7 Oct 2026): its
        # verdict and its plain sentence ('Not testable: ...' when the overlap
        # is too short to test at a useful tolerance - never 'agree' then).
        # Keyed on 'verdict', as mark_inside_checks is, so the page and the
        # gate can never read one check two ways
        c = chk.get("consistent")
        why = str(chk.get("verdict_reason") or "")
        site = html.escape(str(chk.get("site")))
        if c is None:
            # final check of 7 Oct 2026: "which allows a direct check of the
            # anchoring. Not testable: ..." said both things of one station
            why = why.strip() or "Not testable."
            return {"agrees": None,
                    "html": (f"<p>Station <b>{site}</b> sits inside the study area, but the anchoring could not be "
                             f"checked against it here. {html.escape(why)}</p>")}
        return {"agrees": bool(c),
                "html": (f"<p>Station <b>{site}</b> sits inside the study area, which allows a direct check of the "
                         f"anchoring. {html.escape(why)}</p>")}
    lu = (rep.get("gnss") or {}).get("los_up_median") or rep.get("los_up_median")
    if not chk or not lu or chk.get("insar_relative_class_matched_mm_yr") is None:
        return None
    e = html.escape
    lu = float(lu)
    ground = ri.ground_word(rep)
    rel_v = float(chk["insar_relative_class_matched_mm_yr"]) / lu
    thr = chk.get("threshold_95_mm_yr")
    thr_v = (float(thr) if thr is not None else Z95 * float(chk.get("farmland_spread_1sigma_mm_yr") or 0.0)) / lu
    bg = ((rep.get("gnss") or {}).get("background") or {})
    st_rel = None
    if chk.get("gnss_los_rate_over_overlap_mm_yr") is not None and bg.get("los_mm_yr") is not None:
        st_rel = (float(chk["gnss_los_rate_over_overlap_mm_yr"]) - float(bg["los_mm_yr"])) / lu
    dist = chk.get("distinguishable_from_farmland")
    if st_rel is not None:
        agrees = abs(rel_v - st_rel) <= thr_v
    else:
        agrees = (not dist) if dist is not None else abs(rel_v) < thr_v
    ov = chk.get("overlap")
    txt = f"<p>Station <b>{e(str(chk.get('site')))}</b> sits inside the study area, which allows a direct check. "
    if agrees:
        txt += (f"The satellite reads the ground around it moving with the surrounding {e(ground)} "
                f"({_s(rel_v / MM_IN)} in/yr relative, within the ±{thr_v / MM_IN:.2f} in/yr ordinary ground reaches)"
                + (f", and the station moved {_s(st_rel / MM_IN)} in/yr against the region over "
                   f"{ov[0]:.1f}–{ov[1]:.1f}: the two agree." if st_rel is not None and ov else "."))
    else:
        txt += (f"The satellite reads the ground around it {rel_v / MM_IN:+.2f} in/yr relative to the surrounding "
                f"{e(ground)}" + (f", where the station moved {st_rel / MM_IN:+.2f} in/yr against the region" if st_rel is not None else "")
                + f": they differ by more than this check can explain (± {thr_v / MM_IN:.2f} in/yr).")
    txt += (f" This check could only reveal a difference of about {thr_v / MM_IN:.2f} in/yr or more"
            + (f", and over the station's overlap with the record the short-term shapes of the two records "
               f"{'match' if float(chk['detrended_correlation']) >= 0.5 else 'do not match'} (correlation "
               f"{float(chk['detrended_correlation']):+.2f})" if chk.get("detrended_correlation") is not None else "")
            + (f"; the station recorded until {int(ov[1])}, the first {ov[1] - ov[0]:.1f} years of the record" if ov else "")
            + ".</p>")
    return {"agrees": bool(agrees), "html": txt}


def mark_inside_checks(rep: dict) -> None:
    """Each GNSS station inside the study area: `consistent` (True/False) by
    the one rule the page prints (station_check), stored on its
    gnss.inside_checks entry where the release gate reads it (release.blockers,
    'gnss_inside_inconsistent'). Nothing wrote it before 7 Oct 2026, so a page
    saying "they differ" could still be signed off. An entry the check cannot
    read (no satellite figure) gets no verdict."""
    g = rep.get("gnss")
    # no stations inside: leave the record as it was (an added empty list
    # would change its numbers hash and void a sign-off)
    if not isinstance(g, dict) or not g.get("inside_checks"):
        return
    g["inside_checks"] = [dict(c) for c in g.get("inside_checks") or [] if isinstance(c, dict)]
    for c in g["inside_checks"]:
        if "verdict" in c:
            # gnss_tie's same-window check (from 7 Oct 2026) carries its own
            # verdict: True / False / None (not testable) as it is
            c["consistent"] = c.get("consistent")
            continue
        sc = station_check(c, rep)
        if sc is not None:
            c["consistent"] = bool(sc["agrees"])


def _plural_reason(r: str) -> str:
    """A per-date reason said of several dates ('returned it incomplete' ->
    'returned them incomplete'), the date-accounting reasons that open with
    'it' included ('it falls outside ...' -> 'they fall outside ...')."""
    r = " " + r                                   # a reason that opens with 'it' matches too
    for a, b in ((" returned it incomplete", " returned them incomplete"), (" on this date", " on these dates"),
                 (" check this date", " check these dates"), (", so it was not assembled", ", so they were not assembled"),
                 (", so it could not be assembled", ", so they could not be assembled"),
                 (", so it was left out", ", so they were left out"),
                 # quality_assessment.date_accounting / _DROP_REASON (7 Oct 2026)
                 (" it is not in", " they are not in"), (" it is in the", " they are in the"),
                 (" but was not downloaded", " but were not downloaded"),
                 (" but was never assembled", " but were never assembled"),
                 (" it falls outside", " they fall outside"), (", so it was not processed", ", so they were not processed"),
                 (" it was found in", " they were found in"), (", so it could not be added", ", so they could not be added"),
                 (" but it never entered", " but they never entered"), (" of this date", " of these dates"),
                 (" left it out", " left them out"), (" for this DISP-S1 epoch", " for these DISP-S1 epochs"),
                 (" granule could not be read and was skipped", " granules could not be read and were skipped"),
                 (" there is no orbit-stack perpendicular baseline", " there are no orbit-stack perpendicular baselines"),
                 # 30 Oct 2026: every reason a clause after "because"
                 (" it could not be aligned", " they could not be aligned"),
                 (", so it was kept out", ", so they were kept out"),
                 (" it was assembled or coregistered but", " they were assembled or coregistered but"),
                 (" all of its interferograms were", " all of their interferograms were"),
                 (" the time of this pass", " the times of these passes"),
                 (", so its weather delay could not be", ", so their weather delays could not be"),
                 (" serves this date in a form", " serves these dates in a form")):
        r = r.replace(a, b)
    return r[1:]


# A quality check that cannot apply to a record, in plain words for the
# client ("{ground}" is the comparison ground's label); any other is named
# from its key.
_NOT_APPLICABLE_PLAIN = {
    "distributed_scatterer_bias": ("the check that ground measured less clearly does not read differently from "
                                   "ground measured more clearly, because the ordinary {ground} used for comparison "
                                   "here is all of one measurement quality"),
}


def quality_checks_text(rep: dict) -> str:
    """The methods table's "Quality checks" line (HTML), counted as the
    verdict counts them (D2, 7 Oct 2026; quality_assessment.check_counts):
    "N of M applicable checks flagged", any that could not be run, and each
    check that cannot apply to this record named and not counted. A report
    from before D2 carries no applicable count or names: its met + flagged
    is the count, as it printed then."""
    passed, flagged = int(rep.get("checks_passed") or 0), int(rep.get("checks_flagged") or 0)
    untested = int(rep.get("checks_untested") or 0)
    applicable = rep.get("checks_applicable")
    applicable = int(applicable) if applicable is not None else passed + flagged + untested
    text = (f"Checked against field practice: {flagged} of {applicable} applicable "
            f"check{'' if applicable == 1 else 's'} flagged"
            + (f", {untested} could not be run" if untested else "") + f", and {passed} met.")
    na = [str(n) for n in rep.get("checks_not_applicable") or []]
    if na:
        ground = rep.get("control_ground_label") or "farmland"
        what = "; ".join(_NOT_APPLICABLE_PLAIN[n].format(ground=ground) if n in _NOT_APPLICABLE_PLAIN
                         else f"the {n.replace('_', ' ')} check" for n in na)
        one = len(na) == 1
        text += (f" {len(na)} more check{'' if one else 's'} {'does' if one else 'do'} not apply to this record and "
                 f"{'is' if one else 'are'} not counted: {what}.")
    return html.escape(text) + " “Reading the numbers” above says what the flags mean for these figures."


# A dropped date's reason as the quality assessment words it -> the page's
# words (final check of 7 Oct 2026: 'failed the reference-pixel check').
_PLAIN_DROP_WORDS = (("failed the reference-pixel check", "could not be tied to the reference point"),
                     ("reference pixel", "reference point"))


# A tool's own error text has no place on the client's page (16 Oct 2026,
# readiness review ERR-02: a dropped date read "burst assembly failed: exit 1:
# ERROR conda.cli.main_run ..."). The plain part before the first ': ' stays;
# the full reason is kept in client_report.json's record_dates.
_TOOL_ERROR = re.compile(r"\bexit \d+\b|\bERROR\b|Traceback|`|\bconda\b|[A-Za-z]+Error\b|\.py\b|/home/|/tmp/")


def _plain_drop_reason(r: str) -> str:
    head, sep, detail = r.partition(": ")
    if sep and _TOOL_ERROR.search(detail):
        r = head
    for a, b in _PLAIN_DROP_WORDS:
        r = r.replace(a, b)
    return r


def methods_stages(cfg, rep: dict) -> list[tuple[str, str]]:
    """The Methodology entries, each from what this run did (generality-16,
    tests-6, methodology-9, stackrunner-2, tests-2, disp-1)."""
    from . import report_insights as ri
    e = html.escape
    ground = ri.ground_word(rep)
    d0, d1 = rep["dates"]
    ymd = lambda d: datetime.strptime(d, "%Y%m%d").strftime("%B %Y")   # noqa: E731
    route = rep.get("backend")
    gap = rep.get("revisit_days")
    rd = rep.get("record_dates") or {}
    data = (f"{rep['n_dates']} {sensor_name(route)} radar images, {ymd(d0)} to {ymd(d1)}, from "
            f"{'an ascending' if (rep.get('orbit') or '').lower().startswith('asc') else 'a descending'} pass "
            f"(track {rep.get('track')})" + (f", typically one every {gap:.0f} days" if gap else "") + ". ")
    if rd.get("available") and rd.get("processed") is not None:
        drop = rd.get("dropped") or {}
        data += f"{rd['processed']} of {rd['available']} available images were used"
        if drop:
            reasons = {}
            for d_, why in drop.items():
                reasons.setdefault(_plain_drop_reason(str(why).split(";")[0].split(" (")[0]), []).append(d_)
            data += ("; " + "; ".join(f"{len(v)} left out because {_plural_reason(r) if len(v) > 1 else r}"
                                      for r, v in list(reasons.items())[:3]))
        data += ". "
    data += "Radar works day and night and through cloud."
    res = rep.get("resolution") or {}
    if route == "disp":
        proc = ("SCREENING ONLY: OPERA DISP-S1, a displacement product made by NASA/JPL from the same satellite, was "
                "taken as the starting point. It was put through this report's comparison with ordinary ground and "
                "its quality checks, but it is not the full processing, and its figures are for screening, "
                "not for decisions.")
    elif route == "nisar":
        proc = ("COMPARISON, NOT VALIDATED: the starting point was NASA-ISRO's ready-made NISAR interferograms "
                "(the PROVISIONAL release; a validated reprocessing is due), each between two images 12 days apart. "
                "NISAR's L-band radar has a wavelength about four times Sentinel-1's and looks to the left of its "
                "path where Sentinel-1 looks right, so it sees a different mix of up-down and sideways movement. "
                "The interferograms were put on one grid, inverted into a time series for each point and put "
                "through this report's comparison with ordinary ground and its quality checks. NISAR figures have "
                "not yet been validated against independent measurements here; they are for comparison, not for "
                "decisions.")
    elif route == "miaplpy":
        pl = rep.get("phase_linking") or {}
        proc = "The images were aligned to within a few centimetres of each other. "
        if pl.get("sequential") and pl.get("window_months"):
            proc += (f"Phase linking then estimated each patch of ground's movement from the similar ground around "
                     f"it, combining each image with its neighbours in overlapping windows of about "
                     f"{pl['window_months']:.0f} months ({pl['window_images']} images, each window tied to the one "
                     f"before through {pl['time_lag']} shared images), so the whole record keeps one reference. ")
        else:
            proc += ("Phase linking then estimated each patch of ground's movement from the similar ground around it, "
                     "using every image together. ")
        proc += "The radar phase was unwrapped into continuous movement and inverted into a time series for each point."
    else:
        proc = ("The images were aligned to within a few centimetres of each other. Interferograms were formed from "
                "pairs of images, unwrapped into continuous movement and inverted into a time series for each point.")
    pts = ""
    if res.get("text"):
        pts = res["text"] + " Features smaller than that are blurred into the ground around them or missed."
    nind = rep.get("independent_radar_measurements")
    full = next((a for a in rep["areas"] if a.get("kind") == "full"), None)
    if nind and full and full.get("cells"):
        # E25: counted over the study area, and not called independent - the
        # radar cells' own ground overlaps (each value combines the larger
        # ground the resolution sentence gives)
        pts += (f" The {full['cells']:,} points in the study area come from about {int(nind):,} radar resolution "
                "cells: neighbouring points share one, and neighbouring cells share ground, so they are not "
                "independent measurements"
                + (" (an estimate)" if "estimated" in str(rep.get("independent_radar_measurements_basis") or "") else "")
                + ".")
    corr = rep.get("corrections") or []
    ref = rep.get("reference") or {}
    if ref.get("auto"):
        # no "stable" and no "most reliable": the coherence the chooser recorded
        # was for a pixel 60 m from the one MintPy used (empirical-12)
        reft = "A reference point was chosen automatically in the far field, away from the mapped mines"
        if ref.get("clears_buffer") and ref.get("distance_to_nearest_mapped_mine_km") is not None:
            reft += f", about {float(ref['distance_to_nearest_mapped_mine_km']):.1f} km from the nearest mapped mine"
        reft += ". "
    else:
        reft = "Movement is measured from a fixed reference point. "
    reft += f"No figure in this report depends on it: each is measured against the surrounding {e(ground)}."
    buf = ref.get("far_field_buffer_m") or getattr(cfg, "acceptance_far_field_buffer_m", None)
    buf_t = (f"more than {float(buf) / 1000:g} km from any mapped mine or known movement" if buf
             else "away from any mapped mine or known movement")
    anchored = bool(rep.get("anchored"))
    reg = regional_gnss(rep)
    # only an area whose figures were widened is said to be (findings 36/49:
    # Area A1, compared with ~16 square patches and not widened, and
    # Site B's untested project area were both named here)
    tested_a = [a for a in rep["areas"] if a.get("kind") == "area" and a.get("testable") is not False]
    thin = [a for a in tested_a if a.get("null_thin")]
    square = [a for a in tested_a if not a.get("null_thin") and a.get("null_shape") == "compact_square"]
    has_limits = any(a.get("detection_limit_mm_yr") for a in tested_a)
    unc = (f"The likely ranges{' and detection limits' if has_limits else ''} come from repeating each measurement on "
           f"ordinary {e(ground)} around the site - the area's own outline moved across every patch it fits - not "
           "from a formula.")
    if square:
        unc += (" For " + ", ".join(e(a.get("display_name") or a["name"]) for a in square) + " the outline itself fits "
                "only a few times, so square patches of the same area were used instead"
                + "".join(f" ({e(a.get('display_name') or a['name'])}: about {int(a['approx_independent_placements'])} "
                          "separate patches)" for a in square if a.get("approx_independent_placements"))
                + "; those figures are not widened.")
    if thin:
        unc += (" For " + ", ".join(e(a.get("display_name") or a["name"]) for a in thin) + " even those fit only a few "
                "times, so the figures are approximate and have been widened to allow for it.")
    if any(a.get("untested_range95_mm_yr") for a in rep["areas"]):
        unc += (" An area that could not be compared with ordinary ground is shown with the uncertainty that can be "
                "stated without that comparison (the regional GNSS rate's, and the spread of the few patches of its "
                "size that fit); it says nothing about whether the area moves differently.")
    stages = [
        ("Data", e(data)),
        ("Processing", e(proc)),
        ("Measurement points", e(pts.strip()) if pts.strip() else "Each point is a map cell of the product grid."),
        ("Corrections", ("Removed from the signal: " + "; ".join(e(c) for c in corr) + ".") if corr else
         "No corrections beyond the processing itself."),
        ("Reference", e(reft)),
        ("Comparison", f"Each area is compared with ordinary {e(ground)} of the same measurement quality, {buf_t}. "
                       "That cancels effects common to the whole scene, including seasonal soil and crop changes."
                       # E28: an area is never part of its own comparison ground
                       + (" The report areas themselves are kept out of it, so no area is compared with itself."
                          if rep.get("report_areas_out_of_farmland") and any(a.get("kind") == "area" for a in rep["areas"])
                          else "")
                       + (" Local movement found standing out from chance is taken out of it as well."
                          if (rep.get("farmland_second_pass") or {}).get("features_removed") else "")),
        ("Anchoring", ("The regional movement measured by nearby GNSS stations"
                       + (" (" + e(_stations_text(rep)) + ")" if rep.get("gnss_stations_used") else "")
                       + " is added to the report areas' figures in the table, "
                       "so those are absolute, on the assumption that the surrounding "
                       f"{e(ground)} moves with the stations' monuments; features, mine ground and the map stay "
                       f"relative to the {e(ground)}." if anchored and reg.get("anchored") else
                       "No GNSS station could anchor the figures" + (f" ({e(str(rep.get('gnss_reason')).rstrip('.'))})" if rep.get("gnss_reason") else "")
                       + f"; figures are relative to the area's {e(ground)}, and the study area as a whole has no figure.")),
        ("Totals", "“Measured over the record” is how far the ground moved between the first and the last image. It "
                   "is the average rate times the years unless the ground's change of pace stands out from what "
                   f"patches of ordinary {e(ground)} of the same size show; then it is read from a curve fitted to "
                   "every image (it may change pace once a year; the seasonal swing is removed), and where the two "
                   "differ noticeably both are given. Each total has a 95% range from the same comparison with "
                   f"ordinary {e(ground)}" + (", plus the regional GNSS rate's uncertainty over the years" if anchored
                                                else "") + "."),
        ("Uncertainty", unc),
        ("Local features", "Features are found by searching the whole area for patches beyond the normal range, so some "
                           "are expected by chance. Each is kept only if it is larger and stronger than what simulated "
                           f"copies of this area's own {e(ground)} produce under the same search."
                           # F1: the leave-candidate-out test, said when it was made
                           + (f" The strongest patches are also tested with themselves left out of the {e(ground)} "
                              "the copies are made from, so a large patch cannot hide by widening the comparison it "
                              "is judged against; the chance given for a patch kept this way allows for how many "
                              "patches were tested."
                              if (rep.get("feature_leave_out") or {}).get("candidates") else "")
                           # M4 (final check of 7 Oct 2026)
                           + " A feature's outline is drawn where the measurements go furthest beyond the normal "
                             "range, so its rate, total and ranges are for that outline and read somewhat stronger "
                             "than the ground there truly moved."),
        ("Quality checks", quality_checks_text(rep)),
    ]
    return stages


def _stations_text(rep: dict) -> str:
    """The GNSS stations the anchor used, as the background groups them
    ('INES and INEV (one site), INVI, VINU'; finding 51c), else their IDs."""
    sites = [str(x.get("site")) for x in (((rep.get("gnss") or {}).get("background") or {}).get("sites") or [])
             if isinstance(x, dict) and x.get("site")]
    if sites:
        return ", ".join((" and ".join(x.split("+")) + " as one site") if "+" in x else x for x in sites)
    return ", ".join(str(x) for x in rep.get("gnss_stations_used") or [])


def limitations(cfg, rep: dict) -> list[str]:
    """The Limitations list (HTML items), from the run: resolution
    (generality-2), the fast-movement limit from the wavelength, the revisit
    and the look cell (tests-6, empirical-7), the line-of-sight factor with
    the mining horizontal motion (methodology-15, stats-16), the mine
    layers' vintage (gap-licensing-5), the screening route (disp-1)."""
    from . import report_insights as ri
    e = html.escape
    has_limits = any(a.get("kind") == "area" and a.get("detection_limit_mm_yr") for a in rep.get("areas") or [])
    # final check of 7 Oct 2026: Site B's page spoke of 'the detection
    # limits' and gave none (no report area could be tested)
    first = ("Steady movement below the detection limits may be missed; a shorter episode, or movement over only part "
             "of an area, can be larger than the limits and still be missed." if has_limits else
             f"No report area could be compared with ordinary {e(ri.ground_word(rep))}, so no detection limit can "
             "be stated: movement there, steady or not, may be missed.")
    out = [first,
           "Ground that could not be measured (water, woodland, active workings, fresh spoil) is not covered by any "
           "figure; each area says how much of it was measured."]
    res = rep.get("resolution") or {}
    if res.get("text"):
        out.append(e(res["text"]) + " A feature narrower than that is blurred into the ground around it; "
                   "a single point is not a measurement of its own 15-by-15 patch of ground."
                   .replace("15-by-15", f"{res.get('posting_m', 15):.0f}-by-{res.get('posting_m', 15):.0f} m"))
    gl = rep.get("gradient_limit") or {}
    look = (res.get("look_cell_m") or [None])[0]
    if gl.get("quarter_wavelength_mm"):
        q = float(gl["quarter_wavelength_mm"])
        out.append(f"Very fast, localised movement - more than about {q / MM_IN:.2f} in ({q:.0f} mm) of difference "
                   + (f"between neighbouring radar cells about {look:.0f} m apart " if look else "between neighbouring measurements ")
                   + f"within {float(gl.get('median_gap_days') or 12):.0f} days - cannot be followed by the radar and "
                   "may be missed or understated (marked in quality_too_fast_to_measure.tif where it was found).")
    ang = rep.get("look_angle_deg")
    los = ("Movement is measured along the radar's line of sight, "
           + (f"about {float(ang):.0f}° from vertical, " if ang else "") + "and reported as vertical (all of it is "
           "taken as up or down).")
    k = rep.get("los_east_over_up")
    if k:
        los += (f" Sideways movement is read as part of it: from this satellite track, east-west ground movement of "
                f"1 in/yr reads as about {float(k):.1f} in/yr of vertical movement.")
    if rep.get("mine_layers"):
        pct = (float(k) if k else 0.8) * ri.MINING_HORIZONTAL_SHARE * 100
        los += (f" Over mines the ground at the edges of a sinking area also moves sideways (often "
                f"{ri.MINING_HORIZONTAL_SHARE - 0.05:.1f}-{ri.MINING_HORIZONTAL_SHARE + 0.05:.1f} times the sinking), so "
                f"figures at those edges can be off by roughly ±{pct:.0f}%, in opposite directions on the east and "
                "west sides; a second satellite track would separate the two.")
    out.append(los)
    end_year = int(rep["dates"][1][:4])
    for L in rep.get("mine_layer_info") or []:
        v = L.get("vintage")
        if v is None:
            out.append(f"The outlines of the mapped {e(L['label'])} are used as supplied; when they were mapped is not "
                       "recorded, so mining or reclamation since then may be missing from them.")
        elif int(v) < end_year:
            out.append(f"The outlines of the mapped {e(L['label'])} are mapped to {int(v)}; mining or reclamation after "
                       f"that is not shown, while the record runs to {end_year}.")
    if rep.get("route") == "screening" or rep.get("backend") == "disp":
        out.append("This is a SCREENING report from a ready-made product (OPERA DISP-S1), not the full "
                   "processing; its figures are for deciding where to look, not for decisions.")
    if rep.get("backend") == "nisar":
        out.append("This is a COMPARISON report from NISAR's ready-made interferograms (PROVISIONAL release, "
                   "under a year of data). NISAR figures have not yet been validated against independent "
                   "measurements; use them beside the Sentinel-1 results, not for decisions.")
    return out


def glossary(rep: dict) -> list[tuple[str, str]]:
    """The terms the page uses, in plain words (clientout-16)."""
    from . import report_insights as ri
    ground = ri.ground_word(rep)
    # the two area-test terms only when some report area was tested (final
    # check of 7 Oct 2026: Site B's page defined them and used neither);
    # a record that lists no areas at all keeps them
    areas = rep.get("areas")
    tested = areas is None or any(a.get("kind") == "area" and a.get("detection_limit_mm_yr") for a in areas)
    return [
        ("Line of sight", "The direction from the ground to the satellite, slanted from vertical. The radar measures "
                          "movement along it; this report converts it to up or down."),
        ("GNSS", "Global navigation satellite systems (GPS and others). Permanent stations measure their own position "
                 "to millimetres and give the regional movement."),
        ("Comparison ground", f"Ordinary {ground} around the site, away from mapped mines, measured in the same record. "
                              "Every area is compared with it."),
        ("95% range", "Where the true value lies with 95% confidence, given this record's own measured noise."),
        ("1 sigma", "The typical size of a figure's error: the true value lies within ± that amount about two times "
                    "in three. A 95% range is about twice as wide."),
    ] + ([
        ("Detection limit", "The sustained sinking across an area that would be detected 4 times in 5 (rising "
                            "ground is given beside it where its limit differs noticeably)."),
        ("Signal ÷ threshold", f"An area's movement beyond ordinary {ground}, divided by the 95% threshold measured on "
                               f"{ri.ground_noun(ground)} patches of its size: 1 or more stands out."),
    ] if tested else []) + [
        ("Coherence", "How consistently the radar signal from a patch of ground repeats between images; low on crops, "
                      "water and woodland."),
        ("Phase linking", "Estimating each patch's movement from many images and similar neighbouring ground together, "
                          "which keeps more of the ground measurable."),
        ("Reference point", "The point all raw radar measurements are made against. This report's figures are made "
                            "against the comparison ground instead."),
        ("Posting and resolution", "Posting is how far apart the points are; resolution is how much ground each "
                                   "value combines. Resolution is coarser."),
    ]


def footer_html(cfg, rep: dict, final: bool) -> str:
    """Version, preparer, reviewer, date, disclaimer; the project's ID, never
    a local path (clientout-9, gap-licensing-11, methodology-10)."""
    e = html.escape
    ver = rep.get("version") or {}
    vl = (f"insar_pipeline {ver.get('version')} (code {ver.get('code_hash')})" if ver.get("version")
          else (_optional(lambda: _module("version").label(), "footer_version") or "insar_pipeline"))
    ident = _optional(lambda: _module("settings").identity(), "footer_identity") or {}
    who = ", ".join(x for x in (ident.get("preparer"), ident.get("firm")) if x)
    rel = rep.get("release") or {}
    if final:
        rv = f"Reviewed by {e(str(rel.get('reviewed_by') or '?'))} on {e(str(rel.get('review_date') or '?')[:10])}."
    else:
        rv = "Not yet reviewed and signed off: this is a DRAFT."
    gen = str(rep.get("generated") or "")
    try:
        gen = datetime.fromisoformat(gen).strftime("%d %B %Y")
    except ValueError:
        pass
    pid = Path(str(rep.get("project") or getattr(cfg, "project_dir", ""))).name
    # the route only when it is the screening one (final check of 7 Oct 2026:
    # 'route: official' is the operator's label, not a client's word)
    return (f"<footer><p>{e(rep.get('title') or '')} ground movement report · project {e(pid)} · {e(gen)} · {e(vl)}"
            + (f" · route: {e(str(rep.get('route')))}" if rep.get("route") == "screening" else "") + "</p>"
            + (f"<p>Prepared by {e(who)}" + (f" ({e(ident['contact'])})" if ident.get("contact") else "") + ". " if who
               # visible on every draft until set (clientout-9); release.blockers
               # holds a final report for it
               else "<p><b>Prepared by: not set</b> (insar doctor --firm ... --preparer ...). ") + rv + "</p>"
            + f"<p>{e(GENERATED_NOTE)}</p><p>{e(DISCLAIMER)}</p><p>{e(TECHNICAL_RECORD_NOTE)}</p></footer>")


def colour_steps_text(ms: dict) -> str:
    """The map caption's colour steps, from each side's own edges (finding
    35: Site A's caption said 'Colours step at 0.50, 0.75, 1.00 in/yr
    each way' while its sinking side had no 0.50 step - a light-red patch
    could be anything from -0.42 to -0.75 in/yr)."""
    edges = [float(x) for x in ms.get("edges_mm_yr") or []]
    nb = int(ms.get("n_below") or 0)
    if len(edges) < nb + 2:
        return ""
    neg = [abs(x) for x in reversed(edges[:nb])]                      # nearest the band first
    pos = edges[nb + 2:]
    top = f"the darkest means {ms['top_in_per_yr']:g} in/yr or more." if ms.get("top_in_per_yr") is not None else ""
    f = lambda xs: ", ".join(f"{x / MM_IN:.2f}" for x in xs)          # noqa: E731
    if [round(x, 1) for x in neg] == [round(x, 1) for x in pos]:
        return f"Colours step at {f(pos)} in/yr each way; " + top
    return (f"Colours step at {f(neg) or 'none'} in/yr on the sinking side and {f(pos) or 'none'} in/yr on the rising "
            "side; " + top)


def mine_outlines_line(rep: dict) -> str:
    """The sentence (HTML, escaped) naming the mine and known-movement
    outlines the report used and where each came from. The one untyped
    layer's source already names what it holds - 'dated mined areas (Indiana
    Department of Natural Resources, ...)' (report_insights.layer_sources) -
    so its generic label is not put in front of it."""
    e = html.escape
    info = rep.get("mine_layer_info") or [{"label": x, "source": None} for x in rep.get("mine_layers") or []]
    if not info:
        return "No mine outlines were supplied, so features cannot be related to mining."
    return ("Mine and known-movement outlines used (as supplied): " + "; ".join(
        e(L["source"]) if L.get("key") == "mapped" and re.match(r"[a-z][a-z ]* \(", str(L.get("source") or ""))
        else e(L["label"]) + " - " + e(L.get("source") or "source and date not recorded") for L in info) + ".")


def report_revision(cfg) -> dict | None:
    """
    The config's report_revision (M3, final check of 7 Oct 2026) as the
    report prints it - config.report_revision's own note, {number,
    supersedes, supersedes_text, heading "Revision N, replacing the report
    issued <date>", changes}, so the page, the PDF, the KMZ and the release
    gate (reissue_without_note) cannot word it differently. A reissue said
    nothing until then: Site A's report of 7 Oct differed from the one
    delivered on 28 Sep (a feature listed, every figure about 1 mm/yr lower,
    new detection limits and GNSS rate) with no word of it. None for a
    first issue, and for no config (a page rebuilt from its record alone).
    """
    if cfg is None:
        return None
    from .config import report_revision as _note
    return _note(cfg)


def revision_html(rep: dict) -> str:
    """The page's revision box: the heading and what changed (M3)."""
    rv = rep.get("report_revision") or {}
    if not rv.get("heading"):
        return ""
    e = html.escape
    items = "".join(f"<li>{e(c)}</li>" for c in rv.get("changes") or [])
    return (f'<div class="answer prose small"><b>{e(rv["heading"])}.</b>'
            + (f" What changed:<ul>{items}</ul>" if items else "") + "</div>")


def page(cfg, rep, figs) -> str:
    from . import report_insights as ri
    e = html.escape
    A = rep["areas"]
    full = next((a for a in A if a.get("kind") == "full"), None)
    regions = [a for a in A if a.get("kind") == "area"]
    anchored = bool(rep.get("anchored", any(a.get("anchored_to_gnss") for a in A)))
    tie = rep.get("gnss") or {}
    d0, d1 = rep["dates"]
    ymd = lambda d: datetime.strptime(d, "%Y%m%d").strftime("%B %Y")   # noqa: E731
    span = rep["span_years"]
    title = rep.get("title") or "Ground movement"
    ground = ri.ground_word(rep)
    nm = lambda a: e(a.get("display_name") or a["name"])               # noqa: E731
    feats = rep.get("features") or []
    standing = [f for f in feats if f.get("stands_out", True)]
    untested_feats = [f for f in feats if f.get("stands_out", True) is None]
    rel = rep.get("release") or {}
    final = bool(rel.get("final"))
    screening = rep.get("route") == "screening" or rep.get("backend") == "disp"
    reg = regional_gnss(rep)

    def rate_cell(a):
        mm = a.get("vertical_mm_yr")
        if mm is None:
            return ('<span class="big num">relative only</span>' if a.get("relative_only") and a.get("kind") == "full"
                    else '<span class="big num">—</span>')
        word = area_word(a)
        # an anchored area that moves only with the region is not coloured as moving on its own
        cls = "" if moves_with_region(a) else {"sinking": "down", "rising": "up"}.get(word, "")
        word += " with the region" if word and moves_with_region(a) else ""
        # to 0.01 in/yr (E9: '0.000 in/yr' claimed 50 times the precision the data have)
        return (f'<span class="big num {cls}">{_s(mm / MM_IN)} in/yr</span><br><span class="small muted num">'
                f'{mm:+.1f} mm/yr{(" " + word) if word else ""}</span>')

    def range_cell(a):
        rng = a.get("range95_mm_yr")
        if "range95_mm_yr" not in a and a.get("vertical_mm_yr") is not None and a.get("vertical_uncertainty_1sigma_mm_yr"):
            u = a["vertical_uncertainty_1sigma_mm_yr"]
            rng = [a["vertical_mm_yr"] - Z95 * u, a["vertical_mm_yr"] + Z95 * u]
        f = lambda x: _s(x / MM_IN)                             # noqa: E731
        if not rng and a.get("untested_range95_mm_yr"):
            # E9: an untested figure with the uncertainty that can be stated, said to be that
            lo, hi = a["untested_range95_mm_yr"]
            return (f'<span class="num">{f(lo)} to {f(hi)} in/yr</span><br><span class="small muted">not compared '
                    f'with ordinary ground: only {e(a.get("untested_uncertainty_basis") or "the uncertainty that can be stated")}</span>')
        if not rng:
            return '<span class="num">—</span><br><span class="small muted">no comparison possible</span>'
        lo, hi = rng
        return f'<span class="num">{f(lo)} to {f(hi)} in/yr</span><br><span class="small muted num">{lo:+.1f} to {hi:+.1f} mm/yr</span>'

    def total_cell(a):
        tot = a.get("total_mm")
        if tot is None:
            return '<td class="num">—</td>'
        word = area_word(a)
        word += " with the region" if word and moves_with_region(a) else ""
        avg = a.get("total_at_average_rate_mm")
        both = (f'<br><span class="small muted">at the average rate: {_sin(avg)}</span>' if totals_differ(tot, avg) else "")
        # every total with its 95% range (E9)
        trng = a.get("total_range95_mm") or a.get("untested_total_range95_mm")
        tr = (f'<br><span class="small muted">95%: {_s(trng[0] / MM_IN)} to {_s(trng[1] / MM_IN)} in</span>'
              if trng else "")
        return (f'<td class="num">{_sin(tot)}<br><span class="small muted">{tot:+.0f} mm{(" " + word) if word else ""}'
                f'</span>{tr}{both}</td>')

    def size_line(a):
        if a.get("footprint_km2"):
            s_ = f"{a['footprint_km2']:.2f} km² area, {a.get('measured_percent', 0):.0f}% measured ({a.get('cells', 0):,} points)"
            if (a.get("outside_frame_percent") or 0) >= 1:
                s_ += f"; {a['outside_frame_percent']:.0f}% outside the satellite's view"
            return s_ + _wooded_note(a)
        return f"{a.get('area_km2', 0):.2f} km², {a.get('cells', 0):,} measurement points" + _wooded_note(a)

    rows = []
    if full:
        tag = ('<span class="tag gnss">GNSS-anchored</span>' if full.get("vertical_mm_yr") is not None and anchored
               else '<span class="tag">Relative only</span>')
        stand = full.get("cells_standing_out_percent"); chance = full.get("cells_standing_out_by_chance_percent")
        # on the report's own farmland (E14; report_insights.cells_sentence);
        # 'the same as chance' only inside chance's 95% binomial range over
        # the study area's radar cells (F7a)
        pts = (e(ri.cells_sentence(stand, chance, rep.get("independent_radar_measurements")))
               if hasattr(ri, "cells_sentence") else "")
        gl = ""
        if reg.get("anchored") and full.get("relative_vertical_mm_yr") is not None:
            ru = full.get("relative_uncertainty_1sigma_mm_yr")
            gl = (f" The figure is the satellite's measure of this area against its {e(ground)} "
                  f"({full['relative_vertical_mm_yr']:+.1f}" + (f" ± {Z95 * float(ru):.1f}" if ru else "")
                  + " mm/yr" + (", 95%" if ru else "") + ") plus the regional movement from GNSS: "
                  + e(reg["sentence"][0].lower() + reg["sentence"][1:]))
        rows.append(f"""<tr><th scope="row">{e(full['name'])}<br><span class="small muted">whole study area, {full.get('area_km2', 0):.0f} km² measured</span></th>
<td>{rate_cell(full)}</td>{total_cell(full)}
<td>{range_cell(full)}</td><td>{tag}<br><span class="small">{e((full.get('reading') or 'The region as a whole').rstrip('.'))}.{gl}{' Features that stand out are listed under Observations. ' if standing else ' '}{pts}</span></td></tr>""")
    for a in regions:
        if a.get("testable") is False or a.get("vertical_mm_yr") is None:
            tag = '<span class="tag">Not testable</span>'
            why = (a.get("not_testable_reason") or "no comparison with ordinary ground was possible").rstrip(".")
            what = f"{e(why)}. The figure is this area's movement only; it was not checked against ordinary ground."
        else:
            tag = ('<span class="tag hit">Movement detected</span>' if a.get("detected")
                   else '<span class="tag ok">Within normal ground variation</span>')
            what = ('Moving differently from the surrounding ground.' if a.get('detected')
                    else f'Not distinguishable from the surrounding {e(ground)}.')
            if moves_with_region(a):
                # D1: the anchored range excludes 0 only through the regional movement
                what += (f" It is {area_word(a)} with the surrounding region (the regional movement measured by "
                         "GNSS stations), not on its own.")
            what += (" " + detection_sentence(a) if a.get("detection_limit_mm_yr") else "") + thin_null_sentence(a, e(ground))
            rs = recent_sentence(a)
            what += (" " + e(rs)) if rs else ""
        if a.get("resolution_caveat"):
            what += " " + e(a["resolution_caveat"])
        rows.append(f"""<tr><th scope="row">{nm(a)}<br><span class="small muted">{size_line(a)}</span></th>
<td>{rate_cell(a)}</td>{total_cell(a)}
<td>{range_cell(a)}</td><td>{tag}<br><span class="small">{what}</span></td></tr>""")
    table = f"""<div class="tablewrap"><table>
<thead><tr><th>Area</th><th>Vertical movement per year</th><th>Measured over the record</th><th>Likely range (95%)</th><th>What it means</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table></div>"""

    # ---- the short answer: over the areas that could be tested only
    tested = [a for a in regions if a.get("testable") is not False and a.get("vertical_mm_yr") is not None]
    untested = [a for a in regions if a not in tested]
    n_det = sum(1 for a in tested if a.get("detected"))
    bits = []
    if tested:
        if n_det == 0:
            # no totals here (finding 2): they included the regional GNSS
            # movement in a sentence about moving with the farmland, and
            # carried no range; the table gives them with theirs
            who = (f"none of the {len(tested)} report areas that could be tested moves" if len(tested) > 1 else
                   f"{nm(tested[0])}, the report area that could be tested, does not move")
            bits.append(f"Over {ymd(d0)} to {ymd(d1)}, <b>{who} differently from the surrounding {e(ground)}</b>"
                        ", within the range ordinary ground moves.")
        else:
            det = [a for a in tested if a.get("detected")]
            bits.append(f"Over {ymd(d0)} to {ymd(d1)}, <b>{n_det} of {len(tested)} tested report areas move differently from the "
                        f"surrounding {e(ground)}</b>: " + ", ".join(
                            nm(a) + (f" ({area_word(a)}, {_in(a['vertical_mm_yr'], 2)}/yr" if area_word(a)
                                     else f" ({_s(a['vertical_mm_yr'] / MM_IN)} in/yr")
                            + (f", likely range {_s(a['range95_mm_yr'][0] / MM_IN)} to {_s(a['range95_mm_yr'][1] / MM_IN)} in/yr"
                               if a.get("range95_mm_yr") else "") + ")"
                            for a in det) + ".")
    if untested:
        bits.append("<b>" + ", ".join(nm(a) for a in untested) + " could not be tested</b> against ordinary ground (see the table); "
                    + ("their" if len(untested) > 1 else "its") + " figures are not a finding either way.")
    if full and full.get("vertical_mm_yr") is not None and full.get("range95_mm_yr"):
        v = full["vertical_mm_yr"]
        lo, hi = full["range95_mm_yr"]
        word = _dir(v, full["range95_mm_yr"])
        if word:
            bits.append(f"The study area as a whole is {word} slowly, about {_in(v, 2)} per year (likely range "
                        f"{lo / MM_IN:+.2f} to {hi / MM_IN:+.2f} in/yr): the satellite's measure of it against the "
                        f"surrounding {e(ground)}, plus the regional movement measured by GNSS stations.")
        else:
            bits.append(f"The study area as a whole has no movement that can be distinguished from zero (about "
                        f"{v / MM_IN:+.2f} in/yr, likely range {lo / MM_IN:+.2f} to {hi / MM_IN:+.2f} in/yr).")
    if rep.get("features_assessed") is False:
        bits.append(f"Local features <b>could not be assessed</b>: too little ordinary {e(ground)} to measure its normal range.")
    elif standing:
        # 'stands out clearly' only as selection_p decided it (E7, contract
        # C4): the calibrated chance test AND the QA's calibrated stand-out
        # AND copies as strong as the real farmland; any other feature that
        # beats chance is worth a look - counted apart
        conf = [f for f in standing if f.get("stands_out_clearly") is True]
        look = [f for f in standing if f.get("stands_out_clearly") is not True]
        pl = lambda n_, one, many: one if n_ == 1 else many      # noqa: E731
        if conf:
            sent = (f"<b>{len(conf)} local {pl(len(conf), 'feature stands', 'features stand')} out clearly</b> from what "
                    f"ordinary {e(ground)} produces by chance")
            if look:
                sent += (f", and <b>{len(look)} more</b> {pl(len(look), 'goes', 'go')} beyond it but {pl(len(look), 'is', 'are')} "
                         "worth a look rather than a confirmed finding")
        else:
            sent = (f"<b>{len(look)} local {pl(len(look), 'feature goes', 'features go')} beyond</b> what ordinary "
                    f"{e(ground)} produces by chance, worth a look rather than a confirmed finding")
        f0 = standing[0]
        v0 = f0.get('vertical_relative_mm_yr', f0['mean_rel_vertical_mm_yr'])
        rr0 = ri.rate_range_words(v0, f0.get("rate_range95_mm_yr"))
        rr0 = f" ({rr0})" if rr0 else ""
        # the feature's own tense (final check of 7 Oct 2026: Site B's
        # slowed Feature 1 'sinks about 1.01 in/yr' here, 'averaged' below)
        if f0.get("ground_reshaped"):
            what = "changed during the record (dug or filled; no rate can be given)"
        elif ri._past_or_present(f0) == "averaged":
            shape0 = (f0.get("trend") or {}).get("shape")
            what = (f"averaged about {abs(v0) / MM_IN:.2f} in/yr{rr0} {'down' if f0['sign'] < 0 else 'up'} "
                    f"relative to the {e(ground)} over the whole record"
                    + (" (slower in recent years)" if shape0 == "slowing" else
                       " (faster in recent years)" if shape0 == "accelerating" else ""))
        else:
            what = (f"{'sinks' if f0['sign'] < 0 else 'rises'} about {abs(v0) / MM_IN:.2f} in/yr{rr0} relative to "
                    f"the {e(ground)}")
        mines0 = f0.get("mines") or {}
        where = (f" and lies {e(mines0.get('phrase', ''))}" if mines0.get("key") or mines0.get("inside") else "")
        # named by its number, not 'the largest': features are listed by QA
        # stand-out, then rate x area (Site B: Feature 1, 2.2 km², was
        # called the largest beside Feature 2's 6.0 km²)
        n0 = next((i for i, f in enumerate(feats, 1) if f is f0), 1)
        bits.append(sent + f"; {f'Feature {n0}' if len(standing) > 1 else 'the feature'}, about {f0['area_km2']:.1f} km² near "
                    f"{ri.lalo_txt(f0['lat'], f0['lon'])}, {what}{where}. Details are under Observations.")
    elif not untested_feats:
        bits.append(f"No local feature stands out from what ordinary {e(ground)} produces by chance anywhere in the study area.")
    if untested_feats:
        bits.append(f"{len(untested_feats)} patch{'es' if len(untested_feats) > 1 else ''} could not be tested against chance "
                    "and need a specialist's look (Observations).")
    answer = " ".join(bits)

    # ---- the header: points, posting, resolution, precision (methodology-9, empirical-7, clientout-7)
    res = rep.get("resolution") or {}
    n_pts = (full or {}).get("cells") or rep.get("points") or 0
    nind = rep.get("independent_radar_measurements")
    up = float(rep.get("los_up_median") or 0) or None
    lims = sorted(a["detection_limit_mm_yr"] for a in tested if a.get("detection_limit_mm_yr"))
    head = (f"{rep['n_dates']} satellite images over {span:.1f} years. {n_pts:,} measurement points across the study area"
            + (f", {res['posting_m']:.0f} m apart" if res.get("posting_m") else "")
            # E25: radar resolution cells over the study area, not 'independent measurements'
            + (f", from about {int(nind):,} radar resolution cells" if nind else "") + ".")
    # the one per-point uncertainty (E18): the uncertainty raster's own value;
    # a report from before 7 Oct 2026 has only the split-half repeatability
    sp = rep.get("single_point_uncertainty_1sigma_mm_yr")
    sp = float(sp) if sp else (float(rep["cell_repeatability_mm_yr"]) / up
                               if rep.get("cell_repeatability_mm_yr") and up else None)
    if sp:
        head += f" A single point's yearly rate is uncertain by about ±{sp / MM_IN:.2f} in/yr (1 sigma)"
        if lims:
            sinking = all(a.get("detection_limit_basis") == "sinking" for a in tested if a.get("detection_limit_mm_yr"))
            head += (f"; averaged over a report area, sustained {'sinking' if sinking else 'movement'} of "
                     + (f"{lims[0] / MM_IN:.2f}" if len(lims) == 1 or abs(lims[0] - lims[-1]) < 0.5 else
                        f"{lims[0] / MM_IN:.2f} to {lims[-1] / MM_IN:.2f}")
                     + " in/yr or more is detected 4 times in 5")
        head += "."

    # ---- observations
    obs = rep.get("observations") or []
    obs_html = "".join(f"<dt>{e(o['title'])}</dt><dd>{e(o['text'])}</dd>" for o in obs)
    qn_items = list(rep.get("quality_notes") or [])
    geo = rep.get("geolocation") or {}
    if geo.get("text"):
        qn_items.append(str(geo["text"]))
    flags = rep.get("qa_flags") or {}
    if flags.get("dem_error"):
        qn_items.append("Near mine edges the elevation model (made before the newest pits and spoil) is out of date, "
                        "so some points there may be drawn a map cell or two from where they are; a feature at an "
                        "outline's edge may lie on the other side of it.")
    if flags.get("plausibility"):
        qn_items.append("Some ground changes too fast across short distances for this satellite to follow; those "
                        "points are not measurements and are marked in quality_too_fast_to_measure.tif.")
    qn = "".join(f"<li>{e(t)}</li>" for t in qn_items)
    mines_line = mine_outlines_line(rep)

    fd = full.get("deepest_patch") if full else None
    gn = "".join(f"<li><b>{e(s['site'])}</b>, {s['distance_km']:.0f} km away, "
                 f"{'inside the study area, ' if s.get('inside_processed_area') else ''}recording {int(s['data'][0])}–{int(s['data'][1])}"
                 + (f": {s['vertical_over_record_mm_yr']['rate_mm_yr']:+.1f}"
                    + (f" ± {float(s['vertical_over_record_mm_yr']['sigma_mm_yr']):.1f}"
                       if s['vertical_over_record_mm_yr'].get('sigma_mm_yr') is not None else "")
                    + " mm/yr vertical over the record's dates" if s.get('vertical_over_record_mm_yr') else "")
                 + ("" if s.get('used_for_background') else " (not enough overlap with the record to use)") + "</li>"
                 for s in tie.get("stations", []))
    chk = station_check(next(iter(tie.get("inside_checks") or []), None), rep)
    chk_html = chk["html"] if chk else ""

    methods_html = "".join(f"<dt>{e(k)}</dt><dd>{v}</dd>" for k, v in methods_stages(cfg, rep))
    lims_a = [a for a in regions if a.get('detection_limit_mm_yr')]
    detect_txt = ""
    if lims_a:
        big = min(lims_a, key=lambda a: a["detection_limit_mm_yr"])
        small = max(lims_a, key=lambda a: a["detection_limit_mm_yr"])
        # the sinking limits (E5); rising ground named where its limit differs by more than a tenth
        sinking = all(a.get("detection_limit_basis") == "sinking" for a in lims_a)
        rising = [a for a in lims_a if rising_limit_differs(a)]
        detect_txt = (f"<p>In practice: sustained {'sinking' if sinking else 'movement'} of "
                      f"{_in(big['detection_limit_mm_yr'], 2)} per year or more across "
                      f"{nm(big)} would be detected 4 times in 5"
                      + (f"; across {nm(small)}, {_in(small['detection_limit_mm_yr'], 2)} per year" if small is not big else "")
                      + "".join(f"; rising ground across {nm(a)} needs {_in(a['detection_limit_rise_mm_yr'], 2)} per year"
                                for a in rising)
                      + ". These limits are for movement that is steady over the whole period and spread over the whole area: "
                        "a shorter episode, or movement over part of an area, can be larger than this and still be missed.</p>")
    band = rep.get("map_band") or {}
    ms = rep.get("map_scale") or {}
    if band:
        # finding 35: the band is the middle 95% of ordinary farmland, not
        # centred on its typical value
        band_txt = (f"Grey is ground within the middle 95% of ordinary {e(ground)} at this scale "
                    f"({_s(band['low_mm_yr'] / MM_IN)} to {_s(band['high_mm_yr'] / MM_IN)} in/yr)")
        ck = ms.get("chance_km2") or [None, None]
        chance_txt = (f"On ordinary {e(ground)} about {band.get('farmland_beyond_low_percent', 0):.1f}% of the ground shows red and "
                      f"{band.get('farmland_beyond_high_percent', 0):.1f}% blue by chance alone"
                      + (f": across this map's {ms['map_km2']:,.0f} km² that is about {ck[0]:.0f} km² of red and "
                         f"{ck[1]:.0f} km² of blue where nothing unusual is happening." if ms.get("map_km2") and ck[0] is not None else "."))
    else:
        chance_txt = ""
        band_txt = f"There is too little ordinary {e(ground)} to measure its normal range, so the map has no grey band"
    scale_txt = ""
    if ms.get("edges_mm_yr"):
        scale_txt = " " + colour_steps_text(ms)
        if ms.get("peak_mm_yr") is not None and ms.get("peak_lat") is not None:
            pin = ms.get("peak_in") or {}
            scale_txt += (f" The strongest movement on the map is {_s(ms['peak_mm_yr'] / MM_IN)} in/yr near "
                          f"{ri.lalo_txt(ms['peak_lat'], ms['peak_lon'])}"
                          # finding 51i: say which patch it lies in
                          + (f", in Feature {pin['feature']}" if pin.get("feature") else
                             ", in a patch the search found within what ordinary ground produces by chance"
                             if pin.get("within_chance") else "") + ".")
    bm_txt = (f" {e(figs['basemap_credit'])}." if figs.get("basemap_credit") else "")
    # relative to the farmland, as the chance level it is set against (a GNSS-
    # anchored figure printed here had no uncertainty - E9)
    fd_v = (fd or {}).get("relative_vertical_mm_yr", (fd or {}).get("vertical_mm_yr"))
    fd_c = (fd or {}).get("chance_relative_vertical_mm_yr", (fd or {}).get("chance_vertical_mm_yr"))
    rel_w = f" relative to the surrounding {e(ground)}" if (fd or {}).get("relative_vertical_mm_yr") is not None else ""
    deep = (f' The deepest 1 km patch found anywhere ({ri.lalo_txt(fd["lat"], fd["lon"])}, {_in(fd_v, 2)}/yr{rel_w}) is about as deep as chance alone would produce ({_in(fd_c, 2)}/yr).'
            if not feats and fd and fd.get("lat") is not None and fd_v is not None
            and fd_c is not None and not fd.get("stands_out") else "")
    patches = rep.get("map_patches") or []
    patch_html = ""
    if patches:
        names = {a["name"]: a.get("display_name") or a["name"] for a in A}
        # E12: a patch is listed under an area only when at least half of it
        # lies inside, with the share inside; one mostly outside is listed
        # apart (Site B's Feature 1 - 0.16 of 2.21 km² inside - was under
        # the project area)
        def share(p):
            if p.get("inside_share") is not None:
                return float(p["inside_share"])
            return float(p.get("inside_area_km2") or 0.0) / float(p["area_km2"]) if p.get("area_km2") else 1.0

        def rows_html(ps, outside):
            out_ = ""
            for p in ps:
                rate, what = patch_rate_cells(p, feats, rep.get("map_smoothing_m"))
                inside = p.get("inside_area_km2")
                size = (f"{p['area_km2']:.2f} km²" + (f"<br><span class=\"muted\">{float(inside):.2f} km² inside</span>"
                                                      if inside is not None and share(p) < 0.995 else ""))
                where_a = (f"beside {e(names.get(p['area'], p['area']))}" if outside else e(names.get(p['area'], p['area'])))
                out_ += (f"<tr><td>{where_a}</td><td class=\"num\">{ri.lalo_txt(p['lat'], p['lon'])}</td>"
                         f"<td class=\"num\">{size}</td><td class=\"num\">{e(rate)}</td><td>{e(what)}</td></tr>")
            return out_
        head_ = ("<div class=\"tablewrap\"><table class=\"small\"><thead><tr><th>Area</th><th>Where</th><th>Size</th>"
                 "<th>Average, per year</th><th>What it is</th></tr></thead><tbody>")
        ins = [p for p in patches if share(p) >= PATCH_INSIDE_SHARE]
        outs = [p for p in patches if share(p) < PATCH_INSIDE_SHARE]
        if ins:
            patch_html += ("<h3>Coloured patches inside the report areas</h3>" + head_ + rows_html(ins, False)
                           + "</tbody></table></div>")
        if outs:
            patch_html += ("<h3>Coloured patches mostly outside the report areas, touching them</h3><p class=\"small "
                           "muted\">Less than half of each lies inside the area; the size inside is given.</p>"
                           + head_ + rows_html(outs, True) + "</tbody></table></div>")

    # ---- independent tests (methodology-11): technical section, never a client number.
    # The injection test is not on the client page at all (E4, contract C5):
    # it is in the technical record (injection_recovery.json)
    ic = rep.get("independent_check") or {}
    tests_html = "<h3>Independent tests</h3><ul class=\"small\">"
    tests_html += ("<li>" + e(ic["sentence"]) + "</li>" if ic.get("available") and ic.get("sentence") else
                   "<li>No second, independently processed satellite product was compared with this record.</li>")
    cv = rep.get("combined_view") or {}
    if cv.get("sentence"):
        tests_html += "<li>" + e(cv["sentence"]) + "</li>"
    tests_html += "</ul>"
    # what this build left out (16 Oct 2026, readiness review ERR-01): in
    # this technical part of the page, in plain words; the errors are in
    # client_report.json's 'omitted' list and on the release gate's list
    om = [str(o.get("what") or o.get("block") or "a part of the report")
          for o in rep.get("omitted") or [] if isinstance(o, dict)]
    omitted_html = ("<h3>Left out of this build</h3><p class=\"small\">These parts could not be worked out when this "
                    "report was built, so they are missing from it:</p><ul class=\"small\">"
                    + "".join(f"<li>{e(_upper_first(w))}.</li>" for w in om) + "</ul>") if om else ""
    # E3: only the table built on the report's own frame (vertical, its
    # farmland: compute's _years_since_mining) - a record with panel
    # history's line-of-sight, first-pass table prints none
    yrs = (rep.get("by_years_since_mining") or []) if rep.get("by_years_since_mining_frame") == "vertical" else []
    yrs_html = ""
    if yrs:
        yrs_html = ("<section class=\"prose\"><h2>Movement by years since mining</h2><p class=\"small muted\">How fast "
                    "dated mined areas changed, by years since each was mined or reclaimed (negative years: before "
                    "it)." + (f" {e(str(rep.get('by_years_since_mining_note')))}" if rep.get('by_years_since_mining_note') else "")
                    + "</p><div class=\"tablewrap\"><table class=\"small\"><thead><tr><th>Years since mining</th><th>Areas</th>"
                    "<th>Median vertical change per year</th><th>Middle half</th></tr></thead><tbody>"
                    # inches with mm beneath, mm to 0.1 (final check of 7 Oct
                    # 2026: '+0.02 in' sat beside '(0 mm)', '(+0 mm)' and
                    # '(+1 mm)', and the middle half was in mm only)
                    + "".join(f"<tr><td class=\"num\">{r['years_since_mining']}</td><td class=\"num\">{r['n_areas']}</td>"
                              f"<td class=\"num\">{_sin(r['median_change_mm_per_yr'])}<br><span class=\"small muted\">"
                              f"{_s(r['median_change_mm_per_yr'], 1)} mm</span></td>"
                              f"<td class=\"num\">{_s(r['q25_mm'] / MM_IN)} to {_s(r['q75_mm'] / MM_IN)} in<br>"
                              f"<span class=\"small muted\">{_s(r['q25_mm'], 1)} to {_s(r['q75_mm'], 1)} mm</span></td></tr>"
                              for r in yrs)
                    + "</tbody></table></div></section>")

    release_html = ""
    if not final:
        hard = [b for b in (rel.get("blockers") or []) if b.get("hard")]
        release_html = ('<div class="answer prose hold"><b>DRAFT - not for release.</b> '
                        + ("To settle before this report is sent:<ul class=\"small\">" + "".join(f"<li>{e(b.get('page_text') or b['text'])}</li>" for b in hard) + "</ul>"
                           if hard else "It has not been signed off for these numbers.") + "</div>")
    # M3: a reissue names the report it replaces and what changed (the
    # config's report_revision; the record's when the config is not to hand)
    rev = report_revision(cfg) or rep.get("report_revision")
    rev_html = revision_html({"report_revision": rev})
    place = rep.get("location") or ""
    credits_html = _optional(lambda: _module("credits").text_block(cfg, rep, html=True), "credits_section") or ""
    if figs.get("basemap_credit"):
        credits_html += f'<ul class="credits"><li>{e(figs["basemap_credit"])}: basemap.nationalmap.gov</li></ul>'
    nisar = rep.get("backend") == "nisar"
    kicker = ("<span class=\"flag\">Screening - not the full processing</span> · " if screening else "") + \
             ("<span class=\"flag\">Comparison - NISAR, not validated</span> · " if nisar else "") + \
             f"Ground movement survey · {sensor_name(rep.get('backend'))} satellite radar · {ymd(d0)} – {ymd(d1)}"
    h1 = f"{e(title)} ground movement report" + (" (screening)" if screening and "screening" not in title.lower() else "") \
        + (" (NISAR comparison)" if nisar and "comparison" not in title.lower() else "")
    gloss = "".join(f"<dt>{e(k)}</dt><dd>{e(v)}</dd>" for k, v in glossary(rep))
    lim_html = "".join(f"<li>{x}</li>" for x in limitations(cfg, rep))
    gnss_p = (("Permanent GNSS (high-precision GPS) stations near the site measure the regional ground motion "
               "independently of the satellite. Over the record's dates " + e(reg["sentence"][0].lower() + reg["sentence"][1:])
               # final check of 7 Oct 2026: 'every figure above' took in the
               # features, mine ground and map, which are relative
               + f" This regional rate is added to every GNSS-anchored figure (the report areas in the table above), "
               f"on the assumption that the surrounding {e(ground)} moves with the stations' monuments; the features, "
               f"the mine ground and the map are relative to the surrounding {e(ground)}.") if reg.get("anchored") else
              "No GNSS station could anchor these figures" + (f" ({e(str(rep.get('gnss_reason')).rstrip('.'))})" if rep.get('gnss_reason') else '') + ".")
    series_caption = (
        "Each chart: the thin line is the area's average on every image; the thick line is its average rate, or for "
        "a feature whose pace changed the curve its measured change is read from, and starts at zero. "
        f"The darker band is how far the average rate of ordinary {e(ground)} of the same size reaches (95%): an area "
        "stands out only if its straight line ends outside it, which is the table's reading. The lighter band adds the "
        "swings single images show (season, weather): ordinary ground's thin line stays inside it on about 19 images "
        "in 20, so a few dates outside it are expected and are not a detection. Feature charts are relative to the "
        f"surrounding {e(ground)} and show only the lighter band; whether a feature stands out was decided by the "
        "search test described under Observations.")
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{e(title)} Ground Movement{' (draft)' if not final else ''}</title>
<style>{CSS}{DRAFT_CSS if not final else ''}</style>
</head>
<body>
<main>
<header>
  <div class="kicker">{kicker}</div>
  <h1>{h1}</h1>
  <p class="prose muted">{e(place) + '. ' if place else ''}{e(head)}</p>
</header>

<section>
  {release_html}
  {rev_html}
  <div class="answer prose">{answer}</div>
  {table}
  <p class="small muted prose">Movement is vertical (negative = sinking), in inches with millimetres beneath. {'Figures are tied to the regional GNSS network, so they are absolute movement, not just movement relative to a reference point.' if anchored else ''} The likely range is where the true value lies with 95% confidence; for a report area it comes from the same comparison with ordinary ground as the reading{', plus the regional movement measured by GNSS stations' if anchored else ', so it excludes no movement exactly when movement is detected'}. "Sinking" or "rising" is written only where that range excludes no movement{'; where only the regional movement does that, the area is said to move with the surrounding region' if anchored else ''}.</p>
</section>

<section class="prose">
  <h2>Observations</h2>
  <p class="small muted">Calculated by the software from the record itself. Causes are suggestions from the mapped context, not findings. {mines_line}</p>
  <dl>{obs_html}</dl>
  <h3>Reading the numbers</h3>
  <ul class="small">{qn}</ul>
</section>

<section>
  <h2>The map</h2>
  <figure><img src="data:image/png;base64,{_b64(figs['map'])}" alt="Map of vertical ground movement per year across the study area">
  <figcaption>Vertical movement per year relative to the surrounding {e(ground)}, averaged over about {_about_m(rep.get('map_smoothing_m', 500))} m{' (the regional GNSS movement is in the table, not in the colours)' if anchored else ''}. {band_txt}; colour appears only beyond it, red for sinking and blue for rising.{scale_txt} {chance_txt}{' Numbered boxes are the features described under Observations.' if feats else deep} Ground with neither colour nor grey could not be measured; inside a report area it is hatched.{bm_txt} {mines_line}</figcaption></figure>
  {patch_html}
</section>

<section>
  <h2>Movement over time</h2>
  <figure><img src="data:image/png;base64,{_b64(figs['series'])}" alt="Charts of vertical movement over time">
  <figcaption>{series_caption}</figcaption></figure>
</section>
{yrs_html}
<section class="prose">
  <h2>How confident are these numbers?</h2>
  <p>Over {e(ground)}, radar measurements of small areas differ from the true movement by a few millimetres per year, because crops and soil moisture change the signal from season to season. We measured that variation directly on ordinary {e(ground)} around the site, moving each report area's outline across it.</p>
  {detect_txt}
  <h3>Checked against GNSS</h3>
  <p>{gnss_p}</p>
  <ul class="small">{gn}</ul>
  {chk_html}
  {tests_html}
  {omitted_html}
</section>

<section class="prose">
  <h2>Methodology</h2>
  <dl class="small">{methods_html}</dl>
  <h2>Limitations</h2>
  <ul class="small">{lim_html}</ul>
  <h2>Terms used</h2>
  <dl class="small">{gloss}</dl>
  <h2>Data sources and credits</h2>
  <div class="small">{credits_html}</div>
</section>
{footer_html(cfg, rep, final)}
</main>
</body>
</html>"""


def one_pager_lines(rep: dict, a: dict) -> list[tuple[str, str]]:
    """The numbers block of an area's one-page summary, (label, value), every
    value formatted from client_report.json's own fields - the PDF prints
    these and nothing else, so a presenter's page cannot disagree with the
    report (clientout-16). Rates to 0.01 in/yr and every total with its 95%
    range (E9); an untested area's range is the uncertainty that can be
    stated, said so; the recent range with its units (finding 51g)."""
    v, rng = a.get("vertical_mm_yr"), a.get("range95_mm_yr")
    word = area_word(a)                     # the page's word (D1, 7 Oct 2026)
    word += " with the surrounding region" if word and moves_with_region(a) else ""
    d0, d1 = rep["dates"]
    urng = None if rng else a.get("untested_range95_mm_yr")
    out = [("Vertical movement per year",
            f"{_s(v / MM_IN)} in/yr ({v:+.1f} mm/yr){(' ' + word) if word else ''}" if v is not None else
            ("relative only" if a.get("relative_only") else "—")),
           ("Likely range (95%)", f"{_s(rng[0] / MM_IN)} to {_s(rng[1] / MM_IN)} in/yr ({rng[0]:+.1f} to {rng[1]:+.1f} mm/yr)"
            if rng else
            f"{_s(urng[0] / MM_IN)} to {_s(urng[1] / MM_IN)} in/yr - not compared with ordinary ground; only "
            f"{a.get('untested_uncertainty_basis') or 'the uncertainty that can be stated'}" if urng else
            "— (no comparison possible)")]
    tot, avg = a.get("total_mm"), a.get("total_at_average_rate_mm")
    trng = a.get("total_range95_mm") or a.get("untested_total_range95_mm")
    out.append((f"Measured over the record ({d0[:4]}–{d1[:4]})",
                f"{_s(tot / MM_IN)} in ({tot:+.0f} mm)"
                + (f", 95% range {_s(trng[0] / MM_IN)} to {_s(trng[1] / MM_IN)} in" if trng else "")
                + (f"; at the average rate {_s(avg / MM_IN)} in" if totals_differ(tot, avg) else "")
                if tot is not None else "—"))
    if a.get("recent_rate_mm_yr") is not None and len(a.get("recent_window") or []) == 2:
        rw, rr = a["recent_window"], a.get("recent_range95_mm_yr")
        out.append((f"Recent rate ({rw[0][:4]}–{rw[1][:4]})", f"{_s(a['recent_rate_mm_yr'] / MM_IN)} in/yr ({a['recent_rate_mm_yr']:+.1f} mm/yr)"
                    + (f", range {_s(rr[0] / MM_IN)} to {_s(rr[1] / MM_IN)} in/yr" if rr else "")))
    out.append(("Reading", str(a.get("reading") or "—")))
    if a.get("kind") == "area":
        lim = a.get("detection_limit_mm_yr")
        sinking = a.get("detection_limit_basis") == "sinking"
        out.append((f"Reliably detects{' sinking' if sinking else ''} (4 times in 5)",
                    f"{lim / MM_IN:.2f} in/yr ({lim:.0f} mm/yr)" if lim else "—"))
        if rising_limit_differs(a):
            rise = float(a["detection_limit_rise_mm_yr"])
            out.append(("Reliably detects rising (4 times in 5)", f"{rise / MM_IN:.2f} in/yr ({rise:.0f} mm/yr)"))
        if a.get("signal_to_noise") is not None:
            out.append(("Signal ÷ threshold", f"{a['signal_to_noise']:.2f} (1 or more stands out)"))
        if a.get("footprint_km2"):
            out.append(("Measured", f"{a.get('area_km2', 0):.2f} of {a['footprint_km2']:.2f} km² "
                                    f"({a.get('measured_percent', 0):.0f}%, {a.get('cells', 0):,} points)"))
    else:
        out.append(("Measured", f"{a.get('area_km2', 0):,.0f} km² ({a.get('cells', 0):,} points)"))
        g = a.get("regional_gnss_vertical_mm_yr")
        if g is not None:
            gu = a.get("regional_gnss_uncertainty_mm_yr")
            out.append(("Regional GNSS rate", f"{_s(g / MM_IN)} in/yr ({g:+.1f}" + (f" ± {gu:.1f}" if gu else "") + " mm/yr, 1 sigma)"))
    return out


def pdf_caveat(rep: dict) -> str:
    """The caveats every area page of the PDF carries (E26, trust review 7
    Oct 2026: the PDF and the KMZ travel without the page's Limitations, and
    neither said that 'vertical' is an assumption from one track or how much
    ground each value covers): vertical assumed, the ground each value
    combines (resolution.support_m), relative or GNSS-anchored."""
    from . import report_insights as ri
    ground = ri.ground_word(rep)
    sup = (rep.get("resolution") or {}).get("support_m") or []

    def m_(x):
        x = float(x)
        return int(round(x)) if x < 50 else int(5 * round(x / 5)) if x < 100 else int(10 * round(x / 10))
    support = (f" within about {m_(sup[0])} m east-west and {m_(sup[1])} m north-south" if len(sup) >= 2 and all(sup[:2])
               else "")
    frame = (f"Figures are relative to the surrounding {ground} plus the regional movement measured by GNSS stations, "
             "so they are absolute; the map is relative." if rep.get("anchored") else
             f"Figures are relative to the surrounding {ground} (no GNSS anchor), not absolute movement.")
    return PDF_CAVEAT.format(support=support, frame=frame)


def _wrap(text: str, width: int) -> str:
    import textwrap
    return "\n".join(textwrap.wrap(str(text), width)) if text else ""


def pdf_boxes(rep: dict, box) -> list[tuple[int, dict]]:
    """[(number, feature)] of the features whose dashed box an area's PDF map
    draws: those whose outline's box meets the map's extent `box` (west,
    east, south, north); [] without a map."""
    if box is None:
        return []
    w, e_, s_, n_ = box
    out = []
    for i, f in enumerate(rep.get("features") or [], 1):
        try:
            fw, fs, fe, fn = (float(x) for x in f["bbox"])
        except (KeyError, TypeError, ValueError):
            continue
        if fe >= w and fw <= e_ and fn >= s_ and fs <= n_:
            out.append((i, f))
    return out


def pdf_box_lines(rep: dict, drawn: list) -> str:
    """What the numbered boxes on an area's PDF map are (final check of 7
    Oct 2026: they were drawn and never explained, and the PDF travels
    without the page): one clause per box with the feature's rate and its
    95% range (M4), its reading, and the outline caveat. '' with no box."""
    if not drawn:
        return ""
    from . import report_insights as ri
    ground = ri.ground_word(rep)
    bits = []
    for i, f in drawn:
        head = f"{i} = Feature {i}: {ri.feature_kind(f)}, {float(f.get('area_km2') or 0):.1f} km²"
        if f.get("ground_reshaped"):
            bits.append(head + ", changed during the record (no rate can be given)")
            continue
        v = f.get("vertical_relative_mm_yr", f.get("mean_rel_vertical_mm_yr"))
        rr = ri.rate_range_words(v, f.get("rate_range95_mm_yr")) if v is not None else ""
        rate = (f", about {abs(v) / MM_IN:.2f} in/yr {'down' if v < 0 else 'up'}{' (' + rr + ')' if rr else ''} "
                f"against the {ground}" if v is not None else "")
        verdict = ("stands out clearly" if f.get("stands_out_clearly") else
                   "worth a look" if f.get("stands_out") else "not tested against chance")
        bits.append(head + rate + f"; {verdict}")
    return ("Numbered boxes on the map: " + ". ".join(bits) + ". Each is described under Observations in the report; "
            "its outline was drawn from these same measurements, so its figures read somewhat stronger than the "
            "ground there truly moved.")


def area_pdf(cfg, rep: dict, ctx, outlines: dict, path: Path, figs: dict | None = None) -> Path:
    """
    The presenter output (clientout-16): one Letter page per report area -
    the whole study area first - with the area's numbers (one_pager_lines),
    a map of the ground around it on the report map's own colour scale, its
    movement over time with the same bands as the report, and the footer the
    page carries (credits, version, date, disclaimer). DRAFT watermark and
    PDF metadata until the report is final.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from . import report_insights as ri
    final = bool((rep.get("release") or {}).get("final"))
    ground = ri.ground_word(rep)
    ms = rep.get("map_scale") or {}
    rf = getattr(ctx, "report_field", None) or {}
    fld = rf.get("field")
    lo_b, hi_b = rf.get("band_lo_mm_yr"), rf.get("band_hi_mm_yr")
    t = ctx.t
    dt_ = [datetime.strptime(d, "%Y%m%d") for d in rep["dates_list"]]
    items = {it["name"]: it for it in series_items(rep)}
    a_ = ctx.attrs
    x0, y0, dx, dy = (float(a_[k]) for k in ("X_FIRST", "Y_FIRST", "X_STEP", "Y_STEP"))
    ver = rep.get("version") or {}
    meta = {"Title": f"{rep.get('title') or ''} ground movement: area summaries",
            "Subject": "DRAFT - not for release" if not final else "Client report area summaries",
            "Creator": f"insar_pipeline {ver.get('version', '')}".strip(), "Keywords": "DRAFT" if not final else ""}
    path.parent.mkdir(parents=True, exist_ok=True)
    # M3: a reissue says so on every page, and its first page lists what changed
    rev = report_revision(cfg) or rep.get("report_revision") or {}
    rev_head = rev.get("heading") or ""
    with PdfPages(path, metadata=meta) as pdf:
        if rev_head:
            fig = plt.figure(figsize=(8.5, 11))
            fig.text(0.06, 0.965, f"{rep.get('title') or ''} ground movement", fontsize=12, color="#58636e", va="top")
            hw = _wrap(rev_head + ".", 48)
            fig.text(0.06, 0.94, hw, fontsize=18, weight="bold", va="top")
            yy = 0.87 - 0.03 * hw.count("\n")
            if rev.get("changes"):
                fig.text(0.06, yy, "What changed:", fontsize=11, weight="bold", va="top")
                yy -= 0.03
                # each item placed under the last one as drawn; a long list is
                # set smaller rather than run into the footer
                rend = fig.canvas.get_renderer()
                sizes = ((10, 95), (9, 105), (8, 118), (7, 135))
                for k_, (fs, width) in enumerate(sizes):
                    drawn_, y_ = [], yy
                    for c in rev["changes"]:
                        drawn_.append(fig.text(0.06, y_, "•", fontsize=fs, va="top"))
                        t_ = fig.text(0.085, y_, _wrap(c, width), fontsize=fs, va="top", color="#1c2530")
                        drawn_.append(t_)
                        y_ = t_.get_window_extent(renderer=rend).y0 / fig.bbox.height - 0.012
                    if y_ >= 0.16 or k_ == len(sizes) - 1:
                        break
                    for d_ in drawn_:
                        d_.remove()
            fig.text(0.06, 0.085, _wrap(DISCLAIMER, 150), fontsize=7, color="#58636e", va="top")
            _stamp(fig, cfg, rep, final)
            _ri().tidy_figure(fig, rep.get("control_ground_label"))
            pdf.savefig(fig)
            plt.close(fig)
        for a in rep["areas"]:
            fig = plt.figure(figsize=(8.5, 11))
            name = a.get("display_name") or a["name"]
            fig.text(0.06, 0.965, f"{rep.get('title') or ''} ground movement", fontsize=12, color="#58636e", va="top")
            if rev_head:
                fig.text(0.94, 0.965, rev_head, fontsize=8, color="#58636e", va="top", ha="right")
            fig.text(0.06, 0.94, name + (" - whole study area" if a.get("kind") == "full" else ""), fontsize=20,
                     weight="bold", va="top")
            d0, d1 = rep["dates"]
            # two lines: on one, the relative-only wording ran off the page's right edge
            fig.text(0.06, 0.905, f"{sensor_name(rep.get('backend'))} satellite radar, {rep['n_dates']} images, {d0[:4]}-{d0[4:6]}-{d0[6:]} to "
                                  f"{d1[:4]}-{d1[4:6]}-{d1[6:]}.\nVertical movement, negative = sinking"
                                  + (", anchored to GNSS." if rep.get("anchored") else f", relative to the {ground}."),
                     fontsize=9, color="#58636e", va="top")
            y = 0.855
            for lab, val in one_pager_lines(rep, a):
                fig.text(0.06, y, lab, fontsize=9, color="#58636e", va="top")
                txt = _wrap(val, 38)
                fig.text(0.06, y - 0.015, txt, fontsize=10.5, weight="bold", va="top")
                y -= 0.036 + 0.019 * txt.count("\n")
            # the map's extent first: the text block names the feature boxes it draws
            rings = outlines.get(a["name"]) or []
            box = None
            if fld is not None and rings:
                allr = np.vstack(rings)
                w, e_ = float(allr[:, 0].min()), float(allr[:, 0].max())
                s_, n_ = float(allr[:, 1].min()), float(allr[:, 1].max())
                pad = max(0.3 * max(e_ - w, n_ - s_), 0.02)
                box = (w - pad, e_ + pad, s_ - pad, n_ + pad)
            drawn = pdf_boxes(rep, box)
            extra = []
            if a.get("kind") == "area":
                if a.get("testable") is False:
                    extra.append("The figure is this area's movement only; it was not checked against ordinary ground.")
                else:
                    extra.append(detection_sentence(a) + thin_null_sentence(a, ground))
                extra.append(a.get("resolution_caveat") or "")
            # E26: the page's caveats travel with the PDF
            extra.append(pdf_caveat(rep))
            # finding 37: the text block is measured and the chart starts
            # below it (Site A's Area A2 caveat ran into the chart's
            # title); a long block is set smaller rather than overprinted.
            # The map's colour bar sits under the map, so the chart starts
            # below it too
            top = 0.42 if box is not None and ms.get("edges_mm_yr") else 0.44
            body = " ".join(x for x in extra if x).strip()
            chart_top, txt_ = top, None
            for fs, width in ((8.5, 60), (7.5, 68), (6.8, 76)):
                if txt_ is not None:
                    txt_.remove()
                txt_ = fig.text(0.06, y - 0.005, _wrap(body, width), fontsize=fs, va="top", color="#1c2530")
                bottom = txt_.get_window_extent(renderer=fig.canvas.get_renderer()).y0 / fig.bbox.height
                chart_top = min(top, bottom - 0.045)
                if chart_top >= 0.34:
                    break
            # the map around the area
            axm = fig.add_axes([0.55, 0.49, 0.40, 0.35])
            if box is not None:
                w, e_, s_, n_ = box
                r0, r1 = max(0, int((n_ - y0) / dy)), min(fld.shape[0], int((s_ - y0) / dy) + 1)
                c0, c1 = max(0, int((w - x0) / dx)), min(fld.shape[1], int((e_ - x0) / dx) + 1)
                sub = fld[r0:r1, c0:c1]
                ext = (x0 + c0 * dx, x0 + c1 * dx, y0 + r1 * dy, y0 + r0 * dy)
                if ms.get("edges_mm_yr") and lo_b is not None and sub.size:
                    cmap, norm = _map_cmap(ms["edges_mm_yr"], int(ms.get("n_below") or 0))
                    q = (sub > lo_b) & (sub < hi_b)
                    im = axm.imshow(np.where(q, np.nan, sub), extent=ext, cmap=cmap, norm=norm, interpolation="nearest")
                    axm.imshow(np.where(q & np.isfinite(sub), 1.0, np.nan), extent=ext, cmap="Greys", vmin=0, vmax=4,
                               interpolation="nearest")
                    # the colours' values (final check of 7 Oct 2026: shades
                    # with no numbers), the report map's own steps in in/yr
                    edges = [float(v_) for v_ in ms["edges_mm_yr"]]
                    cb = fig.colorbar(im, cax=fig.add_axes([0.55, 0.452, 0.40, 0.011]), orientation="horizontal",
                                      extend="both", ticks=edges, spacing="uniform")
                    cb.ax.set_xticklabels([_s(v_ / MM_IN) for v_ in edges], fontsize=6.5)
                    cb.ax.set_xlabel("in/yr, relative to the surrounding " + ground, fontsize=7, labelpad=1)
                for r_ in rings:
                    axm.plot(r_[:, 0], r_[:, 1], color="#1c2530", lw=1.4)
                for i, f in drawn:
                    fw, fs, fe, fn = f["bbox"]
                    axm.add_patch(plt.Rectangle((fw, fs), fe - fw, fn - fs, fill=False, ec="#1c2530", lw=1.0, ls="--"))
                    axm.annotate(str(i), (fw, fn), fontsize=8, weight="bold")
                axm.set_xlim(w, e_); axm.set_ylim(s_, n_)
                axm.set_aspect(1 / np.cos(np.radians(0.5 * (s_ + n_))))
                from matplotlib.ticker import FuncFormatter, MaxNLocator
                axm.xaxis.set_major_locator(MaxNLocator(3)); axm.yaxis.set_major_locator(MaxNLocator(4))
                axm.xaxis.set_major_formatter(FuncFormatter(_lon_fmt)); axm.yaxis.set_major_formatter(FuncFormatter(_lat_fmt))
                axm.tick_params(labelsize=7)
                _scale_bar_and_north(axm, w, s_, e_, n_)
                axm.set_title(f"Per year, relative to the surrounding {ground}:\ngrey = normal, red = sinking, "
                              "blue = rising, blank = not measured", fontsize=8)
            else:
                axm.set_axis_off()
            # what the numbered boxes are (final check of 7 Oct 2026: never
            # explained), under the map's colour bar; the chart starts below it
            boxes = pdf_box_lines(rep, drawn)
            if boxes:
                bt = fig.text(0.55, 0.418 if ms.get("edges_mm_yr") else 0.475, _wrap(boxes, 74), fontsize=6.5,
                              va="top", color="#1c2530")
                b0 = bt.get_window_extent(renderer=fig.canvas.get_renderer()).y0 / fig.bbox.height
                chart_top = min(chart_top, b0 - 0.045)
            # the movement over time, below the text block
            axs = fig.add_axes([0.08, 0.14, 0.86, max(0.16, chart_top - 0.14)])
            it = items.get(name)
            if it is not None:
                bg = (((rep.get("gnss") or {}).get("background") or {}).get("vertical_mm_yr") or 0.0) \
                    if rep.get("anchored") and not it["relative"] else 0.0
                _series_axes(axs, dt_, t, it, bg, compact=True)
            elif a.get("series_vertical_mm"):
                # the whole study area: its series even when relative only (it
                # used to say 'too little of this area could be measured' of a
                # study area 97-99% measured); no rate line without an anchor
                y_ = np.array([np.nan if v is None else v for v in a["series_vertical_mm"]], float) / MM_IN
                axs.plot(dt_, y_, color="#5b6672", lw=0.9,
                         label="every image" + (f" (relative to the {ground})" if a.get("relative_only") else ""))
                if a.get("vertical_mm_yr") is not None:
                    axs.plot(dt_, float(a["vertical_mm_yr"]) * t / MM_IN, color="#1c2530", lw=2.2, label="average rate")
                axs.set_ylabel("inches"); axs.grid(alpha=0.25); axs.legend(fontsize=8, loc="lower left")
                axs.set_title(f"{name}: movement over the record", fontsize=10, loc="left")
            else:
                axs.set_axis_off()
                axs.text(0.5, 0.5, "No series: too little of this area could be measured.", ha="center",
                         transform=axs.transAxes)
            fig.text(0.06, 0.085, _wrap(DISCLAIMER, 150), fontsize=7, color="#58636e", va="top")
            _stamp(fig, cfg, rep, final)
            _ri().tidy_figure(fig, rep.get("control_ground_label"))
            pdf.savefig(fig)
            plt.close(fig)
    log.info("Wrote %s (%d pages)", path, len(rep["areas"]) + (1 if rev_head else 0))
    return path


def cell_sigma_map(cell_null: dict | None, kcls: np.ndarray, model: dict | None = None) -> np.ndarray:
    """Per-cell farmland null (mm/yr LOS) by coherence class: rep['cell_null_by_class']
    (max of the farmland cell spread and the cell's own repeatability - the
    noise significant_95 and the SNR raster divide by). A class missing from
    it takes the nearest listed class (significance._nearest_class's rule).
    Only when the report carries no table does it fall back to the model's
    repeatability, and says so in the log."""
    out = np.full(kcls.shape, np.nan)
    table = {int(k): float(v) for k, v in (cell_null or {}).items() if v is not None}
    if not table:
        log.warning("No per-cell farmland null in the report; the uncertainty raster uses the repeatability alone")
        table = {int(c["class_index"]): float(c["empirical_mm_yr"]) for c in (model or {}).get("by_class", [])}
    for i in range(int(kcls.max()) + 1 if kcls.size else 0):
        if table:
            out[kcls == i] = table[i] if i in table else table[min(table, key=lambda j: (abs(j - i), j))]
    return out


def write_rasters(cfg, rep, ctx, out_dir: Path | None = None) -> list[Path]:
    """GeoTIFFs in the report's own terms, per 15 m cell - the numbers a
    client gets by opening the rasters match the report and the KMZ:

      vertical_movement_<unit>_per_yr.tif        rate relative to farmland of the
                                                 same coherence, / los_up, + GNSS
                                                 (relative only when not anchored)
      vertical_movement_uncertainty_1sigma_...   that cell's farmland null (by
                                                 coherence class: the larger of the
                                                 farmland cell spread and the cell's
                                                 repeatability - the yardstick the
                                                 per-cell test uses) with the GNSS
                                                 uncertainty, vertical
      vertical_change_at_average_rate_<unit>_<d0>_<d1>
                                                 the average rate x years (per cell;
                                                 the report's area totals are the
                                                 measured change, record_change).
                                                 Named vertical_movement_total_* until
                                                 6 Oct 2026, which a GIS user read as
                                                 the measured total; a file under the
                                                 old name, or of other record dates,
                                                 is moved to superseded/

    Until 5 Oct 2026 the uncertainty raster used the repeatability alone
    (Site B class 0.4-0.6: 5.59 against the 6.69 mm/yr the test uses), so
    a client testing cells with it flagged more ground than the pipeline did.
    The raw products beside them (line of sight, against the reference pixel,
    farmland drift included: ordinary farmland reads about -0.6 in/yr there)
    are for specialists."""
    from . import deliverables as dl
    # out_dir: a check's scratch folder (the golden test); the deliverables otherwise
    D = Path(out_dir) if out_dir else Path(cfg.project_dir) / "deliverables"
    D.mkdir(parents=True, exist_ok=True)
    anchored = bool(rep.get("anchored"))
    bg = ((rep.get("gnss") or {}).get("background") or {}) if anchored else {}
    bg_v, bg_u = bg.get("vertical_mm_yr") or 0.0, bg.get("vertical_uncertainty_mm_yr") or 0.0
    up = _los_up(cfg, ctx.vel.shape)
    try:
        model = json.loads((Path(cfg.project_dir) / "deliverables" / "uncertainty_model.json").read_text())
    except (OSError, ValueError):
        model = {}
    null = cell_sigma_map(rep.get("cell_null_by_class"), ctx.kcls, model)
    rate = ctx.r_cm / up + bg_v
    sd = np.hypot(null / up, bg_u)
    unit, div = ("in", MM_IN) if getattr(cfg, "display_units", "in") == "in" else ("mm", 1.0)
    d0, d1 = rep["dates"]
    rel_to = (f"the surrounding {_ri().ground_word(rep)} of the same coherence class"
              + (", + the regional GNSS motion" if anchored else ""))
    # the assumption travels with the file (methodology-15: the client rasters
    # dropped the _ASSUMES_VERTICAL of the raw ones from their names)
    k = rep.get("los_east_over_up")
    assume = (" ASSUMES ALL MOVEMENT IS VERTICAL (line of sight / its up component)"
              + (f"; east-west movement of 1 reads as {float(k):.2f} vertical" if k else ""))
    specs = ((f"vertical_movement_{unit}_per_yr.tif", rate / div, "vertical velocity", f"{unit}/yr",
              "Vertical movement per year (LOS / cos incidence), relative to " + rel_to + "." + assume),
             (f"vertical_movement_uncertainty_1sigma_{unit}_per_yr.tif", sd / div, "vertical velocity uncertainty (1 sigma)",
              f"{unit}/yr", "Per-cell 1-sigma: the farmland cell null of the cell's coherence class"
              + (" with the GNSS uncertainty" if anchored else "")),
             (f"{AVERAGE_RATE_RASTER}_{unit}_{d0}_{d1}.tif", rate * ctx.span / div,
              "vertical change at the average rate (average rate x years)", unit,
              f"Average rate x {ctx.span:.2f} years per cell - not the measured change the report gives for areas."
              + assume))
    # line of sight in the report's frame (8 Oct 2026: a coworker asked for
    # LOS velocity and LOS cumulative displacement; the raw los_* rasters are
    # against the reference pixel with the farmland drift in them, so
    # farmland reads about -0.3 in/yr there). Relative to farmland of the
    # same coherence class, no GNSS, no vertical assumption.
    from . import field_points as fp
    los_rel_to = f"the surrounding {_ri().ground_word(rep)} of the same coherence class"
    los_specs = ((f"line_of_sight_velocity_{unit}_per_yr.tif", ctx.r_cm / div, "line-of-sight velocity", f"{unit}/yr",
                  "Line-of-sight movement per year, relative to " + los_rel_to + "; + = towards the satellite. "
                  "Not converted to vertical."),
                 (f"line_of_sight_cumulative_displacement_{unit}_{d0}_{d1}.tif", fp.relative_change(ctx) / div,
                  "line-of-sight cumulative displacement", unit,
                  f"Line-of-sight change from the first image ({d0}) to the last ({d1}), each image relative to "
                  + los_rel_to + "; + = towards the satellite. A two-date difference: each date carries its own "
                  "noise (the time series has every date)."))
    out = []
    ri = _ri()
    for name, arr, qty, units, desc in specs + los_specs:
        a = np.where(ctx.good, arr, np.nan).astype(np.float32)
        los = name.startswith("line_of_sight_")
        try:
            # vertical rasters: "negative = down (sinking)", not the LOS note;
            # "farmland" only where the comparison ground is (ri.say_ground)
            tags = dl.product_tags(cfg, ctx.attrs, qty, units, relative_to=los_rel_to if los else rel_to,
                                   vertical=not los,
                                   sign="unsigned (a 1-sigma uncertainty)" if "uncertainty" in qty else None)
            dl._write_geotiff(D / name, a, ctx.attrs, tags=ri.say_ground(tags, rep.get("control_ground_label")),
                              description=ri.say_ground(desc, rep.get("control_ground_label")), units=units)
        except (AttributeError, TypeError):
            dl._write_geotiff(D / name, a, ctx.attrs)
        out.append(D / name)
    _supersede_old_total(D, unit, keep=f"{AVERAGE_RATE_RASTER}_{unit}_{d0}_{d1}.tif",
                         also=(f"line_of_sight_cumulative_displacement_{unit}_*.tif",
                               f"line_of_sight_cumulative_displacement_{unit}_{d0}_{d1}.tif"))
    snr = write_signal_to_noise(cfg, rep, ctx, null, D)
    if snr is not None:
        out.append(snr)
    # E14 (trust review 7 Oct 2026): the point table's relative columns
    # (mean_velocity_relative, cell_noise_1sigma, significant_95) were
    # written by the export on the FIRST farmland pass; put them on this
    # report's farmland and null, so a point reads what
    # rate_signal_to_noise.tif reads. A table from another products run, or
    # none in D, is left alone (labelled first pass).
    try:
        pts = dl.points_on_report_farmland(cfg, rep, ctx, null, D)
        if not pts.get("applied"):
            log.info("Point table left on the first farmland pass: %s", pts.get("reason"))
    except Exception as exc:                                  # noqa: BLE001 - the table keeps its first-pass label
        log.warning("Point table not put on the report's farmland (%s): it stays on the first pass, "
                    "labelled so", exc)
    log.info("Report-consistent rasters: %s", ", ".join(p.name for p in out))
    return out


def _supersede_old_total(D: Path, unit: str, keep: str | None = None, also: tuple | None = None) -> None:
    """Move a per-cell 'total' raster written under the pre-6 Oct name
    (vertical_movement_total_<unit>_<d0>_<d1>.tif, average rate x years) to
    deliverables/superseded/, so a package never carries both names - and
    an average-rate raster of other dates than `keep` (the one just written):
    a record extended and re-run left the older-dated one beside it, and the
    package copied both into data/. Moved, never deleted."""
    old = sorted(D.glob(f"{OLD_TOTAL_RASTER}_{unit}_*.tif"))
    if keep is not None:
        old += sorted(p for p in D.glob(f"{AVERAGE_RATE_RASTER}_{unit}_*.tif") if p.name != keep)
    if also:                                                  # (glob, the one to keep): the dated LOS raster too
        old += sorted(p for p in D.glob(also[0]) if p.name != also[1])
    if not old:
        return
    dest = D / "superseded" / datetime.now().strftime("rasters_%Y%m%d_%H%M%S")
    dest.mkdir(parents=True, exist_ok=True)
    for p in old:
        for q in (p, p.with_name(p.name + ".aux.xml")):          # with its GIS display settings
            if not q.exists():
                continue
            try:
                q.replace(dest / q.name)
                log.info("Moved %s to %s (superseded by %s)", q.name, dest, keep or f"{AVERAGE_RATE_RASTER}_*")
            except OSError as exc:
                log.warning("Could not move the superseded %s aside (%s)", q.name, exc)


def write_signal_to_noise(cfg, rep, ctx, null: np.ndarray, D: Path) -> Path | None:
    """
    rate_signal_to_noise.tif and signal_to_noise.json again, from the
    report's own field: |relative LOS rate after the second farmland pass
    (ctx.r_cm)| / the per-class cell null of rep['cell_null_by_class'] - the
    numerator and denominator of the client vertical raster, the KMZ's point
    fading and the Results stands_out (signal / threshold >= 1 <=> SNR >=
    1.96). deliverables writes both first, before any feature is found, from
    the FIRST farmland pass; once features that beat chance are cut from the
    farmland the class centres move (Site B class 1 by 0.99 mm/yr LOS), so
    a cell could stand out in one delivered product and not in another
    (review outcome check, 6 Oct 2026). Same file names, same keys.
    """
    from . import deliverables as dl
    r = ctx.r_cm
    good = ctx.good & np.isfinite(r) & np.isfinite(null) & (null > 0)
    snr = np.full(r.shape, np.nan, dtype=np.float32)
    snr[good] = np.abs(r[good]) / null[good]
    sp = rep.get("farmland_second_pass") or {}
    second = bool(sp)
    ground = _ri().ground_word(rep)
    cut = [w for w, on in (("features that beat chance", sp.get("features_removed", 1 if sp else 0)),
                           ("the report areas", sp.get("report_areas_removed"))) if on]
    basis = (f"the report's {ground} (second pass: {' and '.join(cut)} taken out of it)" if cut
             else f"the report's {ground}")
    path = D / "rate_signal_to_noise.tif"
    try:
        tags = dl.product_tags(cfg, ctx.attrs, "rate signal-to-noise: |class-matched relative rate| / per-cell "
                               "noise (los_velocity_cell_noise_1sigma)", "unitless",
                               f"far-field {ground} of the same coherence class, as the report measures it",
                               significant_95_at=1.96, farmland=basis, written_by="client_report",
                               sign="unsigned (an absolute value)")
        dl._write_geotiff(path, snr, ctx.attrs, tags=tags, units="unitless",
                          description=(f"|rate relative to {ground} of the same coherence class| / per-cell noise; "
                                       ">= 1.96 stands out (the report's, the KMZ's and the Results' rule)"))
    except Exception as exc:                                  # noqa: BLE001 - deliverables' first-pass file stays
        log.warning("Report-consistent signal-to-noise raster not written (%s)", exc)
        return None
    s_all, s_far = snr[good], snr[good & ctx.far]
    if not s_all.size:
        return path
    pct = lambda x, k: round(100 * float((x >= k).mean()), 1) if x.size else None          # noqa: E731
    stats = {"definition": ("|rate relative to farmland of the same coherence class| / max(farmland cell spread, "
                            "the cell's rate repeatability)"),
             "cells": int(good.sum()), "median": round(float(np.median(s_all)), 2),
             "percent_above": {str(k): pct(s_all, k) for k in (1, 2, 3)},
             "percent_above_on_farmland_by_chance": {str(k): pct(s_far, k) for k in (1, 2, 3)},
             "percent_standing_out_at_1_96": pct(s_all, 1.96),
             "percent_of_farmland_standing_out_at_1_96_by_chance": pct(s_far, 1.96),
             "farmland": basis, "written_by": "client_report (the report's farmland and cell null)",
             "points_table_note": ("points_attributes.gpkg significant_95 / mean_velocity_relative are from the "
                                   "first farmland pass, written before the report cut anything from it; where "
                                   "they differ, this raster is the report's rule" if second else None),
             "reading": "a share above 2 no larger than farmland's own means the area holds no more movement than chance"}
    tmp = D / "signal_to_noise.json.tmp"
    tmp.write_text(json.dumps(stats, indent=1))
    tmp.replace(D / "signal_to_noise.json")
    log.info("Signal-to-noise (report's farmland): median %.2f; %.1f%% of cells >= 1.96 (farmland by chance %s%%)",
             stats["median"], stats["percent_standing_out_at_1_96"],
             stats["percent_of_farmland_standing_out_at_1_96_by_chance"])
    return path


def build(cfg, out: Path | None = None, out_dir: Path | None = None) -> Path:
    """The report into deliverables/ (json, html, figures/, the report-consistent
    rasters) - or, with out_dir, everything but the rasters into that folder
    (a preview that leaves the delivered report alone)."""
    D = Path(out_dir) if out_dir else cfg.project_dir / "deliverables"
    D.mkdir(parents=True, exist_ok=True)
    rep, ctx, masks, outlines = compute(cfg)
    rep.setdefault("omitted", [])
    # the figures and the page add what they leave out to the record's list
    # (ERR-01): the page is rendered before client_report.json is written,
    # so the record holds them too, and the PDF still follows the record.
    # The PDF draws the figures' stamp again; a failure there is the
    # figures' own, already on the list.
    with _collecting(rep["omitted"]):
        n_compute = len(rep["omitted"])
        # a preview reads the project's basemap cache but writes only to its own folder
        figs = figures(cfg, rep, ctx, masks, outlines, out_dir=D / "figures",
                       cache_dir=(D / "basemap_cache") if out_dir else None)
        if out_dir is None:
            write_rasters(cfg, rep, ctx)
        text = page(cfg, rep, figs)
        if len(rep["omitted"]) > n_compute:
            # left out after compute's release block: refresh its reading
            # list, and render again so the page names them all (the footer
            # is drawn after the list). The second render cannot add more:
            # one entry per block.
            _refresh_omitted_items(rep)
            text = page(cfg, rep, figs)
    (D / "client_report.json").write_text(json.dumps(rep, indent=1))
    # written aside and moved into place whole: a failure leaves no PDF, never
    # the last build's beside this client_report.json or a truncated one
    pdf, part = D / PDF_NAME, D / (PDF_NAME + ".partial")
    try:
        area_pdf(cfg, rep, ctx, outlines, part, figs)
        os.replace(part, pdf)
    except Exception as exc:                                  # noqa: BLE001 - the page is the report; the PDF its handout
        log.warning("Area summaries PDF not written: %s", exc)
        for f in (part, pdf):
            try:
                f.unlink(missing_ok=True)
            except OSError as exc2:
                log.warning("Could not remove %s (%s): it is not this report's", f, exc2)
    out = out or (D / "client_report.html")
    # the page's own sentences say "farmland" for the comparison ground: said
    # as the measured ground where it is not farmland (ri.say_ground)
    out.write_text(_ri().no_negative_zero(_ri().say_ground(text, rep.get("control_ground_label"))))
    log.info("Wrote %s", out)
    return out


def _refresh_omitted_items(rep: dict) -> None:
    """The release block's items for the omitted list as it stands now
    (release.omitted_items). Only soft blocks fail after compute (the
    figures' credits, the footer, the credits section: release.OMITTED_HARD
    names compute's), so FINAL / DRAFT - and the DRAFT mark the figures
    already carry - does not change."""
    from . import release
    rel = rep.get("release")
    if not isinstance(rel, dict) or not hasattr(release, "omitted_items"):
        return
    rel["blockers"] = ([b for b in rel.get("blockers") or [] if b.get("code") not in release.OMITTED_CODES]
                       + release.omitted_items(rep))


def main():
    import argparse
    from .config import load_config
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("config"); ap.add_argument("--out"); ap.add_argument("--out-dir", help="preview folder; deliverables untouched")
    a = ap.parse_args()
    print(build(load_config(a.config), Path(a.out) if a.out else None, Path(a.out_dir) if a.out_dir else None))


if __name__ == "__main__":
    main()
