"""Results for any area and any date range, per point and summarised.

    insar results <config.yaml> --area "Area A2" --from 2020-01-01 --to 2024-12-31 [--export]

The dashboard's Results screen (v) calls the same `analyse()`.

Everything is in the client report's terms, recomputed for the chosen window
with the report's own functions (client_report.area_entry / tested_fields),
so over the full record a report area reads here exactly what it reads in
the report, the KMZ and the rasters (empirical-9, clientout-2: until 5 Oct
2026 the whole area got a third rate here and a self-referential farmland
test):

  rate              the area's mean series relative to ordinary farmland of
                    the same temporal-coherence class (the farmland drift
                    cancels; class centres are means, as the report's),
                    its slope, plus the regional GNSS background when this
                    record's GNSS tie is valid (gnss_tie.check) - reported
                    as vertical (LOS / the up component of the look; assumes
                    the motion is vertical). Without a tie every figure is
                    relative to farmland and says so; the whole study area
                    then has no figure of its own (`relative_only`)
  farmland          the report's: when the report took movement it found out
                    of the farmland (client_report.json farmland_second_pass),
                    the same features are taken out here (report_farmland)
  total             the MEASURED change over the window (significance.
                    record_change, which may bend - but only when its change
                    of pace beats farmland copies of the area's footprint
                    over the same window, a calibrated 5% test (E10, 7 Oct
                    2026); else, and for the whole study area, rate x the
                    window's length); rate x the window's length is kept as
                    "at the average rate", and per point only that one exists
  farmland, E28     the report's cut (farmland_second_pass.cut_cells)
                    holds the report areas it left out of its comparison
                    ground as well as the features it found; an older
                    report's outlines are rebuilt with the areas it names
                    (report_areas_removed)
  test              each area's footprint placed over farmland (significance.
                    assess on the placement field): detected <=> |excess| >=
                    the empirical 95% threshold; signal_to_noise = |excess| /
                    threshold (1 or more stands out); the detection limit is
                    the 80%-power limit. The whole study area is never tested
                    against its own farmland. An area without enough farmland
                    or measured outline reads "Not testable", with no range.
                    A tested area's 95% range (the window's and each year's)
                    is on the test's frame - excess +/- threshold, + GNSS when
                    anchored (client_report.tested_range) - so it need not be
                    centred on the area figure
  per point         the cell's farmland null: over the whole record the
                    report's (client_report.json cell_null_by_class, E18: the
                    rasters' and the KMZ's); for a shorter window the larger
                    of the farmland cell spread over the SAME window and the
                    cell's repeatability (significance.cell_null_by_class);
                    signal_to_threshold = |relative rate| / (1.96 x that),
                    1 or more stands out
  per-year rates    each calendar year in the window with >= 6 dates

Statistics (mean, min, max, range, standard deviation, median) take one value
per radar resolution cell: MintPy geocodes by nearest neighbour, so each radar
pixel fills ~7.5 of the 15 m cells, and a cell count overstates the
measurements (deliverables.radar_pixel_ids). Neighbouring radar cells share
ground, so they are not independent measurements either (E25). Min and max over many points
are dominated by single-point noise; the area figure and its 95% range are
the numbers to quote.

The point columns are named for what they are (empirical-8): the LOS columns
are relative to farmland (+ GNSS), NOT the raw line-of-sight rasters in
deliverables/ (los_velocity_*: against the reference pixel, farmland drift
included), which read several times larger for the same cells.
"""
from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

import numpy as np

from .version import png_metadata          # the build, in every PNG (OUT-01)

log = logging.getLogger(__name__)

MM_IN = 25.4
Z95 = 1.96
MIN_DATES = 6
MIN_YEARS = 0.25
# Above this many points the workbook gets no Points sheet (points.csv has
# them all): Excel's limit is 1,048,576 rows, and openpyxl needs several GB
# and minutes on the dashboard's thread well before that. The Site B
# full area (1.81 M points) left a 3-of-7-sheet workbook behind when it hit
# the limit (techout-2).
XLSX_MAX_POINTS = 200_000
SHEETS = ("Charts", "Summary", "Notes", "Sources", "Statistics", "Annual")


@dataclass
class Result:
    area: str
    start: str
    end: str
    unit: str
    n_dates: int
    years: float
    points: dict            # column -> array, one row per point
    stats: dict             # quantity -> {mean, min, max, range, std, median, n, cells}
    area_summary: dict
    annual: list            # [{year, n_dates, area_mean_vertical, point_noise_1sigma, ...}]
    notes: list = field(default_factory=list)
    series: dict = field(default_factory=dict)   # the area mean over the window, for the trend graph
    entry: dict = field(default_factory=dict)    # client_report.area_entry for this window (mm, vertical)
    # release.numbers_hash of the client_report.json analyse read ("" when
    # there was none; None for a Result analyse did not make)
    numbers_hash: str | None = None
    # the single-point farmland null per coherence class this window used
    # (LOS mm/yr; the report's over the whole record, else this window's -
    # see the module notes) and which: what the results viewer's date
    # windows take as a tested window's null (19 Oct 2026)
    cell_null_by_class: dict = field(default_factory=dict)
    null_source: str = ""


def _parse_date(s) -> date:
    return s if isinstance(s, date) else datetime.strptime(str(s)[:10].replace("/", "-"), "%Y-%m-%d").date()


def report_record(cfg, out_dir: Path | None = None) -> dict:
    """client_report.json (a preview's in out_dir first), or {}."""
    for d in ([Path(out_dir)] if out_dir else []) + [Path(cfg.project_dir) / "deliverables"]:
        p = d / "client_report.json"
        if p.exists():
            try:
                rep = json.loads(p.read_text())
                return rep if isinstance(rep, dict) else {}
            except (OSError, ValueError) as exc:
                log.warning("client_report.json not readable (%s)", exc)
    return {}


def ring_mask(ring, attrs: dict, shape) -> np.ndarray:
    """Cells whose centre lies inside a [[lon, lat], ...] ring (a feature's
    outline in client_report.json), on the product grid."""
    from matplotlib.path import Path as MPath
    out = np.zeros(shape, bool)
    ring = np.asarray(ring, dtype=float)
    if ring.ndim != 2 or ring.shape[0] < 3:
        return out
    y0, x0, dy, dx = (float(attrs[k]) for k in ("Y_FIRST", "X_FIRST", "Y_STEP", "X_STEP"))
    c = (ring[:, 0] - x0) / dx - 0.5
    r = (ring[:, 1] - y0) / dy - 0.5
    r0, r1 = max(0, int(np.floor(r.min()))), min(shape[0], int(np.ceil(r.max())) + 1)
    c0, c1 = max(0, int(np.floor(c.min()))), min(shape[1], int(np.ceil(c.max())) + 1)
    if r1 <= r0 or c1 <= c0:
        return out
    rr, cc = np.mgrid[r0:r1, c0:c1]
    inside = MPath(np.c_[c, r]).contains_points(np.c_[cc.ravel(), rr.ravel()]).reshape(rr.shape)
    out[r0:r1, c0:c1] = inside
    return out


def _bbox_ring(b) -> list:
    w, s_, e, n = b
    return [[w, s_], [e, s_], [e, n], [w, n], [w, s_]]


def report_farmland(ctx, rep: dict | None) -> dict:
    """
    Put a freshly loaded _Context on the report's farmland.

    client_report takes the movement its feature search found (features that
    beat chance) out of the farmland, grown by farmland_second_pass.buffer_m,
    and measures every class centre, the placement field and the class tilt
    again (stats-3). A fresh _Context has the first pass: on Site A every
    relative figure sat ~0.9 mm/yr vertical from the report's, and the nulls
    placed footprints over the found movement itself (wider ranges, smaller
    S/N).

    Two steps. (1) The report's own step (client_report._exclude_from_farmland)
    on the exact cells the report cut (farmland_second_pass.cut_cells, from
    5 Oct 2026: the same farmland to the cell, method "recorded cut"), or for
    an older report on the outlines it stores for those features (`ring`,
    else `bbox`). Those are the SECOND search's outlines, which can be larger
    than the first-pass cut (Site A: 1,792,126 farmland cells against the
    report's 1,819,691). (2) The class
    centres are then set to exactly the report's - first pass + the recorded
    class_centre_shift_mm_yr - by a per-class correction applied to r_cm,
    r_null in the box and the class tilt (`correction_mm_yr`, LOS), so every
    rate is the report's to its rounding; consumers apply the same correction
    to fields they derive (results.analyse for a window). Without outlines
    only step (2) is made (method "class shift"). Idempotent: what was done
    is kept on the context (ctx.report_farmland).
    """
    done = getattr(ctx, "report_farmland", None)
    if done is not None:
        return done
    sp = (rep or {}).get("farmland_second_pass") or {}
    info = {"applied": False, "correction_mm_yr": {}}
    want_shift = _recorded_shift(rep)
    if sp and want_shift:
        before = {int(k): float(v) for k, v in ctx.class_means.items() if isinstance(k, (int, np.integer))}
        from . import client_report as cr
        # the exact cells the report cut (farmland_second_pass.cut_cells,
        # written from 5 Oct 2026); older reports only have the outlines
        cut = getattr(cr, "runs_mask", lambda *a: None)(sp.get("cut_cells"), ctx.good.shape)
        n_feat = None
        if cut is not None and cut.any():
            masks, how = [cut], "recorded cut"
        else:
            rings = [f.get("ring") or (_bbox_ring(f["bbox"]) if f.get("bbox") else None)
                     for f in (rep.get("features") or []) if f.get("stands_out") is True]
            rings += [_bbox_ring(f["bbox"]) for f in (rep.get("features_beyond_listed") or []) if f.get("bbox")]
            masks = [m for m in (ring_mask(r, ctx.attrs, ctx.good.shape) for r in rings if r) if m.any()]
            how = "feature outlines" if masks else "class shift"
            n_feat = len(masks)
            # the report areas the report cut too (E28, contract C2: their
            # whole outlines, recorded by name in report_areas_removed)
            masks += _report_area_outlines(ctx, sp.get("report_areas_removed") or [])
        del cut
        want = (sp.get("far_cells") or [None, None])[1]
        info = {"applied": True, "method": "class shift", "report_far_cells": want}
        if masks:
            rec = cr._exclude_from_farmland(ctx, masks, float(sp.get("buffer_m") or 500.0))
            # the recorded cut is ONE mask holding every feature the report
            # cut: its count is the report's (Site B 2, not 1 - the Notes
            # sheet quotes it)
            info.update({"method": how, "far_cells": rec["far_cells"][1],
                         "features": n_feat if n_feat is not None else int(sp.get("features_removed") or 0)})
            if how != "recorded cut" and sp.get("report_areas_removed"):
                info["report_areas_cut"] = list(sp["report_areas_removed"])
        if sp.get("report_areas_removed"):
            info["report_areas"] = list(sp["report_areas_removed"])          # E28: the Notes sheet names them
        after = {int(k): float(v) for k, v in ctx.class_means.items() if isinstance(k, (int, np.integer))}
        corr = {k: (before[k] + want_shift[k]) - after[k] for k in after if k in before and k in want_shift}
        if info["method"] == "recorded cut" and info.get("far_cells") == want \
                and all(abs(v) <= 0.0006 for v in corr.values()):
            # the same farmland to the cell: what is left is the 3-decimal
            # rounding of the recorded shift, not a difference to correct
            info["rounding_mm_yr"] = {k: round(v, 5) for k, v in corr.items()}
            corr = {}
        cmap = _class_map(corr, ctx.kcls)
        if cmap is not None:
            ctx.r_cm = ctx.r_cm - cmap
            if getattr(ctx, "r_null", None) is not None:
                aoi = getattr(ctx, "aoi", None)
                ctx.r_null = ctx.r_null - (np.where(aoi, cmap, 0.0) if aoi is not None else cmap)
                ctx.r_place = ctx.r_null
            ctx.class_tilt = {k: v + corr.get(int(k), 0.0) for k, v in (getattr(ctx, "class_tilt", {}) or {}).items()}
            ctx.class_means = {k: (v + corr.get(int(k), 0.0) if isinstance(k, (int, np.integer)) else v)
                               for k, v in ctx.class_means.items()}
        info["correction_mm_yr"] = {k: round(v, 4) for k, v in corr.items()}
        log.info("Farmland as the report defines it (%s): %s", info["method"], info)
    if rep:
        ctx.farmland_pass = "report"                          # panel_history.farmland_label
    ctx.report_farmland = info
    return info


def _report_area_outlines(ctx, names) -> list[np.ndarray]:
    """The whole outlines (client_report._mask's footprint) of the named
    report areas, for a report that cut them from its comparison ground
    (E28) but recorded no cut_cells."""
    cfg = getattr(ctx, "cfg", None)
    if not names or cfg is None:
        return []
    from . import client_report as cr
    want = {str(n).lower() for n in names}
    out = []
    for name, spec in cr.parse_areas(cfg):
        if name.lower() in want and spec.strip().lower() != "full":
            got = cr._mask(ctx, spec)
            fp = np.asarray(got[2] if len(got) > 2 else got[0], bool)
            if fp.any():
                out.append(fp)
    return out


def _class_map(corr: dict, kcls: np.ndarray) -> np.ndarray | None:
    """{class: value} as a per-cell map; a class not in it takes the nearest
    listed class's value (significance's nearest-class rule)."""
    if not corr or all(abs(v) < 1e-9 for v in corr.values()):
        return None
    out = np.zeros(kcls.shape, np.float64)
    for i in np.unique(kcls[kcls >= 0]):
        i = int(i)
        out[kcls == i] = corr[i] if i in corr else corr[min(corr, key=lambda j: (abs(j - i), j))]
    return out


def _recorded_shift(rep: dict | None) -> dict:
    """{class: the report's class-centre shift} (farmland_second_pass.
    class_centre_shift_mm_yr, LOS mm/yr: second pass minus first)."""
    sh = {}
    for k, v in (((rep or {}).get("farmland_second_pass") or {}).get("class_centre_shift_mm_yr") or {}).items():
        try:
            sh[int(k)] = float(v)
        except (TypeError, ValueError):
            continue
    return sh


def anchor(cfg, dates) -> dict:
    """The GNSS background by the report's rule (client_report._gnss ->
    gnss_tie.check: this record's window, not older than the velocity):
    {anchored, vertical_mm_yr, vertical_uncertainty_mm_yr, los_mm_yr, reason}.
    No tie -> every value None and the reason; never a silent 0.0 (until
    5 Oct 2026 a missing tie made every figure here relative while the notes
    said 'anchored to the regional GNSS network (+0.00 mm/yr)')."""
    try:
        from . import client_report as cr
        tie, why = cr._gnss(cfg, dates)
        # the workbook and notes are client-facing: the report's plain wording,
        # not gnss_tie.check's file names (7 Oct 2026)
        why = cr.client_gnss_reason(tie, why) or why
    except Exception as exc:                                  # noqa: BLE001 - no tie is a state, not a crash
        tie, why = {}, "the GNSS tie could not be read"
        log.warning("GNSS tie not read for the results (%s)", exc)
    bg = (tie or {}).get("background") or {}
    v = bg.get("vertical_mm_yr")
    if v is None:
        return {"anchored": False, "vertical_mm_yr": None, "vertical_uncertainty_mm_yr": None, "los_mm_yr": None,
                "reason": why or "no GNSS station overlapped the record long enough to measure the regional motion"}
    return {"anchored": True, "vertical_mm_yr": float(v),
            # unknown, not zero: ranges are then withheld (the report's rule)
            "vertical_uncertainty_mm_yr": (float(bg["vertical_uncertainty_mm_yr"])
                                           if bg.get("vertical_uncertainty_mm_yr") else None),
            "los_mm_yr": float(bg["los_mm_yr"]) if bg.get("los_mm_yr") is not None else None, "reason": None}


def basis_sentence(anc: dict, window_is_record: bool = True) -> str:
    """One sentence saying what every rate here is relative to."""
    if anc.get("anchored"):
        return ("Rates are relative to ordinary farmland of the same measurement quality, anchored to the regional "
                f"GNSS network ({anc['vertical_mm_yr']:+.2f} mm/yr vertical"
                + (" over the full record" if not window_is_record else "") + "): absolute movement. "
                "Negative = moving away from the satellite / sinking.")
    return ("Rates are relative to ordinary farmland of the same measurement quality only - NOT GNSS-anchored"
            + (f" ({anc['reason'].rstrip('.')})" if anc.get("reason") else "")
            + ", so they are not absolute, and the movement of the study area as a whole is not measured. "
            "Negative = moving away from the satellite / sinking.")


def area_region(cfg, ctx, area: str) -> dict:
    """A report area by name, or coordinates / WKT / GeoJSON / a polygon file:
    {mask (measured cells), footprint (every cell in the outline), label,
    spec, kind ("full" | "area")}."""
    from . import areas as ar
    from . import client_report as cr
    for name, spec in cr.parse_areas(cfg):
        if area.strip().lower() == name.lower():
            got = cr._mask(ctx, spec)
            m = got[0]
            fp = got[2] if len(got) > 2 else m
            kind = "full" if spec.lower() == "full" else "area"
            # the name the client report shows (a file-stem name is tidied the same way)
            from . import report_insights as ri
            return {"mask": m & ctx.good, "footprint": fp, "spec": spec, "kind": kind,
                    "label": ri.display_name(name, spec) if kind != "full" else name}
    a = ar.load_polygon(area) if ar.normalize_path(area).exists() else ar.parse_coordinates(area)
    if a.is_box:
        w, s, e, n = a.bbox
        m = ctx.sg.aoi_mask((w, s, e, n), ctx.attrs, ctx.good.shape)
    else:
        import tempfile
        from .stack_backend import _polygon_mask
        with tempfile.NamedTemporaryFile("w", suffix=".geojson", delete=False) as fh:
            json.dump(a.geojson(), fh)
        m = _polygon_mask(Path(fh.name), ctx.good.shape, ctx.attrs).astype(bool)
        Path(fh.name).unlink()
    return {"mask": m & ctx.good, "footprint": m, "label": a.label, "spec": area, "kind": "area"}


def area_mask(cfg, ctx, area: str) -> tuple[np.ndarray, str]:
    """(measured cells, label) of an area - see area_region."""
    r = area_region(cfg, ctx, area)
    return r["mask"], r["label"]


def _fit_rates(t: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Slope of each column of Y (n_dates x n) against t, NaN-aware."""
    ok = np.isfinite(Y)
    n = ok.sum(0).astype(float)
    tt = np.where(ok, t[:, None], 0.0)
    yy = np.where(ok, Y, 0.0)
    St, Sy, Stt, Sty = tt.sum(0), yy.sum(0), (tt * tt).sum(0), (tt * yy).sum(0)
    den = n * Stt - St * St
    with np.errstate(invalid="ignore", divide="ignore"):
        r = (n * Sty - St * Sy) / den
    r[(n < MIN_DATES) | (den <= 0)] = np.nan
    return r


def _stats(x: np.ndarray, ids: np.ndarray | None = None) -> dict:
    """Across the points - one value per radar resolution cell when `ids`
    (deliverables.radar_pixel_ids) are given; `cells` is the map-cell
    count, `n` the radar cells the statistics rest on."""
    x = np.asarray(x, dtype=float)
    ok = np.isfinite(x)
    cells = int(ok.sum())
    if ids is not None:
        ids = np.asarray(ids)
        keep = ok & (ids >= 0)
        _, first = np.unique(ids[keep], return_index=True)
        x = np.concatenate([x[keep][first], x[ok & (ids < 0)]])   # an untraced cell counts as its own
    else:
        x = x[ok]
    if not x.size:
        return {"n": 0}
    return {"n": int(x.size), "cells": cells, "mean": float(x.mean()), "min": float(x.min()),
            "max": float(x.max()), "range": float(x.max() - x.min()),
            "std": float(x.std(ddof=1)) if x.size > 1 else 0.0, "median": float(np.median(x))}


def _window_summary(sg, t: np.ndarray, rel: np.ndarray, pace_null=()) -> dict | None:
    """panel_history._Context.summarise / totals for a window's relative
    series (LOS mm): slope, the measured change (record_change) and the
    fitted curve's start, so client_report.area_entry treats a window as
    the report treats the record. `pace_null`: the farmland copies of the
    area's footprint over the same window, LOS mm like `rel` (E10, contract
    C3): the bend is used only when its change of pace beats theirs; with
    fewer than 20 copies (or none: the whole study area) the total is rate
    x span and not called non-linear. Until 7 Oct 2026 the uncalibrated
    F-test decided, which bent on every farmland patch tested."""
    ok = np.isfinite(rel)
    if ok.sum() < MIN_DATES:
        return None
    rel = rel - rel[ok][0]
    fit = sg.fit_rate(t, rel)
    if fit is None:
        return None
    avg = sg.trend_total(fit)
    ch = sg.record_change(t, rel, pace_null=[np.asarray(r, float) for r in pace_null])
    tot = float(ch["change_mm"]) if ch and ch.get("change_mm") is not None else avg
    start = (ch or {}).get("start_mm")
    return {"rate_mm_yr": round(fit["slope"], 3), "total_mm": round(tot, 2), "total_at_average_rate_mm": round(avg, 2),
            "total_method": (ch or {}).get("method") or "rate_x_span", "nonlinear": bool((ch or {}).get("nonlinear", False)),
            "pace_change_tested": bool((ch or {}).get("pace_change_tested", False)),
            "pace_null_n": (ch or {}).get("pace_null_n"),
            "start_mm": float(start) if start is not None else None,
            "series_mm": [round(float(v), 3) if np.isfinite(v) else None for v in rel]}


PACE_COPIES = 30            # client_report.SERIES_NOISE_PLACEMENTS: the copies the report's area null uses


def _footprint_copies(cfg, ctx, rep: dict, mask: np.ndarray, area: str) -> list[np.ndarray]:
    """
    Up to PACE_COPIES non-overlapping copies of an area's footprint over the
    report's wide farmland (client_report._placements, on ctx.far_wide kept
    report_insights.NULL_EXCLUDE_M from every patch the report's search
    found - its features and the patches within chance, by their outlines),
    as flat cell indices: the farmland null of the area's change of pace
    over a window (E10). A report area takes the report's seed (100 + its
    position), so over the whole record its copies are the report's own up
    to the patches' outlines. [] when no copy fits.
    """
    from . import client_report as cr
    from . import report_insights as ri
    sg = ctx.sg
    found = np.zeros(ctx.good.shape, bool)
    for f in list((rep or {}).get("features") or []) + list((rep or {}).get("features_beyond_listed") or []) \
            + list((rep or {}).get("features_within_chance") or []):
        ring = f.get("ring") or (_bbox_ring(f["bbox"]) if f.get("bbox") else None)
        if ring:
            found |= ring_mask(ring, ctx.attrs, ctx.good.shape)
    domain = (ctx.far_wide & ~sg.euclidean_buffer(found, ctx.attrs, ri.NULL_EXCLUDE_M)) if found.any() \
        else ctx.far_wide.copy()
    seed = 100
    for ai, (name, _spec) in enumerate(cr.parse_areas(cfg)):
        if area.strip().lower() == name.lower():
            seed = 100 + ai
    n = int(getattr(cr, "SERIES_NOISE_PLACEMENTS", PACE_COPIES) or PACE_COPIES)
    return cr._placements(mask, domain, n, seed=seed)


def analyse(cfg, area: str, start=None, end=None, ctx=None, progress=None, rep: dict | None = None) -> Result:
    """One pass over the chosen dates of the geocoded time series."""
    import h5py
    from . import client_report as cr
    from . import deliverables as dl
    from .panel_history import _Context
    ctx = ctx or _Context(cfg)
    sg = ctx.sg
    D = Path(cfg.project_dir) / "deliverables"
    rep = report_record(cfg) if rep is None else rep
    # the report's farmland (its second pass) before anything reads ctx.far:
    # the far-field class series, the class centres and the null all use it
    farm = report_farmland(ctx, rep)
    anc = anchor(cfg, ctx.dates)
    bg_v, bg_u, bg_l = anc["vertical_mm_yr"], anc["vertical_uncertainty_mm_yr"], anc["los_mm_yr"]
    anchored = anc["anchored"]

    dates = [datetime.strptime(d, "%Y%m%d").date() for d in ctx.dates]
    t0 = _parse_date(start) if start else dates[0]
    t1 = _parse_date(end) if end else dates[-1]
    sel = [i for i, d in enumerate(dates) if t0 <= d <= t1]
    if len(sel) < MIN_DATES:
        raise ValueError(f"only {len(sel)} images between {t0} and {t1}; at least {MIN_DATES} are needed")
    ty = np.array([(dates[i] - dates[0]).days / 365.25 for i in sel])
    span = float(ty[-1] - ty[0])
    if span < MIN_YEARS:
        raise ValueError(f"{t0} to {t1} spans {span * 12:.1f} months of images; at least 3 months are needed")
    whole = len(sel) == len(dates)
    reg = area_region(cfg, ctx, area)
    mask, label, kind = reg["mask"], reg["label"], reg["kind"]
    if mask.sum() == 0:
        raise ValueError(f"no measured points in {label}")

    ny, nx = ctx.good.shape
    classes = sorted(k for k in ctx.class_means if isinstance(k, (int, np.integer)))
    far_idx = {int(k): np.flatnonzero((ctx.far & (ctx.kcls == k)).ravel()) for k in classes}
    a_idx = np.flatnonzero(mask.ravel())
    # farmland copies of the area's footprint, streamed below: the null of its
    # change of pace over THIS window (E10); none for the whole study area
    pl_idx = _footprint_copies(cfg, ctx, rep, mask, area) if kind != "full" else []
    rng = np.random.default_rng(0)
    s_idx = {k: (v if v.size <= 4000 else np.sort(rng.choice(v, 4000, replace=False))) for k, v in far_idx.items()}
    # streaming least squares for every cell (the farmland null needs the full field)
    # counts and time sums in float32 (t <= ~11 yr, <= ~300 dates: the slope's
    # denominator to ~1e-6), the displacement sums in float64, as
    # client_report._stream; float32 temporaries in the loop (a 6 M-cell grid
    # held ~1.8 GB with float64 ones)
    n = np.zeros(ny * nx, np.float32); St = np.zeros(ny * nx, np.float32); Stt = np.zeros(ny * nx, np.float32)
    Sy = np.zeros(ny * nx); Sty = np.zeros(ny * nx)
    A = np.full((len(sel), a_idx.size), np.nan, np.float32)
    F = {k: np.full(len(sel), np.nan) for k in far_idx}
    PL = np.full((len(pl_idx), len(sel)), np.nan)
    Smp = {k: np.full((len(sel), v.size), np.nan, np.float32) for k, v in s_idx.items()}
    pos = {i: j for j, i in enumerate(sel)}
    yr_of = {j: dates[i].year for j, i in enumerate(sel)}

    aoi = getattr(ctx, "aoi", None)
    if aoi is None:
        aoi = sg.aoi_mask(cfg.bbox, ctx.attrs, ctx.good.shape)

    # the report's class centres exactly (report_farmland's correction), for
    # every field derived here; for a window shorter than the record it is the
    # record's correction
    corr = farm.get("correction_mm_yr") or {}
    cmap = _class_map(corr, ctx.kcls)

    def matched(r):
        rel_, _ = sg.class_matched_field(r, ctx.tc, ctx.far)
        pf = getattr(sg, "placement_field", None)
        nul = pf(r, ctx.tc, ctx.far, ctx.far_wide, aoi) if pf is not None else rel_
        if cmap is not None:
            rel_ = rel_ - cmap
            nul = nul - np.where(aoi, cmap, 0.0)
        return rel_, nul

    up_all = cr._los_up(cfg, (ny, nx))
    up_m = float(np.nanmedian(up_all.ravel()[a_idx]))
    px_km2 = float(np.prod(sg.pixel_metres(ctx.attrs))) / 1e6
    cov = cr.coverage(mask, reg["footprint"], cr._imaged(cfg, (ny, nx)), px_km2)

    def area_year(a) -> dict:
        """One calendar year's area figure from that year's rate field (the
        report's rule on it): the whole area relative + GNSS, untested; every
        other area tested on the placement field, three states."""
        nn = a["n"].astype(np.float64)
        with np.errstate(invalid="ignore", divide="ignore"):
            ry = ((nn * a["Sty"] - a["St"] * a["Sy"]) / (nn * a["Stt"] - a["St"].astype(np.float64) ** 2)).reshape(ny, nx)
        ry[(a["n"].reshape(ny, nx) < MIN_DATES) | ~ctx.good] = np.nan
        rel_y, null_y = matched(ry)
        rel_m = float(np.nanmean(rel_y[mask])) / up_m
        if kind == "full":
            # the whole study area is never tested against its own farmland: its
            # figure is relative + GNSS with the GNSS and block-scatter uncertainty
            if not anchored:
                return {"m": None, "rg": None, "s2n": None,
                        "reading": "relative only (no GNSS anchor): not measured for the area as a whole"}
            m_y = rel_m + bg_v
            bsd = cr._block_sd(rel_y / up_all, mask)
            u_y = math.hypot(bg_u, bsd or 0.0) if bg_u is not None else None
            rg = cr._range(m_y, u_y)
            return {"m": m_y, "rg": rg, "s2n": round(abs(m_y) / (Z95 * u_y), 2) if u_y else None,
                    "reading": ("no measurable movement of the region as a whole" if rg is None or rg[0] <= 0 <= rg[1]
                                else f"region {'sinking' if m_y < 0 else 'rising'} (GNSS-anchored)")}
        sig_y = sg.assess(null_y, ctx.far_wide, mask)
        tf = cr.tested_fields(sig_y, up_m, cov)
        m_y = rel_m + (bg_v if anchored else 0.0)
        # three states, one rule: |excess| >= the empirical 95% threshold; the
        # range on the same frame, as the report's (client_report.tested_range)
        rg, _ = cr.tested_range(tf, sig_y, up_m, bg_v if anchored else None, bg_u)
        regional = cr.moves_with_region({"kind": "area", "anchored_to_gnss": anchored, **tf, "range95_mm_yr": rg})
        return {"m": m_y, "rg": rg, "s2n": tf.get("signal_to_noise"),
                "reading": (("stands out" if tf.get("detected") else
                             "within normal variation" + (f"; {cr.REGION_NOTE}" if regional else ""))
                            if tf["testable"]
                            else "not testable" + (f": {tf['not_testable_reason'].rstrip('.')}"
                                                   if tf.get("not_testable_reason") else ""))}

    # each calendar year's sums are kept only until its last image is read and
    # its field is reduced to the area's figure (area_year): one year's grids
    # alive at a time, not all of them (6 M cells x 11 years was ~1.2 GB)
    n_of = {}
    for j in yr_of:
        n_of[yr_of[j]] = n_of.get(yr_of[j], 0) + 1
    year_fig, cur, Ycur, seen = {}, None, None, 0

    def finish(yr, a, n_yr):
        tsub = [ty[j] for j in yr_of if yr_of[j] == yr]
        if n_yr >= MIN_DATES and tsub[-1] - tsub[0] >= MIN_YEARS:
            year_fig[yr] = area_year(a)

    with h5py.File(ctx.ts_file, "r") as f:
        ds = f["timeseries"]
        done = 0
        for i, ep in sg.iter_epochs(ds):
            if i not in pos:
                continue
            j = pos[i]
            v = ep.ravel() * np.float32(1000.0)          # mm, float32 as panel_history's series
            ok = np.isfinite(v)
            tj = np.float32(ty[j])
            okt = ok * tj
            n += ok; St += okt; Stt += okt * tj
            vz = np.where(ok, v, np.float32(0.0))
            Sy += vz; Sty += vz * tj
            A[j] = v[a_idx]
            if yr_of[j] != cur:                   # the same sums, per calendar year
                cur, seen = yr_of[j], 0
                Ycur = {"n": np.zeros(ny * nx, np.uint16), "St": np.zeros(ny * nx, np.float32),
                        "Stt": np.zeros(ny * nx, np.float32), "Sy": np.zeros(ny * nx, np.float32),
                        "Sty": np.zeros(ny * nx, np.float32)}
            Ycur["n"] += ok; Ycur["St"] += okt; Ycur["Stt"] += okt * tj; Ycur["Sy"] += vz; Ycur["Sty"] += vz * tj
            del okt
            seen += 1
            if seen == n_of[cur]:
                finish(cur, Ycur, seen)
                Ycur = None
            for k, idx in far_idx.items():
                F[k][j] = np.nanmean(v[idx]) if idx.size else np.nan
                Smp[k][j] = v[s_idx[k]]
            for g, ix in enumerate(pl_idx):
                pv_ = v[ix]
                pv_ = pv_[np.isfinite(pv_)]
                PL[g, j] = float(pv_.mean()) if pv_.size else np.nan
            done += 1
            if progress:
                progress(done, len(sel))
    with np.errstate(invalid="ignore", divide="ignore"):
        nn64, St64 = n.astype(np.float64), St.astype(np.float64)
        rate = ((nn64 * Sty - St64 * Sy) / (nn64 * Stt - St64 * St64)).reshape(ny, nx)
    rate[(n.reshape(ny, nx) < MIN_DATES) | ~ctx.good] = np.nan
    del n, St, Stt, Sy, Sty, nn64, St64       # ~200 MB on a 6 M-cell grid, before the null's arrays

    rel, r_null = matched(rate)                                               # LOS mm/yr vs farmland
    spread_map, spread_tab = sg.farmland_cell_spread(rel, ctx.tc, ctx.far)
    try:
        model = json.loads((D / "uncertainty_model.json").read_text())
    except (OSError, ValueError):
        model = {}
    # the per-cell null: over the whole record the report's own
    # (rep['cell_null_by_class'] - the one the uncertainty and signal-to-noise
    # rasters, the KMZ and the report's single-point sentence use, E18); for a
    # shorter window the larger of that window's farmland cell spread and the
    # cell's repeatability (significance.cell_null_by_class, the report's
    # rule), by class index - never by list position (empirical-11). Until
    # 7 Oct 2026 the whole record was measured again here on the report's
    # second pass, so a workbook's per-point 1 sigma and stand-out flags
    # differed from the rasters (Area A1: 6,847 cells standing out
    # against 4,604 in rate_signal_to_noise.tif; final verification)
    rep_null = {str(k): float(v) for k, v in ((rep or {}).get("cell_null_by_class") or {}).items()
                if v is not None and str(k).lstrip("-").isdigit()}
    if whole and rep_null:
        cell_null, null_src = rep_null, "report"
    else:
        cell_null = {str(k): v for k, v in sg.cell_null_by_class(model, spread_tab).items()} \
            if hasattr(sg, "cell_null_by_class") else {}
        null_src = "window"
    null_map = cr.cell_sigma_map(cell_null, ctx.kcls, model) if cell_null else spread_map.astype(float)
    # the class tilt of the far-field SERIES (their epoch means move at the
    # class MEAN rate; the centres are means by default, so it is zero unless
    # the centres are made robust) for this window, plus the correction
    plain = sg.class_means(rate, ctx.tc, ctx.far, robust=False)
    robust = sg.class_means(rate, ctx.tc, ctx.far)
    tilt = {k: (float(robust[k]) - float(plain[k]) if isinstance(robust.get(k), (int, float))
                and isinstance(plain.get(k), (int, float)) else 0.0) + corr.get(k, 0.0) for k in F}

    # ---- per point
    rr, cc = np.unravel_index(a_idx, (ny, nx))
    y0, x0, dy, dx = (float(ctx.attrs[k]) for k in ("Y_FIRST", "X_FIRST", "Y_STEP", "X_STEP"))
    up = up_all.ravel()[a_idx].astype(float)
    rel_p = rel.ravel()[a_idx]
    sd_p = null_map.ravel()[a_idx]
    kc = ctx.kcls.ravel()[a_idx]
    tw = ty - ty[0]
    # each cell minus farmland of ITS OWN class, class by class (one float32
    # copy of A, not a per-point float64 stack of the class series)
    rel_series = np.full_like(A, np.nan)                                      # LOS mm, vs farmland
    # a class too small to measure takes the nearest measured class's series,
    # as its rate takes that class's centre (field_points._class_setup; until
    # 19 Oct 2026 its cells had no change between the window ends)
    lend = {i: min(F, key=lambda j: (abs(j - i), j)) for i in range(len(sg.CLASS_EDGES) - 1)} if F else {}
    for k, src in lend.items():
        cols = np.flatnonzero(kc == k)
        if cols.size:
            rel_series[:, cols] = A[:, cols] - (F[src] + tilt.get(src, 0.0) * tw).astype(np.float32)[:, None]
    unit, div = ("in", MM_IN) if getattr(cfg, "display_units", "in") == "in" else ("mm", 1.0)
    los_tag = "rel_farmland_gnss" if anchored else "rel_farmland"
    los_rate = rel_p + (bg_l if bg_l is not None and anchored else 0.0)
    vert_rate = rel_p / up + (bg_v if anchored else 0.0)
    try:
        ids, ids_basis = dl.radar_pixel_ids(cfg, rr, cc, (ny, nx))
    except (AttributeError, ImportError):
        ids, ids_basis = rr.astype(np.int64) * nx + cc, "product-grid index"
    n_indep = int(np.unique(ids[ids >= 0]).size + (ids < 0).sum())

    def ends(series):
        first = np.array([(d - dates[sel[0]]).days <= 30 for d in (dates[i] for i in sel)])
        last = np.array([(dates[sel[-1]] - d).days <= 30 for d in (dates[i] for i in sel)])
        return np.nanmean(series[last], 0) - np.nanmean(series[first], 0)
    snr_t = np.abs(rel_p) / np.where(sd_p > 0, Z95 * sd_p, np.nan)
    pts = {"point_id": np.arange(1, a_idx.size + 1), "lat": y0 + (rr + 0.5) * dy, "lon": x0 + (cc + 0.5) * dx,
           "radar_pixel_id": ids, "temporal_coherence": ctx.tc.ravel()[a_idx],
           f"los_velocity_{los_tag}_{unit}_yr": los_rate / div, f"vertical_velocity_{unit}_yr": vert_rate / div,
           f"los_change_at_average_rate_{los_tag}_{unit}": los_rate * span / div,
           f"vertical_change_at_average_rate_{unit}": vert_rate * span / div,
           f"vertical_change_between_window_ends_{unit}": (ends(rel_series) / up + (bg_v * span if anchored else 0.0)) / div,
           f"vertical_velocity_1sigma_{unit}_yr": np.hypot(sd_p / up, bg_u if anchored and bg_u else 0.0) / div,
           "signal_to_threshold": snr_t,
           "stands_out": np.where(np.isfinite(snr_t), snr_t >= 1.0, False)}

    # ---- the area, the report's way (client_report.area_entry on this window)
    w = {k: int((kc == k).sum()) for k in F}
    tot_w = sum(w.values())
    a_mean = np.array([np.nanmean(r) if np.isfinite(r).any() else np.nan for r in A])
    far_mix = (sum((F[k] + tilt[k] * tw) * (wk / tot_w) for k, wk in w.items() if wk) if tot_w
               else np.nanmean(np.vstack(list(F.values())), 0))
    # each copy matched to farmland by its OWN coherence mix, as the area is
    pace_null = []
    kc_all = ctx.kcls.ravel()
    for g, ix in enumerate(pl_idx):
        wg = {k: int((kc_all[ix] == k).sum()) for k in F}
        tg = sum(wg.values())
        if tg:
            pace_null.append(PL[g] - sum((F[k] + tilt[k] * tw) * (n_ / tg) for k, n_ in wg.items() if n_))
    del PL
    summ = _window_summary(sg, tw, a_mean - far_mix, pace_null) if mask.sum() >= 5 else None
    if summ is not None and summ.get("total_method") not in (None, "rate_x_span") and pace_null:
        # a bent total's 95% range from the copies' own bent totals, vertical
        # (E9: client_report.total_range, given them as measured_change gives
        # the report's); without them it would be the rate's range x years
        pw = [sg.record_change(tw, r) for r in pace_null]
        summ["copies_piecewise_vertical_mm"] = [float(c["piecewise_change_mm"]) / up_m for c in pw
                                                if c and c.get("piecewise_change_mm") is not None]
    sig = None
    if summ is not None and kind != "full":
        sig = sg.assess(r_null, ctx.far_wide, mask)           # footprint placed over all farmland, as the report does
        summ["significance"] = sig
    full_sd = cr._block_sd(rel / up_all, mask) if kind == "full" else None
    entry = cr.area_entry(label, reg["spec"], kind, cov, summ, sig, up_m, bg_v, bg_u, span, tw,
                          full_rel_sd=full_sd)

    def d_(x):
        # + 0.0: a zero total is 0, never -0 in the workbook (rate x span of a 0.00 rate)
        return None if x is None else x / div + 0.0

    def r_(r):
        return [r[0] / div + 0.0, r[1] / div + 0.0] if r and None not in r[:2] else None
    rng95 = entry.get("range95_mm_yr")
    area_summary = {
        # E25: radar resolution cells, not 'independent measurements' (the
        # page says they are not); the old key stays as an alias for the
        # dashboard's Results screen and older readers, out of the workbook
        "area": label, "kind": kind, "points": int(a_idx.size), "radar_resolution_cells": n_indep,
        "independent_measurements": n_indep,
        "images": len(sel), "first_image": str(dates[sel[0]]), "last_image": str(dates[sel[-1]]), "years": round(span, 2),
        "footprint_km2": cov.get("footprint_km2"), "area_km2": cov.get("area_km2"), "measured_percent": cov.get("measured_percent"),
        f"mean_vertical_velocity_{unit}_yr": d_(entry.get("vertical_mm_yr")),
        f"mean_vertical_cumulative_{unit}": d_(entry.get("total_mm")),            # the measured change
        f"mean_vertical_at_average_rate_{unit}": d_(entry.get("total_at_average_rate_mm")),
        "total_method": entry.get("total_method"),
        f"mean_los_velocity_{los_tag}_{unit}_yr": d_(None if summ is None else
                                                     summ["rate_mm_yr"] + (bg_l if anchored and bg_l is not None else 0.0)),
        f"relative_vertical_velocity_{unit}_yr": d_(entry.get("relative_vertical_mm_yr")),
        f"vertical_95pct_range_{unit}_yr": [rng95[0] / div, rng95[1] / div] if rng95 else None,
        # E9: every total the client reads carries its 95% range (the
        # report's total_range), and an area that could not be compared with
        # farmland the range that can be stated (untested_uncertainty)
        f"measured_change_95pct_range_{unit}": r_(entry.get("total_range95_mm")
                                                   or entry.get("untested_total_range95_mm")),
        f"untested_vertical_95pct_range_{unit}_yr": r_(entry.get("untested_range95_mm_yr")),
        # 80% power, 4 times in 5: SINKING (E5, contract C1); the rising
        # limit beside it (both stored; the client sentence adds the rising
        # one only when it differs by more than 10%)
        f"reliably_detects_{unit}_yr": d_(entry.get("detection_limit_mm_yr")),
        f"reliably_detects_rising_{unit}_yr": d_(entry.get("detection_limit_rise_mm_yr")),
        "pace_change_tested": (summ or {}).get("pace_change_tested"),
        "pace_null_copies": (summ or {}).get("pace_null_n"),
        f"stands_out_above_{unit}_yr": d_(entry.get("decision_threshold_mm_yr")),  # the empirical 95% threshold
        "signal_to_noise": entry.get("signal_to_noise"),                          # |excess| / threshold (>= 1 stands out)
        "reading": entry.get("reading"),
        "testable": entry.get("testable"), "not_testable_reason": entry.get("not_testable_reason"),
        "gnss_anchored": anchored, "relative_only": not anchored,
        "gnss_background_vertical_mm_yr": bg_v, "gnss_background_uncertainty_mm_yr": bg_u,
        "gnss_reason": anc.get("reason"),
        "farmland_second_pass": ({k: ({str(c): x for c, x in v.items()} if isinstance(v, dict) else v)
                                  for k, v in farm.items()} if farm.get("applied") else None),
        "basis": basis_sentence(anc, whole),
    }

    # ---- per calendar year
    annual = []
    years = sorted({dates[i].year for i in sel})
    for yr in years:
        jj = [j for j, i in enumerate(sel) if dates[i].year == yr]
        if len(jj) < MIN_DATES:
            continue
        tsub = ty[jj]
        if tsub[-1] - tsub[0] < MIN_YEARS:
            continue
        r_pts = _fit_rates(tsub, rel_series[jj])
        pts[f"vertical_velocity_{yr}_{unit}_yr"] = (r_pts / up + (bg_v if anchored else 0.0)) / div
        noise = []
        for k, smp in Smp.items():                   # how much farmland points differ within that year
            rs = _fit_rates(tsub, smp[jj] - F[k][jj][:, None])
            rs = rs[np.isfinite(rs)]
            if rs.size > 100:
                noise.append((k, 1.4826 * np.median(np.abs(rs - np.median(rs)))))
        nmap = dict(noise)
        pn = np.array([nmap.get(int(k), np.nan) for k in kc]) / up
        yf = year_fig.get(yr) or {"m": None, "rg": None, "s2n": None, "reading": "not measured"}
        m_y, rg, s2n, reading = yf["m"], yf["rg"], yf["s2n"], yf["reading"]
        row = {"year": yr, "n_dates": len(jj), "first": str(dates[sel[jj[0]]]), "last": str(dates[sel[jj[-1]]])}
        row.update({f"area_mean_vertical_{unit}_yr": d_(m_y),
                    f"area_mean_95pct_low_{unit}_yr": rg[0] / div if rg else None,
                    f"area_mean_95pct_high_{unit}_yr": rg[1] / div if rg else None,
                    "area_signal_to_noise": s2n, "area_reading": reading,
                    f"point_noise_1sigma_{unit}_yr": float(np.nanmedian(pn)) / div if np.isfinite(pn).any() else None,
                    "partial_year": bool(tsub[-1] - tsub[0] < 0.9)})
        annual.append(row)

    # ---- the area's movement over the window (the report's graph, for this window)
    ser = entry.get("series_vertical_mm")
    centre = (bg_v * tw) if anchored else np.zeros_like(tw)
    series = {"date": [str(dates[i]) for i in sel]}
    if ser:
        series[f"area_mean_vertical_{unit}"] = [None if v is None else v / div for v in ser]
        if anchored:
            series[f"regional_gnss_{unit}"] = list(centre / div)
        if rng95:
            # the area's own 95% range x time, as the summary row and the KMZ
            # chart give it (it was centred on the GNSS line or zero with the
            # spread-based 1 sigma, not on the area's rate; D1, 7 Oct 2026)
            series[f"range95_low_{unit}"] = list(rng95[0] * tw / div)
            series[f"range95_high_{unit}"] = list(rng95[1] * tw / div)
    stats = {q: _stats(pts[c], ids) for q, c in (
        ("LOS velocity (relative to farmland" + (" + GNSS)" if anchored else ")"), f"los_velocity_{los_tag}_{unit}_yr"),
        ("Vertical velocity", f"vertical_velocity_{unit}_yr"),
        ("Vertical change at the average rate", f"vertical_change_at_average_rate_{unit}"))}
    notes = [f"{len(sel)} images, {dates[sel[0]]} to {dates[sel[-1]]} ({span:.2f} years).",
             basis_sentence(anc, whole),
             "Area total = the measured change over the window: a fitted curve that may change pace once a year "
             "(season removed), used only when its change of pace is larger than ordinary farmland copies of the "
             "same footprint show over the same window (a 5% test on at least 20 copies); otherwise - and for the "
             "whole study area, which no copy fits - rate x window length. 'at the average rate' = rate x window "
             "length. Per point only the average-rate figure is given.",
             "reliably_detects = sinking of that size or more would be detected 4 times in 5; "
             "reliably_detects_rising = the same for rising.",
             "measured_change_95pct_range = the 95% range of the area total: from ordinary farmland of the same "
             "size (with the regional GNSS uncertainty times the years when anchored), or, for an area that could "
             "not be compared with farmland, from the uncertainty that can be stated without that comparison "
             "(untested_vertical_95pct_range is the same for its rate).",
             f"{a_idx.size:,} map cells from about {n_indep:,} radar resolution cells ({ids_basis}). Neighbouring "
             "map cells share a radar cell and neighbouring radar cells share ground, so these are not independent "
             "radar measurements; the statistics take one value per radar resolution cell. Single points are "
             "noisy: the area figure and its 95% range are the numbers to quote.",
             ("Per point, vertical_velocity_1sigma and signal_to_threshold use the report's per-cell null (the one "
              "its uncertainty and signal-to-noise rasters and the KMZ use)." if null_src == "report" else
              "Per point, vertical_velocity_1sigma and signal_to_threshold use this window's own farmland cell "
              "spread (or the cell's repeatability, whichever is larger)"
              + ("; over the whole record the report's per-cell null would be used." if not whole else
                 ": the report carries no per-cell null of its own.")),
             "The LOS columns are relative to farmland" + (" + GNSS" if anchored else "") + "; they are NOT the raw "
             "line-of-sight rasters (los_velocity_*.tif, against the reference pixel with the farmland drift "
             "included), which read several times larger for the same cells.",
             "signal_to_threshold = a point's movement beyond ordinary farmland / its 95% threshold: 1 or more "
             "stands out."]
    if entry.get("not_testable_reason") and kind != "full":
        notes.append(f"Not testable: {entry['not_testable_reason']}.")
    if farm.get("applied"):
        # what the report took out: the movement it found and (E28) the
        # report areas - an area is never compared with itself
        cut_what = ([f"the movement the report found inside the farmland ({farm['features']} feature(s))"]
                    if farm.get("features") else
                    [] if farm.get("report_areas") else ["the movement the report found inside the farmland "
                                                         "(its features)"])
        if farm.get("report_areas"):
            from . import report_insights as ri
            specs = dict(cr.parse_areas(cfg))
            shown = [ri.display_name(str(n), specs.get(n, "")) for n in farm["report_areas"]]
            cut_what.append(f"the report areas {', '.join(shown)} (an area is never compared with itself)")
        notes.append("Farmland is the report's: " + " and ".join(cut_what)
                     + (" are" if len(cut_what) > 1 else " is") + " taken out of it, as in client_report.json "
                     "farmland_second_pass"
                     + (" (only its class-centre shift: the report stored no feature outlines, so the "
                        "comparison patches may still cover that movement)" if farm["method"] == "class shift" else "")
                     + ("" if whole else "; for a window shorter than the record the record's farmland zero is used")
                     + ".")
    if span < 2:
        notes.append("Windows under 2 years carry seasonal soil and crop signals that read as rates; treat them with care.")
    # the words as printed (workbook Notes / Summary, the Results screen):
    # never '-0.00' (report_insights.no_negative_zero)
    from .report_insights import tidy_text
    notes = tidy_text(notes)
    area_summary = {k: (tidy_text(v) if isinstance(v, str) else v) for k, v in area_summary.items()}
    annual = [{k: (tidy_text(v) if isinstance(v, str) else v) for k, v in r.items()} for r in annual]
    from . import release
    return Result(area=label, start=str(dates[sel[0]]), end=str(dates[sel[-1]]), unit=unit, n_dates=len(sel),
                  years=span, points=pts, stats=stats, area_summary=area_summary, annual=annual, notes=notes,
                  series=series, entry=entry, numbers_hash=release.numbers_hash(rep) if rep else "",
                  cell_null_by_class={int(k): float(v) for k, v in (cell_null or {}).items() if v is not None},
                  null_source=null_src)


# ---------------------------------------------------------------- export
def _atomic_text(path: Path, writer) -> None:
    """writer(tmp_path) then rename: a failed write never leaves a truncated
    file under the final name."""
    tmp = path.with_name(path.name + ".partial")
    try:
        writer(tmp)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def export(cfg, res: Result, out_dir: Path | None = None) -> Path:
    """points.csv, statistics.csv, annual.csv, summary.json, a workbook and a chart.

    The workbook is written to <name>.xlsx.partial, re-opened and checked to
    hold every sheet, and only then renamed (techout-2: a failure used to
    leave a plausible-looking 3-of-7-sheet .xlsx behind). A workbook from an
    earlier export of the same window is removed first, so it can never sit
    beside new CSVs. Over XLSX_MAX_POINTS points the Points sheet is left out
    and the Summary and Notes say where they are."""
    import pandas as pd
    from . import deliverables as dl
    from .areas import formula_safe, formula_safe_frame, slug
    out = out_dir or (Path(cfg.project_dir) / "deliverables" / "results"
                      / f"{slug(res.area)}_{res.start.replace('-', '')}_{res.end.replace('-', '')}")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    xlsx = out / f"{slug(res.area)}_{res.start}_{res.end}.xlsx"
    if xlsx.exists():
        xlsx.unlink()
    # every text cell of the CSVs and the workbook through formula_safe: the
    # area's name is the client's (typed, or from an uploaded file) and a
    # name starting '=' was a formula in the workbook (openpyxl stores such a
    # str as one) and in Excel's reading of the CSVs (16 Oct 2026, readiness
    # review SEC-05)
    pts = pd.DataFrame(res.points)
    _atomic_text(out / "points.csv", lambda p: formula_safe_frame(pts).to_csv(p, index=False, float_format="%.5f"))
    st = pd.DataFrame([{"quantity": q, **{k: v for k, v in s.items()}} for q, s in res.stats.items()])
    _atomic_text(out / "statistics.csv",
                 lambda p: formula_safe_frame(st).to_csv(p, index=False, float_format="%.5f"))
    an = pd.DataFrame(res.annual)
    _atomic_text(out / "annual.csv", lambda p: formula_safe_frame(an).to_csv(p, index=False, float_format="%.5f"))
    rep = report_record(cfg)
    notes = list(res.notes)
    too_many = len(pts) > XLSX_MAX_POINTS
    if too_many:
        notes.append(f"Per-point table: points.csv ({len(pts):,} points) - too many rows for a workbook (kept "
                     f"under {XLSX_MAX_POINTS:,}; Excel's limit is 1,048,576); open it in QGIS, or export a "
                     "smaller area for a Points sheet.")
    # The report numbers this export was computed from (#46, 7 Oct 2026):
    # package._results_problem compares it with release.numbers_hash of the
    # report it packages, so the sign-off rebuild (same numbers, a newer
    # client_report.json) no longer leaves the workbook out, and a changed
    # number still does. Taken from the analysis, not from the report on
    # disk now: a report rebuilt between analysing and exporting is not the
    # one these figures came from. None with no report (judged by age).
    nh = res.numbers_hash
    if nh is None:
        from . import release
        nh = release.numbers_hash(rep) if rep else ""
    summ = {"area_summary": res.area_summary, "statistics": res.stats, "annual": res.annual, "notes": notes,
            "units": res.unit, "points_sheet": not too_many, "numbers_hash": nh or None,
            "software": dl._software()}        # the code that wrote it (16 Oct 2026, readiness review OUT-01)
    _atomic_text(out / "summary.json", lambda p: p.write_text(json.dumps(summ, indent=1, default=_json_default)))
    if res.series:
        _atomic_text(out / "series.csv",
                     lambda p: formula_safe_frame(pd.DataFrame(res.series)).to_csv(p, index=False,
                                                                                    float_format="%.5f"))
        _trend_chart(res, out / "trend.png")
    _chart(res, out / "chart.png")

    try:
        from . import credits as cd
        src = cd.credits(cfg, rep or None)
    except Exception as exc:                                  # noqa: BLE001 - never stop an export for the credits
        log.warning("Credits not read (%s)", exc)
        src = [{"source": "Copernicus Sentinel-1", "kind": "data", "url": "",
                "text": "Contains modified Copernicus Sentinel data (Sentinel-1 radar images)."}]
    try:
        from . import settings
        firm = settings.identity().get("firm") or ""
    except Exception:                                         # noqa: BLE001
        firm = ""
    rows = []
    for k, v in res.area_summary.items():
        if k in SUMMARY_ALIASES:
            continue                       # kept in summary.json for older readers, not shown twice
        nd = _decimals(k, res.unit)
        if isinstance(v, list):
            v = (" to ".join(_shown(x, nd) for x in v) if nd is not None and all(isinstance(x, (int, float))
                                                                                   for x in v)
                 else ", ".join(f"{x:.4f}" if isinstance(x, float) else str(x) for x in v))
        elif isinstance(v, dict):
            v = json.dumps(v)
        elif nd is not None and isinstance(v, float) and round(v, nd) == 0:
            v = 0.0                        # shown as 0.00, never -0.00
        rows.append({"item": k, "value": v})
    if too_many:
        rows.append({"item": "per_point_table", "value": f"points.csv ({len(pts):,} points; too many for a workbook)"})
    rows.append({"item": "software", "value": dl._software()})     # which code made it (2 Nov 2026, OUT-01)
    part = xlsx.with_name(xlsx.name + ".partial")
    expected = list(SHEETS) + (["Points"] if not too_many else []) + (["Trend data"] if res.series else [])

    def sheet(df):
        return formula_safe_frame(df, workbook=True)
    try:
        with open(part, "wb") as fh, pd.ExcelWriter(fh, engine="openpyxl") as xw:
            sheet(pd.DataFrame(rows)).to_excel(xw, sheet_name="Summary", index=False)
            _summary_formats(xw.sheets["Summary"], rows, res.unit)
            sheet(pd.DataFrame({"notes": notes})).to_excel(xw, sheet_name="Notes", index=False)   # the caveats, always
            sheet(pd.DataFrame([{k: c.get(k, "") for k in ("source", "kind", "text", "url")} for c in src])).to_excel(
                xw, sheet_name="Sources", index=False)
            # E9 in the workbook (final verification, 7 Oct 2026): figures to
            # the page's precision, not full floats; the CSVs and summary.json
            # beside it keep every digit
            sheet(_rounded_statistics(st, res.unit)).to_excel(xw, sheet_name="Statistics", index=False)
            sheet(_rounded(an, res.unit)).to_excel(xw, sheet_name="Annual", index=False)
            if not too_many:
                sheet(_rounded(pts, res.unit)).to_excel(xw, sheet_name="Points", index=False)
            if res.series:
                sheet(_rounded(pd.DataFrame(res.series), res.unit)).to_excel(xw, sheet_name="Trend data", index=False)
            from openpyxl.drawing.image import Image as XLImage
            ws = xw.book.create_sheet("Charts", 0)
            ws["A1"] = formula_safe(f"{res.area}: {res.start} to {res.end}", workbook=True)
            row = 3
            for png in ("trend.png", "chart.png"):
                if (out / png).exists():
                    img = XLImage(str(out / png)); img.anchor = f"A{row}"; ws.add_image(img)
                    row += int(img.height / 20) + 2
            title = (rep.get("title") or "").strip()
            xw.book.properties.title = f"{title + ': ' if title else ''}{res.area}, {res.start} to {res.end}"
            xw.book.properties.creator = firm or "InSAR ground-movement pipeline"
            xw.book.properties.description = f"Made by {dl._software()}"
        # verify the outcome, not the attempt: every sheet is in the file
        import openpyxl
        with open(part, "rb") as fh:
            got = openpyxl.load_workbook(fh, read_only=True).sheetnames
        missing = [s for s in expected if s not in got]
        if missing:
            raise RuntimeError(f"the workbook was written without {', '.join(missing)}")
        os.replace(part, xlsx)
    finally:
        if part.exists():
            part.unlink()
    log.info("Results exported to %s (%s)", out, ", ".join(expected))
    return out


# area_summary keys kept for older readers (the dashboard's Results screen
# reads independent_measurements) but left out of the workbook's Summary
SUMMARY_ALIASES = ("independent_measurements",)


def _decimals(name: str, unit: str) -> int | None:
    """The decimals a workbook figure is shown with, from its column or item
    name - the page's precision (E9: nothing more precise than the data):
    in/yr and in to 2, mm/yr to 1, mm to 0, ratios and km2 to 2. None for
    anything else (counts, coordinates, coherence, words)."""
    n = str(name)
    if unit == "in" and (n.endswith("_in_yr") or n.endswith("_in")):
        return 2
    if n.endswith("_mm_yr"):
        return 1
    if n.endswith("_mm"):
        return 0
    if n in ("signal_to_noise", "signal_to_threshold", "area_signal_to_noise") or n.endswith("_km2"):
        return 2
    return None


def _shown(x, nd: int) -> str:
    """A figure as the workbook shows it: nd decimals, signed, never -0.00."""
    from .report_insights import signed
    return signed(float(x), nd)


def _rounded(df, unit: str):
    """A copy of a sheet's table with each figure column rounded to the
    page's precision (_decimals); a value that rounds to zero is 0, not -0."""
    out = df.copy()
    for c in out.columns:
        nd = _decimals(c, unit)
        if nd is not None and out[c].dtype.kind == "f":
            out[c] = out[c].round(nd) + 0.0
    return out


def _rounded_statistics(st, unit: str):
    """The Statistics sheet rounded by quantity: rates to the rate's
    precision, totals to the total's (its rows name the quantity, its
    columns the statistic)."""
    out = st.copy()
    for i, q in enumerate(out.get("quantity", [])):
        nd = _decimals(f"x_{unit}_yr" if "velocity" in str(q).lower() else f"x_{unit}", unit)
        for c in ("mean", "min", "max", "range", "std", "median"):
            if c in out.columns:
                v = out.at[out.index[i], c]
                if isinstance(v, (int, float, np.floating)) and np.isfinite(v):
                    out.at[out.index[i], c] = round(float(v), nd) + 0.0
    return out


def _summary_formats(ws, rows: list, unit: str) -> None:
    """The Summary sheet keeps each figure's full value (it is the record the
    report's numbers are checked against) and shows it to the page's
    precision with a number format; a value that rounds to zero was written
    as 0, so it never shows as -0.00."""
    for i, r in enumerate(rows, start=2):                 # row 1 is the header
        nd = _decimals(r["item"], unit)
        if nd is not None and isinstance(r["value"], float):
            ws.cell(row=i, column=2).number_format = "0" if nd == 0 else "0." + "0" * nd


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return str(o)


def _trend_chart(res: Result, path: Path) -> None:
    """The area's movement over the window, as the report draws it: zeroed on
    the fitted curve at the first image, so it ends at the measured total."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    u, s = res.unit, res.series
    d = [datetime.strptime(x, "%Y-%m-%d") for x in s["date"]]
    rate = res.area_summary.get(f"mean_vertical_velocity_{u}_yr")
    rel = rate is None or not res.area_summary.get("gnss_anchored")
    if rate is None:
        rate = res.area_summary.get(f"relative_vertical_velocity_{u}_yr")
    tot = res.area_summary.get(f"mean_vertical_cumulative_{u}")
    fig, ax = plt.subplots(figsize=(10, 3.8))
    if f"range95_low_{u}" in s:
        ax.fill_between(d, s[f"range95_low_{u}"], s[f"range95_high_{u}"], color="#c9d3db", alpha=0.6, lw=0,
                        label="95% range of the rate")
    if f"regional_gnss_{u}" in s:
        ax.plot(d, s[f"regional_gnss_{u}"], color="#5b6672", lw=1, ls="--", label="regional ground motion (GNSS)")
    y = [np.nan if v is None else v for v in s.get(f"area_mean_vertical_{u}", [])]
    if y:
        ax.plot(d, y, color="#a8432f" if (rate or 0) < 0 else "#2d5f7c", lw=1.6, label=res.area)
    ax.axhline(0, color="#8a95a1", lw=0.6)
    ax.set_ylabel("inches" if u == "in" else "mm"); ax.grid(alpha=0.25)
    if u == "in":
        ax.secondary_yaxis("right", functions=(lambda v: v * MM_IN, lambda v: v / MM_IN)).set_ylabel("mm")
    # the page's precision (E9; this chart goes to client/ as the trend
    # graph): in/yr to 2 decimals with mm/yr to 1, never '-0.00'
    rd, td = (2, 2) if u == "in" else (1, 0)
    sd = _shown
    if rate is None:
        head = f"{res.area}"
    elif u == "in":
        head = f"{res.area}: {sd(rate, rd)} in/yr ({sd(rate * MM_IN, 1)} mm/yr)"
    else:
        head = f"{res.area}: {sd(rate, rd)} mm/yr"
    head += ("  (relative to the surrounding farmland)" if rel else "")
    # the total on a line of its own, with its 95% range (E9: never a bare
    # total; on one line the title ran past a 10-inch figure)
    trng = res.area_summary.get(f"measured_change_95pct_range_{u}")
    head += (f"\n{sd(tot, td)} {u} measured over {res.start} to {res.end}"
             + (f" (95% {sd(trng[0], td)} to {sd(trng[1], td)} {u})" if trng else "") if tot is not None else "")
    ax.set_title(head, fontsize=10.5, loc="left")
    ax.legend(fontsize=8.5, loc="lower left")
    fig.suptitle("Vertical movement over the window (negative = sinking; zero = the fitted movement at the first image)",
                 fontsize=11)
    fig.tight_layout(); _tidy(fig); fig.savefig(path, dpi=120, metadata=png_metadata()); plt.close(fig)


def _chart(res: Result, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    u = res.unit
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    v = np.asarray(res.points[f"vertical_velocity_{u}_yr"], dtype=float)
    axs[0].hist(v[np.isfinite(v)], bins=40, color="#2d5f7c")
    m = res.area_summary.get(f"mean_vertical_velocity_{u}_yr")
    if m is not None:
        axs[0].axvline(m, color="#a8432f", lw=2, label="area figure")
        axs[0].legend(fontsize=8)
    axs[0].set_xlabel(f"vertical velocity, {u}/yr"); axs[0].set_ylabel("points")
    axs[0].set_title(f"{res.area}: points, {res.start} to {res.end}", fontsize=10, loc="left")
    rows = [a for a in res.annual if a.get(f"area_mean_vertical_{u}_yr") is not None]
    if rows:
        yrs = [a["year"] for a in rows]
        mm = [a[f"area_mean_vertical_{u}_yr"] for a in rows]
        axs[1].bar(yrs, mm, color=["#a8432f" if x < 0 else "#2d5f7c" for x in mm])
        er = [(y_, x, a[f"area_mean_95pct_low_{u}_yr"], a[f"area_mean_95pct_high_{u}_yr"]) for y_, x, a in zip(yrs, mm, rows)
              if a.get(f"area_mean_95pct_low_{u}_yr") is not None]
        if er:
            # drawn from its own ends: a tested year's range is on the test's
            # frame (excess over the null median +/- threshold; D1, 7 Oct 2026),
            # so the bar need not sit at its middle, and an errorbar about the
            # bar raised "yerr must not contain negative values" when the bar
            # lay outside it
            axs[1].errorbar([e[0] for e in er], [0.5 * (e[2] + e[3]) for e in er],
                            yerr=[0.5 * (e[3] - e[2]) for e in er], fmt="none", ecolor="#55616e",
                            capsize=3, label="95% range of the area figure")
            axs[1].legend(fontsize=8)
        axs[1].axhline(0, color="#8a95a1", lw=0.7)
        axs[1].set_ylabel(f"{u}/yr")
        axs[1].set_title("Area figure, each calendar year", fontsize=10, loc="left")
    fig.tight_layout(); _tidy(fig); fig.savefig(path, dpi=110, metadata=png_metadata()); plt.close(fig)


def _tidy(fig) -> None:
    from .report_insights import tidy_figure
    tidy_figure(fig)


def _f(x, fmt):
    """A number for the console; never '-0.000' (report_insights.no_negative_zero)."""
    if x is None:
        return "—"
    from .report_insights import no_negative_zero
    return no_negative_zero(format(x, fmt))


def main(argv=None) -> int:
    import argparse
    import sys
    from .config import load_config
    ap = argparse.ArgumentParser(prog="insar results", description="Results for an area and a date range")
    ap.add_argument("config")
    ap.add_argument("--area", required=True, help='a report area name, "lat,lon,lat,lon", WKT, or a polygon file')
    ap.add_argument("--from", dest="start"); ap.add_argument("--to", dest="end")
    ap.add_argument("--export", action="store_true", help="write CSV / Excel / chart")
    a = ap.parse_args(argv if argv is not None else sys.argv[1:])
    logging.basicConfig(level=logging.WARNING)
    cfg = load_config(a.config)
    res = analyse(cfg, a.area, a.start, a.end,
                  progress=lambda d, n: print(f"\r  reading images {d}/{n}", end="", flush=True) if d % 10 == 0 or d == n else None)
    print()
    s = res.area_summary; u = res.unit
    n_rc = s.get("radar_resolution_cells", s.get("independent_measurements"))
    print(f"\n{res.area}: {s['points']:,} points (about {n_rc:,} radar resolution cells), {s['images']} images, "
          f"{s['first_image']} to {s['last_image']} ({s['years']:.2f} yr)")
    lo, hi = s[f"vertical_95pct_range_{u}_yr"] or (None, None)
    tlo, thi = s.get(f"measured_change_95pct_range_{u}") or (None, None)
    rf, tf = ("+.2f", "+.2f") if u == "in" else ("+.1f", "+.0f")      # the page's precision (E9)
    print(f"  vertical {_f(s[f'mean_vertical_velocity_{u}_yr'], rf)} {u}/yr (95% {_f(lo, rf)} to {_f(hi, rf)}), "
          f"measured {_f(s[f'mean_vertical_cumulative_{u}'], tf)} {u} (95% {_f(tlo, tf)} to "
          f"{_f(thi, tf)}); signal / threshold "
          f"{_f(s['signal_to_noise'], '.2f')}: {s['reading']}")
    print(f"  {s['basis']}")
    print(f"\n  {'':44s}{'mean':>9}{'min':>9}{'max':>9}{'range':>9}{'std':>9}{'n':>9}")
    for q, st in res.stats.items():
        if not st.get("n"):
            continue
        nd = (2 if u == "in" else 1) if "velocity" in q.lower() else (2 if u == "in" else 0)
        print(f"  {q + ' (' + (u + '/yr' if 'velocity' in q else u) + ')':44s}"
              + "".join(f"{_f(st[k], f'>+9.{nd}f')}" if k not in ("range", "std") else f"{_f(st[k], f'>9.{nd}f')}"
                        for k in ("mean", "min", "max", "range", "std")) + f"{st['n']:>9,}")
    if res.annual:
        print("\n  year  images  area figure (vertical)   95% range              S/T   one point (1 sigma)  reading")
        for r in res.annual:
            lo, hi = r[f"area_mean_95pct_low_{u}_yr"], r[f"area_mean_95pct_high_{u}_yr"]
            print(f"  {r['year']}  {r['n_dates']:5d}  {_f(r[f'area_mean_vertical_{u}_yr'], '+10' + rf[1:])} {u}/yr   "
                  f"{_f(lo, rf)} to {_f(hi, rf)}   {_f(r['area_signal_to_noise'], '4.1f')}   "
                  f"{_f(r[f'point_noise_1sigma_{u}_yr'], rf[1:])}   {r['area_reading']}"
                  f"{'  (part year)' if r['partial_year'] else ''}")
    if a.export:
        try:
            print(f"\n  exported: {export(cfg, res)}")
        except Exception as exc:                              # noqa: BLE001 - say so and fail, never a partial file
            print(f"\n  export FAILED: {exc} (no workbook was left behind)", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
