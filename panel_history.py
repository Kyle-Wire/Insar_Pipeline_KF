"""The full-record history of the mine: every dated mined area, every year,
and the named focus sites.

    python -m insar_pipeline.panel_history <config.yaml> [--sites-only]

Reads the corrected, geocoded time series, the temporal-coherence mask and
the dated mined-area polygons named in the config (mined_year_geojson, with
the year in mined_year_field and the mine/pit name in mined_name_field - the
Indiana DNR CMIS annual mined-area polygons on the Site A record; any
source with a year per polygon elsewhere); writes deliverables/
panel_history.json, panel_history.csv, site_history.json, site_stats.csv and
the figures. Nothing site-specific is hard-coded: until 5 Oct 2026 the file
name of the Site A CMIS layer (looked for beside the mine layer) and its
three mine numbers were, so no other site could have a panel history.

Every series is RELATIVE and CLASS-MATCHED: the region's mean series minus
the far-field series of each temporal-coherence class, weighted by the
region's own coherence mix. That cancels the reference pixel exactly and the
distributed-scatterer bias to the extent it is a function of coherence
(the far field is coherent farmland inside the AOI, outside every panel +
buffer; significance.far_field_mask). Positive = toward the satellite.

Every region also carries `significance`: the same class-matched rate
measured on the region's footprint shifted across ordinary farmland
(significance.assess, on significance.placement_field so the farmland
outside the AOI box is centred on its own class means). Its spread is the
uncertainty to quote. A region whose figure sits inside that spread has not
been shown to move differently from the farmland around it, whatever its rate.

Totals over the record (`total_mm`) are the measured change of a model that
may bend (significance.record_change: yearly hinges plus an annual cycle;
Kyle, 5 Oct 2026), with the fitted rate x span kept as
`total_at_average_rate_mm`; `total_method` says which was possible. The
difference between the last and first dates (`endpoint_change_mm`) is kept
for reference only: it rests on two single epochs (~7 mm of noise each) in
different seasons.

Which farmland (E3, E14, trust review 7 Oct 2026): this stage runs before
the client report, so every relative figure in panel_history.json,
site_history.json and site_stats.csv is on the FIRST farmland pass, in line
of sight, and each file says so (farmland_label,
by_years_since_mining_frame). The client report takes the report areas and
the movement it finds out of the farmland afterwards (exclude_from_farmland)
and builds its own 'Movement by years since mining' table on that farmland,
vertical, by this module's rules (_panel_features, _feature_mask,
_annual_levels, _by_years_since) - never this file's line-of-sight table,
which until 7 Oct 2026 it printed as if it were the report's vertical
figures. site_history(cfg, ctx) on the report's context writes the site
files on its farmland, labelled so.
"""
import json
import logging
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)
MM_IN = 25.4
HI_COH = 0.7
# one colour per mine/pit name, in name order (was keyed to the Gibson names)
PALETTE = ("#2a78d6", "#eb6834", "#1baf7a", "#8e5bd0", "#c9a227", "#d64f8a", "#4a9fb5", "#7a5c3e")
# year properties tried, in order, when mined_year_field is not set
_YEAR_FIELDS = ("year_start", "year_mined", "year", "YEAR", "Year")


def _win(dates, a, b):
    """Indices of dates in [a, b) given YYYYMMDD strings."""
    return [i for i, d in enumerate(dates) if a <= d < b]


def _colors(names) -> dict:
    return {n: PALETTE[i % len(PALETTE)] for i, n in enumerate(sorted(set(names)))}


def _panels_path(cfg) -> Path | None:
    """The dated mined-area layer from the config, resolved like every other
    path in it ($VARS, ~, relative to the project); None when not set."""
    raw = str(getattr(cfg, "mined_year_geojson", "") or "").strip()
    if not raw:
        return None
    p = Path(os.path.expandvars(os.path.expanduser(raw)))
    return p if p.is_absolute() else Path(cfg.project_dir) / p


def _year_of(v) -> int | None:
    """A year from whatever the layer stores: 2016, 2016.0, "2016",
    "2016-05-01", or ArcGIS's epoch milliseconds (1462060800000)."""
    if v is None or isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        if not np.isfinite(v):
            return None
        if abs(v) > 1e11:                                     # epoch milliseconds
            return datetime.fromtimestamp(v / 1000.0, tz=timezone.utc).year
        y = int(v)
        return y if 1800 <= y <= 2100 else None
    m = re.search(r"(?<!\d)(18|19|20)\d\d(?!\d)", str(v))
    return int(m.group(0)) if m else None


def _name_of(v) -> str:
    """A display name from the name property: a bare permit number such as
    802033 reads 'Mine 802033'; anything else is used as written."""
    if v is None:
        return ""
    if isinstance(v, float) and np.isfinite(v) and v.is_integer():
        v = int(v)
    s = str(v).strip()
    return f"Mine {s}" if s.isdigit() else s


def _load_dated(cfg) -> tuple[list, str]:
    """(features, reason). Each feature is normalised to
    {"type": "Feature", "geometry": ..., "properties": {"mine": name, "year": int}}.
    `reason` is a plain sentence when there are none, for the log and for
    panel_history.json, so a skipped history says why."""
    p = _panels_path(cfg)
    if p is None:
        return [], ("no dated mined-area layer is configured (mined_year_geojson), so the panel-by-panel "
                    "history is skipped")
    if not p.exists():
        return [], f"mined_year_geojson {p} does not exist"
    feats = json.loads(p.read_text()).get("features") or []
    yf = str(getattr(cfg, "mined_year_field", "") or "").strip()
    nf = str(getattr(cfg, "mined_name_field", "") or "").strip()
    props = [f.get("properties") or {} for f in feats]
    if not yf:
        yf = next((k for k in _YEAR_FIELDS if any(k in pr for pr in props)), "")
        if yf:
            log.info("mined_year_field not set: using the %r property of %s", yf, p.name)
    if not yf:
        avail = sorted({k for pr in props for k in pr})
        return [], (f"mined_year_field is not set and {p.name} has none of {', '.join(_YEAR_FIELDS)}; "
                    f"its properties are: {', '.join(avail) or 'none'}")
    out, bad = [], 0
    for f, pr in zip(feats, props):
        g = f.get("geometry") or {}
        y = _year_of(pr.get(yf))
        if y is None or g.get("type") not in ("Polygon", "MultiPolygon"):
            bad += 1
            continue
        mine = (_name_of(pr.get(nf)) if nf else "") or "Dated mined areas"
        out.append({"type": "Feature", "geometry": g, "properties": {"mine": mine, "year": y}})
    if bad:
        log.warning("%s: %d of %d features have no readable %r year or are not polygons - left out",
                    p.name, bad, len(feats), yf)
    if not out:
        avail = sorted({k for pr in props for k in pr})
        return [], (f"no feature of {p.name} has a readable year in {yf!r} "
                    f"(properties: {', '.join(avail) or 'none'})")
    return out, ""


def _panel_features(cfg) -> list:
    """The dated mined-area polygons, normalised (see _load_dated). A project
    without such a layer (Site B) has none, and the per-panel history is
    skipped rather than failing the whole figures stage."""
    return _load_dated(cfg)[0]


def _layer_meta(cfg) -> dict:
    p = _panels_path(cfg)
    return {"file": str(p) if p else None,
            "year_field": str(getattr(cfg, "mined_year_field", "") or "") or None,
            "name_field": str(getattr(cfg, "mined_name_field", "") or "") or None}


def _feature_mask(ft_list, shape, attrs, td) -> np.ndarray:
    from insar_pipeline.stack_backend import _polygon_mask
    p = Path(td) / f"f{abs(hash(json.dumps([[f['properties'], f['geometry']] for f in ft_list], sort_keys=True, default=str)))}.geojson"
    p.write_text(json.dumps({"type": "FeatureCollection", "features": ft_list}))
    return _polygon_mask(p, shape, attrs)


def _second_gate(cfg, good: np.ndarray) -> tuple[np.ndarray, dict]:
    """
    The second quality gate (Kyle, 5 Oct 2026: measure, then decide). With
    cfg.min_unwrap_consistency > 0, cells whose MintPy temporal coherence of
    the unwrapped network (geo/geo_temporalCoherence_mintpy.h5: on the
    phase-linked route the linked interferograms close by construction, so
    this residual measures unwrapping consistency) is below it are dropped
    from every figure built on this context. 0 (the default) = off.

    A gate that is asked for but cannot be applied raises: reporting
    ungated figures as gated is exactly the "what it attempted" defect.
    """
    thr = float(getattr(cfg, "min_unwrap_consistency", 0.0) or 0.0)
    info = {"threshold": thr, "applied": False, "layer": None}
    if thr <= 0:
        return good, info
    from insar_pipeline.quality_assessment import _read2d
    from insar_pipeline import deliverables as dl
    p = cfg.mintpy_dir / "geo" / "geo_temporalCoherence_mintpy.h5"
    if not p.exists() and str(getattr(cfg, "backend", "")).lower() in ("stack", "local", "hyp3"):
        # off the phase-linked route MintPy's temporalCoherence.h5 IS the
        # network (unwrapping-consistency) coherence; only miaplpy replaces
        # it with the linked layer and keeps MintPy's as *_mintpy.h5
        p = dl._product(cfg, "temporalCoherence.h5")
    if not p.exists():
        raise FileNotFoundError(
            f"min_unwrap_consistency is {thr:g} but {p} does not exist, so the second quality gate cannot be "
            "applied. Geocode the layer (time_series.geocode_extra_layers) or set min_unwrap_consistency to 0.")
    try:
        um, _ = _read2d(p, "temporalCoherence")
    except KeyError:
        um, _ = _read2d(p)
    if um.shape != good.shape:
        raise ValueError(f"{p.name} is {um.shape}, the product grid is {good.shape}: the second quality gate "
                         "cannot be applied to a layer on another grid")
    keep = np.isfinite(um) & (um >= thr)
    before = int(good.sum())
    gated = good & keep
    after = int(gated.sum())
    info.update({"applied": True, "layer": str(p), "cells_before": before, "cells_after": after,
                 "removed_percent": round(100.0 * (before - after) / before, 2) if before else None})
    log.info("Second quality gate: MintPy unwrapping-consistency coherence >= %g keeps %d of %d cells (%.1f%% removed)",
             thr, after, before, info["removed_percent"] or 0.0)
    return gated, info


def measure_farmland(ctx) -> None:
    """Everything measured against farmland, from ctx.far / ctx.far_wide: the
    class centres (r_cm, class_means), the field the nulls are placed on
    (r_null = r_place) and the class tilt of the far-field series. One
    definition for _Context.__init__ and for the second pass
    (exclude_from_farmland), so the two cannot drift apart. Works on any
    object with sg, vel, tc, far, far_wide and aoi (tests use light ones)."""
    sg = ctx.sg
    ctx.r_cm = ctx.r_null = ctx.r_place = None                # free the old fields before the new
    ctx.r_cm, ctx.class_means = sg.class_matched_field(ctx.vel, ctx.tc, ctx.far)
    # the field the farmland null is placed on. Placements go over all
    # farmland in the product grid (far_wide) so a mine-sized footprint fits
    # more than once or twice, but r_cm is centred on the IN-BOX far field:
    # the outside ring kept its own offset (stats-4) and shifted the null.
    # placement_field re-centres the outside ring on its own class means;
    # inside the AOI it is r_cm, so a region's value is unchanged.
    ctx.r_null = sg.placement_field(ctx.vel, ctx.tc, ctx.far, ctx.far_wide, ctx.aoi)
    ctx.r_place = ctx.r_null
    # The far-field class SERIES are epoch means, so their rate is the class
    # MEAN rate. Each class series is tilted by (centre - mean) x (t - t0) in
    # matched(), so the printed relative rate and the significance value
    # share one zero whatever centre class_means returns. Since 6 Oct 2026
    # the centres ARE the class means (significance.class_means robust=False
    # by default: client figures are area means, and the farmland skew is
    # taken out by the second pass instead), so the tilt is 0 and this is a
    # no-op kept for a caller that asks for median centres (5 Oct: median
    # centres, the series and the values then differed by ~0.65 mm/yr LOS on
    # Site B without it).
    plain = sg.class_means(ctx.vel, ctx.tc, ctx.far, robust=False)
    ctx.class_tilt = {k: float(c) - float(plain[k]) for k, c in ctx.class_means.items()
                      if isinstance(k, int) and isinstance(plain.get(k), (int, float))}


def exclude_from_farmland(ctx, masks, buffer_m: float) -> dict:
    """
    The second pass of the farmland definition (stats-3): movement FOUND in
    the farmland (the masks: features that beat chance) is taken out of it,
    grown by buffer_m (significance.far_without), and everything measured
    against farmland is measured again (measure_farmland). With the class
    MEAN centres used since 6 Oct 2026 (significance.class_means) this pass
    is what takes the pull of unmapped sinking ground out of the zero:
    cutting Site B Features 2 and 3 moves the mean centres 0.52 and 1.03
    mm/yr LOS (median centres, used on 5 Oct, moved 0.21-0.44). Returns
    what changed, as client_report.json farmland_second_pass records it.
    """
    sg = ctx.sg
    kcls = getattr(ctx, "kcls", None)
    kcls = sg.coherence_class(ctx.tc) if kcls is None else kcls

    def per_class(far):
        return {int(k): int(n) for k, n in zip(*np.unique(kcls[far], return_counts=True)) if k >= 0}

    before = {k: float(v) for k, v in ctx.class_means.items() if isinstance(k, int)}
    n0, w0, c0 = int(ctx.far.sum()), int(ctx.far_wide.sum()), per_class(ctx.far)
    ctx.far = sg.far_without(ctx.far, masks, ctx.attrs, buffer_m)
    ctx.far_wide = sg.far_without(ctx.far_wide, masks, ctx.attrs, buffer_m)
    measure_farmland(ctx)
    ctx.farmland_pass = "second"                              # what farmland_label() says
    after = {k: float(v) for k, v in ctx.class_means.items() if isinstance(k, int)}
    c1 = per_class(ctx.far)
    return {"features_removed": len(masks), "buffer_m": float(buffer_m),
            "far_cells": [n0, int(ctx.far.sum())], "far_wide_cells": [w0, int(ctx.far_wide.sum())],
            # per coherence class: farmland cells before/after and the centre's move (LOS mm/yr)
            "far_cells_by_class": {str(k): [c0.get(k, 0), c1.get(k, 0)] for k in sorted(set(c0) | set(c1))},
            "classes_measured": [sorted(before), sorted(after)],
            "class_centre_shift_mm_yr": {str(k): round(after[k] - before[k], 3) for k in after if k in before}}


def farmland_label(ctx) -> str:
    """Which farmland a context's relative figures are on, in words, for
    the technical files that carry them (E14: first-pass values stay in
    technical files only if labelled)."""
    rf = getattr(ctx, "report_farmland", None) or {}
    # results.report_farmland marks every context it saw a report for
    # 'report', but one it could not apply (no second pass recorded, or no
    # class shift) leaves the context on the first pass
    if rf.get("applied") or getattr(ctx, "farmland_pass", "first") == "second":
        return ("the client report's farmland (what the report took out of the first pass - the report areas and "
                "the movement it found - taken out)")
    return ("the FIRST farmland pass (before the client report took the movement it found out of the farmland; "
            "the report's own figures can differ by a constant per coherence class - client_report.json "
            "farmland_second_pass.class_centre_shift_mm_yr)")


def farmland_short(label: str) -> str:
    """farmland_label's words cut to a figure title: 'the FIRST farmland
    pass' or 'the client report's farmland'."""
    return ("the client report's farmland" if str(label).startswith("the client report's")
            else "the FIRST farmland pass")


class _Context:
    """Everything the regional figures share, loaded once."""

    def __init__(self, cfg):
        import h5py
        from insar_pipeline import deliverables as dl
        from insar_pipeline import significance as sg
        from insar_pipeline.quality_assessment import _read2d
        self.cfg, self.sg = cfg, sg
        self.ts_file = dl.timeseries_file(cfg)
        good, self.attrs = _read2d(dl._product(cfg, "maskTempCoh.h5"), "mask")
        vel, _ = _read2d(dl._product(cfg, "velocity.h5"), "velocity")
        self.vel = vel * 1000.0
        self.good = good.astype(bool) & np.isfinite(self.vel)
        self.good, self.second_gate = _second_gate(cfg, self.good)
        from insar_pipeline import landcover
        self.good, self.wooded_mask = landcover.drop_wooded(cfg, self.good, self.attrs)
        self.tc, _ = _read2d(dl._product(cfg, "temporalCoherence.h5"), "temporalCoherence")
        self.far, meta = sg.far_field_mask(cfg, self.good, self.attrs)
        self.far_wide, _ = sg.far_field_mask(cfg, self.good, self.attrs, restrict_to_aoi=False)
        self.union = meta["mine"] & self.good
        self.aoi = meta.get("aoi")
        if self.aoi is None:
            self.aoi = sg.aoi_mask(cfg.bbox, self.attrs, self.good.shape)
        self.null_field = "placement_field (outside-box ring re-centred on its own class means)"
        self.kcls = sg.coherence_class(self.tc)
        self.farmland_pass = "first"                          # exclude_from_farmland makes it "second"
        measure_farmland(self)
        with h5py.File(self.ts_file, "r") as f:
            self.dates = [d.decode() if isinstance(d, bytes) else str(d) for d in f["date"][:]]
        self.t = sg.years(self.dates)
        self.span = float(self.t[-1] - self.t[0])
        self.ref = sg.reference_pixel(self.attrs, cfg)

    def exclude_from_farmland(self, masks, buffer_m: float) -> dict:
        """The second farmland pass on this context (module function
        exclude_from_farmland)."""
        return exclude_from_farmland(self, masks, buffer_m)

    def series(self, regions: dict, envelope_for=()) -> tuple[dict, dict, dict]:
        """Mean series of every region, the far field and each far-field
        coherence class; per-cell cumulative envelopes for `envelope_for`."""
        import h5py
        sg_iter = self.sg.iter_epochs
        regions = dict(regions)
        regions["__far__"] = self.far
        for k in self.class_means:
            regions[f"__far_{k}__"] = self.far & (self.kcls == k)
        idx = {k: np.flatnonzero(v.ravel()) for k, v in regions.items() if v.sum() >= 5}
        out = {k: np.full(len(self.dates), np.nan) for k in idx}
        env = {k: {s: np.full(len(self.dates), np.nan) for s in ("min", "max", "std")} for k in envelope_for if k in idx}
        first = None
        with h5py.File(self.ts_file, "r") as f:
            ds = f["timeseries"]
            for i, ep in sg_iter(ds):
                sl = ep.ravel() * 1000.0
                if i == 0:
                    first = sl.copy()
                for k, ix in idx.items():
                    v = sl[ix]
                    fin = np.isfinite(v)
                    if fin.any():
                        out[k][i] = float(v[fin].mean())
                    if k in env:
                        cum = (sl[ix] - first[ix]); cum = cum[np.isfinite(cum)]
                        if cum.size:
                            env[k]["min"][i], env[k]["max"][i], env[k]["std"][i] = cum.min(), cum.max(), cum.std()
        far_s = out.pop("__far__")
        far_cls = {k: out.pop(f"__far_{k}__") for k in self.class_means if f"__far_{k}__" in out}
        return out, {"far": far_s, "classes": far_cls}, env

    def matched(self, region_series, mask, far):
        """region series minus the far-field class series weighted by the region's
        coherence mix, each class series re-centred on its median rate (class_tilt)"""
        w = {k: int((mask & (self.kcls == k)).sum()) for k in far["classes"]}
        tot = sum(w.values())
        if not tot:
            return region_series - far["far"]
        dt = self.t - self.t[0]
        return region_series - sum((far["classes"][k] + self.class_tilt.get(k, 0.0) * dt) * (n / tot)
                                   for k, n in w.items())

    def totals(self, series, fit, pace_null=None) -> dict:
        """The change over the record: measured (significance.record_change,
        which may bend) when it can be fitted, else rate x span; both kept.
        `pace_null` (contract C3): the farmland copies of the same footprint
        - significance.pace_change_null() of them, or their series in the
        units and frame of `series` - so the bend is used only when its pace
        change beats the farmland's 95th percentile (E10); None keeps the
        uncalibrated F-test (pace_change_tested False)."""
        avg = self.sg.trend_total(fit)
        ch = (self.sg.record_change(self.t, series) if pace_null is None
              else self.sg.record_change(self.t, series, pace_null=pace_null))
        tot = float(ch["change_mm"]) if ch and ch.get("change_mm") is not None else avg
        start = (ch or {}).get("start_mm")
        return {"total_mm": round(tot, 1), "total_in": round(tot / MM_IN, 2),
                "total_at_average_rate_mm": round(avg, 1),
                "total_method": (ch or {}).get("method") or "rate_x_span",
                "nonlinear": bool((ch or {}).get("nonlinear", False)),
                "pace_change_tested": bool((ch or {}).get("pace_change_tested", False)),
                # the fitted curve's value at the first date (season removed): a
                # series zeroed on it ends at total_mm
                "start_mm": round(float(start), 1) if start is not None else None}

    def summarise(self, region_series, mask, far, significance=True, pace_null=None) -> dict:
        """A region's class-matched relative series, rate and totals, and
        (significance) its farmland null - with the local and the
        coherence-matched nulls beside the pooled one (E6, contract C1:
        significance.assess given the coherence grid and the geometry).
        `pace_null`: see totals()."""
        sg = self.sg
        rel = self.matched(region_series, mask, far)
        rel = rel - rel[np.isfinite(rel)][0]
        fit = sg.fit_rate(self.t, rel)
        plain = sg.fit_rate(self.t, region_series - far["far"])
        fin = rel[np.isfinite(rel)]
        out = {"rate_mm_yr": round(fit["slope"], 2), "rate_in_yr": round(fit["slope"] / MM_IN, 3),
               "rate_plain_mm_yr": round(plain["slope"], 2) if plain else None,
               "stderr_white_mm_yr": round(fit["stderr_white"], 2), "stderr_ar1_mm_yr": round(fit["stderr_ar1"], 2),
               **self.totals(rel, fit, pace_null),
               "endpoint_change_mm": round(float(fin[-1] - fin[0]), 1),
               "series_mm": [round(float(v), 2) if np.isfinite(v) else None for v in rel]}
        if significance:
            out["significance"] = sg.assess(self.r_null, self.far_wide, mask, tc=self.tc, attrs=self.attrs)
        return out

    def absolute(self, region_series) -> dict:
        s = region_series - region_series[np.isfinite(region_series)][0]
        fit = self.sg.fit_rate(self.t, s)
        return {"rate_mm_yr": round(fit["slope"], 2), "rate_in_yr": round(fit["slope"] / MM_IN, 3),
                **self.totals(s, fit),
                "series_mm": [round(float(v), 2) if np.isfinite(v) else None for v in s]}


def _annual_levels(dates, rel) -> dict:
    """{calendar year: mean level} over years with at least 4 epochs. The
    difference of two annual means cancels the season, which a slope fitted
    inside one year does not."""
    out = {}
    for y in range(int(dates[0][:4]), int(dates[-1][:4]) + 1):
        w = _win(dates, f"{y}0101", f"{y + 1}0101")
        v = rel[w] if w else np.array([])
        v = v[np.isfinite(v)]
        if v.size >= 4:
            out[y] = float(v.mean())
    return out


def _by_years_since(panels: dict) -> list:
    """Year-on-year change of each dated area's annual mean level, grouped
    by years since its mining/reclamation year (k = calendar year - year
    mined; k = 0 is the change from the year before into the mining year,
    negative k is before mining). The answer to 'how fast does ground settle
    N years after it was mined' across every dated area, as median and
    quartiles over the areas. Their spread is between areas; the farmland
    null of each area's rate is in its own `significance`."""
    by = {}
    for name, p in panels.items():
        for a in p.get("annual_change_mm") or ():
            by.setdefault(a["years_since_mining"], []).append(a["change_mm"])
    rows = []
    for k in sorted(by):
        v = np.array(by[k], float)
        # + 0.0: a figure that rounds to zero is stored as 0.0, never -0.0
        # (it reached the page's 'Movement by years since mining' table)
        rows.append({"years_since_mining": int(k), "n_areas": int(v.size),
                     "median_change_mm_per_yr": round(float(np.median(v)), 1) + 0.0,
                     "q25_mm": round(float(np.percentile(v, 25)), 1) + 0.0,
                     "q75_mm": round(float(np.percentile(v, 75)), 1) + 0.0})
    return rows


def analyse(cfg, ctx: _Context | None = None) -> dict:
    ctx = ctx or _Context(cfg)
    feats = _panel_features(cfg)
    good, attrs = ctx.good, ctx.attrs

    regions = {}
    meta = {}
    groups = {}
    for ft in feats:                                  # one region per (mine, year): a year's polygons together
        pr = ft["properties"]
        groups.setdefault((pr["mine"], pr["year"]), []).append(ft)
    with tempfile.TemporaryDirectory() as td:
        dated_all = _feature_mask(feats, good.shape, attrs, td) if feats else np.zeros_like(good)
        for (mine, y), fts in sorted(groups.items()):
            name = f"{mine} {y}"
            m = _feature_mask(fts, good.shape, attrs, td) & good
            if m.sum() >= 30:
                regions[name] = m
                meta[name] = {"mine": mine, "year": int(y), "n_features": len(fts), "n_pixels": int(m.sum())}
    # the whole workings: the mapped mine layer when there is one (the record
    # always used it), else every dated area together
    if ctx.union.sum() >= 5:
        workings, basis = ctx.union, "the mapped mine layer"
    else:
        workings, basis = dated_all & good, "every dated mined area together (no mapped mine layer)"
    regions["workings"] = workings
    # dated ground the far field still counts as farmland: the far field
    # excludes the mine layer + buffer only, so dated areas outside that layer
    # would sit in the comparison ground (the builder should merge them)
    in_far = int((dated_all & ctx.far).sum())
    if in_far:
        log.warning("%d far-field cells lie inside dated mined areas that are not in the mine layer: the comparison "
                    "ground includes mined ground - add the dated areas to the mine layer (deramp_exclude_geojson)", in_far)
    log.info("Workings %d coherent px, far field %d px, %d dated areas (%d polygons)",
             int(workings.sum()), int(ctx.far.sum()), len(meta), len(feats))
    mines = sorted({mm["mine"] for mm in meta.values()})
    for mine in mines:
        m = np.zeros_like(good)
        for n, mm in meta.items():
            if mm["mine"] == mine:
                m |= regions[n]
        regions[f"{mine} (all panels)"] = m
    series, far, _ = ctx.series(regions)
    dates, t = ctx.dates, ctx.t

    result = {"dates": dates, "reference_pixel": ctx.ref, "far_field_px": int(ctx.far.sum()),
              "layer": _layer_meta(cfg), "workings_basis": basis,
              "dated_area_cells_in_far_field": in_far,
              "second_gate": ctx.second_gate, "null_field": ctx.null_field,
              "sign": ("positive = toward the satellite; subsidence negative; every series is the region minus the "
                       "far field matched by coherence class"),
              "uncertainty_note": ("significance.farmland_spread_1sigma_mm_yr is the uncertainty of a rate: how much "
                                   "ordinary farmland of the same footprint differs. stderr_* are temporal-fit errors "
                                   "only and understate it several-fold."),
              "workings": {"n_pixels": int(workings.sum()), **ctx.summarise(series["workings"], workings, far)},
              "mines": {}, "panels": {}, "workings_by_year": []}
    for mine in mines:
        k = f"{mine} (all panels)"
        if k in series:
            result["mines"][mine] = {"n_pixels": int(regions[k].sum()), **ctx.summarise(series[k], regions[k], far)}

    # year by year, whole workings
    r = np.array([np.nan if v is None else v for v in result["workings"]["series_mm"]])
    prev = None
    for y in range(int(dates[0][:4]), int(dates[-1][:4]) + 1):
        w = _win(dates, f"{y}0101", f"{y + 1}0101")
        if len(w) < 4:
            continue
        mean = float(np.nanmean(r[w])); f_y = ctx.sg.fit_rate(t[w], r[w])
        result["workings_by_year"].append({"year": y, "n_epochs": len(w), "mean_level_mm": round(mean, 1),
                                           "change_from_previous_year_mm": round(mean - prev, 1) if prev is not None else None,
                                           "rate_within_year_mm_yr": round(f_y["slope"], 1) if f_y else None})
        prev = mean

    # each panel: rate, total, significance, and the step across its mining year
    for name, mm in meta.items():
        y = mm["year"]
        s = ctx.summarise(series[name], regions[name], far)
        rel = np.array([np.nan if v is None else v for v in s["series_mm"]])
        pre = _win(dates, f"{y - 1}0701", f"{y}0101"); post = _win(dates, f"{y + 1}0101", f"{y + 1}0701")
        step = float(np.nanmean(rel[post]) - np.nanmean(rel[pre])) if len(pre) >= 3 and len(post) >= 3 else None
        after = _win(dates, f"{y + 1}0101", "99999999")
        f_after = ctx.sg.fit_rate(t[after], rel[after]) if len(after) >= 8 else None
        lv = _annual_levels(dates, rel)
        annual = [{"year": Y, "years_since_mining": Y - y, "change_mm": round(lv[Y] - lv[Y - 1], 1) + 0.0}
                  for Y in sorted(lv) if Y - 1 in lv]
        result["panels"][name] = {**mm, **{k: v for k, v in s.items() if k != "series_mm"},
                                  "step_across_mining_year_mm": round(step, 1) if step is not None else None,
                                  "step_across_mining_year_in": round(step / MM_IN, 2) if step is not None else None,
                                  "step_windows": [f"{y - 1}-07 to {y}-01", f"{y + 1}-01 to {y + 1}-07"] if step is not None else None,
                                  "rate_after_mining_mm_yr": round(f_after["slope"], 2) if f_after else None,
                                  "annual_change_mm": annual,
                                  "series_mm": s["series_mm"]}
    # TECHNICAL copy (E3): line of sight like the rest of this file, on this
    # context's farmland - at the products stage the first pass, before the
    # client report exists - and labelled so. The client page's table is the
    # report's own (client_report._years_since_mining: vertical, on its
    # farmland, by these rules); until 7 Oct 2026 the page printed this one.
    result["by_years_since_mining"] = _by_years_since(result["panels"])
    result["by_years_since_mining_frame"] = "line of sight"
    result["by_years_since_mining_note"] = (
        "TECHNICAL: line of sight (positive = toward the satellite), relative to " + farmland_label(ctx)
        + ", matched by coherence class: change of each dated area's annual mean level from the previous calendar "
        "year, grouped by years since its mining/reclamation year (0 = into the mining year; negative = before "
        "it). Quartiles are across areas; each area's own farmland null is in its significance. Not the client "
        "report's table, which is vertical and on the report's own farmland.")
    result["farmland"] = farmland_label(ctx)
    return result


def write(cfg, res) -> Path:
    out = cfg.project_dir / "deliverables"
    (out / "panel_history.json").write_text(json.dumps(res, indent=1))
    cols = ("mine", "year", "n_pixels", "rate_mm_yr", "rate_in_yr", "farmland_spread_1sigma_mm_yr", "z",
            "distinguishable_from_farmland", "total_mm", "total_in", "total_at_average_rate_mm", "total_method",
            "endpoint_change_mm", "step_across_mining_year_mm", "step_across_mining_year_in", "rate_after_mining_mm_yr")
    from insar_pipeline.areas import formula_safe_text
    from insar_pipeline.deliverables import _software
    rows = [",".join(c if c != "year" else "year_mined" for c in cols)]
    for name, p in sorted(res["panels"].items(), key=lambda kv: (kv[1]["mine"], kv[1]["year"])):
        sig = p.get("significance") or {}
        vals = {**p, **{k: sig.get(k) for k in ("farmland_spread_1sigma_mm_yr", "z", "distinguishable_from_farmland")}}
        # the mine name is a shapefile attribute: on one line and never a
        # formula in a spreadsheet (16 Oct 2026, readiness review SEC-05)
        rows.append(",".join("" if vals.get(k) is None else formula_safe_text(str(vals[k]).replace(",", ";"))
                             for k in cols))
    (out / "panel_history.csv").write_text(
        "# region minus the far field matched by coherence class (" + str(res.get("farmland") or "first farmland pass")
        + "); line of sight, positive = toward the satellite. "
        "farmland_spread_1sigma_mm_yr is the uncertainty of rate_mm_yr (the same measurement over ordinary farmland); "
        "total_mm = measured change over the record (total_method); total_at_average_rate_mm = rate x span; "
        "endpoint_change_mm = last minus first date (noisy)\n"
        "# software: " + _software() + "\n"          # (16 Oct 2026, readiness review OUT-01)
        + "\n".join(rows) + "\n")
    log.info("Wrote %s and .csv", out / "panel_history.json")
    return out / "panel_history.json"


def _fmt_rate(m):
    sig = m.get("significance") or {}
    sd = sig.get("farmland_spread_1sigma_mm_yr")
    return f"{m['rate_mm_yr']:+.2f}" + (f" ± {sd:.1f}" if sd is not None else "") + " mm/yr"


def _epoch_noise(cfg):
    """Single-epoch noise from the QA (robust step RMS), or None."""
    try:
        q = json.loads((cfg.project_dir / "quality_assessment.json").read_text())
        v = ((q.get("checks") or {}).get("single_epoch_noise") or {}).get("robust_mm")
        return float(v) if isinstance(v, (int, float)) else None
    except (OSError, ValueError):
        return None


def plot(cfg, res) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = cfg.project_dir / "deliverables" / "figures"; out.mkdir(parents=True, exist_ok=True)
    dt = [datetime.strptime(d, "%Y%m%d") for d in res["dates"]]
    first_year = int(res["dates"][0][:4])
    recent = {n: p for n, p in res["panels"].items() if p["year"] >= first_year}
    older = {n: p for n, p in res["panels"].items() if p["year"] < first_year}
    n = len(recent); cols = 4; rws = int(np.ceil(n / cols))
    fig = plt.figure(figsize=(16, 4.2 + 2.6 * rws))
    gs = fig.add_gridspec(rws + 1, cols, height_ratios=[1.6] + [1] * rws)
    ax = fig.add_subplot(gs[0, :])
    colors = _colors(list(res["mines"]) + [p["mine"] for p in res["panels"].values()])
    ax.plot(dt, res["workings"]["series_mm"], color="#17202a", lw=2, label=f"all panels ({_fmt_rate(res['workings'])})")
    for mine, m in res["mines"].items():
        ax.plot(dt, m["series_mm"], color=colors.get(mine, "grey"), lw=1.2, label=f"{mine} ({_fmt_rate(m)})")
    ax.axhline(0, color="#8a95a1", lw=0.8); ax.grid(alpha=0.3); ax.legend(fontsize=9, loc="lower left")
    ax.set_ylabel("mm, relative to far field")
    ax2 = ax.secondary_yaxis("right", functions=(lambda v: v / MM_IN, lambda v: v * MM_IN)); ax2.set_ylabel("inches")
    ax.set_title("The workings over the whole record, relative to the surrounding farmland matched by coherence "
                 "(± = spread of ordinary farmland of the same footprint; positive = toward the satellite)", fontsize=10)
    lo = min(np.nanmin(np.array(p["series_mm"], float)) for p in recent.values()) if recent else -30
    hi = max(np.nanmax(np.array(p["series_mm"], float)) for p in recent.values()) if recent else 30
    lo, hi = min(lo, -20), max(hi, 20)
    for k, (name, p) in enumerate(sorted(recent.items(), key=lambda kv: (kv[1]["mine"], kv[1]["year"]))):
        a = fig.add_subplot(gs[1 + k // cols, k % cols])
        a.axvspan(datetime(p["year"], 1, 1), datetime(p["year"] + 1, 1, 1), color=colors.get(p["mine"], "grey"), alpha=0.15, lw=0)
        a.plot(dt, p["series_mm"], color=colors.get(p["mine"], "grey"), lw=1)
        a.axhline(0, color="#8a95a1", lw=0.6); a.set_ylim(lo, hi); a.grid(alpha=0.25); a.tick_params(labelsize=7)
        step = p.get("step_across_mining_year_mm")
        sd = (p.get("significance") or {}).get("farmland_spread_1sigma_mm_yr")
        a.set_title(f"{name} · {p['n_pixels']:,} px · {p['rate_mm_yr']:+.1f}" + (f"±{sd:.0f}" if sd else "") + " mm/yr"
                    + (f" · step {step:+.0f} mm" if step is not None else ""), fontsize=8.5)
        if k % cols == 0:
            a.set_ylabel("mm rel.", fontsize=8)
    old_txt = (f"{len(older)} older panels ({min(p['year'] for p in older.values())}–{first_year - 1}) are in panel_history.csv"
               if older else "no dated area predates the record")
    fig.suptitle(f"Panel-by-panel history, panels mined {first_year}–{res['dates'][-1][:4]} (shaded: the mining year). "
                 + old_txt, fontsize=11, y=0.995)
    fig.tight_layout(); fig.savefig(out / "panel_history.png", dpi=110); plt.close(fig)
    log.info("Wrote %s", out / "panel_history.png")
    return out / "panel_history.png"


def _manifest_upsert(cfg, entry: dict) -> None:
    """Replace (not merely add) a figure's caption, so a corrected caption
    always reaches the page."""
    man = cfg.project_dir / "deliverables" / "figures" / "figures.json"
    m = json.loads(man.read_text()) if man.exists() else []
    m = [x for x in m if x["file"] != entry["file"]] + [entry]
    man.parent.mkdir(parents=True, exist_ok=True)
    man.write_text(json.dumps(m, indent=1))


def _manifest_drop(cfg, name: str) -> None:
    man = cfg.project_dir / "deliverables" / "figures" / "figures.json"
    if man.exists():
        m = json.loads(man.read_text())
        if any(x["file"] == name for x in m):
            man.write_text(json.dumps([x for x in m if x["file"] != name], indent=1))


def _skip(cfg, reason: str) -> None:
    """Record why there is no panel history, and retire a previous one: the
    summary table reads panel_history.json, and a file left from an earlier
    configuration drew rows for panels this run never measured."""
    out = cfg.project_dir / "deliverables"
    out.mkdir(parents=True, exist_ok=True)
    (out / "panel_history.json").write_text(json.dumps(
        {"skipped": True, "reason": reason, "layer": _layer_meta(cfg), "mines": {}, "panels": {}}, indent=1))
    for stale in (out / "panel_history.csv", out / "figures" / "panel_history.png"):
        if stale.exists():
            stale.unlink()
    _manifest_drop(cfg, "panel_history.png")


def run(cfg) -> dict:
    ctx = _Context(cfg)
    feats, reason = _load_dated(cfg)
    if not feats:
        (log.warning if _panels_path(cfg) is not None else log.info)(
            "Panel-by-panel history skipped: %s; the maps, series and summary table still draw", reason)
        _skip(cfg, reason)
        if getattr(cfg, "focus_sites_lalo", ()):
            plot_sites(cfg, site_history(cfg, ctx))
        return {}
    res = analyse(cfg, ctx); write(cfg, res); plot(cfg, res)
    lay = res.get("layer") or {}
    src = Path(lay["file"]).name if lay.get("file") else "the dated mined-area layer"
    noise = _epoch_noise(cfg)
    noise_txt = f"about {noise:.0f} mm" if noise is not None else "several millimetres"
    _manifest_upsert(cfg, {"file": "panel_history.png", "title": "The mine, panel by panel, over the whole record",
                           "caption": ("Top: mean displacement of the workings (" + res.get("workings_basis", "every mapped panel together")
                                       + ") and of each mine's dated areas, relative to the surrounding farmland matched by coherence "
                                       "class, from the first to the last acquisition. The ± after each rate is the spread of the same "
                                       "measurement over ordinary farmland of the same footprint: a rate inside it has "
                                       f"not been shown to differ from the farmland. Below: each dated area in {src} "
                                       f"(year from {lay.get('year_field') or 'its year property'}) worked during the record, with its year shaded; "
                                       "the title gives its rate and the step between the half-year before and the half-year after that year. "
                                       f"Single epochs carry {noise_txt} of noise, so steps of that size are "
                                       "not a signal. Areas dated before the record began are in panel_history.csv; "
                                       "panel_history.json gives the year-on-year change by years since mining.")})
    if getattr(cfg, "focus_sites_lalo", ()):
        plot_sites(cfg, site_history(cfg, ctx))
    return res


def main():
    import sys
    from insar_pipeline.config import load_config
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config(sys.argv[1])
    if len(sys.argv) > 2 and sys.argv[2] == "--sites-only":
        plot_sites(cfg, site_history(cfg))
    else:
        run(cfg)


# ---------------------------------------------------------------------------
# named focus sites (e.g. a client's parcel)
# ---------------------------------------------------------------------------
def parse_focus_sites(cfg) -> list:
    """[(name, S, W, N, E, geojson-or-None)] from cfg.focus_sites_lalo."""
    sites = []
    for spec in getattr(cfg, "focus_sites_lalo", ()) or ():
        name, val = spec.split(":", 1)
        val = val.strip()
        if val.lower().endswith(".geojson"):
            gj = Path(val)
            if not gj.is_absolute():
                gj = Path(cfg.project_dir) / gj
            if not gj.exists():
                log.warning("focus site %r: %s not found - skipped", name, gj); continue
            cs = np.array([c for f in json.loads(gj.read_text())["features"]
                           for poly in (f["geometry"]["coordinates"] if f["geometry"]["type"] == "MultiPolygon" else [f["geometry"]["coordinates"]])
                           for ring in poly for c in ring])
            sites.append((name.strip(), float(cs[:, 1].min()), float(cs[:, 0].min()), float(cs[:, 1].max()), float(cs[:, 0].max()), gj))
        else:
            s_, w, n_, e = (float(v) for v in val.split(","))
            sites.append((name.strip(), min(s_, n_), min(w, e), max(s_, n_), max(w, e), None))
    return sites


def site_history(cfg, ctx: _Context | None = None) -> dict:
    """
    Everything the record says about each named site in cfg.focus_sites_lalo:
    its coherent cells, their class-matched relative series and rate with the
    farmland null, the same on its high-coherence cells, the per-cell rates by
    coherence class, a graph_n_points-cell cluster at its centre (the
    nearest coherent cells, not chosen by rate - how a point-sample survey
    would pick them) in absolute and relative terms, and the same regional
    figures for each mine of the dated mined-area layer.
    """
    from insar_pipeline.stack_backend import _polygon_mask
    sites = parse_focus_sites(cfg)
    if not sites:
        return {}
    ctx = ctx or _Context(cfg)
    sg, good, attrs, tc, vel = ctx.sg, ctx.good, ctx.attrs, ctx.tc, ctx.vel
    ny, nx = good.shape
    lat, _ = sg.rc_to_lalo(attrs, np.arange(ny), 0)
    _, lon = sg.rc_to_lalo(attrs, 0, np.arange(nx))
    hi = tc >= HI_COH
    regions, site_meta = {}, {}
    panels = _panel_features(cfg)
    mine_names = sorted({ft["properties"]["mine"] for ft in panels})
    with tempfile.TemporaryDirectory() as td:
        for name, s_, w, n_, e, gj in sites:
            box = np.outer((lat >= s_) & (lat <= n_), (lon >= w) & (lon <= e))
            if gj is not None:
                box &= _polygon_mask(gj, good.shape, attrs)       # the drawn zone, not its bounding box
            m = box & good
            regions[name] = m; regions[name + "_hi"] = m & hi
            overl = []
            for ft in panels:
                cs = np.array([c for poly in (ft["geometry"]["coordinates"] if ft["geometry"]["type"] == "MultiPolygon" else [ft["geometry"]["coordinates"]]) for ring in poly for c in ring])
                if cs[:, 0].max() < w or cs[:, 0].min() > e or cs[:, 1].max() < s_ or cs[:, 1].min() > n_:
                    continue
                ov = int((_feature_mask([ft], good.shape, attrs, td) & box).sum())
                if ov:
                    pr = ft["properties"]
                    overl.append({"mine": pr["mine"], "year": int(pr["year"]),
                                  "share_of_box_percent": round(100 * ov / max(1, box.sum()), 1)})
            # nearest-cells cluster: the graph_n_points coherent cells nearest the site centre
            cr, cc = sg.lalo_to_rc(attrs, (s_ + n_) / 2, (w + e) / 2)
            py, px = sg.pixel_metres(attrs)
            rr, ccc = np.indices(good.shape)
            dist = np.hypot((rr - cr) * py, (ccc - cc) * px); dist[~good] = np.inf
            order = np.argsort(dist.ravel())[:int(cfg.graph_n_points)]
            cl = np.zeros_like(good); cl.ravel()[order] = True
            regions[name + "_cluster"] = cl
            by_class = []
            for k, lab in enumerate(sg.class_labels()):
                b = m & (ctx.kcls == k); fcl = ctx.far & (ctx.kcls == k)
                if b.sum() >= 10 and fcl.sum() >= 200:
                    by_class.append({"tc": lab, "n_site": int(b.sum()),
                                     "site_mean_mm_yr": round(float(vel[b].mean()), 2),
                                     "far_mean_mm_yr": round(float(vel[fcl].mean()), 2),
                                     "diff_mm_yr": round(float(vel[b].mean() - vel[fcl].mean()), 2)})
            site_meta[name] = {"box_S_W_N_E": [s_, w, n_, e], "polygon": (str(gj) if gj else None),
                               "cells_in_box": int(box.sum()), "coherent_cells": int(m.sum()),
                               "high_coherence_cells": int((m & hi).sum()), "cluster_cells": int(cl.sum()),
                               "cluster_radius_m": round(float(dist.ravel()[order].max()), 0),
                               "overlapping_panels": sorted(overl, key=lambda o: o["year"]), "rate_by_coherence_class": by_class}
        for mine in mine_names:
            fts = [ft for ft in panels if ft["properties"]["mine"] == mine]
            if fts:
                mm = _feature_mask(fts, good.shape, attrs, td) & good
                regions[mine] = mm; regions[mine + "_hi"] = mm & hi
    site_names = [n for n, *_ in sites]
    series, far, envelope = ctx.series(regions, envelope_for=site_names)

    def cell_stats(mask):
        """min / max / mean / median over the cells: rate and trend total (rate x span), absolute and relative."""
        out = {}
        for what, scale in (("rate_mm_yr", 1.0), ("total_mm", ctx.span)):
            for how, arr in (("absolute_vs_reference", vel), ("relative_vs_far_field", ctx.r_cm)):
                a = arr[mask] * scale; a = a[np.isfinite(a)]
                if a.size >= 5:
                    out[f"{what}_{how}"] = {"n": int(a.size), "min": round(float(a.min()), 2), "max": round(float(a.max()), 2),
                                            "mean": round(float(a.mean()), 2), "median": round(float(np.median(a)), 2)}
        return out

    out = {"dates": ctx.dates, "reference_pixel": ctx.ref,
           "sign": "positive = toward the satellite; subsidence negative",
           # which farmland every relative figure here is on (E14): at the
           # products stage the first pass, labelled
           "farmland": farmland_label(ctx), "sites": {}, "mines": {}}
    for name in site_names:
        if name not in series:
            continue
        d = dict(site_meta[name])
        d["cell_stats"] = cell_stats(regions[name])
        d["cell_stats_note"] = ("per-cell statistics over the site's coherent cells; rate = the cell's fitted velocity, "
                                f"total = rate x span ({ctx.span:.2f} years); relative = minus the far-field mean of the "
                                "cell's own coherence class, on " + out["farmland"] + ". Single cells carry several "
                                "mm/yr of noise: quote the mean, and compare it with "
                                "significance.farmland_spread_1sigma_mm_yr")
        env = envelope.get(name)
        if env is not None:
            d["envelope_mm"] = {k: [round(float(v), 2) if np.isfinite(v) else None for v in env[k]] for k in ("min", "max", "std")}
        d["relative_all_cells"] = ctx.summarise(series[name], regions[name], far)
        if name + "_hi" in series:
            d["relative_high_coherence_cells"] = ctx.summarise(series[name + "_hi"], regions[name + "_hi"], far, significance=False)
        d["absolute_all_cells"] = ctx.absolute(series[name])
        if name + "_cluster" in series:
            d["cluster_absolute"] = ctx.absolute(series[name + "_cluster"])
            d["cluster_relative"] = ctx.summarise(series[name + "_cluster"], regions[name + "_cluster"], far)
        out["sites"][name] = d
    for mine in mine_names:
        if mine in series:
            out["mines"][mine] = {"n_pixels": int(regions[mine].sum()), "n_high_coherence": int(regions[mine + "_hi"].sum()),
                                  "relative_all_cells": ctx.summarise(series[mine], regions[mine], far)}
            if mine + "_hi" in series:
                out["mines"][mine]["relative_high_coherence_cells"] = ctx.summarise(
                    series[mine + "_hi"], regions[mine + "_hi"], far, significance=False)
    (cfg.project_dir / "deliverables" / "site_history.json").write_text(json.dumps(out, indent=1))
    from insar_pipeline.areas import formula_safe_text
    from insar_pipeline.deliverables import _software
    rows = ["site,quantity,basis,n,min,max,mean,median,unit"]
    for name, d in out["sites"].items():
        # the site's name as the config or an uploaded layer gives it: one
        # cell, never a formula (16 Oct 2026, readiness review SEC-05)
        cell = formula_safe_text(str(name).replace(",", ";"))
        for key, st in d.get("cell_stats", {}).items():
            how = "absolute vs reference pixel" if "absolute" in key else "relative vs far field (class-matched)"
            q = "rate" if key.startswith("rate") else "total (rate x span)"
            unit_mm, unit_in = ("mm/yr", "in/yr") if key.startswith("rate") else ("mm", "in")
            rows.append(f"{cell},{q},{how},{st['n']},{st['min']},{st['max']},{st['mean']},{st['median']},{unit_mm}")
            rows.append(f"{cell},{q},{how},{st['n']},{st['min'] / MM_IN:.3f},{st['max'] / MM_IN:.3f},{st['mean'] / MM_IN:.3f},{st['median'] / MM_IN:.3f},{unit_in}")
    (cfg.project_dir / "deliverables" / "site_stats.csv").write_text(
        "# per-cell statistics over each site's coherent cells, line of sight; negative = away from the satellite "
        "(subsidence); relative = against " + out["farmland"] + "; see site_history.json for the farmland spread "
        "that a site mean must exceed to count\n"
        "# software: " + _software() + "\n"          # (16 Oct 2026, readiness review OUT-01)
        + "\n".join(rows) + "\n")
    log.info("Wrote %s", cfg.project_dir / "deliverables" / "site_history.json")
    return out


def plot_sites(cfg, res) -> Path | None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from insar_pipeline.deliverables import _software
    if not res.get("sites"):
        return None
    out = cfg.project_dir / "deliverables" / "figures"
    dt = [datetime.strptime(d, "%Y%m%d") for d in res["dates"]]
    n = len(res["sites"]) + len(res["mines"])
    # which farmland every relative line here is on (E14): site_history's own
    # label, in full in the CSV and the caption, short on the figures
    farm = res.get("farmland") or farmland_label(None)
    short = farmland_short(farm)
    fig, axs = plt.subplots(n, 1, figsize=(15, 3.6 * n), sharex=True)
    axs = np.atleast_1d(axs)
    k = 0

    def verdict(m):
        s = m.get("significance") or {}
        return f" — {s['verdict']}" if s.get("verdict") else ""
    for name, d in res["sites"].items():
        a = axs[k]; k += 1
        r = d["relative_all_cells"]
        a.plot(dt, r["series_mm"], color="#2a78d6", lw=1.6,
               label=f"all coherent cells ({d['coherent_cells']:,}): {_fmt_rate(r)}, {r['total_mm']:+.0f} mm ({r['total_in']:+.2f} in) over the record{verdict(r)}")
        if "relative_high_coherence_cells" in d:
            h = d["relative_high_coherence_cells"]
            a.plot(dt, h["series_mm"], color="#1baf7a", lw=1.2, label=f"high-coherence cells only ({d['high_coherence_cells']:,}): {h['rate_mm_yr']:+.2f} mm/yr")
        if "cluster_relative" in d:
            c = d["cluster_relative"]; ca = d["cluster_absolute"]
            a.plot(dt, c["series_mm"], color="#eb6834", lw=1.2, ls="--",
                   label=f"{d['cluster_cells']}-point cluster at the centre: {_fmt_rate(c)} relative, {ca['rate_mm_yr']:+.2f} absolute; {c['total_in']:+.2f} in relative, {ca['total_in']:+.2f} in absolute")
        a.axhline(0, color="#8a95a1", lw=0.8); a.grid(alpha=0.3); a.legend(fontsize=8.5, loc="lower left")
        ov = ", ".join(f"{o['mine']} {o['year']} ({o['share_of_box_percent']}%)" for o in d["overlapping_panels"]) or "no mapped panel"
        a.set_title(f"{name}: {d['box_S_W_N_E'][0]:.5f}–{d['box_S_W_N_E'][2]:.5f} N, {abs(d['box_S_W_N_E'][1]):.5f}–{abs(d['box_S_W_N_E'][3]):.5f} W · overlaps {ov} · relative to the far field, class-matched, {short}", fontsize=10)
        a.set_ylabel("mm"); a.secondary_yaxis("right", functions=(lambda v: v / MM_IN, lambda v: v * MM_IN)).set_ylabel("in")
    for mine, d in res["mines"].items():
        a = axs[k]; k += 1
        r = d["relative_all_cells"]
        a.plot(dt, r["series_mm"], color="#2a78d6", lw=1.6,
               label=f"all coherent cells ({d['n_pixels']:,}): {_fmt_rate(r)}, {r['total_mm']:+.0f} mm ({r['total_in']:+.2f} in){verdict(r)}")
        if "relative_high_coherence_cells" in d:
            h = d["relative_high_coherence_cells"]
            a.plot(dt, h["series_mm"], color="#1baf7a", lw=1.2, label=f"high-coherence cells only ({d['n_high_coherence']:,}): {h['rate_mm_yr']:+.2f} mm/yr")
        a.axhline(0, color="#8a95a1", lw=0.8); a.grid(alpha=0.3); a.legend(fontsize=8.5, loc="lower left")
        a.set_title(f"{mine}, all mapped panels, relative to the far field matched by coherence class, {short}", fontsize=10)
        a.set_ylabel("mm"); a.secondary_yaxis("right", functions=(lambda v: v / MM_IN, lambda v: v * MM_IN)).set_ylabel("in")
    fig.tight_layout(); fig.savefig(out / "site_history.png", dpi=110); plt.close(fig)
    # the project-area graph: every coherent cell in the first site, absolute and relative
    name, d = next(iter(res["sites"].items()))
    if "envelope_mm" in d:
        env = d["envelope_mm"]; ab = d["absolute_all_cells"]["series_mm"]; rl = d["relative_all_cells"]["series_mm"]
        mn, mx = env["min"], env["max"]                 # already cumulative since the first date
        ref = res.get("reference_pixel") or {}
        fig, axs = plt.subplots(2, 1, figsize=(15, 9), sharex=True)
        for a, ser, title, col in ((axs[0], ab, f"Absolute, against the reference pixel ({ref.get('lat', '')}, {ref.get('lon', '')}); carries the reference and the distributed-scatterer bias", "#2a78d6"),
                                   (axs[1], rl, f"Relative, against the surrounding farmland matched by coherence class ({short})", "#1baf7a")):
            y = np.array([np.nan if v is None else v for v in ser], float)
            if a is axs[0]:
                lo = np.array([np.nan if v is None else v for v in mn], float); hi_ = np.array([np.nan if v is None else v for v in mx], float)
                a.fill_between(dt, lo / MM_IN, hi_ / MM_IN, color=col, alpha=0.15, lw=0, label="min–max across the cells")
            a.plot(dt, y / MM_IN, color=col, lw=1.8, marker="o", ms=2.5, label=f"mean of {d['coherent_cells']:,} coherent cells")
            a.axhline(0, color="#8a95a1", lw=0.8); a.grid(alpha=0.3); a.set_ylabel("inches"); a.legend(loc="lower left", fontsize=9)
            a.secondary_yaxis("right", functions=(lambda v: v * MM_IN, lambda v: v / MM_IN)).set_ylabel("mm")
            a.set_title(title, fontsize=10)
        cs = d.get("cell_stats", {})
        ra, rr = cs.get("rate_mm_yr_absolute_vs_reference", {}), cs.get("rate_mm_yr_relative_vs_far_field", {})
        ta, tr = cs.get("total_mm_absolute_vs_reference", {}), cs.get("total_mm_relative_vs_far_field", {})
        sd = (d["relative_all_cells"].get("significance") or {}).get("farmland_spread_1sigma_mm_yr")
        fig.suptitle(f"Project area '{name}': line-of-sight displacement, {res['dates'][0][:4]}-{res['dates'][0][4:6]}-{res['dates'][0][6:]} → {res['dates'][-1][:4]}-{res['dates'][-1][4:6]}-{res['dates'][-1][6:]}  ·  negative = away from the satellite\n"
                     f"rate abs mean {ra.get('mean', 0) / MM_IN:+.2f} in/yr  ·  rel mean {rr.get('mean', 0) / MM_IN:+.2f} in/yr"
                     + (f" ± {sd / MM_IN:.2f} (ordinary farmland of this size)" if sd else "")
                     + f"  ·  total (rate × span) abs mean {ta.get('mean', 0) / MM_IN:+.2f} in, rel mean {tr.get('mean', 0) / MM_IN:+.2f} in"
                     + f"\nTECHNICAL: relative figures on {short}" + ("" if short.startswith("the client report") else
                                                                     " - not the client report's figures"), fontsize=10)
        fig.tight_layout(); fig.savefig(cfg.project_dir / "deliverables" / "los_cumulative_displacement_project_area.png", dpi=110); plt.close(fig)
        csv = ["date,mean_abs_in,min_abs_in,max_abs_in,mean_rel_in"]
        for i, dd in enumerate(res["dates"]):
            f_ = lambda v: "" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v / MM_IN:.3f}"  # noqa: E731
            csv.append(f"{dd},{f_(ab[i])},{f_(mn[i])},{f_(mx[i])},{f_(rl[i])}")
        (cfg.project_dir / "deliverables" / "los_cumulative_displacement_project_area.csv").write_text(
            "# TECHNICAL: every coherent cell in the project area, line of sight; absolute = vs reference pixel, "
            "relative = vs far field matched by coherence class, on " + farm + "; negative = away from the "
            "satellite\n"
            "# software: " + _software() + "\n"           # (16 Oct 2026, readiness review OUT-01)
            + "\n".join(csv) + "\n")
        log.info("Wrote %s", cfg.project_dir / "deliverables" / "los_cumulative_displacement_project_area.png")
    # the caption from what was drawn: the sites' names, the cluster size
    # actually used (graph_n_points), the mines of the dated layer
    names = list(res["sites"])
    n_cl = sorted({d.get("cluster_cells") for d in res["sites"].values() if d.get("cluster_cells")})
    cl_txt = (f"a {'/'.join(str(c) for c in n_cl)}-point cluster of the cells nearest its centre (nearest coherent "
              "cells, not chosen by rate - how a point-sample survey would pick them)") if n_cl else ""
    who = (f"the named site '{names[0]}'" if len(names) == 1 else
           f"each of the {len(names)} named sites ({', '.join(names)})")
    mines_txt = (f" Below: each mine of the dated mined-area layer ({', '.join(res['mines'])}) the same way."
                 if res.get("mines") else "")
    _manifest_upsert(cfg, {"file": "site_history.png",
                           "title": "The named sites" + (" and each mine" if res.get("mines") else "") + ", against the surrounding farmland",
                           "caption": (f"Top: {who}, relative to the surrounding farmland matched by coherence class - "
                                       "every coherent cell, its high-coherence cells" + (f", and {cl_txt}" if cl_txt else "")
                                       + ", in relative and absolute terms. The absolute figure carries the reference pixel and "
                                       "the distributed-scatterer bias and is not a measure of the site's own motion."
                                       + mines_txt + " Relative lines are on " + farm + "." + " The ± after a rate is the spread of the "
                                       "same measurement over ordinary farmland of the same footprint; each legend says "
                                       "whether the rate stands outside it.")})
    log.info("Wrote %s", out / "site_history.png")
    return out / "site_history.png"


if __name__ == "__main__":
    main()
