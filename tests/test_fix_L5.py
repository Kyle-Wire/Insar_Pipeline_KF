"""Lane L5a of the 5 Oct 2026 review: the client report's numbers
(client_report.compute and its helpers) and report_insights.

clientout-1, empirical-1, methodology-7   totals are the measured change
methodology-1, clientout-4, empirical-6,  detection limit = 80% power; S/N =
  stats-11                                |excess| / threshold, one rule
stats-2, gap-surface_mine_site-2,         not testable (no / thin farmland):
  methodology-2                           no range, no verdict, honest wording
generality-1                              no GNSS: relative only, no
                                          zero-width range, no GNSS claim
methodology-4, gap-surface_mine_site-3    measured share of each outline
stats-13                                  an unmeasurable area is shown, not a crash
clientout-2                               the whole area defined once (rel + GNSS)
stats-15                                  uncertainty raster = the per-cell null
stats-7                                   map band centred on farmland
timeseries-9                              corrections from what was applied
methodology-6, stats-12                   trend words tested against farmland
stats-6                                   calibrated feature search, fail closed

Lane L5b (presentation and wording, the second half of this file):
clientout-6, stats-8                      chart bands with per-date noise; the
                                          caption matches the decision
clientout-5, generality-4, clientout-12   data-driven map bins, extend arrows,
                                          scale bar, north arrow, offline basemap
usability-4, clientout-7, empirical-7,    posting and resolution from the run;
  tests-6                                 study-area points; no fixed 15 m / 2 km
methodology-9                             calibrated method, DS-bias size,
                                          confirmed vs worth-a-look features
clientout-10, stats-9, methodology-5,     direction words only when the range
  empirical-5                             excludes zero; GNSS said as GNSS
empirical-12                              reference wording from the measurement
methodology-15, stats-16                  LOS factor and the mining limitation
clientout-18, generality-16, empirical-13 wording, names, hemispheres, labels
gap-surface_mine_site-7                   surface-mine wording, the 'other' layer
clientout-9, gap-licensing-11             DRAFT/final, footer, no path, offline
gap-licensing-5                           mine layer source and vintage
clientout-16                              one-page PDF per area, glossary, recent

Every test checks an OUTCOME on synthetic data with a known answer. The
end-to-end ones build a small geocoded project (120 x 120 cells of ~100 m,
280 dates over 9.2 years) in a temporary folder.
"""
import json
from datetime import datetime
import types
from pathlib import Path

import h5py
import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from insar_pipeline import client_report as cr
from insar_pipeline import report_insights as ri
from insar_pipeline import significance as sg
from insar_pipeline.config import PipelineConfig

NY = NX = 120
X0, Y0 = -87.60, 38.40
DX, DY = 0.00115, -0.0009                     # ~100 m x 100 m at 38.3 N
BBOX = [X0, Y0 + NY * DY, X0 + NX * DX, Y0]
UP = float(np.cos(np.radians(39.5)))         # the synthetic geometry's LOS -> vertical factor
FEATURE = (slice(30, 45), slice(70, 85))     # 15 x 15 cells, ~2.25 km2, unmapped
FIELD_BOX = (slice(80, 100), slice(20, 40))  # ordinary farmland, a report area
PIT_BOX = (slice(5, 25), slice(5, 25))       # a report area 70% unmeasured
BG_V, BG_U = -1.5, 0.6                       # GNSS background, vertical mm/yr


def _lalo(r, c):
    return Y0 + (r + 0.5) * DY, X0 + (c + 0.5) * DX


def _box_spec(sl):
    """The box around the cells of `sl`, its edges a quarter cell outside the
    edge cells' centres (aoi_mask keeps cells whose centre is inside: an edge
    exactly on a centre, rounded to 6 decimals, kept or lost a row by chance)."""
    s_, w = _lalo(sl[0].stop - 1, sl[1].start)
    n, e = _lalo(sl[0].start, sl[1].stop - 1)
    s_, n, w, e = s_ - 0.25 * abs(DY), n + 0.25 * abs(DY), w - 0.25 * DX, e + 0.25 * DX
    return f"{s_:.6f},{w:.6f},{n:.6f},{e:.6f}"


def _dates(n=280, step=12):
    import datetime as dt
    d0 = dt.date(2016, 1, 5)
    return [(d0 + dt.timedelta(days=step * i)).strftime("%Y%m%d") for i in range(n)]


def _attrs():
    return {"X_FIRST": str(X0), "Y_FIRST": str(Y0), "X_STEP": str(DX), "Y_STEP": str(DY),
            "LENGTH": str(NY), "WIDTH": str(NX), "REF_Y": "3", "REF_X": "3",
            "REF_LAT": str(Y0 + 3.5 * DY), "REF_LON": str(X0 + 3.5 * DX), "UNIT": "m/year"}


def _h5(path, data: dict, attrs, dates=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        for k, v in data.items():
            f.create_dataset(k, data=v)
        if dates is not None:
            f.create_dataset("date", data=np.array([d.encode() for d in dates]))
        for k, v in attrs.items():
            f.attrs[k] = v


def feature_truth(t):
    """LOS mm: spoil-like settling, fast first and slowing (the Site B Feature 1 case)."""
    return -250.0 * (1.0 - np.exp(-t / 1.5))


def make_project(tmp_path, *, gnss=True, seed=7) -> PipelineConfig:
    cfg = PipelineConfig(project_dir=tmp_path / "proj", bbox=BBOX, start_date="2016-01-01", end_date="2025-03-31",
                         backend="miaplpy", acceptance_far_field_buffer_m=500.0,
                         project_title="Testsite", project_location="Test County, Indiana",
                         report_areas=("Full Satellite Imaging:full", f"Test Field:{_box_spec(FIELD_BOX)}",
                                       f"Wet Pit:{_box_spec(PIT_BOX)}"))
    geo = cfg.mintpy_dir / "geo"
    dates = _dates()
    t = sg.years(dates)
    rng = np.random.default_rng(seed)
    # farmland: fields differ persistently (a correlated per-cell rate, ~2 mm/yr)
    # plus atmosphere-like correlated noise on every date and a little white noise
    rate = gaussian_filter(rng.normal(0, 1, (NY, NX)), 3)
    rate = rate / rate.std() * 2.0
    ts = np.empty((len(dates), NY, NX), np.float32)
    for i in range(len(dates)):
        atm = gaussian_filter(rng.normal(0, 1, (NY, NX)), 6)
        ts[i] = (rate * t[i] + atm / atm.std() * 3.0 + rng.normal(0, 2.0, (NY, NX))
                 + 6.0 * np.sin(2 * np.pi * t[i])) / 1000.0
    ts[:, FEATURE[0], FEATURE[1]] += (feature_truth(t) / 1000.0)[:, None, None]
    vel = np.polyfit(t, ts.reshape(len(t), -1), 1)[0].reshape(NY, NX).astype(np.float32)
    A = dict(_attrs(), START_DATE=dates[0], END_DATE=dates[-1], REF_DATE=dates[0])
    mask = np.ones((NY, NX), bool)
    pit = np.zeros((NY, NX), bool)
    pit[PIT_BOX] = True
    rr, cc = np.nonzero(pit)
    mask[rr[: int(0.7 * rr.size)], cc[: int(0.7 * rr.size)]] = False      # 70% of the pit unmeasured
    _h5(geo / "geo_velocity.h5", {"velocity": vel}, A)
    _h5(geo / "geo_timeseries_ERA5_ramp_demErr.h5", {"timeseries": ts}, A, dates)
    _h5(geo / "geo_maskTempCoh.h5", {"mask": mask}, A)
    _h5(geo / "geo_temporalCoherence.h5", {"temporalCoherence": np.full((NY, NX), 0.95, np.float32)}, A)
    _h5(geo / "geo_geometryRadar.h5", {"incidenceAngle": np.full((NY, NX), 39.5, np.float32),
                                       "azimuthAngle": np.full((NY, NX), -100.0, np.float32),
                                       "height": np.zeros((NY, NX), np.float32)}, A)
    D = cfg.project_dir / "deliverables"
    (D / "figures").mkdir(parents=True, exist_ok=True)
    (D / "summary.json").write_text(json.dumps({"total_measured_points": NY * NX, "track": 92,
                                                "orbit_direction": "ASCENDING", "look_angle_deg": 39.5}))
    (D / "uncertainty_model.json").write_text(json.dumps({
        "by_class": [{"tc": "0.9-1.0", "class_index": 4, "empirical_mm_yr": 1.0, "n": 14000}],
        "cell_null_by_class": {"4": 2.5}}))
    (cfg.project_dir / "quality_assessment.json").write_text(json.dumps({
        "checks": {"coverage": {"aoi_percent_at_threshold": 97.0, "pass": True},
                   "network": {"connected": True, "pass": True}},
        "findings": {}, "passed": 2, "flagged": 0, "verdict": "MEETS PRACTICE"}))
    if gnss:
        (D / "gnss_tie.json").write_text(json.dumps({
            "record_window": [dates[0], dates[-1]], "stations": [], "inside_checks": [], "los_up_median": UP,
            "uncertainty_floor_mm_yr": BG_U,               # a current tie (gnss_tie.stale_reason)
            "background": {"vertical_mm_yr": BG_V, "vertical_uncertainty_mm_yr": BG_U, "los_mm_yr": BG_V * UP,
                           "uncertainty_basis": "regional floor"}}))
    return cfg


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    cfg = make_project(tmp_path_factory.mktemp("l5"))
    rep, ctx, masks, outlines = cr.compute(cfg)
    return types.SimpleNamespace(cfg=cfg, rep=rep, ctx=ctx, masks=masks, outlines=outlines)


def _area(rep, name):
    return next(a for a in rep["areas"] if a["name"] == name)


def _series(x):
    return np.array([np.nan if v is None else v for v in x], dtype=float)


# ------------------------------------------------------------------ totals (clientout-1, empirical-1, methodology-7)
def test_feature_total_is_the_measured_change_not_rate_x_span(built):
    rep, ctx = built.rep, built.ctx
    assert rep["features"], "the planted settling patch must be found and stand out"
    f = rep["features"][0]
    t = ctx.t
    # the feature is the smoothed-field patch, a little larger than the planted
    # block, so its own series is the truth: its shape is the planted settling
    s = _series(f["series_vertical_mm"])
    shape = feature_truth(t) - feature_truth(t[0])
    k = float(np.nansum(s * shape) / np.nansum(shape * shape))              # dilution by the patch's edge
    true = k * shape[-1]
    assert 0.5 < k / (1 / UP) <= 1.05
    # the old rate x span is far off for movement that slowed; the measured change is not
    assert abs(f["total_at_average_rate_mm"] - true) > 0.15 * abs(true)
    assert abs(f["total_relative_mm"] - true) < 0.05 * abs(true)
    assert f["nonlinear"] is True and f["total_method"] == "piecewise_linear_yearly"
    # the chart ends where its title says: the plotted series' change matches the total
    assert abs((np.nanmean(s[-5:]) - np.nanmean(s[:5])) - f["total_relative_mm"]) < 0.05 * abs(true)
    # the words: an average in the past tense, then the recent pace
    assert "averaged" in f["text"] and "Measured over the record" in f["text"]
    assert f["recent_rate_mm_yr"] is not None and abs(f["recent_rate_mm_yr"]) < 0.5 * abs(f["vertical_relative_mm_yr"])


def test_area_totals_are_measured_and_keep_the_average_rate_figure(built):
    for a in built.rep["areas"]:
        if a.get("series_vertical_mm") is None or a.get("total_mm") is None:
            continue
        assert "total_at_average_rate_mm" in a and a["total_method"] in ("piecewise_linear_yearly", "rate_x_span")
        s = _series(a["series_vertical_mm"])
        span_noise = 3 * 3.0 / UP + 3.0                       # a few mm of date-to-date noise at the ends
        assert abs((np.nanmean(s[-5:]) - np.nanmean(s[:5])) - a["total_mm"]) < span_noise


def test_feature_text_steady_movement_stays_present_tense():
    f = {"sign": -1, "area_km2": 1.0, "lat": 38.3, "lon": -87.5, "mean_rel_vertical_mm_yr": -20.0,
         "vertical_relative_mm_yr": -20.0, "total_relative_mm": -200.0, "nonlinear": False,
         "trend": {"shape": "steady", "text": ""}, "mines": {"phrase": "away from every mapped mine"},
         "chance": {"tested": True, "p": 0.0167, "copies_as_strong": 0, "copies": 59}, "quality_standout": False,
         "cause_hint": "x."}
    txt = ri.feature_text(1, f)
    assert txt.startswith("It moves about 0.79 in/yr") and "averaged" not in txt


# ------------------------------------------------------------------ detection limit (methodology-1, clientout-4, empirical-6, stats-11)
def test_detection_limit_is_detected_four_times_in_five(built):
    """Plant the report's own detection limit into farmland placements of the
    area's footprint: at least ~80% must be called detected by the report's
    rule (|excess| >= threshold). 2 sigma - the old 'limit' - is caught about
    half the time."""
    a = _area(built.rep, "Test Field")
    assert a["testable"] is True
    m = built.masks["Test Field"]
    nl = sg.footprint_null(built.ctx.r_null, built.ctx.far_wide, m & np.isfinite(built.ctx.r_null))
    v = nl["values"]
    d = v - np.median(v)
    thr = a["decision_threshold_mm_yr"] * a["los_up"]
    lim = a["detection_limit_mm_yr"] * a["los_up"]
    for sign in (-1, 1):
        assert np.mean(np.abs(d + sign * lim) >= thr) >= 0.78
    two_sd = 2 * float(d.std())
    assert np.mean(np.abs(d - two_sd) >= thr) < 0.70
    assert a["detection_limit_mm_yr"] > a["decision_threshold_mm_yr"]


def test_signal_to_noise_and_reading_follow_one_rule():
    base = {"testable": True, "farmland_spread_1sigma_mm_yr": 2.0, "spread_for_ranges_mm_yr": 2.0,
            "threshold_95_mm_yr": 4.6, "detectable_80_mm_yr": 6.2, "null_median_mm_yr": 0.5}
    for exc in (-9.0, -4.7, -4.5, 0.3, 4.4, 4.8, 12.0):
        s2t = abs(exc) / 4.6
        tf = cr.tested_fields(dict(base, excess_mm_yr=exc, signal_to_threshold=round(s2t, 2),
                                   distinguishable_from_farmland=abs(exc) >= 4.6), up=0.8)
        assert tf["detected"] == (tf["signal_to_noise"] >= 1)
        assert tf["detection_limit_mm_yr"] == round(6.2 / 0.8, 1) and tf["decision_threshold_mm_yr"] == round(4.6 / 0.8, 1)


def test_page_states_the_80_percent_limit_not_would_have_been_detected(built):
    html = cr.page(built.cfg, built.rep, _fake_figs(built))
    assert "would be detected 4 times in 5" in html
    assert "would have been detected" not in html and "Smallest movement this record could detect" not in html
    rows = cr.summary_rows(built.rep)
    lim_cells = [r[4] for r in rows]
    assert any("mm/yr" in c for c in lim_cells)


# ------------------------------------------------------------------ not testable (stats-2, gap-surface_mine_site-2, methodology-2)
def _untestable_sig():
    return sg._not_testable(-3.0, "Too little ordinary farmland around this area to compare it with.",
                            null_thin=True, approx_independent_placements=2, n_placements=12)


def test_untestable_area_has_no_range_no_verdict_and_never_reads_within_normal():
    cov = {"cells": 400, "area_km2": 4.0, "footprint_km2": 4.0, "measured_percent": 100.0,
           "outside_frame_percent": 0.0, "coverage_balanced": True}
    summ = {"rate_mm_yr": -3.0, "total_mm": -30.0, "series_mm": list(np.linspace(0, -30, 50)), "start_mm": 0.0}
    a = cr.area_entry("Mine Block", "x", "area", cov, summ, _untestable_sig(), 0.78, BG_V, BG_U, 10.0,
                      np.linspace(0, 10, 50))
    assert a["testable"] is False and a["detected"] is None
    assert a["range95_mm_yr"] is None and a["vertical_uncertainty_1sigma_mm_yr"] is None
    assert a["detection_limit_mm_yr"] is None and a["signal_to_noise"] is None
    assert a["reading"].startswith("Not testable") and "within" not in a["reading"].lower()


def test_page_names_untestable_areas_and_does_not_count_them_as_quiet(built):
    rep = json.loads(json.dumps(built.rep))
    html = cr.page(built.cfg, rep, _fake_figs(built))
    pit = _area(rep, "Wet Pit")
    assert pit["testable"] is False
    row = html[html.index('<th scope="row">Wet Pit'):]           # the table row (the DRAFT list names it first)
    row = row[:row.index("</tr>")]
    assert "Not testable" in row and "Within normal ground variation" not in row and "in/yr</span><br>" not in row.split("<td>")[3]
    assert "Wet Pit could not be tested" in html.replace("</b>", "").replace("<b>", "")


def test_thin_null_is_said_and_the_thousands_of_patches_claim_is_gone(built):
    rep = json.loads(json.dumps(built.rep))
    a = _area(rep, "Test Field")
    a.update({"null_thin": True, "approx_independent_placements": 7, "null_shape": "compact_square"})
    html = cr.page(built.cfg, rep, _fake_figs(built))
    assert "Only about 7 comparison areas" in html
    assert "thousands of patches" not in html and "same size and shape as each one reported" not in html


def test_no_farmland_band_means_features_could_not_be_assessed():
    rng = np.random.default_rng(0)
    ctx = _fake_ctx(rng.normal(0, 3, (80, 80)), far=np.zeros((80, 80), bool))
    sm = ri.smoothed_field(ctx, 0.78)
    assert sm["band_ok"] is False
    assert ri.find_features(ctx, sm, {}, np.ones((80, 80), bool)) is None
    obs = ri.observations({"features": [], "features_assessed": False, "features_reason": sm["reason"]})
    assert obs[0]["title"] == "Features could not be assessed"
    assert not any(o["title"] == "No feature stands out" for o in obs)


# ------------------------------------------------------------------ no GNSS (generality-1)
def test_unanchored_whole_area_is_relative_only_with_no_zero_width_range():
    cov = {"cells": 10000, "area_km2": 100.0, "footprint_km2": 100.0, "measured_percent": 99.0,
           "outside_frame_percent": 0.0, "coverage_balanced": True}
    summ = {"rate_mm_yr": 0.6, "total_mm": 6.0, "series_mm": list(np.linspace(0, 6, 50)), "start_mm": 0.0}
    a = cr.area_entry("Full Satellite Imaging", "full", "full", cov, summ, None, 0.78, None, None, 10.0,
                      np.linspace(0, 10, 50))
    assert a["relative_only"] is True and a["vertical_mm_yr"] is None and a["range95_mm_yr"] is None
    assert a["relative_vertical_mm_yr"] == round(0.6 / 0.78, 2)
    rep = {"areas": [a], "span_years": 10.0, "dates": ["20160105", "20260105"], "n_dates": 50, "anchored": False}
    row = cr.summary_rows(rep)[0]
    assert "GNSS" not in row[6] and row[1] == "relative only" and row[3] == "—"
    # and nothing can build a zero-width range from a zero sigma
    assert cr._range(1.0, 0.0) is None and cr._range(1.0, None) is None


def test_unanchored_notes_say_why():
    notes = ri.quality_notes({"checks": {}}, {"anchored": False,
                                            "gnss_reason": "gnss_tie.json covers None, not this record's dates"}, 0.78)
    assert any("No GNSS station could anchor" in n and "covers None" in n for n in notes)


# ------------------------------------------------------------------ coverage (methodology-4, gap-surface_mine_site-3)
def test_measured_share_of_each_outline_is_stated_and_gates_the_verdict(built):
    pit = _area(built.rep, "Wet Pit")
    fp_cells = (PIT_BOX[0].stop - PIT_BOX[0].start) * (PIT_BOX[1].stop - PIT_BOX[1].start)
    assert pit["measured_percent"] == pytest.approx(30.0, abs=1.5)
    assert pit["footprint_km2"] == pytest.approx(fp_cells * np.prod(sg.pixel_metres(built.ctx.attrs)) / 1e6, rel=0.01)
    assert pit["testable"] is False and "could be measured" in pit["not_testable_reason"]
    field = _area(built.rep, "Test Field")
    assert field["measured_percent"] == pytest.approx(100.0) and field["testable"] is True


def test_measured_cells_all_on_one_side_withhold_the_verdict():
    fp = np.zeros((40, 40), bool)
    fp[10:30, 10:30] = True
    m = fp.copy()
    m[10:30, 20:30] = False                      # 50%+ measured but the east half blank
    m[10:13, 20:30] = True
    cov = cr.coverage(m, fp, None, 0.01)
    assert cov["measured_percent"] >= 50 and cov["coverage_balanced"] is False
    tf = cr.tested_fields({"testable": True, "farmland_spread_1sigma_mm_yr": 1.0, "threshold_95_mm_yr": 2.0,
                           "detectable_80_mm_yr": 2.8, "excess_mm_yr": 0.1, "distinguishable_from_farmland": False},
                          0.8, cov)
    assert tf["testable"] is False and tf["detected"] is None


def test_outline_outside_the_satellite_view_is_reported_apart():
    fp = np.zeros((20, 20), bool)
    fp[5:15, 5:15] = True
    imaged = np.zeros((20, 20), bool)
    imaged[:, :10] = True
    cov = cr.coverage(fp & imaged, fp, imaged, 0.01)
    assert cov["outside_frame_percent"] == pytest.approx(50.0) and cov["measured_percent"] == pytest.approx(50.0)


def test_mine_ground_says_x_of_y_measured():
    rep = {"features": [], "mine_ground": [{"key": "surface", "label": "surface mines", "cells": 900,
                                            "area_km2": 12.3, "footprint_km2": 20.5, "relative_vertical_mm_yr": -1.0,
                                            "testable": True, "detected": False}]}
    txt = next(o["text"] for o in ri.observations(rep) if o["kind"] == "mines")
    assert txt.startswith("12.3 of 20.5 km² measured")


# ------------------------------------------------------------------ an unmeasurable area (stats-13)
def test_area_with_too_few_cells_is_shown_not_dropped_and_does_not_crash(built, tmp_path):
    rep = json.loads(json.dumps(built.rep))
    cov = {"cells": 3, "area_km2": 0.03, "footprint_km2": 0.4, "measured_percent": 7.5,
           "outside_frame_percent": 0.0, "coverage_balanced": True}
    rep["areas"].append(cr.area_entry("Pond", "1,2,3,4", "area", cov, None, None, 0.78, BG_V, BG_U, 9.2, built.ctx.t))
    cr.summary_table(rep, tmp_path / "t.png")                 # raised TypeError before
    assert (tmp_path / "t.png").exists()
    row = next(r for r in cr.summary_rows(rep) if r[0].startswith("Pond"))
    assert row[1] == "—" and row[6].startswith("Not measurable")
    html = cr.page(built.cfg, rep, _fake_figs(built))
    assert "Pond" in html and "Not testable" in html[html.index("Pond"):]


# ------------------------------------------------------------------ the whole area once (clientout-2)
def test_every_area_rate_is_its_series_slope_and_the_whole_area_adds_gnss(built):
    t = built.ctx.t
    for a in built.rep["areas"]:
        if a.get("vertical_mm_yr") is None:
            continue
        s = _series(a["series_vertical_mm"])
        ok = np.isfinite(s)
        assert np.polyfit(t[ok], s[ok], 1)[0] == pytest.approx(a["vertical_mm_yr"], abs=0.15)
    full = _area(built.rep, "Full Satellite Imaging")
    assert full["vertical_mm_yr"] == pytest.approx(full["relative_vertical_mm_yr"] + BG_V, abs=0.02)
    assert full["regional_gnss_vertical_mm_yr"] == BG_V and full["regional_gnss_uncertainty_mm_yr"] == BG_U
    assert full["range95_mm_yr"][1] - full["range95_mm_yr"][0] >= 2 * 1.96 * BG_U - 0.02
    assert "GNSS network" not in full["reading"]


# ------------------------------------------------------------------ per-cell uncertainty raster (stats-15)
def test_uncertainty_raster_uses_the_cell_null_not_the_repeatability(built):
    import rasterio
    out = cr.write_rasters(built.cfg, built.rep, built.ctx)
    p = next(x for x in out if "uncertainty" in x.name)
    with rasterio.open(p) as r:
        v = float(np.nanmedian(r.read(1)))
    assert built.rep["cell_null_by_class"] == {"4": 2.5}
    assert v == pytest.approx(np.hypot(2.5 / UP, BG_U) / 25.4, rel=0.01)      # repeatability (1.0) gave ~0.06


# ------------------------------------------------------------------ the map band (stats-7)
def test_map_band_is_centred_on_farmland_and_colours_each_side_equally():
    rng = np.random.default_rng(2)
    f = gaussian_filter(rng.normal(0, 1, (200, 200)), 3)
    f = f / f.std() * 4.0 + 3.0                          # farmland reads +3 mm/yr, not 0
    f = np.where(f < 3.0, 3.0 + 1.4 * (f - 3.0), f)      # and its sinking side is wider
    ctx = _fake_ctx(f, pixel=40.0)
    sm = ri.smoothed_field(ctx, 1.0)
    farv = sm["field"][ctx.far & np.isfinite(sm["field"])]
    red, blue = np.mean(farv <= sm["band_lo_mm_yr"]), np.mean(farv >= sm["band_hi_mm_yr"])
    assert 0.015 <= red <= 0.035 and 0.015 <= blue <= 0.035
    assert sm["centre_mm_yr"] == pytest.approx(np.median(farv), abs=0.3)
    # the zero-centred band (until 5 Oct 2026) coloured far more of one side
    old = 2 * 1.4826 * np.median(np.abs(farv - np.median(farv)))
    assert abs(np.mean(farv <= -old) - np.mean(farv >= old)) > 0.01


def test_map_band_ignores_real_movement_inside_the_farmland():
    rng = np.random.default_rng(4)
    f = gaussian_filter(rng.normal(0, 1, (200, 200)), 3)
    f = f / f.std() * 3.0
    f[20:60, 20:60] -= 25.0                              # unmapped subsidence, 4% of the farmland
    ctx = _fake_ctx(f, pixel=40.0)
    sm = ri.smoothed_field(ctx, 1.0)
    farv = sm["field"][ctx.far & np.isfinite(sm["field"])]
    assert sm["band_lo_mm_yr"] > np.quantile(farv, 0.025) + 2.0       # the raw quantile is set by the patch


# ------------------------------------------------------------------ corrections (timeseries-9)
def test_corrections_come_from_what_was_applied_not_the_decision(tmp_path):
    cfg = PipelineConfig(project_dir=tmp_path / "p", bbox=BBOX, start_date="2016-01-01", end_date="2025-03-31")
    cfg.mintpy_dir.mkdir(parents=True)
    (cfg.mintpy_dir / "corrections.json").write_text(json.dumps({"solid_earth_tides": True, "ionosphere": "gim"}))
    ts = cfg.mintpy_dir / "geo" / "geo_timeseries_ERA5_demErr.h5"
    _h5(ts, {"timeseries": np.zeros((2, 2, 2), np.float32)}, {})
    got = ri.corrections_applied(cfg, ts)
    assert not any("tides" in g or "ionosphere" in g for g in got)       # decided, never applied
    with h5py.File(ts, "a") as f:
        f.attrs["IONO_GIM_APPLIED"] = "jpl global ionosphere maps"
    assert any("ionosphere" in g for g in ri.corrections_applied(cfg, ts))
    (cfg.mintpy_dir / "run_products.json").write_text(json.dumps({"timeseries_geo": str(ts),
                                                                  "corrections_applied": ["ERA5", "SET", "demErr"]}))
    got = ri.corrections_applied(cfg, ts)
    assert any("tides" in g for g in got) and not any("ionosphere" in g for g in got)


# ------------------------------------------------------------------ trend words (methodology-6, stats-12)
def _year_list(t):
    return [2016 + int(x) for x in t]


def test_trend_drops_half_years_and_tests_the_pace_against_farmland():
    t = np.arange(0, 9.45, 12 / 365.25)                    # ends in June: a half year
    years = _year_list(t)
    steady = -20.0 * t + 15.0 * np.sin(2 * np.pi * t) + np.random.default_rng(0).normal(0, 2, t.size)
    null = {"years": sorted(set(years)), "rates": np.random.default_rng(1).normal(0, 4.0, (40, len(set(years))))}
    d = ri.trend_shape(t, steady, years, 2.0, null_yearly=null)
    assert max(years) not in d["yearly_mm_yr"] and max(years) in d["years_not_used"]
    assert d["shape"] == "steady" and "slowed" not in d["text"] and "sped up" not in d["text"]


def test_trend_untested_when_no_year_moves_the_records_way():
    """Sinking within every year, jumping up between them (unwrapping steps):
    the record's rate is upward and no year moves upward. The fastest-years
    contrast had no years and divided by zero (the first city site, 30 Oct 2026)."""
    t = np.arange(0, 6, 12 / 365.25)
    years = _year_list(t)
    s = -5.0 * (t - np.floor(t)) + 40.0 * np.floor(t)
    d = ri.trend_shape(t, s, years, 1.0)
    assert d["rate_mm_yr"] > 0 and all(v < 0 for v in d["yearly_mm_yr"].values())
    assert d["shape"] == "untested" and d["text"] == "" and "no single year" in d["untested_because"]


def test_trend_withholds_slowing_that_farmland_produces_by_itself():
    t = np.arange(0, 10, 12 / 365.25)
    years = _year_list(t)
    rate = np.where(t < 8, -20.0, -11.0)                    # recent 0.55 x the record's pace
    s = np.cumsum(rate * 12 / 365.25)
    noisy = {"years": sorted(set(years)), "rates": np.random.default_rng(3).normal(0, 15.0, (40, 10))}
    quiet = {"years": sorted(set(years)), "rates": np.random.default_rng(3).normal(0, 1.0, (40, 10))}
    assert ri.trend_shape(t, s, years, 1.0, null_yearly=noisy)["shape"] == "steady"
    d = ri.trend_shape(t, s, years, 1.0, null_yearly=quiet)
    assert d["shape"] == "slowing" and "slowed" in d["text"]
    assert "matches" in ri.cause_hint("surface", -1, d["shape"])
    assert "matches" not in ri.cause_hint("surface", -1, "steady")


def test_trend_without_any_null_gives_no_shape_word():
    t = np.arange(0, 10, 12 / 365.25)
    s = -20.0 * t
    d = ri.trend_shape(t, s, _year_list(t), 0.0)
    assert d["shape"] == "untested" and d["text"] == ""


def test_feature_trend_uses_its_own_farmland_placements(built):
    f = built.rep["features"][0]
    assert f["yearly_null_placements"] >= ri.MIN_YEARLY_NULL
    # this synthetic project has no land-cover map: its comparison ground is
    # "ground" (landcover.comparison_ground, 28 Oct 2026)
    assert f["trend"]["yearly_null"].endswith("ground patches of the same footprint")
    assert f["trend"]["shape"] == "slowing"


# ------------------------------------------------------------------ calibrated feature search (stats-6)
def _fake_ctx(field, far=None, pixel=60.0):
    ny, nx = field.shape
    dlat = pixel / 111320.0
    dlon = pixel / (111320.0 * np.cos(np.radians(38.3)))
    attrs = {"X_FIRST": "-87.6", "Y_FIRST": "38.4", "X_STEP": str(dlon), "Y_STEP": str(-dlat)}
    good = np.ones((ny, nx), bool)
    far = good.copy() if far is None else far
    return types.SimpleNamespace(sg=sg, attrs=attrs, good=good, far=far, far_wide=far, r_cm=field, r_null=field)


def _search(field, far=None):
    ctx = _fake_ctx(field, far)
    sm = ri.smoothed_field(ctx, 1.0)
    aoi = np.ones(field.shape, bool)
    cands = ri.find_features(ctx, sm, {}, aoi, limit=None)
    null = ri.feature_null(ctx, sm, cands or [], aoi)
    return [ri.selection_p(f, sm, null) | {"f": f} for f in cands or []], null


def test_feature_search_on_pure_farmland_rarely_finds_anything():
    hits = 0
    for seed in range(12):
        f = gaussian_filter(np.random.default_rng(100 + seed).normal(0, 1, (160, 160)), 5)
        res, null = _search(f / f.std() * 3.0)
        assert null["tested"]
        hits += any(r["stands_out"] for r in res)
    assert hits <= 2                                          # ~5% expected; the old gate had no measured rate at all


def test_feature_search_finds_a_real_bowl():
    """A 1 km bowl about four farmland band widths deep (the band here is
    ~+/-5.5; the review's own case: -40 mm/yr against Site B's ~10 mm/yr
    band). At 15 (2.6 band widths) the search itself missed it in 2 of 4
    seeds - noise of +2.5 sd at its centre left the peak short of STRONG -
    whatever the chance test said."""
    found = 0
    for seed in range(4):
        f = gaussian_filter(np.random.default_rng(200 + seed).normal(0, 1, (160, 160)), 5)
        f = f / f.std() * 3.0
        yy, xx = np.mgrid[:160, :160]
        f = f - 25.0 * np.exp(-((yy - 80) ** 2 + (xx - 80) ** 2) / (2 * 8.0 ** 2))
        res, _ = _search(f)
        found += any(r["stands_out"] and abs(r["f"]["lat"] - (38.4 - 80.5 * 60 / 111320)) < 0.01 for r in res)
    assert found == 4


def test_feature_search_fails_closed_without_enough_farmland():
    f = gaussian_filter(np.random.default_rng(5).normal(0, 1, (160, 160)), 5) * 10
    f[70:90, 70:90] -= 30.0
    far = np.zeros(f.shape, bool)
    far[:12, :] = True                                        # ~7 km2 of farmland
    ctx = _fake_ctx(f, far)
    sm = ri.smoothed_field(ctx, 1.0)
    null = ri.feature_null(ctx, sm, [], np.ones(f.shape, bool))
    assert null["tested"] is False and "farmland" in null["reason"]
    fake = {"sign": -1, "area_km2": 1.0, "beyond_farmland_mm_yr": 30.0}
    p = ri.selection_p(fake, sm, null)
    assert p["tested"] is False and p["stands_out"] is None
    txt = ri.feature_text(1, {"sign": -1, "area_km2": 1.0, "lat": 38.3, "lon": -87.5, "mean_rel_vertical_mm_yr": -30.0,
                              "mines": {"phrase": "away from every mapped mine"}, "chance": p,
                              "quality_standout": False, "cause_hint": ""})
    assert "could not be tested" in txt and "clears what chance" not in txt and "Groundwater" not in txt


def test_report_keeps_only_features_that_beat_chance(built):
    rep = built.rep
    assert rep["feature_search"]["tested"] is True
    for f in rep["features"]:
        assert f["stands_out"] is True and f["selection_p"] <= ri.SELECTION_ALPHA / 2   # G1: p <= 0.025
        assert f["ring"] and len(f["ring"]) >= 4 and f["bbox"]
        assert f["text"] == ri.feature_text(rep["features"].index(f) + 1, f)
    for w in rep["features_within_chance"]:
        assert w["selection_p"] > ri.SELECTION_ALPHA / 2
    assert "clears what chance alone produces" not in json.dumps(rep)


# ------------------------------------------------------------------ the contract's top-level blocks
def test_report_carries_route_release_gnss_and_cell_null(built):
    rep = built.rep
    assert rep["route"] == "official"
    assert rep["release"]["final"] is False and any(b["hard"] for b in rep["release"]["blockers"])
    assert rep["gnss"]["anchored"] is True and rep["gnss"]["reason"] is None
    assert rep["recent_window"][1] == rep["dates"][1]
    html = cr.page(built.cfg, rep, _fake_figs(built))
    assert "DRAFT - not for release" in html


def _fake_figs(built):
    p = Path(built.cfg.project_dir) / "deliverables" / "figures" / "blank.png"
    if not p.exists():
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(1, 1)).savefig(p)
        plt.close("all")
    return {"map": p, "series": p, "summary": p}


# ------------------------------------------------------------------ open requests on these files (numbers and data)
def _class_ctx(vel, far):
    """A _Context-like object with what the second farmland pass rebuilds."""
    ctx = _fake_ctx(vel, far=far)
    ctx.vel, ctx.tc = vel, np.full(vel.shape, 0.95)
    ctx.aoi = np.ones(vel.shape, bool)
    ctx.r_cm, ctx.class_means = sg.class_matched_field(ctx.vel, ctx.tc, ctx.far)
    ctx.r_null = ctx.r_place = ctx.r_cm
    ctx.class_tilt = {}
    return ctx


def test_second_pass_takes_found_movement_out_of_the_farmland_zero():
    """stats-3: unmapped sinking ground inside the farmland pulls the zero
    every 'relative to farmland' figure is measured from. Once the search
    has found it (it beat chance), the farmland is defined again without it."""
    rng = np.random.default_rng(11)
    vel = rng.normal(0.0, 1.0, (200, 200))
    patch = np.zeros(vel.shape, bool)
    patch[:, :70] = True                                   # 35% of the farmland sinks 10 mm/yr
    vel[patch] -= 10.0
    ctx = _class_ctx(vel, np.ones(vel.shape, bool))
    before = ctx.class_means[max(k for k in ctx.class_means if isinstance(k, int))]
    assert before < -0.35                                     # median centre still pulled down
    rec = cr._exclude_from_farmland(ctx, [patch], 500.0)
    after = ctx.class_means[max(k for k in ctx.class_means if isinstance(k, int))]
    assert abs(after) < 0.1 and not ctx.far[patch].any() and not ctx.far_wide[patch].any()
    clean = ~patch
    clean[:, :80] = False
    assert abs(float(np.median(ctx.r_cm[clean]))) < 0.1      # ordinary ground reads zero again
    assert rec["features_removed"] == 1 and rec["far_cells"][1] < rec["far_cells"][0]
    shift = list(rec["class_centre_shift_mm_yr"].values())[0]
    assert shift == pytest.approx(after - before, abs=0.01)


def test_report_runs_the_second_farmland_pass_on_found_features(built):
    sp = built.rep["farmland_second_pass"]
    assert sp and sp["features_removed"] >= 1 and sp["features_after"] >= 1
    assert not built.ctx.far[FEATURE].any()                  # the planted patch is no longer farmland


def test_corrections_say_n_of_m_era5_dates_and_name_the_new_tokens(tmp_path):
    cfg = PipelineConfig(project_dir=tmp_path / "p", bbox=BBOX, start_date="2016-01-01", end_date="2025-03-31")
    cfg.mintpy_dir.mkdir(parents=True)
    dates = _dates(20)
    ts = cfg.mintpy_dir / "geo" / "geo_timeseries_ERA5_tropHgt_demErr.h5"
    _h5(ts, {"timeseries": np.zeros((20, 2, 2), np.float32)}, {}, dates)
    (cfg.mintpy_dir / "run_products.json").write_text(json.dumps({
        "timeseries_geo": str(ts), "corrections_applied": ["ERA5", "tropHgt", "demErr"],
        "era5": {"dates_total": 20, "dates_corrected": 18, "missing_dates": dates[-2:]}}))
    got = ri.corrections_applied(cfg, ts)
    era = next(g for g in got if "ERA5" in g)
    assert "for 18 of 20 dates" in era and "most recent 2" in era
    assert any("height-correlated" in g for g in got)
    assert any("elevation model and today's ground" in g for g in got)
    # missing dates in mid-record are named, not called "the most recent"
    (cfg.mintpy_dir / "run_products.json").write_text(json.dumps({
        "timeseries_geo": str(ts), "corrections_applied": ["ERA5"],
        "era5": {"dates_total": 20, "dates_corrected": 19, "missing_dates": [dates[5]]}}))
    era = ri.corrections_applied(cfg, ts)[0]
    assert "for 19 of 20 dates" in era and dates[5] in era and "most recent" not in era


def test_disp_tides_and_ionosphere_claimed_only_when_measured_on_every_date(tmp_path):
    cfg = PipelineConfig(project_dir=tmp_path / "p", bbox=BBOX, start_date="2016-01-01", end_date="2025-03-31",
                         backend="disp")
    cfg.mintpy_dir.mkdir(parents=True)
    claim = "tides and ionosphere (OPERA product layers)"
    for status, want in (("all", True), ("partial", False), ("none", False), ("off", False)):
        (cfg.mintpy_dir / "corrections.json").write_text(json.dumps({
            "solid_earth_tides": status == "all", "ionosphere": "product layer",
            "disp_corrections": {"status": status, "epochs": 100, "epochs_corrected": 60}}))
        got = ri.corrections_applied(cfg, None)
        assert (claim in got) is want and not any("global ionosphere" in g for g in got)
    (cfg.mintpy_dir / "corrections.json").write_text(json.dumps({"solid_earth_tides": True, "ionosphere": "product layer"}))
    assert ri.corrections_applied(cfg, None) == []           # an older record: unknown, nothing claimed


def test_coverage_note_gives_both_numbers_and_never_says_usable():
    q = {"checks": {"coverage": {"aoi_percent_at_threshold": 97.0, "reliable_percent": 43.0}}}
    n = ri.quality_notes(q, {"anchored": True}, 0.78)[0]
    assert "97% of the study area returned a measurement" in n and "43% passed both quality checks" in n
    old = ri.quality_notes({"checks": {"coverage": {"aoi_percent_at_threshold": 97.0}}}, {"anchored": True}, 0.78)[0]
    assert "returned a measurement" in old and "usable" not in old and "passed both" not in old


def test_inputs_older_than_the_velocity_product_are_flagged(tmp_path):
    import os
    cfg = make_project(tmp_path)
    assert cr._stale_inputs(cfg) == []
    vel = cfg.mintpy_dir / "geo" / "geo_velocity.h5"
    t = vel.stat().st_mtime
    os.utime(cfg.project_dir / "deliverables" / "summary.json", (t - 3600, t - 3600))
    got = cr._stale_inputs(cfg)
    assert len(got) == 1 and "summary.json" in got[0] and "older than the velocity product" in got[0]


def test_standout_on_a_cell_corner_is_placed_like_mintpy_does():
    """gap-geolocation-9: int() put a coordinate that falls a hair short of a
    cell edge (floating point) on the cell before; lalo_to_rc adds 0.01 cell."""
    f = np.zeros((60, 60))
    f[30:40, 30:40] = -40.0                                   # a patch from row 30
    ctx = _fake_ctx(f, far=np.zeros((60, 60), bool))
    a = ctx.attrs
    lat = float(a["Y_FIRST"]) + 30 * float(a["Y_STEP"]) * (1 - 1e-12)   # row 30's top edge, a hair short
    lon = float(a["X_FIRST"]) + 35.5 * float(a["X_STEP"])
    assert int((lat - float(a["Y_FIRST"])) / float(a["Y_STEP"])) == 29     # what the old code computed
    sm = {"field": f, "centre_mm_yr": 0.0, "band_lo_mm_yr": -60.0, "band_hi_mm_yr": 60.0, "band_mm_yr": 60.0,
          "band_ok": True, "pixel_m": (60.0, 60.0)}
    sm["band_lo_mm_yr"] = -20.0                               # the patch is beyond the band but not STRONG
    got = ri.find_features(ctx, sm, {}, np.ones(f.shape, bool), {"lat": lat, "lon": lon}, limit=None)
    assert got and got[0]["quality_standout"] is True


def test_screening_project_keeps_saying_so_in_its_title():
    assert ri._pretty("site_d_disp_screening") == "Site D (screening)"
    assert ri._pretty("site_a_pl") == "Site A"


def test_reverse_geocode_uses_the_user_agent_set_now(tmp_path, monkeypatch):
    from insar_pipeline import areas
    seen = {}

    class R:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return json.dumps({"address": {"county": "Gibson County", "state": "Indiana", "country_code": "us"}}).encode()

    def fake_open(req, timeout=0):
        seen["ua"] = req.get_header("User-agent")
        return R()

    monkeypatch.setattr(areas, "user_agent", lambda: "insar-pipeline (Test Firm; test@example.com)")
    monkeypatch.setattr(ri.urllib.request, "urlopen", fake_open)
    assert ri.reverse_geocode(38.3, -87.5) == "Gibson County, Indiana"
    assert seen["ua"] == "insar-pipeline (Test Firm; test@example.com)"


def test_record_blocks_claim_a_reference_distance_only_when_measured(tmp_path):
    cfg = types.SimpleNamespace(project_dir=tmp_path, reference_auto=False, mintpy_dir=tmp_path / "mintpy",
                                cross_check_disp=True, injection_test=True)
    near = {"configured": None, "far_field_buffer_m": 2000.0,
            "chosen": {"lat": 38.3, "lon": -87.5, "distance_to_nearest_mapped_mine_km": 1.4}}
    b = cr._record_blocks(cfg, {}, {}, near)
    assert b["reference"]["auto"] is True and b["reference"]["clears_buffer"] is False
    far = dict(near, chosen=dict(near["chosen"], distance_to_nearest_mapped_mine_km=3.1))
    assert cr._record_blocks(cfg, {}, {}, far)["reference"]["clears_buffer"] is True
    old = {"configured": None, "chosen": {"lat": 38.3, "lon": -87.5, "rule": "kept"}}
    assert cr._record_blocks(cfg, {}, {}, old)["reference"]["clears_buffer"] is None
    assert b["injection_test"]["status"] == "not run" and b["independent_check"]["available"] is False
    T = tmp_path / "deliverables" / "technical"
    T.mkdir(parents=True)
    now = datetime.now().isoformat(timespec="seconds")      # only this run's records are read (7 Oct 2026)
    (T / "injection_recovery.json").write_text(json.dumps({"status": "complete", "sentence": "Found down to 1 km2.",
                                                           "decision_rule": "rule", "technical_summary": "Technical record",
                                                           "generated": now}))
    (T / "cross_check.json").write_text(json.dumps({"available": True, "status": "complete", "sentence": "Agrees.",
                                                    "generated": now,
                                                    "areas": [{"name": "A", "agree": True, "combined_95_mm_yr": 3.0}]}))
    b = cr._record_blocks(cfg, {}, {}, {})
    # E4 / C5 (trust review 7 Oct 2026): technical only - an older record's
    # sentence is not carried; the rule and the technical summary are
    assert b["injection_test"]["sentence"] is None and b["injection_test"]["status"] == "complete"
    assert b["injection_test"]["decision_rule"] == "rule" and b["injection_test"]["technical_summary"] == "Technical record"
    assert b["independent_check"]["areas"][0]["agree"] is True


def test_send_rule_is_re_evaluated_with_the_tie_only_on_a_qa_that_has_one():
    qa = {"checks": {"coverage": {"aoi_percent_at_threshold": 97.0, "pass": True}}, "findings": {}}
    old = cr._send_rule(qa, {})
    assert old["value"] is None and "predates" in old["reasons"][0]          # never a verdict from checks never run
    qa["send_rule"] = {"value": "send", "reasons": []}
    tied = cr._send_rule(qa, {"background": {"vertical_mm_yr": -1.0, "vertical_uncertainty_mm_yr": 0.5}})
    untied = cr._send_rule(qa, {"background": None, "reason": "no station within 40 km"})
    assert tied["value"] != "hold for specialist" and tied["gnss_considered"] is True
    assert untied["value"] == "hold for specialist" and "no station within 40 km" in untied["reasons"][0]


def test_report_carries_the_send_rule_re_evaluated_with_the_tie_and_no_stale_inputs(built):
    rep = built.rep
    assert rep["send_rule"]["value"] is None                   # the synthetic QA stored no send rule
    assert rep["stale_inputs"] == [] and rep["los_up_median"] == pytest.approx(UP, abs=1e-3)
    assert all("resolution_caveat" in a for a in rep["areas"])


def test_gnss_station_check_uses_the_one_rule(built):
    rep = json.loads(json.dumps(built.rep))
    rep["gnss"]["inside_checks"] = [{"site": "TEST", "insar_relative_class_matched_mm_yr": 3.0,
                                     "farmland_spread_1sigma_mm_yr": 1.0, "threshold_95_mm_yr": 4.0,
                                     "distinguishable_from_farmland": False}]
    html = cr.page(built.cfg, rep, _fake_figs(built))
    assert "moving with the surrounding ground" in html          # no land-cover map: "ground"          # 1.96 sd (1.96 mm/yr) said "outside the normal range"
    rep["gnss"]["inside_checks"] = [{"site": "TEST", "insar_relative_class_matched_mm_yr": None}]
    assert "TEST" not in cr.page(built.cfg, rep, _fake_figs(built)).split("Checked against GNSS")[1].split("</section>")[0]



# ======================================================================== lane L5b
def _page(built, rep=None, cfg=None):
    return cr.page(cfg or built.cfg, rep if rep is not None else json.loads(json.dumps(built.rep)), _fake_figs(built))


def _text(html):
    import re
    html = re.sub(r"<style>.*?</style>", " ", html, flags=re.S)
    return " ".join(re.sub(r"<[^>]+>", " ", html).split())


def _capture_figures(monkeypatch):
    """Every matplotlib figure saved while the patch is on, kept for inspection."""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.figure import Figure
    seen = []
    orig = Figure.savefig

    def keep(self, *a, **k):
        seen.append(self)
        return orig(self, *a, **k)

    monkeypatch.setattr(Figure, "savefig", keep)
    return seen


def _no_network(monkeypatch):
    def refuse(*a, **k):
        raise OSError("offline (test)")
    monkeypatch.setattr(ri.urllib.request, "urlopen", refuse)


# ------------------------------------------------------------------ chart bands (clientout-6, stats-8)
def test_series_band_holds_quiet_ground_and_matches_the_reading():
    """A quiet area's line (per-date noise + season, no movement) stays inside
    the outer band on ~95% of dates; the old rate cone (zero wide at the
    first date) put most first-year dates outside. The straight line ends
    outside the inner band exactly when |excess| passes the threshold."""
    rng = np.random.default_rng(3)
    t = np.linspace(0, 10.6, 270)
    inside_new, first_old = [], []
    for _ in range(20):
        y = rng.normal(0, 5.0, t.size) + 4.0 * np.sin(2 * np.pi * t)
        y -= np.polyval(np.polyfit(t, y, 1), t[0])             # zeroed like the report
        noise = cr._scatter_about_trend(y, t)
        b = cr.series_band(t, 0.0, 1.96 * 0.6, noise)
        inside_new.append(np.mean((y >= b["outer"][0]) & (y <= b["outer"][1])))
        old = np.abs(y) <= 1.96 * 0.6 * t
        first_old.append(np.mean(old[t < 1.0]))
    assert np.mean(inside_new) >= 0.93 and min(inside_new) >= 0.88
    assert np.mean(first_old) < 0.5                              # the old cone failed the first year
    # the decision, by construction: rate v against centre c and threshold thr
    for v in np.linspace(-12, 12, 49):
        for thr in (2.0, 5.0):
            c = 0.7
            b = cr.series_band(t, c, thr, 3.0)
            end = v * t[-1]
            outside = end < b["inner"][0][-1] or end > b["inner"][1][-1]
            assert outside == (abs(v - c) > thr)
    assert cr.series_band(t, 0.0, None, 3.0)["outer"] is None   # no comparison: no band, never zero-width


def test_report_area_line_stays_in_its_band_and_caption_matches(built):
    rep = built.rep
    items = {it["name"]: it for it in cr.series_items(rep)}
    it = items["Test Field"]
    a = _area(rep, "Test Field")
    assert a["series_noise_mm"] and a["series_noise_basis"] and it["decision"]
    t = built.ctx.t
    y = _series(it["series_vertical_mm"])
    b = cr.series_band(t, it["centre_rate"], it["cone_rate"], it["noise_mm"])
    ok = np.isfinite(y)
    assert np.mean(((y >= b["outer"][0]) & (y <= b["outer"][1]))[ok]) >= 0.9
    end_outside = not (b["inner"][0][-1] <= it["rate"] * t[-1] <= b["inner"][1][-1])
    assert end_outside == bool(a["detected"])
    assert "Wet Pit" in items and items["Wet Pit"]["cone_rate"] is None     # not testable: no band
    html = _page(built)
    assert "a line that leaves the band is a detection" not in html
    assert "stands out only if its straight line ends outside it" in html and "19 images in 20" in html


# ------------------------------------------------------------------ the map (clientout-5, generality-4, clientout-12)
def test_map_scale_comes_from_the_data_and_separates_chance_from_features():
    rng = np.random.default_rng(5)
    f = rng.normal(0.0, 4.0, (200, 200))
    f[20:40, 20:40] = -14.0                                      # a chance-level patch
    f[100:130, 100:130] = -40.0                                  # a real feature
    f[150:160, 150:160] = -75.0                                  # its core, ~3 in/yr
    ms = ri.map_scale(f, np.ones(f.shape, bool), -10.2, 10.2)
    e = np.array(ms["edges_mm_yr"])
    assert np.all(np.diff(e) > 0) and -10.2 in e and 10.2 in e
    assert ms["top_in_per_yr"] * MM_IN >= 70 and ms["top_in_per_yr"] != 0.6
    assert np.digitize(-14.0, e) != np.digitize(-40.0, e)        # different colours (the fixed scale: both saturated)
    assert ms["n_below"] == sum(1 for x in e if x < -10.2)
    flat = np.where(np.abs(f) > 20, 0.0, f)                       # nothing but farmland
    assert ri.map_scale(flat, np.ones(f.shape, bool), -10.2, 10.2)["top_in_per_yr"] <= 1.0


MM_IN = 25.4


def test_map_patches_inside_report_areas_are_named():
    f = np.zeros((100, 100))
    f[10:30, 10:30] = -14.0
    f[60:80, 60:80] = -40.0
    area = np.zeros(f.shape, bool)
    area[0:90, 0:90] = True
    attrs = {"X_FIRST": "-87.6", "Y_FIRST": "38.4", "X_STEP": "0.001", "Y_STEP": "-0.001"}
    lon = lambda c: -87.6 + (c + 0.5) * 0.001                        # noqa: E731
    lat = lambda r: 38.4 + (r + 0.5) * -0.001                        # noqa: E731
    feat = {"sign": -1, "bbox": [lon(60), lat(79), lon(79), lat(60)], "lat": lat(70), "lon": lon(70)}
    got = ri.map_patches(f, -10.0, 10.0, {"Zone": area}, [feat], 0.01, attrs)
    kinds = {round(p["mean_rel_vertical_mm_yr"]): p["feature"] for p in got}
    assert kinds == {-40: 1, -14: None} and all(p["area"] == "Zone" for p in got)


def test_client_map_has_scale_bar_north_arrow_full_latitudes_and_extend_arrows(built, tmp_path, monkeypatch):
    import re
    seen = _capture_figures(monkeypatch)
    _no_network(monkeypatch)
    cfg = built.cfg
    cfg.report_basemap = "usgs"                                   # requested, but offline: drawn without it
    figs = cr.figures(cfg, built.rep, built.ctx, built.masks, built.outlines, out_dir=tmp_path,
                      cache_dir=tmp_path / "cache")
    assert figs["basemap_credit"] is None and (tmp_path / "client_map.png").stat().st_size > 10000
    m = seen[0]
    ax = m.axes[0]
    texts = [t.get_text() for t in ax.texts]
    assert "N" in texts and any(t.endswith(" km") for t in texts) and any(t.endswith(" mi") for t in texts)
    m.canvas.draw()
    labs = [t.get_text() for t in ax.get_yticklabels() if t.get_text()]
    assert labs and all(re.fullmatch(r"\d+\.\d{2}°N", x) for x in labs)       # never '25', '00'
    cb = m.axes[1]
    assert "beyond" in cb.get_ylabel()
    assert ax.get_legend() is not None and any("not measured" in t.get_text() for t in ax.get_legend().get_texts())
    stamp = " ".join(t.get_text() for t in m.texts)
    assert "Copernicus Sentinel" in stamp and "insar_pipeline" in stamp and "DRAFT" in stamp


def test_basemap_uses_cached_tiles_and_never_raises_offline(tmp_path, monkeypatch):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _no_network(monkeypatch)
    bbox = (-87.6, 38.3, -87.45, 38.4)
    assert ri.basemap(bbox, [tmp_path], tmp_path) is None          # offline, nothing cached
    w, s_, e, n = bbox
    z = int(np.clip(round(np.log2(1400 / 256.0 * 360.0 / (e - w))), 6, 15))
    x0, y0 = ri._tile_xy(n, w, z)
    x1, y1 = ri._tile_xy(s_, e, z)
    for tx in range(int(x0), int(x1) + 1):
        for ty in range(int(y0), int(y1) + 1):
            p = tmp_path / "usgs_topo" / str(z) / f"{tx}_{ty}.jpg"
            p.parent.mkdir(parents=True, exist_ok=True)
            plt.imsave(p, np.full((256, 256, 3), 0.8), format="jpg")
    got = ri.basemap(bbox, [tmp_path], None, fetch=False)
    assert got and got["fetched"] == 0 and got["image"].ndim == 3 and got["extent"] == (w, e, s_, n)
    assert "public domain" in got["credit"]


# ------------------------------------------------------------------ posting, resolution, points (usability-4, clientout-7, empirical-7, tests-6)
def test_page_states_posting_resolution_and_study_area_points_from_the_run(built):
    rep = json.loads(json.dumps(built.rep))
    t = _text(_page(built, rep))
    res = rep["resolution"]
    assert f"Points are {res['posting_m']:.0f} m apart" in t and "each value combines similar ground within about" in t
    full = _area(rep, "Full Satellite Imaging")
    assert f"{full['cells']:,} measurement points across the study area" in t
    assert "a few millimetres across" not in t and "every 15 m cell" not in t and "about 40 m apart" not in t
    # E25 (trust review 7 Oct 2026): radar resolution cells, not independent measurements
    assert "radar resolution cells" in t and "independent radar measurements" not in t
    # another route, another grid: nothing typed for Site A survives
    rep["resolution"] = dict(res, posting_m=30.0, look_cell_m=[30.0, 30.0], support_m=[200.0, 200.0],
                             text="Points are 30 m apart; each value combines similar ground found within up to about 200 m (OPERA's processing).")
    t2 = _text(_page(built, rep))
    assert "30 m apart" in t2 and "15 m" not in t2 and "15-by-15" not in t2
    lines = dict(cr.one_pager_lines(rep, _area(rep, "Test Field")))
    # E9 (trust review 7 Oct 2026): 2 decimals of an inch
    assert lines["Vertical movement per year"].startswith(f"{_area(rep, 'Test Field')['vertical_mm_yr'] / MM_IN:+.2f} in/yr")


# ------------------------------------------------------------------ calibrated method wording (methodology-9)
SITE_B_DS = {"pass": False, "velocity_by_coherence_class": [
    {"tc": "0.4-0.6", "n": 1051816, "median_mm_yr": -9.57}, {"tc": "0.6-0.7", "n": 59833, "median_mm_yr": -4.88},
    {"tc": "0.7-0.8", "n": 14546, "median_mm_yr": -0.66}, {"tc": "0.8-0.9", "n": 2878, "median_mm_yr": 0.35},
    {"tc": "0.9-1.0", "n": 967, "median_mm_yr": 4.74}], "low_minus_high_coherence_mm_yr": -14.31}


def test_ds_bias_note_gives_its_size_and_the_assumption():
    import re
    n = ri.quality_notes({"checks": {"distributed_scatterer_bias": SITE_B_DS}}, {"anchored": True}, 0.772)
    note = next(x for x in n if "best points" in x)
    v = float(re.search(r"about (\d+\.\d+) in/yr", note).group(1))
    assert 0.3 <= v <= 0.6                                         # 0.45: dominant class vs the weighted >=0.7 classes
    assert "slightly" not in note and "removes it" not in note and "as long as the effect is the same" in note
    small = ri.quality_notes({"checks": {"distributed_scatterer_bias": SITE_B_DS}},
                             {"anchored": True, "detection_limits_mm_yr": [40.0]}, 0.772)
    assert "slightly" in next(x for x in small if "best points" in x)   # only when far under the limits


def test_method_and_feature_count_are_calibrated(built):
    rep = json.loads(json.dumps(built.rep))
    t = _text(_page(built, rep))
    assert "every image at once" not in t and "keeps farmland measurable across the seasons" not in t
    assert "overlapping windows of about 5 months (13 images" in t
    f0 = rep["features"][0]
    # E7 / C4 (trust review 7 Oct 2026): 'clearly' only from the feature's own
    # stands_out_clearly; a QA stand-out alone no longer gives it
    rep["features"] = [dict(f0, quality_standout=True, stands_out=True, stands_out_clearly=True),
                       dict(f0, quality_standout=False, stands_out=True, stands_out_clearly=False),
                       dict(f0, quality_standout=True, stands_out=True)]
    t = _text(_page(built, rep))
    assert "1 local feature stands out clearly" in t and "2 more go beyond it but are worth a look" in t
    assert "3 local features stand out" not in t
    assert "overlapdatum" not in t and "Phase linking: MiaplPy" in rep["technical_methods"]   # technical, not client


# ------------------------------------------------------------------ direction words and the regional figure (clientout-10, stats-9, methodology-5, empirical-5)
def _with_gnss(rep, v, u, **bg):
    rep = json.loads(json.dumps(rep))
    rep["gnss"]["background"] = dict(rep["gnss"]["background"], vertical_mm_yr=v, vertical_uncertainty_mm_yr=u, **bg)
    return rep


def test_regional_rate_is_gnss_with_a_direction_only_when_its_range_excludes_zero(built):
    r = cr.regional_gnss(_with_gnss(built.rep, -1.85, 1.28, n_sites=3, uncertainty_basis="scatter between the sites used"))
    assert "sinking" not in r["sentence"] and "cannot be told apart from no movement" in r["sentence"]
    assert "The 3 GNSS stations used" in r["sentence"] and "scatter between the stations used" in r["sentence"]
    r = cr.regional_gnss(_with_gnss(built.rep, -3.0, 0.5))
    assert "the region is sinking" in r["sentence"]
    rep = _with_gnss(built.rep, -1.85, 0.31)
    rep["gnss"]["leave_one_out"] = {"without": {"VINU": -1.35, "INVI": -2.0}, "max_change_mm_yr": 0.5}
    assert "without VINU it would be -1.4 mm/yr" in cr.regional_gnss(rep)["sentence"]
    t = _text(_page(built, rep))
    assert "also show" not in t and "as GNSS says it does" not in t
    assert "on the assumption that the surrounding ground moves with the stations' monuments" in t


def test_area_direction_words_need_a_range_that_excludes_zero(built):
    rep = json.loads(json.dumps(built.rep))
    a = _area(rep, "Test Field")
    a.update({"vertical_mm_yr": -2.0, "range95_mm_yr": [-5.0, 1.0], "total_mm": -20.0})
    html = _page(built, rep)
    row = html[html.index('<th scope="row">Test Field'):]
    row = row[:row.index("</tr>")]
    # E5 (trust review 7 Oct 2026): the detection limit is said for sinking
    # ("Sustained sinking of ... would be detected"); that is not a direction
    # word for the area
    row = row.replace("Sustained sinking of", "")
    assert "sinking" not in row and "-2.0 mm/yr" in row
    a["range95_mm_yr"] = [-5.0, -0.5]
    row = _page(built, rep)
    row = row[row.index('<th scope="row">Test Field'):]
    assert "sinking" in row[:row.index("</tr>")]
    assert ri.direction(-2.0, None) == "" and ri.direction(2.0, [0.5, 3.5]) == "rising"


def test_station_check_states_what_it_could_find_and_the_shape(built):
    rep = json.loads(json.dumps(built.rep))
    rep["gnss"]["background"]["los_mm_yr"] = -0.86
    chk = {"site": "LAMC", "insar_relative_class_matched_mm_yr": 0.27, "farmland_spread_1sigma_mm_yr": 6.37,
           "gnss_los_rate_over_overlap_mm_yr": -0.91, "detrended_correlation": -0.12, "overlap": [2016.01, 2018.21]}
    got = cr.station_check(chk, rep)
    t = _text(got["html"])
    assert got["agrees"] is True
    lim = 1.96 * 6.37 / rep["gnss"]["los_up_median"] / 25.4
    assert f"could only reveal a difference of about {lim:.2f} in/yr or more" in t
    assert "do not match (correlation -0.12)" in t and "-0.00" not in t


# ------------------------------------------------------------------ reference wording (empirical-12, tests-2)
def test_reference_wording_follows_what_was_measured(built):
    rep = json.loads(json.dumps(built.rep))
    rep["reference"] = {"auto": True, "distance_to_nearest_mapped_mine_km": 1.4, "far_field_buffer_m": 2000.0,
                        "clears_buffer": False}
    t = _text(_page(built, rep))
    assert "stable reference point" not in t and "km from the nearest mapped mine" not in t
    assert "more than 2 km from any mapped mine" in t
    rep["reference"].update(distance_to_nearest_mapped_mine_km=3.1, clears_buffer=True, far_field_buffer_m=1500.0)
    t = _text(_page(built, rep))
    assert "about 3.1 km from the nearest mapped mine" in t and "more than 1.5 km from any mapped mine" in t
    q = {"checks": {"reference_stability": {"pass": False, "value_mm_yr": -3.11, "yardstick_mm_yr": 1.0}}}
    note = ri.quality_notes(q, {"anchored": True, "detection_limits_mm_yr": [3.6]}, 0.772)[0]
    assert "a little" not in note and "-4.0 mm/yr" in note and "more than the smallest movement" in note


# ------------------------------------------------------------------ line of sight (methodology-15, stats-16)
def test_los_limitation_gives_the_east_west_factor_and_the_mining_error(built):
    rep = json.loads(json.dumps(built.rep))
    rep.update(los_east_over_up=0.81, mine_layers=["mine areas"])
    lim = " ".join(cr.limitations(built.cfg, rep))
    assert "east-west ground movement of 1 in/yr reads as about 0.8 in/yr" in lim
    assert "±28%" in lim and "second satellite track" in lim
    f = dict(rep["features"][0], los_relative_mm_yr=-13.0,
             mines={"key": "near_underground", "inside": {"underground": 0.1}, "phrase": "next to mapped workings"})
    assert "measured 13 mm/yr along its line of sight" in ri.feature_text(1, f)


def test_client_rasters_carry_the_vertical_assumption(built):
    import rasterio
    rep = dict(built.rep, los_east_over_up=0.81)
    out = cr.write_rasters(built.cfg, rep, built.ctx)
    with rasterio.open(out[0]) as src:
        desc = src.tags().get("TIFFTAG_IMAGEDESCRIPTION") or (src.descriptions[0] or "")
    assert "ASSUMES ALL MOVEMENT IS VERTICAL" in desc and "0.81" in desc


# ------------------------------------------------------------------ wording, names, hemispheres (clientout-18, generality-16, empirical-13)
def test_names_hemispheres_and_comparison_ground_come_from_the_project(built):
    assert ri.display_name("project_zone") == "Project Zone" and ri.display_name("Area A1") == "Area A1"
    assert ri.lalo_txt(-12.1, 45.2) == "12.100 S, 45.200 E" and ri.lalo_txt(38.3, -87.4) == "38.300 N, 87.400 W"
    rep = json.loads(json.dumps(built.rep))
    rep["control_ground_label"] = "pasture"
    a = _area(rep, "Test Field")
    a["display_name"] = "Test Field (display)"
    t = _text(_page(built, rep))
    assert "ordinary pasture" in t and "Test Field (display)" in t
    assert "Full Satellite Imaging" not in str(rep.get("full_area_name")) or rep["full_area_name"] == "Full Satellite Imaging"


def test_single_area_answer_and_chart_title_read_naturally(built):
    rep = json.loads(json.dumps(built.rep))
    rep["areas"] = [a for a in rep["areas"] if a["name"] != "Wet Pit"]
    t = _text(_page(built, rep))
    assert "Test Field, the report area that could be tested, does not move differently" in t
    assert "over the smallest" not in t and "none of the report area " not in t
    assert cr.totals_differ(-84.1, -76.7) and not cr.totals_differ(-14.3, -15.0)


# ------------------------------------------------------------------ surface mines (gap-surface_mine_site-7)
def test_other_layer_and_surface_mine_wording(tmp_path):
    gj = {"type": "FeatureCollection", "features": [{"type": "Feature", "properties": {}, "geometry": {
        "type": "Polygon", "coordinates": [[[-87.6, 38.3], [-87.5, 38.3], [-87.5, 38.4], [-87.6, 38.3]]]}}]}
    for n in ("surface", "other"):
        (tmp_path / f"{n}.geojson").write_text(json.dumps(gj))
    cfg = types.SimpleNamespace(surface_mines_geojson=str(tmp_path / "surface.geojson"),
                                other_movement_geojson=str(tmp_path / "other.geojson"), underground_mines_geojson="",
                                deramp_exclude_geojson="",
                                mine_layer_sources=("Surface mines: IGWS Coal_Mines_Surface_Dec2016",
                                                    "Other known ground movement: dnr.geojson, file dated 2023-03-28"))
    L = {x["key"]: x for x in ri.mine_layers(cfg)}
    assert set(L) == {"surface", "other"}
    assert L["surface"]["vintage"] == 2016 and L["other"]["vintage"] is None     # a file date is not a vintage
    f = {"sign": 1, "area_km2": 1.0, "lat": 38.3, "lon": -87.5, "mean_rel_vertical_mm_yr": 20.0,
         "vertical_relative_mm_yr": 20.0, "mines": {"key": "surface", "inside": {"surface": 0.6}, "phrase": "on mapped surface-mine ground"},
         "cause_hint": ri.cause_hint("surface", 1, None), "chance": {"tested": True, "p": 0.01, "copies": 59, "copies_as_strong": 0}}
    assert ri.feature_kind(f) == "change on surface-mine ground"
    assert "cannot follow ground built up by metres" in f["cause_hint"]
    assert "Rising" in ri.cause_hint("surface", 1, None)
    txt = ri.feature_text(1, dict(f, ground_reshaped=True))
    assert "no rate is given" in txt and "in/yr (20 mm/yr) up" not in txt
    assert "in/yr (20 mm/yr) up" in ri.feature_text(1, f)              # slow heave on coherent ground stays a measurement


def test_reshaped_gate_reads_the_dem_error_against_ordinary_ground():
    rng = np.random.default_rng(2)
    dh = rng.normal(0, 2.0, (100, 100)).astype(np.float32)
    far = np.ones(dh.shape, bool)
    pit = np.zeros(dh.shape, bool)
    pit[10:20, 10:20] = True
    dh[pit] = 12.0
    far[pit] = False
    other = np.zeros(dh.shape, bool)
    other[50:60, 50:60] = True
    assert cr._reshaped(dh, pit, far) is True and cr._reshaped(dh, other, far) is False
    assert cr._reshaped(None, pit, far) is None


# ------------------------------------------------------------------ release state, footer, offline page (clientout-9, gap-licensing-11, methodology-10, disp-1)
def test_page_is_self_contained_with_footer_and_draft_until_final(built, monkeypatch):
    from insar_pipeline import settings
    monkeypatch.setattr(settings, "identity", lambda: {"firm": "Test Firm", "preparer": "A. Engineer", "contact": "a@x.com"})
    rep = json.loads(json.dumps(built.rep))
    html = _page(built, rep)
    assert html.startswith("<!doctype html>") and '<meta charset="utf-8">' in html and 'name="viewport"' in html
    assert "googleapis" not in html and str(built.cfg.project_dir) not in html
    assert "DRAFT - not for release" in html and 'content:"DRAFT"' in html
    t = _text(html)
    assert "Prepared by A. Engineer, Test Firm (a@x.com)" in t and "insar_pipeline" in t and "not a survey" in t
    assert "overall:" not in t and "Data sources and credits" in t and "Contains modified Copernicus Sentinel" in t
    rep["release"] = {"final": True, "blockers": [], "reviewed_by": "J. Smith", "review_date": "2026-10-06T10:00"}
    html = _page(built, rep)
    assert "DRAFT" not in html                                    # package.build refuses a final report containing it
    assert "Reviewed by J. Smith on 2026-10-06" in _text(html)


def test_screening_route_says_so_everywhere(built):
    rep = json.loads(json.dumps(built.rep))
    rep.update(route="screening", backend="disp")
    t = _text(_page(built, rep))
    assert "Screening - not the full processing" in t and "ground movement report (screening)" in t
    assert "SCREENING ONLY" in t and "same comparison and checks as a full processing" not in t
    assert "route: screening" in t


def test_credits_do_not_name_opera_when_no_cross_check_ran(built):
    from insar_pipeline import credits
    rep = json.loads(json.dumps(built.rep))
    assert rep["independent_check"]["available"] is False
    assert not any("OPERA" in c["source"] for c in rep["credits"])
    # credits itself reads the block (INT2: the fix moved from the report's
    # workaround into credits.opera_used, so every caller gets it)
    assert not any("OPERA" in c["source"] for c in credits.credits(built.cfg, rep))
    ran = dict(rep, independent_check={"available": True, "status": "complete"})
    assert credits.opera_used(built.cfg, ran) is True


def test_record_dates_and_qa_sentences_reach_the_page(built):
    rep = json.loads(json.dumps(built.rep))
    rep["record_dates"] = {"available": 280, "processed": 270, "dropped": {"20200101": "bursts missing from the archive",
                                                                          "20200113": "bursts missing from the archive"}}
    rep["qa_flags"] = {"dem_error": True, "plausibility": True}
    rep["geolocation"] = {"text": "Map positions agree with mapped water edges to within 15 m."}
    rep["injection_test"] = {"status": "complete", "sentence": "Planted bowls of 1 km were found down to 10 mm/yr."}
    t = _text(_page(built, rep))
    assert "270 of 280 available images were used; 2 left out because bursts missing from the archive" in t
    assert "Map positions agree with mapped water edges" in t and "quality_too_fast_to_measure.tif" in t
    assert "elevation model (made before the newest pits and spoil)" in t
    # E4 / C5 (trust review 7 Oct 2026): the injection test is technical only,
    # even when an older record carries a sentence
    assert "Planted bowls of 1 km were found" not in t and "Planting known movement" not in t


# ------------------------------------------------------------------ mine layer sources (gap-licensing-5, clientout-13)
def test_mine_layer_source_and_vintage_are_printed_with_their_limitation(built):
    rep = json.loads(json.dumps(built.rep))
    rep["mine_layers"] = ["underground workings"]
    rep["mine_layer_info"] = [{"key": "underground", "label": "underground workings",
                               "source": "Indiana Geological & Water Survey, Coal_Mines_Underground_Dec2016", "vintage": 2016}]
    t = _text(_page(built, rep))
    assert "underground workings - Indiana Geological &amp; Water Survey" in t or \
           "underground workings - Indiana Geological & Water Survey" in t
    assert f"are mapped to 2016; mining or reclamation after that is not shown, while the record runs to {rep['dates'][1][:4]}" in t
    rep["mine_layer_info"][0].update(source=None, vintage=None)
    t = _text(_page(built, rep))
    assert "source and date not recorded" in t and "when they were mapped is not recorded" in t


# ------------------------------------------------------------------ presenter outputs (clientout-16)
def test_area_pdf_has_one_page_per_area_with_the_report_numbers(built, tmp_path):
    import re
    p = cr.area_pdf(built.cfg, built.rep, built.ctx, built.outlines, tmp_path / "areas.pdf")
    raw = p.read_bytes()
    assert raw.startswith(b"%PDF") and len(re.findall(rb"/Type /Page\b(?!s)", raw)) == len(built.rep["areas"])
    assert b"DRAFT" in raw                                           # metadata + watermark until final
    rep = built.rep
    a = _area(rep, "Test Field")
    lines = dict(cr.one_pager_lines(rep, a))
    # E9 (trust review 7 Oct 2026): in/yr to 2 decimals, no more precise than the data
    assert lines["Vertical movement per year"].startswith(f"{a['vertical_mm_yr'] / 25.4:+.2f} in/yr ({a['vertical_mm_yr']:+.1f} mm/yr)")
    lo, hi = a["range95_mm_yr"]
    assert lines["Likely range (95%)"].startswith(f"{lo / 25.4:+.2f} to {hi / 25.4:+.2f} in/yr")
    # E5 (trust review 7 Oct 2026): the stated limit is the sinking one
    assert a["detection_limit_basis"] == "sinking"
    assert lines["Reliably detects sinking (4 times in 5)"] == f"{a['detection_limit_mm_yr'] / 25.4:.2f} in/yr ({a['detection_limit_mm_yr']:.0f} mm/yr)"
    assert lines["Reading"] == a["reading"]
    rk = next(k for k in lines if k.startswith("Recent rate"))
    assert lines[rk].startswith(f"{a['recent_rate_mm_yr'] / 25.4:+.2f} in/yr")


def test_page_has_glossary_and_each_area_recent_rate(built):
    t = _text(_page(built))
    a = _area(built.rep, "Test Field")
    rw = a["recent_window"]
    assert f"Over {rw[0][:4]}–{rw[1][:4]}: {a['recent_rate_mm_yr'] / 25.4:+.2f} in/yr" in t
    assert "Terms used" in t and "Posting and resolution" in t and "Line of sight" in t


def test_build_writes_page_json_figures_and_pdf_offline(tmp_path, monkeypatch):
    _no_network(monkeypatch)
    cfg = make_project(tmp_path)
    out = cr.build(cfg, out_dir=tmp_path / "preview")
    P = tmp_path / "preview"
    assert out.exists() and (P / "client_report.json").exists() and (P / cr.PDF_NAME).exists()
    assert {p.name for p in (P / "figures").glob("client_*.png")} == {"client_map.png", "client_series.png", "client_summary.png"}
    assert not (cfg.project_dir / "context" / "basemap").exists()   # a preview writes nothing into the project


def test_rebuild_without_changes_keeps_the_numbers_a_sign_off_is_bound_to(tmp_path, monkeypatch):
    """Everything L5b adds to client_report.json (map scale, patches, chart
    noise, layer info, technical paragraph) is measured, not stamped: two
    builds of the same products give the same release.numbers_hash, so a
    sign-off is not voided by rebuilding - and the basemap (online or not)
    never enters the numbers."""
    from insar_pipeline import release
    _no_network(monkeypatch)
    cfg = make_project(tmp_path)
    h1 = release.numbers_hash(cr.compute(cfg)[0])
    h2 = release.numbers_hash(cr.compute(cfg)[0])
    assert h1 == h2
