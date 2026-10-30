"""Trust review of 7 Oct 2026, group cr (client_report.py): the report's side
of contracts C1-C5 and decisions E3, E4, E5, E7, E9, E10, E11, E12, E13,
E14, E18, E20, E25, E26 and E28. Each test failed on the code before these
fixes (backups/pre_trust_20261007/client_report.py) unless its docstring
says otherwise.
"""
import json
import types

import numpy as np
import pytest

from insar_pipeline import client_report as cr
from insar_pipeline import significance as sg

MM_IN = 25.4
UP = 0.7838


def _cov(**kw):
    return {"cells": 900, "area_km2": 2.0, "footprint_km2": 2.0, "measured_percent": 95.0,
            "outside_frame_percent": 0.0, "coverage_balanced": True, **kw}


def _t(n=270, years=10.45):
    return np.linspace(2016.0, 2016.0 + years, n)


def _summ(t, rate=-5.0, total=None, method="rate_x_span", **kw):
    y = rate * (t - t[0])
    return {"rate_mm_yr": rate, "total_mm": rate * (t[-1] - t[0]) if total is None else total,
            "total_method": method, "nonlinear": method != "rate_x_span", "start_mm": 0.0,
            "series_mm": [round(float(v), 2) for v in y], **kw}


# The Site A Area A2 null as significance.assess gives it after E5/E6
# (LOS): worse sign 15.46 (rising), sinking 14.32
AREA_A2 = {"testable": True, "farmland_spread_1sigma_mm_yr": 5.6, "spread_for_ranges_mm_yr": 5.6,
                "null_median_mm_yr": 0.72, "excess_mm_yr": -5.17, "threshold_95_mm_yr": 10.58,
                "detectable_80_mm_yr": 15.46, "detection_limit_sink_mm_yr": 14.32, "detection_limit_rise_mm_yr": 15.46,
                "signal_to_threshold": 0.49, "distinguishable_from_farmland": False, "approx_independent_placements": 30,
                "null_thin": False, "method": "footprint", "null_shape": "footprint",
                "local_null": {"median_mm_yr": 1.09, "spread_mm_yr": 5.4, "n_independent": 12, "n_placements": 400,
                               "disagrees": False, "why": "agrees", "radius_km": 10.0},
                "coherence_null": None, "null_checks_note": "fewer than 10 independent copies within +/-0.025"}


# ------------------------------------------------------------------ E5 / C1: the sinking limit, the rising beside it
def test_detection_limit_is_the_sinking_one_and_the_nulls_are_stored():
    """E5: the limit a client reads is the sinking one (Area A2 19.7 ->
    18.3 mm/yr vertical); the rising limit is stored and printed only when it
    differs by more than a tenth. C1: the local / coherence-matched nulls are
    copied into the area entry."""
    tf = cr.tested_fields(AREA_A2, UP, _cov())
    assert tf["detection_limit_mm_yr"] == pytest.approx(14.32 / UP, abs=0.05)        # 18.3
    assert tf["detection_limit_rise_mm_yr"] == pytest.approx(15.46 / UP, abs=0.05)   # 19.7
    assert tf["detection_limit_either_direction_mm_yr"] == pytest.approx(19.7, abs=0.05)
    assert tf["detection_limit_basis"] == "sinking"
    assert tf["null_local"]["median_mm_yr"] == 1.09 and "line of sight" in tf["null_local"]["units"]
    assert tf["null_coherence_matched"] is None
    s = cr.detection_sentence(tf)
    assert s.startswith("Sustained sinking of 0.72 in/yr (18 mm/yr) or more") and "4 times in 5" in s
    assert "rising" not in s                                         # 19.7 is within 10% of 18.3
    # Site B's mine ground: sinking 6.8, rising 7.7 (13% more) - both said
    fr = dict(AREA_A2, detection_limit_sink_mm_yr=6.8 * UP, detection_limit_rise_mm_yr=7.7 * UP,
              detectable_80_mm_yr=7.7 * UP)
    s2 = cr.detection_sentence(cr.tested_fields(fr, UP, _cov()))
    assert "rising ground needs 0.30 in/yr (8 mm/yr)" in s2
    lines = dict(cr.one_pager_lines({"dates": ["20160105", "20260730"]},
                                    dict(cr.tested_fields(fr, UP, _cov()), kind="area", vertical_mm_yr=-1.0,
                                         range95_mm_yr=[-3.0, 1.0])))
    assert lines["Reliably detects sinking (4 times in 5)"] == "0.27 in/yr (7 mm/yr)"
    assert lines["Reliably detects rising (4 times in 5)"] == "0.30 in/yr (8 mm/yr)"
    # a significance record from before 7 Oct has no per-sign limit: the old wording
    old = {k: v for k, v in AREA_A2.items() if not k.startswith("detection_limit_")}
    assert cr.detection_sentence(cr.tested_fields(old, UP, _cov())).startswith("Sustained movement of 0.78 in/yr")


# ------------------------------------------------------------------ E10 / C3: the pace change against farmland
def _noisy(t, rng, rate, sd=3.0, season=4.0):
    return rate * (t - t[0]) + season * np.sin(2 * np.pi * t) + rng.normal(0, sd, t.size)


def test_measured_change_uses_the_bend_only_when_it_beats_farmland_copies():
    """E10: the hinge total only when the area's change of pace beats the
    95th percentile of farmland copies of its footprint; else rate x span,
    and 'nonlinear' follows. Too few copies (or none: the whole area):
    rate x span, untested. The region's LOS series is tested in the copies'
    vertical frame."""
    rng = np.random.default_rng(1)
    t = _t()
    up = 0.5
    copies = [_noisy(t, rng, rng.normal(0, 2.0)) for _ in range(30)]                 # vertical
    steady = _noisy(t, rng, -6.0) * up                                               # LOS
    s = cr.measured_change(t, {"rate_mm_yr": -3.0, "series_mm": list(steady)}, up, copies)
    assert s["total_method"] == "rate_x_span" and s["nonlinear"] is False
    assert s["pace_change"]["tested"] is True and s["pace_change"]["farmland_copies"] == 30
    p95 = sg.pace_change_null(t, copies)["p95_mm"]
    assert s["pace_change"]["farmland_p95_mm"] == pytest.approx(p95, abs=0.05)      # the copies' frame
    lin = sg.fit_rate(t, steady)
    assert s["total_mm"] == pytest.approx(lin["slope"] * lin["span_years"], abs=0.2)
    # a settling curve: fast, then stopped
    bend = (np.where(t - t[0] < 4, -30.0 * (t - t[0]), -120.0) + 4.0 * np.sin(2 * np.pi * t)
            + rng.normal(0, 3.0, t.size)) * up
    b = cr.measured_change(t, {"rate_mm_yr": -10.0, "series_mm": list(bend)}, up, copies)
    assert b["total_method"] != "rate_x_span" and b["nonlinear"] is True and b["pace_change"]["tested"] is True
    assert b["total_mm"] == pytest.approx(-120.0 * up, abs=8.0)
    assert len(b["copies_piecewise_vertical_mm"]) == 30
    # too few copies, or none
    for few in (copies[:10], []):
        f = cr.measured_change(t, {"rate_mm_yr": -10.0, "series_mm": list(bend)}, up, few)
        assert f["total_method"] == "rate_x_span" and f["nonlinear"] is False and f["pace_change"]["tested"] is False


# ------------------------------------------------------------------ E9: totals and untested figures with ranges
def test_tested_area_total_has_its_range():
    """E9: a total's 95% range - rate x span: the rate's range on the test
    frame times the years (+ GNSS x span, in quadrature, when anchored);
    a bend: the farmland copies' own bent totals."""
    t = _t()
    span = float(t[-1] - t[0])
    summ = _summ(t, rate=-5.0 * UP)
    sig = dict(AREA_A2, excess_mm_yr=-5.0 * UP - 0.72)          # the area's value = its series' rate
    a = cr.area_entry("Area A2", "x", "area", _cov(), summ, sig, UP, None, None, span, t)
    rel_tot = -5.0 * span
    med, thr = 0.72 / UP, 10.58 / UP
    assert a["total_range95_mm"] == pytest.approx([rel_tot - med * span - thr * span,
                                                   rel_tot - med * span + thr * span], abs=0.15)
    # the rate's range x the years (not anchored: the same frame)
    assert a["total_range95_mm"][0] == pytest.approx(a["range95_mm_yr"][0] * span, abs=0.5)
    g = cr.area_entry("Area A2", "x", "area", _cov(), summ, sig, UP, -0.76, 1.01, span, t)
    half = np.hypot(thr * span, 1.96 * 1.01 * span)
    assert g["total_range95_mm"] == pytest.approx([rel_tot - med * span - 0.76 * span - half,
                                                   rel_tot - med * span - 0.76 * span + half], abs=0.15)
    # the bend: the copies' own hinge totals are its null
    P = list(np.linspace(-20, 20, 41))
    hb = cr.area_entry("F", "x", "area", _cov(), _summ(t, rate=-5.0 * UP, total=-60.0 * UP,
                                                       method="piecewise_linear_yearly",
                                                       copies_piecewise_vertical_mm=P), AREA_A2, UP,
                       None, None, span, t)
    c = float(np.quantile(np.abs(np.asarray(P) - np.median(P)), 0.95))
    assert hb["total_range95_mm"] == pytest.approx([-60.0 - c, -60.0 + c], abs=0.15)


def test_untested_area_shows_the_uncertainty_it_can_state():
    """E9 (findings 4, 23, 48): Site B's project area printed '0.000
    in/yr' with no uncertainty. An untested area now carries the GNSS part
    and the few-patches part (widened for their number); its 95% range stays
    None so no direction word is read from it."""
    t = _t()
    span = float(t[-1] - t[0])
    up = 0.7717
    sig = {"testable": False, "not_testable_reason": "Too little ordinary farmland around this area.",
           "thin_null_spread_1sigma_mm_yr": 1.77, "approx_independent_placements": 3, "null_thin": True,
           "null_available": True}
    a = cr.area_entry("Site B project area", "x.geojson", "area", _cov(), _summ(t, rate=1.85 * up), sig, up,
                      -1.85, 1.28, span, t)
    assert a["testable"] is False and a["range95_mm_yr"] is None and cr.area_word(a) == ""
    u = float(np.hypot(1.28, 1.77 * sg.prediction_inflation(3) / up))
    assert a["untested_uncertainty_1sigma_mm_yr"] == pytest.approx(u, abs=0.01)
    assert a["untested_range95_mm_yr"] == pytest.approx([0.0 - 1.96 * u, 0.0 + 1.96 * u], abs=0.02)
    assert a["untested_total_range95_mm"] is not None and "GNSS" in a["untested_uncertainty_basis"]
    # the GNSS part alone when no farmland patch of its size fits at all
    b = cr.area_entry("P", "x", "area", _cov(), _summ(t, rate=1.85 * up), {"testable": False}, up, -1.85, 1.28, span, t)
    assert b["untested_range95_mm_yr"] == pytest.approx([-1.96 * 1.28, 1.96 * 1.28], abs=0.02)
    # the page prints it, said to be not compared, never '0.000'
    rows = cr.summary_rows({"areas": [dict(a, display_name="Site B project area")]})
    assert rows[0][1].startswith("0.00 in/yr") and "0.000" not in rows[0][1] and "not compared with farmland" in rows[0][3]


# ------------------------------------------------------------------ E11: the recent range on the test frame
def test_recent_range_is_the_last_two_years_test():
    """E11: the last two years' range is excess over the farmland median +/-
    the threshold (+ GNSS in quadrature when anchored), as the record's own
    range, not the value +/- 1.96 x the spread."""
    t = _t()
    span = float(t[-1] - t[0])
    rec = {"value_mm_yr": -8.0, "spread_mm_yr": 9.0, "excess_mm_yr": -8.6, "threshold_mm_yr": 17.1,
           "window": ["20240701", "20260630"]}
    a = cr.area_entry("A", "x", "area", _cov(), _summ(t), AREA_A2, UP, None, None, span, t, recent=rec)
    assert a["recent_range95_mm_yr"] == pytest.approx([(-8.6 - 17.1) / UP, (-8.6 + 17.1) / UP], abs=0.01)
    g = cr.area_entry("A", "x", "area", _cov(), _summ(t), AREA_A2, UP, -0.76, 1.01, span, t, recent=rec)
    h = np.hypot(17.1 / UP, 1.96 * 1.01)
    assert g["recent_range95_mm_yr"] == pytest.approx([-8.6 / UP - 0.76 - h, -8.6 / UP - 0.76 + h], abs=0.01)
    f = cr.feature_entry({"sign": -1, "mean_rel_vertical_mm_yr": -20.0}, _summ(t), AREA_A2, UP, span, t,
                         [int(x) for x in t], recent=rec)
    assert f["recent_range95_mm_yr"] == pytest.approx([(-8.6 - 17.1) / UP, (-8.6 + 17.1) / UP], abs=0.01)


# ------------------------------------------------------------------ E3: years since mining on the report's frame
def test_years_since_mining_is_vertical_on_the_reports_farmland():
    """E3: each dated area measured against the report's farmland
    (ctx.matched with the report's far series), vertical (/ its own los_up),
    by panel_history's own rules; rows on fewer than 3 areas kept out of the
    client table."""
    dates = [f"{y}{m:02d}15" for y in range(2016, 2024) for m in range(1, 13)]
    t = sg.years(dates)
    good = np.ones((10, 10), bool)
    seen = []

    def matched(series, mask, far):
        seen.append(far)
        return series - (-2.0) * (t - t[0])                  # less the farmland's own drift (-2 LOS a year)
    ctx = types.SimpleNamespace(dates=dates, t=t, good=good, matched=matched, cfg=None)
    dated = {f"__yr_{j}": {"name": f"M{j} 2018", "year": 2018, "ix": np.arange(j, j + 3), "up": 0.8} for j in range(3)}
    pl = {k: -6.0 * (t - t[0]) for k in dated}                # LOS: -6 - (-2) = -4 LOS = -5 vertical a year
    got = cr._years_since_mining(ctx, dated, pl, {"far": "report's"})
    assert got["by_years_since_mining_frame"] == "vertical" and seen and all(f == {"far": "report's"} for f in seen)
    rows = got["by_years_since_mining"]
    assert rows and all(r["n_areas"] == 3 for r in rows)
    assert all(r["median_change_mm_per_yr"] == pytest.approx(-5.0, abs=0.2) for r in rows)
    two = cr._years_since_mining(ctx, dict(list(dated.items())[:2]), pl, {"far": "report's"})
    assert two["by_years_since_mining"] == [] and two["by_years_since_mining_all_rows"]


# ------------------------------------------------------------------ E20 (finding 35): colour steps per side
def test_colour_steps_are_read_from_each_side():
    ms = {"edges_mm_yr": [-25.4, -19.05, -10.74, 9.6, 12.7, 19.05, 25.4], "n_below": 2, "top_in_per_yr": 1.0}
    s = cr.colour_steps_text(ms)
    assert "0.75, 1.00 in/yr on the sinking side" in s and "0.50, 0.75, 1.00 in/yr on the rising side" in s
    assert "each way" not in s
    sym = {"edges_mm_yr": [-25.4, -19.05, -12.7, -9.2, 9.09, 12.7, 19.05, 25.4], "n_below": 3, "top_in_per_yr": 1.0}
    assert cr.colour_steps_text(sym).startswith("Colours step at 0.50, 0.75, 1.00 in/yr each way")


# ------------------------------------------------------------------ the GNSS inside check (gnss_tie from 7 Oct)
def test_inside_station_check_prints_the_ties_verdict():
    """gnss_tie's same-window check (E16) carries verdict, consistent and a
    plain verdict_reason; the page prints that sentence and the gate keeps
    its consistent (None: not testable)."""
    reason = ("Not testable: over the 2.2 years the station and the radar record share (2016.0–2018.2), this check "
              "could only see a difference of about 1.08 in/yr (27 mm/yr) or more.")
    chk = {"site": "LAMC", "verdict": "not testable", "consistent": None, "verdict_reason": reason}
    rep = {"gnss": {"inside_checks": [chk], "los_up_median": UP}, "los_up_median": UP}
    sc = cr.station_check(chk, rep)
    assert sc is not None and sc["agrees"] is None and "Not testable: over the 2.2 years" in sc["html"]
    assert "agree" not in sc["html"].replace("Not testable", "")
    cr.mark_inside_checks(rep)
    assert rep["gnss"]["inside_checks"][0]["consistent"] is None
    differ = dict(chk, verdict="differ", consistent=False, verdict_reason="Over 2016.0–2018.2 ... they differ.")
    assert cr.station_check(differ, rep)["agrees"] is False


# ------------------------------------------------------------------ C4 / E7: listing by the chance test alone
def test_feature_search_lists_by_the_chance_test_alone(monkeypatch):
    """E7: a patch holding the QA's stand-out but within chance is not listed
    (until 7 Oct it was listed, cut from the farmland and called 'stands out
    clearly')."""
    from insar_pipeline import report_insights as ri
    cand = {"sign": -1, "quality_standout": True, "_mask": np.ones((2, 2), bool)}
    monkeypatch.setattr(ri, "smoothed_field", lambda ctx, up: {"band_ok": True})
    monkeypatch.setattr(ri, "find_features", lambda *a, **k: [dict(cand)])
    monkeypatch.setattr(ri, "feature_null", lambda *a, **k: {"tested": True})
    monkeypatch.setattr(ri, "selection_p", lambda f, sm, null: {"tested": True, "p": 0.25, "stands_out": False})
    _, cands, _, _ = cr._feature_search(None, None, {}, None, [{"lat": 0, "lon": 0}])
    assert cands[0]["stands_out"] is False


def test_calibrated_standouts_only():
    """E7 / C4: the QA's stand-outs reach the search only when calibrated; a
    QA from before 7 Oct (gap > 2, uncalibrated) gives none."""
    old = {"findings": {"monitoring_point": {"standout": {"lat": 38.24, "lon": -87.65, "stands_out": True}}}}
    assert cr._calibrated_standouts(old) == []
    new = {"findings": {"monitoring_point": {"standout_calibration": {"tested": True}, "standouts": [
        {"lat": 38.3044, "lon": -87.3843, "scale_km": 1.0, "stands_out": True, "p_calibrated": 0.01},
        {"lat": 38.2754, "lon": -87.3511, "scale_km": 2.0, "stands_out": True, "p_calibrated": 0.01}]}}}
    assert [s["scale_km"] for s in cr._calibrated_standouts(new)] == [1.0, 2.0]


# ------------------------------------------------------------------ E13: the comparison ground's width
def test_comparison_ground_variants_measure_the_cut_width():
    """E13: the whole area's relative figure with the cut grown by 0 / 500 /
    1000 m; the 500 m one is the report's farmland, and nothing cut gives 0."""
    ny = nx = 80
    attrs = {"X_FIRST": "-87.6", "Y_FIRST": "38.4", "X_STEP": "0.00115", "Y_STEP": "-0.0009"}
    tc = np.full((ny, nx), 0.95)
    vel = np.zeros((ny, nx))
    cut = np.zeros((ny, nx), bool)
    cut[35:45, 35:45] = True
    ring = sg.euclidean_buffer(cut, attrs, 900.0) & ~sg.euclidean_buffer(cut, attrs, 300.0)
    vel[ring] = -10.0
    ctx = types.SimpleNamespace(sg=sg, vel=vel, tc=tc, kcls=sg.coherence_class(tc), attrs=attrs)
    far0 = np.ones((ny, nx), bool)
    up = np.full((ny, nx), UP)
    m = np.ones((ny, nx), bool)
    got = cr._comparison_ground_variants(ctx, far0, cut, m, up)
    v0, v500, v1000 = got["relative_vertical_mm_yr"]
    far500 = far0 & ~sg.euclidean_buffer(cut, attrs, 500.0)
    r500, _ = sg.class_matched_field(vel, tc, far500)
    assert v500 == pytest.approx(float(np.mean(r500 / UP)), abs=1e-3)
    assert v0 > v500 > v1000 and got["sd_mm_yr"] == pytest.approx(float(np.std([v0, v500, v1000], ddof=1)), abs=1e-3)
    assert cr._comparison_ground_variants(ctx, far0, np.zeros((ny, nx), bool), m, up)["sd_mm_yr"] == 0.0


# ------------------------------------------------------------------ a synthetic project through the whole report
import test_fix_L9b as L9b                                     # noqa: E402  (the report's project builder)


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    """L9b's project (farmland, an unmapped settling patch, an ordinary
    farmland field and a 70%-unmeasured pit as report areas, no mine layer)
    through compute, the figures, the page, the PDF (its text recorded) and
    the rasters."""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.figure import Figure
    root = tmp_path_factory.mktemp("trust_cr")
    cfg = L9b.make_project(root)
    rep, ctx, masks, outlines = cr.compute(cfg)
    D = cfg.project_dir / "deliverables"
    figs = cr.figures(cfg, rep, ctx, masks, outlines, out_dir=D / "figures")
    html = cr.page(cfg, rep, figs)
    texts = []
    mp = pytest.MonkeyPatch()
    orig = Figure.text

    def rec(self, x, y, s, *a, **k):
        texts.append((float(y), str(s), k.get("fontsize")))
        return orig(self, x, y, s, *a, **k)
    mp.setattr(Figure, "text", rec)
    axes = []
    orig_axes = Figure.add_axes

    def rec_axes(self, rect, *a, **k):
        axes.append(list(rect))
        return orig_axes(self, rect, *a, **k)
    mp.setattr(Figure, "add_axes", rec_axes)
    try:
        cr.area_pdf(cfg, rep, ctx, outlines, D / cr.PDF_NAME, figs)
    finally:
        mp.undo()
    cr.write_rasters(cfg, rep, ctx)
    return types.SimpleNamespace(cfg=cfg, rep=rep, ctx=ctx, masks=masks, html=html, figs=figs, D=D,
                                 pdf_texts=texts, pdf_axes=axes)


def _area(rep, name):
    return next(a for a in rep["areas"] if a["name"] == name)


def _text(html):
    import re
    import html as h
    return " ".join(h.unescape(re.sub(r"<[^>]+>", " ", html)).split())


def test_report_areas_are_cut_from_the_comparison_ground(built):
    """E28 / C2: the field and the pit lie on farmland (no mine layer): they
    are cut from it (+500 m), before the feature search, and the cut is in
    the recorded farmland so the Results and the KMZ rebuild it."""
    out = built.rep["report_areas_out_of_farmland"]
    assert out["applied"] and set(out["areas_cut"]) == {"Test Field", "Wet Pit"} and out["far_cells_removed"] > 0
    field = sg.euclidean_buffer(built.masks["Test Field"] | ~built.ctx.good & False, built.ctx.attrs, 400.0)
    assert not (built.ctx.far & field).any() and not (built.ctx.far_wide & field).any()
    sp = built.rep["farmland_second_pass"]
    assert set(sp["report_areas_removed"]) == {"Test Field", "Wet Pit"} and sp["features_removed"] >= 1
    cut = cr.runs_mask(sp["cut_cells"], built.ctx.good.shape)
    assert cut[L9b.FIELD_BOX].all() and cut[L9b.FEATURE].any()
    assert sp["far_cells"][1] == int(built.ctx.far.sum())


def test_whole_area_uncertainty_carries_the_comparison_ground_term(built):
    """E13: the whole area's relative 1 sigma = hypot(block scatter, the sd
    of its figure over cut widths 0 / 500 / 1000 m)."""
    full = _area(built.rep, "Full Satellite Imaging")
    cg = full["comparison_ground"]
    assert cg["buffers_m"] == [0.0, 500.0, 1000.0] and cg["sd_mm_yr"] > 0
    assert full["relative_uncertainty_1sigma_mm_yr"] == pytest.approx(
        np.hypot(full["relative_block_sd_mm_yr"], cg["sd_mm_yr"]), abs=0.01)
    assert full["relative_vertical_mm_yr"] == pytest.approx(cg["relative_vertical_mm_yr"][1], abs=0.3)


def test_every_total_on_the_page_pdf_and_table_has_its_range(built):
    """E9: total_range95_mm on every measured area, printed beside the total
    on the page, the PDF and the summary table; the whole area's is its
    rate range x the years; no total in the short answer."""
    rep = built.rep
    full = _area(rep, "Full Satellite Imaging")
    span = rep["span_years"]
    assert full["total_range95_mm"] == pytest.approx([full["total_mm"] - 1.96 * full["vertical_uncertainty_1sigma_mm_yr"] * span,
                                                      full["total_mm"] + 1.96 * full["vertical_uncertainty_1sigma_mm_yr"] * span],
                                                     abs=0.2)
    field = _area(rep, "Test Field")
    lo, hi = field["total_range95_mm"]
    t = _text(built.html)
    assert f"95%: {cr._s(lo / MM_IN)} to {cr._s(hi / MM_IN)} in" in t
    lines = dict(cr.one_pager_lines(rep, field))
    assert f"95% range {cr._s(lo / MM_IN)} to {cr._s(hi / MM_IN)} in" in next(v for k, v in lines.items()
                                                                            if k.startswith("Measured over"))
    assert any("95%:" in r[2] for r in cr.summary_rows(rep))
    answer = built.html.split('<div class="answer prose">')[1].split("</div>")[0]
    assert "over the record" not in answer
    # rates to 0.01 in/yr everywhere (E9: no '0.000')
    import re
    assert not re.search(r"[+-]\d\.\d{3} in/yr", t)
    # the pit could not be compared: the uncertainty it can state, said so
    pit = _area(rep, "Wet Pit")
    assert pit["testable"] is False and pit["range95_mm_yr"] is None and pit["untested_range95_mm_yr"]
    assert "not compared with ordinary ground" in t


def test_totals_use_rate_x_span_unless_the_pace_change_beats_farmland(built):
    """E10: the whole area has no copy of itself: rate x span, untested; a
    report area's pace change is tested against its farmland copies."""
    full = _area(built.rep, "Full Satellite Imaging")
    assert full["total_method"] == "rate_x_span" and full["pace_change"]["tested"] is False
    assert full["total_mm"] == pytest.approx(full["total_at_average_rate_mm"], abs=0.5)
    for f in built.rep["features"]:
        assert f["pace_change"] is not None and f["nonlinear"] == (f["total_method"] != "rate_x_span")


def test_single_point_uncertainty_is_the_rasters(built):
    """E18: the header's single-point 1 sigma is the uncertainty raster's own
    median over the study area (the cell null, vertical, with GNSS), not the
    split-half repeatability."""
    import rasterio
    sp = built.rep["single_point_uncertainty_1sigma_mm_yr"]
    with rasterio.open(built.D / "vertical_movement_uncertainty_1sigma_in_per_yr.tif") as r:
        u = r.read(1) * MM_IN
    m = built.masks["Full Satellite Imaging"]
    assert sp == pytest.approx(float(np.nanmedian(u[m])), abs=0.02)
    assert f"uncertain by about ±{sp / MM_IN:.2f} in/yr (1 sigma)" in _text(built.html)


def test_radar_cells_counted_over_the_study_area_and_not_independent(built):
    """E25: the count is over the study area's points and worded as radar
    resolution cells; 'independent measurements' is gone from the page."""
    t = _text(built.html)
    assert "independent radar measurements" not in t and "independent measurements about" not in t
    assert f"from about {built.rep['independent_radar_measurements']:,} radar resolution cells" in t
    assert built.rep["independent_radar_measurements"] <= _area(built.rep, "Full Satellite Imaging")["cells"]


def test_no_injection_sentence_on_the_client_page(built):
    """E4 / C5: the injection test is technical only - no client sentence,
    whatever the record holds."""
    rep = json.loads(json.dumps(built.rep))
    rep["injection_test"] = {"status": "complete", "sentence": "Patches about 0.3 km across were found 20 of 20 times."}
    t = _text(cr.page(built.cfg, rep, built.figs))
    assert "found 20 of 20" not in t and "injection" not in t.lower() and "Planting known movement" not in t
    assert built.rep["injection_test"]["sentence"] is None


def test_stands_out_clearly_only_from_the_features_own_decision(built):
    """C4 / E7 / E8: the page's 'stands out clearly' comes only from
    f['stands_out_clearly'] (selection_p: both calibrated tests and the null
    check); a QA stand-out alone is 'worth a look'. The null check is
    stored for the release gate."""
    rep = json.loads(json.dumps(built.rep))
    assert "real_beats_all_copies" in rep["feature_null_check"]
    f0 = rep["features"][0]
    rep["features"] = [dict(f0, quality_standout=True, stands_out=True, stands_out_clearly=False)]
    t = _text(cr.page(built.cfg, rep, built.figs))
    assert "stands out clearly" not in t and "worth a look" in t
    rep["features"] = [dict(f0, quality_standout=True, stands_out=True, stands_out_clearly=True)]
    assert "1 local feature stands out clearly" in _text(cr.page(built.cfg, rep, built.figs))


def test_patches_mostly_outside_an_area_are_listed_apart(built):
    """E12 (findings 6, 32, 45): Site B's Feature 1, 0.16 of 2.21 km²
    inside the project area, was listed under it. A patch under half inside
    is listed apart with the size inside."""
    rep = json.loads(json.dumps(built.rep))
    rep["map_patches"] = [
        {"area": "Test Field", "sign": -1, "area_km2": 2.21, "inside_area_km2": 0.16, "inside_share": 0.07,
         "mean_rel_vertical_mm_yr": -35.0, "lat": 38.3044, "lon": -87.3843, "feature": None},
        {"area": "Test Field", "sign": 1, "area_km2": 1.2, "inside_area_km2": 1.2, "inside_share": 1.0,
         "mean_rel_vertical_mm_yr": 12.0, "lat": 38.30, "lon": -87.38, "feature": None}]
    html = cr.page(built.cfg, rep, built.figs)
    inside = html.split("Coloured patches inside the report areas")[1].split("</table>")[0]
    outside = html.split("Coloured patches mostly outside the report areas")[1].split("</table>")[0]
    assert "1.20 km²" in inside and "2.21 km²" not in inside
    assert "beside Test Field" in outside and "0.16 km² inside" in outside
    assert all("inside_share" in p for p in built.rep.get("map_patches") or [])


def test_map_caption_band_and_methods_wording(built):
    """E20 (findings 35, 36, 49): the grey band is the middle 95% of
    farmland; 'widened' only for a thin null."""
    t = _text(built.html)
    # no land-cover map on the synthetic project: "ground" (28 Oct 2026)
    assert "centred on its own typical value" not in t and "middle 95% of ordinary ground" in t
    rep = json.loads(json.dumps(built.rep))
    a = _area(rep, "Test Field")
    a.update({"null_thin": False, "null_shape": "compact_square", "approx_independent_placements": 16, "testable": True})
    m = dict(cr.methods_stages(built.cfg, rep))["Uncertainty"]
    assert "about 16 separate patches" in m and "not widened" in m and "have been widened" not in m
    a.update({"null_thin": True, "approx_independent_placements": 6})
    assert "have been widened" in dict(cr.methods_stages(built.cfg, rep))["Uncertainty"]


def test_years_since_mining_table_needs_the_reports_frame(built):
    """E3: a record carrying panel_history's line-of-sight, first-pass table
    prints no table; one rebuilt on the report's frame prints it vertical."""
    rep = json.loads(json.dumps(built.rep))
    rows = [{"years_since_mining": k, "n_areas": 5, "median_change_mm_per_yr": -14.2, "q25_mm": -20.0, "q75_mm": -9.0}
            for k in (0, 1)]
    rep.update({"by_years_since_mining": rows, "by_years_since_mining_note": "change of each dated area's ..."})
    rep.pop("by_years_since_mining_frame", None)
    assert "Movement by years since mining" not in cr.page(built.cfg, rep, built.figs)
    rep["by_years_since_mining_frame"] = "vertical"
    assert "Median vertical change per year" in cr.page(built.cfg, rep, built.figs)


def test_cells_standing_out_on_the_second_pass(built):
    """E14: the whole area's single-cell shares are measured on the report's
    own farmland and cell null (rate_signal_to_noise.tif >= 1.96), not read
    from summary.json's first pass."""
    import rasterio
    full = _area(built.rep, "Full Satellite Imaging")
    with rasterio.open(built.D / "rate_signal_to_noise.tif") as r:
        snr = r.read(1)
    m = built.masks["Full Satellite Imaging"] & np.isfinite(snr)
    assert full["cells_standing_out_percent"] == pytest.approx(100 * float((snr[m] >= 1.96).mean()), abs=0.15)
    assert full["cells_standing_out_by_chance_percent"] is not None


def test_pdf_pages_carry_the_caveats_and_the_chart_sits_below_the_text(built):
    """E26: every PDF page says vertical is assumed from one track, the
    ground each value combines, and relative vs GNSS-anchored; finding 37:
    the chart starts below the text block."""
    caveats = [s for _, s, _ in built.pdf_texts if "Vertical assumes all movement is up or down" in s]
    assert len(caveats) >= len(built.rep["areas"])            # once a page (again when set smaller to fit)
    one = " ".join(caveats[0].split())
    assert "line of sight from one satellite track" in one and "GNSS" in one
    sup = built.rep["resolution"]["support_m"]
    assert f"about {int(5 * round(sup[0] / 5)) if sup[0] < 100 else int(10 * round(sup[0] / 10))} m east-west" in one
    charts = [r for r in built.pdf_axes if abs(r[0] - 0.08) < 1e-6]
    assert len(charts) == len(built.rep["areas"]) and all(r[1] + r[3] <= 0.44 + 1e-9 for r in charts)


# ------------------------------------------------------------------ reviewer of group cr (7 Oct 2026)
def test_whole_area_standout_names_the_feature_that_holds_it():
    """The whole area's 'standout' block named the FIRST listed feature
    holding any calibrated stand-out, not the one holding this stand-out:
    Site B's largest-gap (1 km) stand-out at 38.3044 N, 87.3843 W lies in
    Feature 2, but the record said feature 1 (which holds the 2 km one)."""
    attrs = {"X_FIRST": "-87.40", "Y_FIRST": "38.32", "X_STEP": "0.001", "Y_STEP": "-0.001"}
    ctx = types.SimpleNamespace(sg=sg, attrs=attrs)
    f1, f2 = np.zeros((60, 60), bool), np.zeros((60, 60), bool)
    f1[40:50, 40:50] = True                                   # around 38.274 N, 87.356 W
    f2[10:20, 12:22] = True                                   # around 38.304 N, 87.384 W
    listed = [{"_mask": f1, "quality_standout": True}, {"_mask": f2, "quality_standout": True}]
    assert cr._listed_feature_holding(ctx, listed, {"lat": 38.3044, "lon": -87.3843}) == 2
    assert cr._listed_feature_holding(ctx, listed, {"lat": 38.2741, "lon": -87.3559}) == 1
    assert cr._listed_feature_holding(ctx, listed, {"lat": 38.2401, "lon": -87.6462}) is None


def test_inside_check_with_a_verdict_is_read_one_way():
    """station_check keyed the tie's own verdict on verdict_reason while
    mark_inside_checks keyed it on verdict: a check carrying a verdict but no
    sentence was printed by the old rule (here: nothing) while the gate kept
    the tie's value. Both now read the verdict."""
    chk = {"site": "LAMC", "verdict": "agree", "consistent": True}
    rep = {"gnss": {"inside_checks": [dict(chk)], "los_up_median": UP}, "los_up_median": UP}
    sc = cr.station_check(chk, rep)
    assert sc is not None and sc["agrees"] is True
    cr.mark_inside_checks(rep)
    assert rep["gnss"]["inside_checks"][0]["consistent"] is True
    nt = cr.station_check({"site": "LAMC", "verdict": "not testable", "consistent": None}, rep)
    assert nt["agrees"] is None and "Not testable" in nt["html"]


def test_recent_sentence_and_glossary_wording():
    """The recent sentence never prints '-0.00' (E9, _s), and the glossary
    does not say the rising limit is always the larger (it is printed when
    it differs by more than a tenth either way)."""
    s = cr.recent_sentence({"recent_rate_mm_yr": -0.1, "recent_window": ["20240701", "20260630"],
                            "recent_range95_mm_yr": [-0.1, 0.05]})
    assert "-0.00" not in s and "0.00 in/yr" in s
    terms = dict(cr.glossary({"control_ground_label": "farmland"}))
    assert "sinking" in terms["Detection limit"] and "needs noticeably more" not in terms["Detection limit"]
