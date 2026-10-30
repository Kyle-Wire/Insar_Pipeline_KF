"""Final check of 7 Oct 2026, group report (client_report.py, report_insights.py)

H1  "stands out clearly" also needs the quality assessment's own stand-out
    calibration to be sound (qa_standout_doubt): flagged copies give at most
    "worth a look" (Site A Feature 1)
M1  "began around <year>" only after ordinary years; a feature already beyond
    farmland at the start "was already sinking when the record starts", and
    the tested change of pace "sped up around / between ..." (trend_shape)
M2  "next to" mapped mining (and the mining causes and sideways caveat) only
    within NEAR_MINE_KM; beyond it the distance, plainly (relation_to_mines)
M3  report_revision: the page and the PDF say "Revision N, replacing the
    report issued <date>" and what changed
M4  every feature rate carries its 95% range and the outline caveat (page,
    PDF, summary table); the mine ground's average its range
L   the chance printed is the listing test's (first_pass_chance); signed()
    never '+0'; 'weaker' patches only when slower; plain words for the
    reference check, the second quality check, field practice and the
    inside station; '1 sigma' in the Terms; no detection-limit words without
    a limit; the footer's route and 'on request'; the disclaimer; the years
    table in inches with mm to 0.1; the charts' thick lines from zero; the
    summary table's whole-area footnote; the PDF's numbered boxes and colour
    scale

Each test failed on the code before these fixes (backups/pre_final_20261007/).

Run: python -m pytest insar_pipeline/tests/test_fix_oct7_last_report.py -q -p no:cacheprovider
"""
import json
import types

import numpy as np
import pytest

from insar_pipeline import client_report as cr
from insar_pipeline import report_insights as ri
from insar_pipeline import significance as sg

import test_fix_L9b as L9b                                     # noqa: E402  (the report's project builder)


# ------------------------------------------------------------------ helpers
def _ctx(field, pixel=60.0):
    ny, nx = field.shape
    dlat = pixel / 111320.0
    dlon = pixel / (111320.0 * np.cos(np.radians(38.3)))
    attrs = {"X_FIRST": "-87.6", "Y_FIRST": "38.4", "X_STEP": str(dlon), "Y_STEP": str(-dlat)}
    good = np.ones((ny, nx), bool)
    return types.SimpleNamespace(sg=sg, attrs=attrs, good=good, far=good, far_wide=good, r_cm=field, r_null=field)


def _null(top=55.5, n=199):
    return {"tested": True, "null_area_km2": 300.0, "searched_area_km2": 400.0,
            "sides": {"sinking": {"tested": True, "max_mass": list(np.linspace(5.0, top, n)),
                                  "median_max_beyond_mm_yr": 18.8, "farmland_strongest_mass": 0.0},
                      "rising": {"tested": True, "max_mass": list(np.linspace(3.0, 20.0, n)),
                                 "median_max_beyond_mm_yr": 13.0, "farmland_strongest_mass": 0.0}}}


def _sm():
    return {"centre_mm_yr": 0.0, "band_lo_mm_yr": -10.0, "band_hi_mm_yr": 8.0}


def _feature(**kw):
    f = {"sign": -1, "area_km2": 23.7, "lat": 38.241, "lon": -87.647, "mean_rel_vertical_mm_yr": -16.3,
         "vertical_relative_mm_yr": -16.3, "total_relative_mm": -172.0, "total_range95_mm": [-243.0, -104.0],
         "rate_range95_mm_yr": [-23.0, -9.84], "nonlinear": False,
         "trend": {"shape": "steady", "text": "it runs through the whole record and its recent pace is not "
                                             "measurably different from the earlier years"},
         "mines": {"key": "none", "phrase": "1.8 km from the nearest mapped mine area"},
         "chance": {"tested": True, "p": 0.015, "stands_out": True, "copies_as_strong": 0, "copies": 199},
         "quality_standout": True, "cause_hint": "The mapped mines do not explain it."}
    f.update(kw)
    return f


def _text(html):
    import re
    import html as h
    html = re.sub(r"<style>.*?</style>", " ", html, flags=re.S)
    return " ".join(h.unescape(re.sub(r"<[^>]+>", " ", html)).split())


# Site A's and Site B's stand-out calibrations as their final QA records hold them
OW_CAL = {"tested": True, "gap_threshold": 2.02, "copies_match_within_band": False,
          "copies_match": {"1": {"ratio": 0.88}, "2": {"ratio": 0.85}, "4": {"ratio": 0.72}}}
FR_CAL = {"tested": True, "gap_threshold": 2.83, "copies_match_within_band": True,
          "copies_match": {"1": {"ratio": 0.91}, "2": {"ratio": 0.97}, "4": {"ratio": 1.01}}}


# ------------------------------------------------------------------ H1: the QA's own calibration
def test_a_flagged_qa_calibration_cannot_confirm_a_feature():
    """Site A's QA flags its own copies (4 km spread ratio 0.72, band
    0.8-1.25): its stand-out no longer makes Feature 1 "stands out clearly".
    Site B's copies match: unchanged."""
    why = ri.qa_standout_doubt(OW_CAL)
    assert why and "4 km: 0.72" in why and "1 km" not in why
    assert ri.qa_standout_doubt(FR_CAL) is None
    assert ri.qa_standout_doubt({"tested": False, "copies_match_within_band": False}) is None
    assert "still holds this patch" in ri.qa_standout_doubt(dict(FR_CAL, candidate_in_farmland=True))
    strong = {"sign": -1, "area_km2": 23.7, "beyond_farmland_mm_yr": 16.9, "quality_standout": True}
    f = dict(strong, quality_standout_doubt=why)
    ch = ri.selection_p(f, _sm(), _null())
    assert ch["stands_out"] is True and f["stands_out_clearly"] is False and ch["stands_out_clearly"] is False
    assert ch["clearly_withheld_by"] == "qa_calibration" and ch["clearly_withheld"] == why
    f2 = dict(strong, quality_standout_doubt=None)
    assert ri.selection_p(f2, _sm(), _null())["stands_out_clearly"] is True and f2["stands_out_clearly"] is True
    t = ri.feature_text(1, _feature(stands_out_clearly=False, chance=ch))
    assert "worth a look rather than a confirmed finding" in t and "stands out clearly" not in t
    assert "not a close enough match to the real farmland here to confirm it" in t
    assert "holds a patch stronger" not in t


def test_the_report_carries_the_qa_doubt_to_the_feature_holding_the_standout():
    """client_report._calibrated_standouts puts the QA's doubt on each of its
    stand-outs and find_features on the feature holding one."""
    Q = {"findings": {"monitoring_point": {"standout_calibration": OW_CAL, "standouts": [
        {"scale_km": 4.0, "lat": 0.0, "lon": 0.0, "stands_out": True, "p_calibrated": 0.03, "gap_in_spreads": 2.2}]}}}
    so = cr._calibrated_standouts(Q)
    assert len(so) == 1 and "0.72" in so[0]["calibration_doubt"]
    Q["findings"]["monitoring_point"]["standout_calibration"] = FR_CAL
    assert cr._calibrated_standouts(Q)[0]["calibration_doubt"] is None
    f = np.zeros((120, 120))
    f[20:40, 20:40] = -40.0
    ctx = _ctx(f)
    a = ctx.attrs
    sm = {"field": f, "centre_mm_yr": 0.0, "band_lo_mm_yr": -10.0, "band_hi_mm_yr": 10.0, "band_mm_yr": 10.0,
          "band_ok": True, "pixel_m": (60.0, 60.0)}
    at = {"lat": float(a["Y_FIRST"]) + 30.5 * float(a["Y_STEP"]), "lon": float(a["X_FIRST"]) + 30.5 * float(a["X_STEP"]),
          "stands_out": True}
    got = ri.find_features(ctx, sm, {}, np.ones(f.shape, bool), [dict(at, calibration_doubt="copies off")], limit=None)
    assert got[0]["quality_standout"] is True and got[0]["quality_standout_doubt"] == "copies off"
    got = ri.find_features(ctx, sm, {}, np.ones(f.shape, bool), [dict(at, calibration_doubt=None)], limit=None)
    assert got[0]["quality_standout"] is True and got[0]["quality_standout_doubt"] is None


# ------------------------------------------------------------------ M1: the onset
def _dates_gap_2018():
    """12-day dates 2016-2026 with 2018's first six weeks missing (2018's
    dates then span 0.85 yr: not used, as Site B's)."""
    t = np.arange(2016.0, 2026.0, 12 / 365.25)
    t = t[~((t >= 2018.0) & (t < 2018.12))]
    return t, [int(x) for x in t]


def _series_from_yearly(t, years, rate_by_year):
    s = np.zeros(t.size)
    for k in range(1, t.size):
        s[k] = s[k - 1] + rate_by_year[years[k]] * (t[k] - t[k - 1])
    return s


FR_F2 = {2016: -17.6, 2017: -22.5, 2018: -45.6, 2019: -75.4, 2020: -54.0, 2021: -40.4, 2022: -42.4,
         2023: -19.7, 2024: -7.8, 2025: -11.8}


def test_a_feature_already_sinking_at_the_start_did_not_begin_later():
    """Site B Feature 2: 2016-2017 average -20 mm/yr against farmland
    placements of sd ~5.7 (z -3.75); it read "it began around 2019". Now:
    already sinking when the record starts, and the tested change of pace
    is placed between the last year used before it and the onset year
    (2018 not used)."""
    t, years = _dates_gap_2018()
    used = [y for y in sorted(set(years)) if y != 2018]
    rows = np.random.default_rng(4).normal(0, 5.7, (40, len(used)))
    d = ri.trend_shape(t - 2016.0, _series_from_yearly(t, years, FR_F2), years, noise_mm_yr=1.0,
                       null_yearly={"years": used, "rates": rows})
    assert 2018 in d["years_not_used"]
    assert "began" not in d and "began" not in d["text"]
    assert d["text"].startswith("it was already sinking when the record starts (2016)")
    assert "it sped up between 2017 and 2019" in d["text"]
    b = d["contrasts"]["before_onset"]
    assert b["years"] == [2016, 2017] and b["z"] > ri.CONTRAST_Z and d.get("already_moving_at_start") is True


def test_began_needs_ordinary_years_before_it():
    """Years before the onset within farmland's spread: "began around 2019"
    still stands, and no years are named for the recent pace (one recent
    window on the page)."""
    t, years = _dates_gap_2018()
    used = [y for y in sorted(set(years)) if y != 2018]
    calm = {**FR_F2, 2016: -1.0, 2017: 1.5, 2018: -2.0}
    rows = np.random.default_rng(4).normal(0, 5.7, (40, len(used)))
    d = ri.trend_shape(t - 2016.0, _series_from_yearly(t, years, calm), years, noise_mm_yr=1.0,
                       null_yearly={"years": used, "rates": rows})
    assert d["began"] == 2019 and "it began between 2017 and 2019" in d["text"]
    assert "already" not in d["text"] and "2024" not in d["text"] and "2025" not in d["text"]
    t2 = np.arange(2016.0, 2026.0, 12 / 365.25)
    y2 = [int(x) for x in t2]
    d = ri.trend_shape(t2 - 2016.0, _series_from_yearly(t2, y2, calm), y2, noise_mm_yr=1.0,
                       null_yearly={"years": sorted(set(y2)), "rates": np.random.default_rng(4).normal(0, 5.7, (40, 10))})
    assert "it began around 2018" in d["text"] or "it began around 2019" in d["text"]


# ------------------------------------------------------------------ M2: next to mapped mining
def test_next_to_mining_only_within_half_a_kilometre():
    """Site A Feature 1 lies 1.84 km from mapped workings and read 'next
    to mapped mine areas ... beside mapped mining' with the mining causes
    and the sideways-movement caveat. Beyond NEAR_MINE_KM the distance is
    given plainly and nothing is attributed to mining."""
    g = np.zeros((200, 200), bool)
    surface = g.copy(); surface[50:100, 50:100] = True
    poly = {"surface": surface}
    near = g.copy(); near[105:115, 60:70] = True                    # 5 cells (75 m) below it
    r = ri.relation_to_mines(near, poly, (15.0, 15.0))
    assert r["key"] == "near_surface" and r["phrase"].startswith("next to mapped surface mines")
    far = g.copy(); far[0:10, 180:195] = True                      # ~1.2 km at 15 m cells
    r = ri.relation_to_mines(far, poly, (15.0, 15.0))
    assert r["key"] == "none" and r["phrase"] == f"{r['nearest_km']:.1f} km from the nearest mapped surface mine"
    assert 1.0 < r["nearest_km"] < 1.5
    assert ri.NEAR_MINE_KM == 0.5
    f = _feature(mines=r, los_relative_mm_yr=-12.6, cause_hint=ri.cause_hint(r["key"], -1, "steady"),
                 stands_out_clearly=True)
    t = ri.feature_text(1, f)
    assert "km from the nearest mapped surface mine" in t and "sideways" not in t and "beside mapped mining" not in t
    assert "The mapped mines do not explain it" in t


# ------------------------------------------------------------------ the report on a synthetic project
@pytest.fixture(scope="module")
def built(tmp_path_factory):
    import matplotlib
    matplotlib.use("Agg")
    root = tmp_path_factory.mktemp("last_report")
    cfg = L9b.make_project(root)
    rep, ctx, masks, outlines = cr.compute(cfg)
    D = cfg.project_dir / "deliverables"
    figs = cr.figures(cfg, rep, ctx, masks, outlines, out_dir=D / "figures")
    return types.SimpleNamespace(cfg=cfg, rep=rep, ctx=ctx, masks=masks, outlines=outlines, figs=figs, D=D)


def _rep(built):
    return json.loads(json.dumps(built.rep))


def _record_pdf_text(built, rep, cfg=None, path=None):
    from matplotlib.figure import Figure
    texts, axes = [], []
    mp = pytest.MonkeyPatch()
    orig, orig_axes = Figure.text, Figure.add_axes

    def rec(self, x, y, s, *a, **k):
        texts.append(str(s))
        return orig(self, x, y, s, *a, **k)

    def rec_axes(self, rect, *a, **k):
        axes.append(list(rect))
        return orig_axes(self, rect, *a, **k)
    mp.setattr(Figure, "text", rec)
    mp.setattr(Figure, "add_axes", rec_axes)
    try:
        cr.area_pdf(cfg or built.cfg, rep, built.ctx, built.outlines, path or (built.D / "test_last.pdf"), built.figs)
    finally:
        mp.undo()
    return texts, axes


def test_the_chance_printed_is_the_test_that_listed_the_feature(built):
    """A feature that beat chance on the first pass is listed by it and cut;
    the second pass, without it in the farmland, scored it against copies
    that never held it. The printed chance is now the first pass's
    (Site B Feature 1: 'about 2 in 200', not 'less than 1 in 200')."""
    second = {"tested": True, "p": 0.005, "stands_out": True, "copies_as_strong": 0, "copies": 199,
              "p_text": "less than 1 in 200", "cluster_mass": 94.0, "stands_out_clearly": True}
    first = {"tested": True, "p": 0.01, "stands_out": True, "copies_as_strong": 1, "copies": 199,
             "p_text": "about 2 in 200", "cluster_mass": 92.8}
    ch = ri.first_pass_chance(second, first)
    assert ch["p_text"] == "about 2 in 200" and ch["copies_as_strong"] == 1 and ch["listed_on"] == "first pass"
    assert ch["second_pass"]["p_text"] == "less than 1 in 200" and ch["stands_out_clearly"] is True
    t = ri.feature_text(1, _feature(chance=ch, stands_out_clearly=True))
    assert "1 produced a patch as large and as strong (a chance of about 2 in 200)" in t
    feats = built.rep["features"]
    assert feats, "the planted settling patch is listed"
    first_pass = [c["first_pass"] for c in built.rep["feature_leave_out"]["candidates"]
                  if c.get("passed") and c.get("left_out") is None]
    assert feats[0]["chance"]["listed_on"] == "first pass"
    assert feats[0]["chance"]["p"] == pytest.approx(first_pass[0]["p"])


def test_feature_rates_carry_their_range_and_the_outline_caveat(built):
    """M4: the rate's 95% range on the footprint test's frame (for a rate x
    span total it is the total's range / the years), in the paragraph, the
    summary sentence and the PDF; the outline caveat beside them."""
    rep = _rep(built)
    f = rep["features"][0]
    lo, hi = f["rate_range95_mm_yr"]
    assert lo < f["vertical_relative_mm_yr"] < hi < 0
    if f["total_method"] == "rate_x_span" and f.get("total_range95_mm"):
        span = rep["span_years"]
        assert lo * span == pytest.approx(f["total_range95_mm"][0], abs=1.5)
    assert "95% range" in f["text"].split("Measured over the record")[0]
    assert "read somewhat stronger than the ground there truly moved" in f["text"]
    t = _text(cr.page(built.cfg, rep, built.figs))
    summary = t.split("Details are under Observations")[0]
    assert f"{abs(lo) / 25.4:.2f}" in summary or f"{abs(hi) / 25.4:.2f}" in summary
    assert "A feature's outline is drawn where the measurements go furthest" in t
    lines = cr.pdf_box_lines(rep, [(1, f)])
    assert lines.startswith("Numbered boxes on the map: 1 = Feature 1: ") and "95% range" in lines
    assert "read somewhat stronger" in lines


def test_mine_ground_average_carries_its_range():
    rep = {"features": [], "mine_ground": [{"key": "mapped", "label": "mine areas", "area_km2": 87.5,
                                             "footprint_km2": 89.4, "relative_vertical_mm_yr": -2.0, "testable": True,
                                             "detected": False, "range95_mm_yr": [-7.4, 3.0]}]}
    obs = ri.observations(rep)
    txt = next(o["text"] for o in obs if o["kind"] == "mines")
    assert "-0.08 in/yr (-2.0 mm/yr; 95% range -0.29 to +0.12 in/yr)" in txt


def test_the_report_records_and_prints_a_revision(built):
    """M3: a reissue says so - 'Revision 1, replacing the report issued 28
    Sep 2026' (config.report_revision's own note) and the changes, in the
    record, on the page and in the PDF (its own first page and a line on
    every area page)."""
    cfg = built.cfg
    assert "report_revision" in built.rep and built.rep["report_revision"] is None     # a first issue
    rv = {"number": 1, "supersedes": "2026-09-28",
          "changes": ["Detection limits are now for 4 detections in 5.", "Feature 1 is listed as worth a look."]}
    got = cr.report_revision(types.SimpleNamespace(report_revision=rv))
    assert got["heading"] == "Revision 1, replacing the report issued 28 Sep 2026"
    assert got["changes"] == rv["changes"]
    assert cr.report_revision(types.SimpleNamespace(report_revision=None)) is None
    assert cr.report_revision(types.SimpleNamespace()) is None and cr.report_revision(None) is None
    import datetime
    assert cr.report_revision(types.SimpleNamespace(report_revision=dict(rv, supersedes=datetime.date(2026, 9, 28))))[
        "supersedes_text"] == "28 Sep 2026"
    old = getattr(cfg, "report_revision", None)
    object.__setattr__(cfg, "report_revision", rv)
    try:
        # the record carries the same note the page prints (release reads it there)
        rec = cr.compute(cfg)[0]["report_revision"]
        assert rec["heading"] == got["heading"] and rec["changes"] == rv["changes"]
        rep = _rep(built)
        t = _text(cr.page(cfg, rep, built.figs))
        assert "Revision 1, replacing the report issued 28 Sep 2026." in t
        assert "Feature 1 is listed as worth a look." in t
        texts, _ = _record_pdf_text(built, rep, cfg)
        assert any("Revision 1, replacing the report issued 28 Sep 2026" in s for s in texts)
        assert any("Detection limits are now for 4 detections in 5." in s.replace("\n", " ") for s in texts)
        assert sum("Revision 1, replacing" in s for s in texts) == 1 + len(rep["areas"])
    finally:
        object.__setattr__(cfg, "report_revision", old)
    assert "replacing the report issued" not in cr.page(cfg, _rep(built), built.figs)


def test_page_words_from_the_final_check(built):
    rep = _rep(built)
    t = _text(cr.page(built.cfg, rep, built.figs))
    foot = t.split("Data sources and credits")[-1]
    assert "route: official" not in foot and "on request" not in foot
    assert "kept with the project" in foot
    assert "Every figure is an estimate with the uncertainty stated beside it" not in t
    assert "area, feature and mine-ground figures and every total carry a 95% range" in t
    assert "1 sigma The typical size of a figure's error" in t
    assert "every GNSS-anchored figure (the report areas in the table above)" in t
    assert "is added to every figure above" not in t
    rep["map_smoothing_m"] = 496
    assert "averaged over about 500 m" in _text(cr.page(built.cfg, rep, built.figs))
    # no area tested: no detection-limit words
    for a in rep["areas"]:
        if a.get("kind") == "area":
            a.update(testable=False, detection_limit_mm_yr=None, not_testable_reason="too little farmland")
    t = _text(cr.page(built.cfg, rep, built.figs))
    lims = t.split("Limitations")[1]
    assert "below the detection limits" not in lims and "no detection limit can be stated" in lims
    assert "Detection limit The sustained" not in t and "Signal ÷ threshold An area" not in t
    assert "The likely ranges come from" in t and "ranges and detection limits come from" not in t


def test_methods_and_station_words():
    rep = {"rd": None}
    reason = "every interferogram of this date failed the reference-pixel check"
    assert cr._plain_drop_reason(reason) == "every interferogram of this date could not be tied to the reference point"
    chk = {"site": "LAMC", "verdict": "not testable", "consistent": None,
           "verdict_reason": "Not testable: over the 2.2 years the station and the radar record share, this check "
                             "could only see a difference of about 1.08 in/yr."}
    sc = cr.station_check(chk, rep)
    assert "which allows a direct check" not in sc["html"] and "could not be checked against it" in sc["html"]
    assert sc["agrees"] is None


def test_quality_notes_say_what_the_second_check_does_and_good_practice():
    Q = {"checks": {"coverage": {"aoi_percent_at_threshold": 97.0, "reliable_percent": 43.1,
                                 "second_gate": {"applied": False}},
                    "reference_stability": {"pass": False, "value_mm_yr": -3.1, "yardstick_mm_yr": 1.0}}}
    notes = ri.quality_notes(Q, {"anchored": True, "detection_limits_mm_yr": [7.0]}, 0.78)
    assert any("All measured points are used in the area figures" in n for n in notes)
    ref = next(n for n in notes if n.startswith("The reference point drifts"))
    assert "good practice is under 0.05 in/yr, 1.3 mm/yr)" in ref and "field practice" not in ref
    Q["checks"]["coverage"]["second_gate"]["applied"] = True
    assert any("Only points that pass both are used" in n for n in ri.quality_notes(Q, {}, 0.78))


def test_weaker_only_when_every_patch_is_slower():
    """Site A: two of the 'weaker' patches averaged faster than Feature 1
    (-18.8 and -17.8 against -16.3 mm/yr)."""
    rep = {"features": [{"sign": -1, "area_km2": 23.7, "lat": 38.24, "lon": -87.65, "mean_rel_vertical_mm_yr": -16.3,
                         "vertical_relative_mm_yr": -16.3, "text": "x"}],
           "features_within_chance": [{"mean_rel_vertical_mm_yr": -18.8, "area_km2": 1.89},
                                      {"mean_rel_vertical_mm_yr": -17.8, "area_km2": 2.59}]}
    o = next(o for o in ri.observations(rep) if "within chance" in o["title"])
    assert o["title"] == "2 smaller patches within chance" and "weaker" not in o["title"]
    assert "up to 0.74 in/yr, over areas up to 2.6 km²" in o["text"]
    rep["features_within_chance"] = [{"mean_rel_vertical_mm_yr": -8.0, "area_km2": 30.0}]
    assert next(o for o in ri.observations(rep) if "within chance" in o["title"])["title"] == "1 weaker patch within chance"
    rep["features_within_chance"] = [{"mean_rel_vertical_mm_yr": -18.0, "area_km2": 30.0}]
    assert next(o for o in ri.observations(rep) if "within chance" in o["title"])["title"] == "1 other patch within chance"


def test_signed_never_prints_a_signed_zero():
    """0.5 at nd=0 rounds half to even, to 0: it printed '+0' (Site A's
    years table, '(+0 mm)' beside '(0 mm)')."""
    assert ri.signed(0.5, 0) == "0" and ri.signed(-0.5, 0) == "0" and ri.signed(0.6, 0) == "+1"
    assert ri.signed(0.004) == "0.00" and ri.signed(-0.004) == "0.00" and ri.signed(-0.006) == "-0.01"


def test_years_table_in_inches_with_mm_to_a_tenth(built):
    rep = _rep(built)
    rep["by_years_since_mining_frame"] = "vertical"
    rep["by_years_since_mining"] = [{"years_since_mining": 1, "n_areas": 19, "median_change_mm_per_yr": 0.5,
                                     "q25_mm": -1.0, "q75_mm": 5.2}]
    t = _text(cr.page(built.cfg, rep, built.figs))
    row = t.split("Middle half")[1]
    assert "+0.02 in +0.5 mm" in row and "-0.04 to +0.20 in -1.0 to +5.2 mm" in row and "(+0 mm)" not in row


def test_feature_charts_draw_the_line_their_total_is_read_on():
    """A feature's chart had no thick line; its thin line, zeroed on the
    fitted start, began at +0.93 in on Site B Feature 1. The line its
    total is read on is now drawn from zero: the average rate for a rate x
    span total, the fitted curve for a pace-change total (ending at it)."""
    t = np.arange(0, 10, 12 / 365.25)
    y = -250.0 * (1.0 - np.exp(-t / 1.5)) + np.random.default_rng(0).normal(0, 2.0, t.size)
    ch = sg.record_change(t, y)
    path = cr.fitted_path(t, y)
    assert path[0] == 0.0 and path[-1] == pytest.approx(ch["piecewise_change_mm"], abs=0.2)
    it = {"feature": True, "rate": -26.0, "total_method": "rate_x_span", "fit_vertical_mm": None}
    line, label = cr.feature_line(it, t)
    assert label == "its average rate" and line[0] == 0.0 and line[-1] == pytest.approx(-26.0 * t[-1])
    it = dict(it, total_method="piecewise_linear_yearly", fit_vertical_mm=list(path))
    line, label = cr.feature_line(it, t)
    assert label.startswith("fitted curve") and line[-1] == pytest.approx(path[-1])
    assert cr.feature_line({"feature": False, "rate": -5.0}, t) is None
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from datetime import datetime, timedelta
    dt_ = [datetime(2016, 1, 1) + timedelta(days=365.25 * x) for x in t]
    fig, ax = plt.subplots()
    item = {"name": "Feature 1", "series_vertical_mm": list(y), "rate": -26.0, "centre_rate": 0.0, "cone_rate": 7.0,
            "noise_mm": 5.0, "decision": False, "feature": True, "relative": True, "total_method": "rate_x_span",
            "fit_vertical_mm": None}
    cr._series_axes(ax, dt_, t, item, 0.0)
    thick = [ln for ln in ax.get_lines() if ln.get_label() == "its average rate"]
    assert thick and thick[0].get_ydata()[0] == 0.0
    plt.close(fig)


def test_summary_table_footnote_says_what_the_whole_areas_ratio_is(built, monkeypatch):
    from matplotlib.figure import Figure
    seen = []
    orig = Figure.text
    monkeypatch.setattr(Figure, "text", lambda self, x, y, s, *a, **k: (seen.append(str(s)), orig(self, x, y, s, *a, **k))[1])
    cr.summary_table(_rep(built), built.D / "figures" / "last_summary.png")
    note = " ".join(s.replace("\n", " ") for s in seen)
    assert "for the whole area: its rate, satellite plus GNSS, ÷ its 95% uncertainty" in note
    assert "the regional rate ÷ its 95%" not in note
    assert "Each feature's outline was drawn from these same measurements" in note


def test_pdf_explains_its_boxes_and_has_a_colour_scale(built):
    rep = _rep(built)
    f = rep["features"][0]
    w, s_, e_, n_ = f["bbox"]
    assert cr.pdf_boxes(rep, (w - 0.01, e_ + 0.01, s_ - 0.01, n_ + 0.01)) == [(1, f)]
    assert cr.pdf_boxes(rep, (w + 5, e_ + 5, s_, n_)) == [] and cr.pdf_boxes(rep, None) == []
    assert cr.pdf_box_lines(rep, []) == ""
    texts, axes = _record_pdf_text(built, rep)
    # the colour bar under each area's map (one per page with a map)
    assert any(r[1] == pytest.approx(0.452) and r[0] == pytest.approx(0.55) for r in axes)


# ------------------------------------------------------------------ the review of this group's fixes
def test_no_onset_words_when_the_years_before_moved_the_other_way():
    """M1's test is two-sided: years before the onset beyond farmland on the
    OTHER side (rising, then sinking) are not ordinary, so no "began" - and
    "it runs through the whole record", which the first draft fell back to,
    is not true of it either: no onset words at all."""
    t = np.arange(2016.0, 2026.0, 12 / 365.25)
    years = [int(x) for x in t]
    rates = {2016: 15.0, 2017: 15.0, 2018: -40.0, 2019: -60.0, 2020: -60.0, 2021: -50.0, 2022: -45.0,
             2023: -40.0, 2024: -40.0, 2025: -40.0}
    rows = np.random.default_rng(4).normal(0, 5.7, (40, 10))
    d = ri.trend_shape(t - 2016.0, _series_from_yearly(t, years, rates), years, noise_mm_yr=1.0,
                       null_yearly={"years": sorted(set(years)), "rates": rows})
    assert d["contrasts"]["before_onset"]["z"] < -ri.CONTRAST_Z
    assert "began" not in d["text"] and "runs through the whole record" not in d["text"]
    assert "already" not in d["text"] and d["text"]
    assert d["moved_the_other_way_before"] == 2018


def test_a_looser_second_gate_is_not_said_to_keep_only_points_passing_both():
    """'52% passed both quality checks' counts the second check at 0.7; a
    config gate of 0.5 leaves some failing points in the figures, so 'Only
    points that pass both are used' would be false."""
    Q = {"checks": {"coverage": {"aoi_percent_at_threshold": 88.0, "reliable_percent": 43.1,
                                 "second_gate": {"applied": True, "applied_threshold": 0.5}}}}
    notes = ri.quality_notes(Q, {}, 0.78)
    assert not any("Only points that pass both" in n for n in notes)
    assert any("leave out only the points that fail a looser form of the second check" in n for n in notes)
    Q["checks"]["coverage"]["second_gate"]["applied_threshold"] = 0.7
    assert any("Only points that pass both are used" in n for n in ri.quality_notes(Q, {}, 0.78))


def test_a_patch_that_is_a_feature_gives_the_features_range():
    """M4: the patches table printed a feature's rate with no range."""
    feats = [{"vertical_relative_mm_yr": -37.3, "mean_rel_vertical_mm_yr": -35.0,
              "rate_range95_mm_yr": [-46.4, -29.0]}]
    p = {"area": "x", "mean_rel_vertical_mm_yr": -35.1, "feature": 1}
    rate, what = cr.patch_rate_cells(p, feats, 500)
    assert rate == "-1.47 in/yr (Feature 1's figure)"
    assert what == ("Feature 1 (Observations); its 95% range -1.83 to -1.14 in/yr; the 500 m map averages "
                    "-1.38 in/yr over this patch")
    _, what = cr.patch_rate_cells(p, [dict(feats[0], rate_range95_mm_yr=None)], 500)
    assert "95% range" not in what
