"""Decision D1 of 7 Oct 2026 (finding #10 of the second review): a tested
report area's 95% range is on the test's own frame.

The reading compares the area's EXCESS over the farmland null median with the
empirical 95% threshold; the range was the printed rate +/- 1.96 x the spread
for ranges. Near the threshold "Movement detected" could come with a range
that included 0 (no direction word), or "Within normal ground variation" with
one that excluded it ("rising"). Now (client_report.tested_range):

  relative part   excess +/- threshold (vertical)
  anchored        + the GNSS background on both ends, half-width
                  hypot(threshold, 1.96 x its uncertainty)
  1 sigma         half-width / 1.96
  printed rate    unchanged (the measured mean)

so for a relative-only area "the range excludes 0" <=> "Movement detected",
always; an anchored area the test did not detect whose range excludes 0
"moves with the surrounding region" (no colour, no 'stands out'). The
Results workbook's per-year rows and its chart band follow the same rule.
Each test failed on the code before the change.

Run: python -m pytest insar_pipeline/tests/test_fix_oct7_dec_D1_range.py -q -p no:cacheprovider
"""
import copy
import json
import re
import sys
import types
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_fix_L5 as L5                                          # noqa: E402  (the client-report project builder)
import test_fix_L9b as L9b                                        # noqa: E402  (the Results project builder)

from insar_pipeline import client_report as cr                   # noqa: E402
from insar_pipeline import significance as sg                    # noqa: E402

T = np.linspace(0.0, 9.0, 60)
COV = {"measured_percent": 90.0, "coverage_balanced": True}
# Area A2's numbers on the Site A record (vertical; up = 1 here)
SD, THR, MED = 7.08, 13.5, 0.92


def _summ(rate):
    return {"rate_mm_yr": rate, "total_mm": rate * 9.0, "series_mm": list(rate * T), "start_mm": 0.0}


def _sig(value, med=MED, thr=THR, sd=SD, det=None):
    """significance.assess's dict for an area of mean `value` (LOS mm/yr)."""
    x = value - med
    return {"testable": True, "farmland_spread_1sigma_mm_yr": sd, "spread_for_ranges_mm_yr": sd,
            "threshold_95_mm_yr": thr, "null_median_mm_yr": med, "excess_mm_yr": round(x, 2),
            "detectable_80_mm_yr": 19.7, "value_mm_yr": value, "signal_to_threshold": round(abs(x) / thr, 2),
            "distinguishable_from_farmland": (abs(x) >= thr) if det is None else det}


def _area(value, up=1.0, bg_v=None, bg_u=None, name="Area A2", **kw):
    return cr.area_entry(name, "x", "area", COV, _summ(value), _sig(value, **kw), up, bg_v, bg_u, 9.0, T)


def _excludes0(r):
    return bool(r) and (r[0] > 0 or r[1] < 0)


# ------------------------------------------------------------------ the record's own fields
def test_within_normal_variation_no_longer_has_a_range_that_excludes_zero():
    """The review's case: value +14.0, excess 13.08 < threshold 13.5. It read
    'Within normal ground variation' with [0.12, 27.88] and 'rising'."""
    a = _area(14.0)
    assert a["reading"] == "Within normal ground variation" and a["detected"] is False
    lo, hi = a["range95_mm_yr"]
    assert lo <= 0 <= hi
    assert (lo, hi) == pytest.approx((14.0 - MED - THR, 14.0 - MED + THR), abs=0.01)
    assert cr.area_word(a) == ""
    assert a["vertical_mm_yr"] == 14.0                      # the printed rate is not moved


def test_movement_detected_has_a_range_that_excludes_zero_and_a_word():
    """value -13.0, excess -13.92: detected, and the range was [-26.88, +0.88]
    (no direction word beside 'Movement detected')."""
    a = _area(-13.0)
    assert a["reading"] == "Movement detected"
    assert _excludes0(a["range95_mm_yr"]) and a["range95_mm_yr"][1] < 0
    assert cr.area_word(a) == "sinking"


def test_range_is_excess_plus_minus_threshold_and_one_sigma_is_its_half_width():
    """Vertical, from LOS with up = 0.77: Area A2 moves from about
    [-19.5, +8.2] to excess +/- threshold, and the 1 sigma is half / 1.96."""
    up = 0.77
    a = _area(-5.64 * up, up=up)
    ex, thr = (-5.64 * up - MED) / up, THR / up
    assert a["range95_mm_yr"] == pytest.approx([ex - thr, ex + thr], abs=0.01)
    assert a["vertical_uncertainty_1sigma_mm_yr"] == pytest.approx(thr / 1.96, abs=0.01)
    lo, hi = a["range95_mm_yr"]
    assert (hi - lo) / 2 == pytest.approx(1.96 * a["vertical_uncertainty_1sigma_mm_yr"], abs=0.02)
    assert a["vertical_mm_yr"] == pytest.approx(-5.64, abs=0.01)


def test_relative_only_range_excludes_zero_exactly_when_detected():
    """Many areas around the threshold, nulls not centred on 0, thresholds
    not 1.96 sd: 'excludes 0' <=> detected, and the word only then."""
    rng = np.random.default_rng(3)
    n_det = n_not = 0
    for _ in range(3000):
        thr = float(rng.uniform(0.5, 20.0))
        med = float(rng.uniform(-1.5, 1.5))
        sd = float(thr / rng.uniform(1.5, 2.6))
        up = float(rng.uniform(0.6, 1.0))
        value = med + float(rng.choice([-1, 1])) * thr * float(rng.uniform(0.9, 1.1))
        a = cr.area_entry("A", "x", "area", COV, _summ(value), _sig(value, med=med, thr=thr, sd=sd), up,
                          None, None, 9.0, T)
        assert _excludes0(a["range95_mm_yr"]) == a["detected"], (value, med, thr, a["range95_mm_yr"])
        assert bool(cr.area_word(a)) == a["detected"]
        n_det += a["detected"]
        n_not += not a["detected"]
    assert n_det > 500 and n_not > 500


@pytest.mark.parametrize("excess, det", [(13.5, True), (-13.5, True), (13.51, False), (-13.51, False)])
def test_rounding_alone_never_contradicts_the_reading(excess, det):
    """At the threshold itself (|excess| == threshold is detected) the inner
    end is 0.00 after rounding; with the stored 0.01-rounded excess a hair
    over a threshold the test did not pass, it is +0.01. The inner end goes
    to the test's side of 0."""
    a = _area(excess + MED, det=det)
    assert a["detected"] is det
    assert _excludes0(a["range95_mm_yr"]) is det
    lo, hi = a["range95_mm_yr"]
    assert min(abs(lo), abs(hi)) <= 0.02


def test_anchored_range_adds_the_gnss_background_and_its_uncertainty():
    bg_v, bg_u = -2.0, 0.5
    a = _area(-5.0, bg_v=bg_v, bg_u=bg_u)
    half = float(np.hypot(THR, 1.96 * bg_u))
    centre = (-5.0 - MED) + bg_v
    assert a["range95_mm_yr"] == pytest.approx([centre - half, centre + half], abs=0.01)
    assert a["vertical_uncertainty_1sigma_mm_yr"] == pytest.approx(half / 1.96, abs=0.01)
    assert a["vertical_mm_yr"] == pytest.approx(-5.0 + bg_v, abs=0.01)
    # no GNSS uncertainty: no range rather than a narrower one
    b = _area(-5.0, bg_v=bg_v, bg_u=None)
    assert b["range95_mm_yr"] is None and b["vertical_uncertainty_1sigma_mm_yr"] is None


def test_anchored_area_that_moves_only_with_the_region_says_so():
    """Not detected (excess -12.92 < 13.5), but the regional -2.0 puts the
    absolute range below 0: it moves with the surrounding region."""
    a = _area(-12.0, bg_v=-2.0, bg_u=0.5)
    assert a["detected"] is False and a["range95_mm_yr"][1] < 0
    assert a["reading"] == "Within normal ground variation; moves with the surrounding region"
    assert cr.moves_with_region(a) and cr.area_word(a) == "sinking"
    # the same area, detected or relative-only, never carries the note
    assert "surrounding region" not in _area(-13.0, bg_v=-2.0, bg_u=0.5)["reading"]
    assert "surrounding region" not in _area(-12.0)["reading"]


def test_not_testable_area_still_has_no_range():
    sig = {"testable": False, "not_testable_reason": "Too little ordinary farmland around this area.",
           "approx_independent_placements": 3}
    a = cr.area_entry("Zone", "x", "area", COV, _summ(-3.0), sig, 1.0, -2.0, 0.5, 9.0, T)
    assert a["range95_mm_yr"] is None and a["vertical_uncertainty_1sigma_mm_yr"] is None
    assert cr.area_word(a) == "" and not cr.moves_with_region(a)


def test_significance_docstring_no_longer_claims_the_spread_range_agrees():
    doc = sg.assess.__doc__
    assert "tested_range" in doc
    assert "so the reading and a 95% range built from\n    spread_for_ranges agree" not in doc


# ------------------------------------------------------------------ the page and the PDF lines
@pytest.fixture(scope="module")
def built(tmp_path_factory):
    """The L5 synthetic project (GNSS-anchored), its report computed once."""
    cfg = L5.make_project(tmp_path_factory.mktemp("d1"))
    rep, ctx, masks, outlines = cr.compute(cfg)
    return types.SimpleNamespace(cfg=cfg, rep=json.loads(json.dumps(rep)), ctx=ctx)


def _with_area(built, a, anchored=True):
    rep = copy.deepcopy(built.rep)
    a = json.loads(json.dumps(a))
    a["name"] = a["display_name"] = "Test Field"
    rep["areas"] = [a if x["name"] == "Test Field" else x for x in rep["areas"]]
    if not anchored:
        rep["anchored"] = False
        rep["gnss"] = {}
        rep["areas"] = [x for x in rep["areas"] if x.get("kind") != "full"]
    return rep


def _row(html, name="Test Field"):
    m = re.search(r'<tr><th scope="row">' + re.escape(name) + r"<br>.*?</tr>", html, re.S)
    assert m, name
    return L5._text(m.group(0))


def test_page_row_has_no_direction_word_beside_within_normal_variation(built):
    rep = _with_area(built, _area(14.0), anchored=False)
    row = _row(cr.page(built.cfg, rep, L5._fake_figs(built)))
    assert "Within normal ground variation" in row
    assert "rising" not in row and "sinking" not in row


def test_page_row_and_short_answer_name_the_direction_of_a_detected_area(built):
    rep = _with_area(built, _area(-13.0), anchored=False)
    html = cr.page(built.cfg, rep, L5._fake_figs(built))
    row = _row(html)
    assert "Movement detected" in row and "sinking" in row
    assert re.search(r"Test Field \(sinking, ", L5._text(html))
    assert 'class="big num down"' in re.search(r'<tr><th scope="row">Test Field<br>.*?</tr>', html, re.S).group(0)


def test_page_row_says_an_anchored_area_moves_with_the_region_not_on_its_own(built):
    rep = _with_area(built, _area(-12.0, bg_v=-2.0, bg_u=0.5))
    html = cr.page(built.cfg, rep, L5._fake_figs(built))
    row = _row(html)
    assert "Within normal ground variation" in row and "Movement detected" not in row
    assert "sinking with the region" in row
    assert "It is sinking with the surrounding region" in row and "not on its own" in row
    assert "stands out" not in row.split("Not distinguishable")[1].split("Sustained movement")[0]
    raw = re.search(r'<tr><th scope="row">Test Field<br>.*?</tr>', html, re.S).group(0)
    assert 'class="big num down"' not in raw            # not coloured as moving on its own
    # the PDF's numbers block and summary table say the same
    lines = dict(cr.one_pager_lines(rep, rep["areas"][[x["name"] for x in rep["areas"]].index("Test Field")]))
    assert lines["Vertical movement per year"].endswith("sinking with the surrounding region")
    assert lines["Reading"].endswith("moves with the surrounding region")
    rows = cr.summary_rows(rep)
    assert any(r[0].startswith("Test Field") and r[-1].endswith("moves with the surrounding region") for r in rows)


def test_one_pager_has_no_word_beside_within_normal_variation(built):
    rep = _with_area(built, _area(14.0), anchored=False)
    a = next(x for x in rep["areas"] if x["name"] == "Test Field")
    lines = dict(cr.one_pager_lines(rep, a))
    assert "rising" not in lines["Vertical movement per year"]
    lo, hi = a["range95_mm_yr"]
    assert f"({lo:+.1f} to {hi:+.1f} mm/yr)" in lines["Likely range (95%)"]


def test_the_projects_own_tested_area_is_on_the_tests_frame(built):
    """The real compute() path: Test Field's range is excess +/- threshold
    + the GNSS background (L5's tie), not the rate +/- 1.96 x the spread."""
    a = next(x for x in built.rep["areas"] if x["name"] == "Test Field")
    assert a["testable"] is True
    half = float(np.hypot(a["decision_threshold_mm_yr"], 1.96 * L5.BG_U))
    centre = a["excess_vertical_mm_yr"] + L5.BG_V
    assert a["range95_mm_yr"] == pytest.approx([centre - half, centre + half], abs=0.08)
    assert a["vertical_uncertainty_1sigma_mm_yr"] == pytest.approx((a["range95_mm_yr"][1] - a["range95_mm_yr"][0])
                                                                   / 2 / 1.96, abs=0.01)


# ------------------------------------------------------------------ the Results workbook
def _fake_assess(real):
    """significance.assess with a tested result whose excess is 1.02 x a
    threshold of 10 mm/yr (detected), and a spread so wide that the old range
    (value +/- 1.96 x 10) always included 0."""
    def assess(field, far, region, *a, **k):
        out = real(field, far, region, *a, **k)
        if not out.get("testable"):
            return out
        value = float(out["value_mm_yr"])
        out.update({"farmland_spread_1sigma_mm_yr": 10.0, "spread_for_ranges_mm_yr": 10.0,
                    "threshold_95_mm_yr": 10.0, "null_median_mm_yr": round(value - 10.2, 2), "excess_mm_yr": 10.2,
                    "signal_to_threshold": 1.02, "distinguishable_from_farmland": True})
        return out
    return assess


@pytest.mark.parametrize("gnss", [False, True])
def test_results_rows_and_band_follow_the_reports_range(tmp_path, monkeypatch, gnss):
    from insar_pipeline import results as rs
    cfg = L9b.make_project(tmp_path, gnss=gnss)
    monkeypatch.setattr(sg, "assess", _fake_assess(sg.assess))
    res = rs.analyse(cfg, "Test Field")
    s = res.area_summary
    # the test's frame, vertical inches: excess / up (+ GNSS) +/- hypot(threshold / up, 1.96 x GNSS sigma)
    bg_v, bg_u = (L9b.BG_V, L9b.BG_U) if gnss else (0.0, 0.0)
    half = float(np.hypot(10.0 / L9b.UP, 1.96 * bg_u))
    want = [(10.2 / L9b.UP + bg_v - half) / 25.4, (10.2 / L9b.UP + bg_v + half) / 25.4]
    rng = s["vertical_95pct_range_in_yr"]
    assert s["reading"] == "Movement detected"
    assert rng == pytest.approx(want, abs=1e-3)
    if not gnss:
        assert rng[0] > 0                                   # detected <=> the range excludes 0
    # the per-year rows: the same rule on each year's own test
    rows = [r for r in res.annual if not r["area_reading"].startswith("not")]
    assert rows
    for r in rows:
        assert r["area_reading"] == "stands out"
        assert [r["area_mean_95pct_low_in_yr"], r["area_mean_95pct_high_in_yr"]] == pytest.approx(want, abs=1e-3)
    # the chart band is the area's own range x time, as the KMZ chart draws it
    tw = np.array([(np.datetime64(d) - np.datetime64(res.series["date"][0])).astype(float) / 365.25
                   for d in res.series["date"]])
    assert res.series["range95_low_in"][-1] == pytest.approx(rng[0] * tw[-1], abs=2e-3)
    assert res.series["range95_high_in"][-1] == pytest.approx(rng[1] * tw[-1], abs=2e-3)


def test_results_year_that_moves_only_with_the_region_says_so(tmp_path, monkeypatch):
    """Anchored, not detected, and the background alone puts the year's
    absolute range below 0: 'within normal variation; moves with the
    surrounding region'."""
    from insar_pipeline import results as rs
    cfg = L9b.make_project(tmp_path, gnss=True)
    real = sg.assess

    def assess(field, far, region, *a, **k):
        out = real(field, far, region, *a, **k)
        if out.get("testable"):
            value = float(out["value_mm_yr"])
            # excess -0.9 x a 1 mm/yr threshold: not detected; the -1.5 mm/yr
            # background (L9b's tie) puts excess + background well below 0
            out.update({"farmland_spread_1sigma_mm_yr": 0.5, "spread_for_ranges_mm_yr": 0.5,
                        "threshold_95_mm_yr": 1.0, "null_median_mm_yr": round(value + 0.9, 2), "excess_mm_yr": -0.9,
                        "signal_to_threshold": 0.9, "distinguishable_from_farmland": False})
        return out
    monkeypatch.setattr(sg, "assess", assess)
    res = rs.analyse(cfg, "Test Field")
    rows = [r for r in res.annual if not r["area_reading"].startswith("not")]
    assert rows
    for r in rows:
        assert r["area_reading"] == "within normal variation; moves with the surrounding region", r
        assert r["area_mean_95pct_high_in_yr"] < 0
    assert res.area_summary["reading"] == "Within normal ground variation; moves with the surrounding region"


def test_results_year_chart_draws_a_range_the_bar_lies_outside(tmp_path, monkeypatch):
    """A tested year's range is centred on the excess over the null median,
    not on the year's figure, so the figure can lie outside it (here the null
    median is 10.2 below the area's value: every year 'stands out' with a
    range above 0 while the figure is near 0). The chart drew the range as an
    errorbar about the bar and the export failed with "'yerr' must not
    contain negative values"; it now draws the range from its own ends."""
    from insar_pipeline import results as rs
    cfg = L9b.make_project(tmp_path, gnss=False)
    monkeypatch.setattr(sg, "assess", _fake_assess(sg.assess))
    res = rs.analyse(cfg, "Test Field")
    rows = [r for r in res.annual if r.get("area_mean_95pct_low_in_yr") is not None]
    assert any(not (r["area_mean_95pct_low_in_yr"] <= r["area_mean_vertical_in_yr"] <= r["area_mean_95pct_high_in_yr"])
               for r in rows)
    rs._chart(res, tmp_path / "chart.png")
    assert (tmp_path / "chart.png").stat().st_size > 0
