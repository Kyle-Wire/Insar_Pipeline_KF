"""Follow-up decisions of 7 Oct 2026, group features (report_insights.py,
client_report.py):

F1     leave-candidate-out: the first pass's strongest candidates are each
       re-tested on farmland with that candidate (+500 m) taken out, the
       simulated copies built from that farmland, under the unchanged
       selection-corrected rule; one that passes is cut and the second pass
       runs (client_report._leave_candidate_out, report_insights.
       leading_candidates / match_candidate / leave_out_result)
F3     the feature null uses 199 simulated copies (p resolution 1 in 200):
       measured on the Site B copy, 59 -> 199 adds well under 10 minutes
       to the products run
F7(a)  "the same as chance" for the cells standing out only inside chance's
       95% binomial range for the study area's radar cells (not 1.5 points)
F7(c)  "fastest in <years>" is never named when fewer than 10 placements
       back the test (the approximate yardstick)

Each test checks an outcome on synthetic data with a known answer and fails
on the code before this round (backups/pre_followup_20261007/).

Run: python -m pytest insar_pipeline/tests/test_fix_oct7_fu_features.py -q -p no:cacheprovider
"""
import types

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from insar_pipeline import client_report as cr
from insar_pipeline import report_insights as ri
from insar_pipeline import significance as sg
from insar_pipeline.panel_history import measure_farmland


# ------------------------------------------------------------------ helpers
def _disc_ctx(depth, radius_cells, n=300, seed=4, pixel=60.0):
    """Correlated farmland noise (sd 3 mm/yr, ~360 m correlation) on an
    18 km square, all of it farmland and searched, with a flat sinking disc
    of `depth` mm/yr - unmapped, so inside the farmland it is judged against."""
    rng = np.random.default_rng(seed)
    f = gaussian_filter(rng.normal(0, 1, (n, n)), 6.0)
    f = f / f.std() * 3.0
    yy, xx = np.mgrid[:n, :n]
    f = f - depth * (((yy - n // 2) ** 2 + (xx - n // 2) ** 2) <= radius_cells ** 2)
    dlat = pixel / 111320.0
    dlon = pixel / (111320.0 * np.cos(np.radians(38.3)))
    attrs = {"X_FIRST": "-87.6", "Y_FIRST": "38.4", "X_STEP": str(dlon), "Y_STEP": str(-dlat)}
    good = np.ones((n, n), bool)
    tc = np.full((n, n), 0.95)
    ctx = types.SimpleNamespace(sg=sg, attrs=attrs, good=good, vel=f, tc=tc, kcls=sg.coherence_class(tc),
                                far=good.copy(), far_wide=good.copy(), aoi=good.copy(), cfg=None)
    measure_farmland(ctx)
    return ctx


# ------------------------------------------------------------------ F1: leave-candidate-out
def test_a_large_patch_hidden_by_its_own_farmland_passes_when_left_out(monkeypatch):
    """Site A Feature 1's mechanism: a 17.6 km² disc 8 mm/yr deep inside
    the farmland puts its own values in the copies, so on the first pass 191
    of 199 copies are as strong as it (never listed, never cut). Left out of
    the farmland it beats every copy (71.4 against at most 7.3) and passes,
    also corrected for the 3 candidates re-tested (G2: p 0.005, adjusted
    0.015; with 59 copies, as until G2, no re-test of 2 or 3 can pass); the
    context's farmland is put back afterwards."""
    monkeypatch.setattr(ri, "N_SURROGATES", 199)   # G2: with 59 copies no re-test of 2-3 can pass
    ctx = _disc_ctx(8.0, 40)
    far0, wide0, means0 = ctx.far, ctx.far_wide, dict(ctx.class_means)
    sm, cands, null, ok = cr._feature_search(ctx, 1.0, {}, ctx.aoi, [])
    lead = ri.leading_candidates(cands, sm)
    big = lead[0]
    assert ok and big["area_km2"] > 15 and big["stands_out"] is False
    assert big["chance"]["copies_as_strong"] >= 30                      # tested against itself
    rec = cr._leave_candidate_out(ctx, sm, cands, null, 1.0, {}, ctx.aoi, [])
    assert rec["tested"] and rec["leading"] == ri.LEAVE_OUT_LEADING == 3 and rec["buffer_m"] == 500.0
    assert len(rec["candidates"]) == len(lead) <= 3
    r = rec["candidates"][0]
    assert r["first_pass"]["strength"] == pytest.approx(ri.cluster_mass(big, sm), abs=0.01)
    assert r["first_pass"]["copies_percentile"] < 20
    lo = r["left_out"]
    assert r["passed"] is True and lo["passed"] is True and lo["copies_as_strong"] == 0
    assert lo["strength"] > 5 * lo["copies_max"] and lo["copies_percentile"] == 100.0
    assert rec["passed"] >= 1 and big["leave_out"] is r
    # ordinary noise patches stay within chance with themselves left out
    assert all(x["passed"] is False for x in rec["candidates"][1:])
    # the context is the first pass's again - nothing added either (the
    # light context had no farmland_pass; the re-test set one to "second")
    assert ctx.far is far0 and ctx.far_wide is wide0 and ctx.class_means == means0
    assert not hasattr(ctx, "farmland_pass")


def test_a_candidate_beating_chance_on_the_first_pass_is_not_rerun(monkeypatch):
    """A 9.9 km² disc that already beats every copy: recorded from the first
    pass (it is cut anyway), no second search for it."""
    monkeypatch.setattr(ri, "N_SURROGATES", 59)
    ctx = _disc_ctx(8.0, 30)
    sm, cands, null, _ = cr._feature_search(ctx, 1.0, {}, ctx.aoi, [])
    calls = []
    orig = cr._feature_search

    def spy(*a, **k):
        calls.append(1)
        return orig(*a, **k)
    monkeypatch.setattr(cr, "_feature_search", spy)
    rec = cr._leave_candidate_out(ctx, sm, cands, null, 1.0, {}, ctx.aoi, [])
    r = rec["candidates"][0]
    assert r["passed"] is True and r["left_out"] is None and "first pass" in r["basis"]
    assert r["first_pass"]["copies_as_strong"] == 0 and r["first_pass"]["copies_percentile"] == 100.0
    assert len(calls) == len(rec["candidates"]) - 1                    # only the others are re-run


def test_a_candidate_outside_the_farmland_takes_the_first_pass_figures(monkeypatch):
    """A leading candidate that does not reach the farmland even with its
    500 m buffer (Site A's third, on mapped mine ground): leaving it out
    changes nothing, so it is not searched again - and its recorded result
    is exactly what the re-run gives."""
    monkeypatch.setattr(ri, "N_SURROGATES", 59)
    ctx = _disc_ctx(0.0, 0)
    sm, cands, _, _ = cr._feature_search(ctx, 1.0, {}, ctx.aoi, [])
    big = ri.leading_candidates(cands, sm)[0]
    # the strongest chance patch's ground (+ 1.5 km) is mapped: no longer farmland
    out = ~sg.euclidean_buffer(big["_mask"], ctx.attrs, 1500.0)
    ctx.far, ctx.far_wide = ctx.far & out, ctx.far_wide & out
    measure_farmland(ctx)
    sm, cands, null, _ = cr._feature_search(ctx, 1.0, {}, ctx.aoi, [])
    f = ri.match_candidate(big, cands)
    grown = sg.euclidean_buffer(f["_mask"], ctx.attrs, 500.0)
    assert f is not None and f["stands_out"] is False and not (grown & (ctx.far | ctx.far_wide)).any()
    # what the re-test gives when made
    keep = {k: getattr(ctx, k) for k in ("far", "far_wide", "r_cm", "r_null", "r_place", "class_means", "class_tilt")}
    cr._exclude_from_farmland(ctx, [f["_mask"]], 500.0)
    sm2, cands2, null2, _ = cr._feature_search(ctx, 1.0, {}, ctx.aoi, [])
    made = ri.leave_out_result(ri.match_candidate(f, cands2), sm2, null2)
    for k, v in keep.items():
        setattr(ctx, k, v)
    del ctx.farmland_pass
    calls = []
    orig = cr._feature_search

    def spy(*a, **k):
        calls.append(1)
        return orig(*a, **k)
    monkeypatch.setattr(cr, "_feature_search", spy)
    rec = cr._leave_candidate_out(ctx, sm, [f], null, 1.0, {}, ctx.aoi, [])
    r = rec["candidates"][0]
    assert calls == [] and "outside the farmland" in r["basis"]
    assert r["left_out"] == made and r["passed"] is made["passed"] is False


def test_leave_out_helpers_match_and_fail_closed():
    a = np.zeros((10, 10), bool)
    a[2:6, 2:6] = True
    b = np.zeros((10, 10), bool)
    b[4:8, 4:8] = True
    c = np.zeros((10, 10), bool)
    c[3:7, 3:7] = True
    f = {"sign": -1, "_mask": a}
    assert ri.match_candidate(f, [{"sign": -1, "_mask": b}, {"sign": -1, "_mask": c}])["_mask"] is c
    assert ri.match_candidate(f, [{"sign": 1, "_mask": a}]) is None             # the other side
    assert ri.match_candidate(f, [{"sign": -1, "_mask": ~a}]) is None           # no overlap
    assert ri.leave_out_result(None, {}, {"tested": True})["passed"] is False
    got = ri.leave_out_result({"sign": -1}, {}, {"tested": False, "reason": "too little farmland"})
    assert got["passed"] is False and got["tested"] is False and "farmland" in got["reason"]
    assert ri.copies_percentile(10.0, [1.0, 5.0, 12.0, 20.0]) == 50.0
    assert ri.copies_percentile(None, [1.0]) is None and ri.copies_percentile(3.0, []) is None


@pytest.fixture(scope="module")
def hidden(tmp_path_factory):
    """L9b's synthetic project through compute with its settling patch made
    to fail the chance test whenever the farmland still holds it - the
    Site A case - and pass once it is out (the patch's own result)."""
    import test_fix_L9b as L9b
    mp = pytest.MonkeyPatch()
    mp.setattr(ri, "N_SURROGATES", 199)          # G2: with 59 copies no re-test of 2-3 can pass
    orig = cr._feature_search

    def self_hidden(ctx, *a, **k):
        sm, cands, null, ok = orig(ctx, *a, **k)
        for f in cands:
            if (f["_mask"] & ctx.far).sum() > 0.5 * f["_mask"].sum() and f["stands_out"]:
                f["stands_out"] = False
                f["chance"] = dict(f["chance"], stands_out=False, p=0.25, copies_as_strong=49, p_text="about 50 in 200")
        return sm, cands, null, ok
    mp.setattr(cr, "_feature_search", self_hidden)
    sent = []
    orig_cs = ri.cells_sentence

    def spy_cells(*a, **k):
        sent.append((a, k))
        return orig_cs(*a, **k)
    mp.setattr(ri, "cells_sentence", spy_cells)
    try:
        root = tmp_path_factory.mktemp("fu_features")
        cfg = L9b.make_project(root)
        rep, ctx, masks, outlines = cr.compute(cfg)
        import matplotlib
        matplotlib.use("Agg")
        figs = cr.figures(cfg, rep, ctx, masks, outlines, out_dir=cfg.project_dir / "deliverables" / "figures")
        html = cr.page(cfg, rep, figs)
    finally:
        mp.undo()
    return types.SimpleNamespace(rep=rep, html=html, sent=sent, L9b=L9b)


def test_compute_cuts_and_lists_a_patch_that_passes_only_when_left_out(hidden):
    """Before F1 the hidden patch was never listed (no second pass); now it
    passes its leave-out test, is cut, and the second pass lists it."""
    rep, L9b = hidden.rep, hidden.L9b
    lat, lon = L9b._lalo((L9b.FEATURE[0].start + L9b.FEATURE[0].stop) / 2,
                         (L9b.FEATURE[1].start + L9b.FEATURE[1].stop) / 2)
    assert len(rep["features"]) >= 1                                  # none before F1
    f = rep["features"][0]
    assert abs(f["lat"] - lat) < 0.01 and abs(f["lon"] - lon) < 0.01 and f["chance"]["stands_out"] is True
    sp = rep["farmland_second_pass"]
    assert sp["features_removed"] == 1 and sp["features_removed_on_leave_out"] == 1
    lo = rep["feature_leave_out"]
    passed = [r for r in lo["candidates"] if r["passed"]]
    assert lo["tested"] and len(passed) == 1 and passed[0]["left_out"]["passed"] is True
    assert passed[0]["first_pass"]["p"] == 0.25 and passed[0]["second_pass"]["stands_out"] is True
    # said on the page, in plain words
    assert "left out of the" in hidden.html and "cannot hide" in hidden.html


def test_page_passes_the_radar_cell_count_to_the_cells_sentence(hidden):
    """F7(a): the page's single-point sentence is judged on the study area's
    radar cells (two arguments until now)."""
    rep = hidden.rep
    assert hidden.sent, "the page did not word the cells standing out"
    a, k = hidden.sent[-1]
    n = a[2] if len(a) > 2 else k.get("n_cells")
    assert n == rep["independent_radar_measurements"] and n


# ------------------------------------------------------------------ F3: 199 copies
def test_feature_null_uses_199_copies_read_when_called(monkeypatch):
    """F3: 59 -> 199 copies (p resolution 1 in 200). Measured on the
    Site B copy: 32 s a null at 59, 47 s at 99, 89 s at 199 - with F1's
    re-tests 4 nulls a run, the report's compute 2.2 -> 6.7 minutes, under
    the 10 the rule allows. The setting is read when the null is made."""
    assert ri.N_SURROGATES == 199
    assert ri.p_text(0, ri.N_SURROGATES) == "less than 1 in 200"
    f = gaussian_filter(np.random.default_rng(3).normal(0, 1, (160, 160)), 4)
    f = f / f.std() * 3.0
    good = np.ones(f.shape, bool)
    attrs = {"X_FIRST": "-87.6", "Y_FIRST": "38.4", "X_STEP": str(60 / (111320.0 * np.cos(np.radians(38.3)))),
             "Y_STEP": str(-60 / 111320.0)}
    ctx = types.SimpleNamespace(sg=sg, attrs=attrs, good=good, far=good, far_wide=good, r_cm=f, r_null=f)
    sm = ri.smoothed_field(ctx, 1.0)
    monkeypatch.setattr(ri, "N_SURROGATES", 7)
    null = ri.feature_null(ctx, sm, [], good)
    assert null["n_surrogates"] == 7 and len(null["sides"]["sinking"]["max_mass"]) == 7
    # a feature beating every one of 199 copies: p = 1/200
    nul = {"tested": True, "sides": {"sinking": {"tested": True, "max_mass": [1.0] * 199}}}
    ch = ri.selection_p({"sign": -1, "area_km2": 5.0, "beyond_farmland_mm_yr": 20.0},
                        {"centre_mm_yr": 0.0, "band_lo_mm_yr": -5.0, "band_hi_mm_yr": 5.0}, nul)
    assert ch["p"] == pytest.approx(0.005) and ch["p_text"] == "less than 1 in 200"


# ------------------------------------------------------------------ F7(a): same as chance
def test_same_as_chance_only_inside_the_binomial_range_for_the_radar_cells():
    """Site B: 5.6% by chance over 242,891 radar cells is 5.51-5.69%; the
    old fixed 1.5 points called 5.8% (and 6.9%) the same as chance."""
    assert ri.chance_share_range(5.6, 242891) == [5.51, 5.69]
    s = ri.cells_sentence(5.8, 5.6, 242891)
    assert "against 5.6% by chance" in s and "same as chance" not in s
    assert "same as chance" in ri.cells_sentence(5.6, 5.6, 242891)
    assert "against" in ri.cells_sentence(6.7, 5.6, 242891)
    # few cells: chance alone spreads widely (4% of 100 cells: 1-8%)
    assert ri.chance_share_range(4.0, 100) == [1.0, 8.0]
    assert "same as chance" in ri.cells_sentence(5.4, 4.0, 100)
    # without a cell count the range cannot be had: never called the same
    assert "same as chance" not in ri.cells_sentence(5.6, 5.5)
    assert ri.cells_sentence(None, 5.0, 1000) == ""


# ------------------------------------------------------------------ F7(c): fastest only with placements
def _dates():
    t = np.arange(2016.0, 2026.0, 12 / 365.25)
    return t, [int(x) for x in t]


def _series_from_yearly(t, years, rate_by_year):
    s = np.zeros(t.size)
    for k in range(1, t.size):
        s[k] = s[k - 1] + rate_by_year[years[k]] * (t[k] - t[k - 1])
    return s


def test_fastest_is_never_named_from_the_approximate_yardstick():
    """A clear fast spell (-75 mm/yr in 2019 against about -15 otherwise):
    with 40 placements it is named; with 9 (under 10) or none the old
    fallback named it on no farmland evidence - now the years are withheld
    and the reason recorded, while the other words still follow the data."""
    t, years = _dates()
    used = sorted(set(years))
    rates = dict(zip(used, [-15.0, -16.0, -20.0, -75.0, -70.0, -18.0, -15.0, -14.0, -15.0, -16.0]))
    s = _series_from_yearly(t, years, rates)
    rng = np.random.default_rng(2)
    rows40 = rng.normal(0, 3.0, (40, len(used)))
    d = ri.trend_shape(t, s, years, noise_mm_yr=1.0, null_yearly={"years": used, "rates": rows40})
    assert d["fastest"] == [2019, 2020] and "fastest in 2019–2020" in d["text"]
    for null in (None, {"years": used, "rates": rows40[:9]}):
        d = ri.trend_shape(t, s, years, noise_mm_yr=1.0, null_yearly=null)
        assert "fastest" not in d and "fastest in" not in d["text"]
        c = d["contrasts"]["fastest"]
        assert c["null_z_q"] is None and "fewer than 10 farmland patches" in c["withheld"]
        assert d["text"] and d["shape"] in ("steady", "slowing", "accelerating")
