"""Calibration decisions of 7 Oct 2026 (report_insights.py, client_report.py):

G1  the exact Monte Carlo rule: a selection-corrected p = (1 + k) / (1 + n)
    passes when p <= SELECTION_ALPHA / 2, compared in whole numbers
    ((1 + k) <= floor(alpha_side x (1 + n))) - an exact 2.5% a side with 199
    copies (k <= 4). Until then p < 0.025 (k <= 3, 2.0% a side): Site B's
    2.2 km² patch, 4 of 199 copies as strong (p 5/200), was not listed.
G2  the leave-out re-tests are corrected for their number: with L
    candidates re-tested, one passes when min(1, L x p_left_out) <=
    SELECTION_ALPHA / 2; the record keeps p, L and p_adjusted, and a
    feature listed because of its re-test prints the adjusted chance
    ("about 3 in 200"). Uncorrected, pure noise listed a feature in 5.6% of
    fields at 540 m correlation (1.9% from the first pass alone).

Each test checks an outcome with a known answer and fails on the code
before this round (backups/pre_cal_20261007/).

Run: python -m pytest insar_pipeline/tests/test_fix_oct7_cal.py -q -p no:cacheprovider
"""
import types

import numpy as np
import pytest

from insar_pipeline import client_kmz
from insar_pipeline import client_report as cr
from insar_pipeline import report_insights as ri


# ------------------------------------------------------------------ helpers
SM = {"centre_mm_yr": 0.0, "band_lo_mm_yr": -2.0, "band_hi_mm_yr": 2.0, "smoothing_m": 500}


def _null(max_mass, tested=True):
    return {"tested": tested, "null_area_km2": 300.0, "searched_area_km2": 300.0,
            "sides": {"sinking": {"tested": True, "max_mass": list(max_mass), "median_max_beyond_mm_yr": 3.0},
                      "rising": {"tested": True, "max_mass": list(max_mass), "median_max_beyond_mm_yr": 3.0}},
            "feature_null_check": {"real_beats_all_copies": False}}


def _copies_with(k, n=199, mass=10.0):
    """n copies' strongest patches of which exactly k are at least `mass`."""
    return [mass + 1.0 + i for i in range(k)] + [1.0 + 0.01 * i for i in range(n - k)]


def _feature(mass=10.0, sign=-1, mask=None, **kw):
    # cluster mass = (beyond - band width 2) x area 1 km²
    f = {"sign": sign, "beyond_farmland_mm_yr": mass + 2.0, "area_km2": 1.0, "lat": 38.3, "lon": -87.4}
    if mask is not None:
        f["_mask"] = mask
    f.update(kw)
    return f


# ------------------------------------------------------------------ G1
def test_four_of_199_copies_as_strong_is_listed_five_is_not():
    """Site B's 2.2 km² patch (k = 4 of 199, p = 5/200 = 0.025): on the
    line, so listed under the exact rule (it was not under p < 0.025)."""
    ch = ri.selection_p(_feature(), SM, _null(_copies_with(4)))
    assert ch["copies_as_strong"] == 4 and ch["p"] == 0.025
    assert ch["stands_out"] is True and ch["p_text"] == "about 5 in 200"
    ch5 = ri.selection_p(_feature(), SM, _null(_copies_with(5)))
    assert ch5["copies_as_strong"] == 5 and ch5["stands_out"] is False


def test_the_rule_is_exact_in_whole_numbers_for_every_copy_count():
    """p <= 1/40 <=> 40 (1 + k) <= 1 + n, for every n and k - including the
    copy counts where (1 + n) / 40 is whole and p lands exactly on 0.025
    (39, 79, 199 copies), where p < 0.025 refused it."""
    for n in range(19, 420):
        for k in range(0, 15):
            assert ri.passes_selection(k, n) is (40 * (1 + k) <= 1 + n), (k, n)
    assert ri.passes_selection(0, 39) and ri.passes_selection(1, 79) and ri.passes_selection(4, 199)
    assert ri.selection_p(_feature(), SM, _null(_copies_with(0, n=39)))["stands_out"] is True


def test_the_level_is_exactly_two_and_a_half_percent_a_side_with_199_copies():
    """Under the null the patch's rank among the 200 is uniform, so the
    rule's size is (passing k values) / 200: exactly 5/200 (it was 4/200 =
    2.0%); corrected for L re-tests it is at most 2.5% / L."""
    assert sum(ri.passes_selection(k, 199) for k in range(200)) == 5
    for L in (1, 2, 3):
        size = sum(ri.passes_selection(k, 199, tests=L) for k in range(200)) / 200
        assert size <= 0.025 / L + 1e-12
    assert [k for k in range(10) if ri.passes_selection(k, 199, tests=2)] == [0, 1]
    assert [k for k in range(10) if ri.passes_selection(k, 199, tests=3)] == [0]


# ------------------------------------------------------------------ G2: the record
@pytest.mark.parametrize("k, L, passed, p_adj, text", [
    (0, 3, True, 0.015, "about 3 in 200"),       # Site A Feature 1: beat all 199 copies, 3 re-tested
    (1, 3, False, 0.03, "about 6 in 200"),       # passed uncorrected (p 0.01)
    (1, 2, True, 0.02, "about 4 in 200"),
    (2, 2, False, 0.03, "about 6 in 200"),
    (4, 1, True, 0.025, "about 5 in 200"),       # one re-test: the first pass's rule
    (0, 1, True, 0.005, "less than 1 in 200"),
    (150, 3, False, 1.0, "about 200 in 200"),    # capped at 1
])
def test_leave_out_result_is_corrected_for_the_number_re_tested(k, L, passed, p_adj, text):
    g = _feature()
    null = _null(_copies_with(k))
    g["chance"] = ri.selection_p(g, SM, null)
    got = ri.leave_out_result(g, SM, null, retested=L)
    assert got["copies_as_strong"] == k and got["p"] == round((1 + k) / 200, 4)
    assert got["retested"] == L and got["p_adjusted"] == pytest.approx(p_adj)
    assert got["passed"] is passed and got["p_adjusted_text"] == text
    assert got["p_text"] == g["chance"]["p_text"]                      # the re-test's own, kept


def _fake_flow(monkeypatch, first, rerun):
    """_leave_candidate_out on candidates `first` (first-pass chance given)
    with every re-run search returning `rerun` (the same patches, the
    re-test's chance given) - no real farmland."""
    monkeypatch.setattr(cr, "_exclude_from_farmland", lambda ctx, masks, b: {})
    monkeypatch.setattr(cr, "_touches_farmland", lambda ctx, m, b: True)
    null2 = _null(_copies_with(0))
    monkeypatch.setattr(cr, "_feature_search", lambda ctx, *a, **k: (dict(SM), [dict(g) for g in rerun], null2, True))
    ctx = types.SimpleNamespace()
    return cr._leave_candidate_out(ctx, dict(SM), first, _null(_copies_with(100)), 1.0, {}, None, [])


def _masks(n):
    out = []
    for i in range(n):
        m = np.zeros((20, 20), bool)
        m[i * 5:i * 5 + 4, 2:6] = True
        out.append(m)
    return out


def _cand(mask, mass, k, stands=False):
    f = _feature(mass, mask=mask)
    f["chance"] = {"tested": True, "cluster_mass": mass, "copies_as_strong": k, "copies": 199,
                   "p": round((1 + k) / 200, 4), "p_text": ri.p_text(k, 199), "stands_out": stands}
    f["stands_out"] = stands
    return f


def test_three_re_tests_need_a_patch_to_beat_every_copy(monkeypatch):
    """Three leading candidates re-tested; the strongest left out has 1 of
    199 copies as strong (p 0.01): it passed uncorrected; corrected for 3
    (p 0.03) it does not. The record keeps p, L and p_adjusted."""
    ms = _masks(3)
    first = [_cand(ms[0], 30.0, 60), _cand(ms[1], 20.0, 120), _cand(ms[2], 10.0, 190)]
    rerun = [_cand(ms[0], 40.0, 1), _cand(ms[1], 20.0, 120), _cand(ms[2], 10.0, 190)]
    rec = _fake_flow(monkeypatch, first, rerun)
    assert rec["retested"] == 3 and len(rec["candidates"]) == 3
    r = rec["candidates"][0]
    lo = r["left_out"]
    assert lo["p"] == 0.01 and lo["retested"] == 3 and lo["p_adjusted"] == pytest.approx(0.03)
    assert lo["passed"] is False and r["passed"] is False and rec["passed"] == 0
    assert "min(1, L x p)" in rec["rule"]


def test_one_re_test_keeps_the_first_pass_rule_and_a_first_pass_listing_is_not_counted(monkeypatch):
    """L counts only the candidates re-tested: one that beats chance on the
    first pass is not (it is cut anyway), so the other's re-test is judged
    alone - 4 of 199 copies as strong passes (p 0.025), 5 does not."""
    ms = _masks(2)
    first = [_cand(ms[0], 30.0, 0, stands=True), _cand(ms[1], 20.0, 60)]
    rec = _fake_flow(monkeypatch, first, [_cand(ms[0], 30.0, 0, True), _cand(ms[1], 25.0, 4)])
    assert rec["retested"] == 1
    assert rec["candidates"][0]["left_out"] is None and rec["candidates"][0]["passed"] is True
    lo = rec["candidates"][1]["left_out"]
    assert lo["retested"] == 1 and lo["p_adjusted"] == 0.025 and lo["passed"] is True
    rec5 = _fake_flow(monkeypatch, [_cand(ms[0], 30.0, 0, stands=True), _cand(ms[1], 20.0, 60)],
                      [_cand(ms[0], 30.0, 0, True), _cand(ms[1], 25.0, 5)])
    assert rec5["candidates"][1]["left_out"]["passed"] is False


# ------------------------------------------------------------------ G2: what is printed
def test_a_feature_listed_on_its_re_test_prints_the_adjusted_chance():
    """The second pass lists it at 'less than 1 in 200'; what is printed is
    the re-test's chance corrected for the 3 re-tested - in the page's
    paragraph and the KMZ - with the second pass's figures kept."""
    second = {"tested": True, "p": 0.005, "copies_as_strong": 0, "copies": 199, "p_text": "less than 1 in 200",
              "stands_out": True, "stands_out_clearly": False, "cluster_mass": 115.4}
    lo = {"p": 0.005, "copies_as_strong": 0, "copies": 199, "retested": 3, "p_adjusted": 0.015,
          "p_adjusted_text": "about 3 in 200", "strength": 115.4, "passed": True}
    ch = ri.leave_out_chance(second, lo)
    assert ch["p"] == 0.015 and ch["p_text"] == "about 3 in 200" and ch["stands_out"] is True
    assert ch["listed_on"] == "leave-out re-test" and ch["retested"] == 3
    assert ch["second_pass"]["p_text"] == "less than 1 in 200" and ch["leave_out"]["p_adjusted"] == 0.015
    assert second["p_text"] == "less than 1 in 200"                    # the input is not changed
    fe = {"sign": -1, "mean_rel_vertical_mm_yr": -16.3, "vertical_relative_mm_yr": -16.3, "area_km2": 23.7,
          "lat": 38.241, "lon": -87.647, "mines": {}, "chance": ch, "stands_out": True,
          "control_ground_label": "farmland"}
    txt = ri.feature_text(1, fe)
    assert "about 3 in 200" in txt and "less than 1 in 200" not in txt
    assert "3 of the strongest patches" in txt and "left out" in txt and "worth a look" in txt
    assert client_kmz.chance_text(fe) == "about 3 in 200"
    # a first-pass (or second-pass) feature is unaffected
    plain = dict(fe, chance=second)
    assert "less than 1 in 200" in ri.feature_text(1, plain) and "strongest patches" not in ri.feature_text(1, plain)


@pytest.fixture(scope="module")
def hidden(tmp_path_factory):
    """test_fix_oct7_fu_features' synthetic project (L9b) whose settling
    patch hides in its own farmland on the first pass, at 199 copies."""
    import test_fix_L9b as L9b
    mp = pytest.MonkeyPatch()
    mp.setattr(ri, "N_SURROGATES", 199)
    orig = cr._feature_search

    def self_hidden(ctx, *a, **k):
        sm, cands, null, ok = orig(ctx, *a, **k)
        for f in cands:
            if (f["_mask"] & ctx.far).sum() > 0.5 * f["_mask"].sum() and f["stands_out"]:
                f["stands_out"] = False
                f["chance"] = dict(f["chance"], stands_out=False, p=0.25, copies_as_strong=49,
                                   p_text="about 50 in 200")
        return sm, cands, null, ok
    mp.setattr(cr, "_feature_search", self_hidden)
    try:
        root = tmp_path_factory.mktemp("cal")
        cfg = L9b.make_project(root)
        rep, ctx, masks, outlines = cr.compute(cfg)
        import matplotlib
        matplotlib.use("Agg")
        figs = cr.figures(cfg, rep, ctx, masks, outlines, out_dir=cfg.project_dir / "deliverables" / "figures")
        html = cr.page(cfg, rep, figs)
    finally:
        mp.undo()
    return types.SimpleNamespace(rep=rep, html=html)


def test_compute_prints_the_re_test_chance_for_the_feature_it_cut(hidden):
    """Through compute and the page: the feature cut on its leave-out test
    and listed by the second pass carries the re-test's chance corrected
    for the number re-tested (not the second pass's), the record keeps
    p, L and p_adjusted, and the method paragraph says so."""
    rep = hidden.rep
    lo_rec = rep["feature_leave_out"]
    passed = [r for r in lo_rec["candidates"] if r["passed"] and r.get("left_out")]
    assert len(passed) == 1 and lo_rec["retested"] >= 1
    lo = passed[0]["left_out"]
    L = lo_rec["retested"]
    assert lo["retested"] == L and lo["p_adjusted"] == pytest.approx(min(1.0, L * lo["p"]), abs=1e-4)
    f = rep["features"][0]
    ch = f["chance"]
    assert ch["listed_on"] == "leave-out re-test" and ch["retested"] == L
    assert ch["p"] == lo["p_adjusted"] and f["selection_p"] == lo["p_adjusted"]
    assert ch["p_text"] == lo["p_adjusted_text"] and "second_pass" in ch
    assert passed[0]["second_pass"]["printed_p_text"] == lo["p_adjusted_text"]
    assert lo["p_adjusted_text"] in f["text"]
    assert "allows for how many patches were tested" in hidden.html
