"""
Trust review of 7 Oct 2026, group rel (release.py, provenance.py): the gate's
side of contracts C1 and C4, and the record of the tests' own checks.

E6 / C1  a report area whose local null (farmland copies within 10 km) or
         coherence-matched null disagrees with the pooled null it was tested
         against is a soft reviewer item, 'null_local_differs', printing both
         nulls (findings #10, #11). Nothing switches.
E8 / C4  when the real comparison ground's strongest patch beats every
         simulated feature copy, a soft reviewer item,
         'feature_null_misspecified' (finding #19: Site A as delivered,
         87.1 against a copy maximum of 61.2, recorded and never acted on).
         A report built before the check is judged from the copy maxima its
         feature_search kept, and told to rebuild.
provenance  the stages record carries each area's nulls, limits and
         pace-change test (E5, E6, E10), each feature's chance and "stands
         out clearly" (E7, E8), the feature null check, the GNSS screening
         (E15, E16) and the injection test's decision rule with no client
         sentence (E4).
Each test fails on the code before this round (backups/pre_trust_20261007).
Synthetic projects in tmp_path; the two read-only trust-check copies are
read (never written) where they exist.
"""
import json

import numpy as np
import pytest

import test_fix_L7b as L7b                                     # the gate's synthetic project
from insar_pipeline import provenance, release
from insar_pipeline import significance as sg


# The Area A2's coherence-matched null as significance.null_checks wrote
# it on the Site A copy (work_sig/e5e6_site_a_pl.json): it agrees.
AGREES = {"median_mm_yr": -0.39, "spread_mm_yr": 5.24, "n_independent": 3768, "n_placements": 1831039,
          "disagrees": False, "coherence_window": [0.505, 0.555],
          "why": "Farmland copies of this area's coherence (0.505-0.555) (about 3768 independent copies): centre "
                 "-0.39 mm/yr against +0.72 for all copies, a difference of 1.10 (2.65, a quarter of the 95% "
                 "threshold 10.60, is allowed); spread 5.24 against 5.55, a ratio of 0.94 (0.80-1.25 allowed). "
                 "It agrees with the pooled comparison ground."}
# The same area's local null moved by a regional offset: it disagrees.
DIFFERS = {"median_mm_yr": -3.61, "spread_mm_yr": 5.31, "n_independent": 630, "n_placements": 306017,
           "disagrees": True, "radius_km": 10.0,
           "why": "Farmland copies centred within 10 km of this area's centre (about 630 independent copies): "
                  "centre -3.61 mm/yr against +0.72 for all copies, a difference of 4.33 (2.65, a quarter of the "
                  "95% threshold 10.60, is allowed); spread 5.31 against 5.55, a ratio of 0.96 (0.80-1.25 "
                  "allowed). It differs from the pooled comparison ground: its centre is shifted, so the 5% "
                  "false-alarm rate and the 4-in-5 detection stated for this area are averages over the scene "
                  "that may not hold here. The pooled figures are still the ones reported."}


@pytest.fixture(autouse=True)
def _identity_set(monkeypatch):
    from insar_pipeline import settings
    monkeypatch.setattr(settings, "identity", lambda: {"firm": "Test Firm", "preparer": "A. Engineer", "contact": ""})


def _codes(b, hard=None):
    return [x["code"] for x in b if hard is None or x["hard"] is hard]


def _item(b, code):
    hits = [x for x in b if x["code"] == code]
    assert len(hits) == 1, b
    return hits[0]


def _with_nulls(local, coherence, name="Area A2"):
    rep = L7b._rep(area_name=name)
    rep["areas"][1].update(null_local=local, null_coherence_matched=coherence)
    return rep


def _signed(tmp_path, rep):
    cfg = L7b._cfg(tmp_path)
    L7b._project(tmp_path, rep, L7b._qa())
    L7b._sign_all(tmp_path, rep)
    return cfg


# ------------------------------------------------------------------ E6 / C1: null_local_differs
def test_an_area_whose_local_null_disagrees_is_a_soft_item_printing_both_nulls(tmp_path):
    rep = _with_nulls(DIFFERS, AGREES)
    cfg = _signed(tmp_path, rep)
    b = release.blockers(cfg)
    it = _item(b, "null_local_differs")
    assert it["hard"] is False
    t = it["text"]
    assert t.startswith("'Area A2': ")
    # both nulls, each with its own and the pooled centre and spread
    assert "centre -3.61 mm/yr against +0.72" in t and "centre -0.39 mm/yr against +0.72" in t
    assert "spread 5.31 against 5.55" in t and "spread 5.24 against 5.55" in t
    # the frame of those figures, and what to look at
    assert "line of sight" in t and "Look at the farmland round the area on the map" in t
    assert ".." not in t
    # soft: a signed report is still FINAL, and nothing hard was added
    assert release.is_final(cfg) and release.status(cfg)["final"] is True
    assert "null_local_differs" not in _codes(release.hard_blockers(cfg))
    # the units client_report gives the copied nulls are the ones printed
    rep_u = _with_nulls(dict(DIFFERS, units="mm/yr along the line of sight"),
                        dict(AGREES, units="mm/yr along the line of sight"))
    t_u = _item(release.blockers(L7b._cfg(tmp_path), rep_u), "null_local_differs")["text"]
    assert "(Figures in mm/yr along the line of sight, as the test computes them; the page's are vertical.)" in t_u


@pytest.mark.parametrize("local,coherence,fires", [
    (AGREES, DIFFERS, True),                     # the coherence-matched null alone
    (None, DIFFERS, True),                       # the local one too thin (None)
    (AGREES, AGREES, False),
    (None, None, False),                         # neither has 10 independent copies
    ("absent", "absent", False),                 # a report from before the check
])
def test_either_null_disagreeing_fires_and_nothing_else_does(tmp_path, local, coherence, fires):
    rep = L7b._rep(area_name="Area A2")
    if local != "absent":
        rep["areas"][1].update(null_local=local, null_coherence_matched=coherence)
    cfg = _signed(tmp_path, rep)
    assert ("null_local_differs" in _codes(release.blockers(cfg))) is fires
    assert release.null_checks_differ(rep) == ([("Area A2", [w["why"] for w in (local, coherence)
                                                                 if isinstance(w, dict)])] if fires else [])
    assert release.is_final(cfg)


def test_the_gate_reads_what_significance_null_checks_writes():
    """End to end on the contract: farmland copies near the area sit 3 mm/yr
    above the rest, so significance.null_checks' local null disagrees, and
    the gate lists exactly that area."""
    rng = np.random.default_rng(0)
    rows, cols = np.meshgrid(np.arange(0, 600, 4), np.arange(0, 600, 4), indexing="ij")
    rows, cols = rows.ravel().astype(float), cols.ravel().astype(float)
    reg = np.zeros((600, 600), bool)
    reg[295:305, 295:305] = True
    near = np.hypot(rows - 299.5, cols - 299.5) * 100.0 <= sg.LOCAL_NULL_KM * 1000.0
    vals = rng.normal(0.0, 1.0, rows.size) + np.where(near, 3.0, 0.0)
    med, sd = float(np.median(vals)), float(vals.std())
    c95 = float(np.quantile(np.abs(vals - med), 0.95))
    nc = sg.null_checks(vals, {"rows": rows, "cols": cols, "coherence": None}, 1.0, reg, med, sd, c95,
                        attrs={"X_FIRST": 500000.0, "Y_FIRST": 4_000_000.0, "X_STEP": 100.0, "Y_STEP": -100.0})
    assert nc["local_null"]["disagrees"] is True
    rep = {"areas": [{"name": "Full Satellite Imaging", "kind": "full"},
                     {"name": "Quiet Field", "kind": "area", "null_local": None, "null_coherence_matched": None},
                     {"name": "Area A1", "kind": "area", "null_local": nc["local_null"],
                      "null_coherence_matched": nc["coherence_null"]}]}
    got = release.null_checks_differ(rep)
    assert [n for n, _ in got] == ["Area A1"] and got[0][1] == [nc["local_null"]["why"]]


def test_mine_ground_whose_null_disagrees_is_listed_too(tmp_path):
    """The ground on a mapped mine layer gets a client verdict from the same
    test (client_report.tested_fields, report_insights' Observations), so
    its nulls are read as an area's are."""
    rep = L7b._rep(area_name="Area A2",
                   mine_ground=[{"key": "surface", "label": "surface mines", "detected": False,
                                 "null_local": DIFFERS, "null_coherence_matched": None},
                                {"key": "underground", "label": "underground mines", "detected": False,
                                 "null_local": AGREES, "null_coherence_matched": None},
                                {"key": "old", "label": "old workings", "note": "too few measured points",
                                 "testable": False}])
    assert release.null_checks_differ(rep) == [("Ground on the mapped surface mines", [DIFFERS["why"]])]
    cfg = _signed(tmp_path, rep)
    it = _item(release.blockers(cfg), "null_local_differs")
    assert it["hard"] is False and it["text"].startswith("'Ground on the mapped surface mines': ")
    assert release.is_final(cfg)


# ------------------------------------------------------------------ E8 / C4: feature_null_misspecified
NEW_CHECK = {"real_beats_all_copies": True, "tested": True, "sides": ["sinking"], "covariance_matched": True,
             "why": "sinking: the real comparison ground's strongest patch (strength 87.1, 5.7 km² near 38.442 N, "
                    "87.843 W) is stronger than the strongest of all 59 simulated copies (61.2). Ordinary ground "
                    "here makes patches the copies never do - movement no layer maps, or copies too light - so the "
                    "chance figures may be too low and no feature is called 'stands out clearly'."}


def _old_search(real=87.08, top=61.214, tested=True):
    """feature_search as a report built before 7 Oct 2026 kept it (Site A's numbers)."""
    return {"tested": tested, "n_surrogates": 59, "seed": 0,
            "sides": {"sinking": {"tested": True, "farmland_strongest_mass": real,
                                  "max_mass": [top] + [25.9] * 58, "median_max_mass": 25.9},
                      "rising": {"tested": True, "farmland_strongest_mass": 11.62, "max_mass": [28.681] * 59}}}


def test_real_farmland_beating_every_copy_is_a_soft_item_with_what_to_look_at(tmp_path):
    rep = L7b._rep(feature_null_check=NEW_CHECK, feature_search=_old_search())
    cfg = _signed(tmp_path, rep)
    it = _item(release.blockers(cfg), "feature_null_misspecified")
    assert it["hard"] is False
    t = it["text"]
    assert t.startswith("The chance test for features may be too lenient for this ground - sinking: ")
    assert "(strength 87.1, 5.7 km² near 38.442 N, 87.843 W)" in t and "(61.2)" in t
    assert "Look at that patch on the map and imagery" in t and "mine layers" in t
    assert "built before" not in t and ".." not in t
    assert release.is_final(cfg)


def test_the_reports_own_check_decides_over_the_stored_copies(tmp_path):
    """A report that stored its check (copies matched to 4 km: Site A
    87.1 against 88.8) is not overruled by recomputing from the maxima."""
    no = dict(NEW_CHECK, real_beats_all_copies=False, sides=[],
              why="the real comparison ground's strongest patch is within what the simulated copies produce")
    rep = L7b._rep(feature_null_check=no, feature_search=_old_search(real=87.08, top=61.2))
    assert "feature_null_misspecified" not in _codes(release.blockers(_signed(tmp_path, rep)))
    # the null's own copy inside feature_search is read when the top-level one is absent
    rep2 = L7b._rep(feature_search=dict(_old_search(real=1.0, top=61.2), feature_null_check=NEW_CHECK))
    chk, src = release.feature_null_check(rep2)
    assert chk is NEW_CHECK and src.endswith("feature_search.feature_null_check")


def test_a_report_built_before_the_check_is_judged_from_its_stored_copies(tmp_path):
    rep = L7b._rep(feature_search=_old_search())
    chk, src = release.feature_null_check(rep)
    assert chk["real_beats_all_copies"] is True and chk["sides"] == ["sinking"] and src.startswith("recomputed")
    cfg = _signed(tmp_path, rep)
    t = _item(release.blockers(cfg), "feature_null_misspecified")["text"]
    assert "(strength 87.1) is stronger than the strongest of all 59 simulated copies (61.2)" in t
    assert "built before that check (7 Oct 2026)" in t and "stands out clearly" in t and "rebuild it" in t
    assert release.is_final(cfg)
    # within the copies, an untested search, or none at all: nothing
    for fs in (_old_search(real=55.0), _old_search(real=0.0), _old_search(tested=False), None):
        r = L7b._rep(feature_search=fs) if fs is not None else L7b._rep()
        assert "feature_null_misspecified" not in _codes(release.blockers(L7b._cfg(tmp_path), r)), fs


# ------------------------------------------------------------------ provenance: the tests' own checks
def _tie_new():
    return {"record_window": ["20160105", "20260617"], "source": "NGL",
            "background": {"vertical_mm_yr": -1.5, "vertical_uncertainty_mm_yr": 1.5,
                           "uncertainty_basis": "scatter between the sites used", "left_out": ["VINU"]},
            "uncertainty_floor_mm_yr": 0.95,
            "estimator": "MIDAS (7 Oct 2026): 1-year pairs closest to one year ...",
            "steps": {"file": "/x/gnss/ngl_steps.txt", "modified": None, "screened": False,
                      "note": "the NGL steps file could not be downloaded"},
            "outlier_test": {"rule": "sites within 10 km share the regional field ...", "z": 1.96,
                             "left_out": ["VINU"],
                             "pairs": [{"sites": ["INVI", "VINU"], "distance_km": 6.9, "common_years": 5.19,
                                        "tested": True, "difference_mm_yr": 4.1, "sigma_mm_yr": 1.15,
                                        "limit_mm_yr": 2.25, "disagree": True, "left_out": "VINU"}]},
            "stations": [{"site": "INEV", "used_for_background": True,
                          "screening": {"steps_not_straddled": [2021.654], "antenna_height_days_dropped": 192,
                                        "antenna_heights_dropped_m": [0.0]}},
                         {"site": "VINU", "used_for_background": False, "left_out": "disagrees with a neighbour",
                          "screening": {"steps_not_straddled": [2014.371], "antenna_height_days_dropped": 99,
                                        "antenna_heights_dropped_m": [-0.0533]}}],
            "radius_sensitivity": {"40": {"vertical_mm_yr": -1.5, "sites": ["INEV"], "evaluated": True},
                                   "50": {"vertical_mm_yr": None, "sites": ["INEV"], "evaluated": False,
                                          "status": "not evaluated", "missing_sites": ["INDA", "INJS"]},
                                   "not_cached": ["INDA", "INJS"]},
            "inside_checks": [{"site": "LAMC", "verdict": "not testable", "consistent": None,
                               "verdict_reason": "Not testable: over the 2.2 years ...", "overlap": [2016.01, 2018.21],
                               "full_record": {"excess_mm_yr": 0.3}}]}


def test_provenance_records_the_reports_checks_the_gnss_screening_and_the_injection_rule(tmp_path):
    rep = _with_nulls(DIFFERS, AGREES)
    # the pace-change test as client_report.measured_change stores it (E10)
    pace = {"tested": False, "pace_change_mm": 7.4, "farmland_p95_mm": 38.8, "farmland_copies": 30,
            "method": "rate_x_span", "reason": "within farmland's own pace changes", "frame": "vertical"}
    rep["areas"][1].update(detection_limit_mm_yr=18.3, detection_limit_rise_mm_yr=19.7, null_checks_note=None,
                           total_method="rate_x_span", nonlinear=False, pace_change=pace,
                           deepest_patch={"pace_change": "another place's", "null_local": "another place's"})
    rep["features"][0].update(stands_out_clearly=False, quality_standout=True, total_method="rate_x_span",
                              nonlinear=False, pace_change=dict(pace, farmland_copies=11),
                              # report_insights.selection_p's keys
                              chance={"tested": True, "p": 0.0167, "copies_as_strong": 0, "copies": 59,
                                      "p_text": "less than 1 in 60", "stands_out": True, "stands_out_clearly": False,
                                      "cluster_mass": 115.4, "clearly_withheld": "the copies are lighter"})
    rep["mine_ground"] = [{"key": "surface", "label": "surface mines", "detected": False,
                           "detection_limit_mm_yr": 9.1, "null_local": None, "null_coherence_matched": AGREES,
                           "null_checks_note": "fewer than 10 independent copies within 10 km"}]
    rep.update(feature_null_check=NEW_CHECK,
               feature_search=dict(_old_search(), covariance_check={"matched": True, "matched_to_m": 4000}))
    cfg = L7b._cfg(tmp_path)
    D = L7b._project(tmp_path, rep, L7b._qa())
    (D / "gnss_tie.json").write_text(json.dumps(_tie_new()))
    (D / "technical").mkdir()
    (D / "technical" / "injection_recovery.json").write_text(json.dumps({
        "status": "complete", "seed": 7, "sizes_km": [0.3, 2.0], "rates_mm_yr": [10.0, 40.0], "types": ["trough"],
        "sentence": None, "decision_rule": "'Found' is the report-area rule, sinking only: ...",
        "technical_summary": "Technical record, not for the client page. ...",
        "control": {"type": "control", "rate_mm_yr": 0.0, "what": "nothing planted"},
        "farmland": {"far_cells": 995819, "features_cut": 1}}))
    st = provenance.stages(cfg)

    d = st["client_report"]["diagnostics"]
    a = next(x for x in d["areas"] if x["name"] == "Area A2")
    assert a["null_local"] == DIFFERS and a["null_coherence_matched"] == AGREES
    assert (a["detection_limit_mm_yr"], a["detection_limit_rise_mm_yr"]) == (18.3, 19.7)
    assert a["pace_change"] == pace
    assert a["total_method"] == "rate_x_span" and a["nonlinear"] is False
    full = next(x for x in d["areas"] if x["kind"] == "full")
    assert "pace_change" not in full and "null_local" not in full    # nothing borrowed from nested blocks
    assert "deepest_patch" not in a
    f = d["features"][0]
    assert f["n"] == 1 and f["stands_out_clearly"] is False and f["quality_standout"] is True
    # the chance as selection_p keeps it: k and n are copies_as_strong and copies
    assert f["chance"] == {"tested": True, "p": 0.0167, "copies_as_strong": 0, "copies": 59,
                           "p_text": "less than 1 in 60", "stands_out": True, "stands_out_clearly": False,
                           "clearly_withheld": "the copies are lighter"}
    assert f["pace_change"]["tested"] is False and f["pace_change"]["farmland_copies"] == 11
    assert d["mine_ground"] == [{"label": "surface mines", "detected": False, "detection_limit_mm_yr": 9.1,
                                 "null_local": None, "null_coherence_matched": AGREES,
                                 "null_checks_note": "fewer than 10 independent copies within 10 km"}]
    assert d["feature_null_check"] == NEW_CHECK and d["feature_null_check_source"] == "client_report.json feature_null_check"
    assert d["feature_copies_covariance"] == {"matched": True, "matched_to_m": 4000}
    assert "note" not in d

    g = st["gnss_tie"]["screening"]
    assert g["estimator"].startswith("MIDAS") and g["steps_file"]["screened"] is False
    assert g["outlier_test"]["left_out"] == ["VINU"] and g["outlier_test"]["pairs"][0]["disagree"] is True
    assert g["background_left_out"] == ["VINU"]
    vinu = next(s for s in g["stations"] if s["site"] == "VINU")
    assert vinu["left_out"] and vinu["antenna_height_days_dropped"] == 99 and vinu["used_for_background"] is False
    assert g["radius_sensitivity"]["50"]["status"] == "not evaluated"
    assert g["radius_sensitivity"]["50"]["missing_sites"] == ["INDA", "INJS"] and "not_cached" not in g["radius_sensitivity"]
    assert g["inside_checks"] == [{"site": "LAMC", "verdict": "not testable", "consistent": None,
                                   "verdict_reason": "Not testable: over the 2.2 years ...", "overlap": [2016.01, 2018.21]}]

    inj = st["injection_test"]
    assert inj["decision_rule"].startswith("'Found' is the report-area rule")
    assert inj["technical_summary"].startswith("Technical record, not for the client page")
    assert inj["control"]["rate_mm_yr"] == 0.0 and inj["farmland"]["features_cut"] == 1
    assert "sentence" not in inj                                            # no client sentence (E4)


def test_provenance_says_when_a_report_or_tie_predates_the_checks(tmp_path):
    rep = L7b._rep(feature_search=_old_search())
    cfg = L7b._cfg(tmp_path)
    D = L7b._project(tmp_path, rep, L7b._qa())                              # its tie has no screening
    (D / "technical").mkdir()
    (D / "technical" / "injection_recovery.json").write_text(json.dumps(
        {"status": "complete", "seed": 1, "sentence": "Patches about 0.3 km across ... were found 20 of 20 times"}))
    st = provenance.stages(cfg)
    d = st["client_report"]["diagnostics"]
    assert "built before 7 Oct 2026" in d["note"]
    assert d["feature_null_check"]["real_beats_all_copies"] is True
    assert d["feature_null_check_source"].startswith("recomputed from feature_search")
    assert d["feature_copies_covariance"] == "not recorded"
    assert st["gnss_tie"]["screening"].startswith("not recorded: a tie written before the 7 Oct 2026 screening")
    assert st["injection_test"]["sentence"].startswith("Patches about 0.3 km")  # an older record keeps its own
