"""Follow-up decisions of 7 Oct 2026, finishing pass: the gaps between the
groups' changes that fall under F1, F7(a) and F8.

F1     the injection test's farmland is the report's second pass, so it cuts
       what the report cuts: a candidate that beats chance only on its
       leave-candidate-out re-test as well (injection_test.report_farmland
       calls client_report._leave_candidate_out, as compute does)
F7(a)  the whole area's record carries the chance range of the radar-cell
       count the page words its sentence with, also when the E25 count could
       not be made (client_report._record_chance_range)
F7(d)  report areas that cover (nearly) all the farmland are not cut, by the
       records' own CUT_MIN_CELLS rule, so the report and the technical
       records read one farmland (client_report._exclude_report_areas; it
       stopped on the empty farmland before)
F8     the Results screen names the project's comparison ground beside an
       untested range (control_ground_label), as the run view and the page do

Each test fails on the code the groups left (backups/pre_followup_20261007/
plus the groups' changes, before this pass).

Run: python -m pytest insar_pipeline/tests/test_fix_oct7_fu_finish.py -q -p no:cacheprovider
"""
from types import SimpleNamespace

import numpy as np

import test_fix_oct7_trust_inj as INJ           # noqa: E402  (its synthetic official project)
from insar_pipeline import client_report as cr
from insar_pipeline import injection_test as it
from insar_pipeline import report_insights as ri
from insar_pipeline import significance as sg


# ------------------------------------------------------------------ F1: the injection test's farmland
def test_injection_farmland_cuts_a_candidate_passing_only_when_left_out(tmp_path, monkeypatch):
    """Site A: Feature 1 is within chance on the first pass (24 of 199
    copies as strong) and beats every copy left out, so the report cuts it.
    The injection test rebuilt the second pass from first-pass stand-outs
    only, so its sites and nulls kept Feature 1 + 500 m - the mismatch its
    7 Oct rebuild was made to remove."""
    O = INJ._project(tmp_path)
    hidden = np.zeros((O.ny, O.nx), bool)
    hidden[40:70, 150:190] = True                       # within chance with itself in the farmland

    def search(ctx, los_up_map, poly, aoi, standout):
        f = {"_mask": hidden & ctx.good, "stands_out": False, "sign": -1}
        return {"field": None}, [f], {"tested": True}, True
    seen = {}

    def leave_out(ctx, sm, cands, null, los_up_map, poly, aoi, standout):
        seen["standout"], seen["n"] = standout, len(cands)
        cands[0]["leave_out"] = {"passed": True, "left_out": {"passed": True}}
        return {"tested": True, "candidates": [cands[0]["leave_out"]], "passed": 1}
    monkeypatch.setattr(cr, "_feature_search", search)
    monkeypatch.setattr(cr, "_leave_candidate_out", leave_out)
    F, info = it.report_farmland(O.cfg)
    assert seen == {"standout": None, "n": 1}            # the report's call, with no QA stand-out (E7)
    grown = sg.euclidean_buffer(hidden, F["attrs"], cr.FARMLAND_SECOND_PASS_M)
    assert not (F["far"] & grown).any() and not (F["far_wide"] & grown).any()
    assert info["features_cut"] == 1 and info["features_cut_on_leave_out"] == 1 and info["leave_out_tested"]
    assert info["far_cells"] < info["far_cells_without_report_areas"]


def test_injection_farmland_keeps_a_candidate_failing_both_tests(tmp_path, monkeypatch):
    """The other side: a candidate within chance on both tests stays in the
    farmland, as in the report (only what beats chance is cut)."""
    O = INJ._project(tmp_path)
    patch = np.zeros((O.ny, O.nx), bool)
    patch[40:70, 150:190] = True

    def search(ctx, los_up_map, poly, aoi, standout):
        return {"field": None}, [{"_mask": patch & ctx.good, "stands_out": False, "sign": -1}], {"tested": True}, True

    def leave_out(ctx, sm, cands, null, los_up_map, poly, aoi, standout):
        cands[0]["leave_out"] = {"passed": False}
        return {"tested": True, "candidates": [cands[0]["leave_out"]], "passed": 0}
    monkeypatch.setattr(cr, "_feature_search", search)
    monkeypatch.setattr(cr, "_leave_candidate_out", leave_out)
    F, info = it.report_farmland(O.cfg)
    assert info["features_cut"] == 0 and info["features_cut_on_leave_out"] == 0
    assert info["far_cells"] == info["far_cells_without_report_areas"]


# ------------------------------------------------------------------ F7(a): the record's chance range
def test_record_carries_the_range_of_the_count_the_page_uses():
    """When deliverables.radar_pixel_ids fails, the page words the sentence
    on summary.json's count (or the estimate); the record carried no range,
    so a reader could not see the range the page's 'same as chance' rested on."""
    rep = {"independent_radar_measurements": 242891,
           "independent_radar_measurements_basis": "estimated: study-area cells x the 15 x 15 m map cell / ...",
           "areas": [{"kind": "full", "cells_standing_out_percent": 6.4, "cells_standing_out_by_chance_percent": 5.6,
                      "cells_standing_out_chance_range95_percent": None},
                     {"kind": "area", "cells_standing_out_by_chance_percent": 5.6}]}
    cr._record_chance_range(rep)
    full, area = rep["areas"]
    assert full["cells_standing_out_chance_range95_percent"] == [5.51, 5.69]
    assert "242,891" in full["cells_standing_out_chance_range_basis"] and "estimated" in full[
        "cells_standing_out_chance_range_basis"]
    assert "cells_standing_out_chance_range95_percent" not in area      # only the whole area has the sentence
    # the page's sentence rests on the same range
    s = ri.cells_sentence(6.4, 5.6, rep["independent_radar_measurements"])
    assert "against 5.6% by chance" in s
    # a range the E25 count already set is kept; no count, no range
    rep2 = {"independent_radar_measurements": 100,
            "areas": [{"kind": "full", "cells_standing_out_by_chance_percent": 4.0,
                       "cells_standing_out_chance_range95_percent": [3.9, 4.1]}]}
    cr._record_chance_range(rep2)
    assert rep2["areas"][0]["cells_standing_out_chance_range95_percent"] == [3.9, 4.1]
    rep3 = {"areas": [{"kind": "full", "cells_standing_out_by_chance_percent": 4.0}]}
    cr._record_chance_range(rep3)
    assert rep3["areas"][0].get("cells_standing_out_chance_range95_percent") is None


# ------------------------------------------------------------------ F8: the Results screen's wording
class _Widget:
    def __init__(self):
        self.shown = None
        self.display = True

    def update(self, x=None, **kw):
        self.shown = x

    def __getattr__(self, name):
        return lambda *a, **k: None


def test_results_screen_names_the_projects_comparison_ground():
    """The untested range added under F8 said 'not compared with farmland'
    whatever the project calls its comparison ground; the page and the run
    view use control_ground_label."""
    from insar_pipeline import results_screen as rsc
    widgets: dict = {}
    s = {"points": 900, "images": 270, "first_image": "2016-01-05", "last_image": "2026-06-17", "years": 10.45,
         "mean_vertical_velocity_in_yr": 0.0, "mean_vertical_cumulative_in": 0.0,
         "vertical_95pct_range_in_yr": None, "untested_vertical_95pct_range_in_yr": [-0.459, 0.459],
         "measured_change_95pct_range_in": [-4.80, 4.80], "testable": False, "signal_to_noise": None,
         "reading": "Not testable: too little ordinary ground", "kind": "area", "gnss_anchored": True}
    res = SimpleNamespace(unit="in", area="P", area_summary=s, stats={}, annual=[], notes=[])
    fake = SimpleNamespace(query_one=lambda sel, *a, **k: widgets.setdefault(sel, _Widget()),
                           cfg=SimpleNamespace(control_ground_label="open rangeland"))
    rsc.ResultsScreen._show(fake, res)
    txt = widgets["#rsummary"].shown.plain
    assert "95% range -0.459 to +0.459 in/yr, not compared with open rangeland" in txt and "farmland" not in txt
    # the default label stays "farmland"
    fake2 = SimpleNamespace(query_one=fake.query_one, cfg=SimpleNamespace(control_ground_label=""))
    rsc.ResultsScreen._show(fake2, res)
    assert "not compared with farmland" in widgets["#rsummary"].shown.plain


# ------------------------------------------------------------------ F7(d): one farmland when areas cover it
def test_report_areas_covering_the_farmland_are_not_cut_as_in_the_records(tmp_path, monkeypatch):
    """quality_assessment.report_farmland_masks (F7(d)) leaves the report
    areas in when cutting them would leave under CUT_MIN_CELLS farmland
    cells; the report cut them anyway, so the two read different farmland
    and compute stopped on the empty farmland (KeyError '__far__'). Now the
    report applies the same rule: it is built, the area reads Not testable,
    and its farmland is the records' cell for cell."""
    import dataclasses

    import test_fix_L9b as L9b
    from insar_pipeline import quality_assessment as qa
    monkeypatch.setattr(ri, "N_SURROGATES", 19)
    cfg = L9b.make_project(tmp_path)
    whole = L9b._box_spec((slice(0, L9b.NY), slice(0, L9b.NX)))
    cfg = dataclasses.replace(cfg, report_areas=("Full Satellite Imaging:full", f"Huge:{whole}"))
    rep, ctx, masks, _ = cr.compute(cfg)
    out = rep["report_areas_out_of_farmland"]
    assert out["applied"] is False and out["report_areas_not_cut"] == ["Huge"]
    # the record's words for its comparison ground: plain "ground" on this
    # synthetic project, which has no land-cover map (28 Oct 2026)
    assert rep["control_ground_label"] == "ground"
    assert "cover nearly all the ground" in out["reason"] and out["far_cells_removed"] == 0
    huge = next(a for a in rep["areas"] if a["name"] == "Huge")
    assert huge["reading"].startswith("Not testable")
    far, far_wide, cut, meta = qa.report_farmland_masks(cfg, ctx.good, ctx.attrs)
    assert cut is None and meta["comparison_ground"]["report_areas_not_cut"] == ["Huge"]
    first = rep.get("farmland_second_pass") or {}
    assert int(far.sum()) == (first.get("far_cells") or [int(ctx.far.sum())])[0]
