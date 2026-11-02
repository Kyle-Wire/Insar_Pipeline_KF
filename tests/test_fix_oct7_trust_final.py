"""Trust review of 7 Oct 2026, the finishing pass: cross-file gaps the
groups left open, closed within the decisions. Each test failed on the code
before this pass (the groups' versions of figures.py, quality_assessment.py,
report_insights.py and client_report.py).

  E21  the coherence caption reads the QA's mode (not the removed median)
       and calls the noise "open water" only when most of it is on water;
  E4   the QA's copy of the injection record is technical: it carries the
       decision rule and the technical summary, no client wording;
  E9   a feature's measured total carries its 95% range on the page, and
       the average-rate total is added by the page's own rule (#37);
  E14  write_rasters puts the point table on the report's farmland.
"""
import json
import types

import numpy as np
import pytest

from insar_pipeline import client_report as cr
from insar_pipeline import figures
from insar_pipeline import quality_assessment as qa
from insar_pipeline import report_insights as ri

MM_IN = 25.4


# ------------------------------------------------------------------ E21: the coherence caption
def test_noise_caption_reads_the_mode_and_names_water_only_when_it_is_water():
    """The QA's E21 record has 'mode' and no 'median': the caption dropped
    the typical level. Site B's noise population is 29% water, Site A's
    88%; the caption said "open water" for both."""
    fr = {"noise_level": {"mode": 0.371, "floor": 0.3941, "population_median": 0.39, "share_on_mapped_water": 0.29}}
    t = figures._noise_sentence(fr, 0.4)
    assert "about 0.371" in t and "up to about 0.394" in t and "+0.006" in t
    assert "decorrelated ground" in t and "open water" not in t
    ow = {"noise_level": {"mode": 0.373, "floor": 0.3982, "share_on_mapped_water": 0.88}}
    t = figures._noise_sentence(ow, 0.4)
    assert "Pure noise (open water) scores about 0.373" in t and "+0.002" in t
    # a record without the share says what the population is, not "open water"
    t = figures._noise_sentence({"noise_level": {"mode": 0.37, "floor": 0.39}}, 0.4)
    assert "open water" not in t and "decorrelated ground and water" in t
    # a QA from before 7 Oct (median) still gives its typical level
    t = figures._noise_sentence({"noise_level": {"median": 0.3856, "floor": 0.4359}}, 0.4)
    assert "about 0.386" in t and "-0.036" in t


# ------------------------------------------------------------------ E4: the QA's injection record
def test_qa_injection_record_is_technical_and_names_the_rule(tmp_path, monkeypatch):
    from insar_pipeline import crosscheck
    T = tmp_path / "deliverables" / "technical"
    T.mkdir(parents=True)
    (T / "injection_recovery.json").write_text(json.dumps({
        "status": "complete", "sentence": None, "seed": 3, "summary": [],
        "decision_rule": "'Found' is the report-area rule, sinking only: ...",
        "technical_summary": "Technical record, not for the client page: ..."}))
    monkeypatch.setattr(crosscheck, "record_is_current", lambda cfg, rec, opt: True)
    rec = qa._injection_record(types.SimpleNamespace(project_dir=tmp_path))
    assert rec["sentence"] is None and rec["gates_verdict"] is False
    assert rec["decision_rule"].startswith("'Found' is the report-area rule")
    assert rec["technical_summary"].startswith("Technical record")
    assert "not for the client page" in rec["note"] and "feature search is not tested" in rec["note"]
    assert "judged by the report's farmland rule" not in rec["note"]


# ------------------------------------------------------------------ E9 / #37: a feature's total on the page
def _feature(**kw):
    f = {"sign": -1, "mean_rel_vertical_mm_yr": -32.0, "vertical_relative_mm_yr": -32.0,
         "total_relative_mm": -329.0, "total_at_average_rate_mm": -334.0, "total_range95_mm": [-437.0, -258.0],
         "mines": {"key": "none", "phrase": "away from every mapped mine"}, "ground_reshaped": False,
         "chance": {"tested": True, "p": 1 / 60, "copies": 59, "copies_as_strong": 0, "stands_out": True,
                    "p_text": "less than 1 in 60"}, "stands_out_clearly": True, "cause_hint": ""}
    f.update(kw)
    return f


def test_feature_total_carries_its_range_and_the_pages_totals_rule():
    """Site B's 2.2 km2 feature: -329 mm measured, 95% range -437 to -258
    (client_report.total_range) - the page printed the total alone."""
    t = ri.feature_text(1, _feature())
    assert "moved 13.0 in (329 mm) (95% range 10.2 to 17.2 in)" in t
    # 5 mm apart on 329: within the page's rule (5 mm or 5%) - one total, as
    # the page's table and the KMZ print it
    assert "at its average rate" not in t
    # 7% apart: the page's rule prints both (the old 10% rule printed one)
    t = ri.feature_text(1, _feature(total_at_average_rate_mm=-352.0))
    assert cr.totals_differ(-329.0, -352.0) and "at its average rate it would have moved 13.9 in" in t
    # a range across zero is signed
    t = ri.feature_text(1, _feature(total_relative_mm=-40.0, total_at_average_rate_mm=-40.0,
                                    total_range95_mm=[-90.0, 12.0]))
    assert "(95% range -3.5 to +0.5 in, negative = down)" in t
    # a record from before 7 Oct has no range: nothing invented
    t = ri.feature_text(1, _feature(total_range95_mm=None))
    assert "95% range" not in t.split("Measured over the record")[1].split(".")[0]


# ------------------------------------------------------------------ E14: the point table follows the report
import test_fix_L9b as L9b                                     # noqa: E402  (the report's project builder)


@pytest.fixture(scope="module")
def computed(tmp_path_factory):
    root = tmp_path_factory.mktemp("trust_final")
    cfg = L9b.make_project(root)
    rep, ctx, masks, outlines = cr.compute(cfg)
    return types.SimpleNamespace(cfg=cfg, rep=rep, ctx=ctx)


def test_write_rasters_puts_the_point_table_on_the_reports_farmland(computed, monkeypatch, tmp_path):
    """deliverables.points_on_report_farmland existed but nothing called it:
    the point table stayed on the first farmland pass (Site A: its
    whole-area mean +0.16 mm/yr where the report reads -0.83)."""
    from insar_pipeline import deliverables as dl
    calls = []

    def fake(cfg, rep, ctx, null_map=None, out_dir=None):
        calls.append((rep, ctx, null_map, out_dir))
        return {"applied": False, "reason": "test"}
    monkeypatch.setattr(dl, "points_on_report_farmland", fake)
    out = tmp_path / "rasters"
    cr.write_rasters(computed.cfg, computed.rep, computed.ctx, out_dir=out)
    assert len(calls) == 1
    rep, ctx, null, D = calls[0]
    assert rep is computed.rep and ctx is computed.ctx and D == out
    want = cr.cell_sigma_map(computed.rep.get("cell_null_by_class"), computed.ctx.kcls, {})
    assert null.shape == computed.ctx.r_cm.shape and np.allclose(null, want, equal_nan=True)

    # a failure there leaves the rasters written and the report unharmed
    def boom(*a, **k):
        raise RuntimeError("synthetic")
    monkeypatch.setattr(dl, "points_on_report_farmland", boom)
    paths = cr.write_rasters(computed.cfg, computed.rep, computed.ctx, out_dir=tmp_path / "r2")
    assert paths and all(p.exists() for p in paths)
