"""
Group I4_qa_figures (7 Oct 2026, integration round): the reason a date the
started stack's plan could not take is given, the QA's reading of stale
cross-check / injection records, the coherence caption's noise level and the
summary table's coherence-bias row. Each test fails on the code before this
round and passes now. No network.

Run: python -m pytest insar_pipeline/tests/test_fix_oct7_int_I4_qa_figures.py -q -p no:cacheprovider
"""
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from insar_pipeline import quality_assessment as qa                 # noqa: E402
from insar_pipeline.config import PipelineConfig                    # noqa: E402

PROCESSED = ["20191208", "20191220", "20200125", "20200206"]
NOT_IN_PLAN = "found in the archive only after this project's processing had started"


def _cfg(tmp_path, **over):
    kw = dict(project_dir=tmp_path / "proj", bbox=[-87.8, 38.2, -87.52, 38.47], start_date="2019-06-01",
              end_date="2020-06-30", backend="miaplpy")
    kw.update(over)
    cfg = PipelineConfig(**kw)
    cfg.mintpy_dir.mkdir(parents=True, exist_ok=True)
    cfg.slc_dir.mkdir(parents=True, exist_ok=True)
    return cfg


# ------------------------------------------------------------------ (a) 'not_in_plan' in plain words
def test_not_in_plan_reason_is_plain_and_the_two_readers_agree(tmp_path):
    """the 'dropped' code (short_date_reasons) and the accepted list
    (date_accounting's beyond-plan branch) give the same client sentence."""
    why = qa._DROP_REASON["not_in_plan"]
    assert NOT_IN_PLAN in why and "not in this project's processing plan" not in why
    # via assembly_status 'dropped'
    cfg = _cfg(tmp_path, allow_missing_dates=("20200101",))
    (cfg.slc_dir / "assembly_status.json").write_text(json.dumps(
        {"catalogue_dates": PROCESSED + ["20200101"], "dropped": {"20200101": "not_in_plan"},
         "accepted_not_in_plan": ["20200101"], "failed": {}}))
    assert qa.short_date_reasons(cfg) == {"20200101": why}
    a = qa.date_accounting(cfg, PROCESSED)["dropped"]["20200101"]
    # via 'accepted_not_in_plan' only (a record without the code)
    cfg2 = _cfg(tmp_path / "b", allow_missing_dates=("20200101",))
    (cfg2.slc_dir / "assembly_status.json").write_text(json.dumps(
        {"catalogue_dates": PROCESSED + ["20200101"], "accepted_not_in_plan": ["20200101"], "failed": {}}))
    b = qa.date_accounting(cfg2, PROCESSED)["dropped"]["20200101"]
    assert a == b and a.startswith(why) and "accepted as missing" in a and "incomplete" not in a


def test_an_older_qa_calling_a_not_in_plan_date_short_is_reworded(tmp_path):
    cfg = _cfg(tmp_path)
    (cfg.slc_dir / "assembly_status.json").write_text(json.dumps({"dropped": {"20200101": "not_in_plan"}}))
    old = {"dropped": {"20200101": qa._SEARCH_SHORT}}
    assert NOT_IN_PLAN in qa.refresh_drop_reasons(cfg, old)["dropped"]["20200101"]


def test_run_summary_page_names_a_not_in_plan_date_in_words(tmp_path):
    from insar_pipeline.scripts import run_summary_page as rsp
    assert rsp._DROP_WHY["not_in_plan"] == qa._DROP_REASON["not_in_plan"]
    h = rsp.build({"project": str(tmp_path), "assembly_status": {"dropped": {"20200101": "not_in_plan"}}})
    li = h[h.index("acquisition(s) not used"):]
    li = li[:li.index("</li>")]
    assert "found in the archive only after" in li and ": not_in_plan" not in li
    # the burst-count sentence is for short dates only
    assert "fewer bursts than the rest" not in li
    h = rsp.build({"project": str(tmp_path), "assembly_status": {"dropped": {"20200101": "not_in_plan",
                                                                             "20200113": "catalogue_short"}}})
    assert "fewer bursts than the rest" in h


# ------------------------------------------------------------------ (b) stale or switched-off records
def _records(cfg, generated):
    T = cfg.project_dir / "deliverables" / "technical"
    T.mkdir(parents=True, exist_ok=True)
    (T / "injection_recovery.json").write_text(json.dumps(
        {"generated": generated, "status": "complete", "sentence": "Found down to 1 km2."}))
    (T / "cross_check.json").write_text(json.dumps(
        {"generated": generated, "status": "complete", "available": True, "sentence": "Agrees."}))


def _velocity(cfg, mtime):
    v = cfg.mintpy_dir / "velocity.h5"
    v.write_bytes(b"")
    os.utime(v, (mtime, mtime))


@pytest.mark.parametrize("fn,option", [(qa._injection_record, "injection_test"),
                                       (qa._independent_check, "cross_check_disp")])
def test_a_record_older_than_the_velocity_reads_as_not_run(tmp_path, fn, option):
    cfg = _cfg(tmp_path, **{option: True})
    now = time.time()
    _records(cfg, datetime.fromtimestamp(now - 86400).isoformat(timespec="seconds"))
    _velocity(cfg, now)
    assert fn(cfg) is None                                  # read before as this run's check
    _records(cfg, datetime.fromtimestamp(now + 60).isoformat(timespec="seconds"))
    rec = fn(cfg)
    assert rec is not None and rec["status"] == "complete" and rec["gates_verdict"] is False


@pytest.mark.parametrize("fn,option", [(qa._injection_record, "injection_test"),
                                       (qa._independent_check, "cross_check_disp")])
def test_a_record_left_after_the_option_was_turned_off_reads_as_not_run(tmp_path, fn, option):
    cfg = _cfg(tmp_path, **{option: False})
    now = time.time()
    _velocity(cfg, now - 86400)
    _records(cfg, datetime.fromtimestamp(now).isoformat(timespec="seconds"))
    assert fn(cfg) is None


# ------------------------------------------------------------------ (c) the caption quotes the QA's floor
def test_noise_sentence_prefers_the_qa_floor():
    from insar_pipeline import figures
    t = figures._noise_sentence({"noise_level": {"linked_noise_level_from_run": 0.384, "floor": 0.436}}, 0.4)
    # the threshold is judged against the floor; the typical level is quoted
    # as typical, never as where the threshold is measured from (7 Oct 2026)
    assert "up to about 0.436" in t and "-0.036" in t and "not reliable" in t and "about 0.384" in t
    # the run's level only when the QA recorded no floor
    t = figures._noise_sentence({"noise_level": {"linked_noise_level_from_run": 0.387, "floor": None}}, 0.4)
    assert "0.387" in t and "+0.013" in t
    assert figures._noise_sentence({}, 0.4) == ""


# ------------------------------------------------------------------ (d) the summary table's coherence-bias row
@pytest.mark.parametrize("ds,reading", [
    ({"low_minus_high_coherence_mm_yr": None, "pass": None, "not_applicable": "one coherence class"},
     "Not applicable"),
    ({"low_minus_high_coherence_mm_yr": None, "pass": None, "velocity_by_coherence_class": []}, "Not measured"),
    ({}, "Not measured"),
    ({"low_minus_high_coherence_mm_yr": -0.5, "pass": True}, "Meets practice"),
    ({"low_minus_high_coherence_mm_yr": -4.0, "pass": False}, "Flagged; absolute rates biased"),
])
def test_summary_table_ds_bias_row_reads_what_was_measured(tmp_path, monkeypatch, ds, reading):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.axes import Axes
    from insar_pipeline import figures
    import test_fix_L9c as L9c                                       # the figures project builder
    seen = {}
    real = Axes.table

    def spy(self, *a, **k):
        seen["rows"] = k.get("cellText")
        return real(self, *a, **k)
    monkeypatch.setattr(Axes, "table", spy)
    cfg = L9c.make_project(tmp_path)
    Q = {"checks": {"distributed_scatterer_bias": ds}, "findings": {}}
    figures.summary_table(cfg, Q, cfg.project_dir / "deliverables" / "figures")
    row = [r for r in seen["rows"] if r[0].startswith("Distributed-scatterer bias")]
    assert len(row) == 1 and row[0][3] == reading
