"""
Decision D2 (7 Oct 2026): a check that cannot apply is not counted. The QA
verdict counts flagged checks out of the APPLICABLE checks ("2 of 5
applicable checks flagged; 1 not applicable"); the distributed-scatterer
bias check with one far-field coherence class is not applicable - stated,
never counted as untested or as a pass - so MEETS PRACTICE is reachable on
such a record. The QA record, its printed report and the figures' summary
table all count this way. Each test fails on the code in
backups/pre_fixes_20261007/ and passes now. No network.

Run: python -m pytest insar_pipeline/tests/test_fix_oct7_dec_D2_verdict.py -q -p no:cacheprovider
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from insar_pipeline import quality_assessment as qa                 # noqa: E402
from insar_pipeline import stack_qc                                  # noqa: E402

import test_fix_L8 as L8                                             # noqa: E402  (the QA project builder)

NA = {"pass": None, "not_applicable": "the far field holds one coherence class (0.9-1.0)"}


def _record(**checks):
    return {"checks": checks}


@pytest.fixture(autouse=True)
def no_osm(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("the QA tried to fetch OSM water in a test")
    monkeypatch.setattr(stack_qc, "fetch_osm_water", boom, raising=False)


# ------------------------------------------------------------------ counting
def test_a_check_that_does_not_apply_is_named_and_not_counted():
    r = _record(reference_stability={"pass": False}, rate_precision={"pass": False}, coverage={"pass": True},
                network={"pass": True}, single_epoch_noise={"pass": True}, distributed_scatterer_bias=dict(NA))
    c = qa.check_counts(r)
    assert (c["passed"], c["flagged"], c["untested"], c["applicable"]) == (3, 2, 0, 5)
    assert c["not_applicable"] == ["distributed_scatterer_bias"]
    assert c["text"] == "2 of 5 applicable checks flagged; 1 not applicable (distributed scatterer bias)"
    assert c["head"] == "2 of 5 applicable checks flagged"


def test_a_check_that_applies_but_was_not_run_stays_untested():
    # pass None with no not_applicable reason: applicable, could not be run
    c = qa.check_counts(_record(coverage={"pass": True}, network={"pass": None},
                                distributed_scatterer_bias={"pass": None, "velocity_by_coherence_class": []}))
    assert (c["passed"], c["flagged"], c["untested"], c["applicable"]) == (1, 0, 2, 3)
    assert c["not_applicable"] == [] and c["text"] == "0 of 3 applicable checks flagged, 2 could not be run"


def test_a_flagged_check_is_counted_even_with_a_stray_reason():
    # only a check with no result can be 'not applicable'
    c = qa.check_counts(_record(coverage={"pass": False, "not_applicable": "x"}))
    assert c["flagged"] == 1 and c["applicable"] == 1 and c["not_applicable"] == []


def test_numpy_booleans_are_counted():
    # the old count tested `is True` / `is False`, so a numpy boolean was
    # neither a pass, a flag nor untested: it vanished from the count
    c = qa.check_counts(_record(coverage={"pass": np.True_}, network={"pass": np.False_}))
    assert (c["passed"], c["flagged"], c["applicable"]) == (1, 1, 2)


# ------------------------------------------------------------------ verdict
def test_meets_practice_is_reachable_with_a_check_that_does_not_apply():
    c = qa.check_counts(_record(coverage={"pass": True}, network={"pass": True}, distributed_scatterer_bias=dict(NA)))
    assert qa.verdict(c) == "MEETS PRACTICE"
    # an untested check still keeps it from meeting practice
    c = qa.check_counts(_record(coverage={"pass": True}, network={"pass": None}))
    assert qa.verdict(c) == "USABLE WITH LIMITATIONS"
    c = qa.check_counts(_record(**{k: {"pass": False} for k in ("a", "b", "c")}, d=dict(NA)))
    assert qa.verdict(c) == "REVIEW"
    # thin farmland is REVIEW whatever the checks say
    assert qa.verdict(qa.check_counts(_record(coverage={"pass": True})), thin=True) == "REVIEW"


def test_nothing_applicable_does_not_meet_practice():
    # the old rule called a record with no applicable check MEETS PRACTICE
    assert qa.verdict(qa.check_counts(_record(distributed_scatterer_bias=dict(NA)))) == "REVIEW"
    assert qa.verdict(qa.check_counts({})) == "REVIEW"


def test_evaluate_counts_applicable_checks_only(tmp_path):
    r = qa.evaluate(L8.build(tmp_path, tc=np.full((L8.NY, L8.NX), 0.95), thr=0.7))
    ds = r["checks"]["distributed_scatterer_bias"]
    assert ds["pass"] is None and ds["not_applicable"]
    assert r["not_applicable"] == ["distributed_scatterer_bias"]
    assert r["untested"] == 0 and r["applicable"] == r["passed"] + r["flagged"] == len(r["checks"]) - 1
    assert r["checks_text"] == qa.check_counts(r)["text"] and "1 not applicable" in r["checks_text"]
    assert r["verdict"] == "MEETS PRACTICE"            # USABLE WITH LIMITATIONS before D2
    # still stated in the send rule, never a hold
    assert any("distributed scatterer bias check does not apply" in n for n in r["send_rule"]["notes"])
    assert not any("distributed scatterer bias" in h for h in r["send_rule"]["hold_reasons"])
    out = qa.format_report(r)
    assert "[N/A ] distributed_scatterer_bias" in out
    assert (f"{r['passed']} of {r['applicable']} applicable checks meet practice, 0 flagged, 0 untested; "
            "not applicable: distributed scatterer bias -> MEETS PRACTICE") in out


def test_format_report_says_not_run_rather_than_not_applicable():
    r = {"checks": {"coverage": {"pass": True}, "network": {"pass": None},
                    "distributed_scatterer_bias": dict(NA)},
         "findings": {}, "verdict": "USABLE WITH LIMITATIONS"}
    out = qa.format_report(r)
    assert "[NOT RUN] network" in out and "[N/A ] distributed_scatterer_bias" in out
    assert "1 of 2 applicable checks meet practice, 0 flagged, 1 untested" in out


# ------------------------------------------------------------------ figures' summary table
def _table_rows(tmp_path, monkeypatch, Q):
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
    figures.summary_table(cfg, Q, cfg.project_dir / "deliverables" / "figures")
    return seen["rows"]


def test_summary_table_counts_applicable_checks_and_names_the_rest(tmp_path, monkeypatch):
    Q = {"verdict": "MEETS PRACTICE", "passed": 4, "flagged": 0, "send_rule": {"value": "send with notes"},
         "checks": {k: {"pass": True} for k in ("reference_stability", "rate_precision", "single_epoch_noise",
                                                "coverage")} | {"distributed_scatterer_bias": dict(NA)},
         "findings": {}}
    rows = _table_rows(tmp_path, monkeypatch, Q)
    i = next(n for n, r in enumerate(rows) if r[0] == "Verdict")
    assert rows[i][1] == "MEETS PRACTICE" and rows[i][2] == "0 of 4 applicable checks flagged"
    assert rows[i + 1][0].startswith("   not applicable") and rows[i + 1][1] == "1 check"
    assert rows[i + 1][2] == "distributed scatterer bias"
    # its own row says so: the old row read "Meets practice" for any
    # distributed-scatterer record without a result
    ds = next(r for r in rows if r[0].startswith("Distributed-scatterer bias"))
    assert ds[3] == "Not applicable"


def test_summary_table_does_not_pass_a_ds_bias_check_that_was_not_run(tmp_path, monkeypatch):
    # no class in the far field (pass None, no reason it cannot apply): not
    # measured, never "Meets practice" as the old row read
    Q = {"verdict": "USABLE WITH LIMITATIONS", "checks": {
        "coverage": {"pass": True},
        "distributed_scatterer_bias": {"pass": None, "velocity_by_coherence_class": [],
                                       "low_minus_high_coherence_mm_yr": None}}, "findings": {}}
    rows = _table_rows(tmp_path, monkeypatch, Q)
    ds = next(r for r in rows if r[0].startswith("Distributed-scatterer bias"))
    assert ds[3] == "Not measured"
    i = next(n for n, r in enumerate(rows) if r[0] == "Verdict")
    assert rows[i][2] == "0 of 2 applicable checks flagged, 1 could not be run"


def test_summary_table_says_an_untested_check(tmp_path, monkeypatch):
    # the old row read "P of P+F checks meet practice", silent on a check
    # that could not be run
    Q = {"verdict": "USABLE WITH LIMITATIONS", "passed": 1, "flagged": 1, "untested": 1,
         "checks": {"coverage": {"pass": True}, "network": {"pass": None}, "rate_precision": {"pass": False}},
         "findings": {}}
    rows = _table_rows(tmp_path, monkeypatch, Q)
    i = next(n for n, r in enumerate(rows) if r[0] == "Verdict")
    assert rows[i][2] == "1 of 3 applicable checks flagged, 1 could not be run"
    assert i == len(rows) - 1                           # no not-applicable line when every check applies
