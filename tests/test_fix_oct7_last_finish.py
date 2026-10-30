"""7 Oct 2026, last pass, finishing: the cross-file gaps the four groups
(report, kmz, outputs, config) left open, closed within decisions H1 and
M1-M4 and the low items of the final verification.

  M3  a revision note must replace the latest issue itself: a supersedes
      date later than it names no report the client holds (a typo such as
      2026-09-29 for 2026-09-28 would have been printed as it was), and
      the dashboard's cached gate notices an issue folder added by hand;
  E9  a feature's total at its average rate carries its 95% range on the
      page, as the KMZ gives it (the rate's range times the years);
  M4  mapped mine ground that could not be compared with farmland gives
      the range that can be stated, or says none can be given (the footer
      promises every mine-ground figure its range);
  E25 the dashboard's Results screen says "about N radar resolution
      cells", not "N independent";
  #13 a package leaves out interferogram figures of another kind of pair
      than the record's route (Site A's raw pairs of 28 Sep on its
      phase-linked record) until the figures stage moves them aside.

Each test failed on the code the groups left (release.py,
report_insights.py, client_report.py, results_screen.py, package.py and
dashboard.py with this pass's edits taken out, checked in a scratch copy).
Synthetic data in tmp_path only."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import test_fix_L7b as L7b
import test_fix_oct7_last_config as LC
from insar_pipeline import client_report as cr
from insar_pipeline import package, release
from insar_pipeline import report_insights as ri

MM_IN = 25.4


@pytest.fixture(autouse=True)
def _identity_set(monkeypatch):
    from insar_pipeline import settings
    monkeypatch.setattr(settings, "identity", lambda: {"firm": "Test Firm", "preparer": "A. Engineer", "contact": ""})


# ------------------------------------------------------------------ M3: the date a note replaces
def test_a_note_replacing_no_real_issue_is_a_hard_blocker(tmp_path):
    """Only 28 Sep was issued: a note 'replacing the report issued 29 Sep'
    would print a date no client report carries."""
    cfg, _ = LC._signed(tmp_path)
    LC._issued(tmp_path)
    cfg.report_revision = dict(LC.NOTE, supersedes="2026-09-29")
    t = next((x["text"] for x in release.hard_blockers(cfg) if x["code"] == "reissue_without_note"), "")
    assert "replaces the one issued 29 Sep 2026, but no report was issued that day" in t
    assert "the latest issue is 28 Sep 2026 (deliverables/final_20260928)" in t
    assert release.is_final(cfg) is False
    cfg.report_revision = dict(LC.NOTE)                       # the real date clears it
    assert "reissue_without_note" not in LC._codes(release.blockers(cfg))
    assert release.is_final(cfg)


# ------------------------------------------------------------------ E9: the average-rate total's range
def _feature(**kw):
    f = {"sign": -1, "mean_rel_vertical_mm_yr": -37.3, "vertical_relative_mm_yr": -37.3,
         "rate_range95_mm_yr": [-46.5, -29.0],
         "total_relative_mm": -329.0, "total_at_average_rate_mm": -390.0, "total_range95_mm": [-437.0, -258.0],
         "mines": {"key": "near_underground", "phrase": "next to mapped mine areas, 0.2 km at the closest"},
         "ground_reshaped": False,
         "chance": {"tested": True, "p": 1 / 200, "copies": 199, "copies_as_strong": 0, "stands_out": True,
                    "p_text": "less than 1 in 200"}, "stands_out_clearly": True, "cause_hint": ""}
    f.update(kw)
    return f


def test_the_average_rate_total_carries_its_range_on_the_page():
    """Site B Feature 2: the KMZ printed '-15.4 in ... at the average rate
    (95% -18.8 to -11.9 in)' and the page 'at its average rate it would have
    moved 15.4 in' bare."""
    t = ri.feature_text(2, _feature(total_at_average_rate_range95_mm=[-478.0, -303.0]))
    assert "at its average rate it would have moved 15.4 in (95% range 11.9 to 18.8 in)" in t
    # a record without the stored range: the rate's range times the years
    # the total was worked over (390 / 37.3 = 10.46 years)
    t = ri.feature_text(2, _feature())
    assert "at its average rate it would have moved 15.4 in (95% range 11.9 to 19.1 in)" in t
    # nothing to build it from: the total alone, nothing invented
    t = ri.feature_text(2, _feature(rate_range95_mm_yr=None))
    assert "at its average rate it would have moved 15.4 in" in t and "moved 15.4 in (95%" not in t


# ------------------------------------------------------------------ M4: untested mine ground
def test_untested_mine_ground_gives_the_range_that_can_be_stated():
    """Mine ground too thin for the farmland test printed no range under a
    footer promising one; a tested one keeps its test's frame."""
    up = 0.8
    sig = {"testable": False, "farmland_spread_1sigma_mm_yr": 2.0}
    rng, basis = cr.mine_ground_range({"testable": False}, sig, up, 1.5)
    assert rng == pytest.approx([1.5 - 1.96 * 2.5, 1.5 + 1.96 * 2.5], abs=0.01)
    assert "ordinary ground of its size" in basis
    assert cr.mine_ground_range({"testable": False}, {"testable": False}, up, 1.5) == (None, None)
    tested = {"testable": True, "detected": False}
    tsig = {"testable": True, "excess_mm_yr": 0.4, "threshold_95_mm_yr": 4.0, "null_median_mm_yr": 0.0}
    rng, basis = cr.mine_ground_range(tested, tsig, up, 0.5)
    assert basis is None and rng == cr.tested_range(tested, tsig, up)[0]


def test_untested_mine_ground_without_a_range_says_so():
    rep = {"features": [], "mine_ground": [
        {"key": "surface", "label": "surface mines", "area_km2": 0.4, "footprint_km2": 3.0,
         "relative_vertical_mm_yr": 1.2, "testable": False, "not_testable_reason": "too little of it was measured",
         "range95_mm_yr": None}]}
    t = next(o["text"] for o in ri.observations(rep) if o["kind"] == "mines")
    assert "could not be compared with ordinary ground (too little of it was measured), so no 95% range can be " \
           "given for it." in t
    rep["mine_ground"][0]["range95_mm_yr"] = [-3.7, 6.1]
    t = next(o["text"] for o in ri.observations(rep) if o["kind"] == "mines")
    assert "95% range -0.15 to +0.24 in/yr" in t and "no 95% range" not in t


# ------------------------------------------------------------------ E25: the Results screen
class _Widget:
    def __init__(self):
        self.shown = None

    def update(self, x=None, **kw):
        self.shown = x

    def __getattr__(self, name):
        return lambda *a, **k: None


@pytest.mark.parametrize("key", ["radar_resolution_cells", "independent_measurements"])
def test_results_screen_counts_radar_resolution_cells(key):
    from insar_pipeline import results_screen as rsc
    widgets: dict = {}
    fake = SimpleNamespace(query_one=lambda sel, *a, **k: widgets.setdefault(sel, _Widget()))
    s = {"points": 1200, key: 300, "images": 270, "first_image": "2016-01-05", "last_image": "2026-06-17",
         "years": 10.45, "mean_vertical_velocity_in_yr": 0.0496, "mean_vertical_cumulative_in": 0.5276,
         "vertical_95pct_range_in_yr": [-0.231, 0.314], "measured_change_95pct_range_in": [-2.398, 3.453],
         "signal_to_noise": 0.2, "reading": "Within normal ground variation", "testable": True, "kind": "area",
         "gnss_anchored": False}
    rsc.ResultsScreen._show(fake, SimpleNamespace(unit="in", area="Area A1", area_summary=s, stats={},
                                                  annual=[], notes=[]))
    txt = widgets["#rsummary"].shown.plain
    assert "1,200 points (about 300 radar resolution cells)" in txt and "independent" not in txt


# ------------------------------------------------------------------ #13: pair figures of another kind
def _pairs(tmp_path, caption: str):
    D = L7b._project(tmp_path, L7b._rep(), L7b._qa())
    F = D / "figures"
    F.mkdir(exist_ok=True)
    for n in ("winter_pair.png", "summer_pair.png"):
        (F / n).write_bytes(b"\x89PNG fake")
    (F / "figures.json").write_text(json.dumps([{"file": "winter_pair.png", "caption": caption},
                                                {"file": "summer_pair.png", "caption": caption},
                                                {"file": "network.png", "caption": "the network"}]))
    return D


def test_a_package_leaves_out_pair_figures_of_another_kind(tmp_path):
    """Site A's phase-linked record kept two raw ISCE interferograms of
    28 Sep, captioned 'Pairs like this are what the record is built from';
    the figures stage moves them aside on its next run, and a package made
    before then copied them into technical/figures/."""
    _pairs(tmp_path, "A raw interferogram: pairs like this are what the record is built from.")
    copies, not_copied, _ = package._plan(L7b._cfg(tmp_path), L7b._rep(), draft=True)
    dests = [d for _, d, _, _ in copies]
    assert not any(d.endswith(("winter_pair.png", "summer_pair.png")) for d in dests)
    why = [w for s, _, w in not_copied if Path(s).name == "winter_pair.png"]
    assert why and "not this record's phase-linked pairs" in why[0]


def test_a_package_keeps_pair_figures_of_the_records_kind(tmp_path):
    _pairs(tmp_path, "A phase-linked interferogram exported from MiaplPy.")
    copies, not_copied, _ = package._plan(L7b._cfg(tmp_path), L7b._rep(), draft=True)
    dests = [d for _, d, _, _ in copies]
    assert "technical/figures/winter_pair.png" in dests and "technical/figures/summer_pair.png" in dests
    # the classic route keeps raw pairs and leaves phase-linked ones out
    copies, not_copied, _ = package._plan(L7b._cfg(tmp_path, backend="isce"), L7b._rep(), draft=True)
    assert not any(d.endswith("winter_pair.png") for _, d, _, _ in copies)


# ------------------------------------------------------------------ M3: the dashboard's cached gate
def test_the_dashboard_sees_an_issue_folder_added_by_hand(tmp_path, monkeypatch):
    """release_status is cached on file times; a final_* folder copied in by
    hand changes the gate's answer (reissue_without_note) with no watched
    file touched, so the projects table kept the old status."""
    import os
    import time
    from insar_pipeline import dashboard as dash
    monkeypatch.setenv("INSAR_SETTINGS", str(tmp_path / "settings.json"))
    dash._RELEASE_CACHE.clear()
    P = tmp_path / "demo"
    (P / "deliverables").mkdir(parents=True)
    (P / "mintpy").mkdir()
    vel = P / "mintpy" / "velocity.h5"
    vel.write_bytes(b"")
    os.utime(vel, (time.time() - 3600, time.time() - 3600))
    rep = {"generated": time.strftime("%Y-%m-%d %H:%M"), "verdict": "REVIEW", "dates": ["20170101", "20200101"],
           "areas": [{"name": "Area A2", "spec": "38.35,-87.66,38.36,-87.65", "kind": "area",
                      "vertical_mm_yr": -5.67, "testable": True, "measured_percent": 90.0}]}
    (P / "deliverables" / "client_report.json").write_text(json.dumps(rep))
    (P / "quality_assessment.json").write_text(json.dumps({"checks": {}, "send_rule": {"value": "send"}}))
    try:
        codes = lambda: {b["code"] for b in (dash.release_status(P) or {}).get("blockers", [])}   # noqa: E731
        assert "reissue_without_note" not in codes()
        old = dict(rep, areas=[dict(rep["areas"][0], vertical_mm_yr=-4.58)])
        d = P / "deliverables" / "final_20260928" / "technical"
        d.mkdir(parents=True)
        (d / "client_report.json").write_text(json.dumps(old))
        assert "reissue_without_note" in codes()
    finally:
        dash._RELEASE_CACHE.clear()
