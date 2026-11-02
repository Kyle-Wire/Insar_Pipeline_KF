"""7 Oct 2026 trust review follow-up, group display (client_kmz.py,
dashboard.py, results_screen.py): decisions F2 and F8. Each test here fails
on the code in backups/pre_followup_20261007/ unless its docstring says
otherwise.

F2  A stand-out found only by the quality assessment is shown to the client
    only when the report lists it as a feature. The KMZ drew it as
    "Deepest patch (quality assessment) ... Worth checking on the ground"
    and the dashboard's run view printed "One area to check on the ground"
    whenever the whole area held a stand-out, also beside a page saying no
    feature stands out (Site A's 4 km stand-out once Feature 1 was no
    longer listed). The run view now lists the report's own features
    (client_report.json 'features'); the KMZ draws the stand-out only as the
    feature that holds it.
F8  The dashboard prints area totals with their 95% range (the run view and
    the Results screen printed them bare; E9: a total is never printed
    without its range). An area that could not be compared with farmland
    also shows its rate's untested range (the report's, said so), so a
    ranged total never sits beside a rate reading "no 95% range".
"""
import copy
import json
import re
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest
from rich.text import Text

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_fix_L9b as L9b                                       # noqa: E402  (the KMZ's synthetic project)
from insar_pipeline import dashboard as dash                     # noqa: E402

MM_IN = 25.4
# the whole area's stand-out as client_report writes it, held by no listed feature
QA_ONLY = {"lat": 38.2401, "lon": -87.6462, "scale_km": 4.0, "vertical_relative_mm_yr": -14.3,
           "vertical_mm_yr": -15.3, "next_deepest_vertical_mm_yr": -9.0, "gap_in_spreads": 2.2,
           "gap_threshold": 2.02, "p_calibrated": 0.03, "in_mapped_mine": False, "inside_mapped_panels": False,
           "within_panel_buffer": False, "buffer_m": 2000.0, "feature": None, "signal_to_noise": 5.9}


# ------------------------------------------------------------------ F2: the KMZ
@pytest.fixture(scope="module")
def kmz_project(tmp_path_factory):
    """L9b's synthetic project (a planted settling patch the report lists as
    Feature 1) and its client report, computed once, nothing written to
    its deliverables."""
    from insar_pipeline import client_report as cr
    cfg = L9b.make_project(tmp_path_factory.mktemp("fu_display"))
    rep, ctx, _, _ = cr.compute(cfg)
    rep = json.loads(json.dumps(rep, default=str))
    assert rep["features"], "the planted patch is a feature in the report"
    return SimpleNamespace(cfg=cfg, rep=rep, ctx=ctx)


def _kml(p, rep, out: Path, monkeypatch) -> str:
    from insar_pipeline import client_kmz as ck
    monkeypatch.setattr(ck, "PER_AREA", 2)
    monkeypatch.setattr(ck, "PER_FEATURE", 1)
    k = ck.build(p.cfg, out_dir=out, rep=rep, ctx=p.ctx)
    return zipfile.ZipFile(k).read("doc.kml").decode()


def _with_standout(rep, so, features):
    r = copy.deepcopy(rep)
    full = next(a for a in r["areas"] if a.get("kind") == "full")
    full["standout"] = so
    r["features"] = features
    return r


def test_kmz_draws_no_stand_out_the_report_does_not_list(kmz_project, tmp_path, monkeypatch):
    """Site A's case: the QA's stand-out lies in no listed feature. The
    old KMZ drew it as 'Deepest patch (quality assessment) ... Worth
    checking on the ground'; it now stays out of the client file, and the
    report's own features are drawn as before."""
    feats = [dict(f, quality_standout=False) for f in kmz_project.rep["features"]]
    kml = _kml(kmz_project, _with_standout(kmz_project.rep, QA_ONLY, feats), tmp_path / "a", monkeypatch)
    assert "Deepest patch" not in kml and "Worth checking on the ground" not in kml
    assert "quality assessment found" not in kml
    folder = re.search(r"<Folder><name>Features</name>(.*?)</Folder>", kml, re.S).group(1)
    assert re.findall(r"<Placemark><name>(Feature \d+)</name>", folder) == [
        f"Feature {i}" for i in range(1, len(feats) + 1)]


def test_kmz_with_no_listed_feature_says_so_and_draws_no_patch(kmz_project, tmp_path, monkeypatch):
    """No feature listed, a QA stand-out: the Features folder carries the
    page's own 'no feature stands out' words and nothing else is drawn as a
    place to check."""
    kml = _kml(kmz_project, _with_standout(kmz_project.rep, QA_ONLY, []), tmp_path / "b", monkeypatch)
    assert "Deepest patch" not in kml and "Worth checking on the ground" not in kml
    folder = re.search(r"<Folder><name>Features</name>(.*?)</Folder>", kml, re.S).group(1)
    assert "<Placemark>" not in folder and "No patch of ground larger than a few fields" in folder
    names = re.findall(r"<Placemark><name>([^<]+)</name>", kml)
    assert not any("patch" in n.lower() for n in names), names


def test_kmz_shows_a_listed_stand_out_as_its_feature(kmz_project, tmp_path, monkeypatch):
    """The stand-out the report lists (Site B: the 1 km stand-out is in
    Feature 1) is drawn once, as that feature, with the report's words.
    (Passes on the old code too: it guards the listed case.)"""
    feats = copy.deepcopy(kmz_project.rep["features"])
    feats[0]["quality_standout"] = True
    so = dict(QA_ONLY, lat=feats[0]["lat"], lon=feats[0]["lon"], scale_km=1.0, feature=1)
    kml = _kml(kmz_project, _with_standout(kmz_project.rep, so, feats), tmp_path / "c", monkeypatch)
    assert "Deepest patch" not in kml
    assert "<Placemark><name>Feature 1</name>" in kml


# ------------------------------------------------------------------ F2 + F8: the dashboard's run view
class _Static:
    def __init__(self):
        self.shown = None

    def update(self, x):
        self.shown = x


def _run_view(P: Path, rep: dict, monkeypatch) -> str:
    """RunScreen._results on a project folder holding `rep`, outside an app."""
    D = P / "deliverables"
    D.mkdir(parents=True, exist_ok=True)
    (D / "client_report.json").write_text(json.dumps(rep))
    monkeypatch.setattr(dash, "release_status", lambda *a, **k: None)
    w = _Static()
    fake = SimpleNamespace(cfg=SimpleNamespace(project_dir=str(P)), query_one=lambda *a, **k: w)
    dash.RunScreen._results(fake, P)
    return w.shown.plain if isinstance(w.shown, Text) else str(w.shown)


AREA = {"name": "Area A1", "kind": "area", "anchored_to_gnss": False, "vertical_mm_yr": 1.26,
        "range95_mm_yr": [-5.87, 7.98], "total_mm": 13.4, "total_range95_mm": [-60.9, 87.7],
        "testable": True, "detected": False, "reading": "Within normal ground variation"}
FULL = {"name": "Full Satellite Imaging", "kind": "full", "anchored_to_gnss": False, "vertical_mm_yr": None,
        "total_mm": None, "range95_mm_yr": None, "reading": "Relative only: no regional anchor",
        "standout": QA_ONLY}
FEATURE = {"sign": -1, "area_km2": 2.21, "mean_rel_vertical_mm_yr": -35.1, "vertical_relative_mm_yr": -37.3,
           "lat": 38.3044, "lon": -87.3843, "stands_out": True, "stands_out_clearly": True,
           "mines": {"key": "near_mapped", "inside": {"mapped": 0.0}}}


def test_run_view_shows_no_quality_only_stand_out(tmp_path, monkeypatch):
    """The run view printed 'One area to check on the ground' for a stand-out
    the report does not list, beside a page that says no feature stands
    out; it now reads the report's feature list."""
    rep = {"verdict": "REVIEW", "anchored": False, "areas": [FULL, AREA], "features": [],
           "features_assessed": True, "generated": "2026-10-07"}
    txt = _run_view(tmp_path / "p", rep, monkeypatch)
    assert "to check on the ground" not in txt
    assert "No local feature stands out from ordinary farmland" in txt


def test_run_view_lists_the_reports_features_with_the_pages_verdict(tmp_path, monkeypatch):
    """Two listed features: both named with the report's number, place,
    size, rate and the page's verdict ('stands out clearly' only with
    stands_out_clearly, else 'worth a look'); a stand-out held by Feature 1
    is not counted again."""
    f2 = dict(FEATURE, area_km2=6.03, lat=38.2741, lon=-87.3506, vertical_relative_mm_yr=-25.7,
              stands_out_clearly=False)
    full = dict(FULL, standout=dict(QA_ONLY, lat=38.3044, lon=-87.3843, feature=1))
    rep = {"verdict": "REVIEW", "anchored": False, "areas": [full, AREA], "features": [FEATURE, f2],
           "features_assessed": True, "generated": "2026-10-07"}
    txt = _run_view(tmp_path / "p", rep, monkeypatch)
    assert "2 features to check on the ground" in txt and "One area to check" not in txt
    assert ("Feature 1: sinking ground, about 2.2 km² near 38.304 N, 87.384 W; -1.47 in/yr relative to the "
            "surrounding farmland · stands out clearly") in txt
    assert "Feature 2: sinking ground, about 6.0 km² near 38.274 N, 87.351 W; -1.01 in/yr" in txt
    assert txt.rstrip().count("worth a look") == 1
    # one listed feature, not tested against chance
    one = dict(rep, features=[dict(FEATURE, stands_out=None, stands_out_clearly=False)])
    txt = _run_view(tmp_path / "q", one, monkeypatch)
    assert "One feature to check on the ground" in txt and "not tested against chance" in txt


def test_run_view_prints_each_total_with_its_95pct_range(tmp_path, monkeypatch):
    """F8: '+0.53 in total' was printed bare. Now with the report's range,
    the untested range for an area that could not be compared, and a
    pointer to rebuild a report that predates the ranges."""
    untested = {"name": "Site B project area", "kind": "area", "anchored_to_gnss": True, "vertical_mm_yr": 0.13,
                "total_mm": 1.4, "total_range95_mm": None, "untested_total_range95_mm": [-30.2, 33.0],
                "untested_range95_mm_yr": [-2.89, 3.15],
                "range95_mm_yr": None, "testable": False, "reading": "Not testable: too little farmland"}
    old = {"name": "Area A2", "kind": "area", "anchored_to_gnss": True, "vertical_mm_yr": -5.67,
           "range95_mm_yr": [-20.4, 9.1], "total_mm": -66.9, "testable": True, "reading": "Within normal ground variation"}
    rep = {"verdict": "REVIEW", "anchored": True, "areas": [AREA, untested, old], "generated": "2026-10-07"}
    txt = _run_view(tmp_path / "p", rep, monkeypatch)
    assert "+0.53 in total (95% range -2.40 to +3.45 in)" in txt
    assert "+0.06 in total (95% range -1.19 to +1.30 in)" in txt
    # its rate too: the range that can be stated, said so, not a bare figure beside a ranged total
    assert "likely range -0.11 to +0.12 in/yr, not compared with farmland" in txt
    assert "-2.63 in total (no 95% range: this report was built before totals carried one; rebuild it)" in txt
    # no date in it: the trust-check copies' reports without the fields were generated on 7 Oct itself
    assert "before 7 Oct" not in txt
    assert not re.search(r"in total\s*\n", txt)                 # never a bare total
    assert dash.total_text({"total_mm": None}) == ""
    assert dash.total_text({"total_mm": -0.1, "total_range95_mm": [-0.2, 0.1]}) == \
        "0.00 in total (95% range -0.01 to 0.00 in)"               # no '-0.00'


# ------------------------------------------------------------------ F8: the Results screen
class _Widget:
    """A stand-in for the screen's widgets: update() is kept, the rest is
    accepted and ignored."""
    def __init__(self):
        self.shown = None
        self.display = True
        self.disabled = False

    def update(self, x=None, **kw):
        self.shown = x

    def __getattr__(self, name):
        return lambda *a, **k: None


def test_results_screen_prints_the_window_total_with_its_range():
    """The Results screen printed '(+0.53 in measured over the window)'
    bare beside a ranged rate; results.area_summary carries the report's
    range of that total (measured_change_95pct_range_<unit>)."""
    from insar_pipeline import results_screen as rsc
    widgets: dict = {}
    fake = SimpleNamespace(query_one=lambda sel, *a, **k: widgets.setdefault(sel, _Widget()))
    s = {"points": 1200, "independent_measurements": 300, "images": 270, "first_image": "2016-01-05",
         "last_image": "2026-06-17", "years": 10.45, "mean_vertical_velocity_in_yr": 0.0496,
         "mean_vertical_cumulative_in": 0.5276, "vertical_95pct_range_in_yr": [-0.231, 0.314],
         "measured_change_95pct_range_in": [-2.398, 3.453], "signal_to_noise": 0.2,
         "reading": "Within normal ground variation", "testable": True, "kind": "area", "gnss_anchored": False}
    res = SimpleNamespace(unit="in", area="Area A1", area_summary=s, stats={}, annual=[], notes=[])
    rsc.ResultsScreen._show(fake, res)
    txt = widgets["#rsummary"].shown.plain
    assert "+0.53 in measured over the window, 95% range -2.40 to +3.45 in" in txt
    # a total the record gives no range for says so, never bare
    s2 = dict(s, measured_change_95pct_range_in=None)
    rsc.ResultsScreen._show(fake, SimpleNamespace(unit="in", area="X", area_summary=s2, stats={}, annual=[], notes=[]))
    assert "+0.53 in measured over the window, no 95% range" in widgets["#rsummary"].shown.plain
    assert rsc.total_text({"mean_vertical_cumulative_mm": None}, "mm") == ""
    # an area that could not be compared with farmland (Site B's project area): its total's range and its
    # rate's untested range, never "no 95% range" beside a ranged total
    s3 = dict(s, vertical_95pct_range_in_yr=None, untested_vertical_95pct_range_in_yr=[-0.459, 0.459],
              mean_vertical_velocity_in_yr=0.0, mean_vertical_cumulative_in=0.0,
              measured_change_95pct_range_in=[-4.80, 4.80], testable=False, signal_to_noise=None,
              reading="Not testable: too little farmland")
    rsc.ResultsScreen._show(fake, SimpleNamespace(unit="in", area="P", area_summary=s3, stats={}, annual=[], notes=[]))
    txt = widgets["#rsummary"].shown.plain
    assert "95% range -0.459 to +0.459 in/yr, not compared with farmland" in txt and "no 95% range" not in txt
    assert "0.00 in measured over the window, 95% range -4.80 to +4.80 in" in txt
