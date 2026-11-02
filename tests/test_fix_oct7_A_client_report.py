"""7 Oct 2026 fixes, group A_client_report (client_report.py): findings 2/7,
3, 4, 5, 6 and 17 of the 5 Oct review, and the client-report side of the
inside-station GNSS contract (each inside_checks entry carries `consistent`,
which release.blockers reads). Each test checks the OUTCOME on synthetic data
(the release gate's blockers, the page's words, what the PDF and the charts
draw, what is left in the deliverables folder) and failed on the code before
these fixes.

Run: python -m pytest insar_pipeline/tests/test_fix_oct7_A_client_report.py -q -p no:cacheprovider
"""
import json
import re
import sys
import types
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_fix_L5 as L5                                          # noqa: E402  (the client-report project builder)

from insar_pipeline import client_report as cr                   # noqa: E402

# a station inside the study area, in the synthetic tie's terms (LOS mm/yr):
# the station moved with the region (its rate minus the background is 0), the
# satellite reads the ground around it 20 mm/yr against its farmland - far
# beyond the check's +/- 4 mm/yr - or 0.3 (agrees)
BG_LOS = L5.BG_V * L5.UP
DISAGREE = {"site": "INSD", "insar_relative_class_matched_mm_yr": 20.0, "threshold_95_mm_yr": 4.0,
            "gnss_los_rate_over_overlap_mm_yr": BG_LOS, "overlap": [2016.0, 2019.0]}
AGREE = dict(DISAGREE, site="INOK", insar_relative_class_matched_mm_yr=0.3)


def _build(root: Path, gnss: bool, inside=()):
    cfg = L5.make_project(root, gnss=gnss)
    if gnss and inside:
        p = cfg.project_dir / "deliverables" / "gnss_tie.json"
        tie = json.loads(p.read_text())
        tie["inside_checks"] = [dict(c) for c in inside]
        p.write_text(json.dumps(tie))
    rep, ctx, masks, outlines = cr.compute(cfg)
    return types.SimpleNamespace(cfg=cfg, rep=rep, ctx=ctx, masks=masks, outlines=outlines)


@pytest.fixture(scope="module")
def anchored(tmp_path_factory):
    """A GNSS-anchored project with a disagreeing station inside it."""
    return _build(tmp_path_factory.mktemp("a_anch"), gnss=True, inside=[DISAGREE])


@pytest.fixture(scope="module")
def relative(tmp_path_factory):
    """No GNSS tie: relative only, like Site A until its tie is re-run."""
    return _build(tmp_path_factory.mktemp("a_rel"), gnss=False)


def _hard(blockers, code):
    return [b for b in blockers if b.get("code") == code and b.get("hard")]


# ------------------------------------------------------------------ 2 / 7: the inside-station check reaches the gate
def test_inside_station_verdict_is_stored_by_the_pages_rule_and_blocks_release(tmp_path):
    """release.blockers blocks on inside_checks[].consistent is False, and
    nothing wrote that key: a page saying 'they differ' could be signed off.
    The verdict is now stored by the rule the page prints."""
    from insar_pipeline import release
    rep = {"los_up_median": 0.78, "control_ground_label": "farmland", "areas": [], "dates": ["20160105", "20250320"],
           "gnss": {"los_up_median": 0.78, "background": {"los_mm_yr": 0.0},
                    "inside_checks": [{"site": "X", "insar_relative_class_matched_mm_yr": 6.0, "threshold_95_mm_yr": 1.0,
                                       "gnss_los_rate_over_overlap_mm_yr": -3.0},
                                      {"site": "Y", "insar_relative_class_matched_mm_yr": 0.2, "threshold_95_mm_yr": 1.0,
                                       "gnss_los_rate_over_overlap_mm_yr": 0.0},
                                      {"site": "Z", "insar_relative_class_matched_mm_yr": None}]}}
    cr.mark_inside_checks(rep)
    x, y, z = rep["gnss"]["inside_checks"]
    assert x["consistent"] is False and y["consistent"] is True and "consistent" not in z
    # the page's own verdict, the same one
    assert cr.station_check(x, rep)["agrees"] is False and cr.station_check(y, rep)["agrees"] is True
    got = _hard(release.blockers(tmp_path, rep, {}), "gnss_inside_inconsistent")
    assert len(got) == 1 and "X" in got[0]["text"]


def test_compute_marks_inside_stations_and_the_report_is_blocked(anchored):
    rep = anchored.rep
    chk = rep["gnss"]["inside_checks"]
    assert [c["site"] for c in chk] == ["INSD"] and chk[0]["consistent"] is False
    assert _hard(rep["release"]["blockers"], "gnss_inside_inconsistent")
    page = L5._text(cr.page(anchored.cfg, json.loads(json.dumps(rep)), L5._fake_figs(anchored)))
    assert "they differ by more than this check can explain" in page
    # an agreeing station: consistent, no blocker
    rep2 = json.loads(json.dumps(rep))
    rep2["gnss"]["inside_checks"] = [dict(AGREE)]
    cr.mark_inside_checks(rep2)
    assert rep2["gnss"]["inside_checks"][0]["consistent"] is True
    from insar_pipeline import release
    assert not _hard(release.blockers(anchored.cfg, rep2, {}), "gnss_inside_inconsistent")


# ------------------------------------------------------------------ 3: the short answer's first feature is not 'the largest'
def _two_features(rep):
    f = rep["features"][0]
    f1 = dict(json.loads(json.dumps(f)), area_km2=2.2, quality_standout=True, stands_out=True)
    f2 = dict(json.loads(json.dumps(f)), area_km2=6.0, quality_standout=False, stands_out=True,
              lat=f["lat"] - 0.02, lon=f["lon"] + 0.02)
    return [f1, f2]


def test_short_answer_names_the_first_feature_by_number_not_as_the_largest(anchored):
    """Features are listed by QA stand-out, then rate x area: Site B's
    Feature 1 (2.2 km²) was called 'the largest' beside a 6.0 km² Feature 2."""
    rep = json.loads(json.dumps(anchored.rep))
    assert rep["features"], "the planted settling patch must be found"
    rep["features"] = _two_features(rep)
    t = L5._text(cr.page(anchored.cfg, rep, L5._fake_figs(anchored)))
    assert "the largest" not in t
    assert "Feature 1, about 2.2 km² near" in t
    # one feature: still 'the feature'
    rep["features"] = rep["features"][:1]
    t = L5._text(cr.page(anchored.cfg, rep, L5._fake_figs(anchored)))
    assert "; the feature, about 2.2 km² near" in t


# ------------------------------------------------------------------ 5: feature charts draw only the lighter band
def _bands(it):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.collections import PolyCollection
    from matplotlib.figure import Figure
    import datetime as dt
    t = np.linspace(0, 9, 40)
    dates = [dt.datetime(2016, 1, 5) + dt.timedelta(days=int(365.25 * x)) for x in t]
    it = dict(it, series_vertical_mm=list(it["rate"] * t))
    ax = Figure().subplots()
    cr._series_axes(ax, dates, t, it, 0.0)
    labels = [h.get_label() for h in ax.get_legend_handles_labels()[0]]
    return sum(isinstance(c, PolyCollection) for c in ax.collections), labels


def test_feature_chart_has_only_the_lighter_band_areas_keep_both():
    feat = {"name": "Feature 1", "rate": -20.0, "total_mm": -180.0, "centre_rate": 0.0, "cone_rate": 8.0,
            "noise_mm": 3.0, "feature": True, "decision": False, "relative": True, "detected": None}
    n, labels = _bands(feat)
    assert n == 1, labels
    assert "ordinary ground's average rate (95%)" not in labels
    area = dict(feat, name="Test Field", feature=False, decision=True, centre_rate=0.5, detected=True)
    n, labels = _bands(area)
    assert n == 2 and "how far ordinary ground's average rate reaches (95%)" in labels


# ------------------------------------------------------------------ 4 and 6: the presenter PDF
def _pdf_figs(built, tmp_path, monkeypatch):
    seen = L5._capture_figures(monkeypatch)
    cr.area_pdf(built.cfg, built.rep, built.ctx, built.outlines, tmp_path / "areas.pdf")
    assert len(seen) == len(built.rep["areas"])
    return seen


def _header(fig):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    r = FigureCanvasAgg(fig).get_renderer()
    hdr = next(tx for tx in fig.texts if tx.get_text().startswith("Sentinel-1 satellite radar"))
    return hdr, hdr.get_window_extent(r), fig.bbox


def test_relative_only_whole_area_page_draws_its_series_not_unmeasurable(relative, tmp_path, monkeypatch):
    """Every relative-only report's first PDF page called a ~97-99% measured
    study area 'too little of this area could be measured' and drew no series."""
    full = next(a for a in relative.rep["areas"] if a["kind"] == "full")
    assert full["relative_only"] and full["vertical_mm_yr"] is None and full["series_vertical_mm"]
    fig = _pdf_figs(relative, tmp_path, monkeypatch)[0]           # the whole study area first
    texts = [tx.get_text() for ax in fig.axes for tx in ax.texts] + [tx.get_text() for tx in fig.texts]
    assert not any("too little of this area could be measured" in s for s in texts)
    lines = [ln.get_label() for ax in fig.axes for ln in ax.get_lines()]
    ground = relative.rep.get("control_ground_label") or "farmland"
    assert f"every image (relative to the {ground})" in lines
    assert "average rate" not in lines                            # no rate without an anchor


def test_anchored_whole_area_page_keeps_series_and_rate(anchored, tmp_path, monkeypatch):
    """Guard (passed before the fix too): the anchored whole area keeps both lines."""
    fig =_pdf_figs(anchored, tmp_path, monkeypatch)[0]
    lines = [ln.get_label() for ax in fig.axes for ln in ax.get_lines()]
    assert "every image" in lines and "average rate" in lines


@pytest.mark.parametrize("which", ["relative", "anchored"])
def test_pdf_header_stays_on_the_page(which, request, tmp_path, monkeypatch):
    """The relative-only header ('... relative to the farmland.') ran past the
    page's right edge (about 640 of 612 pt) and was cut off."""
    built = request.getfixturevalue(which)
    for fig in _pdf_figs(built, tmp_path, monkeypatch):
        hdr, box, page = _header(fig)
        assert box.x1 <= page.x1 - 0.03 * page.width, hdr.get_text()
        # no land-cover map on the synthetic project: "the ground" (28 Oct 2026)
        assert hdr.get_text().endswith("anchored to GNSS." if which == "anchored" else "relative to the ground.")
        # the numbers block starts below the two-line header
        first_label = next(tx for tx in fig.texts if tx.get_text() == "Vertical movement per year")
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        r = FigureCanvasAgg(fig).get_renderer()
        assert first_label.get_window_extent(r).y1 < hdr.get_window_extent(r).y0


# ------------------------------------------------------------------ 17: an older-dated average-rate raster is moved aside
def _touch(p: Path):
    p.write_bytes(b"x")
    return p


def test_older_dated_average_rate_raster_is_superseded(tmp_path):
    D = tmp_path / "deliverables"
    D.mkdir()
    old = _touch(D / "vertical_change_at_average_rate_in_20160105_20250101.tif")
    _touch(D / (old.name + ".aux.xml"))
    pre = _touch(D / "vertical_movement_total_in_20160105_20240101.tif")
    keep = _touch(D / "vertical_change_at_average_rate_in_20160105_20260617.tif")
    other = _touch(D / "vertical_movement_in_per_yr.tif")
    cr._supersede_old_total(D, "in", keep=keep.name)
    top = sorted(p.name for p in D.glob("*.tif*"))
    assert top == sorted([keep.name, other.name])
    moved = sorted(p.name for p in (D / "superseded").glob("*/*"))
    assert moved == sorted([old.name, old.name + ".aux.xml", pre.name])


def test_write_rasters_leaves_only_this_records_average_rate_raster(anchored, tmp_path):
    out = tmp_path / "rasters"
    out.mkdir()
    stale = _touch(out / "vertical_change_at_average_rate_in_20160105_20200101.tif")
    paths = cr.write_rasters(anchored.cfg, anchored.rep, anchored.ctx, out_dir=out)
    d0, d1 = anchored.rep["dates"]
    left = sorted(p.name for p in out.glob(f"{cr.AVERAGE_RATE_RASTER}_*.tif"))
    assert left == [f"{cr.AVERAGE_RATE_RASTER}_in_{d0}_{d1}.tif"]
    assert (out / left[0]) in paths and not stale.exists()
    assert any(p.name == stale.name for p in (out / "superseded").glob("*/*"))
    assert re.match(r"^vertical_change_at_average_rate_in_\d{8}_\d{8}\.tif$", left[0])


def test_a_report_without_inside_stations_is_left_as_it_was(relative):
    """Review: marking added an empty inside_checks list to a report whose
    GNSS block had none (no tie, a stale tie), which changes its numbers hash
    and so voids a sign-off on a rebuild with the same numbers."""
    rep = {"los_up_median": 0.78, "gnss": {"anchored": False, "reason": "no GNSS tie on disk", "stations_used": 0}}
    before = json.dumps(rep, sort_keys=True)
    cr.mark_inside_checks(rep)
    assert json.dumps(rep, sort_keys=True) == before
    assert "inside_checks" not in relative.rep["gnss"]
