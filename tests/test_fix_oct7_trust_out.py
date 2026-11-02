"""7 Oct 2026 trust review, group out (deliverables.py, results.py,
panel_history.py, client_kmz.py): decisions E3, E14, E17, E19, E26, E20
item #37, and the consumer side of contracts C1 / C3 / C4 / E28 in these
files. Each test here fails on the code in backups/pre_trust_20261007/
unless its docstring says otherwise.

E3   panel_history.json's 'by years since mining' table is line of sight on
     the first farmland pass: labelled so (the client page builds its own,
     vertical, on the report's farmland - client_report).
E14  first-pass values stay in technical files only if labelled; the point
     table's relative columns are rewritten on the report's farmland
     (deliverables.points_on_report_farmland).
E17  a products run moves older Results exports to deliverables/superseded/.
E19  the point table carries the second-check columns only when the record
     has the layer, and its README says only what is there.
E26  the KMZ carries the page's caveats (vertical assumed from one track,
     the ground a value covers, the page's Limitations).
#37  the KMZ prints the average-rate total by the page's rule.
C1   the KMZ's detection limit is the sinking one (+ rising when > 10%
     apart); panel_history passes the coherence grid and geometry to
     significance.assess (the local / coherence-matched nulls).
E9   the KMZ prints the ranges the report carries for totals and for areas
     that could not be compared with farmland.
C3   panel_history totals take a pace null; a Results window's total bends
     only when the bend beats farmland copies of its footprint (E10).
C4   the KMZ's chance is 'less than 1 in 60', never 'about 0.02'.
"""
import html as htmlmod
import json
import re
import sys
import types
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_fix_L7a as L7a                                       # noqa: E402  (deliverables project)
import test_fix_L9c as L9c                                       # noqa: E402  (panel-history project)


@pytest.fixture(autouse=True)
def _clear_cache():
    from insar_pipeline import deliverables as dl
    dl.clear_cache()
    yield
    dl.clear_cache()


def _plain(s):
    return htmlmod.unescape(re.sub(r"<[^>]+>", " ", s))


# a report record with what the KMZ reads (Site B's resolution / look geometry)
REP = {"anchored": True, "span_years": 10.4, "control_ground_label": "farmland", "dates": ["20160105", "20260617"],
       "n_dates": 270, "title": "Test", "orbit": "ASCENDING", "track": 19, "look_angle_deg": 39.49,
       "los_east_over_up": 0.811, "mine_layers": ["mine areas"], "mine_layer_info": [], "route": "miaplpy",
       "resolution": {"posting_m": 15.0, "look_cell_m": [40.6, 42.2], "support_m": [92.2, 239.2],
                      "text": "Points are 15 m apart; each value combines similar ground within about 90 m "
                              "east-west and 240 m north-south."},
       "gnss": {"background": {"vertical_mm_yr": -1.5, "vertical_uncertainty_mm_yr": 1.5}}}

AREA = {"name": "Project area", "kind": "area", "testable": True, "vertical_mm_yr": 0.0,
        "total_mm": 3.7, "total_at_average_rate_mm": 0.0, "nonlinear": True, "range95_mm_yr": [-6.0, 6.0],
        "detection_limit_mm_yr": 18.3, "detection_limit_rise_mm_yr": 19.7, "detection_limit_basis": "sinking",
        "signal_to_noise": 0.1,
        "reading": "Within normal ground variation", "cells": 1000, "area_km2": 2.0}


# ------------------------------------------------------------------ #37 the totals rule
def test_kmz_prints_the_average_rate_total_by_the_pages_rule():
    """Site B's project area: measured +3.7 mm, average rate 0.0 mm. The
    page prints one total (|difference| < 5 mm); the KMZ printed '0.00 in
    (+0 mm) at the average rate' beside it (> max(2 mm, 5%), or bent)."""
    from insar_pipeline import client_kmz as ck
    from insar_pipeline import client_report as cr
    t = _plain(ck.area_description(AREA, REP))
    assert "at the average rate" not in t and not cr.totals_differ(3.7, 0.0)
    whole = dict(AREA, kind="full", total_mm=-14.4, total_at_average_rate_mm=-17.4)
    assert "at the average rate" not in _plain(ck.area_description(whole, REP))
    big = dict(AREA, total_mm=-100.0, total_at_average_rate_mm=-120.0)
    assert "at the average rate" in _plain(ck.area_description(big, REP)) and cr.totals_differ(-100.0, -120.0)
    f = {"sign": -1, "lat": 38.3, "lon": -87.38, "area_km2": 2.2, "vertical_relative_mm_yr": -25.9,
         "total_relative_mm": -257.0, "total_at_average_rate_mm": -260.0, "nonlinear": True, "text": "x"}
    assert "at the average rate" not in _plain(ck.feature_description(2, f, REP))


# ------------------------------------------------------------------ C1 / E5 the detection limit
def test_kmz_detection_limit_is_the_sinking_one_and_the_rising_one_only_when_it_differs():
    """Site A's Area A2: sinking 18.3, rising 19.7 mm/yr (7.7% apart):
    only the sinking limit. 13% apart (Site B's mine ground): both."""
    from insar_pipeline import client_kmz as ck
    t = _plain(ck.area_description(AREA, REP))
    assert "sinking of 0.72 in/yr (18 mm/yr) or more" in t and "rising of" not in t
    t = _plain(ck.area_description(dict(AREA, detection_limit_mm_yr=6.8, detection_limit_rise_mm_yr=7.7,
                                        detection_limit_basis="sinking"), REP))
    assert "sinking of 0.27 in/yr (7 mm/yr) or more; rising of 0.30 in/yr (8 mm/yr) or more" in t
    # a report from before 7 Oct 2026: one (worse-sign) limit, printed as it was
    old = {k: v for k, v in AREA.items() if k not in ("detection_limit_rise_mm_yr", "detection_limit_basis")}
    assert "sustained movement of 0.72 in/yr" in _plain(ck.area_description(old, REP))


# ------------------------------------------------------------------ E9 ranges the report carries
def test_kmz_totals_and_untested_areas_carry_the_reports_ranges():
    """Every total and every untested figure the client reads carries its
    95% range (E9); the KMZ printed the report's totals bare and '—' for
    the range of an area that could not be compared with farmland."""
    from insar_pipeline import client_kmz as ck
    a = dict(AREA, total_mm=-70.9, total_at_average_rate_mm=-71.0, total_range95_mm=[-221.0, 75.0])
    assert "-2.79 in (-71 mm), 95% -8.70 to +2.95 in" in _plain(ck.area_description(a, REP))
    u = dict(AREA, testable=False, range95_mm_yr=None, untested_range95_mm_yr=[-5.5, 6.2],
             untested_total_range95_mm=[-57.0, 65.0], not_testable_reason="Too little ordinary farmland")
    t = _plain(ck.area_description(u, REP))
    assert "-0.22 to +0.24 in/yr (-5.5 to +6.2 mm/yr), not compared with ordinary farmland" in t
    assert "95% -2.24 to +2.56 in" in t
    f = {"sign": -1, "lat": 38.3, "lon": -87.38, "area_km2": 2.2, "vertical_relative_mm_yr": -25.9,
         "total_relative_mm": -269.0, "total_range95_mm": [-330.0, -208.0], "text": "x"}
    # feature totals to 0.1 in, as on the page (E9 precision, final check of 7 Oct 2026)
    assert "-10.6 in (-269 mm), 95% -13.0 to -8.2 in" in _plain(ck.feature_description(2, f, REP))


# ------------------------------------------------------------------ C4 / E8 the chance in words
def test_kmz_chance_is_a_bound_when_the_feature_beat_every_copy():
    """A feature that beat all 59 copies: 'less than 1 in 60' (the report's
    p_text), never 'about 0.02'; rebuilt from the copies for an older record."""
    from insar_pipeline import client_kmz as ck
    f = {"sign": -1, "lat": 38.3, "lon": -87.38, "area_km2": 2.2, "vertical_relative_mm_yr": -25.9,
         "total_relative_mm": -257.0, "text": "x", "selection_p": 0.0167,
         "chance": {"tested": True, "p": 0.0167, "copies_as_strong": 0, "copies": 59}}
    t = _plain(ck.feature_description(1, f, REP))
    assert "less than 1 in 60" in t and "0.02" not in t
    f["chance"] = dict(f["chance"], copies_as_strong=3, p_text="about 4 in 60")
    assert "about 4 in 60" in _plain(ck.feature_description(1, f, REP))
    f.pop("chance"); f.pop("selection_p")
    assert "Chance of a patch" not in _plain(ck.feature_description(1, f, REP))


# ------------------------------------------------------------------ E26 the caveats travel
def test_kmz_balloons_say_vertical_is_assumed_and_what_ground_a_value_covers():
    from insar_pipeline import client_kmz as ck
    t = _plain(ck.area_description(AREA, REP))
    assert "Vertical assumes all the movement is up or down" in t
    assert "about 39° from vertical from one ascending satellite track" in t
    assert "east-west ground movement of 1 in/yr reads as about 0.8 in/yr" in t
    assert "90 m east-west and 240 m north-south" in t
    pv = ck.point_values(-10.0, -8.5, 6.0, 1.5, True)
    p = _plain(ck.point_description("Project area", 38.3, -87.5, pv, 10.4, 0.8, None, resolution=REP["resolution"]))
    assert "(~45 m)" in p and "within about 90 m east-west and 240 m north-south" in p
    assert "Vertical assumes all the movement is up or down" in p
    f = {"sign": -1, "lat": 38.3, "lon": -87.38, "area_km2": 2.2, "vertical_relative_mm_yr": -37.7,
         "text": "Near mining the ground also moves sideways, which this single satellite view partly reads as "
                 "vertical; the radar itself measured 29 mm/yr along its line of sight (see Limitations). It runs on."}
    ft = _plain(ck.feature_description(1, f, REP))
    assert "(see Limitations)" not in ft
    assert "(from this track, east-west ground movement of 1 in/yr reads as about 0.8 in/yr of vertical" in ft


def test_kmz_document_carries_the_pages_limitations(tmp_path):
    from insar_pipeline import client_kmz as ck
    from insar_pipeline import client_report as cr
    cfg = types.SimpleNamespace(project_dir=tmp_path / "Test", backend="miaplpy")
    d = ck.doc_description(cfg, REP, False, [], False)
    assert "<b>Limitations</b>" in d
    for item in cr.limitations(cfg, REP):
        assert item in d                                       # the page's items, word for word
    assert "90 m east-west and 240 m north-south" in _plain(d)


# ------------------------------------------------------------------ E17 superseded Results exports
def _stub_exports(monkeypatch, dl):
    for name in ("export_velocity", "export_velocity_uncertainty", "export_empirical_uncertainty",
                 "export_cell_noise", "export_signal_to_noise", "export_vertical_velocity",
                 "export_quality", "export_extra_layers", "export_cumulative_displacement",
                 "plot_displacement_series", "plot_selected_points", "export_kmz_points",
                 "export_measurement_points", "export_coverage_curve"):
        monkeypatch.setattr(dl, name, lambda *a, **k: None)
    monkeypatch.setattr(dl, "summary_stats", lambda cfg: {})
    monkeypatch.setattr(dl.provenance, "write", lambda *a, **k: None)


def test_a_products_run_moves_older_results_exports_aside(tmp_path, monkeypatch):
    """Site A's 28 Sep Area A2 export (-0.223 in/yr against the
    rebuilt report's -0.265) stayed in deliverables/results/ and its CSV/JSON
    went into the package's technical folder."""
    from insar_pipeline import deliverables as dl
    cfg, _ = L7a.make_project(tmp_path, gnss_validation=False)
    D = cfg.project_dir / "deliverables"
    old = D / "results" / "area_a2_20160105_20260730"
    old.mkdir(parents=True)
    (old / "summary.json").write_text(json.dumps({"area_summary": {"mean_vertical_velocity_in_yr": -0.2232}}))
    (old / "annual.csv").write_text("year\n2017\n")
    _stub_exports(monkeypatch, dl)
    dl.export_all(cfg)
    assert not any((D / "results").iterdir())
    moved = list((D / "superseded").glob("results_*/area_a2_20160105_20260730"))
    assert len(moved) == 1 and (moved[0] / "summary.json").exists() and (moved[0] / "annual.csv").exists()
    assert dl.supersede_results(D) == []                       # nothing left to move


# ------------------------------------------------------------------ E19 the second-check columns
def _gpkg_columns(path):
    import sqlite3
    con = sqlite3.connect(str(path))
    try:
        return [r[1] for r in con.execute("PRAGMA table_info(points)")]
    finally:
        con.close()


def test_point_table_leaves_out_the_second_check_without_its_layer(tmp_path):
    """Site A: unwrap_consistency / fails_second_gate empty in all 4.5 M
    rows under a README that described them."""
    from insar_pipeline import deliverables as dl
    cfg, _ = L7a.make_project(tmp_path)
    D = cfg.project_dir / "deliverables"
    p = dl.export_measurement_points(cfg, D)
    cols = _gpkg_columns(p)
    assert "unwrap_consistency" not in cols and "fails_second_gate" not in cols
    readme = (D / "points_README.txt").read_text()
    assert "not available on this record" in readme
    assert "quality_unwrapping_consistency.tif" not in readme and "unwrap_consistency [" not in readme


def test_point_table_carries_the_second_check_when_the_layer_is_there(tmp_path):
    """The guard beside the test above (passes on the old code too): a
    record with the layer keeps both columns."""
    from insar_pipeline import deliverables as dl
    cfg, truth = L7a.make_project(tmp_path)
    um = np.where(truth["tc"] > 0.6, 0.9, 0.5).astype(np.float32)
    L7a._w(cfg.mintpy_dir / "geo" / "geo_temporalCoherence_mintpy.h5", {"temporalCoherence": um}, truth["attrs"])
    D = cfg.project_dir / "deliverables"
    p = dl.export_measurement_points(cfg, D)
    cols = _gpkg_columns(p)
    assert "unwrap_consistency" in cols and "fails_second_gate" in cols
    assert "not available on this record" not in (D / "points_README.txt").read_text()


# ------------------------------------------------------------------ E14 the point table's farmland
def test_point_readme_says_its_relative_columns_are_on_the_first_pass(tmp_path):
    from insar_pipeline import deliverables as dl
    cfg, _ = L7a.make_project(tmp_path)
    D = cfg.project_dir / "deliverables"
    dl.export_measurement_points(cfg, D)
    readme = (D / "points_README.txt").read_text()
    assert "Relative columns: mean_velocity_relative" in readme
    assert "Farmland: the FIRST pass" in readme


def _report_ctx(cfg, truth, shift=(0.85, -0.3)):
    """The report's context, light: the products' relative field moved by a
    per-class shift (what the second pass does to every class centre)."""
    from insar_pipeline import deliverables as dl
    from insar_pipeline import significance as sg
    vel, _ = dl._read(dl._product(cfg, "velocity.h5"), "velocity")
    tc = truth["tc"]
    kcls = sg.coherence_class(tc)
    good = dl.accepted_points(cfg, vel)
    far = good & (kcls == kcls[good].min())
    first, _ = sg.class_matched_field(vel * 1000.0, tc, far)
    r_cm = first - np.where(tc > 0.6, shift[1], shift[0])
    return types.SimpleNamespace(r_cm=r_cm.astype(np.float32), far=far, good=good, kcls=kcls,
                                 dates=list(L7a.DATES))


def test_point_table_is_rewritten_on_the_reports_farmland(tmp_path):
    """Site A: the points' class 0 sat +0.852 mm/yr LOS from the report,
    so averaging them over the whole area gave +0.16 mm/yr where the report
    reads -0.83; significant_95 disagreed with rate_signal_to_noise.tif on
    48,594 points."""
    import pyogrio
    import sqlite3
    from insar_pipeline import deliverables as dl
    cfg, truth = L7a.make_project(tmp_path)
    D = cfg.project_dir / "deliverables"
    p = dl.export_measurement_points(cfg, D)
    ctx = _report_ctx(cfg, truth)
    null = np.full(ctx.r_cm.shape, 2.0, np.float32)
    rep = {"dates": [L7a.DATES[0], L7a.DATES[-1]], "control_ground_label": "farmland",
           "farmland_second_pass": {"features_removed": 1, "buffer_m": 500.0,
                                    "class_centre_shift_mm_yr": {"0": 0.85, "1": -0.3}}}
    out = dl.points_on_report_farmland(cfg, rep, ctx, null)
    assert out["applied"] is True and out["points"] > 0
    vel, _ = dl._read(dl._product(cfg, "velocity.h5"), "velocity")
    rows, cols = np.where(dl.accepted_points(cfg, vel))
    df = pyogrio.read_dataframe(p, layer="points", read_geometry=False, fid_as_index=True)
    assert np.allclose(df["mean_velocity_relative"].to_numpy(), ctx.r_cm[rows, cols], atol=1e-4)
    assert np.allclose(df["cell_noise_1sigma"].to_numpy(), 2.0)
    want = (np.abs(ctx.r_cm[rows, cols]) > 1.96 * 2.0).astype(int)
    assert (df["significant_95"].to_numpy() == want).all() and 0 < want.sum() < want.size
    readme = (D / "points_README.txt").read_text()
    assert "Farmland: the client report's (second pass" in readme and "FIRST pass" not in readme
    assert "class 0 +0.850" in readme
    con = sqlite3.connect(str(p))
    md = con.execute("SELECT metadata FROM gpkg_metadata").fetchone()[0]
    con.close()
    assert "RELATIVE_FARMLAND" in md and "Farmland: the client report" in md and "FIRST pass" not in md
    # the table still reads as a GeoPackage with every point
    info = pyogrio.read_info(p, layer="points", force_feature_count=True)
    assert int(info["features"]) == rows.size


def test_a_point_table_from_another_products_run_is_left_alone(tmp_path, monkeypatch):
    import pyogrio
    from insar_pipeline import deliverables as dl
    cfg, truth = L7a.make_project(tmp_path)
    D = cfg.project_dir / "deliverables"
    p = dl.export_measurement_points(cfg, D)
    before = pyogrio.read_dataframe(p, layer="points", read_geometry=False)["mean_velocity_relative"].to_numpy()
    ctx = _report_ctx(cfg, truth)
    real = dl.accepted_points
    monkeypatch.setattr(dl, "accepted_points", lambda cfg, vel: real(cfg, vel) & (np.arange(vel.shape[1]) > 10))
    out = dl.points_on_report_farmland(cfg, {"dates": [L7a.DATES[0], L7a.DATES[-1]]}, ctx,
                                       np.full(ctx.r_cm.shape, 2.0, np.float32))
    assert out["applied"] is False and "another products run" in out["reason"]
    after = pyogrio.read_dataframe(p, layer="points", read_geometry=False)["mean_velocity_relative"].to_numpy()
    assert np.array_equal(before, after, equal_nan=True)


def test_summary_per_cell_shares_are_labelled_first_pass(tmp_path):
    from insar_pipeline import deliverables as dl
    cfg, _ = L7a.make_project(tmp_path)
    vel, _ = dl._read(dl._product(cfg, "velocity.h5"), "velocity")
    s = dl._cell_significance(cfg, np.isfinite(vel))
    assert s["farmland_pass"] == "first" and "FIRST farmland pass" in s["note"]


# ------------------------------------------------------------------ E3 / E14 panel history and sites
def test_panel_history_years_since_table_is_labelled_line_of_sight_first_pass(tmp_path):
    """Site A's client table was this one: line of sight, first pass,
    25-35% small, under a legend saying 'vertical'."""
    from insar_pipeline import panel_history as ph
    cfg = L9c._dated_project(tmp_path)
    res = ph.run(cfg)
    saved = json.loads((cfg.project_dir / "deliverables" / "panel_history.json").read_text())
    for r in (res, saved):
        assert r["by_years_since_mining_frame"] == "line of sight"
        assert r["by_years_since_mining_note"].startswith("TECHNICAL: line of sight")
        assert "FIRST farmland pass" in r["by_years_since_mining_note"] and "FIRST farmland pass" in r["farmland"]
    csv = (cfg.project_dir / "deliverables" / "panel_history.csv").read_text().splitlines()[0]
    assert "FIRST farmland pass" in csv and "line of sight" in csv


def test_site_files_say_which_farmland_they_are_on(tmp_path):
    from insar_pipeline import panel_history as ph
    cfg = L9c._dated_project(tmp_path, focus_sites_lalo=("Client parcel: 38.33,-87.68,38.37,-87.66",),
                             graph_n_points=30)
    ph.run(cfg)
    D = cfg.project_dir / "deliverables"
    sh = json.loads((D / "site_history.json").read_text())
    assert "FIRST farmland pass" in sh["farmland"]
    assert "FIRST farmland pass" in sh["sites"]["Client parcel"]["cell_stats_note"]
    assert "FIRST farmland pass" in (D / "site_stats.csv").read_text().splitlines()[0]
    # on the report's context (after its cut) the same files say so
    ctx = ph._Context(cfg)
    cut = np.zeros(ctx.good.shape, bool)
    cut[5:10, 5:10] = True
    ph.exclude_from_farmland(ctx, [cut], 500.0)
    ph.site_history(cfg, ctx)
    sh = json.loads((D / "site_history.json").read_text())
    assert sh["farmland"].startswith("the client report's farmland")
    assert "client report's farmland" in (D / "site_stats.csv").read_text().splitlines()[0]


# ------------------------------------------------------------------ C1 / C3 on the context
def test_summarise_gives_assess_the_coherence_grid_and_geometry(tmp_path, monkeypatch):
    """Contract C1: the local and coherence-matched nulls need tc and attrs;
    client_report's area, mine and feature figures come through summarise."""
    from insar_pipeline import panel_history as ph
    from insar_pipeline import significance as sg
    cfg = L9c.make_project(tmp_path)
    ctx = ph._Context(cfg)
    seen = {}
    real = sg.assess

    def spy(field, far, region, *a, **k):
        seen.update(k)
        return real(field, far, region, *a, **k)
    monkeypatch.setattr(sg, "assess", spy)
    m = np.zeros(ctx.good.shape, bool)
    m[20:30, 20:30] = True
    m &= ctx.good
    series, far, _ = ctx.series({"r": m})
    out = ctx.summarise(series["r"], m, far)
    assert seen.get("tc") is ctx.tc and seen.get("attrs") is ctx.attrs
    assert {"local_null", "coherence_null"} <= set(out["significance"])


def _bend(t):
    return np.where(t < 1.5, -20.0 * t, -30.0 + 0.0 * (t - 1.5))


def test_context_totals_take_a_pace_null(tmp_path):
    """Contract C3 / E10: with no copy of the footprint (the whole study
    area) the total is rate x span, untested; a bend the copies never make
    is the measured change."""
    from insar_pipeline import panel_history as ph
    from insar_pipeline import significance as sg
    t = np.linspace(0.0, 4.0, 100)
    rng = np.random.default_rng(1)
    y = _bend(t) + rng.normal(0, 1.0, t.size)
    ctx = object.__new__(ph._Context)
    ctx.sg, ctx.t = sg, t
    fit = sg.fit_rate(t, y)
    none = ctx.totals(y, fit, pace_null=[])
    assert none["total_method"] == "rate_x_span" and none["pace_change_tested"] is False
    assert none["nonlinear"] is False and none["total_mm"] == pytest.approx(none["total_at_average_rate_mm"], abs=0.1)
    copies = [rng.normal(0, 1.0, t.size) for _ in range(25)]
    tested = ctx.totals(y, fit, pace_null=copies)
    assert tested["pace_change_tested"] is True and tested["total_method"] != "rate_x_span"
    assert tested["total_mm"] == pytest.approx(-30.0, abs=3.0)


def test_results_window_total_bends_only_when_farmland_copies_do_not():
    """E10 in the Results workbook (client-facing): the uncalibrated F-test
    bent the total on every farmland patch tested; without copies of the
    footprint (or for the whole study area) the total is rate x span."""
    from insar_pipeline import results as rs
    from insar_pipeline import significance as sg
    t = np.linspace(0.0, 4.0, 100)
    rng = np.random.default_rng(2)
    y = _bend(t) + rng.normal(0, 1.0, t.size)
    s = rs._window_summary(sg, t, y)
    assert s["total_method"] == "rate_x_span" and s["nonlinear"] is False and s["pace_change_tested"] is False
    s = rs._window_summary(sg, t, y, [rng.normal(0, 1.0, t.size) for _ in range(25)])
    assert s["pace_change_tested"] is True and s["total_method"] != "rate_x_span" and s["nonlinear"] is True
    # copies that bend as much: the bend is ordinary, rate x span
    s = rs._window_summary(sg, t, y, [_bend(t) * rng.uniform(0.8, 1.6) + rng.normal(0, 1.0, t.size)
                                      for _ in range(25)])
    assert s["pace_change_tested"] is True and s["total_method"] == "rate_x_span"


# ------------------------------------------------------------------ the built record (L9b's project)
@pytest.fixture(scope="module")
def built(tmp_path_factory):
    import test_fix_L9b as L9B
    return L9B.built.__wrapped__(tmp_path_factory)


def test_built_kmz_carries_the_limitations_and_the_point_ground(built):
    kml = built.kml
    assert "<b>Limitations</b>" in kml and "Vertical assumes all the movement is up or down" in kml
    sup = (built.rep.get("resolution") or {}).get("support_m")
    if sup:
        assert f"within about {round(sup[0], -1):.0f} m east-west" in kml


def test_results_workbook_stores_the_rising_limit_and_the_pace_test(built):
    """C1 / E5: the workbook stores both limits; E10: it says whether the
    bend was tested and on how many copies."""
    s = built.res["Test Field"].area_summary
    assert "reliably_detects_rising_in_yr" in s
    assert "pace_change_tested" in s and "pace_null_copies" in s
    assert any("reliably_detects_rising" in n for n in built.res["Test Field"].notes)


def test_results_rebuild_the_report_areas_the_report_cut_without_its_recorded_cells(built):
    """E28 (contract C2): the report cuts the report areas other than the
    whole study area from its comparison ground with the features it finds.
    A report that lost its recorded cells (cut_cells) names them
    (report_areas_removed): the Results farmland rebuilt from the feature
    outlines alone kept the area in its own comparison ground (L9b's
    project: 13,510 farmland cells against the report's 12,206)."""
    import copy
    from insar_pipeline import results as rs
    from insar_pipeline.panel_history import _Context
    rep = copy.deepcopy(built.rep)
    sp = rep["farmland_second_pass"]
    if not sp.get("report_areas_removed"):
        pytest.skip("this build cut no report area from its farmland")
    sp.pop("cut_cells", None)
    res = rs.analyse(built.cfg, "Test Field", ctx=_Context(built.cfg), rep=rep)
    got = res.area_summary["farmland_second_pass"]
    want = built.rep["farmland_second_pass"]["far_cells"][1]
    assert got["report_areas_cut"] == sp["report_areas_removed"]
    assert abs(got["far_cells"] - want) <= 0.02 * want


# ------------------------------------------------------------------ reviewer, 7 Oct 2026
def test_results_notes_count_the_reports_cut_and_name_the_report_areas(built):
    """The recorded cut (farmland_second_pass.cut_cells) is ONE mask holding
    every feature the report cut, so the Notes sheet said '(1 feature(s))'
    for Site B's 2 (features_removed). E28: the report areas the report
    left out of its comparison ground are named as well, by the name the
    client reads."""
    import copy
    from insar_pipeline import results as rs
    from insar_pipeline.panel_history import _Context
    rep = copy.deepcopy(built.rep)
    sp = rep.get("farmland_second_pass") or {}
    if not sp.get("cut_cells"):
        pytest.skip("this build recorded no cut")
    sp.update({"features_removed": 2, "report_areas_removed": ["Test Field"]})
    res = rs.analyse(built.cfg, "Test Field", ctx=_Context(built.cfg), rep=rep)
    farm = res.area_summary["farmland_second_pass"]
    assert farm["method"] == "recorded cut" and farm["features"] == 2
    note = next(n for n in res.notes if n.startswith("Farmland is the report's"))
    assert "(2 feature(s))" in note
    assert "and the report areas Test Field (an area is never compared with itself) are taken out" in note


def test_farmland_label_is_first_pass_when_the_report_farmland_was_not_applied():
    """results.report_farmland marks a context 'report' whenever it saw a
    report; one it could not apply leaves the context on the first pass,
    and the technical files must say so."""
    from insar_pipeline import panel_history as ph
    ctx = types.SimpleNamespace(farmland_pass="report", report_farmland={"applied": False})
    assert "FIRST farmland pass" in ph.farmland_label(ctx)
    ctx = types.SimpleNamespace(farmland_pass="report", report_farmland={"applied": True})
    assert ph.farmland_label(ctx).startswith("the client report's farmland")
    assert ph.farmland_label(types.SimpleNamespace(farmland_pass="second")).startswith("the client report's")


def test_results_totals_carry_the_reports_95pct_range(built, tmp_path, monkeypatch):
    """E9 in the Results workbook and its trend graph (both go to client/):
    the area total printed bare ('-2.79 in measured'); it now carries the
    report's own range (client_report.total_range through area_entry), the
    Summary sheet lists it, and an untested area states what can be."""
    from insar_pipeline import results as rs
    import matplotlib.axes
    checked = 0
    for a in built.rep["areas"]:
        res = built.res[a["name"]]
        s = res.area_summary
        assert "measured_change_95pct_range_in" in s and "untested_vertical_95pct_range_in_yr" in s
        want = a.get("total_range95_mm") or a.get("untested_total_range95_mm")
        got = s["measured_change_95pct_range_in"]
        if want:
            assert got is not None and got[0] * 25.4 == pytest.approx(want[0], abs=0.6)
            assert got[1] * 25.4 == pytest.approx(want[1], abs=0.6)
            checked += 1
        if s["measured_change_95pct_range_in"] is not None and s.get("mean_vertical_cumulative_in") is not None:
            titles = []
            monkeypatch.setattr(matplotlib.axes.Axes, "set_title",
                                lambda self, t, *k, **kw: titles.append(t))
            rs._trend_chart(res, tmp_path / f"t{checked}.png")
            monkeypatch.undo()
            lo, hi = s["measured_change_95pct_range_in"]
            assert any(f"(95% {lo:+.2f} to {hi:+.2f} in)" in t for t in titles)
    assert checked


def test_a_bent_results_total_takes_its_range_from_the_copies_own_bent_totals(built, monkeypatch):
    """E9 with E10: when a window's total bends (its pace change beat the
    farmland copies), its range comes from those copies' own bent totals,
    as the report's does (client_report.total_range) - not from the rate's
    range x years."""
    from insar_pipeline import client_report as cr
    from insar_pipeline import results as rs
    from insar_pipeline.panel_history import _Context
    real_ws, real_entry, seen = rs._window_summary, cr.area_entry, {}

    def bent(sg_, t, rel, pace_null=()):
        s = real_ws(sg_, t, rel, pace_null)
        if s is not None and len(pace_null):
            s.update(total_method="piecewise_linear_yearly", nonlinear=True)      # as if the bend beat them
        return s

    def spy(*a, **k):
        seen["summ"] = a[4]
        out = real_entry(*a, **k)
        seen["entry"] = out
        return out
    monkeypatch.setattr(rs, "_window_summary", bent)
    monkeypatch.setattr(cr, "area_entry", spy)
    res = rs.analyse(built.cfg, "Test Field", ctx=_Context(built.cfg), rep=built.rep)
    n = res.area_summary["pace_null_copies"]
    assert n and n >= 20
    assert len(seen["summ"]["copies_piecewise_vertical_mm"]) == n
    assert "farmland copies of this footprint" in (seen["entry"].get("total_range_basis") or "")
    assert res.area_summary["measured_change_95pct_range_in"] is not None
