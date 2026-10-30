"""7 Oct 2026, last pass after the final verification, group outputs
(deliverables.py, results.py, panel_history.py, figures.py, package.py).
Each test here fails on the code in backups/pre_final_20261007/.

E25   the point README and the Results workbook say 'about N radar
      resolution cells', not 'independent measurements', with the study
      area's count beside the whole table's (ledger #44); summary.json
      carries the count under that name.
E18   a full-record Results window uses the report's per-cell null (the
      rasters', the KMZ's); the point README and signal_to_noise.json say
      which farmland that null was measured on.
E9    the workbook shows figures to the page's precision (in/yr and in to 2
      decimals); the trend graph's title too.
E14   the project-area series (PNG and CSV) and the site figures say which
      farmland pass they are on; the rate map's class means are labelled.
#1    dem_difference_m.tif is described as MintPy's DEM-error estimate that
      takes up part of the movement, in its GeoTIFF and the package README.
Package README: what the page says of the technical record, quoted; the
      reissue note (report_revision).
Figures: interferogram figures of another kind of pair than the record's are
      moved aside, not presented as current.
Years since mining: no -0.0 stored.
"""
import copy
import json
import re
import sys
import types
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_fix_L7a as L7a                                       # noqa: E402  (deliverables project)
import test_fix_L7b as L7b                                       # noqa: E402  (package project)
import test_fix_L9c as L9c                                       # noqa: E402  (figures / panel-history project)


@pytest.fixture(autouse=True)
def _clear_cache():
    from insar_pipeline import deliverables as dl
    dl.clear_cache()
    yield
    dl.clear_cache()


# ------------------------------------------------------------------ E25 the point README and summary.json
def test_point_readme_counts_radar_resolution_cells_over_the_study_area(tmp_path):
    """points_README.txt said 'Independent measurements: 582,856 radar
    pixels behind 4,485,046 cells' beside a page saying they are not
    independent and giving 413,076 over the study area."""
    import sqlite3
    from insar_pipeline import deliverables as dl
    cfg, _ = L7a.make_project(tmp_path)
    D = cfg.project_dir / "deliverables"
    p = dl.export_measurement_points(cfg, D)
    readme = (D / "points_README.txt").read_text()
    assert "Independent measurements" not in readme
    con = sqlite3.connect(str(p))
    n, n_ids, n_in = con.execute("SELECT count(*), count(DISTINCT radar_pixel_id), sum(in_aoi) FROM points").fetchone()
    n_ids_in = con.execute("SELECT count(DISTINCT radar_pixel_id) FROM points WHERE in_aoi = 1").fetchone()[0]
    con.close()
    assert 0 < n_ids_in < n_ids and 0 < n_in < n
    assert f"Radar resolution cells: about {n_ids:,} lie behind these {n:,} map cells" in readme
    assert f"within the study area (in_aoi = 1) about {n_ids_in:,} behind {n_in:,}" in readme
    assert "they are not independent measurements" in readme


def test_summary_names_the_radar_resolution_cells_and_the_study_area_count(tmp_path):
    from insar_pipeline import deliverables as dl
    cfg, _ = L7a.make_project(tmp_path)
    s = dl.summary_stats(cfg)
    assert s["radar_resolution_cells"] == s["independent_radar_measurements"] > 0
    assert 0 < s["radar_resolution_cells_in_study_area"] < s["radar_resolution_cells"]
    assert "not independent" in s["radar_resolution_cells_note"]
    assert "radar_resolution_cells" in s["total_measured_points_note"]


# ------------------------------------------------------------------ #1 dem_difference_m.tif
def test_dem_difference_is_described_as_a_dem_error_estimate_that_follows_the_movement(tmp_path):
    """Its GeoTIFF said 'reshaped ground (spoil, pits, reclamation since the
    DEM was made) shows here'; MintPy's estimate follows the rate (E1)."""
    import rasterio
    from insar_pipeline import deliverables as dl
    from insar_pipeline import package
    cfg, truth = L7a.make_project(tmp_path)
    L7a._w(cfg.mintpy_dir / "geo" / "geo_demErr.h5", {"dem": np.full((L7a.NY, L7a.NX), 2.0, np.float32)},
           truth["attrs"])
    D = cfg.project_dir / "deliverables"
    dl.export_extra_layers(cfg, D)
    with rasterio.open(D / "dem_difference_m.tif") as r:
        desc, tags = r.descriptions[0] or "", r.tags()
    text = desc + " " + " ".join(tags.values())
    assert "reshaped ground (spoil" not in text and "minus the elevation model" not in text
    assert "MintPy's estimate of the elevation-model error" in desc
    assert "not a map of reshaped ground on its own" in desc and "INCLUDES_MOVEMENT" in tags
    meaning = next(m for pat, m in package.TECH_FILES if pat.match("dem_difference_m.tif"))
    assert "ground during the record minus" not in meaning and "not a map of reshaped ground" in meaning


# ------------------------------------------------------------------ E18 the per-cell null, labelled
def _report_ctx(cfg, truth, shift=(0.85, -0.3)):
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


def _rewrite(tmp_path, scale=1.0):
    from insar_pipeline import deliverables as dl
    cfg, truth = L7a.make_project(tmp_path)
    D = cfg.project_dir / "deliverables"
    dl.export_measurement_points(cfg, D)
    model = json.loads((D / "uncertainty_model.json").read_text())
    null = {k: float(v) * scale for k, v in model["cell_null_by_class"].items()}
    # as client_report.write_signal_to_noise leaves it for a second-pass report
    (D / "signal_to_noise.json").write_text(json.dumps({"points_table_note": "first pass", "median": 1.0}))
    rep = {"dates": [L7a.DATES[0], L7a.DATES[-1]], "control_ground_label": "farmland", "cell_null_by_class": null,
           "farmland_second_pass": {"features_removed": 1, "buffer_m": 500.0,
                                    "class_centre_shift_mm_yr": {"0": 0.85, "1": -0.3}}}
    ctx = _report_ctx(cfg, truth)
    out = dl.points_on_report_farmland(cfg, rep, ctx)
    assert out["applied"] is True
    return D, out


def test_point_table_and_signal_to_noise_say_where_the_cell_null_was_measured(tmp_path):
    """signal_to_noise.json said cell_noise_1sigma was 'rewritten on this
    farmland' (the report's second pass), but the null is uncertainty_model.
    json's, measured before the second pass (Site A 7.592 against 7.091
    mm/yr for class 0)."""
    from insar_pipeline import deliverables as dl
    D, out = _rewrite(tmp_path)
    assert out["cell_noise_basis"] == dl.NULL_FROM_PRODUCTS
    readme = (D / "points_README.txt").read_text()
    assert "cell_noise_1sigma: measured at the products stage (uncertainty_model.json)" in readme
    assert "before the client report takes the movement it finds out of it" in readme
    st = json.loads((D / "signal_to_noise.json").read_text())
    assert "cell_noise_1sigma and significant_95 were rewritten on this farmland" not in st["points_table_note"]
    assert st["cell_null_basis"] == dl.NULL_FROM_PRODUCTS


def test_a_report_with_its_own_cell_null_is_said_to_be_the_reports(tmp_path):
    from insar_pipeline import deliverables as dl
    D, out = _rewrite(tmp_path, scale=1.3)
    assert out["cell_noise_basis"] == dl.NULL_FROM_REPORT
    assert "cell_noise_1sigma: the client report's own per-cell null" in (D / "points_README.txt").read_text()
    assert dl.cell_null_basis({}, {"cell_null_by_class": {"0": 5.0}}).startswith("the cells' rate repeatability")


# ------------------------------------------------------------------ years since mining: no -0.0
def test_years_since_mining_rows_store_no_negative_zero():
    from insar_pipeline import panel_history as ph
    panels = {f"a{i}": {"annual_change_mm": [{"year": 2020, "years_since_mining": 1, "change_mm": v}]}
              for i, v in enumerate((-0.04, -0.02, -0.01))}
    rows = ph._by_years_since(panels)
    assert "-0.0" not in json.dumps(rows)
    assert rows[0]["median_change_mm_per_yr"] == 0.0


# ------------------------------------------------------------------ E14 the project-area series and site figures
def test_project_area_series_and_site_figures_say_which_farmland_pass(tmp_path, monkeypatch):
    """los_cumulative_displacement_project_area.png / .csv carried the first
    pass with no label, while site_history.json beside them said so."""
    import matplotlib.axes
    import matplotlib.figure
    from insar_pipeline import panel_history as ph
    cfg = L9c._dated_project(tmp_path, focus_sites_lalo=("Client parcel: 38.33,-87.68,38.37,-87.66",),
                             graph_n_points=30)
    sup, titles = [], []
    real_sup, real_title = matplotlib.figure.Figure.suptitle, matplotlib.axes.Axes.set_title
    monkeypatch.setattr(matplotlib.figure.Figure, "suptitle",
                        lambda self, t, *a, **k: (sup.append(t), real_sup(self, t, *a, **k))[1])
    monkeypatch.setattr(matplotlib.axes.Axes, "set_title",
                        lambda self, t, *a, **k: (titles.append(t), real_title(self, t, *a, **k))[1])
    ph.run(cfg)
    D = cfg.project_dir / "deliverables"
    head = (D / "los_cumulative_displacement_project_area.csv").read_text().splitlines()[0]
    assert head.startswith("# TECHNICAL") and "FIRST farmland pass" in head
    assert any("TECHNICAL: relative figures on the FIRST farmland pass" in t for t in sup)
    assert any(t.startswith("Relative, against the surrounding farmland") and "FIRST farmland pass" in t
               for t in titles)
    assert any(t.startswith("Client parcel:") and "FIRST farmland pass" in t for t in titles)
    man = {m["file"]: m for m in json.loads((D / "figures" / "figures.json").read_text())}
    assert "Relative lines are on the FIRST farmland pass" in man["site_history.png"]["caption"]


# ------------------------------------------------------------------ figures: the pair figures and the rate map
def _figures_project(tmp_path):
    cfg = L9c.make_project(tmp_path, deramp_exclude_geojson=L9c._write(tmp_path / "mine.geojson",
                                                                        L9c._fc((L9c.MINE, {}))),
                           stack_reference_date="20180105")
    L9c._stack(cfg, ["20180105_20180129", "20180609_20180727", "20180129_20180222"])
    return cfg


def test_pair_figures_record_their_kind_and_explain_the_marks(tmp_path):
    from insar_pipeline import figures
    cfg = _figures_project(tmp_path)
    figures.make_figures(cfg)
    man = {m["file"]: m for m in json.loads((cfg.project_dir / "deliverables" / "figures" / "figures.json").read_text())}
    for name in figures.PAIR_FIGURES:
        assert man[name]["kind"] == figures.PAIR_KIND_LINKED and re.match(r"\d{4}-\d\d-\d\d$", man[name]["drawn"])
        assert "The star is the reference pixel." in man[name]["caption"]
    # the rate map's class means are the QA's: the first farmland pass, said so
    assert "FIRST farmland pass" in man["velocity_relative.png"]["caption"]


def test_interferogram_figures_of_another_kind_are_moved_aside_not_kept(tmp_path):
    """Site A's phase-linked record kept two raw ISCE interferograms of
    28 Sep under 'Pairs like this are what the record is built from' once
    the stack's interferograms were deleted."""
    from insar_pipeline import figures
    cfg = _figures_project(tmp_path)
    figures.make_figures(cfg)
    F = cfg.project_dir / "deliverables" / "figures"
    man = json.loads((F / "figures.json").read_text())
    for m in man:                                   # as a figures.json from before 7 Oct 2026 had them
        if m["file"] == "winter_pair.png":
            m.pop("kind"); m.pop("drawn")
            m["caption"] = "Left: the wrapped phase, the raw measurement. Pairs like this are what the record is built from."
    (F / "figures.json").write_text(json.dumps(man))
    for p in (cfg.stack_dir / "merged" / "interferograms").glob("*/filt_fine.int"):
        p.unlink()                                  # the stack cleaned up to make room
    figures.make_figures(cfg)
    man = {m["file"]: m for m in json.loads((F / "figures.json").read_text())}
    assert "winter_pair.png" not in man and not (F / "winter_pair.png").exists()
    moved = list((cfg.project_dir / "deliverables" / "superseded").glob("figures_*/winter_pair.png"))
    assert len(moved) == 1
    # the phase-linked one stays, saying when it was drawn and that its pairs are gone
    assert (F / "summer_pair.png").exists()
    assert man["summer_pair.png"]["caption"].startswith("Drawn on ")
    assert "since been deleted" in man["summer_pair.png"]["caption"]


# ------------------------------------------------------------------ package README
def test_package_readme_says_what_the_page_says_of_the_technical_record(tmp_path, monkeypatch):
    """The page offered the technical record 'on request' while the README
    said technical/ was 'do NOT send to the client'. The README now quotes
    the page's own sentence (client_report.TECHNICAL_RECORD_NOTE) and says
    what the preparer does with the three files."""
    from insar_pipeline import client_report as crp
    from insar_pipeline import package
    cfg = L7b._cfg(tmp_path)
    L7b._project(tmp_path, L7b._rep(), L7b._qa())
    out = package.build(cfg, force_draft=True)
    readme = (out / "README.txt").read_text()
    flat = " ".join(x.strip() for x in readme.splitlines())
    assert "technical/   for specialists only - do NOT send to the client" not in readme
    assert f'The report says: "{crp.TECHNICAL_RECORD_NOTE}"' in flat
    assert "quality_assessment.json (quality assessment), gnss_tie.json (GNSS tie) and provenance.json (provenance)" in flat
    if "on request" not in crp.TECHNICAL_RECORD_NOTE:
        assert "they stay with the project and are not sent" in flat and "on request" not in flat
    # a page that offers the record on request: the README says what to send
    monkeypatch.setattr(crp, "TECHNICAL_RECORD_NOTE", "Technical record (quality assessment, GNSS tie, provenance) "
                        "available on request.")
    lines = " ".join(x.strip() for x in package._record_note_lines(
        [(None, "technical/quality_assessment.json", "technical", ""), (None, "technical/gnss_tie.json", "technical", "")]))
    assert "When the client asks, send quality_assessment.json" in lines and "never the raw rasters" in lines


def test_package_readme_carries_the_reissue_note(tmp_path):
    from insar_pipeline import package
    cfg = L7b._cfg(tmp_path)
    # as the config holds it ('YYYY-MM-DD'); the README prints the page's
    # heading word for word, the date as the page gives it (review, 7 Oct
    # 2026: it printed 'replacing the report issued 2026-09-28')
    rep = L7b._rep(report_revision={"number": 1, "supersedes": "2026-09-28",
                                    "changes": ["Detection limits are now at 80% power.", "Readings unchanged."]})
    L7b._project(tmp_path, rep, L7b._qa())
    readme = (package.build(cfg, force_draft=True) / "README.txt").read_text()
    assert "Revision 1, replacing the report issued 28 Sep 2026. What changed:" in readme
    assert "2026-09-28" not in readme
    assert "  - Detection limits are now at 80% power." in readme
    assert package._revision_lines(cfg, L7b._rep()) == []
    # the report's record as client_report writes it (config.report_revision's note)
    from insar_pipeline.config import report_revision
    rec = report_revision({"number": 2, "supersedes": "2026-10-07", "changes": ["Feature 1 re-measured."]})
    assert package._revision_lines(cfg, {"report_revision": rec})[0] == (
        "Revision 2, replacing the report issued 7 Oct 2026. What changed:")


# ------------------------------------------------------------------ Results (L9b's built record)
@pytest.fixture(scope="module")
def built(tmp_path_factory):
    import test_fix_L9b as L9B
    return L9B.built.__wrapped__(tmp_path_factory)


def test_full_record_results_use_the_reports_cell_null(built):
    """A workbook exported with the final code measured the null again on
    the report's second pass: Site A's Area A2 read 1 sigma 0.358
    against the raster's 0.383, Area A1 6,847 cells standing out
    against 4,604 in rate_signal_to_noise.tif."""
    from insar_pipeline import results as rs
    from insar_pipeline.panel_history import _Context
    rep = copy.deepcopy(built.rep)
    assert rep.get("cell_null_by_class")
    rep["cell_null_by_class"] = {k: 1.5 * float(v) for k, v in rep["cell_null_by_class"].items()}
    base = rs.analyse(built.cfg, "Test Field", ctx=_Context(built.cfg), rep=built.rep)
    wide = rs.analyse(built.cfg, "Test Field", ctx=_Context(built.cfg), rep=rep)
    a, b = (np.asarray(r.points["signal_to_threshold"], float) for r in (base, wide))
    ok = np.isfinite(a) & np.isfinite(b) & (a > 0)
    assert ok.sum() > 10 and np.allclose(b[ok] / a[ok], 1 / 1.5, rtol=1e-6)
    assert any("use the report's per-cell null" in n for n in wide.notes)
    # a window shorter than the record measures its own
    d = built.rep["dates_list"]
    part = rs.analyse(built.cfg, "Test Field", start=f"{d[0][:4]}-{d[0][4:6]}-{d[0][6:]}",
                      end=f"{d[len(d) // 2][:4]}-{d[len(d) // 2][4:6]}-{d[len(d) // 2][6:]}",
                      ctx=_Context(built.cfg), rep=rep)
    assert any("this window's own farmland cell spread" in n for n in part.notes)


def test_results_say_radar_resolution_cells(built):
    s = built.res["Test Field"].area_summary
    assert s["radar_resolution_cells"] == s["independent_measurements"] > 0
    note = next(n for n in built.res["Test Field"].notes if "radar resolution cells" in n)
    assert "these are not independent radar measurements" in note
    assert "from about" in note and "independent radar measurements (" not in note


def _sheet(wb, name):
    rows = list(wb[name].iter_rows(values_only=True))
    return rows[0], rows[1:]


def _decimals_of(x):
    s = repr(float(x))
    return 0 if "e" in s else len(s.split(".")[1].rstrip("0"))


def test_workbook_figures_are_at_the_pages_precision(built, tmp_path):
    """The workbook stored -0.2515748031496063 in/yr (General format, about
    ten digits on screen) beside ranges of +/- 0.5 in/yr; the page prints
    -0.25."""
    import openpyxl
    from insar_pipeline import results as rs
    res = built.res["Test Field"]
    out = rs.export(built.cfg, res, out_dir=tmp_path)
    with open(next(out.glob("*.xlsx")), "rb") as fh:
        wb = openpyxl.load_workbook(fh)
        summ = {r[0].value: r[1] for r in wb["Summary"].iter_rows(min_row=2)}
        assert "independent_measurements" not in summ and summ["radar_resolution_cells"].value > 0
        assert summ["mean_vertical_velocity_in_yr"].number_format == "0.00"
        v = summ["mean_vertical_velocity_in_yr"].value
        assert v == pytest.approx(res.area_summary["mean_vertical_velocity_in_yr"])       # the full value kept
        rng = summ["vertical_95pct_range_in_yr"].value
        assert re.fullmatch(r"[+-]?\d+\.\d\d to [+-]?\d+\.\d\d", rng), rng
        head, rows = _sheet(wb, "Annual")
        for c, name in enumerate(head):
            if name and str(name).endswith("_in_yr"):
                assert all(_decimals_of(r[c]) <= 2 for r in rows if isinstance(r[c], float))
        head, rows = _sheet(wb, "Points")
        col = head.index("vertical_velocity_in_yr")
        assert rows and all(_decimals_of(r[col]) <= 2 for r in rows if isinstance(r[col], float))
        assert any(_decimals_of(r[head.index("lat")]) > 2 for r in rows)          # coordinates untouched
        head, rows = _sheet(wb, "Statistics")
        for r in rows:
            assert all(_decimals_of(r[head.index(k)]) <= 2 for k in ("mean", "min", "max", "std")
                       if isinstance(r[head.index(k)], float))
    # the CSVs keep every digit
    csv_line = (out / "points.csv").read_text().splitlines()[1]
    assert re.search(r"\.\d{5}", csv_line)


def test_trend_graph_title_is_at_the_pages_precision(built, tmp_path, monkeypatch):
    import matplotlib.axes
    from insar_pipeline import results as rs
    titles = []
    monkeypatch.setattr(matplotlib.axes.Axes, "set_title", lambda self, t, *a, **k: titles.append(t))
    rs._trend_chart(built.res["Test Field"], tmp_path / "t.png")
    head = titles[0]
    assert not re.search(r"\d\.\d{3} in", head), head
    assert re.search(r"Test Field: [+-]?\d+\.\d\d in/yr \([+-]?\d+\.\d mm/yr\)", head), head
