"""Outcome check of the 5 Oct 2026 review fixes (6 Oct 2026): the gaps and
regressions an independent check found after the lanes finished (183 of 218
findings resolved on first check). Each test here failed on the code before
these fixes unless its docstring says otherwise.
"""
import json
import types
from pathlib import Path

import numpy as np
import pytest

DATES = ["20160105", "20250320"]


# ------------------------------------------------------------------ 1. a pre-floor GNSS tie is stale (stats-10)
def _tie_project(tmp_path: Path, tie: dict):
    d = tmp_path / "p" / "deliverables"
    d.mkdir(parents=True, exist_ok=True)
    (d / "gnss_tie.json").write_text(json.dumps(tie))
    return types.SimpleNamespace(project_dir=tmp_path / "p", mintpy_dir=tmp_path / "p" / "mintpy")


def test_pre_floor_tie_is_stale(tmp_path):
    """Site A's 28 Sep tie (+/- 0.73, no regional floor) made the whole
    region read 'sinking'; the floored tie gives +/- 1.01 and a range that
    crosses zero. A tie without uncertainty_basis / uncertainty_floor_mm_yr
    is not used, and says why in plain words."""
    from insar_pipeline import gnss_tie as gt
    old = {"record_window": DATES, "stations": [], "inside_checks": [], "los_up_median": 0.78,
           "background": {"vertical_mm_yr": -1.09, "vertical_uncertainty_mm_yr": 0.73, "los_mm_yr": -0.85}}
    cfg = _tie_project(tmp_path, old)
    tie, why = gt.check(cfg, DATES)
    assert tie is None
    assert "re-run the GNSS tie" in why and "too narrow" in why
    assert gt.load_valid(cfg, DATES) is None
    # the report's reader turns it into an unanchored record with that reason
    from insar_pipeline import client_report as cr
    t2, why2 = cr._gnss(cfg, DATES)
    assert t2 == {} and "re-run the GNSS tie" in why2
    # a current tie is used
    new = dict(old, uncertainty_floor_mm_yr=1.01,
               background=dict(old["background"], vertical_uncertainty_mm_yr=1.01, uncertainty_basis="regional floor"))
    cfg = _tie_project(tmp_path, new)
    assert gt.check(cfg, DATES)[0]["background"]["vertical_uncertainty_mm_yr"] == 1.01


def test_release_lists_a_pre_floor_tie_as_stale(tmp_path):
    from insar_pipeline import release
    old = {"record_window": DATES, "stations": [],
           "background": {"vertical_mm_yr": -1.09, "vertical_uncertainty_mm_yr": 0.73}}
    cfg = _tie_project(tmp_path, old)
    rep = {"dates": DATES, "gnss": {"anchored": True}}
    st = release._stale(cfg, rep)
    assert any("re-run the GNSS tie" in s for s in st), st
    # an unanchored record (no background) is not stale for want of the fields
    cfg = _tie_project(tmp_path, {"record_window": DATES, "background": None, "reason": "no station"})
    assert not any("GNSS tie" in s for s in release._stale(cfg, {"dates": DATES, "gnss": {"anchored": False}}))


# ------------------------------------------------------------------ a synthetic project built by the report (items 2-9)
import test_fix_L9b as L9b                                     # noqa: E402  (the report's project builder)

SITE_B_GEO = {"available": True, "n_water_bodies": 46, "step_m": 2.5, "search_m": 75.0, "dE_m": 5.01, "dN_m": 0.0,
                 "offset_m": 5.01, "score": 0.347,
                 "halves": {"all": {"offset_m": 5.01}, "west": {"dE_m": 12.52, "dN_m": -17.53, "offset_m": 21.54},
                            "east": {"dE_m": 5.01, "dN_m": 2.5, "offset_m": 5.6}},
                 "west_east_disagreement_m": 21.39, "flag": False,
                 "text": "Map positions agree with mapped water edges to within about 5 m (46 water bodies)."}


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    """L9b's project (farmland, an unmapped settling patch, a field and a pit)
    with Site B's 6 Oct geolocation record in its QA and a raster under
    the pre-6 Oct total name, built by client_report.build with no firm or
    preparer set."""
    from insar_pipeline import client_report as cr, settings
    root = tmp_path_factory.mktemp("final")
    cfg = L9b.make_project(root)
    P = cfg.project_dir
    D = P / "deliverables"
    qa = json.loads((P / "quality_assessment.json").read_text())
    qa["geolocation"] = dict(SITE_B_GEO)
    (P / "quality_assessment.json").write_text(json.dumps(qa))
    (D / "vertical_movement_total_in_20160105_20250306.tif").write_bytes(b"old name")
    mp = pytest.MonkeyPatch()
    mp.setattr(settings, "identity", lambda: {"firm": "", "preparer": "", "contact": ""})
    try:
        html = cr.build(cfg).read_text()
    finally:
        mp.undo()
    rep = json.loads((D / "client_report.json").read_text())
    return types.SimpleNamespace(cfg=cfg, D=D, rep=rep, html=html)


def _figs(built):
    p = built.D / "figures" / "blank.png"
    if not p.exists():
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(1, 1)).savefig(p)
        plt.close("all")
    return {"map": p, "series": p, "summary": p}


# ------------------------------------------------------------------ 2. never "-0.00"
def test_signed_zero_is_never_printed():
    from insar_pipeline import client_kmz as ck, client_report as cr, report_insights as ri, results as rs
    assert ri.signed(-0.004) == "0.00" and ri.signed(0.006) == "+0.01" and ri.signed(-1.234, 1) == "-1.2"
    assert ri.no_negative_zero("range -0.15 to -0.00 in/yr (-0 mm; +0.00 in)") == "range -0.15 to 0.00 in/yr (0 mm; 0.00 in)"
    # dates, coordinates, small numbers and base64 are left alone
    for keep in ("2016-01-05", "-87.38430", "-0.05", "-0.001", "ab/+0Q==", "x+0.5"):
        assert ri.no_negative_zero(keep) == keep
    assert ri.tidy_text({"a": ["to -0.00 in/yr", 1.0, -0.0], "b": ("-0.0",)}) == {"a": ["to 0.00 in/yr", 1.0, -0.0],
                                                                              "b": ("0.0",)}
    assert cr._s(-0.001) == "0.00" and ck._in(-0.1) == "0.00" and rs._f(-0.0001, "+.3f") == "0.000"
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.set_title("Area: -0.00 in/yr"); ax.plot([0, 1], [0, 1], label="range -0.12 to -0.00")
    ax.legend()
    ri.tidy_figure(fig)
    assert ax.get_title() == "Area: 0.00 in/yr" and ax.get_legend().get_texts()[0].get_text() == "range -0.12 to 0.00"
    plt.close(fig)


def test_report_page_and_kmz_text_pass_through_the_net(tmp_path, monkeypatch):
    """The page is written through no_negative_zero, whatever an f-string
    upstream printed (Site A's whole area read '-0.15 to -0.00 in/yr')."""
    from insar_pipeline import client_report as cr
    cfg = types.SimpleNamespace(project_dir=tmp_path)
    monkeypatch.setattr(cr, "compute", lambda cfg: ({"areas": []}, None, {}, {}))
    monkeypatch.setattr(cr, "figures", lambda *a, **k: {})
    monkeypatch.setattr(cr, "area_pdf", lambda *a, **k: None)
    monkeypatch.setattr(cr, "page", lambda cfg, rep, figs: "<p>likely range -0.15 to -0.00 in/yr</p>")
    out = cr.build(cfg, out_dir=tmp_path / "preview")
    assert out.read_text() == "<p>likely range -0.15 to 0.00 in/yr</p>"
    from insar_pipeline import client_kmz as ck
    assert ck._no_neg_zero("<td>-0.00 in</td><coordinates>-87.384300,38.304400,0</coordinates>") == \
        "<td>0.00 in</td><coordinates>-87.384300,38.304400,0</coordinates>"


def test_built_report_has_no_signed_zero(built):
    import re
    assert not re.search(r"(?<![\w.])[-−]0\.0+(?!\.?\d)", built.html)
    assert not re.search(r"(?<![\w.])[-−]0\.0+(?!\.?\d)", json.dumps(built.rep))


# ------------------------------------------------------------------ 3. one per-cell rule
def test_delivered_snr_is_the_client_rasters_rule(built):
    """|client vertical - GNSS| x up / cell null == rate_signal_to_noise.tif,
    cell for cell: the second-pass relative field and the per-class cell
    null the vertical raster and the KMZ fading use (deliverables wrote the
    SNR from the FIRST farmland pass)."""
    import rasterio
    assert built.rep.get("farmland_second_pass"), "the planted patch must trigger the second farmland pass"
    with rasterio.open(built.D / "vertical_movement_in_per_yr.tif") as r:
        v = r.read(1).astype(float) * 25.4
    with rasterio.open(built.D / "rate_signal_to_noise.tif") as r:
        snr = r.read(1).astype(float)
        tags = r.tags()
    null = {int(k): float(x) for k, x in built.rep["cell_null_by_class"].items()}
    assert set(null) == {4}                                    # tc 0.95 everywhere: one class
    want = np.abs(v - L9b.BG_V) * L9b.UP / null[4]
    ok = np.isfinite(want) & np.isfinite(snr)
    assert ok.sum() > 0.9 * np.isfinite(v).sum()
    assert np.allclose(want[ok], snr[ok], atol=2e-3)
    assert tags.get("WRITTEN_BY") == "client_report"
    s = json.loads((built.D / "signal_to_noise.json").read_text())
    assert s["written_by"].startswith("client_report") and s["percent_standing_out_at_1_96"] == pytest.approx(
        round(100 * float((snr[np.isfinite(snr)] >= 1.96).mean()), 1), abs=0.11)


# ------------------------------------------------------------------ 4. the KMZ's regional sentence
def test_kmz_basis_sentence_is_range_gated(built):
    from insar_pipeline import client_kmz as ck
    rep = json.loads(json.dumps(built.rep))
    txt = ck.basis_sentence(rep)
    assert "the region itself moves" not in txt and "the region is sinking" in txt       # -1.5 +/- 0.6: excludes 0
    rep["gnss"]["background"]["vertical_uncertainty_mm_yr"] = 1.0                       # -1.5 +/- 1.0: crosses 0
    txt = ck.basis_sentence(rep)
    assert "cannot be told apart from no movement" in txt and "sinking" not in txt.split("anchored")[1].split(".")[1]


# ------------------------------------------------------------------ 5. a feature's rate in the patches table
def test_patch_that_is_a_feature_carries_the_features_figure():
    from insar_pipeline import client_report as cr
    feats = [{"vertical_relative_mm_yr": -37.3, "mean_rel_vertical_mm_yr": -35.0}]
    p = {"area": "x", "mean_rel_vertical_mm_yr": -35.1, "feature": 1}
    rate, what = cr.patch_rate_cells(p, feats, 500)
    assert rate == "-1.47 in/yr (Feature 1's figure)"
    assert "Feature 1 (Observations)" in what and "500 m map averages -1.38 in/yr" in what
    rate, what = cr.patch_rate_cells(dict(p, feature=None), feats, 500)
    assert rate == "-1.38 in/yr" and "chance" in what
    rate, _ = cr.patch_rate_cells(p, [{"ground_reshaped": True}], 500)
    assert rate == "no rate"


def test_page_patch_table_uses_the_feature_figure(built):
    from insar_pipeline import client_report as cr
    rep = json.loads(json.dumps(built.rep))
    assert rep["features"]
    f = rep["features"][0]
    fv = f.get("vertical_relative_mm_yr", f["mean_rel_vertical_mm_yr"])
    full = next(a["name"] for a in rep["areas"] if a["kind"] == "full")
    rep["map_patches"] = [{"area": full, "sign": -1, "area_km2": 1.2, "inside_area_km2": 1.2,
                           "mean_rel_vertical_mm_yr": fv * 0.8, "lat": f["lat"], "lon": f["lon"], "feature": 1}]
    html = cr.page(built.cfg, rep, _figs(built))
    row = html.split("Coloured patches inside the report areas")[1].split("</table>")[0]
    assert f"{cr._s(fv / 25.4)} in/yr (Feature 1&#x27;s figure)" in row
    assert f"averages {cr._s(fv * 0.8 / 25.4)} in/yr" in row


# ------------------------------------------------------------------ 6. geolocation: no "within" claim when the halves disagree
def test_geolocation_inconclusive_when_a_half_is_off():
    from insar_pipeline import quality_assessment as qa
    g = qa.judge_geolocation(SITE_B_GEO)
    assert g["flag"] is None and g["status"].startswith("inconclusive") and "within about" not in g["text"]
    assert qa.judge_geolocation(g) == g                                   # idempotent
    agree = dict(SITE_B_GEO, halves={"west": {"offset_m": 5.2}, "east": {"offset_m": 5.6}},
                 west_east_disagreement_m=1.0)
    g = qa.judge_geolocation(agree)
    assert g["flag"] is False and "within about 5 m" in g["text"]
    off = dict(SITE_B_GEO, offset_m=16.0, halves={"west": {"offset_m": 15.0}, "east": {"offset_m": 17.0}},
               west_east_disagreement_m=3.0)
    g = qa.judge_geolocation(off)
    assert g["flag"] is True and "within" not in g["text"] and "17 m off" in g["text"]
    # E22 (trust review 7 Oct 2026): an 11 m half is within one 15 m cell -
    # not flagged; a half over a cell is, once the split test says the halves
    # differ beyond their own scatter
    one_half = dict(agree, halves={"west": {"offset_m": 11.0}, "east": {"offset_m": 5.0}}, west_east_disagreement_m=6.0)
    assert qa.judge_geolocation(one_half)["flag"] is False
    over_cell = dict(agree, halves={"west": {"offset_m": 21.0}, "east": {"offset_m": 5.0}},
                     west_east_disagreement_m=16.0, split_test={"p_value": 0.01})
    assert qa.judge_geolocation(over_cell)["flag"] is True
    assert qa.judge_geolocation(dict(over_cell, split_test={"p_value": 0.28}))["flag"] is False
    assert qa.send_rule({"checks": {}, "geolocation": qa.judge_geolocation(SITE_B_GEO)})["value"] != "send"


def test_report_built_on_an_old_qa_makes_no_position_claim(built):
    geo = built.rep["geolocation"]
    assert geo["flag"] is None and "within about" not in (geo.get("text") or "")
    assert "agree with mapped water edges to within" not in built.html
    assert "inconclusive" in built.html


# ------------------------------------------------------------------ 7. summary.json after the QA
def test_qa_write_puts_its_verdict_and_coverage_in_summary(tmp_path):
    from insar_pipeline import deliverables as dl, quality_assessment as qa
    cfg = L9b.make_project(tmp_path)
    S0 = dict(json.loads((cfg.project_dir / "deliverables" / "summary.json").read_text()),
              coverage_grade="Good", coverage_percent=98.6, data_quality=None)
    (cfg.project_dir / "deliverables" / "summary.json").write_text(json.dumps(S0))
    r = {"verdict": "REVIEW", "passed": 3, "flagged": 3, "send_rule": {"value": "hold for specialist"},
         "checks": {"coverage": {"threshold": 0.4, "aoi_percent_at_threshold": 98.6, "reliable_percent": 41.0,
                                 "threshold_above_noise": False, "noise_level": {"floor": 0.46},
                                 "yardstick_percent": 70, "pass": False}}, "findings": {}}
    dl.clear_cache()
    qa.write(cfg, r)
    S = json.loads((cfg.project_dir / "deliverables" / "summary.json").read_text())
    assert S["data_quality"] == "REVIEW" and S["send_rule"] == "hold for specialist"
    assert S["reliable_percent"] == 41.0 and S["coverage"]["reliable_percent"] == 41.0
    assert S["coverage_grade"] != "Good" and S["coverage_grade"].startswith("Flagged by the quality assessment")
    assert "noise floor" in S["coverage_grade"] and S["coverage_grade_by_share"] == "Good"
    qa.write(cfg, r)                                                       # idempotent
    assert json.loads((cfg.project_dir / "deliverables" / "summary.json").read_text())["coverage_grade_by_share"] == "Good"


# ------------------------------------------------------------------ 8. sign convention on vertical rasters
def test_vertical_rasters_say_down_is_negative(built):
    import rasterio
    from insar_pipeline import deliverables as dl
    with rasterio.open(built.D / "vertical_movement_in_per_yr.tif") as r:
        assert r.tags()["SIGN_CONVENTION"].lower().startswith("negative = down (sinking), vertical, gnss-anchored where stated")
    with rasterio.open(next(built.D.glob("vertical_change_at_average_rate_in_*.tif"))) as r:
        assert r.tags()["SIGN_CONVENTION"].startswith("Negative = down (sinking)")
    cfg = built.cfg
    assert dl.product_tags(cfg, {}, "line-of-sight velocity", "mm/yr")["SIGN_CONVENTION"] == dl.SIGN_NOTE
    assert dl.product_tags(cfg, {}, "vertical velocity ASSUMING all motion is vertical",
                           "mm/yr")["SIGN_CONVENTION"] == dl.VERTICAL_SIGN_NOTE


# ------------------------------------------------------------------ 9. the average-rate raster's name
def test_average_rate_raster_is_named_for_what_it_is(built):
    from insar_pipeline import package
    names = sorted(p.name for p in built.D.glob("*.tif"))
    assert any(n.startswith("vertical_change_at_average_rate_in_") for n in names)
    assert not any(n.startswith("vertical_movement_total") for n in names)
    moved = list((built.D / "superseded").rglob("vertical_movement_total_in_*.tif"))
    assert moved and moved[0].read_bytes() == b"old name"                  # moved aside, never deleted
    for pat, meaning in package.DATA_FILES:
        if pat.pattern.startswith("^vertical_change_at_average_rate") or "movement_total" in pat.pattern:
            assert "average rate × years" in meaning and "over the record" not in meaning
    src = (Path(__file__).resolve().parents[1] / "client_report.py").read_text()
    assert 'f"vertical_movement_total_{unit}' not in src


# ------------------------------------------------------------------ 10. stale or partial workbooks stay out of client/
def _workbook(path: Path, sheets):
    from openpyxl import Workbook
    wb = Workbook()
    wb.active.title = sheets[0]
    for s in sheets[1:]:
        wb.create_sheet(s)
    wb.save(path)


def test_package_skips_stale_or_partial_results_workbooks(tmp_path):
    import os
    from insar_pipeline import package
    rep_f = tmp_path / "client_report.json"
    rep_f.write_text("{}")
    t = rep_f.stat().st_mtime
    old = tmp_path / "old.xlsx"
    _workbook(old, ["Summary", "Notes", "Sources", "Statistics", "Annual", "Points"])
    os.utime(old, (t - 3600, t - 3600))
    assert "before the report was last rebuilt" in package._results_problem([old], t)
    part = tmp_path / "part.xlsx"
    _workbook(part, ["Summary", "Statistics", "Annual"])
    os.utime(part, (t + 60, t + 60))
    assert "no Notes sheet" in package._results_problem([part], t)
    good = tmp_path / "good.xlsx"
    _workbook(good, ["Summary", "Notes", "Sources", "Statistics", "Annual"])
    os.utime(good, (t + 60, t + 60))
    assert package._results_problem([good], t) is None
    assert "every measured point" not in package._workbook_meaning({"points_sheet": False}, [good])
    assert "every measured point" not in package._workbook_meaning({}, [good])          # read from its sheets
    assert "every measured point" in package._workbook_meaning({"points_sheet": True}, [good])


# ------------------------------------------------------------------ 11. the run-summary page
def test_summary_page_totals_cell_size_and_hemispheres():
    from insar_pipeline.scripts import run_summary_page as rsp
    assert rsp.box_text(-12.1, 8.1, -12.0, 8.3) == "12.10000–12.00000 S, 8.10000–8.30000 E"
    assert rsp.box_text(38.2, -87.4, 38.3, -87.3) == "38.20000–38.30000 N, 87.40000–87.30000 W"
    st = {"n": 50, "min": -5.0, "max": 4.0, "mean": -0.2, "median": -0.1}
    res = {"project": "/x/site", "identity": {"title": "Site", "location": "Somewhere"}, "backend": "miaplpy",
           "config": {"looks": [11, 3]}, "summary": {"resolution": {"posting_m": 20.0}, "span_days": 3650},
           "site_history": {"sites": {"Pad": {"cell_stats": {"rate_mm_yr_relative_vs_far_field": st,
                                                             "total_mm_relative_vs_far_field": st},
                                              "box_S_W_N_E": [-12.1, 8.1, -12.0, 8.3], "coherent_cells": 50,
                                              "cells_in_box": 60, "relative_all_cells": {}}}}}
    page = rsp.build(res)
    assert "12.10000–12.00000 S, 8.10000–8.30000 E" in page and "single 20 m cells" in page
    assert "Totals are each cell's rate" not in page and "rate × span" not in page
    assert "measured change over the record" in page


# ------------------------------------------------------------------ 12. the dashboard says what the code does
def test_dashboard_whole_pipeline_text_and_interrupted_rows():
    from insar_pipeline import dashboard
    src = (Path(__file__).resolve().parents[1] / "dashboard.py").read_text()
    assert "re-plans from the start" not in src and "starts over from the download" not in src
    assert "reuses this" in src and "allow_missing_dates" in src
    now = 2000.0
    L = {"unwrap": {"stage": "unwrap", "status": "running", "t": 1500.0, "started": 1000.0, "done": 400, "total": 804},
         "download": {"stage": "download", "status": "done", "t": 900.0, "started": 100.0}}
    rows = {r["key"]: r for r in dashboard.stage_rows(L, {"started": 50.0}, "interrupted", now=now)}
    assert rows["unwrap"]["icon"] == "■" and rows["unwrap"]["note"].startswith("interrupted")
    assert rows["unwrap"]["style"] == "yellow" and not rows["unwrap"]["live"]
    assert rows["download"]["icon"] == "✓"
    rows = {r["key"]: r for r in dashboard.stage_rows(L, {"started": 50.0}, "running", now=now)}
    assert rows["unwrap"]["icon"] == "▶" and "interrupted" not in rows["unwrap"]["note"]


# ------------------------------------------------------------------ 13. short dates: the search, not the archive
def test_short_dates_are_said_to_be_search_gaps_unless_proven(tmp_path):
    from insar_pipeline import client_report as cr, quality_assessment as qa
    cfg = types.SimpleNamespace(project_dir=tmp_path)
    slc = tmp_path / "slc"
    slc.mkdir()
    (slc / "dropped_short_dates.json").write_text(json.dumps({"modal_burst_count": 3,
                                                              "dropped": ["20160516", "20250517"]}))
    r = qa.short_date_reasons(cfg)
    assert all(v.startswith("the archive search returned it incomplete") for v in r.values())
    (slc / "assembly_status.json").write_text(json.dumps({"dropped": {"20160516": "catalogue_short"}}))
    r = qa.short_date_reasons(cfg)
    assert r["20160516"].startswith("the satellite archive holds only part")
    assert r["20250517"].startswith("the archive search returned it incomplete")
    # an older QA's wording is replaced when the report is rebuilt
    old = {"available": 275, "processed": 270, "dropped": {
        "20160516": "the burst catalogue held fewer bursts than the other dates (3), so the date could not be assembled",
        "20250517": "the burst catalogue held fewer bursts than the other dates (3), so the date could not be "
                    "assembled (accepted as missing by allow_missing_dates)"}}
    new = qa.refresh_drop_reasons(cfg, old)
    assert new["dropped"]["20250517"].startswith("the archive search returned it incomplete")
    assert new["dropped"]["20250517"].endswith("(accepted as missing by allow_missing_dates)")
    assert cr._plural_reason("the archive search returned it incomplete") == "the archive search returned them incomplete"


def test_report_line_names_the_search(built):
    from insar_pipeline import client_report as cr
    rep = json.loads(json.dumps(built.rep))
    why = "the archive search returned it incomplete (fewer bursts than the other dates), so it was not assembled"
    rep["record_dates"] = {"available": 275, "processed": 270, "dropped": {d: why for d in
                                                                          ("1", "2", "3", "4", "5")}}
    txt = dict(cr.methods_stages(built.cfg, rep))
    body = " ".join(txt.values())
    assert "5 left out because the archive search returned them incomplete" in body
    assert "burst catalogue" not in body


# ------------------------------------------------------------------ 14. small items
def test_mine_ground_on_a_thin_null_says_approximate():
    from insar_pipeline import report_insights as ri
    rep = {"features": [], "mine_ground": [{"key": "underground", "label": "underground mines", "area_km2": 87.5,
                                            "footprint_km2": 89.4, "relative_vertical_mm_yr": -0.4, "testable": True,
                                            "detected": False, "null_thin": True, "approx_independent_placements": 7,
                                            "null_shape": "footprint"}]}
    t = next(o["text"] for o in ri.observations(rep) if o["kind"] == "mines")
    assert "Only about 7 comparison areas of this size" in t and "approximate" in t


def test_points_header_states_the_measured_chance_share(monkeypatch):
    from insar_pipeline import deliverables as dl
    monkeypatch.setattr(dl, "_chance_percent", lambda cfg: 7.1)
    monkeypatch.setattr(dl, "resolution_info", lambda cfg, attrs: {"posting_m": 15, "text": "Points are 15 m apart."})
    monkeypatch.setattr(dl, "reference_label", lambda cfg: "ref")
    monkeypatch.setattr(dl, "source_line", lambda cfg, attrs: "src")
    cfg = types.SimpleNamespace(backend="miaplpy", min_temporal_coherence=0.4)
    h = "\n".join(dl._points_header(cfg, {}, {"utm_epsg": 32616, "n_independent": 10, "radar_basis": "x"}, 20,
                                    dl.POINT_FIELDS))
    assert "7.1% of ordinary far-field farmland is flagged by chance" in h and "~5%" not in h


def test_null_method_states_the_count_it_carries():
    """Site A Area A1: 'the region footprint fits only ~3 independent
    times' beside approx_independent_placements 16."""
    src = (Path(__file__).resolve().parents[1] / "significance.py").read_text()
    assert "the region footprint fits only ~{n_ind} independent times)" not in src
    assert "fits ~{best[1]} independent times - the region's own footprint" in src


def test_miaplpy_docstrings_name_the_real_folders():
    from insar_pipeline import miaplpy_backend as mb
    assert "through cfg.time_series_dir" not in mb.__doc__ and "<project>/mintpy" in mb.__doc__
    assert "returns MiaplPy's network directory" not in mb.run.__doc__
    assert '"interferograms"' in mb.run.__doc__


def test_pools_are_capped_to_this_machine(monkeypatch):
    import os
    from insar_pipeline import settings, stack_backend as sb
    cfg = types.SimpleNamespace(stack_num_proc_unwrap=(os.cpu_count() or 1) + 40, stack_num_proc_topo=1,
                                stack_num_proc=1, mintpy_num_workers=16, mintpy_max_memory_gb=16.0)
    monkeypatch.setattr(settings, "machine_resources", lambda: {"mem_gb": 7.8, "swap_gb": 2.0, "cores": 8,
                                                                "mem_available_gb": 0.0})
    assert sb._workers_for(cfg, "run_16_unwrap") <= (os.cpu_count() or 1)
    lines = "\n".join(sb._mintpy_compute_lines(cfg))
    cap = settings.sized_parallelism(7.8, 8)
    assert f"numWorker              = {cap['mintpy_num_workers']}" in lines
    assert f"maxMemory              = {float(cap['mintpy_max_memory_gb'])}" in lines


@pytest.mark.parametrize("tropo,expect", [("pyaps", True), ("no", False)])
def test_era5_prefetch_starts_with_the_phase_linking_chain(tmp_path, monkeypatch, tropo, expect):
    from insar_pipeline import miaplpy_backend as mb, notify, pipeline
    from insar_pipeline.config import load_config
    f = tmp_path / "c.yaml"
    f.write_text(f"project_dir: {tmp_path / 'proj'}\nbbox: [-87.7, 38.2, -87.6, 38.3]\n"
                 "start_date: '2020-01-01'\nend_date: '2021-01-01'\n")
    cfg = load_config(str(f))
    cfg.backend, cfg.mintpy_tropo_method = "miaplpy", tropo
    cfg.project_dir.mkdir(parents=True, exist_ok=True)
    started = []
    monkeypatch.setattr(pipeline, "start_era5_prefetch", lambda cfg: started.append(cfg))
    monkeypatch.setattr(pipeline, "check_preconditions", lambda *a, **k: {})
    monkeypatch.setattr(notify, "send", lambda *a, **k: {})

    def stop(cfg):
        raise RuntimeError("stop here")
    monkeypatch.setattr(mb, "run", stop)
    assert pipeline.run_from(cfg, "miaplpy", kind="miaplpy", watchdog=False, notify=False) == 1
    assert bool(started) is expect


def test_no_firm_or_preparer_is_a_blocker_and_shows_on_the_draft(built, monkeypatch):
    from insar_pipeline import release, settings
    assert "Prepared by: not set" in built.html
    monkeypatch.setattr(settings, "identity", lambda: {"firm": "", "preparer": "", "contact": ""})
    b = release.blockers(built.cfg, built.rep)
    assert any(x["code"] == "identity_missing" and x["hard"] for x in b)
    monkeypatch.setattr(settings, "identity", lambda: {"firm": "Acme Geo", "preparer": "K. W.", "contact": ""})
    assert not any(x["code"] == "identity_missing" for x in release.blockers(built.cfg, built.rep))


# ------------------------------------------------------------------ 15. moderate items
def test_points_and_kmz_popups_carry_the_second_check(tmp_path):
    import h5py
    from insar_pipeline import client_kmz as ck, deliverables as dl
    cfg = types.SimpleNamespace(mintpy_dir=tmp_path / "mintpy", backend="miaplpy", min_unwrap_consistency=0.0,
                                project_dir=tmp_path)
    (cfg.mintpy_dir / "geo").mkdir(parents=True)
    with h5py.File(cfg.mintpy_dir / "geo" / "geo_temporalCoherence_mintpy.h5", "w") as f:
        f["temporalCoherence"] = np.array([[0.9, 0.5], [np.nan, 0.71]], np.float32)
    um, src = dl.unwrap_consistency_grid(cfg, (2, 2))
    assert src == "geo_temporalCoherence_mintpy.h5" and um[0, 1] == pytest.approx(0.5)
    assert dl.unwrap_consistency_grid(cfg, (3, 3))[0] is None
    assert dl.second_gate_level(cfg) == 0.7
    cfg.min_unwrap_consistency = 0.8
    assert dl.second_gate_level(cfg) == 0.8
    assert {"unwrap_consistency", "fails_second_gate"} <= set(dl.POINT_FIELDS)
    pv = ck.point_values(-3.0, -3.0, 2.0, None, False)
    d = ck.point_description("Field", 38.3, -87.5, pv, 9.2, 0.9, None, unwrap=0.5, gate=0.7)
    assert "Unwrapping consistency" in d and "fails the second quality check" in d
    d = ck.point_description("Field", 38.3, -87.5, pv, 9.2, 0.9, None, unwrap=0.82, gate=0.7)
    assert "passes the second quality check" in d


def test_points_table_has_the_second_check_columns(tmp_path):
    """point_attributes (points_attributes.gpkg's columns) carries
    unwrap_consistency and fails_second_gate from the MintPy layer."""
    import h5py
    import test_fix_L7a as L7a
    from insar_pipeline import deliverables as dl
    cfg, _ = L7a.make_project(tmp_path)
    dl.clear_cache()
    vel, attrs = dl._read(dl._product(cfg, "velocity.h5"), "velocity")
    uc = np.full(vel.shape, 0.9, np.float32)
    uc[: vel.shape[0] // 2] = 0.3
    with h5py.File(cfg.mintpy_dir / "geo" / "geo_temporalCoherence_mintpy.h5", "w") as f:
        f["temporalCoherence"] = uc
    rows, cols = np.where(dl.accepted_points(cfg, vel))
    data, _ = dl.point_attributes(cfg, vel, attrs, rows, cols)
    assert set(np.unique(data["fails_second_gate"])) == {0, 1}
    assert np.array_equal(data["unwrap_consistency"] < 0.7, data["fails_second_gate"] == 1)
    dl.clear_cache()


def test_dem_change_flags_reshaped_ground_and_says_so():
    from insar_pipeline import client_report as cr, report_insights as ri
    rng = np.random.default_rng(3)
    ny = nx = 200
    attrs = {"X_FIRST": -87.6, "Y_FIRST": 38.4, "X_STEP": 0.00017, "Y_STEP": -0.000135}
    dh = rng.normal(0, 5.0, (ny, nx)).astype(np.float32)
    good = np.ones((ny, nx), bool)
    spoil = np.zeros((ny, nx), bool)
    spoil[60:140, 60:140] = True                            # ~1.2 km: reshaped by ~15 m
    dh[spoil] -= 15.0
    far = ~spoil
    sm = cr._dem_smoothed(dh, good, attrs)
    # E1 (trust review 7 Oct 2026): tested after what the movement explains
    # is taken out (report_insights.dem_change); the spoil heap is not moving,
    # so its whole difference is left over
    rate = cr._dem_smoothed(rng.normal(0, 1.0, (ny, nx)).astype(np.float32), good, attrs)
    d = ri.dem_change(sm, rate, spoil, far)
    assert d["flag"] is True and d["rate_removed"] and d["raw_flag"] and d["feature_median_m"] < -10
    calm = np.zeros((ny, nx), bool)
    calm[150:190, 10:50] = True
    assert ri.dem_change(sm, rate, calm, far & ~calm)["flag"] is False
    f = {"sign": -1, "mean_rel_vertical_mm_yr": -30.0, "vertical_relative_mm_yr": -30.0, "total_relative_mm": -300.0,
         "mines": {"key": "none", "phrase": "away from every mapped mine"}, "dem_change": d,
         "ground_reshaped_since_dem": True, "ground_reshaped": False, "chance": {"tested": True, "p": 0.01,
                                                                            "copies": 59, "copies_as_strong": 0}}
    f["cause_hint"] = ri.cause_hint("none", -1, None, reshaped_since_dem=True)
    t = ri.feature_text(1, f)
    assert "reshaped since the elevation model was made" in t and "earthmoving" in t
    assert "earthmoving" not in ri.cause_hint("none", -1, None)


def test_comparison_ground_label_reaches_the_kmz_and_the_geocode_step_the_latitude():
    from insar_pipeline import areas, client_kmz as ck
    pv = ck.point_values(-3.0, -3.0, 2.0, None, False)
    d = ck.point_description("Field", 38.3, -87.5, pv, 9.2, 0.9, None, ground="pasture")
    assert "pasture" in d and "farmland" not in d
    assert "pasture" in ck.series_title("Feature 1", -3.0, -30.0, True, "pasture")
    assert "pasture" in ck.standout_text({"lat": 38.3, "lon": -87.4, "vertical_relative_mm_yr": -5.0,
                                          "gap_in_spreads": 2.0}, 0, [], "pasture")
    rep = {"anchored": False, "control_ground_label": "pasture", "gnss_reason": "no station"}
    assert "ordinary pasture" in ck.basis_sentence(rep)
    assert areas.geocode_lalo_step(38.3) == "-0.000135,0.000172"
    assert areas.geocode_lalo_step(55.0).endswith(",0.000235")


def test_write_project_sets_the_geocode_step_for_its_latitude(tmp_path):
    from insar_pipeline import areas
    from insar_pipeline.config import load_config
    aoi = areas.box(55.30, -1.70, 55.34, -1.64, "site")
    out = areas.write_project("Northern Site", aoi, "miaplpy", "2020-01-01", "2024-12-31", track=19,
                              direction="ascending", swaths=[2], report_areas=[("Full Satellite Imaging", "full")],
                              projects_root=tmp_path / "p", configs_dir=tmp_path / "c")
    cfg = load_config(str(out))
    assert cfg.mintpy_geocode_lalo_step == areas.geocode_lalo_step(0.5 * (cfg.bbox[1] + cfg.bbox[3]))
    assert cfg.mintpy_geocode_lalo_step != "-0.000135,0.000172"
