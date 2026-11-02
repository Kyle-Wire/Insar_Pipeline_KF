"""Lane L9b of the 5 Oct 2026 review: the client KMZ (client_kmz.py) and the
Results export (results.py, results_screen.py).

empirical-9, clientout-2/-4 (L5a)   one figure per area: client_report.json
                                    = KMZ balloon = KMZ chart slope = workbook
                                    = report raster mean
empirical-8                         point columns cannot be mistaken for the
                                    raw LOS rasters
techout-6, gap-licensing-3,         no GNSS claim without a GNSS tie; one
  clientout-11                      wording; KMZ named from the title
empirical-11                        per-cell sigma by class index
clientout-3, empirical-2,           the report's features in the KMZ, word for
  gap-geolocation-1, methodology-12 word; no fixed Site A sentence
techout-2                           the workbook is complete or absent; the
                                    dashboard survives a failed export
open requests                       placement-field null and three-state
                                    annual reading (stats-1, clientout-14),
                                    one per-cell rule (tests-5, empirical-3),
                                    independent measurements (gap-geolocation-3),
                                    credits (gap-licensing-1), screening and
                                    draft naming (disp-1), too-fast cells
                                    (gap-surface_mine_site-8), the map's scale

Every test checks an OUTCOME. The end-to-end ones share one small geocoded
project (120 x 120 cells of ~100 m, 280 dates; the builder of
test_fix_L5.py) with its client report built once; the KMZ is built with a
few points per area so the module stays fast.
"""
import asyncio
import csv
import html as htmlmod
import io
import json
import re
import types
import zipfile
from pathlib import Path

import h5py
import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from insar_pipeline import client_kmz as ck
from insar_pipeline import client_report as cr
from insar_pipeline import results as rs
from insar_pipeline import significance as sg
from insar_pipeline.config import PipelineConfig

PKG = Path(__file__).resolve().parents[1]
NY = NX = 120
X0, Y0 = -87.60, 38.40
DX, DY = 0.00115, -0.0009
BBOX = [X0, Y0 + NY * DY, X0 + NX * DX, Y0]
UP = float(np.cos(np.radians(39.5)))
FEATURE = (slice(30, 45), slice(70, 85))
FIELD_BOX = (slice(80, 100), slice(20, 40))
PIT_BOX = (slice(5, 25), slice(5, 25))
FAST_BOX = (slice(82, 92), slice(22, 32))   # cells flagged too fast to measure, inside the Test Field
BG_V, BG_U = -1.5, 0.6


def _lalo(r, c):
    return Y0 + (r + 0.5) * DY, X0 + (c + 0.5) * DX


def _box_spec(sl):
    s_, w = _lalo(sl[0].stop - 1, sl[1].start)
    n, e = _lalo(sl[0].start, sl[1].stop - 1)
    s_, n, w, e = s_ - 0.25 * abs(DY), n + 0.25 * abs(DY), w - 0.25 * DX, e + 0.25 * DX
    return f"{s_:.6f},{w:.6f},{n:.6f},{e:.6f}"


def _dates(n=280, step=12):
    import datetime as dt
    d0 = dt.date(2016, 1, 5)
    return [(d0 + dt.timedelta(days=step * i)).strftime("%Y%m%d") for i in range(n)]


def _h5(path, data: dict, attrs, dates=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        for k, v in data.items():
            f.create_dataset(k, data=v)
        if dates is not None:
            f.create_dataset("date", data=np.array([d.encode() for d in dates]))
        for k, v in attrs.items():
            f.attrs[k] = v


def make_project(root: Path, *, gnss=True, seed=7) -> PipelineConfig:
    """test_fix_L5.make_project: farmland with persistent field-to-field rates
    and atmosphere-like noise, an unmapped settling patch (FEATURE), an
    ordinary-farmland report area and a 70%-unmeasured pit."""
    cfg = PipelineConfig(project_dir=root / "proj", bbox=BBOX, start_date="2016-01-01", end_date="2025-03-31",
                         backend="miaplpy", acceptance_far_field_buffer_m=500.0,
                         project_title="Testsite", project_location="Test County, Indiana",
                         report_areas=("Full Satellite Imaging:full", f"Test Field:{_box_spec(FIELD_BOX)}",
                                       f"Wet Pit:{_box_spec(PIT_BOX)}"))
    geo = cfg.mintpy_dir / "geo"
    dates = _dates()
    t = sg.years(dates)
    rng = np.random.default_rng(seed)
    rate = gaussian_filter(rng.normal(0, 1, (NY, NX)), 3)
    rate = rate / rate.std() * 2.0
    ts = np.empty((len(dates), NY, NX), np.float32)
    for i in range(len(dates)):
        atm = gaussian_filter(rng.normal(0, 1, (NY, NX)), 6)
        ts[i] = (rate * t[i] + atm / atm.std() * 3.0 + rng.normal(0, 2.0, (NY, NX))
                 + 6.0 * np.sin(2 * np.pi * t[i])) / 1000.0
    ts[:, FEATURE[0], FEATURE[1]] += (-250.0 * (1.0 - np.exp(-t / 1.5)) / 1000.0)[:, None, None]
    vel = np.polyfit(t, ts.reshape(len(t), -1), 1)[0].reshape(NY, NX).astype(np.float32)
    A = {"X_FIRST": str(X0), "Y_FIRST": str(Y0), "X_STEP": str(DX), "Y_STEP": str(DY), "LENGTH": str(NY),
         "WIDTH": str(NX), "REF_Y": "3", "REF_X": "3", "REF_LAT": str(Y0 + 3.5 * DY), "REF_LON": str(X0 + 3.5 * DX),
         "UNIT": "m/year", "START_DATE": dates[0], "END_DATE": dates[-1], "REF_DATE": dates[0]}
    mask = np.ones((NY, NX), bool)
    pit = np.zeros((NY, NX), bool)
    pit[PIT_BOX] = True
    rr, cc = np.nonzero(pit)
    mask[rr[: int(0.7 * rr.size)], cc[: int(0.7 * rr.size)]] = False
    _h5(geo / "geo_velocity.h5", {"velocity": vel}, A)
    _h5(geo / "geo_timeseries_ERA5_ramp_demErr.h5", {"timeseries": ts}, A, dates)
    _h5(geo / "geo_maskTempCoh.h5", {"mask": mask}, A)
    _h5(geo / "geo_temporalCoherence.h5", {"temporalCoherence": np.full((NY, NX), 0.95, np.float32)}, A)
    _h5(geo / "geo_geometryRadar.h5", {"incidenceAngle": np.full((NY, NX), 39.5, np.float32),
                                       "azimuthAngle": np.full((NY, NX), -100.0, np.float32),
                                       "height": np.zeros((NY, NX), np.float32)}, A)
    D = cfg.project_dir / "deliverables"
    (D / "figures").mkdir(parents=True, exist_ok=True)
    (D / "summary.json").write_text(json.dumps({"total_measured_points": NY * NX, "track": 92,
                                                "orbit_direction": "ASCENDING", "look_angle_deg": 39.5}))
    # class 4 is the FIRST entry of by_class: by list position it read as class 0 (empirical-11)
    (D / "uncertainty_model.json").write_text(json.dumps({
        "by_class": [{"tc": "0.9-1.0", "class_index": 4, "empirical_mm_yr": 1.0, "n": 14000}],
        "cell_null_by_class": {"4": 2.5}}))
    (cfg.project_dir / "quality_assessment.json").write_text(json.dumps({
        "checks": {"coverage": {"aoi_percent_at_threshold": 97.0, "pass": True},
                   "network": {"connected": True, "pass": True}},
        "findings": {}, "passed": 2, "flagged": 0, "verdict": "MEETS PRACTICE"}))
    if gnss:
        (D / "gnss_tie.json").write_text(json.dumps({
            "record_window": [dates[0], dates[-1]], "stations": [], "inside_checks": [], "los_up_median": UP,
            "uncertainty_floor_mm_yr": BG_U,               # a current tie (gnss_tie.stale_reason)
            "background": {"vertical_mm_yr": BG_V, "vertical_uncertainty_mm_yr": BG_U, "los_mm_yr": BG_V * UP,
                           "uncertainty_basis": "regional floor"}}))
    return cfg


def _write_too_fast(cfg):
    """A quality_too_fast_to_measure.tif on the product grid (1 in FAST_BOX)."""
    import rasterio
    from rasterio.transform import from_origin
    flag = np.zeros((NY, NX), np.uint8)
    flag[FAST_BOX] = 1
    p = cfg.project_dir / "deliverables" / "quality_too_fast_to_measure.tif"
    with rasterio.open(p, "w", driver="GTiff", height=NY, width=NX, count=1, dtype="uint8", nodata=255,
                       crs="EPSG:4326", transform=from_origin(X0, Y0, DX, -DY)) as dst:
        dst.write(flag, 1)


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    """The project, its client report (json, html, figures, rasters), the KMZ
    previewed into another folder, and the Results of every report area."""
    root = tmp_path_factory.mktemp("l9b")
    cfg = make_project(root)
    cr.build(cfg)
    _write_too_fast(cfg)
    D = cfg.project_dir / "deliverables"
    rep = json.loads((D / "client_report.json").read_text())
    mp = pytest.MonkeyPatch()
    mp.setattr(ck, "PER_AREA", 8)
    mp.setattr(ck, "PER_FEATURE", 4)
    before = sorted(p.name for p in D.iterdir())
    kmz = ck.build(cfg, out_dir=root / "preview")
    after = sorted(p.name for p in D.iterdir())
    mp.undo()
    z = zipfile.ZipFile(kmz)
    from insar_pipeline.panel_history import _Context
    ctx = _Context(cfg)
    res = {a["name"]: rs.analyse(cfg, a["name"], ctx=ctx) for a in rep["areas"]}
    return types.SimpleNamespace(cfg=cfg, rep=rep, kmz=kmz, z=z, kml=z.read("doc.kml").decode(), ctx=ctx, res=res,
                                 root=root, D=D, deliverables_before=before, deliverables_after=after)


def _placemark(kml, name):
    m = re.search(r"<Placemark><name>" + re.escape(htmlmod.escape(name)) + r"</name>.*?<!\[CDATA\[(.*?)\]\]>", kml, re.S)
    return m.group(1) if m else None


def _plain(s):
    return htmlmod.unescape(re.sub(r"<[^>]+>", " ", s))


def _points(kml):
    """(style, balloon, lon, lat) as Google Earth shows them (13 Oct 2026: one
    shared balloon per style, each point's values in ExtendedData)."""
    return ck.expand_placemarks(kml)


def _xlsx_summary(path):
    import openpyxl
    with open(path, "rb") as fh:
        wb = openpyxl.load_workbook(fh, read_only=True)
        rows = list(wb["Summary"].iter_rows(values_only=True))[1:]
        return {r[0]: r[1] for r in rows}, wb.sheetnames, wb.properties


# ------------------------------------------------------------------ one figure per area (empirical-9, clientout-2/-4)
def test_every_area_figure_is_the_same_in_json_kmz_chart_workbook_and_raster(built, tmp_path):
    """client_report.json = the KMZ balloon = the KMZ chart's slope = the
    workbook's Summary = the report raster's mean over the area. Until 5 Oct
    2026 Site B's whole area read -1.85 (report), -1.07 (workbook) and
    the KMZ recomputed its own."""
    import rasterio
    t = sg.years(built.rep["dates_list"])
    with rasterio.open(built.D / "vertical_movement_in_per_yr.tif") as r:
        rast = r.read(1).astype(float) * 25.4
    checked = 0
    for i, a in enumerate(built.rep["areas"]):
        v = a.get("vertical_mm_yr")
        assert v is not None
        # the KMZ balloon prints the report's value
        bal = _plain(_placemark(built.kml, a["name"]))
        assert f"({v:+.1f} mm)" in bal, (a["name"], bal[:300])
        assert a["reading"] in bal
        # the chart's data (charts/area<i>.csv) has the report's slope
        rows = list(csv.reader(io.StringIO(built.z.read(f"charts/area{i}.csv").decode())))[1:]
        y = np.array([float(q[1]) if q[1] else np.nan for q in rows])
        ok = np.isfinite(y)
        assert np.polyfit(t[ok], y[ok], 1)[0] == pytest.approx(v, abs=0.02)
        # the workbook's Summary sheet
        out = rs.export(built.cfg, built.res[a["name"]], out_dir=tmp_path / f"x{i}")
        summ, _, _ = _xlsx_summary(next(out.glob("*.xlsx")))
        assert summ["mean_vertical_velocity_in_yr"] * 25.4 == pytest.approx(v, abs=0.02)
        assert summ["reading"] == a["reading"]
        if a.get("signal_to_noise") is not None:
            assert summ["signal_to_noise"] == pytest.approx(a["signal_to_noise"], abs=0.03)
        if a.get("detection_limit_mm_yr"):
            assert summ["reliably_detects_in_yr"] * 25.4 == pytest.approx(a["detection_limit_mm_yr"], abs=0.15)
        if a.get("total_mm") is not None:
            assert summ["mean_vertical_cumulative_in"] * 25.4 == pytest.approx(a["total_mm"], abs=0.2)
        # the report raster, averaged over the area's measured cells
        m = rs.area_region(built.cfg, built.ctx, a["name"])["mask"]
        assert float(np.nanmean(rast[m])) == pytest.approx(v, abs=0.05)
        checked += 1
    assert checked == 3


def test_results_whole_area_is_the_reports_and_never_tested_against_its_own_farmland(built):
    """empirical-9: the whole area ran sg.assess against farmland that is
    mostly itself ('stands out' in 2017 at SNR 7.3) and gave a third rate."""
    full = next(a for a in built.rep["areas"] if a["kind"] == "full")
    s = built.res[full["name"]].area_summary
    assert s["kind"] == "full"
    assert s["mean_vertical_velocity_in_yr"] * 25.4 == pytest.approx(full["vertical_mm_yr"], abs=0.05)
    assert s["reading"] == full["reading"] and "normal ground variation" not in s["reading"]
    assert s["reliably_detects_in_yr"] is None and s["stands_out_above_in_yr"] is None
    lo, hi = s["vertical_95pct_range_in_yr"]
    assert lo * 25.4 == pytest.approx(full["range95_mm_yr"][0], abs=0.05)
    for r in built.res[full["name"]].annual:
        assert r["area_reading"] not in ("stands out", "within normal variation")


def test_results_use_the_reports_farmland_and_placement_null(built):
    """clientout-14 / stats-3: the null is placed on the placement field over
    the report's (second-pass) farmland, so the threshold and the limit are
    the report's, not ones widened by the found movement itself."""
    a = next(a for a in built.rep["areas"] if a["name"] == "Test Field")
    s = built.res["Test Field"].area_summary
    assert built.rep["farmland_second_pass"]["features_removed"] >= 1
    # INT2 (empirical-9 request): the report stores the exact cells it cut,
    # so the workbook's farmland is the report's to the cell
    assert s["farmland_second_pass"]["method"] == "recorded cut"
    want = built.rep["farmland_second_pass"]["far_cells"][1]
    assert s["farmland_second_pass"]["far_cells"] == want
    assert s["stands_out_above_in_yr"] * 25.4 == pytest.approx(a["decision_threshold_mm_yr"], abs=0.03)
    assert s["vertical_95pct_range_in_yr"][0] * 25.4 == pytest.approx(a["range95_mm_yr"][0], abs=0.03)


def test_a_report_without_the_recorded_cut_uses_the_feature_outlines(built):
    """A client_report.json from before the cut was stored (INT2) is still
    read: the farmland is rebuilt from the feature outlines, within 2%."""
    import copy
    from insar_pipeline.panel_history import _Context
    rep = copy.deepcopy(built.rep)
    rep["farmland_second_pass"].pop("cut_cells", None)
    res = rs.analyse(built.cfg, "Test Field", ctx=_Context(built.cfg), rep=rep)
    sp = res.area_summary["farmland_second_pass"]
    want = built.rep["farmland_second_pass"]["far_cells"][1]
    assert sp["method"] == "feature outlines" and abs(sp["far_cells"] - want) <= 0.02 * want


def test_without_feature_outlines_the_reports_class_centres_still_hold(built):
    """A report without stored outlines: only the recorded class-centre shift
    can be applied - the rates are still the report's."""
    import copy
    from insar_pipeline.panel_history import _Context
    rep = copy.deepcopy(built.rep)
    rep["farmland_second_pass"].pop("cut_cells", None)
    for f in rep["features"]:
        f.pop("ring", None)
        f.pop("bbox", None)
    res = rs.analyse(built.cfg, "Test Field", ctx=_Context(built.cfg), rep=rep)
    a = next(a for a in built.rep["areas"] if a["name"] == "Test Field")
    assert res.area_summary["farmland_second_pass"]["method"] == "class shift"
    assert res.area_summary["mean_vertical_velocity_in_yr"] * 25.4 == pytest.approx(a["vertical_mm_yr"], abs=0.02)
    assert any("no feature outlines" in n for n in res.notes)


def test_preview_kmz_goes_to_the_preview_folder_only(built):
    assert built.kmz.parent == built.root / "preview"
    assert built.deliverables_after == built.deliverables_before


# ------------------------------------------------------------------ annual rows (stats-1)
def test_annual_reading_has_three_states_and_follows_one_rule(built):
    rows = built.res["Test Field"].annual
    assert rows
    for r in rows:
        s = r["area_signal_to_noise"]
        head, _, note = r["area_reading"].partition("; ")
        assert head in ("stands out", "within normal variation")
        assert (head == "stands out") == (s >= 1.0)
        # D1 (7 Oct 2026): this project is GNSS-anchored, so a year the test
        # did not detect but whose absolute range still excludes 0 says it
        # moves with the surrounding region - and only such a year says so
        lo = next(v for k, v in r.items() if k.startswith("area_mean_95pct_low_"))
        hi = next(v for k, v in r.items() if k.startswith("area_mean_95pct_high_"))
        assert lo is not None and hi is not None and lo <= hi
        regional = head == "within normal variation" and (lo > 0 or hi < 0)
        assert note == ("moves with the surrounding region" if regional else ""), r
    for r in built.res["Wet Pit"].annual:
        assert r["area_reading"].startswith("not testable")
        assert r["area_signal_to_noise"] is None and r["area_mean_95pct_low_in_yr"] is None


# ------------------------------------------------------------------ column names (empirical-8)
def test_point_columns_cannot_be_confused_with_the_raw_los_rasters(built):
    import rasterio
    res = built.res["Test Field"]
    cols = set(res.points)
    raw = {p.stem.replace("_per_year", "_yr").replace("_per_yr", "_yr") for p in built.D.glob("los_*.tif")}
    assert "los_velocity_in_yr" not in cols and "los_cumulative_in" not in cols and not (cols & raw)
    assert "los_velocity_rel_farmland_gnss_in_yr" in cols
    assert "vertical_change_at_average_rate_in" in cols and "vertical_cumulative_in" not in cols
    # the vertical column IS the delivered vertical raster at those cells
    with rasterio.open(built.D / "vertical_movement_in_per_yr.tif") as r:
        rast = r.read(1)
    m = rs.area_region(built.cfg, built.ctx, "Test Field")["mask"]
    np.testing.assert_allclose(res.points["vertical_velocity_in_yr"], rast[m], atol=2e-3)


# ------------------------------------------------------------------ no false GNSS claim (techout-6, gap-licensing-3)
def test_results_without_a_gnss_tie_say_relative_and_never_anchored(tmp_path):
    cfg = make_project(tmp_path, gnss=False)
    res = rs.analyse(cfg, "Test Field")
    s = res.area_summary
    text = " ".join(res.notes)
    assert s["gnss_anchored"] is False and s["gnss_background_vertical_mm_yr"] is None
    assert "anchored to the regional GNSS" not in text and "+0.00 mm/yr" not in text
    assert "NOT GNSS-anchored" in text and "los_velocity_rel_farmland_in_yr" in res.points
    assert s["vertical_95pct_range_in_yr"] is not None                 # relative range, from farmland alone
    full = rs.analyse(cfg, "Full Satellite Imaging").area_summary
    assert full["mean_vertical_velocity_in_yr"] is None and full["relative_only"] is True
    assert full["reading"].startswith("Relative only")


def test_kmz_wording_follows_the_reports_anchoring(built):
    rep = json.loads(json.dumps(built.rep))
    rep["anchored"] = False
    rep["gnss_reason"] = "no GNSS station overlapped the record"
    rep["gnss"] = {"background": None, "anchored": False, "reason": rep["gnss_reason"]}
    area = next(a for a in rep["areas"] if a["name"] == "Test Field")
    for txt in (ck.doc_description(built.cfg, rep, False, [], False), ck.area_description(area, rep),
                ck.basis_sentence(rep)):
        txt = _plain(txt)
        # no land-cover map on this synthetic project: "ground" (28 Oct 2026)
        assert "anchored to" not in txt and "relative to the area's ordinary ground" in txt
        assert "no GNSS station overlapped the record" in txt
    pv = ck.point_values(-3.0, -3.0, 2.0, None, False)
    assert "relative to the surrounding farmland" in ck.point_description("Test Field", 38.3, -87.5, pv, 9.2, 0.9, None)
    # anchored: the report's own sentence, with its regional rate
    assert "anchored to the regional GNSS network" in ck.basis_sentence(built.rep)
    # the report's regional sentence (range-gated), not "the region itself moves -1.5 mm/yr"
    assert "-1.5" in ck.basis_sentence(built.rep) and "the region itself moves" not in ck.basis_sentence(built.rep)


# ------------------------------------------------------------------ names (clientout-11, disp-1, gap-licensing-1)
def test_kmz_is_named_from_the_title_with_draft_and_screening_marks(tmp_path):
    cfg = PipelineConfig(project_dir=tmp_path / "site_d", bbox=BBOX, start_date="2016-01-01", end_date="2025-03-31")
    rep = {"title": "Site D", "dates": ["20160105", "20250306"]}
    assert ck.kmz_name(cfg, rep, False, False) == "Site_D_ground_movement_DRAFT.kmz"
    assert ck.kmz_name(cfg, rep, True, False) == "Site_D_ground_movement.kmz"
    assert ck.kmz_name(cfg, dict(rep, title="Site D (screening)"), True, True) == "Site_D_SCREENING_ground_movement.kmz"
    assert ck.doc_name(cfg, rep, True, True).startswith("SCREENING – Site D: ground movement 2016–2025")
    cfg.backend = "disp"
    assert ck._screening(cfg, {}) is True
    assert "SCREENING ONLY" in ck.doc_description(cfg, dict(rep, n_dates=10, anchored=False), True, [], True)
    assert ck.latlon_text(-33.9, 151.2, 2) == "33.90 S, 151.20 E"
    assert ck.latlon_text(38.3, -87.38, 2) == "38.30 N, 87.38 W"


def test_built_kmz_is_a_draft_named_for_the_project_with_credits(built, monkeypatch, tmp_path):
    assert built.kmz.name == "Testsite_ground_movement_DRAFT.kmz"
    doc = re.search(r"<Document><name>(.*?)</name>", built.kml).group(1)
    assert doc.startswith("Testsite: ground movement 2016") and doc.endswith("(DRAFT)")
    head = _plain(re.search(r"<Document>.*?<!\[CDATA\[(.*?)\]\]>", built.kml, re.S).group(1))
    assert "DRAFT - not for release" in head and "Copernicus Sentinel data" in head
    assert "Data sources and credits" in head
    assert re.search(r"<Data name='numbers_hash'><value>[0-9a-f]{16,}</value>", built.kml)
    # signed off: no DRAFT in the name or the title
    monkeypatch.setattr(ck, "release_state", lambda cfg, rep: (True, []))
    monkeypatch.setattr(ck, "PER_AREA", 2)
    monkeypatch.setattr(ck, "PER_FEATURE", 1)
    k = ck.build(built.cfg, out_dir=tmp_path, rep=built.rep, ctx=built.ctx)
    assert k.name == "Testsite_ground_movement.kmz" and not list(tmp_path.glob("*.partial"))


# ------------------------------------------------------------------ features, no fixed sentence (clientout-3 & co.)
FIXED = ("Outside every mapped mine panel", "the only area that stands clear", "cannot say why")


def test_kmz_lists_the_reports_features_word_for_word(built):
    feats = built.rep["features"]
    assert feats, "the planted patch is a feature in the report"
    folder = re.search(r"<Folder><name>Features</name>(.*?)</Folder>", built.kml, re.S).group(1)
    names = re.findall(r"<Placemark><name>(Feature \d+)</name>", folder)
    assert names == [f"Feature {i}" for i in range(1, len(feats) + 1)]
    for i, f in enumerate(feats, 1):
        bal = _placemark(built.kml, f"Feature {i}")
        assert htmlmod.escape(f["text"]) in bal                    # the report's paragraph, verbatim
        assert f"{f['vertical_relative_mm_yr']:+.1f} mm" in bal
        assert f"charts/feature{i}.png" in bal and f"charts/feature{i}.csv" in built.z.namelist()
    for s in FIXED:
        assert s not in built.kml


def test_the_fixed_site_a_sentence_is_gone_from_the_code():
    src = (PKG / "client_kmz.py").read_text()
    for s in FIXED:
        assert s not in src


def test_standout_text_is_built_from_its_data():
    so = {"lat": 38.3044, "lon": -87.3843, "scale_km": 1.0, "vertical_mm_yr": -52.5, "vertical_relative_mm_yr": -50.6,
          "gap_in_spreads": 4.28, "inside_mapped_panels": True, "within_panel_buffer": True}
    t = ck.standout_text(so, 3, ["underground workings"])
    assert "inside a mapped mine area" in t and "Outside every" not in t and "the only area" not in t
    assert "Sinking" in t and "1.99 in/yr" in t and "3 features" in t and "4.3 farmland spreads" in t
    t = ck.standout_text(dict(so, inside_mapped_panels=False), 1, ["underground workings"])
    assert "next to mapped mine areas" in t
    t = ck.standout_text(dict(so, inside_mapped_panels=False, within_panel_buffer=False), 0, ["x"])
    assert "outside every mapped mine area and its buffer" in t
    assert "No mine map was supplied" in ck.standout_text(dict(so, inside_mapped_panels=False,
                                                                within_panel_buffer=False), 0, [])
    t = ck.standout_text({"lat": 38.3, "lon": -87.4, "scale_km": 1, "vertical_mm_yr": 3.0, "in_mapped_mine": True}, 0, ["x"])
    assert t.count("Rising") == 1 and "in or next to mapped mine ground" in t


# ------------------------------------------------------------------ per-cell rule (empirical-11, tests-5, empirical-3)
def test_point_balloons_use_the_cell_null_by_class_index_and_one_rule(built):
    """The model lists class 4 first: by list position it was class 0 and the
    class-4 points got a made-up 5 mm/yr. Every point: the farmland range
    excludes zero exactly when signal / threshold >= 1, and faded <=> < 1."""
    pts = _points(built.kml)
    assert len(pts) > 15
    half_in = 1.96 * 2.5 / UP / 25.4                       # cell_null_by_class {"4": 2.5}, vertical, inches
    n_checked = 0
    for style, desc, _, _ in pts:
        txt = _plain(desc)
        if "Not measurable" in txt:
            continue
        # a value that rounds to zero prints unsigned ('0.00'), as on the page
        m = re.search(r"Compared with ordinary ground\s+([-+]?\d+\.\d+) in/yr, 95% ([-+]?\d+\.\d+) to ([-+]?\d+\.\d+)", txt)
        s2t = float(re.search(r"Signal ÷ threshold\s+(\d+\.\d+)", txt).group(1))
        rel, lo, hi = (float(x) for x in m.groups())
        assert (hi - lo) / 2 == pytest.approx(half_in, abs=0.011)
        excl = lo > 0 or hi < 0
        if abs(s2t - 1.0) > 0.06:                           # printed to 0.1 / 0.01 in
            assert excl == (s2t >= 1.0), txt
            assert style.endswith("f") == (s2t < 1.0)
        n_checked += 1
    assert n_checked > 15


def test_cells_too_fast_to_measure_say_so_instead_of_a_rate(built):
    """gap-surface_mine_site-8."""
    hit = 0
    lat_hi, lat_lo = _lalo(FAST_BOX[0].start, 0)[0], _lalo(FAST_BOX[0].stop - 1, 0)[0]
    lon_lo, lon_hi = _lalo(0, FAST_BOX[1].start)[1], _lalo(0, FAST_BOX[1].stop - 1)[1]
    for style, desc, lon, lat in _points(built.kml):
        inside = lat_lo - 1e-6 <= float(lat) <= lat_hi + 1e-6 and lon_lo - 1e-6 <= float(lon) <= lon_hi + 1e-6
        if inside:
            hit += 1
            assert style == "nm" and "Not measurable - too fast for this satellite" in desc
            assert "Vertical, per year" not in desc
        else:
            assert style != "nm"
    assert hit >= 1


def test_point_values_one_rule_at_the_boundary():
    for rel in np.linspace(-8, 8, 33):
        pv = ck.point_values(rel - 1.5, rel, 2.0, 0.6, True)
        lo, hi = pv["rel_range"]
        assert (lo > 0 or hi < 0) == (pv["signal_to_threshold"] > 1.0)
        assert pv["faded"] == (pv["signal_to_threshold"] < 1.0)
    assert ck.point_values(-1.0, -1.0, float("nan"), None, False)["faded"] is True


# ------------------------------------------------------------------ the client map's scale and band
def test_point_colours_and_overlay_are_the_client_maps():
    from matplotlib import colormaps
    from matplotlib.colors import TwoSlopeNorm
    bins = ck.point_bins(0.6)
    edges = [b[1] for b in bins[:-1]]
    np.testing.assert_allclose(edges, [-0.4, -0.2, -0.1, 0.1, 0.2, 0.4])
    want = colormaps["RdBu"](TwoSlopeNorm(0, -0.6, 0.6)(-0.6))[:3]
    np.testing.assert_allclose(bins[0][2], want)
    assert bins[3][2] == ck.GREY
    np.testing.assert_allclose([b[1] for b in ck.point_bins(1.2)[:3]], [-0.8, -0.4, -0.2])
    # the overlay greys the report's (asymmetric, farmland-centred) band, not +/- a half-width around zero
    ny, nx = 40, 40
    field = np.tile(np.linspace(-12, 12, nx), (ny, 1))          # vertical mm/yr relative to farmland
    attrs = {"X_FIRST": "-87.6", "Y_FIRST": "38.4", "X_STEP": "0.001", "Y_STEP": "-0.001"}
    ctx = types.SimpleNamespace(r_cm=field, good=np.ones((ny, nx), bool), attrs=attrs)
    cfg = types.SimpleNamespace(bbox=[-87.6, 38.4 - ny * 0.001, -87.6 + nx * 0.001, 38.4])
    rep = {"map_smoothing_m": 1, "map_band": {"low_mm_yr": -9.0, "high_mm_yr": 3.0}, "map_band_mm_yr": 6.0}
    png, _ = ck._overlay(cfg, ctx, np.ones((ny, nx)), 0.0, rep, 0.6, 100.0)
    from PIL import Image
    im = np.asarray(Image.open(io.BytesIO(png)))
    grey = (np.abs(im[..., :3].astype(int) - 219).max(-1) <= 1) & (im[..., 3] == 140)
    cols = np.where(grey[ny // 2])[0]
    xv = np.linspace(-12, 12, nx)
    assert xv[cols].min() > -9.0 - 0.7 and xv[cols].max() < 3.0 + 0.7
    assert xv[cols].min() < -5.5                                  # +/- 6 around zero would stop at -6


# ------------------------------------------------------------------ workbook (techout-2, gap-licensing-1, gap-geolocation-3)
def test_large_area_workbook_is_complete_without_a_points_sheet(built, monkeypatch, tmp_path):
    monkeypatch.setattr(rs, "XLSX_MAX_POINTS", 10)
    res = built.res["Test Field"]
    out = rs.export(built.cfg, res, out_dir=tmp_path)
    xlsx = next(out.glob("*.xlsx"))
    summ, sheets, props = _xlsx_summary(xlsx)
    assert {"Charts", "Summary", "Notes", "Sources", "Statistics", "Annual", "Trend data"} <= set(sheets)
    assert "Points" not in sheets and "points.csv" in summ["per_point_table"]
    import openpyxl
    with open(xlsx, "rb") as fh:
        wb = openpyxl.load_workbook(fh, read_only=True)
        notes = " ".join(str(r[0]) for r in wb["Notes"].iter_rows(values_only=True))
        sources = " ".join(str(c) for r in wb["Sources"].iter_rows(values_only=True) for c in r)
    assert "points.csv (400 points)" in notes
    assert "Copernicus Sentinel" in sources
    assert props.title.startswith("Testsite: Test Field") and props.creator != "openpyxl"
    with open(out / "points.csv") as fh:
        assert sum(1 for _ in fh) == 401
    assert not list(tmp_path.glob("*.partial"))


def test_a_failed_export_leaves_no_workbook(built, monkeypatch, tmp_path):
    import pandas as pd
    res = built.res["Test Field"]
    out = rs.export(built.cfg, res, out_dir=tmp_path)            # a good earlier export
    assert list(out.glob("*.xlsx"))
    real = pd.DataFrame.to_excel

    def boom(self, writer, sheet_name="Sheet1", **kw):
        if sheet_name == "Statistics":
            raise ValueError("This sheet is too large!")
        return real(self, writer, sheet_name=sheet_name, **kw)
    monkeypatch.setattr(pd.DataFrame, "to_excel", boom)
    with pytest.raises(ValueError):
        rs.export(built.cfg, res, out_dir=tmp_path)
    assert not list(tmp_path.glob("*.xlsx")) and not list(tmp_path.glob("*.partial"))


def test_statistics_count_independent_radar_measurements(built, monkeypatch):
    """gap-geolocation-3: each radar pixel fills ~7.5 map cells; the
    statistics rest on the measurements, not the copies."""
    from insar_pipeline import deliverables as dl
    monkeypatch.setattr(dl, "radar_pixel_ids", lambda cfg, r, c, shape=None: ((np.asarray(r) // 2) * 1000 + np.asarray(c) // 2,
                                                                               "2 x 2 blocks"))
    res = rs.analyse(built.cfg, "Test Field", ctx=built.ctx)
    st = res.stats["Vertical velocity"]
    assert st["cells"] == 400 and st["n"] == 100
    assert res.area_summary["independent_measurements"] == 100
    assert "radar_pixel_id" in res.points and any("independent radar measurements" in n for n in res.notes)


# ------------------------------------------------------------------ the dashboard survives a failed export (techout-2)
def test_results_screen_shows_a_failed_export_and_keeps_running(built, tmp_path, monkeypatch):
    from textual.app import App
    from insar_pipeline import results_screen as rsc
    cfg_yaml = tmp_path / "config_testsite.yaml"
    cfg_yaml.write_text(f"project_dir: {built.cfg.project_dir}\nbbox: {list(BBOX)}\nstart_date: '2016-01-01'\n"
                        "end_date: '2025-03-31'\nbackend: miaplpy\ntrack: 92\nstack_virtual_merge: false\n"
                        "mintpy_deramp: 'no'\nreport_areas: ['Full Satellite Imaging:full', 'Test Field:"
                        + _box_spec(FIELD_BOX) + "']\n")

    def fail(cfg, res, out_dir=None):
        raise ValueError("This sheet is too large!")
    monkeypatch.setattr(rs, "export", fail)
    seen = {}

    class T(App):
        def on_mount(self):
            self.push_screen(rsc.ResultsScreen(str(cfg_yaml)))

    async def go():
        app = T()
        async with app.run_test(size=(160, 50)) as pilot:
            await pilot.pause()
            scr = app.screen
            scr.res = built.res["Test Field"]
            scr.action_export()
            for _ in range(5):
                await app.workers.wait_for_complete()
                await pilot.pause()
            seen["running"] = app.is_running
            seen["last"] = scr.last_export
            seen["notes"] = [str(n.message) for n in app._notifications]
            seen["button"] = scr.query_one("#rexp").disabled
    asyncio.run(go())
    assert seen["running"] and seen["last"] is None and seen["button"] is False
    assert any("Export failed" in n and "too large" in n for n in seen["notes"])


def test_results_screen_colours_readings_from_fields():
    from insar_pipeline.results_screen import reading_style
    assert reading_style({"kind": "area", "testable": True, "signal_to_noise": 1.3}) == "bold yellow"
    assert reading_style({"kind": "area", "testable": True, "signal_to_noise": 0.4}) == "green"
    assert reading_style({"kind": "area", "testable": False}) == "dim"
    assert reading_style({"kind": "full", "gnss_anchored": False}) == "dim"


def test_a_report_from_before_the_changes_still_reads_and_prints_no_old_limit(built):
    """Backward compatibility: a client_report.json written before 5 Oct 2026
    (no testable / range95 / reading) still gives a balloon - its range from
    the old 1-sigma, its reading from `detected` - but never its 2-sigma limit
    or |z| under today's labels."""
    old = {"name": "Old Area", "kind": "area", "vertical_mm_yr": -5.67, "vertical_uncertainty_1sigma_mm_yr": 4.6,
           "total_mm": -59.9, "detection_limit_mm_yr": 15.0, "signal_to_noise": 1.4, "detected": False,
           "area_km2": 4.0, "cells": 486}
    t = _plain(ck.area_description(old, {"anchored": True, "span_years": 10.4,
                                         "gnss": {"background": {"vertical_mm_yr": -1.1}}}))
    assert "Within normal ground variation" in t and "-0.58 to +0.13 in/yr" in t
    assert "Reliably detects" not in t and "Signal ÷ threshold" not in t and "rebuild it" in t
