"""P2/3 T1 (review 8 Oct 2026, group 'digitise'): the digitised truth for
experiment A and the deramp-exclusion polygon built from it.

What is checked (PLAN.md section 3 phase 2 T1; protocol_A.yaml
truth.tahmoor_west_units, truth.end_of_panel_maxima, site.angle_of_draw;
contract C-truth):
  - review_2026-10-08/truth/tahmoor_west_gnss_units.csv is read by
    ground_truth.load_points_csv: the 17 Tahmoor West units of Report 8,
    validation role, Up/East/North with sigmas, a digitising_sigma_mm column
    and a provenance column (source PDF, page, figure, axis calibration);
  - the digitising sigma follows the protocol: the two-pass RMS, at least
    half the plot's gridline spacing, added in quadrature to the unit's own
    scatter; the two passes and Report 7 agree to a few mm;
  - the numbers printed in Report 8's text are reproduced;
  - panels.geojson: every outline has provenance and matches the void length
    and width printed in its report table; CA19 lies over LW902 (GA); the
    Picton Station dot lies where Wikipedia puts the station;
  - configs/experiments/context/appin_tahmoor_deforming.geojson holds every
    panel mined or settling in A's window plus its angle of draw, keeps the
    reference stations out, and the pipeline's own deramp-mask builder
    accepts it on the processed crop; the APPROX file is kept unchanged,
    and the A1/A3 configs name the new polygon (finishing pass, 8 Oct 2026,
    before any run).
None of these files existed before T1, so every test fails on the old tree.
No network; nothing under ~/insar_projects (INSAR_PROJECTS -> tmp_path).
"""
from __future__ import annotations

import csv
import json
import math
from datetime import date
from pathlib import Path

import numpy as np
import pytest

PKG = Path(__file__).resolve().parents[1]
TRUTH = PKG / "review_2026-10-08" / "truth"
from client_files import need as local_only                      # noqa: E402  (local-only files skip on a clone)
UNITS = TRUTH / "tahmoor_west_gnss_units.csv"
DAILY = TRUTH / "tahmoor_west_gnss_daily.csv"
CALIB = TRUTH / "tahmoor_west_gnss_calibration.json"
PANELS = TRUTH / "panels.geojson"
CTX = PKG / "configs" / "experiments" / "context"
DEFORM = CTX / "appin_tahmoor_deforming.geojson"
APPROX = CTX / "appin_tahmoor_deforming_APPROX.geojson"
BOX_A = (150.55, -34.30, 150.78, -34.14)
STATIONS = {"CA07": (150.6392, -34.2216), "PCTN": (150.6052, -34.1576), "MENA": (150.7438, -34.1261),
            "CRDX": (150.7676, -34.3248)}
CA19 = (150.691241, -34.178878)
REPORT8_UNITS = {"TW_G07", "TW_G08", "TW_G09", "TW_G10", "TW_G11", "TW_G12A", "TW_G13", "TW_G15", "TW_G16",
                 "TW_G17", "TW_G18", "TW_G19", "TW_G20", "TW_G21", "TW_G24", "TW_GSR17N", "TW_GSR17S"}


def _rows(path):
    with open(path) as f:
        return list(csv.DictReader(line for line in f if not line.startswith("#")))


@pytest.fixture(scope="module")
def points():
    from insar_pipeline import ground_truth as gt
    return gt.load_points_csv(UNITS)


@pytest.fixture(scope="module")
def rows():
    return _rows(UNITS)


def _mga(lonlat):
    from pyproj import Transformer
    t = Transformer.from_crs(4326, 28356, always_xy=True)
    p = np.asarray(lonlat, float).reshape(-1, 2)
    x, y = t.transform(p[:, 0].tolist(), p[:, 1].tolist())
    return np.c_[x, y]


# ---------------------------------------------------------------- the GNSS units
def test_units_file_loads_with_the_pipeline_reader(points):
    ids = {pt["id"] for pt, _ in points}
    assert ids == REPORT8_UNITS                                     # protocol A: 17 units
    for pt, df in points:
        assert pt["role"] == "validation" and pt["kind"] == "gnss"
        assert BOX_A[0] < pt["lon"] < BOX_A[2] and BOX_A[1] < pt["lat"] < BOX_A[3]
        assert set(df["component"]) == {"u", "e", "n"}
        assert (df["sigma_mm"] > 0).all() and df["value_mm"].notna().all()
        assert "Report 8" in pt["source"] and "page" in pt["source"]
        u = df[df["component"] == "u"]
        assert u["time"].is_monotonic_increasing and len(u) >= 20
        # values are mm since the unit's start: the first epoch is near zero
        assert abs(u["value_mm"].iloc[0]) < 15


def test_contract_columns_digitising_sigma_and_provenance(rows):
    need = {"id", "lat", "lon", "date", "up_mm", "east_mm", "north_mm", "sigma_up_mm", "digitising_sigma_mm",
            "digitising_rms_mm", "gnss_sigma_mm", "source_report", "page", "figure", "provenance"}
    assert need <= set(rows[0])
    for r in rows:
        prov = r["provenance"]
        assert "gm3.au" in prov and f"page {r['page']}" in prov and r["figure"] in prov
        assert "calibration" in prov or "fitted" in prov
        assert r["figure"].startswith("Fig. G") and r["figure"].endswith("-R21")
        date.fromisoformat(r["date"])                               # ISO dates only


def test_digitising_sigma_follows_the_protocol(rows):
    """Protocol A as amended (2026-10-08/p23-1, 8 Oct 2026, before any
    result): digitising error = hypot(two-pass RMS, half the plotted line
    width in mm, axis-calibration residual) per point, added in quadrature
    to the unit's own scatter. The rule as first fixed (the two-pass RMS, at
    least the plot's half gridline) stays in the *_as_first_fixed columns."""
    cal = json.loads(CALIB.read_text())
    for r in rows:
        c = cal[r["id"]]["panels"]["height"]
        half = c["pass_A"]["gridline_mm"] / 2.0
        rms = float(r["digitising_rms_mm"])
        dig = math.sqrt(rms ** 2 + float(r["half_line_width_mm"]) ** 2 + float(r["calibration_mm"]) ** 2)
        assert float(r["digitising_sigma_mm"]) == pytest.approx(dig, abs=0.011)
        assert float(r["sigma_up_mm"]) == pytest.approx(math.hypot(dig, float(r["gnss_sigma_mm"])), abs=0.011)
        assert float(r["half_gridline_mm"]) == half
        assert float(r["sigma_up_as_first_fixed_mm"]) == pytest.approx(
            math.hypot(max(rms, half), float(r["gnss_sigma_mm"])), abs=0.06)


def test_two_passes_and_two_reports_agree():
    cal = json.loads(CALIB.read_text())
    checks = json.loads((TRUTH / "checks" / "gnss_checks.json").read_text())
    assert set(cal) == REPORT8_UNITS
    for uid, c in cal.items():
        assert c["x_axis"]["rms_days"] < 0.2
        for name, p in c["panels"].items():
            assert p["pass_A"]["rms_at_ticks_mm"] < 0.5                        # vector calibration
            assert p["pass_B"]["rms_at_gridlines_mm"] < 1.0                     # raster calibration
            assert p["two_pass"]["rms_A_minus_B_mm"] < 2.0, (uid, name)
            # MSEC's 7-day running average is the centred mean of the plotted daily values
            assert p["avg7_vs_centred_daily_mean"]["rms_mm"] < 2.0, (uid, name)
        for comp, k in checks["units"][uid].items():
            assert k["report8_minus_report7_mm"]["n"] >= 10
            assert k["report8_minus_report7_mm"]["rms"] < 2.0, (uid, comp)


def test_report_text_numbers_are_reproduced():
    """Report 8 text: Sites 24, 7, 8 and 9 recorded an additional 16, 5, 2 and
    4 mm of vertical subsidence between 1 January and 30 June 2023; Site 24
    'approximately 660 mm' (Report 8) and 'approximately 640 mm' (Report 7,
    plots to 15 Nov 2022). The increments are a few mm, the size of each
    unit's daily scatter (2-4 mm), so they agree within 5 mm, not exactly. The
    plots show 7 and 17 mm more than the text's approximate totals. This is
    not rounding: Report 8's 660 is about Report 7's 640 plus its own 16 mm,
    and leaves out the ~9 mm the plot shows for mid-Nov to Dec 2022."""
    pr = json.loads((TRUTH / "checks" / "gnss_checks.json").read_text())["printed_numbers"]
    inc = {c["unit"]: c for c in pr if "additional" in c["report8_text"]}
    assert set(inc) == {"TW_G24", "TW_G07", "TW_G08", "TW_G09"}
    for u in ("TW_G07", "TW_G08", "TW_G09"):
        assert abs(inc[u]["difference_mm"]) <= 5.0, u
    assert abs(inc["TW_G24"]["difference_mm"]) <= 6.0
    tot = {c["report8_text"][:8]: c for c in pr if "approximately" in c["report8_text"]}
    r8 = next(c for c in pr if "660" in c["report8_text"])
    r7 = next(c for c in pr if "640" in c["report8_text"])
    assert tot and 0 <= r8["difference_mm"] <= 20 and 0 <= r7["difference_mm"] <= 10


def test_site24_event_over_lw_w4():
    """Site 24 (above LW W4) sat still before 16 May 2022 and sank about
    600 mm by the end of W4 (13 Sep 2022), moving east and south."""
    r = [x for x in _rows(UNITS) if x["id"] == "TW_G24"]
    by = {x["date"]: x for x in r}
    assert abs(float(by["2022-05-01"]["up_mm"])) < 15
    assert -680 < float(by["2022-09-15"]["up_mm"]) < -560
    assert float(by["2022-09-15"]["east_mm"]) > 80 and float(by["2022-09-15"]["north_mm"]) < -100
    # the position is over LW W4, about 200 m from its commencing (north) end
    from shapely.geometry import Point, Polygon
    fc = json.loads(PANELS.read_text())
    w4 = Polygon(next(f for f in fc["features"] if f["properties"]["id"] == "TW_W4")["properties"]["mga94_corners"])
    p = Point(_mga((float(r[0]["lon"]), float(r[0]["lat"])))[0])
    assert w4.contains(p)
    north_end = max(c[1] for c in w4.exterior.coords)
    assert 120 < north_end - p.y < 300


def test_daily_record_has_every_plotted_value():
    d = _rows(DAILY)
    assert {r["id"] for r in d} == REPORT8_UNITS
    assert {r["component"] for r in d} == {"u", "e", "n"}
    n_daily = sum(1 for r in d if r["daily_mm"])
    assert n_daily > 17 * 300


# ---------------------------------------------------------------- the panels
def test_panels_match_the_printed_dimensions_and_carry_provenance():
    local_only(TRUTH / "sources", "the source documents (300+ MB) stay on the developer's machine")
    fc = json.loads(PANELS.read_text())
    ids = {f["properties"]["id"] for f in fc["features"]}
    for need in ("TW_W1", "TW_W2", "TW_W3", "TW_W4", "TN_LW31", "TN_LW32", "TS_S1A", "AP_LW901", "AP_LW902",
                 "AP_LW903", "AP_LW904", "AP_LW905", "AP_LW707B", "AP_LW708A", "AP_LW708B", "AP_LW709"):
        assert need in ids
    checks = json.loads((TRUTH / "panel_checks.json").read_text())["panels"]
    for f in fc["features"]:
        p = f["properties"]
        pv = p["provenance"]
        assert pv["source_pdf"].endswith(".pdf") and pv["page"] >= 1 and pv["drawing"].startswith("MSEC")
        assert (TRUTH / "sources" / pv["source_pdf"]).exists()
        assert p["position_sigma_m"] > 0 and p["draw_distance_m"] > 150
        c = checks[p["id"]]
        if "length_diff_m" in c:
            assert abs(c["length_diff_m"]) <= 10, p["id"]
        if "width_diff_m" in c:
            assert abs(c["width_diff_m"]) <= 3, p["id"]
        ring = f["geometry"]["coordinates"][0]
        assert ring[0] == ring[-1] and len(ring) == 5
        for lon, lat in ring:
            assert BOX_A[0] - 0.02 < lon < BOX_A[2] + 0.02 and BOX_A[1] - 0.02 < lat < BOX_A[3] + 0.02


def test_georeference_checked_against_independent_positions():
    ck = json.loads((TRUTH / "panel_checks.json").read_text())
    absolute = {a["what"][:12]: a for a in ck["absolute"]}
    ca19 = next(a for a in ck["absolute"] if "CA19" in a["what"])
    assert ca19["inside_LW902"] is True
    picton = next(a for a in ck["absolute"] if "Picton" in a["what"])
    assert picton["offset_m"] < 30
    assert absolute
    # the same mined panels drawn on drawings whose grids were read separately agree to metres
    cross = ck["cross_drawing_edges"]
    for drawing, panel in (("MSEC1394-01", "AP_LW708A"), ("MSEC1250-01", "AP_LW707B"), ("MSEC1250-01", "AP_LW902"),
                           ("MSEC1334-01", "AP_LW903"), ("MSEC1176-01", "AP_LW901"), ("MSEC1263-01", "TN_LW31")):
        assert cross[drawing][panel]["median_offset_m"] < 3.0, (drawing, panel)


def test_lw_w4_is_the_as_mined_outline():
    """LW W4 stopped after 721 m (Report 7); the planned 1004 m outline is kept
    but is not in the window set."""
    fc = {f["properties"]["id"]: f["properties"] for f in json.loads(PANELS.read_text())["features"]}
    assert abs(fc["TW_W4"]["measured"]["length_m"] - 721) < 10 and fc["TW_W4"]["in_window"]
    assert abs(fc["TW_W4_planned"]["measured"]["length_m"] - 1004) < 5 and not fc["TW_W4_planned"]["in_window"]


# ---------------------------------------------------------------- the deforming polygon
def _deform_shapes():
    from shapely.geometry import shape
    gj = json.loads(DEFORM.read_text())
    return gj, [shape(f["geometry"]) for f in gj["features"]]


def test_deforming_polygon_holds_every_window_panel_and_its_draw():
    from shapely.geometry import Point, Polygon
    from shapely.ops import transform, unary_union
    gj, shapes = _deform_shapes()
    assert "panels.geojson" in gj["_source"] and "angle of draw" in gj["_source"]
    assert all(f["properties"]["approximate"] is False for f in gj["features"])
    to_mga = lambda g: transform(lambda x, y, z=None: tuple(_mga(np.c_[x, y]).T), g)   # noqa: E731
    area = unary_union([to_mga(s) for s in shapes])
    fc = json.loads(PANELS.read_text())
    n = 0
    for f in fc["features"]:
        p = f["properties"]
        if not p["in_window"]:
            continue
        n += 1
        reach = Polygon(p["mga94_corners"]).buffer(p["draw_distance_m"] - 1.0)   # 1 m: coordinate rounding
        assert area.contains(reach), p["id"]
    assert n >= 16
    assert any(s.contains(Point(*CA19)) for s in shapes)
    for k, (lo, la) in STATIONS.items():
        assert not any(s.contains(Point(lo, la)) for s in shapes), k


def test_sources_are_recorded_with_hashes():
    local_only(TRUTH / "sources", "the source documents (300+ MB) stay on the developer's machine")
    src = json.loads((TRUTH / "sources.json").read_text())
    for name in ("tahmoor_west_report_7.pdf", "tahmoor_west_report_8.pdf",
                 "tahmoor_west_EP_app_A_subsidence_predictions.pdf", "appin_MSEC1050_LW902_EoP_report_RevA.pdf"):
        assert name in src and len(src[name]["sha256"]) == 64 and src[name]["url"].startswith("https://")
    for name, rec in src.items():
        assert (TRUTH / "sources" / name).stat().st_size == rec["bytes"]


# ---------------------------------------------------------------- the survey lines
def test_survey_lines_carry_the_end_of_panel_maxima_over_their_panels():
    """protocol A truth.end_of_panel_maxima: LW902 263 mm (ARTC line), LW903 432
    and LW904 649 mm (Menangle Road), LW708 990 mm (ARTC) and 813 mm (M31
    West). Each line must run over (or, for LW902's ARTC line, beside) its
    panel; the maxima carry their report page."""
    local_only(TRUTH / "sources", "the source documents (300+ MB) stay on the developer's machine")
    from shapely.geometry import LineString, Polygon
    fc = json.loads((TRUTH / "survey_lines.geojson").read_text())
    panels = {f["properties"]["id"]: Polygon(f["properties"]["mga94_corners"])
              for f in json.loads(PANELS.read_text())["features"]}
    lines = {f["properties"]["id"]: f["properties"] for f in fc["features"]}
    got = {(m["panel"], m["max_mm"]) for p in lines.values() for m in p["end_of_panel_maxima"]}
    assert got == {("LW902", 263), ("LW903", 432), ("LW904", 649), ("LW708", 990), ("LW708", 813)}
    over = {"AP_ARTC_A9": ["AP_LW901"], "AP_MENANGLE_A9": ["AP_LW903", "AP_LW904"],
            "AP_ARTC_A7": ["AP_LW708B"], "AP_M31W_A7": ["AP_LW708B"]}
    for lid, pids in over.items():
        p = lines[lid]
        L = LineString(p["mga94_coords"])
        for pid in pids:
            assert L.intersection(panels[pid]).length > 300, (lid, pid)
        assert p["pegs_used"] > 50 and p["median_peg_offset_m"] < 10
        assert (TRUTH / "sources" / p["provenance"]["source_pdf"]).exists()
        for m in p["end_of_panel_maxima"]:
            assert (TRUTH / "sources" / m["source_pdf"]).exists() and m["page"] >= 1
    # LW902's ARTC line runs over LW901, beside LW902 (why its 263 mm is a lower bound of the panel peak)
    assert LineString(lines["AP_ARTC_A9"]["mga94_coords"]).distance(panels["AP_LW902"]) < 100


def test_menangle_line_is_named_as_the_eop_table_places_it():
    """LW904 EoP Table 3-1: the Menangle Road line 'crosses above the western
    end of Longwall 903 and above the eastern end of Longwall 904'. The
    digitised line must do that, and its name must say so (it said 'west ends
    of LW903, LW904' before the review)."""
    from shapely.geometry import LineString, Polygon
    fc = json.loads((TRUTH / "survey_lines.geojson").read_text())
    p = next(f["properties"] for f in fc["features"] if f["properties"]["id"] == "AP_MENANGLE_A9")
    assert "western end of LW903" in p["name"] and "eastern end of LW904" in p["name"]
    assert "west ends" not in p["name"]
    panels = {f["properties"]["id"]: Polygon(f["properties"]["mga94_corners"])
              for f in json.loads(PANELS.read_text())["features"]}
    L = LineString(p["mga94_coords"])
    for pid, side in (("AP_LW903", "west"), ("AP_LW904", "east")):
        x0, _, x1, _ = panels[pid].bounds
        cx = L.intersection(panels[pid]).centroid.x
        frac = (cx - x0) / (x1 - x0)
        assert (frac < 0.5) if side == "west" else (frac > 0.5), (pid, round(frac, 2))


def test_independent_review_checks():
    """Review of T1: (1) South32's APNSVY00237 (MGA2020, another surveyor's
    drawing, fitted to its own grid and checked against its corner table)
    puts LW709 and LW905 within 3 m of the MSEC-derived outlines, which also
    fixes Appin east-west (CA19 inside LW902 fixes north-south only);
    (2) the mining bars of all 17 GNSS figures, through pass A's date
    calibration, fall on the published W1/W3/W4 dates (absolute date check)."""
    rc = json.loads((TRUTH / "checks" / "review_checks.json").read_text())
    s32 = rc["south32_appin_outlines"]
    assert s32["grid"]["spacing_m_at_1_6000"] == pytest.approx(1000.0, abs=1.0)
    assert s32["grid"]["step_e_m"] == 1000.0 and s32["grid"]["step_n_m"] == 1000.0
    assert s32["corner_table_check"]["corners"] == 24 and s32["corner_table_check"]["max_m"] < 4.0
    for pid in ("AP_LW709", "AP_LW905"):
        rec = s32["panels"][pid]
        assert rec["south32_vertices"] >= 4
        assert min(rec["msec_as_mga94"]["max_m"], rec["msec_as_mga2020"]["max_m"]) < 3.0, pid
        assert rec["msec_as_mga94"]["max_m"] < 5.0, pid           # the datum T1 used stays inside 10 m sigma
    bars = rc["gnss_date_axis_vs_mining_bars"]
    assert len(bars["figures"]) == 17 and bars["checked_ends"] >= 30
    assert bars["max_abs_days"] < 0.5
