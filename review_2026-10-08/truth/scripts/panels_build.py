"""
T1 (8 Oct 2026): longwall panel outlines for experiment A, read from the
mines' own CAD drawings and georeferenced on each drawing's MGA grid.

Writes, in review_2026-10-08/truth/:
  panels.geojson          one Polygon per panel (WGS 84 lon/lat), with the
                          mining dates, depth of cover, draw distance,
                          whether it was mined or settling in A's window,
                          and its provenance (PDF, URL, page, drawing,
                          grid lines used, fit residual, checks)
  georef_drawings.json    every drawing's georeference (grid lines and the
                          values read for them, fit, residuals) and the
                          affine transfers for drawings without a grid
  panel_checks.json       each outline against the void length and width
                          printed in the reports' tables, and the same panel
                          from two independent drawings

How (and why it can be trusted):
  - The drawings are CAD exports: grid lines and panel outlines are vector
    paths. The grid labels are glyph outlines, not text, so the value of
    each grid line was read by eye from a zoomed render (the line used is
    named below by its page coordinate). Every outline's length and width
    are then checked against the void dimensions printed in the report
    tables, which tests the scale; the same panel read from two drawings
    with independently read grids tests the label reading (a misread label
    moves a panel by a whole grid step, 500-2000 m).
  - Drawings without a grid (MSEC1285-01 for LW904/LW905, MSEC1263-01 for
    the as-mined LW W4) are tied by an affine fit to the corners of panels
    already georeferenced elsewhere; the fit residual is reported.
  - Datum: MSEC drawings say "Grid to MGA co-ordinates" without naming
    GDA94 or GDA2020; they are converted as MGA94 (EPSG:28356). The two
    differ by about 1.5-1.8 m here, inside the stated uncertainty.

Position uncertainty (1 sigma) of an outline: 10 m for a gridded drawing
(grid fit < 0.2 m; line placement and the planned-versus-mined edge, e.g.
LW904 finished 2 m short of its approved end and the installation heading
adds about 9 m), and sqrt(10^2 + rms^2) for an affine transfer.
"""
from __future__ import annotations

import json
import os
import tempfile
from datetime import date
from pathlib import Path

import numpy as np

import drawgeo as dg
import pdfvec
from sources import SOURCES

HERE = Path(__file__).resolve().parent
TRUTH = HERE.parent
SRC = TRUTH / "sources"
WORK = Path(os.environ.get("T1_WORKDIR", Path(tempfile.gettempdir()) / "t1_digitise"))   # SVG/PNG renders
WORK.mkdir(parents=True, exist_ok=True)
BASE_SIGMA_M = 10.0
WINDOW = (date(2017, 11, 1), date(2022, 12, 31))       # protocol A window (A1)
SETTLING_FROM = date(2017, 5, 1)                       # completed <= 6 months before the window: still settling

# ---------------------------------------------------------------- drawings with a grid
# east / north: (page axis, [(page coordinate of the grid line, MGA value read by eye), ...])
GRIDDED = {
    "MSEC1019-01": dict(pdf="tahmoor_west_EP_app_A_subsidence_predictions.pdf", page=200,
                        title="Western Domain Extraction Plan LW W1-W2, General Layout", date="2019-06-07",
                        scale="1:20000", report="MSEC1019 Rev B (Jun 2019)",
                        east=("x", [(96.6, 277000), (167.46, 277500), (238.32, 278000), (309.18, 278500),
                                    (380.04, 279000), (450.9, 279500), (521.76, 280000)]),
                        north=("y", [(86.22, 6217000), (157.08, 6216500), (227.94, 6216000), (298.8, 6215500),
                                     (369.66, 6215000), (440.52, 6214500), (511.44, 6214000), (582.3, 6213500),
                                     (653.16, 6213000), (724.02, 6212500), (794.88, 6212000)])),
    "MSEC1112-01": dict(pdf="tahmoor_west_EP_app_A_subsidence_predictions_1.pdf", page=180,
                        title="Western Domain Extraction Plan LW W3-W4, General Layout", date="2021-03-12",
                        scale="1:20000", report="MSEC1112 Rev A (Mar 2021)",
                        east=("x", [(96.6, 277000), (167.46, 277500), (238.32, 278000), (309.18, 278500),
                                    (380.04, 279000), (450.9, 279500), (521.76, 280000)]),
                        north=("y", [(86.22, 6217000), (157.08, 6216500), (227.94, 6216000), (298.8, 6215500),
                                     (369.66, 6215000), (440.52, 6214500), (511.44, 6214000), (582.3, 6213500),
                                     (653.16, 6213000), (724.02, 6212500), (794.88, 6212000)])),
    "MSEC1192-01": dict(pdf="tahmoor_MSEC1192_south_EP_S1A-S6A_RevA.pdf", page=245,
                        title="Tahmoor South Project Extraction Plan LW S1A to LW S6A, General Layout",
                        date="2022-05-12", scale="as shown", report="MSEC1192 Rev A (May 2022)",
                        east=("x", [(178.08, 276000), (348.18, 278000), (518.22, 280000)]),
                        north=("y", [(184.26, 6208000), (354.36, 6206000), (524.4, 6204000), (694.5, 6202000)])),
    "MSEC1050-01": dict(pdf="appin_MSEC1050_LW902_EoP_report_RevA.pdf", page=32,
                        title="Illawarra Coal, Appin Colliery Area 9, General Layout & Monitoring (LW902)",
                        date="2019-07-16", scale="1:12500", report="MSEC1050 Rev A (LW902 End of Panel)",
                        east=("y", [(146.42, 289000), (373.22, 288000), (599.96, 287000)]),
                        north=("x", [(98.8, 6216000), (325.6, 6215000), (552.34, 6214000)])),
    "MSEC1176-01": dict(pdf="appin_LW903_EoP_monitoring_review.pdf", page=37,
                        title="Illawarra Metallurgical Coal, Appin Colliery LW903, Ground Monitoring Lines",
                        date="2021-07-21", scale="1:12500", report="MSEC1176 (LW903 End of Panel monitoring review)",
                        east=("y", [(156.08, 289000), (382.88, 288000), (609.68, 287000)]),
                        north=("x", [(112.84, 6216000), (339.64, 6215000), (566.38, 6214000)])),
    "MSEC1334-01": dict(pdf="appin_LW905_EoP_monitoring_review.pdf", page=41,
                        title="Illawarra Metallurgical Coal, Appin Colliery LW905, Ground Monitoring Lines",
                        date="2023-06-15", scale="1:12500 (printed reduced)",
                        report="MSEC1334 (LW905 End of Panel monitoring review)",
                        east=("x", [(291.62, 287000), (490.1, 288000), (688.52, 289000)]),
                        north=("y", [(147.46, 6217000), (345.88, 6216000), (544.36, 6215000)])),
    "MSEC975-01": dict(pdf="appin_LW707_monitoring_report.pdf", page=51,
                       title="Illawarra Coal, LW707 End of Panel, General Layout and Monitoring Lines",
                       date="2018-10-11", scale="1:20000", report="MSEC975 (LW707 subsidence monitoring report)",
                       east=("y", [(121.04, 293000), (262.76, 292000), (404.54, 291000), (546.26, 290000),
                                   (687.98, 289000)]),
                       north=("x", [(130.6, 6218000), (272.32, 6217000), (414.04, 6216000)])),
    "MSEC1250-01": dict(pdf="appin_LW708_subsidence_assessment.pdf", page=44,
                        title="Illawarra Metallurgical Coal, LW708 End of Panel, General Layout and Monitoring Lines",
                        date="2022-04-28", scale="1:20000", report="MSEC1250 (LW708 subsidence assessment)",
                        east=("x", [(117.36, 288000), (259.14, 289000), (400.86, 290000), (542.58, 291000),
                                    (684.36, 292000)]),
                        north=("y", [(179.5, 6218000), (321.22, 6217000), (462.94, 6216000)])),
    "MSEC1394-01": dict(pdf="appin_LW709_subsidence_assessment.pdf", page=46,
                        title="Appin Colliery, End of Panel LW709, Ground Monitoring Lines",
                        date="2024-01-23", scale="as shown", report="MSEC1394 (LW709 subsidence assessment)",
                        east=("x", [(46.58, 288000), (250.64, 289000), (454.76, 290000), (658.88, 291000)]),
                        north=("y", [(54.22, 6219000), (258.34, 6218000), (462.46, 6217000)])),
}

# drawings tied by an affine fit to panels georeferenced on gridded drawings
AFFINE = {
    "MSEC1263-01": dict(pdf="tahmoor_west_report_8.pdf", page=88,
                        title="Tahmoor North Western Domain Monitoring LW W4, General Layout (Rev 21)",
                        date="2023-08-23", scale="1:10000 (printed reduced)",
                        report="MSEC1263 Rev 21, in Tahmoor West Report 8",
                        # grey outlines from west to east: W1, W2, W3, W4
                        ties=[("W1", "#dcdcdc", "west_rank", 0), ("W2", "#dcdcdc", "west_rank", 1),
                              ("W3", "#dcdcdc", "west_rank", 2)]),
    "MSEC1285-01": dict(pdf="appin_LW904_EoP_monitoring_review.pdf", page=39,
                        title="Illawarra Metallurgical Coal, Appin Colliery LW904, Ground Monitoring Lines",
                        date="2022-12-15", scale="as shown", report="MSEC1285 (LW904 End of Panel monitoring review)",
                        # grey outlines from south to north: LW901, LW902, LW903, LW904, LW905
                        ties=[("LW901", "#808080", "south_rank", 0), ("LW902", "#808080", "south_rank", 1),
                              ("LW903", "#808080", "south_rank", 2)]),
}

# ---------------------------------------------------------------- panels
# near: rough MGA position of the panel's centre (to pick the right outline on the drawing)
# dims: (overall void length, width) printed in the source table, for the check
# depth: (min, max) depth of cover, m; mined: (start, end) ISO dates
TW_DATES = ("Tahmoor West Reports 1-8 (gm3.au); W1 start: Tahmoor 2019 Annual Review gives both 15 and 17 Nov 2019, "
            "Report 1 and the GNSS plots start on 15 Nov 2019")
PANELS = [
    # Tahmoor West (Western Domain), Bulli seam, mined north to south
    dict(id="TW_W1", name="LW W1", mine="Tahmoor", domain="Western Domain", drawing="MSEC1019-01", style="#ffbf00",
         near=(277908, 6215503), dims=(1875, 283), dims_src="MSEC1019 Table 1.1",
         check=[("MSEC1112-01", "#dcdcdc")],
         depth=(455, 535), depth_src="MSEC1019 section 2.1 (depth of cover above LW W1-W2)",
         mined=("2019-11-15", "2020-11-06"), mined_src=TW_DATES),
    dict(id="TW_W2", name="LW W2", mine="Tahmoor", domain="Western Domain", drawing="MSEC1019-01", style="#ffbf00",
         near=(278226, 6215592), dims=(1685, 283), dims_src="MSEC1019 Table 1.1",
         depth=(455, 535), depth_src="MSEC1019 section 2.1",
         mined=("2020-12-07", "2021-06-17"), mined_src=TW_DATES),
    dict(id="TW_W3", name="LW W3", mine="Tahmoor", domain="Western Domain", drawing="MSEC1112-01", style="#ffbf00",
         near=(278549, 6215591), dims=(1552, 283), dims_src="MSEC1112 Table 1.1",
         depth=(470, 550), depth_src="MSEC1112 section 2.1 (depth of cover above LW W3-W4)",
         mined=("2021-09-13", "2022-03-21"), mined_src=TW_DATES),
    dict(id="TW_W4", name="LW W4 (as mined, 721 m of extraction)", mine="Tahmoor", domain="Western Domain",
         drawing="MSEC1263-01", style="#ffbf00", near=None, dims=(721, 285),
         dims_src="Report 7 (MSEC1263 Rev 20 summary table): length of extraction of LW W4 721 m; "
                  "width 285 m (MSEC1112 Table 1.1)",
         depth=(470, 550), depth_src="MSEC1112 section 2.1",
         mined=("2022-05-16", "2022-09-13"), mined_src=TW_DATES,
         note="LW W4 was planned at 1004 m (MSEC1112-01) and stopped after 721 m of extraction; this outline is the "
              "as-mined one from the monitoring report drawing of Aug 2023. The planned outline is kept as "
              "TW_W4_planned (not used for the deforming area)."),
    dict(id="TW_W4_planned", name="LW W4 (planned, not fully mined)", mine="Tahmoor", domain="Western Domain",
         drawing="MSEC1112-01", style="#ffbf00", near=(278889, 6215332), dims=(1004, 285),
         dims_src="MSEC1112 Table 1.1", depth=(470, 550), depth_src="MSEC1112 section 2.1",
         mined=None, mined_src="planned only; see TW_W4", status="planned outline, superseded"),
    # Tahmoor North (mined up-dip, south-east to north-west)
    dict(id="TN_LW30", name="LW30", mine="Tahmoor", domain="Tahmoor North", drawing="MSEC1112-01", style="#dcdcdc",
         near=(278693, 6213058), dims=(2322, 283),
         dims_src="Tahmoor 2019 Annual Review, App. 14 (GeoTerra TA35-R1A) Table 1: length 2322 m; width 283 m",
         depth=(425, 490), depth_src="GeoTerra TA35-R1A Table 1 (Tahmoor 2019 Annual Review)",
         mined=("2016-06-20", "2017-05-15"), mined_src="GeoTerra TA35-R1A Table 1 (Tahmoor 2019 Annual Review)"),
    dict(id="TN_LW31", name="LW31", mine="Tahmoor", domain="Tahmoor North", drawing="MSEC1112-01", style="#dcdcdc",
         near=(278957, 6213242), dims=(2450, 283), dims_src="GeoTerra TA35-R1A Table 1: length 2450 m",
         depth=(450, 500), depth_src="GeoTerra TA35-R1A Table 1",
         mined=("2017-06-28", "2018-08-17"), mined_src="GeoTerra TA35-R1A Table 1 (Tahmoor 2019 Annual Review)"),
    dict(id="TN_LW32", name="LW32", mine="Tahmoor", domain="Tahmoor North", drawing="MSEC1112-01", style="#dcdcdc",
         near=(279201, 6213455), dims=(2380, 283),
         dims_src="Tahmoor 2019 Annual Review, GeoTerra TA35-R1A section 3.1: '283m wide ... approximately 2,380m long' "
                  "(its Table 1 says 2500 m)",
         depth=(450, 500), depth_src="GeoTerra TA35-R1A Table 1",
         mined=("2018-10-29", "2019-09-26"),
         mined_src="GeoTerra TA35-R1A Table 1; the Annual Review text gives completion on 22 and on 25 Sep 2019"),
    # Tahmoor South
    dict(id="TS_S1A", name="LW S1A", mine="Tahmoor", domain="Tahmoor South", drawing="MSEC1192-01", style="#ffbf00",
         near=None, pick="most_north_east", dims=(None, 285), dims_src="MSEC1192 Table 1.1: void widths 283-285 m",
         depth=(365, 410), depth_src="MSEC1192 section 1.2: 365-410 m over LW S1A-S6A, deepest over S1A",
         mined=("2022-10-18", "2023-07-04"), mined_src="review_2026-10-08/longwall/notes.md 3.2 (Tahmoor South reports)",
         note="Planned outline (May 2022); S1A started on 18 Oct 2022, inside A's window."),
    # Appin Area 9 (mined west to east)
    dict(id="AP_LW901", name="LW901", mine="Appin", domain="Area 9", drawing="MSEC1050-01", style="#808080",
         near=None, pick="Area9_901", dims=(2028, 305), dims_src="MSEC927/1050 Table 1.2 (as extracted)",
         check=[("MSEC1176-01", "#808080")],
         depth=(485, 540), depth_src="MSEC927 (LW901 End of Panel) section 1",
         mined=("2016-01-19", "2017-09-08"), mined_src="MSEC927 Table 1.1 (LW901 End of Panel report)"),
    dict(id="AP_LW902", name="LW902", mine="Appin", domain="Area 9", drawing="MSEC1050-01", style="#ffbf00",
         near=None, dims=(2153, 305), dims_src="MSEC1050 Table 1.2 (as extracted)",
         check=[("MSEC1176-01", "#808080")],
         depth=(510, 590), depth_src="MSEC1050 section 1",
         mined=("2018-05-12", "2019-04-03"), mined_src="LW902 End of Panel Summary Report; drawing MSEC1050-01 labels"),
    dict(id="AP_LW903", name="LW903", mine="Appin", domain="Area 9", drawing="MSEC1176-01", style="#ffbf00",
         near=None, dims=(2297, 305), dims_src="MSEC1176/1285/1334 Table 1.2 (as extracted)",
         depth=(510, 630), depth_src="MSEC1176 section 1",
         mined=("2019-11-01", "2021-04-07"), mined_src="LW903 End of Panel Report"),
    dict(id="AP_LW904", name="LW904", mine="Appin", domain="Area 9", drawing="MSEC1285-01", style="#ffbf00",
         near=None, dims=(2038, 325), dims_src="MSEC1285 Table 1.2 (as extracted)",
         depth=(535, 630), depth_src="MSEC1285 section 1",
         mined=("2021-05-20", "2022-08-09"), mined_src="LW904 End of Panel Summary Report"),
    dict(id="AP_LW905", name="LW905", mine="Appin", domain="Area 9", drawing="MSEC1334-01", style="#ffbf00",
         near=None, dims=(858, 325), dims_src="MSEC1334 Table 1.2 (as extracted)",
         check=[("MSEC1285-01", "#808080")],
         depth=(585, 715), depth_src="MSEC1334 section 1",
         mined=("2022-09-25", "2023-02-28"), mined_src="review_2026-10-08/longwall/notes.md 3.1; drawing MSEC1334-01 labels"),
    # Appin Area 7 (mined west to east)
    dict(id="AP_LW707A", name="LW707A", mine="Appin", domain="Area 7", drawing="MSEC975-01", style="#ffbf00",
         near=None, pick="south_west", dims=(1035, 324), dims_src="MSEC1394 Table 1.2 (as extracted)",
         depth=(475, 600), depth_src="MSEC975 section 1 (LW707A-B)",
         mined=("2016-01-07", "2016-08-18"), mined_src="drawing MSEC975-01 labels (start 7-Jan-2016, finish 18-Aug-2016)"),
    dict(id="AP_LW707B", name="LW707B", mine="Appin", domain="Area 7", drawing="MSEC975-01", style="#ffbf00",
         near=None, pick="north_east", dims=(2070, 324), dims_src="MSEC1394 Table 1.2 (as extracted)",
         depth=(475, 600), depth_src="MSEC975 section 1 (LW707A-B)",
         mined=("2016-09-26", "2018-06-19"), mined_src="drawing MSEC975-01 labels (start 26-Sep-2016, finish 19-Jun-2018)"),
    dict(id="AP_LW708A", name="LW708A", mine="Appin", domain="Area 7", drawing="MSEC1250-01", style="#ffbf00",
         near=None, pick="south_west", dims=(1200, 324), dims_src="MSEC1394 Table 1.2 (as extracted)",
         depth=(515, 620), depth_src="MSEC1250 section 1 (LW708A-B)",
         mined=("2019-04-02", "2022-01-03"),
         mined_src="LW708 End of Panel Summary Report (LW708A+B together: 2 Apr 2019 - 3 Jan 2022)"),
    dict(id="AP_LW708B", name="LW708B", mine="Appin", domain="Area 7", drawing="MSEC1250-01", style="#ffbf00",
         near=None, pick="north_east", dims=(2260, 324), dims_src="MSEC1394 Table 1.2 (as extracted)",
         depth=(515, 620), depth_src="MSEC1250 section 1 (LW708A-B)",
         mined=("2019-04-02", "2022-01-03"),
         mined_src="LW708 End of Panel Summary Report (LW708A+B together: 2 Apr 2019 - 3 Jan 2022)"),
    dict(id="AP_LW709", name="LW709", mine="Appin", domain="Area 7", drawing="MSEC1394-01", style="#ffbf00",
         near=None, dims=(2615, 324), dims_src="MSEC1394 Table 1.2 (as extracted)",
         depth=(530, 615), depth_src="MSEC1394 section 1",
         mined=("2022-02-22", "2023-10-08"), mined_src="drawing MSEC1394-01 labels (start 22-Feb-2022, end 8-Oct-2023)"),
]


def load_paths(spec):
    return pdfvec.read_paths(pdfvec.page_svg(SRC / spec["pdf"], spec["page"], WORK))


def gridded_georef(spec, paths):
    xs, ys = dg.long_lines(paths, 150)
    have = {"x": xs, "y": ys}
    fits = {}
    for comp in ("east", "north"):
        axis, pairs = spec[comp]
        for pv, _ in pairs:
            if not len(have[axis]) or np.min(np.abs(have[axis] - pv)) > 0.05:
                raise ValueError(f"{spec['pdf']} p{spec['page']}: no grid line at {axis}={pv}")
        pv = np.array([p for p, _ in pairs]); val = np.array([v for _, v in pairs], float)
        b, a = np.polyfit(pv, val, 1)
        res = val - (a + b * pv)
        fits[comp] = dg.AxisFit(float(a), float(b), float(np.sqrt(np.mean(res ** 2))), len(pv),
                                [(float(p), float(v)) for p, v in pairs])
    east_axis = spec["east"][0]
    return dg.GridGeoref(fits["east"], fits["north"], east_axis)


def candidates(paths, style, min_extent_pt=40.0):
    """Closed four-corner outlines of the style that are large enough to be a
    panel (buildings, symbols and legend boxes are smaller)."""
    return [q for q in dg.dedupe(dg.closed_polygons(paths, stroke=style))
            if len(q) == 4 and float(np.ptp(q, axis=0).max()) > min_extent_pt]


def ordered(q_mga):
    """Corners in a fixed order: by bearing from the centre (north-up maps)."""
    c = q_mga.mean(axis=0)
    ang = np.arctan2(q_mga[:, 0] - c[0], q_mga[:, 1] - c[1])
    return q_mga[np.argsort(ang)]


def pick(cands_mga, rule):
    cen = np.array([q.mean(axis=0) for q in cands_mga])
    if rule is None or isinstance(rule, tuple):
        if rule is None:
            if len(cands_mga) != 1:
                raise ValueError(f"{len(cands_mga)} outlines of this style; give 'near'")
            return 0
        return int(np.argmin(np.hypot(*(cen - np.asarray(rule)).T)))
    if rule == "most_north_east":
        return int(np.argmax(cen[:, 0] + cen[:, 1]))
    if rule == "north_east":
        return int(np.argmax(cen[:, 0] + cen[:, 1]))
    if rule == "south_west":
        return int(np.argmin(cen[:, 0] + cen[:, 1]))
    if rule == "Area9_901":                     # LW901: the southernmost of the grey Area 9 panels
        return int(np.argmin(cen[:, 1]))
    raise ValueError(rule)


def main():
    georefs, drawing_paths, records = {}, {}, {}
    for name, spec in GRIDDED.items():
        paths = load_paths(spec)
        g = gridded_georef(spec, paths)
        georefs[name], drawing_paths[name] = g, paths
        records[name] = {"pdf": spec["pdf"], "url": SOURCES.get(spec["pdf"]), "page": spec["page"],
                         "title": spec["title"], "drawing_date": spec["date"], "scale": spec["scale"],
                         "report": spec["report"], "datum": "MGA zone 56, converted as GDA94 (EPSG:28356)",
                         "labels": "grid labels are glyph outlines; each line's value was read by eye from a zoomed "
                                   "render (page coordinate -> value listed)",
                         "georef": g.record(),
                         "grid_lines": {"east": g.east.lines, "north": g.north.lines}}

    outlines = {}            # id -> (MGA corners, drawing, sigma)

    def outline(p, drawing, style, g, paths):
        cands = [g(q) for q in candidates(paths, style)]
        rule = p.get("pick") or (tuple(p["near"]) if p.get("near") else None)
        i = pick(cands, rule)
        return ordered(cands[i])

    # 1. panels on gridded drawings
    for p in PANELS:
        if p["drawing"] in GRIDDED:
            g, paths = georefs[p["drawing"]], drawing_paths[p["drawing"]]
            outlines[p["id"]] = (outline(p, p["drawing"], p["style"], g, paths), p["drawing"], BASE_SIGMA_M)

    # 2. drawings tied by affine to panels already read
    tie_ids = {"W1": "TW_W1", "W2": "TW_W2", "W3": "TW_W3", "LW901": "AP_LW901", "LW902": "AP_LW902",
               "LW903": "AP_LW903"}
    for name, spec in AFFINE.items():
        paths = load_paths(spec)
        drawing_paths[name] = paths
        page_pts, map_pts, labels = [], [], []
        for tie, style, rank_kind, rank in spec["ties"]:
            ref = outlines[tie_ids[tie]][0]
            cands = candidates(paths, style)
            cen = np.array([q.mean(axis=0) for q in cands])
            # north-up drawings: page x grows eastwards, page y southwards
            order = np.argsort(cen[:, 0]) if rank_kind == "west_rank" else np.argsort(-cen[:, 1])
            page_pts.append(cands[order[rank]]); map_pts.append(ref)
            labels += [f"{tie} corner {k}" for k in range(4)]
        # page corners must be in the same order as the map corners: order the page quad the same way
        # (north-up drawings: page y grows southwards, so flip y for the bearing)
        pp = []
        for q in page_pts:
            c = q.mean(axis=0)
            ang = np.arctan2(q[:, 0] - c[0], -(q[:, 1] - c[1]))
            pp.append(q[np.argsort(ang)])
        A = dg.fit_affine(np.vstack(pp), np.vstack(map_pts), labels)
        georefs[name] = A
        records[name] = {"pdf": spec["pdf"], "url": SOURCES.get(spec["pdf"]), "page": spec["page"],
                         "title": spec["title"], "drawing_date": spec["date"], "scale": spec["scale"],
                         "report": spec["report"], "datum": "as the tie panels (MGA94)",
                         "labels": "no grid on this drawing: affine fit to the corners of " +
                                   ", ".join(t[0] for t in spec["ties"]) + " read from gridded drawings",
                         "georef": A.record()}
    for p in PANELS:
        if p["drawing"] in AFFINE:
            A, paths = georefs[p["drawing"]], drawing_paths[p["drawing"]]
            cands = [A(q) for q in candidates(paths, p["style"])]
            rule = p.get("pick") or (tuple(p["near"]) if p.get("near") else None)
            i = pick(cands, rule)
            outlines[p["id"]] = (ordered(cands[i]), p["drawing"], float(np.hypot(BASE_SIGMA_M, A.rms_m)))

    # 3. checks: printed dimensions, and the same panel on a second drawing
    checks = {}
    for p in PANELS:
        q, drawing, sig = outlines[p["id"]]
        m = dg.rect_metrics(q)
        L, Wd = p["dims"]
        c = {"drawing": drawing, "measured": m, "printed": {"length_m": L, "width_m": Wd, "source": p["dims_src"]}}
        if L:
            c["length_diff_m"] = round(m["length_m"] - L, 1)
        if Wd:
            c["width_diff_m"] = round(m["width_m"] - Wd, 1)
        for d2, st2 in p.get("check", []):
            g2, paths2 = georefs[d2], drawing_paths[d2]
            cands = [ordered(g2(qq)) for qq in candidates(paths2, st2)]
            j = int(np.argmin([np.hypot(*(cc.mean(axis=0) - q.mean(axis=0))) for cc in cands]))
            corner = np.hypot(*(cands[j] - q).T)
            c.setdefault("second_drawing", []).append(
                {"drawing": d2, "centre_offset_m": round(float(np.hypot(*(cands[j].mean(axis=0) - q.mean(axis=0)))), 2),
                 "corner_offsets_m": [round(float(x), 2) for x in corner]})
        checks[p["id"]] = c

    # 3b. every grey (already mined) panel edge drawn on each drawing against the outlines read from
    # the other drawings: a misread grid label would put them a whole grid step (500-2000 m) apart
    from shapely.geometry import Point, Polygon
    polys = {k: Polygon(v[0]) for k, v in outlines.items() if k != "TW_W4_planned"}
    cross = {}
    for name in list(GRIDDED) + list(AFFINE):
        g, paths = georefs[name], drawing_paths[name]
        agree = {}
        for pth in paths:
            if pdfvec.colour_name(pth.stroke) not in ("#dcdcdc", "#808080"):
                continue
            for sub in pth.subpaths:
                sub = np.asarray(sub)
                if len(sub) < 3 or len(sub) > 6 or float(np.ptp(sub, axis=0).max()) < 60:
                    continue
                m = g(sub)
                best = []
                for v in m:
                    d = {k: pl.exterior.distance(Point(v)) for k, pl in polys.items()}
                    k = min(d, key=d.get)
                    best.append((d[k], k))
                ds = np.array([b[0] for b in best])
                ks = {b[1] for b in best}
                if np.median(ds) < 5 and len(ks) == 1:
                    k = ks.pop()
                    if outlines[k][1] != name:
                        agree.setdefault(k, []).append(round(float(np.median(ds)), 2))
        cross[name] = {k: {"edges": len(v), "median_offset_m": float(np.median(v))} for k, v in sorted(agree.items())}

    # 3c. absolute checks against positions published elsewhere
    absolute = []
    ca19 = dg.lonlat_to_mga([150.691241, -34.178878], 28356)[0]
    pl902 = Polygon(outlines["AP_LW902"][0])
    absolute.append({"what": "GA CEMP GNSS site CA19 'located above longwall 902' (Garthwaite & Fuhrmann 2020, "
                             "GA record 144247), at 150.691241 E, 34.178878 S",
                     "inside_LW902": bool(pl902.contains(Point(ca19))),
                     "distance_to_LW902_edge_m": round(float(pl902.exterior.distance(Point(ca19))), 1)})
    g19 = georefs["MSEC1019-01"]
    # the blue 'Picton Station' dot on MSEC1019-01: a filled circle (#0000b8) drawn as a fan of 20
    # triangles; its centre is the mean of the triangles' vertices near the label's dot
    tri = [np.asarray(sub) for pth in drawing_paths["MSEC1019-01"] if pdfvec.colour_name(pth.fill) == "#0000b8"
           for sub in pth.subpaths if len(sub) == 3]
    near = [t for t in tri if np.all(np.hypot(t[:, 0] - 517.3, t[:, 1] - 309.9) < 3.0)]
    st = np.vstack(near).mean(axis=0)
    ll = dg.mga_to_lonlat(g19(st), 28356)[0]
    wiki = (150.612575, -34.179088)
    e1, n1 = dg.lonlat_to_mga(ll, 28356)[0]
    e2, n2 = dg.lonlat_to_mga(wiki, 28356)[0]
    absolute.append({"what": "'Picton Station' dot on MSEC1019-01 against Picton railway station on Wikipedia "
                             "(34.179088 S, 150.612575 E, https://en.wikipedia.org/wiki/Picton_railway_station)",
                     "drawing_lonlat": [round(float(ll[0]), 6), round(float(ll[1]), 6)],
                     "offset_m": round(float(np.hypot(e1 - e2, n1 - n2)), 1)})

    # 4. GeoJSON
    feats = []
    for p in PANELS:
        q, drawing, sig = outlines[p["id"]]
        spec = GRIDDED.get(drawing) or AFFINE[drawing]
        ll = dg.mga_to_lonlat(q, 28356)
        ring = [[round(float(x), 7), round(float(y), 7)] for x, y in ll] + [[round(float(ll[0, 0]), 7),
                                                                             round(float(ll[0, 1]), 7)]]
        dmax = p["depth"][1]
        mined = p.get("mined")
        if mined:
            a, b = date.fromisoformat(mined[0]), date.fromisoformat(mined[1])
            if a <= WINDOW[1] and b >= WINDOW[0]:
                status = "mined in the window"
            elif SETTLING_FROM <= b < WINDOW[0]:
                status = "settling in the window (completed within 6 months before it)"
            else:
                status = "outside the window"
        else:
            status = p.get("status", "not mined")
        in_window = status.startswith("mined in") or status.startswith("settling")
        props = {
            "id": p["id"], "name": p["name"], "mine": p["mine"], "domain": p["domain"],
            "mined_start": mined[0] if mined else None, "mined_end": mined[1] if mined else None,
            "mined_source": p["mined_src"],
            "window_status": status, "in_window": in_window,
            "depth_of_cover_m": list(p["depth"]), "depth_source": p["depth_src"],
            "draw_distance_m": round(dg.TAN_DRAW * dmax, 1),
            "draw_rule": "tan(26.5 deg) x the largest depth of cover over the panel (protocol A angle_of_draw)",
            "position_sigma_m": round(sig, 1),
            "mga94_corners": [[round(float(x), 1), round(float(y), 1)] for x, y in q],
            "measured": checks[p["id"]]["measured"],
            "provenance": {
                "source_pdf": spec["pdf"], "source_url": SOURCES.get(spec["pdf"]), "page": spec["page"],
                "drawing": drawing, "drawing_title": spec["title"], "drawing_date": spec["date"],
                "report": spec["report"],
                "outline": f"vector path, stroke {p['style']}, read from the PDF page (not rasterised)",
                "georeference": ("MGA grid lines on the drawing, values read by eye (georef_drawings.json)"
                                 if drawing in GRIDDED else
                                 "affine fit to panels read on gridded drawings (georef_drawings.json)"),
                "dimension_check": {k: checks[p["id"]][k] for k in ("length_diff_m", "width_diff_m") if k in checks[p["id"]]},
            },
        }
        if p.get("note"):
            props["note"] = p["note"]
        feats.append({"type": "Feature", "properties": props, "geometry": {"type": "Polygon", "coordinates": [ring]}})
    fc = {"type": "FeatureCollection",
          "_source": ("Longwall panel outlines for experiment A, digitised 8 Oct 2026 (PLAN T1) from the mines' CAD "
                      "drawings (vector paths) on each drawing's MGA grid; see review_2026-10-08/truth/README.md, "
                      "georef_drawings.json and panel_checks.json. Coordinates WGS 84 lon/lat (converted from MGA94 "
                      "zone 56). Outlines are the overall void including first workings, as the report tables give "
                      "them. in_window: mined in 2017-11-01 .. 2022-12-31, or completed within the 6 months before."),
          "crs_note": "EPSG:4326 lon/lat; mga94_corners in EPSG:28356",
          "features": feats}
    (TRUTH / "panels.geojson").write_text(json.dumps(fc, indent=1) + "\n")
    (TRUTH / "georef_drawings.json").write_text(json.dumps(records, indent=1) + "\n")
    (TRUTH / "panel_checks.json").write_text(json.dumps({"panels": checks, "cross_drawing_edges": cross,
                                                          "absolute": absolute}, indent=1) + "\n")
    return fc, checks, records, cross, absolute


if __name__ == "__main__":
    fc, checks, records, cross, absolute = main()
    for name, r in records.items():
        g = r["georef"]
        if g["kind"].startswith("grid"):
            print(f"{name:12s} grid  E rms {g['east']['rms_m']:.2f} m  N rms {g['north']['rms_m']:.2f} m  "
                  f"scale {g['scale_m_per_pt']:.3f} m/pt")
        else:
            print(f"{name:12s} affine rms {g['rms_m']:.2f} m over {g['points']} corners")
    for k, c in checks.items():
        m = c["measured"]
        extra = "; ".join(f"{s['drawing']} centre {s['centre_offset_m']} m, corners max {max(s['corner_offsets_m'])} m"
                          for s in c.get("second_drawing", []))
        print(f"{k:14s} {c['drawing']:12s} L {m['length_m']:7.1f} W {m['width_m']:6.1f} "
              f"dL {c.get('length_diff_m')} dW {c.get('width_diff_m')} bearing {m['bearing_deg']:.1f}  {extra}")
    for name, v in cross.items():
        print("cross", name, {k: (x["edges"], x["median_offset_m"]) for k, x in v.items()})
    for a in absolute:
        print("absolute", a)
