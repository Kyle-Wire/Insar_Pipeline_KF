"""
T1 (8 Oct 2026): the survey lines that carry the end-of-panel maxima in
protocol A (truth.end_of_panel_maxima, row appin_peak_ratio), read from the
MSEC drawings' own survey pegs.

  LW902  ARTC (Main Southern Railway) line   263 mm  MSEC1050-01
  LW903  Menangle Road line                  432 mm  MSEC1285-01
  LW904  Menangle Road line                  649 mm  MSEC1285-01
  (the road runs diagonally: it crosses the western end of LW903 and the
  eastern end of LW904, as the LW904 EoP Table 3-1 says)
  LW708  ARTC rail line                      990 mm  MSEC1250-01
  LW708  M31 West line                       813 mm  MSEC1250-01

How: the pegs are filled round dots (triangle fans) of the line's colour.
Every dot within a few points of a hand-placed guide (page coordinates read
off zoomed renders with the dots overlaid; it only says which dots belong
to the line, it never becomes the geometry) is taken, put in order along the
guide, sent through the drawing's georeference, and the line is the median
peg position in 20 m steps along it. Lines are LineStrings in WGS 84 with
their provenance; position sigma 10 m (pegs drawn on CAD; the 100 m
sampling buffer of the protocol is ten times that).
"""
from __future__ import annotations

import json

import numpy as np
from scipy.spatial import cKDTree
from shapely.geometry import LineString, Point

import drawgeo as dg
import panels_build as pb
import pdfvec
from sources import SOURCES

TRUTH = pb.TRUTH

LINES = [
    dict(id="AP_ARTC_A9", name="ARTC (Main Southern Railway) monitoring line, Appin Area 9", drawing="MSEC1050-01",
         colour="#7e00ff", dot=(0.4, 1.6), half_width_pt=5.0,
         guide=[(285, 25), (287, 140), (288, 200), (286, 260), (283, 300), (278, 340), (272, 380), (265, 420),
                (255, 445), (248, 465), (243, 500), (247, 540), (251, 580), (250, 610), (242, 640), (232, 670),
                (222, 700), (219, 740), (228, 780), (238, 815)],
         maxima=[dict(panel="LW902", max_mm=263, source_pdf="appin_LW902_EoP_summary.pdf", page=9,
                      text="ARTC monitoring line ... Maximum vertical subsidence (263 mm)")]),
    dict(id="AP_MENANGLE_A9",
         name="Menangle Road monitoring line, Appin Area 9 (western end of LW903, eastern end of LW904; "
              "LW904 EoP Table 3-1)",
         drawing="MSEC1285-01", colour="#7e00ff", dot=(0.4, 1.6), half_width_pt=5.0,
         guide=[(147, 264), (170, 259), (195, 255), (215, 248), (240, 234), (260, 225), (280, 215), (305, 205),
                (330, 195), (350, 188), (370, 182), (400, 178), (430, 170), (460, 160), (500, 148), (540, 138),
                (570, 133), (605, 124), (625, 118)],
         maxima=[dict(panel="LW903", max_mm=432, source_pdf="appin_LW903_EoP_report.pdf", page=9,
                      text="Menangle Road ... Maximum measured incremental vertical subsidence (432 mm)"),
                 dict(panel="LW904", max_mm=649, source_pdf="appin_LW904_EoP_summary.pdf", page=10,
                      text="Menangle Road ... Maximum measured incremental vertical subsidence (649 mm)")]),
    dict(id="AP_ARTC_A7", name="ARTC rail monitoring line, Appin Area 7 (LW708B)", drawing="MSEC1250-01",
         colour="#ff0000", dot=(0.4, 2.5), half_width_pt=5.0,
         guide=[(568, 35), (566, 80), (556, 120), (549, 140), (553, 180), (565, 215), (575, 245), (577, 275),
                (574, 300), (565, 320)],
         maxima=[dict(panel="LW708", max_mm=990, source_pdf="appin_LW708_EoP_summary.pdf", page=10,
                      text="The maximum measured incremental vertical subsidence along the ARTC Line of 990 mm")]),
    dict(id="AP_M31W_A7", name="M31 West monitoring line, Appin Area 7 (LW708B)", drawing="MSEC1250-01",
         colour="#ff0000", dot=(0.4, 2.5), half_width_pt=5.0,
         guide=[(596, 35), (595, 100), (594, 150), (588, 190), (582, 225), (577, 260), (575, 300), (572, 340),
                (567, 380)],
         maxima=[dict(panel="LW708", max_mm=813, source_pdf="appin_LW708_EoP_summary.pdf", page=9,
                      text="The maximum measured incremental vertical subsidence along the M31 East and M31 West "
                           "Lines of 654 mm and 813 mm")]),
]
STEP_M = 20.0


def dots(paths, colour, dmin, dmax):
    """Centres of filled round dots drawn as fans of triangles."""
    tris = [np.asarray(s) for p in paths if pdfvec.colour_name(p.fill) == colour for s in p.subpaths if len(s) == 3]
    if not tris:
        return np.zeros((0, 2))
    cen = np.array([t.mean(axis=0) for t in tris])
    parent = list(range(len(cen)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    for a, b in cKDTree(cen).query_pairs(0.9):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    groups = {}
    for i in range(len(cen)):
        groups.setdefault(find(i), []).append(i)
    out = []
    for idx in groups.values():
        pts = np.vstack([tris[i] for i in idx])
        e = np.ptp(pts, axis=0)
        if dmin <= e.max() <= dmax and abs(e[0] - e[1]) < 0.35 * e.max():
            out.append(pts.mean(axis=0))
    return np.array(out)


def georef(name):
    if name in pb.GRIDDED:
        spec = pb.GRIDDED[name]
        paths = pb.load_paths(spec)
        return spec, paths, pb.gridded_georef(spec, paths)
    spec = pb.AFFINE[name]
    paths = pb.load_paths(spec)
    M = np.array(json.loads((TRUTH / "georef_drawings.json").read_text())[name]["georef"]["matrix"])
    return spec, paths, (lambda p: np.c_[np.asarray(p, float).reshape(-1, 2),
                                         np.ones(len(np.asarray(p).reshape(-1, 2)))] @ M.T)


def build():
    feats, report = [], {}
    cache = {}
    for L in LINES:
        if L["drawing"] not in cache:
            cache[L["drawing"]] = georef(L["drawing"])
        spec, paths, g = cache[L["drawing"]]
        d = dots(paths, L["colour"], *L["dot"])
        guide = LineString(L["guide"])
        sel = [p for p in d if guide.distance(Point(p)) <= L["half_width_pt"]]
        sel = np.array(sorted(sel, key=lambda p: guide.project(Point(p))))
        m = g(sel)
        # distance along the line in metres: project the MGA pegs onto the guide in MGA
        gm = LineString(g(np.array(L["guide"], float)))
        s = np.array([gm.project(Point(q)) for q in m])
        verts = []
        for a in np.arange(s.min(), s.max() + STEP_M, STEP_M):
            k = (s >= a) & (s < a + STEP_M)
            if k.any():
                verts.append(np.median(m[k], axis=0))
        line_m = LineString(verts).simplify(2.0)
        resid = float(np.median([line_m.distance(Point(q)) for q in m]))
        ll = dg.mga_to_lonlat(np.array(line_m.coords), 28356)
        maxima = [dict(mx, source_url=SOURCES.get(mx["source_pdf"])) for mx in L["maxima"]]
        props = {"id": L["id"], "name": L["name"], "end_of_panel_maxima": maxima,
                 "pegs_used": int(len(sel)), "length_m": round(line_m.length, 0),
                 "median_peg_offset_m": round(resid, 1), "position_sigma_m": 10.0,
                 "mga94_coords": [[round(float(x), 1), round(float(y), 1)] for x, y in line_m.coords],
                 "provenance": {"source_pdf": spec["pdf"], "source_url": SOURCES.get(spec["pdf"]),
                                "page": spec["page"], "drawing": L["drawing"], "drawing_title": spec["title"],
                                "drawing_date": spec["date"],
                                "method": f"survey pegs (filled dots {L['colour']}) within {L['half_width_pt']} pt of a "
                                          "hand-placed guide, ordered along it, median position per 20 m",
                                "georeference": "MGA grid lines" if L["drawing"] in pb.GRIDDED else
                                                "affine fit to panels on gridded drawings (georef_drawings.json)"}}
        feats.append({"type": "Feature", "properties": props,
                      "geometry": {"type": "LineString",
                                   "coordinates": [[round(float(x), 7), round(float(y), 7)] for x, y in ll]}})
        report[L["id"]] = {k: props[k] for k in ("pegs_used", "length_m", "median_peg_offset_m")}
    fc = {"type": "FeatureCollection",
          "_source": ("Survey lines of the Appin end-of-panel maxima in protocol A (appin_peak_ratio), digitised "
                      "8 Oct 2026 (PLAN T1) from the survey pegs on the MSEC drawings; see "
                      "review_2026-10-08/truth/README.md. WGS 84 lon/lat; mga94_coords in EPSG:28356. The maxima "
                      "are survey-LINE maxima (lower bounds of the panel peak), as the protocol says."),
          "features": feats}
    (TRUTH / "survey_lines.geojson").write_text(json.dumps(fc, indent=1) + "\n")
    return report


if __name__ == "__main__":
    print(json.dumps(build(), indent=1))
