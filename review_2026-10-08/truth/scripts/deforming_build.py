"""
T1 (8 Oct 2026): the replacement deramp-exclusion polygon for experiment A,
built from the digitised panel outlines (panels.geojson) by the APPROX
file's rule, with the hand-drawing allowance replaced by the outlines'
stated uncertainty:

  every panel mined, or still settling, in 2017-11-01 .. 2022-12-31
  (panels.geojson in_window: mined in the window, or completed in the six
  months before it), buffered by
      its draw distance  = tan(26.5 deg) x the largest depth of cover over
                           the panel (protocol A angle_of_draw; NSW Southern
                           Coalfield convention, 20 mm limit), plus
      a margin           = 50 m (at least 2 x the outline's position sigma,
                           10-14 m, rounded up)
  then merged; one feature per connected area.

The buffer is round (a distance from the panel edge). Writes
configs/experiments/context/appin_tahmoor_deforming.geojson (the APPROX file
is kept) and review_2026-10-08/truth/checks/deforming_checks.json: areas,
the stations inside or outside, and a comparison with the APPROX envelopes
(which panel + draw areas the APPROX file misses, and how much ground it
excluded beyond them).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from shapely.geometry import Point, Polygon, mapping, shape
from shapely.ops import transform, unary_union

import drawgeo as dg

HERE = Path(__file__).resolve().parent
TRUTH = HERE.parent
PKG = TRUTH.parents[1]
OUT = PKG / "configs" / "experiments" / "context" / "appin_tahmoor_deforming.geojson"
APPROX = PKG / "configs" / "experiments" / "context" / "appin_tahmoor_deforming_APPROX.geojson"
MARGIN_M = 50.0
STATIONS = {"CA19": (150.691241, -34.178878), "CA07": (150.6392, -34.2216), "PCTN": (150.6052, -34.1576),
            "MENA": (150.7438, -34.1261), "CRDX": (150.7676, -34.3248)}
BOX_A = (150.55, -34.30, 150.78, -34.14)


def to_mga(geom):
    return transform(lambda x, y, z=None: tuple(dg.lonlat_to_mga(np.c_[x, y], 28356).T), geom)


def to_ll(geom):
    return transform(lambda x, y, z=None: tuple(dg.mga_to_lonlat(np.c_[x, y], 28356).T), geom)


def build():
    fc = json.loads((TRUTH / "panels.geojson").read_text())
    used, parts = [], []
    for f in fc["features"]:
        p = f["properties"]
        if not p["in_window"]:
            continue
        poly = Polygon(p["mga94_corners"])
        d = p["draw_distance_m"] + MARGIN_M
        parts.append((p["id"], poly.buffer(d, resolution=32)))
        used.append({"id": p["id"], "name": p["name"], "mine": p["mine"], "window_status": p["window_status"],
                     "draw_distance_m": p["draw_distance_m"], "buffer_m": round(d, 1),
                     "position_sigma_m": p["position_sigma_m"]})
    merged = unary_union([g for _, g in parts])
    comps = list(merged.geoms) if merged.geom_type == "MultiPolygon" else [merged]
    feats = []
    for c in sorted(comps, key=lambda g: g.centroid.x):
        ids = [i for i, g in parts if g.intersects(c)]
        mines = sorted({u["mine"] for u in used if u["id"] in ids})
        cs = c.simplify(1.0, preserve_topology=True)
        ll = to_ll(cs)
        feats.append({"type": "Feature",
                      "properties": {"name": f"{' + '.join(mines)}: panels {', '.join(i.split('_', 1)[1] for i in ids)}"
                                             " + angle of draw + 50 m",
                                     "approximate": False, "panels": ids, "area_km2": round(c.area / 1e6, 2)},
                      "geometry": json.loads(json.dumps(mapping(ll)))})
    for f in feats:                                      # round coordinates (0.1 m is 1e-6 deg)
        f["geometry"]["coordinates"] = [[[round(x, 6), round(y, 6)] for x, y in ring]
                                        for ring in f["geometry"]["coordinates"]]
    out = {"type": "FeatureCollection",
           "_source": (
               "Deforming area for experiment A's deramp exclusion, built 8 Oct 2026 (PLAN T1) from the digitised "
               "panel outlines review_2026-10-08/truth/panels.geojson (the mines' CAD drawings on their MGA grids; "
               "outline position sigma 10-14 m). Rule of the APPROX file, with its 0.5-1 km hand-drawing allowance "
               "replaced by the outlines' uncertainty: every longwall panel mined, or completed in the six months "
               "before and so still settling, in 2017-11-01 .. 2022-12-31 inside the A box (Appin LW901-LW905, "
               "LW707A/B, LW708A/B, LW709; Tahmoor West LW W1-W4 as mined; Tahmoor North LW30-LW32; Tahmoor South "
               "LW S1A), buffered by its angle of draw (tan 26.5 deg x the largest depth of cover over the panel, "
               "204-356 m) plus 50 m. Built by review_2026-10-08/truth/scripts/deforming_build.py. The run adds "
               "deramp_buffer_m on top. The APPROX file (appin_tahmoor_deforming_APPROX.geojson) is kept for the "
               "record. Since 8 Oct 2026, before any run, the A1, A2 and A3 configs name this file (protocol A "
               "amendment 2026-10-08/a-deramp-polygon); replacing it once a run has started is a new run."),
           "rule": {"angle_of_draw_deg": 26.5, "draw_distance": "tan(26.5 deg) x max depth of cover over the panel",
                    "margin_m": MARGIN_M, "panels_included": "panels.geojson in_window true",
                    "window": ["2017-11-01", "2022-12-31"], "settling": "completed on or after 2017-05-01"},
           "panels_used": used,
           "features": feats}
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    return out, merged, parts


def checks(out, merged, parts):
    res = {"polygon": str(OUT.relative_to(PKG)), "area_km2": round(merged.area / 1e6, 2),
           "features": [{"name": f["properties"]["name"], "area_km2": f["properties"]["area_km2"]} for f in out["features"]]}
    st = {}
    for k, ll in STATIONS.items():
        pm = Point(dg.lonlat_to_mga(ll, 28356)[0])
        st[k] = {"inside": bool(merged.contains(pm)), "distance_m": round(float(merged.exterior.distance(pm)
                                                                                 if merged.geom_type == "Polygon"
                                                                                 else merged.boundary.distance(pm)), 0)}
    res["stations"] = st
    approx = unary_union([to_mga(shape(f["geometry"])) for f in json.loads(APPROX.read_text())["features"]])
    res["approx_area_km2"] = round(approx.area / 1e6, 2)
    miss = merged.difference(approx)
    res["new_outside_approx_km2"] = round(miss.area / 1e6, 3)
    res["approx_outside_new_km2"] = round(approx.difference(merged).area / 1e6, 2)
    res["panels_not_fully_in_approx"] = []
    for i, g in parts:
        a = g.difference(approx).area
        if a > 1.0:
            res["panels_not_fully_in_approx"].append({"panel": i, "panel_plus_draw_outside_approx_km2": round(a / 1e6, 3)})
    box = to_mga(Polygon([(BOX_A[0], BOX_A[1]), (BOX_A[2], BOX_A[1]), (BOX_A[2], BOX_A[3]), (BOX_A[0], BOX_A[3])]))
    res["box_area_km2"] = round(box.area / 1e6, 1)
    res["share_of_box"] = round(merged.intersection(box).area / box.area, 3)
    res["box_more_than_2km_from_polygon_km2"] = round(box.difference(merged.buffer(2000)).area / 1e6, 1)
    res["approx_box_more_than_2km_km2"] = round(box.difference(approx.buffer(2000)).area / 1e6, 1)
    (TRUTH / "checks" / "deforming_checks.json").write_text(json.dumps(res, indent=1) + "\n")
    return res


if __name__ == "__main__":
    out, merged, parts = build()
    print(json.dumps(checks(out, merged, parts), indent=1))
