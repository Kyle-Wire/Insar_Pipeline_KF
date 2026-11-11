"""The app's live map shows a study's own layers (A3 of
review_2026-11-07/PLAN_oil_gas_slopes.md, viewer part): the config's
feature_layers / event_polygons / event_lines / fast_zone_geojson /
reference_stable_geojson labelled from the study's layer kinds, the wells
coloured by the report's reading of each (study_points), and the before/after
change raster when the event products made one. A default-study project lists
nothing new. On the app tests' fake project (test_fix_L9a); no network."""
import json

import numpy as np
import pytest

from insar_pipeline import app_server as app
import test_fix_L9a as l9

home, _project = l9.home, l9._project


def _fc(*feats):
    return {"type": "FeatureCollection", "features": [{"type": "Feature", "properties": p, "geometry": g}
                                                      for p, g in feats]}


def _oil_gas(home):
    cfg = _project(home)
    P = home / "demo"
    ctx = P / "context"
    ctx.mkdir(exist_ok=True)
    pt = lambda x, y: {"type": "Point", "coordinates": [x, y]}            # noqa: E731
    (ctx / "study_wells.geojson").write_text(json.dumps(_fc(
        ({"name": "W1", "status": "PA", "plug_date": "2019-05-01", "type": "oil"}, pt(-87.66, 38.35)),
        ({"name": "W2", "status": "Active", "type": "gas"}, pt(-87.65, 38.36)))))
    (ctx / "study_faults.geojson").write_text(json.dumps(_fc(
        ({"name": "F1"}, {"type": "LineString", "coordinates": [[-87.69, 38.31], [-87.61, 38.39]]}))))
    sq = lambda w, s, e, n: {"type": "Polygon", "coordinates": [[[w, s], [e, s], [e, n], [w, n], [w, s]]]}  # noqa: E731
    (ctx / "study_pads.geojson").write_text(json.dumps(_fc(({"name": "Pad A", "start": "2018-01-01", "end": "2019-01-01"},
                                                             sq(-87.67, 38.34, -87.66, 38.35)))))
    (ctx / "study_fast_zone.geojson").write_text(json.dumps(_fc(({}, sq(-87.64, 38.32, -87.63, 38.33)))))
    with open(cfg, "a") as f:
        f.write(f"study: oil_gas\nfeature_layers: ['wells:{ctx / 'study_wells.geojson'}', "
                f"'faults:{ctx / 'study_faults.geojson'}']\nevent_products: true\nevent_polygons: {ctx / 'study_pads.geojson'}\n"
                f"fast_zone_geojson: {ctx / 'study_fast_zone.geojson'}\n")
    return cfg, P


def _ids(cfg):
    return {x["id"]: x for x in app.layer_list(app._load_cfg(str(cfg)))}


def test_a_default_study_project_lists_nothing_new(home):
    cfg = _project(home)
    L = app.layer_list(app._load_cfg(str(cfg)))
    assert not [x for x in L if x["id"].startswith("study_") or x["id"] == app.EVENT_RASTER["id"]]
    assert not {"Study layers", "Study results"} & {x["group"] for x in L}
    assert app.study_layers(app._load_cfg(str(cfg))) == []


def test_study_layers_are_listed_with_their_layer_kind_labels(home):
    cfg, P = _oil_gas(home)
    ids = _ids(cfg)
    got = {k: (v["label"], v["style"], v["group"]) for k, v in ids.items() if k.startswith("study_")}
    assert got == {"study_wells": ("Wells", "study_point", "Study layers"),
                   "study_faults": ("Faults", "study_line", "Study layers"),
                   "study_pads": ("Dated pads or patterns", "study_poly", "Study layers"),
                   "study_fast_zone": ("Fast zones", "fast_zone", "Study layers")}
    assert all(ids[k]["ready"] for k in got)
    assert app.EVENT_RASTER["id"] not in ids                            # no change map made yet: not listed
    fault = app.api_layer({"config": str(cfg), "id": "study_faults"})
    assert fault["kind"] == "geojson" and fault["data"]["features"][0]["geometry"]["type"] == "LineString"
    # before the products: the wells as listed, no number yet
    w = app.api_layer({"config": str(cfg), "id": "study_wells"})
    assert w["kind"] == "study_points" and not w["from_report"]
    assert [(p["id"], p["status"], p["attributes"]["status"]) for p in w["points"]] == [
        ("W1", "no_report", "PA"), ("W2", "no_report", "Active")]


def test_wells_take_the_reports_reading_and_the_change_raster_appears(home):
    cfg, P = _oil_gas(home)
    rep = json.loads((P / "deliverables" / "client_report.json").read_text())
    base = {"layer": "wells", "layer_label": "Wells", "radius_m": 100.0, "tier": "SCREENING", "shared_with": []}
    rep["study"] = "oil_gas"
    rep["study_points"] = [
        {**base, "id": "W1", "lat": 38.35, "lon": -87.66, "attributes": {"status": "PA", "plug_date": "2019-05-01",
         "type": "oil"}, "refused": False, "detected": True, "relative_vertical_mm_yr": -41.8,
         "range95_mm_yr": [-47.0, -36.6], "shared_with": ["W2"]},
        {**base, "id": "W2", "lat": 38.36, "lon": -87.65, "attributes": {"status": "Active", "plug_date": None,
         "type": "gas"}, "refused": False, "detected": False, "relative_vertical_mm_yr": 1.2,
         "range95_mm_yr": [-3.0, 5.4], "shared_with": ["W1"]},
        {**base, "id": "W3", "lat": 38.325, "lon": -87.635, "attributes": {}, "refused": True, "tier": "REFUSED",
         "refused_reason": "inside fast zones: faster than this satellite can follow, so no number is given",
         "detected": None, "relative_vertical_mm_yr": None, "range95_mm_yr": None},
        {**base, "id": "W4", "lat": 38.37, "lon": -87.62, "attributes": {}, "refused": False, "testable": False,
         "detected": None, "not_testable_reason": "fewer than 5 reliable points measured",
         "relative_vertical_mm_yr": None, "range95_mm_yr": None}]
    (P / "deliverables" / "client_report.json").write_text(json.dumps(rep))
    w = app.api_layer({"config": str(cfg), "id": "study_wells"})
    assert w["from_report"]
    by = {p["id"]: p for p in w["points"]}
    assert {k: p["status"] for k, p in by.items()} == {"W1": "moving", "W2": "not_distinguishable",
                                                        "W3": "no_number", "W4": "not_measurable"}
    assert by["W1"]["value_in_yr"] == pytest.approx(-41.8 / 25.4, abs=1e-3)
    assert by["W1"]["range_mm_yr"] == [-47.0, -36.6] and by["W1"]["shared_with"] == ["W2"]
    assert by["W1"]["attributes"]["plug_date"] == "2019-05-01" and by["W1"]["tier"] == "SCREENING"
    assert by["W3"]["value_mm_yr"] is None and "faster than this satellite" in by["W3"]["reason"]
    assert [c["name"] for c in w["legend"]["classes"]] == [
        "On moving ground (detected)", "Not distinguishable from ordinary ground", "Not measurable here",
        "No number (too fast for this satellite)"]
    # the before/after change map, once events wrote it
    import rasterio
    from rasterio.transform import from_origin
    ev = P / "deliverables" / "events"
    ev.mkdir()
    a = np.linspace(-30, 10, 40 * 50, dtype="float32").reshape(40, 50)
    with rasterio.open(ev / "los_change_mm_20180101_20200101.tif", "w", driver="GTiff", height=40, width=50, count=1,
                       dtype="float32", crs="EPSG:4326", transform=from_origin(-87.7, 38.4, .002, .0025),
                       nodata=np.nan) as d:
        d.write(a, 1)
    ids = _ids(cfg)
    r = ids[app.EVENT_RASTER["id"]]
    assert r["label"] == "Change between two dates (screening)" and r["ready"] and r["style"] == "raster"
    im = app.api_layer({"config": str(cfg), "id": app.EVENT_RASTER["id"]})
    assert im["kind"] == "image" and im["legend"]["units"] == "mm"
    assert im["source"] == "los_change_mm_20180101_20200101.tif"


def test_the_results_viewer_lists_the_wells_with_their_readings(home):
    """11 Nov 2026: the results viewer's wells (/api/wells) are the live map's
    rows: the point layers only, each well's reading."""
    from insar_pipeline import results_viewer as rv
    cfg, P = _oil_gas(home)
    d = rv.api_wells({"config": str(cfg)})
    assert [L["id"] for L in d["layers"]] == ["study_wells"]          # faults and pads are not points
    L = d["layers"][0]
    assert L["label"] == "Wells" and not L["from_report"]
    assert [(p["id"], p["status"]) for p in L["points"]] == [("W1", "no_report"), ("W2", "no_report")]
    assert L["points"] == app.api_layer({"config": str(cfg), "id": "study_wells"})["points"]
    assert {c["name"] for c in L["legend"]} == {w for k, w, _ in app.WELL_STATUS if k == "no_report"}
    assert "/api/wells" in rv.API
