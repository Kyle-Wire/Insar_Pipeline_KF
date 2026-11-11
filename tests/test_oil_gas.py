"""The oil and gas study, end to end (review_2026-11-07/PLAN_oil_gas_slopes.md, track A).

A1 (9 Nov 2026): the study's layers reach a normal products run.
  - write_project(study="oil_gas", study_layers=...) writes each layer into
    context/ with its properties and names it in its config field (wells and
    faults in feature_layers, pads in event_polygons, the fast zone in
    fast_zone_geojson, the stable area in reference_stable_geojson), turns on
    event_products for the whole record, and keeps pads and fast zones out of
    the ramp fit; wrong geometry, unknown kinds, layers outside the box and a
    fast zone on the DISP route are refused before anything is written;
  - the products run's event write (deliverables.export_event_products)
    gives values near the wells with status, plug date and type carried, and
    profiles along the faults as well as event_lines;
  - `insar events` runs the same write from the command line.

Gate A1 (pre-registered in the plan, fixed before this file ran):
  (a) on a fixture with a planted 40 mm/yr bowl, 12 wells (6 inside, 6
      outside) and 2 dated pads, the run writes values_at.csv with status,
      plug date and type carried through;
  (b) its numbers equal the standalone events.values_at exactly;
  (c) default goldens 19/19 (run separately: review_2026-11-07/phase0/run_gate0b.sh).
No network, nothing under ~/insar_projects.
"""
from __future__ import annotations

import json
import math
import re

import h5py
import numpy as np
import pandas as pd
import pytest
import yaml

from insar_pipeline import areas, config, domains
from insar_pipeline import deliverables as dl
from insar_pipeline import events as E
from insar_pipeline.tests import test_p23_events as P

BOWL_MM_YR = 40.0                      # LOS sinking rate at the bowl's centre
BOWL_SIG_M = 300.0
INSIDE = [(60, 60), (58, 63), (63, 57), (55, 60), (60, 66), (65, 62)]      # within ~120 m of the centre
OUTSIDE = [(100, 15), (105, 30), (110, 100), (20, 20), (12, 60), (95, 108)]  # > 600 m away, coherent ground
STATUS = ["producing", "shut-in", "plugged", "producing", "injection", "plugged"]


@pytest.fixture(autouse=True)
def _clear_cache():
    dl.clear_cache()
    yield
    dl.clear_cache()


def make_bowl(tmp_path):
    """test_p23_events' product, with its series replaced by a steady bowl:
    -40 mm/yr LOS at the centre cell, Gaussian (sigma 300 m), over the
    whole record, relative to the reference pixel (5, 5)."""
    cfg, T = P.make_project(tmp_path)
    geo = cfg.project_dir / "mintpy" / "geo"
    yrs = (T["days"] - T["days"][0]) / 365.25
    shape_ = np.exp(-T["dist"] ** 2 / (2 * BOWL_SIG_M ** 2))
    los = -BOWL_MM_YR * yrs[:, None, None] * shape_[None]
    ts = (los - los[:, 5, 5][:, None, None]) / 1000.0
    with h5py.File(geo / "geo_timeseries_ERA5_demErr.h5", "r+") as f:
        f["timeseries"][...] = ts.astype(np.float32)
    with h5py.File(geo / "geo_velocity.h5", "r+") as f:
        f["velocity"][...] = ((ts[-1] - ts[0]) / (yrs[-1] - yrs[0])).astype(np.float32)
    T["ts_mm"] = ts * 1000.0
    return cfg, T


def _point(T, r, c, props):
    lon, lat = P._lonlat(T, r, c)
    return {"type": "Feature", "properties": props, "geometry": {"type": "Point", "coordinates": [lon, lat]}}


def wells_fc(T):
    feats = []
    for k, (r, c) in enumerate(INSIDE + OUTSIDE):
        props = {"name": f"W-{k + 1:02d}", "status": STATUS[k % len(STATUS)],
                 "plug_date": "2019-06-01" if STATUS[k % len(STATUS)] == "plugged" else "",
                 "type": "oil" if k % 2 else "gas", "operator_note": "ignored column"}
        feats.append(_point(T, r, c, props))
    return P._fc(*feats)


def pads_fc(T):
    return P._fc(P._square(T, 55, 65, 55, 65, {"name": "Pad A", "start": "2018-06-01", "end": "2019-06-01"}),
                 P._square(T, 95, 108, 95, 110, {"name": "Pad B", "start": "2018-03-01", "end": "2020-01-01"}))


def faults_fc(T):
    a, b = P._lonlat(T, 60, 20), P._lonlat(T, 60, 100)
    return P._fc({"type": "Feature", "properties": {"name": "Fault 1"},
                  "geometry": {"type": "LineString", "coordinates": [list(a), list(b)]}})


def oil_cfg(cfg, tmp_path, T, lines=True):
    ctx = tmp_path / "layers"
    ctx.mkdir(exist_ok=True)
    w = P._write(ctx / "wells.geojson", wells_fc(T))
    pads = P._write(ctx / "pads.geojson", pads_fc(T))
    fl = [f"wells:{w}"]
    if lines:
        fl.append(f"faults:{P._write(ctx / 'faults.geojson', faults_fc(T))}")
    cfg.study = "oil_gas"
    cfg.feature_layers = tuple(fl)
    cfg.event_polygons = str(pads)
    cfg.event_products = True
    return cfg


# ------------------------------------------------------------------ write_project
AOI = (38.30, -87.70, 38.34, -87.64)


def _ll_point(lat, lon, **props):
    return {"type": "Feature", "properties": props, "geometry": {"type": "Point", "coordinates": [lon, lat]}}


def _ll_square(lat0, lon0, d, **props):
    ring = [[lon0, lat0], [lon0 + d, lat0], [lon0 + d, lat0 + d], [lon0, lat0 + d], [lon0, lat0]]
    return {"type": "Feature", "properties": props, "geometry": {"type": "Polygon", "coordinates": [ring]}}


def _layers():
    return {
        "wells": P._fc(_ll_point(38.32, -87.67, name="W1", status="producing", plug_date="", type="oil"),
                       _ll_point(38.31, -87.66, name="W2", status="plugged", plug_date="2019-01-01", type="gas"),
                       _ll_point(45.0, -100.0, name="far away")),
        "pads": P._fc(_ll_square(38.315, -87.675, 0.004, name="Pad A", start="2019-01-01", end="2020-01-01")),
        "faults": P._fc({"type": "Feature", "properties": {"name": "F1"},
                         "geometry": {"type": "LineString", "coordinates": [[-87.69, 38.31], [-87.65, 38.33]]}}),
        "stable_area": P._fc(_ll_square(38.335, -87.645, 0.003, name="quarry floor")),
    }


def _kw(tmp_path):
    return dict(track=19, direction="ascending", report_areas=[("All", "full")], disp_frame="04861",
                projects_root=tmp_path / "p", configs_dir=tmp_path / "c")


def test_write_project_puts_the_oil_gas_layers_in_their_fields(tmp_path):
    aoi = areas.box(*AOI, "field")
    out = areas.write_project("Field Test", aoi, "disp", "2020-01-01", "2024-12-31", study="oil_gas",
                              study_layers=_layers(), **_kw(tmp_path))
    raw = yaml.safe_load(out.read_text())
    cfg = config.load_config(str(out))
    kinds = dict(domains.split_feature_layer(s) for s in cfg.feature_layers)
    assert set(kinds) == {"wells", "faults"}
    assert raw["event_products"] is True and cfg.event_polygons and cfg.reference_stable_geojson
    wells = json.loads(open(kinds["wells"]).read())
    # features are kept whole with their properties; the well outside the box is dropped
    assert [f["properties"]["name"] for f in wells["features"]] == ["W1", "W2"]
    assert wells["features"][1]["properties"]["plug_date"] == "2019-01-01"
    pads = json.loads(open(cfg.event_polygons).read())
    assert pads["features"][0]["properties"]["start"] == "2019-01-01"
    # the pad is kept out of the ramp fit (a plane through the bowl would take part of it away);
    # the stable area and the wells are not
    assert raw["mintpy_deramp"] == "linear" and cfg.deramp_exclude_geojson
    # reference_mode is never set for the user (domains.NEVER_PRESET)
    assert "reference_mode" not in raw
    # the products run reads the same layers back
    st = E.settings(cfg)
    assert [k for k, _ in st["points"]] == ["wells"] and [k for k, _ in st["study_lines"]] == ["faults"]


def test_write_project_refuses_bad_study_layers_before_writing(tmp_path):
    aoi = areas.box(*AOI, "field")
    kw = _kw(tmp_path)
    bad = {
        "a point that is a polygon": ({"wells": P._fc(_ll_square(38.31, -87.68, 0.002, name="x"))}, "not a point"),
        "an unknown kind": ({"mines": P._fc(_ll_point(38.32, -87.67))}, "no layer kind 'mines'"),
        "nothing inside the box": ({"wells": P._fc(_ll_point(45.0, -100.0))}, "no feature lies inside"),
        "not a collection": ({"wells": {"type": "Point", "coordinates": [-87.67, 38.32]}}, "FeatureCollection"),
        "a fast zone on DISP": ({"fast_zone": P._fc(_ll_square(38.31, -87.68, 0.002))}, "fast zones"),
    }
    for what, (layers, msg) in bad.items():
        with pytest.raises(ValueError, match=msg):
            areas.write_project("Bad Test", aoi, "disp", "2020-01-01", "2024-12-31", study="oil_gas",
                                study_layers=layers, **kw)
        assert not (tmp_path / "c").exists() or not list((tmp_path / "c").glob("*.yaml")), what
        ctx = next((tmp_path / "p").glob("*/context"), None)
        assert ctx is None or not list(ctx.glob("study_*.geojson")), what


def test_a_fast_zone_on_the_phase_linked_route_is_kept_to_nearest_pairs(tmp_path):
    aoi = areas.box(*AOI, "field")
    kw = dict(_kw(tmp_path), swaths=[2])
    kw.pop("disp_frame")
    layers = {"fast_zone": P._fc(_ll_square(38.31, -87.68, 0.003, name="steam pattern"))}
    out = areas.write_project("Steam Test", aoi, "miaplpy", "2020-01-01", "2024-12-31", study="oil_gas",
                              study_layers=layers, **kw)
    cfg = config.load_config(str(out))                       # aliasing's fast-zone checks pass on these defaults
    assert cfg.fast_zone_geojson and json.loads(open(cfg.fast_zone_geojson).read())["features"]
    assert cfg.mintpy_deramp == "linear" and cfg.deramp_exclude_geojson     # kept out of the ramp fit


def test_the_default_study_writes_what_it_always_wrote(tmp_path):
    aoi = areas.box(*AOI, "site")
    out = areas.write_project("Plain Test", aoi, "disp", "2020-01-01", "2024-12-31", **_kw(tmp_path))
    raw = yaml.safe_load(out.read_text())
    assert "feature_layers" not in raw and "event_products" not in raw and raw["mintpy_deramp"] == "no"


# ------------------------------------------------------------------ the products run's event write
def test_gate_a1_values_near_wells_carry_their_columns_and_equal_the_standalone(tmp_path):
    cfg, T = make_bowl(tmp_path)
    cfg = oil_cfg(cfg, tmp_path, T)
    man = dl.export_event_products(cfg, cfg.project_dir / "deliverables")
    out = cfg.project_dir / "deliverables" / E.EVENTS_DIR
    assert man["status"] == "written", man
    va = pd.read_csv(out / "values_at.csv", keep_default_na=False)
    # (a) 12 wells, each with its own words carried through (and nothing else of theirs)
    assert va["id"].nunique() == 12 and set(va["layer"]) == {"wells"}
    first = va.drop_duplicates("id").set_index("id")
    assert first.loc["W-03", "attr_status"] == "plugged" and first.loc["W-03", "attr_plug_date"] == "2019-06-01"
    assert first.loc["W-02", "attr_type"] == "oil" and first.loc["W-01", "attr_plug_date"] == ""
    assert "attr_operator_note" not in va.columns
    # (b) the numbers are the standalone values_at's, exactly
    pts = E.study_points(E.settings(cfg)["points"])
    alone = E.values_at(cfg, pts, attributes=("status", "plug_date", "type"))
    got = pd.read_csv(out / "values_at.csv")
    pd.testing.assert_series_equal(got["value_mm"], alone["value_mm"].astype(float), check_names=False)
    assert list(got["id"]) == list(alone["id"])
    # the bowl is there: the centre well sank about 40 mm/yr over the record, an outside well hardly moved
    yrs = (T["days"][-1] - T["days"][0]) / 365.25
    w1 = alone[alone["id"] == "W-01"].dropna(subset=["value_mm"])
    w7 = alone[alone["id"] == "W-07"].dropna(subset=["value_mm"])
    assert (w1["value_mm"].iloc[-1] - w1["value_mm"].iloc[0]) / yrs == pytest.approx(-BOWL_MM_YR, rel=0.1)
    assert abs(w7["value_mm"].iloc[-1] - w7["value_mm"].iloc[0]) / yrs < 2.0
    # the pads and the fault profile come out of the same write
    pm = pd.read_csv(out / "panel_max.csv")
    assert set(pm["panel"]) == {"Pad A", "Pad B"}
    prof = pd.read_csv(out / "profiles.csv")
    assert set(prof["layer"]) == {"faults"} and set(prof["line"]) == {"Fault 1"}
    rec = json.loads((out / "events.json").read_text())
    assert [s["kind"] for s in rec["settings"]["study_layers"]] == ["wells", "faults"]
    assert all(s["sha256"] for s in rec["settings"]["study_layers"])


def test_event_lines_and_fault_lines_are_profiled_side_by_side(tmp_path):
    cfg, T = make_bowl(tmp_path)
    cfg = oil_cfg(cfg, tmp_path, T)
    a, b = P._lonlat(T, 20, 60), P._lonlat(T, 100, 60)
    cfg.event_lines = str(P._write(tmp_path / "road.geojson", P._fc(
        {"type": "Feature", "properties": {"name": "Road"},
         "geometry": {"type": "LineString", "coordinates": [list(a), list(b)]}})))
    out = tmp_path / "ev"
    man = E.write_event_products(cfg, out)
    assert man["status"] == "written", man["errors"]
    prof = pd.read_csv(out / "profiles.csv")
    assert set(zip(prof["layer"], prof["line"])) == {("event_lines", "Road"), ("faults", "Fault 1")}


def test_command_line_options_replace_the_study_layers(tmp_path):
    cfg, T = make_bowl(tmp_path)
    cfg = oil_cfg(cfg, tmp_path, T)
    other = P._write(tmp_path / "pts.geojson", P._fc(_point(T, 60, 60, {"name": "Survey 1"})))
    man = E.write_event_products(cfg, tmp_path / "ev", points=str(other), lines="")
    va = pd.read_csv(tmp_path / "ev" / "values_at.csv")
    assert set(va["id"]) == {"Survey 1"} and "layer" not in va.columns
    assert man["profiles"] is None and man["settings"]["study_layers"] is None


def test_a_default_study_config_reads_no_study_layers(tmp_path):
    cfg, T = make_bowl(tmp_path)
    cfg.event_products = True
    st = E.settings(cfg)
    assert st["points"] == [] and st["study_lines"] == []
    man = E.write_event_products(cfg, tmp_path / "ev")
    assert man["values_at"] is None and man["settings"]["study_layers"] is None


def test_study_points_name_parts_and_repeats_uniquely(tmp_path):
    fc = P._fc(_ll_point(38.3, -87.6, name="W1"), _ll_point(38.31, -87.6, name="W1"),
               {"type": "Feature", "properties": {"status": "plugged"},
                "geometry": {"type": "MultiPoint", "coordinates": [[-87.61, 38.3], [-87.62, 38.3]]}})
    pts = E.study_points([("wells", P._write(tmp_path / "w.geojson", fc))])
    assert [p["id"] for p in pts] == ["W1", "W1 (2)", "wells_3 part 1", "wells_3 part 2"]
    assert pts[2]["attributes"]["status"] == "plugged" and all(p["layer"] == "wells" for p in pts)
    assert all(math.isfinite(p["lat"]) for p in pts)


def test_insar_events_is_a_subcommand(tmp_path, monkeypatch, capsys):
    from insar_pipeline import dashboard
    assert "insar events <config>" in dashboard.USAGE
    called = {}
    monkeypatch.setattr(E, "main", lambda argv: called.setdefault("argv", argv) and 0)
    monkeypatch.setattr("sys.argv", ["insar", "events", "x.yaml", "--record"])
    with pytest.raises(SystemExit):
        dashboard.main()
    assert called["argv"] == ["x.yaml", "--record"]


# ------------------------------------------------------------------ A2: numbers per well
# Gate A2 (pre-registered in the plan, fixed before this ran), on a fixture
# with a planted 40 mm/yr bowl:
#   (a) at least 5 of the 6 inside wells read moving, at most 1 of the 6 outside
#   (b) wells 20 m apart are marked as shared
#   (c) wells inside the fast zone have no number
# The fixture is test_fix_L5's farmland (seeded noise, the unmapped FEATURE, the
# report areas) on a finer 30 m grid, so a well's 100 m disc holds ~35 cells.
A2_N = 160
A2_SCALE = 0.3                                 # 100 m cells x 0.3 = ~30 m
A2_BOWL = (80, 80)
A2_BOWL_SIG_M = 400.0
A2_INSIDE = [(80, 80), (78, 83), (83, 77), (76, 80), (80, 85), (85, 82)]          # <= ~160 m from the centre
A2_OUTSIDE = [(130, 30), (135, 130), (110, 150), (150, 90), (60, 140), (100, 10)]  # >= 1.5 km away, farmland
A2_PAIR = (140, 60)                            # two wells 20 m apart (outside the bowl)
A2_FAST = (120, 110)                           # a well inside a drawn fast zone


def make_a2(tmp, pads=True):
    """The A2 field project under tmp (config returned; nothing computed):
    test_fix_L5's farmland on a 30 m grid, the bowl, 15 wells, a fast zone and,
    with pads, two dated pads and event_products on (A3)."""
    from insar_pipeline import significance as sg
    from insar_pipeline.tests import test_fix_L5 as L5
    mp = pytest.MonkeyPatch()
    try:
        mp.setattr(L5, "NY", A2_N)
        mp.setattr(L5, "NX", A2_N)
        mp.setattr(L5, "DX", L5.DX * A2_SCALE)
        mp.setattr(L5, "DY", L5.DY * A2_SCALE)
        mp.setattr(L5, "BBOX", [L5.X0, L5.Y0 + A2_N * L5.DY, L5.X0 + A2_N * L5.DX, L5.Y0])
        cfg = L5.make_project(tmp)
        geo = cfg.mintpy_dir / "geo"
        f = geo / "geo_timeseries_ERA5_ramp_demErr.h5"
        with h5py.File(f, "r") as h:
            dates = [d.decode() for d in h["date"][:]]
        t = sg.years(dates)
        rr, cc = np.mgrid[0:A2_N, 0:A2_N]
        lat0, lon0 = L5._lalo(*A2_BOWL)
        lat, lon = L5._lalo(rr, cc)
        dist = np.hypot((lat - lat0) * 111_320.0, (lon - lon0) * 111_320.0 * math.cos(math.radians(lat0)))
        bowl = np.exp(-dist ** 2 / (2 * A2_BOWL_SIG_M ** 2))
        with h5py.File(f, "r+") as h:
            ts = h["timeseries"][...]
            ts += (-BOWL_MM_YR * t[:, None, None] * bowl[None] / 1000.0).astype(np.float32)
            h["timeseries"][...] = ts
        vel = np.polyfit(t, ts.reshape(len(t), -1), 1)[0].reshape(A2_N, A2_N).astype(np.float32)
        with h5py.File(geo / "geo_velocity.h5", "r+") as h:
            h["velocity"][...] = vel
        del ts
        lay = tmp / "layers"
        lay.mkdir()

        def pt(r, c, **props):
            la, lo = L5._lalo(r, c)
            return {"type": "Feature", "properties": props, "geometry": {"type": "Point", "coordinates": [lo, la]}}
        feats = [pt(r, c, name=f"IN-{k + 1}", status="producing", type="oil") for k, (r, c) in enumerate(A2_INSIDE)]
        feats += [pt(r, c, name=f"OUT-{k + 1}", status="plugged", plug_date="2018-01-01", type="gas")
                  for k, (r, c) in enumerate(A2_OUTSIDE)]
        la, lo = L5._lalo(*A2_PAIR)
        dlon = 20.0 / (111_320.0 * math.cos(math.radians(la)))
        feats += [{"type": "Feature", "properties": {"name": "PAIR-1"}, "geometry": {"type": "Point", "coordinates": [lo, la]}},
                  {"type": "Feature", "properties": {"name": "PAIR-2"},
                   "geometry": {"type": "Point", "coordinates": [lo + dlon, la]}}]
        feats.append(pt(*A2_FAST, name="FAST-1", status="producing"))
        wells = P._write(lay / "wells.geojson", P._fc(*feats))
        fr, fc_ = A2_FAST
        ring = [L5._lalo(fr + dr, fc_ + dc) for dr, dc in ((-6, -6), (-6, 6), (6, 6), (6, -6), (-6, -6))]
        fz = P._write(lay / "fast.geojson", P._fc({"type": "Feature", "properties": {"name": "steam"},
                                                   "geometry": {"type": "Polygon",
                                                                "coordinates": [[[x, y] for y, x in ring]]}}))
        cfg.study = "oil_gas"
        cfg.feature_layers = (f"wells:{wells}",)
        cfg.fast_zone_geojson = str(fz)
        if pads:
            r0, c0 = A2_BOWL
            pad_a = [L5._lalo(r0 + dr, c0 + dc) for dr, dc in ((-8, -8), (-8, 8), (8, 8), (8, -8), (-8, -8))]
            pad_b = [L5._lalo(130 + dr, 30 + dc) for dr, dc in ((-6, -6), (-6, 6), (6, 6), (6, -6), (-6, -6))]
            cfg.event_polygons = str(P._write(lay / "pads.geojson", P._fc(
                {"type": "Feature", "properties": {"name": "Pad A", "start": "2018-01-01", "end": "2021-01-01"},
                 "geometry": {"type": "Polygon", "coordinates": [[[x, y] for y, x in pad_a]]}},
                {"type": "Feature", "properties": {"name": "Pad B", "start": "2019-01-01", "end": "2022-01-01"},
                 "geometry": {"type": "Polygon", "coordinates": [[[x, y] for y, x in pad_b]]}})))
            cfg.event_products = True
    finally:
        mp.undo()
    return cfg


@pytest.fixture(scope="module")
def a2(tmp_path_factory):
    from insar_pipeline import client_report as cr
    cfg = make_a2(tmp_path_factory.mktemp("a2"), pads=False)
    rep, ctx, masks, outlines = cr.compute(cfg)
    return rep


def _rows(rep):
    return {r["id"]: r for r in rep["study_points"]}


def test_gate_a2_wells_on_the_bowl_read_moving_and_wells_off_it_do_not(a2):
    rows = _rows(a2)
    inside = [rows[f"IN-{k + 1}"] for k in range(6)]
    outside = [rows[f"OUT-{k + 1}"] for k in range(6)]
    moving_in = sum(bool(r["detected"]) for r in inside)
    moving_out = sum(bool(r["detected"]) for r in outside)
    print(f"A2(a): inside moving {moving_in}/6, outside moving {moving_out}/6; inside rates "
          f"{[r['relative_vertical_mm_yr'] for r in inside]}, outside {[r['relative_vertical_mm_yr'] for r in outside]}")
    assert moving_in >= 5 and moving_out <= 1
    # the figures are a report area's: a vertical relative rate with its 95% range
    for r in inside:
        assert r["relative_vertical_mm_yr"] < 0 and r["range95_mm_yr"][1] < 0
    # the columns are carried as given, never read
    assert rows["OUT-1"]["attributes"] == {"status": "plugged", "plug_date": "2018-01-01", "type": "gas"}
    assert rows["IN-1"]["attributes"]["plug_date"] is None


def test_gate_a2_wells_closer_than_the_radar_can_separate_share_one_value(a2):
    rows = _rows(a2)
    a, b = rows["PAIR-1"], rows["PAIR-2"]
    assert a["shared_with"] == ["PAIR-2"] and b["shared_with"] == ["PAIR-1"]
    assert a["relative_vertical_mm_yr"] == b["relative_vertical_mm_yr"] and a["n_cells"] == b["n_cells"]
    # wells far apart are not grouped
    assert rows["OUT-1"]["shared_with"] == [] and rows["IN-1"]["shared_with"] != ["OUT-1"]


def test_gate_a2_a_well_in_a_fast_zone_gets_no_number(a2):
    r = _rows(a2)["FAST-1"]
    assert r["refused"] is True and r["relative_vertical_mm_yr"] is None and r["range95_mm_yr"] is None
    assert r["tier"] == "REFUSED" and "no number" in r["refused_reason"]
    assert all(_rows(a2)[k]["tier"] == "SCREENING" for k in ("IN-1", "OUT-1"))


# ------------------------------------------------------------------ A3 + A4: report, KMZ, tiers, release
# Gate A3 (pre-registered): (a) every number in the oil and gas block equals
# client_report.json and the side files; (b) 0 whole-word Gate 1b mining words
# in the rendered report, the PDF's text and doc.kml of the oil_gas fixture;
# (c) each cannot and not_yet line in Limitations with its tag; (d) default
# goldens 19/19 (run_gate0b.sh, separately).
# Gate A4: (a) the oil_gas fixture is titled screening and every table carries
# its tier; (b) release refuses a crafted oil_gas report with a rising-ground
# rate as a number, a fast-zone well with a number, or "stopped moving";
# (c) default goldens 19/19.
GATE_1B = re.compile(r"\b(mine|mines|mining|mined|workings|spoil|stockpiles?|panels?|backfill|longwall|seams?)\b",
                     re.I)


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    import zipfile

    import matplotlib.text as mtext

    from insar_pipeline import client_kmz as kmz
    from insar_pipeline import client_report as cr
    tmp = tmp_path_factory.mktemp("a3")
    cfg = make_a2(tmp / "p")
    dl.export_event_products(cfg, cfg.project_dir / "deliverables")
    drawn, orig = [], mtext.Text.set_text

    def rec(self, t):
        drawn.append(str(t))
        return orig(self, t)
    mp = pytest.MonkeyPatch()
    mp.setattr(mtext.Text, "set_text", rec)
    try:
        out = tmp / "out"
        cr.build(cfg, None, out)
        kmz_path = kmz.build(cfg, out_dir=out)
    finally:
        mp.undo()
    with zipfile.ZipFile(kmz_path) as z:
        kml = z.read("doc.kml").decode()
    rep = json.loads((out / "client_report.json").read_text())
    return {"cfg": cfg, "rep": rep, "html": (out / "client_report.html").read_text(), "kml": kml,
            "drawn": "\n".join(drawn), "events": cfg.project_dir / "deliverables" / E.EVENTS_DIR}


def _text(html_):
    import html as h
    html_ = re.sub(r"<!\[CDATA\[(.*?)\]\]>", lambda m: m.group(1), html_, flags=re.S)
    return h.unescape(re.sub(r"<[^>]+>", " ", html_))


def test_gate_a3_numbers_are_the_records_and_the_side_files(built):
    from insar_pipeline import client_report as cr
    rep, page, kml = built["rep"], _text(built["html"]), built["kml"]
    assert {"study", "study_points", "study_events"} <= set(rep)
    for r in rep["study_points"]:
        reading = cr._well_reading(r)
        assert reading in page, r["id"]
        assert f"<name>{r['id']}</name>" in kml and reading in _text(kml), r["id"]
    side = {r["panel"]: r for r in json.loads((built["events"] / "panel_max.json").read_text())["panels"]}
    for p_ in rep["study_events"]["pads"]:
        assert p_["max_relative_los_mm"] == side[p_["name"]]["max_mm_relative_to_ring"]
        v = p_["max_relative_vertical_mm"]
        assert f"{abs(v) / 25.4:.2f} in {'down' if v < 0 else 'up'}" in page
        assert f"{abs(v) / 25.4:.2f} in {'down' if v < 0 else 'up'} ({v:+.1f} mm)" in _text(kml)
    # the pad on the bowl sank; the one off it barely moved
    pads = {p_["name"]: p_ for p_ in rep["study_events"]["pads"]}
    assert pads["Pad A"]["max_relative_vertical_mm"] < -50 and abs(pads["Pad B"]["max_relative_vertical_mm"]) < 20


def test_gate_a3_no_mining_words_anywhere_the_client_reads(built):
    for name in ("html", "kml", "drawn"):
        txt = _text(built[name]) if name != "drawn" else built[name]
        hits = sorted({m.group(0).lower() for m in GATE_1B.finditer(txt)})
        assert not hits, (name, hits)


def test_gate_a3_the_card_is_in_the_limitations_with_its_tags(built):
    from insar_pipeline import client_report as cr
    page = _text(built["html"])
    st = domains.study("oil_gas")
    for t, tag in st.cannot:
        assert f"This report cannot tell you {t} (reason: {cr.TAG_WORDS[tag]})" in page, t
    for t, tag in st.not_yet:
        assert f"Not yet checked: {t} (reason: {cr.TAG_WORDS[tag]})" in page, t


def test_gate_a4_titled_screening_and_every_row_carries_its_tier(built):
    page, kml = _text(built["html"]), built["kml"]
    assert "ground movement report (screening)" in page and "Screening report - oil and gas fields" in page
    assert "SCREENING" in re.search(r"<Document><name>([^<]+)</name>", kml).group(1)
    for r in built["rep"]["study_points"]:
        assert r["tier"] in ("SCREENING", "REFUSED")
    assert all(p_["tier"] == "SCREENING" for p_ in built["rep"]["study_events"]["pads"])
    # the tables print the tier on every row
    sec = built["html"][built["html"].index("wells and dated pads"):]
    sec = sec[:sec.index("</section>")]
    n_rows = len(built["rep"]["study_points"]) + len(built["rep"]["study_events"]["pads"])
    assert len(re.findall(r"<td class='small'>(screening|refused)</td>", sec)) == n_rows


def test_gate_a4_release_refuses_the_three_crafted_reports(built):
    import copy

    from insar_pipeline import release
    cfg, rep = built["cfg"], built["rep"]
    codes = lambda r: {b["code"]: b["hard"] for b in release.blockers(cfg, r)}       # noqa: E731
    clean = codes(rep)
    # a clean oil and gas report: a screening reading item, never the old hard hold
    assert clean.get("study_screening") is False and "study_not_released" not in clean
    assert not {"study_rising_number", "study_fast_zone_number", "study_stopped_moving"} & set(clean)
    rising = copy.deepcopy(rep)
    r0 = next(r for r in rising["study_points"] if r["id"] == "OUT-1")
    r0.update(detected=True, relative_vertical_mm_yr=12.0, shown_vertical_mm_yr=12.0)
    assert codes(rising).get("study_rising_number") is True
    fast = copy.deepcopy(rep)
    next(r for r in fast["study_points"] if r["id"] == "FAST-1").update(shown_vertical_mm_yr=-30.0)
    assert codes(fast).get("study_fast_zone_number") is True
    stopped = copy.deepcopy(rep)
    stopped["study_points"][0]["note"] = "the ground stopped moving after the shut-in"
    assert codes(stopped).get("study_stopped_moving") is True
    # a config edited back to the default still holds this study's report hard
    import dataclasses
    back = dataclasses.replace(cfg, study=domains.DEFAULT_STUDY)
    assert codes.__call__  # noqa: B018
    assert {b["code"]: b["hard"] for b in release.blockers(back, rep)}.get("study_not_released") is True


def test_a_rising_well_is_shown_without_a_rate(built):
    from insar_pipeline import client_report as cr
    r = {"testable": True, "detected": True, "relative_vertical_mm_yr": 15.0, "range95_mm_yr": [5.0, 25.0]}
    assert "no rate" in cr._well_reading(r) and "in/yr" not in cr._well_reading(r)


# ------------------------------------------------------------------ A2b: dense fields (gate in the plan)
def _pt(lat, lon, name):
    return {"id": name, "lat": lat, "lon": lon}


def test_gate_a2b_b_a_line_of_wells_30_m_apart_does_not_chain():
    from insar_pipeline import client_report as cr
    # cells on a 15 m grid along a line; each well sees the cells within 100 m
    n, step = 101, 30.0                                     # 3 km
    rels = []
    for k in range(n):
        centre = k * step
        cells = np.arange(int((centre - 100) // 15), int((centre + 100) // 15) + 1) + 10_000
        rels.append(np.sort(cells.astype(np.int64)))
    dlon = step / (111_320.0 * math.cos(math.radians(35.0)))
    pts = [_pt(35.0, -119.5 + k * dlon, f"W{k:03d}") for k in range(n)]
    g = cr._group_study_points(pts, rels, 100.0)
    spans = [(np.flatnonzero(g == x).max() - np.flatnonzero(g == x).min()) * step for x in np.unique(g)]
    print("A2b(b): groups", len(spans), "widest", max(spans), "m")
    assert max(spans) <= 200.0 and len(spans) > 5


def test_gate_a2b_c_the_shared_test_agrees_with_each_wells_own(tmp_path, monkeypatch):
    from insar_pipeline import client_report as cr
    from insar_pipeline.tests import test_fix_L5 as L5
    cfg = make_a2(tmp_path, pads=False)
    # 120 wells scattered over the field, on and off the bowl (seeded)
    rng = np.random.default_rng(11)
    r_, c_ = rng.uniform(30, 150, 120), rng.uniform(5, 155, 120)
    dy = L5.DY * A2_SCALE
    dx = L5.DX * A2_SCALE
    la, lo = L5.Y0 + (r_ + 0.5) * dy, L5.X0 + (c_ + 0.5) * dx
    feats = [{"type": "Feature", "properties": {"name": f"S-{k:03d}"},
              "geometry": {"type": "Point", "coordinates": [float(lo[k]), float(la[k])]}} for k in range(120)]
    cfg.feature_layers = (f"wells:{P._write(tmp_path / 'scatter.geojson', P._fc(*feats))}",)
    monkeypatch.setattr(cr, "STUDY_POINT_OWN_TEST_MAX_GROUPS", 10_000)
    own = _rows(cr.compute(cfg)[0])
    monkeypatch.setattr(cr, "STUDY_POINT_OWN_TEST_MAX_GROUPS", 0)
    shared = _rows(cr.compute(cfg)[0])
    ok = [k for k, r in own.items() if r.get("testable") and shared[k].get("testable")]
    assert len(ok) >= 100
    agree = np.mean([bool(own[k]["detected"]) == bool(shared[k]["detected"]) for k in ok])
    ratio = np.array([shared[k]["decision_threshold_mm_yr"] / own[k]["decision_threshold_mm_yr"] for k in ok])
    within = float(np.mean(np.abs(ratio - 1.0) <= 0.25))
    print(f"A2b(c): {len(ok)} wells, calls agree {agree:.3f}, thresholds within 25% {within:.3f}, "
          f"moving (own) {sum(bool(own[k]['detected']) for k in ok)}, ratio median {np.median(ratio):.3f}")
    assert agree >= 0.95 and within >= 0.95
    assert all("size class" in shared[k]["test_basis"] for k in ok)


def test_the_report_table_is_capped_and_the_full_list_is_a_csv(built, tmp_path, monkeypatch):
    from insar_pipeline import client_report as cr
    rep = built["rep"]
    monkeypatch.setattr(cr, "STUDY_POINT_TABLE_MAX_ROWS", 5)
    page = cr.study_section_html(built["cfg"], rep)
    n = len(rep["study_points"])
    assert f"The first 5 of {n} listed points" in page and "IN-1" in page and "OUT-6" not in page
    f = cr.write_study_points_csv(rep, tmp_path)
    df = pd.read_csv(f)
    assert len(df) == n and {"attr_status", "reading", "tier"} <= set(df.columns)


def test_many_wells_get_a_thinned_series_and_the_same_values(tmp_path, monkeypatch):
    cfg, T = make_bowl(tmp_path)
    cfg = oil_cfg(cfg, tmp_path, T, lines=False)
    full = E.write_event_products(cfg, tmp_path / "full")
    assert "values_at_thinned_to_every_nth_date" not in full
    monkeypatch.setattr(E, "STUDY_POINTS_FULL_SERIES", 5)
    thin = E.write_event_products(cfg, tmp_path / "thin")
    assert thin["values_at_thinned_to_every_nth_date"] == E.STUDY_POINTS_THIN_STEP
    a, b = pd.read_csv(tmp_path / "full" / "values_at.csv"), pd.read_csv(tmp_path / "thin" / "values_at.csv")
    assert len(b) < len(a) / 4
    m = b.merge(a, on=["id", "epoch"], suffixes=("_t", "_f"))
    assert len(m) == len(b) and np.allclose(m["value_mm_t"], m["value_mm_f"], equal_nan=True)
