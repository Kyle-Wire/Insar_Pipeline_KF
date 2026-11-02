"""The combined view in the program (PLAN review_2026-10-28, Phase 2): the app
and the dashboard offer to combine two finished projects that look at the
same report areas from opposite passes, list the combined project with its
two sources, run Combine from the Start dialog, and draw its vertical and
east-west layers (east-west red = east, blue = west). Fake projects under a
temporary INSAR_PROJECTS (the dashboard tests' fixture); combine.run is
replaced by a writer of a combined record, since the real one needs MintPy
products."""
import json

import numpy as np
import pytest

from insar_pipeline import app_server as app, combine, dashboard as dash
from insar_pipeline.config import load_config
import test_fix_L9a as l9

home = l9.home


def _pair(root):
    a = l9._project(root, "demo")
    d = l9._project(root, "demo_desc")
    a.write_text(a.read_text() + "orbit_direction: ascending\n")
    d.write_text(d.read_text() + "orbit_direction: descending\n")
    combine._STATUS.clear()
    return a, d


AREAS = [{"name": "Full Satellite Imaging", "combined": True, "reading": "Not testable",
          "vertical": {"testable": False}, "east_west": {"movement_mm_yr": None}},
         {"name": "Area A2", "combined": True,
          "reading": "No vertical movement detected; " + combine.EAST_PHRASE,
          "sentence": "No vertical movement detected (the two directions together).", "note": None,
          "vertical": {"testable": True, "vertical_mm_yr": -1.69, "range95_mm_yr": [-6.69, 3.31], "detected": False},
          "east_west": {"movement_mm_yr": 2.03, "stands_out_indicative": True, "status": "indicative"}}]


def _fake_run(calls):
    def run(path):
        calls.append(str(path))
        cc = combine.load(path)
        src = [{"direction": k, "config": str(getattr(cc, k)),
                "files": combine.source_files(load_config(str(getattr(cc, k))))} for k in ("ascending", "descending")]
        cc.deliverables.mkdir(parents=True, exist_ok=True)
        _raster(cc.deliverables / "east_west_movement_INDICATIVE_in_per_yr.tif")
        _raster(cc.deliverables / "vertical_movement_combined_in_per_yr.tif")
        (cc.deliverables / combine.REPORT).write_text(json.dumps(
            {"kind": "combined", "generated": "2026-10-28T10:00:00", "sources": src, "anchored_to_gnss": True,
             "comparison_ground": {"words": "the surrounding ground (mostly built-up and grassland)"},
             "areas": AREAS}))
        combine._STATUS.clear()
        return cc.deliverables / combine.REPORT
    return run


def _raster(path):
    import rasterio
    from rasterio.transform import from_origin
    a = np.linspace(-0.2, 0.2, 40 * 50, dtype="float32").reshape(40, 50)
    with rasterio.open(path, "w", driver="GTiff", height=40, width=50, count=1, dtype="float32", crs="EPSG:4326",
                       transform=from_origin(-87.7, 38.4, .002, .0025), nodata=np.nan) as d:
        d.write(a, 1)


def test_a_project_with_a_partner_is_offered_combine(home):
    a, d = _pair(home)
    h = app.api_project({"config": str(a)})["head"]
    assert h["combine"]["kind"] == "offer" and h["combine"]["name"] == "demo_desc"
    assert h["combine"]["direction"] == "descending"
    assert any("can be combined" in n["text"] for n in h["notes"])
    # no partner: another direction missing, or other report areas
    combine._STATUS.clear()
    d.write_text(d.read_text().replace("orbit_direction: descending", "orbit_direction: ascending"))
    assert "combine" not in app.api_project({"config": str(a)})["head"]


def test_combine_with_only_the_offered_partner(home, monkeypatch):
    a, d = _pair(home)
    calls = []
    monkeypatch.setattr(combine, "run", _fake_run(calls))
    other = l9._project(home, "other")
    with pytest.raises(ValueError, match="not on offer"):
        app.api_combine_with({"config": str(a), "partner": str(other)})
    r = app.api_combine_with({"config": str(a), "partner": str(d)})
    assert r["name"] == "demo_combined" and len(calls) == 1
    cc = combine.load(r["config"])
    assert cc.ascending == a.resolve() and cc.descending == d.resolve()
    # the source page now points at it, current
    h = app.api_project({"config": str(a)})["head"]
    assert h["combine"]["kind"] == "combined" and h["combine"]["name"] == "demo_combined"
    assert h["combine"]["status"] == "current"


def test_the_combined_project_in_the_list_and_on_its_page(home, monkeypatch):
    a, d = _pair(home)
    monkeypatch.setattr(combine, "run", _fake_run([]))
    c = app.api_combine_with({"config": str(a), "partner": str(d)})["config"]
    row = next(p for p in app.api_home({})["projects"] if p["name"] == "demo_combined")
    assert row["combined"] and row["route"] == "combined ascending + descending" and row["state"] == "done"
    assert row["center"] == pytest.approx([38.35, -87.65])
    a2 = next(x for x in row["areas"] if x["name"] == "Area A2")
    assert a2["rate"].startswith("-0.07 in/yr vertical") and "indicative" in a2["total"]
    pg = app.api_project({"config": c})
    assert [s["name"] for s in pg["head"]["sources"]] == ["demo", "demo_desc"]
    assert pg["phases"] == [] and pg["results"]["combined"]
    assert "built-up and grassland" in pg["results"]["frame"]
    ids = {x["id"]: x for x in pg["layers"]}
    assert ids["combined_vertical"]["ready"] and ids["combined_east"]["ready"]
    assert not ids["combined_vertical_unc"]["ready"]
    assert "frame" not in ids and "coherence" not in ids          # a combined project draws only its own
    # out of date once a source changes
    (home / "demo_desc" / "mintpy" / "geo" / "geo_velocity.h5").write_bytes(b"rebuilt")
    assert any(n["text"].startswith("Out of date") for n in app.api_project({"config": c})["head"]["notes"])


def test_start_on_a_combined_project_is_combine(home, monkeypatch):
    a, d = _pair(home)
    calls = []
    monkeypatch.setattr(combine, "run", _fake_run(calls))
    c = app.api_combine_with({"config": str(a), "partner": str(d)})["config"]
    o = app.api_run_options({"config": c})
    assert [x["kind"] for x in o["options"]] == ["combine"] and not o["options"][0]["preflight"]
    assert app.api_start({"config": c, "kind": "combine"})["finished"] and len(calls) == 2
    with pytest.raises(ValueError, match="combined project only"):
        app.api_start({"config": str(a), "kind": "combine"})


def test_east_west_layer_is_red_east_blue_west(home, monkeypatch):
    a, d = _pair(home)
    monkeypatch.setattr(combine, "run", _fake_run([]))
    c = app.api_combine_with({"config": str(a), "partner": str(d)})["config"]
    im = app.api_layer({"config": c, "id": "combined_east"})
    lg = im["legend"]
    assert im["kind"] == "image" and lg["style"] == "div_ew" and lg["min"] == -lg["max"]
    r_hi, b_hi = int(lg["colors"][-1][1:3], 16), int(lg["colors"][-1][5:7], 16)
    r_lo, b_lo = int(lg["colors"][0][1:3], 16), int(lg["colors"][0][5:7], 16)
    assert r_hi > b_hi and b_lo > r_lo                             # + (east) red, - (west) blue


def test_a_single_project_never_lists_the_combined_layers(home):
    a, _ = _pair(home)
    ids = {x["id"] for x in app.layer_list(app._load_cfg(str(a)))}
    assert not ids & {"combined_vertical", "combined_east", "combined_vertical_unc"}


def test_the_dashboard_opens_a_combined_project_on_its_own_screen(home, monkeypatch):
    a, d = _pair(home)
    monkeypatch.setattr(combine, "run", _fake_run([]))
    c = app.api_combine_with({"config": str(a), "partner": str(d)})["config"]
    assert isinstance(dash.project_screen(c), dash.CombinedScreen)
    assert isinstance(dash.project_screen(str(a)), dash.RunScreen)
    t = dash.combined_text(c).plain
    assert "Ascending:  demo" in t and "Descending: demo_desc" in t
    assert "Area A2" in t and "-0.07 in/yr vertical" in t and "indicative" in t


def test_dashboard_combined_screen_and_the_c_key_on_a_source(home, monkeypatch):
    a, d = _pair(home)
    calls = []
    monkeypatch.setattr(combine, "run", _fake_run(calls))

    async def body(app_, pilot):
        await l9._settle(app_, pilot)
        rs = app_.screen
        assert isinstance(rs, dash.RunScreen)
        await l9._settle(app_, pilot)
        assert rs.combine_status and rs.combine_status["kind"] == "offer"
        await pilot.press("c")                         # combine with demo_desc: asks first
        await l9._settle(app_, pilot)
        app_.screen.query_one("#yes", dash.Button).press()
        await l9._settle(app_, pilot, 6)
        assert len(calls) == 1
        cs = app_.screen
        assert isinstance(cs, dash.CombinedScreen)
        assert "Descending: demo_desc" in cs.query_one("#cbody", dash.Static).render().plain

    l9._pilot(dash.InsarDashboard(str(a)), body)
