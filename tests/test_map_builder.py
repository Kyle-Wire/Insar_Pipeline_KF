"""The map builder's API: uploads (projected, zipped shapefiles), project
creation from the defaults, mine clipping, overwrite with backup, and the
local-only guard."""
import base64
import io
import json
import zipfile

import pytest

from insar_pipeline import areas, map_builder as mb, runs, settings


def _shapefile_zip(tmp_path):
    gpd = pytest.importorskip("geopandas")
    from shapely.geometry import box
    df = gpd.GeoDataFrame({"n": [1, 2, 3]}, crs=4326, geometry=[
        box(-87.50, 38.30, -87.48, 38.32), box(-87.45, 38.35, -87.44, 38.36), box(-86.0, 39.5, -85.9, 39.6)])
    df.to_crs(2966).to_file(tmp_path / "mines.shp")                 # Indiana State Plane West, feet
    zb = io.BytesIO()
    with zipfile.ZipFile(zb, "w") as z:
        for f in tmp_path.glob("mines.*"):
            z.write(f, f"sub/{f.name}")
    return base64.b64encode(zb.getvalue()).decode()


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "projects_root", lambda: tmp_path / "projects")
    monkeypatch.setattr(settings, "configs_dir", lambda: tmp_path / "configs")
    monkeypatch.setattr(runs, "REGISTRY", tmp_path / "registry.json")
    monkeypatch.setattr(mb, "UPLOADS", tmp_path / "uploads")
    monkeypatch.setattr(areas, "subswaths", lambda *a, **k: [2, 3])
    monkeypatch.setattr(areas, "burst_coverage", lambda area, track, direction, targets=None, **k: {
        "swaths": [2, 3], "bursts": 4, "covered": 1.0, "targets": {n: 1.0 for n in (targets or {})}})
    return tmp_path


def test_projected_zipped_shapefile_comes_back_in_lonlat(sandbox):
    d = mb.api_outline({"files": [{"name": "mines.zip", "data": _shapefile_zip(sandbox)}]})
    assert d["polygons"] == 3
    w, s, e, n = d["bbox"]
    assert w == pytest.approx(-87.50, abs=1e-4) and n == pytest.approx(39.6, abs=1e-4)


def test_create_project_from_layers(sandbox):
    up = mb.api_outline({"files": [{"name": "mines.zip", "data": _shapefile_zip(sandbox)}]})
    area = areas.box(38.29, -87.52, 38.37, -87.42, "Zone").geojson()
    body = {"name": "Test Mine", "route": "miaplpy", "start": "2016-01-01", "end": "2026-06-30", "track": 19,
            "direction": "ASCENDING", "reference": "38.33,-87.47", "pad_km": 5,
            "layers": [{"role": "area", "name": "Zone", "geojson": area},
                       {"role": "mine", "kind": "underground", "name": "CMIS", "upload_id": up["upload_id"], "report": True},
                       {"role": "mine", "kind": "surface", "name": "Pit",
                        "geojson": areas.box(38.33, -87.46, 38.34, -87.45).geojson()},
                       {"role": "report", "name": "North field", "geojson": areas.box(38.35, -87.5, 38.36, -87.49).geojson()}]}
    r = mb.api_create(body)
    from insar_pipeline.config import load_config
    cfg = load_config(r["config"])
    assert cfg.backend == "miaplpy" and cfg.track == 19 and cfg.stack_swaths == "2 3"
    assert cfg.reference_lalo == "38.33000,-87.47000"
    assert r["report_areas"] == ["Full Satellite Imaging", "Zone", "CMIS", "North field"]
    assert (r["mine_polygons_uploaded"], r["mine_polygons_kept"]) == (4, 3)   # the far one is outside the box
    assert cfg.mintpy_deramp == "linear" and cfg.deramp_exclude_geojson
    # typed layers kept apart for the report, all of them in the deramp exclusion
    ug = json.loads(open(cfg.underground_mines_geojson).read())["features"]
    sf = json.loads(open(cfg.surface_mines_geojson).read())["features"]
    assert (len(ug), len(sf)) == (2, 1) and cfg.project_title == "Test Mine"
    ctx = sandbox / "projects" / "test_mine" / "context"
    assert (ctx / "sources" / "mine_cmis" / "mines.zip").exists()
    saved = json.loads((ctx / "builder_layers.json").read_text())
    assert [(L["role"], L["kind"]) for L in saved["layers"]] == [("area", ""), ("mine", "underground"), ("mine", "surface"), ("report", "")]
    # reopening gives the same layers back
    again = mb.api_project({"config": r["config"]})
    assert [L["name"] for L in again["layers"]] == ["Zone", "CMIS", "Pit", "North field"]
    # a second create asks first, then keeps a backup
    assert "exists" in mb.api_create(body)
    r2 = mb.api_create({**body, "overwrite": True})
    assert r2["backup"] and (sandbox / "configs").glob("test_mine.yaml.bak_*")


def test_create_refuses_bad_input(sandbox):
    area = {"role": "area", "name": "A", "geojson": areas.box(38.29, -87.52, 38.37, -87.42).geojson()}
    base = {"name": "x", "track": 19, "direction": "ASCENDING"}
    with pytest.raises(ValueError, match="exactly one project area"):
        mb.api_create({**base, "layers": []})
    with pytest.raises(ValueError, match="share a name"):
        mb.api_create({**base, "layers": [area, {**area, "role": "report"}]})
    with pytest.raises(ValueError, match="DISP-S1"):
        mb.api_create({**base, "route": "disp", "layers": [area]})
    with pytest.raises(ValueError, match="bad upload id"):
        mb.api_create({**base, "layers": [{"role": "area", "upload_id": "../../etc"}]})


def test_server_is_local_only(sandbox, monkeypatch):
    import urllib.error
    import urllib.request
    import re
    import socket
    monkeypatch.setattr(mb, "SERVE_DIR", sandbox / "serve")        # not the user's ~/.cache
    monkeypatch.setattr(mb, "_SERVER", {})                          # this test's own server, stopped at the end
    # a free port, not a fixed one: a test run beside another kept 8798 (its
    # server lives as long as its pytest) and this one talked to that server
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    mb.serve(port=port)

    def post(origin=None, host=None):
        h = {"Content-Type": "application/json", mb.TOKEN_HEADER: token}
        if origin:
            h["Origin"] = origin
        if host:
            h["Host"] = host
        req = urllib.request.Request(f"http://127.0.0.1:{port}/api/info", data=b"{}", headers=h)
        try:
            with urllib.request.urlopen(req) as r:
                return r.status
        except urllib.error.HTTPError as e:
            return e.code
    try:
        # every call carries the token the server wrote into the page it served
        # (16 Oct 2026, readiness review NET-06; tests/test_fix_oct16_security.py)
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/{mb.PAGE}") as r:
            token = re.search(r'<meta name="insar-token" content="([^"]+)">', r.read().decode()).group(1)
        assert post(host=f"localhost:{port}") == 200
        assert post(origin=f"http://localhost:{port}", host=f"localhost:{port}") == 200
        assert post(origin="https://example.com", host=f"localhost:{port}") == 403
        assert post(host=f"evil.example:{port}") == 403                    # DNS rebinding
        with pytest.raises(urllib.error.HTTPError):
            urllib.request.urlopen(f"http://127.0.0.1:{port}/uploads/")
    finally:
        mb._SERVER["httpd"].shutdown()
        mb._SERVER["httpd"].server_close()
