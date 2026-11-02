"""The pipeline app (app_server.py, 24 Oct 2026): the phase stepper from the
dashboard's own stage rows, the start gate (a passing preflight of that kind
first, the cleaned-dates confirmation), only listed projects, the live map's
layers appearing as their files do, and the server's guards (token, Host,
exact asset list). On fake projects under a temporary INSAR_PROJECTS (the
dashboard tests' fixture); no network, no real runs."""
import base64
import http.client
import io
import json
import math
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from insar_pipeline import app_server as app, dashboard as dash, runs
import test_fix_L9a as l9

SITE_A_AOI, _project = l9.SITE_A_AOI, l9._project
home = l9.home          # the dashboard tests' fixture: a temporary projects folder


def _row(key, icon="✓", live=False, note="", done=None, total=None):
    return {"key": key, "icon": icon, "live": live, "note": note, "done": done, "total": total, "label": key,
            "elapsed": "1s", "remaining": "", "style": ""}


# ---------------------------------------------------------------- phases
def test_stage_keys_fall_into_the_phases_the_page_shows():
    assert app.phase_of("download") == "data"
    assert app.phase_of("run_02_unpack_secondary_slc") == "stack"
    assert app.phase_of("run_16_unwrap") == "unwrap"
    assert app.phase_of("miaplpy_phase_linking") == "link"
    assert app.phase_of("mintpy_correct_unwrap_error") == "series"     # MintPy's, not the stack's unwrap
    assert app.phase_of("weather_raider") == "series"
    assert app.phase_of("client_report") == "products"
    assert app.phase_of("pipeline") is None and app.phase_of("disk") is None


def test_a_project_never_run_shows_every_phase_not_run():
    ph = app._phases([], "miaplpy", "idle")
    assert [p["status"] for p in ph] == ["not run"] * 6


def test_phases_of_a_run_in_phase_linking():
    rows = [_row("download"), _row("run_02_x"), _row("miaplpy_phase_linking", "▶", True, done=50, total=100)]
    ph = {p["key"]: p for p in app._phases(rows, "miaplpy", "running")}
    assert ph["data"]["status"] == "done" and ph["stack"]["status"] == "done"
    assert ph["link"]["status"] == "active" and ph["link"]["fraction"] == pytest.approx(0.5)
    assert ph["unwrap"]["status"] == "waiting" and ph["products"]["status"] == "waiting"


def test_a_restart_from_later_on_marks_the_phases_before_it_skipped_or_earlier():
    rows = [_row("download", note="earlier run"), _row("client_report", "▶", True)]
    ph = {p["key"]: p for p in app._phases(rows, "miaplpy", "running")}
    assert ph["data"]["status"] == "earlier"
    assert ph["stack"]["status"] == "skipped" and ph["series"]["status"] == "skipped"
    assert ph["products"]["status"] == "active"


def test_the_background_weather_download_does_not_hold_its_phase_active():
    rows = [_row("download"), _row("era5_prefetch", "▶", True, done=10, total=100),
            _row("run_02_x", "▶", True, done=1, total=4)]
    ph = {p["key"]: p for p in app._phases(rows, "miaplpy", "running")}
    assert ph["data"]["status"] == "done" and ph["data"]["background"] is True
    assert ph["stack"]["status"] == "active"


def test_failed_and_interrupted_phases_say_so():
    ph = {p["key"]: p for p in app._phases([_row("download", "✗")], "disp", "failed")}
    assert list(ph) == ["data", "series", "products"] and ph["data"]["status"] == "failed"
    ph = {p["key"]: p for p in app._phases([_row("download", "■")], "stack", "interrupted")}
    assert ph["data"]["status"] == "stopped"


# ---------------------------------------------------------------- the API on a fake project
def test_home_lists_the_project_with_the_reports_own_words(home):
    _project(home)
    d = app.api_home({})
    p = next(x for x in d["projects"] if x["name"] == "demo")
    assert p["state"] == "done" and p["send_rule"] == "send with notes"
    a2 = next(a for a in p["areas"] if a["name"] == "Area A2")
    want = dash.area_view(SITE_A_AOI, True)
    assert (a2["rate"], a2["range"], a2["reading"]) == (want["rate"], want["range"], want["reading"])
    assert p["center"] == pytest.approx([38.35, -87.65])


def test_only_a_listed_project_is_served(home, tmp_path):
    other = tmp_path / "elsewhere.yaml"
    other.write_text("project_dir: /tmp/x\nbbox: [0, 0, 1, 1]\n")
    with pytest.raises(ValueError, match="not one of"):
        app.api_project({"config": str(other)})


def test_project_view_has_the_head_phases_results_and_layers(home):
    cfg = _project(home)
    d = app.api_project({"config": str(cfg)})
    assert d["head"]["state"] == "done" and d["head"]["kind"] == "full"
    assert any("Finished in 10 min" in n["text"] for n in d["head"]["notes"])
    assert d["phases"][-1]["status"] == "done"
    assert d["results"]["send_rule"] == "send with notes" and not d["results"]["final"]
    ids = {l["id"]: l for l in d["layers"]}
    assert ids["bbox"]["ready"] and ids["report_1"]["ready"]          # the box report area
    assert not ids["frame"]["ready"] and not ids["points"]["ready"]


def test_layers_appear_as_their_files_do(home):
    cfg = _project(home)
    P = home / "demo"
    (P / "slc").mkdir()
    (P / "slc" / "frame_footprint.json").write_text(json.dumps(
        {"bbox_wsen": [-88.0, 38.0, -87.0, 39.0], "n_bursts": 10, "distinct_bursts": 2}))
    import rasterio
    from rasterio.transform import from_origin
    a = np.linspace(0, 1, 40 * 50, dtype="float32").reshape(40, 50)
    a[0, 0] = np.nan
    with rasterio.open(P / "deliverables" / "quality_temporal_coherence.tif", "w", driver="GTiff", height=40, width=50,
                       count=1, dtype="float32", crs="EPSG:4326", transform=from_origin(-87.7, 38.4, .002, .0025),
                       nodata=np.nan) as d:
        d.write(a, 1)
    ids = {l["id"]: l for l in app.layer_list(app._load_cfg(str(cfg)))}
    assert ids["frame"]["ready"] and ids["coherence"]["ready"] and ids["coherence"]["source"] == "products"
    fr = app.api_layer({"config": str(cfg), "id": "frame"})
    ring = fr["data"]["features"][0]["geometry"]["coordinates"][0]
    assert ring[0] == [-88.0, 38.0] and ring[2] == [-87.0, 39.0]
    im = app.api_layer({"config": str(cfg), "id": "coherence"})
    assert im["kind"] == "image" and base64.b64decode(im["png"])[:4] == b"\x89PNG"
    assert im["bounds"] == [[pytest.approx(38.3), pytest.approx(-87.7)], [pytest.approx(38.4), pytest.approx(-87.6)]]
    assert im["legend"]["min"] == 0 and im["legend"]["max"] == 1


def test_outline_layers_are_thinned_for_the_map():
    ring = [[float(i) / 1e4, 0.0] for i in range(20000)] + [[0.0, 0.0]]
    out = app._thin_coords([ring], 10)
    assert len(out[0]) < 2100 and out[0][0] == out[0][-1]
    assert app._thin_coords([[[0, 0], [1, 0], [1, 1], [0, 0]]], 10) == [[[0, 0], [1, 0], [1, 1], [0, 0]]]


# ---------------------------------------------------------------- starting a run
def test_a_processing_run_needs_a_passing_preflight_of_that_kind_first(home, monkeypatch):
    cfg = str(_project(home))
    started = []
    monkeypatch.setattr(runs, "start", lambda c, P, kind, extra=None: started.append(kind) or {"pid": 1})
    app._PREFLIGHT.clear()
    with pytest.raises(ValueError, match="preflight"):
        app.api_start({"config": cfg, "kind": "resume"})
    monkeypatch.setattr(dash, "preflight_checks", lambda cfg, config, kind, say=None: [("fail", "no login")])
    assert app.api_preflight({"config": cfg, "kind": "resume"})["ok"] is False
    with pytest.raises(ValueError, match="preflight"):
        app.api_start({"config": cfg, "kind": "resume"})
    monkeypatch.setattr(dash, "preflight_checks", lambda cfg, config, kind, say=None: [("ok", "fine")])
    assert app.api_preflight({"config": cfg, "kind": "mintpy"})["ok"] is True
    with pytest.raises(ValueError, match="preflight"):                 # another kind's pass does not count
        app.api_start({"config": cfg, "kind": "resume"})
    assert app.api_start({"config": cfg, "kind": "mintpy"})["started"] == "mintpy"
    with pytest.raises(ValueError, match="preflight"):                 # one pass, one start
        app.api_start({"config": cfg, "kind": "mintpy"})
    app._PREFLIGHT[(str(Path(cfg).resolve()), "full")] = (time.time() - app.PREFLIGHT_TTL_S - 1, True)
    with pytest.raises(ValueError, match="preflight"):                 # a stale pass does not count
        app.api_start({"config": cfg, "kind": "full"})
    assert app.api_start({"config": cfg, "kind": "products"})["started"] == "products"   # no preflight kind
    assert started == ["mintpy", "products"]


def test_whole_pipeline_on_cleaned_dates_needs_the_box_ticked(home, monkeypatch):
    cfg = str(_project(home))
    monkeypatch.setattr(dash, "cleaned_dates", lambda c: 12)
    monkeypatch.setattr(runs, "start", lambda c, P, kind, extra=None: {"pid": 1})
    opts = {o["kind"]: o for o in app.api_run_options({"config": cfg})["options"]}
    assert "12 dates" in opts["full"]["warning"]
    app._PREFLIGHT[(str(Path(cfg).resolve()), "full")] = (time.time(), True)
    with pytest.raises(ValueError, match="confirmation"):
        app.api_start({"config": cfg, "kind": "full"})
    assert app.api_start({"config": cfg, "kind": "full", "confirm_cleaned": True})["started"] == "full"


def test_run_options_put_the_recommendation_first_and_follow_the_route(home, monkeypatch):
    cfg = str(_project(home, backend="disp"))
    monkeypatch.setattr(runs, "suggest_kind", lambda P, cfg=None: ("products", "the products are older than the report"))
    o = app.api_run_options({"config": cfg})
    assert o["options"][0]["kind"] == "products" and o["options"][0]["recommended"]
    kinds = [x["kind"] for x in o["options"]]
    assert "resume" not in kinds and "miaplpy" not in kinds and "mintpy" not in kinds
    with pytest.raises(ValueError, match="does not apply"):
        app.api_start({"config": cfg, "kind": "miaplpy"})


def test_open_only_opens_the_projects_own_paths(home, monkeypatch):
    cfg = str(_project(home))
    opened = []
    monkeypatch.setattr(dash, "open_path", lambda p: opened.append(Path(p)) or str(p))
    app.api_open({"config": cfg, "what": "report"})
    assert opened[-1].name == "client_report.html"
    with pytest.raises(ValueError):
        app.api_open({"config": cfg, "what": "/etc/passwd"})


# ---------------------------------------------------------------- the server
@pytest.fixture
def server(tmp_path, monkeypatch):
    import socket
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    monkeypatch.setattr(app, "SERVE_DIR", tmp_path / "serve")
    monkeypatch.setattr(app, "_SERVER", {})
    url = app.serve(port)
    yield port, url
    app._SERVER["httpd"].shutdown()


def _req(port, method, path, body=None, headers=None):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    h = {"Host": f"localhost:{port}", **(headers or {})}
    c.request(method, path, body=json.dumps(body).encode() if body is not None else None, headers=h)
    r = c.getresponse()
    return r.status, r.read(), dict(r.getheaders())


def test_every_asset_the_page_loads_is_served_and_nothing_else(server):
    port, _ = server
    for rel, f in app.ASSETS.items():
        assert f.is_file(), rel
    page = (app.ASSET_DIR / app.PAGE).read_text()
    import re
    for ref in re.findall(r'(?:src|href)="(assets/[^"]+)"', page):
        assert ref in app.ASSETS, ref
        st, _, _ = _req(port, "GET", "/" + ref)
        assert st == 200, ref
    assert _req(port, "GET", "/assets/../app_server.py")[0] == 404
    assert _req(port, "GET", "/assets/vendor/leaflet.draw.js")[0] == 404


def test_the_page_carries_a_token_and_the_api_wants_it(server):
    port, _ = server
    st, body, hdr = _req(port, "GET", "/app.html")
    assert st == 200 and "script-src 'self'" in hdr["Content-Security-Policy"]
    tok = body.decode().split('name="insar-token" content="', 1)[1].split('"', 1)[0]
    assert _req(port, "POST", "/api/home", {})[0] == 403
    assert _req(port, "POST", "/api/home", {}, {"X-Insar-Token": "wrong"})[0] == 403
    assert _req(port, "POST", "/api/home", {}, {"X-Insar-Token": tok, "Origin": "http://evil.example"})[0] == 403
    st, _, _ = _req(port, "GET", "/app.html", headers={"Host": "evil.example"})
    assert st == 403
    assert _req(port, "POST", "/api/nothing", {}, {"X-Insar-Token": tok})[0] == 404


def test_quit_ends_main_and_leaves_runs_alone(monkeypatch, tmp_path):
    import socket
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    monkeypatch.setattr(app, "SERVE_DIR", tmp_path / "serve")
    monkeypatch.setattr(app, "_SERVER", {})
    monkeypatch.setattr(app, "_STOP", threading.Event())
    stopped = []
    monkeypatch.setattr(runs, "stop", lambda *a, **k: stopped.append(a))
    th = threading.Thread(target=app.main, args=(["--no-window", "--port", str(port)],), daemon=True)
    th.start()
    for _ in range(50):
        if app._SERVER.get("httpd"):
            break
        time.sleep(0.05)
    app.api_quit({})
    th.join(5)
    assert not th.is_alive() and stopped == []


# ---------------------------------------------------------------- files (file_manager)
def test_files_lists_sizes_and_marks_what_is_kept(home):
    cfg = str(_project(home))
    P = home / "demo"
    (P / "stack").mkdir()
    (P / "stack" / "big.bin").write_bytes(b"x" * 5000)
    d = app.api_files({"config": cfg})
    by = {i["name"]: i for i in d["items"]}
    assert by["stack"]["bytes"] == 5000 and not by["stack"]["protected"] and by["stack"]["about"]
    assert by["logs"]["protected"] and by["run.json"]["protected"] and by["quality_assessment.json"]["protected"]
    with pytest.raises(ValueError, match="outside"):
        app.api_files({"config": cfg, "rel": "../.."})


def test_delete_refuses_kept_paths_and_asks_the_name_for_deliverables(home):
    cfg = str(_project(home))
    P = home / "demo"
    (P / "stack").mkdir()
    (P / "stack" / "a.bin").write_bytes(b"x" * 10)
    (P / "deliverables" / "final_20261001").mkdir()
    plan = app.api_plan_delete({"config": cfg, "rels": ["logs", "run.json", "../x", "deliverables/final_20261001", "stack"]})
    assert len(plan["refused"]) == 4 and [i["rel"] for i in plan["items"]] == ["stack"]
    r = app.api_delete_files({"config": cfg, "rels": ["logs", "stack"]})      # one refused item refuses the lot
    assert not r["applied"] and (P / "stack").exists()
    r = app.api_delete_files({"config": cfg, "rels": ["deliverables/client_report.html"]})
    assert not r["applied"] and "Type the project's name" in r["refused"][0]
    r = app.api_delete_files({"config": cfg, "rels": ["deliverables/client_report.html", "stack"], "confirm_name": "demo"})
    assert r["applied"] and r["freed"] > 0 and not (P / "stack").exists(), r
    log = [json.loads(x) for x in (P / "logs" / "deletions.jsonl").read_text().splitlines()]
    assert {x["path"] for x in log} == {"deliverables/client_report.html", "stack"} and all(x["deleted"] for x in log)


def test_delete_waits_for_a_run_and_holds_the_lock(home, monkeypatch):
    from insar_pipeline import file_manager as fm
    cfg = str(_project(home))
    P = home / "demo"
    (P / "stack").mkdir()
    monkeypatch.setattr(runs, "state", lambda p: "running")
    r = app.api_delete_files({"config": cfg, "rels": ["stack"]})
    assert not r["applied"] and "run is going" in r["refused"][0] and (P / "stack").exists()
    monkeypatch.setattr(runs, "state", lambda p: "done")
    held = []
    real = runs.project_lock

    def spy(*a, **k):
        held.append(k.get("kind"))
        return real(*a, **k)
    monkeypatch.setattr(runs, "project_lock", spy)
    assert fm.delete(app._load_cfg(cfg), ["stack"])["applied"] and held == ["cleanup"]


def test_deleting_a_project_from_the_app_always_wants_its_name(home):
    cfg = str(_project(home))
    r = app.api_project_space({"config": cfg, "action": "delete", "apply": True})
    assert not r["applied"] and "Type the project's name" in r["refused"][0] and (home / "demo").exists()
    plan = app.api_project_space({"config": cfg, "action": "delete"})
    assert plan["applied"] is False and plan["archive_files"] > 0


# ---------------------------------------------------------------- the company and the map export
def test_company_settings_and_logo(home, tmp_path):
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (40, 20), (200, 0, 0)).save(buf, format="JPEG")
    url = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
    c = app.api_company({"firm": "A Firm", "preparer": "Someone", "logo_data": url})
    assert c["firm"] == "A Firm" and c["logo"].startswith("data:image/png;base64,")
    from insar_pipeline import settings
    assert Path(settings.load()["company_logo"]).name == "company_logo.png"
    with pytest.raises(ValueError):
        app.api_company({"logo_data": "data:text/html;base64,PGI+"})
    assert app.api_company({"remove_logo": True})["logo"] is None


def test_map_export_draws_a_page_offline_with_draft_and_the_layers(home, monkeypatch):
    import rasterio
    from rasterio.transform import from_origin
    from insar_pipeline import map_export
    cfg = str(_project(home))
    P = home / "demo"
    a = np.linspace(-0.5, 0.5, 60 * 60, dtype="float32").reshape(60, 60)
    with rasterio.open(P / "deliverables" / "vertical_movement_in_per_yr.tif", "w", driver="GTiff", height=60, width=60, count=1,
                       dtype="float32", crs="EPSG:4326", transform=from_origin(-87.7, 38.4, .1 / 60, .1 / 60), nodata=np.nan) as d:
        d.write(a, 1)
    texts = []
    import matplotlib.figure
    real = matplotlib.figure.Figure.savefig

    def keep(self, *a_, **k):
        texts.extend(t.get_text() for ax in self.axes for t in ax.texts)
        return real(self, *a_, **k)
    monkeypatch.setattr(matplotlib.figure.Figure, "savefig", keep)
    for opts in ({"page": "letter", "orientation": "portrait", "bw": True, "format": "png"},
                 {"page": "tabloid", "orientation": "landscape", "bw": False, "format": "pdf"}):
        f = map_export.export(app._load_cfg(cfg), cfg, {**opts, "basemap": "none", "layers": ["bbox", "vertical", "report_1"],
                                                         "notes": "a note", "title": "Demo map"})
        assert f.exists() and f.stat().st_size > 10_000 and f.parent == P / "exports" / "maps"
        assert json.loads((f.parent / (f.name + ".json")).read_text())["final"] is False
    assert "DRAFT" in texts and any("a note" in t for t in texts) and any("black: going down" in t for t in texts)
    with pytest.raises(ValueError, match="page size"):
        map_export.export(app._load_cfg(cfg), cfg, {"page": "napkin"})


def test_scale_bar_is_true_ground_distance():
    from insar_pipeline import map_export as mx
    ext = mx.fit_extent((-90.5, 41.0, -90.4, 41.1), 1.0)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(4, 4))
    n = mx._scalebar(ax, ext, (4, 4))
    plt.close(fig)
    lon_m = (ext[1] - ext[0]) * math.cos(math.radians(41.05))         # true metres across the frame
    assert n == pytest.approx(lon_m / (4 * 0.0254), rel=0.01)


def test_every_element_the_script_looks_up_is_in_the_page():
    """24 Oct 2026: a button renamed in the page left app.js calling
    addEventListener on null, and the whole page came up blank."""
    import re
    js = (app.ASSET_DIR / "app.js").read_text()
    page = (app.ASSET_DIR / app.PAGE).read_text()
    ids = set(re.findall(r'id="([^"]+)"', page))
    used = set(re.findall(r"\$\('#([\w-]+)['\s)]", js))
    assert used and not (used - ids), sorted(used - ids)


def test_a_project_whose_folder_was_deleted_says_so_with_what_is_kept(home, monkeypatch, tmp_path):
    """24 Oct 2026: the validation experiments' folders were deleted after their
    runs (VALIDATION.md) and the app listed them as 'Not started' with nothing
    behind them."""
    import shutil
    cfg = _project(home, name="exp_x")
    shutil.rmtree(home / "exp_x")
    res = tmp_path / "pkg" / "review_2026-10-08" / "results" / "exp_x"
    (res / "protocol_A_X").mkdir(parents=True)
    (res / "protocol_A_X" / "validation.md").write_text("report")
    (res / "run_result.json").write_text(json.dumps({"status": "done", "purpose": "experiment", "finished": 1.79e9}))
    arch = home / "_archive"
    (arch / "exp_x_20261024").mkdir(parents=True)
    (arch / "exp_x_other_20261024").mkdir()                  # another project's archive: not this one's
    monkeypatch.setattr(app, "PKG", tmp_path / "pkg")
    p = next(x for x in app.api_home({})["projects"] if x["name"] == "exp_x")
    assert p["state"] == "removed" and p["gone"] == "folder"
    kinds = [k["kind"] for k in p["kept"]]
    assert kinds.count("results") == 1 and kinds.count("archive") == 1
    assert next(k for k in p["kept"] if k["kind"] == "results")["purpose"] == "experiment"
    opened = []
    monkeypatch.setattr(dash, "open_path", lambda f: opened.append(Path(f)) or str(f))
    i = [k for k in p["kept"] if k["path"]].index(next(k for k in p["kept"] if k["kind"] == "results"))
    app.api_open_kept({"config": str(cfg), "index": i})
    assert opened[-1].name == "validation.md"
    assert any("not on disk" in n["text"] for n in app.api_project({"config": str(cfg)})["head"]["notes"])
    runs._remember(str(cfg))
    r = app.api_forget({"config": str(cfg)})
    assert r["forgot"] and r["still_listed"]                 # its yaml is in the configs folder the pipeline reads


# ---------------------------------------------------------------- sign-off (30 Oct 2026)
@pytest.fixture
def no_signoffs():
    app._SIGNOFF.clear()
    yield
    app._SIGNOFF.clear()


def _signoff_done(config, timeout=10):
    t0 = time.time()
    while time.time() - t0 < timeout:
        j = app._signoff_job(config)
        if j and j["state"] != "working":
            return j
        time.sleep(0.02)
    raise AssertionError("the sign-off did not finish")


def test_signoff_form_asks_what_the_dashboard_asks(home, no_signoffs):
    """The app's sign-off form: the report's features to tick, the preparer as
    the default reviewer, and only the hard blockers a sign-off does not settle."""
    from insar_pipeline import release, settings
    settings.save(preparer="K. Wire")
    cfg = str(_project(home))
    f = app.api_signoff_form({"config": cfg})
    assert [x["n"] for x in f["features"]] == [1] and f["reviewer"] == "K. Wire"
    assert f["held"] == "" and f["running"] is False and f["job"] is None
    blockers = release.status(home / "demo")["blockers"]
    assert any(b["code"] == "not_signed_off" for b in blockers)
    assert f["other_blockers"] == dash.signoff_leaves(blockers)
    assert not any("signed off" in b for b in f["other_blockers"])


def test_signoff_refuses_what_release_refuses_and_records_nothing(home, monkeypatch, no_signoffs):
    """A refused sign-off is answered at once, with nothing recorded and no
    rebuild: no name, a feature not ticked, a malformed tick, a run going."""
    cfg = str(_project(home))
    rebuilt = []
    monkeypatch.setattr(dash, "rebuild_report", lambda c, P: rebuilt.append(c) or (True, ""))
    with pytest.raises(ValueError, match="every feature must be checked"):
        app.api_signoff({"config": cfg, "reviewer": "K. Wire", "features": []})
    with pytest.raises(ValueError, match="reviewer's name"):
        app.api_signoff({"config": cfg, "reviewer": "  ", "features": [1]})
    with pytest.raises(ValueError, match="feature numbers"):
        app.api_signoff({"config": cfg, "reviewer": "K. Wire", "features": ["1"]})
    with pytest.raises(ValueError, match="feature numbers"):
        app.api_signoff({"config": cfg, "reviewer": "K. Wire", "features": [True]})
    monkeypatch.setattr(runs, "state", lambda P: "running")
    with pytest.raises(ValueError, match="Wait for the run"):
        app.api_signoff({"config": cfg, "reviewer": "K. Wire", "features": [1]})
    monkeypatch.undo()
    assert rebuilt == [] and not (home / "demo" / "deliverables" / "review.json").exists()
    assert app._signoff_job(cfg) is None


def test_signoff_records_at_once_and_the_page_follows_the_rebuild(home, monkeypatch, no_signoffs):
    """Answered once release.sign_off has recorded it; the rebuild goes on in
    the background, one at a time per project, the app does not quit under
    it, and the project page reads its outcome."""
    from insar_pipeline import release
    cfg = str(_project(home))
    go, rebuilt = threading.Event(), []

    def rebuild(config, P):
        go.wait(10)
        rebuilt.append(config)
        return True, str(P / "logs" / "report.log")
    monkeypatch.setattr(dash, "rebuild_report", rebuild)
    r = app.api_signoff({"config": cfg, "reviewer": "K. Wire", "features": [1], "notes": "checked on imagery"})
    assert r["job"]["state"] == "working" and r["job"]["steps"][0].startswith("Signed off by K. Wire")
    rec = release.review_record(home / "demo")
    assert rec["reviewer"] == "K. Wire" and rec["notes"] == "checked on imagery" and rec["acknowledged_features"] == [1]
    assert app.api_project({"config": cfg})["results"]["signoff"]["state"] == "working"
    with pytest.raises(ValueError, match="already going"):
        app.api_signoff({"config": cfg, "reviewer": "K. Wire", "features": [1]})
    with pytest.raises(ValueError, match="rebuilding a report"):
        app.api_quit({})
    go.set()
    j = _signoff_done(cfg)
    assert rebuilt == [str(Path(cfg).resolve())]
    # other hard blockers remain in the fixture: signed and rebuilt, still a DRAFT, no package
    assert j["state"] == "done" and j["final"] is False and j["package"] is None and "still a DRAFT" in j["message"]
    assert app.api_project({"config": cfg})["results"]["signoff"]["message"] == j["message"]


def test_a_rebuild_that_fails_after_the_sign_off_says_it_is_recorded(home, monkeypatch, no_signoffs):
    cfg = str(_project(home))

    def held(config, P):
        raise RuntimeError("a run holds the project")
    monkeypatch.setattr(dash, "rebuild_report", held)
    app.api_signoff({"config": cfg, "reviewer": "K. Wire", "features": [1]})
    j = _signoff_done(cfg)
    assert j["state"] == "failed" and j["message"] == "The sign-off is recorded, but: a run holds the project"


def test_the_signoff_endpoints_want_the_token(server):
    port, _ = server
    for path in ("/api/signoff_form", "/api/signoff"):
        assert _req(port, "POST", path, {"config": "x"})[0] == 403
        assert _req(port, "POST", path, {"config": "x"}, {"X-Insar-Token": "wrong"})[0] == 403
