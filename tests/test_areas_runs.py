"""Area input (coordinates, polygons), project writing and the run manager."""
import json
import time
import zipfile

import pytest

from insar_pipeline import areas, progress, runs


def test_coordinates_point_radius_and_corners():
    a = areas.parse_coordinates("38.35114, -87.66331", radius_km=1.0)
    w, s, e, n = a.bbox
    assert s < 38.35114 < n and w < -87.66331 < e
    wkm, hkm = a.size_km()
    assert wkm == pytest.approx(2.0, rel=0.02) and hkm == pytest.approx(2.0, rel=0.02)
    b = areas.parse_coordinates("38.35114,-87.66331,38.35349,-87.65863")    # report-area order
    assert b.bbox == pytest.approx((-87.66331, 38.35114, -87.65863, 38.35349))
    assert b.is_box


def test_coordinates_wkt_geojson_and_errors():
    a = areas.parse_coordinates("POLYGON((-87.7 38.2, -87.6 38.2, -87.6 38.3, -87.7 38.2))")
    assert a.bbox == pytest.approx((-87.7, 38.2, -87.6, 38.3)) and not a.is_box
    gj = {"type": "Polygon", "coordinates": [[[-87.7, 38.2], [-87.6, 38.2], [-87.65, 38.3], [-87.7, 38.2]]]}
    assert areas.parse_coordinates(json.dumps(gj)).bbox[3] == pytest.approx(38.3)
    with pytest.raises(ValueError):
        areas.parse_coordinates("-87.6, 38.3, 12")
    with pytest.raises(ValueError):
        areas.parse_coordinates("138.3, -87.6")          # latitude out of range


def test_polygon_files(tmp_path):
    kml = """<kml xmlns="http://www.opengis.net/kml/2.2"><Placemark><Polygon><outerBoundaryIs><LinearRing>
    <coordinates>-87.7,38.2,0 -87.6,38.2,0 -87.6,38.3,0 -87.7,38.2,0</coordinates>
    </LinearRing></outerBoundaryIs></Polygon></Placemark></kml>"""
    (tmp_path / "a.kml").write_text(kml)
    assert areas.load_polygon(tmp_path / "a.kml").bbox == pytest.approx((-87.7, 38.2, -87.6, 38.3))
    with zipfile.ZipFile(tmp_path / "a.kmz", "w") as z:
        z.writestr("doc.kml", kml)
    assert areas.load_polygon(tmp_path / "a.kmz").rings
    a = areas.box(38.2, -87.7, 38.3, -87.6, "x")
    (tmp_path / "a.geojson").write_text(json.dumps(a.geojson()))
    assert areas.load_polygon(tmp_path / "a.geojson").is_box


def test_new_areas_offer_phase_linked_and_screening_only():
    assert list(areas.ROUTES) == ["miaplpy", "disp"]          # official first; no classic


@pytest.mark.parametrize("route", ["disp", "miaplpy"])
def test_write_project_is_a_valid_config(tmp_path, route):
    from insar_pipeline.config import load_config
    aoi = areas.box(38.30, -87.70, 38.34, -87.64, "site")
    poly = areas.parse_coordinates("POLYGON((-87.69 38.31, -87.66 38.31, -87.66 38.33, -87.69 38.31))")
    out = areas.write_project("My Test Site", aoi, route, "2020-01-01", "2024-12-31", track=19,
                              direction="ascending", disp_frame="04861", swaths=[2],
                              report_areas=[("Full Satellite Imaging", "full"), ("Site", aoi), ("Pond", poly)],
                              deforming=poly, projects_root=tmp_path / "p", configs_dir=tmp_path / "c")
    cfg = load_config(str(out))
    # disp-3: a DISP screening gets its own project folder (<slug>_disp_screening),
    # so it can never share or overwrite the official phase-linked project.
    assert cfg.backend == route and cfg.project_dir == tmp_path / "p" / areas.project_key("My Test Site", route)
    assert areas.project_key("My Test Site", route) == ("my_test_site_disp_screening" if route == "disp" else "my_test_site")
    w, s, e, n = cfg.bbox
    assert w < -87.70 and e > -87.64 and s < 38.30 and n > 38.34          # padded
    assert cfg.report_areas[0] == "Full Satellite Imaging:full"
    assert cfg.report_areas[2].endswith("pond.geojson")
    assert cfg.mintpy_deramp == "linear" and cfg.deramp_exclude_geojson.endswith("deforming_area.geojson")
    if route == "disp":
        assert cfg.disp_record_chain and not cfg.disp_mask_connected_components


def test_write_project_without_deforming_area_turns_deramp_off(tmp_path):
    from insar_pipeline.config import load_config
    aoi = areas.box(38.30, -87.70, 38.34, -87.64, "site")
    out = areas.write_project("x", aoi, "disp", "2020-01-01", "2024-12-31", track=19, direction="ascending",
                              report_areas=[("All", "full")], disp_frame="04861",
                              projects_root=tmp_path / "p", configs_dir=tmp_path / "c")
    assert load_config(str(out)).mintpy_deramp == "no"


def test_run_manager_start_state_stop(tmp_path, monkeypatch):
    cfgp = tmp_path / "x.yaml"; cfgp.write_text("")
    monkeypatch.setattr(runs, "REGISTRY", tmp_path / "reg.json")
    monkeypatch.setitem(runs.KINDS, "sleep", ("test", ["-c", "import time; time.sleep(30)"]))
    rec = runs.start(str(cfgp), tmp_path, "sleep")
    assert runs.alive(rec) and runs.state(tmp_path) == "running"
    with pytest.raises(RuntimeError):
        runs.start(str(cfgp), tmp_path, "sleep")          # one run per project
    # a recorded PID whose start time differs is NOT our process
    assert not runs.alive({**rec, "start_ticks": (rec["start_ticks"] or 0) + 1})
    assert runs.stop(tmp_path, grace_s=5) in ("stopped", "killed after the grace period")
    assert runs.state(tmp_path) == "stopped"
    assert json.loads((tmp_path / "reg.json").read_text())["configs"][0] == str(cfgp)


def test_run_state_from_progress(tmp_path, monkeypatch):
    cfgp = tmp_path / "x.yaml"; cfgp.write_text("")
    monkeypatch.setattr(runs, "REGISTRY", tmp_path / "reg.json")
    monkeypatch.setitem(runs.KINDS, "quick", ("test", ["-c", "pass"]))
    runs.start(str(cfgp), tmp_path, "quick")
    time.sleep(0.5)
    assert runs.state(tmp_path) == "failed"                # exited without a final event
    progress.emit(tmp_path, "pipeline", "done")
    assert runs.state(tmp_path) == "done"


def test_tail_reads_only_the_end(tmp_path):
    f = tmp_path / "big.log"
    f.write_text("".join(f"line {i}\n" for i in range(100000)))
    assert runs.tail(f, 3) == ["line 99997", "line 99998", "line 99999"]
    assert runs.tail(tmp_path / "missing.log") == []


def test_trash_is_recoverable_and_settings_move_the_projects_root(tmp_path, monkeypatch):
    monkeypatch.setenv("INSAR_PROJECTS", str(tmp_path / "proj"))
    from insar_pipeline import files_screen, settings
    assert settings.projects_root() == tmp_path / "proj"
    victim = tmp_path / "proj" / "old_run" / "big.h5"
    victim.parent.mkdir(parents=True); victim.write_bytes(b"x" * 1000)
    dest = files_screen.move_to_trash(victim.parent)
    assert not victim.parent.exists() and (dest / "big.h5").exists()
    assert files_screen.trash_dir() in dest.parents
    size, n, done = files_screen.dir_size(files_screen.trash_dir())
    assert (size, n, done) == (1000, 1, True)


def test_windows_paths_are_understood():
    n = areas.normalize_path
    assert str(n(r"\\wsl.localhost\Ubuntu\home\me\p\context\zone.geojson")) == "/home/me/p/context/zone.geojson"
    assert str(n("//wsl.localhost/Ubuntu/home/me/zone.geojson")) == "/home/me/zone.geojson"
    assert str(n(r"\\wsl$\Ubuntu-22.04\home\me\zone.kml")) == "/home/me/zone.kml"
    assert str(n(r'"C:\Users\me\Downloads\zone.kmz"')) == "/mnt/c/Users/me/Downloads/zone.kmz"
    assert str(n("/home/me/zone.geojson")) == "/home/me/zone.geojson"


def test_load_polygon_from_a_pasted_windows_path(tmp_path):
    a = areas.box(38.2, -87.7, 38.3, -87.6, "x")
    f = tmp_path / "zone.geojson"
    f.write_text(json.dumps(a.geojson()))
    pasted = "//wsl.localhost/Ubuntu" + str(f)
    assert areas.load_polygon(pasted).is_box


def test_detached_runs_read_the_netrc_login(tmp_path, monkeypatch):
    """A dashboard run has no terminal: credentials must come from ~/.netrc, never a prompt."""
    import io
    from insar_pipeline import data_acquisition as da
    (tmp_path / ".netrc").write_text("machine urs.earthdata.nasa.gov login someone password s3cret!\n")
    (tmp_path / ".netrc").chmod(0o600)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("EARTHDATA_USERNAME", raising=False)
    monkeypatch.delenv("EARTHDATA_PASSWORD", raising=False)
    monkeypatch.setattr("sys.stdin", io.StringIO(""))
    assert da.get_credentials() == ("someone", "s3cret!")
    (tmp_path / ".netrc").unlink()
    with pytest.raises(RuntimeError, match="netrc"):
        da.get_credentials()


def test_config_accepts_windows_paths(tmp_path):
    from insar_pipeline.config import load_config
    f = tmp_path / "c.yaml"
    f.write_text("project_dir: '\\\\wsl.localhost\\Ubuntu\\home\\me\\insar_projects\\site'\n"
                 "bbox: [-87.7, 38.2, -87.6, 38.3]\nstart_date: '2020-01-01'\nend_date: '2021-01-01'\n")
    assert str(load_config(str(f)).project_dir) == "/home/me/insar_projects/site"


def test_partial_restart_is_judged_by_its_own_events(tmp_path):
    """Site B: the full run failed after phase linking; a 'phase linking
    onward' restart emits no 'pipeline' event, so the old 'pipeline failed'
    must not make a finished restart look failed."""
    import json
    import time
    from insar_pipeline import progress, runs
    progress.emit(tmp_path, "pipeline", "failed", msg="IndexError")
    f = progress.path_for(tmp_path)                  # the failure was hours before the restart
    ev = json.loads(f.read_text().splitlines()[-1]); ev["t"] -= 7200
    f.write_text(json.dumps(ev) + "\n")
    (tmp_path / runs.RUNFILE).write_text(json.dumps({"pid": 999999999, "kind": "miaplpy", "started": time.time()}))
    progress.emit(tmp_path, "client_report", "done")
    assert runs.state(tmp_path) == "done"
    progress.emit(tmp_path, "figures", "failed", msg="x")
    assert runs.state(tmp_path) == "failed"
