"""Fixes from the technical readiness re-check of 2 Nov 2026.

BLD-06  a release holds no client data: the local client tests and any client
        report areas in context/ stay out even when built on the developer's
        machine; RELEASE.txt no longer claims client configs ship.
DEP-01  every release carries SBOM.cdx.json, built from the pinned env files.
SEC-06  every release carries MANIFEST.sha256; `insar doctor` checks the
        installed files against it (integrity.check).
ERR-03  INSAR_LOG_LEVEL sets the run log's level; a run log over the cap rolls
        over to .1-.3 and the run keeps writing under the same name.
ERR-04  `insar diagnose <project>` zips the run records, last logs and JSON
        records for the custodian, never context/ or client products.
OUT-01  the Results workbook and every PNG figure name the build.
NET-04  the map builder serves its own Leaflet; no page loads a CDN or web font.
HIL-02  the report says its readings and verdicts are calculated by the software.
RUN-04  a project path with non-ASCII characters runs intact through the shell
        commands built from it.
"""
from __future__ import annotations

import http.server
import json
import os
import re
import shutil
import subprocess
import tarfile
import threading
import urllib.error
import urllib.request
import zipfile
from functools import partial
from pathlib import Path

import pytest

from insar_pipeline import diagnose, integrity, map_builder, runs, version

PKG = Path(__file__).resolve().parents[1]


# ================================================================ BLD-06 / DEP-01 / SEC-06
@pytest.fixture(scope="module")
def release(tmp_path_factory):
    """make_release.sh run on a copy of the package that also holds what a
    developer machine holds: a local client test and a client report area."""
    base = tmp_path_factory.mktemp("rel")
    copy = base / "src" / "insar_pipeline"
    shutil.copytree(PKG, copy, ignore=shutil.ignore_patterns(".git", "__pycache__", ".pytest_cache", "*.pyc"))
    (copy / "tests" / "test_local_client_site.py").write_text("SITE = 'a client site'\n")
    (copy / "context" / "client_report_areas.geojson").write_text('{"type": "FeatureCollection", "features": []}')
    out = base / "out"
    r = subprocess.run(["bash", str(copy / "install" / "make_release.sh"), str(out)],
                       capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stdout + r.stderr
    tar = next(out.glob("insar_pipeline_*.tar.gz"))
    ex = base / "extracted"
    with tarfile.open(tar) as tf:
        names = tf.getnames()
        try:
            tf.extractall(ex, filter="data")
        except TypeError:                                          # a Python without extraction filters
            tf.extractall(ex)
    return {"names": names, "dir": ex / "insar_pipeline", "stdout": r.stdout}


def test_a_release_holds_no_local_client_tests_or_client_context(release):
    names = release["names"]
    assert "insar_pipeline/tests/test_local_client_site.py" not in names
    assert "insar_pipeline/context/client_report_areas.geojson" not in names
    ctx = [n for n in names if n.startswith("insar_pipeline/context/") and n.count("/") == 2]
    assert ctx and all(re.search(r"/(SOURCE\.txt|cmis_[^/]*\.geojson|coal_mine[^/]*\.geojson)$", n) for n in ctx), ctx
    assert "insar_pipeline/tests/client_files.py" in names           # ordinary tests still ship
    rel = (release["dir"] / "RELEASE.txt").read_text()
    assert "do not distribute outside the operating organisation" in rel
    assert "client project configurations" not in rel + release["stdout"]


def test_a_release_carries_an_sbom_of_every_environment(release):
    bom = json.loads((release["dir"] / "SBOM.cdx.json").read_text())
    assert bom["bomFormat"] == "CycloneDX" and bom["specVersion"] == "1.5"
    envs = {p["value"] for c in bom["components"] for p in c["properties"] if p["name"] == "insar:environment"}
    assert envs == {f.stem for f in (PKG / "install" / "envs").glob("*.yml")}
    purls = {c["purl"] for c in bom["components"]}
    assert any(p.startswith("pkg:conda/conda-forge/mintpy@") for p in purls)
    assert any(p.startswith("pkg:pypi/") for p in purls)


def test_the_sbom_lists_every_pinned_line_of_the_env_files():
    import importlib.util
    spec = importlib.util.spec_from_file_location("make_sbom", PKG / "scripts" / "make_sbom.py")
    mk = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mk)
    for f in (PKG / "install" / "envs").glob("*.yml"):
        env = mk.read_env(f)
        text = f.read_text()
        assert len(env["conda"]) == len(re.findall(r"(?m)^\s+- [A-Za-z0-9_.\-]+=[^=\s]", text)), f.name
        assert len(env["pip"]) == len(re.findall(r"(?m)^\s+- [A-Za-z0-9_.\-\[\]]+==", text)), f.name


def test_the_installed_release_matches_its_manifest_and_a_change_is_caught(release):
    d = release["dir"]
    ok, msg, bad = integrity.check(d)
    assert ok and bad == [] and "shipped files match" in msg, msg
    listed = integrity.read(d / integrity.MANIFEST)
    assert "pyproject.toml" in listed and "install/envs/insar.yml" in listed and "SBOM.cdx.json" in listed
    victim = d / "install" / "envs" / "insar.yml"
    keep = victim.read_bytes()
    try:
        victim.write_bytes(keep + b"# edited\n")
        (d / "patches" / "miaplpy" / "overlap_window_datum.patch").rename(d / "moved.patch")
        ok, msg, bad = integrity.check(d)
        assert not ok and "install/envs/insar.yml (changed)" in bad
        assert "patches/miaplpy/overlap_window_datum.patch (missing)" in bad
    finally:
        victim.write_bytes(keep)
        (d / "moved.patch").rename(d / "patches" / "miaplpy" / "overlap_window_datum.patch")


def test_a_checkout_without_a_manifest_is_not_a_fault(tmp_path):
    ok, msg, bad = integrity.check(tmp_path)
    assert ok and bad == [] and "not an installed release" in msg


# ================================================================ ERR-03
@pytest.mark.parametrize("value, level", [("", 20), ("debug", 10), ("WARNING", 30), ("error", 40), ("loud", 20)])
def test_the_run_log_level_comes_from_insar_log_level(monkeypatch, value, level):
    monkeypatch.setenv(runs.LOG_LEVEL_ENV, value)
    assert runs.log_level() == level


def test_a_run_log_over_the_cap_rolls_over_and_writing_carries_on(tmp_path):
    log = tmp_path / "logs" / "full_20261102_0100.log"
    log.parent.mkdir()
    fd = os.open(log, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)   # stands in for the run's stdout
    try:
        os.write(fd, b"first part\n" * 20)
        assert not runs.roll_log(log, max_bytes=10_000, keep=2, fds=(fd,))   # under the cap: untouched
        assert runs.roll_log(log, max_bytes=100, keep=2, fds=(fd,))
        os.write(fd, b"after the roll\n")
        assert (tmp_path / "logs" / f"{log.name}.1").read_text().startswith("first part")
        now = log.read_text()
        assert "log continued" in now and now.endswith("after the roll\n")
        for i in range(3):                                         # keep=2: never more than .1 and .2
            os.write(fd, b"x" * 200)
            assert runs.roll_log(log, max_bytes=100, keep=2, fds=(fd,))
        assert sorted(p.name for p in log.parent.iterdir()) == [log.name, f"{log.name}.1", f"{log.name}.2"]
    finally:
        os.close(fd)


def test_the_roller_only_runs_on_a_run_log():
    # the test runner's stdout is not a project's logs/<kind>_<stamp>.log
    assert runs.run_log_path() is None or runs.run_log_path().parent.name == "logs"


# ================================================================ ERR-04
def _project(tmp_path):
    P = tmp_path / "site_e"
    (P / "logs").mkdir(parents=True)
    (P / "deliverables").mkdir()
    (P / "context" / "sources").mkdir(parents=True)
    for n in ("run_result.json", "run.json", "quality_assessment.json"):
        (P / n).write_text("{}")
    (P / "run_result_history.jsonl").write_text("{}\n")
    (P / "progress.jsonl").write_text("{}\n")
    for i in range(5):
        f = P / "logs" / f"full_2026110{i}_0100.log"
        f.write_text(f"log {i}\n")
        os.utime(f, (1_700_000_000 + i, 1_700_000_000 + i))
    for n in ("client_report.json", "gnss_tie.json", "provenance.json", "config_snapshot.yaml"):
        (P / "deliverables" / n).write_text("{}")
    for n in ("client_report.html", "site_e.kmz", "vertical_velocity.tif", "points_attributes.gpkg"):
        (P / "deliverables" / n).write_bytes(b"client product")
    (P / "context" / "area_of_interest.geojson").write_text("{}")
    (P / "context" / "sources" / "client_mines.zip").write_bytes(b"client original")
    cfg = tmp_path / "configs" / "site_e.yaml"
    cfg.parent.mkdir()
    cfg.write_text("project_dir: x\n")
    return P, cfg


def test_diagnose_bundles_the_records_and_last_logs_but_no_client_products(tmp_path):
    P, cfg = _project(tmp_path)
    z = diagnose.write(P, str(cfg), tmp_path / "diagnostics", "site_e", doctor=False)
    assert z.parent == tmp_path / "diagnostics" and z.name.startswith("site_e_") and z.suffix == ".zip"
    with zipfile.ZipFile(z) as zz:
        names = set(zz.namelist())
        manifest = zz.read("MANIFEST.txt").decode()
    assert {"run_result.json", "run_result_history.jsonl", "progress.jsonl", "quality_assessment.json",
            "deliverables/client_report.json", "deliverables/provenance.json", "deliverables/config_snapshot.yaml",
            "config/site_e.yaml", "version.txt", "MANIFEST.txt"} <= names
    assert sorted(n for n in names if n.startswith("logs/")) == [
        "logs/full_20261102_0100.log", "logs/full_20261103_0100.log", "logs/full_20261104_0100.log"]
    assert not [n for n in names if n.startswith("context/") or n.endswith((".kmz", ".tif", ".gpkg", ".html", ".zip"))]
    for n in names - {"MANIFEST.txt"}:
        assert n in manifest
    assert version.label() in manifest


def test_insar_diagnose_is_a_command(tmp_path, monkeypatch, capsys):
    from insar_pipeline import settings, storage
    P, cfg = _project(tmp_path)
    monkeypatch.setattr(storage, "find", lambda name: ("site_e", P, str(cfg), None))
    monkeypatch.setattr(settings, "projects_root", lambda: tmp_path)
    monkeypatch.setattr(diagnose, "_doctor_quick", lambda timeout=300: "doctor stand-in\n")
    assert diagnose.main(["site_e"]) == 0
    assert "holds no client outlines" in capsys.readouterr().out
    assert len(list((tmp_path / "diagnostics").glob("site_e_*.zip"))) == 1
    from insar_pipeline import dashboard
    assert "insar diagnose <project>" in dashboard.USAGE


# ================================================================ OUT-01
def test_every_png_names_the_build(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    fig = plt.figure()
    fig.savefig(tmp_path / "f.png", metadata=version.png_metadata())
    plt.close(fig)
    assert Image.open(tmp_path / "f.png").info.get("Software") == version.label()
    for f in ("results.py", "figures.py", "client_report.py", "deliverables.py"):
        src = (PKG / f).read_text()
        pngs = [args for obj, args in re.findall(r"(\w+)\.savefig\((.*)\)", src) if obj != "pdf"]
        assert pngs and all("png_metadata()" in m for m in pngs), (f, pngs)


# ================================================================ NET-04
def test_the_map_builder_serves_its_own_leaflet_and_loads_no_cdn(tmp_path):
    page = (PKG / map_builder.PAGE).read_text()
    for cdn in ("cdnjs.", "googleapis", "gstatic", "unpkg.", "jsdelivr"):
        assert cdn not in page, cdn
    wanted = set(re.findall(r'(?:href|src)="(assets/vendor/[^"]+)"', page))
    assert wanted and wanted <= set(map_builder.ASSETS), wanted - set(map_builder.ASSETS)
    assert all(f.is_file() for f in map_builder.ASSETS.values())
    (tmp_path / map_builder.PAGE).write_bytes(page.encode())
    H = type("H", (map_builder._Handler,), {"port": 0, "token": "t" * 32})
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), partial(H, directory=str(tmp_path)))
    H.port = httpd.server_address[1]
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        with urllib.request.urlopen(f"http://localhost:{H.port}/assets/vendor/leaflet.min.js", timeout=10) as r:
            assert r.status == 200 and r.headers["Content-Type"].startswith("text/javascript")
            assert r.read() == (map_builder.VENDOR / "leaflet.min.js").read_bytes()
        for bad in ("/assets/vendor/../../config.py", "/viewer_assets/vendor/leaflet.min.js", "/uploads/x"):
            try:
                urllib.request.urlopen(f"http://localhost:{H.port}{bad}", timeout=10)
                raise AssertionError(bad)
            except urllib.error.HTTPError as e:
                assert e.code == 404, bad
    finally:
        httpd.shutdown()
        httpd.server_close()


# ================================================================ HIL-02
def test_the_report_says_its_verdicts_are_calculated_by_the_software():
    from types import SimpleNamespace
    from insar_pipeline import client_report as cr
    import html as _html
    page = cr.footer_html(SimpleNamespace(project_dir="/x/site_e"), {"title": "Site E"}, final=False)
    assert _html.escape(cr.GENERATED_NOTE) in page
    assert "calculated by the software" in cr.GENERATED_NOTE


# ================================================================ RUN-04
def test_a_project_path_with_non_ascii_characters_runs_intact(tmp_path, monkeypatch):
    from insar_pipeline import miaplpy_backend as mb
    import test_fix_L6b as L6b                                     # the synthetic phase-linked project
    conda_sh = tmp_path / "condá" / "conda.sh"
    monkeypatch.setattr(mb, "CONDA_SH", str(conda_sh))
    cfg = L6b._cfg(tmp_path / "Projets société – Site É 東")
    work = mb.work_dir(cfg)
    work.mkdir(parents=True, exist_ok=True)
    tpl = work / "modèle.txt"
    argv = mb._command_argv(cfg, [tpl, "--dir", work])
    stubs = ("source() { printf 'SOURCE<%s>\\n' \"$@\"; }; conda() { :; }; "
             "miaplpyApp() { printf 'APP<%s>\\n' \"$@\"; pwd; }; ")
    r = subprocess.run(["bash", "-c", stubs + argv[-1]], capture_output=True, text=True, timeout=30,
                       env={**os.environ, "LC_ALL": "C.UTF-8"})
    assert r.returncode == 0, r.stderr
    out = r.stdout.splitlines()
    assert out[0] == f"SOURCE<{conda_sh}>"
    assert [ln for ln in out if ln.startswith("APP<")] == [f"APP<{tpl}>", "APP<--dir>", f"APP<{work}>"]
    assert out[-1] == str(work)


# ================================================================ DEP-03 (credits on the phase-linked NISAR route)
def test_the_phase_linked_nisar_route_credits_the_gslcs_and_dolphin(tmp_path):
    import dataclasses
    import test_fix_oct26_nisar_products as N26                    # its NISAR config helper
    from insar_pipeline import credits
    gunw = N26._cfg(tmp_path, "nisar")
    c = credits.credits(gunw)
    assert "GSLC" not in c[0]["text"] and not any(x["source"] == "dolphin" for x in c)
    pl = dataclasses.replace(gunw, nisar_phase_linked=True) if dataclasses.is_dataclass(gunw) else gunw
    if pl is gunw:
        pl.nisar_phase_linked = True
    c = credits.credits(pl)
    assert c[0]["source"] == "NISAR" and "GSLC" in c[0]["text"] and "GUNW" in c[0]["text"]
    assert [x["source"] for x in c if x["kind"] == "software"] == ["dolphin", "snaphu", "MintPy"]


# ================================================================ SEC-05 (script injection into the rendered report)
def test_hostile_area_and_title_text_is_escaped_in_the_client_report(tmp_path):
    import dataclasses
    import test_fix_L9b as L9b                                     # its synthetic farmland project
    from insar_pipeline import client_report as cr
    cfg = L9b.make_project(tmp_path)
    evil = '<script>alert(1)</script> & "Field"'
    cfg = dataclasses.replace(
        cfg, project_title='<img src=x onerror=alert(2)>',
        report_areas=tuple(f"{evil}:{spec.split(':', 1)[1]}" if spec.startswith("Test Field:") else spec
                           for spec in cfg.report_areas))
    cr.build(cfg)
    html = (cfg.project_dir / "deliverables" / "client_report.html").read_text()
    assert "<script>alert(1)" not in html and "<img src=x onerror" not in html
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html            # shown, as text


# ================================================================ DAT-07 (nothing in the zip or the report record names the login)
def test_the_diagnose_zip_masks_the_home_folder_and_the_earthdata_user(tmp_path, monkeypatch):
    home = tmp_path / "home" / "someone"
    monkeypatch.setenv("HOME", str(home))
    P, cfg = _project(home)
    (P / "logs" / "full_20261104_0100.log").write_text(f"reading {home}/insar_projects/site_e/x.h5\n")
    os.utime(P / "logs" / "full_20261104_0100.log", (1_800_000_000, 1_800_000_000))
    monkeypatch.setattr(diagnose, "_doctor_quick",
                        lambda timeout=300: "The Earthdata login for 'someone' works\n")
    z = diagnose.write(P, str(cfg), tmp_path / "d", "site_e", doctor=True)
    with zipfile.ZipFile(z) as zz:
        text = "".join(zz.read(n).decode() for n in zz.namelist())
    assert str(home) not in text and "'someone'" not in text
    assert "~/insar_projects/site_e/x.h5" in text and "login for '<user>'" in text


def test_the_report_record_shows_the_home_folder_as_a_tilde(tmp_path, monkeypatch):
    from insar_pipeline import provenance
    monkeypatch.setenv("HOME", str(tmp_path))
    assert provenance.portable(str(tmp_path / "insar_projects" / "site_e")) == "~/insar_projects/site_e"
    src = (PKG / "client_report.py").read_text()
    assert '"project": portable(str(P))' in src


# ================================================================ DEP-04 (no noncommercial CS2 code on the NISAR route)
def test_dolphin_is_configured_to_unwrap_without_snaphus_cs2_module(tmp_path, monkeypatch):
    import yaml
    from insar_pipeline import nisar_dolphin as nd

    def fake_config(cmd, **kw):                                     # what `dolphin config` writes: its defaults
        out = Path(cmd[cmd.index("--outfile") + 1])
        out.write_text(yaml.safe_dump({"unwrap_options": {"unwrap_method": "snaphu",
                                                          "snaphu_options": {"init_method": "mcf", "cost": "smooth"}},
                                       "timeseries_options": {}}))
        return subprocess.CompletedProcess(cmd, 0, "", "")
    monkeypatch.setattr(nd.subprocess, "run", fake_config)
    cfg = nd.write_config(tmp_path, {"files": ["a.tif", "b.tif"], "wavelength_m": 0.24}, 11, (30, 60))
    y = yaml.safe_load(cfg.read_text())
    assert y["unwrap_options"]["snaphu_options"]["init_method"] == "mst" == nd.INIT_METHOD
    assert y["unwrap_options"]["snaphu_options"]["cost"] == "smooth"       # nothing else changed


def test_unwrapped_results_from_an_mcf_run_are_moved_aside_once(tmp_path):
    from insar_pipeline import nisar_dolphin as nd
    (tmp_path / "unwrapped").mkdir()
    (tmp_path / "unwrapped" / "a.unw.tif").write_bytes(b"x")
    (tmp_path / "timeseries").mkdir()
    (tmp_path / "nisar_pl_run.json").write_text(json.dumps({"unwrap": "snaphu (smooth, 2 x 2 tiles)"}))
    assert nd.retire_other_init_outputs(tmp_path) == ["unwrapped", "timeseries"]
    assert not (tmp_path / "unwrapped").exists()
    moved = next(tmp_path.glob("superseded_mcf_*"))
    assert (moved / "unwrapped" / "a.unw.tif").exists()
    (tmp_path / "unwrapped").mkdir()                                  # made again, with MST
    (tmp_path / "nisar_pl_run.json").write_text(json.dumps({"init_method": "mst"}))
    assert nd.retire_other_init_outputs(tmp_path) == [] and (tmp_path / "unwrapped").exists()
