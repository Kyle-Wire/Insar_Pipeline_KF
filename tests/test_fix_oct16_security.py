"""
The 16 Oct 2026 Technical Readiness Review, security and integrity batch.

  SEC-02 / RUN-04  conda env names and paths in `bash -c` command strings:
                   the env name is checked, every path quoted
  SEC-03 / SEC-04  outline files from outside the operating organisation: content checked
                   against the extension, size caps (a typed path too), zip
                   members counted and sized from the directory before any
                   is read, DOCTYPE/ENTITY KML refused, feature and vertex
                   caps, parse errors in words
  NET-06           the builder's and viewer's local servers take an /api/
                   call only with the token written into the page they serve
  DAT-03           GDAL's Earthdata cookie jar in a private folder, not /tmp
  DAT-07           run_pipeline.py no longer prints the Earthdata user name
  OUT-03           the linking fingerprint's comparison covers linked dates
                   since removed from the SLC stack and the MiaplPy build;
                   a fingerprint written before (without them) still matches
  (BLD-05, the release checksum, is in test_fix_oct7_dec_D14D18_cfg.py's
  release test.)

Every test here failed on the code at f3a0a0a and passes after the fix,
except the three marked GUARD, which pass on both: they pin that what was
accepted before still is (valid env names give the same command, the public
mine layers still load, and a fingerprint written before this change still
matches - no 12 h re-link for an existing project).

Synthetic files only; no network, no conda, no MiaplPy.
"""
import http.server
import importlib.util
import json
import os
import re
import stat
import subprocess
import sys
import threading
import urllib.error
import urllib.request
import zipfile
from functools import partial
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from insar_pipeline import areas                          # noqa: E402
from insar_pipeline import map_builder                    # noqa: E402
from insar_pipeline import miaplpy_backend as mb          # noqa: E402
from insar_pipeline import results_viewer                 # noqa: E402
from insar_pipeline import stack_backend as sb            # noqa: E402
from insar_pipeline import static_layers                  # noqa: E402

import test_fix_L6b as L6b                                # noqa: E402  (the synthetic phase-linked project)

PKG = Path(__file__).resolve().parents[1]
RING = [[10.0, 45.0], [10.02, 45.0], [10.02, 45.02], [10.0, 45.02], [10.0, 45.0]]


# ================================================================ SEC-02 / RUN-04
# Stand-ins for source / conda / python / miaplpyApp: the command string is run
# by bash exactly as built, and each stand-in prints the arguments it got.
STUBS = ("source() { printf 'SOURCE<%s>\\n' \"$@\"; }; conda() { printf 'CONDA<%s>\\n' \"$@\"; }; "
         "python() { printf 'PYTHON<%s>\\n' \"$@\"; }; miaplpyApp() { printf 'APP<%s>\\n' \"$@\"; pwd; }; ")


_RUN = subprocess.run                                     # the real one: tests replace mb.subprocess.run


def _run_stubbed(cmd: str) -> list:
    r = _RUN(["bash", "-c", STUBS + cmd], capture_output=True, text=True, timeout=30)
    assert r.returncode == 0, r.stderr
    return r.stdout.splitlines()


def test_miaplpy_command_quotes_a_project_path_with_a_space_and_a_quote(tmp_path, monkeypatch):
    conda_sh = tmp_path / "my conda" / "etc" / "conda.sh"
    monkeypatch.setattr(mb, "CONDA_SH", str(conda_sh))
    cfg = L6b._cfg(tmp_path / "Site D's project folder")
    work = mb.work_dir(cfg)
    work.mkdir(parents=True, exist_ok=True)
    tpl = work / "site d's template.txt"
    argv = mb._command_argv(cfg, [tpl, "--dir", work, "--dostep", "load_data"])
    assert argv[:2] == ["bash", "-lc"]
    out = _run_stubbed(argv[-1])
    assert out[0] == f"SOURCE<{conda_sh}>"                         # one argument, space and all
    assert out[1:3] == ["CONDA<activate>", "CONDA<miaplpy>"]
    assert [ln for ln in out if ln.startswith("APP<")] == [
        f"APP<{tpl}>", "APP<--dir>", f"APP<{work}>", "APP<--dostep>", "APP<load_data>"]
    assert out[-1] == str(work)                                    # cd went into the quoted folder


def test_patch_check_command_quotes_its_paths(tmp_path, monkeypatch):
    conda_sh = tmp_path / "my conda" / "conda.sh"
    check = tmp_path / "it's here" / "check_miaplpy_patch.py"
    monkeypatch.setattr(mb, "CONDA_SH", str(conda_sh))
    monkeypatch.setattr(mb, "PATCH_CHECK", check)
    seen = {}

    def fake(argv, **kw):
        seen["cmd"] = argv[-1]
        return subprocess.CompletedProcess(argv, 0, json.dumps({"patched": True, "miaplpy_version": "x"}) + "\n", "")
    monkeypatch.setattr(mb.subprocess, "run", fake)
    mb.verify_patched_miaplpy("miaplpy", ministack=10, time_lag=3)
    out = _run_stubbed(seen["cmd"])
    assert out == [f"SOURCE<{conda_sh}>", "CONDA<activate>", "CONDA<miaplpy>",
                   f"PYTHON<{check}>", "PYTHON<--ministack>", "PYTHON<10>", "PYTHON<--time-lag>", "PYTHON<3>"]


@pytest.mark.parametrize("bad", ["miaplpy; touch /tmp/owned", "my env", "env$(id)", "-n", "x`id`", "a\nb", ""])
def test_a_bad_conda_env_name_is_refused_before_anything_runs(tmp_path, monkeypatch, bad):
    def boom(*a, **k):
        raise AssertionError("a command ran with a bad env name")
    monkeypatch.setattr(mb.subprocess, "run", boom)
    cfg = L6b._cfg(tmp_path, miaplpy_env=bad)
    with pytest.raises(ValueError, match="miaplpy_env = .* is not a conda environment name"):
        mb._command_argv(cfg, ["t.txt", "--dostep", "load_data"])
    with pytest.raises(ValueError, match="is not a conda environment name"):
        mb.verify_patched_miaplpy(bad)
    if bad:                                                        # "" means "the current env" for the stack
        cfg.stack_conda_env = bad
        with pytest.raises(ValueError, match="stack_conda_env = .* is not a conda environment name"):
            sb._wrap_env("echo hi", cfg)


def test_valid_env_names_give_the_same_commands_as_before(tmp_path, monkeypatch):
    """GUARD: for ordinary names and paths nothing changes (the strings the
    code at f3a0a0a built)."""
    monkeypatch.setattr(mb, "CONDA_SH", "/opt/conda/etc/profile.d/conda.sh")
    cfg = L6b._cfg(tmp_path / "plain")
    work = mb.work_dir(cfg)
    cmd = mb._command_argv(cfg, [work / "t.txt", "--dir", work, "--dostep", "phase_linking"])[-1]
    assert cmd.startswith("source /opt/conda/etc/profile.d/conda.sh && conda activate miaplpy && ISCE_HOME=")
    assert cmd.endswith(f"cd {work} && miaplpyApp {work}/t.txt --dir {work} --dostep phase_linking")
    cfg.stack_conda_env = "isce2_gpu-1.0"
    assert sb._wrap_env("topo.py -x", cfg) == ["/usr/bin/time", "-v", "bash", "-lc",
                                                "conda run -n isce2_gpu-1.0 --no-capture-output topo.py -x"]
    cfg.stack_conda_env = ""
    assert sb._wrap_env("topo.py -x", cfg) == ["/usr/bin/time", "-v", "bash", "-lc", "topo.py -x"]


# ================================================================ SEC-03 / SEC-04
def _kml(extra_head: str = "", body_pad: str = "") -> str:
    coords = " ".join(f"{x},{y},0" for x, y in RING)
    return (f'<?xml version="1.0" encoding="UTF-8"?>{extra_head}\n'
            '<kml xmlns="http://www.opengis.net/kml/2.2"><Document><Placemark><name>Pit</name>'
            f'<description><![CDATA[<!DOCTYPE html><p>a balloon</p>]]></description>{body_pad}'
            f'<Polygon><outerBoundaryIs><LinearRing><coordinates>{coords}</coordinates></LinearRing>'
            '</outerBoundaryIs></Polygon></Placemark></Document></kml>')


def _geojson(ring=RING) -> str:
    return json.dumps({"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {"name": "Pit"}, "geometry": {"type": "Polygon", "coordinates": [ring]}}]})


def test_a_zip_bomb_kmz_is_refused_from_its_directory_before_it_is_read(tmp_path, monkeypatch):
    p = tmp_path / "pits.kmz"
    with zipfile.ZipFile(p, "w", zipfile.ZIP_DEFLATED) as z:      # ~20 KB unpacking to 20 MB
        z.writestr("doc.kml", _kml(body_pad=" " * (20 * 2 ** 20)))
    assert p.stat().st_size < 100_000
    opened = []
    real_open = zipfile.ZipFile.open
    monkeypatch.setattr(zipfile.ZipFile, "open", lambda self, *a, **k: opened.append(a) or real_open(self, *a, **k))
    with pytest.raises(ValueError, match=r"doc\.kml in pits\.kmz unpacks to 20 MB .*refused"):
        areas.load_polygon(p)
    assert opened == []                                            # judged from the directory alone


def test_a_zip_with_too_many_members_or_too_much_inside_is_refused(tmp_path, monkeypatch):
    many = tmp_path / "many.zip"
    with zipfile.ZipFile(many, "w") as z:
        z.writestr("mines.geojson", _geojson())
        for k in range(1001):
            z.writestr(f"junk/{k}.txt", "")
    with pytest.raises(ValueError, match="holds 1,002 files"):
        areas.load_polygon(many)
    monkeypatch.setattr(areas, "MAX_UNZIPPED_BYTES", 2 ** 20, raising=False)
    big = tmp_path / "big.zip"
    with zipfile.ZipFile(big, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("mines.geojson", _geojson())
        z.writestr("mines.dbf", b" " * (2 * 2 ** 20))              # a padded attribute table
    with pytest.raises(ValueError, match="unpacking to 2 MB"):
        areas.load_polygon(big)


@pytest.mark.parametrize("name, content, match", [
    ("pits.geojson", _kml(), r"pits\.geojson is not a valid GeoJSON file: its content is XML text"),
    ("pits.kml", _geojson(), r"pits\.kml is not a valid KML file: its content is JSON text"),
    ("pits.kmz", _geojson(), r"pits\.kmz is not a valid KMZ file: its content is JSON text"),
    ("pits.zip", "%PDF-1.7 a scanned map", r"pits\.zip is not a valid zip file: its content is a PDF document"),
])
def test_content_that_is_not_what_the_extension_says_is_refused_in_words(tmp_path, name, content, match):
    p = tmp_path / name
    p.write_text(content)
    with pytest.raises(ValueError, match=match):
        areas.load_polygon(p)


def test_a_kml_declaring_a_doctype_or_entity_is_refused(tmp_path):
    laughs = ('\n<!DOCTYPE kml [<!ENTITY a "ha"><!ENTITY b "&a;&a;&a;&a;&a;&a;&a;&a;&a;&a;">'
              '<!ENTITY c "&b;&b;&b;&b;&b;&b;&b;&b;&b;&b;">]>')
    p = tmp_path / "laughs.kml"
    p.write_text(_kml(extra_head=laughs).replace("<name>Pit</name>", "<name>&c;</name>"))
    with pytest.raises(ValueError, match=r"laughs\.kml is not a valid KML file: it declares a DOCTYPE or ENTITY"):
        areas.load_polygon(p)
    kmz = tmp_path / "laughs.kmz"
    with zipfile.ZipFile(kmz, "w") as z:
        z.writestr("doc.kml", p.read_text())
    with pytest.raises(ValueError, match="declares a DOCTYPE or ENTITY"):
        areas.load_polygon(kmz)
    ok = tmp_path / "balloon.kml"                                  # <!DOCTYPE html> inside CDATA is text
    ok.write_text(_kml())
    assert len(areas.load_polygon(ok).rings) == 1


@pytest.mark.parametrize("name, data, match", [
    ("cut.geojson", b'{"type": "FeatureCollection", "features": [{"type": "Fe',
     r"cut\.geojson is not a valid GeoJSON file: Unterminated string"),
    ("list.json", b'{"type": "Polygon", "coordinates": 7}', r"list is not a valid GeoJSON outline: TypeError"),
    ("latin.kml", _kml().replace("Pit", "Pét").encode("latin-1"),
     r"latin\.kml is not a valid KML file: it is not UTF-8 text"),
    ("cut.kml", _kml()[:300].encode(), r"cut\.kml is not a valid KML file: (unclosed|no element|not well-formed)"),
    ("words.kml", _kml().replace("10.02,45.0,0", "ten,forty-five").encode(),
     r"words\.kml is not a valid KML file: a coordinate is not a longitude,latitude pair"),
])
def test_malformed_files_are_a_plain_value_error(tmp_path, name, data, match):
    p = tmp_path / name
    p.write_bytes(data)
    with pytest.raises(ValueError, match=match):
        areas.load_polygon(p)


def test_too_many_vertices_or_features_are_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(areas, "MAX_VERTICES", 100, raising=False)
    monkeypatch.setattr(areas, "MAX_FEATURES", 3, raising=False)
    dense = [[10.0 + 0.0001 * k, 45.0 + (0.0001 * k) % 0.003] for k in range(200)] + [[10.0, 45.0]]
    p = tmp_path / "dense.geojson"
    p.write_text(_geojson(dense))
    with pytest.raises(ValueError, match="more than 100 vertices"):
        areas.load_polygon(p)
    k = tmp_path / "dense.kml"
    k.write_text(_kml().replace(" ".join(f"{x},{y},0" for x, y in RING), " ".join(f"{x},{y}" for x, y in dense)))
    with pytest.raises(ValueError, match="more than 100 vertices"):
        areas.load_polygon(k)
    four = tmp_path / "four.geojson"
    four.write_text(json.dumps({"type": "FeatureCollection", "features": [json.loads(_geojson())["features"][0]] * 4}))
    with pytest.raises(ValueError, match="more than 3 features"):
        areas.load_polygon(four)
    with pytest.raises(ValueError, match="more than 3 features"):
        areas.attribute_fields(four)                               # the dated layer's reader too


def test_a_typed_path_larger_than_the_upload_limit_is_refused(tmp_path, monkeypatch):
    """The dashboard's path box reads any file; the builder's 300 MB upload
    cap now applies to it (checked from the file size, before reading)."""
    monkeypatch.setattr(areas, "MAX_FILE_BYTES", 1000, raising=False)
    p = tmp_path / "statewide.geojson"
    p.write_text(_geojson() + " " * 2000)
    with pytest.raises(ValueError, match=r"statewide\.geojson is 0 MB; outline files up to 0 MB are read"):
        areas.load_polygon(str(p))


def test_the_public_mine_layers_still_load(tmp_path):
    """GUARD: the limits leave real layers alone (context/: up to 2.1 MB, 53
    features, 52,000 vertices), as a file and inside a KMZ/zip."""
    n = {}
    for f in ("cmis_all_panels_union.geojson", "coal_mines_underground_dec2016.geojson"):
        n[f] = len(areas.load_polygon(PKG / "context" / f).rings)
        z = tmp_path / (Path(f).stem + ".zip")
        with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zz:
            zz.write(PKG / "context" / f, f)
        assert len(areas.load_polygon(z).rings) == n[f]
    assert min(n.values()) >= 1 and max(n.values()) >= 100


# ================================================================ NET-06
def _server(module, tmp_path, token="the-right-token-0123456789"):
    """The module's real handler on a free port, serving its real page from tmp_path."""
    (tmp_path / module.PAGE).write_bytes((PKG / module.PAGE).read_bytes())
    H = type("H", (module._Handler,), {"port": 0, "token": token})
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), partial(H, directory=str(tmp_path)))
    H.port = httpd.server_address[1]
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, H.port


def _call(port, path, token=None, origin=None, body=b"{}"):
    h = {"Content-Type": "application/json"}
    if token is not None:
        h[map_builder.TOKEN_HEADER] = token
    if origin is not None:
        h["Origin"] = origin
    req = urllib.request.Request(f"http://localhost:{port}{path}", data=body, headers=h)
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


@pytest.mark.parametrize("module, endpoint", [(map_builder, "/api/create"), (results_viewer, "/api/save_area")])
def test_state_changing_api_calls_need_the_page_token(tmp_path, monkeypatch, module, endpoint):
    called = []
    monkeypatch.setitem(module.API, endpoint, lambda body: called.append(body) or {"ok": True})
    httpd, port = _server(module, tmp_path)
    try:
        with urllib.request.urlopen(f"http://localhost:{port}/{module.PAGE}", timeout=10) as r:
            page = r.read().decode()
            assert r.status == 200 and r.headers["Cache-Control"] == "no-store"
        token = re.search(r'<meta name="insar-token" content="([^"]+)">', page).group(1)
        # and the page sends it - from its own script files, for the results
        # viewer (19 Oct 2026: its script moved out of the page, CSP script-src 'self')
        sends = page + "".join(f.read_text() for f in getattr(module, "ASSETS", {}).values() if f.suffix == ".js")
        assert token == "the-right-token-0123456789" and "X-Insar-Token" in sends
        here = f"http://localhost:{port}"
        assert _call(port, endpoint)[0] == 403                                   # no Origin, no token
        assert _call(port, endpoint, origin=here)[0] == 403                      # its Origin, no token
        code, err = _call(port, endpoint, token="the-wrong-token-0123456789", origin=here)
        assert code == 403 and "reload the page" in err["error"]
        assert _call(port, endpoint, token=token[:-1])[0] == 403
        assert called == []
        assert _call(port, endpoint, token=token, body=b'{"x": 1}') == (200, {"ok": True})
        assert _call(port, endpoint, token=token, origin=here)[0] == 200
        assert _call(port, endpoint, token=token, origin="https://example.com")[0] == 403
        assert called == [{"x": 1}, {}]
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_each_server_session_has_its_own_token(tmp_path, monkeypatch):
    made = []

    class FakeServer:                                              # nothing listens; serve() runs as it would
        def __init__(self, addr, handler):
            made.append(handler.func)

        def serve_forever(self):
            pass
    monkeypatch.setattr(http.server, "ThreadingHTTPServer", FakeServer)
    monkeypatch.setattr(map_builder, "UPLOADS", tmp_path / "uploads")        # serve() prunes old uploads
    for module in (map_builder, results_viewer, map_builder):
        monkeypatch.setattr(module, "SERVE_DIR", tmp_path / module.__name__)
        monkeypatch.setattr(module, "_port_busy", lambda port: False)
        monkeypatch.setattr(module, "_SERVER", {})                 # a new session each time
        module.serve(port=1)
    tokens = [getattr(h, "token", "") for h in made]
    assert len(tokens) == 3 and len(set(tokens)) == 3 and min(map(len, tokens)) >= 40


# ================================================================ DAT-03
def test_the_earthdata_cookie_jar_is_private_not_in_tmp(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    for k in ("GDAL_HTTP_COOKIEFILE", "GDAL_HTTP_COOKIEJAR", *static_layers._GDAL_ENV):
        monkeypatch.setenv(k, "x")                                 # recorded, so the test's changes are undone
        monkeypatch.delenv(k)
    (home / ".cache" / "insar_pipeline" / "earthdata").mkdir(parents=True, mode=0o755)
    os.chmod(home / ".cache" / "insar_pipeline" / "earthdata", 0o755)   # an older, wider folder
    static_layers._apply_gdal_env()
    jar = Path(os.environ["GDAL_HTTP_COOKIEJAR"])
    assert os.environ["GDAL_HTTP_COOKIEFILE"] == str(jar)
    assert jar == home / ".cache" / "insar_pipeline" / "earthdata" / "gdal_edl_cookies.txt"
    assert not str(jar).startswith("/tmp/gdal")
    assert stat.S_IMODE(jar.parent.stat().st_mode) == 0o700
    assert stat.S_IMODE(jar.stat().st_mode) == 0o600
    assert jar.parent.stat().st_uid == os.getuid() == jar.stat().st_uid


# ================================================================ DAT-07
def test_run_pipeline_does_not_print_the_earthdata_user(tmp_path, monkeypatch, capsys):
    spec = importlib.util.spec_from_file_location("run_pipeline_under_test", PKG / "scripts" / "run_pipeline.py")
    rp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rp)
    home = tmp_path / "home"
    home.mkdir()
    (home / ".netrc").write_text("machine urs.earthdata.nasa.gov login someone_x7q password not-a-real-one\n")
    os.chmod(home / ".netrc", 0o600)
    monkeypatch.setenv("HOME", str(home))
    for k in ("EARTHDATA_USERNAME", "EARTHDATA_PASSWORD"):
        monkeypatch.setenv(k, "x")
        monkeypatch.delenv(k)
    assert rp.earthdata_env() is True
    assert os.environ["EARTHDATA_USERNAME"] == "someone_x7q"      # still exported for the run
    out = capsys.readouterr()
    assert "someone_x7q" not in out.out + out.err and "not-a-real-one" not in out.out + out.err


# ================================================================ OUT-03
BUILD = "0.2.1+overlapdatum"


def _patch_check(cfg, version):
    (mb.work_dir(cfg) / "patch_check.json").write_text(json.dumps(
        {"patched": True, "ratio": 1.29, "ministack": 10, "time_lag": 3, "miaplpy_version": version,
         "checked": "2026-10-16T09:00:00", "env": "miaplpy"}))


def _linked(tmp_path, stack_dates=L6b.DATES, fp_extra=None, drop=()):
    """A complete series of L6b.DATES with a fingerprint as the code before 16
    Oct 2026 wrote it (fp_extra adds or, with None values, removes keys)."""
    cfg = L6b._cfg(tmp_path)
    L6b._patches(cfg)
    L6b._phase_series(cfg)
    L6b._slc_stack(cfg, n=len(stack_dates))
    if list(stack_dates) != list(L6b.DATES[:len(stack_dates)]):
        import h5py
        with h5py.File(mb.work_dir(cfg) / "inputs" / "slcStack.h5", "a") as h:
            del h["date"]
            h["date"] = [d.encode() for d in stack_dates]
    fp = {"method": "sequential_EMI", "ministack_size": 10, "time_lag": 3, "range_window": 15, "azimuth_window": 15,
          "subset_yx": [0, 400, 0, 400], "dates": list(L6b.DATES), "n_dates": len(L6b.DATES), "status": "complete",
          "source": "config at the start of phase linking", "miaplpy_version": BUILD, "patch_check_ratio": 1.29,
          "patch_check": {"patched": True, "ratio": 1.29, "miaplpy_version": BUILD}, "written": "2026-10-14T23:51:31"}
    fp.update(fp_extra or {})
    fp = {k: v for k, v in fp.items() if v is not None and k not in drop}
    fp["sha"] = mb.fingerprint_sha(fp)
    mb.fingerprint_path(cfg).write_text(json.dumps(fp, indent=2))
    return cfg, fp


@pytest.mark.parametrize("drop", [(), ("miaplpy_version", "patch_check", "patch_check_ratio"),
                                  ("miaplpy_version", "patch_check", "patch_check_ratio", "dates", "n_dates")])
def test_a_fingerprint_written_before_still_matches(tmp_path, drop):
    """GUARD (the hard requirement): an existing project's fingerprint - the
    5-14 Oct format, or an older one recording no build or no dates - still
    matches its unchanged project, so a resume re-links nothing; and the
    hash that ties exported pairs to the series is the same as before."""
    cfg, fp = _linked(tmp_path, drop=drop)
    _patch_check(cfg, BUILD if not drop else "0.2.9+someotherbuild")  # not recorded: not compared
    assert mb.step_done(cfg, "phase_linking") is True
    assert mb.step_done(cfg, "concatenate_patches") is True
    assert mb.check_linking_fingerprint(cfg)["sha"] == fp["sha"]
    assert mb.FINGERPRINT_KEYS == ("method", "ministack_size", "time_lag", "range_window", "azimuth_window",
                                   "subset_yx")
    assert mb.fingerprint_sha({**fp, "miaplpy_version": "other"}) == mb.fingerprint_sha(fp)
    if not drop:                                                   # the value the code at f3a0a0a computed
        assert fp["sha"] == "e37801fa652eeda3"


def test_linked_dates_since_removed_from_the_slc_stack_are_refused(tmp_path):
    cfg, _ = _linked(tmp_path, stack_dates=[d for d in L6b.DATES if d != "20230128"])
    _patch_check(cfg, BUILD)
    with pytest.raises(RuntimeError, match=r"dates: 1 linked date\(s\) are no longer in inputs/slcStack\.h5 "
                                           r"\(\['20230128'\]\).*re-link") as e:
        mb.step_done(cfg, "concatenate_patches")
    assert "delete inputs/slcStack.h5 as well" not in str(e.value)    # the stack is already the new record


def test_a_series_linked_with_another_miaplpy_build_is_refused(tmp_path):
    cfg, _ = _linked(tmp_path)
    _patch_check(cfg, "0.2.2+overlapdatum")
    with pytest.raises(RuntimeError, match=r"MiaplPy build: linked with 0\.2\.1\+overlapdatum, installed now "
                                           r"0\.2\.2\+overlapdatum.*MiaplPy build 0\.2\.1\+overlapdatum back"):
        mb.step_done(cfg, "phase_linking")


def test_a_new_linking_records_this_runs_build_not_an_abandoned_ones(tmp_path):
    """A linking started with build A and abandoned before any patch finished
    left its fingerprint; the next start runs with build B. The fingerprint
    must say B, or the build comparison would stop the resume of B's own
    patches."""
    cfg = L6b._cfg(tmp_path)
    L6b._patches(cfg, flagged=0)
    _patch_check(cfg, "0.2.1+overlapdatum")
    mb._before_step(cfg, "phase_linking")
    assert json.loads(mb.fingerprint_path(cfg).read_text())["miaplpy_version"] == "0.2.1+overlapdatum"
    _patch_check(cfg, "0.2.2+overlapdatum")                       # rebuilt before the next start
    mb._before_step(cfg, "phase_linking")
    assert json.loads(mb.fingerprint_path(cfg).read_text())["miaplpy_version"] == "0.2.2+overlapdatum"
    L6b._patches(cfg, flagged=2)                                   # B links some patches, then a resume
    assert mb.step_done(cfg, "phase_linking") is False             # resumes, no refusal


def test_upload_size_cap_is_the_same_in_the_builder_and_the_reader():
    assert areas.MAX_FILE_BYTES == map_builder.MAX_UPLOAD == 300 * 2 ** 20
