"""19 Oct 2026, results viewer plan phase 1: the page's scripts, styles and
libraries moved out of the page and off the CDN (viewer_assets/). The
server serves the page and an exact list of files and nothing else on disk,
under a Content-Security-Policy that runs scripts only from itself; the
data comes from a token-checked binary endpoint."""
import http.server
import re
import threading
import urllib.error
import urllib.request
from functools import partial

import pytest

from insar_pipeline import map_builder
from insar_pipeline import results_viewer as rv

TOKEN = "the-right-token-0123456789"
CDNS = ("cdnjs.", "googleapis", "gstatic", "unpkg.", "jsdelivr", "code.jquery")


@pytest.fixture()
def server(tmp_path, monkeypatch):
    monkeypatch.setitem(rv.BINARY_API, "/api/bundle", lambda body: b"\x05\x00\x00\x00{}   " + b"\x01\x02")
    (tmp_path / rv.PAGE).write_bytes((rv.PKG / rv.PAGE).read_bytes())
    H = type("H", (rv._Handler,), {"port": 0, "token": TOKEN})
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), partial(H, directory=str(tmp_path)))
    H.port = httpd.server_address[1]
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield H.port
    httpd.shutdown()


def _get(port, path, host=None):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}")
    if host:
        req.add_header("Host", host)
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, dict(r.headers), r.read()
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers), e.read()


def test_every_listed_file_exists_and_is_served_with_its_type(server):
    for rel, f in rv.ASSETS.items():
        assert f.is_file(), rel
        code, h, body = _get(server, "/" + rel)
        assert code == 200 and body == f.read_bytes(), rel
        assert h["Content-Type"] == rv.CONTENT_TYPES[f.suffix] and h["X-Content-Type-Options"] == "nosniff", rel


@pytest.mark.parametrize("path", ["/results_viewer.py", "/assets/../results_viewer.py", "/assets/%2e%2e/results_viewer.py",
                                  "/viewer_assets/viewer_core.js", "/assets/viewer_assets/viewer_core.js",
                                  "/assets/tests/parity_runner.js", "/assets/vendor/", "/assets/", "/etc/passwd",
                                  "/assets/vendor/images/../../viewer_core.js"])
def test_nothing_else_on_disk_is_served(server, path):
    assert _get(server, path)[0] == 404


def test_another_host_gets_nothing(server):
    assert _get(server, "/assets/viewer_core.js", host="evil.example:80")[0] == 403
    assert _get(server, "/" + rv.PAGE, host="evil.example:80")[0] == 403


def test_the_page_runs_scripts_only_from_this_server(server):
    code, h, body = _get(server, "/" + rv.PAGE)
    page = body.decode()
    csp = h["Content-Security-Policy"]
    assert code == 200 and "script-src 'self'" in csp and "'unsafe-inline'" not in csp.split("script-src")[1].split(";")[0]
    assert "frame-ancestors 'none'" in csp and "connect-src 'self'" in csp
    # no inline script: every <script> loads a listed file
    srcs = re.findall(r"<script([^>]*)>(.*?)</script>", page, re.S)
    assert srcs and all('src="assets/' in attrs and not inner.strip() for attrs, inner in srcs)
    for m in re.findall(r'(?:src|href)="([^"]+)"', page):
        assert m in rv.ASSETS, m
    # the token goes into the page, never into a file every page can fetch
    assert TOKEN in page
    for f in rv.ASSETS.values():
        assert TOKEN.encode() not in f.read_bytes()


def test_no_cdn_or_font_service_is_left(server):
    for f in [rv.PKG / rv.PAGE] + [f for f in rv.ASSETS.values() if f.suffix in (".js", ".css") and "vendor" not in f.parts]:
        text = f.read_text()
        assert not [c for c in CDNS if c in text], f
    # the only places the page fetches from: this server and the listed basemaps
    app = (rv.ASSET_DIR / "viewer_app.js").read_text()
    hosts = set(re.findall(r"https://([^/'\"]+)/", app)) - {"www.openstreetmap.org"}       # an attribution link
    allowed = {h.replace("https://", "") for h in rv.TILE_HOSTS}
    assert all(h in allowed or ("*" + h[h.index("."):]) in allowed or h.startswith("{s}") for h in hosts), hosts


def test_the_data_endpoint_needs_the_token(server):
    def post(token):
        req = urllib.request.Request(f"http://localhost:{server}/api/bundle", data=b'{"config": "x"}',
                                     headers={"Content-Type": "application/json",
                                              **({map_builder.TOKEN_HEADER: token} if token else {})})
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                return r.status, r.headers["Content-Type"], r.read()
        except urllib.error.HTTPError as e:
            return e.code, None, None
    assert post(None)[0] == 403 and post("nope")[0] == 403
    code, ctype, body = post(TOKEN)
    assert code == 200 and ctype == "application/octet-stream" and body.endswith(b"\x01\x02")


def test_the_vendored_libraries_are_the_published_ones():
    """The SHA-512 hashes cdnjs publishes for Leaflet 1.9.4 and Leaflet.draw
    1.0.4 (viewer_assets/vendor/LICENSES.txt says where they came from)."""
    import base64
    import hashlib
    want = {
        "leaflet.min.js": "sha512-puJW3E/qXDqYp9IfhAI54BJEaWIfloJ7JWs7OeD5i6ruC9JZL1gERT1wjtwXFlh7CjE7ZJ+/vcRZRkIYIb6p4g==",
        "leaflet.min.css": "sha512-h9FcoyWjHcOcmEVkxOfTLnmZFWIH0iZhZT1H2TbOq55xssQGEJHEaIm+PgoUaZbRvQTNTluNOEfb1ZRy6D3BOw==",
        "leaflet.draw.js": "sha512-ozq8xQKq6urvuU6jNgkfqAmT7jKN2XumbrX1JiB3TnF7tI48DPI4Gy1GXKD/V3EExgAs1V+pRO7vwtS1LHg0Gw==",
        "leaflet.draw.css": "sha512-gc3xjCmIy673V6MyOAZhIW93xhM9ei1I+gLbmFjUHIjocENRsLX/QUE1htk5q1XV2D/iie/VQ8DXI6Vu8bexvQ==",
    }
    for name, sri in want.items():
        algo, b64 = sri.split("-", 1)
        got = base64.b64encode(hashlib.new(algo, (rv.ASSET_DIR / "vendor" / name).read_bytes()).digest()).decode()
        assert got == b64, name


def test_a_running_server_keeps_the_files_it_started_with(tmp_path, monkeypatch):
    """19 Oct 2026: a dashboard left open across a code update served the new
    page with its old code, and the page came up unstyled. serve() reads the
    page's files when it starts the server and never touches a running one."""
    monkeypatch.setattr(rv, "SERVE_DIR", tmp_path / "serve")
    monkeypatch.setattr(rv, "_SERVER", {})
    port = 18000 + (hash(str(tmp_path)) % 2000)
    while rv._port_busy(port):
        port += 1
    url = rv.serve(port)
    try:
        page_before = (tmp_path / "serve" / rv.PAGE).read_bytes()
        # the page on disk changes under the running server: a second serve() leaves it alone
        (tmp_path / "serve" / rv.PAGE).write_bytes(page_before)
        monkeypatch.setattr(rv, "PKG", tmp_path)                    # a "new release" with no page at all
        assert rv.serve(port) == url
        assert (tmp_path / "serve" / rv.PAGE).read_bytes() == page_before
        # and the files it serves are the ones read at its start
        H = rv._SERVER["httpd"].RequestHandlerClass.func
        assert H.assets["assets/viewer_core.js"] == rv.ASSETS["assets/viewer_core.js"].read_bytes()
        code, _, body = _get(port, "/assets/viewer_core.js")
        assert code == 200 and body == H.assets["assets/viewer_core.js"]
    finally:
        rv._SERVER["httpd"].shutdown()
