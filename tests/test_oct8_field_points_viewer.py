"""8 Oct 2026: the whole-field KMZ, the line-of-sight client rasters, the
signal-to-noise layer and the results viewer.

A coworker's review of the Site B KMZ: points were dense only on the
features (~400 per report area elsewhere), there was no signal-to-noise
layer, and the LOS products were not in the client set. Every new figure
must read what the delivered rasters read at that cell, and the viewer what
the KMZ reads."""
import base64
import csv
import html as htmlmod
import io
import re
import types
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import numpy as np
import pytest

from insar_pipeline import client_kmz as ck
from insar_pipeline import client_report as cr
from insar_pipeline import field_points as fp
from insar_pipeline import results_viewer as rv
from insar_pipeline.tests import test_fix_L9b as L9b


# ------------------------------------------------------------------ the grid
def test_grid_cells_one_per_block_nearest_the_centre():
    mask = np.ones((70, 70), bool)
    r, c, sp = fp.grid_cells(mask, 15.0, 100.0)              # k = 7 cells
    assert sp == pytest.approx(105.0)
    assert r.size == 100                                      # 10 x 10 blocks
    assert set(r % 7) == {3} and set(c % 7) == {3}            # the centre cell of each block
    mask[:, :35] = False
    r2, c2, _ = fp.grid_cells(mask, 15.0, 100.0)
    assert r2.size == 50 and c2.min() >= 35


def test_grid_cells_widens_to_the_cap():
    mask = np.ones((400, 400), bool)
    r, _, sp = fp.grid_cells(mask, 15.0, 100.0, max_points=1000)
    assert r.size <= 1000 and sp > 100.0


def test_svg_chart_is_valid_svg_and_decimated():
    t = np.linspace(0, 10, 400)
    y = -0.05 * t + np.random.default_rng(1).normal(0, 0.05, t.size)
    s = ck.svg_chart(t, y, "x: movement", centre_in=0 * t, half_in=0.06 * t, start_year=2016.0)
    root = ET.fromstring(s)
    line = [e for e in root.iter() if e.tag.endswith("path")][0]
    assert len(line.get("d").split()) <= ck.SVG_MAX_VERTICES
    assert "2017" in s and "2025" in s                      # every other year on a ten-year record
    assert ck.svg_chart(t[:2], y[:2], "too short") == ""


def test_vectorised_trend_zeroing_matches_the_charts():
    rng = np.random.default_rng(3)
    t = np.linspace(0, 9, 120)
    S = (rng.normal(0, 3, (20, t.size)) + 4.0 * t[None, :] + 7.0).astype(np.float32)
    S[3, :50] = np.nan
    S[5, :] = np.nan
    fd = {"series_los": S, "up": np.full(20, 0.77), "meta": {"t": list(t), "bg_v": -1.2}}
    A = fp.series_vertical_in_rows(fd, np.arange(20))
    B = np.vstack([fp.series_vertical_in(fd, j) for j in range(20)])
    np.testing.assert_allclose(A, B, atol=1e-9, equal_nan=True)


# ------------------------------------------------------------------ the built project
@pytest.fixture(scope="module")
def built(tmp_path_factory):
    root = tmp_path_factory.mktemp("oct8")
    cfg = L9b.make_project(root)
    cr.build(cfg)                                             # the report and its rasters
    L9b._write_too_fast(cfg)
    D = cfg.project_dir / "deliverables"
    before = sorted(p.name for p in D.iterdir())
    kmz = ck.build(cfg, out_dir=root / "preview")
    after = sorted(p.name for p in D.iterdir())
    z = zipfile.ZipFile(kmz)
    kml_all = z.read("doc.kml").decode()
    import rasterio
    rasters = {}
    for name in ("line_of_sight_velocity_in_per_yr.tif", "rate_signal_to_noise.tif"):
        with rasterio.open(D / name) as r:
            rasters[name] = r.read(1).astype(float)
    cum = sorted(D.glob("line_of_sight_cumulative_displacement_in_*.tif"))
    assert len(cum) == 1
    with rasterio.open(cum[0]) as r:
        rasters["cum"] = r.read(1).astype(float)
    return types.SimpleNamespace(cfg=cfg, D=D, root=root, kmz=kmz, z=z, doc=z.read("doc.kml").decode(),
                                 kml_all=kml_all, rasters=rasters, before=before, after=after,
                                 npz=root / "preview" / "viewer" / fp.VIEWER_FILE)


def _points(kml):
    return ck.expand_placemarks(kml)


def _plain(s):
    """The balloon's words, without its tags or its chart."""
    return htmlmod.unescape(re.sub(r"<[^>]+>", " ", re.sub(r"<svg.*?</svg>", " ", s, flags=re.S)))


def _cell(lat, lon):
    return int(round((float(lat) - L9b.Y0) / L9b.DY - 0.5)), int(round((float(lon) - L9b.X0) / L9b.DX - 0.5))


def test_points_cover_the_whole_field_not_just_features(built):
    pts = [p for p in _points(built.kml_all) if not p[0].startswith("s")]
    cells = {_cell(p[3], p[2]) for p in pts}
    # the synthetic grid's cells are ~100 m: the field grid takes every measured cell of the study area
    rows = np.array([r for r, _ in cells]); cols = np.array([c for _, c in cells])
    assert len(cells) > 0.6 * L9b.NY * L9b.NX
    assert rows.min() < 10 and rows.max() > L9b.NY - 10 and cols.min() < 10 and cols.max() > L9b.NX - 10
    # 13 Oct 2026: every point in doc.kml, always loaded (the 8 Oct tiles behind Regions looked blank)
    assert "<NetworkLink>" not in built.doc and "<Region>" not in built.doc
    assert ck.DOT in built.z.namelist() and f"<href>{ck.DOT}</href>" in built.doc


def test_balloon_line_of_sight_and_snr_rows_read_the_rasters(built):
    v_r, s_r, c_r = (built.rasters[k] for k in ("line_of_sight_velocity_in_per_yr.tif", "rate_signal_to_noise.tif", "cum"))
    n = 0
    for style, desc, lon, lat in _points(built.kml_all):
        txt = _plain(desc)
        if "Not measurable" in txt:
            continue
        r, c = _cell(lat, lon)
        m = re.search(r"Line of sight, per year \(vs ordinary \w+\)\s+([-+]?\d+\.\d+) in", txt)
        assert m, txt[:400]
        assert float(m.group(1)) == pytest.approx(v_r[r, c], abs=0.0051)
        m = re.search(r"Line of sight, \d+ \w+ \d{4} to \d+ \w+ \d{4}\s+([-+]?\d+\.\d+) in", txt)
        assert float(m.group(1)) == pytest.approx(c_r[r, c], abs=0.051)
        m = re.search(r"Signal-to-noise ratio\s+(\d+\.\d+)", txt)
        if abs(s_r[r, c] - 1.96) > 0.01:
            assert float(m.group(1)) == pytest.approx(s_r[r, c], abs=0.0051)
            assert ("(stands out)" in txt.split("Signal ÷ threshold")[1][:40]) == (s_r[r, c] >= 1.96)
        assert "<svg" in desc
        n += 1
    assert n > 1000


def test_map_layers_full_resolution_with_every_cell_on(built):
    """The whole field shows when the file opens: every measured cell (ON),
    and line of sight, signal-to-noise and the report's map one at a time."""
    from PIL import Image
    names = built.z.namelist()
    for k in ("cells", "los_velocity", "los_change", "snr"):
        assert f"layers/{k}_0.png" in names, k
    for f in ("legend_los_velocity.png", "legend_los_change.png", "legend_snr.png"):
        assert f in names, f
    im = Image.open(io.BytesIO(built.z.read("layers/cells_0.png")))
    assert im.size == (L9b.NX, L9b.NY)                         # one pixel per cell
    alpha = np.asarray(im)[..., 3]
    assert (alpha > 0).mean() > 0.9                            # nearly every cell is drawn
    folder = re.search(r"<Folder><name>Map layers \(one at a time\)</name>(.*)</Folder><Folder><name>GNSS", built.doc,
                       re.S).group(1)
    assert "radioFolder" in built.doc and folder.startswith("<open>1</open><styleUrl>#radio</styleUrl>")
    on = re.findall(r"<Folder><name>([^<]+)</name><visibility>1</visibility>", folder)
    assert len(on) == 1 and re.fullmatch(r"Vertical movement, every measured \d+ m cell", on[0])
    assert "Signal-to-noise ratio" in folder and "In this file" in built.doc
    assert "<LookAt>" in built.doc


def test_shared_balloon_is_point_description_word_for_word(built):
    """Google Earth's text (the style's balloon filled from the point's
    values) is the balloon point_description writes."""
    fd = fp.load(built.npz)
    m = fd["meta"]
    bins = [(lo, hi, tuple(rgb), lab) for lo, hi, rgb, lab in m["bins"]]
    field = re.search(r"<Folder><name>Measurement points: .*?</Folder>", built.doc, re.S).group(0)
    styles = "".join(re.findall(r"<Style id='\w+'>.*?</Style>", built.doc, re.S))
    shown = {(f"{float(a):.6f}", f"{float(b):.6f}"): (st, d) for st, d, b, a in _points(styles + field)}
    n = 0
    for j in np.flatnonzero(fd["group"] == 0)[::17]:
        p = ck.point_parts(fd, int(j), m["names"][0], bins, bool(m["relative_colours"]), m["dates"][0],
                           m["dates"][-1], ck.decimal_year(m["dates"][0]))
        st, d = shown[(f"{p['lat']:.6f}", f"{p['lon']:.6f}")]
        assert st == p["style"]
        assert d == ck._no_neg_zero(p["desc"])
        n += 1
    assert n > 100
    pm = re.findall(r"<Placemark><styleUrl>#b\w+</styleUrl>.*?</Placemark>", built.doc, re.S)
    assert sum(map(len, pm)) / len(pm) < 2200                  # bytes a point: the whole field fits in doc.kml


def test_preview_writes_the_viewer_copy_beside_it_not_in_deliverables(built):
    assert built.npz.exists()
    assert built.before == built.after
    fd = fp.load(built.npz)
    assert fd["series_los"].shape == (fd["lat"].size, len(fd["meta"]["dates"]))
    assert fd["meta"]["numbers_hash"]


# ------------------------------------------------------------------ the viewer
@pytest.fixture
def viewer(built, monkeypatch, tmp_path):
    """The viewer on the preview's table, with the project's config as a yaml."""
    monkeypatch.setattr(fp, "viewer_path", lambda cfg, out_dir=None: built.npz)
    cfgy = tmp_path / "testsite.yaml"
    c = built.cfg
    cfgy.write_text(
        f"backend: miaplpy\nproject_dir: {c.project_dir}\nbbox: {list(c.bbox)}\nstart_date: '{c.start_date}'\n"
        f"end_date: '{c.end_date}'\nproject_title: Testsite\nproject_location: Test County, Indiana\n"
        "track: 92\nstack_virtual_merge: false\nmintpy_deramp: 'no'\n"
        "report_areas:\n" + "".join(f"- {a}\n" for a in c.report_areas) + "display_units: in\n")
    return types.SimpleNamespace(config=str(cfgy), built=built)


def test_viewer_point_is_the_kmz_balloon(viewer):
    d = rv.api_load({"config": viewer.config})
    lat = np.frombuffer(base64.b64decode(d["lat"]), "<f8")
    lon = np.frombuffer(base64.b64decode(d["lon"]), "<f8")
    # the field's points; a feature's extra sample can share a cell with one (the field folder only)
    field = re.search(r"<Folder><name>Measurement points: .*?</Folder>", viewer.built.doc, re.S).group(0)
    styles = "".join(re.findall(r"<Style id='\w+'>.*?</Style>", viewer.built.doc, re.S))
    kmz_pts = {(f"{float(a):.6f}", f"{float(b):.6f}"): desc for _, desc, b, a in _points(styles + field)}
    group = np.frombuffer(base64.b64decode(d["group"]), "u1")
    checked = 0
    for j in np.flatnonzero(group == 0)[::max(1, d["n"] // 25)]:
        p = rv.api_point({"config": viewer.config, "j": j})
        key = (f"{lat[j]:.6f}", f"{lon[j]:.6f}")
        if key not in kmz_pts:
            continue
        assert _plain(p["html"]).split() == _plain(ck._no_neg_zero(kmz_pts[key])).split()
        checked += 1
    assert checked > 10


def test_polygon_average_exports_and_save(viewer):
    fd = fp.load(viewer.built.npz)
    w, s, e, n = viewer.built.cfg.bbox
    ring = [[w + 0.2 * (e - w), s + 0.2 * (n - s)], [w + 0.6 * (e - w), s + 0.2 * (n - s)],
            [w + 0.6 * (e - w), s + 0.6 * (n - s)], [w + 0.2 * (e - w), s + 0.6 * (n - s)]]
    st = rv.api_polygon({"config": viewer.config, "rings": [ring]})
    inside = ((fd["lon"] > ring[0][0]) & (fd["lon"] < ring[1][0]) & (fd["lat"] > ring[0][1]) & (fd["lat"] < ring[2][1])
              & (fd["group"] == 0))
    assert st["n"] == int(inside.sum())
    ok = inside & ~fd["fast"]
    assert st["vertical_in_yr"] == pytest.approx(float(np.nanmean(fd["rate"][ok])) / 25.4, abs=1e-9)
    c = rv.api_export_csv({"config": viewer.config, "rings": [ring], "name": "North Field"})
    rows = [r for r in csv.reader(io.StringIO(c["text"])) if r and not r[0].startswith("#")]
    assert rows[0][0] == "latitude" and len(rows) == st["n"] + 2 and rows[-1][0] == "polygon average"
    k = rv.api_export_kmz({"config": viewer.config, "rings": [ring], "name": "North Field"})
    kz = zipfile.ZipFile(io.BytesIO(base64.b64decode(k["b64"])))
    doc = kz.read("doc.kml").decode()
    assert doc.count("<Placemark>") == st["n"] + 1 and "North Field" in doc and ck.DOT in kz.namelist()
    assert len(ck.expand_placemarks(doc)) == st["n"]
    # save as a report area: the config gains it (backed up), the outline is in context/
    before = Path(viewer.config).read_text()
    with pytest.raises(ValueError):
        rv.save_report_area(viewer.config, "north_field_v2", [ring])            # reads like a file name
    with pytest.raises(ValueError):
        rv.save_report_area(viewer.config, "Test Field", [ring])                # already a report area
    assert Path(viewer.config).read_text() == before
    out = rv.save_report_area(viewer.config, "North Field", [ring])
    from insar_pipeline.config import load_config
    got = dict(cr.parse_areas(load_config(viewer.config)))
    assert "North Field" in got and Path(out["saved"]).exists() and Path(out["backup"]).read_text() == before
    assert list(got)[:3] == ["Full Satellite Imaging", "Test Field", "Wet Pit"]
    Path(out["saved"]).unlink()


def test_package_puts_the_line_of_sight_rasters_in_data():
    from insar_pipeline import package
    for name in ("line_of_sight_velocity_in_per_yr.tif", "line_of_sight_cumulative_displacement_in_20160105_20260617.tif"):
        assert not package.FORBIDDEN_CLIENT.search(name), name
        assert any(p.match(name) for p, _ in package.DATA_FILES), name
    assert package.FORBIDDEN_CLIENT.search("los_velocity_in_per_year.tif")
    assert package.SKIP_DIRS.match("viewer")


# ------------------------------------------------------------------ 26 Oct 2026: a coworker's review of the Site E KMZ
def test_kmz_points_are_off_on_open_and_each_balloon_names_its_point(built):
    """The points made Google Earth slow to open and every balloon was headed
    'Full Satellite Imaging' (its report area), so it read as the area's
    overview; the 15 m cell layer is what shows on open."""
    for name in ("Measurement points: vertical movement", "Feature points"):
        m = re.search(r"<Folder><name>" + re.escape(name) + r"[^<]*</name>(<visibility>\d</visibility>)?", built.doc)
        if m:
            assert m.group(1) == "<visibility>0</visibility>", name
    assert re.search(r"<name>Vertical movement, every measured \d+ m cell</name><visibility>1</visibility>", built.doc)
    pts = [p for p in _points(built.kml_all) if "Not measurable" not in p[1]]
    for style, desc, lon, lat in pts[:20]:
        head = " ".join(_plain(desc).split("Vertical")[0].split())
        assert re.match(r"Point \d+\.\d{5} [NS], \d+\.\d{5} [EW] in \S", head), head


def test_viewer_exports_every_cell_inside_a_polygon(viewer, monkeypatch):
    """The viewer's 'KMZ, every 15 m cell': every measured product-grid cell
    inside the polygon, computed from the products by the client KMZ's own
    gather, so a cell the ~100 m table also holds reads the same in both;
    over a date window it follows the window; too large a polygon is refused."""
    fd = fp.load(viewer.built.npz)
    w, s, e, n = viewer.built.cfg.bbox
    ring = [[w + 0.2 * (e - w), s + 0.2 * (n - s)], [w + 0.6 * (e - w), s + 0.2 * (n - s)],
            [w + 0.6 * (e - w), s + 0.6 * (n - s)], [w + 0.2 * (e - w), s + 0.6 * (n - s)]]
    body = {"config": viewer.config, "rings": [ring], "name": "North Field"}
    part = rv.api_export_kmz(body)
    full = rv.api_export_kmz(dict(body, resolution="full"))
    assert full["filename"] == "North_Field_every_cell.kmz"
    doc = zipfile.ZipFile(io.BytesIO(base64.b64decode(full["b64"]))).read("doc.kml").decode()
    got = {(f"{float(a):.5f}", f"{float(b):.5f}"): _plain(d) for _, d, b, a in ck.expand_placemarks(doc)}
    # the synthetic grid's cells are ~100 m, so here every measured cell is also a ~100 m point
    ft = rv._full_table(viewer.config, [ring], fd)
    assert full["points"] == ft["row"].size == len(got) >= part["points"] > 10
    pdoc = zipfile.ZipFile(io.BytesIO(base64.b64decode(part["b64"]))).read("doc.kml").decode()
    same = 0
    for _, d, b, a in ck.expand_placemarks(pdoc):
        k = (f"{float(a):.5f}", f"{float(b):.5f}")
        if k in got:
            assert got[k].split() == _plain(d).split()
            same += 1
    assert same == part["points"]
    dates = fd["meta"]["dates"]
    win = rv.api_export_kmz(dict(body, resolution="full", window={"first": dates[len(dates) // 3], "last": dates[-1]}))
    assert win["points"] == full["points"] and win["filename"].endswith("_every_cell.kmz") and win["b64"] != full["b64"]
    monkeypatch.setattr(rv, "FULL_RES_MAX_CELLS", 3)
    with pytest.raises(ValueError, match="at most 3"):
        rv.api_export_kmz(dict(body, resolution="full"))


def test_viewer_map_cells_and_one_cells_balloon_are_the_points(viewer, monkeypatch):
    """26 Oct 2026 (Kyle: every 15 m cell on the viewer map when zoomed in,
    clickable): api_cells gives every measured cell in view with its own
    figures, and api_cell a cell's balloon and series. A cell that is also a
    ~100 m point reads exactly what that point reads."""
    from insar_pipeline import viewer_bundle as vb
    fd = fp.load(viewer.built.npz)
    w, s, e, n = viewer.built.cfg.bbox
    r = rv.api_cells({"config": viewer.config, "bbox": [w, s, e, n]})
    dec = lambda k, dt: np.frombuffer(base64.b64decode(r[k]), dt)                          # noqa: E731
    rows, cols, vert, snr = dec("row", "<i4"), dec("col", "<i4"), dec("vertical", "<f4"), dec("snr", "<f4")
    assert r["n"] == rows.size >= (fd["group"] == 0).sum()
    key = {(int(a), int(b)): k for k, (a, b) in enumerate(zip(rows, cols))}
    T = len(fd["meta"]["dates"])
    field = np.flatnonzero((fd["group"] == 0) & ~fd["fast"] & np.isfinite(fd["rate"]))
    for j in field[:: max(1, field.size // 15)]:
        k = key[(int(fd["row"][j]), int(fd["col"][j]))]
        assert vert[k] == pytest.approx(fd["rate"][j], abs=1e-4)
        assert snr[k] == pytest.approx(fd["snr"][j], abs=1e-4, nan_ok=True)
    j = int(field[len(field) // 2])
    c = rv.api_cell({"config": viewer.config, "row": int(fd["row"][j]), "col": int(fd["col"][j])})
    p = rv.api_point({"config": viewer.config, "j": j})
    assert _plain(c["html"]).split() == _plain(p["html"]).split()
    assert c["html"].startswith("<b>Point ")
    own = np.array([np.nan if v is None else v for v in c["own_vert_mm"]])
    assert np.allclose(own, vb.point_series(fd, j, 0, T - 1, True), atol=1e-3, equal_nan=True)
    assert c["vert"] == p["vert"]
    with pytest.raises(ValueError):
        rv.api_cell({"config": viewer.config, "row": -1, "col": 0})
    monkeypatch.setattr(rv, "CELLS_MAX", 3)
    assert rv.api_cells({"config": viewer.config, "bbox": [w, s, e, n]})["too_many"] == r["n"]


def test_polygon_panel_from_every_cell(viewer, monkeypatch):
    """26 Oct 2026 (Kyle: a small polygon showed one or two ~100 m points):
    the panel's figures from every measured cell inside (api_polygon_cells).
    The synthetic grid's cells are ~100 m, so every cell is a point here and
    the two must agree - over the whole record and over a date window."""
    fd = fp.load(viewer.built.npz)
    w, s, e, n = viewer.built.cfg.bbox
    ring = [[w + 0.2 * (e - w), s + 0.2 * (n - s)], [w + 0.6 * (e - w), s + 0.2 * (n - s)],
            [w + 0.6 * (e - w), s + 0.6 * (n - s)], [w + 0.2 * (e - w), s + 0.6 * (n - s)]]
    dates = fd["meta"]["dates"]
    monkeypatch.setattr(rv, "POLYGON_LINES", 10 ** 6)          # every cell's line, to check the mean
    for win in (None, {"first": dates[len(dates) // 3], "last": dates[-1]}):
        body = {"config": viewer.config, "rings": [ring], **({"window": win} if win else {})}
        cells = rv.api_polygon_cells(body)
        W, fdw = rv._window(fd, body)
        pts = rv._poly_stats(fdw, rv._inside(fd, [ring]))
        assert cells["source"] == "cells" and cells["n"] == pts["n"] > 10
        for k in ("vertical_in_yr", "vertical_in_yr_p10", "vertical_in_yr_p90", "los_in_yr", "los_change_in",
                  "median_snr", "percent_standing_out"):
            assert cells[k] == pytest.approx(pts[k], abs=1e-6, nan_ok=True), k
        assert len(cells["series_vertical_mm"]) == len(W and fdw["meta"]["dates"])
        assert np.allclose([np.nan if v is None else v for v in cells["series_vertical_mm"]],
                           [np.nan if v is None else v * 25.4 for v in pts["series_vertical_in"]], atol=0.01, equal_nan=True)
        assert len(cells["cells_vertical_mm"]) == cells["n"] - cells["n_too_fast"]
        # every cell's own series, drawn grey behind the average: the average is their mean
        lines = np.array([[np.nan if v is None else v for v in ln] for ln in cells["cells_series_vertical_mm"]])
        assert lines.shape == (cells["cells_series_of"], len(cells["series_vertical_mm"]))
        with np.errstate(invalid="ignore"):
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                mean = np.nanmean(lines, axis=0)
        assert np.allclose(mean, [np.nan if v is None else v for v in cells["series_vertical_mm"]], atol=0.06,
                           equal_nan=True)
    c = rv.api_export_csv({"config": viewer.config, "rings": [ring], "name": "North Field", "resolution": "full"})
    assert c["filename"] == "North_Field_every_cell.csv"
    # above POLYGON_LINES an even sample is drawn, and the chart says how many of how many
    monkeypatch.setattr(rv, "POLYGON_LINES", 5)
    few = rv.api_polygon_cells({"config": viewer.config, "rings": [ring]})
    assert len(few["cells_series_vertical_mm"]) == 5 < few["cells_series_of"]


def test_a_wells_own_series_is_its_map_cells(viewer):
    """11 Nov 2026 (Kyle: the well's chart on the well, not "the nearest
    point"): api_well gives the measured map cell the well sits in, as
    api_cell gives it; a well just off a cell's centre still lands in that
    cell, and a well far off the grid is refused."""
    fd = fp.load(viewer.built.npz)
    field = np.flatnonzero((fd["group"] == 0) & ~fd["fast"] & np.isfinite(fd["rate"]))
    j = int(field[len(field) // 2])
    r, c = int(fd["row"][j]), int(fd["col"][j])
    cell = rv.api_cell({"config": viewer.config, "row": r, "col": c})
    w = rv.api_well({"config": viewer.config, "lat": cell["lat"] + 1e-5, "lon": cell["lon"] - 1e-5})
    assert (w["row"], w["col"]) == (r, c) and w["own_vert_mm"] == cell["own_vert_mm"]
    assert 0 < w["offset_m"] < 3
    with pytest.raises(ValueError):
        rv.api_well({"config": viewer.config, "lat": 0.0, "lon": 0.0})
