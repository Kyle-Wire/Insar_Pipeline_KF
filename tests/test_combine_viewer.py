"""A combined project in the results viewer (combine_viewer.py, 28 Oct 2026):
the series solve recovers known vertical and east-west movement from two
passes on different dates; the viewer serves the combined table with the
vertical in its vertical slot (up = 1) and east-west where a single project
has line of sight; 15 m cells and polygons come from the combined fields;
date windows are off and exports refused, each said so."""
from datetime import date, timedelta

import numpy as np
import pytest

from insar_pipeline import combine, combine_viewer as cv, decompose as dc, results_viewer as rv, viewer_bundle as vb
import test_combine_app as tca
import test_fix_L9a as l9

home = l9.home
ASC, DESC = (-0.60, 0.79), (0.58, 0.80)          # (east, up) of each pass's look vector


def test_series_solve_recovers_vertical_and_east_west_from_two_passes():
    d0 = date(2017, 1, 1)
    da = [d0 + timedelta(days=12 * i) for i in range(60)]
    dd = [d0 + timedelta(days=5 + 12 * i) for i in range(-1, 61)]          # another day, one revisit either side
    yrs = lambda ds: np.array([(d - d0).days / 365.25 for d in ds])         # noqa: E731
    up_rate, east_rate = np.array([-10.0, 0.0, 4.0]), np.array([6.0, -3.0, 0.0])

    def los(eu, ds):
        t = yrs(ds)
        return eu[0] * east_rate[:, None] * t + eu[1] * up_rate[:, None] * t + 7.0     # an offset: zeroed away
    epochs = [(i, dc.interpolation_plan([x.strftime("%Y%m%d") for x in dd], d, 24)) for i, d in enumerate(da)]
    assert all(p is not None for _, p in epochs)
    n = up_rate.size
    U, E = cv.solve_series(los(ASC, da), los(DESC, dd), epochs, (np.full(n, ASC[0]), np.full(n, ASC[1])),
                           (np.full(n, DESC[0]), np.full(n, DESC[1])))
    t = yrs(da)
    assert U[:, 0] == pytest.approx(0) and E[:, 0] == pytest.approx(0)
    assert U == pytest.approx(up_rate[:, None] * t, abs=1e-4)
    assert E == pytest.approx(east_rate[:, None] * t, abs=1e-4)


# ---------------------------------------------------------------- the viewer, on a fake combined project
def _combined_project(root, monkeypatch):
    a, d = tca._pair(root)
    monkeypatch.setattr(combine, "run", tca._fake_run([]))
    from insar_pipeline import app_server as app
    c = app.api_combine_with({"config": str(a), "partner": str(d)})["config"]
    return c


def _table(n=6, T=10):
    t = np.linspace(0, 3, T)
    rng = np.random.default_rng(0)
    up = np.array([-4.0, -2.0, 0.0, 1.0, 3.0, 5.0])[:n]
    east = np.array([6.0, 0.0, -2.0, 1.0, 0.0, -6.0])[:n]
    fd = {"row": np.arange(n, dtype=np.int32), "col": np.arange(n, dtype=np.int32), "group": np.zeros(n, np.int16),
          "lat": 38.35 + 0.001 * np.arange(n), "lon": np.full(n, -87.65), "up": np.ones(n),
          "los_rel": up, "rel_v": up, "rate": up - 1.0, "sd_v": np.full(n, 2.0), "snr": np.abs(up) / 2.0,
          "tc": np.full(n, 0.8, np.float32), "unwrap": np.full(n, np.nan, np.float32), "fast": np.zeros(n, bool),
          "change_los": up * 3, "series_los": (up[:, None] * t + rng.normal(0, .1, (n, T))).astype(np.float32),
          "kcls": np.full(n, 3, np.int8), "east": east, "east_change": east * 3,
          "series_east": (east[:, None] * t).astype(np.float32)}
    fd["meta"] = {"combined": True, "title": "demo combined", "names": ["Study area"],
                  "dates": [f"2017{m:02d}01" for m in range(1, T + 1)], "t": [float(x) for x in t],
                  "t_fit": [float(x) for x in t], "fit_exclude": [], "span": 3.0, "anchored": True, "bg_v": -1.0,
                  "bg_u": 1.0, "ground": "ground (mostly grassland)", "relative_colours": True,
                  "bins": [[-1e9, 0, [1, 0, 0], "<0"], [0, 1e9, [0, 0, 1], ">0"]], "areas": [], "features": [],
                  "spacing_m": 100.0, "numbers_hash": "", "window_note": cv.WINDOW_NOTE,
                  "sources": [{"direction": "ascending", "project": "demo"},
                              {"direction": "descending", "project": "demo_desc"}]}
    return fd


def test_the_viewer_serves_a_combined_table(home, monkeypatch):
    c = _combined_project(home, monkeypatch)
    cc = combine.load(c)
    cv.save(_table(), cv.viewer_path(cc))
    row = next(p for p in rv.api_projects({})["projects"] if p["name"] == "demo_combined")
    assert row["ready"]
    _, fd, stale = rv._table(c)
    assert stale is None and fd["meta"]["combined"] and "east" in fd
    h, arrs = vb.decode(rv.bundle_bytes(c))
    m = h["meta"]
    assert m["route"] == "combined" and m["windows"] is False and m["window_note"] == cv.WINDOW_NOTE
    assert arrs["east"] == pytest.approx(fd["east"]) and arrs["up"] == pytest.approx(1.0)
    pt = rv.api_point({"config": c, "j": 0})
    east_in = np.array(pt["los"])
    assert np.polyfit(fd["meta"]["t"], east_in, 1)[0] == pytest.approx(6.0 / 25.4, abs=1e-3)   # east-west, not LOS
    assert "East-west" in pt["html"] and "-0.00" not in pt["html"]
    ring = [[-87.66, 38.3495], [-87.64, 38.3495], [-87.64, 38.3525], [-87.66, 38.3525]]
    pg = rv.api_polygon({"config": c, "rings": [ring]})
    assert pg["n"] == 3 and pg["los_in_yr"] == pytest.approx(np.mean([6.0, 0.0, -2.0]) / 25.4)
    assert pg["vertical_in_yr"] == pytest.approx(np.mean([-5.0, -3.0, -1.0]) / 25.4)
    # exports (28 Oct 2026, Kyle): the vertical and the east-west of the points inside
    csv_ = rv.api_export_csv({"config": c, "rings": [ring], "name": "Area A2"})
    rows = [r for r in csv_["text"].splitlines() if not r.startswith("#")]
    assert csv_["filename"] == "Area_A2_combined_points.csv"
    assert rows[0].startswith("latitude,longitude,vertical_in_per_yr,vertical_relative_in_per_yr,east_west_in_per_yr")
    assert len(rows) == 1 + 3 + 1                                          # header, 3 points, the average
    avg = rows[-1].split(",")
    assert float(avg[2]) == pytest.approx(pg["vertical_in_yr"], abs=1e-4) and float(avg[4]) == pytest.approx(
        pg["los_in_yr"], abs=1e-4)
    k = rv.api_export_kmz({"config": c, "rings": [ring], "name": "Area A2"})
    import base64, io, zipfile, xml.etree.ElementTree as ET                # noqa: E401
    doc = zipfile.ZipFile(io.BytesIO(base64.b64decode(k["b64"]))).read("doc.kml")
    root = ET.fromstring(doc)
    assert k["points"] == 3 and len(root.findall(".//{http://www.opengis.net/kml/2.2}Placemark")) == 4
    assert b"East-west, per year (indicative)" in doc
    with pytest.raises(ValueError, match="not available for the combined view"):
        rv.api_save_area({"config": c, "rings": [ring], "name": "x"})
    # the whole field, as Combine writes it
    out = cv.write_kmz(combine.load(c), fd)
    assert out.name == "demo_combined_DRAFT.kmz"
    whole = zipfile.ZipFile(out).read("doc.kml")
    assert len(ET.fromstring(whole).findall(".//{http://www.opengis.net/kml/2.2}Placemark")) == fd["lat"].size


class _FakeSources(cv.Sources):
    """Sources without the projects: a 20 x 20 common grid with Up and East."""

    def __init__(self):
        attrs = {"X_FIRST": "-87.66", "Y_FIRST": "38.36", "X_STEP": "0.0005", "Y_STEP": "-0.0005",
                 "WIDTH": "20", "LENGTH": "20"}
        self.G = type("G", (), {"grid": dc.Grid.from_attrs(attrs, (20, 20)), "attrs": attrs})()
        yy, xx = np.mgrid[0:20, 0:20]
        good = np.ones((20, 20), bool)
        good[0, 0] = False
        self.F = {"good": good, "up": (-0.5 * xx).astype(np.float32), "east": (0.3 * yy).astype(np.float32),
                  "tc": np.full((20, 20), 0.8, np.float32)}
        self.sd_up = np.full((20, 20), 2.0)
        self.kcls = np.full((20, 20), 3)
        self.lon_c, self.lat_c = self.G.grid.centres()
        self.bg_v, self.anchored, self.bg_u, self.span = -1.0, True, 1.0, 3.0
        self.t = np.linspace(0, 3, 10)
        self.dates = [f"2017{m:02d}01" for m in range(1, 11)]
        self.cc = type("CC", (), {"name": "demo combined"})()

    def usable(self, rows, cols):
        return self.F["good"][np.asarray(rows, int), np.asarray(cols, int)]

    def series(self, rows, cols):
        f = self.figures(rows, cols)
        U = (f["los_rel"][:, None] * self.t).astype(np.float32)
        E = (f["east"][:, None] * self.t).astype(np.float32)
        return {"up3": U, "east3": E, "up1": U, "east1": E}

    def meta(self):
        return {"combined": True, "dates": self.dates, "t": [float(x) for x in self.t], "span": self.span,
                "anchored": True, "bg_v": self.bg_v, "bg_u": self.bg_u, "spacing_m": 100.0}


def test_cells_and_polygons_come_from_the_combined_fields(home, monkeypatch):
    c = _combined_project(home, monkeypatch)
    cv.save(_table(), cv.viewer_path(combine.load(c)))
    S = _FakeSources()
    monkeypatch.setattr(rv, "_combined_sources", lambda config: S)
    rv._CELL_TABLES.clear()
    import base64
    cs = rv.api_cells({"config": c, "bbox": [-87.66, 38.35, -87.65, 38.36]})
    assert cs["n"] == 399                                                    # the one unusable cell left out
    los = np.frombuffer(base64.b64decode(cs["los"]), "<f4")
    rows = np.frombuffer(base64.b64decode(cs["row"]), "<i4")
    assert los == pytest.approx(0.3 * rows)                                  # east-west in the LOS slot
    one = rv.api_cell({"config": c, "row": 4, "col": 6})
    assert one["rate_vert_mm_yr"] == pytest.approx(-3.0 - 1.0) and one["rate_los_mm_yr"] == pytest.approx(1.2)
    assert len(one["own_vert_mm"]) == 10
    with pytest.raises(ValueError, match="both passes"):
        rv.api_cell({"config": c, "row": 0, "col": 0})
    ring = [[-87.6599, 38.3599], [-87.6551, 38.3599], [-87.6551, 38.3551], [-87.6599, 38.3551]]      # 10 x 10 cells
    pc = rv.api_polygon_cells({"config": c, "rings": [ring]})
    assert pc["source"] == "cells" and pc["n"] == 99
    assert pc["los_in_yr"] == pytest.approx(np.mean([0.3 * r for r in range(10) for k in range(10)
                                                     if (r, k) != (0, 0)]) / 25.4, rel=1e-4)
    assert len(pc["cells_series_vertical_mm"]) == 99 and len(pc["series_los_mm"]) == 10
    k = rv.api_export_kmz({"config": c, "rings": [ring], "name": "x", "resolution": "full"})
    assert k["points"] == 99 and k["filename"] == "x_combined_every_cell.kmz"
    csv_ = rv.api_export_csv({"config": c, "rings": [ring], "name": "x", "resolution": "full"})
    assert len([r for r in csv_["text"].splitlines() if not r.startswith("#")]) == 1 + 99 + 1
