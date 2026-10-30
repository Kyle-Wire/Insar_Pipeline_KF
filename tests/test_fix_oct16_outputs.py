"""16 Oct 2026: the outputs batch of the technical readiness review
(Copy of Code_Check_List.xlsx, filled in insar_projects/).

SEC-05  spreadsheet formula injection, the rest of it: every CSV and workbook
        writer that can carry text from a client's file (a mine or panel name
        from a shapefile attribute, a site, polygon, line, window or point
        name, a station id) neutralises a cell that starts like a formula
        (areas.formula_safe / formula_safe_frame / formula_safe_text);
        plain numbers stay numbers.
OUT-01  the build label ('insar_pipeline <version> (code <hash>)') in the
BLD-01  outputs that had none: the client KMZ, the results viewer's CSV and
        KMZ, points_timeseries.parquet, summary.json, gnss_tie.json,
        quality_assessment.json, events.json, a Results export's
        summary.json and the CSVs that already had a '#' header; a code
        change alone still moves no numbers_hash.
QUA-05  deliverables._empirical_context is cached per project, products
        and settings, not under one constant key.
DAT-05  the KMZ icons are fetched over https.

Each test failed on the code at f3a0a0a. Synthetic projects in tmp_path
only; no real project and no network."""
import csv
import dataclasses
import io
import json
import os
import re
import zipfile
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest

import test_fix_L7a as L7a
import test_fix_L9b as L9b
import test_p23_events as EV
from insar_pipeline import areas, deliverables as dl, release, version

PKG = Path(__file__).resolve().parents[1]
LABEL = version.label()
# what a client's file can carry; each must never start a spreadsheet cell
HOSTILE = ("=cmd|' /C calc'!A0", "+cmd|' /C calc'!A0", "-cmd|' /C calc'!A0", "@SUM(1+1)*cmd|' /C calc'!A0")

built = L9b.built                      # the client report, KMZ and Results of L9b's synthetic project


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    from insar_pipeline import settings
    monkeypatch.setattr(settings, "identity", lambda: {"firm": "Test Firm", "preparer": "A. Engineer", "contact": ""})
    dl.clear_cache()
    yield
    dl.clear_cache()


def _cells(text: str):
    """Every cell a spreadsheet makes of a CSV's text, '#' lines included
    (Excel splits those at their commas too), leading spaces and quotes off."""
    return [c.lstrip(' "') for row in csv.reader(io.StringIO(text)) for c in row]


def _no_formula(text: str):
    bad = [c for c in _cells(text) if c.startswith(("=cmd", "+cmd", "-cmd", "@SUM", "\t", "\r"))]
    assert not bad, bad


# ------------------------------------------------------------------ SEC-05: the helpers
def test_formula_safe_neutralises_formulas_and_keeps_numbers():
    for h in HOSTILE + ("=1+1", "\t=1+1", "\r=1+1", "-2+3+cmd|' /C calc'!A0"):
        assert areas.formula_safe(h) == "'" + h
    for keep in ("-1.5", "+2", "-1e-3", "-.5", "Area A1", "North field 2", "", None, -1.5, 3, np.float32(-2.0)):
        assert areas.formula_safe(keep) is keep or areas.formula_safe(keep) == keep
    # a workbook cell: only what openpyxl would store as a formula
    assert areas.formula_safe("=cmd", workbook=True) == "'=cmd"
    assert areas.formula_safe("-0.50 to 0.30", workbook=True) == "-0.50 to 0.30"
    df = pd.DataFrame({"name": ["=cmd|' /C calc'!A0", "North", "-1.5", None], "v": [-1.0, 2.0, 3.0, 4.0],
                       "=col": ["a", "b", "c", "d"]})
    out = areas.formula_safe_frame(df)
    assert list(out["name"][:3]) == ["'=cmd|' /C calc'!A0", "North", "-1.5"] and pd.isna(out["name"][3])
    assert list(out.columns) == ["name", "v", "'=col"] and out["v"].dtype == float
    assert df.loc[0, "name"] == "=cmd|' /C calc'!A0" and "=col" in df        # the caller's frame is untouched
    plain = pd.DataFrame({"a": ["x", "y"], "b": [1, 2]})
    assert areas.formula_safe_frame(plain) is plain
    # a '#' line or a hand-joined row: one line, and no piece a spreadsheet splits off is a formula
    t = areas.formula_safe_text("North,=cmd|x\n@SUM(1);@SUM(2); -87.6, \"=HYPERLINK(1)\"")
    assert t == "North,'=cmd|x @SUM(1);'@SUM(2); -87.6, \"'=HYPERLINK(1)\""
    _no_formula("# " + t)


# ------------------------------------------------------------------ SEC-05: each writer
def test_panel_history_csv_neutralises_a_mine_name_from_the_shapefile(tmp_path):
    from types import SimpleNamespace
    from insar_pipeline import panel_history as ph
    (tmp_path / "deliverables").mkdir()
    panels = {}
    for i, mine in enumerate(HOSTILE + ("North\n=cmd|' /C calc'!A0", "Mine, =cmd|' /C calc'!A0")):
        panels[f"p{i}"] = {"mine": mine, "year": 2019, "n_pixels": 10, "rate_mm_yr": -1.5, "rate_in_yr": -0.06,
                           "total_method": "fitted", "significance": {"z": -2.5}}
    ph.write(SimpleNamespace(project_dir=tmp_path), {"panels": panels, "farmland": "the FIRST farmland pass"})
    text = (tmp_path / "deliverables" / "panel_history.csv").read_text()
    _no_formula(text)
    rows = [r for r in csv.reader(io.StringIO(text)) if r and not r[0].startswith("#")]
    assert len(rows) == 1 + len(panels) and all(len(r) == len(rows[0]) for r in rows)   # one row per panel
    assert {r[rows[0].index("rate_mm_yr")] for r in rows[1:]} == {"-1.5"}                # numbers stay numbers
    assert "# software: " + LABEL in text.splitlines()


def _site_project(tmp_path, name):
    lat0, lon0 = L7a.Y0 + 24 * L7a.DY, L7a.X0 + 30 * L7a.DX
    spec = f"{name}:{lat0 - 0.004},{lon0 - 0.004},{lat0 + 0.004},{lon0 + 0.004}"
    cfg, truth = L7a.make_project(tmp_path, focus_sites_lalo=(spec,))
    L7a._w(cfg.mintpy_dir / "geo" / "geo_maskTempCoh.h5", {"mask": truth["tc"] >= 0.4}, truth["attrs"])
    return cfg


def test_site_stats_and_selected_points_csvs_neutralise_the_site_name(tmp_path):
    from insar_pipeline import panel_history as ph
    cfg = _site_project(tmp_path, "North,=cmd|' /C calc'!A0")
    D = cfg.project_dir / "deliverables"
    assert ph.site_history(cfg)["sites"]
    dl.plot_selected_points(cfg, D)
    for f in ("site_stats.csv", "los_cumulative_displacement_selected_points.csv"):
        text = (D / f).read_text()
        _no_formula(text)
        assert "# software: " + LABEL in text.splitlines(), f
    rows = [r for r in csv.reader(io.StringIO((D / "site_stats.csv").read_text())) if not r[0].startswith("#")]
    assert all(len(r) == 9 for r in rows) and rows[1][0] == "North;'=cmd|' /C calc'!A0"


def test_event_csvs_neutralise_polygon_line_window_and_point_names(tmp_path):
    from insar_pipeline import events as E
    cfg, T = EV.make_project(tmp_path)
    panels = EV._write(tmp_path / "panels.geojson", EV._fc(
        EV._square(T, EV.CR - 18, EV.CR + 18, EV.CC - 18, EV.CC + 18, {"name": HOSTILE[0]})))
    lines = EV._write(tmp_path / "lines.geojson", EV._fc(
        {"type": "Feature", "properties": {"name": HOSTILE[1]},
         "geometry": {"type": "LineString", "coordinates": [list(EV._lonlat(T, EV.CR, 20)),
                                                            list(EV._lonlat(T, EV.CR, 100))]}}))
    lon_c, lat_c = EV._lonlat(T, EV.CR, EV.CC)
    out = tmp_path / "events"
    rec = E.write_event_products(cfg, out, windows=[{"name": HOSTILE[2], "start": EV.T0, "end": EV.T1}],
                                 polygons=str(panels), lines=str(lines),
                                 points={HOSTILE[3]: {"lat": lat_c, "lon": lon_c}})
    assert not rec["errors"]
    for f in ("panel_max.csv", "profiles.csv", "values_at.csv"):
        _no_formula((out / f).read_text())
    pm, prof, va = (pd.read_csv(out / f) for f in ("panel_max.csv", "profiles.csv", "values_at.csv"))
    assert list(pm["panel"]) == ["'" + HOSTILE[0]] and list(pm["window"]) == ["'" + HOSTILE[2]]
    assert set(prof["line"]) == {"'" + HOSTILE[1]} and set(va["id"]) == {"'" + HOSTILE[3]}
    assert pm["max_mm"].dtype == float and pm.loc[0, "max_mm"] < -100                # numbers stay numbers
    assert rec["software"] == LABEL                   # OUT-01: events.json says which code (its CSVs cannot)


def test_results_export_neutralises_the_area_name_in_csvs_and_workbook(built, tmp_path):
    import openpyxl
    from insar_pipeline import results as rs
    res = next(iter(built.res.values()))
    res = dataclasses.replace(res, area=HOSTILE[0],
                              points={**res.points, "label": [HOSTILE[3]] * len(next(iter(res.points.values())))},
                              area_summary={**res.area_summary, "area": HOSTILE[0]})
    out = rs.export(built.cfg, res, out_dir=tmp_path / "res")
    for f in ("points.csv", "statistics.csv", "annual.csv", "series.csv"):
        if (out / f).exists():
            _no_formula((out / f).read_text())
    with open(next(out.glob("*.xlsx")), "rb") as fh:
        wb = openpyxl.load_workbook(fh)
    formulas = [(ws.title, c.coordinate, c.value) for ws in wb for row in ws.iter_rows() for c in row
                if c.data_type == "f" or (isinstance(c.value, str) and c.value.startswith("="))]
    assert not formulas, formulas[:5]
    assert wb["Charts"]["A1"].value.startswith("'" + HOSTILE[0])
    summ = json.loads((out / "summary.json").read_text())
    assert summ["software"] == LABEL                                # OUT-01: the record says which code


def test_validation_epochs_csv_neutralises_a_station_id(tmp_path):
    from insar_pipeline import validation as V
    p = tmp_path / "validation_epochs.csv"
    V.write_epochs_csv(p, [{"station": HOSTILE[0], "date": "2019-01-15", "insar_mm": -1.5, "truth_mm": -2.0},
                           {"station": "TW01", "date": "2019-02-15", "insar_mm": -3.0, "truth_mm": -2.5}])
    text = p.read_text()
    _no_formula(text)
    rows = list(csv.DictReader(io.StringIO(text)))
    assert rows[0]["station"] == "'" + HOSTILE[0] and rows[1]["station"] == "TW01" and rows[0]["insar_mm"] == "-1.5"


# ------------------------------------------------------------------ the results viewer's exports
def _viewer_table(monkeypatch, title="Test site"):
    from insar_pipeline import field_points as fp
    from insar_pipeline import results_viewer as rv
    one = lambda v: np.array([v])                               # noqa: E731
    fd = {"meta": {"title": title, "dates": ["20170101", "20200101"], "t": [0.0, 3.0], "numbers_hash": "abc123",
                   "spacing_m": 100},
          "lat": one(38.3), "lon": one(-87.6), "group": np.zeros(1, "u1"), "fast": np.zeros(1, bool),
          "rate": one(-2.0), "los_rel": one(-1.0), "change_los": one(-3.0), "snr": one(1.2), "tc": one(0.8),
          "unwrap": one(0.9)}
    monkeypatch.setattr(rv, "_table", lambda config: (None, fd, None))
    monkeypatch.setattr(rv, "_inside", lambda fd, rings: np.array([0]))
    monkeypatch.setattr(rv, "_poly_stats", lambda fd, idx: {"series_vertical_in": [0.0, -0.1], "n": 1, "n_too_fast": 0})
    monkeypatch.setattr(fp, "series_vertical_in_rows", lambda fd, idx: np.full((1, 2), 0.1))
    return rv


def test_viewer_csv_keeps_every_piece_of_a_name_out_of_formula_cells_and_carries_the_label(monkeypatch):
    rv = _viewer_table(monkeypatch, title="Site, =cmd|' /C calc'!A0")
    out = rv.api_export_csv({"config": "x", "rings": [[]], "name": "North, =cmd|' /C calc'!A0;@SUM(1+1)"})
    _no_formula(out["text"])
    lines = out["text"].splitlines()
    assert lines[2] == "# software: " + LABEL
    # nine '#' lines since 19 Oct 2026 (+ the signal-to-threshold column's)
    assert all(ln.startswith("#") for ln in lines[:9]) and lines[9].startswith("latitude,")


def test_viewer_kmz_carries_the_numbers_and_the_label(monkeypatch):
    import base64
    from insar_pipeline import client_kmz as ck
    rv = _viewer_table(monkeypatch)
    monkeypatch.setattr(ck, "point_parts", lambda *a, **k: {})
    monkeypatch.setattr(ck, "points_kml", lambda parts, bins: ("", []))
    monkeypatch.setattr(ck, "svg_chart", lambda *a, **k: "")
    out = rv.api_export_kmz({"config": "x", "rings": [[[-87.6, 38.3], [-87.5, 38.3], [-87.5, 38.4]]], "name": "North"})
    kml = zipfile.ZipFile(io.BytesIO(base64.b64decode(out["b64"]))).read("doc.kml").decode()
    assert f"<Data name='software'><value>{LABEL}</value></Data>" in kml
    assert "<Data name='numbers_hash'><value>abc123</value></Data>" in kml


# ------------------------------------------------------------------ OUT-01 / DAT-05: the client KMZ
def test_client_kmz_carries_the_label_and_fetches_its_icons_over_https(built):
    doc = re.search(r"<Document>.*?<ExtendedData>(.*?)</ExtendedData>", built.kml, re.S).group(1)
    assert f"<Data name='software'><value>{LABEL}</value></Data>" in doc
    assert "http://maps.google.com" not in built.kml and "https://maps.google.com/mapfiles/kml/shapes/" in built.kml
    # and nowhere in the package does a KMZ fetch an icon over plain http
    assert not [p.name for p in PKG.glob("*.py") if "http://maps.google.com" in p.read_text()]


# ------------------------------------------------------------------ OUT-01: the records
def test_tie_qa_and_summary_records_carry_the_label(tmp_path, monkeypatch):
    from insar_pipeline import gnss_tie, quality_assessment as qa
    cfg, _ = L7a.make_project(tmp_path)
    D = cfg.project_dir / "deliverables"
    tie = gnss_tie._untied(cfg, "no station in range", dates=list(L7a.DATES))
    assert tie["software"] == LABEL and json.loads((D / "gnss_tie.json").read_text())["software"] == LABEL
    r = {"verdict": "MEETS PRACTICE", "checks": {}}
    qa.write(cfg, r)
    assert json.loads((cfg.project_dir / "quality_assessment.json").read_text())["software"] == LABEL
    assert "software" not in r                                     # the caller's dict is left as it was
    # summary.json from the products stage (every exporter stubbed: only the record is under test)
    for name in ("supersede_results", "export_velocity", "export_velocity_uncertainty", "export_empirical_uncertainty",
                 "export_cell_noise", "export_signal_to_noise", "export_vertical_velocity", "export_quality",
                 "export_extra_layers", "export_aliasing_products", "export_cumulative_displacement",
                 "plot_displacement_series", "plot_selected_points", "export_measurement_points",
                 "export_coverage_curve", "export_event_products", "_apply_qa_and_gnss"):
        monkeypatch.setattr(dl, name, lambda *a, **k: None)
    monkeypatch.setattr(dl, "summary_stats", lambda cfg: {"total_measured_points": 7})
    monkeypatch.setattr(dl.provenance, "write", lambda *a, **k: None)
    cfg.gnss_validation = False
    dl.export_all(cfg)
    assert json.loads((D / "summary.json").read_text())["software"] == LABEL
    # the QA's refresh keeps the label of the code that computed the numbers
    S = json.loads((D / "summary.json").read_text())
    (D / "summary.json").write_text(json.dumps({**S, "software": "insar_pipeline 2026.1.1 (code 000000000000)"}))
    dl.refresh_summary(cfg)
    assert json.loads((D / "summary.json").read_text())["software"] == "insar_pipeline 2026.1.1 (code 000000000000)"


def test_a_code_change_alone_moves_no_numbers_hash():
    """The tie's label reaches client_report.json (rep['gnss'] = {**tie, ...}):
    another build with the same numbers must keep the sign-off binding."""
    rep = {"areas": [{"name": "Area A1", "vertical_mm_yr": -1.2}],
           "gnss": {"background": {"vertical_mm_yr": -1.5}, "software": "insar_pipeline 2026.10.16 (code aaaaaaaaaaaa)"}}
    other = json.loads(json.dumps(rep))
    other["gnss"]["software"] = "insar_pipeline 2026.10.17 (code bbbbbbbbbbbb)"
    other["software"] = "insar_pipeline 2026.10.17 (code bbbbbbbbbbbb)"
    assert release.numbers_hash(rep) == release.numbers_hash(other)
    other["gnss"]["background"]["vertical_mm_yr"] = -1.6
    assert release.numbers_hash(rep) != release.numbers_hash(other)            # a number still does


def test_points_timeseries_parquet_carries_the_label(tmp_path):
    import pyarrow.parquet as pq
    cfg, _ = L7a.make_project(tmp_path)
    D = cfg.project_dir / "deliverables"
    dl.export_measurement_points(cfg, D)
    md = pq.read_metadata(D / "points_timeseries.parquet").metadata
    assert md[b"software"].decode() == LABEL and b"geo" in md


# ------------------------------------------------------------------ QUA-05: one context per project
def test_empirical_context_is_per_project_and_still_cached(tmp_path):
    cfg_a, _ = L7a.make_project(tmp_path / "a")
    cfg_b, truth_b = L7a.make_project(tmp_path / "b")
    # project B sinks 30 mm/yr more in a block of farmland the far field reads
    vel_b = cfg_b.mintpy_dir / "geo" / "geo_velocity.h5"
    with h5py.File(vel_b, "r+") as f:
        v = f["velocity"][:]
        v[30:45, 3:15] -= 0.030
        f["velocity"][...] = v
    ref_b = [np.array(x) for x in (dl._empirical_context(cfg_b)[i] for i in (0, 1, 4))]
    dl.clear_cache()
    a = dl._empirical_context(cfg_a)
    assert dl._empirical_context(cfg_a) is a                      # the same project: still computed once
    b = dl._empirical_context(cfg_b)
    assert b is not a
    for got, want in zip((b[0], b[1], b[4]), ref_b):
        np.testing.assert_array_equal(got, want)                   # B's own far field, rates and noise
    assert not np.array_equal(a[1], b[1], equal_nan=True)
    # B's velocity rewritten: its context is computed again
    st = vel_b.stat()
    os.utime(vel_b, ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))
    assert dl._empirical_context(cfg_b) is not b
