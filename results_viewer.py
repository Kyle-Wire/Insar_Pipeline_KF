"""The results viewer: a map in the browser of a project's measurement points
over the whole study area - colour by vertical movement, line-of-sight
velocity, line-of-sight cumulative displacement or signal-to-noise; click a
point for its figures and its movement over the record; draw a polygon for
the average movement of the points inside it, and export it (CSV, KMZ) or
save it as a report area for the next report. Zoomed in on the whole record
(street level), every measured 15 m cell takes the place of the ~100 m points
(api_cells: each cell's own figures, the client rasters'), and a click gives
that cell's balloon and its own chart (api_cell: field_points.
cells_from_frames, gather's numbers); a link can name a cell (#cell=lat,lon).
"KMZ, every 15 m cell" exports
every measured cell inside the polygon over the chosen dates, computed from
the project's products (_full_table; up to FULL_RES_MAX_CELLS) - the client
KMZ itself keeps the ~100 m points.

    insar viewer           (or m in the dashboard, or python -m insar_pipeline.results_viewer)

It reads <project>/deliverables/viewer/field_points.npz, the table the
client KMZ is built from (field_points; written by every client KMZ build,
or alone by `python -m insar_pipeline.field_points <config>`), so a point
reads the same here as in the KMZ, the rasters and the point table. A table
built from other numbers than the current client_report.json is shown with
a warning (its numbers_hash differs).

A polygon's figures are the plain average of the points inside it (about
100 m apart; neighbouring values share ground, so they are not independent).
They are a look, not a test: the report's area figures - the 95% range
against ordinary ground of the same size, the detection limit - come from
saving the polygon as a report area and rebuilding the client report.

Date windows (19 Oct 2026): a bar at the top picks any window of the record
(6+ images over 3+ months, results' rule) and every map layer, figure, chart
and download follows it. The page computes a window itself
(viewer_assets/viewer_core.js, held to viewer_bundle.py and so to
results.analyse by the tests) from the table's own-cell series; the whole
record shows the stored figures. An untested window's figures are labelled
a look: its single-point noise comes from a sample of the report's farmland.
Every chart downloads as PNG, SVG or CSV with the report's stamp lines.

Like the builder, the server binds to 127.0.0.1 only, refuses requests from
other web pages (Origin/Host checks), and takes an API call only with the
secret it wrote into the page it served this session
(map_builder.page_with_token). It serves the page and an exact list of its
script, style and image files (ASSETS, from viewer_assets/ - nothing else on
disk), under a Content-Security-Policy that allows scripts only from itself.
Map tiles are fetched by the browser from the chosen basemap.
"""
from __future__ import annotations

import base64
import csv
import html
import http.server
import io
import json
import math
import re
import shutil
import socket
import threading
import zipfile
from datetime import datetime
from functools import partial
from pathlib import Path

import numpy as np

PKG = Path(__file__).resolve().parent
PAGE = "results_viewer.html"
PORT = 8766
SERVE_DIR = Path.home() / ".cache" / "insar_viewer"
MM_IN = 25.4
Z95 = 1.96
_SERVER: dict = {}
_CACHE: dict = {}          # npz path -> (mtime, field table)
_BUNDLES: dict = {}        # npz path -> ((npz mtime, report mtime), encoded bundle)
ASSET_DIR = PKG / "viewer_assets"
# every file the page loads besides itself, by the exact path it asks for
# (viewer_assets/ only; make_release.sh ships each one, tests/test_fix_oct7_
# dec_D14D18_cfg.py checks that)
ASSETS = {f"assets/{n}": ASSET_DIR / n for n in (
    "viewer.css", "viewer_core.js", "viewer_charts.js", "viewer_app.js",
    "vendor/leaflet.min.js", "vendor/leaflet.min.css", "vendor/leaflet.draw.js", "vendor/leaflet.draw.css",
    "vendor/LICENSES.txt", "vendor/images/layers.png", "vendor/images/layers-2x.png",
    "vendor/images/marker-icon.png", "vendor/images/marker-icon-2x.png", "vendor/images/marker-shadow.png",
    "vendor/images/spritesheet.png", "vendor/images/spritesheet-2x.png", "vendor/images/spritesheet.svg")}
CONTENT_TYPES = {".js": "text/javascript; charset=utf-8", ".css": "text/css; charset=utf-8", ".png": "image/png",
                 ".svg": "image/svg+xml", ".txt": "text/plain; charset=utf-8"}
# the basemaps the page offers (EXTERNAL_SERVICES.md): the only places it fetches from besides this server
TILE_HOSTS = ("https://basemap.nationalmap.gov", "https://server.arcgisonline.com", "https://tile.openstreetmap.org",
              "https://*.tile.opentopomap.org")
# the one page that may show this one inside it: the app window's results section
# (app_server.PORT; 11 Nov 2026). Any other page still cannot frame it (NET-06).
APP_PORT = 8767
APP_ORIGINS = (f"http://localhost:{APP_PORT}", f"http://127.0.0.1:{APP_PORT}")
CSP = ("default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
       f"img-src 'self' data: blob: {' '.join(TILE_HOSTS)}; connect-src 'self'; font-src 'self'; "
       f"object-src 'none'; base-uri 'none'; form-action 'none'; frame-ancestors {' '.join(APP_ORIGINS)}")


# ---------------------------------------------------------------- data
def _is_combined(config: str) -> bool:
    from . import combine
    return combine.is_combined_config(config)


def _config_paths(config: str):
    from .config import load_config
    from . import field_points as fp
    if _is_combined(config):                     # a combined project (combine_viewer): its own table
        from . import combine
        from . import combine_viewer as cv
        from .app_server import _CombinedCfg
        cfg = _CombinedCfg(str(config))
        return cfg, cv.viewer_path(cfg.cc), cfg.cc.deliverables / combine.REPORT
    cfg = load_config(config)
    return cfg, fp.viewer_path(cfg), Path(cfg.project_dir) / "deliverables" / "client_report.json"


_SOURCES: dict = {}             # combined config -> (combined report mtime, combine_viewer.Sources)


def _combined_sources(config: str):
    """Both projects of a combined config, loaded once per combined report
    (combine_viewer.Sources): its 15 m cells, polygons and clicked cells."""
    from . import combine
    from . import combine_viewer as cv
    rp = combine.load(config).deliverables / combine.REPORT
    if not rp.exists():
        raise ValueError("This combined project has not been combined yet: press Start, then Combine, in the app")
    m = rp.stat().st_mtime
    hit = _SOURCES.get(str(config))
    if hit is None or hit[0] != m:
        _SOURCES.clear()
        _SOURCES[str(config)] = hit = (m, cv.Sources(config))
    return hit[1]


def _not_for_combined(config: str, what: str) -> None:
    if _is_combined(config):
        raise ValueError(f"{what} is not available for the combined view yet.")


def _table(config: str) -> tuple:
    """(cfg, field table, stale reason or None) for a project; the table is
    cached by file modification time."""
    from . import field_points as fp
    cfg, p, rj = _config_paths(config)
    if not p.exists():
        raise ValueError(f"No viewer points for this project yet ({p}). Build the client KMZ (Products in the "
                         f"dashboard), or run: python -m insar_pipeline.field_points {config}")
    m = p.stat().st_mtime
    hit = _CACHE.get(str(p))
    if hit is None or hit[0] != m:
        # a table from before 26 Oct 2026 gets the delivered rate's fit axis here
        fd_ = fp.load(p)
        _CACHE[str(p)] = hit = (m, fd_ if fd_["meta"].get("combined") else fp.add_fit_axis(cfg, fd_))
    fd = hit[1]
    stale = None
    if getattr(cfg, "_combined", False):
        from . import combine
        why = combine.stale_reason(config)
        if why:
            stale = f"The combined view is out of date: {why}."
        elif rj.exists() and rj.stat().st_mtime > m + 1:
            stale = "These points are older than the combined result: press Start, then Combine, in the app."
        return cfg, fd, stale
    try:
        from . import release
        cur = release.numbers_hash(json.loads(rj.read_text()))
        if fd["meta"].get("numbers_hash") and cur != fd["meta"]["numbers_hash"]:
            stale = ("These points were built from an earlier version of the client report's numbers; rebuild the "
                     "client KMZ so they match the report.")
    except (OSError, ValueError, ImportError):
        stale = "No client_report.json beside these points: they cannot be checked against the report."
    return cfg, fd, stale


def _b64(a, dtype="<f4") -> str:
    return base64.b64encode(np.ascontiguousarray(np.asarray(a, dtype=dtype)).tobytes()).decode()


def _nice_limits(fd: dict) -> dict:
    from .client_kmz import nice_limit
    f = fd["group"] == 0
    return {"los": nice_limit(fd["los_rel"][f] / MM_IN), "change": nice_limit(fd["change_los"][f] / MM_IN)}


def _inside(fd: dict, rings: list) -> np.ndarray:
    """Field points (not the features' extra sample: they would count twice)
    inside the polygon (rings: [[lon, lat], ...] - the first the outline,
    the rest holes, as GeoJSON)."""
    from matplotlib.path import Path as MPath
    if not rings or len(rings[0]) < 3:
        raise ValueError("draw a polygon with at least three corners")
    xy = np.column_stack([fd["lon"], fd["lat"]])
    sel = MPath(np.asarray(rings[0], float)[:, :2]).contains_points(xy)
    for hole in rings[1:]:
        sel &= ~MPath(np.asarray(hole, float)[:, :2]).contains_points(xy)
    return np.flatnonzero(sel & (fd["group"] == 0))


def _series(fd: dict, j: int) -> tuple[list, list]:
    from . import field_points as fp
    r = lambda a: [None if not np.isfinite(v) else round(float(v), 4) for v in a]          # noqa: E731
    return r(fp.series_vertical_in(fd, j)), r(fp.series_los_in(fd, j))


def _poly_stats(fd: dict, idx: np.ndarray) -> dict:
    from . import field_points as fp
    m = fd["meta"]
    ok = idx[~fd["fast"][idx]]
    out = {"n": int(idx.size), "n_too_fast": int(idx.size - ok.size)}
    if not ok.size:
        return out
    q = lambda a: None if not np.isfinite(a).any() else float(np.nanmean(a))                # noqa: E731
    pct = lambda a, p: None if not np.isfinite(a).any() else float(np.nanpercentile(a, p))  # noqa: E731
    v_in = fd["rate"][ok] / MM_IN
    snr = fd["snr"][ok]
    with np.errstate(invalid="ignore"):
        stand = float((snr[np.isfinite(snr)] >= Z95).mean() * 100) if np.isfinite(snr).any() else None
    with __import__("warnings").catch_warnings():
        __import__("warnings").simplefilter("ignore", RuntimeWarning)
        if "cell_los" in fd:                             # each point's own cell (the series its figures come from)
            from . import viewer_bundle as vb
            T = len(m["dates"])
            vert = vb.mean_series(fd, ok, 0, T - 1, True) / MM_IN
            los = vb.mean_series(fd, ok, 0, T - 1, False) / MM_IN
        else:
            vert = np.nanmean(fp.series_vertical_in_rows(fd, ok), axis=0)
            los = np.nanmean(fp.series_los_in_rows(fd, ok), axis=0)
    r = lambda a: [None if not np.isfinite(v) else round(float(v), 4) for v in a]          # noqa: E731
    out.update({
        "vertical_in_yr": q(v_in), "vertical_in_yr_p10": pct(v_in, 10), "vertical_in_yr_p90": pct(v_in, 90),
        "relative_in_yr": q(fd["rel_v"][ok] / MM_IN),
        "los_in_yr": q(fd["los_rel"][ok] / MM_IN), "los_change_in": q(fd["change_los"][ok] / MM_IN),
        "median_snr": None if not np.isfinite(snr).any() else float(np.nanmedian(snr)),
        "percent_standing_out": stand, "percent_by_chance": m.get("snr_chance_pct"),
        "mean_coherence": q(fd["tc"][ok]), "series_vertical_in": r(vert), "series_los_in": r(los),
        "anchored": bool(m.get("anchored")), "span_yr": float(m["span"])})
    return out


# ---------------------------------------------------------------- API
def api_projects(_: dict) -> dict:
    from . import runs
    from .config import load_config
    from . import field_points as fp
    out = []
    from . import combine
    from . import combine_viewer as cv
    for c in runs.known_configs():
        try:
            p = cv.viewer_path(combine.load(c)) if combine.is_combined_config(c) else fp.viewer_path(load_config(str(c)))
        except Exception:                                    # noqa: BLE001
            continue
        out.append({"config": str(c), "name": Path(c).stem.replace("config_", ""), "ready": p.exists(),
                    "built": datetime.fromtimestamp(p.stat().st_mtime).strftime("%d %b %Y %H:%M") if p.exists() else None})
    return {"projects": out}


def api_load(body: dict) -> dict:
    cfg, fd, stale = _table(body["config"])
    m = fd["meta"]
    lim = _nice_limits(fd)
    rel_cols = bool(m.get("relative_colours"))
    return {
        "title": m.get("title"), "dates": m["dates"], "span": m["span"], "spacing_m": m.get("spacing_m"),
        "anchored": m.get("anchored"), "ground": m.get("ground"), "stale": stale, "n": int(fd["lat"].size),
        "bins": m.get("bins"), "relative_colours": rel_cols, "limits": lim, "names": m.get("names"),
        "snr_chance_pct": m.get("snr_chance_pct"), "areas": m.get("areas") or [], "features": m.get("features") or [],
        "bbox": list(cfg.bbox),
        "lat": _b64(fd["lat"], "<f8"), "lon": _b64(fd["lon"], "<f8"),
        # the colour value of the movement layer is the map's frame (client_kmz.point_parts)
        "vert": _b64((fd["rel_v"] if rel_cols else fd["rate"]) / MM_IN),
        "los": _b64(fd["los_rel"] / MM_IN), "change": _b64(fd["change_los"] / MM_IN), "snr": _b64(fd["snr"]),
        "fast": _b64(fd["fast"], "u1"), "group": _b64(fd["group"], "u1"),
    }


def bundle_bytes(config: str) -> bytes:
    """The page's data for a project (viewer_bundle.bundle, encoded), cached
    until the table or the report changes."""
    from . import viewer_bundle as vb
    cfg, fd, stale = _table(config)
    from . import field_points as fp
    p = fp.viewer_path(cfg)
    rj = Path(cfg.project_dir) / "deliverables" / "client_report.json"
    key = (p.stat().st_mtime, rj.stat().st_mtime if rj.exists() else None)
    hit = _BUNDLES.get(str(p))
    if hit is None or hit[0] != key:
        try:
            rep = json.loads(rj.read_text())
        except (OSError, ValueError):
            rep = {}
        header, arrays = vb.bundle(cfg, fd, rep, stale=stale)
        if fd["meta"].get("combined"):
            src = fd["meta"].get("sources") or []
            header["meta"].update({
                "route": "combined", "screening": False, "final": False, "windows": False,
                "window_note": fd["meta"].get("window_note"),
                "blockers": ["The combined view is for review: east-west is indicative (not a tested figure) and "
                             "the combined vertical is not yet signed off as a client figure."],
                "stamp": [f"{fd['meta'].get('title')}: vertical and east-west from "
                          + " and ".join(f"{s['project']} ({s['direction']})" for s in src)]})
        _BUNDLES[str(p)] = hit = (key, vb.encode(header, arrays))
    return hit[1]


def _window(fd: dict, body: dict):
    """(window figures, table to export from) for a request's `window`
    ({first, last} as YYYYMMDD; absent = the whole record): a look - a
    tested window's null table arrives with the tested results (phase 2)."""
    from . import viewer_bundle as vb
    w = body.get("window") or {}
    dates = fd["meta"]["dates"]
    i0, i1 = vb.window_index(dates, w.get("first"), w.get("last"))
    if i0 == 0 and i1 == len(dates) - 1:                  # the whole record: the table as stored
        return {"kind": "report", "i0": i0, "i1": i1, "first": dates[0], "last": dates[-1]}, fd
    W = vb.window_figures(fd, i0, i1)
    return W, vb.window_table(fd, W)


def _east_in(fd: dict, rows) -> np.ndarray:
    """East-west series (in) of table rows, each zeroed on its own trend."""
    t = np.asarray(fd["meta"]["t"])
    S = np.asarray(fd["series_east"], float)[np.atleast_1d(rows)]
    out = np.full(S.shape, np.nan)
    for i, s in enumerate(S):
        ok = np.isfinite(s)
        out[i] = s - np.polyfit(t[ok], s[ok], 1)[1] if ok.sum() > 2 else s
    return out / MM_IN


def _combined_html(fd: dict, j: int) -> str:
    """A combined point's figures, in the balloon's place."""
    m, b = fd["meta"], lambda v, d=2: "—" if not np.isfinite(v) else f"{v / MM_IN:+.{d}f} in/yr ({v:+.1f} mm/yr)"
    rows = [("Vertical, per year", b(fd["rate"][j]) + (" with the regional GNSS motion" if m.get("anchored") else "")),
            ("Vertical vs the surrounding ground", b(fd["rel_v"][j])),
            ("East-west, per year (indicative)", b(fd["east"][j]) + "; + = east"),
            ("Signal-to-noise (vertical)", "—" if not np.isfinite(fd["snr"][j]) else f"{fd['snr'][j]:.1f}"),
            ("Measurement quality", f"{float(fd['tc'][j]):.2f}")]
    return ("<table>" + "".join(f"<tr><th>{k}</th><td>{v}</td></tr>" for k, v in rows) + "</table>"
            f"<p>Both passes together, {m['dates'][0]} to {m['dates'][-1]}; relative to {m.get('ground')}.</p>")


def _combined_point(fd: dict, j: int) -> dict:
    from . import field_points as fp
    r = lambda a: [None if not np.isfinite(v) else round(float(v), 4) for v in a]          # noqa: E731
    m = fd["meta"]
    sd_v, bg_u = float(fd["sd_v"][j]), m.get("bg_u")
    u = (sd_v ** 2 + (bg_u or 0.0) ** 2) ** 0.5 if np.isfinite(sd_v) else None
    from . import client_kmz as ck
    return {"html": ck._no_neg_zero(_combined_html(fd, j)), "vert": r(fp.series_vertical_in(fd, j)), "los": r(_east_in(fd, j)[0]),
            "fast": False, "band_vert_in_per_yr": None if u is None else Z95 * u / MM_IN, "band_los_in_per_yr": None,
            "bg_v_in_per_yr": float(m["bg_v"]) / MM_IN if m.get("anchored") else 0.0,
            "lat": float(fd["lat"][j]), "lon": float(fd["lon"][j])}


def _combined_poly(fd: dict, idx: np.ndarray) -> dict:
    """_poly_stats with east-west where a single project has line of sight."""
    st = _poly_stats(fd, idx)
    if idx.size:
        with __import__("warnings").catch_warnings():
            __import__("warnings").simplefilter("ignore", RuntimeWarning)
            st.update({"los_in_yr": float(np.nanmean(fd["east"][idx])) / MM_IN,
                       "los_change_in": float(np.nanmean(fd["east_change"][idx])) / MM_IN,
                       "series_los_in": [None if not np.isfinite(v) else round(float(v), 4)
                                         for v in np.nanmean(_east_in(fd, idx), axis=0)]})
    return st


def api_point(body: dict) -> dict:
    from . import client_kmz as ck
    _, fd, _ = _table(body["config"])
    if fd["meta"].get("combined"):
        j = int(body["j"])
        if not 0 <= j < fd["lat"].size:
            raise ValueError("no such point")
        return _combined_point(fd, j)
    j = int(body["j"])
    if not 0 <= j < fd["lat"].size:
        raise ValueError("no such point")
    m = fd["meta"]
    # the KMZ's own balloon, without its chart (the page draws one you can read off)
    desc = ck.point_parts(fd, j, (m.get("names") or ["Study area"])[int(fd["group"][j])], [(-1e9, 1e9, (0, 0, 0), "")],
                          False, m["dates"][0], m["dates"][-1], ck.decimal_year(m["dates"][0]))["desc"]
    desc = re.sub(r"<svg.*?</svg>", "", desc, flags=re.S)
    vert, los = _series(fd, j)
    sd_v, bg_u = float(fd["sd_v"][j]), m.get("bg_u")
    u = (sd_v ** 2 + (bg_u or 0.0) ** 2) ** 0.5 if np.isfinite(sd_v) else None
    return {"html": ck._no_neg_zero(desc), "vert": vert, "los": los, "fast": bool(fd["fast"][j]),
            "band_vert_in_per_yr": None if u is None else Z95 * u / MM_IN,
            "band_los_in_per_yr": None if u is None else Z95 * sd_v * float(fd["up"][j]) / MM_IN,
            "bg_v_in_per_yr": float(m["bg_v"]) / MM_IN if m.get("anchored") else 0.0,
            "lat": float(fd["lat"][j]), "lon": float(fd["lon"][j])}


def api_polygon(body: dict) -> dict:
    _, fd, _ = _table(body["config"])
    if fd["meta"].get("combined"):
        return _combined_poly(fd, _inside(fd, body["rings"]))
    return _poly_stats(fd, _inside(fd, body["rings"]))


def _safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", name or "").strip("_") or "polygon"


def api_export_csv(body: dict) -> dict:
    """The points inside the polygon and their series: one row per point,
    then the polygon's average; inches, every figure in the report's frame."""
    if _is_combined(body["config"]):                     # both passes (combine_viewer, 28 Oct 2026)
        return _combined_export_csv(body)
    _, fd0, stale = _table(body["config"])
    if body.get("resolution") == "full":                 # every measured 15 m cell inside (_full_table)
        fd0 = _cell_table(body["config"], body["rings"], fd0)
        idx = np.arange(fd0["row"].size)
    else:
        idx = _inside(fd0, body["rings"])
    if not idx.size:
        raise ValueError("no measurement points inside this polygon")
    W, fd = _window(fd0, body)
    m = fd["meta"]
    kind = {"report": "the client report's figures (whole record)",
            "look": "a look, not a tested figure: this window's single-point noise is from a sample of the "
                    "report's farmland cells",
            "tested": "this window as results.analyse tested it"}[W["kind"]]
    # names and the title go into '#' header lines on one line (SEC-05,
    # 16 Oct 2026): a newline in a typed name started a line a spreadsheet
    # read as a cell, so '=cmd|...' ran as a formula; and a spreadsheet
    # splits a '#' line at its commas, so no piece of a name may start one
    # either (formula_safe_text)
    from .areas import formula_safe_text, one_line
    from .deliverables import _software
    name = one_line(body.get("name")) or "polygon"
    st = _poly_stats(fd, idx)
    buf = io.StringIO()
    g = m.get("ground") or "farmland"
    for line in (
            f"# {formula_safe_text(m.get('title'))} - {formula_safe_text(name)}: {idx.size} measurement points "
            f"(about {m.get('spacing_m') or 100:.0f} m apart)",
            f"# dates {m['dates'][0]} to {m['dates'][-1]} ({len(m['dates'])} images); {kind}; numbers_hash "
            f"{m.get('numbers_hash')}" + (f"; {one_line(stale)}" if stale else ""),
            f"# software: {_software()}",           # which code wrote it (16 Oct 2026, readiness review OUT-01)
            "# vertical_in_per_yr: vertical movement per year (line of sight / its up component, ASSUMES all movement "
            f"is vertical), relative to ordinary {g} of the same kind" + (" + the regional GNSS motion" if m.get("anchored") else ""),
            f"# line_of_sight_in_per_yr, line_of_sight_change_in: relative to ordinary {g} of the same kind; + = towards "
            "the satellite; the change is the first image of these dates to the last",
            "# signal_to_noise: |movement relative to ordinary ground| / its noise; 1.96 or more stands out"
            + (f" ({m.get('snr_chance_pct')}% of ordinary {g} does by chance over the whole record)"
               if m.get("snr_chance_pct") is not None else ""),
            "# signal_to_threshold: the same divided by 1.96 - 1 or more stands out",
            "# vertical_in_<date>: the point's own cell, vertical, inches, zeroed on its straight-line fit over these "
            "dates at the first image (the charts' line); the last row is the polygon's average of them",
            "# negative = sinking / away from the satellite"):
        buf.write(line + "\n")
    w = csv.writer(buf)
    w.writerow(["latitude", "longitude", "vertical_in_per_yr", "line_of_sight_in_per_yr", "line_of_sight_change_in",
                "signal_to_noise", "signal_to_threshold", "temporal_coherence", "unwrapping_consistency",
                "too_fast_to_measure"] + [f"vertical_in_{d}" for d in m["dates"]])
    f4 = lambda v: "" if v is None or not np.isfinite(v) else f"{v:.4f}"                    # noqa: E731
    from . import viewer_bundle as vb
    T = len(m["dates"])
    V = np.vstack([vb.point_series(fd, int(j), 0, T - 1) for j in idx]) / MM_IN if "cell_los" in fd else None
    if V is None:                                        # a version-1 table: the 3 x 3 series it has
        from . import field_points as fp
        V = fp.series_vertical_in_rows(fd, idx)
    for row, j in enumerate(idx):
        vert = [None if not np.isfinite(x) else float(x) for x in V[row]]
        fast = bool(fd["fast"][j])
        w.writerow([f"{fd['lat'][j]:.6f}", f"{fd['lon'][j]:.6f}"]
                   + (["", "", "", "", ""] if fast else [
                       f4(fd["rate"][j] / MM_IN), f4(fd["los_rel"][j] / MM_IN), f4(fd["change_los"][j] / MM_IN),
                       f"{fd['snr'][j]:.2f}", f"{fd['snr'][j] / 1.96:.2f}"])
                   + [f"{fd['tc'][j]:.3f}", f4(fd["unwrap"][j]), int(fast)]
                   + ([""] * len(vert) if fast else ["" if v is None else f"{v:.4f}" for v in vert]))
    ok = [r for r, j in enumerate(idx) if not fd["fast"][j]]
    with __import__("warnings").catch_warnings():
        __import__("warnings").simplefilter("ignore", RuntimeWarning)
        avg = np.nanmean(V[ok], axis=0) if ok else np.full(T, np.nan)
    w.writerow(["polygon average", "", f4(st.get("vertical_in_yr")), f4(st.get("los_in_yr")), f4(st.get("los_change_in")),
                "", "", f4(st.get("mean_coherence")), "", ""]
               + ["" if not np.isfinite(v) else f"{v:.4f}" for v in avg])
    tag = "" if W["kind"] == "report" else f"_{W['first']}_{W['last']}"
    what = "every_cell" if body.get("resolution") == "full" else "points"
    return {"filename": f"{_safe_name(name)}{tag}_{what}.csv", "text": buf.getvalue()}


# Full resolution (26 Oct 2026, Kyle: the client KMZ's ~100 m points are
# sparse beside the 15 m cells; export every cell, but only at the sites and
# dates chosen here). ~210 bytes a point in the KMZ (~1.9 kB as KML): 40,000
# cells, about 9 km2 of fully measured ground, is ~8 MB that Google Earth
# Pro opens without a long wait.
FULL_RES_MAX_CELLS = 40_000
_CTX: dict = {}                 # client_report.json -> (mtime, cfg, rep, ctx), as _CACHE


def _too_many_cells(n: int) -> str:
    return (f"{n:,} measured cells inside this polygon; its full-resolution figures and exports take at most "
            f"{FULL_RES_MAX_CELLS:,} (about 9 km2 of fully measured ground, a KMZ Google Earth opens quickly). "
            "Draw a smaller polygon; the KMZ button gives the ~100 m points.")


def _project_context(config: str) -> tuple:
    """(cfg, rep, ctx, cell maps) of a project - the context the client KMZ
    builds its points from (panel_history._Context on the report's farmland)
    and field_points.cell_maps - loaded once per version of
    client_report.json, one project at a time."""
    from . import field_points as fp
    from . import results as rs
    from .panel_history import _Context
    cfg, _, rj = _config_paths(config)
    if not rj.exists():
        raise ValueError("No client_report.json for this project: build the client report first")
    m = rj.stat().st_mtime
    hit = _CTX.get(str(rj))
    if hit is None or hit[0] != m:
        rep = rs.report_record(cfg)
        ctx = _Context(cfg)
        rs.report_farmland(ctx, rep)
        _CTX.clear()                                          # one project's cube frames at a time
        _CTX[str(rj)] = hit = (m, cfg, rep, ctx, fp.cell_maps(cfg, rep, ctx))
    return hit[1:]


# The 15 m cells on the map (26 Oct 2026, Kyle: every cell when zoomed in,
# clickable with its own figures and chart). The page asks for the cells in
# view once it is zoomed in far enough (viewer_app.js CELL_ZOOM); a view
# holding more is refused and the page says "zoom in further".
CELLS_MAX = 150_000


def api_cells(body: dict) -> dict:
    """The measured 15 m cells inside bbox [w, s, e, n]: row and column on the
    product grid and each cell's whole-record figures (field_points.
    cell_figures - what the client rasters and the KMZ read), for the page to
    colour as it colours the points. {"too_many": n} past CELLS_MAX."""
    from . import field_points as fp
    if _is_combined(body["config"]):
        return _combined_cells(body)
    _, _, ctx, maps = _project_context(body["config"])
    w, s_, e, n = (float(v) for v in body["bbox"])
    a = ctx.attrs
    y0, x0, dy, dx = (float(a[k]) for k in ("Y_FIRST", "X_FIRST", "Y_STEP", "X_STEP"))
    ny, nx = ctx.good.shape
    r0, r1 = sorted(((n - y0) / dy, (s_ - y0) / dy))
    c0, c1 = sorted(((w - x0) / dx, (e - x0) / dx))
    r0, r1 = max(0, int(np.floor(r0))), min(ny, int(np.ceil(r1)) + 1)
    c0, c1 = max(0, int(np.floor(c0))), min(nx, int(np.ceil(c1)) + 1)
    if r0 >= r1 or c0 >= c1:
        rows = cols = np.zeros(0, int)
    else:
        rr, cc = np.nonzero(ctx.good[r0:r1, c0:c1])
        rows, cols = rr + r0, cc + c0
    if rows.size > CELLS_MAX:
        return {"too_many": int(rows.size)}
    f = fp.cell_figures(ctx, maps, rows, cols)
    enc = lambda k, dt="<f4": _b64(f[k], dt)                                                  # noqa: E731
    return {"n": int(rows.size), "y0": y0, "x0": x0, "dy": dy, "dx": dx,
            "row": _b64(rows, "<i4"), "col": _b64(cols, "<i4"), "fast": _b64(f["fast"], "u1"),
            "vertical": enc("rate"), "vertical_rel": enc("rel_v"), "los": enc("los_rel"),
            "change_los": enc("change_los"), "snr": enc("snr"), "tc": enc("tc"), "unwrap": enc("unwrap")}


def _combined_cells(body: dict) -> dict:
    """api_cells for a combined project: its common grid's usable cells in
    the view, east-west in the line-of-sight slot (the page labels it)."""
    S = _combined_sources(body["config"])
    a = S.G.attrs
    w, s_, e, n = (float(v) for v in body["bbox"])
    y0, x0, dy, dx = (float(a[k]) for k in ("Y_FIRST", "X_FIRST", "Y_STEP", "X_STEP"))
    ny, nx = S.G.grid.shape
    r0, r1 = sorted(((n - y0) / dy, (s_ - y0) / dy))
    c0, c1 = sorted(((w - x0) / dx, (e - x0) / dx))
    r0, r1 = max(0, int(np.floor(r0))), min(ny, int(np.ceil(r1)) + 1)
    c0, c1 = max(0, int(np.floor(c0))), min(nx, int(np.ceil(c1)) + 1)
    if r0 >= r1 or c0 >= c1:
        rows = cols = np.zeros(0, int)
    else:
        rr, cc = np.nonzero(S.F["good"][r0:r1, c0:c1])
        rows, cols = rr + r0, cc + c0
        keep = S.usable(rows, cols)
        rows, cols = rows[keep], cols[keep]
    if rows.size > CELLS_MAX:
        return {"too_many": int(rows.size)}
    f = S.figures(rows, cols)
    vert_rel = f["rel_v"]
    enc = lambda v, dt="<f4": _b64(np.asarray(v), dt)                                          # noqa: E731
    return {"n": int(rows.size), "y0": y0, "x0": x0, "dy": dy, "dx": dx,
            "row": enc(rows, "<i4"), "col": enc(cols, "<i4"), "fast": enc(f["fast"], "u1"),
            "vertical": enc(f["rate"]), "vertical_rel": enc(vert_rel), "los": enc(f["east"]),
            "change_los": enc(f["east_change"]), "snr": enc(f["snr"]), "tc": enc(f["tc"]), "unwrap": enc(f["unwrap"])}


def _combined_cell(body: dict) -> dict:
    S = _combined_sources(body["config"])
    r, c = int(body["row"]), int(body["col"])
    ny, nx = S.G.grid.shape
    if not (0 <= r < ny and 0 <= c < nx) or not S.usable([r], [c])[0]:
        raise ValueError("no measurement from both passes in that map cell")
    fd = S.table([r], [c])
    T = len(fd["meta"]["dates"])
    rnd = lambda a: [None if not np.isfinite(v) else round(float(v), 4) for v in a]            # noqa: E731
    t = np.asarray(fd["meta"]["t"])
    bg = float(fd["meta"]["bg_v"]) if fd["meta"].get("anchored") else 0.0

    def own(a, vertical):                                     # zeroed on its own fit at the first date (point_series)
        y = np.asarray(a[0], float)
        ok = np.isfinite(y)
        f = np.polyfit(t[ok], y[ok], 1) if ok.sum() > 2 else (0.0, 0.0)
        return y - f[1] + (bg * t if vertical else 0.0)
    pt = _combined_point(fd, 0)
    return {"html": pt["html"], "fast": False, "lat": pt["lat"], "lon": pt["lon"], "row": r, "col": c,
            "own_vert_mm": rnd(own(fd["own_up"], True)), "own_los_mm": rnd(own(fd["own_east"], False)),
            "vert": pt["vert"], "los": pt["los"], "rate_vert_mm_yr": float(fd["rate"][0]),
            "rate_los_mm_yr": float(fd["east"][0]), "sd_vert_mm_yr": float(fd["sd_v"][0]), "up": 1.0,
            "dates_n": T}


def api_cell(body: dict) -> dict:
    """One 15 m cell's balloon (the client KMZ's point balloon, its chart left
    to the page) and its series: its own cell and the 3 x 3 cells around it
    (field_points.cells_from_frames - gather's numbers)."""
    from . import client_kmz as ck
    from . import field_points as fp
    from . import viewer_bundle as vb
    if _is_combined(body["config"]):
        return _combined_cell(body)
    _, fd0, _ = _table(body["config"])
    _, _, ctx, maps = _project_context(body["config"])
    r, c = int(body["row"]), int(body["col"])
    ny, nx = ctx.good.shape
    if not (0 <= r < ny and 0 <= c < nx and ctx.good[r, c]):
        raise ValueError("no measurement in that map cell")
    one = fp.cells_from_frames(ctx, maps, [r], [c], fd0["frame_los"], fd0["meta"])
    m = one["meta"]
    name = (m.get("names") or ["Study area"])[0]
    desc = ck.point_parts(one, 0, name, [(-1e9, 1e9, (0, 0, 0), "")], False, m["dates"][0], m["dates"][-1],
                          ck.decimal_year(m["dates"][0]))["desc"]
    desc = re.sub(r"<svg.*?</svg>", "", desc, flags=re.S)
    T = len(m["dates"])
    rnd = lambda a: [None if not np.isfinite(v) else round(float(v), 4) for v in a]            # noqa: E731
    vert, los = _series(one, 0)
    return {"html": ck._no_neg_zero(desc), "fast": bool(one["fast"][0]),
            "lat": float(one["lat"][0]), "lon": float(one["lon"][0]), "row": r, "col": c,
            "own_vert_mm": rnd(vb.point_series(one, 0, 0, T - 1, True)),
            "own_los_mm": rnd(vb.point_series(one, 0, 0, T - 1, False)),
            "vert": vert, "los": los,
            "rate_vert_mm_yr": float(one["rate"][0]), "rate_los_mm_yr": float(one["los_rel"][0]),
            "sd_vert_mm_yr": float(one["sd_v"][0]), "up": float(one["up"][0])}


def _full_table(config: str, rings: list, fd0: dict) -> dict:
    """Every measured product-grid cell (15 m) inside the polygon, with the
    figures the client KMZ gives its points (field_points.gather: the same
    rule, the same farmland frames, each cell's own series for date
    windows), computed now from the project's products. The context is
    loaded once per version of client_report.json."""
    import math
    from . import field_points as fp
    from . import results as rs
    if not rings or len(rings[0]) < 3:
        raise ValueError("draw a polygon with at least three corners")
    cfg, rep, ctx, _ = _project_context(config)
    mask = rs.ring_mask(rings[0], ctx.attrs, ctx.good.shape)
    for hole in rings[1:]:
        mask &= ~rs.ring_mask(hole, ctx.attrs, ctx.good.shape)
    mask &= ctx.good
    rows, cols = np.nonzero(mask)
    if not rows.size:
        raise ValueError("no measured map cells inside this polygon")
    if rows.size > FULL_RES_MAX_CELLS:
        raise ValueError(_too_many_cells(rows.size))
    fd = fp.gather(cfg, rep, ctx, [(rows, cols)])
    for k, v in fd0["meta"].items():                          # names, colours, title: the viewer table's
        fd["meta"].setdefault(k, v)
    # a date window's single-point noise from the viewer table's own farmland
    # sample (viewer_bundle.far_sample), not one drawn around these cells: the
    # cells then read what the points read over any dates
    from . import viewer_bundle as vb
    S0, K0 = vb.far_sample(fd0)
    sel0 = (fd0["group"] == 0) & fd0["far"].astype(bool)
    fd.update({"far": np.zeros(rows.size, bool), "far_los": np.asarray(S0, np.float32), "far_cls": K0.astype(np.int8),
               "far_row": np.concatenate([fd0["row"][sel0], fd0["far_row"]]).astype(np.int32),
               "far_col": np.concatenate([fd0["col"][sel0], fd0["far_col"]]).astype(np.int32)})
    for k in ("far_cells_by_class", "far_sample_per_class", "far_partial_nan_cells"):
        if k in fd0["meta"]:
            fd["meta"][k] = fd0["meta"][k]
    py, px = ctx.sg.pixel_metres(ctx.attrs)
    fd["meta"]["spacing_m"] = math.sqrt(py * px)
    return fd


POLYGON_LINES = 300            # individual cells drawn behind a polygon's average
_CELL_TABLES: dict = {}         # (config, rings) -> the polygon's 15 m table, the last few


def _cell_table(config: str, rings: list, fd0: dict) -> dict:
    """_full_table, kept for the last few polygons so a change of dates (or
    an export after the panel) does not gather the cells again."""
    key = (str(config), json.dumps(rings), fd0["meta"].get("numbers_hash"))
    hit = _CELL_TABLES.get(key)
    if hit is None:
        hit = _full_table(config, rings, fd0)
        while len(_CELL_TABLES) >= 4:
            _CELL_TABLES.pop(next(iter(_CELL_TABLES)))
        _CELL_TABLES[key] = hit
    elif hit["row"].size > FULL_RES_MAX_CELLS:               # the cap holds for a kept table too
        raise ValueError(_too_many_cells(hit["row"].size))
    return hit


def _combined_cell_fd(body: dict) -> dict:
    """Every usable common-grid cell inside the polygon as a combined table
    (combine_viewer.Sources.table), kept for the last few polygons."""
    from . import results as rs
    rings = body["rings"]
    if not rings or len(rings[0]) < 3:
        raise ValueError("draw a polygon with at least three corners")
    S = _combined_sources(body["config"])
    key = ("combined", str(body["config"]), json.dumps(rings), S.cc.name)
    fd = _CELL_TABLES.get(key)
    if fd is None:
        mask = rs.ring_mask(rings[0], S.G.attrs, S.G.grid.shape)
        for hole in rings[1:]:
            mask &= ~rs.ring_mask(hole, S.G.attrs, S.G.grid.shape)
        rows, cols = np.nonzero(mask & S.F["good"])
        if rows.size > FULL_RES_MAX_CELLS:
            raise ValueError(_too_many_cells(rows.size))
        fd = S.table(rows, cols)
        if not fd["row"].size:
            raise ValueError("no cells measured from both passes inside this polygon")
        fd["meta"]["spacing_m"] = float(math.sqrt(np.prod(S.G.grid.posting_m())))
        while len(_CELL_TABLES) >= 4:
            _CELL_TABLES.pop(next(iter(_CELL_TABLES)))
        _CELL_TABLES[key] = fd
    return fd


def _combined_polygon_cells(body: dict) -> dict:
    """api_polygon_cells for a combined project: every usable common-grid
    cell inside, with its series from both passes (combine_viewer.Sources),
    over the whole record (no date windows in the combined view)."""
    fd = _combined_cell_fd(body)
    idx = np.arange(fd["row"].size)
    st = _combined_poly(fd, idx)
    mm = lambda a: [None if v is None else round(float(v) * MM_IN, 3) for v in a]             # noqa: E731
    r1 = lambda a: [None if not np.isfinite(v) else round(float(v), 1) for v in a]             # noqa: E731
    t = np.asarray(fd["meta"]["t"])
    bg = float(fd["meta"]["bg_v"]) if fd["meta"].get("anchored") else 0.0
    pick = idx if idx.size <= POLYGON_LINES else idx[np.linspace(0, idx.size - 1, POLYGON_LINES).round().astype(int)]

    def zero(y):
        y = np.asarray(y, float)
        ok = np.isfinite(y)
        return y - np.polyfit(t[ok], y[ok], 1)[1] if ok.sum() > 2 else y
    st.update({"source": "cells", "kind": "report", "cell_m": float(fd["meta"]["spacing_m"]),
               "series_vertical_mm": mm(st.pop("series_vertical_in", [])),
               "series_los_mm": mm(st.pop("series_los_in", [])),
               "cells_vertical_mm": [None if not np.isfinite(v) else round(float(v), 3) for v in fd["rate"]],
               "cells_series_vertical_mm": [r1(zero(fd["own_up"][j]) + bg * t) for j in pick],
               "cells_series_los_mm": [r1(zero(fd["own_east"][j])) for j in pick],
               "cells_series_of": int(idx.size)})
    return st


def _combined_export_set(body: dict) -> tuple[dict, np.ndarray, bool]:
    """(table, rows, every cell?) a combined export reads: the ~100 m points
    inside the polygon, or every usable 15 m cell with resolution "full"."""
    if body.get("resolution") == "full":
        fd = _combined_cell_fd(body)
        return fd, np.arange(fd["row"].size), True
    _, fd, _ = _table(body["config"])
    return fd, _inside(fd, body["rings"]), False


def _combined_export_csv(body: dict) -> dict:
    """api_export_csv for a combined project: the vertical (both passes) and
    the east-west (indicative) of every point or cell inside, with each one's
    vertical series; the last row is the polygon's average."""
    from . import field_points as fp
    from .areas import formula_safe_text, one_line
    from .deliverables import _software
    fd, idx, full = _combined_export_set(body)
    if not idx.size:
        raise ValueError("no measurement points inside this polygon")
    m = fd["meta"]
    name = one_line(body.get("name")) or "polygon"
    st = _combined_poly(fd, idx)
    g = m.get("ground") or "ground"
    src = ", ".join(f"{x['project']} ({x['direction']})" for x in m.get("sources") or [])
    buf = io.StringIO()
    for line in (
            f"# {formula_safe_text(m.get('title'))} - {formula_safe_text(name)}: {idx.size} "
            + (f"cells (every measured {m.get('spacing_m') or 15:.0f} m cell)" if full
               else f"measurement points (about {m.get('spacing_m') or 100:.0f} m apart)"),
            f"# both passes together: {formula_safe_text(src)}; dates {m['dates'][0]} to {m['dates'][-1]} "
            f"({len(m['dates'])} images); the whole record (no date windows in the combined view)",
            f"# software: {_software()}",
            f"# vertical_in_per_yr: vertical movement per year from both passes, east-west separated out, relative to "
            f"ordinary {g}" + (" + the regional GNSS motion" if m.get("anchored") else ""),
            f"# vertical_relative_in_per_yr: the same without the regional motion (vs ordinary {g})",
            "# east_west_in_per_yr: east-west movement per year, + = east; INDICATIVE, not a tested figure",
            "# signal_to_noise: |vertical relative to ordinary ground| / its noise; 1.96 or more stands out",
            "# vertical_in_<date>: the point's vertical series (both passes, each date solved), zeroed on its own "
            "straight-line fit; the last row is the polygon's average",
            "# negative vertical = sinking; north-south movement is not measured"):
        buf.write(line + "\n")
    w = csv.writer(buf)
    w.writerow(["latitude", "longitude", "vertical_in_per_yr", "vertical_relative_in_per_yr", "east_west_in_per_yr",
                "signal_to_noise", "temporal_coherence"] + [f"vertical_in_{d}" for d in m["dates"]])
    f4 = lambda v: "" if v is None or not np.isfinite(v) else f"{v:.4f}"                    # noqa: E731
    V = fp.series_vertical_in_rows(fd, idx)
    for row, j in enumerate(idx):
        w.writerow([f"{fd['lat'][j]:.6f}", f"{fd['lon'][j]:.6f}", f4(fd["rate"][j] / MM_IN),
                    f4(fd["rel_v"][j] / MM_IN), f4(fd["east"][j] / MM_IN), f4(fd["snr"][j]), f"{fd['tc'][j]:.3f}"]
                   + [f4(v) for v in V[row]])
    with __import__("warnings").catch_warnings():
        __import__("warnings").simplefilter("ignore", RuntimeWarning)
        avg = np.nanmean(V, axis=0)
    w.writerow(["polygon average", "", f4(st.get("vertical_in_yr")), f4(st.get("relative_in_yr")),
                f4(st.get("los_in_yr")), "", f4(st.get("mean_coherence"))] + [f4(v) for v in avg])
    return {"filename": f"{_safe_name(name)}_combined_{'every_cell' if full else 'points'}.csv",
            "text": buf.getvalue()}


def _combined_export_kmz(body: dict) -> dict:
    """api_export_kmz for a combined project: the polygon (its average and
    chart) and the points or cells inside, through the client KMZ's shared-
    balloon writer (client_kmz.points_kml), each balloon combine_viewer's."""
    from . import client_kmz as ck
    from . import combine_viewer as cv
    from .areas import one_line
    from .deliverables import _software
    fd, idx, full = _combined_export_set(body)
    if not idx.size:
        raise ValueError("no measurement points inside this polygon")
    m = fd["meta"]
    name = one_line(body.get("name")) or "polygon"
    st = _combined_poly(fd, idx)
    bins = [(lo, hi, tuple(rgb), lab) for lo, hi, rgb, lab in (m.get("bins") or [])] or ck.point_bins(m.get("lim_in") or 0.6)
    y0 = ck.decimal_year(m["dates"][0])
    pstyles, pm_list = ck.points_kml([cv.kmz_point_parts(fd, int(j), name, bins, bool(m.get("relative_colours", True)), y0)
                                      for j in idx], bins)
    t = np.asarray(m["t"])
    chart = ck.svg_chart(t, np.array([np.nan if v is None else v for v in st.get("series_vertical_in", [])], float),
                         f"{name}: average vertical movement of {st['n']} " + ("cells" if full else "points"),
                         start_year=y0)
    g = html.escape(m.get("ground") or "ground")
    f2 = lambda v, u="": "—" if v is None or not np.isfinite(v) else f"{v:+.2f}{u}"         # noqa: E731
    src = ", ".join(f"{x['project']} ({x['direction']})" for x in m.get("sources") or [])
    rows = "".join(f"<tr><td>{a}</td><td>{b}</td></tr>" for a, b in (
        ("Points", f"{st['n']} - every measured {m.get('spacing_m') or 15:.0f} m cell" if full
         else f"{st['n']} (about {m.get('spacing_m') or 100:.0f} m apart)"),
        ("Vertical, per year (average)", f2(st.get("vertical_in_yr"), " in")
         + (" with the regional GNSS motion" if m.get("anchored") else f" relative to ordinary {g}")),
        ("Middle 80% of points", f"{f2(st.get('vertical_in_yr_p10'))} to {f2(st.get('vertical_in_yr_p90'))} in/yr"),
        ("East-west, per year (indicative)", f2(st.get("los_in_yr"), " in") + "; + = east"),
        ("Points standing out (vertical signal-to-noise 1.96 or more)",
         "—" if st.get("percent_standing_out") is None else f"{st['percent_standing_out']:.0f}%")))
    desc = (f"<b>{html.escape(name)}</b><table style='font-size:12px'>{rows}</table>{chart}<br><small>"
            + html.escape(f"Both passes together ({src}), the whole record: the plain average of the points inside "
                          "this outline - a look, not a tested figure. East-west is indicative. Negative vertical = "
                          "sinking.") + "</small>")
    ring = " ".join(f"{x:.6f},{y:.6f},0" for x, y in np.asarray(body["rings"][0], float)[:, :2])
    kml = (f"<?xml version='1.0' encoding='UTF-8'?><kml xmlns='http://www.opengis.net/kml/2.2'><Document>"
           f"<name>{html.escape(m.get('title') or '')} - {html.escape(name)}</name><open>1</open>"
           f"<ExtendedData><Data name='combined_from'><value>{html.escape(src)}</value></Data>"
           f"<Data name='software'><value>{html.escape(_software())}</value></Data>"
           f"<Data name='dates'><value>{m['dates'][0]} to {m['dates'][-1]} (whole record)</value></Data></ExtendedData>"
           f"{pstyles}<Style id='poly'><LineStyle><color>ff0b76d9</color><width>2.5</width></LineStyle>"
           f"<PolyStyle><fill>0</fill></PolyStyle></Style>"
           f"<Placemark><name>{html.escape(name)}</name><styleUrl>#poly</styleUrl><description><![CDATA[{desc}]]>"
           f"</description><Polygon><outerBoundaryIs><LinearRing><coordinates>{ring}</coordinates></LinearRing>"
           f"</outerBoundaryIs></Polygon></Placemark>"
           f"<Folder><name>Measurement points ({idx.size}{', every measured cell' if full else ''})</name>"
           f"{''.join(pm_list)}</Folder></Document></kml>")
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("doc.kml", ck._no_neg_zero(kml))
        z.writestr(ck.DOT, ck.dot_png())
    return {"filename": f"{_safe_name(name)}_combined{'_every_cell' if full else ''}.kmz",
            "b64": base64.b64encode(out.getvalue()).decode(), "points": int(idx.size)}


def api_polygon_cells(body: dict) -> dict:
    """The polygon panel from every measured 15 m cell inside (26 Oct 2026,
    Kyle: a small polygon held one or two ~100 m points): _poly_stats over
    the cells for the request's dates - the same figures the page computes
    for the points - with the mean series in mm and each cell's vertical
    rate (the histogram). The dates follow window_figures as for the points."""
    if _is_combined(body["config"]):
        return _combined_polygon_cells(body)
    _, fd0, _ = _table(body["config"])
    fd = _cell_table(body["config"], body["rings"], fd0)
    W, fdw = _window(fd, body)
    idx = np.arange(fdw["row"].size)
    st = _poly_stats(fdw, idx)
    ok = idx[~np.asarray(fdw["fast"], bool)]
    mm = lambda a: [None if v is None else round(float(v) * MM_IN, 3) for v in a]             # noqa: E731
    r3 = lambda a: [None if not np.isfinite(v) else round(float(v), 3) for v in a]             # noqa: E731
    # each cell's own series as the average takes it (point_series: zeroed on
    # its fit at the first image), drawn grey behind the average; an even
    # sample above POLYGON_LINES so the chart stays readable and light
    from . import viewer_bundle as vb
    T = len(fdw["meta"]["dates"])
    pick = ok if ok.size <= POLYGON_LINES else ok[np.linspace(0, ok.size - 1, POLYGON_LINES).round().astype(int)]
    r1 = lambda a: [None if not np.isfinite(v) else round(float(v), 1) for v in a]             # noqa: E731
    st.update({"source": "cells", "kind": W["kind"], "cell_m": float(fd["meta"].get("spacing_m") or 15.0),
               "series_vertical_mm": mm(st.pop("series_vertical_in", [])),
               "series_los_mm": mm(st.pop("series_los_in", [])),
               "cells_vertical_mm": r3(np.asarray(fdw["rate"], float)[ok]),
               "cells_series_vertical_mm": [r1(vb.point_series(fdw, int(j), 0, T - 1, True)) for j in pick],
               "cells_series_los_mm": [r1(vb.point_series(fdw, int(j), 0, T - 1, False)) for j in pick],
               "cells_series_of": int(ok.size)})
    return st


def api_export_kmz(body: dict) -> dict:
    """The polygon (its average and chart) and the points inside it, written
    by the client KMZ's own point code (client_kmz.points_kml). With
    resolution "full", every measured 15 m cell inside it (_full_table)
    instead of the ~100 m points."""
    from . import client_kmz as ck
    if _is_combined(body["config"]):                     # both passes (combine_viewer, 28 Oct 2026)
        return _combined_export_kmz(body)
    _, fd0, stale = _table(body["config"])
    full = body.get("resolution") == "full"
    if full:
        fd0 = _cell_table(body["config"], body["rings"], fd0)
        idx = np.arange(fd0["row"].size)
    else:
        idx = _inside(fd0, body["rings"])
    if not idx.size:
        raise ValueError("no measurement points inside this polygon")
    W, fd = _window(fd0, body)
    m = fd["meta"]
    from .areas import one_line
    name = one_line(body.get("name")) or "polygon"
    st = _poly_stats(fd, idx)
    bins = [(lo, hi, tuple(rgb), lab) for lo, hi, rgb, lab in (m.get("bins") or [])] or ck.point_bins(m.get("lim_in") or 0.6)
    d0, d1 = m["dates"][0], m["dates"][-1]
    y0 = ck.decimal_year(d0)
    span_label = "over the record" if W["kind"] == "report" else "over these dates"
    pstyles, pm_list = ck.points_kml([ck.point_parts(fd, int(j), name, bins, bool(m.get("relative_colours")), d0, d1, y0,
                                                     span_label=span_label) for j in idx], bins)
    pms = "".join(pm_list)
    t = np.asarray(m["t"])
    chart = ck.svg_chart(t, np.array([np.nan if v is None else v for v in st.get("series_vertical_in", [])], float),
                         f"{name}: average movement of {st['n'] - st['n_too_fast']} points", start_year=y0)
    g = html.escape(m.get("ground") or "farmland")
    f2 = lambda v, u="": "—" if v is None else f"{v:+.2f}{u}"                               # noqa: E731
    rows = "".join(f"<tr><td>{a}</td><td>{b}</td></tr>" for a, b in (
        ("Points", (f"{st['n']} - every measured {m.get('spacing_m') or 15:.0f} m cell" if full else
                    f"{st['n']} (about {m.get('spacing_m') or 100:.0f} m apart)")
                   + (f"; {st['n_too_fast']} too fast to measure" if st["n_too_fast"] else "")),
        ("Vertical, per year (average)", f2(st.get("vertical_in_yr"), " in")
         + ("" if m.get("anchored") else f" relative to ordinary {g}")),
        ("Middle 80% of points", f"{f2(st.get('vertical_in_yr_p10'))} to {f2(st.get('vertical_in_yr_p90'))} in/yr"),
        (f"Line of sight, per year (vs ordinary {g})", f2(st.get("los_in_yr"), " in")),
        (f"Line of sight, {ck._date_text(d0)} to {ck._date_text(d1)}", f2(st.get("los_change_in"), " in")),
        ("Points standing out (signal-to-noise 1.96 or more)",
         "—" if st.get("percent_standing_out") is None else f"{st['percent_standing_out']:.0f}%"
         + (f" (ordinary {g}: {m['snr_chance_pct']:.1f}% by chance)" if m.get("snr_chance_pct") is not None else ""))))
    look = ("" if W["kind"] == "report" else
            f" {ck._date_text(d0)} to {ck._date_text(d1)}: each point's single-point noise over these dates is "
            "measured on a sample of the report's farmland cells - a look, not a tested figure.")
    desc = (f"<b>{html.escape(name)}</b><table style='font-size:12px'>{rows}</table>{chart}<br><small>The plain average "
            "of the points inside this outline; neighbouring points share ground, so this is a look, not a tested "
            "figure. Negative = sinking. Vertical assumes all the movement is up or down." + html.escape(look)
            + (f" {html.escape(stale)}" if stale else "") + "</small>")
    ring = " ".join(f"{x:.6f},{y:.6f},0" for x, y in np.asarray(body["rings"][0], float)[:, :2])
    from .deliverables import _software
    kml = (f"<?xml version='1.0' encoding='UTF-8'?><kml xmlns='http://www.opengis.net/kml/2.2'><Document>"
           f"<name>{html.escape(m.get('title') or '')} - {html.escape(name)}</name><open>1</open>"
           # which numbers and which code, as the client KMZ (16 Oct 2026, readiness review OUT-01)
           f"<ExtendedData><Data name='numbers_hash'><value>{html.escape(str(m.get('numbers_hash') or ''))}</value></Data>"
           f"<Data name='software'><value>{html.escape(_software())}</value></Data>"
           f"<Data name='dates'><value>{d0} to {d1} ({W['kind']})</value></Data></ExtendedData>"
           f"{pstyles}<Style id='poly'><LineStyle><color>ff0b76d9</color><width>2.5</width></LineStyle>"
           f"<PolyStyle><fill>0</fill></PolyStyle></Style>"
           f"<Placemark><name>{html.escape(name)}</name><styleUrl>#poly</styleUrl><description><![CDATA[{desc}]]>"
           f"</description><Polygon><outerBoundaryIs><LinearRing><coordinates>{ring}</coordinates></LinearRing>"
           f"</outerBoundaryIs></Polygon></Placemark>"
           f"<Folder><name>Measurement points ({idx.size}{', every measured cell' if full else ''})</name>"
           f"{pms}</Folder></Document></kml>")
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("doc.kml", ck._no_neg_zero(kml))
        z.writestr(ck.DOT, ck.dot_png())
    tag = ("" if W["kind"] == "report" else f"_{W['first']}_{W['last']}") + ("_every_cell" if full else "")
    return {"filename": f"{_safe_name(name)}{tag}.kmz", "b64": base64.b64encode(out.getvalue()).decode(),
            "points": int(idx.size)}


def save_report_area(config: str, name: str, rings: list) -> dict:
    """Add the polygon to the project's report areas: its outline to
    <project>/context/report_<slug>.geojson, a line 'Name:<path>' to the
    config's report_areas (the config backed up first, and restored if it no
    longer loads), and a report layer to the builder's saved layers when
    there are any, so an Update from the builder keeps it."""
    from . import areas
    from . import client_report as cr
    from .config import load_config
    name = (name or "").strip()
    why = areas.client_name_problem(name)
    if why:
        raise ValueError(f"The name {why}")
    cfg = load_config(config)
    if any(lbl.casefold() == name.casefold() for lbl, _ in cr.parse_areas(cfg)):
        raise ValueError(f"The project already has a report area called {name!r}")
    ring = [[float(x), float(y)] for x, y in np.asarray(rings[0], float)[:, :2]]
    if ring[0] != ring[-1]:
        ring.append(ring[0])
    gj = {"type": "FeatureCollection", "features": [{"type": "Feature", "properties": {"name": name},
          "geometry": {"type": "Polygon", "coordinates": [ring] + [
              [[float(x), float(y)] for x, y in np.asarray(h, float)[:, :2]] for h in rings[1:]]}}]}
    P = Path(cfg.project_dir)
    ctx = P / "context"
    ctx.mkdir(parents=True, exist_ok=True)
    f = ctx / f"report_{areas.slug(name)}.geojson"
    if f.exists():
        raise ValueError(f"{f.name} already exists in the project's context folder; choose another name")
    f.write_text(json.dumps(gj))
    from .settings import projects_root
    spec = f"{name}:{areas._portable(f, projects_root())}"
    cp = Path(config)
    text = cp.read_text()
    backup = cp.with_name(cp.name + datetime.now().strftime(".bak_%Y%m%d_%H%M%S"))
    shutil.copy2(cp, backup)
    lines = text.splitlines()
    i = next((k for k, ln in enumerate(lines) if re.match(r"^report_areas\s*:", ln)), None)
    if i is None:
        lines += ["report_areas:", f"- {spec}"]
    elif re.match(r"^report_areas\s*:\s*\[\s*\]\s*$", lines[i]):
        lines[i:i + 1] = ["report_areas:", f"- {spec}"]
    elif re.match(r"^report_areas\s*:\s*$", lines[i]):
        k = i + 1
        while k < len(lines) and re.match(r"^\s*-\s", lines[k]):
            k += 1
        lines.insert(k, f"- {spec}")
    else:
        f.unlink(missing_ok=True)
        raise ValueError("The config writes report_areas in a form this page does not edit; add the line "
                         f"'- {spec}' under report_areas by hand")
    cp.write_text("\n".join(lines) + ("\n" if text.endswith("\n") else ""))
    try:
        got = [lbl for lbl, _ in cr.parse_areas(load_config(config))]
        if name not in got:
            raise ValueError(f"the config loads but has no area {name!r}")
    except Exception as exc:                                  # noqa: BLE001
        shutil.copy2(backup, cp)
        f.unlink(missing_ok=True)
        raise ValueError(f"The config did not take the new area ({exc}); it was put back as it was") from None
    saved = ctx / "builder_layers.json"
    if saved.exists():
        try:
            d = json.loads(saved.read_text())
            d.setdefault("layers", []).append({"role": "report", "name": name, "geojson": gj})
            saved.write_text(json.dumps(d))
        except (OSError, ValueError):
            pass
    return {"saved": str(f), "config": str(cp), "backup": str(backup),
            "message": f"{name!r} is now a report area. Rebuild the client report (Products in the dashboard) for "
                       f"its tested figures: rate with its 95% range, detection limit and chart."}


def api_save_area(body: dict) -> dict:
    _not_for_combined(body["config"], "Saving a report area (save it on one of its two projects)")
    return save_report_area(body["config"], body.get("name") or "", body["rings"])


WELL_CELL_REACH = 3        # cells: how far from a well its own map cell may be looked for


def api_well(body: dict) -> dict:
    """A well's own series (11 Nov 2026): the measured 15 m map cell the well
    sits in - or, when that cell has no measurement, the nearest measured
    cell within WELL_CELL_REACH cells - as api_cell gives it, with how far
    that cell's centre is from the well (offset_m)."""
    if _is_combined(body["config"]):
        raise ValueError("a combined view has no single map grid for a well's own series")
    _, _, ctx, _ = _project_context(body["config"])
    lat, lon = float(body["lat"]), float(body["lon"])
    a = ctx.attrs
    y0, x0, dy, dx = (float(a[k]) for k in ("Y_FIRST", "X_FIRST", "Y_STEP", "X_STEP"))
    ny, nx = ctx.good.shape
    r0, c0 = (lat - y0) / dy - 0.5, (lon - x0) / dx - 0.5      # cell centres sit half a step in
    rc, cc = int(round(r0)), int(round(c0))
    k = WELL_CELL_REACH
    best, bd = None, np.inf
    for r in range(max(0, rc - k), min(ny, rc + k + 1)):
        for c in range(max(0, cc - k), min(nx, cc + k + 1)):
            if ctx.good[r, c]:
                d = (r - r0) ** 2 + (c - c0) ** 2
                if d < bd:
                    best, bd = (r, c), d
    if best is None:
        raise ValueError(f"no measured map cell within {k} cells of this well")
    out = api_cell({"config": body["config"], "row": best[0], "col": best[1]})
    kx = math.cos(math.radians(lat))
    out["offset_m"] = round(math.hypot((out["lat"] - lat) * 111320.0, (out["lon"] - lon) * 111320.0 * kx), 1)
    return out


def api_wells(body: dict) -> dict:
    """The project's study point layers (wells) with each one's reading, the
    same rows the app's live map shows (app_server._study_points: from
    client_report.json's study_points once the products have run, else as
    listed in the file, no number yet). [] for a combined view or a project
    without point layers (11 Nov 2026)."""
    config = body["config"]
    if _is_combined(config):
        return {"layers": []}
    from . import app_server as app
    cfg, _, _ = _config_paths(config)
    out = []
    for L in app.study_layers(cfg):
        if L["geometry"] != "point":
            continue
        d = app._study_points(cfg, L)
        out.append({"id": L["id"], "label": L["label"], "kind": L["kind"], "from_report": d["from_report"],
                    "legend": d["legend"]["classes"], "points": d["points"]})
    return {"layers": out}


API = {"/api/projects": api_projects, "/api/wells": api_wells, "/api/well": api_well, "/api/load": api_load, "/api/point": api_point, "/api/polygon": api_polygon,
       "/api/export_csv": api_export_csv, "/api/export_kmz": api_export_kmz, "/api/save_area": api_save_area,
       "/api/cells": api_cells, "/api/cell": api_cell, "/api/polygon_cells": api_polygon_cells}
BINARY_API = {"/api/bundle": lambda body: bundle_bytes(body["config"])}


# ---------------------------------------------------------------- server
class _Handler(http.server.SimpleHTTPRequestHandler):
    port = PORT
    token = ""        # set per server session by serve() (16 Oct 2026, readiness review NET-06)
    # the page's files as they were when serve() started this server: a
    # dashboard left open across a code update kept serving its old code
    # while the page on disk changed, and the page came up unstyled
    # (19 Oct 2026). None (a handler made outside serve()) reads them from disk.
    assets: dict | None = None

    def log_message(self, *a):
        pass

    def _local(self) -> bool:
        ok = {f"localhost:{self.port}", f"127.0.0.1:{self.port}"}
        origin = self.headers.get("Origin")
        return self.headers.get("Host") in ok and (origin is None or origin.split("//", 1)[-1] in ok)

    def _send(self, code: int, obj) -> None:
        data = json.dumps(obj, default=str).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _send_bytes(self, data: bytes, ctype: str, cache: str = "no-store") -> None:
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", cache)
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        path = self.path.split("?")[0].split("#")[0]
        if path in ("/", ""):
            self.send_response(302)
            self.send_header("Location", "/" + PAGE)
            self.end_headers()
            return
        if not self._local():                       # the token goes only to this machine's own page
            return self._send(403, {"error": "only this machine's browser may open the viewer"})
        if path == "/" + PAGE:
            from .map_builder import send_page      # the page with this session's token
            return send_page(self, PAGE, {"Content-Security-Policy": CSP, "X-Content-Type-Options": "nosniff"})
        rel = path.lstrip("/")
        f = ASSETS.get(rel)                         # an exact name, never a path joined from the request
        data = (self.assets or {}).get(rel) if self.assets is not None else (
            f.read_bytes() if f is not None and f.is_file() else None)
        if f is None or data is None:
            return self._send(404, {"error": "not served"})
        return self._send_bytes(data, CONTENT_TYPES.get(f.suffix, "application/octet-stream"))

    def do_POST(self):
        from .map_builder import STALE_PAGE, token_ok
        path = self.path.split("?")[0]
        fn = API.get(path) or BINARY_API.get(path)
        if fn is None:
            return self._send(404, {"error": "no such endpoint"})
        if not self._local():
            return self._send(403, {"error": "only the viewer page on this machine may call this"})
        if not token_ok(self.headers, self.token):  # every call, Origin or not (NET-06)
            return self._send(403, {"error": STALE_PAGE})
        try:
            n = int(self.headers.get("Content-Length") or 0)
            if n > 20 * 1024 * 1024:
                raise ValueError("request too large")
            body = json.loads(self.rfile.read(n) or b"{}")
            if path in BINARY_API:
                return self._send_bytes(fn(body), "application/octet-stream")
            self._send(200, fn(body))
        except Exception as exc:                             # noqa: BLE001
            self._send(400, {"error": str(exc) or type(exc).__name__})


def _port_busy(port: int) -> bool:
    with socket.socket() as s:
        return s.connect_ex(("127.0.0.1", port)) == 0


def serve(port: int = PORT) -> str:
    """Start the server in a daemon thread (once) and return the page URL.

    The page and its files are read when the server starts and served as
    they were then; a server already running (this process's or another's)
    is left with its own, so a code update on disk can never mix an old
    server with a new page. Restart the dashboard (or `insar viewer`) to
    serve the new page."""
    url = f"http://localhost:{port}/{PAGE}"
    if "httpd" in _SERVER or _port_busy(port):
        return url
    SERVE_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(PKG / PAGE, SERVE_DIR / PAGE)
    assets = {rel: f.read_bytes() for rel, f in ASSETS.items() if f.is_file()}
    import secrets
    handler = type("Handler", (_Handler,), {"port": port, "token": secrets.token_urlsafe(32), "assets": assets})
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", port), partial(handler, directory=str(SERVE_DIR)))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    _SERVER["httpd"] = httpd
    return url


def main() -> None:
    import sys
    from .map_builder import open_in_browser
    url = serve()
    open_in_browser(url)
    print(f"Results viewer at {url}  (Ctrl-C to stop the server)")
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        print()
        sys.exit(0)


if __name__ == "__main__":
    main()
