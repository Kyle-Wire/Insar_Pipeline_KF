"""The project builder: a map in the browser where you draw or upload the
project area, the mine outlines and any extra report areas, check what the
satellite archive holds, and create the project. Every processing setting
comes from the production defaults (areas.write_project); the rare change is
`e` on the dashboard's preflight screen.

    insar builder          (or b in the dashboard, or python -m insar_pipeline.map_builder)

Why a local server and not a file: the page needs Python for the uploads
(shapefiles in any projection), the archive check and writing the project;
and OpenStreetMap's tiles and search refuse pages opened as files. The
server binds to 127.0.0.1 only, refuses requests from other web pages
(Origin/Host checks), and takes an API call only with the secret it wrote
into the page it served this session (page_with_token), so nothing but the
builder page itself can use it.

What leaves this computer: place searches go to OpenStreetMap Nominatim
through /api/geocode (with areas.user_agent naming the firm and a contact,
as its usage policy asks; the page tells the user to type a town or road,
never a client's or site's name); map tiles are fetched by the browser from
the chosen basemap (USGS by default, so the viewed area is seen by that
server); the archive check sends the area to ASF.
"""
from __future__ import annotations

import base64
import hmac
import http.server
import json
import re
import secrets
import shutil
import socket
import subprocess
import sys
import threading
import time
import uuid
from datetime import date
from functools import partial
from pathlib import Path

PKG = Path(__file__).resolve().parent
PAGE = "project_builder.html"
PORT = 8765
SERVE_DIR = Path.home() / ".cache" / "insar_builder"
UPLOADS = SERVE_DIR / "uploads"
MAX_UPLOAD = 300 * 1024 * 1024                      # a statewide mine layer is ~100 MB
DISPLAY_POINTS = 150_000                            # thin the map copy above this; never the saved one
ROLES = ("area", "mine", "report", "study")
MINE_KINDS = ("underground", "surface", "other", "dated")
# The studies step 0 offers besides the default (A5 of
# review_2026-11-07/PLAN_oil_gas_slopes.md, "oil_gas only"): every other study,
# offered in domains.py or not, is listed greyed out with its reason, since its
# report wording has not been checked yet (decision G2).
BUILDER_STUDIES = ("oil_gas",)
NOT_IN_BUILDER = "its report wording is not checked yet; the builder offers oil and gas fields first"
STUDY_SUFFIXES = (".geojson", ".json", ".zip", ".kml", ".kmz", ".shp", ".gpkg", ".fgb")
_SERVER: dict = {}
# the page's map library, served from the package's vendored copy, as the results
# map does (results_viewer.ASSETS; checked against cdnjs's published SRI hashes):
# no CDN and no web fonts (2 Nov 2026, readiness review NET-04). Exact names only.
VENDOR = PKG / "viewer_assets" / "vendor"
ASSETS = {f"assets/vendor/{n}": VENDOR / n for n in (
    "leaflet.min.js", "leaflet.min.css", "leaflet.draw.js", "leaflet.draw.css",
    "images/layers.png", "images/layers-2x.png", "images/marker-icon.png", "images/marker-icon-2x.png",
    "images/marker-shadow.png", "images/spritesheet.png", "images/spritesheet-2x.png", "images/spritesheet.svg")}
CONTENT_TYPES = {".js": "text/javascript; charset=utf-8", ".css": "text/css; charset=utf-8", ".png": "image/png",
                 ".svg": "image/svg+xml"}


# ---------------------------------------------------------------- API
def _area_from_layer(layer: dict, label: str):
    """A layer as sent back by the page: the uploaded file itself when there
    is one (full detail), else the geometry drawn on the map."""
    from . import areas
    uid = layer.get("upload_id")
    if uid:
        return areas.load_polygon(_upload_main(uid), label)
    gj = layer.get("geojson")
    if not gj:
        raise ValueError(f"layer {label!r} has no geometry")
    return areas._from_geojson(gj, label)


def _upload_main(uid: str) -> Path:
    if not re.fullmatch(r"[0-9a-f]{32}", uid or ""):
        raise ValueError("bad upload id")
    d = UPLOADS / uid
    files = [p for p in d.iterdir()] if d.is_dir() else []
    if not files:
        raise ValueError("that upload has expired; upload the file again")
    rank = [".shp", ".zip", ".gpkg", ".geojson", ".json", ".kmz", ".kml", ".fgb"]
    files.sort(key=lambda p: rank.index(p.suffix.lower()) if p.suffix.lower() in rank else 99)
    return files[0]


def _thin(gj: dict) -> tuple[dict, bool]:
    feats = gj["features"]
    n = sum(len(r) for f in feats for r in f["geometry"]["coordinates"])
    if n <= DISPLAY_POINTS:
        return gj, False
    k = -(-n // DISPLAY_POINTS)
    for f in feats:                                          # outer rings and holes alike
        f["geometry"]["coordinates"] = [(r[::k] + [r[0]]) if len(r) > 4 * k else r
                                        for r in f["geometry"]["coordinates"]]
    return gj, True


def api_outline(body: dict) -> dict:
    """Uploaded file(s) -> the outline in lon/lat. A shapefile arrives as its
    .shp/.dbf/.shx/.prj together, or zipped."""
    from . import areas
    files = body.get("files") or []
    if not files:
        raise ValueError("no file")
    uid = uuid.uuid4().hex
    d = UPLOADS / uid
    d.mkdir(parents=True)
    size = 0
    for f in files:
        name = Path(str(f["name"])).name                  # never a path from the browser
        data = base64.b64decode(f["data"])
        size += len(data)
        if size > MAX_UPLOAD:
            shutil.rmtree(d)
            raise ValueError(f"upload larger than {MAX_UPLOAD >> 20} MB")
        (d / name).write_bytes(data)
        try:                                              # the file's own date: the layer's vintage
            t = float(f.get("lastModified") or 0) / 1000.0
            if t > 0:
                import os
                os.utime(d / name, (t, t))
        except (TypeError, ValueError, OSError):
            pass
    main = _upload_main(uid)
    if body.get("study_kind"):
        return _study_upload(uid, main, d, body)
    if main.suffix.lower() not in areas.POLYGON_SUFFIXES:
        shutil.rmtree(d)
        raise ValueError(f"{main.name}: upload a shapefile (.shp with .dbf .shx .prj, or zipped), "
                         "GeoJSON, KML/KMZ or GeoPackage")
    try:
        a = areas.load_polygon(main, body.get("label") or main.stem)
    except Exception:
        shutil.rmtree(d)
        raise
    gj, thinned = _thin(a.geojson())
    out = {"upload_id": uid, "label": a.label, "stem": main.stem, "geojson": gj, "polygons": len(a.rings),
           "holes": sum(len(h) for h in a.holes), "bbox": a.bbox, "size_km": a.size_km(), "thinned": thinned,
           "files": sorted(p.name for p in d.iterdir()), "source": _upload_source(uid)}
    if body.get("kind") == "dated":
        # the attribute table: which field holds the year mined (and the name)
        out["fields"] = areas.attribute_fields(main)
    return out


_GEOMETRY_TYPES = {"point": ("Point", "MultiPoint"), "line": ("LineString", "MultiLineString"),
                   "polygon": ("Polygon", "MultiPolygon")}


def _study_kind(study_key: str, kind: str):
    """The domains.LayerKind `kind` of the study, or ValueError."""
    from . import domains
    try:
        st = domains.study(study_key)
    except KeyError as exc:
        raise ValueError(str(exc).strip("'\"")) from None
    for L in st.layer_kinds:
        if L.key == kind:
            return L
    raise ValueError(f"the {st.label} study has no layer {kind!r}")


def _gis_features(path: str) -> dict:
    import geopandas as gpd
    df = gpd.read_file(path)
    if df.crs is None:
        df = df.set_crs(4326)
    return json.loads(df[df.geometry.notna()].to_crs(4326).to_json())


def study_features(src: Path, geometry: str, label: str) -> dict:
    """A study layer's file as a lon/lat FeatureCollection WITH its attribute
    columns (a well's status, plug date and type are read from the file, never
    typed into a form). Only features of the layer's geometry (point, line or
    polygon) are kept; a file holding none of them is refused. The outline
    files' own checks (size, content against extension) run first."""
    from . import areas
    p = areas.normalize_path(src)
    areas._open_outline(p)
    suf = p.suffix.lower()
    if suf not in STUDY_SUFFIXES:
        raise ValueError(f"{p.name}: upload a shapefile (.shp with .dbf .shx .prj, or zipped), GeoJSON, "
                         "GeoPackage or KML/KMZ")
    if suf in (".geojson", ".json"):
        fc = areas._read_json_file(p)
        crs = str(((fc.get("crs") or {}).get("properties") or {}).get("name") or "")
        if crs and not re.search(r"CRS84|EPSG:+4326$|EPSG::?4326\b", crs, re.I):
            fc = _gis_features(str(p))
    elif suf == ".zip":
        with areas._open_zip(p) as z:
            names = [n for n in z.namelist() if not n.startswith("__MACOSX")]
        inner = next((n for ext in (".shp", ".gpkg", ".geojson", ".json", ".kml") for n in names
                      if n.lower().endswith(ext)), None)
        if inner is None:
            raise ValueError(f"{p.name} holds no shapefile, GeoPackage, GeoJSON or KML")
        fc = _gis_features(f"zip://{p}!{inner}")
    elif suf == ".kmz":
        with areas._open_zip(p) as z:
            inner = next((n for n in z.namelist() if n.lower().endswith(".kml")), None)
        if inner is None:
            raise ValueError(f"{p.name} holds no KML")
        fc = _gis_features(f"zip://{p}!{inner}")
    else:
        fc = _gis_features(str(p))
    if fc.get("type") == "Feature":
        fc = {"type": "FeatureCollection", "features": [fc]}
    feats = fc.get("features") or []
    want = _GEOMETRY_TYPES[geometry]
    keep = [{"type": "Feature", "properties": dict(f.get("properties") or {}), "geometry": f["geometry"]}
            for f in feats if (f.get("geometry") or {}).get("type") in want]
    if not keep:
        found = sorted({str((f.get("geometry") or {}).get("type")) for f in feats}) or ["no features"]
        raise ValueError(f"{label}: needs {geometry}s; {p.name} holds {', '.join(found)}")
    areas._check_features(len(keep), label)
    from shapely.geometry import shape
    for k, f in enumerate(keep):
        try:
            x0, y0, x1, y1 = shape(f["geometry"]).bounds
        except (ValueError, TypeError, AttributeError) as exc:
            raise ValueError(f"{label}: feature {k + 1} has unreadable coordinates ({exc})") from None
        if max(abs(x0), abs(x1)) > 180 or max(abs(y0), abs(y1)) > 90:
            raise ValueError(f"{label}: the coordinates are not longitude/latitude; a projected file must name "
                             "its coordinate system (a .prj beside a shapefile)")
    return {"type": "FeatureCollection", "features": keep}


def _thin_any(c, k: int):
    """Every k-th vertex of every ring or line (ends kept), at any nesting depth."""
    if not c or isinstance(c[0], (int, float)):
        return c
    if isinstance(c[0], list) and c[0] and isinstance(c[0][0], (int, float)):
        return (c[::k] + [c[-1]]) if len(c) > 4 * k else c
    return [_thin_any(x, k) for x in c]


def _study_upload(uid: str, main: Path, d: Path, body: dict) -> dict:
    """An uploaded study layer: its features with their columns, which of the
    layer's attribute columns the file has, and a thinned copy for the map."""
    L = _study_kind(str(body.get("study") or ""), str(body.get("study_kind")))
    try:
        fc = study_features(main, L.geometry, L.label)
    except Exception:
        shutil.rmtree(d)
        raise
    cols = sorted({k for f in fc["features"] for k in f["properties"]})
    low = {c.lower(): c for c in cols}
    n = sum(1 for f in fc["features"] for _ in _bounds(f["geometry"]["coordinates"]))
    k = -(-n // DISPLAY_POINTS) if n > DISPLAY_POINTS else 1
    show = {"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {},
         "geometry": {"type": f["geometry"]["type"], "coordinates": _thin_any(f["geometry"]["coordinates"], k)}}
        for f in fc["features"]]} if k > 1 else fc
    xs = [b for f in fc["features"] for b in _bounds(f["geometry"]["coordinates"])]
    return {"upload_id": uid, "label": L.label, "stem": main.stem, "geojson": show, "features": len(fc["features"]),
            "geometry": L.geometry, "thinned": k > 1, "files": sorted(x.name for x in d.iterdir()),
            "columns": cols, "found": [a for a in L.attributes if a.lower() in low],
            "missing": [a for a in L.attributes if a.lower() not in low],
            "bbox": [min(x[0] for x in xs), min(x[1] for x in xs), max(x[0] for x in xs), max(x[1] for x in xs)]
            if xs else None, "source": _upload_source(uid)}


def _bounds(c):
    """Every (lon, lat) of a coordinate array, at any nesting depth."""
    if c and isinstance(c[0], (int, float)):
        yield (float(c[0]), float(c[1]))
        return
    for x in c or []:
        yield from _bounds(x)


def _card_sections(lines: list) -> dict:
    """evidence.card's lines as the two step-6 cards: what the study will give
    (each output with its tier) and what it cannot tell you / not yet."""
    out = {"will_give": [], "cannot": [], "not_yet": [], "notes": []}
    sec = None
    for ln in lines[1:]:
        t = ln.strip()
        if t in ("Will give:", "Cannot tell you:", "Not yet:"):
            sec = {"Will give:": "will_give", "Cannot tell you:": "cannot", "Not yet:": "not_yet"}[t]
        elif t.startswith("- ") and sec:
            out[sec].append(t[2:])
        elif t and not t.startswith(("Layers:", "Not offered yet:")):
            sec = None
            out["notes"].append(t)
    return out


def api_studies(_: dict) -> dict:
    """Step 0: the default study first (today's builder), then the studies the
    builder offers with their tier badge and their two cards, then the rest
    greyed out with the reason. Map layers and the study choice only: nothing
    here is a processing setting."""
    from . import domains, evidence
    try:
        ev = evidence.load()
    except Exception:                                        # noqa: BLE001 - the cards then say so
        ev = None
    order = ([domains.study(domains.DEFAULT_STUDY)] + [domains.study(k) for k in BUILDER_STUDIES]
             + [s for s in domains.STUDIES if s.key != domains.DEFAULT_STUDY and s.key not in BUILDER_STUDIES])
    out = []
    for st in order:
        default = st.key == domains.DEFAULT_STUDY
        selectable = default or (st.key in BUILDER_STUDIES and st.builder_status == "offered")
        row = {"key": st.key, "label": st.label, "question": st.question, "default": default,
               "selectable": selectable, "badge": None if default else ("Screening" if selectable else None),
               "reason": "" if selectable else (st.builder_reason if st.builder_status != "offered"
                                                else NOT_IN_BUILDER),
               "layer_kinds": [{"key": L.key, "label": L.label, "geometry": L.geometry,
                                "attributes": list(L.attributes)} for L in st.layer_kinds]}
        if selectable and not default:
            try:
                row.update(_card_sections(evidence.card(st, ev)) if ev is not None else
                           {"will_give": [], "cannot": [t for t, _ in st.cannot], "not_yet": [t for t, _ in st.not_yet],
                            "notes": ["evidence.yaml could not be read: tiers not shown"]})
            except Exception as exc:                         # noqa: BLE001 - shown on the card
                row.update({"will_give": [], "cannot": [f"{t} ({g})" for t, g in st.cannot],
                            "not_yet": [f"{t} ({g})" for t, g in st.not_yet], "notes": [f"tiers not shown: {exc}"]})
        out.append(row)
    return {"default": domains.DEFAULT_STUDY, "studies": out}


def _upload_source(uid: str) -> str:
    """'mines.zip, file dated 2023-04-11': what an uploaded layer was, for
    mine_layer_sources when nobody types its source and vintage."""
    main = _upload_main(uid)
    return f"{main.name}, file dated {time.strftime('%Y-%m-%d', time.localtime(main.stat().st_mtime))}"


def api_check(body: dict) -> dict:
    """What the archive holds over the area, the GNSS stations near it and
    the processing box."""
    from . import areas
    a = _area_from_layer(body["area"], "project area")
    start, end = body.get("start") or "2016-01-01", body.get("end") or date.today().isoformat()
    from .config import GNSS_TIE_RADIUS_KM
    pad = float(body.get("pad_km") or 5)
    proc = a.padded(pad)
    r = areas.archive_check(a, start, end, box=proc.bbox)
    try:
        # the stations the tie will USE (inside the tie radius, >= 3 years of
        # overlap with these dates), apart from those nearby that it will not
        g = areas.gnss_preview(a, start, end, GNSS_TIE_RADIUS_KM)
        r["gnss"] = [{k: x.get(k) for k in ("site", "lat", "lon", "distance_km", "will_use", "overlap_years")}
                     for x in g["stations"]]
        r["gnss_used"], r["gnss_not_used"] = g["used"], g["not_used"]
        r["gnss_warning"], r["gnss_source"] = g["warning"], g["source"]
    except Exception as exc:                                 # noqa: BLE001
        r["gnss"], r["gnss_error"] = [], str(exc)
    r["gnss_radius_km"] = GNSS_TIE_RADIUS_KM
    r["processing_box"] = proc.bbox
    r["size_km"] = a.size_km()
    try:
        # the products stage's memory for this box (the planner's rule)
        n_dates = max([t["dates"] for t in r["tracks"]] or [0])
        r["ram"] = areas.ram_estimate(proc, n_dates)
    except Exception as exc:                                 # noqa: BLE001
        r["ram"] = {"error": str(exc)[:120]}
    # sub-swaths and burst coverage of the padded box for each usable track
    # (the dashboard used the first track's swaths for whichever track was
    # chosen; write_project looks them up again for the chosen one)
    for t in [t for t in r["tracks"] if t["dates"] >= 60][:4]:
        try:
            c = areas.burst_coverage(proc, t["track"], t["direction"], {"area": a})
            t["swaths"], t["coverage"] = c["swaths"], min(c["covered"], c["targets"]["area"])
        except Exception as exc:                             # noqa: BLE001 - shown per track, create looks again
            t["swath_error"] = str(exc)[:120]
    return r


def _slug_exists(name: str, route: str = "miaplpy") -> dict | None:
    """What already exists under this name ON THIS ROUTE (a DISP-S1 screening
    has its own <name>_disp_screening folder and config)."""
    from . import areas
    return areas.existing_project(name, route)


def _stem_of(layer: dict) -> str:
    uid = layer.get("upload_id")
    try:
        return _upload_main(uid).stem if uid else ""
    except ValueError:
        return ""


NOT_RECORDED = "source and date not recorded"


def _layer_source(layer: dict) -> str:
    """'Source, vintage' of a mine layer: what was typed, else what the
    upload itself says (file name and date), else that it was drawn. A layer
    reopened from the project (api_project) was not drawn today: with its
    source box cleared it is 'not recorded', never 'drawn on <today>'."""
    typed = str(layer.get("source") or "").strip()
    if typed:
        return typed
    if layer.get("upload_id"):
        try:
            return _upload_source(layer["upload_id"])
        except (ValueError, OSError):
            return "uploaded file"
    if layer.get("reopened"):
        return NOT_RECORDED
    return f"drawn on the project builder map {date.today().isoformat()}"


def api_create(body: dict) -> dict:
    """Write the project: config from the production defaults, outlines into
    <project>/context/, the uploaded originals into context/sources/."""
    from . import areas, runs
    from .config import load_config
    name = (body.get("name") or "").strip()
    if not name:
        raise ValueError("give the project a name")
    layers = body.get("layers") or []
    for L in layers:
        if L.get("role") not in ROLES:
            raise ValueError(f"unknown layer role {L.get('role')!r}")
    area_layers = [L for L in layers if L["role"] == "area"]
    if len(area_layers) != 1:
        raise ValueError("exactly one project area is needed")
    study_key, study_layers = _study_of_body(body, layers)
    stems = tuple(s for s in (_stem_of(L) for L in layers) if s)
    # the project area's row in the report: its own name if one was typed,
    # else the project's (an uploaded area no longer takes the file's name)
    area_name = (area_layers[0].get("name") or "").strip() or name
    area = _area_from_layer(area_layers[0], area_name[:40])
    kinds = {k: [] for k in MINE_KINDS}
    dated_fc = []
    for L in layers:
        if L["role"] == "mine":
            if L.get("reopened") and not L.get("kind"):
                raise ValueError(f"choose the type of the mine layer {L.get('name') or ''!r}: the project saved it "
                                 "without one")
            k = L.get("kind") or "other"
            if k not in kinds:
                raise ValueError(f"unknown mine type {k!r}")
            kinds[k].append(_area_from_layer(L, L.get("name") or "mine"))
            if k == "dated":
                # the year of each polygon drives the mining-year history: kept
                # with its attributes (load_polygon keeps only the outline)
                src = _upload_main(L["upload_id"]) if L.get("upload_id") else L.get("geojson")
                dated_fc += areas.dated_features(src, str(L.get("year_field") or ""),
                                                 str(L.get("name_field") or ""))["features"]
    mines = [a for v in kinds.values() for a in v]
    report = [("Full Satellite Imaging", "full"), (area.label, area)]
    for L in layers:
        if L["role"] == "report" or (L["role"] == "mine" and L.get("report")):
            report.append(((L.get("name") or "").strip()[:40], _area_from_layer(L, L.get("name") or "area")))
    labels = [r[0] for r in report]
    for label in labels:                                     # names the client will read
        problem = areas.client_name_problem(label, stems)
        if problem:
            raise ValueError(f"name in client report: {problem}")
    if len({x.casefold() for x in labels}) != len(labels):
        raise ValueError("two report areas share a name; rename one")
    route = body.get("route") or "miaplpy"
    if route not in areas.ROUTES:
        raise ValueError(f"route must be one of {list(areas.ROUTES)}")
    track, direction = int(body["track"]), str(body["direction"])
    if route == "disp" and not body.get("disp_frame"):
        raise ValueError(f"no DISP-S1 frame on {direction.lower()} track {track} covers this area; choose the "
                         "track a frame is listed for, or use the phase-linked route")

    existing = _slug_exists(name, route)
    if existing and not body.get("overwrite"):
        return {"exists": existing["config"] or existing["project_dir"], **existing}
    sources = {}
    for k in kinds:
        srcs = [_layer_source(L) for L in layers if L["role"] == "mine" and (L.get("kind") or "other") == k]
        if srcs:
            sources[k] = "; ".join(dict.fromkeys(srcs))
    # a study other than the default names itself and its layers; the default
    # study's call is today's, argument for argument
    extra = {"study": study_key, "study_layers": study_layers} if study_key else {}
    out = areas.write_project(
        name, area, route, body.get("start") or "2016-01-01", body.get("end") or date.today().isoformat(),
        track=track, direction=direction, disp_frame=body.get("disp_frame") or "",
        swaths=None,                     # looked up for THIS track over the padded box; refused if it fails
        report_areas=report, mine_sources=sources, file_stems=stems, overwrite=bool(existing),
        deforming=areas.merge(kinds["other"], "other known movement") if kinds["other"] else None,
        underground=areas.merge(kinds["underground"], "underground workings") if kinds["underground"] else None,
        surface=areas.merge(kinds["surface"], "surface mines") if kinds["surface"] else None,
        dated={"type": "FeatureCollection", "features": dated_fc} if dated_fc else None,
        pad_km=float(body.get("pad_km") or 5), reference_lalo=(body.get("reference") or "").strip(), **extra)
    backup = areas.last_backup(out) if existing else None
    cfg = load_config(str(out))                              # paths are $INSAR_PROJECTS-relative in the file
    ctx = Path(cfg.project_dir) / "context"
    for L in layers:                                         # keep what was uploaded, for the record
        if L.get("upload_id"):
            dst = ctx / "sources" / f"{L['role']}_{areas.slug(L.get('name') or 'layer')}"
            shutil.rmtree(dst, ignore_errors=True)
            shutil.copytree(UPLOADS / L["upload_id"], dst)
    saved, empty = [], []
    for L in layers:                                         # what the builder reopens; mines cut to the box
        if L["role"] == "study":
            continue                                         # saved below, one per kind, as the project keeps it
        if L["role"] == "mine" and L.get("kind") == "dated":
            # reopened with its years: the project's own normalised copy
            mp = Path(cfg.mined_year_geojson) if cfg.mined_year_geojson else None
            if mp is None or not mp.exists():
                empty.append(L.get("name") or "dated mined areas")
                continue
            saved.append({"role": "mine", "kind": "dated", "name": L.get("name") or "", "report": bool(L.get("report")),
                          "source": _layer_source(L), "geojson": json.loads(mp.read_text()),
                          "year_field": cfg.mined_year_field, "name_field": cfg.mined_name_field,
                          "fields": {"fields": [cfg.mined_year_field, cfg.mined_name_field]}})
            continue
        a = _area_from_layer(L, L.get("name") or L["role"])
        a = areas.clip_to(a, tuple(cfg.bbox)) if L["role"] == "mine" else a
        if a is None:
            empty.append(L.get("name") or L.get("kind") or "mine layer")
            continue
        saved.append({"role": L["role"], "name": L.get("name") or "", "report": bool(L.get("report")),
                      "kind": L.get("kind") or ("other" if L["role"] == "mine" else ""),
                      "source": _layer_source(L) if L["role"] == "mine" else "", "geojson": a.geojson()})
    kept_study = {}
    for key in (study_layers or {}):
        # the project's own copy (areas._write_study_layers: the features in the
        # box, with their columns), reopened with them
        f = ctx / f"study_{key}.geojson"
        if f.exists():
            fc = json.loads(f.read_text())
            kept_study[key] = len(fc.get("features") or [])
            saved.append({"role": "study", "kind": key, "name": _study_kind(study_key, key).label, "report": False,
                          "source": "", "geojson": fc})
    doc = {"name": name, "reference": body.get("reference") or "", "pad_km": float(body.get("pad_km") or 5),
           "layers": saved}
    if study_key:
        doc["study"] = study_key
    (ctx / "builder_layers.json").write_text(json.dumps(doc))
    runs._remember(out)
    kept = cfg.deramp_exclude_geojson
    n_mine = len(json.loads(Path(kept).read_text())["features"]) if kept else 0
    res = {"config": str(out), "project_dir": str(cfg.project_dir), "backup": str(backup) if backup else None,
            "report_areas": labels, "mine_polygons_kept": n_mine, "mine_layers_outside_box": empty,
            "mine_polygons_uploaded": sum(len(m.rings) for m in mines), "swaths": cfg.stack_swaths.split()
            if route != "disp" else None, "mine_layer_sources": list(cfg.mine_layer_sources),
            "bbox": list(cfg.bbox), "route": areas.ROUTE_NAMES.get(route, route), "title": cfg.project_title,
            "dated_polygons": len(json.loads(Path(cfg.mined_year_geojson).read_text())["features"])
            if cfg.mined_year_geojson else 0,
            "far_field": areas.far_field_check(areas.box(cfg.bbox[1], cfg.bbox[0], cfg.bbox[3], cfg.bbox[2]),
                                               areas.load_polygon(kept, "mines") if kept else None,
                                               report, float(cfg.acceptance_far_field_buffer_m))}
    if study_key:
        from . import domains
        st = domains.study(study_key)
        res["study"] = st.label
        res["study_layers"] = [{"kind": k, "label": _study_kind(study_key, k).label, "features": n}
                               for k, n in kept_study.items()]
    return res


def _study_of_body(body: dict, layers: list) -> tuple[str | None, dict]:
    """(study key, {layer kind: FeatureCollection}) of a create request; (None,
    {}) for the default study, whose request and write_project call are
    today's. Only the default study and BUILDER_STUDIES are taken; a study's
    layers are its own layer kinds (an upload is read again in full, with its
    attribute columns), and mine layers belong to the default study."""
    from . import domains
    key = str(body.get("study") or domains.DEFAULT_STUDY).strip().lower()
    try:
        st = domains.study(key)
    except KeyError as exc:
        raise ValueError(str(exc).strip("'\"")) from None
    study_rows = [L for L in layers if L["role"] == "study"]
    if st.key == domains.DEFAULT_STUDY:
        if study_rows:
            raise ValueError(f"the {st.label} study takes mine layers, not {study_rows[0].get('kind')!r}: choose the "
                             "study the layer belongs to in step 0, or remove it")
        return None, {}
    if st.key not in BUILDER_STUDIES or st.builder_status != "offered":
        raise ValueError(f"the builder does not offer the {st.label} study yet "
                         f"({st.builder_reason if st.builder_status != 'offered' else NOT_IN_BUILDER})")
    mine = next((L for L in layers if L["role"] == "mine"), None)
    if mine is not None:
        raise ValueError(f"mine layers belong to the mining subsidence study, not {st.label}: remove "
                         f"{mine.get('name') or 'the mine layer'!r} or choose that study in step 0")
    out = {}
    for L in study_rows:
        LK = _study_kind(st.key, str(L.get("kind") or ""))
        if L.get("upload_id"):
            fc = study_features(_upload_main(L["upload_id"]), LK.geometry, LK.label)
        else:
            fc = L.get("geojson")
            if not (isinstance(fc, dict) and fc.get("type") == "FeatureCollection"):
                raise ValueError(f"{LK.label}: no geometry")
        out.setdefault(LK.key, {"type": "FeatureCollection", "features": []})["features"] += list(fc["features"])
    return st.key, out


def api_projects(_: dict) -> dict:
    """Existing projects and their outlines, to view or rework on the map."""
    import yaml
    from . import runs
    out = []
    for c in runs.known_configs():
        try:
            d = yaml.safe_load(Path(c).read_text()) or {}
        except Exception:                                    # noqa: BLE001
            continue
        import os
        out.append({"config": str(c), "name": Path(c).stem.replace("config_", ""),
                    "project_dir": os.path.expandvars(str(d.get("project_dir", ""))), "bbox": d.get("bbox")})
    return {"projects": out}


def api_project(body: dict) -> dict:
    """One project's outlines: what the builder saved, else rebuilt from the
    config (projects made before the builder existed)."""
    import yaml
    from . import areas, client_report
    from .config import load_config
    cfg = load_config(body["config"])
    # the project's own record window: the page otherwise keeps its default
    # dates, and Update would rewrite start_date/end_date with them
    window = {"start": cfg.start_date, "end": cfg.end_date}
    saved = Path(cfg.project_dir) / "context" / "builder_layers.json"
    if saved.exists():
        d = json.loads(saved.read_text())
        return {**d, "name": d.get("name") or Path(body["config"]).stem, "bbox": list(cfg.bbox), **window}
    raw = yaml.safe_load(Path(body["config"]).read_text())
    from .report_insights import layer_sources
    recorded = layer_sources(cfg)                            # {kind: "source, vintage"} the config records
    layers, aoi = [], None
    for pz in (Path(cfg.project_dir) / "context" / "area_of_interest.geojson",
               Path(cfg.project_dir) / "context" / "project_zone.geojson"):
        if pz.exists():
            aoi = areas.load_polygon(pz, "project area")
            layers.append({"role": "area", "name": aoi.label, "geojson": aoi.geojson()})
            break
    # each mine layer with the source the config gives it ("reopened": with
    # the source cleared it is 'not recorded', not 'drawn on the map today')
    typed = [(k, f) for k, f in (("underground", cfg.underground_mines_geojson), ("surface", cfg.surface_mines_geojson),
                                 ("other", cfg.other_movement_geojson))
             if f and Path(f).exists()]
    for k, f in typed:
        a = areas.load_polygon(f, "other known movement" if k == "other" else f"{k} mines")
        layers.append({"role": "mine", "kind": k, "name": a.label, "source": recorded.get(k) or NOT_RECORDED,
                       "reopened": True, "geojson": a.geojson()})
    if cfg.mined_year_geojson and Path(cfg.mined_year_geojson).exists():
        layers.append({"role": "mine", "kind": "dated", "name": "dated mined areas",
                       "source": recorded.get("dated") or NOT_RECORDED, "reopened": True,
                       "geojson": json.loads(Path(cfg.mined_year_geojson).read_text()),
                       "year_field": cfg.mined_year_field, "name_field": cfg.mined_name_field,
                       "fields": {"fields": [f for f in (cfg.mined_year_field, cfg.mined_name_field) if f]}})
    if not typed and cfg.deramp_exclude_geojson and Path(cfg.deramp_exclude_geojson).exists():
        # one untyped layer (a project from before mine types): reopened with
        # NO type, so Update waits until one is chosen. Reopened as 'other',
        # an Update made Site B's mapped mines "ground already known to
        # move" in the report, credited as drawn on the builder map that day
        a = areas.load_polygon(cfg.deramp_exclude_geojson, "mapped mines")
        layers.append({"role": "mine", "kind": None, "name": a.label, "source": recorded.get("mapped") or NOT_RECORDED,
                       "reopened": True, "geojson": a.geojson()})
    for label, spec in client_report.parse_areas(cfg):
        if spec.lower() == "full":
            continue
        try:
            if spec.lower().endswith((".geojson", ".json", ".kml", ".kmz", ".shp", ".zip", ".gpkg")):
                a = areas.load_polygon(spec, label)
            else:
                s, w, n, e = (float(v) for v in spec.split(","))
                a = areas.box(min(s, n), min(w, e), max(s, n), max(w, e), label)
        except Exception:                                    # noqa: BLE001
            continue
        if aoi is not None and len(a.rings) == len(aoi.rings) and \
                all(abs(x - y) < 1e-5 for x, y in zip(a.bbox, aoi.bbox)):
            layers[0]["name"] = label                        # the project area's own report entry
            continue
        layers.append({"role": "report", "name": label, "geojson": a.geojson()})
    if not any(L["role"] == "area" for L in layers):
        w, s, e, n = cfg.bbox
        layers.insert(0, {"role": "area", "name": "processing box", "geojson": areas.box(s, w, n, e).geojson()})
    extra = _study_reopened(cfg, layers)
    return {"name": Path(body["config"]).stem.replace("config_", ""), "reference": raw.get("reference_lalo") or "",
            "layers": layers, "bbox": list(cfg.bbox), **window, **extra}


def _study_reopened(cfg, layers: list) -> dict:
    """A project of a study other than the default, written without the
    builder: its study layers from the config fields, and {"study": key}.
    {} for the default study (today's reopening)."""
    from . import domains
    try:
        st = domains.study_of(cfg)
    except KeyError:
        return {}
    if st.key == domains.DEFAULT_STUDY:
        return {}
    for L in st.layer_kinds:
        if L.config_field == "feature_layers":
            files = [p for k, p in (domains.split_feature_layer(x) for x in getattr(cfg, "feature_layers", ()) or ())
                     if k == L.key]
        else:
            files = [getattr(cfg, L.config_field, "") or ""]
        for f in files:
            if f and Path(f).exists():
                layers.append({"role": "study", "kind": L.key, "name": L.label, "reopened": True,
                               "geojson": json.loads(Path(f).read_text())})
    return {"study": st.key}


def api_info(_: dict) -> dict:
    from . import areas, settings
    from .config import GNSS_TIE_RADIUS_KM
    from datetime import timedelta
    from .stack_backend import ERA5_LAG_DAYS
    # the default end date leaves ERA5's ~5-day publication lag behind, so the
    # run can correct every date (a record ending today stops at the
    # troposphere step naming the last dates)
    return {"today": date.today().isoformat(),
            "default_end": (date.today() - timedelta(days=ERA5_LAG_DAYS + 1)).isoformat(), "routes": areas.ROUTES, "configs_dir": str(settings.configs_dir()),
            "projects_root": str(settings.projects_root()), "gnss_radius_km": GNSS_TIE_RADIUS_KM}


def api_exists(body: dict) -> dict:
    """Whether the name is taken on the chosen route, where its folder would
    be, and whether the name itself is fit for the client report."""
    from . import areas
    name, route = (body.get("name") or "").strip(), body.get("route") or "miaplpy"
    if route not in areas.ROUTES:
        route = "miaplpy"
    have = _slug_exists(name, route) if name else None
    key, P, out = areas.project_paths(name or "x", route)
    return {"exists": (have["config"] or have["project_dir"]) if have else None, "holds": (have or {}).get("holds", []),
            "backend": (have or {}).get("backend"), "folder": str(P) if name else None,
            "name_problem": areas.client_name_problem(name) if name else None}


def api_geocode(body: dict) -> dict:
    """Place search through this server, so OpenStreetMap is told who asks
    (areas.user_agent: the firm and a contact) instead of an anonymous
    browser on localhost. The text still goes to OpenStreetMap."""
    from . import areas
    q = (body.get("q") or "").strip()
    if not q:
        raise ValueError("nothing to search for")
    hits = areas.geocode(q, limit=1)
    return {"hits": [{"name": h["name"], "lat": h["lat"], "lon": h["lon"], "bbox": h["area"].bbox} for h in hits]}


API = {"/api/outline": api_outline, "/api/check": api_check, "/api/create": api_create, "/api/studies": api_studies,
       "/api/projects": api_projects, "/api/project": api_project, "/api/info": api_info,
       "/api/exists": api_exists, "/api/geocode": api_geocode}


# ---------------------------------------------------------------- server
# A secret per server session (16 Oct 2026, readiness review NET-06). The
# Host and Origin checks stop other web pages, but a request that sends no
# Origin - any program on this machine, or a Windows one through WSL's
# localhost forwarding - could call /api/create and write a project folder
# and config. The server now writes a token into the page it serves and
# takes an /api/ call only when it carries that token (constant-time
# compare); a page left open across a server restart is told to reload. The
# results viewer (results_viewer.py) uses the same helpers.
TOKEN_HEADER = "X-Insar-Token"


def page_with_token(page: Path, token: str) -> bytes:
    """The page as served: the session's token in <meta name="insar-token">
    right after <head>, where the page's api() reads it."""
    html = page.read_text(encoding="utf-8")
    if "<head>" not in html:
        raise RuntimeError(f"{page} has no <head> to carry the session token")
    return html.replace("<head>", f'<head>\n<meta name="insar-token" content="{token}">', 1).encode("utf-8")


def token_ok(headers, token: str) -> bool:
    """The request carries this session's token (constant-time compare)."""
    got = headers.get(TOKEN_HEADER) or ""
    return bool(token) and hmac.compare_digest(got.encode("utf-8", "replace"), token.encode("utf-8"))


def send_page(handler, page: str, headers: dict | None = None) -> None:
    """Serve `page` from the handler's folder with its session token; never
    cached, so a reload after a restart gets the new token. `headers`: more
    response headers (the results viewer's Content-Security-Policy)."""
    data = page_with_token(Path(handler.directory) / page, handler.token)
    handler.send_response(200)
    handler.send_header("Content-Type", "text/html; charset=utf-8")
    handler.send_header("Content-Length", str(len(data)))
    handler.send_header("Cache-Control", "no-store")
    for k, v in (headers or {}).items():
        handler.send_header(k, v)
    handler.end_headers()
    handler.wfile.write(data)


STALE_PAGE = "this page is from an earlier session of the server, or not from it: reload the page (F5)"


class _Handler(http.server.SimpleHTTPRequestHandler):
    port = PORT
    token = ""                                      # set per server session by serve()

    def log_message(self, *a):                      # keep the terminal / dashboard clean
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

    def do_GET(self):
        path = self.path.split("?")[0].split("#")[0]
        if path in ("/", ""):
            self.send_response(302)
            self.send_header("Location", "/" + PAGE)
            self.end_headers()
            return
        f = ASSETS.get(path.lstrip("/"))            # the vendored map library (NET-04): exact names only
        if f is not None:
            try:
                data = f.read_bytes()
            except OSError:
                return self._send(404, {"error": "not served"})
            self.send_response(200)
            self.send_header("Content-Type", CONTENT_TYPES.get(f.suffix, "application/octet-stream"))
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(data)
            return
        if path != "/" + PAGE:                      # the page only: never the uploads beside it
            return self._send(404, {"error": "not served"})
        if not self._local():                       # the token goes only to this machine's own page
            return self._send(403, {"error": "only this machine's browser may open the builder"})
        return send_page(self, PAGE)

    def do_POST(self):
        fn = API.get(self.path.split("?")[0])
        if fn is None:
            return self._send(404, {"error": "no such endpoint"})
        if not self._local():
            return self._send(403, {"error": "only the builder page on this machine may call this"})
        if not token_ok(self.headers, self.token):  # no Origin is no excuse: the token decides
            return self._send(403, {"error": STALE_PAGE})
        try:
            n = int(self.headers.get("Content-Length") or 0)
            if n > MAX_UPLOAD * 4 // 3 + 65536:
                raise ValueError("request too large")
            body = json.loads(self.rfile.read(n) or b"{}")
            self._send(200, fn(body))
        except Exception as exc:                             # noqa: BLE001
            self._send(400, {"error": str(exc) or type(exc).__name__})


def _port_busy(port: int) -> bool:
    with socket.socket() as s:
        return s.connect_ex(("127.0.0.1", port)) == 0


def _prune_uploads(days: float = 2.0) -> None:
    if UPLOADS.is_dir():
        cutoff = time.time() - days * 86400
        for d in UPLOADS.iterdir():
            if d.is_dir() and d.stat().st_mtime < cutoff:
                shutil.rmtree(d, ignore_errors=True)


def serve(port: int = PORT) -> str:
    """Start the server in a daemon thread (once) and return the page URL."""
    SERVE_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(PKG / PAGE, SERVE_DIR / PAGE)      # always the current builder
    for old in ("insar_config_builder.html",):      # the retired settings form
        (SERVE_DIR / old).unlink(missing_ok=True)
    _prune_uploads()
    url = f"http://localhost:{port}/{PAGE}"
    if "httpd" in _SERVER or _port_busy(port):
        return url                                  # ours, or one started earlier
    handler = type("Handler", (_Handler,), {"port": port, "token": secrets.token_urlsafe(32)})
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", port), partial(handler, directory=str(SERVE_DIR)))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    _SERVER["httpd"] = httpd
    return url


def open_in_browser(url: str) -> None:
    """The Windows default browser from WSL, else xdg-open."""
    if shutil.which("cmd.exe"):
        subprocess.Popen(["cmd.exe", "/c", "start", "", url], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    elif shutil.which("xdg-open"):
        subprocess.Popen(["xdg-open", url], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def main() -> None:
    url = serve()
    open_in_browser(url)
    print(f"Project builder at {url}  (Ctrl-C to stop the server)")
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        print()
        sys.exit(0)


if __name__ == "__main__":
    main()
