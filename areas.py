"""Turning "an area" into a project: three ways in, one outline out.

  coordinates   "lat,lon" (+ a radius), "lat1,lon1,lat2,lon2" (two corners -
                the same order the report areas use), WKT POLYGON, or GeoJSON
  place name    OpenStreetMap Nominatim search -> bounding box
  polygon       a GeoJSON / KML / KMZ / shapefile drawn elsewhere (the
                project builder, Google Earth, a GIS)

Then: what the archive holds over it (Sentinel-1 passes by track and year,
OPERA DISP-S1 frames), which GNSS stations are near, and a config written
from the production defaults. Network calls live here, not in the dashboard,
so they can be tested and reused.
"""
from __future__ import annotations

import json
import math
import re
import urllib.parse
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

PKG = Path(__file__).resolve().parent


def user_agent() -> str:
    """What Nominatim is told about who is asking. Its usage policy asks for a
    User-Agent that identifies the application and a way to reach its operator
    before it blocks; the old string called this "a research pipeline" and
    gave no contact. The firm and contact come from the settings file
    (insar-doctor --firm ... --contact ...); nothing here is hard-coded."""
    try:
        from .version import __version__ as ver
    except Exception:                                        # noqa: BLE001 - never block a lookup
        ver = "1"
    try:
        from .settings import identity
        who = identity()
    except Exception:                                        # noqa: BLE001
        who = {}
    firm, contact = who.get("firm") or "", who.get("contact") or ""
    tail = "; ".join(x for x in (firm, contact) if x) or "operator contact not set: insar-doctor --contact"
    return f"insar-pipeline/{ver} (ground-movement mapping for engineering projects; {tail})"


USER_AGENT = user_agent()          # kept for callers that import the constant (report_insights)


@dataclass
class Area:
    """An outline in lon/lat. bbox is (min_lon, min_lat, max_lon, max_lat),
    the pipeline's own order."""
    label: str
    rings: list = field(default_factory=list)       # [[(lon, lat), ...], ...] outer rings
    source: str = ""
    # holes[i] = the interior rings of rings[i]. Until 5 Oct 2026 only outer
    # rings were kept, so every hole came back filled: unmined islands and
    # reclaimed ground inside a mine outline became "mine" and left the
    # farmland comparison (+0.9 to +3.5% coherent cells on the Site A mine
    # layers, which carry 9-107 holes each) - the bug _points_in_polygons
    # had already fixed for the files read directly.
    holes: list = field(default_factory=list)

    def holes_of(self, i: int) -> list:
        return self.holes[i] if i < len(self.holes) else []

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        xs = [p[0] for r in self.rings for p in r]
        ys = [p[1] for r in self.rings for p in r]
        return (min(xs), min(ys), max(xs), max(ys))

    @property
    def is_box(self) -> bool:
        if len(self.rings) != 1:
            return False
        w, s, e, n = self.bbox
        corners = {(w, s), (e, s), (e, n), (w, n)}
        return {tuple(p) for p in self.rings[0]} == corners

    def size_km(self) -> tuple[float, float]:
        w, s, e, n = self.bbox
        lat = math.radians(0.5 * (s + n))
        return ((e - w) * 111.32 * math.cos(lat), (n - s) * 110.95)

    def wkt(self) -> str:
        w, s, e, n = self.bbox
        return f"POLYGON(({w} {s},{e} {s},{e} {n},{w} {n},{w} {s}))"

    def geojson(self) -> dict:
        feats = [{"type": "Feature", "properties": {"name": self.label},
                  "geometry": {"type": "Polygon", "coordinates": [[list(map(float, p)) for p in ring]
                                                                  for ring in (r, *self.holes_of(i))]}}
                 for i, r in enumerate(self.rings)]
        return {"type": "FeatureCollection", "features": feats}

    def padded(self, km: float) -> "Area":
        """The bounding box grown by km on every side (the processing box
        needs far field around the area of interest)."""
        w, s, e, n = self.bbox
        dlat = km / 110.95
        dlon = km / (111.32 * math.cos(math.radians(0.5 * (s + n))))
        return box(s - dlat, w - dlon, n + dlat, e + dlon, f"{self.label} + {km:g} km")


def box(s: float, w: float, n: float, e: float, label: str = "box") -> Area:
    s, n, w, e = min(s, n), max(s, n), min(w, e), max(w, e)
    return Area(label, [[(w, s), (e, s), (e, n), (w, n), (w, s)]], "coordinates")


def _check_latlon(lat: float, lon: float) -> None:
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        raise ValueError(f"{lat}, {lon} is not a latitude, longitude pair")


def parse_coordinates(text: str, radius_km: float = 2.0, label: str = "") -> Area:
    """Coordinates as typed or pasted. Latitude first, as the report areas
    and every map app's "copy coordinates" give them."""
    t = text.strip()
    if not t:
        raise ValueError("nothing entered")
    if t.startswith("{"):
        return _from_geojson(_json_text(t, "the pasted text"), label or "pasted GeoJSON")
    if t.upper().startswith(("POLYGON", "MULTIPOLYGON")):
        # shapely reads holes ("POLYGON((outer),(hole))"); the regex fallback
        # only finds the rings that open with "((" - the outer ones
        try:
            from shapely import wkt as _wkt
            from shapely.geometry import mapping
        except ImportError:
            rings = []
            for grp in re.findall(r"\(\(([^()]+)\)", t):
                rings.append([tuple(float(v) for v in pt.split()[:2]) for pt in grp.split(",")])
            if not rings:
                raise ValueError("could not read the WKT polygon")
            _check_lonlat(rings, label or "pasted WKT")
            return Area(label or "pasted WKT", rings, "coordinates")
        try:
            geom = _wkt.loads(t)
        except Exception as exc:                             # noqa: BLE001 - shapely's own error types vary
            raise ValueError(f"could not read the WKT polygon ({exc})") from None
        a = _from_geojson(mapping(geom), label or "pasted WKT")
        a.source = "coordinates"
        return a
    nums = [float(v) for v in re.findall(r"-?\d+(?:\.\d+)?", t)]
    if len(nums) == 2:
        lat, lon = nums
        _check_latlon(lat, lon)
        a = box(lat, lon, lat, lon, label or f"Site {lat:.4f} {lon:.4f}")
        return Area(a.label, a.padded(radius_km).rings, "coordinates")
    if len(nums) == 4:
        s, w, n, e = nums
        _check_latlon(s, w); _check_latlon(n, e)
        return box(s, w, n, e, label or "box")
    raise ValueError("enter 'lat, lon', two corners 'lat1, lon1, lat2, lon2', a WKT POLYGON or GeoJSON")


def _check_lonlat(rings: list, label: str) -> None:
    """Every vertex must be longitude/latitude. A GeoJSON exported in UTM or
    State Plane used to load without complaint; clip_to then dropped every
    ring and the mine layer vanished with only "kept 0" to show for it."""
    for r in rings:
        for x, y in r:
            if not (-180 <= x <= 180 and -90 <= y <= 90):
                raise ValueError(f"{label}: the coordinates ({x:.6g}, {y:.6g}) are not longitude/latitude - the file "
                                 "is in a projected system (UTM, State Plane). Export it as WGS84 / EPSG:4326, or "
                                 "upload the shapefile with its .prj and it is reprojected here")


def _from_geojson(gj: dict, label: str) -> Area:
    """The polygons of a GeoJSON object. A file that is JSON but not GeoJSON
    used to stop on a KeyError or TypeError; it is a ValueError in words now,
    and the feature and vertex counts are capped (16 Oct 2026, readiness
    review SEC-03 / SEC-04)."""
    try:
        return _geojson_area(gj, label)
    except (KeyError, TypeError, AttributeError, IndexError) as exc:
        raise ValueError(f"{label or 'the outline'} is not a valid GeoJSON outline: {type(exc).__name__} "
                         f"{str(exc)[:120]}") from None


def _geojson_area(gj: dict, label: str) -> Area:
    feats = gj["features"] if gj.get("type") == "FeatureCollection" else \
        [gj] if gj.get("type") == "Feature" else [{"geometry": gj, "properties": {}}]
    who = label or "the outline"
    _check_features(len(feats), who)
    rings, holes = [], []
    count = [0]

    def ring(r):
        count[0] += len(r)
        _check_vertices(count[0], who)
        try:
            return [(float(p[0]), float(p[1])) for p in r]
        except (TypeError, ValueError, IndexError) as exc:
            raise ValueError(f"{who} is not a valid GeoJSON outline: a position is not a longitude, latitude "
                             f"pair ({str(exc)[:80]})") from None

    def polygons(g):
        t = (g or {}).get("type")
        if t == "Polygon":
            return [g["coordinates"]]
        if t == "MultiPolygon":
            return list(g["coordinates"])
        if t == "GeometryCollection":
            return [p for sub in g.get("geometries") or () for p in polygons(sub)]
        return []
    for f in feats:
        for poly in polygons(f.get("geometry")):
            if not poly or len(poly[0]) < 3:
                continue
            # ring 0 is the exterior, every further ring a hole (RFC 7946 3.1.6)
            rings.append(ring(poly[0]))
            holes.append([ring(h) for h in poly[1:] if len(h) >= 3])
    if not rings:
        raise ValueError("no polygons in that GeoJSON")
    name = next((f.get("properties", {}).get("name") for f in feats if (f.get("properties") or {}).get("name")), None)
    _check_lonlat(rings, label or name or "polygon")
    return Area(label or name or "polygon", rings, "polygon", holes)


def _from_kml(text: str, label: str, name: str = "") -> Area:
    """Every Polygon's outerBoundaryIs and its innerBoundaryIs holes (KML is
    lon,lat[,alt] by definition; checked anyway). `name` is the file's name
    for the messages (default: the label)."""
    name = name or label or "that file"
    root = _parse_kml(text, name)
    rings, holes = [], []
    count = [0, 0]                                   # polygons, vertices

    def coords(el):
        out = []
        for c in el.iter():
            if c.tag.endswith("coordinates") and c.text:
                toks = c.text.split()
                count[1] += len(toks)
                _check_vertices(count[1], name)
                try:
                    pts = [(float(xy[0]), float(xy[1])) for xy in (tok.split(",") for tok in toks)]
                except (ValueError, IndexError) as exc:
                    raise ValueError(f"{name} is not a valid KML file: a coordinate is not a longitude,latitude "
                                     f"pair ({str(exc)[:80]})") from None
                if len(pts) >= 3:
                    out.append(pts)
        return out
    for poly in root.iter():
        if not (poly.tag == "Polygon" or poly.tag.endswith("}Polygon")):
            continue
        count[0] += 1
        _check_features(count[0], name, "polygons")
        outer, inner = [], []
        for b in poly:
            if b.tag.endswith("outerBoundaryIs"):
                outer += coords(b)
            elif b.tag.endswith("innerBoundaryIs"):
                inner += coords(b)           # one LinearRing each, or several in one element
        if outer:
            rings.append(outer[0])
            holes.append(inner)
    if not rings:
        raise ValueError("no polygons in that KML")
    _check_lonlat(rings, label)
    return Area(label, rings, "polygon", holes)


def normalize_path(path: str | Path) -> Path:
    r"""A path as a person pastes it, made usable from Linux/WSL:
      "\\wsl.localhost\Ubuntu\home\me\x.geojson" or //wsl.localhost/Ubuntu/...  -> /home/me/x.geojson
      \\wsl$\Ubuntu\...                                                          -> the same
      C:\Users\me\Downloads\x.kml                                                -> /mnt/c/Users/me/Downloads/x.kml
    plus surrounding quotes (Explorer's "Copy as path") and ~."""
    s = str(path).strip().strip('"').strip("'").strip()
    t = s.replace("\\", "/")
    parts = [x for x in t.split("/") if x]
    if t.startswith("//") and parts and parts[0].lower() in ("wsl.localhost", "wsl$"):
        return Path("/" + "/".join(parts[2:]))            # drop the host and the distro name
    m = re.match(r"^([A-Za-z]):[\\/](.*)$", s)
    if m:
        return Path(f"/mnt/{m.group(1).lower()}/" + m.group(2).replace("\\", "/"))
    return Path(s).expanduser()


GIS_SUFFIXES = (".shp", ".gpkg", ".gdb", ".fgb")
POLYGON_SUFFIXES = (".geojson", ".json", ".kml", ".kmz", ".zip") + GIS_SUFFIXES


# ---------------------------------------------------------------- outline files from outside the operating organisation
# Outline files come from clients, operators and state surveys, uploaded in
# the builder or typed as a path in the dashboard. Until 16 Oct 2026 they were
# typed by extension only, a malformed one stopped on a raw decode error, and
# nothing bounded what a file could unpack to: a 150 KB KMZ inflating to
# 150 MB was read whole (482 MB peak). Readiness review SEC-03 / SEC-04. The
# limits are generous for real mine layers: the public layers in context/
# are 2 KB - 2.1 MB with 1-53 features and up to 52,000 vertices, and a
# statewide mine layer is ~100 MB (map_builder.MAX_UPLOAD's note), about 3
# million vertices as GeoJSON.
MAX_FILE_BYTES = 300 * 1024 * 1024      # = map_builder.MAX_UPLOAD; a typed path had no limit
MAX_ZIP_MEMBERS = 1000                  # a zipped shapefile is 4-8 files, a KMZ one KML and its icons
MAX_UNZIPPED_BYTES = 1024 ** 3          # all members, from the zip directory before anything is read
                                        # (a shapefile's .dbf pads every text field, so it is larger
                                        # than its GeoJSON and can deflate 100x and more)
MAX_TEXT_RATIO = 100                    # a KML/GeoJSON member over 1 MB: real ones deflate 3-10x
                                        # (context/: 2.6-3.8x compact, 10x indented); a bomb 1000x
MAX_FEATURES = 200_000                  # thousands in a statewide layer; 4,000x the largest one here
MAX_VERTICES = 10_000_000               # ~3x a 100 MB statewide layer; ~1-2 GB of memory to read
_ZIP_MAGIC = (b"PK\x03\x04", b"PK\x05\x06")   # a zip's first member, or an empty zip
_KINDS = {".geojson": "GeoJSON", ".json": "GeoJSON", ".kml": "KML", ".kmz": "KMZ", ".zip": "zip"}


def _check_features(n: int, name: str, what: str = "features") -> None:
    if n > MAX_FEATURES:
        raise ValueError(f"{name}: more than {MAX_FEATURES:,} {what} - more than an outline layer here may hold. "
                         "Clip the layer to the region of the project first (in a GIS), then load it again")


def _check_vertices(n: int, name: str) -> None:
    if n > MAX_VERTICES:
        raise ValueError(f"{name}: more than {MAX_VERTICES:,} vertices - more than an outline layer here may "
                         "hold. Clip the layer to the region of the project, or simplify it, first (in a GIS)")


def _content_kind(head: bytes) -> str:
    """What a file's first bytes look like, in words."""
    text = head.lstrip(b"\xef\xbb\xbf \t\r\n")
    if head.startswith(_ZIP_MAGIC):
        return "a zip archive (a KMZ or a zipped shapefile)"
    if text.startswith(b"{"):
        return "JSON text"
    if text.startswith(b"["):
        return "a JSON list (GeoJSON is one object, in braces)"
    if text.startswith(b"<"):
        return "XML text (KML)"
    if head.startswith(b"SQLite format 3"):
        return "a GeoPackage (SQLite) database"
    if head.startswith(b"%PDF"):
        return "a PDF document"
    return "not text or an archive this reads"


def _check_content(p: Path) -> None:
    """The content must be what the extension says: a zip for .zip/.kmz, a
    JSON object for .geojson/.json, XML for .kml. Other extensions go to
    GDAL, which checks its own formats."""
    kind = _KINDS.get(p.suffix.lower())
    if kind is None:
        return
    with open(p, "rb") as fh:
        head = fh.read(4096)
    text = head.lstrip(b"\xef\xbb\xbf \t\r\n")
    ok = (head.startswith(_ZIP_MAGIC) if kind in ("KMZ", "zip") else
          text.startswith(b"{") if kind == "GeoJSON" else text.startswith(b"<"))
    if not ok:
        raise ValueError(f"{p.name} is not a valid {kind} file: its content is {_content_kind(head)}, not what "
                         f"a {p.suffix} file holds. Check that it is the file meant, saved in that format")


def _check_size(p: Path) -> None:
    """A file (with a shapefile's companions, or a folder such as .gdb)
    larger than the builder's upload limit is refused before it is read."""
    if p.is_dir():
        files = [f for f in p.rglob("*") if f.is_file()]
    elif p.suffix.lower() == ".shp":
        files = [f for f in p.parent.iterdir() if f.stem == p.stem and f.is_file()]
    else:
        files = [p]
    total = sum(f.stat().st_size for f in files)
    if total > MAX_FILE_BYTES:
        raise ValueError(f"{p.name} is {total / 2 ** 20:,.0f} MB; outline files up to {MAX_FILE_BYTES >> 20} MB "
                         "are read (the builder's upload limit). Clip the layer to the region of the project first")


def _text(data: bytes, name: str, kind: str) -> str:
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{name} is not a valid {kind} file: it is not UTF-8 text ({exc.reason} at byte "
                         f"{exc.start:,})") from None


def _json_text(text: str, name: str) -> dict:
    try:
        doc = json.loads(text)
    except (ValueError, RecursionError) as exc:
        raise ValueError(f"{name} is not a valid GeoJSON file: {str(exc)[:160]}") from None
    if not isinstance(doc, dict):
        raise ValueError(f"{name} is not a valid GeoJSON file: it holds a JSON {type(doc).__name__}, "
                         "not a GeoJSON object")
    return doc


def _read_json_file(p: Path) -> dict:
    return _json_text(_text(p.read_bytes(), p.name, "GeoJSON"), p.name)


class _RootReached(Exception):
    pass


def _parse_kml(text: str, name: str):
    """The KML's element tree. A DOCTYPE or ENTITY declaration is refused
    first: KML never needs one, and entities are how XML expands to
    gigabytes or reads other files (defusedxml is used when installed; it is
    not in the environment today). Parse errors are a ValueError in words."""
    from xml.parsers import expat

    def refuse(*_):
        raise ValueError(f"{name} is not a valid KML file: it declares a DOCTYPE or ENTITY, which KML never "
                         "needs (refused: such declarations can expand to gigabytes or read other files)")

    def root(*_):
        raise _RootReached
    prolog = expat.ParserCreate()
    prolog.StartDoctypeDeclHandler = refuse
    prolog.EntityDeclHandler = refuse
    prolog.StartElementHandler = root               # the prolog is over: nothing more to look for
    try:
        for i in range(0, len(text), 1 << 16):
            prolog.Parse(text[i:i + (1 << 16)], False)
        prolog.Parse("", True)
    except _RootReached:
        pass
    except expat.ExpatError as exc:
        raise ValueError(f"{name} is not a valid KML file: {exc}") from None
    try:
        from defusedxml.ElementTree import fromstring
    except ImportError:
        from xml.etree.ElementTree import fromstring
    try:
        return fromstring(text)
    except Exception as exc:                         # noqa: BLE001 - ParseError, or defusedxml's own refusals
        raise ValueError(f"{name} is not a valid KML file: {str(exc)[:160]}") from None


def _open_zip(p: Path) -> zipfile.ZipFile:
    """The zip, its member count and unpacked size checked from its directory
    before any member is read."""
    try:
        z = zipfile.ZipFile(p)
    except (zipfile.BadZipFile, OSError, ValueError) as exc:
        raise ValueError(f"{p.name} is not a valid {_KINDS.get(p.suffix.lower(), 'zip')} file: {exc}") from None
    infos = z.infolist()
    total = sum(i.file_size for i in infos)
    if len(infos) > MAX_ZIP_MEMBERS or total > MAX_UNZIPPED_BYTES:
        z.close()
        raise ValueError(f"{p.name} holds {len(infos):,} files unpacking to {total / 2 ** 20:,.0f} MB; up to "
                         f"{MAX_ZIP_MEMBERS:,} files and {MAX_UNZIPPED_BYTES >> 20:,} MB are read. Zip only the "
                         "outline layer, clipped to the region of the project")
    return z


def _read_member(z: zipfile.ZipFile, member: str, archive: str, kind: str) -> str:
    """One KML/GeoJSON member as text: its size and compression ratio checked
    from the directory first, and the read itself bounded (a directory can
    lie about the size)."""
    info = z.getinfo(member)
    where = f"{member} in {archive}"
    ratio = info.file_size / max(1, info.compress_size)
    if info.file_size > MAX_FILE_BYTES or (info.file_size > 2 ** 20 and ratio > MAX_TEXT_RATIO):
        raise ValueError(f"{where} unpacks to {info.file_size / 2 ** 20:,.0f} MB from "
                         f"{info.compress_size / 2 ** 20:,.2f} MB ({ratio:,.0f}x): a real {kind} file deflates "
                         f"about 3-10x, and up to {MAX_FILE_BYTES >> 20} MB is read - refused")
    try:
        with z.open(info) as fh:
            data = fh.read(MAX_FILE_BYTES + 1)
    except Exception as exc:                         # noqa: BLE001 - BadZipFile, zlib.error, EOFError, OSError
        raise ValueError(f"{archive} is not a valid zip file: {member}: {str(exc)[:160]}") from None
    if len(data) > MAX_FILE_BYTES:
        raise ValueError(f"{where} unpacks to more than {MAX_FILE_BYTES >> 20} MB - refused")
    return _text(data, where, kind)


def _open_outline(p: Path) -> None:
    """What every outline file passes before it is read: it exists, its size,
    and its content against its extension."""
    if not p.exists():
        raise FileNotFoundError(p)
    _check_size(p)
    if p.is_file():
        _check_content(p)


def _from_gis(path: str, label: str) -> Area:
    """Anything GDAL reads (shapefile, GeoPackage, zipped shapefile), in any
    projection, reprojected to lon/lat."""
    import geopandas as gpd
    df = gpd.read_file(path)
    if df.crs is None:
        x0, y0, x1, y1 = df.total_bounds
        if not (-180 <= x0 <= x1 <= 180 and -90 <= y0 <= y1 <= 90):
            raise ValueError(f"{Path(path.split('!')[0]).name} has no projection (.prj) and its coordinates are not "
                             "longitude/latitude; include the .prj file")
        df = df.set_crs(4326)
    df = df[df.geometry.notna() & df.geom_type.isin(["Polygon", "MultiPolygon"])].to_crs(4326)
    if df.empty:
        raise ValueError("no polygons in that file (points and lines are not outlines)")
    return _from_geojson(json.loads(df[["geometry"]].to_json()), label)


def load_polygon(path: str | Path, label: str = "") -> Area:
    r"""A drawn or exported outline: GeoJSON, KML, KMZ, a shapefile (.shp with
    its .dbf/.shx/.prj beside it, or all of them in a .zip), GeoPackage.
    Projected shapefiles (State Plane, UTM) are reprojected to lon/lat.
    Windows paths (\\wsl.localhost\..., C:\...) are accepted."""
    p = normalize_path(path)
    _open_outline(p)
    label = label or p.stem
    suf = p.suffix.lower()
    if suf in (".geojson", ".json"):
        doc = _read_json_file(p)
        crs = str(((doc.get("crs") or {}).get("properties") or {}).get("name") or "")
        if crs and not re.search(r"CRS84|EPSG:+4326$|EPSG::?4326\b", crs, re.I):
            # an old-style GeoJSON that names its projection: GDAL reads the
            # "crs" member, so it is reprojected like a shapefile
            return _from_gis(str(p), label)
        return _from_geojson(doc, label)
    if suf == ".kml":
        return _from_kml(_text(p.read_bytes(), p.name, "KML"), label, p.name)
    if suf == ".kmz":
        with _open_zip(p) as z:
            kml = next((n for n in z.namelist() if n.lower().endswith(".kml")), None)
            if kml is None:
                raise ValueError(f"{p.name} is not a valid KMZ file: it holds no .kml")
            text = _read_member(z, kml, p.name, "KML")
        return _from_kml(text, label, f"{kml} in {p.name}")
    if suf == ".zip":
        with _open_zip(p) as z:
            names = [n for n in z.namelist() if not n.startswith("__MACOSX")]
            inner = next((n for ext in (".shp", ".gpkg", ".geojson", ".json", ".kml") for n in names
                          if n.lower().endswith(ext)), None)
            if inner is None:
                raise ValueError(f"{p.name} holds no shapefile, GeoPackage, GeoJSON or KML")
            is_kml = inner.lower().endswith(".kml")
            text = (None if inner.lower().endswith((".shp", ".gpkg"))
                    else _read_member(z, inner, p.name, "KML" if is_kml else "GeoJSON"))
        if text is None:
            return _from_gis(f"zip://{p}!{inner}", label)
        where = f"{inner} in {p.name}"
        return _from_kml(text, label, where) if is_kml else _from_geojson(_json_text(text, where), label)
    if suf in GIS_SUFFIXES:
        if suf == ".shp":
            missing = [e for e in (".dbf", ".shx") if not p.with_suffix(e).exists()
                       and not p.with_suffix(e.upper()).exists()]
            if missing:
                raise ValueError(f"a shapefile is several files; {', '.join(missing)} is missing beside {p.name} "
                                 "(or zip them all together)")
        return _from_gis(str(p), label)
    raise ValueError(f"{suf} is not an outline format this reads (GeoJSON, KML, KMZ, shapefile, zip, GeoPackage)")


def merge(parts: list[Area], label: str) -> Area:
    """Several outlines as one (e.g. every mine file uploaded)."""
    return Area(label, [r for a in parts for r in a.rings], "polygon",
                [a.holes_of(i) for a in parts for i in range(len(a.rings))])


def clip_to(area: Area, bounds: tuple[float, float, float, float]) -> Area | None:
    """Only the rings that reach into bounds (W, S, E, N): a statewide mine
    layer shrinks to the mines that matter. None when nothing is left."""
    w, s, e, n = bounds
    keep = [i for i, r in enumerate(area.rings)
            if min(p[0] for p in r) <= e and max(p[0] for p in r) >= w
            and min(p[1] for p in r) <= n and max(p[1] for p in r) >= s]
    return Area(area.label, [area.rings[i] for i in keep], area.source,
                [area.holes_of(i) for i in keep]) if keep else None


def geocode(query: str, limit: int = 6, timeout: int = 20) -> list[dict]:
    """Place-name search (OpenStreetMap Nominatim; one request per call, as
    its usage policy asks). Returns [{name, lat, lon, area}] best first.
    The text goes to OpenStreetMap's servers: a town or a road, never a
    client's or a site's name (Nominatim's policy asks for no confidential
    material, and a client NDA may too)."""
    url = "https://nominatim.openstreetmap.org/search?" + urllib.parse.urlencode(
        {"q": query, "format": "jsonv2", "limit": limit, "addressdetails": 0})
    req = urllib.request.Request(url, headers={"User-Agent": user_agent(), "Accept-Language": "en"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        hits = json.loads(r.read().decode())
    out = []
    for h in hits:
        s, n, w, e = (float(v) for v in h["boundingbox"])
        out.append({"name": h.get("display_name", query), "lat": float(h["lat"]), "lon": float(h["lon"]),
                    "kind": h.get("type", ""), "area": box(s, w, n, e, h.get("name") or query)})
    return out


# ---------------------------------------------------------------- archive
def archive_check(area: Area, start: str, end: str, box: tuple | None = None) -> dict:
    """What Sentinel-1 and OPERA hold over the area: IW SLC scenes by
    direction, track and year, and DISP-S1 frames with granule counts
    (disp_frames, kept for older callers) and, per frame, its track,
    direction and the share of the processing box `box` (W, S, E, N; the
    area's own box when not given) its footprint covers (disp_frame_detail,
    disp_backend.frame_summary). A frame must be chosen from those on the
    chosen track by overlap: by granule count alone the other track's frame
    can win."""
    import asf_search as asf
    wkt = area.wkt()
    res = asf.search(platform=[asf.PLATFORM.SENTINEL1], processingLevel=asf.PRODUCT_TYPE.SLC,
                     beamMode=asf.BEAMMODE.IW, intersectsWith=wkt, start=start, end=end, maxResults=10000)
    tracks: dict = {}
    for r in res:
        pr = r.properties
        key = (pr.get("flightDirection", "?").upper(), int(pr.get("pathNumber") or 0))
        yr = str(pr.get("startTime", ""))[:4]
        day = str(pr.get("startTime", ""))[:10]
        t = tracks.setdefault(key, {"direction": key[0], "track": key[1], "dates": set(), "by_year": {}})
        if day not in t["dates"]:
            t["dates"].add(day)
            t["by_year"][yr] = t["by_year"].get(yr, 0) + 1
    tr = sorted(({**v, "dates": len(v["dates"])} for v in tracks.values()), key=lambda d: -d["dates"])
    frames: dict = {}
    detail: list = []
    try:
        from . import disp_backend as db
        disp = asf.search(shortName=[db.COLLECTION], intersectsWith=wkt, start=start, end=end, maxResults=5000)
        for g in disp:
            m = re.search(r"_F(\d+)_", g.properties["fileName"])
            if m:
                frames[m.group(1)] = frames.get(m.group(1), 0) + 1
        detail = db.frame_summary(disp, tuple(box) if box else area.bbox)
    except Exception as exc:                                 # noqa: BLE001
        frames = {"error": str(exc)[:120]}
    return {"tracks": tr, "disp_frames": frames, "disp_frame_detail": detail}


def frame_for_track(detail: list, track: int, direction: str) -> dict | None:
    """The DISP-S1 frame to process for the chosen track: on that track and
    direction, the largest share of the processing box (granules break ties).
    None when no frame is on that track."""
    rows = [d for d in detail or [] if d.get("track") == int(track)
            and str(d.get("direction") or "").lower() == str(direction).lower()]
    return max(rows, key=lambda d: (d.get("overlap_fraction") or 0.0, d.get("granules") or 0)) if rows else None


# A processing box, area of interest or report area must lie this much inside
# the union of the track's burst footprints; less and part of it would be
# missing from every product without a word (the finding: an IW2-only config
# for ground in IW3). The footprints tile without gaps, so a box inside the
# track reads 1.000 (Site D, track 19 ascending, 5 Oct 2026: IW2, 1.000).
MIN_BURST_COVERAGE = 0.99


def _shapely(area: Area):
    from shapely.geometry import Polygon
    from shapely.ops import unary_union
    return unary_union([Polygon(r, area.holes_of(i)).buffer(0) for i, r in enumerate(area.rings)])


def burst_coverage(area: Area, track: int, direction: str, targets: dict | None = None,
                   max_results: int = 250) -> dict:
    """
    One track's Sentinel-1 bursts over the area (the ASF burst catalogue):
    {"swaths": [IW numbers whose bursts touch it], "bursts": distinct burst IDs,
     "covered": share of the area inside their union, "targets": {name: share}}
    for any further outlines passed in targets. The catalogue lists every
    date's copy of every burst (and each polarisation), so it is deduplicated
    by burst ID; 250 results hold several whole dates for any box one stack
    can process. Raises when the catalogue cannot be asked - never a guess.
    """
    import asf_search as asf
    from shapely.geometry import shape
    from shapely.ops import unary_union
    res = asf.search(dataset=asf.DATASET.SLC_BURST, intersectsWith=area.wkt(), relativeOrbit=int(track),
                     flightDirection=direction.upper(), maxResults=max_results)
    ids: dict = {}
    for r in res:
        b = r.properties.get("burst") or {}
        m = re.search(r"IW(\d)", str(b.get("subswath", "")))
        if not m:
            continue
        key = b.get("fullBurstID") or (m.group(1), b.get("burstIndex"), str(r.properties.get("startTime"))[:10])
        if key not in ids and r.geometry:
            ids[key] = (int(m.group(1)), shape(r.geometry))
    union = unary_union([g for _, g in ids.values()]) if ids else None

    def share(a: Area) -> float:
        g = _shapely(a)
        if union is None or g.area <= 0:
            return 0.0
        return float(union.intersection(g).area / g.area)
    return {"swaths": sorted({sw for sw, _ in ids.values()}), "bursts": len(ids), "covered": share(area),
            "targets": {k: share(v) for k, v in (targets or {}).items()}}


def subswaths(area: Area, track: int, direction: str) -> list[int]:
    """IW sub-swaths whose bursts touch the area on one track (burst catalogue)."""
    return burst_coverage(area, track, direction)["swaths"]


def gnss_near(area: Area, radius_km: float | None = None) -> list[dict]:
    """GNSS stations within radius_km of the area's centre (UNR/NGL list);
    by default the radius the tie itself uses (config.GNSS_TIE_RADIUS_KM)."""
    from . import gnss_validation as gv
    if radius_km is None:
        from .config import GNSS_TIE_RADIUS_KM
        radius_km = GNSS_TIE_RADIUS_KM
    w, s, e, n = area.bbox
    from .settings import projects_root
    st = gv.load_stations(projects_root() / "gnss")
    return gv.nearby(st, 0.5 * (s + n), 0.5 * (w + e), radius_km)


def gnss_preview(area: Area, start: str, end: str, radius_km: float | None = None) -> dict:
    """The GNSS stations the tie is expected to use for this area and these
    dates (gnss_tie.preview_stations: inside config.GNSS_TIE_RADIUS_KM and at
    least 3 years of overlap with the record), apart from those nearby that it
    will not use. The builder used to list every station within 50 km and say
    'the report is anchored to them' while the Site D tie used one."""
    from . import gnss_tie
    from .config import GNSS_TIE_RADIUS_KM
    r = float(GNSS_TIE_RADIUS_KM if radius_km is None else radius_km)
    w, s, e, n = area.bbox
    d0, d1 = str(start).replace("-", "")[:8], str(end).replace("-", "")[:8]
    rows = gnss_tie.preview_stations(0.5 * (s + n), 0.5 * (w + e), d0, d1, radius=r)
    source = "NGL MIDAS catalogue (station spans checked against the dates)"
    if not rows:
        # no cached catalogue on this machine: the plain station list, spans unchecked
        rows = [{"site": g.get("site"), "lat": g.get("lat"), "lon": g.get("lon"),
                 "distance_km": round(float(g.get("distance_km") or 0.0), 1), "will_use": None}
                for g in gnss_near(area, r)]
        source = "NGL station list (spans not checked: the MIDAS catalogue is not cached)"
    used = [x for x in rows if x.get("will_use")]
    not_used = [x for x in rows if x.get("will_use") is False]
    warn = None
    if rows and rows[0].get("will_use") is not None and len(used) < 3:
        warn = (f"Only {len(used)} station(s) can anchor the report"
                + (": the background is then widened to the spread of station rates in the region"
                   if used else ": results stay relative to the area's own farmland"))
    return {"radius_km": r, "stations": rows, "used": [x["site"] for x in used],
            "not_used": [x["site"] for x in not_used], "source": source, "warning": warn}


def far_field_check(box: Area, mines: Area | None, report_areas: list, buffer_m: float = 2000.0) -> dict:
    """
    Is there enough ordinary ground in the processing box to compare the
    report areas with? The farmland comparison places each area's footprint
    over the box minus the mines and their buffer; under about 10 times the
    largest report area it rests on a handful of independent placements and
    the area reads 'not testable' (Site B's 103 km2 zone: ~4). Areas are
    measured in a local km plane (enough at these sizes).
    """
    import math
    from shapely.affinity import scale
    w, s, e, n = box.bbox
    kx, ky = 111.32 * math.cos(math.radians(0.5 * (s + n))), 110.95

    def km(a: Area):
        return scale(_shapely(a), xfact=kx, yfact=ky, origin=(0, 0))
    far = km(box)
    if mines is not None:
        far = far.difference(km(mines).buffer(buffer_m / 1000.0))
    sizes = {label: km(a).area for label, a in report_areas if not isinstance(a, str)}
    largest = max(sizes, key=sizes.get) if sizes else None
    ratio = far.area / sizes[largest] if largest and sizes[largest] > 0 else None
    out = {"far_field_km2": round(far.area, 1), "largest_report_area": largest,
           "largest_report_area_km2": round(sizes[largest], 2) if largest else None,
           "ratio": round(ratio, 1) if ratio is not None else None, "warning": None}
    if ratio is not None and ratio < FAR_FIELD_MIN_RATIO:
        # the ratio grows roughly with the box area: (1 + 2 pad / side)^2
        side = math.sqrt(max(sizes[largest], 1e-6))
        out["warning"] = (f"Only {ratio:.1f} times the largest report area ({largest}, "
                          f"{sizes[largest]:.1f} km2) of ordinary ground lies in the processing box outside the "
                          f"mines and their {buffer_m / 1000:g} km buffer; under ~{FAR_FIELD_MIN_RATIO:.0f}x the "
                          "farmland comparison can be too thin to test that area. Widen the margin (pad_km; "
                          f"about {max(2.0, side):.0f} km more) or draw smaller report areas.")
    return out


FAR_FIELD_MIN_RATIO = 10.0


def ram_estimate(box: Area, n_dates: int, posting_m: float = 15.0) -> dict:
    """The products-stage memory for this box (stack_backend._plan_memory's
    rule) against this machine's RAM, for the builder's archive check."""
    from . import settings
    from .stack_backend import products_ram_gb
    ew, ns = box.size_km()
    cells = int((ew * 1000.0 / posting_m) * (ns * 1000.0 / posting_m))
    gb = products_ram_gb(cells, int(n_dates))
    mem = float(settings.machine_resources().get("mem_gb") or 0.0)
    level = "bad" if mem and gb > 0.9 * mem else "warn" if mem and gb > 0.5 * mem else "ok"
    return {"cells": cells, "n_dates": int(n_dates), "est_ram_gb": round(gb, 1), "mem_gb": round(mem, 1),
            "level": level,
            "text": (f"The products stage needs about {gb:.0f} GB of this machine's {mem:.0f} GB RAM "
                     f"({cells / 1e6:.1f} M map cells x {n_dates} dates)"
                     + (": that WILL NOT FIT - shrink the area" if level == "bad"
                        else ": run nothing else alongside it" if level == "warn" else ""))}


# ---------------------------------------------------------------- dated mined areas
def attribute_fields(path) -> dict:
    """The attribute table of an uploaded polygon layer: field names, a few
    values of each, and the fields that look like a year and a name - for the
    builder's 'dated mined areas' layer, whose year attribute drives the
    panel-by-panel history."""
    fc = _feature_collection(path)
    props = [f.get("properties") or {} for f in fc.get("features") or []]
    fields = sorted({k for p in props for k in p})
    samples = {k: [str(p.get(k))[:24] for p in props[:400] if p.get(k) not in (None, "")][:3] for k in fields}
    from .panel_history import _year_of
    yearish = [k for k in fields if sum(_year_of(p.get(k)) is not None for p in props[:400]) >= 0.8 * min(400, len(props))]
    year_guess = next((k for k in yearish if re.search(r"year|yr|date|mined|start", k, re.I)), yearish[0] if yearish else "")
    name_guess = next((k for k in fields if re.search(r"name|mine|permit|pit|panel", k, re.I) and k != year_guess), "")
    return {"fields": fields, "samples": samples, "year_guess": year_guess, "name_guess": name_guess,
            "features": len(props)}


def _feature_collection(src) -> dict:
    """A polygon layer as a GeoJSON FeatureCollection in lon/lat WITH its
    attributes (load_polygon keeps only the outline). src: a file path or an
    already-parsed FeatureCollection."""
    if isinstance(src, dict):
        fc = src
    else:
        p = normalize_path(src)
        _open_outline(p)                  # the outline files' checks (16 Oct 2026, SEC-03 / SEC-04)
        suf = p.suffix.lower()
        if suf in (".geojson", ".json"):
            fc = _read_json_file(p)
            crs = str(((fc.get("crs") or {}).get("properties") or {}).get("name") or "")
            if crs and not re.search(r"CRS84|EPSG:+4326$|EPSG::?4326\b", crs, re.I):
                fc = _gis_fc(str(p))
        elif suf == ".zip":
            with _open_zip(p) as z:
                names = [n for n in z.namelist() if not n.startswith("__MACOSX")]
            inner = next((n for ext in (".shp", ".gpkg", ".geojson", ".json") for n in names
                          if n.lower().endswith(ext)), None)
            if inner is None:
                raise ValueError(f"{p.name} holds no shapefile, GeoPackage or GeoJSON with attributes")
            fc = _gis_fc(f"zip://{p}!{inner}")
        elif suf in GIS_SUFFIXES:
            fc = _gis_fc(str(p))
        else:
            raise ValueError(f"{p.name}: a dated layer needs an attribute table (shapefile, GeoPackage or GeoJSON)")
    feats = [f for f in fc.get("features") or [] if (f.get("geometry") or {}).get("type") in ("Polygon", "MultiPolygon")]
    if fc.get("type") == "Feature":
        feats = [fc]
    _check_features(len(feats), "dated mined areas")
    n = 0
    for f in feats:
        g = f["geometry"]
        polys = [g["coordinates"]] if g["type"] == "Polygon" else g["coordinates"]
        n += sum(len(r) for poly in polys for r in poly)
        _check_vertices(n, "dated mined areas")
        _check_lonlat([r for poly in polys for r in poly], "dated mined areas")
    return {"type": "FeatureCollection", "features": feats}


def _gis_fc(path: str) -> dict:
    import geopandas as gpd
    df = gpd.read_file(path)
    if df.crs is None:
        df = df.set_crs(4326)
    df = df[df.geometry.notna() & df.geom_type.isin(["Polygon", "MultiPolygon"])].to_crs(4326)
    return json.loads(df.to_json())


def dated_features(src, year_field: str, name_field: str = "", bounds: tuple | None = None) -> dict:
    """The dated mined areas as panel_history reads them: a FeatureCollection
    whose properties are normalised to {"mined_year": <as stored>,
    "mine_name": <as stored or "">} (so several uploads merge into one file
    with one field name each), cut to the polygons reaching into bounds.
    Raises ValueError in plain words when the year field is missing or holds
    no year."""
    from .panel_history import _year_of
    fc = _feature_collection(src)
    feats = fc["features"]
    if not feats:
        raise ValueError("the dated mined-area layer holds no polygons")
    if not year_field or not any(year_field in (f.get("properties") or {}) for f in feats):
        have = sorted({k for f in feats for k in (f.get("properties") or {})})
        raise ValueError(f"the dated mined-area layer has no attribute {year_field!r}; it has: "
                         + (", ".join(have[:20]) or "no attributes"))
    out = []
    for f in feats:
        p = f.get("properties") or {}
        if _year_of(p.get(year_field)) is None:
            continue
        if bounds is not None:
            a = _from_geojson({"type": "FeatureCollection", "features": [f]}, "x")
            if clip_to(a, tuple(bounds)) is None:
                continue
        out.append({"type": "Feature", "geometry": f["geometry"],
                    "properties": {"mined_year": p.get(year_field),
                                   "mine_name": p.get(name_field, "") if name_field else ""}})
    if not out and bounds is None:
        raise ValueError(f"no polygon in the dated mined-area layer has a readable year in {year_field!r}")
    return {"type": "FeatureCollection", "features": out}


# ---------------------------------------------------------------- config
# What the dashboard offers for a NEW area. Phase-linked is the official route;
# DISP-S1 is fast screening (and an independent check). The classic stack route
# still exists in the pipeline for comparisons but is not offered here.
ROUTES = {
    "miaplpy": "Phase-linked stack: official results (1-2 days, ~300 GB)",
    "disp": "OPERA DISP-S1: fast screening, not official (hours)",
}
ROUTE_NAMES = {"miaplpy": "phase-linked (official)", "disp": "DISP-S1 (screening)",
               "stack": "classic (comparison)", "combined": "combined ascending + descending"}


def slug(text: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return s[:40] or "area"


# ---------------------------------------------------------------- names
# Names the client sees: the project title and every report area row (table,
# summary sentence, map legend, chart titles, KMZ folders). 'project_zone' -
# the uploaded file's stem - reached Site B's client report this way
# ("project_zone moved 0.25 in"). These rules match release.py's file-stem
# test (lower-case words joined by _ or -), so the builder refuses what the
# release gate would block, at the moment the name can still be typed.
_GENERIC_NAME = re.compile(
    r"(project ?(area|zone)|area( of interest)?|aoi|polygons?|layers?|shapes?|features?|outlines?|boundary|"
    r"untitled|new layer|report area|underground workings|surface mines|other known movement)( ?\d+)?", re.I)
_FILE_SUFFIX = re.compile(r"\.(geojson|json|shp|kml|kmz|gpkg|zip|gdb|fgb)$", re.I)
# Line breaks and other control characters (incl. the Unicode line and
# paragraph separators). A name typed into a page can carry a newline, and in
# a CSV's '#' header line that starts a new line a spreadsheet reads as a cell
# - '=cmd|...' then ran as a formula (16 Oct 2026, readiness review SEC-05).
_CONTROL = re.compile(r"[\x00-\x1f\x7f\u2028\u2029]")
# A first character a spreadsheet reads as the start of a formula.
FORMULA_START = ("=", "+", "-", "@")


def one_line(text) -> str:
    """text on one line: every control character a space, runs of spaces one."""
    return re.sub(r"\s+", " ", _CONTROL.sub(" ", str(text or ""))).strip()


# Every CSV and workbook cell the package writes (16 Oct 2026, readiness
# review SEC-05). Refusing a typed report-area name does not cover text that
# arrives in a file: a mine or panel name from a shapefile attribute, a point
# id from a ground-truth CSV, a line name in the events layer. A text cell
# that starts like a formula gets a leading apostrophe (the OWASP convention
# for CSV injection), so a spreadsheet shows it as text; tab and carriage
# return start a formula too once a spreadsheet trims them. A plain number
# ('-1.5', '+2', '-1e-3') is data, not a formula, and is left alone.
CELL_FORMULA_START = FORMULA_START + ("\t", "\r")
_PLAIN_NUMBER = re.compile(r"[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?")


def formula_safe(value, workbook: bool = False):
    """value for one spreadsheet cell: a str a spreadsheet would read as a
    formula, prefixed with an apostrophe; anything else (numbers, None, a
    plain-number string, ordinary text) returned unchanged.

    workbook=True for a cell written into an .xlsx through openpyxl: a string
    cell there is never evaluated, and the one way text becomes a formula is
    openpyxl's own rule that a str starting with '=' IS one - so only that is
    changed, and '-0.50 to 0.30' keeps its look in the client's workbook."""
    if not isinstance(value, str):
        return value
    if workbook:
        return "'" + value if len(value) > 1 and value.startswith("=") else value
    if value.startswith(CELL_FORMULA_START) and not _PLAIN_NUMBER.fullmatch(value):
        return "'" + value
    return value


def formula_safe_frame(df, workbook: bool = False):
    """df with every text cell and column name through formula_safe; numeric
    columns are not touched. df itself is never changed: a copy comes back
    when a cell had to change, else df."""
    out = None
    for i in range(df.shape[1]):
        col = df.iloc[:, i]
        if col.dtype.kind != "O" and str(col.dtype) != "string":
            continue
        safe = col.map(lambda v: formula_safe(v, workbook))
        if not safe.equals(col):
            out = df.copy() if out is None else out
            out.isetitem(i, safe)
    names = [formula_safe(c, workbook) for c in df.columns]
    if names != list(df.columns):
        out = df.copy() if out is None else out
        out.columns = names
    return df if out is None else out


def formula_safe_text(text) -> str:
    """text for a place in a CSV that no writer quotes - a '#' comment line,
    a hand-joined row: on one line (one_line), and every piece a spreadsheet
    splits off at a comma or a semicolon (its delimiter in many locales)
    through formula_safe. A '#' line is read as cells too, so a name
    'North,=cmd|...' put '=cmd|...' in a cell of its own; leading spaces and
    quotes before the formula character do not protect it."""
    out = []
    for piece in re.split(r"([,;])", one_line(text)):
        body = piece.lstrip(' "')
        if body.startswith(FORMULA_START) and not _PLAIN_NUMBER.fullmatch(body.rstrip(' "')):
            piece = piece[:len(piece) - len(body)] + "'" + body
        out.append(piece)
    return "".join(out)


def client_name_problem(name: str, file_stems=()) -> str | None:
    """Why `name` cannot head a row of the client report, or None when it can.
    file_stems are the names of the files the outline came from: a name equal
    to one of them is refused when it reads like a file name (no space), so a
    file called 'Area A1.geojson' can keep its name and 'SiteD_permit_v3'
    or 'Boundary' cannot."""
    n = (name or "").strip()
    if not n:
        return "has no name: type the name the client report should show"
    if _CONTROL.search(n):
        return f"{one_line(n)!r} contains a line break or other control character; type it on one line"
    if n.startswith(FORMULA_START):
        return (f"{n!r} starts with '{n[0]}', which spreadsheets read as the start of a formula; "
                "start it with a letter or a digit")
    if ":" in n:
        return (f"{n!r} contains ':', which the config uses to separate a report area's name from its outline; "
                "use a dash or a comma")
    if _FILE_SUFFIX.search(n):
        return f"{n!r} is a file name; type the name the client report should show"
    if "_" in n or re.fullmatch(r"[a-z0-9]+(-[a-z0-9]+)+", n):
        return (f"{n!r} reads like a file name; type the plain name the client report should show "
                "(e.g. 'Site D permit area')")
    if _GENERIC_NAME.fullmatch(n):
        return f"{n!r} is a placeholder; type the name the client report should show"
    if " " not in n and any(n.casefold() == str(st).strip().casefold() for st in file_stems if st):
        return (f"{n!r} is the uploaded file's name; type the name the client report should show "
                "(it heads the area's row in the report)")
    return None


# ---------------------------------------------------------------- projects
# Files write_project itself puts in <project>/context; report polygons are
# written as report_<slug>.geojson so no area name can overwrite one of them
# (a report area called "Surface mines" used to share surface_mines.geojson).
RESERVED_CONTEXT = ("area_of_interest", "deforming_area", "surface_mines", "underground_mines", "other_movement",
                    "mine_panels", "builder_layers")
SCREENING_SUFFIX = "_disp_screening"
MINE_KIND_LABEL = {"underground": "Underground mine workings", "surface": "Surface mines",
                   "other": "Other known ground movement", "dated": "Dated mined areas"}


def project_key(name: str, route: str) -> str:
    """The folder and config name. A DISP-S1 screening never shares the
    official project's folder: disp_chain deletes mintpy/geo/geo_*.h5 and
    writes timeseries.h5, temporalCoherence.h5 and the client report where the
    phase-linked run keeps them, so 'Site D' screened in the dashboard after
    'Site D' was processed officially replaced the official results."""
    return slug(name) + (SCREENING_SUFFIX if route == "disp" else "")


def project_paths(name: str, route: str, projects_root: Path | None = None,
                  configs_dir: Path | None = None) -> tuple[str, Path, Path]:
    """(key, project folder, config file) a write_project call would use."""
    from . import settings
    key = project_key(name, route)
    root = Path(projects_root or settings.projects_root())
    return key, root / key, Path(configs_dir or settings.configs_dir()) / f"{key}.yaml"


def _results_in(P: Path) -> list[str]:
    """What a project folder already holds that a new config would reuse or
    overwrite, in words."""
    checks = (("a client report", P / "deliverables" / "client_report.json"),
              ("MintPy results", P / "mintpy" / "geo" / "geo_velocity.h5"),
              ("MintPy inputs", P / "mintpy" / "inputs" / "ifgramStack.h5"),
              ("a DISP-S1 time series", P / "mintpy" / "disp_chain.json"),
              ("phase-linked results", P / "miaplpy"),
              ("a coregistered stack", P / "stack"))
    return [what for what, f in checks if f.exists()]


def existing_project(name: str, route: str, projects_root: Path | None = None,
                     configs_dir: Path | None = None) -> dict | None:
    """What is already there under this name and route: {config (or None),
    project_dir, backend, holds: [...]}; None when nothing is."""
    _, P, out = project_paths(name, route, projects_root, configs_dir)
    holds = _results_in(P)
    if not out.exists() and not holds:
        return None
    backend = None
    if out.exists():
        try:
            import yaml
            backend = (yaml.safe_load(out.read_text()) or {}).get("backend")
        except Exception:                                    # noqa: BLE001 - a broken file still exists
            backend = "unreadable"
    return {"config": str(out) if out.exists() else None, "project_dir": str(P), "backend": backend, "holds": holds}


def last_backup(config: Path) -> Path | None:
    """The newest backup write_project kept of this config."""
    config = Path(config)
    baks = sorted(config.parent.glob(f"{config.name}.bak_*"))
    return baks[-1] if baks else None


def _portable(path: Path, root: Path) -> str:
    """A path inside the projects root as $INSAR_PROJECTS/..., which
    load_config expands, so the config keeps working when the projects root
    moves (settings.json, $INSAR_PROJECTS) - before, every path was absolute
    and a moved root silently dropped the mine polygons from the farmland
    comparison. Outside the root, or when this process's $INSAR_PROJECTS is
    another folder (a test, an explicit projects_root), the absolute path."""
    import os
    env = os.environ.get("INSAR_PROJECTS")
    try:
        if env and Path(os.path.expanduser(env)).resolve() == Path(root).resolve():
            return "$INSAR_PROJECTS/" + Path(path).resolve().relative_to(Path(root).resolve()).as_posix()
    except ValueError:
        pass
    return str(path)


POSTING_M = 15.0                   # the geocoded posting every route delivers


def geocode_lalo_step(lat: float, metres: float = POSTING_M) -> str:
    """mintpy_geocode_lalo_step for `metres` x `metres` cells at latitude
    `lat`: "-dlat,dlon" in degrees (MintPy's order and sign). At 38.3 N this
    is the long-standing -0.000135,0.000172."""
    import math
    m_per_deg = 111_320.0
    dlat = metres / m_per_deg
    dlon = metres / (m_per_deg * max(math.cos(math.radians(float(lat))), 0.05))
    return f"-{dlat:.6f},{dlon:.6f}"


# Weather delays for new projects (24 Oct 2026, Kyle: the Copernicus ERA5 queue
# had been clogged since 22 Oct): NOAA HRRR through RAiDER on the MintPy routes,
# where HRRR serves the whole record and its environment is installed; ERA5
# (pyaps, the CDS) otherwise. The DISP-S1 chain keeps the record's ERA5.
HRRR_ENV = "raider-env"


def weather_default(route: str, bbox, start: str) -> tuple[dict, str, str]:
    """({config fields}, start date to use, note for the config's header).
    HRRR covers the continental US from 2016-07-15 (tropo_disp.MODEL_START,
    check_model_range): an earlier start moves to that day, since a series
    corrected only from the middle would carry a step the ground never had."""
    from types import SimpleNamespace
    from . import settings, tropo_disp
    era5 = {"mintpy_tropo_method": "pyaps"}
    if route not in ("miaplpy", "stack"):
        return era5, start, ""
    first = tropo_disp.MODEL_START["HRRR"].strftime("%Y-%m-%d")
    ok, why = tropo_disp.check_model_range(SimpleNamespace(weather_model="HRRR", bbox=tuple(bbox)),
                                           [max(start, first).replace("-", "")])
    if not ok:
        return era5, start, f"Weather delays from ERA5: {why}"
    if not settings.env_prefix(HRRR_ENV).exists():
        return era5, start, (f"Weather delays from ERA5: NOAA HRRR is the default, but the {HRRR_ENV} environment "
                             "is not installed here (install/install.sh makes it).")
    note = "Weather delays from NOAA HRRR (3 km, no Copernicus queue)"
    if start < first:
        note += f"; the record starts {first}, the first day HRRR covers (asked for {start})"
        start = first
    return {"mintpy_tropo_method": "raider", "weather_model": "HRRR", "raider_env": HRRR_ENV}, start, note + "."


def write_project(name: str, area: Area, route: str, start: str, end: str, *, track: int,
                  direction: str, report_areas: list[tuple[str, Area | str]], deforming: Area | None = None,
                  underground: Area | None = None, surface: Area | None = None,
                  disp_frame: str = "", swaths: list[int] | None = None, pad_km: float = 5.0,
                  reference_lalo: str = "", mine_sources: dict | None = None, file_stems: tuple = (),
                  overwrite: bool = False, dated: dict | None = None,
                  projects_root: Path | None = None, configs_dir: Path | None = None) -> Path:
    """Write <configs_dir>/<key>.yaml (and the outlines into the project's
    context/) from the production defaults. Validated by load_config before
    it is returned; an invalid config is not left on disk.

    Refuses, before anything is written:
      - a report-area or project name the client should not see
        (client_name_problem), and report names that share a file name;
      - an existing config or a folder that already holds results, unless
        overwrite (the old config is then kept as <key>.yaml.bak_<time>);
      - on the stack routes, sub-swaths it cannot look up for THIS track over
        the padded box, or bursts that leave part of the box, the area or a
        report area uncovered. swaths=None looks them up (the default for
        callers); a list is trusted as already looked up for this track.
    mine_sources: {"underground"|"surface"|"other"|"dated": "source, vintage"} for
    mine_layer_sources (the report credits each layer with it).
    dated: the dated mined areas (dated_features: properties mined_year /
    mine_name) for panel_history's mining-year history; written to
    context/mine_panels.geojson, kept out of the ramp fit and the farmland
    like every mine layer."""
    import os
    import time

    import yaml

    from . import settings
    from .config import GNSS_TIE_RADIUS_KM, load_config
    if route not in ROUTES and route != "stack":
        raise ValueError(f"route {route!r} is not one of {list(ROUTES)}")
    problem = client_name_problem(name, file_stems)
    if problem:
        raise ValueError(f"project name {problem}")
    # the project area's own row still carrying the outline's default name (the
    # dashboard passes area.label, a file's stem for an upload) takes the
    # project's name instead - the default the review asked for - and the
    # header says so; any other report name with a problem is refused below
    renamed = next((label for label, a in report_areas if a is area and client_name_problem(label, file_stems)), None)
    if renamed is not None:
        report_areas = [(name if (a is area and label == renamed) else label, a) for label, a in report_areas]
    names, slugs = set(), {}
    for label, a in report_areas:
        problem = client_name_problem(label, file_stems)
        if problem:
            raise ValueError(f"report area {problem}")
        if label.strip().casefold() in names:
            raise ValueError(f"two report areas are called {label!r}; rename one")
        names.add(label.strip().casefold())
        if not isinstance(a, str):
            k = slug(label)
            if k in slugs:
                raise ValueError(f"report areas {slugs[k]!r} and {label!r} differ only in punctuation or after "
                                 "40 characters; give them clearly different names")
            slugs[k] = label

    key, P, out = project_paths(name, route, projects_root, configs_dir)
    root = P.parent
    have = existing_project(name, route, projects_root, configs_dir)
    if have and not overwrite:
        what = ", ".join(have["holds"]) or "no results yet"
        raise FileExistsError(f"a project called {name!r} already exists ({have['config'] or have['project_dir']}; "
                              f"{have['backend'] or 'no config'}; holds {what}). Choose another name, or replace "
                              "it: the current config is then kept as a .bak copy beside it")
    proc = area.padded(pad_km)
    w, s, e, n = proc.bbox
    weather, start, weather_note = weather_default(route, (w, s, e, n), start)

    cover = None
    if route != "disp":
        if swaths is None:
            targets = {"the area of interest": area}
            targets.update({label: a for label, a in report_areas if not isinstance(a, str)})
            try:
                cover = burst_coverage(proc, track, direction, targets)
            except Exception as exc:                         # noqa: BLE001 - any lookup failure stops here
                raise RuntimeError(f"could not look up the bursts of {direction.lower()} track {track} over the "
                                   f"processing box ({str(exc)[:160]}). Nothing was written; check the connection "
                                   "and try again - the sub-swaths are not guessed") from exc
            swaths = cover["swaths"]
            if not swaths:
                raise ValueError(f"no Sentinel-1 bursts of {direction.lower()} track {track} cover the processing "
                                 "box; choose a track the archive check lists")
            short = [(what, f) for what, f in (("the processing box", cover["covered"]), *cover["targets"].items())
                     if f < MIN_BURST_COVERAGE]
            if short:
                raise ValueError(f"{direction.lower()} track {track} (IW{', IW'.join(map(str, swaths))}) covers only "
                                 + ", ".join(f"{100 * f:.1f}% of {what}" for what, f in short)
                                 + ": that part would be missing from every product. Reduce the margin, move the "
                                 "area, or choose another track")
        elif not swaths:
            raise ValueError("no sub-swaths given")

    if reference_lalo:                                       # refused before anything is written
        lat, lon = (float(v) for v in reference_lalo.split(","))
        _check_latlon(lat, lon)
    ctx = P / "context"
    ctx.mkdir(parents=True, exist_ok=True)
    # the outlines context/ holds now, put back if the new config is refused:
    # the restored config names the same files, and must not find the
    # rejected outlines in them
    prev = {f: f.read_bytes() for f in ctx.glob("*.geojson")}

    def _restore_context():
        for f in ctx.glob("*.geojson"):
            if f not in prev:
                f.unlink()
        for f, b in prev.items():
            f.write_bytes(b)
    specs = []
    for label, a in report_areas:
        if isinstance(a, str):                              # "full"
            specs.append(f"{label}:{a}")
        elif a.is_box:
            aw, as_, ae, an = a.bbox
            specs.append(f"{label}:{as_:.6f},{aw:.6f},{an:.6f},{ae:.6f}")
        else:
            f = ctx / f"report_{slug(label)}.geojson"
            f.write_text(json.dumps(a.geojson()))
            specs.append(f"{label}:{_portable(f, root)}")
    doc = {"backend": route, "project_dir": _portable(P, root), "bbox": [round(v, 5) for v in (w, s, e, n)],
           "start_date": start, "end_date": end, "track": int(track), "orbit_direction": direction.lower(),
           "report_areas": specs, "gnss_tie_radius_km": float(GNSS_TIE_RADIUS_KM), "display_units": "in",
           "deforming_threshold": 0.5, "acceptance_far_field_buffer_m": 2000,
           "min_temporal_coherence": 0.4, **weather, "gnss_validation": False}
    if reference_lalo:
        doc["reference_lalo"] = f"{lat:.5f},{lon:.5f}"      # a spot known to be stable (checked above)
    # the route is in the title: a screening report must not read like the official one
    doc["project_title"] = f"{name} (DISP-S1 screening)" if route == "disp" else name
    # mapped mines by type: each kept for the report, all of them (plus any other
    # known movement) out of the ramp fit and the farmland comparison
    typed, sources = {}, []
    mine_sources = mine_sources or {}
    for field_name, kind, layer, fname in (
            ("underground_mines_geojson", "underground", underground, "underground_mines.geojson"),
            ("surface_mines_geojson", "surface", surface, "surface_mines.geojson")):
        layer = clip_to(layer, (w, s, e, n)) if layer is not None else None
        if layer is not None:
            (ctx / fname).write_text(json.dumps(layer.geojson()))
            doc[field_name] = _portable(ctx / fname, root)
            typed[field_name] = layer
            sources.append((kind, layer))
    if deforming is not None:
        deforming = clip_to(deforming, (w, s, e, n))
        if deforming is not None:
            # its own file too, so the report can say a feature lies on ground
            # already known to move (gap-surface_mine_site-7); still merged
            # into the ramp-fit exclusion below like every mine layer
            (ctx / "other_movement.geojson").write_text(json.dumps(deforming.geojson()))
            doc["other_movement_geojson"] = _portable(ctx / "other_movement.geojson", root)
            sources.append(("other", deforming))
    dated_area = None
    if dated is not None:
        fc = {"type": "FeatureCollection",
              "features": [f for f in dated.get("features") or []
                           if clip_to(_from_geojson({"type": "FeatureCollection", "features": [f]}, "x"),
                                      (w, s, e, n)) is not None]}
        if fc["features"]:
            (ctx / "mine_panels.geojson").write_text(json.dumps(fc))
            doc.update({"mined_year_geojson": _portable(ctx / "mine_panels.geojson", root),
                        "mined_year_field": "mined_year", "mined_name_field": "mine_name"})
            dated_area = _from_geojson(fc, "dated mined areas")
            sources.append(("dated", dated_area))
    parts = [a for a in (deforming, *typed.values(), dated_area) if a is not None]
    deforming = merge(parts, "mines and known movement") if parts else None
    if deforming is not None:
        f = ctx / "deforming_area.geojson"
        f.write_text(json.dumps(deforming.geojson()))
        doc.update({"mintpy_deramp": "linear", "deramp_exclude_geojson": _portable(f, root)})
    else:
        # the deramp refuses to fit a plane through unknown deformation
        doc["mintpy_deramp"] = "no"
    if sources:
        # where each kept layer came from: the report credits it under the map
        doc["mine_layer_sources"] = [f"{MINE_KIND_LABEL[k]}: {str(mine_sources.get(k) or '').strip() or 'source and date not recorded'}"
                                     for k, _ in sources]
    res = settings.machine_resources()
    pools = settings.sized_parallelism(res["mem_gb"], res["cores"])
    if route == "disp":
        doc.update({"disp_frame_id": disp_frame, "disp_record_chain": True, "disp_use_recommended_mask": False,
                    "disp_use_static_layers": True, "disp_apply_corrections": True,
                    # the record grades on temporal coherence and never masks by
                    # component; DISP-S1's component 0 grows within each segment
                    "disp_mask_connected_components": False, "disp_flag_unwrap_jumps": True,
                    "disp_min_valid_fraction": 0.5, "disp_prefetch_workers": 12})
    else:
        if route == "stack":
            # MintPy's network pruning; the phase-linked route ignores both
            # (stack_backend writes 'no' for it), so they are not written there
            doc.update({"max_temp_baseline": 120, "max_perp_baseline": 350.0})
        doc.update({"stack_swaths": " ".join(str(v) for v in swaths), "stack_polarization": "vv",
                    "stack_num_connections": 3,
                    "stack_range_looks": 11, "stack_azimuth_looks": 3, "stack_filter_strength": 0.4,
                    "stack_unw_method": "snaphu", "stack_unw_full_cost": True, "stack_crop_unwrap": True,
                    "stack_use_burst2safe": True, "stack_cleanup_offsets": True,
                    "stack_cleanup_misreg_inputs": True, "stack_cleanup_burst_geometry": True,
                    "mintpy_unwrap_error_method": "bridging+phase_closure", "subset_to_aoi": True,
                    # MintPy's dask pool (0 = none) sized for this machine; its
                    # maxMemory and the stack pools are left unset, so the config
                    # sizes them for whichever machine runs it (config._sized)
                    "mintpy_num_workers": pools["mintpy_num_workers"],
                    "stack_unw_engine": "snaphu2", "stack_unw_cost_mode": "DEFO"})
        # a stuck snaphu pair cannot hang the run. max_plausible_velocity_mm_yr
        # is left at its default: it is only a count, and the physical limit is
        # the rate DIFFERENCE between neighbouring cells, which the QA measures
        # (quality_assessment.json 'plausibility'), not a cell's absolute rate
        doc.update({"stack_unwrap_timeout_min": 240})
        # ~15 m posting at THIS area's latitude (generality-16): the config
        # default -0.000135,0.000172 is 15 x 15 m only near 38 N (Indiana); at
        # 55 N its 0.000172 deg of longitude is 11 m
        doc["mintpy_geocode_lalo_step"] = geocode_lalo_step(0.5 * (s + n))
        if route == "miaplpy":
            # 120 MB per worker, one core each (measured on Site A); a few cores left free
            doc["miaplpy_num_processors"] = pools["miaplpy_num_processors"]
            doc.update({"stack_virtual_merge": "False", "stack_cleanup_coreg_after_filter": False,
                        "miaplpy_env": "miaplpy", "miaplpy_phase_linking_method": "sequential_EMI",
                        "miaplpy_ministack_size": 10, "miaplpy_range_window": 15, "miaplpy_azimuth_window": 15})
    out.parent.mkdir(parents=True, exist_ok=True)
    backup = None
    if out.exists():                                         # overwrite: keep the old one beside it
        import shutil
        backup = out.with_name(f"{out.name}.bak_{time.strftime('%Y%m%d_%H%M%S')}")
        shutil.copy2(out, backup)
    cov = (f"; bursts cover {100 * cover['covered']:.1f}% of the box" if cover else "")
    ff = far_field_check(proc, deforming, report_areas, float(doc["acceptance_far_field_buffer_m"]))
    if ff["warning"]:
        import logging
        logging.getLogger(__name__).warning("%s: %s", name, ff["warning"])
    header = (f"# {name}: written by the insar dashboard ({ROUTES.get(route, ROUTE_NAMES[route])}).\n"
              f"# Area of interest {area.label} ({area.source}); processing box padded {pad_km:g} km"
              + (f"; IW{', IW'.join(map(str, swaths))} of {direction.lower()} track {track}{cov}" if route != "disp" else "")
              + ".\n"
              + (f"# The project area's report row is named after the project ({renamed!r} reads like a file name).\n"
                 if renamed is not None else "")
              + f"# Pools sized for {res['mem_gb']:.0f} GB RAM / {res['cores']} cores (MintPy {pools['mintpy_num_workers']} "
              f"workers, phase linking {pools['miaplpy_num_processors']}); unset pools follow the machine that runs it.\n"
              + (f"# {weather_note}\n" if weather_note else "")
              + (f"# WARNING: {ff['warning']}\n" if ff["warning"] else ""))
    out.write_text(header + yaml.safe_dump(doc, sort_keys=False))
    try:
        load_config(str(out))
    except Exception:
        if backup is not None:
            os.replace(backup, out)                          # the project keeps its working config
        else:
            out.unlink()
        _restore_context()                                   # ... and the outlines that config names
        raise
    (ctx / "area_of_interest.geojson").write_text(json.dumps(area.geojson()))
    return out
