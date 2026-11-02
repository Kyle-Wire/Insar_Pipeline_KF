"""Map exports for print: a project's map on a chosen page, in colour or black
and white, in an official layout (app_server's Export map).

    export(cfg, config, opts) -> Path        <project>/exports/maps/<name>_<page>_<orientation>_<stamp>.pdf|png

The layout: the map frame with latitude/longitude ticks on its edges, a
scale bar (feet/miles and metres, true ground distance at the map's centre)
and a north arrow; beside it (landscape) or below it (portrait) the title,
the legend of every layer drawn, the notes, the data credits
(credits.short_line, as the report prints them, plus the base map's), and
the title block: the company logo and name (settings: firm, company_logo),
who prepared it, the date, the project, the print scale and the page.

The map is drawn in Web Mercator (EPSG:3857, the base maps' grid); every
raster is reprojected onto it, nearest neighbour, so no value is invented.
The layers are the app's (app_server.RASTERS, the outlines, points, GNSS,
features) with their own colour rules (app_server.colour); in black and white
the rasters use grey ramps and the outlines hatching and dashes. A project
whose report is not FINAL is printed with DRAFT across the map: the map
carries the report's figures' status, whatever the page says otherwise.
"""
from __future__ import annotations

import io
import json
import math
import textwrap
import time
import urllib.request
from datetime import datetime
from pathlib import Path

PAGES = {"letter": ("Letter", 8.5, 11), "legal": ("Legal", 8.5, 14), "tabloid": ("Tabloid (11x17)", 11, 17),
         "ansi_c": ("ANSI C (17x22)", 17, 22), "ansi_d": ("ANSI D (22x34)", 22, 34),
         "a4": ("A4", 8.27, 11.69), "a3": ("A3", 11.69, 16.54)}
BASEMAPS = {
    "imagery": ("https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}", 19,
                "Imagery: Esri, Vantor, Earthstar Geographics, and the GIS User Community"),
    "osm": ("https://tile.openstreetmap.org/{z}/{x}/{y}.png", 19, "Base map: (c) OpenStreetMap contributors"),
    "usgs": ("https://basemap.nationalmap.gov/arcgis/rest/services/USGSTopo/MapServer/tile/{z}/{y}/{x}", 16,
             "Base map: USGS The National Map"),
    "topo": ("https://a.tile.opentopomap.org/{z}/{x}/{y}.png", 17,
             "Base map: (c) OpenStreetMap contributors, SRTM | (c) OpenTopoMap (CC-BY-SA)"),
    "grey": ("https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}",
             16, "Base map: Esri, HERE, Garmin, (c) OpenStreetMap contributors, and the GIS user community"),
    "none": (None, 0, ""),
}
TILE_CACHE = Path.home() / ".cache" / "insar_app" / "tiles"
MAX_TILES = 100
R = 6378137.0
M_PER_FT = 0.3048
MARGIN = 0.4                       # inches round the page


# ---------------------------------------------------------------- geometry
def merc(lon, lat):
    import numpy as np
    lon, lat = np.asarray(lon, float), np.clip(np.asarray(lat, float), -85.0511, 85.0511)
    return R * np.radians(lon), R * np.log(np.tan(np.pi / 4 + np.radians(lat) / 2))


def unmerc(x, y):
    import numpy as np
    return np.degrees(np.asarray(x) / R), np.degrees(2 * np.arctan(np.exp(np.asarray(y) / R)) - np.pi / 2)


def fit_extent(bounds, aspect):
    """Mercator (x0, x1, y0, y1) holding bounds (W, S, E, N), widened to the frame's width/height."""
    W, S, E, N = bounds
    (x0, x1), (y0, y1) = merc([W, E], [S, N])
    cx, cy, w, h = (x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0) * 1.04, (y1 - y0) * 1.04
    if w / h < aspect:
        w = h * aspect
    else:
        h = w / aspect
    return cx - w / 2, cx + w / 2, cy - h / 2, cy + h / 2


# ---------------------------------------------------------------- the base map
def _tile(url: str, z: int, x: int, y: int, ua: str):
    from PIL import Image
    key = TILE_CACHE / (url.split("//", 1)[1].split("/", 1)[0].replace(":", "_")) / str(z) / str(x) / f"{y}.img"
    if key.exists() and time.time() - key.stat().st_mtime < 30 * 86400:
        data = key.read_bytes()
    else:
        req = urllib.request.Request(url.format(z=z, x=x, y=y), headers={"User-Agent": ua})
        with urllib.request.urlopen(req, timeout=20) as r:
            data = r.read()
        key.parent.mkdir(parents=True, exist_ok=True)
        key.write_bytes(data)
    return Image.open(io.BytesIO(data)).convert("RGB")


def basemap(name: str, extent, px_wide: int, bw: bool):
    """(RGB array, mercator extent) of the tiles covering extent, or (None, note)."""
    import numpy as np
    from PIL import Image
    url, zmax, _ = BASEMAPS.get(name, BASEMAPS["none"])
    if not url:
        return None, ""
    x0, x1, y0, y1 = extent
    world = 2 * math.pi * R
    z = int(max(1, min(zmax, math.floor(math.log2(px_wide * world / (256 * (x1 - x0)))))))
    while True:
        n = 2 ** z
        tx0, tx1 = int((x0 + world / 2) / world * n), int((x1 + world / 2) / world * n)
        ty0, ty1 = int((world / 2 - y1) / world * n), int((world / 2 - y0) / world * n)
        if (tx1 - tx0 + 1) * (ty1 - ty0 + 1) <= MAX_TILES or z <= 1:
            break
        z -= 1
    from . import version
    ua = f"insar-pipeline/{version.label().split()[1]} (map export)"
    mosaic = Image.new("RGB", ((tx1 - tx0 + 1) * 256, (ty1 - ty0 + 1) * 256), (230, 230, 230))
    failed = 0
    for tx in range(tx0, tx1 + 1):
        for ty in range(ty0, ty1 + 1):
            try:
                mosaic.paste(_tile(url, z, tx % n, ty, ua), ((tx - tx0) * 256, (ty - ty0) * 256))
            except Exception:                                 # noqa: BLE001 - a missing tile stays grey
                failed += 1
    if failed == (tx1 - tx0 + 1) * (ty1 - ty0 + 1):
        return None, "the base map could not be fetched (offline?)"
    a = np.asarray(mosaic.convert("L").convert("RGB") if bw else mosaic) / 255.0
    if bw:
        a = 0.35 + 0.6 * a                                    # lighter, so the black linework reads over it
    ext = (tx0 / n * world - world / 2, (tx1 + 1) / n * world - world / 2,
           world / 2 - (ty1 + 1) / n * world, world / 2 - ty0 / n * world)
    return a, ext


# ---------------------------------------------------------------- the layers
OUTLINE = {   # style: (colour line, colour fill, alpha, dash, bw line, bw hatch)
    "bbox": ("#ffffff", None, 0, (6, 4), "#000000", None),
    "aoi": ("#ffd43b", "#ffd43b", .08, None, "#000000", None),
    "mine": ("#ff922b", "#ff922b", .22, None, "#000000", "///"),
    "report": ("#74c0fc", None, 0, (2, 3), "#000000", None),
    "frame": ("#da77f2", None, 0, (10, 5), "#555555", None),
    "water": ("#4dabf7", "#4dabf7", .45, None, "#777777", "..."),
}


def _draw_geojson(ax, gj, style: str, bw: bool, label: str, handles: list):
    import matplotlib.patheffects as pe
    from matplotlib.patches import Patch, PathPatch
    from matplotlib.lines import Line2D
    from matplotlib.path import Path as MPath
    line, fill, alpha, dash, bline, hatch = OUTLINE.get(style, OUTLINE["report"])
    col = bline if bw else line
    key_col = "#000000" if style == "bbox" else col           # a white line vanishes on the page's white
    halo = [pe.Stroke(linewidth=3.2, foreground="white" if (bw or style != "bbox") else "black", alpha=.85), pe.Normal()]

    def rings(geom):
        t, c = geom.get("type"), geom.get("coordinates") or []
        if t == "Polygon":
            yield c, True
        elif t == "MultiPolygon":
            for p in c:
                yield p, True
        elif t == "LineString":
            yield [c], False
        elif t == "MultiLineString":
            yield c, False

    for ft in gj.get("features") or []:
        for parts, closed in rings(ft.get("geometry") or {}):
            verts, codes = [], []
            for r in parts:
                if len(r) < 2:
                    continue
                x, y = merc([p[0] for p in r], [p[1] for p in r])
                verts += list(zip(x, y))
                codes += [MPath.MOVETO] + [MPath.LINETO] * (len(r) - 1)
            if not verts:
                continue
            path = MPath(verts, codes)
            filled = closed and (fill or (bw and hatch))
            pp = PathPatch(path, facecolor=("none" if not filled or bw else fill), alpha=None,
                           edgecolor=col, lw=1.4, ls=(0, dash) if dash else "-", hatch=hatch if bw else None,
                           fill=bool(filled), zorder=6, path_effects=halo)
            if filled and not bw:
                ax.add_patch(PathPatch(path, facecolor=fill, edgecolor="none", alpha=alpha, zorder=5))
                pp.set_facecolor("none")
            ax.add_patch(pp)
    handles.append(Patch(facecolor=("white" if bw else (fill or "none")), edgecolor=col, hatch=hatch if bw else None,
                         ls=(0, dash) if dash else "-", lw=1.4, alpha=None if bw or not fill else max(alpha, .35), label=label)
                   if fill or (bw and hatch) else Line2D([], [], color=key_col, lw=1.6, ls=(0, dash) if dash else "-", label=label))


def _draw_raster(ax, cfg, lid, extent, px_wide, bw):
    """Reproject the layer's window onto the frame's mercator grid and draw it; returns its legend."""
    import numpy as np
    from rasterio.transform import from_bounds
    from rasterio.warp import Resampling, reproject
    from . import app_server as S
    f, raw = S.raster_file(cfg, lid)
    if f is None:
        return None
    x0, x1, y0, y1 = extent
    (W, E), (Sb, N) = unmerc([x0, x1], [y0, y1])
    a, (w, s, e, n) = S.read_raster(f, lid, max_px=2400, bounds=(W - .01, Sb - .01, E + .01, N + .01))
    if a.size == 0 or not np.isfinite(a).any():
        return None
    ny, nx = max(64, int(px_wide * (y1 - y0) / (x1 - x0))), max(64, px_wide)
    dst = np.full((ny, nx), np.nan, "float32")
    reproject(a, dst, src_transform=from_bounds(w, s, e, n, a.shape[1], a.shape[0]), src_crs="EPSG:4326",
              dst_transform=from_bounds(x0, y0, x1, y1, nx, ny), dst_crs="EPSG:3857",
              resampling=Resampling.nearest, src_nodata=np.nan, dst_nodata=np.nan)
    rgba, legend = S.colour(dst, lid, bw=bw, alpha=0.82)
    ax.imshow(rgba, extent=(x0, x1, y0, y1), origin="upper", interpolation="nearest", zorder=3)
    legend.update(label=S.RASTER[lid]["label"] + (" (raw)" if raw else ""), id=lid)
    return legend


# ---------------------------------------------------------------- page furniture
def _nice(x: float) -> float:
    e = 10 ** math.floor(math.log10(x))
    for m in (1, 2, 2.5, 5, 10):
        if m * e >= x:
            return m * e
    return 10 * e


def _ticks(lo, hi, n=4):
    step = _nice((hi - lo) / n)
    t = math.ceil(lo / step) * step
    out = []
    while t <= hi + 1e-12:
        out.append(round(t, 10))
        t += step
    return out, step


def _deg(v, pos, neg, step):
    nd = max(0, min(4, -int(math.floor(math.log10(step))) + (1 if step / 10 ** math.floor(math.log10(step)) == 2.5 else 0)))
    return f"{abs(v):.{nd}f}°{pos if v >= 0 else neg}"


def _graticule(ax, extent):
    x0, x1, y0, y1 = extent
    (W, E), (S, N) = unmerc([x0, x1], [y0, y1])
    lons, ls = _ticks(W, E)
    lats, ts = _ticks(S, N)
    ax.set_xticks(merc(lons, [S] * len(lons))[0])
    ax.set_xticklabels([_deg(v, "E", "W", ls) for v in lons], fontsize=6.5)
    ax.set_yticks(merc([W] * len(lats), lats)[1])
    ax.set_yticklabels([_deg(v, "N", "S", ts) for v in lats], fontsize=6.5, rotation=90, va="center")
    ax.tick_params(direction="in", length=4, top=True, right=True, labeltop=False, labelright=False, pad=2)


def _scalebar(ax, extent, frame_in):
    """Feet/miles over metres, true ground distance at the map's centre, in an
    opaque box at the lower left. Returns the print scale (1:N)."""
    from matplotlib.patches import Rectangle
    x0, x1, y0, y1 = extent
    fw, fh = frame_in
    ux, uy = (x1 - x0) / fw, (y1 - y0) / fh                   # map units per inch
    lat = float(unmerc(0, (y0 + y1) / 2)[1])
    k = math.cos(math.radians(lat))                           # true metres per mercator metre
    target = min(2.2, fw * 0.28) * ux * k                     # about 2 inches of ground
    ft = target / M_PER_FT
    if ft >= 2640:
        miles = _nice(ft / 5280) if _nice(ft / 5280) * 5280 * M_PER_FT <= target * 1.15 else _nice(ft / 5280) / 2
        L1, t1 = miles * 5280 * M_PER_FT, f"{miles:g} mile" + ("s" if miles != 1 else "")
    else:
        f_ = _nice(ft) if _nice(ft) * M_PER_FT <= target * 1.15 else _nice(ft) / 2
        L1, t1 = f_ * M_PER_FT, f"{f_:,.0f} ft"
    mm = _nice(target) if _nice(target) <= target * 1.15 else _nice(target) / 2
    L2, t2 = mm, (f"{mm / 1000:g} km" if mm >= 1000 else f"{mm:,.0f} m")
    w1, w2 = L1 / k, L2 / k
    bx, by = x0 + .22 * ux, y0 + .2 * uy
    ax.add_patch(Rectangle((bx - .1 * ux, by - .1 * uy), max(w1, w2) + .5 * ux, .62 * uy, fc="white", ec="black", lw=.6, zorder=20))
    h = .06 * uy
    for L, w, yb in ((L1, w1, by + .3 * uy), (L2, w2, by + .12 * uy)):
        for i in range(4):
            ax.add_patch(Rectangle((bx + i * w / 4, yb), w / 4, h, fc="black" if i % 2 == 0 else "white", ec="black", lw=.5, zorder=21))
    kw = dict(fontsize=6, zorder=22)
    ax.text(bx + w1 + .06 * ux, by + .3 * uy + h / 2, t1, va="center", **kw)
    ax.text(bx + w2 + .06 * ux, by + .12 * uy + h / 2, t2, va="center", **kw)
    ax.text(bx, by + .3 * uy + h + .03 * uy, "0", ha="center", va="bottom", **kw)
    return round((x1 - x0) * k / (fw * 0.0254))


def _north(ax, extent, frame_in):
    from matplotlib.patches import Polygon, Rectangle
    x0, x1, y0, y1 = extent
    ux, uy = (x1 - x0) / frame_in[0], (y1 - y0) / frame_in[1]
    cx, cy = x1 - .42 * ux, y1 - .62 * uy
    ax.add_patch(Rectangle((cx - .26 * ux, cy - .36 * uy), .52 * ux, .78 * uy, fc="white", ec="black", lw=.6, zorder=20))
    ax.add_patch(Polygon([(cx, cy + .22 * uy), (cx - .12 * ux, cy - .24 * uy), (cx, cy - .14 * uy)], fc="black", ec="black", lw=.5, zorder=21))
    ax.add_patch(Polygon([(cx, cy + .22 * uy), (cx + .12 * ux, cy - .24 * uy), (cx, cy - .14 * uy)], fc="white", ec="black", lw=.5, zorder=21))
    ax.text(cx, cy + .25 * uy, "N", ha="center", va="bottom", fontsize=8, fontweight="bold", zorder=22)


def _fit(ax, text: str, w_in: float, h_in: float, fs_max: float, fs_min: float = 4.5, **kw):
    """Write text into its box at the largest font that fits (down to fs_min)."""
    fs = fs_max
    while True:
        t = _wrap(text, w_in * .9, fs)
        need = (t.count("\n") + 1) * fs * 1.32 / 72 + .34
        if need <= h_in or fs <= fs_min:
            break
        fs -= .25
    ax.text(.04, 1 - .3 / h_in, t, transform=ax.transAxes, fontsize=fs, va="top", linespacing=1.3, **kw)


def _wrap(text: str, width_in: float, fs: float) -> str:
    chars = max(12, int(width_in * 72 / (fs * 0.6)))
    return "\n".join(textwrap.fill(p, chars) if p.strip() else "" for p in str(text).split("\n"))


def _box(fig, rect, title=None):
    ax = fig.add_axes(rect)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_linewidth(.8)
    if title:
        ax.text(.03, .97, title.upper(), transform=ax.transAxes, fontsize=6.5, fontweight="bold", va="top", color="#333333")
    return ax


# ---------------------------------------------------------------- the page
def export(cfg, config: str, opts: dict) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.lines import Line2D
    from . import app_server as S, credits, dashboard as db, settings, version

    P = Path(cfg.project_dir)
    page = opts.get("page", "tabloid")
    if page not in PAGES:
        raise ValueError(f"unknown page size {page!r}")
    pname, pw, ph = PAGES[page]
    landscape = opts.get("orientation", "landscape") == "landscape"
    W, H = (max(pw, ph), min(pw, ph)) if landscape else (min(pw, ph), max(pw, ph))
    bw = bool(opts.get("bw"))
    fmt = "png" if opts.get("format") == "png" else "pdf"
    dpi = 300 if fmt == "png" else 200
    rep = S._load_json(P / "deliverables" / "client_report.json") or {}
    rs = db.release_status(cfg) if rep else None
    final = bool(rs and rs.get("final"))
    ident = settings.identity()
    sset = settings.load()
    logo = sset.get("company_logo") if opts.get("logo", True) else None

    fig = plt.figure(figsize=(W, H), dpi=dpi)
    fig.patch.set_facecolor("white")
    m, gap = MARGIN, 0.18
    # map frame and the panel beside (landscape) or below (portrait) it, in page fractions
    if landscape:
        pan_w = min(4.2, max(2.7, 0.27 * W))
        map_rect = (m, m + 0.2, W - 2 * m - pan_w - gap, H - 2 * m - 0.2)
        pan = (W - m - pan_w, m, pan_w, H - 2 * m)
    else:
        pan_h = min(4.6, max(2.9, 0.3 * H))
        map_rect = (m, m + pan_h + gap + 0.2, W - 2 * m, H - 2 * m - pan_h - gap - 0.2)
        pan = (m, m, W - 2 * m, pan_h)
    frac = lambda x, y, w, h: (x / W, y / H, w / W, h / H)   # noqa: E731
    ax = fig.add_axes(frac(*map_rect))
    aspect = map_rect[2] / map_rect[3]
    bounds = opts.get("bounds") or list(cfg.bbox)
    extent = fit_extent(bounds, aspect)
    px_wide = int(map_rect[2] * min(dpi, 220))
    notes_auto = []

    bm, bext = basemap(opts.get("basemap", "imagery"), extent, px_wide, bw)
    if bm is not None:
        ax.imshow(bm, extent=bext, origin="upper", interpolation="bilinear", zorder=1)
    elif bext:
        notes_auto.append(bext)
    ax.set_facecolor("#eeeeee")

    layers = [x for x in (opts.get("layers") or []) if isinstance(x, str)]
    meta = {L["id"]: L for L in S.layer_list(cfg)}
    handles, ramps = [], []
    for lid in layers:                                         # rasters first, outlines over them
        if lid in S.RASTER:
            lg = _draw_raster(ax, cfg, lid, extent, px_wide, bw)
            if lg:
                ramps.append(lg)
    body = {"config": config}
    for lid in layers:
        L = meta.get(lid)
        if not L or not L.get("ready") or lid in S.RASTER:
            continue
        if lid in ("points", "gnss", "features"):
            continue
        try:
            d = S.api_layer({**body, "id": lid})
        except Exception:                                     # noqa: BLE001 - left off the map
            continue
        if d.get("kind") == "geojson":
            _draw_geojson(ax, d["data"], L["style"], bw, L["label"], handles)
    if "points" in layers and meta.get("points", {}).get("ready"):
        d = S._points(config)
        x, y = merc(d["lon"], d["lat"])
        t = np.clip(np.asarray(d["v"]) / d["limit"], -1, 1)
        from matplotlib import colormaps
        cols = colormaps["gray" if bw else "RdBu"]((t + 1) / 2)
        ax.scatter(x, y, s=2.2, c=cols, lw=0, zorder=7)
        ramps.append({"label": "Measurement points, vertical" + (" (relative)" if d["relative"] else ""), "min": -d["limit"],
                      "max": d["limit"], "units": "in/yr", "style": "div", "id": "points",
                      "colors": ["#%02x%02x%02x" % tuple(int(255 * c) for c in colormaps["gray" if bw else "RdBu"](i / 6)[:3])
                                 for i in range(7)]})
    if "gnss" in layers and meta.get("gnss", {}).get("ready"):
        d = S.api_layer({**body, "id": "gnss"})
        for st in d["stations"]:
            x, y = merc(st["lon"], st["lat"])
            if extent[0] <= x <= extent[1] and extent[2] <= y <= extent[3]:
                ax.plot(x, y, marker="^", ms=7, mfc="black" if st["used"] else "white", mec="black", mew=1, zorder=9)
                ax.text(x, y, "  " + st["site"], fontsize=6, va="center", zorder=9,
                        bbox=dict(fc="white", ec="none", alpha=.7, pad=.6))
        handles.append(Line2D([], [], marker="^", ls="", mfc="black", mec="black", label="GNSS station used in the tie"))
    if "features" in layers and meta.get("features", {}).get("ready"):
        d = S.api_layer({**body, "id": "features"})
        for f in d["features"]:
            x, y = merc(f["lon"], f["lat"])
            ax.plot(x, y, marker="o", ms=11, mfc="black", mec="white", mew=1.2, zorder=10)
            ax.text(x, y, str(f["n"]), color="white", fontsize=6.5, ha="center", va="center", fontweight="bold", zorder=11)
        if d["features"]:
            handles.append(Line2D([], [], marker="o", ls="", ms=8, mfc="black", mec="white", label="Feature the report lists (numbered)"))
            notes_auto += [f["text"] for f in d["features"]]
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    _graticule(ax, extent)
    for sp in ax.spines.values():
        sp.set_linewidth(1.2)
    _north(ax, extent, map_rect[2:])
    scale = _scalebar(ax, extent, map_rect[2:])
    if not final:
        import matplotlib.patheffects as pe
        halo = [pe.withStroke(linewidth=4, foreground="white")]
        ax.text(.5, .5, "DRAFT", transform=ax.transAxes, fontsize=min(W, H) * 11, color="black", alpha=.3,
                ha="center", va="center", rotation=30, fontweight="bold", zorder=30, path_effects=halo)
        ax.text(.5, .5, "\n\n\n\nnot for client use: the report is not signed off", transform=ax.transAxes,
                fontsize=min(W, H) * 1.3, color="black", alpha=.6, ha="center", va="center", rotation=30,
                rotation_mode="anchor", zorder=30, path_effects=halo)

    # ---- the panel
    px, py, pw_, ph_ = pan
    title = opts.get("title") or rep.get("title") or P.name
    sub = opts.get("subtitle") or ""
    place = (rep.get("location") or {}).get("place") if isinstance(rep.get("location"), dict) else rep.get("location")
    dates = rep.get("dates") or []
    span = f"{dates[0][:4]}-{dates[-1][:4]}" if len(dates) >= 2 else ""
    if landscape:
        cells = {"title": (px, py + ph_ - 1.25, pw_, 1.25), "block": (px, py, pw_, 1.85)}
        rest_y0, rest_y1 = py + 1.85 + .1, py + ph_ - 1.25 - .1
        avail = rest_y1 - rest_y0
        cells["legend"] = (px, rest_y1 - avail * .45, pw_, avail * .45)
        cells["notes"] = (px, rest_y0 + avail * .2 + .1, pw_, avail * .35 - .2)
        cells["credits"] = (px, rest_y0, pw_, avail * .2)
    else:
        c3 = (pw_ - 2 * .12) / 3
        cells = {"title": (px, py + ph_ - 1.0, c3 * 2 + .12, 1.0), "legend": (px, py, c3, ph_ - 1.1),
                 "notes": (px + c3 + .12, py + (ph_ - 1.1) * .42, c3, (ph_ - 1.1) * .58),
                 "credits": (px + c3 + .12, py, c3, (ph_ - 1.1) * .42 - .1),
                 "block": (px + 2 * (c3 + .12), py, c3, ph_)}
    # title
    a_ = _box(fig, frac(*cells["title"]))
    tw = cells["title"][2]
    a_.text(.04, .8, _wrap(title, tw * .92, 13), transform=a_.transAxes, fontsize=13, fontweight="bold", va="top")
    a_.text(.04, .33, _wrap(" · ".join(x for x in (sub, place, f"Sentinel-1 {span}" if span else "") if x), tw * .92, 8),
            transform=a_.transAxes, fontsize=8, va="top", color="#333333")
    # legend
    a_ = _box(fig, frac(*cells["legend"]), "Legend")
    lw_in, lh_in = cells["legend"][2], cells["legend"][3]
    y = 1 - .32 / lh_in
    for r in ramps:
        a_.text(.04, y, _wrap(r["label"], lw_in * .9, 7), transform=a_.transAxes, fontsize=7, va="top", fontweight="bold")
        y -= (.2 + .12 * r["label"].count("\n")) / lh_in
        if r.get("classes"):
            for c in r["classes"]:
                a_.add_artist(matplotlib.patches.Rectangle((.04, y - .09 / lh_in), .06, .1 / lh_in, fc=c["color"], ec="black", lw=.4,
                                                           transform=a_.transAxes))
                a_.text(.13, y - .04 / lh_in, c["name"], transform=a_.transAxes, fontsize=6.5, va="center")
                y -= .15 / lh_in
        else:
            from matplotlib.colors import LinearSegmentedColormap
            cm = LinearSegmentedColormap.from_list("r", r["colors"])
            ia = a_.inset_axes([.04, y - .14 / lh_in, .7, .1 / lh_in])
            ia.imshow(np.linspace(0, 1, 256)[None, :], aspect="auto", cmap=cm)
            ia.set_xticks([]); ia.set_yticks([])
            fmt_ = (lambda v: f"{v:.2f}") if abs(r["max"]) < 2 else (lambda v: f"{v:.1f}")
            a_.text(.04, y - .17 / lh_in, fmt_(r["min"]), transform=a_.transAxes, fontsize=6, va="top")
            a_.text(.74, y - .17 / lh_in, fmt_(r["max"]) + "  " + (r.get("units") or ""), transform=a_.transAxes, fontsize=6, va="top", ha="right")
            y -= .32 / lh_in
            if r.get("style") == "div":
                los = r["id"].startswith("los")
                words = (("black: moving away from the satellite, white: towards" if los else "black: going down, white: going up")
                         if bw else ("red: moving away from the satellite, blue: towards" if los else "red: going down, blue: going up"))
                a_.text(.04, y, _wrap(words, lw_in * .9, 6), transform=a_.transAxes, fontsize=6, va="top", color="#444444")
                y -= .16 / lh_in
        y -= .06 / lh_in
    if handles:
        a_.legend(handles=handles, loc="upper left", bbox_to_anchor=(.02, max(.02, y)), fontsize=6.5, frameon=False,
                  handlelength=2.2, borderaxespad=0)
    # notes
    a_ = _box(fig, frac(*cells["notes"]), "Notes")
    nw = cells["notes"][2]
    anch = db.anchored_of(rep) if rep else None
    auto = ["Rates are per year; vertical movement is relative to ordinary ground of the same kind"
            + (", GNSS-anchored." if anch else "." if anch is False else "."),
            "The map shows the measurements as pictures; the figures to quote are the client report's"
            + (f" ({rep.get('generated', '')})." if rep.get("generated") else "."),
            "Coordinates: WGS 84 latitude/longitude; drawn in Web Mercator (EPSG:3857)."] + notes_auto
    user = (opts.get("notes") or "").strip()
    text = "\n".join(([user, ""] if user else []) + [f"{i}. {n}" for i, n in enumerate(auto, 1)])
    _fit(a_, text, nw, cells["notes"][3], 6.5)
    # credits
    a_ = _box(fig, frac(*cells["credits"]), "Data sources")
    try:
        cred = credits.short_line(cfg, rep or None)
    except Exception:                                         # noqa: BLE001
        cred = "Contains modified Copernicus Sentinel data."
    bmw = BASEMAPS.get(opts.get("basemap", "imagery"), BASEMAPS["none"])[2]
    _fit(a_, cred.replace(" · ", "\n") + ("\n" + bmw if bmw else ""), cells["credits"][2], cells["credits"][3], 5.8,
         color="#222222")
    # title block
    bx, by, bw_in, bh_in = cells["block"]
    a_ = _box(fig, frac(*cells["block"]))
    top = 1.0
    if logo and Path(logo).is_file():
        from PIL import Image
        im = Image.open(logo).convert("RGBA")
        if bw:
            g = im.convert("LA")
            im = Image.merge("RGBA", (g.getchannel(0),) * 3 + (g.getchannel(1),))
        lh = min(.75, bh_in * .38)
        ia = a_.inset_axes([.06, 1 - (lh + .1) / bh_in, .88, lh / bh_in])
        ia.imshow(np.asarray(im))
        ia.set_aspect("equal", anchor="W")
        ia.axis("off")
        top = 1 - (lh + .2) / bh_in
    rows = [("Prepared by", ident.get("firm") or "—"), ("By", ident.get("preparer") or "—"),
            ("Project", P.name), ("Date", datetime.now().strftime("%d %b %Y")),
            ("Scale", f"1:{scale:,} on {pname}" if scale else pname),
            ("Status", "FINAL" if final else "DRAFT: not signed off")]
    rh = (top * bh_in - .2) / len(rows)
    for i, (k, v) in enumerate(rows):
        yy = top - (.1 + i * rh) / bh_in
        a_.text(.06, yy, k.upper(), transform=a_.transAxes, fontsize=5.5, va="top", color="#555555", fontweight="bold")
        a_.text(.06, yy - .13 / bh_in, _wrap(v, bw_in * .85, 7.5), transform=a_.transAxes, fontsize=7.5, va="top",
                fontweight="bold" if k == "Status" and not final else None)
    fig.text(m / W, (m * .45) / H, f"{version.label()} · map export · {('black and white' if bw else 'colour')} · "
             f"{pname} {'landscape' if landscape else 'portrait'}", fontsize=5, color="#777777")

    out = P / "exports" / "maps"
    out.mkdir(parents=True, exist_ok=True)
    f = out / f"{P.name}_{page}_{'landscape' if landscape else 'portrait'}{'_bw' if bw else ''}_{time.strftime('%Y%m%d_%H%M%S')}.{fmt}"
    fig.savefig(f, dpi=dpi, facecolor="white")
    plt.close(fig)
    (out / (f.name + ".json")).write_text(json.dumps({"opts": {k: v for k, v in opts.items() if k != "notes"},
                                                      "notes": opts.get("notes", ""), "final": final,
                                                      "software": version.label(), "written": datetime.now().isoformat()},
                                                     indent=1, default=str))
    return f
