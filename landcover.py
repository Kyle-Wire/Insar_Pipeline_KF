"""
Land cover: which measured cells are wooded (15 Oct 2026, Kyle).

C-band radar (Sentinel-1) does not see the ground under tree canopy: it
sees the leaves and branches, which move with wind, water and the seasons.
Most wooded cells lose coherence and drop out on their own; the ones that
pass the coherence threshold (phase linking can carry some) are the noisiest
ground in the product. They are LABELLED, not deleted:

  1. they are left out of the comparison ground ("ordinary farmland") that
     every relative figure and every chance test is built on
     (significance.far_field_mask), unless that would take away more than
     MAX_FAR_SHARE_REMOVED of it (a wooded region keeps its trees there, and
     says so);
  2. each feature and report area states its wooded share
     (share_of / wooded_sentence);
  3. the client KMZ carries a "Tree cover" layer to lay over the map
     (client_kmz).

The same map names the comparison ground in the report's words
(comparison_ground, 28 Oct 2026): "farmland" only where cropland dominates.

The map: ESA WorldCover 2021 v200 (10 m, global, free; CC BY 4.0), read
window-by-window from its public cloud-optimised GeoTIFFs and saved once per
project as context/landcover_worldcover.tif (ensure()). A user raster in
`landcover_raster` (NLCD or WorldCover class codes) is used instead when set.
Nothing here raises into the pipeline: no map means no labels, said once in
the log.
"""
from __future__ import annotations

import logging
import math
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

WORLDCOVER_URL = ("https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/"
                  "ESA_WorldCover_10m_2021_v200_{tile}_Map.tif")
WORLDCOVER_CREDIT = "ESA WorldCover 10 m 2021 v200 (Zanaga et al. 2022), CC BY 4.0"
LOCAL_NAME = "landcover_worldcover.tif"         # in <project>/context/
PAD_DEG = 0.15                                  # around the bbox: the product grid reaches past it
MAX_CELLS = 400_000_000                         # a box larger than this (about 2 x 2 degrees) is not fetched
MAX_FAR_SHARE_REMOVED = 0.5                     # trees kept in the comparison ground above this share

# class codes -> scheme. WorldCover: 10 tree cover, 20 shrub, 30 grass, 40
# crop, 50 built-up, 60 bare, 70 snow, 80 water, 90 herbaceous wetland, 95
# mangroves, 100 moss. NLCD: 41/42/43 forest, 90 woody wetlands.
WOODED = {"worldcover": (10, 95), "nlcd": (41, 42, 43, 90)}
NLCD_ONLY = {11, 12, 21, 22, 23, 24, 31, 41, 42, 43, 51, 52, 71, 72, 73, 74, 81, 82}
LABELS = {"worldcover": {10: "trees", 20: "shrubland", 30: "grassland", 40: "cropland", 50: "built-up",
                         60: "bare ground", 70: "snow and ice", 80: "water", 90: "wetland", 95: "mangroves",
                         100: "moss and lichen"},
          "nlcd": {11: "water", 21: "developed", 22: "developed", 23: "developed", 24: "developed",
                   31: "bare ground", 41: "trees", 42: "trees", 43: "trees", 52: "shrubland", 71: "grassland",
                   81: "pasture", 82: "cropland", 90: "wooded wetland", 95: "wetland"}}

_GRID_CACHE: dict = {}


def local_path(cfg) -> Path | None:
    """<project>/context/landcover_worldcover.tif, or None without a project."""
    pd = getattr(cfg, "project_dir", None)
    return Path(str(pd)) / "context" / LOCAL_NAME if pd else None


def raster_path(cfg) -> str | None:
    """The land-cover raster this project uses: the user's landcover_raster
    when set and present, else the downloaded WorldCover map, else None."""
    user = str(getattr(cfg, "landcover_raster", "") or "")
    if user:
        return user if Path(user).exists() else None
    lp = local_path(cfg)
    return str(lp) if lp is not None and lp.exists() else None


def enabled(cfg) -> bool:
    return bool(getattr(cfg, "landcover_label_wooded", True))


def _tiles(w: float, s: float, e: float, n: float) -> list[str]:
    """WorldCover 3 x 3 degree tile names covering the box (named by the
    lower-left corner: N39W093)."""
    out = []
    for la in range(int(math.floor(s / 3)) * 3, int(math.floor((n - 1e-9) / 3)) * 3 + 1, 3):
        for lo in range(int(math.floor(w / 3)) * 3, int(math.floor((e - 1e-9) / 3)) * 3 + 1, 3):
            out.append(f"{'N' if la >= 0 else 'S'}{abs(la):02d}{'E' if lo >= 0 else 'W'}{abs(lo):03d}")
    return out


def ensure(cfg, force: bool = False) -> dict:
    """Fetch the WorldCover map for the project box once. Never raises:
    returns {"path", "status", "note"}. status: user | present | fetched |
    off | failed."""
    if not enabled(cfg):
        return {"path": None, "status": "off", "note": "landcover_label_wooded is off"}
    user = str(getattr(cfg, "landcover_raster", "") or "")
    if user:
        ok = Path(user).exists()
        return {"path": user if ok else None, "status": "user" if ok else "failed",
                "note": None if ok else f"landcover_raster {user} does not exist"}
    lp = local_path(cfg)
    if lp is None:
        return {"path": None, "status": "failed", "note": "no project folder"}
    if lp.exists() and not force:
        return {"path": str(lp), "status": "present", "note": None}
    try:
        w, s, e, n = [float(x) for x in cfg.bbox]
        w, s, e, n = w - PAD_DEG, s - PAD_DEG, e + PAD_DEG, n + PAD_DEG
        _fetch_worldcover(w, s, e, n, lp)
        log.info("Land cover: ESA WorldCover saved to %s", lp)
        return {"path": str(lp), "status": "fetched", "note": None}
    except Exception as exc:                                     # noqa: BLE001 - labels are optional
        note = f"the land-cover map could not be fetched ({type(exc).__name__}: {exc}); no tree-cover labels"
        log.warning("Land cover: %s", note)
        return {"path": None, "status": "failed", "note": note}


def _fetch_worldcover(w: float, s: float, e: float, n: float, out: Path) -> None:
    import rasterio
    from rasterio.transform import from_origin
    from rasterio.windows import from_bounds
    res = 1.0 / 12000.0                                          # WorldCover's own grid (8.33e-5 deg)
    # snap to the tiles' grid so every pixel lands exactly once
    w, n = math.floor(w / res) * res, math.ceil(n / res) * res
    e, s = math.ceil(e / res) * res, math.floor(s / res) * res
    nx, ny = int(round((e - w) / res)), int(round((n - s) / res))
    if nx * ny > MAX_CELLS:
        raise ValueError(f"the box is {nx} x {ny} land-cover cells, over the {MAX_CELLS:,} limit")
    arr = np.zeros((ny, nx), np.uint8)
    for tile in _tiles(w, s, e, n):
        with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", GDAL_HTTP_MAX_RETRY="3",
                          GDAL_HTTP_RETRY_DELAY="2"):
            try:
                ds = rasterio.open("/vsicurl/" + WORLDCOVER_URL.format(tile=tile))
            except rasterio.errors.RasterioIOError:
                # WorldCover has no tile over open ocean: that part stays 0 (no data)
                log.info("Land cover: no WorldCover tile %s (open water?)", tile)
                continue
            with ds:
                bw, bs, be, bn = ds.bounds
                iw, is_, ie, in_ = max(w, bw), max(s, bs), min(e, be), min(n, bn)
                if iw >= ie or is_ >= in_:
                    continue
                win = from_bounds(iw, is_, ie, in_, ds.transform).round_offsets().round_lengths()
                data = ds.read(1, window=win)
                r0, c0 = int(round((n - in_) / res)), int(round((iw - w) / res))
                hh, ww = min(data.shape[0], ny - r0), min(data.shape[1], nx - c0)
                arr[r0:r0 + hh, c0:c0 + ww] = data[:hh, :ww]
    out.parent.mkdir(parents=True, exist_ok=True)
    part = out.with_name(out.name + ".partial")
    with rasterio.open(part, "w", driver="GTiff", width=nx, height=ny, count=1, dtype="uint8", crs="EPSG:4326",
                       transform=from_origin(w, n, res, res), nodata=0, compress="deflate", tiled=True) as dst:
        dst.write(arr, 1)
        dst.update_tags(source=WORLDCOVER_CREDIT, scheme="worldcover")
    part.replace(out)


def scheme_of(path: str, classes=None) -> str:
    """'worldcover' or 'nlcd': the file's own tag, else its class codes."""
    try:
        import rasterio
        with rasterio.open(path) as ds:
            tag = (ds.tags() or {}).get("scheme")
        if tag in WOODED:
            return tag
    except Exception:                                            # noqa: BLE001
        pass
    if classes is not None and set(int(c) for c in np.unique(classes) if c >= 0) & NLCD_ONLY:
        return "nlcd"
    return "worldcover"


def sample_on_grid(path, attrs, shape) -> np.ndarray:
    """A class raster sampled (nearest) at every product cell centre; -1
    where it has no value."""
    import rasterio
    from insar_pipeline import significance as sg
    ny, nx = shape
    lat, _ = sg.rc_to_lalo(attrs, np.arange(ny), 0)
    _, lon = sg.rc_to_lalo(attrs, 0, np.arange(nx))
    out = np.full(shape, -1, int)
    with rasterio.open(path) as ds:
        band = ds.read(1)
        tr = None
        if ds.crs and not ds.crs.is_geographic:
            from pyproj import Transformer
            tr = Transformer.from_crs("EPSG:4326", ds.crs, always_xy=True)
        inv = ~ds.transform
        for r in range(ny):
            xs, ys = (tr.transform(lon, np.full(nx, lat[r])) if tr else (lon, np.full(nx, lat[r])))
            xs, ys = np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)
            cc = np.floor(inv.a * xs + inv.b * ys + inv.c).astype(int)
            rr = np.floor(inv.d * xs + inv.e * ys + inv.f).astype(int)
            ok = (rr >= 0) & (rr < band.shape[0]) & (cc >= 0) & (cc < band.shape[1])
            out[r, ok] = band[rr[ok], cc[ok]]
        if ds.nodata is not None:
            out[out == int(ds.nodata)] = -1
    return out


def classes_on_grid(cfg, attrs, shape) -> tuple[np.ndarray, str] | None:
    """(class codes on the product grid, scheme), or None without a map.
    Cached per raster file and grid (far_field_mask asks many times)."""
    if not enabled(cfg):
        return None
    path = raster_path(cfg)
    if not path:
        return None
    try:
        key = (path, Path(path).stat().st_mtime, tuple(shape),
               tuple(str(attrs.get(k)) for k in ("X_FIRST", "Y_FIRST", "X_STEP", "Y_STEP")))
    except OSError:
        return None
    hit = _GRID_CACHE.get(key)
    if hit is None:
        try:
            cls = sample_on_grid(path, attrs, shape)
        except Exception as exc:                                 # noqa: BLE001
            log.warning("Land cover: %s could not be read (%s); no tree-cover labels", path, exc)
            return None
        hit = (cls, scheme_of(path, cls))
        _GRID_CACHE.clear()                                      # one grid at a time is enough
        _GRID_CACHE[key] = hit
    return hit


def wooded_on_grid(cfg, attrs, shape) -> np.ndarray | None:
    """Boolean grid of wooded cells (trees, mangroves / NLCD forest, woody
    wetland), or None without a map."""
    got = classes_on_grid(cfg, attrs, shape)
    if got is None:
        return None
    cls, scheme = got
    return np.isin(cls, WOODED[scheme])


def exclude_wooded(cfg, far: np.ndarray, attrs: dict) -> tuple[np.ndarray, dict | None]:
    """far without its wooded cells (significance.far_field_mask), unless
    that removes more than MAX_FAR_SHARE_REMOVED of it. Returns (far, meta);
    meta None when there is no map or the setting is off."""
    wood = wooded_on_grid(cfg, attrs, far.shape)
    if wood is None:
        return far, None
    n = int(far.sum())
    gone = int((far & wood).sum())
    share = gone / n if n else 0.0
    meta = {"wooded_cells": gone, "wooded_share_of_comparison_ground": round(share, 3)}
    if share > MAX_FAR_SHARE_REMOVED:
        meta.update(applied=False, reason=(f"trees cover {100 * share:.0f}% of the comparison ground; they were "
                                           "kept in it (taking them out would leave too little)"))
        return far, meta
    meta["applied"] = gone > 0
    return far & ~wood, meta


def drop_wooded(cfg, good: np.ndarray, attrs: dict) -> tuple[np.ndarray, dict]:
    """good without its wooded cells when cfg.mask_wooded (21 Oct 2026): the
    delivered products leave woods out, because woods carry less of the
    farmland bias the relative figures remove and so read as rising. Returns
    (good, meta); meta["applied"] is False without a map or with the setting off."""
    meta = {"applied": False, "wooded_cells_removed": 0}
    if not bool(getattr(cfg, "mask_wooded", True)):
        meta["reason"] = "mask_wooded is off"
        return good, meta
    wood = wooded_on_grid(cfg, attrs, np.shape(good))
    if wood is None:
        meta["reason"] = "no land-cover map"
        return good, meta
    gone = int((good & wood).sum())
    n = int(good.sum())
    meta.update(applied=True, wooded_cells_removed=gone,
                wooded_share_of_measured=round(gone / n, 3) if n else 0.0)
    if gone:
        log.info("Wooded cells left out of the products: %d of %d measured (%.1f%%)",
                 gone, n, 100.0 * gone / n)
    return good & ~wood, meta


def share_of(cfg, attrs, mask: np.ndarray, measured: np.ndarray | None = None) -> dict | None:
    """Land cover of a feature or area: {wooded_percent, wooded_measured_percent,
    main_cover, cover_percent{label: %}}. wooded_percent over its whole
    outline; wooded_measured_percent over the cells with a value (measured)."""
    got = classes_on_grid(cfg, attrs, mask.shape)
    if got is None:
        return None
    cls, scheme = got
    m = np.asarray(mask, bool) & (cls >= 0)
    n = int(m.sum())
    if not n:
        return None
    wood = np.isin(cls, WOODED[scheme])
    labels = LABELS[scheme]
    cover: dict = {}
    vals, cnt = np.unique(cls[m], return_counts=True)
    for v, c in zip(vals, cnt):
        lab = labels.get(int(v), "other")
        cover[lab] = cover.get(lab, 0.0) + 100.0 * c / n
    cover = {k: round(v, 1) for k, v in sorted(cover.items(), key=lambda kv: -kv[1])}
    out = {"wooded_percent": round(100.0 * float((wood & m).sum()) / n, 1),
           "main_cover": next(iter(cover)), "cover_percent": cover, "scheme": scheme,
           "wooded_masked": bool(getattr(cfg, "mask_wooded", True))}
    if measured is not None:
        mm = m & np.asarray(measured, bool)
        out["wooded_measured_percent"] = (round(100.0 * float((wood & mm).sum()) / int(mm.sum()), 1)
                                          if mm.any() else None)
    return out


# ------------------------------------------------------------------ the comparison ground in words (28 Oct 2026)
# control_ground_label's default, "farmland", was written for the Indiana
# coalfield. The first combined site's comparison ground is 53% built-up and 44%
# grassland, with no cropland, yet its report said "farmland" (Kyle). The
# words now come from the measured land cover unless someone set the label
# (combine.ground_label's rule, for the single-project report).
DEFAULT_GROUND_LABEL = "farmland"            # config.control_ground_label's default
FARMLAND_COVERS = ("cropland", "pasture")    # NLCD 81 pasture is farmland too
FARMLAND_SHARE = 50.0                        # % of the comparison ground: "farmland" from here up
GROUND_NAMED_SHARE = 20.0                    # a cover this large or more is named in the words
PLAIN_GROUND = "ground"


def configured_ground_label(cfg) -> str:
    """The config's control_ground_label when someone set it; '' when it is
    unset or still the default (then the words are measured)."""
    own = str(getattr(cfg, "control_ground_label", "") or "").strip()
    return "" if own.lower() == DEFAULT_GROUND_LABEL else own


def ground_label_from(own: str, lc: dict | None) -> str:
    """The comparison ground's words: `own` (a label someone set) wins; else
    "farmland" when cropland (and pasture) is FARMLAND_SHARE % or more of the
    measured ground; else "ground (mostly built-up and grassland)", naming
    every cover of GROUND_NAMED_SHARE % or more in a fixed (alphabetical)
    order, so two measurements of nearly the same ground say the same words;
    "ground" alone without a land-cover map."""
    if own:
        return own
    cover = (lc or {}).get("cover_percent") or {}
    if not cover:
        return PLAIN_GROUND
    if sum(float(cover.get(k) or 0.0) for k in FARMLAND_COVERS) >= FARMLAND_SHARE:
        return DEFAULT_GROUND_LABEL
    main = sorted(k for k, v in cover.items() if float(v) >= GROUND_NAMED_SHARE and k != "other")
    return PLAIN_GROUND + (f" (mostly {' and '.join(main)})" if main else "")


def comparison_ground(cfg, attrs, far) -> dict:
    """{label, source, cover_percent, scheme} for the comparison ground `far`
    (significance.far_field_mask; the report's ctx.far_wide). source: config
    (control_ground_label set), measured (the land-cover map) or unmeasured
    (no map, or no mask). Never raises."""
    own = configured_ground_label(cfg)
    lc = None
    if far is not None and not own:
        try:
            lc = share_of(cfg, attrs, np.asarray(far, bool))
        except Exception as exc:                                 # noqa: BLE001 - words are never worth a failed stage
            log.warning("Land cover of the comparison ground not measured (%s)", exc)
    return {"label": ground_label_from(own, lc),
            "source": "config" if own else "measured" if lc else "unmeasured",
            "cover_percent": (lc or {}).get("cover_percent"), "scheme": (lc or {}).get("scheme")}


WOODED_NOTE_PERCENT = 25.0     # a feature this wooded or more gets the caveat sentence


def wooded_sentence(lc: dict | None, what: str = "It") -> str:
    """One sentence for the report, or '' without a map. Above
    WOODED_NOTE_PERCENT it carries the canopy caveat."""
    if not lc:
        return ""
    w = float(lc.get("wooded_percent") or 0.0)
    main = lc.get("main_cover") or "other"
    main_pc = float((lc.get("cover_percent") or {}).get(main) or 0.0)
    if w >= WOODED_NOTE_PERCENT and lc.get("wooded_masked"):
        return (f" Trees cover about {w:.0f}% of it. The satellite (C-band) sees the canopy, not the ground "
                "beneath, so the wooded part is left out and the figures describe the open ground.")
    if w >= WOODED_NOTE_PERCENT:
        return (f" Trees cover about {w:.0f}% of it. The satellite (C-band) sees the canopy, not the ground "
                "beneath, so movement read on wooded ground is less certain and can be the trees themselves.")
    if w >= 5.0:
        return f" {what} is mostly {main} ({main_pc:.0f}%); trees cover about {w:.0f}%."
    return f" {what} is mostly {main} ({main_pc:.0f}%), with little or no tree cover."


# ------------------------------------------------------------------ phase-linking skip (15 Oct 2026)
# Pixels MiaplPy does not phase-link (miaplpy.inversion.mask). MiaplPy keeps
# them as neighbours for the pixels it does link, so a kept pixel's result is
# unchanged; a skipped one gets its raw phase and temporal coherence 0.1,
# under every threshold. Seasonality decides what is skipped:
#  - water: WorldCover's PERMANENT water only, shrunk WATER_EDGE_CELLS cells
#    (~25 m) from the shore, so banks, reservoir drawdown zones and seasonal
#    shorelines are still linked; herbaceous wetland (dry part of the year)
#    is never skipped;
#  - dense woods (opt-in): only the interior of woods, WOODS_EDGE_CELLS
#    (~50 m) in from the edge. Deciduous woods in leaf-off months can hold
#    C-band coherence, and creek-line trees borrow fields' coherence through
#    the 15 x 15 window (Site C: wooded cells passed the 0.4 threshold at
#    97.5%, rate scatter 6.7 mm/yr against cropland's 7.0), so they are kept
#    by default.
SKIP_MODES = ("none", "water", "water_and_dense_woods")
WATER_EDGE_CELLS = 3
WOODS_EDGE_CELLS = 6
WATER_CODES = {"worldcover": (80,), "nlcd": (11,)}


def skip_mode(cfg) -> str:
    m = str(getattr(cfg, "miaplpy_skip", "water") or "none").strip().lower()
    return m if m in SKIP_MODES else "none"


def _skip_on_map(cls: np.ndarray, scheme: str, mode: str) -> tuple[np.ndarray, np.ndarray]:
    """(skip water, skip woods) on the land-cover map's own grid."""
    from scipy import ndimage
    water = np.isin(cls, WATER_CODES[scheme])
    water = ndimage.binary_erosion(water, iterations=WATER_EDGE_CELLS) if water.any() else water
    woods = np.zeros_like(water)
    if mode == "water_and_dense_woods":
        w = np.isin(cls, WOODED[scheme])
        woods = ndimage.binary_erosion(w, iterations=WOODS_EDGE_CELLS) if w.any() else w
    return water, woods


def skip_on_points(cfg, lat: np.ndarray, lon: np.ndarray, mode: str | None = None) -> tuple[np.ndarray, dict] | None:
    """Boolean 'do not phase-link' at each (lat, lon) - the radar grid's own
    geometry - and its record, or None with no land-cover map or mode none.
    Points outside the map, or with no lat/lon, are linked."""
    mode = skip_mode(cfg) if mode is None else mode
    if mode == "none":
        return None
    path = raster_path(cfg)
    if not path:
        return None
    import rasterio
    with rasterio.open(path) as ds:
        if ds.crs and not ds.crs.is_geographic:
            raise ValueError(f"{path}: the phase-linking skip needs a lon/lat land-cover map")
        cls = ds.read(1).astype(np.int32)
        if ds.nodata is not None:
            cls[cls == int(ds.nodata)] = -1
        inv = ~ds.transform
        scheme = scheme_of(path, cls)
    water, woods = _skip_on_map(cls, scheme, mode)
    skip = np.zeros(lat.shape, bool)
    from_water = np.zeros(lat.shape, bool)
    step = 256
    for r in range(0, lat.shape[0], step):
        la, lo = np.asarray(lat[r:r + step], float), np.asarray(lon[r:r + step], float)
        ok = np.isfinite(la) & np.isfinite(lo) & ~((la == 0) & (lo == 0))
        cc = np.floor(inv.a * lo + inv.b * la + inv.c)
        rr = np.floor(inv.d * lo + inv.e * la + inv.f)
        ok &= (rr >= 0) & (rr < cls.shape[0]) & (cc >= 0) & (cc < cls.shape[1])
        ri, ci = rr[ok].astype(int), cc[ok].astype(int)
        w = np.zeros(la.shape, bool); w[ok] = water[ri, ci]
        d = np.zeros(la.shape, bool); d[ok] = woods[ri, ci]
        skip[r:r + step] = w | d
        from_water[r:r + step] = w
    n = skip.size
    rec = {"mode": mode, "map": path, "scheme": scheme, "pixels": int(n),
           "skipped_percent": round(100.0 * float(skip.sum()) / n, 2) if n else 0.0,
           "water_percent": round(100.0 * float(from_water.sum()) / n, 2) if n else 0.0,
           "dense_woods_percent": round(100.0 * float((skip & ~from_water).sum()) / n, 2) if n else 0.0,
           "water_edge_m": "about 25", "woods_edge_m": "about 50" if mode == "water_and_dense_woods" else None}
    return skip, rec
