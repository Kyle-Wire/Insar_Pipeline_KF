"""OPERA DISP-S1-STATIC: per-pixel line-of-sight geometry and terrain masks.

A DISP-S1 displacement granule states only `near_range_incidence_angle`
and `far_range_incidence_angle` for the whole frame - 30.7 and 45.9
degrees on the Site A frame. Everything per-pixel lives in a separate
DISP-S1-STATIC product, one per frame, whose URL every granule gives in
`identification/static_layers_data_access`.

Two layers matter here:

  * `line_of_sight_enu` - three bands holding the east, north and up
    components of the unit vector pointing FROM THE GROUND TO THE
    SATELLITE. This is what converts a line-of-sight number into a
    vertical one, and what projects a GNSS velocity into the radar's
    line of sight so the two can be compared at all.

    Using a single mid-swath incidence angle instead is wrong by up to
    ~7 degrees at the swath edges, which is ~10% on any vertical
    projection. Over a 7 km AOI it happens to be a much smaller error
    (38.17 to 38.71 degrees measured over Site A), but that is a
    property of the AOI, not of the method.

  * `layover_shadow_mask` - terrain where the radar cannot see the
    ground at all. Those pixels are not "low quality", they are
    unmeasurable, and they belong in the denominator of a coverage
    figure rather than in its numerator.

The static product is a per-frame COG, so a window is read over HTTP
rather than downloading the whole file.
"""
import logging
import os
from pathlib import Path

import numpy as np

from .config import PipelineConfig

log = logging.getLogger(__name__)

STATIC_COLLECTION = "OPERA_L3_DISP-S1-STATIC_V1"
LOS_SUFFIX = "_line_of_sight_enu.tif"
LAYOVER_SUFFIX = "_layover_shadow_mask.tif"
DEM_SUFFIX = "_dem.tif"

# GDAL needs to be told to use ~/.netrc and to keep the Earthdata auth
# cookie across the redirect to the data host, or every read returns the
# login page as a 404 body. The cookie jar (_cookie_jar) holds that session.
_GDAL_ENV = {
    "GDAL_HTTP_NETRC": "YES",
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif",
    "GDAL_HTTP_MAX_RETRY": "3",
    "GDAL_HTTP_RETRY_DELAY": "2",
}


def _cookie_jar() -> str:
    """The Earthdata session cookies' file, private to this user:
    ~/.cache/insar_pipeline/earthdata/ (0700) / gdal_edl_cookies.txt (0600).
    It was the fixed, shared /tmp/gdal_edl_cookies.txt, which any other
    account on the machine could read, or create first and so control (16
    Oct 2026, readiness review DAT-03). The folder is ours alone, so the
    file stays private even if curl rewrites it with the umask's mode.
    Without a writable home, a private folder of this process (mkdtemp,
    0700) is used."""
    d = Path.home() / ".cache" / "insar_pipeline" / "earthdata"
    try:
        d.mkdir(mode=0o700, parents=True, exist_ok=True)
        os.chmod(d, 0o700)                   # an older or hand-made folder is narrowed too
    except OSError:
        import tempfile
        d = Path(tempfile.mkdtemp(prefix="insar_edl_"))
    f = d / "gdal_edl_cookies.txt"
    os.close(os.open(f, os.O_CREAT | os.O_WRONLY, 0o600))
    os.chmod(f, 0o600)
    return str(f)


def _apply_gdal_env() -> None:
    for k, v in _GDAL_ENV.items():
        os.environ.setdefault(k, v)
    if not (os.environ.get("GDAL_HTTP_COOKIEFILE") and os.environ.get("GDAL_HTTP_COOKIEJAR")):
        jar = _cookie_jar()
        os.environ.setdefault("GDAL_HTTP_COOKIEFILE", jar)
        os.environ.setdefault("GDAL_HTTP_COOKIEJAR", jar)


def find(cfg: PipelineConfig, frame: str | int | None = None) -> dict[str, str]:
    """URLs of the static layers for the frame covering this AOI."""
    import asf_search as asf

    results = list(asf.search(shortName=[STATIC_COLLECTION],
                              intersectsWith=cfg.aoi_wkt, maxResults=50))
    if not results:
        log.warning("No DISP-S1-STATIC product found for this AOI")
        return {}

    if frame is not None:
        tag = f"F{int(frame):05d}"
        picked = [p for p in results if tag in p.properties["fileName"]]
        if not picked:
            log.warning("No DISP-S1-STATIC for frame %s; AOI returned %s",
                        tag, ", ".join(sorted(
                            {p.properties["fileName"][:40] for p in results})))
            return {}
        results = picked
    elif len({p.properties.get("frameNumber") for p in results}) > 1:
        # Several frames touch the AOI. Pick the one whose centre is
        # closest to the AOI centre rather than whichever sorted first -
        # the wrong frame gives geometry for the wrong side of the swath
        # and nothing downstream would notice.
        import math
        min_lon, min_lat, max_lon, max_lat = cfg.bbox
        clat, clon = (min_lat + max_lat) / 2, (min_lon + max_lon) / 2

        def _offset(r):
            try:
                return math.hypot(float(r.properties["centerLat"]) - clat,
                                  float(r.properties["centerLon"]) - clon)
            except (KeyError, TypeError, ValueError):
                return float("inf")

        results = sorted(results, key=_offset)
        log.info("AOI touches %d static frames; using frame %s, whose centre "
                 "is nearest the AOI. Pass `frame` to override.",
                 len({r.properties.get("frameNumber") for r in results}),
                 results[0].properties.get("frameNumber"))

    # The newest production of the chosen frame. Sorting by file name and
    # taking the FIRST picked the oldest one whenever OPERA had published the
    # static layers more than once (a reprocessing, a new version).
    if frame is not None:
        p = max(results, key=lambda r: (str(r.properties.get("productVersion") or ""),
                                        str(r.properties.get("processingDate") or ""),
                                        r.properties["fileName"]))
    else:
        p = results[0]
    urls = [p.properties["url"]] + list(p.properties.get("additionalUrls") or [])
    out = {}
    for u in urls:
        if u.endswith(LOS_SUFFIX):
            out["los_enu"] = u
        elif u.endswith(LAYOVER_SUFFIX):
            out["layover_shadow"] = u
        elif u.endswith(DEM_SUFFIX):
            out["dem"] = u
    if "los_enu" not in out:
        log.warning("DISP-S1-STATIC for this frame lists no %s", LOS_SUFFIX)
    return out


def _read_window(url: str, grid: dict, bands=None) -> np.ndarray | None:
    """Read `url` over the AOI grid, nearest-neighbour onto that exact grid."""
    _apply_gdal_env()
    import rasterio
    from rasterio.warp import reproject, Resampling
    from rasterio.transform import Affine

    dst_tf = Affine(grid["dx"], 0, grid["x0"] - grid["dx"] / 2,
                    0, grid["dy"], grid["y0"] - grid["dy"] / 2)
    try:
        with rasterio.open("/vsicurl/" + url) as src:
            n = src.count if bands is None else len(bands)
            idx = list(range(1, src.count + 1)) if bands is None else bands
            out = np.full((n, grid["ny"], grid["nx"]), np.nan, dtype="float32")
            # Same CRS and 30 m posting in practice, but reproject rather
            # than assume: a silent half-pixel shift here would put the
            # geometry on the wrong pixels and nothing downstream would
            # notice.
            reproject(
                source=rasterio.band(src, idx),
                destination=out,
                dst_transform=dst_tf,
                dst_crs=f"EPSG:{grid['epsg']}",
                resampling=Resampling.bilinear,
                dst_nodata=np.nan,
            )
    except Exception as exc:
        log.warning("Could not read %s (%s)", url.rsplit("/", 1)[-1], exc)
        return None
    return out


def load(cfg: PipelineConfig, grid: dict, frame=None,
         want_dem: bool = False) -> dict[str, np.ndarray]:
    """
    Per-pixel LOS unit vector and layover/shadow mask on the AOI grid.

    `grid` is {x0, y0, dx, dy, nx, ny, epsg} describing the displacement
    stack, so the geometry lands on exactly the same pixels. `want_dem`
    additionally fetches the frame DEM, which the tropospheric correction
    needs and nothing else does - it is ~90 MB, so it is opt-in.

    Returns {} when the static product cannot be read - callers must cope,
    because the geometry is an improvement on a nominal incidence angle
    rather than a precondition for producing anything at all.
    """
    urls = find(cfg, frame)
    if not urls:
        return {}

    out: dict[str, np.ndarray] = {}
    if "los_enu" in urls:
        enu = _read_window(urls["los_enu"], grid)
        if enu is not None and enu.shape[0] >= 3:
            norm = np.linalg.norm(enu[:3], axis=0)
            good = np.isfinite(norm) & (np.abs(norm - 1.0) < 0.05)
            if not good.any():
                log.warning("line_of_sight_enu has no unit-length vectors on "
                            "this grid - ignoring it")
            else:
                out["los_east"], out["los_north"], out["los_up"] = enu[:3]
                inc = np.degrees(np.arccos(np.clip(enu[2], -1.0, 1.0)))
                out["incidence_deg"] = inc.astype("float32")
                f = inc[np.isfinite(inc)]
                log.info("Per-pixel LOS geometry: incidence %.2f to %.2f deg "
                         "(mean %.2f) over %d pixels",
                         f.min(), f.max(), f.mean(), f.size)
                bad = int(np.sum(np.isfinite(norm) & ~good))
                if bad:
                    log.warning("%d LOS vectors are not unit length - treated "
                                "as missing", bad)
                for k in ("los_east", "los_north", "los_up", "incidence_deg"):
                    out[k] = np.where(good, out[k], np.nan).astype("float32")

    if want_dem and "dem" in urls:
        # Terrain height on the stack grid. The tropospheric delay is
        # strongly height-dependent, so a delay cube has to be sampled at
        # each pixel's own elevation rather than at a single level - that
        # stratified component is most of what a weather model can
        # actually resolve over a small AOI.
        dem = _read_window(urls["dem"], grid, bands=[1])
        if dem is not None:
            out["height"] = dem[0].astype("float32")
            h = out["height"][np.isfinite(out["height"])]
            if h.size:
                log.info("Terrain height on the AOI grid: %.0f to %.0f m "
                         "(relief %.0f m)", h.min(), h.max(), h.max() - h.min())

    if "layover_shadow" in urls:
        ls = _read_window(urls["layover_shadow"], grid, bands=[1])
        if ls is not None:
            # OPERA layover/shadow: 0 is normal ground, non-zero flags
            # layover, shadow or both. Those pixels cannot be measured at
            # all, as distinct from being measured badly.
            m = ls[0]
            out["layover_shadow"] = m.astype("float32")
            flagged = float(np.nanmean(m > 0))
            log.info("Layover/shadow affects %.2f%% of the AOI grid",
                     flagged * 100)
    return out


def project_to_vertical(los: np.ndarray, los_up: np.ndarray,
                        min_cos: float = 0.2) -> np.ndarray:
    """
    Line-of-sight to vertical, ASSUMING MOTION IS PURELY VERTICAL.

    d_vertical = d_los / los_up.

    That assumption is the whole content of the conversion, and it is
    wrong wherever there is horizontal motion - which a single geometry
    cannot detect. On a subsidence site it is usually defensible and it
    must still be stated. With two geometries, decompose instead.
    """
    up = np.where(np.abs(los_up) >= min_cos, los_up, np.nan)
    return los / up


def los_from_enu(east: float, north: float, up: float,
                 los_east, los_north, los_up):
    """
    Project an east/north/up velocity into the radar line of sight.

    Sign follows the layer definition: the unit vector points from the
    ground to the satellite, so a positive result means motion TOWARD the
    satellite - the same convention as `displacement` and as MintPy.
    """
    return east * los_east + north * los_north + up * los_up
