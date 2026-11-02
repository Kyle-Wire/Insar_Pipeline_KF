"""Stage 6: turn the MintPy time series into the four standard products.

  1. LOS displacement points   -> KMZ, clickable, with per-point time series
  2. LOS cumulative displacement graphs -> PNG + CSV
  3. LOS velocity              -> GeoTIFF (+ KMZ)
  4. Quality / signal-to-noise -> GeoTIFF of temporal coherence

Every rate and displacement product is masked by temporal coherence; the
quality raster deliberately is not (it shows where cells were rejected).
Every GeoTIFF carries its quantity, units, sign convention, reference pixel,
date span, threshold, route and source as tags (_write_geotiff), because
none of that is recoverable from the pixels once a file leaves this folder.

Point products are compact by default (Kyle, 5 Oct 2026):
points_attributes.gpkg (EPSG:4326 points, ~20 columns, EGMS units mm and
mm/yr) and points_timeseries.parquet (GeoParquet, one row per point, one
float32 column per date). The 8-14 GB per-point CSV and MintPy's multi-GB
time-series KMZ are written only when `points_full_csv` /
`full_timeseries_kmz` ask for them.

Trust review, 7 Oct 2026: the point table carries the second-check
columns (unwrap_consistency, fails_second_gate) only when the record has
their layer, and its README says only what is there (E19); its relative
columns are written on the FIRST farmland pass, labelled so, until
points_on_report_farmland rewrites them on the client report's farmland
(E14); and a products run moves every earlier Results export to
superseded/ (supersede_results, E17).

Sign convention throughout: positive LOS displacement means motion TOWARD
the satellite (MintPy's convention; see SIGN_NOTE).
"""
import json
import logging
import shutil
import subprocess
import time
import warnings
import zipfile
from datetime import datetime
from pathlib import Path

import math

import numpy as np
from scipy.ndimage import uniform_filter

from .config import PipelineConfig
from . import provenance
from .version import png_metadata          # the build, in every PNG (OUT-01)

log = logging.getLogger(__name__)

# Verified empirically against a known subsidence site: the deforming
# area reads negative in both the HyP3/MintPy and OPERA DISP-S1 paths,
# and matches the sign on vendor deliverables. MintPy's convention is
# positive toward the satellite. OPERA DISP-S1 states the same convention
# on its `displacement` layer, so the DISP backend passes values through
# UNCHANGED (an earlier version negated on read and inverted the result -
# see disp_backend and provenance.sign_convention).
SIGN_NOTE = ("Negative = motion away from the satellite along the radar "
             "line of sight (subsidence on a near-vertical signal). "
             "Positive = motion toward the satellite.")
# The vertical rasters' own convention (review outcome check, 6 Oct 2026:
# vertical_movement_in_per_yr.tif carried the LOS note above, though it is
# vertical and GNSS-anchored when the report is).
VERTICAL_SIGN_NOTE = ("Negative = down (sinking), vertical, GNSS-anchored where stated (RELATIVE_TO). "
                      "Positive = up.")
# Layers that are not motion carry their own note, not the line-of-sight one
# (review 7 Oct 2026: quality_temporal_coherence.tif, a 0-1 score, said
# "Negative = motion away from the satellite").
SIGN_QUALITY = "not applicable (unitless 0-1 quality)"
SIGN_UNCERTAINTY = "unsigned (a 1-sigma uncertainty)"

MM_PER_INCH = 25.4


def _pixel_metres(attrs) -> float:
    """
    Ground pixel size in metres, from the product's own attributes.

    NEVER ASSUME 30 m. MintPy products arrive in two coordinate systems and
    the step size is the only reliable discriminator: X_STEP is 30.0 for a
    UTM grid and 0.00027 for a geographic one. A geocoded topsStack product
    posted at 15 m and a DISP-S1 product at 30 m both flow through the same
    exporters, so a hardcoded constant silently changes what a "1 km" window
    means - which is a deliverable-visible behaviour change with no error
    raised anywhere.
    """
    try:
        dx = abs(float(attrs["X_STEP"]))
        dy = abs(float(attrs.get("Y_STEP", dx)))
    except (KeyError, TypeError, ValueError):
        log.warning("No X_STEP in the product attributes - assuming 30 m posting. "
                    "Any physical window size derived from this is a guess.")
        return 30.0
    if dx < 0.01:                                   # degrees
        try:
            lat = float(attrs.get("Y_FIRST", 0.0))
        except (TypeError, ValueError):
            lat = 0.0
        mx = dx * 111_320.0 * math.cos(math.radians(lat))
        my = dy * 110_900.0
        return float((mx + my) / 2.0)
    return float((dx + dy) / 2.0)


def _window_px(attrs, metres: float = 1000.0) -> int:
    """Odd kernel width covering `metres` on the ground at this product's posting."""
    px = _pixel_metres(attrs)
    k = int(round(metres / max(px, 1e-6)))
    return max(3, k | 1)


def resolution_info(cfg, attrs: dict | None = None) -> dict:
    """
    Posting and resolution of the product, with one plain sentence ("text")
    for headers and tags. resolution.effective_resolution is the single
    source; until it is there this works it out from the product's own
    attributes. The CSV header used to say "Posting 15 m; independent
    resolution cell ~41 m" as a literal - right for the record's 11 x 3 looks
    and wrong for any other looks, for DISP-S1 at 30 m, or for the
    phase-linked route, whose window averages over more ground.
    """
    try:
        from .resolution import effective_resolution
        r = effective_resolution(cfg, attrs)
        if r and r.get("text"):
            return dict(r)
    except (ImportError, AttributeError):
        pass
    except Exception as exc:                                  # noqa: BLE001
        log.warning("resolution.effective_resolution failed (%s) - using the product attributes", exc)
    a = attrs or {}
    posting = round(_pixel_metres(a), 1) if a else None
    out = {"posting_m": posting, "route": str(getattr(cfg, "backend", "") or ""),
           "basis": "product attributes (resolution.py not available)"}
    try:
        rng, az = float(a["RANGE_PIXEL_SIZE"]), float(a["AZIMUTH_PIXEL_SIZE"])
        inc = _look_angle(a, cfg)
        # RANGE_PIXEL_SIZE is the multilooked SLANT-range spacing (2.33 m x 11
        # looks = 25.6 m on the record); on the ground it is / sin(incidence),
        # 41 m at 38.5 deg. Azimuth is already a ground distance.
        gr = rng / math.sin(math.radians(float(inc))) if inc else None
    except (KeyError, TypeError, ValueError):
        gr = az = None
    if gr and posting:
        out["look_cell_m"] = [round(gr, 1), round(az, 1)]
        extra = (" or more (phase linking also averages similar neighbouring ground)"
                 if out["route"].lower() == "miaplpy" else "")
        out["text"] = (f"Points are {posting:.0f} m apart (posting); each value comes from a radar "
                       f"cell about {gr:.0f} m across-track by {az:.0f} m along-track{extra}.")
    elif posting:
        out["text"] = (f"Points are {posting:.0f} m apart (posting); the product does not record "
                       f"the size of the radar cell behind each value.")
    else:
        out["text"] = "Posting and resolution are not recorded in the product."
    return out


def los_up_or_raise(cfg, shape=None, fill: bool = False) -> np.ndarray:
    """
    The per-cell LOS -> vertical factor (the up component of the unit vector
    toward the satellite, = cos(incidence)), or an exception.

    Ten call sites used to read `... if los_up is not None else 0.78`, with no
    log line: if the geometry file went missing or was renamed, every vertical
    number for a new site was converted with Site A's incidence, 5-15%
    off on another track or swath, silently. This is the one helper; it never
    invents a value. `fill=True` fills NaN cells inside the grid with THIS
    map's own median (logged), never with a constant from another site.
    """
    up = geometry_field(cfg, "los_up")
    if up is None:
        raise FileNotFoundError(
            "No LOS geometry on the product grid (geometryGeo.h5 from the DISP-S1 static layers, "
            f"or geo/geo_geometryRadar.h5 from MintPy's geocode step, in {cfg.mintpy_dir}). "
            "Vertical figures need it; none were made rather than assume another site's incidence.")
    up = np.asarray(up, dtype=np.float32)
    if shape is not None and tuple(up.shape) != tuple(shape):
        raise ValueError(f"The LOS geometry is {up.shape} but the product grid is {tuple(shape)} - "
                         "they come from different geocoding runs; re-run the geocode step.")
    fin = np.isfinite(up)
    if not fin.any():
        raise ValueError("The LOS geometry holds no finite value.")
    bad = fin & ((up <= 0.0) | (up > 1.0))
    if bad.any():
        raise ValueError(f"{int(bad.sum())} cells have a LOS up-component outside (0, 1] - "
                         "the geometry layer is not a unit vector toward the satellite.")
    if fill and not fin.all():
        med = float(np.median(up[fin]))
        log.info("LOS up-component missing in %d cells; filled with this grid's median %.4f",
                 int((~fin).sum()), med)
        up = np.where(fin, up, np.float32(med))
    return up


def _unit_label(cfg) -> tuple[str, str, float]:
    """Return (rate label, displacement label, divisor applied to mm)."""
    if (cfg.display_units or "in").lower().startswith("in"):
        return "in/yr", "in", MM_PER_INCH
    return "mm/yr", "mm", 1.0


def reference_label(cfg) -> str:
    """
    The reference as the PRODUCT carries it: MintPy's REF_LAT/REF_LON and the
    geocoded pixel REF_Y/REF_X. MintPy resolves the configured lat/lon in
    radar coordinates, so the geocoded datum can sit a pixel or two from
    where the config says (two rows, ~36 m, on the Site A record);
    headers that printed the config described a pixel the product was not
    referenced to.
    """
    from .significance import reference_pixel
    try:
        _, attrs = _read(_product(cfg, "velocity.h5"), "velocity")
        ref = reference_pixel(attrs, cfg)
    except Exception:           # before MintPy has run there is nothing to read  # noqa: BLE001
        ref = {}
    if not ref:
        return cfg.reference_lalo or "auto"
    cfgd = (cfg.reference_lalo or "").strip()
    # Quote the CENTRE of the geocoded cell that carries the datum, like every
    # other coordinate in the products. REF_LAT/REF_LON are where MintPy
    # resolved the reference in RADAR coordinates: on the site_b record
    # 38.330006,-87.351326 against a cell centre of 38.329889,-87.351256
    # (13 m S, 6 m E) - kept, labelled as the radar position.
    try:
        lon_c, lat_c = _rc_to_lonlat(attrs, ref["row"], ref["col"])
        lat_c, lon_c = float(lat_c), float(lon_c)
    except Exception:                                         # noqa: BLE001
        lat_c, lon_c = ref["lat"], ref["lon"]
    radar = ""
    if "REF_LAT" in attrs and "REF_LON" in attrs:
        radar = f"; radar position {float(attrs['REF_LAT']):.6f},{float(attrs['REF_LON']):.6f}"
    return (f"{lat_c:.6f},{lon_c:.6f} (cell centre of product pixel row {ref['row']}, col {ref['col']}"
            + radar + (f"; configured {cfgd}" if cfgd else "") + ")")


# One copy of each product per export. Every exporter used to open
# velocity.h5 (10x), the temporal coherence (7x) and the full time-series
# cube (5x) for itself; at 275 dates the cube is 6.75 GB and five copies do
# not fit in 24 GB. Arrays come back READ-ONLY so a consumer that writes in
# place fails loudly instead of corrupting the next product.
_CACHE: dict = {}


def _read(path: Path, dataset: str):
    import h5py
    path = Path(path)
    key = (str(path), dataset, path.stat().st_mtime_ns)
    hit = _CACHE.get(key)
    if hit is None:
        with h5py.File(path, "r") as f:
            arr, attrs = f[dataset][:], dict(f.attrs)
        arr.flags.writeable = False
        # keep at most a handful; the cube is the one that matters
        if len(_CACHE) > 8:
            _CACHE.clear()
        _CACHE[key] = hit = (arr, attrs)
    arr, attrs = hit
    return arr, dict(attrs)


_META: dict = {}            # small per-export values (resolution text)


def clear_cache() -> None:
    _CACHE.clear()
    _META.clear()


def _geotransform(attrs: dict):
    """Affine transform from MintPy's geographic attributes."""
    return (float(attrs["X_FIRST"]), float(attrs["X_STEP"]),
            float(attrs["Y_FIRST"]), float(attrs["Y_STEP"]))


def _crs(attrs: dict) -> str:
    """
    Projection of the MintPy product.

    Never assume EPSG:4326: HyP3 geocodes to UTM, so X_FIRST and Y_FIRST
    are metres. Writing those as degrees puts the raster in the Gulf of
    Guinea. ISCE/ARIA products are usually geographic, so read it from the
    metadata rather than guessing.
    """
    epsg = attrs.get("EPSG") or attrs.get("epsg")
    if epsg:
        return f"EPSG:{int(float(epsg))}"

    zone = attrs.get("UTM_ZONE")
    if zone:
        zone = str(zone).strip()
        num = "".join(c for c in zone if c.isdigit())
        north = not zone.upper().endswith("S")
        return f"EPSG:{(32600 if north else 32700) + int(num)}"

    # Fall back on the coordinate magnitudes: degrees are bounded by 180.
    x0 = abs(float(attrs["X_FIRST"]))
    if x0 > 360:
        log.warning("No EPSG or UTM_ZONE in the metadata and X_FIRST=%.1f "
                    "looks projected - the output CRS may be wrong.", x0)
    return "EPSG:4326"


def _rc_to_lonlat(attrs: dict, rows, cols):
    """(lon, lat) in WGS84 of the CENTRE of product cells (rows, cols), for a
    geographic or projected grid. X_FIRST/Y_FIRST are the outer corner of the
    first pixel (MintPy yx2lalo adds 0.5): corner coordinates put a label,
    a KMZ point or an in-the-box test half a cell (7.5 m) north-west of the
    cell it describes. significance.rc_to_lalo is the same rule for
    geographic grids; this one also transforms a UTM grid."""
    x0, dx, y0, dy = _geotransform(attrs)
    x = x0 + (np.asarray(cols, dtype=float) + 0.5) * dx
    y = y0 + (np.asarray(rows, dtype=float) + 0.5) * dy
    crs = _crs(attrs)
    if crs == "EPSG:4326":
        return x, y
    from pyproj import Transformer
    lon, lat = Transformer.from_crs(crs, "EPSG:4326", always_xy=True).transform(x, y)
    return np.asarray(lon), np.asarray(lat)


def _box_mask(attrs: dict, box, shape) -> np.ndarray:
    """Cells whose CENTRE lies inside box = (W, S, E, N) in degrees."""
    ny, nx = shape
    w, s, e, n = (float(v) for v in box)
    if _crs(attrs) == "EPSG:4326":
        _, lat = _rc_to_lonlat(attrs, np.arange(ny), 0)
        lon, _ = _rc_to_lonlat(attrs, 0, np.arange(nx))
        return np.outer((lat >= s) & (lat <= n), (lon >= w) & (lon <= e))
    rr, cc = np.indices(shape)
    lon, lat = _rc_to_lonlat(attrs, rr, cc)
    return (lat >= s) & (lat <= n) & (lon >= w) & (lon <= e)


# ------------------------------------------------------------ GeoTIFF tags

def _route(cfg) -> tuple[str, str]:
    """(backend, 'official' | 'screening' | 'comparison')."""
    from .config import route_tag
    return str(getattr(cfg, "backend", "") or ""), route_tag(cfg)


def _software() -> str:
    try:
        from . import version
        return version.label()
    except Exception:                                         # noqa: BLE001
        return "insar_pipeline (version unrecorded)"


def source_line(cfg, attrs: dict | None = None) -> str:
    """One line naming the data sources and the software, for GeoTIFF tags and
    table headers. The ESA attribution ("Contains modified Copernicus
    Sentinel data <years>") is a condition of the data licence on anything
    derived from Sentinel-1; credits.short_line names every notice the run
    owes (ERA5, the DEM, GNSS) when that module is present."""
    line = ""
    try:
        from . import credits
        line = " ".join(credits.short_line(cfg).split())
    except Exception:                                         # noqa: BLE001
        pass
    if "Copernicus Sentinel" not in line and str(getattr(cfg, "backend", "")) != "nisar":
        a = attrs or {}
        y0, y1 = str(a.get("START_DATE", ""))[:4], str(a.get("END_DATE", ""))[:4]
        years = f" {y0}-{y1}" if y0 and y1 and y0 != y1 else (f" {y0}" if y0 else "")
        line = f"Contains modified Copernicus Sentinel data{years}" + (f" · {line}" if line else "")
    return f"{line}. Processed with {_software()}."


def product_tags(cfg, attrs: dict, quantity: str, units: str, relative_to: str = "",
                 masked: bool = True, vertical: bool | None = None, sign: str | None = None, **extra) -> dict:
    """
    The metadata every delivered raster carries, so a file copied out of the
    folder (as the client data/ set is) still says what it is: LOS or
    vertical, units, which way is negative, which pixel it is relative to,
    over which dates, gated how, from which route, made by what. Until 5 Oct
    2026 the only tag was GDAL's AREA_OR_POINT=Area and the file name was the
    only label. Other writers (client_report.write_rasters) pass the result of
    this to _write_geotiff so the client rasters carry the same set.

    SIGN_CONVENTION: `sign` when given (e.g. an unsigned ratio); else the
    vertical note for a vertical quantity (`vertical`, or a quantity that
    starts with "vertical"), else the line-of-sight note.
    """
    backend, tag = _route(cfg)
    a = attrs or {}
    if vertical is None:
        vertical = str(quantity).strip().lower().startswith("vertical")
    tags = {
        "QUANTITY": quantity,
        "UNITS": units,
        "SIGN_CONVENTION": sign or (VERTICAL_SIGN_NOTE if vertical else SIGN_NOTE),
        "RELATIVE_TO": relative_to or "the reference pixel (absolute)",
        "REFERENCE_PIXEL": reference_label(cfg),
        "REF_DATE": str(a.get("REF_DATE", "")),
        "DATE_RANGE": (f"{a.get('START_DATE', '')} to {a.get('END_DATE', '')}"
                       if a.get("START_DATE") else ""),
        "TEMPORAL_COHERENCE_MIN": (str(getattr(cfg, "min_temporal_coherence", ""))
                                   if masked else "none: every imaged cell is kept"),
        "ROUTE": f"{backend} ({tag})",
        "SOURCE": source_line(cfg, a),
        "TIFFTAG_SOFTWARE": _software(),
        "TIFFTAG_COPYRIGHT": source_line(cfg, a),
        "PROJECT": Path(getattr(cfg, "project_dir", "") or "").name,
    }
    try:
        # once per export: it reads the incidence layer for the ground-range size
        key = ("resolution_text", str(getattr(cfg, "project_dir", "")), a.get("X_STEP"), a.get("RANGE_PIXEL_SIZE"))
        # (a separate dict: an extra _CACHE entry would make _read evict the cube sooner)
        if key not in _META:
            _META[key] = resolution_info(cfg, a).get("text", "")
        tags["POSTING_AND_RESOLUTION"] = _META[key]
    except Exception:                                         # noqa: BLE001
        pass
    tags.update({k.upper(): str(v) for k, v in extra.items()})
    return {k: v for k, v in tags.items() if v not in (None, "")}


def _write_geotiff(path: Path, data: np.ndarray, attrs: dict, nodata=np.nan,
                   tags: dict | None = None, description: str | None = None,
                   units: str | None = None, dtype: str = "float32"):
    """
    Float32 GeoTIFF on the product grid: DEFLATE (floating-point predictor),
    256 x 256 tiles and internal overviews, so a 14 MB raster becomes a few MB
    and opens at any zoom in QGIS without a pyramid pass. `tags` (normally
    product_tags(...)) are written to the dataset; `description` and `units`
    to the band (and TIFFTAG_IMAGEDESCRIPTION). The written file is re-opened
    and checked: a raster that lost its tags is reported, not assumed.
    `dtype` "int16"/"uint8" writes an integer layer (count, flag) with an
    integer nodata and the integer predictor.
    """
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.transform import from_origin
    x0, dx, y0, dy = _geotransform(attrs)
    transform = from_origin(x0, y0, dx, abs(dy))
    crs = _crs(attrs)
    h, w = data.shape
    tags = dict(tags or {})
    units = units or tags.get("UNITS")
    if description:
        tags.setdefault("TIFFTAG_IMAGEDESCRIPTION", description)
    is_float = np.dtype(dtype).kind == "f"
    profile = dict(driver="GTiff", height=h, width=w, count=1, dtype=dtype,
                   crs=crs, transform=transform, nodata=nodata, compress="deflate",
                   predictor=3 if is_float else 2, zlevel=6, BIGTIFF="IF_SAFER")
    if h >= 256 and w >= 256:
        profile.update(tiled=True, blockxsize=256, blockysize=256)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data.astype(dtype), 1)
        if tags:
            dst.update_tags(**{k: str(v) for k, v in tags.items()})
        if description:
            dst.set_band_description(1, description)
        if units:
            dst.units = (units,)
            dst.update_tags(1, UNITS=units)
        levels, f = [], 2
        while min(h, w) // f >= 128:
            levels.append(f)
            f *= 2
        if levels:
            rs = Resampling.average if is_float else Resampling.nearest
            dst.build_overviews(levels, rs)
            dst.update_tags(ns="rio_overview", resampling=rs.name)
    with rasterio.open(path) as chk:
        lost = [k for k in tags if k not in chk.tags()]
        if lost or (units and chk.units[0] != units):
            log.warning("%s was written but its metadata did not survive (%s) - the file "
                        "name is now its only label.", path.name, ", ".join(lost) or "units")
    log.info("Wrote %s (%s)", path, crs)


def load_mask(cfg: PipelineConfig) -> np.ndarray:
    """Reliability mask from temporal coherence at the configured threshold,
    without wooded cells when cfg.mask_wooded (landcover.drop_wooded)."""
    tc, attrs = _read(_product(cfg, "temporalCoherence.h5"), "temporalCoherence")
    mask = tc >= cfg.min_temporal_coherence

    kept = float(np.nanmean(mask))
    log.info("Temporal coherence >= %.2f keeps %.1f%% of pixels",
             cfg.min_temporal_coherence, kept * 100)
    if np.allclose(np.nanmean(tc), 1.0, atol=1e-6):
        log.warning("Temporal coherence is ~1.000 everywhere. That is not a "
                    "good result - it means the network has no closed loops "
                    "(n_neighbors=1) and the metric is measuring nothing. "
                    "The quality product below is not meaningful until the "
                    "network has redundancy.")
    elif kept < 0.05:
        log.warning("Only %.1f%% of pixels pass. Either the data is poor or "
                    "the threshold is too strict for this land cover.", kept * 100)
    from . import landcover
    mask, _ = landcover.drop_wooded(cfg, mask, attrs)
    return mask


# ---------------------------------------------------------------- velocity

def export_velocity(cfg: PipelineConfig, out_dir: Path) -> Path:
    """LOS velocity in mm/yr, masked."""
    vel, attrs = _read(_product(cfg, "velocity.h5"), "velocity")
    mask = load_mask(cfg)

    vel_mm = vel * 1000.0           # MintPy stores m/yr
    vel_mm[~mask] = np.nan
    rate_label, _, divisor = _unit_label(cfg)
    vel_out = vel_mm / divisor

    import h5py
    with h5py.File(timeseries_file(cfg), "r") as f:
        ds = [d.decode() if isinstance(d, bytes) else d for d in f["date"][:]]
    span_days = (datetime.strptime(ds[-1], "%Y%m%d")
                 - datetime.strptime(ds[0], "%Y%m%d")).days
    if span_days < 365:
        log.warning("This stack spans %d days. Velocity is fit as a linear "
                    "rate and extrapolated to a year, so a few mm of "
                    "atmospheric noise becomes tens of mm/yr. Read the "
                    "cumulative displacement instead, and treat the velocity "
                    "as indicative only.", span_days)

    finite = vel_out[np.isfinite(vel_out)]
    if finite.size:
        log.info("LOS velocity %s: min %.2f, max %.2f, median %.2f",
                 rate_label, finite.min(), finite.max(), np.median(finite))
        peak_mm = float(np.nanmax(np.abs(vel_mm[np.isfinite(vel_mm)])))
        if peak_mm > cfg.max_plausible_velocity_mm_yr:
            # a count against the configured value, kept for continuity; the
            # physical limit is the rate DIFFERENCE between neighbouring cells
            # (quality_too_fast_to_measure.tif, QA 'plausibility')
            log.warning(
                "Peak velocity is %.0f mm/yr, above max_plausible_velocity_mm_yr "
                "(%.0f). A broad, smooth bowl can sink that fast and be measured; "
                "a steep edge cannot - see quality_too_fast_to_measure.tif for "
                "the cells whose neighbour-to-neighbour change is beyond what "
                "unwrapping can follow.",
                peak_mm, cfg.max_plausible_velocity_mm_yr)

    suffix = "in_per_year" if divisor != 1.0 else "mm_per_year"
    path = out_dir / f"los_velocity_{suffix}.tif"
    _write_geotiff(path, vel_out, attrs,
                   tags=product_tags(cfg, attrs, "line-of-sight velocity", rate_label,
                                     "the reference pixel (absolute: low-coherence farmland carries the "
                                     "distributed-scatterer bias; see mean_velocity_relative in the points)"),
                   description=f"LOS velocity ({rate_label}) against the reference pixel; cells below "
                               f"temporal coherence {cfg.min_temporal_coherence} are blank")
    return path


# ----------------------------------------------------------------- quality

def export_quality(cfg: PipelineConfig, out_dir: Path) -> Path:
    """
    Temporal coherence: where the inversion fit the observations.

    Deliberately NOT masked at the threshold - it is the layer that shows
    where cells were rejected (site_a_pl: 124,689 cells below 0.4 in the
    raster, min 0.343). Its tags and band description say so; the package
    README must not describe it as "cells below 0.4 are left out".
    """
    tc, attrs = _read(_product(cfg, "temporalCoherence.h5"), "temporalCoherence")
    path = out_dir / "quality_temporal_coherence.tif"
    thr = cfg.min_temporal_coherence
    # which coherence this is: on the phase-linked route MiaplPy's linked
    # (mini-stack) coherence, not MintPy's network coherence - the two are not
    # comparable (time_series stamps temporalCoherenceSource on the layer)
    src = str(attrs.get("temporalCoherenceSource", "") or "MintPy network-inversion temporal coherence")
    _write_geotiff(path, tc, attrs,
                   tags=product_tags(cfg, attrs, "temporal coherence of the time-series inversion",
                                     "unitless (0-1)", "not applicable", masked=False, sign=SIGN_QUALITY,
                                     reporting_threshold=thr, coherence_source=src,
                                     linked_noise_level=attrs.get("linked_noise_level", "")),
                   description=(f"Temporal coherence ({src}), every imaged cell (UNMASKED); the rate and "
                                f"displacement products keep only cells >= {thr}"),
                   units="unitless")
    return path


EXTRA_LAYER_NAMES = ("quality_unwrapping_consistency.tif", "dem_difference_m.tif",
                     "quality_position_uncertainty_m.tif", "disturbed_ground_runs.tif",
                     "quality_too_fast_to_measure.tif")
POINT_PRODUCT_NAMES = ("points_attributes.gpkg", "points_timeseries.parquet")


class PointProductError(RuntimeError):
    """A point product failed to write (export_measurement_points); export_all
    finishes the summary and provenance, then fails the stage with it."""

# in the MintPy folder while a time-series run is part-way (time_series.py)
RUN_INCOMPLETE = "run_incomplete.json"


def _set_aside(out_dir: Path, names, label: str, why: str) -> list[Path]:
    """Move earlier runs' copies of `names` (with their .aux.xml) from out_dir
    to out_dir/superseded/<label>_<stamp>/, so a product this run did not
    write is never packaged as this run's (package skips superseded*). Moved,
    never deleted. Returns where they went."""
    old = [out_dir / n for n in names if (out_dir / n).exists()]
    if not old:
        return []
    dest = out_dir / "superseded" / f"{label}_{datetime.now():%Y%m%d_%H%M%S}"
    dest.mkdir(parents=True, exist_ok=True)
    moved = []
    for p in old:
        for q in (p, p.with_name(p.name + ".aux.xml")):          # with its GIS display settings
            if not q.exists():
                continue
            try:
                q.replace(dest / q.name)
                moved.append(dest / q.name)
            except OSError as exc:
                log.error("%s is from an earlier run and could not be moved aside (%s): delete it before "
                          "packaging", q.name, exc)
    log.warning("Not written by this run (%s), so the earlier copies were moved to %s and are not delivered: %s",
                why, dest, ", ".join(p.name for p in old))
    return moved


RESULTS_DIR = "results"


def supersede_results(out_dir: Path) -> list[Path]:
    """
    Move every Results export (deliverables/results/<area>_<d0>_<d1>/, the
    dashboard's Results screen and `insar results --export`) to
    deliverables/superseded/results_<stamp>/ (E17, trust review 7 Oct 2026).
    A products run rebuilds every product those exports were computed from,
    so an export from before it states figures the new products may not
    give: Site A's 28 Sep Area A2 workbook read -0.223 in/yr
    against the rebuilt report's -0.265, and Site B's 29 Sep whole-area
    one read 'stands out' in three years the report calls no measurable
    movement. package._results_problem refused their workbooks but copied
    their CSV/JSON into technical/; the package skips superseded*, so moved
    exports reach no package. Moved, never deleted; re-export from the
    Results screen after the report is rebuilt. Returns where they went.
    """
    src = Path(out_dir) / RESULTS_DIR
    if not src.is_dir():
        return []
    old = sorted(p for p in src.iterdir() if not p.name.startswith("."))
    if not old:
        return []
    dest = Path(out_dir) / "superseded" / f"results_{datetime.now():%Y%m%d_%H%M%S}"
    dest.mkdir(parents=True, exist_ok=True)
    moved = []
    for p in old:
        try:
            p.replace(dest / p.name)
            moved.append(dest / p.name)
        except OSError as exc:
            log.error("Results export %s is from before this products run and could not be moved aside (%s): "
                      "delete it before packaging", p, exc)
    if moved:
        log.warning("%d Results export(s) from before this products run moved to %s (re-export them once the "
                    "client report is rebuilt): %s", len(moved), dest, ", ".join(p.name for p in moved))
    return moved


# What dem_difference_m.tif is and is not (E1, ledger #1): the words its
# GeoTIFF carries and the package README quotes
DEM_ERROR_CAVEAT = ("It also takes up part of any ground movement during the record (movement that follows the "
                    "orbit baselines reads as height), so it follows the movement rate and is not a map of reshaped "
                    "ground on its own; the report calls ground reshaped only after removing what the rate explains "
                    "(report_insights.dem_change).")

# Sentinel-1's radar wavelength: c / 5.405 GHz. Every route but nisar reads
# Sentinel-1 (config.BACKENDS), so it is the fallback for a product that
# states no wavelength - and only there (M1, 8 Oct 2026).
S1_WAVELENGTH_M = 0.05546576
_NOT_SENTINEL1 = ("nisar", "alos", "saocom", "lsar", "ssar")


def product_wavelength_m(cfg, attrs: dict) -> tuple[float, str]:
    """
    The radar wavelength (m) the product was made with, and where it came
    from. The product's own WAVELENGTH attribute first (MintPy carries it
    from every reader; prep_nisar takes it from the GUNW's centre frequency,
    ~0.24 m at L-band); Sentinel-1 C-band only when the product states none
    and nothing says it is another sensor. A NISAR product without one
    raises: a quarter wavelength at C-band is 4.3 times too small at L-band,
    and the flag built on it would mark ground L-band follows easily.
    """
    for key in ("WAVELENGTH", "wavelength", "radarWavelength"):
        v = attrs.get(key)
        if isinstance(v, bytes):
            v = v.decode()
        try:
            w = float(v)
        except (TypeError, ValueError):
            continue
        if 0.005 < w < 1.0:                       # X-band 3 cm .. P-band 70 cm
            return w, f"the product's {key} attribute"
    backend = str(getattr(cfg, "backend", "") or "").lower()
    platform = attrs.get("PLATFORM", "")
    platform = (platform.decode() if isinstance(platform, bytes) else str(platform or "")).lower()
    if backend == "nisar" or any(s in platform for s in _NOT_SENTINEL1):
        raise ValueError(f"the product states no radar wavelength and is not Sentinel-1 (backend {backend!r}, "
                         f"platform {platform!r}); a C-band value would be wrong")
    return S1_WAVELENGTH_M, "Sentinel-1 C-band (the product states no wavelength; this route reads Sentinel-1)"


def export_extra_layers(cfg: PipelineConfig, out_dir: Path) -> list[Path]:
    """
    The layers the time-series step adds on the product grid
    (time_series.geocode_extra_layers), each delivered when it exists - an
    older project has none until that function is run on it:
      quality_unwrapping_consistency.tif   MintPy's network temporal coherence
                                           (the second quality gate's layer)
      dem_difference_m.tif                 MintPy's elevation-model error estimate
                                           (dem_error): it also takes up part of
                                           the movement, so it is not a map of
                                           reshaped ground on its own (E1)
      quality_position_uncertainty_m.tif   how far that height error moves the
                                           cell on the map: |dem error| / tan(incidence)
      disturbed_ground_runs.tif            longest run of noise-like consecutive
                                           pairs (int16, -1 = no data)
    and, from the velocity itself,
      quality_too_fast_to_measure.tif      1 where the rate differs from a cell
                                           one look away by more than unwrapping
                                           can follow (stack_qc.gradient_limit_mm_yr)
    None of these masks or changes a delivered figure (Kyle, 5 Oct 2026:
    measure, then decide); the QA records their effect.
    """
    import h5py
    written = []
    geo = cfg.mintpy_dir / "geo"

    def layer(path: Path, ds: str):
        if not path.exists():
            return None, None
        with h5py.File(path, "r") as f:
            if ds not in f:
                return None, None
            return f[ds][:], {**dict(f.attrs), **dict(f[ds].attrs)}
    vel, vattrs = _read(_product(cfg, "velocity.h5"), "velocity")
    shape = vel.shape

    um, ua = layer(geo / "geo_temporalCoherence_mintpy.h5", "temporalCoherence")
    if um is not None and um.shape == shape:
        src = str(ua.get("temporalCoherenceSource", "") or "MintPy network-inversion temporal coherence")
        p = out_dir / "quality_unwrapping_consistency.tif"
        _write_geotiff(p, um.astype(np.float32), ua,
                       tags=product_tags(cfg, ua, "unwrapping consistency (MintPy network temporal coherence)",
                                         "unitless (0-1)", "not applicable", masked=False, coherence_source=src,
                                         sign=SIGN_QUALITY, second_gate=str(cfg.min_unwrap_consistency or 0.0)),
                       description=(f"{src}; 1 = every interferogram agrees with the time series. The second "
                                    f"quality gate (min_unwrap_consistency = {cfg.min_unwrap_consistency or 0.0:g}; "
                                    "0 = not applied to any figure). Every imaged cell."),
                       units="unitless")
        written.append(p)

    dem, da = layer(geo / "geo_demErr.h5", "dem")
    if dem is not None and dem.shape == shape:
        p = out_dir / "dem_difference_m.tif"
        _write_geotiff(p, dem.astype(np.float32), da,
                       tags=product_tags(cfg, da, "elevation-model error estimated by MintPy (dem_error)", "m",
                                         "the elevation model (Copernicus DEM GLO-30 unless dem_path is set)",
                                         masked=False, sign="Positive = ground above the elevation model; "
                                                            "negative = ground below it.",
                                         includes_movement=DEM_ERROR_CAVEAT),
                       # E1 (trust review 7 Oct 2026): this said 'reshaped ground
                       # (spoil, pits, reclamation) shows here', but the estimate
                       # follows the movement rate - Site B's 2.2 km2 feature
                       # read 'reshaped' on movement alone
                       description=("MintPy's estimate of the elevation-model error (m). " + DEM_ERROR_CAVEAT),
                       units="m")
        written.append(p)
        inc = geometry_field(cfg, "incidence")
        if inc is not None and inc.shape == shape:
            with np.errstate(invalid="ignore", divide="ignore"):
                shift = np.abs(dem) / np.tan(np.radians(inc))
            p = out_dir / "quality_position_uncertainty_m.tif"
            _write_geotiff(p, shift.astype(np.float32), da,
                           tags=product_tags(cfg, da, "map-position shift from the elevation-model error", "m",
                                             "not applicable", masked=False, sign="unsigned (a distance)",
                                             formula="|dem error| / tan(incidence)"),
                           description=("how far the elevation-model height error moves this cell's measurement "
                                        "across the ground, in metres (|dem error| / tan(incidence)); compare with "
                                        f"the {_pixel_metres(vattrs):.0f} m posting"),
                           units="m")
            written.append(p)

    dis, dsa = layer(geo / "geo_disturbed.h5", "disturbed")
    if dis is not None and str((dsa or {}).get("usable", "yes")) == "no":
        # time_series marked it unable to separate anything: not a deliverable
        log.warning("Disturbed-ground layer not exported: %s", dsa.get("usable_reason", ""))
        (out_dir / "disturbed_ground_runs.tif").unlink(missing_ok=True)   # this stage's own earlier export
        dis = None
    if dis is not None and dis.shape == shape:
        p = out_dir / "disturbed_ground_runs.tif"
        _write_geotiff(p, np.where(np.isfinite(dis), dis, -1).astype(np.int16), dsa, nodata=-1, dtype="int16",
                       tags=product_tags(cfg, dsa, "longest run of consecutive noise-like image pairs",
                                         "pairs", "not applicable", masked=False, sign="not applicable (a count)",
                                         pair_noise_level=dsa.get("pair_noise_level", ""),
                                         pair_coherence_kind=dsa.get("pair_coherence_kind", "")),
                       description=("per cell, the longest run of consecutive nearest-neighbour image pairs whose "
                                    "coherence is below the measured noise level (-1 = no data). A diagnostic of "
                                    "ground that changed or decorrelated for a stretch - not a map of disturbed "
                                    "ground on its own (see quality_assessment.json coverage.disturbed)"),
                       units="pairs")
        written.append(p)

    try:
        from . import stack_qc
        from .significance import pixel_metres
        with h5py.File(timeseries_file(cfg), "r") as f:
            dates = [d.decode() if isinstance(d, bytes) else str(d) for d in f["date"][:]]
        wavelength, wl_source = product_wavelength_m(cfg, vattrs)
        lim = stack_qc.gradient_limit_mm_yr(wavelength, dates)
        res = resolution_info(cfg, vattrs)
        ew, ns = (res.get("look_cell_m") or [41.0, 42.0])[:2]
        py, px = pixel_metres(vattrs)
        step = (max(1, int(round(float(ns) / py))), max(1, int(round(float(ew) / px))))
        good = load_mask(cfg) & np.isfinite(vel)
        bad = stack_qc.rate_gradient_flags(vel * 1000.0, step, lim["limit_mm_yr"], valid=good)
        flag = np.where(good, bad.astype(np.uint8), 255).astype(np.uint8)
        p = out_dir / "quality_too_fast_to_measure.tif"
        _write_geotiff(p, flag, vattrs, nodata=255, dtype="uint8",
                       tags=product_tags(cfg, vattrs, "cells changing too fast across the ground to measure",
                                         "flag (1 = not measurable, 0 = measurable)", "not applicable",
                                         sign="not applicable (a flag)", gradient_limit_mm_yr=lim["limit_mm_yr"],
                                         median_revisit_days=lim["median_gap_days"],
                                         neighbour_step_cells=f"{step[0]},{step[1]}",
                                         wavelength_m=f"{wavelength:.6g}", wavelength_source=wl_source,
                                         quarter_wavelength_mm=lim["quarter_wavelength_mm"],
                                         lower_bound=("yes: built from the unwrapped velocity, so a cell whose "
                                                      "unwrapping already aliased (a whole fringe lost) reads "
                                                      "smooth here and is not flagged; quality_aliasing_risk.tif, "
                                                      "where written, is made before unwrapping")),
                       description=(f"1 where the rate differs from a cell one look away by more than "
                                    f"{lim['limit_mm_yr']:.0f} mm/yr (a quarter wavelength, "
                                    f"{lim['quarter_wavelength_mm']:.1f} mm, per {lim['median_gap_days']:.0f}-day "
                                    "revisit): too fast a change across the ground for this satellite, so the value "
                                    "there is not a measurement. 255 = below the coherence threshold. A lower bound: "
                                    "it is built from the unwrapped velocity, so cells that already aliased are "
                                    "not marked."),
                       units="flag")
        written.append(p)
        n_bad = int(bad.sum())
        if n_bad:
            log.warning("%d cells change faster across the ground than unwrapping can follow (%.0f mm/yr between "
                        "cells one look apart): marked in %s; their rates are not measurements", n_bad,
                        lim["limit_mm_yr"], p.name)
    except Exception as exc:                                  # noqa: BLE001
        log.warning("quality_too_fast_to_measure.tif not written: %s", exc)
    # a layer not written this time leaves the previous run's copy behind, from
    # another inversion and maybe another grid: move it aside, never ship it
    _set_aside(out_dir, [n for n in EXTRA_LAYER_NAMES if n not in {p.name for p in written}], "layers",
               "its source layer is missing, on another grid or failed")
    missing = [n for n, f in (("unwrapping consistency", geo / "geo_temporalCoherence_mintpy.h5"),
                              ("DEM error", geo / "geo_demErr.h5"), ("disturbed ground", geo / "geo_disturbed.h5"))
               if not f.exists()]
    if missing:
        log.info("Not on disk, so not delivered: %s (time_series.geocode_extra_layers(cfg) adds them to a "
                 "finished project)", ", ".join(missing))
    return written


# ------------------------------------------------------------- time series

def _product(cfg: PipelineConfig, name: str) -> Path:
    """
    Locate a MintPy product, preferring the geocoded copy.

    WHY THIS IS NEEDED. The DISP-S1 and HyP3 backends hand MintPy data that is
    already in map coordinates, so it writes velocity.h5 at the mintpy root.
    topsStack does not geocode - its interferogram workflow ends at unwrapping
    and everything downstream is in radar coordinates - so MintPy's own
    geocode step runs last and writes geo/geo_velocity.h5 instead.

    Every exporter here needs map coordinates to write a GeoTIFF, so the
    geocoded copy wins whenever both exist. Falling back to the flat name
    keeps every existing backend byte-identical.
    """
    geo = cfg.mintpy_dir / "geo" / f"geo_{name}"
    if geo.exists():
        return geo
    return cfg.mintpy_dir / name


def timeseries_file(cfg: PipelineConfig) -> Path:
    """
    The most corrected time series MintPy produced, geocoded if it exists.

    MintPy chains corrections into new FILES rather than editing in place, and
    the name accumulates a token per step:

        timeseries.h5
          -> timeseries_SET.h5            solid earth tide
          -> timeseries_ERA5.h5           troposphere
          -> timeseries_ERA5_ramp.h5      deramp
          -> timeseries_ERA5_ramp_demErr.h5   DEM error

    velocity.h5 is fitted to whichever came LAST, so reading an earlier one
    gives displacements that do not match the velocity map.

    This used to be a hardcoded list of names, which broke the moment
    deramping was enabled: the list knew nothing about the "_ramp" token, so
    it silently fell back to timeseries_ERA5.h5 - less corrected, and in radar
    coordinates while the masks were geocoded, which surfaced only as an
    IndexError about mismatched array dimensions. Scoring the tokens instead
    means a new correction step cannot quietly demote the product again.

    Since 5 Oct 2026 MintPy's own run record decides: time_series writes
    mintpy/run_products.json after a successful run, naming the series the
    velocity was fitted to. That file is used when present and only if it
    agrees with the velocity product (same reference pixel and reference
    date, the velocity's start and end dates in the series, same grid);
    the token ranking below is the logged fallback for older runs. Whatever
    file is chosen is checked the same way, and a series that does not
    match the velocity raises rather than producing displacements that
    disagree with the rate map.
    """
    rp = run_products(cfg)
    if rp:
        for key in ("timeseries_geo", "timeseries_radar"):
            p = rp.get(key)
            if not p or not Path(p).exists():
                continue
            why = _mismatch_with_velocity(cfg, Path(p))
            if why is None:
                log.info("Using %s (named in run_products.json as %s)", Path(p).name, key)
                return Path(p)
            log.error("run_products.json names %s as %s, but %s - falling back to the "
                      "correction-token ranking.", p, key, why)
        log.warning("run_products.json names no usable time series - choosing by the "
                    "correction tokens in the file names instead.")
    else:
        log.info("No mintpy/run_products.json (a run made before 5 Oct 2026) - choosing the "
                 "time series by the correction tokens in the file names.")
    best = _timeseries_by_tokens(cfg)
    why = _mismatch_with_velocity(cfg, best)
    if why is not None:
        raise ValueError(f"The time series {best} does not match the velocity product: {why}. "
                         "Re-run the MintPy geocode step, or the time series, so the two agree.")
    return best


def run_products(cfg) -> dict | None:
    """mintpy/run_products.json (written by time_series after a successful
    MintPy run), or None for runs made before it existed."""
    p = Path(cfg.mintpy_dir) / "run_products.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except (ValueError, OSError) as exc:
        log.warning("Could not read %s (%s)", p, exc)
        return None


def _mismatch_with_velocity(cfg, ts_path: Path) -> str | None:
    """None when the series fits the velocity product, else a plain reason.
    Nothing to compare against (no velocity yet) is not a mismatch."""
    vel_path = _product(cfg, "velocity.h5")
    if not vel_path.exists():
        return None
    import h5py
    try:
        with h5py.File(vel_path, "r") as f:
            va = dict(f.attrs)
            vshape = f["velocity"].shape if "velocity" in f else None
        with h5py.File(ts_path, "r") as f:
            ta = dict(f.attrs)
            tshape = f["timeseries"].shape[1:] if "timeseries" in f else None
            dates = {d.decode() if isinstance(d, bytes) else str(d) for d in f["date"][:]} \
                if "date" in f else set()
    except (OSError, KeyError) as exc:
        return f"it could not be read ({exc})"

    def s(v):
        return v.decode() if isinstance(v, bytes) else str(v)

    for k in ("REF_Y", "REF_X"):
        if k in va and k in ta and int(float(s(va[k]))) != int(float(s(ta[k]))):
            return f"its {k} is {s(ta[k])} where the velocity's is {s(va[k])}"
    if va.get("REF_DATE") and ta.get("REF_DATE") and s(va["REF_DATE"]) != s(ta["REF_DATE"]):
        return f"its reference date is {s(ta['REF_DATE'])} where the velocity's is {s(va['REF_DATE'])}"
    if dates:
        for k in ("START_DATE", "END_DATE"):
            if va.get(k) and s(va[k]) not in dates:
                return f"the velocity's {k} {s(va[k])} is not one of its dates"
    if vshape is not None and tshape is not None and tuple(vshape) != tuple(tshape):
        return f"it is on a {tuple(tshape)} grid where the velocity is {tuple(vshape)}"
    return None


def _timeseries_by_tokens(cfg: PipelineConfig) -> Path:
    """The legacy choice: the most corrected series by its file-name tokens."""
    # Each token MintPy can append, weighted so that more corrections win and
    # the ordering between them never has to be guessed.
    from .corrections import WEATHER_TOKENS
    tokens = ("SET",) + WEATHER_TOKENS + ("ramp", "demErr")

    def score(path: Path) -> tuple:
        name = path.stem
        n = sum(1 for tok in tokens if f"_{tok}" in name)
        return (n, len(name))

    candidates = []
    for d, geocoded in ((cfg.mintpy_dir / "geo", True), (cfg.mintpy_dir, False)):
        if not d.exists():
            continue
        for f in d.glob("*timeseries*.h5"):
            # Residual files are diagnostics - the fit leftovers, not displacement.
            if "Residual" in f.name:
                continue
            candidates.append((geocoded, *score(f), f))
    if not candidates:
        raise FileNotFoundError(f"No timeseries file in {cfg.mintpy_dir}")

    candidates.sort(reverse=True)
    geocoded, n_corr, _, best = candidates[0]

    if n_corr == 0:
        if getattr(cfg, "backend", "local") in ("disp", "nisar"):
            pass          # these arrive already corrected from the producer
        else:
            log.warning("Only the uncorrected %s is present - atmospheric, ramp "
                        "and DEM-error corrections did not run, so these "
                        "displacements are noisier than the velocity map "
                        "implies.", best.name)
    else:
        log.info("Using %s (%d correction step%s applied%s)", best.name, n_corr,
                 "" if n_corr == 1 else "s", ", geocoded" if geocoded else "")
    if not geocoded and (cfg.mintpy_dir / "geo").exists():
        log.warning("A geo/ directory exists but holds no time series, so the "
                    "radar-coordinate %s is being used. Rasters written from it "
                    "will not line up with the geocoded velocity.", best.name)
    return best


class _Cube:
    """
    The geocoded time-series cube left on disk and read as it is used
    (generality-12). Loading it whole held 3.8 GB (Site B, 270 dates) to
    6.4 GB (Site A) plus derived copies, which a 16 GB laptop running WSL
    with its default ~8 GB cannot hold. Supports what the exports do:
      cube[i]                    one epoch (2-D)
      cube[:, r, c]              one cell's series
      cube[:, rows, cols]        many cells (arrays), read in chunk-aligned
                                 row bands of at most BAND_BYTES
      epochs(cube)               (i, epoch) in chunk-aligned blocks
    Values are float32 metres, exactly what h5py returns for the dataset.
    """
    BAND_BYTES = 256e6

    def __init__(self, path: Path, dataset: str = "timeseries"):
        import h5py
        self.path, self.dataset = Path(path), dataset
        with h5py.File(self.path, "r") as f:
            d = f[dataset]
            self.shape, self.dtype, self.chunks = tuple(d.shape), d.dtype, d.chunks
        self.ndim = len(self.shape)

    def __len__(self):
        return self.shape[0]

    def __array__(self, *a, **k):
        # numpy would otherwise build the whole cube through __getitem__
        raise TypeError("the time-series cube is read as it is used (_Cube): stream it with _epochs(), "
                        "take cells with cube[:, rows, cols], or use _anchors / _valid_fraction")

    def __getitem__(self, key):
        import h5py
        with h5py.File(self.path, "r") as f:
            d = f[self.dataset]
            if (isinstance(key, tuple) and len(key) == 3 and key[0] == slice(None)
                    and np.ndim(key[1]) == 1 and np.ndim(key[2]) == 1):
                return self._points(d, np.asarray(key[1], np.intp), np.asarray(key[2], np.intp))
            return d[key]

    def _points(self, d, rows, cols):
        """(n_dates, k) for k cells, reading whole row bands (h5py cannot take
        two index arrays) aligned on the chunk rows."""
        n, ny, nx = self.shape
        out = np.empty((n, rows.size), np.float32)
        if not rows.size:
            return out
        crow = int(self.chunks[1]) if self.chunks else 1
        band = max(crow, int(self.BAND_BYTES // max(1, n * nx * 4)) // crow * crow)
        order = np.argsort(rows, kind="stable")
        rs = rows[order]
        i = 0
        while i < rs.size:
            r0 = int(rs[i]) // crow * crow
            j = int(np.searchsorted(rs, min(ny, r0 + band), side="left"))
            # only the chunk rows these cells need: a 126-point cluster spans a
            # few rows, not a whole band
            r1 = min(ny, r0 + band, -(-(int(rs[j - 1]) + 1) // crow) * crow)
            sel = order[i:j]
            blk = d[:, r0:r1, :]
            out[:, sel] = blk[:, rows[sel] - r0, cols[sel]]
            i = j
        return out


def _epochs(ts):
    """(i, epoch) for an in-memory cube or a _Cube, chunk-aligned on disk
    (significance.iter_epochs: ts[i] in a loop decodes a chunk 9 dates deep
    to return one)."""
    if isinstance(ts, _Cube):
        import h5py
        from .significance import iter_epochs
        with h5py.File(ts.path, "r") as f:
            yield from iter_epochs(f[ts.dataset])
    else:
        for i in range(ts.shape[0]):
            yield i, ts[i]


def _dates_and_series(cfg: PipelineConfig):
    """(dates, the cube as a _Cube, attrs): the cube is read as each export
    uses it, never held whole."""
    path = timeseries_file(cfg)
    ts = _Cube(path, "timeseries")              # (n_dates, rows, cols), metres
    import h5py
    with h5py.File(path, "r") as f:
        attrs = dict(f.attrs)
        raw = f["date"][:]
    dates = [d.decode() if isinstance(d, bytes) else str(d) for d in raw]
    return dates, ts, attrs


def geometry_field(cfg: PipelineConfig, name: str):
    """
    A per-pixel geometry layer on the product grid, or None.

    Two backends write two files: DISP-S1's static layers become
    geometryGeo.h5 (incidence_deg, los_east/north/up, height) and MintPy's
    geocode step writes geo/geo_geometryRadar.h5 for the topsStack path
    (incidenceAngle, azimuthAngle, height, ...). Everything that needs
    incidence, LOS unit vectors, height or the imaged footprint goes
    through here so the stack path stops falling back to a DISP-only file
    that does not exist.

    Names accepted: incidence (deg), azimuth (deg), height (m), los_east,
    los_north, los_up (unit vector, positive toward the satellite).
    """
    disp = _product(cfg, "geometryGeo.h5")
    tops = _product(cfg, "geometryRadar.h5")
    keys_disp = {"incidence": "incidence_deg", "height": "height",
                 "los_east": "los_east", "los_north": "los_north", "los_up": "los_up"}
    keys_tops = {"incidence": "incidenceAngle", "azimuth": "azimuthAngle",
                 "height": "height"}
    candidates = [(disp, keys_disp), (tops, keys_tops)]
    # The NISAR GUNW route (8 Oct 2026): prep_nisar loads a geocoded stack, so
    # MintPy's geometry stays in mintpy/inputs/geometryGeo.h5 with MintPy's own
    # names (incidenceAngle, azimuthAngle with the route's corrected encoding,
    # nisar_backend.los_encoding_for_prep_nisar). Read only on that route and
    # only when neither file above exists, so every other route reads exactly
    # what it did.
    if str(getattr(cfg, "backend", "")).lower() == "nisar" and not disp.exists() and not tops.exists():
        candidates.append((cfg.mintpy_dir / "inputs" / "geometryGeo.h5", keys_tops))
    import h5py
    for path, keys in candidates:
        if not path.exists():
            continue
        with h5py.File(path, "r") as f:
            if name in keys and keys[name] in f:
                arr = f[keys[name]][:].astype(np.float32)
                if name == "incidence":
                    arr[arr == 0] = np.nan      # fill value outside the swath
                return arr
            if keys is keys_tops and name in ("los_east", "los_north", "los_up") \
                    and "incidenceAngle" in f and "azimuthAngle" in f:
                inc = np.radians(f["incidenceAngle"][:].astype(np.float32))
                az = np.radians(f["azimuthAngle"][:].astype(np.float32))
                inc[inc == 0] = np.nan
                # MintPy convention: azimuthAngle measured from north,
                # anti-clockwise, of the look direction's ground projection;
                # unit vector points from ground TOWARD the satellite.
                if name == "los_up":
                    return np.cos(inc)
                if name == "los_east":
                    return np.sin(inc) * np.sin(az) * -1.0
                return np.sin(inc) * np.cos(az)
    return None


def observable_mask(cfg: PipelineConfig, shape=None):
    """
    Pixels the sensor actually imaged, from the geometry rather than from
    the result. DISP-S1 supplies maskObservable.h5; topsStack's footprint is
    wherever the geocoded incidence angle is finite - exactly the velocity
    footprint (verified pixel for pixel). Without this the
    coverage denominator was the whole lat/lon bounding rectangle, 27% of
    which lies outside the swath.
    """
    obs = _product(cfg, "maskObservable.h5")
    if obs.exists():
        m, _ = _read(obs, "mask")
        m = m.astype(bool)
        if shape is None or m.shape == shape:
            return m, "maskObservable.h5"
    inc = geometry_field(cfg, "incidence")
    if inc is not None and (shape is None or inc.shape == shape):
        return np.isfinite(inc), "geometryRadar incidenceAngle footprint"
    return None, None


def _anchors(ts: np.ndarray):
    """
    Each pixel's first and last FINITE value, and where it has any at all.

    Anchoring a displacement on `ts[0]` and `ts[-1]` assumes every pixel is
    valid on those two particular dates. In a gappy stack - DISP-S1 masks
    unwrapping failures per epoch, and over farmland that is most of the
    grid on some dates - that assumption silently turns a pixel with 250
    good epochs into NaN because it happened to be masked on the first one.
    That is what reduced the selected-points graph to "final average nan"
    while the velocity map beside it was fine.

    Anchoring on each pixel's own first and last valid epoch keeps the
    pixel. The cost is that pixels whose record starts late are measured
    over a shorter interval than the file name claims, so the caller is
    given `has` and should say how many pixels that affects.
    """
    if isinstance(ts, _Cube):
        return _anchors_streamed(ts)
    fin = np.isfinite(ts)
    has = fin.any(axis=0)
    first_i = np.argmax(fin, axis=0)
    last_i = ts.shape[0] - 1 - np.argmax(fin[::-1], axis=0)
    first = np.where(has, np.take_along_axis(ts, first_i[None], 0)[0], np.nan)
    last = np.where(has, np.take_along_axis(ts, last_i[None], 0)[0], np.nan)
    return first, last, has, first_i, last_i


def _anchors_streamed(ts: "_Cube"):
    """_anchors in one pass over the epochs on disk, the same values without
    the cube in memory; kept for the export (four products use it)."""
    key = ("anchors", str(ts.path), ts.path.stat().st_mtime_ns)
    if key in _META:
        return _META[key]
    _, ny, nx = ts.shape
    first = np.full((ny, nx), np.nan, np.float32)
    last = np.full((ny, nx), np.nan, np.float32)
    first_i = np.zeros((ny, nx), np.intp)
    last_i = np.full((ny, nx), ts.shape[0] - 1, np.intp)    # the in-memory form's value for an empty cell
    has = np.zeros((ny, nx), bool)
    count = np.zeros((ny, nx), np.int32)
    for i, ep in _epochs(ts):
        fin = np.isfinite(ep)
        new = fin & ~has
        first[new] = ep[new]
        first_i[new] = i
        last[fin] = ep[fin]
        last_i[fin] = i
        has |= fin
        count += fin
    for a in (first, last, first_i, last_i, has, count):
        a.flags.writeable = False
    _META[key] = (first, last, has, first_i, last_i)
    _META[("valid_count",) + key[1:]] = count
    return _META[key]


def _valid_fraction(ts) -> np.ndarray:
    """Each cell's share of epochs with a finite value (one streamed pass on a
    _Cube, shared with _anchors)."""
    if not isinstance(ts, _Cube):
        return np.isfinite(ts).mean(axis=0)
    _anchors_streamed(ts)
    count = _META[("valid_count", str(ts.path), ts.path.stat().st_mtime_ns)]
    return count / float(ts.shape[0])


def export_cumulative_displacement(cfg: PipelineConfig, out_dir: Path) -> Path:
    """Cumulative LOS displacement raster for the last date, plus a CSV of
    the area-average series."""
    dates, ts, attrs = _dates_and_series(cfg)
    mask = load_mask(cfg)

    _, disp_label, divisor = _unit_label(cfg)
    # Cumulative displacement is the CHANGE across the period, so it must
    # be differenced against the first epoch. MintPy's time series is
    # relative to its reference date, and after the reference_date and
    # deramp steps the first epoch is not necessarily zero - taking the
    # last value alone reports a position, not a displacement, and it will
    # not agree with the velocity map.
    first, last, has, first_i, last_i = _anchors(ts)
    cumulative = (last - first) * 1000.0 / divisor
    cumulative[~mask] = np.nan
    late = int(np.sum(mask & has & (first_i > 0)))
    short = int(np.sum(mask & has & (last_i < len(dates) - 1)))
    if late or short:
        log.info("Cumulative displacement is measured over each pixel's own "
                 "valid record: %d reported pixels start after %s and %d end "
                 "before %s. State that where the figure is quoted.",
                 late, dates[0], short, dates[-1])
    unit_tag = "in" if divisor != 1.0 else "mm"
    path = out_dir / (f"los_cumulative_displacement_{unit_tag}_"
                      f"{dates[0]}_{dates[-1]}.tif")
    _write_geotiff(path, cumulative, attrs,
                   tags=product_tags(cfg, attrs, "line-of-sight cumulative displacement", disp_label,
                                     "the reference pixel (absolute)",
                                     measured_over="each cell's first to last valid epoch",
                                     cells_starting_late=late, cells_ending_early=short),
                   description=f"LOS displacement ({disp_label}) from each cell's first to last valid "
                               f"epoch, {dates[0]} to {dates[-1]}, against the reference pixel")

    # Area-average series - a quick sanity read on the whole AOI.
    csv = out_dir / "los_cumulative_displacement_area_mean.csv"
    with open(csv, "w") as f:
        f.write(f"# {SIGN_NOTE}\n")
        f.write(f"# reference point: {reference_label(cfg)}\n")
        f.write(f"# temporal coherence threshold: {cfg.min_temporal_coherence}\n")
        f.write(f"# software: {_software()}\n")       # (16 Oct 2026, readiness review OUT-01)
        f.write(f"date,mean_los_displacement_{disp_label},"
                f"std_{disp_label},n_pixels\n")
        for i, ep in _epochs(ts):
            vals = (ep - first)[mask] * 1000.0 / divisor
            vals = vals[np.isfinite(vals)]
            if vals.size:
                f.write(f"{dates[i]},{vals.mean():.3f},{vals.std():.3f},{vals.size}\n")
    log.info("Wrote %s", csv)
    return path


def plot_displacement_series(cfg: PipelineConfig, out_dir: Path,
                             points: list[tuple[float, float]] | None = None) -> Path | None:
    """
    Cumulative LOS displacement graphs.

    With `points` (lat, lon) it plots those locations; otherwise it plots
    the area mean plus the fastest-moving reliable pixel, which is usually
    where someone's attention should go first.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dates, ts, attrs = _dates_and_series(cfg)
    mask = load_mask(cfg)
    dt = [datetime.strptime(d, "%Y%m%d") for d in dates]
    base, _, _, _, _ = _anchors(ts)

    def rowcol(lat, lon):
        # the cell CONTAINING the point (MintPy's lalo2yx), not the nearest
        # corner: round() on corner coordinates picked the next cell half the time
        return _lalo_to_rc(attrs, lat, lon)

    fig, ax = plt.subplots(figsize=(9, 5))

    if points:
        for lat, lon in points:
            r, c = rowcol(lat, lon)
            if not (0 <= r < ts.shape[1] and 0 <= c < ts.shape[2]):
                log.warning("Point %.4f,%.4f is outside the AOI - skipped", lat, lon)
                continue
            series = (ts[:, r, c] - base[r, c]) * 1000.0
            label = f"{lat:.4f}, {lon:.4f}"
            if not mask[r, c]:
                label += " (below coherence threshold)"
            ax.plot(dt, series, marker="o", ms=3, label=label)
    else:
        area = np.array([np.nanmean((ep - base)[mask]) for _, ep in _epochs(ts)]) * 1000.0
        ax.plot(dt, area, marker="o", ms=3, color="#2E63C8", label="AOI mean")

        vel, _ = _read(_product(cfg, "velocity.h5"), "velocity")
        v = np.where(mask, np.abs(vel), np.nan)
        if np.any(np.isfinite(v)):
            r, c = np.unravel_index(np.nanargmax(v), v.shape)
            lon, lat = (float(z) for z in _rc_to_lonlat(attrs, r, c))     # the cell centre
            ax.plot(dt, (ts[:, r, c] - base[r, c]) * 1000.0,
                    marker="o", ms=3, color="#C42A66",
                    label=f"fastest pixel ({lat:.4f}, {lon:.4f})")

    ax.set_ylabel("Cumulative LOS displacement (mm)")
    ax.set_xlabel("Date")
    ax.set_title("Line-of-sight cumulative displacement")
    ax.grid(alpha=.3)
    ax.legend(fontsize=8)
    fig.text(0.01, 0.01,
             f"{SIGN_NOTE}  Reference: {reference_label(cfg)}. "
             f"Temporal coherence >= {cfg.min_temporal_coherence}.",
             fontsize=7, color="#697083")
    fig.tight_layout(rect=[0, 0.04, 1, 1])

    path = out_dir / "los_cumulative_displacement.png"
    fig.savefig(path, dpi=160, metadata=png_metadata())
    plt.close(fig)
    log.info("Wrote %s", path)
    return path


def _lalo_to_rc(attrs, lat, lon):
    """Row/col of the cell containing a lat/lon, from the MintPy attrs (any CRS
    _crs recognises - a UTM_ZONE-only product used to be read as degrees)."""
    from pyproj import Transformer
    crs = _crs(attrs)
    x0, y0 = float(attrs["X_FIRST"]), float(attrs["Y_FIRST"])
    dx, dy = float(attrs["X_STEP"]), float(attrs["Y_STEP"])
    if crs != "EPSG:4326":
        x, y = Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform(lon, lat)
    else:
        x, y = lon, lat
    # MintPy's convention (coord.lalo2yx): floor, with X/Y_FIRST the outer corner
    return int(np.floor((y - y0) / dy + 0.01)), int(np.floor((x - x0) / dx + 0.01))


def plot_selected_points(cfg: PipelineConfig, out_dir: Path,
                         n_points: int | None = None) -> Path:
    """
    Cumulative displacement over a selected cluster, vendor style.

    Plots each selected point faintly, their average in bold, and the
    spread as an envelope. This is what a monitoring report shows: an
    AOI-wide mean is dominated by stable ground and understates the
    movement, while a single pixel is not defensible on its own.

    The points are a COMPACT CLUSTER on the deforming feature, which is
    what a vendor plot shows - not the n fastest pixels in the scene. See
    the note on the selection below; the two rules differ by 3x.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_points = n_points or cfg.graph_n_points
    dates, ts, attrs = _dates_and_series(cfg)
    mask = load_mask(cfg)
    vel, _ = _read(_product(cfg, "velocity.h5"), "velocity")
    _, disp_label, divisor = _unit_label(cfg)
    dt = [datetime.strptime(d, "%Y%m%d") for d in dates]

    # A COMPACT CLUSTER, not the n fastest pixels anywhere.
    #
    # This is a vendor-comparison graph, so the selection rule has to be
    # the vendor's. 3vGeomatics average 126 SELECTED points inside the
    # deforming area - a cluster on one feature. Taking the n fastest
    # pixels across the whole scene instead pools the single worst pixel
    # from every unrelated feature, which is not a place and does not have
    # a displacement history. Over the 50 km Site A run the two rules
    # differ by nearly 3x on the same data:
    #
    #     126 fastest anywhere in the AOI   -43.8 mm/yr   -14.43 in
    #     1.2 km cluster at the bowl        -14.6 mm/yr    -4.20 in
    #     vendor, 126 selected points       -13.4 mm/yr    -5.0  in
    #
    # The cluster rule reproduces the vendor; the fastest-n rule overstates
    # by 3x and is not comparable to anything they publish.
    #
    # The cluster is centred on the deepest point of the rate field after
    # smoothing, so it lands on the feature rather than on one noisy pixel,
    # and is then grown to n_points by nearest distance from that centre.
    rate = np.where(mask & np.isfinite(vel), vel, np.nan)
    if not np.isfinite(rate).any():
        log.warning("No reliable pixels to plot")
        return None

    # Only pixels present through most of the record may join: a tight
    # cluster decorrelates together, and if every member is blank on the
    # same date the average for that date is nan.
    valid_frac = _valid_fraction(ts)
    eligible = np.isfinite(rate) & (valid_frac >= cfg.graph_min_valid_fraction)
    if eligible.sum() < n_points:
        log.warning("Only %d pixels meet graph_min_valid_fraction=%.2f; "
                    "relaxing for this graph", int(eligible.sum()),
                    cfg.graph_min_valid_fraction)
        eligible = np.isfinite(rate)

    site_label = ""
    sites = []
    try:
        from .panel_history import parse_focus_sites
        sites = parse_focus_sites(cfg)
    except Exception as exc:                                   # noqa: BLE001
        log.warning("focus sites not parsed: %s", exc)
    if cfg.graph_center_lalo.strip():
        lat_c, lon_c = (float(v) for v in cfg.graph_center_lalo.split(","))
        cr, cc = _lalo_to_rc(attrs, lat_c, lon_c)
        log.info("Selected-points cluster pinned to %.5f, %.5f", lat_c, lon_c)
    elif sites:
        # The project area comes first: the cluster sits at its centre and is
        # drawn only from its cells, so the graph describes the project area.
        name, s_, w, n_, e, _gj = sites[0]
        lat_c, lon_c = (s_ + n_) / 2, (w + e) / 2
        cr, cc = _lalo_to_rc(attrs, lat_c, lon_c)
        eligible &= _box_mask(attrs, (w, s_, e, n_), eligible.shape)      # cell centres
        site_label = name
        log.info("Selected-points cluster centred on project area %r at %.5f, %.5f", name, lat_c, lon_c)
    else:
        # Inside the AOI only: the product grid is the bounding rectangle of a
        # rotated frame and its margins are the noisiest ground in the scene.
        # Same rule as quality_assessment's "strongest subsidence in the data".
        eligible &= _box_mask(attrs, cfg.bbox, eligible.shape)           # cell centres
        k = _window_px(attrs, 1000.0)           # 1 km at THIS product's posting
        ok = eligible.astype(float)
        num = uniform_filter(np.where(eligible, np.nan_to_num(rate), 0.0), k)
        den = uniform_filter(ok, k)
        smooth = np.where(den > 0.3, num / np.maximum(den, 1e-9), np.nan)
        if not np.isfinite(smooth).any():
            smooth = np.where(eligible, rate, np.nan)
        cr, cc = np.unravel_index(np.nanargmin(smooth), smooth.shape)

    rr, ccc = np.indices(rate.shape)
    dist = np.hypot(rr - cr, ccc - cc)
    dist[~eligible] = np.inf
    order = np.argsort(dist.ravel())
    picks = [np.unravel_index(i, rate.shape) for i in order[:n_points]]
    picks = [(r, c) for r, c in picks if np.isfinite(rate[r, c])]
    if picks:
        span_px = float(np.max(dist[tuple(np.array(picks).T)]))
        log.info("Selected-points cluster: %d points within %.0f m of "
                 "%d,%d (mean rate %.2f mm/yr)", len(picks),
                 span_px * _pixel_metres(attrs),
                 cr, cc, float(np.nanmean(rate[tuple(np.array(picks).T)])) * 1000)
    if not picks:
        log.warning("No reliable pixels to plot")
        return None

    # Rebase each pixel on its own first valid epoch, not on ts[0] - see
    # _anchors(). Then aggregate nan-aware: one masked epoch in one of the
    # 126 picked pixels would otherwise blank the average for that date,
    # which is exactly how this graph came to report "final average nan".
    first, _, _, _, _ = _anchors(ts)
    pr, pc = (np.array(v, np.intp) for v in zip(*picks))
    series = (ts[:, pr, pc] - first[pr, pc]).T          # (points, dates), float32; one band read
    series = series * 1000.0 / divisor
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN dates
        mean = np.nanmean(series, axis=0)
        lo, hi = np.nanmin(series, axis=0), np.nanmax(series, axis=0)
    n_per_date = np.isfinite(series).sum(axis=0)
    if n_per_date.min() < len(picks):
        log.info("Selected-points average is over %d-%d of the %d points "
                 "depending on the date; dates where a point is masked are "
                 "averaged over the rest.",
                 int(n_per_date.min()), int(n_per_date.max()), len(picks))
    if n_per_date[-1] == 0:
        log.warning("No selected point has a valid value on the final date - "
                    "the last average is undefined.")

    fig, ax = plt.subplots(figsize=(11, 5.2), facecolor="#2E3138")
    ax.set_facecolor("#3A3E46")

    for k, row in enumerate(series):
        ax.plot(dt, row, color="#9AA0AA", lw=0.5, alpha=0.30,
                label=(f"each of the {len(picks)} points" if k == 0 else None))
    ax.plot(dt, lo, color="#E8863C", lw=0.9, alpha=0.75, label="min and max across the points")
    ax.plot(dt, hi, color="#E8863C", lw=0.9, alpha=0.75)
    ax.plot(dt, mean, color="#F08A24", lw=1.6, marker="o", ms=3.5,
            label=f"Average ({len(picks)} points)")
    ax.axhline(0, color="#D8DBE0", lw=0.8)

    ax.set_title(f"Line of Sight Cumulative Displacement from "
                 f"{dates[0][:4]}-{dates[0][4:6]}-{dates[0][6:]} to "
                 f"{dates[-1][:4]}-{dates[-1][4:6]}-{dates[-1][6:]}",
                 color="#E4E6EA", fontsize=11, loc="left", pad=12)
    ax.set_ylabel(f"Line of Sight Displacement ({disp_label})", color="#B9BEC6")
    for spine in ax.spines.values():
        spine.set_color("#5A5F68")
    ax.tick_params(colors="#B9BEC6", labelsize=8)
    for lbl in ax.get_xticklabels():
        lbl.set_rotation(30)
        lbl.set_ha("right")
    ax.grid(color="#4A4E57", lw=0.5)

    # Mirror axis naming the sign convention, as monitoring reports do.
    right = ax.twinx()
    right.set_ylim(ax.get_ylim())
    right.set_yticks([])
    # A rotated label reads bottom to top: negative (bottom) is AWAY from the
    # satellite, i.e. subsidence; positive (top) is toward it.
    right.set_ylabel("\u2193 Away from satellite (subsidence)          Toward satellite \u2191",
                     color="#B9BEC6", fontsize=8)
    for spine in right.spines.values():
        spine.set_color("#5A5F68")

    leg = ax.legend(loc="lower left", facecolor="#3A3E46",
                    edgecolor="#5A5F68", fontsize=8)
    for text in leg.get_texts():
        text.set_color("#E4E6EA")

    fig.text(0.01, 0.015,
             f"Reference: {reference_label(cfg)}   |   "
             f"temporal coherence >= {cfg.min_temporal_coherence}   |   "
             + (f"{len(picks)}-point cluster in the project area '{site_label}'" if site_label
                else f"{len(picks)}-point cluster on the deforming feature"),
             color="#868C96", fontsize=7)
    fig.tight_layout(rect=[0, 0.03, 1, 1])

    path = out_dir / "los_cumulative_displacement_selected_points.png"
    fig.savefig(path, dpi=170, facecolor=fig.get_facecolor(), metadata=png_metadata())
    plt.close(fig)

    csv = out_dir / "los_cumulative_displacement_selected_points.csv"
    from .areas import formula_safe_text
    with open(csv, "w") as f:
        f.write(f"# {SIGN_NOTE}\n")
        # the project area's name is the config's: on the '#' line and no
        # formula in the cells a spreadsheet splits it into (16 Oct 2026,
        # readiness review SEC-05)
        f.write(f"# average of a {len(picks)}-point cluster centred on "
                + (f"the project area '{formula_safe_text(site_label)}'" if site_label else "the deforming feature")
                + " (vendor-comparable selection)\n")
        f.write(f"# reference point: {reference_label(cfg)}\n")
        f.write(f"# software: {_software()}\n")       # (16 Oct 2026, readiness review OUT-01)
        f.write(f"date,mean_{disp_label},min_{disp_label},max_{disp_label}\n")
        for i, d in enumerate(dates):
            f.write(f"{d},{mean[i]:.3f},{lo[i]:.3f},{hi[i]:.3f}\n")

    log.info("Wrote %s (%d points, final average %.2f %s)",
             path, len(picks), mean[-1], disp_label)
    return path


# --------------------------------------------------------------------- KMZ

def export_kmz_points(cfg: PipelineConfig, out_dir: Path) -> Path:
    """
    Displacement points as KMZ, sampled on a grid.

    MintPy ships save_kmz_timeseries.py, which produces a richer product
    with an embedded plot per point - it is tried first. This fallback
    writes a plain point KMZ carrying velocity and cumulative displacement,
    which is enough for Google Earth review and has no extra dependencies.

    OPT-IN since 5 Oct 2026 (`full_timeseries_kmz`, export_all): on a
    ten-year stack MintPy's KMZ is 3.8 GB (site_b) to 16.6 GB
    (site_a_pl), more than Google Earth opens, and it shows absolute LOS
    in cm/yr that disagrees with the report's relative vertical in/yr. The
    client KMZ (client_kmz.py) is the Google Earth product.
    """
    ts_path = timeseries_file(cfg)
    # save_kmz_timeseries.py writes <stem>.kmz BESIDE ITS INPUT (its
    # work_dir is dirname(ts_file)), i.e. in mintpy/geo/ for a geocoded
    # series. The first version of this looked in mintpy/ and, finding
    # nothing, threw away a complete 491 MB product every run while logging
    # that none had been made.
    expected = ts_path.with_suffix(".kmz")
    try:
        t0 = time.time()
        result = subprocess.run(
            ["save_kmz_timeseries.py", str(ts_path)],
            cwd=ts_path.parent, capture_output=True, text=True, timeout=1800,
        )
        fresh = expected.exists() and expected.stat().st_mtime >= t0 - 1
        if result.returncode == 0 and fresh and expected.stat().st_size > 0:
            target = out_dir / expected.name
            shutil.copyfile(expected, target)
            log.info("Wrote %s (%.0f MB, via save_kmz_timeseries.py in %.0f s)",
                     target, target.stat().st_size / 1e6, time.time() - t0)
            return target
        log.warning("save_kmz_timeseries.py did not produce %s (rc %s, fresh=%s): %s "
                    "- writing the simple point KMZ instead", expected.name,
                    result.returncode, fresh, (result.stderr or result.stdout or "")[-600:])
    except Exception as exc:                                  # noqa: BLE001
        log.warning("save_kmz_timeseries.py unavailable (%s) - writing the "
                    "simple point KMZ instead", exc)

    dates, ts, attrs = _dates_and_series(cfg)
    vel, _ = _read(_product(cfg, "velocity.h5"), "velocity")
    mask = load_mask(cfg)
    step = max(1, cfg.kmz_point_step)
    rate_label, disp_label, divisor = _unit_label(cfg)

    # The same quantities, in the same units, as every other product: the
    # cumulative figure is the CHANGE between each cell's first and last
    # valid epoch (export_cumulative_displacement), not ts[-1], which is a
    # position relative to the reference DATE (20161206 on site_b), and
    # the units follow display_units instead of a hard-coded mm.
    first, last, _, _, _ = _anchors(ts)
    cumulative = (last - first) * 1000.0 / divisor
    vel_mm = vel * 1000.0 / divisor
    vmax = np.nanmax(np.abs(vel_mm[mask])) if np.any(mask) else 1.0
    vmax = vmax or 1.0

    def colour(v):
        # KML is aabbggrr. Blue for motion toward the satellite, red away.
        t = max(-1.0, min(1.0, v / vmax))
        if t >= 0:
            return "ff%02x%02x%02x" % (int(255 * (1 - t)), int(255 * (1 - t)), 255)
        return "ffff%02x%02x" % (int(255 * (1 + t)), int(255 * (1 + t)))

    rows = []
    for r in range(0, ts.shape[1], step):
        for c in range(0, ts.shape[2], step):
            if not mask[r, c] or not np.isfinite(vel_mm[r, c]):
                continue
            # the cell CENTRE, in lon/lat whatever the product CRS (KML needs degrees)
            lon, lat = (float(z) for z in _rc_to_lonlat(attrs, r, c))
            rows.append((lat, lon, vel_mm[r, c], cumulative[r, c]))

    log.info("KMZ: %d points at every %d pixels", len(rows), step)

    parts = ['<?xml version="1.0" encoding="UTF-8"?>',
             '<kml xmlns="http://www.opengis.net/kml/2.2"><Document>',
             '<name>LOS displacement points</name>',
             f'<description>{SIGN_NOTE} Reference point: '
             f'{reference_label(cfg)}. Temporal coherence &gt;= '
             f'{cfg.min_temporal_coherence}. Period {dates[0]} to {dates[-1]}.'
             f' Processed with {_software()}.'          # (16 Oct 2026, readiness review OUT-01)
             '</description>']

    styles = {}
    for lat, lon, v, cum in rows:
        col = colour(v)
        if col not in styles:
            styles[col] = f"s{len(styles)}"
            # the icon over https (16 Oct 2026, readiness review DAT-05)
            parts.append(
                f'<Style id="{styles[col]}"><IconStyle><color>{col}</color>'
                '<scale>0.5</scale><Icon><href>https://maps.google.com/mapfiles/'
                'kml/shapes/placemark_circle.png</href></Icon></IconStyle>'
                '<LabelStyle><scale>0</scale></LabelStyle></Style>')

    for lat, lon, v, cum in rows:
        parts.append(
            f'<Placemark><styleUrl>#{styles[colour(v)]}</styleUrl>'
            f'<description><![CDATA['
            f'<b>LOS velocity:</b> {v:.2f} {rate_label}<br/>'
            f'<b>Cumulative LOS displacement:</b> {cum:.2f} {disp_label}<br/>'
            f'<b>Location:</b> {lat:.5f}, {lon:.5f}<br/>'
            f'<i>{SIGN_NOTE}</i>]]></description>'
            f'<Point><coordinates>{lon:.6f},{lat:.6f},0</coordinates></Point>'
            '</Placemark>')
    parts.append('</Document></kml>')

    path = out_dir / "los_displacement_points.kmz"
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("doc.kml", "\n".join(parts))
    log.info("Wrote %s", path)
    return path


# ------------------------------------------------------- summary statistics

def _look_angle(attrs: dict, cfg: PipelineConfig | None = None):
    """
    Incidence angle: measured if available, computed if not.

    Order of preference, because having two values for one quantity is how
    this project keeps getting caught:

      1. `geometryGeo.h5`, written from DISP-S1-STATIC. This is the real
         per-pixel field, and its AOI mean is the number to quote.
      2. An explicit attribute, if the product states one.
      3. Law of cosines on the (Earth centre, satellite, target) triangle,
         from attributes MintPy does write. Good to about a degree - it
         returns 37.46 where the static layers say 38.51 over the same
         AOI, because it uses a single mid-swath slant range.

    A null here silently turns any line-of-sight to vertical projection
    downstream into a no-op, so a computed value beats a blank.

    MintPy does not write INCIDENCE_ANGLE into velocity.h5 on the HyP3 or
    DISP paths - different attribute keys than the ISCE path - which is
    why this came back null. It is recoverable: the law of cosines on the
    (Earth centre, satellite, target) triangle gives it to a fraction of a
    degree from attributes those files do carry. A null here silently
    turns any LOS-to-vertical projection downstream into a no-op, so it is
    worth computing rather than leaving blank.
    """
    if cfg is not None:
        try:
            inc = geometry_field(cfg, "incidence")
        except Exception as exc:                              # noqa: BLE001
            log.warning("Could not read the incidence layer (%s)", exc)
            inc = None
        if inc is not None:
            inc = inc[np.isfinite(inc)]
            if inc.size:
                return round(float(inc.mean()), 2)

    for key in ("INCIDENCE_ANGLE", "incidenceAngle", "INCIDENCE_ANGLE_DEG"):
        v = attrs.get(key)
        if v not in (None, "", "None"):
            try:
                return round(float(v), 2)
            except (TypeError, ValueError):
                pass
    try:
        re_m = float(attrs["EARTH_RADIUS"])
        h = float(attrs["HEIGHT"])
        rng = float(attrs["STARTING_RANGE"])
        # Move to mid-swath only on a RADAR grid. After geocoding, WIDTH is
        # a count of longitude columns while RANGE_PIXEL_SIZE is still the
        # slant-range spacing; multiplying them added 30 km of range that
        # does not exist and reported 40.38 deg where the geometry says 38.37.
        if "X_FIRST" not in attrs and "Y_FIRST" not in attrs:
            width = float(attrs.get("WIDTH", 0) or 0)
            rng += 0.5 * width * float(attrs.get("RANGE_PIXEL_SIZE", 0) or 0)
    except (KeyError, TypeError, ValueError):
        return None
    cos_inc = ((re_m + h) ** 2 - re_m ** 2 - rng ** 2) / (2.0 * rng * re_m)
    if not -1.0 <= cos_inc <= 1.0:
        return None
    return round(float(np.degrees(np.arccos(cos_inc))), 2)


def _track(cfg: PipelineConfig, attrs: dict):
    """Relative orbit, under whichever key this backend happened to use."""
    for key in ("trackNumber", "relative_orbit", "relativeOrbit",
                "PATH_NUMBER", "pathNumber", "TRACK_NUMBER"):
        v = attrs.get(key)
        if v not in (None, "", "None"):
            try:
                return int(float(v))
            except (TypeError, ValueError):
                return v
    # The config knows it even when the product metadata does not.
    return cfg.track


def _precision(cfg: PipelineConfig, vel, gated, rate_label) -> dict:
    """Formal rate uncertainty, quoted over all fitted pixels, not the gated ones."""
    import h5py
    f = _product(cfg, "velocity.h5")
    try:
        with h5py.File(f, "r") as h:
            if "velocityStd" not in h:
                return {"available": False,
                        "note": "this backend produced no velocityStd"}
            std = h["velocityStd"][:]
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "note": str(exc)}

    _, _, divisor = _unit_label(cfg)
    fitted = np.isfinite(std) & np.isfinite(vel)
    if not fitted.any():
        return {"available": False, "note": "no fitted pixels"}
    s_all = (std[fitted] * 1000.0 / divisor)
    sig95 = np.abs(vel[fitted]) > 1.96 * std[fitted]
    out = {
        "available": True,
        "basis": "all fitted pixels, before the quality gate",
        "n_fitted": int(fitted.sum()),
        "median_1sigma": round(float(np.median(s_all)), 4),
        "p90_1sigma": round(float(np.percentile(s_all, 90)), 4),
        "percent_significant_95": round(100 * float(np.mean(sig95)), 1),
        "percent_significant_95_note": ("against the FORMAL sigma and on absolute rates: counts the "
                                        "distributed-scatterer bias as motion. Use significance_per_cell."),
        "units": rate_label,
    }
    if gated is not None and gated.any():
        out["percent_significant_95_of_reported"] = round(
            100 * float(np.mean(np.abs(vel[gated]) > 1.96 * std[gated])), 1)
    return out


def _cell_significance(cfg, reported) -> dict:
    try:
        far, rel, emp, model, null = _empirical_context(cfg)
    except Exception as exc:                                  # noqa: BLE001
        return {"available": False, "note": str(exc)}
    flag = np.abs(rel) > 1.96 * null
    out = {"available": True,
           "percent_of_reported_cells_significant_95": round(100 * float(flag[reported].mean()), 1),
           "percent_of_far_field_cells_significant_95": round(100 * float(flag[far].mean()), 1),
           "empirical_rate_error_by_class": model.get("by_class"),
           "farmland_cell_spread_by_class": model.get("farmland_cell_spread"),
           # first-pass values stay in technical files only if labelled (E14)
           "farmland_pass": "first",
           "note": ("a cell counts as significant when its class-matched relative rate exceeds 1.96 x the spread of "
                    "farmland cells of its coherence class (or its own repeatability if larger). The far-field share "
                    "is the chance rate; only an excess over it means anything. These shares are on the FIRST "
                    "farmland pass (the report areas other than the whole study area already left out, as the "
                    "report leaves them out), over every reported cell of the product grid, written before the "
                    "client report took the movement it found out of the farmland; the report's own shares are in "
                    "signal_to_noise.json (rate_signal_to_noise.tif).")}
    from .significance import reference_mode
    mode = reference_mode(cfg)
    if mode == "stable_polygon":
        out["reference_mode"] = mode
        out["note"] = ("a cell counts as significant when its rate relative to the stable ground of its coherence "
                       "class exceeds 1.96 x the spread of the stable ground's cells of that class (or its own "
                       "repeatability if larger). The far-field share here is the share of the stable ground "
                       f"flagged by chance; only an excess over it means anything ({REFERENCE_RECORD}).")
    elif mode == "gnss_absolute":
        out["reference_mode"] = mode
        out.pop("farmland_pass")
        try:
            rec = json.loads((Path(cfg.project_dir) / "deliverables" / REFERENCE_RECORD).read_text())
            out["gnss_datum"] = {"stations": sum(1 for m in rec.get("stations", []) if m.get("in_fit")),
                                 "rate_added_mm_yr": (rec.get("rate") or {}).get("offset_mm_yr"),
                                 "leave_one_out": (rec.get("leave_one_out") or {}).get("summary")}
        except (OSError, ValueError):
            pass
        out["note"] = ("absolute mode: a cell counts as significant when its rate on the GNSS datum exceeds 1.96 x "
                       "its per-cell noise (the farmland cell spread or its repeatability, with the datum's "
                       "leave-one-out rate error added). The far-field share is NOT a chance rate here: where the "
                       f"whole area moves, the farmland is flagged too ({REFERENCE_RECORD}).")
    return out


def summary_stats(cfg: PipelineConfig) -> dict:
    """
    The headline numbers vendor reports lead with.

    Coverage and measured-point count say more about whether a result is
    usable than any single velocity does: a beautiful velocity map over
    12% of the site is not a monitoring product.

    What changed on 5 Oct 2026, and why: this file used to tell a coworker
    that 2,142,698 Site A cells (48% of the measured ground) were
    "deforming" faster than -0.5 in/yr and that the data quality was "Good",
    while the QA said REVIEW and the report said nothing stands out. Both
    came from ABSOLUTE rates, which on low-coherence farmland carry the
    distributed-scatterer bias (about -12 mm/yr), and from a grade on
    coverage alone. Now:
      deforming_area                 on the class-matched RELATIVE field with
                                     the far-field chance share (the QA's rule;
                                     148,942 cells on Site A, not 2.1 M)
      raw_los_rate_below_threshold   the old absolute count, named for what it is
      raw_los_velocity_vs_reference_*  the old velocity_min/max/median
      data_quality                   the QA verdict (refresh_summary after QA);
                                     coverage_grade is the old Good/Fair/Poor
    read_summary() maps files written before this onto the new keys.
    """
    import h5py

    mask = load_mask(cfg)
    vel, attrs = _read(_product(cfg, "velocity.h5"), "velocity")
    with h5py.File(timeseries_file(cfg), "r") as f:
        dates = [d.decode() if isinstance(d, bytes) else d for d in f["date"][:]]

    # Two denominators, because they answer different questions:
    #   - of the ground that could be measured at all, how much is
    #     reliable enough to report? (data quality)
    #   - of the AOI grid, how much is actually monitored? (site coverage)
    # A vendor quoting a high coverage figure is usually quoting the
    # first. Reporting only one invites a false comparison.
    #
    # The quality denominator used to be isfinite(velocity), which is not
    # a property of the ground at all - it is a property of this pipeline's
    # own masking. Where the velocity fit succeeded everywhere the two
    # figures collapsed to the same number and the quality figure said
    # nothing. Prefer an observable mask written by the backend (land,
    # excluding water and permanently-unmeasurable pixels) and fall back
    # only when there is none, saying so in coverage_basis.
    valid = np.isfinite(vel)
    observable, obs_source = observable_mask(cfg, vel.shape)
    if observable is not None:
        basis = f"share of imaged pixels ({obs_source})"
    else:
        observable = valid
        basis = ("share of pixels with a velocity - no observable mask "
                 "available, so this is not independent of the masking above")
    measured = int(np.sum(mask & valid))
    n_valid = int(np.sum(observable))
    n_grid = int(vel.size)
    coverage = float(measured / n_valid) if n_valid else 0.0
    grid_coverage = float(measured / n_grid) if n_grid else 0.0

    # The product grid is the geocoded bounding rectangle of a rotated radar
    # frame and can be far larger than the AOI (at Site A the AOI is
    # 53% of it). The figure a vendor quotes is IN the AOI, so report that
    # too, on the imaged pixels inside cfg.bbox (cell CENTRES inside it).
    aoi_measured = aoi_imaged = None
    inside = None
    try:
        inside = _box_mask(attrs, cfg.bbox, vel.shape)
        aoi_measured = int(np.sum(mask & valid & inside))
        aoi_imaged = int(np.sum(observable & inside))
    except (KeyError, ValueError, TypeError):
        pass
    aoi_coverage = (aoi_measured / aoi_imaged) if aoi_imaged else None

    rate_label, _, divisor = _unit_label(cfg)
    vel_out = (vel * 1000.0 / divisor)[mask & valid]

    span_days = (datetime.strptime(dates[-1], "%Y%m%d")
                 - datetime.strptime(dates[0], "%Y%m%d")).days

    # A coverage grade, in the spirit of the "Data Quality" field on a
    # vendor summary sheet. Coverage and stack length drive it, nothing
    # else - which is why it is no longer called data_quality: a "Good"
    # beside a QA verdict of REVIEW (site_b) read as a contradiction.
    # Graded on the AOI figure when there is one: that is the area the
    # client asked about and the basis a vendor quotes on.
    grade_on = aoi_coverage if aoi_coverage is not None else coverage
    if grade_on >= 0.7 and span_days >= 365:
        grade = "Good"
    elif grade_on >= 0.4 and span_days >= 180:
        grade = "Fair"
    else:
        grade = "Poor"

    # Radar resolution cells: MintPy geocodes by nearest neighbour, so each
    # radar pixel (about 40 x 47 m) fills about 7.5 of the 15 m map cells
    # (site_b: 2,621,392 geocoded cells hold 348,670 radar pixels). The
    # cell count is not a count of measurements - and neighbouring radar
    # cells share ground, so neither are these independent (E25, 7 Oct 2026:
    # the count is 'about N radar resolution cells', over the study area
    # where the report gives it).
    n_independent, id_basis, n_in_aoi = None, None, None
    try:
        rr, cc = np.nonzero(mask & valid)
        ids, id_basis = radar_pixel_ids(cfg, rr, cc, vel.shape)
        n_independent = int(np.unique(ids[ids >= 0]).size)
        if inside is not None:
            ins = inside[rr, cc]
            n_in_aoi = int(np.unique(ids[ins & (ids >= 0)]).size)
    except Exception as exc:                                  # noqa: BLE001
        log.warning("Radar resolution cell count not made (%s)", exc)

    rel_area = _relative_deforming_area_from_context(cfg, attrs, mask & valid, inside)

    stats = {
        "date_range": f"{dates[0]} to {dates[-1]}",
        "n_acquisitions": len(dates),
        "span_days": span_days,
        "total_measured_points": measured,
        "total_measured_points_note": ("15 m map cells with a reported value; nearest-neighbour geocoding "
                                       "copies each radar measurement into several cells - see "
                                       "radar_resolution_cells"),
        "radar_resolution_cells": n_independent,
        "radar_resolution_cells_in_study_area": n_in_aoi,
        "radar_resolution_cells_basis": id_basis,
        "radar_resolution_cells_note": ("distinct radar resolution cells behind the measured map cells (the whole "
                                        "product grid; _in_study_area: inside the processing box, the count the "
                                        "report gives). Neighbouring radar cells share ground, so they are not "
                                        "independent measurements. independent_radar_measurements is the same "
                                        "whole-grid count under its older name, kept for older readers."),
        "independent_radar_measurements": n_independent,
        "independent_radar_measurements_basis": id_basis,
        "resolution": resolution_info(cfg, attrs),
        "route": dict(zip(("backend", "tag"), _route(cfg))),
        "coverage_percent": round(coverage * 100, 1),
        "coverage_basis": basis,
        "aoi_coverage_percent": (round(aoi_coverage * 100, 1)
                                 if aoi_coverage is not None else None),
        "aoi_measured_points": aoi_measured,
        "aoi_imaged_cells": aoi_imaged,
        "product_grid_coverage_percent": round(grid_coverage * 100, 1),
        "product_grid_cells": n_grid,
        "imaged_cells": n_valid,
        "grade_basis": ("AOI coverage" if aoi_coverage is not None
                        else "product-grid coverage"),
        "temporal_coherence_threshold": cfg.min_temporal_coherence,
        "coverage_grade": grade,
        "coverage_grade_note": ("coverage and record length only - NOT a quality verdict; "
                                "data_quality is the quality assessment's verdict"),
        "data_quality": None,
        "data_quality_source": "pending: the quality assessment runs after this file is written",
        "rate_units": rate_label,
        "raw_los_velocity_vs_reference_min": round(float(vel_out.min()), 3) if vel_out.size else None,
        "raw_los_velocity_vs_reference_max": round(float(vel_out.max()), 3) if vel_out.size else None,
        "raw_los_velocity_vs_reference_median": round(float(np.median(vel_out)), 3) if vel_out.size else None,
        "raw_los_velocity_note": ("line of sight against the reference pixel, single cells; on low-coherence "
                                  "farmland this carries the distributed-scatterer bias, so the median is "
                                  "farmland drift, not ground motion"),
        # Ground faster than the threshold RELATIVE to farmland of the same
        # coherence class, with the share of farmland that crosses it by chance.
        "deforming_area": rel_area,
        # The old absolute count, kept and named for what it is.
        "raw_los_rate_below_threshold": _deforming_area(cfg, vel_out, vel, mask & valid,
                                                        divisor, cfg.deforming_threshold),
        # The formal figure, over the pixels that were FIT rather than the
        # ones that passed the quality gate. Filtering on a quality metric
        # that correlates with sigma and then quoting median sigma is
        # selection on the outcome: it describes the pixels whose noise
        # happened to be small and calls that the product's precision.
        "velocity_std_formal": _precision(cfg, vel, mask & valid, rate_label),
        # the per-cell test the point table carries, and how often ordinary
        # farmland passes it by chance
        "significance_per_cell": _cell_significance(cfg, mask & valid),
        # And the honest one, which absorbs the reference point, the
        # troposphere and any ramp. Quote the LARGER of the two.
        "stable_area_dispersion": stable_area_dispersion(cfg, vel, mask & valid),
        "egms_reference_target": (
            "EGMS publishes 0.7 mm/yr (1 sigma) for points with coherence "
            "> 0.7; compare velocity_std_formal against that, and note that "
            "the stable-area dispersion is the figure that includes "
            "systematics."),
        "look_angle_deg": _look_angle(attrs, cfg),
        "orbit_direction": attrs.get("ORBIT_DIRECTION"),
        "wavelength_m": attrs.get("WAVELENGTH"),
        "track": _track(cfg, attrs),
    }
    _apply_qa_and_gnss(cfg, stats)

    log.info("Summary: %d measured cells (about %s radar resolution cells) | %s in the AOI | "
             "%.1f%% of imaged pixels, %.1f%% of the product grid | %d acquisitions over %d days "
             "| coverage grade %s",
             measured, n_independent if n_independent is not None else "n/a",
             (f"{aoi_coverage * 100:.1f}%" if aoi_coverage is not None else "n/a"),
             coverage * 100, grid_coverage * 100, len(dates), span_days, grade)
    if grade != "Good":
        log.warning("Coverage grades as %s (%.1f%% %s over %d days). Vendor "
                    "products typically quote 70%%+ in the AOI.", grade,
                    grade_on * 100, stats["grade_basis"], span_days)
    return stats


# Keys renamed on 5 Oct 2026: old -> new. read_summary maps files written
# before then, so a reader asks for the new name only.
SUMMARY_RENAMED = {
    "velocity_min": "raw_los_velocity_vs_reference_min",
    "velocity_max": "raw_los_velocity_vs_reference_max",
    "velocity_median": "raw_los_velocity_vs_reference_median",
}


def read_summary(path) -> dict:
    """deliverables/summary.json (a project dir, a deliverables dir or the
    file), with files written before 5 Oct 2026 mapped onto the new keys:
    velocity_* -> raw_los_velocity_vs_reference_*, the old coverage-only
    data_quality -> coverage_grade (data_quality then None until the QA
    verdict is attached), and the old absolute deforming_area ->
    raw_los_rate_below_threshold. Missing file -> {}."""
    p = Path(path)
    if p.is_dir():
        p = p / "summary.json" if (p / "summary.json").exists() else p / "deliverables" / "summary.json"
    if not p.exists():
        return {}
    S = json.loads(p.read_text())
    for old, new in SUMMARY_RENAMED.items():
        if old in S and new not in S:
            S[new] = S[old]
    if "coverage_grade" not in S and S.get("data_quality") in ("Good", "Fair", "Poor"):
        S["coverage_grade"] = S["data_quality"]
        S["data_quality"] = None
        S["data_quality_source"] = "not recorded (summary written before 5 Oct 2026)"
    if "raw_los_rate_below_threshold" not in S and isinstance(S.get("deforming_area"), dict) \
            and "relative_to" not in S["deforming_area"]:
        S["raw_los_rate_below_threshold"] = S.pop("deforming_area")
    return S


def _fresh(path: Path, than: Path) -> bool:
    try:
        return path.exists() and path.stat().st_mtime >= than.stat().st_mtime
    except OSError:
        return False


def _apply_qa_coverage(stats: dict, Q: dict) -> None:
    """The QA's coverage judgement in summary.json, replacing the stand-alone
    share-and-span grade: Site B's summary said coverage_grade 'Good' at
    98.6% (first gate only) while its QA flagged coverage (the threshold sits
    inside the noise floor) and said 'hold for specialist' (review outcome
    check, 6 Oct 2026). The old letter is kept as coverage_grade_by_share.
    Idempotent: refresh_summary runs after the QA and again after the tie."""
    cov = (Q.get("checks") or {}).get("coverage") or {}
    if not cov:
        return
    old = stats.get("coverage_grade_by_share")
    if old is None and stats.get("coverage_grade") in ("Good", "Fair", "Poor"):
        old = stats["coverage_grade"]
    stats["coverage_grade_by_share"] = old
    rel = cov.get("reliable_percent")
    stats["reliable_percent"] = rel
    stats["coverage"] = {k: cov.get(k) for k in ("threshold", "aoi_percent_at_threshold", "aoi_percent_first_gate",
                                                  "reliable_percent", "threshold_above_noise", "pass")}
    stats["coverage"]["noise_floor"] = (cov.get("noise_level") or {}).get("floor")
    both = f"; {rel:.1f}% of the area passes both quality checks" if rel is not None else ""
    verdict = Q.get("verdict")
    if cov.get("pass") is False:
        why = []
        if cov.get("threshold_above_noise") is False:
            why.append(f"the coherence threshold ({cov.get('threshold')}) is inside the noise floor "
                       f"({stats['coverage']['noise_floor']}), so coverage does not mean reliability")
        elif cov.get("aoi_percent_at_threshold") is not None:
            why.append(f"{cov['aoi_percent_at_threshold']:.1f}% of the area measured, under the "
                       f"{cov.get('yardstick_percent')}% yardstick")
        stats["coverage_grade"] = ("Flagged by the quality assessment" + (": " + "; ".join(why) if why else "")
                                   + both + ".")
    else:
        stats["coverage_grade"] = (f"{old or 'Graded'} coverage, met the quality assessment's yardstick"
                                   + (f" ({cov['aoi_percent_at_threshold']:.1f}% measured)"
                                      if cov.get("aoi_percent_at_threshold") is not None else "") + both
                                   + (f"; overall verdict {verdict}" if verdict else "") + ".")
    stats["coverage_grade_note"] = ("from the quality assessment's coverage check (checks.coverage; reliable_percent "
                                    "also passes unwrapping consistency >= 0.7); coverage_grade_by_share is the "
                                    "share-and-record-length grade alone, NOT a quality verdict")


def _apply_qa_and_gnss(cfg, stats: dict) -> dict:
    """Put the QA verdict in data_quality and the GNSS tie in gnss_tie, when
    each exists and is at least as new as the velocity product; leave a
    plain 'pending' otherwise. An older file belongs to another run. With a
    tie on disk, send_rule is the QA's rule re-evaluated with it, as the
    report and the release list have it."""
    vel_path = _product(cfg, "velocity.h5")
    qa_path = Path(cfg.project_dir) / "quality_assessment.json"
    qa_used = None
    if _fresh(qa_path, vel_path):
        try:
            Q = json.loads(qa_path.read_text())
            if Q.get("verdict"):
                qa_used = Q
                stats["data_quality"] = Q["verdict"]
                # counted as the verdict counts them (D2, 7 Oct 2026): out of
                # the applicable checks, naming any that cannot apply
                from .quality_assessment import check_counts
                stats["data_quality_source"] = ("quality_assessment.json verdict"
                                                + (f" ({check_counts(Q)['text']})" if Q.get("checks") else ""))
                if isinstance(Q.get("send_rule"), dict):
                    stats["send_rule"] = Q["send_rule"].get("value")
                _apply_qa_coverage(stats, Q)
                fd = (Q.get("findings") or {}).get("deforming_area")
                if isinstance(fd, dict) and fd.get("n_pixels") is not None:
                    stats["deforming_area_qa_n_pixels"] = fd["n_pixels"]
        except (ValueError, OSError) as exc:
            log.warning("Could not read %s (%s)", qa_path, exc)
    elif qa_path.exists():
        stats["data_quality"] = None
        stats["data_quality_source"] = ("pending: quality_assessment.json is older than the velocity "
                                        "product (from an earlier run)")
    tie_path = Path(cfg.project_dir) / "deliverables" / "gnss_tie.json"
    tie = None
    dates = str(stats.get("date_range") or "").split(" to ")
    if tie_path.exists() and len(dates) == 2:
        # the report's rule (gnss_tie.check, via client_report._tie_for_send_rule):
        # a tie made before the regional floor, for another record window,
        # older than the velocity product or unreadable is a refusal - the
        # figures go relative-only and the send rule holds. Until 7 Oct 2026
        # only the pre-floor case held here; an old or other-window tie was
        # ignored or counted as anchored while the report held.
        try:
            from . import gnss_tie as _gt
            tie, why = _gt.check(cfg, dates)
        except Exception as exc:                          # noqa: BLE001 - summary still written
            tie, why = None, f"the GNSS tie could not be checked ({exc})"
        if tie is None:
            try:
                raw = json.loads(tie_path.read_text())
            except (ValueError, OSError):
                raw = {}
            tie = {"background": None, "reason": why, "stations_used": raw.get("stations_used"),
                   "record_window": raw.get("record_window")}
    elif _fresh(tie_path, vel_path):
        try:
            tie = json.loads(tie_path.read_text())
        except (ValueError, OSError):
            tie = None
        if tie is not None:
            try:
                from . import gnss_tie as _gt
                old_tie = _gt.stale_reason(tie)
            except Exception:                             # noqa: BLE001 - summary still written
                old_tie = None
            if old_tie:
                # a pre-floor tie is not this record's anchor (gnss_tie.check)
                tie = {"background": None, "reason": old_tie, "stations_used": tie.get("stations_used"),
                       "record_window": tie.get("record_window")}
    if tie is not None:
        bg = tie.get("background")
        stats["gnss_tie"] = {"anchored": bool(bg), "background": bg, "reason": tie.get("reason"),
                             "stations_used": tie.get("stations_used"),
                             "record_window": tie.get("record_window")}
        # rewritten from the current tie every time (a re-run tie must not
        # leave 'not tied ... re-run the GNSS tie' beside gnss_tie.anchored);
        # a gnss_validation verdict is left alone
        if "gnss" not in stats or str(stats.get("gnss", "")).startswith(
                ("skipped", "tied to GNSS", "not tied to GNSS")):
            stats["gnss"] = ("tied to GNSS (gnss_tie.json)" if bg else
                             f"not tied to GNSS: {tie.get('reason') or 'see gnss_tie.json'}")
    if qa_used is not None and isinstance(qa_used.get("send_rule"), dict):
        # the QA runs before the tie, so its stored rule never saw it; the
        # report and the release list re-evaluate it with the tie
        # (client_report._send_rule), and summary.json says the same
        stats["send_rule_gnss_considered"] = False
        if tie is not None:
            try:
                from . import quality_assessment as _qa
                stats["send_rule"] = _qa.send_rule(qa_used, gnss=tie)["value"]
                stats["send_rule_gnss_considered"] = True
            except Exception as exc:                          # noqa: BLE001 - summary still written
                log.warning("Send rule not re-evaluated with the GNSS tie (%s); summary.json keeps the "
                            "quality assessment's", exc)
    return stats


def refresh_summary(cfg) -> dict:
    """Re-read the QA verdict and the GNSS tie into deliverables/summary.json.
    The pipeline writes summary.json before the quality assessment and the
    GNSS tie run, so it calls this after each of them; a later rebuild picks
    both up directly. Returns the updated summary ({} when there is none)."""
    p = Path(cfg.project_dir) / "deliverables" / "summary.json"
    if not p.exists():
        return {}
    S = read_summary(p)
    _apply_qa_and_gnss(cfg, S)          # S keeps its 'software': the code that computed its numbers
    p.write_text(json.dumps(S, indent=2, default=_json_default))
    log.info("summary.json: data_quality %s (%s)", S.get("data_quality"), S.get("data_quality_source"))
    return S


def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def relative_deforming_area(cfg, rel: np.ndarray, far: np.ndarray, good: np.ndarray,
                            inside: np.ndarray | None, attrs: dict) -> dict:
    """
    Ground faster than deforming_threshold RELATIVE to farmland of the same
    coherence class (rel, mm/yr), inside the AOI, beside the share of
    far-field farmland that crosses the same threshold by chance - the same
    rule as quality_assessment.findings.deforming_area. Single cells are
    noisy (several mm/yr on low-coherence ground), so only an excess over
    the chance share means anything.
    """
    thr = float(getattr(cfg, "deforming_threshold", 0.5))
    rate_label, _, divisor = _unit_label(cfg)
    thr_mm = thr * divisor
    region = good & np.isfinite(rel) & (inside if inside is not None else True)
    moving = region & (rel < -thr_mm)
    far_ok = far & np.isfinite(rel)
    far_share = float((far_ok & (rel < -thr_mm)).sum() / max(1, far_ok.sum()))
    try:
        from .significance import pixel_metres
        py, px = pixel_metres(attrs)
        area_km2 = round(float(moving.sum()) * py * px / 1e6, 2)
    except Exception:                                         # noqa: BLE001
        area_km2 = None
    from .significance import reference_mode
    mode = reference_mode(cfg)
    threshold = f"faster than -{thr} {rate_label} relative to the far field, matched by coherence class"
    note = ("compare share_of_measured_aoi_percent with far_field_share_beyond_threshold_percent: "
            "that is how much ordinary farmland crosses the same threshold by chance")
    if mode == "stable_polygon":
        threshold = f"faster than -{thr} {rate_label} relative to the stable ground, matched by coherence class"
        note = ("compare share_of_measured_aoi_percent with far_field_share_beyond_threshold_percent: "
                "that is how much of the stable ground crosses the same threshold by chance")
    elif mode == "gnss_absolute":
        threshold = f"faster than -{thr} {rate_label} on the GNSS datum (absolute, not relative to farmland)"
        note = ("absolute rates: far_field_share_beyond_threshold_percent is the share of the far-field farmland "
                "moving this fast on the GNSS datum. Where the whole area moves it is not a chance rate.")
    return {
        "relative_to": reference_ground(cfg),
        "threshold": threshold,
        "n_pixels": int(moving.sum()),
        "area_km2": area_km2,
        "share_of_measured_aoi_percent": round(100 * float(moving.sum() / max(1, region.sum())), 1),
        "median_rate_mm_yr": round(float(np.median(rel[moving])), 2) if moving.any() else None,
        "fastest_mm_yr": round(float(np.min(rel[moving])), 2) if moving.any() else None,
        "far_field_share_beyond_threshold_percent": round(100 * far_share, 1),
        "note": note,
    }


def _relative_deforming_area_from_context(cfg, attrs, good, inside) -> dict:
    try:
        far, rel, _, _, _ = _empirical_context(cfg)
    except Exception as exc:                                  # noqa: BLE001
        log.warning("Relative deforming area not computed (%s)", exc)
        return {"available": False, "note": f"class-matched field unavailable: {exc}"}
    return relative_deforming_area(cfg, rel, far, good, inside, attrs)


def _deforming_area(cfg, vel_out, vel, valid_mask, divisor, threshold_in_yr=1.0):
    """
    ABSOLUTE line-of-sight rate below the threshold (summary.json
    raw_los_rate_below_threshold). Not a deforming area: on low-coherence
    farmland the absolute rate carries the distributed-scatterer bias, so
    this counts farmland drift (2.1 M Site A cells, 48% of the measured
    ground). deforming_area is the relative figure.

    The threshold is a reporting choice, not a property of the data - a
    site whose peak rate sits just under it will report "none found" while
    clearly subsiding. Set deforming_threshold to suit the site.
    """
    # The threshold is in DISPLAY units (deforming_threshold: "faster than
    # 0.5 in/yr" or "faster than 12 mm/yr"); convert to mm with the same
    # divisor the products use. Both branches of the old expression were
    # 25.4, so a mm configuration was off by 25x.
    thresh = -threshold_in_yr * divisor
    rate_mm = vel * 1000.0
    unit = _unit_label(cfg)[0]
    sel = valid_mask & (rate_mm < thresh)
    n = int(np.sum(sel))
    if n < 50:
        # Say what the fastest pixel actually is, so a threshold set just
        # above the site's real rate is obvious rather than looking like
        # an absence of movement.
        finite = rate_mm[valid_mask]
        finite = finite[np.isfinite(finite)]
        peak = round(float(np.min(finite) / divisor), 3) if finite.size else None
        return {
            "n_pixels": n,
            "threshold": f"faster than -{threshold_in_yr} {unit}",
            "fastest_pixel": peak,
            "note": ("no coherent area below the threshold - the fastest "
                     f"pixel is {peak} {unit}, so lower deforming_threshold if "
                     "that is the signal you are after"),
        }
    return {
        "n_pixels": n,
        "threshold": f"faster than -{threshold_in_yr} {unit}",
        "basis": "absolute line-of-sight rate against the reference pixel (includes farmland drift)",
        "median_velocity": round(float(np.median(rate_mm[sel]) / divisor), 3),
        "max_velocity": round(float(np.min(rate_mm[sel]) / divisor), 3),
    }


# ------------------------------------------------------------------ driver

def export_vertical_velocity(cfg: PipelineConfig, out_dir: Path) -> Path | None:
    """
    Vertical velocity, IF the geometry is available and the assumption holds.

    d_vertical = d_los / los_up. That division is the entire conversion,
    and it assumes ALL motion is vertical. A single satellite geometry
    cannot test that assumption - it measures one projection of a
    three-component vector, and any horizontal component is silently
    folded into the answer.

    On a subsidence site the assumption is usually defensible. It is never
    free: at 38 degrees incidence the conversion multiplies every number
    by about 1.28, so quoting a vertical figure where the vendor quotes
    line-of-sight (or the reverse) manufactures a 28% discrepancy out of
    nothing. The raster is written with the assumption in its filename so
    it cannot be mistaken for a measurement.
    """
    vel, attrs = _read(_product(cfg, "velocity.h5"), "velocity")
    try:
        los_up = los_up_or_raise(cfg, vel.shape)
    except (FileNotFoundError, ValueError) as exc:
        log.warning("Vertical projection skipped: %s", exc)
        return None
    mask = load_mask(cfg)
    rate_label, _, divisor = _unit_label(cfg)

    from .static_layers import project_to_vertical
    vert = project_to_vertical(vel, los_up) * 1000.0 / divisor
    vert[~mask] = np.nan

    good = np.isfinite(vert) & np.isfinite(vel)
    factor = None
    if good.any():
        factor = float(np.nanmedian(1.0 / los_up[good]))
        log.info("Vertical projection assumes purely vertical motion; median "
                 "1/cos(incidence) is %.3f, so every number is %.0f%% larger "
                 "than the line-of-sight value it came from.",
                 factor, (factor - 1) * 100)

    path = out_dir / f"vertical_velocity_{rate_label.replace('/', '_per_')}_ASSUMES_VERTICAL.tif"
    _write_geotiff(path, vert, attrs,
                   tags=product_tags(cfg, attrs, "vertical velocity ASSUMING all motion is vertical "
                                     "(LOS velocity / LOS up-component)", rate_label,
                                     "the reference pixel (absolute, not GNSS-anchored, farmland drift included)",
                                     median_los_to_vertical_factor=(round(factor, 4) if factor else "")),
                   description=f"LOS velocity / cos(incidence), {rate_label}; assumes purely vertical motion")
    return path


def export_velocity_uncertainty(cfg: PipelineConfig, out_dir: Path) -> Path | None:
    """
    The 1-sigma rate uncertainty raster, where the backend produced one.

    Formal error only: MintPy's residue-based velocityStd, which treats the
    epochs as independent. They are not (lag-1 residual autocorrelation ~0.8
    on the Site A record), and the empirical repeatability of a cell's
    rate is several times larger - see export_empirical_uncertainty, which
    writes that figure beside this one. Neither contains the reference point
    or the distributed-scatterer bias. Treat this one as a floor, never as
    the uncertainty.
    """
    import h5py
    with h5py.File(_product(cfg, "velocity.h5"), "r") as f:
        if "velocityStd" not in f:
            log.info("velocity.h5 carries no velocityStd - skipping the "
                     "uncertainty raster.")
            return None
        std = f["velocityStd"][:]
        attrs = dict(f.attrs)

    mask = load_mask(cfg)
    rate_label, _, divisor = _unit_label(cfg)
    out = std * 1000.0 / divisor
    out[~mask] = np.nan
    finite = out[np.isfinite(out)]
    if finite.size:
        log.info("Rate uncertainty (1 sigma, formal): median %.3f %s, 90th "
                 "percentile %.3f %s", float(np.median(finite)), rate_label,
                 float(np.percentile(finite, 90)), rate_label)
    path = out_dir / f"los_velocity_uncertainty_1sigma_{rate_label.replace('/', '_per_')}.tif"
    _write_geotiff(path, out, attrs,
                   tags=product_tags(cfg, attrs, "FORMAL 1-sigma LOS rate uncertainty (MintPy velocityStd; "
                                     "assumes independent dates - a floor, not the uncertainty)", rate_label,
                                     "not applicable", sign=SIGN_UNCERTAINTY),
                   description=f"Formal 1-sigma LOS rate uncertainty, {rate_label}: a floor; use "
                               "los_velocity_cell_noise_1sigma for significance")
    return path


def _cell_null_table(model: dict, spread_rows: list) -> dict:
    """
    {coherence class index: per-cell noise, mm/yr LOS} = max(farmland cell
    spread, empirical repeatability) - the one yardstick for a single cell.
    significance.cell_null_by_class is the shared definition; this local copy
    reproduces np.fmax(spread_map, emp_map) exactly (a class without 200
    far-field cells has no spread and takes its repeatability; a class too
    small to measure takes the nearest measured class's repeatability) for
    when that function is not there yet.
    """
    try:
        from .significance import cell_null_by_class
        got = cell_null_by_class(model, spread_rows)
        # non-integer keys (e.g. "_borrowed") are annotations, not classes
        return {int(k): float(v) for k, v in got.items()
                if str(k).lstrip("-").isdigit() and v is not None and np.isfinite(float(v))}
    except (ImportError, AttributeError):
        pass
    from .significance import CLASS_EDGES
    emp = {int(r["class_index"]): float(r["empirical_mm_yr"]) for r in model.get("by_class", [])}
    spr = {int(r["class_index"]): float(r["farmland_cell_spread_mm_yr"]) for r in spread_rows or []}
    out = {}
    for i in range(len(CLASS_EDGES) - 1):
        e = emp.get(i, emp[min(emp, key=lambda j: abs(j - i))] if emp else None)
        vals = [v for v in (e, spr.get(i)) if v is not None]
        if vals:
            out[i] = max(vals)
    return out


def _record_cell_null(cfg, model: dict, spread_rows: list, null_by_class: dict,
                      comparison_ground: dict | None = None) -> dict:
    """
    Write the farmland cell spread and the per-cell noise into
    uncertainty_model.json beside the repeatability, per class. Until 5 Oct
    2026 the spread lived only in memory, so the client rasters and the KMZ
    (which read this file) used repeatability alone - 1.4-1.8x smaller than
    the noise the SNR raster divides by - and showed 3x more 'significant'
    ground than the report's own test. The `key` the cache checks is kept,
    so the cached repeatability stays valid. `comparison_ground` records the
    farmland both were measured on (F7(d): without the report areas other
    than the whole study area, as the report's).
    """
    from .significance import class_labels
    spr = {int(r["class_index"]): r["farmland_cell_spread_mm_yr"] for r in spread_rows or []}
    by_class = []
    for r in model.get("by_class", []):
        r = dict(r)
        i = int(r["class_index"])
        r["farmland_cell_spread_mm_yr"] = spr.get(i)
        r["cell_null_mm_yr"] = round(null_by_class[i], 3) if i in null_by_class else None
        by_class.append(r)
    labels = class_labels()
    model = dict(model, by_class=by_class, farmland_cell_spread=spread_rows,
                 cell_null_by_class={str(i): round(v, 3) for i, v in sorted(null_by_class.items())},
                 cell_null_by_class_label={labels[i]: round(v, 3) for i, v in sorted(null_by_class.items())},
                 cell_null_basis=("per coherence class, max(farmland cell spread, empirical repeatability), "
                                  "mm/yr line of sight: the noise a single cell's class-matched relative "
                                  "rate is tested against (rate_signal_to_noise.tif, significant_95)"))
    if comparison_ground is not None:
        from .significance import reference_mode
        stable = reference_mode(cfg) == "stable_polygon"
        model["comparison_ground"] = dict(comparison_ground, definition=(
            ("the stable ground (reference_mode stable_polygon): reliable cells inside "
             f"{getattr(cfg, 'reference_stable_geojson', '')}, outside the mapped panels themselves" if stable else
             "far-field farmland: coherent cells inside the AOI, outside every mapped panel plus buffer")
            + (", and outside the report areas other than the whole study area plus "
               f"{float(comparison_ground.get('buffer_m') or 500):.0f} m (as the client report)"
               if comparison_ground.get("applied") else "")))
    path = None
    try:
        from .significance import uncertainty_cache_path
        path = uncertainty_cache_path(cfg)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(model, indent=1, default=_json_default))
    except Exception as exc:                                  # noqa: BLE001
        log.warning("Could not record the per-cell noise in %s (%s)", path, exc)
    return model


# ---------------------------------------------------------------- reference modes (B1, 8 Oct 2026)
# What the relative products are relative to (significance.reference_mode;
# its docstring has what each mode costs). 'farmland' (default) is unchanged:
# no new file, no new tag, the same strings.
FARMLAND_GROUND = "far-field farmland of the same coherence class"
REFERENCE_RECORD = "reference_datum.json"
STABLE_BASIS = ("pixels of the stable ground (reference_mode stable_polygon: inside the stable polygon, outside the "
                "mapped panels) whose series is finite on every date; empirical = robust sd of (second-half rate - "
                "first-half rate) / 2, the full-record rate error under random-walk-like noise")
GNSS_SIGMA_FLOOR_MM = 1.0       # no station outweighs the rest on a small formal sigma


def reference_ground(cfg, farmland: str = FARMLAND_GROUND) -> str:
    """What this run's relative products are relative to, in its
    reference_mode: `farmland` (the text the caller has always used) in
    the default mode."""
    from .significance import reference_mode
    mode = reference_mode(cfg)
    if mode == "stable_polygon":
        return ("the stable ground drawn in " + Path(str(getattr(cfg, "reference_stable_geojson", "") or "")).name
                + ", matched by coherence class")
    if mode == "gnss_absolute":
        return f"absolute: the GNSS datum fitted to the stations in {REFERENCE_RECORD}"
    return farmland


def _median_or_none(vals, nd: int = 2):
    v = [float(x) for x in vals if x is not None and np.isfinite(x)]
    return round(float(np.median(v)), nd) if v else None


def _nan_list(a, nd: int = 3) -> list:
    return [None if v is None or not np.isfinite(v) else round(float(v), nd) for v in np.asarray(a, dtype=float)]


def _write_reference_record(cfg, rec: dict) -> Path | None:
    """deliverables/reference_datum.json: what the relative products are
    relative to in a non-default reference_mode, for a person and for the
    validation harness. Never written in farmland mode."""
    path = Path(cfg.project_dir) / "deliverables" / REFERENCE_RECORD
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(rec, indent=1, default=_json_default))
        return path
    except OSError as exc:
        log.warning("Could not write %s (%s)", path, exc)
        return None


def _stable_record(cfg, fmeta: dict, centres: dict) -> dict:
    from .significance import class_labels
    labels = class_labels()
    gj = str(getattr(cfg, "reference_stable_geojson", "") or "")
    return {"reference_mode": "stable_polygon", "available": True,
            "written": datetime.now().isoformat(timespec="seconds"),
            "what": (f"Every relative figure is relative to the stable ground drawn in {Path(gj).name}: each cell "
                     "minus the mean rate of the polygon's reliable cells of its own coherence class."),
            "polygon": gj, "cells_in_polygon": fmeta.get("stable_polygon_cells"),
            "cells_in_mapped_panels_left_out": fmeta.get("stable_polygon_cells_in_panels"),
            "cells_used": fmeta.get("n_cells"), "cells_used_inside_aoi": fmeta.get("stable_polygon_cells_in_aoi"),
            "report_area_cut": fmeta.get("comparison_ground"),
            "class_centres_mm_yr_los": {labels[i]: round(float(v), 3) for i, v in sorted(centres.items())},
            "caution": ("Right only if the polygon is stable; nothing here checks that against GNSS. Coherence "
                        "classes the polygon lacks take the nearest measured class's centre.")}


def _load_gnss_series(station_ids, window, cache_dir):
    """The GNSS series (ground_truth.load_gnss_series: NGL IGS20 daily E/N/U
    in mm with equipment steps removed, positions in .attrs). One place, so
    tests can stand in for the download.

    Screened about a line over +-15 days (ground_truth model 'local'), not
    one line over the record: the datum is for any site, and where the
    ground's motion is not a straight line (Corcoran's water years: -295
    then +43 mm) the scatter about one line is tens of mm (a CRCN-like
    series: 87 mm in Up, against 5 about the local line). That scatter
    became each day's sigma (ground_truth takes the larger of it and the
    formal one), so the stations moving most would have counted for almost
    nothing in the datum (review 8 Oct 2026)."""
    from . import ground_truth
    return ground_truth.load_gnss_series(station_ids, window=window, cache_dir=cache_dir, model="local")


def _gnss_in_frame(cfg, series, frame: str | None = None, plate: str | None = None) -> tuple:
    """
    The datum's GNSS in the configured frame (Decision 5, 8 Oct 2026;
    significance.gnss_frame): 'plate' (default) puts NGL's IGS20 series
    relative to their tectonic plate (ground_truth.to_plate_frame, the
    ITRF2020 plate motion model, one plate for all the stations); 'itrf'
    leaves them as published. `frame`/`plate` override the config.

    Only series that state they are IGS20 are converted. When none states
    a frame they are left as given and the record says the plate's motion
    was not removed (never a plate label on unconverted numbers); a mix of
    stated and unstated, or another frame, raises ValueError.

    Returns (series, record): record {requested, plate_setting, applied,
    frame ('IGS20', 'IGS20 assumed', 'plate:<CODE>', or None when no
    series was loaded), plate (the choice, when applied), reason (when
    not)}.
    """
    from . import ground_truth as gt
    from . import significance as sg
    f_cfg, p_cfg = sg.gnss_frame(cfg)
    if frame is not None:
        f = str(frame).strip().lower()
        frame = {"igs20": "itrf", "itrf2020": "itrf"}.get(f, f)
        if frame not in sg.REFERENCE_FRAMES:
            raise ValueError(f"frame {frame!r} is not one of {', '.join(sg.REFERENCE_FRAMES)}")
    frame = f_cfg if frame is None else frame
    plate = p_cfg if plate is None else plate
    rec = {"requested": frame, "plate_setting": plate if frame == "plate" else None}
    frames = {sid: gt.series_frame(df) for sid, df in (series or {}).items()}
    if frame == "itrf":
        return series, {**rec, "applied": False, "frame": "IGS20",
                        "reason": "reference_gnss_frame is 'itrf': the series as NGL publishes them"}
    other = sorted(s for s, f in frames.items() if f not in ("IGS20", None))
    if other:
        raise ValueError("the datum's GNSS series are not all IGS20 (" + ", ".join(f"{s}: {frames[s]}" for s in other)
                         + "), so the plate's motion cannot be removed from them with the ITRF2020 plate model")
    stated = sorted(s for s, f in frames.items() if f == "IGS20")
    unstated = sorted(s for s, f in frames.items() if f is None)
    if stated and unstated:
        raise ValueError(f"the series of {', '.join(unstated)} do not state their frame while "
                         f"{', '.join(stated)} are IGS20: one datum needs one frame")
    if not frames:
        return series, {**rec, "applied": False, "frame": None,
                        "reason": "no GNSS series were loaded, so there was nothing to put in a frame"}
    if not stated:
        log.warning("The GNSS series do not state their frame: the plate's motion was not removed "
                    "(reference_datum.json, frame)")
        return series, {**rec, "applied": False, "frame": "IGS20 assumed",
                        "reason": ("the GNSS series do not state their frame, so the plate's motion was not "
                                   "removed; they are taken as IGS20, as NGL publishes them")}
    conv = gt.to_plate_frame(series, plate=plate)
    choice = getattr(conv, "plate", None) or {}
    return conv, {**rec, "applied": True, "frame": f"plate:{choice.get('plate')}", "plate": choice}


def _is_geographic(attrs: dict) -> bool:
    return abs(float(attrs["X_STEP"])) < 0.01


def _station_cells(attrs: dict, lat: float, lon: float, shape, radius_m: float):
    """(rows, cols) of the grid cells whose centres lie within radius_m of
    (lat, lon), on a geographic or a projected grid; empty off the grid."""
    from . import significance as sg
    ny, nx = shape
    py, px = sg.pixel_metres(dict(attrs, LENGTH=ny))
    if _is_geographic(attrs):
        r, c = sg.lalo_to_rc(attrs, lat, lon)
    else:
        from pyproj import Transformer
        from .stack_qc import product_epsg
        epsg = product_epsg(attrs, lonlat=(lon, lat), shape=shape)
        x, y = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True).transform(lon, lat)
        r = int(np.floor((y - float(attrs["Y_FIRST"])) / float(attrs["Y_STEP"])))
        c = int(np.floor((x - float(attrs["X_FIRST"])) / float(attrs["X_STEP"])))
    hy, hx = int(np.ceil(radius_m / py)) + 1, int(np.ceil(radius_m / px)) + 1
    r0, r1, c0, c1 = max(0, r - hy), min(ny, r + hy + 1), max(0, c - hx), min(nx, c + hx + 1)
    if r0 >= r1 or c0 >= c1:
        return np.array([], int), np.array([], int)
    rr, cc = np.mgrid[r0:r1, c0:c1]
    if _is_geographic(attrs):
        clat, clon = sg.rc_to_lalo(attrs, rr, cc)
        dy = (clat - lat) * 111_320.0
        dx = (clon - lon) * 111_320.0 * np.cos(np.radians(lat))
    else:
        dy = float(attrs["Y_FIRST"]) + (rr + 0.5) * float(attrs["Y_STEP"]) - y
        dx = float(attrs["X_FIRST"]) + (cc + 0.5) * float(attrs["X_STEP"]) - x
    keep = np.hypot(dx, dy) <= radius_m
    return rr[keep], cc[keep]


def _grid_latlon(attrs: dict, shape, lonlat=None):
    """Cell-centre latitude and longitude of the whole grid (2-D). `lonlat`
    places a projected grid that carries no EPSG (stack_qc.product_epsg)."""
    from . import significance as sg
    ny, nx = shape
    if _is_geographic(attrs):
        lat, _ = sg.rc_to_lalo(attrs, np.arange(ny), 0)
        _, lon = sg.rc_to_lalo(attrs, 0, np.arange(nx))
        return np.broadcast_to(lat[:, None], shape), np.broadcast_to(lon[None, :], shape)
    from pyproj import Transformer
    from .stack_qc import product_epsg
    epsg = product_epsg(attrs, lonlat=lonlat, shape=shape)
    yy, xx = np.mgrid[0:ny, 0:nx]
    X = float(attrs["X_FIRST"]) + (xx + 0.5) * float(attrs["X_STEP"])
    Y = float(attrs["Y_FIRST"]) + (yy + 0.5) * float(attrs["Y_STEP"])
    lon, lat = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True).transform(X, Y)
    return lat, lon


GNSS_COLUMNS = ("e_mm", "n_mm", "u_mm", "se_mm", "sn_mm", "su_mm")


def _gnss_at_dates(df, dates, half_days: float) -> dict:
    """Mean of each GNSS column over the days within +-half_days of each
    radar date (YYYYMMDD); NaN where no day is that close. Gaps are never
    filled. The sigmas are averaged too, not divided by sqrt(n): GNSS noise
    is coloured (protocol gnss_screening)."""
    df = df.sort_index()
    day = np.asarray(df.index.values.astype("datetime64[D]").astype(np.int64), dtype=float)
    sar = np.array([np.datetime64(f"{d[:4]}-{d[4:6]}-{d[6:8]}", "D") for d in dates]).astype(np.int64).astype(float)
    lo = np.searchsorted(day, sar - half_days, side="left")
    hi = np.searchsorted(day, sar + half_days, side="right")
    out = {}
    for col in GNSS_COLUMNS:
        if col not in df.columns:
            out[col] = np.full(len(dates), np.nan)
            continue
        v = df[col].to_numpy(dtype=float)
        fin = np.isfinite(v)
        cs = np.r_[0.0, np.cumsum(np.where(fin, v, 0.0))]
        cn = np.r_[0, np.cumsum(fin)]
        n = cn[hi] - cn[lo]
        out[col] = np.where(n > 0, (cs[hi] - cs[lo]) / np.maximum(n, 1), np.nan)
        if col == "u_mm":
            out["n_days"] = n
    return out


def _velocity_model(attrs: dict, dates, cfg=None) -> dict:
    """The model timeseries2velocity fitted, from velocity.h5's attributes
    (mintpy.timeFunc.*), so the datum's rate is taken exactly as the
    velocity's was: {polynomial, periodic (years), steps (years from the
    first date), use (the dates it used: startDate/endDate and excludeDate,
    a date-list file read from mintpy/ as injection_test does), t_ref (its
    reference date, REF_DATE, in years from the first date; with a
    polynomial of degree 2 or more the velocity is the slope there), note}.
    A straight line on every date when they say nothing."""
    def items(v):
        if v is None:
            return []
        if isinstance(v, bytes):
            v = v.decode()
        s = str(v).strip().strip("[]")
        if s.lower() in ("", "no", "none", "auto", "false"):
            return []
        return [x.strip().strip("'\"") for x in s.replace(";", ",").split(",") if x.strip()]

    def day(d):
        return datetime.strptime(d, "%Y%m%d")

    def yrs(d):
        return (day(d) - day(dates[0])).days / 365.25        # significance.years' axis

    try:
        poly = int(float((items(attrs.get("mintpy.timeFunc.polynomial")) or [1])[0]))
        periodic = [float(x) for x in items(attrs.get("mintpy.timeFunc.periodic"))]
        steps = [yrs(d) for d in items(attrs.get("mintpy.timeFunc.stepDate")) if d.isdigit() and len(d) == 8]
        note = "from velocity.h5 (mintpy.timeFunc)"
    except (TypeError, ValueError):
        poly, periodic, steps = 1, [], []
        note = "velocity.h5's time function could not be read: a straight line assumed"
    other = [k for k in ("exp", "log") if items(attrs.get(f"mintpy.timeFunc.{k}"))]
    if other:
        note += f"; its {' and '.join(other)} terms are not modelled here"
    raw = attrs.get("mintpy.timeFunc.excludeDate")
    try:
        from .injection_test import _exclude_dates
        excl = set(_exclude_dates(cfg, raw)) if cfg is not None and getattr(cfg, "mintpy_dir", None) else None
    except Exception:                                                  # noqa: BLE001
        excl = None
    if excl is None:
        excl = {d for d in items(raw) if d.isdigit() and len(d) == 8}
    use = np.array([d not in excl for d in dates])
    for key, keep in (("startDate", lambda d, x: d >= x), ("endDate", lambda d, x: d <= x)):
        lim = [d for d in items(attrs.get(f"mintpy.timeFunc.{key}")) if d.isdigit() and len(d) == 8]
        if lim:
            use &= np.array([keep(d, lim[0]) for d in dates])
    ref = attrs.get("REF_DATE")
    ref = ref.decode() if isinstance(ref, bytes) else (str(ref) if ref is not None else "")
    t_ref = yrs(ref) if ref in dates else 0.0
    return {"polynomial": poly, "periodic": periodic, "steps": steps, "use": use, "t_ref": t_ref, "note": note}


def gnss_station_series(cfg, stations=None, radius_m: float | None = None, match_days: float | None = None,
                        cache_dir=None, frame: str | None = None, plate: str | None = None) -> dict:
    """
    The InSAR and the GNSS line of sight at each station, on the radar dates
    of the delivered time series (timeseries_file), for gnss_absolute and
    for the validation harness.

      InSAR   median of the reliable cells (temporal coherence at the
              configured floor, a fitted velocity) whose centres lie within
              radius_m of the station, at each date with at least
              GNSS_ABS_MIN_CELLS of them; mm, positive towards the satellite.
      GNSS    ground_truth.load_gnss_series (stations, the record's dates
              +-match_days, cache_dir), the mean of the days within
              +-match_days of each radar date, projected onto the median
              per-pixel LOS unit vector of the same cells:
              l_e e + l_n n + l_u u. In the frame of _gnss_in_frame
              (reference_gnss_frame: by default relative to the stations'
              tectonic plate; `frame`/`plate` override the config).
      sigma   per station and date: the projected GNSS sigma and the cells'
              robust spread (1.4826 MAD) in quadrature, at least
              GNSS_SIGMA_FLOOR_MM; NaN where either value is missing.

    Station ids are upper-cased (NGL's). The protocol's further rule of 2
    independent radar pixels per station is the harness's (it has the
    radar ids); this reports the cell count.
    Returns {dates, t, ids, insar_mm, gnss_los_mm, sigma_mm (stations,
    dates), xy_km (stations, 2; local east/north about origin_latlon),
    origin_latlon, stations: [{id, lat, lon, n_cells, los_vector,
    horizontal_los_rate_mm_yr (the straight-line rate of the GNSS east and
    north projected onto the line of sight, in the series' frame: in IGS20
    mostly the plate's motion, plate-fixed what the station moves relative
    to its plate), plate-fixed only: plate_removed_mm_yr {e, n},
    plate_los_rate_mm_yr (the plate's motion on the line of sight that was
    removed) and residual_horizontal_mm_yr; dates_with_both, usable,
    reason}], radius_m, match_days, cache_dir, sources, frame (the
    _gnss_in_frame record; plate-fixed it also has plate_los_ramp, what
    the plate's motion on the line of sight does across the grid relative
    to the stations: _plate_los_ramp)}.
    """
    import h5py
    from . import significance as sg
    ids = [str(s).strip().upper() for s in (stations if stations is not None
                                            else getattr(cfg, "reference_gnss_stations", None) or [])]
    # once each: a station listed twice would be scored leave-one-out on a datum fitted with its twin
    ids = list(dict.fromkeys(ids))
    if not ids:
        raise ValueError("no GNSS stations named (reference_gnss_stations)")
    radius = float(radius_m if radius_m is not None else getattr(cfg, "reference_gnss_radius_m", 100.0) or 100.0)
    days = float(match_days if match_days is not None else getattr(cfg, "reference_gnss_match_days", 3))
    cache = Path(cache_dir or getattr(cfg, "reference_gnss_cache_dir", "") or Path(cfg.project_dir).parent / "gnss")
    vel, attrs = _read(_product(cfg, "velocity.h5"), "velocity")
    good = load_mask(cfg) & np.isfinite(vel)
    ts_path = timeseries_file(cfg)
    with h5py.File(ts_path, "r") as f:
        dates = [d.decode() if isinstance(d, bytes) else str(d) for d in f["date"][:]]
    los = {k: geometry_field(cfg, k) for k in ("los_east", "los_north", "los_up")}
    if any(v is None or v.shape != vel.shape for v in los.values()):
        raise ValueError("this record has no per-pixel line-of-sight vector on the product grid (geometry "
                         "layer), which is needed to project the GNSS onto the line of sight")
    from datetime import timedelta
    d0 = datetime.strptime(dates[0], "%Y%m%d") - timedelta(days=days)
    d1 = datetime.strptime(dates[-1], "%Y%m%d") + timedelta(days=days)
    series = _load_gnss_series(ids, (d0.strftime("%Y-%m-%d"), d1.strftime("%Y-%m-%d")), cache) or {}
    series, frame_rec = _gnss_in_frame(cfg, series, frame, plate)
    missing = getattr(series, "missing", None) or {}
    n_s, n_e = len(ids), len(dates)
    insar = np.full((n_s, n_e), np.nan)
    gl = np.full((n_s, n_e), np.nan)
    gs = np.full((n_s, n_e), np.nan)
    spread = np.full((n_s, n_e), np.nan)
    pos = np.full((n_s, 2), np.nan)
    meta, sources = [], {}
    t_yr = sg.years(dates)
    with h5py.File(ts_path, "r") as f:
        cube = f["timeseries"]
        for i, sid in enumerate(ids):
            m = {"id": sid, "usable": False}
            meta.append(m)
            df = series.get(sid)
            if df is None or len(df) == 0:
                m["reason"] = missing.get(sid) or "no GNSS series for this station in the record's dates"
                continue
            sources[sid] = {k: df.attrs.get(k) for k in ("source_url", "steps_removed") if k in df.attrs}
            try:
                lat, lon = float(df.attrs["lat"]), float(df.attrs["lon"])
            except (KeyError, TypeError, ValueError):
                m["reason"] = "the GNSS series carries no station position"
                continue
            pos[i] = (lat, lon)
            m.update(lat=round(lat, 6), lon=round(lon, 6))
            rr, cc = _station_cells(attrs, lat, lon, vel.shape, radius)
            sel = good[rr, cc] if rr.size else np.array([], bool)
            rr, cc = rr[sel], cc[sel]
            m["n_cells"] = int(rr.size)
            if rr.size < sg.GNSS_ABS_MIN_CELLS:
                m["reason"] = (f"{rr.size} reliable cell(s) within {radius:g} m (off the grid or on incoherent "
                               f"ground); {sg.GNSS_ABS_MIN_CELLS} are needed")
                continue
            lv = np.array([float(np.nanmedian(los[k][rr, cc])) for k in ("los_east", "los_north", "los_up")])
            if not np.isfinite(lv).all():
                m["reason"] = "no line-of-sight vector at the station's cells"
                continue
            m["los_vector"] = [round(float(x), 5) for x in lv]
            r0, c0 = int(rr.min()), int(cc.min())
            block = np.asarray(cube[:, r0:int(rr.max()) + 1, c0:int(cc.max()) + 1], dtype=float)
            vals = block[:, rr - r0, cc - c0] * 1000.0                  # (dates, cells), mm
            enough = np.isfinite(vals).sum(1) >= sg.GNSS_ABS_MIN_CELLS
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                med = np.nanmedian(vals, axis=1)
                mad = 1.4826 * np.nanmedian(np.abs(vals - med[:, None]), axis=1)
            insar[i] = np.where(enough, med, np.nan)
            spread[i] = np.where(enough, mad, np.nan)
            g = _gnss_at_dates(df, dates, days)
            gl[i] = lv[0] * g["e_mm"] + lv[1] * g["n_mm"] + lv[2] * g["u_mm"]
            hr = sg._rate_coefficient(t_yr, lv[0] * g["e_mm"] + lv[1] * g["n_mm"])
            m["horizontal_los_rate_mm_yr"] = None if hr is None else round(hr, 2)
            pmo = df.attrs.get("plate_motion") if frame_rec.get("applied") else None
            if pmo:
                rm = pmo["removed_mm_yr"]
                m.update(plate_removed_mm_yr=rm, plate_los_rate_mm_yr=round(lv[0] * rm["e"] + lv[1] * rm["n"], 3),
                         residual_horizontal_mm_yr=pmo.get("residual_horizontal_mm_yr"))
            gs[i] = np.sqrt((lv[0] * g["se_mm"]) ** 2 + (lv[1] * g["sn_mm"]) ** 2 + (lv[2] * g["su_mm"]) ** 2)
            both = np.isfinite(insar[i]) & np.isfinite(gl[i])
            m["dates_with_both"] = int(both.sum())
            if not both.any():
                m["reason"] = "no radar date with both an InSAR value and a GNSS value"
                continue
            m["usable"] = True
    sig = np.sqrt(np.nan_to_num(gs) ** 2 + np.nan_to_num(spread) ** 2)
    sig = np.where(np.isfinite(insar) & np.isfinite(gl), np.maximum(sig, GNSS_SIGMA_FLOOR_MM), np.nan)
    have = np.isfinite(pos[:, 0])
    origin = (float(pos[have, 0].mean()), float(pos[have, 1].mean())) if have.any() else (np.nan, np.nan)
    x, y = sg.local_xy_km(pos[:, 0], pos[:, 1], origin[0], origin[1])
    if frame_rec.get("applied"):
        frame_rec["plate_los_ramp"] = _plate_los_ramp(
            attrs, vel.shape, good, los, frame_rec["plate"]["plate"],
            [m["plate_los_rate_mm_yr"] for m in meta if m.get("usable") and m.get("plate_los_rate_mm_yr") is not None],
            origin)
    return {"dates": dates, "t": t_yr, "ids": ids, "insar_mm": insar, "gnss_los_mm": gl,
            "sigma_mm": sig, "xy_km": np.c_[x, y], "origin_latlon": origin, "stations": meta,
            "radius_m": radius, "match_days": days, "cache_dir": str(cache), "sources": sources,
            "frame": frame_rec}


def _plate_los_ramp(attrs: dict, shape, good: np.ndarray, los: dict, plate: str, at_stations,
                    origin) -> dict | None:
    """
    How far a cell away from the stations is from plate-fixed, for the
    record (never applied). The datum makes the stations plate-fixed, but
    the plate's motion on the line of sight changes across the scene as the
    line of sight does (Stephenson et al. 2022; plate_motion), so a cell
    keeps (plate's LOS rate at the cell - its mean at the stations) unless
    the run's deramp took that out with every other ramp. Evaluated on
    about 20000 reliable cells (every n-th row and column).
    Returns {cells_sampled, every_nth_cell, stations_mean_plate_los_rate_mm_yr,
    min_mm_yr, max_mm_yr, span_mm_yr} or None with nothing to evaluate.
    """
    from . import plate_motion as pm
    vals = [float(v) for v in at_stations if v is not None and np.isfinite(v)]
    if not vals or not good.any():
        return None
    step = max(1, int(np.ceil(np.sqrt(good.sum() / 20000.0))))
    sl = (slice(None, None, step), slice(None, None, step))
    le, ln = np.asarray(los["los_east"])[sl], np.asarray(los["los_north"])[sl]
    sub = good[sl] & np.isfinite(le) & np.isfinite(ln)
    if not sub.any():
        return None
    ll = (origin[1], origin[0]) if np.isfinite(origin[0]) else None
    lat, lon = _grid_latlon(attrs, shape, lonlat=ll)
    pr = pm.plate_los_rate(le[sub], ln[sub], np.asarray(lat)[sl][sub], np.asarray(lon)[sl][sub], plate)
    ref = float(np.mean(vals))
    d = pr - ref
    return {"cells_sampled": int(sub.sum()), "every_nth_cell": step,
            "stations_mean_plate_los_rate_mm_yr": round(ref, 3), "min_mm_yr": round(float(d.min()), 3),
            "max_mm_yr": round(float(d.max()), 3), "span_mm_yr": round(float(np.ptp(d)), 3)}


def _frame_record(cfg, S: dict) -> dict:
    """reference_datum.json 'frame': the frame the absolute rates are in
    (gnss_station_series' frame record), the stations' horizontal motion on
    the line of sight in that frame and, plate-fixed, the plate, the model,
    what was removed per station and the line-of-sight ramp left away from
    the stations."""
    from . import plate_motion as pm
    fr = S.get("frame") or {}
    usable = [m for m in S["stations"] if m.get("usable")]
    los_rates = {m["id"]: m.get("horizontal_los_rate_mm_yr") for m in usable}
    base = {"requested": fr.get("requested"), "station_horizontal_los_rate_mm_yr": los_rates,
            "station_horizontal_los_rate_median_mm_yr": _median_or_none(list(los_rates.values()))}
    if not fr.get("applied"):
        name = ("IGS20, as NGL publishes the GNSS" if fr.get("requested") == "itrf" else
                "IGS20 assumed: the GNSS series did not state their frame, so the plate's motion was not removed")
        return {"name": name, "code": "IGS20", **base, "reason": fr.get("reason"),
                "note": ("The absolute rates include the plate's own motion: the stations' horizontal motion "
                         "projected onto the line of sight (straight-line rate over the record, listed here; "
                         "mostly the plate's). Ground that is stable on its plate reads about the median, not "
                         "zero.")}
    pl = fr.get("plate") or {}
    m = pm.PLATE_MODELS["ITRF2020"]
    ramp = fr.get("plate_los_ramp")
    deramp = str(getattr(cfg, "mintpy_deramp", "") or "no").strip().lower()
    note = (f"Each station's IGS20 series had the {pl.get('name')} plate's rigid motion removed ({m['name']}, the "
            "published ITRF2020 plate motion model), so ground that rides on the plate reads about zero and the "
            "absolute rates are motion relative to the plate. station_horizontal_los_rate_mm_yr is what the "
            "stations still move horizontally, on the line of sight, relative to the plate.")
    if pl.get("boundary_zone") and pl.get("note"):
        note += " " + pl["note"]
    plane = bool(getattr(cfg, "reference_gnss_plane", False))
    if ramp:
        if deramp not in ("no", "none", "false", ""):
            fate = (f"the run's settings ask for a {deramp} deramp, which takes a ramp like this out with every "
                    "other ramp.")
        elif plane:
            # the plate's LOS motion is close to linear across a scene, and the datum's per-date plane is fitted
            # to the stations' GNSS minus InSAR, so it carries this ramp's linear part with it
            fate = ("this run asks for no deramp, but the datum's plane (reference_gnss_plane) takes the linear "
                    "part of it out between the stations; beyond them the plane is extrapolated.")
        else:
            fate = "this run's settings ask for no deramp and no datum plane, so it is in the product."
        note += (f" Away from the stations a cell keeps the change of the plate's line-of-sight motion across the "
                 f"scene, between {ramp['min_mm_yr']:+.2f} and {ramp['max_mm_yr']:+.2f} mm/yr on this grid "
                 "(plate_los_ramp, relative to the stations' mean): " + fate)
    return {"name": (f"plate-fixed: {pl.get('name')} plate ({pl.get('plate')}), {m['name']} rigid motion removed "
                     "from NGL's IGS20 series"),
            "code": fr.get("frame"), **base,
            "plate": {k: pl.get(k) for k in ("plate", "name", "how", "region", "boundary_zone", "candidates")},
            "model": {"name": m["name"], "citation": m["citation"], "source_url": m["source_url"],
                      "sha256": m["sha256"], "residual_rms_mm_yr": dict(zip("en", m["residual_rms_mm_yr"].get(
                          pl.get("plate"), (None, None))))},
            "station_plate_removed_mm_yr": {s["id"]: s.get("plate_removed_mm_yr") for s in usable},
            "station_plate_los_rate_mm_yr": {s["id"]: s.get("plate_los_rate_mm_yr") for s in usable},
            "station_residual_horizontal_mm_yr": {s["id"]: s.get("residual_horizontal_mm_yr") for s in usable},
            "plate_los_ramp": ramp, "deramp_setting": deramp, "datum_plane": plane, "note": note}


def gnss_absolute_reference(cfg, write: bool = True) -> tuple:
    """
    reference_mode gnss_absolute: the GNSS datum of this record and the rate
    it adds to every cell (significance.reference_mode has the trade-offs).

    1. gnss_station_series: the InSAR and the LOS-projected GNSS at the
       configured stations (reference_gnss_stations).
    2. significance.gnss_datum_fit on all of them: one offset per date and
       one constant per station (reference_gnss_plane: a plane per date as
       well, the joint fit).
    3. significance.gnss_absolute_loo: each station scored with the datum
       fitted without it. The RMS of those leave-one-out rate errors is the
       datum's 1-sigma rate uncertainty, which every cell shares.
    4. significance.datum_rate with velocity.h5's own time function: the
       rate to add to velocity.h5 (plus the plane's rate times each cell's
       east/north km from the stations' centre).

    Returns (shift, record): shift is a scalar or a (rows, cols) map in
    mm/yr LOS; the record goes to deliverables/reference_datum.json (write).
    Raises ValueError with the reason when fewer than GNSS_ABS_MIN_STATIONS
    stations (GNSS_ABS_MIN_STATIONS_PLANE, 5, with the plane) have both
    values, when the datum has no rate, or when no station can be scored
    leave-one-out (its uncertainty would be unknown).
    """
    from . import significance as sg
    plane = bool(getattr(cfg, "reference_gnss_plane", False))
    S = gnss_station_series(cfg)
    d = S["gnss_los_mm"] - S["insar_mm"]
    has = np.isfinite(d).any(1)
    need = sg.GNSS_ABS_MIN_STATIONS_PLANE if plane else sg.GNSS_ABS_MIN_STATIONS
    if int(has.sum()) < need:
        why = "; ".join(f"{m['id']}: {m['reason']}" for m in S["stations"] if not m.get("usable"))
        raise ValueError(f"gnss_absolute needs at least {need} stations with an InSAR value and a GNSS value on the "
                         f"same radar dates{' (plane)' if plane else ''}; this record has {int(has.sum())}"
                         + (f" ({why})" if why else ""))
    t = S["t"]
    fit = sg.gnss_datum_fit(t, d, S["sigma_mm"], S["xy_km"], plane=plane)
    loo = sg.gnss_absolute_loo(t, S["insar_mm"], S["gnss_los_mm"], S["sigma_mm"], S["xy_km"], plane=plane,
                               ids=S["ids"])
    vel, attrs = _read(_product(cfg, "velocity.h5"), "velocity")
    vm = _velocity_model(attrs, S["dates"], cfg)
    rate = sg.datum_rate(fit, t, vm["polynomial"], vm["periodic"], vm["use"], steps=vm["steps"], t_ref=vm["t_ref"])
    model_note = vm["note"]
    if rate["offset_mm_yr"] is None:
        raise ValueError(f"the GNSS datum covers {rate['epochs']} radar date(s), too few for a rate")
    if rate["epoch_share"] < 1.0:
        log.warning("The GNSS datum covers %.0f%% of the radar dates velocity.h5 was fitted to: the rate added is "
                    "the datum's over those dates only (reference_datum.json, rate.epoch_share)",
                    100 * rate["epoch_share"])
    shift = float(rate["offset_mm_yr"])
    if plane:
        if rate["east_mm_yr_per_km"] is None or rate["north_mm_yr_per_km"] is None:
            raise ValueError("the GNSS plane covers too few radar dates for a rate (4 stations are needed on a date)")
        lat, lon = _grid_latlon(attrs, vel.shape, lonlat=S["origin_latlon"][::-1])
        gx, gy = sg.local_xy_km(lat, lon, *S["origin_latlon"])
        shift = (shift + rate["east_mm_yr_per_km"] * gx + rate["north_mm_yr_per_km"] * gy).astype(np.float64)
    sigma = loo["summary"]["rate_bias_rms_mm_yr"]
    if sigma is None:
        # without it the datum's error would be left out of every cell's
        # noise, and the absolute flags would claim more than they know
        why = "; ".join(f"{k}: {v['reason']}" for k, v in loo["per_station"].items() if not v.get("scored"))
        raise ValueError("the GNSS datum's uncertainty cannot be measured: no station could be scored on a datum "
                         "fitted without it" + (f" ({why})" if why else "")
                         + (". With the plane each station is scored on a plane fitted to the others, which needs "
                            f"{sg.GNSS_ABS_MIN_PER_EPOCH_PLANE} of them at a date" if plane else ""))
    ids = S["ids"]
    for k, m in enumerate(S["stations"]):
        m["in_fit"] = bool(fit["used"][k])
        m["constant_mm"] = None if not np.isfinite(fit["constants_mm"][k]) else round(float(fit["constants_mm"][k]), 3)
    p = fit.get("plane")
    rec = {
        "reference_mode": "gnss_absolute", "available": True,
        "written": datetime.now().isoformat(timespec="seconds"),
        "what": ("The relative products (mean_velocity_relative, rate_signal_to_noise.tif, the deforming area) are "
                 "ABSOLUTE in this mode: velocity.h5 plus the rate of a GNSS datum fitted at each radar date to "
                 f"{int(fit['used'].sum())} GNSS stations projected onto the line of sight, mm/yr line of sight, "
                 "positive towards the satellite, in the GNSS frame named under 'frame'"
                 + (f" (relative to the {S['frame']['plate'].get('name')} plate)"
                    if (S.get("frame") or {}).get("applied") else "")
                 + ". They are not relative to farmland."),
        "variant": loo["summary"]["variant"],
        "stations": S["stations"], "fixed_station": ids[fit["fixed"]],
        "datum": {"dates": S["dates"], "offset_mm": _nan_list(fit["offset_mm"]),
                  "stations_per_date": [int(x) for x in fit["n_per_epoch"]],
                  "rms_residual_mm": round(fit["rms_residual_mm"], 3), "dof": fit["dof"],
                  "plane": None if p is None else {
                      "method": p["method"], "offset_mm": _nan_list(p["offset_mm"]),
                      "east_mm_per_km": _nan_list(p["east_mm_per_km"], 4),
                      "north_mm_per_km": _nan_list(p["north_mm_per_km"], 4),
                      "reference_date": None if p["reference_epoch"] is None else S["dates"][p["reference_epoch"]],
                      "dates_with_plane": p["epochs"],
                      "origin_latlon": [round(float(v), 6) for v in S["origin_latlon"]]},
                  "how_to_apply": ("add offset_mm (or, with the plane, offset_mm + east_mm_per_km x + "
                                   "north_mm_per_km y, km from origin_latlon) to the InSAR line of sight at each "
                                   "date to put it on the GNSS datum; the time-series products are left on the "
                                   "reference pixel")},
        "rate": {**{k: (round(v, 4) if isinstance(v, float) else v) for k, v in rate.items()},
                 "model_source": model_note},
        "datum_rate_sigma_mm_yr": sigma,
        "datum_rate_sigma_basis": ("RMS of the leave-one-out rate errors (each station's InSAR on the datum fitted "
                                   "without it, minus its GNSS); added in quadrature to every cell's noise"),
        "leave_one_out": loo,
        "settings": {"radius_m": S["radius_m"], "match_days": S["match_days"], "cache_dir": S["cache_dir"],
                     "min_cells": sg.GNSS_ABS_MIN_CELLS, "min_stations_per_date": sg.GNSS_ABS_MIN_PER_EPOCH,
                     "min_stations_per_date_plane": sg.GNSS_ABS_MIN_PER_EPOCH_PLANE,
                     "sigma_floor_mm": GNSS_SIGMA_FLOOR_MM},
        "gnss_sources": S["sources"],
        "frame": _frame_record(cfg, S),
        "caution": ("Absolute rates carry the distributed-scatterer bias that farmland centring cancels (several "
                    "mm/yr on low-coherence farmland)"
                    + (", and what the stations move relative to their plate, plus the plate's line-of-sight "
                       "ramp away from the stations (frame)" if (S.get("frame") or {}).get("applied")
                       else ", and the plate's motion on the line of sight (frame)")
                    + ". With few stations the datum is poorly known: read the leave-one-out figures, not the "
                      "fit's own residuals."),
    }
    if write:
        _write_reference_record(cfg, rec)
    log.info("GNSS datum: %d stations, %+.2f mm/yr LOS added%s; leave-one-out rate error RMS %s mm/yr",
             int(fit["used"].sum()), rate["offset_mm_yr"], " (+ plane)" if plane else "", sigma)
    return shift, rec


def _absolute_field(cfg, rel: np.ndarray, vel: np.ndarray, null: np.ndarray):
    """gnss_absolute: (the rate field on the GNSS datum where the relative
    one is defined, the per-cell noise with the datum's rate uncertainty
    added in quadrature). A failure is recorded and raised, never replaced
    by the farmland field under an absolute label."""
    try:
        shift, rec = gnss_absolute_reference(cfg)
    except Exception as exc:
        _write_reference_record(cfg, {"reference_mode": "gnss_absolute", "available": False,
                                      "written": datetime.now().isoformat(timespec="seconds"),
                                      "reason": f"{type(exc).__name__}: {exc}"})
        raise
    absolute = np.where(np.isfinite(rel), vel * 1000.0 + shift, np.nan)
    s = rec.get("datum_rate_sigma_mm_yr")
    if s:
        null = np.sqrt(null.astype(np.float64) ** 2 + float(s) ** 2).astype(np.float32)
    return absolute, null


def _empirical_context(cfg):
    """(far-field mask, class-matched relative rate in mm/yr, empirical sigma
    map in mm/yr, model, per-cell noise map in mm/yr) on the geocoded grid -
    computed once per export. The noise map is built FROM the per-class
    table written to uncertainty_model.json, so every product that reads
    either gets the same number.

    reference_mode (B1, 8 Oct 2026): 'farmland' is the above, unchanged.
    'stable_polygon': the far field IS the stable polygon (far_field_mask),
    so the relative rate, the spread and the repeatability are all measured
    on it; reference_datum.json records it. 'gnss_absolute': the noise is
    still measured on the farmland's class-matched field, then the relative
    rate is replaced by the rate on the GNSS datum (gnss_absolute_reference)
    where it is defined, and the datum's leave-one-out rate error is added
    to the noise in quadrature. A datum that cannot be fitted raises; the
    farmland field is never passed off under the absolute label."""
    from . import significance as sg
    mode = sg.reference_mode(cfg)                 # a wrong setting fails here, before any work
    # One context per export, keyed on what it is computed from (16 Oct
    # 2026, readiness review QUA-05): the key was the constant "empirical",
    # so a second project in the same process (the dashboard, a test run)
    # got the first project's far field, rates and noise. Now: the project,
    # the reference mode, the products it reads and the settings (the report
    # areas, the coherence threshold, the stable polygon and the GNSS
    # settings all live in cfg) - _empirical_key. Only the newest context is
    # kept: each holds four grid-sized arrays.
    key = _empirical_key(cfg, mode)
    hit = _CACHE.get("empirical")
    if hit is not None and hit[0] == key:
        return hit[1]
    _CACHE.pop("empirical", None)
    if mode == "farmland":
        # this stage's own earlier record from a run in another mode: left in
        # place it would tell a reader (and the harness) that these products
        # are absolute or centred on a polygon when they are on farmland
        try:
            (Path(cfg.project_dir) / "deliverables" / REFERENCE_RECORD).unlink(missing_ok=True)
        except OSError as exc:
            log.warning("Could not remove the earlier %s (%s)", REFERENCE_RECORD, exc)
    vel, attrs = _read(_product(cfg, "velocity.h5"), "velocity")
    tc, _ = _read(_product(cfg, "temporalCoherence.h5"), "temporalCoherence")
    good = load_mask(cfg) & np.isfinite(vel)
    # the report areas other than the whole study area are not comparison
    # ground (E28, as the client report; F7(d), 7 Oct 2026): the per-cell
    # null the report quotes and the rasters divide by is measured without them
    # (stable_polygon: far_field_mask returns the stable polygon instead)
    from .quality_assessment import report_farmland_masks
    far, _, _, fmeta = report_farmland_masks(cfg, good, attrs, wide=False)
    rel, centres = sg.class_matched_field(vel * 1000.0, tc, far)
    model = sg.empirical_rate_model(timeseries_file(cfg), tc, far, cache=sg.uncertainty_cache_path(cfg),
                                    basis=STABLE_BASIS if mode == "stable_polygon" else None)
    emp = sg.empirical_sigma_map(model, tc)
    _, table = sg.farmland_cell_spread(rel, tc, far)
    # the per-cell test: stands out from farmland cells of the same class,
    # and by more than its own rate's repeatability
    null_by_class = _cell_null_table(model, table)
    model = _record_cell_null(cfg, model, table, null_by_class, comparison_ground=fmeta["comparison_ground"])
    if mode == "stable_polygon":
        _write_reference_record(cfg, _stable_record(cfg, fmeta, centres))
    fmeta = None
    kcls = sg.coherence_class(tc)
    null = np.full(tc.shape, np.nan, dtype=np.float32)
    for i, v in null_by_class.items():
        null[kcls == i] = v
    if mode == "gnss_absolute":
        rel, null = _absolute_field(cfg, rel, vel, null)
    _CACHE["empirical"] = (key, (far, rel, emp, model, null))
    return _CACHE["empirical"][1]


def _empirical_key(cfg, mode: str) -> tuple:
    """What _empirical_context is computed from: the project folder, the
    reference mode, the velocity and temporal coherence products (path,
    mtime, size; None when there is none) and the settings (the config's
    repr: every field). The time series is not stamped on its own:
    timeseries_file opens and checks it, too slow for a cache key, and
    MintPy fits the velocity after it, so a new series is a new velocity."""
    def stamp(name):
        try:
            p = _product(cfg, name)
            st = p.stat()
            return str(p), st.st_mtime_ns, st.st_size
        except (OSError, AttributeError, TypeError):           # no product yet: the read fails later, as before
            return None
    return (str(Path(cfg.project_dir).resolve()), mode, stamp("velocity.h5"), stamp("temporalCoherence.h5"),
            repr(cfg))


def export_signal_to_noise(cfg: PipelineConfig, out_dir: Path) -> Path | None:
    """
    Per-cell signal-to-noise of the rate: |rate relative to farmland of the
    same coherence class| divided by the noise a cell of that class carries -
    the larger of how much farmland cells of the class differ and the cell's
    own rate repeatability (significance.py). The same yardstick as the
    points table's significant_95 flag: SNR >= 1.96 is that flag. Writes the
    raster and deliverables/signal_to_noise.json (share of cells above 1, 2
    and 3, against the same share on far-field farmland = chance).
    """
    try:
        far, rel, _, _, null = _empirical_context(cfg)
    except Exception as exc:  # noqa: BLE001
        log.warning("Signal-to-noise not computed (%s)", exc)
        return None
    _, attrs = _read(_product(cfg, "velocity.h5"), "velocity")
    good = load_mask(cfg) & np.isfinite(rel) & np.isfinite(null) & (null > 0)
    snr = np.full(rel.shape, np.nan, dtype=np.float32)
    snr[good] = np.abs(rel[good]) / null[good]
    path = out_dir / "rate_signal_to_noise.tif"
    from .significance import reference_mode
    mode = reference_mode(cfg)
    quantity = "rate signal-to-noise: |class-matched relative rate| / per-cell noise (los_velocity_cell_noise_1sigma)"
    description = ("|rate relative to farmland of the same coherence class| / per-cell noise; "
                   ">= 1.96 is the significant_95 flag")
    definition = ("|rate relative to farmland of the same coherence class| / max(farmland cell spread, the cell's "
                  "rate repeatability)")
    reading = "a share above 2 no larger than farmland's own means the area holds no more movement than chance"
    first_pass = "first pass (before the client report; it rewrites this file from the report's farmland)"
    extra = {}
    if mode == "stable_polygon":
        quantity = quantity.replace("class-matched relative rate", "rate relative to the stable ground")
        description = ("|rate relative to the stable ground of the same coherence class| / per-cell noise; "
                       ">= 1.96 is the significant_95 flag")
        definition = ("|rate relative to the stable ground of the same coherence class| / max(the stable ground's "
                      "cell spread, the cell's rate repeatability)")
        reading = ("a share above 2 no larger than the stable ground's own means the area holds no more movement "
                   "than chance")
        first_pass = "first pass (before the client report, which rewrites this file from its own comparison ground)"
    elif mode == "gnss_absolute":
        quantity = ("rate signal-to-noise: |rate on the GNSS datum (absolute)| / per-cell noise with the datum's rate "
                    "uncertainty (los_velocity_cell_noise_1sigma)")
        description = "|rate on the GNSS datum| / per-cell noise; >= 1.96 is the significant_95 flag"
        definition = ("|rate on the GNSS datum (absolute)| / sqrt(max(farmland cell spread, the cell's rate "
                      "repeatability)^2 + the datum's leave-one-out rate error^2)")
        reading = ("absolute: a cell above 2 moves on the GNSS datum. The farmland share is not a chance rate when "
                   "the whole area moves")
        first_pass = None
        extra = {"reference_mode": f"gnss_absolute (see {REFERENCE_RECORD})"}
    if first_pass is not None:
        extra["farmland"] = first_pass
    _write_geotiff(path, snr, attrs,
                   tags=product_tags(cfg, attrs, quantity, "unitless", reference_ground(cfg),
                                     significant_95_at=1.96, sign="unsigned (an absolute value)", **extra),
                   description=description, units="unitless")
    s_all, s_far = snr[good], snr[good & far]
    stats = {"definition": definition,
             "cells": int(good.sum()), "median": round(float(np.median(s_all)), 2),
             "percent_above": {str(k): round(100 * float((s_all >= k).mean()), 1) for k in (1, 2, 3)},
             "percent_above_on_farmland_by_chance": {str(k): round(100 * float((s_far >= k).mean()), 1) for k in (1, 2, 3)},
             "reading": reading}
    if mode != "farmland":
        stats["reference_mode"] = mode
        stats["relative_to"] = reference_ground(cfg)
    (out_dir / "signal_to_noise.json").write_text(json.dumps(stats, indent=1))
    log.info("Signal-to-noise: median %.2f; %.1f%% of cells >= 2 (farmland by chance %.1f%%)", stats["median"],
             stats["percent_above"]["2"], stats["percent_above_on_farmland_by_chance"]["2"])
    return path


def export_cell_noise(cfg: PipelineConfig, out_dir: Path) -> Path | None:
    """
    The per-cell noise the significance test uses (1 sigma, LOS): per
    coherence class, the larger of the farmland cell spread and the rate's
    repeatability - the denominator of rate_signal_to_noise.tif, cell for
    cell, so |relative rate| / this raster IS that raster. This is the
    uncertainty to quote for a single cell's relative rate; the EMPIRICAL
    raster beside it is repeatability alone and the formal one a floor.
    """
    try:
        _, _, _, model, null = _empirical_context(cfg)
    except Exception as exc:                                  # noqa: BLE001
        log.warning("Per-cell noise raster not written (%s)", exc)
        return None
    _, attrs = _read(_product(cfg, "velocity.h5"), "velocity")
    rate_label, _, divisor = _unit_label(cfg)
    out = (null / divisor).astype(np.float32)
    out[~load_mask(cfg)] = np.nan
    path = out_dir / f"los_velocity_cell_noise_1sigma_{rate_label.replace('/', '_per_')}.tif"
    from .significance import reference_mode
    quantity = ("per-cell 1-sigma noise of the class-matched relative LOS rate: max(farmland cell spread, rate "
                "repeatability) by coherence class")
    extra = {}
    ground = NULL_PRODUCTS_FARMLAND
    mode = reference_mode(cfg)
    if mode == "gnss_absolute":
        quantity = ("per-cell 1-sigma noise of the LOS rate on the GNSS datum: max(farmland cell spread, rate "
                    "repeatability) by coherence class, with the datum's leave-one-out rate error added in quadrature")
        extra = {"reference_mode": f"gnss_absolute (see {REFERENCE_RECORD})"}
    elif mode == "stable_polygon":
        quantity = ("per-cell 1-sigma noise of the LOS rate relative to the stable ground: max(the stable ground's "
                    "cell spread, rate repeatability) by coherence class")
        ground = ("measured at the products stage (uncertainty_model.json) on the stable ground with the report "
                  "areas taken out")
    _write_geotiff(path, out, attrs,
                   tags=product_tags(cfg, attrs, quantity, rate_label, reference_ground(cfg),
                                     sign=SIGN_UNCERTAINTY,
                                     by_class_mm_yr=json.dumps(model.get("cell_null_by_class_label", {})),
                                     farmland=ground, **extra),
                   description=f"Per-cell 1-sigma noise of the relative LOS rate, {rate_label} (the "
                               "signal-to-noise denominator)")
    return path


def export_empirical_uncertainty(cfg: PipelineConfig, out_dir: Path) -> Path | None:
    """
    The repeatability of a single cell's rate: over the far field, by
    coherence class (significance.empirical_rate_model), mapped onto every
    reported cell. Several times the formal raster beside it. It is NOT the
    significance yardstick: fields differ persistently, so a cell's rate can
    be repeatable and still say nothing about motion - the per-cell noise
    raster (export_cell_noise, max of this and the farmland cell spread) is
    the figure to test a cell against. It still excludes the reference pixel
    and the distributed-scatterer bias, which relative, class-matched
    figures cancel.
    """
    try:
        _, _, emp, model, _ = _empirical_context(cfg)
    except Exception as exc:  # noqa: BLE001
        log.warning("Empirical uncertainty not computed (%s)", exc)
        return None
    _, attrs = _read(_product(cfg, "velocity.h5"), "velocity")
    rate_label, _, divisor = _unit_label(cfg)
    out = emp / divisor
    out[~load_mask(cfg)] = np.nan
    path = out_dir / f"los_velocity_uncertainty_1sigma_EMPIRICAL_{rate_label.replace('/', '_per_')}.tif"
    _write_geotiff(path, out.astype(np.float32), attrs,
                   tags=product_tags(cfg, attrs, "1-sigma repeatability of a single cell's LOS rate by "
                                     "coherence class (split-half, far field) - NOT the significance "
                                     "yardstick; see los_velocity_cell_noise_1sigma", rate_label,
                                     "not applicable", sign=SIGN_UNCERTAINTY),
                   description=f"Rate repeatability 1 sigma, {rate_label}; the per-cell noise raster is "
                               "the larger of this and the farmland cell spread")
    log.info("Rate uncertainty (1 sigma, empirical by coherence class): %s", ", ".join(
        f"{r['tc']} {r['empirical_mm_yr']:.1f} mm/yr" for r in model.get("by_class", [])))
    return path


# EGMS Basic/Calibrated per-measurement-point schema. Field names and
# units follow the Copernicus European Ground Motion Service product
# specification (v3), because it is the only PUBLIC, operational,
# per-point convention there is - the commercial vendor publishes none.
# Matching it means our acceptance criteria can be audited against a
# European service rather than against a threshold we invented. This is
# NOT an EGMS product: the field names and units follow it, nothing else.
#
# Units are EGMS's - mm and mm/yr - whatever display_units says. Until
# 5 Oct 2026 the columns were in the delivery's display units (in/yr), so
# anything reading the table with EGMS assumptions was 25.4x off; the
# client-facing inches live in the report, the KMZ and the rasters.
#
# EGMS's own stated target, worth quoting beside ours: mean velocity
# standard deviation 0.7 mm/yr (1 sigma) for points with coherence > 0.7.
EGMS_FIELDS = [
    ("pid", "point identifier"),
    ("radar_pixel_id", "radar pixel the value came from (row x width + col); cells sharing it share one measurement"),
    ("mp_type", "DS = distributed scatterer"),
    ("easting", "m, UTM (EPSG in the header)"),
    ("northing", "m, UTM (EPSG in the header)"),
    ("latitude", "degrees, WGS84 (EPSG:4326), cell centre"),
    ("longitude", "degrees, WGS84 (EPSG:4326), cell centre"),
    ("height", "m, height of the processing DEM at the cell"),
    ("incidence_angle", "degrees from vertical"),
    ("los_east", "LOS direction cosine, ground to satellite"),
    ("los_north", "LOS direction cosine, ground to satellite"),
    ("los_up", "LOS direction cosine, ground to satellite"),
    ("mean_velocity", "mm/yr, against the reference pixel (absolute)"),
    ("mean_velocity_std", "mm/yr, 1 sigma, FORMAL (assumes independent dates) - see the header note"),
    ("acceleration", "mm/yr/yr"),
    ("acceleration_std", "mm/yr/yr"),
    ("seasonality", "mm, annual oscillation amplitude"),
    ("seasonality_std", "mm"),
    ("rmse", "mm, RMS residual about the fitted line"),
    ("temporal_coherence", "unitless"),
    # not EGMS fields: what this site needs to read a point honestly
    ("mean_velocity_relative", "mm/yr, minus the mean of ordinary far-field farmland of the point's coherence "
                               "class (which farmland: the header's 'Relative columns')"),
    ("mean_velocity_std_empirical", "mm/yr, 1 sigma, rate repeatability for the point's coherence class"),
    ("cell_noise_1sigma", "mm/yr, 1 sigma: max(far-field cell spread, repeatability) for the point's "
                          "coherence class - the noise significant_95 tests against"),
    ("significant_95", "1 where |mean_velocity_relative| > 1.96 x cell_noise_1sigma"),
    ("in_aoi", "1 where the cell centre lies inside the configured bbox (the processing box)"),
    # the second quality check, per point (phaselinked-6)
    ("unwrap_consistency", "unitless (0-1): MintPy's network temporal coherence - how well the point's "
                           "interferograms agree with its time series (quality_unwrapping_consistency.tif)"),
    ("fails_second_gate", "1 where unwrap_consistency is below 0.7 (MintPy's reliability threshold; the QA's "
                          "'passes both quality checks'), or below min_unwrap_consistency when that is higher; "
                          "no figure is gated on it unless min_unwrap_consistency > 0"),
]

# The compact attribute table (points_attributes.gpkg): the EGMS fields a
# reader needs, without the empty acceleration/seasonality columns this
# processing does not estimate. The full list stays in the opt-in CSV.
POINT_FIELDS = [f for f, _ in EGMS_FIELDS
                if f not in ("acceleration", "acceleration_std", "seasonality", "seasonality_std")]


def egms_field_units(cfg=None) -> list[tuple[str, str]]:
    """EGMS_FIELDS with the units the files are written in (always mm, mm/yr)."""
    return list(EGMS_FIELDS)


def _point_coords(attrs: dict, rows, cols):
    """
    (lon, lat, easting, northing, utm_epsg) of the CENTRE of each cell.

    lon/lat are always WGS84 and are what the point GEOMETRY is built from.
    easting/northing are UTM metres in a named zone - for a projected product
    its own CRS, for a geographic one the zone of the scene centre. Until
    5 Oct 2026 the UTM metres overwrote the variables the geometry was then
    built from, under an EPSG:4326 tag: every delivered GeoPackage
    (site_b 10.8 GB, site_a_pl 18.7 GB) put its points at
    "longitude" 450,000, off the globe.
    """
    from pyproj import Transformer
    lon, lat = _rc_to_lonlat(attrs, rows, cols)
    lon, lat = np.asarray(lon, dtype=np.float64), np.asarray(lat, dtype=np.float64)
    crs = _crs(attrs)
    if crs == "EPSG:4326":
        zone = int((float(np.nanmean(lon)) + 180) // 6) + 1
        epsg = (32600 if np.nanmean(lat) >= 0 else 32700) + zone
        e, n = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True).transform(lon, lat)
    else:
        epsg = int(crs.split(":")[1])
        x0, dx, y0, dy = _geotransform(attrs)
        e = x0 + (np.asarray(cols, dtype=float) + 0.5) * dx
        n = y0 + (np.asarray(rows, dtype=float) + 0.5) * dy
    return lon, lat, np.asarray(e), np.asarray(n), epsg


def radar_pixel_ids(cfg, rows, cols, shape=None) -> tuple[np.ndarray, str]:
    """
    The radar pixel each geocoded cell's value came from, as row x width +
    col on the radar grid, and a sentence saying how it was found.

    MintPy geocodes by nearest neighbour, so one radar pixel (about 40 x 47 m
    at 11 x 3 looks) fills about 7.5 of the 15 m map cells, and the cell
    count overstates the number of measurements 7.5x (site_b: 2,621,392
    geocoded cells, 348,670 distinct radar pixels). The geocoded
    latitude/longitude layer is itself a nearest-neighbour copy of the radar
    lookup table, so matching its float32 values against
    inputs/geometryRadar.h5 recovers the source pixel EXACTLY - verified on
    site_b: every one of the 2,621,392 cells matched, and every geocoded
    velocity equals its radar pixel's. Products made natively in map
    coordinates (DISP-S1, HyP3) have no radar grid: each cell is its own
    measurement and its id is its product-grid index.
    """
    import h5py
    rows = np.asarray(rows); cols = np.asarray(cols)
    geo = Path(cfg.mintpy_dir) / "geo" / "geo_geometryRadar.h5"
    rad = Path(cfg.mintpy_dir) / "inputs" / "geometryRadar.h5"
    if geo.exists() and rad.exists():
        try:
            with h5py.File(rad, "r") as f:
                if "latitude" in f and "longitude" in f:
                    rla, rlo = f["latitude"][:], f["longitude"][:]
                else:
                    rla = rlo = None
            if rla is not None:
                with h5py.File(geo, "r") as f:
                    gla, glo = f["latitude"][:], f["longitude"][:]
                if shape is not None and gla.shape != tuple(shape):
                    raise ValueError(f"geo_geometryRadar is {gla.shape}, the product {tuple(shape)}")

                def key(la, lo):
                    la = np.ascontiguousarray(la, dtype=np.float32)
                    lo = np.ascontiguousarray(lo, dtype=np.float32)
                    return (la.view(np.uint32).astype(np.uint64) << np.uint64(32)) |                         lo.view(np.uint32).astype(np.uint64)

                rk = key(rla, rlo).ravel()
                order = np.argsort(rk, kind="stable")
                srt = rk[order]
                gk = key(gla[rows, cols], glo[rows, cols])
                pos = np.clip(np.searchsorted(srt, gk), 0, srt.size - 1)
                hit = (srt[pos] == gk) & np.isfinite(gla[rows, cols]) & np.isfinite(glo[rows, cols])
                ids = np.where(hit, order[pos], -1).astype(np.int64)
                miss = int((~hit).sum())
                if miss:
                    log.warning("%d of %d cells could not be traced to a radar pixel (radar_pixel_id -1)",
                                miss, hit.size)
                return ids, (f"radar pixel row x {rla.shape[1]} + col, traced through MintPy's "
                             f"nearest-neighbour geocoding lookup ({rad.name})")
        except Exception as exc:                              # noqa: BLE001
            log.warning("Radar pixel ids not traced (%s) - using the product-grid index", exc)
    width = int(shape[1]) if shape is not None else int(cols.max()) + 1 if cols.size else 1
    return (rows.astype(np.int64) * width + cols.astype(np.int64),
            "product-grid index (no radar lookup table: each cell is its own measurement)")


def accepted_points(cfg: PipelineConfig, vel: np.ndarray) -> np.ndarray:
    """
    The hard acceptance gate: unwrapping validity and coherence.

    Deliberately NOT a precision gate, and the reason is the whole design:
    a phase-unwrapping error produces a LOW `mean_velocity_std`. The
    series is smooth, the linear fit is excellent, the residuals are
    small - and the rate is wrong by whole 2*pi cycles. Sigma is a
    VARIANCE statistic; an unwrapping error is a BIAS. Selecting on sigma
    would preferentially admit exactly the pixels that look cleanest and
    are most wrong.

    So validity gates and precision is reported as an attribute. A pixel
    whose rate is not distinguishable from zero is still a measurement -
    "monitored, no motion detected" is a real and useful finding on a
    subsidence site, and deleting it would throw that away.

    `isfinite(vel)` already encodes the unwrapping-validity requirement:
    the backend only fits a pixel that survived `disp_min_valid_fraction`
    of epochs after connected-component masking.
    """
    return load_mask(cfg) & np.isfinite(vel)


def stable_area_dispersion(cfg: PipelineConfig, vel: np.ndarray,
                           mask: np.ndarray) -> dict:
    """
    An empirical precision figure, to sit beside the formal one.

    `mean_velocity_std` propagates the scatter of the observations about
    the fitted model. It contains NOTHING about the reference point, the
    uncorrected troposphere, or an orbital ramp - all of which are
    systematic and, at this site, larger. Quoting it alone conveys false
    precision and is the obvious thing for a reviewer to attack.

    The scatter of velocity across ground known not to be moving absorbs
    all of that. That needs `stable_area_bbox`; there is no way to infer
    it from the data.

    WHAT THIS DELIBERATELY DOES NOT DO. The first version fell back to
    the least-deforming decile of the AOI when no control area was set.
    That selects pixels whose velocity is near zero BY CONSTRUCTION, so
    its dispersion is near zero by construction too: measured on a real
    run it returned 0.016 in/yr against a formal sigma of 0.228 - it
    understated by fifteen times. A caveat in the field name does not fix
    that, because people read the number and not the caveat. **A figure
    that flatters the product by an order of magnitude is worse than no
    figure**, so when no control area is configured this reports that it
    is unavailable and says what to set.

    What IS reported without a control area is `local_scatter`: the
    typical pixel-to-pixel spread of velocity within small windows. That
    is honest and useful - it bounds the SHORT-WAVELENGTH noise - but it
    is blind to exactly the long-wavelength systematics that matter most,
    so it is never called a precision figure.
    """
    rate_label, _, divisor = _unit_label(cfg)
    vmm = vel * 1000.0 / divisor

    out: dict = {"units": rate_label}

    # Short-wavelength scatter: always computable, never a substitute.
    try:
        v = np.where(mask, vmm, np.nan)
        if np.isfinite(v).sum() >= 100:
            # Windowed std as sqrt(E[x^2] - E[x]^2) over the finite pixels
            # only, with uniform_filter. The generic_filter(np.std) version
            # called a Python function once per pixel - 6 M calls, 39 s on
            # the Site A grid - to compute a number the summary itself
            # labels "NOT a precision figure". This is the same statistic
            # in ~0.2 s, and it ignores NaN neighbours instead of counting
            # them as zero velocity.
            fin = np.isfinite(v)
            z = np.where(fin, v, 0.0).astype(np.float64)
            cnt = uniform_filter(fin.astype(np.float64), size=5, mode="nearest") * 25.0
            s1 = uniform_filter(z, size=5, mode="nearest") * 25.0
            s2 = uniform_filter(z * z, size=5, mode="nearest") * 25.0
            with np.errstate(invalid="ignore", divide="ignore"):
                mean = s1 / cnt
                var = s2 / cnt - mean * mean
            win = np.sqrt(np.clip(var, 0.0, None))
            good = fin & (cnt >= 12)
            if good.any():
                out["local_scatter_1sigma"] = round(
                    float(np.median(win[good])), 4)
                out["local_scatter_note"] = (
                    "pixel-to-pixel spread within 5x5 windows. Bounds the "
                    "short-wavelength noise only - blind to the reference "
                    "point, troposphere and orbital ramp, which are the "
                    "larger terms. NOT a precision figure.")
    except Exception as exc:  # noqa: BLE001
        log.debug("local scatter unavailable: %s", exc)

    if not getattr(cfg, "stable_area_bbox", None):
        out["available"] = False
        out["note"] = (
            "No stable_area_bbox configured, so the empirical precision - "
            "the figure that would include the reference point, the "
            "troposphere and any ramp - cannot be measured. Set it to ground "
            "argued not to be moving. Do NOT substitute velocity_std_formal: "
            "it is a floor, not the uncertainty.")
        return out

    _, attrs = _read(_product(cfg, "velocity.h5"), "velocity")
    x0, dx, y0, dy = _geotransform(attrs)
    from pyproj import Transformer
    tr = Transformer.from_crs("EPSG:4326", _crs(attrs), always_xy=True)
    lo0, la0, lo1, la1 = cfg.stable_area_bbox
    (xa, ya), (xb, yb) = tr.transform(lo0, la0), tr.transform(lo1, la1)
    r, c = np.mgrid[0:vel.shape[0], 0:vel.shape[1]]
    X, Y = x0 + (c + 0.5) * dx, y0 + (r + 0.5) * dy        # cell centres
    sel = (mask & (X >= min(xa, xb)) & (X <= max(xa, xb))
           & (Y >= min(ya, yb)) & (Y <= max(ya, yb)))

    vals = vmm[sel]
    vals = vals[np.isfinite(vals)]
    if vals.size < 30:
        out["available"] = False
        out["note"] = (f"stable_area_bbox contains only {vals.size} accepted "
                       f"points - too few to measure dispersion.")
        return out

    out.update({
        "available": True,
        "basis": "configured stable control area",
        "n_points": int(vals.size),
        "dispersion_1sigma": round(float(np.std(vals)), 4),
        "mean_velocity": round(float(np.mean(vals)), 4),
        "note": ("Includes the reference point, troposphere and ramp. Quote "
                 "the LARGER of this and velocity_std_formal as the minimum "
                 "detectable velocity. A non-zero mean here is a reference-"
                 "frame offset: the control area should read about zero."),
    })
    return out


def export_coverage_curve(cfg: PipelineConfig, out_dir: Path,
                          vendor_coverage_percent: float | None = None,
                          vendor_points: int | None = None) -> Path | None:
    """
    Coverage and point count against coherence threshold.

    A single coverage figure is close to meaningless on its own, because
    the threshold behind it is a choice and the number is violently
    sensitive to it. Measured over Site A: 82.4% of the grid passes
    at 0.4 and 16.9% at 0.5 - a factor of five for one decimal place.
    Publishing the curve instead of a point makes the choice visible and
    lets a reviewer put their own threshold on it.

    Where a vendor figure is supplied it is drawn as a horizontal line.
    That is an AUDIT LINE, not a target: matching someone's coverage by
    tuning a threshold imports an error budget nobody can see. It is
    useful only to say where their number falls on our curve.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tc, _ = _read(_product(cfg, "temporalCoherence.h5"), "temporalCoherence")
    vel, _ = _read(_product(cfg, "velocity.h5"), "velocity")
    fitted = np.isfinite(vel)

    is_disp = getattr(cfg, "backend", "") == "disp"
    observable, obs_source = observable_mask(cfg, fitted.shape)
    if observable is not None:
        denom_label = ("observable (land, seen at least once)" if is_disp
                       else f"imaged pixels ({obs_source})")
    else:
        observable = np.ones_like(fitted, dtype=bool)
        denom_label = "product grid (no observable mask available)"
    n_denom = max(int(observable.sum()), 1)

    thresholds = np.round(np.arange(0.20, 0.95, 0.025), 3)
    pts, cov = [], []
    for th in thresholds:
        k = int(np.sum(fitted & observable & (np.nan_to_num(tc) >= th)))
        pts.append(k)
        cov.append(100.0 * k / n_denom)
    pts, cov = np.array(pts), np.array(cov)

    # Coverage is lost at several gates, not one, and the coherence
    # threshold is only the last of them. Quoting a coverage figure
    # without the breakdown invites the reader to assume a looser
    # coherence cut would recover the rest; usually it would not.
    # The gate NAMES depend on the backend: the DISP-S1 path has a
    # valid-epoch fraction and OPERA's recommended_mask; the MintPy
    # inversion has neither, and printing them on a topsStack run made a
    # geocoding fill artefact read as a quality gate.
    g_obs = int(observable.sum())
    g_fit = int(np.sum(fitted & observable))
    g_qual = int(np.sum(fitted & observable & np.isfinite(tc)))
    ceiling = 100.0 * g_qual / n_denom
    if is_disp:
        gate_fit = (f"has a fitted velocity (valid-epoch >="
                    f"{getattr(cfg, 'disp_min_valid_fraction', float('nan')):.2f})")
        gate_qual = "passes OPERA's recommended_mask"
    else:
        gate_fit = "has an inverted velocity"
        gate_qual = "has a temporal coherence"
    n_reported = int(np.sum(fitted & observable & (np.nan_to_num(tc) >= cfg.min_temporal_coherence)))
    log.info("Coverage is gated in stages, each applied to the survivors of "
             "the last:")
    log.info("  %-44s %6d  %5.1f%%", denom_label, g_obs, 100.0 * g_obs / n_denom)
    log.info("  + %-42s %6d  %5.1f%%", gate_fit, g_fit, 100.0 * g_fit / n_denom)
    log.info("  + %-42s %6d  %5.1f%%  <- the ceiling; no coherence threshold "
             "can exceed it", gate_qual, g_qual, ceiling)
    log.info("  + temporal coherence >= %.2f                %6d  %5.1f%%  "
             "<- what is reported", cfg.min_temporal_coherence, n_reported,
             100.0 * n_reported / n_denom)
    if vendor_coverage_percent and vendor_coverage_percent > ceiling:
        log.warning("The vendor's %.1f%% coverage is ABOVE this run's ceiling "
                    "of %.1f%%, so it cannot be reached by relaxing coherence "
                    "alone.%s", vendor_coverage_percent, ceiling,
                    (" It would also need a lower disp_min_valid_fraction or "
                     "dropping OPERA's own recommended_mask - both buy points "
                     "at the cost of trusting each one less." if is_disp else
                     " The pixels above the ceiling were never imaged or "
                     "never inverted."))

    csv = out_dir / "coverage_vs_threshold.csv"
    with open(csv, "w") as f:
        f.write(f"# coverage denominator: {denom_label} ({n_denom} cells)\n")
        f.write("# gates, each applied to the survivors of the last:\n")
        f.write(f"#   {denom_label} {g_obs} | + {gate_fit} {g_fit} "
                f"| + {gate_qual} {g_qual}\n")
        f.write(f"# ceiling {ceiling:.2f}% - no coherence threshold exceeds it\n")
        f.write(f"# current setting: min_temporal_coherence="
                f"{cfg.min_temporal_coherence}\n")
        if vendor_coverage_percent:
            f.write(f"# vendor reference: {vendor_coverage_percent}% / "
                    f"{vendor_points} points (acceptance criteria not published)\n")
        f.write(f"# software: {_software()}\n")       # (16 Oct 2026, readiness review OUT-01)
        f.write("temporal_coherence_threshold,points,coverage_percent\n")
        for th, k, c in zip(thresholds, pts, cov):
            f.write(f"{th},{k},{c:.2f}\n")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(thresholds, cov, color="#2E63C8", lw=2, label="this product")
    ax.axhline(ceiling, color="#2E63C8", ls="-.", lw=1,
               label=f"ceiling from the gates above ({ceiling:.0f}%)")
    ax.axvline(cfg.min_temporal_coherence, color="#C42A66", ls="--", lw=1.2,
               label=f"our threshold ({cfg.min_temporal_coherence})")
    if vendor_coverage_percent:
        ax.axhline(vendor_coverage_percent, color="#7A7F88", ls=":", lw=1.4,
                   label=f"vendor coverage ({vendor_coverage_percent}%)")
        # Where does the vendor's coverage fall on our curve?
        if cov.max() >= vendor_coverage_percent >= cov.min():
            th_match = float(np.interp(vendor_coverage_percent,
                                       cov[::-1], thresholds[::-1]))
            ax.plot([th_match], [vendor_coverage_percent], "o",
                    color="#7A7F88", ms=6)
            ax.annotate(f"matched at {th_match:.2f}",
                        (th_match, vendor_coverage_percent),
                        textcoords="offset points", xytext=(8, 8), fontsize=8,
                        color="#4A4E57")
            log.info("Vendor coverage of %.1f%% corresponds to a coherence "
                     "threshold of %.2f on this product. That is an audit "
                     "line, not a target - matching it would import an error "
                     "budget we cannot see.",
                     vendor_coverage_percent, th_match)
    ax.set_xlabel("temporal coherence threshold")
    ax.set_ylabel(f"coverage (% of {denom_label})")
    ax.set_title("Coverage is a choice of threshold, not a property of the data",
                 fontsize=11, loc="left")
    ax.grid(alpha=.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    png = out_dir / "coverage_vs_threshold.png"
    fig.savefig(png, dpi=150, metadata=png_metadata())
    plt.close(fig)
    log.info("Wrote %s and %s", csv, png)
    return png


def _n_dates(cfg) -> int:
    import h5py
    with h5py.File(timeseries_file(cfg), "r") as f:
        return int(f["date"].shape[0])


def point_attributes(cfg: PipelineConfig, vel: np.ndarray, attrs: dict,
                     rows: np.ndarray, cols: np.ndarray) -> tuple[dict, dict]:
    """
    Every EGMS_FIELDS column for the accepted cells (rows, cols), in mm and
    mm/yr; a field this run did not estimate is None. Returns (data, meta)
    with meta = {lon, lat, utm_epsg, radar_basis, n_independent (the
    distinct radar resolution cells behind the table - its old name),
    n_points_in_aoi, n_radar_cells_in_aoi}.
    """
    import h5py
    n = int(rows.size)
    lon, lat, east, north, utm_epsg = _point_coords(attrs, rows, cols)

    def _grid(path, name, scale=1.0):
        # _product, not a raw path: rows/cols index the GEOCODED grid, and on
        # the topsStack backend the flat velocity.h5 is still in radar
        # coordinates at a different shape. Indexing one with the other raises
        # an out-of-bounds error that names neither file.
        f = _product(cfg, path)
        if not f.exists():
            return None
        with h5py.File(f, "r") as h:
            if name not in h:
                return None
            arr = h[name][:]
            if arr.shape != vel.shape:
                log.warning("%s/%s is %s but the point grid is %s - skipping "
                            "this column rather than mixing geometries.",
                            f.name, name, arr.shape, vel.shape)
                return None
            return (arr[rows, cols] * scale).astype(np.float32)

    def _geom(name):
        arr = geometry_field(cfg, name)
        if arr is None:
            return None
        if arr.shape != vel.shape:
            log.warning("geometry %s is %s but the point grid is %s - skipping",
                        name, arr.shape, vel.shape)
            return None
        return arr[rows, cols].astype(np.float32)

    ids, id_basis = radar_pixel_ids(cfg, rows, cols, vel.shape)
    mm = 1000.0
    data = {
        "pid": np.arange(1, n + 1, dtype=np.int64),
        "radar_pixel_id": ids,
        # every route here is multilooked or phase-linked: distributed scatterers
        "mp_type": np.full(n, "DS", dtype=object),
        "easting": east, "northing": north,
        "latitude": lat, "longitude": lon,
        "height": _geom("height"),
        "incidence_angle": _geom("incidence"),
        "los_east": _geom("los_east"),
        "los_north": _geom("los_north"),
        "los_up": _geom("los_up"),
        "mean_velocity": (vel[rows, cols] * mm).astype(np.float32),
        "mean_velocity_std": _grid("velocity.h5", "velocityStd", mm),
        "acceleration": _grid("velocity.h5", "acceleration", mm),
        "acceleration_std": _grid("velocity.h5", "accelerationStd", mm),
        "seasonality": _grid("velocity.h5", "annualAmplitude", mm),
        "seasonality_std": _grid("velocity.h5", "annualAmplitudeStd", mm),
        # MintPy's `residue` is sqrt(SUM of squared residuals); the RMS is
        # that over sqrt(number of dates)
        "rmse": _grid("velocity.h5", "residue", mm / np.sqrt(_n_dates(cfg))),
        "temporal_coherence": _grid("temporalCoherence.h5", "temporalCoherence"),
        "in_aoi": _box_mask(attrs, cfg.bbox, vel.shape)[rows, cols].astype(np.int8),
    }
    # Significance against what a point on this ground can actually resolve:
    # its rate relative to farmland of the same coherence class, against the
    # per-cell noise of that class. Testing the absolute rate against the
    # formal sigma flagged 96% of the Site A points as significant - the
    # distributed-scatterer bias read as motion.
    um, um_src = unwrap_consistency_grid(cfg, vel.shape)
    has_um = um is not None
    if um is not None:
        u = um[rows, cols].astype(np.float32)
        data["unwrap_consistency"] = u
        # NaN (not measured) is not evidence of consistency: it fails, as in
        # the QA's reliable_percent
        data["fails_second_gate"] = (np.nan_to_num(u, nan=-1.0) < second_gate_level(cfg)).astype("int8")
        del um
    try:
        _, rel, emp, _, null = _empirical_context(cfg)
        data["mean_velocity_relative"] = rel[rows, cols].astype(np.float32)
        data["mean_velocity_std_empirical"] = emp[rows, cols].astype(np.float32)
        data["cell_noise_1sigma"] = null[rows, cols].astype(np.float32)
        with np.errstate(invalid="ignore"):
            data["significant_95"] = (np.abs(rel[rows, cols]) > 1.96 * null[rows, cols]).astype("int8")
    except Exception as exc:                                  # noqa: BLE001
        log.warning("No empirical significance (%s): significant_95 left empty rather than "
                    "tested against the formal sigma", exc)
    # n_independent keeps its old name for older readers: it is the number of
    # distinct radar resolution cells behind the table, not a count of
    # independent measurements (E25, 7 Oct 2026). The study area's counts
    # (in_aoi = 1) are the ones the report gives.
    in_aoi = data["in_aoi"].astype(bool)
    meta = {"lon": lon, "lat": lat, "utm_epsg": utm_epsg, "radar_basis": id_basis,
            "n_independent": int(np.unique(ids[ids >= 0]).size),
            "n_points_in_aoi": int(in_aoi.sum()),
            "n_radar_cells_in_aoi": int(np.unique(ids[in_aoi & (ids >= 0)]).size)}
    if not has_um:
        meta["second_gate_missing"] = um_src                 # why there is no second-check column (E19)
    return data, meta


def second_gate_level(cfg) -> float:
    """The unwrapping consistency a point must reach to 'pass both quality
    checks': MintPy's 0.7 (quality_assessment.RELIABLE_UNWRAP_CONSISTENCY,
    the QA's reliable_percent), or the configured gate when higher."""
    try:
        from .quality_assessment import RELIABLE_UNWRAP_CONSISTENCY as rel
    except ImportError:                                       # pragma: no cover
        rel = 0.7
    return max(float(getattr(cfg, "min_unwrap_consistency", 0.0) or 0.0), float(rel))


def unwrap_consistency_grid(cfg, shape=None) -> tuple[np.ndarray | None, str]:
    """(MintPy's network temporal coherence on the product grid, its source)
    - geo/geo_temporalCoherence_mintpy.h5 on the phase-linked route, where
    the delivered temporal coherence is MiaplPy's; off it, MintPy's own
    temporalCoherence.h5 IS that layer. (None, why) when absent or on
    another grid."""
    import h5py
    p = Path(cfg.mintpy_dir) / "geo" / "geo_temporalCoherence_mintpy.h5"
    if not p.exists() and str(getattr(cfg, "backend", "")).lower() in ("stack", "local", "hyp3"):
        try:
            p = _product(cfg, "temporalCoherence.h5")
        except Exception:                                     # noqa: BLE001
            return None, "no temporal coherence layer"
    if not p.exists():
        return None, f"{p.name} not on disk (time_series.geocode_extra_layers adds it)"
    try:
        with h5py.File(p, "r") as f:
            if "temporalCoherence" not in f:
                return None, f"{p.name} has no temporalCoherence dataset"
            if shape is not None and tuple(f["temporalCoherence"].shape) != tuple(shape):
                return None, f"{p.name} is on another grid"
            return f["temporalCoherence"][:].astype(np.float32), p.name
    except OSError as exc:
        return None, f"{p.name} not readable ({exc})"


def _chance_percent(cfg) -> float | None:
    """The MEASURED share of far-field farmland cells significant_95 flags by
    chance (summary.json significance_per_cell
    percent_of_far_field_cells_significant_95) - 7.1% on Site B, where the
    header said '~5%' (stats-15)."""
    try:
        far, rel, _, _, null = _empirical_context(cfg)
        with np.errstate(invalid="ignore"):
            flag = np.abs(rel) > 1.96 * null
        return round(100 * float(flag[far].mean()), 1) if far.any() else None
    except Exception as exc:                                  # noqa: BLE001 - the header says where to look
        log.info("Chance share for the points header not measured (%s)", exc)
        return None


# The README block that says which farmland the relative columns are on: it
# starts with RELATIVE_BLOCK_START and runs to the 'Reference point:' line,
# and points_on_report_farmland replaces it when it rewrites the columns
RELATIVE_BLOCK_START = "Relative columns:"
RELATIVE_BLOCK_END = "Reference point:"
# the second quality check per point (phaselinked-6): only when the layer is there (E19)
SECOND_GATE_FIELDS = ("unwrap_consistency", "fails_second_gate")


def _mode_relative_lines(cfg, chance: float | None) -> list[str] | None:
    """The README's relative-columns lines in a non-default reference_mode
    (B1), or None in farmland mode."""
    from .significance import reference_mode
    mode = reference_mode(cfg)
    if mode == "stable_polygon":
        name = Path(str(getattr(cfg, "reference_stable_geojson", "") or "")).name
        return [f"# {RELATIVE_BLOCK_START} mean_velocity_relative = mean_velocity minus the mean of the stable ground",
                f"#   drawn in {name} (reference_mode stable_polygon) for the point's coherence class;",
                "#   cell_noise_1sigma = the noise one cell's relative rate is tested against;",
                "#   significant_95 = 1 where |mean_velocity_relative| > 1.96 x cell_noise_1sigma; "
                + (f"{chance:.1f}% of the stable ground is flagged by chance on this record," if chance is not None
                   else "the share of the stable ground flagged by chance is in summary.json (significance_per_cell),"),
                "#   so only an excess over that share means anything. The stable ground is assumed not to move;",
                f"#   nothing here checks that ({REFERENCE_RECORD})."]
    if mode == "gnss_absolute":
        rec_frame = _recorded_frame(cfg)
        code = str(rec_frame.get("code") or "")
        if code.startswith("plate:"):
            pname = (rec_frame.get("plate") or {}).get("name") or code[6:]
            frame_lines = [f"#   bias that farmland centring would cancel. It is relative to the {pname} plate: the",
                           "#   GNSS had the plate's rigid motion removed (ITRF2020 plate motion model), so ground that",
                           "#   rides on the plate reads about zero (what the stations still move relative to it, and the",
                           f"#   plate's line-of-sight ramp away from them, are under 'frame' in {REFERENCE_RECORD}).",
                           "#   cell_noise_1sigma = the per-cell noise with the"]
        else:
            frame_lines = ["#   bias that farmland centring would cancel, and it is in the GNSS frame (IGS20): the plate's",
                           "#   own motion on the line of sight is in it, so stable ground does not read zero (the size is",
                           f"#   under 'frame' in {REFERENCE_RECORD}). cell_noise_1sigma = the per-cell noise with the"]
        return [f"# {RELATIVE_BLOCK_START} mean_velocity_relative = mean_velocity on the GNSS datum: ABSOLUTE, NOT",
                "#   relative to farmland (reference_mode gnss_absolute; the datum, its stations and their",
                f"#   leave-one-out errors are in {REFERENCE_RECORD}). It still carries the distributed-scatterer",
                *frame_lines,
                "#   datum's leave-one-out rate error added; significant_95 = 1 where |mean_velocity_relative| >",
                "#   1.96 x cell_noise_1sigma: the point moves on the GNSS datum. Where the whole area moves,",
                "#   most points are flagged. The client report does not rewrite these columns in this mode."]
    return None


def _recorded_frame(cfg) -> dict:
    """The 'frame' of deliverables/reference_datum.json (gnss_absolute), or
    {} when there is none to read."""
    try:
        rec = json.loads((Path(cfg.project_dir) / "deliverables" / REFERENCE_RECORD).read_text())
    except (OSError, ValueError):
        return {}
    fr = rec.get("frame") if isinstance(rec, dict) else None
    return fr if isinstance(fr, dict) else {}


def _relative_block(chance: float | None, report: dict | None, cfg=None) -> list[str]:
    """
    The README lines on mean_velocity_relative, cell_noise_1sigma and
    significant_95 (E14, trust review 7 Oct 2026). At export they are on the
    FIRST farmland pass, written before the client report took the movement
    it found out of the farmland; they differ from the report by a constant
    per coherence class (Site A class 0: +0.85 mm/yr line of sight, so
    the points averaged over the whole area read +0.16 mm/yr vertical where
    the report reads -0.83), and say so. `report`: the record
    points_on_report_farmland writes once it has rewritten them on the
    report's farmland ({basis, written, chance_percent, class_centre_shift}).
    `cfg`: in a non-default reference_mode the lines say what the columns
    are relative to instead (_mode_relative_lines); None or farmland mode
    changes nothing.
    """
    mode_head = _mode_relative_lines(cfg, chance) if cfg is not None else None
    if mode_head is not None:
        from .significance import reference_mode
        if reference_mode(cfg) == "gnss_absolute":
            return mode_head
        if report:
            return mode_head + [f"#   Rewritten from the client report ({report.get('basis')}), "
                                f"{report.get('written')}."] \
                + (["#   cell_noise_1sigma: " + report["cell_noise_basis"] + "."]
                   if report.get("cell_noise_basis") else [])
        return mode_head + ["#   Written at the products stage, before the client report; the report may rewrite",
                            "#   them on its own comparison ground."]
    head = [f"# {RELATIVE_BLOCK_START} mean_velocity_relative = mean_velocity minus the mean of ordinary "
            "far-field farmland of the point's",
            "#   coherence class; cell_noise_1sigma = the noise one cell's relative rate is tested against;",
            "#   significant_95 = 1 where |mean_velocity_relative| > 1.96 x cell_noise_1sigma; "
            + (f"{chance:.1f}% of ordinary far-field farmland is flagged by chance on this record," if chance is not None
               else "the share of ordinary farmland flagged by chance is in summary.json (significance_per_cell),"),
            "#   so only an excess over that share means anything. A regional figure needs the farmland",
            "#   spread in site_history.json / quality_assessment.json instead."]
    if report:
        shift = report.get("class_centre_shift_mm_yr") or {}
        null_lines = []
        if report.get("cell_noise_basis"):
            # which farmland the per-cell null was measured on, said apart
            # from the relative rate's (final verification, 7 Oct 2026: the
            # note said all three columns were on the report's farmland)
            null_lines = ["#   cell_noise_1sigma: " + report["cell_noise_basis"] + "."]
        return head + [
            f"#   Farmland: the client report's ({report.get('basis')}), rewritten {report.get('written')}",
            "#   from client_report.json: these three columns agree with rate_signal_to_noise.tif, the",
            "#   client rasters and the report's area figures (mean_velocity_relative / los_up averaged over",
            "#   an area gives about the report's relative vertical rate)."
            + (" Class-centre shift from the first pass (mm/yr, LOS): "
               + ", ".join(f"class {k} {float(v):+.3f}" for k, v in sorted(shift.items())) + "." if shift else "")] \
            + null_lines
    return head + [
        "#   Farmland: the FIRST pass, written before the client report found any movement in the farmland.",
        "#   Once the report takes what it found out of the farmland (client_report.json",
        "#   farmland_second_pass) these three columns can differ from its figures by a constant per",
        "#   coherence class (class_centre_shift_mm_yr) until the report rewrites them; this note then",
        "#   says so (the report rewrites rate_signal_to_noise.tif on its own farmland too)."]


def _second_gate_lines(meta: dict, fields) -> list[str]:
    """E19: the README says what the point table holds about the second
    quality check - nothing about columns it does not have."""
    if all(f in fields for f in SECOND_GATE_FIELDS):
        return []
    why = (meta.get("second_gate_missing") or "the layer is not on the product grid").split(" (")[0].rstrip(".")
    return ["# Unwrapping consistency (the second quality check): not available on this record (" + why + "),",
            "#   so this table has no unwrap_consistency or fails_second_gate column and the QA's 'passed both",
            "#   quality checks' share cannot be recomputed from it (time_series.geocode_extra_layers adds the",
            "#   layer to a finished project; the next products run then writes both columns)."]


def _radar_cells_lines(meta: dict, n: int) -> list[str]:
    """E25 (trust review 7 Oct 2026) in the point README: 'about N radar
    resolution cells', not 'independent measurements' (the README said
    'Independent measurements: 582,856 radar pixels behind 4,485,046 cells'
    beside a report saying they are not independent), and the study area's
    count (in_aoi = 1) beside the whole table's, as the report gives it."""
    head = (f"# Radar resolution cells: about {meta['n_independent']:,} lie behind these {n:,} map cells")
    if meta.get("n_radar_cells_in_aoi") is not None and meta.get("n_points_in_aoi") is not None:
        head += (f"; within the study area (in_aoi = 1) about {meta['n_radar_cells_in_aoi']:,} behind "
                 f"{meta['n_points_in_aoi']:,}, the count the report gives")
    return [head + ".",
            "#   Geocoding is nearest-neighbour, so map cells sharing a radar_pixel_id carry one value, and",
            "#   neighbouring radar cells share ground: they are not independent measurements. Count and",
            "#   average over distinct radar_pixel_id, not rows."]


def _points_header(cfg, attrs: dict, meta: dict, n: int, fields) -> list[str]:
    """The '#' header of the point CSV and the body of points_README.txt."""
    backend, tag = _route(cfg)
    res = resolution_info(cfg, attrs)
    chance = _chance_percent(cfg)
    return [
        f"# Measurement points, one per accepted {res.get('posting_m') or _pixel_metres(attrs):.0f} m map cell. "
        "Field NAMES follow the Copernicus",
        "#   European Ground Motion Service (EGMS) per-point specification; this is NOT an EGMS product.",
        "# UNITS are EGMS's: displacement mm, velocity mm/yr (whatever the report's display units).",
        f"# {SIGN_NOTE}",
        f"# Coordinates: latitude/longitude are WGS84 (EPSG:4326) cell centres; easting/northing are "
        f"UTM metres, EPSG:{meta['utm_epsg']}.",
        f"# {res.get('text', '')}",
        *_radar_cells_lines(meta, n),
        f"#   radar_pixel_id: {meta['radar_basis']}.",
        f"# Acceptance: temporal coherence >= {cfg.min_temporal_coherence} AND a",
        "#   fitted velocity. NOT gated on uncertainty: an unwrapping error yields",
        "#   a LOW std, so selecting on std admits the cells that look cleanest",
        "#   and are most wrong. in_aoi = 1 inside the configured processing box; the",
        "#   rest is the rectangle the radar frame was geocoded to, the noisiest ground.",
        "# mean_velocity is ABSOLUTE (against the reference pixel) and on",
        "#   low-coherence farmland carries a distributed-scatterer bias of up to",
        ("#   ~-12 mm/yr; mean_velocity_relative is on the GNSS datum here and keeps it."
         if reference_ground(cfg).startswith("absolute") else
         "#   ~-12 mm/yr; mean_velocity_relative removes it by coherence class."),
        "# mean_velocity_std is FORMAL (independent dates assumed) and several",
        "#   times too small; mean_velocity_std_empirical is the rate's repeatability",
        "#   for the cell's coherence class; cell_noise_1sigma is the larger of that and",
        "#   the spread of far-field cells of the class (uncertainty_model.json).",
        *_relative_block(chance, None, cfg),
        *_second_gate_lines(meta, fields),
        f"# Reference point: {reference_label(cfg)}",
        f"# Dates: {attrs.get('START_DATE', '?')} to {attrs.get('END_DATE', '?')}; time series in "
        "points_timeseries.parquet, mm from each point's first valid epoch.",
        f"# Route: {backend} ({tag}).",
        f"# Source: {source_line(cfg, attrs)}",
        "# Fields: " + "; ".join(f"{f} [{u}]" for f, u in EGMS_FIELDS if f in fields),
    ]


def _column(data: dict, f: str, n: int):
    v = data.get(f)
    return v if v is not None else np.full(n, np.nan, dtype=np.float32)


def _write_points_gpkg(path: Path, data: dict, fields, meta: dict, header: list[str],
                       grid_bounds, chunk: int = 500_000) -> Path | None:
    """
    The attribute GeoPackage, written in chunks to a .partial file and
    renamed only after reading it back: EPSG:4326, feature count, total
    bounds inside the product grid, and geometry x/y equal to the
    longitude/latitude columns. A file that fails is renamed *.INVALID and
    reported - "Wrote" is logged only for one that passed.
    """
    import pandas as pd
    import pyogrio
    import shapely
    import geopandas as gpd
    lon, lat = meta["lon"], meta["lat"]
    n = int(lon.size)
    tmp = path.with_name(path.stem + ".partial" + path.suffix)      # GDAL wants the .gpkg extension
    if tmp.exists():
        tmp.unlink()
    for i0 in range(0, n, chunk):
        sl = slice(i0, min(n, i0 + chunk))
        df = pd.DataFrame({f: _column(data, f, n)[sl] for f in fields})
        gdf = gpd.GeoDataFrame(df, geometry=shapely.points(lon[sl], lat[sl]), crs="EPSG:4326")
        if i0 == 0:
            pyogrio.write_dataframe(
                gdf, tmp, layer="points", driver="GPKG",
                layer_options={"DESCRIPTION": " ".join(h.lstrip("# ") for h in header[:5])},
                layer_metadata={"README": "\n".join(h.lstrip("#") for h in header),
                                "UNITS": "mm, mm/yr (EGMS)", "SIGN_CONVENTION": SIGN_NOTE})
        else:
            pyogrio.write_dataframe(gdf, tmp, layer="points", driver="GPKG", append=True)
    problem = None
    try:
        info = pyogrio.read_info(tmp, layer="points", force_total_bounds=True, force_feature_count=True)
        crs = str(info.get("crs") or "")
        x0, y0, x1, y1 = (float(v) for v in info["total_bounds"])
        gw, gs, ge, gn = grid_bounds
        tol = 1e-6
        if "4326" not in crs:
            problem = f"layer CRS is {crs!r}, not EPSG:4326"
        elif int(info["features"]) != n:
            problem = f"{info['features']} features written of {n}"
        elif not (gw - tol <= x0 <= x1 <= ge + tol and gs - tol <= y0 <= y1 <= gn + tol):
            problem = (f"points span {x0:.4f}..{x1:.4f} E, {y0:.4f}..{y1:.4f} N, outside the product grid "
                       f"{gw:.4f}..{ge:.4f} E, {gs:.4f}..{gn:.4f} N")
        else:
            head = pyogrio.read_dataframe(tmp, layer="points", max_features=min(n, 5))
            if not (np.allclose(head.geometry.x, head["longitude"]) and np.allclose(head.geometry.y, head["latitude"])):
                problem = "point geometry does not equal the longitude/latitude columns"
    except Exception as exc:                                  # noqa: BLE001
        problem = f"could not be read back ({exc})"
    if problem:
        bad = path.with_name(path.name + ".INVALID")
        tmp.replace(bad)
        log.error("%s FAILED its check (%s) - kept as %s, not delivered.", path.name, problem, bad.name)
        return None
    tmp.replace(path)
    log.info("Wrote %s (%d points, %d fields, %.1f MB; checked: EPSG:4326, bounds inside the grid)",
             path, n, len(fields), path.stat().st_size / 1e6)
    return path


def _write_points_timeseries(path: Path, pid: np.ndarray, meta: dict, rows, cols,
                             dates: list[str], ts: np.ndarray, chunk: int = 100_000) -> Path | None:
    """
    points_timeseries.parquet: GeoParquet (WKB points, lon/lat WGS84), one
    row per point (pid joins it to points_attributes.gpkg), one float32
    column per date, mm from each point's own first valid epoch (_anchors),
    zstd-compressed, written in row groups so memory stays bounded. This is
    the 8-14 GB CSV's time series in a form GIS and pandas both read.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq
    import shapely
    lon, lat = meta["lon"], meta["lat"]
    n = int(lon.size)
    geo = {"version": "1.0.0", "primary_column": "geometry",
           # no "crs" member = OGC:CRS84, i.e. WGS84 longitude/latitude
           "columns": {"geometry": {"encoding": "WKB", "geometry_types": ["Point"],
                                    "bbox": [float(np.nanmin(lon)), float(np.nanmin(lat)),
                                             float(np.nanmax(lon)), float(np.nanmax(lat))]}}}
    schema = pa.schema([pa.field("pid", pa.int64()), pa.field("geometry", pa.binary())]
                       + [pa.field(d, pa.float32()) for d in dates],
                       metadata={b"geo": json.dumps(geo).encode(),
                                 b"units": b"mm, line of sight, from each point's first valid epoch",
                                 b"sign_convention": SIGN_NOTE.encode(),
                                 # the code that wrote it, as the GeoTIFFs' TIFFTAG_SOFTWARE
                                 # (16 Oct 2026, readiness review OUT-01)
                                 b"software": _software().encode()})
    tmp = path.with_name(path.stem + ".partial" + path.suffix)
    with pq.ParquetWriter(tmp, schema, compression="zstd") as w:
        for i0 in range(0, n, chunk):
            sl = slice(i0, min(n, i0 + chunk))
            block = np.asarray(ts[:, rows[sl], cols[sl]], dtype=np.float32)      # (dates, k), metres
            fin = np.isfinite(block)
            first_i = np.argmax(fin, axis=0)
            base = block[first_i, np.arange(block.shape[1])]
            base[~fin.any(axis=0)] = np.nan
            disp = (block - base) * np.float32(1000.0)
            arrays = [pa.array(pid[sl], type=pa.int64()),
                      pa.array(shapely.to_wkb(shapely.points(lon[sl], lat[sl])), type=pa.binary())]
            arrays += [pa.array(disp[j], type=pa.float32(), from_pandas=True) for j in range(len(dates))]
            w.write_table(pa.Table.from_arrays(arrays, schema=schema))
    md = pq.read_metadata(tmp)
    if md.num_rows != n or b"geo" not in (md.metadata or {}):
        bad = path.with_name(path.name + ".INVALID")
        tmp.replace(bad)
        log.error("%s FAILED its check (%d rows of %d) - kept as %s", path.name, md.num_rows, n, bad.name)
        return None
    tmp.replace(path)
    log.info("Wrote %s (%d points x %d dates, %.1f MB)", path, n, len(dates), path.stat().st_size / 1e6)
    return path


# rounding for the opt-in CSV: full float64 digits made it 8-14 GB
_CSV_DECIMALS = {"latitude": 7, "longitude": 7, "easting": 2, "northing": 2, "height": 1,
                 "incidence_angle": 2, "los_east": 4, "los_north": 4, "los_up": 4,
                 "temporal_coherence": 3}


def _write_points_csv(path: Path, data: dict, header: list[str], n: int, rows, cols,
                      dates: list[str] | None, ts: np.ndarray | None, chunk: int = 200_000,
                      omit=()) -> Path:
    """The big per-point CSV (opt-in, points_full_csv): every EGMS field but
    `omit` (the second-check columns of a record without their layer) plus
    one column per date (mm), written in chunks."""
    import pandas as pd
    fields = [f for f, _ in EGMS_FIELDS if f not in set(omit)]
    with open(path, "w") as fh:
        fh.write("\n".join(header) + "\n")
        for i0 in range(0, n, chunk):
            sl = slice(i0, min(n, i0 + chunk))
            cols_ = {}
            for f in fields:
                v = _column(data, f, n)[sl]
                if v.dtype.kind == "f":
                    v = np.round(v, _CSV_DECIMALS.get(f, 2))
                cols_[f] = v
            if dates is not None and ts is not None:
                block = np.asarray(ts[:, rows[sl], cols[sl]], dtype=np.float32)
                fin = np.isfinite(block)
                base = block[np.argmax(fin, axis=0), np.arange(block.shape[1])]
                base[~fin.any(axis=0)] = np.nan
                disp = np.round((block - base) * np.float32(1000.0), 2)
                for j, d in enumerate(dates):
                    cols_[d] = disp[j]
            pd.DataFrame(cols_).to_csv(fh, index=False, header=(i0 == 0))
    log.info("Wrote %s (%d points, %.1f MB)", path, n, path.stat().st_size / 1e6)
    return path


def export_measurement_points(cfg: PipelineConfig, out_dir: Path,
                              include_timeseries: bool = True) -> Path | None:
    """
    The measurement point products: one row per accepted cell.

    This is the defensible artefact. The vendor-format summary sheet sits
    beside it for comparability with the go-by the client already knows,
    but this is the one whose acceptance rule can be pointed at.

    Compact by default (Kyle, 5 Oct 2026):
      points_attributes.gpkg      EPSG:4326 points, ~20 columns (POINT_FIELDS),
                                  mm and mm/yr, in_aoi, radar_pixel_id
      points_timeseries.parquet   GeoParquet, one float32 column per date (mm)
      points_README.txt           fields, units, sign, CRS, reference, source
    The full CSV (measurement_points_egms.csv, 8-14 GB at 2.6-4.5 M points x
    275 dates) only when cfg.points_full_csv. The old 293-column GeoPackage
    (11-19 GB, and with broken geometry) is gone; the two files above carry
    the same data. Returns the attribute GeoPackage (or, if it failed its
    check, whichever product was written).
    """
    vel, attrs = _read(_product(cfg, "velocity.h5"), "velocity")
    mask = accepted_points(cfg, vel)
    n = int(mask.sum())
    if not n:
        log.warning("No points pass the acceptance gate - nothing to export")
        _set_aside(out_dir, POINT_PRODUCT_NAMES + ("points_README.txt",), "points",
                   "no point passed the acceptance gate")
        return None
    rows, cols = np.where(mask)
    data, meta = point_attributes(cfg, vel, attrs, rows, cols)

    # The second quality check's columns only when this record has its layer
    # (E19, 7 Oct 2026): Site A's table carried both, empty in all 4.5 M
    # rows (fails_second_gate typed FLOAT), under a README that described
    # them and pointed to a raster that was not delivered
    absent = [f for f in SECOND_GATE_FIELDS if data.get(f) is None]
    if absent:
        log.info("No unwrapping-consistency layer on this record (%s): the point products leave out %s",
                 meta.get("second_gate_missing") or "not on the product grid", ", ".join(absent))
    missing = [f for f, _ in EGMS_FIELDS if data.get(f) is None and f not in absent]
    if missing:
        # Every other field is written, whether or not this run estimated it. A
        # field that is absent breaks anything reading the file by name; a
        # field that is present and empty says truthfully "not estimated here".
        log.info("Point fields carried as empty in this run (not estimated): %s", ", ".join(missing))
    fields = [f for f in POINT_FIELDS if f not in absent]
    header = _points_header(cfg, attrs, meta, n, fields)
    credits_block = ""
    try:
        from . import credits
        credits_block = "\n" + credits.text_block(cfg) + "\n"
    except Exception:                                         # noqa: BLE001
        pass
    (out_dir / "points_README.txt").write_text(
        "\n".join(h[2:] if h.startswith("# ") else h.lstrip("#") for h in header) + "\n" + credits_block)

    # the product grid's lon/lat extent (cell centres), for the read-back check
    ny, nx = vel.shape
    clon, clat = _rc_to_lonlat(attrs, np.array([0, 0, ny - 1, ny - 1]), np.array([0, nx - 1, 0, nx - 1]))
    grid_bounds = (float(np.min(clon)), float(np.min(clat)), float(np.max(clon)), float(np.max(clat)))
    if _crs(attrs) != "EPSG:4326":          # a rotated UTM grid: its corners do not bound it in lon/lat
        pad = 0.01
        grid_bounds = (grid_bounds[0] - pad, grid_bounds[1] - pad, grid_bounds[2] + pad, grid_bounds[3] + pad)

    # a product that failed is listed in `failed`: the earlier run's copy under
    # the final name would otherwise be packaged beside this run's other file,
    # and the two no longer join on pid
    written, failed = [], []
    try:
        g = _write_points_gpkg(out_dir / "points_attributes.gpkg", data, fields, meta, header, grid_bounds)
        if g:
            written.append(g)
        else:
            failed.append("points_attributes.gpkg")
    except Exception as exc:                                  # noqa: BLE001
        log.error("points_attributes.gpkg not written (%s)", exc)
        failed.append("points_attributes.gpkg")

    dates = ts = None
    if include_timeseries or cfg.points_full_csv:
        dates, ts, _ = _dates_and_series(cfg)
    if include_timeseries:
        try:
            p = _write_points_timeseries(out_dir / "points_timeseries.parquet", data["pid"], meta,
                                         rows, cols, dates, ts)
            if p:
                written.append(p)
            else:
                failed.append("points_timeseries.parquet")
        except Exception as exc:                              # noqa: BLE001
            log.error("points_timeseries.parquet not written (%s)", exc)
            failed.append("points_timeseries.parquet")
    _set_aside(out_dir, failed, "points", "the write failed")

    if cfg.points_full_csv:
        written.append(_write_points_csv(out_dir / "measurement_points_egms.csv", data, header, n,
                                         rows, cols, dates if include_timeseries else None,
                                         ts if include_timeseries else None, omit=absent))
    else:
        log.info("The full per-point CSV is off (points_full_csv: false); points_attributes.gpkg and "
                 "points_timeseries.parquet carry the same data.")
    # products of earlier runs, left alone but named so nobody opens them by mistake
    for old, why in (("measurement_points_egms.gpkg", "its point geometry is UTM metres tagged EPSG:4326 "
                      "(off the globe) - superseded by points_attributes.gpkg"),
                     ("measurement_points_egms.csv", "written by an earlier run (display units, before "
                      "5 Oct 2026) - superseded by points_attributes.gpkg + points_timeseries.parquet")):
        f = out_dir / old
        if f.exists() and f not in written:
            log.warning("%s is from an earlier run and is NOT this run's product: %s. Delete it with "
                        "scripts/cleanup.py when no longer needed.", old, why)
    if failed:
        raise PointProductError(f"Point product(s) not written: {', '.join(failed)} (see the errors above). "
                                "Any earlier copy was moved to deliverables/superseded/; re-run the deliverables.")
    return written[0] if written else None


def _replace_relative_block(text: str, block: list[str], prefix: str = "") -> str:
    """`text` (points_README.txt, or the GeoPackage README with its lines
    left-stripped of '#') with the RELATIVE_BLOCK lines replaced by `block`
    (header lines, '# ' included), each written with `prefix` in front of
    the line's text. A README written before 7 Oct 2026 has no such block:
    its lines from 'significant_95:' to the reference line are replaced.
    Unchanged when neither is found."""
    lines = text.split("\n")
    key = [ln.lstrip("# ").lstrip() for ln in lines]
    i0 = next((i for i, k in enumerate(key) if k.startswith(RELATIVE_BLOCK_START)), None)
    i1 = next((i for i, k in enumerate(key) if k.startswith(RELATIVE_BLOCK_END)), None)
    new = [prefix + (b[2:] if b.startswith("# ") else b.lstrip("#")) for b in block]
    if i0 is None:
        if i1 is None:
            return text
        i0 = next((i for i, k in enumerate(key) if k.startswith("significant_95:")), i1)
    if i1 is None or i1 <= i0:
        return text
    return "\n".join(lines[:i0] + new + lines[i1:])


# where the per-cell null was measured, in the words the point README, the
# GeoPackage record and signal_to_noise.json carry (cell_null_basis)
NULL_PRODUCTS_FARMLAND = ("measured at the products stage (uncertainty_model.json) on far-field farmland with the "
                          "report areas taken out, before the client report takes the movement it finds out of it")
NULL_FROM_PRODUCTS = (NULL_PRODUCTS_FARMLAND + " - the same per-cell null the report, rate_signal_to_noise.tif, the "
                      "uncertainty raster and the KMZ use")
NULL_FROM_REPORT = ("the client report's own per-cell null (client_report.json cell_null_by_class), the one "
                    "rate_signal_to_noise.tif, the uncertainty raster and the KMZ use")


def cell_null_basis(rep: dict | None, model: dict | None) -> str:
    """Which farmland the report's per-cell null (rep['cell_null_by_class'])
    was measured on, in words: the products stage's (uncertainty_model.json,
    before the report's second pass) when the report's table is that one,
    else the report's own. The point table's mean_velocity_relative is on
    the report's second pass while its cell_noise_1sigma came from
    uncertainty_model.json, and signal_to_noise.json said both were on the
    second pass (final verification, 7 Oct 2026)."""
    got = {str(k): v for k, v in ((rep or {}).get("cell_null_by_class") or {}).items() if v is not None}
    um = {str(k): v for k, v in ((model or {}).get("cell_null_by_class") or {}).items() if v is not None}
    try:
        same = bool(got) and all(k in um and abs(float(got[k]) - float(um[k])) <= 0.0015 for k in got)
    except (TypeError, ValueError):
        same = False
    if not got:
        # client_report.cell_sigma_map's fallback
        return "the cells' rate repeatability alone (the report carries no per-cell null)"
    return NULL_FROM_PRODUCTS if same else NULL_FROM_REPORT


def points_on_report_farmland(cfg, rep: dict, ctx=None, null_map: np.ndarray | None = None,
                              out_dir: Path | None = None) -> dict:
    """
    points_attributes.gpkg's relative columns on the client report's
    farmland (E14, trust review 7 Oct 2026). The export writes
    mean_velocity_relative, cell_noise_1sigma and significant_95 before the
    report exists, on the FIRST farmland pass; once the report takes the
    movement it found out of the farmland (client_report.json
    farmland_second_pass) every class centre moves, and the points disagreed
    with the report by a constant per coherence class: Site A class 0
    +0.852 mm/yr LOS (whole-area mean of the points +0.16 mm/yr vertical
    against the report's -0.83, Area A2 -4.58 against -5.64), Site B
    +0.43 / +0.99 / +0.34; significant_95 disagreed with
    rate_signal_to_noise.tif on 21,725 and 48,594 points.

    Rewrites the three columns in place from the report's own fields - the
    relative rate ctx.r_cm (client_report.compute's context after its second
    pass, or a fresh one put on the report's farmland by
    results.report_farmland when ctx is None) and the report's per-cell null
    (`null_map`, client_report.cell_sigma_map of rep['cell_null_by_class'];
    built here when None) - so a point reads what rate_signal_to_noise.tif
    and the report's rasters read; then the README block that says which
    farmland, in points_README.txt and the GeoPackage's metadata, and the
    record (metadata key RELATIVE_FARMLAND). The rows are re-derived from the
    products (accepted_points, the export's own order) and checked against
    the table's count and coordinates: a table from another products run is
    left alone. Cheap: three columns updated by row id, no rewrite of the
    file. Returns {applied, reason | points, basis, written, ...}.

    For client_report.write_rasters to call after its second pass, beside
    rate_signal_to_noise.tif (dl.points_on_report_farmland(cfg, rep, ctx,
    null, D)); callable on its own for a built report (ctx None). Measured
    on the 7 Oct copies: 6.7 s for Site B's 2.58 M points, 11.6 s for
    Site A's 4.49 M; the points then average to the report's relative
    figures (Site A whole area -0.83, Area A1 +0.20, Area A2
    -5.63 against -5.64) and significant_95 agrees with
    rate_signal_to_noise.tif on every point.
    """
    import sqlite3
    import xml.etree.ElementTree as ET
    D = Path(out_dir) if out_dir else Path(cfg.project_dir) / "deliverables"
    path = D / "points_attributes.gpkg"
    if not path.exists():
        return {"applied": False, "reason": f"no {path.name} in {D}"}
    from .significance import reference_mode
    if reference_mode(cfg) == "gnss_absolute":
        # the report is still on farmland; its figures would undo the datum (B1)
        return {"applied": False, "reason": ("reference_mode is gnss_absolute: the table's relative columns are on "
                                             "the GNSS datum and the client report's farmland would replace them")}
    if not rep:
        return {"applied": False, "reason": "no client report: the table stays on the first farmland pass"}
    from . import client_report as cr
    if ctx is None:
        from .panel_history import _Context
        from .results import report_farmland
        ctx = _Context(cfg)
        report_farmland(ctx, rep)
    if list(rep.get("dates") or [])[:1] != list(getattr(ctx, "dates", [None]))[:1] \
            or list(rep.get("dates") or [])[-1:] != list(getattr(ctx, "dates", [None]))[-1:]:
        return {"applied": False, "reason": "the client report is from another record than the products"}
    vel, attrs = _read(_product(cfg, "velocity.h5"), "velocity")
    if vel.shape != ctx.r_cm.shape:
        return {"applied": False, "reason": f"the report's grid {ctx.r_cm.shape} is not the products' {vel.shape}"}
    try:
        model = json.loads((D / "uncertainty_model.json").read_text())
    except (OSError, ValueError):
        model = {}
    null_basis = cell_null_basis(rep, model)
    if null_map is None:
        null_map = cr.cell_sigma_map(rep.get("cell_null_by_class"), ctx.kcls, model)
    rows, cols = np.where(accepted_points(cfg, vel))
    n = int(rows.size)
    con = sqlite3.connect(str(path))

    def _no_geometry_change(*_a):
        raise RuntimeError("a point geometry would change: refused")
    # GDAL's R-tree triggers name these functions (they run only when a
    # point's fid or geometry changes - never here), and plain sqlite3 does
    # not have them: the statements would not even prepare without them
    for fn in ("ST_IsEmpty", "ST_MinX", "ST_MaxX", "ST_MinY", "ST_MaxY"):
        con.create_function(fn, 1, _no_geometry_change)
    try:
        cnt, f0, f1 = con.execute("SELECT count(*), min(fid), max(fid) FROM points").fetchone()
        off = con.execute("SELECT count(*) FROM points WHERE pid IS NOT fid").fetchone()[0]
        if (cnt, f0, f1, off) != (n, 1, n, 0):
            return {"applied": False, "reason": (f"the table holds {cnt:,} points (fid {f0}-{f1}) where this record "
                                                 f"accepts {n:,}: it is from another products run")}
        step = max(1, n // 2000)
        smp = con.execute("SELECT fid, latitude, longitude FROM points WHERE (fid - 1) % ? = 0 OR fid = ?",
                          (step, n)).fetchall()
        ix = np.array([r[0] for r in smp], int) - 1
        lon, lat = _rc_to_lonlat(attrs, rows[ix], cols[ix])
        got = np.array([[r[1], r[2]] for r in smp], float)
        if not np.allclose(got, np.c_[lat, lon], atol=1e-5, rtol=0):
            return {"applied": False, "reason": "the table's coordinates are not this record's accepted cells"}
        rel = ctx.r_cm[rows, cols].astype(np.float32)
        nz = np.asarray(null_map, np.float32)[rows, cols]
        with np.errstate(invalid="ignore"):
            sig = (np.abs(rel) > np.float32(1.96) * nz).astype(np.int8)
        before = int(con.execute("SELECT count(*) FROM points WHERE significant_95 = 1").fetchone()[0])
        # the report's chance share on its own farmland (signal_to_noise.json's)
        far = ctx.far & ctx.good & np.isfinite(ctx.r_cm) & np.isfinite(null_map) & (np.asarray(null_map) > 0)
        with np.errstate(invalid="ignore"):
            chance = (round(100.0 * float((np.abs(ctx.r_cm[far]) > 1.96 * np.asarray(null_map)[far]).mean()), 1)
                      if far.any() else None)
        sp = rep.get("farmland_second_pass") or {}
        ground = rep.get("control_ground_label") or "farmland"
        cut = ([f"the {sp['features_removed']} feature(s) that beat chance"] if sp.get("features_removed") else []) \
            + ([f"the report areas {', '.join(map(str, sp['report_areas_removed']))}"]
               if sp.get("report_areas_removed") else [])
        basis = (f"second pass: {' and '.join(cut) or 'what the report found'} taken out of the {ground}, plus "
                 f"{float(sp.get('buffer_m') or 500):.0f} m" if sp else
                 f"the report took nothing out of the {ground}")
        try:
            from . import release
            nh = release.numbers_hash(rep)
        except Exception:                                     # noqa: BLE001 - the record without the hash
            nh = None
        record = {"basis": basis, "written": datetime.now().isoformat(timespec="seconds"), "numbers_hash": nh,
                  "points": n, "chance_percent": chance,
                  "class_centre_shift_mm_yr": sp.get("class_centre_shift_mm_yr") or {},
                  "cell_noise_basis": null_basis}

        def nn(a):
            return [None if x != x else x for x in a.tolist()]           # NaN -> NULL
        fid = np.arange(1, n + 1)
        with con:                                             # one transaction: all of it or none
            for i0 in range(0, n, 200_000):
                sl = slice(i0, min(n, i0 + 200_000))
                con.executemany("UPDATE points SET mean_velocity_relative = ?, cell_noise_1sigma = ?, "
                                "significant_95 = ? WHERE fid = ?",
                                zip(nn(rel[sl]), nn(nz[sl]), sig[sl].tolist(), fid[sl].tolist()))
            block = _relative_block(chance, record, cfg)
            row = con.execute("SELECT m.id, m.metadata FROM gpkg_metadata m JOIN gpkg_metadata_reference r "
                              "ON r.md_file_id = m.id WHERE r.table_name = 'points'").fetchone()
            if row is not None:
                root = ET.fromstring(row[1])
                md = root.find("Metadata")
                if md is None:
                    md = ET.SubElement(root, "Metadata")
                items = {el.get("key"): el for el in md.findall("MDI")}
                if "README" in items:
                    items["README"].text = _replace_relative_block(items["README"].text or "", block, " ")
                el = items.get("RELATIVE_FARMLAND")
                if el is None:
                    el = ET.SubElement(md, "MDI", key="RELATIVE_FARMLAND")
                el.text = json.dumps(record)
                con.execute("UPDATE gpkg_metadata SET metadata = ? WHERE id = ?",
                            (ET.tostring(root, encoding="unicode"), row[0]))
    finally:
        con.close()
    after = int(sig.sum())
    rd = D / "points_README.txt"
    if rd.exists():
        rd.write_text(_replace_relative_block(rd.read_text(), block))
    sn = D / "signal_to_noise.json"
    if sn.exists():
        # its note said the points were first pass; they are this farmland now
        try:
            st = json.loads(sn.read_text())
            if st.get("points_table_note"):
                st["points_table_note"] = ("points_attributes.gpkg mean_velocity_relative and significant_95 were "
                                           "rewritten on this farmland, with cell_noise_1sigma = the per-cell null "
                                           "this raster divides by "
                                           f"({record['written']}; deliverables.points_on_report_farmland)")
            # the null's own farmland, which is not always the relative rate's
            st["cell_null_basis"] = null_basis
            sn.write_text(json.dumps(st, indent=1))
        except (OSError, ValueError) as exc:
            log.warning("signal_to_noise.json note not updated (%s)", exc)
    log.info("Points on the report's farmland (%s): %d points rewritten; significant_95 %d -> %d (chance on the "
             "report's farmland %s%%)", basis, n, before, after, chance)
    return {"applied": True, **record, "significant_before": before, "significant_after": after}


def export_aliasing_products(cfg, out_dir: Path) -> dict | None:
    """
    deliverables/quality_aliasing_risk.tif and aliasing_risk.json (F2,
    aliasing.export_products, contract C-alias) on this run's product grid,
    after export_extra_layers (it relabels quality_too_fast_to_measure.tif a
    lower bound) and before the event products (they report each panel's
    flagged share). None when the diagnostic is off for this config
    (aliasing_diagnostic; "auto" is off for a client project), so such a
    project is exactly as before; an earlier run's flag files, if any, are
    then moved to superseded/ (nothing is moved or made when there are none),
    so a flag from another inversion is never delivered or read as this
    run's. Never raises: a failure is logged and returned, and the other
    products do not depend on it.
    """
    try:
        from . import aliasing
        if not aliasing.enabled(cfg):
            _set_aside(out_dir, list(getattr(aliasing, "EXPORT_NAMES", ("quality_aliasing_risk.tif",
                                                                        "aliasing_risk.json"))),
                       "aliasing", "aliasing_diagnostic is off for this config")
            return None
        rec = aliasing.export_products(cfg, out_dir)
    except Exception as exc:                                  # noqa: BLE001 - recorded, said in words
        log.error("quality_aliasing_risk.tif not written (%s: %s)", type(exc).__name__, exc)
        return {"error": f"{type(exc).__name__}: {exc}"}
    return rec if isinstance(rec, dict) else None


def export_event_products(cfg, out_dir: Path) -> dict | None:
    """
    deliverables/events/ (F5, events.write_event_products) when the config
    sets event_products: change maps between dates, the largest subsidence
    and its date per event_polygons polygon, profiles along event_lines.
    None when it is not set, and nothing is written, so a project without
    the key is exactly as before; an earlier deliverables/events/ (from
    another inversion), if there is one, is then moved to superseded/. A failure is logged as an error
    and recorded in events.json and summary.json; it does not stop the
    products stage, whose other products do not depend on it.
    """
    from . import events
    folder = Path(out_dir) / events.EVENTS_DIR
    if not getattr(cfg, "event_products", None):
        # an earlier run's event products are from another inversion: never left beside this run's
        moved = events._set_aside(folder) if folder.is_dir() else None
        if moved:
            log.warning("event_products is off: the earlier event products were moved to %s", moved)
        return None
    try:
        man = events.write_event_products(cfg, folder)
    except Exception as exc:                                  # noqa: BLE001 - recorded, said in words
        log.error("Event products not written (%s: %s); see %s", type(exc).__name__, exc, folder / "events.json")
        if folder.is_dir():
            events._set_aside(folder)                 # an earlier run's files never sit beside this failure
        events.record_failure(folder, exc)
        return {"status": "failed", "folder": str(folder), "error": f"{type(exc).__name__}: {exc}"}
    if man is None:
        return None
    return {"status": man["status"], "folder": str(folder), "maps": len(man["maps"]),
            "panels": len(man["panels"]), "profile_lines": len((man.get("profiles") or {}).get("lines") or []),
            "errors": man["errors"]}


def export_all(cfg: PipelineConfig,
               points: list[tuple[float, float]] | None = None) -> Path:
    """Write every deliverable, plus the provenance record, to deliverables/."""
    # a MintPy run that started and did not finish (time_series writes this
    # marker as it moves the previous products aside and removes it once the
    # new ones pass their checks): what is in mintpy/ is neither run's set
    for marker in dict.fromkeys((Path(cfg.mintpy_dir) / RUN_INCOMPLETE,
                                 Path(cfg.project_dir) / "mintpy" / RUN_INCOMPLETE)):
        if marker.exists():
            try:
                m = json.loads(marker.read_text())
            except (OSError, ValueError):
                m = {}
            m = m if isinstance(m, dict) else {}
            raise RuntimeError(
                f"The time-series run started {m.get('started') or '(time not recorded)'} did not finish "
                f"({marker} is still there), so the products in {marker.parent} are incomplete. "
                "Re-run the time series before writing deliverables.")
    out_dir = cfg.project_dir / "deliverables"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not (_product(cfg, "velocity.h5")).exists():
        raise FileNotFoundError(
            f"No velocity.h5 in {cfg.mintpy_dir} - run the time series first.")
    # every Results export was computed from the products this run replaces (E17)
    supersede_results(out_dir)

    export_velocity(cfg, out_dir)
    export_velocity_uncertainty(cfg, out_dir)
    export_empirical_uncertainty(cfg, out_dir)
    export_cell_noise(cfg, out_dir)
    export_signal_to_noise(cfg, out_dir)
    export_vertical_velocity(cfg, out_dir)
    export_quality(cfg, out_dir)
    export_extra_layers(cfg, out_dir)
    aliasing_record = export_aliasing_products(cfg, out_dir)  # None unless the F2 diagnostic is on
    export_cumulative_displacement(cfg, out_dir)
    plot_displacement_series(cfg, out_dir, points=points)
    plot_selected_points(cfg, out_dir)
    if cfg.full_timeseries_kmz:
        export_kmz_points(cfg, out_dir)
    else:
        log.info("MintPy's full time-series KMZ is off (full_timeseries_kmz: false; 4-17 GB on a "
                 "ten-year stack). The client KMZ is the Google Earth product.")
    points_error = None
    try:
        export_measurement_points(cfg, out_dir)
    except PointProductError as exc:
        points_error = exc              # the rest is still written; the stage fails at the end
    export_coverage_curve(cfg, out_dir,
                          vendor_coverage_percent=cfg.vendor_coverage_percent,
                          vendor_points=cfg.vendor_points)
    events_record = export_event_products(cfg, out_dir)       # None unless event_products is set

    stats = summary_stats(cfg)
    if aliasing_record is not None:
        stats["aliasing_risk"] = aliasing_record
    if events_record is not None:
        stats["event_products"] = events_record

    # A tie to an absolute frame, a bound on the reference point, or an
    # explicit statement that neither was possible. Never silence.
    if cfg.gnss_validation:
        try:
            from . import gnss_validation
            stats["gnss"] = gnss_validation.validate(
                cfg, out_dir, radius_km=cfg.gnss_search_radius_km)["verdict"]
        except Exception as exc:  # noqa: BLE001
            log.warning("GNSS validation failed (%s) - the reference point "
                        "remains unbounded and this product is relative "
                        "only.", exc)
            stats["gnss"] = f"not available: {exc}"
    else:
        stats["gnss"] = "skipped (gnss_validation is false)"

    _apply_qa_and_gnss(cfg, stats)          # a rebuild after QA / the GNSS tie picks them up
    # the code that computed these numbers (16 Oct 2026, readiness review
    # OUT-01); refresh_summary keeps it when it copies the QA verdict in
    stats["software"] = _software()
    (out_dir / "summary.json").write_text(json.dumps(stats, indent=2, default=_json_default))
    log.info("Wrote %s", out_dir / "summary.json")

    provenance.write(cfg, out_dir, extra={"summary": stats})

    if points_error is not None:
        raise points_error
    log.info("Deliverables written to %s", out_dir)
    return out_dir
