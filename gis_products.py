"""Stage 3b: turn ISCE2's geocoded output into GIS-ready GeoTIFFs.

ISCE2's products aren't directly viewable in QGIS: the wrapped
interferogram is complex-valued (QGIS can't render complex rasters or
compute phase from them), and the unwrapped product is in radians rather
than a physical unit. This writes plain single-band Float32 GeoTIFFs
alongside them.
"""
import logging
from pathlib import Path

import numpy as np
import rasterio

log = logging.getLogger(__name__)

# Sentinel-1 C-band wavelength (m). One 2*pi fringe ~= 2.8 cm of
# line-of-sight displacement.
S1_WAVELENGTH = 0.0555


def _write_float(path: Path, data: np.ndarray, profile: dict,
                 invalid: np.ndarray | None = None) -> None:
    """nodata is NaN, not 0: a genuine zero phase or zero displacement (the
    reference area, by construction) is a measurement, and nodata=0 hid it.
    `invalid` marks the cells with no data (zero amplitude outside the
    geocoded footprint), which become NaN."""
    profile = dict(profile)
    data = np.array(data, dtype="float32")
    if invalid is not None:
        data[invalid] = np.nan
    profile.update(driver="GTiff", dtype="float32", count=1, nodata=np.nan)
    for key in ("blockxsize", "blockysize"):
        profile.pop(key, None)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data, 1)


def make_wrapped_phase(merged_dir: Path) -> Path | None:
    """Extract phase (-pi..pi) from the complex wrapped interferogram."""
    src_path = merged_dir / "filt_topophase.flat.geo"
    if not src_path.exists():
        log.warning("No wrapped interferogram at %s — skipping", src_path)
        return None

    out_path = merged_dir / "wrapped_phase.tif"
    with rasterio.open(src_path) as src:
        data = src.read(1)
        profile = src.profile
    # ISCE fills outside the geocoded footprint with 0+0j: no amplitude, no phase
    _write_float(out_path, np.angle(data), profile, invalid=(data == 0))
    log.info("Wrote %s", out_path)
    return out_path


def make_los_displacement(merged_dir: Path,
                          wavelength: float = S1_WAVELENGTH) -> Path | None:
    """
    Convert unwrapped phase (radians) to line-of-sight displacement (metres).

    Sign: d = -phase * wavelength / (4 pi), so POSITIVE means motion TOWARD
    the satellite (range decrease) and subsidence reads negative - MintPy's
    convention and deliverables.SIGN_NOTE. (This docstring used to say
    "positive ... away from the satellite", which contradicts the formula.)
    This is LOS only — decomposing into true 3D motion needs both
    ascending and descending tracks.

    Note this inherits any unwrapping errors in the source: where coherence
    collapses (right along a rupture, for instance), SNAPHU can introduce
    jumps of whole 2*pi multiples, which show up here as displacements far
    larger than physically plausible. Check against the coherence layer
    before trusting the numbers.
    """
    src_path = merged_dir / "filt_topophase.unw.geo"
    if not src_path.exists():
        log.warning("No unwrapped interferogram at %s — skipping", src_path)
        return None

    out_path = merged_dir / "los_displacement_m.tif"
    with rasterio.open(src_path) as src:
        # Band 1 is amplitude, band 2 is the unwrapped phase.
        amp = src.read(1)
        phase = src.read(2)
        profile = src.profile
    _write_float(out_path, -phase * wavelength / (4 * np.pi), profile,
                 invalid=~np.isfinite(phase) | (amp == 0))
    log.info("Wrote %s", out_path)
    return out_path


def make_gis_products(pair_dir: Path) -> None:
    """Write both GIS-ready rasters for one processed pair.

    Each raster is tried on its own, and a failure is logged, not raised
    (16 Oct 2026, readiness review TST-03): both used to share one try, so
    an unreadable wrapped interferogram (e.g. its .xml sidecar never
    written) also cost the pair its displacement raster.
    """
    merged_dir = pair_dir / "merged"
    for make in (make_wrapped_phase, make_los_displacement):
        try:
            make(merged_dir)
        except Exception as exc:  # noqa: BLE001
            log.warning("Could not write GIS products for %s (%s): %s",
                        pair_dir.name, make.__name__, exc)
