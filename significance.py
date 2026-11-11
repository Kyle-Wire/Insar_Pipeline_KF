"""
How far a site figure can be trusted, measured from the product itself.

WHY THIS MODULE EXISTS
----------------------
The review of 26 Sep 2026 found that every "+/-" the pipeline quoted was the
ordinary least-squares error of a line fitted to a region's mean series. That
assumes the epochs are independent and the region is the only thing that
varies, and both are false on this ground:

  * the residuals of a region-minus-farmland series are strongly correlated
    from one epoch to the next (lag-1 0.5-0.8), so the white-noise error is
    two to three times too small even as a temporal error;
  * far more importantly, ordinary farmland patches of the same size and
    coherence differ from each other by several mm/yr over the record
    (land cover, crop rotation, the distributed-scatterer bias varying field
    by field). A mine or a parcel is one such patch. Its figure means
    something only if it stands out from that spread.

So the uncertainty of a regional relative rate is measured EMPIRICALLY: the
region's exact footprint is shifted across the far field (every offset,
computed as an FFT correlation) and the class-matched relative rate of each
shifted copy is taken. That set of values is what the same measurement reads
on ground with no mine under it - the null distribution. A region figure is
"distinguishable from farmland" only if it falls in the tails of it.

ONE RULE (review of 5 Oct 2026). The test is taken about the null's own
centre (its median), never about zero: on both projects every null median
was positive (0.2-1.0 mm/yr), so z = value/sd and p folded about zero could
disagree (Site B full area z 1.68 but p 0.42). assess() now returns the
excess over the null median, the empirical 95% threshold of |null - median|,
and the 80%-power detection limit (the smallest sustained shift detected in
at least 4 of 5 placements - about 2.8 sd on a Gaussian null, not the 2 sd
that was printed as "the smallest movement that would have been detected",
which is caught only about half the time). Detected <=> |excess| >= threshold
<=> p < 0.05, so the reading, the threshold and signal/threshold cannot
disagree. A null that rests on fewer than 5 independent placements is not
testable at all (Site B's 103 km2 project zone: ~4), and one on fewer than
10 has its threshold, detection limit and range widened (prediction interval
of a spread measured from n samples).

Trust review, 7 Oct 2026. The decision rule stays symmetric, but the 80%
limit is also given per direction (E5: the null has a long sinking tail, so
sinking is caught sooner than the worse-sign limit says - Site A Specific
AOI 18.3 against 19.7 mm/yr vertical); the null restricted to placements
within 10 km and to placements of the region's own coherence is reported
beside the pooled one as a diagnostic, never a switch (E6); the measured
change uses the hinge model only when its departure from rate x span beats
farmland copies of the same footprint (E10, record_change); and report areas
can be taken out of the comparison ground (E28, far_field_mask exclude).

The class centres that define "relative to farmland" are MEANS (Kyle's
decision, 6 Oct 2026). Medians were tried on 5 Oct: unmapped subsidence in the
far field (Site B features 2 and 3) pulls the means down, but median
centring made ordinary farmland read about -0.8 mm/yr instead of 0. The second
farmland pass removes such features from the comparison ground instead.

The per-pixel rate uncertainty is measured the same way in time: each pixel's
rate is fitted separately to the first and second halves of the record, and
the spread of the difference over the far field, by coherence class, gives
the repeatability of a rate (empirical_rate_model).

Reference modes (B1, 8 Oct 2026). "Relative to farmland" cannot see motion
that covers the whole area: the farmland moves with it. The config key
reference_mode chooses what the figures are relative to: 'farmland' (the
default; everything above, unchanged), 'gnss_absolute' (a datum per date
fitted to three or more GNSS stations projected onto the line of sight,
each station scored with the datum fitted without it; by default the GNSS
is relative to the stations' tectonic plate, reference_gnss_frame), or 'stable_polygon'
(a user-drawn polygon of stable ground replaces the far field). The costs
of each are in reference_mode()'s docstring.

Conventions follow MintPy exactly (mintpy.objects.coord):
  lat/lon -> row/col   floor((lat - Y_FIRST) / Y_STEP + 0.01)
  row/col -> lat/lon   (row + 0.5) * Y_STEP + Y_FIRST     (pixel CENTRE)
and the pixel that carries the datum is REF_Y/REF_X in the product's own
attributes - not the configured lat/lon, which MintPy resolves in radar
coordinates and which lands on a different geocoded pixel (four rows apart on
the Site A record).
"""
import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

CLASS_EDGES = (0.4, 0.6, 0.7, 0.8, 0.9, 1.01)
MM_PER_IN = 25.4
Z95 = 1.959964                  # two-sided 95%
Z80 = 0.841621                  # one-sided 80% power
MIN_TESTABLE_INDEPENDENT = 5    # below this the null is not a measurement of anything
RANGE_INFLATION_BELOW = 10      # prediction-interval widening below this many independent placements


def class_labels() -> list[str]:
    return [f"{lo:.1f}-{min(hi, 1.0):.1f}" for lo, hi in zip(CLASS_EDGES[:-1], CLASS_EDGES[1:])]


def coherence_class(tc: np.ndarray) -> np.ndarray:
    """Index into class_labels() per pixel; -1 below the first edge or NaN."""
    k = np.digitize(np.nan_to_num(tc, nan=-1.0), CLASS_EDGES) - 1
    k[(k < 0) | (k >= len(CLASS_EDGES) - 1)] = -1
    return k


# ---------------------------------------------------------------- geometry
def lalo_to_rc(attrs: dict, lat: float, lon: float) -> tuple[int, int]:
    """MintPy's own geo lat/lon -> row/col (coord.lalo2yx)."""
    return (int(np.floor((lat - float(attrs["Y_FIRST"])) / float(attrs["Y_STEP"]) + 0.01)),
            int(np.floor((lon - float(attrs["X_FIRST"])) / float(attrs["X_STEP"]) + 0.01)))


def rc_to_lalo(attrs: dict, rows, cols):
    """Pixel CENTRE lat/lon of row/col (coord.yx2lalo). X_FIRST/Y_FIRST are the
    outer corner of the first pixel, so a corner-based coordinate sits half a
    cell (7.5 m at 15 m posting) north-west of the cell it describes."""
    rows = np.asarray(rows, dtype=float); cols = np.asarray(cols, dtype=float)
    return ((rows + 0.5) * float(attrs["Y_STEP"]) + float(attrs["Y_FIRST"]),
            (cols + 0.5) * float(attrs["X_STEP"]) + float(attrs["X_FIRST"]))


def reference_pixel(attrs: dict, cfg=None) -> dict:
    """The pixel that actually carries the datum in this product."""
    if "REF_Y" in attrs and "REF_X" in attrs:
        r, c = int(float(attrs["REF_Y"])), int(float(attrs["REF_X"]))
        lat, lon = rc_to_lalo(attrs, r, c)
        return {"row": r, "col": c, "lat": round(float(attrs.get("REF_LAT", lat)), 6),
                "lon": round(float(attrs.get("REF_LON", lon)), 6),
                "source": "REF_Y/REF_X of the product (the pixel MintPy referenced)"}
    if cfg is not None and (getattr(cfg, "reference_lalo", "") or "").strip():
        la, lo = (float(v) for v in cfg.reference_lalo.split(","))
        r, c = lalo_to_rc(attrs, la, lo)
        return {"row": r, "col": c, "lat": la, "lon": lo,
                "source": "configured reference_lalo (product carries no REF_Y/REF_X)"}
    return {}


def pixel_metres(attrs: dict) -> tuple[float, float]:
    """(row, col) pixel size in metres for a geographic or projected grid."""
    dy, dx = abs(float(attrs["Y_STEP"])), abs(float(attrs["X_STEP"]))
    if abs(float(attrs.get("Y_FIRST", 0))) <= 90 and dy < 1:      # degrees
        lat = float(attrs["Y_FIRST"]) + float(attrs["Y_STEP"]) * float(attrs.get("LENGTH", 0)) / 2
        return dy * 111_320.0, dx * 111_320.0 * np.cos(np.radians(lat))
    return dy, dx


def aoi_mask(bbox, attrs: dict, shape) -> np.ndarray:
    """Cells whose CENTRE lies inside bbox = (W, S, E, N)."""
    ny, nx = shape
    lat, _ = rc_to_lalo(attrs, np.arange(ny), 0)
    _, lon = rc_to_lalo(attrs, 0, np.arange(nx))
    w, s, e, n = bbox
    return np.outer((lat >= s) & (lat <= n), (lon >= w) & (lon <= e))


def euclidean_buffer(mask: np.ndarray, attrs: dict, metres: float) -> np.ndarray:
    """mask dilated by a true distance. binary_dilation with the default cross
    element grows an L1 diamond, only 71% of the stated width on diagonals."""
    from scipy.ndimage import distance_transform_edt
    if not mask.any():
        return mask.copy()
    dy, dx = pixel_metres(attrs)
    return distance_transform_edt(~mask, sampling=(dy, dx)) <= metres


_MASKS: dict = {}


def panel_mask(geojson: str, shape, attrs: dict) -> np.ndarray:
    """The mapped panels rasterised onto the grid (cell centres), cached.
    No layer configured ("" / None) is no panels. A layer that IS configured
    but missing raises: an all-False mask would silently count the mines as
    farmland (the far field, the null, every relative figure)."""
    from insar_pipeline.stack_backend import _polygon_mask
    if not geojson:
        return np.zeros(shape, bool)
    if not Path(geojson).exists():
        raise FileNotFoundError(f"mapped-mine layer {geojson} is configured but not on disk: the mines would be "
                                "treated as ordinary ground. Restore the file or correct the path in the config.")
    key = (str(geojson), Path(geojson).stat().st_mtime, tuple(shape),
           *(str(attrs.get(k)) for k in ("X_FIRST", "Y_FIRST", "X_STEP", "Y_STEP")))
    if key not in _MASKS:
        _MASKS[key] = _polygon_mask(Path(geojson), shape, attrs)
    return _MASKS[key].copy()


EXCLUDE_BUFFER_M = 500.0       # far_field_mask(exclude=...): the found-feature cut's width (far_without)


def far_field_mask(cfg, good: np.ndarray, attrs: dict, buffer_m: float | None = None,
                   restrict_to_aoi: bool = True, exclude: np.ndarray | None = None,
                   exclude_buffer_m: float = EXCLUDE_BUFFER_M) -> tuple[np.ndarray, dict]:
    """
    The local control: coherent cells inside the AOI, outside every mapped
    panel plus buffer_m. One definition, used by every figure. (Until 27 Sep
    2026 the far field also took in the geocoded grid outside the AOI - 35% of
    it on the Site A record - which shifted every relative figure by
    about 0.3 mm/yr.)

    `exclude` (E28, 7 Oct 2026): a boolean grid of cells that must not be
    comparison ground - the report areas other than the whole study area -
    taken out, grown by `exclude_buffer_m` (500 m by default, the width the
    second farmland pass cuts round a found feature), of this mask and so of
    the class centres and the null built on it. An area must not be compared
    with itself: Site B's project area (not in the mine layer) had 11.5%
    of its cells in its own farmland and read +1.85 mm/yr vertical relative
    to it; kept out (+500 m) it reads +1.99, and every other Site B
    relative figure moves by about +0.13 (the class centres move). On
    Site A no report area touches the farmland: nothing moves. None or
    an empty mask changes nothing.

    reference_mode (B1, 8 Oct 2026; see reference_mode()): 'farmland', the
    default, is everything above, unchanged. 'stable_polygon' returns the
    user's stable-ground polygon instead (stable_far_field). 'gnss_absolute'
    keeps the farmland here: the farmland is still the ground the noise and
    the nulls are measured on, and the datum comes from GNSS
    (deliverables.gnss_absolute_reference).
    """
    if reference_mode(cfg) == "stable_polygon":
        return stable_far_field(cfg, good, attrs, buffer_m=buffer_m, exclude=exclude,
                                exclude_buffer_m=exclude_buffer_m)
    buf = float(getattr(cfg, "acceptance_far_field_buffer_m", 2000.0) if buffer_m is None else buffer_m)
    gj = getattr(cfg, "deramp_exclude_geojson", "") or ""
    mine = panel_mask(gj, good.shape, attrs)
    far = good & ~euclidean_buffer(mine, attrs, buf)
    aoi = aoi_mask(cfg.bbox, attrs, good.shape)
    if restrict_to_aoi:
        far &= aoi
    # wooded cells are not ordinary ground (landcover.exclude_wooded, 15 Oct
    # 2026): C-band sees the canopy. No land-cover map: unchanged.
    from insar_pipeline import landcover
    far, wood_meta = landcover.exclude_wooded(cfg, far, attrs)
    n_before = int(far.sum())
    if exclude is not None:
        exclude = np.asarray(exclude, dtype=bool)
        if exclude.shape != good.shape:
            raise ValueError(f"exclude mask {exclude.shape} is not on the product grid {good.shape}")
        if exclude.any():
            far &= ~euclidean_buffer(exclude, attrs, float(exclude_buffer_m))
    return far, {"buffer_m": buf, "restricted_to_aoi": restrict_to_aoi, "panel_polygon": gj or None,
                 "n_cells": int(far.sum()), "mine_cells": int((mine & good).sum()), "aoi": aoi, "mine": mine,
                 "excluded_cells": n_before - int(far.sum()),
                 "exclude_buffer_m": float(exclude_buffer_m) if exclude is not None and exclude.any() else None,
                 "wooded": wood_meta}


# ---------------------------------------------------------------- reference modes (B1, 8 Oct 2026)
REFERENCE_MODES = ("farmland", "gnss_absolute", "stable_polygon")
_MODE_ALIASES = {"": "farmland", "farmland_relative": "farmland"}     # gaps/notes.md spells it farmland_relative
# The settings the modes read, with the defaults that keep today's behaviour.
# They are read with getattr, so a config without them is 'farmland';
# config.py declares them (name: (default, what it does)).
REFERENCE_SETTINGS = {
    "reference_mode": ("farmland", "what every relative figure is relative to: farmland (default), "
                                   "gnss_absolute or stable_polygon"),
    "reference_stable_geojson": ("", "stable_polygon: GeoJSON (lon/lat) of ground argued to be stable, e.g. a "
                                     "ring outside the angle of draw"),
    "reference_gnss_stations": ([], "gnss_absolute: the NGL station ids that set the datum (at least 3; "
                                    "at least 5 with the plane, so that each can be left out)"),
    "reference_gnss_plane": (False, "gnss_absolute: fit a plane at each date as well as the offset (needs 4 "
                                    "stations at a date, and a fifth so each can be scored without it)"),
    "reference_gnss_radius_m": (100.0, "gnss_absolute: the InSAR value at a station is the median of the "
                                       "reliable cells within this distance"),
    "reference_gnss_match_days": (3, "gnss_absolute: GNSS days within this many days of a radar date are "
                                     "averaged for that date"),
    "reference_gnss_cache_dir": ("", "gnss_absolute: where the GNSS series are downloaded; empty = the "
                                     "pipeline's GNSS cache beside the project (as gnss_tie)"),
}
# The frame of the gnss_absolute datum (Decision 5, 8 Oct 2026). Kept out of REFERENCE_SETTINGS until
# config.py declares them (test_p1_finish requires every key there to be a config field with the same
# default); read with getattr, so a config without them gets these defaults.
REFERENCE_FRAME_SETTINGS = {
    "reference_gnss_frame": ("plate", "gnss_absolute: the frame of the GNSS datum. 'plate' (default): the "
                                      "stations' series relative to their tectonic plate (the ITRF2020 plate "
                                      "motion model removed), so ground that rides on the plate reads about "
                                      "zero. 'itrf': NGL's IGS20 series as published; the plate's motion on the "
                                      "line of sight is then in every absolute rate"),
    "reference_gnss_plate": ("auto", "gnss_absolute with frame 'plate': the plate. 'auto' takes it from where the "
                                     "stations are, and from their velocities where two plates meet "
                                     "(plate_motion.PLATE_REGIONS); or a code, e.g. NOAM, AUST, EURA, PCFC"),
}
REFERENCE_FRAMES = ("plate", "itrf")
_FRAME_ALIASES = {"": "plate", "igs20": "itrf", "itrf2020": "itrf"}
GNSS_ABS_MIN_STATIONS = 3          # stations with InSAR and GNSS values needed for gnss_absolute at all
GNSS_ABS_MIN_PER_EPOCH = 2         # calibration stations at a date for its offset (protocol_B1.yaml)
GNSS_ABS_MIN_PER_EPOCH_PLANE = 4   # ... for its plane (protocol_B1.yaml offset_plus_plane)
GNSS_ABS_MIN_STATIONS_PLANE = 5    # with the plane: 4 at a date + 1, or no station can be scored leave-one-out
GNSS_ABS_MIN_CELLS = 5             # reliable cells within the radius for a station value (protocol sampling)
STABLE_POLYGON_MIN_CELLS = 200     # class_means' min_n: below it not even a pooled centre is measured
KM_PER_DEG = 111.32                # local east/north coordinates for the plane (equirectangular)


def reference_mode(cfg) -> str:
    """
    What every relative figure is measured against: the config key
    `reference_mode`. Missing or empty means 'farmland' (today's behaviour,
    unchanged). Raises ValueError for anything else.

    THE THREE MODES AND WHAT EACH COSTS

    farmland (default). Each cell minus the mean of far-field ground of the
    same coherence class (far_field_mask, class_matched_field).
      + Cancels the distributed-scatterer bias, which depends on coherence
        (several mm/yr on low-coherence farmland), and anything common to
        the whole area: atmosphere, orbit ramps, the reference pixel.
      + Needs no outside data, and its noise and nulls are measured on the
        same ground it is centred on.
      - Blind to motion that covers the whole area. Where the whole basin
        sinks (groundwater, experiment B) the farmland sinks too and the
        signal cancels (gaps/notes.md 1b).

    gnss_absolute. The InSAR line of sight is tied to GNSS: at each date one
    offset (and, optionally, a plane) is fitted so that the InSAR at three
    or more stations matches their GNSS projected onto the line of sight,
    with one constant per station (gnss_datum_fit). Each station is also
    scored with the datum fitted without it (gnss_absolute_loo), and that
    leave-one-out scatter is the mode's uncertainty.
      + Sees motion of the whole area, in the GNSS frame.
      - The frame (gnss_frame; Decision 5, 8 Oct 2026). By default
        (reference_gnss_frame 'plate') the GNSS is put relative to the
        stations' tectonic plate with the published ITRF2020 plate motion
        model (plate_motion, ground_truth.to_plate_frame), so ground that
        rides on its plate reads about zero. Whatever the stations move
        relative to their plate stays in: in a plate-boundary zone that is
        real motion (Corcoran: 12-16 mm/yr north-west relative to stable
        North America, the Sierra Nevada-Great Valley block), and
        reference_datum.json lists it per station. With 'itrf' the series
        stay in IGS20 as NGL publishes them and the plate's own motion is
        in every absolute rate: its horizontal velocity projected onto the
        line of sight, several mm/yr and up to about 20 (Site A: east
        -14.5 mm/yr x LOS east -0.62 = about +9 mm/yr on an ascending
        track), so stable ground does not read zero. The harness compares
        like with like (GNSS in IGS20 too), so its statistics are not
        affected by either.
      - Plate-fixed at the stations, not at every cell: the plate's motion
        on the line of sight changes across a scene because the line of
        sight does (Stephenson et al. 2022, plate_motion), and a little
        because the plate's velocity does (about 0.2 mm/yr over 30 km in
        Australia, 0.05 in Indiana), so a cell far
        from the stations keeps that difference unless the run's deramp
        or the datum's plane (reference_gnss_plane; its linear part,
        between the stations) took it out. reference_datum.json gives
        its size on the grid
        (frame.plate_los_ramp; with Sentinel-1, 0.3-0.7 mm/yr across a 30 km
        box, the most at Appin ascending, where the plate moves 57 mm/yr).
      - Rests on the InSAR being right at the stations. A station on
        low-coherence ground carries its distributed-scatterer bias into the
        datum, and the bias is no longer cancelled anywhere else. Absolute
        rates over farmland include it.
      - With 3-6 stations the datum is poorly known: the leave-one-out RMSE
        has a 95% interval of about 0.6x-2.4x its value at 6 stations.
      - A station monument can move unlike the ground around it (pumping at
        the well, local compaction). The value at a station is a median over
        100 m, so a single bad cell does not decide it.
      - The plane takes 3 parameters at each date (the offset and two
        slopes). It removes a real regional tilt along with orbit and
        atmospheric ramps. With 4-5 stations it is a sensitivity check, not
        a result (protocol_B1.yaml). Only the CHANGE of tilt since the first
        plane date can be measured; a tilt that never changes is absorbed
        by the station constants (gnss_datum_fit, plane_method).
      - Plate motion: the per-date offset absorbs the translation common to
        the stations. What is left comes from the LOS vector varying across
        the box: about 0.5 mm/yr for 50 mm/yr of plate motion and a change of
        0.01 in the LOS vector.
      - The farmland is still where the per-cell noise is measured
        (far_field_mask is unchanged in this mode), and the time-series
        products stay on the reference pixel. The datum per date is recorded
        (deliverables reference_datum.json) for anyone who needs it.

    stable_polygon. Each cell minus the mean of reliable cells of the same
    coherence class inside a polygon the user draws on ground argued to be
    stable: bedrock, an urban core, or a ring outside a longwall's angle of
    draw (stable_far_field).
      + Sees motion relative to that ground, including motion over the
        whole area of interest. Keeps the coherence-class matching.
      - Right only if the polygon really is stable. Nothing checks this
        against GNSS. A ring drawn inside the angle of draw subtracts the
        subsidence from every figure.
      - A small polygon gives noisy class centres. Classes it lacks borrow
        the nearest measured class's centre (class_means). The null drawn
        inside it needs enough independent placements (the 5-placement rule
        of assess).
      - Ground far from the area of interest differs from it by atmosphere
        and orbit gradients that the farmland, lying around the area, shares.
    """
    raw = getattr(cfg, "reference_mode", None) if cfg is not None else None
    m = str(raw if raw is not None else "farmland").strip().lower()
    m = _MODE_ALIASES.get(m, m)
    if m not in REFERENCE_MODES:
        raise ValueError(f"reference_mode {raw!r} is not one of {', '.join(REFERENCE_MODES)}")
    return m


def gnss_frame(cfg) -> tuple[str, str]:
    """(frame, plate) of the gnss_absolute datum from the config keys
    reference_gnss_frame ('plate', the default, or 'itrf'; 'IGS20' is
    'itrf') and reference_gnss_plate ('auto', the default, or a plate code
    of plate_motion.ITRF2020_PMM, returned upper-case). Raises ValueError
    for anything else."""
    raw = getattr(cfg, "reference_gnss_frame", None) if cfg is not None else None
    f = str(raw if raw is not None else REFERENCE_FRAME_SETTINGS["reference_gnss_frame"][0]).strip().lower()
    f = _FRAME_ALIASES.get(f, f)
    if f not in REFERENCE_FRAMES:
        raise ValueError(f"reference_gnss_frame {raw!r} is not one of {', '.join(REFERENCE_FRAMES)}")
    rawp = getattr(cfg, "reference_gnss_plate", None) if cfg is not None else None
    p = str(rawp if rawp not in (None, "") else REFERENCE_FRAME_SETTINGS["reference_gnss_plate"][0]).strip()
    if p.lower() == "auto":
        return f, "auto"
    from .plate_motion import plate_code
    try:
        return f, plate_code(p)
    except ValueError as exc:
        raise ValueError(f"reference_gnss_plate: {exc}") from None


def check_reference_settings(cfg) -> list[str]:
    """Plain-English problems with the reference settings (empty when they
    are usable), for the config check to print."""
    try:
        mode = reference_mode(cfg)
    except ValueError as exc:
        return [str(exc)]
    out = []
    if mode == "stable_polygon":
        gj = str(getattr(cfg, "reference_stable_geojson", "") or "").strip()
        if not gj:
            out.append("reference_mode is stable_polygon but reference_stable_geojson is empty: give the file "
                       "with the stable ground drawn on it")
        elif not Path(gj).exists():
            out.append(f"reference_stable_geojson {gj} does not exist")
    if mode == "gnss_absolute":
        st = list(getattr(cfg, "reference_gnss_stations", None) or [])
        plane = bool(getattr(cfg, "reference_gnss_plane", False))
        need = GNSS_ABS_MIN_STATIONS_PLANE if plane else GNSS_ABS_MIN_STATIONS
        uniq = {str(x).strip().upper() for x in st}                   # ids as NGL spells them
        if len(uniq) < need:
            out.append(f"reference_mode is gnss_absolute but reference_gnss_stations names {len(uniq)} station(s); "
                       f"at least {need} are needed"
                       + (" for the plane (4 to fit it at a date, and one more so that each station can be "
                          "scored on a plane fitted without it)" if plane else ""))
        if float(getattr(cfg, "reference_gnss_radius_m", 100.0) or 0) <= 0:
            out.append("reference_gnss_radius_m must be more than 0")
        if float(getattr(cfg, "reference_gnss_match_days", 3) or 0) < 0:
            out.append("reference_gnss_match_days cannot be negative")
        try:
            gnss_frame(cfg)
        except ValueError as exc:
            out.append(str(exc))
    return out


def stable_polygon_mask(cfg, shape, attrs: dict) -> np.ndarray:
    """The stable-ground polygon (reference_stable_geojson) on the product
    grid, by cell centre. Raises when it is not configured or not on disk:
    an empty mask would leave nothing to centre on."""
    gj = str(getattr(cfg, "reference_stable_geojson", "") or "").strip()
    if not gj:
        raise ValueError("reference_mode is stable_polygon but reference_stable_geojson is empty: give the file "
                         "with the stable ground drawn on it")
    if not Path(gj).exists():
        raise FileNotFoundError(f"stable-ground polygon {gj} is configured but not on disk. Restore the file or "
                                "correct reference_stable_geojson.")
    return panel_mask(gj, shape, attrs)


def stable_far_field(cfg, good: np.ndarray, attrs: dict, buffer_m: float | None = None,
                     exclude: np.ndarray | None = None,
                     exclude_buffer_m: float = EXCLUDE_BUFFER_M) -> tuple[np.ndarray, dict]:
    """
    The comparison ground in stable_polygon mode: reliable cells inside the
    stable polygon and outside the mapped panels themselves. The panels are
    not grown by the far-field buffer: a ring 1-3 km outside the angle of
    draw would lose most of itself to a 2 km buffer, and the user drew it
    there on purpose. The polygon is used wherever it lies on the product
    grid, inside or outside the area of interest, so restrict_to_aoi does
    not apply. `exclude` (report areas) is cut out as in far_field_mask.

    Returns (mask, meta) with far_field_mask's keys (buffer_m is the
    configured one, not used here) plus reference_mode, stable_polygon and
    the polygon's cell counts. Raises ValueError when the polygon holds
    fewer than STABLE_POLYGON_MIN_CELLS reliable cells: no class centre, not
    even a pooled one, could be measured on it.
    """
    buf = float(getattr(cfg, "acceptance_far_field_buffer_m", 2000.0) if buffer_m is None else buffer_m)
    gj = getattr(cfg, "deramp_exclude_geojson", "") or ""
    mine = panel_mask(gj, good.shape, attrs)
    poly = stable_polygon_mask(cfg, good.shape, attrs)
    far = good & poly & ~mine
    if int(far.sum()) < STABLE_POLYGON_MIN_CELLS:
        raise ValueError(f"the stable-ground polygon {getattr(cfg, 'reference_stable_geojson', '')} holds "
                         f"{int(far.sum())} reliable cells outside the mapped panels; at least "
                         f"{STABLE_POLYGON_MIN_CELLS} are needed to centre on it. Draw it larger or on more "
                         "coherent ground.")
    aoi = aoi_mask(cfg.bbox, attrs, good.shape)
    n_before = int(far.sum())
    if exclude is not None:
        exclude = np.asarray(exclude, dtype=bool)
        if exclude.shape != good.shape:
            raise ValueError(f"exclude mask {exclude.shape} is not on the product grid {good.shape}")
        if exclude.any():
            far &= ~euclidean_buffer(exclude, attrs, float(exclude_buffer_m))
    return far, {"buffer_m": buf, "restricted_to_aoi": False, "panel_polygon": gj or None,
                 "n_cells": int(far.sum()), "mine_cells": int((mine & good).sum()), "aoi": aoi, "mine": mine,
                 "excluded_cells": n_before - int(far.sum()),
                 "exclude_buffer_m": float(exclude_buffer_m) if exclude is not None and exclude.any() else None,
                 "reference_mode": "stable_polygon",
                 "stable_polygon": str(getattr(cfg, "reference_stable_geojson", "")),
                 "stable_polygon_cells": int((poly & good).sum()),
                 "stable_polygon_cells_in_panels": int((poly & good & mine).sum()),
                 "stable_polygon_cells_in_aoi": int((far & aoi).sum())}


def _weights(sigma, shape) -> np.ndarray:
    """1/sigma^2 broadcast to (stations, dates); None = equal weights."""
    if sigma is None:
        return np.ones(shape)
    s = np.asarray(sigma, dtype=float)
    if s.ndim == 1:
        s = s[:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.broadcast_to(1.0 / (s * s), shape).astype(float)


def _longest_record(ok: np.ndarray, t: np.ndarray) -> int:
    """The station whose values span the longest time (then the most dates,
    then the first listed): its constant is fixed to 0."""
    best, key = -1, None
    for s in np.flatnonzero(ok.any(1)):
        e = np.flatnonzero(ok[s])
        k = (float(t[e[-1]] - t[e[0]]), int(e.size), -int(s))
        if key is None or k > key:
            best, key = int(s), k
    return best


def _connected(ok: np.ndarray, start: int) -> np.ndarray:
    """Stations linked to `start` through dates they share (the ones whose
    constants the fit can tell apart from the datum)."""
    sta = np.zeros(ok.shape[0], bool)
    sta[start] = True
    while True:
        ep = ok[sta].any(0)
        new = ok[:, ep].any(1) | sta
        if (new == sta).all():
            return sta
        sta = new


def _prune(ok: np.ndarray, fixed: int, min_n: int) -> np.ndarray:
    """ok restricted to dates with at least min_n stations and to the
    stations linked to `fixed` through shared dates, repeated until stable
    (dropping a group can drop dates, which can split groups). All False
    when the fixed station is left with nothing."""
    ok = ok & (ok.sum(0) >= min_n)[None, :]
    while ok[fixed].any():
        new = ok & _connected(ok, fixed)[:, None]
        new &= (new.sum(0) >= min_n)[None, :]
        if (new == ok).all():
            return ok
        ok = new
    return np.zeros_like(ok)


def _solve_datum(d, w, ok, fixed: int, xy=None, ref: int | None = None):
    """Weighted least squares of d[s, e] = o(e) [+ east(e) x_s + north(e)
    y_s] + b_s over the cells in ok; b_fixed = 0 and, with a plane,
    east(ref) = north(ref) = 0. Returns (offset, constants, east, north,
    residual, dof); east/north are None without xy."""
    n_s, n_e = d.shape
    ei = np.flatnonzero(ok.any(0))
    free = [int(s) for s in np.flatnonzero(ok.any(1)) if s != fixed]
    col_o = np.full(n_e, -1)
    col_o[ei] = np.arange(ei.size)
    nc = ei.size
    col_p = np.full(n_e, -1)
    if xy is not None:
        pe = [e for e in ei if e != ref]
        col_p[pe] = nc + 2 * np.arange(len(pe))
        nc += 2 * len(pe)
    col_s = {s: nc + j for j, s in enumerate(free)}
    nc += len(free)
    rs, re_ = np.nonzero(ok)
    A = np.zeros((rs.size, nc))
    A[np.arange(rs.size), col_o[re_]] = 1.0
    for k, (s, e) in enumerate(zip(rs, re_)):
        if s != fixed:
            A[k, col_s[int(s)]] = 1.0
        if xy is not None and col_p[e] >= 0:
            A[k, col_p[e]] = xy[s, 0]
            A[k, col_p[e] + 1] = xy[s, 1]
    sw = np.sqrt(w[rs, re_])
    x, _, rank, _ = np.linalg.lstsq(A * sw[:, None], d[rs, re_] * sw, rcond=None)
    if rank < nc:
        # lstsq would return a minimum-norm answer without a word: a datum
        # and a tilt that the data do not decide (review 8 Oct 2026: two
        # station groups linked by one station gave -2.2 mm/yr/km for a
        # true 0.5 and an offset rate of +12 for a true -35)
        raise ValueError(
            "the stations do not tie the dates together firmly enough to fit "
            + ("a plane at each date: from one period to the next at least 3 stations not in a line must "
               "carry on, or the tilt of one period cannot be told from the station constants of the next. "
               "Fit without the plane, or add stations that span the periods"
               if xy is not None else
               "one offset per date: a date's stations share no date with the rest")
            + f" ({nc - rank} of {nc} unknowns undetermined)")
    offset = np.full(n_e, np.nan)
    offset[ei] = x[col_o[ei]]
    const = np.full(n_s, np.nan)
    const[fixed] = 0.0
    for s in free:
        const[s] = x[col_s[s]]
    east = north = None
    model = offset[re_] + const[rs]
    if xy is not None:
        east = np.full(n_e, np.nan)
        north = np.full(n_e, np.nan)
        east[ei] = 0.0
        north[ei] = 0.0
        has = col_p[ei] >= 0
        east[ei[has]] = x[col_p[ei[has]]]
        north[ei[has]] = x[col_p[ei[has]] + 1]
        model = model + east[re_] * xy[rs, 0] + north[re_] * xy[rs, 1]
    res = np.full(d.shape, np.nan)
    res[rs, re_] = d[rs, re_] - model
    return offset, const, east, north, res, int(rs.size - nc)


def gnss_datum_fit(t, d, sigma=None, xy=None, plane: bool = False, fixed: int | None = None,
                   min_per_epoch: int = GNSS_ABS_MIN_PER_EPOCH,
                   min_per_epoch_plane: int = GNSS_ABS_MIN_PER_EPOCH_PLANE,
                   plane_method: str = "joint") -> dict:
    """
    The GNSS datum of an InSAR line-of-sight record (protocol_B1.yaml,
    reference.modes.gnss_absolute_loo), by weighted least squares:

        d[c, t] = GNSS_LOS[c, t] - InSAR[c, t] = o(t) + b_c

    One offset o(t) per date and one constant b_c per station. Each
    station's GNSS origin and InSAR reference differ from every other's;
    without b_c the datum would step whenever a station joins or leaves.
    The constant of the station with the longest record is fixed to 0
    (`fixed` overrides). That choice moves o(t) by a constant and nothing
    else. InSAR + o(t) is then the InSAR on the GNSS datum.

    t: (dates,) decimal years. d: (stations, dates) mm, NaN where either
    value is missing. sigma: (stations, dates) or (stations,) mm, or None
    for equal weights (weights 1/sigma^2).

    A date gets an offset only with at least `min_per_epoch` stations.
    Stations that never share a date with the fixed station's group cannot
    be told apart from the datum and are left out (`used` False).

    plane=True adds a plane per date, o_p(t) + e(t) x + n(t) y, on dates
    with at least `min_per_epoch_plane` stations not all in a line. xy:
    (stations, 2) km east and north on any local grid; the field must use
    the same grid. Two ways:
      'joint' (default): offset, plane and station constants in one fit.
        A tilt that never changes cannot be told apart from the station
        constants (each station's GNSS origin is arbitrary), so the tilt is
        fixed to 0 on the first plane date: the plane is the tilt CHANGE
        since then, and its rate is exact.
      'protocol': protocol_B1.yaml offset_plus_plane as written, a plane
        per date fitted to d - b_c with the constants of the offset-only
        fit. Exact when every station has every date. When stations join
        or leave, each constant absorbs the tilt averaged over that
        station's own dates, and the plane's rate is biased (a 0.5
        mm/yr/km tilt read 0.28 in a test with one station joining and one
        leaving). Kept so the harness can reproduce the fixed protocol.
    Dates without a plane have none (NaN), and no datum in plane mode.

    Returns {offset_mm (dates,), constants_mm (stations,), fixed, used,
    n_per_epoch, epochs_with_offset, residual_mm (stations, dates),
    rms_residual_mm, dof, plane: None | {offset_mm, east_mm_per_km,
    north_mm_per_km, constants_mm, epochs, reference_epoch, method,
    rms_residual_mm}}. Raises ValueError when no date has enough stations,
    and (joint plane) when the plane dates are tied together by fewer than 3
    stations not in a line: the tilt of one period cannot then be carried to
    the next, and least squares would return an arbitrary one.
    """
    t = np.asarray(t, dtype=float)
    d = np.asarray(d, dtype=float)
    if d.ndim != 2 or d.shape[1] != t.size:
        raise ValueError(f"d must be (stations, dates) with {t.size} dates; got {d.shape}")
    if plane and plane_method not in ("joint", "protocol"):
        raise ValueError(f"plane_method {plane_method!r} is not 'joint' or 'protocol'")
    n_s, n_e = d.shape
    w = _weights(sigma, d.shape)
    ok0 = np.isfinite(d) & np.isfinite(w) & (w > 0)
    ok = ok0 & (ok0.sum(0) >= min_per_epoch)[None, :]
    if not ok.any():
        raise ValueError(f"no date has {min_per_epoch} stations with both an InSAR and a GNSS value")
    if fixed is None:
        fixed = _longest_record(ok, t)
    elif not ok[fixed].any():
        raise ValueError(f"the fixed station (index {fixed}) has no value on a date with {min_per_epoch} stations")
    ok = _prune(ok, fixed, min_per_epoch)
    if not ok.any():
        raise ValueError("no date has enough stations linked to the station with the longest record")
    offset, const, _, _, res, dof = _solve_datum(d, w, ok, fixed)
    out = {"offset_mm": offset, "constants_mm": const, "fixed": int(fixed), "used": ok.any(1),
           "n_per_epoch": ok.sum(0), "epochs_with_offset": int(ok.any(0).sum()), "residual_mm": res,
           "rms_residual_mm": float(np.sqrt(np.nanmean(res[ok] ** 2))),
           "dof": dof, "plane": None}
    if not plane:
        return out
    if xy is None:
        raise ValueError("the plane needs the stations' positions (xy, km)")
    xy = np.asarray(xy, dtype=float)
    okp = ok0.copy()
    for e in range(n_e):                        # a date whose stations lie in a line has no plane
        ss = np.flatnonzero(okp[:, e])
        if ss.size < min_per_epoch_plane or np.linalg.matrix_rank(np.c_[np.ones(ss.size), xy[ss]]) < 3:
            okp[:, e] = False
    pl = {"method": plane_method}
    if plane_method == "protocol":
        okp &= ok
        pe = np.full((n_e, 3), np.nan)
        for e in np.flatnonzero(okp.any(0)):
            ss = np.flatnonzero(okp[:, e])
            s2 = np.sqrt(w[ss, e])
            G = np.c_[np.ones(ss.size), xy[ss]]
            pe[e] = np.linalg.lstsq(G * s2[:, None], (d[ss, e] - const[ss]) * s2, rcond=None)[0]
        rp = np.where(okp, d - const[:, None] - (pe[:, 0][None] + pe[:, 1][None] * xy[:, :1]
                                                 + pe[:, 2][None] * xy[:, 1:2]), np.nan)
        pl.update(offset_mm=pe[:, 0], east_mm_per_km=pe[:, 1], north_mm_per_km=pe[:, 2], constants_mm=const,
                  reference_epoch=None)
    else:
        fx = fixed if okp[fixed].any() else _longest_record(okp, t)
        okp = _prune(okp, fx, min_per_epoch_plane) if fx >= 0 else np.zeros_like(okp)
        if not okp.any():
            pl.update(offset_mm=np.full(n_e, np.nan), east_mm_per_km=np.full(n_e, np.nan),
                      north_mm_per_km=np.full(n_e, np.nan), constants_mm=np.full(n_s, np.nan),
                      reference_epoch=None)
            rp = np.full(d.shape, np.nan)
        else:
            ref = int(np.flatnonzero(okp.any(0))[0])
            po, pc, pe_, pn, rp, _ = _solve_datum(d, w, okp, fx, xy=xy, ref=ref)
            pl.update(offset_mm=po, east_mm_per_km=pe_, north_mm_per_km=pn, constants_mm=pc, reference_epoch=ref)
    pl["epochs"] = int(np.isfinite(pl["offset_mm"]).sum())
    pl["rms_residual_mm"] = float(np.sqrt(np.nanmean(rp[okp] ** 2))) if okp.any() else None
    out["plane"] = pl
    return out


def datum_correction(fit: dict, xy=None) -> np.ndarray:
    """The per-date correction to ADD to InSAR to put it on the GNSS datum:
    the offset, or the plane at `xy` (km, the fit's grid) when the fit has
    one. NaN on dates without a datum."""
    p = fit.get("plane")
    if p is None or xy is None:
        return np.asarray(fit["offset_mm"], dtype=float)
    x, y = float(xy[0]), float(xy[1])
    return p["offset_mm"] + p["east_mm_per_km"] * x + p["north_mm_per_km"] * y


def _rate_coefficient(t: np.ndarray, y: np.ndarray, polynomial: int = 1, periodic=(), steps=(),
                      t_ref: float | None = None) -> float | None:
    """The straight-line coefficient of the model MintPy's
    timeseries2velocity fits (polynomial + sin/cos of each period, years,
    + a step at each of `steps`, years), over the finite values; None with
    too few of them. Time runs from t_ref (MintPy's reference date; the
    first date when None): with a polynomial of degree 2 or more, MintPy's
    'velocity' is the slope AT its reference date, and the slope at year 0
    of a decimal-year axis would be something else entirely."""
    ok = np.isfinite(y)
    t = np.asarray(t, dtype=float)
    tt = t - (float(t_ref) if t_ref is not None else float(t[0]))
    cols = [tt ** p for p in range(max(1, int(polynomial)) + 1)]
    for P in periodic or ():
        cols += [np.sin(2 * np.pi * tt / float(P)), np.cos(2 * np.pi * tt / float(P))]
    for s in steps or ():
        cols.append((t > float(s)).astype(float))            # MintPy: Heaviside, t > t_step
    G = np.vstack(cols).T
    if ok.sum() <= G.shape[1] or np.ptp(t[ok]) <= 0:
        return None
    return float(np.linalg.lstsq(G[ok], y[ok], rcond=None)[0][1])


def datum_rate(fit: dict, t, polynomial: int = 1, periodic=(), use=None, steps=(),
               t_ref: float | None = None) -> dict:
    """
    The rate of the GNSS datum, to add to a velocity product: the
    straight-line coefficient of the same model the velocity was fitted
    with (polynomial, periodic, steps, the dates it used, time from its
    reference date t_ref; the first date when None) applied to o(t), and
    to the plane's coefficients when the fit has a plane. By linearity this
    is exactly rate(InSAR + datum) - rate(InSAR) when the datum covers
    every date the velocity used. `epoch_share` says how much of that it
    covers.

    Returns {offset_mm_yr, east_mm_yr_per_km, north_mm_yr_per_km (None
    without a plane), epochs, epoch_share, model}.
    """
    t = np.asarray(t, dtype=float)
    use = np.ones(t.size, bool) if use is None else np.asarray(use, bool)
    t_ref = float(t[0]) if t_ref is None else float(t_ref)
    p = fit.get("plane")
    src = p["offset_mm"] if p is not None else np.asarray(fit["offset_mm"], dtype=float)
    y = np.where(use, src, np.nan)
    out = {"offset_mm_yr": _rate_coefficient(t, y, polynomial, periodic, steps, t_ref),
           "east_mm_yr_per_km": None, "north_mm_yr_per_km": None,
           "epochs": int(np.isfinite(y).sum()), "epoch_share": round(float(np.isfinite(y).sum() / max(1, use.sum())), 3),
           "model": {"polynomial": int(polynomial), "periodic_years": [float(x) for x in periodic or ()]}}
    if steps:
        out["model"]["steps_years"] = [round(float(s), 4) for s in steps]
    if p is not None:
        out["east_mm_yr_per_km"] = _rate_coefficient(t, np.where(use, p["east_mm_per_km"], np.nan), polynomial,
                                                     periodic, steps, t_ref)
        out["north_mm_yr_per_km"] = _rate_coefficient(t, np.where(use, p["north_mm_per_km"], np.nan), polynomial,
                                                      periodic, steps, t_ref)
    return out


def gnss_absolute_loo(t, insar, gnss, sigma=None, xy=None, plane: bool = False, ids=None,
                      min_per_epoch: int = GNSS_ABS_MIN_PER_EPOCH,
                      min_per_epoch_plane: int = GNSS_ABS_MIN_PER_EPOCH_PLANE,
                      plane_method: str = "joint") -> dict:
    """
    Leave-one-out scoring of the GNSS datum (protocol_B1.yaml): for each
    station k the datum is fitted WITHOUT k (gnss_datum_fit on the others),
    and InSAR(k, t) + datum(t) is compared with GNSS(k, t). k's own constant
    (the plain mean of its residuals over its matched dates, as
    protocol_B1.yaml epoch_matching.series_offset says) is removed. No
    station is scored on a datum it helped to fit.

    insar, gnss: (stations, dates) mm line of sight; sigma as gnss_datum_fit.
    Per station: n_epochs, constant_mm, rmse_mm (residual series, constant
    removed), rate_bias_mm_yr (straight-line slope of the residual = InSAR
    rate on the datum - GNSS rate; the harness fits its own protocol
    models), span_years and residual_mm (list, None where missing), or
    scored False with the reason. Summary over the scored stations: n,
    mean and RMS of the rate biases, median and pooled RMSE.
    """
    t = np.asarray(t, dtype=float)
    insar = np.asarray(insar, dtype=float)
    gnss = np.asarray(gnss, dtype=float)
    n_s = insar.shape[0]
    if plane and xy is None:
        raise ValueError("the plane needs the stations' positions (xy, km)")
    ids = [str(i) for i in (ids if ids is not None else range(n_s))]
    d = gnss - insar
    w = _weights(sigma, d.shape)
    xy = None if xy is None else np.asarray(xy, dtype=float)
    per = {}
    pooled = []
    for k in range(n_s):
        if not np.isfinite(d[k]).any():
            per[ids[k]] = {"scored": False, "reason": "no date with both an InSAR and a GNSS value"}
            continue
        dk = d.copy()
        dk[k] = np.nan
        try:
            fit = gnss_datum_fit(t, dk, sigma, xy, plane, None, min_per_epoch, min_per_epoch_plane,
                                 plane_method)
        except ValueError as exc:
            per[ids[k]] = {"scored": False, "reason": f"the other stations give no datum ({exc})"}
            continue
        r = insar[k] + datum_correction(fit, xy[k] if (plane and xy is not None) else None) - gnss[k]
        ok = np.isfinite(r) & np.isfinite(w[k]) & (w[k] > 0)
        if ok.sum() < 2:
            per[ids[k]] = {"scored": False, "reason": f"{int(ok.sum())} date(s) with a datum from the other stations"}
            continue
        c = float(np.mean(r[ok]))                    # the protocol's series offset: a plain mean
        r0 = np.where(ok, r - c, np.nan)
        slope = float(np.polyfit(t[ok], r0[ok], 1)[0]) if ok.sum() >= 3 and np.ptp(t[ok]) > 0 else None
        pooled.append(r0[ok])
        per[ids[k]] = {"scored": True, "n_epochs": int(ok.sum()), "constant_mm": round(c, 3),
                       "rmse_mm": round(float(np.sqrt(np.mean(r0[ok] ** 2))), 3),
                       "rate_bias_mm_yr": None if slope is None else round(slope, 3),
                       "span_years": round(float(np.ptp(t[ok])), 3),
                       "residual_mm": [None if not np.isfinite(v) else round(float(v), 3) for v in r0]}
    scored = [v for v in per.values() if v.get("scored")]
    biases = np.array([v["rate_bias_mm_yr"] for v in scored if v["rate_bias_mm_yr"] is not None], dtype=float)
    allr = np.concatenate(pooled) if pooled else np.array([])
    summary = {"stations_scored": len(scored), "stations": n_s,
               "rate_bias_mean_mm_yr": round(float(biases.mean()), 3) if biases.size else None,
               "rate_bias_rms_mm_yr": round(float(np.sqrt(np.mean(biases ** 2))), 3) if biases.size else None,
               "series_rmse_median_mm": round(float(np.median([v["rmse_mm"] for v in scored])), 3) if scored else None,
               "series_rmse_pooled_mm": round(float(np.sqrt(np.mean(allr ** 2))), 3) if allr.size else None,
               "variant": f"offset_plus_plane ({plane_method})" if plane else "offset_only"}
    return {"per_station": per, "summary": summary}


def local_xy_km(lat, lon, lat0: float, lon0: float):
    """East and north km from (lat0, lon0), equirectangular: the grid the
    plane is fitted and applied on (better than 0.1% over 30 km)."""
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    return (lon - lon0) * KM_PER_DEG * np.cos(np.radians(lat0)), (lat - lat0) * KM_PER_DEG


# ---------------------------------------------------------------- class matching
def _centre(x: np.ndarray, robust: bool) -> float:
    return float(np.median(x) if robust else x.mean())


def _measured_centres(vel, k, far, min_n, robust) -> dict:
    """{class index: centre} for the classes with at least min_n far cells."""
    out = {}
    for i in range(len(CLASS_EDGES) - 1):
        sel = far & (k == i) & np.isfinite(vel)
        if sel.sum() >= min_n:
            out[i] = _centre(vel[sel], robust)
    return out


def class_means(vel: np.ndarray, tc: np.ndarray, far: np.ndarray, min_n: int = 200,
                robust: bool = False) -> dict:
    """
    Far-field centre of each coherence class {class index: centre}.

    The MEAN by default; robust=True gives the median. Measured 6 Oct 2026:
    every client figure is an AREA mean, and with median centres ordinary
    farmland areas read 0.58 (Site A) / 0.65 (Site B) mm/yr LOS low -
    the farmland rate distribution is skewed (unwrapping-error tail), so the
    median zeroes the typical CELL but not the typical area. The skew from
    real unmapped features is removed instead by the second pass
    (far_without: the found features cut out of the comparison ground), after
    which the mean zeroes ordinary ground. Earlier rationale for the median:
    robust: the MEDIAN. The far field is "everything outside the
    mapped mines", so unmapped movement inside it - Site B features 2 and
    3, 5.6 and 0.9 km2 sinking 20-26 mm/yr, both wholly in the far field -
    skews the distribution; the plain mean put the zero 0.65 mm/yr LOS low on
    Site B and 0.58 low on the Site A record (5 Oct 2026), so every
    relative figure read that much too high. Cutting those two features'
    boxes out of the far field moves the median by 0.21 (0.4-0.6) and 0.44
    (0.6-0.7) mm/yr, the mean by 0.52 and 1.03: the median takes out about
    60% of the sensitivity; far_without() takes out the rest once the
    features are known (a second pass).

    A class with fewer than min_n far cells is not measured: it BORROWS the
    centre of the nearest measured class, so its cells still get a relative
    rate (before 5 Oct 2026 they were silently NaN - a high-coherence pad or
    building inside a mine dropped out of every figure). The borrowing
    classes are listed under "_borrowed" (sorted class indices); callers
    iterate only the int keys. With no class measured but min_n far cells in
    all, every class takes the pooled centre and is listed as borrowed.
    """
    k = coherence_class(tc)
    out = _measured_centres(vel, k, far, min_n, robust)
    n_cls = len(CLASS_EDGES) - 1
    borrowed = []
    if out:
        measured = dict(out)                 # lend only from measured classes, never from a borrower
        for i in range(n_cls):
            if i not in measured:
                out[i] = measured[min(measured, key=lambda j: (abs(j - i), j))]
                borrowed.append(i)
    else:
        pool = far & (k >= 0) & np.isfinite(vel)
        if pool.sum() >= min_n:
            c = _centre(vel[pool], robust)
            out = {i: c for i in range(n_cls)}
            borrowed = list(range(n_cls))
    out = dict(sorted(out.items()))
    out["_borrowed"] = sorted(borrowed)
    return out


def far_without(far: np.ndarray, masks, attrs: dict, buffer_m: float = 500.0) -> np.ndarray:
    """The far field minus found features (each mask grown by buffer_m): the
    second pass of the farmland definition, so movement found inside the
    far field no longer shapes the zero it is measured against."""
    cut = np.zeros_like(far)
    for m in masks:
        cut |= m
    return far & ~euclidean_buffer(cut, attrs, buffer_m) if cut.any() else far.copy()


def _fine_centres(vel, tc, far, step, min_n, robust) -> tuple[np.ndarray, dict]:
    """Fine coherence bins (width `step`) with at least min_n far cells: the
    bin index map and {bin: centre}."""
    edges = np.arange(CLASS_EDGES[0], 1.0 + step / 2, step)
    edges[-1] = 1.01
    b = np.digitize(np.nan_to_num(tc, nan=-1.0), edges) - 1
    b[(b < 0) | (b >= edges.size - 1)] = -1
    cen = {}
    for i in range(edges.size - 1):
        sel = far & (b == i) & np.isfinite(vel)
        if sel.sum() >= min_n:
            cen[i] = _centre(vel[sel], robust)
    return b, cen


def class_matched_field(vel: np.ndarray, tc: np.ndarray, far: np.ndarray, robust: bool = False,
                        fine_step: float | None = None, fine_min_n: int = 2000) -> tuple[np.ndarray, dict]:
    """
    vel minus the far-field centre of the pixel's own coherence class
    (class_means: the mean by default). Averaged over any region this is
    the region-minus-farmland rate with the coherence mix matched, so the
    distributed-scatterer bias (a function of coherence) cancels even where
    the region is less coherent than the farmland. Its median over the far
    field is zero by construction (its MEAN was, while the centres were means).

    Returns (field, centres). The centres are the MEASURED classes only, as
    before 5 Oct 2026, so callers that loop over them (per-class far-field
    series, QA tables) see the same keys; cells of a class too small to
    measure are filled from the nearest measured class, and class_means()
    ["_borrowed"] says which.

    fine_step (option, off by default): match on coherence bins this wide
    wherever a bin has at least fine_min_n far cells, falling back to the
    coarse class. Measured 5 Oct 2026: the far-field rate varies by ~5 mm/yr
    inside the 0.4-0.6 class (Site B -6.7 at 0.40, -11.1 at 0.50, -8.2 at
    0.575; Site A -8.1/-15.1/-9.9), but 0.025 bins narrowed the 0.5-4 km
    farmland nulls by only 4-10% on both projects while moving region values
    by up to 0.5 mm/yr (Site A mines -1.29 -> -1.79): the bias is not
    monotonic in coherence (land cover), so finer bins can absorb real
    differences. Not a clear enough gain to change the default.
    """
    allc = class_means(vel, tc, far, robust=robust)
    borrowed = set(allc["_borrowed"])
    k = coherence_class(tc)
    r = np.full(vel.shape, np.nan, dtype=np.float64)
    for i, m in allc.items():
        if isinstance(i, int):
            sel = k == i
            r[sel] = vel[sel] - m
    if fine_step:
        b, cen = _fine_centres(vel, tc, far, float(fine_step), int(fine_min_n), robust)
        for i, m in cen.items():
            sel = b == i
            r[sel] = vel[sel] - m
    return r, {i: m for i, m in allc.items() if isinstance(i, int) and i not in borrowed}


def placement_field(vel: np.ndarray, tc: np.ndarray, far: np.ndarray, far_wide: np.ndarray,
                    aoi: np.ndarray, min_n: int = 200, robust: bool = False,
                    fine_step: float | None = None) -> np.ndarray:
    """
    The class-matched field to place null footprints on when they range over
    `far_wide` (the farmland outside the processing box as well).

    Inside the AOI it IS class_matched_field(vel, tc, far), so a region's
    value is unchanged. Outside the AOI each cell is matched against the
    centre of its class over the outside ring itself (far_wide & ~aoi): that
    ring is a different population - on Site B its class-matched mean was
    +2.24 mm/yr against 0.00 inside the box (+0.32 on the Site A record),
    because it lies farther from the reference - and without this the null's
    centre and tails mixed the two (Site B 1 km null: median 1.45 over
    far_wide against 0.82 in the box). Re-centring keeps the room the wide
    field gives a mine-sized footprint without shifting the null. A class
    with fewer than min_n ring cells keeps the in-box centre.
    """
    r, _ = class_matched_field(vel, tc, far, robust=robust, fine_step=fine_step)
    ring = far_wide & ~aoi
    if not ring.any():
        return r
    k = coherence_class(tc)
    out_c = _measured_centres(vel, k, ring, min_n, robust)
    outside = ~aoi
    for i, m in out_c.items():
        sel = outside & (k == i)
        r[sel] = vel[sel] - m
    return r


# ---------------------------------------------------------------- the farmland null
def _placements(S, C, n_req, S_tot, N_tot, where: bool = False):
    """Placement means from window sums S and counts C, keeping windows with
    at least n_req valid cells, and the same RE-REFERENCED to the far field
    outside each window (mean of the window minus mean of the rest).
    where=True also returns the kept windows (boolean grid like S) and which
    of them have a re-referenced value (boolean, aligned with the plain ones)."""
    ok = C >= n_req
    vals = S[ok] / C[ok]
    rest = N_tot - C[ok]
    with np.errstate(invalid="ignore", divide="ignore"):
        rr = vals - (S_tot - S[ok]) / rest
    keep = (rest > 0) & np.isfinite(rr)
    if where:
        return vals, rr[keep], float(ok.sum()), ok, keep
    return vals, rr[keep], float(ok.sum())


def _where_info(ok, keep, C, T, offset) -> dict:
    """Per-placement centre (row, col on the product grid) and mean coherence
    of the cells it averaged, for the plain and the re-referenced values:
    what the local and coherence-matched nulls (null_checks) select on."""
    r, c = np.nonzero(ok)
    r = r + float(offset[0]); c = c + float(offset[1])
    coh = (T[ok] / np.maximum(C[ok], 1.0)) if T is not None else None
    return {"rows": r, "cols": c, "coherence": coh,
            "rows_rr": r[keep], "cols_rr": c[keep], "coherence_rr": None if coh is None else coh[keep]}


def _compact(F, W, n_cells, min_valid, Tc=None, where: bool = False):
    """The compact equal-area square's placements. Tc (coherence x valid, as
    F is field x valid) and where=True add each placement's centre and mean
    coherence (_where_info) as a fifth return value."""
    from scipy.ndimage import uniform_filter
    side = int(np.ceil(np.sqrt(n_cells)))
    a = float(side * side)
    S = uniform_filter(F, side, mode="constant") * a
    C = uniform_filter(W, side, mode="constant") * a
    vals, rr, area, ok, keep = _placements(S, C, min_valid * a - 1e-6, float(F.sum()), float(W.sum()), where=True)
    n_ind = max(1, int(round(area / a)))
    if not where:
        return vals, rr, side, n_ind
    T = uniform_filter(Tc, side, mode="constant") * a if Tc is not None else None
    return vals, rr, side, n_ind, _where_info(ok, keep, C, T, (0.0, 0.0))


def footprint_null(field: np.ndarray, far: np.ndarray, region: np.ndarray, min_valid: float = 0.5,
                   tc: np.ndarray | None = None, where: bool = False) -> dict:
    """
    Mean of `field` over the region's footprint shifted to every offset at
    which at least `min_valid` of the footprint falls on far-field cells.
    Returns the values and how they were obtained. Falls back to a compact
    square of the same cell count when the footprint does not fit 20 times.

    `values_rereferenced` are the same placements each taken relative to the
    far field OUTSIDE it. For a footprint that covers a large share of the
    far field the plain values shrink towards the field's mean (a finite-
    domain effect: Site B's 103 km2 project zone, as a 676-cell square over
    the wide far field, measured 1.37 mm/yr plain against 1.61 re-referenced,
    5 Oct 2026); the real region is compared with farmland it does not
    overlap, which is what the re-referenced values mimic.

    where=True adds "where": each placement's centre on the grid (the
    region's centroid carried by the shift) and, when `tc` is given, the
    mean coherence of the cells it averaged (null_checks), plus
    "footprint_area" (the cells one independent placement occupies).
    """
    from scipy.signal import fftconvolve
    valid = far & np.isfinite(field)
    F = np.where(valid, field, 0.0)
    W = valid.astype(np.float64)
    Tc = np.where(valid & np.isfinite(tc), tc, 0.0) if (where and tc is not None) else None
    rows, cols = np.nonzero(region)
    if rows.size == 0:
        return {"values": np.array([]), "values_rereferenced": np.array([]), "method": "empty region",
                "shape": "footprint", "n_placements": 0, "approx_independent": 0, "region_cells": 0}
    K = region[rows.min():rows.max() + 1, cols.min():cols.max() + 1].astype(np.float64)
    n = K.sum()
    method, shape = "region footprint shifted over the far field", "footprint"
    S_tot, N_tot = float(F.sum()), float(W.sum())
    info = None
    if K.shape[0] < F.shape[0] and K.shape[1] < F.shape[1]:
        Kf = K[::-1, ::-1]
        S = fftconvolve(F, Kf, mode="valid")
        C = np.rint(fftconvolve(W, Kf, mode="valid"))
        vals, rr, placements_area, ok, keep = _placements(S, C, min_valid * n, S_tot, N_tot, where=True)
        footprint_area = float(K.shape[0] * K.shape[1])
        if where and vals.size >= 20:
            T = fftconvolve(Tc, Kf, mode="valid") if Tc is not None else None
            info = _where_info(ok, keep, C, T, (rows.mean() - rows.min(), cols.mean() - cols.min()))
        del S, C
    else:
        vals, rr, placements_area, footprint_area = np.array([]), np.array([]), 0.0, 1.0
    n_ind = max(1, int(round(placements_area / footprint_area))) if placements_area else 0
    if vals.size < 20:
        if where:
            vals, rr, side, n_ind, info = _compact(F, W, n, min_valid, Tc, where=True)
        else:
            vals, rr, side, n_ind = _compact(F, W, n, min_valid)
        footprint_area = float(side * side)
        method = f"compact {side}x{side}-cell square (the footprint does not fit in the far field)"
        shape = "compact_square"
    out = {"values": vals, "values_rereferenced": rr, "method": method, "shape": shape,
           "n_placements": int(vals.size), "approx_independent": int(n_ind) if vals.size else 0,
           "region_cells": int(n)}
    if where:
        out["where"], out["footprint_area"] = info, footprint_area
    return out


def prediction_inflation(n_independent: int) -> float:
    """How much wider a 95% interval must be when the spread itself was
    measured from n independent samples: t(0.975, n-1) sqrt(1 + 1/n) / 1.96
    (a prediction interval for one new draw). 1.55 at n = 5, 1.24 at n = 9;
    1.0 from RANGE_INFLATION_BELOW up, where the empirical null is used as is."""
    from scipy.stats import t as student_t
    n = int(n_independent)
    if n >= RANGE_INFLATION_BELOW:
        return 1.0
    n = max(n, 2)
    return float(student_t.ppf(0.975, n - 1) * np.sqrt(1 + 1 / n) / Z95)


def _power_limit(d_sorted: np.ndarray, c: float, power: float = 0.8, sign: int = 0) -> float:
    """Smallest sustained shift s such that, added to (and subtracted from)
    every farmland placement, at least `power` of them read |d + s| >= c.
    d_sorted: the centred null, sorted. sign 0: the worse of the two signs
    (detectable_80); +1: rising only (d + s); -1: sinking only (d - s,
    negative LOS movement) - the same symmetric decision rule |x| >= c, the
    power counted for one direction (E5, 7 Oct 2026)."""
    n = d_sorted.size
    if n == 0 or not np.isfinite(c):
        return float("nan")

    def ge(x):                                   # placements reading >= x
        return n - np.searchsorted(d_sorted, x, side="left")

    def le(x):                                   # placements reading <= x
        return np.searchsorted(d_sorted, x, side="right")

    def rise(s):                                 # detection rate of a shift +s
        return (ge(c - s) + le(-c - s)) / n

    def sink(s):                                 # detection rate of a shift -s
        return (ge(c + s) + le(s - c)) / n

    def pw(s):                                   # the sign asked for, or the worse sign
        return rise(s) if sign > 0 else sink(s) if sign < 0 else min(rise(s), sink(s))

    hi = c + float(d_sorted[-1] - d_sorted[0]) + 1e-9
    grid = np.linspace(0.0, hi, 513)
    k = next((i for i, s in enumerate(grid) if pw(s) >= power), len(grid) - 1)
    if k == 0:
        return 0.0
    lo, hi = grid[k - 1], grid[k]
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        lo, hi = (lo, mid) if pw(mid) >= power else (mid, hi)
    return float(hi)


def _not_testable(value, reason: str, **extra) -> dict:
    return {"testable": False, "not_testable_reason": reason,
            "value_mm_yr": None if value is None else round(float(value), 2),
            "distinguishable_from_farmland": None, "null_available": False,
            "verdict": "not testable: " + reason[0].lower() + reason[1:],
            "detection_limit_sink_mm_yr": None, "detection_limit_rise_mm_yr": None,
            "local_null": None, "coherence_null": None, **extra}


# The local and coherence-matched nulls beside the pooled one (E6, 7 Oct 2026):
# diagnostics for the reviewer, never a switch. Measured on the two projects
# (audit of 7 Oct 2026): the pooled 5% / 80% figures are scene averages - the
# same threshold's false-alarm rate ran from 0% to 23% across 5 km blocks
# (shared regional offsets of -2 to -5 mm/yr LOS), and farmland placements'
# median moved with their mean coherence (+2.6, -0.5, +1.5 mm/yr LOS at
# 0.40-0.50, 0.50-0.55, 0.55-0.60 on Site A, all inside the one 0.4-0.6
# class the matching uses).
LOCAL_NULL_KM = 10.0             # local null: placements centred within this distance of the region's centre
COHERENCE_MATCH = 0.025          # coherence-matched null: placements within +/- this of the region's mean coherence
NULL_CHECK_MIN_INDEPENDENT = RANGE_INFLATION_BELOW    # fewer independent placements: no diagnostic (None)
NULL_CENTRE_SHARE = 0.25         # disagrees: |centre difference| > this x threshold_95 ...
NULL_SPREAD_RATIO = (0.8, 1.25)  # ... or spread ratio (subset / all) outside this


def _subset_null(vals, sel, area, med, sd, c95, what: str) -> dict | None:
    """One diagnostic null: the placements `sel` of the null actually used,
    against all of them. None below NULL_CHECK_MIN_INDEPENDENT independent
    placements (sel.sum() / the cells one placement occupies)."""
    n_ind = int(round(float(sel.sum()) / float(area))) if area else 0
    if n_ind < NULL_CHECK_MIN_INDEPENDENT:
        return None
    x = vals[sel]
    m_sub, sd_sub = float(np.median(x)), float(x.std())
    diff = abs(m_sub - med)
    ratio = sd_sub / sd if sd > 0 else float("inf")
    shift = diff > NULL_CENTRE_SHARE * c95
    wide = not (NULL_SPREAD_RATIO[0] <= ratio <= NULL_SPREAD_RATIO[1])
    why = (f"{what} (about {n_ind} independent copies): centre {m_sub:+.2f} mm/yr against {med:+.2f} for all "
           f"copies, a difference of {diff:.2f} ({NULL_CENTRE_SHARE * c95:.2f}, a quarter of the 95% threshold "
           f"{c95:.2f}, is allowed); spread {sd_sub:.2f} against {sd:.2f}, a ratio of {ratio:.2f} "
           f"({NULL_SPREAD_RATIO[0]:.2f}-{NULL_SPREAD_RATIO[1]:.2f} allowed). ")
    if shift or wide:
        why += ("It differs from the pooled comparison ground: "
                + " and ".join(w for w, on in (("its centre is shifted", shift), ("its spread is different", wide)) if on)
                + ", so the 5% false-alarm rate and the 4-in-5 detection stated for this area are averages over "
                  "the scene that may not hold here. The pooled figures are still the ones reported.")
    else:
        why += "It agrees with the pooled comparison ground."
    return {"median_mm_yr": round(m_sub, 2), "spread_mm_yr": round(sd_sub, 2), "n_independent": n_ind,
            "n_placements": int(sel.sum()), "disagrees": bool(shift or wide), "why": why}


def null_checks(vals, where: dict | None, area: float, reg: np.ndarray, med: float, sd: float, c95: float,
                tc: np.ndarray | None = None, attrs: dict | None = None) -> dict:
    """
    The local and coherence-matched nulls of a region (E6, 7 Oct 2026), as
    diagnostics beside the pooled null `vals` (the null assess used, with
    each placement's centre and mean coherence in `where`, from
    footprint_null(where=True)):

      local_null      placements whose centre lies within LOCAL_NULL_KM
                      (10 km) of the region's centre (needs `attrs`)
      coherence_null  placements whose mean coherence is within
                      +/-COHERENCE_MATCH (0.025) of the region's (needs `tc`)

    Each is None, or {median_mm_yr, spread_mm_yr, n_independent,
    n_placements, disagrees, why} once it rests on at least
    NULL_CHECK_MIN_INDEPENDENT (10) independent placements. `disagrees`:
    |its median - the pooled median| > 0.25 x threshold_95, or its sd / the
    pooled sd outside 0.8-1.25. Nothing switches: the pooled null stays the
    test; a disagreement is a reviewer item (release: null_local_differs).
    `null_checks_note` says why either is None.
    """
    out = {"local_null": None, "coherence_null": None}
    notes = []
    if where is None:
        out["null_checks_note"] = "placement positions not available for this null"
        return out
    rows, cols = np.nonzero(reg)
    if attrs is None:
        notes.append("local null not computed (no grid geometry given)")
    else:
        dy, dx = pixel_metres(attrs)
        dist = np.hypot((where["rows"] - rows.mean()) * dy, (where["cols"] - cols.mean()) * dx)
        out["local_null"] = _subset_null(
            vals, dist <= LOCAL_NULL_KM * 1000.0, area, med, sd, c95,
            f"Farmland copies centred within {LOCAL_NULL_KM:g} km of this area's centre")
        if out["local_null"] is None:
            notes.append(f"fewer than {NULL_CHECK_MIN_INDEPENDENT} independent copies within {LOCAL_NULL_KM:g} km")
    coh = where.get("coherence")
    if tc is None or coh is None:
        notes.append("coherence-matched null not computed (no coherence given)")
    else:
        rc = tc[reg]
        rc = rc[np.isfinite(rc)]
        if not rc.size:
            notes.append("the area has no coherence values")
        else:
            c0 = float(rc.mean())
            out["coherence_null"] = _subset_null(
                vals, np.abs(coh - c0) <= COHERENCE_MATCH, area, med, sd, c95,
                f"Farmland copies of this area's coherence ({c0 - COHERENCE_MATCH:.3f}-{c0 + COHERENCE_MATCH:.3f})")
            if out["coherence_null"] is None:
                notes.append(f"fewer than {NULL_CHECK_MIN_INDEPENDENT} independent copies within "
                             f"+/-{COHERENCE_MATCH:g} of the area's coherence {c0:.3f}")
            else:
                out["coherence_null"]["coherence_window"] = [round(c0 - COHERENCE_MATCH, 3),
                                                             round(c0 + COHERENCE_MATCH, 3)]
    if out["local_null"] is not None:
        out["local_null"]["radius_km"] = LOCAL_NULL_KM
    out["null_checks_note"] = "; ".join(notes) or None
    return out


def rescore_null(sig: dict, value: float) -> dict:
    """assess()'s result for another region of the same size class: its null
    (median, threshold, limits, spread) kept and `value` (the region's mean)
    tested against it, the rule of assess. The placement diagnostics are not
    recomputed (local_null and coherence_null None; p_two_sided None). A
    null that was not testable stays so, with this value."""
    out = dict(sig, value_mm_yr=round(float(value), 2), local_null=None, coherence_null=None, p_two_sided=None,
               percentile_in_farmland=None, null_checks_note="not computed (a shared null of the region's size class)")
    if not sig.get("testable"):
        return out
    med, c95 = float(sig["null_median_mm_yr"]), float(sig["threshold_95_mm_yr"])
    sd = float(sig["farmland_spread_1sigma_mm_yr"])
    x = float(value) - med
    distinct = bool(abs(x) >= c95) if c95 > 0 else bool(x != 0)
    out.update({"excess_mm_yr": round(x, 2), "signal_to_threshold": round(abs(x) / c95, 2) if c95 > 0 else None,
                "z": round(x / sd, 2) if sd > 0 else None, "distinguishable_from_farmland": distinct,
                "verdict": ("distinguishable from the surrounding ground" if distinct
                            else "within the spread of ordinary ground")})
    return out


def assess(field: np.ndarray, far: np.ndarray, region: np.ndarray, min_valid: float = 0.5,
           min_independent: int = 10, tc: np.ndarray | None = None, attrs: dict | None = None) -> dict:
    """
    The region's mean of `field` (normally class_matched_field, or
    placement_field when `far` reaches outside the AOI) set against the
    farmland null: the same mean over the region's footprint shifted across
    `far`. Never returns {}.

    `far` here is where placements may go - pass the widest farmland the
    product has (every coherent cell outside the panels + buffer), not only
    the AOI's, so that a mine-sized footprint fits more than once or twice.

    The test (one rule, 5 Oct 2026):
      excess_mm_yr        value - null median (the null is not centred on 0:
                          its median was 0.2-1.0 on every region measured)
      threshold_95_mm_yr  95th percentile of |null - median|
      distinguishable     |excess| >= threshold_95  (<=> p_two_sided < 0.05)
      signal_to_threshold |excess| / threshold_95  (>= 1 <=> distinguishable)
      detectable_80_mm_yr the smallest sustained shift that, added to every
                          farmland placement, is detected in at least 80% of
                          them (both signs) - ~2.8 sd on a Gaussian null
      detection_limit_sink_mm_yr / detection_limit_rise_mm_yr
                          the same limit for one direction only (E5, 7 Oct
                          2026): sinking = a negative LOS shift, rising =
                          positive; both positive magnitudes in LOS mm/yr,
                          against the same symmetric rule. detectable_80 is
                          the larger of the two (the worse sign). The null is
                          skewed (a long sinking tail: Site A Specific
                          AOI median +0.72, mean +0.03 LOS), so its worse
                          sign was rising and a sinking movement of the
                          stated limit was caught about 9 times in 10, not 4
                          in 5: sinking limit 14.32 LOS (18.3 mm/yr vertical)
                          against the stated 15.46 (19.7). The client is told
                          the sinking limit (client_report).
      z                   excess / sd, for scale only
    When the footprint fits fewer than `min_independent` independent times
    (whatever the method) the compact equal-area square and the re-referenced
    placements (see footprint_null) are tried as well and the LARGEST spread
    is used - with a handful of placements the spread itself is barely
    measured and overstating it is the safe error; `null_thin` says whether
    the null finally used is still below `min_independent`. Below 10
    independent placements the spread is also widened into a prediction
    interval (`spread_for_ranges_mm_yr` = sd x prediction_inflation(n)), and
    the threshold and the detection limits are floored at 1.96 and 2.8 times
    that widened spread.

    `local_null` and `coherence_null` (E6): the same null restricted to the
    placements within 10 km of the region and to those within +/-0.025 of its
    mean coherence - diagnostics only, the pooled null stays the test (see
    null_checks). They need `attrs` (the grid geometry) and `tc` (temporal
    coherence on the same grid); without them they are None.

    A report area's 95% range is built on this test's own frame - excess +/-
    threshold_95 (client_report.tested_range; + the GNSS background, with
    half-width hypot(threshold, 1.96 x its uncertainty), when anchored) - so
    for an area that is not anchored "the range excludes 0" <=>
    distinguishable, always. A range built from spread_for_ranges about the
    plain value (the report's until 7 Oct 2026) did NOT always agree with the
    reading: the null median is not 0 and c95 need not be 1.96 sd.
    spread_for_ranges stays what a range that is not this test uses (the
    report's recent-rate range).

    Not testable (`testable` False, `distinguishable_from_farmland` None,
    `not_testable_reason` a plain sentence that says "ordinary ground" and
    "the comparison ground", never "farmland": these sentences are printed
    as they are by the QA, the report, the workbook and the combined view,
    whatever the ground is - 28 Oct 2026) when the region has fewer than 5
    cells with a value, when fewer than 20 placements fit, or when the null
    rests on fewer than 5 independent placements. A missing null is never
    "within normal variation".
    """
    reg = region & np.isfinite(field)
    n_reg = int(reg.sum())
    if n_reg < 5:
        x = field[region & np.isfinite(field)] if region.any() else np.array([])
        return _not_testable(x.mean() if x.size else None,
                             f"Too few measured points in this area ({n_reg}) to compare it with ordinary ground.",
                             null_thin=True, approx_independent_placements=0, n_placements=0)
    value = float(field[reg].mean())
    diag = tc is not None or attrs is not None
    nl = footprint_null(field, far, reg, min_valid, tc=tc, where=diag)
    v, method, shape, n_ind = nl["values"], nl["method"], nl["shape"], nl["approx_independent"]
    wh, area = nl.get("where"), nl.get("footprint_area")
    if v.size < 20:
        return dict(_not_testable(value, "Too little ordinary ground around this area to compare it with: a "
                                         "patch of its size fits fewer than 20 times in the comparison ground "
                                         "measured.",
                                  null_thin=True, approx_independent_placements=int(n_ind),
                                  n_placements=int(v.size), method=method, null_shape=shape),
                    null="unavailable (too few far-field placements)")
    if n_ind < min_independent:
        # every honest way of measuring the spread of a thin null; keep the largest.
        # Re-referenced placements (each minus the far field outside it) are put
        # back in the region's frame by adding the mean of the far field outside
        # the REGION, so their median is comparable with the region's plain value
        # (without this a far field whose mean is not zero - -0.65 mm/yr on
        # Site B with median class centres - would read as an excess).
        valid = far & np.isfinite(field)
        in_reg = reg & valid
        n_rest = float(valid.sum() - in_reg.sum())
        frame = (float(field[valid].sum() - field[in_reg].sum()) / n_rest) if n_rest > 0 else np.nan

        def _rr_where(info, fin):
            if info is None:
                return None
            return {"rows": info["rows_rr"][fin], "cols": info["cols_rr"][fin],
                    "coherence": None if info["coherence_rr"] is None else info["coherence_rr"][fin]}

        cands = [(v, n_ind, method, shape, wh, area)]
        rr = nl["values_rereferenced"] + frame
        fin = np.isfinite(rr)
        rr = rr[fin]
        if rr.size >= 20:
            cands.append((rr, n_ind, method + ", each placement relative to the far field outside it", shape,
                          _rr_where(wh, fin), area))
        if shape == "footprint":
            Tc = np.where(valid & np.isfinite(tc), tc, 0.0) if (diag and tc is not None) else None
            got = _compact(np.where(valid, field, 0.0), valid.astype(np.float64), n_reg, min_valid, Tc, where=diag)
            vs, rr_sq, side, n_sq = got[:4]
            sq_wh = got[4] if diag else None
            rr_sq = rr_sq + frame
            fin_sq = np.isfinite(rr_sq)
            rr_sq = rr_sq[fin_sq]
            sq = f"compact {side}x{side}-cell square"
            if vs.size >= 20:
                cands.append((vs, n_sq, sq, "compact_square", sq_wh, float(side * side)))
            if rr_sq.size >= 20:
                cands.append((rr_sq, n_sq, sq + ", each placement relative to the far field outside it",
                              "compact_square", _rr_where(sq_wh, fin_sq), float(side * side)))
        best = max(cands, key=lambda c_: float(np.std(c_[0])))
        if best[0] is not v:
            # the count said is the one approx_independent_placements carries
            # (the null used); the footprint's own count follows it (Site A
            # Area A1 said 'fits only ~3' beside approx_independent 16)
            method = (f"{best[2]} (largest spread; fits ~{best[1]} independent times - the region's own footprint "
                      f"fits only ~{n_ind})")
        else:
            method += f" (fits ~{n_ind} independent times; the alternatives' spreads are smaller)"
        v, n_ind, shape, wh, area = best[0], best[1], best[3], best[4], best[5]
    thin = n_ind < min_independent            # of the null actually used
    if n_ind < MIN_TESTABLE_INDEPENDENT:
        return _not_testable(value, f"Too little ordinary ground around this area to compare it with: an area "
                                    f"this large fits only about {n_ind} separate times in the comparison ground "
                                    f"measured ({MIN_TESTABLE_INDEPENDENT} needed).",
                             null_available=True, null_thin=True, approx_independent_placements=int(n_ind),
                             n_placements=int(v.size), method=method, null_shape=shape,
                             thin_null_spread_1sigma_mm_yr=round(float(v.std()), 2))
    med = float(np.median(v))
    d = np.sort(v - med)
    x = value - med
    sd = float(d.std())
    small = thin or n_ind < RANGE_INFLATION_BELOW
    infl = prediction_inflation(n_ind) if small else 1.0
    c95 = float(np.quantile(np.abs(d), 0.95))
    if small:
        c95 = max(c95, Z95 * sd * infl)
    s80 = _power_limit(d, c95)
    s_sink, s_rise = _power_limit(d, c95, sign=-1), _power_limit(d, c95, sign=+1)
    if small:
        floor = (Z95 + Z80) * sd * infl
        s80, s_sink, s_rise = max(s80, floor), max(s_sink, floor), max(s_rise, floor)
    p = float(np.mean(np.abs(d) >= abs(x)))
    distinct = bool(abs(x) >= c95) if c95 > 0 else bool(x != 0)
    out = {
        "testable": True, "not_testable_reason": None, "null_available": True,
        "value_mm_yr": round(value, 2),
        "farmland_spread_1sigma_mm_yr": round(sd, 2),
        "spread_for_ranges_mm_yr": round(sd * infl, 2),
        "null_median_mm_yr": round(med, 2),
        "excess_mm_yr": round(x, 2),
        "threshold_95_mm_yr": round(c95, 2),
        "detectable_80_mm_yr": round(s80, 2),
        "detection_limit_sink_mm_yr": round(s_sink, 2),
        "detection_limit_rise_mm_yr": round(s_rise, 2),
        "signal_to_threshold": round(abs(x) / c95, 2) if c95 > 0 else None,
        "z": round(x / sd, 2) if sd > 0 else None,
        "percentile_in_farmland": round(100 * float(np.mean(v < value)), 1),
        "p_two_sided": round(p, 3),
        "n_placements": int(v.size), "approx_independent_placements": int(n_ind),
        "null_thin": bool(thin), "method": method, "null_shape": shape,
        "distinguishable_from_farmland": distinct,
        "verdict": ("distinguishable from the surrounding ground" if distinct
                    else "within the spread of ordinary ground"),
        "local_null": None, "coherence_null": None,
    }
    if diag:
        out.update(null_checks(v, wh, area, reg, med, sd, c95, tc=tc, attrs=attrs))
    else:
        out["null_checks_note"] = "not computed (assess was not given the coherence and grid geometry)"
    return out


# ---------------------------------------------------------------- time
def iter_epochs(ds):
    """
    Yield (i, epoch i as a 2-D array) reading the cube in chunk-aligned
    blocks. MintPy's geocoded cube is chunked several dates deep (8 x 82 x 74
    on the Site A record) and h5py's default chunk cache is 1 MB, so
    `ds[i]` in a loop decodes 8 dates to return one: 51 GB read per pass of a
    6.4 GB file. One block of chunk-depth dates at a time reads each chunk once.
    """
    n = ds.shape[0]
    b = max(1, int(ds.chunks[0])) if ds.chunks else 1
    for i0 in range(0, n, b):
        blk = ds[i0:i0 + b]
        for j in range(blk.shape[0]):
            yield i0 + j, blk[j]


def years(dates) -> np.ndarray:
    d0 = datetime.strptime(dates[0], "%Y%m%d")
    return np.array([(datetime.strptime(d, "%Y%m%d") - d0).days / 365.25 for d in dates])


def fit_rate(t: np.ndarray, y: np.ndarray) -> dict | None:
    """
    Straight-line rate with two errors: the white-noise OLS error (what
    MintPy's `residue` uncertainty and the old stderr fields are) and the
    same inflated for lag-1 autocorrelation of the residuals,
    sqrt((1 + rho) / (1 - rho)). Both are TEMPORAL errors only; neither
    knows how different ordinary farmland patches are (see assess()).
    """
    ok = np.isfinite(y)
    if ok.sum() < 6:
        return None
    tt, yy = t[ok], y[ok]
    A = np.c_[tt, np.ones(tt.size)]
    coef = np.linalg.lstsq(A, yy, rcond=None)[0]
    e = yy - A @ coef
    sxx = float(((tt - tt.mean()) ** 2).sum())
    se = float(np.sqrt(e @ e / max(1, tt.size - 2) / sxx))
    rho = float((e[1:] @ e[:-1]) / (e @ e)) if e @ e > 0 else 0.0
    r = min(max(rho, 0.0), 0.95)
    return {"slope": float(coef[0]), "intercept": float(coef[1]), "stderr_white": se,
            "lag1_autocorrelation": round(rho, 3), "stderr_ar1": se * float(np.sqrt((1 + r) / (1 - r))),
            "n": int(ok.sum()), "span_years": float(tt[-1] - tt[0])}


def trend_total(fit: dict) -> float:
    """Displacement over the record AT THE AVERAGE RATE: rate x span.
    Robust to the single-epoch noise (~7 mm) and to the season of the end
    dates, which a last-minus-first difference is not - but it is the
    measured change only when the ground moved at a steady pace. Site B
    Feature 1 (fast in 2019-20, slow since) read 415 mm this way against
    ~348 mm in its own series. The measured change is record_change();
    this stays as `total_at_average_rate_mm`."""
    return fit["slope"] * fit["span_years"]


def _yearly_knots(tt: np.ndarray, min_epochs: int = 6, min_tail_years: float = 0.5) -> list[float]:
    """Whole years after the first date (tt starts at 0), merged so that every
    segment holds at least min_epochs epochs and the last one spans at least
    min_tail_years: an end segment fitted to a few months would set the total."""
    span = float(tt[-1])
    knots = []
    for k in range(1, int(np.floor(span)) + 1):
        if span - k < min_tail_years:
            break
        lo = knots[-1] if knots else 0.0
        if np.sum((tt >= lo) & (tt < k)) >= min_epochs:
            knots.append(float(k))
    while knots and np.sum(tt >= knots[-1]) < min_epochs:
        knots.pop()
    return knots


def _change_design(tt: np.ndarray, knots) -> np.ndarray:
    """intercept, rate, annual sine and cosine, then one hinge max(t - k, 0) per knot"""
    w = 2 * np.pi * tt
    return np.c_[np.ones(tt.size), tt, np.sin(w), np.cos(w), *[np.maximum(tt - k, 0.0) for k in knots]]


def _ls_fit(A: np.ndarray, yy: np.ndarray):
    """coefficients, residual sum of squares, lag-1 residual autocorrelation clipped to [0, 0.95]"""
    b, *_ = np.linalg.lstsq(A, yy, rcond=None)
    e = yy - A @ b
    rss = float(e @ e)
    rho = float((e[1:] @ e[:-1]) / rss) if rss > 0 else 0.0
    return b, rss, min(max(rho, 0.0), 0.95)


def _nested_p(rss0: float, rss1: float, q: int, dof: int, rho: float) -> float:
    """F-test of q added terms with the statistic and the residual degrees of
    freedom deflated by the AR(1) variance inflation (1 + rho) / (1 - rho)."""
    from scipy.stats import f as f_dist
    if dof <= 2 or rss1 <= 0:
        return 1.0
    infl = (1 + rho) / (1 - rho)
    F = max(rss0 - rss1, 0.0) / q / (rss1 / dof) / infl
    return float(f_dist.sf(F, q, max(1.0, dof / infl)))


PACE_NULL_MIN = 20     # farmland copies needed before their 95th percentile tests a change of pace (E10)
PACE_ALPHA = 0.05      # the pace-change test's false-alarm rate on ordinary farmland


def _rank_threshold(vals: np.ndarray, alpha: float = PACE_ALPHA) -> float | None:
    """The farmland copies' (1 - alpha) percentile as a test threshold: the
    ceil((1 - alpha)(n + 1))-th smallest of n copies (a Monte Carlo test -
    the real footprint is one more draw like the copies, so it exceeds that
    copy by chance with probability (n + 1 - k)/(n + 1) <= alpha). None when
    n is too small for any copy to give alpha (beating all n copies happens
    by chance 1 in n + 1: 19 copies are needed for 5%). np.quantile's
    interpolated 95th percentile is NOT a 5% test with this few copies:
    simulated, ordinary farmland exceeds it 9.1% of the time with 20 copies,
    7.7% with 30 and 7.1% with 40 (reviewer, 7 Oct 2026)."""
    v = np.sort(np.asarray(vals, dtype=float))
    k = int(np.ceil(round((1.0 - alpha) * (v.size + 1), 9)))
    return float(v[k - 1]) if 1 <= k <= v.size else None


def _pace_model(t: np.ndarray, y: np.ndarray, min_years: float):
    """The straight line and the hinge model of record_change: (lin, out, fit)
    with `out` the rate x span result and `fit` None when no hinge model can
    be fitted, else {p, change, sigma, start, n_segments}."""
    t = np.asarray(t, dtype=float); y = np.asarray(y, dtype=float)
    lin = fit_rate(t, y)
    if lin is None:
        return None, None, None
    ok = np.isfinite(y)
    t0 = float(t[ok][0])
    tt, yy = t[ok] - t0, y[ok]
    span = float(tt[-1])
    linear = lin["slope"] * lin["span_years"]
    start_lin = lin["intercept"] + lin["slope"] * t0
    out = {"change_mm": linear, "linear_change_mm": linear, "nonlinear": False,
           "sigma_mm": lin["stderr_ar1"] * span, "method": "rate_x_span",
           "start_mm": start_lin, "end_mm": start_lin + linear,
           "curvature_p": None, "piecewise_change_mm": None, "n_segments": 1}
    knots = _yearly_knots(tt) if span >= min_years else []
    if span < min_years or not knots:
        out["reason"] = (f"record shorter than {min_years:g} years" if span < min_years
                         else "too few dates to follow a change of pace")
        return lin, out, None
    # refine the end segments: one half-way hinge each, kept only if it is needed
    for lo, hi in ((0.0, knots[0]), (knots[-1], span)):
        mid = 0.5 * (lo + hi)
        if np.sum((tt >= lo) & (tt < mid)) < 6 or np.sum(tt >= mid) - np.sum(tt > hi) < 6:
            continue
        _, rss_a, _ = _ls_fit(_change_design(tt, knots), yy)
        trial = sorted(knots + [mid])
        A = _change_design(tt, trial)
        _, rss_b, rho_b = _ls_fit(A, yy)
        if _nested_p(rss_a, rss_b, 1, tt.size - A.shape[1], rho_b) < 0.01:
            knots = trial
    A0, A1 = _change_design(tt, []), _change_design(tt, knots)
    _, rss0, _ = _ls_fit(A0, yy)
    b1, rss1, rho = _ls_fit(A1, yy)
    dof = tt.size - A1.shape[1]
    if dof <= 2 or rss1 <= 0:
        out["reason"] = "too few dates to follow a change of pace"
        return lin, out, None
    p = _nested_p(rss0, rss1, len(knots), dof, rho)
    # the model's change with the seasonal terms set to zero, and its error
    cvec = np.r_[0.0, span, 0.0, 0.0, [span - k for k in knots]]
    cov = np.linalg.pinv(A1.T @ A1) * (rss1 / dof) * (1 + rho) / (1 - rho)
    fit = {"p": p, "change": float(cvec @ b1), "sigma": float(np.sqrt(max(cvec @ cov @ cvec, 0.0))),
           "start": float(b1[0]), "n_segments": len(knots) + 1}
    out.update({"curvature_p": p, "piecewise_change_mm": fit["change"], "n_segments": fit["n_segments"]})
    return lin, out, fit


def pace_change_statistic(t: np.ndarray, y: np.ndarray, min_years: float = 2.0) -> float | None:
    """|piecewise change - rate x span| of a series (the units of y): how much
    the measured change departs from the steady-pace total, by exactly
    record_change's model. None when no hinge model can be fitted."""
    lin, out, fit = _pace_model(t, y, min_years)
    if fit is None:
        return None
    return abs(fit["change"] - out["linear_change_mm"])


def pace_change_null(t: np.ndarray, series, min_years: float = 2.0) -> dict:
    """
    The farmland null of the pace-change statistic (E10, 7 Oct 2026): the
    statistic (pace_change_statistic) of every farmland copy's series of the
    region's own footprint. `series`: an iterable of 1-D series on the dates
    `t`, in the SAME units and frame as the region's series they will be set
    against (client_report's placement series are vertical mm; the region's
    LOS series must then be divided by the same los_up, or these multiplied).
    Returns {p95_mm, median_mm, n, values_mm (sorted), basis}; p95_mm is the
    copies' 95th percentile taken as a 5% test (_rank_threshold: the
    ceil(0.95 (n + 1))-th smallest copy - the largest of 20 or 30 copies, the
    39th of 40), None below PACE_NULL_MIN copies with a statistic.
    """
    vals = []
    for s in series if series is not None else []:
        v = pace_change_statistic(t, np.asarray(s, dtype=float), min_years)
        if v is not None and np.isfinite(v):
            vals.append(v)
    vals = np.asarray(vals, dtype=float)
    enough = vals.size >= PACE_NULL_MIN
    thr = _rank_threshold(vals) if enough else None
    return {"p95_mm": round(thr, 2) if thr is not None else None,
            "median_mm": round(float(np.median(vals)), 2) if vals.size else None,
            "n": int(vals.size), "values_mm": [round(float(v), 1) for v in np.sort(vals)],
            "basis": (f"{vals.size} farmland copies of the same footprint: 95th percentile of |measured change - "
                      f"rate x span|" + ("" if enough else f" (fewer than {PACE_NULL_MIN}: not a test)"))}


def record_change(t: np.ndarray, y: np.ndarray, min_years: float = 2.0, pace_null=None) -> dict | None:
    """
    How far the ground moved over the record, measured from the series rather
    than from a straight line.

    Model (least squares): intercept + rate + a hinge max(t - k, 0) at every
    whole year k after the first date (a continuous piecewise-linear path that
    may change pace once a year) + an annual sine/cosine. The change is the
    model's end-to-end difference with the seasonal terms removed. No
    quadratic: it gave 422 mm for Site B Feature 1, no better than the
    line's 415, where the series itself shows ~347.

    The two END segments set the total, so each may take one extra hinge
    half-way along when that improves the fit significantly (p < 0.01).
    Measured on synthetic 10.45-yr series (270 dates, 7 mm noise, 5 mm
    season, 40 seeds, 5 Oct 2026): yearly hinges alone understated a decay
    with a 1-yr time constant starting at the first date by 5.8% (it bends
    inside the first year); with the end refinement 2.1%. Half-yearly hinges
    throughout gave 1.6% but doubled the noise under a year-to-year varying
    season (sd 7.8 vs 3.8-4.8 mm, and 7 of 60 steady series called
    nonlinear against 1-2). A mid-record S-curve and a step are within 1%.

    WHEN THE HINGE TOTAL IS USED (E10, 7 Oct 2026). The pace change is
    |piecewise change - rate x span| (pace_change_statistic). `pace_null` is
    the farmland null of that statistic for the region's own footprint:
    either pace_change_null(t, series) or the farmland copies' series
    themselves (an iterable of 1-D series on `t`, same units and frame as
    `y`; pace_change_null is applied). Given a null with at least
    PACE_NULL_MIN (20) copies, the hinge total is used only when the pace
    change exceeds the null's 95th percentile (taken so that ordinary
    farmland exceeds it at most 5% of the time: _rank_threshold - the
    largest of 20-38 copies, the second largest of 39-58), else the change
    is rate x span;
    `nonlinear` is that same outcome; `pace_change_tested` True. Given a null
    with fewer copies (or none - a region too large for copies of itself to
    fit), the change cannot be tested: rate x span, `nonlinear` False,
    `pace_change_tested` False.
    With NO `pace_null` (None, the technical callers) the old gate stands: an
    F-test of all the hinges against line + season, the statistic and its
    residual degrees of freedom deflated by the residuals' lag-1
    autocorrelation; hinges when p < 0.05, `nonlinear` when also the two
    totals differ by more than max(2 sigma, 10%); `pace_change_tested` False.
    That F-test is NOT calibrated: on independent class-matched farmland
    patches of both projects it gave p < 0.05 on every one (60/60, 40/40,
    12/12; audit of 7 Oct 2026) and 'nonlinear' on 50-65% of them, so it
    never fell back to rate x span. The farmland null replaces it wherever a
    client reads the total.

    Returns {change_mm, linear_change_mm (rate x span, fit_rate's rate),
    nonlinear, sigma_mm, method, reason, start_mm, end_mm, curvature_p,
    piecewise_change_mm, n_segments, pace_change_mm, pace_change_tested,
    pace_null_p95_mm, pace_null_n}. sigma_mm is the TEMPORAL error of the
    change (AR(1)-inflated), not the farmland spread. start_mm/end_mm are the
    model (no season) at the first and last dates, so a series zeroed on
    start_mm ends near change_mm. Falls back to rate x span (method
    "rate_x_span") below `min_years` or when too few epochs leave no hinge.
    None when fit_rate gives no rate (fewer than 6 epochs).
    """
    lin, out, fit = _pace_model(t, y, min_years)
    if lin is None:
        return None
    null = None
    if isinstance(pace_null, dict) and not {"p95_mm", "n"} <= set(pace_null):
        # a dict of named series would otherwise read as a null of 0 copies and
        # silently give rate x span, untested (reviewer, 7 Oct 2026)
        raise ValueError("pace_null must be pace_change_null(...) or a list of series, not a dict of series")
    if pace_null is not None:
        null = pace_null if isinstance(pace_null, dict) else pace_change_null(t, pace_null, min_years)
    out.update({"pace_change_mm": None, "pace_change_tested": False,
                "pace_null_p95_mm": (null or {}).get("p95_mm"), "pace_null_n": (null or {}).get("n")})
    if fit is None:
        return _round_change(out)
    linear = out["linear_change_mm"]
    pace = abs(fit["change"] - linear)
    out["pace_change_mm"] = pace
    p = fit["p"]
    hinge = {"change_mm": fit["change"], "sigma_mm": fit["sigma"], "method": "piecewise_linear_yearly",
             "start_mm": fit["start"], "end_mm": fit["start"] + fit["change"]}
    if null is None:
        if p < 0.05:
            out.update(hinge, reason=f"the pace changed during the record (p = {p:.2g})")
            out["nonlinear"] = bool(abs(fit["change"] - linear) > max(2 * fit["sigma"], 0.1 * abs(linear)))
        else:
            out["reason"] = f"no significant change of pace (p = {p:.2g}): rate x span"
        return _round_change(out)
    p95 = null.get("p95_mm")
    n = int(null.get("n") or 0)
    if p95 is None or n < PACE_NULL_MIN:
        out["reason"] = (f"too few farmland copies of this footprint ({n}; {PACE_NULL_MIN} needed) to test a "
                         f"change of pace: rate x span")
        return _round_change(out)
    out["pace_change_tested"] = True
    if pace > float(p95):
        out.update(hinge, reason=(f"the pace changed during the record: the measured change departs from rate x "
                                  f"span by {pace:.1f}, more than {float(p95):.1f} (the 95th percentile of {n} "
                                  f"farmland copies, a 5% test)"))
        out["nonlinear"] = True
    else:
        out["reason"] = (f"no change of pace beyond farmland's: the measured change departs from rate x span by "
                         f"{pace:.1f}, within {float(p95):.1f} (the 95th percentile of {n} farmland copies, a 5% "
                         f"test): rate x span")
    return _round_change(out)


def _round_change(d: dict) -> dict:
    for k in ("change_mm", "linear_change_mm", "sigma_mm", "start_mm", "end_mm", "piecewise_change_mm",
              "pace_change_mm"):
        if d.get(k) is not None:
            d[k] = round(float(d[k]), 1)
    if d.get("curvature_p") is not None:
        d["curvature_p"] = round(float(d["curvature_p"]), 4)
    return d


# ---------------------------------------------------------------- per-pixel repeatability
def empirical_rate_model(ts_file: Path, tc: np.ndarray, far: np.ndarray, cache: Path | None = None,
                         basis: str | None = None) -> dict:
    """
    How repeatable a single pixel's rate is, by coherence class, measured
    over the far field:

      formal       white-noise OLS error of the full-record rate (= MintPy's
                   residue-based velocityStd)
      ar1          the same inflated for the residuals' lag-1 autocorrelation
      split_half   robust spread of (rate fitted to the second half of the
                   record - rate fitted to the first half)
      empirical    split_half / 2: the full-record rate error if the noise
                   behaves like a random walk (error variance ~ 1/span),
                   which the residual autocorrelation (~0.8) indicates. Under
                   white noise it would be split_half / 4 (each half has half
                   the epochs over half the span, so its rate variance is 8x
                   the full record's and the difference's 16x; simulated: 4.0,
                   against 1.97 for a random walk); that lower figure is given
                   as `empirical_white_scaling` (it read split_half / (2 sqrt 2)
                   until 5 Oct 2026, an informational field only).

    Streams the geocoded cube epoch by epoch (two passes), holding only the
    far-field pixels. Cached in `cache` against the cube's size and mtime.
    `basis` replaces the recorded description of the ground (B1: the stable
    polygon instead of the far field); None keeps the far-field text.
    """
    import h5py
    st = ts_file.stat()
    key = {"file": str(ts_file), "size": st.st_size, "mtime": st.st_mtime, "far_cells": int(far.sum())}
    if cache is not None and cache.exists():
        try:
            c = json.loads(cache.read_text())
            if c.get("key") == key:
                return c
        except (ValueError, OSError):
            pass
    idx = np.flatnonzero(far.ravel())
    kcls = coherence_class(tc).ravel()[idx]
    with h5py.File(ts_file, "r") as f:
        dates = [d.decode() if isinstance(d, bytes) else str(d) for d in f["date"][:]]
        ds = f["timeseries"]
        t = years(dates); n = len(t)
        half = t < t.mean()
        groups = {"full": np.ones(n, bool), "a": half, "b": ~half}
        acc = {g: [np.zeros(idx.size), np.zeros(idx.size)] for g in groups}     # sum y, sum (t - tbar) y
        tbar = {g: float(t[m].mean()) for g, m in groups.items()}
        sxx = {g: float(((t[m] - tbar[g]) ** 2).sum()) for g, m in groups.items()}
        finite = np.ones(idx.size, bool)
        for i, ep in iter_epochs(ds):
            y = ep.ravel()[idx].astype(np.float64) * 1000.0
            finite &= np.isfinite(y); y = np.nan_to_num(y)
            for g, m in groups.items():
                if m[i]:
                    acc[g][0] += y; acc[g][1] += (t[i] - tbar[g]) * y
        slope = {g: acc[g][1] / sxx[g] for g in groups}
        icpt = acc["full"][0] / n - slope["full"] * tbar["full"]
        see = np.zeros(idx.size); sel1 = np.zeros(idx.size); prev = None
        for i, ep in iter_epochs(ds):
            y = np.nan_to_num(ep.ravel()[idx].astype(np.float64) * 1000.0)
            e = y - (icpt + slope["full"] * t[i])
            see += e * e
            if prev is not None:
                sel1 += e * prev
            prev = e
    ok = finite & (kcls >= 0)
    sig_w = np.sqrt(see / max(1, n - 2) / sxx["full"])
    rho = np.clip(np.where(see > 0, sel1 / np.maximum(see, 1e-12), 0.0), 0.0, 0.95)
    sig_ar1 = sig_w * np.sqrt((1 + rho) / (1 - rho))
    dv = slope["b"] - slope["a"]

    def robust_sd(x):
        return float(1.4826 * np.median(np.abs(x - np.median(x)))) if x.size else None

    rows = []
    for i, lab in enumerate(class_labels()):
        m = ok & (kcls == i)
        if m.sum() < 200:
            continue
        sh = robust_sd(dv[m])
        rows.append({"tc": lab, "class_index": i, "n": int(m.sum()),
                     "formal_median_mm_yr": round(float(np.median(sig_w[m])), 3),
                     "ar1_median_mm_yr": round(float(np.median(sig_ar1[m])), 3),
                     "lag1_autocorrelation_median": round(float(np.median(rho[m])), 3),
                     "split_half_robust_sd_mm_yr": round(sh, 3),
                     "empirical_mm_yr": round(sh / 2, 3),
                     "empirical_white_scaling_mm_yr": round(sh / 4, 3)})
    allsh = robust_sd(dv[ok])
    out = {"key": key, "by_class": rows,
           "all": {"n": int(ok.sum()), "formal_median_mm_yr": round(float(np.median(sig_w[ok])), 3),
                   "ar1_median_mm_yr": round(float(np.median(sig_ar1[ok])), 3),
                   "lag1_autocorrelation_median": round(float(np.median(rho[ok])), 3),
                   "split_half_robust_sd_mm_yr": round(allsh, 3), "empirical_mm_yr": round(allsh / 2, 3)},
           "halves": [dates[int(np.argmax(half))], dates[int(np.flatnonzero(half)[-1])],
                      dates[int(np.argmax(~half))], dates[-1]],
           "basis": basis or ("far-field pixels (inside the AOI, outside every mapped panel + buffer) whose series "
                              "is finite on every date; empirical = robust sd of (second-half rate - first-half "
                              "rate) / 2, the full-record rate error under random-walk-like noise"),
           "computed": datetime.now().isoformat(timespec="seconds")}
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(out, indent=1))
    log.info("Empirical rate model: formal %.2f, AR(1) %.2f, empirical %.2f mm/yr (median over %d far-field px)",
             out["all"]["formal_median_mm_yr"], out["all"]["ar1_median_mm_yr"], out["all"]["empirical_mm_yr"], out["all"]["n"])
    return out


def empirical_sigma_map(model: dict, tc: np.ndarray) -> np.ndarray:
    """Per-pixel empirical rate error, looked up by coherence class (mm/yr)."""
    k = coherence_class(tc)
    out = np.full(tc.shape, np.nan, dtype=np.float32)
    table = {r["class_index"]: r["empirical_mm_yr"] for r in model.get("by_class", [])}
    for i in range(len(CLASS_EDGES) - 1):
        if i in table:
            out[k == i] = table[i]
        elif table:            # a class too small to measure takes the nearest measured one
            out[k == i] = table[min(table, key=lambda j: abs(j - i))]
    return out


def farmland_cell_spread(rel: np.ndarray, tc: np.ndarray, far: np.ndarray) -> tuple[np.ndarray, list]:
    """
    The single-cell farmland null: robust spread of the class-matched relative
    rate over far-field cells of each coherence class. Larger than a cell's
    own repeatability (7.4 vs 5.7 mm/yr at 0.4-0.6 on the record) because
    fields differ PERSISTENTLY from one another - the bias varies field by
    field and repeats in both halves of the record - so a cell's rate can be
    repeatable and still say nothing about motion. Returns the per-cell map
    (mm/yr) and the table.
    """
    k = coherence_class(tc)
    out = np.full(tc.shape, np.nan, dtype=np.float32); rows = []
    for i, lab in enumerate(class_labels()):
        x = rel[far & (k == i) & np.isfinite(rel)]
        if x.size < 200:
            continue
        sd = float(1.4826 * np.median(np.abs(x - np.median(x))))
        out[k == i] = sd
        rows.append({"tc": lab, "class_index": i, "n": int(x.size), "farmland_cell_spread_mm_yr": round(sd, 3)})
    return out, rows


def _nearest_class(table: dict, i: int):
    """table[i], or the nearest measured class's value (empirical_sigma_map's rule)."""
    if i in table:
        return table[i]
    return table[min(table, key=lambda j: (abs(j - i), j))] if table else None


def cell_null_by_class(model: dict, spread_rows: list | None = None) -> dict[int, float]:
    """
    The single-cell farmland null per coherence class, mm/yr LOS: the larger
    of the farmland cell spread (farmland_cell_spread - fields differ
    persistently) and the cell's own repeatability (empirical_rate_model).
    The same rule as the rate signal-to-noise raster (np.fmax of the two
    maps), written down once so the uncertainty raster, the KMZ and the
    workbook stop using three different per-cell figures (Site A
    Area A2: 0.38 vs 0.27 in/yr for the same cells). A class too small
    to have a repeatability takes the nearest measured class's; one too small
    for a spread uses the repeatability alone. `spread_rows` defaults to
    model["farmland_cell_spread"] when deliverables has stored it there.
    """
    emp = {int(r["class_index"]): float(r["empirical_mm_yr"]) for r in (model or {}).get("by_class", [])
           if r.get("empirical_mm_yr") is not None}
    rows = spread_rows if spread_rows is not None else (model or {}).get("farmland_cell_spread") or []
    spr = {int(r["class_index"]): float(r["farmland_cell_spread_mm_yr"]) for r in rows
           if r.get("farmland_cell_spread_mm_yr") is not None}
    out = {}
    for i in range(len(CLASS_EDGES) - 1):
        vals = [v for v in (_nearest_class(emp, i), spr.get(i)) if v is not None and np.isfinite(v)]
        if vals:
            out[i] = round(max(vals), 3)
    return out


def uncertainty_cache_path(cfg) -> Path:
    return cfg.project_dir / "deliverables" / "uncertainty_model.json"
