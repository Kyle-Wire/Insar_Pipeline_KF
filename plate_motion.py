"""
Remove the long-wavelength ramp that rigid plate motion imprints on InSAR.

THE PHYSICS, AND WHY IT IS NOT OBVIOUS
--------------------------------------
A tectonic plate moves as a rigid body at centimetres per year. That sounds
like something a relative InSAR product removes for free: if every pixel
moves the same, referencing to one pixel subtracts it.

It does not, because the LINE-OF-SIGHT VECTOR IS NOT CONSTANT ACROSS THE
SCENE. Sentinel-1 IW incidence runs about 30 deg at near range to 46 deg at
far range, so the east component of the look vector changes across the
swath. A spatially uniform ground velocity therefore projects into a
SPATIALLY VARYING line-of-sight velocity - a ramp.

Stephenson et al. (2022, GRL 10.1029/2022GL099835) show this is the
dominant source of long-wavelength velocity gradient in multi-year stacks
once ionosphere and troposphere are corrected: centimetre-per-year plate
motion induces millimetre-per-year gradients over hundreds of kilometres.

WHY THIS BEATS FITTING AN EMPIRICAL PLANE
-----------------------------------------
An empirical ramp fit removes whatever happens to be long-wavelength,
including real regional deformation - it cannot tell the two apart. This
removes a term that is independently KNOWN, from GNSS, and leaves
everything else alone. The empirical plane is then useful as a CHECK: if
the two agree, the ramp is understood; if the empirical one is much
larger, something else is in there and blanket removal would be wrong.

WHAT IS ACTUALLY SUBTRACTED
---------------------------
Not the absolute plate velocity - the DIFFERENTIAL relative to the
reference pixel, which is all a referenced product contains:

    ramp(x) = (los_e(x) - los_e(ref)) * Ve + (los_n(x) - los_n(ref)) * Vn

Over Site A the GNSS field gives Ve about -14.5 mm/yr, Vn about
+0.6 mm/yr, near-identical at every station within 120 km - which is what
makes treating it as rigid defensible here.

THE PUBLISHED PLATE MODEL (Decision 5, 8 Oct 2026)
--------------------------------------------------
The second half of this module puts GNSS in a PLATE-FIXED frame: the
station's velocity minus the rigid motion of its tectonic plate, so that
ground which simply rides on its plate reads zero. NGL's daily series are
in IGS20, which is aligned to ITRF2020, so the model is the ITRF2020 plate
motion model (ITRF2020-PMM):

    Altamimi, Z., Metivier, L., Rebischung, P., Collilieux, X., Chanard, K.,
    Barneoud, J. (2023) ITRF2020 plate motion model. Geophysical Research
    Letters 50(24), e2023GL106373. https://doi.org/10.1029/2023GL106373

The 13 rotation poles and the origin rate bias (ORB) are copied from the
IERS ITRS Centre's file ITRF2020-PMM.dat (ITRF2020_PMM_URL; its sha256 is
ITRF2020_PMM_SHA256), and the plate velocity at a point is

    v = omega x X + ORB,   horizontal components only

as the ITRF site advises: "add the estimated ORB to the horizontal
velocities predicted by the ITRF2020 PMM rotation poles for full
consistency with ITRF2020. However, the predicted vertical velocities
resulting from the addition of the ORB should be discarded." Checked on
8 Oct 2026 against the published residual table (ITRF2020-PMM-residuals.dat,
518 sites): observed minus this prediction reproduces the published
residuals to 0.005 mm/yr rms (0.04 at most, from the 3-decimal
coordinates); without the ORB it is off by 0.51 mm/yr rms (0.90 at
most). Copies of both files and the check (check_published_residuals.py
and .out) are in review_2026-10-08/p23/plate/.

How well a rigid plate describes the ground: the model's own residuals
are about 0.2-0.4 mm/yr rms per component on the plates that matter here
(ITRF2020_PMM_RESIDUAL_RMS; Australia 0.23/0.22, North America 0.33/0.37,
Eurasia 0.25/0.25), at sites chosen for being on stable plate interiors.
Inside a plate-boundary zone the ground does not move rigidly. In the
western United States, ground east of the San Andreas fault system moves
up to about 15 mm/yr relative to stable North America (the Sierra
Nevada-Great Valley block, the Basin and Range). That motion is real; a
plate-fixed frame shows it rather than removing it.

Checked on real stations, 8 Oct 2026 (review_2026-10-08/p23/plate/ngl_igs20/,
NGL's IGS20 MIDAS velocities midas.IGS.txt, .NA.txt and .AU.txt from
gps_timeseries/IGS20/midas/, records to 2026.74): of 2366 stations in
stable North America (30-45 N, 100-75 W; at least 5 years, rate sigma under
0.5 mm/yr), the median velocity relative to the plate is -0.07 east, -0.01
north mm/yr, and 91% are under 1 mm/yr (IGS20: 15 mm/yr). Of 648
Australian stations: -0.02 east, -0.15 north, 88% under 1 mm/yr (IGS20:
60 mm/yr). NGL's own Australia-fixed frame agrees with this one to about
0.1 mm/yr; its North America frame (NGL's own pole) sits 0.8 mm/yr
further north in Indiana. From the daily series (straight line,
2016-2026): CA07 (Appin's reference station) -0.30 east, +0.78 north
(57 mm/yr in IGS20); LAMC (Indiana) +0.23 east, +0.30 north (15 mm/yr).
CA07's north residual is the station's own: its IGS20 MIDAS rate gives
+0.57 relative to this model and NGL's Australia-fixed rate +0.63.
(A first pass read NGL's /velocities/ folder, which NGL's README there
says is IGS14 and superseded; the same check on it is kept as
check_midas_networks_igs14.out and gives the same picture within about
0.2 mm/yr.)

ITRF2014-PMM (Altamimi et al. 2017) is the model for IGS14 series. The
pipeline's ground truth reads only IGS20 (ground_truth.load_gnss_series
refuses IGS14), so it is not included.
"""
import logging

import numpy as np

from .config import PipelineConfig

log = logging.getLogger(__name__)

# A station must be at least this consistent with its neighbours to count
# toward the plate velocity. Rejects the sites whose own solution is poor.
MAX_SIGMA_MM_YR = 1.5


def plate_velocity_from_gnss(cfg: PipelineConfig, radius_km: float = 250.0
                             ) -> dict | None:
    """
    Horizontal plate velocity from the far-field GNSS field.

    Uses the MEDIAN rather than a fitted Euler pole. Over a few hundred
    kilometres a rigid plate is indistinguishable from a constant, the
    median is unaffected by one bad station, and it needs no pole
    geometry. Reports the spread so the rigid assumption can be checked
    rather than assumed: if east velocities scatter by more than about a
    mm/yr across the network, the plate is not behaving rigidly here and
    this correction should not be applied.
    """
    from pathlib import Path
    from . import gnss_validation as gv

    stations = gv.load_stations(Path(cfg.project_dir) / "gnss")
    if not stations:
        log.warning("No GNSS catalogue available - cannot estimate plate motion")
        return None
    min_lon, min_lat, max_lon, max_lat = cfg.bbox
    clat, clon = (min_lat + max_lat) / 2.0, (min_lon + max_lon) / 2.0
    near = [s for s in gv.nearby(stations, clat, clon, radius_km)
            if max(s["sigma_east_mm_yr"], s["sigma_north_mm_yr"]) <= MAX_SIGMA_MM_YR]
    if len(near) < 5:
        log.warning("Only %d usable GNSS stations within %.0f km - too few to "
                    "characterise plate motion", len(near), radius_km)
        return None

    ve = np.array([s["v_east_mm_yr"] for s in near])
    vn = np.array([s["v_north_mm_yr"] for s in near])
    out = {
        "n_stations": len(near),
        "v_east_mm_yr": float(np.median(ve)),
        "v_north_mm_yr": float(np.median(vn)),
        "spread_east_mm_yr": float(np.std(ve)),
        "spread_north_mm_yr": float(np.std(vn)),
        "radius_km": radius_km,
    }
    log.info("Plate motion from %d GNSS stations within %.0f km: "
             "E %+.2f (spread %.2f), N %+.2f (spread %.2f) mm/yr",
             out["n_stations"], radius_km, out["v_east_mm_yr"],
             out["spread_east_mm_yr"], out["v_north_mm_yr"],
             out["spread_north_mm_yr"])
    if out["spread_east_mm_yr"] > 1.5 or out["spread_north_mm_yr"] > 1.5:
        log.warning("Station-to-station spread exceeds 1.5 mm/yr - the plate is "
                    "NOT behaving rigidly across this network, so a single "
                    "velocity is a poor description. Treat the correction as "
                    "indicative and check it against the empirical ramp.")
    return out


def predicted_ramp(los_east: np.ndarray, los_north: np.ndarray,
                   plate: dict, ref_rc: tuple | None = None) -> np.ndarray:
    """
    The line-of-sight ramp a rigid plate velocity imprints, in mm/yr.

    Returned relative to the reference pixel, because that is the only
    part a referenced product contains. With no reference given, the scene
    median stands in - the shape of the ramp is identical either way, only
    its datum differs, and the datum is unmeasurable regardless.
    """
    ve, vn = plate["v_east_mm_yr"], plate["v_north_mm_yr"]
    ramp = los_east * ve + los_north * vn
    if ref_rc is not None:
        r, c = ref_rc
        if 0 <= r < ramp.shape[0] and 0 <= c < ramp.shape[1] and np.isfinite(ramp[r, c]):
            return ramp - ramp[r, c]
    with np.errstate(all="ignore"):
        return ramp - np.nanmedian(ramp)


def compare_with_empirical(velocity: np.ndarray, ramp: np.ndarray,
                           mask: np.ndarray | None = None) -> dict:
    """
    Check the predicted ramp against a plane fitted to the data.

    This is the test that decides whether the correction is safe to apply.
    The plane is fitted on the robust middle of the velocity distribution
    so real deformation does not drag it. If the two agree in magnitude
    and orientation, the long-wavelength signal is understood and removing
    it is defensible. If the empirical plane is much larger, something
    else is present - residual troposphere, an unwrapping offset, real
    regional motion - and subtracting the plate term alone will leave it
    while subtracting the empirical plane would remove real signal too.
    """
    ok = np.isfinite(velocity) & np.isfinite(ramp)
    if mask is not None:
        ok &= mask
    if ok.sum() < 500:
        return {"ok": False, "reason": "too few valid pixels"}

    ny, nx = velocity.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    lo, hi = np.nanpercentile(velocity[ok], [20, 80])
    fit = ok & (velocity >= lo) & (velocity <= hi)
    A = np.c_[xx[fit].ravel(), yy[fit].ravel(), np.ones(fit.sum())]
    sol, *_ = np.linalg.lstsq(A, velocity[fit].ravel(), rcond=None)
    plane = sol[0] * xx + sol[1] * yy + sol[2]

    emp = float(np.ptp(plane[ok]))
    pred = float(np.ptp(ramp[ok]))
    corr = float(np.corrcoef(plane[ok], ramp[ok])[0, 1])
    out = {"ok": True, "empirical_span_mm_yr": round(emp, 2),
           "predicted_span_mm_yr": round(pred, 2),
           "correlation": round(corr, 3),
           "ratio": round(emp / pred, 2) if pred > 1e-6 else float("inf")}
    log.info("Long-wavelength ramp: empirical plane spans %.2f mm/yr, plate "
             "motion predicts %.2f mm/yr, correlation %+.3f",
             emp, pred, corr)
    if corr < 0.5:
        log.warning("The fitted plane does not resemble the plate-motion "
                    "prediction (correlation %+.2f). The scene's "
                    "long-wavelength signal is NOT dominated by plate motion, "
                    "so removing the plate term will not flatten it and "
                    "removing the empirical plane would take real signal with "
                    "it. Investigate before correcting.", corr)
    elif out["ratio"] > 2.0:
        log.warning("The empirical plane is %.1fx the plate-motion prediction. "
                    "Plate motion explains only part of the ramp; the rest is "
                    "unexplained and should not be removed blindly.",
                    out["ratio"])
    return out


def correct(cfg: PipelineConfig, velocity: np.ndarray, grid: dict,
            frame=None, ref_rc=None, apply: bool = False) -> tuple:
    """
    Estimate, report and optionally remove the plate-motion ramp.

    Returns (velocity, record). `apply=False` estimates and reports only,
    which is the default everywhere: the report is the point, and a
    correction that has not been looked at is how a real signal quietly
    disappears.
    """
    from . import static_layers as sl

    record = {"applied": False}
    plate = plate_velocity_from_gnss(cfg)
    if plate is None:
        record["reason"] = "no plate velocity available"
        return velocity, record
    record["plate"] = plate

    geom = sl.load(cfg, grid, frame=frame)
    if "los_east" not in geom or "los_north" not in geom:
        log.warning("No per-pixel line-of-sight vectors (DISP-S1-STATIC not "
                    "available). A scene-constant look vector cannot produce a "
                    "ramp, so there is nothing to correct.")
        record["reason"] = "no per-pixel LOS geometry"
        return velocity, record

    ramp = predicted_ramp(geom["los_east"], geom["los_north"], plate, ref_rc)
    record["comparison"] = compare_with_empirical(velocity, ramp)
    record["ramp_span_mm_yr"] = float(np.nanmax(ramp) - np.nanmin(ramp))

    if apply:
        velocity = velocity - ramp
        record["applied"] = True
        log.info("Removed the plate-motion ramp (%.2f mm/yr across the scene)",
                 record["ramp_span_mm_yr"])
    return velocity, record


# ---------------------------------------------------------------- the published plate model (Decision 5)
ITRF2020_PMM_URL = "https://itrf.ign.fr/docs/solutions/itrf2020/ITRF2020-PMM.dat"
ITRF2020_PMM_SHA256 = "9862e83f3596f2886ab5dbe8bea4a66a8e9b3518c8f0124815986fcc37ea1b5f"   # fetched 8 Oct 2026
ITRF2020_PMM_CITATION = ("Altamimi, Z., Metivier, L., Rebischung, P., Collilieux, X., Chanard, K., Barneoud, J. "
                         "(2023) ITRF2020 plate motion model. Geophysical Research Letters 50(24), e2023GL106373, "
                         "https://doi.org/10.1029/2023GL106373")
# Table S1 of ITRF2020-PMM.dat: rotation pole cartesian components, degrees per million years, as published
ITRF2020_PMM = {
    "AMUR": (-0.0364, -0.1532, 0.2325),
    "ANTA": (-0.0746, -0.0866, 0.1882),
    "ARAB": (0.3135, -0.0406, 0.3994),
    "AUST": (0.4132, 0.3265, 0.3398),
    "CARB": (0.0576, -0.3949, 0.2017),
    "EURA": (-0.0237, -0.1442, 0.2091),
    "INDI": (0.3159, 0.0037, 0.4010),
    "NAZC": (-0.0907, -0.4336, 0.4459),
    "NOAM": (0.0126, -0.1849, -0.0272),
    "NUBI": (0.0250, -0.1625, 0.1991),
    "PCFC": (-0.1122, 0.2836, -0.5984),
    "SOAM": (-0.0725, -0.0784, -0.0437),
    "SOMA": (-0.0225, -0.1997, 0.2401),
}
ITRF2020_ORB_MM_YR = (0.37, 0.35, 0.74)          # origin rate bias TX, TY, TZ (same file)
PLATE_NAMES = {"AMUR": "Amurian", "ANTA": "Antarctic", "ARAB": "Arabian", "AUST": "Australian",
               "CARB": "Caribbean", "EURA": "Eurasian", "INDI": "Indian", "NAZC": "Nazca",
               "NOAM": "North American", "NUBI": "Nubian", "PCFC": "Pacific", "SOAM": "South American",
               "SOMA": "Somalian"}
# rms of the model's post-fit horizontal residuals (east, north; mm/yr) over the sites each plate was fitted
# to, from ITRF2020-PMM-residuals.dat (Table S2): how far a stable site on that plate departs from rigid motion
ITRF2020_PMM_RESIDUAL_RMS = {"AMUR": (0.18, 0.32), "ANTA": (0.29, 0.40), "ARAB": (0.14, 0.12),
                             "AUST": (0.23, 0.22), "CARB": (0.23, 0.78), "EURA": (0.25, 0.25),
                             "INDI": (0.41, 0.35), "NAZC": (0.60, 0.61), "NOAM": (0.33, 0.37),
                             "NUBI": (0.28, 0.35), "PCFC": (0.53, 0.34), "SOAM": (0.41, 0.37),
                             "SOMA": (0.54, 0.35)}
PLATE_MODELS = {"ITRF2020": {"poles_deg_per_myr": ITRF2020_PMM, "orb_mm_yr": ITRF2020_ORB_MM_YR,
                             "name": "ITRF2020-PMM", "source_url": ITRF2020_PMM_URL,
                             "sha256": ITRF2020_PMM_SHA256, "citation": ITRF2020_PMM_CITATION,
                             "residual_rms_mm_yr": ITRF2020_PMM_RESIDUAL_RMS,
                             "series_frames": ("IGS20", "ITRF2020")}}

# Where 'auto' may choose a plate. Coarse boxes, deliberately: each lies well inside the continent named, or
# is flagged as a boundary zone where the stations' own velocities decide between the plates listed (they
# differ by tens of mm/yr there, far more than any station's error). Anywhere else the plate must be named.
PLATE_REGIONS = (
    # Australia in two boxes, so that Sumba, Savu, Rote and Timor (8-11 S, 119-127 E: the Banda arc, where the
    # Sunda and Australian plates meet) and southern New Guinea fall outside and must be named
    {"region": "Australian continent south of 13 S, and Tasmania", "lat": (-45.0, -13.0), "lon": (112.0, 155.0),
     "candidates": ("AUST",), "boundary_zone": False},
    {"region": "northern Australia (the Top End and Cape York)", "lat": (-13.0, -10.5), "lon": (129.0, 146.0),
     "candidates": ("AUST",), "boundary_zone": False},
    {"region": "North America east of the Rocky Mountain front (stable interior, Gulf and Atlantic coasts)",
     "lat": (24.0, 72.0), "lon": (-105.0, -52.0), "candidates": ("NOAM",), "boundary_zone": False},
    {"region": "western North America (the Pacific-North America plate-boundary zone)",
     "lat": (30.0, 60.0), "lon": (-128.0, -105.0), "candidates": ("NOAM", "PCFC"), "boundary_zone": True,
     "note": ("East of the San Andreas fault system the ground is on the North American side but moves up to "
              "about 15 mm/yr relative to stable North America (the Sierra Nevada-Great Valley block, the Basin "
              "and Range); west of it the ground moves with the Pacific plate. The plate is chosen by the "
              "stations' velocities, and what is left after its removal is real motion, not an error.")},
    # Europe in two boxes, so that the Po plain and the north Adriatic (Adria, which moves a few mm/yr relative to
    # Eurasia) fall outside and must be named
    {"region": "western Europe north of 45 N, west of the Alps (France, the Low Countries, the British Isles)",
     "lat": (45.0, 72.0), "lon": (-11.0, 6.5), "candidates": ("EURA",), "boundary_zone": False},
    {"region": "central, northern and eastern Europe north of the Alps", "lat": (46.5, 72.0), "lon": (6.5, 40.0),
     "candidates": ("EURA",), "boundary_zone": False},
)
MIN_VELOCITY_SPAN_YR = 0.5       # a station's velocity decides between plates only with this much record


def _model(model: str) -> dict:
    key = str(model or "ITRF2020").upper().replace("-PMM", "")
    if key not in PLATE_MODELS:
        raise ValueError(f"plate model {model!r} is not one of {', '.join(PLATE_MODELS)} (IGS20 series are "
                         "aligned to ITRF2020, so ITRF2020 is the one to use)")
    return PLATE_MODELS[key]


def plate_code(plate: str, model: str = "ITRF2020") -> str:
    """The model's code for `plate` ('noam', 'NOAM' or 'North American'
    all give NOAM); ValueError when the model has no such plate."""
    s = str(plate or "").strip()
    poles = _model(model)["poles_deg_per_myr"]
    if s.upper() in poles:
        return s.upper()
    for code, name in PLATE_NAMES.items():
        if s.lower() in (name.lower(), f"{name.lower()} plate") and code in poles:
            return code
    raise ValueError(f"plate {plate!r} is not in {_model(model)['name']}: use one of {', '.join(sorted(poles))}")


def _ecef_mm(lat, lon, h=0.0):
    """Earth-centred X, Y, Z in mm of geodetic lat/lon (degrees) and
    ellipsoidal height h (m), on the GRS80 ellipsoid (ITRF's)."""
    a, f = 6378137.0, 1.0 / 298.257222101
    e2 = f * (2.0 - f)
    la, lo = np.radians(np.asarray(lat, dtype=float)), np.radians(np.asarray(lon, dtype=float))
    h = np.asarray(h, dtype=float)
    N = a / np.sqrt(1.0 - e2 * np.sin(la) ** 2)
    return ((N + h) * np.cos(la) * np.cos(lo) * 1000.0, (N + h) * np.cos(la) * np.sin(lo) * 1000.0,
            (N * (1.0 - e2) + h) * np.sin(la) * 1000.0)


def plate_velocity_enu(plate: str, lat, lon, h=0.0, model: str = "ITRF2020"):
    """
    The horizontal velocity (east, north; mm/yr) in ITRF2020 of a point
    that moves rigidly with `plate` at geodetic lat/lon (degrees), from
    the published model: omega x X + ORB, the ORB's vertical part
    discarded as the ITRF advises (module docstring). Scalars or arrays.
    Subtracting it from an IGS20 velocity gives the velocity relative to
    the plate.
    """
    m = _model(model)
    code = plate_code(plate, model)
    w = np.radians(np.asarray(m["poles_deg_per_myr"][code], dtype=float)) / 1e6        # rad/yr
    x, y, z = _ecef_mm(lat, lon, h)
    tx, ty, tz = m["orb_mm_yr"]
    vx = w[1] * z - w[2] * y + tx
    vy = w[2] * x - w[0] * z + ty
    vz = w[0] * y - w[1] * x + tz
    la, lo = np.radians(np.asarray(lat, dtype=float)), np.radians(np.asarray(lon, dtype=float))
    ve = -np.sin(lo) * vx + np.cos(lo) * vy
    vn = -np.sin(la) * np.cos(lo) * vx - np.sin(la) * np.sin(lo) * vy + np.cos(la) * vz
    if np.ndim(ve) == 0:
        return float(ve), float(vn)
    return ve, vn


def plate_los_rate(los_east, los_north, lat, lon, plate: str, model: str = "ITRF2020"):
    """The plate's rigid motion on the line of sight (mm/yr, positive
    towards the satellite with a ground-to-satellite unit vector): what a
    plate-fixed frame removes at each point. Vertical: none."""
    ve, vn = plate_velocity_enu(plate, lat, lon, model=model)
    return np.asarray(los_east, dtype=float) * ve + np.asarray(los_north, dtype=float) * vn


def plate_at(lat: float, lon: float) -> dict | None:
    """The PLATE_REGIONS entry holding (lat, lon), or None where 'auto'
    does not choose (the plate must be named)."""
    lon = (float(lon) + 180.0) % 360.0 - 180.0
    for r in PLATE_REGIONS:
        if r["lat"][0] <= float(lat) <= r["lat"][1] and r["lon"][0] <= lon <= r["lon"][1]:
            return r
    return None


def choose_plate(lats, lons, ve=None, vn=None, plate: str = "auto", model: str = "ITRF2020") -> dict:
    """
    One plate for a set of stations (one frame for one datum).

    plate: a code or name (used as given), or 'auto': the PLATE_REGIONS
    entry each station lies in. In a boundary zone with two candidate
    plates, the stations' own IGS20 horizontal velocities `ve`, `vn`
    (mm/yr; NaN where a station's record is too short) decide: each
    station takes the plate its velocity is closest to, and they must
    agree. Raises ValueError, in words, when a station lies where 'auto'
    does not choose, when the stations lie in regions with no plate in
    common, or when the velocities are missing or disagree.

    Returns {plate, name, model, how, region, boundary_zone, note,
    candidates: {code: median horizontal misfit mm/yr} (boundary zones)}.
    """
    m = _model(model)
    lats = np.atleast_1d(np.asarray(lats, dtype=float))
    lons = np.atleast_1d(np.asarray(lons, dtype=float))
    base = {"model": m["name"], "candidates": None, "boundary_zone": None, "region": None, "note": None}
    if str(plate or "auto").strip().lower() != "auto":
        code = plate_code(plate, model)
        return {**base, "plate": code, "name": PLATE_NAMES.get(code, code), "how": "named in the settings"}
    if lats.size == 0:
        raise ValueError("no station positions to choose a plate from")
    regions = []
    for la, lo in zip(lats, lons):
        r = plate_at(la, lo)
        if r is None:
            raise ValueError(f"a station at {la:.3f}, {lo:.3f} lies outside the regions where the plate is chosen "
                             "automatically (plate_motion.PLATE_REGIONS): name it (reference_gnss_plate, e.g. "
                             f"{', '.join(sorted(m['poles_deg_per_myr']))})")
        regions.append(r)
    common = [c for c in regions[0]["candidates"] if all(c in r["candidates"] for r in regions)]
    if not common:
        raise ValueError("the stations lie in regions with no plate in common ("
                         + "; ".join(sorted({r['region'] for r in regions})) + "): one datum needs one frame, so "
                         "name the plate or use stations on one plate")
    zone = any(r["boundary_zone"] for r in regions)
    names = sorted({r["region"] for r in regions})
    note = " ".join(sorted({r["note"] for r in regions if r.get("note")})) or None
    out = {**base, "region": "; ".join(names), "boundary_zone": zone, "note": note}
    if len(common) == 1:
        code = common[0]
        return {**out, "plate": code, "name": PLATE_NAMES.get(code, code),
                "how": f"automatic: the stations lie in the region '{out['region']}'"}
    if ve is None or vn is None:
        raise ValueError(f"the stations lie in {out['region']}, where {' or '.join(common)} is possible: their "
                         "velocities are needed to choose (or name the plate)")
    ve = np.atleast_1d(np.asarray(ve, dtype=float))
    vn = np.atleast_1d(np.asarray(vn, dtype=float))
    have = np.isfinite(ve) & np.isfinite(vn)
    if not have.any():
        raise ValueError(f"the stations lie in {out['region']}, where {' or '.join(common)} is possible, and no "
                         f"station has {MIN_VELOCITY_SPAN_YR:g} yr of record to choose by: name the plate")
    mis = {}
    for c in common:
        pe, pn = plate_velocity_enu(c, lats[have], lons[have], model=model)
        mis[c] = np.hypot(ve[have] - pe, vn[have] - pn)
    best_each = [min(common, key=lambda c: mis[c][i]) for i in range(int(have.sum()))]
    if len(set(best_each)) > 1:
        raise ValueError(f"the stations' velocities put them on different plates ({', '.join(sorted(set(best_each)))}"
                         "): one datum needs one frame, so name the plate or use stations on one plate")
    code = best_each[0]
    return {**out, "plate": code, "name": PLATE_NAMES.get(code, code),
            "candidates": {c: round(float(np.median(v)), 2) for c, v in mis.items()},
            "how": (f"automatic: the stations lie in the region '{out['region']}', and their velocities are "
                    f"closest to {PLATE_NAMES.get(code, code)} plate motion")}
