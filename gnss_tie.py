"""
Anchor the record to GNSS: the regional background motion over the record's
own dates, and a check at any station inside the processed area.

WHY NOT A SINGLE-STATION SHIFT
------------------------------
The textbook tie shifts the whole field so the InSAR rate at a station equals
the station's rate. That imports whatever the station's radar cell carries:
on this farmland a cell of coherence 0.7-0.8 reads ~4.5 mm/yr low from the
distributed-scatterer bias alone, and one cell's rate is repeatable only to a
few mm/yr. So the anchor here is the quantity GNSS measures well and InSAR
does not measure at all - how the region moves as a whole:

    absolute rate (site) = class-matched relative rate (site)
                           + regional background, projected into the LOS

The background is the vertical rate of the continuous stations within
`radius_km`, each fitted over the RECORD's dates (MIDAS: the median of
one-year differences - insensitive to seasonal cycles and to steps from
equipment changes; Blewitt et al. 2016). It rests on the assumption that the
far-field farmland moves with the regional GNSS field; a station inside the
processed area can test exactly that, over the years it and the record
share (see THE CHECK AT A STATION INSIDE THE AREA).

THE ESTIMATOR (E15, 7 Oct 2026). Until 7 Oct midas_rate paired each day with
the first one at least 0.97 yr later (a median gap of 0.972 yr), took a plain
median and quoted 1.2533 x sd / sqrt(n/4) - MIDAS without its trimming or
its error scaling, and with no step screening. It now follows MIDAS
(Blewitt et al. 2016, JGR 121, 2054-2068) in everything that moves a rate
here (MIDAS also pairs backwards in time where gaps leave days unpaired;
NGL's daily series have few gaps, so forward pairs alone are kept): each
day paired with the one closest to one year later (within 0.01 yr); no pair
may straddle a step - an
equipment change or earthquake in NGL's steps file, or a change of the
antenna height the series was processed with; slopes beyond 2 sigma of the
median trimmed and the median re-taken; sigma = 3 x 1.2533 x sd / sqrt(n/4)
(the factor 3 is MIDAS's own, which makes its errors realistic: the old
ones were 2-2.6 x smaller than NGL's published sigmas on the same spans).
An antenna height held for fewer than 365 daily solutions is a metadata
slip, not a change: INEV carries 0 m instead of 0.035 m on 192 of 1,930
days and those days sit +35 mm high, so they are dropped. The steps file
also moved INVI: a 6 Jun 2024 antenna change put a step of several mm into
every pair across it (-1.51 -> -1.09 mm/yr over the record).

The background is ROBUST to a local mover: two sites within 10 km share the
regional field, so their differenced daily series measures local motion
precisely (common-mode noise cancels). When that difference's MIDAS rate
lies beyond the 95% limit of its own sigma (familywise over the pairs
tested), the site of the pair farther from the median of the other sites
is left out of the background before the weighted mean (recorded in
outlier_test). Site B's VINU, 6.9 km from INVI, is the case it was built
for: VINU - INVI = -1.83 +/- 1.15 mm/yr over 2016.0-2021.2, inside the
2.25 mm/yr limit, so VINU stays - a 1.6 sigma local excess is not evidence
enough to discard a station, and its disagreement widens the background's
uncertainty (scatter between the sites) instead.

Horizontal motion is left out of the background: in IGS20 the stations carry
~15 mm/yr of North America plate motion, uniform over the scene and already
removed by a relative product; their horizontal motion relative to the plate
is a fraction of a mm/yr, below every uncertainty here. Vertical-equivalent
figures divide LOS by the LOS up-component and assume the motion is vertical,
which is the standard reading of subsidence and is said wherever used.

UNCERTAINTY OF THE BACKGROUND (review of 5 Oct 2026). The formal error of a
station's rate (0.3-0.4 mm/yr) says how well that station's own motion is
known, not how well it stands for the ground 10-40 km away: on Site B the
three sites differ by 1.28 mm/yr (INES+INEV -1.10, INVI -1.54, VINU -3.51),
and at Site D only one station (INLN) lies within 40 km. So the background
uncertainty is the largest of the formal error, the scatter between the sites
used, and a REGIONAL FLOOR - the scatter of the MIDAS vertical rates of every
NGL station within 100 km whose record overlaps ours by 3 years or more
(cached catalogue, monuments within 1 km averaged, 5 Oct 2026: 0.95
Site B, 1.01 Site A, 0.91 Site D mm/yr - so the Site A
background reads -1.09 +/- 1.01 instead of +/- 0.73, Site B is unchanged at
+/- 1.28, and Site D's INLN-only tie +/- 0.91 instead of its 0.29 formal
error). Those are the 5 Oct estimator's figures; with the 7 Oct one (THE
ESTIMATOR) the station errors are 0.9-1.4 mm/yr and the backgrounds
Site B -1.50 +/- 1.50 (scatter between INES+INEV -0.92, INVI -1.09 and
VINU -3.60) and Site A -0.76 +/- 1.01 (regional floor; ILWS +0.21,
INES+INEV -0.94, INVI -1.09). gnss_tie.json also records how much the background moves
when each site is left out and at radii of 30/40/50 km - from stations
actually downloaded only: a radius within which a catalogue station that
would qualify (3 yr or more of overlap) was never downloaded reads "not
evaluated", never a number (Site B's 50 km read -1.85, "unchanged", while
INDA and INJS at 44-45 km, which would move it by about 0.6, were missing).

THE CHECK AT A STATION INSIDE THE AREA (E16, 7 Oct 2026). Until 7 Oct it set
the radar's whole-record (2016-2026) rate at the station against the
station's 2.2-year rate minus the whole-record background, with a 10-year
threshold, and could not fail (Site A LAMC "the two agree"). Now both
sides cover the same dates only - the SAR dates inside the station's record:
the radar side is each cell's rate over those dates, class-matched against
farmland over the same dates, tested against the farmland null of that
same-window field (significance.assess); the GNSS side is the station minus
the background sites over the same dates (so the region's own movement,
which a relative radar record cannot see, cancels). When the 95% tolerance
of their difference exceeds 0.25 in/yr (6.35 mm/yr) vertical, the overlap
is too short to test anything that matters: the verdict is "not testable",
`consistent` None - never "agree" (a difference beyond even that tolerance
is still "differ"). LAMC: 41 dates over 2016.0-2018.2, a tolerance near
1 in/yr, so not testable. Its farmland leaves out the report areas other
than the whole study area plus 500 m, as the report's comparison ground
does (E28; F7(d), 7 Oct 2026; recorded as comparison_ground).

WHETHER THE STATIONS WERE SCREENED (F7(b), 7 Oct 2026). steps_file_read
(and each station's screening.steps_file_read) says whether NGL's steps file
was read; when it was not, equipment changes and earthquakes were not
screened and release.py puts that on the reviewer's list
(unscreened_reason), as it does for a tie made before the screening.

ALWAYS A FRESH FILE. run() writes deliverables/gnss_tie.json on every call,
with "background": null and a plain-language "reason" when no tie is
possible, so a tie from an earlier run (another radius, other dates) can
never be read as this run's. Readers use load_valid(cfg, dates), which also
refuses a file whose record window is not the record's or that is older than
the velocity product.

Data: Nevada Geodetic Laboratory daily solutions, IGS20
(https://geodesy.unr.edu), cached under <projects>/gnss/<SITE>/.
"""
import json
import logging
import math
from datetime import date, datetime
from pathlib import Path
from statistics import NormalDist

import numpy as np

log = logging.getLogger(__name__)
NGL_TENV3 = "https://geodesy.unr.edu/gps_timeseries/IGS20/tenv3/IGS20/{site}.tenv3"
NGL_STEPS = "https://geodesy.unr.edu/NGLStationPages/steps.txt"
STEPS_FILE = "ngl_steps.txt"      # cached beside the series, refreshed whenever a series is downloaded
MM_PER_IN = 25.4
MIN_OVERLAP_YEARS = 3.0           # a station's record must cover this much of ours to count
REGIONAL_FLOOR_RADIUS_KM = 100.0  # stations whose scatter floors the background uncertainty
SENSITIVITY_RADII_KM = (30.0, 40.0, 50.0)
UNTIED_SUFFIX = "so every figure is relative to the surrounding farmland."
# Nominal incidence by sensor, only when the product has no geometry layer
# (M1, 8 Oct 2026). Sentinel-1 IW mid-swath 38.5 deg (the value used since
# Sep 2026). NISAR L-SAR's swath spans roughly 33-47 deg, so ~40 deg mid-swath
# (approximate); NISAR products always carry incidence (prep_nisar
# interpolates it from the GUNW radar-grid cube), so a NISAR run should never
# reach this table. stack_qc.product_wavelength_m / product_look_side read the
# rest of the sensor from the product.
NOMINAL_INCIDENCE_DEG = {"sentinel-1": 38.5, "nisar": 40.0}
NOMINAL_LOS_UP = math.cos(math.radians(NOMINAL_INCIDENCE_DEG["sentinel-1"]))   # only when the product has no geometry layer
NOMINAL_LOS_UP_BY_SENSOR = {k: math.cos(math.radians(v)) for k, v in NOMINAL_INCIDENCE_DEG.items()}
# MIDAS (Blewitt et al. 2016)
PAIR_TOLERANCE_YEARS = 0.01       # a pair is the day closest to one year later, within this
MIDAS_TRIM_SIGMA = 2.0            # slopes beyond this many sigma of the median are trimmed
MIDAS_SIGMA_SCALE = 3.0           # MIDAS's empirical factor for realistic rate errors
MIN_ANTENNA_HEIGHT_DAYS = 365     # an antenna height held for fewer daily solutions is a metadata slip
# robust background: a local mover shows against a close neighbour
NEIGHBOUR_KM = 10.0
OUTLIER_FALSE_ALARM = 0.05        # familywise, over the neighbour pairs tested
# the check at a station inside the area
SAR_GNSS_MATCH_DAYS = 6           # GNSS days averaged around each SAR date
INSIDE_CHECK_USEFUL_MM_YR = 0.25 * MM_PER_IN    # vertical; a coarser check tests nothing that matters
Z95 = 1.959964


def default_radius_km() -> float:
    """The one radius the config default, the builder and the dashboard share."""
    try:
        from .config import GNSS_TIE_RADIUS_KM
        return float(GNSS_TIE_RADIUS_KM)
    except ImportError:                                   # pragma: no cover - config always has it
        return 40.0


def radius_km(cfg) -> float:
    v = getattr(cfg, "gnss_tie_radius_km", None)
    return float(default_radius_km() if v is None else (v or 0.0))


def fetch_tenv3(site: str, cache_root: Path, timeout: int = 180) -> Path:
    """The station's daily series (cached), and NGL's steps file with it: a
    series newer than the cached steps file may carry an equipment change
    the file does not name yet, so the file is fetched when missing or older
    than the series. A steps file that cannot be had is logged, and the tie
    records that equipment changes were not screened."""
    import urllib.request
    d = cache_root / site; d.mkdir(parents=True, exist_ok=True)
    p = d / f"{site}.tenv3"
    if not p.exists() or p.stat().st_size < 1000:
        with urllib.request.urlopen(NGL_TENV3.format(site=site), timeout=timeout) as r:
            p.write_bytes(r.read())
    s = cache_root / STEPS_FILE
    if not s.exists() or s.stat().st_mtime < p.stat().st_mtime - 86400:
        try:
            fetch_steps(cache_root, timeout)
        except Exception as exc:                          # noqa: BLE001 - recorded by tie() as not screened
            log.warning("NGL steps file not downloaded (%s): equipment changes cannot be screened", exc)
    return p


def fetch_steps(cache_root: Path, timeout: int = 180) -> Path:
    """Download NGL's list of equipment changes and earthquakes (steps.txt)."""
    import urllib.request
    p = Path(cache_root) / STEPS_FILE
    p.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(NGL_STEPS, timeout=timeout) as r:
        data = r.read()
    if len(data) < 1000:
        raise ValueError(f"steps file is only {len(data)} bytes")
    tmp = p.with_suffix(".tmp")
    tmp.write_bytes(data)
    if not read_steps(tmp):                               # an error page, not the list: keep the cached file
        tmp.unlink()
        raise ValueError("the download names no equipment change or earthquake (not NGL's steps file)")
    tmp.replace(p)
    return p


_MONTHS = {m: i + 1 for i, m in enumerate(("JAN", "FEB", "MAR", "APR", "MAY", "JUN",
                                           "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"))}


def ngl_date(yymmmdd: str) -> date:
    """NGL's date text (21AUG27) as a calendar date (shared with ground_truth)."""
    y = int(yymmmdd[:2])
    y += 1900 if y >= 80 else 2000
    return date(y, _MONTHS[yymmmdd[2:5].upper()], int(yymmmdd[5:7]))


def _ngl_decyear(yymmmdd: str) -> float:
    """NGL's date (21AUG27) as NGL's own decimal year (2000 + days since
    2000-01-01 / 365.25), the convention of the tenv3 time column, so a step
    lands exactly on its day's sample."""
    return 2000.0 + (ngl_date(yymmmdd) - date(2000, 1, 1)).days / 365.25


def read_steps(path: Path) -> dict[str, list[dict]]:
    """NGL steps file -> {site: [{"t": decimal year, "event": text}]}: code 1
    (equipment: antenna, radome, receiver, elevation cut-off) and code 2
    (a possible earthquake offset)."""
    out: dict[str, list[dict]] = {}
    for line in Path(path).read_text(errors="replace").splitlines():
        p = line.split()
        if len(p) < 3 or p[2] not in ("1", "2"):
            continue
        try:
            t = _ngl_decyear(p[1])
        except (KeyError, ValueError):
            continue
        what = (p[3] if len(p) > 3 else "equipment change") if p[2] == "1" else "possible earthquake offset"
        out.setdefault(p[0], []).append({"t": round(t, 4), "event": what})
    return out


def read_tenv3(path: Path, full: bool = False) -> dict:
    """decimal year and east/north/up in mm (NGL tenv3: integer part + fraction,
    metres), and the antenna height (m) the solution was processed with.

    full=True (the validation harness, ground_truth.py) adds each day's
    calendar date ("date"), NGL's formal 1-sigma of east/north/up in mm
    ("se", "sn", "su") and the station height in metres ("h"). The default
    output is unchanged."""
    rows = [l.split() for l in Path(path).read_text().splitlines()[1:] if l.strip()]
    a = lambda i: np.array([float(r[i]) for r in rows])  # noqa: E731
    out = {"site": rows[0][0], "t": a(2),
           "e": (a(7) + a(8)) * 1000.0, "n": (a(9) + a(10)) * 1000.0, "u": (a(11) + a(12)) * 1000.0,
           "ant": a(13), "lat": float(rows[-1][20]), "lon": float(rows[-1][21])}
    if full:
        out.update({"date": [ngl_date(r[1]) for r in rows],
                    "se": a(14) * 1000.0, "sn": a(15) * 1000.0, "su": a(16) * 1000.0,
                    "h": float(rows[-1][22])})
    return out


def screen_series(g: dict, steps=()) -> dict:
    """
    The station's vertical series made ready for MIDAS (E15): the step
    epochs no 1-year pair may straddle - `steps` (NGL steps file, decimal
    years) plus every change of the antenna height the series was processed
    with - and the days dropped. An antenna height held for fewer than
    MIN_ANTENNA_HEIGHT_DAYS daily solutions is a metadata slip, not an
    equipment change (INEV: 0 m instead of 0.035 m on 192 days, +35 mm
    high): those days are dropped. Such a run is shorter than a year, so
    as a step it could give no pair anyway. A series without the antenna
    column is screened by `steps` only.

    Returns {"t", "u", "steps", "antenna_epochs_dropped",
    "antenna_heights_dropped_m"}.
    """
    t, u = np.asarray(g["t"], dtype=float), np.asarray(g["u"], dtype=float)
    keep = np.ones(t.size, dtype=bool)
    events = {float(s) for s in steps}
    dropped: list[float] = []
    ant = g.get("ant")
    if ant is not None and len(ant) == t.size:
        ant = np.asarray(ant, dtype=float)
        vals, cnt = np.unique(ant, return_counts=True)
        if vals.size > 1:
            held = vals[cnt >= MIN_ANTENNA_HEIGHT_DAYS]
            if held.size:                                 # else no height held a year: every change is a step
                keep = np.isin(ant, held)
                dropped = [round(float(v), 4) for v in vals[~np.isin(vals, held)]]
            a_k, t_k = ant[keep], t[keep]
            events |= {float(t_k[i + 1]) for i in np.flatnonzero(np.diff(a_k) != 0)}
    return {"t": t[keep], "u": u[keep], "steps": sorted(events),
            "antenna_epochs_dropped": int((~keep).sum()), "antenna_heights_dropped_m": dropped}


def midas_rate(t: np.ndarray, x: np.ndarray, steps=()) -> dict | None:
    """
    MIDAS trend (Blewitt et al. 2016): each sample paired with the one
    closest to one year later (within PAIR_TOLERANCE_YEARS), no pair across
    a step epoch in `steps` (decimal years; a pair touching the step's day
    is left out too); the median of the pairs' slopes, slopes beyond 2 sigma
    (1.4826 x MAD) of it trimmed and the median re-taken. Robust to annual
    cycles (they cancel over a year) and to steps. Uncertainty as MIDAS:
    3 x 1.2533 x sd(trimmed) / sqrt(n/4) - about a quarter of the pairs are
    independent, and the factor 3 makes the error realistic.
    """
    t, x = np.asarray(t, dtype=float), np.asarray(x, dtype=float)
    if t.size < 60 or t[-1] - t[0] < 1.5:
        return None
    i = np.arange(t.size)
    j = np.searchsorted(t, t + 1.0)
    lo, hi = np.clip(j - 1, 0, t.size - 1), np.clip(j, 0, t.size - 1)
    g_lo, g_hi = np.abs(t[lo] - t - 1.0), np.abs(t[hi] - t - 1.0)
    j = np.where(g_lo <= g_hi, lo, hi)
    ok = np.minimum(g_lo, g_hi) <= PAIR_TOLERANCE_YEARS
    half_day = 0.5 / 365.25
    for s in steps:
        ok &= ~((t < s + half_day) & (t[j] > s - half_day))
    i, j = i[ok], j[ok]
    if i.size < 30:
        return None
    s = (x[j] - x[i]) / (t[j] - t[i])
    med = float(np.median(s))
    sd = 1.4826 * float(np.median(np.abs(s - med)))
    # a noise-free series (sd ~ 0, rounding only) is not trimmed: there is no tail
    kept = s[np.abs(s - med) <= MIDAS_TRIM_SIGMA * sd] if sd > 1e-6 else s
    if kept.size < 30:
        kept = s
    med = float(np.median(kept))
    sd = 1.4826 * float(np.median(np.abs(kept - med)))
    n_eff = max(1.0, kept.size / 4.0)                     # MIDAS: ~1/4 of pairs independent
    return {"rate_mm_yr": med, "sigma_mm_yr": MIDAS_SIGMA_SCALE * 1.2533 * sd / math.sqrt(n_eff),
            "n_pairs": int(kept.size), "n_trimmed": int(s.size - kept.size), "span_years": float(t[-1] - t[0])}


def _decyear(d: str) -> float:
    dt = datetime.strptime(d, "%Y%m%d")
    y0 = datetime(dt.year, 1, 1)
    return dt.year + (dt - y0).days / ((datetime(dt.year + 1, 1, 1) - y0).days)


def _haversine_km(lat1, lon1, lat2, lon2) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 6371.0 * 2 * math.asin(math.sqrt(a))


def _centre(cfg) -> tuple[float, float]:
    return (cfg.bbox[1] + cfg.bbox[3]) / 2, (cfg.bbox[0] + cfg.bbox[2]) / 2


def discover(cfg, cache_root: Path, radius: float | None = None) -> list[str]:
    """Stations within the tie radius of the AOI centre (NGL + JPL catalogues)."""
    from .gnss_validation import load_stations, nearby
    clat, clon = _centre(cfg)
    found = nearby(load_stations(cache_root), clat, clon, radius_km(cfg) if radius is None else float(radius))
    return sorted({s["site"] for s in found if s.get("site")})


def _record_dates(cfg) -> list[str] | None:
    """The record's dates (YYYYMMDD) from the time series the products use, or None."""
    try:
        import h5py
        from . import deliverables as dl
        with h5py.File(dl.timeseries_file(cfg), "r") as f:
            return [d.decode() if isinstance(d, bytes) else str(d) for d in f["date"][:]]
    except Exception as exc:                              # noqa: BLE001 - only the window is lost
        log.warning("GNSS tie: record dates not readable (%s)", exc)
        return None


def _tie_path(cfg) -> Path:
    return Path(cfg.project_dir) / "deliverables" / "gnss_tie.json"


def _write(cfg, out: dict) -> dict:
    path = _tie_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    # 'software': the code that made the tie (16 Oct 2026, readiness review
    # OUT-01); release.numbers_hash drops it with 'written' when the tie is
    # copied into client_report.json, so a code change alone moves no hash
    from . import deliverables as dl
    out = dict(out, written=datetime.now().isoformat(timespec="seconds"), software=dl._software())
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(out, indent=1))
    tmp.replace(path)                                     # never a half-written tie
    return out


def _untied(cfg, reason: str, dates=None, **extra) -> dict:
    """Record that this run has NO tie, replacing any earlier file."""
    dates = dates if dates is not None else _record_dates(cfg)
    out = {"background": None, "anchored": False, "reason": reason,
           "record_window": [dates[0], dates[-1]] if dates else None,
           "radius_km": radius_km(cfg), "stations_used": [], **extra}
    log.warning("GNSS tie: none - %s", reason)
    return _write(cfg, out)


def run(cfg) -> dict:
    """
    The pipeline's entry point: discover, fetch, tie. ALWAYS writes
    deliverables/gnss_tie.json - with "background": null and the reason when
    anchoring is off, no station is in range or none has enough data - so a
    file from an earlier run is never left behind to be read as this one's.
    If the tie itself fails, the unanchored file is written first and the
    error is re-raised.
    """
    radius = radius_km(cfg)
    if radius <= 0:
        return _untied(cfg, "GNSS anchoring is switched off for this project (gnss_tie_radius_km is 0), "
                       + UNTIED_SUFFIX)
    cache = Path(cfg.project_dir).parent / "gnss"
    try:
        sites = discover(cfg, cache)
        if not sites:
            return _untied(cfg, f"No continuous GNSS station lies within {radius:g} km of the area, " + UNTIED_SUFFIX)
        return tie(cfg, sites, cache)
    except Exception as exc:
        try:
            _untied(cfg, f"The GNSS tie could not be computed ({type(exc).__name__}: {exc}), " + UNTIED_SUFFIX)
        except Exception as exc2:                         # noqa: BLE001 - keep the original error
            log.error("GNSS tie: could not record the failure either (%s)", exc2)
        raise


def _site_groups(stations: list[dict]) -> list[tuple[str, float, float, list[str]]]:
    """
    (name, rate, sigma, members) per SITE: stations within 1 km of each other
    are averaged (inverse variance) first. INES and INEV share a site: one
    monument's record followed by its replacement, so a site is not counted
    twice. Each station needs "lat", "lon", "rate_mm_yr", "sigma_mm_yr".
    """
    groups: list[list[dict]] = []
    for s in stations:
        for gset in groups:
            if _haversine_km(s["lat"], s["lon"], gset[0]["lat"], gset[0]["lon"]) < 1.0:
                gset.append(s); break
        else:
            groups.append([s])
    out = []
    for gset in groups:
        v = np.array([s["rate_mm_yr"] for s in gset])
        sg_ = np.array([s["sigma_mm_yr"] for s in gset])
        wts = 1 / np.maximum(sg_, 0.1) ** 2
        out.append(("+".join(s["site"] for s in gset), float((v * wts).sum() / wts.sum()),
                    float(1 / math.sqrt(wts.sum())), [s["site"] for s in gset]))
    return out


def combine(stations: list[dict]) -> dict | None:
    """Inverse-variance mean of the sites, its formal error and the scatter between sites."""
    site_rates = _site_groups(stations)
    if not site_rates:
        return None
    v = np.array([x[1] for x in site_rates]); s_ = np.array([x[2] for x in site_rates])
    wts = 1 / np.maximum(s_, 0.1) ** 2
    return {"site_rates": site_rates, "mean": float((v * wts).sum() / wts.sum()),
            "formal": float(1 / math.sqrt(wts.sum())),
            "scatter": float(v.std(ddof=1)) if v.size > 1 else float("nan")}


def _day_index(t: np.ndarray) -> np.ndarray:
    """NGL's decimal years back to whole days since 2000-01-01, to match days across stations."""
    return np.rint((np.asarray(t, dtype=float) - 2000.0) * 365.25).astype(np.int64)


def outlier_test(use: list[dict], series: dict, t0: float, t1: float,
                 min_overlap_years: float = MIN_OVERLAP_YEARS) -> tuple[list[dict], dict]:
    """
    The robust background (E15): which sites may enter the weighted mean.
    Two sites (monument groups) within NEIGHBOUR_KM share the regional field,
    so the MIDAS rate of their DIFFERENCED daily series over their common
    days in the record [t0, t1] measures local motion with the common-mode
    noise cancelled. A pair disagrees when that rate lies beyond z x its
    MIDAS sigma, z set for OUTLIER_FALSE_ALARM familywise over the pairs
    tested (1.96 for one pair). Of a disagreeing pair, the site farther from
    the median of the OTHER sites' rates is left out; with no other site to
    tell which of the two moves locally, both stay (their scatter widens
    the uncertainty). A pair with less than `min_overlap_years` in common is
    not tested.

    `use`: combine() inputs (site, lat, lon, rate_mm_yr, sigma_mm_yr);
    `series`: {station: screen_series(...)}. Returns (the sites kept, the
    record for gnss_tie.json "outlier_test").
    """
    groups = _site_groups(use)
    group_of = {m: name for name, _, _, members in groups for m in members}
    rate = {name: r for name, r, _, _ in groups}
    pairs = []
    for k, a in enumerate(use):
        for b in use[k + 1:]:
            if group_of[a["site"]] == group_of[b["site"]]:
                continue
            dist = _haversine_km(a["lat"], a["lon"], b["lat"], b["lon"])
            if dist > NEIGHBOUR_KM:
                continue
            p = {"sites": [a["site"], b["site"]], "distance_km": round(dist, 1)}
            sa, sb = series.get(a["site"]), series.get(b["site"])
            m, span = None, 0.0
            if sa is not None and sb is not None:
                _, ia, ib = np.intersect1d(_day_index(sa["t"]), _day_index(sb["t"]), return_indices=True)
                tt = sa["t"][ia]
                w = (tt >= t0) & (tt <= t1)
                span = float(tt[w][-1] - tt[w][0]) if w.sum() > 1 else 0.0
                if span >= min_overlap_years:
                    m = midas_rate(tt[w], (sa["u"][ia] - sb["u"][ib])[w], sorted(set(sa["steps"]) | set(sb["steps"])))
            p["common_years"] = round(span, 2)
            if m is None:
                p.update(tested=False, why=f"only {span:.1f} years of common days in the record "
                                           f"({min_overlap_years:g} needed)")
            else:
                p.update(tested=True, difference_mm_yr=m["rate_mm_yr"], sigma_mm_yr=m["sigma_mm_yr"])
            pairs.append(p)
    tested = [p for p in pairs if p["tested"]]
    z = NormalDist().inv_cdf(1 - OUTLIER_FALSE_ALARM / (2 * len(tested))) if tested else None
    left_out: set[str] = set()
    for p in tested:
        lim = z * p["sigma_mm_yr"]
        p["disagree"] = bool(abs(p["difference_mm_yr"]) > lim)
        p["limit_mm_yr"] = round(lim, 2)
        p["difference_mm_yr"], p["sigma_mm_yr"] = round(p["difference_mm_yr"], 2), round(p["sigma_mm_yr"], 2)
        if not p["disagree"]:
            continue
        ga, gb = group_of[p["sites"][0]], group_of[p["sites"][1]]
        others = [r for name, r in rate.items() if name not in (ga, gb)]
        if not others:
            p["left_out"] = None
            p["why"] = "no third site to tell which of the two moves locally: both kept"
            continue
        ref = float(np.median(others))
        out_g = ga if abs(rate[ga] - ref) >= abs(rate[gb] - ref) else gb
        p["left_out"] = out_g
        left_out.add(out_g)
    kept = [u for u in use if group_of[u["site"]] not in left_out]
    if not kept:                                          # never empty the background by this test
        kept, left_out = list(use), set()
    rec = {"rule": (f"sites within {NEIGHBOUR_KM:g} km share the regional field: the MIDAS rate of their differenced "
                    f"daily series over common days must lie within z x its sigma (z for {OUTLIER_FALSE_ALARM:.0%} "
                    "false alarms over the pairs tested); of a pair that does not, the site farther from the median "
                    "of the other sites is left out of the background"),
           "z": round(z, 2) if z is not None else None, "pairs": pairs, "left_out": sorted(left_out)}
    return kept, rec


def _midas_catalogue(cache_root: Path) -> list[dict]:
    """The cached NGL MIDAS catalogue with each solution's time span: SITE, lat,
    lon, t0, t1 (decimal years), vertical rate and sigma (mm/yr). Reads the
    files gnss_validation.load_stations cached; never downloads. [] if absent."""
    llh, midas = cache_root / "unr_llh.out", cache_root / "unr_midas_igs14.txt"
    if not (llh.exists() and midas.exists()):
        return []
    pos = {}
    for line in llh.read_text(errors="replace").splitlines():
        p = line.split()
        if len(p) >= 3:
            try:
                pos[p[0]] = (float(p[1]), ((float(p[2]) + 180) % 360) - 180)
            except ValueError:
                continue
    out = []
    for line in midas.read_text(errors="replace").splitlines():
        p = line.split()
        if len(p) < 14 or p[0] not in pos:
            continue
        try:
            t0, t1, vu, su = float(p[2]), float(p[3]), float(p[10]), float(p[13])
        except ValueError:
            continue
        out.append({"site": p[0], "lat": pos[p[0]][0], "lon": pos[p[0]][1], "t0": t0, "t1": t1,
                    "rate_mm_yr": vu * 1000.0, "sigma_mm_yr": su * 1000.0})
    return out


def regional_scatter(cache_root: Path, lat: float, lon: float, t0: float, t1: float,
                     radius: float = REGIONAL_FLOOR_RADIUS_KM, min_overlap_years: float = MIN_OVERLAP_YEARS) -> dict:
    """
    The floor on the background uncertainty: the scatter (sample sd) of the
    MIDAS vertical rates of the NGL stations within `radius` km whose MIDAS
    span overlaps the record [t0, t1] (decimal years) by at least
    `min_overlap_years`, monuments within 1 km averaged into one site. It is
    what one station's rate says about the ground tens of km away. From the
    cached catalogue only (the IGS14 MIDAS file; its spans end at the file's
    date, so the overlap is a lower bound). None with fewer than 3 sites
    or no cached catalogue - and then tie() does not anchor.
    """
    cat = _midas_catalogue(cache_root)
    sel = [s for s in cat if _haversine_km(lat, lon, s["lat"], s["lon"]) <= radius
           and min(s["t1"], t1) - max(s["t0"], t0) >= min_overlap_years]
    groups = _site_groups(sel)
    rates = np.array([g[1] for g in groups])
    floor = float(rates.std(ddof=1)) if rates.size >= 3 else None
    return {"uncertainty_floor_mm_yr": round(floor, 2) if floor is not None else None,
            "n_sites": int(rates.size), "radius_km": radius, "catalogue_cached": bool(cat),
            "sites": [g[0] for g in groups],
            "basis": (f"sample sd of the MIDAS vertical rates (cached NGL IGS14 catalogue) of the {rates.size} sites "
                      f"within {radius:g} km whose solution overlaps the record by {min_overlap_years:g} yr or more"
                      if cat else "NGL MIDAS catalogue not cached: no regional floor")}


def _station_record(site: str, g: dict, t0: float, t1: float, clat: float, clon: float,
                    steps=(), sc: dict | None = None) -> dict:
    """The station's entry in gnss_tie.json: its MIDAS rates over its whole
    span and over the record [t0, t1], from the screened series `sc`
    (screen_series(g, steps) when not given)."""
    sc = screen_series(g, steps) if sc is None else sc
    t, u = sc["t"], sc["u"]
    full = midas_rate(t, u, sc["steps"])
    w = (t >= t0) & (t <= t1)
    rec = midas_rate(t[w], u[w], sc["steps"]) if w.sum() else None
    return {"site": site, "lat": round(g["lat"], 5), "lon": round(g["lon"], 5),
            "distance_km": round(_haversine_km(clat, clon, g["lat"], g["lon"]), 1),
            "data": [round(float(g["t"][0]), 2), round(float(g["t"][-1]), 2)],
            "vertical_full_span_mm_yr": {k: round(v, 2) for k, v in full.items() if k.endswith("yr")} if full else None,
            "vertical_over_record_mm_yr": ({k: round(v, 2) for k, v in rec.items() if k.endswith("yr") or k == "span_years"}
                                           if rec else None),
            "overlap_years": round(float(t[w][-1] - t[w][0]), 2) if w.sum() > 1 else 0.0,
            "screening": {"steps_not_straddled": [round(s, 3) for s in sc["steps"] if t.size and t[0] <= s <= t[-1]],
                          "antenna_height_days_dropped": sc["antenna_epochs_dropped"],
                          "antenna_heights_dropped_m": sc["antenna_heights_dropped_m"]}}


def _usable(st: dict, min_overlap_years: float) -> dict | None:
    """The station as a combine() input when it may anchor the background."""
    rec = st.get("vertical_over_record_mm_yr")
    if not rec or st.get("overlap_years", 0.0) < min_overlap_years:
        return None
    return {"site": st["site"], "lat": st["lat"], "lon": st["lon"],
            "rate_mm_yr": rec["rate_mm_yr"], "sigma_mm_yr": rec["sigma_mm_yr"]}


def _cached_tenv3(cache_root: Path, site: str) -> Path | None:
    p = cache_root / site / f"{site}.tenv3"
    return p if p.exists() and p.stat().st_size >= 1000 else None


def _radius_sensitivity(cfg, cache_root: Path, stations: list[dict], t0: float, t1: float,
                        min_overlap_years: float, series: dict | None = None,
                        steps: dict | None = None) -> dict:
    """The background at 30/40/50 km (and the run's radius), from stations
    already processed or already downloaded - never a new download - by the
    tie's own rule (screened series, outlier_test, combine). A radius within
    which a catalogue station was never downloaded that would qualify (its
    catalogue span overlaps the record by `min_overlap_years`, or the
    catalogue does not say) is "not evaluated" (vertical_mm_yr None,
    missing_sites): a number from the downloaded stations alone would read
    as the background at that radius when it is not (E15)."""
    from .gnss_validation import load_stations, nearby
    clat, clon = _centre(cfg)
    radii = sorted(set(SENSITIVITY_RADII_KM) | {radius_km(cfg)})
    known = {s["site"]: s for s in stations}
    series = dict(series or {})
    steps = steps or {}
    not_cached = []
    try:
        cand = nearby(load_stations(cache_root), clat, clon, max(radii))
    except Exception as exc:                              # noqa: BLE001 - sensitivity is informational
        return {"note": f"catalogue not readable: {exc}"}
    catalogue = {s["site"]: s for s in _midas_catalogue(cache_root)}
    # The cached IGS14 catalogue stops at its own date (2024.74): a span that
    # runs to that date is a station still recording, whose daily series -
    # what the tie would measure - reaches the record's end. Judged by the
    # catalogue span alone, a station begun in 2022 would "not qualify" and
    # the radius read as evaluated without it.
    cat_end = max((s["t1"] for s in catalogue.values()), default=None)
    for c in cand:
        site = c.get("site")
        if not site or site in known:
            continue
        p = _cached_tenv3(cache_root, site)
        if p is None:
            cat = catalogue.get(site)
            if cat is None:
                qualifies = True                          # the catalogue does not say: it might
            else:
                end = max(cat["t1"], t1) if cat["t1"] >= cat_end - 0.1 else cat["t1"]
                qualifies = min(end, t1) - max(cat["t0"], t0) >= min_overlap_years
            not_cached.append((site, float(c.get("distance_km", 0.0)), qualifies)); continue
        try:
            g = read_tenv3(p)
            sc = screen_series(g, [e["t"] for e in steps.get(site, [])])
            known[site] = _station_record(site, g, t0, t1, clat, clon, sc=sc)
            series[site] = sc
        except Exception as exc:                          # noqa: BLE001
            log.info("GNSS %s (sensitivity only) not readable: %s", site, exc)
    out = {}
    for r in radii:
        use = [u for s in known.values() if s["distance_km"] <= r for u in [_usable(s, min_overlap_years)] if u]
        missing = sorted(s for s, d, q in not_cached if d <= r and q)
        kept, rec = outlier_test(use, series, t0, t1, min_overlap_years) if use else ([], {"left_out": []})
        c = combine(kept)
        e = {"vertical_mm_yr": round(c["mean"], 2) if c else None, "sites": [x[0] for x in c["site_rates"]] if c else []}
        if rec["left_out"]:
            e["left_out"] = rec["left_out"]
        if missing:
            e.update(vertical_mm_yr=None, evaluated=False, status="not evaluated", missing_sites=missing)
        else:
            e["evaluated"] = True
        out[f"{r:g}"] = e
    out["not_cached"] = sorted(s for s, _, _ in not_cached)
    return out


def _los_up(cfg, shape) -> tuple:
    """(LOS up-component map or None, where it came from). The background is
    vertical; only its LOS copy needs this, so a missing geometry is recorded
    and the nominal 38.5-degree incidence used rather than failing the tie."""
    from . import deliverables as dl
    fn = getattr(dl, "los_up_or_raise", None)
    try:
        m = fn(cfg, shape) if fn is not None else dl.geometry_field(cfg, "los_up")
    except Exception as exc:                              # noqa: BLE001 - recorded below
        log.warning("GNSS tie: no LOS geometry (%s); LOS copy of the background assumes 38.5 deg incidence", exc)
        return None, f"assumed 38.5 deg incidence (no geometry: {exc})"
    if m is None:
        return None, "assumed 38.5 deg incidence (no geometry layer)"
    return m, "product geometry"


def _load_steps(cache_root: Path, steps_file: Path | None) -> tuple[dict, dict]:
    """({site: [step events]}, the record of where they came from). No steps
    file is not fatal - MIDAS's trimming limits what an unscreened step can
    do - but it is said."""
    p = Path(steps_file) if steps_file is not None else Path(cache_root) / STEPS_FILE
    try:
        steps = read_steps(p)
        if not steps:                                     # NGL's list names thousands of sites
            raise ValueError("it names no equipment change or earthquake")
        mod = datetime.fromtimestamp(p.stat().st_mtime).isoformat(timespec="seconds")
        return steps, {"file": str(p), "modified": mod, "screened": True}
    except (OSError, ValueError) as exc:
        log.warning("GNSS tie: NGL steps file not readable (%s); equipment changes not screened", exc)
        return {}, {"file": str(p), "modified": None, "screened": False,
                    "note": "NGL steps file not available: equipment changes and earthquakes were not screened "
                            "(antenna-height changes in the series were)"}


def tie(cfg, sites: list[str], cache_root: Path, min_overlap_years: float = MIN_OVERLAP_YEARS,
        steps_file: Path | None = None) -> dict:
    """
    The regional background over the record's dates and the check at every
    station inside the processed area. Writes deliverables/gnss_tie.json -
    with "background": null and the reason when no station has enough data.
    `steps_file`: NGL's steps file (default: the cached <cache_root>/ngl_steps.txt).
    """
    from . import deliverables as dl
    from . import significance as sg
    from .quality_assessment import _read2d, report_farmland_masks

    vel, attrs = _read2d(dl._product(cfg, "velocity.h5"), "velocity"); vel = vel * 1000.0
    tc, _ = _read2d(dl._product(cfg, "temporalCoherence.h5"), "temporalCoherence")
    good = dl.load_mask(cfg) & np.isfinite(vel)
    # the inside-station check's farmland leaves out the report areas other
    # than the whole study area (+ 500 m), as the report's comparison ground
    # does (E28; F7(d), 7 Oct 2026)
    far, _, _, fmeta = report_farmland_masks(cfg, good, attrs, wide=False)
    ground = fmeta["comparison_ground"]
    fmeta = None
    rel, _ = sg.class_matched_field(vel, tc, far)
    los_up_map, los_src = _los_up(cfg, vel.shape)
    import h5py
    with h5py.File(dl.timeseries_file(cfg), "r") as f:
        dates = [d.decode() if isinstance(d, bytes) else str(d) for d in f["date"][:]]
    t0, t1 = _decyear(dates[0]), _decyear(dates[-1])
    los_up = float(np.nanmedian(los_up_map[good])) if los_up_map is not None else NOMINAL_LOS_UP
    clat, clon = _centre(cfg)
    ny, nx = vel.shape
    stations, inside, failed, series = [], [], [], {}
    steps = None
    for site in sites:
        try:
            fp = fetch_tenv3(site, cache_root)
            g = read_tenv3(fp)
        except Exception as exc:                              # noqa: BLE001
            log.warning("GNSS %s not available: %s", site, exc); failed.append(site); continue
        if steps is None:                                     # after the first fetch, which refreshes the file
            steps, steps_rec = _load_steps(cache_root, steps_file)
        sc = screen_series(g, [e["t"] for e in steps.get(site, [])])
        series[site] = sc
        st = _station_record(site, g, t0, t1, clat, clon, sc=sc)
        # an empty steps_not_straddled means "no step" only when the list was read
        st["screening"]["steps_file_read"] = bool(steps_rec.get("screened"))
        # The cache is fetched once and never refreshed, so record WHEN this
        # station's series was downloaded: a tie made from a two-year-old file
        # ends where that file ends, whatever the record window says.
        st["data_file"] = str(fp)
        try:
            st["data_file_modified"] = datetime.fromtimestamp(Path(fp).stat().st_mtime).isoformat(timespec="seconds")
        except (OSError, TypeError, ValueError):
            st["data_file_modified"] = None
        r, c = sg.lalo_to_rc(attrs, g["lat"], g["lon"])
        is_in = bool(0 <= r < ny and 0 <= c < nx and sg.aoi_mask(cfg.bbox, attrs, good.shape)[r, c])
        st["inside_processed_area"] = is_in
        stations.append(st)
        if is_in:
            inside.append((site, sc, r, c, g["lat"], g["lon"]))
    if steps is None:
        steps, steps_rec = _load_steps(cache_root, steps_file)
    use = [u for s in stations for u in [_usable(s, min_overlap_years)] if u]
    kept, outliers = outlier_test(use, series, t0, t1, min_overlap_years)
    kept_sites = {u["site"] for u in kept}
    for st in stations:
        st["used_for_background"] = st["site"] in kept_sites
        if any(st["site"] == u["site"] for u in use) and st["site"] not in kept_sites:
            st["left_out"] = ("disagrees with a neighbour within "
                              f"{NEIGHBOUR_KM:g} km beyond the uncertainty of their difference (outlier_test)")
    inside_checks = []
    for site, sc, r, c, lat, lon in inside:
        win = np.zeros_like(good); win[max(0, r - 3):r + 4, max(0, c - 3):c + 4] = True
        win &= good & np.isfinite(rel)
        # the radar's WHOLE-RECORD rate at the station's window against
        # farmland: a diagnostic only - the station recorded only part of the
        # record, so it is never set against the station (E16)
        a = sg.assess(rel, far, win) if win.sum() >= 5 else {}
        chk = {"site": site, "row": r, "col": c, "cells": int(win.sum()),
               "median_temporal_coherence": round(float(np.nanmedian(tc[win])), 2) if win.any() else None,
               "full_record": {
                   "insar_relative_class_matched_mm_yr": round(float(rel[win].mean()), 2) if win.any() else None,
                   "farmland_spread_1sigma_mm_yr": a.get("farmland_spread_1sigma_mm_yr"),
                   "threshold_95_mm_yr": a.get("threshold_95_mm_yr"),
                   "excess_mm_yr": a.get("excess_mm_yr"),
                   "distinguishable_from_farmland": a.get("distinguishable_from_farmland"),
                   "note": ("the radar's whole-record rate at the station's 7x7 window against farmland (LOS mm/yr); "
                            "NOT compared with the station, which recorded only part of the record")}}
        # the station minus the OTHER background sites: never minus a monument
        # at its own site (within 1 km, as _site_groups), which would cancel
        # the station's own movement and read as agreement - also when the
        # station itself is too short to be in the background
        own = {m for name, _, _, members in _site_groups(kept) if site in members for m in members}
        regional = [dict(series[u["site"]], site=u["site"]) for u in kept
                    if u["site"] not in own and u["site"] != site and u["site"] in series
                    and _haversine_km(lat, lon, u["lat"], u["lon"]) >= 1.0]
        chk.update(_series_check(cfg, dates, sc, r, c, los_up_map, good, tc=tc, far=far, regional=regional))
        chk["comparison_ground"] = ground
        if "verdict" not in chk:
            chk.update(verdict="not testable", consistent=None,
                       verdict_reason="Not testable: " + str(chk.get("series_check") or "no comparison was possible") + ".")
        inside_checks.append(chk)
    comb = combine(kept)
    floor = regional_scatter(cache_root, clat, clon, t0, t1, min_overlap_years=min_overlap_years)
    out = {"record_window": [dates[0], dates[-1]], "radius_km": radius_km(cfg),
           "stations": stations, "inside_checks": inside_checks,
           "stations_used": [s["site"] for s in kept],
           "uncertainty_floor_mm_yr": floor["uncertainty_floor_mm_yr"], "regional_scatter": floor,
           "los_up_median": round(los_up, 4), "los_up_source": los_src,
           "source": "Nevada Geodetic Laboratory daily solutions, IGS20; MIDAS rates (Blewitt et al. 2016)",
           "estimator": ("MIDAS (7 Oct 2026): 1-year pairs closest to one year (within 0.01 yr), none across an NGL "
                         "step or an antenna-height change, 2-sigma trim, sigma x 3; antenna heights held under 365 "
                         "days dropped"),
           "steps": steps_rec, "steps_file_read": bool(steps_rec.get("screened")), "outlier_test": outliers}
    if failed:
        out["not_downloaded"] = failed
    out["radius_sensitivity"] = _radius_sensitivity(cfg, cache_root, stations, t0, t1, min_overlap_years,
                                                    series=series, steps=steps)
    if comb is None:
        why = (f"None of the {len(sites)} GNSS station(s) within {radius_km(cfg):g} km has {min_overlap_years:g} or "
               f"more years of data during the record" if stations else
               f"The {len(sites)} GNSS station(s) within {radius_km(cfg):g} km could not be downloaded")
        out.update({"background": None, "anchored": False, "reason": why + ", " + UNTIED_SUFFIX, "leave_one_out": None})
        out = _write(cfg, out)
        log.warning("GNSS tie: none - %s", out["reason"])
        return out
    mean, formal, scatter = comb["mean"], comb["formal"], comb["scatter"]
    fl = floor["uncertainty_floor_mm_yr"]
    if fl is None:
        # No regional floor, no tie (the 6 Oct rule, stale_reason): without it
        # a single-station tie reads +/- its formal error (Site D's INLN
        # 0.29 mm/yr against a 0.91 floor), the too-narrow range that made
        # Site A's whole area read "sinking". Each station's own rate
        # stays in "stations".
        why = ("The regional list of GNSS stations could not be downloaded; it is" if not floor.get("catalogue_cached")
               else f"Fewer than 3 GNSS stations within {REGIONAL_FLOOR_RADIUS_KM:g} km have {min_overlap_years:g} "
                    f"or more years of data during the record; at least 3 are")
        out.update({"background": None, "anchored": False, "leave_one_out": None,
                    "reason": why + " needed to measure how much the ground's movement differs across the region, "
                              + UNTIED_SUFFIX})
        out = _write(cfg, out)
        log.warning("GNSS tie: none - %s", out["reason"])
        return out
    parts = {"formal error": formal, "scatter between the sites used": scatter if np.isfinite(scatter) else 0.0,
             "regional floor": fl}
    basis = max(parts, key=parts.get)
    unc = parts[basis]
    n_sites = len(comb["site_rates"])
    loo = None
    if n_sites >= 2:
        loo = {}
        for name, _, _, members in comb["site_rates"]:
            c2 = combine([u for u in kept if u["site"] not in members])
            loo[name] = round(c2["mean"], 2)
        loo = {"without": loo, "max_change_mm_yr": round(max(abs(v - mean) for v in loo.values()), 2)}
    out["leave_one_out"] = loo
    out["anchored"], out["reason"] = True, None
    out["background"] = {
        "sites": [{"site": n, "vertical_mm_yr": round(r, 2), "sigma_mm_yr": round(s, 2)} for n, r, s, _ in comb["site_rates"]],
        "n_sites": n_sites,
        "vertical_mm_yr": round(mean, 2), "vertical_uncertainty_mm_yr": round(unc, 2),
        "los_mm_yr": round(mean * los_up, 2), "los_uncertainty_mm_yr": round(unc * los_up, 2),
        "formal_mm_yr": round(formal, 2), "between_site_scatter_mm_yr": round(scatter, 2) if np.isfinite(scatter) else None,
        "uncertainty_basis": basis,
        "left_out": outliers["left_out"],
        "note": ("inverse-variance mean of the sites' vertical MIDAS rates over the record's dates, after the "
                 "neighbour outlier test (outlier_test); uncertainty = the largest of the formal error, the scatter "
                 "between the sites used and the regional floor (scatter "
                 f"of the NGL stations within {REGIONAL_FLOOR_RADIUS_KM:g} km). Absolute rate of a region = its "
                 "class-matched relative rate + this background (LOS), assuming the surrounding farmland moves "
                 "with the regional GNSS field - tested, over the years they share, at stations inside the "
                 "processed area (inside_checks)."),
    }
    out = _write(cfg, out)
    log.info("GNSS tie: background %.2f +/- %.2f mm/yr vertical (%s) from %s -> %s",
             mean, unc, basis, [x[0] for x in comb["site_rates"]], _tie_path(cfg))
    return out


# Fields every anchored tie has carried since 5 Oct 2026. A tie without them
# was computed before the regional floor existed: its uncertainty is the
# formal / between-site figure only (Site A's 28 Sep tie: +/- 0.73 mm/yr
# where the floored tie gives +/- 1.01), and that too-narrow band was enough
# to make Site A's whole area read "sinking" (range -3.8 to -0.1 mm/yr;
# with +/- 1.01 the range crosses zero). Such a file is stale, not used -
# and so is one that carries the field as null (no floor could be measured:
# fewer than 3 stations within 100 km or no catalogue), which tie() no
# longer anchors.
REQUIRED_TIE_FIELDS = ("uncertainty_floor_mm_yr",)          # top level
REQUIRED_BACKGROUND_FIELDS = ("uncertainty_basis",)          # inside "background"


def stale_reason(tie_: dict | None) -> str | None:
    """A plain sentence when an ANCHORED tie predates the regional
    uncertainty floor (no uncertainty_floor_mm_yr / uncertainty_basis) or
    was anchored without one (the field null), else None. An unanchored
    record ("background": null) needs neither field."""
    if not isinstance(tie_, dict):
        return None
    bg = tie_.get("background")
    if not isinstance(bg, dict) or bg.get("vertical_mm_yr") is None:
        return None
    missing = [k for k in REQUIRED_TIE_FIELDS if tie_.get(k) is None] + \
              [k for k in REQUIRED_BACKGROUND_FIELDS if bg.get(k) is None]
    if not missing:
        return None
    log.info("GNSS tie has no regional floor (no %s)", ", ".join(missing))
    if all(k in tie_ for k in REQUIRED_TIE_FIELDS) and all(k in bg for k in REQUIRED_BACKGROUND_FIELDS):
        return ("the GNSS tie on disk was anchored without measuring how much GNSS stations across the region "
                "differ, so its range is too narrow to use; re-run the GNSS tie")
    return ("the GNSS tie on disk was made before its uncertainty allowed for how much GNSS stations across the "
            "region differ, so its range is too narrow to use; re-run the GNSS tie")


def unscreened_reason(tie_: dict | None) -> str | None:
    """
    A plain sentence for the reviewer when the stations behind what a report
    takes from this tie - the anchored background, or an inside-station check
    that reached "agree" or "differ" - were NOT screened for equipment
    changes and earthquakes (decision F7(b), 7 Oct 2026; release.py lists it
    as a soft item): NGL's steps file could not be read when the tie was
    made (steps_file_read False; steps.screened False in a tie written
    before that field), or the tie predates the screening (no steps record
    at all). None when they were screened, or nothing rests on them.
    """
    if not isinstance(tie_, dict):
        return None
    bg = tie_.get("background")
    uses = []
    if isinstance(bg, dict) and bg.get("vertical_mm_yr") is not None:
        uses.append("the regional movement the figures are anchored to")
    if any(isinstance(c, dict) and c.get("consistent") is not None for c in tie_.get("inside_checks") or []):
        uses.append("the check at the GNSS station inside the area")
    if not uses:
        return None
    read = tie_.get("steps_file_read")
    if read is None and isinstance(tie_.get("steps"), dict):
        read = tie_["steps"].get("screened")
    if read is True:
        return None
    what = " and ".join(uses)
    if read is False:
        return ("The GNSS stations' records were not checked for equipment changes and earthquakes: the Nevada "
                "Geodetic Laboratory's list of them could not be read when the GNSS tie was made. A jump in a "
                "station's record at such a change moves its rate (INVI's 2024 antenna change moved it by about "
                f"0.4 mm/yr), and with it {what}. Re-run the GNSS tie once the list can be downloaded, then rebuild "
                "the report.")
    return ("The GNSS tie was made before the stations' records were checked for equipment changes and earthquakes "
            f"(7 Oct 2026), so a jump in a station's record may have moved {what}. Re-run the GNSS tie, then "
            "rebuild the report.")


def check(cfg, dates) -> tuple[dict | None, str]:
    """(the tie, "") when deliverables/gnss_tie.json belongs to this record,
    else (None, why): missing, unreadable, another record window, older
    than the velocity product (written before the product it would anchor),
    or anchored without the regional uncertainty floor (stale_reason)."""
    path = _tie_path(cfg)
    if not path.exists():
        return None, "no gnss_tie.json (the GNSS tie has not run)"
    try:
        tie_ = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        return None, f"gnss_tie.json is not readable ({exc})"
    old = stale_reason(tie_)
    if old:
        return None, old
    want = [str(dates[0]), str(dates[-1])] if dates is not None and len(dates) else None
    rw = tie_.get("record_window")
    if want is None or rw is None or [str(x) for x in rw] != want:
        return None, f"gnss_tie.json covers {rw}, not this record's {want}"
    try:
        from . import deliverables as dl
        vel = dl._product(cfg, "velocity.h5")
        if vel.exists() and path.stat().st_mtime < vel.stat().st_mtime:
            return None, "gnss_tie.json is older than the velocity product it would anchor"
    except Exception as exc:                              # noqa: BLE001 - no product to compare is not stale
        log.debug("GNSS tie freshness not checked: %s", exc)
    return tie_, ""


def load_valid(cfg, dates) -> dict | None:
    """
    The GNSS tie of THIS record, or None. A file is used only if its
    record_window is [dates[0], dates[-1]], it is not older than the
    velocity product and (when anchored) it carries the regional
    uncertainty floor - a tie left by an earlier run (other dates, another
    radius, anchoring since switched off, the pre-5 Oct uncertainty) is
    never read as this one's. A valid file may still carry
    "background": null with its "reason".
    """
    tie_, why = check(cfg, dates)
    if tie_ is None:
        log.warning("GNSS tie not used: %s", why)
    return tie_


def preview_stations(lat: float, lon: float, d0: str, d1: str, radius: float | None = None,
                     cache_root: Path | None = None, min_overlap_years: float = MIN_OVERLAP_YEARS) -> list[dict]:
    """
    What the tie is EXPECTED to use, for the builder and the dashboard: every
    catalogue station within max(radius, 50) km with its distance, its MIDAS
    span and `will_use` = inside the tie radius AND overlapping d0..d1
    (YYYYMMDD) by at least min_overlap_years. "Expected": the catalogue's spans
    end at the file's date, and the run measures the overlap from the daily
    series. From the cached catalogue only.
    """
    if cache_root is None:
        from . import settings
        cache_root = Path(settings.projects_root()) / "gnss"
    r = default_radius_km() if radius is None else float(radius)
    t0, t1 = _decyear(d0), _decyear(d1)
    out = []
    for s in _midas_catalogue(Path(cache_root)):
        d = _haversine_km(lat, lon, s["lat"], s["lon"])
        if d > max(r, max(SENSITIVITY_RADII_KM)):
            continue
        ov = min(s["t1"], t1) - max(s["t0"], t0)
        out.append({"site": s["site"], "lat": s["lat"], "lon": s["lon"],
                    "distance_km": round(d, 1), "span": [round(s["t0"], 2), round(s["t1"], 2)],
                    "overlap_years": round(max(ov, 0.0), 2),
                    "will_use": bool(d <= r and ov >= min_overlap_years)})
    return sorted(out, key=lambda x: x["distance_km"])


def _sample(g: dict, tt: np.ndarray) -> np.ndarray:
    """The series' mean vertical (mm) within SAR_GNSS_MATCH_DAYS of each
    date in tt (decimal years); NaN where it has no day that close."""
    half = SAR_GNSS_MATCH_DAYS / 365.25
    t, u = np.asarray(g["t"], dtype=float), np.asarray(g["u"], dtype=float)
    lo, hi = np.searchsorted(t, tt - half, side="left"), np.searchsorted(t, tt + half, side="right")
    cs = np.r_[0.0, np.cumsum(u)]
    n = hi - lo
    return np.where(n > 0, (cs[hi] - cs[lo]) / np.maximum(n, 1), np.nan)


def _difference_sigma(g: dict, regional: list[dict], t_a: float, t_b: float) -> float | None:
    """MIDAS sigma (vertical mm/yr) of the station minus the mean of the
    regional sites, day by day over [t_a, t_b] (days all of them have), or
    None when that is too short for MIDAS."""
    days = [_day_index(g["t"])] + [_day_index(s["t"]) for s in regional]
    common = days[0]
    for d in days[1:]:
        common = np.intersect1d(common, d)
    cols = []
    for s, d in zip([g] + list(regional), days):
        cols.append(np.asarray(s["u"], dtype=float)[np.searchsorted(d, common)])
    t = 2000.0 + common / 365.25
    w = (t >= t_a) & (t <= t_b)
    diff = cols[0] - np.mean(cols[1:], axis=0)
    steps = sorted({float(x) for s in [g] + list(regional) for x in s.get("steps", ())})
    m = midas_rate(t[w], diff[w], steps)
    return m["sigma_mm_yr"] if m else None


def _series_check(cfg, dates, g, r, c, los_up_map, good, tc=None, far=None, regional=None) -> dict:
    """
    The check at a station inside the area (E16, 7 Oct 2026), over the SAR
    dates inside the station's record ONLY - never a whole-record radar rate
    against a few years of GNSS.

    Radar side: every cell's LOS rate over those dates, class-matched
    against the farmland `far` over the same dates (coherence `tc`), and the
    station's 7x7 window tested against that same-window field's farmland
    null (significance.assess): `insar_relative_over_overlap_mm_yr` is its
    excess over the null median. GNSS side: the station minus the mean of
    the `regional` sites (background sites with data on at least 80% of the
    dates; dicts with t, u, site and steps) on the same dates, so the
    region's own movement - which a relative radar record cannot see -
    cancels; its MIDAS sigma from the differenced daily series. Their
    difference is tested against the 95% tolerance hypot(radar threshold,
    1.96 x GNSS sigma). A tolerance above INSIDE_CHECK_USEFUL_MM_YR vertical
    (0.25 in/yr) cannot confirm the anchoring: "not testable", consistent
    None - never "agree" - unless the difference exceeds even that
    tolerance, which is found whatever its size ("differ", 5% false
    alarms). Under 1.5 years of daily data common to the station and the
    regional sites there is no MIDAS rate or sigma for their difference:
    "not testable". The shape check correlates the detrended radar series
    with the detrended station-minus-region series.

    Returns {"series_check": why} alone when nothing could be compared
    (fewer than 20 dates, no radar cell at the station, no geometry);
    otherwise the numbers, `verdict` ("agree" | "differ" | "not testable"),
    `consistent` (True | False | None) and a plain `verdict_reason`. All
    rates mm/yr, LOS unless the key says vertical. Called without tc / far
    (the old signature) the farmland is every good cell outside the window,
    one coherence class.
    """
    import h5py
    from . import deliverables as dl
    from . import significance as sg
    t_sar = np.array([_decyear(d) for d in dates])
    ok = (t_sar >= g["t"][0] + 6 / 365) & (t_sar <= g["t"][-1] - 6 / 365)
    if ok.sum() < 20:
        return {"series_check": "fewer than 20 radar images were taken while the station was recording"}
    los_up = float(los_up_map[r, c]) if los_up_map is not None else NOMINAL_LOS_UP
    win = np.zeros(good.shape, dtype=bool)
    win[max(0, r - 3):r + 4, max(0, c - 3):c + 4] = True
    win &= good
    # a station in trees, by water or on ground below the coherence mask has
    # no radar cell to compare: say so rather than fail the whole tie
    if not win.any() or not np.isfinite(los_up):
        return {"series_check": "no usable radar measurements at the station over the years it recorded"}
    both_msg = "fewer than 20 radar images have both a usable radar measurement and GNSS data at the station"
    idx = np.flatnonzero(ok)
    tt = t_sar[idx]
    gl = _sample(g, tt)
    if np.isfinite(gl).sum() < 20:
        return {"series_check": both_msg}
    tc = np.ones(good.shape) if tc is None else tc
    far = (good & ~win) if far is None else (far & good & ~win)
    k_cls = sg.coherence_class(tc)
    wcls, wn = np.unique(k_cls[win], return_counts=True)
    farc = [far & (k_cls == kc) for kc in wcls]
    farc = [m if m.any() else far for m in farc]          # a class the farmland lacks: all of it
    tm = float(tt.mean())
    acc = np.zeros(good.shape)
    n_fin = np.zeros(good.shape, dtype=np.int32)
    ins = np.full(idx.size, np.nan)       # window minus class-matched farmland, per date (LOS mm)
    pos = {int(i): q for q, i in enumerate(idx)}
    with h5py.File(dl.timeseries_file(cfg), "r") as f:
        ds = f["timeseries"]
        b = max(1, int(ds.chunks[0])) if ds.chunks else 1
        for i0 in range((int(idx[0]) // b) * b, int(idx[-1]) + 1, b):     # chunk-aligned blocks
            blk = ds[i0:min(i0 + b, int(idx[-1]) + 1)]
            for jj in range(blk.shape[0]):
                q = pos.get(i0 + jj)
                if q is None:
                    continue
                x = blk[jj].astype(np.float64) * 1000.0
                fin = np.isfinite(x)
                acc[fin] += (tt[q] - tm) * x[fin]
                n_fin += fin
                wv = x[win & fin]
                cen = [x[m & fin].mean() if (m & fin).any() else np.nan for m in farc]
                if wv.size and np.all(np.isfinite(cen)):
                    ins[q] = wv.mean() - float(np.dot(wn, cen) / wn.sum())
    have = np.isfinite(gl) & np.isfinite(ins)
    if have.sum() < 20:
        return {"series_check": both_msg}
    # the radar's rate over the overlap dates, cell by cell, against farmland over the same dates
    v_w = np.where(n_fin == idx.size, acc / float(((tt - tm) ** 2).sum()), np.nan)
    del acc
    fw = np.isfinite(v_w)
    rel_w, _ = sg.class_matched_field(v_w, tc, far & fw)
    a = sg.assess(rel_w, far & fw, win & fw)
    th = tt[have]
    out = {"overlap_sar_dates": int(have.sum()), "overlap": [round(float(th[0]), 2), round(float(th[-1]), 2)],
           "overlap_years": round(float(th[-1] - th[0]), 2),
           "gnss_los_rate_over_overlap_mm_yr": round(float(np.polyfit(th, gl[have], 1)[0] * los_up), 2),
           "insar_relative_over_overlap_mm_yr": a.get("excess_mm_yr"),
           "insar_relative_over_overlap_vertical_mm_yr": (round(a["excess_mm_yr"] / los_up, 2)
                                                          if a.get("excess_mm_yr") is not None else None),
           "insar_null_over_overlap": {k: a.get(k) for k in ("threshold_95_mm_yr", "farmland_spread_1sigma_mm_yr",
                                                             "null_median_mm_yr", "approx_independent_placements",
                                                             "method")},
           "useful_tolerance_vertical_mm_yr": round(INSIDE_CHECK_USEFUL_MM_YR, 2)}
    y0, y1 = float(th[0]), float(th[-1])
    span = f"over the {y1 - y0:.1f} years the station and the radar record share ({y0:.1f}–{y1:.1f})"

    def _untestable(why: str) -> dict:
        out.update(verdict="not testable", consistent=None, verdict_reason="Not testable: " + why + ".")
        return out

    reg_sites, R = [], []
    for s in regional or []:
        v = _sample(s, tt)
        if np.isfinite(v[have]).mean() >= 0.8:
            reg_sites.append(s); R.append(v)
    out["regional_sites_over_overlap"] = [s.get("site") for s in reg_sites]
    if not a.get("testable"):
        return _untestable(str(a.get("not_testable_reason") or "the radar's farmland comparison is not available")
                           .rstrip("."))
    if not reg_sites:
        return _untestable(f"none of the regional GNSS stations recorded {span}, so the station's own movement "
                           "cannot be told apart from the region's")
    R = np.vstack(R)
    sel = have & np.all(np.isfinite(R), axis=0)
    if sel.sum() < 20:
        return _untestable(f"fewer than 20 radar images {span} have GNSS data at the station and at every "
                           "regional station")
    ts_ = tt[sel]
    reg = (R[:, sel] - R[:, sel].mean(axis=1, keepdims=True)).mean(axis=0)
    d_v = gl[sel] - reg                                    # station minus region, vertical mm
    rate_v = float(np.polyfit(ts_, d_v, 1)[0])
    sig_v = _difference_sigma(g, reg_sites, y0, y1)
    out["station_minus_regional_vertical_mm_yr"] = round(rate_v, 2)
    if sig_v is None:
        # under 1.5 years of common daily data: no GNSS rate free of the
        # seasonal cycle, and no error for it - a tolerance of the radar's
        # threshold alone would call GNSS noise a difference
        return _untestable(f"{span}, the station and the regional stations share too little daily data for a "
                           "reliable GNSS rate (at least 1.5 years are needed)")
    thr = float(a["threshold_95_mm_yr"])
    tol = math.hypot(thr, Z95 * sig_v * los_up)
    x_sat = float(a["excess_mm_yr"])
    diff = x_sat - rate_v * los_up
    tol_v, xs_v = tol / los_up, x_sat / los_up
    ra = (d_v - d_v.mean()) * los_up
    rb = ins[sel] - ins[sel].mean()
    ra, rb = ra - np.polyval(np.polyfit(ts_, ra, 1), ts_), rb - np.polyval(np.polyfit(ts_, rb, 1), ts_)
    out.update({"station_minus_regional_los_mm_yr": round(rate_v * los_up, 2),
                "station_minus_regional_vertical_mm_yr": round(rate_v, 2),
                "station_minus_regional_sigma_vertical_mm_yr": round(sig_v, 2),
                "difference_vertical_mm_yr": round(diff / los_up, 2),
                "tolerance_95_vertical_mm_yr": round(tol_v, 2),
                "detrended_rms_insar_mm": round(float(rb.std()), 1),
                "detrended_rms_gnss_mm": round(float(ra.std()), 1),
                "detrended_rms_difference_mm": round(float((ra - rb).std()), 1),
                "detrended_correlation_vs_regional": (round(float(np.corrcoef(ra, rb)[0, 1]), 2)
                                                      if ra.std() > 0 and rb.std() > 0 else None),
                "note": ("same dates on both sides: radar = the window's rate over the overlap minus class-matched "
                         "farmland over the overlap (excess over that field's farmland null); GNSS = the station "
                         "minus the regional background sites over the same dates, projected into the LOS")})
    agree = abs(diff) <= tol
    # a coarse check cannot confirm agreement, but a difference beyond even
    # its tolerance is still found (5% false alarms whatever the tolerance)
    if agree and tol_v > INSIDE_CHECK_USEFUL_MM_YR:
        return _untestable(f"{span}, this check could only see a difference of about {tol_v / MM_PER_IN:.2f} in/yr "
                           f"({tol_v:.0f} mm/yr) or more between the station and the radar, too coarse to test the "
                           f"anchoring (a useful check resolves {INSIDE_CHECK_USEFUL_MM_YR / MM_PER_IN:.2f} in/yr)")
    said = (f"Over {y0:.1f}–{y1:.1f} the radar reads the ground at the station {xs_v / MM_PER_IN:+.2f} in/yr against "
            f"ordinary ground around it, and the station moved {rate_v / MM_PER_IN:+.2f} in/yr against the regional "
            "stations")
    out.update(verdict="agree" if agree else "differ", consistent=bool(agree),
               verdict_reason=said + (f"; the two agree within the ±{tol_v / MM_PER_IN:.2f} in/yr this check can "
                                      "resolve." if agree else
                                      f"; they differ by more than this check can explain (±{tol_v / MM_PER_IN:.2f} "
                                      "in/yr)."))
    return out
