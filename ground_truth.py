"""
Ground truth for the validation harness (PLAN phase 1, V2): GNSS daily
series, survey and digitised points, and the station inventory, in one shape
the harness (validation.py, V1) reads.

WHAT IT PROVIDES
----------------
    Point = dict(id, kind, lat, lon, h_m, source, role)
        kind: 'gnss' | 'survey' | 'extensometer' | 'insar_product'
        role: 'reference' | 'calibration' | 'validation' | 'null' (from the
              protocol; None = named by the protocol but not used)

    load_gnss_series(ids, window, cache_dir, frame='IGS20', plate='auto')
        -> {id: DataFrame(index=date; e_mm, n_mm, u_mm, se_mm, sn_mm, su_mm)}
        frame 'IGS20' (default; 'itrf' is the same) as NGL publishes it,
        or 'plate': relative to the stations' tectonic plate (to_plate_frame)
    to_plate_frame(series, plate='auto') -> the same, plate-fixed
    load_points_csv(path) -> [(Point, DataFrame(time, value_mm, sigma_mm, component))]
    load_ga_xlsx(path)    -> {id: DataFrame like load_gnss_series}
    station_rate(df, window, component='u', method='linear'|'midas')
        -> dict(rate_mm_yr, sigma_mm_yr, n, span_yr, ...)
    stations_in_box(bbox, catalogue_dir, min_span_yr) -> [Point]

plus station_displacement and match_epochs (the change between two dates and
the +-3 day GNSS mean at SAR dates), protocol_points (the stations a fixed
protocol names, with their roles) and fetch_ga_xlsx.

WHY NOT THE PIPELINE'S OWN GNSS PATH
------------------------------------
gnss_validation.py reads NGL's frozen MIDAS IGS14 file, whose records end by
2024.74, and gnss_tie.py anchors the record to a regional background, which
is the wrong model where the whole area sinks. The protocols
(review_2026-10-08/protocol_*.yaml) forbid both as truth. This module reads
NGL's live IGS20 daily series and steps file directly and reuses
gnss_tie's readers (read_tenv3, read_steps, screen_series, midas_rate), so
the two paths parse NGL the same way. Nothing in gnss_tie changes.

HOW A GNSS SERIES IS PREPARED (protocol "gnss_screening")
--------------------------------------------------------
1. The series and NGL's steps file are downloaded into `cache_dir` (never into
   a project folder). Each series records the file's sha256, when it was
   fetched and its last epoch.
2. Days processed with an antenna height held for under a year are dropped
   (a metadata slip, gnss_tie.screen_series); other antenna-height changes
   are equipment steps.
3. Equipment steps (NGL code 1, and antenna-height changes) inside the window
   are removed as offsets. Possible earthquake offsets (code 2) are removed
   only when earthquake_steps=True; otherwise they are listed as not removed.
   Each offset is estimated LOCALLY: a line plus a step fitted to the days
   within 60 days either side (widened to 120 or 180 days when fewer than 10
   days lie on one side), never past a neighbouring step or earthquake (one
   left in the series too). The days come from the whole record, not only
   the window, so a step just inside a window edge (P307's antenna change
   11 days into the NISAR window) is still estimated. A local fit does
   not let a step absorb the misfit of a whole-record model, which over a
   non-linear signal (San Joaquin: -290 mm/yr in drought, +40 mm in a wet
   year) would remove real motion (a synthetic drought-to-wet turn: ~55 mm
   taken by a whole-record fit, ~1 mm by the local one). A step with fewer
   than 3 days on one side is not estimated and is listed as such. Floor,
   measured by planting steps that are not there every 7 days in real
   series: about 2 mm rms on stable ground (CA07) and 3-4 mm rms on San
   Joaquin farmland (CAHA, QCAP), whatever the local window (30-90 days) or
   trend shape.
4. Daily outliers beyond 5 robust sigma (1.4826 x MAD of the residuals about
   the screening model) are removed, iterating until nothing changes. A day is
   removed when any of its three components is an outlier: a daily solution is
   one estimate, and the line of sight mixes all three. The screening model:
     'linear'    offset + rate + annual (sin, cos)  (protocol A, stable ground)
     'piecewise' as 'linear' with hinges, e.g. water_year_hinges(window)
                 (protocol B, whose rate model is piecewise per water year)
     'local'     a straight line fitted to the days within +-15 days of each
                 day (an event series such as CA19 over a longwall panel,
                 where no rate model fits)
5. Each day's sigma is the robust sigma of the residuals about that model
   over the window (not divided by sqrt(n): GNSS noise is coloured), or the
   day's own formal sigma if that is larger.
6. Values are displacements in mm, zero at the screening model's value on the
   first day kept. Gaps are never filled or interpolated.

THE FRAME (Decision 5, 8 Oct 2026)
----------------------------------
NGL's series are in IGS20 (aligned to ITRF2020): a station on stable
ground moves by its plate's rigid motion, 15-60 mm/yr horizontally
(Appin about +18 east, +55 north; Indiana about -15 east). frame='plate'
removes that motion: from each day's east and north it subtracts the
plate's velocity at the station (plate_motion.plate_velocity_enu, the
published ITRF2020 plate motion model, Altamimi et al. 2023) times the
time since the series' first day. Up is not touched (a rigid plate has
no vertical motion, and the ITRF says to discard the model's vertical
part). Steps, outliers and sigmas are as above: every screening model
has a straight line in it, so removing one changes none of them, and the
datum (zero on the first day) stays.

One call has one plate (one frame): 'auto' takes it from where the
stations are and, where two plates meet (western North America), from
their velocities (plate_motion.choose_plate). Each series records the
plate, the velocity removed, how the plate was chosen and the station's
horizontal motion left after removal, with a warning when that is over
PLATE_RESIDUAL_WARN_MM_YR (the station does not ride rigidly on its
plate: a boundary zone, or local motion; with under
PLATE_RESIDUAL_MIN_SPAN_YR of record the warning says the annual cycle
may be what sets the rate).

Only series that state they are IGS20 are converted. GA's CEMP workbook
is in GA's own published frame, and its series are refused.

The validation harness reads IGS20 (the default) and double-differences
between stations, which cancels the plate's motion anyway; nothing there
changes.

Units: mm, mm/yr, decimal years only internally. Dates are calendar days (UTC).
"""
from __future__ import annotations

import hashlib
import logging
import math
import time
import urllib.error
import urllib.request
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import gnss_tie

log = logging.getLogger(__name__)

KINDS = ("gnss", "survey", "extensometer", "insar_product")
ROLES = ("reference", "calibration", "validation", "null")
COMPONENTS = ("u", "los", "e", "n")
SERIES_COLUMNS = ("e_mm", "n_mm", "u_mm", "se_mm", "sn_mm", "su_mm")

NGL_TENV3 = gnss_tie.NGL_TENV3
NGL_STEPS = gnss_tie.NGL_STEPS
STEPS_FILE = gnss_tie.STEPS_FILE                  # same cache layout as gnss_tie: <cache>/ngl_steps.txt, <cache>/<SITE>/<SITE>.tenv3
NGL_HOLDINGS = "https://geodesy.unr.edu/NGLStationPages/DataHoldings.txt"
HOLDINGS_FILE = "ngl_DataHoldings.txt"
GA_CEMP_XLSX = "https://d28rz98at9flks.cloudfront.net/144247/144247_01_0.xlsx"
GA_FRAME = ("GA CEMP displacements as published (not IGS20): never difference their east/north "
            "with NGL east/north (they differ by plate motion)")

OUTLIER_SIGMA = 5.0               # protocol gnss_screening: daily outliers beyond 5 robust sigma
LOCAL_HALF_DAYS = 15              # 'local' screening model: a line fitted over +-15 days
STEP_WINDOWS_DAYS = (60, 120, 180)
STEP_MIN_DAYS_GOOD = 10           # days wanted on each side of a step
STEP_MIN_DAYS = 3                 # fewer on either side: the offset is not estimated
MATCH_DAYS = 3                    # protocol epoch_matching: GNSS mean within +-3 days of a SAR date
MIN_RATE_SPAN_YR = 1.0            # under this, compare displacement between matched dates, not rates
MAX_CACHE_AGE_DAYS = 7            # 'auto' refresh of a series that ends before the window does
STEPS_MAX_AGE_DAYS = 1
PLATE_RESIDUAL_WARN_MM_YR = 3.0   # plate-fixed: a station moving faster than this does not ride rigidly on its plate
PLATE_RESIDUAL_MIN_SPAN_YR = 2.5  # ... judged only on this much record (shorter: the annual cycle sets the rate)
FRAMES = ("IGS20", "plate")
_FRAME_ALIASES = {"igs20": "IGS20", "itrf": "IGS20", "itrf2020": "IGS20", "plate": "plate"}
YEAR_DAYS = 365.25
_EPOCH = date(2000, 1, 1)


class SeriesSet(dict):
    """{id: DataFrame} with `.missing` = {id: plain-language reason} for the
    stations that could not be loaded (never silently dropped), and
    `.plate` = the plate choice (plate_motion.choose_plate) when the set
    is plate-fixed, else None."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.missing: dict[str, str] = {}
        self.plate: dict | None = None


# ----------------------------------------------------------------- small helpers
def _to_date(x) -> date:
    if isinstance(x, datetime):
        return x.date()
    if isinstance(x, date):
        return x
    if isinstance(x, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(x).date()
    s = str(x).strip()
    if len(s) == 8 and s.isdigit():
        return datetime.strptime(s, "%Y%m%d").date()
    return date.fromisoformat(s[:10])


def _window(window) -> tuple[date | None, date | None]:
    if window is None:
        return None, None
    d0, d1 = window
    d0 = None if d0 is None else _to_date(d0)
    d1 = None if d1 is None else _to_date(d1)
    if d0 is not None and d1 is not None and d1 < d0:
        raise ValueError(f"window ends ({d1}) before it starts ({d0})")
    return d0, d1


def _ordinal(dates) -> np.ndarray:
    return np.array([d.toordinal() for d in dates], dtype=np.int64)


def _decyear_ord(o) -> np.ndarray:
    """NGL's decimal-year convention (2000 + days since 2000-01-01 / 365.25),
    the one gnss_tie.midas_rate and the steps file use."""
    return 2000.0 + (np.asarray(o, dtype=float) - _EPOCH.toordinal()) / YEAR_DAYS


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _mtime_iso(path: Path) -> str:
    return datetime.fromtimestamp(Path(path).stat().st_mtime, tz=timezone.utc).isoformat(timespec="seconds")


def _age_days(path: Path) -> float:
    return (time.time() - Path(path).stat().st_mtime) / 86400.0


def _download(url: str, dest: Path, timeout: int, check=None) -> Path:
    """Fetch url to dest atomically; `check(tmp_path)` may reject the content."""
    dest = Path(dest)
    with urllib.request.urlopen(url, timeout=timeout) as r:
        data = r.read()
    dest.parent.mkdir(parents=True, exist_ok=True)       # only once there is something to keep
    tmp = dest.with_name(dest.name + ".part")
    tmp.write_bytes(data)
    try:
        if check is not None:
            check(tmp)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    tmp.replace(dest)
    return dest


def _robust_sigma(r: np.ndarray) -> float:
    r = r[np.isfinite(r)]
    if r.size < 3:
        return 0.0
    return 1.4826 * float(np.median(np.abs(r - np.median(r))))


def make_point(id, kind, lat, lon, h_m=None, source="", role="validation", **extra) -> dict:
    """A Point, checked: kind in KINDS, role in ROLES (or None = not used)."""
    if kind not in KINDS:
        raise ValueError(f"point {id}: kind {kind!r} is not one of {KINDS}")
    if role is not None and role not in ROLES:
        raise ValueError(f"point {id}: role {role!r} is not one of {ROLES}")
    p = {"id": str(id), "kind": kind, "lat": float(lat), "lon": float(lon),
         "h_m": None if h_m is None or (isinstance(h_m, float) and math.isnan(h_m)) else float(h_m),
         "source": str(source), "role": role}
    p.update(extra)
    return p


def point_from_series(sid: str, df: pd.DataFrame, role: str = "validation", kind: str = "gnss") -> dict:
    """The Point of a series from load_gnss_series / load_ga_xlsx (role is the
    caller's, from the protocol; by default a station only validates)."""
    a = df.attrs
    if a.get("lat") is None or a.get("lon") is None:
        raise ValueError(f"{sid}: the series has no coordinates (pass coords= to load_ga_xlsx)")
    return make_point(sid, kind, a["lat"], a["lon"], a.get("h"), a.get("source_url", ""), role)


def split_roles(points) -> dict[str, list[str]]:
    """{role: [ids]} - the anchor/validation split, for the harness's record.
    Raises when one id carries two roles."""
    seen: dict[str, str] = {}
    out: dict[str, list[str]] = {r: [] for r in ROLES}
    out["not_used"] = []
    for p in points:
        r = p.get("role")
        if p["id"] in seen and seen[p["id"]] != r:
            raise ValueError(f"{p['id']} has two roles: {seen[p['id']]} and {r}")
        seen[p["id"]] = r
        out[r if r in ROLES else "not_used"].append(p["id"])
    return out


def water_year_hinges(window) -> list[date]:
    """1 October of every water year boundary strictly inside the window (the
    hinges of protocol B's piecewise-linear rate model)."""
    d0, d1 = _window(window)
    if d0 is None or d1 is None:
        raise ValueError("water_year_hinges needs a window with both ends")
    return [date(y, 10, 1) for y in range(d0.year, d1.year + 1) if d0 < date(y, 10, 1) < d1]


# ----------------------------------------------------------------- screening model
def _design(t: np.ndarray, hinges: np.ndarray, annual: bool) -> np.ndarray:
    cols = [np.ones_like(t), t - t[0]]
    cols += [np.maximum(0.0, t - h) for h in hinges]
    if annual:
        w = 2 * math.pi * t
        cols += [np.sin(w), np.cos(w)]
    return np.column_stack(cols)


def _local_line(o: np.ndarray, x: np.ndarray, use: np.ndarray, half: int) -> np.ndarray:
    """At every day, the value of a straight line fitted to the used days
    within +-half days (the mean where fewer than 3 days or one day only).
    A line, not a median: at the edge of a gap during fast motion a median
    lags the motion and would flag real days as outliers."""
    ou, xu = o[use].astype(float), x[use]
    lo = np.searchsorted(ou, o - half, side="left")
    hi = np.searchsorted(ou, o + half, side="right")
    out = np.full(o.size, np.nan)
    for i, (a, b) in enumerate(zip(lo, hi)):
        if b <= a:
            continue
        dx, xx = ou[a:b] - o[i], xu[a:b]
        s0, s1, s2 = float(b - a), dx.sum(), (dx * dx).sum()
        det = s0 * s2 - s1 * s1
        if b - a < 3 or det <= 1e-9 * max(1.0, s0 * s2):
            out[i] = xx.mean()
        else:
            out[i] = (s2 * xx.sum() - s1 * (xx * dx).sum()) / det
    return out


def _model_values(o, X, use, model, hinges_o, local_half):
    """The screening model fitted to the used days, evaluated on every day."""
    if model == "local":
        return np.column_stack([_local_line(o, X[:, k], use, local_half) for k in range(X.shape[1])])
    t = _decyear_ord(o)
    h = _decyear_ord(np.asarray(hinges_o, dtype=float)) if len(hinges_o) else np.array([])
    tu = t[use]
    h = h[(h > tu.min()) & (h < tu.max())] if tu.size else h[:0]
    annual = tu.size > 0 and (tu.max() - tu.min()) >= 1.0
    A = _design(t, h, annual)
    coef, *_ = np.linalg.lstsq(A[use], X[use], rcond=None)
    return A @ coef


def _screen(o, X, model, hinges_o, nsig, local_half, max_iter=10):
    """Outlier days (any component beyond nsig robust sigma of the residuals
    about the model), iterated to a fixed point. Returns (keep, model values,
    robust sigma per component)."""
    keep = np.ones(o.size, dtype=bool)
    M, sig = np.repeat(X[:1], o.size, axis=0), np.zeros(X.shape[1])
    if o.size < 3:                                        # nothing to screen against
        return keep, M, sig
    for _ in range(max_iter):
        if keep.sum() < 3:
            break
        M = _model_values(o, X, keep, model, hinges_o, local_half)
        R = X - M
        sig = np.array([_robust_sigma(R[keep, k]) for k in range(X.shape[1])])
        with np.errstate(invalid="ignore"):
            bad = np.zeros(o.size, dtype=bool)
            for k in range(X.shape[1]):
                if sig[k] > 0:
                    bad |= np.abs(R[:, k]) > nsig * sig[k]
        new = ~bad
        if np.array_equal(new, keep):
            break
        keep = new
    return keep, M, sig


def _check_model(model: str, hinges) -> str:
    if model not in ("linear", "piecewise", "local"):
        raise ValueError(f"screening model {model!r}: use 'linear', 'piecewise' or 'local'")
    if model == "piecewise" and not hinges:
        raise ValueError("model 'piecewise' needs hinges (e.g. water_year_hinges(window))")
    return model


# ----------------------------------------------------------------- steps
def _local_offset(o, X, s, prev_o, next_o):
    """Offset at day s from a line + step fitted to the days within 60 (120,
    180) days either side, never past a neighbouring step; robust (one refit
    without residuals beyond 5 sigma). None when too few days on a side."""
    for w in STEP_WINDOWS_DAYS:
        sel = (o >= max(s - w, prev_o)) & (o < min(s + w, next_o))
        nb, na = int((sel & (o < s)).sum()), int((sel & (o >= s)).sum())
        if nb >= STEP_MIN_DAYS_GOOD and na >= STEP_MIN_DAYS_GOOD:
            break
    if nb < STEP_MIN_DAYS or na < STEP_MIN_DAYS:
        return None, nb, na, w
    oo, XX = o[sel].astype(float), X[sel]
    A = np.column_stack([np.ones_like(oo), (oo - s) / YEAR_DAYS, (oo >= s).astype(float)])
    off, sig = np.zeros(X.shape[1]), np.zeros(X.shape[1])
    for k in range(X.shape[1]):
        use = np.ones(oo.size, dtype=bool)
        for _ in range(2):
            c, *_ = np.linalg.lstsq(A[use], XX[use, k], rcond=None)
            r = XX[:, k] - A @ c
            rs = _robust_sigma(r[use])
            nxt = np.abs(r) <= OUTLIER_SIGMA * rs if rs > 0 else use
            if (nxt & (oo < s)).sum() < STEP_MIN_DAYS or (nxt & (oo >= s)).sum() < STEP_MIN_DAYS:
                break
            use = nxt
        c, *_ = np.linalg.lstsq(A[use], XX[use, k], rcond=None)
        r = XX[use, k] - A[use] @ c
        dof = max(1, int(use.sum()) - 3)
        cov = np.linalg.pinv(A[use].T @ A[use]) * float(r @ r) / dof
        off[k], sig[k] = c[2], math.sqrt(max(cov[2, 2], 0.0))
    return (off, sig), nb, na, w


def _steps_for(site: str, steps_by_site: dict) -> list[dict]:
    """NGL steps for a site as dated events: [{date, code, event}]."""
    out = []
    for s in steps_by_site.get(site, []):
        d = _EPOCH + timedelta(days=int(round((s["t"] - 2000.0) * YEAR_DAYS)))
        code = 2 if s["event"] == "possible earthquake offset" else 1
        out.append({"date": d, "code": code, "event": s["event"]})
    return out


# ----------------------------------------------------------------- downloads
def _check_tenv3(site: str):
    def check(p: Path):
        lines = p.read_text(errors="replace").splitlines()
        if not lines or not lines[0].lower().startswith("site") or len(lines) < 2 \
                or lines[1].split()[0].upper() != site:
            raise ValueError(f"the download is not NGL's {site}.tenv3")
    return check


def _check_steps(p: Path):
    if not gnss_tie.read_steps(p):
        raise ValueError("the download names no equipment change or earthquake (not NGL's steps file)")


def _check_holdings(p: Path):
    if not _read_holdings(p):
        raise ValueError("the download lists no station (not NGL's DataHoldings.txt)")


def _last_epoch_tenv3(p: Path) -> date | None:
    try:
        with open(p, "rb") as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - 4096))
            tail = f.read().decode(errors="replace").splitlines()
        for line in reversed(tail):
            parts = line.split()
            if len(parts) > 2 and not parts[0].lower().startswith("site"):
                return gnss_tie.ngl_date(parts[1])
    except (OSError, KeyError, ValueError):
        return None
    return None


def _fetch_tenv3(site: str, cache: Path, d1: date | None, refresh, timeout: int) -> tuple[Path, str]:
    """(path, note). refresh: True = always download; False = never (cache
    only); 'auto' = when missing, or when the cached copy is more than
    MAX_CACHE_AGE_DAYS old and ends before the window does (or no window end
    was given)."""
    p = cache / site / f"{site}.tenv3"
    have = p.exists() and p.stat().st_size > 200
    want = refresh is True or not have
    if refresh == "auto" and have:
        last = _last_epoch_tenv3(p)
        ends_early = d1 is None or last is None or last < d1
        want = ends_early and _age_days(p) > MAX_CACHE_AGE_DAYS
    if refresh is False and not have:
        raise FileNotFoundError(f"no cached series at {p} and downloads are off (refresh=False)")
    note = "cached copy used"
    if want:
        try:
            _download(NGL_TENV3.format(site=site), p, timeout, _check_tenv3(site))
            note = "downloaded"
        except urllib.error.HTTPError as exc:
            if not have:
                if exc.code == 404:
                    raise FileNotFoundError(f"NGL has no IGS20 daily series for {site} (HTTP 404)") from exc
                raise
            note = f"refresh failed (HTTP {exc.code}); cached copy of {_mtime_iso(p)} used"
        except (OSError, ValueError) as exc:
            if not have:
                raise
            note = f"refresh failed ({exc}); cached copy of {_mtime_iso(p)} used"
    return p, note


def _steps_file(cache: Path, refresh, timeout: int, newest_series: float | None) -> tuple[Path, str]:
    p = cache / STEPS_FILE
    have = p.exists() and p.stat().st_size > 1000
    want = refresh is True or not have
    if refresh == "auto" and have:
        stale = _age_days(p) > STEPS_MAX_AGE_DAYS
        older = newest_series is not None and p.stat().st_mtime < newest_series - 86400
        want = stale and older
    note = "cached copy used"
    if want and refresh is not False:
        try:
            _download(NGL_STEPS, p, timeout, _check_steps)
            note = "downloaded"
        except (OSError, ValueError) as exc:
            if not have:
                raise RuntimeError(
                    f"NGL's steps file could not be downloaded ({exc}) and none is cached in {cache}: "
                    "equipment changes cannot be screened, so no series is loaded") from exc
            note = f"refresh failed ({exc}); cached copy of {_mtime_iso(p)} used"
    if not p.exists():
        raise RuntimeError(f"no NGL steps file in {cache} and downloads are off: equipment changes "
                           "cannot be screened, so no series is loaded")
    return p, note


# ----------------------------------------------------------------- GNSS series
def load_gnss_series(station_ids, window=None, cache_dir=None, frame: str = "IGS20", *,
                     earthquake_steps: bool = False, model="linear", hinges=(),
                     outlier_sigma: float = OUTLIER_SIGMA, local_half_days: int = LOCAL_HALF_DAYS,
                     refresh="auto", timeout: int = 180, plate: str = "auto") -> SeriesSet:
    """
    NGL IGS20 daily east/north/up of each station over `window` (d0, d1),
    prepared as the module docstring says: equipment steps removed as offsets
    (earthquake steps only with earthquake_steps=True), daily outliers beyond
    `outlier_sigma` robust sigma removed, never a gap filled.

    `model` (str, or {id: str}) is the screening model: 'linear' (default),
    'piecewise' with `hinges` (dates; e.g. water_year_hinges(window) for
    protocol B), or 'local' (a line over +-15 days, for event series).

    `frame`: 'IGS20' (default; 'itrf' means the same) as NGL publishes it,
    or 'plate': each series relative to the stations' tectonic plate, one
    plate for the call (`plate`: 'auto' or a code such as NOAM or AUST;
    to_plate_frame, module docstring "THE FRAME"). A plate that cannot be
    chosen raises ValueError rather than return IGS20 under a plate label.

    Returns SeriesSet {id: DataFrame(index=date; e_mm, n_mm, u_mm, se_mm,
    sn_mm, su_mm)} with `.missing` {id: reason}. Each frame's .attrs holds
    lat, lon, h, steps_removed, steps_not_removed, source_url, sha256,
    fetched, last_epoch_in_file, outliers_removed, robust_sigma_mm, model,
    window, frame, frame_code ('IGS20' or 'plate:<CODE>'), datum, and with
    frame='plate' plate_motion (what was removed and why).
    """
    fr = _FRAME_ALIASES.get(str(frame).strip().lower())
    if fr is None:
        raise ValueError(f"frame {frame!r}: only NGL's IGS20 series are read (IGS14 is frozen); frame='plate' "
                         "puts them relative to the stations' tectonic plate")
    if fr == "plate" and str(plate or "auto").strip().lower() != "auto":
        from . import plate_motion
        plate_motion.plate_code(plate)                     # an unknown plate fails before any download
    if cache_dir is None:
        raise ValueError("cache_dir is required (downloads never go into a project folder)")
    if refresh not in (True, False, "auto"):
        raise ValueError("refresh must be True, False or 'auto'")
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    d0, d1 = _window(window)
    ids = [station_ids] if isinstance(station_ids, str) else list(station_ids)
    ids = [str(s).strip().upper() for s in ids]
    hinge_o = [_to_date(h).toordinal() for h in (hinges or ())]
    models = {sid: (model.get(sid, "linear") if isinstance(model, dict) else model) for sid in ids}
    for m in set(models.values()):
        _check_model(m, hinge_o)

    out = SeriesSet()
    files: dict[str, tuple[Path, str]] = {}
    for sid in ids:
        try:
            files[sid] = _fetch_tenv3(sid, cache, d1, refresh, timeout)
        except Exception as exc:                          # noqa: BLE001 - recorded per station
            out.missing[sid] = str(exc) if isinstance(exc, FileNotFoundError) else f"download failed: {exc}"
            log.warning("GNSS %s not loaded: %s", sid, out.missing[sid])
    if not files:
        return out
    newest = max(p.stat().st_mtime for p, _ in files.values())
    steps_path, steps_note = _steps_file(cache, refresh, timeout, newest)
    steps_by_site = gnss_tie.read_steps(steps_path)
    steps_meta = {"file": str(steps_path), "sha256": _sha256(steps_path), "fetched": _mtime_iso(steps_path),
                  "note": steps_note}

    for sid, (path, note) in files.items():
        try:
            df = _prepare(sid, path, note, d0, d1, steps_by_site, steps_meta, earthquake_steps,
                          models[sid], hinge_o, outlier_sigma, local_half_days)
        except (ValueError, IndexError, KeyError) as exc:
            out.missing[sid] = f"series not readable: {exc}" if not isinstance(exc, ValueError) else str(exc)
            log.warning("GNSS %s not loaded: %s", sid, exc)
            continue
        out[sid] = df
    if fr == "plate" and out:
        return to_plate_frame(out, plate=plate)
    return out


def _prepare(sid, path, note, d0, d1, steps_by_site, steps_meta, earthquake_steps,
             model, hinge_o, nsig, local_half) -> pd.DataFrame:
    g = gnss_tie.read_tenv3(path, full=True)
    if g["site"].upper() != sid:
        raise ValueError(f"{path} holds {g['site']}, not {sid}")
    dates_all = g["date"]
    o_all = _ordinal(dates_all)
    file_first, file_last = dates_all[0], dates_all[-1]
    # hinges belong to the 'piecewise' model only (a 'linear' station in a mixed {id: model} stays linear)
    hinge_o = list(hinge_o) if model == "piecewise" else []
    # antenna heights held under a year are metadata slips (dropped); the rest are steps
    scr = gnss_tie.screen_series({"t": g["t"], "u": g["u"], "ant": g["ant"]}, steps=())
    keep_ant = np.isin(g["t"], scr["t"])
    t_to_o = dict(zip(g["t"].tolist(), o_all.tolist()))
    ant_steps = sorted({t_to_o[t] for t in scr["steps"] if t in t_to_o})
    inwin = np.ones(o_all.size, dtype=bool)
    if d0 is not None:
        inwin &= o_all >= d0.toordinal()
    if d1 is not None:
        inwin &= o_all <= d1.toordinal()
    sel = keep_ant & inwin
    if not sel.any():
        raise ValueError(f"no daily solution inside the window (the record runs {file_first} .. {file_last})")
    o = o_all[sel]
    X = np.column_stack([g["e"][sel], g["n"][sel], g["u"][sel]])
    formal = np.column_stack([g["se"][sel], g["sn"][sel], g["su"][sel]])
    slip_days = int((~keep_ant & inwin).sum())
    # the whole record (slips dropped) for the step offsets: a step just inside the window
    # edge is estimated from the days on both sides of it, not only from those in the window
    o_rec = o_all[keep_ant]
    X_rec = np.column_stack([g["e"][keep_ant], g["n"][keep_ant], g["u"][keep_ant]])

    # the steps inside the data's span: equipment (code 1 + antenna-height changes), earthquakes (code 2)
    ev: dict[int, dict] = {}
    for s in _steps_for(sid, steps_by_site):
        so = s["date"].toordinal()
        e = ev.setdefault(so, {"codes": set(), "events": []})
        e["codes"].add(s["code"])
        e["events"].append(s["event"])
    for so in ant_steps:
        e = ev.setdefault(so, {"codes": set(), "events": []})
        e["codes"].add(1)
        e["events"].append("antenna height changed in the series")
    removed, not_removed, use_steps = [], [], []
    for so in sorted(ev):
        if not (o[0] < so <= o[-1]):
            continue
        e = ev[so]
        evs = sorted(set(e["events"]))
        if 1 in e["codes"] or earthquake_steps:
            use_steps.append((so, evs))
        else:
            not_removed.append({"date": date.fromordinal(so).isoformat(), "events": evs,
                                "why": "possible earthquake offset (NGL code 2): removed only with earthquake_steps=True"})
    # every known event bounds a local fit, including one left in the series (an earthquake
    # step without earthquake_steps=True) and those outside the window
    bounds = np.array(sorted(ev), dtype=np.int64)
    offsets = np.zeros_like(X)
    for so, evs in use_steps:
        k = int(np.searchsorted(bounds, so))
        prev_o = int(bounds[k - 1]) if k else -10**9
        next_o = int(bounds[k + 1]) if k + 1 < bounds.size else 10**9
        res, nb, na, w = _local_offset(o_rec, X_rec, so, prev_o, next_o)
        rec = {"date": date.fromordinal(so).isoformat(), "events": evs, "days_before": nb, "days_after": na,
               "fit_window_days": w}
        if res is None:
            rec["why"] = f"offset not estimated: fewer than {STEP_MIN_DAYS} days on one side within {w} days"
            not_removed.append(rec)
            continue
        off, sig = res
        offsets[o >= so] += off
        rec.update({"offset_mm": dict(zip("enu", np.round(off, 2).tolist())),
                    "offset_sigma_mm": dict(zip("enu", np.round(sig, 2).tolist()))})
        removed.append(rec)
    X = X - offsets

    keep, M, sig = _screen(o, X, model, hinge_o, nsig, local_half)
    outlier_days = [date.fromordinal(int(x)).isoformat() for x in o[~keep]]
    o, X, M, formal = o[keep], X[keep], M[keep], formal[keep]
    datum = M[0].copy()
    X = X - datum
    S = np.maximum(formal, sig[None, :])
    df = pd.DataFrame({"e_mm": X[:, 0], "n_mm": X[:, 1], "u_mm": X[:, 2],
                       "se_mm": S[:, 0], "sn_mm": S[:, 1], "su_mm": S[:, 2]},
                      index=pd.DatetimeIndex(pd.to_datetime([date.fromordinal(int(x)) for x in o]), name="date"))
    span = (o[-1] - o[0]) / YEAR_DAYS
    df.attrs.update({
        # NGL writes eastern longitudes as lon - 360 (CA19: -209.31 for 150.69 E)
        "id": sid, "kind": "gnss", "lat": float(g["lat"]), "lon": round(float((g["lon"] + 180.0) % 360.0 - 180.0), 10),
        "h": float(g["h"]),
        "frame": "IGS20 (NGL daily solutions, as published; plate motion included)", "frame_code": "IGS20",
        "source_url": NGL_TENV3.format(site=sid), "file": str(path), "sha256": _sha256(path),
        "fetched": _mtime_iso(path), "fetch_note": note,
        "first_epoch_in_file": file_first.isoformat(), "last_epoch_in_file": file_last.isoformat(),
        "ends_before_window_end": bool(d1 is not None and file_last < d1),
        "window": [None if d0 is None else d0.isoformat(), None if d1 is None else d1.isoformat()],
        "n_days": int(o.size), "span_yr": round(float(span), 3),
        "first_day": date.fromordinal(int(o[0])).isoformat(), "last_day": date.fromordinal(int(o[-1])).isoformat(),
        "steps_file": steps_meta, "steps_removed": removed, "steps_not_removed": not_removed,
        "earthquake_steps_removed": bool(earthquake_steps),
        "antenna_slip_days_dropped": slip_days, "antenna_heights_dropped_m": scr["antenna_heights_dropped_m"],
        "outliers_removed": len(outlier_days), "outlier_days": outlier_days, "outlier_sigma": float(nsig),
        "model": {"name": model, "hinges": [date.fromordinal(int(h)).isoformat() for h in hinge_o],
                  "local_half_days": local_half if model == "local" else None},
        "robust_sigma_mm": dict(zip("enu", np.round(sig, 3).tolist())),
        "sigma_rule": "per day: the larger of the robust sigma of the residuals about the screening model "
                      "over the window and the day's formal sigma (not divided by sqrt(n))",
        "datum": "zero at the screening model's value on the first day kept",
        "gaps_filled": False,
    })
    return df


# ----------------------------------------------------------------- plate-fixed frame (Decision 5)
def series_frame(df: pd.DataFrame) -> str | None:
    """'IGS20', 'plate:<CODE>' or another frame a series states (its
    attrs), or None when it states none."""
    a = getattr(df, "attrs", {}) or {}
    code = a.get("frame_code")
    if code:
        return str(code)
    text = str(a.get("frame") or "")
    if text.startswith("IGS20"):
        return "IGS20"
    return text or None


def horizontal_rate(df: pd.DataFrame, min_span_yr: float = 0.5) -> tuple[float, float] | None:
    """Straight-line east and north rates (mm/yr) of a series over all its
    days, or None with under `min_span_yr` of record or under 10 days."""
    if df is None or len(df) < 10:
        return None
    o = _ordinal(df.index)
    t = (o - o[0]) / YEAR_DAYS
    if np.ptp(t) < min_span_yr:
        return None
    rates = []
    for col in ("e_mm", "n_mm"):
        y = df[col].to_numpy(dtype=float)
        ok = np.isfinite(y)
        if ok.sum() < 10 or np.ptp(t[ok]) < min_span_yr:
            return None
        rates.append(float(np.polyfit(t[ok], y[ok], 1)[0]))
    return rates[0], rates[1]


def to_plate_frame(series, plate: str = "auto", *, model: str = "ITRF2020"):
    """
    IGS20 series -> the same series relative to their tectonic plate (module
    docstring "THE FRAME"). `series`: a SeriesSet or {id: DataFrame} from
    load_gnss_series, or one DataFrame. One plate for all of them
    (plate_motion.choose_plate: `plate` 'auto' or a code). From each day's
    e_mm and n_mm the plate's velocity at the station times the years since
    the series' first day is subtracted; u_mm and the sigmas are unchanged.

    Returns the same shape (a SeriesSet keeps `.missing` and gains `.plate`).
    Each frame's attrs gain frame, frame_code ('plate:<CODE>') and
    plate_motion {plate, plate_name, model, citation, source_url, how,
    removed_mm_yr {e, n}, igs20_horizontal_mm_yr, residual_horizontal_mm_yr
    {e, n, total, span_yr} (straight line over the series, None under
    0.5 yr; over PLATE_RESIDUAL_WARN_MM_YR it is warned, as off the plate
    only with PLATE_RESIDUAL_MIN_SPAN_YR of record),
    model_residual_rms_mm_yr, reference_day, note}. Raises ValueError for a
    series that is not stated IGS20 (GA's CEMP frame, one already
    plate-fixed, or one with no frame) and when no plate can be chosen.
    """
    import copy
    from . import plate_motion as pm
    single = isinstance(series, pd.DataFrame)
    items = {"series": series} if single else dict(series)
    if not items:
        return series
    for sid, df in items.items():
        f = series_frame(df)
        if f != "IGS20":
            raise ValueError(f"{sid}: the series is in {f or 'an unstated frame'}, not IGS20: only NGL's IGS20 "
                             "series can be put in a plate frame with the ITRF2020 plate motion model")
    for sid, df in items.items():
        if "lat" not in df.attrs or "lon" not in df.attrs:
            raise ValueError(f"{sid}: the series carries no station position, so its plate's motion is unknown")
    ids = list(items)
    lats = np.array([float(items[s].attrs["lat"]) for s in ids])
    lons = np.array([float(items[s].attrs["lon"]) for s in ids])
    before = {s: horizontal_rate(items[s], pm.MIN_VELOCITY_SPAN_YR) for s in ids}
    ve = np.array([np.nan if before[s] is None else before[s][0] for s in ids])
    vn = np.array([np.nan if before[s] is None else before[s][1] for s in ids])
    choice = pm.choose_plate(lats, lons, ve, vn, plate=plate, model=model)
    code = choice["plate"]
    mdl = pm._model(model)
    out = {}
    for k, sid in enumerate(ids):
        df = items[sid]
        pe, pn = pm.plate_velocity_enu(code, lats[k], lons[k], float(df.attrs.get("h") or 0.0), model=model)
        o = _ordinal(df.index)
        t = (o - o[0]) / YEAR_DAYS if o.size else np.zeros(0)
        span = float(t[-1]) if t.size else 0.0
        new = df.copy()
        new.attrs = copy.deepcopy(dict(df.attrs))
        new["e_mm"] = df["e_mm"].to_numpy(dtype=float) - pe * t
        new["n_mm"] = df["n_mm"].to_numpy(dtype=float) - pn * t
        after = horizontal_rate(new)
        res = None if after is None else {"e": round(after[0], 2), "n": round(after[1], 2),
                                          "total": round(float(np.hypot(*after)), 2), "span_yr": round(span, 2)}
        note = None
        if res is not None and res["total"] > PLATE_RESIDUAL_WARN_MM_YR:
            if span < PLATE_RESIDUAL_MIN_SPAN_YR:
                # a straight line through under ~2.5 years carries the annual cycle (several mm), so a big
                # rate there is not yet evidence of motion off the plate
                note = (f"{sid} moves {res['total']:.1f} mm/yr horizontally relative to the {choice['name']} plate "
                        f"over a record of only {span:.1f} yr, where seasonal motion can dominate a straight-line "
                        "rate: not yet evidence that it moves off its plate. Only the plate's motion was removed.")
            else:
                note = (f"{sid} moves {res['total']:.1f} mm/yr horizontally relative to the {choice['name']} plate "
                        f"over {span:.1f} yr: it does not ride rigidly on it (a plate-boundary zone, or local "
                        "motion). That motion stays in the series.")
            log.warning(note)
        new.attrs.update({
            "frame": (f"plate-fixed: {choice['name']} plate ({code}), {mdl['name']} rigid motion removed from "
                      "NGL's IGS20 daily solutions"),
            "frame_code": f"plate:{code}",
            "plate_motion": {
                "plate": code, "plate_name": choice["name"], "model": mdl["name"], "citation": mdl["citation"],
                "source_url": mdl["source_url"], "how": choice["how"], "region": choice.get("region"),
                "removed_mm_yr": {"e": round(pe, 3), "n": round(pn, 3)},
                "vertical_removed": "none (a rigid plate has no vertical motion; the model's vertical part is "
                                    "discarded, as the ITRF advises)",
                "igs20_horizontal_mm_yr": None if before[sid] is None else
                {"e": round(before[sid][0], 2), "n": round(before[sid][1], 2)},
                "residual_horizontal_mm_yr": res,
                "model_residual_rms_mm_yr": dict(zip("en", mdl["residual_rms_mm_yr"].get(code, (None, None)))),
                "reference_day": pd.Timestamp(df.index[0]).date().isoformat() if len(df) else None,
                "how_removed": "e_mm and n_mm minus the plate's velocity x years since reference_day; u_mm unchanged",
                "note": note or choice.get("note"),
            },
        })
        out[sid] = new
    if single:
        return out["series"]
    res_set = SeriesSet(out)
    res_set.missing = dict(getattr(series, "missing", {}) or {})
    res_set.plate = choice
    return res_set


# ----------------------------------------------------------------- GA CEMP xlsx
def fetch_ga_xlsx(cache_dir, refresh: bool = False, timeout: int = 180) -> Path:
    """Geoscience Australia's CEMP GNSS workbook (369 kB, public) into cache_dir."""
    p = Path(cache_dir) / "ga" / "144247_01_0.xlsx"
    if refresh or not p.exists():
        _download(GA_CEMP_XLSX, p, timeout)
    return p


def load_ga_xlsx(path, sites=None, *, window=None, coords=None, model="local",
                 outlier_sigma: float = OUTLIER_SIGMA, local_half_days: int = LOCAL_HALF_DAYS) -> SeriesSet:
    """
    Geoscience Australia CEMP daily (or campaign) E/N/U displacements: one
    sheet per site, columns date (yyyymmdd), east/north/up displacement and
    their uncertainties in mm (GA's datum: zero at the site's first epoch).

    Screened like a GNSS series (outliers beyond 5 robust sigma about the
    screening model; default 'local', since CA19 is an event series). No
    step is removed: the workbook carries no equipment record. Each day's sigma is the larger of GA's stated
    sigma and the robust sigma (protocol A: "GA xlsx: its stated sigma if
    larger"). The workbook has no coordinates: pass coords={id: (lat, lon)}
    (e.g. from protocol_points), else lat/lon are None. Its east/north are
    not IGS20 (attrs['frame']).
    """
    import openpyxl
    path = Path(path)
    d0, d1 = _window(window)
    want = None if sites is None else {str(s).upper() for s in ([sites] if isinstance(sites, str) else sites)}
    wb = openpyxl.load_workbook(path, data_only=True, read_only=True)
    sha = _sha256(path)
    out = SeriesSet()
    try:
        names = {n.upper(): n for n in wb.sheetnames}
        for sid in sorted(want or names):
            if sid not in names:
                out.missing[sid] = f"no sheet {sid} in {path.name}"
                continue
            m = model.get(sid, "local") if isinstance(model, dict) else model
            _check_model(m, ())
            rows = list(wb[names[sid]].iter_rows(values_only=True))
            head = [str(h or "").lower() for h in rows[0]] if rows else []
            if len(head) < 7 or "date" not in head[0] or "uncertainty" not in head[4]:
                out.missing[sid] = f"sheet {sid} does not have GA's columns (date, E, N, U, sigma E, N, U)"
                continue
            recs = []
            for r in rows[1:]:
                if r is None or r[0] is None or any(v is None for v in r[1:7]):
                    continue
                d = _to_date(str(int(r[0])))
                if (d0 and d < d0) or (d1 and d > d1):
                    continue
                recs.append((d.toordinal(), *[float(v) for v in r[1:7]]))
            if not recs:
                out.missing[sid] = "no epoch inside the window"
                continue
            a = np.array(sorted(recs))
            o = a[:, 0].astype(np.int64)
            X, stated = a[:, 1:4], a[:, 4:7]
            keep, _, sig = _screen(o, X, m, [], outlier_sigma, local_half_days)
            outl = [date.fromordinal(int(x)).isoformat() for x in o[~keep]]
            o, X, stated = o[keep], X[keep], stated[keep]
            S = np.maximum(stated, sig[None, :])
            df = pd.DataFrame({"e_mm": X[:, 0], "n_mm": X[:, 1], "u_mm": X[:, 2],
                               "se_mm": S[:, 0], "sn_mm": S[:, 1], "su_mm": S[:, 2]},
                              index=pd.DatetimeIndex(pd.to_datetime([date.fromordinal(int(x)) for x in o]),
                                                     name="date"))
            c = (coords or {}).get(sid)
            if isinstance(c, dict):
                c = (c.get("lat"), c.get("lon"), c.get("h_m"))
            df.attrs.update({
                "id": sid, "kind": "gnss", "lat": None if not c else float(c[0]),
                "lon": None if not c else float(c[1]), "h": None if not c or len(c) < 3 or c[2] is None else float(c[2]),
                "frame": GA_FRAME, "source_url": GA_CEMP_XLSX, "file": str(path), "sha256": sha,
                "fetched": _mtime_iso(path), "window": [None if d0 is None else d0.isoformat(),
                                                        None if d1 is None else d1.isoformat()],
                "n_days": int(o.size), "span_yr": round(float((o[-1] - o[0]) / YEAR_DAYS), 3),
                "first_day": date.fromordinal(int(o[0])).isoformat(),
                "last_day": date.fromordinal(int(o[-1])).isoformat(),
                "steps_removed": [], "steps_not_removed": [],
                "steps_note": ("not screened for steps: the workbook has no equipment record; "
                               "GA's displacements are used as published"),
                "outliers_removed": len(outl), "outlier_days": outl, "outlier_sigma": float(outlier_sigma),
                "model": {"name": m, "hinges": [], "local_half_days": local_half_days if m == "local" else None},
                "robust_sigma_mm": dict(zip("enu", np.round(sig, 3).tolist())),
                "stated_sigma_median_mm": dict(zip("enu", np.round(np.median(stated, axis=0), 2).tolist())),
                "sigma_rule": "per epoch: the larger of GA's stated sigma and the robust sigma of the "
                              "residuals about the screening model",
                "datum": "GA's: zero at the site's first epoch in the workbook",
                "gaps_filled": False,
            })
            out[sid] = df
    finally:
        wb.close()
    return out


# ----------------------------------------------------------------- user / digitised points
_ALIASES = {"u": "u", "up": "u", "vertical": "u", "e": "e", "east": "e", "n": "n", "north": "n", "los": "los"}
_WIDE = {"u": ("up_mm", "u_mm", "vertical_mm"), "e": ("east_mm", "e_mm"), "n": ("north_mm", "n_mm"),
         "los": ("los_mm",)}
_SIGMA_WIDE = {"u": ("sigma_up_mm", "sigma_u_mm"), "e": ("sigma_east_mm", "sigma_e_mm"),
               "n": ("sigma_north_mm", "sigma_n_mm"), "los": ("sigma_los_mm",)}


def load_points_csv(path, *, roles: dict | None = None, default_role: str = "validation",
                    default_kind: str = "survey") -> list[tuple[dict, pd.DataFrame]]:
    """
    User-supplied or digitised truth: levelling, survey marks, digitised
    GNSS plots, extensometers. One CSV, either

      wide: id, lat, lon, date, up_mm[, east_mm, north_mm, los_mm], sigma_mm
            (or sigma_up_mm, sigma_east_mm, ...)
      long: id, lat, lon, date, component (u|e|n|los), value_mm, sigma_mm

    with optional kind, role, h_m, source (or source_report and page). A role
    in `roles` ({id: role}) wins over the file's role column, which wins over
    `default_role` ('validation': a point only validates unless the protocol
    says otherwise). Rows without a value are dropped, never filled; a value
    without a sigma (or a sigma not above zero), a row without a date or a
    point without lat/lon is an error. Dates are YYYY-MM-DD or YYYYMMDD
    only: a day-first date such as 05/02/2022 is refused, not guessed.

    Returns [(Point, DataFrame(time, value_mm, sigma_mm, component))], by id.
    """
    path = Path(path)
    # only empty cells are missing: "null" is a role, "NA" could be an id
    raw = pd.read_csv(path, comment="#", dtype={"id": str}, skipinitialspace=True,
                      keep_default_na=False, na_values=["", "nan", "NaN"])
    raw.columns = [str(c).strip().lower() for c in raw.columns]
    tcol = "date" if "date" in raw.columns else "time" if "time" in raw.columns else None
    for need in ("id", "lat", "lon"):
        if need not in raw.columns:
            raise ValueError(f"{path.name}: column '{need}' is missing")
    if tcol is None:
        raise ValueError(f"{path.name}: a 'date' (or 'time') column is needed")
    raw["id"] = raw["id"].astype(str).str.strip()
    # ISO dates only (YYYY-MM-DD or YYYYMMDD): a guessed format would read an Australian
    # 05/02/2022 (5 Feb) as 2 May without a word
    tx = raw[tcol].astype(str).str.strip()
    blank = raw[tcol].isna() | tx.isin(["", "nan", "NaN"])
    if blank.any():
        r = raw[blank].iloc[0]
        raise ValueError(f"{path.name}: {r['id']} has a row without a date")
    try:
        raw["_time"] = pd.to_datetime(tx, format="ISO8601")
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{path.name}: dates must be YYYY-MM-DD (or YYYYMMDD); {exc}".split("\n")[0]) from exc
    for c in ("lat", "lon"):
        bad = ~np.isfinite(pd.to_numeric(raw[c], errors="coerce").to_numpy(dtype=float))
        if bad.any():
            raise ValueError(f"{path.name}: {raw['id'][bad].iloc[0]} has no {c}")

    if "component" in raw.columns and "value_mm" in raw.columns:
        if "sigma_mm" not in raw.columns:
            raise ValueError(f"{path.name}: column 'sigma_mm' is missing")
        comp = raw["component"].astype(str).str.strip().str.lower().map(_ALIASES)
        bad = raw["component"][comp.isna()]
        if len(bad):
            raise ValueError(f"{path.name}: unknown component(s) {sorted(set(map(str, bad)))}; use u, e, n or los")
        long = pd.DataFrame({"id": raw["id"], "time": raw["_time"], "value_mm": raw["value_mm"].astype(float),
                             "sigma_mm": raw["sigma_mm"].astype(float), "component": comp, "_row": raw.index})
    else:
        parts = []
        for c, names in _WIDE.items():
            col = next((n for n in names if n in raw.columns), None)
            if col is None:
                continue
            scol = next((n for n in _SIGMA_WIDE[c] if n in raw.columns), "sigma_mm")
            if scol not in raw.columns:
                raise ValueError(f"{path.name}: '{col}' has no sigma column (sigma_mm or {_SIGMA_WIDE[c][0]})")
            parts.append(pd.DataFrame({"id": raw["id"], "time": raw["_time"], "value_mm": raw[col].astype(float),
                                       "sigma_mm": raw[scol].astype(float), "component": c, "_row": raw.index}))
        if not parts:
            raise ValueError(f"{path.name}: no value column (up_mm, east_mm, north_mm, los_mm, or component + value_mm)")
        long = pd.concat(parts, ignore_index=True)
    long = long[np.isfinite(long["value_mm"])]
    missing_sigma = long[~np.isfinite(long["sigma_mm"])]
    if len(missing_sigma):
        r = missing_sigma.iloc[0]
        raise ValueError(f"{path.name}: {r['id']} on {r['time'].date()} has a value but no sigma")
    not_positive = long[long["sigma_mm"] <= 0]
    if len(not_positive):
        r = not_positive.iloc[0]
        raise ValueError(f"{path.name}: {r['id']} on {r['time'].date()} has sigma {r['sigma_mm']:g} mm; "
                         "a sigma must be above zero")

    sha = _sha256(path)
    out = []
    for pid, grp in long.groupby("id", sort=True):
        rows = raw.loc[grp["_row"].unique()]
        lat, lon = rows["lat"].astype(float).unique(), rows["lon"].astype(float).unique()
        if lat.size != 1 or lon.size != 1:
            raise ValueError(f"{path.name}: point {pid} has more than one position")
        first = rows.iloc[0]
        kind = str(first["kind"]).strip() if "kind" in rows.columns and pd.notna(first["kind"]) else default_kind
        if roles and pid in roles:
            role = roles[pid]
        elif "role" in rows.columns and pd.notna(first["role"]):
            role = str(first["role"]).strip()
        else:
            role = default_role
        h = None
        for hc in ("h_m", "height_m"):
            if hc in rows.columns and pd.notna(first[hc]):
                h = float(first[hc])
        if "source" in rows.columns and pd.notna(first["source"]):
            src = str(first["source"])
        elif "source_report" in rows.columns and pd.notna(first["source_report"]):
            src = str(first["source_report"]) + (f", page {first['page']}" if "page" in rows.columns
                                                  and pd.notna(first["page"]) else "")
        else:
            src = path.name
        pt = make_point(pid, kind, lat[0], lon[0], h, src, role)
        df = (grp[["time", "value_mm", "sigma_mm", "component"]]
              .sort_values(["component", "time"]).reset_index(drop=True))
        df.attrs.update({"id": pid, "file": str(path), "sha256": sha, "gaps_filled": False})
        out.append((pt, df))
    return out


# ----------------------------------------------------------------- inventory
def _read_holdings(path: Path) -> list[dict]:
    rows = []
    for line in Path(path).read_text(errors="replace").splitlines():
        p = line.split()
        if len(p) < 11 or p[0] == "Sta":
            continue
        try:
            rows.append({"id": p[0], "lat": float(p[1]), "lon": round(((float(p[2]) + 180) % 360) - 180, 6),
                         "h_m": float(p[3]), "start": date.fromisoformat(p[7]), "end": date.fromisoformat(p[8]),
                         "n_solutions": int(p[10])})
        except ValueError:
            continue
    return rows


def _read_llh_midas(llh: Path, midas: Path) -> list[dict]:
    """Fallback: the pipeline's cached station list (unr_llh.out) with spans
    from the frozen MIDAS IGS14 file (records end by 2024.74)."""
    pos = {}
    for line in Path(llh).read_text(errors="replace").splitlines():
        p = line.split()
        try:
            pos[p[0]] = (float(p[1]), round(((float(p[2]) + 180) % 360) - 180, 6), float(p[3]) if len(p) > 3 else None)
        except (IndexError, ValueError):
            continue

    def dy(x: float) -> date:
        y = int(x)
        return date(y, 1, 1) + timedelta(days=round((x - y) * (date(y + 1, 1, 1) - date(y, 1, 1)).days))
    rows = []
    for line in Path(midas).read_text(errors="replace").splitlines():
        p = line.split()
        if len(p) < 14 or p[0] not in pos:
            continue
        try:
            t0, t1, nsol = float(p[2]), float(p[3]), int(float(p[6]))
        except ValueError:
            continue
        lat, lon, h = pos[p[0]]
        rows.append({"id": p[0], "lat": lat, "lon": lon, "h_m": h, "start": dy(t0), "end": dy(t1),
                     "n_solutions": nsol})
    return rows


def _in_box(lat, lon, bbox) -> bool:
    w, s, e, n = bbox
    if not (s <= lat <= n):
        return False
    return (w <= lon <= e) if w <= e else (lon >= w or lon <= e)    # a box across 180 deg


def stations_in_box(bbox, catalogue_dir, min_span_yr: float = 0.0, *, window=None, roles: dict | None = None,
                    default_role: str = "validation", refresh="auto", timeout: int = 180) -> list[dict]:
    """
    NGL stations inside bbox (west, south, east, north; degrees) whose data
    span is at least `min_span_yr` - the span inside `window` when one is
    given, else the whole record. From NGL's live inventory
    (DataHoldings.txt, IGS20; refreshed into catalogue_dir when the cached
    copy is over a day old, refresh='auto'), else the cached copy, else the
    pipeline's cached station list (unr_llh.out + midas.IGS14 spans, frozen
    at 2024.74, said in each Point's 'inventory').

    Roles: `roles` ({id: role}), else `default_role` ('validation': nothing
    anchors unless the protocol says so). Points are sorted by id and carry
    start, end, years_in_window and n_solutions.
    """
    cat = Path(catalogue_dir)
    d0, d1 = _window(window)
    hold = cat / HOLDINGS_FILE
    note = None
    have = hold.exists() and hold.stat().st_size > 1000
    if refresh is True or (refresh == "auto" and (not have or _age_days(hold) > 1.0)):
        try:
            cat.mkdir(parents=True, exist_ok=True)
            _download(NGL_HOLDINGS, hold, timeout, _check_holdings)
            have, note = True, "NGL DataHoldings.txt (IGS20), downloaded"
        except (OSError, ValueError) as exc:
            log.warning("NGL inventory not refreshed (%s)", exc)
            note = f"refresh failed ({exc})"
    if have:
        rows = _read_holdings(hold)
        inv = f"NGL DataHoldings.txt (IGS20) of {_mtime_iso(hold)}" + ("" if note is None or "downloaded" in note
                                                                        else f"; {note}")
    else:
        llh, mid = cat / "unr_llh.out", cat / "unr_midas_igs14.txt"
        if not (llh.exists() and mid.exists()):
            raise FileNotFoundError(f"no NGL inventory in {cat} (DataHoldings.txt, or unr_llh.out + "
                                    "unr_midas_igs14.txt) and none could be downloaded")
        rows = _read_llh_midas(llh, mid)
        inv = ("pipeline cache unr_llh.out + midas.IGS14 spans (FROZEN: records end by 2024.74; "
               "live IGS20 inventory not available)")
    out = []
    for r in rows:
        if not _in_box(r["lat"], r["lon"], bbox):
            continue
        a = max(r["start"], d0) if d0 else r["start"]
        b = min(r["end"], d1) if d1 else r["end"]
        yrs = max(0, (b - a).days) / YEAR_DAYS
        if yrs < min_span_yr:
            continue
        role = (roles or {}).get(r["id"], default_role)
        out.append(make_point(r["id"], "gnss", r["lat"], r["lon"], r["h_m"], inv, role,
                              start=r["start"].isoformat(), end=r["end"].isoformat(),
                              years_in_window=round(yrs, 2), n_solutions=r["n_solutions"], inventory=inv))
    return sorted(out, key=lambda p: p["id"])


# ----------------------------------------------------------------- protocols
def _protocol_role(roles) -> str | None:
    rs = [roles] if isinstance(roles, str) else list(roles or [])
    if "reference" in rs:
        return "reference"
    if "validation" in rs:
        return "validation"
    if "calibration" in rs or "calibration_pool" in rs:
        return "calibration"
    if "null_test" in rs or "reference_check" in rs or "null" in rs:
        return "null"
    return None                                            # not_used, duplicate_check, ...


# the one protocol-role -> Point-role mapping (validation.py records it too)
protocol_role = _protocol_role


def protocol_points(protocol) -> list[dict]:
    """The GNSS stations a fixed protocol (path or loaded dict) names, as
    Points with the API role: reference -> 'reference'; validation ->
    'validation'; calibration_pool (B: calibrates the others, scored only in
    the leave-one-out fold that excludes it) -> 'calibration'; null_test /
    reference_check -> 'null'; not_used / duplicate_check -> None. The
    protocol's own role list is kept as 'protocol_roles'. NISAR-only
    stations (truth.nisar_stations) are 'calibration' with protocol_roles
    ['nisar'] (leave-one-out within their sub-area)."""
    import yaml
    if not isinstance(protocol, dict):
        src = str(protocol)
        protocol = yaml.safe_load(Path(protocol).read_text())
    else:
        src = f"protocol {protocol.get('protocol', '')}".strip()
    truth = protocol.get("truth") or {}
    out = []
    for sid, s in (truth.get("stations") or {}).items():
        if not isinstance(s, dict) or "lat" not in s:
            continue
        pr = s.get("roles", s.get("role"))
        pr = [pr] if isinstance(pr, str) else list(pr or [])
        out.append(make_point(sid, "gnss", s["lat"], s["lon"], None, src, _protocol_role(pr),
                              protocol_roles=pr))
    for area, group in (truth.get("nisar_stations") or {}).items():
        if not isinstance(group, dict):
            continue
        for sid, s in group.items():
            if not isinstance(s, dict) or "lat" not in s or any(p["id"] == sid for p in out):
                continue
            role = None if s.get("role") == "not_used" else "calibration"
            out.append(make_point(sid, "gnss", s["lat"], s["lon"], None, src, role,
                                  protocol_roles=["nisar"], sub_area=area))
    return out


# ----------------------------------------------------------------- rates and changes
def _select(df: pd.DataFrame, window, column: str) -> pd.Series:
    if column not in df.columns:
        raise ValueError(f"no column {column!r} (have {list(df.columns)})")
    s = df[column].dropna().sort_index()
    d0, d1 = _window(window)
    if d0 is not None:
        s = s[s.index >= pd.Timestamp(d0)]
    if d1 is not None:
        s = s[s.index <= pd.Timestamp(d1)]
    return s


def station_rate(df: pd.DataFrame, window=None, component: str = "u", method: str = "linear", *,
                 annual: bool = True, min_span_yr: float = MIN_RATE_SPAN_YR) -> dict:
    """
    The station's rate over `window` in one component ('e', 'n', 'u').

    'linear': least squares, offset + rate (+ annual sin/cos when the span
    is a year or more); sigma = the least-squares error with the residuals'
    lag-1 autocorrelation r1 (effective sample size n(1-r1)/(1+r1), Santer
    et al. 2000), as the protocols specify.
    'midas': gnss_tie.midas_rate (Blewitt et al. 2016), no pair across a
    step, removed or left in the series (steps_not_removed).

    Returns dict(rate_mm_yr, sigma_mm_yr, n, span_yr, method, component,
    comparison). Under `min_span_yr` (1 yr by default; MIDAS itself needs
    1.5 yr) no rate is given: rate_mm_yr and sigma_mm_yr are None,
    comparison is 'displacement' and `reason` says why - compare the change
    between matched dates instead (station_displacement).
    """
    if component not in ("e", "n", "u"):
        raise ValueError("component must be 'e', 'n' or 'u' (LOS is projected by the harness)")
    if method not in ("linear", "midas"):
        raise ValueError("method must be 'linear' or 'midas'")
    s = _select(df, window, f"{component}_mm")
    o = np.array([d.toordinal() for d in s.index.date], dtype=np.int64)
    x = s.to_numpy(dtype=float)
    n = int(x.size)
    span = float((o[-1] - o[0]) / YEAR_DAYS) if n else 0.0
    out = {"rate_mm_yr": None, "sigma_mm_yr": None, "n": n, "span_yr": round(span, 3), "method": method,
           "component": component, "comparison": "rate"}
    if n < 3 or span < min_span_yr:
        out.update(comparison="displacement",
                   reason=f"{span:.2f} yr of data in the window (under {min_span_yr:g} yr): compare the "
                          "displacement between matched dates, not a rate")
        return out
    t = _decyear_ord(o)
    if method == "midas":
        # no pair across a step: those removed (residual error) and those left in the series
        recs = list(df.attrs.get("steps_removed", [])) + list(df.attrs.get("steps_not_removed", []))
        steps = sorted({_decyear_ord([_to_date(r["date"]).toordinal()])[0] for r in recs})
        m = gnss_tie.midas_rate(t, x, steps)
        if m is None:
            out.update(comparison="displacement",
                       reason="too few one-year pairs for MIDAS (it needs about 1.5 yr of daily data)")
            return out
        out.update(rate_mm_yr=float(m["rate_mm_yr"]), sigma_mm_yr=float(m["sigma_mm_yr"]),
                   n_pairs=int(m["n_pairs"]))
        return out
    use_annual = annual and span >= 1.0
    A = _design(t, np.array([]), use_annual)
    coef, *_ = np.linalg.lstsq(A, x, rcond=None)
    r = x - A @ coef
    p = A.shape[1]
    r1 = float(np.corrcoef(r[:-1], r[1:])[0, 1]) if n > 3 and np.std(r) > 0 else 0.0
    r1 = 0.0 if not np.isfinite(r1) else min(max(r1, 0.0), 0.99)
    n_eff = n * (1 - r1) / (1 + r1)
    if n_eff <= p:
        out.update(rate_mm_yr=float(coef[1]), sigma_mm_yr=None, r1=round(r1, 3), n_eff=round(n_eff, 1),
                   reason="too few effectively independent days for an uncertainty")
        return out
    s2 = float(r @ r) / (n_eff - p)
    cov = np.linalg.pinv(A.T @ A) * s2
    out.update(rate_mm_yr=float(coef[1]), sigma_mm_yr=float(math.sqrt(cov[1, 1])), r1=round(r1, 3),
               n_eff=round(n_eff, 1), annual=bool(use_annual))
    return out


def match_epochs(df: pd.DataFrame, dates, tol_days: int = MATCH_DAYS) -> pd.DataFrame:
    """The mean of the daily solutions within +-tol_days of each date (protocol
    epoch_matching), with the number of days used (n_days) and the mean
    per-day sigma (not divided by sqrt(n): GNSS noise is coloured). NaN where
    no day is that close - never filled."""
    idx = pd.DatetimeIndex(pd.to_datetime([_to_date(d) for d in dates]), name="date")
    df = df.sort_index()
    o = np.array([d.toordinal() for d in df.index.date], dtype=np.int64)
    q = np.array([d.toordinal() for d in idx.date], dtype=np.int64)
    lo, hi = np.searchsorted(o, q - tol_days, "left"), np.searchsorted(o, q + tol_days, "right")
    cols = [c for c in SERIES_COLUMNS if c in df.columns]
    vals = df[cols].to_numpy(dtype=float)
    res = np.full((q.size, len(cols)), np.nan)
    for i, (a, b) in enumerate(zip(lo, hi)):
        if b > a:
            res[i] = np.nanmean(vals[a:b], axis=0)
    out = pd.DataFrame(res, index=idx, columns=cols)
    out["n_days"] = (hi - lo).astype(int)
    return out


def station_displacement(df: pd.DataFrame, d0, d1, component: str = "u", tol_days: int = MATCH_DAYS) -> dict:
    """The change from d0 to d1 (mm): the mean within +-tol_days of each date,
    sigma = hypot of the two dates' per-day sigmas. For records too short
    for a rate. change_mm is None when either date has no day close enough."""
    m = match_epochs(df, [d0, d1], tol_days)
    v, s = m[f"{component}_mm"].to_numpy(), m[f"s{component}_mm"].to_numpy()
    n = m["n_days"].tolist()
    out = {"d0": _to_date(d0).isoformat(), "d1": _to_date(d1).isoformat(), "component": component,
           "n_days": n, "change_mm": None, "sigma_mm": None}
    if not (np.isfinite(v).all() and np.isfinite(s).all()):
        out["reason"] = f"no daily solution within {tol_days} days of " + \
            " and ".join(d for d, k in zip((out["d0"], out["d1"]), n) if k == 0)
        return out
    out.update(change_mm=float(v[1] - v[0]), sigma_mm=float(math.hypot(s[0], s[1])))
    return out
