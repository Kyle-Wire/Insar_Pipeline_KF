"""
Local change layer (Route B, 22 Oct 2026): vendor-style gated consecutive-date
steps at full resolution, read without phase unwrapping.

Why: the phase-linked record blends each value over ~90 x 240 m and fits one
continuous series through every pair, which spreads farmland's season-driven
phase drift into the good intervals. The vendor's point product instead counts, per point,
only the steps that pass a quality test (a failed step adds nothing). This
layer does the same from a full-resolution stack (4 x 1 looks, ~16 x 14 m).
It is a LOCAL layer: each step is read against the pixel's surroundings
(local_change_sigma_km), so it says how a spot moves against the ground around
it; anything broader than a few km is removed with the surroundings. A rejected
step adds nothing, but the gate does not remove crop effects. The counted
12-day steps keep farmland's fading bias, which is largest at 12 days (cropland
minus built-up about -15 to -18 mm/yr LOS, drift test and its re-analysis,
Site D, 22 Oct 2026). Under the shipped gate, cropland near two towns read -2.9
and -2.1 mm/yr against +7.2 and +5.5 for the few built-up pixels that pass, a
contrast of about 8-10 mm/yr, about half the pit-edge signal the layer raised
(a post-hoc check on 2-6% of built-up pixels; its size is uncertain). A spot
must be compared with ground of the same land cover and crop history, or
confirmed with long pairs (feature_check.py).
Regional figures stay with the GNSS-anchored record.

SCREENING ONLY. On 22 Oct 2026 the layer failed its pre-registered validation
against the vendor at Site D in both configurations (configuration 1: kept
share, group differences; configuration 2: kept share, town control, 4 vs 8 km
reference stability). It is not to be quoted as a finding; it points to places
to check by other means. See CHANGELOG and USER_GUIDE 6.10.

Steps (run() runs the ones named; each writes atomically and the next checks
what it reads):
  1. partners: for every pixel, the neighbours in a (2R+1)^2 window whose
     amplitude behaves alike (mean amplitude within +-partner_dmu, log-amplitude
     time-series correlation >= partner_corr), chosen separately for each period
     of local_change_partner_window_days (default ~2 years) over the whole record;
     each step uses the partners of the period its first date falls in. Over a
     10-year record chosen once, the correlation is measured so precisely that
     real neighbours on rotating cropland fall below 0.35 (Site D, 22 Oct 2026:
     median 3 partners of 120, against ~14 over 20 months), and the ground itself
     changes (crop rotation, pit advance). Each period also records each pixel's
     amplitude dispersion (sd/mean of its amplitude), the persistent-scatterer test.
     Phase similarity (the vendor's published PtSel) finds almost no partners on
     farmland that decorrelates every season, so amplitude is used.
  2. filter: each consecutive-date interferogram (normally 12 days; longer
     across missing dates) averaged over each pixel's partners, with the sample
     coherence of that average from the two dates' SLC powers.
  3. steps: the ERA5 delay change is removed from each filtered pair where the
     ERA5 grid covers the pixel, the pair is referenced to its coherence-weighted
     complex low-pass (Gaussian, sigma km), and the wrapped residual phase is
     the step. A step is unambiguous within +-lambda/4 (14 mm LOS), i.e. rates
     below ~420 mm/yr on a 12-day step (~140 mm/yr on 36 days).
  4. summary: a step counts where its coherence passes a threshold that also
     depends on how many samples the coherence rests on (step_gate: a pixel
     with few partners has a coherence estimate that noise passes most of the
     time) and the pixel has at least min_partners partners - or is a persistent
     scatterer (amplitude dispersion < local_change_ps_dispersion), which counts on
     its own coherence like the vendor's persistent targets (at 4 x 1 looks speckle
     alone gives ~0.3, so at 0.15 this path is effectively off: 0.04% of pixels at
     Site D, 22 Oct 2026); rejected steps add
     nothing; rate = counted displacement / counted time (the vendor's "while
     observed"). Each pixel's noise comes from the scatter of its own counted
     steps; significance is a two-sided t-test with n-1 degrees of freedom at
     the probability of a Gaussian min_snr (3 -> p 0.0027), with at least
     min_steps counted steps.

Needs a project with real merged SLCs (stack_virtual_merge: "False") at fine
looks (e.g. stack_range_looks 4, stack_azimuth_looks 1); unwrapping is not used.
The consecutive-date interferograms come from either source (_setup picks):
  * fine.int: a stack-route project processed to the filter step;
  * slc: formed here from the merged SLCs, when no fine.int pairs exist - the
    rlks x 1 MEAN of a x conj(b) over the window, which is what topsStack's
    burst interferogram (flatten False) merged with ISCE Looks (a mean,
    checked 22 Oct 2026) gives. This lets a project merge the phase-linked way
    (backend miaplpy, --to-step merge_reference_secondary_slc), which frees each
    date's bursts as it goes: the stack route's merge keeps both copies, ~475
    GB at Site D against ~241 GB.
Optionally (local_change_dem_error) each step loses the topographic residual
of a MintPy DEM-error map: dBperp x demErr / (R sin(inc)) - the vendor fits rate plus
DEM error; without it a 25 m DEM error (a spoil heap) puts ~3.8 mm into every
12-day step at Site D's baselines.
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

WAVELENGTH = 0.05546576                       # Sentinel-1 C band (m)
RAD2MM = -WAVELENGTH / (4 * np.pi) * 1000.0   # MintPy convention: + = towards the satellite
YEAR = 365.25
OUT = "local_change"
STEPS = ("partners", "filter", "steps", "summary")


# ----------------------------------------------------------------------------- files
def raster_info(path: Path) -> tuple[int, int, type]:
    """(rows, cols, dtype) of an ISCE raster from its .vrt."""
    t = Path(str(path) + ".vrt").read_text()
    w = int(re.search(r'rasterXSize="(\d+)"', t).group(1))
    h = int(re.search(r'rasterYSize="(\d+)"', t).group(1))
    dt = re.search(r'dataType="(\w+)"', t).group(1)
    return h, w, {"Float64": np.float64, "Float32": np.float32, "CFloat32": np.complex64}[dt]


def _mm(path, shape, dtype, mode="r"):
    return np.memmap(path, dtype=dtype, mode=mode, shape=shape)


def out_dir(cfg) -> Path:
    d = Path(cfg.project_dir) / (str(getattr(cfg, "local_change_out", "") or "") or OUT)
    d.mkdir(parents=True, exist_ok=True)
    return d


def consecutive_pairs(igram_dir: Path, start: str | None = None, end: str | None = None) -> list[str]:
    """Pair folder names between consecutive dates (normally 12 days) that have fine.int."""
    pairs = sorted(p.name for p in igram_dir.iterdir() if (p / "fine.int").exists()) if igram_dir.exists() else []
    dates = sorted({d for p in pairs for d in p.split("_")})
    want = [f"{a}_{b}" for a, b in zip(dates[:-1], dates[1:])]
    s, e = (start or "00000000").replace("-", ""), (end or "99999999").replace("-", "")
    return [p for p in want if p in pairs and p[:8] >= s and p[9:] <= e]


def date_pairs(dates: list[str], start: str | None = None, end: str | None = None) -> list[str]:
    """Consecutive-date pair names over the dates within [start, end] (the SLC source)."""
    s, e = (start or "00000000").replace("-", ""), (end or "99999999").replace("-", "")
    d = sorted(x for x in dates if s <= x <= e)
    return [f"{a}_{b}" for a, b in zip(d[:-1], d[1:])]


def offsets(R: int) -> list[tuple[int, int]]:
    return [(dy, dx) for dy in range(-R, R + 1) for dx in range(-R, R + 1)]


def partner_periods(dates: list[str], window_days: float) -> list[list[str]]:
    """The dates split into consecutive periods of about window_days (0 or less:
    one period, the whole record). A last period under half a window joins the one
    before, so no period rests on a handful of dates."""
    dates = sorted(dates)
    if not dates or not window_days or window_days <= 0:
        return [dates]
    t = np.array([np.datetime64(f"{d[:4]}-{d[4:6]}-{d[6:]}") for d in dates])
    k = ((t - t[0]) / np.timedelta64(1, "D") // float(window_days)).astype(int)
    out = [[d for d, j in zip(dates, k) if j == i] for i in range(int(k.max()) + 1)]
    out = [p for p in out if p]
    span = lambda p: (np.datetime64(f"{p[-1][:4]}-{p[-1][4:6]}-{p[-1][6:]}") -  # noqa: E731
                      np.datetime64(f"{p[0][:4]}-{p[0][4:6]}-{p[0][6:]}")) / np.timedelta64(1, "D")
    if len(out) > 1 and span(out[-1]) < window_days / 2:
        last = out.pop()                      # pop first: "out[-2] = out[-2] + out.pop()" writes to the wrong period
        out[-1] = out[-1] + last
    return out


def pair_periods(pairs: list[str], periods: list[list[str]]) -> list[int]:
    """The period each pair's FIRST date falls in (a pair across a boundary uses
    the earlier period's partners)."""
    where = {d: k for k, p in enumerate(periods) for d in p}
    return [where[p.split("_")[0]] for p in pairs]


def _days(pair: str) -> float:
    a, b = pair.split("_")
    return float((np.datetime64(f"{b[:4]}-{b[4:6]}-{b[6:]}") - np.datetime64(f"{a[:4]}-{a[4:6]}-{a[6:]}"))
                 / np.timedelta64(1, "D"))


# ----------------------------------------------------------------------------- 1. partners
_G: dict = {}


def _init(state: dict) -> None:
    _G.clear()
    _G.update(state)


def _slc(date: str, ya: int, yb: int, xa: int, xb: int) -> np.ndarray:
    """One date's merged full-resolution SLC over interferogram rows/cols (rlks x 1 grid)."""
    rl = _G["rlks"]
    return np.asarray(_mm(_G["slc_dir"] / date / f"{date}.slc.full", _G["slc_shape"], np.complex64)[ya:yb, rl * xa:rl * xb])


def looks_mean(a: np.ndarray, rlks: int) -> np.ndarray:
    """rlks x 1 mean along range (ISCE Looks, as topsStack's merge multilooks)."""
    return a.reshape(a.shape[0], -1, rlks).mean(2)


def _power(date: str, ya: int, yb: int, xa: int, xb: int) -> np.ndarray:
    """Multilooked (rlks x 1) power of one date's merged SLC over interferogram rows/cols."""
    s = _slc(date, ya, yb, xa, xb)
    return looks_mean((s.real ** 2 + s.imag ** 2).astype(np.float32), _G["rlks"])


def pair_from_slcs(s1: np.ndarray, s2: np.ndarray, rlks: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(interferogram, power 1, power 2) on the rlks x 1 grid from two full-resolution
    SLC windows: the same mean of a x conj(b) as topsStack's fine.int at these looks."""
    z = looks_mean(s1 * np.conj(s2), rlks).astype(np.complex64)
    return z, looks_mean((s1.real ** 2 + s1.imag ** 2).astype(np.float32), rlks), \
        looks_mean((s2.real ** 2 + s2.imag ** 2).astype(np.float32), rlks)


def amplitude_partners(A: np.ndarray, R: int, corr: float, dmu: float) -> np.ndarray:
    """Partner flags (n_offsets, h - 2R, w - 2R) uint8 for the interior of an
    amplitude stack A (dates, h, w): mean amplitude within +-dmu (relative) and
    log-amplitude time-series correlation >= corr. The centre is always 1."""
    mu = A.mean(0)
    L = np.log(np.maximum(A, 1e-6))
    L = L - L.mean(0)
    L = L / np.maximum(np.sqrt((L ** 2).mean(0)), 1e-6)
    hb, wb = A.shape[1] - 2 * R, A.shape[2] - 2 * R
    cm, cL = mu[R:R + hb, R:R + wb], L[:, R:R + hb, R:R + wb]
    out = np.zeros((len(offsets(R)), hb, wb), np.uint8)
    for o, (dy, dx) in enumerate(offsets(R)):
        if dy == 0 and dx == 0:
            out[o] = 1
            continue
        qm = mu[R + dy:R + dy + hb, R + dx:R + dx + wb]
        r = (cL * L[:, R + dy:R + dy + hb, R + dx:R + dx + wb]).mean(0)
        out[o] = ((np.abs(qm - cm) <= dmu * cm) & (r >= corr)).astype(np.uint8)
    return out


def amplitude_dispersion(A: np.ndarray, R: int) -> np.ndarray:
    """sd/mean of each interior pixel's amplitude over the dates of A (dates, h, w):
    the persistent-scatterer test. Four independent looks of speckle give about
    0.26, but range samples are oversampled, and at Site D's 4 x 1 looks the
    median was 0.34-0.35 (22 Oct 2026). A stable bright scatterer stays well below."""
    hb, wb = A.shape[1] - 2 * R, A.shape[2] - 2 * R
    a = A[:, R:R + hb, R:R + wb]
    return (a.std(0) / np.maximum(a.mean(0), 1e-12)).astype(np.float16)


def _partner_block(rows):
    a, b = rows
    r0, r1, c0, c1 = _G["win"]; R = _G["R"]
    A = np.stack([np.sqrt(_power(d, r0 + a - R, r0 + b + R, c0 - R, c1 + R)) for d in _G["pdates"]])
    return a, b, amplitude_partners(A, R, _G["corr"], _G["dmu"]), amplitude_dispersion(A, R)


# ----------------------------------------------------------------------------- 2. filter
def partner_filter(z: np.ndarray, p1: np.ndarray, p2: np.ndarray, W: np.ndarray, R: int) -> tuple[np.ndarray, np.ndarray]:
    """Average a (h+2R, w+2R) interferogram over each pixel's partners W
    (n_off, h, w); coherence = |sum z| / sqrt(sum P1 sum P2). Returns the mean
    complex value and the coherence, (h, w) each."""
    hb, wb = W.shape[1:]
    F = np.zeros((hb, wb), np.complex64); Q1 = np.zeros((hb, wb), np.float32)
    Q2 = np.zeros((hb, wb), np.float32); S = np.zeros((hb, wb), np.float32)
    for o, (dy, dx) in enumerate(offsets(R)):
        w = W[o].astype(np.float32)
        sl = (slice(R + dy, R + dy + hb), slice(R + dx, R + dx + wb))
        F += w * z[sl]; Q1 += w * p1[sl]; Q2 += w * p2[sl]; S += w
    den = np.sqrt(Q1 * Q2)
    coh = np.where(den > 0, np.abs(F) / np.where(den > 0, den, 1), 0).astype(np.float32)
    return (F / np.where(S > 0, S, 1)).astype(np.complex64), np.clip(coh, 0, 1)


def _filter_one(pair):
    r0, r1, c0, c1 = _G["win"]; R = _G["R"]
    hb, wb = r1 - r0, c1 - c0
    W = np.asarray(_mm(_G["partner_files"][_G["pair_period"][pair]], (len(offsets(R)), hb, wb), np.uint8))
    d1, d2 = pair.split("_")
    box = (r0 - R, r1 + R, c0 - R, c1 + R)
    if _G.get("source") == "slc":
        z, p1, p2 = pair_from_slcs(_slc(d1, *box), _slc(d2, *box), _G["rlks"])
    else:
        z = np.asarray(_mm(_G["igram_dir"] / pair / "fine.int", _G["ifg_shape"], np.complex64)[r0 - R:r1 + R, c0 - R:c1 + R])
        p1, p2 = _power(d1, *box), _power(d2, *box)
    f, c = partner_filter(z, p1, p2, W, R)
    d = _G["pairs_dir"]
    for arr, name in ((f, f"{pair}.c8"), (c, f"{pair}.coh.f4")):
        arr.tofile(d / f"{name}.tmp")
        os.replace(d / f"{name}.tmp", d / name)
    return pair, float(np.median(c))


# ----------------------------------------------------------------------------- 3. steps
def lowpass(z: np.ndarray, w: np.ndarray, sig_y: float, sig_x: float, k: int = 16) -> np.ndarray:
    """Weighted complex Gaussian low-pass of unit phasors (computed on a k x k
    block grid, then upsampled): the smooth reference each step is read against."""
    from scipy import ndimage
    h, wd = z.shape
    H, Wd = max(k, h // k * k), max(k, wd // k * k)
    u = np.where(np.abs(z) > 0, z / np.maximum(np.abs(z), 1e-12), 0) * w
    u = np.pad(u, ((0, max(0, H - h)), (0, max(0, Wd - wd))))[:H, :Wd]
    b = u.reshape(H // k, k, Wd // k, k).mean((1, 3))
    br = ndimage.gaussian_filter(b.real, (sig_y / k, sig_x / k), mode="nearest")
    bi = ndimage.gaussian_filter(b.imag, (sig_y / k, sig_x / k), mode="nearest")
    lp = ndimage.zoom(br + 1j * bi, (h / br.shape[0], wd / br.shape[1]), order=1)
    return lp[:h, :wd]


def step_mm(z: np.ndarray, coh: np.ndarray, sig_y: float, sig_x: float, weather_mm: np.ndarray | None = None) -> np.ndarray:
    """One consecutive-date step (mm LOS, + = towards the satellite): the wrapped
    phase of the pair against its own low-pass, after removing the modelled
    weather delay change (mm, same sign convention; 0 where not covered)."""
    if weather_mm is not None:
        z = z * np.exp(-1j * (np.nan_to_num(weather_mm) / RAD2MM)).astype(np.complex64)
    lp = lowpass(z, coh ** 2, sig_y, sig_x)
    return (np.angle(z * np.conj(lp)) * RAD2MM).astype(np.float32)


# ----------------------------------------------------------------------------- 4. gate, rate, noise
def coherence_floor(n_partners: np.ndarray, rlks: int, alpha: float = 0.05, eff_ratio: float = 0.65) -> np.ndarray:
    """The coherence that pure noise exceeds with probability alpha, for a
    sample coherence over rlks x (partners + 1) samples (eff_ratio of them
    independent; neighbouring range samples are correlated): for N independent
    samples P(|gamma| >= g) = (1 - g^2)^(N - 1), so g = sqrt(1 - alpha^(1/(N-1)))."""
    n_eff = np.maximum(2.0, eff_ratio * rlks * (np.asarray(n_partners, float) + 1))
    return np.sqrt(1 - alpha ** (1 / (n_eff - 1)))


def step_gate(coh: np.ndarray, n_partners: np.ndarray, thr: float, rlks: int, min_partners: int,
              alpha: float = 0.05, eff_ratio: float = 0.65, ps: np.ndarray | None = None) -> np.ndarray:
    """Which steps count: coherence >= max(thr, coherence_floor) and at least
    min_partners partners, or (ps) a persistent scatterer, which stands on its own
    coherence. Without the floor, a pixel with 0-3 partners passes thr 0.3 in
    50-75% of pure-noise steps (review, 22 Oct 2026); with it, noise passes at
    about alpha whatever the partner count, so the floor, not min_partners, is
    what keeps a persistent scatterer with no partners honest (0 partners at 4
    looks: coherence >= 0.92)."""
    floor = coherence_floor(n_partners, rlks, alpha, eff_ratio)
    enough = np.asarray(n_partners) >= min_partners
    if ps is not None:
        enough = enough | np.asarray(ps, bool)
    return enough & (coh >= np.maximum(thr, floor))


def gated_rate(steps: np.ndarray, accepted: np.ndarray, days: np.ndarray) -> dict:
    """The vendor's bookkeeping over steps (n, ...): accepted steps add their
    movement and their time; rejected steps add nothing. Returns counted
    displacement (mm), counted years, counted steps, rate while observed (mm/yr;
    NaN under 0.05 yr) and accepted share."""
    acc = np.asarray(accepted, bool) & np.isfinite(steps)
    disp = np.where(acc, steps, 0).sum(0)
    yrs = (np.asarray(days, float).reshape((-1,) + (1,) * (steps.ndim - 1)) * acc).sum(0) / YEAR
    rate = np.where(yrs > 0.05, disp / np.maximum(yrs, 1e-9), np.nan)
    return {"displacement_mm": disp, "years": yrs, "n_steps": acc.sum(0), "rate_mm_yr": rate,
            "accepted_share": acc.mean(0)}


def rate_noise(steps: np.ndarray, accepted: np.ndarray, days: np.ndarray) -> np.ndarray:
    """1-sigma of the rate while observed (mm/yr) from the scatter of each
    pixel's own counted steps: sd(steps) x sqrt(n) / counted years. rate/noise
    is then a t statistic with n - 1 degrees of freedom (see significant)."""
    acc = np.asarray(accepted, bool) & np.isfinite(steps)
    n = acc.sum(0)
    yrs = (np.asarray(days, float).reshape((-1,) + (1,) * (steps.ndim - 1)) * acc).sum(0) / YEAR
    mean = np.where(n > 0, np.where(acc, steps, 0).sum(0) / np.maximum(n, 1), 0)
    sd = np.sqrt(np.where(acc, (steps - mean) ** 2, 0).sum(0) / np.maximum(n - 1, 1))
    return np.where((n > 1) & (yrs > 0.05), sd * np.sqrt(n) / np.maximum(yrs, 1e-9), np.nan)


def significant(t: np.ndarray, n_steps: np.ndarray, min_snr: float = 3.0, min_steps: int = 10) -> np.ndarray:
    """Two-sided t-test of rate/noise with n - 1 degrees of freedom at the
    probability a Gaussian min_snr has (3 -> p 0.0027), and at least min_steps
    counted steps. A fixed |t| >= 3 passes 20% of pure noise at n = 2."""
    from scipy import stats
    p_max = 2 * stats.norm.sf(min_snr)
    n = np.asarray(n_steps)
    df = np.maximum(n - 1, 1)
    p = 2 * stats.t.sf(np.abs(np.nan_to_num(t)), df)
    return (n >= max(2, min_steps)) & np.isfinite(t) & (p <= p_max)


# ----------------------------------------------------------------------------- driver
def _setup(cfg) -> dict:
    stack = Path(cfg.stack_dir); M = stack / "merged"; I = M / "interferograms"
    slc_dir = M / "SLC"
    dates = sorted(d.name for d in slc_dir.iterdir() if (d / f"{d.name}.slc.full").exists()) if slc_dir.exists() else []
    if not dates:
        virtual = slc_dir.exists() and any(slc_dir.glob("*/*.slc.full.vrt"))
        raise RuntimeError(f"no merged SLC files under {slc_dir}: "
                           + ("the merge was virtual; set stack_virtual_merge: \"False\" and re-merge"
                              if virtual else "keep them (they are removed only by a manual clean-up)"))
    slc_shape = raster_info(slc_dir / dates[0] / f"{dates[0]}.slc.full")[:2]
    rlks = int(cfg.stack_range_looks); alks = int(cfg.stack_azimuth_looks)
    start, end = str(cfg.start_date or ""), str(cfg.end_date or "")
    pairs = consecutive_pairs(I, start, end)
    if pairs:
        source = "fine_int"
        ifg_shape = raster_info(I / pairs[0] / "fine.int")[:2]
    else:
        # no stack-route interferograms (a phase-linked-style merge): form them here
        source = "slc"
        pairs = date_pairs(dates, start, end)
        ifg_shape = (slc_shape[0], slc_shape[1] // max(1, rlks))
        if not pairs:
            raise RuntimeError(f"no consecutive-date pairs: no fine.int under {I} and under two merged SLC dates in "
                               f"{slc_dir} within {start or 'start'}..{end or 'end'}")
        lat = M / "geom_reference" / "lat.rdr"
        if Path(str(lat) + ".vrt").exists() and raster_info(lat)[:2] != ifg_shape:
            raise RuntimeError(f"merged geometry {raster_info(lat)[:2]} is not on the {rlks} x {alks} grid of the "
                               f"SLCs {ifg_shape}: re-merge the geometry at these looks")
    if alks != 1 or slc_shape[1] // rlks != ifg_shape[1]:
        raise RuntimeError(f"local_change needs azimuth looks 1 and SLC width / range looks = interferogram width "
                           f"(got {cfg.stack_range_looks} x {cfg.stack_azimuth_looks}, SLC {slc_shape}, ifg {ifg_shape})")
    from .unwrap_crop import compute_crop_window
    cw = compute_crop_window(cfg, write=False)
    R = int(cfg.local_change_window)
    win = (max(cw["r0"], R), min(cw["r1"], ifg_shape[0] - R), max(cw["c0"], R), min(cw["c1"], ifg_shape[1] - R))
    od = out_dir(cfg)
    wdays = float(getattr(cfg, "local_change_partner_window_days", 0) or 0)
    periods = partner_periods(dates, wdays)
    one = len(periods) == 1
    pfiles = [od / f"partners_R{R}.u8"] if one else [od / f"partners_R{R}_p{k}.u8" for k in range(len(periods))]
    dfiles = [od / f"dispersion_R{R}.f2"] if one else [od / f"dispersion_R{R}_p{k}.f2" for k in range(len(periods))]
    pp = pair_periods(pairs, periods)
    return {"igram_dir": I, "pairs": pairs, "source": source, "ifg_shape": ifg_shape, "slc_dir": slc_dir, "dates": dates,
            "slc_shape": slc_shape, "rlks": rlks, "win": win, "R": R, "corr": float(cfg.local_change_partner_corr),
            "dmu": float(cfg.local_change_partner_dmu), "periods": periods, "window_days": wdays,
            "partner_files": pfiles, "dispersion_files": dfiles, "pair_period": dict(zip(pairs, pp)),
            "partners_meta": od / f"partners_R{R}.json", "pairs_dir": od / "pairs",
            "geom": M / "geom_reference", "out": od}


def _partner_meta(st) -> dict:
    return {"window": list(st["win"]), "R": st["R"], "corr": st["corr"], "dmu": st["dmu"], "dates": st["dates"],
            "window_days": st["window_days"], "periods": [[p[0], p[-1], len(p)] for p in st["periods"]]}


def variant_dir(cfg, tag: str) -> Path:
    """<project>/local_change/variants/<tag>: steps.h5, GeoTIFFs and record of a
    re-run of the 'steps' and 'summary' steps with other settings (e.g. sigma
    8 km), reading the base run's partners and filtered pairs."""
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", tag or ""):
        raise ValueError(f"variant tag {tag!r}: letters, digits, '_', '.', '-' only")
    d = out_dir(cfg) / "variants" / tag
    d.mkdir(parents=True, exist_ok=True)
    return d


def run(cfg, steps=STEPS, workers: int | None = None, variant: str | None = None) -> dict:
    """Run the steps named, in order. Each checks what it reads (the partner
    flags' window and settings, every filtered pair) and writes atomically, so a
    failed run leaves the previous good files in place. A variant re-runs only
    'steps' and 'summary' into variant_dir(cfg, variant)."""
    bad = [s for s in steps if s not in STEPS]
    if bad:
        raise ValueError(f"unknown local change step(s) {bad}; choose from {STEPS}")
    if "steps" in steps or "summary" in steps:
        weather_source(cfg)                      # an explicit, wrong ERA5 path fails now, not hours later
        dem_error_source(cfg)
    st = _setup(cfg)
    if variant:
        if set(steps) - {"steps", "summary"}:
            raise ValueError("a variant re-runs only the 'steps' and 'summary' steps")
        st["out"] = variant_dir(cfg, variant)
    # the SLC source reads two full-resolution windows per pair (~0.4 GB each at Site D)
    workers = workers or max(1, min(8 if st["source"] == "slc" else 12, (os.cpu_count() or 2) - 2))
    r0, r1, c0, c1 = st["win"]; hb, wb = r1 - r0, c1 - c0
    rec_f = st["out"] / "local_change.json"
    rec = json.loads(rec_f.read_text()) if rec_f.exists() else {}
    t0 = time.time()
    if "partners" in steps:
        st["partners_meta"].unlink(missing_ok=True)      # stale until every period is rewritten
        per = []
        for k, pdates in enumerate(st["periods"]):
            tmp = Path(str(st["partner_files"][k]) + ".tmp"); dtmp = Path(str(st["dispersion_files"][k]) + ".tmp")
            W = _mm(tmp, (len(offsets(st["R"])), hb, wb), np.uint8, "w+")
            D = _mm(dtmp, (hb, wb), np.float16, "w+")
            blocks = [(i, min(i + 48, hb)) for i in range(0, hb, 48)]
            with Pool(workers, initializer=_init, initargs=({**st, "pdates": pdates},)) as pool:
                for a, b, w, da in pool.imap_unordered(_partner_block, blocks):
                    W[:, a:b] = w
                    D[a:b] = da
            W.flush(); D.flush()
            n = (np.asarray(W[:, ::7, ::7]) > 0).sum(0) - 1
            dsp = np.asarray(D[::7, ::7], np.float32)
            del W, D
            os.replace(tmp, st["partner_files"][k]); os.replace(dtmp, st["dispersion_files"][k])
            per.append({"first": pdates[0], "last": pdates[-1], "dates": len(pdates), "median": float(np.median(n)),
                        "p10": float(np.percentile(n, 10)), "p90": float(np.percentile(n, 90)),
                        "dispersion_median": float(np.nanmedian(dsp))})
            log.info("Local change: partners %s-%s (%d dates): median %.0f (p10 %.0f, p90 %.0f) of %d; amplitude "
                     "dispersion median %.2f", pdates[0], pdates[-1], len(pdates), per[-1]["median"], per[-1]["p10"],
                     per[-1]["p90"], len(offsets(st["R"])) - 1, per[-1]["dispersion_median"])
        st["partners_meta"].write_text(json.dumps(_partner_meta(st)))
        rec["partners"] = {"window_px": 2 * st["R"] + 1, "corr": st["corr"], "dmu": st["dmu"], "dates": len(st["dates"]),
                           "window_days": st["window_days"], "periods": per,
                           "median": float(np.median([x["median"] for x in per]))}
    if "filter" in steps or "steps" in steps:
        meta = json.loads(st["partners_meta"].read_text()) if st["partners_meta"].exists() else None
        if meta != _partner_meta(st):
            raise RuntimeError("the partner flags on disk were built for a different window, settings or date list; "
                               "run the 'partners' step again")
    if "filter" in steps:
        st["pairs_dir"].mkdir(parents=True, exist_ok=True)
        with Pool(workers, initializer=_init, initargs=(st,)) as pool:
            med = dict(pool.imap_unordered(_filter_one, st["pairs"]))
        days = [_days(p) for p in st["pairs"]]
        counts = {str(int(d)): days.count(d) for d in sorted(set(days))}
        rec["filter"] = {"pairs": len(med), "source": st["source"], "median_coherence": float(np.median(list(med.values()))),
                         "days_counts": counts}
        log.info("Local change: %d consecutive-date pairs filtered (days: %s), median coherence %.2f", len(med), counts,
                 rec["filter"]["median_coherence"])
    if "steps" in steps:
        rec["steps"] = _steps(cfg, st)
    if "summary" in steps:
        rec["summary"] = summarize(cfg, st["out"] / "steps.h5")
    rec.update({"window": [r0, r1, c0, c1], "pairs": st["pairs"], "source": st["source"], "variant": variant,
                "sigma_km": float(cfg.local_change_sigma_km), "seconds": round(time.time() - t0),
                "written": time.strftime("%Y-%m-%dT%H:%M:%S")})
    rec_f.write_text(json.dumps(rec, indent=1))
    return rec


def weather_source(cfg) -> Path | None:
    """The ERA5.h5 to use. An explicit local_change_era5 that is missing, or has
    no geometryRadar.h5 beside it, is an error; the default <mintpy>/inputs one
    is optional (then no weather is removed, with a warning)."""
    p = str(getattr(cfg, "local_change_era5", "") or "")
    if p:
        if not Path(p).exists():
            raise FileNotFoundError(f"local_change_era5 {p} does not exist")
        if not (Path(p).parent / "geometryRadar.h5").exists():
            raise FileNotFoundError(f"local_change_era5: no geometryRadar.h5 beside {p}")
        return Path(p)
    c = Path(cfg.mintpy_dir) / "inputs" / "ERA5.h5"
    return c if c.exists() and (c.parent / "geometryRadar.h5").exists() else None


def geo_sample(h5: Path, name: str, LAT, LON) -> np.ndarray:
    """Nearest-cell values of a MintPy geocoded dataset (Y_FIRST/X_FIRST/Y_STEP/X_STEP)
    at each LAT/LON; NaN outside its grid."""
    import h5py
    with h5py.File(h5, "r") as f:
        a = f.attrs
        y0, x0, dy, dx = (float(a[k]) for k in ("Y_FIRST", "X_FIRST", "Y_STEP", "X_STEP"))
        d = f[name]
        ny, nx = d.shape[-2:]
        r = np.floor((np.asarray(LAT) - y0) / dy).astype(int)
        c = np.floor((np.asarray(LON) - x0) / dx).astype(int)
        ok = (r >= 0) & (r < ny) & (c >= 0) & (c < nx) & np.isfinite(LAT) & np.isfinite(LON)
        if not ok.any():
            return np.full(np.shape(LAT), np.nan)
        rr, cc = r[ok], c[ok]
        y_lo, y_hi, x_lo, x_hi = rr.min(), rr.max() + 1, cc.min(), cc.max() + 1
        block = np.asarray(d[y_lo:y_hi, x_lo:x_hi] if d.ndim == 2 else d[0, y_lo:y_hi, x_lo:x_hi], np.float64)
    out = np.full(np.shape(LAT), np.nan)
    out[ok] = block[rr - y_lo, cc - x_lo]
    return out


def topo_factor(dem_error_file: Path, LAT, LON) -> np.ndarray:
    """demErr / (R sin(inc)) per pixel (m of LOS per m of perpendicular baseline),
    from a MintPy geo_demErr.h5 with geo_geometryRadar.h5 beside it; 0 where
    either is missing (no correction rather than a guess)."""
    geom = Path(dem_error_file).parent / "geo_geometryRadar.h5"
    dz = geo_sample(dem_error_file, "dem", LAT, LON)
    rng = geo_sample(geom, "slantRangeDistance", LAT, LON)
    inc = np.radians(geo_sample(geom, "incidenceAngle", LAT, LON))
    k = dz / (rng * np.sin(inc))
    return np.where(np.isfinite(k), k, 0.0)


def topo_step_mm(k: np.ndarray, bperp_a: float, bperp_b: float) -> np.ndarray:
    """The DEM-error residual of one step (mm LOS, MintPy convention, as
    mintpy dem_error's G = bperp / (R sin(inc)))."""
    return 1000.0 * (float(bperp_b) - float(bperp_a)) * k


def bperp_by_date(baseline_dir: Path) -> dict:
    """Perpendicular baseline per date (m) from a topsStack baselines/ folder."""
    from mintpy.utils import isce_utils
    bl = isce_utils.read_baseline_timeseries(str(baseline_dir), processor="tops") or {}
    return {d: float(np.mean(v)) for d, v in bl.items()}


def dem_error_source(cfg) -> tuple[Path, Path] | None:
    """(geo_demErr.h5, baselines dir) when local_change_dem_error is set; a set
    path that is missing, or has no geo_geometryRadar.h5 or baselines, is an error."""
    p = str(getattr(cfg, "local_change_dem_error", "") or "")
    if not p:
        return None
    f = Path(p)
    if not f.exists() or not (f.parent / "geo_geometryRadar.h5").exists():
        raise FileNotFoundError(f"local_change_dem_error {p}: missing, or no geo_geometryRadar.h5 beside it")
    b = Path(str(getattr(cfg, "local_change_baselines", "") or "") or Path(cfg.stack_dir) / "baselines")
    if not b.exists():
        raise FileNotFoundError(f"local_change_dem_error needs per-date baselines: {b} does not exist "
                                "(set local_change_baselines)")
    return f, b


def _weather_map(era: Path, LAT, LON, coslat):
    """(file, date index, rows, cols, covered) mapping each window pixel to its
    nearest ERA5-grid pixel; covered is False beyond 3 grid spacings, where no
    weather is removed rather than extrapolating."""
    import h5py
    from scipy.spatial import cKDTree
    f = h5py.File(era, "r")
    g = h5py.File(era.parent / "geometryRadar.h5", "r")
    elat, elon = g["latitude"][:], g["longitude"][:]
    ok = np.isfinite(elat) & (elat != 0)
    tree = cKDTree(np.c_[elat[ok], elon[ok] * coslat])
    sp = float(np.median(tree.query(tree.data[:: max(1, len(tree.data) // 2000)], k=2)[0][:, 1]))
    d, nn = tree.query(np.c_[LAT.ravel(), LON.ravel() * coslat])
    ey, ex = np.nonzero(ok)
    return f, {x.decode(): i for i, x in enumerate(f["date"][:])}, ey[nn], ex[nn], (d <= 3 * sp).reshape(LAT.shape)


def _steps(cfg, st) -> dict:
    import h5py
    r0, r1, c0, c1 = st["win"]; hb, wb = r1 - r0, c1 - c0
    missing = [p for p in st["pairs"] if not ((st["pairs_dir"] / f"{p}.c8").exists() and (st["pairs_dir"] / f"{p}.coh.f4").exists())]
    if missing:
        raise RuntimeError(f"{len(missing)} filtered pair(s) missing ({missing[:4]}...); run the 'filter' step")
    lh, lw, ldt = raster_info(st["geom"] / "lat.rdr")
    LAT = np.asarray(_mm(st["geom"] / "lat.rdr", (lh, lw), ldt)[r0:r1, c0:c1], np.float64)
    LON = np.asarray(_mm(st["geom"] / "lon.rdr", (lh, lw), ldt)[r0:r1, c0:c1], np.float64)
    coslat = np.cos(np.radians(float(np.nanmedian(LAT))))
    dy_m = abs(float(np.nanmedian(np.diff(LAT, axis=0)))) * 111320 or 14.0
    dx_m = abs(float(np.nanmedian(np.diff(LON, axis=1)))) * 111320 * coslat or 16.0
    sig = float(cfg.local_change_sigma_km) * 1000
    era = weather_source(cfg)
    E = _weather_map(era, LAT, LON, coslat) if era is not None else None
    if E is None:
        log.warning("Local change: no ERA5 delays (local_change_era5, or mintpy/inputs/ERA5.h5 with geometryRadar.h5); "
                    "steps keep the weather that the low-pass does not remove")
    dem = dem_error_source(cfg)
    K = bp = None
    if dem is not None:
        K = topo_factor(dem[0], LAT, LON)
        bp = bperp_by_date(dem[1])
        log.info("Local change: DEM-error term from %s (|demErr/(R sin inc)| median %.2e, %d dates with baselines)",
                 dem[0], float(np.median(np.abs(K))), len(bp))
    no_baseline = []
    npart = np.zeros((len(st["periods"]), hb, wb), np.uint8)
    disp = np.zeros((len(st["periods"]), hb, wb), np.float16)
    for k, f in enumerate(st["partner_files"]):
        W = _mm(f, (len(offsets(st["R"])), hb, wb), np.uint8)
        for a in range(0, hb, 256):
            npart[k, a:a + 256] = (np.asarray(W[:, a:a + 256]).sum(0) - 1).astype(np.uint8)
        del W
        if st["dispersion_files"][k].exists():
            disp[k] = np.fromfile(st["dispersion_files"][k], np.float16).reshape(hb, wb)
        else:
            disp[k] = np.nan
    n = len(st["pairs"])
    sf = st["out"] / "steps.h5"; tmp = Path(str(sf) + ".tmp")
    no_weather = []
    with h5py.File(tmp, "w") as h:
        S = h.create_dataset("step_mm", (n, hb, wb), np.float32, chunks=(1, min(256, hb), min(256, wb)))
        C = h.create_dataset("coherence", (n, hb, wb), np.float16, chunks=(1, min(256, hb), min(256, wb)))
        h["latitude"] = LAT.astype(np.float32); h["longitude"] = LON.astype(np.float32)
        h["partners"] = npart                                   # (periods, h, w)
        h["dispersion"] = disp
        h["pair_period"] = np.array([st["pair_period"][p] for p in st["pairs"]], np.int16)
        h["pair"] = np.array(st["pairs"], dtype="S17"); h["days"] = np.array([_days(p) for p in st["pairs"]])
        if E is not None:
            h["weather_covered"] = E[4].astype(np.uint8)
        for k, p in enumerate(st["pairs"]):
            z = np.fromfile(st["pairs_dir"] / f"{p}.c8", np.complex64).reshape(hb, wb)
            c = np.fromfile(st["pairs_dir"] / f"{p}.coh.f4", np.float32).reshape(hb, wb)
            wmm = None
            if E is not None:
                f, ix, eyy, exx, cov = E
                a, b = p.split("_")
                if a in ix and b in ix:
                    wmm = np.where(cov, (f["timeseries"][ix[b]] - f["timeseries"][ix[a]])[eyy, exx].reshape(hb, wb) * 1000.0, 0.0)
                else:
                    no_weather.append(p)
            if K is not None:
                a, b = p.split("_")
                if a in bp and b in bp:
                    wmm = topo_step_mm(K, bp[a], bp[b]) + (0.0 if wmm is None else wmm)
                else:
                    no_baseline.append(p)
            S[k] = step_mm(z, c, sig / dy_m, sig / dx_m, wmm)
            C[k] = c.astype(np.float16)
        h.attrs.update({"sigma_km": float(cfg.local_change_sigma_km), "weather": str(era) if era else "none",
                        "dem_error": str(dem[0]) if dem else "none", "source": st.get("source", "fine_int"),
                        "convention": "mm LOS, + towards the satellite", "window": [r0, r1, c0, c1],
                        "rlks": int(st["rlks"]), "complete": True})
    os.replace(tmp, sf)
    out = {"file": str(sf), "pairs": n, "sigma_km": float(cfg.local_change_sigma_km),
           "weather": str(era) if era else None, "pairs_without_weather": no_weather,
           "dem_error": str(dem[0]) if dem else None, "pairs_without_baseline": no_baseline}
    if E is not None:
        out["weather_coverage"] = round(float(E[4].mean()), 4)
    return out


def _open_steps(steps_file: Path):
    import h5py
    if not Path(steps_file).exists():
        raise FileNotFoundError(f"{steps_file} missing; run the 'steps' step")
    h = h5py.File(steps_file, "r")
    if not h.attrs.get("complete") or len(h["pair"]) != h["step_mm"].shape[0]:
        h.close()
        raise RuntimeError(f"{steps_file} is incomplete; run the 'steps' step again")
    return h


def _gate_args(cfg) -> dict:
    return {"thr": float(cfg.local_change_threshold), "min_partners": int(getattr(cfg, "local_change_min_partners", 4)),
            "alpha": float(getattr(cfg, "local_change_noise_alpha", 0.05)),
            "eff_ratio": float(getattr(cfg, "local_change_eff_sample_ratio", 0.65)),
            "ps_dispersion": float(getattr(cfg, "local_change_ps_dispersion", 0.0) or 0.0)}


def _gate_inputs(h, rows=None, cols=None):
    """(partner counts (periods, ...), persistent-scatterer flags (periods, ...) or
    None, each step's period) from a steps.h5; a file from before per-period
    partners has one 2-D partners layer and no dispersion."""
    sl = (slice(None),) if rows is None else (slice(None), rows, cols)
    npart = h["partners"]
    if npart.ndim == 2:
        npart = npart[sl[1:]][None] if rows is not None else npart[:][None]
        disp = None
    else:
        npart = npart[sl] if rows is not None else npart[:]
        disp = (h["dispersion"][sl] if rows is not None else h["dispersion"][:]) if "dispersion" in h else None
    pp = h["pair_period"][:] if "pair_period" in h else np.zeros(h["step_mm"].shape[0], np.int16)
    return npart, disp, pp


def _ps(disp, thr):
    if disp is None or not thr or thr <= 0:
        return None
    return np.nan_to_num(np.asarray(disp, np.float32), nan=np.inf) < thr


def summarize(cfg, steps_file: Path) -> dict:
    """GeoTIFFs: rate while observed, accepted share, counted displacement, noise,
    t statistic, counted steps and the significance-masked rate (a map cell shows
    only where most of its radar pixels are significant)."""
    ga = _gate_args(cfg)
    min_snr = float(getattr(cfg, "local_change_min_snr", 3.0)); min_steps = int(getattr(cfg, "local_change_min_steps", 10))
    h = _open_steps(steps_file)
    try:
        days = h["days"][:]; LAT = h["latitude"][:]; LON = h["longitude"][:]
        npart, disp, pp = _gate_inputs(h)
        ps_all = _ps(disp, ga["ps_dispersion"])
        rlks = int(h.attrs.get("rlks", getattr(cfg, "stack_range_looks", 4)))
        keys = ("displacement_mm", "years", "n_steps", "rate_mm_yr", "accepted_share", "noise_mm_yr")
        g = {k: np.full(LAT.shape, np.nan, np.float32) for k in keys}
        for a in range(0, LAT.shape[0], 256):            # row blocks: a 10-year record has ~270 steps per pixel
            b = min(a + 256, LAT.shape[0])
            s_, c_ = h["step_mm"][:, a:b], h["coherence"][:, a:b].astype(np.float32)
            acc = step_gate(c_, npart[:, a:b][pp], ga["thr"], rlks, ga["min_partners"], ga["alpha"], ga["eff_ratio"],
                            None if ps_all is None else ps_all[:, a:b][pp])
            part = gated_rate(s_, acc, days)
            part["noise_mm_yr"] = rate_noise(s_, acc, days)
            for k in g:
                g[k][a:b] = part[k]
    finally:
        h.close()
    t = np.where(g["noise_mm_yr"] > 0, g["rate_mm_yr"] / g["noise_mm_yr"], np.nan)
    sig = significant(t, g["n_steps"], min_snr, min_steps)
    sig_rate = np.where(sig, g["rate_mm_yr"], np.nan)
    od = Path(steps_file).parent
    files = {}
    for name, arr in (("local_change_rate_mm_yr", g["rate_mm_yr"]), ("local_change_accepted_share", g["accepted_share"]),
                      ("local_change_counted_displacement_mm", g["displacement_mm"]), ("local_change_counted_steps", g["n_steps"]),
                      ("local_change_noise_1sigma_mm_yr", g["noise_mm_yr"]), ("local_change_t", t),
                      ("local_change_rate_significant_mm_yr", sig_rate)):
        files[name] = str(_grid_tif(od / f"{name}.tif", LAT, LON, np.asarray(arr, np.float32), cfg, name,
                                    majority=name == "local_change_rate_significant_mm_yr"))
    # only pixels with enough counted steps can be significant; counting the
    # rest pulled the share below what chance gives (review, 22 Oct 2026)
    v = np.isfinite(t) & (np.nan_to_num(g["n_steps"]) >= max(2, min_steps))
    return {**ga, "min_snr": min_snr, "min_steps": min_steps, "files": files, "eligible_pixels": int(v.sum()),
            "persistent_share": None if ps_all is None else float(ps_all.mean()),
            "accepted_share_median": float(np.nanmedian(g["accepted_share"])),
            "noise_1sigma_median_mm_yr": float(np.nanmedian(g["noise_mm_yr"])),
            "share_significant": float(sig[v].mean()) if v.any() else None,
            "expected_by_chance": float(2 * __import__("scipy").stats.norm.sf(min_snr)),
            "meaning": "rate = counted displacement / counted time (steps that pass the gate); against the surroundings "
                       "within ~sigma km, not a fixed reference; + = towards the satellite. Screening only: the layer "
                       "failed its pre-registered gate (22 Oct 2026). Read only the significant rate, as places to check "
                       "by other means, and compare its share with what chance gives. share_significant counts the whole "
                       "processed window, not only the map's bbox."}


def _grid_tif(path: Path, lat, lon, val, cfg, name, step_m: float = 15.0, majority: bool = False) -> Path:
    """Mean of radar cells per ~15 m geographic cell over cfg.bbox (step in
    degrees worked out at the area's latitude). A cell no radar pixel fell in
    takes its nearest neighbour's value within 1.5 cells; a cell whose radar
    pixels are all NaN stays NaN. majority: show a cell only where more than
    half of its radar pixels hold a value. That is for the significance map,
    where NaN means "not significant": one significant pixel among others is
    not a significant cell (review lc-3, 22 Oct 2026: one pixel became a 3 x 3
    patch, and the map showed 2-2.5 times the significant share of the radar
    pixels in its area; now 1.34% of cells against 1.44% of pixels at Site D)."""
    import rasterio
    from rasterio.transform import from_origin
    from scipy import ndimage
    w, s, e, n = cfg.bbox
    lat0 = float(np.nanmedian(lat))
    dlat = step_m / 111320.0; dlon = step_m / (111320.0 * np.cos(np.radians(lat0)))
    ny, nx = int(np.ceil((n - s) / dlat)), int(np.ceil((e - w) / dlon))
    geo = np.isfinite(lat) & np.isfinite(lon)                  # no position: outside the grid
    r = np.floor((n - np.where(geo, lat, n + dlat)) / dlat).astype(int)
    c = np.floor((np.where(geo, lon, w - dlon) - w) / dlon).astype(int)
    inside = (r >= 0) & (r < ny) & (c >= 0) & (c < nx)
    ok = inside & np.isfinite(val)
    acc = np.zeros(ny * nx); cnt = np.zeros(ny * nx); radar = np.zeros(ny * nx)
    np.add.at(acc, r[ok] * nx + c[ok], val[ok]); np.add.at(cnt, r[ok] * nx + c[ok], 1)
    np.add.at(radar, r[inside] * nx + c[inside], 1)
    show = cnt > (0.5 * radar if majority else 0)
    out = np.where(show, acc / np.maximum(cnt, 1), np.nan).reshape(ny, nx).astype(np.float32)
    hole = (radar == 0).reshape(ny, nx)
    if hole.any() and (~hole).any():
        dist, (iy, ix) = ndimage.distance_transform_edt(hole, return_indices=True)
        fill = hole & (dist <= 1.5)
        out[fill] = out[iy[fill], ix[fill]]
    with rasterio.open(path, "w", driver="GTiff", width=nx, height=ny, count=1, dtype="float32", crs="EPSG:4326",
                       nodata=np.nan, transform=from_origin(w, n, dlon, dlat), compress="deflate") as d:
        d.write(out, 1)
        d.update_tags(layer=name, method="local_change (Route B): gated consecutive-date steps, see local_change.py",
                      sigma_km=str(cfg.local_change_sigma_km), threshold=str(cfg.local_change_threshold))
    return path


def sample(cfg, steps_file: Path, lat, lon, start: str | None = None, end: str | None = None) -> dict:
    """Gated figures at the nearest step cell of each point, over the pairs
    within [start, end] (YYYYMMDD; None = all). Points farther than 2.5 cell
    spacings from any cell (outside the processed window) are NaN."""
    from scipy.spatial import cKDTree
    ga = _gate_args(cfg)
    h = _open_steps(steps_file)
    try:
        LAT = h["latitude"][:]; LON = h["longitude"][:]
        cos = np.cos(np.radians(float(np.nanmedian(LAT))))
        ok = np.isfinite(LAT)
        yy, xx = np.nonzero(ok)
        tree = cKDTree(np.c_[LAT[ok], LON[ok] * cos])
        spacing = float(np.median(tree.query(tree.data[:: max(1, len(tree.data) // 2000)], k=2)[0][:, 1]))
        pts = np.c_[np.asarray(lat, float), np.asarray(lon, float) * cos]
        good = np.isfinite(pts).all(1)
        d = np.full(len(pts), np.inf); j = np.zeros(len(pts), int)
        if good.any():
            d[good], j[good] = tree.query(pts[good])
        near = good & (d <= 2.5 * spacing)
        pairs = [x.decode() for x in h["pair"][:]]
        sel = np.array([(start is None or p[:8] >= start) and (end is None or p[9:] <= end) for p in pairs])
        ry, rx = yy[j], xx[j]
        r0, r1, c0, c1 = ry.min(), ry.max() + 1, rx.min(), rx.max() + 1
        S = h["step_mm"][:, r0:r1, c0:c1][:, ry - r0, rx - c0][sel]
        C = h["coherence"][:, r0:r1, c0:c1][:, ry - r0, rx - c0][sel].astype(np.float32)
        npart, disp, pp = _gate_inputs(h, slice(r0, r1), slice(c0, c1))
        npart = npart[:, ry - r0, rx - c0]
        ps_pts = _ps(None if disp is None else disp[:, ry - r0, rx - c0], ga["ps_dispersion"])
        days = h["days"][:][sel]
        rlks = int(h.attrs.get("rlks", 4))
    finally:
        h.close()
    acc = step_gate(C, npart[pp][sel], ga["thr"], rlks, ga["min_partners"], ga["alpha"], ga["eff_ratio"],
                    None if ps_pts is None else ps_pts[pp][sel])
    g = gated_rate(S, acc, days)
    for k in ("rate_mm_yr", "accepted_share"):
        g[k] = np.where(near, g[k], np.nan)
    g["near"] = near
    g["pairs"] = [p for p, s_ in zip(pairs, sel) if s_]
    return g
