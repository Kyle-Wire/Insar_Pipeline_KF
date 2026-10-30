"""Injection-recovery: plant movement of known size and see whether the
report-area rule reads it as moving. A TECHNICAL record only: it writes no
client sentence (7 Oct 2026, E4).

WHY THIS EXISTS. No positive-control site exists (no surveyed subsidence
inside a processed frame), so "nothing detected" had no measured
sensitivity behind it: whether the linear deramp, the DEM-error regression
and the velocity fit attenuate a realistic mining signal had never been
measured, and that is the first question in a deposition. This plants
signals of known size into the official product and runs the decision the
report makes for a REPORT AREA (DECISION_RULE): an area drawn in advance
over the patch, set against the report's own farmland.

WHAT "FOUND" MEANS, AND WHAT IT DOES NOT. "Found" is the report-area rule
over an area drawn exactly on the planted patch, sinking only. It is NOT
the map's search for unmapped movement (client_report._feature_search: 500 m
smoothing, a size floor and a chance test against 59 surrogate maps), which
is far less sensitive: measured 7 Oct 2026 on Site A, the search listed
2 of 10 planted cases that this rule had all read as moving (a 0.3 km bowl
sinking 40 mm/yr at its centre was not even coloured on the map). Until
7 Oct 2026 the record's sentence ("Patches about 0.3 km across ... were
found 20 of 20 times") read as a claim about the search; there is no client
sentence now, and the record states the rule (decision_rule). One difference
from a report area remains and is stated there: the drawn area is not taken
out of its own comparison ground (a report area is, with 500 m round it,
E28). Measured 7 Oct 2026 on Site A (20 farmland sites; 0.3, 1 and 2 km;
nothing planted and 10-40 mm/yr), taking it out changed none of the 240
decisions; the median threshold and excess moved 0.08 mm/yr LOS or less.

THE COMPARISON GROUND IS THE REPORT'S (report_farmland): the second farmland
pass is rebuilt inside this stage, because this stage runs before the client
report. Until 7 Oct 2026 the sites and the nulls came from the first pass,
which still held unmapped sinking ground (Site A Feature 1): 11-18% of
farmland sites lay on it and read "found" 32-55% of the time with nothing
planted. A ZERO-RATE CONTROL (nothing planted, the same footprints) is run
for every site and width, so each detection rate is read beside the rate at
which the same rule fires on unplanted ground; every rate carries its
Clopper-Pearson 95% range (20 sites: 16 of 20 is 56-94%).

HOW, WITHOUT RERUNNING MINTPY OR COPYING THE CUBE. The three steps after
the troposphere are linear in the data:
  deramp      a per-date least-squares plane [row, col, 1] over the deramp
              mask (MintPy utils1.deramp: cells sampled when > 1e6, plane
              re-referenced to REF_Y/REF_X), fitted in the grid MintPy used
              - radar coordinates on the stack and phase-linked routes;
  DEM error   per cell, the bperp column of [bperp, 1, t, t^2(, steps)] fitted
              over the non-excluded dates and removed (dem_error.py; the
              per-cell range/incidence scale does not change the span);
  velocity    the linear-rate coefficient of timeseries2velocity's design.
A planted signal s(x)*f(t) therefore comes out as
    (s(x) - plane_s(x) + plane_s(ref)) * rate(f - bperp*delta_f)
and the product's own velocity is already the processed real data, so
    velocity with the planting = official velocity + that term
exactly, by linearity. Only the dates, bperp and masks are read; the cube is
never touched. The planted velocity is then class-matched with the real
field and set against the farmland null with significance.assess - the
rule behind every report-area reading - over the patch's half-peak footprint.

WHAT IT DOES NOT COVER, and the record says so: the feature search (above),
phase linking, unwrapping (aliasing above the stated ceiling; unwrapping
errors), and the coherence mask removing the most active cells. A planted
patch is measured on the cells the real product kept. The recovery here is
therefore an UPPER bound on what the full chain achieves. Patches wider than
the default 2 km, planted on farmland, pull the farmland zero and the deramp
(a 7.3 km disk read 2 of 20 on Site A, recovered/planted 0.78) and are
not representative of a mine-sized area, whose limit rests on its own null:
such widths are flagged, and so is any farmland row recovered below 0.95 of
planted once the temporal gain (which a mine suffers too) is divided out.

Signals (vertical, converted to line of sight with the site's own incidence),
all SINKING - planted with a negative sign, since a positive rate in the
product moves toward the satellite; `rate_mm_yr` is the size, the planted_*
values carry the sign:
  trough       Gaussian bowl, steady rate; widths are the full width at half
               the peak; rates are the peak vertical rate;
  settlement   the same footprint settling as A(1 - exp(-(t - t_s)/tau)),
               starting 30% into the record with tau = 1.5 yr - spoil or a
               reclaimed pit - scaled so its straight-line rate over the
               record is the stated rate.
  control      nothing planted, the same half-peak footprints (rate 0): on
               farmland, the false-alarm rate of the same rule; inside the
               mines, the mine ground's own reading (it may really move).
Sites: random cells on the report's farmland (second pass) and inside the
mapped mine polygons, seeded and recorded so a rerun gives the same table.

    python -m insar_pipeline.injection_test <config.yaml> --out-dir <dir>
writes <dir>/injection_recovery.json (default deliverables/technical/).
"""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

SIZES_KM = (0.3, 0.6, 1.0, 2.0)          # full width at half the peak
RATES_MM_YR = (10.0, 20.0, 40.0)         # vertical rate at the patch centre (the control adds rate 0)
TYPES = ("trough", "settlement")
CONTROL = "control"                      # the zero-rate control's type: nothing planted
REPRESENTATIVE_MIN_RATIO = 0.95          # recovered/planted below this: the planting moved the zero/deramp
DECISION_RULE = (
    "'Found' is the report-area rule, sinking only: an area drawn in advance exactly over the planted patch "
    "(its reliable cells where the planted signal is at least half its peak) has its class-matched mean set "
    "against the report's farmland null (significance.assess), and is found when that mean differs from the "
    "null median by at least the null's 95% threshold AND the difference is sinking (negative). A difference of "
    "the other sign is not found (counted as wrong_sign). This is the rule behind a report area's reading, with "
    "one difference: the drawn area stays in its own comparison ground, where a report area is taken out of it "
    "with 500 m round it (E28); on Site A taking it out changed none of 240 decisions at widths of 2 km or "
    "less. It is not the map's search for unmapped movement (the feature search and its chance test), which is "
    "less sensitive and is not tested here. The control row is the same rule with nothing planted.")
SETTLEMENT_START = 0.30                  # share of the record before settlement begins
SETTLEMENT_TAU_YR = 1.5
N_FARMLAND = 20
N_MINE = 6
SEED = 20261005
WAVELENGTH_M = 0.05546576                # Sentinel-1 C band: only when the product states no WAVELENGTH
                                         # (M1, 8 Oct 2026: _aliasing reads the product's own)
MM_IN = 25.4
NO_INCIDENCE = ("no incidence angle in the product geometry: the vertical-to-LOS conversion is unknown "
                "(no 0.78 default)")
NOT_COVERED = ("the map's search for unmapped movement (the feature search) is not tested: 'found' is the "
               "report-area rule over an area drawn on the patch. Phase linking, phase unwrapping (including "
               "aliasing of fast, narrow movement) and the coherence mask are not exercised: planted movement "
               "is added after them, on the cells the real product kept, so the recovery is an upper bound "
               "for the full chain")


# ---------------------------------------------------------------- MintPy settings
def _attr_list(v) -> list[str]:
    """MintPy writes list attributes as their Python repr: "['a', 'b']", "[]",
    "exclude_date.txt", "None"."""
    if v is None:
        return []
    s = v.decode() if isinstance(v, bytes) else str(v)
    s = s.strip().strip("[]")
    out = [x.strip().strip("'\"") for x in s.split(",")]
    return [x for x in out if x and x.lower() not in ("none", "no", "auto")]


def _exclude_dates(cfg, attr_value) -> list[str]:
    """Dates a MintPy step left out of its fit; a file name is read from mintpy/."""
    out = []
    for item in _attr_list(attr_value):
        if item.isdigit() and len(item) == 8:
            out.append(item)
            continue
        f = Path(item) if Path(item).is_absolute() else Path(cfg.mintpy_dir) / item
        if f.exists():
            out += [ln.split()[0] for ln in f.read_text().splitlines()
                    if ln.strip() and not ln.startswith("#") and ln.split()[0].isdigit()]
    return sorted(set(out))


def chain_settings(cfg, ts_file: Path) -> dict:
    """What the official run applied after the troposphere, read from the
    product's own attributes (the name tokens as a fallback)."""
    import h5py
    with h5py.File(ts_file, "r") as f:
        a = {k: (v.decode() if isinstance(v, bytes) else v) for k, v in f.attrs.items()}
    vel_attrs = {}
    try:
        from .crosscheck import product
        with h5py.File(product(cfg, "velocity.h5"), "r") as f:
            vel_attrs = {k: (v.decode() if isinstance(v, bytes) else v) for k, v in f.attrs.items()}
    except (OSError, ImportError):
        pass
    name = ts_file.name
    deramp = str(a.get("mintpy.deramp", "linear" if "ramp" in name else "no")).lower()
    if "ramp" not in name:
        deramp = "no"
    mask = a.get("mintpy.deramp.maskFile") or ""
    if not mask or not Path(mask).exists():
        mask = str(Path(cfg.mintpy_dir) / "maskDeramp.h5")
    poly = vel_attrs.get("mintpy.timeFunc.polynomial", 1)
    try:
        poly = int(str(poly).strip("[]").split(",")[0] or 1)
    except ValueError:
        poly = 1
    return {
        "deramp": deramp, "deramp_mask": mask if deramp != "no" else None,
        "dem_error": "demErr" in name,
        "dem_poly_order": int(str(a.get("mintpy.topographicResidual.polyOrder", 2)).strip() or 2),
        "dem_step_dates": _attr_list(a.get("mintpy.topographicResidual.stepDate")),
        "dem_exclude_dates": _exclude_dates(cfg, a.get("mintpy.topographicResidual.excludeDate")),
        "velocity_polynomial": poly,
        "velocity_periodic": [float(x) for x in _attr_list(vel_attrs.get("mintpy.timeFunc.periodic"))],
        "velocity_exclude_dates": _exclude_dates(cfg, vel_attrs.get("mintpy.timeFunc.excludeDate")),
        "timeseries_file": str(ts_file),
    }


# ---------------------------------------------------------------- the temporal operators
def years_of(dates) -> np.ndarray:
    d0 = datetime.strptime(dates[0], "%Y%m%d")
    return np.array([(datetime.strptime(d, "%Y%m%d") - d0).days / 365.25 for d in dates])


def dem_error_operator(f: np.ndarray, t: np.ndarray, bperp: np.ndarray, poly_order: int = 2,
                       step_idx=(), use: np.ndarray | None = None) -> np.ndarray:
    """f minus its bperp component, as dem_error.py removes it: the bperp
    coefficient of [bperp, 1, t, .., t^p, steps] fitted over `use` dates,
    times bperp, subtracted from every date."""
    cols = [bperp] + [t ** p for p in range(poly_order + 1)]
    for i in step_idx:
        cols.append((np.arange(t.size) >= i).astype(float))
    G = np.vstack(cols).T
    use = np.ones(t.size, bool) if use is None else use
    coef = np.linalg.lstsq(G[use], f[use], rcond=None)[0]
    return f - bperp * coef[0]


def velocity_operator(f: np.ndarray, t: np.ndarray, polynomial: int = 1, periodic=(),
                      use: np.ndarray | None = None) -> float:
    """The linear-rate coefficient timeseries2velocity fits (per year)."""
    cols = [t ** p for p in range(polynomial + 1)]
    for P in periodic:
        cols += [np.sin(2 * np.pi * t / P), np.cos(2 * np.pi * t / P)]
    G = np.vstack(cols).T
    use = np.ones(t.size, bool) if use is None else use
    return float(np.linalg.lstsq(G[use], f[use], rcond=None)[0][1])


def time_function(kind: str, t: np.ndarray) -> np.ndarray:
    """Unit time functions whose straight-line rate over the record is 1/yr -
    the rate an exact chain would report - so recovered/planted measures what
    the chain does, not how a line fits a curve. (Scaled instead to its
    average rate, total/span, a settlement starting 30% in read 1.39 on
    Site B with no chain attenuation at all.)"""
    T = float(t[-1] - t[0])
    if kind == "trough":
        return t - t[0]
    ts_ = t[0] + SETTLEMENT_START * T
    g = np.where(t > ts_, 1.0 - np.exp(-(t - ts_) / SETTLEMENT_TAU_YR), 0.0)
    slope = float(np.polyfit(t, g, 1)[0]) if np.ptp(g) > 0 else 0.0
    return g / slope if slope > 0 else g


def temporal_gain(kind: str, t, bperp, S: dict, dates) -> float:
    """Recovered rate per unit planted straight-line rate after DEM error + velocity."""
    f = time_function(kind, t)
    if S["dem_error"] and bperp is not None and np.ptp(bperp) > 0:
        use = np.array([d not in set(S["dem_exclude_dates"]) for d in dates])
        steps = [i for i, d in enumerate(dates) if d in set(S["dem_step_dates"])]
        f = dem_error_operator(f, t, bperp, S["dem_poly_order"], steps, use)
    usev = np.array([d not in set(S["velocity_exclude_dates"]) for d in dates])
    return velocity_operator(f, t, S["velocity_polynomial"], S["velocity_periodic"], usev)


# ---------------------------------------------------------------- the spatial operator
class Deramp:
    """MintPy's linear deramp of a planted field, evaluated on the geo grid.

    The plane is fitted in the grid MintPy fitted it in (the mask's own
    rows/cols - radar on the stack routes). Geo cells are placed in that grid
    by an affine fit of (row, col) on (lat, lon) over the mask grid; a plane
    of an affine map is a plane, so this is exact up to the non-affinity of
    the radar geometry (`affine_rms_px`, recorded)."""

    def __init__(self, cfg, mask_file: str | None, ramp: str = "linear"):
        import h5py
        self.active = bool(mask_file) and Path(mask_file).exists() and ramp == "linear"
        self.note = None
        if not self.active:
            if ramp not in ("no", "linear"):
                self.note = (f"the run removed a {ramp} ramp, which is not modelled here; the recovery "
                             "does not include its attenuation and is overstated")
            elif ramp == "linear":
                self.note = (f"the run deramped but its mask ({mask_file}) is gone; the plane's attenuation "
                             "is not included, so the recovery is overstated")
            return
        with h5py.File(mask_file, "r") as f:
            key = "mask" if "mask" in f else next(iter(f.keys()))
            m = f[key][:].astype(bool)
            a = {k: (v.decode() if isinstance(v, bytes) else v) for k, v in f.attrs.items()}
        ny, nx = m.shape
        if "X_FIRST" in a:
            self.coords = "geo"
            lat = float(a["Y_FIRST"]) + (np.arange(ny) + 0.5) * float(a["Y_STEP"])
            lon = float(a["X_FIRST"]) + (np.arange(nx) + 0.5) * float(a["X_STEP"])
            LAT = np.repeat(lat[:, None], nx, 1)
            LON = np.repeat(lon[None, :], ny, 0)
        else:
            self.coords = "radar"
            g = Path(cfg.mintpy_dir) / "inputs" / "geometryRadar.h5"
            with h5py.File(g, "r") as f:
                LAT, LON = f["latitude"][:].astype("float64"), f["longitude"][:].astype("float64")
            if LAT.shape != m.shape:
                raise ValueError(f"deramp mask {m.shape} and geometryRadar {LAT.shape} differ")
        rows, cols = np.mgrid[0:ny, 0:nx]
        ok = np.isfinite(LAT) & np.isfinite(LON) & (LAT != 0)
        msk = m & ok
        # MintPy samples uniformly above 1e6 cells
        n = int(msk.sum())
        if n > 1e6:
            step = int(np.ceil(np.sqrt(n / 1e6)))
            samp = np.zeros_like(msk)
            samp[step // 2::step, step // 2::step] = True
            msk &= samp
        self.lat, self.lon = LAT[msk], LON[msk]
        G = np.c_[rows[msk].astype("float64"), cols[msk].astype("float64"), np.ones(int(msk.sum()))]
        self.P = np.linalg.pinv(G)                       # (3, N): plane coefficients = P @ s
        # (row, col) as an affine function of (lat, lon), fitted on every valid cell
        sel = ok.copy()
        if sel.sum() > 400_000:
            k = int(np.ceil(np.sqrt(sel.sum() / 400_000)))
            thin = np.zeros_like(sel); thin[::k, ::k] = True; sel &= thin
        A = np.c_[LAT[sel], LON[sel], np.ones(int(sel.sum()))]
        self.aff_r = np.linalg.lstsq(A, rows[sel].astype("float64"), rcond=None)[0]
        self.aff_c = np.linalg.lstsq(A, cols[sel].astype("float64"), rcond=None)[0]
        res = np.hypot(A @ self.aff_r - rows[sel], A @ self.aff_c - cols[sel])
        self.affine_rms_px = float(np.sqrt(np.mean(res ** 2)))
        self.ref = (int(float(a["REF_Y"])), int(float(a["REF_X"]))) if "REF_Y" in a and "REF_X" in a else None
        self.n_cells = int(msk.sum())
        del LAT, LON, rows, cols

    def plane(self, s_at_mask: np.ndarray, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
        """Plane fitted to the planted field over the mask, at (lat, lon),
        minus its value at the reference pixel (MintPy re-references the ramp)."""
        c = self.P @ s_at_mask
        r = self.aff_r[0] * lat + self.aff_r[1] * lon + self.aff_r[2]
        k = self.aff_c[0] * lat + self.aff_c[1] * lon + self.aff_c[2]
        out = c[0] * r + c[1] * k + c[2]
        if self.ref is not None:
            out = out - (c[0] * self.ref[0] + c[1] * self.ref[1] + c[2])
        return out


def gaussian(lat, lon, lat0, lon0, fwhm_km) -> np.ndarray:
    """Unit-peak Gaussian bowl of the given full width at half maximum."""
    sigma = fwhm_km * 1000.0 / 2.3548
    dy = (lat - lat0) * 110_950.0
    dx = (lon - lon0) * 111_320.0 * np.cos(np.radians(lat0))
    return np.exp(-(dx * dx + dy * dy) / (2 * sigma * sigma))


# ---------------------------------------------------------------- run
def _choose_sites(F: dict, rng, n_farm: int, n_mine: int, max_km: float) -> list[dict]:
    """Seeded random site centres: on the report's farmland in the AOI (F
    from report_farmland: the second pass, so never on a feature the report
    found) and on reliable cells inside the mapped mines, each at least one largest
    width from the grid edge (the half-peak footprint stays on the grid) and
    1.5 widths from the reference pixel (3.5 sigma: the planted signal there
    is 0.2% of its peak, so referencing does not subtract it)."""
    from . import significance as sg
    attrs, good = F["attrs"], F["good"]
    ny, nx = good.shape
    dy_m, dx_m = sg.pixel_metres(attrs)
    pad_r = int(np.ceil(max_km * 1000 / dy_m)); pad_c = int(np.ceil(max_km * 1000 / dx_m))
    inner = np.zeros_like(good)
    if ny > 2 * pad_r and nx > 2 * pad_c:
        inner[pad_r:ny - pad_r, pad_c:nx - pad_c] = True
    ref = sg.reference_pixel(attrs)
    if ref:
        rr, cc = np.mgrid[0:ny, 0:nx]
        inner &= ((rr - ref["row"]) * dy_m) ** 2 + ((cc - ref["col"]) * dx_m) ** 2 > (1.5 * max_km * 1000) ** 2
    out = []
    mine = F.get("mine")
    for kind, pool, n in (("farmland", F["far"] & inner, n_farm),
                          ("mine", (mine & good & inner) if mine is not None else None, n_mine)):
        if pool is None or not pool.any() or n <= 0:
            continue
        idx = np.flatnonzero(pool)
        pick = rng.choice(idx, size=min(n, idx.size), replace=False)
        for j in pick:
            r, c = divmod(int(j), nx)
            la, lo = sg.rc_to_lalo(attrs, r, c)
            out.append({"kind": kind, "row": r, "col": c, "lat": round(float(la), 6), "lon": round(float(lo), 6)})
    return out


def _los_up_map(cfg, shape) -> np.ndarray | None:
    from .crosscheck import _read2d, product
    for name in ("geometryRadar.h5", "geometryGeo.h5"):
        f = product(cfg, name)
        if f.exists():
            try:
                inc, _ = _read2d(f, "incidenceAngle")
            except (KeyError, StopIteration):
                continue
            if inc.shape == shape:
                inc = inc.astype("float64")
                inc[inc == 0] = np.nan
                return np.cos(np.radians(inc))
    return None


def _aliasing(cfg, attrs, dates, los_up: float) -> dict:
    """The unwrapping ceiling the test cannot see: a quarter wavelength of
    line-of-sight change between neighbouring cells within one revisit.
    The wavelength and the fallback revisit are the product's own
    (stack_qc.product_wavelength_m / revisit_days; M1, 8 Oct 2026): a quarter
    wavelength is 13.9 mm for Sentinel-1 and ~61 mm for NISAR L-band. A
    non-Sentinel-1 record also names its wavelength; a Sentinel-1 record is
    unchanged."""
    from . import significance as sg
    from . import stack_qc
    wl = stack_qc.product_wavelength_m(attrs, WAVELENGTH_M)
    posting = None
    try:
        from . import resolution
        posting = float(resolution.effective_resolution(cfg, attrs).get("posting_m"))
    except Exception:                            # noqa: BLE001 - geo posting as the fallback
        posting = None
    if not posting:
        posting = float(np.mean(sg.pixel_metres(attrs)))
    dd = np.diff([datetime.strptime(d, "%Y%m%d").toordinal() for d in dates])
    revisit = float(np.median(dd)) if dd.size else stack_qc.revisit_days([], attrs)
    q_mm = wl / 4 * 1000
    grad = q_mm / (posting * revisit / 365.25)           # mm/yr of LOS rate per metre
    rows = []
    for w in SIZES_KM:
        sigma = w * 1000 / 2.3548
        rows.append({"width_km": w, "max_peak_vertical_mm_yr": round(grad * sigma / np.exp(-0.5) / los_up, 0)})
    out = {"quarter_wavelength_mm": round(q_mm, 1), "revisit_days": revisit, "posting_m": round(posting, 1),
           "max_los_rate_gradient_mm_yr_per_m": round(grad, 2), "by_width": rows,
           "note": ("a Gaussian trough faster than this changes by more than a quarter wavelength between "
                    "neighbouring cells within one revisit and cannot be unwrapped; not tested here")}
    if not stack_qc.is_sentinel1_wavelength(wl):
        out["wavelength_m"] = wl
        out["band"] = stack_qc.sensor_band(wl)
    return out


def _report_area_cells(cfg, ctx) -> tuple[np.ndarray | None, list[str]]:
    """The union of the report areas other than the whole study area (their
    whole outlines on the grid, client_report._mask's footprint) and their
    names: not comparison ground, since an area must not be compared with
    itself (E28; client_report passes the same union to
    significance.far_field_mask(exclude=...))."""
    from . import client_report as cr
    union, names = None, []
    for name, spec in cr.parse_areas(cfg):
        if spec.strip().lower() == "full":
            continue
        _, _, fp = cr._mask(ctx, spec)
        union = fp.astype(bool) if union is None else (union | fp.astype(bool))
        names.append(name)
    return union, names


def report_farmland(cfg) -> tuple[dict, dict]:
    """
    The fields the report reads and the comparison ground it reads them
    against, built the way client_report.compute builds them - inside this
    stage, because the stage runs BEFORE the quality assessment and the
    client report (pipeline.py): this run's client_report.json, with its
    farmland_second_pass cut, does not exist yet, and the previous run's
    would be stale.
      1. panel_history._Context: the reliable cells (the second gate
         included), the far field and the wide far field (first pass);
      2. the report areas other than the whole study area taken out of the
         comparison ground (_report_area_cells), by the report's own rule:
         significance.far_field_mask(exclude=...), which grows them by its
         exclude buffer (500 m), for the far field and the wide far field;
      3. the report's feature search (client_report._feature_search) with no
         QA stand-out - the QA has not run, and the chance test alone decides
         what is listed and cut (E7) - and its leave-candidate-out re-test
         (client_report._leave_candidate_out, F1); when a feature beats
         chance, either way, its cells grown by
         client_report.FARMLAND_SECOND_PASS_M are taken out
         (panel_history.exclude_from_farmland): the report's second pass.
    Until 7 Oct 2026 sites and nulls came from crosscheck.load_fields, the
    first pass: on Site A 11-18% of farmland sites lay on Feature 1 plus
    500 m and read "found" 32-55% of the time with nothing planted, and the
    95% thresholds were wider than the report's (2 km: 8.84 against 7.23 mm/yr
    LOS). Returns (F, what was done): F carries the keys of
    crosscheck.load_fields this module reads.
    """
    from . import client_report as cr
    from . import panel_history as ph
    from . import report_insights as ri
    ctx = ph._Context(cfg)
    sg = ctx.sg
    info = {"definition": ("the client report's comparison ground: reliable cells outside the mapped mines plus "
                           "buffer, without the report areas other than the whole study area (plus "
                           "report_area_buffer_m), and without the "
                           "features that beat chance plus their buffer (the second farmland pass), rebuilt in "
                           "this stage"),
            "far_cells_first_pass": int(ctx.far.sum()), "far_wide_cells_first_pass": int(ctx.far_wide.sum())}
    excl, names = _report_area_cells(cfg, ctx)
    info["report_areas_excluded"] = names
    if excl is not None and excl.any():
        # the same calls _Context made, with the areas out (E28, contract C2)
        far, meta = sg.far_field_mask(cfg, ctx.good, ctx.attrs, exclude=excl)
        far_wide, _ = sg.far_field_mask(cfg, ctx.good, ctx.attrs, restrict_to_aoi=False, exclude=excl)
        info["report_area_buffer_m"] = meta.get("exclude_buffer_m")
        if not (np.array_equal(far, ctx.far) and np.array_equal(far_wide, ctx.far_wide)):
            ctx.far, ctx.far_wide = far, far_wide
            ph.measure_farmland(ctx)
        del far, far_wide, meta
    info["far_cells_without_report_areas"] = int(ctx.far.sum())
    del excl
    try:
        los_up_map = cr._los_up(cfg, ctx.good.shape)
    except FileNotFoundError:
        # a geometry with the incidence but no azimuth: cos(incidence) is the
        # same up factor (deliverables.geometry_field), filled with its own median
        los_up_map = _los_up_map(cfg, ctx.good.shape)
        if los_up_map is None or not np.isfinite(los_up_map).any():
            raise RuntimeError(NO_INCIDENCE) from None
        los_up_map = np.where(np.isfinite(los_up_map), los_up_map, float(np.nanmedian(los_up_map)))
    poly = ri.polygon_masks(ctx, ri.mine_layers(cfg))
    aoi = sg.aoi_mask(cfg.bbox, ctx.attrs, ctx.good.shape)
    sm, cands, null, assessed = cr._feature_search(ctx, los_up_map, poly, aoi, None)
    # F1 (follow-up of 7 Oct 2026): the report also cuts a leading candidate
    # that beats chance only with itself left out of the farmland
    # (client_report._leave_candidate_out, the same call); without it the
    # sites and nulls would keep Site A's Feature 1 + 500 m, which the
    # report cuts. Up to LEAVE_OUT_LEADING (3) more nulls in this stage.
    loo = cr._leave_candidate_out(ctx, sm, cands, null, los_up_map, poly, aoi, None)
    del sm, null, los_up_map, poly
    found = [f["_mask"] for f in cands if f.get("stands_out") or (f.get("leave_out") or {}).get("passed")]
    info.update({"feature_search_made": bool(assessed), "features_cut": len(found),
                 "features_cut_on_leave_out": sum(1 for f in cands if not f.get("stands_out")
                                                  and (f.get("leave_out") or {}).get("passed")),
                 "leave_out_tested": bool(loo.get("tested")),
                 "buffer_m": float(cr.FARMLAND_SECOND_PASS_M)})
    if found:
        second = ph.exclude_from_farmland(ctx, found, cr.FARMLAND_SECOND_PASS_M)
        cut = np.zeros(ctx.good.shape, bool)
        for m in found:
            cut |= m
        info["second_pass"] = {k: second.get(k) for k in ("far_cells", "far_wide_cells", "class_centre_shift_mm_yr")}
        info["cut_cells"] = cr.mask_runs(cut)
        del cut
    del found, cands
    info.update({"far_cells": int(ctx.far.sum()), "far_wide_cells": int(ctx.far_wide.sum())})
    F = {"attrs": ctx.attrs, "vel": ctx.vel, "tc": ctx.tc, "good": ctx.good, "far": ctx.far,
         "far_wide": ctx.far_wide, "aoi": ctx.aoi, "placement": ctx.r_place, "mine": ctx.union}
    return F, info


def run(cfg, out_dir: str | Path | None = None, *, n_farmland: int = N_FARMLAND, n_mine: int = N_MINE,
        sizes_km=SIZES_KM, rates_mm_yr=RATES_MM_YR, types=TYPES, seed: int = SEED) -> Path:
    """Plant, process, decide; write injection_recovery.json. Returns its path.

    Every site and width also gets the zero-rate control (nothing planted),
    whatever the rates asked; a rate of 0 or less in `rates_mm_yr` is that
    control and is not planted again. The record carries no client sentence
    ('sentence' is None): it is a technical record (decision_rule,
    technical_summary, the summary table).

    On failure the file is written with `status: failed` and the cause, then
    the error is raised - never a silent "done"."""
    from . import crosscheck as cc
    from . import significance as sg
    t0 = time.time()
    out = Path(out_dir) if out_dir else Path(cfg.project_dir) / "deliverables" / "technical"
    path = out / "injection_recovery.json"
    planted_rates = [float(r) for r in rates_mm_yr if float(r) > 0]
    rec: dict = {"test": "injection-recovery", "generated": datetime.now().isoformat(timespec="seconds"),
                 "project": str(cfg.project_dir), "seed": int(seed), "sizes_km": list(sizes_km),
                 "rates_mm_yr": planted_rates, "types": list(types),
                 "covers": ("deramp, DEM error, velocity fit, class matching against the report's farmland "
                            "(second pass) and the report-area decision"),
                 "not_covered": NOT_COVERED, "decision_rule": DECISION_RULE,
                 "control": {"type": CONTROL, "rate_mm_yr": 0.0,
                             "what": ("nothing planted, the same half-peak footprints at every site and width: on "
                                      "farmland the false-alarm rate of the rule; inside the mines the mine "
                                      "ground's own reading, which may be real movement")},
                 # a technical record only: no sentence for the client (E4, 7 Oct 2026)
                 "sentence": None,
                 "settlement": {"start_share_of_record": SETTLEMENT_START, "tau_yr": SETTLEMENT_TAU_YR},
                 "rate_definition": ("vertical rate of SINKING at the centre: for the trough its steady rate, "
                                     "for the settlement the straight-line rate of the planted curve over the "
                                     "record (what an exact chain would report); planted and recovered values "
                                     "are negative, as subsidence reads in the product. The average over the "
                                     "half-peak footprint is about 0.72 of the centre rate "
                                     "(planted_region_vertical_mm_yr)"),
                 "footprint": "cells where the planted signal is at least half its peak (and reliable)"}
    if len(planted_rates) != len(list(rates_mm_yr)):
        rec["rates_dropped"] = {"rates_mm_yr": [float(r) for r in rates_mm_yr if float(r) <= 0],
                                "why": "a rate of 0 is the control, run for every site and width; nothing "
                                       "below 0 is planted (every planted patch sinks)"}
    try:
        F, rec["farmland"] = report_farmland(cfg)
        ts_file = cc.timeseries_file(cfg)
        dates, bperp = cc.read_dates(ts_file)
        t = years_of(dates)
        S = chain_settings(cfg, ts_file)
        rec["chain"] = {k: v for k, v in S.items() if k != "dem_exclude_dates"} | \
            {"dem_exclude_dates": len(S["dem_exclude_dates"])}
        gains = {k: temporal_gain(k, t, bperp, S, dates) for k in types}
        rec["temporal_gain"] = {k: round(v, 4) for k, v in gains.items()}
        D = Deramp(cfg, S["deramp_mask"], S["deramp"])
        rec["deramp"] = ({"modelled": True, "coordinates": D.coords, "mask_cells_used": D.n_cells,
                          "affine_rms_px": round(D.affine_rms_px, 3), "mask": S["deramp_mask"]}
                         if D.active else {"modelled": False, "note": D.note or "no deramp in this chain"})
        attrs, good, shape = F["attrs"], F["good"], F["good"].shape
        lat_c = float(attrs["Y_FIRST"]) + (np.arange(shape[0]) + 0.5) * float(attrs["Y_STEP"])
        lon_c = float(attrs["X_FIRST"]) + (np.arange(shape[1]) + 0.5) * float(attrs["X_STEP"])
        up_map = _los_up_map(cfg, shape)
        up_med = cc.los_up_median(cfg, good)
        if up_map is None and up_med is None:
            raise RuntimeError(NO_INCIDENCE)
        # sites must hold the largest patch; on a grid too small for it,
        # the sizes that do not fit are left out (and listed), not the test
        sites, sizes_km = [], sorted(sizes_km)
        while sizes_km and not sites:
            sites = _choose_sites(F, np.random.default_rng(seed), n_farmland, n_mine, max(sizes_km))
            if not sites:
                rec.setdefault("sizes_skipped_km", []).append(sizes_km.pop())
        if not sites:
            raise RuntimeError("no reliable farmland cell far enough from the edge and the reference pixel")
        if rec.get("sizes_skipped_km"):
            log.warning("injection: patches of %s km do not fit on this grid away from its edge and the "
                        "reference pixel; left out", rec["sizes_skipped_km"])
        wide = [w for w in sizes_km if w > max(SIZES_KM)]
        if wide:
            rec["sizes_beyond_default_km"] = {
                "widths_km": wide,
                "note": (f"wider than the default {max(SIZES_KM):g} km: planted on farmland, such a patch pulls "
                         "the farmland zero and the deramp, so it reads lower than the same movement on a mine "
                         "(which lies outside both); not representative of a mine-sized area, whose limit "
                         "rests on its own farmland null")}
            log.warning("injection: widths %s km exceed the default %g km - %s", wide, max(SIZES_KM),
                        rec["sizes_beyond_default_km"]["note"])
        rec["sites"] = sites
        dy_m, dx_m = sg.pixel_metres(attrs)
        cases, controls = [], []
        for si, st in enumerate(sites):
            up = float(up_map[st["row"], st["col"]]) if up_map is not None and \
                np.isfinite(up_map[st["row"], st["col"]]) else float(up_med)
            for w in sizes_km:
                # the window the planted field is evaluated in (5 sigma)
                half_r = int(np.ceil(5 * w * 1000 / 2.3548 / dy_m)); half_c = int(np.ceil(5 * w * 1000 / 2.3548 / dx_m))
                r0, r1 = max(0, st["row"] - half_r), min(shape[0], st["row"] + half_r + 1)
                c0, c1 = max(0, st["col"] - half_c), min(shape[1], st["col"] + half_c + 1)
                LA, LO = np.meshgrid(lat_c[r0:r1], lon_c[c0:c1], indexing="ij")
                s_win = gaussian(LA, LO, st["lat"], st["lon"], w) * up         # LOS per unit vertical peak
                fp = np.zeros(shape, bool)
                fp[r0:r1, c0:c1] = s_win >= 0.5 * up
                region = fp & good
                # planted vertical mean per unit peak over the cells the
                # decision uses (the reliable half-peak footprint)
                reg_win = region[r0:r1, c0:c1]
                planted_region_mean = (float(s_win[reg_win].mean()) / up if reg_win.any()
                                       else float(s_win[s_win >= 0.5 * up].mean()) / up)
                # the zero-rate control: the same footprint, nothing planted
                controls.append(_control(F, region, {
                    "site": si, "kind": st["kind"], "type": CONTROL, "width_km": w, "rate_mm_yr": 0.0,
                    "planted_region_vertical_mm_yr": 0.0, "planted_region_los_mm_yr": 0.0,
                    "los_up": round(up, 4), "region_cells": int(region.sum())}))
                if not planted_rates:
                    continue
                if D.active:
                    s_mask = gaussian(D.lat, D.lon, st["lat"], st["lon"], w) * up
                    plane_full = D.plane(s_mask, lat_c[:, None], lon_c[None, :])  # broadcast (ny, nx)
                else:
                    plane_full = None
                for kind in types:
                    g = gains[kind]
                    for R in planted_rates:
                        # planted as SINKING: in the product a positive rate
                        # moves toward the satellite (uplift), so subsidence
                        # of R mm/yr is -R (it was planted as uplift until
                        # 7 Oct 2026, which cancelled real mine subsidence).
                        # rate_mm_yr stays the positive size the record names.
                        dv = np.zeros(shape)
                        dv[r0:r1, c0:c1] = -s_win * R * g
                        if plane_full is not None:
                            dv += plane_full * R * g
                        case = _decide(F, dv, region, cases_meta={
                            "site": si, "kind": st["kind"], "type": kind, "width_km": w, "rate_mm_yr": R,
                            "planted_region_vertical_mm_yr": round(-R * planted_region_mean, 2),
                            "planted_region_los_mm_yr": round(-R * planted_region_mean * up, 2),
                            "los_up": round(up, 4), "region_cells": int(region.sum())})
                        cases.append(case)
            log.info("injection: site %d/%d (%s) done", si + 1, len(sites), st["kind"])
        rec["cases"] = cases                      # planted cases only
        rec["control_cases"] = controls           # nothing planted (type 'control', rate 0)
        rec["summary"] = summarise(cases + controls, sizes_km, planted_rates, types, gains)
        rec["aliasing_ceiling"] = _aliasing(cfg, attrs, dates, float(up_med or 0.78))
        rec["technical_summary"] = technical_summary(rec["summary"])
        rec.update({"status": "complete", "wall_s": round(time.time() - t0)})
    except Exception as exc:
        rec.update({"status": "failed", "reason": f"{type(exc).__name__}: {exc}"})
        _write(path, rec)
        raise
    _write(path, rec)
    log.info("Injection-recovery (technical record): %s (%d planted cases, %d controls, %.0f s)",
             rec["technical_summary"], len(rec["cases"]), len(rec["control_cases"]), time.time() - t0)
    return path


def found(s: dict) -> bool | None:
    """The record's "found" (DECISION_RULE) from a significance.assess
    result: distinguishable from farmland AND sinking (excess < 0); None when
    the area could not be tested. Until 7 Oct 2026 it was the two-sided
    distinguishable alone, so a planted SINKING patch reading as uplift
    counted as found."""
    if not s or not s.get("testable", True) or s.get("distinguishable_from_farmland") is None:
        return None
    exc = s.get("excess_mm_yr")
    return bool(s["distinguishable_from_farmland"]) and exc is not None and float(exc) < 0


def _verdict(s: dict, out: dict) -> dict:
    """The decision fields every case carries, from significance.assess."""
    d = s.get("distinguishable_from_farmland")
    exc = s.get("excess_mm_yr")
    out.update({"testable": s.get("testable", True), "detected": found(s),
                "distinguishable": d, "wrong_sign": bool(d) and exc is not None and float(exc) > 0,
                "value_mm_yr": s.get("value_mm_yr"), "excess_mm_yr": exc,
                "threshold_95_mm_yr": s.get("threshold_95_mm_yr")})
    if not out["testable"]:
        out["note"] = s.get("not_testable_reason")
    return out


def _control(F: dict, region: np.ndarray, cases_meta: dict) -> dict:
    """The zero-rate control: the footprint read on the report's own field
    with nothing planted, by the same rule as every planted case."""
    from . import significance as sg
    out = dict(cases_meta)
    if region.sum() < 5:
        out.update({"testable": False, "detected": None, "note": "fewer than 5 reliable cells in the footprint"})
        return out
    return _verdict(sg.assess(F["placement"], F["far_wide"], region) or {}, out)


def _decide(F: dict, dv: np.ndarray, region: np.ndarray, cases_meta: dict) -> dict:
    """The planted velocity through the report-area rule (DECISION_RULE):
    class-matched with the real field, set against the report's farmland
    null over the footprint drawn on the patch."""
    from . import significance as sg
    vel = F["vel"] + dv
    if hasattr(sg, "placement_field"):
        field = sg.placement_field(vel, F["tc"], F["far"], F["far_wide"], F["aoi"])
    else:
        field, _ = sg.class_matched_field(vel, F["tc"], F["far"])
    base = F["placement"]
    out = dict(cases_meta)
    if region.sum() < 5:
        out.update({"testable": False, "detected": None, "note": "fewer than 5 reliable cells in the footprint"})
        return out
    s = sg.assess(field, F["far_wide"], region) or {}
    rec_v = float(np.nanmean(field[region]) - np.nanmean(base[region]))     # what the planting added
    _verdict(s, out)
    out.update({"recovered_los_mm_yr": round(rec_v, 2),
                "recovered_over_planted": (round(rec_v / out["planted_region_los_mm_yr"], 3)
                                           if out["planted_region_los_mm_yr"] else None),
                "chain_attenuation": round(float(np.nanmean(dv[region])) / out["planted_region_los_mm_yr"], 3)
                if out["planted_region_los_mm_yr"] else None})
    return out


def clopper_pearson(k: int, n: int, level: float = 0.95) -> list[float] | None:
    """Exact (Clopper-Pearson) range of a rate k of n: 16 of 20 is 0.563-0.943,
    20 of 20 is 0.832-1.0. None when n is 0."""
    if not n:
        return None
    from scipy.stats import beta
    a = (1.0 - level) / 2.0
    lo = 0.0 if k <= 0 else float(beta.ppf(a, k, n - k + 1))
    hi = 1.0 if k >= n else float(beta.ppf(1.0 - a, k + 1, n - k))
    return [round(lo, 3), round(hi, 3)]


def summarise(cases: list[dict], sizes, rates, types, gains: dict | None = None) -> list[dict]:
    """Detection rate (with its 95% range) and median recovered/planted by
    type, width, rate and site kind (farmland vs inside the mapped mines),
    each planted row beside its control: the same kind and width with
    nothing planted (control_rate, control_ci95 - on farmland the rule's
    false-alarm rate). A FARMLAND row recovered below REPRESENTATIVE_MIN_RATIO
    of planted, once the type's temporal gain (`gains`, the DEM-error and
    velocity-fit loss every area suffers, a mine too) is divided out, is
    flagged: the planting moved the farmland zero or the deramp, which a
    real signal on a mine does not. A row planted on a mine is what a mine
    reads, so it is never flagged."""
    rows = []

    def row(kind, ty, w, R, sel):
        tst = [c for c in sel if c.get("testable") and c.get("detected") is not None]
        det = sum(1 for c in tst if c["detected"])
        return {"kind": kind, "type": ty, "width_km": w, "rate_mm_yr": R, "sites": len(sel),
                "testable": len(tst), "detected": det,
                "detection_rate": round(det / len(tst), 3) if tst else None,
                "detection_ci95": clopper_pearson(det, len(tst)),
                "wrong_sign": sum(1 for c in tst if c.get("wrong_sign")),
                "distinguishable_either_sign": sum(1 for c in tst if c.get("distinguishable")),
                "tst": tst}

    for kind in ("farmland", "mine"):
        for w in sizes:
            sel = [c for c in cases if c["kind"] == kind and c["type"] == CONTROL and c["width_km"] == w]
            ctrl = row(kind, CONTROL, w, 0.0, sel) if sel else None
            if ctrl is not None:
                ctrl.pop("tst")
                ctrl["meaning"] = ("false alarms: farmland with nothing planted read as sinking" if kind == "farmland"
                                   else "mine ground with nothing planted read as sinking (it may really be moving)")
                rows.append(ctrl)
            for ty in types:
                for R in rates:
                    sel = [c for c in cases if c["kind"] == kind and c["type"] == ty and c["width_km"] == w
                           and c["rate_mm_yr"] == R]
                    if not sel:
                        continue
                    r = row(kind, ty, w, R, sel)
                    tst = r.pop("tst")
                    ratio = [c["recovered_over_planted"] for c in tst if c.get("recovered_over_planted") is not None]
                    att = [c["chain_attenuation"] for c in tst if c.get("chain_attenuation") is not None]
                    pv = [c["planted_region_vertical_mm_yr"] for c in tst
                          if c.get("planted_region_vertical_mm_yr") is not None]
                    med_ratio = round(float(np.median(ratio)), 3) if ratio else None
                    g = float((gains or {}).get(ty) or 0.0)
                    farm_ratio = (med_ratio / g if g > 0 else med_ratio) if med_ratio is not None else None
                    r.update({"median_recovered_over_planted": med_ratio,
                              "median_chain_attenuation": round(float(np.median(att)), 3) if att else None,
                              "median_planted_area_mean_vertical_mm_yr": round(float(np.median(pv)), 2) if pv else None,
                              "control_rate": ctrl["detection_rate"] if ctrl else None,
                              "control_ci95": ctrl["detection_ci95"] if ctrl else None,
                              "representative": (kind != "farmland" or farm_ratio is None
                                                 or farm_ratio >= REPRESENTATIVE_MIN_RATIO)})
                    if not r["representative"]:
                        r["note"] = ("recovered below 0.95 of planted beyond the time-fit loss: planted on farmland, "
                                     "so the patch pulls the farmland zero and the deramp; not representative of a "
                                     "mine area")
                    rows.append(r)
    return rows


def technical_summary(summary: list[dict]) -> str:
    """One paragraph for the technical record (never the client page): by
    width, the steady sinking troughs on farmland found of tested, beside the
    control (nothing planted), under the report-area rule."""
    rows = [r for r in summary if r["kind"] == "farmland" and r["type"] in ("trough", CONTROL)]
    if not any(r["testable"] for r in rows):
        return "Technical record: the planted-movement test could not be evaluated on this product."
    parts = []
    for w in sorted({r["width_km"] for r in rows}):
        cells = []
        for r in sorted((r for r in rows if r["width_km"] == w), key=lambda r: r["rate_mm_yr"]):
            what = "nothing planted" if r["type"] == CONTROL else f"{r['rate_mm_yr']:g} mm/yr"
            ci = r.get("detection_ci95")
            rng = f" ({ci[0] * 100:.0f}-{ci[1] * 100:.0f}%)" if ci else ""
            cells.append(f"{what} {r['detected']} of {r['testable']}{rng}")
        flag = "" if all(r.get("representative", True) for r in rows if r["width_km"] == w) else " [not representative]"
        parts.append(f"{w:g} km{flag}: " + ", ".join(cells))
    return ("Technical record, not for the client page. Report-area rule over an area drawn on each planted patch "
            "(sinking only; see decision_rule), steady sinking troughs on the report's farmland, found of tested "
            "with the 95% range, rate at the patch centre (about 0.72 of it averaged over the patch): "
            + "; ".join(parts) + ". The map's search for unmapped movement, phase linking, unwrapping and the "
            "coherence mask are not tested.")


def _write(path: Path, rec: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.part")
    tmp.write_text(json.dumps(rec, indent=1, default=str))
    tmp.replace(path)
    return path


def main(argv=None) -> int:
    import argparse
    from .config import load_config
    ap = argparse.ArgumentParser(description="Injection-recovery test on a finished project's products.")
    ap.add_argument("config")
    ap.add_argument("--out-dir", default=None,
                    help="write injection_recovery.json here instead of <project>/deliverables/technical")
    ap.add_argument("--sites", type=int, default=N_FARMLAND, help="farmland sites (default %(default)s)")
    ap.add_argument("--mine-sites", type=int, default=N_MINE, help="sites inside mapped mines (default %(default)s)")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--sizes", default=",".join(f"{w:g}" for w in SIZES_KM),
                    help="patch widths in km, comma-separated (default %(default)s; wider ones are "
                         "flagged as not representative of a mine-sized area)")
    ap.add_argument("--rates", default=",".join(f"{r:g}" for r in RATES_MM_YR),
                    help="vertical sinking rates in mm/yr, comma-separated (default %(default)s; the "
                         "zero-rate control runs whatever is given)")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = run(load_config(a.config), out_dir=a.out_dir, n_farmland=a.sites, n_mine=a.mine_sites, seed=a.seed,
            sizes_km=tuple(float(x) for x in a.sizes.split(",")),
            rates_mm_yr=tuple(float(x) for x in a.rates.split(",")))
    print(p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
