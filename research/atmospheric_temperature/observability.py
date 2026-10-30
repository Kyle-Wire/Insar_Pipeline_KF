"""How much of the radar residual could be temperature? Measured noise
against the forward operator's sensitivities.

Phase 1 of research/atmospheric_temperature/ASSESSMENT.md (gate G1: a written
detectability number). Works on radar_obs.h5 (radar_obs.py) and nothing else
from the project except one RAiDER cube for the site pressure.

DEFORMATION
-----------
The residual (observation - model) still holds ground motion. Every measure
here is taken on SHORT-INTERVAL DIFFERENCES, residual(t_i) - residual(t_i-1)
for consecutive dates at most max_gap_days apart: slow motion and the
reference date's own atmosphere cancel, and what is left is the difference
of two dates' unmodelled delay plus noise. Two independent dates add in
quadrature, so per-date figures are the difference figures / sqrt(2) (the
"per_date" fields); that is an assumption (independent dates), recorded as such.

THE TWO ROUTES TO TEMPERATURE (ASSESSMENT.md 4.1)
-------------------------------------------------
  pixel scale   two pixels ~80 m apart share everything above the surface
                layer, so only a surface-layer anomaly that DIFFERS between
                them moves their delay difference: the per-K response of a
                100 m layer (fixed e, fixed RH), against the measured RMS
                difference at the shortest lag.
  stratified    the slope of the residual against height, per date: a
                layer-mean temperature anomaly between the lowest and highest
                ground changes it through the hypsometric equation (+ the wet
                term), against the slope's own precision (spatial-block
                bootstrap) and its spread from date to date.
Both humidity and temperature move both; the moisture equivalent of every
number is reported beside the temperature one.
"""
from __future__ import annotations

import json
import logging
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np

from . import refractivity as rf

log = logging.getLogger(__name__)


# ---------------------------------------------------------------- grid spacing
def pixel_spacing_m(lat: np.ndarray, lon: np.ndarray, valid: np.ndarray) -> tuple[float, float]:
    """(metres per row, metres per column): the median distance between
    valid neighbours on a local tangent plane."""
    R = 6371008.8
    la, lo = np.radians(lat), np.radians(lon)

    def step(a0, a1, o0, o1, v):
        d = R * np.hypot(a1 - a0, (o1 - o0) * np.cos(0.5 * (a0 + a1)))
        return float(np.median(d[v]))

    vr = valid[1:, :] & valid[:-1, :]
    vc = valid[:, 1:] & valid[:, :-1]
    return (step(la[:-1, :], la[1:, :], lo[:-1, :], lo[1:, :], vr),
            step(la[:, :-1], la[:, 1:], lo[:, :-1], lo[:, 1:], vc))


# ---------------------------------------------------------------- structure function
def offsets(max_px: int, n: int = 24) -> list[tuple[int, int]]:
    ks = sorted(set(np.unique(np.round(np.geomspace(1, max(max_px, 1), n)).astype(int)).tolist()))
    out = []
    for k in ks:
        out += [(0, k), (k, 0), (k, k), (k, -k)]
    return out


def structure_function(field: np.ndarray, mask: np.ndarray, sy: float, sx: float,
                       lag_edges_m: np.ndarray, max_px: int | None = None) -> dict:
    """RMS of f(x+r) - f(x) (field units) in lag bins, from shifted copies of
    the grid along rows, columns and both diagonals. Deterministic; every
    valid pair at those offsets counts."""
    f = np.where(mask, field, np.nan)
    ny, nx = f.shape
    max_px = max_px or int(lag_edges_m[-1] / min(sy, sx)) + 1
    sums = np.zeros(len(lag_edges_m) - 1)
    counts = np.zeros(len(lag_edges_m) - 1)
    for dy, dx in offsets(min(max_px, max(ny, nx) - 1)):
        if dy >= ny or abs(dx) >= nx:
            continue
        d = np.hypot(dy * sy, dx * sx)
        b = np.searchsorted(lag_edges_m, d) - 1
        if b < 0 or b >= len(counts):
            continue
        a = f[dy:, max(dx, 0):nx + min(dx, 0)]
        c = f[:ny - dy, max(-dx, 0):nx - max(dx, 0)]
        diff = a - c
        ok = np.isfinite(diff)
        sums[b] += float(np.sum(diff[ok] ** 2))
        counts[b] += int(ok.sum())
    with np.errstate(invalid="ignore", divide="ignore"):
        rms = np.sqrt(sums / counts)
    centres = np.sqrt(lag_edges_m[:-1] * lag_edges_m[1:])
    return {"lag_m": centres.tolist(), "rms": rms.tolist(), "pairs": counts.astype(int).tolist()}


# ---------------------------------------------------------------- stratified slope
def block_ids(shape: tuple[int, int], sy: float, sx: float, block_m: float) -> np.ndarray:
    by, bx = max(1, int(round(block_m / sy))), max(1, int(round(block_m / sx)))
    r, c = np.mgrid[0:shape[0], 0:shape[1]]
    nbx = (shape[1] + bx - 1) // bx
    return (r // by) * nbx + (c // bx)


def height_slope(field: np.ndarray, height: np.ndarray, mask: np.ndarray, blocks: np.ndarray,
                 n_boot: int = 200, rng: np.random.Generator | None = None) -> dict:
    """OLS slope of field against height (field units per metre) with a
    spatial-block bootstrap standard error: the blocks, not the pixels, are
    resampled, because the residual is spatially correlated well beyond a
    pixel (a pixel bootstrap would understate the error many times over)."""
    rng = rng or np.random.default_rng(0)
    ok = mask & np.isfinite(field) & np.isfinite(height)
    if ok.sum() < 50:
        return {"slope": np.nan, "se_boot": np.nan, "n": int(ok.sum()), "blocks": 0}
    h, y, b = height[ok], field[ok], blocks[ok]
    ub, inv = np.unique(b, return_inverse=True)
    S = np.zeros((len(ub), 5))
    np.add.at(S, inv, np.column_stack([np.ones_like(h), h, y, h * h, h * y]))

    def fit(s):
        n, sh, sy_, shh, shy = s[..., 0], s[..., 1], s[..., 2], s[..., 3], s[..., 4]
        den = n * shh - sh * sh
        with np.errstate(invalid="ignore", divide="ignore"):
            return (n * shy - sh * sy_) / den

    slope = float(fit(S.sum(0)))
    cnt = rng.multinomial(len(ub), np.full(len(ub), 1.0 / len(ub)), size=n_boot)
    boots = fit(cnt @ S)
    return {"slope": slope, "se_boot": float(np.nanstd(boots)), "n": int(ok.sum()), "blocks": int(len(ub))}


def _lagwise(fn, a, *args):
    """fn over axis 0 of a pairs x lags array. Lag bins no offset reaches are
    NaN in every pair; their result stays NaN, without numpy's warning."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return fn(a, *args, axis=0)


# ---------------------------------------------------------------- pairs of dates
def consecutive_pairs(dates: list[str], max_gap_days: int = 24) -> list[tuple[int, int]]:
    t = [datetime.strptime(d, "%Y%m%d") for d in dates]
    return [(i - 1, i) for i in range(1, len(t)) if (t[i] - t[i - 1]).days <= max_gap_days]


# ---------------------------------------------------------------- site conditions
def site_pressure_hpa(cube_path: Path, lat: float, lon: float, height_m: float) -> float:
    """Surface pressure at the site implied by RAiDER's hydrostatic cube
    (Saastamoinen inverted): a model value, labelled as such where used."""
    from .radar_obs import sample_cube_components
    hyd, _ = sample_cube_components(cube_path, np.array([lon]), np.array([lat]), np.array([height_m]))
    f = 1.0 - 0.00266 * np.cos(2 * np.radians(lat)) - 0.28e-6 * height_m
    return float(hyd[0] * f / 0.0022768)


def sensitivities(z_low: float, z_high: float, p_hpa: float, cos_inc: float,
                  t_surface: tuple[float, ...] = (268.15, 293.15), rh: float = 0.6) -> dict:
    """LOS (mm) responses at the site from the forward operator, on SYNTHETIC
    columns started at the site's lowest ground and pressure, for a cold and a
    warm surface temperature (the operator's T dependence is weak; the pair
    brackets it)."""
    out = {"synthetic_columns": True, "z_low": z_low, "z_high": z_high, "p_surface_hPa": p_hpa,
           "cos_incidence": cos_inc, "cases": []}
    for ts in t_surface:
        col = rf.synthetic_column(z_surface=z_low, p_surface=p_hpa, t_surface=ts, rh_surface=rh)
        z, t, e = col["z"], col["t"], col["e"]
        layer_e = rf.layer_anomaly_response(z, t, e, p_hpa, 100.0, 1.0, "e")
        layer_rh = rf.layer_anomaly_response(z, t, e, p_hpa, 100.0, 1.0, "rh")
        hum = rf.humidity_anomaly_response(z, t, e, p_hpa, 100.0, 1.0)
        relief = max(z_high - z_low, 1.0)
        rel_e = rf.relief_response(z, t, e, p_hpa, z[0], z_high, 1.0, "e")
        rel_rh = rf.relief_response(z, t, e, p_hpa, z[0], z_high, 1.0, "rh")
        hum_rel = rf.humidity_anomaly_response(z, t, e, p_hpa, relief, 1.0)
        los = 1e3 / cos_inc
        out["cases"].append({
            "t_surface_K": ts, "label": col["label"],
            "pixel_1K_100m_fixed_e_mm": abs(layer_e["d_ztd_m"]) * los,
            "pixel_1K_100m_fixed_rh_mm": abs(layer_rh["d_ztd_m"]) * los,
            "pixel_1hPa_e_100m_mm": abs(hum["d_ztd_m"]) * los,
            "slope_1K_layer_fixed_e_mm_per_km": abs(rel_e["d_delta_ztd_m"]) * los / (relief / 1e3),
            "slope_1K_layer_fixed_rh_mm_per_km": abs(rel_rh["d_delta_ztd_m"]) * los / (relief / 1e3),
            "slope_1hPa_e_layer_mm_per_km": abs(hum_rel["d_ztd_m"]) * los / (relief / 1e3),
        })
    return out


# ---------------------------------------------------------------- the whole measure
def assess(obs_dir: Path, project_dir: Path, max_gap_days: int = 24, block_m: float = 2000.0, target_lag_m: float = 80.0,
           lag_edges_m: np.ndarray | None = None) -> dict:
    """observability.json for one exported project (module docstring)."""
    import h5py
    from .radar_obs import cube_for_date, geometry
    obs_dir, project_dir = Path(obs_dir), Path(project_dir)
    rec = json.loads((obs_dir / "radar_obs.json").read_text())
    geom = geometry(project_dir / "mintpy")
    mask_file = project_dir / "mintpy" / "maskTempCoh.h5"
    with h5py.File(mask_file, "r") as m:
        mask = m["mask"][()].astype(bool) & geom["valid"]
    with h5py.File(obs_dir / "radar_obs.h5", "r") as f:
        dates = [d.decode() for d in f["date"][()]]
        res = f["residual"][()].astype("float64")
        mod = (f["model_hydro"][()] + f["model_wet"][()]).astype("float64")
    sy, sx = pixel_spacing_m(geom["lat"], geom["lon"], geom["valid"])
    if lag_edges_m is None:
        lag_edges_m = np.geomspace(0.5 * min(sy, sx), 12000.0, 16)
    pairs = consecutive_pairs(dates, max_gap_days)
    blocks = block_ids(res.shape[1:], sy, sx, block_m)
    hgt = geom["height"]
    h_lo, h_hi = (float(v) for v in np.nanpercentile(hgt[mask], [2, 98]))

    sf_res, sf_mod, slopes, slopes_mod = [], [], [], []
    rng = np.random.default_rng(0)
    for i0, i1 in pairs:
        d = res[i1] - res[i0]
        dm = mod[i1] - mod[i0]
        sf_res.append(structure_function(d, mask, sy, sx, lag_edges_m)["rms"])
        sf_mod.append(structure_function(dm, mask, sy, sx, lag_edges_m)["rms"])
        s = height_slope(d, hgt, mask, blocks, rng=rng)
        slopes.append((s["slope"], s["se_boot"]))
        slopes_mod.append(height_slope(dm, hgt, mask, blocks, n_boot=2, rng=rng)["slope"])
    sf_res, sf_mod = np.array(sf_res) * 1e3, np.array(sf_mod) * 1e3       # mm
    lags = structure_function(res[0], mask, sy, sx, lag_edges_m)["lag_m"]
    sl = np.array(slopes) * 1e6                                            # mm per km
    slm = np.array(slopes_mod) * 1e6
    r2 = np.sqrt(2.0)

    cube = cube_for_date(project_dir / "tropo_corrections", rec["model"], dates[0])
    lat0, lon0 = float(np.median(geom["lat"][mask])), float(np.median(geom["lon"][mask]))
    p_site = site_pressure_hpa(cube, lat0, lon0, h_lo)
    cos_inc = float(np.cos(np.radians(np.median(geom["incidence"][mask]))))
    sens = sensitivities(h_lo, h_hi, p_site, cos_inc)

    med = _lagwise(np.nanmedian, sf_res)
    have = np.isfinite(med)
    shortest = int(np.nanargmin(np.where(have, lags, np.inf)))
    near = int(np.nanargmin(np.where(have, np.abs(np.log(np.array(lags) / target_lag_m)), np.inf)))
    slope_se_date = float(np.nanmedian(sl[:, 1])) / r2
    slope_spread_date = float(1.4826 * np.nanmedian(np.abs(sl[:, 0] - np.nanmedian(sl[:, 0])))) / r2
    # most favourable (largest) response over the two synthetic surface temperatures, per closure
    worst = {k: max(c[k] for c in sens["cases"]) for k in sens["cases"][0] if k.endswith(("_mm", "_km"))}
    best_case_pixel = max(worst["pixel_1K_100m_fixed_e_mm"], worst["pixel_1K_100m_fixed_rh_mm"])
    best_case_slope = max(worst["slope_1K_layer_fixed_e_mm_per_km"], worst["slope_1K_layer_fixed_rh_mm_per_km"])

    def pixel(i):
        rms = float(np.nanmedian(sf_res[:, i])) / r2
        return {"lag_m": lags[i], "rms_mm": rms,
                "min_dT_K_2sigma_fixed_e": 2 * rms / worst["pixel_1K_100m_fixed_e_mm"],
                "min_dT_K_2sigma_fixed_rh": 2 * rms / worst["pixel_1K_100m_fixed_rh_mm"],
                "equivalent_e_hPa_of_1sigma": rms / worst["pixel_1hPa_e_100m_mm"]}

    # does the residual stratification follow the model's own? slope = gain * model slope + scatter
    okp = np.isfinite(sl[:, 0]) & np.isfinite(slm)
    gain = float(np.polyfit(slm[okp], sl[okp, 0], 1)[0]) if okp.sum() > 3 else float("nan")
    corr = float(np.corrcoef(slm[okp], sl[okp, 0])[0, 1]) if okp.sum() > 3 else float("nan")
    months = np.array([int(dates[b][4:6]) for _a, b in pairs])
    by_month = {int(mo): {"pairs": int((months == mo).sum()),
                          "median_abs_residual_slope": float(np.nanmedian(np.abs(sl[months == mo, 0])))}
                for mo in sorted(set(months.tolist()))}
    out = {
        "project": str(project_dir), "obs_dir": str(obs_dir), "model": rec["model"],
        "dates": len(dates), "pairs_used": len(pairs), "max_gap_days": max_gap_days,
        "assumption_independent_dates": "per-date = pair figure / sqrt(2)",
        "grid_m": {"row": sy, "col": sx}, "mask": f"{mask_file.name} & valid geometry: {int(mask.sum())} pixels",
        "height_m": {"p2": h_lo, "p98": h_hi, "relief": h_hi - h_lo},
        "site": {"lat": round(lat0, 2), "lon": round(lon0, 2), "pressure_hPa_from_model_cube": p_site,
                 "pressure_source": f"{cube.name} hydro at {h_lo:.0f} m, Saastamoinen inverted"},
        "structure_function_mm": {"lag_m": lags, "residual_pair_median": _lagwise(np.nanmedian, sf_res).tolist(),
                                  "residual_pair_p90": _lagwise(np.nanpercentile, sf_res, 90).tolist(),
                                  "model_pair_median": _lagwise(np.nanmedian, sf_mod).tolist()},
        "stratified_mm_per_km": {"residual_pair_slope_median_abs": float(np.nanmedian(np.abs(sl[:, 0]))),
                                 "residual_pair_slope_robust_spread": slope_spread_date * r2,
                                 "residual_pair_slope_se_boot_median": slope_se_date * r2,
                                 "model_pair_slope_robust_spread": float(1.4826 * np.nanmedian(
                                     np.abs(slm - np.nanmedian(slm)))),
                                 "residual_vs_model_slope_gain": gain, "residual_vs_model_slope_corr": corr,
                                 "by_month_of_later_date": by_month,
                                 "block_m": block_m},
        "sensitivity": sens,
        "detectability_per_date": {
            "pixel_shortest_lag": pixel(shortest),
            "pixel_near_target": {**pixel(near), "target_lag_m": target_lag_m},
            "pixel_note": "two pixels this far apart share the column above the surface layer; only a "
                          "surface-layer anomaly that differs between them moves their difference",
            "slope_se_mm_per_km": slope_se_date,
            "slope_min_dT_K_2sigma_best_case": 2 * slope_se_date / best_case_slope,
            "slope_spread_mm_per_km": slope_spread_date,
            "slope_spread_as_K_if_all_temperature": slope_spread_date / best_case_slope,
            "slope_spread_as_e_hPa_if_all_moisture": slope_spread_date / worst["slope_1hPa_e_layer_mm_per_km"],
            "best_case_means": "the largest response over fixed-e / fixed-RH and the two synthetic surface "
                               "temperatures: the most favourable sensitivity, so the smallest dT",
            "pixel_best_case_dT_K_2sigma_near_target": 2 * pixel(near)["rms_mm"] / best_case_pixel,
        },
        "pairs": [{"d0": dates[a], "d1": dates[b], "slope_mm_per_km": float(s[0]), "se": float(s[1]),
                   "model_slope_mm_per_km": float(m)} for (a, b), s, m in zip(pairs, sl, slm)],
    }
    (obs_dir / "observability.json").write_text(json.dumps(out, indent=2, default=float))
    return out
