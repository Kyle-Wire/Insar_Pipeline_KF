"""Refractivity forward operator: atmospheric state -> radar zenith delay.

Phase 1 of research/atmospheric_temperature/ASSESSMENT.md. The pipeline only
ever holds delays (RAiDER's cubes, inputs/<MODEL>.h5); this module holds the
physics that maps a state (pressure P, temperature T, water-vapour pressure e
on height levels) to those delays, so that the question "how much does a
temperature change move the delay?" has a tested, reproducible answer rather
than a hand calculation.

UNITS AND CONVENTIONS
---------------------
  z   metres above the ellipsoid or geoid (whatever the caller's heights are;
      RAiDER's cubes use the height axis of the weather model, see U2 in
      ASSESSMENT.md), ascending
  P,e hPa;  T K;  q kg/kg
  N   refractivity, N-units (n - 1) * 1e6
  delays: metres, ONE-WAY, ZENITH, positive = longer path. The LOS delay is
          the zenith delay / cos(incidence) in the pipeline (projected mode,
          not ray-traced: tropo_disp._config_text).

N = k1 P/T + k2 e/T + k3 e/T^2 with TOTAL pressure in the first term, as
RAiDER writes it (models/weatherModel.py _get_hydro_refractivity /
_get_wet_refractivity; hrrr.py and ecmwf.py: k1 0.776, k2 0.233 K/Pa, k3
3.75e3 K^2/Pa). k1 P/T is "hydrostatic" and the rest "wet", which is the
split RAiDER's `hydro` and `wet` cube variables carry.

THE HYDROSTATIC POINT
---------------------
k1 P/T = k1 R_d rho (ideal gas, dry air; moist air to the virtual-temperature
correction), so the integral of the hydrostatic refractivity over a column
in hydrostatic balance is k1 R_d P_s / g: it depends on the surface pressure
ONLY. A temperature change at fixed surface pressure redistributes the mass
in height but leaves the integral unchanged. column_delay() therefore
rebuilds the pressure profile from the temperature profile (hypsometric
equation) instead of holding P fixed; holding P fixed while changing T
breaks hydrostatic balance and overstates the temperature sensitivity of the
hydrostatic delay (local_jacobian() shows that version, labelled as such).

Nothing here reads data. synthetic_column() is a SYNTHETIC test atmosphere
and every result computed from it says so.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

G0 = 9.80665          # m s-2
R_D = 287.05          # J kg-1 K-1, dry air
R_V = 461.5           # J kg-1 K-1, water vapour
EPS = R_D / R_V       # 0.622
L_V = 2.5e6           # J kg-1, latent heat of vaporisation
TOP_M = 26000.0       # RAiDER's integration top (constants._ZREF)


@dataclass(frozen=True)
class Constants:
    """Refractivity constants for N = k1 P/T + k2 e/T + k3 e/T^2 (P total)."""
    k1: float          # K/hPa
    k2: float          # K/hPa
    k3: float          # K^2/hPa
    source: str


# RAiDER 0.6.0 (HRRR, ECMWF models): the constants the cubes were made with.
K_RAIDER = Constants(77.6, 23.3, 3.75e5, "RAiDER 0.6.0 models/hrrr.py, ecmwf.py")
# Bevis et al. (1994): k1 77.60, k2' 22.1, k3 3.739e5 (total-pressure form).
K_BEVIS94 = Constants(77.60, 22.1, 3.739e5, "Bevis et al. 1994, J. Appl. Meteorol.")


# ---------------------------------------------------------------- moisture
def saturation_vapour_pressure(t_k):
    """hPa over water (Bolton 1980)."""
    tc = np.asarray(t_k, dtype="float64") - 273.15
    return 6.112 * np.exp(17.67 * tc / (tc + 243.5))


def vapour_pressure(q, p_hpa):
    """e (hPa) from specific humidity q (kg/kg) and total pressure: the exact
    relation e = q P / (eps + (1 - eps) q). (RAiDER's _find_e_from_q uses an
    approximation through the saturation pressure; the cross-check in the
    tests therefore feeds e, not q, to both.)"""
    q = np.asarray(q, dtype="float64")
    return q * np.asarray(p_hpa, dtype="float64") / (EPS + (1.0 - EPS) * q)


def specific_humidity(e_hpa, p_hpa):
    e = np.asarray(e_hpa, dtype="float64")
    return EPS * e / (np.asarray(p_hpa, dtype="float64") - (1.0 - EPS) * e)


def virtual_temperature(t_k, e_hpa, p_hpa):
    return np.asarray(t_k, dtype="float64") / (1.0 - np.asarray(e_hpa) / np.asarray(p_hpa) * (1.0 - EPS))


# ---------------------------------------------------------------- refractivity and integration
def refractivity(p_hpa, t_k, e_hpa, k: Constants = K_RAIDER):
    """(N_hydrostatic, N_wet) in N-units."""
    p, t, e = (np.asarray(v, dtype="float64") for v in (p_hpa, t_k, e_hpa))
    return k.k1 * p / t, k.k2 * e / t + k.k3 * e / t ** 2


def integrate_up(z, n):
    """10^-6 * integral of n from each level to the top, trapezoidal, as
    RAiDER's _getZTD does per level. z ascending (m); returns metres per
    level, the last level 0."""
    z = np.asarray(z, dtype="float64")
    n = np.asarray(n, dtype="float64")
    seg = 0.5 * (n[1:] + n[:-1]) * np.diff(z)
    out = np.zeros_like(n)
    out[:-1] = np.cumsum(seg[::-1])[::-1]
    return 1e-6 * out


def hydrostatic_pressure(z, t_k, p_surface_hpa, e_hpa=None):
    """Pressure on the levels z (ascending, z[0] the surface) from the
    hypsometric equation, d ln P / dz = -g / (R_d T_v), trapezoidal in 1/T_v.
    With e given, T_v uses e at the dry-air estimate of P (one pass, enough
    below 1e-4 relative for any terrestrial e)."""
    z = np.asarray(z, dtype="float64")
    t = np.asarray(t_k, dtype="float64")

    def build(tv):
        inv = 1.0 / tv
        dln = -G0 / R_D * 0.5 * (inv[1:] + inv[:-1]) * np.diff(z)
        return p_surface_hpa * np.exp(np.concatenate([[0.0], np.cumsum(dln)]))

    p = build(t)
    if e_hpa is not None:
        p = build(virtual_temperature(t, e_hpa, p))
    return p


def saastamoinen_zhd(p_surface_hpa, lat_deg, h_m):
    """Zenith hydrostatic delay (m) from surface pressure alone (Saastamoinen
    1972, Davis et al. 1985 gravity term)."""
    f = 1.0 - 0.00266 * np.cos(2 * np.radians(lat_deg)) - 0.28e-6 * np.asarray(h_m, dtype="float64")
    return 0.0022768 * np.asarray(p_surface_hpa, dtype="float64") / f


def column_delay(z, t_k, e_hpa, p_surface_hpa, k: Constants = K_RAIDER, hydrostatic: bool = True):
    """(zhd, zwd) in metres at the lowest level z[0], integrated to z[-1].

    hydrostatic=True: the pressure profile is rebuilt from T (and e) with
    P(z[0]) = p_surface_hpa, so a temperature change keeps the column in
    balance. hydrostatic=False: p_surface_hpa must then be the pressure
    PROFILE, held as given (the local, non-balanced sensitivity)."""
    z = np.asarray(z, dtype="float64")
    e = np.asarray(e_hpa, dtype="float64")
    p = hydrostatic_pressure(z, t_k, p_surface_hpa, e) if hydrostatic else np.asarray(p_surface_hpa, dtype="float64")
    nh, nw = refractivity(p, t_k, e, k)
    return float(integrate_up(z, nh)[0]), float(integrate_up(z, nw)[0])


# ---------------------------------------------------------------- sensitivities
def local_jacobian(p_hpa, t_k, e_hpa, k: Constants = K_RAIDER) -> dict:
    """Analytic partial derivatives of N at fixed P (NOT hydrostatically
    balanced) and, for the wet term, at fixed relative humidity (Clausius-
    Clapeyron, d ln e_s/dT = L_v / (R_v T^2)). N-units per K / per hPa."""
    p, t, e = (np.asarray(v, dtype="float64") for v in (p_hpa, t_k, e_hpa))
    dnh_dt = -k.k1 * p / t ** 2
    dnw_dt_e = -k.k2 * e / t ** 2 - 2 * k.k3 * e / t ** 3
    dnw_de = k.k2 / t + k.k3 / t ** 2
    dlne_dt = L_V / (R_V * t ** 2)
    return {"dNh_dT_fixedP": dnh_dt, "dNw_dT_fixed_e": dnw_dt_e, "dNw_de": dnw_de,
            "dNw_dT_fixed_RH": dnw_dt_e + dnw_de * e * dlne_dt, "dlne_s_dT": dlne_dt}


def layer_anomaly_response(z, t_k, e_hpa, p_surface_hpa, depth_m: float, dt_k: float = 1.0,
                           hold: str = "e", k: Constants = K_RAIDER) -> dict:
    """Change in the surface zenith delay (m) when the lowest depth_m of the
    column warms by dt_k, the surface pressure held fixed and the column
    rebalanced. hold: 'e' (vapour pressure unchanged) or 'rh' (relative
    humidity unchanged, e scales with e_s). Returns the hydrostatic, wet and
    total changes, plus the same with P held fixed (unbalanced) for contrast."""
    z = np.asarray(z, dtype="float64")
    t0 = np.asarray(t_k, dtype="float64")
    e0 = np.asarray(e_hpa, dtype="float64")
    inside = (z - z[0]) <= depth_m
    t1 = t0 + np.where(inside, dt_k, 0.0)
    if hold == "e":
        e1 = e0
    elif hold == "rh":
        e1 = e0 * saturation_vapour_pressure(t1) / saturation_vapour_pressure(t0)
    else:
        raise ValueError("hold must be 'e' or 'rh'")
    h0, w0 = column_delay(z, t0, e0, p_surface_hpa, k)
    h1, w1 = column_delay(z, t1, e1, p_surface_hpa, k)
    p_fixed = hydrostatic_pressure(z, t0, p_surface_hpa, e0)
    hf0, wf0 = column_delay(z, t0, e0, p_fixed, k, hydrostatic=False)
    hf1, wf1 = column_delay(z, t1, e1, p_fixed, k, hydrostatic=False)
    return {"depth_m": depth_m, "dT_K": dt_k, "hold": hold,
            "d_zhd_m": h1 - h0, "d_zwd_m": w1 - w0, "d_ztd_m": (h1 + w1) - (h0 + w0),
            "fixedP_d_zhd_m": hf1 - hf0, "fixedP_d_ztd_m": (hf1 + wf1) - (hf0 + wf0)}


def humidity_anomaly_response(z, t_k, e_hpa, p_surface_hpa, depth_m: float, de_hpa: float = 1.0,
                              k: Constants = K_RAIDER) -> dict:
    """Change in the surface zenith delay (m) when e rises by de_hpa in the
    lowest depth_m, temperature and surface pressure unchanged."""
    z = np.asarray(z, dtype="float64")
    e0 = np.asarray(e_hpa, dtype="float64")
    e1 = e0 + np.where((z - z[0]) <= depth_m, de_hpa, 0.0)
    h0, w0 = column_delay(z, t_k, e0, p_surface_hpa, k)
    h1, w1 = column_delay(z, t_k, e1, p_surface_hpa, k)
    return {"depth_m": depth_m, "de_hPa": de_hpa, "d_zhd_m": h1 - h0, "d_zwd_m": w1 - w0,
            "d_ztd_m": (h1 + w1) - (h0 + w0)}


def relief_response(z, t_k, e_hpa, p_surface_hpa, z_low: float, z_high: float, dt_k: float = 1.0,
                    hold: str = "e", k: Constants = K_RAIDER) -> dict:
    """Change in the DIFFERENCE of zenith delay between two pixels at heights
    z_low and z_high (z_low = z[0], the column's surface) when the layer
    between them warms by dt_k, the pressure at z_low held fixed. This is the
    hypsometric route by which layer temperature reaches a stratified,
    elevation-correlated delay; at one pixel it is invisible."""
    z = np.asarray(z, dtype="float64")
    if abs(z[0] - z_low) > 1e-6:
        raise ValueError("z_low must be the column's lowest level z[0]")
    t0 = np.asarray(t_k, dtype="float64")
    e0 = np.asarray(e_hpa, dtype="float64")
    hi = int(np.argmin(np.abs(z - z_high)))
    t1 = t0 + np.where(z <= z[hi], dt_k, 0.0)
    e1 = e0 if hold == "e" else e0 * saturation_vapour_pressure(t1) / saturation_vapour_pressure(t0)

    def diff(t, e):
        p = hydrostatic_pressure(z, t, p_surface_hpa, e)
        nh, nw = refractivity(p, t, e, k)
        dh, dw = integrate_up(z, nh), integrate_up(z, nw)
        return dh[0] - dh[hi], dw[0] - dw[hi]

    h0, w0 = diff(t0, e0)
    h1, w1 = diff(t1, e1)
    return {"z_low": float(z[0]), "z_high": float(z[hi]), "dT_K": dt_k, "hold": hold,
            "delta_zhd_m": h0, "delta_zwd_m": w0,
            "d_delta_zhd_m": h1 - h0, "d_delta_zwd_m": w1 - w0, "d_delta_ztd_m": (h1 + w1) - (h0 + w0)}


# ---------------------------------------------------------------- a SYNTHETIC column
def synthetic_column(z_surface: float = 0.0, p_surface: float = 1000.0, t_surface: float = 293.15,
                     rh_surface: float = 0.6, lapse_k_per_km: float = 6.5, tropopause_m: float = 11000.0,
                     moisture_scale_m: float = 2000.0, top_m: float = TOP_M, step_m: float = 15.0) -> dict:
    """A SYNTHETIC atmosphere for tests and sensitivity numbers: a constant
    lapse rate to an isothermal tropopause, relative humidity decaying with
    height (e-folding moisture_scale_m), hydrostatic pressure. step_m 15 m is
    RAiDER's integration step (constants._STEP). Not an observation."""
    z = np.arange(z_surface, top_m + step_m / 2, step_m, dtype="float64")
    zz = z - z_surface
    trop = max(tropopause_m - z_surface, 0.0)
    t = t_surface - lapse_k_per_km * 1e-3 * np.minimum(zz, trop)
    rh = rh_surface * np.exp(-zz / moisture_scale_m)
    e = rh * saturation_vapour_pressure(t)
    p = hydrostatic_pressure(z, t, p_surface, e)
    return {"z": z, "t": t, "e": e, "p": p, "rh": rh, "synthetic": True,
            "label": f"SYNTHETIC column: surface {z_surface:g} m, {p_surface:g} hPa, {t_surface:g} K, "
                     f"RH {rh_surface:g}, lapse {lapse_k_per_km:g} K/km"}


def sensitivity_table(col: dict | None = None, k: Constants = K_RAIDER) -> dict:
    """The numbers of ASSESSMENT.md section 4.1, computed: the delay response
    to a 1 K anomaly in the lowest 100 m (fixed e, fixed RH, fixed P for
    contrast), to 1 hPa of e in the same layer, and the hypsometric relief
    response per K. On a SYNTHETIC column unless one is given."""
    col = col or synthetic_column()
    z, t, e, ps = col["z"], col["t"], col["e"], float(col["p"][0])
    out = {"column": col.get("label", "given column"), "synthetic": bool(col.get("synthetic")),
           "constants": k.source}
    for hold in ("e", "rh"):
        r = layer_anomaly_response(z, t, e, ps, 100.0, 1.0, hold, k)
        out[f"1K_100m_fixed_{hold}_mm"] = {"zhd": r["d_zhd_m"] * 1e3, "zwd": r["d_zwd_m"] * 1e3,
                                           "ztd": r["d_ztd_m"] * 1e3, "fixedP_zhd": r["fixedP_d_zhd_m"] * 1e3}
    h = humidity_anomaly_response(z, t, e, ps, 100.0, 1.0, k)
    out["1hPa_e_100m_mm"] = h["d_ztd_m"] * 1e3
    out["relief_per_K_mm"] = {}
    for dz in (50, 100, 500, 1000):
        r = relief_response(z, t, e, ps, z[0], z[0] + dz, 1.0, "e", k)
        # keyed by the relief actually used: the nearest level to z[0] + dz
        out["relief_per_K_mm"][round(r["z_high"] - r["z_low"])] = {
            "delta_zhd_mm": r["delta_zhd_m"] * 1e3, "per_K_zhd_mm": r["d_delta_zhd_m"] * 1e3,
            "per_K_ztd_mm": r["d_delta_ztd_m"] * 1e3}
    return out
