"""Atmospheric-temperature research, Phase 1 (research/atmospheric_temperature):
the refractivity forward operator, the read-only radar-observation exporter
and the observability measures. Every array here is SYNTHETIC; no real
project is read."""
import json

import h5py
import numpy as np
import pytest

from insar_pipeline.research.atmospheric_temperature import observability as ob
from insar_pipeline.research.atmospheric_temperature import radar_obs as ro
from insar_pipeline.research.atmospheric_temperature import raider_crosscheck as rx
from insar_pipeline.research.atmospheric_temperature import refractivity as rf


# ---------------------------------------------------------------- refractivity
@pytest.fixture(scope="module")
def col():
    return rf.synthetic_column()


def test_integrate_up_constant_refractivity():
    z = np.linspace(0, 1000, 101)
    d = rf.integrate_up(z, np.full_like(z, 300.0))
    assert d[0] == pytest.approx(300e-6 * 1000)
    assert d[-1] == 0 and d[50] == pytest.approx(300e-6 * 500)


def test_hydrostatic_integral_is_saastamoinen(col):
    zhd, _ = rf.column_delay(col["z"], col["t"], col["e"], col["p"][0])
    # Saastamoinen covers the whole column; the integration stops at 26 km
    want = rf.saastamoinen_zhd(col["p"][0], 45, 0) - rf.saastamoinen_zhd(col["p"][-1], 45, rf.TOP_M)
    assert zhd == pytest.approx(want, rel=5e-3)


def test_temperature_at_fixed_surface_pressure_barely_moves_the_hydrostatic_delay(col):
    r = rf.layer_anomaly_response(col["z"], col["t"], col["e"], col["p"][0], 100.0, 1.0, "e")
    # balanced: ~0 (only the moist-air term remains); unbalanced (P held): k1 P/T^2 * 100 m
    assert abs(r["d_zhd_m"]) < 0.05 * abs(r["fixedP_d_zhd_m"])
    assert r["fixedP_d_zhd_m"] == pytest.approx(-77.6 * col["p"][0] / col["t"][0] ** 2 * 100e-6, rel=0.05)


def test_humidity_dominates_temperature(col):
    z, t, e, ps = col["z"], col["t"], col["e"], col["p"][0]
    fixed_e = rf.layer_anomaly_response(z, t, e, ps, 100.0, 1.0, "e")["d_ztd_m"]
    fixed_rh = rf.layer_anomaly_response(z, t, e, ps, 100.0, 1.0, "rh")["d_ztd_m"]
    one_hpa = rf.humidity_anomaly_response(z, t, e, ps, 100.0, 1.0)["d_ztd_m"]
    jac = rf.local_jacobian(col["p"][0], t[0], e[0])
    assert one_hpa == pytest.approx(jac["dNw_de"] * 100e-6, rel=0.05)
    assert fixed_e < 0 < fixed_rh
    assert one_hpa > 5 * abs(fixed_e) and one_hpa > fixed_rh


def test_relief_response_is_the_hypsometric_term(col):
    z, t, e, ps = col["z"], col["t"], col["e"], col["p"][0]
    r = rf.relief_response(z, t, e, ps, z[0], 1000.0, 1.0, "e")
    # d(delta ZHD)/dT ~ -delta ZHD / T_mean for the layer
    tmean = float(np.mean(t[z <= 1000]))
    assert r["d_delta_zhd_m"] == pytest.approx(-r["delta_zhd_m"] / tmean, rel=0.1)


def test_sensitivity_table_is_labelled_synthetic():
    t = rf.sensitivity_table()
    assert t["synthetic"] and "SYNTHETIC" in t["column"]
    assert 990 <= max(t["relief_per_K_mm"]) <= 1010


def test_matches_raider_own_code():
    py = rx.default_python()
    if not py.exists():
        pytest.skip("raider-env is not installed")
    c = rf.synthetic_column(z_surface=1850.0, p_surface=810.0, t_surface=268.0)
    r = rx.run(c, py)
    nh, nw = rf.refractivity(c["p"], c["t"], c["e"])
    h, w = rf.integrate_up(c["z"], nh), rf.integrate_up(c["z"], nw)
    np.testing.assert_allclose(np.array(r["hydro"])[:-1], h[:-1], rtol=1e-9)
    np.testing.assert_allclose(np.array(r["wet"])[:-1], w[:-1], rtol=1e-9)
    assert r["k_per_Pa"] == pytest.approx([0.776, 0.233, 3750.0])


# ---------------------------------------------------------------- a SYNTHETIC MintPy project
DATES = ["20200101", "20200113", "20200125", "20200206"]
NY, NX = 30, 40


def _cube(path, scale_wet, scale_hyd):
    import xarray as xr
    z = np.array([-500.0, 0.0, 500.0, 1000.0, 2000.0, 5000.0])
    y = np.linspace(39.2, 38.6, 7)            # north to south, as RAiDER writes
    x = np.linspace(-105.2, -104.4, 9)
    Z, Y, X = np.meshgrid(z, y, x, indexing="ij")
    hydro = scale_hyd * 2.3 * np.exp(-Z / 8000.0) * (1 + 0.001 * (X + 105))
    wet = scale_wet * 0.1 * np.exp(-Z / 2000.0) * (1 + 0.05 * (Y - 38.6))
    ds = xr.Dataset({"hydro": (("z", "y", "x"), hydro), "wet": (("z", "y", "x"), wet)},
                    coords={"z": z, "y": y, "x": x})
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(path)


@pytest.fixture
def project(tmp_path):
    P = tmp_path / "proj"
    M = P / "mintpy"
    (M / "inputs").mkdir(parents=True)
    lat, lon = np.meshgrid(np.linspace(39.0, 38.8, NY), np.linspace(-105.0, -104.6, NX), indexing="ij")
    hgt = 1800.0 + 600.0 * (lat - 38.8) / 0.2
    inc = np.full((NY, NX), 37.0)
    with h5py.File(M / "inputs" / "geometryRadar.h5", "w") as g:
        for k, v in (("latitude", lat), ("longitude", lon), ("height", hgt), ("incidenceAngle", inc)):
            g[k] = v.astype("float32")
    for i, d in enumerate(DATES):
        _cube(P / "tropo_corrections" / "HRRR" / d / f"HRRR_tropo_{d}T000000_ztd.nc", 1 + 0.3 * i, 1 + 0.001 * i)
    rng = np.random.default_rng(1)
    truth_motion = np.zeros((len(DATES), NY, NX))
    truth_motion[:, 10:20, 10:20] = np.arange(len(DATES))[:, None, None] * -0.002
    # the "true" model the run applied, built through the exporter's own sampler
    zen = np.array([np.sum(ro.sample_cube_components(
        P / "tropo_corrections" / "HRRR" / d / f"HRRR_tropo_{d}T000000_ztd.nc", lon, lat, hgt), axis=0) for d in DATES])
    slant = -zen / np.cos(np.radians(inc))
    ref_y, ref_x = 2, 3
    applied = ro._reference(slant, 0, ref_y, ref_x)
    noise = rng.normal(0, 0.001, truth_motion.shape)
    noise -= noise[0]
    noise -= noise[:, ref_y:ref_y + 1, ref_x:ref_x + 1]
    obs = truth_motion + applied + noise
    attrs = {"REF_DATE": DATES[0], "REF_Y": str(ref_y), "REF_X": str(ref_x), "LENGTH": str(NY), "WIDTH": str(NX),
             "UNIT": "m", "FILE_TYPE": "timeseries", "WAVELENGTH": "0.0555"}
    for name, arr in (("timeseries.h5", obs), ("timeseries_HRRR.h5", obs - applied)):
        with h5py.File(M / name, "w") as f:
            f.attrs.update(attrs)
            f["date"] = np.array([d.encode() for d in DATES])
            f["timeseries"] = arr.astype("float32")
    with h5py.File(M / "maskTempCoh.h5", "w") as m:
        m["mask"] = np.ones((NY, NX), dtype=bool)
    (M / "smallbaselineApp.cfg").write_text("mintpy.deramp = no\nmintpy.topographicResidual = yes\n"
                                            "mintpy.troposphericDelay.method = pyaps  # comment\n")
    return {"dir": P, "truth_motion": truth_motion, "noise": noise}


def test_export_reproduces_the_applied_correction(project, tmp_path):
    out = tmp_path / "out"
    rec = ro.export(project["dir"], out)
    assert rec["check_against_applied_correction"]["passed"]
    assert rec["check_against_applied_correction"]["rms_mm"] < 0.1
    assert rec["check_against_applied_correction"]["pixel_dates_over_0.1mm"] == 0
    with h5py.File(out / "radar_obs.h5", "r") as f:
        assert {f[k].attrs["kind"] for k in ("model_hydro", "model_wet", "residual")} == {"model", "residual"}
        res = f["residual"][()]
        # residual = what is not the model: the motion plus the noise (float32 rounding)
        np.testing.assert_allclose(res, project["truth_motion"] + project["noise"], atol=2e-5)
        assert np.abs(f["model_wet"][()]).max() > 0 and np.abs(f["model_hydro"][()]).max() > 0
    j = json.loads((out / "radar_obs.json").read_text())
    assert j["step_history"]["config"]["mintpy.troposphericDelay.method"] == "pyaps"
    assert j["step_history"]["after_weather_step_not_in_these_products"] == [
        "topographicResidual (DEM error + polynomial deformation)"]
    assert len(j["observation"]["sha256"]) == 64


def test_export_check_catches_a_wrong_sign(project, tmp_path):
    M = project["dir"] / "mintpy"
    with h5py.File(M / "timeseries.h5", "r") as f, h5py.File(M / "timeseries_HRRR.h5", "r+") as g:
        g["timeseries"][...] = 2 * f["timeseries"][()] - g["timeseries"][()]     # correction added, not removed
    rec = ro.export(project["dir"], tmp_path / "out", digest=False)
    assert not rec["check_against_applied_correction"]["passed"]


def test_export_refuses_to_write_into_the_project(project):
    with pytest.raises(ValueError, match="read-only"):
        ro.export(project["dir"], project["dir"] / "mintpy" / "atmos")


def test_observability_runs_and_records_its_assumptions(project, tmp_path):
    out = tmp_path / "out"
    ro.export(project["dir"], out, digest=False)
    o = ob.assess(out, project["dir"], block_m=500.0)
    assert o["pairs_used"] == 3 and o["sensitivity"]["synthetic_columns"]
    det = o["detectability_per_date"]
    assert det["pixel_near_target"]["rms_mm"] > 0
    assert det["pixel_near_target"]["min_dT_K_2sigma_fixed_e"] > det["pixel_near_target"]["min_dT_K_2sigma_fixed_rh"]
    # three pairs are too few for the residual-vs-model slope gain: NaN, not a number from nothing
    assert np.isnan(o["stratified_mm_per_km"]["residual_vs_model_slope_gain"])
    assert "sqrt(2)" in o["assumption_independent_dates"]


# ---------------------------------------------------------------- measures
def test_structure_function_of_a_ramp():
    ny, nx = 50, 60
    f = np.tile(np.arange(nx, dtype=float) * 0.001, (ny, 1))      # 1 mm per column
    s = ob.structure_function(f, np.ones((ny, nx), bool), 10.0, 10.0, np.array([5.0, 12.0, 15.0]))
    # lag 10 m: the column shift differs by 1 mm, the row shift by 0 (about as many pairs each)
    assert s["rms"][0] == pytest.approx(np.sqrt(0.5) * 0.001, rel=2e-3)
    # lag 14 m: both diagonals differ by 1 mm
    assert s["rms"][1] == pytest.approx(0.001, rel=1e-9)


def test_height_slope_recovers_a_known_slope():
    rng = np.random.default_rng(3)
    h = rng.uniform(1000, 2000, (40, 40))
    f = 2e-6 * h + rng.normal(0, 1e-4, h.shape)
    s = ob.height_slope(f, h, np.ones(h.shape, bool), ob.block_ids(h.shape, 50.0, 50.0, 200.0))
    assert s["slope"] == pytest.approx(2e-6, rel=0.05) and 0 < s["se_boot"] < 2e-7


def test_consecutive_pairs_respect_the_gap():
    assert ob.consecutive_pairs(["20200101", "20200113", "20200301", "20200313"], 24) == [(0, 1), (2, 3)]
