"""Phase 1, D1 (review_2026-10-08/PLAN.md section 3): ascending + descending
decomposition into Up and East (decompose.py).

Every test builds synthetic MintPy-style runs with a KNOWN Up / East (and,
where it matters, North) field and checks that the decomposition gets it back:

  * the per-cell solve recovers Up and East with per-cell LOS vectors, and is
    the same system MintPy's asc_desc2horz_vert.py solves;
  * its propagated rate error matches a Monte Carlo, and North (not
    estimated) leaks by exactly the stated factor;
  * look directions too alike are refused;
  * runs on different grids (another posting, a half-cell shift, a UTM 56S
    grid) are put on one lat/lon grid;
  * runs with different date ranges are decomposed over the common window
    only (rates), and their series are interpolated to common epochs, never
    paired as if acquired on the same day; an epoch with no SAR date within
    12 days is left empty;
  * runs referenced to different pixels are put on the same ground;
  * unreliable cells stay empty;
  * the CLI works from two pipeline configs and every output says North is
    not estimated.

Synthetic inputs only; no network, nothing under insar_projects.
"""
import json
import subprocess
import sys
from datetime import date, timedelta
from pathlib import Path

import h5py
import numpy as np
import pytest
import yaml

from insar_pipeline import decompose as dc

PKG = Path(__file__).resolve().parents[1]
T0 = date(2017, 1, 1)
ASC_AZ, DESC_AZ = 102.0, -102.0          # MintPy azimuthAngle (ground -> satellite, anticlockwise from north)


# ------------------------------------------------------------------ helpers

def unit(inc_deg, az_deg):
    inc, az = np.radians(inc_deg), np.radians(az_deg)
    return -np.sin(inc) * np.sin(az), np.sin(inc) * np.cos(az), np.cos(inc)


def dates_every(first: date, last: date, step: int = 12):
    out, d = [], first
    while d <= last:
        out.append(d)
        d += timedelta(days=step)
    return out


def yrs(d: date) -> float:
    return (d - T0).days / 365.25


def make_run(folder: Path, *, dates, x0, dx, y0, dy, nx, ny, az, up, east, north=None,
             inc=(33.0, 44.0), ref_rc=None, coh=None, depth=4, epsg=None, name="run",
             velocity=False):
    """Write a MintPy-style run: <name>_ts.h5, <name>_geom.h5, <name>_coh.h5
    (and <name>_vel.h5, a straight line over all its dates, with velocity=True).

    up / east / north(t_years, lon, lat) -> mm. The series is zero on the
    first date; with ref_rc it is also referenced like MintPy's, zero at that
    cell on every date (without, it is absolute, which isolates the other
    behaviours from referencing)."""
    folder.mkdir(parents=True, exist_ok=True)
    xs = x0 + (np.arange(nx) + 0.5) * dx
    ys = y0 + (np.arange(ny) + 0.5) * dy
    X, Y = np.meshgrid(xs, ys)
    if epsg:
        from pyproj import Transformer
        lon, lat = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True).transform(X, Y)
    else:
        lon, lat = X, Y
    inc_deg = np.broadcast_to(np.linspace(inc[0], inc[1], nx)[None, :], (ny, nx)).astype(np.float32)
    az_deg = np.full((ny, nx), az, np.float32) + np.linspace(-1.5, 1.5, ny)[:, None].astype(np.float32)
    e, n, u = unit(inc_deg, az_deg)
    ts = np.empty((len(dates), ny, nx), np.float32)
    for i, d in enumerate(dates):
        t = yrs(d)
        v = e * east(t, lon, lat) + u * up(t, lon, lat)
        if north is not None:
            v = v + n * north(t, lon, lat)
        ts[i] = v / 1000.0
    attrs = {}
    if ref_rc is not None:
        r, c = ref_rc
        ts -= ts[:, r:r + 1, c:c + 1]
        attrs.update({"REF_Y": str(r), "REF_X": str(c)})
    ts -= ts[0:1]
    attrs.update({"X_FIRST": repr(x0), "Y_FIRST": repr(y0), "X_STEP": repr(dx), "Y_STEP": repr(dy),
             "WIDTH": str(nx), "LENGTH": str(ny),
             "REF_DATE": dates[0].strftime("%Y%m%d"), "FILE_TYPE": "timeseries", "UNIT": "m",
             "X_UNIT": "meters" if epsg else "degrees", "Y_UNIT": "meters" if epsg else "degrees",
             "START_DATE": dates[0].strftime("%Y%m%d"), "END_DATE": dates[-1].strftime("%Y%m%d")})
    if epsg:
        attrs["EPSG"] = str(epsg)
    p_ts, p_g, p_c = (folder / f"{name}_{k}.h5" for k in ("ts", "geom", "coh"))
    with h5py.File(p_ts, "w") as f:
        f.create_dataset("timeseries", data=ts, chunks=(min(depth, len(dates)), ny, nx))
        f.create_dataset("date", data=np.array([d.strftime("%Y%m%d").encode() for d in dates]))
        f.create_dataset("bperp", data=np.zeros(len(dates), np.float32))
        f.attrs.update(attrs)
    with h5py.File(p_g, "w") as f:
        f.create_dataset("incidenceAngle", data=inc_deg)
        f.create_dataset("azimuthAngle", data=az_deg)
        f.attrs.update({k: attrs[k] for k in ("X_FIRST", "Y_FIRST", "X_STEP", "Y_STEP", "WIDTH", "LENGTH")})
    with h5py.File(p_c, "w") as f:
        f.create_dataset("temporalCoherence", data=np.ones((ny, nx), np.float32) if coh is None
                         else np.asarray(coh, np.float32))
        f.attrs.update(attrs)
    p_v = None
    if velocity:
        tt = np.array([yrs(d) for d in dates])
        A = np.c_[tt - tt.mean(), np.ones_like(tt)]
        m = np.linalg.lstsq(A, ts.reshape(len(dates), -1), rcond=None)[0]
        p_v = folder / f"{name}_vel.h5"
        with h5py.File(p_v, "w") as f:
            f.create_dataset("velocity", data=m[0].reshape(ny, nx).astype(np.float32))
            f.attrs.update(attrs)
            f.attrs["FILE_TYPE"] = "velocity"
    return p_ts, p_g, p_c, p_v, (lon, lat, e, n, u)


def geom(paths, label, floor=0.4):
    p_ts, p_g, p_c, p_v, _ = paths
    return dc.geometry_from_files(p_ts, p_g, p_c, label=label, min_coherence=floor, velocity_path=p_v)


# A smooth truth: rates linear in lon/lat (so averaging cells onto another grid
# keeps them), Up sinking, East a few mm/yr.
LON0, LAT0 = 150.60, -34.20


def up_rate(lon, lat):
    return -30.0 + 400.0 * (lon - LON0) - 250.0 * (lat - LAT0)


def east_rate(lon, lat):
    return 6.0 + 150.0 * (lat - LAT0)


def UP(t, lon, lat):
    return up_rate(lon, lat) * t


def EAST(t, lon, lat):
    return east_rate(lon, lat) * t


GRID = dict(x0=LON0, dx=0.002, y0=LAT0, dy=-0.002, nx=30, ny=24)


def _read_tif(path):
    import rasterio
    with rasterio.open(path) as r:
        return r.read(1), r.tags(), r.transform


def _grid_lonlat(summary):
    g = summary["grid"]
    lon = g["west"] + (np.arange(g["width"]) + 0.5) * g["lon_step_deg"]
    lat = g["north"] + (np.arange(g["length"]) + 0.5) * g["lat_step_deg"]
    return np.meshgrid(lon, lat)


# ------------------------------------------------------------------ the solve

def test_solver_recovers_known_up_east_with_per_cell_vectors():
    rng = np.random.default_rng(1)
    shape = (40, 50)
    inc = np.stack([rng.uniform(30, 46, shape), rng.uniform(30, 46, shape)])
    az = np.stack([rng.uniform(95, 110, shape), rng.uniform(-110, -95, shape)])
    e, n, u = unit(inc, az)
    U, E = rng.normal(-20, 15, shape), rng.normal(0, 8, shape)
    los = e * E + u * U
    sol = dc.solve_up_east(los, e, u, n)
    assert np.allclose(sol["up"], U, atol=1e-4)
    assert np.allclose(sol["east"], E, atol=1e-4)
    # a scene-mean LOS vector (nisar_mining.decompose) would not: per-cell vectors matter
    em, um = e.reshape(2, -1).mean(1), u.reshape(2, -1).mean(1)
    Ainv = np.linalg.inv(np.c_[em, um])
    up_mean = Ainv[1, 0] * los[0] + Ainv[1, 1] * los[1]
    assert np.nanmax(np.abs(up_mean - U)) > 1.0


def test_solver_is_the_system_mintpy_asc_desc2horz_vert_solves():
    from mintpy.asc_desc2horz_vert import asc_desc2horz_vert
    rng = np.random.default_rng(2)
    shape = (12, 15)
    inc = np.stack([rng.uniform(31, 45, shape), rng.uniform(31, 45, shape)]).astype(np.float32)
    az = np.stack([rng.uniform(98, 106, shape), rng.uniform(-106, -98, shape)]).astype(np.float32)
    e, n, u = unit(inc, az)
    los = (e * rng.normal(0, 10, shape) + u * rng.normal(-30, 10, shape)).astype(np.float32)
    h, v = asc_desc2horz_vert(los, inc, az, horz_az_angle=-90, step=1)
    sol = dc.solve_up_east(los, e, u)
    assert np.allclose(sol["up"], v, atol=1e-3)
    assert np.allclose(sol["east"], h, atol=1e-3)


def test_propagated_rate_error_matches_monte_carlo():
    rng = np.random.default_rng(3)
    k = 40000
    e, n, u = unit(np.array([[34.0], [41.0]]), np.array([[ASC_AZ], [DESC_AZ]]))
    e, n, u = (np.broadcast_to(a, (2, k)) for a in (e, n, u))
    sig = np.broadcast_to(np.array([[2.0], [3.5]]), (2, k))
    U, E = -25.0, 4.0
    los = e * E + u * U + rng.normal(size=(2, k)) * sig
    sol = dc.solve_up_east(los, e, u, sigma=sig)
    assert np.std(sol["up"]) == pytest.approx(float(sol["sigma_up"][0]), rel=0.03)
    assert np.std(sol["east"]) == pytest.approx(float(sol["sigma_east"][0]), rel=0.03)
    assert np.mean(sol["up"]) == pytest.approx(U, abs=0.05)
    # without stated errors there is no propagated sigma, only the amplification
    sol2 = dc.solve_up_east(los, e, u)
    assert np.all(np.isnan(sol2["sigma_up"])) and np.all(np.isfinite(sol2["amplification_up"]))


def test_north_is_not_estimated_and_its_leak_is_stated():
    shape = (5, 6)
    e, n, u = unit(np.stack([np.full(shape, 36.0), np.full(shape, 40.0)]),
                   np.stack([np.full(shape, ASC_AZ), np.full(shape, DESC_AZ)]))
    U, E, N = -40.0, 5.0, 20.0
    los = e * E + n * N + u * U
    sol = dc.solve_up_east(los, e, u, n)
    assert np.allclose(sol["up"] - U, sol["leak_up"] * N, atol=1e-4)
    assert np.allclose(sol["east"] - E, sol["leak_east"] * N, atol=1e-4)
    assert -0.2 < float(np.median(sol["leak_up"])) < -0.08          # Sentinel-1: about -0.16
    assert "North is not estimated" in dc.NORTH_NOTE


def test_look_directions_too_alike_are_refused(tmp_path):
    shape = (4, 4)
    e, n, u = unit(np.stack([np.full(shape, 34.0), np.full(shape, 42.0)]),
                   np.stack([np.full(shape, ASC_AZ), np.full(shape, ASC_AZ + 2)]))
    sol = dc.solve_up_east(np.zeros((2,) + shape), e, u, n)
    assert np.all(np.isnan(sol["up"])) and np.all(sol["condition"] > 10)
    d = dates_every(date(2020, 1, 4), date(2021, 6, 1))
    a = make_run(tmp_path / "a", dates=d, az=ASC_AZ, up=UP, east=EAST, name="a", **GRID)
    b = make_run(tmp_path / "b", dates=d, az=ASC_AZ + 3, up=UP, east=EAST, name="b", inc=(30, 35), **GRID)
    with pytest.raises(ValueError, match="too alike"):
        dc.decompose([geom(a, "asc_t9"), geom(b, "asc_t11")], tmp_path / "out", epochs="none")


# ------------------------------------------------------------------ end to end

def test_runs_on_different_grids_and_date_ranges(tmp_path):
    """A3-like ascending run (starts 2019-11-09) and A1-like descending run
    (from 2017-11-03, on a coarser grid shifted by a fraction of a cell). The
    ground moved at a DIFFERENT rate before the common window: only a
    common-window rate gets the window's Up and East back."""
    def up(t, lon, lat):                       # sinking 3x faster before 2019-11-01
        tb = yrs(date(2019, 11, 1))
        return np.where(t < tb, 3 * up_rate(lon, lat) * (t - tb), up_rate(lon, lat) * (t - tb))

    asc_d = dates_every(date(2019, 11, 9), date(2022, 12, 10))
    desc_d = dates_every(date(2017, 11, 3), date(2022, 12, 25))
    assert asc_d[-1] < desc_d[-1]
    a = make_run(tmp_path / "a", dates=asc_d, az=ASC_AZ, up=up, east=EAST, name="a", velocity=True, **GRID)
    dg = dict(x0=LON0 - 0.0013, dx=0.003, y0=LAT0 + 0.0011, dy=-0.003, nx=22, ny=18)
    b = make_run(tmp_path / "b", dates=desc_d, az=DESC_AZ, up=up, east=EAST, name="b", inc=(36, 46), **dg)
    s = dc.decompose([geom(a, "asc_t9"), geom(b, "desc_t147")], tmp_path / "out", epochs="none",
                     ref="none")
    assert s["common_window"] == ["2019-11-09", asc_d[-1].isoformat()]
    assert s["rates"]["runs"]["desc_t147"]["first_date"] >= "2019-11-09"
    assert s["rates"]["runs"]["desc_t147"]["dates_in_window"] < len(desc_d)
    assert s["grid"]["lon_step_deg"] == pytest.approx(0.003)
    files = s["rates"]["files"]
    up_r, tags, _ = _read_tif(files["vertical_velocity"])
    ea_r, _, _ = _read_tif(files["east_west_velocity"])
    LON, LAT = _grid_lonlat(s)
    ok = np.isfinite(up_r)
    assert ok.mean() > 0.6
    # the coarser descending grid sets the common grid; the ascending run is averaged onto it
    assert "average" in s["runs"][0]["resampling"]
    assert "cell for cell" in s["runs"][1]["resampling"]
    assert np.nanmax(np.abs(up_r - up_rate(LON, LAT))[ok]) < 0.15
    assert np.nanmax(np.abs(ea_r - east_rate(LON, LAT))[ok]) < 0.15
    assert "North is not estimated" in tags["NORTH"]
    assert "Positive = up" in tags["SIGN"]
    # the refit reads the same series as the run's own velocity
    chk = s["rates"]["runs"]["asc_t9"]["check_against_run_velocity"]
    assert chk["compared"] and chk["median_abs_difference_mm_yr"] < 1e-3


def test_whole_record_rate_would_be_wrong(tmp_path):
    """The same set-up decomposed over the descending run's whole record is
    biased - so the common window is what makes the result right."""
    def up(t, lon, lat):
        tb = yrs(date(2019, 11, 1))
        return np.where(t < tb, 3 * up_rate(lon, lat) * (t - tb), up_rate(lon, lat) * (t - tb))
    desc_d = dates_every(date(2017, 11, 3), date(2022, 12, 25))
    b = make_run(tmp_path / "b", dates=desc_d, az=DESC_AZ, up=up, east=EAST, name="b", **GRID)
    g = geom(b, "desc_t147")
    whole = dc._RateFit(g.dates, desc_d[0], desc_d[-1], "linear", g.grid.shape)
    win = dc._RateFit(g.dates, date(2019, 11, 9), desc_d[-1], "linear", g.grid.shape)
    with h5py.File(b[0]) as f:
        ts = f["timeseries"][:]
    for i in range(len(desc_d)):
        whole.add(i, ts[i])
        win.add(i, ts[i])
    _, _, _, _, (lon, lat, e, n, u) = b
    truth = e * east_rate(lon, lat) + u * up_rate(lon, lat)
    assert np.allclose(win.solve()[0] * 1000, truth, atol=0.05)
    assert np.nanmax(np.abs(whole.solve()[0] * 1000 - truth)) > 5.0


def test_series_interpolated_to_common_epochs_never_paired_by_day(tmp_path):
    """Seasonal motion (+-50 mm), ascending and descending acquired 5 days
    apart: interpolation to the 1st of each month is within 0.5 mm; pairing
    the nearest dates as if simultaneous would be off by several mm. A
    descending gap longer than 12 days around one epoch leaves it empty."""
    def up(t, lon, lat):
        return 50.0 * np.sin(2 * np.pi * t) + 0 * lon

    def east(t, lon, lat):
        return 8.0 * np.cos(2 * np.pi * t) + 0 * lat

    asc_d = dates_every(date(2020, 1, 3), date(2021, 12, 30))
    desc_d = dates_every(date(2020, 1, 8), date(2021, 12, 30))
    gap = (date(2021, 3, 20), date(2021, 4, 15))
    desc_d = [d for d in desc_d if not gap[0] <= d <= gap[1]]
    a = make_run(tmp_path / "a", dates=asc_d, az=ASC_AZ, up=up, east=east, name="a", **GRID)
    b = make_run(tmp_path / "b", dates=desc_d, az=DESC_AZ, up=up, east=east, name="b", **GRID)
    s = dc.decompose([geom(a, "asc_t9"), geom(b, "desc_t147")], tmp_path / "out", ref="none")
    ser = s["series"]
    assert ser["base_epoch"] == "2020-02-01"
    assert "2021-04-01" in ser["epochs_without_value"]
    with h5py.File(ser["files"]["vertical"]) as f:
        dates = [d.decode() for d in f["date"][:]]
        up_c = f["timeseries"][:] * 1000
        assert f.attrs["UNIT"] == "m" and "North is not estimated" in f.attrs["DECOMPOSITION_NORTH"]
    with h5py.File(ser["files"]["east_west"]) as f:
        ea_c = f["timeseries"][:] * 1000
    t = np.array([yrs(date(int(d[:4]), int(d[4:6]), int(d[6:]))) for d in dates])
    want_u = 50 * np.sin(2 * np.pi * t) - 50 * np.sin(2 * np.pi * t[0])
    want_e = 8 * np.cos(2 * np.pi * t) - 8 * np.cos(2 * np.pi * t[0])
    k_gap = dates.index("20210401")
    for k in range(len(dates)):
        if k == k_gap:
            assert np.all(np.isnan(up_c[k]))
            continue
        assert np.nanmax(np.abs(up_c[k] - want_u[k])) < 0.7, dates[k]
        assert np.nanmax(np.abs(ea_c[k] - want_e[k])) < 0.7, dates[k]
    # what pairing the nearest acquisitions as simultaneous would have given at one epoch
    ep = date(2020, 6, 1)
    da = min(asc_d, key=lambda d: abs((d - ep).days))
    dd = min(desc_d, key=lambda d: abs((d - ep).days))
    assert da != dd
    assert abs(50 * np.sin(2 * np.pi * yrs(da)) - 50 * np.sin(2 * np.pi * yrs(dd))) > 2.0
    assert Path(ser["files"]["vertical_change"]).exists()


def test_interpolation_plan_rules():
    d = ["20200101", "20200113", "20200210"]
    assert dc.interpolation_plan(d, date(2020, 1, 13), 12) == [(1, 1.0)]
    p = dc.interpolation_plan(d, date(2020, 1, 7), 12)
    assert [i for i, _ in p] == [0, 1] and sum(w for _, w in p) == pytest.approx(1.0)
    assert p[0][1] == pytest.approx(0.5)
    assert dc.interpolation_plan(d, date(2020, 1, 27), 12) is None          # 14 days either side
    assert dc.interpolation_plan(d, date(2019, 12, 31), 12) is None         # before the first date
    assert dc.monthly_epochs(date(2020, 1, 9), date(2020, 4, 1)) == [
        date(2020, 2, 1), date(2020, 3, 1), date(2020, 4, 1)]


def test_runs_referenced_to_different_pixels_are_put_on_one_ground(tmp_path):
    """Each run is referenced (MintPy) to its own pixel, both of which move.
    With a common reference point the result is Up/East relative to that
    point; without one it carries an offset."""
    d = dates_every(date(2020, 1, 4), date(2022, 6, 1))
    a = make_run(tmp_path / "a", dates=d, az=ASC_AZ, up=UP, east=EAST, name="a", ref_rc=(3, 4),
                 inc=(38, 38.001), **GRID)
    b = make_run(tmp_path / "b", dates=d, az=DESC_AZ, up=UP, east=EAST, name="b", ref_rc=(20, 25),
                 inc=(40, 40.001), **GRID)
    lat_r, lon_r = -34.240, 150.655
    s = dc.decompose([geom(a, "asc_t9"), geom(b, "desc_t147")], tmp_path / "pt", epochs="none",
                     ref=(lat_r, lon_r), ref_radius_m=400)
    up_r, tags, _ = _read_tif(s["rates"]["files"]["vertical_velocity"])
    LON, LAT = _grid_lonlat(s)
    near = (np.hypot((LON - lon_r) * 111320 * np.cos(np.radians(lat_r)), (LAT - lat_r) * 111320) <= 400)
    want = up_rate(LON, LAT) - np.median(up_rate(LON, LAT)[near])
    assert np.nanmax(np.abs(up_r - want)) < 0.6
    assert s["reference"]["mode"] == "point" and "same ground" in tags["REFERENCE"]
    s0 = dc.decompose([geom(a, "asc_t9"), geom(b, "desc_t147")], tmp_path / "none", epochs="none", ref="none")
    up0, _, _ = _read_tif(s0["rates"]["files"]["vertical_velocity"])
    assert np.nanmedian(np.abs(up0 - want)) > 3.0
    # auto: the first run's reference point, used when both runs have reliable cells there
    sa = dc.decompose([geom(a, "asc_t9"), geom(b, "desc_t147")], tmp_path / "auto", epochs="none",
                      ref="auto", ref_radius_m=300)
    assert sa["reference"]["mode"] == "point" and sa["reference"]["source"] == "first run's reference point"
    with pytest.raises(ValueError, match="Too few reliable cells"):
        dc.decompose([geom(a, "asc_t9"), geom(b, "desc_t147")], tmp_path / "far", epochs="none",
                     ref=(-30.0, 140.0))


def test_unreliable_cells_stay_empty(tmp_path):
    d = dates_every(date(2020, 1, 4), date(2022, 6, 1))
    coh = np.ones((GRID["ny"], GRID["nx"]), np.float32)
    coh[5:10, 5:12] = 0.2
    a = make_run(tmp_path / "a", dates=d, az=ASC_AZ, up=UP, east=EAST, name="a", **GRID)
    b = make_run(tmp_path / "b", dates=d, az=DESC_AZ, up=UP, east=EAST, name="b", coh=coh, **GRID)
    s = dc.decompose([geom(a, "asc_t9"), geom(b, "desc_t147", floor=0.4)], tmp_path / "out", ref="none",
                     epochs=["2020-06-01", "2021-06-01"])
    up_r, _, _ = _read_tif(s["rates"]["files"]["vertical_velocity"])
    assert "none: cell for cell" in s["runs"][1]["resampling"]
    assert np.all(np.isnan(up_r[5:10, 5:12])) and np.isfinite(up_r[12:, 15:]).all()
    with h5py.File(s["series"]["files"]["vertical"]) as f:
        assert np.all(np.isnan(f["timeseries"][1, 5:10, 5:12]))
    # a floor of 0.1 keeps them
    s2 = dc.decompose([geom(a, "asc_t9"), geom(b, "desc_t147", floor=0.1)], tmp_path / "out2", ref="none",
                      epochs="none")
    up2, _, _ = _read_tif(s2["rates"]["files"]["vertical_velocity"])
    assert np.isfinite(up2[5:10, 5:12]).all()


def test_projected_utm56s_run_is_resampled_to_the_lat_lon_grid(tmp_path):
    from pyproj import Transformer
    d = dates_every(date(2020, 1, 4), date(2022, 6, 1))
    a = make_run(tmp_path / "a", dates=d, az=ASC_AZ, up=UP, east=EAST, name="a", **GRID)
    x, y = Transformer.from_crs("EPSG:4326", "EPSG:32756", always_xy=True).transform(LON0 - 0.01, LAT0 + 0.01)
    b = make_run(tmp_path / "b", dates=d, az=DESC_AZ, up=UP, east=EAST, name="b", epsg=32756,
                 x0=x, dx=90.0, y0=y, dy=-90.0, nx=90, ny=80)
    s = dc.decompose([geom(a, "asc_t9"), geom(b, "desc_t147")], tmp_path / "out", epochs="none", ref="none")
    up_r, _, tr = _read_tif(s["rates"]["files"]["vertical_velocity"])
    LON, LAT = _grid_lonlat(s)
    ok = np.isfinite(up_r)
    assert ok.mean() > 0.8
    assert np.nanmax(np.abs(up_r - up_rate(LON, LAT))[ok]) < 0.5
    assert s["runs"][1]["grid_crs"] == "EPSG:32756"
    assert tr.a == pytest.approx(0.002)             # the lat/lon grid (asc posting is the coarser)


def test_linear_annual_rate_model(tmp_path):
    def up(t, lon, lat):
        return up_rate(lon, lat) * t + 15 * np.sin(2 * np.pi * t)
    d = dates_every(date(2020, 1, 4), date(2021, 9, 1))
    a = make_run(tmp_path / "a", dates=d, az=ASC_AZ, up=up, east=EAST, name="a", **GRID)
    b = make_run(tmp_path / "b", dates=d, az=DESC_AZ, up=up, east=EAST, name="b", **GRID)
    runs = [geom(a, "asc_t9"), geom(b, "desc_t147")]
    LON = None
    errs = {}
    for model in ("linear", "linear_annual"):
        s = dc.decompose(runs, tmp_path / model, epochs="none", ref="none", rate_model=model)
        up_r, _, _ = _read_tif(s["rates"]["files"]["vertical_velocity"])
        LON, LAT = _grid_lonlat(s)
        errs[model] = float(np.nanmax(np.abs(up_r - up_rate(LON, LAT))))
    assert errs["linear_annual"] < 0.05 < errs["linear"]


# ------------------------------------------------------------------ the CLI, from configs

def _project(tmp: Path, cfg_name: str, *, track, direction, az, dates, inc, ref_rc):
    # a phase-linked config of the run's own (until 28 Oct 2026 the A1 / A3
    # experiment configs, which then left the repository)
    proj = tmp / cfg_name.replace(".yaml", "")
    raw = {"backend": "miaplpy", "project_dir": str(proj), "bbox": [150.55, -34.30, 150.78, -34.14],
           "start_date": "2017-11-01", "end_date": "2022-12-31", "track": track, "orbit_direction": direction,
           "min_temporal_coherence": 0.4, "purpose": "experiment", "stack_virtual_merge": "False",
           "mintpy_deramp": "no"}
    cfg = tmp / cfg_name
    cfg.write_text(yaml.safe_dump(raw))
    geo = proj / "mintpy" / "geo"
    g = dict(x0=150.60, dx=0.002, y0=-34.16, dy=-0.002, nx=40, ny=36)
    p_ts, p_g, p_c, p_v, _ = make_run(geo, dates=dates, az=az, up=UP, east=EAST, name="tmp", inc=inc,
                                      ref_rc=ref_rc, velocity=True, **g)
    p_ts.rename(geo / "geo_timeseries_ERA5_demErr.h5")
    p_g.rename(geo / "geo_geometryRadar.h5")
    p_c.rename(geo / "geo_temporalCoherence.h5")
    p_v.rename(geo / "geo_velocity.h5")
    return cfg


def test_cli_from_two_project_configs(tmp_path):
    # MintPy-referenced runs (each to its own cell); near-constant incidence so
    # that 'relative to the reference point' is exact (see decompose: with
    # varying look vectors the point's own motion is seen ~1.3% per degree off)
    asc_d = dates_every(date(2019, 11, 9), date(2022, 12, 27))
    desc_d = dates_every(date(2017, 11, 3), date(2022, 12, 25))
    asc = _project(tmp_path, "asc_t9.yaml", track=9, direction="ascending", az=ASC_AZ, inc=(36, 36.001),
                   ref_rc=(5, 5), dates=asc_d)
    desc = _project(tmp_path, "desc_t147.yaml", track=147, direction="descending", az=DESC_AZ, inc=(41, 41.001),
                    ref_rc=(30, 30), dates=desc_d)
    w = (asc_d[0].strftime("%Y%m%d"), min(asc_d[-1], desc_d[-1]).strftime("%Y%m%d"))
    out = tmp_path / "d1"
    r = subprocess.run([sys.executable, "-m", "insar_pipeline.decompose", str(asc), str(desc),
                        "--out", str(out), "--window", "2019-11-01", "2022-12-31", "--ref-radius-m", "400",
                        "--epochs", "2020-01-01,2021-01-01,2022-01-01"],
                       cwd=str(PKG.parent), capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-3000:]
    assert "North" in r.stdout and "not estimated" in r.stdout
    s = json.loads((out / "decomposition_summary.json").read_text())
    assert [x["label"] for x in s["runs"]] == ["asc_t9", "desc_t147"]
    assert s["runs"][0]["coherence_floor"] == 0.4                 # the config's floor
    assert s["runs"][0]["config"].endswith("asc_t9.yaml")
    assert s["common_window"] == [asc_d[0].isoformat(), min(asc_d[-1], desc_d[-1]).isoformat()]
    assert s["reference"]["mode"] == "point"                       # auto: the first run's reference cell
    assert s["rates"]["runs"]["desc_t147"]["check_against_run_velocity"]["compared"] is False
    assert "North is not estimated" in s["north"]
    assert s["series"]["epochs"] == ["2020-01-01", "2021-01-01", "2022-01-01"]
    readme = (out / "decomposition_README.txt").read_text()
    assert "North is not estimated" in readme
    up_r, tags, _ = _read_tif(s["rates"]["files"]["vertical_velocity"])
    LON, LAT = _grid_lonlat(s)
    ref = (s["reference"]["lon"], s["reference"]["lat"])
    want = up_rate(LON, LAT) - up_rate(*ref)
    assert np.nanmax(np.abs(up_r - want)) < 0.6
    for name in (f"vertical_velocity_mm_per_year_{w[0]}_{w[1]}_decomposed.tif",
                 f"east_west_velocity_mm_per_year_{w[0]}_{w[1]}.tif",
                 "north_leak_into_vertical.tif", "decomposition_condition_number.tif",
                 "vertical_timeseries_decomposed.h5", "east_west_timeseries.h5",
                 "decomposition_layers.h5", "vertical_change_mm_20200101_20220101_decomposed.tif"):
        assert (out / name).exists(), name
    with h5py.File(out / "decomposition_layers.h5") as f:
        assert {"los_velocity_asc_t9", "los_up_desc_t147", "north_leak_vertical"} <= set(f)
        assert "not estimated" in f.attrs["NORTH"]


# ------------------------------------------------------------------ review fixes (8 Oct)

def test_explicit_epochs_are_kept_to_the_common_window(tmp_path):
    """--window narrows the common window; an explicit epoch outside it is
    left out (and listed), not decomposed."""
    d = dates_every(date(2020, 1, 4), date(2022, 6, 1))
    a = make_run(tmp_path / "a", dates=d, az=ASC_AZ, up=UP, east=EAST, name="a", **GRID)
    b = make_run(tmp_path / "b", dates=d, az=DESC_AZ, up=UP, east=EAST, name="b", **GRID)
    s = dc.decompose([geom(a, "asc_t9"), geom(b, "desc_t147")], tmp_path / "out", ref="none",
                     window=("2021-01-01", "2021-12-31"),
                     epochs=["2020-06-01", "2021-06-01", "2021-09-01", "2022-03-01"])
    assert s["common_window"] == ["2021-01-01", "2021-12-31"]
    assert s["series"]["epochs"] == ["2021-06-01", "2021-09-01"]
    assert s["series"]["epochs_outside_window"] == ["2020-06-01", "2022-03-01"]
    assert s["series"]["epochs_before_base_without_value"] == []
    assert "outside the common window: 2020-06-01, 2022-03-01" in \
        (tmp_path / "out" / "decomposition_README.txt").read_text()


def test_epoch_file_comments_are_ignored(tmp_path):
    p = tmp_path / "epochs.txt"
    p.write_text("# survey epochs, digitised\n2020-01-01\n  # indented comment, with a comma\n"
                 "2020-02-01, 2020-03-01  # two on one line\n")
    assert dc._parse_epochs(f"@{p}") == ["2020-01-01", "2020-02-01", "2020-03-01"]


def test_failed_layers_write_leaves_no_partial_file(tmp_path, monkeypatch):
    d = dates_every(date(2020, 1, 4), date(2021, 6, 1))
    a = make_run(tmp_path / "a", dates=d, az=ASC_AZ, up=UP, east=EAST, name="a", **GRID)
    b = make_run(tmp_path / "b", dates=d, az=DESC_AZ, up=UP, east=EAST, name="b", **GRID)

    def boom(path, *args, **kw):
        Path(path).write_bytes(b"half")
        raise OSError("disk full")
    monkeypatch.setattr(dc, "_write_layers", boom)
    with pytest.raises(OSError, match="disk full"):
        dc.decompose([geom(a, "asc_t9"), geom(b, "desc_t147")], tmp_path / "out", ref="none", epochs="none")
    assert not list((tmp_path / "out").glob("*.partial"))
    assert not (tmp_path / "out" / "decomposition_layers.h5").exists()


def test_reference_point_as_text_through_the_api(tmp_path):
    d = dates_every(date(2020, 1, 4), date(2021, 6, 1))
    a = make_run(tmp_path / "a", dates=d, az=ASC_AZ, up=UP, east=EAST, name="a", **GRID)
    b = make_run(tmp_path / "b", dates=d, az=DESC_AZ, up=UP, east=EAST, name="b", **GRID)
    s = dc.decompose([geom(a, "asc_t9"), geom(b, "desc_t147")], tmp_path / "out", epochs="none",
                     ref="-34.240, 150.655", ref_radius_m=400)
    assert s["reference"]["mode"] == "point"
    assert (s["reference"]["lat"], s["reference"]["lon"]) == (-34.24, 150.655)
