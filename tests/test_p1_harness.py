"""PLAN phase 1, V1 (review 8 Oct 2026, group 'harness'): the validation
harness (validation.py) and `insar validate` (cli.py).

What is checked:
  - the three fixed protocols are read as written: sampling, reliability,
    matching, models, water years, bootstrap, station roles, the reference
    station and its fallbacks, B-2's inheritance from B-1, and every graded
    threshold row parses and grades boundary values the way its text says
    (including the count rule "at most 1 outside" and A's max(30 mm, 10%));
  - York regression (errors in both), the bootstrap (seeded, the pair
    double-difference formula equal to recomputing the pairs), the protocol
    model fit (piecewise by water year + annual, lag-1 inflation);
  - CRS-aware sampling: median of reliable cells within the radius on a
    lat/lon grid and on a UTM grid, the minimum cells and independent
    pixels, never filled; GNSS epoch matching within +-3 days;
  - GNSS projected with the per-pixel LOS vector after double-differencing
    (plate motion cancels); the leave-one-out datum never uses the scored
    station;
  - end to end on a synthetic product with known motion: the network design
    (B1) and the reference-station design (A) recover zero error and pass;
    planted errors fail; a masked event is honest, an unflagged wrong one
    is an unflagged failure; outputs (json, md, csv, plots) are written;
  - `python -m insar_pipeline.cli validate` dispatches and refuses while a
    run holds the project.
None of this existed before V1, so every test fails on the old tree.
No network, nothing under ~/insar_projects: everything is in tmp_path, and
ground truth comes from a fake ground_truth module with the agreed API.
"""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import yaml

from insar_pipeline import validation as V

PKG = Path(__file__).resolve().parents[1]
REVIEW = PKG / "review_2026-10-08"
PA, PB1, PB2 = REVIEW / "protocol_A.yaml", REVIEW / "protocol_B1.yaml", REVIEW / "protocol_B2.yaml"


# ------------------------------------------------------------------ protocols as written

def test_fixed_protocols_load_and_are_recognised_as_fixed():
    for p in (PA, PB1, PB2):
        st = V.fixed_hash_status(p)
        assert st["status"] == "fixed", st
    P = V.load_protocol(PB2)
    # B-2 inherits sampling, reliability, matching from B-1 (its own keys win)
    assert P["sampling"]["radius_m"] == 100 and P["reliability"]["coherence_floor"] == 0.4
    assert "power_statement" in P["statistics_reported"] and "confidence_intervals" in P["statistics_reported"]
    assert P["_inherited"]["file"].endswith("protocol_B1.yaml")


def test_ad_hoc_and_edited_protocols_are_labelled(tmp_path):
    q = tmp_path / "protocol_B1.yaml"
    q.write_text(PB1.read_text() + "\n# edited\n")
    assert V.fixed_hash_status(q)["status"] == "changed"
    r = tmp_path / "smoke.yaml"
    r.write_text(PB1.read_text())
    assert V.fixed_hash_status(r)["status"] == "ad_hoc"


def test_protocol_params_B1():
    prm = V.protocol_params(V.load_protocol(PB1), run="B1", backend="miaplpy")
    assert prm["design"] == "network"
    assert (prm["radius_m"], prm["min_cells"], prm["min_independent_pixels"]) == (100.0, 5, 2)
    assert prm["floor"] == 0.4 and prm["sensitivity_floors"] == [0.3, 0.5, 0.7]
    assert prm["match_days"] == 3
    assert prm["hinges"] == ["2022-10-01", "2023-10-01", "2024-10-01", "2025-10-01"]
    assert [w[0] for w in prm["water_years"]] == ["WY2022", "WY2023", "WY2024", "WY2025", "WY2026"]
    assert prm["wy_min_years"] == 0.75 and prm["wy_min_epochs"] == 10
    assert prm["n_boot"] == 10000 and prm["seed"] == 20261008
    assert sorted(prm["rate_stations"]) == sorted(["CRCN", "LEMA", "CAHA", "KDL1", "QCAP", "KDL2"])
    assert "HANF" in prm["series_stations"] and "HANF" not in prm["rate_stations"]
    assert prm["window"] == ["2021-10-01", "2026-09-30"]
    nis = V.protocol_params(V.load_protocol(PB1), run="B5", backend="nisar")
    assert nis["radius_m"] == 160.0 and nis["min_cells"] == 3


def test_protocol_params_B2_and_A():
    b2 = V.protocol_params(V.load_protocol(PB2), run="B-2a")
    assert b2["design"] == "network" and b2["hinges"] == [] and b2["water_years"] == [] and b2["annual"]
    assert len(b2["rate_stations"]) == 7 and "UHKS" not in b2["rate_stations"] and "TXKY" not in b2["stations"]
    assert b2["radius_m"] == 100.0 and b2["match_days"] == 3
    a = V.protocol_params(V.load_protocol(PA), run="A3")
    assert a["design"] == "reference_station" and a["reference_station"] == "CA07"
    assert a["reference_fallback"] == ["PCTN", "MENA"] and a["usable_share"] == 0.8
    assert a["source_order"] == ["ga", "ngl"]
    assert a["validation_stations"] == ["CA19"] and sorted(a["null_stations"]) == ["CRDX", "MENA", "PCTN"]
    assert a["window"] == ["2019-11-01", "2022-12-31"]          # the run's own window
    assert "CA10" not in a["stations"]                          # not_used
    assert V._screen_models(a)[0] == {"CA19": "local", "CA07": "linear", "PCTN": "linear",
                                      "MENA": "linear", "CRDX": "linear"}


def test_every_graded_row_of_the_fixed_protocols_parses():
    n = 0
    for p in (PA, PB1, PB2):
        P = V.load_protocol(p)
        for name, c in (P.get("comparisons") or {}).items():
            specs = [c.get("grade")] if isinstance(c, dict) else []
            specs += [v.get("grade") for v in c.values() if isinstance(v, dict)] if isinstance(c, dict) else []
            for g in specs:
                if isinstance(g, dict) and "pass" in g:
                    pg = V.parse_grade_text(g["pass"])
                    share = next((x["metric"] for grp in pg for x in grp if x["metric"].startswith("share")), None)
                    if g.get("marginal"):
                        V.parse_grade_text(g["marginal"], share)
                    n += 1
    assert n >= 20


def _grade(P, row, **M):
    return V.grade_row(P["comparisons"][row]["grade"], M)["grade"]


def test_grades_follow_the_threshold_text():
    B1, B2, A = (V.load_protocol(p) for p in (PB1, PB2, PA))
    assert [_grade(B1, "rate_rmse_dd", rmse=v) for v in (4.9, 5.0, 7.0, 10.1)] == ["pass", "pass", "marginal", "fail"]
    assert [_grade(B1, "rate_bias_loo", abs_bias=v) for v in (1.5, 4.0, 6.0)] == ["pass", "marginal", "fail"]
    assert _grade(B1, "york_slope", slope=1.02, ci_includes_1=True) == "pass"
    assert _grade(B1, "york_slope", slope=1.02, ci_includes_1=False) == "marginal"     # CI excludes 1
    assert _grade(B1, "york_slope", slope=1.07, ci_includes_1=True) == "marginal"
    assert _grade(B1, "york_slope", slope=1.2, ci_includes_1=True) == "fail"
    assert _grade(B2, "york_slope", slope=1.07, ci_includes_1=True) == "pass"         # Houston: 0.9-1.1
    assert [_grade(B1, "rate_correlation", r=v) for v in (0.99, 0.96, 0.9)] == ["pass", "marginal", "fail"]
    assert [_grade(B2, "rate_correlation", r=v) for v in (0.91, 0.85, 0.7)] == ["pass", "marginal", "fail"]
    assert [_grade(B1, "series_rmse", median_rmse=v) for v in (17, 25, 31)] == ["pass", "marginal", "fail"]
    # uncertainty calibration: the protocol's own count rule for its station count
    assert _grade(B1, "uncertainty_calibration", share_inside=5 / 6, n_outside=1, n=6) == "pass"
    assert _grade(B1, "uncertainty_calibration", share_inside=4 / 6, n_outside=2, n=6) == "marginal"
    assert _grade(B1, "uncertainty_calibration", share_inside=3 / 6, n_outside=3, n=6) == "fail"
    assert _grade(B1, "uncertainty_calibration", share_inside=0.92, n_outside=2, n=25) == "pass"    # many points: the share
    # the count rule holds at the station count it was written for: 1 of 3 outside (67% inside) is no pass
    assert _grade(B1, "uncertainty_calibration", share_inside=2 / 3, n_outside=1, n=3) == "marginal"
    assert _grade(B2, "uncertainty_calibration", share_inside=5 / 6, n_outside=1, n=6) == "marginal"   # "7 sites"
    assert _grade(B2, "uncertainty_calibration", share_inside=6 / 7, n_outside=1, n=7) == "pass"
    assert [_grade(B1, "tre_vertical", share_cells=v) for v in (0.95, 0.8, 0.5)] == ["pass", "marginal", "fail"]
    # A: |err| <= max(30 mm, 10% of truth) / 20%
    assert _grade(A, "ca19_lw902", abs_err=31.0, truth=-326.0) == "pass"
    assert _grade(A, "ca19_lw902", abs_err=29.0, truth=-100.0) == "pass"
    assert _grade(A, "ca19_lw902", abs_err=50.0, truth=-326.0) == "marginal"
    assert _grade(A, "ca19_lw902", abs_err=70.0, truth=-326.0) == "fail"
    assert [_grade(A, "appin_peak_ratio", ratio=v) for v in (0.85, 0.7, 0.5)] == ["pass", "marginal", "fail"]
    assert _grade(A, "stable_ring", share_beyond=0.04, rmse_vs_zero=4.0) == "pass"
    assert _grade(A, "stable_ring", share_beyond=0.04, rmse_vs_zero=6.0) == "marginal"
    assert _grade(A, "stable_ring", share_beyond=0.2, rmse_vs_zero=1.0) == "fail"
    assert [_grade(A, "unflagged_failure_rate", rate=v) for v in (0.05, 0.2, 0.3)] == ["pass", "marginal", "fail"]
    # numpy inputs (what the harness computes) grade the same as Python floats
    assert _grade(A, "ca19_lw902", abs_err=np.float64(119.0), truth=np.float64(-227.6)) == "fail"
    assert _grade(B1, "rate_rmse_dd", rmse=np.float32(11.0)) == "fail"
    assert _grade(B1, "york_slope", slope=np.float64(1.0), ci_includes_1=np.bool_(False)) == "marginal"
    assert _grade(B1, "rate_rmse_dd", rmse=np.nan) == "not_testable"
    # a needed metric missing is 'not testable', never a pass
    assert _grade(B1, "york_slope", slope=1.0) == "not_testable"
    assert V.grade_row({"pass": "wibble >= 3"}, {})["grade"] == "not_computed"
    assert V.grade_row("report only", {})["grade"] == "report_only"


# ------------------------------------------------------------------ statistics

def test_york_corrects_the_attenuation_ordinary_least_squares_suffers():
    rng = np.random.default_rng(1)
    xt = np.linspace(-100, 100, 40)
    x = xt + rng.normal(0, 25, xt.size)          # truth with large errors
    y = 1.0 * xt + rng.normal(0, 2, xt.size)
    ols = np.polyfit(x, y, 1)[0]
    f = V.york_fit(x, y, 25.0, 2.0)
    assert ols < 0.95                             # attenuated
    assert abs(f["slope"] - 1.0) < 0.06 and f["slope_se"] > 0
    g = V.york_fit(xt, 2 * xt + 3, 0.1, 0.1)
    assert abs(g["slope"] - 2) < 1e-6 and abs(g["intercept"] - 3) < 1e-5


def test_bootstrap_is_seeded_and_the_pair_formula_matches_recomputing_pairs():
    vals = np.array([1.0, 2.0, 3.0, 10.0])
    stat = lambda idx, _r: float(vals[idx].mean())  # noqa: E731
    a, b = V.bootstrap_ci(4, stat, 500, 7), V.bootstrap_ci(4, stat, 500, 7)
    assert a == b and a["lo"] < 4 < a["hi"]
    assert V.bootstrap_ci(1, stat, 500, 7)["lo"] is None
    # network_component's double-difference RMSE over resampled stations
    rng = np.random.default_rng(3)
    S = 4
    err = {(i, j): rng.normal(0, 3, 3) for i in range(S) for j in range(i + 1, S)}
    idx = np.array([0, 0, 2, 3])                  # station 0 drawn twice, 1 not at all
    pos = [(p, q) for p in range(S) for q in range(p + 1, S) if idx[p] != idx[q]]
    pooled = np.concatenate([err[tuple(sorted((idx[p], idx[q])))] for p, q in pos])
    SS, NN = np.zeros((S, S)), np.zeros((S, S))
    for (i, j), e in err.items():
        SS[i, j] = SS[j, i] = np.sum(e ** 2)
        NN[i, j] = NN[j, i] = e.size
    c = np.bincount(idx, minlength=S).astype(float)
    w = np.triu(np.outer(c, c), 1)
    assert math.isclose(math.sqrt(np.sum(w * SS) / np.sum(w * NN)), math.sqrt(np.mean(pooled ** 2)))


def test_model_rates_by_water_year_and_lag1_inflation():
    prm = V.protocol_params(V.load_protocol(PB1), run="B1")
    t = np.arange(V._day("2021-10-05"), V._day("2024-09-28"), 12)
    rates = {2022: -250.0, 2023: 40.0, 2024: -60.0}
    y, v = [], 0.0
    for k, d in enumerate(t):
        yr = int(str(np.datetime64(int(d), "D"))[:4]) + (1 if str(np.datetime64(int(d), "D"))[5:7] >= "10" else 0)
        v += rates[yr] * (12 / 365.25 if k else 0)
        y.append(v + 8 * math.sin(2 * math.pi * d / 365.25))
    r = V.model_rates(t, np.array(y), prm)
    for wy, want in rates.items():
        assert abs(r["rates"][f"WY{wy}"]["rate"] - want) < 3, (wy, r["rates"])
    assert "WY2025" not in r["rates"]                              # no data: no rate
    noise = np.cumsum(np.random.default_rng(2).normal(0, 2, t.size))   # random walk: r1 ~ 1
    f_white = V.fit_series(t, np.random.default_rng(2).normal(0, 2, t.size), (), True)
    f_rw = V.fit_series(t, noise, (), True)
    assert f_rw["r1"] > 0.5 and f_rw["n_eff"] < 0.4 * f_rw["n"]
    assert abs(f_white["r1"]) < 0.3


# ------------------------------------------------------------------ synthetic product

LAT0, LON0, DLAT, DLON = 36.30, -119.70, -0.0005, 0.0005
NY = NX = 220
PLATE = np.array([-15.0, 20.0, 0.0])          # mm/yr, common motion (IGS20-like)


def _day_years(days):
    return (np.asarray(days, float) - V._day("2021-10-01")) / 365.25


def _vertical(lat, lon, days):
    """Ground vertical (mm): a linear rate field plus an annual term."""
    rate = -40.0 + 400.0 * (lat - 36.25) - 300.0 * (lon + 119.65)
    t = _day_years(days)
    return rate * t + 6.0 * np.sin(2 * np.pi * np.asarray(days, float) / 365.25)


def _enu(lat, lon, days):
    t = _day_years(days)
    e = PLATE[0] * t
    n = PLATE[1] * t
    return np.column_stack([e * np.ones_like(t), n * np.ones_like(t), _vertical(lat, lon, days)])


def _geometry():
    rr, cc = np.mgrid[0:NY, 0:NX]
    inc = 33.0 + 10.0 * cc / NX                 # varies across the grid, so LOS vectors differ
    az = np.full(inc.shape, -100.0)             # descending, right-looking (MintPy: ground-to-satellite, anticlockwise from north; ascending is ~ +100)
    lat = LAT0 + (rr + 0.5) * DLAT
    lon = LON0 + (cc + 0.5) * DLON
    return inc, az, lat, lon


def _los(inc, az):
    i, a = np.radians(inc), np.radians(az)
    return np.stack([-np.sin(i) * np.sin(a), np.sin(i) * np.cos(a), np.cos(i)])


def make_product(root: Path, dates, tcoh=None, extra=None, epsg=None):
    """A geocoded MintPy-like product: time series (m) whose every cell is
    l(cell) . ENU(cell) minus the reference pixel's, temporal coherence,
    geometry and velocity, under root/mintpy/geo."""
    import h5py
    geo = root / "mintpy" / "geo"
    geo.mkdir(parents=True)
    days = V._days(dates)
    inc, az, lat, lon = _geometry()
    L = _los(inc, az)                                     # (3, NY, NX)
    t = _day_years(days)[:, None, None]
    rate = -40.0 + 400.0 * (lat - 36.25) - 300.0 * (lon + 119.65)
    U = rate[None] * t + 6.0 * np.sin(2 * np.pi * days / 365.25)[:, None, None]
    los = L[0][None] * PLATE[0] * t + L[1][None] * PLATE[1] * t + L[2][None] * U
    ref = los[:, NY // 2, NX // 2].copy()
    ts = (los - ref[:, None, None]) / 1000.0
    if extra is not None:
        ts = ts + extra / 1000.0
    if epsg:
        attrs = {"X_FIRST": "500000.0", "Y_FIRST": "4020000.0", "X_STEP": "45.0", "Y_STEP": "-55.0", "EPSG": str(epsg)}
    else:
        attrs = {"X_FIRST": str(LON0), "Y_FIRST": str(LAT0), "X_STEP": str(DLON), "Y_STEP": str(DLAT)}
    attrs.update({"WIDTH": str(NX), "LENGTH": str(NY), "REF_Y": str(NY // 2), "REF_X": str(NX // 2),
                  "REF_DATE": dates[0], "START_DATE": dates[0], "END_DATE": dates[-1]})
    with h5py.File(geo / "geo_timeseries_ERA5_demErr.h5", "w") as f:
        f.create_dataset("timeseries", data=ts.astype(np.float32), chunks=(4, 32, 32))
        f.create_dataset("date", data=np.array([d.encode() for d in dates]))
        f.attrs.update({**attrs, "UNIT": "m"})
    with h5py.File(geo / "geo_velocity.h5", "w") as f:
        f.create_dataset("velocity", data=(ts[-1] - ts[0]).astype(np.float32))
        f.create_dataset("velocityStd", data=np.full((NY, NX), 0.0005, np.float32))     # 0.5 mm/yr
        f.attrs.update({**attrs, "UNIT": "m/year"})
    with h5py.File(geo / "geo_temporalCoherence.h5", "w") as f:
        f.create_dataset("temporalCoherence", data=(np.full((NY, NX), 0.9) if tcoh is None else tcoh).astype(np.float32))
        f.attrs.update(attrs)
    with h5py.File(geo / "geo_geometryRadar.h5", "w") as f:
        f.create_dataset("incidenceAngle", data=inc.astype(np.float32))
        f.create_dataset("azimuthAngle", data=az.astype(np.float32))
        f.create_dataset("height", data=np.zeros((NY, NX), np.float32))
        f.attrs.update(attrs)
    return SimpleNamespace(project_dir=root, mintpy_dir=root / "mintpy", backend="miaplpy",
                           min_temporal_coherence=0.4)


def make_d1(path: Path, d0="2021-11-01", d1="2024-09-01", step=0.0004):
    """A decompose.py-like Up series: own coarser lat/lon grid, monthly epochs,
    metres, change since the first epoch, referenced to the grid centre."""
    import h5py
    months = pd.date_range(d0, d1, freq="MS")
    days = V._days([m.strftime("%Y%m%d") for m in months])
    ny = nx = int(NY * abs(DLAT) / step)
    rr, cc = np.mgrid[0:ny, 0:nx]
    lat, lon = LAT0 - (rr + 0.5) * step, LON0 + (cc + 0.5) * step
    up = np.stack([_vertical(lat, lon, np.full(lat.shape, d)) for d in days])
    # referenced to the product's reference pixel (as decompose.py does by default: the first run's own)
    rlat, rlon = LAT0 + (NY // 2 + 0.5) * DLAT, LON0 + (NX // 2 + 0.5) * DLON
    up = up - _vertical(rlat, rlon, days)[:, None, None]
    up = up - up[0]
    with h5py.File(path, "w") as f:
        f.create_dataset("timeseries", data=(up / 1000.0).astype(np.float32))
        f.create_dataset("date", data=np.array([m.strftime("%Y%m%d").encode() for m in months]))
        f.attrs.update({"X_FIRST": str(LON0), "Y_FIRST": str(LAT0), "X_STEP": str(step), "Y_STEP": str(-step),
                        "EPSG": "4326", "UNIT": "m", "DECOMPOSITION_NORTH": "North is not estimated."})
    return path


def _rc(lat, lon):
    return int((lat - LAT0) / DLAT), int((lon - LON0) / DLON)


def _cell_centre(lat, lon):
    r, c = _rc(lat, lon)
    return LAT0 + (r + 0.5) * DLAT, LON0 + (c + 0.5) * DLON


class FakeSeriesSet(dict):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.missing = {}


class FakeGT:
    """The agreed ground_truth API over synthetic stations."""

    def __init__(self, stations, noise=(0.5, 0.5, 1.0), seed=0, overrides=None, spans=None):
        self.st, self.noise, self.seed = stations, noise, seed
        self.overrides = overrides or {}
        self.spans = spans or {}
        self.points = []
        self.calls = []

    def _frame(self, sid, d0, d1, model):
        lat, lon = self.st[sid]
        if sid in self.spans:
            d0, d1 = max(d0, self.spans[sid][0]), min(d1, self.spans[sid][1])
        days = np.arange(V._day(d0), V._day(d1) + 1)
        enu = _enu(lat, lon, days)
        if sid in self.overrides:
            enu = self.overrides[sid](days, enu)
        rng = np.random.default_rng(self.seed + sum(map(ord, sid)))
        enu = enu + rng.normal(0, 1, enu.shape) * np.array(self.noise)
        df = pd.DataFrame({"e_mm": enu[:, 0], "n_mm": enu[:, 1], "u_mm": enu[:, 2],
                           "se_mm": self.noise[0], "sn_mm": self.noise[1], "su_mm": self.noise[2]},
                          index=pd.DatetimeIndex(np.array(days, "datetime64[D]"), name="date"))
        df.attrs.update({"lat": lat, "lon": lon, "sha256": "f" * 64, "fetched": "2026-10-08T00:00:00Z",
                         "source_url": f"https://example.invalid/{sid}.tenv3", "last_epoch_in_file": str(d1),
                         "robust_sigma_mm": dict(zip("enu", self.noise)), "model": {"name": model, "hinges": []},
                         "steps_removed": [], "outliers_removed": 0})
        return df

    def load_gnss_series(self, ids, window=None, cache_dir=None, frame="IGS20", model="linear", hinges=(),
                         earthquake_steps=False):
        self.calls.append(("gnss", list(ids), window, model, list(hinges)))
        self.earthquake_steps = earthquake_steps
        out = FakeSeriesSet()
        for sid in ids:
            if sid not in self.st:
                out.missing[sid] = "no such station"
                continue
            out[sid] = self._frame(sid, window[0], window[1], model.get(sid) if isinstance(model, dict) else model)
        return out

    def load_ga_xlsx(self, path, sites=None, window=None, coords=None, model="local"):
        self.calls.append(("ga", path))
        return FakeSeriesSet()

    def fetch_ga_xlsx(self, cache_dir):
        raise OSError("offline test")

    def load_points_csv(self, path):
        self.calls.append(("points", path))
        return self.points


STATIONS_B = {"AAAA": (36.272, -119.672), "BBBB": (36.255, -119.640), "CCCC": (36.235, -119.618),
              "DDDD": (36.222, -119.665), "EEEE": (36.248, -119.605)}


def b1_protocol(tmp_path: Path, stations=STATIONS_B) -> Path:
    """protocol_B1 with this synthetic site's stations and box (a copy: the
    fixed file is never touched)."""
    P = yaml.safe_load(PB1.read_text())
    # the protocol as first fixed: these tests pin its rows (rate_correlation and the rate-based York slope
    # graded). The dated amendments of 8 Oct 2026 are tested in test_p23_amend (its _b1_proto adds them back).
    P["amendments"] = []
    P["site"]["aoi"]["bbox_wsen"] = [LON0, LAT0 + NY * DLAT, LON0 + NX * DLON, LAT0]
    P["truth"]["stations"] = {s: {"lat": la, "lon": lo, "roles": ["rate", "series", "calibration_pool"]}
                              for s, (la, lo) in stations.items()}
    P["runs"] = {"T1": {"config": "synthetic.yaml", "window": ["2021-10-01", "2024-09-30"]}}
    for c in P["comparisons"].values():               # the synthetic run is T1 (a row lists the runs it applies to)
        if isinstance(c, dict) and isinstance(c.get("runs"), list):
            c["runs"] = ["T1"]
    p = tmp_path / "protocol_synthetic_b.yaml"
    p.write_text(yaml.safe_dump(P, sort_keys=False))
    return p


def _dates(d0="2021-10-03", d1="2024-09-25", step=12):
    return [str(np.datetime64(int(d), "D")).replace("-", "") for d in range(V._day(d0), V._day(d1) + 1, step)]


# ------------------------------------------------------------------ sampling and matching

def test_sampling_is_a_median_of_reliable_cells_within_the_radius(tmp_path):
    dates = _dates(step=48)
    tc = np.full((NY, NX), 0.9)
    lat, lon = _cell_centre(36.272, -119.672)
    r, c = _rc(lat, lon)
    extra = np.zeros((len(dates), NY, NX))
    extra[:, r, c + 1] = 5000.0                         # one wild cell: the median ignores it
    cfg = make_product(tmp_path, dates, tcoh=tc, extra=extra)
    prod = V.InsarProduct(cfg)
    rows, cols, d = prod.cells_near(lat, lon, 100.0)
    # 100 m on a ~45 x 55 m grid: about pi 100^2 / (45 x 55) ~ 13 cells
    assert 9 <= rows.size <= 17 and d.max() <= 100.0
    s = V.sample_points(prod, {"P": {"lat": lat, "lon": lon}}, 100.0, [0.4], 5, 2)["P"][0.4]
    assert np.all(np.isfinite(s["series"]))
    # equal to the cell's own value up to the field's gradient across the disc, not pulled by the wild cell
    ts = prod.read_cells(np.array([r]), np.array([c]))[:, 0]
    assert np.nanmax(np.abs(s["series"] - ts)) < 1.5
    # the LOS vector is the per-pixel one at the station (unit length, ground to satellite)
    want = _los(*[g[r, c] for g in _geometry()[:2]])
    assert np.allclose(s["los"], want, atol=0.01) and abs(np.linalg.norm(s["los"]) - 1) < 1e-9
    # below the floor: no value (never filled)
    tc2 = tc.copy()
    tc2[r - 3:r + 4, c - 3:c + 4] = 0.2
    cfg2 = make_product(tmp_path / "b", dates, tcoh=tc2)
    s2 = V.sample_points(V.InsarProduct(cfg2), {"P": {"lat": lat, "lon": lon}}, 100.0, [0.4, 0.1], 5, 2)["P"]
    assert np.all(np.isnan(s2[0.4]["series"])) and s2[0.4]["reliable_cells"] == 0
    assert np.all(np.isfinite(s2[0.1]["series"]))


def test_sampling_on_a_projected_grid_uses_metres(tmp_path):
    cfg = make_product(tmp_path, _dates(step=96), epsg=32611)
    prod = V.InsarProduct(cfg)
    from pyproj import Transformer
    lon, lat = Transformer.from_crs("EPSG:32611", "EPSG:4326", always_xy=True).transform(500000 + 45 * 100.5,
                                                                                         4020000 - 55 * 100.5)
    rows, cols, d = prod.cells_near(lat, lon, 100.0)
    assert 9 <= rows.size <= 17
    assert abs(rows.mean() - 100) < 1.5 and abs(cols.mean() - 100) < 1.5


def test_sampling_needs_independent_radar_pixels(tmp_path, monkeypatch):
    cfg = make_product(tmp_path, _dates(step=96))
    prod = V.InsarProduct(cfg)
    lat, lon = _cell_centre(36.272, -119.672)
    monkeypatch.setattr(prod, "pixel_ids", lambda r, c: (np.zeros(np.size(r), np.int64), "one pixel"))
    s = V.sample_points(prod, {"P": {"lat": lat, "lon": lon}}, 100.0, [0.4], 5, 2)["P"][0.4]
    assert np.all(np.isnan(s["series"])) and s["n_independent"].max() == 1


def test_gnss_epoch_matching_is_plus_minus_three_days():
    days = np.array([100, 101, 102, 110, 111])
    st = {"days": days, "enu": np.column_stack([days * 1.0, days * 0.0, days * 2.0])}
    m, n = V.match_epochs(st, np.array([101, 106, 107, 114, 115]), 3)
    # 101 +-3: 100,101,102 | 106: none (103..109) | 107: 110 | 114: 111 only | 115: none (112..118)
    assert n.tolist() == [3, 0, 1, 1, 0]
    assert m[0, 0] == 101.0 and np.isnan(m[1, 0]) and m[2, 0] == 110.0 and m[3, 0] == 111.0 and np.isnan(m[4, 0])
    m2, n2 = V.match_epochs(st, np.array([107]), 6)          # the pipeline's own tie used +-6 days
    assert n2.tolist() == [4]


# ------------------------------------------------------------------ datum

def test_loo_datum_never_uses_the_scored_station():
    rng = np.random.default_rng(5)
    n = 40
    o_true = np.cumsum(rng.normal(0, 3, n))
    b = {"A": 0.0, "B": 7.0, "C": -4.0, "K": 2.0}
    Y = {s: o_true + b[s] + rng.normal(0, 0.1, n) for s in b}
    Y["B"][:10] = np.nan                                   # joins late: the constants keep the datum continuous
    W = {s: np.ones(n) for s in b}
    o, consts, info = V._loo_datum(["A", "B", "C"], Y, W, n)
    assert np.nanmax(np.abs((o - o.mean()) - (o_true - o_true.mean()))) < 0.3
    assert abs((consts["B"] - consts["A"]) - 7.0) < 0.2 and info["constant_fixed_at"] in ("A", "C")
    # corrupting K changes nothing in K's own datum (K is not a calibration station)
    Y2 = dict(Y, K=Y["K"] + 1000.0)
    o2, _, _ = V._loo_datum(["A", "B", "C"], Y2, W, n)
    assert np.allclose(o, o2, equal_nan=True)
    # with fewer than 2 calibration stations at an epoch there is no value there
    o3, _, info3 = V._loo_datum(["A"], Y, W, n)
    assert np.all(np.isnan(o3)) and "at least 2" in info3["why"]


# ------------------------------------------------------------------ end to end, network design

def _run_b(tmp_path, stations=STATIONS_B, extra=None, gt=None, **kw):
    cfg = make_product(tmp_path / "proj", _dates(), extra=extra)
    proto = b1_protocol(tmp_path, stations)
    gt = gt or FakeGT(stations)
    out = tmp_path / "out"
    res = V.validate(cfg, proto, out, run="T1", cache_dir=tmp_path / "cache", gt=gt, sensitivity=kw.get("sens", False),
                     epoch_coverage=kw.get("cov", False), config_path=tmp_path / "synthetic.yaml", up=kw.get("up"))
    return res, out, gt


def test_network_end_to_end_recovers_known_motion_and_passes(tmp_path):
    res, out, gt = _run_b(tmp_path, sens=True, cov=True)
    rows = res["rows"]
    # ground_truth was asked for the protocol's screening model: piecewise by water year
    assert gt.calls[0][3] == {s: "piecewise" for s in STATIONS_B}
    assert gt.calls[0][4] == ["2022-10-01", "2023-10-01", "2024-10-01", "2025-10-01"]
    assert rows["rate_rmse_dd"]["grade"] == "pass" and rows["rate_rmse_dd"]["metrics"]["rmse"] < 1.0
    assert rows["rate_rmse_dd"]["metrics"]["n"] == 10 * 3                   # 10 pairs x 3 water years
    assert rows["rate_bias_loo"]["grade"] == "pass"
    assert rows["york_slope"]["grade"] == "pass"
    assert rows["rate_correlation"]["grade"] == "pass"
    assert rows["series_rmse"]["grade"] == "pass" and rows["series_rmse"]["metrics"]["median_rmse"] < 2.0
    assert rows["series_correlation"]["grade"] == "pass"
    assert rows["uncertainty_calibration"]["grade"] in ("pass", "marginal", "fail")      # graded, any outcome
    assert rows["tre_vertical"]["grade"] == "not_computed"
    v = res["verdict"]
    assert v["route_rows"] == ["rate_rmse_dd", "rate_bias_loo", "uncertainty_calibration"]
    for name in ("rate_rmse_dd", "rate_bias_loo", "york_slope", "rate_correlation", "series_rmse"):
        assert rows[name].get("ci"), name
    js = json.loads((out / "validation.json").read_text())
    assert js["protocol"]["fixed"]["status"] == "ad_hoc" and js["truth"]["stations"]["AAAA"]["sha256"] == "f" * 64
    assert set(js["sensitivity"]) == {"0.3", "0.5", "0.7"}
    assert js["coverage"]["by_floor"]["0.4"]["share_reliable"] == 1.0
    assert "per_date" in js["coverage"]["by_floor"]["0.4"]
    md = (out / "validation.md").read_text()
    assert "rate_rmse_dd" in md and "ad hoc" in md
    assert (out / "validation_epochs.csv").read_text().count("\n") > 100
    for p in ("rates_scatter.png", "dd_rates_scatter.png", "series_stations.png"):
        assert (out / p).stat().st_size > 1000, p
    # the vertical rows are reported, not graded, and labelled
    assert rows["vertical_secondary"]["grade"] == "report_only"


def test_plate_motion_cancels_only_with_the_per_pixel_los_vector(tmp_path):
    """A pair's GNSS difference projected with each station's own LOS vector
    matches the InSAR; projecting Up only (no LOS vector) leaves the plate
    motion seen through the LOS vector's horizontal part as an error."""
    res, _, _ = _run_b(tmp_path)
    assert res["rows"]["rate_rmse_dd"]["metrics"]["rmse"] < 1.0
    va = res["rows"]["vertical_secondary"]["assumes_vertical"]
    # LOS / cos(incidence) misreads the horizontal plate motion as vertical
    assert va["rate_bias_loo"]["metrics"]["abs_bias"] > 1.0 or va["rate_rmse_dd"]["metrics"]["rmse"] > 1.0


def test_network_vertical_from_d1_on_its_own_grid_and_epochs(tmp_path):
    """D1 Up (its own grid, monthly epochs) against GNSS Up matched at those
    epochs agrees; LOS / cos(incidence) minus D1 Up is reported at each
    station and equals what the geometry predicts."""
    up = make_d1(tmp_path / "vertical_timeseries_decomposed.h5")
    res, out, _ = _run_b(tmp_path, up=up)
    vs = res["rows"]["vertical_secondary"]
    assert vs["grade"] == "report_only"
    d1 = vs["d1_up"]
    assert d1["rate_rmse_dd"]["metrics"]["rmse"] < 1.0 and d1["series_rmse"]["metrics"]["median_rmse"] < 2.0
    assert d1["rate_rmse_dd"]["metrics"]["n"] >= 10
    vob = res["rows"]["vertical_only_bias"]
    assert vob["grade"] == "report_only" and set(vob["stations"]) == set(STATIONS_B)
    inc, az = _geometry()[:2]
    lr = _los(inc[NY // 2, NX // 2], az[NY // 2, NX // 2])
    rlat, rlon = LAT0 + (NY // 2 + 0.5) * DLAT, LON0 + (NX // 2 + 0.5) * DLON
    r_ref = -40.0 + 400.0 * (rlat - 36.25) - 300.0 * (rlon + 119.65)
    wants = []
    for s_, v in vob["stations"].items():
        lat, lon = _cell_centre(*STATIONS_B[s_])
        r, c = _rc(lat, lon)
        l = _los(inc[r, c], az[r, c])
        # LOS/cos - Up relative to the reference pixel: the plate's horizontal motion seen through the
        # difference in horizontal sensitivity, and the reference's vertical scaled by the cosines
        want = ((l[0] - lr[0]) * PLATE[0] + (l[1] - lr[1]) * PLATE[1]) / l[2] + (1 - lr[2] / l[2]) * r_ref
        assert abs(v["difference"] - want) < 0.5, (s_, v, want)
        wants.append(abs(want))
    assert max(wants) > 1.0                                     # a real effect somewhere, measured right


def test_network_planted_error_fails(tmp_path):
    """InSAR at one station sinks 25 mm/yr too fast: the double differences and
    the leave-one-out bias see it."""
    lat, lon = _cell_centre(*STATIONS_B["CCCC"])
    r, c = _rc(lat, lon)
    dates = _dates()
    extra = np.zeros((len(dates), NY, NX), np.float32)
    extra[:, r - 4:r + 5, c - 4:c + 5] = (-25.0 * _day_years(V._days(dates)))[:, None, None]
    res, _, _ = _run_b(tmp_path, extra=extra)
    rows = res["rows"]
    assert rows["rate_rmse_dd"]["grade"] in ("marginal", "fail") and rows["rate_rmse_dd"]["metrics"]["rmse"] > 8
    assert res["verdict"]["route"]["grade"] in ("fail", "marginal")
    errs = {p["pair"]: p["error"] for p in rows["rate_rmse_dd"]["pairs"]}
    assert all(abs(e) > 15 for k, es in errs.items() if "CCCC" in k for e in es.values())
    assert all(abs(e) < 2 for k, es in errs.items() if "CCCC" not in k for e in es.values())


def test_missing_station_is_reported_not_dropped_silently(tmp_path):
    st = dict(STATIONS_B, ZZZZ=(36.26, -119.63))
    gt = FakeGT(STATIONS_B)                        # ZZZZ has no GNSS
    res, out, _ = _run_b(tmp_path, stations=st, gt=gt)
    assert "ZZZZ" in res["truth"]["not_loaded"]
    assert "ZZZZ" in (out / "validation.md").read_text()


# ------------------------------------------------------------------ end to end, reference-station design

STATIONS_A = {"VVVV": (36.262, -119.652), "RRRR": (36.236, -119.632), "NNNN": (36.270, -119.612)}


def a_protocol(tmp_path: Path) -> Path:
    P = yaml.safe_load(PA.read_text())
    P["site"]["aoi"]["bbox_wsen"] = [LON0, LAT0 + NY * DLAT, LON0 + NX * DLON, LAT0]
    P["window"] = {"start": "2021-10-01", "end": "2024-09-30"}
    P["runs"] = {"T1": {"config": "synthetic_a.yaml", "window": ["2021-10-01", "2024-09-30"]}}
    P["truth"]["stations"] = {
        "VVVV": {"lat": STATIONS_A["VVVV"][0], "lon": STATIONS_A["VVVV"][1], "role": "validation"},
        "RRRR": {"lat": STATIONS_A["RRRR"][0], "lon": STATIONS_A["RRRR"][1], "role": "reference"},
        "NNNN": {"lat": STATIONS_A["NNNN"][0], "lon": STATIONS_A["NNNN"][1], "role": ["reference_check", "null_test"]}}
    P["reference"]["primary"]["station"] = "RRRR"
    P["reference"]["primary"]["fallback_if_unusable"] = ("RRRR is unusable when it has an InSAR value at fewer than "
                                                         "80% of the compared epochs. Then NNNN.")
    P["reference"]["primary"]["source_order"] = ["GA xlsx VVVV - GA xlsx RRRR", "NGL VVVV - NGL RRRR"]
    P["reference"]["sensitivity"] = ["same, with NNNN as the reference station"]
    C = P["comparisons"]
    C["ev_main"] = copy.deepcopy(C["ca19_lw902"])
    C["ev_main"]["what"] = "VVVV cumulative LOS change over the event, double-differenced to RRRR"
    C["ev_main"]["epochs"] = "t0 = last T1 date <= 2022-05-01 with a GA epoch within +-3 d; t1 = first T1 date >= 2023-03-01 with a GA epoch within +-3 d"
    C["ev_series"] = copy.deepcopy(C["ca19_series"])
    C["ev_series"]["what"] = "VVVV matched-epoch LOS series, double-differenced to RRRR"
    C["ev_series"]["epochs"] = "every T1 date in 2021-10-01 .. 2024-09-30 with a GA epoch within +-3 d"
    for k in ("ca19_lw902", "ca19_outage", "ca19_series"):
        del C[k]
    for c in C.values():                              # the synthetic run is T1 (a row lists the runs it applies to)
        if isinstance(c, dict) and isinstance(c.get("runs"), list):
            c["runs"] = ["T1"]
    rg = P["verdict"]["rows_graded"]
    P["verdict"]["rows_graded"] = ["ev_main", "ev_series"] + [r for r in rg if not r.startswith("ca19")]
    P["verdict"]["route_trustworthy_for_longwall"] = P["verdict"]["route_trustworthy_for_longwall"].replace(
        "ca19_lw902", "ev_main")
    p = tmp_path / "protocol_synthetic_a.yaml"
    p.write_text(yaml.safe_dump(P, sort_keys=False))
    return p


def _event(days):
    """An event at VVVV: 300 mm of subsidence between 2022-06 and 2023-02."""
    t0, t1 = V._day("2022-06-01"), V._day("2023-02-01")
    f = np.clip((np.asarray(days, float) - t0) / (t1 - t0), 0, 1)
    return -300.0 * f


def _run_a(tmp_path, insar_scale=1.0, mask_event=False, spans=None, flag_event=False, points=None, up=None,
           outlines=None):
    dates = _dates()
    days = V._days(dates)
    lat, lon = _cell_centre(*STATIONS_A["VVVV"])
    r, c = _rc(lat, lon)
    inc, az = _geometry()[:2]
    lu = _los(inc[r, c], az[r, c])[2]
    extra = np.zeros((len(dates), NY, NX), np.float32)
    extra[:, r - 4:r + 5, c - 4:c + 5] = (insar_scale * lu * _event(days))[:, None, None]
    tc = np.full((NY, NX), 0.9)
    if mask_event:
        tc[r - 4:r + 5, c - 4:c + 5] = 0.1
    cfg = make_product(tmp_path / "proj", dates, tcoh=tc, extra=extra)
    gt = FakeGT(STATIONS_A, overrides={"VVVV": lambda d, enu: enu + np.column_stack(
        [np.zeros(d.size), np.zeros(d.size), _event(d)])}, spans=spans)
    if flag_event:
        import rasterio
        fl = np.zeros((NY, NX), np.float32)
        fl[r - 4:r + 5, c - 4:c + 5] = 1.0
        (tmp_path / "proj" / "deliverables").mkdir(parents=True, exist_ok=True)
        with rasterio.open(tmp_path / "proj" / "deliverables" / "quality_aliasing_risk.tif", "w", driver="GTiff",
                           height=NY, width=NX, count=1, dtype="float32") as dst:
            dst.write(fl, 1)
    if points:
        gt.points = points
    out = tmp_path / "out"
    res = V.validate(cfg, a_protocol(tmp_path), out, run="T1", cache_dir=tmp_path / "cache", gt=gt,
                     sensitivity=False, epoch_coverage=False, points="units.csv" if points else None, up=up,
                     outlines=outlines)
    return res, out, gt


def _event_outline(tmp_path):
    """A small panel outline over the event (its stable ring is 1-3 km outside it)."""
    lat, lon = _cell_centre(*STATIONS_A["VVVV"])
    h = 0.003
    p = tmp_path / "outline.geojson"
    p.write_text(json.dumps({"type": "FeatureCollection", "features": [{
        "type": "Feature", "properties": {"name": "LW999"}, "geometry": {"type": "Polygon", "coordinates": [[
            [lon - h, lat - h], [lon + h, lat - h], [lon + h, lat + h], [lon - h, lat + h], [lon - h, lat - h]]]}}]}))
    return p


def test_reference_design_end_to_end(tmp_path):
    res, out, gt = _run_a(tmp_path)
    rows = res["rows"]
    # ground_truth screened the event station with the local model, the others linear
    assert gt.calls[0][3] == {"VVVV": "local", "RRRR": "linear", "NNNN": "linear"}
    assert res["detail"]["reference_choice"]["chosen"] == "RRRR"
    ev = rows["ev_main"]
    assert ev["grade"] == "pass" and abs(ev["metrics"]["err"]) < 5 and ev["metrics"]["truth"] < -150
    assert rows["ev_series"]["grade"] == "pass" and rows["ev_series"]["metrics"]["rmse"] < 5
    assert rows["stable_stations"]["grade"] == "report_only"
    nn = rows["stable_stations"]["stations"]["NNNN"]
    assert abs(nn["insar_rate"] - nn["gnss_rate"]) < 1.5
    # sensitivity with the other reference station is reported
    assert "sensitivity_reference_NNNN" in rows
    for name in ("tahmoor_w_units", "appin_peak_ratio", "stable_ring"):
        assert rows[name]["grade"] == "not_computed" and rows[name]["why"]
    uf = rows["unflagged_failure_rate"]
    assert uf["metrics"]["rate"] == 0.0 and uf["grade"] == "report_only"      # no F2 flag yet: does not grade
    assert res["verdict"]["route"]["grade"] == "not_reached"                  # T1 rows missing
    assert (out / "events_scatter.png").exists() and (out / "series_points.png").exists()
    # the GA workbook could not be had (offline): said so, and NGL (ranked second) was used
    assert "GA CEMP workbook could not be had" in " ".join(res["warnings"])
    assert rows["ev_main"]["comparison"]["source"] == "ngl"


def test_reference_design_underestimated_event_fails(tmp_path):
    """C-band missing half the bowl (aliasing) is a failure, and with no flag
    it is an unflagged failure."""
    res, _, _ = _run_a(tmp_path, insar_scale=0.5)
    rows = res["rows"]
    assert rows["ev_main"]["grade"] == "fail"
    assert rows["unflagged_failure_rate"]["metrics"]["unflagged_failures"] >= 1


def test_masked_point_is_honest_not_a_failure(tmp_path):
    res, _, _ = _run_a(tmp_path, insar_scale=0.5, mask_event=True)
    rows = res["rows"]
    assert rows["ev_main"]["grade"] == "not_testable"
    assert rows["ev_main"]["comparison"]["truth"] < -150         # the truth still exists; the InSAR does not
    uf = rows["unflagged_failure_rate"]["metrics"]
    assert uf["unflagged_failures"] == 0 and uf["share_masked"] > 0.9


def test_f2_flag_makes_the_wrong_event_honest_and_grades(tmp_path):
    """With F2's flag on the event cells, the under-measured event is flagged:
    no unflagged failure, the row grades (F2 grades; the old flag would not),
    and the flagged share is reported beside it. (Protocol A's amendment
    2026-10-08/p23-2: the row passes only with the flag shown selective on
    stable ground, so the event's panel outline is given; tests/test_protdec.py.)"""
    res, _, _ = _run_a(tmp_path, insar_scale=0.5, flag_event=True, outlines=_event_outline(tmp_path))
    uf = res["rows"]["unflagged_failure_rate"]
    assert "F2" in uf["flag_layer"] and uf["grade"] == "pass"
    assert uf["flag_selectivity"]["stable_flagged_share"] == 0.0
    assert uf["metrics"]["unflagged_failures"] == 0 and uf["metrics"]["share_flagged"] > 0.9
    assert uf["metrics"]["flag_false_alarm_share"] is not None


def test_digitised_points_are_double_differenced_to_the_reference_up(tmp_path):
    """A digitised unit (Up only) in the reference station's column (same
    incidence, so LOS / cos(incidence) carries no plate leakage): its changes
    since its first epoch, minus the reference station's GNSS Up change,
    against the InSAR vertical change double-differenced to the reference."""
    lat_u, lon_u = 36.252, STATIONS_A["RRRR"][1]
    lat_u, lon_u = _cell_centre(lat_u, lon_u)
    lat_r, lon_r = _cell_centre(*STATIONS_A["RRRR"])
    assert _rc(lat_u, lon_u)[1] == _rc(lat_r, lon_r)[1]
    days = V._days(["20211215", "20220601", "20230110", "20230801", "20240501"])
    pt = {"id": "UNIT1", "kind": "survey", "lat": lat_u, "lon": lon_u, "h_m": None, "source": "digitised", "role": "validation"}
    df = pd.DataFrame({"time": pd.to_datetime(np.array(days, "datetime64[D]")),
                       "value_mm": _vertical(lat_u, lon_u, days) + 1234.0, "sigma_mm": 5.0, "component": "u"})
    res, _, gt = _run_a(tmp_path, points=[(pt, df)])
    row = res["rows"]["tahmoor_w_units"]
    assert ("points", "units.csv") in gt.calls
    assert row["comparisons"] == 4 and row["metrics"]["rmse"] < 3.0
    assert row["grade"] == "report_only" and "F2" in row["why"]          # no F2 flag yet: reported, not graded


def test_reference_design_d1_up(tmp_path):
    up = make_d1(tmp_path / "vertical_timeseries_decomposed.h5")
    res, _, _ = _run_a(tmp_path, up=up)
    d = res["rows"]["decomposition_up_at_stations"]
    assert d["grade"] == "report_only"
    v = d["stations"]["VVVV"]
    # D1 here has no event (the synthetic D1 is the background field only): the event shows as the difference
    assert v["change_gnss_mm"] < -250 and abs(v["change_insar_mm"]) < 50
    n = d["stations"]["NNNN"]
    assert n["rmse_after_offset_mm"] < 3.0


def test_reference_falls_back_in_the_protocol_order_on_coverage_alone(tmp_path):
    # RRRR's GNSS stops early: GNSS at < 80% of the compared epochs -> NNNN, the next in the protocol's order
    res, _, _ = _run_a(tmp_path / "a", spans={"RRRR": ("2021-09-01", "2022-06-30")})
    ch = res["detail"]["reference_choice"]
    assert ch["chosen"] == "NNNN" and not ch["candidates"][0]["usable"]
    assert res["rows"]["ev_main"]["grade"] == "pass" and res["rows"]["ev_main"]["comparison"]["reference"] == "NNNN"
    # neither usable: every row says why, none passes
    res2, out2, _ = _run_a(tmp_path / "b", spans={"RRRR": ("2021-09-01", "2022-06-30"),
                                                  "NNNN": ("2021-09-01", "2022-06-30")})
    assert res2["detail"]["reference_choice"]["chosen"] is None
    for name in ("ev_main", "ev_series"):
        assert res2["rows"][name]["grade"] == "not_testable"
        assert "no reference station is usable" in res2["rows"][name]["why"]
    assert "no reference station is usable" in (out2 / "validation.md").read_text()


def test_run_is_the_pipeline_stage_entry(tmp_path, monkeypatch):
    """pipeline.run_validation (V3) calls validation.run(cfg): the protocol from
    cfg.validation_protocol, the run found from the config's file name, the
    output under <project>/validation/protocol_<id>_<run>."""
    cfg = make_product(tmp_path / "proj", _dates(step=24))
    proto = b1_protocol(tmp_path)
    cfg.validation_protocol = str(proto)
    cfg._source_path = str(tmp_path / "configs" / "synthetic.yaml")
    monkeypatch.setattr(V, "_ground_truth", lambda: FakeGT(STATIONS_B))
    out = V.run(cfg)
    assert out == tmp_path / "proj" / "validation" / "protocol_B1_T1"
    js = json.loads((out / "validation.json").read_text())
    assert js["protocol"]["run"] == "T1" and js["rows"]["rate_rmse_dd"]["grade"] == "pass"
    cfg.validation_protocol = ""
    with pytest.raises(ValueError):
        V.run(cfg)


# ------------------------------------------------------------------ CLI

def test_cli_validate_dispatches_and_refuses_while_running(tmp_path, monkeypatch):
    from insar_pipeline import cli, runs
    called = {}

    def fake_main(argv, cfg=None):
        called["argv"], called["cfg"] = argv, cfg
        return 0

    cfgfile = tmp_path / "c.yaml"
    cfgfile.write_text("x: 1\n")
    fake_cfg = SimpleNamespace(project_dir=tmp_path / "proj")
    monkeypatch.setattr(cli, "load_cfg", lambda p: fake_cfg)
    monkeypatch.setattr(V, "main", fake_main)
    monkeypatch.setattr(runs, "state", lambda p: "done")
    with pytest.raises(SystemExit) as e:
        cli.main(["validate", str(cfgfile), "--protocol", str(PB1), "--out", str(tmp_path / "o")])
    assert e.value.code == 0
    assert called["argv"] == [str(cfgfile), "--protocol", str(PB1), "--out", str(tmp_path / "o")]
    assert called["cfg"] is fake_cfg
    monkeypatch.setattr(runs, "state", lambda p: "running")
    called.clear()
    with pytest.raises(SystemExit) as e:
        cli.main(["validate", str(cfgfile), "--protocol", str(PB1)])
    assert e.value.code == 3 and not called
    with pytest.raises(SystemExit) as e:
        cli.main(["validate"])
    assert e.value.code == 2


# ------------------------------------------------------------------ independent review, 8 Oct 2026

def _unit_points(days_iso, lat=36.252):
    lat_u, lon_u = _cell_centre(lat, STATIONS_A["RRRR"][1])
    days = V._days(days_iso)
    pt = {"id": "UNIT1", "kind": "survey", "lat": lat_u, "lon": lon_u, "h_m": None, "source": "digitised",
          "role": "validation"}
    df = pd.DataFrame({"time": pd.to_datetime(np.array(days, "datetime64[D]")),
                       "value_mm": _vertical(lat_u, lon_u, days) + 50.0, "sigma_mm": 5.0, "component": "u"})
    return [(pt, df)]


def test_a_run_after_the_matched_series_still_gets_its_reference(tmp_path, monkeypatch):
    """A3 and D1 start 2019-11, after CA19's GA record: the matched-series row
    has no date in such a run. The reference is then judged on the run's own
    dates (it was 'no reference station is usable', which left the Tahmoor
    West units - a row A3 is meant to grade - with no datum at all); the CA19
    rows the protocol does not run on this run say so."""
    orig = a_protocol

    def before_this_run(tp):
        p = orig(tp)
        P = yaml.safe_load(p.read_text())
        C = P["comparisons"]
        C["ev_series"]["epochs"] = "every T1 date in 2019-01-01 .. 2019-07-13 with a GA epoch within +-3 d"
        C["ev_main"]["runs"] = ["A1", "A2"]                       # as ca19_lw902: not run on this run
        p.write_text(yaml.safe_dump(P, sort_keys=False))
        return p
    monkeypatch.setitem(globals(), "a_protocol", before_this_run)      # _run_a builds its protocol with it
    res, out, _ = _run_a(tmp_path, points=_unit_points(["20211215", "20220601", "20230110", "20240501"]))
    ch = res["detail"]["reference_choice"]
    assert ch["chosen"] == "RRRR" and ch["rows_not_in_this_run"][0]["row"] == "ev_series"
    assert ch["candidates"][0]["checks"][0]["row"] == "this run's dates in the window"
    units = res["rows"]["tahmoor_w_units"]
    assert units["metrics"]["n"] == 3 and units["metrics"]["rmse"] < 3.0
    ev = res["rows"]["ev_main"]
    assert ev["grade"] == "not_testable" and "not on T1" in ev["why"]
    assert res["verdict"]["route"]["grade"] == "not_reached"


def test_a_gap_at_the_reference_is_no_comparison_not_an_honest_mask(tmp_path):
    """The honest-or-flagged share counts a point as honest when ITS cells are
    masked. A gap in the reference station's InSAR is no comparison at all:
    it was counted as masked, which lowered the unflagged-failure rate."""
    dates = _dates()
    days = V._days(dates)
    lat, lon = _cell_centre(*STATIONS_A["VVVV"])
    r, c = _rc(lat, lon)
    rl, rc_ = _rc(*_cell_centre(*STATIONS_A["RRRR"]))
    inc, az = _geometry()[:2]
    lu = _los(inc[r, c], az[r, c])[2]
    extra = np.zeros((len(dates), NY, NX), np.float32)
    extra[:, r - 4:r + 5, c - 4:c + 5] = (0.5 * lu * _event(days))[:, None, None]      # half the event: wrong
    gap = (days >= V._day("2023-02-20")) & (days <= V._day("2023-03-20"))     # around t1; still >= 80% coverage
    extra[np.ix_(gap, np.arange(rl - 4, rl + 5), np.arange(rc_ - 4, rc_ + 5))] = np.nan  # the reference's cells
    cfg = make_product(tmp_path / "proj", dates, extra=extra)
    gt = FakeGT(STATIONS_A, overrides={"VVVV": lambda d, enu: enu + np.column_stack(
        [np.zeros(d.size), np.zeros(d.size), _event(d)])})
    res = V.validate(cfg, a_protocol(tmp_path), tmp_path / "out", run="T1", cache_dir=tmp_path / "cache", gt=gt,
                     sensitivity=False, epoch_coverage=False)
    ev = res["rows"]["ev_main"]
    assert ev["grade"] == "not_testable" and "RRRR has no InSAR value" in ev["why"]
    assert ev["comparison"]["masked"] is False
    uf = res["rows"]["unflagged_failure_rate"]["metrics"]
    assert res["detail"]["reference_choice"]["chosen"] == "RRRR"
    assert uf["share_masked"] == 0.0 and uf["not_compared"] == int(gap.sum()) + 1     # the gap's epochs + the event
    assert uf["n"] + uf["not_compared"] == len(res["detail"]["comparisons"])


def test_digitised_epochs_outside_the_run_are_not_counted_honest(tmp_path):
    """Tahmoor West units run to 2023-06, A1 to 2022-12: epochs after the run
    (or before it) have no InSAR value because the run never looked, not
    because the cell is masked."""
    pts = _unit_points(["20211215", "20220601", "20230110", "20240501", "20250101", "20251201"])
    res, _, _ = _run_a(tmp_path, points=pts)
    units = res["rows"]["tahmoor_w_units"]
    assert units["comparisons"] == 3 and units["metrics"]["n"] == 3     # 2025 epochs lie outside the window
    pc = [c for c in res["detail"]["comparisons"] if c["kind"] == "point_epoch"]
    assert not any(c["masked"] for c in pc)


def test_an_edited_parent_protocol_is_warned(tmp_path):
    """B-2 takes sampling, matching and statistics from B-1: an edited B-1
    changes B-2's result while B-2's own hash still reads 'fixed'."""
    parent = tmp_path / "protocol_B1.yaml"                     # the fixed name, edited content
    parent.write_text(b1_protocol(tmp_path).read_text())
    P = yaml.safe_load(parent.read_text())
    secs = ["insar_input", "sampling", "reliability", "decomposition", "epoch_matching", "gnss_screening",
            "statistics_reported"]
    child = {k: v for k, v in P.items() if k not in secs}
    child["protocol"] = "B1X"
    child["inherits"] = {"file": str(parent), "version": 1, "sections": secs}
    cp = tmp_path / "child.yaml"
    cp.write_text(yaml.safe_dump(child, sort_keys=False))
    cfg = make_product(tmp_path / "proj", _dates(step=24))
    res = V.validate(cfg, cp, tmp_path / "out", run="T1", cache_dir=tmp_path / "cache", gt=FakeGT(STATIONS_B),
                     sensitivity=False, epoch_coverage=False)
    assert res["protocol"]["inherited"]["fixed"]["status"] == "changed"
    assert any("inherits from (protocol_B1.yaml) differs" in w for w in res["warnings"])


def test_cli_validate_says_when_the_config_does_not_load(tmp_path, capsys):
    from insar_pipeline import cli
    assert cli.validate_main([str(tmp_path / "missing.yaml"), "--protocol", str(PB1)]) == 1
    assert "does not load" in capsys.readouterr().out


def test_common_mode_removed_rate_statistics_are_reported_beside_the_pooled_ones(tmp_path):
    """Corcoran's whole box sinks by water year (-250, +40, -60 mm/yr). The
    leave-one-out datum brings that motion from the other stations' GNSS into
    the InSAR, so pooled water-year rates line up near slope 1 even when the
    InSAR reads only 70% of every motion. The same York slope and r with each
    water year's mean removed show it; they are reported, not graded."""
    dates = _dates()
    days = V._days(dates)
    inc, az, lat, lon = _geometry()
    L = _los(inc, az)
    wy = {2022: -250.0, 2023: 40.0, 2024: -60.0}

    def common(d):
        d = np.asarray(d, float)
        out = np.zeros(d.size)
        for y, r in wy.items():
            a, b = V._day(f"{y - 1}-10-01"), V._day(f"{y}-10-01")
            out += r * np.clip(d - a, 0, b - a) / 365.25
        return out

    t = _day_years(days)[:, None, None]
    rate = -40.0 + 400.0 * (lat - 36.25) - 300.0 * (lon + 119.65)
    U = rate[None] * t + 6.0 * np.sin(2 * np.pi * days / 365.25)[:, None, None]
    base = L[0][None] * PLATE[0] * t + L[1][None] * PLATE[1] * t + L[2][None] * U
    cpart = L[2][None] * common(days)[:, None, None]
    whole = base + cpart
    rel = whole - whole[:, NY // 2, NX // 2][:, None, None]
    base_rel = base - base[:, NY // 2, NX // 2][:, None, None]
    extra = (0.7 * rel - base_rel).astype(np.float32)         # the product then holds 70% of every motion
    gt = FakeGT(STATIONS_B, overrides={s: (lambda d, enu: enu + np.column_stack(
        [np.zeros(d.size), np.zeros(d.size), common(d)])) for s in STATIONS_B})
    res, out, _ = _run_b(tmp_path, extra=extra, gt=gt)
    ys = res["rows"]["york_slope"]
    cmr = ys["common_mode_removed"]
    # leave-one-out with n = 5 stations: the demeaned slope is (n x 0.7 - 1) / (n - 1) = 0.625
    assert cmr["n"] >= 10 and abs(cmr["york_slope"] - 0.625) < 0.03
    assert ys["metrics"]["slope"] > cmr["york_slope"] + 0.15            # the pooled slope hides it
    assert res["rows"]["rate_correlation"]["common_mode_removed"] == cmr
    assert "With each water year's mean removed" in (out / "validation.md").read_text()
