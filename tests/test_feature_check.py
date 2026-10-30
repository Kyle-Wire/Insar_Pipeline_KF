"""Long-pair feature check (22 Oct 2026): pairing by days, the yearly least
squares, and recovery of a planted onset of movement from a synthetic stack -
including the aliasing of fast movement on ~1-year pairs that the docs describe."""
from pathlib import Path

import numpy as np
import pytest

from insar_pipeline import feature_check as fc
from insar_pipeline import local_change as lc


def _vrt(path, h, w, dt):
    Path(str(path) + ".vrt").write_text(f'<VRTDataset rasterXSize="{w}" rasterYSize="{h}">'
                                        f'<VRTRasterBand dataType="{dt}" band="1"/></VRTDataset>')


def _dates(start="2019-01-04", years=6, step=12):
    t = np.arange(np.datetime64(start), np.datetime64(start) + int(365.25 * years), step).astype("datetime64[D]")
    return [f"{d:%Y%m%d}" for d in t.tolist()]


def test_pairs_by_days_matches_within_the_tolerance():
    dates = _dates(years=2)
    pairs = fc.pairs_by_days(dates, 182, tol=18)
    t = np.array([fc._ordinal(d) for d in dates])
    assert pairs and all(abs(t[j] - t[i] - 182) <= 18 for i, j in pairs)
    assert all(j > i for i, j in pairs)
    assert fc.pairs_by_days(dates[:3], 365) == []


def test_yearly_velocity_recovers_piecewise_rates_exactly():
    dates = _dates(years=5)
    pairs = fc.pairs_by_days(dates, 182)
    A, years = fc.year_design(dates, pairs)
    truth = np.array([0.0, 0.0, -10.0, -10.0, -4.0, -4.0])[:len(years)]
    d = A @ truth
    r = fc.yearly_velocity(A, d, np.arange(len(pairs)) % 7)
    ok = np.isfinite(r["v"])
    assert np.allclose(r["v"][ok], truth[ok], atol=1e-6)
    assert np.all(r["se"][ok] < 1e-6)


def test_a_year_covered_only_by_pairs_into_an_unsolved_year_reads_unknown_not_zero():
    """Review fc-1 (22 Oct 2026): 2021's only pairs run on into 2022, which has too
    little cover to solve. Dropping them left 2021 with no pair, and least squares
    printed 0.0 mm/yr - "stable". It must read unknown; covered years stay exact."""
    dates = [d for d in _dates(start="2019-01-04", years=5) if d < "20240101"]
    six = fc.pairs_by_days(dates, 182)
    inside = [(i, j) for i, j in six if dates[i][:4] == dates[j][:4] and dates[i][:4] in ("2019", "2020", "2023")]
    straddle = [(i, j) for i, j in six if "20210701" <= dates[i] <= "20210831"][:3]
    pairs = inside + straddle
    A, years = fc.year_design(dates, pairs)
    assert years == [2019, 2020, 2021, 2022, 2023]
    assert A[len(inside):, 2].sum() >= 0.5 and A[:, 3].sum() < 0.5           # 2021 "covered", 2022 not
    truth = np.array([0.0, -3.0, -10.0, -10.0, -4.0])
    r = fc.yearly_velocity(A, A @ truth, np.arange(len(pairs)) % 7)
    assert np.isnan(r["v"][2]) and np.isnan(r["v"][3])                      # was 0.0 and NaN
    assert np.allclose(r["v"][[0, 1, 4]], truth[[0, 1, 4]], atol=1e-6)
    assert np.isnan(r["se"][2]) and r["pairs_used"] == len(inside)


def test_one_split_shared_by_two_years_solves_neither():
    """Two years seen only through pairs that split the same way over both (rank
    deficient): lstsq would print the minimum-norm split (-5, -5 here). Neither
    year is determined; the third year is."""
    A = np.array([[0.5, 0.5, 0.0], [0.5, 0.5, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
    r = fc.yearly_velocity(A, A @ np.array([-10.0, 0.0, -4.0]), np.arange(4))
    assert np.isnan(r["v"][:2]).all() and np.isclose(r["v"][2], -4.0)
    assert r["groups_per_year"].tolist() == [0, 0, 2]


def test_dropping_a_year_can_leave_the_next_one_short_in_turn():
    """Year 2 is short (0.2 yr), so the pair reaching it goes; that leaves year 1
    with 0.3 yr, so it goes; that leaves year 0 with 0.3 yr. Nothing is solved,
    and no pair is counted as used (the original solved years 0 and 1 from 0.3 yr)."""
    A = np.array([[0.3, 0.0, 0.0], [0.3, 0.3, 0.0], [0.0, 0.3, 0.2]])
    r = fc.yearly_velocity(A, A @ np.array([-5.0, -5.0, -5.0]), np.arange(3))
    assert np.isnan(r["v"]).all() and r["pairs_used"] == 0 and np.isnan(r["se"]).all()


def test_no_usable_pair_gives_no_velocity():
    dates = ["20200101", "20200701", "20210701", "20220101"]
    A, _ = fc.year_design(dates, [(1, 2)])                                  # one pair, 2020-07..2021-07
    r = fc.yearly_velocity(A, np.array([-20.0]), np.array([0]))
    assert np.isnan(r["v"]).all() and r["pairs_used"] == 0                 # was 0.0 for 2020
    A0, y0 = fc.year_design([], [])
    assert A0.shape == (0, 0) and y0 == []


def test_yearly_error_holds_its_false_alarm_rate_on_6_month_pairs():
    """Review fc-2 (22 Oct 2026): 10 jackknife blocks of ~1 year left each year's
    error on 1-2 blocks, so |v/se| >= 2 passed pure noise in 7.9% of years in this
    set-up with the 22 Oct code (nominal 4.6%). With ~3-month blocks it holds.
    Null: 10.5 years of 12-day dates, 3 mm date noise, 1 mm pair noise, no
    movement."""
    t = np.arange(np.datetime64("2016-01-05"), np.datetime64("2026-06-20"), 12).astype("datetime64[D]")
    dates = [f"{d:%Y%m%d}" for d in t.tolist()]
    pairs = fc.pairs_by_days(dates, 182.0)
    A, _ = fc.year_design(dates, pairs)
    import inspect
    n_groups = inspect.signature(fc.check_area).parameters["n_groups"].default
    groups = np.arange(len(pairs)) * n_groups // len(pairs)                 # as check_area: start-date order
    rng = np.random.default_rng(1)
    hits = years = solved = 0
    for _ in range(150):
        nd = rng.normal(0, 3.0, len(dates))
        d = np.array([nd[j] - nd[i] for i, j in pairs]) + rng.normal(0, 1.0, len(pairs))
        r = fc.yearly_velocity(A, d, groups)
        ok = np.isfinite(r["v"]) & np.isfinite(r["se"])
        hits += int((np.abs(r["v"][ok] / r["se"][ok]) >= 2).sum()); years += int(ok.sum())
        solved += int(np.isfinite(r["v"]).sum())
    assert years == solved                                                  # every solved year has an error
    assert hits / years < 0.065, hits / years


def _stack(root, dates, motion_mm, rlks=4, h=60, w=60, seed=0):
    """Merged SLCs on a small frame: a stable random scene plus noise; pixels within
    120 m of the centre move by motion_mm(date) (mm LOS, + towards the satellite)."""
    rng = np.random.default_rng(seed)
    M = Path(root) / "stack" / "merged"
    W = w * rlks
    yy, xx = np.mgrid[0:h, 0:w]
    lat = 45.10 + yy * 14.0 / 111320.0
    lon = -100.05 + xx * 16.0 / (111320.0 * np.cos(np.radians(45.10)))
    (M / "geom_reference").mkdir(parents=True)
    for name, a in (("lat", lat), ("lon", lon)):
        f = M / "geom_reference" / f"{name}.rdr"; a.astype(np.float64).tofile(f); _vrt(f, h, w, "Float64")
    c0 = (45.10 + 30 * 14.0 / 111320.0, -100.05 + 30 * 16.0 / (111320.0 * np.cos(np.radians(45.10))))
    d = np.hypot((lat - c0[0]) * 111320.0, (lon - c0[1]) * 111320.0 * np.cos(np.radians(45.10)))
    feat = np.repeat(d <= 120.0, rlks, axis=1)
    base = rng.standard_normal((h, W)) + 1j * rng.standard_normal((h, W))
    for k, dt in enumerate(dates):
        (M / "SLC" / dt).mkdir(parents=True)
        s = base + 0.3 * (rng.standard_normal((h, W)) + 1j * rng.standard_normal((h, W)))
        s = np.where(feat, s * np.exp(-1j * motion_mm(dt) / lc.RAD2MM), s)      # angle(a conj b) x RAD2MM = d_b - d_a
        f = M / "SLC" / dt / f"{dt}.slc.full"; s.astype(np.complex64).tofile(f); _vrt(f, h, W, "CFloat32")
    return fc.Stack(Path(root) / "stack", rlks), c0


@pytest.mark.parametrize("rate", [-8.0, -20.0])
def test_a_planted_onset_is_recovered_year_by_year(tmp_path, rate):
    dates = _dates(start="2019-01-04", years=6)
    stack, c0 = _stack(tmp_path, dates, lambda d: (fc._ordinal(d) - fc._ordinal("20220101")) / 365.25 * rate
                       if d > "20220101" else 0.0)
    area = {"name": "feature", "lat": c0[0], "lon": c0[1], "radius_m": 120.0, "ring_m": (250.0, 400.0)}
    r = fc.check_area(stack, area, separations=(182.0, 365.0))
    six = r["separations"]["182"]
    v = dict(zip(six["years"], six["velocity_mm_yr"]))
    assert all(abs(v[y]) < 1.5 for y in (2019, 2020, 2021) if v[y] is not None)
    assert all(abs(v[y] - rate) < 1.5 for y in (2022, 2023) if v[y] is not None)
    one = dict(zip(r["separations"]["365"]["years"], r["separations"]["365"]["velocity_mm_yr"]))
    if abs(rate) < 13:                                      # inside +-lambda/4 per year: both agree
        assert abs(one[2023] - rate) < 1.5
    else:                                                   # beyond it the 1-year pairs alias - the documented limit
        assert abs(one[2023] - rate) > 5


@pytest.mark.parametrize("start, end", [("20190101", "20190301"), ("20300101", "20300301")])
def test_cli_writes_its_record_when_a_separation_has_no_pairs(tmp_path, monkeypatch, capsys, start, end):
    """Review fc-3 (22 Oct 2026): over a span shorter than the separation there are
    no pairs, the coherence medians are None and the print line crashed before
    the JSON was written. A span holding no date at all crashed earlier, in
    year_design."""
    import importlib.util
    import json
    from types import SimpleNamespace
    spec = importlib.util.spec_from_file_location("run_feature_check",
                                                  Path(__file__).resolve().parents[1] / "scripts" / "run_feature_check.py")
    rfc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rfc)
    dates = _dates(start="2019-01-04", years=1)
    _, c0 = _stack(tmp_path, dates, lambda d: 0.0)
    monkeypatch.setattr(rfc, "load_config", lambda p: SimpleNamespace(stack_dir=tmp_path / "stack", stack_range_looks=4,
                                                                      project_dir=tmp_path))
    out = tmp_path / "fc.json"
    assert rfc.main(["cfg.yaml", "--area", f"spot={c0[0]},{c0[1]},120,250,400", "--start", start,
                     "--end", end, "--out", str(out)]) == 0
    rec = json.loads(out.read_text())["areas"]["spot"]["separations"]
    assert rec["182"]["pairs"] == rec["365"]["pairs"] == 0 and rec["182"]["velocity_mm_yr"] is None
    assert "coherence feature n/a" in capsys.readouterr().out


def test_cli_records_the_jackknife_block_count(tmp_path, monkeypatch):
    """Review of the F8 fix (23 Oct 2026): the default went from 10 to 40 blocks, so a
    re-run must say which it used; --groups sets it and must reach check_area."""
    import importlib.util
    import json
    from types import SimpleNamespace
    spec = importlib.util.spec_from_file_location("run_feature_check",
                                                  Path(__file__).resolve().parents[1] / "scripts" / "run_feature_check.py")
    rfc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rfc)
    dates = _dates(start="2019-01-04", years=1)
    _, c0 = _stack(tmp_path, dates, lambda d: 0.0)
    monkeypatch.setattr(rfc, "load_config", lambda p: SimpleNamespace(stack_dir=tmp_path / "stack", stack_range_looks=4,
                                                                      project_dir=tmp_path))
    args = ["cfg.yaml", "--area", f"spot={c0[0]},{c0[1]},120,250,400", "--sep-days", "182"]
    for extra, want in (([], 40), (["--groups", "12"], 12)):
        out = tmp_path / f"fc_{want}.json"
        assert rfc.main(args + extra + ["--out", str(out)]) == 0
        rec = json.loads(out.read_text())
        assert rec["jackknife_groups"] == want and rec["areas"]["spot"]["jackknife_groups"] == want
    with pytest.raises(SystemExit):
        rfc.main(args + ["--groups", "2", "--out", str(tmp_path / "bad.json")])
