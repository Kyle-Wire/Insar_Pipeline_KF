"""
Lane L1 of the 5 Oct 2026 review: the statistical core (significance.py,
gnss_tie.py, resolution.py). Every test here failed on the code before the
fixes and checks an OUTCOME on synthetic data with a known answer - the
detection rate of a planted shift, the false-alarm rate on pure noise, the
total of a series whose true change is known - not that a function ran.
"""
import json
import os
import types
from pathlib import Path

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from insar_pipeline import significance as sg

T = np.linspace(0, 10.45, 270)          # the record's shape: 270 dates over 10.45 yr


def _series(truth, seed, noise=7.0, season=5.0):
    rng = np.random.default_rng(seed)
    return truth(T) + season * np.sin(2 * np.pi * T + 0.7) + rng.normal(0, noise, T.size)


def _corr_field(seed, n=200, s=8.0, sd=3.0):
    f = gaussian_filter(np.random.default_rng(seed).normal(0, 1, (n, n)), s, mode="wrap")
    return f / f.std() * sd


# ---------------------------------------------------------------- stats-5: totals are the measured change
def test_record_change_follows_a_mid_record_s_curve_and_an_early_decay():
    s_curve = lambda x: -350 / (1 + np.exp(-(x - 3.5) * 2.0))            # noqa: E731 - mined mid-record
    decay = lambda x: -300 * (1 - np.exp(-x / 1.0))                       # noqa: E731 - spoil settling from day one
    for truth, rate_x_span_off in ((s_curve, 0.12), (decay, 0.20)):
        true_change = truth(T[-1]) - truth(T[0])
        r = sg.record_change(T, _series(truth, 11))
        assert r["method"] == "piecewise_linear_yearly" and r["nonlinear"]
        assert abs(r["change_mm"] - true_change) <= 0.05 * abs(true_change)
        # and the old figure (rate x span) is visibly wrong, so the test can tell them apart
        assert abs(r["linear_change_mm"] - true_change) > rate_x_span_off * abs(true_change)
        assert r["end_mm"] - r["start_mm"] == pytest.approx(r["change_mm"], abs=0.2)


def test_record_change_of_a_step_and_of_a_steady_movement():
    step = sg.record_change(T, _series(lambda x: np.where(x > 5.2, -100.0, 0.0), 12))
    assert abs(step["change_mm"] + 100) < 15 and step["nonlinear"]
    assert step["linear_change_mm"] < -130                               # rate x span says ~-150
    steady = sg.record_change(T, _series(lambda x: -20.0 * x, 13))
    fit = sg.fit_rate(T, _series(lambda x: -20.0 * x, 13))
    assert not steady["nonlinear"]
    assert steady["change_mm"] == pytest.approx(sg.trend_total(fit), rel=0.03)


def test_record_change_falls_back_to_rate_x_span_on_a_short_record():
    t = np.linspace(0, 1.5, 40)
    r = sg.record_change(t, -10 * t + np.random.default_rng(0).normal(0, 3, t.size))
    assert r["method"] == "rate_x_span" and r["change_mm"] == r["linear_change_mm"]
    assert sg.record_change(t[:5], t[:5]) is None                         # fewer than 6 dates: no rate at all
    assert "AVERAGE RATE" in sg.trend_total.__doc__.upper()


# ---------------------------------------------------------------- tests-5 / clientout-4: one rule, 80% power
def test_reading_threshold_and_signal_to_threshold_never_disagree():
    """Heavy-tailed, off-centre null: the decision, signal/threshold and p
    must say the same thing for every planted offset (the Site B annual
    sheet read 'within' at S/N 2.73 and 'stands out' at 2.83)."""
    rng = np.random.default_rng(5)
    base = gaussian_filter(rng.standard_t(3, (260, 260)), 2.0)
    base = base / base.std() * 3.0 + 0.4 * 3.0                           # +0.4 sd off zero
    far = np.ones(base.shape, bool); far[100:160, 100:160] = False
    region = np.zeros_like(far); region[115:145, 115:145] = True
    seen = set()
    for shift in np.linspace(-4, 4, 33):
        f = base.copy(); f[region] += shift
        a = sg.assess(f, far, region)
        assert a["testable"] and not a["null_thin"]
        assert a["distinguishable_from_farmland"] == (a["signal_to_threshold"] >= 1)
        if a["distinguishable_from_farmland"]:
            assert a["p_two_sided"] <= 0.05 + 1e-3
        else:
            assert a["p_two_sided"] >= 0.05 - 1e-3
        assert a["excess_mm_yr"] == pytest.approx(a["value_mm_yr"] - a["null_median_mm_yr"], abs=0.02)
        assert a["detectable_80_mm_yr"] > a["threshold_95_mm_yr"]
        seen.add(a["distinguishable_from_farmland"])
    assert seen == {True, False}


def test_planted_shift_of_2_8_sigma_is_detected_four_times_in_five():
    """The detection limit is 80% power: a sustained shift of 2.8 farmland
    sigma is caught at least 4 times in 5; the 2 sigma once printed as 'the
    smallest movement that would have been detected' only about half the time."""
    hits = {2.0: 0, 2.8: 0}
    d80 = []
    n = 300
    for i in range(n):
        rr = np.random.default_rng(1000 + i)
        f0 = rr.normal(0, 5.0, (100, 100))                               # region mean sigma = 5/10 = 0.5
        far = np.ones_like(f0, bool); far[40:60, 40:60] = False
        reg = np.zeros_like(far); reg[45:55, 45:55] = True
        for k in hits:
            f = f0.copy(); f[reg] += k * 0.5
            hits[k] += sg.assess(f, far, reg)["distinguishable_from_farmland"]
        if i < 20:
            d80.append(sg.assess(f0, far, reg)["detectable_80_mm_yr"] / 0.5)
    assert hits[2.8] / n >= 0.80                                          # theory 0.80; measured 0.81
    assert hits[2.0] / n <= 0.65                                          # theory 0.52
    assert 2.6 <= np.mean(d80) <= 3.1                                     # ~2.8 sigma on a Gaussian null


# ---------------------------------------------------------------- stats-4: centred test, re-centred outside ring
def test_assess_is_centred_on_the_null_not_on_zero():
    rng = np.random.default_rng(6)
    f = rng.normal(0, 5.0, (200, 200))
    far = np.ones_like(f, bool); far[80:120, 80:120] = False
    reg = np.zeros_like(far); reg[90:100, 90:100] = True
    sd = 0.5
    f[far] += 1.5 * sd                                                    # farmland as a whole reads +c
    f[reg] = f[reg] - f[reg].mean() + 1.5 * sd + rng.normal(0, 0.01)      # the region reads exactly the same
    a = sg.assess(f, far, reg)
    assert abs(a["z"]) < 0.3 and a["p_two_sided"] > 0.5
    assert a["distinguishable_from_farmland"] is False


def _ring_case(offset):
    rng = np.random.default_rng(8)
    n = 240
    tc = np.full((n, n), 0.5)
    vel = gaussian_filter(rng.normal(0, 1, (n, n)), 3.0)
    vel = vel / vel.std() * 4.0 - 10.0                                    # class 0.4-0.6, DS bias -10
    aoi = np.zeros((n, n), bool); aoi[40:200, 40:200] = True
    vel[~aoi] += offset                                                   # the ring outside the box reads differently
    far_wide = np.ones((n, n), bool); far_wide[100:140, 100:140] = False
    far = far_wide & aoi
    reg = np.zeros((n, n), bool); reg[110:130, 110:130] = True
    return vel, tc, far, far_wide, aoi, reg


def test_outside_ring_offset_does_not_shift_the_null():
    vel, tc, far, far_wide, aoi, reg = _ring_case(0.0)
    ref = sg.assess(sg.placement_field(vel, tc, far, far_wide, aoi), far_wide, reg)
    vel2, *_ = _ring_case(2.0)
    plain = sg.assess(sg.class_matched_field(vel2, tc, far)[0], far_wide, reg)
    fixed_field = sg.placement_field(vel2, tc, far, far_wide, aoi)
    fixed = sg.assess(fixed_field, far_wide, reg)
    for k in ("farmland_spread_1sigma_mm_yr", "threshold_95_mm_yr"):
        assert fixed[k] == pytest.approx(ref[k], rel=0.05)
    assert fixed["null_median_mm_yr"] == pytest.approx(ref["null_median_mm_yr"], abs=0.1)
    assert abs(plain["null_median_mm_yr"] - ref["null_median_mm_yr"]) > 0.3      # what the old field did
    # inside the box the placement field IS the class-matched field
    r_cm, _ = sg.class_matched_field(vel2, tc, far)
    assert np.allclose(fixed_field[aoi], r_cm[aoi], equal_nan=True)


# ---------------------------------------------------------------- stats-1 / generality-2: thin and missing nulls
def test_thin_null_false_alarms_stay_at_or_below_five_percent():
    """Correlated noise, a region the footprint fits only ~9 times: the old
    code (thin flagged only on the footprint path, no widening) called 8% of
    noise-only regions 'distinguishable'."""
    n = 200
    region = np.zeros((n, n), bool); o = (n - 64) // 2; region[o:o + 64, o:o + 64] = True
    far = ~region
    alarms = untestable = 0
    N = 120
    for i in range(N):
        a = sg.assess(_corr_field(7000 + i), far, region)
        untestable += not a["testable"]
        alarms += bool(a["distinguishable_from_farmland"])
        if a["testable"]:
            assert a["null_thin"] == (a["approx_independent_placements"] < 10)
            if a["approx_independent_placements"] < 10:
                assert a["spread_for_ranges_mm_yr"] > a["farmland_spread_1sigma_mm_yr"]
                assert a["threshold_95_mm_yr"] >= 1.96 * a["spread_for_ranges_mm_yr"] - 0.02
                assert a["detectable_80_mm_yr"] >= 2.8 * a["spread_for_ranges_mm_yr"] - 0.03
    assert alarms / N <= 0.05
    assert untestable > 0                                                # the thinnest are not judged at all


def test_thin_null_is_compared_in_the_regions_own_frame():
    """A field offset by +5 everywhere and no signal: when the re-referenced
    placements (larger spread) are chosen, the region must be compared in the
    same frame, or the offset reads as an excess (it read 4-7 mm/yr before)."""
    n = 200
    reg = np.zeros((n, n), bool); o = (n - 72) // 2; reg[o:o + 72, o:o + 72] = True
    chosen = 0
    for seed in range(4):
        a = sg.assess(_corr_field(seed) + 5.0, ~reg, reg)
        if not a["testable"]:
            continue
        chosen += "outside it" in a["method"]
        assert abs(a["excess_mm_yr"]) < a["threshold_95_mm_yr"]
        assert a["excess_mm_yr"] == pytest.approx(a["value_mm_yr"] - a["null_median_mm_yr"], abs=0.02)
    assert chosen >= 2


def test_large_region_on_the_compact_path_is_flagged_thin_or_untestable():
    """A region covering a quarter of the grid goes straight to the compact
    square; the old code never flagged that path as thin."""
    f = _corr_field(3, n=240, s=10.0)
    region = np.zeros(f.shape, bool); region[40:200, 40:200] = True
    assert sg.footprint_null(f, ~region, region)["shape"] == "compact_square"
    a = sg.assess(f, ~region, region)
    assert a["null_thin"] is True
    assert a["approx_independent_placements"] < 10
    assert a["testable"] is False or a["spread_for_ranges_mm_yr"] > a["farmland_spread_1sigma_mm_yr"]


def test_mine_drawn_as_the_area_with_a_thin_ring_is_not_judged_within():
    f = _corr_field(4, n=240, s=15.0)
    region = np.zeros(f.shape, bool); region[40:200, 40:200] = True
    from scipy.ndimage import binary_dilation
    far = ~binary_dilation(region, iterations=24)
    a = sg.assess(f, far, region)
    assert a != {} and a["testable"] is False
    assert a["distinguishable_from_farmland"] is None
    # significance says "ordinary ground" at the source (28 Oct 2026)
    assert a["not_testable_reason"] and "ordinary ground" in a["not_testable_reason"]
    assert "farmland_spread_1sigma_mm_yr" not in a                       # no range can be built from it
    # no far field at all: the null is unavailable, still not {}
    b = sg.assess(f, np.zeros_like(region), region)
    assert b["testable"] is False and b["null_available"] is False and b["distinguishable_from_farmland"] is None
    # an area with no measured cell: not testable, never {}
    c = sg.assess(f, far, np.zeros_like(region))
    assert c["testable"] is False and c["value_mm_yr"] is None


def test_well_sampled_white_noise_is_unchanged():
    rng = np.random.default_rng(1)
    field = rng.normal(0, 5.0, (400, 400))
    far = np.ones_like(field, bool)
    region = np.zeros_like(far); region[10:20, 10:20] = True
    a = sg.assess(field, far, region)
    assert a["testable"] and not a["null_thin"] and a["null_shape"] == "footprint"
    assert a["spread_for_ranges_mm_yr"] == a["farmland_spread_1sigma_mm_yr"] == pytest.approx(0.5, rel=0.1)
    assert a["threshold_95_mm_yr"] == pytest.approx(1.96 * 0.5, rel=0.1)


# ---------------------------------------------------------------- stats-3: robust class centres, borrowing
def test_unmapped_subsidence_in_the_far_field_does_not_move_the_zero():
    """stats-3: a sinking patch left in the comparison ground pulls the class
    mean; the second pass (far_without: the found feature cut out) puts the
    zero back on ordinary ground."""
    rng = np.random.default_rng(9)
    n = 200
    tc = np.full((n, n), 0.5)
    vel = rng.normal(-10.0, 3.0, (n, n))
    far = np.ones((n, n), bool); far[:, 150:] = False
    vel[20:60, 20:80] -= 30.0                                            # 2400 far cells sinking, unmapped
    reg = np.zeros((n, n), bool); reg[100:140, 160:190] = True           # ordinary ground outside the far field
    r1, _ = sg.class_matched_field(vel, tc, far)
    assert r1[reg].mean() > 2.0                                          # 8% of the far field 30 mm/yr low: 2.4
    attrs = {"X_FIRST": 500000.0, "Y_FIRST": 4_000_000.0, "X_STEP": 15.0, "Y_STEP": -15.0}
    feat = np.zeros_like(far); feat[20:60, 20:80] = True
    r2, _ = sg.class_matched_field(vel, tc, sg.far_without(far, [feat], attrs, 500.0))
    assert abs(r2[reg].mean()) < 0.15


def test_mean_centres_keep_an_ordinary_area_at_zero_on_skewed_farmland():
    """6 Oct 2026: farmland rates are skewed (an unwrapping-error tail), so the
    median zeroes the typical cell but leaves the typical AREA mean off zero
    (0.58-0.65 mm/yr LOS on the real records). Client figures are area means:
    the default centre is the mean; robust=True stays available."""
    rng = np.random.default_rng(11)
    n = 200
    tc = np.full((n, n), 0.5)
    vel = rng.normal(0.0, 1.0, (n, n)) - 10.0 * (rng.random((n, n)) < 0.1)  # a 10% error tail everywhere, no feature
    far = np.ones((n, n), bool); far[:, 150:] = False
    reg = np.zeros((n, n), bool); reg[100:140, 160:190] = True
    r_mean, _ = sg.class_matched_field(vel, tc, far)
    r_med, _ = sg.class_matched_field(vel, tc, far, robust=True)
    assert abs(r_mean[reg].mean()) < 0.15
    assert r_med[reg].mean() < -0.4


def test_sparse_class_borrows_and_says_so():
    rng = np.random.default_rng(10)
    n = 120
    tc = np.full((n, n), 0.5); tc[:10, :10] = 0.95                       # 100 class-4 cells, all in the far field
    tc[60:80, 60:80] = 0.95                                              # a high-coherence pad in the region
    vel = rng.normal(0, 1.0, (n, n))
    far = np.ones((n, n), bool); far[50:90, 50:90] = False
    means = sg.class_means(vel, tc, far)
    assert means["_borrowed"] == [1, 2, 3, 4] and means[4] == means[0]
    r, centres = sg.class_matched_field(vel, tc, far)
    assert set(centres) == {0}                                           # callers loop over measured classes only
    pad = np.zeros_like(far); pad[60:80, 60:80] = True
    assert np.isfinite(r[pad]).all()                                     # was NaN: the pad dropped out
    # nearest MEASURED class lends, never a borrower
    tc2 = np.where(np.arange(n)[None, :] < 60, 0.5, 0.95) * np.ones((n, 1))
    m2 = sg.class_means(rng.normal(0, 1, (n, n)) + np.where(tc2 > 0.9, 5.0, 0.0), tc2, np.ones((n, n), bool))
    assert m2["_borrowed"] == [1, 2, 3] and m2[3] == m2[4] and m2[1] == m2[0]


# ---------------------------------------------------------------- stats-14: finer coherence bins (option)
def test_fine_coherence_bins_remove_within_class_bias_when_asked():
    rng = np.random.default_rng(12)
    n = 200
    fields = rng.uniform(0.4, 0.6, (n // 10, n // 10))
    tc = np.kron(fields, np.ones((10, 10)))                              # one coherence per 10x10 field
    vel = -6.0 - 40.0 * (0.1 - np.abs(tc - 0.5)) + rng.normal(0, 1.0, (n, n))   # -10 at 0.5, -6 at the edges
    far = np.ones((n, n), bool); far[80:120, 80:120] = False
    reg = np.zeros_like(far); reg[90:100, 90:100] = True
    coarse = sg.assess(sg.class_matched_field(vel, tc, far)[0], far, reg)
    default = sg.assess(sg.class_matched_field(vel, tc, far, fine_step=None)[0], far, reg)
    fine = sg.assess(sg.class_matched_field(vel, tc, far, fine_step=0.025, fine_min_n=500)[0], far, reg)
    assert default["farmland_spread_1sigma_mm_yr"] == coarse["farmland_spread_1sigma_mm_yr"]
    assert fine["farmland_spread_1sigma_mm_yr"] < 0.8 * coarse["farmland_spread_1sigma_mm_yr"]


# ---------------------------------------------------------------- stats-18: white-noise scaling of split_half
def test_empirical_white_scaling_is_split_half_over_four(tmp_path):
    import h5py
    rng = np.random.default_rng(13)
    n_px, n_t = 40, 50                                                   # 2000 pixels, white noise
    d0 = np.datetime64("2016-01-05")
    dates = [str(d0 + np.timedelta64(12 * i, "D")).replace("-", "") for i in range(270)]
    ts = rng.normal(0, 0.007, (270, n_px, n_t)).astype(np.float32)       # 7 mm per epoch, metres
    p = tmp_path / "ts.h5"
    with h5py.File(p, "w") as f:
        f["date"] = np.array(dates, dtype="S8")
        f.create_dataset("timeseries", data=ts, chunks=(8, n_px, n_t))
    tc = np.full((n_px, n_t), 0.5)
    far = np.ones((n_px, n_t), bool)
    m = sg.empirical_rate_model(p, tc, far)
    row = m["by_class"][0]
    assert row["empirical_white_scaling_mm_yr"] == pytest.approx(row["split_half_robust_sd_mm_yr"] / 4, abs=0.002)
    # and it IS the full-record rate error under white noise (formal OLS error), not 1.41x it
    assert row["empirical_white_scaling_mm_yr"] == pytest.approx(row["formal_median_mm_yr"], rel=0.12)


# ---------------------------------------------------------------- contract: one per-cell null
def test_cell_null_by_class_is_the_larger_of_spread_and_repeatability():
    model = {"by_class": [{"class_index": 0, "empirical_mm_yr": 5.5}, {"class_index": 4, "empirical_mm_yr": 2.0}]}
    spread = [{"class_index": 0, "farmland_cell_spread_mm_yr": 7.4}, {"class_index": 1, "farmland_cell_spread_mm_yr": 3.0}]
    c = sg.cell_null_by_class(model, spread)
    assert c[0] == 7.4                       # spread wins
    assert c[1] == 5.5                       # repeatability of the nearest measured class (0) wins over 3.0
    assert c[4] == 2.0 and c[3] == 2.0       # no spread: repeatability alone
    assert sg.cell_null_by_class(dict(model, farmland_cell_spread=spread)) == c


# ---------------------------------------------------------------- methodology-8 / gap-geolocation-5: resolution
ATTRS_PL = {"PROCESSOR": "isce", "RANGE_PIXEL_SIZE": "25.629341337932303", "AZIMUTH_PIXEL_SIZE": "46.850480731040605",
            "RLOOKS": "11", "ALOOKS": "3", "CENTER_INCIDENCE_ANGLE": "39.197147", "HEIGHT": "700371.92",
            "X_STEP": "0.000172", "Y_STEP": "-0.000135", "Y_FIRST": "38.46293142211914", "LENGTH": "1882"}


def _cfg(**kw):
    base = dict(backend="miaplpy", miaplpy_range_window=15, miaplpy_azimuth_window=15, stack_range_looks=11,
                stack_azimuth_looks=3, mintpy_geocode_lalo_step="-0.000135,0.000172",
                bbox=(-87.56, 38.24, -87.33, 38.43), project_dir=Path("/nonexistent"))
    base.update(kw)
    return types.SimpleNamespace(**base)


def test_phase_linked_resolution_is_the_window_not_the_look_cell():
    from insar_pipeline import resolution as rs
    r = rs.effective_resolution(_cfg(), ATTRS_PL)
    assert r["posting_m"] == pytest.approx(15.0, abs=0.3)
    assert 80 <= r["support_m"][0] <= 100 and 220 <= r["support_m"][1] <= 250
    assert r["look_cell_m"] == pytest.approx([40.6, 42.2], abs=0.5)
    assert "41 m" not in r["text"] and "15 m apart" in r["text"] and "240 m north-south" in r["text"]
    stack = rs.effective_resolution(_cfg(backend="stack"), ATTRS_PL)
    assert stack["support_m"] == pytest.approx([40.6, 42.2], abs=0.5)
    assert "41 m east-west by 42 m north-south" in stack["text"]


def test_resolution_text_follows_the_config_and_route():
    from insar_pipeline import resolution as rs
    sq = rs.effective_resolution(_cfg(miaplpy_range_window=21, miaplpy_azimuth_window=5), ATTRS_PL)
    assert sq["support_m"][1] < 120 and sq["support_m"][0] > 110                   # a near-square window says so
    no_attrs = rs.effective_resolution(_cfg())
    assert "defaults" in no_attrs["source"] and 220 <= no_attrs["support_m"][1] <= 250
    disp = rs.effective_resolution(_cfg(backend="disp", disp_decimate=1))
    assert disp["posting_m"] == 30.0 and "30 m apart" in disp["text"]
    r = rs.effective_resolution(_cfg(), ATTRS_PL)
    assert rs.area_caveat(r, 260, 410) and "north-south" in rs.area_caveat(r, 260, 410)   # Area A2-sized
    assert rs.area_caveat(r, 2600, 4100) is None


# ---------------------------------------------------------------- gnss_tie fixture: a tiny synthetic project
SITES = {  # name: (lat, lon, t0, t1, rate mm/yr) - centre of the box is 39.0, -87.3
    "AAAA": (39.10, -87.30, 2010.5, 2026.6, -1.0),     # 11 km, long record: used
    "BBBB": (39.3417, -87.30, 2024.0, 2026.0, -2.0),   # 38 km but only 2 yr in the record: not used
    "CCCC": (39.375, -87.30, 2010.5, 2026.6, -4.0),    # 41.7 km: outside the 40 km radius
}
REGIONAL = {  # catalogue-only stations 60-90 km away (MIDAS rates for the floor)
    "DDDD": (39.6, -87.30, 2010.5, 2024.7, -3.5), "EEEE": (38.4, -87.30, 2010.5, 2024.7, -0.1),
    "FFFF": (39.0, -86.5, 2010.5, 2024.7, -1.2),
}
DATES = ["20160105", "20260617"]


def _tenv3(site):
    lat, lon, t0, t1, rate = SITES[site]
    t = np.arange(t0, t1, 1 / 365.25)
    rng = np.random.default_rng(sum(map(ord, site)))
    return {"site": site, "t": t, "e": t * 0, "n": t * 0, "u": rate * (t - t0) + rng.normal(0, 1.0, t.size),
            "lat": lat, "lon": lon}


@pytest.fixture
def project(tmp_path, monkeypatch):
    import h5py
    from insar_pipeline import deliverables as dl
    from insar_pipeline import gnss_tie as gt
    from insar_pipeline import gnss_validation as gv
    geo = tmp_path / "proj" / "mintpy" / "geo"; geo.mkdir(parents=True)
    attrs = {"X_FIRST": -87.36, "Y_FIRST": 39.06, "X_STEP": 0.003, "Y_STEP": -0.003, "LENGTH": 40, "WIDTH": 40}
    rng = np.random.default_rng(0)
    for name, ds, arr in (("geo_velocity.h5", "velocity", rng.normal(0, 0.001, (40, 40))),
                          ("geo_temporalCoherence.h5", "temporalCoherence", np.full((40, 40), 0.6))):
        with h5py.File(geo / name, "w") as f:
            f[ds] = arr.astype(np.float32)
            for k, v in attrs.items():
                f.attrs[k] = str(v)
    ts = geo / "geo_timeseries_test.h5"
    with h5py.File(ts, "w") as f:
        f["date"] = np.array(DATES, dtype="S8")
        f["timeseries"] = np.zeros((2, 40, 40), np.float32)
    cache = tmp_path / "gnss"; cache.mkdir()
    allst = {**SITES, **REGIONAL}
    (cache / "unr_llh.out").write_text("".join(f"{s} {la} {lo} 100.0\n" for s, (la, lo, *_) in allst.items()))
    (cache / "unr_midas_igs14.txt").write_text("".join(
        f"{s} MIDAS5 {t0} {t1} {t1 - t0:.4f} 3000 3000 5000 0.0 0.0 {r / 1000:.6f} 0.0003 0.0003 0.0006 x\n"
        for s, (la, lo, t0, t1, r) in allst.items()))
    cfg = types.SimpleNamespace(project_dir=tmp_path / "proj", mintpy_dir=tmp_path / "proj" / "mintpy",
                                bbox=(-87.35, 38.95, -87.25, 39.05), gnss_tie_radius_km=40.0,
                                min_temporal_coherence=0.4, deramp_exclude_geojson="",
                                acceptance_far_field_buffer_m=2000.0)
    monkeypatch.setattr(dl, "timeseries_file", lambda c: ts)
    monkeypatch.setattr(gv, "load_stations", lambda root, *a, **k: {
        s: {"lat": la, "lon": lo} for s, (la, lo, *_) in allst.items()})
    monkeypatch.setattr(gt, "fetch_tenv3", lambda site, root, timeout=180: site)
    monkeypatch.setattr(gt, "read_tenv3", lambda p: _tenv3(str(p)))
    return types.SimpleNamespace(cfg=cfg, cache=cache, geo=geo)


# ---------------------------------------------------------------- generality-3 / stats-10: one station is not 0.3 mm/yr
def test_single_station_tie_is_floored_by_the_regional_scatter(project):
    from insar_pipeline import gnss_tie as gt
    out = gt.run(project.cfg)                                       # cache root = project_dir.parent / "gnss"
    bg = out["background"]
    assert out["stations_used"] == ["AAAA"] and bg["n_sites"] == 1
    floor = np.std([-1.0, -4.0, -3.5, -0.1, -1.2], ddof=1)   # catalogue sites within 100 km, >= 3 yr overlap (not BBBB)
    assert out["uncertainty_floor_mm_yr"] == pytest.approx(floor, abs=0.01)
    assert bg["vertical_uncertainty_mm_yr"] >= floor - 0.01 > bg["formal_mm_yr"] * 3
    assert bg["uncertainty_basis"] == "regional floor"
    assert out["leave_one_out"] is None                              # nothing to leave out
    assert out["radius_sensitivity"]["30"]["sites"] == ["AAAA"]
    assert json.loads((project.cfg.project_dir / "deliverables" / "gnss_tie.json").read_text())["background"] == bg


def test_two_station_tie_does_not_rest_on_a_two_value_sd(project, monkeypatch):
    from insar_pipeline import gnss_tie as gt
    monkeypatch.setitem(SITES, "GGGG", (39.18, -87.30, 2010.5, 2026.6, -1.4))     # 20 km, agrees with AAAA
    monkeypatch.setattr(gt, "discover", lambda cfg, root, radius=None: ["AAAA", "GGGG"])
    out = gt.run(project.cfg)
    bg = out["background"]
    assert bg["n_sites"] == 2 and bg["between_site_scatter_mm_yr"] < 0.5
    assert bg["vertical_uncertainty_mm_yr"] == pytest.approx(out["uncertainty_floor_mm_yr"], abs=0.01)
    assert bg["uncertainty_basis"] == "regional floor"
    assert set(out["leave_one_out"]["without"]) == {"AAAA", "GGGG"}
    assert out["leave_one_out"]["max_change_mm_yr"] == pytest.approx(0.2, abs=0.1)


def test_builder_preview_matches_what_the_tie_uses(project):
    from insar_pipeline import config
    from insar_pipeline import gnss_tie as gt
    assert gt.radius_km(types.SimpleNamespace()) == config.GNSS_TIE_RADIUS_KM == 40.0
    out = gt.run(project.cfg)
    pv = gt.preview_stations(39.0, -87.3, DATES[0], DATES[1], cache_root=project.cache)
    assert {p["site"] for p in pv if p["will_use"]} == set(out["stations_used"]) == {"AAAA"}
    assert {p["site"] for p in pv} >= {"BBBB", "CCCC"}                  # shown, but marked not used


# ---------------------------------------------------------------- tests-4 / generality-14: no stale tie
def test_switching_the_tie_off_replaces_an_old_file(project):
    from insar_pipeline import gnss_tie as gt
    gt.run(project.cfg)
    project.cfg.gnss_tie_radius_km = 0
    out = gt.run(project.cfg)
    on_disk = json.loads((project.cfg.project_dir / "deliverables" / "gnss_tie.json").read_text())
    assert on_disk["background"] is None and on_disk["anchored"] is False
    assert "switched off" in on_disk["reason"] and on_disk["record_window"] == DATES
    assert gt.load_valid(project.cfg, DATES)["background"] is None and out["background"] is None


def test_no_station_or_a_failed_tie_still_writes_an_unanchored_file(project, monkeypatch):
    from insar_pipeline import gnss_tie as gt
    gt.run(project.cfg)
    monkeypatch.setattr(gt, "discover", lambda cfg, root, radius=None: [])
    assert gt.run(project.cfg)["background"] is None
    assert "No continuous GNSS station" in gt.load_valid(project.cfg, DATES)["reason"]
    monkeypatch.setattr(gt, "discover", lambda cfg, root, radius=None: ["AAAA"])
    monkeypatch.setattr(gt, "tie", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    with pytest.raises(RuntimeError):
        gt.run(project.cfg)
    on_disk = json.loads((project.cfg.project_dir / "deliverables" / "gnss_tie.json").read_text())
    assert on_disk["background"] is None and "boom" in on_disk["reason"]


def test_load_valid_refuses_another_record_or_an_older_file(project):
    from insar_pipeline import gnss_tie as gt
    gt.run(project.cfg)
    assert gt.load_valid(project.cfg, DATES)["background"]["vertical_mm_yr"] == pytest.approx(-1.0, abs=0.2)
    assert gt.load_valid(project.cfg, ["20170101", DATES[1]]) is None          # extended / shortened record
    tie_f = project.cfg.project_dir / "deliverables" / "gnss_tie.json"
    vel = project.geo / "geo_velocity.h5"
    os.utime(vel, (tie_f.stat().st_mtime + 60, tie_f.stat().st_mtime + 60))   # products rebuilt after the tie
    assert gt.load_valid(project.cfg, DATES) is None
    ok, why = gt.check(project.cfg, DATES)
    assert ok is None and "older" in why


def test_old_tie_files_on_disk_still_validate(tmp_path):
    """Products written before 5 Oct 2026 (no anchored/stations_used keys) still
    load when they carry the regional floor; an ANCHORED tie without it is
    stale since 6 Oct (test_fix_final.test_pre_floor_tie_is_stale)."""
    from insar_pipeline import gnss_tie as gt
    d = tmp_path / "p" / "deliverables"; d.mkdir(parents=True)
    old = {"record_window": DATES, "stations": [], "uncertainty_floor_mm_yr": 1.01,
           "background": {"vertical_mm_yr": -1.09, "vertical_uncertainty_mm_yr": 1.01,
                          "uncertainty_basis": "regional floor"}}
    (d / "gnss_tie.json").write_text(json.dumps(old))
    cfg = types.SimpleNamespace(project_dir=tmp_path / "p", mintpy_dir=tmp_path / "p" / "mintpy")
    assert gt.load_valid(cfg, DATES)["background"]["vertical_mm_yr"] == -1.09
    # an untied record needs neither field
    (d / "gnss_tie.json").write_text(json.dumps({"record_window": DATES, "background": None,
                                                 "reason": "no station within 40 km"}))
    assert gt.load_valid(cfg, DATES)["background"] is None
