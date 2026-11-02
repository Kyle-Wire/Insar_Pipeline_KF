"""
Trust review of 7 Oct 2026, group sig (significance.py): decisions E5, E6, E10
and E28, and the producer side of contracts C1, C2 and C3.

E5   per-sign 80% detection limits against the unchanged symmetric rule
     (detection_limit_sink_mm_yr / detection_limit_rise_mm_yr); on the
     Site A copy the Area A2's sinking limit is 14.32 LOS = 18.3
     mm/yr vertical against the stated 15.46 (19.7), finding 13.
E6   local (10 km) and coherence-matched (+/-0.025) nulls as diagnostics,
     never a switch (local_null / coherence_null).
E10  record_change uses the hinge total only when the pace change beats the
     farmland copies' 95th percentile (pace_null; pace_change_tested), taken
     as a 5% test with 20-40 copies (reviewer: the largest of 20-38 copies,
     not np.quantile's interpolated value, which farmland beats 7-9% of the time).
E28  far_field_mask(exclude=...) takes report areas out of the comparison
     ground (class centres and null).
Each test fails on the code before this round (backups/pre_trust_20261007).
"""
import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from insar_pipeline import significance as sg

# a projected grid (metres), so distances are simple: 100 m cells
ATTRS_100M = {"X_FIRST": 500000.0, "Y_FIRST": 4_000_000.0, "X_STEP": 100.0, "Y_STEP": -100.0}


def _skewed_null_field(n=260, seed=0):
    """Farmland whose placements have a long SINKING tail (as the real nulls:
    Area A2 median +0.72, mean +0.03 LOS): smooth noise plus sparse deep
    sinking blobs."""
    rng = np.random.default_rng(seed)
    f = gaussian_filter(rng.normal(0, 1, (n, n)), 2) * 6
    pits = np.zeros((n, n))
    idx = rng.integers(0, n, (60, 2))
    pits[idx[:, 0], idx[:, 1]] = -400.0
    f += gaussian_filter(pits, 3)
    return f


# ------------------------------------------------------------------ E5 / C1
def _sampled_power(d, c, shift):
    """detection rate of a shift added to every centred placement, same rule |x| >= c"""
    return float(np.mean(np.abs(d + shift) >= c))


def test_assess_returns_a_sinking_and_a_rising_limit_on_the_same_symmetric_rule():
    f = _skewed_null_field()
    far = np.ones(f.shape, bool)
    reg = np.zeros(f.shape, bool); reg[100:104, 100:104] = True
    far &= ~reg
    a = sg.assess(f, far, reg)
    assert a["testable"] and not a["null_thin"]
    sink, rise, worse = a["detection_limit_sink_mm_yr"], a["detection_limit_rise_mm_yr"], a["detectable_80_mm_yr"]
    # the skewed null: sinking is caught sooner, the stated (worse-sign) limit is the rising one
    assert sink < rise
    assert max(sink, rise) == pytest.approx(worse, abs=0.011)
    # the decision rule is unchanged: |excess| >= the 95th percentile of |null - median|
    nl = sg.footprint_null(f, far, reg)
    v = nl["values"]
    d = np.sort(v - np.median(v))
    c95 = float(np.quantile(np.abs(d), 0.95))
    assert a["threshold_95_mm_yr"] == pytest.approx(c95, abs=0.006)
    # sampled power on the farmland placements themselves: 80% at each sign's own limit
    # (the unrounded limits; assess stores them to 0.01)
    s_ex, r_ex = sg._power_limit(d, c95, sign=-1), sg._power_limit(d, c95, sign=+1)
    assert (round(s_ex, 2), round(r_ex, 2)) == (sink, rise)
    assert _sampled_power(d, c95, -s_ex) >= 0.8 - 1e-9
    assert _sampled_power(d, c95, -0.97 * s_ex) < 0.8
    assert _sampled_power(d, c95, +r_ex) >= 0.8 - 1e-9
    assert _sampled_power(d, c95, +0.97 * r_ex) < 0.8
    # and a sinking movement of the old (worse-sign) limit is caught more than 4 times in 5
    assert _sampled_power(d, c95, -worse) > 0.8


def test_power_limit_by_sign_brackets_the_worse_sign():
    d = np.sort(np.r_[np.random.default_rng(1).normal(0, 1, 4000), -np.abs(np.random.default_rng(2).normal(0, 6, 300))])
    d = d - np.median(d)
    c = float(np.quantile(np.abs(d), 0.95))
    worse = sg._power_limit(d, c)
    sink, rise = sg._power_limit(d, c, sign=-1), sg._power_limit(d, c, sign=+1)
    assert sink <= worse + 1e-9 and rise <= worse + 1e-9
    assert max(sink, rise) == pytest.approx(worse, rel=1e-6)


def test_thin_null_floors_both_signs_at_the_widened_spread():
    rng = np.random.default_rng(4)
    f = gaussian_filter(rng.normal(0, 1, (120, 120)), 6) * 20
    far = np.zeros(f.shape, bool); far[:, :60] = True
    reg = np.zeros(f.shape, bool); reg[30:56, 70:96] = True
    a = sg.assess(f, far, reg)
    assert a["testable"] and a["null_thin"]
    assert sg.MIN_TESTABLE_INDEPENDENT <= a["approx_independent_placements"] < sg.RANGE_INFLATION_BELOW
    floor = (sg.Z95 + sg.Z80) * a["spread_for_ranges_mm_yr"]
    assert a["detection_limit_sink_mm_yr"] >= floor - 0.02
    assert a["detection_limit_rise_mm_yr"] >= floor - 0.02


def test_not_testable_carries_the_new_keys_as_none():
    f = np.zeros((40, 40)); far = np.zeros_like(f, bool)
    reg = np.zeros_like(far); reg[10:12, 10:11] = True
    a = sg.assess(f, far, reg)
    assert a["testable"] is False
    for k in ("detection_limit_sink_mm_yr", "detection_limit_rise_mm_yr", "local_null", "coherence_null"):
        assert k in a and a[k] is None


# ------------------------------------------------------------------ E6 / C1
def _grid(n=300, seed=5):
    rng = np.random.default_rng(seed)
    return gaussian_filter(rng.normal(0, 1, (n, n)), 1.5) * 8


def test_local_null_flags_a_shifted_neighbourhood_and_never_switches():
    f = _grid()
    # the region's own 10 km (100 cells at 100 m) sits on ground 4 mm/yr higher
    rr, cc = np.mgrid[:f.shape[0], :f.shape[1]]
    near = np.hypot(rr - 60, cc - 60) <= 100
    f_shift = f + np.where(near, 4.0, 0.0)
    far = np.ones(f.shape, bool)
    reg = np.zeros(f.shape, bool); reg[58:63, 58:63] = True
    far &= ~reg
    plain = sg.assess(f_shift, far, reg)
    a = sg.assess(f_shift, far, reg, attrs=ATTRS_100M)
    ln = a["local_null"]
    assert ln is not None and ln["n_independent"] >= sg.NULL_CHECK_MIN_INDEPENDENT
    assert ln["disagrees"] is True
    assert abs(ln["median_mm_yr"] - a["null_median_mm_yr"]) > sg.NULL_CENTRE_SHARE * a["threshold_95_mm_yr"]
    assert "differs from the pooled comparison ground" in ln["why"]
    assert a["coherence_null"] is None                 # no coherence given
    # a diagnostic only: the test itself is the pooled one, unchanged
    for k in ("excess_mm_yr", "threshold_95_mm_yr", "detectable_80_mm_yr", "distinguishable_from_farmland",
              "detection_limit_sink_mm_yr", "detection_limit_rise_mm_yr", "p_two_sided"):
        assert a[k] == plain[k]
    # homogeneous ground: the local null agrees
    b = sg.assess(f, far, reg, attrs=ATTRS_100M)
    assert b["local_null"] is not None and b["local_null"]["disagrees"] is False
    assert "agrees" in b["local_null"]["why"]


def test_coherence_matched_null_flags_a_coherence_dependent_offset():
    rng = np.random.default_rng(6)
    f = _grid(seed=7)
    tc = gaussian_filter(rng.uniform(0.4, 0.7, f.shape), 4)
    tc = 0.4 + (tc - tc.min()) / (tc.max() - tc.min()) * 0.3
    reg = np.zeros(f.shape, bool); reg[100:105, 100:105] = True
    c0 = float(tc[reg].mean())
    # ground of the region's own coherence reads 5 mm/yr higher (the DS bias inside one class)
    f_b = f + np.where(np.abs(tc - c0) <= 0.03, 5.0, 0.0)
    far = ~reg
    a = sg.assess(f_b, far, reg, tc=tc, attrs=ATTRS_100M)
    cn = a["coherence_null"]
    assert cn is not None and cn["disagrees"] is True
    assert cn["coherence_window"] == [pytest.approx(round(c0 - 0.025, 3)), pytest.approx(round(c0 + 0.025, 3))]
    b = sg.assess(f, far, reg, tc=tc, attrs=ATTRS_100M)
    assert b["coherence_null"] is not None and b["coherence_null"]["disagrees"] is False


def test_diagnostic_nulls_need_ten_independent_placements():
    f = _grid(n=60, seed=8)
    reg = np.zeros(f.shape, bool); reg[10:30, 10:30] = True
    tc = np.full(f.shape, 0.5)
    a = sg.assess(f, ~reg, reg, tc=tc, attrs=ATTRS_100M)
    assert a["testable"] and a["approx_independent_placements"] < sg.NULL_CHECK_MIN_INDEPENDENT
    assert a["local_null"] is None and a["coherence_null"] is None
    assert "fewer than 10 independent copies within 10 km" in a["null_checks_note"]
    assert "fewer than 10 independent copies within +/-0.025" in a["null_checks_note"]
    # with room for 11 independent copies both are computed (all of this ground is within 10 km)
    g = _grid(n=70, seed=8)
    reg2 = np.zeros(g.shape, bool); reg2[10:26, 10:26] = True
    b = sg.assess(g, ~reg2, reg2, tc=np.full(g.shape, 0.5), attrs=ATTRS_100M)
    assert b["approx_independent_placements"] >= sg.NULL_CHECK_MIN_INDEPENDENT
    assert b["local_null"]["n_independent"] == b["coherence_null"]["n_independent"] >= sg.NULL_CHECK_MIN_INDEPENDENT
    # without the coherence and grid geometry nothing is computed, and the record says so
    c = sg.assess(g, ~reg2, reg2)
    assert c["local_null"] is None and c["coherence_null"] is None and "not computed" in c["null_checks_note"]


def test_null_checks_spread_ratio_alone_can_disagree():
    vals = np.r_[np.random.default_rng(9).normal(0, 1, 2000), np.random.default_rng(10).normal(0, 3, 2000)]
    where = {"rows": np.r_[np.zeros(2000), np.full(2000, 500.0)], "cols": np.zeros(4000), "coherence": None}
    reg = np.zeros((600, 10), bool); reg[0, 0] = True
    med, sd = float(np.median(vals)), float(vals.std())
    c95 = float(np.quantile(np.abs(vals - med), 0.95))
    out = sg.null_checks(vals, where, 10.0, reg, med, sd, c95, attrs=ATTRS_100M)
    ln = out["local_null"]          # the first 2000 (within 10 km): sd 1 against a pooled ~2.2
    assert ln["n_independent"] == 200 and ln["disagrees"] is True
    assert ln["spread_mm_yr"] / sd < sg.NULL_SPREAD_RATIO[0]


# ------------------------------------------------------------------ E10 / C3
T = np.linspace(0.0, 10.45, 270)


def _farm(seed, n=270):
    """farmland-like series: a slow wander (random walk), weather noise and a
    season whose amplitude changes from year to year - on which the old
    AR(1)-deflated F-test fires (audit: p < 0.05 on every real farmland patch)"""
    rng = np.random.default_rng(seed)
    amp = np.repeat(rng.uniform(2, 8, 12), 23)[:n]
    return np.cumsum(rng.normal(0, 1.6, n)) + rng.normal(0, 4, n) + amp * np.sin(2 * np.pi * T)


def test_steady_movement_on_farmland_noise_is_rate_x_span_once_tested():
    null = [_farm(s) for s in range(200, 240)]
    olds = [sg.record_change(T, s)["method"] for s in null[:20]]
    assert olds.count("piecewise_linear_yearly") >= 10      # the uncalibrated gate fires on ordinary ground
    y = -8.0 * T + _farm(7)
    old = sg.record_change(T, y)
    new = sg.record_change(T, y, pace_null=null)
    assert new["pace_change_tested"] is True and new["pace_null_n"] == 40
    assert new["pace_change_mm"] <= new["pace_null_p95_mm"]
    assert new["method"] == "rate_x_span" and new["nonlinear"] is False
    assert new["change_mm"] == new["linear_change_mm"] == old["linear_change_mm"]
    assert "rate x span" in new["reason"]


def test_a_real_change_of_pace_beats_the_farmland_null_and_keeps_the_hinge_total():
    null = [_farm(s) for s in range(300, 340)]
    decay = -400.0 * (1 - np.exp(-T / 1.0))                  # fast early, then slow (Site B Feature 1's shape)
    y = decay + _farm(11)
    r = sg.record_change(T, y, pace_null=sg.pace_change_null(T, null))
    assert r["pace_change_tested"] is True
    assert r["pace_change_mm"] > r["pace_null_p95_mm"]
    assert r["method"] == "piecewise_linear_yearly" and r["nonlinear"] is True
    assert abs(r["change_mm"] - decay[-1]) < abs(r["linear_change_mm"] - decay[-1])


def test_too_few_farmland_copies_cannot_test_a_change_of_pace():
    decay = -400.0 * (1 - np.exp(-T / 1.0)) + _farm(12)
    few = [_farm(s) for s in range(400, 411)]                # 11 copies (Site A Feature 1 had 11)
    r = sg.record_change(T, decay, pace_null=few)
    assert r["pace_change_tested"] is False and r["pace_null_n"] == 11 and r["pace_null_p95_mm"] is None
    assert r["method"] == "rate_x_span" and r["nonlinear"] is False
    assert "too few farmland copies" in r["reason"]
    none_fit = sg.record_change(T, decay, pace_null=[])      # a region too large for any copy of itself
    assert none_fit["method"] == "rate_x_span" and none_fit["pace_change_tested"] is False


def test_a_dict_of_named_copies_is_refused_not_read_as_an_empty_null():
    """Reviewer, 7 Oct 2026: {name: series} would otherwise pass as a null
    dict with no copies and silently give an untested rate x span."""
    y = -400.0 * (1 - np.exp(-T / 1.0)) + _farm(16)
    named = {f"pl_{s}": _farm(s) for s in range(600, 625)}
    with pytest.raises(ValueError):
        sg.record_change(T, y, pace_null=named)
    r = sg.record_change(T, y, pace_null=list(named.values()))
    assert r["pace_change_tested"] is True and r["pace_null_n"] == 25


def test_without_a_null_record_change_is_unchanged_and_says_it_was_not_tested():
    y = -8.0 * T + _farm(13)
    r = sg.record_change(T, y)
    assert r["pace_change_tested"] is False and r["pace_null_p95_mm"] is None
    # the old gate: hinges when the F-test fires
    assert r["method"] == ("piecewise_linear_yearly" if r["curvature_p"] < 0.05 else "rate_x_span")


def test_pace_change_null_matches_the_statistic_and_dict_or_series_agree():
    null = [_farm(s) for s in range(500, 525)]
    stats = [sg.pace_change_statistic(T, s) for s in null]
    nl = sg.pace_change_null(T, null)
    # 25 copies: a 5% test must beat the largest (ceil(0.95 x 26) = 25th smallest)
    assert nl["n"] == 25 and nl["p95_mm"] == pytest.approx(max(stats), abs=0.006)
    assert nl["values_mm"] == sorted(nl["values_mm"])
    y = -400.0 * (1 - np.exp(-T / 1.5)) + _farm(14)
    a, b = sg.record_change(T, y, pace_null=nl), sg.record_change(T, y, pace_null=null)
    assert a == b
    # the statistic is |piecewise - rate x span| of record_change's own model
    r = sg.record_change(T, y)
    assert sg.pace_change_statistic(T, y) == pytest.approx(abs(r["piecewise_change_mm"] - r["linear_change_mm"]), abs=0.1)


def test_pace_threshold_is_a_5pct_test_with_few_copies():
    """Reviewer, 7 Oct 2026: np.quantile's interpolated 95th percentile of 20-40
    copies is exceeded by an ordinary (exchangeable) footprint 7-9% of the time;
    the threshold used must keep that at or below 5%."""
    rng = np.random.default_rng(21)
    for n, k in ((20, 20), (30, 30), (40, 39)):
        draws = rng.standard_normal((20000, n + 1))
        thr = np.array([sg._rank_threshold(row[:n]) for row in draws])
        assert np.array_equal(thr, np.sort(draws[:, :n], axis=1)[:, k - 1])
        assert float(np.mean(draws[:, n] > thr)) <= 0.05 + 0.005
        assert float(np.mean(draws[:, n] > np.quantile(draws[:, :n], 0.95, axis=1))) > 0.065
    assert sg._rank_threshold(rng.standard_normal(18)) is None          # 1 in 19 cannot reach 5%


def test_a_pace_change_between_the_interpolated_and_the_calibrated_percentile_is_not_a_change():
    null = [_farm(s) for s in range(700, 720)]                           # 20 copies
    stats = np.array([sg.pace_change_statistic(T, s) for s in null])
    lo, hi = float(np.quantile(stats, 0.95)), float(stats.max())
    y = next((-8.0 * T + _farm(s) for s in range(720, 1200)
              if lo + 0.05 < sg.pace_change_statistic(T, -8.0 * T + _farm(s)) < hi - 0.05), None)
    assert y is not None
    r = sg.record_change(T, y, pace_null=null)
    assert r["pace_change_tested"] is True and r["pace_null_p95_mm"] == pytest.approx(hi, abs=0.006)
    assert r["method"] == "rate_x_span" and r["nonlinear"] is False


# ------------------------------------------------------------------ E28 / C2
class _Cfg:
    deramp_exclude_geojson = ""
    acceptance_far_field_buffer_m = 2000.0

    def __init__(self, shape):
        ny, nx = shape
        self.bbox = (ATTRS_100M["X_FIRST"], ATTRS_100M["Y_FIRST"] - 100.0 * ny,
                     ATTRS_100M["X_FIRST"] + 100.0 * nx, ATTRS_100M["Y_FIRST"])


def test_far_field_mask_excludes_report_areas_from_the_comparison_ground():
    rng = np.random.default_rng(15)
    shape = (200, 200)
    good = np.ones(shape, bool)
    tc = np.full(shape, 0.5)
    area = np.zeros(shape, bool); area[20:180, 20:180] = True        # a large report area on farmland
    vel = rng.normal(0, 0.5, shape) + np.where(area, 3.0, 0.0)       # moving 3 mm/yr against its surroundings
    cfg = _Cfg(shape)
    far0, m0 = sg.far_field_mask(cfg, good, ATTRS_100M)
    far1, m1 = sg.far_field_mask(cfg, good, ATTRS_100M, exclude=area)
    assert (far0 & area).sum() == area.sum() and m0["excluded_cells"] == 0
    assert not (far1 & area).any()
    # grown by the 500 m cut (5 cells at 100 m)
    assert not (far1 & sg.euclidean_buffer(area, ATTRS_100M, 500.0)).any()
    assert m1["excluded_cells"] == int((far0 & ~far1).sum()) and m1["exclude_buffer_m"] == 500.0
    # compared with itself the area reads attenuated; kept out it reads its true offset
    r0, _ = sg.class_matched_field(vel, tc, far0)
    r1, _ = sg.class_matched_field(vel, tc, far1)
    assert float(r0[area].mean()) < 1.5
    assert float(r1[area].mean()) == pytest.approx(3.0, abs=0.1)


def test_far_field_mask_exclude_none_or_empty_changes_nothing_and_checks_the_grid():
    shape = (80, 80)
    good = np.ones(shape, bool)
    cfg = _Cfg(shape)
    base, _ = sg.far_field_mask(cfg, good, ATTRS_100M)
    same, meta = sg.far_field_mask(cfg, good, ATTRS_100M, exclude=np.zeros(shape, bool))
    assert np.array_equal(base, same) and meta["excluded_cells"] == 0 and meta["exclude_buffer_m"] is None
    wide, _ = sg.far_field_mask(cfg, good, ATTRS_100M, restrict_to_aoi=False, exclude=None)
    assert wide.sum() >= base.sum()
    with pytest.raises(ValueError):
        sg.far_field_mask(cfg, good, ATTRS_100M, exclude=np.zeros((10, 10), bool))
