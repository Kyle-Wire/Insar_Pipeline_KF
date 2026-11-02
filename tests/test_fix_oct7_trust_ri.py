"""Trust review of 7 Oct 2026, group ri (report_insights.py): decisions

E1   "ground reshaped since the elevation model" is tested after removing
     what the movement itself puts into MintPy's DEM error (dem_change)
E2   "fastest in <years>" is tested against the same pick of years made on
     steady movement plus farmland's noise (trend_shape)
E7   "stands out clearly" needs the calibrated chance test AND the QA
     stand-out; a QA patch is no longer excused from STRONG (find_features,
     selection_p, feature_text)
E8   copies keep the farmland's measured correlation out to 4 km and say
     so; "less than 1 in 60", never "about 0.02"; real farmland stronger
     than every copy withholds "clearly" (feature_null, feature_null_check,
     p_text)
E11  the feature's recent-rate clause carries its 95% range and a direction
     only when the range excludes zero
E14  the single-cell shares on the report's own (second-pass) frame
     (cells_standing_out)

Each test checks an outcome on synthetic data with a known answer and failed
on the code before these fixes.

Run: python -m pytest insar_pipeline/tests/test_fix_oct7_trust_ri.py -q -p no:cacheprovider
"""
import types

import numpy as np
from scipy.ndimage import gaussian_filter

from insar_pipeline import report_insights as ri
from insar_pipeline import significance as sg


# ------------------------------------------------------------------ helpers
def _ctx(field, far=None, pixel=60.0):
    ny, nx = field.shape
    dlat = pixel / 111320.0
    dlon = pixel / (111320.0 * np.cos(np.radians(38.3)))
    attrs = {"X_FIRST": "-87.6", "Y_FIRST": "38.4", "X_STEP": str(dlon), "Y_STEP": str(-dlat)}
    good = np.ones((ny, nx), bool)
    far = good.copy() if far is None else far
    return types.SimpleNamespace(sg=sg, attrs=attrs, good=good, far=far, far_wide=far, r_cm=field, r_null=field)


def _dates():
    t = np.arange(2016.0, 2026.0, 12 / 365.25)
    return t, [int(x) for x in t]


def _series_from_yearly(t, years, rate_by_year):
    s = np.zeros(t.size)
    for k in range(1, t.size):
        s[k] = s[k - 1] + rate_by_year[years[k]] * (t[k] - t[k - 1])
    return s


def _feature(**kw):
    f = {"sign": -1, "area_km2": 2.0, "lat": 38.3, "lon": -87.4, "mean_rel_vertical_mm_yr": -30.0,
         "vertical_relative_mm_yr": -30.0, "total_relative_mm": -300.0, "nonlinear": True,
         "trend": {"shape": "slowing", "text": ""}, "mines": {"key": "none", "phrase": "away from every mapped mine"},
         "chance": {"tested": True, "p": 0.0167, "stands_out": True, "copies_as_strong": 0, "copies": 59},
         "quality_standout": False, "cause_hint": "x."}
    f.update(kw)
    return f


# ------------------------------------------------------------------ E1: DEM change with the movement taken out
def _dem_case(extra_m):
    """Farmland whose 500 m DEM error follows its rate (0.25 m per mm/yr, the
    Site B farmland fit) with 3 m of scatter; a sinking patch at -60
    mm/yr whose DEM error is what that rate explains (-15 m) plus extra_m."""
    rng = np.random.default_rng(11)
    ny = nx = 300
    rate = gaussian_filter(rng.normal(0, 1, (ny, nx)), 4)
    rate = rate / rate.std() * 8.0
    dem = 0.25 * rate + rng.normal(0, 3.0, (ny, nx))
    feat = np.zeros((ny, nx), bool)
    feat[100:160, 100:160] = True
    rate[feat] = -60.0 + rng.normal(0, 3.0, feat.sum())
    dem[feat] = 0.25 * rate[feat] + extra_m + rng.normal(0, 3.0, feat.sum())
    return dem.astype(np.float32), rate.astype(np.float32), feat, ~feat


def test_dem_change_takes_the_movement_out_before_flagging():
    """Site B Feature 1 (E1): -14.9 m raw, flagged at 3 robust sd; -7.2 m
    after its rate (1.8 sd), not. Here the patch's DEM error is all movement
    (-15 m from -60 mm/yr): the raw gate fires, the residual one must not;
    with 25 m more on top (earthmoving) it must."""
    dem, rate, feat, far = _dem_case(0.0)
    d = ri.dem_change(dem, rate, feat, far)
    assert d["raw_flag"] is True and abs(d["feature_median_m"]) > d["threshold_m"]
    assert d["rate_removed"] is True and d["flag"] is False
    assert 0.2 < d["slope_m_per_mm_yr"] < 0.3 and abs(d["residual_m"]) < d["residual_threshold_m"]
    dem2, rate2, feat2, far2 = _dem_case(-25.0)
    d2 = ri.dem_change(dem2, rate2, feat2, far2)
    assert d2["flag"] is True and d2["residual_m"] < -20
    # no rate field: no test, no claim (the raw numbers kept)
    d3 = ri.dem_change(dem, None, feat, far)
    assert d3["flag"] is False and d3["rate_removed"] is False and d3["raw_flag"] is True


def test_reshaped_sentence_and_earthmoving_hint_need_the_rate_removed_test():
    raw_only = {"flag": True, "feature_median_m": -14.8, "farmland_median_m": -0.1, "farmland_robust_sd_m": 4.12,
                "threshold_m": 12.4, "smoothing_m": 500.0}            # client_report._dem_change, the old gate
    hint = ri.cause_hint("near_mapped", -1, "slowing", reshaped_since_dem=True)
    assert "earthmoving" in hint
    t = ri.feature_text(1, _feature(dem_change=raw_only, ground_reshaped_since_dem=True, cause_hint=hint))
    assert "reshaped since the elevation model" not in t and "earthmoving" not in t
    assert "beside mapped mining" in t                                 # the rest of the hint stays
    dem2, rate2, feat2, far2 = _dem_case(-25.0)
    d2 = ri.dem_change(dem2, rate2, feat2, far2)
    t2 = ri.feature_text(1, _feature(dem_change=d2, ground_reshaped_since_dem=True, cause_hint=hint))
    assert "Its ground was reshaped since the elevation model was made" in t2 and "earthmoving" in t2
    assert "from the movement alone" in t2 and "the other way" not in t2


def test_reshaped_sentence_adds_up_when_the_movement_points_the_other_way():
    """Fill on ground whose rate puts -12 m into the DEM error: it reads +2 m,
    a residual of +14 m. The sentence said 'the remaining 14 m' of a 2 m
    difference; the three figures must add up and say which way."""
    dc = {"flag": True, "rate_removed": True, "raw_flag": False, "feature_median_m": 1.9, "farmland_median_m": -0.1,
          "farmland_robust_sd_m": 4.1, "threshold_m": 12.4, "smoothing_m": 500.0, "slope_m_per_mm_yr": 0.27,
          "intercept_m": -0.2, "feature_rate_mm_yr": -45.0, "expected_from_rate_m": -12.3, "residual_m": 14.2,
          "residual_robust_sd_m": 4.0, "residual_threshold_m": 12.0, "residual_sigmas": 3.55}
    t = ri.feature_text(1, _feature(dem_change=dc, ground_reshaped_since_dem=True))
    assert ("its height differs from the model by about 2 m, where ordinary farmland moving as fast as this differs "
            "by about 12 m the other way from the movement alone. That leaves about 14 m the movement does not "
            "explain, more than ordinary farmland shows (under 12 m).") in t
    assert "remaining" not in t
    dc_same = dict(dc, feature_median_m=-24.1, residual_m=-12.6)
    t = ri.feature_text(1, _feature(dem_change=dc_same, ground_reshaped_since_dem=True))
    assert ("differs from the model by about 24 m, where ordinary farmland moving as fast as this differs by about "
            "12 m from the movement alone. That leaves about 13 m") in t


# ------------------------------------------------------------------ E2: "fastest in" against its own selection
def test_fastest_years_are_not_named_for_steady_movement():
    """Steady movement (constant true rate, independent yearly noise, null
    rows from the same noise): the old fixed-weight test named 'fastest' in
    43-66% of runs (audit_features_verify/trend_v.py); now about 2% (the
    null's 99% point, measured 1.2-2.2% with 20-40 placements)."""
    t, years = _dates()
    used = sorted(set(years))
    rng = np.random.default_rng(5)
    named = 0
    runs = 150
    for _ in range(runs):
        e = rng.normal(0, 5.5, len(used))
        s = _series_from_yearly(t, years, {y: -37.3 + e[i] for i, y in enumerate(used)})
        rows = rng.normal(0, 5.5, (40, len(used))) + rng.normal(0, 3, (40, 1))
        out = ri.trend_shape(t, s, years, noise_mm_yr=1.0, null_yearly={"years": used, "rates": rows})
        named += "fastest" in out
    assert named <= 0.05 * runs


def test_site_a_feature1_loses_fastest_and_site_b_feature1_keeps_it():
    """The delivered yearly rates. Site A Feature 1 (11 placements, z 3.3)
    printed 'fastest in 2019-2022'; Site B Feature 1 (40 placements, z
    7.2) keeps 'fastest in 2019-2020'."""
    t, years = _dates()
    used = sorted(set(years))
    ow = dict(zip(used, [-3.1, -11.8, -14.0, -16.1, -18.2, -19.6, -19.2, -12.0, -9.6, -8.5]))
    rng = np.random.default_rng(1)
    rows11 = rng.normal(0, 4.2, (11, len(used)))
    d = ri.trend_shape(t, _series_from_yearly(t, years, ow), years, noise_mm_yr=1.0,
                       null_yearly={"years": used, "rates": rows11})
    assert "fastest" not in d and "fastest in" not in d["text"]
    assert "withheld" in d["contrasts"]["fastest"]
    fr = dict(zip(used, [-16.4, -22.5, -48.0, -75.0, -54.1, -39.9, -42.5, -19.4, -8.1, -11.5]))
    rows40 = rng.normal(0, 7.2, (40, len(used)))
    d = ri.trend_shape(t, _series_from_yearly(t, years, fr), years, noise_mm_yr=1.0,
                       null_yearly={"years": used, "rates": rows40})
    assert d["fastest"] == [2019, 2020] and "fastest in 2019–2020" in d["text"]
    c = d["contrasts"]["fastest"]
    assert c["z"] > c["null_z_q"] > ri.CONTRAST_Z


# ------------------------------------------------------------------ E7: both tests for "stands out clearly"
def test_quality_standout_no_longer_excuses_a_patch_from_strong():
    """A shallow patch (peak under STRONG band widths) holding the QA
    stand-out was kept as a candidate before; the copies only count STRONG
    patches, so it cannot be tested like them and is not a candidate now."""
    f = gaussian_filter(np.random.default_rng(7).normal(0, 1, (160, 160)), 5)
    f = f / f.std() * 3.0
    yy, xx = np.mgrid[:160, :160]
    f = f - 9.0 * np.exp(-((yy - 80) ** 2 + (xx - 80) ** 2) / (2 * 10.0 ** 2))
    ctx = _ctx(f)
    sm = ri.smoothed_field(ctx, 1.0)
    lat = float(ctx.attrs["Y_FIRST"]) + 80.5 * float(ctx.attrs["Y_STEP"])
    lon = float(ctx.attrs["X_FIRST"]) + 80.5 * float(ctx.attrs["X_STEP"])
    cands = ri.find_features(ctx, sm, {}, np.ones(f.shape, bool), {"lat": lat, "lon": lon}, limit=None)
    c, width = sm["centre_mm_yr"], ri._side(sm, -1)[1]
    assert sm["field"][80, 80] - c > -ri.STRONG * width                # the bowl is under STRONG here
    assert not any(x["quality_standout"] for x in cands)
    for x in cands:
        assert x["sign"] * (x["peak_rel_vertical_mm_yr"] - c) >= ri.STRONG * x["band_mm_yr"] - 0.1


def test_every_calibrated_standout_marks_its_patch_and_a_failed_one_none():
    """The QA now keeps every scale that passes its calibrated rule, each
    with its own location (Site B: 1 km in Feature 1, 2 km in Feature
    2). find_features takes that list; an entry that did not pass
    (stands_out False, or None when the calibration could not run) marks
    nothing - before, any dict passed in marked its patch."""
    f = np.zeros((120, 120))
    f[20:40, 20:40] = -40.0
    f[70:95, 70:95] = -40.0
    ctx = _ctx(f, far=np.zeros(f.shape, bool))
    a = ctx.attrs
    sm = {"field": f, "centre_mm_yr": 0.0, "band_lo_mm_yr": -10.0, "band_hi_mm_yr": 10.0, "band_mm_yr": 10.0,
          "band_ok": True, "pixel_m": (60.0, 60.0)}

    def at(r, c, **kw):
        return dict({"lat": float(a["Y_FIRST"]) + (r + 0.5) * float(a["Y_STEP"]),
                     "lon": float(a["X_FIRST"]) + (c + 0.5) * float(a["X_STEP"]), "stands_out": True}, **kw)
    got = ri.find_features(ctx, sm, {}, np.ones(f.shape, bool), [at(30, 30, scale_km=1.0), at(80, 80, scale_km=2.0)],
                           limit=None)
    assert len(got) == 2 and all(x["quality_standout"] for x in got)
    got = ri.find_features(ctx, sm, {}, np.ones(f.shape, bool), at(30, 30, stands_out=False), limit=None)
    assert len(got) == 2 and not any(x["quality_standout"] for x in got)
    got = ri.find_features(ctx, sm, {}, np.ones(f.shape, bool), [at(80, 80, stands_out=None)], limit=None)
    assert not any(x["quality_standout"] for x in got)


def _null(real=0.0, top=55.5):
    return {"tested": True, "null_area_km2": 300.0, "searched_area_km2": 400.0,
            "sides": {"sinking": {"tested": True, "max_mass": list(np.linspace(5.0, top, 59)),
                                  "median_max_beyond_mm_yr": 18.8, "farmland_strongest_mass": real},
                      "rising": {"tested": True, "max_mass": list(np.linspace(3.0, 20.0, 59)),
                                 "median_max_beyond_mm_yr": 13.0, "farmland_strongest_mass": 0.0}}}


def _sm():
    return {"centre_mm_yr": 0.0, "band_lo_mm_yr": -10.0, "band_hi_mm_yr": 8.0}


def test_stands_out_clearly_needs_both_tests_and_says_less_than_1_in_60():
    strong = {"sign": -1, "area_km2": 6.0, "beyond_farmland_mm_yr": 26.0}     # mass 96 > every copy
    weak = {"sign": -1, "area_km2": 2.0, "beyond_farmland_mm_yr": 30.0}       # mass 40: copies reach it
    for qs, f0, expect in ((True, strong, True), (False, strong, False), (True, weak, False)):
        f = dict(f0, quality_standout=qs)
        ch = ri.selection_p(f, _sm(), _null())
        assert f["stands_out_clearly"] is expect and ch["stands_out_clearly"] is expect
    ch = ri.selection_p(dict(strong, quality_standout=True), _sm(), _null())
    assert ch["p_text"] == "less than 1 in 60" and ch["copies_as_strong"] == 0
    weak_ch = ri.selection_p(dict(weak, quality_standout=True), _sm(), _null())
    assert weak_ch["stands_out"] is False and weak_ch["p_text"].startswith("about ") and "in 60" in weak_ch["p_text"]
    # the paragraph follows the label, and never prints "about 0.02"
    f = _feature(quality_standout=True, stands_out_clearly=True, chance=ch)
    t = ri.feature_text(1, f)
    assert "stands out clearly" in t and "less than 1 in 60" in t and "about 0.02" not in t
    f = _feature(quality_standout=False, stands_out_clearly=False, chance=ri.selection_p(dict(strong), _sm(), _null()))
    t = ri.feature_text(1, f)
    assert "worth a look" in t and "stands out clearly" not in t and "about 0.02" not in t
    # a QA stand-out the calibrated test does not support is within chance, not "clearly"
    f = _feature(quality_standout=True, stands_out_clearly=False, chance=weak_ch)
    t = ri.feature_text(1, f)
    assert "stands out clearly" not in t and "passes the quality assessment" not in t and "within chance" in t


def test_a_record_without_the_decision_is_never_clearly_and_chance_gets_no_cause():
    """A feature built before selection_p decided "clearly" (its QA
    stand-out was the uncalibrated gap rule) is worth a look at most; a
    feature within chance is given no cause."""
    ch = {"tested": True, "p": 0.0167, "stands_out": True, "copies_as_strong": 0, "copies": 59}
    t = ri.feature_text(1, _feature(quality_standout=True, chance=ch, cause_hint="Subsidence over workings."))
    assert "worth a look" in t and "stands out clearly" not in t and "passes the quality assessment" not in t
    assert "less than 1 in 60" in t and "Subsidence over workings." in t
    ch = {"tested": True, "p": 0.4, "stands_out": False, "copies_as_strong": 23, "copies": 59}
    t = ri.feature_text(1, _feature(chance=ch, cause_hint="Subsidence over workings."))
    assert "within chance" in t and "about 24 in 60" in t and "Subsidence over workings." not in t


# ------------------------------------------------------------------ E8: the null against the real farmland
def test_real_farmland_stronger_than_every_copy_withholds_clearly():
    """Site A (7 Oct): the comparison ground's own strongest patch 87.1
    against a copy maximum of 61.2. The null is mis-specified for that
    ground: a reviewer item, and nothing is called 'stands out clearly'."""
    chk = ri.feature_null_check(_null(real=87.1, top=61.2))
    assert chk["real_beats_all_copies"] is True and chk["sides"] == ["sinking"] and "87.1" in chk["why"]
    assert ri.feature_null_check(_null(real=40.0, top=61.2))["real_beats_all_copies"] is False
    assert ri.feature_null_check(_null(real=0.0, top=0.0))["real_beats_all_copies"] is False   # no patch, no claim
    assert ri.feature_null_check({"tested": False, "reason": "x"})["real_beats_all_copies"] is False
    f = {"sign": -1, "area_km2": 23.7, "beyond_farmland_mm_yr": 16.9, "quality_standout": True}
    ch = ri.selection_p(f, {"centre_mm_yr": 0.0, "band_lo_mm_yr": -12.0, "band_hi_mm_yr": 8.0},
                        _null(real=87.1, top=61.2))
    assert ch["stands_out"] is True and f["stands_out_clearly"] is False and ch["clearly_withheld"]
    t = ri.feature_text(1, _feature(quality_standout=True, stands_out_clearly=False, chance=ch))
    assert "worth a look" in t and "stands out clearly" not in t
    assert "holds a patch stronger than any of the simulated copies, so the chance figure may be too low" in t


def test_copies_keep_the_measured_correlation_out_to_4_km():
    """The copies' correlation was the farmland's times a cosine over 0-3 km
    (half at 1.5 km, a quarter at 2 km: 0.59 and 0.11 here against the
    field's 0.80 and 0.44). On a 30 km field correlated over ~2 km they now
    match it at 1 and 2 km, and feature_null records the check."""
    rng = np.random.default_rng(3)
    f = gaussian_filter(rng.normal(0, 1, (500, 500)), 20)
    f = f / f.std() * 3.0
    ctx = _ctx(f)
    sm = ri.smoothed_field(ctx, 1.0)
    null = ri.feature_null(ctx, sm, [], np.ones(f.shape, bool), n_surrogates=5)
    cc = null["covariance_check"]
    lags = cc["lags_m"]
    for lag in (1000, 2000):
        i = lags.index(lag)
        # the field's own correlation (Gaussian kernel of 20 cells: exp(-d^2 / 4 sigma^2) = 0.84, 0.50)
        expect = np.exp(-(lag / 60.0) ** 2 / (4 * 20.0 ** 2))
        assert abs(cc["farmland"][i] - expect) < 0.1
        assert cc["farmland"][i] > 0.3 and abs(cc["farmland"][i] - cc["copies"][i]) <= 0.05
    assert null["feature_null_check"]["real_beats_all_copies"] in (True, False)
    assert "covariance_matched" in null["feature_null_check"]


def test_p_text_is_a_bound_when_every_copy_is_beaten():
    assert ri.p_text(0, 59) == "less than 1 in 60"
    assert ri.p_text(2, 59) == "about 3 in 60"


# ------------------------------------------------------------------ E11: recent rate words
def test_recent_clause_has_a_direction_only_when_its_range_excludes_zero():
    win = ["20240709", "20260617"]
    f = _feature(recent_rate_mm_yr=-9.6, recent_range95_mm_yr=[-25.56, 6.39], recent_window=win)
    t = ri.feature_text(1, f)
    assert "over 2024–2026 its rate cannot be told apart from no movement (95% range -1.01 to +0.25 in/yr)" in t
    assert "0.38 in/yr" not in t
    f = _feature(recent_rate_mm_yr=-10.5, recent_range95_mm_yr=[-20.5, -0.45], recent_window=win)
    t = ri.feature_text(1, f)
    assert "about 0.41 in/yr (10 mm/yr) down over 2024–2026 (95% range -0.81 to -0.02 in/yr)" in t
    f = _feature(recent_rate_mm_yr=-10.5, recent_range95_mm_yr=None, recent_window=win)
    assert "0.41" not in ri.feature_text(1, f)


def test_recent_clause_word_follows_the_range_not_a_value_across_zero():
    """A range on the test's frame (the excess over farmland's median) can
    exclude zero on the other side of the measured value: the word follows
    the range and the value is not printed beside it."""
    win = ["20240709", "20260617"]
    t = ri.feature_text(1, _feature(recent_rate_mm_yr=-0.5, recent_range95_mm_yr=[0.3, 12.0], recent_window=win))
    assert "over 2024–2026 it moved up (95% range +0.01 to +0.47 in/yr)" in t and "0.02 in/yr" not in t


# ------------------------------------------------------------------ E14: single cells on the report's frame
def test_cells_standing_out_uses_the_given_second_pass_frame():
    """Site B: summary.json's first pass read 6.8% against 7.1% by chance;
    the second-pass S/N raster 6.7% against 5.6%. The shares come from the
    field and the comparison ground passed in, by the S/N raster's rule."""
    rng = np.random.default_rng(2)
    r = rng.normal(0, 1.0, (100, 100))
    null = np.full(r.shape, 1.0)
    far = np.zeros(r.shape, bool)
    far[:, :50] = True
    reported = np.ones(r.shape, bool)
    r[:, 50:60] = 5.0                                                     # moving ground, off the farmland
    d = ri.cells_standing_out(r, null, reported, far)
    assert abs(d["cells_standing_out_by_chance_percent"] - 5.0) < 1.5
    assert d["cells_standing_out_percent"] > d["cells_standing_out_by_chance_percent"] + 5
    s = ri.cells_sentence(d["cells_standing_out_percent"], d["cells_standing_out_by_chance_percent"])
    assert "against" in s and "same as chance" not in s
    # F7(a), follow-up of 7 Oct 2026: 'the same' only inside chance's 95% binomial
    # range for the radar cells (4% of 100 cells: 1-8%), never without a count
    assert "same as chance" in ri.cells_sentence(5.4, 4.0, 100) and ri.cells_sentence(None, 4.0) == ""
    assert "same as chance" not in ri.cells_sentence(5.4, 4.0)
