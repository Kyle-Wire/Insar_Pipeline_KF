"""Tests for significance.py: the conventions and the empirical null behave as stated."""
import numpy as np
import pytest

from insar_pipeline import significance as sg

ATTRS = {"X_FIRST": -87.86474054101562, "Y_FIRST": 38.510596064453125,
         "X_STEP": 0.000172, "Y_STEP": -0.000135, "LENGTH": 2609, "WIDTH": 2367}


def test_lalo_to_rc_matches_mintpy_floor():
    # the Site A configured reference: MintPy's geo lalo2yx gives (1616, 1784)
    assert sg.lalo_to_rc(ATTRS, 38.292436, -87.557893) == (1616, 1784)


def test_rc_to_lalo_is_pixel_centre_and_round_trips():
    lat, lon = sg.rc_to_lalo(ATTRS, 1618, 1784)
    assert lat == pytest.approx(38.29209856445313)          # MintPy yx2lalo
    assert lon == pytest.approx(-87.55780654101562)
    assert sg.lalo_to_rc(ATTRS, float(lat), float(lon)) == (1618, 1784)


def test_reference_pixel_prefers_product_attributes():
    a = dict(ATTRS, REF_Y="1618", REF_X="1784", REF_LAT="38.292114", REF_LON="-87.55792")

    class Cfg:
        reference_lalo = "38.292436,-87.557893"
    ref = sg.reference_pixel(a, Cfg())
    assert (ref["row"], ref["col"]) == (1618, 1784)
    assert "REF_Y" in ref["source"]
    ref2 = sg.reference_pixel(ATTRS, Cfg())
    assert (ref2["row"], ref2["col"]) == (1616, 1784)


def test_euclidean_buffer_is_round_not_a_diamond():
    a = {"X_FIRST": 500000.0, "Y_FIRST": 4_000_000.0, "X_STEP": 10.0, "Y_STEP": -10.0}
    m = np.zeros((201, 201), bool); m[100, 100] = True
    b = sg.euclidean_buffer(m, a, 500.0)
    # a diagonal cell at 45 deg, 480 m away (34 px each way) must be inside
    assert b[100 + 34, 100 + 34]
    # and one at 520 m must not
    assert not b[100 + 37, 100 + 37]


def test_coherence_class_edges():
    tc = np.array([0.3, 0.4, 0.59, 0.6, 0.95, 1.0, np.nan])
    assert list(sg.coherence_class(tc)) == [-1, 0, 0, 1, 4, 4, -1]


def test_class_matched_field_cancels_a_coherence_dependent_bias():
    rng = np.random.default_rng(0)
    tc = rng.uniform(0.4, 1.0, (200, 200))
    bias = -12 * (1 - tc)                           # the DS drift, a function of coherence only
    vel = bias + rng.normal(0, 0.1, tc.shape)
    far = np.ones_like(tc, bool); far[80:120, 80:120] = False
    r, means = sg.class_matched_field(vel, tc, far)
    # a low-coherence region with no motion reads ~0 class-matched, but strongly negative plain
    region = np.zeros_like(far); region[80:120, 80:120] = True
    low = region & (tc < 0.6)
    assert abs(r[low].mean()) < 1.0
    assert vel[low].mean() - vel[far].mean() < -2.0


def test_footprint_null_matches_white_noise_theory():
    rng = np.random.default_rng(1)
    sigma, n = 5.0, 100
    field = rng.normal(0, sigma, (400, 400))
    far = np.ones_like(field, bool)
    region = np.zeros_like(far); region[10:20, 10:20] = True           # 100 cells
    out = sg.assess(field, far, region)
    assert out["farmland_spread_1sigma_mm_yr"] == pytest.approx(sigma / np.sqrt(n), rel=0.1)
    assert out["method"].startswith("region footprint")


def test_assess_flags_a_real_signal_and_not_noise():
    rng = np.random.default_rng(2)
    field = rng.normal(0, 3.0, (300, 300))
    far = np.ones_like(field, bool); far[100:140, 100:140] = False
    region = np.zeros_like(far); region[110:130, 110:130] = True
    quiet = sg.assess(field, far, region)
    assert not quiet["distinguishable_from_farmland"]
    field2 = field.copy(); field2[region] -= 5.0                        # 5 mm/yr against a 0.15 spread
    loud = sg.assess(field2, far, region)
    assert loud["distinguishable_from_farmland"] and loud["z"] < -10


def test_fit_rate_ar1_inflates_for_correlated_residuals():
    rng = np.random.default_rng(3)
    t = np.arange(250) / 25.0
    e = np.zeros(250)
    for i in range(1, 250):
        e[i] = 0.8 * e[i - 1] + rng.normal(0, 3)
    f = sg.fit_rate(t, 2.0 * t + e)
    assert f["lag1_autocorrelation"] > 0.6
    assert f["stderr_ar1"] > 2.5 * f["stderr_white"]
    assert sg.trend_total(f) == pytest.approx(f["slope"] * (t[-1] - t[0]))


def test_empirical_sigma_map_falls_back_to_nearest_class():
    model = {"by_class": [{"class_index": 0, "empirical_mm_yr": 5.5}, {"class_index": 4, "empirical_mm_yr": 2.0}]}
    m = sg.empirical_sigma_map(model, np.array([0.45, 0.65, 0.95, 0.2]))
    assert list(m[:3]) == [5.5, 5.5, 2.0] and np.isnan(m[3])
