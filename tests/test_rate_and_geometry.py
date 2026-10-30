"""
Rate fitting, uncertainty, geometry projection and unwrap-spike rejection.

Each of these produces a number that looks reasonable whatever it does,
which is the failure mode this project keeps meeting. The tests below all
work the same way: build data whose right answer is known by construction,
then check the code recovers it.

Run: python tests/test_rate_and_geometry.py   (or: python -m pytest tests/)
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from insar_pipeline.disp_backend import (          # noqa: E402
    _design, _rate_and_sigma, _noise_inflation, _flag_unwrap_jumps,
    _fit, _term_index,
    _check_boundary_steps, _repair_boundary_steps,
)
from insar_pipeline.static_layers import (          # noqa: E402
    project_to_vertical, los_from_enu,
)

WAVELENGTH = 0.05546576
CYCLE = WAVELENGTH / 2.0


# ----------------------------------------------------------------- rate fit

def test_linear_rate_is_recovered_exactly_without_noise():
    t = np.linspace(0, 9, 100)
    truth = -0.012
    y = truth * t + 0.5
    rate, sigma = _rate_and_sigma(_design(t, False), y)
    assert np.isclose(rate, truth, rtol=1e-9)
    assert sigma < 1e-9


def test_seasonal_signal_biases_a_linear_fit_and_the_seasonal_model_fixes_it():
    """The reason disp_seasonal_model exists."""
    # Sample only the first half of each year, as a gappy farmland stack
    # over a cropped field can easily do. A pure linear fit then mistakes
    # part of the annual cycle for a trend.
    years = np.arange(0, 10)
    t = np.concatenate([years + f for f in (0.05, 0.15, 0.25, 0.35)])
    t.sort()
    truth = -0.012
    y = truth * t + 0.02 * np.sin(2 * np.pi * t)

    lin, _ = _rate_and_sigma(_design(t, False), y)
    sea, _ = _rate_and_sigma(_design(t, True), y)

    assert abs(sea - truth) < abs(lin - truth), (
        f"seasonal fit ({sea:.5f}) should beat linear ({lin:.5f}) "
        f"against truth {truth}")
    assert np.isclose(sea, truth, atol=1e-4)


def test_sigma_grows_with_noise_and_shrinks_with_record_length():
    rng = np.random.default_rng(0)
    t_short = np.linspace(0, 1, 30)
    t_long = np.linspace(0, 10, 300)
    quiet = 0.001
    loud = 0.010

    def sig(t, s):
        y = -0.012 * t + rng.normal(0, s, t.size)
        return _rate_and_sigma(_design(t, False), y)[1]

    assert sig(t_short, loud) > sig(t_short, quiet)
    assert sig(t_long, loud) < sig(t_short, loud), (
        "a decade must constrain a rate better than a single year")


def test_correlated_noise_widens_the_error_bar():
    """White residuals must not be claimed for a red-noise process."""
    rng = np.random.default_rng(1)
    n = 400
    white = rng.normal(0, 1, (n, 50))
    red = _ar1(rng, n, 50, 0.8)

    assert _noise_inflation(white) < 1.2
    assert _noise_inflation(red) > 1.8, "AR(1) r=0.8 implies ~3x widening"


def _ar1(rng, n, m, r):
    x = np.zeros((n, m))
    e = rng.normal(0, 1, (n, m))
    for i in range(1, n):
        x[i] = r * x[i - 1] + e[i]
    return x


# --------------------------------------------------------- unwrapping spikes

def _clean_stack(n=40, ny=6, nx=6, rate_per_epoch=-0.002):
    t = np.arange(n)[:, None, None]
    return (rate_per_epoch * t * np.ones((1, ny, nx))).astype("float64")


def test_a_reverting_whole_cycle_spike_is_flagged():
    ts = _clean_stack()
    ts[20, 3, 3] += CYCLE          # one epoch jumps a full cycle and returns
    bad, info = _flag_unwrap_jumps(ts, WAVELENGTH)
    assert bad[20, 3, 3], "the spiked epoch should be flagged"
    assert info["flagged"] == 1, f"only that epoch, got {info['flagged']}"


def test_multi_cycle_spikes_are_flagged_too():
    ts = _clean_stack()
    ts[15, 1, 1] += 3 * CYCLE
    bad, _ = _flag_unwrap_jumps(ts, WAVELENGTH)
    assert bad[15, 1, 1]


def test_real_motion_is_not_flagged():
    """A permanent step is ground movement, not an unwrapping error."""
    ts = _clean_stack()
    ts[25:, 2, 2] += CYCLE         # steps and STAYS - e.g. a real event
    bad, info = _flag_unwrap_jumps(ts, WAVELENGTH)
    assert not bad.any(), (
        f"a non-reverting step must not be flagged, got {info['flagged']}")


def test_a_smooth_stack_is_left_alone():
    ts = _clean_stack()
    bad, info = _flag_unwrap_jumps(ts, WAVELENGTH)
    assert info["flagged"] == 0
    assert not bad.any()


def test_sub_cycle_noise_is_not_mistaken_for_an_unwrapping_error():
    rng = np.random.default_rng(3)
    ts = _clean_stack(n=60, ny=10, nx=10)
    ts += rng.normal(0, 0.15 * CYCLE, ts.shape)   # 4 mm atmospheric noise
    _, info = _flag_unwrap_jumps(ts, WAVELENGTH)
    assert info["fraction"] < 0.01, (
        f"noise well below a cycle should not trip the detector, "
        f"got {info['fraction']:.3%}")


def test_realistic_atmospheric_noise_does_not_trip_the_detector():
    """
    The case that got through on a real run and had to be fixed.

    Successive differences of a noisy series are anti-correlated by
    construction, so "jumped and reverted" is what noise does. On a
    sparsely sampled stack whose scatter approaches a cycle, a detector
    built only on "near-integer and reverts" flagged 3.4% of all samples
    and threw away good data. 13 mm is the residual scatter measured on
    the real Site A stack.
    """
    rng = np.random.default_rng(7)
    ts = _clean_stack(n=40, ny=40, nx=40, rate_per_epoch=-0.003)
    ts += rng.normal(0, 0.013, ts.shape)          # 13 mm, as measured
    _, info = _flag_unwrap_jumps(ts, WAVELENGTH)
    assert info["fraction"] < 0.005, (
        f"realistic atmospheric noise must not read as unwrapping errors, "
        f"got {info['fraction']:.3%}")


def test_a_real_spike_is_still_caught_in_the_presence_of_noise():
    """
    The noise test above must not be passed by switching the detector off.

    Thresholds were tuned against this pair: 3 mm per-epoch noise is a
    reasonably coherent pixel, and 1-, 2- and 4-cycle spikes on it are all
    caught at tolerance 0.30 / n_sigma 3.0.
    """
    for k in (1, 2, 4):
        rng = np.random.default_rng(8)
        ts = _clean_stack(n=40, ny=12, nx=12, rate_per_epoch=-0.003)
        ts += rng.normal(0, 0.003, ts.shape)
        ts[20, 5, 5] += k * CYCLE
        bad, _ = _flag_unwrap_jumps(ts, WAVELENGTH)
        assert bad[20, 5, 5], f"a {k}-cycle spike on a quiet pixel must be caught"


def test_detection_degrades_when_noise_approaches_a_cycle():
    """
    A stated limit, not a bug.

    The reverting leg of a spike carries the noise of two epochs. Once
    that noise is a sizeable fraction of a cycle, the reversal stops
    looking like a whole number of cycles and the spike is missed. The
    detector is for stacks whose per-epoch scatter is well under a cycle;
    on noisier ones it finds little, which is the safe direction to fail.
    """
    rng = np.random.default_rng(11)
    ts = _clean_stack(n=40, ny=12, nx=12, rate_per_epoch=-0.003)
    ts += rng.normal(0, 0.40 * CYCLE, ts.shape)   # ~11 mm per epoch
    ts[20, 5, 5] += 2 * CYCLE
    _, info = _flag_unwrap_jumps(ts, WAVELENGTH)
    # It must stay QUIET rather than flag the whole scene.
    assert info["fraction"] < 0.01


# --------------------------------------------------- stitching-error steps
#
# INDEX CONVENTION. OPERA starts each new segment at the last secondary date
# of the one before - the ANCHOR, epoch a - and the carry copies ts[a] into
# every later epoch. The first epoch of the new segment is b = a + 1
# (assemble_stack's boundary_idx). A carried error therefore enters at a and
# stays: it is injected at ts[a:] here, and the checks take the ANCHORS.
# Until 5 Oct 2026 these tests injected at ts[b:] and checked b - one epoch
# late, a case the stitching cannot produce - so they passed while the real
# anchor errors went undetected.

BOUNDARIES = [9, 17, 25, 33]                 # first epoch of each new segment
ANCHORS = [b - 1 for b in BOUNDARIES]        # the epoch the carry copies


def test_noise_alone_shows_no_excess_at_boundaries():
    """
    The control that stops this check crying wolf.

    Noise produces whole-cycle-looking steps everywhere. The first version
    counted only at boundaries, reported 7% of the grid per boundary on a
    real stack, and that was almost entirely noise. Anchors must be
    compared against ordinary epochs.
    """
    rng = np.random.default_rng(21)
    ts = _clean_stack(n=40, ny=30, nx=30, rate_per_epoch=-0.003)
    ts += rng.normal(0, 0.013, ts.shape)          # realistic 13 mm
    info = _check_boundary_steps(ts, ANCHORS, WAVELENGTH)
    assert abs(info["excess"]) < 0.01, (
        f"pure noise must show no anchor excess, got "
        f"{info['excess']:.3%} (anchors {info['boundary_rate']:.3%}, "
        f"control {info['control_rate']:.3%})")


def test_real_stitching_errors_do_show_an_excess():
    """And the check must fire when the carried anchor values are wrong -
    and only when it looks at the anchors."""
    rng = np.random.default_rng(22)
    ts = _clean_stack(n=40, ny=30, nx=30, rate_per_epoch=-0.003)
    ts += rng.normal(0, 0.003, ts.shape)
    # A fifth of the grid inherits a whole cycle at each anchor.
    for a in ANCHORS:
        ts[a:, :6, :] += CYCLE
    info = _check_boundary_steps(ts, ANCHORS, WAVELENGTH)
    assert info["excess"] > 0.01, (
        f"injected stitching errors must show up, got {info['excess']:.3%}")
    hits = info["suspect_count"]
    assert (hits[:6] > 0).all() and hits[:6].mean() > 3.0 and (hits[6:] > 0).mean() < 0.01
    # The old index (the new segment's first epoch) sees nothing of it.
    assert _check_boundary_steps(ts, BOUNDARIES, WAVELENGTH)["suspect_pixels"] == 0


def test_whole_cycle_step_on_a_boundary_is_reported():
    """
    The case _flag_unwrap_jumps deliberately cannot see.

    A carry that inherited an unwrapping error puts a permanent step into
    every later epoch. Permanent steps look like real movement, so the
    only thing distinguishing this one is that it lands exactly on the
    anchor OPERA re-referenced to.
    """
    ts = _clean_stack(n=40)
    a = 20
    ts[a:, 2, 2] += CYCLE                    # permanent, from the anchor on
    info = _check_boundary_steps(ts, [a], WAVELENGTH)
    assert info["boundaries_checked"] == 1
    assert info["suspect_pixels"] == 1 and info["pixels_with_any"] == 1
    assert _check_boundary_steps(ts, [a + 1], WAVELENGTH)["suspect_pixels"] == 0

    # The spike detector must NOT see it - that is why this check exists.
    bad, _ = _flag_unwrap_jumps(ts, WAVELENGTH)
    assert not bad.any()


def test_a_step_away_from_a_boundary_is_not_reported():
    ts = _clean_stack(n=40)
    ts[20:, 2, 2] += CYCLE
    assert _check_boundary_steps(ts, [30], WAVELENGTH)["suspect_pixels"] == 0


def test_sub_cycle_motion_across_a_boundary_is_not_reported():
    """Real ground movement is not a whole number of cycles."""
    ts = _clean_stack(n=40)
    ts[20:, 2, 2] += 0.4 * CYCLE
    assert _check_boundary_steps(ts, [20], WAVELENGTH)["suspect_pixels"] == 0


def test_repair_removes_injected_stitching_errors_and_leaves_the_trend():
    """
    The repair must remove the steps WITHOUT eating the real signal.

    Two checks, because passing only the first is easy: the anchor
    excess must go away, AND the recovered rate must still match the rate
    that was put in.
    """
    rng = np.random.default_rng(31)
    n, rate = 40, -0.003
    ts = _clean_stack(n=n, ny=24, nx=24, rate_per_epoch=rate)
    ts += rng.normal(0, 0.003, ts.shape)
    for a in ANCHORS:
        ts[a:, :8, :] += CYCLE          # a third of the grid inherits a cycle

    before = _check_boundary_steps(ts, ANCHORS, WAVELENGTH)
    assert before["excess"] > 0.01

    rep = _repair_boundary_steps(ts, ANCHORS, WAVELENGTH)
    after = _check_boundary_steps(ts, ANCHORS, WAVELENGTH)

    assert rep["repaired_pixels"] > 0
    assert after["excess"] < 0.01, f"excess remains: {after['excess']:.3%}"

    # The affected pixels must still carry the rate they started with.
    t = np.arange(n, dtype="float64")
    A = np.vstack([t, np.ones(n)]).T
    recovered = np.linalg.lstsq(A, ts[:, :8, :].reshape(n, -1), rcond=None)[0][0]
    assert np.allclose(np.median(recovered), rate, atol=2e-4), (
        f"repair changed the rate: {np.median(recovered):.5f} vs {rate}")


def test_repair_leaves_a_reverting_spike_after_the_anchor_alone():
    """A spike in the new segment's first epoch reverts by itself; the old
    repair (which tested that epoch) made it a permanent -1 cycle."""
    rng = np.random.default_rng(33)
    ts = _clean_stack(n=40, ny=12, nx=12, rate_per_epoch=-0.003)
    ts += rng.normal(0, 0.002, ts.shape)
    truth = ts.copy()
    ts[BOUNDARIES[0], :4, :] += CYCLE
    _repair_boundary_steps(ts, ANCHORS, WAVELENGTH)
    assert np.allclose(ts[-1], truth[-1])


def test_repair_is_a_no_op_on_a_clean_stack():
    rng = np.random.default_rng(32)
    ts = _clean_stack(n=40, ny=20, nx=20, rate_per_epoch=-0.003)
    ts += rng.normal(0, 0.003, ts.shape)
    before = ts.copy()
    rep = _repair_boundary_steps(ts, ANCHORS, WAVELENGTH)
    assert rep["repaired_pixels"] == 0
    assert np.allclose(ts, before)


# ------------------------------------------------------------------ geometry

def test_vertical_projection_matches_one_over_cos_incidence():
    inc = np.radians(38.44)
    los_up = np.full((4, 4), np.cos(inc))
    los = np.full((4, 4), -0.010)
    vert = project_to_vertical(los, los_up)
    assert np.allclose(vert, -0.010 / np.cos(inc))
    # Vertical is always LARGER in magnitude than its line-of-sight value.
    assert np.all(np.abs(vert) > np.abs(los))


def test_vertical_projection_refuses_a_near_horizontal_geometry():
    """Dividing by a tiny los_up would manufacture enormous displacements."""
    out = project_to_vertical(np.array([0.01]), np.array([0.01]))
    assert np.isnan(out).all()


def test_pure_uplift_projects_positive_toward_the_satellite():
    # los vector points ground -> satellite, so uplift is toward it.
    assert los_from_enu(0, 0, 1.0, -0.612, -0.110, 0.783) > 0


def test_plate_motion_dominates_an_absolute_los_rate():
    """
    Why gnss_validation quotes only vertical rates as regional context.

    North America runs ~15 mm/yr west here. Projected into an ascending
    line of sight that swamps a 1-2 mm/yr vertical signal.
    """
    horizontal_only = los_from_enu(-14.9, 1.1, 0.0, -0.612, -0.110, 0.783)
    vertical_only = los_from_enu(0.0, 0.0, -1.2, -0.612, -0.110, 0.783)
    assert abs(horizontal_only) > 5 * abs(vertical_only)
    assert horizontal_only > 0        # westward motion reads as approaching


# ------------------------------------------------- reference-pixel selection

def _pick_reference(flat, finite, t, criterion):
    """The two candidate criteria, side by side on the same data."""
    n = flat.shape[1]
    rate = np.zeros(n)
    scatter = np.zeros(n)
    for j in range(n):
        m = finite[:, j]
        A = np.vstack([t[m], np.ones(m.sum())]).T
        sol, *_ = np.linalg.lstsq(A, flat[m, j], rcond=None)
        rate[j] = sol[0]
        scatter[j] = np.std(flat[m, j] - A @ sol)
    if criterion == "scatter_only":          # the bug
        return int(np.argmin(scatter)), rate
    pool = np.where(scatter <= np.median(scatter))[0]
    return int(pool[np.argmin(np.abs(rate[pool]))]), rate


def test_scatter_only_reference_picks_a_moving_pixel_and_shifts_everything():
    """
    Why ranking reference candidates by residual scatter alone is wrong.

    A pixel subsiding at a steady rate has almost no scatter about its own
    straight line. Picking it as the reference subtracts its trend from
    every other pixel, shifting the entire velocity field uniformly - and
    a uniform shift is invisible to every internal check, including
    correlation against another backend.
    """
    rng = np.random.default_rng(41)
    n_ep, n_px = 40, 60
    t = np.linspace(0, 9.5, n_ep)

    rates = rng.normal(-0.004, 0.002, n_px)   # mostly subsiding
    rates[0] = -0.010                          # one steady, fast, very smooth
    rates[1] = 0.0                             # one genuinely stable
    noise = np.full(n_px, 0.004)
    noise[0] = 0.0002                          # the fast pixel is the quietest
    noise[1] = 0.0004

    flat = (t[:, None] * rates[None, :]
            + rng.normal(0, 1, (n_ep, n_px)) * noise[None, :])
    finite = np.ones_like(flat, dtype=bool)

    bad_j, rate = _pick_reference(flat, finite, t, "scatter_only")
    good_j, _ = _pick_reference(flat, finite, t, "quiet_and_still")

    assert bad_j == 0, "scatter-only should be seduced by the smooth fast pixel"
    assert abs(rate[good_j]) < abs(rate[bad_j]), (
        "the fixed criterion must pick a pixel closer to zero rate")

    # The consequence: the whole field shifts by the reference's own rate.
    shift_bad = -rate[bad_j]
    assert abs(shift_bad) > 0.008, "a ~10 mm/yr shift applied to every pixel"
    # Enough to flip the sign of the field median.
    assert np.median(rate) < 0 < np.median(rate - rate[bad_j]), (
        "the bad reference flips the sign of the median velocity")
    # The fixed one does not.
    assert np.median(rate - rate[good_j]) < 0


def test_fixed_criterion_keeps_the_reference_quiet_too():
    """It must not fix the trend problem by choosing a noisy pixel."""
    rng = np.random.default_rng(42)
    n_ep, n_px = 40, 60
    t = np.linspace(0, 9.5, n_ep)
    rates = rng.normal(-0.004, 0.002, n_px)
    noise = np.full(n_px, 0.002)
    rates[5] = 0.0
    noise[5] = 0.030                           # zero rate, hopelessly noisy
    rates[6] = 0.0002
    noise[6] = 0.001                           # near-zero rate AND quiet
    flat = (t[:, None] * rates[None, :]
            + rng.normal(0, 1, (n_ep, n_px)) * noise[None, :])
    finite = np.ones_like(flat, dtype=bool)
    j, _ = _pick_reference(flat, finite, t, "quiet_and_still")
    assert j != 5, "must not choose the noisy pixel just because its rate is 0"


# ------------------------------------------- tropospheric correction, referenced

def _tropo_case(seed=51, n_ep=18, ny=12, nx=12):
    """
    A stack referenced to a pixel, and a delay cube that is not.

    The delay is mostly a large AOI-wide offset with a small spatial
    structure on top - which is what ERA5 actually looks like over a
    7 km AOI, measured at 176 mm uniform against 2.6 mm of structure.
    """
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 9.5, n_ep)

    uniform = rng.normal(0, 0.060, n_ep)                    # ~60 mm, AOI-wide
    struct = rng.normal(0, 0.003, (n_ep, ny, nx))           # ~3 mm, spatial
    delay = uniform[:, None, None] + struct                 # the true delay

    truth = (-0.010 * t)[:, None, None] * np.ones((1, ny, nx))
    obs = truth - delay + rng.normal(0, 0.004, (n_ep, ny, nx))

    # Referencing the observations removes the uniform part, as it does
    # in the real pipeline.
    r, c = 0, 0
    ts = obs - obs[:, r, c][:, None, None]
    return ts, delay, t, (r, c)


def _scatter(ts, t):
    rms = []
    for j in range(ts.shape[1] * ts.shape[2]):
        y = ts.reshape(ts.shape[0], -1)[:, j]
        A = np.vstack([t, np.ones(t.size)]).T
        sol, *_ = np.linalg.lstsq(A, y, rcond=None)
        rms.append(np.sqrt(np.mean((y - A @ sol) ** 2)))
    return float(np.median(rms))


def test_unreferenced_delay_cube_makes_the_series_much_worse():
    """
    The bug the guard caught on a real run: +300% scatter.

    The observations have had a reference pixel subtracted, so the
    AOI-wide part of the delay is already gone from them. Adding the
    absolute cube puts it back in.
    """
    ts, delay, t, _ = _tropo_case()
    before = _scatter(ts, t)
    after = _scatter(ts + delay, t)
    assert after > 3 * before, (
        f"an unreferenced cube should be badly worse, got "
        f"{before * 1000:.1f} -> {after * 1000:.1f} mm")


def test_referencing_the_cube_makes_the_correction_help():
    ts, delay, t, (r, c) = _tropo_case()
    cube = delay - delay[:, r, c][:, None, None]
    before = _scatter(ts, t)
    after = _scatter(ts + cube, t)
    assert after < before, (
        f"a referenced cube should tighten the series, got "
        f"{before * 1000:.1f} -> {after * 1000:.1f} mm")


def test_referenced_cube_is_far_smaller_than_the_raw_delay():
    """The size difference is the whole point, and it is large."""
    _, delay, _, (r, c) = _tropo_case()
    cube = delay - delay[:, r, c][:, None, None]
    raw_p95 = np.nanpercentile(np.abs(delay), 95)
    ref_p95 = np.nanpercentile(np.abs(cube), 95)
    assert ref_p95 < raw_p95 / 5, (
        f"referencing should strip the uniform part: {raw_p95 * 1000:.1f} mm "
        f"-> {ref_p95 * 1000:.1f} mm")


# --------------------------------------------- full coefficient extraction

def test_fit_recovers_every_coefficient_from_known_truth():
    """
    The refactor that lets one solve produce the whole EGMS field set.

    Build a series with a known rate, annual amplitude and acceleration,
    then check each comes back. Getting the rate right while silently
    mislabelling the seasonal or quadratic column is exactly the kind of
    error that produces a plausible table of wrong numbers.
    """
    t = np.linspace(0, 9.5, 240)
    rate, amp, phase, accel = -0.012, 0.006, 0.7, 0.0008
    y = (rate * t
         + amp * np.sin(2 * np.pi * t + phase)
         + 0.5 * accel * t ** 2
         + 0.020)

    A = _design(t, seasonal=True, acceleration=True)
    idx = _term_index(seasonal=True, acceleration=True)
    sol, sigmas, rmse = _fit(A, y)

    assert np.isclose(sol[idx["rate"]], rate, rtol=1e-6)
    got_amp = np.hypot(sol[idx["sin"]], sol[idx["cos"]])
    assert np.isclose(got_amp, amp, rtol=1e-6)
    assert np.isclose(2.0 * sol[idx["accel"]], accel, rtol=1e-6)
    assert rmse < 1e-9, "a noiseless series should fit exactly"
    assert np.all(sigmas[: A.shape[1]] < 1e-9)


def test_term_index_matches_the_design_matrix_width():
    for seasonal in (False, True):
        for accel in (False, True):
            A = _design(np.linspace(0, 5, 50), seasonal, accel)
            idx = _term_index(seasonal, accel)
            used = [v for v in idx.values() if v is not None]
            assert max(used) == A.shape[1] - 1, (
                f"index/width mismatch for seasonal={seasonal} accel={accel}")
            assert len(set(used)) == len(used), "duplicate column index"


def test_rate_is_always_column_zero_whatever_the_model():
    """Callers read the rate by fixed index; a reordering must not happen."""
    t = np.linspace(0, 9.5, 200)
    y = -0.012 * t + 0.004 * np.sin(2 * np.pi * t)
    for seasonal in (False, True):
        for accel in (False, True):
            A = _design(t, seasonal, accel)
            r, _ = _rate_and_sigma(A, y)
            assert abs(r - (-0.012)) < 2e-3, (
                f"rate off with seasonal={seasonal} accel={accel}: {r}")


def test_rmse_reports_actual_residual_scatter():
    rng = np.random.default_rng(61)
    t = np.linspace(0, 9.5, 300)
    noise = 0.005
    y = -0.012 * t + rng.normal(0, noise, t.size)
    _, _, rmse = _fit(_design(t, seasonal=False), y)
    assert abs(rmse - noise) < 0.1 * noise, f"rmse {rmse:.5f} vs {noise}"


def test_acceleration_inflates_the_rate_error_bar_on_a_short_record():
    """Why disp_acceleration_model is off by default."""
    rng = np.random.default_rng(62)
    t = np.linspace(0, 1.5, 40)          # short record
    y = -0.012 * t + rng.normal(0, 0.004, t.size)
    _, s_lin = _rate_and_sigma(_design(t, False, False), y)
    _, s_acc = _rate_and_sigma(_design(t, False, True), y)
    assert s_acc > 2 * s_lin, (
        f"a quadratic should trade against the rate: {s_lin:.5f} -> {s_acc:.5f}")


# ------------------------------------------- acceptance and precision policy

def test_acceptance_keeps_points_whose_rate_is_not_significant():
    """
    "Monitored, no motion detected" is a finding, not a failure.

    Deleting points because their rate is indistinguishable from zero
    throws away the statement a subsidence client most needs: that this
    ground WAS measured and is NOT moving.
    """
    vel = np.array([[0.0001, -0.010], [0.0002, -0.008]])
    std = np.full_like(vel, 0.004)
    significant = np.abs(vel) > 1.96 * std
    assert significant.sum() == 2, "two pixels should be significant"
    # The gate is validity, so all four survive; significance is a column.
    accepted = np.isfinite(vel)
    assert accepted.all(), "insignificant points must still be delivered"


def test_uncertainty_cannot_see_an_unwrapping_error():
    """
    The reason acceptance is not gated on uncertainty.

    A pixel offset by a whole 2*pi cycle has a smooth series, an
    excellent linear fit, and a SMALL formal sigma - while its rate is
    wrong by tens of mm/yr. Sigma is a variance statistic; this is a bias.
    """
    t = np.linspace(0, 9.5, 240)
    truth_rate = -0.010
    clean = truth_rate * t
    # A step of one cycle partway through, exactly the stitching failure.
    corrupted = clean.copy()
    corrupted[120:] += CYCLE

    A = _design(t, seasonal=False)
    r_clean, s_clean = _rate_and_sigma(A, clean)
    r_bad, s_bad = _rate_and_sigma(A, corrupted)

    # The rate is badly wrong ...
    assert abs(r_bad - truth_rate) > 0.002, "the cycle error should bias the rate"
    # ... and sigma stays small enough that a precision gate would admit it.
    assert s_bad < 0.004, (
        f"sigma {s_bad:.5f} should stay small - that is the whole problem")


def test_least_deforming_decile_would_understate_dispersion():
    """
    Why the automatic stable-area fallback was removed.

    Selecting the pixels whose velocity is nearest zero and then
    measuring their spread measures the selection, not the ground. On a
    real run this returned 0.016 in/yr against a formal sigma of 0.228.
    """
    rng = np.random.default_rng(71)
    vel = rng.normal(-0.004, 0.005, 20000)
    decile = vel[np.abs(vel) <= np.percentile(np.abs(vel), 10)]
    assert np.std(decile) < 0.25 * np.std(vel), (
        "the decile's spread is a property of the cut, not of the data")


# ------------------------------------------------ weather model plumbing

def test_delay_cache_is_keyed_on_model_as_well_as_date():
    """
    Keyed on date alone, switching weather_model on a project that already
    had delays would silently reuse the other model's cubes and label the
    result with the new model's name - an ERA5 correction reported as
    HRRR, with nothing anywhere to reveal it.
    """
    from insar_pipeline import tropo_disp
    import inspect as _inspect
    src = _inspect.getsource(tropo_disp.delay_cube_for_date)
    assert "cfg.tropo_dir / model" in src, (
        "the cache path must include the weather model")


def test_model_range_rejects_dates_before_the_archive_starts():
    """
    A partial correction is worse than none - it puts a step the size of
    the correction into the series - so an out-of-range date must stop the
    whole thing before any download, not fail partway through.
    """
    from insar_pipeline import tropo_disp
    from insar_pipeline.config import PipelineConfig
    from pathlib import Path

    cfg = PipelineConfig(project_dir=Path("/tmp/x"),
                         bbox=(-87.69, 38.31, -87.60, 38.39),
                         start_date="2016-01-01", end_date="2026-01-01")

    cfg.weather_model = "HRRR"
    ok, msg = tropo_disp.check_model_range(cfg, ["20160703", "20200101"])
    assert not ok and "2016-07-15" in msg

    ok, msg = tropo_disp.check_model_range(cfg, ["20160715", "20200101"])
    assert ok, msg

    # ERA5 reaches back to 1950, so the same dates are fine.
    cfg.weather_model = "ERA5"
    ok, _ = tropo_disp.check_model_range(cfg, ["20160703", "20200101"])
    assert ok


def test_conus_only_model_rejects_an_aoi_outside_conus():
    from insar_pipeline import tropo_disp
    from insar_pipeline.config import PipelineConfig
    from pathlib import Path

    cfg = PipelineConfig(project_dir=Path("/tmp/x"),
                         bbox=(85.0, 28.0, 85.5, 28.5),      # Nepal
                         start_date="2020-01-01", end_date="2021-01-01")
    cfg.weather_model = "HRRR"
    ok, msg = tropo_disp.check_model_range(cfg, ["20200101"])
    assert not ok and "continental US" in msg


def test_underdetermined_model_returns_nan_rather_than_a_wrong_number():
    """
    Three epochs cannot support a rate + intercept + annual sine/cosine.

    `_fit` returns NaN, which is correct - but the caller must notice.
    A run that fits an underdetermined model reports its pixel counts
    cheerfully and then writes an entirely empty raster; the failure is
    invisible until someone opens it. `build_stack` now drops optional
    terms the stack cannot support.
    """
    t = np.linspace(0, 0.1, 3)
    y = -0.012 * t
    A = _design(t, seasonal=True)          # 4 parameters, 3 observations
    assert A.shape[0] <= A.shape[1]
    sol, sigmas, rmse = _fit(A, y)
    assert np.isnan(sol).all() and np.isnan(sigmas).all()

    # Dropping the seasonal term makes it estimable again.
    A2 = _design(t, seasonal=False)
    assert A2.shape[0] > A2.shape[1]
    r, _ = _rate_and_sigma(A2, y)
    assert np.isclose(r, -0.012, rtol=1e-6)


if __name__ == "__main__":
    import traceback
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except Exception:
            failed += 1
            print(f"FAIL  {fn.__name__}")
            traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
