"""Results for an area and a date range: the fitting and statistics helpers."""
import numpy as np
import pytest

from insar_pipeline import results


def test_fit_rates_recovers_slopes_and_ignores_gaps():
    t = np.linspace(0, 3, 40)
    Y = np.stack([2.0 * t + 1, -5.0 * t, 0 * t], axis=1)
    Y[5, 0] = np.nan                                    # a missing date does not bias the fit
    r = results._fit_rates(t, Y)
    assert r == pytest.approx([2.0, -5.0, 0.0], abs=1e-9)
    few = results._fit_rates(t[:3], Y[:3])              # too few dates -> no rate
    assert np.isnan(few).all()


def test_stats_are_across_points():
    s = results._stats(np.array([1.0, 2.0, 3.0, np.nan]))
    assert s["n"] == 3 and s["mean"] == 2.0 and s["min"] == 1.0 and s["max"] == 3.0
    assert s["range"] == 2.0 and s["std"] == pytest.approx(1.0) and s["median"] == 2.0
    assert results._stats(np.array([np.nan])) == {"n": 0}


def test_dates_parse_both_ways():
    assert str(results._parse_date("2021/03/05")) == "2021-03-05"
    assert str(results._parse_date("2021-03-05T00:00")) == "2021-03-05"
