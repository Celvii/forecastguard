import numpy as np
import pytest

from src.interval_backtesting import conformal_margin, interval_metrics, repair_bounds


def test_repair_bounds_handles_quantile_crossing_and_negative_sales():
    lower, upper = repair_bounds([-1, 10], [5, 7])
    np.testing.assert_allclose(lower, [0, 7])
    np.testing.assert_allclose(upper, [5, 10])


def test_conformal_margin_finite_sample_rank():
    actual = np.arange(56, dtype=float)
    lower = actual + np.arange(56, dtype=float)
    upper = lower + 1
    # ceil((56 + 1) * .90) = 52, so the 52nd sorted score is 51.
    assert conformal_margin(actual, lower, upper, alpha=0.10) == 51


def test_conformal_margin_rejects_insufficient_calibration():
    with pytest.raises(ValueError):
        conformal_margin(np.zeros(5), np.zeros(5), np.ones(5), alpha=0.10)


def test_interval_metrics_penalizes_missed_observations():
    m = interval_metrics([2, 8, 10], [1, 4, 7], [3, 7, 9], alpha=0.10)
    assert m["coverage_pct"] == pytest.approx(100 / 3)
    assert m["avg_width"] == pytest.approx(7 / 3)
    assert m["interval_score"] == pytest.approx(47 / 3)
