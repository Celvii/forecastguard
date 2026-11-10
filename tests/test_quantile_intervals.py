import numpy as np
import pytest

from src.intervals import non_crossing_quantiles


def test_non_crossing_quantiles_preserve_values():
    lo, mid, hi, repaired = non_crossing_quantiles(
        [5, 12], [20, 25], [30, 40]
    )
    np.testing.assert_array_equal(lo, [5, 12])
    np.testing.assert_array_equal(mid, [20, 25])
    np.testing.assert_array_equal(hi, [30, 40])
    assert repaired.tolist() == [False, False]


def test_crossing_is_reported_and_q80_is_preserved():
    lo, mid, hi, repaired = non_crossing_quantiles(
        [35, -2], [20, 10], [30, 5]
    )
    np.testing.assert_array_equal(lo, [20, 0])
    np.testing.assert_array_equal(mid, [20, 10])
    np.testing.assert_array_equal(hi, [35, 10])
    assert repaired.tolist() == [True, True]


def test_invalid_quantiles_rejected():
    with pytest.raises(ValueError, match="matching shapes"):
        non_crossing_quantiles([1, 2], [2], [3, 4])
    with pytest.raises(ValueError, match="finite"):
        non_crossing_quantiles([1, np.nan], [2, 2], [3, 4])
