
import numpy as np
import pytest

from src.inventory import inventory_cost, optimal_quantile


def test_inventory_cost():
    actual = np.array([10, 10])
    order = np.array([12, 7])

    costs = inventory_cost(
        actual,
        order,
        overstock_cost=2,
        understock_cost=8
    )

    # First order: 2 extra units * $2 = $4
    # Second order: 3 missing units * $8 = $24
    np.testing.assert_array_equal(costs, [4, 24])


def test_optimal_quantile():
    assert optimal_quantile(2, 8) == pytest.approx(0.80)
    assert optimal_quantile(2, 2) == pytest.approx(0.50)
    assert optimal_quantile(2, 18) == pytest.approx(0.90)

    # Negative costs should be rejected
    with pytest.raises(ValueError):
        optimal_quantile(-2, 8)
