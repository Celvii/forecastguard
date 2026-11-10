
import numpy as np


def optimal_quantile(overstock_cost=2, understock_cost=8):
    """Return the newsvendor target demand quantile."""
    if overstock_cost <= 0 or understock_cost <= 0:
        raise ValueError("Costs must be positive.")

    return understock_cost / (overstock_cost + understock_cost)


def inventory_cost(actual, order, overstock_cost=2, understock_cost=8):
    """Return simulated inventory cost for each observation."""
    actual = np.asarray(actual, dtype=float)
    order = np.asarray(order, dtype=float)

    if actual.shape != order.shape:
        raise ValueError("Actual and order arrays must have matching shapes.")

    surplus = np.maximum(order - actual, 0)
    shortage = np.maximum(actual - order, 0)

    return overstock_cost * surplus + understock_cost * shortage
