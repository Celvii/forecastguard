
import numpy as np
import pandas as pd

from .data_pipeline import FEATURES
from .model import build_quantile_model
from .inventory import inventory_cost, optimal_quantile


DEFAULT_CUTOFFS = [
    1493, 1549, 1605, 1661,
    1717, 1773, 1829, 1885
]


def run_backtest(
    df,
    overstock_cost=2,
    understock_cost=8,
    cutoffs=DEFAULT_CUTOFFS
):
    """Run expanding-window, one-day-ahead backtesting."""

    q = optimal_quantile(overstock_cost, understock_cost)
    weekly_cols = [f"weekly_lag_{i}" for i in range(1, 9)]

    fold_results = []
    predictions = []

    for fold, cutoff in enumerate(cutoffs, start=1):

        # Training uses only observations before this fold
        train = df[
            (df["day_number"] <= cutoff) &
            (df["price_available"])
        ].copy()

        test = df[
            df["day_number"].between(
                cutoff + 1, cutoff + 56
            )
        ].copy()

        # Create a new model for this fold
        model = build_quantile_model(q)

        model.fit(
            train[FEATURES],
            train["units_sold"]
        )

        # Forecast product demand
        predicted_quantile = np.maximum(
            0,
            model.predict(test[FEATURES])
        )

        actual = test["units_sold"].to_numpy()

        ml_order = np.ceil(predicted_quantile)

        # Seasonal baseline uses the matching quantile
        seasonal_q = test[weekly_cols].quantile(
            q, axis=1
        ).to_numpy()

        seasonal_order = np.ceil(
            np.maximum(0, seasonal_q)
        )

        # Calculate simulated costs
        ml_cost = inventory_cost(
            actual, ml_order,
            overstock_cost, understock_cost
        )

        seasonal_cost = inventory_cost(
            actual, seasonal_order,
            overstock_cost, understock_cost
        )

        ml_total = ml_cost.sum()
        seasonal_total = seasonal_cost.sum()

        savings = (
            (seasonal_total - ml_total)
            / seasonal_total * 100
        )

        fold_results.append({
            "Fold": fold,
            "Start Day": cutoff + 1,
            "End Day": cutoff + 56,
            "ML Cost": ml_total,
            "Seasonal Cost": seasonal_total,
            "Savings (%)": savings,
            "Quantile Coverage (%)": (
                np.mean(actual <= predicted_quantile) * 100
            )
        })

        # Store each prediction for later analysis
        detail = test[
            ["item_id", "date", "day_number", "units_sold"]
        ].copy()

        detail["Fold"] = fold
        detail["predicted_quantile"] = predicted_quantile
        detail["ml_order"] = ml_order
        detail["seasonal_order"] = seasonal_order
        detail["ml_cost"] = ml_cost
        detail["seasonal_cost"] = seasonal_cost

        predictions.append(detail)

    return (
        pd.DataFrame(fold_results),
        pd.concat(predictions, ignore_index=True)
    )
