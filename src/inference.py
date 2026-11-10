
from pathlib import Path

import numpy as np
import pandas as pd

from .data_pipeline import FEATURES


def build_next_day_features(project_root, artifact):
    """Build forecasts inputs without using future sales."""

    root = Path(project_root)
    last_day = int(artifact["last_training_day"])
    next_day = last_day + 1
    product_ids = artifact["product_ids"]

    # Load the historical product sales
    history = pd.read_csv(
        root / "data/selected_product_sales.csv"
    )

    # Avoid using a model trained on stale history
    if history["day_number"].max() != last_day:
        raise ValueError(
            "Saved model and sales history are out of sync. "
            "Retrain the model before forecasting."
        )

    # Read the known calendar for the next day
    calendar = pd.read_csv(
        root / "data/raw/calendar.csv"
    )

    future_calendar = calendar[
        calendar["d"] == f"d_{next_day}"
    ]

    if len(future_calendar) != 1:
        raise ValueError(
            f"Calendar entry d_{next_day} is missing or duplicated."
        )

    cal = future_calendar.iloc[0]
    forecast_date = pd.to_datetime(cal["date"])

    records = []

    for product_id in product_ids:

        product = history[
            history["item_id"] == product_id
        ].copy()

        if product["day_number"].duplicated().any():
            raise ValueError(
                f"Duplicate sales days for {product_id}"
            )

        sales = product.set_index(
            "day_number"
        )["units_sold"]

        # All 28 previous calendar days must exist
        required_days = np.arange(
            next_day - 28, next_day
        )

        recent_sales = sales.reindex(required_days)

        if recent_sales.isna().any():
            raise ValueError(
                f"Incomplete historical sales for {product_id}"
            )

        # Create the same features used during training
        row = {
            "item_id": product_id,
            "date": forecast_date,
            "day_number": next_day,
            "day_of_week": forecast_date.dayofweek,
            "month": forecast_date.month,
            "lag_1": sales.loc[next_day - 1],
            "lag_7": sales.loc[next_day - 7],
            "lag_14": sales.loc[next_day - 14],
            "lag_28": sales.loc[next_day - 28],
            "rolling_mean_7": recent_sales.iloc[-7:].mean(),
            "rolling_mean_28": recent_sales.mean(),
        }

        for col in ["event_name_1", "event_name_2"]:
            value = cal[col]
            row[col] = (
                "No_Event"
                if pd.isna(value) else value
            )

        records.append(row)

    forecast_features = pd.DataFrame(records)

    # Ensure all trained features exist
    missing = set(FEATURES) - set(forecast_features.columns)

    if missing:
        raise ValueError(f"Missing model features: {missing}")

    return forecast_features
