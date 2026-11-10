
from pathlib import Path
import pandas as pd


NUMERIC_FEATURES = [
    "lag_1", "lag_7", "lag_14", "lag_28",
    "rolling_mean_7", "rolling_mean_28"
]

CATEGORICAL_FEATURES = [
    "item_id", "day_of_week", "month",
    "event_name_1", "event_name_2"
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def prepare_product_data(project_root):
    """
    Load selected Walmart products and create
    leakage-safe, one-day-ahead forecasting features.
    """
    root = Path(project_root)

    # Load historical sales
    df = pd.read_csv(
        root / "data/selected_product_sales.csv",
        parse_dates=["date"]
    )

    df = df.sort_values(
        ["item_id", "day_number"]
    ).reset_index(drop=True)

    # Load calendar information
    calendar = pd.read_csv(
        root / "data/raw/calendar.csv",
        usecols=[
            "d", "wm_yr_wk",
            "event_name_1", "event_name_2"
        ]
    )

    df = df.merge(
        calendar,
        on="d",
        how="left",
        validate="many_to_one"
    )

    # Product-specific lag features
    for lag in [1, 7, 14, 28]:
        df[f"lag_{lag}"] = (
            df.groupby("item_id")["units_sold"]
            .shift(lag)
        )

    # Historical rolling features
    for window in [7, 28]:
        df[f"rolling_mean_{window}"] = (
            df.groupby("item_id")["units_sold"]
            .transform(
                lambda s: s.shift(1)
                .rolling(window).mean()
            )
        )

    # Same-weekday historical observations
    weekly_cols = []

    for week in range(1, 9):
        col = f"weekly_lag_{week}"

        df[col] = (
            df.groupby("item_id")["units_sold"]
            .shift(7 * week)
        )

        weekly_cols.append(col)

    # Baseline quantile
    df["seasonal_q80"] = (
        df[weekly_cols].quantile(0.80, axis=1)
    )

    # Calendar features
    df["day_of_week"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month

    for col in ["event_name_1", "event_name_2"]:
        df[col] = df[col].fillna("No_Event")

    # Selling-price records
    prices = pd.read_csv(
        root / "data/raw/sell_prices.csv"
    )

    prices = prices[
        (prices["store_id"] == "CA_1") &
        (prices["item_id"].isin(df["item_id"].unique()))
    ]

    df = df.merge(
        prices[["item_id", "wm_yr_wk", "sell_price"]],
        on=["item_id", "wm_yr_wk"],
        how="left",
        validate="many_to_one"
    )

    df["price_available"] = df["sell_price"].notna()

    # Remove rows without required historical features
    df = df.dropna(
        subset=NUMERIC_FEATURES + ["seasonal_q80"]
    ).copy()

    assert not df.duplicated(
        ["item_id", "day_number"]
    ).any(), "Duplicate product-day observations detected"

    return df
