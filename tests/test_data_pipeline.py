
import pandas as pd

from src.data_pipeline import prepare_product_data


def make_test_data(root):
    raw = root / "data" / "raw"
    raw.mkdir(parents=True)

    dates = pd.date_range("2020-01-01", periods=70)

    # Two distinct products with predictable sales
    records = []

    for item, base in [("A", 0), ("B", 1000)]:
        for day in range(1, 71):
            records.append({
                "item_id": item,
                "d": f"d_{day}",
                "units_sold": base + day,
                "date": dates[day - 1],
                "day_number": day
            })

    pd.DataFrame(records).to_csv(
        root / "data" / "selected_product_sales.csv",
        index=False
    )

    # Artificial calendar
    calendar = pd.DataFrame({
        "d": [f"d_{i}" for i in range(1, 71)],
        "wm_yr_wk": [(i - 1) // 7 + 1 for i in range(1, 71)],
        "event_name_1": [None] * 70,
        "event_name_2": [None] * 70
    })

    calendar.to_csv(raw / "calendar.csv", index=False)

    # Artificial product prices
    prices = pd.DataFrame([
        {
            "store_id": "CA_1",
            "item_id": item,
            "wm_yr_wk": week,
            "sell_price": 5.0
        }
        for item in ["A", "B"]
        for week in range(1, 11)
    ])

    prices.to_csv(raw / "sell_prices.csv", index=False)


def test_product_features(tmp_path):
    make_test_data(tmp_path)
    df = prepare_product_data(tmp_path)

    a = df[
        (df["item_id"] == "A") &
        (df["day_number"] == 29)
    ].iloc[0]

    b = df[
        (df["item_id"] == "B") &
        (df["day_number"] == 29)
    ].iloc[0]

    # Product A's previous day is 28
    assert a["lag_1"] == 28
    assert a["lag_7"] == 22
    assert a["rolling_mean_7"] == 25

    # Product B must use its own history
    assert b["lag_1"] == 1028

    # 70 days minus first 28 days per product
    assert len(df) == 84


def test_no_current_target_leakage(tmp_path):
    make_test_data(tmp_path)

    original = prepare_product_data(tmp_path)

    path = tmp_path / "data" / "selected_product_sales.csv"
    sales = pd.read_csv(path)

    # Change actual sales on day 40 for product A
    mask = (
        (sales["item_id"] == "A") &
        (sales["day_number"] == 40)
    )

    sales.loc[mask, "units_sold"] = 9999
    sales.to_csv(path, index=False)

    modified = prepare_product_data(tmp_path)

    def get_row(frame, day):
        return frame[
            (frame["item_id"] == "A") &
            (frame["day_number"] == day)
        ].iloc[0]

    # Day 40 features must not depend on
    # day 40's actual sales
    before = get_row(original, 40)
    after = get_row(modified, 40)

    assert before["lag_1"] == after["lag_1"]
    assert before["rolling_mean_7"] == after["rolling_mean_7"]

    # Day 41 may use day 40's observed sales
    assert get_row(modified, 41)["lag_1"] == 9999
