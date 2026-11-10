
import pandas as pd
import pytest

from src.inference import build_next_day_features


def test_next_day_uses_only_past_sales(tmp_path):
    data_dir = tmp_path / "data"
    raw_dir = data_dir / "raw"

    raw_dir.mkdir(parents=True)

    # Create 40 historical sales days for two products
    records = []

    for product_id, offset in [("A", 0), ("B", 1000)]:
        for day in range(1, 41):
            records.append({
                "item_id": product_id,
                "day_number": day,
                "units_sold": offset + day
            })

    pd.DataFrame(records).to_csv(
        data_dir / "selected_product_sales.csv",
        index=False
    )

    # Day 41 is in the calendar but has no sales record
    calendar = pd.DataFrame({
        "d": ["d_41"],
        "date": ["2020-02-10"],
        "event_name_1": [None],
        "event_name_2": [None]
    })

    calendar.to_csv(
        raw_dir / "calendar.csv",
        index=False
    )

    artifact = {
        "last_training_day": 40,
        "product_ids": ["A", "B"]
    }

    features = build_next_day_features(
        tmp_path, artifact
    )

    a = features.set_index("item_id").loc["A"]
    b = features.set_index("item_id").loc["B"]

    assert a["day_number"] == 41
    assert a["lag_1"] == 40
    assert a["lag_7"] == 34
    assert a["lag_14"] == 27
    assert a["lag_28"] == 13

    assert a["rolling_mean_7"] == pytest.approx(37)
    assert a["rolling_mean_28"] == pytest.approx(26.5)

    # Product histories remain independent
    assert b["lag_1"] == 1040

    # Day 41 sales are not part of the input
    assert "units_sold" not in features.columns
    assert len(features) == 2
