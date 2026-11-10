"""Phase 15 tests; no Streamlit dependency needed."""
import pandas as pd
import pytest

from src.reliability_reporting import (
    prepare_interval_predictions,
    interval_summary,
    summarize_groups,
)


def sample_frame():
    return pd.DataFrame({
        "fold": [1, 1, 2, 2],
        "item_id": ["A", "B", "A", "B"],
        "day_number": [10, 10, 11, 11],
        "date": ["2016-01-01", "2016-01-01", "2016-01-02", "2016-01-02"],
        "units_sold": [10, 20, 30, 40],
        "raw_lower": [9, 22, 27, 35],
        "raw_upper": [11, 24, 33, 45],
        "calibrated_lower": [8, 19, 26, 34],
        "calibrated_upper": [12, 25, 34, 46],
    })


def test_recomputed_coverage_and_scores():
    clean = prepare_interval_predictions(sample_frame())
    metrics = interval_summary(clean)
    assert metrics["n_forecasts"] == 4
    assert metrics["raw_coverage_pct"] == pytest.approx(75)
    assert metrics["calibrated_coverage_pct"] == pytest.approx(100)
    assert metrics["raw_avg_width"] == pytest.approx(5)
    assert metrics["calibrated_avg_width"] == pytest.approx(7.5)
    assert metrics["raw_interval_score"] == pytest.approx(15)
    assert metrics["calibrated_interval_score"] == pytest.approx(7.5)


def test_per_product_and_fold():
    clean = prepare_interval_predictions(sample_frame())
    products = summarize_groups(clean, "item_id")
    folds = summarize_groups(clean, "fold")
    assert products["n_forecasts"].sum() == 4
    assert folds["n_forecasts"].sum() == 4
    assert len(products) == len(folds) == 2


def test_rejects_inverted_intervals():
    df = sample_frame()
    df.loc[0, "raw_lower"] = 100
    with pytest.raises(ValueError, match="Invalid interval"):
        prepare_interval_predictions(df)


def test_rejects_duplicate_forecasts():
    df = pd.concat([sample_frame(), sample_frame().iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="Duplicate"):
        prepare_interval_predictions(df)


def test_rejects_missing_columns():
    with pytest.raises(ValueError, match="Missing interval"):
        prepare_interval_predictions(sample_frame().drop(columns="raw_upper"))
