"""Phase 17 regression tests; run `python -m pytest -q` from project root."""
from pathlib import Path

import pandas as pd
import pytest

from scripts.audit_release import validate_cost_backtest, validate_next_day
from scripts.build_demo_snapshot import build_snapshot


def _example_cost_frames():
    detail = pd.DataFrame({
        "Fold": [1, 1], "item_id": ["A", "B"], "day_number": [90, 90],
        "units_sold": [10, 10], "ml_order": [12, 7], "seasonal_order": [10, 9],
        "ml_cost": [4, 24], "seasonal_cost": [0, 8]
    })
    summary = pd.DataFrame({
        "Fold": [1], "ML Cost": [28], "Seasonal Cost": [8], "Savings (%)": [-250]
    })
    return summary, detail


def test_cost_audit_reconciles_folds_and_arithmetic():
    summary, detail = _example_cost_frames()
    result = validate_cost_backtest(summary, detail)
    assert result["product_day_forecasts"] == 2
    assert result["ml_cost"] == 28
    assert result["seasonal_cost"] == 8


def test_cost_audit_rejects_incorrect_cost():
    summary, detail = _example_cost_frames()
    detail.loc[0, "ml_cost"] = 5
    with pytest.raises(ValueError, match="cost formula"):
        validate_cost_backtest(summary, detail)


def test_cost_audit_rejects_duplicate_observation():
    summary, detail = _example_cost_frames()
    detail.loc[1, "item_id"] = "A"
    with pytest.raises(ValueError, match="duplicate"):
        validate_cost_backtest(summary, detail)


def _next_day_frame():
    return pd.DataFrame({
        "date": ["2016-05-23"], "day_number": [1942], "item_id": ["FOODS_3_090"],
        "lower_q05": [31.82], "predicted_q80": [64.67], "upper_q95": [84.53],
        "suggested_order_units": [65], "conformal_margin": [3.23],
        "calibrated_lower": [28.59], "calibrated_upper": [87.76]
    })


def test_next_day_audit_accepts_correct_calibration():
    meta = validate_next_day(_next_day_frame(), calibrated=True)
    assert meta["forecast_day"] == 1942


def test_next_day_audit_rejects_wrong_calibrated_bounds():
    df = _next_day_frame()
    df.loc[0, "calibrated_upper"] = 150
    with pytest.raises(ValueError, match="upper"):
        validate_next_day(df, calibrated=True)


def test_public_build_requires_explicit_data_terms_review(tmp_path):
    root = tmp_path / "ForecastGuard"
    root.mkdir()
    with pytest.raises(ValueError, match="Public distribution is blocked"):
        build_snapshot(root, tmp_path / "public", mode="public")
    assert not (tmp_path / "public").exists()


def test_demo_build_excludes_raw_data_and_models(tmp_path, monkeypatch):
    root = tmp_path / "ForecastGuard"
    for rel in (
        "app/dashboard.py", "app/interval_reliability.py",
        "app/calibrated_recommendations.py", "src/reliability_reporting.py",
    ):
        file = root / rel
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text("# test component\n", encoding="utf-8")
    (root / "data" / "raw").mkdir(parents=True)
    (root / "data" / "raw" / "private_sales.csv").write_text("secret", encoding="utf-8")
    (root / "models").mkdir()
    (root / "models" / "test.joblib").write_bytes(b"do not copy")
    from scripts import build_demo_snapshot as builder
    from scripts import audit_release as auditor
    for name in auditor.REQUIRED_FILES:
        file = root / "results" / name
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text("header\nvalue\n", encoding="utf-8")
    monkeypatch.setattr(builder, "audit_release", lambda root: {
        "backtest": {"product_day_forecasts": 2240, "folds": 8, "ml_cost": 87976,
                     "seasonal_cost": 112278, "savings_pct": 21.64},
        "reliability": {"raw_coverage_pct": 87.19, "calibrated_coverage_pct": 91.29}
    })
    monkeypatch.setattr(builder, "_version", lambda package: "1.0.0")
    destination = tmp_path / "local-demo"
    result = build_snapshot(root, destination, mode="local")
    assert result["mode"] == "local"
    assert (destination / "app" / "dashboard.py").is_file()
    assert (destination / "results" / "backtest_summary.csv").is_file()
    assert (destination / "MANIFEST.json").is_file()
    assert (destination / "requirements.txt").is_file()
    assert not (destination / "data").exists()
    assert not (destination / "models").exists()
