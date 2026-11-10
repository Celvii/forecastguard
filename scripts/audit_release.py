"""ForecastGuard Phase 17: reproducibility and public-demo input audit.

Run from ForecastGuard root:
    python -m scripts.audit_release

Reads CSV exports only; does not train models or alter data.
"""
from __future__ import annotations

from pathlib import Path
import math

import numpy as np
import pandas as pd


REQUIRED_FILES = (
    "backtest_summary.csv",
    "backtest_predictions.csv",
    "interval_backtest_predictions.csv",
    "next_day_forecast.csv",
    "next_day_calibrated_forecast.csv",
)
OPTIONAL_FILES = (
    "interval_backtest_summary.csv",
    "interval_product_summary.csv",
    "interval_calibration_margins.csv",
)


def _numeric(df: pd.DataFrame, names: list[str], context: str) -> None:
    for name in names:
        if name not in df.columns:
            raise ValueError(f"{context}: required column {name!r} is missing")
        values = pd.to_numeric(df[name], errors="raise").to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"{context}: {name!r} contains missing or nonfinite numbers")


def validate_cost_backtest(folds: pd.DataFrame, predictions: pd.DataFrame) -> dict:
    """Validate cost arithmetic, row uniqueness, and per-fold reconciliation."""
    context = "Cost backtest"
    _numeric(folds, ["Fold", "ML Cost", "Seasonal Cost", "Savings (%)"], context)
    _numeric(predictions, [
        "Fold", "day_number", "units_sold", "ml_order", "seasonal_order", "ml_cost", "seasonal_cost"
    ], context)
    if predictions.empty or folds.empty:
        raise ValueError(f"{context}: empty results")
    if "item_id" not in predictions or predictions["item_id"].isna().any():
        raise ValueError(f"{context}: product IDs are missing")
    if predictions.duplicated(["item_id", "day_number"]).any():
        raise ValueError(f"{context}: duplicate product/day records")
    if folds["Fold"].duplicated().any():
        raise ValueError(f"{context}: duplicate fold IDs")
    if (predictions[["units_sold", "ml_order", "seasonal_order"]] < 0).any().any():
        raise ValueError(f"{context}: negative demand or order")
    y = predictions["units_sold"].to_numpy(dtype=float)
    for order_col, cost_col in (("ml_order", "ml_cost"), ("seasonal_order", "seasonal_cost")):
        q = predictions[order_col].to_numpy(dtype=float)
        cost = 2 * np.maximum(q - y, 0) + 8 * np.maximum(y - q, 0)
        if not np.allclose(cost, predictions[cost_col].to_numpy(dtype=float), atol=0.001):
            raise ValueError(f"{context}: {cost_col} does not match the 2/8 cost formula")

    grouped = predictions.groupby("Fold", as_index=True)[["ml_cost", "seasonal_cost"]].sum()
    indexed = folds.set_index("Fold")
    if set(grouped.index) != set(indexed.index):
        raise ValueError(f"{context}: fold IDs differ between summary and predictions")
    if not np.allclose(grouped.loc[indexed.index, "ml_cost"], indexed["ML Cost"], atol=0.01):
        raise ValueError(f"{context}: fold ML totals do not reconcile")
    if not np.allclose(grouped.loc[indexed.index, "seasonal_cost"], indexed["Seasonal Cost"], atol=0.01):
        raise ValueError(f"{context}: fold seasonal totals do not reconcile")
    expected_saving = 100 * (indexed["Seasonal Cost"] - indexed["ML Cost"]) / indexed["Seasonal Cost"]
    if (indexed["Seasonal Cost"] <= 0).any() or not np.allclose(expected_saving, indexed["Savings (%)"], atol=0.01):
        raise ValueError(f"{context}: fold savings do not reconcile")
    ml_cost = float(predictions["ml_cost"].sum())
    base_cost = float(predictions["seasonal_cost"].sum())
    return {
        "product_day_forecasts": int(len(predictions)),
        "products": int(predictions["item_id"].nunique()),
        "folds": int(len(folds)),
        "ml_cost": ml_cost,
        "seasonal_cost": base_cost,
        "savings_pct": 100 * (base_cost - ml_cost) / base_cost,
    }


def validate_interval_history(frame: pd.DataFrame) -> dict:
    """Recompute reliability metrics from the observation rows."""
    from src.reliability_reporting import prepare_interval_predictions, interval_summary

    validated = prepare_interval_predictions(frame)
    return interval_summary(validated)


def validate_next_day(frame: pd.DataFrame, calibrated: bool = False) -> dict:
    """Check saved next-day forecast schema, ordering and no duplicate products."""
    context = "Calibrated next day" if calibrated else "Nominal next day"
    columns = ["day_number", "lower_q05", "predicted_q80", "upper_q95", "suggested_order_units"]
    if calibrated:
        columns += ["conformal_margin", "calibrated_lower", "calibrated_upper"]
    _numeric(frame, columns, context)
    if frame.empty or "item_id" not in frame.columns or "date" not in frame.columns:
        raise ValueError(f"{context}: empty or missing required fields")
    if frame["item_id"].isna().any() or frame["item_id"].duplicated().any():
        raise ValueError(f"{context}: duplicate or missing product")
    if frame["day_number"].nunique() != 1 or frame["date"].nunique() != 1:
        raise ValueError(f"{context}: inconsistent forecast dates/day numbers")
    if pd.to_datetime(frame["date"], errors="coerce").isna().any():
        raise ValueError(f"{context}: invalid dates")
    lo = frame["lower_q05"].to_numpy(dtype=float)
    q80 = frame["predicted_q80"].to_numpy(dtype=float)
    hi = frame["upper_q95"].to_numpy(dtype=float)
    order = frame["suggested_order_units"].to_numpy(dtype=float)
    if (lo < 0).any() or (lo > q80 + 1e-6).any() or (q80 > hi + 1e-6).any():
        raise ValueError(f"{context}: nominal bounds do not contain Q80")
    if not np.allclose(order, np.ceil(q80), atol=1e-6):
        raise ValueError(f"{context}: suggested order not equal to ceil(Q80)")
    if calibrated:
        margin = frame["conformal_margin"].to_numpy(dtype=float)
        calibrated_lo = frame["calibrated_lower"].to_numpy(dtype=float)
        calibrated_hi = frame["calibrated_upper"].to_numpy(dtype=float)
        if (margin < 0).any():
            raise ValueError(f"{context}: negative conformal margin")
        if not np.allclose(calibrated_lo, np.maximum(0, lo - margin), atol=0.02):
            raise ValueError(f"{context}: calibrated lower not equal to max(0, Q05-margin)")
        if not np.allclose(calibrated_hi, hi + margin, atol=0.02):
            raise ValueError(f"{context}: calibrated upper not equal to Q95+margin")
    return {
        "products": int(len(frame)),
        "forecast_date": str(pd.to_datetime(frame["date"].iloc[0]).date()),
        "forecast_day": int(frame["day_number"].iloc[0]),
    }


def audit_release(root: Path) -> dict:
    """Audit all required result CSVs without touching training files."""
    root = Path(root)
    folder = root / "results"
    for name in REQUIRED_FILES:
        if not (folder / name).is_file():
            raise FileNotFoundError(f"Required result CSV missing: {folder / name}")
    frames = {name: pd.read_csv(folder / name) for name in REQUIRED_FILES}
    summary = validate_cost_backtest(frames["backtest_summary.csv"], frames["backtest_predictions.csv"])
    reliability = validate_interval_history(frames["interval_backtest_predictions.csv"])
    nominal = validate_next_day(frames["next_day_forecast.csv"])
    calibrated = validate_next_day(frames["next_day_calibrated_forecast.csv"], calibrated=True)
    if nominal["forecast_day"] != calibrated["forecast_day"]:
        raise ValueError("Nominal and calibrated forecast target days differ")
    return {"backtest": summary, "reliability": reliability, "nominal_next_day": nominal,
            "calibrated_next_day": calibrated,
            "optional_missing": [name for name in OPTIONAL_FILES if not (folder / name).is_file()]}


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    try:
        report = audit_release(root)
    except (ValueError, KeyError, FileNotFoundError, pd.errors.ParserError) as exc:
        print("RELEASE AUDIT FAILED:", exc)
        raise SystemExit(1) from exc

    b = report["backtest"]
    r = report["reliability"]
    print("RELEASE AUDIT PASSED")
    print(f"Cost backtest: {b['product_day_forecasts']} forecasts; {b['folds']} folds; {b['products']} products")
    print(f"Cost totals: ML ${b['ml_cost']:,.0f}, baseline ${b['seasonal_cost']:,.0f}; savings {b['savings_pct']:.2f}%")
    print(f"Historical 90% interval coverage: raw {r['raw_coverage_pct']:.2f}% / calibrated {r['calibrated_coverage_pct']:.2f}%")
    print(f"Interval forecasts: {r['n_forecasts']:,}")
    print(f"Nominal next day: {report['nominal_next_day']['forecast_date']}")
    print(f"Calibrated next day: {report['calibrated_next_day']['forecast_date']}")
    if report["optional_missing"]:
        print("Optional auxiliary CSVs missing:", ", ".join(report["optional_missing"]))
    print("Manual follow-up: verify M5 redistribution terms, app copy, security, and deployment dependencies.")


if __name__ == "__main__":
    main()
