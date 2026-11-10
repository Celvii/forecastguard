"""Validated, aggregation-safe metrics for historical interval backtests.

This module is deliberately independent of Streamlit so results can be tested
without launching a dashboard.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = {
    "fold", "item_id", "day_number", "date", "units_sold",
    "raw_lower", "raw_upper", "calibrated_lower", "calibrated_upper",
}


def prepare_interval_predictions(frame: pd.DataFrame, alpha: float = 0.10) -> pd.DataFrame:
    """Validate Phase 14 CSV and independently recalculate coverage and scores."""
    if not 0 < alpha < 1:
        raise ValueError("alpha must be strictly between 0 and 1")
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Missing interval result columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("Interval prediction data is empty")

    data = frame.copy()
    numeric_cols = [
        "fold", "day_number", "units_sold", "raw_lower", "raw_upper",
        "calibrated_lower", "calibrated_upper",
    ]
    for col in numeric_cols:
        data[col] = pd.to_numeric(data[col], errors="raise")
        if not np.isfinite(data[col].to_numpy(dtype=float)).all():
            raise ValueError(f"Non-finite values in {col}")
    if data["item_id"].isna().any():
        raise ValueError("Missing product IDs")
    if (data["fold"] <= 0).any() or (data["day_number"] <= 0).any():
        raise ValueError("Fold and day_number must be positive")
    if data.duplicated(["fold", "item_id", "day_number"]).any():
        raise ValueError("Duplicate fold-product-day rows")
    data["date"] = pd.to_datetime(data["date"], errors="raise")
    if data["date"].isna().any():
        raise ValueError("Missing dates")
    if (data["units_sold"] < 0).any():
        raise ValueError("Observed sales must be nonnegative")
    for lo, hi in (("raw_lower", "raw_upper"), ("calibrated_lower", "calibrated_upper")):
        if (data[lo] > data[hi]).any():
            raise ValueError(f"Invalid interval: {lo} exceeds {hi}")

    y = data["units_sold"].to_numpy(dtype=float)
    for label in ("raw", "calibrated"):
        lo = data[f"{label}_lower"].to_numpy(dtype=float)
        hi = data[f"{label}_upper"].to_numpy(dtype=float)
        data[f"{label}_covered"] = (y >= lo) & (y <= hi)
        data[f"{label}_width"] = hi - lo
        data[f"{label}_interval_score"] = (
            (hi - lo)
            + (2.0 / alpha) * np.maximum(lo - y, 0)
            + (2.0 / alpha) * np.maximum(y - hi, 0)
        )
    return data.sort_values(["fold", "item_id", "day_number"]).reset_index(drop=True)


def interval_summary(data: pd.DataFrame) -> dict:
    """Compute aggregate metrics directly from observation rows (weighted by count)."""
    if data.empty:
        raise ValueError("Cannot summarize an empty interval evaluation")
    result = {"n_forecasts": int(len(data))}
    for label in ("raw", "calibrated"):
        result[f"{label}_coverage_pct"] = float(data[f"{label}_covered"].mean() * 100)
        result[f"{label}_avg_width"] = float(data[f"{label}_width"].mean())
        result[f"{label}_interval_score"] = float(data[f"{label}_interval_score"].mean())
    result["coverage_delta_pp"] = (
        result["calibrated_coverage_pct"] - result["raw_coverage_pct"]
    )
    result["width_delta"] = (
        result["calibrated_avg_width"] - result["raw_avg_width"]
    )
    result["score_delta"] = (
        result["calibrated_interval_score"] - result["raw_interval_score"]
    )
    return result


def summarize_groups(data: pd.DataFrame, group_by: str) -> pd.DataFrame:
    """Return independently calculated per-product or per-fold results."""
    if group_by not in ("item_id", "fold"):
        raise ValueError("group_by must be item_id or fold")
    rows = []
    for group_value, subset in data.groupby(group_by, sort=True):
        rows.append({group_by: group_value, **interval_summary(subset)})
    return pd.DataFrame(rows)
