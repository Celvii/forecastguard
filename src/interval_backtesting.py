"""Product-level, leakage-aware evaluation of nominal and CQR-calibrated intervals.

Inputs come from src.data_pipeline.prepare_product_data().
This module evaluates one-day-ahead forecasts, updated with observed sales daily.
It does not claim exchangeability guarantees for nonstationary time series.
"""

from __future__ import annotations

from math import ceil

import numpy as np
import pandas as pd

from .data_pipeline import FEATURES
from .model import build_quantile_model


DEFAULT_CUTOFFS = (1493, 1549, 1605, 1661, 1717, 1773, 1829, 1885)


def _as_vector(values, name):
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a non-empty, finite, one-dimensional array")
    return array


def repair_bounds(lower, upper):
    """Ensure non-negative, ordered lower/upper quantile predictions."""
    a, b = _as_vector(lower, "lower"), _as_vector(upper, "upper")
    if a.shape != b.shape:
        raise ValueError("lower and upper must have the same shape")
    return np.maximum(0, np.minimum(a, b)), np.maximum(0, np.maximum(a, b))


def conformal_margin(y, lower, upper, alpha=0.10):
    """Conformalized quantile regression margin from a held-out calibration set.

    Nonconformity scores: max(lower - actual, actual - upper, 0).
    With n observations, choose rank ceil((n+1)*(1-alpha)), one-indexed.
    """
    if not 0 < alpha < 1:
        raise ValueError("alpha must be strictly between 0 and 1")
    actual = _as_vector(y, "y")
    lo, hi = repair_bounds(lower, upper)
    if not (actual.shape == lo.shape == hi.shape):
        raise ValueError("y, lower and upper must have the same shape")
    n = len(actual)
    rank = ceil((n + 1) * (1 - alpha))
    if rank > n:
        raise ValueError(f"Need more calibration observations for alpha={alpha}; got n={n}")
    scores = np.maximum.reduce((lo - actual, actual - hi, np.zeros(n)))
    return float(np.sort(scores)[rank - 1])


def interval_metrics(y, lower, upper, alpha=0.10):
    """Coverage (%), average width (units), and mean Winkler interval score."""
    if not 0 < alpha < 1:
        raise ValueError("alpha must be strictly between 0 and 1")
    actual = _as_vector(y, "y")
    lo = _as_vector(lower, "lower")
    hi = _as_vector(upper, "upper")
    if not (actual.shape == lo.shape == hi.shape):
        raise ValueError("y, lower and upper must have the same shape")
    if np.any(lo > hi):
        raise ValueError("lower bounds exceed upper bounds")
    covered = (actual >= lo) & (actual <= hi)
    widths = hi - lo
    scores = widths + (2 / alpha) * np.maximum(lo - actual, 0) + (2 / alpha) * np.maximum(actual - hi, 0)
    return {
        "coverage_pct": float(100 * covered.mean()),
        "avg_width": float(widths.mean()),
        "interval_score": float(scores.mean()),
    }


def run_interval_backtest(df, *, cutoffs=DEFAULT_CUTOFFS,
                          calibration_days=56, horizon_days=56, alpha=0.10,
                          price_filter=True):
    """Return (fold_results, observation_results, product_results, margins).

    For each cutoff: train <= cutoff-calibration_days; calibrate the next
    calibration_days; evaluate days cutoff+1 through cutoff+horizon_days.
    Models are refitted per fold and the CQR margin is calculated per product.
    """
    if calibration_days < 1 or horizon_days < 1:
        raise ValueError("calibration_days and horizon_days must be positive")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be strictly between 0 and 1")
    required = set(FEATURES) | {"item_id", "day_number", "date", "units_sold"}
    if price_filter:
        required.add("price_available")
    if required - set(df.columns):
        raise ValueError(f"Dataset missing required columns: {sorted(required - set(df.columns))}")
    if df.duplicated(["item_id", "day_number"]).any():
        raise ValueError("Duplicate item_id/day_number observations")

    products = sorted(df["item_id"].unique().tolist())
    fold_records, details, margin_records = [], [], []

    for fold_idx, cutoff in enumerate(cutoffs, start=1):
        train_end = cutoff - calibration_days
        train = df.loc[df["day_number"] <= train_end].copy()
        if price_filter:
            train = train.loc[train["price_available"]].copy()
        calibration = df.loc[df["day_number"].between(train_end + 1, cutoff)].copy()
        evaluation = df.loc[df["day_number"].between(cutoff + 1, cutoff + horizon_days)].copy()

        if train.empty:
            raise ValueError(f"Fold {fold_idx}: no training observations")
        for name, segment, expected in (("calibration", calibration, calibration_days),
                                        ("evaluation", evaluation, horizon_days)):
            counts = segment.groupby("item_id")["day_number"].nunique()
            if set(counts.index) != set(products) or not (counts == expected).all():
                raise ValueError(f"Fold {fold_idx}: incomplete {name} history per product")

        lower_model = build_quantile_model(alpha / 2)
        upper_model = build_quantile_model(1 - alpha / 2)
        for model in (lower_model, upper_model):
            model.fit(train[FEATURES], train["units_sold"])

        cal_lower, cal_upper = repair_bounds(
            lower_model.predict(calibration[FEATURES]),
            upper_model.predict(calibration[FEATURES]),
        )
        test_lower, test_upper = repair_bounds(
            lower_model.predict(evaluation[FEATURES]),
            upper_model.predict(evaluation[FEATURES]),
        )
        calibration = calibration.assign(raw_lower=cal_lower, raw_upper=cal_upper)
        evaluation = evaluation.assign(raw_lower=test_lower, raw_upper=test_upper)

        for product_id in products:
            cal = calibration.loc[calibration["item_id"] == product_id].sort_values("day_number")
            test = evaluation.loc[evaluation["item_id"] == product_id].sort_values("day_number").copy()
            margin = conformal_margin(
                cal["units_sold"].to_numpy(),
                cal["raw_lower"].to_numpy(),
                cal["raw_upper"].to_numpy(), alpha=alpha,
            )
            test["calibrated_lower"] = np.maximum(0, test["raw_lower"] - margin)
            test["calibrated_upper"] = test["raw_upper"] + margin
            test["conformal_margin"] = margin
            test["fold"] = fold_idx
            y = test["units_sold"].to_numpy()
            test["raw_covered"] = (y >= test["raw_lower"]) & (y <= test["raw_upper"])
            test["calibrated_covered"] = (y >= test["calibrated_lower"]) & (y <= test["calibrated_upper"])
            test["raw_width"] = test["raw_upper"] - test["raw_lower"]
            test["calibrated_width"] = test["calibrated_upper"] - test["calibrated_lower"]
            test["raw_interval_score"] = (
                test["raw_width"]
                + (2 / alpha) * np.maximum(test["raw_lower"] - y, 0)
                + (2 / alpha) * np.maximum(y - test["raw_upper"], 0)
            )
            test["calibrated_interval_score"] = (
                test["calibrated_width"]
                + (2 / alpha) * np.maximum(test["calibrated_lower"] - y, 0)
                + (2 / alpha) * np.maximum(y - test["calibrated_upper"], 0)
            )
            details.append(test[[
                "fold", "item_id", "day_number", "date", "units_sold", "raw_lower", "raw_upper",
                "calibrated_lower", "calibrated_upper", "conformal_margin",
                "raw_covered", "calibrated_covered", "raw_width", "calibrated_width",
                "raw_interval_score", "calibrated_interval_score",
            ]])
            margin_records.append({
                "fold": fold_idx, "item_id": product_id,
                "train_end_day": train_end,
                "calibration_start_day": train_end + 1,
                "calibration_end_day": cutoff,
                "evaluation_start_day": cutoff + 1,
                "evaluation_end_day": cutoff + horizon_days,
                "calibration_observations": len(cal),
                "calibration_raw_coverage_pct": 100 * (
                    (cal["units_sold"] >= cal["raw_lower"]) &
                    (cal["units_sold"] <= cal["raw_upper"])
                ).mean(),
                "conformal_margin": margin,
            })

        fold_detail = pd.concat(details[-len(products):], ignore_index=True)
        fold_records.append({
            "fold": fold_idx, "train_end_day": train_end,
            "calibration_start_day": train_end + 1,
            "calibration_end_day": cutoff,
            "evaluation_start_day": cutoff + 1,
            "evaluation_end_day": cutoff + horizon_days,
            "n_forecasts": len(fold_detail),
            "raw_coverage_pct": 100 * fold_detail["raw_covered"].mean(),
            "calibrated_coverage_pct": 100 * fold_detail["calibrated_covered"].mean(),
            "raw_avg_width": fold_detail["raw_width"].mean(),
            "calibrated_avg_width": fold_detail["calibrated_width"].mean(),
            "raw_interval_score": fold_detail["raw_interval_score"].mean(),
            "calibrated_interval_score": fold_detail["calibrated_interval_score"].mean(),
        })

    observations = pd.concat(details, ignore_index=True)
    product_summary = (
        observations.groupby("item_id")
        .agg(
            forecasts=("units_sold", "size"),
            raw_coverage_pct=("raw_covered", lambda x: 100 * x.mean()),
            calibrated_coverage_pct=("calibrated_covered", lambda x: 100 * x.mean()),
            raw_avg_width=("raw_width", "mean"),
            calibrated_avg_width=("calibrated_width", "mean"),
            raw_interval_score=("raw_interval_score", "mean"),
            calibrated_interval_score=("calibrated_interval_score", "mean"),
        ).reset_index()
    )
    return (
        pd.DataFrame(fold_records),
        observations,
        product_summary,
        pd.DataFrame(margin_records),
    )
