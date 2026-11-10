"""Chronological per-product CQR calibration for ForecastGuard deployment.

Never use calibration targets to fit the same three quantile models.
This experiment's historical evaluation periods have already informed design,
so calibration is not evidence of fresh out-of-sample performance.
"""
import math
import numpy as np
import pandas as pd

from .data_pipeline import FEATURES
from .intervals import non_crossing_quantiles
from .model import build_quantile_model

QUANTILES = {"q05": 0.05, "q80": 0.80, "q95": 0.95}


def conformal_margin(actual, lower, upper, alpha=0.10):
    """Finite-sample split-conformal rank for nonnegative CQR errors.

    Returns a marginal adjustment under exchangeability, which chronological
    retail demand generally does not satisfy. Validate coverage historically.
    """
    y = np.asarray(actual, dtype=float)
    lo = np.asarray(lower, dtype=float)
    hi = np.asarray(upper, dtype=float)
    if not (y.shape == lo.shape == hi.shape) or y.ndim != 1 or len(y) == 0:
        raise ValueError("Expected equally-sized, nonempty 1D arrays")
    if not (np.isfinite(y).all() and np.isfinite(lo).all() and np.isfinite(hi).all()):
        raise ValueError("Inputs must contain only finite numbers")
    if np.any(lo > hi):
        raise ValueError("Lower bounds must not exceed upper bounds")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")
    scores = np.maximum.reduce([lo - y, y - hi, np.zeros(len(y))])
    k = math.ceil((len(scores) + 1) * (1 - alpha))
    if k > len(scores):
        raise ValueError("Calibration set is too small for a finite margin at this alpha")
    return float(np.sort(scores)[k - 1])


def fit_deployment_bundle(
    df, *, training_end_day=1885, calibration_end_day=1941, alpha=0.10
):
    """Train through 1885, calibrate 1886-1941, forecast day 1942.

    All columns in FEATURES must have been constructed as one-day-ahead inputs.
    The model never trains on the calibration targets.
    """
    if training_end_day >= calibration_end_day:
        raise ValueError("Training must end before calibration")
    if df.duplicated(["item_id", "day_number"]).any():
        raise ValueError("Duplicate product-day rows")
    if int(df["day_number"].max()) != calibration_end_day:
        raise ValueError("Historical data must end on calibration_end_day")
    train = df.loc[(df["day_number"] <= training_end_day) & df["price_available"]].copy()
    cal = df.loc[df["day_number"].between(training_end_day + 1, calibration_end_day)].copy()
    product_ids = sorted(train["item_id"].unique().tolist())
    if not product_ids or len(cal) == 0 or set(cal["item_id"]) != set(product_ids):
        raise ValueError("Training and calibration products differ or are missing")
    if set(cal["day_number"].unique()) != set(range(training_end_day+1, calibration_end_day+1)):
        raise ValueError("Calibration calendar has missing days")
    # Train three distinct quantile models ONLY on training rows.
    models = {}
    for key, quantile in QUANTILES.items():
        model = build_quantile_model(quantile)
        model.fit(train[FEATURES], train["units_sold"])
        models[key] = model
    raw = {key: model.predict(cal[FEATURES]) for key, model in models.items()}
    lo, center, hi, repaired = non_crossing_quantiles(
        raw["q05"], raw["q80"], raw["q95"]
    )
    cal = cal[["item_id", "day_number", "units_sold"]].copy()
    cal["lower"] = lo
    cal["upper"] = hi
    cal["repaired"] = repaired
    margins, sample_counts = {}, {}
    for product_id in product_ids:
        p = cal.loc[cal["item_id"] == product_id]
        expected_count = calibration_end_day - training_end_day
        if len(p) != expected_count:
            raise ValueError(
                f"Expected {expected_count} calibration days for {product_id}, got {len(p)}"
            )
        margins[product_id] = conformal_margin(
            p["units_sold"].to_numpy(),
            p["lower"].to_numpy(),
            p["upper"].to_numpy(),
            alpha=alpha,
        )
        sample_counts[product_id] = len(p)
    bundle = {
        "models": models,
        "features": list(FEATURES),
        "product_ids": product_ids,
        "conformal_margins": margins,
        "alpha": float(alpha),
        "target_coverage": float(1 - alpha),
        "last_training_day": int(training_end_day),
        "last_history_day": int(calibration_end_day),
        "calibration_first_day": int(training_end_day + 1),
        "calibration_last_day": int(calibration_end_day),
        "calibration_counts": sample_counts,
        "training_rows": int(len(train)),
        "calibration_rows": int(len(cal)),
        "crossings_repaired_in_calibration": int(np.sum(repaired)),
        "calibration_method": "per-product nonnegative split CQR; chronological calibration",
    }
    return bundle


def predict_with_calibration(bundle, feature_frame):
    """Use matching trained models + per-product calibration margins."""
    if feature_frame["item_id"].duplicated().any():
        raise ValueError("Each product should appear once in a next-day forecast")
    expected = set(bundle["product_ids"])
    if set(feature_frame["item_id"]) != expected:
        raise ValueError("Prediction products do not match fitted models")
    X = feature_frame[bundle["features"]]
    raw = {k: bundle["models"][k].predict(X) for k in QUANTILES}
    lo, q80, hi, repaired = non_crossing_quantiles(
        raw["q05"], raw["q80"], raw["q95"]
    )
    margins = feature_frame["item_id"].map(bundle["conformal_margins"])
    if margins.isna().any():
        raise ValueError("Missing calibration margin for at least one product")
    margins = margins.to_numpy(dtype=float)
    out = feature_frame[["date", "day_number", "item_id"]].copy()
    out["lower_q05"] = lo
    out["predicted_q80"] = q80
    out["upper_q95"] = hi
    out["conformal_margin"] = margins
    out["calibrated_lower"] = np.maximum(0, lo-margins)
    out["calibrated_upper"] = hi+margins
    out["calibrated_interval_width"] = out["calibrated_upper"] - out["calibrated_lower"]
    out["suggested_order_units"] = np.ceil(q80).astype(int)
    out["interval_repaired"] = repaired
    out["calibration_method"] = "chronological per-product CQR"
    return out.sort_values("item_id").reset_index(drop=True)
