"""Generate q05/q80/q95 next-day forecasts with saved ForecastGuard models.

From the ForecastGuard project root:
    python -m scripts.predict_next_day
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.inference import build_next_day_features
from src.intervals import non_crossing_quantiles

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
OUTPUT_PATH = ROOT / "results" / "next_day_forecast.csv"

ARTIFACTS = {
    "q05": ("quantile_q05.joblib", 0.05),
    "q80": ("quantile_q80.joblib", 0.80),
    "q95": ("quantile_q95.joblib", 0.95),
}


def load_artifacts():
    artifacts = {}
    for key, (filename, expected_quantile) in ARTIFACTS.items():
        path = MODEL_DIR / filename
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing {path}. Train interval models first: "
                "python -m scripts.train_interval_models"
            )
        artifact = joblib.load(path)
        if not np.isclose(float(artifact["quantile"]), expected_quantile):
            raise ValueError(f"Unexpected quantile metadata in {path.name}")
        artifacts[key] = artifact

    reference = artifacts["q80"]
    for key, other in artifacts.items():
        if int(other["last_training_day"]) != int(reference["last_training_day"]):
            raise ValueError(f"Model {key} was trained through a different day")
        if list(other["features"]) != list(reference["features"]):
            raise ValueError(f"Model {key} has a different feature schema")
        if set(other["product_ids"]) != set(reference["product_ids"]):
            raise ValueError(f"Model {key} was trained on different products")
    return artifacts


def main():
    artifacts = load_artifacts()
    frame = build_next_day_features(ROOT, artifacts["q80"])
    X = frame[artifacts["q80"]["features"]]

    raw = {
        key: artifacts[key]["model"].predict(X)
        for key in ("q05", "q80", "q95")
    }
    lower, q80, upper, repaired = non_crossing_quantiles(
        raw["q05"], raw["q80"], raw["q95"]
    )

    output = frame[["date", "day_number", "item_id"]].copy()
    output["lower_q05"] = lower
    output["predicted_q80"] = q80
    output["upper_q95"] = upper
    output["interval_width"] = upper - lower
    output["suggested_order_units"] = np.ceil(q80).astype(int)
    output["interval_repaired"] = repaired
    output = output.sort_values("item_id").reset_index(drop=True)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(OUTPUT_PATH, index=False)
    print("NEXT-DAY FORECAST (nominal, UNCALIBRATED 90% intervals)")
    print(f"Forecast date: {pd.to_datetime(output['date'].iloc[0]).date()}")
    print(output[["item_id", "lower_q05", "predicted_q80", "upper_q95",
                  "suggested_order_units", "interval_repaired"]]
          .round(2).to_string(index=False))
    print(f"\nIntervals requiring non-crossing repair: {int(repaired.sum())}")
    print(f"Saved: {OUTPUT_PATH}")
    print("These are illustrative order-up-to quantities, not actual purchase orders.")


if __name__ == "__main__":
    main()
