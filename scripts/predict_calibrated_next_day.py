"""Predict next historical day with the separately calibrated bundle.

Run: python -m scripts.predict_calibrated_next_day
"""
from pathlib import Path
import joblib
import pandas as pd
from src.inference import build_next_day_features
from src.deployment_calibration import predict_with_calibration

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / "calibrated_deployment_bundle.joblib"
OUTPUT = ROOT / "results" / "next_day_calibrated_forecast.csv"

def main():
    if not MODEL.is_file():
        raise FileNotFoundError(
            "Train the calibrated bundle first: python -m scripts.train_calibrated_bundle"
        )
    # Load only joblib artifacts you generated/trust (pickle-based format).
    bundle = joblib.load(MODEL)
    # Existing inference helper uses last_training_day to check *history freshness*.
    # Pass the last available sales day, not the older model-fitting cutoff.
    history_reference = {
        "last_training_day": bundle["last_history_day"],
        "product_ids": bundle["product_ids"],
    }
    features = build_next_day_features(ROOT, history_reference)
    if features["day_number"].nunique() != 1 or int(features["day_number"].iloc[0]) != bundle["last_history_day"] + 1:
        raise ValueError("Next-day feature date is not consistent with bundle")
    output = predict_with_calibration(bundle, features)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(OUTPUT, index=False)
    print("CALIBRATED HISTORICAL NEXT-DAY FORECAST")
    print("Forecast day:", int(output["day_number"].iloc[0]))
    print("Date:", pd.to_datetime(output["date"].iloc[0]).date())
    print(output[["item_id", "calibrated_lower", "predicted_q80", "calibrated_upper", "suggested_order_units", "conformal_margin"]].round(2).to_string(index=False))
    print("Saved:", OUTPUT)
    print("Intervals are calibrated using historical days, but future coverage is not guaranteed.")
    print("Order recommendations are illustrative; stock on hand and lead times are unknown.")

if __name__ == "__main__":
    main()
