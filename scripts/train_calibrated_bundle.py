"""Train a separate deployed bundle with a reserved trailing calibration set.

Run: python -m scripts.train_calibrated_bundle
"""
from pathlib import Path
import joblib
from src.data_pipeline import prepare_product_data
from src.deployment_calibration import fit_deployment_bundle

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / "calibrated_deployment_bundle.joblib"

def main():
    df = prepare_product_data(ROOT)
    bundle = fit_deployment_bundle(
        df,
        training_end_day=1885,
        calibration_end_day=1941,
        alpha=0.10,
    )
    MODEL.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, MODEL)
    print("Saved:", MODEL)
    print("Model training: days 1–1885 (price-present rows only)")
    print("Per-product calibration: days 1886–1941")
    print("Training rows:", bundle["training_rows"])
    print("Calibration rows:", bundle["calibration_rows"])
    print("Per-product conformal margins:")
    for item_id, q in bundle["conformal_margins"].items():
        print(f"  {item_id}: {q:.2f} units ({bundle['calibration_counts'][item_id]} days)")
    print("This calibration period is NOT an independent final-test evaluation.")

if __name__ == "__main__":
    main()
