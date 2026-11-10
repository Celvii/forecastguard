
from pathlib import Path
import joblib
import numpy as np

from src.data_pipeline import (
    prepare_product_data,
    FEATURES
)
from src.model import build_quantile_model
from src.inventory import optimal_quantile


# Locate project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODEL_DIR / "quantile_q80.joblib"


def main():
    # Load all historical observations
    df = prepare_product_data(PROJECT_ROOT)

    # Use the previously selected training policy
    train = df[df["price_available"]].copy()

    # Business cost assumptions
    q = optimal_quantile(
        overstock_cost=2,
        understock_cost=8
    )

    # Build and train the final model
    model = build_quantile_model(q)

    model.fit(
        train[FEATURES],
        train["units_sold"]
    )

    # Save the fitted pipeline and metadata
    artifact = {
        "model": model,
        "features": list(FEATURES),
        "quantile": q,
        "training_rows": len(train),
        "last_training_day": int(
            train["day_number"].max()
        ),
        "product_ids": sorted(
            train["item_id"].unique().tolist()
        )
    }

    joblib.dump(artifact, MODEL_PATH)

    print("Model saved successfully!")
    print("Training observations:", len(train))
    print("Target quantile:", q)
    print("Model location:", MODEL_PATH)

    # Verify the saved model
    loaded = joblib.load(MODEL_PATH)

    sample = train[FEATURES].tail(10)

    original_predictions = model.predict(sample)
    loaded_predictions = loaded["model"].predict(
        sample[loaded["features"]]
    )

    np.testing.assert_allclose(
        original_predictions,
        loaded_predictions,
        rtol=1e-12,
        atol=1e-12
    )

    print("\nModel reload verification passed!")
    print(
        "Sample predictions:",
        np.round(loaded_predictions, 2)
    )


if __name__ == "__main__":
    main()
