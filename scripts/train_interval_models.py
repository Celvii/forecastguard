
from pathlib import Path
import joblib

from src.data_pipeline import (
    prepare_product_data,
    FEATURES
)
from src.model import build_quantile_model

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

# Load historical dataset
df = prepare_product_data(PROJECT_ROOT)

# Match the final model's training policy
train = df[df["price_available"]].copy()

# Train lower and upper quantile models
for quantile in [0.05, 0.95]:

    print(f"Training quantile: {quantile}")

    model = build_quantile_model(quantile)

    model.fit(
        train[FEATURES],
        train["units_sold"]
    )

    artifact = {
        "model": model,
        "features": list(FEATURES),
        "quantile": quantile,
        "last_training_day": int(
            train["day_number"].max()
        ),
        "product_ids": sorted(
            train["item_id"].unique().tolist()
        )
    }

    filename = f"quantile_q{int(quantile * 100):02d}.joblib"
    path = MODEL_DIR / filename

    joblib.dump(artifact, path)

    print(f"Saved: {path.name}")

print("\nBoth interval models saved successfully!")
