
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from lightgbm import LGBMRegressor

from .data_pipeline import (
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES
)


def build_quantile_model(quantile=0.80):
    """Create a fresh LightGBM quantile forecasting pipeline."""

    if not 0 < quantile < 1:
        raise ValueError("Quantile must be between 0 and 1.")

    preprocessor = ColumnTransformer([
        ("numeric", "passthrough", NUMERIC_FEATURES),
        ("categorical",
         OneHotEncoder(handle_unknown="ignore"),
         CATEGORICAL_FEATURES)
    ])

    model = LGBMRegressor(
        objective="quantile",
        alpha=quantile,
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=15,
        min_child_samples=30,
        random_state=42,
        verbosity=-1
    )

    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

median_model = build_quantile_model(0.50)
inventory_model = build_quantile_model(0.80)
upper_model = build_quantile_model(0.95)