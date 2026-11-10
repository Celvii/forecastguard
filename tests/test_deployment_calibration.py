import numpy as np
import pandas as pd
import pytest

from src.deployment_calibration import conformal_margin, predict_with_calibration

def test_conformal_margin_finite_sample_rank():
    # 9 errors, alpha=0.1 => ceil((9+1)*0.9)=9: largest score.
    y = np.arange(1.0, 10.0)
    lower = np.zeros(9)
    upper = np.zeros(9)
    assert conformal_margin(y, lower, upper, alpha=0.1) == 9.0

def test_conformal_margin_rejects_invalid_intervals():
    with pytest.raises(ValueError):
        conformal_margin([1.0], [2.0], [1.0])
    with pytest.raises(ValueError):
        conformal_margin([1.0], [0.0], [2.0], alpha=0.01)

class FakeModel:
    def __init__(self, output):
        self.output = np.asarray(output, dtype=float)
    def predict(self, frame):
        return self.output.copy()

def test_predict_with_calibration_has_correct_bounds():
    bundle = {
        "models": {
            "q05": FakeModel([4, 9]),
            "q80": FakeModel([8, 7]),
            "q95": FakeModel([10, 6]),
        },
        "features": ["item_id"],
        "product_ids": ["A", "B"],
        "conformal_margins": {"A": 2.0, "B": 3.0},
    }
    frame = pd.DataFrame({
        "item_id": ["A", "B"],
        "date": ["2016-05-23", "2016-05-23"],
        "day_number": [1942, 1942],
    })
    out = predict_with_calibration(bundle, frame).set_index("item_id")
    assert out.loc["A", "calibrated_lower"] == 2
    assert out.loc["A", "calibrated_upper"] == 12
    assert out.loc["A", "suggested_order_units"] == 8
    # Q05=9, Q80=7, Q95=6 cross; repaired to [6, 9]
    assert out.loc["B", "calibrated_lower"] == 3
    assert out.loc["B", "calibrated_upper"] == 12
    assert bool(out.loc["B", "interval_repaired"])

def test_prediction_product_mismatch_is_rejected():
    bundle = {"product_ids": ["A"]}
    frame = pd.DataFrame({"item_id": ["B"]})
    with pytest.raises(ValueError):
        predict_with_calibration(bundle, frame)
