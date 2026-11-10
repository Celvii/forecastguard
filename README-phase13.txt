ForecastGuard Phase 13: Next-day prediction intervals
====================================================

Prerequisites
-------------
- Existing ForecastGuard project, src/inference.py, and 3 fitted model files:
  models/quantile_q05.joblib, models/quantile_q80.joblib, models/quantile_q95.joblib
- Existing Streamlit dashboard and results CSVs

Install
-------
1. Back up your customized app/dashboard.py and scripts/predict_next_day.py if needed.
2. Extract this archive into the ForecastGuard PROJECT ROOT, merging directories.
   Do not replace data, trained models, notebooks, or existing results.
3. From the project root:
   python -m pip install -r phase13-requirements.txt
   python -m pytest -q
   python -m scripts.predict_next_day
   python -m streamlit run app/dashboard.py
4. Open the "Next-day recommendations" tab. It should show Q05, Q80, Q95,
   an interval visualization, and a download button.

What it does
------------
- Loads the three saved pipeline artifacts and validates model metadata.
- Builds one-day-ahead features from the M5 sales history (historical day 1942).
- Writes results/next_day_forecast.csv with Q05, Q80, Q95, suggested orders,
  interval width, and a non-crossing adjustment flag.
- Preserves the old predicted_q80 and suggested_order_units columns for compatibility.
- Uses conservative min/max bound adjustments if independently fitted quantiles
  cross; it does NOT alter Q80 inventory-order advice.

Limitations
-----------
- Q05-Q95 is NOMINAL 90% only. No product-level conformal calibration or empirical
  interval coverage validation has yet been run.
- Forecasts are illustrative historical M5 forecasts, not live 2026 forecasts.
- Only load joblib artifacts that you produced and trust.
