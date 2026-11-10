# ForecastGuard — Phase 14: Product-Level Interval Calibration

## Installation

1. Unzip this add-on into your ForecastGuard project root, merging directories. These are **new files** and should not replace your existing modules or dashboard.
2. From the project root, run:

   ```powershell
   python -m pytest -q
   python -m scripts.run_interval_backtest
   ```

3. The script produces four files in `results/`:

   - `interval_backtest_summary.csv` — eight chronological evaluation windows.
   - `interval_backtest_predictions.csv` — 2,240 per-product, per-day forecasts with raw/calibrated bounds, coverage and scores.
   - `interval_product_summary.csv` — coverage, width, interval score by product.
   - `interval_calibration_margins.csv` — per-fold, per-product conformal adjustments and chronology.

## Method

For each of eight folds, fit independent Q05 and Q95 LightGBM models on data through `cutoff - 56` (training targets with observed price records only), calibrate each product on the following **56 days**, then measure coverage and width on the subsequent **56 days**. Calibration and evaluation targets do not overlap. Rolling lags during the evaluation use past *observed* sales (one-day-ahead repeated daily), not a 56-day blind forecast. Each fold fits fresh models. Calibration is product-specific, not pooled across item IDs.

Conformal margin is the 52nd smallest of 56 nonconformity scores for 90% nominal coverage. Time-series observations need not be exchangeable; results are empirical historical coverage only, not guaranteed coverage on future sales.

The previously saved `models/quantile_q05.joblib`, `quantile_q80.joblib`, and `quantile_q95.joblib` remain untouched. **Do not apply these fold-specific calibration margins directly to those final deployment models**, because those models were fit on different data and contain the calibration dates. A separate chronologically trained and calibrated deployment artifact will be constructed after reviewing the backtest.

No inventory cost simulation changes are made in this phase. The existing Streamlit dashboard remains functional.
