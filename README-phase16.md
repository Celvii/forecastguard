# ForecastGuard Phase 16 — Calibrated Next-Day Deployment

The previous dashboard's historical interval-calibration evaluation reported 87.19% raw and 91.29% calibrated coverage across 2,240 product-day predictions. These numbers are **historical estimates**, not guaranteed future coverage.

## What this adds

- `src/deployment_calibration.py`: separate training/calibration model fit and inference.
- `scripts/train_calibrated_bundle.py`: fits Q05, Q80, Q95 using days through **1885** (only price-present training targets), then calculates a per-product nonnegative CQR margin using **days 1886–1941**.
- `scripts/predict_calibrated_next_day.py`: predicts **day 1942**, using observed history through day 1941 and scheduled calendar inputs. Does **not** read day 1942 sales.
- `app/calibrated_recommendations.py`: interactive chart for the new calibrated intervals.
- `app/dashboard.py`: Phase 15 dashboard with an additional **Calibrated next-day** tab.
- `tests/test_deployment_calibration.py`: four new tests.

## Install

1. Back up `app/dashboard.py` before extraction. Extract this ZIP into the existing ForecastGuard root, preserving paths.
2. Ensure Phase 13 `src/intervals.py`, Phase 15 `app/interval_reliability.py`, and existing `src/data_pipeline.py`, `src/model.py`, `src/inference.py` are present.
3. From the ForecastGuard root, run:

```powershell
python -m pytest -q
python -m scripts.train_calibrated_bundle
python -m scripts.predict_calibrated_next_day
python -m streamlit run app/dashboard.py
```

4. Open **Calibrated next-day**. The existing **Next-day recommendations** page deliberately remains *uncalibrated*, using your previous all-history models. The new calibrated deployment bundle excludes the 56-day calibration period from model fitting, so Q80 forecasts may differ from those on the original page.

## Artifacts

- `models/calibrated_deployment_bundle.joblib`: the three fitted models, metadata, and per-product conformal margins. Load only trusted model artifacts (joblib is pickle-based).
- `results/next_day_calibrated_forecast.csv`: product-specific calibrated lower/upper bounds, Q80 orders, and adjustment margins. Does not overwrite `results/next_day_forecast.csv`.

## Methodological details

These calibration margins are based on nonconformity scores `max(lower - actual, actual - upper, 0)` with the finite-sample rank `ceil((n+1)*(1-alpha))`. For n=56 and alpha=0.1, the 52nd ordered score is selected per product. Crossing quantile forecasts are first repaired by creating a nonnegative outer envelope of Q05, Q80 and Q95 estimates, both during calibration and deployment.

Product-wise CQR using a chronological calibration block does **not** automatically guarantee 90% future coverage: product demand can shift, the calibration block is small, and the data do not contain actual stock-on-hand observations. Do not report coverage on the calibration block itself as an independent out-of-sample performance measure; rely on the historical Phase 14 rolling evaluation to assess past empirical performance. The phase 14 window results also informed development, and are not prospectively collected holdout evidence.

The Q80 decision target is **not** shifted by interval calibration. It is a hypothetical single-period order-up-to quantity given a $2 per-unit surplus and $8 per-unit shortage cost; it is not a purchase order. Forecast day 1942 is in May 2016, **not 2026 live forecasts**.
