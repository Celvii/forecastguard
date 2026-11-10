# ForecastGuard — Phase 15: Historical uncertainty reliability dashboard

This is an add-on to an **existing Phase 13 dashboard** and the **Phase 14 interval backtest**. No trained models, user data, or historical CSV results are included or overwritten.

## Install

1. Back up any locally customized `app/dashboard.py`.
2. Unzip this package **into the ForecastGuard root**, merging the `app/`, `src/`, and `tests/` folders. The updated dashboard retains the Phase 13 tabs and adds **Interval reliability**.
3. Ensure `altair` and `streamlit` are installed (already required for the Phase 13 dashboard).
4. From the **ForecastGuard project root**, run:

   ```powershell
   python -m pytest -q
   python -m scripts.run_interval_backtest
   python -m streamlit run app/dashboard.py
   ```

5. Open the **Interval reliability** tab. If the Phase 14 CSVs are missing, the tab shows a clear instruction instead of stopping the rest of the app.

## Reads Phase 14 files

Mandatory: `results/interval_backtest_predictions.csv`.

Optional (to display the calibration margin): `results/interval_calibration_margins.csv`.

Expected mandatory columns: `fold`, `item_id`, `day_number`, `date`, `units_sold`, `raw_lower`, `raw_upper`, `calibrated_lower`, `calibrated_upper`.

The dashboard **recalculates** coverage, average width, and interval score from individual forecast outcomes. It does not merely echo summary CSV values. It also checks for missing columns, non-finite numbers, and invalid intervals.

## What it shows

- Empirical raw and calibrated coverage vs a nominal 90% target.
- Changes in interval width and mean interval score (smaller is better).
- Fold-level coverage and sharpness charts.
- Product-level reliability comparison.
- Product + fold explorer plotting observed sales, raw interval band, and calibrated interval band.
- Optional fold-specific conformal margin readout.

## Critical distinction

Phase 14 backtests train models through `cutoff - 56`, calibrate each product on the next 56 days, and evaluate the subsequent 56 days. The saved Phase 13 Q05/Q95 models are trained on the full available dataset, so **the historical fold-specific conformal margins are not valid for those final saved models**. This dashboard DOES NOT change the next-day forecasts or label them calibrated.

Results reflect historical, development-informed backtests. Neither exchangeability assumptions nor reliable future coverage are guaranteed for time-varying sales data. Observed sales are an imperfect demand proxy; inventory state is unavailable.

## Files

- `app/dashboard.py` — Phase 13 dashboard with additional tab.
- `app/interval_reliability.py` — Streamlit reliability views.
- `app/__init__.py` — package marker.
- `src/reliability_reporting.py` — independent metric validation and aggregation.
- `tests/test_reliability_reporting.py` — five additional unit tests.
