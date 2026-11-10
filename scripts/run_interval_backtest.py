"""Run the Phase 14 eight-fold product-level interval backtest.

From the ForecastGuard project root:
    python -m scripts.run_interval_backtest
"""

from pathlib import Path

from src.data_pipeline import prepare_product_data
from src.interval_backtesting import run_interval_backtest

ROOT = Path(__file__).resolve().parents[1]


def main():
    df = prepare_product_data(ROOT)
    folds, predictions, products, margins = run_interval_backtest(df)

    result_dir = ROOT / "results"
    result_dir.mkdir(exist_ok=True)
    folds.to_csv(result_dir / "interval_backtest_summary.csv", index=False)
    predictions.to_csv(result_dir / "interval_backtest_predictions.csv", index=False)
    products.to_csv(result_dir / "interval_product_summary.csv", index=False)
    margins.to_csv(result_dir / "interval_calibration_margins.csv", index=False)

    print("\nPHASE 14: FOLD SUMMARY")
    print(folds.round(2).to_string(index=False))
    print("\nPRODUCT SUMMARY")
    print(products.round(2).to_string(index=False))
    print("\nOVERALL COVERAGE AND SHARPNESS")
    print(f"Target coverage: 90.00%")
    print(f"Raw coverage: {100 * predictions['raw_covered'].mean():.2f}%")
    print(f"Calibrated coverage: {100 * predictions['calibrated_covered'].mean():.2f}%")
    print(f"Raw average interval width: {predictions['raw_width'].mean():.2f} units")
    print(f"Calibrated average interval width: {predictions['calibrated_width'].mean():.2f} units")
    print(f"Raw mean interval score: {predictions['raw_interval_score'].mean():.2f}")
    print(f"Calibrated mean interval score: {predictions['calibrated_interval_score'].mean():.2f}")
    print(f"Total evaluated forecasts: {len(predictions)}")
    print("\nSaved four CSV reports in:", result_dir)
    print("Note: This is historical empirical validation; time-series exchangeability is not assured.")


if __name__ == "__main__":
    main()
