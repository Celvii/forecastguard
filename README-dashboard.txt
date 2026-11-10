ForecastGuard dashboard add-on

1. Unzip and COPY the `app/` and `.streamlit/` folders into your existing ForecastGuard project root, merging folders if needed.
2. Install dependencies from the ForecastGuard root:
   python -m pip install -r dashboard-requirements.txt
3. Ensure you have generated these files in `results/`:
   backtest_summary.csv
   backtest_predictions.csv
   next_day_forecast.csv
4. Launch:
   python -m streamlit run app/dashboard.py

The app reads saved results. It does not retrain models.
Figures represent historical hypothetical costs, not verified real-world savings.
