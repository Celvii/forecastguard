# ForecastGuard — portfolio text

## GitHub repository description (short)

Uncertainty-aware retail sales forecasting with LightGBM quantile regression, conformal calibration, rolling backtests, and simulated inventory decisions.

## LinkedIn project entry

**Title:** ForecastGuard — Uncertainty-Aware Retail Demand Forecasting

**Description:**
Built an end-to-end Python forecasting research system using Walmart M5 historical sales data. Developed pooled LightGBM quantile models for five product series, evaluated inventory decisions over eight expanding-window historical backtests (2,240 product-day forecasts), and measured a **21.64% reduction in hypothetical inventory costs** relative to a seasonal quantile baseline. Evaluated product-level prediction uncertainty using chronological conformal calibration, increasing aggregate historical interval coverage from **87.19% to 91.29%**. Organized the project into reusable Python pipelines, unit tests, saved model inference, and Streamlit visualizations. Costs are simulations, not verified retailer savings.

**Suggested skills:** Python, Pandas, Scikit-learn, LightGBM, Time Series Forecasting, Quantile Regression, Conformal Prediction, Machine Learning, Pytest, Streamlit, Git.

**Repository:** https://github.com/Celvii/forecastguard
**Live demo:** Add the tested Streamlit URL after deployment.

## Resume (choose one or two bullets)

- Developed a Python/LightGBM quantile-forecasting system for five Walmart M5 product series; achieved **21.64% lower simulated inventory cost** versus a seasonal quantile baseline across **2,240 product-day predictions** and eight chronological backtesting windows.
- Implemented product-level conformal calibration, improving empirical 90%-interval coverage from **87.19% to 91.29%**, and packaged reproducible pipelines, automated tests, saved-model inference, and a Streamlit research dashboard.

## Optional LinkedIn announcement (post)

I recently completed ForecastGuard, a Python data science project exploring how uncertainty-aware forecasting can improve inventory decisions.

Using historical Walmart M5 sales data, I built pooled LightGBM quantile models, evaluated them across eight chronological backtesting windows, and compared cost-sensitive ordering decisions with seasonal baselines. The model achieved **21.64% lower simulated inventory costs** across 2,240 product-day forecasts under the experiment's stated cost assumptions.

I also evaluated prediction intervals using conformal calibration. Historical empirical coverage improved from 87.19% to 91.29% against a nominal 90% target.

The main lesson: accurate predictions and good decisions aren't the same thing. Quantifying uncertainty matters when the costs of overforecasting and underforecasting differ.

The repository includes the modeling pipeline, validation approach, tests, documentation, and a Streamlit research demonstration. These findings are historical simulations, not real-world Walmart savings.

GitHub: https://github.com/Celvii/forecastguard
Live demo: [add URL after deployment]
