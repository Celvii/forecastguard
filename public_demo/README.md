# ForecastGuard — Public Results Companion

This is a lightweight, stand-alone Streamlit overview of **aggregate historical project metrics**, not the complete private/local ForecastGuard research dashboard. No M5 row-level sales data, derived forecast rows, or serialized trained models are included.

**Data caveat:** The displayed aggregate results come from a Walmart M5 historical backtest using hypothetical inventory costs, and are not real retailer financial outcomes. See the main project for model, calibration, and uncertainty limitations.

Run locally:

```bash
python -m pip install -r requirements.txt
python -m streamlit run streamlit_app.py
```

On Streamlit Community Cloud, use this folder as repository root and set entrypoint `streamlit_app.py`.
