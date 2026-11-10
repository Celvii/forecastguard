"""ForecastGuard Streamlit dashboard for previously generated historical backtests."""
from pathlib import Path
import sys

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.interval_reliability import render_interval_reliability
from app.calibrated_recommendations import render_calibrated_recommendations

st.set_page_config(
    page_title="ForecastGuard | Retail Demand Intelligence",
    page_icon="📦",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def load_csv(path: str, last_modified_ns: int, parse_dates: bool = False) -> pd.DataFrame:
    """Load a CSV and refresh the cache whenever its file timestamp changes."""
    _ = last_modified_ns
    return pd.read_csv(path, parse_dates=["date"] if parse_dates else None)


def result_file(name: str, parse_dates: bool = False, required: bool = True):
    path = RESULTS_DIR / name
    if not path.is_file():
        if required:
            st.error(
                f"Missing {path}. Run the backtesting / next-day forecast scripts "
                "before opening this dashboard."
            )
            st.stop()
        return None
    return load_csv(str(path), path.stat().st_mtime_ns, parse_dates)


def check_columns(frame: pd.DataFrame, required: set[str], file_name: str) -> None:
    missing = required.difference(frame.columns)
    if missing:
        st.error(f"{file_name} is missing columns: {', '.join(sorted(missing))}")
        st.stop()


folds = result_file("backtest_summary.csv")
predictions = result_file("backtest_predictions.csv", parse_dates=True)
next_day = result_file("next_day_forecast.csv", parse_dates=True, required=False)

check_columns(
    folds,
    {"Fold", "ML Cost", "Seasonal Cost", "Savings (%)", "Quantile Coverage (%)"},
    "backtest_summary.csv",
)
check_columns(
    predictions,
    {"date", "item_id", "day_number", "units_sold", "predicted_quantile",
     "ml_order", "seasonal_order", "ml_cost", "seasonal_cost"},
    "backtest_predictions.csv",
)
if next_day is not None:
    check_columns(
        next_day,
        {"date", "item_id", "predicted_q80", "suggested_order_units"},
        "next_day_forecast.csv",
    )

st.title("ForecastGuard")
st.caption("Retail demand forecasting · quantile ML · simulated inventory decisions")
st.info(
    "Historical Walmart M5 research prototype. Costs are hypothetical: "
    "USD 2 per surplus unit and USD 8 per shortage unit, not verified retail savings. "
    "Sales are used as a proxy for demand; inventory and stockouts are not observed."
)

ml_total = float(predictions["ml_cost"].sum())
seasonal_total = float(predictions["seasonal_cost"].sum())
reduction = (
    (seasonal_total - ml_total) / seasonal_total * 100
    if seasonal_total > 0 else float("nan")
)
win_count = int((folds["ML Cost"] < folds["Seasonal Cost"]).sum())
coverage = float((predictions["units_sold"] <= predictions["predicted_quantile"]).mean() * 100)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Simulated cost reduction", f"{reduction:.2f}%")
m2.metric("Total ML cost", f"${ml_total:,.0f}")
m3.metric("Winning backtest folds", f"{win_count}/{len(folds)}")
m4.metric("Empirical Q80 coverage", f"{coverage:.1f}%")

st.caption(
    f"Evaluation: {len(predictions):,} product-day predictions, "
    f"{predictions['item_id'].nunique()} products, "
    f"{len(folds)} expanding-window backtests."
)

overview_tab, products_tab, forecast_tab, calibrated_tab, reliability_tab, methods_tab = st.tabs(
    ["Backtest overview", "Product explorer", "Next-day recommendations",
     "Calibrated next-day", "Interval reliability", "Methodology"]
)

with overview_tab:
    st.subheader("Backtest performance")
    left, right = st.columns(2)
    with left:
        st.write("Simulated cost per fold (lower is better)")
        st.bar_chart(folds.set_index("Fold")[["ML Cost", "Seasonal Cost"]])
    with right:
        st.write("ML cost reduction vs seasonal quantile (%)")
        st.bar_chart(folds.set_index("Fold")[["Savings (%)"]])
    st.dataframe(folds, use_container_width=True, hide_index=True)
    st.download_button(
        "Download fold results (CSV)",
        data=folds.to_csv(index=False).encode("utf-8"),
        file_name="forecastguard_backtest_summary.csv",
        mime="text/csv",
    )

with products_tab:
    st.subheader("Product-level forecasts")
    product_ids = sorted(predictions["item_id"].unique().tolist())
    product_id = st.selectbox("Choose product", product_ids)
    product = (
        predictions.loc[predictions["item_id"] == product_id]
        .sort_values("date")
        .copy()
    )
    product_ml_cost = float(product["ml_cost"].sum())
    product_seasonal_cost = float(product["seasonal_cost"].sum())
    product_savings = (
        100 * (product_seasonal_cost - product_ml_cost) / product_seasonal_cost
        if product_seasonal_cost else float("nan")
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("Cost reduction", f"{product_savings:.2f}%")
    c2.metric("ML cost", f"${product_ml_cost:,.0f}")
    c3.metric(
        "Q80 coverage",
        f"{(product['units_sold'] <= product['predicted_quantile']).mean() * 100:.1f}%",
    )
    st.write("Observed unit sales vs 80th-percentile prediction")
    st.line_chart(
        product.set_index("date")[["units_sold", "predicted_quantile"]],
        use_container_width=True,
    )
    st.caption(
        "The Q80 curve is a conditional upper quantile, not a point-forecast mean "
        "or a 90% prediction interval."
    )
    st.dataframe(
        product[["date", "units_sold", "predicted_quantile", "ml_order", "seasonal_order", "ml_cost", "seasonal_cost"]].tail(28),
        use_container_width=True,
        hide_index=True,
    )
    st.download_button(
        "Download selected product predictions",
        data=product.to_csv(index=False).encode("utf-8"),
        file_name=f"forecastguard_{product_id}_predictions.csv",
        mime="text/csv",
    )
    st.subheader("Comparison across all products")
    by_product = predictions.groupby("item_id", as_index=True)[["ml_cost", "seasonal_cost"]].sum()
    st.bar_chart(by_product, use_container_width=True)

with forecast_tab:
    st.subheader("Saved next-day recommendations")
    if next_day is None:
        st.warning(
            "No next_day_forecast.csv exists. From the project root, run "
            "`python -m scripts.predict_next_day` and refresh."
        )
    else:
        forecast_date = next_day["date"].iloc[0].date()
        st.write(f"Forecast date: **{forecast_date}**")
        interval_cols = {"lower_q05", "predicted_q80", "upper_q95"}
        has_intervals = interval_cols.issubset(next_day.columns)

        if has_intervals:
            import altair as alt

            st.caption(
                "Nominal 90% prediction interval: estimated 5th–95th percentiles. "
                "These saved next-day models are NOT conformal-calibrated. "
                "The separate historical calibration backtest appears in Interval reliability; "
                "those results do not calibrate this saved model."
            )
            st.dataframe(
                next_day[["item_id", "lower_q05", "predicted_q80", "upper_q95",
                          "suggested_order_units"]].round(2),
                use_container_width=True,
                hide_index=True,
            )
            plot = next_day.copy()
            plot["item_id"] = plot["item_id"].astype(str)
            base = alt.Chart(plot).encode(
                y=alt.Y("item_id:N", title="Product", sort=sorted(plot["item_id"].unique())),
            )
            intervals = base.mark_rule(strokeWidth=7, color="#32C8C3").encode(
                x=alt.X("lower_q05:Q", title="Forecast units sold", scale=alt.Scale(zero=True)),
                x2="upper_q95:Q",
                tooltip=[
                    "item_id:N",
                    alt.Tooltip("lower_q05:Q", format=".1f", title="Q05"),
                    alt.Tooltip("predicted_q80:Q", format=".1f", title="Q80"),
                    alt.Tooltip("upper_q95:Q", format=".1f", title="Q95"),
                ],
            )
            points = base.mark_circle(size=135, color="#FBBF24").encode(
                x="predicted_q80:Q",
                tooltip=["item_id:N", alt.Tooltip("predicted_q80:Q", format=".1f")],
            )
            st.altair_chart((intervals + points).properties(height=240), use_container_width=True)
            st.caption("Teal line: estimated Q05–Q95 range. Yellow dot: Q80 inventory target.")
            if "interval_repaired" in next_day.columns:
                repaired = (
                    next_day["interval_repaired"].astype(str).str.lower() == "true"
                ).sum()
                if repaired:
                    st.warning(
                        f"{repaired} product(s) had crossing quantile estimates. "
                        "Bounds were expanded to contain Q80; these are not "
                        "calibration adjustments."
                    )
        else:
            st.warning(
                "This is a Q80-only forecast file. To add prediction intervals, "
                "run `python -m scripts.predict_next_day` after installing Phase 13."
            )
            st.dataframe(
                next_day[["item_id", "predicted_q80", "suggested_order_units"]],
                use_container_width=True,
                hide_index=True,
            )

        st.metric("Total illustrative order units", f"{int(next_day['suggested_order_units'].sum()):,}")
        st.download_button(
            "Download order recommendations",
            data=next_day.to_csv(index=False).encode("utf-8"),
            file_name="forecastguard_next_day_forecast.csv",
            mime="text/csv",
        )
        st.caption(
            "These are historical out-of-sample forecasts for the day following "
            "the M5 sales data (2016), not live 2026 predictions or actual purchase orders. "
            "Suggested quantities ignore inventory on hand, lead times, and stockouts."
        )

with calibrated_tab:
    render_calibrated_recommendations(RESULTS_DIR)

with reliability_tab:
    render_interval_reliability(RESULTS_DIR)


with methods_tab:
    st.subheader("Evaluation design")
    st.markdown(
        """
- **Dataset:** Walmart M5 historical daily sales, five food products from store CA_1.
- **Prediction:** One-day-ahead conditional 80th percentile of units sold.
- **Models:** Pooled LightGBM quantile regressor compared with a same-weekday seasonal quantile baseline.
- **Backtesting:** Eight expanding training windows, 56 days each; one-day-ahead features use prior observed sales.
- **Training filter:** Omit training targets on days without a weekly selling-price record; evaluation rows are unchanged.
- **Costs:** $2 per surplus unit and $8 per shortage unit, selected for a *hypothetical* simulation.
- **Limitations:** Observed sales are not necessarily unconstrained demand. Zero-sales streaks may reflect availability, and no inventory-on-hand or replenishment constraints are modeled. The same historical windows informed project development, so results are descriptive rather than independently confirmed prospective savings.
- **Uncertainty:** The original Next-day recommendations page uses uncalibrated Q05–Q95 intervals, while the separate Calibrated next-day tab uses a bundle trained through day 1885 and calibrated on days 1886–1941. Both forecast day 1942; the calibrated bundle deliberately uses less training history to reserve calibration data. Historical evaluation does not guarantee future coverage. The separate Phase 14 historical interval backtest uses held-out chronological calibration periods and can be explored under Interval reliability. Its results do not automatically calibrate the saved next-day models. Crossing estimates, if present, are conservatively expanded to include Q80.
"""
    )
