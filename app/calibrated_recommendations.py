"""Phase 16 dashboard tab: calibrated next-day order-up-to forecasts."""
from pathlib import Path
import pandas as pd
import streamlit as st

REQUIRED = {
    "date", "item_id", "day_number", "lower_q05", "predicted_q80", "upper_q95",
    "calibrated_lower", "calibrated_upper", "conformal_margin", "suggested_order_units",
}

def render_calibrated_recommendations(results_dir):
    path = Path(results_dir) / "next_day_calibrated_forecast.csv"
    st.subheader("Calibrated next-day recommendations")
    st.caption(
        "Historical research forecast: trained through M5 day 1885, "
        "calibrated using days 1886–1941, predicting day 1942. "
        "This is not a live 2026 forecast."
    )
    if not path.is_file():
        st.warning("Run `python -m scripts.train_calibrated_bundle`, then "
                   "`python -m scripts.predict_calibrated_next_day` and refresh.")
        return
    df = pd.read_csv(path, parse_dates=["date"])
    missing = REQUIRED - set(df.columns)
    if missing:
        st.error("Calibrated CSV is missing: " + ", ".join(sorted(missing)))
        return
    if len(df) == 0:
        st.warning("Calibrated CSV contains no predictions.")
        return
    if df["item_id"].duplicated().any():
        st.error("Calibrated file contains duplicate products")
        return
    st.info(
        "The Q05–Q95 bounds have been expanded using separate product-specific "
        "conformal margins. The Q80 order target is unchanged by this interval "
        "calibration. Historical coverage is not a guarantee of future coverage."
    )
    st.write("Forecast date: **" + str(df["date"].dt.date.iloc[0]) + "**")
    view = df[["item_id", "lower_q05", "predicted_q80", "upper_q95",
               "conformal_margin", "calibrated_lower", "calibrated_upper",
               "suggested_order_units"]].copy()
    st.dataframe(view.round(2), use_container_width=True, hide_index=True)
    import altair as alt
    domain = df.copy()
    domain["item_id"] = domain["item_id"].astype(str)
    labels = sorted(domain["item_id"].unique().tolist())
    base = alt.Chart(domain).encode(
        y=alt.Y("item_id:N", title="Product", sort=labels)
    )
    bands = base.mark_rule(strokeWidth=8, color="#32C8C3").encode(
        x=alt.X("calibrated_lower:Q", title="Units sold (calibrated interval)", scale=alt.Scale(zero=True)),
        x2="calibrated_upper:Q",
        tooltip=["item_id:N", alt.Tooltip("calibrated_lower:Q", format=".1f"),
                 alt.Tooltip("calibrated_upper:Q", format=".1f")],
    )
    targets = base.mark_circle(size=145, color="#FBBF24").encode(
        x="predicted_q80:Q",
        tooltip=["item_id:N", alt.Tooltip("predicted_q80:Q", format=".1f")],
    )
    st.altair_chart((bands + targets).properties(height=270), use_container_width=True)
    st.caption("Turquoise: calibrated bounds. Yellow: Q80 inventory decision target. "
               "Inventory on hand and lead times are not considered.")
    st.download_button(
        "Download calibrated recommendations",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="forecastguard_calibrated_forecast.csv",
        mime="text/csv",
    )
