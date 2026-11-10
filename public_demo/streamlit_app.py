"""Public aggregated-results companion: no M5 data or model artifact dependencies."""
import pandas as pd
import streamlit as st

st.set_page_config(page_title="ForecastGuard | Research Summary", page_icon="📈", layout="wide")
st.title("ForecastGuard")
st.caption("Historical uncertainty-aware retail forecasting · public research summary")
st.info("Historical M5 research prototype. All inventory costs are hypothetical: "
        "USD 2 per surplus unit and USD 8 per shortage unit in the primary experiment. "
        "This is NOT live retail forecasting or verified Walmart savings.")

c1,c2,c3,c4=st.columns(4)
c1.metric("Simulated cost reduction", "21.64%")
c2.metric("ML simulated cost", "$87,976")
c3.metric("Winning backtest folds", "7/8")
c4.metric("Product-days evaluated", "2,240")

tab1,tab2,tab3 = st.tabs(["Backtesting", "Uncertainty", "Methodology"])
with tab1:
    st.subheader("Expanding-window cost evaluation")
    folds=pd.DataFrame({
        "Fold":[1,2,3,4,5,6,7,8],
        "Seasonal quantile":[14292,16696,14286,10304,16118,16472,13130,10980],
        "Pooled quantile ML":[12108,10930,11104,10306,12948,9840,10994,9746]
    }).set_index("Fold")
    st.bar_chart(folds)
    st.caption("Eight historical 56-day blocks across five selected products. The data were inspected during iterative development; not a blind prospective trial.")
    scenario=pd.DataFrame({
        "Scenario":["Equal costs (Q50)","Shortage 4× (Q80)","Shortage 9× (Q90)"],
        "ML simulated cost":[45152,87976,118980],
        "Seasonal baseline cost":[65212,112278,156200],
        "Cost reduction (%)":[30.76,21.64,23.83]
    })
    st.dataframe(scenario, hide_index=True, use_container_width=True)
with tab2:
    st.subheader("Historical calibration evaluation")
    a,b,c=st.columns(3)
    a.metric("Raw coverage", "87.19%")
    b.metric("Calibrated coverage", "91.29%", "+4.10 pp")
    c.metric("Nominal target", "90%")
    width=pd.DataFrame({"Average interval width (units)":[38.7,44.7]},index=["Raw","Calibrated"])
    st.bar_chart(width)
    st.caption("Per-product conformal margins widen interval bounds. Historical marginal coverage does not guarantee future or per-product coverage, particularly under distribution shift.")
with tab3:
    st.subheader("What the experiment measures")
    st.markdown("""**Scope:** five high-volume food products at CA_1 in the 2011–2016 Walmart M5 historical dataset.  
**Features:** prior-day and weekly lags, trailing averages, known calendar events and product ID.  
**Decision policy:** target conditional demand quantiles for a simplified single-period inventory-cost objective.  
**Comparison:** pooled LightGBM quantile model versus historical same-weekday quantile baseline.  
**Deployment sample:** model forecasts M5 day 1942 (2016-05-23), not today's sales.""")
    st.warning("Recorded sales may not equal unconstrained demand. The experiment lacks inventory-on-hand, supplier lead time, and confirmed stockout labels. The cost figures are simulations, not real business savings.")
