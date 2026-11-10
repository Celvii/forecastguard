"""The Phase 15 historical interval reliability tab for ForecastGuard."""
from __future__ import annotations

from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from src.reliability_reporting import (
    prepare_interval_predictions,
    interval_summary,
    summarize_groups,
)


@st.cache_data(show_spinner=False)
def _read_interval_csv(path_str: str, modified_ns: int) -> pd.DataFrame:
    _ = modified_ns
    return pd.read_csv(path_str)


def _load_frame(path: Path) -> pd.DataFrame:
    return _read_interval_csv(str(path), path.stat().st_mtime_ns)


def render_interval_reliability(results_dir: Path) -> None:
    """Render historical reliability diagnostics; never alter next-day predictions."""
    st.subheader("Product-level interval reliability")
    st.caption(
        "Historical, repeated one-day-ahead predictions from Phase 14. "
        "The nominal target is 90% coverage. Lower interval score is better."
    )
    predictions_path = Path(results_dir) / "interval_backtest_predictions.csv"
    if not predictions_path.is_file():
        st.warning(
            "Phase 14 results are not available yet. From the ForecastGuard project root, "
            "run `python -m scripts.run_interval_backtest`, then refresh this page. "
            "Other dashboard tabs continue to work."
        )
        return

    try:
        obs = prepare_interval_predictions(_load_frame(predictions_path))
    except (ValueError, TypeError, KeyError, pd.errors.ParserError) as exc:
        st.error(f"The Phase 14 prediction CSV could not be validated: {exc}")
        return

    overall = interval_summary(obs)
    per_fold = summarize_groups(obs, "fold")
    per_product = summarize_groups(obs, "item_id")

    a, b, c, d = st.columns(4)
    a.metric("Raw coverage", f"{overall['raw_coverage_pct']:.2f}%")
    b.metric(
        "Calibrated coverage", f"{overall['calibrated_coverage_pct']:.2f}%",
        delta=f"{overall['coverage_delta_pp']:+.2f} pp",
    )
    c.metric("Target coverage", "90%")
    d.metric("Historical predictions", f"{overall['n_forecasts']:,}")

    a, b, c = st.columns(3)
    a.metric("Raw average width", f"{overall['raw_avg_width']:.1f} units")
    b.metric(
        "Calibrated average width", f"{overall['calibrated_avg_width']:.1f} units",
        delta=f"{overall['width_delta']:+.1f} units",
        delta_color="off",
    )
    c.metric(
        "Interval score (calibrated)", f"{overall['calibrated_interval_score']:.1f}",
        delta=f"{overall['score_delta']:+.1f} vs. raw",
        delta_color="inverse",
    )

    if overall["calibrated_coverage_pct"] < 90:
        st.warning(
            "Historical calibrated coverage is below the nominal 90% target. "
            "Inspect the product-level and fold-level failures below."
        )
    else:
        st.info(
            "Historical calibrated coverage meets or exceeds 90% in aggregate. "
            "This does not guarantee future coverage or adequate coverage for every product."
        )

    st.markdown("#### Stability across evaluation windows")
    fold_chart = per_fold[["fold", "raw_coverage_pct", "calibrated_coverage_pct"]].melt(
        id_vars="fold", var_name="method", value_name="coverage"
    )
    fold_chart["method"] = fold_chart["method"].map({
        "raw_coverage_pct": "Raw", "calibrated_coverage_pct": "Calibrated"
    })
    coverage = alt.Chart(fold_chart).mark_line(point=True).encode(
        x=alt.X("fold:O", title="Backtest fold"),
        y=alt.Y("coverage:Q", title="Coverage (%)", scale=alt.Scale(zero=False)),
        color=alt.Color("method:N", title="Interval"),
        tooltip=["fold:O", "method:N", alt.Tooltip("coverage:Q", format=".2f")],
    )
    target = alt.Chart(pd.DataFrame({"target": [90]})).mark_rule(
        strokeDash=[5, 4], color="#9CA3AF"
    ).encode(y="target:Q")
    st.altair_chart((coverage + target).properties(height=290), use_container_width=True)

    left, right = st.columns(2)
    with left:
        st.write("Average interval width per fold (lower = sharper)")
        st.bar_chart(per_fold.set_index("fold")[["raw_avg_width", "calibrated_avg_width"]])
    with right:
        st.write("Mean interval score per fold (lower = better)")
        st.line_chart(per_fold.set_index("fold")[["raw_interval_score", "calibrated_interval_score"]])

    st.markdown("#### Product-level summary")
    st.dataframe(per_product.round(2), hide_index=True, use_container_width=True)
    st.download_button(
        "Download reliability summary",
        data=per_product.to_csv(index=False).encode("utf-8"),
        file_name="forecastguard_reliability_by_product.csv", mime="text/csv",
    )

    st.markdown("#### Inspect actual forecasts")
    products = sorted(obs["item_id"].unique().tolist())
    selected_item = st.selectbox("Product", products, key="interval_product")
    folds = sorted(obs["fold"].unique().tolist())
    selected_fold = st.selectbox("Backtest fold", folds, key="interval_fold")
    selected = obs.loc[
        (obs["item_id"] == selected_item) & (obs["fold"] == selected_fold)
    ].sort_values("date")
    selected_metrics = interval_summary(selected)
    m1, m2, m3 = st.columns(3)
    m1.metric("Raw coverage", f"{selected_metrics['raw_coverage_pct']:.1f}%")
    m2.metric("Calibrated coverage", f"{selected_metrics['calibrated_coverage_pct']:.1f}%")
    m3.metric("Calibrated width", f"{selected_metrics['calibrated_avg_width']:.1f} units")

    # A colored interval band shows whether actual values fell outside either method.
    raw_band = alt.Chart(selected).mark_area(opacity=0.16, color="#64748B").encode(
        x=alt.X("date:T", title="Date"),
        y=alt.Y("raw_lower:Q", title="Sales units"),
        y2="raw_upper:Q",
    )
    calibrated_band = alt.Chart(selected).mark_area(opacity=0.23, color="#28BEB8").encode(
        x="date:T", y="calibrated_lower:Q", y2="calibrated_upper:Q",
    )
    actual_line = alt.Chart(selected).mark_line(
        color="#FBBF24", strokeWidth=2
    ).encode(
        x="date:T", y="units_sold:Q",
        tooltip=[
            alt.Tooltip("date:T", title="Date"),
            alt.Tooltip("units_sold:Q", title="Actual sales"),
            alt.Tooltip("raw_lower:Q", format=".1f"),
            alt.Tooltip("raw_upper:Q", format=".1f"),
            alt.Tooltip("calibrated_lower:Q", format=".1f"),
            alt.Tooltip("calibrated_upper:Q", format=".1f"),
        ],
    )
    st.altair_chart(
        (raw_band + calibrated_band + actual_line).properties(height=340),
        use_container_width=True,
    )
    st.caption(
        "Gray band: raw Q05–Q95 interval. Teal band: historically conformal-calibrated "
        "interval. Yellow line: observed sales. Calibration and evaluation use "
        "separate time periods within each fold."
    )
    st.dataframe(
        selected[["date", "units_sold", "raw_lower", "raw_upper",
                  "calibrated_lower", "calibrated_upper", "raw_covered",
                  "calibrated_covered"]].round(2),
        hide_index=True, use_container_width=True,
    )

    margin_path = Path(results_dir) / "interval_calibration_margins.csv"
    if margin_path.is_file():
        margin_df = _load_frame(margin_path)
        if {"fold", "item_id", "conformal_margin"}.issubset(margin_df.columns):
            filtered = margin_df.loc[
                (margin_df["item_id"] == selected_item)
                & (margin_df["fold"] == selected_fold),
                "conformal_margin",
            ]
            if len(filtered) == 1:
                st.caption(f"Historical product/fold conformal margin: {float(filtered.iloc[0]):.2f} units")

    st.info(
        "These are historical evaluation intervals, not the intervals displayed in "
        "Next-day recommendations. The saved full-history Q05/Q95 models have not "
        "been conformal-calibrated against independent later data. Do not apply "
        "fold-specific margins directly to those saved models. Time-series shifts "
        "can weaken nominal coverage."
    )
