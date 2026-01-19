"""SHAP Analysis tab.

Displays individual trade SHAP explanations and global feature importance.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from ml4t.diagnostic.evaluation.trade_dashboard.types import DashboardBundle


def render_tab(st: Any, bundle: DashboardBundle) -> None:
    """Render the SHAP Analysis tab.

    Parameters
    ----------
    st : streamlit
        Streamlit module instance.
    bundle : DashboardBundle
        Normalized dashboard data.
    """
    st.header("SHAP Analysis")

    st.info(
        "Explore SHAP (SHapley Additive exPlanations) values for individual trades "
        "to understand which features drove model predictions."
    )

    explanations = bundle.explanations

    if not explanations:
        st.warning("No trade explanations available.")
        return

    # Check for trade selected from worst trades tab
    selected_from_tab2 = st.session_state.get("selected_trade_for_shap")
    selected_trade_idx = 0

    if selected_from_tab2:
        for i, exp in enumerate(explanations):
            if exp.get("trade_id") == selected_from_tab2:
                selected_trade_idx = i
                break

    # Trade selector
    st.subheader("Trade Selection")

    if selected_from_tab2:
        st.success(f"Currently viewing: **{selected_from_tab2}** (selected in Worst Trades tab)")

    trade_options = [exp.get("trade_id", f"Trade_{i}") for i, exp in enumerate(explanations)]

    selected_trade_idx = st.selectbox(
        "Select trade to view SHAP explanation:",
        range(len(trade_options)),
        index=selected_trade_idx,
        format_func=lambda x: trade_options[x],
    )

    if selected_trade_idx is not None:
        _render_trade_shap(st, explanations[selected_trade_idx])

    # Global feature importance
    st.divider()
    _render_global_importance(st, explanations)


def _render_trade_shap(st: Any, explanation: dict[str, Any]) -> None:
    """Render SHAP explanation for a single trade."""
    trade_id = explanation.get("trade_id", "Unknown")
    timestamp = explanation.get("timestamp")
    top_features = explanation.get("top_features", [])

    st.divider()
    st.subheader(f"Trade: {trade_id}")
    if timestamp:
        st.caption(f"Timestamp: {timestamp}")

    # Note: Renamed from "Waterfall" - this is actually a bar chart
    st.subheader("Top SHAP Contributions")

    if not top_features:
        st.warning("No SHAP features available for this trade.")
        return

    # Prepare data for visualization
    features_data = []
    cumulative = 0.0

    for item in top_features[:15]:
        if len(item) >= 2:
            feature, shap_val = item[0], item[1]
            cumulative += shap_val
            features_data.append(
                {
                    "Feature": feature,
                    "SHAP Value": shap_val,
                    "Cumulative": cumulative,
                    "Impact": "Positive" if shap_val > 0 else "Negative",
                }
            )

    if not features_data:
        st.warning("Could not parse SHAP features.")
        return

    df_shap = pd.DataFrame(features_data)

    # Create bar chart
    import plotly.graph_objects as go

    colors = ["#FF6B6B" if val < 0 else "#51CF66" for val in df_shap["SHAP Value"]]

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=df_shap["SHAP Value"],
            y=df_shap["Feature"],
            orientation="h",
            marker={"color": colors},
            text=[f"{val:.4f}" for val in df_shap["SHAP Value"]],
            textposition="auto",
            hovertemplate="<b>%{y}</b><br>SHAP: %{x:.4f}<extra></extra>",
        )
    )

    fig.update_layout(
        title="SHAP Feature Contributions (Top 15 Features)",
        xaxis_title="SHAP Value",
        yaxis_title="Feature",
        height=max(400, len(df_shap) * 30),
        yaxis={"autorange": "reversed"},
        showlegend=False,
    )

    st.plotly_chart(fig, use_container_width=True)

    # Feature values table
    st.subheader("Feature Values")

    display_df = df_shap[["Feature", "SHAP Value", "Impact"]].copy()
    display_df["SHAP Value"] = display_df["SHAP Value"].apply(lambda x: f"{x:.4f}")

    st.dataframe(
        display_df,
        hide_index=True,
        use_container_width=True,
        column_config={
            "Feature": st.column_config.TextColumn("Feature Name", width="medium"),
            "SHAP Value": st.column_config.TextColumn("SHAP Value", width="small"),
            "Impact": st.column_config.TextColumn("Impact", width="small"),
        },
    )

    # Interpretation guide
    with st.expander("How to Interpret SHAP Values"):
        st.markdown(
            """
            **SHAP Value Interpretation:**

            - **Positive SHAP value (green)**: Feature pushed prediction higher
            - **Negative SHAP value (red)**: Feature pushed prediction lower
            - **Magnitude**: Larger absolute values indicate stronger influence

            **For a losing trade:**
            - Large positive values contributed to an incorrect bullish prediction
            - Large negative values contributed to an incorrect bearish prediction

            **Actionable insights:**
            - Identify which features consistently mislead the model
            - Look for patterns across multiple losing trades (see Patterns tab)
            """
        )

    # Summary statistics
    st.divider()
    st.subheader("SHAP Summary Statistics")

    shap_values = [item[1] for item in top_features if len(item) >= 2]

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        total_shap = sum(shap_values)
        st.metric("Total SHAP", f"{total_shap:.4f}")

    with col2:
        positive_shap = sum(v for v in shap_values if v > 0)
        st.metric("Positive Contrib.", f"{positive_shap:.4f}")

    with col3:
        negative_shap = sum(v for v in shap_values if v < 0)
        st.metric("Negative Contrib.", f"{negative_shap:.4f}")

    with col4:
        mean_abs_shap = float(np.mean([abs(v) for v in shap_values])) if shap_values else 0.0
        st.metric("Mean Abs. SHAP", f"{mean_abs_shap:.4f}")


def _render_global_importance(st: Any, explanations: list[dict[str, Any]]) -> None:
    """Render global feature importance across all trades."""
    st.subheader("Global Feature Importance")

    st.markdown(
        "Aggregate SHAP importance across all analyzed trades to identify "
        "which features are most influential overall."
    )

    # Calculate global importance
    all_features: dict[str, list[float]] = {}

    for exp in explanations:
        top_features = exp.get("top_features", [])

        for item in top_features:
            if len(item) >= 2:
                feature, shap_val = item[0], item[1]
                if feature not in all_features:
                    all_features[feature] = []
                all_features[feature].append(abs(shap_val))

    if not all_features:
        st.warning("No feature importance data available.")
        return

    # Calculate mean absolute SHAP for each feature
    feature_importance = [
        {
            "Feature": feature,
            "Mean Abs SHAP": float(np.mean(values)),
            "Frequency": len(values),
            "Total Impact": sum(values),
        }
        for feature, values in all_features.items()
    ]

    # Sort by mean absolute SHAP
    feature_importance.sort(key=lambda x: cast(float, x["Mean Abs SHAP"]), reverse=True)

    # Display top 20
    df_importance = pd.DataFrame(feature_importance[:20])

    # Create bar chart
    import plotly.express as px

    fig = px.bar(
        df_importance,
        x="Mean Abs SHAP",
        y="Feature",
        orientation="h",
        title="Top 20 Most Important Features (Mean Absolute SHAP)",
        color="Mean Abs SHAP",
        color_continuous_scale="Blues",
    )

    fig.update_layout(
        yaxis={"autorange": "reversed"},
        height=600,
    )

    st.plotly_chart(fig, use_container_width=True)

    # Display table
    st.subheader("Feature Importance Table")

    display_importance = df_importance.copy()
    display_importance["Mean Abs SHAP"] = display_importance["Mean Abs SHAP"].apply(
        lambda x: f"{x:.4f}"
    )
    display_importance["Total Impact"] = display_importance["Total Impact"].apply(
        lambda x: f"{x:.4f}"
    )

    st.dataframe(display_importance, hide_index=True, use_container_width=True)
