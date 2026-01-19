"""Error Patterns tab.

Displays clustered error patterns with hypotheses and recommended actions.
Applies Benjamini-Hochberg FDR correction to feature significance.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    from ml4t.diagnostic.evaluation.trade_dashboard.types import DashboardBundle


def render_tab(st: Any, bundle: DashboardBundle) -> None:
    """Render the Error Patterns tab.

    Parameters
    ----------
    st : streamlit
        Streamlit module instance.
    bundle : DashboardBundle
        Normalized dashboard data.
    """
    st.header("Error Patterns & Recommendations")

    st.info(
        "Error patterns are identified by clustering trades with similar "
        "SHAP profiles. Each pattern includes hypotheses and actionable "
        "recommendations for improvement."
    )

    patterns_df = bundle.patterns_df

    if patterns_df.empty:
        st.warning("No error patterns found. Ensure clustering was performed during analysis.")
        return

    patterns = patterns_df.to_dict("records")

    # Sidebar filters
    with st.sidebar:
        st.divider()
        st.subheader("Pattern Filters")

        sort_by = st.selectbox(
            "Sort patterns by",
            options=[
                "Pattern ID",
                "Number of Trades (Desc)",
                "Confidence (Desc)",
                "Distinctiveness (Desc)",
            ],
            index=1,
        )

        max_n_trades = max(p.get("n_trades", 0) for p in patterns) if patterns else 1
        min_trades = st.slider(
            "Min trades in pattern",
            min_value=1,
            max_value=max(1, max_n_trades),
            value=1,
        )

        show_only_with_hypothesis = st.checkbox(
            "Only patterns with hypotheses",
            value=False,
        )

    # Filter patterns
    filtered_patterns = [
        p
        for p in patterns
        if p.get("n_trades", 0) >= min_trades
        and (not show_only_with_hypothesis or p.get("hypothesis"))
    ]

    # Sort patterns
    if sort_by == "Pattern ID":
        filtered_patterns.sort(key=lambda p: p.get("cluster_id", 0))
    elif sort_by == "Number of Trades (Desc)":
        filtered_patterns.sort(key=lambda p: p.get("n_trades", 0), reverse=True)
    elif sort_by == "Confidence (Desc)":
        filtered_patterns.sort(key=lambda p: p.get("confidence") or 0, reverse=True)
    elif sort_by == "Distinctiveness (Desc)":
        filtered_patterns.sort(key=lambda p: p.get("distinctiveness") or 0, reverse=True)

    # Display count
    if len(filtered_patterns) < len(patterns):
        st.caption(f"Showing {len(filtered_patterns)} of {len(patterns)} patterns")

    # Display patterns
    for i, pattern in enumerate(filtered_patterns):
        _render_pattern_card(st, pattern, expanded=(i == 0))

    # Summary statistics
    st.divider()
    _render_pattern_summary(st, patterns, filtered_patterns)


def _render_pattern_card(st: Any, pattern: dict[str, Any], expanded: bool = False) -> None:
    """Render a single pattern card."""
    cluster_id = pattern.get("cluster_id", 0)
    n_trades = pattern.get("n_trades", 0)
    description = pattern.get("description", "No description")
    hypothesis = pattern.get("hypothesis")
    actions = pattern.get("actions", [])
    confidence = pattern.get("confidence")
    separation_score = pattern.get("separation_score")
    distinctiveness = pattern.get("distinctiveness")
    top_features = pattern.get("top_features", [])

    with st.expander(
        f"**Pattern {cluster_id}**: {n_trades} trades - {description}",
        expanded=expanded,
    ):
        # Metrics row
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Trades", n_trades)

        with col2:
            if confidence is not None:
                st.metric("Confidence", f"{confidence:.1%}")
            else:
                st.metric("Confidence", "N/A")

        with col3:
            if separation_score is not None:
                st.metric(
                    "Separation",
                    f"{separation_score:.2f}",
                    help="Distance to nearest cluster (higher = more distinct)",
                )
            else:
                st.metric("Separation", "N/A")

        with col4:
            if distinctiveness is not None:
                st.metric(
                    "Distinctiveness",
                    f"{distinctiveness:.2f}",
                    help="Uniqueness of SHAP profile (higher = more unique)",
                )
            else:
                st.metric("Distinctiveness", "N/A")

        st.divider()

        # Hypothesis
        st.markdown("### Hypothesis")
        if hypothesis:
            st.markdown(f"> {hypothesis}")
        else:
            st.markdown("*No hypothesis generated for this pattern.*")

        st.divider()

        # Actions
        st.markdown("### Recommended Actions")
        if actions:
            for idx, action in enumerate(actions, 1):
                st.markdown(f"{idx}. {action}")
        else:
            st.markdown("*No actions suggested for this pattern.*")

        # Top features with FDR correction
        if top_features:
            st.divider()
            _render_pattern_features(st, top_features)


def _render_pattern_features(st: Any, top_features: list[Any]) -> None:
    """Render pattern features with BH-FDR correction."""
    st.markdown("### Key Features")

    # Apply BH-FDR correction to p-values
    features_with_fdr = _apply_fdr_correction(top_features)

    feature_names = [f["feature"] for f in features_with_fdr[:10]]
    feature_shap = [f["shap_value"] for f in features_with_fdr[:10]]
    feature_sig = [f["significant_fdr"] for f in features_with_fdr[:10]]

    # Create visualization
    import plotly.graph_objects as go

    colors = ["#51CF66" if sig else "#ADB5BD" for sig in feature_sig]

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=feature_shap,
            y=feature_names,
            orientation="h",
            marker={"color": colors},
            text=[f"{val:.4f}" for val in feature_shap],
            textposition="auto",
            hovertemplate="<b>%{y}</b><br>Mean SHAP: %{x:.4f}<extra></extra>",
        )
    )

    fig.update_layout(
        title="Top Features for This Pattern (FDR-corrected significance)",
        xaxis_title="Mean SHAP Value",
        yaxis_title="Feature",
        height=max(300, len(feature_names) * 30),
        yaxis={"autorange": "reversed"},
        showlegend=False,
    )

    st.plotly_chart(fig, use_container_width=True)

    # Features table with FDR info
    features_df = pd.DataFrame(
        [
            {
                "Feature": f["feature"],
                "Mean SHAP": f"{f['shap_value']:.4f}",
                "P-value": f"{f['p_value']:.4f}" if f["p_value"] is not None else "N/A",
                "Adj. P-value (BH)": f"{f['adjusted_p']:.4f}"
                if f["adjusted_p"] is not None
                else "N/A",
                "Significant": "Yes" if f["significant_fdr"] else "No",
            }
            for f in features_with_fdr[:10]
        ]
    )

    st.dataframe(features_df, hide_index=True, use_container_width=True)

    with st.expander("Understanding Feature Significance (FDR-corrected)"):
        st.markdown(
            """
            **Benjamini-Hochberg FDR Correction:**

            Raw p-values are corrected for multiple comparisons using the
            Benjamini-Hochberg procedure, which controls the False Discovery Rate.

            - **P-value**: Raw p-value from statistical test
            - **Adj. P-value (BH)**: FDR-adjusted p-value
            - **Significant**: Yes if adjusted p-value < 0.05

            **Interpretation:**
            - Green bars: Statistically significant after FDR correction
            - Gray bars: Not significant (may be noise)

            Focus on significant features for most reliable insights.
            """
        )


def _apply_fdr_correction(top_features: list[Any]) -> list[dict[str, Any]]:
    """Apply Benjamini-Hochberg FDR correction to feature p-values.

    Parameters
    ----------
    top_features : list
        List of feature tuples. Expected formats:
        - (name, shap_value)
        - (name, shap_value, p_value)
        - (name, shap_value, p_value_t, p_value_mw, significant)

    Returns
    -------
    list[dict]
        Features with FDR-corrected significance.
    """
    from ml4t.diagnostic.evaluation.trade_dashboard.stats import benjamini_hochberg_fdr

    # Parse features into consistent format
    parsed = []
    for item in top_features:
        if len(item) >= 2:
            feature = item[0]
            shap_val = item[1]
            # Use t-test p-value if available (index 2), otherwise None
            p_value = item[2] if len(item) > 2 else None
            parsed.append(
                {
                    "feature": feature,
                    "shap_value": shap_val,
                    "p_value": p_value,
                }
            )

    if not parsed:
        return []

    # Extract p-values for FDR correction
    p_values = [f["p_value"] for f in parsed if f["p_value"] is not None]

    if len(p_values) >= 2:
        # Apply BH-FDR correction
        fdr_result = benjamini_hochberg_fdr(p_values, alpha=0.05)
        adjusted_p_values = fdr_result["adjusted_p_values"]
        rejected = fdr_result["rejected"]

        # Map back to features
        p_idx = 0
        for f in parsed:
            if f["p_value"] is not None:
                f["adjusted_p"] = adjusted_p_values[p_idx]
                f["significant_fdr"] = rejected[p_idx]
                p_idx += 1
            else:
                f["adjusted_p"] = None
                f["significant_fdr"] = False
    else:
        # Not enough p-values for FDR
        for f in parsed:
            f["adjusted_p"] = f["p_value"]
            f["significant_fdr"] = f["p_value"] is not None and f["p_value"] < 0.05

    return parsed


def _render_pattern_summary(
    st: Any,
    all_patterns: list[dict[str, Any]],
    filtered_patterns: list[dict[str, Any]],
) -> None:
    """Render pattern summary statistics."""
    st.subheader("Pattern Summary")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Patterns", len(all_patterns))
        if len(filtered_patterns) < len(all_patterns):
            st.caption(f"({len(filtered_patterns)} shown)")

    with col2:
        patterns_with_hypotheses = sum(1 for p in all_patterns if p.get("hypothesis"))
        st.metric("With Hypotheses", patterns_with_hypotheses)
        st.caption(f"{patterns_with_hypotheses}/{len(all_patterns)} patterns")

    with col3:
        total_trades = sum(p.get("n_trades", 0) for p in all_patterns)
        st.metric("Total Trades", total_trades)

    with col4:
        avg_confidence = [
            float(p["confidence"]) for p in all_patterns if p.get("confidence") is not None
        ]
        if avg_confidence:
            import numpy as np

            st.metric("Avg Confidence", f"{float(np.mean(avg_confidence)):.1%}")
        else:
            st.metric("Avg Confidence", "N/A")
