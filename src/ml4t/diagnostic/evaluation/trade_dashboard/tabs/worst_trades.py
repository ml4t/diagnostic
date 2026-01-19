"""Worst Trades tab.

Displays a table of trades with sorting/filtering and detailed view.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    from ml4t.diagnostic.evaluation.trade_dashboard.types import DashboardBundle


def render_tab(st: Any, bundle: DashboardBundle) -> None:
    """Render the Worst Trades tab.

    Parameters
    ----------
    st : streamlit
        Streamlit module instance.
    bundle : DashboardBundle
        Normalized dashboard data.
    """
    st.header("Worst Trades Analysis")

    st.info(
        "This tab shows the trades analyzed for error patterns. "
        "Select a trade to see detailed SHAP explanations."
    )

    trades_df = bundle.trades_df

    if trades_df.empty:
        st.warning("No trade data available.")
        return

    # Sidebar filters
    with st.sidebar:
        st.divider()
        st.subheader("Trade Filters")

        # Sort options
        sort_options = ["PnL (Low to High)", "PnL (High to Low)", "Entry Time", "Return %"]
        sort_by = st.selectbox("Sort by", options=sort_options, index=0)

        # Max trades slider
        max_trades = st.slider("Max trades to display", min_value=5, max_value=100, value=20)

    # Apply sorting
    sorted_df = trades_df.copy()

    if sort_by == "PnL (Low to High)" and "pnl" in sorted_df.columns:
        sorted_df = sorted_df.sort_values("pnl", ascending=True, na_position="last")
    elif sort_by == "PnL (High to Low)" and "pnl" in sorted_df.columns:
        sorted_df = sorted_df.sort_values("pnl", ascending=False, na_position="last")
    elif sort_by == "Entry Time" and "entry_time" in sorted_df.columns:
        sorted_df = sorted_df.sort_values("entry_time", ascending=True, na_position="last")
    elif sort_by == "Return %" and "return_pct" in sorted_df.columns:
        sorted_df = sorted_df.sort_values("return_pct", ascending=True, na_position="last")

    # Limit display
    sorted_df = sorted_df.head(max_trades)

    # Build display DataFrame
    display_columns = {
        "trade_id": "Trade ID",
        "symbol": "Symbol",
        "entry_time": "Entry Time",
        "pnl": "PnL",
        "return_pct": "Return %",
        "duration_days": "Duration (days)",
        "top_feature": "Top Feature",
        "top_shap_value": "Top SHAP",
    }

    display_df = sorted_df[[c for c in display_columns if c in sorted_df.columns]].copy()
    display_df = display_df.rename(
        columns={k: v for k, v in display_columns.items() if k in display_df.columns}
    )

    # Format timestamp for display
    if "Entry Time" in display_df.columns:
        display_df["Entry Time"] = display_df["Entry Time"].apply(
            lambda x: x.strftime("%Y-%m-%d %H:%M") if pd.notna(x) else "N/A"
        )

    # Configure column formatting
    column_config = {
        "Trade ID": st.column_config.TextColumn("Trade ID", width="medium"),
        "Symbol": st.column_config.TextColumn("Symbol", width="small"),
        "Entry Time": st.column_config.TextColumn("Entry Time", width="medium"),
        "PnL": st.column_config.NumberColumn(
            "PnL",
            format="%.2f",
            help="Profit/Loss for this trade",
        ),
        "Return %": st.column_config.NumberColumn(
            "Return %",
            format="%.2f%%",
            help="Return as percentage",
        ),
        "Duration (days)": st.column_config.NumberColumn(
            "Duration (days)",
            format="%.1f",
            help="Trade duration in days",
        ),
        "Top Feature": st.column_config.TextColumn(
            "Top Feature",
            help="Feature with highest absolute SHAP value",
        ),
        "Top SHAP": st.column_config.NumberColumn(
            "Top SHAP",
            format="%.4f",
            help="SHAP value for top feature",
        ),
    }

    # Display table with selection
    st.subheader("Trade Table")

    # Initialize session state for selected trade
    if "selected_trade_idx" not in st.session_state:
        st.session_state.selected_trade_idx = None

    # Use dataframe with on_select callback
    event = st.dataframe(
        display_df,
        hide_index=True,
        use_container_width=True,
        column_config={k: v for k, v in column_config.items() if k in display_df.columns},
        on_select="rerun",
        selection_mode="single-row",
    )

    # Handle row selection
    selection = getattr(event, "selection", None)
    if selection is not None:
        rows = getattr(selection, "rows", [])
        if rows:
            st.session_state.selected_trade_idx = rows[0]

    # Display trade details if selected
    if (
        st.session_state.selected_trade_idx is not None
        and st.session_state.selected_trade_idx < len(sorted_df)
    ):
        _render_trade_details(st, sorted_df, bundle, st.session_state.selected_trade_idx)


def _render_trade_details(
    st: Any,
    sorted_df: pd.DataFrame,
    bundle: DashboardBundle,
    selected_idx: int,
) -> None:
    """Render detailed view of selected trade."""
    st.divider()
    st.subheader("Trade Details")

    row = sorted_df.iloc[selected_idx]
    trade_id = row.get("trade_id", "")

    # Find corresponding explanation
    explanation = next(
        (exp for exp in bundle.explanations if exp.get("trade_id") == trade_id),
        None,
    )

    # Basic metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Trade ID", trade_id)
        if pd.notna(row.get("symbol")):
            st.metric("Symbol", row["symbol"])

    with col2:
        pnl = row.get("pnl")
        if pd.notna(pnl):
            st.metric("PnL", f"${pnl:.2f}")
        else:
            st.metric("PnL", "N/A")

    with col3:
        return_pct = row.get("return_pct")
        if pd.notna(return_pct):
            st.metric("Return", f"{return_pct:.2f}%")
        else:
            st.metric("Return", "N/A")

    with col4:
        duration = row.get("duration_days")
        if pd.notna(duration):
            st.metric("Duration", f"{duration:.1f} days")
        else:
            st.metric("Duration", "N/A")

    # Entry/Exit prices
    col1, col2 = st.columns(2)

    with col1:
        entry_price = row.get("entry_price")
        if pd.notna(entry_price):
            st.metric("Entry Price", f"${entry_price:.4f}")
        else:
            st.caption("Entry price not available")

    with col2:
        exit_price = row.get("exit_price")
        if pd.notna(exit_price):
            st.metric("Exit Price", f"${exit_price:.4f}")
        else:
            st.caption("Exit price not available")

    # Top features from explanation
    if explanation and explanation.get("top_features"):
        st.subheader("Top SHAP Contributions")

        top_features = explanation["top_features"]
        feature_data = [
            {"Feature": f[0], "SHAP Value": f[1]}
            for f in top_features[:10]  # Limit to top 10
        ]

        if feature_data:
            st.dataframe(
                pd.DataFrame(feature_data),
                hide_index=True,
                use_container_width=True,
                column_config={
                    "Feature": st.column_config.TextColumn("Feature"),
                    "SHAP Value": st.column_config.NumberColumn("SHAP Value", format="%.4f"),
                },
            )
