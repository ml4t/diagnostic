"""Trade-level visualizations for backtest analysis.

Provides interactive Plotly plots for deep trade analysis:
- MFE/MAE scatter plot with exit efficiency
- Exit reason breakdown (sunburst/treemap)
- Trade PnL waterfall
- Duration distribution
- Size vs return analysis
- Consecutive wins/losses

These visualizations exceed QuantStats by providing trade-level insights
rather than just portfolio-level aggregates.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ml4t.diagnostic.visualization.core import (
    create_base_figure,
    get_color_scheme,
    get_theme_config,
    validate_theme,
)

if TYPE_CHECKING:
    import polars as pl


# =============================================================================
# MFE/MAE Analysis
# =============================================================================


def plot_mfe_mae_scatter(
    trades_df: pl.DataFrame,
    *,
    color_by: Literal["pnl", "pnl_pct", "duration", "exit_reason", "direction"] = "pnl",
    size_by: Literal["quantity", "notional", "uniform"] = "uniform",
    show_efficiency_frontier: bool = True,
    show_edge_ratio: bool = True,
    show_quadrants: bool = True,
    mfe_col: str = "mfe",
    mae_col: str = "mae",
    theme: str | None = None,
    height: int = 600,
    width: int | None = None,
) -> go.Figure:
    """Create MFE vs MAE scatter plot with exit efficiency analysis.

    Maximum Favorable Excursion (MFE) shows the best unrealized return
    during each trade. Maximum Adverse Excursion (MAE) shows the worst.
    This plot reveals exit timing efficiency.

    Parameters
    ----------
    trades_df : pl.DataFrame
        Trade data with mfe, mae, pnl columns
    color_by : str, default "pnl"
        Field to use for color encoding
    size_by : str, default "uniform"
        Field to use for marker size
    show_efficiency_frontier : bool, default True
        Show diagonal line where exit equals MFE (perfect efficiency)
    show_edge_ratio : bool, default True
        Show aggregate edge ratio annotation
    show_quadrants : bool, default True
        Show quadrant labels (Q1: winners, Q2-4: losers by type)
    mfe_col : str, default "mfe"
        Column name for MFE
    mae_col : str, default "mae"
        Column name for MAE
    theme : str, optional
        Plot theme
    height : int, default 600
        Figure height
    width : int, optional
        Figure width

    Returns
    -------
    go.Figure
        Interactive scatter plot

    Examples
    --------
    >>> fig = plot_mfe_mae_scatter(trades_df, color_by="exit_reason")
    >>> fig.show()

    Notes
    -----
    Quadrant Interpretation:
    - Q1 (MFE > |MAE|, PnL > 0): Healthy winners with controlled drawdown
    - Q2 (MFE < |MAE|, PnL > 0): Lucky winners that recovered from large drawdown
    - Q3 (MFE < |MAE|, PnL < 0): Losers with insufficient profit opportunity
    - Q4 (MFE > |MAE|, PnL < 0): Poor exit timing - had profit but lost it
    """

    theme = validate_theme(theme)

    # Extract data
    mfe = trades_df[mfe_col].to_numpy()
    mae = np.abs(trades_df[mae_col].to_numpy())  # MAE as positive values
    pnl = trades_df["pnl"].to_numpy() if "pnl" in trades_df.columns else np.zeros(len(mfe))

    # Color encoding
    if color_by == "pnl" and "pnl" in trades_df.columns:
        color_values = pnl
        colorscale = "RdYlGn"
        color_label = "PnL ($)"
    elif color_by == "pnl_pct" and "pnl_pct" in trades_df.columns:
        color_values = trades_df["pnl_pct"].to_numpy()
        colorscale = "RdYlGn"
        color_label = "Return (%)"
    elif color_by == "duration" and "bars_held" in trades_df.columns:
        color_values = trades_df["bars_held"].to_numpy()
        colorscale = "Viridis"
        color_label = "Bars Held"
    elif color_by == "exit_reason" and "exit_reason" in trades_df.columns:
        # Categorical - use discrete colors
        color_values = None
        exit_reasons = trades_df["exit_reason"].to_list()
    elif color_by == "direction" and "direction" in trades_df.columns:
        color_values = None
        directions = trades_df["direction"].to_list()
    else:
        color_values = pnl
        colorscale = "RdYlGn"
        color_label = "PnL ($)"

    # Size encoding
    if size_by == "quantity" and "quantity" in trades_df.columns:
        sizes = np.abs(trades_df["quantity"].to_numpy())
        sizes = 5 + 20 * (sizes - sizes.min()) / (sizes.max() - sizes.min() + 1e-10)
    elif (
        size_by == "notional"
        and "entry_price" in trades_df.columns
        and "quantity" in trades_df.columns
    ):
        notional = np.abs(trades_df["entry_price"].to_numpy() * trades_df["quantity"].to_numpy())
        sizes = 5 + 20 * (notional - notional.min()) / (notional.max() - notional.min() + 1e-10)
    else:
        sizes = 10  # Uniform size

    # Create figure
    fig = create_base_figure(
        title="MFE vs MAE Analysis (Exit Efficiency)",
        xaxis_title="MAE (Max Adverse Excursion) - % Loss from Entry",
        yaxis_title="MFE (Max Favorable Excursion) - % Gain from Entry",
        height=height,
        width=width,
        theme=theme,
    )

    # Hover template
    hover_template = (
        "<b>Trade</b><br>"
        "MFE: %{y:.2%}<br>"
        "MAE: %{x:.2%}<br>"
        "PnL: $%{customdata[0]:.2f}<br>"
        "Return: %{customdata[1]:.2%}<br>"
        "<extra></extra>"
    )

    # Custom data for hover
    custom_data = np.column_stack(
        [
            pnl,
            trades_df["pnl_pct"].to_numpy() / 100
            if "pnl_pct" in trades_df.columns
            else pnl / 10000,
        ]
    )

    # Add scatter trace
    if color_by == "exit_reason" and "exit_reason" in trades_df.columns:
        # Discrete color by exit reason
        unique_reasons = list(set(exit_reasons))
        colors = get_color_scheme("set2")

        for i, reason in enumerate(unique_reasons):
            mask = [r == reason for r in exit_reasons]
            fig.add_trace(
                go.Scatter(
                    x=mae[mask],
                    y=mfe[mask],
                    mode="markers",
                    name=reason,
                    marker={
                        "size": sizes if isinstance(sizes, int) else sizes[mask],
                        "color": colors[i % len(colors)],
                        "opacity": 0.7,
                        "line": {"width": 1, "color": "white"},
                    },
                    customdata=custom_data[mask],
                    hovertemplate=hover_template.replace(
                        "<extra></extra>", f"Exit: {reason}<extra></extra>"
                    ),
                )
            )
    elif color_by == "direction" and "direction" in trades_df.columns:
        # Long vs Short
        for direction in ["long", "short"]:
            mask = [d == direction for d in directions]
            color = "#28A745" if direction == "long" else "#DC3545"
            fig.add_trace(
                go.Scatter(
                    x=mae[mask],
                    y=mfe[mask],
                    mode="markers",
                    name=direction.title(),
                    marker={
                        "size": sizes if isinstance(sizes, int) else sizes[mask],
                        "color": color,
                        "opacity": 0.7,
                        "line": {"width": 1, "color": "white"},
                    },
                    customdata=custom_data[mask],
                    hovertemplate=hover_template,
                )
            )
    else:
        # Continuous color scale
        fig.add_trace(
            go.Scatter(
                x=mae,
                y=mfe,
                mode="markers",
                marker={
                    "size": sizes,
                    "color": color_values,
                    "colorscale": colorscale,
                    "colorbar": {"title": color_label, "thickness": 15},
                    "opacity": 0.7,
                    "line": {"width": 1, "color": "white"},
                },
                customdata=custom_data,
                hovertemplate=hover_template,
                showlegend=False,
            )
        )

    # Add efficiency frontier (diagonal)
    if show_efficiency_frontier:
        max_val = max(mfe.max(), mae.max()) * 1.1
        fig.add_trace(
            go.Scatter(
                x=[0, max_val],
                y=[0, max_val],
                mode="lines",
                name="Perfect Efficiency (Exit at MFE)",
                line={"color": "gray", "dash": "dash", "width": 2},
                hoverinfo="skip",
            )
        )

    # Add quadrant annotations
    if show_quadrants:
        annotations = [
            {
                "x": mae.max() * 0.8,
                "y": mfe.max() * 0.9,
                "text": "Q1: Healthy Winners",
                "color": "#28A745",
            },
            {
                "x": mae.max() * 0.2,
                "y": mfe.max() * 0.9,
                "text": "Q2: Lucky Recovery",
                "color": "#FFC107",
            },
            {
                "x": mae.max() * 0.2,
                "y": mfe.max() * 0.1,
                "text": "Q3: No Opportunity",
                "color": "#DC3545",
            },
            {
                "x": mae.max() * 0.8,
                "y": mfe.max() * 0.1,
                "text": "Q4: Poor Exit",
                "color": "#DC3545",
            },
        ]

        for ann in annotations:
            fig.add_annotation(
                x=ann["x"],
                y=ann["y"],
                text=ann["text"],
                showarrow=False,
                font={"size": 10, "color": ann["color"]},
                opacity=0.7,
            )

    # Add edge ratio annotation
    if show_edge_ratio:
        edge_ratio = np.mean(mfe) / np.mean(mae) if np.mean(mae) > 0 else np.inf
        efficiency = np.mean(pnl[pnl > 0] / mfe[pnl > 0]) if (pnl > 0).sum() > 0 else 0

        fig.add_annotation(
            x=0.02,
            y=0.98,
            xref="paper",
            yref="paper",
            text=f"<b>Edge Ratio:</b> {edge_ratio:.2f}<br><b>Exit Efficiency:</b> {efficiency:.1%}",
            showarrow=False,
            font={"size": 12},
            align="left",
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="gray",
            borderwidth=1,
        )

    fig.update_layout(
        legend={"yanchor": "top", "y": 0.99, "xanchor": "right", "x": 0.99},
    )

    return fig


# =============================================================================
# Exit Reason Analysis
# =============================================================================


def plot_exit_reason_breakdown(
    trades_df: pl.DataFrame,
    *,
    chart_type: Literal["sunburst", "treemap", "bar", "pie"] = "sunburst",
    show_pnl_contribution: bool = True,
    show_win_loss_split: bool = True,
    exit_reason_col: str = "exit_reason",
    pnl_col: str = "pnl",
    theme: str | None = None,
    height: int = 500,
    width: int | None = None,
) -> go.Figure:
    """Create exit reason breakdown visualization.

    Shows distribution of exit reasons and their PnL contribution.

    Parameters
    ----------
    trades_df : pl.DataFrame
        Trade data with exit_reason and pnl columns
    chart_type : str, default "sunburst"
        Type of chart: "sunburst", "treemap", "bar", or "pie"
    show_pnl_contribution : bool, default True
        Show PnL contribution rather than just count
    show_win_loss_split : bool, default True
        Split by winner/loser within each exit reason
    exit_reason_col : str, default "exit_reason"
        Column name for exit reason
    pnl_col : str, default "pnl"
        Column name for PnL
    theme : str, optional
        Plot theme
    height : int, default 500
        Figure height
    width : int, optional
        Figure width

    Returns
    -------
    go.Figure
        Exit reason breakdown chart

    Examples
    --------
    >>> fig = plot_exit_reason_breakdown(trades_df, chart_type="sunburst")
    >>> fig.show()
    """
    import polars as pl

    theme = validate_theme(theme)
    theme_config = get_theme_config(theme)

    # Prepare data
    if show_win_loss_split:
        # Add win/loss classification
        trades_with_outcome = trades_df.with_columns(
            pl.when(pl.col(pnl_col) > 0)
            .then(pl.lit("Winner"))
            .otherwise(pl.lit("Loser"))
            .alias("outcome")
        )

        grouped = trades_with_outcome.group_by([exit_reason_col, "outcome"]).agg(
            [
                pl.count().alias("count"),
                pl.col(pnl_col).sum().alias("total_pnl"),
                pl.col(pnl_col).mean().alias("avg_pnl"),
            ]
        )

        # Build hierarchical data
        labels = ["All Trades"]
        parents = [""]
        values = []
        colors = []

        total_trades = len(trades_df)
        total_pnl = trades_df[pnl_col].sum()

        values.append(total_trades if not show_pnl_contribution else abs(total_pnl))
        colors.append("#6C757D")

        # Add exit reasons
        for reason in grouped[exit_reason_col].unique().to_list():
            reason_data = grouped.filter(pl.col(exit_reason_col) == reason)
            reason_count = reason_data["count"].sum()
            reason_pnl = reason_data["total_pnl"].sum()

            labels.append(reason)
            parents.append("All Trades")
            values.append(reason_count if not show_pnl_contribution else abs(reason_pnl))
            colors.append("#3498DB" if reason_pnl > 0 else "#E74C3C")

            # Add win/loss under each reason
            for outcome in ["Winner", "Loser"]:
                outcome_data = reason_data.filter(pl.col("outcome") == outcome)
                if len(outcome_data) > 0:
                    outcome_count = outcome_data["count"].sum()
                    outcome_pnl = outcome_data["total_pnl"].sum()

                    labels.append(f"{reason} - {outcome}")
                    parents.append(reason)
                    values.append(outcome_count if not show_pnl_contribution else abs(outcome_pnl))
                    colors.append("#28A745" if outcome == "Winner" else "#DC3545")
    else:
        # Simple grouping
        grouped = trades_df.group_by(exit_reason_col).agg(
            [
                pl.count().alias("count"),
                pl.col(pnl_col).sum().alias("total_pnl"),
            ]
        )

        labels = grouped[exit_reason_col].to_list()
        values = (
            grouped["count"].to_list()
            if not show_pnl_contribution
            else [abs(p) for p in grouped["total_pnl"].to_list()]
        )
        colors = ["#28A745" if p > 0 else "#DC3545" for p in grouped["total_pnl"].to_list()]
        parents = None

    # Create chart
    if chart_type == "sunburst" and show_win_loss_split:
        fig = go.Figure(
            go.Sunburst(
                labels=labels,
                parents=parents,
                values=values,
                marker={"colors": colors},
                branchvalues="total",
                hovertemplate="<b>%{label}</b><br>Count: %{value}<extra></extra>",
            )
        )
    elif chart_type == "treemap" and show_win_loss_split:
        fig = go.Figure(
            go.Treemap(
                labels=labels,
                parents=parents,
                values=values,
                marker={"colors": colors},
                branchvalues="total",
                hovertemplate="<b>%{label}</b><br>Count: %{value}<extra></extra>",
            )
        )
    elif chart_type == "pie":
        fig = go.Figure(
            go.Pie(
                labels=labels if not show_win_loss_split else grouped[exit_reason_col].to_list(),
                values=values if not show_win_loss_split else grouped["count"].to_list(),
                marker={"colors": colors if not show_win_loss_split else None},
                hole=0.4,
                hovertemplate="<b>%{label}</b><br>Count: %{value}<br>%{percent}<extra></extra>",
            )
        )
    else:  # bar
        exit_reasons = grouped[exit_reason_col].to_list()
        counts = grouped["count"].to_list()
        pnls = grouped["total_pnl"].to_list()
        bar_colors = ["#28A745" if p > 0 else "#DC3545" for p in pnls]

        fig = go.Figure()

        fig.add_trace(
            go.Bar(
                x=exit_reasons,
                y=counts,
                marker_color=bar_colors,
                text=[f"${p:,.0f}" for p in pnls],
                textposition="outside",
                hovertemplate="<b>%{x}</b><br>Count: %{y}<br>Total PnL: %{text}<extra></extra>",
            )
        )

        fig.update_layout(
            xaxis_title="Exit Reason",
            yaxis_title="Number of Trades",
        )

    value_type = "PnL Contribution" if show_pnl_contribution else "Trade Count"
    fig.update_layout(
        title=f"Exit Reason Breakdown ({value_type})",
        height=height,
        width=width,
        **{k: v for k, v in theme_config["layout"].items() if k != "margin"},
    )

    return fig


# =============================================================================
# Trade Waterfall
# =============================================================================


def plot_trade_waterfall(
    trades_df: pl.DataFrame,
    *,
    n_trades: int | None = None,
    sort_by: Literal["time", "pnl", "abs_pnl"] = "time",
    show_cumulative_line: bool = True,
    group_by_day: bool = False,
    initial_equity: float = 100000.0,
    pnl_col: str = "pnl",
    time_col: str = "exit_time",
    theme: str | None = None,
    height: int = 500,
    width: int | None = None,
) -> go.Figure:
    """Create trade-by-trade PnL waterfall chart.

    Shows each trade's contribution to cumulative PnL.

    Parameters
    ----------
    trades_df : pl.DataFrame
        Trade data with pnl column
    n_trades : int, optional
        Limit to last N trades. None for all.
    sort_by : str, default "time"
        How to order trades: "time", "pnl", or "abs_pnl"
    show_cumulative_line : bool, default True
        Overlay cumulative PnL line
    group_by_day : bool, default False
        Aggregate by day (useful for high-frequency strategies)
    initial_equity : float, default 100000.0
        Starting equity for cumulative calculation
    pnl_col : str, default "pnl"
        Column name for PnL
    time_col : str, default "exit_time"
        Column name for trade time
    theme : str, optional
        Plot theme
    height : int, default 500
        Figure height
    width : int, optional
        Figure width

    Returns
    -------
    go.Figure
        Waterfall chart of trade PnL

    Examples
    --------
    >>> fig = plot_trade_waterfall(trades_df, n_trades=50, show_cumulative_line=True)
    >>> fig.show()
    """
    import polars as pl

    theme = validate_theme(theme)
    theme_config = get_theme_config(theme)

    # Sort and limit
    if sort_by == "time" and time_col in trades_df.columns:
        trades = trades_df.sort(time_col)
    elif sort_by == "pnl":
        trades = trades_df.sort(pnl_col, descending=True)
    elif sort_by == "abs_pnl":
        trades = trades_df.with_columns(pl.col(pnl_col).abs().alias("_abs_pnl")).sort(
            "_abs_pnl", descending=True
        )
    else:
        trades = trades_df

    if n_trades is not None:
        trades = trades.tail(n_trades)

    # Group by day if requested
    if group_by_day and time_col in trades.columns:
        trades = (
            trades.with_columns(pl.col(time_col).dt.date().alias("date"))
            .group_by("date")
            .agg(
                [
                    pl.col(pnl_col).sum().alias(pnl_col),
                    pl.count().alias("n_trades"),
                ]
            )
            .sort("date")
        )
        x_labels = [str(d) for d in trades["date"].to_list()]
        hover_extra = "<br>Trades: %{customdata[1]}"
        custom_data = np.column_stack(
            [
                trades[pnl_col].to_numpy(),
                trades["n_trades"].to_numpy(),
            ]
        )
    else:
        x_labels = [f"Trade {i + 1}" for i in range(len(trades))]
        hover_extra = ""
        custom_data = trades[pnl_col].to_numpy().reshape(-1, 1)

    pnl_values = trades[pnl_col].to_numpy()
    cumulative = np.cumsum(pnl_values)

    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Waterfall bars
    colors = ["#28A745" if p > 0 else "#DC3545" for p in pnl_values]

    fig.add_trace(
        go.Bar(
            x=x_labels,
            y=pnl_values,
            marker_color=colors,
            name="Trade PnL",
            hovertemplate=f"<b>%{{x}}</b><br>PnL: $%{{y:,.2f}}{hover_extra}<extra></extra>",
            customdata=custom_data,
        ),
        secondary_y=False,
    )

    # Cumulative line
    if show_cumulative_line:
        fig.add_trace(
            go.Scatter(
                x=x_labels,
                y=initial_equity + cumulative,
                mode="lines+markers",
                name="Cumulative Equity",
                line={"color": "#2E86AB", "width": 2},
                marker={"size": 4},
                hovertemplate="<b>%{x}</b><br>Equity: $%{y:,.2f}<extra></extra>",
            ),
            secondary_y=True,
        )

    # Add zero line
    fig.add_hline(y=0, line_dash="solid", line_color="gray", line_width=1, secondary_y=False)

    # Update layout
    fig.update_layout(
        title="Trade PnL Waterfall",
        height=height,
        width=width,
        legend={"yanchor": "top", "y": 0.99, "xanchor": "left", "x": 0.01},
        **{k: v for k, v in theme_config["layout"].items() if k != "margin"},
    )

    fig.update_yaxes(title_text="Trade PnL ($)", secondary_y=False)
    fig.update_yaxes(title_text="Cumulative Equity ($)", secondary_y=True)
    fig.update_xaxes(title_text="Trade" if not group_by_day else "Date")

    # Rotate x labels if many trades
    if len(x_labels) > 20:
        fig.update_xaxes(tickangle=45)

    return fig


# =============================================================================
# Duration Distribution
# =============================================================================


def plot_trade_duration_distribution(
    trades_df: pl.DataFrame,
    *,
    duration_col: str = "bars_held",
    split_by: Literal["outcome", "exit_reason", "direction", "none"] = "outcome",
    pnl_col: str = "pnl",
    bin_count: int = 30,
    show_statistics: bool = True,
    theme: str | None = None,
    height: int = 450,
    width: int | None = None,
) -> go.Figure:
    """Plot distribution of trade holding periods.

    Parameters
    ----------
    trades_df : pl.DataFrame
        Trade data with duration column
    duration_col : str, default "bars_held"
        Column name for holding period
    split_by : str, default "outcome"
        How to split distribution: "outcome", "exit_reason", "direction", or "none"
    pnl_col : str, default "pnl"
        Column name for PnL (used for outcome split)
    bin_count : int, default 30
        Number of histogram bins
    show_statistics : bool, default True
        Show mean/median annotations
    theme : str, optional
        Plot theme
    height : int, default 450
        Figure height
    width : int, optional
        Figure width

    Returns
    -------
    go.Figure
        Duration distribution histogram

    Examples
    --------
    >>> fig = plot_trade_duration_distribution(trades_df, split_by="outcome")
    >>> fig.show()
    """

    theme = validate_theme(theme)
    theme_config = get_theme_config(theme)

    fig = create_base_figure(
        title="Trade Duration Distribution",
        xaxis_title="Holding Period (bars)",
        yaxis_title="Number of Trades",
        height=height,
        width=width,
        theme=theme,
    )

    durations = trades_df[duration_col].to_numpy()

    if split_by == "outcome" and pnl_col in trades_df.columns:
        winners = durations[trades_df[pnl_col].to_numpy() > 0]
        losers = durations[trades_df[pnl_col].to_numpy() <= 0]

        fig.add_trace(
            go.Histogram(
                x=winners,
                name="Winners",
                marker_color="#28A745",
                opacity=0.7,
                nbinsx=bin_count,
            )
        )
        fig.add_trace(
            go.Histogram(
                x=losers,
                name="Losers",
                marker_color="#DC3545",
                opacity=0.7,
                nbinsx=bin_count,
            )
        )
        fig.update_layout(barmode="overlay")

    elif split_by == "exit_reason" and "exit_reason" in trades_df.columns:
        exit_reasons = trades_df["exit_reason"].unique().to_list()
        colors = get_color_scheme("set2")

        for i, reason in enumerate(exit_reasons):
            mask = trades_df["exit_reason"].to_numpy() == reason
            fig.add_trace(
                go.Histogram(
                    x=durations[mask],
                    name=reason,
                    marker_color=colors[i % len(colors)],
                    opacity=0.7,
                    nbinsx=bin_count,
                )
            )
        fig.update_layout(barmode="stack")

    elif split_by == "direction" and "direction" in trades_df.columns:
        for direction in ["long", "short"]:
            mask = trades_df["direction"].to_numpy() == direction
            color = "#28A745" if direction == "long" else "#DC3545"
            fig.add_trace(
                go.Histogram(
                    x=durations[mask],
                    name=direction.title(),
                    marker_color=color,
                    opacity=0.7,
                    nbinsx=bin_count,
                )
            )
        fig.update_layout(barmode="overlay")

    else:
        fig.add_trace(
            go.Histogram(
                x=durations,
                name="All Trades",
                marker_color=theme_config["colorway"][0],
                opacity=0.7,
                nbinsx=bin_count,
            )
        )

    # Add statistics
    if show_statistics:
        mean_dur = np.mean(durations)
        median_dur = np.median(durations)

        fig.add_vline(
            x=mean_dur,
            line_dash="dash",
            line_color="#2E86AB",
            annotation_text=f"Mean: {mean_dur:.1f}",
            annotation_position="top",
        )
        fig.add_vline(
            x=median_dur,
            line_dash="dot",
            line_color="#E74C3C",
            annotation_text=f"Median: {median_dur:.1f}",
            annotation_position="bottom",
        )

    fig.update_layout(
        legend={"yanchor": "top", "y": 0.99, "xanchor": "right", "x": 0.99},
    )

    return fig


# =============================================================================
# Size vs Return Analysis
# =============================================================================


def plot_trade_size_vs_return(
    trades_df: pl.DataFrame,
    *,
    size_metric: Literal["quantity", "notional", "risk_amount"] = "notional",
    return_metric: Literal["pnl", "pnl_pct"] = "pnl_pct",
    show_regression: bool = True,
    show_correlation: bool = True,
    color_by: Literal["outcome", "exit_reason", "none"] = "outcome",
    theme: str | None = None,
    height: int = 500,
    width: int | None = None,
) -> go.Figure:
    """Analyze relationship between position size and returns.

    Useful for detecting if larger positions perform differently.

    Parameters
    ----------
    trades_df : pl.DataFrame
        Trade data
    size_metric : str, default "notional"
        Size measure: "quantity", "notional", or "risk_amount"
    return_metric : str, default "pnl_pct"
        Return measure: "pnl" or "pnl_pct"
    show_regression : bool, default True
        Show regression line
    show_correlation : bool, default True
        Show correlation annotation
    color_by : str, default "outcome"
        Color points by: "outcome", "exit_reason", or "none"
    theme : str, optional
        Plot theme
    height : int, default 500
        Figure height
    width : int, optional
        Figure width

    Returns
    -------
    go.Figure
        Size vs return scatter plot
    """

    theme = validate_theme(theme)
    theme_config = get_theme_config(theme)

    # Calculate size metric
    if size_metric == "quantity" and "quantity" in trades_df.columns:
        sizes = np.abs(trades_df["quantity"].to_numpy())
        x_label = "Position Size (units)"
    elif (
        size_metric == "notional"
        and "entry_price" in trades_df.columns
        and "quantity" in trades_df.columns
    ):
        sizes = np.abs(trades_df["entry_price"].to_numpy() * trades_df["quantity"].to_numpy())
        x_label = "Notional Value ($)"
    elif size_metric == "risk_amount" and "entry_price" in trades_df.columns:
        sizes = np.abs(trades_df["entry_price"].to_numpy() * trades_df["quantity"].to_numpy())
        x_label = "Risk Amount ($)"
    else:
        sizes = (
            np.abs(trades_df["quantity"].to_numpy())
            if "quantity" in trades_df.columns
            else np.ones(len(trades_df))
        )
        x_label = "Position Size"

    # Get returns
    if return_metric == "pnl_pct" and "pnl_pct" in trades_df.columns:
        returns = trades_df["pnl_pct"].to_numpy()
        y_label = "Return (%)"
    else:
        returns = trades_df["pnl"].to_numpy()
        y_label = "PnL ($)"

    fig = create_base_figure(
        title="Position Size vs Return",
        xaxis_title=x_label,
        yaxis_title=y_label,
        height=height,
        width=width,
        theme=theme,
    )

    # Color by outcome or exit reason
    if color_by == "outcome" and "pnl" in trades_df.columns:
        winners = trades_df["pnl"].to_numpy() > 0

        fig.add_trace(
            go.Scatter(
                x=sizes[winners],
                y=returns[winners],
                mode="markers",
                name="Winners",
                marker={"color": "#28A745", "size": 8, "opacity": 0.6},
            )
        )
        fig.add_trace(
            go.Scatter(
                x=sizes[~winners],
                y=returns[~winners],
                mode="markers",
                name="Losers",
                marker={"color": "#DC3545", "size": 8, "opacity": 0.6},
            )
        )
    elif color_by == "exit_reason" and "exit_reason" in trades_df.columns:
        exit_reasons = trades_df["exit_reason"].unique().to_list()
        colors = get_color_scheme("set2")

        for i, reason in enumerate(exit_reasons):
            mask = trades_df["exit_reason"].to_numpy() == reason
            fig.add_trace(
                go.Scatter(
                    x=sizes[mask],
                    y=returns[mask],
                    mode="markers",
                    name=reason,
                    marker={"color": colors[i % len(colors)], "size": 8, "opacity": 0.6},
                )
            )
    else:
        fig.add_trace(
            go.Scatter(
                x=sizes,
                y=returns,
                mode="markers",
                name="Trades",
                marker={"color": theme_config["colorway"][0], "size": 8, "opacity": 0.6},
            )
        )

    # Add regression line
    if show_regression:
        from scipy import stats

        # Filter NaN values
        valid = np.isfinite(sizes) & np.isfinite(returns)
        if valid.sum() > 2:
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                sizes[valid], returns[valid]
            )

            x_line = np.array([sizes[valid].min(), sizes[valid].max()])
            y_line = slope * x_line + intercept

            fig.add_trace(
                go.Scatter(
                    x=x_line,
                    y=y_line,
                    mode="lines",
                    name=f"Regression (R²={r_value**2:.3f})",
                    line={"color": "gray", "dash": "dash", "width": 2},
                )
            )

    # Add correlation annotation
    if show_correlation:
        valid = np.isfinite(sizes) & np.isfinite(returns)
        if valid.sum() > 2:
            corr = np.corrcoef(sizes[valid], returns[valid])[0, 1]

            fig.add_annotation(
                x=0.02,
                y=0.98,
                xref="paper",
                yref="paper",
                text=f"<b>Correlation:</b> {corr:.3f}",
                showarrow=False,
                font={"size": 12},
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="gray",
                borderwidth=1,
            )

    # Add zero line
    fig.add_hline(y=0, line_dash="solid", line_color="gray", line_width=1)

    fig.update_layout(
        legend={"yanchor": "top", "y": 0.99, "xanchor": "right", "x": 0.99},
    )

    return fig


# =============================================================================
# Consecutive Wins/Losses Analysis
# =============================================================================


def plot_consecutive_analysis(
    trades_df: pl.DataFrame,
    *,
    metric: Literal["wins", "losses", "pnl"] = "wins",
    pnl_col: str = "pnl",
    theme: str | None = None,
    height: int = 450,
    width: int | None = None,
) -> go.Figure:
    """Analyze consecutive wins/losses and streaks.

    Parameters
    ----------
    trades_df : pl.DataFrame
        Trade data with pnl column
    metric : str, default "wins"
        What to analyze: "wins", "losses", or "pnl" (for cumulative)
    pnl_col : str, default "pnl"
        Column name for PnL
    theme : str, optional
        Plot theme
    height : int, default 450
        Figure height
    width : int, optional
        Figure width

    Returns
    -------
    go.Figure
        Streak analysis visualization
    """
    theme = validate_theme(theme)
    theme_config = get_theme_config(theme)

    pnl = trades_df[pnl_col].to_numpy()
    is_win = pnl > 0

    # Calculate streaks
    streaks = []
    current_streak = 0
    current_type = None

    for win in is_win:
        if current_type is None:
            current_type = win
            current_streak = 1
        elif win == current_type:
            current_streak += 1
        else:
            streaks.append((current_type, current_streak))
            current_type = win
            current_streak = 1

    if current_streak > 0:
        streaks.append((current_type, current_streak))

    win_streaks = [s[1] for s in streaks if s[0]]
    loss_streaks = [s[1] for s in streaks if not s[0]]

    # Create subplot
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Win Streak Distribution", "Loss Streak Distribution"),
    )

    # Win streaks histogram
    if win_streaks:
        fig.add_trace(
            go.Histogram(
                x=win_streaks,
                name="Win Streaks",
                marker_color="#28A745",
                opacity=0.7,
                nbinsx=max(win_streaks) if win_streaks else 10,
            ),
            row=1,
            col=1,
        )

    # Loss streaks histogram
    if loss_streaks:
        fig.add_trace(
            go.Histogram(
                x=loss_streaks,
                name="Loss Streaks",
                marker_color="#DC3545",
                opacity=0.7,
                nbinsx=max(loss_streaks) if loss_streaks else 10,
            ),
            row=1,
            col=2,
        )

    # Add statistics annotation
    max_win_streak = max(win_streaks) if win_streaks else 0
    max_loss_streak = max(loss_streaks) if loss_streaks else 0
    avg_win_streak = np.mean(win_streaks) if win_streaks else 0
    avg_loss_streak = np.mean(loss_streaks) if loss_streaks else 0

    fig.add_annotation(
        x=0.25,
        y=1.15,
        xref="paper",
        yref="paper",
        text=f"Max: {max_win_streak} | Avg: {avg_win_streak:.1f}",
        showarrow=False,
        font={"size": 11, "color": "#28A745"},
    )

    fig.add_annotation(
        x=0.75,
        y=1.15,
        xref="paper",
        yref="paper",
        text=f"Max: {max_loss_streak} | Avg: {avg_loss_streak:.1f}",
        showarrow=False,
        font={"size": 11, "color": "#DC3545"},
    )

    fig.update_layout(
        title="Consecutive Trade Streak Analysis",
        height=height,
        width=width,
        showlegend=False,
        **{k: v for k, v in theme_config["layout"].items() if k != "margin"},
    )

    fig.update_xaxes(title_text="Streak Length", row=1, col=1)
    fig.update_xaxes(title_text="Streak Length", row=1, col=2)
    fig.update_yaxes(title_text="Frequency", row=1, col=1)
    fig.update_yaxes(title_text="Frequency", row=1, col=2)

    return fig
