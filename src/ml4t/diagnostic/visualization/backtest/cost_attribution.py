"""Cost attribution visualizations for backtest analysis.

Provides interactive Plotly visualizations for understanding
the impact of transaction costs on strategy performance.

Key visualizations:
- Cost waterfall (Gross → Commission → Slippage → Net)
- Cost sensitivity analysis (Sharpe degradation as costs increase)
- Cost over time (rolling cost impact)
- Cost by asset (identify high-cost positions)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ml4t.diagnostic.visualization.core import get_theme_config

if TYPE_CHECKING:
    import polars as pl


def plot_cost_waterfall(
    gross_pnl: float,
    commission: float,
    slippage: float,
    net_pnl: float | None = None,
    other_costs: dict[str, float] | None = None,
    title: str = "Cost Attribution Waterfall",
    show_percentages: bool = True,
    theme: str | None = None,
    height: int = 500,
    width: int | None = None,
) -> go.Figure:
    """Create a waterfall chart showing gross-to-net PnL decomposition.

    Visualizes how transaction costs (commission, slippage) erode
    gross trading profits into net returns.

    Parameters
    ----------
    gross_pnl : float
        Gross profit/loss before costs
    commission : float
        Total commission costs (should be positive, will be shown as negative)
    slippage : float
        Total slippage costs (should be positive, will be shown as negative)
    net_pnl : float, optional
        Net PnL after all costs. If not provided, calculated from inputs.
    other_costs : dict[str, float], optional
        Additional cost categories (e.g., {"Financing": 500, "Fees": 200})
    title : str
        Chart title
    show_percentages : bool
        Whether to show cost as percentage of gross
    theme : str, optional
        Theme name (default, dark, print, presentation)
    height : int
        Figure height in pixels
    width : int, optional
        Figure width in pixels

    Returns
    -------
    go.Figure
        Plotly figure with waterfall chart

    Examples
    --------
    >>> fig = plot_cost_waterfall(
    ...     gross_pnl=100000,
    ...     commission=2500,
    ...     slippage=1500,
    ... )
    >>> fig.show()
    """
    theme_config = get_theme_config(theme)

    # Build cost categories
    labels = ["Gross PnL"]
    values = [gross_pnl]
    measures = ["absolute"]

    # Add commission
    labels.append("Commission")
    values.append(-abs(commission))
    measures.append("relative")

    # Add slippage
    labels.append("Slippage")
    values.append(-abs(slippage))
    measures.append("relative")

    # Add other costs if provided
    if other_costs:
        for name, cost in other_costs.items():
            labels.append(name)
            values.append(-abs(cost))
            measures.append("relative")

    # Calculate net PnL
    if net_pnl is None:
        total_costs = commission + slippage
        if other_costs:
            total_costs += sum(other_costs.values())
        net_pnl = gross_pnl - total_costs

    labels.append("Net PnL")
    values.append(net_pnl)
    measures.append("total")

    # Create hover text with percentages
    if show_percentages and gross_pnl != 0:
        text = [f"${gross_pnl:,.0f}"]
        for val in values[1:-1]:
            pct = abs(val) / abs(gross_pnl) * 100
            text.append(f"${val:,.0f} ({pct:.1f}%)")
        text.append(f"${net_pnl:,.0f}")
    else:
        text = [f"${v:,.0f}" for v in values]

    # Determine colors
    colors = theme_config["colorway"]
    increasing_color = colors[0]  # Usually green/blue
    decreasing_color = colors[1] if len(colors) > 1 else "#EF553B"  # Red for costs
    totals_color = colors[2] if len(colors) > 2 else "#636EFA"  # Blue for totals

    fig = go.Figure(
        go.Waterfall(
            name="Cost Attribution",
            orientation="v",
            x=labels,
            y=values,
            measure=measures,
            text=text,
            textposition="outside",
            increasing={"marker": {"color": increasing_color}},
            decreasing={"marker": {"color": decreasing_color}},
            totals={"marker": {"color": totals_color}},
            connector={"line": {"color": "rgba(128, 128, 128, 0.5)", "width": 2}},
        )
    )

    # Build layout
    layout_updates = {
        "title": {"text": title, "font": {"size": 18}},
        "height": height,
        "yaxis": {"title": "PnL ($)", "tickformat": "$,.0f"},
        "showlegend": False,
    }
    if width:
        layout_updates["width"] = width

    # Merge theme layout without overwriting explicit settings
    for key, value in theme_config["layout"].items():
        if key not in layout_updates:
            layout_updates[key] = value

    fig.update_layout(**layout_updates)

    return fig


def plot_cost_sensitivity(
    returns: pl.Series | np.ndarray,
    base_costs_bps: float = 10.0,
    cost_multipliers: list[float] | None = None,
    trades_per_year: int = 252,
    risk_free_rate: float = 0.0,
    title: str = "Cost Sensitivity Analysis",
    show_breakeven: bool = True,
    theme: str | None = None,
    height: int = 500,
    width: int | None = None,
) -> go.Figure:
    """Analyze how Sharpe ratio degrades as transaction costs increase.

    Shows the sensitivity of risk-adjusted returns to transaction costs,
    helping identify the breakeven point where strategy becomes unprofitable.

    Parameters
    ----------
    returns : pl.Series or np.ndarray
        Gross daily returns (before costs)
    base_costs_bps : float
        Base transaction cost in basis points (e.g., 10 = 0.1%)
    cost_multipliers : list[float], optional
        Multipliers to test (default: [0, 0.5, 1, 1.5, 2, 3, 5])
    trades_per_year : int
        Estimated number of trades per year for cost impact
    risk_free_rate : float
        Annual risk-free rate for Sharpe calculation
    title : str
        Chart title
    show_breakeven : bool
        Whether to annotate the breakeven cost level
    theme : str, optional
        Theme name
    height : int
        Figure height in pixels
    width : int, optional
        Figure width in pixels

    Returns
    -------
    go.Figure
        Plotly figure with cost sensitivity chart
    """
    import polars as pl

    theme_config = get_theme_config(theme)

    # Convert to numpy
    if isinstance(returns, pl.Series):
        returns_arr = returns.to_numpy()
    else:
        returns_arr = np.asarray(returns)

    # Default multipliers
    if cost_multipliers is None:
        cost_multipliers = [0, 0.5, 1, 1.5, 2, 3, 5]

    # Calculate metrics at each cost level
    cost_levels = []
    sharpe_values = []
    cagr_values = []

    gross_mean = np.mean(returns_arr)
    gross_std = np.std(returns_arr, ddof=1)

    for mult in cost_multipliers:
        # Cost per trade in decimal
        cost_per_trade = (base_costs_bps * mult) / 10000

        # Estimate daily cost drag (assuming uniform trading)
        daily_cost_drag = cost_per_trade * (trades_per_year / 252)

        # Net returns
        net_mean = gross_mean - daily_cost_drag

        # Calculate Sharpe
        if gross_std > 0:
            sharpe = (net_mean - risk_free_rate / 252) / gross_std * np.sqrt(252)
        else:
            sharpe = 0

        # Calculate CAGR (approximate)
        cagr = (1 + net_mean) ** 252 - 1

        cost_levels.append(base_costs_bps * mult)
        sharpe_values.append(sharpe)
        cagr_values.append(cagr * 100)  # As percentage

    colors = theme_config["colorway"]

    # Create subplot with Sharpe and CAGR
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Sharpe Ratio vs Costs", "CAGR vs Costs"),
        horizontal_spacing=0.12,
    )

    # Sharpe trace
    fig.add_trace(
        go.Scatter(
            x=cost_levels,
            y=sharpe_values,
            mode="lines+markers",
            name="Sharpe Ratio",
            line={"color": colors[0], "width": 3},
            marker={"size": 10},
            hovertemplate="Cost: %{x:.1f} bps<br>Sharpe: %{y:.2f}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    # Add zero line for Sharpe
    fig.add_hline(
        y=0,
        line_dash="dash",
        line_color="gray",
        row=1,
        col=1,
    )

    # CAGR trace
    fig.add_trace(
        go.Scatter(
            x=cost_levels,
            y=cagr_values,
            mode="lines+markers",
            name="CAGR (%)",
            line={"color": colors[1] if len(colors) > 1 else colors[0], "width": 3},
            marker={"size": 10},
            hovertemplate="Cost: %{x:.1f} bps<br>CAGR: %{y:.1f}%<extra></extra>",
        ),
        row=1,
        col=2,
    )

    # Add zero line for CAGR
    fig.add_hline(
        y=0,
        line_dash="dash",
        line_color="gray",
        row=1,
        col=2,
    )

    # Find breakeven point (where Sharpe crosses zero)
    if show_breakeven:
        for i in range(len(sharpe_values) - 1):
            if sharpe_values[i] > 0 and sharpe_values[i + 1] <= 0:
                # Linear interpolation
                breakeven = cost_levels[i] + (
                    (0 - sharpe_values[i])
                    / (sharpe_values[i + 1] - sharpe_values[i])
                    * (cost_levels[i + 1] - cost_levels[i])
                )
                fig.add_vline(
                    x=breakeven,
                    line_dash="dot",
                    line_color="red",
                    annotation_text=f"Breakeven: {breakeven:.1f} bps",
                    annotation_position="top",
                    row=1,
                    col=1,
                )
                break

    # Mark current cost level
    if base_costs_bps in cost_levels:
        idx = cost_levels.index(base_costs_bps)
        fig.add_annotation(
            x=base_costs_bps,
            y=sharpe_values[idx],
            text="Current",
            showarrow=True,
            arrowhead=2,
            row=1,
            col=1,
        )

    # Build layout
    layout_updates = {
        "title": {"text": title, "font": {"size": 18}},
        "height": height,
        "showlegend": False,
        "xaxis": {"title": "Transaction Cost (bps)"},
        "xaxis2": {"title": "Transaction Cost (bps)"},
        "yaxis": {"title": "Sharpe Ratio"},
        "yaxis2": {"title": "CAGR (%)"},
    }
    if width:
        layout_updates["width"] = width

    for key, value in theme_config["layout"].items():
        if key not in layout_updates:
            layout_updates[key] = value

    fig.update_layout(**layout_updates)

    return fig


def plot_cost_over_time(
    dates: pl.Series | np.ndarray,
    gross_returns: pl.Series | np.ndarray,
    net_returns: pl.Series | np.ndarray,
    rolling_window: int = 63,
    title: str = "Cost Impact Over Time",
    theme: str | None = None,
    height: int = 450,
    width: int | None = None,
) -> go.Figure:
    """Visualize how transaction costs impact returns over time.

    Shows the difference between gross and net returns on a rolling basis,
    helping identify periods of high cost impact.

    Parameters
    ----------
    dates : pl.Series or np.ndarray
        Date index
    gross_returns : pl.Series or np.ndarray
        Gross daily returns (before costs)
    net_returns : pl.Series or np.ndarray
        Net daily returns (after costs)
    rolling_window : int
        Rolling window for smoothing (default: 63 = ~3 months)
    title : str
        Chart title
    theme : str, optional
        Theme name
    height : int
        Figure height in pixels
    width : int, optional
        Figure width in pixels

    Returns
    -------
    go.Figure
        Plotly figure with rolling cost impact
    """
    import polars as pl

    theme_config = get_theme_config(theme)
    colors = theme_config["colorway"]

    # Convert to numpy
    if isinstance(dates, pl.Series):
        dates_arr = dates.to_list()
    else:
        dates_arr = list(dates)

    if isinstance(gross_returns, pl.Series):
        gross_arr = gross_returns.to_numpy()
    else:
        gross_arr = np.asarray(gross_returns)

    if isinstance(net_returns, pl.Series):
        net_arr = net_returns.to_numpy()
    else:
        net_arr = np.asarray(net_returns)

    # Calculate cost drag
    cost_drag = gross_arr - net_arr

    # Rolling metrics
    def rolling_mean(arr: np.ndarray, window: int) -> np.ndarray:
        """Simple rolling mean with edge handling."""
        result = np.full(len(arr), np.nan)
        for i in range(window - 1, len(arr)):
            result[i] = np.mean(arr[i - window + 1 : i + 1])
        return result

    rolling_gross = rolling_mean(gross_arr, rolling_window) * 252 * 100
    rolling_net = rolling_mean(net_arr, rolling_window) * 252 * 100
    rolling_cost = rolling_mean(cost_drag, rolling_window) * 252 * 100

    fig = go.Figure()

    # Gross returns
    fig.add_trace(
        go.Scatter(
            x=dates_arr,
            y=rolling_gross,
            name="Gross Returns (ann.)",
            mode="lines",
            line={"color": colors[0], "width": 2},
            hovertemplate="%{x}<br>Gross: %{y:.1f}%<extra></extra>",
        )
    )

    # Net returns
    fig.add_trace(
        go.Scatter(
            x=dates_arr,
            y=rolling_net,
            name="Net Returns (ann.)",
            mode="lines",
            line={"color": colors[1] if len(colors) > 1 else colors[0], "width": 2},
            hovertemplate="%{x}<br>Net: %{y:.1f}%<extra></extra>",
        )
    )

    # Cost drag (as filled area)
    fig.add_trace(
        go.Scatter(
            x=dates_arr,
            y=rolling_cost,
            name="Cost Drag (ann.)",
            mode="lines",
            fill="tozeroy",
            line={"color": "rgba(239, 85, 59, 0.7)", "width": 1},
            fillcolor="rgba(239, 85, 59, 0.3)",
            hovertemplate="%{x}<br>Cost Drag: %{y:.1f}%<extra></extra>",
        )
    )

    # Build layout
    layout_updates = {
        "title": {"text": title, "font": {"size": 18}},
        "height": height,
        "xaxis": {"title": "Date"},
        "yaxis": {"title": "Annualized Return (%)"},
        "legend": {"yanchor": "top", "y": 0.99, "xanchor": "left", "x": 0.01},
        "hovermode": "x unified",
    }
    if width:
        layout_updates["width"] = width

    for key, value in theme_config["layout"].items():
        if key not in layout_updates:
            layout_updates[key] = value

    fig.update_layout(**layout_updates)

    return fig


def plot_cost_by_asset(
    trades: pl.DataFrame,
    top_n: int = 10,
    cost_column: str = "cost",
    symbol_column: str = "symbol",
    sort_by: Literal["total", "per_trade", "percentage"] = "total",
    title: str = "Transaction Costs by Asset",
    theme: str | None = None,
    height: int = 450,
    width: int | None = None,
) -> go.Figure:
    """Show transaction cost breakdown by asset.

    Helps identify which assets incur the highest costs and may need
    different position sizing or execution strategies.

    Parameters
    ----------
    trades : pl.DataFrame
        Trade records with symbol and cost columns
    top_n : int
        Number of top assets to show
    cost_column : str
        Name of the cost column
    symbol_column : str
        Name of the symbol column
    sort_by : {"total", "per_trade", "percentage"}
        How to rank assets:
        - "total": Total cost in dollars
        - "per_trade": Average cost per trade
        - "percentage": Cost as % of gross PnL
    title : str
        Chart title
    theme : str, optional
        Theme name
    height : int
        Figure height in pixels
    width : int, optional
        Figure width in pixels

    Returns
    -------
    go.Figure
        Plotly figure with cost breakdown by asset
    """
    import polars as pl

    theme_config = get_theme_config(theme)
    colors = theme_config["colorway"]

    # Check if required columns exist
    if cost_column not in trades.columns:
        # Try to calculate cost from pnl columns
        if "gross_pnl" in trades.columns and "net_pnl" in trades.columns:
            trades = trades.with_columns((pl.col("gross_pnl") - pl.col("net_pnl")).alias("cost"))
            cost_column = "cost"
        else:
            raise ValueError(f"Cost column '{cost_column}' not found and cannot be calculated")

    if symbol_column not in trades.columns:
        raise ValueError(f"Symbol column '{symbol_column}' not found")

    # Aggregate by symbol
    agg_cols = [
        pl.col(cost_column).sum().alias("total_cost"),
        pl.col(cost_column).mean().alias("avg_cost"),
        pl.col(cost_column).count().alias("n_trades"),
    ]

    if "gross_pnl" in trades.columns:
        agg_cols.append(pl.col("gross_pnl").sum().alias("total_gross"))

    cost_by_symbol = trades.group_by(symbol_column).agg(agg_cols)

    # Calculate percentage if we have gross PnL
    if "total_gross" in cost_by_symbol.columns:
        cost_by_symbol = cost_by_symbol.with_columns(
            (pl.col("total_cost") / pl.col("total_gross").abs() * 100).alias("cost_pct")
        )

    # Sort based on criteria
    if sort_by == "total":
        cost_by_symbol = cost_by_symbol.sort("total_cost", descending=True)
    elif sort_by == "per_trade":
        cost_by_symbol = cost_by_symbol.sort("avg_cost", descending=True)
    elif sort_by == "percentage" and "cost_pct" in cost_by_symbol.columns:
        cost_by_symbol = cost_by_symbol.sort("cost_pct", descending=True)

    # Take top N
    top_assets = cost_by_symbol.head(top_n)

    symbols = top_assets[symbol_column].to_list()
    total_costs = top_assets["total_cost"].to_list()
    n_trades = top_assets["n_trades"].to_list()

    # Determine what to show on secondary axis
    show_pct = "cost_pct" in top_assets.columns and sort_by == "percentage"

    if show_pct:
        secondary_values = top_assets["cost_pct"].to_list()
        secondary_name = "Cost %"
        secondary_format = ".1f"
    else:
        secondary_values = [c / n for c, n in zip(total_costs, n_trades)]
        secondary_name = "Avg/Trade"
        secondary_format = "$,.0f"

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Bar chart for total costs
    fig.add_trace(
        go.Bar(
            x=symbols,
            y=total_costs,
            name="Total Cost",
            marker_color=colors[0],
            hovertemplate="%{x}<br>Total: $%{y:,.0f}<extra></extra>",
        ),
        secondary_y=False,
    )

    # Line for secondary metric
    fig.add_trace(
        go.Scatter(
            x=symbols,
            y=secondary_values,
            name=secondary_name,
            mode="lines+markers",
            line={"color": colors[1] if len(colors) > 1 else "red", "width": 2},
            marker={"size": 8},
            hovertemplate=f"%{{x}}<br>{secondary_name}: %{{y:{secondary_format}}}<extra></extra>",
        ),
        secondary_y=True,
    )

    # Build layout
    layout_updates = {
        "title": {"text": title, "font": {"size": 18}},
        "height": height,
        "xaxis": {"title": "Asset", "tickangle": -45},
        "yaxis": {"title": "Total Cost ($)", "tickformat": "$,.0f"},
        "legend": {"yanchor": "top", "y": 0.99, "xanchor": "right", "x": 0.99},
        "bargap": 0.3,
    }
    if width:
        layout_updates["width"] = width

    for key, value in theme_config["layout"].items():
        if key not in layout_updates:
            layout_updates[key] = value

    fig.update_layout(**layout_updates)

    # Update secondary y-axis
    if show_pct:
        fig.update_yaxes(title_text="Cost (% of Gross)", tickformat=".1f%", secondary_y=True)
    else:
        fig.update_yaxes(title_text="Avg Cost/Trade ($)", tickformat="$,.0f", secondary_y=True)

    return fig


def plot_cost_pie(
    commission: float,
    slippage: float,
    other_costs: dict[str, float] | None = None,
    title: str = "Cost Breakdown",
    theme: str | None = None,
    height: int = 400,
    width: int | None = None,
) -> go.Figure:
    """Create a pie chart showing the breakdown of transaction costs.

    Parameters
    ----------
    commission : float
        Commission costs
    slippage : float
        Slippage costs
    other_costs : dict[str, float], optional
        Additional cost categories
    title : str
        Chart title
    theme : str, optional
        Theme name
    height : int
        Figure height in pixels
    width : int, optional
        Figure width in pixels

    Returns
    -------
    go.Figure
        Plotly pie chart figure
    """
    theme_config = get_theme_config(theme)
    colors = theme_config["colorway"]

    # Build labels and values
    labels = ["Commission", "Slippage"]
    values = [abs(commission), abs(slippage)]

    if other_costs:
        for name, cost in other_costs.items():
            labels.append(name)
            values.append(abs(cost))

    # Calculate percentages for text
    total = sum(values)
    text_info = [f"${v:,.0f}<br>({v / total * 100:.1f}%)" for v in values]

    fig = go.Figure(
        go.Pie(
            labels=labels,
            values=values,
            text=text_info,
            textinfo="text",
            hovertemplate="%{label}<br>$%{value:,.0f}<br>%{percent}<extra></extra>",
            marker={"colors": colors[: len(labels)]},
            hole=0.4,  # Donut chart
        )
    )

    # Add total in center
    fig.add_annotation(
        text=f"Total<br>${total:,.0f}",
        x=0.5,
        y=0.5,
        font={"size": 16},
        showarrow=False,
    )

    # Build layout
    layout_updates = {
        "title": {"text": title, "font": {"size": 18}},
        "height": height,
        "showlegend": True,
        "legend": {
            "orientation": "h",
            "yanchor": "bottom",
            "y": -0.1,
            "xanchor": "center",
            "x": 0.5,
        },
    }
    if width:
        layout_updates["width"] = width

    for key, value in theme_config["layout"].items():
        if key not in layout_updates:
            layout_updates[key] = value

    fig.update_layout(**layout_updates)

    return fig
