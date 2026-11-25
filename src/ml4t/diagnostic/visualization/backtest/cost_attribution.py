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

from ml4t.diagnostic.visualization._colors import COLORS as _ML4T_COLORS
from ml4t.diagnostic.visualization.backtest._formatting import (
    format_currency_adaptive,
    format_percent_adaptive,
)
from ml4t.diagnostic.visualization.core import get_theme_config, validate_theme

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
    theme = validate_theme(theme)
    theme_config = get_theme_config(theme)

    # Build cost categories (use "relative" for Gross PnL so color follows sign)
    labels = ["Gross PnL"]
    values = [gross_pnl]
    measures = ["relative"]

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
        text = [format_currency_adaptive(gross_pnl)]
        for val in values[1:-1]:
            pct = abs(val) / abs(gross_pnl) * 100
            text.append(f"{format_currency_adaptive(val)} ({format_percent_adaptive(pct)})")
        text.append(format_currency_adaptive(net_pnl))
    else:
        text = [format_currency_adaptive(v) for v in values]

    # Colors: green for positive, red for negative
    positive_color = _ML4T_COLORS.get("positive", "#10b981")
    negative_color = _ML4T_COLORS.get("negative", "#ef4444")

    fig = go.Figure(
        go.Waterfall(
            name="Cost Attribution",
            orientation="v",
            x=labels,
            y=values,
            measure=measures,
            text=text,
            textposition="outside",
            increasing={"marker": {"color": positive_color}},
            decreasing={"marker": {"color": negative_color}},
            totals={"marker": {"color": positive_color if net_pnl >= 0 else negative_color}},
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
    base_costs_bps: float | None = None,
    cost_multipliers: list[float] | None = None,
    trades_per_year: float | None = None,
    risk_free_rate: float = 0.0,
    title: str = "Cost Sensitivity Analysis",
    show_breakeven: bool = True,
    theme: str | None = None,
    height: int = 500,
    width: int | None = None,
    *,
    periods_per_year: int = 252,
    realized_costs: pl.Series | np.ndarray | None = None,
) -> go.Figure:
    """Show how Sharpe ratio and CAGR change as transaction costs are scaled.

    Each point is the return series with its transaction costs multiplied by
    ``k`` for every ``k`` in ``cost_multipliers``: ``k = 0`` is the return before
    costs and ``k = 1`` ("Current") is the cost level the input states. The
    caller says which returns it passes:

    - **Realized costs (recommended).** ``returns`` are net of costs and
      ``realized_costs`` holds the cost each period paid, as a return aligned with
      ``returns``. Point ``k`` is ``returns + (1 - k) * realized_costs``, so ``k = 1``
      reproduces the net series exactly. ``BacktestProfile.cost_returns`` provides
      this series for an ``ml4t-backtest`` result. ``base_costs_bps``, when given,
      is the average cost the run paid, and labels the x axis in bps.
    - **Assumed uniform costs.** ``returns`` are gross, and costs are spread evenly
      over the periods: ``base_costs_bps`` per unit traded, ``trades_per_year``
      times the full equity per year. Point ``k`` is ``returns - k * cost``.

    Parameters
    ----------
    returns : pl.Series or np.ndarray
        Periodic returns: net of ``realized_costs`` when given, gross otherwise.
    base_costs_bps : float, optional
        Cost per unit of traded notional at ``k = 1``, in basis points. Defaults to
        10 for gross returns. With realized costs it only scales the x axis.
    cost_multipliers : list[float], optional
        Non-negative cost multiples to evaluate (default: [0, 0.5, 1, 1.5, 2, 3, 5]).
    trades_per_year : float, optional
        Times per year the full equity is traded, defaulting to 252 for gross returns.
    risk_free_rate : float
        Annual risk-free rate for the Sharpe ratio.
    title : str
        Chart title.
    show_breakeven : bool
        Whether to mark the cost level at which the Sharpe ratio reaches zero.
    theme : str, optional
        Theme name.
    height : int
        Figure height in pixels.
    width : int, optional
        Figure width in pixels.
    periods_per_year : int
        Positive return observations per year, independent of trade frequency.
    realized_costs : pl.Series or np.ndarray, optional
        Per-period cost paid, as a return, aligned with net ``returns``.

    Returns
    -------
    go.Figure
        Sharpe ratio and CAGR, stacked, against the cost level.
    """
    import polars as pl

    from ml4t.diagnostic.evaluation.portfolio_analysis.metrics import annual_return
    from ml4t.diagnostic.metrics.risk_adjusted import _periodic_risk_free_rate, sharpe_ratio

    if (
        isinstance(periods_per_year, bool)
        or not isinstance(periods_per_year, (int, np.integer))
        or periods_per_year <= 0
    ):
        raise ValueError("periods_per_year must be a positive integer")
    theme = validate_theme(theme)
    theme_config = get_theme_config(theme)

    returns_arr = np.asarray(
        returns.to_numpy() if isinstance(returns, pl.Series) else returns, dtype=float
    )
    if realized_costs is not None:
        if trades_per_year is not None:
            raise ValueError(
                "trades_per_year applies to gross returns; realized_costs already "
                "state what each period paid"
            )
        costs = np.asarray(
            realized_costs.to_numpy() if isinstance(realized_costs, pl.Series) else realized_costs,
            dtype=float,
        )
        if costs.shape != returns_arr.shape:
            raise ValueError("realized_costs must align with returns")
        before_costs = returns_arr + costs
    else:
        base_costs_bps = 10.0 if base_costs_bps is None else base_costs_bps
        trades_per_year = 252 if trades_per_year is None else trades_per_year
        before_costs = returns_arr
        costs = np.full_like(
            returns_arr, base_costs_bps / 10_000 * trades_per_year / periods_per_year
        )
    if not (np.isfinite(before_costs).all() and np.isfinite(costs).all()):
        raise ValueError("returns and costs must be finite")
    if base_costs_bps is not None and not (np.isfinite(base_costs_bps) and base_costs_bps >= 0):
        raise ValueError("base_costs_bps must be finite and non-negative")

    multipliers = sorted(float(k) for k in (cost_multipliers or [0, 0.5, 1, 1.5, 2, 3, 5]))
    if not multipliers or any(not np.isfinite(k) or k < 0 for k in multipliers):
        raise ValueError("cost_multipliers must be finite and non-negative")

    def cost_level(k: float) -> float:
        return k * base_costs_bps if base_costs_bps is not None else k

    sharpe_values: list[float] = []
    cagr_values: list[float] = []
    for k in multipliers:
        # k = 1 adds nothing to net input, so it reproduces the reported series exactly.
        adjusted = (
            returns_arr + (1.0 - k) * costs
            if realized_costs is not None
            else before_costs - k * costs
        )
        sharpe_values.append(
            float(
                sharpe_ratio(
                    adjusted, risk_free_rate=risk_free_rate, periods_per_year=periods_per_year
                )
            )
        )
        cagr_values.append(annual_return(adjusted, periods_per_year=periods_per_year) * 100)
    cost_levels = [cost_level(k) for k in multipliers]

    if base_costs_bps is None:
        x_title = "Cost multiple of realized (×)"
        x_hover = "Cost: %{x:.2f}× realized"
    elif realized_costs is not None:
        x_title = "Cost per fill (bps of traded notional)"
        x_hover = "Cost: %{x:.1f} bps"
    else:
        x_title = "Cost per trade (bps)"
        x_hover = "Cost: %{x:.1f} bps"

    colors = theme_config["colorway"]
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        subplot_titles=("Sharpe ratio", "CAGR (%)"),
        vertical_spacing=0.14,
    )
    fig.add_trace(
        go.Scatter(
            x=cost_levels,
            y=sharpe_values,
            mode="lines+markers",
            name="Sharpe Ratio",
            cliponaxis=False,
            line={"color": colors[0], "width": 3},
            marker={"size": 8},
            hovertemplate=x_hover + "<br>Sharpe: %{y:.2f}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=cost_levels,
            y=cagr_values,
            mode="lines+markers",
            name="CAGR (%)",
            cliponaxis=False,
            line={"color": colors[1] if len(colors) > 1 else colors[0], "width": 3},
            marker={"size": 8},
            hovertemplate=x_hover + "<br>CAGR: %{y:.1f}%<extra></extra>",
        ),
        row=2,
        col=1,
    )
    for row in (1, 2):
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=row, col=1)

    # The mean return is linear in k and the Sharpe ratio has its sign, so the
    # breakeven multiple is exact rather than interpolated between points.
    excess_before_costs = float(np.mean(before_costs)) - _periodic_risk_free_rate(
        risk_free_rate, periods_per_year
    )
    mean_cost = float(np.mean(costs))
    if show_breakeven and mean_cost > 0 and excess_before_costs > 0:
        breakeven_k = excess_before_costs / mean_cost
        if breakeven_k <= multipliers[-1]:
            breakeven = cost_level(breakeven_k)
            label = (
                f"Breakeven {breakeven_k:.2f}×"
                if base_costs_bps is None
                else f"Breakeven {breakeven:.1f} bps"
            )
            fig.add_vline(
                x=breakeven,
                line_dash="dot",
                line_color="red",
                annotation_text=label,
                annotation_position="bottom right",
                row=1,
                col=1,
            )

    if 1.0 in multipliers:
        idx = multipliers.index(1.0)
        fig.add_annotation(
            x=cost_levels[idx],
            y=sharpe_values[idx],
            text="Current",
            showarrow=True,
            arrowhead=2,
            ax=24,
            ay=-28,
            row=1,
            col=1,
        )

    layout_updates: dict[str, object] = {
        "title": {"text": title, "font": {"size": 18}},
        "height": height,
        "showlegend": False,
        "xaxis": {"rangemode": "nonnegative"},
        "xaxis2": {"title": x_title, "rangemode": "nonnegative"},
        # Transparent ticks keep y labels clear of the unclipped markers at zero cost.
        "yaxis": {"ticks": "outside", "ticklen": 8, "tickcolor": "rgba(0,0,0,0)"},
        "yaxis2": {"ticks": "outside", "ticklen": 8, "tickcolor": "rgba(0,0,0,0)"},
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

    theme = validate_theme(theme)
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

    theme = validate_theme(theme)
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
        secondary_values = [c / n for c, n in zip(total_costs, n_trades, strict=False)]
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
    theme = validate_theme(theme)
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
