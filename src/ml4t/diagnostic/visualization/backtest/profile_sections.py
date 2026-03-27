"""Profile-native sections for backtest tearsheets."""

from __future__ import annotations

from typing import TYPE_CHECKING

import plotly.graph_objects as go
import polars as pl
from plotly.subplots import make_subplots

from ml4t.diagnostic.visualization._colors import COLORS, SERIES_COLORS
from ml4t.diagnostic.visualization.core import get_theme_config, validate_theme

if TYPE_CHECKING:
    from ml4t.diagnostic.integration.backtest_profile import BacktestProfile


def _table_figure(
    headers: list[str],
    values: list[list[object]],
    *,
    title: str,
    theme: str | None = None,
    height: int = 420,
) -> go.Figure:
    theme = validate_theme(theme)
    theme_config = get_theme_config(theme)
    fig = go.Figure(
        data=[
            go.Table(
                header={
                    "values": headers,
                    "align": "left",
                    "font": {"size": 12},
                },
                cells={
                    "values": values,
                    "align": "left",
                    "font": {"size": 11},
                },
            )
        ]
    )
    fig.update_layout(theme_config["layout"])
    fig.update_layout(title={"text": title, "font": {"size": 18}}, height=height)
    return fig


def plot_activity_overview(
    profile: BacktestProfile, theme: str | None = None,
) -> go.Figure | str:
    """Plot profile-native activity diagnostics.

    Returns the turnover chart as a Plotly figure. If rebalance data is
    available, returns an HTML string with the chart + a collapsible
    rebalance events table.
    """
    import html as html_mod

    theme = validate_theme(theme)
    timeline = profile.activity["turnover_timeline"]
    rebalance = profile.activity["rebalance_summary"].head(15)

    fig = go.Figure()

    if not timeline.is_empty():
        fig.add_trace(
            go.Scatter(
                x=timeline["timestamp"].to_list(),
                y=timeline["turnover"].to_list(),
                mode="lines",
                name="Turnover",
                line={"width": 1.5},
            ),
        )
        cost_drag = timeline["cost_drag"]
        if cost_drag.drop_nulls().len() > 0 and cost_drag.drop_nulls().max() > 0:
            fig.add_trace(
                go.Scatter(
                    x=timeline["timestamp"].to_list(),
                    y=cost_drag.to_list(),
                    mode="lines",
                    name="Cost Drag",
                    line={"width": 1.5, "dash": "dot", "color": COLORS["negative"]},
                    yaxis="y2",
                ),
            )

    theme_config = get_theme_config(theme)
    fig.update_layout(theme_config["layout"])
    fig.update_layout(
        title={"text": "Activity", "font": {"size": 16}},
        height=350,
        yaxis={"title": "Turnover"},
        yaxis2={
            "overlaying": "y",
            "side": "right",
            "title": "Cost Drag",
            "showgrid": False,
            "tickformat": ".2%",
        },
    )

    # If no rebalance data, return the figure directly
    if rebalance.is_empty():
        return fig

    # Build collapsible rebalance table as HTML
    n_events = rebalance.height
    rows_html = ""
    for i in range(n_events):
        key = html_mod.escape(str(rebalance["rebalance_key"][i]))
        ts = html_mod.escape(str(rebalance["timestamp"][i])[:19])
        notional = f"{float(rebalance['filled_notional'][i]):,.0f}"
        cost = f"{float(rebalance['implementation_cost'][i]):,.2f}"
        syms = str(rebalance["symbols_touched"][i])
        rows_html += (
            f"<tr><td>{key}</td><td>{ts}</td>"
            f"<td>{notional}</td><td>{cost}</td><td>{syms}</td></tr>"
        )

    table_html = (
        f'<details class="section-detail" style="margin-top:8px">'
        f'<summary class="section-detail-summary">'
        f'Rebalance Events ({n_events})</summary>'
        f'<div class="metrics-table-wrap">'
        f'<table class="metrics-table">'
        f'<thead><tr><th>Rebalance</th><th>Timestamp</th>'
        f'<th>Notional</th><th>Cost</th><th>Symbols</th></tr></thead>'
        f'<tbody>{rows_html}</tbody></table></div></details>'
    )

    # Serialize chart to HTML and combine with table
    from .tearsheet import _figure_to_clean_html

    fig.update_layout(autosize=True, width=None, margin={"t": 40})
    chart_html = f'<div class="chart-container">{_figure_to_clean_html(fig)}</div>'
    return chart_html + table_html


def plot_overview_snapshot(profile: BacktestProfile, theme: str | None = None) -> go.Figure:
    """Plot equity curve + drawdown subplot for the overview workspace.

    Top panel (75%): cumulative return line.
    Bottom panel (25%): drawdown filled area (red/salmon).
    Shared x-axis — drawdown acts as context for the equity curve.
    """
    theme = validate_theme(theme)
    equity = profile.equity_df
    theme_config = get_theme_config(theme)

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.75, 0.25],
    )

    if not equity.is_empty():
        timestamps = equity["timestamp"].to_list()
        cum_ret = equity["cumulative_return"].to_list()

        # Top: equity curve
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=cum_ret,
                mode="lines",
                name="Strategy",
                line={"color": theme_config["colorway"][0], "width": 2},
                hovertemplate="%{y:.1%}<extra></extra>",
                showlegend=False,
            ),
            row=1,
            col=1,
        )

        # Bottom: drawdown area
        if "drawdown" in equity.columns:
            dd = equity["drawdown"].to_list()
        else:
            import numpy as np

            cum_arr = np.array(cum_ret)
            hwm = np.maximum.accumulate(cum_arr)
            dd = ((cum_arr - hwm) / np.where(hwm > 0, hwm, 1.0)).tolist()

        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=dd,
                fill="tozeroy",
                mode="lines",
                name="Drawdown",
                line={"color": SERIES_COLORS["drawdown_line"], "width": 0.5},
                fillcolor=SERIES_COLORS["drawdown"],
                hovertemplate="%{y:.1%}<extra></extra>",
                showlegend=False,
            ),
            row=2,
            col=1,
        )

    fig.update_layout(theme_config["layout"])
    fig.update_layout(
        title="Cumulative Return",
        height=450,
        margin={"t": 40, "l": 48, "r": 24, "b": 32},
        hovermode="x unified",
    )
    fig.update_yaxes(tickformat=".0%", row=1, col=1)
    fig.update_yaxes(tickformat=".0%", row=2, col=1)
    fig.update_xaxes(showticklabels=False, row=1, col=1)
    return fig


def plot_occupancy_overview(profile: BacktestProfile, theme: str | None = None) -> go.Figure:
    """Plot occupancy and realized structure diagnostics."""
    theme = validate_theme(theme)
    timeline = profile.occupancy["timeline"]
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.12,
        subplot_titles=("Exposure Fractions", "Open Positions"),
    )

    if not timeline.is_empty():
        fig.add_trace(
            go.Scatter(
                x=timeline["timestamp"].to_list(),
                y=timeline["gross_exposure_fraction"].to_list(),
                mode="lines",
                name="Gross Exposure",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=timeline["timestamp"].to_list(),
                y=timeline["net_exposure_fraction"].to_list(),
                mode="lines",
                name="Net Exposure",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=timeline["timestamp"].to_list(),
                y=timeline["open_positions"].to_list(),
                mode="lines",
                name="Open Positions",
            ),
            row=2,
            col=1,
        )

    theme_config = get_theme_config(theme)
    fig.update_layout(theme_config["layout"])
    fig.update_layout(title={"text": "Exposure", "font": {"size": 16}}, height=350)
    return fig


def plot_attribution_overview(profile: BacktestProfile, theme: str | None = None) -> go.Figure:
    """Plot symbol contribution and burden diagnostics."""
    theme = validate_theme(theme)
    by_symbol = profile.attribution["by_symbol"]
    if by_symbol.is_empty():
        return _table_figure(
            ["Message"],
            [["Attribution unavailable"]],
            title="Attribution",
            theme=theme,
            height=220,
        )

    # Filter to symbols with actual trades or non-zero PnL
    if "trade_count" in by_symbol.columns:
        by_symbol = by_symbol.filter(pl.col("trade_count") > 0)
    elif "net_pnl" in by_symbol.columns:
        by_symbol = by_symbol.filter(pl.col("net_pnl").abs() > 1e-10)
    if by_symbol.is_empty():
        return _table_figure(
            ["Message"],
            [["No traded symbols for attribution"]],
            title="Attribution",
            theme=theme,
            height=220,
        )
    by_symbol = by_symbol.sort("net_pnl", descending=True)
    top = by_symbol.head(8)
    fig = make_subplots(
        rows=2,
        cols=1,
        specs=[[{"type": "bar"}], [{"type": "table"}]],
        vertical_spacing=0.12,
        row_heights=[0.56, 0.44],
    )
    # Use contribution share (%) for cross-time comparability
    bar_col = "pnl_contribution_share" if "pnl_contribution_share" in top.columns else "net_pnl"
    bar_label = "PnL Contribution" if bar_col == "pnl_contribution_share" else "Net PnL"
    bar_values = top[bar_col].to_list()
    colors = [COLORS["positive"] if v >= 0 else COLORS["negative"] for v in bar_values]
    fig.add_trace(
        go.Bar(
            x=top["symbol"].to_list(),
            y=bar_values,
            name=bar_label,
            marker_color=colors,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Table(
            header={
                "values": [
                    "Symbol",
                    "PnL Share",
                    "Turnover Share",
                    "Cost Share",
                    "Burden Score",
                ]
            },
            cells={
                "values": [
                    top["symbol"].to_list(),
                    top["pnl_contribution_share"].round(4).to_list(),
                    top["turnover_contribution_share"].round(4).to_list(),
                    top["cost_contribution_share"].round(4).to_list(),
                    top["burden_score"].round(4).to_list(),
                ]
            },
        ),
        row=2,
        col=1,
    )
    theme_config = get_theme_config(theme)
    fig.update_layout(theme_config["layout"])
    y_axis_config: dict[str, object] = {"title": bar_label}
    if bar_col == "pnl_contribution_share":
        y_axis_config["tickformat"] = ".1%"
    fig.update_layout(
        title={"text": "Attribution", "font": {"size": 16}},
        height=380,
        yaxis=y_axis_config,
    )
    return fig


def plot_drawdown_anatomy(profile: BacktestProfile, theme: str | None = None) -> go.Figure:
    """Plot drawdown episodes as a profile-native table."""
    episodes = profile.drawdown["episodes"]
    if episodes.is_empty():
        return _table_figure(
            ["Message"],
            [["No drawdown episodes detected"]],
            title="Drawdown Anatomy",
            theme=theme,
            height=220,
        )

    return _table_figure(
        [
            "Episode",
            "Peak",
            "Trough",
            "Recovery",
            "Depth",
            "Bars",
            "Status",
            "Top Contributors",
        ],
        [
            episodes["episode_id"].to_list(),
            episodes["peak_timestamp"].to_list(),
            episodes["trough_timestamp"].to_list(),
            episodes["recovery_timestamp"].to_list(),
            episodes["depth"].round(4).to_list(),
            episodes["peak_to_trough_bars"].to_list(),
            episodes["status"].to_list(),
            episodes["top_contributors"].to_list(),
        ],
        title="Drawdown Anatomy",
        theme=theme,
        height=420,
    )


def plot_cost_bridge(profile: BacktestProfile, theme: str | None = None) -> go.Figure:
    """Plot a profile-native gross-to-net cost bridge."""
    from .cost_attribution import plot_cost_waterfall

    total_pnl = float(
        profile.edge["metrics"].get("avg_trade", 0.0) * profile.edge["metrics"].get("num_trades", 0)
    )
    total_commission = float(profile.edge["metrics"].get("total_commission", 0.0))
    total_slippage = float(profile.edge["metrics"].get("total_slippage", 0.0))
    gross_pnl = total_pnl + total_commission + total_slippage

    return plot_cost_waterfall(
        gross_pnl=gross_pnl,
        commission=total_commission,
        slippage=total_slippage,
        net_pnl=total_pnl,
        title="Gross-To-Net Cost Bridge",
        theme=theme,
    )


def plot_stability_overview(profile: BacktestProfile, theme: str | None = None) -> go.Figure:
    """Plot 2x2 rolling stability diagnostics (return, Sharpe, volatility, beta)."""
    import numpy as np

    theme = validate_theme(theme)
    rolling = profile.performance["rolling"]
    returns = profile.daily_returns.to_numpy() if hasattr(profile.daily_returns, "to_numpy") else np.asarray(profile.daily_returns)

    theme_config = get_theme_config(theme)

    fig = make_subplots(
        rows=2,
        cols=2,
        shared_xaxes=True,
        subplot_titles=(
            "Rolling Return (252d)",
            "Rolling Sharpe (252d)",
            "Rolling Volatility (252d)",
            "Rolling Sortino (252d)",
        ),
        vertical_spacing=0.14,
        horizontal_spacing=0.08,
    )

    if not rolling.is_empty():
        if profile.equity_df.height == rolling.height and "timestamp" in profile.equity_df.columns:
            x_axis = profile.equity_df["timestamp"].to_list()
        else:
            x_axis = rolling["period_index"].to_list()

        line_kw = {"width": 1.5}
        colorway = theme_config.get("colorway", ["#2563eb"])
        line_color = colorway[0] if colorway else "#2563eb"

        # Row 1, Col 1: Rolling Return (252d)
        fig.add_trace(
            go.Scatter(
                x=x_axis, y=rolling["rolling_return_252"].to_list(),
                mode="lines", line={**line_kw, "color": line_color},
                hovertemplate="%{y:.1%}<extra></extra>", showlegend=False,
            ),
            row=1, col=1,
        )

        # Row 1, Col 2: Rolling Sharpe (252d)
        fig.add_trace(
            go.Scatter(
                x=x_axis, y=rolling["rolling_sharpe_252"].to_list(),
                mode="lines", line={**line_kw, "color": line_color},
                hovertemplate="%{y:.2f}<extra></extra>", showlegend=False,
            ),
            row=1, col=2,
        )

        # Row 2, Col 1: Rolling Volatility (252d)
        n = len(returns)
        vol_252 = np.full(n, np.nan)
        sqrt_252 = float(np.sqrt(252))
        for i in range(251, n):
            vol_252[i] = float(np.std(returns[i - 251 : i + 1], ddof=1)) * sqrt_252
        fig.add_trace(
            go.Scatter(
                x=x_axis, y=vol_252.tolist(),
                mode="lines", line={**line_kw, "color": line_color},
                hovertemplate="%{y:.1%}<extra></extra>", showlegend=False,
            ),
            row=2, col=1,
        )

        # Row 2, Col 2: Rolling Sortino (252d)
        sortino_252 = np.full(n, np.nan)
        for i in range(251, n):
            window = returns[i - 251 : i + 1]
            mean_r = float(np.mean(window))
            downside = window[window < 0]
            if len(downside) > 0:
                downside_std = float(np.std(downside, ddof=1)) * sqrt_252
                if downside_std > 0:
                    sortino_252[i] = mean_r * 252 / downside_std
        fig.add_trace(
            go.Scatter(
                x=x_axis, y=sortino_252.tolist(),
                mode="lines", line={**line_kw, "color": line_color},
                hovertemplate="%{y:.2f}<extra></extra>", showlegend=False,
            ),
            row=2, col=2,
        )

    fig.update_layout(theme_config["layout"])
    fig.update_layout(
        height=400,
        margin={"t": 40, "l": 48, "r": 24, "b": 32},
        showlegend=False,
    )
    fig.update_yaxes(tickformat=".0%", row=1, col=1)
    fig.update_yaxes(tickformat=".0%", row=2, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color=COLORS["neutral"], line_width=0.5, row=1, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color=COLORS["neutral"], line_width=0.5, row=1, col=2)
    return fig
