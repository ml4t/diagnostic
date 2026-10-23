"""Transaction plots using native kernels and Plotly, not Matplotlib axes.

The financial arguments retain Pyfolio conventions. Returned Plotly figures
support show/write_html/to_dict; existing Matplotlib axes cannot be reused.
Install Diagnostic's viz extra when calling a rendering function.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import polars as pl

from ml4t.diagnostic.evaluation.portfolio_analysis.inputs import positive_integer
from ml4t.diagnostic.evaluation.portfolio_analysis.metrics import _safe_cumprod, annual_return
from ml4t.diagnostic.evaluation.portfolio_analysis.transaction_metrics import (
    execution_timing_frame,
    filled_notional_expr,
)

from . import txn
from .cone_plotting import plot_cones
from .position_plotting import (
    plot_exposures,
    plot_gross_leverage,
    plot_holdings,
    plot_long_short_holdings,
    plot_max_median_position_concentration,
    plot_sector_allocations,
    show_and_plot_top_positions,
)
from .return_panels import (
    plot_annual_returns,
    plot_drawdown_periods,
    plot_drawdown_underwater,
    plot_monthly_returns_dist,
    plot_monthly_returns_heatmap,
    plot_perf_stats,
    plot_return_quantiles,
    plot_rolling_beta,
    plot_rolling_sharpe,
    plot_rolling_volatility,
)
from .return_plotting import plot_returns, plot_rolling_returns, show_perf_stats
from .round_trip_plotting import (
    plot_prob_profit_trade,
    plot_round_trip_lifetimes,
    show_profit_attribution,
)

if TYPE_CHECKING:
    import plotly.graph_objects as go


def _figure(title: str, units: str, ax: Any, kwargs: dict[str, Any]) -> go.Figure:
    from ml4t.diagnostic.visualization.core import create_base_figure

    if ax is not None:
        raise TypeError(
            "The adapter returns Plotly figures; remove Matplotlib ax and use show/write_html"
        )
    width = kwargs.pop("width", None)
    figure = create_base_figure(
        title=title,
        yaxis_title=units,
        theme=kwargs.pop("theme", "default"),
        height=kwargs.pop("height", 420),
        width=width,
    )
    figure.update_layout(
        showlegend=True,
        legend={"orientation": "h", "y": -0.3},
        margin={"l": 65, "r": 25, "t": 45, "b": 145},
        width=width,
        autosize=width is None,
        font_size=14,
        title_font_size=16,
        legend_font_size=14,
    )
    figure.update_xaxes(tickfont_size=14, title_font_size=14)
    figure.update_yaxes(tickfont_size=14, title_font_size=14)
    return figure


def _finish(figure: go.Figure, description: str) -> go.Figure:
    figure.update_layout(
        meta={
            **(figure.layout.meta or {}),
            "analysis": "legacy_transactions",
            "description": description,
        }
    )
    return figure


def _date_ticks(figure: go.Figure, index: pd.DatetimeIndex) -> None:
    if not len(index):
        return
    selected = sorted({0, len(index) // 2, len(index) - 1})
    format_string = "%b %d" if index.normalize().is_unique else "%b %d<br>%H:%M"
    figure.update_xaxes(
        tickmode="array",
        tickvals=[index[position] for position in selected],
        ticktext=[index[position].strftime(format_string) for position in selected],
        tickangle=0,
    )


def plot_turnover(
    returns: pd.Series,
    transactions: pd.DataFrame,
    positions: pd.DataFrame,
    turnover_denom: str = "AGB",
    legend_loc: str = "best",
    ax: Any = None,
    **kwargs: Any,
) -> go.Figure:
    """Daily, civil-month mean and all-time mean of explicit legacy turnover."""
    txn._dated(returns, "returns")
    figure = _figure("Daily Turnover", "turnover fraction", ax, kwargs)
    values = txn.get_turnover(positions, transactions, turnover_denom)
    monthly = values.resample(pd.offsets.MonthEnd()).mean()
    for name, series in (("Daily", values), ("Monthly mean", monthly)):
        figure.add_scatter(x=series.index, y=series.to_list(), name=name, mode="lines", **kwargs)
    figure.add_hline(y=values.mean(), line_dash="dash")
    figure.update_xaxes(range=[returns.index[0], returns.index[-1]])
    _date_ticks(figure, returns.index)
    figure.update_layout(
        meta={
            "denominator": turnover_denom,
            "daily_mean": values.mean(),
            "legend_loc": legend_loc,
            "returns_window": [str(returns.index[0]), str(returns.index[-1])],
        }
    )
    return _finish(
        figure,
        "Legacy gross turnover over the selected closing-value or zero-seeded AGB denominator, with civil-month and all-time daily means. Values above 2 are not clipped.",
    )


def plot_daily_turnover_hist(
    transactions: pd.DataFrame,
    positions: pd.DataFrame,
    turnover_denom: str = "AGB",
    ax: Any = None,
    **kwargs: Any,
) -> go.Figure:
    """Distribution of finite defined legacy turnover, retaining the input sample."""
    figure = _figure("Distribution of Daily Turnover", "observations", ax, kwargs)
    values = txn.get_turnover(positions, transactions, turnover_denom)
    finite = np.isfinite(values)
    figure.add_histogram(x=values[finite].to_list(), name="Turnover", **kwargs)
    figure.update_xaxes(title_text="turnover fraction")
    figure.update_layout(
        meta={"denominator": turnover_denom, "excluded_nonfinite": int((~finite).sum())}
    )
    return _finish(
        figure,
        "Finite defined legacy turnover sample. Nonfinite exclusions are counted in metadata; missingness is not replaced with zero risk. Plotly histogram bins can differ from Seaborn.",
    )


def plot_daily_volume(
    returns: pd.Series,
    transactions: pd.DataFrame,
    ax: Any = None,
    **kwargs: Any,
) -> go.Figure:
    """Daily absolute execution units and their active-date mean, as in Pyfolio."""
    txn._dated(returns, "returns")
    figure = _figure("Daily Trading Volume", "absolute execution units", ax, kwargs)
    values = txn.get_txn_vol(transactions)["txn_shares"]
    figure.add_scatter(
        x=values.index, y=values.to_list(), name="Units", mode="lines+markers", **kwargs
    )
    if not values.empty:
        figure.add_hline(y=float(values.mean()), line_dash="dash")
    figure.update_xaxes(range=[returns.index[0], returns.index[-1]])
    _date_ticks(figure, returns.index)
    figure.update_layout(
        meta={
            "active_date_mean": None if values.empty else float(values.mean()),
            "returns_window": [str(returns.index[0]), str(returns.index[-1])],
        }
    )
    return _finish(
        figure,
        "Absolute supplied execution units on activity dates, with their active-date mean. Mixed instrument units are not necessarily shares.",
    )


def plot_txn_time_hist(
    transactions: pd.DataFrame,
    bin_minutes: int = 5,
    tz: str = "America/New_York",
    ax: Any = None,
    **kwargs: Any,
) -> go.Figure:
    """Legacy inclusive09:30-16:00 notional distribution in the selected timezone.

    This deliberately differs from native complete-day timing. Empty selected
    notional is undefined, not a zero distribution. Other actual event hours
    remain in the input and are counted explicitly as excluded here.
    """
    figure = _figure("Transaction Time Distribution", "fraction of selected notional", ax, kwargs)
    txn._dated(transactions, "transactions")
    fills = (
        pl.from_pandas(transactions[["amount", "price"]], nan_to_null=True)
        .rename({"amount": "quantity"})
        .with_columns(
            pl.from_pandas(transactions.index).rename("timestamp"), pl.lit(1).alias("multiplier")
        )
        .with_columns(filled_notional_expr().alias("filled_notional"))
    )
    timing = execution_timing_frame(
        fills, bin_minutes=bin_minutes, timezone=tz, start_minute=570, end_minute=961
    )
    figure.add_bar(
        x=timing["time"].to_list(),
        y=timing["notional_fraction"].to_list(),
        name="Selected notional",
        **kwargs,
    )
    figure.update_xaxes(
        title_text=f"civil time ({tz})", tickmode="array", tickvals=["09:30", "12:00", "16:00"]
    )
    figure.update_layout(
        meta={
            "timezone": tz,
            "bin_minutes": bin_minutes,
            "window": "09:30-16:00 inclusive",
            "included_fills": timing["num_fills"].sum(),
            "excluded_fills": fills.height - timing["num_fills"].sum(),
        }
    )
    return _finish(
        figure,
        f"Notional fractions selected from 09:30 through the 16:00 minute in {tz}. Outside fills are counted as excluded. Empty selected notional is undefined, not zero.",
    )


def plot_slippage_sweep(
    returns: pd.Series,
    positions: pd.DataFrame,
    transactions: pd.DataFrame,
    slippage_params: tuple[float, ...] = (3, 8, 10, 12, 15, 20, 50),
    ax: Any = None,
    **kwargs: Any,
) -> go.Figure:
    """Legacy additional-bps wealth paths starting at one, using native compounding."""
    figure = _figure("Additional Slippage Wealth Paths", "wealth (initial1)", ax, kwargs)
    if not slippage_params:
        raise ValueError("slippage_params must contain at least one scenario")
    for bps in slippage_params:
        adjusted = txn.adjust_returns_for_slippage(returns, positions, transactions, bps)
        figure.add_scatter(
            x=adjusted.index,
            y=_safe_cumprod(1 + adjusted.to_numpy()),
            mode="lines",
            name=f"{bps:g} bps",
            **kwargs,
        )
    figure.update_layout(
        meta={
            "denominator": "legacy closing position value",
            "missing_returns": "legacy zero contribution to wealth product",
        }
    )
    _date_ticks(figure, adjusted.index)
    return _finish(
        figure,
        "Wealth starts at 1, compounding legacy closing-value extra-bps adjusted returns. Observed fees are retained once. Missing returns contribute no wealth change under the explicit legacy convention.",
    )


def plot_slippage_sensitivity(
    returns: pd.Series,
    positions: pd.DataFrame,
    transactions: pd.DataFrame,
    ax: Any = None,
    *,
    periods_per_year: int = 252,
    **kwargs: Any,
) -> go.Figure:
    """Legacy1..99bps CAGR sensitivity, with an explicit optional annualization scale."""
    scale = positive_integer(periods_per_year, "periods_per_year")
    figure = _figure("Additional Slippage Sensitivity", "annual return (CAGR)", ax, kwargs)
    values = [
        annual_return(
            txn.adjust_returns_for_slippage(returns, positions, transactions, bps).to_numpy(), scale
        )
        for bps in range(1, 100)
    ]
    figure.add_scatter(
        x=list(range(1, 100)), y=values, mode="lines", name="Annual return", **kwargs
    )
    figure.update_xaxes(title_text="additional cash bps")
    figure.update_layout(
        meta={"periods_per_year": scale, "denominator": "legacy closing position value"}
    )
    return _finish(
        figure,
        f"CAGR for the original 1 through 99 additional cash-bps scenarios, with explicit {scale} observations per year. This is compounded annual return, not an arithmetic mean.",
    )


def plot_capacity_sweep(
    returns: pd.Series,
    transactions: pd.DataFrame,
    market_data: pd.DataFrame,
    bt_starting_capital: float,
    min_pv: int = 100000,
    max_pv: int = 300000000,
    step_size: int = 1000000,
    ax: Any = None,
    *,
    periods_per_year: int = 252,
    impact: float = 0.1,
    **kwargs: Any,
) -> go.Figure:
    """Familiar fixed-share capital sweep with explicit annualization/model.

    Stop before a ratio below -1, as in Pyfolio. Original missing returns retain
    metric-specific omission; a newly unavailable active market penalty cannot
    silently shorten that sample. The returned object is Plotly, not an Axes.
    """
    from ml4t.diagnostic.compat import empyrical as ep

    from . import capacity

    minimum = positive_integer(min_pv, "min_pv")
    maximum = positive_integer(max_pv, "max_pv")
    step = positive_integer(step_size, "step_size")
    scale = positive_integer(periods_per_year, "periods_per_year")
    if maximum <= minimum:
        raise ValueError("max_pv must exceed min_pv (exclusive)")
    figure = _figure("Capital Base Performance Sweep", "annualized Sharpe ratio", ax, kwargs)
    daily = capacity.daily_txns_with_bar_data(transactions, market_data)
    capitals, ratios, stopped_at = [], [], None
    for capital in range(minimum, maximum, step):
        adjusted = capacity.apply_slippage_penalty(
            returns, daily, capital, bt_starting_capital, impact=impact
        )
        unavailable = (adjusted.isna() & returns.notna()).any()
        ratio = np.nan if unavailable else ep.sharpe_ratio(adjusted, annualization=scale)
        if ratio < -1:
            stopped_at = capital
            break
        capitals.append(capital / 1e6)
        ratios.append(float(ratio) if np.isfinite(ratio) else None)
    figure.add_scatter(x=capitals, y=ratios, mode="lines+markers", name="Extra impact", **kwargs)
    if not any(value is not None for value in ratios):
        figure.add_annotation(
            text="Unavailable: no finite Sharpe under the supplied sample and impact model.",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
            font_size=14,
        )
    figure.update_xaxes(title_text="starting capital (millions)")
    figure.update_layout(
        meta={
            "analysis": "legacy_capacity",
            "capital_units": "millions of the supplied common account/market currency; no FX",
            "periods_per_year": scale,
            "impact_coefficient": impact,
            "nav_timing": "compounded_closing",
            "stopped_at_capital": stopped_at,
            "description": "Retrospective quadratic volume-share scenarios using observed closing marks and volume, not prospective executable liquidity. Net returns lose only extra hypothetical impact. Newly missing active market coverage remains unavailable; Sharpe is dimensionless.",
        }
    )
    return figure


__all__ = [
    "plot_exposures",
    "plot_gross_leverage",
    "plot_holdings",
    "plot_long_short_holdings",
    "plot_max_median_position_concentration",
    "plot_sector_allocations",
    "show_and_plot_top_positions",
    "plot_annual_returns",
    "plot_drawdown_periods",
    "plot_drawdown_underwater",
    "plot_monthly_returns_dist",
    "plot_monthly_returns_heatmap",
    "plot_perf_stats",
    "plot_return_quantiles",
    "plot_rolling_beta",
    "plot_rolling_sharpe",
    "plot_rolling_volatility",
    "plot_returns",
    "plot_rolling_returns",
    "show_perf_stats",
    "plot_cones",
    "plot_capacity_sweep",
    "plot_round_trip_lifetimes",
    "plot_prob_profit_trade",
    "show_profit_attribution",
    "plot_turnover",
    "plot_daily_turnover_hist",
    "plot_daily_volume",
    "plot_txn_time_hist",
    "plot_slippage_sweep",
    "plot_slippage_sensitivity",
]
