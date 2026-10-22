"""Complete familiar return-report panels over existing analytical kernels."""

from __future__ import annotations

import calendar
from html import escape
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from ml4t.diagnostic.compat import empyrical as ep

from . import timeseries
from .return_plotting import _live_boundary, _return_series, _set_legend, _time_axis

if TYPE_CHECKING:
    import plotly.graph_objects as go


def _figure(title: str, units: str, ax: Any, options: dict[str, Any]) -> go.Figure:
    from .plotting import _figure as base

    figure = base(title, units, ax, options)
    if options:
        raise TypeError(f"Unknown Plotly layout options: {', '.join(sorted(options))}")
    return figure


def _line(figure: go.Figure, values: pd.Series, label: str) -> None:
    figure.add_scatter(
        x=[value.isoformat() for value in values.index],
        y=values.tolist(),
        mode="lines",
        name=escape(label),
        connectgaps=False,
    )


def _rolling_panel(
    returns: pd.Series,
    factor_returns: pd.Series | None,
    window: int,
    metric: str,
    legend_loc: str | None,
    ax: Any,
    scale: int,
    options: dict[str, Any],
) -> go.Figure:
    supplied = _return_series(returns)
    label = "Volatility" if metric == "rolling_volatility" else "Sharpe"
    figure = _figure(f"Rolling {label} ({window} observations)", label, ax, options)
    _set_legend(figure, legend_loc)
    actual = getattr(timeseries, metric)(supplied, window, periods_per_year=scale)
    _line(figure, actual, label)
    dates = supplied.index
    if factor_returns is not None:
        benchmark = _return_series(factor_returns)
        _line(
            figure,
            getattr(timeseries, metric)(benchmark, window, periods_per_year=scale),
            "Benchmark",
        )
        dates = dates.union(benchmark.index)
    if np.isfinite(actual.mean()):
        figure.add_hline(y=actual.mean(), line_dash="dash")
    figure.add_hline(y=0)
    _time_axis(figure, dates)
    figure.update_layout(
        meta={
            "analysis": "legacy_returns",
            "periods_per_year": scale,
            "window_observations": window,
            "description": "Full observed row windows, not calendar months or rebalances. Warmup and incomplete windows stay unavailable. Benchmark windows use their supplied clock. Average overlays omit undefined windows.",
        },
    )
    if metric == "rolling_volatility":
        figure.update_yaxes(tickformat=".1%")
    return figure


def plot_rolling_volatility(
    returns: pd.Series,
    factor_returns: pd.Series | None = None,
    rolling_window: int = 126,
    legend_loc: str | None = "best",
    ax: Any = None,
    *,
    periods_per_year: int = 252,
    **kwargs: Any,
) -> go.Figure:
    return _rolling_panel(
        returns,
        factor_returns,
        rolling_window,
        "rolling_volatility",
        legend_loc,
        ax,
        periods_per_year,
        kwargs,
    )


def plot_rolling_sharpe(
    returns: pd.Series,
    factor_returns: pd.Series | None = None,
    rolling_window: int = 126,
    legend_loc: str | None = "best",
    ax: Any = None,
    *,
    periods_per_year: int = 252,
    **kwargs: Any,
) -> go.Figure:
    return _rolling_panel(
        returns,
        factor_returns,
        rolling_window,
        "rolling_sharpe",
        legend_loc,
        ax,
        periods_per_year,
        kwargs,
    )


def plot_rolling_beta(
    returns: pd.Series,
    factor_returns: pd.Series,
    legend_loc: str | None = "best",
    ax: Any = None,
    **kwargs: Any,
) -> go.Figure:
    supplied = _return_series(returns)
    benchmark = _return_series(factor_returns)
    figure = _figure("Rolling portfolio beta", "Beta", ax, kwargs)
    _set_legend(figure, legend_loc)
    for window in (126, 252):
        values = timeseries.rolling_beta(supplied, benchmark, window)
        _line(figure, values, f"{window}+1 observations")
        if window == 126 and np.isfinite(values.mean()):
            figure.add_hline(y=values.mean(), line_dash="dash")
    figure.add_hline(y=1)
    _time_axis(figure, supplied.index)
    figure.update_layout(
        meta={
            "analysis": "legacy_returns",
            "window_observations": [127, 253],
            "description": "Historical inclusive endpoint convention: named126/252windows contain127/253observations. Existing paired regression omits missing pairs within each selected window. Beta is dimensionless.",
        },
    )
    return figure


def plot_drawdown_underwater(returns: pd.Series, ax: Any = None, **kwargs: Any) -> go.Figure:
    supplied = _return_series(returns)
    figure = _figure("Underwater", "Drawdown", ax, kwargs)
    _line(figure, ep.drawdown_series(supplied), "Drawdown")
    figure.data[0].fill = "tozeroy"
    _time_axis(figure, supplied.index)
    figure.update_yaxes(tickformat=".1%")
    figure.update_layout(
        meta={
            "analysis": "legacy_returns",
            "initial_capital_included": True,
            "description": "Shared drawdown kernel includes initial capital, even when the first return loses money. This corrects Pyfolio's report-only omission of that initial peak. Missing returns are neutral product factors, not observed cash returns.",
        }
    )
    return figure


def plot_drawdown_periods(
    returns: pd.Series, top: int = 10, ax: Any = None, **kwargs: Any
) -> go.Figure:
    supplied = _return_series(returns)
    table = timeseries.gen_drawdown_table(supplied, top=top)
    figure = _figure("Largest drawdown episodes", "Wealth multiple", ax, kwargs)
    _line(figure, ep.cum_returns(supplied, starting_value=1), "Wealth")
    for _, row in table.dropna(subset=["Valley date"]).iterrows():
        peak = supplied.index[0] if pd.isna(row["Peak date"]) else row["Peak date"]
        recovery = supplied.index[-1] if pd.isna(row["Recovery date"]) else row["Recovery date"]
        figure.add_vrect(
            x0=peak.isoformat(),
            x1=recovery.isoformat(),
            fillcolor="salmon",
            opacity=0.2,
            line_width=0,
        )
    _time_axis(figure, supplied.index)
    figure.update_layout(
        meta={
            "analysis": "legacy_returns",
            "initial_capital_included": True,
            "description": "Shared native episodes sorted by depth. A pre-observation initial peak has no invented timestamp; shading begins at the first observed point. Unrecovered episodes end at the final observed point. Flat/gaining samples do not invent zero-depth episodes.",
        }
    )
    return figure


def plot_monthly_returns_heatmap(returns: pd.Series, ax: Any = None, **kwargs: Any) -> go.Figure:
    supplied = _return_series(returns)
    months = ep.aggregate_returns(supplied, "monthly").unstack().reindex(columns=range(1, 13))
    figure = _figure("Monthly returns (%)", "Year", ax, kwargs)
    figure.update_layout(width=max(650, figure.layout.width or 0), autosize=False)
    figure.add_heatmap(
        x=[calendar.month_abbr[i] for i in range(1, 13)],
        y=months.index.astype(str).tolist(),
        z=(months.round(3) * 100).to_numpy(),
        zmid=0,
        colorscale="RdYlGn",
        texttemplate="%{z:.1f}",
        textfont_size=14,
        showscale=False,
        hovertemplate="%{y} %{x}: %{z:.3g}%<extra></extra>",
    )
    figure.update_yaxes(type="category")
    figure.update_layout(
        meta={
            "analysis": "legacy_returns",
            "units": "percent",
            "description": "Shared civil-month return products, rounded to three decimal return units before percentage display as in Pyfolio. Absent months remain blank, not invented zero returns. An observed all-NaN month retains the legacy neutral-product convention. Scroll horizontally on narrow screens to retain readable monthly values.",
        }
    )
    return figure


def plot_annual_returns(returns: pd.Series, ax: Any = None, **kwargs: Any) -> go.Figure:
    supplied = _return_series(returns)
    values = ep.aggregate_returns(supplied, "yearly").sort_index(ascending=False) * 100
    figure = _figure("Annual returns", "Year", ax, kwargs)
    figure.add_bar(
        x=values.tolist(), y=values.index.astype(str).tolist(), orientation="h", name="Return"
    )
    figure.add_vline(x=values.mean(), line_dash="dash")
    figure.add_vline(x=0)
    figure.update_xaxes(title_text="Return (%)")
    figure.update_yaxes(type="category")
    figure.update_layout(
        meta={
            "analysis": "legacy_returns",
            "units": "percent",
            "description": "Shared observed civil-year products, including partial first/last years. These are actual period returns, not annualized CAGR. Dashed line is their arithmetic mean.",
        }
    )
    return figure


def plot_monthly_returns_dist(returns: pd.Series, ax: Any = None, **kwargs: Any) -> go.Figure:
    supplied = _return_series(returns)
    values = ep.aggregate_returns(supplied, "monthly") * 100
    figure = _figure("Monthly return distribution", "Months", ax, kwargs)
    figure.add_histogram(x=values.tolist(), name="Monthly returns")
    figure.add_vline(x=values.mean(), line_dash="dash")
    figure.add_vline(x=0)
    figure.update_xaxes(title_text="Monthly return (%)")
    figure.update_layout(
        meta={
            "analysis": "legacy_returns",
            "units": "percent",
            "description": "Actual shared civil-month products, including partial months. Histogram presentation/bins differ from Seaborn; the underlying sample is retained.",
        }
    )
    return figure


def plot_return_quantiles(
    returns: pd.Series,
    live_start_date: Any = None,
    ax: Any = None,
    **kwargs: Any,
) -> go.Figure:
    supplied = _return_series(returns)
    split = None if live_start_date is None else _live_boundary(live_start_date, supplied.index)
    figure = _figure("Return quantiles", "Return (%)", ax, kwargs)
    samples = [("Backtest", supplied if split is None else supplied.loc[supplied.index < split])]
    if split is not None:
        samples.append(("Live", supplied.loc[supplied.index >= split]))
    for label, sample in samples:
        for period in ("daily", "weekly", "monthly"):
            values = sample if period == "daily" else ep.aggregate_returns(sample, period)
            if label == "Backtest":
                figure.add_box(
                    y=(values * 100).tolist(),
                    name=period.title(),
                    boxpoints=False,
                    quartilemethod="linear",
                )
            else:
                figure.add_scatter(
                    x=[period.title()] * len(values),
                    y=(values * 100).tolist(),
                    name="Live " + period.title(),
                    mode="markers",
                    marker_symbol="diamond",
                )
    figure.update_layout(
        meta={
            "analysis": "legacy_returns",
            "units": "percent",
            "description": "Backtest daily/ISO-week/civil-month samples use shared return products; live observations overlay as points. Aggregation occurs independently within each live split, retaining partial boundary weeks/months. Missing raw daily values remain undefined. Plotly boxes replace Seaborn artists.",
        }
    )
    return figure


def plot_perf_stats(
    returns: pd.Series,
    factor_returns: pd.Series,
    ax: Any = None,
    *,
    periods_per_year: int = 252,
    n_samples: int = 1000,
    random_seed: int | None = None,
) -> go.Figure:
    figure = _figure("IID metric sampling distributions", "Metric", ax, {})
    values = timeseries.perf_stats_bootstrap(
        returns,
        factor_returns=factor_returns,
        return_stats=False,
        periods_per_year=periods_per_year,
        n_samples=n_samples,
        random_seed=random_seed,
    ).drop(columns="Kurtosis")
    for label in values:
        figure.add_box(x=values[label].tolist(), name=label, orientation="h", boxpoints=False)
    figure.update_layout(
        height=max(500, len(values.columns) * 36 + 150),
        meta={
            "analysis": "legacy_returns",
            "periods_per_year": periods_per_year,
            "n_samples": n_samples,
            "random_seed": random_seed,
            "description": "Actual IID metric draws, with Kurtosis omitted as in Pyfolio. Row scales differ by named metric; ratios are dimensionless and return metrics use decimal returns. These are sampling summaries, not calibrated confidence coverage; nonfinite/undefined draws are not zero risk.",
        },
    )
    return figure
