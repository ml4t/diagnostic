"""Plot familiar supplied analytical tables with native Plotly presentation.

Figures replace Matplotlib axes. Call ``show`` or ``write_html`` explicitly,
including when a notebook cell suppresses its final expression with a semicolon.
No factor sample, asset count or uncertainty is inferred from a summary table.
"""

from __future__ import annotations

from html import escape
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from ml4t.diagnostic.metrics.ic_inference import compute_ic_summary_stats

if TYPE_CHECKING:
    import plotly.graph_objects as go

    from ml4t.diagnostic.results.signal_results import SignalICResult

__all__ = [
    "plot_ic_ts",
    "plot_ic_hist",
    "plot_ic_qq",
    "plot_monthly_ic_heatmap",
    "plot_ic_by_group",
    "plot_quantile_returns_bar",
    "plot_quantile_returns_violin",
    "plot_mean_quantile_returns_spread_time_series",
    "plot_cumulative_returns",
    "plot_cumulative_returns_by_quantile",
    "plot_top_bottom_quantile_turnover",
    "plot_factor_rank_auto_correlation",
]


def _axes(ax: Any) -> None:
    if ax is not None:
        raise TypeError(
            "Matplotlib ax is incompatible; use the returned Plotly figure and show/write_html"
        )


def _finish(figure: go.Figure) -> go.Figure:
    figure.update_layout(width=None, autosize=True, height=500, font_size=14, title_font_size=16)
    for annotation in figure.layout.annotations:
        annotation.font.size = 13
    return figure


def _figure(title: str, ylabel: str) -> go.Figure:
    from ml4t.diagnostic.visualization.core import create_base_figure

    return _finish(create_base_figure(title=escape(title), yaxis_title=ylabel))


def _dates(index: pd.Index) -> list:
    return [value.isoformat() if isinstance(value, pd.Timestamp) else value for value in index]


def _ic_model(ic: pd.DataFrame) -> SignalICResult:
    from ml4t.diagnostic.results.signal_results import SignalICResult

    if not isinstance(ic, pd.DataFrame) or not isinstance(ic.index, pd.DatetimeIndex):
        raise TypeError("ic must be a pandas DataFrame indexed by observation dates")
    summary = {column: compute_ic_summary_stats(ic[column].to_numpy(dtype=float)) for column in ic}
    return SignalICResult(
        ic_by_date={column: ic[column].to_numpy(dtype=float).tolist() for column in ic},
        dates=[value.isoformat() for value in ic.index],
        ic_mean={column: float(row["mean_ic"]) for column, row in summary.items()},
        ic_std={column: float(row["std_ic"]) for column, row in summary.items()},
        ic_t_stat={column: float(row["t_stat"]) for column, row in summary.items()},
        ic_p_value={column: float(row["p_value"]) for column, row in summary.items()},
        ic_positive_pct={
            column: 100 * float(row["pct_positive"]) for column, row in summary.items()
        },
        ic_ir={
            column: float(row["mean_ic"]) / float(row["std_ic"])
            if row["std_ic"] > 0
            else float("nan")
            for column, row in summary.items()
        },
    )


def plot_ic_ts(ic: pd.DataFrame, ax: Any = None) -> list[go.Figure]:
    """Render actual dated ICs and complete 22-observation rolling means."""
    from ml4t.diagnostic.visualization.signal.ic_plots import plot_ic_ts as native

    _axes(ax)
    model = _ic_model(ic)
    return [
        _finish(
            native(model, period=column, rolling_window=22, min_periods=22, show_significance=False)
        )
        for column in ic
    ]


def plot_ic_hist(ic: pd.DataFrame, ax: Any = None) -> list[go.Figure]:
    """Reuse native empirical IC distributions without filling missing observations."""
    from ml4t.diagnostic.visualization.signal.ic_plots import plot_ic_histogram

    _axes(ax)
    model = _ic_model(ic)
    return [_finish(plot_ic_histogram(model, period=column, bins=25)) for column in ic]


def plot_ic_qq(ic: pd.DataFrame, theoretical_dist: Any = None, ax: Any = None) -> list[go.Figure]:
    """Reuse fitted native Q-Q samples; the default theoretical family is Normal."""
    from ml4t.diagnostic.visualization.signal.ic_plots import plot_ic_qq as native

    _axes(ax)
    model = _ic_model(ic)
    return [
        _finish(native(model, period=column, fitted=True, theoretical_dist=theoretical_dist))
        for column in ic
    ]


def plot_monthly_ic_heatmap(mean_monthly_ic: pd.DataFrame, ax: Any = None) -> list[go.Figure]:
    """Render supplied civil-month means using the existing native month heatmap."""
    from ml4t.diagnostic.visualization.signal.ic_plots import plot_monthly_ic_heatmap as native

    _axes(ax)
    if pd.MultiIndex.from_arrays(
        [mean_monthly_ic.index.year, mean_monthly_ic.index.month]
    ).has_duplicates:
        raise ValueError("monthly IC requires one supplied row per civil month")
    model = _ic_model(mean_monthly_ic)
    return [_finish(native(model, period=column)) for column in mean_monthly_ic]


def plot_ic_by_group(ic_group: pd.DataFrame, ax: Any = None) -> go.Figure:
    """Render supplied group means without treating them as per-asset observations."""
    import plotly.graph_objects as go

    _axes(ax)
    figure = _figure("Mean IC by group", "Mean IC")
    for column in ic_group:
        figure.add_trace(
            go.Bar(x=list(ic_group.index), y=ic_group[column].tolist(), name=escape(str(column)))
        )
    figure.update_layout(
        barmode="group", meta={"units": "dimensionless", "sample": "supplied group means"}
    )
    return figure


def _limits(
    figure: go.Figure, frame: pd.DataFrame, percentiles: tuple[float, float] | None
) -> None:
    if percentiles is None:
        return
    if len(percentiles) != 2 or not 0 <= percentiles[0] < percentiles[1] <= 100:
        raise ValueError("ylim_percentiles must be an increasing pair within0..100")
    values = frame.to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    if len(values):
        figure.update_yaxes(range=(np.percentile(values, percentiles) * 10000).tolist())


def plot_quantile_returns_bar(
    mean_ret_by_q: pd.DataFrame,
    by_group: bool = False,
    ylim_percentiles: tuple[float, float] | None = None,
    ax: Any = None,
) -> go.Figure | list[go.Figure]:
    """Render supplied means in bps; absent errors/counts remain absent."""
    import plotly.graph_objects as go

    _axes(ax)
    if not isinstance(by_group, bool | np.bool_):
        raise TypeError("by_group must be boolean")
    if by_group:
        figures = []
        for group, sample in mean_ret_by_q.groupby(level="group", observed=True):
            figure = plot_quantile_returns_bar(sample.droplevel("group"))
            _limits(figure, mean_ret_by_q, ylim_percentiles)
            figure.update_layout(title=escape(f"Mean quantile returns: {group}"))
            figures.append(figure)
        return figures
    figure = _figure("Mean return by factor quantile", "Mean return (bps)")
    for column in mean_ret_by_q:
        figure.add_trace(
            go.Bar(
                x=list(mean_ret_by_q.index),
                y=(10000 * mean_ret_by_q[column]).tolist(),
                name=escape(str(column)),
            )
        )
    _limits(figure, mean_ret_by_q, ylim_percentiles)
    figure.update_layout(
        barmode="group",
        meta={
            "units": "bps",
            "sample": "supplied quantile means",
            "counts": "unavailable",
            "errors": "unavailable",
        },
    )
    return figure


def plot_quantile_returns_violin(
    return_by_q: pd.DataFrame,
    ylim_percentiles: tuple[float, float] | None = None,
    ax: Any = None,
) -> go.Figure:
    """Render the supplied dated bucket observations, not synthetic distributions."""
    import plotly.graph_objects as go

    _axes(ax)
    buckets = return_by_q.index.get_level_values("factor_quantile")
    figure = _figure("Observed returns by factor quantile", "Return (bps)")
    for column in return_by_q:
        figure.add_trace(
            go.Violin(
                x=list(buckets),
                y=(10000 * return_by_q[column]).tolist(),
                name=escape(str(column)),
                box_visible=True,
                spanmode="hard",
            )
        )
    _limits(figure, return_by_q, ylim_percentiles)
    figure.update_layout(
        violinmode="group", meta={"units": "bps", "sample": "supplied dated bucket means"}
    )
    return figure


def plot_mean_quantile_returns_spread_time_series(
    mean_returns_spread: pd.Series | pd.DataFrame,
    std_err: pd.Series | pd.DataFrame | None = None,
    bandwidth: float = 1,
    ax: Any = None,
) -> go.Figure | list[go.Figure]:
    """Render supplied exposure-scaled spreads and descriptive SE bands in bps."""
    import plotly.graph_objects as go

    _axes(ax)
    if isinstance(mean_returns_spread, pd.DataFrame):
        return [
            plot_mean_quantile_returns_spread_time_series(
                mean_returns_spread[column], None if std_err is None else std_err[column], bandwidth
            )
            for column in mean_returns_spread
        ]
    if not np.isfinite(bandwidth) or bandwidth < 0:
        raise ValueError("bandwidth must be nonnegative and finite")
    values = mean_returns_spread * 10000
    figure = _figure(f"Quantile spread: {mean_returns_spread.name}", "Spread return (bps)")
    dates = _dates(values.index)
    figure.add_trace(go.Scatter(x=dates, y=values.tolist(), name="Observed spread", mode="lines"))
    figure.add_trace(
        go.Scatter(
            x=dates, y=values.rolling(22).mean().tolist(), name="22-observation mean", mode="lines"
        )
    )
    if std_err is not None:
        errors = std_err.reindex(values.index) * (10000 * bandwidth)
        figure.add_trace(
            go.Scatter(
                x=dates,
                y=(values - errors).tolist(),
                name="Lower SE band",
                mode="lines",
                line_width=0,
                showlegend=False,
            )
        )
        figure.add_trace(
            go.Scatter(
                x=dates,
                y=(values + errors).tolist(),
                name="Descriptive SE band",
                mode="lines",
                line_width=0,
                fill="tonexty",
            )
        )
    figure.update_layout(
        meta={
            "units": "bps",
            "bandwidth": bandwidth,
            "exposure": "unchanged supplied spread",
            "inference": "descriptive SE, not confidence interval",
        }
    )
    return figure


def plot_cumulative_returns(
    factor_returns: pd.Series,
    period: str | pd.Timedelta,
    freq: Any = None,
    title: str | None = None,
    ax: Any = None,
) -> go.Figure:
    """Compound supplied raw returns; period/freq do not create holdings or daily accounting."""
    import plotly.graph_objects as go

    from ml4t.diagnostic.compat.alphalens.performance import cumulative_returns

    _axes(ax)
    wealth = cumulative_returns(factor_returns)
    figure = _figure(title or f"Cumulative factor wealth ({period})", "Wealth multiple")
    figure.add_trace(
        go.Scatter(x=_dates(wealth.index), y=wealth.tolist(), name="Wealth", mode="lines")
    )
    figure.update_layout(
        meta={
            "units": "wealth multiple",
            "period": str(period),
            "freq": str(freq),
            "accounting": "supplied observation-order product",
        }
    )
    return figure


def plot_cumulative_returns_by_quantile(
    quantile_returns: pd.Series,
    period: str | pd.Timedelta,
    freq: Any = None,
    ax: Any = None,
) -> go.Figure:
    """Compound each supplied bucket series with the same native wealth kernel."""
    import plotly.graph_objects as go

    from ml4t.diagnostic.compat.alphalens.performance import cumulative_returns

    _axes(ax)
    frame = quantile_returns.unstack("factor_quantile")
    wealth = cumulative_returns(frame)
    figure = _figure(f"Cumulative quantile wealth ({period})", "Wealth multiple")
    for bucket in reversed(wealth.columns):
        figure.add_trace(
            go.Scatter(
                x=_dates(wealth.index),
                y=wealth[bucket].tolist(),
                name=escape(str(bucket)),
                mode="lines",
            )
        )
    figure.update_layout(
        meta={
            "units": "wealth multiple",
            "period": str(period),
            "freq": str(freq),
            "scale": "linear presentation; no synthetic holdings",
        }
    )
    return figure


def plot_top_bottom_quantile_turnover(
    quantile_turnover: pd.DataFrame,
    period: int = 1,
    ax: Any = None,
) -> go.Figure:
    """Render the actual highest/lowest supplied bucket membership turnover."""
    import plotly.graph_objects as go

    _axes(ax)
    figure = _figure(f"Quantile turnover ({period} observations)", "Turnover fraction")
    for bucket in dict.fromkeys([max(quantile_turnover.columns), min(quantile_turnover.columns)]):
        figure.add_trace(
            go.Scatter(
                x=_dates(quantile_turnover.index),
                y=quantile_turnover[bucket].tolist(),
                name=escape(str(bucket)),
                mode="lines",
            )
        )
    return figure


def plot_factor_rank_auto_correlation(
    factor_autocorrelation: pd.Series,
    period: int = 1,
    ax: Any = None,
) -> go.Figure:
    """Render supplied rank correlations and their available-sample mean."""
    import plotly.graph_objects as go

    _axes(ax)
    figure = _figure(f"Factor rank correlation ({period} observations)", "Rank correlation")
    figure.add_trace(
        go.Scatter(
            x=_dates(factor_autocorrelation.index),
            y=factor_autocorrelation.tolist(),
            name="Observed correlation",
            mode="lines",
        )
    )
    mean = factor_autocorrelation.mean()
    if np.isfinite(mean):
        figure.add_hline(y=mean)
    return figure
