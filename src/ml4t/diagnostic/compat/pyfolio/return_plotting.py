"""Familiar return-report presentation using existing metrics and forecast kernels."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from html import escape
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from ml4t.diagnostic.compat import empyrical as ep
from ml4t.diagnostic.compat._returns import as_array
from ml4t.diagnostic.signal.quantile import _sample_return_statistics

from . import timeseries
from .txn import _dated

if TYPE_CHECKING:
    import plotly.graph_objects as go


def _return_series(returns: pd.Series) -> pd.Series:
    if not isinstance(returns, pd.Series):
        raise TypeError("returns must be a dated pandas Series")
    _dated(returns, "returns")
    if returns.empty or not returns.index.is_unique or not returns.index.is_monotonic_increasing:
        raise ValueError("return clock must be nonempty, unique and chronological")
    if np.isinf(as_array(returns)).any():
        raise ValueError("return observations must not contain infinity")
    return returns.copy(deep=True)


def _live_boundary(value: Any, index: pd.DatetimeIndex) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp):
        raise ValueError("live_start_date must be nonmissing")
    timestamp = (
        timestamp.tz_localize("UTC") if timestamp.tzinfo is None else timestamp.tz_convert("UTC")
    )
    return timestamp.tz_localize(None) if index.tz is None else timestamp.tz_convert(index.tz)


def _time_axis(figure: go.Figure, index: pd.DatetimeIndex) -> None:
    exact = [stamp.isoformat() for stamp in index]
    selected = sorted({0, len(index) // 2, len(index) - 1})
    precise = (index.as_unit("ns").asi8 % 1_000_000 != 0).any()
    figure.update_xaxes(
        type="category",
        categoryorder="array",
        categoryarray=exact,
        tickmode="array",
        tickvals=[exact[position] for position in selected],
        ticktext=[
            str(position + 1) if precise else index[position].strftime("%b %d")
            for position in selected
        ],
        tickangle=0,
        title_text="Observation (time on hover)" if precise else "Observed date",
    )


def _set_legend(figure: go.Figure, location: str | int | None) -> None:
    from .stress_sheet import _LOCATIONS

    if isinstance(location, int) and 0 <= location < len(_LOCATIONS):
        location = _LOCATIONS[location]
    if location is not None and location not in _LOCATIONS:
        raise ValueError("legend_loc must be a supported location or None")
    figure.update_layout(showlegend=location is not None)
    if location not in (None, "best"):
        x = 0 if "left" in location else 1 if "right" in location else 0.5
        y = 1 if "upper" in location else 0 if "lower" in location else 0.5
        figure.update_layout(
            legend={
                "orientation": "v",
                "x": x,
                "y": y,
                "xanchor": "left" if x == 0 else "right" if x == 1 else "center",
                "yanchor": "top" if y == 1 else "bottom" if y == 0 else "middle",
            }
        )


def plot_returns(returns: pd.Series, live_start_date: Any = None, ax: Any = None) -> go.Figure:
    """Render the original raw return sample; NaNs remain gaps, not cash returns."""
    import plotly.graph_objects as go

    from .plotting import _figure

    supplied = _return_series(returns)
    split = None if live_start_date is None else _live_boundary(live_start_date, supplied.index)
    samples = (
        [("Backtest", supplied)]
        if split is None
        else [
            ("Backtest", supplied.loc[supplied.index < split]),
            ("Live", supplied.loc[supplied.index >= split]),
        ]
    )
    figure = _figure("Observed returns", "Periodic return", ax, {})
    for label, sample in samples:
        figure.add_trace(
            go.Scatter(
                x=[stamp.isoformat() for stamp in sample.index],
                y=as_array(sample).tolist(),
                name=label,
                mode="lines",
                connectgaps=False,
            )
        )
    _time_axis(figure, supplied.index)
    figure.update_yaxes(tickformat=".1%")
    figure.update_layout(
        meta={
            "analysis": "legacy_returns",
            "units": "decimal periodic returns",
            "description": "Actual original return observations and optional live split; missing values remain gaps. Display spacing counts observations, not elapsed sessions.",
        }
    )
    return figure


def plot_rolling_returns(
    returns: pd.Series,
    factor_returns: pd.Series | None = None,
    live_start_date: Any = None,
    logy: bool = False,
    cone_std: float | tuple[float, ...] | None = None,
    legend_loc: str | int | None = "best",
    volatility_match: bool = False,
    cone_function: Callable = timeseries.forecast_cone_bootstrap,
    ax: Any = None,
    *,
    random_seed: int | None = None,
    num_samples: int = 1000,
    **kwargs: Any,
) -> go.Figure:
    """Plot original full-clock wealth, optional volatility matching and live cones.

    The name is historical: this is a cumulative path, not a rolling CAGR.
    Missing returns are neutral only in compounding. A custom cone sees a copy
    of pre-live observations and their actual final wealth, never future returns.
    Plotly categorical dates preserve exact keys with observation-order spacing.
    """
    import plotly.graph_objects as go

    from .plotting import _figure

    supplied = _return_series(returns)
    if not all(isinstance(flag, bool | np.bool_) for flag in (logy, volatility_match)):
        raise TypeError("logy and volatility_match must be boolean")
    benchmark = None
    if factor_returns is not None:
        benchmark = _return_series(factor_returns).reindex(supplied.index)
        if not supplied.index.isin(factor_returns.index).all():
            raise ValueError("factor_returns must cover every return date")
    if volatility_match:
        if benchmark is None:
            raise ValueError("volatility_match requires factor_returns")
        strategy_std = _sample_return_statistics(as_array(supplied.dropna()))[1]
        benchmark_std = _sample_return_statistics(as_array(benchmark.dropna()))[1]
        if not np.isfinite(strategy_std) or strategy_std <= 0 or not np.isfinite(benchmark_std):
            raise ValueError("volatility matching requires defined positive strategy dispersion")
        supplied = supplied / strategy_std * benchmark_std
    wealth = ep.cum_returns(supplied, starting_value=1.0)
    split = None if live_start_date is None else _live_boundary(live_start_date, supplied.index)
    training = supplied if split is None else supplied.loc[supplied.index < split]
    live = supplied.iloc[:0] if split is None else supplied.loc[supplied.index >= split]
    title = (
        "Volatility-matched wealth"
        if volatility_match
        else "Cumulative wealth (log)"
        if logy
        else "Cumulative wealth"
    )
    figure = _figure(title, "Wealth multiple", ax, kwargs)
    _set_legend(figure, legend_loc)
    if kwargs:
        raise TypeError(f"Unknown Plotly layout options: {', '.join(sorted(kwargs))}")
    if benchmark is not None:
        baseline = ep.cum_returns(benchmark, starting_value=1.0)
        figure.add_trace(
            go.Scatter(
                x=[stamp.isoformat() for stamp in baseline.index],
                y=as_array(baseline).tolist(),
                name=escape(str(factor_returns.name or "Benchmark")),
                mode="lines",
                connectgaps=False,
            )
        )
    for label, sample in [
        ("Backtest", wealth.loc[training.index]),
        *([] if live.empty else [("Live", wealth.loc[live.index])]),
    ]:
        figure.add_trace(
            go.Scatter(
                x=[stamp.isoformat() for stamp in sample.index],
                y=as_array(sample).tolist(),
                name=label,
                mode="lines",
                connectgaps=False,
            )
        )
    bounds_record = None
    if not live.empty and cone_std is not None:
        if training.empty:
            raise ValueError("a cone requires at least one pre-live observation")
        levels = [cone_std] if isinstance(cone_std, int | float | np.floating) else list(cone_std)
        start = float(wealth.loc[training.index[-1]])
        options = {"cone_std": levels, "starting_value": start}
        if cone_function is timeseries.forecast_cone_bootstrap:
            options.update(num_samples=num_samples, random_seed=random_seed)
        elif random_seed is not None:
            raise ValueError("custom cone functions own their seed; omit random_seed")
        bounds = cone_function(training.copy(deep=True), len(live), **options)
        required = [sign * float(level) for level in levels for sign in (-1, 1)]
        if (
            not isinstance(bounds, pd.DataFrame)
            or len(bounds) != len(live)
            or any(column not in bounds for column in required)
        ):
            raise ValueError(
                "cone output requires both bounds for each requested level and every live row"
            )
        if not np.isfinite(bounds[required].to_numpy(dtype=float)).all():
            raise ValueError("cone bounds must be finite")
        bounds_record = {str(column): bounds[column].tolist() for column in bounds}
        for level in levels:
            for sign in (-1, 1):
                figure.add_trace(
                    go.Scatter(
                        x=[stamp.isoformat() for stamp in live.index],
                        y=bounds[sign * float(level)].tolist(),
                        name=f"{sign * float(level):+g} std forecast",
                        mode="lines",
                        line_width=0.5,
                        showlegend=False,
                        fill="tonexty" if sign == 1 else None,
                        fillcolor="rgba(70,130,180,0.18)",
                    )
                )
    _time_axis(figure, supplied.index)
    figure.update_yaxes(type="log" if logy else "linear")
    figure.add_hline(y=1.0, line_dash="dash")
    figure.update_layout(
        meta={
            "analysis": "legacy_returns",
            "units": "wealth multiple",
            "training_observations": len(training),
            "live_observations": len(live),
            "cone_bounds": bounds_record,
            "description": "Full-clock products start at one, not net cumulative returns. NaNs are neutral product factors, not imputed observations. Cones sample only actual pre-live returns; bounds are mean +/- population std, not confidence coverage. Date spacing counts observations; exact original timestamps remain on hover.",
        },
    )
    return figure


def show_perf_stats(
    returns: pd.Series,
    factor_returns: pd.Series | None = None,
    positions: pd.DataFrame | None = None,
    transactions: pd.DataFrame | None = None,
    turnover_denom: str = "AGB",
    live_start_date: Any = None,
    bootstrap: bool = False,
    header_rows: Mapping[str, Any] | None = None,
    return_df: bool = False,
    *,
    periods_per_year: int = 252,
    n_samples: int = 1000,
    random_seed: int | None = None,
) -> pd.DataFrame | str:
    """Display actual split statistics as pandas or escaped HTML, not terminal printouts.

    Selected return rows remain the sample; the benchmark is separately aligned
    by the existing regression. Bootstrap summary columns are retained instead
    of the original no-live display's destructive Backtest-column reindex.
    """
    supplied = _return_series(returns)
    if not all(isinstance(flag, bool | np.bool_) for flag in (bootstrap, return_df)):
        raise TypeError("bootstrap and return_df must be boolean")
    cutoff = None if live_start_date is None else _live_boundary(live_start_date, supplied.index)
    samples = (
        [("Backtest", supplied)]
        if cutoff is None
        else [
            ("In-sample", supplied.loc[supplied.index < cutoff]),
            ("Out-of-sample", supplied.loc[supplied.index >= cutoff]),
            ("All", supplied),
        ]
    )
    frames = {}
    for label, sample in samples:
        selected = []
        for frame in (positions, transactions):
            if frame is None or label in {"Backtest", "All"}:
                selected.append(frame)
            else:
                boundary = _live_boundary(live_start_date, frame.index)
                selected.append(
                    frame.loc[frame.index < boundary]
                    if label == "In-sample"
                    else frame.loc[frame.index >= boundary]
                )
        options = {
            "factor_returns": factor_returns,
            "positions": selected[0],
            "transactions": selected[1],
            "turnover_denom": turnover_denom,
            "periods_per_year": periods_per_year,
        }
        if bootstrap:
            if factor_returns is not None:
                if not sample.index.isin(factor_returns.index).all():
                    raise ValueError(
                        "bootstrap factor_returns must cover every sampled return date"
                    )
                options["factor_returns"] = factor_returns.reindex(sample.index)
            options.update(n_samples=n_samples, random_seed=random_seed)
        frames[label] = (timeseries.perf_stats_bootstrap if bootstrap else timeseries.perf_stats)(
            sample, **options
        )
    table = (
        next(iter(frames.values())).copy()
        if bootstrap and cutoff is None
        else pd.concat(frames, axis=1)
    )
    table = table.astype(object)
    percentages = {
        "Annual return",
        "Cumulative returns",
        "Annual volatility",
        "Max drawdown",
        "Daily value at risk",
        "Daily turnover",
    }
    for column in table:
        for label in table.index:
            if label in percentages:
                table.loc[label, column] = str(np.round(table.loc[label, column] * 100, 3)) + "%"
    headers = {str(key): str(value) for key, value in (header_rows or {}).items()}
    headers.update(
        {
            "Start date": supplied.index[0].strftime("%Y-%m-%d"),
            "End date": supplied.index[-1].strftime("%Y-%m-%d"),
        }
    )
    if cutoff is None:
        headers["Total months"] = str(len(supplied) // 21)
    else:
        headers["In-sample months"] = str(len(samples[0][1]) // 21)
        headers["Out-of-sample months"] = str(len(samples[1][1]) // 21)
    table.attrs["diagnostic"] = {
        "header_rows": headers,
        "periods_per_year": periods_per_year,
        "bootstrap": bootstrap,
        "sampling_summary_not_confidence_coverage": bootstrap,
    }
    if return_df:
        return table
    header_html = "".join(
        f"<li>{escape(key)}: {escape(value)}</li>" for key, value in headers.items()
    )
    return "<ul>" + header_html + "</ul>" + table.to_html(escape=True)
