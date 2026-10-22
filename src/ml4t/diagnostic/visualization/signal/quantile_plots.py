"""Quantile returns visualization plots.

This module provides interactive Plotly visualizations for quantile analysis:
- plot_quantile_returns_bar: Mean returns by quantile (bar chart)
- plot_quantile_returns_violin: Return distributions by quantile
- plot_cumulative_returns: Cumulative returns by quantile over time
- plot_spread_timeseries: Top-bottom spread time series
"""

from __future__ import annotations

from html import escape
from numbers import Real
from typing import TYPE_CHECKING, Any

import numpy as np
import plotly.graph_objects as go
import polars as pl

from ml4t.diagnostic.utils.formatting import format_finite
from ml4t.diagnostic.visualization.core import (
    create_base_figure,
    get_quantile_colors,
    get_theme_config,
    validate_theme,
)

if TYPE_CHECKING:
    import pandas as pd

    from ml4t.diagnostic.results.signal_results import QuantileAnalysisResult


def _get_quantile_colors(n_quantiles: int, theme_config: dict[str, Any]) -> list[str]:
    """Get diverging colors for quantiles (red -> green progression)."""
    return get_quantile_colors(n_quantiles, theme_config)


def plot_quantile_returns_bar(
    quantile_result: QuantileAnalysisResult,
    period: str | None = None,
    show_error_bars: bool = True,
    show_spread: bool = True,
    theme: str | None = None,
    width: int | None = None,
    height: int | None = None,
) -> go.Figure:
    """Plot mean returns by quantile as a bar chart.

    Parameters
    ----------
    quantile_result : QuantileAnalysisResult
        Quantile analysis result from a signal analysis workflow.
    period : str | None
        Period to plot (e.g., "1D", "5D"). If None, uses first period.
    show_error_bars : bool, default True
        Show defined standard errors. Missing errors have no cap; known zeros remain defined.
    show_spread : bool, default True
        Show top-bottom spread annotation
    theme : str | None
        Plot theme (default, dark, print, presentation)
    width : int | None
        Figure width in pixels
    height : int | None
        Figure height in pixels

    Returns
    -------
    go.Figure
        Interactive Plotly figure

    Examples
    --------
    >>> quantile_result = analyzer.compute_quantile_analysis()
    >>> fig = plot_quantile_returns_bar(quantile_result, period="5D")
    >>> fig.show()
    """
    theme = validate_theme(theme)
    theme_config = get_theme_config(theme)

    # Get period data
    periods = quantile_result.periods
    if period is None:
        period = periods[0]
    elif period not in periods:
        raise ValueError(f"Period '{period}' not found. Available: {periods}")

    n_quantiles = quantile_result.n_quantiles
    quantile_labels = quantile_result.quantile_labels
    mean_returns = quantile_result.mean_returns[period]
    std_returns = quantile_result.std_returns[period]
    counts = quantile_result.count_by_quantile

    # Get colors
    colors = _get_quantile_colors(n_quantiles, theme_config)

    # Create figure
    period_lines = [escape(period[index : index + 24]) for index in range(0, len(period), 24)]
    fig = create_base_figure(
        title="Mean Returns by Quantile<br>(" + "<br>".join(period_lines) + ")",
        xaxis_title="Quantile",
        yaxis_title="Mean Forward Return",
        width=width or theme_config["defaults"]["bar_height"],
        height=height or theme_config["defaults"]["bar_height"],
        theme=theme,
    )
    label_size = max(14, fig.layout.font.size or 14)
    fig.update_layout(font_size=label_size)
    fig.update_layout(
        margin_t=max(
            fig.layout.margin.t or 0,
            (len(period_lines) + 1) * (fig.layout.title.font.size or 16) * 1.3 + 24,
        )
    )
    fig.update_xaxes(tickfont_size=label_size, title_font_size=label_size)
    fig.update_yaxes(tickfont_size=label_size, title_font_size=label_size)

    # Prepare data
    x_labels = quantile_labels
    y_values = [mean_returns.get(q, 0) for q in quantile_labels]
    y_std = [std_returns.get(q, 0) for q in quantile_labels]

    # Compute standard errors
    y_stderr = []
    for q, std in zip(quantile_labels, y_std, strict=False):
        count = counts.get(q, 1)
        y_stderr.append(std / np.sqrt(count) if count > 0 and np.isfinite(std) else None)
    defined = [index for index, error in enumerate(y_stderr) if error is not None]
    mixed = 0 < len(defined) < len(y_stderr)

    # Bar chart
    fig.add_trace(
        go.Bar(
            x=x_labels,
            y=y_values,
            marker_color=colors,
            error_y={
                "type": "data",
                "array": y_stderr,
                "visible": bool(defined) and not mixed,
                "color": "gray",
            }
            if show_error_bars
            else None,
            customdata=[
                [counts[q], format_finite(error, ".4%")]
                for q, error in zip(quantile_labels, y_stderr, strict=True)
            ],
            hovertemplate=(
                "Quantile: %{x}<br>Mean Return: %{y:.4f}<br>"
                "Observations: %{customdata[0]}<br>SE: %{customdata[1]}<extra></extra>"
            ),
            name="Mean Return",
            legendgroup="quantile_means",
        )
    )
    if show_error_bars and mixed:
        # Plotly draws a zero-length cap even for null bar errors. Draw only defined SEs.
        fig.add_trace(
            go.Scatter(
                x=[x_labels[index] for index in defined],
                y=[y_values[index] for index in defined],
                mode="markers",
                marker={"size": 0, "opacity": 0},
                error_y={
                    "type": "data",
                    "array": [y_stderr[index] for index in defined],
                    "visible": True,
                    "color": "gray",
                },
                name="Defined standard errors",
                legendgroup="quantile_means",
                showlegend=False,
                hoverinfo="skip",
            )
        )
    fig.update_layout(
        meta={
            "uncertainty": "sample_standard_error",
            "return_column": period,
            "defined_error_count": len(defined),
            "missing_error_count": len(y_stderr) - len(defined),
        }
    )

    # Zero line
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)

    # Spread annotation
    if show_spread:
        spread = quantile_result.spread_mean.get(period, 0)
        spread_t = quantile_result.spread_t_stat.get(period, 0)
        spread_p = quantile_result.spread_p_value.get(period, 1)
        monotonic, direction = quantile_result.get_monotonicity(period)
        monotonic_display = (
            "N/A" if monotonic is None else "✓ " + direction if monotonic else "✗ No"
        )

        spread_text = (
            f"<b>Spread Analysis:</b><br>"
            f"Top - Bottom: {format_finite(spread, '.1%')}<br>"
            f"t-stat: {format_finite(spread_t, '.2f')} "
            f"(p={format_finite(spread_p, '.4f')})<br>"
            f"Monotonic: {monotonic_display}"
        )

        fig.add_annotation(
            text=spread_text,
            xref="paper",
            yref="paper",
            x=0,
            y=-0.25,
            showarrow=False,
            bgcolor="rgba(255,255,255,0.8)" if theme != "dark" else "rgba(50,50,50,0.8)",
            bordercolor="gray",
            borderwidth=1,
            align="left",
            xanchor="left",
            yanchor="top",
            font={"size": 14},
        )
        fig.update_layout(margin_b=max(180, fig.layout.margin.b or 0))

    # Format y-axis as percentage
    fig.update_yaxes(tickformat=".2%")

    return fig


def plot_quantile_returns_violin(
    quantile_result: QuantileAnalysisResult,
    factor_data: dict | pl.DataFrame | pd.DataFrame | None = None,
    period: str | None = None,
    show_box: bool = True,
    theme: str | None = None,
    width: int | None = None,
    height: int | None = None,
    *,
    observations: pl.DataFrame | pd.DataFrame | None = None,
    ylim_percentiles: tuple[float, float] | None = None,
) -> go.Figure:
    """Plot actual observed-date bucket means, or explicitly supplied return values.

    No normal samples are inferred from summary means/deviations. A result with
    no empirical observations requires explicit input or raises ValueError.
    The default result/report path uses dated bucket means, not asset samples.
    Plotly's KDE rendering is a presentation choice, not a numerical density
    equivalence claim; callers retain figure bandwidth/style/embedding control.

    Parameters
    ----------
    quantile_result : QuantileAnalysisResult
        Quantile analysis result from a signal analysis workflow.
    factor_data : dict or DataFrame, optional
        Explicit supplied values with quantile and {period}_fwd_return columns.
        These values are not reduced to date/bucket means. Null/nonfinite values
        are omitted, and the sample unit remains supplied_value.
    period : str | None
        Period to plot. If None, uses first period.
    show_box : bool, default True
        Show box plot inside violin
    theme : str | None
        Plot theme
    width : int | None
        Figure width
    height : int | None
        Figure height
    observations : DataFrame, optional
        Canonical by-date compute_quantile_statistics output, with return_column
        matching the selected period. Shared adjustments/counts are retained.
        Select one group before plotting grouped statistics. Cannot be combined
        with factor_data. Otherwise use the result's owned observations.
    ylim_percentiles : tuple[float, float], optional
        Ordered percentiles in [0,100] of defined selected-period observations.
        Sets visible y limits without discarding or modifying trace samples.

    Returns
    -------
    go.Figure
        Interactive Plotly figure
    """
    theme = validate_theme(theme)
    theme_config = get_theme_config(theme)

    periods = quantile_result.periods
    if not periods:
        raise ValueError("No return periods are available for empirical plotting")
    if period is None:
        period = periods[0]
    elif period not in periods:
        raise ValueError(f"Period '{period}' not found. Available: {periods}")

    if factor_data is not None and observations is not None:
        raise ValueError("supply observations or factor_data, not both")
    from ml4t.diagnostic.signal._utils import _signal_frame
    from ml4t.diagnostic.signal.quantile import _quantile_observations

    empirical = factor_data is None
    adjustment = None
    if empirical:
        data = (
            _quantile_observations(observations)
            if observations is not None
            else quantile_result.get_dataframe("observations")
        )
        if "group" in data.columns and data["group"].n_unique() > 1:
            raise ValueError("select one group before plotting empirical quantile observations")
        if data.height and period not in data["return_column"].to_list():
            raise ValueError(f"empirical observations do not contain selected period {period!r}")
        data = data.filter(pl.col("return_column") == period)
        methods = data["return_adjustment"].unique().to_list()
        if len(methods) > 1:
            raise ValueError("selected observations mix return-adjustment methods")
        adjustment = methods[0] if methods else None
        value_column = "mean"
    else:
        data = _signal_frame(
            pl.DataFrame(factor_data) if isinstance(factor_data, dict) else factor_data
        )
        canonical = f"{period}_fwd_return"
        if period in data.columns and canonical in data.columns:
            raise ValueError(f"ambiguous supplied return columns: {period} and {canonical}")
        value_column = period if period in data.columns else canonical
        if not {"quantile", value_column}.issubset(data.columns):
            raise ValueError(f"supplied return values require quantile and {value_column} columns")
    if not data["quantile"].dtype.is_numeric():
        raise TypeError("supplied quantile labels must be numeric positive integers")
    buckets = data["quantile"]
    if (
        buckets.null_count()
        or not buckets.is_finite().all()
        or not ((buckets >= 1) & (buckets == buckets.floor())).all()
    ):
        raise ValueError("supplied quantile labels must be positive integers")
    bucket_ids = quantile_result.quantile_ids or list(range(1, quantile_result.n_quantiles + 1))
    if set(buckets.to_list()).difference(bucket_ids):
        raise ValueError("observed buckets are not represented by the result's quantile labels")
    if not (data[value_column].dtype.is_numeric() or data[value_column].dtype == pl.Null):
        raise TypeError("supplied return observations must be numeric")
    data = data.with_columns(pl.col(value_column).cast(pl.Float64)).filter(
        pl.col(value_column).is_finite()
    )
    limits = None
    if ylim_percentiles is not None:
        try:
            percentiles = tuple(ylim_percentiles)
        except TypeError as exc:
            raise ValueError(
                "ylim_percentiles must contain two ordered percentiles in [0,100]"
            ) from exc
        if (
            len(percentiles) != 2
            or any(isinstance(value, bool) or not isinstance(value, Real) for value in percentiles)
            or not 0 <= percentiles[0] <= percentiles[1] <= 100
        ):
            raise ValueError("ylim_percentiles must contain two ordered percentiles in [0,100]")
        if data.height:
            limits = np.percentile(data[value_column].to_numpy(), percentiles).tolist()

    n_quantiles = quantile_result.n_quantiles
    quantile_labels = quantile_result.quantile_labels
    colors = _get_quantile_colors(n_quantiles, theme_config)

    # Create figure
    fig = create_base_figure(
        title=f"{'Observed quantile means' if empirical else 'Supplied quantile returns'} ({period})",
        xaxis_title="Quantile",
        yaxis_title="Date/bucket mean return" if empirical else "Supplied return",
        width=width or theme_config["defaults"]["bar_height"] + 200,
        height=height or theme_config["defaults"]["bar_height"],
        theme=theme,
    )

    for i, q_label in enumerate(quantile_labels):
        sample = data.filter(pl.col("quantile") == bucket_ids[i])
        custom = None
        if empirical:
            sample = sample.sort("date")
            custom = list(
                zip(
                    sample["date"].to_pandas().astype(str).to_list(),
                    sample["n_obs"].to_list(),
                    strict=True,
                )
            )
        fig.add_trace(
            go.Violin(
                y=sample[value_column].to_numpy(),
                name=q_label,
                customdata=custom,
                hovertemplate="Date: %{customdata[0]}<br>Mean: %{y:.4%}<br>Assets: %{customdata[1]}<extra>%{fullData.name}</extra>"
                if empirical
                else None,
                box_visible=show_box,
                meanline_visible=True,
                spanmode="hard",
                fillcolor=colors[i],
                line_color=colors[i],
                opacity=0.6,
            )
        )
    if data.is_empty():
        fig.add_annotation(text="No defined empirical observations", showarrow=False)
    fig.update_layout(
        meta={
            "sample_unit": "date_bucket_mean" if empirical else "supplied_value",
            "return_unit": "decimal",
            "synthetic": False,
            "return_adjustment": adjustment,
        }
    )
    if limits is not None:
        fig.update_yaxes(range=limits)

    # Zero line
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)

    # Format y-axis
    fig.update_yaxes(tickformat=".2%")
    fig.update_layout(showlegend=False)

    return fig


def plot_cumulative_returns(
    quantile_result: QuantileAnalysisResult,
    period: str | None = None,
    show_spread: bool = True,
    theme: str | None = None,
    width: int | None = None,
    height: int | None = None,
) -> go.Figure:
    """Plot cumulative returns by quantile over time.

    Parameters
    ----------
    quantile_result : QuantileAnalysisResult
        Quantile analysis result with cumulative_returns computed.
    period : str | None
        Period to plot. If None, uses first period.
    show_spread : bool, default True
        Show top-bottom spread as shaded area
    theme : str | None
        Plot theme
    width : int | None
        Figure width
    height : int | None
        Figure height

    Returns
    -------
    go.Figure
        Interactive Plotly figure

    Raises
    ------
    ValueError
        If cumulative_returns not available in result
    """
    theme = validate_theme(theme)
    theme_config = get_theme_config(theme)

    # Check for cumulative returns
    if quantile_result.cumulative_returns is None:
        raise ValueError(
            "Cumulative returns not computed. Set cumulative_returns=True in SignalConfig."
        )

    # Get period data
    periods = quantile_result.periods
    if period is None:
        period = periods[0]
    elif period not in periods:
        raise ValueError(f"Period '{period}' not found. Available: {periods}")

    n_quantiles = quantile_result.n_quantiles
    quantile_labels = quantile_result.quantile_labels
    colors = _get_quantile_colors(n_quantiles, theme_config)

    cum_returns = quantile_result.cumulative_returns[period]
    dates_raw = quantile_result.cumulative_dates
    precise_keys = False

    # Convert dates or create fallback indices
    if dates_raw is not None and len(dates_raw) > 0:
        if isinstance(dates_raw[0], str):
            import pandas as pd

            try:
                timestamps = [pd.Timestamp(value) for value in dates_raw]
                dates: list[Any] = [value.isoformat() for value in timestamps]
                precise_keys = any(value.value % 1_000_000 for value in timestamps)
            except ValueError:
                dates = list(dates_raw)
        else:
            dates = list(dates_raw)
    else:
        # Fallback to integer indices if no dates provided
        max_len = max(len(v) for v in cum_returns.values()) if cum_returns else 0
        dates = list(range(max_len))

    # Create figure
    fig = create_base_figure(
        title=f"Cumulative Returns by Quantile ({period})",
        xaxis_title="Date",
        yaxis_title="Cumulative Return",
        width=width or theme_config["defaults"]["line_height"] + 300,
        height=height or theme_config["defaults"]["line_height"],
        theme=theme,
    )

    # Plot each quantile
    for i, q_label in enumerate(quantile_labels):
        cum_ret = cum_returns.get(q_label, [])
        if len(cum_ret) == 0:
            continue

        fig.add_trace(
            go.Scatter(
                x=dates[: len(cum_ret)],
                y=cum_ret,
                mode="lines",
                name=q_label,
                line={"color": colors[i], "width": 2},
                hovertemplate=f"{q_label}<br>Date: %{{x}}<br>Cum. Return: %{{y:.2%}}<extra></extra>",
            )
        )

    # Spread area (top minus bottom)
    if show_spread and n_quantiles >= 2:
        top_ret = cum_returns.get(quantile_labels[-1], [])
        bottom_ret = cum_returns.get(quantile_labels[0], [])

        if len(top_ret) > 0 and len(bottom_ret) > 0:
            min_len = min(len(top_ret), len(bottom_ret))
            spread = [top_ret[i] - bottom_ret[i] for i in range(min_len)]

            fig.add_trace(
                go.Scatter(
                    x=dates[:min_len],
                    y=spread,
                    mode="lines",
                    name="Spread (Top - Bottom)",
                    line={"color": "purple", "width": 2, "dash": "dash"},
                    hovertemplate="Spread<br>Date: %{x}<br>Spread: %{y:.2%}<extra></extra>",
                )
            )

    # Zero line
    fig.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.5)

    # Format y-axis
    fig.update_yaxes(tickformat=".0%")
    if precise_keys:
        fig.update_xaxes(
            type="category",
            tickmode="array",
            tickvals=dates,
            ticktext=[str(index + 1) for index in range(len(dates))],
            title_text="Observed date order",
        )

    fig.update_layout(
        legend={
            "yanchor": "top",
            "y": 0.99,
            "xanchor": "left",
            "x": 0.01,
        },
    )

    return fig


def plot_spread_timeseries(
    quantile_result: QuantileAnalysisResult,
    period: str | None = None,
    rolling_window: int = 21,
    show_confidence: bool = True,
    theme: str | None = None,
    width: int | None = None,
    height: int | None = None,
    *,
    observations: pl.DataFrame | pd.DataFrame | None = None,
    upper_quantile: int | None = None,
    lower_quantile: int | None = None,
    bandwidth: float = 1.0,
    exposure_scale: float = 1.0,
) -> go.Figure:
    """Plot actual dated bucket-mean differences and propagated sample errors.

    Reuse compute_quantile_spread on retained or explicitly supplied canonical
    by-date statistics. No returns are reconstructed from cumulative increments
    or pooled summaries. Missing empirical data requires explicit observations.
    Grouped statistics require one group selected before plotting.

    The raw +1 upper / -1 lower mean difference has gross-two exposure, not a
    unit-gross portfolio return. Optional bands are bandwidth times independent
    propagated standard error, not inferred confidence coverage or rolling sample
    deviation. Undefined errors remain unavailable. The existing pooled Student
    statistics on quantile_result are not substituted for this dated sample.

    Parameters
    ----------
    quantile_result : QuantileAnalysisResult
        Result with actual dated quantile observations.
    period : str | None
        Period to plot. If None, uses first period.
    rolling_window : int, default 21
        Number of observations for a complete-window moving mean, not sessions.
    show_confidence : bool, default True
        Show propagated standard-error bands around the observed spread. The
        historical argument name does not establish a confidence level.
    theme : str | None
        Plot theme
    width : int | None
        Figure width
    height : int | None
        Figure height
    observations : DataFrame, optional
        Explicit canonical by-date compute_quantile_statistics output. Its
        return_column matches period, with shared adjustment and asset counts.
    upper_quantile, lower_quantile : int, optional
        Positive bucket selectors. Upper defaults to the result's top bucket;
        the automatic top/bottom path requires at least two buckets.
    bandwidth : float
        Finite nonnegative multiplier of actual propagated standard errors.
    exposure_scale : float
        Explicit positive multiplier applied to both the difference and its SE.
        The native default remains raw gross-two exposure; 0.5 selects unit gross.

    Returns
    -------
    go.Figure
        Interactive Plotly figure
    """
    theme = validate_theme(theme)
    theme_config = get_theme_config(theme)

    import pandas as pd

    from ml4t.diagnostic.signal._prepared import positive_count
    from ml4t.diagnostic.signal.quantile import _quantile_observations, compute_quantile_spread

    window = positive_count(rolling_window, "rolling_window")
    if not isinstance(show_confidence, bool | np.bool_):
        raise ValueError("show_confidence must be boolean")
    if (
        isinstance(bandwidth, bool)
        or not isinstance(bandwidth, Real)
        or not np.isfinite(bandwidth)
        or bandwidth < 0
    ):
        raise ValueError("bandwidth must be a finite nonnegative real multiplier")
    if (
        isinstance(exposure_scale, bool)
        or not isinstance(exposure_scale, Real)
        or not np.isfinite(exposure_scale)
        or exposure_scale <= 0
    ):
        raise ValueError("exposure_scale must be a finite positive real multiplier")
    periods = quantile_result.periods
    if not periods:
        raise ValueError("No return periods are available for empirical spread plotting")
    if period is None:
        period = periods[0]
    elif period not in periods:
        raise ValueError(f"Period '{period}' not found. Available: {periods}")

    if upper_quantile is None and quantile_result.n_quantiles < 2:
        raise ValueError("automatic spread plotting requires at least two quantile buckets")
    values = (
        _quantile_observations(observations)
        if observations is not None
        else quantile_result.get_dataframe("observations")
    )
    if values.height and period not in values["return_column"].to_list():
        raise ValueError(f"empirical observations do not contain selected period {period!r}")
    values = values.filter(pl.col("return_column") == period)
    if "group" in values.columns and values["group"].n_unique() > 1:
        raise ValueError("select one group before plotting empirical spread observations")
    methods = values["return_adjustment"].unique().to_list()
    if len(methods) > 1:
        raise ValueError("selected observations mix return-adjustment methods")
    bucket_ids = quantile_result.quantile_ids or list(range(1, quantile_result.n_quantiles + 1))
    spread_data = compute_quantile_spread(
        values,
        upper_quantile=max(bucket_ids) if upper_quantile is None else upper_quantile,
        lower_quantile=min(bucket_ids) if lower_quantile is None else lower_quantile,
        include_standard_error=bool(show_confidence),
    ).sort("date")
    spread = spread_data["spread"].cast(pl.Float64).to_numpy() * exposure_scale
    errors = spread_data["standard_error"].cast(pl.Float64).to_numpy() * exposure_scale
    if spread_data["date"].dtype.is_temporal():
        timestamps = [pd.Timestamp(value) for value in spread_data["date"].to_pandas()]
        dates = [value.isoformat() for value in timestamps]
        categories = any(value.value % 1_000_000 for value in timestamps)
    else:
        dates = [str(value) for value in spread_data["date"]]
        categories = True
    customdata = list(
        zip(
            dates,
            spread_data["upper_n_obs"].to_list(),
            spread_data["lower_n_obs"].to_list(),
            strict=True,
        )
    )
    rolling_mean = pd.Series(spread).rolling(window, min_periods=window).mean().to_numpy()

    # Create figure
    fig = create_base_figure(
        title=f"Observed Quantile Spread ({period})",
        xaxis_title="Observed date order" if categories else "Date",
        yaxis_title="Upper - lower mean return",
        width=width or theme_config["defaults"]["line_height"] + 300,
        height=height or theme_config["defaults"]["line_height"],
        theme=theme,
    )

    fig.add_trace(
        go.Scatter(
            x=dates,
            y=spread,
            customdata=customdata,
            mode="lines+markers",
            name="Observed Spread",
            marker={
                "size": 4,
                "color": theme_config["colorway"][0],
                "opacity": 0.4,
            },
            hovertemplate="Date: %{customdata[0]}<br>Spread: %{y:.4%}<br>Upper assets: %{customdata[1]}<br>Lower assets: %{customdata[2]}<extra></extra>",
        )
    )

    # Rolling mean
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=rolling_mean,
            mode="lines",
            name=f"{window}-Observation Rolling Mean",
            line={"color": theme_config["colorway"][1], "width": 2},
            hovertemplate="Date: %{x}<br>Rolling Mean: %{y:.4f}<extra></extra>",
        )
    )

    if show_confidence:
        upper = spread + bandwidth * errors
        lower = spread - bandwidth * errors

        fig.add_trace(
            go.Scatter(
                x=list(dates) + list(reversed(dates)),
                y=list(upper) + list(reversed(lower)),
                mode="lines",
                fill="toself",
                fillcolor="rgba(128, 128, 128, 0.2)",
                line={"width": 0},
                showlegend=True,
                name="Propagated SE band",
                hoverinfo="skip",
            )
        )

    # Zero line
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)

    summary_text = f"+{exposure_scale:g} upper / -{exposure_scale:g} lower (gross {2 * exposure_scale:g})<br>Defined observed pairs: {int(np.isfinite(spread).sum())}"
    if show_confidence:
        summary_text += f"<br>Band: {bandwidth:g} × independent propagated SE"
        if not np.isfinite(errors).all():
            summary_text += "<br>Some propagated errors unavailable"
    if not np.isfinite(spread).any():
        summary_text += "<br>No defined observed spread"

    fig.add_annotation(
        text=summary_text,
        xref="paper",
        yref="paper",
        x=0.02,
        y=0.98,
        showarrow=False,
        bgcolor="rgba(255,255,255,0.8)" if theme != "dark" else "rgba(50,50,50,0.8)",
        bordercolor="gray",
        borderwidth=1,
        align="left",
    )

    # Format y-axis
    fig.update_yaxes(tickformat=".2%")
    if categories:
        fig.update_xaxes(
            type="category",
            tickmode="array",
            tickvals=dates,
            ticktext=[str(index + 1) for index in range(len(dates))],
        )
    fig.update_layout(
        showlegend=True,
        legend={"orientation": "h", "x": 0, "y": -0.2},
        meta={
            "return_unit": "decimal",
            "sample_unit": "date_bucket_mean_difference",
            "exposure_convention": "long_upper_short_lower_unit_each"
            if exposure_scale == 1
            else "long_upper_short_lower_scaled",
            "exposure_scale": float(exposure_scale),
            "error_method": "independent_propagation" if show_confidence else "unavailable",
            "bandwidth": float(bandwidth),
            "rolling_window_observations": window,
            "return_adjustment": methods[0] if methods else None,
            "x_clock": "observation_order" if categories else "dated",
            "timestamp_keys": dates,
        },
    )

    return fig
