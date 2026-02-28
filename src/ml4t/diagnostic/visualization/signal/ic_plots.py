"""IC (Information Coefficient) visualization plots.

This module provides interactive Plotly visualizations for IC analysis:
- plot_ic_ts: IC time series with rolling mean and significance bands
- plot_ic_histogram: IC distribution with mean and confidence intervals
- plot_ic_qq: Q-Q plot for normality assessment
- plot_ic_heatmap: Monthly IC heatmap for seasonality analysis
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

import numpy as np
import plotly.graph_objects as go
from scipy import stats

from ml4t.diagnostic.visualization.core import (
    create_base_figure,
    format_percentage,
    get_colorscale,
    get_theme_config,
    validate_theme,
)

if TYPE_CHECKING:
    from ml4t.diagnostic.results.signal_results import SignalICResult


def plot_ic_ts(
    ic_result: SignalICResult,
    period: str | None = None,
    rolling_window: int = 21,
    show_significance: bool = True,
    significance_level: float = 0.05,
    theme: str | None = None,
    width: int | None = None,
    height: int | None = None,
) -> go.Figure:
    """Plot IC time series with rolling mean and significance bands.

    Parameters
    ----------
    ic_result : SignalICResult
        IC analysis result from a signal analysis workflow.
    period : str | None
        Period to plot (e.g., "1D", "5D"). If None, uses first period.
    rolling_window : int, default 21
        Window size for rolling mean calculation
    show_significance : bool, default True
        Show significance bands (±1.96 * std for 95% CI)
    significance_level : float, default 0.05
        Significance level for bands (0.05 = 95% CI)
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
    >>> ic_result = analyzer.compute_ic_analysis()
    >>> fig = plot_ic_ts(ic_result, period="5D", rolling_window=21)
    >>> fig.show()
    """
    theme = validate_theme(theme)
    theme_config = get_theme_config(theme)

    # Get period data
    periods = list(ic_result.ic_by_date.keys())
    if period is None:
        period = periods[0]
    elif period not in periods:
        raise ValueError(f"Period '{period}' not found. Available: {periods}")

    ic_series = np.array(ic_result.ic_by_date[period])
    dates = ic_result.dates

    # Convert dates to datetime if strings
    if dates and isinstance(dates[0], str):
        try:
            dates = [datetime.fromisoformat(d) for d in dates]
        except ValueError:
            pass  # Keep as strings if conversion fails

    # Compute rolling mean
    valid_mask = ~np.isnan(ic_series)
    np.where(valid_mask, ic_series, 0)

    rolling_mean = np.full_like(ic_series, np.nan)
    for i in range(rolling_window - 1, len(ic_series)):
        window = ic_series[i - rolling_window + 1 : i + 1]
        window_valid = window[~np.isnan(window)]
        if len(window_valid) >= rolling_window // 2:
            rolling_mean[i] = np.mean(window_valid)

    # Create figure
    fig = create_base_figure(
        title=f"IC Time Series ({period})",
        xaxis_title="Date",
        yaxis_title="Information Coefficient",
        width=width or theme_config["defaults"]["line_height"] + 300,
        height=height or theme_config["defaults"]["line_height"],
        theme=theme,
    )

    # IC scatter points
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=ic_series,
            mode="markers",
            name="Daily IC",
            marker={
                "size": 4,
                "color": theme_config["colorway"][0],
                "opacity": 0.5,
            },
            hovertemplate="Date: %{x}<br>IC: %{y:.4f}<extra></extra>",
        )
    )

    # Rolling mean line
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=rolling_mean,
            mode="lines",
            name=f"{rolling_window}-Day Rolling Mean",
            line={
                "color": theme_config["colorway"][1],
                "width": 2,
            },
            hovertemplate="Date: %{x}<br>Rolling IC: %{y:.4f}<extra></extra>",
        )
    )

    # Zero line
    fig.add_hline(
        y=0,
        line_dash="dash",
        line_color="gray",
        opacity=0.5,
    )

    # Mean IC line
    mean_ic = ic_result.ic_mean.get(period, 0)
    fig.add_hline(
        y=mean_ic,
        line_dash="dot",
        line_color=theme_config["colorway"][2],
        annotation_text=f"Mean IC: {mean_ic:.4f}",
        annotation_position="right",
    )

    # Significance bands
    if show_significance:
        ic_std = ic_result.ic_std.get(period, 0)
        z_score = stats.norm.ppf(1 - significance_level / 2)
        upper = z_score * ic_std / np.sqrt(len(ic_series))
        lower = -upper

        fig.add_hrect(
            y0=lower,
            y1=upper,
            fillcolor="gray",
            opacity=0.1,
            line_width=0,
            annotation_text=f"{100 * (1 - significance_level):.0f}% CI",
            annotation_position="top right",
        )

    # Add summary annotation
    positive_pct = ic_result.ic_positive_pct.get(period, 0)
    ir = ic_result.ic_ir.get(period, 0)
    t_stat = ic_result.ic_t_stat.get(period, 0)
    p_value = ic_result.ic_p_value.get(period, 1)

    summary_text = (
        f"<b>Summary:</b><br>"
        f"Mean IC: {mean_ic:.4f}<br>"
        f"IC IR: {ir:.3f}<br>"
        f"Positive %: {format_percentage(positive_pct)}<br>"
        f"t-stat: {t_stat:.2f} (p={p_value:.4f})"
    )

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

    fig.update_layout(
        legend={
            "yanchor": "top",
            "y": 0.99,
            "xanchor": "right",
            "x": 0.99,
        },
        showlegend=True,
    )

    return fig


def plot_ic_histogram(
    ic_result: SignalICResult,
    period: str | None = None,
    bins: int = 50,
    show_kde: bool = True,
    show_stats: bool = True,
    theme: str | None = None,
    width: int | None = None,
    height: int | None = None,
) -> go.Figure:
    """Plot IC distribution histogram with optional KDE.

    Parameters
    ----------
    ic_result : SignalICResult
        IC analysis result from a signal analysis workflow.
    period : str | None
        Period to plot. If None, uses first period.
    bins : int, default 50
        Number of histogram bins
    show_kde : bool, default True
        Show kernel density estimate overlay
    show_stats : bool, default True
        Show summary statistics annotation
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
    """
    theme = validate_theme(theme)
    theme_config = get_theme_config(theme)

    # Get period data
    periods = list(ic_result.ic_by_date.keys())
    if period is None:
        period = periods[0]
    elif period not in periods:
        raise ValueError(f"Period '{period}' not found. Available: {periods}")

    ic_series = np.array(ic_result.ic_by_date[period])
    ic_clean = ic_series[~np.isnan(ic_series)]

    # Create figure
    fig = create_base_figure(
        title=f"IC Distribution ({period})",
        xaxis_title="Information Coefficient",
        yaxis_title="Frequency",
        width=width or theme_config["defaults"]["bar_height"],
        height=height or theme_config["defaults"]["bar_height"],
        theme=theme,
    )

    # Histogram
    fig.add_trace(
        go.Histogram(
            x=ic_clean,
            nbinsx=bins,
            name="IC Distribution",
            marker_color=theme_config["colorway"][0],
            opacity=0.7,
            hovertemplate="IC: %{x:.4f}<br>Count: %{y}<extra></extra>",
        )
    )

    # KDE overlay
    if show_kde and len(ic_clean) > 10:
        kde = stats.gaussian_kde(ic_clean)
        x_kde = np.linspace(ic_clean.min(), ic_clean.max(), 200)
        y_kde = kde(x_kde)

        # Scale KDE to match histogram
        hist_counts, _ = np.histogram(ic_clean, bins=bins)
        bin_width = (ic_clean.max() - ic_clean.min()) / bins
        y_kde_scaled = y_kde * len(ic_clean) * bin_width

        fig.add_trace(
            go.Scatter(
                x=x_kde,
                y=y_kde_scaled,
                mode="lines",
                name="KDE",
                line={"color": theme_config["colorway"][1], "width": 2},
            )
        )

    # Mean line
    mean_ic = ic_result.ic_mean.get(period, 0)
    fig.add_vline(
        x=mean_ic,
        line_dash="dash",
        line_color=theme_config["colorway"][2],
        annotation_text=f"Mean: {mean_ic:.4f}",
    )

    # Zero line
    fig.add_vline(
        x=0,
        line_dash="dot",
        line_color="gray",
        opacity=0.7,
    )

    # Statistics annotation
    if show_stats:
        ic_std = ic_result.ic_std.get(period, 0)
        skewness = float(stats.skew(ic_clean)) if len(ic_clean) > 2 else 0
        kurtosis = float(stats.kurtosis(ic_clean)) if len(ic_clean) > 3 else 0

        stats_text = (
            f"<b>Statistics:</b><br>"
            f"N: {len(ic_clean)}<br>"
            f"Mean: {mean_ic:.4f}<br>"
            f"Std: {ic_std:.4f}<br>"
            f"Skew: {skewness:.3f}<br>"
            f"Kurt: {kurtosis:.3f}"
        )

        fig.add_annotation(
            text=stats_text,
            xref="paper",
            yref="paper",
            x=0.98,
            y=0.98,
            showarrow=False,
            bgcolor="rgba(255,255,255,0.8)" if theme != "dark" else "rgba(50,50,50,0.8)",
            bordercolor="gray",
            borderwidth=1,
            align="left",
            xanchor="right",
            yanchor="top",
        )

    return fig


def plot_ic_qq(
    ic_result: SignalICResult,
    period: str | None = None,
    theme: str | None = None,
    width: int | None = None,
    height: int | None = None,
) -> go.Figure:
    """Plot Q-Q plot for IC normality assessment.

    Parameters
    ----------
    ic_result : SignalICResult
        IC analysis result from a signal analysis workflow.
    period : str | None
        Period to plot. If None, uses first period.
    theme : str | None
        Plot theme
    width : int | None
        Figure width
    height : int | None
        Figure height

    Returns
    -------
    go.Figure
        Interactive Plotly Q-Q plot
    """
    theme = validate_theme(theme)
    theme_config = get_theme_config(theme)

    # Get period data
    periods = list(ic_result.ic_by_date.keys())
    if period is None:
        period = periods[0]
    elif period not in periods:
        raise ValueError(f"Period '{period}' not found. Available: {periods}")

    ic_series = np.array(ic_result.ic_by_date[period])
    ic_clean = ic_series[~np.isnan(ic_series)]
    ic_sorted = np.sort(ic_clean)

    # Theoretical quantiles
    n = len(ic_sorted)
    theoretical_quantiles = stats.norm.ppf(
        (np.arange(1, n + 1) - 0.5) / n,
        loc=np.mean(ic_clean),
        scale=np.std(ic_clean, ddof=1),
    )

    # Create figure
    fig = create_base_figure(
        title=f"IC Q-Q Plot ({period})",
        xaxis_title="Theoretical Quantiles (Normal)",
        yaxis_title="Sample Quantiles (IC)",
        width=width or theme_config["defaults"]["scatter_height"],
        height=height or theme_config["defaults"]["scatter_height"],
        theme=theme,
    )

    # Q-Q scatter
    fig.add_trace(
        go.Scatter(
            x=theoretical_quantiles,
            y=ic_sorted,
            mode="markers",
            name="IC Values",
            marker={
                "size": 6,
                "color": theme_config["colorway"][0],
                "opacity": 0.6,
            },
            hovertemplate="Theoretical: %{x:.4f}<br>Sample: %{y:.4f}<extra></extra>",
        )
    )

    # Reference line (45-degree)
    min_val = min(theoretical_quantiles.min(), ic_sorted.min())
    max_val = max(theoretical_quantiles.max(), ic_sorted.max())

    fig.add_trace(
        go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode="lines",
            name="Normal Reference",
            line={"color": theme_config["colorway"][1], "dash": "dash", "width": 2},
        )
    )

    # Normality test
    if len(ic_clean) >= 8:
        _, shapiro_p = stats.shapiro(ic_clean[:5000])  # Shapiro-Wilk limited to 5000
        _, jb_stat, jb_p = stats.jarque_bera(ic_clean)

        normality_text = (
            f"<b>Normality Tests:</b><br>"
            f"Shapiro-Wilk p: {shapiro_p:.4f}<br>"
            f"Jarque-Bera p: {jb_p:.4f}<br>"
            f"{'✓ Normal' if min(shapiro_p, jb_p) > 0.05 else '✗ Non-normal'}"
        )

        fig.add_annotation(
            text=normality_text,
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

    return fig


def plot_ic_heatmap(
    ic_result: SignalICResult,
    period: str | None = None,
    colorscale: str = "rdbu",
    theme: str | None = None,
    width: int | None = None,
    height: int | None = None,
) -> go.Figure:
    """Plot monthly IC heatmap for seasonality analysis.

    Parameters
    ----------
    ic_result : SignalICResult
        IC analysis result from a signal analysis workflow.
    period : str | None
        Period to plot. If None, uses first period.
    colorscale : str, default "rdbu"
        Plotly colorscale name (rdbu for diverging red-blue)
    theme : str | None
        Plot theme
    width : int | None
        Figure width
    height : int | None
        Figure height

    Returns
    -------
    go.Figure
        Interactive Plotly heatmap
    """
    theme = validate_theme(theme)
    theme_config = get_theme_config(theme)

    # Get period data
    periods = list(ic_result.ic_by_date.keys())
    if period is None:
        period = periods[0]
    elif period not in periods:
        raise ValueError(f"Period '{period}' not found. Available: {periods}")

    ic_series = np.array(ic_result.ic_by_date[period])
    dates = ic_result.dates

    # Parse dates and create year-month structure
    parsed_dates = []
    for d in dates:
        if isinstance(d, str):
            try:
                parsed_dates.append(datetime.fromisoformat(d))
            except ValueError:
                try:
                    parsed_dates.append(datetime.strptime(d, "%Y-%m-%d"))
                except ValueError:
                    continue
        elif isinstance(d, datetime):
            parsed_dates.append(d)
        else:
            try:
                parsed_dates.append(datetime.fromisoformat(str(d)))
            except ValueError:
                continue

    if len(parsed_dates) != len(ic_series):
        raise ValueError("Date parsing failed - length mismatch")

    # Build year-month matrix
    import pandas as pd

    df = pd.DataFrame(
        {
            "date": parsed_dates,
            "ic": ic_series,
        }
    )
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month

    # Pivot to get mean IC by year-month
    pivot = df.pivot_table(values="ic", index="year", columns="month", aggfunc="mean")
    pivot = pivot.sort_index(ascending=False)  # Most recent year at top

    # Month names
    month_names = [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    ]

    # Create figure
    fig = create_base_figure(
        title=f"Monthly IC Heatmap ({period})",
        xaxis_title="Month",
        yaxis_title="Year",
        width=width or theme_config["defaults"]["heatmap_height"],
        height=height or 400 + 30 * len(pivot),
        theme=theme,
    )

    # Get colorscale
    try:
        colors = get_colorscale(colorscale)
    except ValueError:
        colors = "RdBu"

    # Symmetric color scale around zero
    ic_values = pivot.values.flatten()
    ic_clean = ic_values[~np.isnan(ic_values)]
    if len(ic_clean) > 0:
        max_abs = max(abs(ic_clean.min()), abs(ic_clean.max()))
        zmin, zmax = -max_abs, max_abs
    else:
        zmin, zmax = -0.1, 0.1

    # Heatmap
    fig.add_trace(
        go.Heatmap(
            z=pivot.values,
            x=month_names[: int(pivot.columns.max())],
            y=pivot.index.astype(str).tolist(),
            colorscale=colors if isinstance(colors, str) else "RdBu",
            zmid=0,
            zmin=zmin,
            zmax=zmax,
            colorbar={"title": "Mean IC"},
            hovertemplate="Year: %{y}<br>Month: %{x}<br>IC: %{z:.4f}<extra></extra>",
        )
    )

    # Add text annotations
    for _i, year in enumerate(pivot.index):
        for _j, month in enumerate(pivot.columns):
            val = pivot.loc[year, month]
            if not np.isnan(val):
                text_color = "white" if abs(val) > max_abs * 0.5 else "black"
                fig.add_annotation(
                    x=month_names[int(month) - 1],
                    y=str(year),
                    text=f"{val:.3f}",
                    showarrow=False,
                    font={"size": 10, "color": text_color},
                )

    fig.update_layout(
        xaxis={"side": "bottom", "tickangle": 0},
        yaxis={"autorange": "reversed"},
    )

    return fig
