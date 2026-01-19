"""Barrier analysis visualization plots.

This module provides interactive Plotly visualizations for triple barrier analysis:
- plot_hit_rate_heatmap: Heatmap of hit rates by quantile and outcome type
- plot_profit_factor_bar: Bar chart of profit factor by quantile
- plot_precision_recall_curve: Precision/recall curve with F1 peak
- plot_time_to_target_box: Box plots of bars to exit by quantile and outcome

All plots follow the consistent API pattern:
- Accept result objects from BarrierAnalysis methods
- Return go.Figure objects
- Support theme customization
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import plotly.graph_objects as go

from ml4t.diagnostic.visualization.core import (
    create_base_figure,
    get_colorscale,
    get_theme_config,
    validate_theme,
)

if TYPE_CHECKING:
    from ml4t.diagnostic.results.barrier_results import (
        HitRateResult,
        PrecisionRecallResult,
        ProfitFactorResult,
        TimeToTargetResult,
    )


def _get_quantile_colors(n_quantiles: int, theme_config: dict) -> list[str]:
    """Get diverging colors for quantiles (red → green progression)."""
    colors: list[str]
    if n_quantiles <= 5:
        colors = ["#D32F2F", "#F57C00", "#FBC02D", "#689F38", "#388E3C"][:n_quantiles]
    else:
        try:
            raw_colors = get_colorscale("rdylgn", n_colors=n_quantiles, reverse=False)
            if isinstance(raw_colors[0], tuple):
                colors = [str(c[1]) if isinstance(c, tuple) else str(c) for c in raw_colors]
            else:
                colors = [str(c) for c in raw_colors]
        except (ValueError, IndexError):
            colorway = theme_config["colorway"]
            repeated = colorway * ((n_quantiles // len(colorway)) + 1)
            colors = [str(c) for c in repeated[:n_quantiles]]
    return colors


def _get_outcome_colors() -> dict[str, str]:
    """Get colors for barrier outcomes."""
    return {
        "tp": "#2ECC71",  # Green for take-profit
        "sl": "#E74C3C",  # Red for stop-loss
        "timeout": "#95A5A6",  # Gray for timeout
    }


def plot_hit_rate_heatmap(
    hit_rate_result: HitRateResult,
    show_counts: bool = True,
    show_chi2: bool = True,
    theme: str | None = None,
    width: int | None = None,
    height: int | None = None,
) -> go.Figure:
    """Plot hit rates as a heatmap (quantile x outcome type).

    Creates a heatmap showing hit rates for each outcome type (TP, SL, timeout)
    across signal quantiles. Includes chi-square test annotation.

    Parameters
    ----------
    hit_rate_result : HitRateResult
        Hit rate analysis result from BarrierAnalysis.compute_hit_rates()
    show_counts : bool, default True
        Show observation counts in cell text
    show_chi2 : bool, default True
        Show chi-square test results annotation
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
    >>> hit_rates = analysis.compute_hit_rates()
    >>> fig = plot_hit_rate_heatmap(hit_rates)
    >>> fig.show()
    """
    theme = validate_theme(theme)
    theme_config = get_theme_config(theme)

    quantile_labels = hit_rate_result.quantile_labels
    outcome_labels = ["Take-Profit", "Stop-Loss", "Timeout"]

    # Build heatmap data matrix [outcomes x quantiles]
    z_values = [
        [hit_rate_result.hit_rate_tp[q] for q in quantile_labels],
        [hit_rate_result.hit_rate_sl[q] for q in quantile_labels],
        [hit_rate_result.hit_rate_timeout[q] for q in quantile_labels],
    ]

    # Build text annotations (rate % and optionally count)
    text_values = []
    for i, outcome in enumerate(["tp", "sl", "timeout"]):
        row_text = []
        for q in quantile_labels:
            rate = z_values[i][quantile_labels.index(q)]
            count_dict = {
                "tp": hit_rate_result.count_tp,
                "sl": hit_rate_result.count_sl,
                "timeout": hit_rate_result.count_timeout,
            }
            count = count_dict[outcome][q]
            if show_counts:
                row_text.append(f"{rate:.1%}<br>n={count:,}")
            else:
                row_text.append(f"{rate:.1%}")
        text_values.append(row_text)

    # Create figure
    fig = create_base_figure(
        title="Hit Rate by Signal Quantile and Outcome",
        xaxis_title="Signal Quantile",
        yaxis_title="Outcome Type",
        width=width or theme_config["defaults"]["heatmap_height"],
        height=height or 400,
        theme=theme,
    )

    # Add heatmap
    fig.add_trace(
        go.Heatmap(
            z=z_values,
            x=quantile_labels,
            y=outcome_labels,
            text=text_values,
            texttemplate="%{text}",
            textfont={"size": 11},
            colorscale="RdYlGn",
            colorbar={
                "title": "Hit Rate",
                "tickformat": ".0%",
            },
            hovertemplate=(
                "Quantile: %{x}<br>Outcome: %{y}<br>Hit Rate: %{z:.2%}<br><extra></extra>"
            ),
        )
    )

    # Chi-square annotation
    if show_chi2:
        sig_text = "✓" if hit_rate_result.is_significant else "✗"
        chi2_text = (
            f"<b>Chi-Square Test:</b><br>"
            f"χ² = {hit_rate_result.chi2_statistic:.2f}<br>"
            f"p = {hit_rate_result.chi2_p_value:.4f}<br>"
            f"Significant: {sig_text} (α={hit_rate_result.significance_level})"
        )

        fig.add_annotation(
            text=chi2_text,
            xref="paper",
            yref="paper",
            x=1.02,
            y=1.0,
            showarrow=False,
            bgcolor="rgba(255,255,255,0.9)" if theme != "dark" else "rgba(50,50,50,0.9)",
            bordercolor="gray",
            borderwidth=1,
            align="left",
            xanchor="left",
            yanchor="top",
            font={"size": 10},
        )

    fig.update_layout(
        xaxis={"side": "bottom"},
        yaxis={"autorange": "reversed"},
    )

    return fig


def plot_profit_factor_bar(
    profit_factor_result: ProfitFactorResult,
    show_reference_line: bool = True,
    show_average_return: bool = True,
    theme: str | None = None,
    width: int | None = None,
    height: int | None = None,
) -> go.Figure:
    """Plot profit factor by quantile as a bar chart.

    Creates a bar chart showing profit factor for each signal quantile,
    with reference line at PF=1.0 (breakeven).

    Parameters
    ----------
    profit_factor_result : ProfitFactorResult
        Profit factor result from BarrierAnalysis.compute_profit_factor()
    show_reference_line : bool, default True
        Show horizontal line at PF=1.0 (breakeven)
    show_average_return : bool, default True
        Show average return as secondary y-axis
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
    >>> pf = analysis.compute_profit_factor()
    >>> fig = plot_profit_factor_bar(pf)
    >>> fig.show()
    """
    theme = validate_theme(theme)
    theme_config = get_theme_config(theme)

    quantile_labels = profit_factor_result.quantile_labels
    n_quantiles = profit_factor_result.n_quantiles

    # Get colors
    colors = _get_quantile_colors(n_quantiles, theme_config)

    # Prepare data
    pf_values = [profit_factor_result.profit_factor[q] for q in quantile_labels]
    avg_returns = [profit_factor_result.avg_return[q] for q in quantile_labels]

    # Create figure
    fig = create_base_figure(
        title="Profit Factor by Signal Quantile",
        xaxis_title="Signal Quantile",
        yaxis_title="Profit Factor",
        width=width or theme_config["defaults"]["bar_height"],
        height=height or theme_config["defaults"]["bar_height"],
        theme=theme,
    )

    # Bar chart for profit factor
    fig.add_trace(
        go.Bar(
            x=quantile_labels,
            y=pf_values,
            marker_color=colors,
            name="Profit Factor",
            hovertemplate=("Quantile: %{x}<br>Profit Factor: %{y:.2f}<br><extra></extra>"),
        )
    )

    # Reference line at PF=1.0
    if show_reference_line:
        fig.add_hline(
            y=1.0,
            line_dash="dash",
            line_color="gray",
            line_width=2,
            annotation_text="Breakeven (PF=1)",
            annotation_position="right",
            annotation={"font_size": 10, "font_color": "gray"},
        )

    # Secondary y-axis for average return
    if show_average_return:
        fig.add_trace(
            go.Scatter(
                x=quantile_labels,
                y=avg_returns,
                mode="lines+markers",
                name="Avg Return",
                yaxis="y2",
                line={"color": theme_config["colorway"][1], "width": 2},
                marker={"size": 8},
                hovertemplate=("Quantile: %{x}<br>Avg Return: %{y:.4%}<br><extra></extra>"),
            )
        )

        # Update layout for secondary y-axis
        fig.update_layout(
            yaxis2={
                "title": "Average Return",
                "overlaying": "y",
                "side": "right",
                "tickformat": ".2%",
                "showgrid": False,
            },
            legend={
                "yanchor": "top",
                "y": 0.99,
                "xanchor": "left",
                "x": 0.01,
            },
        )

    # Monotonicity annotation
    direction = profit_factor_result.pf_direction
    monotonic = profit_factor_result.pf_monotonic
    rho = profit_factor_result.pf_spearman

    mono_text = (
        f"<b>Monotonicity:</b><br>"
        f"Monotonic: {'✓' if monotonic else '✗'} ({direction})<br>"
        f"Spearman ρ: {rho:.3f}<br>"
        f"Overall PF: {profit_factor_result.overall_profit_factor:.2f}"
    )

    fig.add_annotation(
        text=mono_text,
        xref="paper",
        yref="paper",
        x=0.02,
        y=0.98,
        showarrow=False,
        bgcolor="rgba(255,255,255,0.9)" if theme != "dark" else "rgba(50,50,50,0.9)",
        bordercolor="gray",
        borderwidth=1,
        align="left",
        xanchor="left",
        yanchor="top",
        font={"size": 10},
    )

    return fig


def plot_precision_recall_curve(
    precision_recall_result: PrecisionRecallResult,
    show_f1_peak: bool = True,
    show_lift: bool = True,
    theme: str | None = None,
    width: int | None = None,
    height: int | None = None,
) -> go.Figure:
    """Plot cumulative precision/recall curves with F1 score.

    Creates a line chart showing cumulative precision, recall, and F1 score
    as signal quantile threshold moves from top (D10) to bottom (D1).

    Parameters
    ----------
    precision_recall_result : PrecisionRecallResult
        Precision/recall result from BarrierAnalysis.compute_precision_recall()
    show_f1_peak : bool, default True
        Highlight the quantile with best F1 score
    show_lift : bool, default True
        Show lift curve on secondary y-axis
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
    >>> pr = analysis.compute_precision_recall()
    >>> fig = plot_precision_recall_curve(pr)
    >>> fig.show()
    """
    theme = validate_theme(theme)
    theme_config = get_theme_config(theme)

    quantile_labels = precision_recall_result.quantile_labels

    # Prepare data - reversed order (from D10 to D1 for cumulative threshold)
    reversed_labels = list(reversed(quantile_labels))

    cum_precision = [precision_recall_result.cumulative_precision_tp[q] for q in reversed_labels]
    cum_recall = [precision_recall_result.cumulative_recall_tp[q] for q in reversed_labels]
    cum_f1 = [precision_recall_result.cumulative_f1_tp[q] for q in reversed_labels]
    cum_lift = [precision_recall_result.cumulative_lift_tp[q] for q in reversed_labels]

    # Create figure
    fig = create_base_figure(
        title="Cumulative Precision/Recall Curve (Top Quantiles)",
        xaxis_title="Include Down to Quantile (from top)",
        yaxis_title="Rate",
        width=width or theme_config["defaults"]["line_height"] + 200,
        height=height or theme_config["defaults"]["line_height"],
        theme=theme,
    )

    # Precision line
    fig.add_trace(
        go.Scatter(
            x=reversed_labels,
            y=cum_precision,
            mode="lines+markers",
            name="Cumulative Precision",
            line={"color": "#3498DB", "width": 2},
            marker={"size": 8},
            hovertemplate=("Threshold: %{x}<br>Precision: %{y:.2%}<br><extra></extra>"),
        )
    )

    # Recall line
    fig.add_trace(
        go.Scatter(
            x=reversed_labels,
            y=cum_recall,
            mode="lines+markers",
            name="Cumulative Recall",
            line={"color": "#E74C3C", "width": 2},
            marker={"size": 8},
            hovertemplate=("Threshold: %{x}<br>Recall: %{y:.2%}<br><extra></extra>"),
        )
    )

    # F1 score line
    fig.add_trace(
        go.Scatter(
            x=reversed_labels,
            y=cum_f1,
            mode="lines+markers",
            name="Cumulative F1",
            line={"color": "#9B59B6", "width": 3, "dash": "dash"},
            marker={"size": 10, "symbol": "diamond"},
            hovertemplate=("Threshold: %{x}<br>F1 Score: %{y:.4f}<br><extra></extra>"),
        )
    )

    # Baseline horizontal line
    baseline = precision_recall_result.baseline_tp_rate
    fig.add_hline(
        y=baseline,
        line_dash="dot",
        line_color="gray",
        line_width=1,
        annotation_text=f"Baseline: {baseline:.1%}",
        annotation_position="right",
        annotation={"font_size": 10, "font_color": "gray"},
    )

    # F1 peak marker
    if show_f1_peak:
        best_q = precision_recall_result.best_f1_quantile
        best_f1 = precision_recall_result.best_f1_score

        fig.add_trace(
            go.Scatter(
                x=[best_q],
                y=[best_f1],
                mode="markers+text",
                name=f"Best F1 ({best_q})",
                marker={"size": 15, "color": "#F39C12", "symbol": "star"},
                text=[f"Best F1: {best_f1:.4f}"],
                textposition="top center",
                hovertemplate=(
                    f"<b>Best F1 Score</b><br>"
                    f"Quantile: {best_q}<br>"
                    f"F1: {best_f1:.4f}<br>"
                    f"<extra></extra>"
                ),
            )
        )

    # Lift curve on secondary axis
    if show_lift:
        fig.add_trace(
            go.Scatter(
                x=reversed_labels,
                y=cum_lift,
                mode="lines+markers",
                name="Cumulative Lift",
                yaxis="y2",
                line={"color": "#2ECC71", "width": 2},
                marker={"size": 6, "symbol": "triangle-up"},
                hovertemplate=("Threshold: %{x}<br>Lift: %{y:.2f}x<br><extra></extra>"),
            )
        )

        fig.update_layout(
            yaxis2={
                "title": "Lift (vs baseline)",
                "overlaying": "y",
                "side": "right",
                "showgrid": False,
            },
        )

    # Format y-axis as percentage
    fig.update_yaxes(tickformat=".0%")

    fig.update_layout(
        legend={
            "yanchor": "bottom",
            "y": 0.01,
            "xanchor": "right",
            "x": 0.99,
        },
    )

    return fig


def plot_time_to_target_box(
    time_to_target_result: TimeToTargetResult,
    outcome_type: str = "all",
    show_mean: bool = True,
    show_median_line: bool = True,
    theme: str | None = None,
    width: int | None = None,
    height: int | None = None,
) -> go.Figure:
    """Plot time-to-target as box plots by quantile.

    Creates box plots showing the distribution of bars to exit for each
    signal quantile. Can show all outcomes or filter by type.

    Parameters
    ----------
    time_to_target_result : TimeToTargetResult
        Time-to-target result from BarrierAnalysis.compute_time_to_target()
    outcome_type : str, default "all"
        Which outcomes to show: "all", "tp", "sl", "comparison"
        "comparison" shows TP and SL side by side
    show_mean : bool, default True
        Show mean marker on box plots
    show_median_line : bool, default True
        Show overall median as horizontal line
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
    >>> ttt = analysis.compute_time_to_target()
    >>> fig = plot_time_to_target_box(ttt, outcome_type="comparison")
    >>> fig.show()
    """
    theme = validate_theme(theme)
    theme_config = get_theme_config(theme)

    quantile_labels = time_to_target_result.quantile_labels
    n_quantiles = time_to_target_result.n_quantiles
    outcome_colors = _get_outcome_colors()

    # Create figure
    title_suffix = {
        "all": "(All Outcomes)",
        "tp": "(Take-Profit)",
        "sl": "(Stop-Loss)",
        "comparison": "(TP vs SL)",
    }
    fig = create_base_figure(
        title=f"Time to Target by Signal Quantile {title_suffix.get(outcome_type, '')}",
        xaxis_title="Signal Quantile",
        yaxis_title="Bars to Exit",
        width=width or theme_config["defaults"]["bar_height"] + 200,
        height=height or theme_config["defaults"]["bar_height"],
        theme=theme,
    )

    if outcome_type == "comparison":
        # Side-by-side comparison of TP and SL
        for i, q in enumerate(quantile_labels):
            # TP box
            mean_tp = time_to_target_result.mean_bars_tp[q]
            median_tp = time_to_target_result.median_bars_tp[q]
            std_tp = time_to_target_result.std_bars_tp[q]
            count_tp = time_to_target_result.count_tp[q]

            # Create synthetic box data from statistics
            if count_tp > 0 and not np.isnan(mean_tp):
                q1_tp = max(0, mean_tp - 0.675 * std_tp)
                q3_tp = mean_tp + 0.675 * std_tp
                whisker_low_tp = max(0, mean_tp - 1.5 * std_tp)
                whisker_high_tp = mean_tp + 1.5 * std_tp

                fig.add_trace(
                    go.Box(
                        x=[q],
                        q1=[q1_tp],
                        median=[median_tp],
                        q3=[q3_tp],
                        lowerfence=[whisker_low_tp],
                        upperfence=[whisker_high_tp],
                        mean=[mean_tp] if show_mean else None,
                        boxmean=show_mean,
                        name="Take-Profit" if i == 0 else None,
                        legendgroup="tp",
                        showlegend=(i == 0),
                        marker_color=outcome_colors["tp"],
                        offsetgroup="tp",
                        hovertemplate=(
                            f"Quantile: {q}<br>"
                            f"Outcome: Take-Profit<br>"
                            f"Mean: {mean_tp:.1f} bars<br>"
                            f"Median: {median_tp:.1f} bars<br>"
                            f"Count: {count_tp}<br>"
                            "<extra></extra>"
                        ),
                    )
                )

            # SL box
            mean_sl = time_to_target_result.mean_bars_sl[q]
            median_sl = time_to_target_result.median_bars_sl[q]
            std_sl = time_to_target_result.std_bars_sl[q]
            count_sl = time_to_target_result.count_sl[q]

            if count_sl > 0 and not np.isnan(mean_sl):
                q1_sl = max(0, mean_sl - 0.675 * std_sl)
                q3_sl = mean_sl + 0.675 * std_sl
                whisker_low_sl = max(0, mean_sl - 1.5 * std_sl)
                whisker_high_sl = mean_sl + 1.5 * std_sl

                fig.add_trace(
                    go.Box(
                        x=[q],
                        q1=[q1_sl],
                        median=[median_sl],
                        q3=[q3_sl],
                        lowerfence=[whisker_low_sl],
                        upperfence=[whisker_high_sl],
                        mean=[mean_sl] if show_mean else None,
                        boxmean=show_mean,
                        name="Stop-Loss" if i == 0 else None,
                        legendgroup="sl",
                        showlegend=(i == 0),
                        marker_color=outcome_colors["sl"],
                        offsetgroup="sl",
                        hovertemplate=(
                            f"Quantile: {q}<br>"
                            f"Outcome: Stop-Loss<br>"
                            f"Mean: {mean_sl:.1f} bars<br>"
                            f"Median: {median_sl:.1f} bars<br>"
                            f"Count: {count_sl}<br>"
                            "<extra></extra>"
                        ),
                    )
                )

        fig.update_layout(boxmode="group")

    else:
        # Single outcome type or all
        if outcome_type == "tp":
            mean_bars = time_to_target_result.mean_bars_tp
            median_bars = time_to_target_result.median_bars_tp
            std_bars = time_to_target_result.std_bars_tp
            counts = time_to_target_result.count_tp
            color = outcome_colors["tp"]
        elif outcome_type == "sl":
            mean_bars = time_to_target_result.mean_bars_sl
            median_bars = time_to_target_result.median_bars_sl
            std_bars = time_to_target_result.std_bars_sl
            counts = time_to_target_result.count_sl
            color = outcome_colors["sl"]
        else:  # all
            mean_bars = time_to_target_result.mean_bars_all
            median_bars = time_to_target_result.median_bars_all
            std_bars = time_to_target_result.std_bars_all
            counts = {
                q: time_to_target_result.count_tp[q]
                + time_to_target_result.count_sl[q]
                + time_to_target_result.count_timeout[q]
                for q in quantile_labels
            }
            color = theme_config["colorway"][0]

        # Get quantile colors
        colors = _get_quantile_colors(n_quantiles, theme_config)

        for i, q in enumerate(quantile_labels):
            mean = mean_bars[q]
            median = median_bars[q]
            std = std_bars[q]
            count = counts[q]

            if count > 0 and not np.isnan(mean):
                q1 = max(0, mean - 0.675 * std)
                q3 = mean + 0.675 * std
                whisker_low = max(0, mean - 1.5 * std)
                whisker_high = mean + 1.5 * std

                fig.add_trace(
                    go.Box(
                        x=[q],
                        q1=[q1],
                        median=[median],
                        q3=[q3],
                        lowerfence=[whisker_low],
                        upperfence=[whisker_high],
                        mean=[mean] if show_mean else None,
                        boxmean=show_mean,
                        name=q,
                        showlegend=False,
                        marker_color=colors[i] if outcome_type == "all" else color,
                        hovertemplate=(
                            f"Quantile: {q}<br>"
                            f"Mean: {mean:.1f} bars<br>"
                            f"Median: {median:.1f} bars<br>"
                            f"Std: {std:.1f}<br>"
                            f"Count: {count}<br>"
                            "<extra></extra>"
                        ),
                    )
                )

    # Overall median line
    if show_median_line:
        overall_median = time_to_target_result.overall_median_bars
        fig.add_hline(
            y=overall_median,
            line_dash="dash",
            line_color="gray",
            line_width=2,
            annotation_text=f"Overall Median: {overall_median:.1f}",
            annotation_position="right",
            annotation={"font_size": 10, "font_color": "gray"},
        )

    # Summary annotation
    summary_text = (
        f"<b>Overall Statistics:</b><br>"
        f"Mean: {time_to_target_result.overall_mean_bars:.1f} bars<br>"
        f"Median: {time_to_target_result.overall_median_bars:.1f} bars<br>"
        f"TP Mean: {time_to_target_result.overall_mean_bars_tp:.1f} bars<br>"
        f"SL Mean: {time_to_target_result.overall_mean_bars_sl:.1f} bars"
    )

    fig.add_annotation(
        text=summary_text,
        xref="paper",
        yref="paper",
        x=0.98,
        y=0.98,
        showarrow=False,
        bgcolor="rgba(255,255,255,0.9)" if theme != "dark" else "rgba(50,50,50,0.9)",
        bordercolor="gray",
        borderwidth=1,
        align="left",
        xanchor="right",
        yanchor="top",
        font={"size": 10},
    )

    return fig


__all__ = [
    "plot_hit_rate_heatmap",
    "plot_profit_factor_bar",
    "plot_precision_recall_curve",
    "plot_time_to_target_box",
]
