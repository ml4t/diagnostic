"""Statistical validity visualizations for backtest analysis.

Provides interactive Plotly visualizations for statistical rigor:
- DSR (Deflated Sharpe Ratio) gauge with probability zones
- Confidence interval forest plots
- RAS (Rademacher Anti-Serum) overfitting detection
- MinTRL (Minimum Track Record Length) analysis

These visualizations help traders understand whether their backtest results
are statistically significant or likely due to overfitting/chance.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import plotly.graph_objects as go

from ml4t.diagnostic.visualization.core import get_theme_config

if TYPE_CHECKING:
    pass


def plot_dsr_gauge(
    dsr_probability: float,
    observed_sharpe: float,
    expected_max_sharpe: float | None = None,
    n_trials: int | None = None,
    title: str = "Deflated Sharpe Ratio",
    show_legend: bool = True,
    theme: str | None = None,
    height: int = 350,
    width: int = 500,
) -> go.Figure:
    """Create a gauge chart showing DSR probability.

    The Deflated Sharpe Ratio corrects for selection bias when choosing
    the best strategy from multiple tests. A DSR probability < 0.05
    suggests the performance is statistically significant.

    Parameters
    ----------
    dsr_probability : float
        DSR probability value (0-1), where lower is more significant.
        Typically displayed as 1 - dsr for "confidence" interpretation.
    observed_sharpe : float
        The observed Sharpe ratio being tested
    expected_max_sharpe : float, optional
        The expected maximum Sharpe under null hypothesis
    n_trials : int, optional
        Number of trials/strategies tested (for annotation)
    title : str
        Chart title
    show_legend : bool
        Whether to show the color zone legend
    theme : str, optional
        Theme name (default, dark, print, presentation)
    height : int
        Figure height in pixels
    width : int
        Figure width in pixels

    Returns
    -------
    go.Figure
        Plotly figure with gauge chart

    Examples
    --------
    >>> fig = plot_dsr_gauge(
    ...     dsr_probability=0.03,
    ...     observed_sharpe=2.1,
    ...     n_trials=100,
    ... )
    >>> fig.show()
    """
    theme_config = get_theme_config(theme)

    # Convert to "confidence" (1 - p-value style)
    # High confidence = good, Low confidence = bad
    confidence = (1 - dsr_probability) * 100

    # Color zones: Red (not significant) -> Yellow (marginal) -> Green (significant)
    # Standard thresholds: p < 0.05 (95%), p < 0.01 (99%)
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=confidence,
            number={"suffix": "%", "font": {"size": 36}},
            title={"text": title, "font": {"size": 18}},
            gauge={
                "axis": {
                    "range": [0, 100],
                    "tickwidth": 1,
                    "tickcolor": "darkgray",
                    "tickvals": [0, 50, 90, 95, 99, 100],
                    "ticktext": ["0%", "50%", "90%", "95%", "99%", "100%"],
                },
                "bar": {"color": "darkblue"},
                "bgcolor": "white",
                "borderwidth": 2,
                "bordercolor": "gray",
                "steps": [
                    {"range": [0, 50], "color": "#EF553B"},  # Red - not significant
                    {"range": [50, 90], "color": "#FFA15A"},  # Orange - weak
                    {"range": [90, 95], "color": "#FECB52"},  # Yellow - marginal
                    {"range": [95, 99], "color": "#00CC96"},  # Green - significant
                    {"range": [99, 100], "color": "#19D3F3"},  # Cyan - highly significant
                ],
                "threshold": {
                    "line": {"color": "black", "width": 4},
                    "thickness": 0.75,
                    "value": confidence,
                },
            },
        )
    )

    # Add annotations
    annotations = []

    # DSR probability annotation
    annotations.append(
        {
            "x": 0.5,
            "y": 0.25,
            "text": f"DSR p-value: {dsr_probability:.4f}",
            "showarrow": False,
            "font": {"size": 14},
            "xref": "paper",
            "yref": "paper",
        }
    )

    # Observed Sharpe
    annotations.append(
        {
            "x": 0.5,
            "y": 0.15,
            "text": f"Observed Sharpe: {observed_sharpe:.2f}",
            "showarrow": False,
            "font": {"size": 12},
            "xref": "paper",
            "yref": "paper",
        }
    )

    # Expected max Sharpe if provided
    if expected_max_sharpe is not None:
        annotations.append(
            {
                "x": 0.5,
                "y": 0.08,
                "text": f"E[max SR]: {expected_max_sharpe:.2f}",
                "showarrow": False,
                "font": {"size": 12},
                "xref": "paper",
                "yref": "paper",
            }
        )

    # Number of trials
    if n_trials is not None:
        annotations.append(
            {
                "x": 0.5,
                "y": 0.01,
                "text": f"(N={n_trials} trials)",
                "showarrow": False,
                "font": {"size": 11, "color": "gray"},
                "xref": "paper",
                "yref": "paper",
            }
        )

    # Build layout
    layout_updates = {
        "height": height,
        "width": width,
        "annotations": annotations,
        "margin": {"l": 40, "r": 40, "t": 60, "b": 40},
    }

    for key, value in theme_config["layout"].items():
        if key not in layout_updates:
            layout_updates[key] = value

    fig.update_layout(**layout_updates)

    return fig


def plot_confidence_intervals(
    metrics: dict[str, dict[str, float]],
    confidence_levels: list[float] | None = None,
    title: str = "Metric Confidence Intervals",
    orientation: Literal["h", "v"] = "h",
    show_point_estimate: bool = True,
    theme: str | None = None,
    height: int = 400,
    width: int | None = None,
) -> go.Figure:
    """Create a forest plot showing confidence intervals for multiple metrics.

    Visualizes bootstrap or analytical confidence intervals at multiple
    confidence levels (e.g., 90%, 95%, 99%).

    Parameters
    ----------
    metrics : dict[str, dict[str, float]]
        Dictionary mapping metric names to their CI values.
        Each value should have keys: 'point', 'lower_90', 'upper_90',
        'lower_95', 'upper_95', 'lower_99', 'upper_99' (based on levels).
    confidence_levels : list[float], optional
        Confidence levels to display (default: [0.90, 0.95, 0.99])
    title : str
        Chart title
    orientation : {"h", "v"}
        Horizontal or vertical orientation
    show_point_estimate : bool
        Whether to show the point estimate marker
    theme : str, optional
        Theme name
    height : int
        Figure height in pixels
    width : int, optional
        Figure width in pixels

    Returns
    -------
    go.Figure
        Plotly figure with forest plot

    Examples
    --------
    >>> metrics = {
    ...     "Sharpe": {"point": 1.5, "lower_95": 0.8, "upper_95": 2.2},
    ...     "CAGR": {"point": 0.15, "lower_95": 0.08, "upper_95": 0.22},
    ... }
    >>> fig = plot_confidence_intervals(metrics)
    >>> fig.show()
    """
    theme_config = get_theme_config(theme)
    colors = theme_config["colorway"]

    if confidence_levels is None:
        confidence_levels = [0.90, 0.95, 0.99]

    # Sort confidence levels (widest first for plotting)
    confidence_levels = sorted(confidence_levels, reverse=True)

    fig = go.Figure()

    metric_names = list(metrics.keys())
    n_metrics = len(metric_names)

    # Colors for different confidence levels (lighter to darker)
    level_colors = {
        0.99: "rgba(99, 110, 250, 0.3)",  # Lightest - widest CI
        0.95: "rgba(99, 110, 250, 0.5)",
        0.90: "rgba(99, 110, 250, 0.7)",  # Darkest - narrowest CI
    }

    for i, metric_name in enumerate(metric_names):
        metric_data = metrics[metric_name]
        point = metric_data.get("point", metric_data.get("estimate", 0))

        # Plot confidence intervals from widest to narrowest
        for level in confidence_levels:
            level_pct = int(level * 100)
            lower_key = f"lower_{level_pct}"
            upper_key = f"upper_{level_pct}"

            if lower_key in metric_data and upper_key in metric_data:
                lower = metric_data[lower_key]
                upper = metric_data[upper_key]

                color = level_colors.get(level, "rgba(99, 110, 250, 0.5)")

                if orientation == "h":
                    fig.add_trace(
                        go.Scatter(
                            x=[lower, upper],
                            y=[i, i],
                            mode="lines",
                            line={"color": color, "width": 8 if level == 0.95 else 5},
                            name=f"{level_pct}% CI" if i == 0 else None,
                            showlegend=(i == 0),
                            hovertemplate=f"{metric_name}<br>{level_pct}% CI: [{lower:.3f}, {upper:.3f}]<extra></extra>",
                        )
                    )
                else:
                    fig.add_trace(
                        go.Scatter(
                            x=[i, i],
                            y=[lower, upper],
                            mode="lines",
                            line={"color": color, "width": 8 if level == 0.95 else 5},
                            name=f"{level_pct}% CI" if i == 0 else None,
                            showlegend=(i == 0),
                            hovertemplate=f"{metric_name}<br>{level_pct}% CI: [{lower:.3f}, {upper:.3f}]<extra></extra>",
                        )
                    )

        # Add point estimate
        if show_point_estimate:
            if orientation == "h":
                fig.add_trace(
                    go.Scatter(
                        x=[point],
                        y=[i],
                        mode="markers",
                        marker={"color": colors[0], "size": 12, "symbol": "diamond"},
                        name="Point Estimate" if i == 0 else None,
                        showlegend=(i == 0),
                        hovertemplate=f"{metric_name}: {point:.3f}<extra></extra>",
                    )
                )
            else:
                fig.add_trace(
                    go.Scatter(
                        x=[i],
                        y=[point],
                        mode="markers",
                        marker={"color": colors[0], "size": 12, "symbol": "diamond"},
                        name="Point Estimate" if i == 0 else None,
                        showlegend=(i == 0),
                        hovertemplate=f"{metric_name}: {point:.3f}<extra></extra>",
                    )
                )

    # Add zero reference line for Sharpe-like metrics
    if orientation == "h":
        fig.add_vline(x=0, line_dash="dash", line_color="gray", line_width=1)
    else:
        fig.add_hline(y=0, line_dash="dash", line_color="gray", line_width=1)

    # Build layout
    if orientation == "h":
        layout_updates = {
            "title": {"text": title, "font": {"size": 18}},
            "height": max(height, n_metrics * 60 + 100),
            "xaxis": {"title": "Value", "zeroline": True},
            "yaxis": {
                "tickvals": list(range(n_metrics)),
                "ticktext": metric_names,
                "autorange": "reversed",
            },
            "legend": {"yanchor": "top", "y": 0.99, "xanchor": "right", "x": 0.99},
        }
    else:
        layout_updates = {
            "title": {"text": title, "font": {"size": 18}},
            "height": height,
            "yaxis": {"title": "Value", "zeroline": True},
            "xaxis": {
                "tickvals": list(range(n_metrics)),
                "ticktext": metric_names,
            },
            "legend": {"yanchor": "top", "y": 0.99, "xanchor": "right", "x": 0.99},
        }
    if width:
        layout_updates["width"] = width

    for key, value in theme_config["layout"].items():
        if key not in layout_updates:
            layout_updates[key] = value

    fig.update_layout(**layout_updates)

    return fig


def plot_ras_analysis(
    original_ic: float,
    adjusted_ic: float,
    rademacher_complexity: float,
    kappa: float = 0.02,
    n_features: int | None = None,
    n_observations: int | None = None,
    title: str = "Rademacher Anti-Serum Analysis",
    theme: str | None = None,
    height: int = 400,
    width: int = 600,
) -> go.Figure:
    """Visualize Rademacher Anti-Serum (RAS) overfitting adjustment.

    The RAS method adjusts Information Coefficients for data mining bias
    by estimating the Rademacher complexity of the strategy search space.

    Parameters
    ----------
    original_ic : float
        Original (unadjusted) Information Coefficient
    adjusted_ic : float
        RAS-adjusted Information Coefficient
    rademacher_complexity : float
        Estimated Rademacher complexity R̂
    kappa : float
        The practical bound parameter used (default: 0.02)
    n_features : int, optional
        Number of features/strategies tested
    n_observations : int, optional
        Number of observations
    title : str
        Chart title
    theme : str, optional
        Theme name
    height : int
        Figure height in pixels
    width : int
        Figure width in pixels

    Returns
    -------
    go.Figure
        Plotly figure with RAS analysis

    Notes
    -----
    The RAS adjustment is:
    IC_adj = max(0, IC_original - 2 * (R̂ + κ))

    where R̂ is the Rademacher complexity and κ is a practical bound.
    """
    theme_config = get_theme_config(theme)
    colors = theme_config["colorway"]

    # Calculate the haircut percentage
    haircut_pct = (1 - adjusted_ic / original_ic) * 100 if original_ic != 0 else 100

    # Create waterfall chart
    fig = go.Figure()

    categories = ["Original IC", "Rademacher (2R̂)", "Practical κ", "Adjusted IC"]
    values = [original_ic, -2 * rademacher_complexity, -2 * kappa, adjusted_ic]
    measures = ["absolute", "relative", "relative", "total"]

    fig.add_trace(
        go.Waterfall(
            name="RAS Adjustment",
            orientation="v",
            x=categories,
            y=values,
            measure=measures,
            text=[f"{v:.4f}" for v in values],
            textposition="outside",
            decreasing={"marker": {"color": "#EF553B"}},
            increasing={"marker": {"color": colors[0]}},
            totals={"marker": {"color": "#00CC96" if adjusted_ic > 0 else "#EF553B"}},
            connector={"line": {"color": "rgba(128, 128, 128, 0.5)", "width": 2}},
        )
    )

    # Add annotations
    annotations = []

    # Haircut percentage
    annotations.append(
        {
            "x": 0.5,
            "y": -0.15,
            "text": f"IC Haircut: {haircut_pct:.1f}%  |  R̂ = {rademacher_complexity:.4f}  |  κ = {kappa:.4f}",
            "showarrow": False,
            "font": {"size": 12},
            "xref": "paper",
            "yref": "paper",
        }
    )

    # Significance indicator
    if adjusted_ic > 0:
        sig_text = "Statistically significant after RAS adjustment"
        sig_color = "#00CC96"
    else:
        sig_text = "Not significant after RAS adjustment (IC ≤ 0)"
        sig_color = "#EF553B"

    annotations.append(
        {
            "x": 0.5,
            "y": -0.22,
            "text": sig_text,
            "showarrow": False,
            "font": {"size": 13, "color": sig_color, "weight": "bold"},
            "xref": "paper",
            "yref": "paper",
        }
    )

    # N and T if provided
    if n_features is not None and n_observations is not None:
        annotations.append(
            {
                "x": 0.5,
                "y": 1.08,
                "text": f"N={n_features} features, T={n_observations} observations",
                "showarrow": False,
                "font": {"size": 11, "color": "gray"},
                "xref": "paper",
                "yref": "paper",
            }
        )

    # Build layout
    layout_updates = {
        "title": {"text": title, "font": {"size": 18}},
        "height": height,
        "width": width,
        "yaxis": {"title": "Information Coefficient"},
        "showlegend": False,
        "annotations": annotations,
        "margin": {"l": 60, "r": 40, "t": 80, "b": 100},
    }

    for key, value in theme_config["layout"].items():
        if key not in layout_updates:
            layout_updates[key] = value

    fig.update_layout(**layout_updates)

    return fig


def plot_minimum_track_record(
    observed_sharpe: float,
    current_periods: int,
    sr_benchmark: float = 0.0,
    confidence: float = 0.95,
    max_periods: int | None = None,
    periods_per_year: int = 252,
    title: str = "Minimum Track Record Length",
    theme: str | None = None,
    height: int = 400,
    width: int | None = None,
) -> go.Figure:
    """Visualize minimum track record length (MinTRL) analysis.

    Shows how many periods are needed to achieve statistical significance
    for the observed Sharpe ratio, and whether the current track record
    is sufficient.

    Parameters
    ----------
    observed_sharpe : float
        The observed Sharpe ratio (annualized)
    current_periods : int
        Current number of observation periods
    sr_benchmark : float
        Benchmark Sharpe ratio for comparison (default: 0)
    confidence : float
        Target confidence level (default: 0.95)
    max_periods : int, optional
        Maximum periods to show on x-axis
    periods_per_year : int
        Periods per year for time conversion (default: 252 for daily)
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
        Plotly figure with MinTRL analysis

    Notes
    -----
    The minimum track record length formula is:
    MinTRL = 1 + (1 - γ₃*SR + γ₄*SR²/4) * (z_α / SR)²

    where γ₃ is skewness, γ₄ is excess kurtosis, and z_α is the
    critical value for confidence level α.
    """
    from scipy import stats

    theme_config = get_theme_config(theme)
    colors = theme_config["colorway"]

    # Calculate MinTRL (simplified, assuming normal returns)
    z_alpha = stats.norm.ppf(confidence)
    sharpe_diff = observed_sharpe - sr_benchmark

    if sharpe_diff <= 0:
        min_trl = float("inf")
    else:
        # Simplified MinTRL (assuming γ₃=0, γ₄=3)
        min_trl = (z_alpha / sharpe_diff) ** 2

    # Convert to years
    min_trl_years = min_trl / periods_per_year if min_trl != float("inf") else float("inf")
    current_years = current_periods / periods_per_year

    # Determine max periods for x-axis
    if max_periods is None:
        if min_trl != float("inf"):
            max_periods = int(max(min_trl * 1.5, current_periods * 1.2))
        else:
            max_periods = current_periods * 2

    # Generate data for the required SR curve at different track record lengths
    periods_range = np.linspace(10, max_periods, 100)

    # Required SR to achieve significance at each track record length
    # SR_required = z_alpha / sqrt(T)
    required_sr = z_alpha / np.sqrt(periods_range) + sr_benchmark

    fig = go.Figure()

    # Required SR curve
    fig.add_trace(
        go.Scatter(
            x=periods_range / periods_per_year,
            y=required_sr,
            mode="lines",
            name=f"{int(confidence * 100)}% Significance Threshold",
            line={"color": colors[1] if len(colors) > 1 else "orange", "width": 2, "dash": "dash"},
            fill="tozeroy",
            fillcolor="rgba(239, 85, 59, 0.2)",
            hovertemplate="Track Record: %{x:.1f} years<br>Required SR: %{y:.2f}<extra></extra>",
        )
    )

    # Horizontal line at observed Sharpe
    fig.add_trace(
        go.Scatter(
            x=[0, max_periods / periods_per_year],
            y=[observed_sharpe, observed_sharpe],
            mode="lines",
            name=f"Observed SR: {observed_sharpe:.2f}",
            line={"color": colors[0], "width": 3},
            hovertemplate="Observed Sharpe: %{y:.2f}<extra></extra>",
        )
    )

    # Current position marker
    is_significant = current_periods >= min_trl
    marker_color = "#00CC96" if is_significant else "#EF553B"

    fig.add_trace(
        go.Scatter(
            x=[current_years],
            y=[observed_sharpe],
            mode="markers",
            name="Current Position",
            marker={"color": marker_color, "size": 15, "symbol": "star"},
            hovertemplate=f"Current: {current_years:.1f} years<br>SR: {observed_sharpe:.2f}<extra></extra>",
        )
    )

    # Add vertical line at MinTRL
    if min_trl != float("inf") and min_trl <= max_periods:
        fig.add_vline(
            x=min_trl_years,
            line_dash="dot",
            line_color="gray",
            annotation_text=f"MinTRL: {min_trl_years:.1f}y",
            annotation_position="top",
        )

    # Add significance zone annotation
    annotations = []

    if is_significant:
        status_text = (
            f"Track record sufficient ({current_years:.1f}y ≥ MinTRL {min_trl_years:.1f}y)"
        )
        status_color = "#00CC96"
    elif min_trl == float("inf"):
        status_text = "Cannot achieve significance (SR ≤ benchmark)"
        status_color = "#EF553B"
    else:
        deficit = min_trl_years - current_years
        status_text = f"Need {deficit:.1f} more years (MinTRL: {min_trl_years:.1f}y)"
        status_color = "#FFA15A"

    annotations.append(
        {
            "x": 0.5,
            "y": -0.15,
            "text": status_text,
            "showarrow": False,
            "font": {"size": 13, "color": status_color, "weight": "bold"},
            "xref": "paper",
            "yref": "paper",
        }
    )

    # Build layout
    layout_updates = {
        "title": {"text": title, "font": {"size": 18}},
        "height": height,
        "xaxis": {"title": "Track Record Length (Years)", "rangemode": "tozero"},
        "yaxis": {"title": "Sharpe Ratio", "rangemode": "tozero"},
        "legend": {"yanchor": "top", "y": 0.99, "xanchor": "right", "x": 0.99},
        "annotations": annotations,
        "margin": {"b": 80},
    }
    if width:
        layout_updates["width"] = width

    for key, value in theme_config["layout"].items():
        if key not in layout_updates:
            layout_updates[key] = value

    fig.update_layout(**layout_updates)

    return fig


def plot_statistical_summary_card(
    metrics: dict[str, Any],
    title: str = "Statistical Validity Summary",
    theme: str | None = None,
    height: int = 300,
    width: int = 700,
) -> go.Figure:
    """Create an executive summary card for statistical validity checks.

    Combines multiple statistical tests into a single traffic-light display
    showing overall strategy validity.

    Parameters
    ----------
    metrics : dict[str, Any]
        Dictionary with statistical metrics. Expected keys:
        - dsr_probability: DSR p-value
        - dsr_significant: bool
        - min_trl: minimum track record length
        - current_trl: current track record length
        - trl_sufficient: bool
        - ras_adjusted_ic: RAS-adjusted IC (optional)
        - ras_significant: bool (optional)
    title : str
        Chart title
    theme : str, optional
        Theme name
    height : int
        Figure height in pixels
    width : int
        Figure width in pixels

    Returns
    -------
    go.Figure
        Plotly figure with summary card
    """
    theme_config = get_theme_config(theme)

    # Extract metrics with defaults
    dsr_prob = metrics.get("dsr_probability", None)
    dsr_sig = metrics.get("dsr_significant", None)
    min_trl = metrics.get("min_trl", None)
    current_trl = metrics.get("current_trl", None)
    trl_sufficient = metrics.get("trl_sufficient", None)
    ras_ic = metrics.get("ras_adjusted_ic", None)
    ras_sig = metrics.get("ras_significant", None)

    # Build indicators
    indicators = []

    # DSR check
    if dsr_prob is not None:
        if dsr_sig:
            indicators.append(("DSR", f"p={dsr_prob:.3f}", "green", "Significant"))
        elif dsr_prob < 0.10:
            indicators.append(("DSR", f"p={dsr_prob:.3f}", "yellow", "Marginal"))
        else:
            indicators.append(("DSR", f"p={dsr_prob:.3f}", "red", "Not Significant"))

    # MinTRL check
    if min_trl is not None and current_trl is not None:
        if trl_sufficient:
            indicators.append(
                ("Track Record", f"{current_trl:.0f}/{min_trl:.0f}", "green", "Sufficient")
            )
        else:
            indicators.append(
                ("Track Record", f"{current_trl:.0f}/{min_trl:.0f}", "red", "Insufficient")
            )

    # RAS check
    if ras_ic is not None:
        if ras_sig:
            indicators.append(("RAS IC", f"{ras_ic:.4f}", "green", "Significant"))
        else:
            indicators.append(("RAS IC", f"{ras_ic:.4f}", "red", "Not Significant"))

    if not indicators:
        indicators = [("No Data", "-", "gray", "No statistical tests available")]

    # Create table-like figure
    n_cols = len(indicators)

    # Color mapping
    color_map = {
        "green": "#00CC96",
        "yellow": "#FECB52",
        "red": "#EF553B",
        "gray": "#888888",
    }

    fig = go.Figure()

    for i, (name, value, color, status) in enumerate(indicators):
        x_pos = (i + 0.5) / n_cols

        # Status icon (colored circle)
        fig.add_annotation(
            x=x_pos,
            y=0.75,
            text="●",
            showarrow=False,
            font={"size": 40, "color": color_map[color]},
            xref="paper",
            yref="paper",
        )

        # Metric name
        fig.add_annotation(
            x=x_pos,
            y=0.5,
            text=f"<b>{name}</b>",
            showarrow=False,
            font={"size": 14},
            xref="paper",
            yref="paper",
        )

        # Value
        fig.add_annotation(
            x=x_pos,
            y=0.35,
            text=value,
            showarrow=False,
            font={"size": 12},
            xref="paper",
            yref="paper",
        )

        # Status text
        fig.add_annotation(
            x=x_pos,
            y=0.2,
            text=status,
            showarrow=False,
            font={"size": 11, "color": color_map[color]},
            xref="paper",
            yref="paper",
        )

    # Build layout
    layout_updates = {
        "title": {"text": title, "font": {"size": 18}, "x": 0.5},
        "height": height,
        "width": width,
        "xaxis": {"visible": False, "range": [0, 1]},
        "yaxis": {"visible": False, "range": [0, 1]},
        "margin": {"l": 20, "r": 20, "t": 60, "b": 20},
    }

    for key, value in theme_config["layout"].items():
        if key not in layout_updates:
            layout_updates[key] = value

    fig.update_layout(**layout_updates)

    return fig
