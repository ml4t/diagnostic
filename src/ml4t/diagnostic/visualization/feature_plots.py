"""Feature importance visualization functions.

This module provides functions for visualizing ML feature importance analysis results
from analyze_ml_importance() and related functions.

All plot functions follow the standard API defined in docs/plot_api_standards.md:
- Consume results dicts from analyze_*() functions
- Return plotly.graph_objects.Figure instances
- Support theme customization via global or per-plot settings
- Use keyword-only arguments (after results)
- Provide comprehensive hover information and interactivity

Example workflow:
    >>> from ml4t.diagnostic.evaluation import analyze_ml_importance
    >>> from ml4t.diagnostic.visualization import plot_importance_bar, set_plot_theme
    >>>
    >>> # Analyze feature importance
    >>> results = analyze_ml_importance(model, X, y, methods=["mdi", "pfi"])
    >>>
    >>> # Set global theme
    >>> set_plot_theme("dark")
    >>>
    >>> # Create visualizations
    >>> fig_bar = plot_importance_bar(results, top_n=15)
    >>> fig_heatmap = plot_importance_heatmap(results)
    >>> fig_dist = plot_importance_distribution(results)
    >>> fig_summary = plot_importance_summary(results)
    >>>
    >>> # Display or save
    >>> fig_bar.show()
    >>> fig_summary.write_html("importance_report.html")
"""

from typing import Any

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ml4t.diagnostic.visualization.core import (
    apply_responsive_layout,
    format_number,
    get_color_scheme,
    get_colorscale,
    get_theme_config,
    validate_plot_results,
    validate_positive_int,
    validate_theme,
)

__all__ = [
    "plot_importance_bar",
    "plot_importance_heatmap",
    "plot_importance_distribution",
    "plot_importance_summary",
]


def plot_importance_bar(
    results: dict[str, Any],
    *,
    title: str | None = None,
    top_n: int | None = 20,
    theme: str | None = None,
    color_scheme: str | None = None,
    width: int | None = None,
    height: int | None = None,
    show_values: bool = True,
) -> go.Figure:
    """Plot horizontal bar chart of consensus feature importance rankings.

    Creates an interactive bar chart showing features ranked by consensus importance
    (average rank across all methods). Bars are color-coded by importance score using
    a continuous colorscale.

    Parameters
    ----------
    results : dict[str, Any]
        Results from analyze_ml_importance() containing:
        - "consensus_ranking": list[str] - Features in order of importance
        - "method_results": dict - Individual method results with importances
    title : str | None, optional
        Plot title. If None, uses "Feature Importance - Consensus Ranking"
    top_n : int | None, optional
        Number of top features to display. If None, shows all features.
        Default is 20 to avoid overcrowding.
    theme : str | None, optional
        Theme name ("default", "dark", "print", "presentation").
        If None, uses current global theme.
    color_scheme : str | None, optional
        Color scheme for bars. If None, uses "viridis".
        Recommended: "viridis", "cividis", "plasma", "blues", "greens"
    width : int | None, optional
        Figure width in pixels. If None, uses theme default (typically 1000).
    height : int | None, optional
        Figure height in pixels. If None, auto-sizes based on feature count
        (25px per feature + 100px padding).
    show_values : bool, optional
        Whether to show importance values on bars. Default is True.

    Returns
    -------
    go.Figure
        Interactive Plotly figure with:
        - Horizontal bars sorted by consensus importance
        - Continuous color gradient indicating importance scores
        - Hover info showing exact importance values
        - Responsive layout for different screen sizes

    Raises
    ------
    ValueError
        If results dict is missing required keys or has invalid structure.
    TypeError
        If parameters have incorrect types.

    Examples
    --------
    >>> from ml4t.diagnostic.evaluation import analyze_ml_importance
    >>> from ml4t.diagnostic.visualization import plot_importance_bar
    >>>
    >>> # Analyze importance
    >>> results = analyze_ml_importance(model, X, y)
    >>>
    >>> # Plot top 10 features
    >>> fig = plot_importance_bar(results, top_n=10)
    >>> fig.show()
    >>>
    >>> # Custom styling for print
    >>> fig = plot_importance_bar(
    ...     results,
    ...     title="Key Predictive Features",
    ...     top_n=15,
    ...     theme="print",
    ...     color_scheme="blues",
    ...     height=600
    ... )
    >>> fig.write_image("feature_importance.pdf")

    Notes
    -----
    - Importance scores are computed as the mean importance across all methods
    - Features are ranked by consensus (average rank), not absolute importance
    - Use top_n to focus on most important features and improve readability
    - For very long feature names, consider increasing width parameter
    """
    # Validate inputs
    validate_plot_results(
        results,
        required_keys=["consensus_ranking", "method_results"],
        function_name="plot_importance_bar",
    )
    theme = validate_theme(theme)
    if top_n is not None:
        validate_positive_int(top_n, "top_n")

    # Note: color_scheme validation happens in get_colorscale()

    # Extract data
    all_features = results["consensus_ranking"]
    features = all_features[:top_n] if top_n is not None else all_features

    # Calculate average importance across methods for each feature
    method_results = results["method_results"]
    importance_scores = []

    for feat in features:
        scores = []
        for method_name, method_result in method_results.items():
            # Get feature importances from method result
            if method_name == "pfi":
                # PFI uses importances_mean
                importances = method_result["importances_mean"]
            else:
                # MDI, MDA, SHAP use importances
                importances = method_result["importances"]

            # Get feature names for this method
            method_features = method_result["feature_names"]

            # Find this feature's importance
            if feat in method_features:
                idx = method_features.index(feat)
                scores.append(importances[idx])

        # Average importance across methods
        if scores:
            importance_scores.append(float(np.mean(scores)))
        else:
            importance_scores.append(0.0)

    # Get theme configuration
    theme_config = get_theme_config(theme)

    # Get colors
    colors = get_colorscale(color_scheme or "viridis")

    # Create figure
    fig = go.Figure()

    # Add bar trace
    fig.add_trace(
        go.Bar(
            x=importance_scores,
            y=features,
            orientation="h",
            marker={
                "color": importance_scores,
                "colorscale": colors,
                "showscale": True,
                "colorbar": {
                    "title": "Importance",
                    "tickformat": ".3f",
                },
            },
            text=[format_number(v, precision=3) for v in importance_scores]
            if show_values
            else None,
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>",
        )
    )

    # Update layout
    fig.update_layout(
        title=title or "Feature Importance - Consensus Ranking",
        xaxis_title="Consensus Importance Score",
        yaxis_title="Features",
        **theme_config["layout"],
        width=width or 1000,
        height=height or max(400, len(features) * 25 + 100),
        showlegend=False,
    )

    # Apply responsive layout
    apply_responsive_layout(fig)

    return fig


def plot_importance_heatmap(
    results: dict[str, Any],
    *,
    title: str | None = None,
    theme: str | None = None,
    color_scheme: str | None = None,
    width: int | None = None,
    height: int | None = None,
    show_values: bool = True,
) -> go.Figure:
    """Plot heatmap showing correlation between importance ranking methods.

    Creates a symmetric correlation matrix showing Spearman rank correlations between
    different feature importance methods (MDI, PFI, MDA, SHAP). High correlations
    indicate method agreement; low correlations suggest different aspects being measured.

    Parameters
    ----------
    results : dict[str, Any]
        Results from analyze_ml_importance() containing:
        - "method_agreement": dict - Pairwise Spearman correlations
        - "methods_run": list[str] - Names of methods that ran successfully
    title : str | None, optional
        Plot title. If None, uses "Method Agreement - Ranking Correlations"
    theme : str | None, optional
        Theme name ("default", "dark", "print", "presentation").
        If None, uses current global theme.
    color_scheme : str | None, optional
        Diverging color scheme for correlation values. If None, uses "rdbu".
        Recommended: "rdbu", "rdylgn", "brbg", "blues_oranges"
    width : int | None, optional
        Figure width in pixels. If None, uses 800.
    height : int | None, optional
        Figure height in pixels. If None, uses 800.
    show_values : bool, optional
        Whether to show correlation values in cells. Default is True.

    Returns
    -------
    go.Figure
        Interactive Plotly heatmap with:
        - Symmetric correlation matrix
        - Diverging colorscale (red = negative, blue = positive)
        - Annotated cells with correlation coefficients
        - Hover showing method pairs and correlation

    Raises
    ------
    ValueError
        If results dict is missing required keys or has invalid structure.
        If fewer than 2 methods were run (can't compute correlations).
    TypeError
        If parameters have incorrect types.

    Examples
    --------
    >>> from ml4t.diagnostic.evaluation import analyze_ml_importance
    >>> from ml4t.diagnostic.visualization import plot_importance_heatmap
    >>>
    >>> # Analyze with multiple methods
    >>> results = analyze_ml_importance(
    ...     model, X, y,
    ...     methods=["mdi", "pfi", "shap"]
    ... )
    >>>
    >>> # Plot method agreement
    >>> fig = plot_importance_heatmap(results)
    >>> fig.show()
    >>>
    >>> # Custom styling
    >>> fig = plot_importance_heatmap(
    ...     results,
    ...     title="Feature Ranking Method Correlations",
    ...     theme="presentation",
    ...     color_scheme="rdylgn"
    ... )

    Notes
    -----
    - Correlations range from -1 (perfect disagreement) to +1 (perfect agreement)
    - High correlations (>0.7) indicate methods are measuring similar aspects
    - Low correlations (<0.5) suggest methods capture different information
    - Diagonal is always 1.0 (perfect self-correlation)
    - Matrix is symmetric (corr(A,B) = corr(B,A))
    """
    # Validate inputs
    validate_plot_results(
        results,
        required_keys=["method_agreement", "methods_run"],
        function_name="plot_importance_heatmap",
    )
    theme = validate_theme(theme)

    # Note: color_scheme validation happens in get_colorscale()

    methods = results["methods_run"]
    if len(methods) < 2:
        raise ValueError(f"plot_importance_heatmap requires at least 2 methods, got {len(methods)}")

    # Build correlation matrix from pairwise comparisons
    n_methods = len(methods)
    correlation_matrix = np.eye(n_methods)  # Diagonal = 1.0

    method_agreement = results["method_agreement"]

    for i, method1 in enumerate(methods):
        for j, method2 in enumerate(methods):
            if i < j:  # Upper triangle
                key1 = f"{method1}_vs_{method2}"
                key2 = f"{method2}_vs_{method1}"

                # Try both key orders
                if key1 in method_agreement:
                    corr = method_agreement[key1]
                elif key2 in method_agreement:
                    corr = method_agreement[key2]
                else:
                    # Shouldn't happen, but handle gracefully
                    corr = 0.0

                correlation_matrix[i, j] = corr
                correlation_matrix[j, i] = corr  # Symmetric

    # Get theme configuration
    theme_config = get_theme_config(theme)

    # Get colors (diverging colorscale for correlations)
    colors = get_colorscale(color_scheme or "rdbu")

    # Create figure
    fig = go.Figure()

    # Create hover text
    hover_text = []
    for i, method1 in enumerate(methods):
        row = []
        for j, method2 in enumerate(methods):
            corr = correlation_matrix[i, j]
            row.append(
                f"<b>{method1.upper()}</b> vs <b>{method2.upper()}</b><br>Correlation: {corr:.3f}"
            )
        hover_text.append(row)

    # Add heatmap trace
    fig.add_trace(
        go.Heatmap(
            z=correlation_matrix,
            x=[m.upper() for m in methods],
            y=[m.upper() for m in methods],
            colorscale=colors,
            zmid=0,  # Center diverging scale at 0
            zmin=-1,
            zmax=1,
            colorbar={
                "title": "Correlation",
                "tickmode": "linear",
                "tick0": -1,
                "dtick": 0.5,
            },
            text=np.round(correlation_matrix, 3) if show_values else None,
            texttemplate="%{text}" if show_values else None,
            textfont={"size": 12},
            hovertext=hover_text,
            hovertemplate="%{hovertext}<extra></extra>",
        )
    )

    # Update layout
    fig.update_layout(
        title=title or "Method Agreement - Ranking Correlations",
        xaxis={
            "title": "",
            "side": "bottom",
        },
        yaxis={
            "title": "",
            "autorange": "reversed",  # Top to bottom
        },
        **theme_config["layout"],
        width=width or 800,
        height=height or 800,
    )

    # Apply responsive layout
    apply_responsive_layout(fig)

    return fig


def plot_importance_distribution(
    results: dict[str, Any],
    *,
    title: str | None = None,
    method: str | None = None,
    theme: str | None = None,
    color_scheme: str | None = None,
    width: int | None = None,
    height: int | None = None,
    bins: int = 30,
    overlay: bool = False,
) -> go.Figure:
    """Plot distribution of feature importance scores across methods.

    Creates histogram(s) showing the distribution of importance scores. Can either
    overlay all methods in a single plot or show them separately in subplots.
    Useful for understanding the spread and concentration of importance values.

    Parameters
    ----------
    results : dict[str, Any]
        Results from analyze_ml_importance() containing:
        - "method_results": dict - Individual method results with importances
        - "methods_run": list[str] - Names of methods that ran successfully
    title : str | None, optional
        Plot title. If None, uses "Feature Importance Distribution"
    method : str | None, optional
        Show distribution for a single method only. If None, shows all methods.
        Valid values: "mdi", "pfi", "mda", "shap" (must be in methods_run)
    theme : str | None, optional
        Theme name ("default", "dark", "print", "presentation").
        If None, uses current global theme.
    color_scheme : str | None, optional
        Color scheme for histogram bars. If None, uses "set2".
        Recommended: "set2", "set3", "pastel" for qualitative
    width : int | None, optional
        Figure width in pixels. If None, uses 1000.
    height : int | None, optional
        Figure height in pixels. If None, uses 600 (overlay) or 400 per method.
    bins : int, optional
        Number of histogram bins. Default is 30.
    overlay : bool, optional
        If True and method is None, overlay all methods in single plot.
        If False and method is None, create subplot for each method.
        Default is False (subplots).

    Returns
    -------
    go.Figure
        Interactive Plotly histogram with:
        - Distribution of importance scores
        - Optional multiple methods overlaid or in subplots
        - Statistics annotations (mean, median, quartiles)
        - Hover showing bin ranges and counts

    Raises
    ------
    ValueError
        If results dict is missing required keys or has invalid structure.
        If specified method was not run or doesn't exist.
    TypeError
        If parameters have incorrect types.

    Examples
    --------
    >>> from ml4t.diagnostic.evaluation import analyze_ml_importance
    >>> from ml4t.diagnostic.visualization import plot_importance_distribution
    >>>
    >>> # Analyze importance
    >>> results = analyze_ml_importance(model, X, y)
    >>>
    >>> # Show all methods (subplots)
    >>> fig = plot_importance_distribution(results)
    >>> fig.show()
    >>>
    >>> # Overlay for comparison
    >>> fig = plot_importance_distribution(results, overlay=True)
    >>> fig.show()
    >>>
    >>> # Single method with custom bins
    >>> fig = plot_importance_distribution(
    ...     results,
    ...     method="pfi",
    ...     bins=50,
    ...     theme="dark"
    ... )

    Notes
    -----
    - Distributions reveal whether importance is concentrated or spread out
    - Overlay mode is best for comparing 2-3 methods; use subplots for more
    - Very skewed distributions may benefit from log scale (not implemented yet)
    - Consider binning strategy for features with very different importance ranges
    """
    # Validate inputs
    validate_plot_results(
        results,
        required_keys=["method_results", "methods_run"],
        function_name="plot_importance_distribution",
    )
    theme = validate_theme(theme)
    validate_positive_int(bins, "bins")

    # Note: color_scheme validation happens in get_color_scheme()

    methods_run = results["methods_run"]
    method_results = results["method_results"]

    # Determine which methods to plot
    if method is not None:
        if method not in methods_run:
            raise ValueError(
                f"Method '{method}' not found in results. Available methods: {methods_run}"
            )
        methods_to_plot = [method]
    else:
        methods_to_plot = methods_run

    # Get theme configuration
    theme_config = get_theme_config(theme)

    # Get colors (get full scheme and use first N colors)
    color_list = get_color_scheme(color_scheme or "set2")
    colors = (
        color_list[: len(methods_to_plot)]
        if len(methods_to_plot) <= len(color_list)
        else color_list
    )

    # Extract importance scores for each method
    method_scores = {}
    for method_name in methods_to_plot:
        result = method_results[method_name]
        scores = result["importances_mean"] if method_name == "pfi" else result["importances"]
        method_scores[method_name] = scores

    # Create figure
    if overlay or len(methods_to_plot) == 1:
        # Single plot with overlaid histograms
        fig = go.Figure()

        for i, (method_name, scores) in enumerate(method_scores.items()):
            fig.add_trace(
                go.Histogram(
                    x=scores,
                    name=method_name.upper(),
                    nbinsx=bins,
                    marker_color=colors[i],
                    opacity=0.7 if overlay else 1.0,
                    hovertemplate=(
                        f"<b>{method_name.upper()}</b><br>Importance: %{{x:.4f}}<br>Count: %{{y}}<extra></extra>"
                    ),
                )
            )

        fig.update_layout(
            title=title or "Feature Importance Distribution",
            xaxis_title="Importance Score",
            yaxis_title="Frequency",
            barmode="overlay" if overlay else "stack",
            **theme_config["layout"],
            width=width or 1000,
            height=height or 600,
        )

    else:
        # Subplots for each method
        n_methods = len(methods_to_plot)
        fig = make_subplots(
            rows=n_methods,
            cols=1,
            subplot_titles=[m.upper() for m in methods_to_plot],
            vertical_spacing=0.1,
        )

        for i, (method_name, scores) in enumerate(method_scores.items(), start=1):
            fig.add_trace(
                go.Histogram(
                    x=scores,
                    nbinsx=bins,
                    marker_color=colors[i - 1],
                    name=method_name.upper(),
                    showlegend=False,
                    hovertemplate=(
                        f"<b>{method_name.upper()}</b><br>Importance: %{{x:.4f}}<br>Count: %{{y}}<extra></extra>"
                    ),
                ),
                row=i,
                col=1,
            )

            # Update subplot axes
            fig.update_xaxes(title_text="Importance Score", row=i, col=1)
            fig.update_yaxes(title_text="Frequency", row=i, col=1)

        fig.update_layout(
            title=title or "Feature Importance Distribution by Method",
            **theme_config["layout"],
            width=width or 1000,
            height=height or (400 * n_methods),
        )

    # Apply responsive layout
    apply_responsive_layout(fig)

    return fig


def plot_importance_summary(
    results: dict[str, Any],
    *,
    title: str | None = None,
    top_n: int = 15,
    theme: str | None = None,
    width: int | None = None,
    height: int | None = None,
) -> go.Figure:
    """Create comprehensive multi-panel feature importance summary visualization.

    Combines multiple views into a single figure:
    - Top-left: Bar chart of consensus rankings
    - Top-right: Method agreement heatmap
    - Bottom: Distribution of importance scores

    This provides a complete overview of feature importance analysis in one plot,
    ideal for reports and presentations.

    Parameters
    ----------
    results : dict[str, Any]
        Results from analyze_ml_importance() containing all required data
    title : str | None, optional
        Overall figure title. If None, uses "Feature Importance Analysis - Summary"
    top_n : int, optional
        Number of top features to show in bar chart. Default is 15.
    theme : str | None, optional
        Theme name ("default", "dark", "print", "presentation").
        If None, uses current global theme.
    width : int | None, optional
        Figure width in pixels. If None, uses 1400.
    height : int | None, optional
        Figure height in pixels. If None, uses 1000.

    Returns
    -------
    go.Figure
        Multi-panel Plotly figure with comprehensive importance summary

    Raises
    ------
    ValueError
        If results dict is missing required keys or has invalid structure.
    TypeError
        If parameters have incorrect types.

    Examples
    --------
    >>> from ml4t.diagnostic.evaluation import analyze_ml_importance
    >>> from ml4t.diagnostic.visualization import plot_importance_summary
    >>>
    >>> # Analyze importance
    >>> results = analyze_ml_importance(model, X, y)
    >>>
    >>> # Create comprehensive summary
    >>> fig = plot_importance_summary(results)
    >>> fig.show()
    >>>
    >>> # Save for report
    >>> fig = plot_importance_summary(
    ...     results,
    ...     title="Model Feature Importance Analysis",
    ...     theme="print",
    ...     top_n=20
    ... )
    >>> fig.write_html("importance_summary.html")
    >>> fig.write_image("importance_summary.pdf")

    Notes
    -----
    - This is the recommended visualization for comprehensive reports
    - All panels use consistent theming and color schemes
    - Interactive hover works independently for each panel
    - May require large display or high resolution for optimal viewing
    - Consider using individual plot functions for more customization
    """
    # Validate inputs
    validate_plot_results(
        results,
        required_keys=["consensus_ranking", "method_results", "method_agreement", "methods_run"],
        function_name="plot_importance_summary",
    )
    theme = validate_theme(theme)
    validate_positive_int(top_n, "top_n")

    # Get theme configuration
    theme_config = get_theme_config(theme)

    # Create subplots: 2x2 layout
    # Row 1: Bar chart (left), Heatmap (right)
    # Row 2: Distribution (spans both columns)
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Consensus Rankings (Top Features)",
            "Method Agreement",
            "Importance Score Distributions",
            "",  # Empty subtitle for merged cell
        ),
        specs=[
            [{"type": "bar"}, {"type": "heatmap"}],
            [{"type": "histogram", "colspan": 2}, None],
        ],
        vertical_spacing=0.15,
        horizontal_spacing=0.12,
    )

    # === Panel 1: Bar chart ===
    all_features = results["consensus_ranking"]
    features = all_features[:top_n]
    method_results = results["method_results"]

    # Calculate average importance
    importance_scores = []
    for feat in features:
        scores = []
        for method_name, method_result in method_results.items():
            importances = (
                method_result["importances_mean"]
                if method_name == "pfi"
                else method_result["importances"]
            )
            method_features = method_result["feature_names"]
            if feat in method_features:
                idx = method_features.index(feat)
                scores.append(importances[idx])
        if scores:
            importance_scores.append(float(np.mean(scores)))
        else:
            importance_scores.append(0.0)

    colors_bar = get_colorscale("viridis")

    fig.add_trace(
        go.Bar(
            x=importance_scores,
            y=features,
            orientation="h",
            marker={
                "color": importance_scores,
                "colorscale": colors_bar,
                "showscale": False,
            },
            hovertemplate="<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>",
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    # === Panel 2: Heatmap ===
    methods = results["methods_run"]
    n_methods = len(methods)
    correlation_matrix = np.eye(n_methods)
    method_agreement = results["method_agreement"]

    for i, method1 in enumerate(methods):
        for j, method2 in enumerate(methods):
            if i < j:
                key1 = f"{method1}_vs_{method2}"
                key2 = f"{method2}_vs_{method1}"
                corr = method_agreement.get(key1, method_agreement.get(key2, 0.0))
                correlation_matrix[i, j] = corr
                correlation_matrix[j, i] = corr

    colors_heatmap = get_colorscale("rdbu")

    fig.add_trace(
        go.Heatmap(
            z=correlation_matrix,
            x=[m.upper() for m in methods],
            y=[m.upper() for m in methods],
            colorscale=colors_heatmap,
            zmid=0,
            zmin=-1,
            zmax=1,
            showscale=True,
            colorbar={
                "title": "Correlation",
                "x": 1.15,  # Position to right of subplot
                "len": 0.4,
            },
            text=np.round(correlation_matrix, 2),
            texttemplate="%{text}",
            textfont={"size": 10},
            hovertemplate=("<b>%{x}</b> vs <b>%{y}</b><br>Correlation: %{z:.3f}<extra></extra>"),
        ),
        row=1,
        col=2,
    )

    # === Panel 3: Distribution (overlay) ===
    color_list_dist = get_color_scheme("set2")
    colors_dist = (
        color_list_dist[: len(methods)] if len(methods) <= len(color_list_dist) else color_list_dist
    )

    for i, method_name in enumerate(methods):
        result = method_results[method_name]
        scores = result["importances_mean"] if method_name == "pfi" else result["importances"]

        fig.add_trace(
            go.Histogram(
                x=scores,
                name=method_name.upper(),
                nbinsx=30,
                marker_color=colors_dist[i],
                opacity=0.7,
                hovertemplate=(
                    f"<b>{method_name.upper()}</b><br>Importance: %{{x:.4f}}<br>Count: %{{y}}<extra></extra>"
                ),
            ),
            row=2,
            col=1,
        )

    # Update axes
    fig.update_xaxes(title_text="Importance Score", row=1, col=1)
    fig.update_yaxes(title_text="Features", row=1, col=1)
    fig.update_xaxes(title_text="", row=1, col=2)
    fig.update_yaxes(title_text="", autorange="reversed", row=1, col=2)
    fig.update_xaxes(title_text="Importance Score", row=2, col=1)
    fig.update_yaxes(title_text="Frequency", row=2, col=1)

    # Update layout
    fig.update_layout(
        title={
            "text": title or "Feature Importance Analysis - Summary",
            "x": 0.5,
            "xanchor": "center",
        },
        barmode="overlay",
        **theme_config["layout"],
        width=width or 1400,
        height=height or 1000,
        showlegend=True,
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1,
        },
    )

    # Apply responsive layout
    apply_responsive_layout(fig)

    return fig
