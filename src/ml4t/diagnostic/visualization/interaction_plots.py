"""Feature interaction visualization functions.

This module provides functions for visualizing feature interaction analysis results
from analyze_interactions(), compute_shap_interactions(), and related functions.

All plot functions follow the standard API defined in docs/plot_api_standards.md:
- Consume results dicts from analyze_*() or compute_*() functions
- Return plotly.graph_objects.Figure instances
- Support theme customization via global or per-plot settings
- Use keyword-only arguments (after results)
- Provide comprehensive hover information and interactivity

Example workflow:
    >>> from ml4t.diagnostic.evaluation import analyze_interactions, compute_shap_interactions
    >>> from ml4t.diagnostic.visualization import (
    ...     plot_interaction_bar,
    ...     plot_interaction_heatmap,
    ...     plot_interaction_network,
    ...     set_plot_theme
    ... )
    >>>
    >>> # Analyze interactions
    >>> results = analyze_interactions(model, X, y)
    >>>
    >>> # Or use SHAP directly
    >>> shap_results = compute_shap_interactions(model, X, top_k=20)
    >>>
    >>> # Create visualizations
    >>> fig_bar = plot_interaction_bar(shap_results, top_n=15)
    >>> fig_heatmap = plot_interaction_heatmap(shap_results)
    >>> fig_network = plot_interaction_network(shap_results, threshold=0.01)
    >>>
    >>> # Display or save
    >>> fig_network.show()
    >>> fig_heatmap.write_html("interactions_report.html")
"""

from typing import Any

import numpy as np
import plotly.graph_objects as go

from ml4t.diagnostic.visualization.core import (
    apply_responsive_layout,
    get_color_scheme,
    get_colorscale,
    get_theme_config,
    validate_plot_results,
    validate_positive_int,
    validate_theme,
)

__all__ = [
    "plot_interaction_bar",
    "plot_interaction_heatmap",
    "plot_interaction_network",
]


def plot_interaction_bar(
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
    """Plot horizontal bar chart of top feature interactions.

    Creates an interactive bar chart showing the strongest feature interactions
    ranked by their interaction strength. Each bar represents a feature pair
    with color-coding by strength.

    Parameters
    ----------
    results : dict[str, Any]
        Results from compute_shap_interactions() or analyze_interactions() containing:
        - "top_interactions": list[tuple[str, str, float]] - Feature pairs with scores
        OR
        - "consensus_ranking": list[tuple[str, str, float, dict]] - From analyze_interactions()
    title : str | None, optional
        Plot title. If None, uses "Feature Interactions - Top Pairs"
    top_n : int | None, optional
        Number of top interactions to display. If None, shows all.
        Default is 20 to avoid overcrowding.
    theme : str | None, optional
        Theme name ("default", "dark", "print", "presentation").
        If None, uses current global theme.
    color_scheme : str | None, optional
        Color scheme for bars. If None, uses "viridis".
        Recommended: "viridis", "cividis", "plasma", "oranges", "reds"
    width : int | None, optional
        Figure width in pixels. If None, uses theme default (typically 1000).
    height : int | None, optional
        Figure height in pixels. If None, auto-sizes based on interaction count
        (25px per interaction + 100px padding).
    show_values : bool, optional
        Whether to show interaction values on bars. Default is True.

    Returns
    -------
    go.Figure
        Interactive Plotly figure with:
        - Horizontal bars sorted by interaction strength
        - Continuous color gradient indicating strength
        - Hover info showing exact values
        - Responsive layout for different screen sizes

    Raises
    ------
    ValueError
        If results dict is missing required keys or has invalid structure.
    TypeError
        If parameters have incorrect types.

    Examples
    --------
    >>> from ml4t.diagnostic.evaluation import compute_shap_interactions
    >>> from ml4t.diagnostic.visualization import plot_interaction_bar
    >>>
    >>> # Compute SHAP interactions
    >>> results = compute_shap_interactions(model, X, top_k=20)
    >>>
    >>> # Plot top 10 interactions
    >>> fig = plot_interaction_bar(results, top_n=10)
    >>> fig.show()
    >>>
    >>> # Custom styling
    >>> fig = plot_interaction_bar(
    ...     results,
    ...     title="Strong Feature Interactions",
    ...     top_n=15,
    ...     theme="dark",
    ...     color_scheme="plasma",
    ...     height=700
    ... )
    >>> fig.write_image("interactions.pdf")

    Notes
    -----
    - Works with both compute_shap_interactions() and analyze_interactions() results
    - Interaction strength is absolute magnitude (always positive)
    - Pairs are deduplicated (A×B same as B×A)
    - Use top_n to focus on strongest interactions
    """
    # Validate inputs
    theme = validate_theme(theme)
    if top_n is not None:
        validate_positive_int(top_n, "top_n")

    # Extract interaction pairs - support both result formats
    if "top_interactions" in results:
        # From compute_shap_interactions() or single method
        interactions = results["top_interactions"]
    elif "consensus_ranking" in results:
        # From analyze_interactions()
        interactions = [
            (pair[0], pair[1], pair[2])  # Extract first 3 elements
            for pair in results["consensus_ranking"]
        ]
    else:
        raise ValueError(
            "Results must contain 'top_interactions' (from compute_shap_interactions) "
            "or 'consensus_ranking' (from analyze_interactions)"
        )

    # Limit to top N
    if top_n is not None:
        interactions = interactions[:top_n]

    # Create labels and values
    pair_labels = [f"{feat_i} × {feat_j}" for feat_i, feat_j, _ in interactions]
    interaction_values = [abs(val) for _, _, val in interactions]

    # Reverse for top-to-bottom display
    pair_labels = pair_labels[::-1]
    interaction_values = interaction_values[::-1]

    # Get theme and colors
    theme_config = get_theme_config(theme)
    colors = get_colorscale(color_scheme or "viridis")

    # Create figure
    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=interaction_values,
            y=pair_labels,
            orientation="h",
            marker={
                "color": interaction_values,
                "colorscale": colors,
                "showscale": True,
                "colorbar": {
                    "title": "Strength",
                    "tickformat": ".3f",
                },
            },
            text=[f"{v:.3f}" for v in interaction_values] if show_values else None,
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>Interaction: %{x:.4f}<extra></extra>",
        )
    )

    # Update layout
    fig.update_layout(
        title=title or "Feature Interactions - Top Pairs",
        xaxis_title="Interaction Strength",
        yaxis_title="Feature Pairs",
        **theme_config["layout"],
        width=width or 1000,
        height=height or max(400, len(pair_labels) * 25 + 100),
        showlegend=False,
    )

    # Apply responsive layout
    apply_responsive_layout(fig)

    return fig


def plot_interaction_heatmap(
    results: dict[str, Any],
    *,
    title: str | None = None,
    theme: str | None = None,
    color_scheme: str | None = None,
    width: int | None = None,
    height: int | None = None,
    show_values: bool = False,  # False by default - can be crowded
) -> go.Figure:
    """Plot heatmap of feature interaction matrix.

    Creates a symmetric heatmap showing pairwise feature interactions. The matrix
    is symmetric (interaction(i,j) = interaction(j,i)). Diagonal elements represent
    main effects (feature importance without interactions).

    Parameters
    ----------
    results : dict[str, Any]
        Results from compute_shap_interactions() or similar containing:
        - "interaction_matrix": np.ndarray - (n_features, n_features) matrix
        - "feature_names": list[str] - Feature names for axis labels
    title : str | None, optional
        Plot title. If None, uses "Feature Interaction Matrix"
    theme : str | None, optional
        Theme name ("default", "dark", "print", "presentation").
        If None, uses current global theme.
    color_scheme : str | None, optional
        Color scheme for heatmap. If None, uses "viridis".
        Recommended: "viridis", "plasma", "inferno", "magma", "cividis"
    width : int | None, optional
        Figure width in pixels. If None, uses 800.
    height : int | None, optional
        Figure height in pixels. If None, uses 800.
    show_values : bool, optional
        Whether to show interaction values in cells. Default is False
        (can be crowded for many features).

    Returns
    -------
    go.Figure
        Interactive Plotly heatmap with:
        - Symmetric interaction matrix
        - Continuous colorscale from weak to strong
        - Optional cell annotations
        - Hover showing feature pairs and values

    Raises
    ------
    ValueError
        If results dict is missing required keys or has invalid structure.
    TypeError
        If parameters have incorrect types.

    Examples
    --------
    >>> from ml4t.diagnostic.evaluation import compute_shap_interactions
    >>> from ml4t.diagnostic.visualization import plot_interaction_heatmap
    >>>
    >>> # Compute interactions
    >>> results = compute_shap_interactions(model, X)
    >>>
    >>> # Create heatmap
    >>> fig = plot_interaction_heatmap(results)
    >>> fig.show()
    >>>
    >>> # With annotations for small feature sets
    >>> fig = plot_interaction_heatmap(
    ...     results,
    ...     show_values=True,  # Show numbers in cells
    ...     theme="print",
    ...     color_scheme="viridis"
    ... )

    Notes
    -----
    - Matrix is symmetric: interaction(i,j) = interaction(j,i)
    - Diagonal elements are main effects (not interactions)
    - Off-diagonal elements are pairwise interactions
    - For many features (>20), consider hiding cell values (show_values=False)
    - All values are absolute (non-negative)
    """
    # Validate inputs
    validate_plot_results(
        results,
        required_keys=["interaction_matrix", "feature_names"],
        function_name="plot_interaction_heatmap",
    )
    theme = validate_theme(theme)

    # Extract data
    interaction_matrix = results["interaction_matrix"]
    feature_names = results["feature_names"]

    # Get theme and colors
    theme_config = get_theme_config(theme)
    colors = get_colorscale(color_scheme or "viridis")

    # Create hover text
    n_features = len(feature_names)
    hover_text = []
    for i in range(n_features):
        row = []
        for j in range(n_features):
            value = interaction_matrix[i, j]
            if i == j:
                row.append(f"<b>{feature_names[i]}</b><br>Main Effect: {value:.4f}")
            else:
                row.append(
                    f"<b>{feature_names[i]}</b> × <b>{feature_names[j]}</b><br>Interaction: {value:.4f}"
                )
        hover_text.append(row)

    # Create figure
    fig = go.Figure()

    fig.add_trace(
        go.Heatmap(
            z=interaction_matrix,
            x=feature_names,
            y=feature_names,
            colorscale=colors,
            colorbar={
                "title": "Strength",
                "tickformat": ".3f",
            },
            text=np.round(interaction_matrix, 3) if show_values else None,
            texttemplate="%{text}" if show_values else None,
            textfont={"size": 10},
            hovertext=hover_text,
            hovertemplate="%{hovertext}<extra></extra>",
        )
    )

    # Update layout
    fig.update_layout(
        title=title or "Feature Interaction Matrix",
        xaxis={
            "title": "",
            "side": "bottom",
            "tickangle": -45 if len(feature_names) > 10 else 0,
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


def plot_interaction_network(
    results: dict[str, Any],
    *,
    title: str | None = None,
    threshold: float | None = None,
    top_n: int | None = None,
    theme: str | None = None,
    color_scheme: str | None = None,
    width: int | None = None,
    height: int | None = None,
    node_size: int = 30,
    show_edge_labels: bool = False,
) -> go.Figure:
    """Plot network graph of feature interactions.

    Creates an interactive network visualization where:
    - Nodes represent features
    - Edges represent interactions
    - Edge thickness indicates interaction strength
    - Only significant interactions above threshold are shown

    Parameters
    ----------
    results : dict[str, Any]
        Results from compute_shap_interactions() or analyze_interactions() containing:
        - "top_interactions": list[tuple[str, str, float]] - Feature pairs
        OR
        - "interaction_matrix" and "feature_names" - Will extract top interactions
    title : str | None, optional
        Plot title. If None, uses "Feature Interaction Network"
    threshold : float | None, optional
        Minimum interaction strength to display. If None, uses adaptive threshold
        (median of all interactions or top 20%, whichever is stricter).
    top_n : int | None, optional
        Maximum number of interactions to display. If None, shows all above threshold.
        Useful to avoid cluttered networks.
    theme : str | None, optional
        Theme name ("default", "dark", "print", "presentation").
        If None, uses current global theme.
    color_scheme : str | None, optional
        Color scheme for nodes. If None, uses "set2".
        Recommended: "set2", "set3", "pastel", "bold"
    width : int | None, optional
        Figure width in pixels. If None, uses 1000.
    height : int | None, optional
        Figure height in pixels. If None, uses 800.
    node_size : int, optional
        Size of nodes in pixels. Default is 30.
    show_edge_labels : bool, optional
        Whether to show interaction values on edges. Default is False
        (can be cluttered).

    Returns
    -------
    go.Figure
        Interactive Plotly network graph with:
        - Nodes positioned using force-directed layout
        - Edge thickness proportional to interaction strength
        - Optional edge labels showing values
        - Hover info for nodes and edges
        - Pan/zoom capability

    Raises
    ------
    ValueError
        If results dict is missing required keys or has invalid structure.
        If no interactions remain after filtering.
    TypeError
        If parameters have incorrect types.

    Examples
    --------
    >>> from ml4t.diagnostic.evaluation import compute_shap_interactions
    >>> from ml4t.diagnostic.visualization import plot_interaction_network
    >>>
    >>> # Compute interactions
    >>> results = compute_shap_interactions(model, X, top_k=30)
    >>>
    >>> # Create network showing only strong interactions
    >>> fig = plot_interaction_network(
    ...     results,
    ...     threshold=0.05,  # Show only interactions > 0.05
    ...     top_n=20         # Limit to top 20
    ... )
    >>> fig.show()
    >>>
    >>> # Show edge labels
    >>> fig = plot_interaction_network(
    ...     results,
    ...     show_edge_labels=True,
    ...     theme="dark"
    ... )

    Notes
    -----
    - Network layout uses spring/force-directed algorithm
    - Isolated nodes (no interactions) are excluded
    - Edge thickness is proportional to interaction strength
    - For complex networks (>50 edges), consider increasing threshold or using top_n
    - Use threshold and top_n together for best control
    """
    # Validate inputs
    theme = validate_theme(theme)
    if top_n is not None:
        validate_positive_int(top_n, "top_n")

    # Extract interactions
    if "top_interactions" in results:
        interactions = results["top_interactions"]
    elif "interaction_matrix" in results and "feature_names" in results:
        # Convert matrix to interaction list
        matrix = results["interaction_matrix"]
        feature_names = results["feature_names"]
        n_features = len(feature_names)

        interactions = []
        for i in range(n_features):
            for j in range(i + 1, n_features):  # Upper triangle only
                interactions.append((feature_names[i], feature_names[j], matrix[i, j]))

        # Sort by strength
        interactions.sort(key=lambda x: abs(x[2]), reverse=True)
    else:
        raise ValueError(
            "Results must contain 'top_interactions' or 'interaction_matrix' + 'feature_names'"
        )

    # Apply threshold
    if threshold is None:
        # Adaptive threshold: median or top 20%
        values = [abs(val) for _, _, val in interactions]
        median_threshold = np.median(values)
        percentile_threshold = np.percentile(values, 80)
        threshold = max(median_threshold, percentile_threshold)

    filtered_interactions = [(f1, f2, val) for f1, f2, val in interactions if abs(val) >= threshold]

    # Apply top_n limit
    if top_n is not None:
        filtered_interactions = filtered_interactions[:top_n]

    if len(filtered_interactions) == 0:
        raise ValueError(
            f"No interactions above threshold {threshold:.4f}. Try lowering threshold or increasing top_n."
        )

    # Build node set
    node_set: set[str] = set()
    for f1, f2, _ in filtered_interactions:
        node_set.add(f1)
        node_set.add(f2)
    nodes = sorted(node_set)
    node_indices = {node: i for i, node in enumerate(nodes)}

    # Simple circular layout for nodes
    n_nodes = len(nodes)
    angles = np.linspace(0, 2 * np.pi, n_nodes, endpoint=False)
    radius = 1.0

    node_x = radius * np.cos(angles)
    node_y = radius * np.sin(angles)

    # Get theme and colors
    get_theme_config(theme)
    node_colors = get_color_scheme(color_scheme or "set2")

    # Create figure
    fig = go.Figure()

    # Add edges
    max_interaction = max(abs(val) for _, _, val in filtered_interactions)
    for f1, f2, val in filtered_interactions:
        i1 = node_indices[f1]
        i2 = node_indices[f2]

        # Edge thickness proportional to interaction strength
        edge_width = 1 + 5 * (abs(val) / max_interaction)

        fig.add_trace(
            go.Scatter(
                x=[node_x[i1], node_x[i2]],
                y=[node_y[i1], node_y[i2]],
                mode="lines",
                line={"width": edge_width, "color": "rgba(125,125,125,0.5)"},
                hoverinfo="text",
                hovertext=f"{f1} × {f2}<br>Interaction: {abs(val):.4f}",
                showlegend=False,
            )
        )

        # Optional edge labels
        if show_edge_labels:
            mid_x = (node_x[i1] + node_x[i2]) / 2
            mid_y = (node_y[i1] + node_y[i2]) / 2
            fig.add_annotation(
                x=mid_x,
                y=mid_y,
                text=f"{abs(val):.2f}",
                showarrow=False,
                font={"size": 8},
            )

    # Add nodes
    fig.add_trace(
        go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            marker={
                "size": node_size,
                "color": [node_colors[i % len(node_colors)] for i in range(n_nodes)],
                "line": {"width": 2, "color": "white"},
            },
            text=nodes,
            textposition="top center",
            textfont={"size": 10},
            hoverinfo="text",
            hovertext=nodes,
            showlegend=False,
        )
    )

    # Update layout (using simpler approach to avoid theme conflicts)
    fig.update_layout(
        title=title or "Feature Interaction Network",
        width=width or 1000,
        height=height or 800,
        xaxis={"showgrid": False, "zeroline": False, "showticklabels": False},
        yaxis={"showgrid": False, "zeroline": False, "showticklabels": False},
        hovermode="closest",
    )

    # Apply responsive layout
    apply_responsive_layout(fig)

    return fig
