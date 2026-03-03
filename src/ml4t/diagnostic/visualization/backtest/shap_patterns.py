"""SHAP error pattern visualizations for backtest tearsheets.

Provides two visualizations for understanding why a model's worst trades
lose money, based on trade-level SHAP analysis:

- plot_shap_error_patterns: Grouped bars showing mean |SHAP| per feature
  per error cluster, with a summary table.
- plot_shap_worst_trades: Stacked horizontal bars showing per-trade SHAP
  feature contributions for the worst trades.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ml4t.diagnostic.visualization._colors import COLORS as _ML4T_COLORS
from ml4t.diagnostic.visualization.core import get_quantile_colors, get_theme_config

if TYPE_CHECKING:
    from ml4t.diagnostic.evaluation.trade_shap.models import TradeShapResult


def plot_shap_error_patterns(
    shap_result: TradeShapResult,
    *,
    top_n_features: int = 8,
    title: str = "SHAP Error Patterns",
    theme: str | None = None,
    height: int = 600,
    width: int | None = None,
) -> go.Figure:
    """Visualize error patterns identified by SHAP-based clustering.

    Top subplot: grouped horizontal bars showing mean |SHAP| per feature
    per cluster, colored by cluster. Bottom: summary table with cluster
    metadata (n_trades, hypothesis, confidence, separation score).

    Parameters
    ----------
    shap_result : TradeShapResult
        Result from TradeShapPipeline.analyze_worst_trades().
    top_n_features : int
        Number of top features to show per cluster (default: 8).
    title : str
        Chart title.
    theme : str, optional
        Theme name.
    height : int
        Figure height in pixels.
    width : int, optional
        Figure width in pixels.

    Returns
    -------
    go.Figure
        Plotly figure with error pattern visualization.
    """
    theme_config = get_theme_config(theme)
    patterns = shap_result.error_patterns

    if not patterns:
        return _empty_placeholder(
            "No error patterns identified (insufficient trades or clustering failed)",
            theme_config,
            height,
            width,
        )

    n_clusters = len(patterns)
    cluster_colors = get_quantile_colors(max(n_clusters, 2), theme_config)[:n_clusters]

    # Collect the union of top features across all patterns
    feature_set: dict[str, float] = {}
    for pattern in patterns:
        for feat_name, mean_shap, *_ in pattern.top_features[:top_n_features]:
            existing = feature_set.get(feat_name, 0.0)
            feature_set[feat_name] = max(existing, abs(mean_shap))

    # Sort by max absolute SHAP (most important first)
    sorted_features = sorted(feature_set.keys(), key=lambda f: feature_set[f], reverse=True)
    top_features = sorted_features[:top_n_features]

    # Build figure: bar chart (top) + table (bottom)
    fig = make_subplots(
        rows=2,
        cols=1,
        row_heights=[0.65, 0.35],
        specs=[[{"type": "xy"}], [{"type": "table"}]],
        vertical_spacing=0.12,
    )

    # --- Top panel: grouped horizontal bars ---
    for i, pattern in enumerate(patterns):
        # Build a lookup from feature name to mean SHAP for this cluster
        feat_shap = {f[0]: f[1] for f in pattern.top_features}

        y_vals = list(reversed(top_features))  # reversed so top feature is at top
        x_vals = [abs(feat_shap.get(f, 0.0)) for f in reversed(top_features)]

        # Bold label for significant features
        significant_set = {f[0] for f in pattern.top_features if f[4]}  # is_significant
        y_labels = [f"<b>{f}</b>" if f in significant_set else f for f in y_vals]

        fig.add_trace(
            go.Bar(
                x=x_vals,
                y=y_labels,
                orientation="h",
                name=f"Cluster {pattern.cluster_id} ({pattern.n_trades} trades)",
                marker_color=cluster_colors[i],
                opacity=0.85,
                hovertemplate=(
                    "%{y}<br>"
                    f"Cluster {pattern.cluster_id}<br>"
                    "Mean |SHAP|: %{x:.4f}<extra></extra>"
                ),
            ),
            row=1,
            col=1,
        )

    # --- Bottom panel: summary table ---
    header_vals = ["Cluster", "Trades", "Hypothesis", "Confidence", "Separation"]
    cluster_ids = [str(p.cluster_id) for p in patterns]
    n_trades = [str(p.n_trades) for p in patterns]
    hypotheses = [
        (p.hypothesis[:80] + "...")
        if p.hypothesis and len(p.hypothesis) > 80
        else (p.hypothesis or "N/A")
        for p in patterns
    ]
    confidences = [f"{p.confidence:.0%}" if p.confidence is not None else "N/A" for p in patterns]
    separations = [f"{p.separation_score:.2f}" for p in patterns]

    table_header_fill = _ML4T_COLORS["slate"]
    table_cell_fill = theme_config["layout"].get("paper_bgcolor", "#FFFFFF")
    table_font_color = theme_config["layout"].get("font", {}).get("color", "#333333")

    fig.add_trace(
        go.Table(
            header={
                "values": [f"<b>{h}</b>" for h in header_vals],
                "fill_color": table_header_fill,
                "font": {"color": "#FFFFFF", "size": 12},
                "align": "left",
                "height": 30,
            },
            cells={
                "values": [cluster_ids, n_trades, hypotheses, confidences, separations],
                "fill_color": table_cell_fill,
                "font": {"color": table_font_color, "size": 11},
                "align": ["center", "center", "left", "center", "center"],
                "height": 25,
            },
        ),
        row=2,
        col=1,
    )

    # Layout
    layout_updates: dict = {
        "title": {"text": title, "font": {"size": 18}},
        "height": height,
        "barmode": "group",
        "xaxis": {"title": "Mean |SHAP Value|"},
        "legend": {"yanchor": "top", "y": 0.99, "xanchor": "right", "x": 0.99},
        "margin": {"l": 140, "r": 20, "t": 80, "b": 20},
    }
    if width:
        layout_updates["width"] = width

    for key, value in theme_config["layout"].items():
        if key not in layout_updates:
            layout_updates[key] = value

    fig.update_layout(**layout_updates)

    return fig


def plot_shap_worst_trades(
    shap_result: TradeShapResult,
    *,
    n_trades: int = 10,
    max_features: int = 8,
    title: str = "SHAP Contributions: Worst Trades",
    theme: str | None = None,
    height: int = 500,
    width: int | None = None,
) -> go.Figure:
    """Stacked horizontal bar chart of SHAP contributions per worst trade.

    Each row is one trade; segments show individual feature SHAP contributions.
    Positive SHAP in green, negative in red.

    Parameters
    ----------
    shap_result : TradeShapResult
        Result from TradeShapPipeline.analyze_worst_trades().
    n_trades : int
        Number of worst trades to display (default: 10).
    max_features : int
        Maximum features to show per trade (default: 8). Remaining are
        aggregated into "Other".
    title : str
        Chart title.
    theme : str, optional
        Theme name.
    height : int
        Figure height in pixels.
    width : int, optional
        Figure width in pixels.

    Returns
    -------
    go.Figure
        Plotly figure with stacked bar chart.
    """
    theme_config = get_theme_config(theme)

    explanations = shap_result.explanations[:n_trades]
    if not explanations:
        return _empty_placeholder(
            "No trade explanations available",
            theme_config,
            height,
            width,
        )

    # Collect all features that appear in top features across trades
    feature_union: set[str] = set()
    for exp in explanations:
        for feat_name, _ in exp.top_features[:max_features]:
            feature_union.add(feat_name)

    # Sort features by total absolute SHAP across shown trades
    feature_importance: dict[str, float] = {}
    for feat_name in feature_union:
        total = 0.0
        for exp in explanations:
            feat_lookup = dict(exp.top_features)
            total += abs(feat_lookup.get(feat_name, 0.0))
        feature_importance[feat_name] = total

    sorted_features = sorted(
        feature_importance.keys(), key=lambda f: feature_importance[f], reverse=True
    )
    display_features = sorted_features[:max_features]

    # Trade labels (y-axis)
    trade_labels = [
        f"{exp.trade_id[:25]}" if len(exp.trade_id) > 25 else exp.trade_id
        for exp in reversed(explanations)
    ]

    fig = go.Figure()

    for feat_name in display_features:
        values = []
        for exp in reversed(explanations):
            feat_lookup = dict(exp.top_features)
            values.append(feat_lookup.get(feat_name, 0.0))

        # Color by sign: use positive/negative colors
        colors = [_ML4T_COLORS["positive"] if v >= 0 else _ML4T_COLORS["negative"] for v in values]

        fig.add_trace(
            go.Bar(
                x=values,
                y=trade_labels,
                orientation="h",
                name=feat_name,
                marker_color=colors,
                hovertemplate=f"{feat_name}<br>SHAP: %{{x:.4f}}<extra></extra>",
            )
        )

    # Layout
    layout_updates: dict = {
        "title": {"text": title, "font": {"size": 18}},
        "height": max(height, len(explanations) * 45 + 120),
        "barmode": "relative",
        "xaxis": {"title": "SHAP Value", "zeroline": True},
        "legend": {"yanchor": "top", "y": 0.99, "xanchor": "right", "x": 0.99},
        "margin": {"l": 180, "r": 20, "t": 80, "b": 60},
    }
    if width:
        layout_updates["width"] = width

    for key, value in theme_config["layout"].items():
        if key not in layout_updates:
            layout_updates[key] = value

    fig.update_layout(**layout_updates)

    return fig


def _empty_placeholder(
    message: str,
    theme_config: dict,
    height: int,
    width: int | None,
) -> go.Figure:
    """Create a placeholder figure when no data is available."""
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font={"size": 14, "color": _ML4T_COLORS["neutral"]},
    )
    layout: dict = {
        "height": height,
        "xaxis": {"visible": False},
        "yaxis": {"visible": False},
    }
    if width:
        layout["width"] = width
    for key, value in theme_config["layout"].items():
        if key not in layout:
            layout[key] = value
    fig.update_layout(**layout)
    return fig
