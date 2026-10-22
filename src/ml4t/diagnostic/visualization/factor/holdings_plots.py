"""Supplied-holdings exposures and additive contributions, separate from estimated betas."""

from __future__ import annotations

from typing import TYPE_CHECKING

import plotly.graph_objects as go

from ml4t.diagnostic.visualization._colors import FACTOR_COLORS
from ml4t.diagnostic.visualization.core import get_theme_config

if TYPE_CHECKING:
    from ml4t.diagnostic.evaluation.factor.holdings_attribution import HoldingsAttributionResult


def plot_holdings_exposures(
    result: HoldingsAttributionResult,
    *,
    theme: str | None = None,
    height: int = 380,
) -> go.Figure:
    """Plot supplied NAV-weighted loadings, not estimated regression coefficients."""
    figure = go.Figure()
    dates = result.exposures["timestamp"].to_pandas()
    colors = get_theme_config(theme)["colorway"]
    for index, name in enumerate(result.factor_names):
        figure.add_trace(
            go.Scatter(
                x=dates,
                y=result.exposures[name].to_numpy(),
                name=name,
                mode="lines+markers" if len(dates) < 5 else "lines",
                line={"color": FACTOR_COLORS.get(name, colors[index % len(colors)]), "width": 2},
            )
        )
    figure.update_layout(get_theme_config(theme)["layout"])
    figure.update_layout(
        title="Supplied holdings exposures",
        height=height,
        font={"size": 14},
        xaxis_title="Return interval end",
        yaxis_title="NAV-weighted loading",
        showlegend=True,
        legend={"orientation": "h", "y": 1.02, "yanchor": "bottom", "x": 0},
        xaxis={"nticks": 4, "tickformat": "%b %d<br>%H:%M", "tickangle": 0, "title_font_size": 14},
        yaxis_title_font_size=14,
        margin={"l": 70, "r": 24, "t": 70, "b": 80},
    )
    return figure


def plot_holdings_attribution(
    result: HoldingsAttributionResult,
    *,
    theme: str | None = None,
    height: int = 500,
) -> go.Figure:
    """Plot already calculated cumulative contributions without compounding them again."""
    figure = go.Figure()
    dates = result.cumulative["timestamp"].to_pandas()
    columns = [
        ("common_returns", "Common"),
        ("specific_returns", "Specific"),
        *([("cash_returns", "Explicit cash")] if result.cash_attributed else []),
        ("total_returns", "Total"),
        *[(name, f"Factor: {name}") for name in result.factor_names],
    ]
    colors = get_theme_config(theme)["colorway"]
    for index, (key, label) in enumerate(columns):
        figure.add_trace(
            go.Scatter(
                x=dates,
                y=result.cumulative[key].to_numpy(),
                name=label,
                mode="lines+markers" if len(dates) < 5 else "lines",
                visible="legendonly" if key in result.factor_names else True,
                line={
                    "color": FACTOR_COLORS.get(label, colors[index % len(colors)]),
                    "width": 2,
                    "dash": "dash" if key == "total_returns" else "solid",
                },
            )
        )
    label = "Wealth-linked" if result.cumulative_method == "wealth_linked" else "Arithmetic"
    figure.update_layout(get_theme_config(theme)["layout"])
    figure.update_layout(
        title=f"{label} supplied holdings attribution",
        height=height,
        font={"size": 14},
        xaxis_title="Return interval end",
        yaxis_title=f"{label} return contribution",
        yaxis_tickformat=".1%",
        showlegend=True,
        legend={"orientation": "h", "y": 1.02, "yanchor": "bottom", "x": 0},
        xaxis={"nticks": 4, "tickformat": "%b %d<br>%H:%M", "tickangle": 0, "title_font_size": 14},
        yaxis_title_font_size=14,
        margin={"l": 70, "r": 24, "t": 140, "b": 80},
    )
    return figure
