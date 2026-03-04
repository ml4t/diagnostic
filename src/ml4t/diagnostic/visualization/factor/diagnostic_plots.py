"""Model diagnostic visualizations: residuals, correlation, VIF."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats as sp_stats

from ml4t.diagnostic.visualization.core import get_theme_config

if TYPE_CHECKING:
    from ml4t.diagnostic.evaluation.factor.data import FactorData
    from ml4t.diagnostic.evaluation.factor.results import (
        FactorModelResult,
        RollingExposureResult,
    )


def plot_residual_diagnostics(
    result: FactorModelResult,
    *,
    theme: str | None = None,
    height: int = 600,
    width: int | None = None,
) -> go.Figure:
    """2x2 subplot: residual time series, QQ plot, ACF, histogram.

    Parameters
    ----------
    result : FactorModelResult
        Static model result with residuals.
    theme : str | None
        Plot theme name.
    height : int
        Figure height.
    width : int | None
        Figure width.

    Returns
    -------
    go.Figure
        4-panel diagnostic figure.
    """
    theme_config = get_theme_config(theme)
    colorway = theme_config["colorway"]

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=["Residuals", "Q-Q Plot", "ACF", "Histogram"],
        vertical_spacing=0.12,
        horizontal_spacing=0.1,
    )

    residuals = result.residuals
    T = len(residuals)

    # 1. Residual time series
    fig.add_trace(
        go.Scatter(
            x=list(range(T)),
            y=residuals,
            mode="lines",
            line={"color": colorway[0], "width": 0.8},
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=1)

    # 2. QQ plot
    sorted_resid = np.sort(residuals)
    theoretical = sp_stats.norm.ppf(np.linspace(0.01, 0.99, T))
    # Scale theoretical to match residual distribution
    theoretical = theoretical * np.std(residuals) + np.mean(residuals)

    fig.add_trace(
        go.Scatter(
            x=theoretical,
            y=sorted_resid,
            mode="markers",
            marker={"color": colorway[0], "size": 3},
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    # 45-degree reference line
    min_val = min(theoretical.min(), sorted_resid.min())
    max_val = max(theoretical.max(), sorted_resid.max())
    fig.add_trace(
        go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode="lines",
            line={"color": "gray", "dash": "dash"},
            showlegend=False,
        ),
        row=1,
        col=2,
    )

    # 3. ACF
    max_lag = min(20, T // 4)
    acf_values = _compute_acf(residuals, max_lag)
    ci_bound = 1.96 / np.sqrt(T)

    fig.add_trace(
        go.Bar(
            x=list(range(1, max_lag + 1)),
            y=acf_values[1:],
            marker_color=colorway[0],
            showlegend=False,
        ),
        row=2,
        col=1,
    )
    fig.add_hline(y=ci_bound, line_dash="dot", line_color="gray", row=2, col=1)
    fig.add_hline(y=-ci_bound, line_dash="dot", line_color="gray", row=2, col=1)
    fig.add_hline(y=0, line_color="gray", line_width=0.5, row=2, col=1)

    # 4. Histogram
    fig.add_trace(
        go.Histogram(
            x=residuals,
            nbinsx=50,
            marker_color=colorway[0],
            opacity=0.7,
            showlegend=False,
        ),
        row=2,
        col=2,
    )

    fig.update_layout(theme_config["layout"])
    fig.update_layout(
        title="Residual Diagnostics",
        height=height,
        width=width or theme_config["defaults"]["width"],
        showlegend=False,
    )

    return fig


def plot_factor_correlation_heatmap(
    factor_data: FactorData,
    *,
    theme: str | None = None,
    height: int = 500,
    width: int | None = None,
) -> go.Figure:
    """Heatmap of factor correlations (diverging RdBu colorscale).

    Parameters
    ----------
    factor_data : FactorData
        Factor data container.
    theme : str | None
        Plot theme name.
    height : int
        Figure height.
    width : int | None
        Figure width.

    Returns
    -------
    go.Figure
        Plotly heatmap.
    """
    theme_config = get_theme_config(theme)

    X = factor_data.get_factor_array()
    factor_names = factor_data.factor_names
    corr = np.corrcoef(X, rowvar=False)

    fig = go.Figure(
        go.Heatmap(
            z=corr,
            x=factor_names,
            y=factor_names,
            colorscale="RdBu",
            zmid=0,
            zmin=-1,
            zmax=1,
            text=[
                [f"{corr[i, j]:.2f}" for j in range(len(factor_names))]
                for i in range(len(factor_names))
            ],
            texttemplate="%{text}",
            hovertemplate=("%{x} vs %{y}<br>Correlation: %{z:.4f}<extra></extra>"),
        )
    )

    fig.update_layout(theme_config["layout"])
    fig.update_layout(
        title="Factor Correlation Matrix",
        height=height,
        width=width or theme_config["defaults"]["width"],
        xaxis={"side": "bottom"},
        yaxis={"autorange": "reversed"},
    )

    return fig


def plot_vif_bar(
    result: FactorModelResult | RollingExposureResult,
    *,
    factor_data: FactorData | None = None,
    theme: str | None = None,
    height: int = 400,
    width: int | None = None,
) -> go.Figure:
    """Bar chart of Variance Inflation Factors with threshold lines.

    VIF > 5 indicates moderate multicollinearity.
    VIF > 10 indicates severe multicollinearity.

    Parameters
    ----------
    result : FactorModelResult | RollingExposureResult
        Model result (uses stability.vif if RollingExposureResult).
    factor_data : FactorData | None
        Factor data for computing VIF if not in result.
    theme : str | None
        Plot theme name.
    height : int
        Figure height.
    width : int | None
        Figure width.

    Returns
    -------
    go.Figure
        Plotly bar chart with threshold lines.
    """
    from ml4t.diagnostic.evaluation.factor.results import RollingExposureResult as _Rolling

    theme_config = get_theme_config(theme)
    colorway = theme_config["colorway"]

    # Extract VIF values
    vif: dict[str, float] | None = None
    if isinstance(result, _Rolling) and result.stability.vif is not None:
        vif = result.stability.vif
    elif factor_data is not None:
        from ml4t.diagnostic.evaluation.factor.rolling_model import _compute_vif

        X = factor_data.get_factor_array()
        vif = _compute_vif(X, factor_data.factor_names)

    if vif is None:
        raise ValueError(
            "VIF not available. Pass compute_vif=True to rolling model or provide factor_data."
        )

    factors = list(vif.keys())
    values = list(vif.values())

    # Color by severity
    colors = []
    for v in values:
        if v > 10:
            colors.append("#ef4444")  # Red
        elif v > 5:
            colors.append("#D4A84B")  # Amber
        else:
            colors.append(colorway[0])

    fig = go.Figure(
        go.Bar(
            x=factors,
            y=values,
            marker_color=colors,
            hovertemplate="<b>%{x}</b><br>VIF: %{y:.2f}<extra></extra>",
        )
    )

    # Threshold lines
    fig.add_hline(
        y=5,
        line_dash="dash",
        line_color="#D4A84B",
        annotation_text="Moderate (5)",
        annotation_position="top right",
    )
    fig.add_hline(
        y=10,
        line_dash="dash",
        line_color="#ef4444",
        annotation_text="Severe (10)",
        annotation_position="top right",
    )

    fig.update_layout(theme_config["layout"])
    fig.update_layout(
        title="Variance Inflation Factor (VIF)",
        yaxis_title="VIF",
        height=height,
        width=width or theme_config["defaults"]["width"],
    )

    return fig


def _compute_acf(x: np.ndarray, max_lag: int) -> np.ndarray:
    """Compute autocorrelation function."""
    n = len(x)
    mean = np.mean(x)
    centered = x - mean
    var = np.sum(centered**2) / n

    if var == 0:
        return np.zeros(max_lag + 1)

    acf = np.zeros(max_lag + 1)
    acf[0] = 1.0
    for k in range(1, max_lag + 1):
        acf[k] = np.sum(centered[: n - k] * centered[k:]) / (n * var)

    return acf
