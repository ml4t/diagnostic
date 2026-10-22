"""Render prepared portfolio regimes/cones in the existing report, without refitting."""

from __future__ import annotations

from html import escape
from typing import TYPE_CHECKING

import numpy as np
import plotly.graph_objects as go
import polars as pl

from ml4t.diagnostic.evaluation.portfolio_analysis import PortfolioRegimeResult, WealthConeResult
from ml4t.diagnostic.evaluation.portfolio_analysis.metrics import _safe_cumprod
from ml4t.diagnostic.evaluation.portfolio_analysis.regimes import _window_mask
from ml4t.diagnostic.visualization.core import create_base_figure

if TYPE_CHECKING:
    from ml4t.diagnostic.evaluation import PortfolioAnalysis


def validate_regime_result(
    analysis: PortfolioAnalysis,
    result: PortfolioRegimeResult,
    kind: str,
) -> PortfolioRegimeResult:
    """Reject a prepared sample/scale from another account before output mutation."""
    if not isinstance(result, PortfolioRegimeResult):
        raise TypeError(f"{kind} must be a prepared PortfolioRegimeResult")
    metadata = result.metadata
    if metadata["kind"] != ("stress" if kind == "stress" else "live_split"):
        raise ValueError(f"{kind} requires its corresponding prepared regime kind")
    expected = pl.DataFrame({"date": analysis.dates, "return": analysis.returns})
    if analysis.benchmark is not None:
        expected = expected.with_columns(pl.Series("benchmark", analysis.benchmark))
    label = metadata["label_column"]
    for sample in result.samples.partition_by(label, maintain_order=True):
        actual = sample.drop(label)
        selected = expected.filter(pl.col("date").is_in(actual["date"].implode()))
        if not actual.equals(selected):
            raise ValueError(f"{kind} must match the exact analysis observations")
    if kind == "live" and not result.samples.drop(label).sort("date").equals(expected):
        raise ValueError("live must partition the exact analysis observations")
    if (
        metadata["periods_per_year"] != analysis._periods_per_year
        or metadata["annual_risk_free"] != analysis._risk_free
    ):
        raise ValueError(f"{kind} must use the exact analysis scale and annual risk-free rate")
    return PortfolioRegimeResult.from_json(result.to_json())


def validate_cone_result(analysis: PortfolioAnalysis, result: WealthConeResult) -> WealthConeResult:
    """Validate actual training/live inputs without drawing another set of paths."""
    if not isinstance(result, WealthConeResult):
        raise TypeError("wealth_cones must be a prepared WealthConeResult")
    expected = pl.DataFrame({"date": analysis.dates, "return": analysis.returns})
    if analysis.benchmark is not None:
        expected = expected.with_columns(pl.Series("benchmark", analysis.benchmark))
    metadata = result.metadata
    mask = _window_mask(expected["date"], None, metadata["live_start"], end_exclusive=True)
    live = result.live.select(expected.columns)
    if not result.training.equals(expected.filter(mask)) or not live.equals(
        expected.filter(~mask).head(metadata["horizon"])
    ):
        raise ValueError("wealth_cones must match the exact analysis training/live observations")
    return WealthConeResult.from_json(result.to_json())


def _style(figure: go.Figure, height: int, width: int | None) -> go.Figure:
    figure.update_layout(
        width=width,
        autosize=width is None,
        height=max(height, 360) + 22 * len(figure.data),
        font_size=14,
        title_font_size=16,
        legend_font_size=14,
        showlegend=True,
        legend={"orientation": "h", "y": -0.25, "x": 0},
        margin={"l": 58, "r": 16, "t": 35, "b": 110},
    )
    figure.update_xaxes(tickfont_size=14, title_font_size=14, automargin=True)
    figure.update_yaxes(tickfont_size=14, title_font_size=14, automargin=True)
    return figure


def regime_figures(
    result: PortfolioRegimeResult,
    title: str,
    theme: str | None,
    height: int,
    width: int | None,
) -> dict[str, go.Figure]:
    """Plot actual per-window products; overlapping stress memberships remain explicit."""
    figure = create_base_figure(
        title=title, xaxis_title="Supplied observation", yaxis_title="Wealth (start 1)", theme=theme
    )
    label = result.metadata["label_column"]
    for sample in result.samples.partition_by(label, maintain_order=True):
        name = sample[label][0]
        exact = sample["date"].cast(pl.String).to_list()
        for column in ("return", "benchmark"):
            if column in sample.columns:
                figure.add_trace(
                    go.Scatter(
                        x=exact,
                        y=_safe_cumprod(1 + sample[column].to_numpy()).tolist(),
                        name=f"{name}" + (" benchmark" if column == "benchmark" else ""),
                        mode="lines+markers",
                        line={"dash": "dot" if column == "benchmark" else "solid"},
                        customdata=exact,
                        hovertemplate="%{customdata}: %{y:.6g}<extra>%{fullData.name}</extra>",
                    )
                )
    if not figure.data:
        figure.add_annotation(
            text="No observed samples in the requested windows",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
            font_size=13,
        )
    figure.update_xaxes(type="category", nticks=3)
    figure.update_layout(
        meta={
            "analysis": "portfolio_regimes",
            "description": "Each window compounds only its actual selected returns from wealth 1. Overlapping stress windows are separate samples. Exact observation keys remain on hover/export; gaps are not fabricated sessions.",
            "kind": result.metadata["kind"],
        }
    )
    return {title: _style(figure, height, width)}


def wealth_cone_figure(
    result: WealthConeResult,
    theme: str | None,
    height: int,
    width: int | None,
) -> go.Figure:
    """Display declared distribution bands and the actual held-out live path."""
    figure = create_base_figure(
        title="Training-only wealth cones",
        xaxis_title="Forward observation",
        yaxis_title="Wealth (capital units)",
        theme=theme,
    )
    for definition, level in (
        result.bands.select("definition", "level").unique(maintain_order=True).iter_rows()
    ):
        values = result.bands.filter(
            (pl.col("definition") == definition) & (pl.col("level") == level)
        )
        name = (
            "Mean"
            if definition == "mean"
            else f"Quantile {level:g}"
            if definition == "quantile"
            else f"Mean {level:+g} std"
        )
        figure.add_trace(
            go.Scatter(
                x=values["step"].to_list(),
                y=values["wealth"].to_list(),
                name=name,
                mode="lines",
                line={"dash": "dot" if definition == "mean_std" else "solid"},
            )
        )
    if result.live.height:
        figure.add_trace(
            go.Scatter(
                x=result.live["step"].to_list(),
                y=result.live["wealth"].to_list(),
                name="Actual live wealth",
                mode="lines+markers",
                line={"color": figure.layout.font.color, "width": 3},
                customdata=result.live["date"].cast(pl.String).to_list(),
                hovertemplate="%{customdata}: %{y:.6g}<extra>Actual live wealth</extra>",
            )
        )
    horizon = result.metadata["horizon"]
    figure.update_xaxes(
        tickmode="array",
        tickvals=np.unique(np.linspace(1, horizon, min(horizon, 4), dtype=int)).tolist(),
    )
    figure.update_layout(
        meta={
            "analysis": "wealth_cones",
            "method": result.metadata["method"],
            "description": f"Training-only {result.metadata['method']} resampling of {result.metadata['n_training']} strictly pre-live observations. Steps count observations, not future sessions. Mean +/- population std bounds are not quantiles or mean confidence intervals; quantile lines are separately labeled. Negative std bounds are not clipped. Actual live outcomes never train the forecast.",
        }
    )
    return _style(figure, max(height, 480), width)


def regime_tables_html(result: PortfolioRegimeResult, title: str) -> str:
    """Show every requested window, exact sample counts and existing metric units."""
    sections = [
        f'<section class="metrics-summary"><h2>{escape(title)}</h2>',
        "<p>Returns, drawdowns and volatility are fractions; Sharpe and beta are dimensionless. Annualized metrics use the declared common scale. Coverage compares requested and observed extents, not an inferred trading grid.</p>",
    ]
    for name in ("coverage", "statistics"):
        table = getattr(result, name)
        sections.append(f"<h3>{escape(name.title())}</h3>")
        if table.is_empty():
            sections.append("<p>No windows requested.</p>")
            continue
        sections.append(
            '<div style="overflow-x:auto"><table style="font-size:14px;border-spacing:12px 6px;white-space:nowrap"><thead><tr>'
        )
        sections.extend(f"<th>{escape(column)}</th>" for column in table.columns)
        sections.append("</tr></thead><tbody>")
        for row in table.iter_rows():
            sections.append("<tr>")
            for value in row:
                text = (
                    "unavailable"
                    if value is None or isinstance(value, float) and not np.isfinite(value)
                    else str(value)
                )
                sections.append(f"<td>{escape(text)}</td>")
            sections.append("</tr>")
        sections.append("</tbody></table></div>")
    sections.append("</section>")
    return "".join(sections)
