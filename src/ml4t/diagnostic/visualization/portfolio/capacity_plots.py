"""Render prepared liquidity without refitting or inventing market coverage."""

from __future__ import annotations

from html import escape
from typing import TYPE_CHECKING, Any

import numpy as np
import plotly.graph_objects as go

from ml4t.diagnostic.metrics import sharpe_ratio
from ml4t.diagnostic.visualization.core import create_base_figure

from .holdings_plots import _exact_dates

if TYPE_CHECKING:
    from ml4t.diagnostic.evaluation.portfolio_analysis import CapacityAnalysisResult


def _capacity_tables(result: CapacityAnalysisResult) -> str:
    def format_value(value: float) -> str:
        return "Unbounded" if np.isposinf(value) else f"{value:.6g}"

    tables = []
    for title, frame in (
        ("Worst liquidation: full/recent observations", result.worst_positions),
        ("Worst traded participation: full/recent observations", result.worst_transactions),
    ):
        frame = frame.select(
            [
                name
                for name in (
                    "sample",
                    "asset",
                    "date",
                    "value",
                    "lagged_dollar_volume",
                    "liquidation_duration",
                    "absolute_units",
                    "participation",
                )
                if name in frame.columns
            ]
        )
        tables.append(
            f'<section class="metrics-summary"><h2>{escape(title)}</h2><p>Scroll horizontally to read all columns on narrow screens.</p><div style="overflow-x:auto;max-width:100%">'
            + frame.to_pandas().to_html(
                index=False,
                escape=True,
                na_rep="Unavailable",
                float_format=format_value,
                border=0,
                classes="capacity-table",
            )
            + "</div></section>"
        )
    meta = result.metadata
    explanation = (
        f"Liquidation: prior {meta['lookback']} {meta['window_unit']} rows, participation {meta['participation']:g}. "
        f"{meta['missing_liquidation_rows']} unavailable position rows; {meta['unbounded_liquidation_rows']} unbounded rows. "
        f"Currency {meta['currency']}; price/volume basis {meta['price_basis']}. "
        f"{meta['book_assumption']}. Impact: {meta['model']}; {meta['calibration']}. "
        "Finite chart lines leave gaps for unknown or unbounded durations; the tables distinguish them. "
        "Participation uses observed same-bar volume, not prospective liquidity. No fills or market data are fabricated."
    )
    return (
        f'<section class="metrics-summary"><h2>Capacity assumptions and coverage</h2><p>{escape(explanation)}</p></section>'
        + "".join(tables)
    )


def _capacity_figures(
    result: CapacityAnalysisResult,
    theme: str,
    height: int,
    width: int | None,
    periods_per_year: int,
    risk_free: float,
) -> tuple[dict[str, go.Figure], dict[str, dict[str, Any]]]:
    figures = {}
    metadata = result.metadata

    def figure(title: str, units: str) -> go.Figure:
        fig = create_base_figure(
            title=title, yaxis_title=units, theme=theme, height=max(height, 360), width=width
        )
        fig.update_layout(
            showlegend=True,
            legend={"orientation": "h", "y": -0.3},
            margin={"l": 65, "r": 25, "t": 45, "b": 140},
            meta={
                "analysis": "prepared_capacity",
                "description": "Supplied market coverage only. Unknown and unbounded liquidation values leave chart gaps; consult the explicit coverage tables. Hypothetical impact is not an executable capacity estimate.",
                **metadata,
            },
        )
        figures[title] = fig
        return fig

    liquidation = figure("Lagged Market Liquidation", metadata["window_unit"] + "s")
    values = result.liquidation
    for asset in values["asset"].unique(maintain_order=True):
        sample = values.filter(values["asset"] == asset)
        y = sample["liquidation_duration"].to_list()
        liquidation.add_scatter(
            x=_exact_dates(sample["date"]),
            y=[value if value is not None and np.isfinite(value) else None for value in y],
            mode="lines+markers",
            name=str(asset),
            connectgaps=False,
        )
    participation = figure("Observed Market Participation", "fraction of same-bar volume")
    values = result.participation
    for asset in values["asset"].unique(maintain_order=True):
        sample = values.filter(values["asset"] == asset)
        participation.add_scatter(
            x=_exact_dates(sample["date"]),
            y=sample["participation"].to_list(),
            mode="lines+markers",
            name=str(asset),
            connectgaps=False,
        )
    scenarios = result.scenarios
    if not scenarios.is_empty():
        sweep = figure("Hypothetical Capital Sweep", "annualized Sharpe ratio (dimensionless)")
        capitals, ratios = [], []
        for capital in scenarios["capital"].unique().sort():
            sample = scenarios.filter(scenarios["capital"] == capital)
            values = sample["adjusted_return"].to_numpy()
            capitals.append(capital)
            ratios.append(
                sharpe_ratio(values, periods_per_year=periods_per_year, risk_free_rate=risk_free)
                if np.isfinite(values).all()
                else None
            )
        sweep.add_scatter(x=capitals, y=ratios, mode="lines+markers", name="Extra impact")
        sweep.update_xaxes(title_text=metadata["currency"] + " starting capital")
        sweep.update_layout(
            meta={
                **sweep.layout.meta,
                "periods_per_year": periods_per_year,
                "risk_free_annual": risk_free,
            }
        )
    return figures, {
        "capacity": {
            "status": "partial" if metadata["missing_liquidation_rows"] else "available",
            "reason": f"{metadata['missing_liquidation_rows']} unavailable liquidation rows (including lag warmup); {metadata['unbounded_liquidation_rows']} unbounded rows. Model is hypothetical.",
        }
    }
