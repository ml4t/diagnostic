"""Render prepared liquidity without refitting or inventing market coverage."""

from __future__ import annotations

from html import escape
from typing import TYPE_CHECKING, Any

import numpy as np
import plotly.graph_objects as go
import polars as pl

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
        frame = frame.with_columns(
            [
                pl.col(name).cast(pl.String)
                for name, dtype in frame.schema.items()
                if dtype == pl.Int128
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

    def clock_axis(fig: go.Figure, dates: pl.Series, title: str) -> None:
        keys = _exact_dates(dates)
        selected = sorted({0, len(keys) // 2, len(keys) - 1}) if keys else []
        unique_days = dates.dtype == pl.Date or dates.dt.date().n_unique() == len(dates)
        fig.update_xaxes(
            type="category",
            categoryorder="array",
            categoryarray=keys,
            tickmode="array",
            tickvals=[keys[index] for index in selected],
            ticktext=[
                dates[index].strftime("%b %d") if unique_days else f"Obs {index + 1}"
                for index in selected
            ],
            tickangle=0,
            title_text=title,
        )

    liquidation = figure("Lagged Market Liquidation", metadata["window_unit"] + "s")
    values = result.liquidation
    for asset in values["asset"].unique(maintain_order=True):
        sample = result.timeline.select("date").join(
            values.filter(values["asset"] == asset).select("date", "value", "liquidation_duration"),
            on="date",
            how="left",
        )
        y = sample["liquidation_duration"].to_list()
        liquidation.add_scatter(
            x=_exact_dates(sample["date"]),
            y=[
                0.0
                if position is None
                else value
                if value is not None and np.isfinite(value)
                else None
                for position, value in zip(sample["value"].to_list(), y, strict=True)
            ],
            mode="lines+markers",
            name=str(asset),
            connectgaps=False,
        )
    clock_axis(liquidation, result.timeline["date"], "Observation")
    liquidation.update_layout(
        meta={
            **liquidation.layout.meta,
            "description": "Exact observation keys use equal-spaced categories so nanosecond instants remain distinct. Complete-book absent assets have zero exposure; unknown market coverage and unbounded duration remain gaps, with explicit tables.",
        }
    )
    participation = figure("Observed Market Participation", "fraction of same-bar volume")
    values = result.participation
    for asset in values["asset"].unique(maintain_order=True):
        sample = values.filter(values["asset"] == asset)
        participation.add_scatter(
            x=_exact_dates(sample["date"]),
            y=sample["participation"].to_list(),
            mode="markers",
            name=str(asset),
            connectgaps=False,
        )
    clock_axis(participation, values["date"].unique().sort(), "Activity observation")
    if values.is_empty():
        participation.add_annotation(
            text="Actual transactions were not supplied"
            if not metadata["transactions_supplied"]
            else "No recorded executions on the supplied clock",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
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
