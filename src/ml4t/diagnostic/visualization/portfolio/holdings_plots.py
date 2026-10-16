"""Render prepared holdings without recomputing portfolio analytics."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd
import plotly.graph_objects as go
import polars as pl

from ml4t.diagnostic.visualization.core import create_base_figure

if TYPE_CHECKING:
    from ml4t.diagnostic.evaluation.portfolio_analysis import HoldingsAnalysisResult


def _exact_dates(dates: pl.Series) -> list[str]:
    if dates.dtype == pl.Date:
        return [date.isoformat() for date in dates.to_list()]
    return [date.isoformat() for date in pd.DatetimeIndex(dates.to_pandas())]


def _holdings_records(frame: pl.DataFrame | None) -> list[dict[str, Any]] | None:
    if frame is None:
        return None
    rows = frame.to_dicts()
    if "date" in frame.columns:
        for row, date in zip(rows, _exact_dates(frame["date"]), strict=True):
            row["date"] = date
    return rows


def _holdings_figures(
    result: HoldingsAnalysisResult, theme: str, height: int, width: int | None
) -> tuple[dict[str, go.Figure], dict[str, dict[str, Any]]]:
    timeline = result.timeline
    dates = _exact_dates(timeline["date"])
    n_defined = result.coverage["n_normalized_observations"]
    normalized = n_defined > 0
    units = "fraction of reported NAV" if normalized else "account currency"
    suffix = "_fraction" if normalized else ""
    figures: dict[str, go.Figure] = {}
    availability: dict[str, dict[str, Any]] = {
        "holdings_normalization": {
            "status": "available"
            if n_defined == timeline.height
            else "partial"
            if normalized
            else "unavailable",
            "reason": (
                f"NAV normalization defined for {n_defined} of {timeline.height} observations. "
                + (
                    "Gaps remain missing; raw account values are retained in the export."
                    if normalized
                    else "Raw account-currency exposures are shown instead of undefined NAV fractions."
                )
            ),
        }
    }

    def figure(title: str, unit: str) -> go.Figure:
        fig = create_base_figure(
            title=title,
            xaxis_title="Observation key",
            yaxis_title=unit,
            theme=theme,
            height=max(height, 320),
            width=width,
        )
        fig.update_layout(
            showlegend=True,
            legend={"orientation": "h", "x": 0, "y": -0.3},
            margin={"l": 65, "r": 25, "t": 45, "b": 115},
            meta={
                "analysis": "prepared_holdings",
                "units": unit,
                "snapshot_kind": result.snapshot_kind,
                "n_observations": timeline.height,
                "description": f"{timeline.height} supplied {result.snapshot_kind} observations; {unit}. No calendar grouping or intraday reconstruction. Exact observation keys and raw values remain in the export.",
            },
        )
        figures[title] = fig
        return fig

    exposure = figure("Holdings Exposure", units)
    for name, column, sign in [
        ("Long", "long_exposure", 1),
        ("Short (signed)", "short_exposure", -1),
        ("Gross", "gross_exposure", 1),
        ("Net", "net_exposure", 1),
        ("Account cash", "cash", 1),
    ]:
        values = [
            None if value is None else sign * value
            for value in timeline[f"{column}{suffix}"].to_list()
        ]
        exposure.add_trace(
            go.Scatter(
                x=dates,
                y=values,
                name=name,
                mode="lines+markers",
                connectgaps=False,
                customdata=[
                    [date, raw] for date, raw in zip(dates, timeline[column].to_list(), strict=True)
                ],
                hovertemplate="Observed %{customdata[0]}<br>Value: %{y}<br>Raw account value: %{customdata[1]}<extra>%{fullData.name}</extra>",
            )
        )
    if timeline["cash"].null_count() == timeline.height:
        availability["cash"] = {
            "status": "unavailable",
            "reason": "Reported account cash is unavailable; cash is not inferred from holdings or NAV.",
        }

    counts = figure("Position Counts", "positions")
    for name, column in [
        ("Open", "open_positions"),
        ("Long", "long_positions"),
        ("Short", "short_positions"),
    ]:
        counts.add_trace(
            go.Scatter(
                x=dates,
                y=timeline[column].to_list(),
                name=name,
                mode="lines+markers",
                connectgaps=False,
            )
        )

    if timeline["gross_hhi"].null_count() < timeline.height:
        concentration = figure("Gross-Capital Concentration", "dimensionless share / HHI")
        for name, column in [
            ("Largest gross share", "largest_gross_share"),
            ("Gross HHI", "gross_hhi"),
        ]:
            concentration.add_trace(
                go.Scatter(
                    x=dates,
                    y=timeline[column].to_list(),
                    name=name,
                    mode="lines+markers",
                    connectgaps=False,
                )
            )
        availability["concentration"] = {"status": "available", "reason": None}
    else:
        availability["concentration"] = {
            "status": "unavailable",
            "reason": "No invested capital in the supplied observations; gross-share concentration is undefined, not zero.",
        }

    if (
        timeline["max_long_weight"].null_count() < timeline.height
        or timeline["min_short_weight"].null_count() < timeline.height
    ):
        weights = figure("Signed Position Weights", "fraction of reported NAV")
        for name, column in [
            ("Maximum long", "max_long_weight"),
            ("Median long", "median_long_weight"),
            ("Median short", "median_short_weight"),
            ("Minimum short", "min_short_weight"),
        ]:
            weights.add_trace(
                go.Scatter(
                    x=dates,
                    y=timeline[column].to_list(),
                    name=name,
                    mode="lines+markers",
                    connectgaps=False,
                )
            )
        availability["signed_weights"] = {"status": "available", "reason": None}
    else:
        availability["signed_weights"] = {
            "status": "unavailable",
            "reason": "No defined signed NAV weights: the supplied observations have no invested side or usable NAV.",
        }

    for side in ["long", "short", "absolute"]:
        top = result.top_positions(side=side)
        availability[f"top_{side}"] = {
            "status": "available",
            "reason": None if top.height else f"No {side} holdings in the supplied observations.",
        }
        if top.is_empty():
            continue
        fig = figure(f"Top {side.title()} Holdings", "account currency")
        fig.add_trace(
            go.Bar(x=top["value"].to_list(), y=top["asset"].to_list(), orientation="h", name=side)
        )
        fig.update_layout(
            showlegend=False,
            xaxis_title="Sample-extreme marked value (account currency)",
            yaxis_title="Asset ID",
        )
        fig.update_yaxes(type="category", autorange="reversed")

    if result.sectors is None:
        availability["sectors"] = {
            "status": "not_requested",
            "reason": "Sector analysis was not requested; prepare holdings with sector_mapping to include it.",
        }
    elif result.sectors.is_empty():
        availability["sectors"] = {
            "status": "available",
            "reason": "No asset holdings to classify; account cash remains separate.",
        }
    else:
        availability["sectors"] = {
            "status": "available",
            "reason": "Asset sectors only. Missing labels remain unclassified; account cash is not a sector.",
        }
        for name, column in [
            ("Sector Net Exposure", "net_exposure"),
            ("Sector Gross Exposure", "gross_exposure"),
        ]:
            sectors = figure(name, units)
            for label in result.sectors["sector"].unique().sort().to_list():
                group = result.sectors.filter(
                    pl.col("sector").is_null() if label is None else pl.col("sector") == label
                )
                aligned = timeline.select("date").join(group, on="date", how="left")
                sectors.add_trace(
                    go.Scatter(
                        x=dates,
                        y=aligned[f"{column}{suffix}"].to_list(),
                        name="Unclassified (missing label)" if label is None else str(label),
                        mode="lines+markers",
                        connectgaps=False,
                        meta={"sector": label},
                        customdata=aligned[column].to_list(),
                        hovertemplate="Value: %{y}<br>Raw account value: %{customdata}<extra>%{fullData.name}</extra>",
                    )
                )
    return figures, availability
