"""Round-trip views composed by the existing portfolio HTML/notebook renderer."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from html import escape
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import polars as pl
from ml4t.specs import UNSET_CALENDAR, CalendarUnset
from scipy.stats import beta

from ml4t.diagnostic.evaluation.portfolio_analysis import RoundTripResult
from ml4t.diagnostic.visualization._colors import COLORS
from ml4t.diagnostic.visualization.core import create_base_figure, validate_theme

if TYPE_CHECKING:
    from ml4t.diagnostic.evaluation.portfolio_analysis import (
        PortfolioAnalysis,
        RoundTripStatisticsResult,
    )

    from .dashboard import PortfolioTearSheet


def _records(frame: pl.DataFrame | None) -> list[dict[str, Any]] | None:
    if frame is None:
        return None
    temporal = []
    for name, dtype in frame.schema.items():
        if isinstance(dtype, pl.Duration):
            temporal.append(
                pl.Series(
                    name,
                    [
                        None
                        if value is None
                        else pd.Timedelta(value, unit=dtype.time_unit).isoformat()
                        for value in frame[name].cast(pl.Int64).to_list()
                    ],
                    dtype=pl.String,
                )
            )
        elif isinstance(dtype, pl.Datetime) or dtype == pl.Date:
            temporal.append(frame[name].cast(pl.String))
    return frame.with_columns(temporal).to_dicts()


def _round_trip_export(report: PortfolioTearSheet) -> dict[str, Any] | None:
    result, statistics = report.round_trips, report.round_trip_statistics
    if result is None or statistics is None:
        return None
    return {
        "metadata": deepcopy(result.metadata),
        "metrics": deepcopy(result.metrics),
        "executions": _records(result.executions),
        "segments": _records(result.segments),
        "open_lots": _records(result.open_lots),
        "opening_lots": _records(result.opening_lots),
        "marks": _records(result.marks),
        "synthetic_segments": _records(result.synthetic_segments),
        "matching_groups": _records(result.matching_groups),
        "statistics": {
            "metadata": deepcopy(statistics.metadata),
            "tables": {name: _records(frame) for name, frame in statistics.tables.items()},
            "coverage": _records(statistics.coverage),
            "observations": _records(statistics.observations),
        },
        "attribution": _records(report.round_trip_attribution),
    }


def _round_trip_tables_html(result: RoundTripStatisticsResult) -> str:
    sections = [
        '<section class="metrics-summary"><h2>Round-trip statistics</h2>',
        f"<p>Cash view: {escape(result.metadata['pnl'])}. Return denominator: "
        f"{escape(result.metadata['return_basis'])}. Synthetic closes included: "
        f"{result.metadata['include_synthetic']}. Missing observations are not zero.</p>",
        '<p class="figure-note">Cash PnL is in supplied account currency. Returns are '
        "fractions of the named denominator, counts are observations, and duration "
        "retains nanoseconds. Asset attribution divides by signed total PnL, not NAV.</p>",
        '<p class="figure-note">Scroll wide tables horizontally. Long labels and exact values are not split across lines.</p>',
    ]
    for name, frame in {**result.tables, "coverage": result.coverage}.items():
        sections.append(f"<h3>{escape(name.title())}</h3>")
        if frame.is_empty():
            sections.append("<p>No completed observations.</p>")
            continue
        records = _records(frame) or []
        if name != "coverage":
            key = "asset" if name == "symbols" else "scope"
            categories = frame[key].unique(maintain_order=True).to_list()
            labels = {"all": "All trades", "short": "Short trades", "long": "Long trades"}
            columns = ["Metric"] + [
                str(value) if key == "asset" else labels[value] for value in categories
            ]
            values = {(row["metric"], row[key]): row["value"] for row in records}
            displayed = [
                [metric.replace("_", " ")] + [values.get((metric, value)) for value in categories]
                for metric in frame["metric"].unique(maintain_order=True).to_list()
            ]
        else:
            columns = frame.columns
            displayed = [list(row.values()) for row in records]
        headers = "".join(f'<th scope="col">{escape(col)}</th>' for col in columns)
        body = "".join(
            "<tr>"
            + "".join(
                f"<td>{escape('Unavailable' if value is None else str(value))}</td>"
                for value in row
            )
            + "</tr>"
            for row in displayed
        )
        sections.append(
            f'<div tabindex="0" role="region" aria-label="Round-trip {escape(name)} table" style="overflow-x:auto">'
            '<table style="font-size:14px;border-spacing:12px 6px;white-space:nowrap">'
            f"<thead><tr>{headers}</tr></thead><tbody>{body}</tbody></table></div>"
        )
    sections.append("</section>")
    return "".join(sections)


def create_round_trip_dashboard(
    analysis: PortfolioAnalysis,
    round_trips: RoundTripResult,
    *,
    pnl: Literal["gross", "net"] = "net",
    return_basis: Literal["lot", "nav"] = "lot",
    portfolio_value: pl.DataFrame | pd.DataFrame | pd.Series | None = None,
    sector_mappings: Mapping[Any, str] | None = None,
    include_synthetic: bool = False,
    calendar: str | CalendarUnset | None = UNSET_CALENDAR,
    alignment: Literal["declared", "civil_date"] = "declared",
    theme: str | None = None,
    height_per_row: int = 360,
    width: int | None = None,
) -> PortfolioTearSheet:
    """Add prepared FIFO tables and full round-trip panels to a portfolio report.

    Supplied net account returns remain independent of closing-trade attribution.
    Native FIFO/grouping, gross/net costs, lot/NAV denominators and optional
    hypothetical closes are explicit in the result and report. No positions,
    marks, NAV or terminal fills are inferred. All assets are displayed without
    random sampling. Exact ns instants remain in hover/export; the browser's date
    axis has millisecond display precision. Fewer than five observations remain
    visible with their count and Beta(1,1) prior, rather than being suppressed.
    """
    from .dashboard import create_portfolio_dashboard

    if not isinstance(round_trips, RoundTripResult):
        raise TypeError("round_trips must be a prepared RoundTripResult")
    supplied = analysis.input_data
    if supplied is not None and (
        supplied.transactions is None or not supplied.transactions.equals(round_trips.executions)
    ):
        raise ValueError("round trips must match this analysis's exact actual transactions")
    theme = validate_theme(theme)
    result = deepcopy(round_trips)
    statistics = result.statistics_tables(
        pnl=pnl,
        return_basis=return_basis,
        portfolio_value=portfolio_value,
        include_synthetic=include_synthetic,
        calendar=calendar,
        alignment=alignment,
    )
    attribution = result.profit_attribution(
        pnl=pnl, sector_mappings=sector_mappings, include_synthetic=include_synthetic
    )
    posterior = result.win_probability(pnl=pnl, include_synthetic=include_synthetic)
    report = create_portfolio_dashboard(
        analysis,
        theme=theme,
        include_positions=False,
        include_transactions=False,
        height_per_row=height_per_row,
        width=width,
    )
    report.round_trips = result
    report.round_trip_statistics = statistics
    report.round_trip_attribution = attribution
    observations = statistics.observations
    total = observations.height
    synthetic_count = observations.filter(pl.col("is_synthetic")).height
    description = (
        f"{total} closing observations; {synthetic_count} explicitly synthetic. "
        f"Matching: {statistics.metadata['matching_method']}; {pnl} cash PnL; "
        f"return denominator: {return_basis}. Supplied account returns are not reconstructed "
        "from realized trade PnL. Open lots remain open; missing marks are not terminal prices."
    )
    figures = {}

    def figure(name: str, xaxis: str, yaxis: str) -> go.Figure:
        item = create_base_figure(
            title=None,
            xaxis_title=xaxis,
            yaxis_title=yaxis,
            theme=theme,
            height=max(360, height_per_row),
            width=width,
        )
        item.update_layout(
            font_size=14,
            title_font_size=16,
            legend_font_size=14,
            margin={"l": 65, "r": 25, "t": 30, "b": 115},
            showlegend=True,
            legend={"orientation": "h", "x": 0, "y": -0.3},
            meta={"analysis": "prepared_round_trips", "description": description},
        )
        item.update_xaxes(tickfont_size=14, title_font_size=14)
        item.update_yaxes(tickfont_size=14, title_font_size=14)
        if width is None:
            item.update_layout(width=None, autosize=True)
        figures[name] = item
        return item

    lifetimes = figure("Round-trip Lifetimes", "Actual time (ms display)", "Asset")
    assets = observations["asset"].unique(maintain_order=True).to_list()
    asset_rows = {asset: index for index, asset in enumerate(assets)}
    dated = observations.filter(pl.col("entry_timestamp").is_not_null()).to_pandas()
    for _, row in dated.iterrows():
        side = row["direction"]
        exact = [row["entry_timestamp"].isoformat(), row["exit_timestamp"].isoformat()]
        lifetimes.add_trace(
            go.Scatter(
                x=exact,
                y=[asset_rows[row["asset"]]] * 2,
                name=("Synthetic " if row["is_synthetic"] else "Actual ") + side,
                mode="lines+markers",
                showlegend=False,
                line={
                    "color": COLORS["blue"] if side == "long" else COLORS["amber"],
                    "width": 6,
                    "dash": "dash" if row["is_synthetic"] else "solid",
                },
                customdata=exact,
                hovertemplate="%{customdata}<extra>%{fullData.name}</extra>",
            )
        )
    lifetimes.update_xaxes(type="date", nticks=3)
    lifetimes.update_yaxes(
        tickmode="array", tickvals=list(asset_rows.values()), ticktext=[str(a) for a in assets]
    )
    lifetimes.update_layout(height=max(360, height_per_row, 120 + len(assets) * 28))
    lifetimes.update_layout(
        meta={
            "analysis": "prepared_round_trips",
            "description": description
            + f" Unknown entry time omits {total - len(dated)} lifelines, not their cash observations. "
            "Blue: long; amber: short; dashed: synthetic. All assets retained; no random sample.",
        }
    )
    probability = figure(
        "Round-trip Win Probability",
        "P(positive cash PnL)",
        "Posterior density",
    )
    x = np.linspace(0, 1, 501)
    probability.add_trace(
        go.Scatter(
            x=x,
            y=beta.pdf(x, posterior["posterior_alpha"], posterior["posterior_beta"]),
            name=f"Beta({posterior['posterior_alpha']:g}, {posterior['posterior_beta']:g})",
        )
    )
    for bound in ("lower", "upper"):
        probability.add_vline(x=posterior[bound], line_dash="dot")
    probability.update_xaxes(range=[0, 1], nticks=3)
    probability.update_layout(
        meta={
            "analysis": "prepared_round_trips",
            "posterior": posterior,
            "description": description
            + " Explicit Beta(1,1) prior, equal-tail95% interval; break-even is nonwinning. "
            "Empty samples display the prior, not evidence of performance.",
        }
    )
    for name, column, units, scale in (
        ("Round-trip Holding Duration", "duration", "Elapsed days (fractional)", 1 / 86400e9),
        ("Round-trip Cash PnL", pnl + "_pnl", "Cash PnL (account currency)", 1),
        (
            "Round-trip Returns",
            statistics.metadata["return_col"],
            "Lot return" if return_basis == "lot" else "NAV contribution",
            1,
        ),
    ):
        item = figure(name, units, "Closing observations")
        for synthetic in (False, True):
            sample = observations.filter(pl.col("is_synthetic") == synthetic)[column].drop_nulls()
            if column == "duration":
                sample = sample.cast(pl.Int64)
            if len(sample):
                item.add_trace(
                    go.Histogram(
                        x=[value * scale for value in sample.to_list()],
                        name="Synthetic" if synthetic else "Actual",
                    )
                )
        item.update_layout(barmode="overlay")
    for scope, label in (("asset", "Asset"), ("sector", "Sector")):
        if scope == "sector" and sector_mappings is None:
            continue
        sample = attribution.filter(pl.col("scope") == scope)
        item = figure(f"Round-trip {label} Attribution", "PnL share", label)
        item.add_trace(
            go.Bar(
                x=sample["pnl_fraction"].to_list(),
                y=[str(value) for value in sample[scope].to_list()],
                orientation="h",
                customdata=sample["pnl"].to_list(),
                name=f"{pnl.title()} cash share",
                hovertemplate="%{y}: %{x:.3f}; cash %{customdata}<extra></extra>",
            )
        )
        item.update_layout(height=max(360, height_per_row, 120 + sample.height * 28))
        item.update_layout(
            meta={
                "analysis": "prepared_round_trips",
                "description": description
                + " Shares divide by signed total PnL, may be negative or exceed one, and are "
                "undefined at zero total. No NAV normalization or complete-case row dropping.",
            }
        )
    report.figures.update(figures)
    report.availability["round_trips"] = {
        "status": "available" if total else "empty",
        "reason": description,
    }
    report.availability["round_trip_lifetimes"] = {
        "status": "available"
        if len(dated) == total and total
        else "partial"
        if len(dated)
        else "unavailable",
        "reason": f"Known entry and exit times for {len(dated)} of {total} completed observations.",
    }
    known_returns = total - observations[statistics.metadata["return_col"]].null_count()
    report.availability["round_trip_return_denominators"] = {
        "status": "available"
        if known_returns == total and total
        else "partial"
        if known_returns
        else "unavailable"
        if total
        else "empty",
        "reason": f"Selected return denominator known for {known_returns} of {total} observations; "
        "undefined selected denominators remain missing, not zero.",
    }
    return report
