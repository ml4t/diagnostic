"""Round-trip views composed by the existing portfolio HTML/notebook renderer."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from html import escape
from math import isfinite
from numbers import Integral, Real
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
        HoldingsAnalysisResult,
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
                str(result.metadata.get("asset_labels", {}).get(value, value))
                if key == "asset"
                else labels[value]
                for value in categories
            ]
            values = {(row["metric"], row[key]): row["value"] for row in records}
            displayed = [
                [metric.replace("_", " ")] + [values.get((metric, value)) for value in categories]
                for metric in frame["metric"].unique(maintain_order=True).to_list()
            ]
        else:
            columns = frame.columns
            displayed = [
                [
                    result.metadata.get("asset_labels", {}).get(value, value)
                    if column == "asset"
                    else value
                    for column, value in row.items()
                ]
                for row in records
            ]
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


def _draw_lifetimes(
    figure: go.Figure,
    observations: pl.DataFrame,
    *,
    disp_amount: int | None = None,
    lsize: float = 6,
    asset_labels: Mapping[Any, str] | None = None,
) -> None:
    if disp_amount is not None and (
        isinstance(disp_amount, bool) or not isinstance(disp_amount, Integral) or disp_amount < 0
    ):
        raise ValueError("disp_amount must be a nonnegative integer or None for all assets")
    if isinstance(lsize, bool) or not isinstance(lsize, Real) or not isfinite(lsize) or lsize <= 0:
        raise ValueError("lsize must be a finite positive line width")
    assets = observations["asset"].drop_nulls().unique(maintain_order=True).to_list()
    if disp_amount is not None:
        # Match the reference's sample without modifying the process-wide RNG.
        indices = np.random.RandomState(1).choice(
            len(assets), size=min(disp_amount, len(assets)), replace=False
        )
        assets = [assets[index] for index in indices]
    asset_rows = {asset: index for index, asset in enumerate(assets)}
    dated = observations.filter(
        pl.col("asset").is_in(assets)
        & pl.col("entry_timestamp").is_not_null()
        & pl.col("exit_timestamp").is_not_null()
    ).to_pandas()
    for _, row in dated.iterrows():
        side = row["direction"]
        exact = [row["entry_timestamp"].isoformat(), row["exit_timestamp"].isoformat()]
        figure.add_trace(
            go.Scatter(
                x=exact,
                y=[asset_rows[row["asset"]]] * 2,
                name=("Synthetic " if row["is_synthetic"] else "Actual ") + side,
                mode="lines+markers",
                showlegend=False,
                line={
                    "color": COLORS["blue"] if side == "long" else COLORS["amber"],
                    "width": lsize,
                    "dash": "dash" if row["is_synthetic"] else "solid",
                },
                customdata=exact,
                hovertemplate="%{customdata}<extra>%{fullData.name}</extra>",
            )
        )
    figure.update_xaxes(type="date", nticks=3)
    figure.update_yaxes(
        tickmode="array",
        tickvals=list(asset_rows.values()),
        ticktext=[str(asset_labels.get(a, a) if asset_labels else a) for a in assets],
    )
    metadata = dict(figure.layout.meta or {})
    figure.update_layout(
        height=max(360, figure.layout.height or 360, 120 + len(assets) * 28),
        meta={
            **metadata,
            "selected_assets": assets,
            "sampling": "all assets"
            if disp_amount is None
            else "local RandomState(1) without replacement",
            "description": metadata.get("description", "")
            + " Blue: long; amber: short; dashed: synthetic. "
            + (
                "All assets retained; no random sample. "
                if disp_amount is None
                else f"Display sample: {len(assets)} assets, local RandomState(1); cash/statistics are not sampled. "
            )
            + "Lifetime axes display milliseconds; exact nanosecond instants stay on hover/export. "
            + f"{len(dated)} known sampled lifelines; missing times do not remove cash observations.",
        },
    )
    if not len(dated):
        figure.update_xaxes(visible=False)
        figure.update_yaxes(visible=False)
        figure.add_annotation(
            text="Unavailable for this sample.<br>No known sampled lifetimes.",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
            font_size=13,
        )


def _draw_probability(
    figure: go.Figure,
    posterior: Mapping[str, Any],
    *,
    points: int = 501,
    legacy_limits: bool = False,
) -> None:
    defined = posterior["lower"] is not None
    if defined:
        x = np.linspace(0, 1, points)
        a, b = posterior["posterior_alpha"], posterior["posterior_beta"]
        figure.add_trace(go.Scatter(x=x, y=beta.pdf(x, a, b), name=f"Beta({a:g}, {b:g})"))
        for bound in ("lower", "upper"):
            figure.add_vline(x=posterior[bound], line_dash="dot")
        figure.update_xaxes(
            range=beta.ppf([0.001, 0.999], a, b).tolist() if legacy_limits else [0, 1], nticks=3
        )
        figure.update_yaxes(rangemode="tozero")
    else:
        figure.add_annotation(
            text="Undefined zero-prior posterior.<br>Both wins and nonwins are required.",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
            font_size=13,
        )
        figure.update_xaxes(visible=False)
        figure.update_yaxes(visible=False)
    metadata = dict(figure.layout.meta or {})
    figure.update_layout(
        meta={
            **metadata,
            "posterior": dict(posterior),
            "description": metadata.get("description", "")
            + f" Prior: Beta({posterior['prior_alpha']:g}, {posterior['prior_beta']:g}); equal-tail95% interval. "
            + "Break-even is nonwinning. Missing cash is excluded and counted. "
            + (
                "The zero-prior method is undefined for empty or one-sided samples; no NaN density is plotted."
                if posterior["prior_method"] == "legacy_zero_prior"
                else "Empty samples display the Beta(1,1) prior, not evidence of performance."
            ),
        }
    )


def create_round_trip_dashboard(
    analysis: PortfolioAnalysis | None,
    round_trips: RoundTripResult,
    *,
    pnl: Literal["gross", "net"] = "net",
    return_basis: Literal["lot", "nav"] = "lot",
    portfolio_value: pl.DataFrame | pd.DataFrame | pd.Series | None = None,
    sector_mappings: Mapping[Any, str] | None = None,
    intraday: HoldingsAnalysisResult | None = None,
    include_synthetic: bool = False,
    calendar: str | CalendarUnset | None = UNSET_CALENDAR,
    alignment: Literal["declared", "civil_date"] = "declared",
    theme: str | None = None,
    height_per_row: int = 360,
    width: int | None = None,
    statistics_method: Literal["native", "legacy_pyfolio"] = "native",
    probability_method: Literal["uniform_prior", "legacy_zero_prior"] = "uniform_prior",
    lifetime_sample: int | None = None,
    lifetime_linewidth: float = 6,
    asset_labels: Mapping[Any, str] | None = None,
    holding_duration_method: Literal["elapsed_days", "legacy_whole_days"] = "elapsed_days",
    return_display: Literal["fraction", "percent"] = "fraction",
) -> PortfolioTearSheet:
    """Add prepared FIFO tables and full round-trip panels to a portfolio report.

    Supplied net account returns remain independent of closing-trade attribution.
    Native FIFO/grouping, gross/net costs, lot/NAV denominators and optional
    hypothetical closes are explicit in the result and report. No positions,
    marks, NAV or terminal fills are inferred. Default displays retain all assets;
    explicit lifetime_sample uses a local deterministic display-only sample.
    Exact ns instants remain in hover/export; the browser's date
    axis has millisecond display precision. Fewer than five observations remain
    visible with their count and Beta(1,1) prior, rather than being suppressed.
    Optional intraday contains complete prepared snapshots on their independent
    clock; they are not resampled onto return or end-of-day observations.
    analysis=None builds an execution-only report with no fabricated performance
    summary. Legacy statistics/probability/whole-day/percent display methods are
    explicit selections; native defaults remain unchanged.
    """
    from .dashboard import create_portfolio_dashboard

    if not isinstance(round_trips, RoundTripResult):
        raise TypeError("round_trips must be a prepared RoundTripResult")
    if holding_duration_method not in {"elapsed_days", "legacy_whole_days"}:
        raise ValueError("holding_duration_method must be elapsed_days or legacy_whole_days")
    if return_display not in {"fraction", "percent"}:
        raise ValueError("return_display must be fraction or percent")
    supplied = analysis.input_data if analysis is not None else None
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
        method=statistics_method,
    )
    attribution = result.profit_attribution(
        pnl=pnl, sector_mappings=sector_mappings, include_synthetic=include_synthetic
    )
    posterior = result.win_probability(
        pnl=pnl, include_synthetic=include_synthetic, method=probability_method
    )
    if analysis is None:
        from .dashboard import PortfolioTearSheet

        if intraday is not None:
            raise ValueError("supply an analysis for additional intraday exposure panels")
        report = PortfolioTearSheet(
            metrics=None,
            availability={
                "portfolio_returns": {
                    "status": "not_requested",
                    "reason": "Execution-only report: no performance summary or fabricated return series.",
                }
            },
        )
    else:
        report = create_portfolio_dashboard(
            analysis,
            theme=theme,
            include_positions=False,
            include_transactions=False,
            intraday=intraday,
            height_per_row=height_per_row,
            width=width,
        )
    if asset_labels is not None:
        statistics.metadata["asset_labels"] = dict(asset_labels)
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

    lifetimes = figure("Round-trip Lifetimes", "Time", "Asset")
    lifetimes.update_layout(
        meta={
            **lifetimes.layout.meta,
            "description": description
            + " Lifetime axes use browser millisecond date precision; exact nanosecond instants remain on hover and in exports.",
        }
    )
    dated = observations.filter(pl.col("entry_timestamp").is_not_null())
    _draw_lifetimes(
        lifetimes,
        observations,
        disp_amount=lifetime_sample,
        lsize=lifetime_linewidth,
        asset_labels=asset_labels,
    )
    probability = figure(
        "Round-trip Win Probability",
        "P(positive cash PnL)",
        "Posterior density",
    )
    _draw_probability(probability, posterior)
    for name, column, units, scale in (
        (
            "Round-trip Holding Duration",
            "duration",
            "Holding days (whole)"
            if holding_duration_method == "legacy_whole_days"
            else "Elapsed days (fractional)",
            1 / 86400e9,
        ),
        ("Round-trip Cash PnL", pnl + "_pnl", "Cash PnL (account currency)", 1),
        (
            "Round-trip Returns",
            statistics.metadata["return_col"],
            ("Lot return" if return_basis == "lot" else "NAV contribution")
            + (" (%)" if return_display == "percent" else " (fraction)"),
            100 if return_display == "percent" else 1,
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
                        x=[
                            np.floor(value * scale)
                            if column == "duration"
                            and holding_duration_method == "legacy_whole_days"
                            else value * scale
                            for value in sample.to_list()
                        ],
                        name="Synthetic" if synthetic else "Actual",
                    )
                )
        item.update_layout(barmode="overlay")
        item.update_layout(
            meta={
                **item.layout.meta,
                "holding_duration_method": holding_duration_method,
                "return_display": return_display,
            }
        )
    for scope, label in (("asset", "Asset"), ("sector", "Sector")):
        if scope == "sector" and sector_mappings is None:
            continue
        sample = attribution.filter(pl.col("scope") == scope)
        item = figure(f"Round-trip {label} Attribution", "PnL share", label)
        item.add_trace(
            go.Bar(
                x=sample["pnl_fraction"].to_list(),
                y=[
                    str(
                        asset_labels.get(value, value)
                        if scope == "asset" and asset_labels
                        else value
                    )
                    for value in sample[scope].to_list()
                ],
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
    for item in figures.values():
        if not item.layout.annotations and not any(
            trace.x is not None and any(value is not None for value in trace.x)
            for trace in item.data
        ):
            item.update_xaxes(visible=False)
            item.update_yaxes(visible=False)
            item.update_layout(showlegend=False)
            item.add_annotation(
                text="Unavailable for this sample.<br>See input and sample coverage.",
                x=0.5,
                y=0.5,
                xref="paper",
                yref="paper",
                showarrow=False,
                font_size=13,
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
