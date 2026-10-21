"""Render prepared actual activity without estimating fills or repricing scenarios."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import plotly.graph_objects as go
import polars as pl

from ml4t.diagnostic.visualization.core import create_base_figure

from .holdings_plots import _exact_dates

if TYPE_CHECKING:
    from ml4t.diagnostic.evaluation.portfolio_analysis import TransactionAnalysisResult


def _transaction_figures(
    result: TransactionAnalysisResult, theme: str, height: int, width: int | None
) -> tuple[dict[str, go.Figure], dict[str, dict[str, Any]]]:
    timeline = result.timeline
    dates = _exact_dates(timeline["date"])
    figures = {}
    description = (
        f"Clock: {result.metadata['clock']}. Turnover denominator: {result.metadata['turnover_denom']}; "
        f"method: {result.metadata['turnover_method']}. "
        "Observed costs are already in supplied net returns; scenarios subtract only extra bps. "
        "Missing denominators/costs stay undefined. No fills are estimated from weights."
        " Exact original keys are retained on hover and in the export."
    )

    def figure(name: str, units: str) -> go.Figure:
        fig = create_base_figure(
            title=name, yaxis_title=units, theme=theme, height=max(320, height), width=width
        )
        fig.update_layout(
            showlegend=True,
            legend={"orientation": "h", "x": 0, "y": -0.3},
            margin={"l": 65, "r": 25, "t": 45, "b": 115},
            meta={"analysis": "prepared_transactions", "units": units, "description": description},
        )
        selected = sorted({0, len(dates) // 2, len(dates) - 1})
        ticks = (
            [timeline["date"][index].strftime("%b %d") for index in selected]
            if timeline["date"].dtype == pl.Date
            else [f"Obs {index + 1}" for index in selected]
        )
        fig.update_xaxes(
            type="category",
            tickmode="array",
            tickvals=[dates[index] for index in selected],
            ticktext=ticks,
            tickangle=0,
            title_text="Exchange session" if timeline["date"].dtype == pl.Date else "Observation",
        )
        figures[name] = fig
        return fig

    notional = figure("Actual Traded Notional", "account currency")
    notional.add_trace(
        go.Bar(x=dates, y=timeline["filled_notional"].to_list(), name="Gross actual notional")
    )
    units = figure("Absolute Executed Units", "supplied execution units")
    units.add_trace(go.Bar(x=dates, y=timeline["absolute_units"].to_list(), name="Absolute units"))
    units.update_layout(
        meta={
            "analysis": "prepared_transactions",
            "units": "execution units",
            "description": result.metadata["quantity_units"],
        }
    )
    turnover = figure("Transaction Turnover", "fraction of selected denominator")
    turnover.add_trace(
        go.Scatter(
            x=dates,
            y=timeline["turnover"].to_list(),
            name=result.metadata["turnover_method"],
            connectgaps=False,
            mode="lines+markers",
        )
    )
    distribution = create_base_figure(
        title="Turnover Distribution",
        xaxis_title="turnover fraction",
        yaxis_title="observations",
        theme=theme,
        height=max(320, height),
        width=width,
    )
    for label, frame in (
        ("All defined", timeline),
        ("Active defined", timeline.filter(pl.col("num_fills") > 0)),
    ):
        distribution.add_trace(go.Histogram(x=frame["turnover"].drop_nulls().to_list(), name=label))
    distribution.update_layout(
        barmode="overlay",
        showlegend=True,
        legend={"orientation": "h", "y": -0.3},
        margin={"l": 65, "r": 25, "t": 45, "b": 110},
        meta={
            "analysis": "prepared_transactions",
            "description": "Finite defined turnover only; active excludes no-fill observations. Unknown denominators are omitted, not zero-filled.",
        },
    )
    figures["Turnover Distribution"] = distribution
    timing_availability = {
        "status": "available",
        "reason": "All actual event hours retained; no US market-hours filter.",
    }
    try:
        timing = result.execution_timing()
    except ValueError as error:
        timing_availability = {"status": "unavailable", "reason": str(error)}
    else:
        timing_figure = create_base_figure(
            title="Execution Time Distribution",
            xaxis_title="civil time (retained timezone)",
            yaxis_title="fraction of actual notional",
            theme=theme,
            height=max(320, height),
            width=width,
        )
        timing_figure.add_trace(
            go.Bar(x=timing["time"].to_list(), y=timing["notional_fraction"].to_list())
        )
        active_timing = timing.filter(pl.col("num_fills") > 0)
        timing_figure.add_trace(
            go.Scatter(
                x=active_timing["time"].to_list(),
                y=active_timing["notional_fraction"].to_list(),
                mode="markers",
                marker={"size": 6},
                showlegend=False,
            )
        )
        timing_figure.update_xaxes(
            type="category", tickmode="array", tickvals=["00:00", "06:00", "12:00", "18:00"]
        )
        timing_figure.update_layout(
            meta={
                "analysis": "prepared_transactions",
                "timezone": timing["timezone"][0],
                "bin_minutes": 5,
                "description": "All actual event hours in the retained timezone, in five-minute bins; no US equity-hours filter.",
            }
        )
        figures["Execution Time Distribution"] = timing_figure
    costs = figure("Observed Execution Costs", "account currency")
    costs.add_trace(
        go.Scatter(
            x=dates,
            y=timeline["implementation_cost"].to_list(),
            name="Known costs",
            connectgaps=False,
            mode="lines+markers",
        )
    )
    scenarios = figure("Incremental Cost Return Scenarios", "net return fraction")
    for frame in result.scenarios.partition_by("incremental_bps", maintain_order=True):
        scenarios.add_trace(
            go.Scatter(
                x=_exact_dates(frame["date"]),
                y=frame["net_return"].to_list(),
                name=f"Extra {frame['incremental_bps'][0]:g} bps",
                mode="lines+markers",
                connectgaps=False,
            )
        )
    defined = result.metrics["n_normalized_observations"]
    total = result.metrics["n_observations"]
    return figures, {
        "execution_timing": timing_availability,
        "transaction_normalization": {
            "status": "available" if defined == total else "partial" if defined else "unavailable",
            "reason": f"Turnover defined for {defined} of {total} supplied clock observations; raw notional remains available. Means use only defined ratios; both all-clock and active counts are exported.",
        },
        "observed_execution_costs": {
            "status": "available"
            if result.executions["implementation_cost"].null_count() == 0
            else "partial",
            "reason": "Both commission and per-unit price slippage must be reported. Unknown costs are not zero; extra-bps scenarios do not require observed costs.",
        },
    }
