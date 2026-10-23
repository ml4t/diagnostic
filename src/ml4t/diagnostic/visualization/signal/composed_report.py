"""Compose owned signal results through the existing Plotly and HTML renderer."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl

from ml4t.diagnostic.utils.formatting import format_finite
from ml4t.diagnostic.visualization.signal.event_plots import _publish_event_html
from ml4t.diagnostic.visualization.signal.information_plots import plot_signal_information
from ml4t.diagnostic.visualization.signal.returns_plots import plot_signal_returns
from ml4t.diagnostic.visualization.signal.turnover_plots import plot_signal_turnover

if TYPE_CHECKING:
    import plotly.graph_objects as go

    from ml4t.diagnostic.results.signal_results import TurnoverAnalysisResult
    from ml4t.diagnostic.signal.information import SignalInformationResult
    from ml4t.diagnostic.signal.returns_report import SignalReturnsResult


def plot_signal_report(
    returns_result: SignalReturnsResult,
    information_result: SignalInformationResult,
    turnover_result: TurnoverAnalysisResult,
    *,
    summary: bool = False,
    theme: str | None = None,
    rolling_window: int = 21,
    bandwidth: float = 1.0,
    ylim_percentiles: tuple[float, float] | None = None,
) -> list[go.Figure]:
    """Compose existing report figures without estimating or altering their samples.

    Return and information components must own the identical prepared input and
    horizon order. Supply dated turnover/rank components from that input; their
    recorded observation clocks remain distinct from return-horizon labels.
    """
    if not isinstance(summary, bool):
        raise TypeError("summary must be boolean")
    if returns_result.metadata["return_columns"] != information_result.metadata[
        "return_columns"
    ] or not returns_result.get_table("prepared_data").equals(
        information_result.get_table("prepared_data")
    ):
        raise ValueError(
            "return and information components must retain the same prepared sample and horizon order"
        )
    if (
        turnover_result.dated_turnover_ipc_base64 is None
        or turnover_result.dated_autocorrelation_ipc_base64 is None
    ):
        raise ValueError("composed reports require actual dated persistence observations")
    families = [
        (
            "returns",
            plot_signal_returns(
                returns_result,
                theme=theme,
                rolling_window=rolling_window,
                bandwidth=bandwidth,
                ylim_percentiles=ylim_percentiles,
            ),
        )
    ]
    if summary:
        families = [
            (
                "returns",
                [
                    figure
                    for figure in families[0][1]
                    if figure.layout.meta["analysis"] == "quantile_mean"
                ],
            )
        ]
    else:
        families.extend(
            [
                ("information", plot_signal_information(information_result, theme=theme)),
                ("turnover", plot_signal_turnover(turnover_result, theme=theme)),
            ]
        )
    figures = []
    for family, collection in families:
        for figure in collection:
            figure.update_layout(
                meta={**dict(figure.layout.meta or {}), "report_component": family}
            )
            figures.append(figure)
    return figures


def generate_signal_report(
    returns_result: SignalReturnsResult,
    information_result: SignalInformationResult,
    turnover_result: TurnoverAnalysisResult,
    *,
    output_file: str | Path | None = None,
    summary: bool = False,
    theme: str | None = None,
    factor_col: str = "factor",
    rolling_window: int = 21,
    bandwidth: float = 1.0,
    ylim_percentiles: tuple[float, float] | None = None,
) -> str:
    """Publish summary/full analytical views of explicit owned components.

    Tables reuse the existing factor, regression, IC and persistence reducers.
    No component is rebuilt from a figure or from a scalar summary. Supplied
    forward outcomes remain mathematical research paths, not account P&L.
    """
    from ml4t.diagnostic.signal.event_study import _factor_statistics
    from ml4t.diagnostic.visualization.report_generation import combine_figures_to_html

    figures = plot_signal_report(
        returns_result,
        information_result,
        turnover_result,
        summary=summary,
        theme=theme,
        rolling_window=rolling_window,
        bandwidth=bandwidth,
        ylim_percentiles=ylim_percentiles,
    )
    spread = returns_result.get_table("spread_observations")
    spread_summary = spread.group_by("return_column", maintain_order=True).agg(
        pl.col("spread").mean().alias("mean_spread"),
        pl.col("spread").is_not_null().sum().alias("n_defined_dates"),
    )
    tables = [
        (
            "factor-statistics",
            "Factor statistics",
            _factor_statistics(returns_result.get_table("prepared_data"), factor_col),
        ),
        (
            "return-regression",
            "Return regression",
            returns_result.get_table("regression_statistics"),
        ),
        ("quantile-means", "Quantile means", returns_result.get_table("quantile_statistics")),
        ("observed-spreads", "Observed spread means", spread_summary),
        ("information-summary", "Information summary", information_result.get_table("summary")),
        ("turnover-summary", "Membership turnover", turnover_result.get_dataframe("turnover")),
        ("rank-summary", "Rank persistence", turnover_result.get_dataframe("autocorrelation")),
        (
            "group-quantile-means",
            "Group quantile means",
            returns_result.get_table("group_quantile_statistics"),
        ),
        ("group-information", "Group information", information_result.get_table("mean_group_ic")),
    ]
    sections = [
        {
            "title": "Methods and samples",
            "text": f"Return means and spreads are decimal fractions; IC, beta and wealth multiples are dimensionless. Annual alpha uses the retained explicit horizon scales. Return demeaning: {returns_result.metadata['demeaned']}; return group adjustment: {returns_result.metadata['group_adjust']}; information group adjustment: {information_result.metadata['group_adjust']}. IC inference is the naive two-sided one-sample t-test, not HAC. Quantile means give each defined date equal weight. Spreads retain {returns_result.metadata['spread_exposure']} exposure; bands are descriptive propagated errors, not confidence coverage. Cumulative paths compound raw supplied horizons and are not self-financing account P&L. Turnover is entrants/current membership; rank persistence uses full-date average ranks and paired Pearson correlation. Each recorded lag counts observations on its supplied/observed clock, not inferred sessions. Undefined observations and inference remain unavailable.",
        }
    ]
    for identifier, title, frame in tables:
        if frame.is_empty():
            continue
        table = frame.to_pandas().to_html(
            index=False,
            escape=True,
            na_rep="Undefined",
            border=0,
            table_id=identifier,
            float_format=lambda value: format_finite(value, ".6g"),
        )
        sections.append(
            {
                "title": title,
                "text": '<div style="overflow-x:auto;font-size:14px">' + table + "</div>",
            }
        )
    described = set()
    descriptions = {
        "returns": "Mean bars give each defined date equal weight. Distribution points are actual dated bucket means. Wealth paths compound raw supplied-horizon outcomes, not display-converted rates.",
        "information": "IC panels retain actual dated ICs and their defined group/month aggregates. Rolling means require complete22-observation windows; insufficient or degenerate samples have no inferred density or fitted quantiles.",
        "turnover": "Persistence panels retain every supplied observation lag and actual dated counts. Missing pairs are not zero turnover or rank correlation; no half-life or confidence interval is inferred.",
    }
    for index, figure in enumerate(figures):
        component = figure.layout.meta["report_component"]
        sections.append(
            {
                "title": f"{component.capitalize()} panel {index + 1}",
                "figure_index": index,
                "text": descriptions[component] if component not in described else "",
            }
        )
        described.add(component)
    html = combine_figures_to_html(
        figures,
        title="Signal summary" if summary else "Signal analysis",
        sections=sections,
        theme=theme,
        include_plotlyjs=True,
    )
    return _publish_event_html(html, output_file)
