"""Original dated stress report content composed through the existing HTML renderer."""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from datetime import date, datetime
from html import escape
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from ml4t.diagnostic.evaluation.portfolio_analysis.metrics import _safe_cumprod
from ml4t.diagnostic.evaluation.portfolio_analysis.regimes import _PERIOD_DATES

if TYPE_CHECKING:
    import plotly.graph_objects as go

_LOCATIONS = [
    "best",
    "upper right",
    "upper left",
    "lower left",
    "lower right",
    "right",
    "center left",
    "center right",
    "lower center",
    "upper center",
    "center",
]


def _prepare_interesting_times_tear_sheet(
    returns: pd.Series,
    benchmark_rets: pd.Series | None = None,
    periods: Mapping[str, tuple[date | datetime | str, date | datetime | str]] | None = None,
    legend_loc: str | int = "best",
    *,
    theme: str | None = None,
) -> tuple[list[go.Figure], list[dict[str, Any]], dict[str, pd.DataFrame]] | None:
    """Prepare original stress membership, tables and paths once for both views."""
    from ml4t.diagnostic.visualization.portfolio.regime_plots import _style

    from . import plotting, timeseries

    location = (
        _LOCATIONS[legend_loc]
        if isinstance(legend_loc, int) and 0 <= legend_loc < len(_LOCATIONS)
        else legend_loc
    )
    if location not in _LOCATIONS:
        raise ValueError("legend_loc must be a supported Matplotlib location name or0..10")
    samples = timeseries.extract_interesting_date_ranges(returns, periods)
    if not samples:
        warnings.warn(
            "Passed returns do not overlap with any interesting times.", UserWarning, stacklevel=2
        )
        return None
    benchmarks = (
        {}
        if benchmark_rets is None
        else timeseries.extract_interesting_date_ranges(benchmark_rets, periods)
    )
    statistics = pd.DataFrame(samples).describe().transpose()[["mean", "min", "max"]] * 100
    requested = list(_PERIOD_DATES if periods is None else periods)
    coverage = pd.DataFrame(
        [
            {
                "period": name,
                "n_observations": len(samples[name]) if name in samples else 0,
                "n_finite": int(np.isfinite(samples[name]).sum()) if name in samples else 0,
                "status": "observed" if name in samples else "unobserved",
            }
            for name in requested
        ]
    )
    figures, sections = [], []
    summary_html = (
        "<h3>Stress Events</h3><p>Mean, minimum and maximum in percent; no annualization. "
        "Missing observations are not zero returns. Only path compounding treats an absent return as a neutral factor.</p>"
        '<div style="overflow-x:auto;font-size:14px">'
        + statistics.to_html(float_format=lambda value: f"{value:.2f}%")
        + coverage.to_html(index=False)
        + "</div>"
    )
    for name, sample in samples.items():
        figure = plotting._figure(str(name), "Cumulative net return", None, {"theme": theme})
        for label, values in (("Algo", sample), ("benchmark", benchmarks.get(name))):
            if values is None:
                continue
            exact = values.index.astype(str).to_list()
            figure.add_scatter(
                x=exact,
                y=(_safe_cumprod(1 + values.to_numpy(dtype=float)) - 1).tolist(),
                customdata=exact,
                name=label,
                mode="lines+markers",
                hovertemplate="%{customdata}: %{y:.6g}<extra>%{fullData.name}</extra>",
            )
        exact = sample.index.astype(str).to_list()
        selected = sorted({0, len(exact) // 2, len(exact) - 1})
        figure.update_xaxes(
            type="category",
            tickmode="array",
            tickvals=[exact[i] for i in selected],
            ticktext=[sample.index[i].strftime("%b %d") for i in selected],
        )
        figure.update_yaxes(tickformat=".1%")
        figure.update_layout(
            meta={
                "analysis": "legacy_stress_events",
                "n_observations": len(sample),
                "n_finite": int(np.isfinite(sample).sum()),
                "legend_loc": legend_loc,
                "statistics_percent": statistics.loc[name].to_dict(),
                "benchmark_status": "not_requested"
                if benchmark_rets is None
                else "observed"
                if name in benchmarks
                else "unobserved",
                "description": "Actual selected event observations and shared cumulative-return products, with independent benchmark membership. Missing observations are retained, not zero returns. Percent statistics omit missing values; path compounding uses neutral missing factors. Explicit periods freeze event boundaries; default open-ended Covid ends at call time, not module import.",
            }
        )
        _style(figure, 360, None)
        if location != "best":
            x = (
                0
                if "left" in location
                else 1
                if "right" in location or location == "right"
                else 0.5
            )
            y = 1 if "upper" in location else 0 if "lower" in location else 0.5
            figure.update_layout(
                legend={
                    "orientation": "v",
                    "x": x,
                    "y": y,
                    "xanchor": "left" if x == 0 else "right" if x == 1 else "center",
                    "yanchor": "top" if y == 1 else "bottom" if y == 0 else "middle",
                }
            )
        sections.append(
            {
                "title": escape(str(name)),
                "text": summary_html if not sections else "",
                "figure_index": len(figures),
            }
        )
        figures.append(figure)
    return figures, sections, {"Stress events (%)": statistics, "Stress coverage": coverage}


def create_interesting_times_tear_sheet(
    returns: pd.Series,
    benchmark_rets: pd.Series | None = None,
    periods: Mapping[str, tuple[date | datetime | str, date | datetime | str]] | None = None,
    legend_loc: str | int = "best",
    return_fig: bool = False,
    *,
    output_file: str | Path | None = None,
    theme: str | None = None,
) -> str | list[go.Figure] | None:
    """Render original percent statistics and event paths through the existing renderer.

    return_fig=True returns per-event Plotly figures. Explicit output_file saves
    all tables in either mode. Metric-specific missingness and neutral missing
    compounding remain unchanged; absent windows are not observed zero returns.
    """
    from ml4t.diagnostic.visualization.report_generation import combine_figures_to_html

    if not isinstance(return_fig, bool | np.bool_):
        raise TypeError("return_fig must be a boolean")
    prepared = _prepare_interesting_times_tear_sheet(
        returns, benchmark_rets, periods, legend_loc, theme=theme
    )
    if prepared is None:
        return None
    figures, sections, _ = prepared
    html = combine_figures_to_html(
        figures,
        title="Interesting times stress analysis",
        sections=sections,
        output_file=output_file,
        theme=theme,
        include_plotlyjs=True,
    )
    return figures if return_fig else html
