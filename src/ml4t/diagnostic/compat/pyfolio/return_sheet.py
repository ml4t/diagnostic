"""Original returns manifest using the existing native report renderer."""

from __future__ import annotations

from html import escape
from numbers import Real
from textwrap import wrap
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from ml4t.diagnostic.evaluation.portfolio_analysis.inputs import positive_integer
from ml4t.diagnostic.signal.quantile import _sample_return_statistics

from .return_plotting import _return_series

if TYPE_CHECKING:
    import plotly.graph_objects as go

    from ml4t.diagnostic.visualization.portfolio import PortfolioTearSheet


def _table_figure(table: pd.DataFrame, title: str, description: str) -> go.Figure:
    import plotly.graph_objects as go

    from .plotting import _figure

    def cell(value: Any) -> str:
        if value is None or pd.isna(value):
            return "Unavailable"
        if isinstance(value, Real):
            return format(value, ".6g")
        return escape(str(value))

    headers = table.attrs.get("diagnostic", {}).get("header_rows", {})
    rows = [[cell(label), *[cell(value) for value in row]] for label, row in table.iterrows()]
    rows = [
        [cell(key), cell(value), *[""] * (len(table.columns) - 1)] for key, value in headers.items()
    ] + rows
    figure = _figure(title, "", None, {"height": max(360, 150 + len(rows) * 30)})
    header_height = 48 if isinstance(table.columns, pd.MultiIndex) else 32
    figure.update_layout(
        width=240 + 150 * len(table.columns),
        autosize=False,
        height=max(360, 60 + header_height + len(rows) * 30),
        margin={"l": 20, "r": 20, "t": 20, "b": 20},
    )
    figure.add_trace(
        go.Table(
            columnwidth=[200, *[150] * len(table.columns)],
            header={
                "values": [
                    cell(table.index.name) if table.index.name is not None else "Metric",
                    *[
                        "<br>".join(cell(part) for part in column)
                        if isinstance(column, tuple)
                        else cell(column)
                        for column in table.columns
                    ],
                ],
                "font_size": 14,
                "height": header_height,
            },
            cells={
                "values": list(zip(*rows, strict=True))
                if rows
                else [[] for _ in range(len(table.columns) + 1)],
                "font_size": 14,
                "height": 30,
                "align": "left",
            },
        )
    )
    figure.update_layout(
        meta={
            "analysis": "legacy_returns",
            "description": description
            + " Tables scroll horizontally on narrow screens; numeric display uses six significant digits unless a percentage format is specified.",
            **table.attrs.get("diagnostic", {}),
        }
    )
    return figure


def create_returns_tear_sheet(
    returns: pd.Series,
    positions: pd.DataFrame | None = None,
    transactions: pd.DataFrame | None = None,
    live_start_date: Any = None,
    cone_std: float | tuple[float, ...] | None = (1.0, 1.5, 2.0),
    benchmark_rets: pd.Series | None = None,
    bootstrap: bool = False,
    turnover_denom: str = "AGB",
    header_rows: dict[str, Any] | None = None,
    return_fig: bool = False,
    *,
    periods_per_year: int = 252,
    num_samples: int = 1000,
    n_samples: int = 1000,
    random_seed: int | None = None,
) -> PortfolioTearSheet | None:
    """Compose all original returns panels as a native Plotly PortfolioTearSheet.

    return_fig=True returns the owned report; otherwise display it and return
    None. Scale counts return observations, not rebalance dates. Native summary
    defaults stay unchanged. Missing returns keep each metric's own convention.
    Matplotlib axes/style contexts are presentation migration, not inputs here.
    """
    from ml4t.diagnostic.visualization.portfolio import PortfolioTearSheet

    from . import plotting, timeseries

    if not all(isinstance(flag, bool | np.bool_) for flag in (bootstrap, return_fig)):
        raise TypeError("bootstrap and return_fig must be boolean")
    scale = positive_integer(periods_per_year, "periods_per_year")
    if bootstrap and benchmark_rets is None:
        raise ValueError("bootstrap report requires benchmark_rets")
    supplied = _return_series(returns)
    benchmark = None if benchmark_rets is None else _return_series(benchmark_rets)
    original_count = len(supplied)
    if benchmark is not None:
        supplied = supplied.loc[
            (supplied.index >= benchmark.index[0]) & (supplied.index <= benchmark.index[-1])
        ]
        if supplied.empty:
            raise ValueError("returns and benchmark dates do not overlap")
        if not supplied.index.isin(benchmark.index).all():
            raise ValueError(
                "benchmark_rets must cover every return observation in the selected range"
            )
    stats = plotting.show_perf_stats(
        supplied,
        factor_returns=benchmark,
        positions=positions,
        transactions=transactions,
        live_start_date=live_start_date,
        bootstrap=bootstrap,
        turnover_denom=turnover_denom,
        header_rows=header_rows,
        return_df=True,
        periods_per_year=scale,
        n_samples=n_samples,
        random_seed=random_seed,
    )
    figures = {
        "Performance statistics": _table_figure(
            stats,
            "Performance statistics",
            "Familiar metric definitions, percentage rows and actual live-split statistics. Undefined values stay unavailable; header month counts use 21 observations as in Pyfolio, not exchange-calendar durations.",
        ),
        "Top drawdowns": _table_figure(
            timeseries.gen_drawdown_table(supplied, top=5),
            "Top drawdowns",
            "Shared native drawdowns include initial capital. Depth is percent; duration uses inclusive Monday-Friday civil dates only when both endpoints are known. An unknown initial peak date is not fabricated.",
        ),
    }
    options = {"factor_returns": benchmark, "live_start_date": live_start_date}
    match_status = {
        "status": "not_requested",
        "reason": "Supply benchmark returns to match sample volatility.",
    }
    if benchmark is not None:
        strategy_std = _sample_return_statistics(supplied.dropna().to_numpy(dtype=float))[1]
        benchmark_std = _sample_return_statistics(
            benchmark.reindex(supplied.index).dropna().to_numpy(dtype=float)
        )[1]
        match_status = (
            {"status": "available"}
            if np.isfinite(strategy_std) and strategy_std > 0 and np.isfinite(benchmark_std)
            else {
                "status": "unavailable",
                "reason": "Volatility matching requires defined dispersion and a nonconstant strategy; the supplied sample does not establish it.",
            }
        )
    if match_status["status"] == "available":
        matched = plotting.plot_rolling_returns(
            supplied, **options, volatility_match=True, legend_loc=None
        )
    else:
        matched = plotting._figure("Volatility matching", "Wealth multiple", None, {})
        matched.update_xaxes(visible=False)
        matched.update_yaxes(visible=False)
        matched.add_annotation(
            text="<br>".join(wrap(match_status["reason"], width=28)),
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
            font_size=14,
        )
        matched.update_layout(
            meta={"description": match_status["reason"], "status": match_status["status"]}
        )
    figures.update(
        {
            "Cumulative wealth": plotting.plot_rolling_returns(
                supplied,
                **options,
                cone_std=cone_std,
                num_samples=num_samples,
                random_seed=random_seed,
            ),
            "Volatility-matched wealth": matched,
            "Log wealth": plotting.plot_rolling_returns(
                supplied,
                **options,
                cone_std=cone_std,
                logy=True,
                num_samples=num_samples,
                random_seed=random_seed,
            ),
            "Observed returns": plotting.plot_returns(supplied, live_start_date=live_start_date),
        }
    )
    if benchmark is not None:
        figures["Rolling beta"] = plotting.plot_rolling_beta(supplied, benchmark)
    figures.update(
        {
            "Rolling volatility": plotting.plot_rolling_volatility(
                supplied, factor_returns=benchmark, periods_per_year=scale
            ),
            "Rolling Sharpe": plotting.plot_rolling_sharpe(supplied, periods_per_year=scale),
            "Drawdown periods": plotting.plot_drawdown_periods(supplied, top=5),
            "Underwater": plotting.plot_drawdown_underwater(supplied),
            "Monthly returns": plotting.plot_monthly_returns_heatmap(supplied),
            "Annual returns": plotting.plot_annual_returns(supplied),
            "Monthly distribution": plotting.plot_monthly_returns_dist(supplied),
            "Return quantiles": plotting.plot_return_quantiles(
                supplied, live_start_date=live_start_date
            ),
        }
    )
    if bootstrap:
        figures["Bootstrap distributions"] = plotting.plot_perf_stats(
            supplied,
            benchmark.reindex(supplied.index),
            periods_per_year=scale,
            n_samples=n_samples,
            random_seed=random_seed,
        )
    report = PortfolioTearSheet(
        metrics=None,
        figures=figures,
        availability={
            "returns": {
                "status": "available",
                "reason": "Original raw return-report profile over existing native kernels; not reconstructed account returns.",
                "n_observations": len(supplied),
                "n_missing": int(supplied.isna().sum()),
                "benchmark_range_exclusions": original_count - len(supplied),
                "periods_per_year": scale,
            },
            "benchmark": {"status": "not_requested" if benchmark is None else "available"},
            "bootstrap": {"status": "available" if bootstrap else "not_requested"},
            "volatility_match": match_status,
        },
    )
    if return_fig:
        return report
    report.show()
    return None
