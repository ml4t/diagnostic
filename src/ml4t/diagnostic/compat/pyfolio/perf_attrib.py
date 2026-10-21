"""Pyfolio supplied-holdings tables over one native attribution calculation.

Dollar books retain their complete account denominator when loadings are
missing. The published reference drops uncovered assets before normalization;
that changes covered NAV weights and is deliberately not reproduced here.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import polars as pl

from ml4t.diagnostic.compat import empyrical as ep
from ml4t.diagnostic.evaluation.portfolio_analysis.inputs import positive_integer
from ml4t.diagnostic.evaluation.portfolio_analysis.transaction_metrics import cash_fraction_expr

from .txn import _books, _dated, get_turnover

if TYPE_CHECKING:
    import plotly.graph_objects as go

PERF_ATTRIB_TURNOVER_THRESHOLD = 0.25


def _stack_positions(positions: pd.DataFrame, pos_in_dollars: bool = True) -> pd.Series:
    if not isinstance(positions, pd.DataFrame):
        raise TypeError("positions must be a dated wide DataFrame")
    _dated(positions, "positions")
    if "cash" not in positions.columns:
        raise ValueError("positions require an explicit account cash column")
    _, nav = _books(positions)
    assets = positions.drop(columns="cash").copy(deep=True)
    if pos_in_dollars:
        for column in assets:
            frame = pl.DataFrame({"value": assets[column].to_numpy(), "nav": nav.to_numpy()})
            assets[column] = (
                frame.select(
                    cash_fraction_expr(pl.col("value"), pl.col("nav"), positive_finite=False)
                )
                .to_series()
                .to_numpy()
            )
    weights = assets.stack(future_stack=True).dropna()
    weights.index = weights.index.set_names(["dt", "ticker"])
    return weights


def _align_and_warn(
    returns: pd.Series,
    positions: pd.DataFrame,
    factor_returns: pd.DataFrame,
    factor_loadings: pd.DataFrame,
    transactions: pd.DataFrame | None = None,
    pos_in_dollars: bool = True,
) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    supplied = tuple(
        item.copy(deep=True) for item in (returns, positions, factor_returns, factor_loadings)
    )
    returns, positions, factor_returns, factor_loadings = supplied
    _dated(returns, "returns")
    _dated(positions, "positions")
    _dated(factor_returns, "factor_returns")
    if not isinstance(factor_loadings.index, pd.MultiIndex) or factor_loadings.index.nlevels != 2:
        raise TypeError("factor_loadings requires a date/asset MultiIndex")
    factor_dates = factor_loadings.index.get_level_values(0)
    if not isinstance(factor_dates, pd.DatetimeIndex):
        raise TypeError("factor loading observations require datetime labels")
    if (
        returns.index.tz != positions.index.tz
        or returns.index.tz != factor_dates.tz
        or returns.index.tz != factor_returns.index.tz
    ):
        raise ValueError("portfolio, position and factor observations must use the same timezone")
    missing_dates = positions.index.difference(factor_dates.unique())
    if len(missing_dates):
        warnings.warn(
            f"Factor loadings unavailable for {len(missing_dates)} observed dates; "
            "the familiar report excludes these dates, without imputing loadings.",
            UserWarning,
            stacklevel=3,
        )
        returns = returns.drop(missing_dates, errors="ignore")
        positions = positions.drop(missing_dates, errors="ignore")
        factor_returns = factor_returns.drop(missing_dates, errors="ignore")
    if transactions is not None and pos_in_dollars:
        turnover = get_turnover(positions, transactions).mean()
        if turnover > PERF_ATTRIB_TURNOVER_THRESHOLD:
            warnings.warn(
                "High turnover: end-of-observation holdings do not identify intraday factor exposure.",
                UserWarning,
                stacklevel=3,
            )
    return returns, positions, factor_returns, factor_loadings


def perf_attrib(
    returns: pd.Series,
    positions: pd.DataFrame,
    factor_returns: pd.DataFrame,
    factor_loadings: pd.DataFrame,
    transactions: pd.DataFrame | None = None,
    pos_in_dollars: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Attribute observed holdings; uncovered assets never change the account NAV.

    This familiar route preserves observed-label timing, not beginning-period
    availability. Missing loading cells retain the native adapter's warning;
    complete native inputs use strict coverage/known-at validation instead.
    """
    returns, positions, factor_returns, factor_loadings = _align_and_warn(
        returns,
        positions,
        factor_returns,
        factor_loadings,
        transactions=transactions,
        pos_in_dollars=pos_in_dollars,
    )
    return ep.perf_attrib(
        returns, _stack_positions(positions, pos_in_dollars), factor_returns, factor_loadings
    )


def compute_exposures(
    positions: pd.DataFrame | pd.Series,
    factor_loadings: pd.DataFrame,
    stack_positions: bool = True,
    pos_in_dollars: bool = True,
) -> pd.DataFrame:
    """Project supplied loadings without fitting betas or mutating source labels."""
    weights = _stack_positions(positions, pos_in_dollars) if stack_positions else positions
    return ep.compute_exposures(weights, factor_loadings)


def create_perf_attrib_stats(
    perf_attrib: pd.DataFrame,
    risk_exposures: pd.DataFrame,
    *,
    periods_per_year: int = 252,
) -> tuple[pd.Series, pd.DataFrame]:
    """Keep the familiar component-compounded statistics, with an explicit scale.

    Independently compounded component returns are descriptive growth paths,
    not an additive wealth-linked decomposition of the total portfolio return.
    """
    scale = positive_integer(periods_per_year, "periods_per_year")
    total, specific, common = (
        perf_attrib[name] for name in ("total_returns", "specific_returns", "common_returns")
    )
    summary = pd.Series(
        {
            "Annualized Specific Return": ep.annual_return(specific, annualization=scale),
            "Annualized Common Return": ep.annual_return(common, annualization=scale),
            "Annualized Total Return": ep.annual_return(total, annualization=scale),
            "Specific Sharpe Ratio": ep.sharpe_ratio(specific, annualization=scale),
            "Cumulative Specific Return": ep.cum_returns_final(specific),
            "Cumulative Common Return": ep.cum_returns_final(common),
            "Total Returns": ep.cum_returns_final(total),
        },
        name="",
    )
    factors = pd.DataFrame(
        {
            "Average Risk Factor Exposure": risk_exposures.mean(axis="rows"),
            "Annualized Return": [
                ep.annual_return(perf_attrib[name], annualization=scale)
                for name in risk_exposures.columns
            ],
            "Cumulative Return": [
                ep.cum_returns_final(perf_attrib[name]) for name in risk_exposures.columns
            ],
        },
        index=risk_exposures.columns,
    )
    return summary, factors


def show_perf_attrib_stats(
    returns: pd.Series,
    positions: pd.DataFrame,
    factor_returns: pd.DataFrame,
    factor_loadings: pd.DataFrame,
    transactions: pd.DataFrame | None = None,
    pos_in_dollars: bool = True,
    *,
    periods_per_year: int = 252,
) -> tuple[pd.Series, pd.DataFrame]:
    """Print and return the actual summary tables without requiring IPython."""
    exposures, table = perf_attrib(
        returns, positions, factor_returns, factor_loadings, transactions, pos_in_dollars
    )
    summary, factors = create_perf_attrib_stats(table, exposures, periods_per_year=periods_per_year)
    print(summary.to_string())
    print(factors.to_string())
    return summary, factors


def _plot_lines(
    paths: list[tuple[str, pd.Series]],
    title: str,
    units: str,
    ax: Any,
    theme: str | None,
    height: int,
) -> go.Figure:
    import plotly.graph_objects as go

    from .plotting import _date_ticks, _figure

    figure = _figure(title, units, ax, {"theme": theme, "height": height})
    for name, values in paths:
        _dated(values, name)
        figure.add_trace(
            go.Scatter(
                x=values.index,
                y=values.to_numpy(dtype=float, na_value=np.nan),
                name=name,
                mode="lines+markers" if len(values) < 5 else "lines",
            )
        )
    if paths:
        _date_ticks(figure, paths[0][1].index)
    figure.update_layout(
        title_font_size=14,
        meta={
            "analysis": "supplied_holdings_attribution",
            "cumulative_method": "component_compounded",
            "interpretation": "Independently compounded component paths are not an additive wealth-linked decomposition.",
        },
    )
    return figure


def plot_returns(
    perf_attrib_data: pd.DataFrame,
    cost: pd.Series | None = None,
    ax: Any = None,
    *,
    theme: str | None = None,
    height: int = 420,
) -> go.Figure:
    """Plot component growth; explicit periodic costs adjust gross total returns once."""
    total = perf_attrib_data["total_returns"]
    paths = [
        (
            "Total returns" + (" (adjusted)" if cost is not None else ""),
            ep.cum_returns(total if cost is None else total - cost),
        ),
        ("Cumulative specific returns", ep.cum_returns(perf_attrib_data["specific_returns"])),
        ("Cumulative common returns", ep.cum_returns(perf_attrib_data["common_returns"])),
    ]
    if cost is not None:
        paths.append(("Cumulative cost spent", -ep.cum_returns(cost)))
    return _plot_lines(paths, "Time series of cumulative returns", "Returns", ax, theme, height)


def plot_factor_contribution_to_perf(
    perf_attrib_data: pd.DataFrame,
    ax: Any = None,
    title: str = "Cumulative common returns attribution",
    *,
    theme: str | None = None,
    height: int = 420,
) -> go.Figure:
    """Preserve specific returns alongside factors, excluding named aggregate paths."""
    components = perf_attrib_data.drop(
        columns=["total_returns", "common_returns", "tilt_returns", "timing_returns"],
        errors="ignore",
    )
    figure = _plot_lines(
        [(str(name), ep.cum_returns(components[name])) for name in components],
        title,
        "Cumulative returns by factor",
        ax,
        theme,
        height,
    )
    figure.add_hline(y=0, line_color="gray")
    return figure


def plot_risk_exposures(
    exposures: pd.DataFrame,
    ax: Any = None,
    title: str = "Daily risk factor exposures",
    *,
    theme: str | None = None,
    height: int = 420,
) -> go.Figure:
    """Render supplied exposures without fitting or compounding their loading values."""
    figure = _plot_lines(
        [(str(name), exposures[name]) for name in exposures],
        title,
        "Factor exposures",
        ax,
        theme,
        height,
    )
    figure.update_layout(
        meta={
            "analysis": "supplied_holdings_exposures",
            "interpretation": "NAV-weighted supplied loadings, not estimated regression betas.",
        }
    )
    return figure


def plot_alpha_returns(
    alpha_returns: pd.Series,
    ax: Any = None,
    *,
    theme: str | None = None,
    height: int = 420,
) -> go.Figure:
    """Render ten explicit observed-value bins; count excluded nonfinite values."""
    import plotly.graph_objects as go

    from .plotting import _figure

    values = alpha_returns.to_numpy(dtype=float, na_value=np.nan)
    finite = values[np.isfinite(values)]
    counts, edges = np.histogram(finite, bins=10)
    figure = _figure("Histogram of alphas", "Observations", ax, {"theme": theme, "height": height})
    figure.add_trace(
        go.Bar(
            x=(edges[1:] + edges[:-1]) / 2,
            y=counts,
            width=np.diff(edges),
            name="Multi-factor alpha",
        )
    )
    figure.add_vline(x=0, line_dash="dash", annotation_text="Zero")
    if len(finite):
        mean = float(np.mean(finite))
        figure.add_vline(x=mean, annotation_text=f"Mean = {mean: 0.5f}")
    figure.update_xaxes(title_text="Periodic specific return", tickformat=".2%")
    figure.update_layout(
        title_font_size=14,
        meta={
            "analysis": "supplied_holdings_specific_returns",
            "n_observations": len(values),
            "excluded_nonfinite": int(len(values) - len(finite)),
            "bin_edges": edges.tolist(),
        },
    )
    return figure
