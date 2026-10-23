"""Familiar position panels using shared holdings kernels and the native figure factory."""

from __future__ import annotations

from html import escape
from numbers import Integral
from textwrap import wrap
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import polars as pl

from ml4t.diagnostic.evaluation.portfolio_analysis.transaction_metrics import cash_fraction_expr

from . import pos, timeseries
from .return_plotting import _return_series, _set_legend
from .txn import _books, _dated

if TYPE_CHECKING:
    import plotly.graph_objects as go


def _positions(values: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(values, pd.DataFrame):
        raise TypeError("positions must be a pandas DataFrame")
    _dated(values, "positions")
    if values.empty or not values.index.is_monotonic_increasing:
        raise ValueError("position observations must be nonempty and chronological")
    if "cash" not in values:
        raise ValueError("positions require separate account cash")
    pos._frame(values, cash=True)
    return values.copy(deep=True)


def _display_mode(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value not in (0, 1, 2):
        raise ValueError("show_and_plot must be 0 (plot), 1 (table), or 2 (both)")
    return int(value)


def _hidden(values: pd.DataFrame) -> pd.DataFrame:
    result = values.copy(deep=True)
    result.columns = [
        "cash" if label == "cash" else f"Holding {i + 1}" for i, label in enumerate(values.columns)
    ]
    result.attrs = {}
    return result


def _lines(
    returns: pd.Series | pd.DatetimeIndex,
    samples: list[tuple[str, pd.Series]],
    title: str,
    units: str,
    ax: Any,
    kwargs: dict[str, Any],
    *,
    legend_loc: str | int | None = "best",
    fill: bool = False,
    description: str,
) -> go.Figure:
    import plotly.graph_objects as go

    from .plotting import _figure

    clock = returns if isinstance(returns, pd.DatetimeIndex) else _return_series(returns).index
    figure = _figure(title, units, ax, kwargs)
    if kwargs:
        raise TypeError(f"Unknown Plotly layout options: {', '.join(sorted(kwargs))}")
    _set_legend(figure, legend_loc)
    keys = {stamp.isoformat(): stamp for stamp in clock}
    nonfinite = 0
    finite = 0
    for number, (name, series) in enumerate(samples):
        exact = [stamp.isoformat() for stamp in series.index]
        labels = [
            f"{stamp} [observation {row + 1}]" if series.index.has_duplicates else stamp
            for row, stamp in enumerate(exact)
        ]
        keys.update(zip(labels, series.index, strict=True))
        values = series.to_numpy(dtype=float)
        nonfinite += int((~np.isfinite(values)).sum())
        finite += int(np.isfinite(values).sum())
        figure.add_trace(
            go.Scatter(
                x=labels,
                y=values.tolist(),
                customdata=exact,
                name="<br>".join(escape(line) for line in wrap(str(name), width=28)),
                mode="lines",
                connectgaps=False,
                fill="tozeroy" if fill and number < 2 else None,
                hovertemplate="%{customdata}<br>%{y}<extra>%{fullData.name}</extra>",
            )
        )
    ordered = sorted(keys, key=keys.__getitem__)
    visible = [i for i, label in enumerate(ordered) if clock[0] <= keys[label] <= clock[-1]]
    ticks = sorted({visible[0], visible[len(visible) // 2], visible[-1]})
    figure.update_xaxes(
        type="category",
        categoryorder="array",
        categoryarray=ordered,
        range=[visible[0] - 0.5, visible[-1] + 0.5],
        tickmode="array",
        tickvals=[ordered[i] for i in ticks],
        ticktext=[keys[ordered[i]].strftime("%b %d") for i in ticks],
        tickangle=0,
        title_text="Observed key (exact time on hover)",
    )
    figure.update_layout(
        meta={
            "analysis": "legacy_positions",
            "units": units,
            "nonfinite_values": nonfinite,
            "clock_spacing": "chronological categorical observation and civil-month keys; not inferred exchange sessions",
            "description": description,
        }
    )
    if samples and not finite:
        figure.add_annotation(
            text="No finite sample values",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
            font_size=14,
        )
    return figure


def plot_exposures(
    returns: pd.Series, positions: pd.DataFrame, ax: Any = None, **kwargs: Any
) -> go.Figure:
    """Legacy long/short/net fractions using the skip-missing closing row sum."""
    values = _positions(positions)
    totals = pos._totals(values)
    _, nav = _books(values)
    totals = totals.with_columns(pl.Series("nav", nav.to_numpy())).select(
        cash_fraction_expr(pl.col("long_exposure"), pl.col("nav"), positive_finite=False).alias(
            "Long"
        ),
        cash_fraction_expr(-pl.col("short_exposure"), pl.col("nav"), positive_finite=False).alias(
            "Short"
        ),
        cash_fraction_expr(pl.col("net_exposure"), pl.col("nav"), positive_finite=False).alias(
            "Net"
        ),
    )
    return _lines(
        returns,
        [(name, pd.Series(totals[name].to_numpy(), index=values.index)) for name in totals.columns],
        "Exposure",
        "NAV fraction",
        ax,
        kwargs,
        fill=True,
        description="Long, signed short and net marked value divided by the closing row sum, including cash. As in Pyfolio plotting, that sum skips missing cash. Zero/negative NAV follows legacy IEEE division.",
    )


def plot_max_median_position_concentration(
    positions: pd.DataFrame, ax: Any = None, **kwargs: Any
) -> go.Figure:
    """Shared max/median signed allocation weights on the original position clock."""
    values = _positions(positions)
    summary = pos.get_max_median_position_concentration(values)
    clock = values.index.drop_duplicates()
    return _lines(
        clock,
        [(name, summary[name]) for name in summary],
        "Position concentration",
        "NAV fraction",
        ax,
        kwargs,
        description="Maximum and median positive/negative normalized allocation. Classification follows allocation sign at negative NAV; undefined sides remain gaps.",
    )


def plot_holdings(
    returns: pd.Series,
    positions: pd.DataFrame,
    legend_loc: str | int | None = "best",
    ax: Any = None,
    **kwargs: Any,
) -> go.Figure:
    """Observed nonzero/nonmissing counts and their civil-month/all-sample means."""
    values = _positions(positions)
    counts = pd.Series(pos._totals(values)["open_positions"].to_numpy(), index=values.index)
    monthly = counts.resample(pd.offsets.MonthEnd()).mean()
    figure = _lines(
        returns,
        [("Daily holdings", counts), ("Monthly mean", monthly)],
        "Holdings",
        "Position count",
        ax,
        kwargs,
        legend_loc=legend_loc,
        description="Counts use the supplied raw or allocation frame, as in the original caller. Zero and missing cells are not active holdings. Civil-month averages weight supplied rows, including explicitly retained duplicate candidates.",
    )
    figure.add_hline(y=float(counts.mean()), line_dash="dash")
    figure.update_layout(meta={**figure.layout.meta, "daily_mean": float(counts.mean())})
    return figure


def plot_long_short_holdings(
    returns: pd.Series,
    positions: pd.DataFrame,
    legend_loc: str | int | None = "upper left",
    ax: Any = None,
    **kwargs: Any,
) -> go.Figure:
    """Positive and negative nonmissing counts on the caller's input definition."""
    values = _positions(positions)
    counts = pos._totals(values)
    figure = _lines(
        returns,
        [
            (label, pd.Series(counts[column].to_numpy(), index=values.index))
            for label, column in (("Long", "long_positions"), ("Short", "short_positions"))
        ],
        "Long and short holdings",
        "Position count",
        ax,
        kwargs,
        fill=True,
        legend_loc=legend_loc,
        description="Separate positive/negative cell counts; both filled regions start at zero, so overlap does not add positions. The sheet passes allocations, retaining the original negative-NAV sign convention.",
    )
    figure.update_layout(
        meta={
            **figure.layout.meta,
            "long_min": counts["long_positions"].min(),
            "long_max": counts["long_positions"].max(),
            "short_min": counts["short_positions"].min(),
            "short_max": counts["short_positions"].max(),
        }
    )
    return figure


def plot_gross_leverage(
    returns: pd.Series, positions: pd.DataFrame, ax: Any = None, **kwargs: Any
) -> go.Figure:
    """Shared familiar gross book divided by signed closing NAV, plus its mean."""
    values = _positions(positions)
    leverage = timeseries.gross_lev(values)
    figure = _lines(
        returns,
        [("Gross leverage", leverage)],
        "Gross leverage",
        "Gross / NAV",
        ax,
        kwargs,
        description="Familiar gross marked book over signed closing row NAV. Zero/negative NAV is not converted to native positive-NAV normalization. The sample mean omits missing values but does not discard infinity.",
    )
    mean = float(leverage.mean())
    if np.isfinite(mean):
        figure.add_hline(y=mean, line_dash="dash")
    figure.update_layout(meta={**figure.layout.meta, "sample_mean": mean})
    return figure


def _top_tables(positions_alloc: pd.DataFrame) -> dict[str, go.Figure]:
    from .return_sheet import _table_figure

    tables = {}
    for side, values in zip(
        ("long", "short", "absolute"), pos.get_top_long_short_abs(positions_alloc), strict=True
    ):
        frame = pd.DataFrame({"Maximum (%)": values * 100})
        frame.attrs["diagnostic"] = {
            "analysis": "legacy_positions",
            "units": "percent of signed NAV",
        }
        frame.index = pd.Index(
            [getattr(value, "symbol", value) for value in frame.index], name="Asset"
        )
        title = f"Top {side} positions"
        tables[title] = _table_figure(
            frame,
            title,
            "Sample extrema of original per-asset NAV allocation, not time-weighted rankings or final holdings. Values are percent; short extrema remain signed.",
        )
    return tables


def _top_plot(
    returns: pd.Series,
    positions_alloc: pd.DataFrame,
    legend_loc: str | int | None,
    ax: Any,
    kwargs: dict[str, Any],
) -> go.Figure:
    top = pos.get_top_long_short_abs(positions_alloc)[2]
    figure = _lines(
        returns,
        [(str(getattr(label, "symbol", label)), positions_alloc[label]) for label in top.index],
        "Top positions",
        "NAV allocation",
        ax,
        kwargs,
        legend_loc="best" if legend_loc == "real_best" else legend_loc,
        description="Full allocation paths for up to ten assets ranked by their maximum absolute allocation over the supplied sample; ties retain input-column order. Gaps stay missing.",
    )
    lines = sum(
        max(1, len(wrap(str(getattr(label, "symbol", label)), width=28))) for label in top.index
    )
    if legend_loc == "real_best":
        bottom = max(145, 30 + lines * 26)
        figure.update_layout(
            margin_b=bottom,
            height=max(figure.layout.height, 300 + bottom),
            legend={"orientation": "h", "x": 0, "y": -0.3},
        )
    if top.empty:
        figure.add_annotation(
            text="No asset holdings to rank",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
            font_size=14,
        )
    return figure


def show_and_plot_top_positions(
    returns: pd.Series,
    positions_alloc: pd.DataFrame,
    show_and_plot: int = 2,
    hide_positions: bool = False,
    legend_loc: str | int | None = "real_best",
    ax: Any = None,
    **kwargs: Any,
) -> go.Figure | None:
    """Display selected top tables and/or return their Plotly allocation paths.

    Hidden identities are replaced before constructing any table or trace; hide
    also forces plot-only output, including for this standalone adapter.
    """
    from ml4t.diagnostic.visualization.portfolio import PortfolioTearSheet

    mode = _display_mode(show_and_plot)
    if not isinstance(hide_positions, bool | np.bool_):
        raise TypeError("hide_positions must be boolean")
    values = _positions(positions_alloc)
    if hide_positions:
        values, mode = _hidden(values), 0
    figure = _top_plot(returns, values, legend_loc, ax, kwargs)
    if mode in (1, 2):
        PortfolioTearSheet(metrics=None, figures=_top_tables(values)).show()
    return figure if mode in (0, 2) else None


def plot_sector_allocations(
    returns: pd.Series, sector_alloc: pd.DataFrame, ax: Any = None, **kwargs: Any
) -> go.Figure:
    """Caller-prepared sector fractions, not inferred full-NAV classification."""
    _dated(sector_alloc, "sector_alloc")
    figure = _lines(
        returns,
        [(str(label), sector_alloc[label]) for label in sector_alloc],
        "Sector allocations",
        "NAV allocation",
        ax,
        kwargs,
        description="Caller-prepared sector paths. The familiar sheet excludes unmapped assets and re-normalizes mapped sectors plus cash, unlike native full-NAV sector results. Account cash is not an asset sector.",
    )
    if not len(sector_alloc.columns):
        figure.add_annotation(
            text="No mapped risky holdings",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
            font_size=14,
        )
    return figure


__all__ = [
    "plot_exposures",
    "show_and_plot_top_positions",
    "plot_max_median_position_concentration",
    "plot_holdings",
    "plot_long_short_holdings",
    "plot_gross_leverage",
    "plot_sector_allocations",
]
