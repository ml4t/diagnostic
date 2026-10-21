"""Familiar round-trip views using native cash summaries and the shared renderer."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
import polars as pl

if TYPE_CHECKING:
    import plotly.graph_objects as go


def _figure(ax: go.Figure | None, x: str, y: str) -> go.Figure:
    import plotly.graph_objects as go

    from ml4t.diagnostic.visualization.core import create_base_figure

    if ax is not None and not isinstance(ax, go.Figure):
        raise TypeError("ax must be a Plotly Figure, not a Matplotlib Axes")
    figure = ax if ax is not None else create_base_figure(xaxis_title=x, yaxis_title=y, height=360)
    figure.update_layout(font_size=14, margin={"l": 65, "r": 25, "t": 30, "b": 115})
    figure.update_layout(meta={**(figure.layout.meta or {}), "analysis": "prepared_round_trips"})
    figure.update_xaxes(tickfont_size=14, title_font_size=14)
    figure.update_yaxes(tickfont_size=14, title_font_size=14)
    return figure


def _observations(frame: pd.DataFrame, required: set[str]) -> tuple[pl.DataFrame, pd.Index]:
    from .round_trips import _helper_frame

    frame = _helper_frame(frame, required)
    codes, labels = pd.factorize(frame["symbol"], sort=False)
    if (codes < 0).any():
        raise ValueError("round-trip asset IDs must be known")
    output = pd.DataFrame({"asset": codes})
    for original, native in (
        ("pnl", "gross_pnl"),
        ("open_dt", "entry_timestamp"),
        ("close_dt", "exit_timestamp"),
    ):
        if original in frame:
            output[native] = frame[original].reset_index(drop=True)
    if "long" in frame:
        if frame["long"].isna().any() or not frame["long"].isin([True, False]).all():
            raise ValueError("lifetime direction must be a known boolean")
        output["direction"] = ["long" if value else "short" for value in frame["long"]]
    output["is_synthetic"] = frame["is_synthetic"].to_list() if "is_synthetic" in frame else False
    return pl.from_pandas(output), labels


def plot_round_trip_lifetimes(
    round_trips: pd.DataFrame, disp_amount: int = 16, lsize: float = 18, ax: go.Figure | None = None
) -> go.Figure:
    """Plot the original deterministic asset sample without resetting global RNG.

    Returns/appends to Plotly rather than Matplotlib. Exact timestamps remain on
    hover; browser date axes display milliseconds. Statistics are not sampled.
    """
    from ml4t.diagnostic.visualization.portfolio.round_trip_plots import _draw_lifetimes

    observations, labels = _observations(round_trips, {"symbol", "open_dt", "close_dt", "long"})
    for name in ("entry_timestamp", "exit_timestamp"):
        if not isinstance(observations[name].dtype, pl.Datetime):
            raise TypeError("lifetimes require actual datetime instants, not bare integer times")
    if observations.filter(pl.col("exit_timestamp") < pl.col("entry_timestamp")).height:
        raise ValueError("closing timestamps must not precede entry timestamps")
    figure = _figure(ax, "Time", "Asset")
    _draw_lifetimes(
        figure,
        observations,
        disp_amount=disp_amount,
        lsize=lsize,
        asset_labels={
            index: str(getattr(label, "symbol", label)) for index, label in enumerate(labels)
        },
    )
    return figure


def plot_prob_profit_trade(round_trips: pd.DataFrame, ax: go.Figure | None = None) -> go.Figure:
    """Plot the explicit legacy Beta(wins, nonwins), excluding unknown cash.

    One-sided/empty samples have no proper zero-prior posterior. They show an
    unavailable note instead of the reference's NaN-density/axis crash. Native
    RoundTripResult.win_probability retains its separate Beta(1,1) default.
    """
    from ml4t.diagnostic.evaluation.portfolio_analysis.fifo_summary import summarize_win_probability
    from ml4t.diagnostic.visualization.portfolio.round_trip_plots import _draw_probability

    from .round_trips import _helper_frame

    frame = _helper_frame(round_trips, {"pnl"})
    posterior = summarize_win_probability(pl.from_pandas(frame["pnl"]), method="legacy_zero_prior")
    figure = _figure(ax, "P(positive cash PnL)", "Density")
    _draw_probability(figure, posterior, points=500, legacy_limits=True)
    return figure


def _attribution(round_trips: pd.DataFrame) -> tuple[pl.DataFrame, pd.Index]:
    from ml4t.diagnostic.evaluation.portfolio_analysis.fifo_statistics import (
        summarize_profit_attribution,
    )

    observations, labels = _observations(round_trips, {"symbol", "pnl"})
    return summarize_profit_attribution(
        observations, pnl_col="gross_pnl", missing="exclude"
    ), labels


def show_profit_attribution(round_trips: pd.DataFrame) -> None:
    """Print native signed cash shares, not a second financial calculation."""
    native, labels = _attribution(round_trips)
    pairs = [
        (
            str(getattr(labels[int(row["asset"])], "symbol", labels[int(row["asset"])])),
            row["pnl_fraction"],
        )
        for row in native.to_dicts()
    ]
    result = pd.Series(
        [value for _, value in pairs], index=[label for label, _ in pairs], dtype=float
    ).sort_values(ascending=False)
    print("Profitability (PnL / PnL total) per name")
    print(result.to_string(float_format=lambda value: f"{value:.2%}", na_rep="Unavailable"))


__all__ = ["plot_round_trip_lifetimes", "plot_prob_profit_trade", "show_profit_attribution"]
