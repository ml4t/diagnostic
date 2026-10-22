"""Familiar supplied-cone redraw presentation over shared native return products."""

from __future__ import annotations

from collections.abc import Sequence
from numbers import Integral
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from ml4t.diagnostic.evaluation.portfolio_analysis.metrics import _safe_cumprod

if TYPE_CHECKING:
    import plotly.graph_objects as go


def plot_cones(
    name: str | None,
    bounds: pd.DataFrame,
    oos_returns: pd.Series,
    num_samples: int = 1000,
    ax: Any = None,
    cone_std: Sequence[float] = (1.0, 1.5, 2.0),
    random_seed: int | None = None,
    num_strikes: int = 3,
    *,
    theme: str | None = None,
) -> go.Figure:
    """Draw original lower-two-std crossings as retrospective shifted overlays.

    This route consumes already supplied bounds. As in Pyfolio, neither
    num_samples nor random_seed draws any new forecast. Each redraw shifts the
    same bounds additively from actual wealth at the crossing; it does not retrain.
    Returns/appends to Plotly instead of a Matplotlib Figure/Axes. Use native
    compute_wealth_cones for an owned training-only forecast and live overlay.
    """
    import plotly.graph_objects as go

    from ml4t.diagnostic.visualization.core import create_base_figure
    from ml4t.diagnostic.visualization.portfolio.regime_plots import _style

    from .txn import _dated

    _dated(oos_returns, "oos_returns")
    if not isinstance(oos_returns, pd.Series) or oos_returns.empty:
        raise ValueError("oos_returns must be a nonempty dated pandas Series")
    if not oos_returns.index.is_unique or not oos_returns.index.is_monotonic_increasing:
        raise ValueError("oos_returns must have unique chronological observations")
    if (
        isinstance(num_strikes, bool)
        or not isinstance(num_strikes, Integral)
        or not 0 <= num_strikes <= 3
    ):
        raise ValueError("num_strikes must be an integer from zero through three")
    if not isinstance(bounds, pd.DataFrame) or len(bounds) < len(oos_returns):
        raise ValueError("bounds must supply every live observation's forecast step")
    levels = list(cone_std)
    needed = {sign * float(level) for level in levels for sign in (1, -1)}
    if num_strikes:
        needed.add(-2.0)
    if not needed <= set(bounds.columns):
        raise ValueError(
            "bounds require both signs of every cone_std and -2 for legacy redraws; use num_strikes=0 to disable redraws"
        )
    if ax is not None and not isinstance(ax, go.Figure):
        raise TypeError("ax must be a Plotly Figure, not a Matplotlib Axes")
    figure = (
        ax
        if ax is not None
        else create_base_figure(
            title=name,
            xaxis_title="Actual observation",
            yaxis_title="Wealth (start 1)",
            theme=theme,
        )
    )
    wealth = _safe_cumprod(1 + oos_returns.to_numpy(dtype=float))
    start, shift = 0, 0.0
    starts = []
    colors = (
        "rgba(0,128,0,0.18)",
        "rgba(255,165,0,0.18)",
        "rgba(255,69,0,0.18)",
        "rgba(139,0,0,0.18)",
    )
    for strike in range(num_strikes + 1):
        if strike:
            previous_lower = bounds[-2.0].iloc[: len(wealth) - start].to_numpy(dtype=float) + shift
            crossings = np.flatnonzero(wealth[start:] < previous_lower)
            if not len(crossings):
                break
            start += int(crossings[0])
            shift = float(wealth[start] - 1)
        starts.append(start)
        dates = oos_returns.index[start:].astype(str).to_list()
        for level in levels:
            for sign in (-1, 1):
                values = (
                    bounds[sign * float(level)].iloc[: len(dates)].to_numpy(dtype=float) + shift
                )
                figure.add_trace(
                    go.Scatter(
                        x=dates,
                        y=values.tolist(),
                        customdata=dates,
                        name=f"Cone {strike} {sign * float(level):+g} std",
                        mode="lines",
                        line={"width": 0.5},
                        showlegend=sign == 1,
                        fill="tonexty" if sign == 1 else None,
                        fillcolor=colors[strike],
                        hovertemplate="%{customdata}: %{y:.6g}<extra>%{fullData.name}</extra>",
                    )
                )
    exact = oos_returns.index.astype(str).to_list()
    figure.add_trace(
        go.Scatter(
            x=exact,
            y=wealth.tolist(),
            customdata=exact,
            name=f"Actual cumulative return {(wealth[-1] - 1) * 100:.2f}%",
            mode="lines+markers",
            line={"width": 3, "color": figure.layout.font.color},
        )
    )
    selected = sorted({0, len(exact) // 2, len(exact) - 1})
    figure.update_xaxes(
        type="category",
        tickmode="array",
        tickvals=[exact[i] for i in selected],
        ticktext=[oos_returns.index[i].strftime("%b %d") for i in selected],
    )
    figure.add_hline(y=1, opacity=0.2)
    figure.update_layout(
        meta={
            "analysis": "legacy_cone_redraws",
            "segment_start_rows": starts,
            "cone_std": levels,
            "num_strikes": int(num_strikes),
            "num_samples": "unused by original supplied-bounds plotting route",
            "random_seed": "unused by original supplied-bounds plotting route",
            "description": "Original retrospective lower-two-std crossing and additive shifts of the same supplied bounds, not retrained forecasts. These std cones are not quantiles; actual wealth uses the shared neutral-NaN product convention. Exact instants remain on hover/export.",
        }
    )
    return _style(figure, 420, None)
