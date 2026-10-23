"""Familiar transaction manifest over the existing actual-fill and cost panels."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from ml4t.diagnostic.evaluation.portfolio_analysis.inputs import positive_integer

from . import utils
from .position_plotting import _positions
from .return_plotting import _return_series
from .txn import _dated

if TYPE_CHECKING:
    from ml4t.diagnostic.visualization.portfolio import PortfolioTearSheet


def create_txn_tear_sheet(
    returns: pd.Series,
    positions: pd.DataFrame,
    transactions: pd.DataFrame,
    turnover_denom: str = "AGB",
    unadjusted_returns: pd.Series | None = None,
    estimate_intraday: bool | np.bool_ | str | None = "infer",
    return_fig: bool = False,
    *,
    periods_per_year: int = 252,
) -> PortfolioTearSheet | None:
    """Compose turnover, actual units, timing and optional extra-slippage panels.

    return_fig=True returns a native PortfolioTearSheet; otherwise display it
    and return None. Actual fills remain separate from the explicitly selected
    position estimate. No trade, commission, rebalance or account return is
    inferred from position changes. Scale counts return observations per year.
    """
    from ml4t.diagnostic.visualization.portfolio import PortfolioTearSheet

    from . import plotting

    if not isinstance(return_fig, bool | np.bool_):
        raise TypeError("return_fig must be boolean")
    if turnover_denom not in ("AGB", "portfolio_value"):
        raise ValueError("turnover_denom must be AGB or portfolio_value")
    scale = positive_integer(periods_per_year, "periods_per_year")
    supplied_returns = _return_series(returns)
    if not isinstance(transactions, pd.DataFrame):
        raise TypeError("transactions must be a pandas DataFrame")
    _dated(transactions, "transactions")
    actual_fills = transactions.copy(deep=True)
    supplied_positions = _positions(positions)
    if (actual_fills.index.tz is None) != (supplied_positions.index.tz is None):
        raise ValueError("transactions and positions must declare compatible timezone-aware clocks")
    if actual_fills.index.tz is not None:
        actual_fills.index = actual_fills.index.tz_convert(supplied_positions.index.tz)
    selected = utils.check_intraday(
        estimate_intraday, supplied_returns, supplied_positions, actual_fills
    )
    if selected is None:
        raise ValueError("positions are required for the transaction report")
    selected = _positions(selected)
    figures = {
        "Turnover": plotting.plot_turnover(
            supplied_returns, actual_fills, selected, turnover_denom
        ),
        "Daily volume": plotting.plot_daily_volume(supplied_returns, actual_fills),
        "Turnover distribution": plotting.plot_daily_turnover_hist(
            actual_fills, selected, turnover_denom
        ),
        "Transaction timing": plotting.plot_txn_time_hist(actual_fills),
    }
    if unadjusted_returns is not None:
        unadjusted = _return_series(unadjusted_returns)
        figures["Slippage sweep"] = plotting.plot_slippage_sweep(unadjusted, selected, actual_fills)
        figures["Slippage sensitivity"] = plotting.plot_slippage_sensitivity(
            unadjusted, selected, actual_fills, periods_per_year=scale
        )
    for title, figure in figures.items():
        figure.update_layout(title_text=title)
    report = PortfolioTearSheet(
        metrics=None,
        figures=figures,
        availability={
            "transactions": {
                "status": "available",
                "n_actual_fills": len(actual_fills),
                "turnover_denom": turnover_denom,
                "source_fill_timezone": str(transactions.index.tz),
                "civil_day_timezone": str(supplied_positions.index.tz),
                "positions_kind": selected.attrs.get("diagnostic", {}).get(
                    "snapshot_kind", "supplied"
                ),
                "reason": "Actual fill activity with the explicit legacy closing-value or zero-seeded AGB denominator. Volume counts activity dates; timing selects inclusive09:30-16:00 New York notional. Histogram bins are Plotly presentation, not copied Seaborn bins.",
            },
            "slippage": {
                "status": "not_requested" if unadjusted_returns is None else "available",
                "periods_per_year": scale,
                "reason": "Only additional hypothetical cash bps are subtracted. Observed fees are not removed or inferred; supplied unadjusted returns are the scenario baseline.",
            },
        },
    )
    if return_fig:
        return report
    report.show()
    return None


__all__ = ["create_txn_tear_sheet"]
