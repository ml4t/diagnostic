"""Complete familiar position manifest using the existing portfolio report renderer."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from . import pos, utils
from . import position_plotting as plots
from .return_plotting import _return_series

if TYPE_CHECKING:
    from ml4t.diagnostic.visualization.portfolio import PortfolioTearSheet


def create_position_tear_sheet(
    returns: pd.Series,
    positions: pd.DataFrame,
    show_and_plot_top_pos: int = 2,
    hide_positions: bool = False,
    sector_mappings: Mapping[Any, Any] | pd.Series | None = None,
    transactions: pd.DataFrame | None = None,
    estimate_intraday: bool | np.bool_ | str | None = "infer",
    return_fig: bool = False,
) -> PortfolioTearSheet | None:
    """Compose all familiar holdings panels and requested top/sector tables.

    return_fig=True returns a native PortfolioTearSheet. Otherwise display it
    and return None. Inputs are owned, not rewritten account returns. The
    explicit intraday selector reuses the native estimate; it is not an observed
    marked account reconstruction. Plotly replaces Matplotlib presentation.
    """
    from ml4t.diagnostic.visualization.portfolio import PortfolioTearSheet

    mode = plots._display_mode(show_and_plot_top_pos)
    if not all(isinstance(flag, bool | np.bool_) for flag in (hide_positions, return_fig)):
        raise TypeError("hide_positions and return_fig must be boolean")
    supplied_returns = _return_series(returns)
    selected = utils.check_intraday(estimate_intraday, supplied_returns, positions, transactions)
    if selected is None:
        raise ValueError("positions are required for the position report")
    supplied = plots._positions(selected)
    if (supplied.index.tz is None) != (supplied_returns.index.tz is None):
        raise ValueError("returns and positions must both be timezone-aware or both naive")
    snapshot_kind = supplied.attrs.get("diagnostic", {}).get("snapshot_kind", "supplied")
    if hide_positions:
        original_labels = supplied.columns
        supplied = plots._hidden(supplied)
        if sector_mappings is not None:
            if isinstance(sector_mappings, pd.Series):
                if not sector_mappings.index.is_unique:
                    raise ValueError("sector mapping keys must be unique")
                sector_mappings = sector_mappings.to_dict()
            if not isinstance(sector_mappings, Mapping):
                raise TypeError("sector_mappings must be a mapping or pandas Series")
            sectors: dict[Any, str] = {}
            anonymous = {}
            for original, label in zip(original_labels, supplied.columns, strict=True):
                if original == "cash" or original not in sector_mappings:
                    continue
                sector = sector_mappings[original]
                if pd.isna(sector):
                    anonymous[label] = None
                else:
                    sectors.setdefault(sector, f"Sector {len(sectors) + 1}")
                    anonymous[label] = sectors[sector]
            sector_mappings = anonymous
        mode = 0
    allocations = pos.get_percent_alloc(supplied)
    figures = {"Exposure": plots.plot_exposures(supplied_returns, supplied)}
    if mode in (0, 2):
        figures["Top positions"] = plots._top_plot(
            supplied_returns, allocations, "real_best", None, {}
        )
    figures.update(
        {
            "Position concentration": plots.plot_max_median_position_concentration(supplied),
            "Holdings": plots.plot_holdings(supplied_returns, allocations),
            "Long and short holdings": plots.plot_long_short_holdings(
                supplied_returns, allocations
            ),
            "Gross leverage": plots.plot_gross_leverage(supplied_returns, supplied),
        }
    )
    if mode in (1, 2):
        figures.update(plots._top_tables(allocations))
    availability = {
        "positions": {
            "status": "available",
            "snapshot_kind": snapshot_kind,
            "n_observations": len(supplied),
            "duplicate_time_observations": int(supplied.index.duplicated().sum()),
            "nonfinite_nav_fractions": figures["Exposure"].layout.meta["nonfinite_values"],
            "identities": "anonymized assets and sectors" if hide_positions else "supplied",
            "reason": "Legacy signed-NAV profile; each panel states its denominator. Exact timestamps and tied rows are retained. Nonfinite fractions are unavailable. Account NAV is not reconstructed.",
        },
        "sectors": {"status": "not_requested"},
    }
    if sector_mappings is not None:
        sectors = pos.get_sector_exposures(supplied, sector_mappings)
        sector_allocations = pos.get_percent_alloc(sectors).drop(columns="cash")
        figures["Sector allocations"] = plots.plot_sector_allocations(
            supplied_returns, sector_allocations
        )
        availability["sectors"] = {
            "status": "available" if len(sector_allocations.columns) else "unavailable",
            "denominator": "mapped_sector_values_plus_cash",
            "reason": "Mapped sectors plus cash form the denominator; unmapped assets are excluded with warnings. Native full-NAV weights are a separate route. An empty mapped book has no sector estimate.",
        }
    report = PortfolioTearSheet(metrics=None, figures=figures, availability=availability)
    if return_fig:
        return report
    report.show()
    return None


__all__ = ["create_position_tear_sheet"]
