"""Compose familiar account workflows from the existing analytical report components."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from ml4t.diagnostic.evaluation.portfolio_analysis.inputs import positive_integer

from . import pos, utils
from .position_plotting import _positions, _top_plot
from .return_plotting import _return_series
from .return_sheet import _table_figure
from .txn import _dated

if TYPE_CHECKING:
    from ml4t.diagnostic.visualization.portfolio import PortfolioTearSheet


def _owned_inputs(
    returns: pd.Series,
    positions: pd.DataFrame | None,
    transactions: pd.DataFrame | None,
) -> tuple[pd.Series, pd.DataFrame | None, pd.DataFrame | None]:
    supplied = _return_series(returns)
    book = None if positions is None else _positions(positions)
    fills = None
    if book is not None and (book.index.tz is None) != (supplied.index.tz is None):
        raise ValueError("returns and positions must both be timezone-aware or both naive")
    if transactions is not None:
        if book is None:
            raise ValueError("transactions require supplied positions and separate cash")
        if not isinstance(transactions, pd.DataFrame):
            raise TypeError("transactions must be a pandas DataFrame")
        _dated(transactions, "transactions")
        fills = transactions.copy(deep=True)
        if (fills.index.tz is None) != (book.index.tz is None):
            raise ValueError(
                "transactions and positions must declare compatible timezone-aware clocks"
            )
        if fills.index.tz is not None:
            fills.index = fills.index.tz_convert(book.index.tz)
    return supplied, book, fills


def _anonymous_inputs(
    book: pd.DataFrame,
    fills: pd.DataFrame | None,
    sectors: Mapping[Any, Any] | pd.Series | None,
    market: pd.DataFrame | None,
    loadings: pd.DataFrame | None,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame | None,
    dict[Any, Any] | None,
    pd.DataFrame | None,
    pd.DataFrame | None,
]:
    labels = {label: f"Holding {i + 1}" for i, label in enumerate(book.columns) if label != "cash"}
    extra_labels = []
    if fills is not None:
        extra_labels.extend(fills["symbol"])
    if market is not None:
        extra_labels.extend(market.columns)
    if loadings is not None and isinstance(loadings.index, pd.MultiIndex):
        extra_labels.extend(loadings.index.get_level_values(1))
    for label in extra_labels:
        if label not in labels:
            labels[label] = f"Holding {len(labels) + 1}"
    if fills is not None:
        fills = fills.copy(deep=True)
        fills["symbol"] = fills["symbol"].map(labels)
        # Only the actual-fill fields consumed by these workflows belong in exports.
        fills = fills[["symbol", "amount", "price"]]
        fills.attrs = {}
    book = book.rename(columns=labels)
    book.attrs = {}
    anonymous_sectors = None
    if sectors is not None:
        if isinstance(sectors, pd.Series):
            if not sectors.index.is_unique:
                raise ValueError("sector mapping keys must be unique")
            sectors = sectors.to_dict()
        if not isinstance(sectors, Mapping):
            raise TypeError("sector_mappings must be a mapping or pandas Series")
        names = {}
        anonymous_sectors = {}
        for asset, sector in sectors.items():
            if asset not in labels:
                continue
            if pd.isna(sector):
                anonymous_sectors[labels[asset]] = None
            else:
                names.setdefault(sector, f"Sector {len(names) + 1}")
                anonymous_sectors[labels[asset]] = names[sector]
    if market is not None:
        market = market.rename(columns=labels).copy(deep=True)
        market.attrs = {}
    if loadings is not None:
        loadings = loadings.copy(deep=True)
        if isinstance(loadings.index, pd.MultiIndex) and loadings.index.nlevels == 2:
            loadings.index = pd.MultiIndex.from_arrays(
                [
                    loadings.index.get_level_values(0),
                    loadings.index.get_level_values(1).map(labels),
                ],
                names=loadings.index.names,
            )
        loadings.attrs = {}
    return book, fills, anonymous_sectors, market, loadings


def _create_account_tear_sheet(
    *,
    simple: bool,
    returns: pd.Series,
    positions: pd.DataFrame | None = None,
    transactions: pd.DataFrame | None = None,
    benchmark_rets: pd.Series | None = None,
    slippage: float | None = None,
    estimate_intraday: bool | np.bool_ | str | None = "infer",
    live_start_date: Any = None,
    turnover_denom: str = "AGB",
    header_rows: dict[str, Any] | None = None,
    market_data: pd.DataFrame | None = None,
    sector_mappings: Mapping[Any, Any] | pd.Series | None = None,
    round_trips: bool = False,
    hide_positions: bool = False,
    cone_std: float | tuple[float, ...] | None = (1.0, 1.5, 2.0),
    bootstrap: bool = False,
    unadjusted_returns: pd.Series | None = None,
    set_context: bool = True,
    factor_returns: pd.DataFrame | None = None,
    factor_loadings: pd.DataFrame | None = None,
    pos_in_dollars: bool = True,
    factor_partitions: Mapping[str, Any] | None = None,
    return_fig: bool = False,
    periods_per_year: int = 252,
    num_samples: int = 1000,
    n_samples: int = 1000,
    random_seed: int | None = None,
) -> PortfolioTearSheet | None:
    from ml4t.diagnostic.visualization.portfolio import PortfolioTearSheet

    from . import plotting, tears, txn
    from .capacity_sheet import _prepare_capacity_tear_sheet
    from .stress_sheet import _prepare_interesting_times_tear_sheet

    if not all(
        isinstance(flag, bool | np.bool_)
        for flag in (
            return_fig,
            round_trips,
            hide_positions,
            bootstrap,
            set_context,
            pos_in_dollars,
        )
    ):
        raise TypeError("report flags must be boolean")
    scale = positive_integer(periods_per_year, "periods_per_year")
    if turnover_denom not in ("AGB", "portfolio_value"):
        raise ValueError("turnover_denom must be AGB or portfolio_value")
    if (factor_returns is None) != (factor_loadings is None):
        raise ValueError("factor_returns and factor_loadings must both be supplied")
    for name, frame in (
        ("market_data", market_data),
        ("factor_returns", factor_returns),
        ("factor_loadings", factor_loadings),
    ):
        if frame is not None and not isinstance(frame, pd.DataFrame):
            raise TypeError(f"{name} must be a pandas DataFrame")
    if factor_returns is not None and positions is None:
        raise ValueError("factor attribution requires positions")
    if (market_data is not None or round_trips or slippage is not None) and (
        positions is None or transactions is None
    ):
        raise ValueError(
            "capacity, round trips and slippage require positions and actual transactions"
        )
    supplied, book, fills = _owned_inputs(returns, positions, transactions)
    if hide_positions and book is not None:
        book, fills, sector_mappings, market_data, factor_loadings = _anonymous_inputs(
            book, fills, sector_mappings, market_data, factor_loadings
        )
    actual_book = book
    if not simple and unadjusted_returns is None and slippage is not None:
        unadjusted_returns = supplied.copy(deep=True)
        supplied = txn.adjust_returns_for_slippage(supplied, book, fills, slippage)
    selected = utils.check_intraday(estimate_intraday, supplied, book, fills)
    if simple and slippage is not None:
        supplied = txn.adjust_returns_for_slippage(supplied, selected, fills, slippage)
    report = PortfolioTearSheet(metrics=None)

    def include(prefix: str, child: PortfolioTearSheet) -> None:
        for name, figure in child.figures.items():
            report.figures[f"{prefix}/{name}"] = figure
        report.availability.update(child.availability)

    if simple:
        stats = plotting.show_perf_stats(
            supplied,
            factor_returns=benchmark_rets,
            positions=selected,
            transactions=fills,
            turnover_denom=turnover_denom,
            live_start_date=live_start_date,
            header_rows=header_rows,
            periods_per_year=scale,
            return_df=True,
        )
        report.figures = {
            "Returns/Performance statistics": _table_figure(
                stats, "Performance statistics", "Supplied returns and actual account observations."
            ),
            "Returns/Cumulative wealth": plotting.plot_rolling_returns(
                supplied,
                factor_returns=benchmark_rets,
                live_start_date=live_start_date,
                cone_std=(1.0, 1.5, 2.0),
                num_samples=num_samples,
                random_seed=random_seed,
            ),
            "Returns/Rolling Sharpe": plotting.plot_rolling_sharpe(
                supplied, periods_per_year=scale
            ),
            "Returns/Underwater": plotting.plot_drawdown_underwater(supplied),
        }
        if benchmark_rets is not None:
            report.figures["Returns/Rolling beta"] = plotting.plot_rolling_beta(
                supplied, benchmark_rets
            )
    else:
        returns_report = tears.create_returns_tear_sheet(
            supplied,
            selected,
            fills,
            live_start_date,
            cone_std,
            benchmark_rets,
            bootstrap,
            turnover_denom,
            header_rows,
            True,
            periods_per_year=scale,
            num_samples=num_samples,
            n_samples=n_samples,
            random_seed=random_seed,
        )
        assert returns_report is not None
        include("Returns", returns_report)
        stress = _prepare_interesting_times_tear_sheet(supplied, benchmark_rets)
        if stress is not None:
            figures, _, tables = stress
            for name, table in tables.items():
                report.figures[f"Stress/{name}"] = _table_figure(
                    table, name, "Original event membership; mean, minimum and maximum are percent."
                )
            for number, figure in enumerate(figures):
                report.figures[f"Stress/Event {number + 1}"] = figure
        report.availability["stress"] = {
            "status": "available" if stress is not None else "unavailable",
            "reason": "Default event boundaries select actual observations; an absent event is not a zero return.",
        }
    report.availability["positions"] = {"status": "not_requested"}
    report.availability["transactions"] = {"status": "not_requested"}
    if selected is not None:
        if simple:
            allocations = pos.get_percent_alloc(selected)
            report.figures.update(
                {
                    "Positions/Exposure": plotting.plot_exposures(supplied, selected),
                    "Positions/Top positions": _top_plot(
                        supplied, allocations, "real_best", None, {}
                    ),
                    "Positions/Holdings": plotting.plot_holdings(supplied, allocations),
                    "Positions/Long and short holdings": plotting.plot_long_short_holdings(
                        supplied, allocations
                    ),
                }
            )
            report.availability["positions"] = {"status": "available"}
        else:
            positions_report = tears.create_position_tear_sheet(
                supplied,
                selected,
                hide_positions=hide_positions,
                sector_mappings=sector_mappings,
                estimate_intraday=False,
                return_fig=True,
            )
            assert positions_report is not None
            include("Positions", positions_report)
        if fills is not None:
            if simple:
                report.figures["Transactions/Turnover"] = plotting.plot_turnover(
                    supplied, fills, selected, turnover_denom
                )
                report.figures["Transactions/Transaction timing"] = plotting.plot_txn_time_hist(
                    fills
                )
                report.availability["transactions"] = {"status": "available"}
            else:
                transactions_report = tears.create_txn_tear_sheet(
                    supplied,
                    selected,
                    fills,
                    turnover_denom,
                    unadjusted_returns,
                    False,
                    True,
                    periods_per_year=scale,
                )
                assert transactions_report is not None
                include("Transactions", transactions_report)
            if round_trips:
                trips = tears.create_round_trip_tear_sheet(
                    supplied,
                    actual_book,
                    fills,
                    sector_mappings,
                    estimate_intraday,
                    True,
                )
                assert trips is not None
                include("Round trips", trips)
                report.round_trips = trips.round_trips
                report.round_trip_statistics = trips.round_trip_statistics
                report.round_trip_attribution = trips.round_trip_attribution
            if market_data is not None:
                figure, sections, tables = _prepare_capacity_tear_sheet(
                    supplied,
                    selected,
                    fills,
                    market_data,
                    last_n_days=125,
                    estimate_intraday=False,
                    periods_per_year=scale,
                )
                report.figures["Capacity/Capital base performance sweep"] = figure
                description = sections[0]["description"]
                for name, table in tables.items():
                    report.figures[f"Capacity/{name}"] = _table_figure(table, name, description)
                report.availability["capacity"] = {"status": "available", "reason": description}
        if factor_returns is not None:
            figures, _, tables = tears._prepare_perf_attrib_tear_sheet(
                supplied,
                selected,
                factor_returns,
                factor_loadings,
                fills,
                pos_in_dollars,
                factor_partitions,
                periods_per_year=scale,
            )
            for name, table in tables.items():
                report.figures[f"Attribution/{name}"] = _table_figure(
                    table,
                    name,
                    f"Supplied holdings/loadings attribution; {scale} return observations per year.",
                )
            for number, figure in enumerate(figures):
                report.figures[f"Attribution/Panel {number + 1}"] = figure
            report.availability["attribution"] = {"status": "available"}
    for key, requested in (
        ("capacity", market_data is not None),
        ("attribution", factor_returns is not None),
        ("round_trips", round_trips),
    ):
        report.availability.setdefault(
            key, {"status": "available" if requested else "not_requested"}
        )
    report.availability["composition"] = {
        "status": "available",
        "workflow": "simple" if simple else "full",
        "periods_per_year": scale,
        "turnover_denom": turnover_denom,
        "source_fill_timezone": None if transactions is None else str(transactions.index.tz),
        "civil_day_timezone": None if book is None else str(book.index.tz),
        "positions_kind": "not_supplied"
        if selected is None
        else selected.attrs.get("diagnostic", {}).get("snapshot_kind", "supplied"),
        "identities": "anonymized assets and sectors" if hide_positions else "supplied",
        "reason": "Component-specific samples, missingness and definitions remain explicit. Returns are supplied account facts; fills are actual executions. Scale counts return observations, not rebalances. set_context does not mutate Matplotlib global styling. Plotly replaces the original artist API.",
    }
    if return_fig:
        return report
    report.show()
    return None
