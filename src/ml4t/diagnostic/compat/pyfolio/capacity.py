"""Familiar capacity shapes over the native market-liquidity calculations.

Legacy windows count supplied market rows, recent slices use civil-day cutoffs,
positions are normalized to a fixed capital base, and impact uses compounded
closing NAV. Native defaults instead use observed marked dollars and beginning
NAV. Known zero-volume exposure is unbounded in both routes. Unlike the
reference, a missing active market bar cannot silently become zero impact.
"""

from __future__ import annotations

from numbers import Real

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from ml4t.diagnostic.compat import empyrical as ep
from ml4t.diagnostic.evaluation.portfolio_analysis.capacity import (
    _lagged_dollar_volume,
    _liquidation_values,
    _positive_real,
    _volume_share_cost,
)
from ml4t.diagnostic.evaluation.portfolio_analysis.inputs import positive_integer

from .txn import _books, _dated


def _numeric(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    for column, dtype in frame.dtypes.items():
        if pd.api.types.is_bool_dtype(dtype) or (
            not is_numeric_dtype(dtype)
            and not all(
                isinstance(value, Real) and not isinstance(value, bool)
                for value in frame[column].dropna()
            )
        ):
            raise TypeError(f"{name} must contain numeric values")
    result = frame.astype(float)
    if np.isinf(result.to_numpy()).any():
        raise ValueError(f"{name} must be finite or missing")
    return result


def _market(market_data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if (
        not isinstance(market_data, pd.DataFrame)
        or not isinstance(market_data.index, pd.MultiIndex)
        or market_data.index.nlevels != 2
    ):
        raise TypeError("market_data requires the familiar (date, field) MultiIndex")
    if market_data.index.has_duplicates or market_data.columns.has_duplicates:
        raise ValueError("market date/field keys and asset columns must be unique")
    price, volume = (
        market_data.xs(field, level=1).copy(deep=True) for field in ("price", "volume")
    )
    for name, frame in (("price", price), ("volume", volume)):
        _dated(frame, name)
        if not frame.index.is_monotonic_increasing or not frame.index.is_unique:
            raise ValueError("market dates must be sorted and unique")
    if not price.index.equals(volume.index) or not price.columns.equals(volume.columns):
        raise ValueError(
            "market price and volume must share their declared observation clock/assets"
        )
    price, volume = _numeric(price, "market prices"), _numeric(volume, "market volumes")
    if (price <= 0).any().any() or (volume < 0).any().any():
        raise ValueError("observed market price must be positive and volume nonnegative")
    return price, volume


def _allocations(positions: pd.DataFrame) -> pd.DataFrame:
    _dated(positions, "positions")
    if not positions.index.is_monotonic_increasing:
        raise ValueError("position dates must be sorted")
    _, nav = _books(positions)
    return _numeric(positions, "positions").drop(columns="cash").div(nav, axis=0)


def daily_txns_with_bar_data(transactions: pd.DataFrame, market_data: pd.DataFrame) -> pd.DataFrame:
    """Civil-date absolute units and supplied closing price/volume, without mutation.

    Multiple assets at one date remain distinct rows. The reference's final
    asfreq fails on such a real ledger; this route retains those keys. Empty
    civil days between activity dates keep the familiar all-missing row.
    """
    _dated(transactions, "transactions")
    price, volume = _market(market_data)
    values = _numeric(transactions[["amount"]], "transaction quantities")
    if values.isna().any().any() or transactions["symbol"].isna().any():
        raise ValueError("actual transaction quantities and stable symbols must be nonmissing")
    source = transactions[["symbol"]].copy(deep=True)
    integral = pd.api.types.is_integer_dtype(transactions["amount"].dtype)
    source["amount"] = (
        pd.Series(
            [abs(int(value)) for value in transactions["amount"]], index=source.index, dtype=object
        )
        if integral
        else values["amount"].abs()
    )
    source["date"] = source.index.normalize()
    daily = (
        source.groupby(["date", "symbol"], sort=False, observed=True)["amount"]
        .sum()
        .reset_index()
        .set_index("date")
        .sort_index(kind="stable")
    )
    if integral:
        daily["amount"] = pd.Series(daily["amount"].tolist(), index=daily.index)
    keys = pd.MultiIndex.from_arrays([daily.index, daily["symbol"]])
    for name, frame in (("price", price), ("volume", volume)):
        flattened = pd.Series(
            frame.to_numpy().reshape(-1),
            index=pd.MultiIndex.from_product([frame.index, frame.columns]),
        )
        daily[name] = flattened.reindex(keys).to_numpy()
    if not daily.empty:
        absent = pd.date_range(daily.index.min(), daily.index.max(), freq="D").difference(
            daily.index.unique()
        )
        if len(absent) and integral and daily["amount"].max() > 2**53:
            daily["amount"] = daily["amount"].astype(object)
        daily = (
            pd.concat(
                [
                    daily,
                    pd.DataFrame(
                        {
                            "symbol": pd.Series(index=absent, dtype=object),
                            "amount": np.nan,
                            "price": np.nan,
                            "volume": np.nan,
                        },
                        index=absent,
                    ),
                ]
            ).sort_index(kind="stable")
            if len(absent)
            else daily
        )
    return daily[["symbol", "amount", "price", "volume"]].rename_axis("date")


def days_to_liquidate_positions(
    positions: pd.DataFrame,
    market_data: pd.DataFrame,
    max_bar_consumption: float = 0.2,
    capital_base: float = 1e6,
    mean_volume_window: int = 5,
) -> pd.DataFrame:
    """Fixed-capital allocation liquidation with the legacy trimmed row window."""
    return _liquidation_frames(
        positions, market_data, max_bar_consumption, capital_base, mean_volume_window
    )[0]


def _liquidation_frames(
    positions: pd.DataFrame,
    market_data: pd.DataFrame,
    max_bar_consumption: float,
    capital_base: float,
    mean_volume_window: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    limit = _positive_real(max_bar_consumption, "max_bar_consumption")
    if limit > 1:
        raise ValueError("max_bar_consumption must not exceed one")
    capital = _positive_real(capital_base, "capital_base")
    window = positive_integer(mean_volume_window, "mean_volume_window")
    price, volume = _market(market_data)
    denominator = _lagged_dollar_volume(price * volume, window)
    allocation, denominator = _allocations(positions).align(denominator)
    result = pd.DataFrame(
        _liquidation_values(allocation.to_numpy() * capital, denominator.to_numpy(), limit),
        index=allocation.index,
        columns=allocation.columns,
    )
    result = result.mask(denominator.isna())
    return result.iloc[window:], allocation.iloc[window:]


def get_max_days_to_liquidate_by_ticker(
    positions: pd.DataFrame,
    market_data: pd.DataFrame,
    max_bar_consumption: float = 0.2,
    capital_base: float = 1e6,
    mean_volume_window: int = 5,
    last_n_days: int | None = None,
) -> pd.DataFrame:
    """Actual worst rows per asset, full history or inclusive civil-day cutoff.

    Tied durations choose the earliest observation deterministically. Each
    returned date/allocation belongs to that row, never columnwise first-nonnull
    fragments from different observations.
    """
    values, allocation = _liquidation_frames(
        positions, market_data, max_bar_consumption, capital_base, mean_volume_window
    )
    if last_n_days is not None:
        days = positive_integer(last_n_days, "last_n_days")
        keep = values.index >= pd.Timestamp(values.index.max()) - pd.Timedelta(days=days)
        values, allocation = values.loc[keep], allocation.loc[keep]
    rows = []
    for asset in values.columns:
        ordered = (
            values[asset]
            .reset_index(drop=True)
            .sort_values(ascending=False, kind="stable", na_position="last")
        )
        if ordered.empty:
            continue
        observation = int(ordered.index[0])
        day = values.index[observation]
        rows.append(
            {
                "symbol": asset,
                "date": day,
                "days_to_liquidate": ordered.iloc[0],
                "pos_alloc_pct": allocation.iloc[observation][asset] * 100
                if asset in allocation
                else np.nan,
            }
        )
    return pd.DataFrame(
        rows, columns=["symbol", "date", "days_to_liquidate", "pos_alloc_pct"]
    ).set_index("symbol")


def get_low_liquidity_transactions(
    transactions: pd.DataFrame,
    market_data: pd.DataFrame,
    last_n_days: int | None = None,
) -> pd.DataFrame:
    """Worst actual daily participation by asset, with legacy strict recent cutoff."""
    daily = daily_txns_with_bar_data(transactions, market_data).reset_index()
    if last_n_days is not None:
        days = positive_integer(last_n_days, "last_n_days")
        daily = daily[daily["date"] > daily["date"].max() - pd.Timedelta(days=days)]
    daily["max_pct_bar_consumed"] = daily["amount"] / daily["volume"] * 100
    selected = (
        daily.sort_values(
            "max_pct_bar_consumed", ascending=False, kind="stable", na_position="last"
        )
        .dropna(subset=["symbol"])
        .drop_duplicates("symbol")
    )
    return selected.set_index("symbol")[["date", "max_pct_bar_consumed"]]


def apply_slippage_penalty(
    returns: pd.Series,
    txn_daily: pd.DataFrame,
    simulate_starting_capital: float,
    backtest_starting_capital: float,
    impact: float = 0.1,
) -> pd.Series:
    """Legacy compounded closing-NAV extra impact with unavailable bars retained.

    This uses actual observed daily bar volume, not the lagged liquidation
    denominator. It is a retrospective hypothetical capital sweep. Observed
    fees in net returns are not removed a second time. Nonpositive NAV leaves
    the percentage penalty undefined rather than returning a sign-reversed fee.
    """
    _dated(returns, "returns")
    _dated(txn_daily, "txn_daily")
    capital = _positive_real(simulate_starting_capital, "simulate_starting_capital")
    base = _positive_real(backtest_starting_capital, "backtest_starting_capital")
    coefficient = _positive_real(impact, "impact", zero=True)
    values = _numeric(txn_daily[["amount", "price", "volume"]], "daily market transactions")
    active = values["amount"].notna()
    costs = pd.Series(
        _volume_share_cost(
            values["amount"].to_numpy(),
            values["price"].to_numpy(),
            values["volume"].to_numpy(),
            np.ones(len(values)),
            capital / base,
            coefficient,
        ),
        index=values.index,
    )
    daily = costs.resample("D").sum(min_count=1)
    unknown = (active & costs.isna()).resample("D").max().reindex(daily.index, fill_value=False)
    daily = daily.fillna(0.0).mask(unknown)
    aligned = daily.reindex(returns.index)
    aligned.loc[~returns.index.isin(daily.index)] = 0.0
    nav = ep.cum_returns(returns, starting_value=capital)
    return returns - aligned / nav.where((nav > 0) & np.isfinite(nav))


__all__ = [
    "daily_txns_with_bar_data",
    "days_to_liquidate_positions",
    "get_max_days_to_liquidate_by_ticker",
    "get_low_liquidity_transactions",
    "apply_slippage_penalty",
]
