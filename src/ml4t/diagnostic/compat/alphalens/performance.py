"""Alphalens statistical and portfolio shapes over shared Diagnostic kernels."""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
from ml4t.specs import UNSET_CALENDAR, CalendarUnset, FeedSpec
from pandas.tseries.offsets import BusinessDay

from ml4t.diagnostic.compat.alphalens.utils import _Keys, _profile, get_forward_returns_columns
from ml4t.diagnostic.evaluation.portfolio_analysis.metrics import _safe_cumprod
from ml4t.diagnostic.signal._prepared import signal_input_contract
from ml4t.diagnostic.signal.portfolio import (
    _select_factor_sample,
    compute_factor_cumulative_returns,
    compute_factor_returns,
    compute_factor_weights,
)
from ml4t.diagnostic.signal.portfolio_inputs import build_factor_portfolio_inputs
from ml4t.diagnostic.signal.positions import (
    _aggregate_allocations,
    _expiry_keys,
    compute_factor_positions,
)
from ml4t.diagnostic.signal.quantile import _quantile_mean_spread, compute_quantile_statistics
from ml4t.diagnostic.signal.signal_ic import compute_signal_ic
from ml4t.diagnostic.utils.legacy_time import add_custom_calendar_timedelta

__all__ = [
    "factor_information_coefficient",
    "mean_information_coefficient",
    "mean_return_by_quantile",
    "compute_mean_returns_spread",
    "factor_weights",
    "factor_returns",
    "cumulative_returns",
    "positions",
    "factor_cumulative_returns",
    "factor_positions",
    "create_pyfolio_input",
]


def _native(factor_data: pd.DataFrame) -> tuple[_Keys, pl.DataFrame]:
    if not isinstance(factor_data, pd.DataFrame):
        raise TypeError("factor_data must be a pandas DataFrame")
    keys = _Keys(factor_data["factor"])
    return keys, pl.from_pandas(keys.encode(factor_data).reset_index())


def factor_weights(
    factor_data: pd.DataFrame,
    demeaned: bool = True,
    group_adjust: bool = False,
    equal_weight: bool = False,
    *,
    reference_version: str = "0.5.0",
) -> pd.Series:
    """Adapt shared date/group weights, preserving opaque pandas asset IDs."""
    _profile(reference_version)
    keys, data = _native(factor_data)
    weights = compute_factor_weights(
        data, demeaned=demeaned, group_adjust=group_adjust, equal_weight=equal_weight
    )
    result = keys.restore(
        weights.select("date", "asset", "weight").to_pandas().set_index(["date", "asset"])
    )
    result = result.reindex(keys.original.index)
    result.index = keys.original.index.copy()
    return result["weight"].rename("factor")


def factor_returns(
    factor_data: pd.DataFrame,
    demeaned: bool = True,
    group_adjust: bool = False,
    equal_weight: bool = False,
    by_asset: bool = False,
    *,
    reference_version: str = "0.5.0",
) -> pd.DataFrame:
    """Adapt shared contributions and aggregate returns with legacy frequency padding."""
    _profile(reference_version)
    keys, data = _native(factor_data)
    columns = list(get_forward_returns_columns(factor_data.columns))
    output = compute_factor_returns(
        data,
        return_columns=columns,
        demeaned=demeaned,
        group_adjust=group_adjust,
        equal_weight=equal_weight,
        by_asset=by_asset,
    )
    result = output.to_pandas().pivot(
        index=["date", "asset"] if by_asset else "date",
        columns="return_column",
        values="weighted_return",
    )
    result = result.reindex(columns=columns)
    result.columns.name = None
    if by_asset:
        result = keys.restore(result).reindex(keys.original.index)
        result.index = keys.original.index.copy()
    else:
        freq = factor_data.index.levels[0].freq
        if freq is not None:
            result = result.asfreq(freq)
    return result


def cumulative_returns(returns: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    """Compound simple returns with the existing NaN-aware wealth kernel."""
    if isinstance(returns, pd.DataFrame):
        return returns.apply(
            lambda column: pd.Series(_safe_cumprod(1 + column.to_numpy()), index=column.index)
        )
    if not isinstance(returns, pd.Series):
        raise TypeError("returns must be a pandas Series or DataFrame")
    return pd.Series(_safe_cumprod(1 + returns.to_numpy()), index=returns.index)


def _expiry_dates(
    dates: pd.DatetimeIndex, period: str | pd.Timedelta, freq: Any, calendar: Any
) -> pd.DatetimeIndex:
    duration = pd.Timedelta(period)
    if duration <= pd.Timedelta(0):
        raise ValueError("period must be a strictly positive holding duration")
    if calendar is None:
        if freq is not None:
            raise ValueError("explicit calendar=None disables calendar offsets; omit freq")
        return dates + duration
    if freq is None:
        freq = dates.freq
    if freq is None:
        freq = BusinessDay()
        warnings.warn("'freq' not set, using business day calendar", UserWarning, stacklevel=3)
    return pd.DatetimeIndex(add_custom_calendar_timedelta(dates, duration, freq))


def _wide_positions(result, keys: _Keys) -> pd.DataFrame:
    frame = keys.restore(result.positions.to_pandas().set_index(["date", "asset"]))
    wide = frame["weight"].unstack()
    wide.columns.name = "asset"
    return wide


def positions(
    weights: pd.Series,
    period: str | pd.Timedelta,
    freq: Any = None,
    *,
    reference_version: str = "0.5.0",
    feed_spec: FeedSpec | Mapping[str, Any] | None = None,
    calendar: str | CalendarUnset | None = UNSET_CALENDAR,
) -> pd.DataFrame:
    """Net overlapping allocations using explicit legacy-offset expiry conversion.

    Corrected all-due expiry semantics apply to both reference profiles; the
    frozen single-pop bug is not reproduced. Canonical calendar=None uses
    elapsed duration without offsets. Other calendar declarations are metadata,
    not a claim that a pandas business-day convention validates an exchange.
    """
    _profile(reference_version)
    keys = _Keys(weights)
    values = keys.encode(keys.original).rename("weight").reset_index()
    values["weight"] = values["weight"].fillna(0.0)
    if not np.isfinite(values["weight"].to_numpy()).all():
        raise ValueError("weights must be finite")
    _, _, contract = signal_input_contract(
        feed_spec=feed_spec,
        timestamp_col=None,
        entity_col=None,
        date_col="date",
        asset_col="asset",
        calendar=calendar,
    )
    dates = weights.unstack().index
    expiry_dates = _expiry_dates(dates, period, freq, contract["calendar"])
    lookup = pd.Series(expiry_dates, index=dates)
    values["expiry"] = values["date"].map(lookup)
    native = pl.from_pandas(values)
    result = _aggregate_allocations(
        _expiry_keys(native.drop("expiry"), native, "expiry"),
        {
            "input_contract": contract,
            "representation": "synthetic_expiring_allocations",
        },
    )
    return _wide_positions(result, keys).reindex(dates.union(expiry_dates).drop_duplicates())


def _selected_positions(
    factor_data,
    period,
    long_short,
    group_neutral,
    equal_weight,
    quantiles,
    groups,
    feed_spec,
    calendar,
):
    keys, data = _native(factor_data)
    if period not in get_forward_returns_columns(factor_data.columns):
        raise ValueError(f"Period {period!r} not found")
    data = _select_factor_sample(data, quantiles, groups, "factor_quantile", "group")
    _, _, contract = signal_input_contract(
        feed_spec=feed_spec,
        timestamp_col=None,
        entity_col=None,
        date_col="date",
        asset_col="asset",
        calendar=calendar,
    )
    # Match legacy unstack's observed entry clock, not unused MultiIndex levels.
    observed = data["date"].unique().to_pandas()
    dates = (
        factor_data.loc[factor_data.index.get_level_values(0).isin(observed), "factor"]
        .unstack()
        .index
    )
    expiry_dates = _expiry_dates(dates, period, None, contract["calendar"])
    expiry_lookup = pl.from_pandas(pd.DataFrame({"date": dates, "expiry": expiry_dates}))
    expiries = data.select("date", "asset").join(expiry_lookup, on="date", how="left")
    result = compute_factor_positions(
        data,
        expiries=expiries,
        demeaned=long_short,
        group_adjust=group_neutral,
        equal_weight=equal_weight,
        feed_spec=feed_spec,
        calendar=calendar,
    )
    return keys, data, result, dates.union(expiry_dates).drop_duplicates()


def factor_positions(
    factor_data: pd.DataFrame,
    period: str,
    long_short: bool = True,
    group_neutral: bool = False,
    equal_weight: bool = False,
    quantiles: Sequence[int] | None = None,
    groups: Sequence[str] | None = None,
    *,
    reference_version: str = "0.5.0",
    feed_spec: FeedSpec | Mapping[str, Any] | None = None,
    calendar: str | CalendarUnset | None = UNSET_CALENDAR,
) -> pd.DataFrame:
    """Select before shared sizing and aggregate the actual union-event holdings."""
    _profile(reference_version)
    keys, _, result, event_index = _selected_positions(
        factor_data,
        period,
        long_short,
        group_neutral,
        equal_weight,
        quantiles,
        groups,
        feed_spec,
        calendar,
    )
    return _wide_positions(result, keys).reindex(event_index)


def factor_cumulative_returns(
    factor_data: pd.DataFrame,
    period: str,
    long_short: bool = True,
    group_neutral: bool = False,
    equal_weight: bool = False,
    quantiles: Sequence[int] | None = None,
    groups: Sequence[str] | None = None,
    *,
    reference_version: str = "0.5.0",
) -> pd.Series:
    """Select before weighting and compound one supplied horizon, not account P&L."""
    _profile(reference_version)
    _, data = _native(factor_data)
    selected = _select_factor_sample(data, quantiles, groups, "factor_quantile", "group")
    if period not in get_forward_returns_columns(factor_data.columns):
        raise ValueError(f"Period {period!r} not found")
    frequency = factor_data.index.levels[0].freq
    dates = selected["date"].to_pandas()
    grid = (
        pd.DataFrame({"date": pd.date_range(dates.min(), dates.max(), freq=frequency)})
        if frequency is not None and len(dates)
        else None
    )
    path = compute_factor_cumulative_returns(
        selected,
        return_column=period,
        demeaned=long_short,
        group_adjust=group_neutral,
        equal_weight=equal_weight,
        date_grid=grid,
    )
    result = path.select("date", "wealth").to_pandas().set_index("date")["wealth"].rename(None)
    if frequency is not None:
        result = result.asfreq(frequency)
    return result


def create_pyfolio_input(
    factor_data: pd.DataFrame,
    period: str,
    capital: float | None = None,
    long_short: bool = True,
    group_neutral: bool = False,
    equal_weight: bool = False,
    quantiles: Sequence[int] | None = None,
    groups: Sequence[str] | None = None,
    benchmark_period: str = "1D",
    *,
    reference_version: str = "0.5.0",
    feed_spec: FeedSpec | Mapping[str, Any] | None = None,
    calendar: str | CalendarUnset | None = UNSET_CALENDAR,
) -> tuple[pd.Series, pd.DataFrame, pd.Series | None]:
    """Adapt the legacy daily-last wealth/event-sum holdings interchange.

    The first daily return is zero. Empty event-sum days are all cash, even
    when a mathematical allocation remains active. Capital scales these
    analytical values; no fills or transaction costs are fabricated.
    """
    _profile(reference_version)
    keys, _, positions_result, _ = _selected_positions(
        factor_data,
        period,
        long_short,
        group_neutral,
        equal_weight,
        quantiles,
        groups,
        feed_spec,
        calendar,
    )
    _, all_data = _native(factor_data)
    inputs = build_factor_portfolio_inputs(
        all_data,
        positions=positions_result,
        return_column=period,
        benchmark_column=benchmark_period,
        capital=capital,
        position_method="legacy_event_sum",
        first_return="zero",
    )
    returns, values, benchmark = inputs.to_pyfolio()
    values = values.rename(columns=dict(enumerate(keys.assets)))
    if "cash" in keys.assets:
        raise ValueError("the Pyfolio wide format reserves asset ID 'cash'; use native long tables")
    return returns, values, benchmark


def _statistical_inputs(factor_data: pd.DataFrame) -> tuple[pl.DataFrame, list[str]]:
    if not isinstance(factor_data, pd.DataFrame):
        raise TypeError("factor_data must be a pandas DataFrame")
    index_sample = factor_data.get("factor", pd.Series(index=factor_data.index, dtype=float))
    keys = _Keys(index_sample)
    return pl.from_pandas(keys.encode(factor_data).reset_index()), list(
        get_forward_returns_columns(factor_data.columns)
    )


def _wide_statistics(
    observations: pl.DataFrame,
    *,
    value: str,
    dimensions: list[str],
    columns: list[str],
    original: pd.DataFrame,
) -> pd.DataFrame:
    result = observations.to_pandas().pivot(index=dimensions, columns="return_column", values=value)
    result = result.reindex(columns=columns)
    result.columns.name = None
    if "group" in dimensions and isinstance(original["group"].dtype, pd.CategoricalDtype):
        result.index = result.index.set_levels(
            pd.CategoricalIndex(
                result.index.levels[dimensions.index("group")], dtype=original["group"].dtype
            ),
            level="group",
        )
    return result.sort_index()


def factor_information_coefficient(
    factor_data: pd.DataFrame,
    group_adjust: bool = False,
    by_group: bool = False,
    *,
    reference_version: str = "0.5.0",
) -> pd.DataFrame:
    """Adapt native dated ICs, retaining legacy per-column NaN propagation."""
    profile = _profile(reference_version)
    data, columns = _statistical_inputs(factor_data)
    observations = compute_signal_ic(
        data, return_columns=columns, group_adjust=group_adjust, by_group=by_group
    )
    result = _wide_statistics(
        observations,
        value="ic",
        dimensions=["date", *(["group"] if by_group else [])],
        columns=columns,
        original=factor_data,
    )
    invalid = (
        factor_data[columns].isna().apply(lambda column: column | factor_data["factor"].isna())
    )
    grouping = [
        factor_data.index.get_level_values("date"),
        *([factor_data["group"]] if by_group else []),
    ]
    result = result.mask(
        invalid.groupby(grouping, observed=True).any().reindex(result.index).fillna(False)
    )
    if not by_group:
        frequency = factor_data.index.levels[0].freq
        if frequency is not None or profile == "0.4.6":
            result = result.asfreq(frequency)
    return result


def mean_information_coefficient(
    factor_data: pd.DataFrame,
    group_adjust: bool = False,
    by_group: bool = False,
    by_time: str | pd.DateOffset | None = None,
    *,
    reference_version: str = "0.5.0",
) -> pd.Series | pd.DataFrame:
    """Aggregate the same IC observations using the caller's pandas time rule."""
    result = factor_information_coefficient(
        factor_data, group_adjust, by_group, reference_version=reference_version
    )
    grouping = ([pd.Grouper(freq=by_time)] if by_time is not None else []) + (
        ["group"] if by_group else []
    )
    if not grouping:
        return result.mean()
    return result.reset_index().set_index("date").groupby(grouping, observed=False).mean()


def mean_return_by_quantile(
    factor_data: pd.DataFrame,
    by_date: bool = False,
    by_group: bool = False,
    demeaned: bool = True,
    group_adjust: bool = False,
    *,
    reference_version: str = "0.5.0",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Adapt shared equal-date means/errors without re-bucketing or score filtering."""
    _profile(reference_version)
    data, columns = _statistical_inputs(factor_data)
    statistics = compute_quantile_statistics(
        data,
        return_columns=columns,
        by_date=by_date,
        by_group=by_group,
        demeaned=demeaned,
        group_adjust=group_adjust,
        factor_col=None,
        quantile_col="factor_quantile",
    ).rename({"quantile": "factor_quantile"})
    dimensions = [
        "factor_quantile",
        *(["date"] if by_date else []),
        *(["group"] if by_group else []),
    ]
    return tuple(
        _wide_statistics(
            statistics, value=value, dimensions=dimensions, columns=columns, original=factor_data
        )
        for value in ("mean", "standard_error")
    )


def compute_mean_returns_spread(
    mean_returns: pd.DataFrame,
    upper_quant: int,
    lower_quant: int,
    std_err: pd.DataFrame | None = None,
    *,
    reference_version: str = "0.5.0",
) -> tuple[pd.Series | pd.DataFrame, pd.Series | pd.DataFrame | None]:
    """Use shared spread/error arithmetic with explicit unit-gross or frozen raw scale."""
    profile = _profile(reference_version)
    upper, lower = (
        mean_returns.xs(upper_quant, level="factor_quantile"),
        mean_returns.xs(lower_quant, level="factor_quantile"),
    )
    upper, lower = upper.align(lower)
    error_upper = error_lower = None
    if std_err is not None:
        error_upper, error_lower = (
            std_err.xs(upper_quant, level="factor_quantile"),
            std_err.xs(lower_quant, level="factor_quantile"),
        )
        error_upper, error_lower = error_upper.align(error_lower)
    spread, error = _quantile_mean_spread(
        upper.to_numpy(dtype=float, na_value=np.nan),
        lower.to_numpy(dtype=float, na_value=np.nan),
        error_upper.to_numpy(dtype=float, na_value=np.nan) if error_upper is not None else None,
        error_lower.to_numpy(dtype=float, na_value=np.nan) if error_lower is not None else None,
    )
    scale = 0.5 if profile == "0.5.0" else 1.0
    difference = upper.copy()
    difference.iloc[...] = spread * scale
    if error_upper is None:
        return difference, None
    uncertainty = error_upper.copy()
    uncertainty.iloc[...] = error * scale
    return difference, uncertainty
