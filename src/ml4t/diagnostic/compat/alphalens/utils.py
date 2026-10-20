"""Pandas preparation adapters, not a second factor-analysis engine.

Legacy duration labels describe an inferred observation calendar. They do not
validate an exchange schedule. Native owned endpoints remain available through
``prepare_signal_panel``; this module returns familiar pandas tables instead.
"""

from __future__ import annotations

import warnings
from collections import Counter
from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any, TypeVar

import numpy as np
import pandas as pd
import polars as pl
from ml4t.specs import UNSET_CALENDAR, CalendarUnset, FeedSpec
from pandas.tseries.offsets import BusinessDay, CustomBusinessDay, Day

from ml4t.diagnostic.signal import MaxLossExceededError, prepare_signal_panel
from ml4t.diagnostic.signal._prepared import positive_count, signal_input_contract
from ml4t.diagnostic.signal._price_outcomes import price_outcomes
from ml4t.diagnostic.signal._utils import _signal_frame

__all__ = [
    "MaxLossExceededError",
    "NonMatchingTimezoneError",
    "compute_forward_returns",
    "get_clean_factor",
    "get_clean_factor_and_forward_returns",
]

_Frame = TypeVar("_Frame", pd.Series, pd.DataFrame)


def _profile(version: str) -> str:
    if version not in ("0.5.0", "0.4.6"):
        raise ValueError("reference_version must be 0.5.0 or 0.4.6")
    return version


class NonMatchingTimezoneError(ValueError):
    """Legacy factor and price date keys declare different timezones."""


class _Keys:
    def __init__(self, factor: pd.Series):
        if not isinstance(factor, pd.Series) or not isinstance(factor.index, pd.MultiIndex):
            raise TypeError("factor must be a pandas Series with date/asset MultiIndex")
        if factor.index.nlevels != 2 or not isinstance(factor.index.levels[0], pd.DatetimeIndex):
            raise TypeError("factor requires exactly two index levels, datetime then asset")
        if not factor.index.is_unique or factor.index.to_frame(index=False).isna().any().any():
            raise ValueError("factor date/asset keys must be unique and nonmissing")
        self.original = factor.copy(deep=True)
        self.original.index = self.original.index.set_names(["date", "asset"])
        self.assets = factor.index.get_level_values(1).unique()
        self.codes = {asset: code for code, asset in enumerate(self.assets)}

    def encode(self, data: _Frame) -> _Frame:
        if not isinstance(data.index, pd.MultiIndex) or data.index.nlevels != 2:
            raise TypeError("supplied tables require date/asset MultiIndexes")
        selected = data.index.get_level_values(1).isin(self.assets)
        result = data.loc[selected].copy(deep=True)
        result.index = pd.MultiIndex.from_arrays(
            [result.index.get_level_values(0), result.index.get_level_values(1).map(self.codes)],
            names=["date", "asset"],
        )
        return result

    def restore(self, data: pd.DataFrame) -> pd.DataFrame:
        result = data.copy(deep=True)
        result.index = pd.MultiIndex.from_arrays(
            [
                result.index.get_level_values(0),
                pd.Index(
                    [self.assets[int(code)] for code in result.index.get_level_values(1)],
                    tupleize_cols=False,
                ),
            ],
            names=["date", "asset"],
        )
        return result


def _frequency(factor_dates: pd.DatetimeIndex, price_dates: pd.DatetimeIndex) -> Any:
    if isinstance(factor_dates.freq, Day | BusinessDay | CustomBusinessDay):
        return factor_dates.freq
    dates = factor_dates.union(price_dates)
    weekdays, holidays = [], []
    for day, name in enumerate(("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")):
        observed = dates[dates.dayofweek == day].normalize()
        if not len(observed):
            continue
        weekdays.append(name)
        expected = pd.date_range(
            dates.min(), dates.max(), freq=CustomBusinessDay(weekmask=name)
        ).normalize()
        holidays.extend(expected.difference(observed).date)
    return CustomBusinessDay(weekmask=" ".join(weekdays), holidays=holidays)


def _legacy_labels(
    factor: pd.Series, prices: pd.DataFrame, periods: Sequence[int]
) -> tuple[dict[str, int], Any]:
    dates = factor.index.levels[0]
    if dates.tz != prices.index.tz:
        raise NonMatchingTimezoneError("factor and prices must declare the same timezone")
    frequency = _frequency(dates, prices.index)
    common = dates.intersection(prices.index)
    if not len(common):
        raise ValueError("factor and prices have no matching date keys")
    calendar = np.busdaycalendar(
        weekmask=getattr(frequency, "weekmask", None)
        or ("1111111" if isinstance(frequency, Day) else "1111100"),
        holidays=np.asarray(getattr(frequency, "holidays", None) or [], dtype="datetime64[D]"),
    )
    labels = {}
    for period in sorted({positive_count(period, "periods") for period in periods}):
        durations, days = [], []
        for position in prices.index.get_indexer(common[:30]):
            if position + period >= len(prices):
                continue
            start, end = prices.index[position], prices.index[position + period]
            naive_start = start.tz_convert(None) if start.tzinfo is not None else start
            naive_end = end.tz_convert(None) if end.tzinfo is not None else end
            count = int(np.busday_count(naive_start.date(), naive_end.date(), busdaycal=calendar))
            duration = end - start
            durations.append(duration - pd.Timedelta(days=duration.components.days - count))
            days.append(count)
        if not durations:
            raise ValueError(f"not enough price observations for period {period}")
        counts = Counter(days)
        mode = min(counts, key=lambda day: (-counts[day], day))
        duration = durations[-1] - pd.Timedelta(days=days[-1] - mode)
        components = duration.components
        label = "".join(
            f"{getattr(components, name)}{unit}"
            for name, unit in (
                ("days", "D"),
                ("hours", "h"),
                ("minutes", "m"),
                ("seconds", "s"),
                ("milliseconds", "ms"),
                ("microseconds", "us"),
                ("nanoseconds", "ns"),
            )
            if getattr(components, name)
        )
        if not label or label in labels:
            raise ValueError(
                "legacy inferred horizon labels are empty or ambiguous; supply explicit native selectors"
            )
        labels[label] = period
    if not labels:
        raise ValueError("periods must contain positive observation counts")
    return labels, frequency


def _set_frequency(frame: pd.DataFrame, frequency: Any) -> None:
    if frequency is None:
        return
    try:
        frame.index.levels[0].freq = frequency
    except ValueError:
        warnings.warn(
            "Factor dates do not conform to the inferred legacy calendar; retaining observed frequency.",
            UserWarning,
            stacklevel=3,
        )


def compute_forward_returns(
    factor: pd.Series,
    prices: pd.DataFrame,
    periods: Sequence[int] = (1, 5, 10),
    filter_zscore: float | None = None,
    cumulative_returns: bool = True,
    *,
    reference_version: str = "0.5.0",
    price_basis: str = "legacy_unspecified",
    feed_spec: FeedSpec | Mapping[str, Any] | None = None,
    calendar: str | CalendarUnset | None = UNSET_CALENDAR,
) -> pd.DataFrame:
    """Adapt legacy ffilled observation-count returns and inferred duration labels.

    Filtering fits all factor analysis dates before sparse factor-row selection
    and uses future outcomes. The adapter does not infer an authoritative
    exchange calendar or claim imputed quotes are observed execution prices.
    """
    reference_version = _profile(reference_version)
    keys = _Keys(factor)
    if not isinstance(prices, pd.DataFrame) or not isinstance(prices.index, pd.DatetimeIndex):
        raise TypeError("prices must be a wide pandas DataFrame with DatetimeIndex")
    if (
        not prices.index.is_unique
        or not prices.index.is_monotonic_increasing
        or not prices.columns.is_unique
    ):
        raise ValueError("price dates must be unique and increasing; asset columns must be unique")
    if not isinstance(cumulative_returns, bool | np.bool_):
        raise TypeError("cumulative_returns must be boolean")
    labels, frequency = _legacy_labels(factor, prices, periods)
    encoded_factor = keys.encode(keys.original).rename("factor")
    encoded_prices = prices.loc[:, prices.columns.isin(keys.assets)].copy(deep=True)
    encoded_prices.columns = encoded_prices.columns.map(keys.codes)
    _, _, contract = signal_input_contract(
        feed_spec=feed_spec,
        timestamp_col="date",
        entity_col="asset",
        date_col=None,
        asset_col=None,
        calendar=calendar,
    )
    _, outcomes, metadata = price_outcomes(
        _signal_frame(encoded_factor.to_frame()),
        encoded_prices,
        horizons=labels,
        price_basis=price_basis,
        date_col="date",
        asset_col="asset",
        price_col="close",
        index_mode="common_grid",
        endpoint_method="cumulative" if cumulative_returns else "exit_bar",
        fill_method="ffill",
        return_filter_zscore=filter_zscore,
        available_at_col=None,
        contract=contract,
    )
    result = (
        outcomes.select("date", "asset", "horizon", "forward_return")
        .pivot(
            on="horizon",
            index=["date", "asset"],
            values="forward_return",
        )
        .select("date", "asset", *labels)
        .to_pandas()
        .set_index(["date", "asset"])
    )
    result = keys.restore(result).reindex(keys.original.index)
    _set_frequency(result, frequency)
    result.attrs["diagnostic"] = {
        **metadata,
        "reference_version": reference_version,
        "calendar_label_method": "legacy_inferred_observation_calendar",
        "calendar_contract": contract,
    }
    return result


def get_clean_factor(
    factor: pd.Series,
    forward_returns: pd.DataFrame,
    groupby: pd.Series | Mapping | None = None,
    binning_by_group: bool = False,
    quantiles: int | Sequence[float] | None = 5,
    bins: int | Sequence[float] | None = None,
    groupby_labels: Mapping | None = None,
    max_loss: float = 0.35,
    zero_aware: bool = False,
    *,
    reference_version: str = "0.5.0",
    feed_spec: FeedSpec | Mapping[str, Any] | None = None,
    calendar: str | CalendarUnset | None = UNSET_CALENDAR,
) -> pd.DataFrame:
    """Translate legacy pandas keys, then use the single native panel cleaner."""
    reference_version = _profile(reference_version)
    keys = _Keys(factor)
    if not isinstance(forward_returns, pd.DataFrame):
        raise TypeError("forward_returns must be a pandas DataFrame")
    encoded_outcomes = keys.encode(forward_returns)
    price_metadata = deepcopy(forward_returns.attrs.get("diagnostic", {}))
    contract = price_metadata.get("calendar_contract")
    horizon_col = None
    if contract is not None:
        if feed_spec is None:
            feed_spec = contract["feed_spec"]
        encoded_outcomes = (
            _signal_frame(encoded_outcomes)
            .unpivot(index=["date", "asset"], variable_name="horizon", value_name="forward_return")
            .with_columns(
                pl.lit(contract["calendar"], dtype=pl.String).alias("calendar"),
                pl.lit(contract["calendar_source"] != "default").alias("calendar_declared"),
            )
        )
        horizon_col = "horizon"
    groups = (
        {keys.codes[asset]: value for asset, value in groupby.items() if asset in keys.codes}
        if isinstance(groupby, Mapping)
        else keys.encode(groupby)
        if groupby is not None
        else None
    )
    panel = prepare_signal_panel(
        keys.encode(keys.original).rename("factor"),
        encoded_outcomes,
        quantiles=quantiles,
        bins=bins,
        groups=groups,
        group_labels=groupby_labels,
        binning_by_group=binning_by_group,
        max_loss=max_loss,
        zero_aware=zero_aware,
        return_columns=list(forward_returns.columns),
        horizon_col=horizon_col,
        feed_spec=feed_spec,
        calendar=calendar,
    )
    result = (
        panel.data.to_pandas()
        .set_index(["date", "asset"])
        .rename(columns={"quantile": "factor_quantile"})
    )
    result = keys.restore(result)
    result = result.reindex(keys.original.index[keys.original.index.isin(result.index)])
    columns = [*forward_returns.columns, "factor"]
    if "group" in result:
        result["group"] = result["group"].astype("category")
        columns.append("group")
    result = result[[*columns, "factor_quantile"]]
    result["factor_quantile"] = result["factor_quantile"].astype("int64")
    _set_frequency(result, forward_returns.index.levels[0].freq)
    result.attrs["diagnostic"] = panel.metadata
    result.attrs["diagnostic"]["reference_version"] = reference_version
    result.attrs["diagnostic"]["asset_key_encoding"] = (
        "call-local integer codes restored to original pandas asset IDs; hashes describe normalized inputs"
    )
    if contract is not None:
        result.attrs["diagnostic"]["price_preparation"] = price_metadata
    return result


def get_clean_factor_and_forward_returns(
    factor: pd.Series,
    prices: pd.DataFrame,
    groupby: pd.Series | Mapping | None = None,
    binning_by_group: bool = False,
    quantiles: int | Sequence[float] | None = 5,
    bins: int | Sequence[float] | None = None,
    periods: Sequence[int] = (1, 5, 10),
    filter_zscore: float | None = 20,
    groupby_labels: Mapping | None = None,
    max_loss: float = 0.35,
    zero_aware: bool = False,
    cumulative_returns: bool = True,
    *,
    reference_version: str = "0.5.0",
    price_basis: str = "legacy_unspecified",
    feed_spec: FeedSpec | Mapping[str, Any] | None = None,
    calendar: str | CalendarUnset | None = UNSET_CALENDAR,
) -> pd.DataFrame:
    """Run the same price view and cleaner with explicit legacy-only defaults."""
    outcomes = compute_forward_returns(
        factor,
        prices,
        periods,
        filter_zscore,
        cumulative_returns,
        reference_version=reference_version,
        price_basis=price_basis,
        feed_spec=feed_spec,
        calendar=calendar,
    )
    result = get_clean_factor(
        factor,
        outcomes,
        groupby,
        binning_by_group,
        quantiles,
        bins,
        groupby_labels,
        max_loss,
        zero_aware,
        reference_version=reference_version,
        feed_spec=feed_spec,
        calendar=calendar,
    )
    return result
