"""Explicit retrospective daily transaction-notional projection, not actual account replay."""

from __future__ import annotations

from bisect import bisect_right
from collections import defaultdict
from collections.abc import Mapping
from math import copysign, isfinite
from numbers import Real
from typing import Any, Literal

import pandas as pd
import polars as pl
from ml4t.specs import UNSET_CALENDAR, CalendarUnset, FeedSpec

from .frame_inputs import (
    PortfolioInputData,
    _frame_declarations,
    _merge_cash,
    _positions,
    _valuation_snapshot,
    normalize_executions,
)
from .holdings import HoldingsAnalysisResult, analyze_holdings
from .transaction_metrics import execution_value_expr


def estimate_intraday_positions(
    positions: pl.DataFrame | pd.DataFrame,
    transactions: pl.DataFrame | pd.DataFrame,
    valuations: pl.DataFrame | pd.DataFrame | None = None,
    *,
    first_return: float | None,
    complete_snapshots: bool,
    method: Literal["legacy_peak_notional"],
    capital_method: Literal["defined_only", "legacy_zero_fill"] = "defined_only",
    feed_spec: FeedSpec | Mapping[str, Any] | None = None,
    timestamp_col: str | None = None,
    entity_col: str | None = None,
    transaction_timestamp_col: str | None = None,
    positions_format: Literal["auto", "long", "wide"] = "auto",
    calendar: str | CalendarUnset | None = UNSET_CALENDAR,
) -> HoldingsAnalysisResult:
    """Project daily representative marked positions using an explicit legacy heuristic.

    Select every maximum-absolute cumulative transaction-notional candidate per
    civil day and add it to previous supplied EOD marked values. Initial risky
    positions are zero and initial cash is inferred from the first EOD marked
    total divided by one plus the first supplied return. Unknown inferred cash
    remains unknown. This is not quantity/quote reconstruction, observed NAV,
    cost-aware cash or causal daily peak information. Same-instant flows sum;
    the published reference's pivot-mean defect is not reproduced.
    first_return is a decimal first-EOD-period return, not an annual rate;
    None/NaN means unknown. No fabricated return series is needed.
    """
    if complete_snapshots is not True:
        raise ValueError("intraday estimate requires complete_snapshots=True")
    if positions_format not in {"auto", "long", "wide"}:
        raise ValueError("positions_format must be auto, long or wide")
    feed, timestamp, entity = _frame_declarations(feed_spec, timestamp_col, entity_col)
    owned, cash = _positions(positions, timestamp, entity, positions_format)
    account = _merge_cash(_valuation_snapshot(valuations, timestamp), cash)
    if account is None:
        account = owned.select("date").unique().sort("date")
    fills = normalize_executions(
        transactions,
        timestamp_col=transaction_timestamp_col
        or (feed.timestamp_col if feed_spec is not None else "timestamp"),
        entity_col=entity,
    )
    return _estimate_owned(
        owned,
        account,
        fills,
        first_return=first_return,
        declarations={"feed_spec": feed.to_dict()},
        method=method,
        capital_method=capital_method,
        calendar=calendar,
    )


def _estimate_from_inputs(
    inputs: PortfolioInputData,
    *,
    method: Literal["legacy_peak_notional"],
    calendar: str | CalendarUnset | None,
    capital_method: Literal["defined_only", "legacy_zero_fill"],
) -> HoldingsAnalysisResult:
    from ml4t.diagnostic.utils.validation import _align_datetime_key_resolution

    positions, account, fills = inputs.positions, inputs.valuations, inputs.transactions
    if positions is None or fills is None or inputs.metadata["complete_snapshots"] is not True:
        raise ValueError(
            "intraday estimate requires complete EOD positions and actual transactions"
        )
    keys = inputs.returns.select("date")
    if account is None:
        account = keys
    else:
        account, keys = _align_datetime_key_resolution(account, keys, "date")
        account = account.join(keys, on="date", how="semi").sort("date")
    return _estimate_owned(
        positions,
        account,
        fills,
        first_return=inputs.returns["return"][0],
        declarations=inputs.metadata,
        method=method,
        capital_method=capital_method,
        calendar=calendar,
    )


def _estimate_owned(
    positions: pl.DataFrame,
    account: pl.DataFrame,
    fills: pl.DataFrame,
    *,
    first_return: float | None,
    declarations: dict[str, Any],
    method: Literal["legacy_peak_notional"],
    capital_method: Literal["defined_only", "legacy_zero_fill"],
    calendar: str | CalendarUnset | None,
) -> HoldingsAnalysisResult:
    if method != "legacy_peak_notional":
        raise ValueError("method='legacy_peak_notional' must explicitly select this estimate")
    if capital_method not in {"defined_only", "legacy_zero_fill"}:
        raise ValueError("capital_method must be defined_only or explicit legacy_zero_fill")
    if account.is_empty():
        raise ValueError("at least one complete EOD observation is required")
    feed = FeedSpec.from_any(declarations["feed_spec"])
    selection = calendar
    if selection is UNSET_CALENDAR and declarations.get("calendar_source") == "argument":
        selection = declarations["calendar_argument"]
    if feed.resolve_calendar(calendar=selection) is not None:
        raise ValueError("legacy_peak_notional is a civil-date profile; select calendar=None")
    dates = account["date"]
    source = analyze_holdings(
        positions, account, dates, complete_snapshots=True, snapshot_kind="end_of_day"
    )
    eod_clock = pd.DatetimeIndex(dates.to_pandas())
    if not (eod_clock == eod_clock.normalize()).all():
        raise ValueError(
            "legacy_peak_notional requires daily EOD labels, not intraday observations"
        )
    raw_clock = pd.DatetimeIndex(fills["timestamp"].to_pandas())
    group_clock = raw_clock
    if feed.timezone is not None:
        group_clock = (
            group_clock.tz_localize(feed.timezone)
            if group_clock.tz is None
            else group_clock.tz_convert(feed.timezone)
        )
    elif eod_clock.tz is not None and str(eod_clock.tz) != str(group_clock.tz):
        raise ValueError(
            "civil-date projection needs consistent clocks or an explicit feed timezone"
        )
    asset_dtype = source.positions.schema["asset"]
    if (
        source.positions.height
        and fills.height
        and (asset_dtype == pl.String) != (fills.schema["asset"] == pl.String)
    ):
        raise TypeError(
            "positions and transactions must retain the same string/integer asset namespace"
        )
    if source.positions.is_empty():
        asset_dtype = fills.schema["asset"]
    elif fills.height and asset_dtype != fills.schema["asset"]:
        asset_dtype = pl.Int128
    assets = list(dict.fromkeys([*source.positions["asset"].to_list(), *fills["asset"].to_list()]))
    labels = pd.Index(assets, dtype=object)
    days = [value.date() for value in eod_clock]
    if source.positions.height:
        observed = pd.DataFrame(
            {
                "date": pd.DatetimeIndex(source.positions["date"].to_pandas()).date,
                "asset": pd.Series(source.positions["asset"].to_list(), dtype=object),
                "value": source.positions["value"].to_list(),
            }
        )
        base = (
            observed.pivot(index="date", columns="asset", values="value")
            .reindex(index=days, columns=labels)
            .fillna(0.0)
        )
    else:
        base = pd.DataFrame(0.0, index=pd.Index(days), columns=labels)
    shifted = base.shift(1).fillna(0.0)
    supplied_cash = pd.Series(source.timeline["cash"].to_list(), index=base.index, dtype=float)
    lagged_cash = supplied_cash.shift(1)
    if first_return is not None and (
        isinstance(first_return, bool) or not isinstance(first_return, Real)
    ):
        raise TypeError("first_return must be a real numeric period return or None")
    first_return = float(first_return) if first_return is not None else float("nan")
    if not pd.isna(first_return) and not isfinite(first_return):
        raise ValueError("first_return must be finite or explicitly unknown")
    first_total = supplied_cash.iloc[0] + base.iloc[0].sum()
    if pd.isna(first_return) or pd.isna(first_total):
        inferred_capital = float("nan")
    elif first_return == -1:
        inferred_capital = float("nan") if first_total == 0 else copysign(float("inf"), first_total)
    else:
        inferred_capital = first_total / (1 + first_return)
    capital = (
        inferred_capital
        if isfinite(inferred_capital)
        else (
            0.0
            if capital_method == "legacy_zero_fill" and pd.isna(inferred_capital)
            else float("nan")
        )
    )
    lagged_cash.iloc[0] = capital
    cash_values = fills.select(execution_value_expr().alias("value"))["value"].to_list()
    if not all(isfinite(value) for value in cash_values):
        raise ValueError("transaction cash values must remain finite")
    transactions = pd.DataFrame(
        {"asset": pd.Series(fills["asset"].to_list(), dtype=object), "value": cash_values}
    )
    transactions.index = group_clock
    if fills.height:
        pivot = transactions.pivot_table(
            index=transactions.index, columns="asset", values="value", aggfunc="sum", fill_value=0
        ).reindex(columns=labels, fill_value=0)
        running = pivot.groupby(pivot.index.date).cumsum()
        if not all(isfinite(value) for value in running.to_numpy().ravel()):
            raise ValueError("cumulative transaction cash values must remain finite")
        exposure = running.abs().sum(axis=1)
        peaks = running.loc[exposure == exposure.groupby(running.index.date).transform("max")]
    else:
        peaks = pd.DataFrame(columns=labels, index=group_clock)
    peak_days = dict(iter(peaks.groupby(peaks.index.date, sort=False)))
    records_by_day = defaultdict(list)
    original_times = {}
    ordered_events = []
    for index in sorted(range(fills.height), key=lambda index: (group_clock[index].value, index)):
        day = group_clock[index].date()
        records_by_day[day].append(group_clock[index].value)
        original_times.setdefault(group_clock[index].value, raw_clock[index])
        ordered_events.append(
            {
                "record_id": fills["record_id"][index],
                "timestamp": raw_clock[index].isoformat(),
                "civil_date": day.isoformat(),
                "ordinal": len(records_by_day[day]) - 1,
                "signed_cash_notional": cash_values[index],
            }
        )
    projected, accounts = [], []
    for day in sorted(set(days) | {value.date() for value in group_clock}):
        candidates = peak_days.get(day, peaks.iloc[:0])
        baseline = shifted.loc[day] if day in shifted.index else pd.Series(0.0, index=labels)
        previous_cash = lagged_cash.loc[day] if day in lagged_cash.index else 0.0
        for candidate in range(max(1, len(candidates))):
            flow = candidates.iloc[candidate] if len(candidates) else pd.Series(0.0, index=labels)
            values = baseline + flow
            cash = previous_cash - flow.sum()
            identifier = len(accounts)
            peak = candidates.index[candidate] if len(candidates) else None
            original = original_times.get(peak.value) if peak is not None else None
            accounts.append(
                {
                    "date": day,
                    "observation_id": identifier,
                    "cash": None if pd.isna(cash) else float(cash),
                    "equity": None if pd.isna(cash) else float(cash + values.sum()),
                    "candidate_timestamp": None if original is None else original.value,
                    "member_stop": bisect_right(records_by_day[day], peak.value)
                    if peak is not None
                    else 0,
                }
            )
            projected.append(values.to_numpy(dtype=float, copy=True))
    clock_dtype = pl.Datetime("ns", fills.schema["timestamp"].time_zone)
    timeline = pl.DataFrame(
        accounts, schema_overrides={"candidate_timestamp": pl.Int64}
    ).with_columns(pl.col("candidate_timestamp").cast(clock_dtype))
    if assets:
        names = [f"holding{index}" for index in range(len(assets))]
        wide = pl.DataFrame(projected, schema=names, orient="row").with_columns(
            timeline["date"], timeline["observation_id"]
        )
        mapping = pl.DataFrame({"holding": names, "asset": pl.Series(assets, dtype=asset_dtype)})
        holdings = (
            wide.unpivot(on=names, index=["date", "observation_id"], variable_name="holding")
            .join(mapping, on="holding", how="left", maintain_order="left")
            .drop("holding")
        )
    else:
        holdings = pl.DataFrame(
            schema={
                "date": pl.Date,
                "observation_id": pl.Int64,
                "asset": asset_dtype,
                "value": pl.Float64,
            }
        )
    result = analyze_holdings(
        holdings,
        timeline,
        timeline["date"],
        complete_snapshots=True,
        snapshot_kind="estimated_intraday",
        observation_ids=timeline["observation_id"],
    )
    result.timeline = result.timeline.join(
        timeline.select("observation_id", "candidate_timestamp", "member_stop"),
        on="observation_id",
        how="left",
        maintain_order="left",
    )
    result.metadata.update(
        method=method,
        capital_method=capital_method,
        initial_capital_identified=isfinite(inferred_capital),
        calendar=None,
        feed_spec=feed.to_dict(),
        first_return=None if pd.isna(first_return) else first_return,
        first_return_unit="decimal return for the first EOD period, not annualized",
        group_timezone=str(group_clock.tz) if group_clock.tz is not None else None,
        calendar_source="argument" if selection is not UNSET_CALENDAR else "feed_spec",
        selection="all equal-maximum absolute cumulative transaction-notional candidates per civil day; simultaneous flows sum",
        positions="previous supplied EOD marked values plus cumulative signed execution cash notionals, not inferred units/quotes",
        cash="first EOD marked total/(1+first supplied return), then lagged EOD cash minus cumulative signed transaction notionals; not reported account cash",
        costs="fees ignored by explicit legacy estimate; actual execution prices already include slippage",
        peak_semantics="retrospective transaction-notional selection, not observed or continuous marked exposure peaks",
        average_semantics="equal-weight selected candidates, including ties; not equal-day or elapsed-time weighting",
        input_clock="daily civil-date labels, not intraday instants; candidate_timestamp retains original selected execution clock",
        membership="execution_order filtered by civil_date, first member_stop records; source same-instant order retained without quadratic prefix copies",
        execution_order=ordered_events,
        initial_capital=None if pd.isna(capital) else float(capital),
        unknown_cash="defined_only keeps unidentified capital unknown; explicit legacy_zero_fill substitutes zero for NaN capital as the published projection does; infinite implied capital stays unavailable",
        native_alternatives="analyze_intraday_exposure for observed snapshots; RoundTripResult.reconstruct_intraday_exposure for explicit cash-price account replay",
    )
    return result
