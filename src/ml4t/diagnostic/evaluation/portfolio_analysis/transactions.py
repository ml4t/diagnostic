"""Actual execution activity and incremental costs over the owned account clock."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Real
from typing import Any

import numpy as np
import pandas as pd
import polars as pl

from ml4t.diagnostic.utils.validation import _align_datetime_key_resolution

from .frame_inputs import PortfolioInputData


@dataclass
class TransactionAnalysisResult:
    """Owned actual fills, full/active-clock turnover and hypothetical extra costs.

    Observed implementation cost requires commission and per-unit price slippage.
    Incremental bps uses actual gross traded cash notional and beginning NAV,
    independent of the selected turnover denominator or one-way convention.
    Input net returns are authoritative and never have observed fees removed again.
    """

    executions: pl.DataFrame
    timeline: pl.DataFrame
    scenarios: pl.DataFrame
    metrics: dict[str, Any]
    metadata: dict[str, Any]
    observations: pl.DataFrame


def _clock(
    dates: pl.Series,
    calendar: str | None,
    semantics: str | None,
    *,
    executions: bool = False,
    timezone: str | None = None,
) -> pl.Series:
    if calendar is None:
        return dates.rename("date")
    from ml4t.diagnostic.splitters.calendar import TradingCalendar

    index = pd.DatetimeIndex(dates.to_pandas())
    labels = not executions and (dates.dtype == pl.Date or semantics == "session_label")
    if labels:
        # Labels designate dates, not midnight instants interpreted in an exchange zone.
        if not (index == index.normalize()).all():
            raise ValueError("session_label observations must be date labels at midnight")
        index = index.tz_localize(None) if index.tz is not None else index
    elif index.tz is None:
        if timezone is None:
            raise ValueError(
                "execution/event instants require timezone-aware timestamps or a declared timezone"
            )
        index = index.tz_localize(timezone)
    sessions, mask = TradingCalendar(calendar).get_sessions_and_mask(
        index, include_daily_labels=labels, bar_close=not executions and semantics == "bar_close"
    )
    if not mask.all():
        raise ValueError("timestamps fall outside the declared exchange sessions")
    return pl.Series("date", sessions.dt.date.to_list(), dtype=pl.Date)


def _strict_sum(column: str) -> pl.Expr:
    return (
        pl.when(pl.col(column).null_count() == 0)
        .then(pl.col(column).sum())
        .otherwise(None)
        .alias(column)
    )


def analyze_transactions(
    inputs: PortfolioInputData,
    *,
    turnover_denom: str,
    turnover_method: str,
    incremental_bps: Sequence[float],
) -> TransactionAnalysisResult:
    from ml4t.diagnostic.integration.backtest_analytics import prepare_activity_fills

    if turnover_denom not in ("beginning_nav", "average_gross_book"):
        raise ValueError("turnover_denom must be beginning_nav or average_gross_book")
    if turnover_method not in ("gross", "one_way"):
        raise ValueError("turnover_method must be gross or one_way")
    bps = list(incremental_bps)
    if not bps or any(
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not np.isfinite(value)
        or value < 0
        for value in bps
    ):
        raise ValueError("incremental_bps must contain finite nonnegative real values")
    if len(set(bps)) != len(bps):
        raise ValueError("incremental_bps must contain distinct values")
    fills = inputs.transactions
    if fills is None:
        raise ValueError("transactions were not supplied; absent executions are not zero activity")
    meta = inputs.metadata
    calendar = meta["calendar"]
    semantics = meta["feed_spec"].get("timestamp_semantics")
    timezone = meta["feed_spec"].get("timezone")
    observations = inputs.returns.with_columns(
        _clock(inputs.returns["date"], calendar, semantics, timezone=timezone)
    )
    clock = observations.group_by("date", maintain_order=True).agg(
        pl.when(pl.len() == 1)
        .then(pl.col("return").first())
        .otherwise((1 + pl.col("return")).product() - 1)
        .alias("return"),
        pl.len().alias("return_observations"),
    )
    # No grouping when calendars are disabled. Preserve the original exact return values.
    if calendar is None:
        clock = observations.with_columns(pl.lit(1, dtype=pl.UInt32).alias("return_observations"))
    enriched = prepare_activity_fills(fills, unknown_costs=True)
    if not enriched["filled_notional"].is_finite().all():
        raise ValueError("execution notional overflows the account-currency calculation")
    session_keys = (
        _clock(fills["timestamp"], calendar, None, executions=True, timezone=timezone)
        if not fills.is_empty()
        else pl.Series("date", [], dtype=clock["date"].dtype)
    )
    clock, fill_keys = _align_datetime_key_resolution(clock, session_keys.to_frame(), "date")
    session_keys = fill_keys["date"]
    if session_keys.dtype != clock["date"].dtype:
        # A Date label is not an instant and cannot be matched in calendar=None row mode.
        raise ValueError(
            "transaction keys must match return observation types in calendar=None mode"
        )
    enriched = enriched.with_columns(session_keys.alias("session"))
    if enriched.select(pl.col("session").is_in(clock["date"].implode()).all()).item() is False:
        raise ValueError("transactions have no matching return observation/session")
    activity = (
        enriched.group_by("session")
        .agg(
            pl.len().alias("num_fills"),
            pl.col("filled_notional").sum(),
            _strict_sum("implementation_cost"),
        )
        .rename({"session": "date"})
    )
    timeline = (
        clock.join(activity, on="date", how="left")
        .with_columns(
            pl.col("num_fills").fill_null(0),
            pl.col("filled_notional").fill_null(0.0),
        )
        .with_columns(
            pl.when(pl.col("num_fills") == 0)
            .then(0.0)
            .otherwise(pl.col("implementation_cost"))
            .alias("implementation_cost")
        )
    )

    account = inputs.valuations
    if account is None:
        timeline = timeline.with_columns(pl.lit(None, dtype=pl.Float64).alias("beginning_nav"))
    else:
        account = account.sort("date")
        account = account.with_columns(
            _clock(account["date"], calendar, semantics, timezone=timezone)
        )
        account = account.group_by("date", maintain_order=True).agg(pl.col("equity").last())
        timeline, account = _align_datetime_key_resolution(timeline, account, "date")
        # Retain the immediately previous reported valuation, including an explicit initial
        # value. An unreported/missing close is not forward-filled across return observations.
        keys = pl.concat([timeline.select("date"), account.select("date")]).unique().sort("date")
        nav = keys.join(account, on="date", how="left").with_columns(
            pl.col("equity").shift(1).alias("beginning_nav")
        )
        timeline = timeline.join(nav.select("date", "beginning_nav"), on="date", how="left")
    if turnover_denom == "average_gross_book":
        positions = inputs.positions
        if positions is None or not meta["complete_snapshots"]:
            raise ValueError("average_gross_book requires declared complete positions")
        if {"asset", "value"}.issubset(positions.columns):
            gross = positions.group_by("date").agg(pl.col("value").abs().sum().alias("gross"))
        else:
            gross = positions.select("date").with_columns(pl.lit(0.0).alias("gross"))
        gross = gross.sort("date")
        gross = gross.with_columns(_clock(gross["date"], calendar, semantics, timezone=timezone))
        gross = gross.group_by("date", maintain_order=True).agg(pl.col("gross").last()).sort("date")
        timeline, gross = _align_datetime_key_resolution(timeline, gross, "date")
        gross = gross.with_columns(
            ((pl.col("gross") + pl.col("gross").shift(1)) / 2).alias("denominator")
        )
        timeline = timeline.join(gross.select("date", "denominator"), on="date", how="left")
    else:
        timeline = timeline.with_columns(pl.col("beginning_nav").alias("denominator"))
    valid = pl.col("denominator").is_finite() & (pl.col("denominator") > 0)
    nav_valid = pl.col("beginning_nav").is_finite() & (pl.col("beginning_nav") > 0)
    timeline = timeline.with_columns(
        pl.when(valid)
        .then(
            pl.col("filled_notional")
            / pl.col("denominator")
            / (2 if turnover_method == "one_way" else 1)
        )
        .otherwise(None)
        .alias("turnover"),
        pl.when(nav_valid)
        .then(pl.col("implementation_cost") / pl.col("beginning_nav"))
        .otherwise(None)
        .alias("observed_cost_return"),
        pl.when(valid)
        .then(pl.lit("defined"))
        .otherwise(pl.lit("unavailable_denominator"))
        .alias("normalization_status"),
    )
    scenarios = []
    for value in bps:
        scenario = (
            timeline.select("date", "return", "filled_notional", "beginning_nav")
            .with_columns(
                pl.lit(float(value)).alias("incremental_bps"),
                (pl.col("filled_notional") * float(value) / 10000).alias("incremental_cost"),
            )
            .with_columns(
                pl.when(pl.col("incremental_cost") == 0)
                .then(0.0)
                .when(nav_valid)
                .then(pl.col("incremental_cost") / pl.col("beginning_nav"))
                .otherwise(None)
                .alias("incremental_cost_return")
            )
            .with_columns(
                (pl.col("return") - pl.col("incremental_cost_return")).alias("net_return")
            )
        )
        scenarios.append(scenario)
    return TransactionAnalysisResult(
        enriched,
        timeline,
        pl.concat(scenarios),
        {
            "num_fills": enriched.height,
            "total_filled_notional": float(enriched["filled_notional"].sum()),
            "total_implementation_cost": None
            if enriched["implementation_cost"].null_count()
            else float(enriched["implementation_cost"].sum()),
            "n_observations": timeline.height,
            "n_active_observations": timeline.filter(pl.col("num_fills") > 0).height,
            "n_normalized_observations": timeline["turnover"].len()
            - timeline["turnover"].null_count(),
            "mean_turnover_all": timeline["turnover"].mean(),
            "mean_turnover_active": timeline.filter(pl.col("num_fills") > 0)["turnover"].mean(),
        },
        {
            "calendar": calendar,
            "turnover_denom": turnover_denom,
            "turnover_method": turnover_method,
            "cost_denominator": "beginning_nav",
            "returns_kind": "supplied net, observed costs not subtracted again",
            "first_agb": "undefined without a previous complete gross snapshot",
            "unknown_costs": "null, not zero",
            "clock": "exchange sessions" if calendar is not None else "exact supplied observations",
        },
        inputs.returns,
    )
