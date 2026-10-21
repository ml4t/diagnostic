"""Supplied-market liquidity and declared volume-share impact scenarios.

No orders, provider data or account state are generated here. Marked position
values and market price times volume times multiplier must share a currency.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from io import BytesIO
from numbers import Real
from typing import Any, Literal

import numpy as np
import pandas as pd
import polars as pl
from ml4t.specs import UNSET_CALENDAR, CalendarUnset, FeedSpec

from ml4t.diagnostic.utils.validation import (
    _align_datetime_key_resolution,
    _validate_observation_keys,
)

from .frame_inputs import _frame, _rename, _temporal
from .holdings import _numeric
from .inputs import positive_integer
from .transactions import _clock

_TABLES = (
    "liquidation",
    "participation",
    "timeline",
    "worst_positions",
    "worst_transactions",
    "scenarios",
)


@dataclass(frozen=True)
class CapacityAnalysisResult:
    """Owned dated inputs/intermediates, coverage and hypothetical scenarios.

    Liquidation durations count selected market observations or sessions, not
    elapsed days. A known zero-volume open position is unbounded; missing market
    coverage is unavailable. Book liquidation assumes parallel asset liquidation
    with an independent participation budget per asset, not a shared order book.
    """

    _tables: tuple[bytes, ...] = field(repr=False)
    _metadata: str = field(repr=False)

    def _table(self, name: str) -> pl.DataFrame:
        return pl.read_ipc(BytesIO(self._tables[_TABLES.index(name)]))

    @property
    def liquidation(self) -> pl.DataFrame:
        """Dated signed marked values and their lagged market denominators."""
        return self._table("liquidation")

    @property
    def participation(self) -> pl.DataFrame:
        """Actual absolute supplied units and same-bar retrospective participation."""
        return self._table("participation")

    @property
    def timeline(self) -> pl.DataFrame:
        """Complete-key book duration and explicit missing-market counts."""
        return self._table("timeline")

    @property
    def worst_positions(self) -> pl.DataFrame:
        """Per-asset full/recent extrema, retaining unavailable assets."""
        return self._table("worst_positions")

    @property
    def worst_transactions(self) -> pl.DataFrame:
        """Per-asset full/recent participation extrema."""
        return self._table("worst_transactions")

    @property
    def scenarios(self) -> pl.DataFrame:
        """Capital-scaled extra impact; never a claim of executable fills."""
        return self._table("scenarios")

    @property
    def metadata(self) -> dict[str, Any]:
        """Independent declarations and missing/undefined reasons."""
        return json.loads(self._metadata)


def _positive_real(value: float, name: str, *, zero: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value):
        raise ValueError(f"{name} must be finite and {'nonnegative' if zero else 'positive'}")
    if value < 0 if zero else value <= 0:
        raise ValueError(f"{name} must be {'nonnegative' if zero else 'positive'}")
    return float(value)


def _lagged_dollar_volume(values: pd.DataFrame, lookback: int) -> pd.DataFrame:
    return values.rolling(lookback, min_periods=lookback).mean().shift(1)


def _liquidation_values(
    values: np.ndarray, dollar_volume: np.ndarray, participation: float
) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        result = np.abs(values) / (participation * dollar_volume)
    return np.where(values == 0, 0.0, result)


def _volume_share_cost(
    units: np.ndarray,
    price: np.ndarray,
    volume: np.ndarray,
    multiplier: np.ndarray,
    capital_ratio: float,
    impact: float,
) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        scaled = np.abs(units) * capital_ratio
        result = (scaled / volume) ** 2 * impact * price * scaled * multiplier
    return np.where((units == 0) | (impact == 0), 0.0, result)


def _worst(frame: pl.DataFrame, value: str, recent: int | None) -> pl.DataFrame:
    outputs = []
    dates = frame["date"].unique().sort()
    for label, selected in (
        ("full", frame),
        (
            "recent",
            frame.filter(pl.col("date").is_in(dates.tail(recent).implode())) if recent else frame,
        ),
    ):
        selected = selected.sort([value, "date"], descending=[True, False], nulls_last=True)
        outputs.append(
            selected.unique("asset", keep="first", maintain_order=True).with_columns(
                pl.lit(label).alias("sample")
            )
        )
    return pl.concat(outputs)


def analyze_capacity(
    positions: pl.DataFrame | pd.DataFrame,
    market_data: pl.DataFrame | pd.DataFrame,
    *,
    currency: str,
    price_basis: str,
    volume_basis: str,
    complete_snapshots: bool,
    transactions: pl.DataFrame | pd.DataFrame | None = None,
    returns: pl.DataFrame | pd.DataFrame | None = None,
    participation: float = 0.1,
    lookback: int = 5,
    window_unit: Literal["observation", "session"] = "observation",
    recent_window: int | None = 126,
    feed_spec: FeedSpec | Mapping[str, Any] | None = None,
    timestamp_col: str | None = None,
    entity_col: str | None = None,
    calendar: str | CalendarUnset | None = UNSET_CALENDAR,
    known_at_col: str | None = None,
    base_capital: float | None = None,
    scenario_capitals: Sequence[float] = (),
    impact: float = 0.1,
) -> CapacityAnalysisResult:
    """Analyze canonical long positions and mapped supplied market bars.

    Positions use date/asset/value (signed account-currency marks). Transactions
    use timestamp/asset/quantity, with optional multiplier; they must be actual
    complete records, not inferred from weights. Market FeedSpec maps timestamp,
    asset, price and volume. A market multiplier column defaults to one. Callers
    declare compatible price/volume adjustment basis and one common currency;
    this function does not perform FX or corporate-action adjustments.

    Lookback uses mean(price*volume*multiplier).shift(1) on the common selected
    clock. Session windows include missing scheduled rows; observation windows
    include the supplied market/position union, without compressing asset holes.
    Named calendars map bars to sessions; None retains exact observation keys.
    With no known_at_col, supplied bar timestamps declare availability. Delayed
    observations remain unavailable until their supplied known-at time.

    Participation is retrospective same-bar absolute units/volume. Impact uses
    closing-mark notional and a declared quadratic volume-share coefficient,
    not an empirically fitted execution model. Capital scenarios require supplied
    simple returns and beginning capital, subtract extra costs once, and use
    beginning-period NAV. Legacy closing-NAV scenarios are separately named in
    the Pyfolio adapter. Missing active market data never becomes a free trade.
    """
    if complete_snapshots is not True:
        raise ValueError("complete_snapshots=True must declare the supplied holdings complete")
    lookback = positive_integer(lookback, "lookback")
    recent_window = (
        positive_integer(recent_window, "recent_window") if recent_window is not None else None
    )
    participation = _positive_real(participation, "participation")
    if participation > 1:
        raise ValueError("participation must not exceed one")
    impact = _positive_real(impact, "impact", zero=True)
    if not isinstance(currency, str) or not currency.strip():
        raise ValueError("currency must declare the common account/market currency")
    if not isinstance(price_basis, str) or not price_basis or price_basis != volume_basis:
        raise ValueError("price_basis and volume_basis must declare the same adjustment basis")
    feed = (
        FeedSpec.from_any(feed_spec)
        if feed_spec is not None
        else FeedSpec(
            timestamp_col="date", entity_col="asset", price_col="price", time_contract_version=2
        )
    )
    if timestamp_col is not None and feed_spec is not None and timestamp_col != feed.timestamp_col:
        raise ValueError("timestamp_col conflicts with feed_spec")
    if entity_col is not None and feed_spec is not None and entity_col != feed.entity_col:
        raise ValueError("entity_col conflicts with feed_spec")
    entity = entity_col or feed.entity_col
    if not isinstance(entity, str):
        raise ValueError("market capacity requires one declared stable entity column")
    effective_calendar = feed.resolve_calendar(calendar=calendar)
    if (
        window_unit not in ("observation", "session")
        or window_unit == "session"
        and effective_calendar is None
    ):
        raise ValueError("window_unit must be observation or session; session requires a calendar")
    marked = _frame(positions, "date")
    market = _frame(market_data, timestamp_col or feed.timestamp_col)
    market = _rename(
        _rename(_rename(market, entity, "asset"), feed.price_col, "price"),
        feed.volume_col,
        "volume",
    )
    for name, frame in (("positions", marked), ("market", market)):
        _temporal(frame)
        _validate_observation_keys(frame, ["date", "asset"], name)
    if market.is_empty() or marked.is_empty() and returns is None:
        raise ValueError("market_data and a complete position/return clock are required")
    if marked["asset"].dtype != market["asset"].dtype:
        raise TypeError("position and market asset IDs must have the same stable type")
    marked = _numeric(marked, "value")
    for name in ("price", "volume"):
        market = _numeric(market, name, nullable=True)
    if market.filter((pl.col("price") <= 0) | (pl.col("volume") < 0)).height:
        raise ValueError("observed market prices must be positive and volume nonnegative")
    if "multiplier" not in market.columns:
        market = market.with_columns(pl.lit(1.0).alias("multiplier"))
    market = _numeric(market, "multiplier")
    if market.filter(pl.col("multiplier") <= 0).height:
        raise ValueError("market multipliers must be positive")
    if known_at_col is not None and known_at_col not in market.columns:
        raise ValueError("market data is missing the declared known-at column")
    market = market.with_columns(pl.col(known_at_col or "date").alias("known_at"))
    _validate_observation_keys(market, ["known_at"], "market availability", unique=False)
    marked, market = _align_datetime_key_resolution(marked, market, "date")
    if (
        marked["date"].dtype != market["date"].dtype
        or market["known_at"].dtype != market["date"].dtype
    ):
        raise TypeError(
            "market/position/known-at keys must have matching temporal types and timezone"
        )
    semantics = getattr(feed.timestamp_semantics, "value", feed.timestamp_semantics)

    def clock(frame: pl.DataFrame, *, executions: bool = False) -> pl.DataFrame:
        return frame.with_columns(
            _clock(
                frame["date"],
                effective_calendar,
                semantics,
                executions=executions,
                timezone=feed.timezone,
            ).alias("clock")
        )

    marked, market = clock(marked), clock(market)
    for name, frame in (("positions", marked), ("market", market)):
        _validate_observation_keys(frame, ["clock", "asset"], name)
    if returns is not None:
        observed = _frame(returns, "date")
        _temporal(observed)
        _validate_observation_keys(observed, ["date"], "returns")
        observed = clock(observed).sort("date")
        decisions = observed.select("clock", "date")
        if not marked["date"].is_in(observed["date"].implode()).all():
            raise ValueError("position keys must belong to the supplied return observation clock")
    else:
        decisions = marked.select("clock", "date").unique().sort("clock")
    if decisions["clock"].n_unique() != decisions.height:
        raise ValueError("capacity requires one position decision instant per selected clock row")
    grid = pd.DatetimeIndex(
        pd.concat([market["clock"].to_pandas(), decisions["clock"].to_pandas()]).unique()
    ).sort_values()
    if window_unit == "session":
        import pandas_market_calendars as mcal

        grid = pd.DatetimeIndex(
            mcal.get_calendar(effective_calendar).schedule(grid.min(), grid.max()).index
        )
    assets = pd.Index(marked["asset"].unique(maintain_order=True).to_list())
    supplied = market.to_pandas().set_index(["clock", "asset"])

    def wide(column: str) -> pd.DataFrame:
        return supplied[column].unstack("asset").reindex(index=grid, columns=assets)

    dollar_volume = wide("price") * wide("volume") * wide("multiplier")
    if np.isinf(dollar_volume.to_numpy()).any():
        raise ValueError("market dollar volume overflows")
    lagged = _lagged_dollar_volume(dollar_volume, lookback)
    availability = wide("known_at")
    cutoffs = decisions.to_pandas().set_index("clock")["date"].reindex(grid)
    for lag in range(1, lookback + 1):
        lagged = lagged.mask(availability.shift(lag).gt(cutoffs, axis=0))
    lookup = {
        (pd.Timestamp(day), asset): value
        for day, row in lagged.iterrows()
        for asset, value in row.items()
    }
    denominators = np.array(
        [
            lookup.get((pd.Timestamp(day), asset), np.nan)
            for day, asset in zip(
                marked["clock"].to_pandas(), marked["asset"].to_list(), strict=True
            )
        ]
    )
    durations = _liquidation_values(marked["value"].to_numpy(), denominators, participation)
    liquidation = marked.with_columns(
        pl.Series("lagged_dollar_volume", denominators, nan_to_null=True),
        pl.Series("liquidation_duration", durations, nan_to_null=True),
    ).sort(["date", "asset"])
    timeline = liquidation.group_by("date", maintain_order=True).agg(
        (pl.col("value") != 0).sum().alias("open_positions"),
        ((pl.col("value") != 0) & pl.col("liquidation_duration").is_null())
        .sum()
        .alias("unavailable_positions"),
        pl.when(pl.col("liquidation_duration").null_count() == 0)
        .then(pl.col("liquidation_duration").max())
        .otherwise(None)
        .alias("book_liquidation_duration"),
    )
    timeline = (
        decisions.select("date")
        .join(timeline, on="date", how="left")
        .with_columns(
            pl.when(pl.col("open_positions").is_null())
            .then(0.0)
            .otherwise(pl.col("book_liquidation_duration"))
            .alias("book_liquidation_duration"),
            pl.col("open_positions").fill_null(0),
            pl.col("unavailable_positions").fill_null(0),
        )
    )
    activity = pl.DataFrame(
        schema={
            "date": marked["date"].dtype,
            "asset": marked["asset"].dtype,
            "absolute_units": pl.Float64,
            "price": pl.Float64,
            "volume": pl.Float64,
            "multiplier": pl.Float64,
            "participation": pl.Float64,
        }
    )
    scenarios = pl.DataFrame(
        schema={
            "date": marked["date"].dtype,
            "capital": pl.Float64,
            "extra_impact_cash": pl.Float64,
            "extra_impact_return": pl.Float64,
            "adjusted_return": pl.Float64,
        }
    )
    capitals = [_positive_real(value, "scenario capital") for value in scenario_capitals]
    if len(set(capitals)) != len(capitals):
        raise ValueError("scenario_capitals must be distinct")
    if transactions is not None:
        fills = _frame(transactions, "timestamp")
        _temporal(fills)
        _validate_observation_keys(fills, ["date", "asset"], "transactions", unique=False)
        fills = _numeric(fills, "quantity")
        if fills["asset"].dtype != marked["asset"].dtype:
            raise TypeError("transaction asset IDs must match positions")
        fills = clock(fills, executions=True)
        if not fills["clock"].is_in(decisions["clock"].implode()).all():
            raise ValueError("transactions have no matching position/return observation clock")
        if "multiplier" in fills.columns:
            fills = _numeric(fills, "multiplier")
            check = fills.join(
                market.select("clock", "asset", pl.col("multiplier").alias("market_multiplier")),
                on=["clock", "asset"],
                how="left",
            )
            if check.filter(
                (pl.col("multiplier") <= 0) | (pl.col("multiplier") != pl.col("market_multiplier"))
            ).height:
                raise ValueError("transaction and market multipliers must be positive and agree")
        activity = fills.group_by("clock", "asset").agg(
            pl.col("quantity").abs().sum().alias("absolute_units")
        )
        activity = activity.join(
            market.select("clock", "asset", "price", "volume", "multiplier"),
            on=["clock", "asset"],
            how="left",
        ).rename({"clock": "date"})
        with np.errstate(divide="ignore", invalid="ignore"):
            ratios = activity["absolute_units"].to_numpy() / activity["volume"].to_numpy()
        activity = activity.with_columns(pl.Series("participation", ratios, nan_to_null=True)).sort(
            ["date", "asset"]
        )
    if capitals:
        if returns is None or transactions is None or base_capital is None:
            raise ValueError(
                "capital scenarios require returns, actual transactions and base_capital"
            )
        starting = _positive_real(base_capital, "base_capital")
        observed = _numeric(_frame(returns, "date"), "return")
        _validate_observation_keys(observed, ["date"], "returns")
        if not observed["date"].equals(decisions["date"]):
            raise ValueError(
                "scenario returns must match the sorted complete position decision keys"
            )
        values = observed["return"].to_numpy()
        if np.any(values <= -1):
            raise ValueError("capital scenarios require strictly positive beginning NAV")
        beginning = starting * np.cumprod(np.r_[1.0, 1 + values[:-1]])
        scenario_tables = []
        for capital in capitals:
            costs = _volume_share_cost(
                activity["absolute_units"].to_numpy(),
                activity["price"].to_numpy(),
                activity["volume"].to_numpy(),
                activity["multiplier"].to_numpy(),
                capital / starting,
                impact,
            )
            daily = (
                activity.with_columns(pl.Series("extra_impact_cash", costs, nan_to_null=True))
                .group_by("date")
                .agg(
                    pl.when(pl.col("extra_impact_cash").null_count() == 0)
                    .then(pl.col("extra_impact_cash").sum())
                    .otherwise(None)
                    .alias("extra_impact_cash")
                )
            )
            decision_clock = decisions.rename({"date": "observation", "clock": "date"})
            daily = decision_clock.join(
                daily.with_columns(pl.lit(True).alias("had_fills")), on="date", how="left"
            ).with_columns(
                pl.when(pl.col("had_fills").is_null())
                .then(0.0)
                .otherwise(pl.col("extra_impact_cash"))
                .alias("extra_impact_cash")
            )
            drag = daily["extra_impact_cash"].to_numpy() / (beginning * capital / starting)
            scenario_tables.append(
                observed.with_columns(
                    pl.lit(capital).alias("capital"),
                    daily["extra_impact_cash"],
                    pl.Series("extra_impact_return", drag, nan_to_null=True),
                    pl.Series("adjusted_return", values - drag, nan_to_null=True),
                ).drop("return")
            )
        scenarios = pl.concat(scenario_tables)
    metadata = {
        "currency": currency,
        "price_basis": price_basis,
        "volume_basis": volume_basis,
        "basis_validation": "caller-declared compatible units; no FX/adjustment engine",
        "feed_spec": feed.to_dict(),
        "calendar": effective_calendar,
        "window_unit": window_unit,
        "lookback": lookback,
        "availability_lag": 1,
        "participation": participation,
        "recent_window": recent_window,
        "impact_coefficient": impact,
        "model": "declared quadratic volume-share, closing-mark notional, extra cost only",
        "calibration": "hypothetical coefficient, not estimated or execution-guaranteed",
        "scenario_nav": "beginning_period",
        "transactions_supplied": transactions is not None,
        "complete_snapshots": True,
        "missing_liquidation_rows": liquidation["liquidation_duration"].null_count(),
        "unbounded_liquidation_rows": liquidation["liquidation_duration"].is_infinite().sum(),
        "book_assumption": "parallel liquidation with independent per-asset participation budgets",
    }
    tables = (
        liquidation,
        activity,
        timeline,
        _worst(liquidation, "liquidation_duration", recent_window),
        _worst(activity, "participation", recent_window),
        scenarios,
    )
    return CapacityAnalysisResult(
        tuple(table.write_ipc(None).getvalue() for table in tables),
        json.dumps(metadata, allow_nan=False),
    )
