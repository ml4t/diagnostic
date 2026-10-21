"""Observed complete holdings snapshots over the shared occupancy calculations."""

from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Integral
from typing import Any, Literal

import pandas as pd
import polars as pl

from .inputs import date_index, date_snapshot


@dataclass
class HoldingsAnalysisResult:
    """Owned raw holdings, account timeline and defined-only occupancy summary.

    Short exposure is a positive magnitude; signed per-asset NAV weights remain
    negative for shorts. Concentration uses absolute invested capital, excluding
    cash. An all-cash observation has zero exposure but undefined concentration.
    Counts and averages describe supplied observations, not inferred sessions.
    Optional sectors contains asset-sector aggregates; null labels retain unknown
    classifications. Account cash stays in timeline, outside the sector namespace.
    A None sector table means not requested; a requested all-cash table is empty.
    Optional metadata retains method/calendar provenance for independent-clock
    consumers without implying return observations or continuous-time coverage.
    """

    positions: pl.DataFrame
    timeline: pl.DataFrame
    metrics: dict[str, Any]
    coverage: dict[str, Any]
    snapshot_kind: str
    sectors: pl.DataFrame | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def top_positions(
        self, top: int = 10, *, side: Literal["long", "short", "absolute"] = "absolute"
    ) -> pl.DataFrame:
        """Return asset/value extrema across supplied observations, excluding cash.

        Long uses each asset's largest positive marked value; short uses its most
        negative value. Absolute uses its largest magnitude, including explicitly
        zero-only assets. No NAV normalization or time weighting is applied.
        Ties sort by stable asset ID. Zero top returns a typed empty table.
        """
        if isinstance(top, bool) or not isinstance(top, Integral):
            raise TypeError("top must be a nonnegative integer")
        if top < 0:
            raise ValueError("top must be a nonnegative integer")
        if side not in ("long", "short", "absolute"):
            raise ValueError("side must be long, short or absolute")
        values = pl.col("value")
        extreme = {
            "long": values.max(),
            "short": values.min(),
            "absolute": values.abs().max(),
        }[side]
        result = self.positions.group_by("asset").agg(extreme.alias("value"))
        if side != "absolute":
            result = result.filter(pl.col("value") > 0 if side == "long" else pl.col("value") < 0)
        return result.sort(["value", "asset"], descending=[side != "short", False]).head(
            min(int(top), result.height)
        )


def _exposure_aggregations() -> list[pl.Expr]:
    """Use the same marked-value totals for portfolios and their classifications."""
    values = pl.col("value")
    return [
        values.clip(lower_bound=0).abs().sum().alias("long_exposure"),
        values.clip(upper_bound=0).abs().sum().alias("short_exposure"),
        values.abs().sum().alias("gross_exposure"),
        values.sum().alias("net_exposure"),
        (values != 0).sum().alias("open_positions"),
        (values > 0).sum().alias("long_positions"),
        (values < 0).sum().alias("short_positions"),
    ]


def position_snapshot(frame: pl.DataFrame | pd.DataFrame) -> pl.DataFrame:
    if isinstance(frame, pl.DataFrame):
        return frame.clone()
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("positions must be a pandas or Polars DataFrame")
    if any(not isinstance(column, str) for column in frame.columns):
        raise TypeError("wide position columns must be string asset IDs; use long format otherwise")
    owned = pl.from_pandas(frame.copy(deep=True))
    if isinstance(frame.index, pd.DatetimeIndex):
        dates = pl.from_pandas(frame.index.copy(deep=True)).rename("date")
        if "date" in owned.columns:
            actual, expected = date_index(owned["date"]), date_index(dates)
            if actual.tz != expected.tz or not (actual == expected).all():
                raise ValueError("position index and date column must agree")
        owned = owned.with_columns(dates)
    return owned


def _same_dates(actual: pl.Series, expected: pl.Series, name: str) -> None:
    actual = date_snapshot(actual, len(expected), f"{name} dates")
    left, right = date_index(actual), date_index(expected)
    if left.tz != right.tz or not (left == right).all():
        raise ValueError(f"{name} dates must match the exact return observation keys and timezone")


def _numeric(frame: pl.DataFrame, column: str, *, nullable: bool = False) -> pl.DataFrame:
    if nullable and frame[column].null_count() == frame.height:
        return frame.with_columns(pl.col(column).cast(pl.Float64))
    if not frame[column].dtype.is_numeric():
        raise TypeError(f"{column} must contain real numeric values")
    frame = frame.with_columns(pl.col(column).cast(pl.Float64).fill_nan(None))
    values = frame[column]
    if (not nullable and values.null_count()) or not values.drop_nulls().is_finite().all():
        raise ValueError(f"{column} must contain finite values; missing positions are not zero")
    return frame


def analyze_holdings(
    positions: pl.DataFrame,
    valuations: pl.DataFrame | pd.DataFrame,
    dates: pl.Series,
    *,
    complete_snapshots: bool,
    snapshot_kind: Literal["observed", "end_of_day", "intraday"],
) -> HoldingsAnalysisResult:
    from ml4t.diagnostic.integration.backtest_analytics import compute_occupancy_metrics

    if complete_snapshots is not True:
        raise ValueError(
            "complete_snapshots=True must explicitly declare complete holdings snapshots"
        )
    if snapshot_kind not in {"observed", "end_of_day", "intraday"}:
        raise ValueError("snapshot_kind must be observed, end_of_day or intraday")
    account = position_snapshot(valuations)
    if "date" not in account.columns or "date" not in positions.columns:
        raise ValueError("positions and valuations require explicit date observation keys")
    _same_dates(account["date"], dates, "valuation")
    account = account.with_columns(pl.col("date").cast(dates.dtype))
    account = account.select(
        "date", *[name for name in ["equity", "cash"] if name in account.columns]
    )
    for name in ["equity", "cash"]:
        if name not in account.columns:
            account = account.with_columns(pl.lit(None, dtype=pl.Float64).alias(name))
        account = _numeric(account, name, nullable=True)

    if "asset" in positions.columns or "value" in positions.columns:
        if not {"date", "asset", "value"}.issubset(positions.columns):
            raise ValueError("long positions require date, asset and value columns")
        long = positions.select("date", "asset", "value")
        asset = long["asset"]
        if asset.dtype != pl.Utf8 and not asset.dtype.is_integer():
            raise TypeError("asset IDs must be stable strings or integers")
        if asset.null_count() or (asset.dtype == pl.Utf8 and (asset == "").any()):
            raise ValueError("asset IDs must be nonmissing and nonempty")
        snapshot_dates = long["date"].unique().sort()
    else:
        _same_dates(positions["date"], dates, "position")
        assets = [name for name in positions.columns if name not in {"date", "cash"}]
        for asset in assets:
            positions = _numeric(positions, asset)
        if "cash" in positions.columns:
            cash = _numeric(positions.select("cash"), "cash", nullable=True)["cash"]
            supplied = account["cash"]
            if ((supplied.is_not_null()) & (cash.is_null() | (supplied != cash))).any():
                raise ValueError("cash in positions must agree with reported valuation cash")
            account = account.with_columns(pl.coalesce(supplied, cash).alias("cash"))
        if not assets:
            long = pl.DataFrame(schema={"date": dates.dtype, "asset": pl.Utf8, "value": pl.Float64})
        elif hasattr(positions, "unpivot"):
            long = positions.unpivot(on=assets, index="date", variable_name="asset")
        else:
            long = positions.melt(id_vars="date", value_vars=assets, variable_name="asset")
        snapshot_dates = positions["date"]
    _same_dates(snapshot_dates, dates, "position observation")
    long = long.with_columns(pl.col("date").cast(dates.dtype))
    if long.select(pl.struct("date", "asset").is_duplicated().any()).item():
        raise ValueError("position date/asset pairs must be unique")
    long = _numeric(long, "value")
    values = pl.col("value")
    aggregates = long.group_by("date").agg(_exposure_aggregations())
    raw_columns = [name for name in aggregates.columns if name != "date"]
    state = account.join(aggregates, on="date", how="left").with_columns(
        pl.col(raw_columns).fill_null(0)
    )
    occupancy = compute_occupancy_metrics(state.rename({"date": "timestamp"}))
    timeline = occupancy["timeline"].rename({"timestamp": "date"})
    defined = pl.col("normalization_status") == "defined"
    timeline = timeline.with_columns(
        *[
            pl.when(defined)
            .then(pl.col(name) / pl.col("equity"))
            .otherwise(None)
            .alias(f"{name}_fraction")
            for name in ["long_exposure", "short_exposure", "cash"]
        ]
    )
    long = long.join(
        timeline.select("date", "equity", "gross_exposure", "normalization_status"),
        on="date",
        how="left",
    ).with_columns(
        pl.when(pl.col("gross_exposure") > 0)
        .then(values.abs() / pl.col("gross_exposure"))
        .otherwise(None)
        .alias("gross_share"),
        pl.when(defined).then(values / pl.col("equity")).otherwise(None).alias("nav_weight"),
        pl.when(values != 0)
        .then(values.abs().rank(method="dense", descending=True).over("date"))
        .otherwise(None)
        .alias("gross_rank"),
    )
    concentration = long.group_by("date").agg(
        pl.col("gross_share").max().alias("largest_gross_share"),
        pl.when(pl.col("gross_share").count() > 0)
        .then(pl.col("gross_share").pow(2).sum())
        .otherwise(None)
        .alias("gross_hhi"),
        pl.col("nav_weight").filter(values > 0).max().alias("max_long_weight"),
        pl.col("nav_weight").filter(values > 0).median().alias("median_long_weight"),
        pl.col("nav_weight").filter(values < 0).median().alias("median_short_weight"),
        pl.col("nav_weight").filter(values < 0).min().alias("min_short_weight"),
    )
    timeline = timeline.join(concentration, on="date", how="left")
    return HoldingsAnalysisResult(
        positions=long.select("date", "asset", "value", "nav_weight", "gross_share", "gross_rank"),
        timeline=timeline,
        metrics=occupancy["metrics"],
        coverage=occupancy["coverage"],
        snapshot_kind=snapshot_kind,
    )
