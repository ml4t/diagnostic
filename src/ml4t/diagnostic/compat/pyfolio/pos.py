"""Familiar pandas position projections over shared native holdings calculations."""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from typing import Any

import pandas as pd
import polars as pl

from ml4t.diagnostic.evaluation.portfolio_analysis.holdings import (
    _exposure_aggregations,
    _top_position_extrema,
    _weight_concentration_aggregations,
)
from ml4t.diagnostic.evaluation.portfolio_analysis.holdings_sectors import _mapping_snapshot
from ml4t.diagnostic.evaluation.portfolio_analysis.transaction_metrics import (
    cash_fraction_expr,
    execution_value_expr,
)

from .txn import _books, _dated


def _frame(positions: pd.DataFrame, *, cash: bool = False) -> pl.DataFrame:
    if not isinstance(positions, pd.DataFrame):
        raise TypeError("positions must be a pandas DataFrame")
    if positions.columns.has_duplicates:
        raise ValueError("position columns must identify distinct assets")
    columns = []
    for ordinal, label in enumerate(positions.columns):
        if label == "cash" and not cash:
            continue
        values = pl.from_pandas(positions.iloc[:, ordinal], nan_to_null=True)
        if values.dtype != pl.Null and not values.dtype.is_numeric():
            raise TypeError("position values must be real numeric values")
        columns.append(
            pl.DataFrame(
                {"row": range(len(positions)), "value": values.cast(pl.Float64)}
            ).with_columns(pl.lit(ordinal, dtype=pl.Int64).alias("asset"))
        )
    return (
        pl.concat(columns)
        if columns
        else pl.DataFrame(schema={"row": pl.Int64, "value": pl.Float64, "asset": pl.Int64})
    )


def _totals(positions: pd.DataFrame) -> pl.DataFrame:
    clock = pl.DataFrame({"row": range(len(positions))}, schema={"row": pl.Int64})
    grouped = _frame(positions).group_by("row").agg(_exposure_aggregations())
    return clock.join(grouped, on="row", how="left").fill_null(0).sort("row")


def get_percent_alloc(values: pd.DataFrame) -> pd.DataFrame:
    """Divide by signed row NAV including cash, omitting NaNs only in that sum.

    Original asset labels and each missing cell remain intact. Zero and negative
    NAV use explicit legacy IEEE division; native positive-NAV defaults do not change.
    """
    frame = _frame(values, cash=True)
    _, nav = _books(values)
    frame = frame.join(
        pl.DataFrame({"row": range(len(values)), "nav": nav.to_numpy()}), on="row", how="left"
    ).with_columns(
        cash_fraction_expr(pl.col("value"), pl.col("nav"), positive_finite=False).alias("weight")
    )
    result = pd.DataFrame(
        index=values.index.copy(deep=True), columns=values.columns.copy(), dtype=float
    )
    for ordinal in range(len(values.columns)):
        result.iloc[:, ordinal] = (
            frame.filter(pl.col("asset") == ordinal).sort("row")["weight"].to_numpy()
        )
    return result


def get_top_long_short_abs(
    positions: pd.DataFrame, top: int = 10
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Shared extrema with legacy input-column tie order and original asset labels."""
    frame = _frame(positions.drop(columns="cash"))
    labels = positions.columns.drop("cash")
    if positions.empty:
        frame = pl.DataFrame(
            {"asset": range(len(labels)), "value": [None] * len(labels)},
            schema={"asset": pl.Int64, "value": pl.Float64},
        )
    outputs = []
    for side in ("long", "short", "absolute"):
        result = _top_position_extrema(frame, top, side=side)
        outputs.append(
            pd.Series(
                result["value"].to_numpy(),
                index=labels.take(result["asset"].to_list()),
                dtype=float,
            )
        )
    return outputs[0], outputs[1], outputs[2]


def get_max_median_position_concentration(positions: pd.DataFrame) -> pd.DataFrame:
    """Positive/negative allocation extrema, including the legacy negative-NAV flip."""
    weights = get_percent_alloc(positions).drop(columns="cash")
    frame = _frame(weights).with_columns(pl.col("value").fill_nan(None).alias("nav_weight"))
    grouped = frame.group_by("row").agg(
        _weight_concentration_aggregations(sign_column="nav_weight")
    )
    result = (
        pl.DataFrame({"row": range(len(positions))}, schema={"row": pl.Int64})
        .join(grouped, on="row", how="left")
        .sort("row")
    )
    return pd.DataFrame(
        dict(
            zip(
                ("max_long", "median_long", "median_short", "max_short"),
                [result[name].to_numpy() for name in result.columns if name != "row"],
                strict=True,
            )
        ),
        index=positions.index.copy(deep=True) if len(positions) else pd.RangeIndex(0),
    )


def get_long_short_pos(positions: pd.DataFrame) -> pd.DataFrame:
    """Signed long, short and net allocations; missing account cash stays missing."""
    cash = positions["cash"]
    totals = _totals(positions).with_columns(pl.from_pandas(cash, nan_to_null=True).alias("cash"))
    denominator = pl.col("net_exposure") + pl.col("cash")
    result = totals.select(
        cash_fraction_expr(pl.col("long_exposure"), denominator, positive_finite=False).alias(
            "long"
        ),
        cash_fraction_expr(-pl.col("short_exposure"), denominator, positive_finite=False).alias(
            "short"
        ),
    ).with_columns((pl.col("long") + pl.col("short")).alias("net exposure"))
    return pd.DataFrame(
        {name: result[name].to_numpy() for name in result.columns},
        index=positions.index.copy(deep=True),
    )


def get_sector_exposures(
    positions: pd.DataFrame, symbol_sector_map: Mapping[Any, Any] | pd.Series
) -> pd.DataFrame:
    """Group supplied marked dollars, excluding/warning on unmapped assets.

    The legacy wide schema reserves cash for account cash. A sector named cash
    is rejected instead of silently overwriting its exposure. Native sector
    results keep sector labels and account cash in separate namespaces.
    """
    cash = positions["cash"]
    labels = positions.columns.drop("cash")
    column_name = (
        symbol_sector_map.name
        if isinstance(symbol_sector_map, pd.Series)
        else positions.columns.name
    )
    if isinstance(symbol_sector_map, pd.Series):
        if not symbol_sector_map.index.is_unique:
            raise ValueError("sector mapping keys must be unique")
        symbol_sector_map = symbol_sector_map.to_dict()
    if not isinstance(symbol_sector_map, Mapping):
        raise TypeError("symbol_sector_map must be a mapping or pandas Series")
    mapping = _mapping_snapshot(
        {i: symbol_sector_map.get(label) for i, label in enumerate(labels)}, pl.Int64
    )
    if mapping["sector"].dtype == pl.String and (mapping["sector"] == "cash").any():
        raise ValueError(
            "sector label cash conflicts with reserved account cash; use native sector results"
        )
    unknown = [str(label) for label in labels if label not in symbol_sector_map]
    if unknown:
        warnings.warn(
            "Symbols "
            + ", ".join(unknown)
            + " have no sector mapping and are excluded from sector allocations",
            UserWarning,
            stacklevel=2,
        )
    frame = _frame(positions.drop(columns="cash")).join(mapping, on="asset", how="left")
    grouped = (
        frame.filter(pl.col("sector").is_not_null())
        .group_by("row", "sector")
        .agg(_exposure_aggregations())
        .sort("row", "sector")
    )
    result = pd.DataFrame(index=positions.index.copy(deep=True))
    for label in mapping["sector"].drop_nulls().unique().sort().to_list():
        group = grouped.filter(pl.col("sector") == label).sort("row")
        result[label] = pd.Series(group["net_exposure"].to_numpy(), index=positions.index)
    result.columns.name = column_name
    result["cash"] = cash.copy(deep=True)
    return result


def extract_pos(positions: pd.DataFrame, cash: pd.Series) -> pd.DataFrame:
    """Project actual amount/mark records without inventing returns or losing cash dates.

    Multipliers declared by asset objects are honored without importing Zipline.
    Repeated date/asset records retain the original pivot-mean convention. The
    supplied cash clock is complete; absent assets on that clock are explicitly zero.
    """
    from ml4t.diagnostic.integration.zipline import _asset

    _dated(positions, "positions")
    _dated(cash, "cash")
    if not cash.index.is_unique or not cash.index.is_monotonic_increasing:
        raise ValueError("cash requires unique chronological observation dates")
    if positions.index.tz != cash.index.tz or not positions.index.isin(cash.index).all():
        raise ValueError("cash must cover every position observation with the same timezone")
    required = {"sid", "amount", "last_sale_price"}
    if missing := required.difference(positions.columns):
        raise ValueError(f"positions require columns: {sorted(missing)}")
    codes, labels = pd.factorize(positions["sid"], sort=True)
    if (codes < 0).any() or any(label == "cash" for label in labels):
        raise ValueError("position asset IDs must be nonmissing and distinct from reserved cash")
    multipliers = [_asset(label, {})[2] if hasattr(label, "sid") else 1.0 for label in labels]
    frame = pl.DataFrame(
        {
            "quantity": pl.from_pandas(positions["amount"]).cast(pl.Float64),
            "price": pl.from_pandas(positions["last_sale_price"]).cast(pl.Float64),
            "multiplier": pl.Series([multipliers[code] for code in codes], dtype=pl.Float64),
        }
    )
    marked = frame.select(execution_value_expr().alias("value"))["value"].to_numpy()
    owned = pd.DataFrame({"asset": codes, "value": marked}, index=positions.index.copy(deep=True))
    projected = owned.pivot_table(index=owned.index, columns="asset", values="value")
    projected = projected.rename(columns=dict(enumerate(labels)))
    result = projected.reindex(cash.index).fillna(0.0)
    result["cash"] = cash.copy(deep=True)
    result.index = cash.index.copy(deep=True)
    result.index.freq = None
    result.columns.name = "sid"
    return result


__all__ = [
    "get_percent_alloc",
    "get_top_long_short_abs",
    "get_max_median_position_concentration",
    "get_long_short_pos",
    "get_sector_exposures",
    "extract_pos",
]
