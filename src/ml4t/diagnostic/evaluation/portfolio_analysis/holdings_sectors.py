"""Sector aggregation on owned holdings and their existing normalization status."""

from __future__ import annotations

from collections.abc import Mapping
from math import isfinite, isnan
from numbers import Integral, Real

import pandas as pd
import polars as pl

from .holdings import _exposure_aggregations, position_snapshot
from .inputs import date_index, date_snapshot

type SectorMapping = (
    Mapping[str | int, str | int | float | None] | pd.Series | pd.DataFrame | pl.DataFrame
)


def _mapping_labels(values: list) -> pl.Series:
    labels = [
        None
        if value is None
        or value is pd.NA
        or (isinstance(value, Real) and not isinstance(value, Integral) and isnan(value))
        else value
        for value in values
    ]
    present = [value for value in labels if value is not None]
    if not present or all(isinstance(value, str) for value in present):
        return pl.Series("sector", labels, dtype=pl.Utf8)
    if not all(isinstance(value, Real) and not isinstance(value, bool) for value in present):
        raise TypeError("sector labels must be homogeneous strings or real numbers, or missing")
    if not all(isfinite(value) for value in present):
        raise ValueError("sector labels must be finite or missing")
    if any(not isinstance(value, Integral) for value in present):
        if any(isinstance(value, Integral) and float(value) != value for value in present):
            raise ValueError("mixed numeric sector labels must preserve exact integer identity")
        return pl.Series("sector", labels, dtype=pl.Float64)
    return pl.Series("sector", labels)


def _mapping_snapshot(mapping: SectorMapping, asset_dtype: pl.DataType) -> pl.DataFrame:
    if isinstance(mapping, (pd.DataFrame, pl.DataFrame)):
        frame = position_snapshot(mapping)
    elif isinstance(mapping, pd.Series):
        frame = position_snapshot(pd.DataFrame({"asset": mapping.index, "sector": mapping.array}))
    elif isinstance(mapping, Mapping):
        ids = list(mapping)
        valid = (
            all(isinstance(key, str) for key in ids)
            if asset_dtype == pl.Utf8
            else all(isinstance(key, Integral) and not isinstance(key, bool) for key in ids)
        )
        if not valid:
            raise TypeError("sector mapping asset IDs must match the holdings string/integer type")
        frame = pl.DataFrame(
            {
                "asset": pl.Series("asset", ids, dtype=asset_dtype, strict=True),
                "sector": _mapping_labels([mapping[key] for key in ids]),
            }
        )
    else:
        raise TypeError("sector_mapping must be a mapping, Series or DataFrame")
    if not {"asset", "sector"}.issubset(frame.columns) or set(frame.columns) - {
        "date",
        "asset",
        "sector",
    }:
        raise ValueError("sector mapping columns must be asset, sector and optional date")
    if frame["asset"].null_count():
        raise ValueError("sector mapping asset IDs must be nonmissing")
    actual = frame["asset"].dtype
    if actual == pl.Categorical:
        frame = frame.with_columns(pl.col("asset").cast(pl.Utf8))
        actual = pl.Utf8
    if actual != pl.Null and not (
        actual == asset_dtype == pl.Utf8 or actual.is_integer() and asset_dtype.is_integer()
    ):
        raise TypeError("sector mapping asset IDs must match the holdings string/integer type")
    frame = frame.with_columns(pl.col("asset").cast(asset_dtype, strict=True))
    if asset_dtype == pl.Utf8 and (frame["asset"] == "").any():
        raise ValueError("sector mapping asset IDs must be nonempty")
    sector_dtype = frame["sector"].dtype
    if sector_dtype == pl.Null or sector_dtype == pl.Categorical:
        frame = frame.with_columns(pl.col("sector").cast(pl.Utf8))
    elif sector_dtype.is_float():
        frame = frame.with_columns(pl.col("sector").fill_nan(None))
    sector = frame["sector"]
    if sector.dtype == pl.Utf8:
        if (sector == "").any():
            raise ValueError("sector labels must be nonempty or missing")
    elif not sector.dtype.is_numeric():
        raise TypeError("sector labels must be strings or real numbers, or missing")
    elif not sector.drop_nulls().is_finite().all():
        raise ValueError("sector labels must be finite or missing")
    return frame


def analyze_sector_exposures(
    positions: pl.DataFrame, timeline: pl.DataFrame, mapping: SectorMapping
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Retain unknown classes and keep account cash in the separate timeline."""
    frame = _mapping_snapshot(mapping, positions.schema["asset"])
    keys = ["asset"]
    if "date" in frame.columns:
        if frame.height:
            unique = frame["date"].unique().sort()
            date_snapshot(unique, len(unique), "sector mapping dates")
            if date_index(unique).tz != date_index(timeline["date"]).tz:
                raise ValueError("sector mapping dates and holdings must use the same timezone")
            converted = frame["date"].cast(timeline.schema["date"])
            if not (date_index(converted) == date_index(frame["date"])).all():
                raise ValueError("sector mapping date conversion must preserve exact precision")
        frame = frame.with_columns(pl.col("date").cast(timeline.schema["date"]))
        keys = ["date", "asset"]
    if frame.select(pl.struct(*keys).is_duplicated().any()).item():
        raise ValueError("sector mapping keys must be unique")
    labeled = positions.join(frame, on=keys, how="left")
    sectors = (
        labeled.group_by("date", "sector")
        .agg(_exposure_aggregations())
        .join(timeline.select("date", "equity", "normalization_status"), on="date", how="left")
    )
    sectors = sectors.with_columns(
        *[
            pl.when(pl.col("normalization_status") == "defined")
            .then(pl.col(f"{name}_exposure") / pl.col("equity"))
            .otherwise(None)
            .alias(f"{name}_exposure_fraction")
            for name in ["long", "short", "gross", "net"]
        ]
    )
    return labeled, sectors.select(
        "date",
        "sector",
        "equity",
        "normalization_status",
        "long_exposure",
        "short_exposure",
        "gross_exposure",
        "net_exposure",
        "open_positions",
        "long_positions",
        "short_positions",
        "long_exposure_fraction",
        "short_exposure_fraction",
        "gross_exposure_fraction",
        "net_exposure_fraction",
    ).sort("date", "sector")
