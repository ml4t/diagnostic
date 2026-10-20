"""Owned analytical price views feeding the existing return kernel and cleaner."""

from __future__ import annotations

from collections.abc import Mapping
from hashlib import sha256
from typing import Any, Literal

import numpy as np
import pandas as pd
import polars as pl

from ml4t.diagnostic.metrics.basic import compute_forward_returns
from ml4t.diagnostic.signal._prepared import positive_count
from ml4t.diagnostic.signal._utils import _signal_frame
from ml4t.diagnostic.signal.quantile import _finite_return_coordinates, _sample_return_statistics
from ml4t.diagnostic.utils.validation import (
    _align_datetime_key_resolution,
    _validate_observation_keys,
)


def _filter_statistics(sample: pl.DataFrame, asset_col: str) -> pl.DataFrame:
    rows = []
    for group in sample.partition_by(asset_col, maintain_order=True):
        values = (
            group["_research_filter_return"]
            .filter(group["_research_filter_return"].is_finite())
            .to_numpy()
        )
        if len(values):
            coordinates, scale, offset = _finite_return_coordinates(values)
            mean, std, _, _ = _sample_return_statistics(coordinates)
        else:
            scale, offset, mean, std = 1.0, 0.0, float("nan"), float("nan")
        rows.append(
            group.select(asset_col)
            .head(1)
            .with_columns(
                pl.lit(scale or 1.0).alias("_research_scale"),
                pl.lit(offset).alias("_research_offset"),
                pl.lit(mean).alias("_research_mean"),
                pl.lit(std).alias("_research_std"),
            )
        )
    return (
        pl.concat(rows)
        if rows
        else pl.DataFrame(
            schema={
                asset_col: sample.schema[asset_col],
                **dict.fromkeys(
                    ("_research_scale", "_research_offset", "_research_mean", "_research_std"),
                    pl.Float64,
                ),
            }
        )
    )


def price_outcomes(
    factors: pl.DataFrame,
    prices: pl.DataFrame | pd.DataFrame,
    *,
    horizons: Mapping[str, int],
    price_basis: str | None,
    date_col: str,
    asset_col: str,
    price_col: str,
    index_mode: Literal["common_grid", "per_asset"],
    endpoint_method: Literal["cumulative", "exit_bar"],
    fill_method: Literal["ffill"] | None,
    return_filter_zscore: float | None,
    available_at_col: str | None,
    contract: dict[str, Any],
    filter_dates: pl.DataFrame | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame, dict[str, Any]]:
    """Construct observation-count research outcomes without calendar inference."""
    if not isinstance(horizons, Mapping) or not horizons:
        raise ValueError("horizons must be a nonempty mapping of selectors to observation counts")
    if any(not isinstance(name, str) or not name for name in horizons):
        raise ValueError("horizon names must be nonempty strings")
    periods = {name: positive_count(value, "horizons") for name, value in horizons.items()}
    if not isinstance(price_basis, str) or not price_basis.strip():
        raise ValueError("price_basis must explicitly declare the supplied price basis")
    if index_mode not in ("common_grid", "per_asset"):
        raise ValueError("index_mode must be common_grid or per_asset")
    if endpoint_method not in ("cumulative", "exit_bar"):
        raise ValueError("endpoint_method must be cumulative or exit_bar")
    if fill_method not in (None, "ffill"):
        raise ValueError("fill_method must be None or ffill")
    if return_filter_zscore is not None and (
        isinstance(return_filter_zscore, bool)
        or not np.isfinite(return_filter_zscore)
        or return_filter_zscore < 0
    ):
        raise ValueError("return_filter_zscore must be finite and nonnegative, or None")
    roles = [date_col, asset_col, price_col, *([available_at_col] if available_at_col else [])]
    if len(set(roles)) != len(roles) or any(name.startswith("_research_") for name in roles):
        raise ValueError("price roles must be distinct and cannot use the _research_ prefix")
    wide = (
        isinstance(prices, pd.DataFrame)
        and not {date_col, asset_col}.issubset(prices.columns)
        and not isinstance(prices.index, pd.MultiIndex)
    )
    if wide:
        assert isinstance(prices, pd.DataFrame)
        if not isinstance(prices.index, pd.DatetimeIndex):
            raise TypeError("wide prices require a DatetimeIndex")
        if not prices.columns.is_unique:
            raise ValueError("wide price asset IDs must be unique")
        temporary = "_research_wide_time"
        while temporary in prices.columns:
            temporary += "_"
        value = "_research_wide_price"
        while value in prices.columns or value == temporary:
            value += "_"
        prices = (
            prices.rename_axis(temporary)
            .reset_index()
            .melt(id_vars=[temporary], var_name=asset_col, value_name=value)
            .rename(columns={temporary: date_col, value: price_col})
        )
    values = _signal_frame(prices)
    _validate_observation_keys(values, [date_col, asset_col], "price")
    if price_col not in values.columns or not (
        values.schema[price_col].is_numeric() or values.schema[price_col] == pl.Null
    ):
        raise TypeError("price_col must identify numeric prices")
    input_hash = sha256(values.write_ipc(None).getvalue()).hexdigest()
    factor_hash = sha256(factors.write_ipc(None).getvalue()).hexdigest()
    input_schema = {name: str(dtype) for name, dtype in values.schema.items()}
    if values.schema[price_col] == pl.Null:
        values = values.with_columns(pl.col(price_col).cast(pl.Float64))
    if values.is_empty():
        values = values.cast(
            {date_col: factors.schema[date_col], asset_col: factors.schema[asset_col]}
        )
    if values.schema[asset_col] != factors.schema[asset_col] and all(
        dtype in (pl.String, pl.Categorical) or isinstance(dtype, pl.Enum)
        for dtype in (values.schema[asset_col], factors.schema[asset_col])
    ):
        values = values.with_columns(pl.col(asset_col).cast(pl.String))
        factors = factors.with_columns(pl.col(asset_col).cast(pl.String))
    factors, values = _align_datetime_key_resolution(factors, values, date_col)
    if (
        values.schema[date_col] != factors.schema[date_col]
        or values.schema[asset_col] != factors.schema[asset_col]
    ):
        raise ValueError("factor and price keys must have matching types and timezones")
    if values.schema[date_col] != pl.Date and not isinstance(values.schema[date_col], pl.Datetime):
        raise TypeError("price keys must be typed dates or datetimes")
    if available_at_col is not None:
        if available_at_col not in values.columns or not isinstance(
            values.schema[available_at_col], pl.Datetime
        ):
            raise TypeError("price availability must contain datetime instants")
        dtype = values.schema[available_at_col]
        availability = pl.col(available_at_col)
        if dtype.time_zone is None:
            timezone = contract["feed_spec"].get("timezone")
            if not timezone:
                raise ValueError("naive price availability requires declared feed timezone")
            availability = availability.dt.replace_time_zone(
                timezone, ambiguous="raise", non_existent="raise"
            )
        availability = availability.dt.convert_time_zone("UTC").cast(pl.Datetime("ns", "UTC"))
    else:
        availability = pl.lit(None, dtype=pl.Datetime("ns", "UTC"))
    keys = [date_col, asset_col]
    observed = values.select(
        *keys,
        pl.col(price_col).alias("_research_raw_price"),
        pl.lit(True).alias("_research_present"),
        pl.col(price_col).cast(pl.Float64).is_finite().fill_null(False).alias("_research_observed"),
        availability.alias("_research_available"),
    )
    if index_mode == "common_grid":
        clock = values.select(date_col).unique()
        assets = factors.select(asset_col).unique()
        observed = clock.join(assets, how="cross").join(
            observed, on=keys, how="left", validate="1:1"
        )
    observed = observed.sort(asset_col, date_col).with_columns(
        pl.col("_research_present", "_research_observed").fill_null(False),
        pl.when(pl.col("_research_observed"))
        .then(pl.col("_research_raw_price"))
        .otherwise(None)
        .alias("_research_price"),
        pl.when(pl.col("_research_observed"))
        .then(pl.col("_research_available"))
        .otherwise(None)
        .alias("_research_available"),
        pl.lit(True).alias("_research_clock_entry"),
    )
    if fill_method == "ffill":
        observed = observed.with_columns(
            pl.col("_research_price", "_research_available").forward_fill().over(asset_col)
        )
    safe_periods = {name: min(count, observed.height + 1) for name, count in periods.items()}
    produced = compute_forward_returns(
        observed,
        periods=list(dict.fromkeys([1, *safe_periods.values()])),
        price_col="_research_price",
        group_col=asset_col,
        date_col=date_col,
        output_col_template="_research_return_{period}",
    )
    assert isinstance(produced, pl.DataFrame)
    analysis_clock = factors.select(date_col).unique() if filter_dates is None else filter_dates
    analysis_clock = analysis_clock.select(date_col).unique().sort(date_col)
    produced, analysis_clock = _align_datetime_key_resolution(produced, analysis_clock, date_col)
    factors, produced = _align_datetime_key_resolution(factors, produced, date_col)
    results = []

    def shifted(column: str, steps: int) -> pl.Expr:
        return pl.col(column).shift(-steps).over(asset_col)

    for name, count in safe_periods.items():
        reference = 0 if endpoint_method == "cumulative" else count - 1
        sample = produced.with_columns(
            shifted(date_col, count).alias("requested_exit_time"),
            shifted(date_col, count).alias("exit_time"),
            shifted(date_col, reference).alias("reference_time"),
            shifted("_research_price", count).alias("exit_price"),
            shifted("_research_price", reference).alias("reference_price"),
            shifted("_research_present", count).alias("exit_row_present"),
            shifted("_research_present", reference).alias("reference_row_present"),
            shifted("_research_observed", count).alias("exit_price_observed"),
            shifted("_research_observed", reference).alias("reference_price_observed"),
            shifted("_research_available", count).alias("exit_available_at"),
            shifted("_research_available", reference).alias("reference_available_at"),
            (
                pl.col(f"_research_return_{count}")
                if endpoint_method == "cumulative"
                else shifted("_research_return_1", reference)
            ).alias("_research_raw_return"),
        )
        sample = factors.select(keys).join(sample, on=keys, how="left", validate="1:1")
        status = (
            pl.when(~pl.col("_research_clock_entry").fill_null(False))
            .then(pl.lit("missing_entry"))
            .when(pl.col("requested_exit_time").is_null())
            .then(pl.lit("insufficient_tail"))
            .when(pl.col("exit_price").is_null() & ~pl.col("exit_row_present").fill_null(False))
            .then(pl.lit("missing_endpoint"))
            .when(pl.col("exit_price").is_null() | pl.col("reference_price").is_null())
            .then(pl.lit("missing_price"))
            .when(~pl.col("_research_raw_return").is_finite().fill_null(False))
            .then(pl.lit("undefined_return"))
            .otherwise(pl.lit("ok"))
        )
        sample = sample.with_columns(status.alias("outcome_status"))
        if return_filter_zscore is not None:
            filter_sample = (
                produced.with_columns(
                    (
                        pl.col(f"_research_return_{count}")
                        if endpoint_method == "cumulative"
                        else shifted("_research_return_1", reference)
                    ).alias("_research_filter_return")
                )
                .join(analysis_clock, on=date_col, how="semi")
                .sort(asset_col, date_col)
            )
            moments = _filter_statistics(filter_sample, asset_col)
            sample = sample.join(moments, on=asset_col, how="left", validate="m:1").with_columns(
                pl.when(
                    (pl.col("outcome_status") == "ok")
                    & (
                        (
                            pl.col("_research_raw_return") / pl.col("_research_scale")
                            - pl.col("_research_offset")
                            - pl.col("_research_mean")
                        ).abs()
                        > return_filter_zscore * pl.col("_research_std")
                    )
                )
                .then(pl.lit("filtered_outlier"))
                .otherwise(pl.col("outcome_status"))
                .alias("outcome_status")
            )
        analytical_fill = (
            pl.col("exit_price").is_not_null() & ~pl.col("exit_price_observed").fill_null(False)
        ) | (
            pl.col("reference_price").is_not_null()
            & ~pl.col("reference_price_observed").fill_null(False)
        )
        results.append(
            sample.select(
                *keys,
                "requested_exit_time",
                "exit_time",
                "reference_time",
                "exit_price",
                "reference_price",
                pl.col("_research_present").fill_null(False).alias("entry_row_present"),
                pl.col("_research_observed").fill_null(False).alias("entry_price_observed"),
                "exit_row_present",
                "reference_row_present",
                "exit_price_observed",
                "reference_price_observed",
                "exit_available_at",
                "reference_available_at",
                analytical_fill.alias("analytical_fill_used"),
                pl.when(
                    pl.col("exit_available_at").is_not_null()
                    & pl.col("reference_available_at").is_not_null()
                    & ~analytical_fill
                    & pl.lit(return_filter_zscore is None)
                )
                .then(pl.max_horizontal("exit_available_at", "reference_available_at"))
                .otherwise(None)
                .alias("outcome_available_at"),
                pl.when(pl.col("outcome_status") == "ok")
                .then(pl.col("_research_raw_return"))
                .otherwise(None)
                .alias("forward_return"),
                pl.col("_research_raw_return").alias("unfiltered_forward_return"),
                "outcome_status",
                pl.lit(name).alias("horizon"),
                pl.lit(str(periods[name])).alias("horizon_value"),
                pl.lit("observation").alias("window_unit"),
                pl.lit(endpoint_method).alias("endpoint_method"),
                pl.lit("returns").alias("return_method"),
                pl.lit(index_mode).alias("index_mode"),
                pl.lit(price_basis).alias("price_basis"),
                pl.lit(contract["calendar"], dtype=pl.String).alias("calendar"),
                pl.lit(contract["calendar_source"] != "default").alias("calendar_declared"),
            )
        )
    return (
        factors,
        pl.concat(results),
        {
            "input_format": "wide cells" if wide else "long keyed quotes",
            "price_basis": price_basis,
            "index_mode": index_mode,
            "endpoint_method": endpoint_method,
            "window_unit": "observation",
            "horizons": periods,
            "fill_method": fill_method,
            "return_filter_zscore": return_filter_zscore,
            "filter_scope": "per asset over supplied legacy factor index date levels before reindexing; lookahead-sensitive research view"
            if filter_dates is not None
            else "per asset over actual factor analysis dates before cleaning; lookahead-sensitive research view",
            "filter_clock_rows": analysis_clock.height,
            "filter_clock_sha256": sha256(analysis_clock.write_ipc(None).getvalue()).hexdigest(),
            "price_input_sha256": input_hash,
            "factor_input_sha256": factor_hash,
            "price_input_schema": input_schema,
            "price_col": price_col,
            "availability_policy": "observed reference and exit publications only; analytical fills and lookahead filters leave outcome availability unknown",
            "calendar_validation": "not performed; observation shifts are not session or expected-bar counts",
            "execution_limit": "Analytical fills are not new observed or executable prices",
        },
    )
