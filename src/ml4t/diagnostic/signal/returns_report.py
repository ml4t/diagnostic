"""Owned return-report tables composed from the existing signal analytical kernels."""

from __future__ import annotations

import base64
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from io import BytesIO
from numbers import Real
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
import polars as pl

from ml4t.diagnostic.evaluation.portfolio_analysis.metrics import _safe_cumprod
from ml4t.diagnostic.signal.factor_regression import compute_factor_alpha_beta
from ml4t.diagnostic.signal.portfolio import compute_factor_returns, compute_factor_weights
from ml4t.diagnostic.signal.quantile import (
    _display_returns,
    compute_quantile_spread,
    compute_quantile_statistics,
)
from ml4t.diagnostic.utils.validation import _align_datetime_key_resolution

if TYPE_CHECKING:
    from ml4t.diagnostic.signal.panel import PreparedSignalPanel

_TABLES = (
    "prepared_data",
    "factor_weights",
    "factor_returns",
    "regression_statistics",
    "regression_observations",
    "quantile_statistics",
    "quantile_observations",
    "group_quantile_statistics",
    "quantile_output_grid",
    "spread_observations",
    "factor_paths",
    "quantile_paths",
)


@dataclass(frozen=True)
class SignalReturnsResult:
    """Owned samples, raw/display statistics and explicitly mathematical wealth paths."""

    _tables: tuple[bytes, ...] = field(repr=False)
    _metadata: str = field(repr=False)

    @property
    def table_names(self) -> tuple[str, ...]:
        """Name the full-precision analytical tables available for inspection."""
        return _TABLES

    def get_table(self, name: str) -> pl.DataFrame:
        """Read an independent typed table, including source counts and missing cells."""
        if name not in _TABLES:
            raise ValueError(f"Unknown returns table {name!r}; choose from {_TABLES}")
        return pl.read_ipc(BytesIO(self._tables[_TABLES.index(name)]))

    @property
    def metadata(self) -> dict:
        """Read independent method, exposure, clock and unit declarations."""
        return json.loads(self._metadata)

    def to_json(self) -> str:
        """Serialize typed snapshots without rounding or recomputing their samples."""
        return json.dumps(
            {
                "format_version": 1,
                "tables": {
                    name: base64.b64encode(value).decode("ascii")
                    for name, value in zip(_TABLES, self._tables, strict=True)
                },
                "metadata": self.metadata,
            },
            allow_nan=False,
        )

    @classmethod
    def from_json(cls, value: str) -> SignalReturnsResult:
        """Restore the supported owned format, retaining exact datetime precision."""
        parsed = json.loads(value)
        if parsed.get("format_version") != 1 or set(parsed.get("tables", {})) != set(_TABLES):
            raise ValueError("unsupported signal returns result format")
        tables = tuple(base64.b64decode(parsed["tables"][name], validate=True) for name in _TABLES)
        for table in tables:
            pl.read_ipc(BytesIO(table))
        return cls(tables, json.dumps(parsed["metadata"], allow_nan=False))


def _display_statistics(frame: pl.DataFrame, powers: dict[str, float]) -> pl.DataFrame:
    converted = _display_returns(frame, powers)
    for source, target in [("std", "display_std"), ("standard_error", "display_standard_error")]:
        converted = converted.with_columns(
            (pl.col(source) * pl.col("conversion_ratio").sqrt()).alias(target)
        ).with_columns(
            pl.when(pl.col(target).is_finite()).then(pl.col(target)).otherwise(None).alias(target)
        )
    return converted.with_columns(
        pl.when(pl.col("standard_error").is_null())
        .then(pl.lit("missing_error"))
        .when(pl.col("display_standard_error").is_null())
        .then(pl.lit("undefined_scaled_error"))
        .otherwise(pl.lit("defined"))
        .alias("display_error_status")
    )


def _powers(columns: list[str], supplied: Mapping[str, float] | None) -> dict[str, float]:
    if supplied is None:
        return dict.fromkeys(columns, 1.0)
    if not isinstance(supplied, Mapping) or set(supplied) != set(columns):
        raise ValueError("rate_conversion_factors must name exactly the selected return columns")
    result = {}
    for column in columns:
        value = supplied[column]
        if isinstance(value, bool | np.bool_) or not isinstance(value, Real):
            raise ValueError("rate_conversion_factors must contain positive finite numbers")
        try:
            result[column] = float(value)
        except (OverflowError, ValueError) as error:
            raise ValueError(
                "rate_conversion_factors must contain positive finite numbers"
            ) from error
        if not np.isfinite(result[column]) or result[column] <= 0:
            raise ValueError("rate_conversion_factors must contain positive finite numbers")
    if result[columns[0]] != 1:
        raise ValueError("the first return column must have conversion factor one")
    return result


def analyze_signal_returns(
    panel: PreparedSignalPanel,
    *,
    annualization_factors: Mapping[str, float],
    demeaned: bool = True,
    group_adjust: bool = False,
    by_group: bool = False,
    rate_conversion_factors: Mapping[str, float] | None = None,
    date_grid: pl.DataFrame | pd.DataFrame | None = None,
    cumulative_columns: Sequence[str] | None = None,
    upper_quantile: int | None = None,
    lower_quantile: int | None = None,
    spread_exposure: Literal["raw_difference", "unit_gross"] = "raw_difference",
) -> SignalReturnsResult:
    """Compose returns, regressions, quantile samples and paths without new estimators.

    Annualization factors explicitly scale each regression intercept; beta is
    dimensionless. Optional first-horizon display powers convert means and scale
    errors by their square roots, retaining the original raw statistics. These
    display errors are descriptive conventions, not confidence intervals.

    Date grids must retain every observed input date. Added output cells have
    unknown returns, zero observed counts and explicit row_present=False. Wealth
    uses the existing NaN-as-flat product, not an assertion of zero missing returns.
    Forward-horizon products are mathematical observation-order paths, not a
    self-financing execution account or calendar-validated daily P&L.
    """
    from ml4t.diagnostic.signal.panel import PreparedSignalPanel

    if not isinstance(panel, PreparedSignalPanel):
        raise TypeError("panel must be an owned PreparedSignalPanel")
    if not isinstance(by_group, bool | np.bool_):
        raise TypeError("by_group must be boolean")
    if spread_exposure not in {"raw_difference", "unit_gross"}:
        raise ValueError("spread_exposure must be raw_difference or unit_gross")
    data = panel.data
    if data.is_empty():
        raise ValueError("returns report requires at least one prepared row")
    columns = panel.metadata["return_columns"]
    powers = _powers(columns, rate_conversion_factors)
    selected = list(columns) if cumulative_columns is None else list(cumulative_columns)
    if (
        isinstance(cumulative_columns, str | bytes)
        or len(set(selected)) != len(selected)
        or not set(selected) <= set(columns)
    ):
        raise ValueError("cumulative_columns must be unique supplied return selectors")
    keys = {
        "factor_col": panel.factor_col,
        "date_col": panel.date_col,
        "asset_col": panel.asset_col,
    }
    weights = compute_factor_weights(data, demeaned=demeaned, group_adjust=group_adjust, **keys)
    returns = compute_factor_returns(
        data,
        return_columns=columns,
        demeaned=demeaned,
        group_adjust=group_adjust,
        date_grid=date_grid,
        **keys,
    )
    observed, clock = _align_datetime_key_resolution(
        data.select(pl.col(panel.date_col).alias("date")).unique(),
        returns.select("date").unique().sort("date"),
        "date",
    )
    if observed.join(clock, on="date", how="anti").height:
        raise ValueError("date_grid must retain every observed date for a complete returns report")
    regression = compute_factor_alpha_beta(
        data,
        return_columns=columns,
        annualization_factors=annualization_factors,
        demeaned=demeaned,
        group_adjust=group_adjust,
        **keys,
    )
    common = dict(return_columns=columns, demeaned=demeaned, group_adjust=group_adjust, **keys)
    statistics = compute_quantile_statistics(data, **common)
    observations = compute_quantile_statistics(data, by_date=True, **common)
    groups = (
        compute_quantile_statistics(data, by_group=True, **common)
        if by_group
        else statistics.head(0).with_columns(
            pl.lit(None, dtype=data.schema.get("group", pl.String)).alias("group")
        )
    )
    buckets = data.select(pl.col("quantile")).unique().sort("quantile")
    grid = clock.join(buckets, how="cross").join(
        pl.DataFrame({"return_column": columns}), how="cross"
    )
    grid, observations = _align_datetime_key_resolution(grid, observations, "date")
    adjustment = "date_group_mean" if group_adjust else "date_mean" if demeaned else "none"
    grid = grid.join(
        observations.with_columns(pl.lit(True).alias("row_present")),
        on=["date", "quantile", "return_column"],
        how="left",
    ).with_columns(
        pl.col("n_obs").fill_null(0),
        pl.col("row_present").fill_null(False),
        pl.col("sample_unit").fill_null("asset"),
        pl.col("return_adjustment").fill_null(adjustment),
    )
    statistics, observations, groups, grid = [
        _display_statistics(frame, powers) for frame in (statistics, observations, groups, grid)
    ]
    scale = 0.5 if spread_exposure == "unit_gross" else 1.0
    if buckets.height >= 2 or upper_quantile is not None or lower_quantile is not None:
        spread = compute_quantile_spread(
            grid.with_columns(
                pl.col("display_mean").alias("mean"),
                pl.col("display_standard_error").alias("standard_error"),
            ),
            upper_quantile=buckets["quantile"][-1] if upper_quantile is None else upper_quantile,
            lower_quantile=buckets["quantile"][0] if lower_quantile is None else lower_quantile,
        ).with_columns(
            (pl.col("spread") * scale).alias("spread"),
            (pl.col("standard_error") * scale).alias("standard_error"),
            pl.lit(spread_exposure).alias("exposure_convention"),
            pl.lit(scale).alias("exposure_scale"),
        )
    else:
        spread = pl.DataFrame(
            schema={
                "date": clock.schema["date"],
                "return_column": pl.String,
                "spread": pl.Float64,
                "standard_error": pl.Float64,
            }
        )
    factor_paths, quantile_paths = [], []
    for column in selected:
        part = returns.filter(pl.col("return_column") == column).sort("date")
        factor_paths.append(
            part.with_columns(
                pl.Series("wealth", _safe_cumprod(1 + part["weighted_return"].to_numpy()))
            )
        )
        for bucket in buckets["quantile"]:
            part = grid.filter(
                (pl.col("return_column") == column) & (pl.col("quantile") == bucket)
            ).sort("date")
            quantile_paths.append(
                part.with_columns(pl.Series("wealth", _safe_cumprod(1 + part["mean"].to_numpy())))
            )
    factor_path = (
        pl.concat(factor_paths)
        if factor_paths
        else returns.head(0).with_columns(pl.lit(None, dtype=pl.Float64).alias("wealth"))
    )
    quantile_path = (
        pl.concat(quantile_paths)
        if quantile_paths
        else grid.head(0).with_columns(pl.lit(None, dtype=pl.Float64).alias("wealth"))
    )
    tables = (
        data,
        weights,
        returns,
        regression.statistics,
        regression.observations,
        statistics,
        observations,
        groups,
        grid,
        spread,
        factor_path,
        quantile_path,
    )
    metadata = {
        **panel.metadata,
        "demeaned": bool(demeaned),
        "group_adjust": bool(group_adjust),
        "by_group": bool(by_group),
        "annualization_factors": {
            row["return_column"]: row["annualization_factor"]
            for row in regression.statistics.iter_rows(named=True)
        },
        "rate_conversion_factors": powers,
        "display_rate_conversion": rate_conversion_factors is not None,
        "base_return_column": columns[0] if rate_conversion_factors is not None else None,
        "cumulative_columns": selected,
        "wealth_representation": "observation_order_supplied_forward_returns",
        "missing_wealth_policy": "flat_wealth_not_observed_zero_return",
        "execution_account": False,
        "spread_exposure": spread_exposure,
        "spread_exposure_scale": scale,
        "spread_availability": "defined_sample"
        if not spread.is_empty()
        else "at_least_two_buckets_required",
        "upper_quantile": upper_quantile if upper_quantile is not None else buckets["quantile"][-1],
        "lower_quantile": lower_quantile if lower_quantile is not None else buckets["quantile"][0],
        "clock_source": "observed" if date_grid is None else "explicit_grid",
        "quantile_aggregation": "equal_date_means",
        "error_scale": "sqrt_horizon_power_not_confidence_coverage",
        "regression_metadata": regression.metadata,
    }
    return SignalReturnsResult(
        tuple(table.write_ipc(None).getvalue() for table in tables),
        json.dumps(metadata, allow_nan=False),
    )
