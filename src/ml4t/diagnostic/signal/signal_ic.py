"""Information Coefficient (IC) computation.

Simple, pure functions for IC analysis.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
import polars as pl

from ml4t.diagnostic.metrics.ic import cross_sectional_ic_series
from ml4t.diagnostic.metrics.ic_inference import compute_ic_summary_stats
from ml4t.diagnostic.signal._prepared import positive_count
from ml4t.diagnostic.signal._utils import _signal_frame, _signal_inputs, precise_dates
from ml4t.diagnostic.signal.returns import _paired_demean
from ml4t.diagnostic.utils.validation import (
    _align_datetime_key_resolution,
    _validate_observation_keys,
)


def compute_signal_ic(
    data: pl.DataFrame | pd.DataFrame,
    *,
    return_columns: Sequence[str],
    method: str = "spearman",
    group_adjust: bool = False,
    by_group: bool = False,
    min_obs: int = 2,
    factor_col: str = "factor",
    date_col: str = "date",
    asset_col: str = "asset",
    group_col: str = "group",
    date_grid: pl.DataFrame | pd.DataFrame | None = None,
) -> pl.DataFrame:
    """Compute inspectable per-date ICs through the existing correlation kernel.

    Return columns are supplied selectors, not inferred durations. Native
    finite pairs are selected separately per column; owned common-sample panels
    already have the same finite sample across their requested columns.
    Group adjustment subtracts the paired date/group mean outcome, not the
    factor. By-group output is independent of that adjustment and bucketing.

    Parameters
    ----------
    data : DataFrame
        Unique date/asset rows with factor, outcomes and optional group values.
        Pandas legacy MultiIndex tables are accepted without modifying callers.
    return_columns : sequence of str
        Nonempty distinct supplied outcomes, retained in the requested order.
    method : str
        Spearman (average ranks) or Pearson correlation.
    group_adjust, by_group : bool
        Demean outcomes within date/group and/or retain the group dimension.
        These do not construct a group-neutral portfolio.
    min_obs : int
        Minimum finite paired observations; undefined ICs retain their counts.
    factor_col, date_col, asset_col, group_col : str
        Explicit input names. Groups must be nonmissing when used.
    date_grid : DataFrame, optional
        Explicit requested date_col values, or date_col/group_col pairs for
        by-group output. String and categorical group labels match by value.
        No calendar or grid is inferred. Missing requested pairs have null IC,
        zero observations and a no_pairs status, not a zero correlation.

    Returns
    -------
    polars.DataFrame
        Canonical date, optional group, return_column, ic, n_obs, status and
        method/adjustment/minimum metadata. Input column names cannot shadow
        those result fields. Unavailable ICs are null, never zero.
    """
    min_obs = positive_count(min_obs, "min_obs")
    if method not in ("spearman", "pearson"):
        raise ValueError(f"Unknown method: {method!r}. Use 'spearman' or 'pearson'.")
    for name, flag in [("group_adjust", group_adjust), ("by_group", by_group)]:
        if not isinstance(flag, bool | np.bool_):
            raise ValueError(f"{name} must be boolean")
    values, columns = _signal_inputs(
        data,
        return_columns,
        factor_col,
        date_col,
        asset_col,
        group_col if group_adjust or by_group else None,
    )
    grid_keys = [date_col, group_col] if by_group else [date_col]
    requested = _signal_frame(date_grid) if date_grid is not None else None
    if requested is not None:
        _validate_observation_keys(requested, grid_keys, "IC grid")
        requested = requested.select(
            pl.col(date_col).alias("date"),
            *([pl.col(group_col).alias("group")] if by_group else []),
        )
    outcome_names = [f"__return_{index}" for index in range(len(columns))]
    values = values.select(
        pl.col(date_col).alias("date"),
        pl.col(asset_col).alias("asset"),
        pl.col(factor_col).alias("factor"),
        *(pl.col(column).alias(name) for column, name in zip(columns, outcome_names, strict=True)),
        *([pl.col(group_col).alias("group")] if group_adjust or by_group else []),
    )
    date_col, asset_col, factor_col, group_col = "date", "asset", "factor", "group"
    keys = [date_col, asset_col]
    grid_keys = [date_col, group_col] if by_group else [date_col]
    if group_adjust:
        values = _paired_demean(values, outcome_names, factor_col, date_col, group_col)
    outputs = []
    partitions = values.partition_by(group_col, maintain_order=True) if by_group else [values]
    for column, label in zip(outcome_names, columns, strict=True):
        parts = []
        for part in partitions:
            observations = cross_sectional_ic_series(
                part.select(*keys, factor_col),
                part.select(*keys, column),
                pred_col=factor_col,
                ret_col=column,
                date_col=date_col,
                entity_col=asset_col,
                method=method,
                min_obs=min_obs,
            )
            finite = part.filter(pl.col(factor_col).is_finite() & pl.col(column).is_finite())
            distinct = finite.group_by(date_col).agg(
                pl.col(factor_col).n_unique().alias("__factor_unique"),
                pl.col(column).n_unique().alias("__outcome_unique"),
            )
            observations = observations.join(distinct, on=date_col, how="left")
            if by_group:
                observations = observations.join(part.select(group_col).head(1), how="cross")
            parts.append(observations)
        observed = (
            pl.concat(parts)
            if parts
            else pl.DataFrame(
                schema={
                    **{key: values.schema[key] for key in grid_keys},
                    "ic": pl.Float64,
                    "n_obs": pl.UInt32,
                    "__factor_unique": pl.UInt32,
                    "__outcome_unique": pl.UInt32,
                }
            )
        )
        if requested is not None:
            grid = requested
            for key in grid_keys:
                observed, grid = _align_datetime_key_resolution(observed, grid, key)
            if by_group:
                if observed.is_empty() and observed.schema[group_col] == pl.Null:
                    observed = observed.with_columns(pl.col(group_col).cast(grid.schema[group_col]))
                label_types = (pl.String, pl.Categorical, pl.Enum)
                if (
                    observed.schema[group_col].base_type() in label_types
                    and grid.schema[group_col].base_type() in label_types
                ):
                    observed = observed.with_columns(pl.col(group_col).cast(pl.String))
                    grid = grid.with_columns(pl.col(group_col).cast(pl.String))
            observed = grid.join(observed, on=grid_keys, how="left")
        observed = observed.with_columns(pl.col("n_obs").fill_null(0))
        observed = observed.with_columns(
            pl.when(pl.col("n_obs") == 0)
            .then(pl.lit("no_pairs"))
            .when(pl.col("n_obs") < max(min_obs, 2))
            .then(pl.lit("insufficient_pairs"))
            .when(pl.col("__factor_unique") < 2)
            .then(pl.lit("constant_factor"))
            .when(pl.col("__outcome_unique") < 2)
            .then(pl.lit("constant_outcome"))
            .when(pl.col("ic").is_finite())
            .then(pl.lit("defined"))
            .otherwise(pl.lit("undefined"))
            .alias("status"),
            pl.lit(label).alias("return_column"),
            pl.lit(method).alias("ic_method"),
            pl.lit("date_group_mean" if group_adjust else "none").alias("return_adjustment"),
            pl.lit(min_obs).alias("min_obs"),
        )
        outputs.append(
            observed.select(
                *grid_keys,
                "return_column",
                "ic",
                "n_obs",
                "status",
                "ic_method",
                "return_adjustment",
                "min_obs",
            ).sort(grid_keys)
        )
    return pl.concat(outputs)


def extract_signal_ic_series(
    data: pl.DataFrame,
    period: int | str,
    method: str = "spearman",
    factor_col: str = "factor",
    date_col: str = "date",
    asset_col: str = "asset",
    min_obs: int = 10,
) -> tuple[list[Any], list[float]]:
    """Extract valid per-date IC values for a single signal horizon.

    Parameters
    ----------
    data : pl.DataFrame
        Factor data with factor and forward return columns.
    period : int or str
        Integer-day convenience or exact supplied return-column selector.
    method : str, default "spearman"
        Correlation method ("spearman" or "pearson").
    factor_col : str, default "factor"
        Factor column name.
    date_col : str, default "date"
        Date column name.
    asset_col : str, default "asset"
        Asset/entity column used for panel joins.
    min_obs : int, default 10
        Minimum observations per date.

    Returns
    -------
    tuple[list[Any], list[float]]
        (dates, ic_values) for dates with valid IC.
    """
    return_col = period if isinstance(period, str) else f"{period}D_fwd_return"

    ic_df = compute_signal_ic(
        data,
        return_columns=[return_col],
        factor_col=factor_col,
        date_col=date_col,
        asset_col=asset_col,
        method=method,
        min_obs=min_obs,
    )

    if ic_df.height == 0:
        return [], []

    ic_clean = ic_df.filter(
        (pl.col("n_obs") >= min_obs) & pl.col("ic").cast(pl.Float64).is_finite()
    )
    dates = precise_dates(ic_clean["date"])
    ic_values = ic_clean["ic"].cast(pl.Float64).to_list()
    return dates, ic_values


def compute_ic_summary(
    ic_series: list[float],
) -> dict[str, float]:
    """Compute summary statistics for an IC series.

    Parameters
    ----------
    ic_series : list[float]
        IC values over time.

    Returns
    -------
    dict[str, float]
        mean, std, t_stat, p_value, pct_positive
    """
    summary = compute_ic_summary_stats(ic_series)
    return {
        "mean": float(summary["mean_ic"]),
        "std": float(summary["std_ic"]),
        "t_stat": float(summary["t_stat"]),
        "p_value": float(summary["p_value"]),
        "pct_positive": float(summary["pct_positive"]),
    }


__all__ = ["compute_signal_ic", "extract_signal_ic_series", "compute_ic_summary"]
