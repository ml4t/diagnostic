"""Quantile analysis functions.

Simple, pure functions for analyzing returns by quantile.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import pandas as pd
import polars as pl
from scipy.stats import spearmanr, ttest_ind

from ml4t.diagnostic.signal._prepared import positive_count
from ml4t.diagnostic.signal._utils import _signal_frame, _signal_inputs
from ml4t.diagnostic.signal.returns import _paired_demean
from ml4t.diagnostic.utils.validation import _validate_observation_keys


def _finite_return_coordinates(values: np.ndarray) -> tuple[np.ndarray, float, float]:
    floating = np.asarray(values, dtype=np.float64)
    scale = float(np.max(np.abs(floating)))
    scaled = floating / scale if scale > 0 else floating
    offset = float(scaled[0])
    return scaled - offset, scale, offset


def _sample_return_statistics(values: np.ndarray) -> tuple[float, float, int, float]:
    count = len(values)
    if not count:
        return float("nan"), float("nan"), 0, float("nan")
    floating = np.asarray(values, dtype=np.float64)
    if np.isfinite(floating).all():
        shifted, scale, offset = _finite_return_coordinates(floating)
        with np.errstate(over="ignore", invalid="ignore"):
            mean = float(np.mean(values))
        if not math.isfinite(mean):
            mean = (float(np.mean(shifted)) + offset) * scale
        scaled_std = float(np.std(shifted, ddof=1)) if count > 1 else float("nan")
        std = scaled_std * scale
        error = (scaled_std / math.sqrt(count)) * scale
    else:
        mean = float(np.mean(floating))
        std = float(np.std(floating, ddof=1)) if count > 1 else float("nan")
        error = std / math.sqrt(count)
    return mean, std, count, error


def _quantile_stat_frame(identity: pl.DataFrame, values: np.ndarray) -> pl.DataFrame:
    mean, std, count, error = _sample_return_statistics(values)
    return identity.with_columns(
        pl.lit(mean if count else None, dtype=pl.Float64).alias("mean"),
        pl.lit(std if count > 1 else None, dtype=pl.Float64).alias("std"),
        pl.lit(error if count > 1 else None, dtype=pl.Float64).alias("standard_error"),
        pl.lit(count, dtype=pl.UInt32).alias("n_obs"),
    )


def _quantile_observations(data: pl.DataFrame | pd.DataFrame) -> pl.DataFrame:
    """Validate canonical dated bucket statistics without re-estimating their sample."""
    values = _signal_frame(data)
    required = {
        "date",
        "quantile",
        "return_column",
        "mean",
        "n_obs",
        "sample_unit",
        "return_adjustment",
    }
    if missing := required.difference(values.columns):
        raise ValueError(f"empirical quantile observations are missing columns: {sorted(missing)}")
    _validate_observation_keys(
        values,
        ["date", "quantile", "return_column", *(["group"] if "group" in values.columns else [])],
        "empirical quantile observations",
    )
    for column in ["quantile", "mean", "n_obs"]:
        if not (
            values[column].dtype.is_numeric()
            or (column == "mean" and values[column].dtype == pl.Null)
        ):
            raise TypeError(f"empirical quantile {column} must be numeric")
    buckets = values["quantile"]
    counts = values["n_obs"]
    if not ((buckets >= 1) & (buckets == buckets.floor())).all():
        raise ValueError("empirical bucket labels must be positive integers")
    if (
        counts.null_count()
        or not counts.is_finite().all()
        or not ((counts >= 0) & (counts == counts.floor())).all()
    ):
        raise ValueError("empirical asset counts must be nonnegative integers")
    if not values["sample_unit"].is_in(["asset"]).fill_null(False).all():
        raise ValueError(
            "empirical observations require per-date bucket statistics, not date aggregates"
        )
    if (
        not values["return_adjustment"]
        .is_in(["none", "date_mean", "date_group_mean"])
        .fill_null(False)
        .all()
    ):
        raise ValueError("empirical return adjustment must name the shared statistics method")
    return values


def compute_quantile_statistics(
    data: pl.DataFrame | pd.DataFrame,
    *,
    return_columns: Sequence[str],
    by_date: bool = False,
    by_group: bool = False,
    demeaned: bool = True,
    group_adjust: bool = False,
    factor_col: str = "factor",
    date_col: str = "date",
    asset_col: str = "asset",
    quantile_col: str = "quantile",
    group_col: str = "group",
) -> pl.DataFrame:
    """Inspect date/bucket return statistics or equal-date mean statistics.

    Buckets and outcomes are supplied, not constructed. With by_date=True,
    finite factor/outcome pairs within each date/bucket define the mean,
    sample standard deviation and standard error. Otherwise the same sample
    reduction operates on defined daily means, giving dates equal weight.
    n_obs counts assets or dates as identified by sample_unit, never pooled
    asset-date rows. Singleton samples have a defined mean but null sample
    dispersion/error. Empty observed samples retain zero counts and null means.
    Across-date group summaries retain observed group/bucket label combinations
    even when a combination has no defined dates. Horizon order follows the
    explicit return_columns, not alphabetical labels.

    Group adjustment takes precedence over date demeaning. This uses the same
    paired outcome transform as grouped IC. Factors, buckets and callers are
    unchanged; group-relative buckets do not imply adjusted returns.

    Parameters
    ----------
    data : DataFrame
        Unique date/asset rows. Pandas MultiIndex input is accepted.
    return_columns : sequence of str
        Explicit supplied outcomes, not inferred duration labels.
    by_date, by_group : bool
        Retain date and/or group dimensions. Defaults aggregate daily means.
    demeaned, group_adjust : bool
        Subtract date or date/group paired return means. Defaults demean dates.
    factor_col, date_col, asset_col, quantile_col, group_col : str
        Explicit physical input names. Buckets must be positive integer labels;
        they need not be contiguous or present on every date.

    Returns
    -------
    polars.DataFrame
        Canonical optional date/group, quantile, return_column, mean, std,
        standard_error, n_obs, sample_unit and return_adjustment columns.
        Raw observations remain inspectable in the supplied/owned input;
        adjusted observations are exposed by demean_signal_returns.
    """
    for name, value in [
        ("by_date", by_date),
        ("by_group", by_group),
        ("demeaned", demeaned),
        ("group_adjust", group_adjust),
    ]:
        if not isinstance(value, bool | np.bool_):
            raise ValueError(f"{name} must be boolean")
    values, columns = _signal_inputs(
        data,
        return_columns,
        factor_col,
        date_col,
        asset_col,
        group_col if group_adjust or by_group else None,
    )
    _validate_observation_keys(values, [quantile_col], "signal buckets", unique=False)
    if not values[quantile_col].dtype.is_numeric():
        raise TypeError("signal buckets must contain numeric integer labels")
    buckets = values[quantile_col].cast(pl.Float64)
    if not (buckets.is_finite() & (buckets >= 1) & (buckets == buckets.floor())).all():
        raise ValueError("signal buckets must contain positive integer labels")
    if quantile_col in columns:
        raise ValueError("return columns must be distinct from the bucket column")
    outcomes = [f"__return_{index}" for index in range(len(columns))]
    values = values.select(
        pl.col(date_col).alias("date"),
        pl.col(asset_col).alias("asset"),
        pl.col(factor_col).alias("factor"),
        pl.col(quantile_col).alias("quantile"),
        *[pl.col(column).alias(alias) for column, alias in zip(columns, outcomes, strict=True)],
        *([pl.col(group_col).alias("group")] if group_adjust or by_group else []),
    )
    if group_adjust or demeaned:
        values = _paired_demean(
            values, outcomes, "factor", "date", "group" if group_adjust else None
        )
    keys = ["date", *(["group"] if by_group else []), "quantile"]
    parts = []
    for part in values.partition_by(keys, maintain_order=True):
        identity = part.select(keys).head(1)
        for column, label in zip(outcomes, columns, strict=True):
            returns = part.filter(pl.col("factor").is_finite() & pl.col(column).is_finite())[
                column
            ].to_numpy()
            parts.append(
                _quantile_stat_frame(
                    identity.with_columns(pl.lit(label).alias("return_column")), returns
                )
            )
    dimensions = {key: values.schema[key] for key in keys}
    schema = {
        **dimensions,
        "return_column": pl.String,
        "mean": pl.Float64,
        "std": pl.Float64,
        "standard_error": pl.Float64,
        "n_obs": pl.UInt32,
    }
    result = pl.concat(parts) if parts else pl.DataFrame(schema=schema)
    if not by_date:
        summary_keys = [key for key in keys if key != "date"] + ["return_column"]
        parts = []
        for part in result.partition_by(summary_keys, maintain_order=True):
            identity = part.select(summary_keys).head(1)
            parts.append(
                _quantile_stat_frame(
                    identity, part.filter(pl.col("mean").is_finite())["mean"].to_numpy()
                )
            )
        result = (
            pl.concat(parts)
            if parts
            else pl.DataFrame(schema={key: value for key, value in schema.items() if key != "date"})
        )
        if by_group:
            grid = (
                values.select("group")
                .unique()
                .join(values.select("quantile").unique(), how="cross")
            )
            grid = grid.join(pl.DataFrame({"return_column": columns}), how="cross")
            result = grid.join(result, on=summary_keys, how="left").with_columns(
                pl.col("n_obs").fill_null(0)
            )
    return (
        result.with_columns(
            pl.lit("asset" if by_date else "date").alias("sample_unit"),
            pl.lit(
                "date_group_mean" if group_adjust else "date_mean" if demeaned else "none"
            ).alias("return_adjustment"),
            pl.col("return_column")
            .replace_strict({column: index for index, column in enumerate(columns)})
            .alias("__horizon_order"),
        )
        .sort(["__horizon_order", *[key for key in keys if by_date or key != "date"]])
        .drop("__horizon_order")
    )


def compute_quantile_spread(
    statistics: pl.DataFrame | pd.DataFrame,
    *,
    upper_quantile: int,
    lower_quantile: int,
    include_standard_error: bool = True,
) -> pl.DataFrame:
    """Subtract arbitrary bucket means from canonical quantile statistics.

    Consume compute_quantile_statistics output without re-estimating samples.
    Retain optional date/group keys, each leg's mean/error/count, sample unit
    and adjustment. Missing counterpart legs remain null. Standard error uses
    hypot(upper_error, lower_error), the independent-error propagation used by
    Alphalens, even when the same bucket is selected twice. This is not a
    covariance-aware estimate or the existing pooled two-sample Student test.

    The raw difference represents +1 upper / -1 lower exposure (gross two),
    not a unit-gross portfolio return. Return-column labels retain their input
    order and are not interpreted as durations. Empty statistics retain an
    empty typed output; absent requested buckets in a nonempty input fail.
    Neither caller-owned Pandas/Polars statistics nor prepared panels change.
    """
    upper = positive_count(upper_quantile, "upper_quantile")
    lower = positive_count(lower_quantile, "lower_quantile")
    if not isinstance(include_standard_error, bool | np.bool_):
        raise ValueError("include_standard_error must be boolean")
    values = _signal_frame(statistics)
    dimensions = [name for name in ["date", "group"] if name in values.columns]
    keys = [*dimensions, "return_column", "sample_unit", "return_adjustment"]
    _validate_observation_keys(values, [*keys, "quantile"], "quantile statistics")
    for name in [
        "quantile",
        "mean",
        "n_obs",
        *(["standard_error"] if include_standard_error else []),
    ]:
        if name not in values.columns:
            raise ValueError(f"quantile statistics is missing {name}")
        if not (values[name].dtype.is_numeric() or values[name].dtype == pl.Null):
            raise TypeError(f"quantile statistics {name} must be numeric")
    buckets = values["quantile"].cast(pl.Float64)
    if not ((buckets >= 1) & (buckets == buckets.floor()) & buckets.is_finite()).all():
        raise ValueError("quantile statistics buckets must be positive integer labels")
    counts = values["n_obs"].cast(pl.Float64)
    if (
        counts.null_count()
        or not ((counts >= 0) & (counts == counts.floor()) & counts.is_finite()).all()
    ):
        raise ValueError("quantile statistics n_obs must be nonmissing nonnegative integer counts")
    expected_unit = "asset" if "date" in dimensions else "date"
    if not (values["sample_unit"] == expected_unit).all():
        raise ValueError("quantile statistics sample_unit must agree with date dimensions")
    if not values["return_adjustment"].is_in(["none", "date_mean", "date_group_mean"]).all():
        raise ValueError("quantile statistics return_adjustment is not supported")
    if values.height and not {upper, lower}.issubset(set(buckets.to_list())):
        raise ValueError("requested quantile bucket is absent from statistics")
    leg_columns = ["mean", "n_obs"]
    if "standard_error" in values.columns:
        leg_columns.append("standard_error")
    legs = [
        values.filter(pl.col("quantile") == bucket).select(
            *keys,
            *[pl.col(name).alias(f"{side}_{name}") for name in leg_columns],
            *(
                []
                if "standard_error" in leg_columns
                else [pl.lit(None, dtype=pl.Float64).alias(f"{side}_standard_error")]
            ),
        )
        for side, bucket in [("upper", upper), ("lower", lower)]
    ]
    result = legs[0].join(legs[1], on=keys, how="full", coalesce=True)
    error = (
        np.hypot(
            result["upper_standard_error"].cast(pl.Float64).to_numpy(),
            result["lower_standard_error"].cast(pl.Float64).to_numpy(),
        )
        if include_standard_error
        else np.full(result.height, np.nan)
    )
    order = (
        values.select("return_column").unique(maintain_order=True).with_row_index("__horizon_order")
    )
    return (
        result.with_columns(
            (pl.col("upper_mean") - pl.col("lower_mean"))
            .cast(pl.Float64)
            .fill_nan(None)
            .alias("spread"),
            pl.Series("standard_error", error).fill_nan(None),
            pl.lit(upper).alias("upper_quantile"),
            pl.lit(lower).alias("lower_quantile"),
            pl.lit("long_upper_short_lower_unit_each").alias("exposure_convention"),
        )
        .join(order, on="return_column", how="left")
        .sort(["__horizon_order", *dimensions])
        .drop("__horizon_order")
    )


def _compute_quantile_return_statistics(
    data: pl.DataFrame,
    period: int,
    n_quantiles: int,
    quantile_col: str = "quantile",
) -> tuple[dict[int, float], dict[int, float]]:
    return_col = f"{period}D_fwd_return"

    if return_col not in data.columns:
        missing = dict.fromkeys(range(1, n_quantiles + 1), float("nan"))
        return missing, missing.copy()

    means: dict[int, float] = {}
    standard_deviations: dict[int, float] = {}
    valid_data = data.filter(pl.col(return_col).is_not_null())

    # Polars parallel group reductions may add floats in a different order on
    # repeated calls. Reduce rows within each quantile in input order instead.
    for partition in valid_data.partition_by(quantile_col, maintain_order=True):
        quantile = partition.item(0, quantile_col)
        returns = partition.get_column(return_col).to_numpy()
        quantile_number = int(quantile)
        means[quantile_number], standard_deviations[quantile_number], _, _ = (
            _sample_return_statistics(returns)
        )

    # Fill missing quantiles
    for q in range(1, n_quantiles + 1):
        means.setdefault(q, float("nan"))
        standard_deviations.setdefault(q, float("nan"))

    return dict(sorted(means.items())), dict(sorted(standard_deviations.items()))


def compute_quantile_returns(
    data: pl.DataFrame,
    period: int,
    n_quantiles: int,
    quantile_col: str = "quantile",
) -> dict[int, float]:
    """Compute mean forward returns by quantile.

    Parameters
    ----------
    data : pl.DataFrame
        Data with quantile and forward return columns.
    period : int
        Forward return period in days.
    n_quantiles : int
        Number of quantiles.
    quantile_col : str, default "quantile"
        Quantile column name.

    Returns
    -------
    dict[int, float]
        Mean return by quantile (1 = lowest factor).
    """
    means, _ = _compute_quantile_return_statistics(data, period, n_quantiles, quantile_col)

    return means


def compute_spread(
    data: pl.DataFrame,
    period: int,
    n_quantiles: int,
    quantile_col: str = "quantile",
) -> dict[str, float]:
    """Compute long-short spread and statistics.

    Parameters
    ----------
    data : pl.DataFrame
        Data with quantile and forward return columns.
    period : int
        Forward return period in days.
    n_quantiles : int
        Number of quantiles.
    quantile_col : str, default "quantile"
        Quantile column name.

    Returns
    -------
    dict[str, float]
        spread, t_stat, p_value
    """
    return_col = f"{period}D_fwd_return"

    if n_quantiles < 2 or return_col not in data.columns:
        return {
            "spread": float("nan"),
            "t_stat": float("nan"),
            "p_value": float("nan"),
        }

    top_returns = data.filter(pl.col(quantile_col) == n_quantiles)[return_col].to_numpy()
    bottom_returns = data.filter(pl.col(quantile_col) == 1)[return_col].to_numpy()

    top_returns = top_returns[~np.isnan(top_returns)]
    bottom_returns = bottom_returns[~np.isnan(bottom_returns)]

    if len(top_returns) < 2 or len(bottom_returns) < 2:
        return {
            "spread": float("nan"),
            "t_stat": float("nan"),
            "p_value": float("nan"),
        }

    if np.isfinite(top_returns).all() and np.isfinite(bottom_returns).all():
        combined = np.concatenate([top_returns, bottom_returns])
        shifted, scale, _ = _finite_return_coordinates(combined)
        top = shifted[: len(top_returns)]
        bottom = shifted[len(top_returns) :]
        spread = float(np.mean(top) - np.mean(bottom)) * scale
        t_stat, p_value = ttest_ind(top, bottom)
    else:
        spread = float(np.mean(top_returns) - np.mean(bottom_returns))
        t_stat, p_value = ttest_ind(top_returns, bottom_returns)

    return {
        "spread": spread,
        "t_stat": float(t_stat),
        "p_value": float(p_value),
    }


def monotonicity_score(
    quantile_returns: dict[int, float],
) -> float:
    """Compute monotonicity score from pre-computed quantile returns.

    A simple Spearman correlation between quantile ranks and their mean returns.
    For the full-featured version that accepts raw DataFrames, use
    ``ml4t.diagnostic.metrics.compute_monotonicity``.

    Parameters
    ----------
    quantile_returns : dict[int, float]
        Mean return by quantile (from ``compute_quantile_returns``).

    Returns
    -------
    float
        Spearman rho (-1 to 1). 1.0 = perfect monotonic increase.
    """
    # Sort by quantile
    sorted_items = sorted(quantile_returns.items())
    quantiles = [q for q, r in sorted_items if not np.isnan(r)]
    returns = [r for q, r in sorted_items if not np.isnan(r)]

    if len(quantiles) < 3:
        return float("nan")

    rho, _ = spearmanr(quantiles, returns)
    return float(rho) if not np.isnan(rho) else float("nan")


__all__ = [
    "compute_quantile_statistics",
    "compute_quantile_spread",
    "compute_quantile_returns",
    "compute_spread",
    "monotonicity_score",
]
