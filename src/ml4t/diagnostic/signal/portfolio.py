"""Factor exposure and supplied-return contributions for owned signal samples."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
import polars as pl

from ml4t.diagnostic.signal._utils import _signal_frame, _signal_inputs
from ml4t.diagnostic.utils.validation import (
    _align_datetime_key_resolution,
    _validate_observation_keys,
)


def _weight_values(values: np.ndarray, demeaned: bool, equal_weight: bool) -> np.ndarray:
    scale = np.max(np.abs(values)) if len(values) else 0.0
    scaled = values / scale if scale > 0 else np.zeros(len(values))
    if demeaned and len(scaled):
        scaled = scaled - scaled[0]
    if equal_weight:
        weights = np.sign(scaled - np.median(scaled) if demeaned else scaled)
        if demeaned:
            for sign in [-1, 1]:
                mask = weights == sign
                if mask.any():
                    weights[mask] /= mask.sum()
    else:
        weights = scaled - scaled.mean() if demeaned else scaled
    gross = np.abs(weights).sum()
    return weights / gross if gross > 0 else np.zeros(len(values))


def _weight_flags(**options: bool) -> None:
    for name, flag in options.items():
        if not isinstance(flag, bool | np.bool_):
            raise ValueError(f"{name} must be boolean")


def compute_factor_weights(
    data: pl.DataFrame | pd.DataFrame,
    *,
    demeaned: bool = True,
    group_adjust: bool = False,
    equal_weight: bool = False,
    factor_col: str = "factor",
    date_col: str = "date",
    asset_col: str = "asset",
    group_col: str = "group",
) -> pl.DataFrame:
    """Construct unit-gross factor weights, or zero weights for no signal.

    Magnitude weights optionally subtract the factor mean. Equal-sign weights
    optionally subtract the median, retain exact ties at zero and equalize
    active positive/negative sides before gross normalization. Group adjustment
    applies this rule within each date/group, then normalizes across the date.
    Active groups receive equal gross exposure; inactive groups stay zero.

    Inputs must have unique nonmissing date/asset keys and finite numeric factors.
    Normalization rescales finite magnitudes and removes a common offset before
    centering to avoid overflow and near-constant exposure cancellation.
    This helper does not clean, filter by future outcomes or construct buckets.
    Zero dispersion produces zero weights, not the undefined zero-normalization
    result in released Alphalens 0.4.6. No positions, expirations, capital or
    calendar accounting is inferred. Caller data and row order are preserved.

    Parameters
    ----------
    data : DataFrame
        Factor observations. Pandas date/asset MultiIndex input is accepted.
    demeaned, group_adjust, equal_weight : bool
        Mean/median centering, group-neutral normalization and equal-sign sizing.
        Group adjustment alone does not imply dollar neutrality.
    factor_col, date_col, asset_col, group_col : str
        Explicit physical input names. Nonmissing groups are required when used.

    Returns
    -------
    polars.DataFrame
        Canonical date/asset/factor, optional group, weight and explicit weighting
        flags. zero_weight_policy records zero exposure for no signal. Gross
        exposure is one when any signal remains, otherwise zero.
    """
    _weight_flags(demeaned=demeaned, group_adjust=group_adjust, equal_weight=equal_weight)
    values = _signal_frame(data)
    _validate_observation_keys(values, list(dict.fromkeys([date_col, asset_col])), "factor weights")
    if factor_col not in values.columns:
        raise pl.exceptions.ColumnNotFoundError(f"factor column {factor_col!r} was not found")
    if not values[factor_col].dtype.is_numeric():
        raise TypeError("factor weights require numeric factors")
    if not values[factor_col].cast(pl.Float64).is_finite().fill_null(False).all():
        raise ValueError("factor weights require finite factors; prepare the sample explicitly")
    if group_adjust:
        _validate_observation_keys(values, [group_col], "factor weight groups", unique=False)
    has_group = group_col in values.columns
    values = values.select(
        pl.col(date_col).alias("date"),
        pl.col(asset_col).alias("asset"),
        pl.col(factor_col).cast(pl.Float64).alias("factor"),
        *([pl.col(group_col).alias("group")] if has_group else []),
    ).with_row_index("__row_order")
    parts = []
    for part in values.partition_by(
        ["date", *(["group"] if group_adjust else [])], maintain_order=True
    ):
        weights = _weight_values(part["factor"].to_numpy(), demeaned, equal_weight)
        parts.append(part.with_columns(pl.Series("weight", weights)))
    result = (
        pl.concat(parts)
        if parts
        else values.with_columns(pl.lit(None, dtype=pl.Float64).alias("weight"))
    )
    if group_adjust:
        parts = [
            part.with_columns(
                pl.Series("weight", _weight_values(part["weight"].to_numpy(), False, False))
            )
            for part in result.partition_by("date", maintain_order=True)
        ]
        if parts:
            result = pl.concat(parts)
    return (
        result.sort("__row_order")
        .drop("__row_order")
        .with_columns(
            pl.lit(demeaned).alias("demeaned"),
            pl.lit(group_adjust).alias("group_adjust"),
            pl.lit(equal_weight).alias("equal_weight"),
            pl.lit("zero").alias("zero_weight_policy"),
        )
    )


def compute_factor_returns(
    data: pl.DataFrame | pd.DataFrame,
    *,
    return_columns: Sequence[str],
    demeaned: bool = True,
    group_adjust: bool = False,
    equal_weight: bool = False,
    by_asset: bool = False,
    factor_col: str = "factor",
    date_col: str = "date",
    asset_col: str = "asset",
    group_col: str = "group",
    date_grid: pl.DataFrame | pd.DataFrame | None = None,
) -> pl.DataFrame:
    """Multiply shared factor weights by supplied outcomes without reallocating.

    The weight universe depends only on factors and optional groups, never on
    a horizon's future availability. Nonfinite outcomes produce null contributions.
    Aggregate returns sum defined contributions with a minimum of one, retaining
    actual paired asset counts. All-zero weights yield zero contributions when
    outcomes are finite. These are forward-horizon arithmetic returns, not a
    self-financing daily wealth path or an annualized statistic.

    Parameters
    ----------
    data : DataFrame
        Unique date/asset factors, groups and numeric supplied outcomes.
    return_columns : sequence of str
        Explicit supplied selectors in output order, not inferred durations.
    demeaned, group_adjust, equal_weight : bool
        See compute_factor_weights. One shared weight calculation serves every
        horizon, including those with unavailable future tails.
    by_asset : bool
        Retain per-asset contributions instead of date-level sums.
    factor_col, date_col, asset_col, group_col : str
        Explicit physical input names.
    date_grid : DataFrame, optional
        Requested date_col keys for aggregate output, including zero-count/null
        holes. No calendar is inferred. Not accepted with by_asset=True.

    Returns
    -------
    polars.DataFrame
        Canonical date, optional asset/group/weight, return_column, weighted_return,
        n_obs and weighting flags. Counts concern defined contributions, not
        the original factor universe. No missing-outcome weight renormalization.
    """
    _weight_flags(
        demeaned=demeaned, group_adjust=group_adjust, equal_weight=equal_weight, by_asset=by_asset
    )
    if by_asset and date_grid is not None:
        raise ValueError("date_grid is supported only for aggregate factor returns")
    values, columns = _signal_inputs(
        data, return_columns, factor_col, date_col, asset_col, group_col if group_adjust else None
    )
    weights = compute_factor_weights(
        values,
        demeaned=demeaned,
        group_adjust=group_adjust,
        equal_weight=equal_weight,
        factor_col=factor_col,
        date_col=date_col,
        asset_col=asset_col,
        group_col=group_col,
    )
    has_group = "group" in weights.columns
    output = []
    for column in columns:
        contribution = pl.Series("outcome", values[column].cast(pl.Float64)) * weights["weight"]
        part = weights.select(
            "date", "asset", *(["group"] if has_group else []), "weight"
        ).with_columns(contribution.alias("weighted_return"))
        part = part.with_columns(
            pl.when(pl.col("weighted_return").is_finite())
            .then(pl.col("weighted_return"))
            .otherwise(None)
            .alias("weighted_return")
        )
        if by_asset:
            part = part.with_columns(
                pl.col("weighted_return").is_not_null().cast(pl.UInt32).alias("n_obs")
            )
        else:
            part = part.group_by("date").agg(
                pl.col("weighted_return").count().alias("n_obs"),
                pl.when(pl.col("weighted_return").count() > 0)
                .then(pl.col("weighted_return").sum())
                .otherwise(None)
                .alias("weighted_return"),
            )
            if date_grid is not None:
                grid = _signal_frame(date_grid)
                _validate_observation_keys(grid, [date_col], "factor return grid")
                grid = grid.select(pl.col(date_col).alias("date"))
                part, grid = _align_datetime_key_resolution(part, grid, "date")
                part = grid.join(part, on="date", how="left").with_columns(
                    pl.col("n_obs").fill_null(0)
                )
            part = part.sort("date")
        output.append(
            part.with_columns(
                pl.lit(column).alias("return_column"),
                pl.lit(demeaned).alias("demeaned"),
                pl.lit(group_adjust).alias("group_adjust"),
                pl.lit(equal_weight).alias("equal_weight"),
                pl.lit("zero").alias("zero_weight_policy"),
            )
        )
    return pl.concat(output)


def compute_factor_cumulative_returns(
    data: pl.DataFrame | pd.DataFrame,
    *,
    return_column: str,
    demeaned: bool = True,
    group_adjust: bool = False,
    equal_weight: bool = False,
    quantiles: Sequence[int] | None = None,
    groups: Sequence[str] | None = None,
    date_grid: pl.DataFrame | pd.DataFrame | None = None,
    factor_col: str = "factor",
    date_col: str = "date",
    asset_col: str = "asset",
    group_col: str = "group",
    quantile_col: str = "quantile",
) -> pl.DataFrame:
    """Compound selected factor returns in observation order, starting at one.

    Select buckets/groups before the existing shared weight calculation. Reuse
    compute_factor_returns without outcome-dependent weight renormalization.
    Nonfinite outcomes remain unavailable contributions; a date with no defined
    return has null weighted_return, zero n_obs and a flat wealth step. This
    matches the existing NaN-aware cumulative-product convention, not evidence
    that an unobserved return was zero. Caller inputs remain unchanged.

    return_column is an exact supplied selector, not a duration or calendar.
    Overlapping forward horizons compound mathematically in observation order;
    this is not self-financing daily holdings, expiration or capital accounting.
    Wealth starts at one; zero signal retains zero exposure. No annualization
    or calendar is inferred. Optional date_grid uses compute_factor_returns'
    explicit output-grid contract, including null unscored dates.

    Parameters
    ----------
    data : DataFrame
        Unique date/asset factors, supplied outcomes and optional group/bucket
        labels. Pandas date/asset MultiIndex inputs are accepted.
    return_column : str
        One exact supplied outcome column to compound.
    demeaned, group_adjust, equal_weight : bool
        Shared factor-weight options, applied after selection.
    quantiles, groups : sequence, optional
        Keep rows matching both selections before normalization. Empty or
        unmatched selections produce an empty path unless date_grid is supplied.
    date_grid : DataFrame, optional
        Explicit date_col keys for aggregate output. No schedule inferred.
    factor_col, date_col, asset_col, group_col, quantile_col : str
        Physical input column names.

    Returns
    -------
    polars.DataFrame
        Actual factor returns/counts/options plus wealth and explicit
        representation, missing-return policy and observed compounding clock.
    """
    from ml4t.diagnostic.evaluation.portfolio_analysis.metrics import _safe_cumprod
    from ml4t.diagnostic.signal._prepared import positive_count

    if not isinstance(return_column, str) or not return_column:
        raise ValueError("return_column must name one supplied return column")
    values = _signal_frame(data)
    for name, selection, column in [
        ("quantiles", quantiles, quantile_col),
        ("groups", groups, group_col),
    ]:
        if selection is not None:
            if isinstance(selection, str | bytes):
                raise ValueError(f"{name} must be a sequence, not a string")
            try:
                selected = list(selection)
            except TypeError as exc:
                raise ValueError(f"{name} must be a sequence") from exc
            if name == "quantiles":
                selected = [positive_count(value, "quantile") for value in selected]
            if column not in values.columns:
                raise ValueError(f"{name} selection requires column {column!r}")
            values = values.filter(pl.col(column).is_in(selected))
    returns = compute_factor_returns(
        values,
        return_columns=[return_column],
        demeaned=demeaned,
        group_adjust=group_adjust,
        equal_weight=equal_weight,
        date_grid=date_grid,
        factor_col=factor_col,
        date_col=date_col,
        asset_col=asset_col,
        group_col=group_col,
    )
    wealth = _safe_cumprod(1 + returns["weighted_return"].to_numpy())
    return returns.with_columns(
        pl.Series("wealth", wealth),
        pl.lit("wealth").alias("representation"),
        pl.lit("observed").alias("compounding_clock"),
        pl.lit("flat_wealth").alias("missing_return_policy"),
    )


__all__ = [
    "compute_factor_weights",
    "compute_factor_returns",
    "compute_factor_cumulative_returns",
]
