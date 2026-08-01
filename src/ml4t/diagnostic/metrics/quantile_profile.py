"""Cross-sectional quantile profiles for panel data."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import polars as pl
from scipy.stats import spearmanr


@dataclass(frozen=True)
class QuantileProfile:
    """Aggregated label behavior across feature quantiles.

    Attributes
    ----------
    means
        Mean label value for each one-based quantile.
    counts
        Number of valid observations in each quantile.
    monotonicity
        Spearman correlation between quantile number and mean label. This is
        NaN when any quantile has fewer than ``min_per_bucket`` observations.
    is_pooled
        Whether quantiles were assigned across all rows rather than within a
        grouping column.
    """

    means: dict[int, float]
    counts: dict[int, int]
    monotonicity: float
    is_pooled: bool


def _normalize_keys(keys: str | Sequence[str] | None) -> list[str]:
    if keys is None:
        return []
    if isinstance(keys, str):
        normalized = [keys]
    else:
        normalized = list(keys)
    if any(not isinstance(key, str) or not key for key in normalized):
        raise ValueError("keys must contain non-empty column names")
    if len(normalized) != len(set(normalized)):
        raise ValueError("keys must not contain duplicate column names")
    return normalized


def _validate_request(
    panel: pl.DataFrame,
    *,
    feature: str,
    label: str,
    n_quantiles: int,
    by: str | None,
    keys: list[str],
    min_per_bucket: int,
) -> None:
    if isinstance(n_quantiles, bool) or not isinstance(n_quantiles, int) or n_quantiles < 2:
        raise ValueError("n_quantiles must be at least 2")
    if (
        isinstance(min_per_bucket, bool)
        or not isinstance(min_per_bucket, int)
        or min_per_bucket < 1
    ):
        raise ValueError("min_per_bucket must be at least 1")
    if not isinstance(feature, str) or not feature:
        raise ValueError("feature must be a non-empty column name")
    if not isinstance(label, str) or not label:
        raise ValueError("label must be a non-empty column name")
    if by is not None and (not isinstance(by, str) or not by):
        raise ValueError("by must be a non-empty column name or None")

    required = {feature, label, *keys}
    if by is not None:
        required.add(by)
    missing = sorted(required.difference(panel.columns))
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    for column in (feature, label):
        if not panel.schema[column].is_numeric():
            raise ValueError(f"{column!r} must be numeric")

    if keys:
        key_frame = panel.select(keys)
        if key_frame.null_count().row(0) != tuple(0 for _ in keys):
            raise ValueError("keys cannot contain null values")
        if key_frame.is_duplicated().any():
            raise ValueError("keys must uniquely identify rows")


def quantile_profile(
    panel: pl.DataFrame,
    *,
    feature: str,
    label: str,
    n_quantiles: int = 5,
    by: str | None = "timestamp",
    keys: str | Sequence[str] | None = None,
    min_per_bucket: int = 20,
) -> QuantileProfile:
    """Summarize mean labels across feature quantiles.

    Quantiles are assigned within each value of ``by`` by default, preserving
    the cross-sectional ordering used by per-date IC. Pass ``by=None``
    explicitly to assign pooled quantiles across all observations.

    Dates or other groups with fewer than ``n_quantiles`` valid feature-label
    pairs are excluded before bucketing. The returned means remain available
    when aggregate bucket counts are small, but ``monotonicity`` is NaN unless
    every bucket contains at least ``min_per_bucket`` observations.

    Parameters
    ----------
    panel
        Polars panel containing the feature, label, grouping, and optional key
        columns.
    feature
        Numeric feature column used to assign quantiles.
    label
        Numeric outcome column averaged within each quantile.
    n_quantiles
        Number of quantiles. Must be at least two.
    by
        Column used for within-group quantile assignment. The default is
        ``"timestamp"``. Pass ``None`` for pooled assignment.
    keys
        Columns that uniquely identify panel rows. When provided, uniqueness is
        validated. Equal feature values receive the same average rank and the
        same quantile regardless of row order.
    min_per_bucket
        Minimum aggregate count required in every bucket before monotonicity is
        scored.

    Returns
    -------
    QuantileProfile
        Bucket means, counts, monotonicity, and pooled-assignment status.
    """
    if not isinstance(panel, pl.DataFrame):
        raise TypeError("panel must be a Polars DataFrame")

    key_columns = _normalize_keys(keys)
    _validate_request(
        panel,
        feature=feature,
        label=label,
        n_quantiles=n_quantiles,
        by=by,
        keys=key_columns,
        min_per_bucket=min_per_bucket,
    )

    valid_expr = (
        pl.col(feature).is_not_null()
        & pl.col(label).is_not_null()
        & pl.col(feature).is_finite()
        & pl.col(label).is_finite()
    )
    if by is not None:
        valid_expr &= pl.col(by).is_not_null()

    valid = panel.filter(valid_expr)
    if valid.is_empty():
        raise ValueError("No finite feature-label pairs are available")

    if key_columns:
        valid = valid.sort(key_columns)

    if by is None:
        if valid.height < n_quantiles:
            raise ValueError(f"Pooled profiles require at least {n_quantiles} valid observations")
        rank = pl.col(feature).rank(method="average")
        denominator: int | pl.Expr = valid.height
    else:
        valid = valid.filter(pl.len().over(by) >= n_quantiles)
        if valid.is_empty():
            raise ValueError(f"No groups contain at least {n_quantiles} valid observations")
        rank = pl.col(feature).rank(method="average").over(by)
        denominator = pl.len().over(by)

    bucketed = valid.with_columns(
        ((rank - 1) * n_quantiles / denominator).floor().cast(pl.Int64).add(1).alias("__quantile")
    )
    summary = (
        bucketed.group_by("__quantile")
        .agg(
            pl.col(label).mean().alias("__mean"),
            pl.len().alias("__count"),
        )
        .sort("__quantile")
    )

    means = dict.fromkeys(range(1, n_quantiles + 1), float("nan"))
    counts = dict.fromkeys(range(1, n_quantiles + 1), 0)
    for quantile, mean, count in summary.iter_rows():
        means[int(quantile)] = float(mean)
        counts[int(quantile)] = int(count)

    mean_values = np.asarray(list(means.values()), dtype=float)
    enough_observations = all(count >= min_per_bucket for count in counts.values())
    if (
        enough_observations
        and np.isfinite(mean_values).all()
        and not np.all(mean_values == mean_values[0])
    ):
        monotonicity = float(spearmanr(range(1, n_quantiles + 1), mean_values).statistic)
    else:
        monotonicity = float("nan")

    return QuantileProfile(
        means=means,
        counts=counts,
        monotonicity=monotonicity,
        is_pooled=by is None,
    )
