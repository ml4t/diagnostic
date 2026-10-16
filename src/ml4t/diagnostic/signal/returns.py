"""Inspectable outcome adjustments shared by prepared signal analyses."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
import polars as pl

from ml4t.diagnostic.signal._utils import _signal_inputs


def _paired_demean(
    values: pl.DataFrame,
    columns: Sequence[str],
    factor_col: str,
    date_col: str,
    group_col: str | None,
) -> pl.DataFrame:
    if values.is_empty():
        return values.clone()
    grouping = [date_col, *([group_col] if group_col is not None else [])]
    selected = list(dict.fromkeys([*grouping, factor_col, *columns]))
    grouped = values.select(selected).to_pandas()
    for column in columns:
        grouped[column] = grouped[column].where(
            np.isfinite(grouped[factor_col]) & np.isfinite(grouped[column])
        )
    adjusted = grouped.groupby(grouping, observed=True, sort=False)[list(columns)].transform(
        lambda outcome: outcome - outcome.mean()
    )
    return values.with_columns(pl.Series(column, adjusted[column].to_numpy()) for column in columns)


def demean_signal_returns(
    data: pl.DataFrame | pd.DataFrame,
    *,
    return_columns: Sequence[str],
    by_group: bool = False,
    factor_col: str = "factor",
    date_col: str = "date",
    asset_col: str = "asset",
    group_col: str = "group",
) -> pl.DataFrame:
    """Subtract the paired date mean or date/group mean from supplied outcomes.

    Finite factor/outcome pairs define each horizon's mean. Unavailable pairs
    become NaN outcomes; rows, keys, factors, groups and buckets are retained.
    Owned common-sample panels already contain finite values at every horizon.
    No new returns, buckets, weights, durations or calendars are constructed.

    Parameters
    ----------
    data : DataFrame
        Unique date/asset observations; Pandas MultiIndex input is also accepted.
        The caller is not modified. The returned frame preserves other columns.
    return_columns : sequence of str
        Explicit supplied outcome columns, not inferred duration labels.
    by_group : bool
        Use date/group means rather than date means. This is not group-neutral
        portfolio construction or a group-level statistical summary.
    factor_col, date_col, asset_col, group_col : str
        Explicit input column/index names.

    Returns
    -------
    polars.DataFrame
        Independent data with adjusted outcome columns in the original row order.
    """
    if not isinstance(by_group, bool | np.bool_):
        raise ValueError("by_group must be boolean")
    values, columns = _signal_inputs(
        data, return_columns, factor_col, date_col, asset_col, group_col if by_group else None
    )
    return _paired_demean(values, columns, factor_col, date_col, group_col if by_group else None)


__all__ = ["demean_signal_returns"]
