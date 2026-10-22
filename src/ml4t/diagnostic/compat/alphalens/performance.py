"""Familiar statistical tables over the existing Diagnostic signal kernels."""

from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl

from ml4t.diagnostic.compat.alphalens.utils import _Keys, _profile, get_forward_returns_columns
from ml4t.diagnostic.signal.quantile import _quantile_mean_spread, compute_quantile_statistics
from ml4t.diagnostic.signal.signal_ic import compute_signal_ic

__all__ = [
    "factor_information_coefficient",
    "mean_information_coefficient",
    "mean_return_by_quantile",
    "compute_mean_returns_spread",
]


def _statistical_inputs(factor_data: pd.DataFrame) -> tuple[pl.DataFrame, list[str]]:
    if not isinstance(factor_data, pd.DataFrame):
        raise TypeError("factor_data must be a pandas DataFrame")
    index_sample = factor_data.get("factor", pd.Series(index=factor_data.index, dtype=float))
    keys = _Keys(index_sample)
    return pl.from_pandas(keys.encode(factor_data).reset_index()), list(
        get_forward_returns_columns(factor_data.columns)
    )


def _wide_statistics(
    observations: pl.DataFrame,
    *,
    value: str,
    dimensions: list[str],
    columns: list[str],
    original: pd.DataFrame,
) -> pd.DataFrame:
    result = observations.to_pandas().pivot(index=dimensions, columns="return_column", values=value)
    result = result.reindex(columns=columns)
    result.columns.name = None
    if "group" in dimensions and isinstance(original["group"].dtype, pd.CategoricalDtype):
        result.index = result.index.set_levels(
            pd.CategoricalIndex(
                result.index.levels[dimensions.index("group")], dtype=original["group"].dtype
            ),
            level="group",
        )
    return result.sort_index()


def factor_information_coefficient(
    factor_data: pd.DataFrame,
    group_adjust: bool = False,
    by_group: bool = False,
    *,
    reference_version: str = "0.5.0",
) -> pd.DataFrame:
    """Adapt native dated ICs, retaining legacy per-column NaN propagation."""
    profile = _profile(reference_version)
    data, columns = _statistical_inputs(factor_data)
    observations = compute_signal_ic(
        data, return_columns=columns, group_adjust=group_adjust, by_group=by_group
    )
    result = _wide_statistics(
        observations,
        value="ic",
        dimensions=["date", *(["group"] if by_group else [])],
        columns=columns,
        original=factor_data,
    )
    invalid = (
        factor_data[columns].isna().apply(lambda column: column | factor_data["factor"].isna())
    )
    grouping = [
        factor_data.index.get_level_values("date"),
        *([factor_data["group"]] if by_group else []),
    ]
    result = result.mask(
        invalid.groupby(grouping, observed=True).any().reindex(result.index).fillna(False)
    )
    if not by_group:
        frequency = factor_data.index.levels[0].freq
        if frequency is not None or profile == "0.4.6":
            result = result.asfreq(frequency)
    return result


def mean_information_coefficient(
    factor_data: pd.DataFrame,
    group_adjust: bool = False,
    by_group: bool = False,
    by_time: str | pd.DateOffset | None = None,
    *,
    reference_version: str = "0.5.0",
) -> pd.Series | pd.DataFrame:
    """Aggregate the same IC observations using the caller's pandas time rule."""
    result = factor_information_coefficient(
        factor_data, group_adjust, by_group, reference_version=reference_version
    )
    grouping = ([pd.Grouper(freq=by_time)] if by_time is not None else []) + (
        ["group"] if by_group else []
    )
    if not grouping:
        return result.mean()
    return result.reset_index().set_index("date").groupby(grouping, observed=False).mean()


def mean_return_by_quantile(
    factor_data: pd.DataFrame,
    by_date: bool = False,
    by_group: bool = False,
    demeaned: bool = True,
    group_adjust: bool = False,
    *,
    reference_version: str = "0.5.0",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Adapt shared equal-date means/errors without re-bucketing or score filtering."""
    _profile(reference_version)
    data, columns = _statistical_inputs(factor_data)
    statistics = compute_quantile_statistics(
        data,
        return_columns=columns,
        by_date=by_date,
        by_group=by_group,
        demeaned=demeaned,
        group_adjust=group_adjust,
        factor_col=None,
        quantile_col="factor_quantile",
    ).rename({"quantile": "factor_quantile"})
    dimensions = [
        "factor_quantile",
        *(["date"] if by_date else []),
        *(["group"] if by_group else []),
    ]
    return tuple(
        _wide_statistics(
            statistics, value=value, dimensions=dimensions, columns=columns, original=factor_data
        )
        for value in ("mean", "standard_error")
    )


def compute_mean_returns_spread(
    mean_returns: pd.DataFrame,
    upper_quant: int,
    lower_quant: int,
    std_err: pd.DataFrame | None = None,
    *,
    reference_version: str = "0.5.0",
) -> tuple[pd.Series | pd.DataFrame, pd.Series | pd.DataFrame | None]:
    """Use shared spread/error arithmetic with explicit unit-gross or frozen raw scale."""
    profile = _profile(reference_version)
    upper, lower = (
        mean_returns.xs(upper_quant, level="factor_quantile"),
        mean_returns.xs(lower_quant, level="factor_quantile"),
    )
    upper, lower = upper.align(lower)
    error_upper = error_lower = None
    if std_err is not None:
        error_upper, error_lower = (
            std_err.xs(upper_quant, level="factor_quantile"),
            std_err.xs(lower_quant, level="factor_quantile"),
        )
        error_upper, error_lower = error_upper.align(error_lower)
    spread, error = _quantile_mean_spread(
        upper.to_numpy(dtype=float, na_value=np.nan),
        lower.to_numpy(dtype=float, na_value=np.nan),
        error_upper.to_numpy(dtype=float, na_value=np.nan) if error_upper is not None else None,
        error_lower.to_numpy(dtype=float, na_value=np.nan) if error_lower is not None else None,
    )
    scale = 0.5 if profile == "0.5.0" else 1.0
    difference = upper.copy()
    difference.iloc[...] = spread * scale
    if error_upper is None:
        return difference, None
    uncertainty = error_upper.copy()
    uncertainty.iloc[...] = error * scale
    return difference, uncertainty
