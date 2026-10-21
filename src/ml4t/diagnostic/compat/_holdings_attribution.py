"""Labeled Empyrical inputs over the shared supplied-holdings attribution kernel."""

import warnings

import pandas as pd
import polars as pl

from ml4t.diagnostic.evaluation.factor.holdings_attribution import (
    _attribute_exposures,
    _exposures,
)


def _panel(data: pd.Series | pd.DataFrame, *, weights: bool) -> pl.DataFrame:
    if not isinstance(data.index, pd.MultiIndex) or data.index.nlevels != 2:
        raise TypeError("holdings and loadings require a two-level date/asset MultiIndex")
    dates = data.index.get_level_values(0)
    if not isinstance(dates, pd.DatetimeIndex):
        raise TypeError("holdings and loadings require explicit datetime observation labels")
    assets = data.index.get_level_values(1)
    frame = pd.DataFrame({"timestamp": dates, "asset": assets})
    if weights:
        frame["weight"] = data.array
    else:
        for name in data.columns:
            frame[name] = data[name].array
    result = pl.from_pandas(frame)
    if not len(result) and result["asset"].dtype == pl.Null:
        template = pl.from_pandas(pd.Series(data.index.levels[1]))
        asset = template.head(0) if template.dtype != pl.Null else pl.Series([], dtype=pl.String)
        result = result.with_columns(asset.rename("asset"))
    return result


def compute_exposures(positions: pd.Series, factor_loadings: pd.DataFrame) -> pd.DataFrame:
    """Preserve observed-label alignment and warn when missing loadings are skipped.

    Positions are fractional NAV weights, not dollar holdings. The familiar
    observed-label method does not assert beginning-period information timing.
    """
    if not isinstance(positions, pd.Series) or not isinstance(factor_loadings, pd.DataFrame):
        raise TypeError("positions must be a Series and factor_loadings a DataFrame")
    if not factor_loadings.columns.is_unique or any(
        not isinstance(name, str) or not name or name in {"timestamp", "asset", "weight"}
        for name in factor_loadings.columns
    ):
        raise ValueError("factor names must be unique nonempty unambiguous strings")
    weights = _panel(positions.copy(deep=True), weights=True)
    loadings = _panel(factor_loadings.copy(deep=True), weights=False)
    left, right = positions.index.get_level_values(0), factor_loadings.index.get_level_values(0)
    if left.tz != right.tz:
        raise ValueError("weight and loading dates must use the same timezone")
    dates = left.union(right).unique().sort_values()
    labels = pl.from_pandas(dates)
    result, coverage = _exposures(
        weights,
        loadings,
        labels,
        tuple(factor_loadings.columns),
        starts=None,
        timing="legacy_observed",
        missing_loadings="legacy_skip",
    )
    if coverage["uncovered_gross_weight"].sum() > 0:
        warnings.warn(
            "Legacy attribution skips uncovered asset/factor loadings; zero contribution "
            "does not establish zero exposure. Native attribution rejects these gaps by default.",
            UserWarning,
            stacklevel=2,
        )
    if len(dates.difference(left)) or positions.isna().any():
        warnings.warn(
            "Legacy attribution treats absent or missing weights as zero. This does not "
            "establish complete observed account snapshots or causal availability.",
            UserWarning,
            stacklevel=2,
        )
    frame = result.select(factor_loadings.columns).to_pandas()
    frame.index = pd.DatetimeIndex(dates, name="dt")
    frame.index.freq = None
    return frame


def perf_attrib(
    returns: pd.Series,
    positions: pd.Series,
    factor_returns: pd.DataFrame,
    factor_loadings: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Continue the familiar supplied-weight workflow without fitting exposures.

    Legacy skip-missing sums and observation labels are deliberate. These inputs
    do not prove weights/loadings were available before each return interval.
    Caller tables and index names are never modified.
    """
    if not isinstance(returns, pd.Series) or not isinstance(returns.index, pd.DatetimeIndex):
        raise TypeError("returns must be a Series with datetime observation labels")
    if len(returns) == 0:
        raise ValueError("returns must contain at least one dated observation")
    if not isinstance(factor_returns, pd.DataFrame) or not isinstance(
        factor_returns.index, pd.DatetimeIndex
    ):
        raise TypeError("factor_returns must be a DataFrame with datetime observation labels")
    if returns.index.tz != factor_returns.index.tz:
        raise ValueError("factor and portfolio dates must use the same timezone")
    supplied = returns.copy(deep=True)
    factors = factor_returns.copy(deep=True).loc[supplied.index[0] : supplied.index[-1]]
    loadings = factor_loadings.copy(deep=True).loc[supplied.index[0] : supplied.index[-1]]
    exposures = compute_exposures(positions, loadings)
    if supplied.index.freq is not None:
        exposures = exposures.asfreq(supplied.index.freq)
    return exposures, _attribute_exposures(exposures, factors, supplied, legacy_missing=True)
