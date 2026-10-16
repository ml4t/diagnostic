"""Core signal analysis functions.

`analyze_signal()` starts from factors/prices; `analyze_prepared_signal()` retains
an existing return panel and its buckets. Both use the same analysis functions.
"""

from __future__ import annotations

from dataclasses import replace
from importlib.metadata import version
from typing import TYPE_CHECKING

import polars as pl

from ml4t.diagnostic.signal._prepared import positive_count, prepared_snapshot
from ml4t.diagnostic.signal._utils import (
    QuantileMethod,
    compute_forward_returns,
    ensure_polars,
    filter_outliers,
    precise_dates,
    quantize_factor,
)
from ml4t.diagnostic.signal.quantile import (
    _compute_quantile_return_statistics,
    compute_quantile_statistics,
    compute_spread,
    monotonicity_score,
)
from ml4t.diagnostic.signal.result import SignalResult
from ml4t.diagnostic.signal.signal_ic import compute_ic_summary, extract_signal_ic_series
from ml4t.diagnostic.signal.turnover import (
    compute_autocorrelation,
    compute_quantile_turnover,
    compute_rank_autocorrelation,
    compute_turnover,
    estimate_half_life,
)

if TYPE_CHECKING:
    import pandas as pd


def prepare_data(
    factor: pl.DataFrame | pd.DataFrame,
    prices: pl.DataFrame | pd.DataFrame,
    periods: tuple[int, ...] = (1, 5, 21),
    quantiles: int = 5,
    filter_zscore: float | None = 3.0,
    quantile_method: str = "quantile",
    factor_col: str = "factor",
    date_col: str = "date",
    asset_col: str = "asset",
    price_col: str = "price",
) -> pl.DataFrame:
    """Prepare factor data for analysis.

    Joins factor with prices, computes forward returns, filters outliers,
    and assigns quantiles.

    Parameters
    ----------
    factor : DataFrame
        Factor data with columns: date, asset, factor.
    prices : DataFrame
        Price data with columns: date, asset, price.
    periods : tuple[int, ...]
        Forward return periods in trading days.
    quantiles : int
        Number of quantiles.
    filter_zscore : float | None
        Z-score threshold for outlier filtering. None disables.
    quantile_method : str
        "qcut" uses interpolated equal-frequency edges on finite factors.
        "quantile" (default) retains legacy rank buckets; "uniform" retains
        legacy epsilon-adjusted equal-width buckets. Unknown methods raise.
    factor_col, date_col, asset_col, price_col : str
        Column names.

    Returns
    -------
    pl.DataFrame
        Prepared data with: date, asset, factor, quantile, {period}D_fwd_return.
    """
    method = QuantileMethod(quantile_method)

    # Convert to Polars
    factor_pl = ensure_polars(factor)
    prices_pl = ensure_polars(prices)

    # Compute forward returns
    data = compute_forward_returns(factor_pl, prices_pl, periods, date_col, asset_col, price_col)

    # Filter outliers
    if filter_zscore is not None and filter_zscore > 0:
        data = filter_outliers(data, filter_zscore, factor_col, date_col)

    # Assign quantiles
    data = quantize_factor(data, quantiles, method, factor_col, date_col)

    return data


def analyze_signal(
    factor: pl.DataFrame | pd.DataFrame,
    prices: pl.DataFrame | pd.DataFrame,
    *,
    periods: tuple[int, ...] = (1, 5, 21),
    quantiles: int = 5,
    filter_zscore: float | None = 3.0,
    quantile_method: str = "quantile",
    ic_method: str = "spearman",
    compute_turnover_flag: bool = True,
    autocorrelation_lags: int = 10,
    min_assets: int = 10,
    factor_col: str = "factor",
    date_col: str = "date",
    asset_col: str = "asset",
    price_col: str = "price",
) -> SignalResult:
    """Analyze a factor signal.

    This is the main entry point for signal analysis. Computes IC, quantile
    returns, spread, monotonicity, and optionally turnover/autocorrelation.

    Parameters
    ----------
    factor : DataFrame
        Factor data with columns: date, asset, factor.
        Higher factor values should predict higher returns.
    prices : DataFrame
        Price data with columns: date, asset, price.
    periods : tuple[int, ...]
        Forward return periods in trading days (default: 1, 5, 21 days).
    quantiles : int
        Number of quantiles for grouping assets (default: 5 quintiles).
    filter_zscore : float | None
        Z-score threshold for outlier filtering. None disables.
    quantile_method : str
        "qcut" uses interpolated equal-frequency edges on finite factors.
        "quantile" (default) and "uniform" retain their distinct legacy methods.
    ic_method : str
        "spearman" (rank correlation) or "pearson" (linear correlation).
    compute_turnover_flag : bool
        Whether to compute turnover and autocorrelation metrics.
    autocorrelation_lags : int
        Number of lags for autocorrelation analysis.
    min_assets : int
        Minimum assets per date for IC computation.
    factor_col, date_col, asset_col, price_col : str
        Column names.

    Returns
    -------
    SignalResult
        Analysis results with IC, quantile returns, spread, monotonicity,
        and optionally turnover metrics.

    Examples
    --------
    Basic usage:

    >>> result = analyze_signal(factor_df, prices_df)
    >>> print(result.summary())
    >>> result.to_json("results.json")

    With custom parameters:

    >>> result = analyze_signal(
    ...     factor_df, prices_df,
    ...     periods=(1, 5, 21, 63),
    ...     quantiles=10,
    ...     ic_method="pearson",
    ... )
    """
    # Prepare data
    data = prepare_data(
        factor,
        prices,
        periods,
        quantiles,
        filter_zscore,
        quantile_method,
        factor_col,
        date_col,
        asset_col,
        price_col,
    )
    result = _analyze_data(
        data,
        periods,
        quantiles,
        ic_method,
        compute_turnover_flag,
        autocorrelation_lags,
        min_assets,
        factor_col,
        date_col,
        asset_col,
    )
    if quantile_method == QuantileMethod.QCUT:
        result = replace(
            result,
            preparation_metadata={
                "bucket_method": "pandas.qcut",
                "pandas_version": version("pandas"),
                "numpy_version": version("numpy"),
                "sample_policy": "existing price path; no complete-case cleaner or loss audit",
            },
        )
    return result


def analyze_prepared_signal(
    data: pl.DataFrame | pd.DataFrame,
    *,
    periods: tuple[int, ...] = (1, 5, 21),
    quantiles: int = 5,
    ic_method: str = "spearman",
    compute_turnover_flag: bool = False,
    autocorrelation_lags: int = 10,
    min_assets: int = 10,
    factor_col: str = "factor",
    date_col: str = "date",
    asset_col: str = "asset",
) -> SignalResult:
    """Analyze an owned prepared sample without prices or reassigned buckets.

    Required columns are date, asset, factor, quantile and ``{period}D_fwd_return``
    for every requested period. An Alphalens pandas MultiIndex table can instead
    supply ``factor_quantile`` and ``{period}D`` columns. Both representations
    together are ambiguous and rejected. Duplicate keys, missing/nonfinite values
    and invalid buckets raise before calculations; this route does not clean data.

    Supplied horizon labels are selectors, not computed durations. No calendar,
    session grid or return endpoint is inferred or validated. Optional turnover
    retains the existing observed-factor-date lag convention. IC/quantile/spread
    calculations retain their native definitions, including pooled quantile
    statistics and two-sample spread inference, not all Alphalens conventions.

    The result exposes an independent ``prepared_data`` snapshot and records its
    sample metadata. JSON preserves that snapshot through an Arrow IPC payload.
    This ingestion route does not implement grouped/neutral analytics or complete
    legacy report workflows.
    """
    periods = tuple(positive_count(period, "periods") for period in periods)
    quantiles = positive_count(quantiles, "quantiles")
    min_assets = positive_count(min_assets, "min_assets")
    if compute_turnover_flag:
        autocorrelation_lags = positive_count(autocorrelation_lags, "autocorrelation_lags")
    owned, payload, metadata = prepared_snapshot(
        data, periods, quantiles, factor_col, date_col, asset_col
    )
    result = _analyze_data(
        owned,
        tuple(periods),
        quantiles,
        ic_method,
        compute_turnover_flag,
        autocorrelation_lags,
        min_assets,
        factor_col,
        date_col,
        asset_col,
    )
    return replace(result, _prepared_data=payload, preparation_metadata=metadata)


def _analyze_data(
    data: pl.DataFrame,
    periods: tuple[int, ...],
    quantiles: int,
    ic_method: str,
    compute_turnover_flag: bool,
    autocorrelation_lags: int,
    min_assets: int,
    factor_col: str,
    date_col: str,
    asset_col: str,
) -> SignalResult:
    # Extract metadata
    n_assets = data.select(asset_col).n_unique()
    n_dates = data.select(date_col).n_unique()
    all_dates = precise_dates(data.select(date_col).unique().sort(date_col).to_series())
    date_range = (str(all_dates[0]), str(all_dates[-1])) if all_dates else ("", "")

    # Count by quantile (period-independent, compute once)
    count_by_quantile: dict[int, int] = {}
    for q_val, cnt in data.group_by("quantile").len().sort("quantile").iter_rows():
        count_by_quantile[int(q_val)] = cnt

    # Initialize result dicts
    ic: dict[str, float] = {}
    ic_std: dict[str, float] = {}
    ic_t_stat: dict[str, float] = {}
    ic_p_value: dict[str, float] = {}
    ic_ir: dict[str, float] = {}
    ic_positive_pct: dict[str, float] = {}
    ic_series: dict[str, list[float]] = {}
    ic_dates: dict[str, list[str]] = {}
    quantile_returns: dict[str, dict[int, float]] = {}
    quantile_returns_std: dict[str, dict[int, float]] = {}
    spread: dict[str, float] = {}
    spread_t_stat: dict[str, float] = {}
    spread_p_value: dict[str, float] = {}
    spread_std: dict[str, float] = {}
    monotonicity: dict[str, float] = {}

    # Compute metrics for each period
    for period in periods:
        period_key = f"{period}D"

        # IC
        dates, ic_vals = extract_signal_ic_series(
            data, period, ic_method, factor_col, date_col, asset_col, min_assets
        )
        summary = compute_ic_summary(ic_vals)

        ic[period_key] = summary["mean"]
        ic_std[period_key] = summary["std"]
        ic_t_stat[period_key] = summary["t_stat"]
        ic_p_value[period_key] = summary["p_value"]
        ic_series[period_key] = ic_vals
        ic_dates[period_key] = [str(d) for d in dates]

        # IC Information Ratio and positive percentage
        if summary["std"] > 0:
            ic_ir[period_key] = summary["mean"] / summary["std"]
        else:
            ic_ir[period_key] = float("nan")
        ic_positive_pct[period_key] = summary["pct_positive"] * 100

        # Quantile returns
        q_returns, q_std = _compute_quantile_return_statistics(data, period, quantiles)
        quantile_returns[period_key] = q_returns
        quantile_returns_std[period_key] = q_std

        # Spread
        spread_stats = compute_spread(data, period, quantiles)
        spread[period_key] = spread_stats["spread"]
        spread_t_stat[period_key] = spread_stats["t_stat"]
        spread_p_value[period_key] = spread_stats["p_value"]

        # Spread std (derive from identity: se = spread / t_stat)
        t = spread_stats["t_stat"]
        if abs(t) > 1e-12:
            spread_std[period_key] = abs(spread_stats["spread"] / t)
        else:
            spread_std[period_key] = float("nan")

        # Monotonicity
        monotonicity[period_key] = monotonicity_score(q_returns)

    quantile_observations: bytes | None = None
    if periods:
        columns = {f"{period}D_fwd_return": f"{period}D" for period in periods}
        observations = compute_quantile_statistics(
            data,
            return_columns=list(columns),
            by_date=True,
            demeaned=False,
            factor_col=factor_col,
            date_col=date_col,
            asset_col=asset_col,
        ).with_columns(pl.col("return_column").replace_strict(columns))
        quantile_observations = observations.write_ipc(None).getvalue()

    # Turnover (optional)
    turnover_dict: dict[str, float] | None = None
    autocorr: list[float] | None = None
    half_life: float | None = None
    dated_turnover: bytes | None = None
    dated_autocorrelation: bytes | None = None

    if compute_turnover_flag:
        turnover_dict = {
            f"{p}D": compute_turnover(data, quantiles, date_col, asset_col, lag=p) for p in periods
        }

        lags = list(range(1, autocorrelation_lags + 1))
        autocorr = compute_autocorrelation(data, lags, date_col, asset_col, factor_col)
        half_life = estimate_half_life(autocorr)
        dated_turnover = (
            compute_quantile_turnover(
                data, lags=tuple(dict.fromkeys(periods)), date_col=date_col, asset_col=asset_col
            )
            .write_ipc(None)
            .getvalue()
        )
        dated_autocorrelation = (
            compute_rank_autocorrelation(
                data, lags=lags, date_col=date_col, asset_col=asset_col, factor_col=factor_col
            )
            .write_ipc(None)
            .getvalue()
        )

    return SignalResult(
        ic=ic,
        ic_std=ic_std,
        ic_t_stat=ic_t_stat,
        ic_p_value=ic_p_value,
        ic_ir=ic_ir,
        ic_positive_pct=ic_positive_pct,
        ic_series=ic_series,
        ic_dates=ic_dates,
        quantile_returns=quantile_returns,
        quantile_returns_std=quantile_returns_std,
        count_by_quantile=count_by_quantile,
        spread=spread,
        spread_t_stat=spread_t_stat,
        spread_p_value=spread_p_value,
        spread_std=spread_std,
        monotonicity=monotonicity,
        turnover=turnover_dict,
        autocorrelation=autocorr,
        half_life=half_life,
        _dated_turnover=dated_turnover,
        _dated_autocorrelation=dated_autocorrelation,
        _quantile_observations=quantile_observations,
        n_assets=n_assets,
        n_dates=n_dates,
        date_range=date_range,
        periods=periods,
        quantiles=quantiles,
    )


__all__ = ["prepare_data", "analyze_signal", "analyze_prepared_signal"]
