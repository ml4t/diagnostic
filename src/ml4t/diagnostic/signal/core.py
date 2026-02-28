"""Core signal analysis functions.

The main entry point is `analyze_signal()` - one function for 95% of use cases.
For power users, `prepare_data()` allows custom workflows.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from ml4t.diagnostic.signal._utils import (
    QuantileMethod,
    compute_forward_returns,
    ensure_polars,
    filter_outliers,
    quantize_factor,
)
from ml4t.diagnostic.signal.quantile import (
    compute_monotonicity,
    compute_quantile_returns,
    compute_spread,
)
from ml4t.diagnostic.signal.result import SignalResult
from ml4t.diagnostic.signal.signal_ic import compute_ic_series, compute_ic_summary
from ml4t.diagnostic.signal.turnover import (
    compute_autocorrelation,
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
        "quantile" (equal frequency) or "uniform" (equal width).
    factor_col, date_col, asset_col, price_col : str
        Column names.

    Returns
    -------
    pl.DataFrame
        Prepared data with: date, asset, factor, quantile, {period}D_fwd_return.
    """
    # Convert to Polars
    factor_pl = ensure_polars(factor)
    prices_pl = ensure_polars(prices)

    # Compute forward returns
    data = compute_forward_returns(factor_pl, prices_pl, periods, date_col, asset_col, price_col)

    # Filter outliers
    if filter_zscore is not None and filter_zscore > 0:
        data = filter_outliers(data, filter_zscore, factor_col, date_col)

    # Assign quantiles
    method = QuantileMethod.QUANTILE if quantile_method == "quantile" else QuantileMethod.UNIFORM
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
        "quantile" (equal frequency) or "uniform" (equal width).
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

    # Extract metadata
    n_assets = data.select(asset_col).n_unique()
    n_dates = data.select(date_col).n_unique()
    all_dates = data.select(date_col).unique().sort(date_col).to_series().to_list()
    date_range = (str(all_dates[0]), str(all_dates[-1])) if all_dates else ("", "")

    # Initialize result dicts
    ic: dict[str, float] = {}
    ic_std: dict[str, float] = {}
    ic_t_stat: dict[str, float] = {}
    ic_p_value: dict[str, float] = {}
    ic_ir: dict[str, float] = {}
    ic_positive_pct: dict[str, float] = {}
    ic_series: dict[str, list[float]] = {}
    quantile_returns: dict[str, dict[int, float]] = {}
    spread: dict[str, float] = {}
    spread_t_stat: dict[str, float] = {}
    spread_p_value: dict[str, float] = {}
    monotonicity: dict[str, float] = {}

    # Compute metrics for each period
    for period in periods:
        period_key = f"{period}D"

        # IC
        dates, ic_vals = compute_ic_series(
            data, period, ic_method, factor_col, date_col, asset_col, min_assets
        )
        summary = compute_ic_summary(ic_vals)

        ic[period_key] = summary["mean"]
        ic_std[period_key] = summary["std"]
        ic_t_stat[period_key] = summary["t_stat"]
        ic_p_value[period_key] = summary["p_value"]
        ic_series[period_key] = ic_vals

        # IC Information Ratio and positive percentage
        if summary["std"] > 0:
            ic_ir[period_key] = summary["mean"] / summary["std"]
        else:
            ic_ir[period_key] = 0.0
        if ic_vals:
            ic_positive_pct[period_key] = sum(1 for x in ic_vals if x > 0) / len(ic_vals) * 100
        else:
            ic_positive_pct[period_key] = 0.0

        # Quantile returns
        q_returns = compute_quantile_returns(data, period, quantiles)
        quantile_returns[period_key] = q_returns

        # Spread
        spread_stats = compute_spread(data, period, quantiles)
        spread[period_key] = spread_stats["spread"]
        spread_t_stat[period_key] = spread_stats["t_stat"]
        spread_p_value[period_key] = spread_stats["p_value"]

        # Monotonicity
        monotonicity[period_key] = compute_monotonicity(q_returns)

    # Turnover (optional)
    turnover_dict: dict[str, float] | None = None
    autocorr: list[float] | None = None
    half_life: float | None = None

    if compute_turnover_flag:
        turnover_val = compute_turnover(data, quantiles, date_col, asset_col)
        turnover_dict = {f"{p}D": turnover_val for p in periods}

        lags = list(range(1, autocorrelation_lags + 1))
        autocorr = compute_autocorrelation(data, lags, date_col, asset_col, factor_col)
        half_life = estimate_half_life(autocorr)

    return SignalResult(
        ic=ic,
        ic_std=ic_std,
        ic_t_stat=ic_t_stat,
        ic_p_value=ic_p_value,
        ic_ir=ic_ir,
        ic_positive_pct=ic_positive_pct,
        ic_series=ic_series,
        quantile_returns=quantile_returns,
        spread=spread,
        spread_t_stat=spread_t_stat,
        spread_p_value=spread_p_value,
        monotonicity=monotonicity,
        turnover=turnover_dict,
        autocorrelation=autocorr,
        half_life=half_life,
        n_assets=n_assets,
        n_dates=n_dates,
        date_range=date_range,
        periods=periods,
        quantiles=quantiles,
    )


__all__ = ["prepare_data", "analyze_signal"]
