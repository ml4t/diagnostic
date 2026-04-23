"""Dashboard statistical computations.

Pure statistical functions for the dashboard, including Probabilistic Sharpe
Ratio calculations for single-strategy analysis.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy import stats

from ml4t.diagnostic.evaluation.trade_dashboard.types import ReturnSummary

logger = logging.getLogger(__name__)


def compute_return_summary(returns: np.ndarray) -> ReturnSummary:
    """Compute summary statistics for a returns series.

    Parameters
    ----------
    returns : np.ndarray
        Array of returns (can be return_pct or pnl).

    Returns
    -------
    ReturnSummary
        Summary statistics including mean, std, Sharpe, skewness, kurtosis.
    """
    n = len(returns)
    if n == 0:
        return ReturnSummary(
            n_samples=0,
            mean=np.nan,
            std=np.nan,
            sharpe=np.nan,
            skewness=np.nan,
            kurtosis=np.nan,
            min_val=np.nan,
            max_val=np.nan,
            win_rate=np.nan,
        )

    mean = float(np.mean(returns))
    std = float(np.std(returns, ddof=1)) if n > 1 else 0.0
    sharpe = mean / std if std > 0 else np.nan

    # Skewness and kurtosis require minimum samples
    skewness = float(stats.skew(returns)) if n > 2 else 0.0
    # Use Fisher=False to get actual kurtosis (3.0 for normal), not excess
    kurtosis = float(stats.kurtosis(returns, fisher=False)) if n > 3 else 3.0

    win_rate = float(np.mean(returns > 0))

    return ReturnSummary(
        n_samples=n,
        mean=mean,
        std=std,
        sharpe=sharpe,
        skewness=skewness,
        kurtosis=kurtosis,
        min_val=float(np.min(returns)),
        max_val=float(np.max(returns)),
        win_rate=win_rate,
    )


def compute_distribution_tests(
    returns: np.ndarray,
) -> pd.DataFrame:
    """Compute distribution tests for returns.

    Parameters
    ----------
    returns : np.ndarray
        Array of returns.

    Returns
    -------
    pd.DataFrame
        DataFrame with test results:
        - test: Test name
        - statistic: Test statistic
        - p_value: P-value
        - interpretation: Human-readable interpretation
    """
    results = []

    n = len(returns)

    # Shapiro-Wilk test (for n <= 5000)
    if 3 <= n <= 5000:
        try:
            from scipy.stats import shapiro

            stat, p = shapiro(returns)
            results.append(
                {
                    "test": "Shapiro-Wilk",
                    "statistic": stat,
                    "p_value": p,
                    "interpretation": "Normal" if p > 0.05 else "Non-normal",
                }
            )
        except Exception:
            logger.debug("Shapiro-Wilk test failed", exc_info=True)

    # Anderson-Darling test
    if n >= 4:
        try:
            from scipy.stats import anderson

            result = anderson(returns, dist="norm")
            # Use 5% significance level
            critical_idx = 2  # Index for 5% level
            stat = result.statistic
            critical = result.critical_values[critical_idx]
            is_normal = stat < critical
            results.append(
                {
                    "test": "Anderson-Darling",
                    "statistic": stat,
                    "p_value": None,  # Anderson doesn't provide p-value directly
                    "interpretation": "Normal" if is_normal else "Non-normal",
                }
            )
        except Exception:
            logger.debug("Anderson-Darling test failed", exc_info=True)

    # Jarque-Bera test
    if n >= 20:
        try:
            from scipy.stats import jarque_bera

            stat, p = jarque_bera(returns)
            results.append(
                {
                    "test": "Jarque-Bera",
                    "statistic": stat,
                    "p_value": p,
                    "interpretation": "Normal" if p > 0.05 else "Non-normal",
                }
            )
        except Exception:
            logger.debug("Jarque-Bera test failed", exc_info=True)

    if not results:
        return pd.DataFrame(columns=["test", "statistic", "p_value", "interpretation"])

    return pd.DataFrame(results)


def compute_time_series_tests(
    returns: np.ndarray,
    max_lags: int = 10,
) -> pd.DataFrame:
    """Compute time-series tests (requires chronologically sorted data).

    Parameters
    ----------
    returns : np.ndarray
        Array of returns (MUST be in chronological order).
    max_lags : int, default 10
        Maximum lags for Ljung-Box test.

    Returns
    -------
    pd.DataFrame
        DataFrame with test results.

    Notes
    -----
    These tests are only meaningful on chronologically ordered data.
    The dashboard normalizes data by sorting trades by entry_time.
    """
    results = []

    n = len(returns)

    # Ljung-Box test for autocorrelation
    if n > max_lags + 5:
        try:
            from statsmodels.stats.diagnostic import acorr_ljungbox

            lb_result = acorr_ljungbox(returns, lags=[max_lags], return_df=True)
            stat = lb_result["lb_stat"].iloc[0]
            p = lb_result["lb_pvalue"].iloc[0]
            results.append(
                {
                    "test": f"Ljung-Box (lag={max_lags})",
                    "statistic": stat,
                    "p_value": p,
                    "interpretation": "No autocorrelation"
                    if p > 0.05
                    else "Autocorrelation detected",
                }
            )
        except Exception:
            logger.debug("Ljung-Box autocorrelation test failed", exc_info=True)

    # ADF test for stationarity
    if n >= 20:
        try:
            from statsmodels.tsa.stattools import adfuller

            adf_result = adfuller(returns, autolag="AIC")
            stat = adf_result[0]
            p = adf_result[1]
            results.append(
                {
                    "test": "ADF (stationarity)",
                    "statistic": stat,
                    "p_value": p,
                    "interpretation": "Stationary" if p < 0.05 else "Non-stationary",
                }
            )
        except Exception:
            logger.debug("ADF stationarity test failed", exc_info=True)

    if not results:
        return pd.DataFrame(columns=["test", "statistic", "p_value", "interpretation"])

    return pd.DataFrame(results)
