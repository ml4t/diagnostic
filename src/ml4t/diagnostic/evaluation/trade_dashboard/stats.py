"""Dashboard statistical computations.

Pure statistical functions for the dashboard, including PSR (Probabilistic
Sharpe Ratio) which replaces the incorrectly-used DSR for single-strategy analysis.
"""

from __future__ import annotations

from typing import Any, Literal, overload

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import norm

from ml4t.diagnostic.evaluation.trade_dashboard.types import ReturnSummary


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


@overload
def probabilistic_sharpe_ratio(
    observed_sharpe: float,
    benchmark_sharpe: float = ...,
    n_samples: int = ...,
    skewness: float = ...,
    kurtosis: float = ...,
    return_components: Literal[False] = ...,
) -> float: ...


@overload
def probabilistic_sharpe_ratio(
    observed_sharpe: float,
    benchmark_sharpe: float = ...,
    n_samples: int = ...,
    skewness: float = ...,
    kurtosis: float = ...,
    return_components: Literal[True] = ...,
) -> dict[str, float]: ...


def probabilistic_sharpe_ratio(
    observed_sharpe: float,
    benchmark_sharpe: float = 0.0,
    n_samples: int = 1,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
    return_components: bool = False,
) -> float | dict[str, float]:
    """Calculate Probabilistic Sharpe Ratio (PSR).

    PSR gives the probability that the true Sharpe ratio exceeds a benchmark,
    accounting for sample size and return distribution characteristics.

    Unlike DSR (which corrects for multiple testing across K strategies),
    PSR is applicable to a SINGLE strategy's performance evaluation.

    Parameters
    ----------
    observed_sharpe : float
        Observed Sharpe ratio of the strategy.
    benchmark_sharpe : float, default 0.0
        Benchmark Sharpe ratio (typically 0 for testing significance).
    n_samples : int, default 1
        Number of return observations (T).
    skewness : float, default 0.0
        Skewness of returns distribution.
    kurtosis : float, default 3.0
        Kurtosis of returns (3.0 for normal, NOT excess kurtosis).
    return_components : bool, default False
        If True, return dict with intermediate calculations.

    Returns
    -------
    float or dict
        PSR probability in [0, 1], or dict with 'psr', 'z_score', 'std_sr'.

    Notes
    -----
    Formula (Bailey & Lopez de Prado 2012):

        PSR = Phi[(SR - SR_0) * sqrt(T-1) / sqrt(1 - gamma_3*SR + (gamma_4-1)/4*SR^2)]

    where:
        - SR = observed Sharpe ratio
        - SR_0 = benchmark Sharpe ratio
        - T = number of samples
        - gamma_3 = skewness
        - gamma_4 = kurtosis (not excess)
        - Phi = standard normal CDF

    Interpretation:
        - PSR > 0.95: 95% confidence true SR > benchmark (significant at alpha=0.05)
        - PSR < 0.50: More likely true SR < benchmark
        - PSR = 0.50: No evidence either way

    Examples
    --------
    >>> psr = probabilistic_sharpe_ratio(
    ...     observed_sharpe=1.5,
    ...     benchmark_sharpe=0.0,
    ...     n_samples=252,
    ...     skewness=-0.5,
    ...     kurtosis=4.0,
    ... )
    >>> print(f"PSR: {psr:.3f}")
    PSR: 0.987

    References
    ----------
    Bailey, D. H., & Lopez de Prado, M. (2012).
    "The Sharpe Ratio Efficient Frontier."
    Journal of Risk, 15(2), 3-44.
    """
    if n_samples < 2:
        # Need at least 2 samples for meaningful calculation
        if return_components:
            return {"psr": 0.5, "z_score": 0.0, "std_sr": np.inf}
        return 0.5

    # Calculate denominator of z-score
    # V[SR] = 1 - gamma_3*SR + (gamma_4-1)/4*SR^2
    sr_squared = observed_sharpe**2
    variance_component = 1 - skewness * observed_sharpe + (kurtosis - 1) / 4 * sr_squared

    # Guard against negative variance (can happen with extreme skewness)
    if variance_component <= 0:
        variance_component = 0.01  # Small positive value

    std_sr = np.sqrt(variance_component / (n_samples - 1))

    # Calculate z-score
    if std_sr > 0:
        z_score = (observed_sharpe - benchmark_sharpe) / std_sr
    else:
        z_score = np.inf if observed_sharpe > benchmark_sharpe else -np.inf

    # Convert to probability
    psr = float(norm.cdf(z_score))

    if return_components:
        return {
            "psr": psr,
            "z_score": float(z_score) if np.isfinite(z_score) else 0.0,
            "std_sr": float(std_sr),
        }

    return psr


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
            pass

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
            pass

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
            pass

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
            pass

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
            pass

    if not results:
        return pd.DataFrame(columns=["test", "statistic", "p_value", "interpretation"])

    return pd.DataFrame(results)


def benjamini_hochberg_fdr(
    p_values: list[float] | np.ndarray,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Apply Benjamini-Hochberg FDR correction.

    Parameters
    ----------
    p_values : list or ndarray
        Raw p-values.
    alpha : float, default 0.05
        Target FDR level.

    Returns
    -------
    dict
        - rejected: boolean array of rejected hypotheses
        - adjusted_p_values: BH-adjusted p-values
        - n_rejected: number of rejections
    """
    from ml4t.diagnostic.evaluation.stats import benjamini_hochberg_fdr as bh_fdr

    result = bh_fdr(p_values, alpha=alpha, return_details=True)
    return {
        "rejected": result["rejected"],
        "adjusted_p_values": result["adjusted_p_values"],
        "n_rejected": result["n_rejected"],
    }
