"""Autocorrelation analysis for time series features.

Provides ACF (autocorrelation function) and PACF (partial autocorrelation function)
analysis with confidence intervals and ARIMA order suggestions.

Key Functions:
    compute_acf: Autocorrelation function with confidence intervals
    compute_pacf: Partial autocorrelation function with confidence intervals
    analyze_autocorrelation: Combined ACF/PACF analysis with ARIMA order suggestion
"""

from __future__ import annotations

from typing import Any, Literal, cast

import numpy as np
import pandas as pd
import polars as pl
from statsmodels.tsa.stattools import acf, pacf

from ml4t.diagnostic.backends.adapter import DataFrameAdapter
from ml4t.diagnostic.errors import ComputationError, ValidationError
from ml4t.diagnostic.logging import get_logger

logger = get_logger(__name__)


# =============================================================================
# Result Class
# =============================================================================


class CorrelationResult:
    """Results from autocorrelation (ACF) or partial autocorrelation (PACF) analysis.

    Attributes:
        values: Correlation coefficients for each lag (length: nlags+1).
                values[0] = 1.0 (correlation with itself).
        conf_int: Confidence intervals (shape: (nlags+1, 2)).
        lags: Lag indices (0, 1, 2, ..., nlags).
        alpha: Significance level for confidence intervals.
        n_obs: Number of observations used.
        method: Estimation method used.
        kind: Type of correlation ('acf' or 'pacf').
    """

    def __init__(
        self,
        values: np.ndarray | None = None,
        conf_int: np.ndarray | None = None,
        lags: np.ndarray | None = None,
        alpha: float = 0.05,
        n_obs: int = 0,
        method: str = "standard",
        kind: Literal["acf", "pacf"] | None = None,
        # Backward compat aliases
        acf_values: np.ndarray | None = None,
        pacf_values: np.ndarray | None = None,
    ):
        # Handle backward compat: acf_values/pacf_values -> values + kind
        if acf_values is not None:
            values = acf_values
            kind = "acf"
        elif pacf_values is not None:
            values = pacf_values
            kind = "pacf"

        if values is None:
            raise ValueError("Must provide values, acf_values, or pacf_values")
        if conf_int is None:
            raise ValueError("Must provide conf_int")
        if lags is None:
            raise ValueError("Must provide lags")
        if kind is None:
            raise ValueError("Must provide kind")

        self.values = values
        self.conf_int = conf_int
        self.lags = lags
        self.alpha = alpha
        self.n_obs = n_obs
        self.method = method
        self.kind = kind

    @property
    def significant_lags(self) -> list[int]:
        """Lags where correlation is significantly different from zero."""
        significant = []
        for i in range(1, len(self.values)):
            if self.conf_int[i, 0] > 0 or self.conf_int[i, 1] < 0:
                significant.append(int(self.lags[i]))
        return significant

    # Backward compatibility aliases
    @property
    def acf_values(self) -> np.ndarray:
        """ACF values (alias for values when kind='acf')."""
        return self.values

    @property
    def pacf_values(self) -> np.ndarray:
        """PACF values (alias for values when kind='pacf')."""
        return self.values

    def __repr__(self) -> str:
        sig_count = len(self.significant_lags)
        total_lags = len(self.lags) - 1
        kind_upper = self.kind.upper()
        base = f"{kind_upper}Result(n_obs={self.n_obs}, nlags={total_lags}, significant={sig_count}/{total_lags}, alpha={self.alpha}"
        if self.kind == "pacf":
            base += f", method='{self.method}'"
        return base + ")"

    def __str__(self) -> str:
        kind_upper = self.kind.upper()
        lines = [
            f"{kind_upper} Analysis Results:",
            f"  Observations: {self.n_obs}",
            f"  Lags analyzed: {len(self.lags) - 1}",
            f"  Significance level: {self.alpha}",
            f"  Method: {self.method}",
            f"  Significant lags: {len(self.significant_lags)}",
        ]
        if self.significant_lags:
            lines.append(f"  Lags: {self.significant_lags[:10]}")
            if len(self.significant_lags) > 10:
                lines[-1] += " ..."
        return "\n".join(lines)


# Backward compatibility type aliases
ACFResult = CorrelationResult
PACFResult = CorrelationResult


# =============================================================================
# Shared Validation
# =============================================================================


def _validate_and_prepare(
    data: pl.Series | pd.Series | np.ndarray,
    nlags: int | None,
    kind: Literal["acf", "pacf"],
    missing: Literal["none", "raise", "conservative", "drop"] = "none",
) -> tuple[np.ndarray, int]:
    """Validate input data and prepare for ACF/PACF computation.

    Args:
        data: Time series data.
        nlags: Number of lags (None for auto).
        kind: Type of correlation ('acf' or 'pacf').
        missing: How to handle missing values.

    Returns:
        Tuple of (clean_values, nlags).

    Raises:
        ValidationError: If data is invalid.
    """
    # Convert to 1D numpy consistently across pandas/polars/ndarray inputs.
    values = DataFrameAdapter.to_numpy(data).astype(np.float64).reshape(-1)

    # Check empty
    if len(values) == 0:
        raise ValidationError(
            f"Cannot compute {kind.upper()} for empty data",
            context={"data_length": 0},
        )

    # Handle missing values
    if missing == "raise" and np.any(np.isnan(values)):
        nan_count = int(np.sum(np.isnan(values)))
        raise ValidationError(
            "Data contains NaN values",
            context={"nan_count": nan_count, "total_count": len(values)},
        )
    elif missing in ["conservative", "drop"] and np.any(np.isnan(values)):
        original_length = len(values)
        values = values[~np.isnan(values)]
        logger.info(
            "Dropped NaN values",
            original_length=original_length,
            clean_length=len(values),
        )

    # Check all NaN
    if len(values) == 0:
        raise ValidationError("All data is NaN after missing value handling")

    # Minimum observations
    min_obs = 5 if kind == "pacf" else 3
    if len(values) < min_obs:
        raise ValidationError(
            f"Insufficient data for {kind.upper()} computation (need at least {min_obs} observations)",
            context={"n_obs": len(values)},
        )

    n_obs = len(values)

    # Determine nlags
    if nlags is None:
        max_lag = n_obs // 2 - 1 if kind == "pacf" else n_obs - 1
        nlags = int(min(10 * np.log10(n_obs), max_lag))
        logger.debug(f"Auto-selected nlags for {kind.upper()}", nlags=nlags, n_obs=n_obs)
    else:
        if nlags < 0:
            raise ValidationError("nlags must be non-negative", context={"nlags": nlags})

        max_lag = n_obs // 2 if kind == "pacf" else n_obs
        if nlags >= max_lag:
            msg = (
                "nlags must be less than n_obs/2 for PACF"
                if kind == "pacf"
                else "nlags must be less than number of observations"
            )
            raise ValidationError(
                msg,
                context={"nlags": nlags, "n_obs": n_obs, "max_nlags": max_lag - 1},
            )

        if nlags > n_obs // 4:
            logger.warning(
                "Large nlags may produce unreliable results",
                nlags=nlags,
                n_obs=n_obs,
            )

    return values, nlags


# =============================================================================
# Public API
# =============================================================================


def compute_acf(
    data: pl.Series | pd.Series | np.ndarray,
    nlags: int | None = None,
    alpha: float = 0.05,
    fft: bool = False,
    missing: Literal["none", "raise", "conservative", "drop"] = "none",
) -> CorrelationResult:
    """Compute autocorrelation function (ACF) with confidence intervals.

    Args:
        data: Time series data.
        nlags: Number of lags. If None, uses min(10*log10(n), n-1).
        alpha: Significance level for confidence intervals.
        fft: Use FFT for faster computation.
        missing: How to handle missing values.

    Returns:
        CorrelationResult with ACF values and confidence intervals.

    Raises:
        ValidationError: If data is invalid.
        ComputationError: If computation fails.
    """
    logger.debug("Computing ACF", fft=fft, missing_handling=missing)

    values, nlags = _validate_and_prepare(data, nlags, "acf", missing)
    n_obs = len(values)

    try:
        acf_values, conf_int = acf(values, nlags=nlags, alpha=alpha, fft=fft, missing=missing)
    except Exception as e:
        raise ComputationError(
            f"Failed to compute ACF: {e}",
            context={"n_obs": n_obs, "nlags": nlags},
            cause=e,
        ) from None

    result = CorrelationResult(
        values=acf_values,
        conf_int=conf_int,
        lags=np.arange(len(acf_values)),
        alpha=alpha,
        n_obs=n_obs,
        method="fft" if fft else "standard",
        kind="acf",
    )

    logger.info("ACF computed", n_obs=n_obs, nlags=nlags, significant=len(result.significant_lags))
    return result


def compute_pacf(
    data: pl.Series | pd.Series | np.ndarray,
    nlags: int | None = None,
    alpha: float = 0.05,
    method: Literal[
        "ywadjusted", "yw_adjusted", "ols", "ld", "ldadjusted", "ld_adjusted"
    ] = "ywadjusted",
) -> CorrelationResult:
    """Compute partial autocorrelation function (PACF) with confidence intervals.

    PACF measures direct correlation with lag k, controlling for intermediate lags.
    Key for identifying AR order: PACF cuts off after lag p for AR(p) processes.

    Args:
        data: Time series data.
        nlags: Number of lags. If None, uses min(10*log10(n), n//2-1).
        alpha: Significance level for confidence intervals.
        method: Estimation method ('ywadjusted', 'ols', 'ld', etc.).

    Returns:
        CorrelationResult with PACF values and confidence intervals.

    Raises:
        ValidationError: If data is invalid.
        ComputationError: If computation fails.
    """
    logger.debug("Computing PACF", method=method)

    # PACF always drops NaN (statsmodels.pacf doesn't have missing parameter)
    values, nlags = _validate_and_prepare(data, nlags, "pacf", missing="drop")
    n_obs = len(values)

    try:
        method_normalized = cast(Any, method.replace("_", ""))
        pacf_values, conf_int = pacf(values, nlags=nlags, alpha=alpha, method=method_normalized)
    except Exception as e:
        raise ComputationError(
            f"Failed to compute PACF: {e}",
            context={"n_obs": n_obs, "nlags": nlags, "method": method},
            cause=e,
        ) from None

    result = CorrelationResult(
        values=pacf_values,
        conf_int=conf_int,
        lags=np.arange(len(pacf_values)),
        alpha=alpha,
        n_obs=n_obs,
        method=method,
        kind="pacf",
    )

    logger.info("PACF computed", n_obs=n_obs, nlags=nlags, significant=len(result.significant_lags))
    return result


# =============================================================================
# Analysis
# =============================================================================


class AutocorrelationAnalysisResult:
    """Combined ACF and PACF analysis with ARIMA order suggestions.

    Attributes:
        acf_result: ACF analysis result.
        pacf_result: PACF analysis result.
        suggested_ar_order: AR order (p) from PACF cutoff.
        suggested_ma_order: MA order (q) from ACF cutoff.
        suggested_d_order: Always 0 (assess stationarity separately).
        is_white_noise: True if no significant autocorrelation.
        summary_df: DataFrame with ACF/PACF side-by-side.
    """

    def __init__(
        self,
        acf_result: CorrelationResult,
        pacf_result: CorrelationResult,
        suggested_ar_order: int,
        suggested_ma_order: int,
        is_white_noise: bool,
        summary_df: pd.DataFrame,
        # Backward compat - allow passing these explicitly
        significant_acf_lags: list[int] | None = None,
        significant_pacf_lags: list[int] | None = None,
    ):
        self.acf_result = acf_result
        self.pacf_result = pacf_result
        self.suggested_ar_order = suggested_ar_order
        self.suggested_ma_order = suggested_ma_order
        self.suggested_d_order = 0
        # Use passed values if provided, otherwise derive from results
        self.significant_acf_lags = (
            significant_acf_lags
            if significant_acf_lags is not None
            else acf_result.significant_lags
        )
        self.significant_pacf_lags = (
            significant_pacf_lags
            if significant_pacf_lags is not None
            else pacf_result.significant_lags
        )
        self.is_white_noise = is_white_noise
        self.summary_df = summary_df

    @property
    def suggested_arima_order(self) -> tuple[int, int, int]:
        """Suggested ARIMA(p, d, q) order."""
        return (self.suggested_ar_order, self.suggested_d_order, self.suggested_ma_order)

    def __repr__(self) -> str:
        p, d, q = self.suggested_arima_order
        return (
            f"AutocorrelationAnalysisResult(n_obs={self.acf_result.n_obs}, "
            f"ARIMA({p},{d},{q}), white_noise={self.is_white_noise})"
        )

    def __str__(self) -> str:
        lines = [
            "Autocorrelation Analysis Results:",
            f"  Observations: {self.acf_result.n_obs}",
            f"  Lags analyzed: {len(self.acf_result.lags) - 1}",
            f"  Significance level: {self.acf_result.alpha}",
            "",
            f"ACF: {len(self.significant_acf_lags)} significant lags",
            f"PACF: {len(self.significant_pacf_lags)} significant lags",
            "",
            f"White noise: {self.is_white_noise}",
            f"Suggested ARIMA order: {self.suggested_arima_order}",
        ]

        if self.is_white_noise:
            lines.append("Interpretation: No autocorrelation detected (random process)")
        elif self.suggested_ar_order > 0 and self.suggested_ma_order == 0:
            lines.append(f"Interpretation: AR({self.suggested_ar_order}) process detected")
        elif self.suggested_ar_order == 0 and self.suggested_ma_order > 0:
            lines.append(f"Interpretation: MA({self.suggested_ma_order}) process detected")
        elif self.suggested_ar_order > 0 and self.suggested_ma_order > 0:
            lines.append(
                f"Interpretation: ARMA({self.suggested_ar_order},{self.suggested_ma_order}) process detected"
            )

        return "\n".join(lines)


def analyze_autocorrelation(
    data: pl.Series | pd.Series | np.ndarray,
    max_lags: int | None = None,
    alpha: float = 0.05,
    acf_method: Literal["standard", "fft"] = "standard",
    pacf_method: Literal[
        "ywadjusted", "yw_adjusted", "ols", "ld", "ldadjusted", "ld_adjusted"
    ] = "ywadjusted",
) -> AutocorrelationAnalysisResult:
    """Perform combined ACF/PACF analysis with ARIMA order suggestion.

    Args:
        data: Time series data.
        max_lags: Maximum lags for both ACF and PACF.
        alpha: Significance level for confidence intervals.
        acf_method: ACF method ('standard' or 'fft').
        pacf_method: PACF estimation method.

    Returns:
        AutocorrelationAnalysisResult with suggested ARIMA orders.
    """
    logger.info("Starting autocorrelation analysis")

    # Compute both ACF and PACF
    acf_result = compute_acf(data, nlags=max_lags, alpha=alpha, fft=(acf_method == "fft"))
    pacf_result = compute_pacf(data, nlags=max_lags, alpha=alpha, method=pacf_method)

    # Determine if white noise
    is_white_noise = (
        len(acf_result.significant_lags) == 0 and len(pacf_result.significant_lags) == 0
    )

    # Suggest ARIMA orders
    suggested_ar_order = _suggest_order(pacf_result)
    suggested_ma_order = _suggest_order(acf_result)

    # Create summary DataFrame
    summary_df = _create_summary_dataframe(acf_result, pacf_result)

    result = AutocorrelationAnalysisResult(
        acf_result=acf_result,
        pacf_result=pacf_result,
        suggested_ar_order=suggested_ar_order,
        suggested_ma_order=suggested_ma_order,
        is_white_noise=is_white_noise,
        summary_df=summary_df,
    )

    logger.info(
        "Autocorrelation analysis completed",
        arima_order=result.suggested_arima_order,
        white_noise=is_white_noise,
    )
    return result


# =============================================================================
# Helpers
# =============================================================================


def _suggest_order(result: CorrelationResult) -> int:
    """Suggest AR order (from PACF) or MA order (from ACF) based on cutoff pattern.

    For AR(p): PACF cuts off after lag p.
    For MA(q): ACF cuts off after lag q.
    """
    significant_set = set(result.significant_lags)
    if not significant_set:
        return 0

    cutoff_lag = 0
    for lag in range(1, len(result.lags)):
        if lag in significant_set and lag == cutoff_lag + 1:
            cutoff_lag = lag
        else:
            break
    return cutoff_lag


def _create_summary_dataframe(
    acf_result: CorrelationResult, pacf_result: CorrelationResult
) -> pd.DataFrame:
    """Create DataFrame with ACF and PACF side-by-side (excluding lag 0)."""
    lags = acf_result.lags[1:]
    acf_sig_set = set(acf_result.significant_lags)
    pacf_sig_set = set(pacf_result.significant_lags)

    return pd.DataFrame(
        {
            "lag": lags,
            "acf_value": acf_result.values[1:],
            "acf_significant": [lag in acf_sig_set for lag in lags],
            "acf_ci_lower": acf_result.conf_int[1:, 0],
            "acf_ci_upper": acf_result.conf_int[1:, 1],
            "pacf_value": pacf_result.values[1:],
            "pacf_significant": [lag in pacf_sig_set for lag in lags],
            "pacf_ci_lower": pacf_result.conf_int[1:, 0],
            "pacf_ci_upper": pacf_result.conf_int[1:, 1],
        }
    )
