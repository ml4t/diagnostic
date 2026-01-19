"""Tail analysis for heavy-tailed distributions.

This module provides tools for analyzing tail behavior:
- Hill estimator for tail index estimation
- Q-Q plots for distribution comparison
- Comprehensive tail analysis combining multiple methods

Tail Classification:
    - Heavy tails (α ≤ 2): Infinite variance regime, extreme power law behavior
    - Medium tails (2 < α ≤ 4): Finite variance, infinite 4th moment
    - Thin tails (α > 4): All moments finite, close to normal

References:
    - Hill, B. M. (1975). A simple general approach to inference about the tail
      of a distribution. The Annals of Statistics, 3(5), 1163-1174.
    - Mandelbrot, B. (1963). The variation of certain speculative prices.
      The Journal of Business, 36(4), 394-419.
    - Clauset, A., Shalizi, C. R., & Newman, M. E. (2009). Power-law distributions
      in empirical data. SIAM Review, 51(4), 661-703.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from ml4t.diagnostic.errors import ComputationError, ValidationError
from ml4t.diagnostic.logging import get_logger

logger = get_logger(__name__)


@dataclass
class HillEstimatorResult:
    """Hill estimator for tail index of heavy-tailed distributions.

    The Hill estimator estimates the tail index α (alpha) for power law distributions.
    Higher α indicates thinner tails. The tail index characterizes how quickly the
    probability density decays in the tail:

        P(X > x) ~ x^(-α)

    Tail Classification:
        - Heavy tails (α ≤ 2): Infinite variance regime, extreme power law behavior
        - Medium tails (2 < α ≤ 4): Finite variance, infinite 4th moment
        - Thin tails (α > 4): All moments finite, approaching normal

    Attributes:
        tail_index: Estimated tail index α (higher = thinner tail)
        tail_index_se: Standard error of tail index estimate
        k: Number of upper order statistics used in estimation
        classification: Tail classification ("heavy", "medium", "thin")
        tail: Which tail was analyzed ("upper", "lower", "both")
        n_obs: Total number of observations

    Notes:
        - SE(α̂) = α̂ / sqrt(k)
        - Financial returns typically have α ∈ [2, 4] (medium tails)
        - Normal distribution has α → ∞ (exponential tail decay)
    """

    tail_index: float
    tail_index_se: float
    k: int
    classification: str
    tail: str
    n_obs: int

    def __repr__(self) -> str:
        """String representation."""
        return f"HillEstimatorResult(alpha={self.tail_index:.4f}, classification='{self.classification}', k={self.k})"

    def summary(self) -> str:
        """Human-readable summary of Hill estimator analysis.

        Returns:
            Formatted summary string
        """
        lines = [
            "Hill Estimator - Tail Index Analysis",
            "=" * 50,
            f"Tail Index (α):     {self.tail_index:.4f}",
            f"Standard Error:     {self.tail_index_se:.4f}",
            f"Z-score:            {self.tail_index / self.tail_index_se:.4f}",
            f"Order Statistics:   k={self.k}",
            f"Total Observations: n={self.n_obs}",
            f"Tail Analyzed:      {self.tail}",
        ]

        lines.append("")
        lines.append(f"Classification: {self.classification.upper()}")

        if self.classification == "heavy":
            interpretation = [
                "  - Infinite variance regime (α ≤ 2)",
                "  - Extreme power law behavior",
                "  - Mean may not exist for α ≤ 1",
                "  - Very high probability of extreme events",
                "  - Standard risk measures (VaR, Sharpe) unreliable",
            ]
        elif self.classification == "medium":
            interpretation = [
                "  - Finite variance but heavy-tailed (2 < α ≤ 4)",
                "  - Fourth moment may not exist",
                "  - Higher extreme event probability than normal",
                "  - Typical for financial returns",
                "  - Use robust risk measures (CVaR, drawdown)",
            ]
        else:  # thin
            interpretation = [
                "  - All moments finite (α > 4)",
                "  - Tail behavior approaching normal distribution",
                "  - Standard statistical methods applicable",
                "  - Lower extreme event probability",
            ]

        lines.extend(interpretation)

        lines.append("")
        lines.append("Methodology:")
        lines.append("  - Hill estimator: α̂ = k / Σ(log(X_i) - log(X_{k+1}))")
        lines.append(f"  - Uses k={self.k} largest order statistics")
        lines.append("  - Asymptotic SE: α̂ / sqrt(k)")

        return "\n".join(lines)


@dataclass
class QQPlotData:
    """Q-Q plot data for distribution comparison.

    Quantile-Quantile (Q-Q) plots compare empirical quantiles against theoretical
    quantiles from a reference distribution. If data follows the reference distribution,
    points should lie on the diagonal line y=x.

    Attributes:
        theoretical_quantiles: Quantiles from reference distribution
        sample_quantiles: Empirical quantiles from data
        distribution: Reference distribution name ("normal", "t", "uniform", etc.)
        r_squared: R² goodness of fit (closer to 1 = better fit)
        df: Degrees of freedom (for t-distribution, None otherwise)
        n_obs: Number of observations
    """

    theoretical_quantiles: np.ndarray
    sample_quantiles: np.ndarray
    distribution: str
    r_squared: float
    df: int | None = None
    n_obs: int = 0

    def __repr__(self) -> str:
        """String representation."""
        df_str = f", df={self.df}" if self.df is not None else ""
        return f"QQPlotData(distribution='{self.distribution}', R²={self.r_squared:.4f}{df_str})"

    def summary(self) -> str:
        """Human-readable summary of QQ plot analysis.

        Returns:
            Formatted summary string
        """
        lines = [
            f"Q-Q Plot Analysis - {self.distribution.title()} Distribution",
            "=" * 50,
            f"Reference Dist:     {self.distribution}",
            f"R² (Goodness):      {self.r_squared:.4f}",
            f"Observations:       {self.n_obs}",
        ]

        if self.df is not None:
            lines.append(f"Degrees of Freedom: {self.df}")

        lines.append("")
        if self.r_squared >= 0.99:
            fit_quality = "Excellent"
            interpretation = "Data closely follows reference distribution"
        elif self.r_squared >= 0.95:
            fit_quality = "Good"
            interpretation = "Data reasonably follows reference distribution"
        elif self.r_squared >= 0.90:
            fit_quality = "Moderate"
            interpretation = "Some deviation from reference distribution"
        else:
            fit_quality = "Poor"
            interpretation = "Significant deviation from reference distribution"

        lines.append(f"Fit Quality: {fit_quality}")
        lines.append(f"  {interpretation}")

        lines.append("")
        lines.append("Interpretation:")
        lines.append("  - Points on diagonal => data follows reference distribution")
        lines.append("  - Deviations in tails => different tail behavior")
        lines.append("  - S-shaped pattern => skewness difference")
        lines.append("  - Curved pattern => kurtosis difference")

        return "\n".join(lines)


@dataclass
class TailAnalysisResult:
    """Comprehensive tail analysis combining Hill estimator and QQ plots.

    Analyzes tail behavior by:
    1. Estimating tail index using Hill estimator
    2. Comparing against normal distribution (QQ plot)
    3. Comparing against Student's t distribution (QQ plot)
    4. Determining best-fit distribution

    This multi-method approach provides robust characterization of tail behavior
    and helps identify appropriate distributional assumptions for modeling.

    Attributes:
        hill_result: Hill estimator analysis results
        qq_normal: QQ plot comparison with normal distribution
        qq_t: QQ plot comparison with Student's t (None if not computed)
        best_fit: Best fitting distribution ("normal", "t", "heavy-tailed")
    """

    hill_result: HillEstimatorResult
    qq_normal: QQPlotData
    qq_t: QQPlotData | None
    best_fit: str

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"TailAnalysisResult(tail_index={self.hill_result.tail_index:.4f}, "
            f"classification='{self.hill_result.classification}', "
            f"best_fit='{self.best_fit}')"
        )

    def summary(self) -> str:
        """Human-readable summary of comprehensive tail analysis.

        Returns:
            Formatted summary string
        """
        lines = [
            "Comprehensive Tail Analysis",
            "=" * 50,
        ]

        # Hill estimator summary
        lines.append("")
        lines.append("TAIL INDEX ESTIMATION:")
        lines.append(f"  Hill α:            {self.hill_result.tail_index:.4f}")
        lines.append(f"  Classification:    {self.hill_result.classification}")
        lines.append(f"  Tail Type:         {self.hill_result.tail}")

        # QQ plot comparisons
        lines.append("")
        lines.append("DISTRIBUTION COMPARISON:")
        lines.append(f"  Normal R²:         {self.qq_normal.r_squared:.4f}")
        if self.qq_t is not None:
            lines.append(f"  Student's t R²:    {self.qq_t.r_squared:.4f} (df={self.qq_t.df})")

        lines.append(f"  Best Fit:          {self.best_fit}")

        # Interpretation
        lines.append("")
        lines.append("INTERPRETATION:")
        if self.best_fit == "normal":
            interpretation = [
                "  - Data is consistent with normal distribution",
                "  - Thin tails (low extreme event probability)",
                "  - Standard statistical methods appropriate",
            ]
        elif self.best_fit == "t":
            interpretation = [
                f"  - Data best fit by Student's t (df={self.qq_t.df if self.qq_t else 'unknown'})",
                "  - Heavier tails than normal but finite variance",
                "  - Moderate extreme event probability",
                "  - Use robust statistical methods",
            ]
        else:  # heavy-tailed
            interpretation = [
                "  - Data exhibits heavy tail behavior",
                "  - Power law distribution indicated",
                "  - High extreme event probability",
                "  - Standard risk measures may be unreliable",
                "  - Consider tail risk models (CVaR, extreme value theory)",
            ]

        lines.extend(interpretation)

        # Recommendations
        lines.append("")
        lines.append("RECOMMENDATIONS:")
        if self.hill_result.classification == "heavy":
            lines.append("  - Use tail risk measures (CVaR, expected shortfall)")
            lines.append("  - Consider extreme value theory for VaR")
            lines.append("  - Apply robust portfolio optimization")
            lines.append("  - Monitor for regime changes")
        elif self.hill_result.classification == "medium":
            lines.append("  - Use robust Sharpe ratio alternatives")
            lines.append("  - Consider CVaR alongside VaR")
            lines.append("  - Account for non-normality in models")
        else:
            lines.append("  - Standard statistical methods appropriate")
            lines.append("  - Monitor for changes in tail behavior")

        return "\n".join(lines)


def hill_estimator(
    data: pd.Series | np.ndarray,
    k: int | None = None,
    tail: str = "both",
) -> HillEstimatorResult:
    """Estimate tail index using Hill estimator.

    The Hill estimator computes the tail index α for power law distributions.
    For a power law tail P(X > x) ~ x^(-α), the Hill estimator is:

        α̂ = k / Σ(log(X_i) - log(X_{k+1}))

    where X_1 ≥ X_2 ≥ ... ≥ X_n are order statistics and k is the number of
    upper order statistics used.

    Tail Classification:
        - Heavy tails (α ≤ 2): Infinite variance regime
        - Medium tails (2 < α ≤ 4): Finite variance, heavy-tailed
        - Thin tails (α > 4): All moments finite

    Args:
        data: Time series data (1D array or Series)
        k: Number of upper order statistics (default: sqrt(n))
        tail: Which tail to analyze - "upper", "lower", or "both" (default)

    Returns:
        HillEstimatorResult with tail index and classification

    Raises:
        ValidationError: If data is invalid
        ComputationError: If estimation fails

    Example:
        >>> import numpy as np
        >>> # Student's t distribution (df=3) has heavy tails
        >>> t_data = np.random.standard_t(df=3, size=1000)
        >>> result = hill_estimator(t_data)
        >>> print(f"Tail index: {result.tail_index:.2f}")
        >>> print(f"Classification: {result.classification}")
        >>>
        >>> # Normal distribution has thin tails (large α)
        >>> normal_data = np.random.normal(0, 1, 1000)
        >>> result = hill_estimator(normal_data)
        >>> print(f"Tail index: {result.tail_index:.2f}")

    Notes:
        - Optimal k selection is an open research problem
        - Default k = sqrt(n) is a common heuristic
        - SE(α̂) = α̂ / sqrt(k)
        - Works best for truly power law tails
        - For "both" tails, returns minimum of upper and lower estimates

    References:
        - Hill, B. M. (1975). A simple general approach to inference about the
          tail of a distribution. The Annals of Statistics, 3(5), 1163-1174.
    """
    # Input validation
    if data is None:
        raise ValidationError("Data cannot be None", context={"function": "hill_estimator"})

    # Convert to numpy array
    if isinstance(data, pd.Series):
        arr = data.to_numpy()
    elif isinstance(data, np.ndarray):
        arr = data
    else:
        raise ValidationError(
            f"Data must be pandas Series or numpy array, got {type(data)}",
            context={"function": "hill_estimator", "data_type": type(data).__name__},
        )

    # Check array properties
    if arr.ndim != 1:
        raise ValidationError(
            f"Data must be 1-dimensional, got {arr.ndim}D",
            context={"function": "hill_estimator", "shape": arr.shape},
        )

    if len(arr) == 0:
        raise ValidationError(
            "Data cannot be empty", context={"function": "hill_estimator", "length": 0}
        )

    # Check for missing/infinite values
    if np.any(~np.isfinite(arr)):
        n_invalid = np.sum(~np.isfinite(arr))
        raise ValidationError(
            f"Data contains {n_invalid} NaN or infinite values",
            context={"function": "hill_estimator", "n_invalid": n_invalid, "length": len(arr)},
        )

    # Check minimum length
    min_length = 50  # Need sufficient data for tail estimation
    if len(arr) < min_length:
        raise ValidationError(
            f"Insufficient data for Hill estimator (need at least {min_length} observations)",
            context={"function": "hill_estimator", "length": len(arr), "min_length": min_length},
        )

    # Validate tail parameter
    if tail not in ["upper", "lower", "both"]:
        raise ValidationError(
            f"Invalid tail parameter: {tail}. Must be 'upper', 'lower', or 'both'",
            context={"function": "hill_estimator", "tail": tail},
        )

    # Set k if not provided (common heuristic: sqrt(n))
    n = len(arr)
    if k is None:
        k = int(np.sqrt(n))
    elif k < 2:
        raise ValidationError(
            f"k must be at least 2, got {k}",
            context={"function": "hill_estimator", "k": k},
        )
    elif k >= n:
        raise ValidationError(
            f"k must be less than n={n}, got {k}",
            context={"function": "hill_estimator", "k": k, "n": n},
        )

    logger.info("Computing Hill estimator", n_obs=n, k=k, tail=tail)

    try:

        def compute_hill_alpha(data_sorted: np.ndarray, k: int) -> tuple[float, float]:
            """Compute Hill estimator for sorted data (descending order)."""
            # Get k largest values and the (k+1)th value
            X_k_plus_1 = data_sorted[k]  # (k+1)th largest value

            # Check for zero or negative values (can't take log)
            if X_k_plus_1 <= 0:
                raise ComputationError(
                    "Hill estimator requires positive data for log transform",
                    context={"function": "hill_estimator", "X_k_plus_1": float(X_k_plus_1)},
                )

            # Compute Hill estimator: α̂ = k / Σ(log(X_i) - log(X_{k+1}))
            log_ratios = np.log(data_sorted[:k]) - np.log(X_k_plus_1)
            alpha = float(k / np.sum(log_ratios))

            # Standard error: SE(α̂) = α̂ / sqrt(k)
            alpha_se = float(alpha / np.sqrt(k))

            return alpha, alpha_se

        # Compute for requested tail(s)
        if tail == "upper":
            # Sort descending for upper tail
            sorted_data = np.sort(arr)[::-1]
            alpha, alpha_se = compute_hill_alpha(sorted_data, k)

        elif tail == "lower":
            # For lower tail, analyze absolute values of negative tail
            # Take absolute values to ensure positive data for log transform
            sorted_data = np.sort(np.abs(arr))[::-1]
            alpha, alpha_se = compute_hill_alpha(sorted_data, k)

        else:  # both
            # Compute both tails and take minimum (more conservative)
            sorted_upper = np.sort(arr)[::-1]
            alpha_upper, alpha_se_upper = compute_hill_alpha(sorted_upper, k)

            # For lower tail, use absolute values
            sorted_lower = np.sort(np.abs(arr))[::-1]
            alpha_lower, alpha_se_lower = compute_hill_alpha(sorted_lower, k)

            # Use minimum (heavier tail)
            if alpha_upper < alpha_lower:
                alpha, alpha_se = alpha_upper, alpha_se_upper
            else:
                alpha, alpha_se = alpha_lower, alpha_se_lower

        # Classify tail
        if alpha <= 2.0:
            classification = "heavy"
        elif alpha <= 4.0:
            classification = "medium"
        else:
            classification = "thin"

        logger.info(
            "Hill estimator computed",
            alpha=alpha,
            classification=classification,
            k=k,
        )

        return HillEstimatorResult(
            tail_index=alpha,
            tail_index_se=alpha_se,
            k=k,
            classification=classification,
            tail=tail,
            n_obs=n,
        )

    except ComputationError:
        raise
    except Exception as e:
        logger.error("Hill estimator failed", error=str(e), n_obs=n, k=k)
        raise ComputationError(  # noqa: B904
            f"Hill estimator computation failed: {e}",
            context={"function": "hill_estimator", "n_obs": n, "k": k, "tail": tail},
            cause=e,
        )


def generate_qq_data(
    data: pd.Series | np.ndarray,
    distribution: str = "normal",
    df: int | None = None,
) -> QQPlotData:
    """Generate Q-Q plot data for distribution comparison.

    Computes empirical quantiles and theoretical quantiles from a reference
    distribution. Q-Q plots visualize how well data follows a theoretical
    distribution - points on the diagonal indicate good fit.

    Args:
        data: Time series data (1D array or Series)
        distribution: Reference distribution ("normal", "t", "uniform", "exponential")
        df: Degrees of freedom for Student's t (required if distribution="t")

    Returns:
        QQPlotData with quantiles and R² goodness of fit

    Raises:
        ValidationError: If data or parameters are invalid
        ComputationError: If computation fails

    Example:
        >>> import numpy as np
        >>> # Normal data should fit normal QQ plot well
        >>> normal_data = np.random.normal(0, 1, 1000)
        >>> qq = generate_qq_data(normal_data, distribution="normal")
        >>> print(f"R²: {qq.r_squared:.4f}")  # Should be close to 1
        >>>
        >>> # Heavy-tailed data fits t-distribution better
        >>> t_data = np.random.standard_t(df=3, size=1000)
        >>> qq_normal = generate_qq_data(t_data, distribution="normal")
        >>> qq_t = generate_qq_data(t_data, distribution="t", df=3)
        >>> print(f"Normal R²: {qq_normal.r_squared:.4f}")
        >>> print(f"t R²: {qq_t.r_squared:.4f}")  # Better fit

    Notes:
        - Uses scipy.stats.probplot for QQ data generation
        - R² measures goodness of fit (1 = perfect fit)
        - Deviations in tails indicate different tail behavior
        - Works for any sample size, but more reliable for n > 100
    """
    # Input validation
    if data is None:
        raise ValidationError("Data cannot be None", context={"function": "generate_qq_data"})

    # Convert to numpy array
    if isinstance(data, pd.Series):
        arr = data.to_numpy()
    elif isinstance(data, np.ndarray):
        arr = data
    else:
        raise ValidationError(
            f"Data must be pandas Series or numpy array, got {type(data)}",
            context={"function": "generate_qq_data", "data_type": type(data).__name__},
        )

    # Check array properties
    if arr.ndim != 1:
        raise ValidationError(
            f"Data must be 1-dimensional, got {arr.ndim}D",
            context={"function": "generate_qq_data", "shape": arr.shape},
        )

    if len(arr) == 0:
        raise ValidationError(
            "Data cannot be empty", context={"function": "generate_qq_data", "length": 0}
        )

    # Check for missing/infinite values
    if np.any(~np.isfinite(arr)):
        n_invalid = np.sum(~np.isfinite(arr))
        raise ValidationError(
            f"Data contains {n_invalid} NaN or infinite values",
            context={"function": "generate_qq_data", "n_invalid": n_invalid, "length": len(arr)},
        )

    # Validate distribution parameter
    valid_distributions = ["normal", "t", "uniform", "exponential"]
    if distribution not in valid_distributions:
        raise ValidationError(
            f"Invalid distribution: {distribution}. Must be one of {valid_distributions}",
            context={"function": "generate_qq_data", "distribution": distribution},
        )

    # Validate df for t-distribution
    if distribution == "t":
        if df is None:
            raise ValidationError(
                "Degrees of freedom (df) required for t-distribution",
                context={"function": "generate_qq_data", "distribution": distribution},
            )
        if df < 1:
            raise ValidationError(
                f"Degrees of freedom must be >= 1, got {df}",
                context={"function": "generate_qq_data", "df": df},
            )

    logger.info("Generating QQ plot data", n_obs=len(arr), distribution=distribution)

    try:
        # Generate QQ plot data using scipy
        if distribution == "normal":
            # Default: compare to standard normal
            (theoretical_q, sample_q), (slope, intercept, r) = stats.probplot(arr, dist="norm")
        elif distribution == "t":
            # Student's t distribution with specified df
            (theoretical_q, sample_q), (slope, intercept, r) = stats.probplot(
                arr, dist="t", sparams=(df,)
            )
        elif distribution == "uniform":
            (theoretical_q, sample_q), (slope, intercept, r) = stats.probplot(arr, dist="uniform")
        elif distribution == "exponential":
            (theoretical_q, sample_q), (slope, intercept, r) = stats.probplot(arr, dist="expon")
        else:
            raise ValidationError(
                f"Distribution '{distribution}' not implemented",
                context={"function": "generate_qq_data", "distribution": distribution},
            )

        # Compute R² from correlation coefficient
        r_squared = float(r**2)

        logger.info(
            "QQ plot data generated",
            distribution=distribution,
            r_squared=r_squared,
        )

        return QQPlotData(
            theoretical_quantiles=theoretical_q,
            sample_quantiles=sample_q,
            distribution=distribution,
            r_squared=r_squared,
            df=df,
            n_obs=len(arr),
        )

    except Exception as e:
        logger.error("QQ plot generation failed", error=str(e), distribution=distribution)
        raise ComputationError(  # noqa: B904
            f"QQ plot generation failed: {e}",
            context={
                "function": "generate_qq_data",
                "distribution": distribution,
                "n_obs": len(arr),
            },
            cause=e,
        )


def analyze_tails(
    data: pd.Series | np.ndarray,
    k: int | None = None,
) -> TailAnalysisResult:
    """Comprehensive tail analysis combining Hill estimator and QQ plots.

    Performs multi-method tail analysis:
    1. Hill estimator for tail index
    2. QQ plot comparison with normal distribution
    3. QQ plot comparison with Student's t (if heavy-tailed)
    4. Best-fit distribution determination

    This provides robust characterization of tail behavior and helps identify
    appropriate distributional assumptions for risk modeling.

    Args:
        data: Time series data (1D array or Series)
        k: Number of order statistics for Hill estimator (default: sqrt(n))

    Returns:
        TailAnalysisResult with comprehensive tail diagnostics

    Raises:
        ValidationError: If data is invalid
        ComputationError: If analysis fails

    Example:
        >>> import numpy as np
        >>> # Analyze heavy-tailed data
        >>> t_data = np.random.standard_t(df=3, size=1000)
        >>> result = analyze_tails(t_data)
        >>> print(result.summary())
        >>>
        >>> # Check best fit
        >>> print(f"Best fit: {result.best_fit}")
        >>> print(f"Tail classification: {result.hill_result.classification}")
        >>>
        >>> # Analyze normal data for comparison
        >>> normal_data = np.random.normal(0, 1, 1000)
        >>> result = analyze_tails(normal_data)
        >>> print(f"Best fit: {result.best_fit}")

    Notes:
        - Combines multiple methods for robust analysis
        - Best fit selected based on Hill estimator and R² values
        - Heavy tails (α ≤ 2) automatically compared to t-distribution
        - Provides actionable recommendations for risk modeling
    """
    # Input validation (basic check, detailed checks in subfunctions)
    if data is None:
        raise ValidationError("Data cannot be None", context={"function": "analyze_tails"})

    logger.info("Starting comprehensive tail analysis")

    try:
        # 1. Hill estimator for tail index
        hill_result = hill_estimator(data, k=k, tail="both")

        # 2. QQ plot with normal distribution
        qq_normal = generate_qq_data(data, distribution="normal")

        # 3. QQ plot with Student's t (if heavy or medium tails)
        qq_t = None
        if hill_result.classification in ["heavy", "medium"]:
            # Estimate df based on tail index
            # For Student's t: tail index α ≈ df
            # Use tail index as starting point, clamp to reasonable range
            estimated_df = max(2, min(30, int(round(hill_result.tail_index))))

            qq_t = generate_qq_data(data, distribution="t", df=estimated_df)

        # 4. Determine best fit
        if hill_result.classification == "thin" and qq_normal.r_squared >= 0.95:
            best_fit = "normal"
        elif qq_t is not None and qq_t.r_squared > qq_normal.r_squared + 0.02:
            # t-distribution fits significantly better
            best_fit = "t"
        elif hill_result.classification == "heavy":
            best_fit = "heavy-tailed"
        elif qq_normal.r_squared >= 0.90:
            best_fit = "normal"
        else:
            # Neither fits well, classify based on Hill estimator
            best_fit = "heavy-tailed" if hill_result.classification != "thin" else "normal"

        logger.info(
            "Tail analysis completed",
            tail_index=hill_result.tail_index,
            classification=hill_result.classification,
            best_fit=best_fit,
        )

        return TailAnalysisResult(
            hill_result=hill_result,
            qq_normal=qq_normal,
            qq_t=qq_t,
            best_fit=best_fit,
        )

    except (ValidationError, ComputationError):
        raise
    except Exception as e:
        logger.error("Tail analysis failed", error=str(e))
        raise ComputationError(  # noqa: B904
            f"Tail analysis failed: {e}",
            context={"function": "analyze_tails"},
            cause=e,
        )
