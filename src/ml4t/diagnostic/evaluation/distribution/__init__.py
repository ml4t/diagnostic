"""Distribution diagnostics for financial returns analysis.

This module provides statistical tests and metrics for analyzing the distribution
properties of financial returns:

- Moments (skewness and excess kurtosis) with significance tests
- Jarque-Bera test for normality (based on moments)
- Shapiro-Wilk test for normality (more powerful for small samples)
- Heavy tail detection using Hill estimator and QQ plots
- Tail classification (thin, medium, heavy) for power law analysis

Distribution analysis is critical for understanding return characteristics and
validating modeling assumptions. Many financial models assume normally distributed
returns, but real financial data often exhibits:
- Skewness (asymmetry): Negative skew common in equity returns
- Excess kurtosis (fat tails): More extreme events than normal distribution
- Non-normality: Violations of Gaussian assumptions
- Heavy tails: Power law behavior in extreme events

Example:
    >>> import numpy as np
    >>> from ml4t.diagnostic.evaluation.distribution import (
    ...     compute_moments, jarque_bera_test, shapiro_wilk_test,
    ...     hill_estimator, analyze_tails, analyze_distribution
    ... )
    >>>
    >>> # Quick comprehensive analysis (recommended)
    >>> returns = np.random.standard_t(df=5, size=1000) * 0.01
    >>> result = analyze_distribution(returns)
    >>> print(result.summary())
    >>> print(f"Recommended: {result.recommended_distribution}")
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ml4t.diagnostic.errors import ComputationError, ValidationError

# Import from submodules
from ml4t.diagnostic.evaluation.distribution.moments import (
    MomentsResult,
    compute_moments,
)
from ml4t.diagnostic.evaluation.distribution.tails import (
    HillEstimatorResult,
    QQPlotData,
    TailAnalysisResult,
    analyze_tails,
    generate_qq_data,
    hill_estimator,
)
from ml4t.diagnostic.evaluation.distribution.tests import (
    JarqueBeraResult,
    ShapiroWilkResult,
    jarque_bera_test,
    shapiro_wilk_test,
)
from ml4t.diagnostic.logging import get_logger

logger = get_logger(__name__)

# Public API
__all__ = [
    # Result classes
    "MomentsResult",
    "JarqueBeraResult",
    "ShapiroWilkResult",
    "HillEstimatorResult",
    "QQPlotData",
    "TailAnalysisResult",
    "DistributionAnalysisResult",
    # Functions
    "compute_moments",
    "jarque_bera_test",
    "shapiro_wilk_test",
    "hill_estimator",
    "generate_qq_data",
    "analyze_tails",
    "analyze_distribution",
]


@dataclass
class DistributionAnalysisResult:
    """Comprehensive distribution analysis results.

    Combines moments, normality tests, and tail analysis to provide complete
    characterization of distribution properties. This unified analysis helps
    determine appropriate statistical methods and risk models.

    Attributes:
        moments_result: Distribution moments (skewness, kurtosis) with significance
        jarque_bera_result: Jarque-Bera normality test result
        shapiro_wilk_result: Shapiro-Wilk normality test result (more powerful for small n)
        tail_analysis_result: Comprehensive tail analysis (Hill, QQ plots)
        is_normal: Consensus normality assessment from all tests
        recommended_distribution: Best-fit distribution ("normal", "t", "stable", "heavy-tailed")
        recommended_df: Degrees of freedom for t-distribution (None otherwise)
        interpretation: Human-readable summary of key findings
    """

    moments_result: MomentsResult
    jarque_bera_result: JarqueBeraResult
    shapiro_wilk_result: ShapiroWilkResult
    tail_analysis_result: TailAnalysisResult | None
    is_normal: bool
    recommended_distribution: str
    recommended_df: int | None
    interpretation: str

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"DistributionAnalysisResult(is_normal={self.is_normal}, "
            f"recommended='{self.recommended_distribution}', "
            f"n={self.moments_result.n_obs})"
        )

    def summary(self) -> str:
        """Comprehensive distribution analysis summary.

        Returns:
            Formatted summary string combining all analyses
        """
        lines = [
            "=" * 70,
            "COMPREHENSIVE DISTRIBUTION ANALYSIS",
            "=" * 70,
            f"Sample Size:             {self.moments_result.n_obs}",
            f"Mean:                    {self.moments_result.mean:.6f}",
            f"Std Dev:                 {self.moments_result.std:.6f}",
            "",
        ]

        # Moments summary
        lines.append("MOMENTS:")
        lines.append(f"  Skewness:              {self.moments_result.skewness:.4f}")
        if self.moments_result.skewness_significant:
            skew_interp = "right-skewed" if self.moments_result.skewness > 0 else "left-skewed"
            lines.append(f"                         (Significantly {skew_interp})")
        else:
            lines.append("                         (Not significantly different from 0)")

        lines.append(f"  Excess Kurtosis:       {self.moments_result.excess_kurtosis:.4f}")
        if self.moments_result.excess_kurtosis_significant:
            kurt_interp = "fat tails" if self.moments_result.excess_kurtosis > 0 else "thin tails"
            lines.append(f"                         (Significantly {kurt_interp})")
        else:
            lines.append("                         (Not significantly different from 0)")

        # Normality tests summary
        lines.append("")
        lines.append("NORMALITY TESTS:")
        lines.append(
            f"  Jarque-Bera:           p={self.jarque_bera_result.p_value:.4f} "
            f"({'PASS' if self.jarque_bera_result.is_normal else 'FAIL'})"
        )
        lines.append(
            f"  Shapiro-Wilk:          p={self.shapiro_wilk_result.p_value:.4f} "
            f"({'PASS' if self.shapiro_wilk_result.is_normal else 'FAIL'})"
        )
        lines.append(f"  Consensus:             {'NORMAL' if self.is_normal else 'NON-NORMAL'}")

        # Tail analysis summary (if computed)
        if self.tail_analysis_result is not None:
            lines.append("")
            lines.append("TAIL ANALYSIS:")
            lines.append(
                f"  Hill Tail Index:       {self.tail_analysis_result.hill_result.tail_index:.4f}"
            )
            lines.append(
                f"  Tail Classification:   {self.tail_analysis_result.hill_result.classification.upper()}"
            )
            lines.append(
                f"  Normal R²:             {self.tail_analysis_result.qq_normal.r_squared:.4f}"
            )
            if self.tail_analysis_result.qq_t is not None:
                lines.append(
                    f"  Student's t R²:        {self.tail_analysis_result.qq_t.r_squared:.4f} "
                    f"(df={self.tail_analysis_result.qq_t.df})"
                )
            lines.append(f"  Best Fit:              {self.tail_analysis_result.best_fit.upper()}")

        # Recommendation
        lines.append("")
        lines.append("=" * 70)
        lines.append("RECOMMENDATION:")
        lines.append(f"  Distribution:          {self.recommended_distribution.upper()}")
        if self.recommended_df is not None:
            lines.append(f"  Degrees of Freedom:    {self.recommended_df}")

        # Interpretation
        lines.append("")
        lines.append("INTERPRETATION:")
        for line in self.interpretation.split("\n"):
            lines.append(f"  {line}")

        # Risk implications
        lines.append("")
        lines.append("RISK IMPLICATIONS:")
        if self.recommended_distribution == "normal":
            lines.append("  - Standard normal-based risk measures appropriate (VaR, Sharpe)")
            lines.append("  - Classical portfolio optimization methods valid")
            lines.append("  - Parametric statistical inference reliable")
        elif self.recommended_distribution == "t":
            lines.append(
                f"  - Use Student's t distribution (df={self.recommended_df}) for modeling"
            )
            lines.append("  - Heavier tails than normal => higher extreme event probability")
            lines.append("  - Consider robust Sharpe ratio alternatives (e.g., Sortino)")
            lines.append("  - VaR should account for fat tails")
        elif self.recommended_distribution in ["stable", "heavy-tailed"]:
            lines.append("  - WARNING: Heavy tails detected => use extreme value theory")
            lines.append("  - Standard risk measures (VaR, Sharpe) may be unreliable")
            lines.append("  - Use CVaR (Expected Shortfall) instead of VaR")
            lines.append("  - Consider tail risk hedging strategies")
            lines.append("  - Apply robust portfolio optimization methods")

        lines.append("=" * 70)

        return "\n".join(lines)


def analyze_distribution(
    data: pd.Series | np.ndarray,
    alpha: float = 0.05,
    compute_tails: bool = True,
) -> DistributionAnalysisResult:
    """Comprehensive distribution analysis combining all methods.

    Performs complete statistical characterization of distribution properties:
    1. Computes moments (skewness, kurtosis) with significance tests
    2. Runs normality tests (Jarque-Bera, Shapiro-Wilk)
    3. Analyzes tail behavior (Hill estimator, QQ plots) if compute_tails=True
    4. Determines consensus and recommends appropriate distribution

    This unified analysis provides actionable guidance for selecting statistical
    methods and risk models appropriate for the data characteristics.

    Args:
        data: Time series data (1D array or Series), typically financial returns
        alpha: Significance level for statistical tests (default 0.05)
        compute_tails: Whether to run tail analysis (default True, can be slow for large n)

    Returns:
        DistributionAnalysisResult with comprehensive analysis and recommendations

    Raises:
        ValidationError: If data is invalid (empty, wrong shape, etc.)
        ComputationError: If analysis fails

    Example:
        >>> import numpy as np
        >>> from ml4t.diagnostic.evaluation.distribution import analyze_distribution
        >>>
        >>> # Analyze financial returns
        >>> returns = np.random.standard_t(df=5, size=1000) * 0.01  # Heavy-tailed returns
        >>> result = analyze_distribution(returns, alpha=0.05, compute_tails=True)
        >>>
        >>> # Print comprehensive summary
        >>> print(result.summary())
        >>>
        >>> # Get recommendation for risk modeling
        >>> print(f"Use {result.recommended_distribution} distribution")
        >>> if result.recommended_df:
        ...     print(f"Degrees of freedom: {result.recommended_df}")
        >>>
        >>> # Check if standard methods are appropriate
        >>> if result.is_normal:
        ...     print("Standard normal-based methods OK")
        ... else:
        ...     print("Use robust methods for non-normal data")
        >>>
        >>> # Quick analysis without tail computation (faster)
        >>> result_fast = analyze_distribution(returns, compute_tails=False)

    Notes:
        - Tail analysis (compute_tails=True) adds Hill estimator and QQ plots
        - Skip tail analysis for very large datasets or when speed is critical
        - Consensus normality requires both JB and SW to accept H0
        - Recommendation logic prioritizes tail analysis over simple normality tests
        - For n < 50, Shapiro-Wilk test may be unreliable (warning issued)
    """
    # Input validation (basic check, detailed checks in subfunctions)
    if data is None:
        raise ValidationError("Data cannot be None", context={"function": "analyze_distribution"})

    logger.info(
        "Starting comprehensive distribution analysis",
        compute_tails=compute_tails,
        alpha=alpha,
    )

    try:
        # 1. Compute moments
        moments_result = compute_moments(data, test_significance=True, alpha=alpha)

        # 2. Jarque-Bera test
        jarque_bera_result = jarque_bera_test(data, alpha=alpha)

        # 3. Shapiro-Wilk test
        shapiro_wilk_result = shapiro_wilk_test(data, alpha=alpha)

        # 4. Tail analysis (optional)
        tail_analysis_result = None
        if compute_tails:
            try:
                tail_analysis_result = analyze_tails(data)
            except Exception as e:
                logger.warning(f"Tail analysis failed, skipping: {e}")
                # Continue without tail analysis

        # 5. Determine consensus normality
        # Both tests must accept H0 for consensus normality
        is_normal = jarque_bera_result.is_normal and shapiro_wilk_result.is_normal

        # 6. Recommend distribution
        recommended_distribution, recommended_df = _recommend_distribution(
            is_normal=is_normal,
            moments_result=moments_result,
            tail_analysis_result=tail_analysis_result,
        )

        # 7. Generate interpretation
        interpretation = _generate_interpretation(
            is_normal=is_normal,
            moments_result=moments_result,
            jarque_bera_result=jarque_bera_result,
            shapiro_wilk_result=shapiro_wilk_result,
            tail_analysis_result=tail_analysis_result,
            recommended_distribution=recommended_distribution,
        )

        logger.info(
            "Distribution analysis completed",
            is_normal=is_normal,
            recommended=recommended_distribution,
            n_obs=moments_result.n_obs,
        )

        return DistributionAnalysisResult(
            moments_result=moments_result,
            jarque_bera_result=jarque_bera_result,
            shapiro_wilk_result=shapiro_wilk_result,
            tail_analysis_result=tail_analysis_result,
            is_normal=is_normal,
            recommended_distribution=recommended_distribution,
            recommended_df=recommended_df,
            interpretation=interpretation,
        )

    except (ValidationError, ComputationError):
        raise
    except Exception as e:
        logger.error("Distribution analysis failed", error=str(e))
        raise ComputationError(  # noqa: B904
            f"Distribution analysis failed: {e}",
            context={"function": "analyze_distribution"},
            cause=e,
        )


def _recommend_distribution(
    is_normal: bool,
    moments_result: MomentsResult,
    tail_analysis_result: TailAnalysisResult | None,
) -> tuple[str, int | None]:
    """Internal: Recommend distribution based on analysis results.

    Logic:
    1. If tail analysis available, prioritize its recommendation
    2. If both normality tests pass, recommend normal
    3. If heavy tails detected (alpha <= 2), recommend stable/heavy-tailed
    4. If medium tails (2 < alpha <= 4), recommend Student's t with estimated df
    5. Otherwise, recommend t-distribution for non-normal data

    Returns:
        Tuple of (distribution_name, degrees_of_freedom)
    """
    # If tail analysis available, use its recommendation
    if tail_analysis_result is not None:
        best_fit = tail_analysis_result.best_fit
        tail_index = tail_analysis_result.hill_result.tail_index
        classification = tail_analysis_result.hill_result.classification

        if best_fit == "normal":
            return ("normal", None)
        elif best_fit == "t":
            # Use df from QQ plot if available
            if tail_analysis_result.qq_t is not None:
                return ("t", tail_analysis_result.qq_t.df)
            else:
                # Estimate df from tail index: df ≈ 2*alpha for medium tails
                df = max(2, min(30, int(round(2 * tail_index))))
                return ("t", df)
        elif best_fit == "heavy-tailed":
            # Very heavy tails
            if classification == "heavy" and tail_index <= 2.0:
                return ("stable", None)  # Stable distribution for alpha <= 2
            else:
                return ("heavy-tailed", None)

    # Fallback: Use normality tests and moments
    if is_normal:
        return ("normal", None)

    # Non-normal: check excess kurtosis
    if moments_result.excess_kurtosis > 2.0:
        # Very fat tails => recommend heavy-tailed
        return ("heavy-tailed", None)
    elif moments_result.excess_kurtosis > 0.5:
        # Moderate fat tails => recommend t with estimated df
        # Heuristic: df ≈ 6/excess_kurtosis + 4 (for excess kurtosis)
        df = max(3, min(30, int(round(6 / moments_result.excess_kurtosis + 4))))
        return ("t", df)
    else:
        # Slight deviation from normal => t with higher df
        return ("t", 10)


def _generate_interpretation(
    is_normal: bool,
    moments_result: MomentsResult,
    jarque_bera_result: JarqueBeraResult,
    shapiro_wilk_result: ShapiroWilkResult,
    tail_analysis_result: TailAnalysisResult | None,
    recommended_distribution: str,
) -> str:
    """Internal: Generate human-readable interpretation.

    Returns:
        Multi-line interpretation string
    """
    lines = []

    # Normality assessment
    if is_normal:
        lines.append("Data is consistent with normal distribution (both tests pass).")
        lines.append("Standard statistical methods and risk measures are appropriate.")
    else:
        lines.append("Data deviates from normality (at least one test rejects H0).")

        # Explain why
        if jarque_bera_result.is_normal and not shapiro_wilk_result.is_normal:
            lines.append("Shapiro-Wilk test rejects normality (more powerful for small samples).")
        elif not jarque_bera_result.is_normal and shapiro_wilk_result.is_normal:
            lines.append("Jarque-Bera test rejects normality (based on skewness/kurtosis).")
        else:
            lines.append("Both normality tests reject H0.")

    # Moments interpretation
    if moments_result.skewness_significant:
        if moments_result.skewness > 0:
            lines.append(
                f"Significant positive skewness ({moments_result.skewness:.3f}) indicates right tail is heavier."
            )
        else:
            lines.append(
                f"Significant negative skewness ({moments_result.skewness:.3f}) "
                "indicates left tail is heavier (common for equity returns)."
            )

    if moments_result.excess_kurtosis_significant and moments_result.excess_kurtosis > 0:
        lines.append(
            f"Significant excess kurtosis ({moments_result.excess_kurtosis:.3f}) "
            "indicates fat tails and higher extreme event probability."
        )

    # Tail analysis interpretation
    if tail_analysis_result is not None:
        classification = tail_analysis_result.hill_result.classification
        tail_index = tail_analysis_result.hill_result.tail_index

        if classification == "heavy":
            lines.append(
                f"Heavy tails detected (α={tail_index:.2f} ≤ 2): power law behavior in extremes."
            )
        elif classification == "medium":
            lines.append(
                f"Medium-heavy tails (α={tail_index:.2f}): heavier than normal but finite variance."
            )
        else:
            lines.append(
                f"Thin tails detected (α={tail_index:.2f} > 4): approaching normal tail behavior."
            )

    # Recommendation rationale
    if recommended_distribution == "normal":
        lines.append("Normal distribution provides adequate fit for this data.")
    elif recommended_distribution == "t":
        lines.append("Student's t distribution recommended for heavier tails than normal.")
    elif recommended_distribution in ["stable", "heavy-tailed"]:
        lines.append("Heavy-tailed distribution required due to extreme power law behavior.")

    return "\n".join(lines)
