"""Comprehensive volatility analysis combining ARCH-LM and GARCH.

This module provides a unified interface for volatility analysis, combining
the ARCH-LM test for detecting volatility clustering with GARCH model
fitting for estimating conditional volatility dynamics.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ml4t.diagnostic.errors import ComputationError, ValidationError
from ml4t.diagnostic.logging import get_logger

from .arch import ARCHLMResult, arch_lm_test
from .garch import GARCHResult, fit_garch

logger = get_logger(__name__)


class VolatilityAnalysisResult:
    """Comprehensive volatility analysis results combining ARCH-LM and GARCH.

    This class provides a unified interface for volatility analysis, combining
    the ARCH-LM test for detecting volatility clustering with GARCH model
    fitting for estimating conditional volatility dynamics.

    Workflow:
        1. Run ARCH-LM test to detect volatility clustering
        2. If clustering detected AND fit_garch=True, fit GARCH model
        3. Provide comprehensive summary and recommendations

    Attributes:
        arch_lm_result: Results from ARCH-LM test
        garch_result: Results from GARCH fitting (None if not fitted or no ARCH effects)
        has_volatility_clustering: Whether volatility clustering was detected
        persistence: Overall volatility persistence (alpha + beta from GARCH, None if not fitted)
        interpretation: Human-readable interpretation of results
    """

    def __init__(
        self,
        arch_lm_result: ARCHLMResult,
        garch_result: GARCHResult | None = None,
    ):
        """Initialize volatility analysis result.

        Args:
            arch_lm_result: Results from ARCH-LM test
            garch_result: Results from GARCH fitting (optional)
        """
        self.arch_lm_result = arch_lm_result
        self.garch_result = garch_result
        self.has_volatility_clustering = arch_lm_result.has_arch_effects

        # Extract persistence if GARCH was fitted
        self.persistence: float | None
        if garch_result is not None:
            self.persistence = garch_result.persistence
        else:
            self.persistence = None

        # Generate interpretation
        self.interpretation = self._generate_interpretation()

    def _generate_interpretation(self) -> str:
        """Generate human-readable interpretation of results."""
        lines = []

        # ARCH-LM test interpretation
        if self.has_volatility_clustering:
            lines.append("✓ Volatility clustering detected (ARCH effects present)")
            lines.append("  - Time-varying volatility in returns")
            lines.append("  - Large changes tend to follow large changes")
        else:
            lines.append("✗ No volatility clustering detected (no ARCH effects)")
            lines.append("  - Constant variance assumption reasonable")
            lines.append("  - Classical methods with homoscedasticity appropriate")

        # GARCH model interpretation (if fitted)
        if self.garch_result is not None:
            lines.append("")
            lines.append("GARCH Model Results:")

            persistence = self.persistence
            if persistence is not None:
                lines.append(f"  - Persistence (α+β): {persistence:.4f}")

                if persistence >= 1.0:
                    lines.append("  ⚠ WARNING: Non-stationary (persistence ≥ 1)")
                    lines.append("  - Volatility shocks do not decay")
                    lines.append("  - Consider IGARCH or alternative models")
                elif persistence >= 0.99:
                    lines.append("  ⚠ Very high persistence (near unit root)")
                    lines.append("  - Volatility shocks decay very slowly")
                    lines.append("  - Risk forecasts remain elevated for long periods")
                elif persistence > 0.95:
                    lines.append("  → High persistence (slow mean reversion)")
                    lines.append("  - Typical for daily financial returns")
                    lines.append("  - Volatility shocks persist for many periods")
                else:
                    lines.append("  → Moderate persistence (faster mean reversion)")
                    lines.append("  - Volatility shocks decay relatively quickly")

                # Compute half-life if stationary and positive
                # Guard against persistence <= 0 which would make log undefined
                if 0.0 < persistence < 1.0:
                    half_life = np.log(0.5) / np.log(persistence)
                    lines.append(f"  - Shock half-life: {half_life:.1f} periods")

        # Recommendations
        lines.append("")
        lines.append("Recommendations:")
        if self.has_volatility_clustering:
            if self.garch_result is not None:
                lines.append("  1. Use fitted GARCH model for volatility forecasting")
                lines.append("  2. Apply conditional volatility in risk models (VaR, CVaR)")
                lines.append("  3. Consider HAC-adjusted standard errors for inference")
                lines.append("  4. Account for volatility clustering in trading strategies")
            else:
                lines.append("  1. Consider fitting GARCH/EGARCH models")
                lines.append("  2. Use HAC-adjusted standard errors")
                lines.append("  3. Account for time-varying volatility in risk models")
        else:
            lines.append("  1. Constant variance models appropriate")
            lines.append("  2. Standard OLS methods valid")
            lines.append("  3. Classical risk models acceptable")

        return "\n".join(lines)

    def __repr__(self) -> str:
        """String representation."""
        garch_info = (
            f", persistence={self.persistence:.4f}"
            if self.persistence is not None
            else ", no_garch"
        )
        return (
            f"VolatilityAnalysisResult("
            f"has_clustering={self.has_volatility_clustering}, "
            f"arch_p={self.arch_lm_result.p_value:.4f}"
            f"{garch_info})"
        )

    def summary(self) -> str:
        """Comprehensive volatility analysis summary.

        Returns:
            Formatted summary string with all analysis results
        """
        lines = [
            "=" * 70,
            "Comprehensive Volatility Analysis",
            "=" * 70,
        ]

        # Section 1: ARCH-LM Test
        lines.append("")
        lines.append("1. ARCH-LM Test for Volatility Clustering")
        lines.append("-" * 70)
        lines.append(f"Test Statistic: {self.arch_lm_result.test_statistic:.4f}")
        lines.append(f"P-value:        {self.arch_lm_result.p_value:.4f}")
        lines.append(f"Lags Used:      {self.arch_lm_result.lags}")
        lines.append(f"Observations:   {self.arch_lm_result.n_obs}")
        lines.append("")
        conclusion = (
            "ARCH effects detected (volatility clustering present)"
            if self.has_volatility_clustering
            else "No ARCH effects (constant variance)"
        )
        lines.append(f"Conclusion: {conclusion}")

        # Section 2: GARCH Model (if fitted)
        if self.garch_result is not None:
            lines.append("")
            lines.append("2. GARCH Model Fitting Results")
            lines.append("-" * 70)
            # Infer p and q from coefficient shapes
            p = (
                len(self.garch_result.alpha)
                if isinstance(self.garch_result.alpha, tuple | list)
                else 1
            )
            q = (
                len(self.garch_result.beta)
                if isinstance(self.garch_result.beta, tuple | list)
                else 1
            )
            lines.append(f"Model:          GARCH({p},{q})")
            lines.append(f"Converged:      {'Yes' if self.garch_result.converged else 'No'}")
            lines.append(f"Iterations:     {self.garch_result.iterations}")
            lines.append("")
            lines.append("Parameters:")
            lines.append(f"  ω (omega):    {self.garch_result.omega:.6f}")

            if isinstance(self.garch_result.alpha, tuple | list):
                for i, a in enumerate(self.garch_result.alpha, 1):
                    lines.append(f"  α{i} (alpha):  {a:.6f}")
            else:
                lines.append(f"  α (alpha):    {self.garch_result.alpha:.6f}")

            if isinstance(self.garch_result.beta, tuple | list):
                for i, b in enumerate(self.garch_result.beta, 1):
                    lines.append(f"  β{i} (beta):   {b:.6f}")
            else:
                lines.append(f"  β (beta):     {self.garch_result.beta:.6f}")

            lines.append("")
            lines.append(f"Persistence (α+β): {self.persistence:.6f}")

            # Model fit statistics
            lines.append("")
            lines.append("Model Fit:")
            lines.append(f"  Log-Likelihood: {self.garch_result.log_likelihood:.4f}")
            lines.append(f"  AIC:            {self.garch_result.aic:.4f}")
            lines.append(f"  BIC:            {self.garch_result.bic:.4f}")

        elif self.has_volatility_clustering:
            lines.append("")
            lines.append("2. GARCH Model")
            lines.append("-" * 70)
            lines.append("Not fitted (fit_garch=False or fitting skipped)")

        # Section 3: Interpretation
        lines.append("")
        lines.append("3. Interpretation")
        lines.append("-" * 70)
        lines.append(self.interpretation)

        lines.append("")
        lines.append("=" * 70)

        return "\n".join(lines)


def analyze_volatility(
    returns: pd.Series | np.ndarray,
    arch_lags: int = 12,
    fit_garch_model: bool = True,
    garch_p: int = 1,
    garch_q: int = 1,
    alpha: float = 0.05,
) -> VolatilityAnalysisResult:
    """Comprehensive volatility analysis combining ARCH-LM and GARCH.

    This function provides a complete workflow for volatility analysis:
    1. Tests for volatility clustering using ARCH-LM test
    2. If clustering detected AND fit_garch=True, fits GARCH model
    3. Returns comprehensive summary with interpretation and recommendations

    The ARCH-LM test detects autoregressive conditional heteroscedasticity
    (volatility clustering), and the GARCH model quantifies the dynamics
    of time-varying volatility.

    Args:
        returns: Returns series (NOT prices) to analyze
        arch_lags: Number of lags for ARCH-LM test (default 12)
        fit_garch_model: Whether to fit GARCH model if ARCH effects detected (default True)
        garch_p: GARCH AR order (default 1)
        garch_q: GARCH MA order (default 1)
        alpha: Significance level for ARCH-LM test (default 0.05)

    Returns:
        VolatilityAnalysisResult with comprehensive analysis

    Raises:
        ValidationError: If data is invalid
        ComputationError: If analysis fails

    Notes:
        - Always run ARCH-LM test first (even if fit_garch_model=False)
        - GARCH fitting only attempted if ARCH effects detected
        - Set fit_garch_model=False to skip GARCH (faster, detection only)
        - GARCH requires 'arch' package (pip install arch)
        - Default GARCH(1,1) sufficient for most financial applications
        - Results include interpretation and actionable recommendations

    References:
        Engle, R. F. (1982). Autoregressive Conditional Heteroscedasticity.
        Bollerslev, T. (1986). Generalized Autoregressive Conditional Heteroskedasticity.
    """
    logger.debug(
        f"Running comprehensive volatility analysis: "
        f"arch_lags={arch_lags}, fit_garch_model={fit_garch_model}, "
        f"garch_p={garch_p}, garch_q={garch_q}"
    )

    # Step 1: Run ARCH-LM test
    try:
        arch_result = arch_lm_test(returns, lags=arch_lags, demean=True, alpha=alpha)
        logger.info(
            f"ARCH-LM test complete: has_arch={arch_result.has_arch_effects}, p_value={arch_result.p_value:.4f}"
        )
    except ValidationError:
        # Let validation errors pass through (invalid inputs)
        raise
    except Exception as e:
        # Wrap other errors as computation errors
        logger.error(f"ARCH-LM test failed: {e}")
        raise ComputationError(  # noqa: B904
            f"ARCH-LM test failed during volatility analysis: {e}",
            context={"arch_lags": arch_lags},
            cause=e,
        )

    # Step 2: Fit GARCH if ARCH effects detected and requested
    garch_result = None
    if arch_result.has_arch_effects and fit_garch_model:
        logger.debug(
            f"ARCH effects detected (p={arch_result.p_value:.4f}), fitting GARCH({garch_p},{garch_q}) model"
        )
        try:
            garch_result = fit_garch(returns, p=garch_p, q=garch_q)
            logger.info(
                f"GARCH({garch_p},{garch_q}) fitted successfully: "
                f"persistence={garch_result.persistence:.4f}, "
                f"converged={garch_result.converged}"
            )
        except ValidationError as e:
            # If arch package not installed, log warning but continue
            if "arch" in str(e).lower() and "package" in str(e).lower():
                logger.warning(
                    "GARCH fitting skipped: arch package not installed. Install with: pip install arch"
                )
            else:
                # Re-raise other validation errors
                raise
        except Exception as e:
            # Log error but continue with ARCH-LM results only
            logger.warning(f"GARCH fitting failed: {e}. Continuing with ARCH-LM results only.")
    elif not arch_result.has_arch_effects:
        logger.info(
            f"No ARCH effects detected (p={arch_result.p_value:.4f}), skipping GARCH fitting"
        )
    else:
        logger.debug("fit_garch_model=False, skipping GARCH fitting")

    # Step 3: Create comprehensive result
    result = VolatilityAnalysisResult(
        arch_lm_result=arch_result,
        garch_result=garch_result,
    )

    logger.info(
        f"Volatility analysis complete: "
        f"has_clustering={result.has_volatility_clustering}, "
        f"persistence={result.persistence}"
    )

    return result
