"""Feature-level diagnostic analysis for quantitative trading signals.

This module provides the main API for comprehensive feature diagnostic testing,
orchestrating all Module A diagnostic capabilities:

- Stationarity analysis (ADF, KPSS, PP tests)
- Autocorrelation analysis (ACF, PACF)
- Volatility clustering (ARCH-LM, GARCH)
- Distribution diagnostics (moments, normality, heavy tails)

The FeatureDiagnostics class provides a unified interface for running all
diagnostic tests on trading features/signals, with configurable test selection
and batch processing capabilities.

Key Concept:
    Before using features in ML models or calculating feature-outcome relationships,
    you must understand their statistical properties. FeatureDiagnostics provides
    a comprehensive health check of feature quality.

Typical Workflow:
    1. Create FeatureDiagnosticsConfig specifying which tests to run
    2. Initialize FeatureDiagnostics with config
    3. Call run_diagnostics() on feature time series
    4. Review FeatureDiagnosticsResult for insights
    5. Transform features based on diagnostic results
    6. Re-run diagnostics on transformed features

Example:
    >>> import numpy as np
    >>> from ml4t.diagnostic.evaluation.feature_diagnostics import (
    ...     FeatureDiagnostics,
    ...     FeatureDiagnosticsConfig
    ... )
    >>>
    >>> # Create feature (e.g., returns signal)
    >>> feature = np.random.randn(1000) * 0.02  # ~2% volatility white noise
    >>>
    >>> # Configure diagnostics
    >>> config = FeatureDiagnosticsConfig(
    ...     run_stationarity=True,
    ...     run_autocorrelation=True,
    ...     run_volatility=True,
    ...     run_distribution=True
    ... )
    >>>
    >>> # Run diagnostics
    >>> diagnostics = FeatureDiagnostics(config)
    >>> result = diagnostics.run_diagnostics(feature, name="momentum_signal")
    >>>
    >>> # Review results
    >>> print(result.summary())
    >>> print(result.summary_df)
    >>>
    >>> # Check specific properties
    >>> if result.stationarity.consensus == "strong_stationary":
    ...     print("Feature is stationary - safe to use directly")
    >>> if result.volatility.has_clustering:
    ...     print("Feature has volatility clustering - consider GARCH modeling")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

from ml4t.diagnostic.errors import ValidationError
from ml4t.diagnostic.logging import get_logger

# Import all diagnostic modules
from .autocorrelation import AutocorrelationAnalysisResult, analyze_autocorrelation
from .distribution import DistributionAnalysisResult, analyze_distribution
from .stationarity import StationarityAnalysisResult, analyze_stationarity
from .volatility import VolatilityAnalysisResult, analyze_volatility

logger = get_logger(__name__)


@dataclass
class FeatureDiagnosticsConfig:
    """Configuration for feature diagnostic analysis.

    Controls which diagnostic tests are run and their parameters.

    Attributes:
        run_stationarity: Whether to run stationarity tests (ADF, KPSS, PP)
        run_autocorrelation: Whether to run autocorrelation analysis (ACF, PACF)
        run_volatility: Whether to run volatility clustering tests (ARCH-LM, GARCH)
        run_distribution: Whether to run distribution diagnostics (moments, normality, tails)
        alpha: Significance level for statistical tests (default: 0.05)
        stationarity_tests: Which stationarity tests to run
        max_acf_lags: Maximum lags for ACF/PACF (None = auto-determine)
        arch_lm_lags: Lags for ARCH-LM test (None = auto-determine)
        fit_garch: Whether to fit GARCH model when ARCH effects detected
        normality_tests: Which normality tests to run
        tail_analysis: Whether to perform heavy tail analysis
        verbose: Whether to log detailed progress information
    """

    run_stationarity: bool = True
    run_autocorrelation: bool = True
    run_volatility: bool = True
    run_distribution: bool = True

    alpha: float = 0.05

    # Stationarity options
    stationarity_tests: list[Literal["adf", "kpss", "pp"]] | None = None

    # Autocorrelation options
    max_acf_lags: int | None = None

    # Volatility options
    arch_lags: int = 12
    fit_garch: bool = True

    # Distribution options
    compute_tails: bool = True

    # Logging
    verbose: bool = False

    def __post_init__(self):
        """Validate configuration."""
        if not (0 < self.alpha < 1):
            raise ValidationError(f"alpha must be in (0, 1), got {self.alpha}")

        if not any(
            [
                self.run_stationarity,
                self.run_autocorrelation,
                self.run_volatility,
                self.run_distribution,
            ]
        ):
            raise ValidationError("At least one diagnostic module must be enabled")


@dataclass
class FeatureDiagnosticsResult:
    """Results from comprehensive feature diagnostic analysis.

    Aggregates results from all diagnostic modules with high-level summary.

    Attributes:
        feature_name: Name/identifier for the feature
        n_obs: Number of observations in feature
        stationarity: Stationarity analysis result (None if not run)
        autocorrelation: Autocorrelation analysis result (None if not run)
        volatility: Volatility clustering analysis result (None if not run)
        distribution: Distribution diagnostics result (None if not run)
        summary_df: DataFrame summarizing all test results
        recommendations: List of recommendations based on diagnostic results
        health_score: Overall feature health score (0.0 to 1.0)
        flags: List of warning flags raised by diagnostics
    """

    feature_name: str
    n_obs: int

    stationarity: StationarityAnalysisResult | None = None
    autocorrelation: AutocorrelationAnalysisResult | None = None
    volatility: VolatilityAnalysisResult | None = None
    distribution: DistributionAnalysisResult | None = None

    summary_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    recommendations: list[str] = field(default_factory=list)
    health_score: float = 0.0
    flags: list[str] = field(default_factory=list)

    def __post_init__(self):
        """Calculate derived fields after initialization."""
        if self.summary_df.empty:
            self.summary_df = self._create_summary_df()

        if not self.recommendations:
            self.recommendations = self._generate_recommendations()

        if self.health_score == 0.0:
            self.health_score = self._calculate_health_score()

        if not self.flags:
            self.flags = self._identify_flags()

    def _create_summary_df(self) -> pd.DataFrame:
        """Create summary DataFrame from all diagnostic results.

        Returns:
            DataFrame with one row per test showing key statistics
        """
        rows = []

        # Stationarity tests
        if self.stationarity is not None:
            if self.stationarity.adf_result is not None:
                rows.append(
                    {
                        "Module": "Stationarity",
                        "Test": "ADF",
                        "Statistic": self.stationarity.adf_result.test_statistic,
                        "P-Value": self.stationarity.adf_result.p_value,
                        "Result": (
                            "Stationary"
                            if self.stationarity.adf_result.is_stationary
                            else "Non-stationary"
                        ),
                    }
                )

            if self.stationarity.kpss_result is not None:
                rows.append(
                    {
                        "Module": "Stationarity",
                        "Test": "KPSS",
                        "Statistic": self.stationarity.kpss_result.test_statistic,
                        "P-Value": self.stationarity.kpss_result.p_value,
                        "Result": (
                            "Stationary"
                            if self.stationarity.kpss_result.is_stationary
                            else "Non-stationary"
                        ),
                    }
                )

            if self.stationarity.pp_result is not None:
                rows.append(
                    {
                        "Module": "Stationarity",
                        "Test": "PP",
                        "Statistic": self.stationarity.pp_result.test_statistic,
                        "P-Value": self.stationarity.pp_result.p_value,
                        "Result": (
                            "Stationary"
                            if self.stationarity.pp_result.is_stationary
                            else "Non-stationary"
                        ),
                    }
                )

            # Add consensus row
            rows.append(
                {
                    "Module": "Stationarity",
                    "Test": "Consensus",
                    "Statistic": None,
                    "P-Value": None,
                    "Result": self.stationarity.consensus,
                }
            )

        # Autocorrelation tests
        if self.autocorrelation is not None:
            n_significant_acf = len(self.autocorrelation.significant_acf_lags)
            n_significant_pacf = len(self.autocorrelation.significant_pacf_lags)

            rows.append(
                {
                    "Module": "Autocorrelation",
                    "Test": "ACF",
                    "Statistic": None,  # No single max ACF statistic
                    "P-Value": None,
                    "Result": f"{n_significant_acf} significant lags",
                }
            )

            rows.append(
                {
                    "Module": "Autocorrelation",
                    "Test": "PACF",
                    "Statistic": None,  # No single max PACF statistic
                    "P-Value": None,
                    "Result": f"{n_significant_pacf} significant lags",
                }
            )

            rows.append(
                {
                    "Module": "Autocorrelation",
                    "Test": "Consensus",
                    "Statistic": None,
                    "P-Value": None,
                    "Result": (
                        "No autocorrelation"
                        if self.autocorrelation.is_white_noise
                        else "Has autocorrelation"
                    ),
                }
            )

        # Volatility tests
        if self.volatility is not None:
            rows.append(
                {
                    "Module": "Volatility",
                    "Test": "ARCH-LM",
                    "Statistic": self.volatility.arch_lm_result.test_statistic,
                    "P-Value": self.volatility.arch_lm_result.p_value,
                    "Result": (
                        "ARCH effects"
                        if self.volatility.arch_lm_result.has_arch_effects
                        else "No ARCH effects"
                    ),
                }
            )

            if self.volatility.garch_result is not None:
                # Note: Currently always GARCH(1,1) - p and q not stored in result
                rows.append(
                    {
                        "Module": "Volatility",
                        "Test": "GARCH",
                        "Statistic": None,
                        "P-Value": None,
                        "Result": "GARCH(1,1) fit",
                    }
                )

            rows.append(
                {
                    "Module": "Volatility",
                    "Test": "Consensus",
                    "Statistic": None,
                    "P-Value": None,
                    "Result": (
                        "Has clustering"
                        if self.volatility.has_volatility_clustering
                        else "No clustering"
                    ),
                }
            )

        # Distribution tests
        if self.distribution is not None:
            if self.distribution.moments_result is not None:
                rows.append(
                    {
                        "Module": "Distribution",
                        "Test": "Skewness",
                        "Statistic": self.distribution.moments_result.skewness,
                        "P-Value": None,
                        "Result": (
                            "Significant"
                            if self.distribution.moments_result.skewness_significant
                            else "Not significant"
                        ),
                    }
                )

                rows.append(
                    {
                        "Module": "Distribution",
                        "Test": "Excess Kurtosis",
                        "Statistic": self.distribution.moments_result.excess_kurtosis,
                        "P-Value": None,
                        "Result": (
                            "Significant"
                            if self.distribution.moments_result.excess_kurtosis_significant
                            else "Not significant"
                        ),
                    }
                )

            if self.distribution.jarque_bera_result is not None:
                rows.append(
                    {
                        "Module": "Distribution",
                        "Test": "Jarque-Bera",
                        "Statistic": self.distribution.jarque_bera_result.statistic,
                        "P-Value": self.distribution.jarque_bera_result.p_value,
                        "Result": (
                            "Normal"
                            if self.distribution.jarque_bera_result.is_normal
                            else "Not normal"
                        ),
                    }
                )

            if self.distribution.shapiro_wilk_result is not None:
                rows.append(
                    {
                        "Module": "Distribution",
                        "Test": "Shapiro-Wilk",
                        "Statistic": self.distribution.shapiro_wilk_result.statistic,
                        "P-Value": self.distribution.shapiro_wilk_result.p_value,
                        "Result": (
                            "Normal"
                            if self.distribution.shapiro_wilk_result.is_normal
                            else "Not normal"
                        ),
                    }
                )

            if (
                self.distribution.tail_analysis_result is not None
                and self.distribution.tail_analysis_result.hill_result is not None
            ):
                rows.append(
                    {
                        "Module": "Distribution",
                        "Test": "Hill Estimator",
                        "Statistic": self.distribution.tail_analysis_result.hill_result.tail_index,
                        "P-Value": None,
                        "Result": self.distribution.tail_analysis_result.hill_result.classification.replace(
                            "_", " "
                        ).title(),
                    }
                )

            rows.append(
                {
                    "Module": "Distribution",
                    "Test": "Recommended",
                    "Statistic": None,
                    "P-Value": None,
                    "Result": self.distribution.recommended_distribution,
                }
            )

        return pd.DataFrame(rows)

    def _generate_recommendations(self) -> list[str]:
        """Generate actionable recommendations based on diagnostic results.

        Returns:
            List of recommendation strings
        """
        recommendations = []

        # Stationarity recommendations
        if self.stationarity is not None:
            if self.stationarity.consensus in ["strong_nonstationary", "likely_nonstationary"]:
                recommendations.append(
                    "Feature is non-stationary. Consider differencing or detrending before use."
                )
            elif self.stationarity.consensus == "inconclusive":
                recommendations.append(
                    "Stationarity tests are inconclusive. Try longer time series or alternative transformations."
                )

        # Autocorrelation recommendations
        if self.autocorrelation is not None and not self.autocorrelation.is_white_noise:
            max_lag = max(
                self.autocorrelation.significant_acf_lags
                + self.autocorrelation.significant_pacf_lags,
                default=0,
            )
            recommendations.append(
                f"Feature has significant autocorrelation up to lag {max_lag}. "
                "Consider AR/MA modeling or including lagged values as features."
            )

        # Volatility recommendations
        if self.volatility is not None and self.volatility.has_volatility_clustering:
            if self.volatility.garch_result is not None:
                recommendations.append(
                    "Feature exhibits volatility clustering. GARCH(1,1) "
                    "model provides good fit. Consider using conditional volatility."
                )
            else:
                recommendations.append(
                    "Feature exhibits volatility clustering (ARCH effects). "
                    "Consider GARCH modeling or volatility-adjusted features."
                )

        # Distribution recommendations
        if self.distribution is not None:
            rec_dist = self.distribution.recommended_distribution

            if rec_dist != "normal":
                recommendations.append(
                    f"Feature distribution is not normal (recommended: {rec_dist}). "
                    "Consider robust statistics or distribution-specific modeling."
                )

            if (
                self.distribution.tail_analysis_result is not None
                and self.distribution.tail_analysis_result.hill_result is not None
            ):
                tail_index = self.distribution.tail_analysis_result.hill_result.tail_index

                if tail_index <= 2:
                    recommendations.append(
                        f"Feature has very heavy tails (α={tail_index:.2f}, variance may not exist). "
                        "Use robust statistics and be cautious with moment-based methods."
                    )
                elif tail_index <= 4:
                    recommendations.append(
                        f"Feature has heavy tails (α={tail_index:.2f}, kurtosis may not exist). "
                        "Consider Student-t or stable distributions."
                    )

        # General recommendations
        if not recommendations:
            recommendations.append(
                "Feature passes all diagnostic checks. Safe to use in modeling without transformation."
            )

        return recommendations

    def _calculate_health_score(self) -> float:
        """Calculate overall feature health score (0.0 to 1.0).

        Combines results from all modules into single score.
        Higher score = better feature quality.

        Returns:
            Health score from 0.0 (poor) to 1.0 (excellent)
        """
        score = 0.0
        max_score = 0.0

        # Stationarity contribution (0.25 weight)
        if self.stationarity is not None:
            max_score += 0.25
            if self.stationarity.consensus == "strong_stationary":
                score += 0.25
            elif self.stationarity.consensus == "likely_stationary":
                score += 0.20
            elif self.stationarity.consensus == "inconclusive":
                score += 0.10

        # Autocorrelation contribution (0.25 weight)
        # Note: Having some autocorrelation is OK, extreme is bad
        if self.autocorrelation is not None:
            max_score += 0.25
            n_significant_acf = len(self.autocorrelation.significant_acf_lags)
            if n_significant_acf == 0:
                score += 0.25  # No autocorrelation is good
            elif n_significant_acf <= 5:
                score += 0.20  # Some autocorrelation is manageable
            elif n_significant_acf <= 10:
                score += 0.10  # Moderate autocorrelation
            # else: 0.0 for excessive autocorrelation

        # Volatility contribution (0.25 weight)
        # Note: Having ARCH effects is OK if we can model them
        if self.volatility is not None:
            max_score += 0.25
            if not self.volatility.has_volatility_clustering:
                score += 0.25  # No clustering is ideal
            elif (
                self.volatility.garch_result is not None and self.volatility.garch_result.converged
            ):
                score += 0.20  # Has clustering but GARCH fits well
            else:
                score += 0.10  # Has clustering but harder to model

        # Distribution contribution (0.25 weight)
        if self.distribution is not None:
            max_score += 0.25
            rec_dist = self.distribution.recommended_distribution

            if rec_dist == "normal":
                score += 0.25  # Normal is ideal
            elif rec_dist in ["t", "heavy-tailed"]:
                score += 0.15  # Heavy tails but manageable
            elif rec_dist == "stable":
                score += 0.05  # Extreme tails, difficult to work with
            # else: "lognormal", "uniform" get default 0.0

        # Normalize to 0.0-1.0 range
        if max_score > 0:
            return score / max_score
        return 0.0

    def _identify_flags(self) -> list[str]:
        """Identify warning flags from diagnostic results.

        Returns:
            List of warning flag strings
        """
        flags = []

        # Stationarity flags
        if self.stationarity is not None:
            if self.stationarity.consensus in ["strong_nonstationary", "likely_nonstationary"]:
                flags.append("NON_STATIONARY")

        # Autocorrelation flags
        if self.autocorrelation is not None:
            n_significant_acf = len(self.autocorrelation.significant_acf_lags)
            if n_significant_acf > 10:
                flags.append("EXCESSIVE_AUTOCORRELATION")

        # Volatility flags
        if self.volatility is not None:
            if self.volatility.has_volatility_clustering:
                flags.append("VOLATILITY_CLUSTERING")
            if (
                self.volatility.garch_result is not None
                and not self.volatility.garch_result.converged
            ):
                flags.append("GARCH_NO_CONVERGENCE")

        # Distribution flags
        if self.distribution is not None:
            if not self.distribution.is_normal:
                flags.append("NON_NORMAL")

            if (
                self.distribution.tail_analysis_result is not None
                and self.distribution.tail_analysis_result.hill_result is not None
            ):
                tail_index = self.distribution.tail_analysis_result.hill_result.tail_index
                if tail_index <= 2:
                    flags.append("VERY_HEAVY_TAILS")
                elif tail_index <= 4:
                    flags.append("HEAVY_TAILS")

            if self.distribution.moments_result is not None:
                if abs(self.distribution.moments_result.skewness) > 1:
                    flags.append("HIGH_SKEWNESS")
                if self.distribution.moments_result.excess_kurtosis > 10:
                    flags.append("HIGH_EXCESS_KURTOSIS")

        return flags

    def summary(self) -> str:
        """Generate human-readable summary of diagnostic results.

        Returns:
            Multi-line summary string
        """
        lines = []
        lines.append(f"Feature Diagnostics: {self.feature_name}")
        lines.append(f"Observations: {self.n_obs}")
        lines.append(f"Health Score: {self.health_score:.2f}/1.00")
        lines.append("")

        # Module summaries
        if self.stationarity is not None:
            lines.append(f"Stationarity: {self.stationarity.consensus}")

        if self.autocorrelation is not None:
            n_significant = len(self.autocorrelation.significant_acf_lags)
            lines.append(f"Autocorrelation: {n_significant} significant lags")

        if self.volatility is not None:
            vol_str = (
                "Has clustering" if self.volatility.has_volatility_clustering else "No clustering"
            )
            lines.append(f"Volatility: {vol_str}")

        if self.distribution is not None:
            lines.append(f"Distribution: {self.distribution.recommended_distribution}")

        lines.append("")

        # Flags
        if self.flags:
            lines.append(f"Flags: {', '.join(self.flags)}")
            lines.append("")

        # Recommendations
        lines.append("Recommendations:")
        for i, rec in enumerate(self.recommendations, 1):
            lines.append(f"  {i}. {rec}")

        return "\n".join(lines)


class FeatureDiagnostics:
    """Main API class for feature-level diagnostic analysis.

    Orchestrates all Module A diagnostic tests (stationarity, autocorrelation,
    volatility, distribution) with configurable options and batch processing.

    Example:
        >>> import numpy as np
        >>> from ml4t.diagnostic.evaluation.feature_diagnostics import (
        ...     FeatureDiagnostics,
        ...     FeatureDiagnosticsConfig
        ... )
        >>>
        >>> # Configure and run diagnostics
        >>> config = FeatureDiagnosticsConfig()
        >>> diagnostics = FeatureDiagnostics(config)
        >>>
        >>> feature = np.random.randn(1000)
        >>> result = diagnostics.run_diagnostics(feature, name="my_feature")
        >>>
        >>> print(result.summary())
        >>> print(f"Health Score: {result.health_score:.2f}")
    """

    def __init__(self, config: FeatureDiagnosticsConfig | None = None):
        """Initialize FeatureDiagnostics with configuration.

        Args:
            config: Configuration object. If None, uses defaults (all tests enabled).
        """
        self.config = config or FeatureDiagnosticsConfig()

    def run_diagnostics(
        self,
        data: pd.Series | np.ndarray,
        name: str = "feature",
    ) -> FeatureDiagnosticsResult:
        """Run comprehensive diagnostic analysis on a single feature.

        Args:
            data: Feature time series (1D array or Series)
            name: Name/identifier for the feature

        Returns:
            FeatureDiagnosticsResult with all test results and recommendations

        Raises:
            ValidationError: If data is invalid (empty, wrong shape, etc.)

        Example:
            >>> import numpy as np
            >>> diagnostics = FeatureDiagnostics()
            >>> feature = np.random.randn(1000)
            >>> result = diagnostics.run_diagnostics(feature, name="returns")
            >>> print(result.summary())
        """
        # Validate input
        if isinstance(data, pd.Series):
            data_array = data.to_numpy()
            n_obs = len(data)
        elif isinstance(data, np.ndarray):
            if data.ndim != 1:
                raise ValidationError(f"Data must be 1-dimensional, got shape {data.shape}")
            data_array = data
            n_obs = len(data)
        else:
            raise ValidationError(
                f"Data must be pd.Series or np.ndarray, got {type(data).__name__}"
            )

        if n_obs == 0:
            raise ValidationError("Data must not be empty")

        if self.config.verbose:
            logger.info(f"Running diagnostics on feature '{name}' ({n_obs} observations)")

        # Initialize result containers
        stationarity_result = None
        autocorrelation_result = None
        volatility_result = None
        distribution_result = None

        # Run stationarity tests
        if self.config.run_stationarity:
            if self.config.verbose:
                logger.info("  Running stationarity tests...")

            try:
                stationarity_result = analyze_stationarity(
                    data_array,
                    alpha=self.config.alpha,
                    include_tests=self.config.stationarity_tests,
                )
            except Exception as e:
                logger.warning(f"Stationarity analysis failed: {e}")

        # Run autocorrelation analysis
        if self.config.run_autocorrelation:
            if self.config.verbose:
                logger.info("  Running autocorrelation analysis...")

            try:
                autocorrelation_result = analyze_autocorrelation(
                    data_array,
                    alpha=self.config.alpha,
                    max_lags=self.config.max_acf_lags,
                )
            except Exception as e:
                logger.warning(f"Autocorrelation analysis failed: {e}")

        # Run volatility clustering tests
        if self.config.run_volatility:
            if self.config.verbose:
                logger.info("  Running volatility clustering tests...")

            try:
                volatility_result = analyze_volatility(
                    data_array,
                    arch_lags=self.config.arch_lags,
                    fit_garch_model=self.config.fit_garch,
                    alpha=self.config.alpha,
                )
            except Exception as e:
                logger.warning(f"Volatility analysis failed: {e}")

        # Run distribution diagnostics
        if self.config.run_distribution:
            if self.config.verbose:
                logger.info("  Running distribution diagnostics...")

            try:
                distribution_result = analyze_distribution(
                    data_array,
                    alpha=self.config.alpha,
                    compute_tails=self.config.compute_tails,
                )
            except Exception as e:
                logger.warning(f"Distribution analysis failed: {e}")

        if self.config.verbose:
            logger.info("  Diagnostics complete")

        # Create and return result
        result = FeatureDiagnosticsResult(
            feature_name=name,
            n_obs=n_obs,
            stationarity=stationarity_result,
            autocorrelation=autocorrelation_result,
            volatility=volatility_result,
            distribution=distribution_result,
        )

        return result

    def run_batch_diagnostics(
        self,
        data: pd.DataFrame,
        feature_names: list[str] | None = None,
    ) -> dict[str, FeatureDiagnosticsResult]:
        """Run diagnostics on multiple features in batch.

        Args:
            data: DataFrame with features as columns
            feature_names: Column names to analyze. If None, analyzes all columns.

        Returns:
            Dictionary mapping feature name to FeatureDiagnosticsResult

        Example:
            >>> import pandas as pd
            >>> import numpy as np
            >>>
            >>> # Create multi-feature DataFrame
            >>> df = pd.DataFrame({
            ...     'momentum': np.random.randn(1000),
            ...     'mean_reversion': np.random.randn(1000),
            ...     'volatility': np.abs(np.random.randn(1000))
            ... })
            >>>
            >>> diagnostics = FeatureDiagnostics()
            >>> results = diagnostics.run_batch_diagnostics(df)
            >>>
            >>> for name, result in results.items():
            ...     print(f"\n{name}:")
            ...     print(f"  Health: {result.health_score:.2f}")
            ...     print(f"  Flags: {result.flags}")
        """
        if not isinstance(data, pd.DataFrame):
            raise ValidationError(
                f"Data must be pd.DataFrame for batch processing, got {type(data).__name__}"
            )

        if feature_names is None:
            feature_names = list(data.columns)

        if self.config.verbose:
            logger.info(f"Running batch diagnostics on {len(feature_names)} features")

        results = {}
        for name in feature_names:
            if name not in data.columns:
                logger.warning(f"Feature '{name}' not found in DataFrame, skipping")
                continue

            if self.config.verbose:
                logger.info(f"\nProcessing feature: {name}")

            results[name] = self.run_diagnostics(data[name], name=name)

        return results
