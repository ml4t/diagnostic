"""Result schemas for feature evaluation modules (A, B, C).

Module A: Feature Diagnostics (stationarity, ACF, volatility clustering)
Module B: Cross-Feature Analysis (correlations, PCA, clustering)
Module C: Feature-Outcome Relationships (IC analysis, threshold analysis)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import polars as pl
from pydantic import Field

from ml4t.diagnostic.results.base import BaseResult

if TYPE_CHECKING:
    from ml4t.diagnostic.integration.engineer_contract import EngineerConfig


# =============================================================================
# Module A: Feature Diagnostics
# =============================================================================


class StationarityTestResult(BaseResult):
    """Results from stationarity tests (ADF, KPSS, PP).

    Tests whether a time series is stationary (mean-reverting) or has unit root.

    Attributes:
        feature_name: Name of feature tested
        adf_statistic: Augmented Dickey-Fuller test statistic
        adf_pvalue: ADF p-value (reject H0 if < alpha => stationary)
        adf_is_stationary: Whether ADF indicates stationarity
        adf_critical_values: ADF critical values at 1%, 5%, 10% levels
        adf_lags_used: Number of lags used in ADF test
        adf_n_obs: Number of observations used in ADF test
        kpss_statistic: KPSS test statistic
        kpss_pvalue: KPSS p-value (reject H0 if < alpha => non-stationary)
        kpss_is_stationary: Whether KPSS indicates stationarity
        pp_statistic: Phillips-Perron test statistic
        pp_pvalue: PP p-value
        pp_is_stationary: Whether PP indicates stationarity
    """

    analysis_type: str = "stationarity_test"
    feature_name: str = Field(..., description="Feature name")

    # ADF test
    adf_statistic: float | None = Field(None, description="ADF test statistic")
    adf_pvalue: float | None = Field(None, description="ADF p-value")
    adf_is_stationary: bool | None = Field(None, description="ADF stationarity")
    adf_critical_values: dict[str, float] | None = Field(
        None, description="ADF critical values (1%, 5%, 10%)"
    )
    adf_lags_used: int | None = Field(None, description="Lags used in ADF test")
    adf_n_obs: int | None = Field(None, description="Observations in ADF test")

    # KPSS test
    kpss_statistic: float | None = Field(None, description="KPSS test statistic")
    kpss_pvalue: float | None = Field(None, description="KPSS p-value")
    kpss_is_stationary: bool | None = Field(None, description="KPSS stationarity")

    # Phillips-Perron test
    pp_statistic: float | None = Field(None, description="PP test statistic")
    pp_pvalue: float | None = Field(None, description="PP p-value")
    pp_is_stationary: bool | None = Field(None, description="PP stationarity")

    def list_available_dataframes(self) -> list[str]:
        """List available DataFrame views.

        Returns:
            List with single 'primary' view containing all test results
        """
        return ["primary"]

    def get_dataframe(self, name: str | None = None) -> pl.DataFrame:
        """Get test results as DataFrame.

        Args:
            name: DataFrame name (ignored, only 'primary' available)

        Returns:
            DataFrame with test statistics and conclusions

        Raises:
            ValueError: If name is provided but not 'primary'
        """
        if name is not None and name != "primary":
            raise ValueError(
                f"Unknown DataFrame name: {name}. Available: {self.list_available_dataframes()}"
            )

        data = {
            "feature": [self.feature_name],
            "adf_statistic": [self.adf_statistic],
            "adf_pvalue": [self.adf_pvalue],
            "adf_stationary": [self.adf_is_stationary],
            "adf_lags_used": [self.adf_lags_used],
            "adf_n_obs": [self.adf_n_obs],
            "kpss_statistic": [self.kpss_statistic],
            "kpss_pvalue": [self.kpss_pvalue],
            "kpss_stationary": [self.kpss_is_stationary],
            "pp_statistic": [self.pp_statistic],
            "pp_pvalue": [self.pp_pvalue],
            "pp_stationary": [self.pp_is_stationary],
        }
        return pl.DataFrame(data)

    def summary(self) -> str:
        """Human-readable summary of stationarity tests."""
        lines = [f"Stationarity Tests: {self.feature_name}"]
        if self.adf_is_stationary is not None:
            lines.append(
                f"  ADF: {'Stationary' if self.adf_is_stationary else 'Non-stationary'} (p={self.adf_pvalue:.4f})"
            )
        if self.kpss_is_stationary is not None:
            lines.append(
                f"  KPSS: {'Stationary' if self.kpss_is_stationary else 'Non-stationary'} (p={self.kpss_pvalue:.4f})"
            )
        if self.pp_is_stationary is not None:
            lines.append(
                f"  PP: {'Stationary' if self.pp_is_stationary else 'Non-stationary'} (p={self.pp_pvalue:.4f})"
            )
        return "\n".join(lines)


class ACFResult(BaseResult):
    """Autocorrelation Function (ACF) and Partial ACF analysis results.

    Detects serial correlation and lag structure in time series.

    Attributes:
        feature_name: Name of feature analyzed
        acf_values: ACF values at each lag
        pacf_values: PACF values at each lag
        significant_lags_acf: List of lags with significant ACF
        significant_lags_pacf: List of lags with significant PACF
        ljung_box_statistic: Ljung-Box test statistic
        ljung_box_pvalue: Ljung-Box p-value (reject H0 => autocorrelation present)
    """

    analysis_type: str = "acf_analysis"
    feature_name: str = Field(..., description="Feature name")

    acf_values: list[float] = Field(..., description="ACF at each lag")
    pacf_values: list[float] = Field(..., description="PACF at each lag")
    significant_lags_acf: list[int] = Field(
        default_factory=list, description="Lags with significant ACF"
    )
    significant_lags_pacf: list[int] = Field(
        default_factory=list, description="Lags with significant PACF"
    )

    ljung_box_statistic: float | None = Field(None, description="Ljung-Box statistic")
    ljung_box_pvalue: float | None = Field(None, description="Ljung-Box p-value")

    def list_available_dataframes(self) -> list[str]:
        """List available DataFrame views.

        Returns:
            List with single 'primary' view containing ACF/PACF values
        """
        return ["primary"]

    def get_dataframe(self, name: str | None = None) -> pl.DataFrame:
        """Get ACF/PACF values as DataFrame.

        Args:
            name: DataFrame name (ignored, only 'primary' available)

        Returns:
            DataFrame with lag, ACF, and PACF values

        Raises:
            ValueError: If name is provided but not 'primary'
        """
        if name is not None and name != "primary":
            raise ValueError(
                f"Unknown DataFrame name: {name}. Available: {self.list_available_dataframes()}"
            )

        n_lags = len(self.acf_values)
        data = {
            "lag": list(range(n_lags)),
            "acf": self.acf_values,
            "pacf": self.pacf_values,
        }
        return pl.DataFrame(data)

    def summary(self) -> str:
        """Human-readable summary of autocorrelation analysis."""
        lines = [f"ACF/PACF Analysis: {self.feature_name}"]
        lines.append(f"  Lags analyzed: {len(self.acf_values)}")
        lines.append(f"  Significant ACF lags: {self.significant_lags_acf}")
        lines.append(f"  Significant PACF lags: {self.significant_lags_pacf}")
        if self.ljung_box_pvalue is not None:
            lines.append(
                f"  Ljung-Box test: p={self.ljung_box_pvalue:.4f} "
                f"({'Autocorrelation present' if self.ljung_box_pvalue < 0.05 else 'No autocorrelation'})"
            )
        return "\n".join(lines)


class FeatureDiagnosticsResultSchema(BaseResult):
    """Complete results from Module A: Feature Diagnostics.

    Comprehensive analysis of individual feature properties:
    - Stationarity testing (ADF, KPSS, PP)
    - Autocorrelation structure (ACF, PACF)
    - Volatility clustering (GARCH effects)
    - Distribution characteristics (normality, skewness, kurtosis)

    Attributes:
        stationarity_tests: Stationarity test results for each feature
        acf_results: ACF/PACF analysis for each feature
        volatility_clustering: GARCH detection results
        distribution_stats: Distribution characteristics
    """

    analysis_type: str = "feature_diagnostics"

    stationarity_tests: list[StationarityTestResult] = Field(
        default_factory=list, description="Stationarity test results"
    )
    acf_results: list[ACFResult] = Field(
        default_factory=list, description="ACF/PACF analysis results"
    )
    volatility_clustering: dict[str, Any] = Field(
        default_factory=dict, description="GARCH detection results"
    )
    distribution_stats: dict[str, Any] = Field(
        default_factory=dict, description="Distribution characteristics"
    )

    def get_stationarity_dataframe(self) -> pl.DataFrame:
        """Get stationarity test results as DataFrame.

        Returns:
            DataFrame with all stationarity tests
        """
        if not self.stationarity_tests:
            return pl.DataFrame()

        # Combine all test results
        dfs = [test.get_dataframe() for test in self.stationarity_tests]
        return pl.concat(dfs)

    def get_acf_dataframe(self, feature_name: str | None = None) -> pl.DataFrame:
        """Get ACF/PACF results as DataFrame.

        Args:
            feature_name: Optional filter by feature

        Returns:
            DataFrame with ACF/PACF values
        """
        if not self.acf_results:
            return pl.DataFrame()

        results = self.acf_results
        if feature_name:
            results = [r for r in results if r.feature_name == feature_name]

        dfs = []
        for result in results:
            df = result.get_dataframe()
            df = df.with_columns(pl.lit(result.feature_name).alias("feature"))
            dfs.append(df)

        return pl.concat(dfs) if dfs else pl.DataFrame()

    def get_dataframe(self, name: str | None = None) -> pl.DataFrame:
        """Get results as DataFrame.

        Args:
            name: 'stationarity' or 'acf'

        Returns:
            Requested DataFrame
        """
        if name == "stationarity":
            return self.get_stationarity_dataframe()
        elif name == "acf":
            return self.get_acf_dataframe()
        else:
            return self.get_stationarity_dataframe()

    def summary(self) -> str:
        """Human-readable summary of diagnostics."""
        lines = ["Feature Diagnostics Summary", "=" * 40]
        lines.append(f"Features analyzed: {len(self.stationarity_tests)}")
        lines.append("")

        # Stationarity summary
        if self.stationarity_tests:
            stationary = sum(
                1 for t in self.stationarity_tests if t.adf_is_stationary or t.kpss_is_stationary
            )
            lines.append(f"Stationary features: {stationary}/{len(self.stationarity_tests)}")

        # ACF summary
        if self.acf_results:
            with_autocorr = sum(1 for r in self.acf_results if r.significant_lags_acf)
            lines.append(f"Features with autocorrelation: {with_autocorr}/{len(self.acf_results)}")

        return "\n".join(lines)

    def to_engineer_config(self) -> EngineerConfig:
        """Generate preprocessing recommendations for ML4T Engineer.

        Analyzes diagnostic results to recommend appropriate transforms:
        - Non-stationary → DIFF (first difference)
        - High skewness (>2) → LOG or SQRT transform
        - Outliers detected → WINSORIZE
        - Already good quality → NONE

        Returns:
            EngineerConfig with preprocessing recommendations

        Example:
            >>> diagnostics = evaluator.evaluate_diagnostics(features_df)
            >>> eng_config = diagnostics.to_engineer_config()
            >>> preprocessing_dict = eng_config.to_dict()
        """
        from ml4t.diagnostic.integration.engineer_contract import (
            EngineerConfig,
            PreprocessingRecommendation,
            TransformType,
        )

        recommendations = []

        # Process stationarity tests
        for stationarity in self.stationarity_tests:
            feature_name = stationarity.feature_name

            # Check if non-stationary (both ADF and KPSS should agree ideally)
            adf_non_stationary = (
                stationarity.adf_is_stationary is not None and not stationarity.adf_is_stationary
            )
            kpss_non_stationary = (
                stationarity.kpss_is_stationary is not None and not stationarity.kpss_is_stationary
            )
            pp_non_stationary = (
                stationarity.pp_is_stationary is not None and not stationarity.pp_is_stationary
            )

            # Count non-stationary signals
            non_stationary_count = sum([adf_non_stationary, kpss_non_stationary, pp_non_stationary])

            if non_stationary_count >= 2:
                # At least 2 tests indicate non-stationarity
                confidence = 0.9 if non_stationary_count == 3 else 0.8
                diagnostics_dict = {}
                if stationarity.adf_pvalue is not None:
                    diagnostics_dict["adf_pvalue"] = stationarity.adf_pvalue
                if stationarity.kpss_pvalue is not None:
                    diagnostics_dict["kpss_pvalue"] = stationarity.kpss_pvalue

                recommendations.append(
                    PreprocessingRecommendation(
                        feature_name=feature_name,
                        transform=TransformType.DIFF,
                        reason=f"Feature is non-stationary ({non_stationary_count}/3 tests)",
                        confidence=confidence,
                        diagnostics=diagnostics_dict if diagnostics_dict else None,
                    )
                )
            elif non_stationary_count == 1:
                # Only 1 test indicates non-stationarity - lower confidence
                test_name = "ADF" if adf_non_stationary else "KPSS" if kpss_non_stationary else "PP"
                pvalue: float | None = getattr(stationarity, f"{test_name.lower()}_pvalue")
                single_test_diagnostics: dict[str, float] | None = (
                    {f"{test_name.lower()}_pvalue": pvalue} if pvalue is not None else None
                )
                recommendations.append(
                    PreprocessingRecommendation(
                        feature_name=feature_name,
                        transform=TransformType.DIFF,
                        reason=f"Possible non-stationarity ({test_name} test)",
                        confidence=0.6,
                        diagnostics=single_test_diagnostics,
                    )
                )
            else:
                # Stationary - no transform needed
                recommendations.append(
                    PreprocessingRecommendation(
                        feature_name=feature_name,
                        transform=TransformType.NONE,
                        reason="Feature is stationary (all tests)",
                        confidence=0.9,
                    )
                )

        # Check distribution stats for skewness/outliers
        # (This is a placeholder - actual implementation depends on what's in distribution_stats)
        if self.distribution_stats:
            for feature_name, stats in self.distribution_stats.items():
                # Skip if already recommended differencing
                if any(
                    r.feature_name == feature_name and r.transform == TransformType.DIFF
                    for r in recommendations
                ):
                    continue

                # Check for high skewness
                skewness = stats.get("skewness")
                if skewness is not None and abs(skewness) > 2:
                    # High positive skew → log transform
                    if skewness > 2:
                        recommendations.append(
                            PreprocessingRecommendation(
                                feature_name=feature_name,
                                transform=TransformType.LOG,
                                reason=f"High right skew (skewness={skewness:.2f})",
                                confidence=0.85,
                                diagnostics={"skewness": skewness},
                            )
                        )
                    # High negative skew → reflect and log (but we'll use sqrt as milder)
                    else:
                        recommendations.append(
                            PreprocessingRecommendation(
                                feature_name=feature_name,
                                transform=TransformType.SQRT,
                                reason=f"High left skew (skewness={skewness:.2f})",
                                confidence=0.75,
                                diagnostics={"skewness": skewness},
                            )
                        )

                # Check for outliers
                has_outliers = stats.get("has_outliers", False)
                if has_outliers:
                    recommendations.append(
                        PreprocessingRecommendation(
                            feature_name=feature_name,
                            transform=TransformType.WINSORIZE,
                            reason="Outliers detected at tail percentiles",
                            confidence=0.8,
                        )
                    )

        return EngineerConfig(
            recommendations=recommendations,
            metadata={
                "created_at": self.created_at,
                "diagnostic_version": self.version,
            },
        )


# =============================================================================
# Module B: Cross-Feature Analysis
# =============================================================================


class CrossFeatureResult(BaseResult):
    """Results from Module B: Cross-Feature Analysis.

    Analysis of relationships between features:
    - Correlation matrix
    - PCA (dimensionality reduction)
    - Clustering (feature groups)
    - Redundancy detection

    Attributes:
        correlation_matrix: Correlation matrix (stored as nested list for JSON)
        feature_names: List of feature names
        pca_results: PCA analysis results (variance explained, loadings)
        clustering_results: Feature clustering results
        redundant_features: Highly correlated feature pairs
    """

    analysis_type: str = "cross_feature"

    correlation_matrix: list[list[float]] = Field(
        ..., description="Correlation matrix as nested list"
    )
    feature_names: list[str] = Field(..., description="Feature names in matrix order")

    pca_results: dict[str, Any] | None = Field(
        None, description="PCA analysis (variance explained, loadings)"
    )
    clustering_results: dict[str, Any] | None = Field(
        None, description="Feature clustering results"
    )
    redundant_features: list[tuple[str, str, float]] | None = Field(
        None, description="Redundant pairs: (feature1, feature2, correlation)"
    )

    def get_correlation_dataframe(self) -> pl.DataFrame:
        """Get correlation matrix as DataFrame.

        Returns:
            DataFrame with correlations in long format
        """
        # Convert to long format for easier manipulation
        n = len(self.feature_names)
        rows = []
        for i in range(n):
            for j in range(n):
                rows.append(
                    {
                        "feature_1": self.feature_names[i],
                        "feature_2": self.feature_names[j],
                        "correlation": self.correlation_matrix[i][j],
                    }
                )
        return pl.DataFrame(rows)

    def get_redundancy_dataframe(self) -> pl.DataFrame:
        """Get redundant feature pairs as DataFrame.

        Returns:
            DataFrame with redundant pairs
        """
        if not self.redundant_features:
            return pl.DataFrame(
                schema={"feature_1": pl.Utf8, "feature_2": pl.Utf8, "correlation": pl.Float64}
            )

        rows = [
            {"feature_1": f1, "feature_2": f2, "correlation": corr}
            for f1, f2, corr in self.redundant_features
        ]
        return pl.DataFrame(rows)

    def get_dataframe(self, name: str | None = None) -> pl.DataFrame:
        """Get results as DataFrame.

        Args:
            name: 'correlation' or 'redundancy'

        Returns:
            Requested DataFrame
        """
        if name == "redundancy":
            return self.get_redundancy_dataframe()
        else:
            return self.get_correlation_dataframe()

    def summary(self) -> str:
        """Human-readable summary of cross-feature analysis."""
        lines = ["Cross-Feature Analysis Summary", "=" * 40]
        lines.append(f"Features analyzed: {len(self.feature_names)}")

        if self.redundant_features:
            lines.append(f"Redundant pairs detected: {len(self.redundant_features)}")
            for f1, f2, corr in self.redundant_features[:5]:  # Show top 5
                lines.append(f"  {f1} <-> {f2}: {corr:.3f}")
            if len(self.redundant_features) > 5:
                lines.append(f"  ... and {len(self.redundant_features) - 5} more")

        if self.pca_results:
            variance = self.pca_results.get("variance_explained", [])
            if variance:
                lines.append(
                    f"PCA: {len(variance)} components explain {sum(variance):.1%} variance"
                )

        return "\n".join(lines)


# =============================================================================
# Module C: Feature-Outcome Relationships
# =============================================================================


class ICAnalysisResult(BaseResult):
    """Information Coefficient (IC) analysis for a single feature.

    Measures correlation between feature ranks and outcome ranks,
    with HAC adjustment for autocorrelation.

    Attributes:
        feature_name: Feature being analyzed
        ic_values: IC at each lag (if lagged analysis)
        mean_ic: Average IC across lags
        ic_std: Standard deviation of IC
        ic_ir: Information Ratio (mean_ic / ic_std)
        pvalue: P-value for IC significance
        hac_adjusted_pvalue: HAC-adjusted p-value
    """

    analysis_type: str = "ic_analysis"
    feature_name: str = Field(..., description="Feature name")

    ic_values: list[float] = Field(..., description="IC at each lag")
    mean_ic: float = Field(..., description="Mean IC")
    ic_std: float = Field(..., description="IC standard deviation")
    ic_ir: float = Field(..., description="Information Ratio (mean / std)")

    pvalue: float | None = Field(None, description="P-value for IC significance")
    hac_adjusted_pvalue: float | None = Field(None, description="HAC-adjusted p-value")

    def get_dataframe(self, name: str | None = None) -> pl.DataFrame:
        """Get IC values as DataFrame.

        Args:
            name: Unused, included for base class compatibility.

        Returns:
            DataFrame with lag and IC values
        """
        del name  # Unused, base class compatibility
        data = {
            "feature": [self.feature_name] * len(self.ic_values),
            "lag": list(range(len(self.ic_values))),
            "ic": self.ic_values,
        }
        return pl.DataFrame(data)

    def summary(self) -> str:
        """Human-readable summary of IC analysis."""
        lines = [f"IC Analysis: {self.feature_name}"]
        lines.append(f"  Mean IC: {self.mean_ic:.4f}")
        lines.append(f"  IC IR: {self.ic_ir:.4f}")
        if self.hac_adjusted_pvalue is not None:
            sig = "Significant" if self.hac_adjusted_pvalue < 0.05 else "Not significant"
            lines.append(f"  HAC p-value: {self.hac_adjusted_pvalue:.4f} ({sig})")
        return "\n".join(lines)


class ThresholdAnalysisResult(BaseResult):
    """Binary classification threshold analysis for a single feature.

    Evaluates feature as binary signal using optimal threshold.

    Attributes:
        feature_name: Feature being analyzed
        optimal_threshold: Threshold value that optimizes target metric
        precision: Precision at optimal threshold
        recall: Recall at optimal threshold
        f1_score: F1 score at optimal threshold
        lift: Lift over base rate
        coverage: Fraction of observations with positive signal
    """

    analysis_type: str = "threshold_analysis"
    feature_name: str = Field(..., description="Feature name")

    optimal_threshold: float = Field(..., description="Optimal threshold value")
    precision: float = Field(..., description="Precision at optimal threshold")
    recall: float = Field(..., description="Recall at optimal threshold")
    f1_score: float = Field(..., description="F1 score at optimal threshold")
    lift: float = Field(..., description="Lift over base rate")
    coverage: float = Field(..., description="Signal coverage (fraction positive)")

    def get_dataframe(self, name: str | None = None) -> pl.DataFrame:
        """Get threshold analysis as DataFrame.

        Args:
            name: Unused, included for base class compatibility.

        Returns:
            Single-row DataFrame with all metrics
        """
        del name  # Unused, base class compatibility
        data = {
            "feature": [self.feature_name],
            "threshold": [self.optimal_threshold],
            "precision": [self.precision],
            "recall": [self.recall],
            "f1_score": [self.f1_score],
            "lift": [self.lift],
            "coverage": [self.coverage],
        }
        return pl.DataFrame(data)

    def summary(self) -> str:
        """Human-readable summary of threshold analysis."""
        lines = [f"Threshold Analysis: {self.feature_name}"]
        lines.append(f"  Optimal threshold: {self.optimal_threshold:.4f}")
        lines.append(f"  Precision: {self.precision:.2%}")
        lines.append(f"  Recall: {self.recall:.2%}")
        lines.append(f"  F1 Score: {self.f1_score:.2%}")
        lines.append(f"  Lift: {self.lift:.2f}x")
        lines.append(f"  Coverage: {self.coverage:.2%}")
        return "\n".join(lines)


class FeatureOutcomeResultSchema(BaseResult):
    """Complete results from Module C: Feature-Outcome Relationships.

    Analysis of how features relate to outcomes:
    - IC analysis (rank correlations)
    - Threshold analysis (binary classification)
    - ML feature importance (if applicable)

    Attributes:
        ic_results: IC analysis for each feature
        threshold_results: Threshold analysis for each feature
        ml_importance: ML feature importance scores
    """

    analysis_type: str = "feature_outcome"

    ic_results: list[ICAnalysisResult] = Field(
        default_factory=list, description="IC analysis per feature"
    )
    threshold_results: list[ThresholdAnalysisResult] | None = Field(
        None, description="Threshold analysis per feature"
    )
    ml_importance: dict[str, float] | None = Field(
        None, description="ML feature importance: {feature: importance}"
    )

    def get_ic_dataframe(self) -> pl.DataFrame:
        """Get IC analysis as DataFrame.

        Returns:
            DataFrame with IC metrics for all features
        """
        if not self.ic_results:
            return pl.DataFrame()

        rows = []
        for result in self.ic_results:
            rows.append(
                {
                    "feature": result.feature_name,
                    "mean_ic": result.mean_ic,
                    "ic_std": result.ic_std,
                    "ic_ir": result.ic_ir,
                    "pvalue": result.pvalue,
                    "hac_pvalue": result.hac_adjusted_pvalue,
                }
            )
        return pl.DataFrame(rows)

    def get_threshold_dataframe(self) -> pl.DataFrame:
        """Get threshold analysis as DataFrame.

        Returns:
            DataFrame with threshold metrics for all features
        """
        if not self.threshold_results:
            return pl.DataFrame()

        dfs = [result.get_dataframe() for result in self.threshold_results]
        return pl.concat(dfs)

    def get_dataframe(self, name: str | None = None) -> pl.DataFrame:
        """Get results as DataFrame.

        Args:
            name: 'ic' or 'threshold'

        Returns:
            Requested DataFrame
        """
        if name == "threshold":
            return self.get_threshold_dataframe()
        else:
            return self.get_ic_dataframe()

    def summary(self) -> str:
        """Human-readable summary of feature-outcome relationships."""
        lines = ["Feature-Outcome Analysis Summary", "=" * 40]

        if self.ic_results:
            lines.append(f"IC analysis: {len(self.ic_results)} features")
            significant = sum(
                1 for r in self.ic_results if r.hac_adjusted_pvalue and r.hac_adjusted_pvalue < 0.05
            )
            lines.append(f"  Significant features: {significant}")

            # Top features by IC
            top = sorted(self.ic_results, key=lambda r: abs(r.mean_ic), reverse=True)[:3]
            lines.append("  Top 3 by |IC|:")
            for r in top:
                lines.append(f"    {r.feature_name}: IC={r.mean_ic:.4f}, IR={r.ic_ir:.4f}")

        if self.threshold_results:
            lines.append("")
            lines.append(f"Threshold analysis: {len(self.threshold_results)} features")
            # Top features by F1
            top = sorted(self.threshold_results, key=lambda r: r.f1_score, reverse=True)[:3]
            lines.append("  Top 3 by F1:")
            for r in top:
                lines.append(f"    {r.feature_name}: F1={r.f1_score:.2%}, Lift={r.lift:.2f}x")

        return "\n".join(lines)
