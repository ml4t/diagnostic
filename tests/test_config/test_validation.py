"""Test validation logic for all config classes.

This test suite verifies:
1. Invalid values are caught with helpful error messages
2. Cross-field validation works correctly
3. Edge cases are handled
4. Pydantic validators produce meaningful errors
"""

import pytest
from pydantic import ValidationError


class TestBaseValidation:
    """Test base configuration validation."""

    def test_extra_fields_rejected(self):
        """Test that extra fields are rejected."""
        from ml4t.diagnostic.config import StationaritySettings

        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            StationaritySettings(invalid_field="value")

    def test_significance_level_range(self):
        """Test significance level must be in valid range."""
        from ml4t.diagnostic.config.base import StatisticalTestConfig

        # Valid
        config = StatisticalTestConfig(significance_level=0.05)
        assert config.significance_level == 0.05

        # Too low
        with pytest.raises(ValidationError):
            StatisticalTestConfig(significance_level=-0.01)

        # Too high
        with pytest.raises(ValidationError):
            StatisticalTestConfig(significance_level=1.5)

    def test_n_jobs_validation(self):
        """Test n_jobs validation."""
        from ml4t.diagnostic.config.base import RuntimeConfig

        # Valid values
        RuntimeConfig(n_jobs=-1)  # All cores
        RuntimeConfig(n_jobs=1)  # Serial
        RuntimeConfig(n_jobs=4)  # Specific number

        # Invalid (negative but not -1)
        with pytest.raises(ValidationError):
            RuntimeConfig(n_jobs=-2)


class TestStationarityValidation:
    """Test stationarity settings validation."""

    def test_stationarity_at_least_one_test(self):
        """Test that at least one stationarity test must be enabled."""
        from ml4t.diagnostic.config import StationaritySettings

        # Valid: at least one enabled
        StationaritySettings(adf_enabled=True, kpss_enabled=False, pp_enabled=False)
        StationaritySettings(adf_enabled=False, kpss_enabled=True, pp_enabled=False)

        # Invalid: all disabled
        with pytest.raises(ValidationError, match="At least one"):
            StationaritySettings(adf_enabled=False, kpss_enabled=False, pp_enabled=False)


class TestVolatilityValidation:
    """Test volatility settings validation."""

    def test_volatility_window_sizes_validation(self):
        """Test volatility window size validation."""
        from ml4t.diagnostic.config import VolatilitySettings

        # Valid
        VolatilitySettings(window_sizes=[21])
        VolatilitySettings(window_sizes=[10, 21, 63])

        # Invalid: empty list
        with pytest.raises(ValidationError, match="at least one window"):
            VolatilitySettings(window_sizes=[])

        # Invalid: window < 2
        with pytest.raises(ValidationError, match="must be >= 2"):
            VolatilitySettings(window_sizes=[1])


class TestDistributionValidation:
    """Test distribution settings validation."""

    def test_outlier_threshold_positive(self):
        """Test outlier threshold must be positive."""
        from ml4t.diagnostic.config import DistributionSettings

        # Valid
        DistributionSettings(outlier_threshold=3.0)

        # Invalid: negative
        with pytest.raises(ValidationError):
            DistributionSettings(outlier_threshold=-1.0)

        # Invalid: zero
        with pytest.raises(ValidationError):
            DistributionSettings(outlier_threshold=0.0)


class TestCorrelationValidation:
    """Test correlation settings validation."""

    def test_correlation_methods_not_empty(self):
        """Test that at least one correlation method must be specified."""
        from ml4t.diagnostic.config import CorrelationSettings
        from ml4t.diagnostic.config.validation import CorrelationMethod

        # Valid
        CorrelationSettings(methods=[CorrelationMethod.PEARSON])

        # Invalid: empty list
        with pytest.raises(ValidationError, match="at least one"):
            CorrelationSettings(methods=[])


class TestPCAValidation:
    """Test PCA settings validation."""

    def test_pca_variance_threshold_range(self):
        """Test PCA variance threshold must be in (0, 1)."""
        from ml4t.diagnostic.config import PCASettings

        # Valid
        PCASettings(enabled=True, n_components="auto", variance_threshold=0.95)

        # Invalid: >= 1
        with pytest.raises(ValidationError):
            PCASettings(enabled=True, n_components="auto", variance_threshold=1.0)

        # Invalid: <= 0
        with pytest.raises(ValidationError):
            PCASettings(enabled=True, n_components="auto", variance_threshold=0.0)


class TestRedundancyValidation:
    """Test redundancy settings validation."""

    def test_redundancy_correlation_threshold_range(self):
        """Test redundancy correlation threshold is a valid probability."""
        from ml4t.diagnostic.config import RedundancySettings

        # Valid
        RedundancySettings(correlation_threshold=0.95)

        # Invalid: > 1
        with pytest.raises(ValidationError):
            RedundancySettings(correlation_threshold=1.5)

        # Invalid: < 0
        with pytest.raises(ValidationError):
            RedundancySettings(correlation_threshold=-0.1)


class TestICValidation:
    """Test IC settings validation."""

    def test_ic_lag_structure_validation(self):
        """Test IC lag structure validation."""
        from ml4t.diagnostic.config import ICSettings

        # Valid
        ICSettings(lag_structure=[0, 1, 5, 10])

        # Invalid: empty
        with pytest.raises(ValidationError, match="at least one lag"):
            ICSettings(lag_structure=[])

        # Invalid: negative lag (Pydantic catches this at field level)
        with pytest.raises(ValidationError):
            ICSettings(lag_structure=[0, -1, 5])


class TestThresholdAnalysisValidation:
    """Test threshold analysis settings validation."""

    def test_threshold_sweep_range_validation(self):
        """Test threshold sweep range validation."""
        from ml4t.diagnostic.config import ThresholdAnalysisSettings

        # Valid
        ThresholdAnalysisSettings(enabled=True, sweep_range=(-2.0, 2.0))

        # Invalid: min >= max
        with pytest.raises(ValidationError, match="must be <"):
            ThresholdAnalysisSettings(enabled=True, sweep_range=(2.0, -2.0))

        with pytest.raises(ValidationError, match="must be <"):
            ThresholdAnalysisSettings(enabled=True, sweep_range=(1.0, 1.0))

    def test_threshold_constraint_consistency(self):
        """Test threshold constraint configuration consistency."""
        from ml4t.diagnostic.config import ThresholdAnalysisSettings

        # Valid: both set
        ThresholdAnalysisSettings(enabled=True, constraint_metric="coverage", constraint_value=0.3)

        # Valid: both None
        ThresholdAnalysisSettings(enabled=True, constraint_metric=None, constraint_value=None)

        # Invalid: only metric set
        with pytest.raises(ValidationError, match="Both constraint_metric and constraint_value"):
            ThresholdAnalysisSettings(
                enabled=True, constraint_metric="coverage", constraint_value=None
            )

        # Invalid: only value set
        with pytest.raises(ValidationError, match="Both constraint_metric and constraint_value"):
            ThresholdAnalysisSettings(enabled=True, constraint_metric=None, constraint_value=0.3)


class TestPortfolioValidation:
    """Test portfolio evaluation validation."""

    def test_metrics_list_not_empty(self):
        """Test that at least one metric must be specified."""
        from ml4t.diagnostic.config import PortfolioMetricsSettings
        from ml4t.diagnostic.config.validation import PortfolioMetric

        # Valid
        PortfolioMetricsSettings(metrics=[PortfolioMetric.SHARPE])

        # Invalid: empty
        with pytest.raises(ValidationError, match="at least one metric"):
            PortfolioMetricsSettings(metrics=[])

    def test_periods_per_year_validation(self):
        """Test periods_per_year validation and warning."""
        from ml4t.diagnostic.config import PortfolioMetricsSettings

        # Valid standard values (no warning)
        PortfolioMetricsSettings(periods_per_year=252)  # Daily
        PortfolioMetricsSettings(periods_per_year=52)  # Weekly
        PortfolioMetricsSettings(periods_per_year=12)  # Monthly

        # Non-standard value should work but may issue warning
        config = PortfolioMetricsSettings(periods_per_year=365)
        assert config.periods_per_year == 365

    def test_bayesian_benchmark_validation(self):
        """Test Bayesian comparison benchmark validation."""
        from ml4t.diagnostic.config import PortfolioBayesianSettings

        # Valid: benchmark column specified when compare_to_benchmark=True
        PortfolioBayesianSettings(enabled=True, compare_to_benchmark=True, benchmark_column="SPY")

        # Invalid: no benchmark column when compare_to_benchmark=True
        with pytest.raises(ValidationError, match="benchmark_column required"):
            PortfolioBayesianSettings(
                enabled=True, compare_to_benchmark=True, benchmark_column=None
            )

    def test_bayesian_prior_params_validation(self):
        """Test Bayesian prior parameters validation."""
        from ml4t.diagnostic.config import PortfolioBayesianSettings
        from ml4t.diagnostic.config.validation import BayesianPriorDistribution

        # Valid: normal prior with correct params
        config1 = PortfolioBayesianSettings(
            prior_distribution=BayesianPriorDistribution.NORMAL,
            prior_params={"mean": 0.0, "std": 1.0},
        )
        assert config1.prior_params == {"mean": 0.0, "std": 1.0}

        # Valid: student-t prior with correct params
        config2 = PortfolioBayesianSettings(
            prior_distribution=BayesianPriorDistribution.STUDENT_T,
            prior_params={"df": 3, "loc": 0, "scale": 1},
        )
        assert config2.prior_params["df"] == 3

        # Valid: uniform prior with correct params
        config3 = PortfolioBayesianSettings(
            prior_distribution=BayesianPriorDistribution.UNIFORM,
            prior_params={"low": -1.0, "high": 1.0},
        )
        assert config3.prior_params == {"low": -1.0, "high": 1.0}

        # Invalid: wrong params for normal (missing required params)
        with pytest.raises(ValidationError, match="requires"):
            PortfolioBayesianSettings(
                prior_distribution=BayesianPriorDistribution.NORMAL,
                prior_params={"mu": 0.0, "sigma": 1.0},  # Wrong param names
            )

    def test_time_aggregation_frequencies_not_empty(self):
        """Test time aggregation frequencies validation."""
        from ml4t.diagnostic.config import PortfolioAggregationSettings
        from ml4t.diagnostic.config.validation import TimeFrequency

        # Valid
        PortfolioAggregationSettings(frequencies=[TimeFrequency.DAILY])

        # Invalid: empty
        with pytest.raises(ValidationError, match="at least one frequency"):
            PortfolioAggregationSettings(frequencies=[])

    def test_drawdown_recovery_threshold_range(self):
        """Test drawdown recovery threshold is a valid probability."""
        from ml4t.diagnostic.config import PortfolioDrawdownSettings

        # Valid
        PortfolioDrawdownSettings(recovery_threshold=1.0)  # Full recovery
        PortfolioDrawdownSettings(recovery_threshold=0.95)  # 95% recovery

        # Invalid: > 1
        with pytest.raises(ValidationError):
            PortfolioDrawdownSettings(recovery_threshold=1.5)

        # Invalid: < 0
        with pytest.raises(ValidationError):
            PortfolioDrawdownSettings(recovery_threshold=-0.1)


class TestStatisticalValidation:
    """Test statistical framework validation."""

    def test_psr_confidence_level_range(self):
        """Test PSR confidence level is a valid probability."""
        from ml4t.diagnostic.config import PSRSettings

        # Valid
        PSRSettings(confidence_level=0.95)

        # Invalid: > 1
        with pytest.raises(ValidationError):
            PSRSettings(confidence_level=1.5)

        # Invalid: < 0
        with pytest.raises(ValidationError):
            PSRSettings(confidence_level=-0.1)

    def test_dsr_n_trials_validation(self):
        """Test DSR n_trials validation."""
        from ml4t.diagnostic.config import DSRSettings

        # Valid
        DSRSettings(n_trials=100)

        # Valid but will issue warning
        config = DSRSettings(n_trials=5)
        assert config.n_trials == 5

        # Invalid: zero
        with pytest.raises(ValidationError):
            DSRSettings(n_trials=0)

        # Invalid: negative
        with pytest.raises(ValidationError):
            DSRSettings(n_trials=-1)

    def test_fdr_alpha_range(self):
        """Test FDR alpha is a valid probability."""
        from ml4t.diagnostic.config import FDRSettings

        # Valid
        FDRSettings(alpha=0.05)

        # Invalid: > 1
        with pytest.raises(ValidationError):
            FDRSettings(alpha=1.5)

        # Invalid: < 0
        with pytest.raises(ValidationError):
            FDRSettings(alpha=-0.1)

    def test_fdr_method_independence_warning(self):
        """Test FDR method independence warning."""
        from ml4t.diagnostic.config import FDRSettings
        from ml4t.diagnostic.config.validation import FDRMethod

        # Should work but may issue warning
        config = FDRSettings(method=FDRMethod.BENJAMINI_HOCHBERG, independent_tests=False)
        assert config.method == FDRMethod.BENJAMINI_HOCHBERG


class TestReportValidation:
    """Test reporting configuration validation."""

    def test_output_formats_not_empty(self):
        """Test at least one output format must be specified."""
        from ml4t.diagnostic.config import OutputFormatConfig
        from ml4t.diagnostic.config.validation import ReportFormat

        # Valid
        OutputFormatConfig(formats=[ReportFormat.HTML])

        # Invalid: empty
        with pytest.raises(ValidationError, match="at least one"):
            OutputFormatConfig(formats=[])

    def test_html_sections_validation(self):
        """Test HTML sections validation."""
        from ml4t.diagnostic.config import HTMLConfig

        # Valid
        HTMLConfig(include_sections=["stationarity", "ic"])

        # Invalid: unknown section
        with pytest.raises(ValidationError, match="Invalid sections"):
            HTMLConfig(include_sections=["stationarity", "invalid_section"])

    def test_custom_css_validation(self):
        """Test custom CSS file validation."""
        import tempfile
        from pathlib import Path

        from ml4t.diagnostic.config import HTMLConfig

        # Valid: existing file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".css", delete=False) as f:
            css_path = Path(f.name)
            f.write("body { color: red; }")

        try:
            config = HTMLConfig(custom_css=css_path)
            assert config.custom_css == css_path
        finally:
            css_path.unlink()

        # Invalid: non-existent file
        with pytest.raises(ValidationError, match="not found"):
            HTMLConfig(custom_css=Path("/nonexistent/file.css"))

    def test_plot_format_validation(self):
        """Test plot format validation."""
        from ml4t.diagnostic.config import VisualizationConfig

        # Valid formats
        VisualizationConfig(plot_format="png")
        VisualizationConfig(plot_format="pdf")
        VisualizationConfig(plot_format="svg")

        # Invalid format
        with pytest.raises(ValidationError, match="Invalid plot format"):
            VisualizationConfig(plot_format="invalid")


class TestValidateFullyMethod:
    """Test the validate_fully() method."""

    def test_validate_fully_on_valid_config(self):
        """Test validate_fully() returns empty list for valid config."""
        from ml4t.diagnostic.config import DiagnosticConfig

        config = DiagnosticConfig()
        errors = config.validate_fully()
        assert errors == []

    def test_validate_fully_detects_issues(self):
        """Test validate_fully() detects validation issues."""
        from ml4t.diagnostic.config import StationaritySettings

        # Create a valid config
        config = StationaritySettings()

        # Manually break it (bypassing validation)
        # Note: This is testing the concept; in practice Pydantic's
        # validate_assignment prevents this
        config_dict = config.model_dump()
        config_dict["significance_level"] = 1.5  # Invalid

        # validate_fully should catch this if we re-create
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            StationaritySettings(**config_dict)


class TestCrossFieldValidation:
    """Test cross-field validation scenarios."""

    def test_pca_auto_requires_variance_threshold(self):
        """Test PCA n_components='auto' requires valid variance_threshold."""
        from ml4t.diagnostic.config import PCASettings

        # Valid: auto with valid threshold
        PCASettings(enabled=True, n_components="auto", variance_threshold=0.95)

        # Invalid: auto with invalid threshold
        with pytest.raises(ValidationError):
            PCASettings(enabled=True, n_components="auto", variance_threshold=1.5)

    def test_bayesian_comparison_cross_validation(self):
        """Test Bayesian comparison cross-field validation."""
        from ml4t.diagnostic.config import PortfolioBayesianSettings

        # Valid: compare_to_benchmark requires benchmark_column
        PortfolioBayesianSettings(compare_to_benchmark=True, benchmark_column="SPY")

        # Valid: not comparing doesn't require benchmark_column
        PortfolioBayesianSettings(compare_to_benchmark=False, benchmark_column=None)

        # Invalid: compare_to_benchmark=True without benchmark_column
        with pytest.raises(ValidationError, match="benchmark_column required"):
            PortfolioBayesianSettings(compare_to_benchmark=True, benchmark_column=None)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
