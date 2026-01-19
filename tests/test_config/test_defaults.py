"""Test that all config classes have sensible defaults and can be instantiated.

This test suite verifies:
1. All config classes can be instantiated with no arguments
2. Default configs produce reasonable analysis settings
3. Preset methods work correctly and differ from each other
4. Default values are appropriate for common use cases
"""

import pytest


class TestBaseConfigDefaults:
    """Test BaseConfig and common infrastructure defaults."""

    def test_runtime_config_default(self):
        """Test RuntimeConfig has sensible defaults."""
        from ml4t.diagnostic.config import RuntimeConfig

        config = RuntimeConfig()
        assert config.n_jobs == -1  # Use all cores by default
        assert config.cache_enabled is True  # Caching on by default
        assert config.verbose is False  # Quiet by default
        assert config.cache_dir.exists()  # Cache dir should be created

    def test_statistical_test_config_default(self):
        """Test StatisticalTestConfig has sensible defaults."""
        from ml4t.diagnostic.config.base import StatisticalTestConfig

        config = StatisticalTestConfig()
        assert config.enabled is True  # Tests enabled by default
        assert config.significance_level == 0.05  # Standard alpha


class TestStationarityDefaults:
    """Test stationarity settings defaults."""

    def test_stationarity_settings_default(self):
        """Test StationaritySettings defaults."""
        from ml4t.diagnostic.config import StationaritySettings

        config = StationaritySettings()
        assert config.adf_enabled is True
        assert config.kpss_enabled is True
        assert config.pp_enabled is False  # PP is optional (similar to ADF)
        assert config.significance_level == 0.05
        assert config.max_lag == "auto"


class TestACFDefaults:
    """Test ACF settings defaults."""

    def test_acf_settings_default(self):
        """Test ACFSettings defaults."""
        from ml4t.diagnostic.config import ACFSettings

        config = ACFSettings()
        assert config.enabled is True
        assert config.n_lags == 40
        assert config.compute_pacf is True
        assert config.use_fft is True


class TestVolatilityDefaults:
    """Test volatility settings defaults."""

    def test_volatility_settings_default(self):
        """Test VolatilitySettings defaults."""
        from ml4t.diagnostic.config import VolatilitySettings

        config = VolatilitySettings()
        assert config.enabled is True
        assert config.window_sizes == [21]  # Standard monthly window
        assert config.detect_clustering is True
        assert config.compute_rolling_vol is True


class TestDistributionDefaults:
    """Test distribution settings defaults."""

    def test_distribution_settings_default(self):
        """Test DistributionSettings defaults."""
        from ml4t.diagnostic.config import DistributionSettings

        config = DistributionSettings()
        assert config.enabled is True
        assert config.test_normality is True
        assert config.compute_moments is True
        assert config.detect_outliers is False  # Opt-in (expensive)


class TestCorrelationDefaults:
    """Test correlation settings defaults."""

    def test_correlation_settings_default(self):
        """Test CorrelationSettings defaults."""
        from ml4t.diagnostic.config import CorrelationSettings
        from ml4t.diagnostic.config.validation import CorrelationMethod

        config = CorrelationSettings()
        assert config.enabled is True
        assert CorrelationMethod.PEARSON in config.methods
        assert config.compute_pairwise is True
        assert config.min_periods == 30
        assert config.lag_correlations is False  # Expensive, opt-in


class TestPCADefaults:
    """Test PCA settings defaults."""

    def test_pca_settings_default(self):
        """Test PCASettings defaults."""
        from ml4t.diagnostic.config import PCASettings

        config = PCASettings()
        assert config.enabled is False  # Opt-in (expensive)
        assert config.n_components == "auto"
        assert config.variance_threshold == 0.95
        assert config.standardize is True


class TestClusteringDefaults:
    """Test clustering settings defaults."""

    def test_clustering_settings_default(self):
        """Test ClusteringSettings defaults."""
        from ml4t.diagnostic.config import ClusteringSettings

        config = ClusteringSettings()
        assert config.enabled is False  # Opt-in


class TestRedundancyDefaults:
    """Test redundancy settings defaults."""

    def test_redundancy_settings_default(self):
        """Test RedundancySettings defaults."""
        from ml4t.diagnostic.config import RedundancySettings

        config = RedundancySettings()
        assert config.enabled is True
        assert config.correlation_threshold == 0.95  # High threshold
        assert config.compute_vif is False  # Expensive, opt-in
        assert config.keep_strategy == "highest_ic"


class TestICDefaults:
    """Test IC settings defaults."""

    def test_ic_settings_default(self):
        """Test ICSettings defaults."""
        from ml4t.diagnostic.config import ICSettings
        from ml4t.diagnostic.config.validation import CorrelationMethod

        config = ICSettings()
        assert config.enabled is True
        assert config.method == CorrelationMethod.PEARSON
        assert config.lag_structure == [0, 1, 5]
        assert config.hac_adjustment is False  # Expensive, opt-in
        assert config.compute_t_stats is True


class TestBinaryClassificationDefaults:
    """Test binary classification settings defaults."""

    def test_binary_classification_settings_default(self):
        """Test BinaryClassificationSettings defaults."""
        from ml4t.diagnostic.config import BinaryClassificationSettings

        config = BinaryClassificationSettings()
        assert config.enabled is False  # Requires threshold selection
        assert config.thresholds == [0.0]
        assert "precision" in config.metrics


class TestThresholdAnalysisDefaults:
    """Test threshold analysis settings defaults."""

    def test_threshold_analysis_settings_default(self):
        """Test ThresholdAnalysisSettings defaults."""
        from ml4t.diagnostic.config import ThresholdAnalysisSettings

        config = ThresholdAnalysisSettings()
        assert config.enabled is False  # Expensive, opt-in
        assert config.sweep_range == (-2.0, 2.0)
        assert config.n_points == 50


class TestMLDiagnosticsDefaults:
    """Test ML diagnostics settings defaults."""

    def test_ml_diagnostics_settings_default(self):
        """Test MLDiagnosticsSettings defaults."""
        from ml4t.diagnostic.config import MLDiagnosticsSettings

        config = MLDiagnosticsSettings()
        assert config.enabled is True
        assert config.feature_importance is True
        assert config.shap_analysis is False  # Very expensive
        assert config.drift_detection is False


class TestDiagnosticConfigDefaults:
    """Test DiagnosticConfig top-level defaults."""

    def test_diagnostic_config_default(self):
        """Test DiagnosticConfig has all settings configured."""
        from ml4t.diagnostic.config import DiagnosticConfig

        config = DiagnosticConfig()
        assert config.stationarity is not None
        assert config.acf is not None
        assert config.volatility is not None
        assert config.distribution is not None
        assert config.correlation is not None
        assert config.pca is not None
        assert config.clustering is not None
        assert config.redundancy is not None
        assert config.ic is not None
        assert config.binary_classification is not None
        assert config.threshold_analysis is not None
        assert config.ml_diagnostics is not None
        assert config.export_recommendations is True
        assert config.return_dataframes is True
        assert config.n_jobs == -1

    def test_diagnostic_config_presets_differ(self):
        """Test that presets produce different configurations."""
        from ml4t.diagnostic.config import DiagnosticConfig

        quick = DiagnosticConfig.for_quick_analysis()
        research = DiagnosticConfig.for_research()
        production = DiagnosticConfig.for_production()

        # Quick should be faster
        assert not quick.volatility.detect_clustering
        assert research.volatility.detect_clustering

        # Research should have more analyses
        assert not quick.ic.hac_adjustment
        assert research.ic.hac_adjustment

        # Production should focus on drift
        assert production.ml_diagnostics.drift_detection
        assert not quick.ml_diagnostics.drift_detection


class TestPortfolioConfigDefaults:
    """Test PortfolioConfig defaults."""

    def test_portfolio_metrics_settings_default(self):
        """Test PortfolioMetricsSettings defaults."""
        from ml4t.diagnostic.config import PortfolioMetricsSettings
        from ml4t.diagnostic.config.validation import PortfolioMetric

        config = PortfolioMetricsSettings()
        assert PortfolioMetric.SHARPE in config.metrics
        assert PortfolioMetric.MAX_DRAWDOWN in config.metrics
        assert config.risk_free_rate == 0.0
        assert config.confidence_level == 0.95
        assert config.periods_per_year == 252  # Daily

    def test_portfolio_bayesian_settings_default(self):
        """Test PortfolioBayesianSettings defaults."""
        from ml4t.diagnostic.config import PortfolioBayesianSettings

        config = PortfolioBayesianSettings()
        assert config.enabled is False  # Opt-in
        assert config.n_samples == 10000
        assert config.credible_interval == 0.95

    def test_portfolio_aggregation_settings_default(self):
        """Test PortfolioAggregationSettings defaults."""
        from ml4t.diagnostic.config import PortfolioAggregationSettings
        from ml4t.diagnostic.config.validation import TimeFrequency

        config = PortfolioAggregationSettings()
        assert TimeFrequency.DAILY in config.frequencies
        assert config.compute_rolling is False
        assert config.align_to_calendar is True

    def test_portfolio_drawdown_settings_default(self):
        """Test PortfolioDrawdownSettings defaults."""
        from ml4t.diagnostic.config import PortfolioDrawdownSettings

        config = PortfolioDrawdownSettings()
        assert config.enabled is True
        assert config.compute_underwater_curve is True
        assert config.top_n_drawdowns == 5
        assert config.recovery_threshold == 1.0

    def test_portfolio_config_default(self):
        """Test PortfolioConfig has all components."""
        from ml4t.diagnostic.config import PortfolioConfig

        config = PortfolioConfig()
        assert config.metrics is not None
        assert config.bayesian is not None
        assert config.aggregation is not None
        assert config.drawdown is not None

    def test_portfolio_config_presets_differ(self):
        """Test that presets produce different configurations."""
        from ml4t.diagnostic.config import PortfolioConfig

        quick = PortfolioConfig.for_quick_analysis()
        research = PortfolioConfig.for_research()
        production = PortfolioConfig.for_production()

        # Research should have more metrics
        assert len(research.metrics.metrics) > len(quick.metrics.metrics)

        # Research should have Bayesian enabled
        assert research.bayesian.enabled
        assert not quick.bayesian.enabled

        # Production should have rolling windows
        assert production.aggregation.compute_rolling
        assert not quick.aggregation.compute_rolling


class TestStatisticalConfigDefaults:
    """Test StatisticalConfig defaults."""

    def test_psr_settings_default(self):
        """Test PSRSettings defaults."""
        from ml4t.diagnostic.config import PSRSettings

        config = PSRSettings()
        assert config.enabled is True
        assert config.confidence_level == 0.95
        assert config.target_sharpe == 0.0
        assert config.adjustment_factor == "auto"

    def test_mintrl_settings_default(self):
        """Test MinTRLSettings defaults."""
        from ml4t.diagnostic.config import MinTRLSettings

        config = MinTRLSettings()
        assert config.enabled is True
        assert config.confidence_level == 0.95
        assert config.target_sharpe == 0.0

    def test_dsr_settings_default(self):
        """Test DSRSettings defaults."""
        from ml4t.diagnostic.config import DSRSettings

        config = DSRSettings()
        assert config.enabled is True
        assert config.n_trials == 100
        assert config.prob_zero_sharpe == 0.5
        assert config.variance_inflation == 1.0

    def test_fdr_settings_default(self):
        """Test FDRSettings defaults."""
        from ml4t.diagnostic.config import FDRSettings
        from ml4t.diagnostic.config.validation import FDRMethod

        config = FDRSettings()
        assert config.enabled is True
        assert config.alpha == 0.05
        assert config.method == FDRMethod.BENJAMINI_HOCHBERG
        assert config.independent_tests is False

    def test_statistical_config_default(self):
        """Test StatisticalConfig has all components."""
        from ml4t.diagnostic.config import StatisticalConfig

        config = StatisticalConfig()
        assert config.psr is not None
        assert config.mintrl is not None
        assert config.dsr is not None
        assert config.fdr is not None
        assert all(
            [config.psr.enabled, config.mintrl.enabled, config.dsr.enabled, config.fdr.enabled]
        )

    def test_statistical_config_presets_differ(self):
        """Test that presets produce different configurations."""
        from ml4t.diagnostic.config import StatisticalConfig

        quick = StatisticalConfig.for_quick_check()
        research = StatisticalConfig.for_research()
        publication = StatisticalConfig.for_publication()

        # Quick should disable some analyses
        assert not quick.mintrl.enabled
        assert research.mintrl.enabled

        # Publication should be most conservative
        assert publication.dsr.n_trials > research.dsr.n_trials
        assert publication.fdr.alpha < research.fdr.alpha


class TestReportConfigDefaults:
    """Test ReportConfig defaults."""

    def test_output_format_config_default(self):
        """Test OutputFormatConfig defaults."""
        from ml4t.diagnostic.config import OutputFormatConfig
        from ml4t.diagnostic.config.validation import ReportFormat

        config = OutputFormatConfig()
        assert ReportFormat.HTML in config.formats
        assert ReportFormat.JSON in config.formats
        assert config.output_dir.exists()
        assert config.overwrite_existing is True

    def test_html_config_default(self):
        """Test HTMLConfig defaults."""
        from ml4t.diagnostic.config import HTMLConfig
        from ml4t.diagnostic.config.validation import ReportTemplate, ReportTheme

        config = HTMLConfig()
        assert config.template == ReportTemplate.FULL
        assert config.theme == ReportTheme.LIGHT
        assert config.interactive_plots is True
        assert config.include_toc is True
        assert config.include_summary is True

    def test_visualization_config_default(self):
        """Test VisualizationConfig defaults."""
        from ml4t.diagnostic.config import VisualizationConfig

        config = VisualizationConfig()
        assert config.plot_dpi == 100
        assert config.plot_width == 800
        assert config.plot_height == 600
        assert config.correlation_heatmap is True
        assert config.save_plots is False

    def test_json_config_default(self):
        """Test JSONConfig defaults."""
        from ml4t.diagnostic.config import JSONConfig
        from ml4t.diagnostic.config.validation import DataFrameExportFormat

        config = JSONConfig()
        assert config.pretty_print is True
        assert config.include_metadata is True
        assert config.export_dataframes == DataFrameExportFormat.RECORDS
        assert config.include_raw_data is False

    def test_report_config_default(self):
        """Test ReportConfig has all components."""
        from ml4t.diagnostic.config import ReportConfig

        config = ReportConfig()
        assert config.output_format is not None
        assert config.html is not None
        assert config.visualization is not None
        assert config.json_config is not None

    def test_report_config_presets_differ(self):
        """Test that presets produce different configurations."""
        from ml4t.diagnostic.config import ReportConfig

        quick = ReportConfig.for_quick_report()
        publication = ReportConfig.for_publication()
        programmatic = ReportConfig.for_programmatic_access()

        # Publication should have higher resolution
        assert publication.visualization.plot_dpi > quick.visualization.plot_dpi

        # Programmatic should be JSON only
        assert len(programmatic.output_format.formats) == 1

        # Quick should have fewer plots
        assert not quick.visualization.time_series_plots
        assert publication.visualization.time_series_plots


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
