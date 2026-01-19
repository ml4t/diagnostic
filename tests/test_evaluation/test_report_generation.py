"""Tests for report generation module."""

import numpy as np
import pytest

from ml4t.diagnostic.evaluation import FeatureDiagnostics, FeatureDiagnosticsConfig
from ml4t.diagnostic.evaluation.report_generation import (
    generate_html_report,
    generate_multi_feature_html_report,
)


@pytest.fixture
def sample_diagnostic_result():
    """Create sample diagnostic result for testing."""
    # Generate simple white noise data
    np.random.seed(42)
    data = np.random.randn(500)

    # Run diagnostics
    config = FeatureDiagnosticsConfig(
        run_stationarity=True,
        run_autocorrelation=True,
        run_volatility=True,
        run_distribution=True,
    )
    diagnostics = FeatureDiagnostics(config)
    result = diagnostics.run_diagnostics(data, name="test_feature")

    return result


@pytest.fixture
def multiple_diagnostic_results():
    """Create multiple diagnostic results for multi-feature testing."""
    np.random.seed(42)
    results = []

    # White noise
    data1 = np.random.randn(500)
    config = FeatureDiagnosticsConfig()
    diagnostics = FeatureDiagnostics(config)
    result1 = diagnostics.run_diagnostics(data1, name="white_noise")
    results.append(result1)

    # Trending data
    data2 = np.cumsum(np.random.randn(500)) * 0.01
    result2 = diagnostics.run_diagnostics(data2, name="trending")
    results.append(result2)

    # Heavy-tailed data
    data3 = np.random.standard_t(df=3, size=500) * 0.02
    result3 = diagnostics.run_diagnostics(data3, name="heavy_tails")
    results.append(result3)

    return results


class TestHTMLReportGeneration:
    """Tests for HTML report generation."""

    def test_generate_basic_html_report(self, sample_diagnostic_result):
        """Test basic HTML report generation."""
        html = generate_html_report(sample_diagnostic_result, include_plots=False)

        # Check structure
        assert isinstance(html, str)
        assert len(html) > 0
        assert "<!DOCTYPE html>" in html
        assert "<html" in html
        assert "</html>" in html

        # Check content
        assert "test_feature" in html
        assert "Health Score" in html
        assert str(sample_diagnostic_result.n_obs) in html

    def test_html_report_with_custom_title(self, sample_diagnostic_result):
        """Test HTML report with custom title."""
        custom_title = "My Custom Diagnostic Report"
        html = generate_html_report(
            sample_diagnostic_result,
            include_plots=False,
            title=custom_title,
        )

        assert custom_title in html

    def test_html_report_includes_plotly_script(self, sample_diagnostic_result):
        """Test that HTML report includes Plotly CDN."""
        html = generate_html_report(sample_diagnostic_result, include_plots=True)

        assert "plotly" in html.lower()

    def test_html_report_includes_stationarity(self, sample_diagnostic_result):
        """Test that HTML includes stationarity results."""
        html = generate_html_report(sample_diagnostic_result, include_plots=False)

        assert "Stationarity" in html
        if sample_diagnostic_result.stationarity is not None:
            assert sample_diagnostic_result.stationarity.consensus in html

    def test_html_report_includes_summary_table(self, sample_diagnostic_result):
        """Test that HTML includes summary table."""
        html = generate_html_report(sample_diagnostic_result, include_plots=False)

        # Check for table elements
        assert "<table" in html
        assert "dataframe" in html

    def test_html_report_includes_recommendations(self, sample_diagnostic_result):
        """Test that HTML includes recommendations."""
        html = generate_html_report(sample_diagnostic_result, include_plots=False)

        assert "Recommendations" in html
        for rec in sample_diagnostic_result.recommendations:
            assert rec in html

    def test_html_report_includes_flags(self, sample_diagnostic_result):
        """Test that HTML includes flags if present."""
        html = generate_html_report(sample_diagnostic_result, include_plots=False)

        if sample_diagnostic_result.flags:
            assert "Flags" in html or "⚠️" in html
            for flag in sample_diagnostic_result.flags:
                assert flag in html

    def test_html_report_styling(self, sample_diagnostic_result):
        """Test that HTML includes CSS styling."""
        html = generate_html_report(sample_diagnostic_result, include_plots=False)

        assert "<style>" in html
        assert "</style>" in html
        assert "font-family" in html
        assert ".dataframe" in html


class TestMultiFeatureHTMLReport:
    """Tests for multi-feature HTML reports."""

    def test_generate_multi_feature_report(self, multiple_diagnostic_results):
        """Test basic multi-feature report generation."""
        html = generate_multi_feature_html_report(
            multiple_diagnostic_results,
            include_plots=False,
        )

        assert isinstance(html, str)
        assert "<!DOCTYPE html>" in html
        assert "Multi-Feature Diagnostic Report" in html

    def test_multi_feature_report_custom_title(self, multiple_diagnostic_results):
        """Test multi-feature report with custom title."""
        custom_title = "Portfolio Diagnostics"
        html = generate_multi_feature_html_report(
            multiple_diagnostic_results,
            include_plots=False,
            title=custom_title,
        )

        assert custom_title in html

    def test_multi_feature_report_includes_all_features(self, multiple_diagnostic_results):
        """Test that all features are included."""
        html = generate_multi_feature_html_report(
            multiple_diagnostic_results,
            include_plots=False,
        )

        for result in multiple_diagnostic_results:
            assert result.feature_name in html

    def test_multi_feature_report_includes_comparison_table(self, multiple_diagnostic_results):
        """Test that comparison table is included."""
        html = generate_multi_feature_html_report(
            multiple_diagnostic_results,
            include_plots=False,
        )

        # Should have comparison table
        assert "Comparison Table" in html
        assert "<table" in html
        assert "dataframe" in html

        # Should have all feature names in table
        for result in multiple_diagnostic_results:
            assert result.feature_name in html

    def test_multi_feature_report_empty_list_raises(self):
        """Test that empty results list raises error."""
        with pytest.raises(ValueError, match="results list cannot be empty"):
            generate_multi_feature_html_report([])

    def test_multi_feature_report_individual_sections(self, multiple_diagnostic_results):
        """Test that individual feature sections are included."""
        html = generate_multi_feature_html_report(
            multiple_diagnostic_results,
            include_plots=False,
        )

        # Each feature should have its own section
        for result in multiple_diagnostic_results:
            assert f"Feature: {result.feature_name}" in html

    def test_multi_feature_report_overview(self, multiple_diagnostic_results):
        """Test that overview section is included."""
        html = generate_multi_feature_html_report(
            multiple_diagnostic_results,
            include_plots=False,
        )

        assert "Overview" in html
        assert f"Features analyzed:</strong> {len(multiple_diagnostic_results)}" in html
        assert "Generated:" in html
