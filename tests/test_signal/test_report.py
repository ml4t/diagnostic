"""Tests for signal report generation.

Tests _report.py: generate_html, _generate_text_html.
"""

import pytest

from ml4t.diagnostic.signal._report import _generate_text_html, generate_html
from ml4t.diagnostic.signal.result import SignalResult

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def minimal_result():
    """Minimal SignalResult for testing."""
    return SignalResult(
        ic={"1D": 0.05},
        ic_std={"1D": 0.02},
        ic_t_stat={"1D": 2.5},
        ic_p_value={"1D": 0.01},
        ic_ir={"1D": 2.5},
        ic_positive_pct={"1D": 60.0},
        ic_series={"1D": [0.04, 0.05, 0.06, 0.03, 0.07]},
        quantile_returns={"1D": {1: 0.001, 2: 0.002, 3: 0.003, 4: 0.004, 5: 0.005}},
        spread={"1D": 0.004},
        spread_t_stat={"1D": 3.0},
        spread_p_value={"1D": 0.005},
        monotonicity={"1D": 1.0},
        n_assets=100,
        n_dates=20,
        date_range=("2024-01-01", "2024-01-20"),
        periods=(1,),
        quantiles=5,
    )


@pytest.fixture
def multi_period_result():
    """SignalResult with multiple periods."""
    return SignalResult(
        ic={"1D": 0.05, "5D": 0.08, "21D": 0.10},
        ic_std={"1D": 0.02, "5D": 0.03, "21D": 0.04},
        ic_t_stat={"1D": 2.5, "5D": 2.7, "21D": 2.5},
        ic_p_value={"1D": 0.01, "5D": 0.008, "21D": 0.012},
        ic_ir={"1D": 2.5, "5D": 2.67, "21D": 2.5},
        ic_positive_pct={"1D": 60.0, "5D": 65.0, "21D": 70.0},
        ic_series={
            "1D": [0.04, 0.05, 0.06],
            "5D": [0.07, 0.08, 0.09],
            "21D": [0.09, 0.10, 0.11],
        },
        quantile_returns={
            "1D": {1: 0.001, 2: 0.002, 3: 0.003, 4: 0.004, 5: 0.005},
            "5D": {1: 0.005, 2: 0.010, 3: 0.015, 4: 0.020, 5: 0.025},
            "21D": {1: 0.02, 2: 0.04, 3: 0.06, 4: 0.08, 5: 0.10},
        },
        spread={"1D": 0.004, "5D": 0.020, "21D": 0.080},
        spread_t_stat={"1D": 3.0, "5D": 4.0, "21D": 5.0},
        spread_p_value={"1D": 0.005, "5D": 0.001, "21D": 0.0001},
        monotonicity={"1D": 1.0, "5D": 1.0, "21D": 1.0},
        n_assets=100,
        n_dates=60,
        date_range=("2024-01-01", "2024-03-01"),
        periods=(1, 5, 21),
        quantiles=5,
    )


@pytest.fixture
def empty_ic_series_result():
    """SignalResult with empty IC series."""
    return SignalResult(
        ic={"1D": 0.05},
        ic_std={"1D": 0.02},
        ic_t_stat={"1D": 2.5},
        ic_p_value={"1D": 0.01},
        ic_series={"1D": []},  # Empty series
        quantile_returns={"1D": {}},  # Empty quantile returns
        spread={"1D": 0.004},
        spread_t_stat={"1D": 3.0},
        spread_p_value={"1D": 0.005},
        monotonicity={"1D": 1.0},
        n_assets=100,
        n_dates=20,
        date_range=("2024-01-01", "2024-01-20"),
        periods=(1,),
        quantiles=5,
    )


@pytest.fixture
def no_periods_result():
    """SignalResult with no periods."""
    return SignalResult(
        ic={},
        ic_std={},
        ic_t_stat={},
        ic_p_value={},
        n_assets=0,
        n_dates=0,
        date_range=("", ""),
        periods=(),
        quantiles=5,
    )


# =============================================================================
# Tests: generate_html (with Plotly)
# =============================================================================


class TestGenerateHtml:
    """Tests for generate_html function."""

    def test_creates_html_file(self, minimal_result, tmp_path):
        """Test that HTML file is created."""
        output_path = tmp_path / "report.html"
        generate_html(minimal_result, str(output_path))

        assert output_path.exists()
        content = output_path.read_text()
        assert "<html" in content.lower()

    def test_html_contains_plotly(self, minimal_result, tmp_path):
        """Test that HTML contains Plotly scripts."""
        output_path = tmp_path / "report.html"
        generate_html(minimal_result, str(output_path))

        content = output_path.read_text()
        # Plotly includes its JS library
        assert "plotly" in content.lower()

    def test_html_contains_title(self, minimal_result, tmp_path):
        """Test that HTML contains result summary in title."""
        output_path = tmp_path / "report.html"
        generate_html(minimal_result, str(output_path))

        content = output_path.read_text()
        assert "100" in content  # n_assets
        assert "20" in content  # n_dates

    def test_multi_period_report(self, multi_period_result, tmp_path):
        """Test report with multiple periods."""
        output_path = tmp_path / "report.html"
        generate_html(multi_period_result, str(output_path))

        content = output_path.read_text()
        assert output_path.exists()
        # Should contain period labels
        assert "1D" in content
        assert "5D" in content
        assert "21D" in content

    def test_empty_ic_series(self, empty_ic_series_result, tmp_path):
        """Test report handles empty IC series gracefully."""
        output_path = tmp_path / "report.html"
        # Should not raise
        generate_html(empty_ic_series_result, str(output_path))

        assert output_path.exists()

    def test_no_periods(self, no_periods_result, tmp_path):
        """Test report handles no periods gracefully."""
        output_path = tmp_path / "report.html"
        # Should not raise
        generate_html(no_periods_result, str(output_path))

        assert output_path.exists()


# =============================================================================
# Tests: _generate_text_html (fallback without Plotly)
# =============================================================================


class TestGenerateTextHtml:
    """Tests for _generate_text_html fallback function."""

    def test_creates_html_file(self, minimal_result, tmp_path):
        """Test that text HTML file is created."""
        output_path = tmp_path / "report.html"
        _generate_text_html(minimal_result, str(output_path))

        assert output_path.exists()

    def test_contains_doctype(self, minimal_result, tmp_path):
        """Test that output is valid HTML with DOCTYPE."""
        output_path = tmp_path / "report.html"
        _generate_text_html(minimal_result, str(output_path))

        content = output_path.read_text()
        assert "<!DOCTYPE html>" in content

    def test_contains_summary(self, minimal_result, tmp_path):
        """Test that output contains result summary."""
        output_path = tmp_path / "report.html"
        _generate_text_html(minimal_result, str(output_path))

        content = output_path.read_text()
        # Should contain summary output
        assert "Signal Analysis" in content
        assert "100 assets" in content
        assert "20 dates" in content

    def test_contains_ic_values(self, minimal_result, tmp_path):
        """Test that output contains IC values."""
        output_path = tmp_path / "report.html"
        _generate_text_html(minimal_result, str(output_path))

        content = output_path.read_text()
        assert "IC" in content
        assert "1D" in content

    def test_styled_output(self, minimal_result, tmp_path):
        """Test that output contains CSS styling."""
        output_path = tmp_path / "report.html"
        _generate_text_html(minimal_result, str(output_path))

        content = output_path.read_text()
        assert "<style>" in content
        assert "monospace" in content

    def test_multi_period(self, multi_period_result, tmp_path):
        """Test text report with multiple periods."""
        output_path = tmp_path / "report.html"
        _generate_text_html(multi_period_result, str(output_path))

        content = output_path.read_text()
        assert "1D" in content
        assert "5D" in content
        assert "21D" in content
