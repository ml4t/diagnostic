"""Week 4 tests for BarrierAnalysis - Integration and End-to-End workflow.

Tests cover:
- End-to-end tear sheet with visualizations
- HTML export with embedded figures
- Figure generation integration
- Theme customization
- Performance with larger datasets
"""

from __future__ import annotations

import tempfile
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from ml4t.diagnostic.config.barrier_config import AnalysisSettings, BarrierConfig
from ml4t.diagnostic.evaluation.barrier_analysis import BarrierAnalysis

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_dates() -> list[date]:
    """Generate 100 trading dates."""
    start = date(2020, 1, 1)
    return [start + timedelta(days=i) for i in range(100)]


@pytest.fixture
def sample_assets() -> list[str]:
    """Generate 10 assets."""
    return [f"ASSET_{i:02d}" for i in range(10)]


@pytest.fixture
def signal_data(sample_dates: list[date], sample_assets: list[str]) -> pl.DataFrame:
    """Create synthetic signal data with predictive signal."""
    np.random.seed(42)
    rows = []
    for d in sample_dates:
        for asset in sample_assets:
            signal = np.random.uniform(-1, 1)
            rows.append({"date": d, "asset": asset, "signal": signal})
    return pl.DataFrame(rows)


@pytest.fixture
def barrier_labels(sample_dates: list[date], sample_assets: list[str]) -> pl.DataFrame:
    """Create synthetic barrier labels correlated with signal."""
    np.random.seed(42)
    rows = []
    for d in sample_dates:
        for asset in sample_assets:
            signal = np.random.uniform(-1, 1)

            # Probability of TP increases with signal
            p_tp = 0.3 + 0.4 * (signal + 1) / 2
            p_sl = 0.3 - 0.2 * (signal + 1) / 2
            p_timeout = 1 - p_tp - p_sl

            outcome = np.random.choice([1, -1, 0], p=[p_tp, p_sl, p_timeout])

            if outcome == 1:
                ret = np.random.uniform(0.01, 0.03)
            elif outcome == -1:
                ret = np.random.uniform(-0.02, -0.01)
            else:
                ret = np.random.uniform(-0.005, 0.005)

            bars = np.random.randint(1, 20)

            rows.append(
                {
                    "date": d,
                    "asset": asset,
                    "label": outcome,
                    "label_return": ret,
                    "label_bars": bars,
                }
            )
    return pl.DataFrame(rows)


@pytest.fixture
def analysis(signal_data: pl.DataFrame, barrier_labels: pl.DataFrame) -> BarrierAnalysis:
    """Create BarrierAnalysis instance."""
    config = BarrierConfig(analysis=AnalysisSettings(n_quantiles=10), signal_name="test_signal")
    return BarrierAnalysis(signal_data, barrier_labels, config=config)


# =============================================================================
# Integration Tests - Tear Sheet with Figures
# =============================================================================


class TestTearSheetWithFigures:
    """Tests for tear sheet with integrated visualizations."""

    def test_tear_sheet_includes_figures(self, analysis: BarrierAnalysis) -> None:
        """Test that tear sheet includes generated figures."""
        tear_sheet = analysis.create_tear_sheet(include_figures=True)

        assert tear_sheet.figures is not None
        assert len(tear_sheet.figures) > 0

    def test_tear_sheet_has_all_expected_figures(self, analysis: BarrierAnalysis) -> None:
        """Test that all expected figures are generated."""
        tear_sheet = analysis.create_tear_sheet(include_figures=True)

        expected_figures = [
            "hit_rate_heatmap",
            "profit_factor_bar",
            "precision_recall_curve",
            "time_to_target_comparison",
        ]

        for fig_name in expected_figures:
            assert fig_name in tear_sheet.figures, f"Missing figure: {fig_name}"

    def test_figures_are_valid_json(self, analysis: BarrierAnalysis) -> None:
        """Test that figures are valid JSON strings."""
        import json

        tear_sheet = analysis.create_tear_sheet(include_figures=True)

        for name, fig_json in tear_sheet.figures.items():
            # Should be valid JSON
            try:
                data = json.loads(fig_json)
                assert "data" in data, f"Figure {name} missing 'data' key"
                assert "layout" in data, f"Figure {name} missing 'layout' key"
            except json.JSONDecodeError as e:
                pytest.fail(f"Figure {name} is not valid JSON: {e}")

    def test_figures_can_be_reconstructed(self, analysis: BarrierAnalysis) -> None:
        """Test that figures can be loaded back as Plotly figures."""
        import plotly.io as pio

        tear_sheet = analysis.create_tear_sheet(include_figures=True)

        for name, fig_json in tear_sheet.figures.items():
            fig = pio.from_json(fig_json)
            assert fig is not None
            assert len(fig.data) > 0, f"Figure {name} has no data traces"

    def test_tear_sheet_without_figures(self, analysis: BarrierAnalysis) -> None:
        """Test tear sheet creation with figures disabled."""
        tear_sheet = analysis.create_tear_sheet(include_figures=False)

        # Figures dict should be empty
        assert tear_sheet.figures == {}

    def test_tear_sheet_with_theme(self, analysis: BarrierAnalysis) -> None:
        """Test tear sheet with custom theme."""
        tear_sheet = analysis.create_tear_sheet(include_figures=True, theme="dark")

        assert len(tear_sheet.figures) > 0

        # Verify dark theme is applied (check background color in layout)
        import json

        for fig_json in tear_sheet.figures.values():
            data = json.loads(fig_json)
            layout = data.get("layout", {})
            paper_bgcolor = layout.get("paper_bgcolor", "").lower()
            # Dark theme uses dark backgrounds
            assert paper_bgcolor in ["#1e1e1e", "#2d2d2d", ""], (
                f"Unexpected bgcolor: {paper_bgcolor}"
            )
            break  # Check just one figure


# =============================================================================
# Integration Tests - HTML Export
# =============================================================================


class TestHTMLExport:
    """Tests for HTML export with embedded figures."""

    def test_save_html_creates_file(self, analysis: BarrierAnalysis) -> None:
        """Test that save_html creates a valid HTML file."""
        tear_sheet = analysis.create_tear_sheet(include_figures=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "barrier_report.html"
            result_path = tear_sheet.save_html(path)

            assert result_path.exists()
            assert result_path == path

    def test_html_contains_figures(self, analysis: BarrierAnalysis) -> None:
        """Test that HTML contains embedded Plotly figures."""
        tear_sheet = analysis.create_tear_sheet(include_figures=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "barrier_report.html"
            tear_sheet.save_html(path)

            content = path.read_text()

            # Should contain figure divs
            assert "plot-container" in content
            # Should contain figure titles
            assert "Hit Rate Heatmap" in content
            assert "Profit Factor Bar" in content
            assert "Precision Recall Curve" in content

    def test_html_contains_metadata(self, analysis: BarrierAnalysis) -> None:
        """Test that HTML contains tear sheet metadata."""
        tear_sheet = analysis.create_tear_sheet(include_figures=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "barrier_report.html"
            tear_sheet.save_html(path)

            content = path.read_text()

            # Should contain metadata
            assert "test_signal" in content  # signal name
            assert "Assets" in content
            assert "Dates" in content
            assert "Observations" in content

    def test_html_is_valid_structure(self, analysis: BarrierAnalysis) -> None:
        """Test that HTML has valid structure."""
        tear_sheet = analysis.create_tear_sheet(include_figures=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "barrier_report.html"
            tear_sheet.save_html(path)

            content = path.read_text()

            # Should have proper HTML structure
            assert "<!DOCTYPE html>" in content
            assert "<html>" in content
            assert "</html>" in content
            assert "<head>" in content
            assert "<body>" in content

    def test_html_includes_plotly_js(self, analysis: BarrierAnalysis) -> None:
        """Test that HTML includes Plotly.js reference."""
        tear_sheet = analysis.create_tear_sheet(include_figures=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "barrier_report.html"
            tear_sheet.save_html(path, include_plotlyjs="cdn")

            content = path.read_text()

            # Should include Plotly.js CDN reference
            assert "plotly" in content.lower()


# =============================================================================
# Integration Tests - JSON Export
# =============================================================================


class TestJSONExport:
    """Tests for JSON export."""

    def test_save_json_creates_file(self, analysis: BarrierAnalysis) -> None:
        """Test that save_json creates a valid JSON file."""
        tear_sheet = analysis.create_tear_sheet(include_figures=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "barrier_report.json"
            result_path = tear_sheet.save_json(path)

            assert result_path.exists()
            assert result_path == path

    def test_json_contains_all_results(self, analysis: BarrierAnalysis) -> None:
        """Test that JSON contains all analysis results."""
        import json

        tear_sheet = analysis.create_tear_sheet(include_figures=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "barrier_report.json"
            tear_sheet.save_json(path)

            with open(path) as f:
                data = json.load(f)

            # Should contain all result keys
            assert "hit_rate_result" in data
            assert "profit_factor_result" in data
            assert "precision_recall_result" in data
            assert "time_to_target_result" in data
            assert "figures" in data

    def test_json_exclude_figures(self, analysis: BarrierAnalysis) -> None:
        """Test that JSON can exclude figures for smaller file size."""
        import json

        tear_sheet = analysis.create_tear_sheet(include_figures=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "barrier_report.json"
            tear_sheet.save_json(path, exclude_figures=True)

            with open(path) as f:
                data = json.load(f)

            # Should not contain figures
            assert "figures" not in data


# =============================================================================
# Integration Tests - Without label_bars Column
# =============================================================================


class TestWithoutLabelBars:
    """Tests for tear sheet without label_bars column."""

    @pytest.fixture
    def barrier_labels_no_bars(
        self, sample_dates: list[date], sample_assets: list[str]
    ) -> pl.DataFrame:
        """Create barrier labels without label_bars column."""
        np.random.seed(42)
        rows = []
        for d in sample_dates:
            for asset in sample_assets:
                signal = np.random.uniform(-1, 1)
                p_tp = 0.3 + 0.4 * (signal + 1) / 2
                p_sl = 0.3 - 0.2 * (signal + 1) / 2
                p_timeout = 1 - p_tp - p_sl

                outcome = np.random.choice([1, -1, 0], p=[p_tp, p_sl, p_timeout])

                if outcome == 1:
                    ret = np.random.uniform(0.01, 0.03)
                elif outcome == -1:
                    ret = np.random.uniform(-0.02, -0.01)
                else:
                    ret = np.random.uniform(-0.005, 0.005)

                rows.append(
                    {
                        "date": d,
                        "asset": asset,
                        "label": outcome,
                        "label_return": ret,
                        # No label_bars column
                    }
                )
        return pl.DataFrame(rows)

    def test_tear_sheet_without_time_to_target(
        self,
        signal_data: pl.DataFrame,
        barrier_labels_no_bars: pl.DataFrame,
    ) -> None:
        """Test tear sheet creation without time-to-target analysis."""
        config = BarrierConfig(analysis=AnalysisSettings(n_quantiles=10))
        analysis = BarrierAnalysis(signal_data, barrier_labels_no_bars, config=config)

        tear_sheet = analysis.create_tear_sheet(include_time_to_target=True)

        # Time-to-target should be None (gracefully handled)
        assert tear_sheet.time_to_target_result is None

        # Other results should be present
        assert tear_sheet.hit_rate_result is not None
        assert tear_sheet.profit_factor_result is not None
        assert tear_sheet.precision_recall_result is not None

    def test_figures_without_time_to_target(
        self,
        signal_data: pl.DataFrame,
        barrier_labels_no_bars: pl.DataFrame,
    ) -> None:
        """Test that figures work without time-to-target."""
        config = BarrierConfig(analysis=AnalysisSettings(n_quantiles=10))
        analysis = BarrierAnalysis(signal_data, barrier_labels_no_bars, config=config)

        tear_sheet = analysis.create_tear_sheet(include_figures=True)

        # Should have 3 figures (no time-to-target)
        assert len(tear_sheet.figures) == 3
        assert "time_to_target_comparison" not in tear_sheet.figures


# =============================================================================
# Integration Tests - Full Workflow
# =============================================================================


class TestFullWorkflow:
    """Tests for the complete end-to-end workflow."""

    def test_complete_workflow(
        self,
        signal_data: pl.DataFrame,
        barrier_labels: pl.DataFrame,
    ) -> None:
        """Test complete workflow from data to HTML report."""
        # 1. Create config
        config = BarrierConfig(
            analysis=AnalysisSettings(n_quantiles=10, significance_level=0.05),
            signal_name="momentum_signal",
        )

        # 2. Create analysis
        analysis = BarrierAnalysis(signal_data, barrier_labels, config=config)

        # 3. Verify properties
        assert analysis.n_observations == 1000
        assert analysis.n_assets == 10
        assert analysis.n_dates == 100

        # 4. Create tear sheet with figures
        tear_sheet = analysis.create_tear_sheet(
            include_figures=True,
            include_time_to_target=True,
            theme="default",
        )

        # 5. Verify tear sheet contents
        assert tear_sheet.hit_rate_result is not None
        assert tear_sheet.profit_factor_result is not None
        assert tear_sheet.precision_recall_result is not None
        assert tear_sheet.time_to_target_result is not None
        assert len(tear_sheet.figures) == 4

        # 6. Export to HTML
        with tempfile.TemporaryDirectory() as tmpdir:
            html_path = Path(tmpdir) / "report.html"
            result = tear_sheet.save_html(html_path)

            assert result.exists()
            content = result.read_text()
            assert len(content) > 10000  # Should be a substantial file

        # 7. Export to JSON
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "report.json"
            result = tear_sheet.save_json(json_path)

            assert result.exists()

        # 8. Verify summary
        summary = tear_sheet.summary()
        assert "momentum_signal" in summary
        assert "Hit Rate" in summary
        assert "Profit Factor" in summary

    def test_workflow_with_all_themes(
        self,
        signal_data: pl.DataFrame,
        barrier_labels: pl.DataFrame,
    ) -> None:
        """Test workflow works with all available themes."""
        config = BarrierConfig(analysis=AnalysisSettings(n_quantiles=5))
        analysis = BarrierAnalysis(signal_data, barrier_labels, config=config)

        themes = ["default", "dark", "print", "presentation"]

        for theme in themes:
            tear_sheet = analysis.create_tear_sheet(
                include_figures=True,
                theme=theme,
            )

            assert len(tear_sheet.figures) > 0, f"No figures generated for theme: {theme}"
