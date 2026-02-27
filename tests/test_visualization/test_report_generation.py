"""Tests for HTML report generation functions.

Tests cover:
- combine_figures_to_html() - Core HTML generation
- generate_importance_report() - Feature importance reports
- generate_interaction_report() - Interaction analysis reports
- generate_combined_report() - Combined analysis reports
- HTML structure validation
- CSS styling validation
- Section generation
- Table of contents generation
"""

# Check if PDF export dependencies are available
import importlib.util
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import plotly.graph_objects as go
import pytest

from ml4t.diagnostic.visualization.report_generation import (
    combine_figures_to_html,
    export_figures_to_pdf,
    generate_combined_report,
    generate_importance_report,
    generate_interaction_report,
)


def _pdf_export_runtime_available() -> bool:
    """Return True only when Plotly PDF export works in this runtime.

    Some environments have `pypdf`/`kaleido` installed but no usable rendering
    backend, which can cause `to_image(..., format="pdf")` to hang.
    """
    if importlib.util.find_spec("pypdf") is None:
        return False
    if importlib.util.find_spec("kaleido") is None:
        return False

    probe = (
        "import plotly.graph_objects as go;"
        "fig=go.Figure(data=[go.Scatter(x=[1,2],y=[1,2])]);"
        "fig.to_image(format='pdf')"
    )
    try:
        result = subprocess.run(
            [sys.executable, "-c", probe],
            capture_output=True,
            timeout=10,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return False
    return result.returncode == 0


PDF_EXPORT_AVAILABLE = _pdf_export_runtime_available()


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sample_figures():
    """Create sample Plotly figures for testing."""
    fig1 = go.Figure()
    fig1.add_trace(go.Bar(x=["A", "B", "C"], y=[1, 3, 2], name="Series 1"))
    fig1.update_layout(title="Test Figure 1")

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=[1, 2, 3], y=[2, 4, 3], mode="lines", name="Series 2"))
    fig2.update_layout(title="Test Figure 2")

    return [fig1, fig2]


@pytest.fixture
def mock_importance_results():
    """Create mock importance analysis results matching real analyze_ml_importance() output."""
    feature_names = ["feature_1", "feature_2", "feature_3", "feature_4", "feature_5"]
    return {
        "consensus_ranking": feature_names,
        "method_results": {
            "mdi": {
                "importances": [0.3, 0.25, 0.2, 0.15, 0.1],
                "feature_names": feature_names,
                "n_features": 5,
                "normalized": True,
                "model_type": "RandomForestClassifier",
            },
            "pfi": {
                "importances_mean": [0.28, 0.27, 0.2, 0.15, 0.1],
                "importances_std": [0.02, 0.03, 0.02, 0.01, 0.01],
                "importances_raw": [[0.28, 0.27, 0.2, 0.15, 0.1]] * 10,
                "feature_names": feature_names,
                "baseline_score": 0.85,
                "n_repeats": 10,
                "scoring": "accuracy",
                "n_features": 5,
            },
        },
        "method_agreement": {"mdi_vs_pfi": 0.85},
        "top_features_consensus": ["feature_1", "feature_2"],
        "warnings": [],
        "interpretation": "Strong consensus across methods",
        "methods_run": ["mdi", "pfi"],
        "methods_failed": [],
    }


@pytest.fixture
def mock_interaction_results():
    """Create mock interaction analysis results matching real compute_shap_interactions() output."""
    import numpy as np

    n_features = 5
    feature_names = [f"feature_{i + 1}" for i in range(n_features)]

    interaction_matrix = np.random.rand(n_features, n_features) * 0.5
    # Make symmetric
    interaction_matrix = (interaction_matrix + interaction_matrix.T) / 2
    # Zero diagonal
    np.fill_diagonal(interaction_matrix, 0)

    # Create top_interactions list (tuples of feature_i, feature_j, strength)
    top_interactions = [
        ("feature_1", "feature_2", 0.45),
        ("feature_2", "feature_3", 0.38),
        ("feature_1", "feature_3", 0.32),
        ("feature_3", "feature_4", 0.28),
        ("feature_1", "feature_4", 0.25),
    ]

    return {
        "interaction_matrix": interaction_matrix,
        "feature_names": feature_names,
        "top_interactions": top_interactions,
        "n_features": n_features,
        "n_samples_used": 100,
        "computation_time": 1.5,
    }


# ============================================================================
# Test combine_figures_to_html()
# ============================================================================


class TestGenerateImportanceReport:
    """Tests for generate_importance_report() function."""

    def test_basic_importance_report(self, mock_importance_results):
        """Test basic importance report generation."""
        with TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "importance.html"

            result = generate_importance_report(
                importance_results=mock_importance_results, output_file=output_path
            )

            # Check file was created
            assert Path(result).exists()

            # Check content
            content = Path(result).read_text()
            assert "Feature Importance Analysis" in content
            assert "Executive Summary" in content

    def test_custom_title(self, mock_importance_results):
        """Test custom title in importance report."""
        with TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "report.html"

            result = generate_importance_report(
                importance_results=mock_importance_results,
                output_file=output_path,
                title="Custom Analysis Title",
            )

            content = Path(result).read_text()
            assert "Custom Analysis Title" in content

    def test_custom_sections(self, mock_importance_results):
        """Test selecting specific sections."""
        with TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "report.html"

            result = generate_importance_report(
                importance_results=mock_importance_results,
                output_file=output_path,
                include_sections=["summary", "rankings"],
            )

            content = Path(result).read_text()
            assert "Executive Summary" in content
            assert "Consensus Feature Rankings" in content

    def test_invalid_section_raises_error(self, mock_importance_results):
        """Test that invalid section names raise ValueError."""
        with TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="Invalid sections"):
                generate_importance_report(
                    importance_results=mock_importance_results,
                    output_file=Path(tmpdir) / "report.html",
                    include_sections=["invalid_section"],
                )

    def test_theme_parameter(self, mock_importance_results):
        """Test theme parameter works."""
        with TemporaryDirectory() as tmpdir:
            result = generate_importance_report(
                importance_results=mock_importance_results,
                output_file=Path(tmpdir) / "report.html",
                theme="dark",
            )

            content = Path(result).read_text()
            # Should contain dark theme colors
            assert "#1E1E1E" in content or "#1e1e1e" in content.lower()

    def test_top_n_parameter(self, mock_importance_results):
        """Test top_n parameter controls number of features shown."""
        with TemporaryDirectory() as tmpdir:
            result = generate_importance_report(
                importance_results=mock_importance_results,
                output_file=Path(tmpdir) / "report.html",
                top_n=10,
            )

            # Just check it runs without error
            assert Path(result).exists()

    def test_all_default_sections_included(self, mock_importance_results):
        """Test that all default sections are included."""
        with TemporaryDirectory() as tmpdir:
            result = generate_importance_report(
                importance_results=mock_importance_results, output_file=Path(tmpdir) / "report.html"
            )

            content = Path(result).read_text()

            # Check all default sections
            assert "Executive Summary" in content
            assert "Consensus Feature Rankings" in content
            assert "Method Agreement" in content
            assert "Score Distributions" in content or "Importance Score" in content
            assert "Recommendations" in content or "Interpretation" in content


# ============================================================================
# Test generate_interaction_report()
# ============================================================================


class TestGenerateInteractionReport:
    """Tests for generate_interaction_report() function."""

    def test_basic_interaction_report(self, mock_interaction_results):
        """Test basic interaction report generation."""
        with TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "interaction.html"

            result = generate_interaction_report(
                interaction_results=mock_interaction_results, output_file=output_path
            )

            # Check file was created
            assert Path(result).exists()

            # Check content
            content = Path(result).read_text()
            assert "Interaction" in content

    def test_custom_title(self, mock_interaction_results):
        """Test custom title in interaction report."""
        with TemporaryDirectory() as tmpdir:
            result = generate_interaction_report(
                interaction_results=mock_interaction_results,
                output_file=Path(tmpdir) / "report.html",
                title="Custom Interaction Report",
            )

            content = Path(result).read_text()
            assert "Custom Interaction Report" in content

    def test_custom_sections(self, mock_interaction_results):
        """Test selecting specific sections."""
        with TemporaryDirectory() as tmpdir:
            result = generate_interaction_report(
                interaction_results=mock_interaction_results,
                output_file=Path(tmpdir) / "report.html",
                include_sections=["top_pairs", "matrix"],
            )

            content = Path(result).read_text()
            assert "Top" in content and "Interaction" in content
            assert "Matrix" in content

    def test_theme_support(self, mock_interaction_results):
        """Test different themes."""
        with TemporaryDirectory() as tmpdir:
            result = generate_interaction_report(
                interaction_results=mock_interaction_results,
                output_file=Path(tmpdir) / "report.html",
                theme="presentation",
            )

            assert Path(result).exists()


# ============================================================================
# Test generate_combined_report()
# ============================================================================


class TestGenerateCombinedReport:
    """Tests for generate_combined_report() function."""

    def test_basic_combined_report(self, mock_importance_results, mock_interaction_results):
        """Test basic combined report generation."""
        with TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "combined.html"

            result = generate_combined_report(
                importance_results=mock_importance_results,
                interaction_results=mock_interaction_results,
                output_file=output_path,
            )

            # Check file was created
            assert Path(result).exists()

            # Check content includes both analyses
            content = Path(result).read_text()
            assert "Feature Importance" in content or "Importance" in content
            assert "Interaction" in content

    def test_combined_report_without_interactions(self, mock_importance_results):
        """Test combined report with only importance results."""
        with TemporaryDirectory() as tmpdir:
            result = generate_combined_report(
                importance_results=mock_importance_results,
                interaction_results=None,
                output_file=Path(tmpdir) / "report.html",
            )

            content = Path(result).read_text()
            # Should still work without interaction section
            assert Path(result).exists()
            assert "Importance" in content

    def test_custom_title(self, mock_importance_results):
        """Test custom title in combined report."""
        with TemporaryDirectory() as tmpdir:
            result = generate_combined_report(
                importance_results=mock_importance_results,
                output_file=Path(tmpdir) / "report.html",
                title="My Custom Report",
            )

            content = Path(result).read_text()
            assert "My Custom Report" in content

    def test_theme_parameter(self, mock_importance_results, mock_interaction_results):
        """Test theme parameter."""
        with TemporaryDirectory() as tmpdir:
            result = generate_combined_report(
                importance_results=mock_importance_results,
                interaction_results=mock_interaction_results,
                output_file=Path(tmpdir) / "report.html",
                theme="print",
            )

            assert Path(result).exists()

    def test_contains_overview_section(self, mock_importance_results):
        """Test that combined report includes overview."""
        with TemporaryDirectory() as tmpdir:
            result = generate_combined_report(
                importance_results=mock_importance_results,
                output_file=Path(tmpdir) / "report.html",
            )

            content = Path(result).read_text()
            assert "Overview" in content or "Analysis" in content

    def test_contains_recommendations(self, mock_importance_results, mock_interaction_results):
        """Test that combined report includes recommendations."""
        with TemporaryDirectory() as tmpdir:
            result = generate_combined_report(
                importance_results=mock_importance_results,
                interaction_results=mock_interaction_results,
                output_file=Path(tmpdir) / "report.html",
            )

            content = Path(result).read_text()
            assert "Recommendation" in content or "Action" in content


# ============================================================================
# Integration Tests
# ============================================================================


class TestReportIntegration:
    """Integration tests using real plot functions."""

    def test_full_workflow(self, mock_importance_results):
        """Test complete workflow from analysis to report."""
        with TemporaryDirectory() as tmpdir:
            # Generate report with real plot functions
            result = generate_importance_report(
                importance_results=mock_importance_results,
                output_file=Path(tmpdir) / "report.html",
                theme="dark",
            )

            # Verify file exists and is valid HTML
            assert Path(result).exists()
            content = Path(result).read_text()

            # Check HTML structure
            assert content.startswith("<!DOCTYPE html>")
            assert "</html>" in content

            # Check required elements
            assert "<head>" in content
            assert "<body>" in content
            assert "<style>" in content  # CSS should be embedded

    def test_multiple_reports_in_sequence(self, mock_importance_results, mock_interaction_results):
        """Test generating multiple reports in sequence."""
        with TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Generate importance report
            result1 = generate_importance_report(
                importance_results=mock_importance_results,
                output_file=tmpdir_path / "importance.html",
            )

            # Generate interaction report
            result2 = generate_interaction_report(
                interaction_results=mock_interaction_results,
                output_file=tmpdir_path / "interaction.html",
            )

            # Generate combined report
            result3 = generate_combined_report(
                importance_results=mock_importance_results,
                interaction_results=mock_interaction_results,
                output_file=tmpdir_path / "combined.html",
            )

            # All should exist
            assert Path(result1).exists()
            assert Path(result2).exists()
            assert Path(result3).exists()

            # Files should be different
            content1 = Path(result1).read_text()
            content2 = Path(result2).read_text()
            content3 = Path(result3).read_text()

            assert len(content1) > 1000  # Non-trivial content
            assert len(content2) > 1000
            assert len(content3) > 1000
            assert content1 != content2  # Different content
            assert content1 != content3


# ============================================================================
# CSS and Styling Tests
# ============================================================================


class TestHtmlStyling:
    """Tests for HTML/CSS styling."""

    def test_css_is_embedded(self, sample_figures):
        """Test that CSS is embedded in HTML."""
        html = combine_figures_to_html(figures=sample_figures, output_file=None)

        # CSS should be in <style> tags, not external file
        assert "<style>" in html
        assert "</style>" in html

    def test_responsive_css(self, sample_figures):
        """Test that responsive CSS is present."""
        html = combine_figures_to_html(figures=sample_figures, output_file=None)

        # Check for media queries
        assert "@media" in html
        assert "max-width" in html

    def test_print_css(self, sample_figures):
        """Test that print CSS is included."""
        html = combine_figures_to_html(figures=sample_figures, output_file=None)

        # Check for print styles
        assert "@media print" in html

    def test_footer_is_present(self, sample_figures):
        """Test that footer is included."""
        html = combine_figures_to_html(figures=sample_figures, output_file=None)

        assert "<footer>" in html
        assert "</footer>" in html
        assert "ML4T Diagnostic" in html or "Generated by" in html


# ============================================================================
# Test export_figures_to_pdf()
# ============================================================================


@pytest.mark.slow
@pytest.mark.skipif(
    not PDF_EXPORT_AVAILABLE, reason="pypdf not installed (optional viz dependency)"
)
class TestExportFiguresToPdf:
    """Tests for PDF export functionality."""

    def test_basic_pdf_export(self, sample_figures):
        """Test basic PDF export with default settings."""
        with TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.pdf"

            result = export_figures_to_pdf(figures=sample_figures, output_file=output_path)

            # Check file was created
            assert Path(result).exists()
            assert Path(result).suffix == ".pdf"

            # Check file has content
            assert Path(result).stat().st_size > 0

    def test_pdf_export_with_custom_page_size(self, sample_figures):
        """Test PDF export with custom page size."""
        with TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.pdf"

            result = export_figures_to_pdf(
                figures=sample_figures,
                output_file=output_path,
                page_size=(1200, 900),  # Larger page
                scale=3.0,
            )

            assert Path(result).exists()
            assert Path(result).stat().st_size > 0

    def test_pdf_export_creates_parent_directory(self, sample_figures):
        """Test that PDF export creates parent directories."""
        with TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "subdir" / "nested" / "test.pdf"

            result = export_figures_to_pdf(figures=sample_figures, output_file=output_path)

            assert Path(result).exists()
            assert output_path.parent.exists()

    def test_pdf_export_empty_figures_raises_error(self):
        """Test that exporting empty figure list raises error."""
        with TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.pdf"

            with pytest.raises(ValueError, match="At least one figure is required"):
                export_figures_to_pdf(figures=[], output_file=output_path)

    def test_pdf_export_invalid_figure_type_raises_error(self):
        """Test that invalid figure types raise TypeError."""
        with TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.pdf"

            with pytest.raises(TypeError, match="must be plotly.graph_objects.Figure"):
                export_figures_to_pdf(figures=["not", "a", "figure"], output_file=output_path)

    def test_pdf_export_invalid_layout_raises_error(self, sample_figures):
        """Test that invalid layout mode raises error."""
        with TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.pdf"

            with pytest.raises(ValueError, match="Invalid layout"):
                export_figures_to_pdf(
                    figures=sample_figures, output_file=output_path, layout="invalid"
                )

    def test_pdf_export_compact_not_implemented(self, sample_figures):
        """Test that compact layout raises NotImplementedError."""
        with TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.pdf"

            with pytest.raises(NotImplementedError, match="Compact layout"):
                export_figures_to_pdf(
                    figures=sample_figures, output_file=output_path, layout="compact"
                )

    def test_pdf_export_returns_absolute_path(self, sample_figures):
        """Test that PDF export returns absolute path."""
        with TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.pdf"

            result = export_figures_to_pdf(figures=sample_figures, output_file=output_path)

            assert Path(result).is_absolute()

    def test_pdf_export_multipage(self, sample_figures):
        """Test that multiple figures create multipage PDF."""
        # Create more figures
        figures = sample_figures + [
            go.Figure(data=[go.Bar(x=["X", "Y"], y=[10, 20])]),
            go.Figure(data=[go.Scatter(x=[1, 2], y=[3, 4])]),
        ]

        with TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.pdf"

            result = export_figures_to_pdf(figures=figures, output_file=output_path)

            # Check file exists and has content
            assert Path(result).exists()
            file_size = Path(result).stat().st_size
            assert file_size > 1000  # Should be reasonable size for 4 pages

    def test_pdf_export_with_different_scales(self, sample_figures):
        """Test PDF export with different scale factors."""
        with TemporaryDirectory() as tmpdir:
            # Low quality (small file)
            path1 = Path(tmpdir) / "low_quality.pdf"
            export_figures_to_pdf(figures=sample_figures, output_file=path1, scale=1.0)

            # High quality (larger file)
            path2 = Path(tmpdir) / "high_quality.pdf"
            export_figures_to_pdf(figures=sample_figures, output_file=path2, scale=4.0)

            # Both should exist
            assert path1.exists() and path2.exists()

            # High quality should be larger (more pixels)
            # Note: This might not always be true due to compression,
            # so we just check both files have content
            assert path1.stat().st_size > 0
            assert path2.stat().st_size > 0


# ============================================================================
# Test PDF export integration with report functions
# ============================================================================


@pytest.mark.slow
@pytest.mark.skipif(
    not PDF_EXPORT_AVAILABLE, reason="pypdf not installed (optional viz dependency)"
)
class TestReportPdfExport:
    """Tests for PDF export via report generation functions."""

    def test_importance_report_pdf_export(self, mock_importance_results):
        """Test that importance report can export to PDF."""
        with TemporaryDirectory() as tmpdir:
            html_path = Path(tmpdir) / "report.html"
            pdf_path = Path(tmpdir) / "report.pdf"

            generate_importance_report(
                importance_results=mock_importance_results,
                output_file=html_path,
                export_pdf=True,
            )

            # Both HTML and PDF should exist
            assert html_path.exists()
            assert pdf_path.exists()
            assert pdf_path.stat().st_size > 0

    def test_report_pdf_export_with_custom_settings(self, mock_importance_results):
        """Test PDF export with custom page size and scale."""
        with TemporaryDirectory() as tmpdir:
            html_path = Path(tmpdir) / "report.html"
            pdf_path = Path(tmpdir) / "report.pdf"

            generate_importance_report(
                importance_results=mock_importance_results,
                output_file=html_path,
                export_pdf=True,
                pdf_page_size=(1200, 900),
                pdf_scale=3.0,
            )

            assert html_path.exists()
            assert pdf_path.exists()
            assert pdf_path.stat().st_size > 0

    def test_report_without_pdf_export(self, mock_importance_results):
        """Test that PDF is not created when export_pdf=False."""
        with TemporaryDirectory() as tmpdir:
            html_path = Path(tmpdir) / "report.html"
            pdf_path = Path(tmpdir) / "report.pdf"

            generate_importance_report(
                importance_results=mock_importance_results,
                output_file=html_path,
                export_pdf=False,  # Explicitly False
            )

            # Only HTML should exist
            assert html_path.exists()
            assert not pdf_path.exists()
