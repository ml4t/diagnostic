"""Tests for report rendering modules."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import polars as pl
import pytest
from pydantic import Field

from ml4t.diagnostic.reporting.base import (
    ReportFactory,
    ReportFormat,
    ReportGenerator,
)
from ml4t.diagnostic.results.base import BaseResult


# Create a concrete test result class
class SimpleTestResult(BaseResult):
    """Simple result class for testing."""

    analysis_type: str = Field(default="test_analysis")
    test_value: float = 0.5
    test_list: list[str] = Field(default_factory=lambda: ["a", "b", "c"])

    def summary(self) -> str:
        """Return summary."""
        return f"Test Result: value={self.test_value}"

    def get_dataframe(self, name: str | None = None) -> pl.DataFrame:
        """Return test DataFrame."""
        return pl.DataFrame(
            {
                "feature": ["f1", "f2", "f3"],
                "importance": [0.5, 0.3, 0.2],
            }
        )

    def list_available_dataframes(self) -> list[str]:
        """List available DataFrames."""
        return ["primary", "features"]


class NoDataFrameResult(BaseResult):
    """Result without DataFrame support."""

    analysis_type: str = Field(default="no_df_analysis")
    value: str = "test"

    def summary(self) -> str:
        """Return summary."""
        return "No DataFrame Result"


class TestReportFormat:
    """Tests for ReportFormat enum."""

    def test_html_format(self):
        """Test HTML format value."""
        assert ReportFormat.HTML.value == "html"

    def test_json_format(self):
        """Test JSON format value."""
        assert ReportFormat.JSON.value == "json"

    def test_markdown_format(self):
        """Test Markdown format value."""
        assert ReportFormat.MARKDOWN.value == "markdown"


class TestReportGenerator:
    """Tests for ReportGenerator base class."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_save_creates_file(self, temp_dir):
        """Test that save creates file."""
        # Need a concrete generator - use JSON since it's simple
        from ml4t.diagnostic.reporting.json_renderer import JSONReportGenerator

        generator = JSONReportGenerator()
        result = SimpleTestResult()
        content = generator.render(result)

        filepath = temp_dir / "report.json"
        generator.save(content, filepath)

        assert filepath.exists()
        assert filepath.read_text() == content

    def test_save_creates_parent_directories(self, temp_dir):
        """Test that save creates parent directories."""
        from ml4t.diagnostic.reporting.json_renderer import JSONReportGenerator

        generator = JSONReportGenerator()
        result = SimpleTestResult()
        content = generator.render(result)

        filepath = temp_dir / "nested" / "deep" / "report.json"
        generator.save(content, filepath)

        assert filepath.exists()


class TestJSONReportGenerator:
    """Tests for JSONReportGenerator."""

    @pytest.fixture
    def result(self):
        """Create test result."""
        return SimpleTestResult()

    def test_render_basic(self, result):
        """Test basic JSON rendering."""
        from ml4t.diagnostic.reporting.json_renderer import JSONReportGenerator

        generator = JSONReportGenerator()
        output = generator.render(result)

        assert isinstance(output, str)
        data = json.loads(output)
        assert data["analysis_type"] == "test_analysis"
        assert data["test_value"] == 0.5

    def test_render_with_indent(self, result):
        """Test JSON rendering with custom indent."""
        from ml4t.diagnostic.reporting.json_renderer import JSONReportGenerator

        generator = JSONReportGenerator()
        output = generator.render(result, indent=4)

        # Should have proper indentation
        lines = output.split("\n")
        assert len(lines) > 1  # Multi-line

    def test_render_exclude_none(self):
        """Test JSON rendering with exclude_none."""
        from ml4t.diagnostic.reporting.json_renderer import JSONReportGenerator

        result = SimpleTestResult()
        generator = JSONReportGenerator()

        output = generator.render(result, exclude_none=True)
        data = json.loads(output)

        # Should still contain required fields
        assert "analysis_type" in data


class TestHTMLReportGenerator:
    """Tests for HTMLReportGenerator."""

    @pytest.fixture
    def result(self):
        """Create test result."""
        return SimpleTestResult()

    def test_render_basic(self, result):
        """Test basic HTML rendering."""
        from ml4t.diagnostic.reporting.html_renderer import HTMLReportGenerator

        generator = HTMLReportGenerator()
        output = generator.render(result)

        assert isinstance(output, str)
        assert "<!DOCTYPE html>" in output
        assert "<html" in output
        assert "Test Analysis Report" in output

    def test_render_includes_metadata(self, result):
        """Test HTML includes metadata."""
        from ml4t.diagnostic.reporting.html_renderer import HTMLReportGenerator

        generator = HTMLReportGenerator()
        output = generator.render(result, include_metadata=True)

        assert "Metadata" in output
        assert "Analysis Type" in output
        assert "test_analysis" in output

    def test_render_excludes_metadata(self, result):
        """Test HTML can exclude metadata."""
        from ml4t.diagnostic.reporting.html_renderer import HTMLReportGenerator

        generator = HTMLReportGenerator()
        output = generator.render(result, include_metadata=False)

        assert '<div class="metadata">' not in output

    def test_render_includes_summary(self, result):
        """Test HTML includes summary."""
        from ml4t.diagnostic.reporting.html_renderer import HTMLReportGenerator

        generator = HTMLReportGenerator()
        output = generator.render(result)

        assert "Summary" in output
        assert "Test Result: value=0.5" in output

    def test_render_includes_dataframes(self, result):
        """Test HTML includes DataFrame tables."""
        from ml4t.diagnostic.reporting.html_renderer import HTMLReportGenerator

        generator = HTMLReportGenerator()
        output = generator.render(result, include_dataframes=True)

        assert "<table>" in output
        assert "feature" in output
        assert "importance" in output

    def test_render_excludes_dataframes(self, result):
        """Test HTML can exclude DataFrames."""
        from ml4t.diagnostic.reporting.html_renderer import HTMLReportGenerator

        generator = HTMLReportGenerator()
        output = generator.render(result, include_dataframes=False)

        # Table should not be present
        assert "<tbody>" not in output or "Primary" not in output

    def test_render_handles_no_dataframe_result(self):
        """Test HTML handles results without DataFrames."""
        from ml4t.diagnostic.reporting.html_renderer import HTMLReportGenerator

        result = NoDataFrameResult()
        generator = HTMLReportGenerator()

        # Should not raise even though get_dataframe raises
        output = generator.render(result, include_dataframes=True)
        assert "No Df Analysis Report" in output

    def test_render_limits_rows(self, result):
        """Test HTML limits DataFrame rows."""
        from ml4t.diagnostic.reporting.html_renderer import HTMLReportGenerator

        generator = HTMLReportGenerator()
        output = generator.render(result, include_dataframes=True, max_rows=2)

        assert "Showing first 2" in output

    def test_custom_template(self, result):
        """Test HTML with custom template."""
        from ml4t.diagnostic.reporting.html_renderer import HTMLReportGenerator

        custom_template = "<html><body><h1>{title}</h1>{summary}</body></html>"
        generator = HTMLReportGenerator()
        output = generator.render(result, custom_template=custom_template)

        assert "<html><body>" in output
        assert "Test Analysis Report" in output

    def test_escape_html(self):
        """Test HTML escaping."""
        from ml4t.diagnostic.reporting.html_renderer import HTMLReportGenerator

        generator = HTMLReportGenerator()
        escaped = generator._escape_html("<script>alert('xss')</script>")

        assert "<script>" not in escaped
        assert "&lt;script&gt;" in escaped

    def test_format_cell(self):
        """Test cell formatting."""
        from ml4t.diagnostic.reporting.html_renderer import HTMLReportGenerator

        generator = HTMLReportGenerator()

        assert generator._format_cell(None) == "<em>null</em>"
        assert generator._format_cell(1.23456) == "1.2346"
        assert generator._format_cell("test") == "test"

    def test_dataframe_to_html(self):
        """Test DataFrame to HTML table conversion."""
        from ml4t.diagnostic.reporting.html_renderer import HTMLReportGenerator

        generator = HTMLReportGenerator()
        df = pl.DataFrame(
            {
                "name": ["a", "b"],
                "value": [1.23456, 2.34567],
            }
        )

        table = generator._dataframe_to_html(df, title="Test Table", caption="Test caption")

        assert "<h2>Test Table</h2>" in table
        assert "<em>Test caption</em>" in table
        assert "<th>" in table
        assert "1.2346" in table


class TestMarkdownReportGenerator:
    """Tests for MarkdownReportGenerator."""

    @pytest.fixture
    def result(self):
        """Create test result."""
        return SimpleTestResult()

    def test_render_basic(self, result):
        """Test basic Markdown rendering."""
        from ml4t.diagnostic.reporting.markdown_renderer import MarkdownReportGenerator

        generator = MarkdownReportGenerator()
        output = generator.render(result)

        assert isinstance(output, str)
        assert "# " in output  # Has title
        assert "Test Analysis Report" in output  # Title from analysis_type

    def test_render_includes_summary(self, result):
        """Test Markdown includes summary."""
        from ml4t.diagnostic.reporting.markdown_renderer import MarkdownReportGenerator

        generator = MarkdownReportGenerator()
        output = generator.render(result)

        assert "Summary" in output
        assert "Test Result: value=0.5" in output

    def test_render_includes_metadata(self, result):
        """Test Markdown includes metadata."""
        from ml4t.diagnostic.reporting.markdown_renderer import MarkdownReportGenerator

        generator = MarkdownReportGenerator()
        output = generator.render(result, include_metadata=True)

        assert "Metadata" in output
        assert "Analysis Type" in output

    def test_render_excludes_metadata(self, result):
        """Test Markdown can exclude metadata."""
        from ml4t.diagnostic.reporting.markdown_renderer import MarkdownReportGenerator

        generator = MarkdownReportGenerator()
        output = generator.render(result, include_metadata=False)

        assert "Metadata" not in output

    def test_render_includes_dataframes(self, result):
        """Test Markdown includes DataFrame tables."""
        from ml4t.diagnostic.reporting.markdown_renderer import MarkdownReportGenerator

        generator = MarkdownReportGenerator()
        output = generator.render(result, include_dataframes=True)

        assert "Data Tables" in output
        assert "| feature" in output  # Table header

    def test_render_excludes_dataframes(self, result):
        """Test Markdown can exclude DataFrames."""
        from ml4t.diagnostic.reporting.markdown_renderer import MarkdownReportGenerator

        generator = MarkdownReportGenerator()
        output = generator.render(result, include_dataframes=False)

        assert "Data Tables" not in output

    def test_render_handles_no_dataframe_result(self):
        """Test Markdown handles results without DataFrames."""
        from ml4t.diagnostic.reporting.markdown_renderer import MarkdownReportGenerator

        result = NoDataFrameResult()
        generator = MarkdownReportGenerator()

        # Should not raise even though get_dataframe raises
        output = generator.render(result, include_dataframes=True)
        assert "No Df Analysis Report" in output

    def test_dataframe_to_markdown(self):
        """Test DataFrame to Markdown table conversion."""
        from ml4t.diagnostic.reporting.markdown_renderer import MarkdownReportGenerator

        generator = MarkdownReportGenerator()
        df = pl.DataFrame(
            {
                "name": ["a", "b"],
                "value": [1.23456, 2.34567],
            }
        )

        table = generator._dataframe_to_markdown(df)

        assert "| name |" in table
        assert "| --- |" in table
        assert "1.2346" in table  # 4 decimal places


class TestReportFactory:
    """Tests for ReportFactory."""

    def test_create_json_generator(self):
        """Test creating JSON generator."""
        generator = ReportFactory.create(ReportFormat.JSON)
        assert generator is not None

    def test_create_html_generator(self):
        """Test creating HTML generator."""
        generator = ReportFactory.create(ReportFormat.HTML)
        assert generator is not None

    def test_create_markdown_generator(self):
        """Test creating Markdown generator."""
        generator = ReportFactory.create(ReportFormat.MARKDOWN)
        assert generator is not None

    def test_create_invalid_format_raises(self):
        """Test creating generator for unregistered format raises."""
        # Clear registry temporarily
        original = ReportFactory._generators.copy()
        ReportFactory._generators = {}

        try:
            with pytest.raises(ValueError) as exc_info:
                ReportFactory.create(ReportFormat.JSON)
            assert "No generator registered" in str(exc_info.value)
        finally:
            ReportFactory._generators = original

    def test_render_convenience_method(self):
        """Test render convenience method."""
        result = SimpleTestResult()
        output = ReportFactory.render(result, ReportFormat.JSON)

        data = json.loads(output)
        assert data["analysis_type"] == "test_analysis"

    def test_available_formats(self):
        """Test listing available formats."""
        formats = ReportFactory.available_formats()
        assert isinstance(formats, list)
        assert len(formats) >= 2  # At least JSON and Markdown

    def test_register_custom_generator(self):
        """Test registering custom generator."""

        class CustomGenerator(ReportGenerator):
            def render(self, result, **options):
                return "CUSTOM"

        # Use a different format to avoid conflicts
        len(ReportFactory._generators)

        # Register doesn't add new format, just replaces
        ReportFactory.register(ReportFormat.JSON, CustomGenerator)

        generator = ReportFactory.create(ReportFormat.JSON)
        assert isinstance(generator, CustomGenerator)

        # Restore original
        from ml4t.diagnostic.reporting.json_renderer import JSONReportGenerator

        ReportFactory.register(ReportFormat.JSON, JSONReportGenerator)
