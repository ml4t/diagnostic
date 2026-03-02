"""Report generation module for ML4T Diagnostic results.

Provides flexible report generation in multiple formats:
- HTML: Rich, styled reports with tables and charts
- JSON: Machine-readable structured output
- Markdown: Human-readable documentation

Examples:
    >>> from ml4t.diagnostic.reporting import ReportFactory, ReportFormat
    >>> from ml4t.diagnostic.results import FeatureDiagnosticsResultSchema
    >>>
    >>> # Generate HTML report
    >>> html_report = ReportFactory.render(result, ReportFormat.HTML)
    >>>
    >>> # Generate JSON report
    >>> json_report = ReportFactory.render(result, ReportFormat.JSON, indent=4)
    >>>
    >>> # Generate Markdown report
    >>> md_report = ReportFactory.render(result, ReportFormat.MARKDOWN)
    >>>
    >>> # Save to file
    >>> generator = ReportFactory.create(ReportFormat.HTML)
    >>> html = generator.render(result)
    >>> generator.save(html, "report.html")
"""

from ml4t.diagnostic.reporting.base import ReportFactory, ReportFormat, ReportGenerator

# Import renderers to trigger registration
from ml4t.diagnostic.reporting.html_renderer import HTMLReportGenerator
from ml4t.diagnostic.reporting.json_renderer import JSONReportGenerator
from ml4t.diagnostic.reporting.markdown_renderer import MarkdownReportGenerator

__all__ = [
    # Factory and base
    "ReportFactory",
    "ReportFormat",
    "ReportGenerator",
    # Renderers
    "HTMLReportGenerator",
    "JSONReportGenerator",
    "MarkdownReportGenerator",
]
