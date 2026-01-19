"""Markdown report renderer."""

from __future__ import annotations

from typing import Any

from ml4t.diagnostic.reporting.base import ReportFormat, ReportGenerator
from ml4t.diagnostic.results.base import BaseResult


class MarkdownReportGenerator(ReportGenerator):
    """Generate Markdown reports from evaluation results.

    Creates human-readable Markdown documentation from results.

    Examples:
        >>> generator = MarkdownReportGenerator()
        >>> report = generator.render(result, include_metadata=True)
        >>> generator.save(report, "report.md")
    """

    def render(self, result: BaseResult, **options: Any) -> str:
        """Render result to Markdown format.

        Args:
            result: Evaluation result to render
            **options: Markdown rendering options
                - include_metadata: Include metadata section (default: True)
                - include_dataframes: Include DataFrame tables (default: True)
                - max_rows: Max rows to show in DataFrame tables (default: 10)

        Returns:
            Markdown string
        """
        include_metadata = options.get("include_metadata", True)
        include_dataframes = options.get("include_dataframes", True)
        max_rows = options.get("max_rows", 10)

        sections = []

        # Title
        analysis_type = result.analysis_type.replace("_", " ").title()
        sections.append(f"# {analysis_type} Report\n")

        # Metadata section
        if include_metadata:
            sections.append("## Metadata\n")
            sections.append(f"- **Analysis Type**: {result.analysis_type}")
            sections.append(f"- **Created**: {result.created_at}")
            sections.append(f"- **ML4T Diagnostic Version**: {result.version}\n")

        # Summary section (all results have summary())
        sections.append("## Summary\n")
        sections.append(f"```\n{result.summary()}\n```\n")

        # DataFrame sections (if available)
        if include_dataframes:
            try:
                available_dfs = result.list_available_dataframes()
                if available_dfs:
                    sections.append("## Data Tables\n")
                    for df_name in available_dfs:
                        df = result.get_dataframe(df_name)
                        sections.append(f"### {df_name.replace('_', ' ').title()}\n")

                        # Limit rows if needed
                        if len(df) > max_rows:
                            display_df = df.head(max_rows)
                            sections.append(f"*Showing first {max_rows} of {len(df)} rows*\n")
                        else:
                            display_df = df

                        # Convert to markdown table (basic implementation)
                        sections.append(self._dataframe_to_markdown(display_df))
                        sections.append("")
            except (NotImplementedError, ValueError):
                # Some results may not have DataFrames
                pass

        return "\n".join(sections)

    def _dataframe_to_markdown(self, df) -> str:
        """Convert Polars DataFrame to Markdown table.

        Args:
            df: Polars DataFrame

        Returns:
            Markdown table string
        """

        # Get column names
        cols = df.columns

        # Header
        header = "| " + " | ".join(cols) + " |"
        separator = "| " + " | ".join(["---"] * len(cols)) + " |"

        # Rows
        rows = []
        for row in df.iter_rows():
            formatted = []
            for val in row:
                # Format values
                if isinstance(val, float):
                    formatted.append(f"{val:.4f}")
                else:
                    formatted.append(str(val))
            rows.append("| " + " | ".join(formatted) + " |")

        return "\n".join([header, separator] + rows)


# Register with factory
from ml4t.diagnostic.reporting.base import ReportFactory  # noqa: E402

ReportFactory.register(ReportFormat.MARKDOWN, MarkdownReportGenerator)
