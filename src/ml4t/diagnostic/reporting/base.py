"""Base report generation functionality."""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Any

from ml4t.diagnostic.results.base import BaseResult


class ReportFormat(str, Enum):
    """Supported report formats."""

    HTML = "html"
    JSON = "json"
    MARKDOWN = "markdown"


class ReportGenerator(ABC):
    """Base class for report generators.

    Provides abstract interface for rendering evaluation results
    into different formats (HTML, JSON, Markdown).

    Subclasses must implement render() method.

    Examples:
        >>> generator = HTMLReportGenerator()
        >>> report = generator.render(result)
        >>> generator.save(report, "output.html")
    """

    @abstractmethod
    def render(self, result: BaseResult, **options: Any) -> str:
        """Render result to string format.

        Args:
            result: Evaluation result to render
            **options: Format-specific rendering options

        Returns:
            Rendered report as string
        """
        ...

    def save(self, content: str, filepath: str | Path) -> None:
        """Save rendered report to file.

        Args:
            content: Rendered report content
            filepath: Output file path
        """
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")


class ReportFactory:
    """Factory for creating report generators.

    Provides convenient access to all report generators via format enum.

    Examples:
        >>> generator = ReportFactory.create(ReportFormat.HTML)
        >>> report = generator.render(result)

        >>> # Or use convenience method
        >>> report = ReportFactory.render(result, ReportFormat.MARKDOWN)
    """

    _generators: dict[ReportFormat, type[ReportGenerator]] = {}

    @classmethod
    def register(cls, format: ReportFormat, generator_class: type[ReportGenerator]) -> None:
        """Register a report generator for a format.

        Args:
            format: Report format enum
            generator_class: Generator class to register
        """
        cls._generators[format] = generator_class

    @classmethod
    def create(cls, format: ReportFormat) -> ReportGenerator:
        """Create report generator for format.

        Args:
            format: Desired report format

        Returns:
            Report generator instance

        Raises:
            ValueError: If format not registered
        """
        if format not in cls._generators:
            raise ValueError(
                f"No generator registered for format: {format}. Available: {list(cls._generators.keys())}"
            )
        return cls._generators[format]()

    @classmethod
    def render(cls, result: BaseResult, format: ReportFormat, **options: Any) -> str:
        """Convenience method to create generator and render in one call.

        Args:
            result: Evaluation result to render
            format: Desired report format
            **options: Format-specific rendering options

        Returns:
            Rendered report as string

        Examples:
            >>> report = ReportFactory.render(result, ReportFormat.HTML)
            >>> report = ReportFactory.render(result, ReportFormat.JSON, indent=4)
        """
        generator = cls.create(format)
        return generator.render(result, **options)

    @classmethod
    def available_formats(cls) -> list[ReportFormat]:
        """List all registered report formats.

        Returns:
            List of available formats
        """
        return list(cls._generators.keys())
