"""JSON report renderer."""

from __future__ import annotations

from typing import Any

from ml4t.diagnostic.reporting.base import ReportFormat, ReportGenerator
from ml4t.diagnostic.results.base import BaseResult


class JSONReportGenerator(ReportGenerator):
    """Generate JSON reports from evaluation results.

    Leverages BaseResult's built-in JSON serialization.

    Examples:
        >>> generator = JSONReportGenerator()
        >>> report = generator.render(result, indent=2)
        >>> generator.save(report, "report.json")
    """

    def render(self, result: BaseResult, **options: Any) -> str:
        """Render result to JSON format.

        Args:
            result: Evaluation result to render
            **options: JSON rendering options
                - indent: Indentation level (default: 2)
                - exclude_none: Exclude None values (default: False)

        Returns:
            JSON string
        """
        indent = options.get("indent", 2)
        exclude_none = options.get("exclude_none", False)

        # Use BaseResult's built-in JSON export
        if exclude_none:
            # Get dict first, then convert to JSON
            data = result.to_dict(exclude_none=True)
            import json

            return json.dumps(data, indent=indent)
        else:
            return result.to_json_string(indent=indent)


# Register with factory
from ml4t.diagnostic.reporting.base import ReportFactory  # noqa: E402

ReportFactory.register(ReportFormat.JSON, JSONReportGenerator)
