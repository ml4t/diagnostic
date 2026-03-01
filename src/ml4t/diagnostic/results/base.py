"""Base result class with common functionality for all evaluation results."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import polars as pl
from pydantic import BaseModel, ConfigDict, Field


class BaseResult(BaseModel):
    """Base class for all evaluation results.

    Provides common functionality:
    - Metadata (timestamp, version, analysis type)
    - JSON export via model_dump_json()
    - Dict export via to_dict()
    - DataFrame conversion via get_dataframe()
    - Human-readable summary via summary()

    All result schemas inherit from this class to ensure consistent behavior.

    Examples:
        >>> result = FeatureDiagnosticsResultSchema(...)
        >>> json_str = result.model_dump_json(indent=2)
        >>> df = result.get_dataframe()
        >>> available = result.list_available_dataframes()
        >>> print(result.summary())
    """

    model_config = ConfigDict(
        extra="forbid",  # Catch typos
        validate_assignment=True,  # Validate on mutation
        arbitrary_types_allowed=True,  # Allow Polars types if needed
        use_enum_values=True,  # Serialize enums as values
    )

    # Metadata fields - present in all results
    created_at: str = Field(
        default_factory=lambda: datetime.now(UTC).isoformat(),
        description="ISO timestamp of result creation (UTC)",
    )
    analysis_type: str = Field(
        ...,
        description="Type of analysis performed (e.g., 'feature_diagnostics')",
    )
    version: str = Field(
        default="2.0.0",
        description="ML4T Diagnostic version used to generate results",
    )

    def to_dict(self, *, exclude_none: bool = False) -> dict[str, Any]:
        """Export to Python dictionary.

        Args:
            exclude_none: Exclude fields with None values

        Returns:
            Dictionary representation of result
        """
        return self.model_dump(exclude_none=exclude_none, mode="python")

    def to_json_string(self, *, indent: int | None = 2) -> str:
        """Export to JSON string.

        Args:
            indent: Indentation level (None for compact)

        Returns:
            JSON string representation
        """
        return self.model_dump_json(indent=indent)

    def get_dataframe(self, name: str | None = None) -> pl.DataFrame:
        """Get results as Polars DataFrame.

        Provides programmatic access to underlying data for QEngine storage
        and further analysis. Subclasses should override to provide specific
        DataFrame views.

        Args:
            name: Optional DataFrame name to retrieve specific view.
                  If None, returns primary/default DataFrame.
                  Use list_available_dataframes() to see available names.

        Returns:
            Polars DataFrame with results

        Raises:
            NotImplementedError: If not implemented by subclass
            ValueError: If requested DataFrame name not available

        Examples:
            >>> result = FeatureDiagnosticsResultSchema(...)
            >>> df = result.get_dataframe()  # Primary DataFrame
            >>> df = result.get_dataframe("stationarity")  # Specific view
            >>> available = result.list_available_dataframes()
            >>> for name in available:
            ...     df = result.get_dataframe(name)
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement get_dataframe()")

    def list_available_dataframes(self) -> list[str]:
        """List available DataFrame views for this result.

        Returns names that can be passed to get_dataframe() to retrieve
        specific data views. Useful for discovery and QEngine integration.

        Returns:
            List of available DataFrame names

        Examples:
            >>> result.list_available_dataframes()
            ['primary', 'stationarity', 'autocorrelation', 'volatility']
        """
        # Default implementation - subclasses should override
        return ["primary"]

    def get_dataframe_schema(self, name: str | None = None) -> dict[str, str]:
        """Get schema information for a DataFrame.

        Returns column names and types for a DataFrame without loading data.
        Useful for QEngine to understand data structure before retrieval.

        Args:
            name: DataFrame name (None for primary)

        Returns:
            Dictionary mapping column names to Polars dtype strings

        Examples:
            >>> result.get_dataframe_schema("stationarity")
            {
                'feature': 'String',
                'adf_pvalue': 'Float64',
                'is_stationary': 'Boolean'
            }
        """
        # Get actual DataFrame and extract schema
        df = self.get_dataframe(name)
        return {col: str(dtype) for col, dtype in zip(df.columns, df.dtypes, strict=False)}

    def summary(self) -> str:
        """Get human-readable summary of results.

        This method should be overridden by subclasses to provide
        meaningful summaries of their specific data.

        Returns:
            Formatted summary string

        Raises:
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement summary()")

    def interpret(self) -> list[str]:
        """Get human-readable interpretation of results.

        Returns actionable insights and recommendations based on
        the analysis results. Default implementation returns empty list.

        Returns:
            List of interpretation strings with insights and recommendations

        Examples:
            >>> result.interpret()
            ['Strategy is statistically significant (DSR=98.2% > 95%)',
             'Recommendation: Strategy shows robust performance']
        """
        # Default implementation - subclasses can override
        return []

    def __repr__(self) -> str:
        """Concise representation showing type and timestamp."""
        return f"{self.__class__.__name__}(analysis_type={self.analysis_type!r}, created_at={self.created_at!r})"
