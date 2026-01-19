"""ML4T Data integration contract for data quality validation.

This module defines the API contract between ML4T Data and ML4T Diagnostic for
data quality assessment. ML4T Data can use these contracts to report data quality
to ML4T Diagnostic for validation before feature engineering.

Example workflow:
    >>> from ml4t.data import DataManager
    >>> from ml4t.diagnostic.integration import DataQualityReport, DataAnomaly
    >>>
    >>> # 1. Load data with quality report
    >>> dm = DataManager(storage_config)
    >>> data, quality = dm.load_with_quality("AAPL", start="2020-01-01")
    >>>
    >>> # 2. Check quality before proceeding
    >>> if not quality.is_acceptable():
    ...     print(quality.summary())
    ...     raise DataQualityError(quality.recommendations)
    >>>
    >>> # 3. Proceed with feature engineering
    >>> features = compute_features(data)
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class AnomalyType(str, Enum):
    """Types of data anomalies that can be detected.

    These anomaly types align with common data quality issues in financial data:

    - MISSING_DATA: Gaps in the data (e.g., missing trading days)
    - STALE_DATA: Same value repeated (stuck price feed)
    - PRICE_SPIKE: Abnormal price movement (e.g., >10 std devs)
    - NEGATIVE_PRICE: Invalid negative price (should be positive)
    - ZERO_VOLUME: No trading activity (suspicious for liquid assets)
    - OHLC_VIOLATION: High < Low or similar OHLC logic errors
    - TIMESTAMP_GAP: Unexpected gap in timestamps
    - DUPLICATE_TIMESTAMP: Same timestamp appears multiple times
    - OUTLIER: Statistical outlier (not necessarily error)
    """

    MISSING_DATA = "missing_data"
    STALE_DATA = "stale_data"
    PRICE_SPIKE = "price_spike"
    NEGATIVE_PRICE = "negative_price"
    ZERO_VOLUME = "zero_volume"
    OHLC_VIOLATION = "ohlc_violation"
    TIMESTAMP_GAP = "timestamp_gap"
    DUPLICATE_TIMESTAMP = "duplicate_timestamp"
    OUTLIER = "outlier"


class Severity(str, Enum):
    """Severity level of data anomalies.

    - INFO: Informational, no action needed
    - WARNING: Potential issue, review recommended
    - ERROR: Definite issue, correction needed
    - CRITICAL: Severe issue, data may be unusable
    """

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class DataAnomaly(BaseModel):
    """Record of a single data anomaly detected.

    Represents a specific data quality issue found during validation.
    Used to communicate detailed findings from ML4T Data to ML4T Diagnostic.

    Attributes:
        anomaly_type: Type of anomaly detected
        severity: Severity level
        timestamp: When the anomaly occurred
        symbol: Which asset (if multi-asset)
        description: Human-readable description
        value: The problematic value (if applicable)
        expected_range: Expected value range (if applicable)
        suggested_fix: Recommended correction

    Example:
        >>> anomaly = DataAnomaly(
        ...     anomaly_type=AnomalyType.PRICE_SPIKE,
        ...     severity=Severity.ERROR,
        ...     timestamp=datetime(2024, 1, 15, 10, 30),
        ...     symbol="AAPL",
        ...     description="Price moved 15 std devs in 1 minute",
        ...     value=999.99,
        ...     expected_range=(150.0, 200.0),
        ...     suggested_fix="Replace with interpolated value"
        ... )
    """

    anomaly_type: AnomalyType = Field(..., description="Type of anomaly")
    severity: Severity = Field(..., description="Severity level")
    timestamp: datetime = Field(..., description="When anomaly occurred")
    symbol: str | None = Field(None, description="Asset symbol (if applicable)")
    description: str = Field(..., description="Human-readable description")
    value: float | None = Field(None, description="Problematic value")
    expected_range: tuple[float, float] | None = Field(None, description="Expected value range")
    suggested_fix: str | None = Field(None, description="Recommended correction")


class DataQualityMetrics(BaseModel):
    """Quantitative metrics for data quality assessment.

    These metrics provide a numerical summary of data quality that can be
    used for automated quality gates.

    Attributes:
        completeness: Fraction of expected data points present [0.0, 1.0]
        timeliness: How up-to-date the data is (e.g., minutes since last update)
        accuracy_score: Estimated data accuracy based on validation checks [0.0, 1.0]
        consistency_score: How consistent the data is (no OHLC violations, etc.) [0.0, 1.0]
        n_records: Total number of records
        n_anomalies: Total anomalies detected
        n_critical: Number of critical severity anomalies
        n_error: Number of error severity anomalies
        n_warning: Number of warning severity anomalies

    Example:
        >>> metrics = DataQualityMetrics(
        ...     completeness=0.98,
        ...     timeliness=5.0,
        ...     accuracy_score=0.95,
        ...     consistency_score=1.0,
        ...     n_records=10000,
        ...     n_anomalies=12,
        ...     n_critical=0,
        ...     n_error=2,
        ...     n_warning=10
        ... )
    """

    completeness: float = Field(..., ge=0.0, le=1.0, description="Data completeness [0,1]")
    timeliness: float = Field(..., ge=0.0, description="Minutes since last update")
    accuracy_score: float = Field(..., ge=0.0, le=1.0, description="Accuracy score [0,1]")
    consistency_score: float = Field(..., ge=0.0, le=1.0, description="Consistency score [0,1]")
    n_records: int = Field(..., ge=0, description="Total records")
    n_anomalies: int = Field(..., ge=0, description="Total anomalies")
    n_critical: int = Field(default=0, ge=0, description="Critical anomalies")
    n_error: int = Field(default=0, ge=0, description="Error anomalies")
    n_warning: int = Field(default=0, ge=0, description="Warning anomalies")


class DataQualityReport(BaseModel):
    """Complete data quality report from ML4T Data.

    This is the primary output format for data quality validation.
    ML4T Data generates this report when loading data, and ML4T Diagnostic
    can use it to decide whether to proceed with analysis.

    Attributes:
        symbol: Asset symbol or identifier
        source: Data source/provider name
        date_range: Start and end dates of the data
        frequency: Data frequency (e.g., "1min", "1d", "tick")
        metrics: Quantitative quality metrics
        anomalies: List of detected anomalies
        recommendations: Human-readable recommendations
        is_production_ready: Whether data meets production quality standards
        created_at: When this report was generated

    Example:
        >>> report = DataQualityReport(
        ...     symbol="AAPL",
        ...     source="databento",
        ...     date_range=(datetime(2024, 1, 1), datetime(2024, 6, 30)),
        ...     frequency="1min",
        ...     metrics=DataQualityMetrics(
        ...         completeness=0.995,
        ...         timeliness=1.0,
        ...         accuracy_score=0.99,
        ...         consistency_score=1.0,
        ...         n_records=100000,
        ...         n_anomalies=3,
        ...     ),
        ...     anomalies=[],
        ...     recommendations=["Data quality is excellent"],
        ...     is_production_ready=True
        ... )
    """

    symbol: str = Field(..., description="Asset symbol")
    source: str = Field(..., description="Data source/provider")
    date_range: tuple[datetime, datetime] = Field(..., description="Data date range")
    frequency: str = Field(..., description="Data frequency (1min, 1d, tick)")
    metrics: DataQualityMetrics = Field(..., description="Quality metrics")
    anomalies: list[DataAnomaly] = Field(default_factory=list, description="Detected anomalies")
    recommendations: list[str] = Field(default_factory=list, description="Recommendations")
    is_production_ready: bool = Field(..., description="Meets production standards")
    created_at: datetime = Field(
        default_factory=datetime.utcnow, description="Report generation time"
    )

    def is_acceptable(
        self,
        min_completeness: float = 0.95,
        max_critical: int = 0,
        max_errors: int = 5,
    ) -> bool:
        """Check if data quality meets acceptance criteria.

        Parameters
        ----------
        min_completeness : float, default 0.95
            Minimum acceptable completeness ratio
        max_critical : int, default 0
            Maximum allowed critical anomalies
        max_errors : int, default 5
            Maximum allowed error anomalies

        Returns
        -------
        bool
            True if data meets all criteria
        """
        return (
            self.metrics.completeness >= min_completeness
            and self.metrics.n_critical <= max_critical
            and self.metrics.n_error <= max_errors
        )

    def summary(self) -> str:
        """Generate human-readable summary of data quality.

        Returns
        -------
        str
            Formatted summary string
        """
        lines = [
            "=" * 50,
            f"Data Quality Report: {self.symbol}",
            "=" * 50,
            "",
            f"Source: {self.source}",
            f"Date range: {self.date_range[0].date()} to {self.date_range[1].date()}",
            f"Frequency: {self.frequency}",
            f"Records: {self.metrics.n_records:,}",
            "",
            "--- Quality Metrics ---",
            f"Completeness: {self.metrics.completeness:.1%}",
            f"Accuracy: {self.metrics.accuracy_score:.1%}",
            f"Consistency: {self.metrics.consistency_score:.1%}",
            "",
            "--- Anomalies ---",
            f"Critical: {self.metrics.n_critical}",
            f"Errors: {self.metrics.n_error}",
            f"Warnings: {self.metrics.n_warning}",
            "",
            f"Production Ready: {'YES' if self.is_production_ready else 'NO'}",
        ]

        if self.recommendations:
            lines.append("")
            lines.append("--- Recommendations ---")
            for rec in self.recommendations:
                lines.append(f"  - {rec}")

        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Export to dictionary format.

        Returns
        -------
        dict
            Dictionary representation suitable for JSON serialization
        """
        return self.model_dump(mode="json")


class DataValidationRequest(BaseModel):
    """Request from ML4T Diagnostic to ML4T Data for validation.

    Allows ML4T Diagnostic to specify what validation checks are needed.
    ML4T Data can use this to customize the quality report.

    Attributes:
        symbol: Asset to validate
        date_range: Date range to validate (optional)
        checks: Specific checks to run
        thresholds: Custom thresholds for validation
        include_details: Whether to include detailed anomaly records

    Example:
        >>> request = DataValidationRequest(
        ...     symbol="AAPL",
        ...     checks=["completeness", "price_spikes", "ohlc_validation"],
        ...     thresholds={"price_spike_std": 5.0},
        ...     include_details=True
        ... )
    """

    symbol: str = Field(..., description="Asset to validate")
    date_range: tuple[datetime, datetime] | None = Field(
        None, description="Optional date range to validate"
    )
    checks: list[str] = Field(
        default_factory=lambda: ["completeness", "stale_data", "price_spikes", "ohlc_validation"],
        description="Validation checks to run",
    )
    thresholds: dict[str, float] = Field(
        default_factory=dict, description="Custom thresholds for validation checks"
    )
    include_details: bool = Field(default=True, description="Include detailed anomaly records")
