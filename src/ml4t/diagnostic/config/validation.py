"""Custom validators and validation utilities.

This module provides reusable validators, custom types, and validation
helpers used across the configuration system.
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated

from pydantic import Field

# Custom type aliases for common constraints
PositiveInt = Annotated[int, Field(gt=0)]
NonNegativeInt = Annotated[int, Field(ge=0)]
PositiveFloat = Annotated[float, Field(gt=0.0)]
NonNegativeFloat = Annotated[float, Field(ge=0.0)]
Probability = Annotated[float, Field(ge=0.0, le=1.0)]
CorrelationValue = Annotated[float, Field(ge=-1.0, le=1.0)]


class SignificanceLevel(float, Enum):
    """Standard significance levels for hypothesis testing."""

    LEVEL_01 = 0.01
    LEVEL_05 = 0.05
    LEVEL_10 = 0.10


class CorrelationMethod(str, Enum):
    """Correlation calculation methods."""

    PEARSON = "pearson"
    SPEARMAN = "spearman"
    KENDALL = "kendall"


class StationarityTest(str, Enum):
    """Stationarity test types."""

    ADF = "adf"  # Augmented Dickey-Fuller
    KPSS = "kpss"  # Kwiatkowski-Phillips-Schmidt-Shin
    PP = "pp"  # Phillips-Perron


class RegressionType(str, Enum):
    """Regression types for stationarity tests."""

    CONSTANT = "c"  # Constant only
    CONSTANT_TREND = "ct"  # Constant and trend
    CONSTANT_TREND_SQUARED = "ctt"  # Constant, trend, and trend squared
    NONE = "n"  # No constant or trend


class ClusteringMethod(str, Enum):
    """Clustering algorithm types."""

    HIERARCHICAL = "hierarchical"
    KMEANS = "kmeans"
    DBSCAN = "dbscan"


class LinkageMethod(str, Enum):
    """Linkage methods for hierarchical clustering."""

    WARD = "ward"
    COMPLETE = "complete"
    AVERAGE = "average"
    SINGLE = "single"


class DistanceMetric(str, Enum):
    """Distance metrics for clustering."""

    EUCLIDEAN = "euclidean"
    CORRELATION = "correlation"
    MANHATTAN = "manhattan"
    COSINE = "cosine"


class NormalityTest(str, Enum):
    """Normality test types."""

    JARQUE_BERA = "jarque_bera"
    SHAPIRO = "shapiro"
    KOLMOGOROV_SMIRNOV = "ks"
    ANDERSON = "anderson"


class OutlierMethod(str, Enum):
    """Outlier detection methods."""

    ZSCORE = "zscore"
    IQR = "iqr"
    ISOLATION_FOREST = "isolation_forest"


class VolatilityClusterMethod(str, Enum):
    """Methods for detecting volatility clustering."""

    LJUNG_BOX = "ljung_box"
    ENGLE_ARCH = "engle_arch"


class ThresholdOptimizationTarget(str, Enum):
    """Optimization targets for threshold analysis."""

    SHARPE = "sharpe"
    PRECISION = "precision"
    RECALL = "recall"
    F1 = "f1"
    INFORMATION_COEFFICIENT = "ic"


class DriftDetectionMethod(str, Enum):
    """Feature drift detection methods."""

    KOLMOGOROV_SMIRNOV = "ks"
    WASSERSTEIN = "wasserstein"
    PSI = "psi"  # Population Stability Index


class PortfolioMetric(str, Enum):
    """Portfolio performance metrics."""

    SHARPE = "sharpe"
    SORTINO = "sortino"
    CALMAR = "calmar"
    MAX_DRAWDOWN = "max_dd"
    VAR = "var"  # Value at Risk
    CVAR = "cvar"  # Conditional Value at Risk
    OMEGA = "omega"


class TimeFrequency(str, Enum):
    """Time aggregation frequencies."""

    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    ANNUAL = "annual"


class FDRMethod(str, Enum):
    """False Discovery Rate control methods."""

    BONFERRONI = "bonferroni"
    HOLM = "holm"
    BENJAMINI_HOCHBERG = "bh"
    BENJAMINI_YEKUTIELI = "by"


class BayesianPriorDistribution(str, Enum):
    """Prior distributions for Bayesian analysis."""

    NORMAL = "normal"
    STUDENT_T = "student_t"
    UNIFORM = "uniform"


class ReportFormat(str, Enum):
    """Report output formats."""

    HTML = "html"
    JSON = "json"
    PDF = "pdf"


class ReportTemplate(str, Enum):
    """Report templates."""

    FULL = "full"
    SUMMARY = "summary"
    DIAGNOSTIC = "diagnostic"


class ReportTheme(str, Enum):
    """Report visual themes."""

    LIGHT = "light"
    DARK = "dark"
    PROFESSIONAL = "professional"


class TableFormat(str, Enum):
    """Table formatting styles."""

    STYLED = "styled"
    PLAIN = "plain"
    DATATABLES = "datatables"


class DataFrameExportFormat(str, Enum):
    """DataFrame serialization formats for JSON."""

    RECORDS = "records"  # list of dicts
    SPLIT = "split"  # {index: [...], columns: [...], data: [...]}
    INDEX = "index"  # {index: {column: value}}


def validate_positive_int(v: int, field_name: str = "value") -> int:
    """Validate that an integer is positive.

    Args:
        v: Value to validate
        field_name: Name of field for error messages

    Returns:
        Validated value

    Raises:
        ValueError: If value is not positive
    """
    if v <= 0:
        raise ValueError(f"{field_name} must be positive (got {v})")
    return v


def validate_probability(v: float, field_name: str = "probability") -> float:
    """Validate that a float is in [0, 1].

    Args:
        v: Value to validate
        field_name: Name of field for error messages

    Returns:
        Validated value

    Raises:
        ValueError: If value is not in [0, 1]
    """
    if not 0.0 <= v <= 1.0:
        raise ValueError(f"{field_name} must be in [0, 1] (got {v})")
    return v


def validate_significance_level(v: float) -> float:
    """Validate significance level is a standard value.

    Args:
        v: Significance level

    Returns:
        Validated significance level

    Raises:
        ValueError: If not a standard significance level
    """
    standard_levels = {0.01, 0.05, 0.10}
    if v not in standard_levels:
        raise ValueError(
            f"Significance level {v} is non-standard. Consider using 0.01, 0.05, or 0.10 for interpretability."
        )
    return v


def validate_min_max_range(
    min_val: float, max_val: float, field_prefix: str = "range"
) -> tuple[float, float]:
    """Validate that min < max.

    Args:
        min_val: Minimum value
        max_val: Maximum value
        field_prefix: Prefix for error messages

    Returns:
        Validated (min, max) tuple

    Raises:
        ValueError: If min >= max
    """
    if min_val >= max_val:
        raise ValueError(
            f"{field_prefix}_min must be < {field_prefix}_max (got {min_val} >= {max_val})"
        )
    return min_val, max_val
