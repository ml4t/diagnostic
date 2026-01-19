"""Threshold Analysis for Trading Signal Optimization.

This module provides tools for evaluating trading signals across multiple thresholds
and finding optimal threshold values. Essential for:
    - Identifying optimal thresholds for indicator-based signals
    - Understanding metric behavior and trade-offs
    - Detecting monotonicity violations
    - Assessing threshold sensitivity

The module integrates with binary_metrics to compute metrics at each threshold.

Usage Example:
    >>> import polars as pl
    >>> from ml4t.diagnostic.evaluation.threshold_analysis import (
    ...     evaluate_threshold_sweep,
    ...     find_optimal_threshold,
    ...     check_monotonicity,
    ... )
    >>>
    >>> # Indicator values and labels
    >>> indicator = pl.Series([45, 55, 65, 75, 85, 35, 72, 68, 52, 88])
    >>> labels = pl.Series([0, 0, 1, 1, 1, 0, 1, 1, 0, 1])
    >>>
    >>> # Sweep across thresholds
    >>> thresholds = [50, 60, 70, 80]
    >>> results = evaluate_threshold_sweep(indicator, labels, thresholds)
    >>> print(results)
    >>>
    >>> # Find optimal threshold
    >>> optimal = find_optimal_threshold(results, metric="lift", min_coverage=0.1)
    >>> print(f"Optimal threshold: {optimal['threshold']}")
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import polars as pl

from .binary_metrics import (
    binary_classification_report,
)

# ============================================================================
# Core Threshold Sweep
# ============================================================================


def evaluate_threshold_sweep(
    indicator: pl.Series,
    labels: pl.Series,
    thresholds: list[float],
    direction: Literal["above", "below"] = "above",
    returns: pl.Series | None = None,
) -> pl.DataFrame:
    """Evaluate binary classification metrics across multiple thresholds.

    For each threshold, generates binary signals and computes precision, recall,
    F1, lift, coverage, and statistical significance.

    Parameters
    ----------
    indicator : pl.Series
        Continuous indicator values (e.g., RSI, momentum score)
    labels : pl.Series
        Binary labels (1=positive outcome, 0=negative outcome)
    thresholds : list[float]
        List of threshold values to evaluate
    direction : {'above', 'below'}, default 'above'
        Signal direction:
        - 'above': signal=1 when indicator > threshold
        - 'below': signal=1 when indicator < threshold
    returns : pl.Series, optional
        Returns for additional return analysis

    Returns
    -------
    pl.DataFrame
        DataFrame with columns:
        - threshold: threshold value
        - precision, recall, f1_score, specificity, lift, coverage
        - n_signals, n_positives, n_total
        - binomial_pvalue, z_test_pvalue
        - is_significant: whether precision > base_rate at p<0.05
        - mean_return_on_signal (if returns provided)

    Examples
    --------
    >>> indicator = pl.Series([45, 55, 65, 75, 85, 35, 72, 68, 52, 88])
    >>> labels = pl.Series([0, 0, 1, 1, 1, 0, 1, 1, 0, 1])
    >>> results = evaluate_threshold_sweep(indicator, labels, [50, 60, 70, 80])
    >>> print(results.select(["threshold", "precision", "lift", "coverage"]))
    """
    if len(indicator) != len(labels):
        raise ValueError(
            f"indicator and labels must have same length, got {len(indicator)} and {len(labels)}"
        )

    if len(thresholds) == 0:
        raise ValueError("thresholds must not be empty")

    results = []

    for threshold in sorted(thresholds):
        # Generate signals based on direction
        if direction == "above":
            signals = (indicator > threshold).cast(pl.Int8)
        else:
            signals = (indicator < threshold).cast(pl.Int8)

        # Get comprehensive report
        report = binary_classification_report(signals, labels, returns=returns)

        row = {
            "threshold": threshold,
            "precision": report.precision,
            "recall": report.recall,
            "f1_score": report.f1_score,
            "specificity": report.specificity,
            "lift": report.lift,
            "coverage": report.coverage,
            "n_signals": report.confusion_matrix.n_signals,
            "n_positives": report.confusion_matrix.n_positives,
            "n_total": report.confusion_matrix.n_total,
            "base_rate": report.base_rate,
            "binomial_pvalue": report.binomial_pvalue,
            "z_test_pvalue": report.z_test_pvalue,
            "is_significant": 1.0 if report.is_significant else 0.0,
        }

        # Add return metrics if available
        if returns is not None and report.mean_return_on_signal is not None:
            row["mean_return_on_signal"] = report.mean_return_on_signal or 0.0
            row["mean_return_no_signal"] = report.mean_return_no_signal or 0.0
            row["return_lift"] = report.return_lift or 0.0

        results.append(row)

    return pl.DataFrame(results)


def evaluate_percentile_thresholds(
    indicator: pl.Series,
    labels: pl.Series,
    percentiles: list[float] | None = None,
    direction: Literal["above", "below"] = "above",
    returns: pl.Series | None = None,
) -> pl.DataFrame:
    """Evaluate thresholds at indicator percentiles.

    Instead of specifying absolute threshold values, this function computes
    thresholds based on the indicator's distribution. Useful when the indicator
    scale varies across assets or time periods.

    Parameters
    ----------
    indicator : pl.Series
        Continuous indicator values
    labels : pl.Series
        Binary labels
    percentiles : list[float], optional
        Percentiles to evaluate (default: [10, 25, 50, 75, 90])
    direction : {'above', 'below'}, default 'above'
        Signal direction
    returns : pl.Series, optional
        Returns for additional analysis

    Returns
    -------
    pl.DataFrame
        Same as evaluate_threshold_sweep, with additional 'percentile' column

    Examples
    --------
    >>> results = evaluate_percentile_thresholds(indicator, labels)
    >>> print(results.select(["percentile", "threshold", "precision"]))
    """
    if percentiles is None:
        percentiles = [10.0, 25.0, 50.0, 75.0, 90.0]

    # Compute threshold values at each percentile
    thresholds = []
    for p in percentiles:
        q_val = indicator.quantile(p / 100.0)
        threshold = float(q_val) if q_val is not None else 0.0
        thresholds.append(threshold)

    # Evaluate
    results = evaluate_threshold_sweep(indicator, labels, thresholds, direction, returns)

    # Add percentile column
    results = results.with_columns(pl.Series("percentile", percentiles))

    # Reorder columns
    cols = ["percentile", "threshold"] + [
        c for c in results.columns if c not in ["percentile", "threshold"]
    ]
    return results.select(cols)


# ============================================================================
# Optimal Threshold Finding
# ============================================================================


@dataclass
class OptimalThresholdResult:
    """Result of optimal threshold search.

    Attributes
    ----------
    threshold : float | None
        Optimal threshold value, or None if no valid threshold found
    found : bool
        Whether a valid threshold was found
    metric : str
        Metric that was optimized
    metric_value : float | None
        Value of the optimized metric at optimal threshold
    precision : float | None
        Precision at optimal threshold
    recall : float | None
        Recall at optimal threshold
    f1_score : float | None
        F1 score at optimal threshold
    lift : float | None
        Lift at optimal threshold
    coverage : float | None
        Coverage at optimal threshold
    n_signals : int | None
        Number of signals at optimal threshold
    is_significant : bool
        Whether optimal threshold is statistically significant
    reason : str | None
        Reason if no threshold found
    """

    threshold: float | None
    found: bool
    metric: str
    metric_value: float | None = None
    precision: float | None = None
    recall: float | None = None
    f1_score: float | None = None
    lift: float | None = None
    coverage: float | None = None
    n_signals: int | None = None
    is_significant: bool = False
    reason: str | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "threshold": self.threshold,
            "found": self.found,
            "metric": self.metric,
            "metric_value": self.metric_value,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "lift": self.lift,
            "coverage": self.coverage,
            "n_signals": self.n_signals,
            "is_significant": self.is_significant,
            "reason": self.reason,
        }


def find_optimal_threshold(
    results_df: pl.DataFrame,
    metric: str = "lift",
    min_coverage: float = 0.01,
    max_coverage: float = 1.0,
    require_significant: bool = False,
    min_signals: int = 1,
) -> OptimalThresholdResult:
    """Find optimal threshold based on specified metric and constraints.

    Parameters
    ----------
    results_df : pl.DataFrame
        DataFrame from evaluate_threshold_sweep()
    metric : str, default 'lift'
        Metric to optimize ('lift', 'precision', 'f1_score', 'recall')
    min_coverage : float, default 0.01
        Minimum required signal coverage (1%)
    max_coverage : float, default 1.0
        Maximum allowed signal coverage
    require_significant : bool, default False
        Only consider statistically significant thresholds
    min_signals : int, default 1
        Minimum number of signals required

    Returns
    -------
    OptimalThresholdResult
        Result containing optimal threshold and associated metrics

    Examples
    --------
    >>> results = evaluate_threshold_sweep(indicator, labels, thresholds)
    >>> optimal = find_optimal_threshold(results, metric="lift", min_coverage=0.05)
    >>> if optimal.found:
    ...     print(f"Optimal threshold: {optimal.threshold}")
    ...     print(f"Lift: {optimal.lift:.2f}")
    """
    if metric not in results_df.columns:
        return OptimalThresholdResult(
            threshold=None,
            found=False,
            metric=metric,
            reason=f"Metric '{metric}' not in results",
        )

    # Apply filters
    filtered = results_df

    # Coverage constraints
    if "coverage" in filtered.columns:
        filtered = filtered.filter(
            (pl.col("coverage") >= min_coverage) & (pl.col("coverage") <= max_coverage)
        )

    # Minimum signals
    if "n_signals" in filtered.columns:
        filtered = filtered.filter(pl.col("n_signals") >= min_signals)

    # Statistical significance
    if require_significant and "is_significant" in filtered.columns:
        filtered = filtered.filter(pl.col("is_significant") == 1.0)

    # Filter out NaN values in target metric
    filtered = filtered.filter(pl.col(metric).is_not_nan())

    if len(filtered) == 0:
        return OptimalThresholdResult(
            threshold=None,
            found=False,
            metric=metric,
            reason="No thresholds meet constraints",
        )

    # Find maximum
    optimal_idx_result = filtered[metric].arg_max()
    if optimal_idx_result is None:
        return OptimalThresholdResult(
            threshold=0.0,
            found=False,
            metric=metric,
        )
    optimal_idx: int = optimal_idx_result

    # Extract values
    def safe_float(col: str) -> float | None:
        if col in filtered.columns:
            val = filtered[col][optimal_idx]
            return float(val) if val is not None else None
        return None

    def safe_int(col: str) -> int | None:
        if col in filtered.columns:
            val = filtered[col][optimal_idx]
            return int(val) if val is not None else None
        return None

    is_sig = False
    if "is_significant" in filtered.columns:
        is_sig = bool(filtered["is_significant"][optimal_idx])

    threshold_val = filtered["threshold"][optimal_idx]
    return OptimalThresholdResult(
        threshold=float(threshold_val) if threshold_val is not None else 0.0,
        found=True,
        metric=metric,
        metric_value=safe_float(metric),
        precision=safe_float("precision"),
        recall=safe_float("recall"),
        f1_score=safe_float("f1_score"),
        lift=safe_float("lift"),
        coverage=safe_float("coverage"),
        n_signals=safe_int("n_signals"),
        is_significant=is_sig,
    )


def find_threshold_for_target_coverage(
    results_df: pl.DataFrame,
    target_coverage: float,
    tolerance: float = 0.05,
) -> OptimalThresholdResult:
    """Find threshold that achieves target coverage.

    Useful when you want a specific signal frequency regardless of metric value.

    Parameters
    ----------
    results_df : pl.DataFrame
        DataFrame from evaluate_threshold_sweep()
    target_coverage : float
        Target signal coverage (e.g., 0.10 for 10%)
    tolerance : float, default 0.05
        Acceptable deviation from target (e.g., 0.05 means 5-15% for target=10%)

    Returns
    -------
    OptimalThresholdResult
        Result with threshold closest to target coverage
    """
    if "coverage" not in results_df.columns:
        return OptimalThresholdResult(
            threshold=None,
            found=False,
            metric="coverage",
            reason="No coverage column in results",
        )

    # Find threshold closest to target coverage
    results_df = results_df.with_columns(
        (pl.col("coverage") - target_coverage).abs().alias("_coverage_diff")
    )

    # Filter by tolerance
    filtered = results_df.filter(pl.col("_coverage_diff") <= tolerance)

    if len(filtered) == 0:
        return OptimalThresholdResult(
            threshold=None,
            found=False,
            metric="coverage",
            reason=f"No threshold within {tolerance:.1%} of target {target_coverage:.1%}",
        )

    # Find closest
    closest_idx_result = filtered["_coverage_diff"].arg_min()
    if closest_idx_result is None:
        return OptimalThresholdResult(
            threshold=None,
            found=False,
            metric="coverage",
            reason="No threshold found",
        )
    closest_idx: int = closest_idx_result

    def safe_float(col: str) -> float | None:
        if col in filtered.columns:
            val = filtered[col][closest_idx]
            return float(val) if val is not None else None
        return None

    threshold_val = filtered["threshold"][closest_idx]
    n_signals_val = filtered["n_signals"][closest_idx] if "n_signals" in filtered.columns else None
    is_sig_val = (
        filtered["is_significant"][closest_idx] if "is_significant" in filtered.columns else False
    )

    return OptimalThresholdResult(
        threshold=float(threshold_val) if threshold_val is not None else 0.0,
        found=True,
        metric="coverage",
        metric_value=safe_float("coverage"),
        precision=safe_float("precision"),
        recall=safe_float("recall"),
        f1_score=safe_float("f1_score"),
        lift=safe_float("lift"),
        coverage=safe_float("coverage"),
        n_signals=int(n_signals_val) if n_signals_val is not None else None,
        is_significant=bool(is_sig_val) if is_sig_val is not None else False,
    )


# ============================================================================
# Monotonicity Analysis
# ============================================================================


@dataclass
class MonotonicityResult:
    """Result of monotonicity analysis.

    Attributes
    ----------
    metric : str
        Metric that was analyzed
    is_monotonic : bool
        Whether metric is monotonic (either direction)
    is_monotonic_increasing : bool
        Whether metric is monotonically increasing
    is_monotonic_decreasing : bool
        Whether metric is monotonically decreasing
    direction_changes : int
        Number of direction reversals
    violations : list[tuple[int, float, float]]
        List of (index, previous_value, current_value) for violations
    max_violation : float | None
        Largest decrease (for increasing expectation)
    """

    metric: str
    is_monotonic: bool
    is_monotonic_increasing: bool
    is_monotonic_decreasing: bool
    direction_changes: int
    violations: list[tuple[int, float, float]]
    max_violation: float | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "metric": self.metric,
            "is_monotonic": self.is_monotonic,
            "is_monotonic_increasing": self.is_monotonic_increasing,
            "is_monotonic_decreasing": self.is_monotonic_decreasing,
            "direction_changes": self.direction_changes,
            "n_violations": len(self.violations),
            "max_violation": self.max_violation,
        }


def check_monotonicity(
    results_df: pl.DataFrame,
    metric: str,
) -> MonotonicityResult:
    """Check if metric exhibits monotonic behavior across thresholds.

    Non-monotonic behavior can indicate:
        - Regime changes in the data
        - Data quality issues
        - Complex indicator dynamics
        - Overfitting at certain thresholds

    Parameters
    ----------
    results_df : pl.DataFrame
        DataFrame from evaluate_threshold_sweep(), sorted by threshold
    metric : str
        Metric to analyze ('precision', 'recall', 'lift', 'f1_score', etc.)

    Returns
    -------
    MonotonicityResult
        Analysis result with monotonicity status and violations

    Examples
    --------
    >>> results = evaluate_threshold_sweep(indicator, labels, thresholds)
    >>> mono = check_monotonicity(results, "lift")
    >>> if not mono.is_monotonic:
    ...     print(f"Warning: {mono.direction_changes} direction changes")
    """
    if metric not in results_df.columns:
        raise ValueError(f"Metric '{metric}' not in results")

    # Sort by threshold to ensure proper ordering
    sorted_df = results_df.sort("threshold")
    values = sorted_df[metric].to_numpy()

    # Handle NaN values
    valid_mask = ~np.isnan(values)
    if not np.any(valid_mask):
        return MonotonicityResult(
            metric=metric,
            is_monotonic=False,
            is_monotonic_increasing=False,
            is_monotonic_decreasing=False,
            direction_changes=0,
            violations=[],
        )

    # Use only valid values for analysis
    valid_values = values[valid_mask]

    if len(valid_values) < 2:
        return MonotonicityResult(
            metric=metric,
            is_monotonic=True,
            is_monotonic_increasing=True,
            is_monotonic_decreasing=True,
            direction_changes=0,
            violations=[],
        )

    # Compute differences
    diffs = np.diff(valid_values)

    # Check increasing/decreasing
    is_increasing = bool(np.all(diffs >= -1e-10))  # Small tolerance for floating point
    is_decreasing = bool(np.all(diffs <= 1e-10))

    # Count direction changes (ignoring zero differences)
    nonzero_diffs = diffs[np.abs(diffs) > 1e-10]
    if len(nonzero_diffs) > 0:
        signs = np.sign(nonzero_diffs)
        direction_changes = int(np.sum(np.abs(np.diff(signs)) > 0))
    else:
        direction_changes = 0

    # Find violations (decreases when we'd expect increase)
    violations = []
    for i in range(1, len(valid_values)):
        if valid_values[i] < valid_values[i - 1] - 1e-10:
            violations.append((i, float(valid_values[i - 1]), float(valid_values[i])))

    # Max violation
    max_violation = None
    if violations:
        max_violation = max(v[1] - v[2] for v in violations)

    return MonotonicityResult(
        metric=metric,
        is_monotonic=is_increasing or is_decreasing,
        is_monotonic_increasing=is_increasing,
        is_monotonic_decreasing=is_decreasing,
        direction_changes=direction_changes,
        violations=violations,
        max_violation=max_violation,
    )


def analyze_all_metrics_monotonicity(
    results_df: pl.DataFrame,
    metrics: list[str] | None = None,
) -> dict[str, MonotonicityResult]:
    """Analyze monotonicity for multiple metrics.

    Parameters
    ----------
    results_df : pl.DataFrame
        DataFrame from evaluate_threshold_sweep()
    metrics : list[str], optional
        Metrics to analyze (default: precision, recall, lift, f1_score, coverage)

    Returns
    -------
    dict[str, MonotonicityResult]
        Dictionary mapping metric name to monotonicity result
    """
    if metrics is None:
        metrics = ["precision", "recall", "lift", "f1_score", "coverage"]

    results = {}
    for metric in metrics:
        if metric in results_df.columns:
            results[metric] = check_monotonicity(results_df, metric)

    return results


# ============================================================================
# Threshold Sensitivity Analysis
# ============================================================================


@dataclass
class SensitivityResult:
    """Result of threshold sensitivity analysis.

    Attributes
    ----------
    metric : str
        Metric analyzed
    mean_value : float
        Mean metric value across thresholds
    std_value : float
        Standard deviation of metric
    min_value : float
        Minimum metric value
    max_value : float
        Maximum metric value
    range_value : float
        Range (max - min)
    coefficient_of_variation : float
        CV = std / mean (relative variability)
    is_stable : bool
        Whether metric is relatively stable (CV < 0.2)
    """

    metric: str
    mean_value: float
    std_value: float
    min_value: float
    max_value: float
    range_value: float
    coefficient_of_variation: float
    is_stable: bool

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "metric": self.metric,
            "mean": self.mean_value,
            "std": self.std_value,
            "min": self.min_value,
            "max": self.max_value,
            "range": self.range_value,
            "cv": self.coefficient_of_variation,
            "is_stable": self.is_stable,
        }


def analyze_threshold_sensitivity(
    results_df: pl.DataFrame,
    metric: str,
    stability_threshold: float = 0.2,
) -> SensitivityResult:
    """Analyze how sensitive a metric is to threshold changes.

    A highly sensitive metric changes dramatically across thresholds,
    suggesting careful threshold selection is important.

    Parameters
    ----------
    results_df : pl.DataFrame
        DataFrame from evaluate_threshold_sweep()
    metric : str
        Metric to analyze
    stability_threshold : float, default 0.2
        CV threshold for considering metric stable

    Returns
    -------
    SensitivityResult
        Sensitivity analysis result

    Examples
    --------
    >>> sensitivity = analyze_threshold_sensitivity(results, "lift")
    >>> if not sensitivity.is_stable:
    ...     print(f"Warning: {metric} varies significantly (CV={sensitivity.cv:.2f})")
    """
    if metric not in results_df.columns:
        raise ValueError(f"Metric '{metric}' not in results")

    values = results_df[metric].drop_nulls().drop_nans()

    if len(values) == 0:
        return SensitivityResult(
            metric=metric,
            mean_value=float("nan"),
            std_value=float("nan"),
            min_value=float("nan"),
            max_value=float("nan"),
            range_value=float("nan"),
            coefficient_of_variation=float("nan"),
            is_stable=False,
        )

    mean_result = values.mean()
    std_result = values.std()
    min_result = values.min()
    max_result = values.max()

    mean_val = (
        float(mean_result)
        if mean_result is not None and isinstance(mean_result, int | float)
        else 0.0
    )
    std_val = (
        float(std_result) if std_result is not None and isinstance(std_result, int | float) else 0.0
    )
    min_val = (
        float(min_result) if min_result is not None and isinstance(min_result, int | float) else 0.0
    )
    max_val = (
        float(max_result) if max_result is not None and isinstance(max_result, int | float) else 0.0
    )
    range_val = max_val - min_val

    # Coefficient of variation
    cv = std_val / mean_val if mean_val != 0 else float("inf")

    return SensitivityResult(
        metric=metric,
        mean_value=mean_val,
        std_value=std_val,
        min_value=min_val,
        max_value=max_val,
        range_value=range_val,
        coefficient_of_variation=cv,
        is_stable=cv < stability_threshold,
    )


# ============================================================================
# Summary Functions
# ============================================================================


@dataclass
class ThresholdAnalysisSummary:
    """Complete threshold analysis summary.

    Attributes
    ----------
    n_thresholds : int
        Number of thresholds evaluated
    optimal : OptimalThresholdResult
        Optimal threshold result
    monotonicity : dict[str, MonotonicityResult]
        Monotonicity results per metric
    sensitivity : dict[str, SensitivityResult]
        Sensitivity results per metric
    significant_count : int
        Number of statistically significant thresholds
    best_per_metric : dict[str, float]
        Best threshold for each metric
    """

    n_thresholds: int
    optimal: OptimalThresholdResult
    monotonicity: dict[str, MonotonicityResult]
    sensitivity: dict[str, SensitivityResult]
    significant_count: int
    best_per_metric: dict[str, float]

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "n_thresholds": self.n_thresholds,
            "optimal": self.optimal.to_dict(),
            "monotonicity": {k: v.to_dict() for k, v in self.monotonicity.items()},
            "sensitivity": {k: v.to_dict() for k, v in self.sensitivity.items()},
            "significant_count": self.significant_count,
            "best_per_metric": self.best_per_metric,
        }


def create_threshold_analysis_summary(
    results_df: pl.DataFrame,
    optimize_metric: str = "lift",
    min_coverage: float = 0.01,
    metrics: list[str] | None = None,
) -> ThresholdAnalysisSummary:
    """Create comprehensive threshold analysis summary.

    Parameters
    ----------
    results_df : pl.DataFrame
        DataFrame from evaluate_threshold_sweep()
    optimize_metric : str, default 'lift'
        Metric to optimize for optimal threshold
    min_coverage : float, default 0.01
        Minimum coverage for optimal threshold
    metrics : list[str], optional
        Metrics to analyze

    Returns
    -------
    ThresholdAnalysisSummary
        Complete analysis summary

    Examples
    --------
    >>> results = evaluate_threshold_sweep(indicator, labels, thresholds)
    >>> summary = create_threshold_analysis_summary(results)
    >>> print(f"Optimal threshold: {summary.optimal.threshold}")
    >>> print(f"Significant thresholds: {summary.significant_count}")
    """
    if metrics is None:
        metrics = ["precision", "recall", "lift", "f1_score", "coverage"]

    # Filter to available metrics
    available_metrics = [m for m in metrics if m in results_df.columns]

    # Optimal threshold
    optimal = find_optimal_threshold(results_df, metric=optimize_metric, min_coverage=min_coverage)

    # Monotonicity analysis
    monotonicity = {}
    for metric in available_metrics:
        monotonicity[metric] = check_monotonicity(results_df, metric)

    # Sensitivity analysis
    sensitivity = {}
    for metric in available_metrics:
        sensitivity[metric] = analyze_threshold_sensitivity(results_df, metric)

    # Significant count
    sig_count = 0
    if "is_significant" in results_df.columns:
        sig_count = int(results_df.filter(pl.col("is_significant") == 1.0).height)

    # Best threshold per metric
    best_per_metric = {}
    for metric in available_metrics:
        result = find_optimal_threshold(results_df, metric=metric, min_coverage=0.0)
        if result.found and result.threshold is not None:
            best_per_metric[metric] = result.threshold

    return ThresholdAnalysisSummary(
        n_thresholds=len(results_df),
        optimal=optimal,
        monotonicity=monotonicity,
        sensitivity=sensitivity,
        significant_count=sig_count,
        best_per_metric=best_per_metric,
    )


def format_threshold_analysis(summary: ThresholdAnalysisSummary) -> str:
    """Format threshold analysis summary as human-readable string.

    Parameters
    ----------
    summary : ThresholdAnalysisSummary
        Summary from create_threshold_analysis_summary()

    Returns
    -------
    str
        Formatted string
    """
    lines = [
        "Threshold Analysis Summary",
        "=" * 50,
        "",
        f"Thresholds Evaluated: {summary.n_thresholds}",
        f"Statistically Significant: {summary.significant_count}",
        "",
    ]

    # Optimal threshold
    lines.append("Optimal Threshold:")
    if summary.optimal.found:
        lines.append(f"  Threshold: {summary.optimal.threshold}")
        lines.append(f"  Optimized Metric: {summary.optimal.metric}")
        lines.append(f"  {summary.optimal.metric}: {summary.optimal.metric_value:.3f}")
        lines.append(f"  Precision: {summary.optimal.precision:.3f}")
        lines.append(f"  Recall: {summary.optimal.recall:.3f}")
        lines.append(f"  Coverage: {summary.optimal.coverage:.1%}")
        lines.append(f"  Significant: {summary.optimal.is_significant}")
    else:
        lines.append(f"  Not found: {summary.optimal.reason}")

    # Monotonicity
    lines.extend(["", "Monotonicity Analysis:"])
    for metric, mono in summary.monotonicity.items():
        status = "[+] Monotonic" if mono.is_monotonic else f"[-] {mono.direction_changes} reversals"
        lines.append(f"  {metric}: {status}")

    # Sensitivity
    lines.extend(["", "Sensitivity Analysis:"])
    for metric, sens in summary.sensitivity.items():
        status = (
            "[+] Stable"
            if sens.is_stable
            else f"[-] Variable (CV={sens.coefficient_of_variation:.2f})"
        )
        lines.append(f"  {metric}: {status}")

    # Best per metric
    lines.extend(["", "Best Threshold per Metric:"])
    for metric, threshold in summary.best_per_metric.items():
        lines.append(f"  {metric}: {threshold}")

    return "\n".join(lines)
