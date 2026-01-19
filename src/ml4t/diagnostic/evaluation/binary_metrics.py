"""Binary Classification Metrics for Trading Signal Evaluation.

This module provides precision, recall, lift, and coverage metrics for evaluating
binary trading signals against labeled outcomes. Designed to complement the
existing Signal Analysis and Feature Diagnostics capabilities.

Key Features:
    - Polars-native implementation (fast, memory-efficient)
    - Statistical significance testing (binomial test, proportions z-test)
    - Confidence intervals via Wilson score
    - Sparse signal support (handles low coverage gracefully)
    - Comprehensive report generation

Usage Example:
    >>> import polars as pl
    >>> from ml4t.diagnostic.evaluation.binary_metrics import (
    ...     precision, recall, lift, coverage, binary_classification_report
    ... )
    >>>
    >>> # Example data
    >>> signals = pl.Series([1, 0, 1, 1, 0, 1, 0, 0, 1, 0])
    >>> labels = pl.Series([1, 0, 1, 0, 0, 1, 0, 1, 1, 0])
    >>>
    >>> # Compute metrics
    >>> prec = precision(signals, labels)
    >>> rec = recall(signals, labels)
    >>> print(f"Precision: {prec:.3f}, Recall: {rec:.3f}")

References:
    Wilson, E.B. (1927). "Probable inference, the law of succession,
    and statistical inference". Journal of the American Statistical
    Association.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import polars as pl
from scipy import stats

# ============================================================================
# Core Metrics
# ============================================================================


def precision(signals: pl.Series, labels: pl.Series) -> float:
    """Compute precision: P(label=1 | signal=1).

    Precision measures the accuracy of positive predictions. In trading:
        - High precision = most signals lead to profitable outcomes
        - Low precision = many false positives (unprofitable trades)

    Parameters
    ----------
    signals : pl.Series
        Binary series (1=signal, 0=no signal)
    labels : pl.Series
        Binary series (1=positive outcome, 0=negative outcome)

    Returns
    -------
    float
        Precision value in [0, 1], or NaN if no signals

    Formula
    -------
    precision = TP / (TP + FP)
    where TP = true positives, FP = false positives
    """
    n_signals = signals.sum()
    if n_signals == 0:
        return float("nan")

    tp = ((signals == 1) & (labels == 1)).sum()
    fp = ((signals == 1) & (labels == 0)).sum()

    return float(tp / (tp + fp))


def recall(signals: pl.Series, labels: pl.Series) -> float:
    """Compute recall (sensitivity): P(signal=1 | label=1).

    Recall measures coverage of positive outcomes. In trading:
        - High recall = captures most profitable opportunities
        - Low recall = misses many profitable opportunities

    Parameters
    ----------
    signals : pl.Series
        Binary series (1=signal, 0=no signal)
    labels : pl.Series
        Binary series (1=positive outcome, 0=negative outcome)

    Returns
    -------
    float
        Recall value in [0, 1], or NaN if no positive labels

    Formula
    -------
    recall = TP / (TP + FN)
    where TP = true positives, FN = false negatives
    """
    n_positives = labels.sum()
    if n_positives == 0:
        return float("nan")

    tp = ((signals == 1) & (labels == 1)).sum()
    fn = ((signals == 0) & (labels == 1)).sum()

    return float(tp / (tp + fn))


def coverage(signals: pl.Series) -> float:
    """Compute signal coverage: fraction of observations with signals.

    Coverage measures how frequently the indicator generates signals:
        - High coverage (>20%) = many trading opportunities
        - Low coverage (<5%) = sparse/rare signals

    Parameters
    ----------
    signals : pl.Series
        Binary series (1=signal, 0=no signal)

    Returns
    -------
    float
        Coverage value in [0, 1]

    Formula
    -------
    coverage = (# signals) / (# total observations)
    """
    n = len(signals)
    if n == 0:
        return float("nan")
    return float(signals.sum() / n)


def lift(signals: pl.Series, labels: pl.Series) -> float:
    """Compute lift: precision / base_rate.

    Lift measures improvement over random selection:
        - Lift > 1.0 = signal better than random
        - Lift < 1.0 = signal worse than random
        - Lift = 1.0 = signal no better than random

    Parameters
    ----------
    signals : pl.Series
        Binary series (1=signal, 0=no signal)
    labels : pl.Series
        Binary series (1=positive outcome, 0=negative outcome)

    Returns
    -------
    float
        Lift value (typically 0.5 - 3.0), or NaN if no signals or labels

    Formula
    -------
    lift = precision / base_rate
    where base_rate = P(label=1) overall
    """
    n = len(labels)
    if n == 0:
        return float("nan")

    base_rate = labels.sum() / n
    if base_rate == 0 or signals.sum() == 0:
        return float("nan")

    prec = precision(signals, labels)
    return float(prec / base_rate)


def f1_score(signals: pl.Series, labels: pl.Series) -> float:
    """Compute F1 score: harmonic mean of precision and recall.

    F1 balances precision and recall:
        - F1 = 1.0 = perfect precision and recall
        - F1 = 0.0 = zero precision or recall

    Parameters
    ----------
    signals : pl.Series
        Binary series (1=signal, 0=no signal)
    labels : pl.Series
        Binary series (1=positive outcome, 0=negative outcome)

    Returns
    -------
    float
        F1 score in [0, 1], or NaN if undefined

    Formula
    -------
    F1 = 2 * (precision * recall) / (precision + recall)
    """
    prec = precision(signals, labels)
    rec = recall(signals, labels)

    if np.isnan(prec) or np.isnan(rec) or (prec + rec) == 0:
        return float("nan")

    return 2 * (prec * rec) / (prec + rec)


def specificity(signals: pl.Series, labels: pl.Series) -> float:
    """Compute specificity: P(signal=0 | label=0).

    Specificity measures the true negative rate:
        - High specificity = correctly avoids bad trades
        - Low specificity = many false positives

    Parameters
    ----------
    signals : pl.Series
        Binary series (1=signal, 0=no signal)
    labels : pl.Series
        Binary series (1=positive outcome, 0=negative outcome)

    Returns
    -------
    float
        Specificity value in [0, 1], or NaN if no negative labels

    Formula
    -------
    specificity = TN / (TN + FP)
    where TN = true negatives, FP = false positives
    """
    n_negatives = (labels == 0).sum()
    if n_negatives == 0:
        return float("nan")

    tn = ((signals == 0) & (labels == 0)).sum()
    fp = ((signals == 1) & (labels == 0)).sum()

    return float(tn / (tn + fp))


def balanced_accuracy(signals: pl.Series, labels: pl.Series) -> float:
    """Compute balanced accuracy: average of recall and specificity.

    Balanced accuracy is useful when classes are imbalanced:
        - Equal weight to both positive and negative class performance
        - Range [0, 1], where 0.5 = random classifier

    Parameters
    ----------
    signals : pl.Series
        Binary series (1=signal, 0=no signal)
    labels : pl.Series
        Binary series (1=positive outcome, 0=negative outcome)

    Returns
    -------
    float
        Balanced accuracy in [0, 1], or NaN if undefined

    Formula
    -------
    balanced_accuracy = (recall + specificity) / 2
    """
    rec = recall(signals, labels)
    spec = specificity(signals, labels)

    if np.isnan(rec) or np.isnan(spec):
        return float("nan")

    return (rec + spec) / 2


# ============================================================================
# Confidence Intervals
# ============================================================================


def wilson_score_interval(
    n_successes: int,
    n_trials: int,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Compute Wilson score confidence interval for a proportion.

    More accurate than normal approximation, especially for small samples
    or extreme proportions. Recommended for trading signal evaluation.

    Parameters
    ----------
    n_successes : int
        Number of successes (e.g., true positives)
    n_trials : int
        Total number of trials (e.g., total signals)
    confidence : float, default 0.95
        Confidence level for the interval

    Returns
    -------
    tuple[float, float]
        (lower_bound, upper_bound) of the confidence interval

    References
    ----------
    Wilson, E.B. (1927). "Probable inference, the law of succession,
    and statistical inference". Journal of the American Statistical
    Association.

    Examples
    --------
    >>> lower, upper = wilson_score_interval(45, 100, confidence=0.95)
    >>> print(f"95% CI: [{lower:.3f}, {upper:.3f}]")
    """
    if n_trials == 0:
        return (float("nan"), float("nan"))

    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    p_hat = n_successes / n_trials

    denominator = 1 + z**2 / n_trials
    center = (p_hat + z**2 / (2 * n_trials)) / denominator
    margin = z * np.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * n_trials)) / n_trials) / denominator

    return (float(center - margin), float(center + margin))


# ============================================================================
# Statistical Tests
# ============================================================================


def binomial_test_precision(
    tp: int,
    n: int,
    prevalence: float,
    alternative: Literal["greater", "less", "two-sided"] = "greater",
) -> float:
    """Test if precision is significantly better than random using binomial test.

    Null hypothesis: precision = prevalence (signal no better than random)
    Alternative: precision > prevalence (signal better than random)

    Parameters
    ----------
    tp : int
        True positives (# signals with positive outcomes)
    n : int
        Total signals (# times signal=1)
    prevalence : float
        Base rate P(label=1) in population
    alternative : {'greater', 'less', 'two-sided'}, default 'greater'
        Alternative hypothesis direction

    Returns
    -------
    float
        p-value for the binomial test

    Notes
    -----
    Interpretation:
        - p < 0.05 => precision significantly > prevalence (good signal!)
        - p >= 0.05 => precision not significantly better than random
    """
    if n == 0:
        return float("nan")

    # Handle edge case where prevalence is 0 or 1
    if prevalence <= 0 or prevalence >= 1:
        return float("nan")

    result = stats.binomtest(tp, n, prevalence, alternative=alternative)
    return float(result.pvalue)


def proportions_z_test(
    signals: pl.Series,
    labels: pl.Series,
    alternative: Literal["greater", "less", "two-sided"] = "greater",
) -> tuple[float, float]:
    """Test if precision differs from base rate using z-test.

    More powerful than binomial test for large samples (n > 30).
    Null hypothesis: precision = base_rate

    Parameters
    ----------
    signals : pl.Series
        Binary series (1=signal, 0=no signal)
    labels : pl.Series
        Binary series (1=positive outcome, 0=negative outcome)
    alternative : {'greater', 'less', 'two-sided'}, default 'greater'
        Alternative hypothesis direction

    Returns
    -------
    tuple[float, float]
        (z_statistic, p_value)

    Notes
    -----
    Interpretation:
        - p < 0.05 => precision significantly different from base rate
        - z > 0 => precision > base rate (good)
        - z < 0 => precision < base rate (bad)
    """
    n_signals = int(signals.sum())
    n_total = len(labels)

    if n_signals == 0 or n_total == 0:
        return (float("nan"), float("nan"))

    # Signal group precision
    tp = int(((signals == 1) & (labels == 1)).sum())
    p1 = tp / n_signals

    # Population base rate
    p2 = float(labels.sum() / n_total)
    n2 = n_total - n_signals

    if n2 == 0:
        return (float("nan"), float("nan"))

    # Pooled proportion
    p_pool = float(labels.sum() / n_total)

    # Standard error
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / n_signals + 1 / n2))

    if se == 0:
        return (float("nan"), float("nan"))

    # Z-statistic
    z = (p1 - p2) / se

    # P-value
    if alternative == "greater":
        p_value = 1 - stats.norm.cdf(z)
    elif alternative == "less":
        p_value = stats.norm.cdf(z)
    else:  # two-sided
        p_value = 2 * (1 - stats.norm.cdf(abs(z)))

    return (float(z), float(p_value))


def compare_precisions_z_test(
    signals1: pl.Series,
    labels1: pl.Series,
    signals2: pl.Series,
    labels2: pl.Series,
    alternative: Literal["greater", "less", "two-sided"] = "two-sided",
) -> tuple[float, float]:
    """Compare precision between two strategies using z-test.

    Tests whether strategy 1 has significantly different precision than strategy 2.

    Parameters
    ----------
    signals1 : pl.Series
        Binary signals from strategy 1
    labels1 : pl.Series
        Binary labels for strategy 1
    signals2 : pl.Series
        Binary signals from strategy 2
    labels2 : pl.Series
        Binary labels for strategy 2
    alternative : {'greater', 'less', 'two-sided'}, default 'two-sided'
        Alternative hypothesis direction

    Returns
    -------
    tuple[float, float]
        (z_statistic, p_value)
    """
    n1 = int(signals1.sum())
    n2 = int(signals2.sum())

    if n1 == 0 or n2 == 0:
        return (float("nan"), float("nan"))

    tp1 = int(((signals1 == 1) & (labels1 == 1)).sum())
    tp2 = int(((signals2 == 1) & (labels2 == 1)).sum())

    p1 = tp1 / n1
    p2 = tp2 / n2

    # Pooled proportion
    p_pool = (tp1 + tp2) / (n1 + n2)

    # Standard error
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))

    if se == 0:
        return (float("nan"), float("nan"))

    z = (p1 - p2) / se

    if alternative == "greater":
        p_value = 1 - stats.norm.cdf(z)
    elif alternative == "less":
        p_value = stats.norm.cdf(z)
    else:
        p_value = 2 * (1 - stats.norm.cdf(abs(z)))

    return (float(z), float(p_value))


# ============================================================================
# Confusion Matrix
# ============================================================================


@dataclass
class ConfusionMatrix:
    """Confusion matrix for binary classification.

    Attributes
    ----------
    tp : int
        True positives
    fp : int
        False positives
    tn : int
        True negatives
    fn : int
        False negatives
    """

    tp: int
    fp: int
    tn: int
    fn: int

    @property
    def n_signals(self) -> int:
        """Total positive predictions."""
        return self.tp + self.fp

    @property
    def n_positives(self) -> int:
        """Total actual positives."""
        return self.tp + self.fn

    @property
    def n_negatives(self) -> int:
        """Total actual negatives."""
        return self.tn + self.fp

    @property
    def n_total(self) -> int:
        """Total observations."""
        return self.tp + self.fp + self.tn + self.fn

    def to_dict(self) -> dict[str, int]:
        """Convert to dictionary."""
        return {
            "tp": self.tp,
            "fp": self.fp,
            "tn": self.tn,
            "fn": self.fn,
            "n_signals": self.n_signals,
            "n_positives": self.n_positives,
            "n_negatives": self.n_negatives,
            "n_total": self.n_total,
        }


def compute_confusion_matrix(signals: pl.Series, labels: pl.Series) -> ConfusionMatrix:
    """Compute confusion matrix from signals and labels.

    Parameters
    ----------
    signals : pl.Series
        Binary series (1=signal, 0=no signal)
    labels : pl.Series
        Binary series (1=positive outcome, 0=negative outcome)

    Returns
    -------
    ConfusionMatrix
        Confusion matrix with tp, fp, tn, fn
    """
    tp = int(((signals == 1) & (labels == 1)).sum())
    fp = int(((signals == 1) & (labels == 0)).sum())
    tn = int(((signals == 0) & (labels == 0)).sum())
    fn = int(((signals == 0) & (labels == 1)).sum())

    return ConfusionMatrix(tp=tp, fp=fp, tn=tn, fn=fn)


# ============================================================================
# Comprehensive Report
# ============================================================================


@dataclass
class BinaryClassificationReport:
    """Comprehensive binary classification report.

    Attributes
    ----------
    precision : float
        Precision (positive predictive value)
    recall : float
        Recall (sensitivity, true positive rate)
    f1_score : float
        Harmonic mean of precision and recall
    specificity : float
        True negative rate
    balanced_accuracy : float
        Average of recall and specificity
    lift : float
        Improvement over random selection
    coverage : float
        Fraction of observations with signals
    confusion_matrix : ConfusionMatrix
        Confusion matrix details
    base_rate : float
        Population prevalence of positive class
    precision_ci : tuple[float, float]
        Wilson score CI for precision
    recall_ci : tuple[float, float]
        Wilson score CI for recall
    binomial_pvalue : float
        P-value for binomial test of precision > base_rate
    z_test_stat : float
        Z-statistic for precision vs base_rate
    z_test_pvalue : float
        P-value for z-test
    mean_return_on_signal : float | None
        Mean return when signal=1 (if returns provided)
    mean_return_no_signal : float | None
        Mean return when signal=0 (if returns provided)
    return_lift : float | None
        Ratio of signal return to no-signal return (if returns provided)
    """

    precision: float
    recall: float
    f1_score: float
    specificity: float
    balanced_accuracy: float
    lift: float
    coverage: float
    confusion_matrix: ConfusionMatrix
    base_rate: float
    precision_ci: tuple[float, float]
    recall_ci: tuple[float, float]
    binomial_pvalue: float
    z_test_stat: float
    z_test_pvalue: float
    mean_return_on_signal: float | None = None
    mean_return_no_signal: float | None = None
    return_lift: float | None = None

    def to_dict(self) -> dict:
        """Convert report to dictionary."""
        result = {
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "specificity": self.specificity,
            "balanced_accuracy": self.balanced_accuracy,
            "lift": self.lift,
            "coverage": self.coverage,
            "base_rate": self.base_rate,
            "precision_ci": self.precision_ci,
            "recall_ci": self.recall_ci,
            "binomial_pvalue": self.binomial_pvalue,
            "z_test_stat": self.z_test_stat,
            "z_test_pvalue": self.z_test_pvalue,
            **self.confusion_matrix.to_dict(),
        }
        if self.mean_return_on_signal is not None:
            result["mean_return_on_signal"] = self.mean_return_on_signal
            result["mean_return_no_signal"] = self.mean_return_no_signal
            result["return_lift"] = self.return_lift
        return result

    @property
    def is_significant(self) -> bool:
        """Whether precision is significantly better than base rate at p<0.05."""
        return self.binomial_pvalue < 0.05

    @property
    def is_sparse(self) -> bool:
        """Whether signal coverage is below 5%."""
        return self.coverage < 0.05


def binary_classification_report(
    signals: pl.Series,
    labels: pl.Series,
    returns: pl.Series | None = None,
    confidence: float = 0.95,
) -> BinaryClassificationReport:
    """Generate comprehensive binary classification report for trading signal.

    Computes all key metrics with confidence intervals and statistical tests.

    Parameters
    ----------
    signals : pl.Series
        Binary series (1=signal, 0=no signal)
    labels : pl.Series
        Binary series (1=positive outcome, 0=negative outcome)
    returns : pl.Series, optional
        Series of returns for additional return analysis
    confidence : float, default 0.95
        Confidence level for Wilson score intervals

    Returns
    -------
    BinaryClassificationReport
        Comprehensive report with all metrics, CIs, and statistical tests

    Examples
    --------
    >>> report = binary_classification_report(signals, labels)
    >>> print(f"Precision: {report.precision:.3f} "
    ...       f"[{report.precision_ci[0]:.3f}, {report.precision_ci[1]:.3f}]")
    >>> print(f"Statistical significance: p={report.binomial_pvalue:.4f}")
    >>> if report.is_significant:
    ...     print("Signal is significantly better than random!")
    """
    # Compute confusion matrix
    cm = compute_confusion_matrix(signals, labels)

    # Basic metrics
    prec = precision(signals, labels)
    rec = recall(signals, labels)
    f1 = f1_score(signals, labels)
    spec = specificity(signals, labels)
    bal_acc = balanced_accuracy(signals, labels)
    lift_val = lift(signals, labels)
    cov = coverage(signals)

    # Base rate
    base_rate = cm.n_positives / cm.n_total if cm.n_total > 0 else float("nan")

    # Confidence intervals
    prec_ci = wilson_score_interval(cm.tp, cm.n_signals, confidence)
    rec_ci = wilson_score_interval(cm.tp, cm.n_positives, confidence)

    # Statistical tests
    binom_pvalue = binomial_test_precision(cm.tp, cm.n_signals, base_rate)
    z_stat, z_pvalue = proportions_z_test(signals, labels)

    # Returns analysis (if provided)
    mean_ret_signal = None
    mean_ret_no_signal = None
    ret_lift = None

    if returns is not None:
        signal_mask = signals == 1
        no_signal_mask = signals == 0

        if signal_mask.sum() > 0:
            val = returns.filter(signal_mask).mean()
            if val is not None and isinstance(val, int | float):
                mean_ret_signal = float(val)
        if no_signal_mask.sum() > 0:
            val = returns.filter(no_signal_mask).mean()
            if val is not None and isinstance(val, int | float):
                mean_ret_no_signal = float(val)

        if (
            mean_ret_signal is not None
            and mean_ret_no_signal is not None
            and mean_ret_no_signal != 0
        ):
            ret_lift = mean_ret_signal / mean_ret_no_signal

    return BinaryClassificationReport(
        precision=prec,
        recall=rec,
        f1_score=f1,
        specificity=spec,
        balanced_accuracy=bal_acc,
        lift=lift_val,
        coverage=cov,
        confusion_matrix=cm,
        base_rate=base_rate,
        precision_ci=prec_ci,
        recall_ci=rec_ci,
        binomial_pvalue=binom_pvalue,
        z_test_stat=z_stat,
        z_test_pvalue=z_pvalue,
        mean_return_on_signal=mean_ret_signal,
        mean_return_no_signal=mean_ret_no_signal,
        return_lift=ret_lift,
    )


def format_classification_report(report: BinaryClassificationReport) -> str:
    """Format binary classification report as human-readable string.

    Parameters
    ----------
    report : BinaryClassificationReport
        Report from binary_classification_report()

    Returns
    -------
    str
        Formatted string with metrics and interpretation
    """
    cm = report.confusion_matrix

    lines = [
        "Binary Classification Report",
        "=" * 50,
        "",
        f"Sample Size: {cm.n_total:,}",
        f"Base Rate: {report.base_rate:.3f} ({cm.n_positives:,} positives)",
        "",
        "Metrics:",
        f"  Precision:    {report.precision:.3f} "
        f"[{report.precision_ci[0]:.3f}, {report.precision_ci[1]:.3f}]",
        f"  Recall:       {report.recall:.3f} "
        f"[{report.recall_ci[0]:.3f}, {report.recall_ci[1]:.3f}]",
        f"  F1 Score:     {report.f1_score:.3f}",
        f"  Specificity:  {report.specificity:.3f}",
        f"  Balanced Acc: {report.balanced_accuracy:.3f}",
        f"  Lift:         {report.lift:.3f}",
        f"  Coverage:     {report.coverage:.3f} ({cm.n_signals:,} signals)",
        "",
        "Confusion Matrix:",
        f"  TP: {cm.tp:>6,}  FP: {cm.fp:>6,}",
        f"  FN: {cm.fn:>6,}  TN: {cm.tn:>6,}",
        "",
        "Statistical Significance:",
        f"  Binomial test p-value: {report.binomial_pvalue:.4f}",
        f"  Z-test statistic:      {report.z_test_stat:.3f}",
        f"  Z-test p-value:        {report.z_test_pvalue:.4f}",
    ]

    # Add returns analysis if available
    if report.mean_return_on_signal is not None:
        lines.extend(
            [
                "",
                "Returns Analysis:",
                f"  Mean return (signal):    {report.mean_return_on_signal:.4f}",
                f"  Mean return (no signal): {report.mean_return_no_signal:.4f}",
                f"  Return lift:             {report.return_lift:.3f}",
            ]
        )

    # Interpretation
    lines.extend(["", "Interpretation:"])

    if report.is_significant:
        lines.append("  [+] Signal precision significantly > base rate (p < 0.05)")
    else:
        lines.append("  [-] Signal precision NOT significantly > base rate (p >= 0.05)")

    if report.lift > 1.2:
        lines.append("  [+] Strong lift (>1.2x better than random)")
    elif report.lift > 1.0:
        lines.append("  [~] Moderate lift (>1.0x better than random)")
    else:
        lines.append("  [-] No lift (<= 1.0x, not better than random)")

    if report.is_sparse:
        lines.append("  [!] Very sparse signals (<5% coverage)")
    elif report.coverage > 0.20:
        lines.append("  [+] High signal frequency (>20% coverage)")

    return "\n".join(lines)


# ============================================================================
# Convenience Functions
# ============================================================================


def compute_all_metrics(
    signals: pl.Series,
    labels: pl.Series,
) -> dict[str, float]:
    """Compute all binary classification metrics.

    Parameters
    ----------
    signals : pl.Series
        Binary series (1=signal, 0=no signal)
    labels : pl.Series
        Binary series (1=positive outcome, 0=negative outcome)

    Returns
    -------
    dict[str, float]
        Dictionary with all metric values
    """
    return {
        "precision": precision(signals, labels),
        "recall": recall(signals, labels),
        "f1_score": f1_score(signals, labels),
        "specificity": specificity(signals, labels),
        "balanced_accuracy": balanced_accuracy(signals, labels),
        "lift": lift(signals, labels),
        "coverage": coverage(signals),
    }
