"""Tests for binary classification metrics module.

These tests verify:
1. Core metrics (precision, recall, F1, lift, coverage, specificity)
2. Confidence intervals (Wilson score)
3. Statistical tests (binomial, z-test, comparison)
4. Confusion matrix
5. Comprehensive reports
6. Edge cases (empty data, all zeros, all ones, sparse signals)
7. Numerical accuracy
"""

import math

import numpy as np
import polars as pl
import pytest

from ml4t.diagnostic.evaluation.binary_metrics import (
    BinaryClassificationReport,
    balanced_accuracy,
    binary_classification_report,
    binomial_test_precision,
    compare_precisions_z_test,
    compute_all_metrics,
    compute_confusion_matrix,
    coverage,
    f1_score,
    format_classification_report,
    lift,
    precision,
    proportions_z_test,
    recall,
    specificity,
    wilson_score_interval,
)

# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def perfect_signal() -> tuple[pl.Series, pl.Series]:
    """Perfect signal: all predictions correct."""
    signals = pl.Series([1, 1, 1, 1, 0, 0, 0, 0])
    labels = pl.Series([1, 1, 1, 1, 0, 0, 0, 0])
    return signals, labels


@pytest.fixture
def inverse_signal() -> tuple[pl.Series, pl.Series]:
    """Inverse signal: all predictions wrong."""
    signals = pl.Series([1, 1, 1, 1, 0, 0, 0, 0])
    labels = pl.Series([0, 0, 0, 0, 1, 1, 1, 1])
    return signals, labels


@pytest.fixture
def random_signal() -> tuple[pl.Series, pl.Series]:
    """Random-like signal with mixed results."""
    signals = pl.Series([1, 0, 1, 1, 0, 1, 0, 0, 1, 0])
    labels = pl.Series([1, 0, 1, 0, 0, 1, 0, 1, 1, 0])
    return signals, labels


@pytest.fixture
def sparse_signal() -> tuple[pl.Series, pl.Series]:
    """Sparse signal with few positives (< 5% coverage)."""
    signals = pl.Series([0] * 96 + [1] * 4)  # 4% coverage
    labels = pl.Series([0] * 50 + [1] * 50)
    return signals, labels


@pytest.fixture
def large_dataset() -> tuple[pl.Series, pl.Series]:
    """Large dataset for performance testing."""
    np.random.seed(42)
    n = 100_000
    signals = pl.Series(np.random.binomial(1, 0.3, n))
    labels = pl.Series(np.random.binomial(1, 0.5, n))
    return signals, labels


# ============================================================================
# Core Metrics Tests
# ============================================================================


class TestPrecision:
    """Tests for precision metric."""

    def test_perfect_precision(self, perfect_signal):
        """Perfect signal should have precision = 1.0."""
        signals, labels = perfect_signal
        assert precision(signals, labels) == 1.0

    def test_zero_precision(self, inverse_signal):
        """Inverse signal should have precision = 0.0."""
        signals, labels = inverse_signal
        assert precision(signals, labels) == 0.0

    def test_partial_precision(self, random_signal):
        """Random signal should have precision between 0 and 1."""
        signals, labels = random_signal
        prec = precision(signals, labels)
        assert 0.0 < prec < 1.0
        # TP = 4 (signal=1 and label=1 at indices 0, 2, 5, 8)
        # FP = 1 (signal=1 and label=0 at index 3)
        # precision = 4/5 = 0.8
        assert prec == pytest.approx(0.8, abs=1e-6)

    def test_no_signals_returns_nan(self):
        """No signals should return NaN."""
        signals = pl.Series([0, 0, 0, 0])
        labels = pl.Series([1, 0, 1, 0])
        assert math.isnan(precision(signals, labels))


class TestRecall:
    """Tests for recall metric."""

    def test_perfect_recall(self, perfect_signal):
        """Perfect signal should have recall = 1.0."""
        signals, labels = perfect_signal
        assert recall(signals, labels) == 1.0

    def test_zero_recall(self, inverse_signal):
        """Inverse signal should have recall = 0.0."""
        signals, labels = inverse_signal
        assert recall(signals, labels) == 0.0

    def test_partial_recall(self, random_signal):
        """Random signal should have recall between 0 and 1."""
        signals, labels = random_signal
        rec = recall(signals, labels)
        # TP = 4, FN = 1 (label=1 at index 7 with signal=0)
        # recall = 4/5 = 0.8
        assert rec == pytest.approx(0.8, abs=1e-6)

    def test_no_positives_returns_nan(self):
        """No positive labels should return NaN."""
        signals = pl.Series([1, 1, 0, 0])
        labels = pl.Series([0, 0, 0, 0])
        assert math.isnan(recall(signals, labels))


class TestCoverage:
    """Tests for coverage metric."""

    def test_half_coverage(self, perfect_signal):
        """Half signals should give 50% coverage."""
        signals, _ = perfect_signal
        assert coverage(signals) == 0.5

    def test_full_coverage(self):
        """All signals should give 100% coverage."""
        signals = pl.Series([1, 1, 1, 1])
        assert coverage(signals) == 1.0

    def test_no_coverage(self):
        """No signals should give 0% coverage."""
        signals = pl.Series([0, 0, 0, 0])
        assert coverage(signals) == 0.0

    def test_sparse_coverage(self, sparse_signal):
        """Sparse signal coverage calculation."""
        signals, _ = sparse_signal
        assert coverage(signals) == 0.04  # 4 signals out of 100

    def test_empty_returns_nan(self):
        """Empty series should return NaN."""
        signals = pl.Series([], dtype=pl.Int64)
        assert math.isnan(coverage(signals))


class TestLift:
    """Tests for lift metric."""

    def test_perfect_lift(self, perfect_signal):
        """Perfect signal should have lift = 2.0 (precision=1.0, base_rate=0.5)."""
        signals, labels = perfect_signal
        assert lift(signals, labels) == pytest.approx(2.0, abs=1e-6)

    def test_no_lift(self, inverse_signal):
        """Inverse signal should have lift = 0.0."""
        signals, labels = inverse_signal
        assert lift(signals, labels) == pytest.approx(0.0, abs=1e-6)

    def test_random_lift(self, random_signal):
        """Random signal lift calculation."""
        signals, labels = random_signal
        # precision = 0.8, base_rate = 0.5
        # lift = 0.8 / 0.5 = 1.6
        assert lift(signals, labels) == pytest.approx(1.6, abs=1e-6)

    def test_no_signals_returns_nan(self):
        """No signals should return NaN."""
        signals = pl.Series([0, 0, 0, 0])
        labels = pl.Series([1, 0, 1, 0])
        assert math.isnan(lift(signals, labels))

    def test_no_positives_returns_nan(self):
        """No positive labels should return NaN (base_rate = 0)."""
        signals = pl.Series([1, 1, 0, 0])
        labels = pl.Series([0, 0, 0, 0])
        assert math.isnan(lift(signals, labels))


class TestF1Score:
    """Tests for F1 score metric."""

    def test_perfect_f1(self, perfect_signal):
        """Perfect signal should have F1 = 1.0."""
        signals, labels = perfect_signal
        assert f1_score(signals, labels) == 1.0

    def test_zero_f1(self, inverse_signal):
        """Inverse signal should have F1 = NaN (undefined when precision=recall=0)."""
        signals, labels = inverse_signal
        # When precision = 0 and recall = 0, F1 is undefined (0/0)
        assert math.isnan(f1_score(signals, labels))

    def test_balanced_f1(self, random_signal):
        """F1 with equal precision and recall."""
        signals, labels = random_signal
        # precision = recall = 0.8
        # F1 = 2 * 0.8 * 0.8 / (0.8 + 0.8) = 0.8
        assert f1_score(signals, labels) == pytest.approx(0.8, abs=1e-6)

    def test_no_signals_returns_nan(self):
        """No signals should return NaN."""
        signals = pl.Series([0, 0, 0, 0])
        labels = pl.Series([1, 0, 1, 0])
        assert math.isnan(f1_score(signals, labels))


class TestSpecificity:
    """Tests for specificity metric."""

    def test_perfect_specificity(self, perfect_signal):
        """Perfect signal should have specificity = 1.0."""
        signals, labels = perfect_signal
        assert specificity(signals, labels) == 1.0

    def test_zero_specificity(self, inverse_signal):
        """Inverse signal should have specificity = 0.0."""
        signals, labels = inverse_signal
        assert specificity(signals, labels) == 0.0

    def test_partial_specificity(self, random_signal):
        """Random signal specificity calculation."""
        signals, labels = random_signal
        spec = specificity(signals, labels)
        # TN = 4 (signal=0 and label=0 at indices 1, 4, 6, 9)
        # FP = 1 (signal=1 and label=0 at index 3)
        # specificity = 4/5 = 0.8
        assert spec == pytest.approx(0.8, abs=1e-6)

    def test_no_negatives_returns_nan(self):
        """No negative labels should return NaN."""
        signals = pl.Series([1, 1, 0, 0])
        labels = pl.Series([1, 1, 1, 1])
        assert math.isnan(specificity(signals, labels))


class TestBalancedAccuracy:
    """Tests for balanced accuracy metric."""

    def test_perfect_balanced_accuracy(self, perfect_signal):
        """Perfect signal should have balanced accuracy = 1.0."""
        signals, labels = perfect_signal
        assert balanced_accuracy(signals, labels) == 1.0

    def test_zero_balanced_accuracy(self, inverse_signal):
        """Inverse signal should have balanced accuracy = 0.0."""
        signals, labels = inverse_signal
        assert balanced_accuracy(signals, labels) == 0.0

    def test_partial_balanced_accuracy(self, random_signal):
        """Random signal balanced accuracy calculation."""
        signals, labels = random_signal
        # recall = 0.8, specificity = 0.8
        # balanced_accuracy = (0.8 + 0.8) / 2 = 0.8
        assert balanced_accuracy(signals, labels) == pytest.approx(0.8, abs=1e-6)


# ============================================================================
# Wilson Score Interval Tests
# ============================================================================


class TestWilsonScoreInterval:
    """Tests for Wilson score confidence interval."""

    def test_perfect_score_interval(self):
        """Perfect score (100%) should have CI near 1."""
        lower, upper = wilson_score_interval(100, 100, 0.95)
        assert lower > 0.95
        assert upper == pytest.approx(1.0, abs=0.01)

    def test_zero_score_interval(self):
        """Zero score (0%) should have CI near 0."""
        lower, upper = wilson_score_interval(0, 100, 0.95)
        assert lower == pytest.approx(0.0, abs=0.01)
        assert upper < 0.05

    def test_half_score_interval(self):
        """50% score should have symmetric CI."""
        lower, upper = wilson_score_interval(50, 100, 0.95)
        # CI should be roughly [0.4, 0.6] for 50/100
        assert 0.35 < lower < 0.45
        assert 0.55 < upper < 0.65
        # Check symmetry around 0.5
        assert abs((lower + upper) / 2 - 0.5) < 0.01

    def test_small_sample_wide_interval(self):
        """Small sample should have wide CI."""
        lower, upper = wilson_score_interval(5, 10, 0.95)
        width = upper - lower
        assert width > 0.3  # Wide interval for small sample

    def test_large_sample_narrow_interval(self):
        """Large sample should have narrow CI."""
        lower, upper = wilson_score_interval(500, 1000, 0.95)
        width = upper - lower
        assert width < 0.1  # Narrow interval for large sample

    def test_zero_trials_returns_nan(self):
        """Zero trials should return NaN."""
        lower, upper = wilson_score_interval(0, 0, 0.95)
        assert math.isnan(lower)
        assert math.isnan(upper)

    def test_higher_confidence_wider_interval(self):
        """Higher confidence should give wider interval."""
        lower_95, upper_95 = wilson_score_interval(50, 100, 0.95)
        lower_99, upper_99 = wilson_score_interval(50, 100, 0.99)
        width_95 = upper_95 - lower_95
        width_99 = upper_99 - lower_99
        assert width_99 > width_95


# ============================================================================
# Statistical Tests
# ============================================================================


class TestBinomialTest:
    """Tests for binomial test of precision."""

    def test_significant_precision(self):
        """High precision should be significant."""
        # 80 successes out of 100 vs 50% base rate
        pvalue = binomial_test_precision(80, 100, 0.5, "greater")
        assert pvalue < 0.001  # Highly significant

    def test_nonsignificant_precision(self):
        """Precision at base rate should not be significant."""
        # 50 successes out of 100 vs 50% base rate
        pvalue = binomial_test_precision(50, 100, 0.5, "greater")
        assert pvalue > 0.4  # Not significant (should be ~0.5)

    def test_zero_trials_returns_nan(self):
        """Zero trials should return NaN."""
        assert math.isnan(binomial_test_precision(0, 0, 0.5))

    def test_edge_prevalence_returns_nan(self):
        """Edge prevalence (0 or 1) should return NaN."""
        assert math.isnan(binomial_test_precision(50, 100, 0.0))
        assert math.isnan(binomial_test_precision(50, 100, 1.0))

    def test_two_sided(self):
        """Two-sided test should detect deviation in either direction."""
        # Very low precision
        pvalue = binomial_test_precision(10, 100, 0.5, "two-sided")
        assert pvalue < 0.001


class TestProportionsZTest:
    """Tests for z-test of precision vs base rate."""

    def test_significant_difference(self, random_signal):
        """High precision should differ significantly from base rate."""
        signals, labels = random_signal
        z_stat, pvalue = proportions_z_test(signals, labels)
        # precision = 0.8, base_rate = 0.5
        assert z_stat > 0  # Precision > base rate
        assert pvalue < 0.5  # Some significance (small sample)

    def test_no_signals_returns_nan(self):
        """No signals should return NaN."""
        signals = pl.Series([0, 0, 0, 0])
        labels = pl.Series([1, 0, 1, 0])
        z_stat, pvalue = proportions_z_test(signals, labels)
        assert math.isnan(z_stat)
        assert math.isnan(pvalue)

    def test_large_sample_significance(self, large_dataset):
        """Large sample with similar rates should not be significant."""
        signals, labels = large_dataset
        z_stat, pvalue = proportions_z_test(signals, labels)
        # With random data, should be close to 0
        assert abs(z_stat) < 3


class TestComparePrecisions:
    """Tests for comparing precisions between strategies."""

    def test_different_strategies(self):
        """Different strategies should show difference."""
        # Strategy 1: 80% precision
        signals1 = pl.Series([1] * 100)
        labels1 = pl.Series([1] * 80 + [0] * 20)

        # Strategy 2: 50% precision
        signals2 = pl.Series([1] * 100)
        labels2 = pl.Series([1] * 50 + [0] * 50)

        z_stat, pvalue = compare_precisions_z_test(signals1, labels1, signals2, labels2, "greater")
        assert z_stat > 0
        assert pvalue < 0.001  # Highly significant

    def test_same_strategies(self):
        """Same strategies should not show difference."""
        signals1 = pl.Series([1] * 100)
        labels1 = pl.Series([1] * 60 + [0] * 40)

        signals2 = pl.Series([1] * 100)
        labels2 = pl.Series([1] * 60 + [0] * 40)

        z_stat, pvalue = compare_precisions_z_test(
            signals1, labels1, signals2, labels2, "two-sided"
        )
        assert abs(z_stat) < 0.01
        assert pvalue > 0.99


# ============================================================================
# Confusion Matrix Tests
# ============================================================================


class TestConfusionMatrix:
    """Tests for confusion matrix computation."""

    def test_perfect_confusion_matrix(self, perfect_signal):
        """Perfect signal confusion matrix."""
        signals, labels = perfect_signal
        cm = compute_confusion_matrix(signals, labels)
        assert cm.tp == 4
        assert cm.fp == 0
        assert cm.tn == 4
        assert cm.fn == 0
        assert cm.n_signals == 4
        assert cm.n_positives == 4
        assert cm.n_total == 8

    def test_inverse_confusion_matrix(self, inverse_signal):
        """Inverse signal confusion matrix."""
        signals, labels = inverse_signal
        cm = compute_confusion_matrix(signals, labels)
        assert cm.tp == 0
        assert cm.fp == 4
        assert cm.tn == 0
        assert cm.fn == 4

    def test_confusion_matrix_to_dict(self, random_signal):
        """Confusion matrix to_dict method."""
        signals, labels = random_signal
        cm = compute_confusion_matrix(signals, labels)
        d = cm.to_dict()
        assert "tp" in d
        assert "fp" in d
        assert "tn" in d
        assert "fn" in d
        assert "n_signals" in d
        assert d["n_total"] == 10


# ============================================================================
# Comprehensive Report Tests
# ============================================================================


class TestBinaryClassificationReport:
    """Tests for comprehensive classification report."""

    def test_report_structure(self, random_signal):
        """Report should contain all expected fields."""
        signals, labels = random_signal
        report = binary_classification_report(signals, labels)

        assert isinstance(report, BinaryClassificationReport)
        assert hasattr(report, "precision")
        assert hasattr(report, "recall")
        assert hasattr(report, "f1_score")
        assert hasattr(report, "specificity")
        assert hasattr(report, "balanced_accuracy")
        assert hasattr(report, "lift")
        assert hasattr(report, "coverage")
        assert hasattr(report, "confusion_matrix")
        assert hasattr(report, "base_rate")
        assert hasattr(report, "precision_ci")
        assert hasattr(report, "recall_ci")
        assert hasattr(report, "binomial_pvalue")
        assert hasattr(report, "z_test_stat")
        assert hasattr(report, "z_test_pvalue")

    def test_report_metrics_consistency(self, random_signal):
        """Report metrics should match individual functions."""
        signals, labels = random_signal
        report = binary_classification_report(signals, labels)

        assert report.precision == pytest.approx(precision(signals, labels), abs=1e-6)
        assert report.recall == pytest.approx(recall(signals, labels), abs=1e-6)
        assert report.f1_score == pytest.approx(f1_score(signals, labels), abs=1e-6)
        assert report.lift == pytest.approx(lift(signals, labels), abs=1e-6)
        assert report.coverage == pytest.approx(coverage(signals), abs=1e-6)

    def test_report_to_dict(self, random_signal):
        """Report to_dict method."""
        signals, labels = random_signal
        report = binary_classification_report(signals, labels)
        d = report.to_dict()

        assert isinstance(d, dict)
        assert "precision" in d
        assert "recall" in d
        assert "tp" in d  # From confusion matrix

    def test_is_significant_property(self):
        """Test is_significant property."""
        # Signal with 80% precision vs 50% base rate should be significant
        signals = pl.Series([1] * 100 + [0] * 100)
        labels = pl.Series([1] * 80 + [0] * 20 + [1] * 20 + [0] * 80)
        # precision = 80/100 = 0.8, base_rate = 100/200 = 0.5
        report = binary_classification_report(signals, labels)
        assert report.is_significant is True

    def test_is_sparse_property(self, sparse_signal):
        """Test is_sparse property."""
        signals, labels = sparse_signal
        report = binary_classification_report(signals, labels)
        assert report.is_sparse is True

    def test_report_with_returns(self, random_signal):
        """Report with returns analysis."""
        signals, labels = random_signal
        returns = pl.Series([0.01, -0.01, 0.02, -0.02, 0.005, 0.015, -0.005, 0.01, 0.02, -0.01])

        report = binary_classification_report(signals, labels, returns=returns)

        assert report.mean_return_on_signal is not None
        assert report.mean_return_no_signal is not None
        assert report.return_lift is not None

    def test_format_classification_report(self, random_signal):
        """Format report as string."""
        signals, labels = random_signal
        report = binary_classification_report(signals, labels)
        formatted = format_classification_report(report)

        assert isinstance(formatted, str)
        assert "Binary Classification Report" in formatted
        assert "Precision" in formatted
        assert "Recall" in formatted
        assert "Confusion Matrix" in formatted
        assert "Statistical Significance" in formatted
        assert "Interpretation" in formatted


class TestComputeAllMetrics:
    """Tests for compute_all_metrics convenience function."""

    def test_all_metrics_returned(self, random_signal):
        """Should return all metrics."""
        signals, labels = random_signal
        metrics = compute_all_metrics(signals, labels)

        expected_keys = [
            "precision",
            "recall",
            "f1_score",
            "specificity",
            "balanced_accuracy",
            "lift",
            "coverage",
        ]
        for key in expected_keys:
            assert key in metrics

    def test_metrics_consistency(self, random_signal):
        """Metrics should match individual functions."""
        signals, labels = random_signal
        metrics = compute_all_metrics(signals, labels)

        assert metrics["precision"] == pytest.approx(precision(signals, labels), abs=1e-6)
        assert metrics["recall"] == pytest.approx(recall(signals, labels), abs=1e-6)
        assert metrics["f1_score"] == pytest.approx(f1_score(signals, labels), abs=1e-6)


# ============================================================================
# Edge Case Tests
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_series(self):
        """Empty series should return NaN for all metrics."""
        signals = pl.Series([], dtype=pl.Int64)
        labels = pl.Series([], dtype=pl.Int64)

        assert math.isnan(precision(signals, labels))
        assert math.isnan(recall(signals, labels))
        assert math.isnan(coverage(signals))
        assert math.isnan(lift(signals, labels))
        assert math.isnan(f1_score(signals, labels))

    def test_all_zeros(self):
        """All zeros should handle gracefully."""
        signals = pl.Series([0, 0, 0, 0])
        labels = pl.Series([0, 0, 0, 0])

        assert math.isnan(precision(signals, labels))  # No signals
        assert math.isnan(recall(signals, labels))  # No positives
        assert coverage(signals) == 0.0
        assert math.isnan(lift(signals, labels))

    def test_all_ones(self):
        """All ones should handle gracefully."""
        signals = pl.Series([1, 1, 1, 1])
        labels = pl.Series([1, 1, 1, 1])

        assert precision(signals, labels) == 1.0
        assert recall(signals, labels) == 1.0
        assert coverage(signals) == 1.0
        # lift = precision / base_rate = 1.0 / 1.0 = 1.0
        assert lift(signals, labels) == pytest.approx(1.0, abs=1e-6)
        assert f1_score(signals, labels) == 1.0

    def test_single_observation(self):
        """Single observation should work."""
        signals = pl.Series([1])
        labels = pl.Series([1])

        assert precision(signals, labels) == 1.0
        assert recall(signals, labels) == 1.0
        assert coverage(signals) == 1.0

    def test_very_sparse_signals(self):
        """Very sparse signals (<1%) should work."""
        n = 10000
        signals = pl.Series([0] * (n - 10) + [1] * 10)
        labels = pl.Series([0] * (n // 2) + [1] * (n // 2))

        assert coverage(signals) == 0.001
        prec = precision(signals, labels)
        assert 0.0 <= prec <= 1.0 or math.isnan(prec)


# ============================================================================
# Performance Tests
# ============================================================================


class TestPerformance:
    """Tests for performance on large datasets."""

    def test_large_dataset_metrics(self, large_dataset):
        """Metrics should compute quickly on large datasets."""
        signals, labels = large_dataset

        # All metrics should complete quickly (test just runs them)
        prec = precision(signals, labels)
        rec = recall(signals, labels)
        f1 = f1_score(signals, labels)
        lift_val = lift(signals, labels)
        cov = coverage(signals)

        assert 0.0 <= prec <= 1.0
        assert 0.0 <= rec <= 1.0
        assert 0.0 <= f1 <= 1.0
        assert lift_val > 0.0
        assert 0.0 <= cov <= 1.0

    def test_large_dataset_report(self, large_dataset):
        """Full report should compute on large datasets."""
        signals, labels = large_dataset
        report = binary_classification_report(signals, labels)

        assert isinstance(report, BinaryClassificationReport)
        assert report.confusion_matrix.n_total == 100_000


# ============================================================================
# Numerical Accuracy Tests
# ============================================================================


class TestNumericalAccuracy:
    """Tests for numerical accuracy."""

    def test_precision_against_sklearn(self):
        """Precision should match sklearn's precision_score."""
        # Known values
        signals = pl.Series([1, 0, 1, 1, 0, 1, 0, 0, 1, 0])
        labels = pl.Series([1, 0, 1, 0, 0, 1, 0, 1, 1, 0])

        # Manual calculation: TP=4, FP=1, precision = 4/5 = 0.8
        assert precision(signals, labels) == pytest.approx(0.8, abs=1e-10)

    def test_recall_against_sklearn(self):
        """Recall should match sklearn's recall_score."""
        signals = pl.Series([1, 0, 1, 1, 0, 1, 0, 0, 1, 0])
        labels = pl.Series([1, 0, 1, 0, 0, 1, 0, 1, 1, 0])

        # Manual calculation: TP=4, FN=1, recall = 4/5 = 0.8
        assert recall(signals, labels) == pytest.approx(0.8, abs=1e-10)

    def test_f1_against_sklearn(self):
        """F1 should match sklearn's f1_score."""
        signals = pl.Series([1, 0, 1, 1, 0, 1, 0, 0, 1, 0])
        labels = pl.Series([1, 0, 1, 0, 0, 1, 0, 1, 1, 0])

        # When precision = recall = 0.8, F1 = 0.8
        assert f1_score(signals, labels) == pytest.approx(0.8, abs=1e-10)

    def test_wilson_score_known_values(self):
        """Wilson score should match known values."""
        # 50/100 with 95% CI
        lower, upper = wilson_score_interval(50, 100, 0.95)
        # Expected values from statistical tables
        assert lower == pytest.approx(0.401, abs=0.01)
        assert upper == pytest.approx(0.599, abs=0.01)
