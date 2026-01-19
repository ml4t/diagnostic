"""Tests for threshold analysis module.

These tests verify:
1. Threshold sweep evaluation
2. Percentile-based thresholds
3. Optimal threshold finding
4. Monotonicity analysis
5. Sensitivity analysis
6. Edge cases and error handling
7. Integration with real-world scenarios
"""

import numpy as np
import polars as pl
import pytest

from ml4t.diagnostic.evaluation.threshold_analysis import (
    MonotonicityResult,
    OptimalThresholdResult,
    SensitivityResult,
    ThresholdAnalysisSummary,
    analyze_all_metrics_monotonicity,
    analyze_threshold_sensitivity,
    check_monotonicity,
    create_threshold_analysis_summary,
    evaluate_percentile_thresholds,
    evaluate_threshold_sweep,
    find_optimal_threshold,
    find_threshold_for_target_coverage,
    format_threshold_analysis,
)

# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def rsi_data() -> tuple[pl.Series, pl.Series]:
    """RSI-like indicator with corresponding labels.

    Higher RSI values tend to predict negative outcomes (overbought).
    """
    np.random.seed(42)
    n = 200

    # Generate RSI-like values (0-100)
    rsi = np.random.beta(2, 2) * 100
    rsi = np.clip(np.random.normal(50, 15, n), 0, 100)

    # Labels: higher RSI = more likely negative outcome
    prob_positive = 1 - (rsi / 100) * 0.6  # 70% at RSI=0, 40% at RSI=100
    labels = np.random.binomial(1, prob_positive)

    return pl.Series(rsi), pl.Series(labels)


@pytest.fixture
def momentum_data() -> tuple[pl.Series, pl.Series]:
    """Momentum indicator with corresponding labels.

    Higher momentum tends to predict positive outcomes.
    """
    np.random.seed(123)
    n = 300

    # Generate momentum values (-10 to 10)
    momentum = np.random.normal(0, 3, n)

    # Labels: positive momentum = more likely positive outcome
    prob_positive = 0.3 + 0.4 * (momentum > 0).astype(float) + 0.1 * (momentum > 2).astype(float)
    prob_positive = np.clip(prob_positive, 0.1, 0.9)
    labels = np.random.binomial(1, prob_positive)

    return pl.Series(momentum), pl.Series(labels)


@pytest.fixture
def simple_indicator() -> tuple[pl.Series, pl.Series]:
    """Simple test case with predictable results."""
    # Indicator: 0-100 in steps of 10
    indicator = pl.Series([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0])
    # Labels: higher values more likely positive
    labels = pl.Series([0, 0, 0, 1, 0, 1, 1, 1, 1, 1])
    return indicator, labels


@pytest.fixture
def monotonic_increasing_results() -> pl.DataFrame:
    """Results with monotonically increasing lift."""
    return pl.DataFrame(
        {
            "threshold": [10.0, 20.0, 30.0, 40.0, 50.0],
            "precision": [0.50, 0.55, 0.60, 0.65, 0.70],
            "recall": [0.90, 0.80, 0.70, 0.60, 0.50],
            "f1_score": [0.64, 0.65, 0.65, 0.62, 0.58],
            "lift": [1.0, 1.1, 1.2, 1.3, 1.4],
            "coverage": [0.50, 0.40, 0.30, 0.20, 0.10],
            "n_signals": [100, 80, 60, 40, 20],
            "n_positives": [100, 100, 100, 100, 100],
            "n_total": [200, 200, 200, 200, 200],
            "base_rate": [0.5, 0.5, 0.5, 0.5, 0.5],
            "binomial_pvalue": [0.5, 0.3, 0.1, 0.05, 0.02],
            "z_test_pvalue": [0.5, 0.3, 0.1, 0.05, 0.02],
            "is_significant": [0.0, 0.0, 0.0, 1.0, 1.0],
        }
    )


@pytest.fixture
def non_monotonic_results() -> pl.DataFrame:
    """Results with non-monotonic behavior."""
    return pl.DataFrame(
        {
            "threshold": [10.0, 20.0, 30.0, 40.0, 50.0],
            "precision": [0.50, 0.60, 0.55, 0.65, 0.58],  # Non-monotonic
            "recall": [0.90, 0.80, 0.70, 0.60, 0.50],
            "f1_score": [0.64, 0.69, 0.62, 0.62, 0.54],
            "lift": [1.0, 1.2, 1.1, 1.3, 1.16],  # Non-monotonic
            "coverage": [0.50, 0.40, 0.30, 0.20, 0.10],
            "n_signals": [100, 80, 60, 40, 20],
            "n_positives": [100, 100, 100, 100, 100],
            "n_total": [200, 200, 200, 200, 200],
            "base_rate": [0.5, 0.5, 0.5, 0.5, 0.5],
            "binomial_pvalue": [0.5, 0.1, 0.2, 0.03, 0.15],
            "z_test_pvalue": [0.5, 0.1, 0.2, 0.03, 0.15],
            "is_significant": [0.0, 0.0, 0.0, 1.0, 0.0],
        }
    )


# ============================================================================
# Threshold Sweep Tests
# ============================================================================


class TestEvaluateThresholdSweep:
    """Tests for evaluate_threshold_sweep function."""

    def test_basic_sweep(self, simple_indicator):
        """Basic threshold sweep should return expected columns."""
        indicator, labels = simple_indicator
        thresholds = [25.0, 50.0, 75.0]

        results = evaluate_threshold_sweep(indicator, labels, thresholds)

        # Check expected columns
        expected_cols = [
            "threshold",
            "precision",
            "recall",
            "f1_score",
            "specificity",
            "lift",
            "coverage",
            "n_signals",
            "n_positives",
            "n_total",
            "base_rate",
            "binomial_pvalue",
            "z_test_pvalue",
            "is_significant",
        ]
        for col in expected_cols:
            assert col in results.columns, f"Missing column: {col}"

        # Check number of rows
        assert len(results) == 3

        # Check thresholds are sorted
        assert results["threshold"].to_list() == [25.0, 50.0, 75.0]

    def test_above_direction(self, simple_indicator):
        """Test 'above' direction generates correct signals."""
        indicator, labels = simple_indicator
        # indicator = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        # Threshold 50: signals where indicator > 50: indices 5,6,7,8,9 (5 signals)

        results = evaluate_threshold_sweep(indicator, labels, [50.0], direction="above")

        assert results["n_signals"][0] == 5
        assert results["coverage"][0] == 0.5

    def test_below_direction(self, simple_indicator):
        """Test 'below' direction generates correct signals."""
        indicator, labels = simple_indicator
        # Threshold 50: signals where indicator < 50: indices 0,1,2,3 (4 signals)

        results = evaluate_threshold_sweep(indicator, labels, [50.0], direction="below")

        assert results["n_signals"][0] == 4
        assert results["coverage"][0] == 0.4

    def test_precision_calculation(self, simple_indicator):
        """Verify precision calculation is correct."""
        indicator, labels = simple_indicator
        # indicator = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        # labels    = [0,  0,  0,  1,  0,  1,  1,  1,  1,  1]
        # Threshold 50 (above): signals at indices 5,6,7,8,9
        # Labels at those indices: 1,1,1,1,1 -> TP=5, FP=0
        # Precision = 5/5 = 1.0

        results = evaluate_threshold_sweep(indicator, labels, [50.0], direction="above")

        assert results["precision"][0] == pytest.approx(1.0, abs=1e-6)

    def test_recall_calculation(self, simple_indicator):
        """Verify recall calculation is correct."""
        indicator, labels = simple_indicator
        # Total positives: indices 3,5,6,7,8,9 = 6
        # Threshold 50 (above): signals at 5,6,7,8,9 -> TP=5
        # Recall = 5/6 = 0.833...

        results = evaluate_threshold_sweep(indicator, labels, [50.0], direction="above")

        assert results["recall"][0] == pytest.approx(5 / 6, abs=1e-6)

    def test_with_returns(self, simple_indicator):
        """Test with returns data included."""
        indicator, labels = simple_indicator
        returns = pl.Series([0.01, -0.01, 0.02, 0.03, -0.02, 0.05, 0.02, 0.01, 0.03, 0.04])

        results = evaluate_threshold_sweep(
            indicator, labels, [50.0], direction="above", returns=returns
        )

        assert "mean_return_on_signal" in results.columns
        assert "mean_return_no_signal" in results.columns
        assert "return_lift" in results.columns

    def test_length_mismatch_raises(self):
        """Mismatched lengths should raise error."""
        indicator = pl.Series([1, 2, 3])
        labels = pl.Series([0, 1])

        with pytest.raises(ValueError, match="same length"):
            evaluate_threshold_sweep(indicator, labels, [1.0])

    def test_empty_thresholds_raises(self, simple_indicator):
        """Empty thresholds should raise error."""
        indicator, labels = simple_indicator

        with pytest.raises(ValueError, match="not be empty"):
            evaluate_threshold_sweep(indicator, labels, [])

    def test_realistic_rsi_sweep(self, rsi_data):
        """Test with realistic RSI-like data."""
        indicator, labels = rsi_data
        thresholds = [30, 40, 50, 60, 70, 80]

        results = evaluate_threshold_sweep(indicator, labels, thresholds, direction="below")

        # RSI below threshold = signal (expecting reversal)
        # Lower thresholds should have fewer signals but potentially higher precision
        assert len(results) == 6
        assert results["coverage"][0] < results["coverage"][-1]  # Lower threshold = less coverage


class TestEvaluatePercentileThresholds:
    """Tests for percentile-based thresholds."""

    def test_default_percentiles(self, momentum_data):
        """Test with default percentiles."""
        indicator, labels = momentum_data

        results = evaluate_percentile_thresholds(indicator, labels)

        # Default percentiles: [10, 25, 50, 75, 90]
        assert len(results) == 5
        assert "percentile" in results.columns
        assert results["percentile"].to_list() == [10.0, 25.0, 50.0, 75.0, 90.0]

    def test_custom_percentiles(self, momentum_data):
        """Test with custom percentiles."""
        indicator, labels = momentum_data
        percentiles = [20.0, 50.0, 80.0]

        results = evaluate_percentile_thresholds(indicator, labels, percentiles=percentiles)

        assert len(results) == 3
        assert results["percentile"].to_list() == [20.0, 50.0, 80.0]

    def test_threshold_values_match_percentiles(self, momentum_data):
        """Threshold values should match indicator percentiles."""
        indicator, labels = momentum_data
        percentiles = [25.0, 50.0, 75.0]

        results = evaluate_percentile_thresholds(indicator, labels, percentiles=percentiles)

        # Verify thresholds match actual percentiles
        expected_p25 = float(indicator.quantile(0.25))
        expected_p50 = float(indicator.quantile(0.50))
        expected_p75 = float(indicator.quantile(0.75))

        assert results["threshold"][0] == pytest.approx(expected_p25, abs=1e-6)
        assert results["threshold"][1] == pytest.approx(expected_p50, abs=1e-6)
        assert results["threshold"][2] == pytest.approx(expected_p75, abs=1e-6)


# ============================================================================
# Optimal Threshold Finding Tests
# ============================================================================


class TestFindOptimalThreshold:
    """Tests for find_optimal_threshold function."""

    def test_basic_optimal_finding(self, monotonic_increasing_results):
        """Find optimal threshold with basic settings."""
        results = monotonic_increasing_results

        optimal = find_optimal_threshold(results, metric="lift")

        assert isinstance(optimal, OptimalThresholdResult)
        assert optimal.found is True
        # Highest lift is at threshold 50
        assert optimal.threshold == 50.0
        assert optimal.lift == 1.4

    def test_min_coverage_constraint(self, monotonic_increasing_results):
        """Min coverage constraint should filter results."""
        results = monotonic_increasing_results
        # Coverage: [0.50, 0.40, 0.30, 0.20, 0.10]

        # With min_coverage=0.25, thresholds 40 (0.20) and 50 (0.10) are excluded
        optimal = find_optimal_threshold(results, metric="lift", min_coverage=0.25)

        assert optimal.found is True
        assert optimal.threshold == 30.0  # Highest lift with coverage >= 0.25
        assert optimal.coverage >= 0.25

    def test_max_coverage_constraint(self, monotonic_increasing_results):
        """Max coverage constraint should filter results."""
        results = monotonic_increasing_results

        # With max_coverage=0.35, thresholds 10 (0.50) and 20 (0.40) are excluded
        optimal = find_optimal_threshold(
            results, metric="lift", min_coverage=0.0, max_coverage=0.35
        )

        assert optimal.found is True
        assert optimal.coverage <= 0.35

    def test_require_significant(self, monotonic_increasing_results):
        """Require significant constraint should filter results."""
        results = monotonic_increasing_results
        # is_significant: [0, 0, 0, 1, 1]

        optimal = find_optimal_threshold(results, metric="lift", require_significant=True)

        assert optimal.found is True
        assert optimal.is_significant is True
        # Only thresholds 40 and 50 are significant, 50 has higher lift
        assert optimal.threshold == 50.0

    def test_no_valid_threshold(self, monotonic_increasing_results):
        """Should return found=False when no threshold meets constraints."""
        results = monotonic_increasing_results

        # Impossible constraints
        optimal = find_optimal_threshold(
            results, metric="lift", min_coverage=0.9, require_significant=True
        )

        assert optimal.found is False
        assert optimal.threshold is None
        assert "No thresholds" in optimal.reason

    def test_missing_metric(self, monotonic_increasing_results):
        """Should handle missing metric gracefully."""
        results = monotonic_increasing_results

        optimal = find_optimal_threshold(results, metric="nonexistent")

        assert optimal.found is False
        assert "not in results" in optimal.reason

    def test_different_metrics(self, monotonic_increasing_results):
        """Should find optimal for different metrics."""
        results = monotonic_increasing_results

        # Precision is highest at threshold 50
        optimal_prec = find_optimal_threshold(results, metric="precision")
        assert optimal_prec.threshold == 50.0

        # Recall is highest at threshold 10
        optimal_rec = find_optimal_threshold(results, metric="recall")
        assert optimal_rec.threshold == 10.0

        # F1 is highest at threshold 20
        optimal_f1 = find_optimal_threshold(results, metric="f1_score")
        assert optimal_f1.threshold == 20.0

    def test_to_dict(self, monotonic_increasing_results):
        """Test to_dict method."""
        results = monotonic_increasing_results
        optimal = find_optimal_threshold(results, metric="lift")

        d = optimal.to_dict()

        assert isinstance(d, dict)
        assert "threshold" in d
        assert "found" in d
        assert "metric" in d


class TestFindThresholdForTargetCoverage:
    """Tests for find_threshold_for_target_coverage function."""

    def test_exact_coverage_match(self, monotonic_increasing_results):
        """Should find threshold with exact coverage match."""
        results = monotonic_increasing_results
        # Coverage: [0.50, 0.40, 0.30, 0.20, 0.10]

        optimal = find_threshold_for_target_coverage(results, target_coverage=0.30)

        assert optimal.found is True
        assert optimal.threshold == 30.0
        assert optimal.coverage == 0.30

    def test_closest_coverage(self, monotonic_increasing_results):
        """Should find threshold closest to target coverage."""
        results = monotonic_increasing_results

        # Target 0.25 is between 0.20 and 0.30
        optimal = find_threshold_for_target_coverage(results, target_coverage=0.25, tolerance=0.1)

        assert optimal.found is True
        # Could be either 30.0 (0.30) or 40.0 (0.20) depending on which is closer
        assert optimal.coverage in [0.20, 0.30]

    def test_no_threshold_in_tolerance(self, monotonic_increasing_results):
        """Should return found=False when no threshold in tolerance."""
        results = monotonic_increasing_results

        # Target 0.70 with small tolerance - no match
        optimal = find_threshold_for_target_coverage(results, target_coverage=0.70, tolerance=0.05)

        assert optimal.found is False
        assert "within" in optimal.reason


# ============================================================================
# Monotonicity Analysis Tests
# ============================================================================


class TestCheckMonotonicity:
    """Tests for check_monotonicity function."""

    def test_monotonic_increasing(self, monotonic_increasing_results):
        """Should detect monotonic increasing pattern."""
        results = monotonic_increasing_results

        mono = check_monotonicity(results, "lift")

        assert isinstance(mono, MonotonicityResult)
        assert mono.is_monotonic is True
        assert mono.is_monotonic_increasing is True
        assert mono.is_monotonic_decreasing is False
        assert mono.direction_changes == 0
        assert len(mono.violations) == 0

    def test_monotonic_decreasing(self, monotonic_increasing_results):
        """Should detect monotonic decreasing pattern (recall)."""
        results = monotonic_increasing_results
        # Recall: [0.90, 0.80, 0.70, 0.60, 0.50] is decreasing

        mono = check_monotonicity(results, "recall")

        assert mono.is_monotonic is True
        assert mono.is_monotonic_increasing is False
        assert mono.is_monotonic_decreasing is True

    def test_non_monotonic(self, non_monotonic_results):
        """Should detect non-monotonic pattern."""
        results = non_monotonic_results
        # Precision: [0.50, 0.60, 0.55, 0.65, 0.58] is non-monotonic

        mono = check_monotonicity(results, "precision")

        assert mono.is_monotonic is False
        assert mono.direction_changes > 0
        assert len(mono.violations) > 0

    def test_violations_list(self, non_monotonic_results):
        """Violations should contain correct information."""
        results = non_monotonic_results

        mono = check_monotonicity(results, "precision")

        # Violations: index 2 (0.60 -> 0.55) and index 4 (0.65 -> 0.58)
        assert len(mono.violations) == 2

        # Each violation is (index, prev_value, curr_value)
        for _idx, prev_val, curr_val in mono.violations:
            assert prev_val > curr_val  # Decrease

    def test_max_violation(self, non_monotonic_results):
        """Max violation should be computed correctly."""
        results = non_monotonic_results

        mono = check_monotonicity(results, "precision")

        # Violations: 0.60->0.55 (diff=0.05) and 0.65->0.58 (diff=0.07)
        assert mono.max_violation == pytest.approx(0.07, abs=1e-6)

    def test_missing_metric_raises(self, monotonic_increasing_results):
        """Should raise for missing metric."""
        results = monotonic_increasing_results

        with pytest.raises(ValueError, match="not in results"):
            check_monotonicity(results, "nonexistent")

    def test_single_row(self):
        """Single row should be monotonic."""
        results = pl.DataFrame(
            {
                "threshold": [50.0],
                "lift": [1.5],
            }
        )

        mono = check_monotonicity(results, "lift")

        assert mono.is_monotonic is True

    def test_to_dict(self, monotonic_increasing_results):
        """Test to_dict method."""
        mono = check_monotonicity(monotonic_increasing_results, "lift")

        d = mono.to_dict()

        assert isinstance(d, dict)
        assert "is_monotonic" in d
        assert "direction_changes" in d


class TestAnalyzeAllMetricsMonotonicity:
    """Tests for analyze_all_metrics_monotonicity function."""

    def test_analyze_multiple_metrics(self, non_monotonic_results):
        """Should analyze all specified metrics."""
        results = non_monotonic_results

        analysis = analyze_all_metrics_monotonicity(results)

        assert isinstance(analysis, dict)
        assert "precision" in analysis
        assert "recall" in analysis
        assert "lift" in analysis
        assert "f1_score" in analysis
        assert "coverage" in analysis

    def test_custom_metrics(self, non_monotonic_results):
        """Should respect custom metrics list."""
        results = non_monotonic_results

        analysis = analyze_all_metrics_monotonicity(results, metrics=["lift", "recall"])

        assert len(analysis) == 2
        assert "lift" in analysis
        assert "recall" in analysis
        assert "precision" not in analysis


# ============================================================================
# Sensitivity Analysis Tests
# ============================================================================


class TestAnalyzeThresholdSensitivity:
    """Tests for analyze_threshold_sensitivity function."""

    def test_basic_sensitivity(self, monotonic_increasing_results):
        """Basic sensitivity analysis."""
        results = monotonic_increasing_results

        sens = analyze_threshold_sensitivity(results, "lift")

        assert isinstance(sens, SensitivityResult)
        assert sens.metric == "lift"
        assert sens.mean_value > 0
        assert sens.std_value >= 0
        assert sens.min_value <= sens.max_value
        assert sens.range_value == sens.max_value - sens.min_value

    def test_coefficient_of_variation(self, monotonic_increasing_results):
        """CV should be computed correctly."""
        results = monotonic_increasing_results
        # Lift: [1.0, 1.1, 1.2, 1.3, 1.4]

        sens = analyze_threshold_sensitivity(results, "lift")

        # Manual calculation
        values = [1.0, 1.1, 1.2, 1.3, 1.4]
        expected_mean = np.mean(values)
        expected_std = np.std(values, ddof=1)  # Polars uses ddof=1
        expected_cv = expected_std / expected_mean

        assert sens.coefficient_of_variation == pytest.approx(expected_cv, abs=1e-4)

    def test_stability_detection(self):
        """Should correctly detect stable vs variable metrics."""
        # Stable metric (low CV)
        stable_results = pl.DataFrame(
            {
                "threshold": [10.0, 20.0, 30.0, 40.0, 50.0],
                "stable_metric": [0.50, 0.51, 0.49, 0.50, 0.51],  # Low variance
                "variable_metric": [0.2, 0.8, 0.3, 0.9, 0.4],  # High variance
            }
        )

        sens_stable = analyze_threshold_sensitivity(stable_results, "stable_metric")
        sens_variable = analyze_threshold_sensitivity(stable_results, "variable_metric")

        assert sens_stable.is_stable is True
        assert sens_variable.is_stable is False

    def test_missing_metric_raises(self, monotonic_increasing_results):
        """Should raise for missing metric."""
        results = monotonic_increasing_results

        with pytest.raises(ValueError, match="not in results"):
            analyze_threshold_sensitivity(results, "nonexistent")

    def test_to_dict(self, monotonic_increasing_results):
        """Test to_dict method."""
        sens = analyze_threshold_sensitivity(monotonic_increasing_results, "lift")

        d = sens.to_dict()

        assert isinstance(d, dict)
        assert "mean" in d
        assert "std" in d
        assert "cv" in d
        assert "is_stable" in d


# ============================================================================
# Summary and Formatting Tests
# ============================================================================


class TestCreateThresholdAnalysisSummary:
    """Tests for create_threshold_analysis_summary function."""

    def test_summary_structure(self, monotonic_increasing_results):
        """Summary should have expected structure."""
        results = monotonic_increasing_results

        summary = create_threshold_analysis_summary(results)

        assert isinstance(summary, ThresholdAnalysisSummary)
        assert summary.n_thresholds == 5
        assert isinstance(summary.optimal, OptimalThresholdResult)
        assert isinstance(summary.monotonicity, dict)
        assert isinstance(summary.sensitivity, dict)
        assert summary.significant_count == 2  # Two significant thresholds

    def test_best_per_metric(self, monotonic_increasing_results):
        """Should compute best threshold per metric."""
        results = monotonic_increasing_results

        summary = create_threshold_analysis_summary(results)

        assert "lift" in summary.best_per_metric
        assert "precision" in summary.best_per_metric
        assert "recall" in summary.best_per_metric

        # Verify correct values
        assert summary.best_per_metric["lift"] == 50.0  # Highest lift
        assert summary.best_per_metric["recall"] == 10.0  # Highest recall

    def test_to_dict(self, monotonic_increasing_results):
        """Test to_dict method."""
        summary = create_threshold_analysis_summary(monotonic_increasing_results)

        d = summary.to_dict()

        assert isinstance(d, dict)
        assert "n_thresholds" in d
        assert "optimal" in d
        assert "monotonicity" in d
        assert "sensitivity" in d


class TestFormatThresholdAnalysis:
    """Tests for format_threshold_analysis function."""

    def test_format_output(self, monotonic_increasing_results):
        """Should produce formatted string output."""
        summary = create_threshold_analysis_summary(monotonic_increasing_results)

        formatted = format_threshold_analysis(summary)

        assert isinstance(formatted, str)
        assert "Threshold Analysis Summary" in formatted
        assert "Optimal Threshold" in formatted
        assert "Monotonicity" in formatted
        assert "Sensitivity" in formatted

    def test_format_with_not_found(self, monotonic_increasing_results):
        """Should handle case when optimal not found."""
        # Create impossible constraints
        results = monotonic_increasing_results.filter(pl.col("coverage") < 0.01)
        summary = create_threshold_analysis_summary(results)

        formatted = format_threshold_analysis(summary)

        # Should still produce output
        assert isinstance(formatted, str)


# ============================================================================
# Integration Tests
# ============================================================================


class TestEndToEndWorkflow:
    """End-to-end integration tests."""

    def test_full_workflow(self, momentum_data):
        """Test complete workflow from raw data to analysis."""
        indicator, labels = momentum_data

        # Step 1: Sweep thresholds
        thresholds = list(np.linspace(-3, 3, 13))
        results = evaluate_threshold_sweep(indicator, labels, thresholds, direction="above")

        # Verify results structure
        assert len(results) == 13
        assert all(results["coverage"] >= 0)
        assert all(results["coverage"] <= 1)

        # Step 2: Find optimal threshold
        optimal = find_optimal_threshold(results, metric="lift", min_coverage=0.05)

        assert optimal.found is True
        assert optimal.coverage >= 0.05

        # Step 3: Check monotonicity
        check_monotonicity(results, "precision")

        # Step 4: Analyze sensitivity
        analyze_threshold_sensitivity(results, "lift")

        # Step 5: Create summary
        summary = create_threshold_analysis_summary(results, optimize_metric="lift")

        assert summary.n_thresholds == 13
        assert summary.optimal.found is True

    def test_percentile_workflow(self, rsi_data):
        """Test workflow with percentile-based thresholds."""
        indicator, labels = rsi_data

        # Use percentiles instead of absolute values
        results = evaluate_percentile_thresholds(
            indicator, labels, percentiles=[10, 30, 50, 70, 90], direction="above"
        )

        assert len(results) == 5
        assert "percentile" in results.columns

        # Find optimal
        optimal = find_optimal_threshold(results, metric="lift")
        assert optimal.found is True

    def test_with_returns_analysis(self, momentum_data):
        """Test workflow with returns analysis."""
        indicator, labels = momentum_data

        # Generate synthetic returns
        np.random.seed(42)
        returns = pl.Series(np.random.normal(0.001, 0.02, len(indicator)))

        thresholds = [-1.0, 0.0, 1.0, 2.0]
        results = evaluate_threshold_sweep(
            indicator, labels, thresholds, direction="above", returns=returns
        )

        assert "mean_return_on_signal" in results.columns
        assert "return_lift" in results.columns


# ============================================================================
# Edge Case Tests
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_constant_indicator(self):
        """Constant indicator should handle gracefully."""
        indicator = pl.Series([50.0] * 100)
        labels = pl.Series([1, 0] * 50)

        # All values same, so all thresholds give same signals
        results = evaluate_threshold_sweep(indicator, labels, [40.0, 50.0, 60.0])

        # Should not raise, but coverage will be 0 or 1
        assert len(results) == 3

    def test_all_positive_labels(self):
        """All positive labels should handle gracefully."""
        indicator = pl.Series([10.0, 20.0, 30.0, 40.0, 50.0])
        labels = pl.Series([1, 1, 1, 1, 1])

        results = evaluate_threshold_sweep(indicator, labels, [25.0])

        # Recall should be computed, lift might be NaN (base_rate = 1)
        assert len(results) == 1

    def test_all_negative_labels(self):
        """All negative labels should handle gracefully."""
        indicator = pl.Series([10.0, 20.0, 30.0, 40.0, 50.0])
        labels = pl.Series([0, 0, 0, 0, 0])

        results = evaluate_threshold_sweep(indicator, labels, [25.0])

        # Recall should be NaN (no positives)
        assert len(results) == 1

    def test_nan_handling_in_monotonicity(self):
        """Monotonicity should handle NaN values."""
        results = pl.DataFrame(
            {
                "threshold": [10.0, 20.0, 30.0, 40.0],
                "lift": [1.0, float("nan"), 1.2, 1.3],
            }
        )

        mono = check_monotonicity(results, "lift")

        # Should not raise, should analyze valid values only
        assert isinstance(mono, MonotonicityResult)

    def test_single_threshold(self):
        """Single threshold should work."""
        indicator = pl.Series([10.0, 50.0, 90.0])
        labels = pl.Series([0, 1, 1])

        results = evaluate_threshold_sweep(indicator, labels, [50.0])

        assert len(results) == 1
        assert results["threshold"][0] == 50.0

    def test_extreme_thresholds(self, momentum_data):
        """Extreme thresholds (0% or 100% coverage) should work."""
        indicator, labels = momentum_data

        # Very low threshold -> 100% coverage
        # Very high threshold -> 0% coverage
        min_val = float(indicator.min())
        max_val = float(indicator.max())

        results = evaluate_threshold_sweep(indicator, labels, [min_val - 1, max_val + 1])

        # First threshold should give ~100% coverage (all above)
        # Second threshold should give ~0% coverage (none above)
        assert results["coverage"][0] > 0.95
        assert results["coverage"][1] < 0.05
