"""Week 2 tests for BarrierAnalysis - Precision/Recall and Time-to-Target.

Tests cover:
- PrecisionRecallResult dataclass creation and serialization
- TimeToTargetResult dataclass creation and serialization
- compute_precision_recall() implementation
- compute_time_to_target() implementation
- Updated tear sheet with new components
- F1 score and lift calculations
- Time analysis correctness
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from ml4t.diagnostic.config.barrier_config import (
    AnalysisSettings,
    BarrierConfig,
)
from ml4t.diagnostic.evaluation.barrier_analysis import BarrierAnalysis
from ml4t.diagnostic.results.barrier_results import (
    PrecisionRecallResult,
    TimeToTargetResult,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_dates() -> list[date]:
    """Generate 100 trading dates."""
    start = date(2020, 1, 1)
    return [start + timedelta(days=i) for i in range(100)]


@pytest.fixture
def sample_assets() -> list[str]:
    """Generate 10 assets."""
    return [f"ASSET_{i:02d}" for i in range(10)]


@pytest.fixture
def signal_data(sample_dates: list[date], sample_assets: list[str]) -> pl.DataFrame:
    """Create synthetic signal data with unique date/asset pairs."""
    np.random.seed(42)
    rows = []
    for d in sample_dates:
        for asset in sample_assets:
            signal = np.random.uniform(-1, 1)
            rows.append({"date": d, "asset": asset, "signal": signal})
    return pl.DataFrame(rows)


@pytest.fixture
def barrier_labels_with_bars(sample_dates: list[date], sample_assets: list[str]) -> pl.DataFrame:
    """Create synthetic barrier labels with label_bars column.

    Labels are correlated with signal values for meaningful precision/recall.
    """
    np.random.seed(42)
    rows = []
    for d in sample_dates:
        for asset in sample_assets:
            signal = np.random.uniform(-1, 1)

            # Probability of TP increases with signal
            p_tp = 0.3 + 0.4 * (signal + 1) / 2
            p_sl = 0.3 - 0.2 * (signal + 1) / 2
            p_timeout = 1 - p_tp - p_sl

            outcome = np.random.choice([1, -1, 0], p=[p_tp, p_sl, p_timeout])

            # Return based on outcome
            if outcome == 1:
                ret = np.random.uniform(0.01, 0.03)
                bars = np.random.randint(5, 15)  # TP tends to be faster
            elif outcome == -1:
                ret = np.random.uniform(-0.02, -0.01)
                bars = np.random.randint(3, 10)  # SL can be fast
            else:
                ret = np.random.uniform(-0.005, 0.005)
                bars = np.random.randint(15, 25)  # Timeout is longer

            rows.append(
                {
                    "date": d,
                    "asset": asset,
                    "label": outcome,
                    "label_return": ret,
                    "label_bars": bars,
                }
            )
    return pl.DataFrame(rows)


@pytest.fixture
def barrier_labels_no_bars(sample_dates: list[date], sample_assets: list[str]) -> pl.DataFrame:
    """Create barrier labels without label_bars column."""
    np.random.seed(42)
    rows = []
    for d in sample_dates:
        for asset in sample_assets:
            signal = np.random.uniform(-1, 1)
            p_tp = 0.3 + 0.4 * (signal + 1) / 2
            p_sl = 0.3 - 0.2 * (signal + 1) / 2
            p_timeout = 1 - p_tp - p_sl
            outcome = np.random.choice([1, -1, 0], p=[p_tp, p_sl, p_timeout])
            ret = 0.02 if outcome == 1 else (-0.01 if outcome == -1 else 0.0)
            rows.append(
                {
                    "date": d,
                    "asset": asset,
                    "label": outcome,
                    "label_return": ret,
                }
            )
    return pl.DataFrame(rows)


@pytest.fixture
def analysis_with_bars(
    signal_data: pl.DataFrame,
    barrier_labels_with_bars: pl.DataFrame,
) -> BarrierAnalysis:
    """Create BarrierAnalysis instance with label_bars."""
    return BarrierAnalysis(signal_data, barrier_labels_with_bars)


@pytest.fixture
def analysis_without_bars(
    signal_data: pl.DataFrame,
    barrier_labels_no_bars: pl.DataFrame,
) -> BarrierAnalysis:
    """Create BarrierAnalysis instance without label_bars."""
    return BarrierAnalysis(signal_data, barrier_labels_no_bars)


# =============================================================================
# PrecisionRecallResult Tests
# =============================================================================


class TestPrecisionRecallResult:
    """Tests for PrecisionRecallResult dataclass."""

    def test_result_creation(self) -> None:
        """Test creating PrecisionRecallResult with valid data."""
        result = PrecisionRecallResult(
            n_quantiles=5,
            quantile_labels=["D1", "D2", "D3", "D4", "D5"],
            precision_tp={"D1": 0.2, "D2": 0.3, "D3": 0.4, "D4": 0.5, "D5": 0.6},
            recall_tp={"D1": 0.1, "D2": 0.15, "D3": 0.2, "D4": 0.25, "D5": 0.3},
            cumulative_precision_tp={"D1": 0.2, "D2": 0.25, "D3": 0.3, "D4": 0.35, "D5": 0.4},
            cumulative_recall_tp={"D1": 0.3, "D2": 0.45, "D3": 0.65, "D4": 0.85, "D5": 1.0},
            cumulative_f1_tp={"D1": 0.24, "D2": 0.32, "D3": 0.4, "D4": 0.5, "D5": 0.57},
            lift_tp={"D1": 0.5, "D2": 0.75, "D3": 1.0, "D4": 1.25, "D5": 1.5},
            cumulative_lift_tp={"D1": 0.5, "D2": 0.63, "D3": 0.75, "D4": 0.88, "D5": 1.0},
            baseline_tp_rate=0.4,
            total_tp_count=400,
            n_observations=1000,
            best_f1_quantile="D5",
            best_f1_score=0.57,
        )
        assert result.n_quantiles == 5
        assert result.baseline_tp_rate == 0.4
        assert result.best_f1_quantile == "D5"

    def test_get_dataframe_default(self) -> None:
        """Test get_dataframe returns precision_recall view."""
        result = PrecisionRecallResult(
            n_quantiles=3,
            quantile_labels=["D1", "D2", "D3"],
            precision_tp={"D1": 0.3, "D2": 0.4, "D3": 0.5},
            recall_tp={"D1": 0.1, "D2": 0.2, "D3": 0.3},
            cumulative_precision_tp={"D1": 0.5, "D2": 0.45, "D3": 0.4},
            cumulative_recall_tp={"D1": 0.3, "D2": 0.5, "D3": 1.0},
            cumulative_f1_tp={"D1": 0.37, "D2": 0.47, "D3": 0.57},
            lift_tp={"D1": 0.75, "D2": 1.0, "D3": 1.25},
            cumulative_lift_tp={"D1": 1.25, "D2": 1.12, "D3": 1.0},
            baseline_tp_rate=0.4,
            total_tp_count=120,
            n_observations=300,
            best_f1_quantile="D3",
            best_f1_score=0.57,
        )
        df = result.get_dataframe()
        assert "quantile" in df.columns
        assert "precision_tp" in df.columns
        assert "recall_tp" in df.columns
        assert df.height == 3

    def test_get_dataframe_cumulative(self) -> None:
        """Test get_dataframe cumulative view."""
        result = PrecisionRecallResult(
            n_quantiles=3,
            quantile_labels=["D1", "D2", "D3"],
            precision_tp={"D1": 0.3, "D2": 0.4, "D3": 0.5},
            recall_tp={"D1": 0.1, "D2": 0.2, "D3": 0.3},
            cumulative_precision_tp={"D1": 0.5, "D2": 0.45, "D3": 0.4},
            cumulative_recall_tp={"D1": 0.3, "D2": 0.5, "D3": 1.0},
            cumulative_f1_tp={"D1": 0.37, "D2": 0.47, "D3": 0.57},
            lift_tp={"D1": 0.75, "D2": 1.0, "D3": 1.25},
            cumulative_lift_tp={"D1": 1.25, "D2": 1.12, "D3": 1.0},
            baseline_tp_rate=0.4,
            total_tp_count=120,
            n_observations=300,
            best_f1_quantile="D3",
            best_f1_score=0.57,
        )
        df = result.get_dataframe("cumulative")
        assert "cumulative_precision_tp" in df.columns
        assert "cumulative_f1_tp" in df.columns

    def test_summary_output(self) -> None:
        """Test summary method returns formatted string."""
        result = PrecisionRecallResult(
            n_quantiles=3,
            quantile_labels=["D1", "D2", "D3"],
            precision_tp={"D1": 0.3, "D2": 0.4, "D3": 0.5},
            recall_tp={"D1": 0.1, "D2": 0.2, "D3": 0.3},
            cumulative_precision_tp={"D1": 0.5, "D2": 0.45, "D3": 0.4},
            cumulative_recall_tp={"D1": 0.3, "D2": 0.5, "D3": 1.0},
            cumulative_f1_tp={"D1": 0.37, "D2": 0.47, "D3": 0.57},
            lift_tp={"D1": 0.75, "D2": 1.0, "D3": 1.25},
            cumulative_lift_tp={"D1": 1.25, "D2": 1.12, "D3": 1.0},
            baseline_tp_rate=0.4,
            total_tp_count=120,
            n_observations=300,
            best_f1_quantile="D3",
            best_f1_score=0.57,
        )
        summary = result.summary()
        assert "Precision/Recall" in summary
        assert "Baseline TP Rate" in summary

    def test_list_available_dataframes(self) -> None:
        """Test list_available_dataframes returns correct list."""
        result = PrecisionRecallResult(
            n_quantiles=3,
            quantile_labels=["D1", "D2", "D3"],
            precision_tp={"D1": 0.3, "D2": 0.4, "D3": 0.5},
            recall_tp={"D1": 0.1, "D2": 0.2, "D3": 0.3},
            cumulative_precision_tp={"D1": 0.5, "D2": 0.45, "D3": 0.4},
            cumulative_recall_tp={"D1": 0.3, "D2": 0.5, "D3": 1.0},
            cumulative_f1_tp={"D1": 0.37, "D2": 0.47, "D3": 0.57},
            lift_tp={"D1": 0.75, "D2": 1.0, "D3": 1.25},
            cumulative_lift_tp={"D1": 1.25, "D2": 1.12, "D3": 1.0},
            baseline_tp_rate=0.4,
            total_tp_count=120,
            n_observations=300,
            best_f1_quantile="D3",
            best_f1_score=0.57,
        )
        available = result.list_available_dataframes()
        assert "precision_recall" in available
        assert "cumulative" in available
        assert "summary" in available


# =============================================================================
# TimeToTargetResult Tests
# =============================================================================


class TestTimeToTargetResult:
    """Tests for TimeToTargetResult dataclass."""

    def test_result_creation(self) -> None:
        """Test creating TimeToTargetResult with valid data."""
        result = TimeToTargetResult(
            n_quantiles=3,
            quantile_labels=["D1", "D2", "D3"],
            mean_bars_tp={"D1": 10.0, "D2": 9.0, "D3": 8.0},
            mean_bars_sl={"D1": 8.0, "D2": 7.0, "D3": 6.0},
            mean_bars_timeout={"D1": 20.0, "D2": 20.0, "D3": 20.0},
            mean_bars_all={"D1": 12.0, "D2": 11.0, "D3": 10.0},
            median_bars_tp={"D1": 9.0, "D2": 8.0, "D3": 7.0},
            median_bars_sl={"D1": 7.0, "D2": 6.0, "D3": 5.0},
            median_bars_all={"D1": 11.0, "D2": 10.0, "D3": 9.0},
            std_bars_tp={"D1": 2.0, "D2": 2.0, "D3": 2.0},
            std_bars_sl={"D1": 1.5, "D2": 1.5, "D3": 1.5},
            std_bars_all={"D1": 3.0, "D2": 3.0, "D3": 3.0},
            count_tp={"D1": 30, "D2": 40, "D3": 50},
            count_sl={"D1": 40, "D2": 35, "D3": 30},
            count_timeout={"D1": 30, "D2": 25, "D3": 20},
            overall_mean_bars=11.0,
            overall_median_bars=10.0,
            overall_mean_bars_tp=9.0,
            overall_mean_bars_sl=7.0,
            n_observations=300,
            tp_faster_than_sl={"D1": False, "D2": False, "D3": False},
            speed_advantage_tp={"D1": -2.0, "D2": -2.0, "D3": -2.0},
        )
        assert result.n_quantiles == 3
        assert result.overall_mean_bars == 11.0
        assert result.n_observations == 300

    def test_get_dataframe_default(self) -> None:
        """Test get_dataframe returns time_to_target view."""
        result = TimeToTargetResult(
            n_quantiles=2,
            quantile_labels=["D1", "D2"],
            mean_bars_tp={"D1": 10.0, "D2": 8.0},
            mean_bars_sl={"D1": 8.0, "D2": 6.0},
            mean_bars_timeout={"D1": 20.0, "D2": 20.0},
            mean_bars_all={"D1": 12.0, "D2": 10.0},
            median_bars_tp={"D1": 9.0, "D2": 7.0},
            median_bars_sl={"D1": 7.0, "D2": 5.0},
            median_bars_all={"D1": 11.0, "D2": 9.0},
            std_bars_tp={"D1": 2.0, "D2": 2.0},
            std_bars_sl={"D1": 1.5, "D2": 1.5},
            std_bars_all={"D1": 3.0, "D2": 3.0},
            count_tp={"D1": 30, "D2": 50},
            count_sl={"D1": 40, "D2": 30},
            count_timeout={"D1": 30, "D2": 20},
            overall_mean_bars=11.0,
            overall_median_bars=10.0,
            overall_mean_bars_tp=9.0,
            overall_mean_bars_sl=7.0,
            n_observations=200,
            tp_faster_than_sl={"D1": False, "D2": False},
            speed_advantage_tp={"D1": -2.0, "D2": -2.0},
        )
        df = result.get_dataframe()
        assert "quantile" in df.columns
        assert "mean_bars_tp" in df.columns
        assert "tp_faster" in df.columns
        assert df.height == 2

    def test_get_dataframe_detailed(self) -> None:
        """Test get_dataframe detailed view."""
        result = TimeToTargetResult(
            n_quantiles=2,
            quantile_labels=["D1", "D2"],
            mean_bars_tp={"D1": 10.0, "D2": 8.0},
            mean_bars_sl={"D1": 8.0, "D2": 6.0},
            mean_bars_timeout={"D1": 20.0, "D2": 20.0},
            mean_bars_all={"D1": 12.0, "D2": 10.0},
            median_bars_tp={"D1": 9.0, "D2": 7.0},
            median_bars_sl={"D1": 7.0, "D2": 5.0},
            median_bars_all={"D1": 11.0, "D2": 9.0},
            std_bars_tp={"D1": 2.0, "D2": 2.0},
            std_bars_sl={"D1": 1.5, "D2": 1.5},
            std_bars_all={"D1": 3.0, "D2": 3.0},
            count_tp={"D1": 30, "D2": 50},
            count_sl={"D1": 40, "D2": 30},
            count_timeout={"D1": 30, "D2": 20},
            overall_mean_bars=11.0,
            overall_median_bars=10.0,
            overall_mean_bars_tp=9.0,
            overall_mean_bars_sl=7.0,
            n_observations=200,
            tp_faster_than_sl={"D1": False, "D2": False},
            speed_advantage_tp={"D1": -2.0, "D2": -2.0},
        )
        df = result.get_dataframe("detailed")
        assert "median_bars_tp" in df.columns
        assert "std_bars_tp" in df.columns
        assert "count_tp" in df.columns

    def test_summary_output(self) -> None:
        """Test summary method returns formatted string."""
        result = TimeToTargetResult(
            n_quantiles=2,
            quantile_labels=["D1", "D2"],
            mean_bars_tp={"D1": 10.0, "D2": 8.0},
            mean_bars_sl={"D1": 8.0, "D2": 6.0},
            mean_bars_timeout={"D1": 20.0, "D2": 20.0},
            mean_bars_all={"D1": 12.0, "D2": 10.0},
            median_bars_tp={"D1": 9.0, "D2": 7.0},
            median_bars_sl={"D1": 7.0, "D2": 5.0},
            median_bars_all={"D1": 11.0, "D2": 9.0},
            std_bars_tp={"D1": 2.0, "D2": 2.0},
            std_bars_sl={"D1": 1.5, "D2": 1.5},
            std_bars_all={"D1": 3.0, "D2": 3.0},
            count_tp={"D1": 30, "D2": 50},
            count_sl={"D1": 40, "D2": 30},
            count_timeout={"D1": 30, "D2": 20},
            overall_mean_bars=11.0,
            overall_median_bars=10.0,
            overall_mean_bars_tp=9.0,
            overall_mean_bars_sl=7.0,
            n_observations=200,
            tp_faster_than_sl={"D1": False, "D2": False},
            speed_advantage_tp={"D1": -2.0, "D2": -2.0},
        )
        summary = result.summary()
        assert "Time-to-Target" in summary
        assert "Mean Bars" in summary


# =============================================================================
# compute_precision_recall() Tests
# =============================================================================


class TestComputePrecisionRecall:
    """Tests for compute_precision_recall method."""

    def test_basic_precision_recall(self, analysis_with_bars: BarrierAnalysis) -> None:
        """Test basic precision/recall computation."""
        result = analysis_with_bars.compute_precision_recall()

        assert isinstance(result, PrecisionRecallResult)
        assert result.n_quantiles == 10
        assert len(result.quantile_labels) == 10
        assert result.n_observations == 1000

    def test_precision_values_in_range(self, analysis_with_bars: BarrierAnalysis) -> None:
        """Test precision values are in valid range [0, 1]."""
        result = analysis_with_bars.compute_precision_recall()

        for q in result.quantile_labels:
            assert 0.0 <= result.precision_tp[q] <= 1.0
            assert 0.0 <= result.recall_tp[q] <= 1.0

    def test_recall_sums_to_one(self, analysis_with_bars: BarrierAnalysis) -> None:
        """Test recall values sum to 1.0 (coverage of all TPs)."""
        result = analysis_with_bars.compute_precision_recall()

        total_recall = sum(result.recall_tp.values())
        assert abs(total_recall - 1.0) < 0.01  # Allow small floating point error

    def test_cumulative_recall_increases(self, analysis_with_bars: BarrierAnalysis) -> None:
        """Test cumulative recall increases monotonically from top quantile."""
        result = analysis_with_bars.compute_precision_recall()

        # Cumulative recall from top (D10) should increase as we include more quantiles
        reversed_labels = list(reversed(result.quantile_labels))
        prev_recall = 0.0
        for q in reversed_labels:
            cum_recall = result.cumulative_recall_tp[q]
            assert cum_recall >= prev_recall
            prev_recall = cum_recall

        # Final cumulative recall should be 1.0
        assert abs(result.cumulative_recall_tp[result.quantile_labels[0]] - 1.0) < 0.01

    def test_lift_values(self, analysis_with_bars: BarrierAnalysis) -> None:
        """Test lift is computed correctly (precision / baseline)."""
        result = analysis_with_bars.compute_precision_recall()

        for q in result.quantile_labels:
            expected_lift = result.precision_tp[q] / result.baseline_tp_rate
            assert abs(result.lift_tp[q] - expected_lift) < 0.001

    def test_f1_calculation(self, analysis_with_bars: BarrierAnalysis) -> None:
        """Test F1 score is harmonic mean of precision and recall."""
        result = analysis_with_bars.compute_precision_recall()

        for q in result.quantile_labels:
            p = result.cumulative_precision_tp[q]
            r = result.cumulative_recall_tp[q]
            if p + r > 0:
                expected_f1 = 2 * p * r / (p + r)
            else:
                expected_f1 = 0.0
            assert abs(result.cumulative_f1_tp[q] - expected_f1) < 0.001

    def test_best_f1_quantile(self, analysis_with_bars: BarrierAnalysis) -> None:
        """Test best_f1_quantile is correct."""
        result = analysis_with_bars.compute_precision_recall()

        # Find best F1 manually
        best_f1 = 0.0
        best_q = result.quantile_labels[-1]
        for q in result.quantile_labels:
            if result.cumulative_f1_tp[q] > best_f1:
                best_f1 = result.cumulative_f1_tp[q]
                best_q = q

        assert result.best_f1_quantile == best_q
        assert abs(result.best_f1_score - best_f1) < 0.001

    def test_caching(self, analysis_with_bars: BarrierAnalysis) -> None:
        """Test that results are cached."""
        result1 = analysis_with_bars.compute_precision_recall()
        result2 = analysis_with_bars.compute_precision_recall()
        assert result1 is result2

    def test_signal_correlation_effect(self) -> None:
        """Test that strong signals have higher precision in top quantile."""
        # Create data where high signals strongly predict TP
        np.random.seed(123)
        dates = [date(2020, 1, 1) + timedelta(days=i) for i in range(50)]
        assets = ["A", "B"]

        signal_rows = []
        barrier_rows = []
        for d in dates:
            for asset in assets:
                signal = np.random.uniform(-1, 1)
                signal_rows.append({"date": d, "asset": asset, "signal": signal})

                # Strong correlation: high signal -> TP
                if signal > 0.5:
                    label = 1  # Very likely TP
                elif signal < -0.5:
                    label = -1  # Very likely SL
                else:
                    label = 0

                barrier_rows.append(
                    {
                        "date": d,
                        "asset": asset,
                        "label": label,
                        "label_return": 0.02 if label == 1 else -0.01,
                        "label_bars": 10,
                    }
                )

        signal_df = pl.DataFrame(signal_rows)
        barrier_df = pl.DataFrame(barrier_rows)

        config = BarrierConfig(analysis=AnalysisSettings(n_quantiles=5))
        analysis = BarrierAnalysis(signal_df, barrier_df, config=config)
        result = analysis.compute_precision_recall()

        # Top quantile (D5) should have higher precision than bottom (D1)
        assert result.precision_tp["D5"] > result.precision_tp["D1"]


# =============================================================================
# compute_time_to_target() Tests
# =============================================================================


class TestComputeTimeToTarget:
    """Tests for compute_time_to_target method."""

    def test_basic_time_to_target(self, analysis_with_bars: BarrierAnalysis) -> None:
        """Test basic time-to-target computation."""
        result = analysis_with_bars.compute_time_to_target()

        assert isinstance(result, TimeToTargetResult)
        assert result.n_quantiles == 10
        assert result.n_observations == 1000

    def test_mean_bars_positive(self, analysis_with_bars: BarrierAnalysis) -> None:
        """Test all mean bars values are non-negative."""
        result = analysis_with_bars.compute_time_to_target()

        for q in result.quantile_labels:
            assert result.mean_bars_tp[q] >= 0.0
            assert result.mean_bars_sl[q] >= 0.0
            assert result.mean_bars_all[q] >= 0.0

    def test_overall_stats(self, analysis_with_bars: BarrierAnalysis) -> None:
        """Test overall statistics are computed."""
        result = analysis_with_bars.compute_time_to_target()

        assert result.overall_mean_bars > 0
        assert result.overall_median_bars > 0
        assert result.overall_mean_bars_tp > 0
        assert result.overall_mean_bars_sl > 0

    def test_speed_analysis(self, analysis_with_bars: BarrierAnalysis) -> None:
        """Test speed advantage calculation."""
        result = analysis_with_bars.compute_time_to_target()

        for q in result.quantile_labels:
            # Speed advantage = mean_sl - mean_tp (positive means TP is faster)
            expected_adv = result.mean_bars_sl[q] - result.mean_bars_tp[q]
            assert abs(result.speed_advantage_tp[q] - expected_adv) < 0.001

            # tp_faster should match the sign
            if result.mean_bars_tp[q] > 0 and result.mean_bars_sl[q] > 0:
                expected_faster = result.mean_bars_tp[q] < result.mean_bars_sl[q]
                assert result.tp_faster_than_sl[q] == expected_faster

    def test_missing_label_bars_raises(self, analysis_without_bars: BarrierAnalysis) -> None:
        """Test that missing label_bars column raises ValueError."""
        with pytest.raises(ValueError, match="label_bars"):
            analysis_without_bars.compute_time_to_target()

    def test_caching(self, analysis_with_bars: BarrierAnalysis) -> None:
        """Test that results are cached."""
        result1 = analysis_with_bars.compute_time_to_target()
        result2 = analysis_with_bars.compute_time_to_target()
        assert result1 is result2

    def test_timeout_bars_longest(self) -> None:
        """Test that timeout outcomes typically have longest bars."""
        # Create data where timeouts are much longer
        np.random.seed(456)
        dates = [date(2020, 1, 1) + timedelta(days=i) for i in range(30)]
        assets = ["A", "B", "C"]

        signal_rows = []
        barrier_rows = []
        for d in dates:
            for asset in assets:
                signal = np.random.uniform(-1, 1)
                signal_rows.append({"date": d, "asset": asset, "signal": signal})

                outcome = np.random.choice([1, -1, 0], p=[0.4, 0.3, 0.3])
                if outcome == 1:
                    bars = np.random.randint(5, 10)  # Fast TP
                elif outcome == -1:
                    bars = np.random.randint(3, 8)  # Fast SL
                else:
                    bars = np.random.randint(20, 30)  # Slow timeout

                barrier_rows.append(
                    {
                        "date": d,
                        "asset": asset,
                        "label": outcome,
                        "label_return": 0.02 if outcome == 1 else -0.01,
                        "label_bars": bars,
                    }
                )

        signal_df = pl.DataFrame(signal_rows)
        barrier_df = pl.DataFrame(barrier_rows)

        analysis = BarrierAnalysis(signal_df, barrier_df)
        result = analysis.compute_time_to_target()

        # Overall timeout bars should be much larger than TP/SL
        for q in result.quantile_labels:
            if result.mean_bars_timeout[q] > 0:
                assert result.mean_bars_timeout[q] > result.mean_bars_tp[q]


# =============================================================================
# Updated Tear Sheet Tests
# =============================================================================


class TestUpdatedTearSheet:
    """Tests for updated BarrierTearSheet with new components."""

    def test_tear_sheet_includes_precision_recall(
        self, analysis_with_bars: BarrierAnalysis
    ) -> None:
        """Test tear sheet includes precision/recall results."""
        tear_sheet = analysis_with_bars.create_tear_sheet()

        assert tear_sheet.precision_recall_result is not None
        assert isinstance(tear_sheet.precision_recall_result, PrecisionRecallResult)

    def test_tear_sheet_includes_time_to_target(self, analysis_with_bars: BarrierAnalysis) -> None:
        """Test tear sheet includes time-to-target results."""
        tear_sheet = analysis_with_bars.create_tear_sheet()

        assert tear_sheet.time_to_target_result is not None
        assert isinstance(tear_sheet.time_to_target_result, TimeToTargetResult)

    def test_tear_sheet_without_time_to_target(
        self, analysis_without_bars: BarrierAnalysis
    ) -> None:
        """Test tear sheet gracefully handles missing label_bars."""
        # Should not raise, just skip time_to_target
        tear_sheet = analysis_without_bars.create_tear_sheet()

        assert tear_sheet.precision_recall_result is not None
        assert tear_sheet.time_to_target_result is None

    def test_tear_sheet_explicit_skip_time_to_target(
        self, analysis_with_bars: BarrierAnalysis
    ) -> None:
        """Test explicitly skipping time-to-target."""
        tear_sheet = analysis_with_bars.create_tear_sheet(include_time_to_target=False)

        assert tear_sheet.precision_recall_result is not None
        assert tear_sheet.time_to_target_result is None

    def test_tear_sheet_summary_includes_new_results(
        self, analysis_with_bars: BarrierAnalysis
    ) -> None:
        """Test summary includes new analysis sections."""
        tear_sheet = analysis_with_bars.create_tear_sheet()
        summary = tear_sheet.summary()

        assert "Precision/Recall" in summary
        assert "Time-to-Target" in summary

    def test_tear_sheet_list_dataframes(self, analysis_with_bars: BarrierAnalysis) -> None:
        """Test list_available_dataframes includes new components."""
        tear_sheet = analysis_with_bars.create_tear_sheet()
        available = tear_sheet.list_available_dataframes()

        assert any("precision_recall" in df for df in available)
        assert any("time_to_target" in df for df in available)

    def test_tear_sheet_get_dataframe_precision_recall(
        self, analysis_with_bars: BarrierAnalysis
    ) -> None:
        """Test getting precision_recall dataframe from tear sheet."""
        tear_sheet = analysis_with_bars.create_tear_sheet()
        df = tear_sheet.get_dataframe("precision_recall_precision_recall")

        assert "precision_tp" in df.columns

    def test_tear_sheet_get_dataframe_time_to_target(
        self, analysis_with_bars: BarrierAnalysis
    ) -> None:
        """Test getting time_to_target dataframe from tear sheet."""
        tear_sheet = analysis_with_bars.create_tear_sheet()
        df = tear_sheet.get_dataframe("time_to_target_time_to_target")

        assert "mean_bars_tp" in df.columns

    def test_tear_sheet_build_summary_df(self, analysis_with_bars: BarrierAnalysis) -> None:
        """Test _build_summary_df includes new metrics."""
        tear_sheet = analysis_with_bars.create_tear_sheet()
        df = tear_sheet.get_dataframe("summary")

        metrics = df["metric"].to_list()
        assert "baseline_tp_rate" in metrics
        assert "best_f1_score" in metrics
        assert "overall_mean_bars" in metrics


# =============================================================================
# Edge Cases and Statistical Correctness
# =============================================================================


class TestEdgeCasesWeek2:
    """Edge case tests for Week 2 features."""

    def test_single_quantile(self) -> None:
        """Test with n_quantiles=2 (minimum)."""
        np.random.seed(789)
        dates = [date(2020, 1, 1) + timedelta(days=i) for i in range(20)]
        assets = ["A"]

        signal_rows = []
        barrier_rows = []
        for d in dates:
            for asset in assets:
                signal = np.random.uniform(-1, 1)
                signal_rows.append({"date": d, "asset": asset, "signal": signal})
                outcome = np.random.choice([1, -1, 0])
                barrier_rows.append(
                    {
                        "date": d,
                        "asset": asset,
                        "label": outcome,
                        "label_return": 0.01,
                        "label_bars": 10,
                    }
                )

        config = BarrierConfig(analysis=AnalysisSettings(n_quantiles=2))
        analysis = BarrierAnalysis(
            pl.DataFrame(signal_rows),
            pl.DataFrame(barrier_rows),
            config=config,
        )

        pr_result = analysis.compute_precision_recall()
        assert pr_result.n_quantiles == 2

        ttt_result = analysis.compute_time_to_target()
        assert ttt_result.n_quantiles == 2

    def test_all_tp_outcomes(self) -> None:
        """Test when all outcomes are TP."""
        np.random.seed(101)
        dates = [date(2020, 1, 1) + timedelta(days=i) for i in range(10)]
        assets = ["A", "B"]

        signal_rows = []
        barrier_rows = []
        for d in dates:
            for asset in assets:
                signal = np.random.uniform(-1, 1)
                signal_rows.append({"date": d, "asset": asset, "signal": signal})
                barrier_rows.append(
                    {
                        "date": d,
                        "asset": asset,
                        "label": 1,  # All TP
                        "label_return": 0.02,
                        "label_bars": np.random.randint(5, 15),
                    }
                )

        analysis = BarrierAnalysis(
            pl.DataFrame(signal_rows),
            pl.DataFrame(barrier_rows),
        )

        pr_result = analysis.compute_precision_recall()
        # All precision should be 1.0 (everything is TP)
        for q in pr_result.quantile_labels:
            assert pr_result.precision_tp[q] == 1.0
        assert pr_result.baseline_tp_rate == 1.0

    def test_no_tp_outcomes(self) -> None:
        """Test when no outcomes are TP."""
        np.random.seed(102)
        dates = [date(2020, 1, 1) + timedelta(days=i) for i in range(10)]
        assets = ["A", "B"]

        signal_rows = []
        barrier_rows = []
        for d in dates:
            for asset in assets:
                signal = np.random.uniform(-1, 1)
                signal_rows.append({"date": d, "asset": asset, "signal": signal})
                outcome = np.random.choice([-1, 0])  # Only SL and timeout
                barrier_rows.append(
                    {
                        "date": d,
                        "asset": asset,
                        "label": outcome,
                        "label_return": -0.01 if outcome == -1 else 0.0,
                        "label_bars": 10,
                    }
                )

        analysis = BarrierAnalysis(
            pl.DataFrame(signal_rows),
            pl.DataFrame(barrier_rows),
        )

        pr_result = analysis.compute_precision_recall()
        # All precision should be 0.0 (no TP)
        for q in pr_result.quantile_labels:
            assert pr_result.precision_tp[q] == 0.0
        assert pr_result.baseline_tp_rate == 0.0
        assert pr_result.total_tp_count == 0

    def test_empty_quantile_time_to_target(self) -> None:
        """Test time-to-target with quantiles having no data for some outcomes."""
        # Create data where some quantiles have only TP, no SL
        np.random.seed(103)
        dates = [date(2020, 1, 1) + timedelta(days=i) for i in range(20)]
        assets = ["A"]

        signal_rows = []
        barrier_rows = []
        for i, d in enumerate(dates):
            for asset in assets:
                signal = i / 20.0  # Linearly increasing signal
                signal_rows.append({"date": d, "asset": asset, "signal": signal})

                # High signals -> all TP, low signals -> all SL
                if i >= 15:
                    outcome = 1
                elif i < 5:
                    outcome = -1
                else:
                    outcome = 0

                barrier_rows.append(
                    {
                        "date": d,
                        "asset": asset,
                        "label": outcome,
                        "label_return": 0.02 if outcome == 1 else -0.01,
                        "label_bars": 10,
                    }
                )

        config = BarrierConfig(analysis=AnalysisSettings(n_quantiles=4))
        analysis = BarrierAnalysis(
            pl.DataFrame(signal_rows),
            pl.DataFrame(barrier_rows),
            config=config,
        )

        result = analysis.compute_time_to_target()
        # Should handle gracefully without errors
        assert result.n_observations == 20
