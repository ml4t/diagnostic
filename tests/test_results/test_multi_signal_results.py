"""Tests for MultiSignalResults module result classes.

These tests focus on the result class methods:
- get_dataframe()
- summary()
- list_available_dataframes()
- get_significant_signals()
- get_ranking()
- filter_signals()
- get_tear_sheet_data()
- get_pairwise_correlation()
"""

from __future__ import annotations

import polars as pl
import pytest

from ml4t.diagnostic.results.multi_signal_results import (
    ComparisonResult,
    MultiSignalSummary,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_multi_signal_summary() -> MultiSignalSummary:
    """Create a sample MultiSignalSummary for testing."""
    return MultiSignalSummary(
        summary_data={
            "signal_name": ["sig_1", "sig_2", "sig_3", "sig_4", "sig_5"],
            "ic_mean": [0.08, 0.05, 0.03, 0.01, -0.02],
            "ic_std": [0.02, 0.02, 0.02, 0.02, 0.02],
            "ic_ir": [4.0, 2.5, 1.5, 0.5, -1.0],
            "ic_t_stat": [4.0, 2.5, 1.5, 0.5, -1.0],
            "ic_p_value": [0.0001, 0.02, 0.15, 0.6, 0.3],
            "fdr_p_value": [0.0005, 0.05, 0.25, 0.75, 0.5],
            "fwer_p_value": [0.0005, 0.10, 0.50, 1.0, 1.0],
            "fdr_significant": [True, True, False, False, False],
            "fwer_significant": [True, False, False, False, False],
            "turnover_mean": [0.15, 0.20, 0.25, 0.30, 0.35],
        },
        n_signals=5,
        n_fdr_significant=2,
        n_fwer_significant=1,
        periods=(1, 5, 21),
        fdr_alpha=0.05,
        fwer_alpha=0.05,
    )


@pytest.fixture
def sample_multi_signal_summary_with_correlation() -> MultiSignalSummary:
    """Create MultiSignalSummary with correlation data."""
    return MultiSignalSummary(
        summary_data={
            "signal_name": ["sig_1", "sig_2", "sig_3"],
            "ic_mean": [0.08, 0.05, 0.03],
            "ic_ir": [4.0, 2.5, 1.5],
            "fdr_significant": [True, True, False],
            "fwer_significant": [True, False, False],
        },
        n_signals=3,
        n_fdr_significant=2,
        n_fwer_significant=1,
        periods=(1, 5),
        fdr_alpha=0.05,
        fwer_alpha=0.05,
        correlation_data={
            "sig_1": [1.0, 0.5, 0.2],
            "sig_2": [0.5, 1.0, 0.3],
            "sig_3": [0.2, 0.3, 1.0],
        },
    )


@pytest.fixture
def sample_comparison_result() -> ComparisonResult:
    """Create a sample ComparisonResult for testing."""
    return ComparisonResult(
        signals=["sig_1", "sig_2", "sig_3"],
        selection_method="top_n",
        selection_params={"n": 3, "metric": "ic_ir"},
        tear_sheets={
            "sig_1": {
                "signal_name": "sig_1",
                "n_assets": 30,
                "n_dates": 100,
                "ic_analysis": {
                    "ic_mean": {"1D": 0.08, "5D": 0.10},
                    "ic_ir": {"1D": 4.0, "5D": 3.5},
                },
            },
            "sig_2": {
                "signal_name": "sig_2",
                "n_assets": 30,
                "n_dates": 100,
                "ic_analysis": {
                    "ic_mean": {"1D": 0.05, "5D": 0.07},
                    "ic_ir": {"1D": 2.5, "5D": 2.2},
                },
            },
            "sig_3": {
                "signal_name": "sig_3",
                "n_assets": 30,
                "n_dates": 100,
                "ic_analysis": {
                    "ic_mean": {"1D": 0.03, "5D": 0.04},
                    "ic_ir": {"1D": 1.5, "5D": 1.3},
                },
            },
        },
        correlation_matrix={
            "sig_1": [1.0, 0.5, 0.2],
            "sig_2": [0.5, 1.0, 0.3],
            "sig_3": [0.2, 0.3, 1.0],
        },
    )


@pytest.fixture
def comparison_result_no_ic() -> ComparisonResult:
    """Create ComparisonResult with tear sheets without IC analysis."""
    return ComparisonResult(
        signals=["sig_a", "sig_b"],
        selection_method="manual",
        selection_params={},
        tear_sheets={
            "sig_a": {"signal_name": "sig_a", "n_assets": 10, "n_dates": 50},
            "sig_b": {"signal_name": "sig_b", "n_assets": 10, "n_dates": 50},
        },
        correlation_matrix={
            "sig_a": [1.0, 0.4],
            "sig_b": [0.4, 1.0],
        },
    )


# =============================================================================
# MultiSignalSummary Tests
# =============================================================================


class TestMultiSignalSummaryGetDataframe:
    """Tests for MultiSignalSummary.get_dataframe method."""

    def test_get_dataframe_default(self, sample_multi_signal_summary: MultiSignalSummary) -> None:
        """Test get_dataframe with default (summary)."""
        df = sample_multi_signal_summary.get_dataframe()

        assert isinstance(df, pl.DataFrame)
        assert "signal_name" in df.columns
        assert "ic_mean" in df.columns
        assert "ic_ir" in df.columns
        assert "fdr_significant" in df.columns
        assert len(df) == 5

    def test_get_dataframe_summary(self, sample_multi_signal_summary: MultiSignalSummary) -> None:
        """Test get_dataframe with name='summary'."""
        df = sample_multi_signal_summary.get_dataframe("summary")

        assert isinstance(df, pl.DataFrame)
        assert len(df) == 5

    def test_get_dataframe_correlation_available(
        self, sample_multi_signal_summary_with_correlation: MultiSignalSummary
    ) -> None:
        """Test get_dataframe with name='correlation' when available."""
        df = sample_multi_signal_summary_with_correlation.get_dataframe("correlation")

        assert isinstance(df, pl.DataFrame)
        assert "sig_1" in df.columns
        assert "sig_2" in df.columns
        assert "sig_3" in df.columns

    def test_get_dataframe_correlation_not_available(
        self, sample_multi_signal_summary: MultiSignalSummary
    ) -> None:
        """Test get_dataframe raises when correlation not available."""
        with pytest.raises(ValueError, match="Correlation data not computed"):
            sample_multi_signal_summary.get_dataframe("correlation")

    def test_get_dataframe_invalid_name(
        self, sample_multi_signal_summary: MultiSignalSummary
    ) -> None:
        """Test get_dataframe raises for invalid name."""
        with pytest.raises(ValueError, match="Unknown DataFrame"):
            sample_multi_signal_summary.get_dataframe("invalid")


class TestMultiSignalSummaryListDataframes:
    """Tests for MultiSignalSummary.list_available_dataframes method."""

    def test_list_available_dataframes_without_correlation(
        self, sample_multi_signal_summary: MultiSignalSummary
    ) -> None:
        """Test list_available_dataframes without correlation."""
        available = sample_multi_signal_summary.list_available_dataframes()

        assert isinstance(available, list)
        assert "summary" in available
        assert "correlation" not in available

    def test_list_available_dataframes_with_correlation(
        self, sample_multi_signal_summary_with_correlation: MultiSignalSummary
    ) -> None:
        """Test list_available_dataframes with correlation."""
        available = sample_multi_signal_summary_with_correlation.list_available_dataframes()

        assert isinstance(available, list)
        assert "summary" in available
        assert "correlation" in available


class TestMultiSignalSummaryGetSignificantSignals:
    """Tests for MultiSignalSummary.get_significant_signals method."""

    def test_get_significant_signals_fdr(
        self, sample_multi_signal_summary: MultiSignalSummary
    ) -> None:
        """Test get_significant_signals with FDR method."""
        signals = sample_multi_signal_summary.get_significant_signals(method="fdr")

        assert isinstance(signals, list)
        assert len(signals) == 2
        assert "sig_1" in signals
        assert "sig_2" in signals

    def test_get_significant_signals_fwer(
        self, sample_multi_signal_summary: MultiSignalSummary
    ) -> None:
        """Test get_significant_signals with FWER method."""
        signals = sample_multi_signal_summary.get_significant_signals(method="fwer")

        assert isinstance(signals, list)
        assert len(signals) == 1
        assert "sig_1" in signals

    def test_get_significant_signals_invalid_method(
        self, sample_multi_signal_summary: MultiSignalSummary
    ) -> None:
        """Test get_significant_signals raises for invalid method."""
        with pytest.raises(ValueError, match="not in summary data"):
            sample_multi_signal_summary.get_significant_signals(method="invalid")


class TestMultiSignalSummaryGetRanking:
    """Tests for MultiSignalSummary.get_ranking method."""

    def test_get_ranking_descending(self, sample_multi_signal_summary: MultiSignalSummary) -> None:
        """Test get_ranking with descending order (default)."""
        ranking = sample_multi_signal_summary.get_ranking(metric="ic_ir")

        assert isinstance(ranking, list)
        assert len(ranking) == 5
        assert ranking[0] == "sig_1"  # Highest IC IR
        assert ranking[4] == "sig_5"  # Lowest IC IR

    def test_get_ranking_ascending(self, sample_multi_signal_summary: MultiSignalSummary) -> None:
        """Test get_ranking with ascending order."""
        ranking = sample_multi_signal_summary.get_ranking(metric="ic_ir", ascending=True)

        assert isinstance(ranking, list)
        assert ranking[0] == "sig_5"  # Lowest IC IR
        assert ranking[4] == "sig_1"  # Highest IC IR

    def test_get_ranking_with_n(self, sample_multi_signal_summary: MultiSignalSummary) -> None:
        """Test get_ranking with limited n."""
        ranking = sample_multi_signal_summary.get_ranking(metric="ic_ir", n=3)

        assert isinstance(ranking, list)
        assert len(ranking) == 3
        assert ranking[0] == "sig_1"

    def test_get_ranking_different_metric(
        self, sample_multi_signal_summary: MultiSignalSummary
    ) -> None:
        """Test get_ranking with different metric."""
        ranking = sample_multi_signal_summary.get_ranking(metric="ic_mean")

        assert isinstance(ranking, list)
        assert ranking[0] == "sig_1"  # Highest IC mean


class TestMultiSignalSummaryFilterSignals:
    """Tests for MultiSignalSummary.filter_signals method."""

    def test_filter_signals_min_ic(self, sample_multi_signal_summary: MultiSignalSummary) -> None:
        """Test filter_signals with min_ic."""
        df = sample_multi_signal_summary.filter_signals(min_ic=0.04)

        assert isinstance(df, pl.DataFrame)
        assert len(df) == 2
        signal_names = df["signal_name"].to_list()
        assert "sig_1" in signal_names
        assert "sig_2" in signal_names

    def test_filter_signals_min_ic_ir(
        self, sample_multi_signal_summary: MultiSignalSummary
    ) -> None:
        """Test filter_signals with min_ic_ir."""
        df = sample_multi_signal_summary.filter_signals(min_ic_ir=2.0)

        assert isinstance(df, pl.DataFrame)
        assert len(df) == 2

    def test_filter_signals_max_turnover(
        self, sample_multi_signal_summary: MultiSignalSummary
    ) -> None:
        """Test filter_signals with max_turnover."""
        df = sample_multi_signal_summary.filter_signals(max_turnover=0.22)

        assert isinstance(df, pl.DataFrame)
        assert len(df) == 2

    def test_filter_signals_significant_only_fdr(
        self, sample_multi_signal_summary: MultiSignalSummary
    ) -> None:
        """Test filter_signals with significant_only using FDR."""
        df = sample_multi_signal_summary.filter_signals(
            significant_only=True, significance_method="fdr"
        )

        assert isinstance(df, pl.DataFrame)
        assert len(df) == 2

    def test_filter_signals_significant_only_fwer(
        self, sample_multi_signal_summary: MultiSignalSummary
    ) -> None:
        """Test filter_signals with significant_only using FWER."""
        df = sample_multi_signal_summary.filter_signals(
            significant_only=True, significance_method="fwer"
        )

        assert isinstance(df, pl.DataFrame)
        assert len(df) == 1

    def test_filter_signals_multiple_criteria(
        self, sample_multi_signal_summary: MultiSignalSummary
    ) -> None:
        """Test filter_signals with multiple criteria."""
        df = sample_multi_signal_summary.filter_signals(
            min_ic=0.02, min_ic_ir=1.0, max_turnover=0.30
        )

        assert isinstance(df, pl.DataFrame)
        assert len(df) >= 1


class TestMultiSignalSummarySummary:
    """Tests for MultiSignalSummary.summary method."""

    def test_summary_basic(self, sample_multi_signal_summary: MultiSignalSummary) -> None:
        """Test summary returns formatted string."""
        summary = sample_multi_signal_summary.summary()

        assert isinstance(summary, str)
        assert "Multi-Signal Analysis Summary" in summary
        assert "Signals Analyzed: 5" in summary
        assert "FDR" in summary
        assert "FWER" in summary
        assert "Top 5 Signals by IC IR:" in summary
        assert "sig_1" in summary

    def test_summary_without_ic_ir(self) -> None:
        """Test summary when ic_ir not in data."""
        summary_result = MultiSignalSummary(
            summary_data={
                "signal_name": ["sig_1", "sig_2"],
                "ic_mean": [0.05, 0.03],
                "fdr_significant": [True, False],
                "fwer_significant": [True, False],
            },
            n_signals=2,
            n_fdr_significant=1,
            n_fwer_significant=1,
            periods=(1,),
            fdr_alpha=0.05,
            fwer_alpha=0.05,
        )

        summary = summary_result.summary()

        assert isinstance(summary, str)
        assert "Multi-Signal Analysis Summary" in summary
        # Should not have top signals section since no ic_ir
        assert "Top 5 Signals by IC IR:" not in summary


# =============================================================================
# ComparisonResult Tests
# =============================================================================


class TestComparisonResultGetDataframe:
    """Tests for ComparisonResult.get_dataframe method."""

    def test_get_dataframe_default(self, sample_comparison_result: ComparisonResult) -> None:
        """Test get_dataframe with default (summary)."""
        df = sample_comparison_result.get_dataframe()

        assert isinstance(df, pl.DataFrame)
        assert "signal_name" in df.columns
        assert len(df) == 3

    def test_get_dataframe_summary(self, sample_comparison_result: ComparisonResult) -> None:
        """Test get_dataframe with name='summary'."""
        df = sample_comparison_result.get_dataframe("summary")

        assert isinstance(df, pl.DataFrame)
        assert "signal_name" in df.columns
        # Should have IC metrics from first period
        assert "ic_mean_1D" in df.columns or len(df.columns) >= 1

    def test_get_dataframe_correlation(self, sample_comparison_result: ComparisonResult) -> None:
        """Test get_dataframe with name='correlation'."""
        df = sample_comparison_result.get_dataframe("correlation")

        assert isinstance(df, pl.DataFrame)
        assert "sig_1" in df.columns
        assert "sig_2" in df.columns
        assert "sig_3" in df.columns

    def test_get_dataframe_invalid_name(self, sample_comparison_result: ComparisonResult) -> None:
        """Test get_dataframe raises for invalid name."""
        with pytest.raises(ValueError, match="Unknown DataFrame"):
            sample_comparison_result.get_dataframe("invalid")

    def test_get_dataframe_summary_without_ic(
        self, comparison_result_no_ic: ComparisonResult
    ) -> None:
        """Test get_dataframe summary when tear sheets have no IC data."""
        df = comparison_result_no_ic.get_dataframe("summary")

        assert isinstance(df, pl.DataFrame)
        assert "signal_name" in df.columns
        assert len(df) == 2


class TestComparisonResultListDataframes:
    """Tests for ComparisonResult.list_available_dataframes method."""

    def test_list_available_dataframes(self, sample_comparison_result: ComparisonResult) -> None:
        """Test list_available_dataframes returns expected list."""
        available = sample_comparison_result.list_available_dataframes()

        assert isinstance(available, list)
        assert "summary" in available
        assert "correlation" in available


class TestComparisonResultGetTearSheetData:
    """Tests for ComparisonResult.get_tear_sheet_data method."""

    def test_get_tear_sheet_data_valid(self, sample_comparison_result: ComparisonResult) -> None:
        """Test get_tear_sheet_data for valid signal."""
        data = sample_comparison_result.get_tear_sheet_data("sig_1")

        assert isinstance(data, dict)
        assert data["signal_name"] == "sig_1"
        assert "ic_analysis" in data

    def test_get_tear_sheet_data_invalid(self, sample_comparison_result: ComparisonResult) -> None:
        """Test get_tear_sheet_data raises for invalid signal."""
        with pytest.raises(ValueError, match="not in comparison"):
            sample_comparison_result.get_tear_sheet_data("nonexistent")


class TestComparisonResultGetCorrelationDataframe:
    """Tests for ComparisonResult.get_correlation_dataframe method."""

    def test_get_correlation_dataframe(self, sample_comparison_result: ComparisonResult) -> None:
        """Test get_correlation_dataframe returns DataFrame."""
        df = sample_comparison_result.get_correlation_dataframe()

        assert isinstance(df, pl.DataFrame)
        assert "sig_1" in df.columns


class TestComparisonResultGetPairwiseCorrelation:
    """Tests for ComparisonResult.get_pairwise_correlation method."""

    def test_get_pairwise_correlation_valid(
        self, sample_comparison_result: ComparisonResult
    ) -> None:
        """Test get_pairwise_correlation for valid signals."""
        corr = sample_comparison_result.get_pairwise_correlation("sig_1", "sig_2")

        assert isinstance(corr, float)
        assert corr == 0.5

    def test_get_pairwise_correlation_diagonal(
        self, sample_comparison_result: ComparisonResult
    ) -> None:
        """Test get_pairwise_correlation for same signal (diagonal)."""
        corr = sample_comparison_result.get_pairwise_correlation("sig_1", "sig_1")

        assert corr == 1.0

    def test_get_pairwise_correlation_invalid_signal1(
        self, sample_comparison_result: ComparisonResult
    ) -> None:
        """Test get_pairwise_correlation raises for invalid signal1."""
        with pytest.raises(ValueError, match="not found"):
            sample_comparison_result.get_pairwise_correlation("nonexistent", "sig_2")

    def test_get_pairwise_correlation_invalid_signal2(
        self, sample_comparison_result: ComparisonResult
    ) -> None:
        """Test get_pairwise_correlation raises for invalid signal2."""
        with pytest.raises(ValueError, match="not found"):
            sample_comparison_result.get_pairwise_correlation("sig_1", "nonexistent")


class TestComparisonResultSummary:
    """Tests for ComparisonResult.summary method."""

    def test_summary_with_params(self, sample_comparison_result: ComparisonResult) -> None:
        """Test summary with selection params."""
        summary = sample_comparison_result.summary()

        assert isinstance(summary, str)
        assert "Signal Comparison" in summary
        assert "Selection Method: top_n" in summary
        assert "Signals Compared: 3" in summary
        assert "sig_1" in summary
        assert "sig_2" in summary
        assert "sig_3" in summary
        assert "Selection Parameters:" in summary
        assert "n: 3" in summary

    def test_summary_without_params(self, comparison_result_no_ic: ComparisonResult) -> None:
        """Test summary without selection params."""
        summary = comparison_result_no_ic.summary()

        assert isinstance(summary, str)
        assert "Signal Comparison" in summary
        assert "Selection Method: manual" in summary
        # Should not have selection params section since empty
        assert "Selection Parameters:" not in summary
