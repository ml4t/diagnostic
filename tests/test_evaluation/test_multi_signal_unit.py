"""Fast unit tests for MultiSignalAnalysis and SignalSelector.

These tests focus on logic paths and error handling without heavy computation.
They mock expensive operations to run in <1 second per test.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import numpy as np
import polars as pl
import pytest

from ml4t.diagnostic.config.multi_signal_config import MultiSignalAnalysisConfig
from ml4t.diagnostic.evaluation.multi_signal import MultiSignalAnalysis
from ml4t.diagnostic.evaluation.signal_selector import SignalSelector
from ml4t.diagnostic.results.multi_signal_results import ComparisonResult, MultiSignalSummary

# =============================================================================
# Fixtures for small, fast test data
# =============================================================================


@pytest.fixture
def small_dates() -> list[datetime]:
    """Generate 10 trading dates."""
    base = datetime(2024, 1, 1)
    return [base + timedelta(days=i) for i in range(10)]


@pytest.fixture
def small_assets() -> list[str]:
    """Generate 5 assets."""
    return ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]


@pytest.fixture
def small_signals(small_dates, small_assets) -> dict[str, pl.DataFrame]:
    """Generate 3 small signals for testing."""
    rng = np.random.RandomState(42)
    signals = {}

    for name in ["signal_a", "signal_b", "signal_c"]:
        rows = []
        for date in small_dates:
            for asset in small_assets:
                rows.append(
                    {
                        "date": date,
                        "asset": asset,
                        "factor": rng.randn(),
                    }
                )
        signals[name] = pl.DataFrame(rows)

    return signals


@pytest.fixture
def small_prices(small_dates, small_assets) -> pl.DataFrame:
    """Generate small price data."""
    rng = np.random.RandomState(123)
    rows = []
    for date in small_dates:
        for asset in small_assets:
            rows.append(
                {
                    "date": date,
                    "asset": asset,
                    "price": 100 + rng.randn() * 10,
                }
            )
    return pl.DataFrame(rows)


@pytest.fixture
def mock_analyze_signal():
    """Create mock analyze_signal function."""
    with patch("ml4t.diagnostic.evaluation.multi_signal.analyze_signal") as mock:
        # Setup mock SignalResult
        mock_result = MagicMock()
        mock_result.ic = {"1D": 0.05}
        mock_result.ic_std = {"1D": 0.02}
        mock_result.ic_t_stat = {"1D": 2.5}
        mock_result.ic_p_value = {"1D": 0.01}
        mock_result.ic_ir = {"1D": 0.5}
        mock_result.ic_positive_pct = {"1D": 60.0}
        mock_result.ic_series = {"1D": [0.05] * 10}
        mock_result.n_dates = 10
        mock_result.n_assets = 5
        mock_result.turnover = {"1D": 0.3}
        mock_result.autocorrelation = [0.7]
        mock_result.to_dict.return_value = {"ic": {"1D": 0.05}}

        mock.return_value = mock_result

        yield mock


# =============================================================================
# MultiSignalAnalysis Initialization Tests
# =============================================================================


class TestMultiSignalAnalysisInit:
    """Test MultiSignalAnalysis initialization and validation."""

    def test_init_basic(self, small_signals, small_prices, mock_analyze_signal):
        """Test basic initialization."""
        analyzer = MultiSignalAnalysis(small_signals, small_prices)

        assert analyzer.n_signals == 3
        assert set(analyzer.signal_names) == {"signal_a", "signal_b", "signal_c"}

    def test_init_with_config(self, small_signals, small_prices, mock_analyze_signal):
        """Test initialization with custom config."""
        config = MultiSignalAnalysisConfig(
            fdr_alpha=0.01,
            fwer_alpha=0.01,
            cache_enabled=False,
        )
        analyzer = MultiSignalAnalysis(small_signals, small_prices, config=config)

        assert analyzer.config.fdr_alpha == 0.01
        assert analyzer.config.cache_enabled is False

    def test_init_empty_signals(self, small_prices):
        """Test error on empty signals dict."""
        with pytest.raises(ValueError, match="No signals provided"):
            MultiSignalAnalysis({}, small_prices)

    def test_init_missing_factor_column(self, small_prices, small_dates, small_assets):
        """Test error when signal missing 'factor' column."""
        bad_signal = pl.DataFrame(
            {
                "date": small_dates[:5],
                "asset": small_assets,
                # Missing 'factor' column
            }
        )

        with pytest.raises(ValueError, match="missing required columns.*factor"):
            MultiSignalAnalysis({"bad": bad_signal}, small_prices)

    def test_init_missing_date_column(self, small_prices, small_assets):
        """Test error when signal missing 'date' column."""
        bad_signal = pl.DataFrame(
            {
                # Missing 'date' column
                "asset": small_assets,
                "factor": [1.0] * len(small_assets),
            }
        )

        with pytest.raises(ValueError, match="missing required columns.*date"):
            MultiSignalAnalysis({"bad": bad_signal}, small_prices)

    def test_init_missing_price_columns(self, small_signals, small_dates, small_assets):
        """Test error when prices missing required columns."""
        bad_prices = pl.DataFrame(
            {
                "date": small_dates[:5],
                "asset": small_assets,
                # Missing 'price' column
            }
        )

        with pytest.raises(ValueError, match="Price data missing required columns.*price"):
            MultiSignalAnalysis(small_signals, bad_prices)

    def test_init_pandas_input(self, small_signals, small_prices, mock_analyze_signal):
        """Test initialization with pandas input (auto-converted to polars)."""

        # Convert to pandas
        signals_pd = {name: df.to_pandas() for name, df in small_signals.items()}
        prices_pd = small_prices.to_pandas()

        analyzer = MultiSignalAnalysis(signals_pd, prices_pd)

        assert analyzer.n_signals == 3


class TestMultiSignalAnalysisProperties:
    """Test MultiSignalAnalysis properties."""

    def test_signal_names(self, small_signals, small_prices, mock_analyze_signal):
        """Test signal_names property."""
        analyzer = MultiSignalAnalysis(small_signals, small_prices)

        names = analyzer.signal_names
        assert isinstance(names, list)
        assert len(names) == 3
        assert "signal_a" in names

    def test_n_signals(self, small_signals, small_prices, mock_analyze_signal):
        """Test n_signals property."""
        analyzer = MultiSignalAnalysis(small_signals, small_prices)

        assert analyzer.n_signals == 3


class TestMultiSignalAnalysisGetIndividual:
    """Test get_individual method."""

    def test_get_individual_valid(self, small_signals, small_prices, mock_analyze_signal):
        """Test getting individual analyzer for valid signal."""
        analyzer = MultiSignalAnalysis(small_signals, small_prices)

        individual = analyzer.get_individual("signal_a")

        # Should be a SignalAnalysis instance (mocked)
        assert individual is not None

    def test_get_individual_cached(self, small_signals, small_prices, mock_analyze_signal):
        """Test that individual analyzers are cached."""
        analyzer = MultiSignalAnalysis(small_signals, small_prices)

        ind1 = analyzer.get_individual("signal_a")
        ind2 = analyzer.get_individual("signal_a")

        # Should be same object
        assert ind1 is ind2

    def test_get_individual_unknown_signal(self, small_signals, small_prices, mock_analyze_signal):
        """Test error for unknown signal."""
        analyzer = MultiSignalAnalysis(small_signals, small_prices)

        with pytest.raises(ValueError, match="Signal 'unknown' not found"):
            analyzer.get_individual("unknown")


class TestMultiSignalAnalysisCacheOperations:
    """Test cache-related operations."""

    def test_cache_stats_enabled(self, small_signals, small_prices, mock_analyze_signal):
        """Test cache_stats when cache enabled."""
        config = MultiSignalAnalysisConfig(cache_enabled=True)
        analyzer = MultiSignalAnalysis(small_signals, small_prices, config=config)

        stats = analyzer.cache_stats()

        assert stats is not None
        assert isinstance(stats, dict)

    def test_cache_stats_disabled(self, small_signals, small_prices, mock_analyze_signal):
        """Test cache_stats when cache disabled."""
        config = MultiSignalAnalysisConfig(cache_enabled=False)
        analyzer = MultiSignalAnalysis(small_signals, small_prices, config=config)

        stats = analyzer.cache_stats()

        assert stats is None

    def test_clear_cache(self, small_signals, small_prices, mock_analyze_signal):
        """Test clear_cache clears all cached data."""
        analyzer = MultiSignalAnalysis(small_signals, small_prices)

        # Access individual to populate cache
        analyzer.get_individual("signal_a")
        assert len(analyzer._individual_results) > 0

        analyzer.clear_cache()

        assert len(analyzer._individual_results) == 0
        assert analyzer._summary is None
        assert analyzer._correlation_matrix is None


class TestMultiSignalAnalysisRepr:
    """Test __repr__ method."""

    def test_repr_basic(self, small_signals, small_prices, mock_analyze_signal):
        """Test repr contains expected information."""
        analyzer = MultiSignalAnalysis(small_signals, small_prices)

        result = repr(analyzer)

        assert "MultiSignalAnalysis" in result
        assert "n_signals=3" in result
        # Cache state may vary based on initialization
        assert "cache=" in result

    def test_repr_without_cache(self, small_signals, small_prices, mock_analyze_signal):
        """Test repr with cache disabled."""
        config = MultiSignalAnalysisConfig(cache_enabled=False)
        analyzer = MultiSignalAnalysis(small_signals, small_prices, config=config)

        result = repr(analyzer)

        assert "cache=disabled" in result
        # Verify cache is actually disabled
        assert analyzer._cache is None


class TestMultiSignalAnalysisCompare:
    """Test compare method error handling."""

    def test_compare_manual_without_signals(self, small_signals, small_prices, mock_analyze_signal):
        """Test error when manual selection without signals list."""
        analyzer = MultiSignalAnalysis(small_signals, small_prices)

        with pytest.raises(ValueError, match="signals parameter required"):
            analyzer.compare(selection="manual", signals=None)

    def test_compare_unknown_selection(self, small_signals, small_prices, mock_analyze_signal):
        """Test error for unknown selection method."""
        analyzer = MultiSignalAnalysis(small_signals, small_prices)

        with pytest.raises(ValueError, match="Unknown selection method"):
            analyzer.compare(selection="invalid_method")


# =============================================================================
# SignalSelector Tests
# =============================================================================


class TestSignalSelectorSelectTopN:
    """Test select_top_n method."""

    @pytest.fixture
    def summary_df(self) -> pl.DataFrame:
        """Create summary DataFrame for testing."""
        return pl.DataFrame(
            {
                "signal_name": ["sig_a", "sig_b", "sig_c", "sig_d", "sig_e"],
                "ic_ir": [0.5, 0.3, 0.8, 0.1, 0.6],
                "ic_mean": [0.05, 0.03, 0.08, 0.01, 0.06],
                "turnover_mean": [0.2, 0.1, 0.4, 0.05, 0.3],
                "fdr_significant": [True, False, True, False, True],
            }
        )

    def test_select_top_n_basic(self, summary_df):
        """Test basic top N selection."""
        selected = SignalSelector.select_top_n(summary_df, n=3, metric="ic_ir")

        assert len(selected) == 3
        assert selected[0] == "sig_c"  # Highest IC IR
        assert selected[1] == "sig_e"
        assert selected[2] == "sig_a"

    def test_select_top_n_ascending(self, summary_df):
        """Test top N with ascending (e.g., lowest turnover)."""
        selected = SignalSelector.select_top_n(
            summary_df, n=3, metric="turnover_mean", ascending=True
        )

        assert len(selected) == 3
        assert selected[0] == "sig_d"  # Lowest turnover
        assert selected[1] == "sig_b"
        assert selected[2] == "sig_a"

    def test_select_top_n_filter_significant(self, summary_df):
        """Test filtering to significant signals only."""
        selected = SignalSelector.select_top_n(
            summary_df, n=10, metric="ic_ir", filter_significant=True
        )

        # Only 3 are significant
        assert len(selected) == 3
        assert all(s in ["sig_a", "sig_c", "sig_e"] for s in selected)

    def test_select_top_n_unknown_metric(self, summary_df):
        """Test error for unknown metric."""
        with pytest.raises(ValueError, match="Metric 'unknown' not found"):
            SignalSelector.select_top_n(summary_df, n=3, metric="unknown")


class TestSignalSelectorSelectUncorrelated:
    """Test select_uncorrelated method."""

    @pytest.fixture
    def summary_df(self) -> pl.DataFrame:
        """Create summary DataFrame."""
        return pl.DataFrame(
            {
                "signal_name": ["sig_a", "sig_b", "sig_c", "sig_d"],
                "ic_ir": [0.5, 0.4, 0.3, 0.2],
            }
        )

    @pytest.fixture
    def correlation_matrix(self) -> pl.DataFrame:
        """Create correlation matrix."""
        # sig_a and sig_b are highly correlated
        # sig_c and sig_d are independent
        return pl.DataFrame(
            {
                "sig_a": [1.0, 0.9, 0.1, 0.0],
                "sig_b": [0.9, 1.0, 0.2, 0.1],
                "sig_c": [0.1, 0.2, 1.0, 0.3],
                "sig_d": [0.0, 0.1, 0.3, 1.0],
            }
        )

    def test_select_uncorrelated_basic(self, summary_df, correlation_matrix):
        """Test basic uncorrelated selection."""
        selected = SignalSelector.select_uncorrelated(
            summary_df, correlation_matrix, n=3, max_correlation=0.5
        )

        # sig_a should be selected first (best IC IR)
        # sig_b should be rejected (too correlated with sig_a)
        # sig_c and sig_d should be selected
        assert len(selected) == 3
        assert "sig_a" in selected
        assert "sig_b" not in selected  # Too correlated with sig_a
        assert "sig_c" in selected
        assert "sig_d" in selected

    def test_select_uncorrelated_with_min_metric(self, summary_df, correlation_matrix):
        """Test filtering by minimum metric value."""
        selected = SignalSelector.select_uncorrelated(
            summary_df, correlation_matrix, n=10, min_metric_value=0.35
        )

        # Only sig_a and sig_b have IC IR >= 0.35
        # sig_b is too correlated with sig_a
        assert len(selected) == 1
        assert "sig_a" in selected

    def test_select_uncorrelated_empty_after_filter(self, summary_df, correlation_matrix):
        """Test returns empty list when no signals pass filter."""
        selected = SignalSelector.select_uncorrelated(
            summary_df,
            correlation_matrix,
            n=10,
            min_metric_value=1.0,  # Too high
        )

        assert selected == []


class TestSignalSelectorSelectParetoFrontier:
    """Test select_pareto_frontier method."""

    @pytest.fixture
    def summary_df(self) -> pl.DataFrame:
        """Create summary DataFrame with tradeoff scenarios."""
        return pl.DataFrame(
            {
                "signal_name": ["sig_a", "sig_b", "sig_c", "sig_d", "sig_e"],
                "turnover_mean": [0.1, 0.2, 0.3, 0.4, 0.25],
                "ic_ir": [0.3, 0.5, 0.7, 0.8, 0.4],
            }
        )

    def test_select_pareto_basic(self, summary_df):
        """Test basic Pareto selection (minimize turnover, maximize IC)."""
        selected = SignalSelector.select_pareto_frontier(
            summary_df, x_metric="turnover_mean", y_metric="ic_ir"
        )

        # Pareto optimal:
        # sig_a (0.1, 0.3) - best turnover, decent IC
        # sig_b (0.2, 0.5) - pareto optimal
        # sig_c (0.3, 0.7) - pareto optimal
        # sig_d (0.4, 0.8) - best IC, worst turnover
        # sig_e (0.25, 0.4) - dominated by sig_b
        assert "sig_a" in selected
        assert "sig_d" in selected
        assert "sig_e" not in selected  # Dominated

    def test_select_pareto_missing_metric(self, summary_df):
        """Test error for missing metric column."""
        with pytest.raises(ValueError, match="Metrics not found"):
            SignalSelector.select_pareto_frontier(summary_df, x_metric="unknown", y_metric="ic_ir")


class TestSignalSelectorSelectByCluster:
    """Test select_by_cluster method."""

    @pytest.fixture
    def summary_df(self) -> pl.DataFrame:
        """Create summary DataFrame."""
        return pl.DataFrame(
            {
                "signal_name": ["sig_a", "sig_b", "sig_c", "sig_d"],
                "ic_ir": [0.5, 0.4, 0.7, 0.2],
            }
        )

    @pytest.fixture
    def correlation_matrix(self) -> pl.DataFrame:
        """Create correlation matrix with 2 clear clusters."""
        # Cluster 1: sig_a, sig_b (correlated)
        # Cluster 2: sig_c, sig_d (correlated)
        return pl.DataFrame(
            {
                "sig_a": [1.0, 0.9, 0.1, 0.0],
                "sig_b": [0.9, 1.0, 0.0, 0.1],
                "sig_c": [0.1, 0.0, 1.0, 0.8],
                "sig_d": [0.0, 0.1, 0.8, 1.0],
            }
        )

    def test_select_by_cluster_basic(self, summary_df, correlation_matrix):
        """Test basic cluster selection."""
        selected = SignalSelector.select_by_cluster(
            correlation_matrix, summary_df, n_clusters=2, signals_per_cluster=1
        )

        # Should select best from each cluster
        # Cluster 1: sig_a (0.5) > sig_b (0.4) → sig_a
        # Cluster 2: sig_c (0.7) > sig_d (0.2) → sig_c
        assert len(selected) == 2
        assert "sig_a" in selected or "sig_b" in selected
        assert "sig_c" in selected or "sig_d" in selected


class TestSignalSelectorGetSelectionInfo:
    """Test get_selection_info method."""

    def test_get_selection_info(self):
        """Test selection info generation."""
        summary_df = pl.DataFrame(
            {
                "signal_name": ["sig_a", "sig_b", "sig_c"],
                "ic_ir": [0.5, 0.3, 0.7],
            }
        )

        info = SignalSelector.get_selection_info(
            summary_df,
            selected_signals=["sig_a", "sig_c"],
            method="top_n",
            n=2,
        )

        assert info["method"] == "top_n"
        assert info["n_selected"] == 2
        assert info["n_total"] == 3
        assert info["signals"] == ["sig_a", "sig_c"]
        assert info["method_params"]["n"] == 2


# =============================================================================
# MultiSignalSummary Result Tests
# =============================================================================


class TestMultiSignalSummary:
    """Test MultiSignalSummary result class."""

    @pytest.fixture
    def summary_result(self) -> MultiSignalSummary:
        """Create MultiSignalSummary for testing."""
        return MultiSignalSummary(
            summary_data={
                "signal_name": ["sig_a", "sig_b", "sig_c"],
                "ic_mean": [0.05, 0.03, 0.08],
                "ic_ir": [0.5, 0.3, 0.7],
                "fdr_significant": [True, False, True],
                "fwer_significant": [True, False, False],
            },
            n_signals=3,
            n_fdr_significant=2,
            n_fwer_significant=1,
            periods=(1, 5, 10),
            fdr_alpha=0.05,
            fwer_alpha=0.05,
        )

    def test_get_dataframe(self, summary_result):
        """Test getting summary as DataFrame."""
        df = summary_result.get_dataframe()

        assert len(df) == 3
        assert "signal_name" in df.columns
        assert "ic_ir" in df.columns

    def test_get_significant_signals_fdr(self, summary_result):
        """Test getting FDR-significant signals."""
        sig = summary_result.get_significant_signals(method="fdr")

        assert len(sig) == 2
        assert "sig_a" in sig
        assert "sig_c" in sig
        assert "sig_b" not in sig

    def test_get_significant_signals_fwer(self, summary_result):
        """Test getting FWER-significant signals."""
        sig = summary_result.get_significant_signals(method="fwer")

        assert len(sig) == 1
        assert "sig_a" in sig

    def test_get_ranking(self, summary_result):
        """Test getting ranked signal names."""
        ranked = summary_result.get_ranking(metric="ic_ir", n=2)

        assert len(ranked) == 2
        assert ranked[0] == "sig_c"  # Highest IC IR
        assert ranked[1] == "sig_a"

    def test_filter_signals(self, summary_result):
        """Test filtering signals by criteria."""
        filtered = summary_result.filter_signals(min_ic_ir=0.4)

        assert len(filtered) == 2
        assert "sig_a" in filtered["signal_name"].to_list()
        assert "sig_c" in filtered["signal_name"].to_list()

    def test_filter_significant_only(self, summary_result):
        """Test filtering to significant signals only."""
        filtered = summary_result.filter_signals(significant_only=True)

        assert len(filtered) == 2

    def test_summary_method(self, summary_result):
        """Test summary() string generation."""
        summary_str = summary_result.summary()

        assert "Multi-Signal Analysis Summary" in summary_str
        assert "Signals Analyzed: 3" in summary_str
        assert "FDR" in summary_str


class TestComparisonResult:
    """Test ComparisonResult class."""

    @pytest.fixture
    def comparison_result(self) -> ComparisonResult:
        """Create ComparisonResult for testing."""
        return ComparisonResult(
            signals=["sig_a", "sig_b"],
            selection_method="top_n",
            selection_params={"n": 2, "metric": "ic_ir"},
            tear_sheets={
                "sig_a": {"ic_analysis": {"ic_mean": {"1D": 0.05}}},
                "sig_b": {"ic_analysis": {"ic_mean": {"1D": 0.03}}},
            },
            correlation_matrix={
                "sig_a": [1.0, 0.3],
                "sig_b": [0.3, 1.0],
            },
        )

    def test_get_tear_sheet_data(self, comparison_result):
        """Test getting tear sheet data."""
        data = comparison_result.get_tear_sheet_data("sig_a")

        assert "ic_analysis" in data

    def test_get_tear_sheet_data_unknown_signal(self, comparison_result):
        """Test error for unknown signal."""
        with pytest.raises(ValueError, match="Signal 'unknown' not in comparison"):
            comparison_result.get_tear_sheet_data("unknown")

    def test_get_correlation_dataframe(self, comparison_result):
        """Test getting correlation as DataFrame."""
        df = comparison_result.get_correlation_dataframe()

        assert len(df) == 2
        assert "sig_a" in df.columns

    def test_get_pairwise_correlation(self, comparison_result):
        """Test getting pairwise correlation."""
        corr = comparison_result.get_pairwise_correlation("sig_a", "sig_b")

        assert corr == 0.3

    def test_summary_method(self, comparison_result):
        """Test summary() string generation."""
        summary_str = comparison_result.summary()

        assert "Signal Comparison" in summary_str
        assert "Selection Method: top_n" in summary_str
        assert "Signals Compared: 2" in summary_str

    def test_list_available_dataframes(self, comparison_result):
        """Test listing available DataFrames."""
        available = comparison_result.list_available_dataframes()

        assert "summary" in available
        assert "correlation" in available
