"""Tests for Phase 3 Week 2 components.

Tests cover:
- MultiSignalSummary result class
- ComparisonResult result class
- MultiSignalAnalysis core class

Tests complete quickly (~5s) with module-scoped fixtures.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from ml4t.diagnostic.config.multi_signal_config import MultiSignalAnalysisConfig
from ml4t.diagnostic.evaluation.multi_signal import MultiSignalAnalysis
from ml4t.diagnostic.results.multi_signal_results import ComparisonResult, MultiSignalSummary

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_summary_data() -> dict[str, list]:
    """Create sample summary data for testing."""
    return {
        "signal_name": ["sig_a", "sig_b", "sig_c", "sig_d", "sig_e"],
        "ic_mean": [0.05, 0.03, 0.08, 0.02, 0.06],
        "ic_std": [0.03, 0.04, 0.04, 0.04, 0.03],
        "ic_t_stat": [2.5, 1.1, 3.0, 0.8, 2.8],
        "ic_p_value": [0.01, 0.27, 0.003, 0.42, 0.005],
        "ic_ir": [1.67, 0.75, 2.0, 0.5, 2.0],
        "ic_positive_pct": [0.65, 0.52, 0.70, 0.48, 0.68],
        "n_observations": [250, 250, 250, 250, 250],
        "turnover_mean": [0.3, 0.1, 0.5, 0.05, 0.2],
        "fdr_significant": [True, False, True, False, True],
        "fdr_adjusted_p": [0.025, 0.27, 0.015, 0.42, 0.0125],
        "fwer_significant": [True, False, True, False, True],
        "fwer_adjusted_p": [0.05, 0.81, 0.015, 0.84, 0.02],
    }


@pytest.fixture
def sample_multi_signal_summary(sample_summary_data: dict) -> MultiSignalSummary:
    """Create sample MultiSignalSummary for testing."""
    return MultiSignalSummary(
        summary_data=sample_summary_data,
        n_signals=5,
        n_fdr_significant=3,
        n_fwer_significant=3,
        periods=(1, 5, 10),
        fdr_alpha=0.05,
        fwer_alpha=0.05,
    )


@pytest.fixture
def sample_comparison_result() -> ComparisonResult:
    """Create sample ComparisonResult for testing."""
    return ComparisonResult(
        signals=["sig_a", "sig_c", "sig_e"],
        selection_method="top_n",
        selection_params={"n": 3, "metric": "ic_ir"},
        tear_sheets={
            "sig_a": {"ic_analysis": {"ic_mean": {"1D": 0.05}}},
            "sig_c": {"ic_analysis": {"ic_mean": {"1D": 0.08}}},
            "sig_e": {"ic_analysis": {"ic_mean": {"1D": 0.06}}},
        },
        correlation_matrix={
            "sig_a": [1.0, 0.3, 0.2],
            "sig_c": [0.3, 1.0, 0.4],
            "sig_e": [0.2, 0.4, 1.0],
        },
    )


@pytest.fixture(scope="module")
def trading_dates() -> list[datetime]:
    """Generate 60 trading days (reduced from 300→100→60 for faster tests)."""
    start = datetime(2022, 1, 3)
    dates = []
    current = start
    while len(dates) < 60:
        if current.weekday() < 5:
            dates.append(current)
        current += timedelta(days=1)
    return dates


@pytest.fixture(scope="module")
def sample_signals(trading_dates: list[datetime]) -> dict[str, pl.DataFrame]:
    """Create sample signals for MultiSignalAnalysis testing.

    Note: Reduced from 20 assets × 5 signals to 10 assets × 3 signals for faster tests.
    """
    np.random.seed(42)
    n_assets = 8  # Reduced from 20→10→8
    assets = [f"ASSET_{i:02d}" for i in range(n_assets)]

    signals = {}
    for sig_num in range(3):  # Reduced from 5
        records = []
        for date in trading_dates:
            for asset in assets:
                factor = np.random.randn() * (1 + sig_num * 0.1)
                records.append(
                    {
                        "date": date,
                        "asset": asset,
                        "factor": factor,
                    }
                )
        signals[f"signal_{sig_num}"] = pl.DataFrame(records)

    return signals


@pytest.fixture(scope="module")
def sample_prices(trading_dates: list[datetime]) -> pl.DataFrame:
    """Create sample prices for testing (reduced from 20 to 10 assets)."""
    np.random.seed(123)
    n_assets = 8  # Reduced from 20→10→8
    assets = [f"ASSET_{i:02d}" for i in range(n_assets)]

    records = []
    for asset in assets:
        price = 100.0
        for date in trading_dates:
            price *= 1 + np.random.randn() * 0.02
            records.append(
                {
                    "date": date,
                    "asset": asset,
                    "price": price,
                }
            )

    return pl.DataFrame(records)


# =============================================================================
# MultiSignalSummary Tests
# =============================================================================


class TestMultiSignalSummary:
    """Test MultiSignalSummary result class."""

    def test_creation(self, sample_multi_signal_summary: MultiSignalSummary) -> None:
        """Test basic creation."""
        summary = sample_multi_signal_summary
        assert summary.n_signals == 5
        assert summary.n_fdr_significant == 3
        assert summary.n_fwer_significant == 3

    def test_get_dataframe(self, sample_multi_signal_summary: MultiSignalSummary) -> None:
        """Test DataFrame retrieval."""
        df = sample_multi_signal_summary.get_dataframe()
        assert isinstance(df, pl.DataFrame)
        assert len(df) == 5
        assert "signal_name" in df.columns
        assert "ic_mean" in df.columns

    def test_get_significant_signals_fdr(
        self, sample_multi_signal_summary: MultiSignalSummary
    ) -> None:
        """Test getting FDR-significant signals."""
        sig_signals = sample_multi_signal_summary.get_significant_signals("fdr")
        assert len(sig_signals) == 3
        assert "sig_a" in sig_signals
        assert "sig_c" in sig_signals
        assert "sig_b" not in sig_signals

    def test_get_significant_signals_fwer(
        self, sample_multi_signal_summary: MultiSignalSummary
    ) -> None:
        """Test getting FWER-significant signals."""
        sig_signals = sample_multi_signal_summary.get_significant_signals("fwer")
        assert len(sig_signals) == 3

    def test_get_ranking(self, sample_multi_signal_summary: MultiSignalSummary) -> None:
        """Test ranking by metric."""
        ranked = sample_multi_signal_summary.get_ranking("ic_ir")
        # ic_ir: sig_c=2.0, sig_e=2.0, sig_a=1.67, sig_b=0.75, sig_d=0.5
        assert ranked[0] in ["sig_c", "sig_e"]  # Top 2 tied
        assert ranked[-1] == "sig_d"

    def test_get_ranking_with_n(self, sample_multi_signal_summary: MultiSignalSummary) -> None:
        """Test ranking with limit."""
        ranked = sample_multi_signal_summary.get_ranking("ic_ir", n=3)
        assert len(ranked) == 3

    def test_filter_signals(self, sample_multi_signal_summary: MultiSignalSummary) -> None:
        """Test signal filtering."""
        filtered = sample_multi_signal_summary.filter_signals(min_ic=0.04)
        assert len(filtered) == 3  # sig_a, sig_c, sig_e

    def test_filter_significant_only(self, sample_multi_signal_summary: MultiSignalSummary) -> None:
        """Test filtering to significant signals only."""
        filtered = sample_multi_signal_summary.filter_signals(significant_only=True)
        assert len(filtered) == 3

    def test_summary_text(self, sample_multi_signal_summary: MultiSignalSummary) -> None:
        """Test human-readable summary."""
        text = sample_multi_signal_summary.summary()
        assert "Multi-Signal Analysis Summary" in text
        assert "5" in text  # n_signals
        assert "3" in text  # n_significant

    def test_list_available_dataframes(
        self, sample_multi_signal_summary: MultiSignalSummary
    ) -> None:
        """Test listing available DataFrames."""
        available = sample_multi_signal_summary.list_available_dataframes()
        assert "summary" in available


class TestMultiSignalSummaryCorrelation:
    """Test MultiSignalSummary with correlation data."""

    def test_with_correlation(self, sample_summary_data: dict) -> None:
        """Test creation with correlation data."""
        corr_data = {
            "sig_a": [1.0, 0.5, 0.3, 0.2, 0.1],
            "sig_b": [0.5, 1.0, 0.4, 0.3, 0.2],
            "sig_c": [0.3, 0.4, 1.0, 0.5, 0.4],
            "sig_d": [0.2, 0.3, 0.5, 1.0, 0.3],
            "sig_e": [0.1, 0.2, 0.4, 0.3, 1.0],
        }

        summary = MultiSignalSummary(
            summary_data=sample_summary_data,
            n_signals=5,
            n_fdr_significant=3,
            n_fwer_significant=3,
            periods=(1,),
            fdr_alpha=0.05,
            fwer_alpha=0.05,
            correlation_data=corr_data,
        )

        corr_df = summary.get_dataframe("correlation")
        assert isinstance(corr_df, pl.DataFrame)
        assert "sig_a" in corr_df.columns

    def test_correlation_not_available_raises(
        self, sample_multi_signal_summary: MultiSignalSummary
    ) -> None:
        """Test that missing correlation raises error."""
        with pytest.raises(ValueError, match="not computed"):
            sample_multi_signal_summary.get_dataframe("correlation")


# =============================================================================
# ComparisonResult Tests
# =============================================================================


class TestComparisonResult:
    """Test ComparisonResult class."""

    def test_creation(self, sample_comparison_result: ComparisonResult) -> None:
        """Test basic creation."""
        result = sample_comparison_result
        assert len(result.signals) == 3
        assert result.selection_method == "top_n"

    def test_get_dataframe_summary(self, sample_comparison_result: ComparisonResult) -> None:
        """Test getting summary DataFrame."""
        df = sample_comparison_result.get_dataframe()
        assert isinstance(df, pl.DataFrame)
        assert len(df) == 3

    def test_get_correlation_dataframe(self, sample_comparison_result: ComparisonResult) -> None:
        """Test getting correlation DataFrame."""
        corr = sample_comparison_result.get_correlation_dataframe()
        assert isinstance(corr, pl.DataFrame)
        assert len(corr.columns) == 3

    def test_get_tear_sheet_data(self, sample_comparison_result: ComparisonResult) -> None:
        """Test getting tear sheet data."""
        data = sample_comparison_result.get_tear_sheet_data("sig_c")
        assert "ic_analysis" in data

    def test_get_tear_sheet_invalid_signal(
        self, sample_comparison_result: ComparisonResult
    ) -> None:
        """Test that invalid signal raises error."""
        with pytest.raises(ValueError, match="not in comparison"):
            sample_comparison_result.get_tear_sheet_data("invalid")

    def test_get_pairwise_correlation(self, sample_comparison_result: ComparisonResult) -> None:
        """Test getting pairwise correlation."""
        corr = sample_comparison_result.get_pairwise_correlation("sig_a", "sig_c")
        assert corr == pytest.approx(0.3)

    def test_summary_text(self, sample_comparison_result: ComparisonResult) -> None:
        """Test human-readable summary."""
        text = sample_comparison_result.summary()
        assert "Signal Comparison" in text
        assert "top_n" in text
        assert "sig_a" in text


# =============================================================================
# MultiSignalAnalysis Tests
# =============================================================================


class TestMultiSignalAnalysisInit:
    """Test MultiSignalAnalysis initialization."""

    def test_basic_init(
        self,
        sample_signals: dict[str, pl.DataFrame],
        sample_prices: pl.DataFrame,
    ) -> None:
        """Test basic initialization."""
        analyzer = MultiSignalAnalysis(sample_signals, sample_prices)
        assert analyzer.n_signals == 3  # Reduced from 5
        assert len(analyzer.signal_names) == 3

    def test_with_config(
        self,
        sample_signals: dict[str, pl.DataFrame],
        sample_prices: pl.DataFrame,
    ) -> None:
        """Test initialization with custom config."""
        config = MultiSignalAnalysisConfig(
            fdr_alpha=0.01,
            cache_enabled=False,
        )
        analyzer = MultiSignalAnalysis(sample_signals, sample_prices, config=config)
        assert analyzer.config.fdr_alpha == 0.01

    def test_empty_signals_raises(self, sample_prices: pl.DataFrame) -> None:
        """Test that empty signals raises error."""
        with pytest.raises(ValueError, match="No signals provided"):
            MultiSignalAnalysis({}, sample_prices)

    def test_missing_columns_raises(self, sample_prices: pl.DataFrame) -> None:
        """Test that missing columns raises error."""
        bad_signal = pl.DataFrame({"date": [1, 2], "asset": ["A", "B"]})  # missing factor
        with pytest.raises(ValueError, match="missing required columns"):
            MultiSignalAnalysis({"bad": bad_signal}, sample_prices)


class TestMultiSignalAnalysisComputeSummary:
    """Test compute_summary method."""

    def test_compute_summary_serial(
        self,
        sample_signals: dict[str, pl.DataFrame],
        sample_prices: pl.DataFrame,
    ) -> None:
        """Test summary computation with serial execution."""
        config = MultiSignalAnalysisConfig(n_jobs=1, cache_enabled=False)
        analyzer = MultiSignalAnalysis(sample_signals, sample_prices, config=config)

        summary = analyzer.compute_summary(progress=False)

        assert isinstance(summary, MultiSignalSummary)
        assert summary.n_signals == 3  # Reduced from 5
        assert summary.n_fdr_significant >= 0
        assert summary.n_fwer_significant >= 0
        assert summary.n_fwer_significant <= summary.n_fdr_significant

    def test_summary_caching(
        self,
        sample_signals: dict[str, pl.DataFrame],
        sample_prices: pl.DataFrame,
    ) -> None:
        """Test that summary is cached."""
        config = MultiSignalAnalysisConfig(n_jobs=1, cache_enabled=True)
        analyzer = MultiSignalAnalysis(sample_signals, sample_prices, config=config)

        summary1 = analyzer.compute_summary(progress=False)
        summary2 = analyzer.compute_summary(progress=False)

        assert summary1 is summary2  # Same object

    def test_summary_has_fdr_fwer_columns(
        self,
        sample_signals: dict[str, pl.DataFrame],
        sample_prices: pl.DataFrame,
    ) -> None:
        """Test that summary includes FDR/FWER columns."""
        config = MultiSignalAnalysisConfig(n_jobs=1, cache_enabled=False)
        analyzer = MultiSignalAnalysis(sample_signals, sample_prices, config=config)

        summary = analyzer.compute_summary(progress=False)
        df = summary.get_dataframe()

        assert "fdr_significant" in df.columns
        assert "fwer_significant" in df.columns
        assert "fdr_adjusted_p" in df.columns
        assert "fwer_adjusted_p" in df.columns


class TestMultiSignalAnalysisCorrelation:
    """Test correlation matrix computation."""

    def test_correlation_matrix(
        self,
        sample_signals: dict[str, pl.DataFrame],
        sample_prices: pl.DataFrame,
    ) -> None:
        """Test correlation matrix computation."""
        analyzer = MultiSignalAnalysis(sample_signals, sample_prices)
        corr = analyzer.correlation_matrix()

        assert isinstance(corr, pl.DataFrame)
        assert len(corr.columns) == 3  # Reduced from 5
        # Diagonal should be 1
        for i, col in enumerate(corr.columns):
            assert corr[col][i] == pytest.approx(1.0)


class TestMultiSignalAnalysisCompare:
    """Test compare method."""

    def test_compare_top_n(
        self,
        sample_signals: dict[str, pl.DataFrame],
        sample_prices: pl.DataFrame,
    ) -> None:
        """Test top-N comparison."""
        config = MultiSignalAnalysisConfig(n_jobs=1, cache_enabled=False)
        analyzer = MultiSignalAnalysis(sample_signals, sample_prices, config=config)

        comparison = analyzer.compare(selection="top_n", n=3)

        assert isinstance(comparison, ComparisonResult)
        assert len(comparison.signals) <= 3
        assert comparison.selection_method == "top_n"

    def test_compare_manual(
        self,
        sample_signals: dict[str, pl.DataFrame],
        sample_prices: pl.DataFrame,
    ) -> None:
        """Test manual signal selection."""
        config = MultiSignalAnalysisConfig(n_jobs=1, cache_enabled=False)
        analyzer = MultiSignalAnalysis(sample_signals, sample_prices, config=config)

        comparison = analyzer.compare(
            selection="manual",
            signals=["signal_0", "signal_1"],
        )

        assert len(comparison.signals) == 2
        assert "signal_0" in comparison.signals

    def test_compare_manual_without_signals_raises(
        self,
        sample_signals: dict[str, pl.DataFrame],
        sample_prices: pl.DataFrame,
    ) -> None:
        """Test that manual selection without signals raises error."""
        analyzer = MultiSignalAnalysis(sample_signals, sample_prices)
        with pytest.raises(ValueError, match="signals parameter required"):
            analyzer.compare(selection="manual")


class TestMultiSignalAnalysisIndividual:
    """Test individual signal access."""

    def test_get_individual(
        self,
        sample_signals: dict[str, pl.DataFrame],
        sample_prices: pl.DataFrame,
    ) -> None:
        """Test getting individual SignalResult."""
        from ml4t.diagnostic.signal import SignalResult

        analyzer = MultiSignalAnalysis(sample_signals, sample_prices)
        individual = analyzer.get_individual("signal_0")

        assert isinstance(individual, SignalResult)

    def test_get_individual_invalid_raises(
        self,
        sample_signals: dict[str, pl.DataFrame],
        sample_prices: pl.DataFrame,
    ) -> None:
        """Test that invalid signal name raises error."""
        analyzer = MultiSignalAnalysis(sample_signals, sample_prices)
        with pytest.raises(ValueError, match="not found"):
            analyzer.get_individual("nonexistent")


class TestMultiSignalAnalysisCache:
    """Test caching behavior."""

    def test_cache_enabled(
        self,
        sample_signals: dict[str, pl.DataFrame],
        sample_prices: pl.DataFrame,
    ) -> None:
        """Test cache is created when enabled."""
        config = MultiSignalAnalysisConfig(cache_enabled=True)
        analyzer = MultiSignalAnalysis(sample_signals, sample_prices, config=config)

        assert analyzer._cache is not None

    def test_cache_disabled(
        self,
        sample_signals: dict[str, pl.DataFrame],
        sample_prices: pl.DataFrame,
    ) -> None:
        """Test cache is not created when disabled."""
        config = MultiSignalAnalysisConfig(cache_enabled=False)
        analyzer = MultiSignalAnalysis(sample_signals, sample_prices, config=config)

        assert analyzer._cache is None

    def test_cache_stats(
        self,
        sample_signals: dict[str, pl.DataFrame],
        sample_prices: pl.DataFrame,
    ) -> None:
        """Test cache statistics retrieval."""
        config = MultiSignalAnalysisConfig(cache_enabled=True, n_jobs=1)
        analyzer = MultiSignalAnalysis(sample_signals, sample_prices, config=config)

        stats = analyzer.cache_stats()
        assert stats is not None
        assert "hits" in stats

    def test_clear_cache(
        self,
        sample_signals: dict[str, pl.DataFrame],
        sample_prices: pl.DataFrame,
    ) -> None:
        """Test cache clearing."""
        config = MultiSignalAnalysisConfig(cache_enabled=True, n_jobs=1)
        analyzer = MultiSignalAnalysis(sample_signals, sample_prices, config=config)

        # Compute to populate cache
        analyzer.compute_summary(progress=False)
        assert analyzer._summary is not None

        # Clear
        analyzer.clear_cache()
        assert analyzer._summary is None


class TestMultiSignalAnalysisRepr:
    """Test string representation."""

    def test_repr(
        self,
        sample_signals: dict[str, pl.DataFrame],
        sample_prices: pl.DataFrame,
    ) -> None:
        """Test __repr__."""
        analyzer = MultiSignalAnalysis(sample_signals, sample_prices)
        repr_str = repr(analyzer)

        assert "MultiSignalAnalysis" in repr_str
        assert "n_signals=3" in repr_str  # Reduced from 5
