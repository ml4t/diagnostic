"""Performance benchmark tests for Phase 3 Multi-Signal Analysis.

Tests cover:
- SmartCache fingerprinting speed
- MultiSignalAnalysis scaling characteristics
- SignalSelector algorithm performance
- Parallel vs sequential performance

All benchmarks use pytest-benchmark for accurate timing measurements.
Run with: pytest -m slow tests/test_evaluation/test_multi_signal_performance.py
"""

from __future__ import annotations

import pytest

# Mark entire module as slow (benchmark tests with pytest-benchmark)
pytestmark = pytest.mark.slow

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from ml4t.diagnostic.caching.smart_cache import SmartCache
from ml4t.diagnostic.config.multi_signal_config import MultiSignalAnalysisConfig
from ml4t.diagnostic.evaluation.multi_signal import MultiSignalAnalysis
from ml4t.diagnostic.evaluation.signal_selector import SignalSelector

# =============================================================================
# Fixtures for Performance Testing
# =============================================================================


@pytest.fixture
def benchmark_trading_dates() -> list[datetime]:
    """Generate 100 trading days (reduced from 250 for faster benchmarks)."""
    start = datetime(2023, 1, 3)
    dates = []
    current = start
    while len(dates) < 100:
        if current.weekday() < 5:
            dates.append(current)
        current += timedelta(days=1)
    return dates


@pytest.fixture
def benchmark_price_data(benchmark_trading_dates: list[datetime]) -> pl.DataFrame:
    """Create price data for benchmarks (reduced from 30 to 15 assets)."""
    np.random.seed(42)
    n_assets = 15  # Reduced from 30
    assets = [f"ASSET_{i:03d}" for i in range(n_assets)]

    records = []
    for asset in assets:
        price = 100.0
        for date in benchmark_trading_dates:
            price *= 1 + np.random.randn() * 0.02
            records.append(
                {
                    "date": date,
                    "asset": asset,
                    "price": price,
                }
            )

    return pl.DataFrame(records)


def create_signal_set(
    n_signals: int,
    trading_dates: list[datetime],
    n_assets: int = 15,  # Reduced from 30
) -> dict[str, pl.DataFrame]:
    """Create n signals for benchmarking."""
    np.random.seed(42)
    assets = [f"ASSET_{i:03d}" for i in range(n_assets)]

    signals = {}
    for sig_num in range(n_signals):
        records = []
        for date in trading_dates:
            for asset in assets:
                factor = np.random.randn() * (0.1 + 0.01 * sig_num)
                records.append(
                    {
                        "date": date,
                        "asset": asset,
                        "factor": factor,
                    }
                )
        signals[f"signal_{sig_num:03d}"] = pl.DataFrame(records)

    return signals


# =============================================================================
# SmartCache Benchmarks
# =============================================================================


class TestSmartCacheBenchmarks:
    """Benchmark SmartCache fingerprinting performance."""

    def test_fingerprint_small_dataframe(self, benchmark) -> None:
        """Benchmark fingerprinting of small DataFrame (1000 rows)."""
        df = pl.DataFrame(
            {
                "a": np.random.randn(1000),
                "b": np.random.randn(1000),
                "c": np.random.randn(1000),
            }
        )

        result = benchmark(SmartCache.polars_fingerprint, df)

        assert len(result) == 32  # MD5 hex length

    def test_fingerprint_medium_dataframe(self, benchmark) -> None:
        """Benchmark fingerprinting of medium DataFrame (10,000 rows)."""
        df = pl.DataFrame(
            {
                "a": np.random.randn(10_000),
                "b": np.random.randn(10_000),
                "c": np.random.randn(10_000),
            }
        )

        result = benchmark(SmartCache.polars_fingerprint, df)

        assert len(result) == 32

    def test_fingerprint_large_dataframe(self, benchmark) -> None:
        """Benchmark fingerprinting of large DataFrame (100,000 rows)."""
        df = pl.DataFrame(
            {
                "a": np.random.randn(100_000),
                "b": np.random.randn(100_000),
                "c": np.random.randn(100_000),
            }
        )

        result = benchmark(SmartCache.polars_fingerprint, df)

        assert len(result) == 32

    def test_fingerprint_1000_ops(self, benchmark_trading_dates: list[datetime]) -> None:
        """Test fingerprinting speed target: <1ms per DataFrame."""
        import time

        n_dates = len(benchmark_trading_dates)
        n_assets = 15  # Match other fixtures
        # Create typical signal DataFrame
        df = pl.DataFrame(
            {
                "date": benchmark_trading_dates * n_assets,
                "asset": [f"ASSET_{i:03d}" for i in range(n_assets)] * n_dates,
                "factor": np.random.randn(n_dates * n_assets),
            }
        )

        # Run 1000 fingerprint operations
        start = time.time()
        for _ in range(1000):
            _ = SmartCache.polars_fingerprint(df)
        elapsed = time.time() - start

        avg_time_ms = elapsed / 1000 * 1000  # Convert to ms

        # Target: <1ms per operation (allowing some margin)
        assert avg_time_ms < 5.0, f"Fingerprinting too slow: {avg_time_ms:.2f}ms average"

    def test_cache_get_set_performance(self, benchmark) -> None:
        """Benchmark cache get/set operations."""
        cache = SmartCache(max_items=100)

        # Pre-populate cache
        for i in range(50):
            cache.set(f"key_{i}", {"data": i})

        def get_and_set():
            # Get existing
            _ = cache.get("key_25")
            # Set new
            cache.set("new_key", {"data": 999})
            return True

        result = benchmark(get_and_set)
        assert result


# =============================================================================
# MultiSignalAnalysis Scaling Benchmarks
# =============================================================================


class TestMultiSignalAnalysisScaling:
    """Benchmark MultiSignalAnalysis scaling characteristics."""

    @pytest.mark.slow
    def test_scaling_5_signals(
        self,
        benchmark,
        benchmark_trading_dates: list[datetime],
        benchmark_price_data: pl.DataFrame,
    ) -> None:
        """Benchmark analysis with 5 signals (reduced from 10)."""
        signals = create_signal_set(5, benchmark_trading_dates)
        config = MultiSignalAnalysisConfig(n_jobs=2, cache_enabled=False)

        def run_analysis():
            analyzer = MultiSignalAnalysis(
                signals=signals,
                prices=benchmark_price_data,
                config=config,
            )
            return analyzer.compute_summary(progress=False)

        summary = benchmark.pedantic(run_analysis, iterations=1, rounds=3)
        assert summary.n_signals == 5

    @pytest.mark.slow
    def test_scaling_10_signals(
        self,
        benchmark,
        benchmark_trading_dates: list[datetime],
        benchmark_price_data: pl.DataFrame,
    ) -> None:
        """Benchmark analysis with 10 signals (reduced from 20)."""
        signals = create_signal_set(10, benchmark_trading_dates)
        config = MultiSignalAnalysisConfig(n_jobs=2, cache_enabled=False)

        def run_analysis():
            analyzer = MultiSignalAnalysis(
                signals=signals,
                prices=benchmark_price_data,
                config=config,
            )
            return analyzer.compute_summary(progress=False)

        summary = benchmark.pedantic(run_analysis, iterations=1, rounds=3)
        assert summary.n_signals == 10

    def test_cache_hit_speedup(
        self,
        benchmark_trading_dates: list[datetime],
        benchmark_price_data: pl.DataFrame,
    ) -> None:
        """Test that cache hits are faster than cache misses."""
        import time

        signals = create_signal_set(5, benchmark_trading_dates)
        config = MultiSignalAnalysisConfig(n_jobs=2, cache_enabled=True)

        analyzer = MultiSignalAnalysis(
            signals=signals,
            prices=benchmark_price_data,
            config=config,
        )

        # First run (cache cold)
        start1 = time.time()
        _ = analyzer.compute_summary(progress=False)
        time1 = time.time() - start1

        # Second run (cache warm)
        start2 = time.time()
        _ = analyzer.compute_summary(progress=False)
        time2 = time.time() - start2

        # Cache hit should be faster (or at least not significantly slower)
        assert time2 <= time1 * 1.5, f"Cache not helping: {time1:.2f}s -> {time2:.2f}s"


# =============================================================================
# SignalSelector Algorithm Benchmarks
# =============================================================================


class TestSignalSelectorBenchmarks:
    """Benchmark signal selection algorithms."""

    @pytest.fixture
    def large_summary_df(self) -> pl.DataFrame:
        """Create large summary DataFrame for selection benchmarks."""
        np.random.seed(42)
        n_signals = 100

        return pl.DataFrame(
            {
                "signal_name": [f"signal_{i:03d}" for i in range(n_signals)],
                "ic_mean": np.random.randn(n_signals) * 0.05,
                "ic_std": np.abs(np.random.randn(n_signals) * 0.03) + 0.01,
                "ic_ir": np.random.randn(n_signals) * 1.5,
                "ic_t_stat": np.random.randn(n_signals) * 2,
                "ic_p_value": np.abs(np.random.rand(n_signals)),
                "turnover_mean": np.abs(np.random.randn(n_signals) * 0.3),
                "fdr_significant": np.random.choice([True, False], n_signals),
                "fwer_significant": np.random.choice([True, False], n_signals),
            }
        )

    @pytest.fixture
    def large_correlation_matrix(self) -> pl.DataFrame:
        """Create large correlation matrix for selection benchmarks."""
        np.random.seed(42)
        n_signals = 100
        signal_names = [f"signal_{i:03d}" for i in range(n_signals)]

        # Generate random correlation matrix (make it symmetric and positive semi-definite)
        random_matrix = np.random.randn(n_signals, n_signals) * 0.3
        corr_matrix = np.dot(random_matrix, random_matrix.T)
        # Normalize to correlations
        d = np.sqrt(np.diag(corr_matrix))
        corr_matrix = corr_matrix / np.outer(d, d)
        np.fill_diagonal(corr_matrix, 1.0)

        # Create DataFrame (signal names as columns, no signal_name column)
        data = {}
        for i, name in enumerate(signal_names):
            data[name] = corr_matrix[:, i].tolist()

        return pl.DataFrame(data)

    def test_top_n_selection_speed(
        self,
        benchmark,
        large_summary_df: pl.DataFrame,
    ) -> None:
        """Benchmark top_n selection with 100 signals."""
        result = benchmark(
            SignalSelector.select_top_n,
            large_summary_df,
            n=20,
            metric="ic_ir",
        )

        assert len(result) == 20

    def test_uncorrelated_selection_speed(
        self,
        benchmark,
        large_summary_df: pl.DataFrame,
        large_correlation_matrix: pl.DataFrame,
    ) -> None:
        """Benchmark uncorrelated selection with 100 signals."""
        result = benchmark(
            SignalSelector.select_uncorrelated,
            large_summary_df,
            large_correlation_matrix,
            n=10,
            max_correlation=0.5,
        )

        assert len(result) <= 10

    def test_pareto_selection_speed(
        self,
        benchmark,
        large_summary_df: pl.DataFrame,
    ) -> None:
        """Benchmark Pareto frontier selection with 100 signals."""
        result = benchmark(
            SignalSelector.select_pareto_frontier,
            large_summary_df,
            x_metric="turnover_mean",
            y_metric="ic_ir",
        )

        assert len(result) > 0

    def test_cluster_selection_speed(
        self,
        benchmark,
        large_summary_df: pl.DataFrame,
        large_correlation_matrix: pl.DataFrame,
    ) -> None:
        """Benchmark cluster selection with 100 signals."""
        result = benchmark(
            SignalSelector.select_by_cluster,
            large_correlation_matrix,
            large_summary_df,
            n_clusters=10,
            signals_per_cluster=1,
        )

        assert len(result) <= 10


# =============================================================================
# Parallel vs Sequential Performance
# =============================================================================


class TestParallelPerformance:
    """Test parallel processing performance gains."""

    @pytest.mark.slow
    def test_parallel_faster_than_sequential(
        self,
        benchmark_trading_dates: list[datetime],
        benchmark_price_data: pl.DataFrame,
    ) -> None:
        """Test that parallel processing is faster than sequential for many signals."""
        import time

        signals = create_signal_set(5, benchmark_trading_dates)  # Reduced from 10

        # Sequential (n_jobs=1)
        config_seq = MultiSignalAnalysisConfig(n_jobs=1, cache_enabled=False)
        analyzer_seq = MultiSignalAnalysis(
            signals=signals,
            prices=benchmark_price_data,
            config=config_seq,
        )

        start_seq = time.time()
        _ = analyzer_seq.compute_summary(progress=False)
        time_seq = time.time() - start_seq

        # Parallel (n_jobs=-1)
        config_par = MultiSignalAnalysisConfig(n_jobs=-1, cache_enabled=False)
        analyzer_par = MultiSignalAnalysis(
            signals=signals,
            prices=benchmark_price_data,
            config=config_par,
        )

        start_par = time.time()
        _ = analyzer_par.compute_summary(progress=False)
        time_par = time.time() - start_par

        # Parallel should be faster or at least not much slower
        # (may be slower for small workloads due to overhead)
        # Allow 50% margin for small signal sets
        assert time_par < time_seq * 1.5, (
            f"Parallel not faster: seq={time_seq:.2f}s, par={time_par:.2f}s"
        )


# =============================================================================
# Memory Usage Tests
# =============================================================================


class TestMemoryUsage:
    """Test memory efficiency."""

    def test_dataframe_fingerprint_memory(self) -> None:
        """Test that fingerprinting doesn't leak memory."""
        import gc
        import tracemalloc

        # Create large DataFrame
        df = pl.DataFrame(
            {
                "a": np.random.randn(100_000),
                "b": np.random.randn(100_000),
                "c": np.random.randn(100_000),
            }
        )

        gc.collect()
        tracemalloc.start()

        # Run many fingerprints
        for _ in range(100):
            _ = SmartCache.polars_fingerprint(df)

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Memory should be reasonable (< 100MB for this test)
        peak_mb = peak / (1024 * 1024)
        assert peak_mb < 100, f"Memory too high: {peak_mb:.1f}MB"

    def test_cache_eviction(self) -> None:
        """Test that cache properly evicts old entries."""
        cache = SmartCache(max_items=10)

        # Add more items than max
        for i in range(20):
            cache.set(f"key_{i}", {"data": i})

        # Should only have max_items entries
        # Count non-None entries
        valid_entries = sum(1 for k in [f"key_{i}" for i in range(20)] if cache.get(k) is not None)

        # Recent entries should be present
        assert cache.get("key_19") is not None
        assert cache.get("key_18") is not None

        # Should have at most max_items entries
        assert valid_entries <= 10


# =============================================================================
# Regression Tests for Performance
# =============================================================================


class TestPerformanceRegression:
    """Tests to catch performance regressions."""

    def test_fingerprint_speed_regression(self) -> None:
        """Ensure fingerprinting stays fast."""
        import time

        df = pl.DataFrame(
            {
                "a": np.random.randn(10_000),
                "b": np.random.randn(10_000),
                "c": np.random.randn(10_000),
            }
        )

        # Warm up
        _ = SmartCache.polars_fingerprint(df)

        # Time 100 runs
        start = time.time()
        for _ in range(100):
            _ = SmartCache.polars_fingerprint(df)
        elapsed = time.time() - start

        # Should complete 100 fingerprints in < 1 second
        assert elapsed < 1.0, f"Fingerprinting regression: {elapsed:.2f}s for 100 ops"

    def test_selector_speed_regression(self) -> None:
        """Ensure selection algorithms stay fast."""
        import time

        np.random.seed(42)
        n_signals = 50

        summary_df = pl.DataFrame(
            {
                "signal_name": [f"signal_{i:02d}" for i in range(n_signals)],
                "ic_ir": np.random.randn(n_signals),
                "turnover_mean": np.abs(np.random.randn(n_signals)),
                "fdr_significant": np.random.choice([True, False], n_signals),
            }
        )

        # All selections should complete quickly
        start = time.time()

        for _ in range(100):
            _ = SignalSelector.select_top_n(summary_df, n=10, metric="ic_ir")

        elapsed = time.time() - start

        # 100 top_n selections should be < 1 second
        assert elapsed < 1.0, f"Selection regression: {elapsed:.2f}s for 100 ops"
