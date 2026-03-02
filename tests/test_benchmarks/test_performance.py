"""Benchmark tests for performance-critical operations.

Run with: pytest tests/test_benchmarks/ --benchmark-only -m slow
"""

import numpy as np
import pytest

# Mark entire module as slow (benchmark tests)
pytestmark = pytest.mark.slow

# Skip if benchmark plugin not available
pytest.importorskip("pytest_benchmark")


@pytest.fixture
def small_data():
    """Small dataset for quick benchmarks."""
    np.random.seed(42)
    n = 1000
    return {
        "X": np.random.randn(n, 10),
        "y": np.random.randn(n),
        "times": np.arange(n),
    }


@pytest.fixture
def medium_data():
    """Medium dataset for realistic benchmarks."""
    np.random.seed(42)
    n = 10_000
    return {
        "X": np.random.randn(n, 20),
        "y": np.random.randn(n),
        "times": np.arange(n),
    }


@pytest.fixture
def large_returns():
    """Large returns series for metrics benchmarks."""
    np.random.seed(42)
    return np.random.randn(252 * 10) * 0.02  # 10 years daily


class TestCrossValidationBenchmarks:
    """Benchmarks for cross-validation operations."""

    def test_cpcv_split_small(self, benchmark, small_data):
        """Benchmark CPCV split generation on small data."""
        from ml4t.diagnostic.splitters import CombinatorialCV

        cv = CombinatorialCV(n_groups=5, embargo_pct=0.01)
        X, y, times = small_data["X"], small_data["y"], small_data["times"]

        def run():
            return list(cv.split(X, y, times))

        result = benchmark(run)
        assert len(result) > 0

    def test_cpcv_split_medium(self, benchmark, medium_data):
        """Benchmark CPCV split generation on medium data."""
        from ml4t.diagnostic.splitters import CombinatorialCV

        cv = CombinatorialCV(n_groups=10, embargo_pct=0.01)
        X, y, times = medium_data["X"], medium_data["y"], medium_data["times"]

        def run():
            return list(cv.split(X, y, times))

        result = benchmark(run)
        assert len(result) > 0


class TestMetricsBenchmarks:
    """Benchmarks for metric calculations."""

    def test_sharpe_ratio(self, benchmark, large_returns):
        """Benchmark Sharpe ratio calculation."""

        def compute_sharpe(returns):
            mean = np.mean(returns)
            std = np.std(returns, ddof=1)
            if std == 0:
                return 0.0
            return mean / std * np.sqrt(252)

        result = benchmark(compute_sharpe, large_returns)
        assert isinstance(result, float)

    def test_max_drawdown(self, benchmark, large_returns):
        """Benchmark maximum drawdown calculation."""

        def max_drawdown(returns):
            cumulative = np.cumprod(1 + returns)
            running_max = np.maximum.accumulate(cumulative)
            drawdowns = (cumulative - running_max) / running_max
            return np.min(drawdowns)

        result = benchmark(max_drawdown, large_returns)
        assert result <= 0

    def test_information_coefficient(self, benchmark, medium_data):
        """Benchmark IC calculation."""
        from scipy.stats import spearmanr

        X, y = medium_data["X"][:, 0], medium_data["y"]

        def compute_ic():
            return spearmanr(X, y)[0]

        result = benchmark(compute_ic)
        assert -1 <= result <= 1


class TestDataContractBenchmarks:
    """Benchmarks for data contract operations."""

    def test_metrics_creation(self, benchmark):
        """Benchmark DataQualityMetrics creation."""
        from ml4t.diagnostic.integration.data_contract import DataQualityMetrics

        def create_metrics():
            return DataQualityMetrics(
                completeness=0.98,
                timeliness=1.0,
                accuracy_score=0.99,
                consistency_score=1.0,
                n_records=100000,
                n_anomalies=5,
                n_critical=0,
                n_error=2,
                n_warning=3,
            )

        result = benchmark(create_metrics)
        assert result.n_records == 100000

    def test_report_summary(self, benchmark):
        """Benchmark report summary generation."""
        from datetime import datetime

        from ml4t.diagnostic.integration.data_contract import (
            DataQualityMetrics,
            DataQualityReport,
        )

        metrics = DataQualityMetrics(
            completeness=0.98,
            timeliness=1.0,
            accuracy_score=0.99,
            consistency_score=1.0,
            n_records=100000,
            n_anomalies=5,
        )
        report = DataQualityReport(
            symbol="AAPL",
            source="databento",
            date_range=(datetime(2024, 1, 1), datetime(2024, 6, 30)),
            frequency="1min",
            metrics=metrics,
            is_production_ready=True,
        )

        result = benchmark(report.summary)
        assert "AAPL" in result

    def test_report_to_dict(self, benchmark):
        """Benchmark report serialization."""
        from datetime import datetime

        from ml4t.diagnostic.integration.data_contract import (
            DataQualityMetrics,
            DataQualityReport,
        )

        metrics = DataQualityMetrics(
            completeness=0.98,
            timeliness=1.0,
            accuracy_score=0.99,
            consistency_score=1.0,
            n_records=100000,
            n_anomalies=5,
        )
        report = DataQualityReport(
            symbol="AAPL",
            source="databento",
            date_range=(datetime(2024, 1, 1), datetime(2024, 6, 30)),
            frequency="1min",
            metrics=metrics,
            is_production_ready=True,
        )

        result = benchmark(report.to_dict)
        assert result["symbol"] == "AAPL"


class TestValidatedCVBenchmarks:
    """Benchmarks for ValidatedCrossValidation."""

    def test_evaluate_sharpes(self, benchmark):
        """Benchmark Sharpe evaluation without model fitting."""
        from ml4t.diagnostic.evaluation.validated_cv import ValidatedCrossValidation

        vcv = ValidatedCrossValidation()
        sharpes = [0.5, 0.6, 0.4, 0.7, 0.55, 0.45, 0.65, 0.5, 0.6, 0.55]

        result = benchmark(vcv.evaluate_sharpes, sharpes)
        assert result.n_folds == 10

    def test_vcv_full_workflow_small(self, benchmark, small_data):
        """Benchmark full VCV workflow on small data."""
        from ml4t.diagnostic.config import ValidatedCrossValidationConfig
        from ml4t.diagnostic.evaluation.validated_cv import (
            ValidatedCrossValidation,
        )

        class SimpleModel:
            def fit(self, X, y):
                self.mean_ = np.mean(y)
                return self

            def predict(self, X):
                return np.full(len(X), self.mean_)

        config = ValidatedCrossValidationConfig(n_groups=5, n_test_groups=1)
        vcv = ValidatedCrossValidation(config)
        model = SimpleModel()
        X, y, times = small_data["X"], small_data["y"], small_data["times"]

        def run():
            return vcv.fit_evaluate(X, y, model, times=times)

        result = benchmark(run)
        assert result.n_folds > 0


class TestCachingBenchmarks:
    """Benchmarks for caching operations."""

    def test_cache_hit(self, benchmark):
        """Benchmark cache hit performance."""
        from ml4t.diagnostic.caching import Cache, CacheConfig, CacheKey

        cache = Cache(CacheConfig(enabled=True, max_memory_items=1000))

        # Pre-populate cache
        keys = []
        for i in range(100):
            key = CacheKey.generate(key_id=i)
            cache.set(key, f"value_{i}")
            keys.append(key)

        # Key to look up
        target_key = keys[50]

        def cache_hit():
            return cache.get(target_key)

        result = benchmark(cache_hit)
        assert result == "value_50"

    def test_cache_miss(self, benchmark):
        """Benchmark cache miss performance."""
        from ml4t.diagnostic.caching import Cache, CacheConfig, CacheKey

        cache = Cache(CacheConfig(enabled=True, max_memory_items=1000))
        miss_key = CacheKey.generate(key_id="nonexistent")

        def cache_miss():
            return cache.get(miss_key)

        result = benchmark(cache_miss)
        assert result is None
