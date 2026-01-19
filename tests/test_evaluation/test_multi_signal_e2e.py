"""End-to-end tests for Phase 3 Multi-Signal Analysis.

Tests cover:
- Scale test: 20 signals with ~6 months of data (reduced for faster tests)
- Memory test: Verify <4GB memory usage
- Speed test: Complete analysis <300s
- Round-trip test: Save/load dashboard HTML
- Edge cases: Empty signals, single signal, all NaN periods
- Full workflow: Analysis -> Selection -> Comparison -> Dashboard

NOTE: These are E2E tests marked as slow.
Run with: pytest -m slow tests/test_evaluation/test_multi_signal_e2e.py
"""

from __future__ import annotations

import gc
import tempfile
import time
import tracemalloc
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

# Mark entire module as slow (large-scale E2E tests)
pytestmark = pytest.mark.slow

from ml4t.diagnostic.config.multi_signal_config import MultiSignalAnalysisConfig
from ml4t.diagnostic.evaluation.multi_signal import MultiSignalAnalysis
from ml4t.diagnostic.evaluation.signal_selector import SignalSelector
from ml4t.diagnostic.visualization.signal.multi_signal_dashboard import (
    MultiSignalDashboard,
)
from ml4t.diagnostic.visualization.signal.multi_signal_plots import (
    plot_ic_ridge,
    plot_pareto_frontier,
    plot_signal_correlation_heatmap,
    plot_signal_ranking_bar,
)

# =============================================================================
# Fixtures for Large-Scale Testing
# =============================================================================


@pytest.fixture(scope="module")
def trading_dates_2y() -> list[datetime]:
    """Generate ~100 trading days (reduced from 150 for faster tests)."""
    start = datetime(2021, 1, 4)
    dates = []
    current = start
    while len(dates) < 100:  # Reduced from 150 to 100 (~4 months)
        if current.weekday() < 5:  # Mon-Fri only
            dates.append(current)
        current += timedelta(days=1)
    return dates


@pytest.fixture(scope="module")
def large_signal_set(
    trading_dates_2y: list[datetime], price_data_2y: pl.DataFrame
) -> dict[str, pl.DataFrame]:
    """Generate 10 synthetic signals with realistic characteristics.

    Reduced from 100→30→20→10 signals for faster tests while maintaining test coverage.

    Signals have varying characteristics:
    - 30% have strong predictive power (IC > 0.03)
    - 50% have weak predictive power (IC 0.01-0.03)
    - 20% are noise (IC ~ 0)

    IMPORTANT: Signals are generated to be correlated with actual forward returns
    from price_data_2y, so IC calculations will show the expected significance.
    """
    n_signals = 10  # Reduced from 100 → 30 → 20 → 10
    n_assets = 10  # Reduced from 50 → 20 → 15 → 10
    n_dates = len(trading_dates_2y)
    [f"ASSET_{i:03d}" for i in range(n_assets)]

    # Compute forward returns from price data for signal correlation (vectorized)
    returns_df = (
        price_data_2y.sort(["asset", "date"])
        .with_columns(pl.col("price").shift(-1).over("asset").alias("next_price"))
        .with_columns(((pl.col("next_price") / pl.col("price")) - 1).alias("forward_return"))
        .fill_null(0.0)
        .select(["date", "asset", "forward_return"])
    )

    signals = {}

    for sig_num in range(n_signals):
        # Determine signal category and correlation strength
        # With 10 signals: 3 strong (30%), 5 weak (50%), 2 noise (20%)
        np.random.seed(42 + sig_num)
        if sig_num < 3:
            # Strong signals - high correlation with forward returns
            correlation_strength = np.random.uniform(0.15, 0.25)
            noise_level = 0.3
        elif sig_num < 8:
            # Weak signals - low correlation with forward returns
            correlation_strength = np.random.uniform(0.05, 0.12)
            noise_level = 0.5
        else:
            # Noise signals - no correlation
            correlation_strength = 0.0
            noise_level = 1.0

        # Generate vectorized noise for all date/asset combinations
        np.random.seed(42 + sig_num * 1000)
        noise = np.random.randn(n_dates * n_assets) * noise_level

        # Create signal DataFrame by joining with returns and adding noise
        signal_df = (
            returns_df.clone()
            .with_columns(
                (
                    pl.col("forward_return") * correlation_strength * 50 + pl.Series("noise", noise)
                ).alias("factor")
            )
            .select(["date", "asset", "factor"])
        )

        signals[f"signal_{sig_num:03d}"] = signal_df

    return signals


@pytest.fixture(scope="module")
def medium_signal_set(trading_dates_2y: list[datetime]) -> dict[str, pl.DataFrame]:
    """Generate 8 signals for medium-scale tests (reduced from 50→15→8)."""
    np.random.seed(42)
    n_signals = 8  # Reduced from 50 → 15 → 8
    n_assets = 10  # Reduced from 30 → 15 → 10
    assets = [f"ASSET_{i:03d}" for i in range(n_assets)]

    signals = {}
    for sig_num in range(n_signals):
        signal_strength = np.random.uniform(0.0, 0.05)
        records = []
        for date in trading_dates_2y:
            for asset in assets:
                factor = np.random.randn() * (0.1 + signal_strength)
                records.append(
                    {
                        "date": date,
                        "asset": asset,
                        "factor": factor,
                    }
                )
        signals[f"signal_{sig_num:02d}"] = pl.DataFrame(records)

    return signals


@pytest.fixture(scope="module")
def small_signal_set(trading_dates_2y: list[datetime]) -> dict[str, pl.DataFrame]:
    """Generate 5 signals for quick tests (reduced from 10)."""
    np.random.seed(42)
    n_signals = 5  # Reduced from 10
    n_assets = 10  # Reduced from 15 → 10
    assets = [f"ASSET_{i:02d}" for i in range(n_assets)]

    signals = {}
    for sig_num in range(n_signals):
        signal_strength = 0.02 + 0.01 * sig_num  # Increasing IC
        records = []
        for date in trading_dates_2y[:80]:  # Use first 80 dates (reduced from 100)
            for asset in assets:
                factor = np.random.randn() * (0.1 + signal_strength)
                records.append(
                    {
                        "date": date,
                        "asset": asset,
                        "factor": factor,
                    }
                )
        signals[f"signal_{sig_num:02d}"] = pl.DataFrame(records)

    return signals


@pytest.fixture(scope="module")
def price_data_2y(trading_dates_2y: list[datetime]) -> pl.DataFrame:
    """Create price data matching signal assets (reduced from 50→15→10 assets)."""
    np.random.seed(123)
    n_assets = 10  # Reduced from 50 → 20 → 15 → 10 (match large_signal_set)
    assets = [f"ASSET_{i:03d}" for i in range(n_assets)]

    records = []
    for asset in assets:
        price = 100.0
        for date in trading_dates_2y:
            # Random walk with drift
            price *= 1 + np.random.randn() * 0.02 + 0.0002
            records.append(
                {
                    "date": date,
                    "asset": asset,
                    "price": price,
                }
            )

    return pl.DataFrame(records)


@pytest.fixture(scope="module")
def price_data_small(trading_dates_2y: list[datetime]) -> pl.DataFrame:
    """Create price data for small signal set."""
    np.random.seed(123)
    n_assets = 10  # Reduced from 15 → 10, match small_signal_set
    assets = [f"ASSET_{i:02d}" for i in range(n_assets)]

    records = []
    for asset in assets:
        price = 100.0
        for date in trading_dates_2y[:80]:  # Match small_signal_set (reduced from 100)
            price *= 1 + np.random.randn() * 0.02 + 0.0002
            records.append(
                {
                    "date": date,
                    "asset": asset,
                    "price": price,
                }
            )

    return pl.DataFrame(records)


# =============================================================================
# Scale Tests
# =============================================================================


class TestMultiSignalEndToEnd:
    """End-to-end tests with realistic signal counts."""

    @pytest.mark.slow
    def test_10_signals_analysis(
        self, large_signal_set: dict[str, pl.DataFrame], price_data_2y: pl.DataFrame
    ) -> None:
        """Complete analysis workflow with 10 signals (reduced from 100→30→20→10)."""
        config = MultiSignalAnalysisConfig(
            n_jobs=4,  # Limit parallelism for test stability
            cache_enabled=True,
            fdr_alpha=0.05,
            fwer_alpha=0.05,
        )

        analyzer = MultiSignalAnalysis(
            signals=large_signal_set,
            prices=price_data_2y,
            config=config,
        )

        # Verify setup
        assert analyzer.n_signals == 10  # Reduced from 100 → 30 → 20 → 10
        assert len(analyzer.signal_names) == 10

        # Compute summary
        summary = analyzer.compute_summary(progress=False)

        # Verify summary structure
        assert summary.n_signals == 10  # Reduced from 100 → 30 → 20 → 10
        df = summary.get_dataframe()
        assert len(df) == 10

        # Check required columns
        required_cols = [
            "signal_name",
            "ic_mean",
            "ic_std",
            "ic_t_stat",
            "ic_p_value",
            "ic_ir",
            "fdr_significant",
            "fwer_significant",
        ]
        for col in required_cols:
            assert col in df.columns, f"Missing column: {col}"

        # FWER should be more conservative than FDR
        assert summary.n_fwer_significant <= summary.n_fdr_significant

        # Some signals should be significant (we designed strong signals)
        assert summary.n_fdr_significant > 0

    @pytest.mark.slow
    def test_memory_usage_under_4gb(
        self, large_signal_set: dict[str, pl.DataFrame], price_data_2y: pl.DataFrame
    ) -> None:
        """Verify memory stays under 4GB during analysis."""
        gc.collect()
        tracemalloc.start()

        config = MultiSignalAnalysisConfig(n_jobs=2)

        analyzer = MultiSignalAnalysis(
            signals=large_signal_set,
            prices=price_data_2y,
            config=config,
        )

        summary = analyzer.compute_summary(progress=False)

        # Check memory
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        peak_gb = peak / (1024**3)

        assert peak_gb < 4.0, f"Memory usage too high: {peak_gb:.2f}GB"

        # Cleanup
        del analyzer, summary
        gc.collect()

    @pytest.mark.slow
    def test_analysis_completes_under_300s(
        self, medium_signal_set: dict[str, pl.DataFrame], price_data_2y: pl.DataFrame
    ) -> None:
        """Performance benchmark: 8 signals analysis <300s (reduced from 50→15→8)."""
        # Adjust price data to match medium signal set (10 assets)
        price_assets = set(medium_signal_set["signal_00"]["asset"].unique())
        price_df = price_data_2y.filter(pl.col("asset").is_in(list(price_assets)))

        config = MultiSignalAnalysisConfig(n_jobs=-1, cache_enabled=False)

        start_time = time.time()

        analyzer = MultiSignalAnalysis(
            signals=medium_signal_set,
            prices=price_df,
            config=config,
        )
        summary = analyzer.compute_summary(progress=False)

        elapsed = time.time() - start_time

        # 300s should be very easy with reduced parameters
        assert elapsed < 300.0, f"Analysis too slow: {elapsed:.1f}s"
        assert summary.n_signals == 8  # Reduced from 50 → 15 → 8

    def test_5_signals_fast(
        self, small_signal_set: dict[str, pl.DataFrame], price_data_small: pl.DataFrame
    ) -> None:
        """Quick sanity test with 5 signals (reduced from 10)."""
        config = MultiSignalAnalysisConfig(n_jobs=2)

        analyzer = MultiSignalAnalysis(
            signals=small_signal_set,
            prices=price_data_small,
            config=config,
        )

        summary = analyzer.compute_summary(progress=False)

        assert summary.n_signals == 5  # Reduced from 10
        assert summary.n_fdr_significant >= 0
        assert summary.n_fwer_significant >= 0


# =============================================================================
# Dashboard and Round-Trip Tests
# =============================================================================


class TestDashboardRoundTrip:
    """Test dashboard generation and file operations."""

    def test_dashboard_generation_and_save(
        self, small_signal_set: dict[str, pl.DataFrame], price_data_small: pl.DataFrame
    ) -> None:
        """Full dashboard workflow including save."""
        config = MultiSignalAnalysisConfig(n_jobs=2)

        analyzer = MultiSignalAnalysis(
            signals=small_signal_set,
            prices=price_data_small,
            config=config,
        )

        summary = analyzer.compute_summary(progress=False)
        correlation_matrix = analyzer.correlation_matrix()

        # Create dashboard
        dashboard = MultiSignalDashboard(title="E2E Test Dashboard", theme="light")

        html = dashboard.generate(
            analysis_results=summary,
            correlation_matrix=correlation_matrix,
        )

        # Verify HTML structure
        assert "<!DOCTYPE html>" in html or "<!doctype html>" in html.lower()
        assert "E2E Test Dashboard" in html
        assert "signal_" in html  # Signal names present

        # Save and verify file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".html", delete=False) as f:
            f.write(html)
            temp_path = f.name

        try:
            # Verify file exists and is valid
            path = Path(temp_path)
            assert path.exists()
            assert path.stat().st_size > 1000  # Should be substantial

            # Reload and verify
            loaded_html = path.read_text()
            assert "E2E Test Dashboard" in loaded_html
        finally:
            path.unlink()  # Cleanup

    def test_dashboard_save_method(
        self, small_signal_set: dict[str, pl.DataFrame], price_data_small: pl.DataFrame
    ) -> None:
        """Test dashboard save method directly."""
        config = MultiSignalAnalysisConfig(n_jobs=2)

        analyzer = MultiSignalAnalysis(
            signals=small_signal_set,
            prices=price_data_small,
            config=config,
        )

        summary = analyzer.compute_summary(progress=False)

        dashboard = MultiSignalDashboard(title="Save Test")

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_dashboard.html"

            saved_path = dashboard.save(
                output_path=str(output_path),
                analysis_results=summary,
            )

            assert Path(saved_path).exists()
            content = Path(saved_path).read_text()
            assert "Save Test" in content

    def test_dashboard_with_comparison(
        self, small_signal_set: dict[str, pl.DataFrame], price_data_small: pl.DataFrame
    ) -> None:
        """Test dashboard with comparison tab."""
        config = MultiSignalAnalysisConfig(n_jobs=2)

        analyzer = MultiSignalAnalysis(
            signals=small_signal_set,
            prices=price_data_small,
            config=config,
        )

        summary = analyzer.compute_summary(progress=False)
        comparison = analyzer.compare(selection="top_n", n=3)
        correlation_matrix = analyzer.correlation_matrix()

        dashboard = MultiSignalDashboard(title="Comparison Test")

        html = dashboard.generate(
            analysis_results=summary,
            correlation_matrix=correlation_matrix,
            comparison=comparison,
        )

        # Should have comparison tab
        assert "comparison" in html.lower() or "Comparison" in html


# =============================================================================
# Selection Algorithm Tests at Scale
# =============================================================================


class TestSelectionAlgorithmsAtScale:
    """Test all 4 selection algorithms work at scale."""

    @pytest.mark.slow
    def test_top_n_selection_8_signals(
        self, medium_signal_set: dict[str, pl.DataFrame], price_data_2y: pl.DataFrame
    ) -> None:
        """Test top_n selection with 8 signals (reduced from 50→15→8)."""
        # Filter price data to match medium_signal_set (10 assets)
        price_assets = [f"ASSET_{i:03d}" for i in range(10)]
        price_df = price_data_2y.filter(pl.col("asset").is_in(price_assets))

        config = MultiSignalAnalysisConfig(n_jobs=4)

        analyzer = MultiSignalAnalysis(
            signals=medium_signal_set,
            prices=price_df,
            config=config,
        )

        summary = analyzer.compute_summary(progress=False)

        # Test top_n selection - ask for 5 (of 8 signals)
        selected = SignalSelector.select_top_n(summary.get_dataframe(), n=5, metric="ic_ir")

        assert len(selected) == 5
        assert all(s in summary.get_dataframe()["signal_name"].to_list() for s in selected)

    def test_uncorrelated_selection(
        self, small_signal_set: dict[str, pl.DataFrame], price_data_small: pl.DataFrame
    ) -> None:
        """Test uncorrelated selection algorithm."""
        config = MultiSignalAnalysisConfig(n_jobs=2)

        analyzer = MultiSignalAnalysis(
            signals=small_signal_set,
            prices=price_data_small,
            config=config,
        )

        summary = analyzer.compute_summary(progress=False)
        correlation_matrix = analyzer.correlation_matrix()

        selected = SignalSelector.select_uncorrelated(
            summary_df=summary.get_dataframe(),
            correlation_matrix=correlation_matrix,
            n=5,
            max_correlation=0.7,
        )

        assert len(selected) <= 5
        assert len(selected) > 0

    def test_pareto_selection(
        self, small_signal_set: dict[str, pl.DataFrame], price_data_small: pl.DataFrame
    ) -> None:
        """Test Pareto frontier selection algorithm."""
        config = MultiSignalAnalysisConfig(n_jobs=2)

        analyzer = MultiSignalAnalysis(
            signals=small_signal_set,
            prices=price_data_small,
            config=config,
        )

        summary = analyzer.compute_summary(progress=False)

        selected = SignalSelector.select_pareto_frontier(
            summary_df=summary.get_dataframe(),
            x_metric="turnover_mean",
            y_metric="ic_ir",
        )

        # Pareto frontier should exist
        assert len(selected) > 0
        assert len(selected) <= summary.n_signals

    def test_cluster_selection(
        self, small_signal_set: dict[str, pl.DataFrame], price_data_small: pl.DataFrame
    ) -> None:
        """Test cluster-based selection algorithm."""
        config = MultiSignalAnalysisConfig(n_jobs=2)

        analyzer = MultiSignalAnalysis(
            signals=small_signal_set,
            prices=price_data_small,
            config=config,
        )

        summary = analyzer.compute_summary(progress=False)
        correlation_matrix = analyzer.correlation_matrix()

        selected = SignalSelector.select_by_cluster(
            correlation_matrix=correlation_matrix,
            summary_df=summary.get_dataframe(),
            n_clusters=3,
            signals_per_cluster=1,
        )

        assert len(selected) <= 3
        assert len(selected) > 0

    def test_compare_method_all_selections(
        self, small_signal_set: dict[str, pl.DataFrame], price_data_small: pl.DataFrame
    ) -> None:
        """Test compare() method with all selection types."""
        config = MultiSignalAnalysisConfig(n_jobs=2)

        analyzer = MultiSignalAnalysis(
            signals=small_signal_set,
            prices=price_data_small,
            config=config,
        )

        # First compute summary (required before compare)
        _ = analyzer.compute_summary(progress=False)

        # Test all selection methods
        for selection in ["top_n", "uncorrelated", "pareto"]:
            comparison = analyzer.compare(selection=selection, n=3)

            assert comparison is not None
            assert len(comparison.signals) <= 3
            assert comparison.selection_method == selection


# =============================================================================
# Visualization Tests at Scale
# =============================================================================


class TestVisualizationAtScale:
    """Test visualization functions with larger datasets."""

    @pytest.mark.slow
    def test_ic_ridge_8_signals(
        self, medium_signal_set: dict[str, pl.DataFrame], price_data_2y: pl.DataFrame
    ) -> None:
        """Test IC ridge plot with 8 signals (reduced from 50→15→8)."""
        price_assets = [f"ASSET_{i:03d}" for i in range(10)]
        price_df = price_data_2y.filter(pl.col("asset").is_in(price_assets))

        config = MultiSignalAnalysisConfig(n_jobs=4)

        analyzer = MultiSignalAnalysis(
            signals=medium_signal_set,
            prices=price_df,
            config=config,
        )

        summary = analyzer.compute_summary(progress=False)

        fig = plot_ic_ridge(summary, max_signals=30)

        assert fig is not None
        assert len(fig.data) > 0

    @pytest.mark.slow
    def test_correlation_heatmap_8_signals(
        self, medium_signal_set: dict[str, pl.DataFrame], price_data_2y: pl.DataFrame
    ) -> None:
        """Test correlation heatmap with 8 signals (reduced from 50→15→8)."""
        price_assets = [f"ASSET_{i:03d}" for i in range(10)]
        price_df = price_data_2y.filter(pl.col("asset").is_in(price_assets))

        config = MultiSignalAnalysisConfig(n_jobs=4)

        analyzer = MultiSignalAnalysis(
            signals=medium_signal_set,
            prices=price_df,
            config=config,
        )

        _ = analyzer.compute_summary(progress=False)
        correlation_matrix = analyzer.correlation_matrix()

        fig = plot_signal_correlation_heatmap(correlation_matrix, cluster=True)

        assert fig is not None
        assert len(fig.data) > 0

    @pytest.mark.slow
    def test_pareto_frontier_8_signals(
        self, medium_signal_set: dict[str, pl.DataFrame], price_data_2y: pl.DataFrame
    ) -> None:
        """Test Pareto frontier plot with 8 signals (reduced from 50→15→8)."""
        price_assets = [f"ASSET_{i:03d}" for i in range(10)]
        price_df = price_data_2y.filter(pl.col("asset").is_in(price_assets))

        config = MultiSignalAnalysisConfig(n_jobs=4)

        analyzer = MultiSignalAnalysis(
            signals=medium_signal_set,
            prices=price_df,
            config=config,
        )

        summary = analyzer.compute_summary(progress=False)

        fig = plot_pareto_frontier(summary)

        assert fig is not None
        assert len(fig.data) >= 1  # At least scatter trace

    @pytest.mark.slow
    def test_ranking_bar_8_signals(
        self, medium_signal_set: dict[str, pl.DataFrame], price_data_2y: pl.DataFrame
    ) -> None:
        """Test ranking bar chart with 8 signals (reduced from 50→15→8)."""
        price_assets = [f"ASSET_{i:03d}" for i in range(10)]
        price_df = price_data_2y.filter(pl.col("asset").is_in(price_assets))

        config = MultiSignalAnalysisConfig(n_jobs=4)

        analyzer = MultiSignalAnalysis(
            signals=medium_signal_set,
            prices=price_df,
            config=config,
        )

        summary = analyzer.compute_summary(progress=False)

        fig = plot_signal_ranking_bar(summary, top_n=20)

        assert fig is not None
        assert len(fig.data) > 0


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_signal(self, trading_dates_2y: list[datetime]) -> None:
        """Test with just one signal."""
        np.random.seed(42)
        n_assets = 10
        assets = [f"ASSET_{i:02d}" for i in range(n_assets)]

        # Single signal
        records = []
        for date in trading_dates_2y[:100]:
            for asset in assets:
                records.append(
                    {
                        "date": date,
                        "asset": asset,
                        "factor": np.random.randn(),
                    }
                )

        signals = {"only_signal": pl.DataFrame(records)}

        # Price data
        price_records = []
        for asset in assets:
            price = 100.0
            for date in trading_dates_2y[:100]:
                price *= 1 + np.random.randn() * 0.02
                price_records.append(
                    {
                        "date": date,
                        "asset": asset,
                        "price": price,
                    }
                )

        prices = pl.DataFrame(price_records)

        analyzer = MultiSignalAnalysis(
            signals=signals,
            prices=prices,
        )

        summary = analyzer.compute_summary(progress=False)

        assert summary.n_signals == 1
        assert len(summary.get_dataframe()) == 1

    def test_signals_with_missing_data(self, trading_dates_2y: list[datetime]) -> None:
        """Test signals with some missing data."""
        np.random.seed(42)
        n_assets = 10
        assets = [f"ASSET_{i:02d}" for i in range(n_assets)]

        signals = {}

        # Complete signal
        records = []
        for date in trading_dates_2y[:100]:
            for asset in assets:
                records.append(
                    {
                        "date": date,
                        "asset": asset,
                        "factor": np.random.randn(),
                    }
                )
        signals["complete"] = pl.DataFrame(records)

        # Signal with gaps (every other day)
        records_sparse = []
        for idx, date in enumerate(trading_dates_2y[:100]):
            if idx % 2 == 0:  # Skip every other day
                for asset in assets:
                    records_sparse.append(
                        {
                            "date": date,
                            "asset": asset,
                            "factor": np.random.randn(),
                        }
                    )
        signals["sparse"] = pl.DataFrame(records_sparse)

        # Price data
        price_records = []
        for asset in assets:
            price = 100.0
            for date in trading_dates_2y[:100]:
                price *= 1 + np.random.randn() * 0.02
                price_records.append(
                    {
                        "date": date,
                        "asset": asset,
                        "price": price,
                    }
                )

        prices = pl.DataFrame(price_records)

        analyzer = MultiSignalAnalysis(
            signals=signals,
            prices=prices,
        )

        summary = analyzer.compute_summary(progress=False)

        assert summary.n_signals == 2
        df = summary.get_dataframe()
        assert len(df) == 2

    def test_signals_with_nan_values(self, trading_dates_2y: list[datetime]) -> None:
        """Test signals with NaN values."""
        np.random.seed(42)
        n_assets = 10
        assets = [f"ASSET_{i:02d}" for i in range(n_assets)]

        # Signal with some NaNs
        records = []
        for idx, date in enumerate(trading_dates_2y[:100]):
            for asset in assets:
                value = np.random.randn() if idx % 10 != 0 else np.nan
                records.append(
                    {
                        "date": date,
                        "asset": asset,
                        "factor": value,
                    }
                )

        signals = {"signal_with_nan": pl.DataFrame(records)}

        # Price data
        price_records = []
        for asset in assets:
            price = 100.0
            for date in trading_dates_2y[:100]:
                price *= 1 + np.random.randn() * 0.02
                price_records.append(
                    {
                        "date": date,
                        "asset": asset,
                        "price": price,
                    }
                )

        prices = pl.DataFrame(price_records)

        analyzer = MultiSignalAnalysis(
            signals=signals,
            prices=prices,
        )

        # Should handle NaNs gracefully
        summary = analyzer.compute_summary(progress=False)

        assert summary.n_signals == 1

    def test_very_short_history(self) -> None:
        """Test with minimal history (edge case)."""
        np.random.seed(42)

        # Only 30 days (minimum for analysis)
        dates = [datetime(2023, 1, 1) + timedelta(days=i) for i in range(30)]
        assets = ["A", "B", "C"]

        records = []
        for date in dates:
            for asset in assets:
                records.append(
                    {
                        "date": date,
                        "asset": asset,
                        "factor": np.random.randn(),
                    }
                )

        signals = {"short_signal": pl.DataFrame(records)}

        price_records = []
        for asset in assets:
            price = 100.0
            for date in dates:
                price *= 1 + np.random.randn() * 0.02
                price_records.append(
                    {
                        "date": date,
                        "asset": asset,
                        "price": price,
                    }
                )

        prices = pl.DataFrame(price_records)

        config = MultiSignalAnalysisConfig(min_observations=20)

        analyzer = MultiSignalAnalysis(
            signals=signals,
            prices=prices,
            config=config,
        )

        summary = analyzer.compute_summary(progress=False)

        # Should work with minimal data
        assert summary.n_signals == 1


# =============================================================================
# Full Workflow Integration Test
# =============================================================================


class TestFullWorkflow:
    """Integration test for complete analysis workflow."""

    def test_complete_workflow(
        self, small_signal_set: dict[str, pl.DataFrame], price_data_small: pl.DataFrame
    ) -> None:
        """Test complete workflow: analysis -> selection -> comparison -> dashboard."""
        # Step 1: Create analyzer
        config = MultiSignalAnalysisConfig(
            n_jobs=2,
            cache_enabled=True,
            fdr_alpha=0.05,
            fwer_alpha=0.05,
        )

        analyzer = MultiSignalAnalysis(
            signals=small_signal_set,
            prices=price_data_small,
            config=config,
        )

        # Step 2: Compute summary with FDR/FWER
        summary = analyzer.compute_summary(progress=False)

        assert summary.n_signals == 5  # small_signal_set has 5 signals
        assert "fdr_significant" in summary.get_dataframe().columns
        assert "fwer_significant" in summary.get_dataframe().columns

        # Step 3: Compute correlation matrix
        correlation_matrix = analyzer.correlation_matrix()

        assert correlation_matrix is not None
        assert len(correlation_matrix.columns) > 1  # signal names

        # Step 4: Select best signals
        selected = SignalSelector.select_top_n(summary.get_dataframe(), n=3, metric="ic_ir")

        assert len(selected) == 3

        # Step 5: Get detailed comparison
        comparison = analyzer.compare(selection="top_n", n=3)

        assert comparison is not None
        assert len(comparison.signals) == 3

        # Step 6: Generate all visualizations
        fig_ridge = plot_ic_ridge(summary)
        fig_ranking = plot_signal_ranking_bar(summary, top_n=5)
        fig_corr = plot_signal_correlation_heatmap(correlation_matrix)
        fig_pareto = plot_pareto_frontier(summary)

        assert all(f is not None for f in [fig_ridge, fig_ranking, fig_corr, fig_pareto])

        # Step 7: Generate dashboard
        dashboard = MultiSignalDashboard(title="Complete Workflow Test")

        html = dashboard.generate(
            analysis_results=summary,
            correlation_matrix=correlation_matrix,
            comparison=comparison,
        )

        # Verify dashboard content
        assert len(html) > 10000  # Should be substantial
        assert "Complete Workflow Test" in html

        # Step 8: Save dashboard
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "complete_workflow.html"
            saved_path = dashboard.save(
                str(output_path),
                analysis_results=summary,
                correlation_matrix=correlation_matrix,
                comparison=comparison,
            )

            assert Path(saved_path).exists()
            assert Path(saved_path).stat().st_size > 10000


# =============================================================================
# Cache Effectiveness Tests
# =============================================================================


class TestCacheEffectiveness:
    """Test that caching improves performance."""

    def test_cache_reduces_recomputation_time(
        self, small_signal_set: dict[str, pl.DataFrame], price_data_small: pl.DataFrame
    ) -> None:
        """Test that cached results are faster than fresh computation."""
        config = MultiSignalAnalysisConfig(n_jobs=2, cache_enabled=True)

        analyzer = MultiSignalAnalysis(
            signals=small_signal_set,
            prices=price_data_small,
            config=config,
        )

        # First computation (cache cold)
        start1 = time.time()
        summary1 = analyzer.compute_summary(progress=False)
        time1 = time.time() - start1

        # Second computation (should use cache)
        start2 = time.time()
        summary2 = analyzer.compute_summary(progress=False)
        time2 = time.time() - start2

        # Results should be identical
        assert summary1.n_signals == summary2.n_signals
        assert summary1.n_fdr_significant == summary2.n_fdr_significant

        # Second should be faster (or at least not slower)
        # Allow some variance due to system load
        assert time2 <= time1 * 1.2  # Allow 20% variance

    def test_cache_invalidation_on_data_change(
        self, small_signal_set: dict[str, pl.DataFrame], price_data_small: pl.DataFrame
    ) -> None:
        """Test that cache produces different fingerprints for different data."""
        from ml4t.diagnostic.caching.smart_cache import SmartCache

        # Get original signal
        original_df = small_signal_set["signal_00"]

        # Create modified signal with different values
        modified_df = original_df.with_columns((pl.col("factor") + 10.0).alias("factor"))

        # Fingerprints should be different
        cache = SmartCache()
        fp1 = cache.polars_fingerprint(original_df)
        fp2 = cache.polars_fingerprint(modified_df)

        assert fp1 != fp2, "Fingerprints should differ for different data"

        # Verify that same data produces same fingerprint
        fp3 = cache.polars_fingerprint(original_df)
        assert fp1 == fp3, "Same data should produce same fingerprint"
