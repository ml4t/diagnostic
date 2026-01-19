"""Tests for Phase 3 Week 1 components.

Tests cover:
- SmartCache with Polars fingerprinting
- MultiSignalAnalysisConfig validation
- holm_bonferroni FWER correction
- SignalSelector algorithms (top_n, uncorrelated, pareto, cluster)
"""

from __future__ import annotations

import time

import numpy as np
import polars as pl
import pytest

from ml4t.diagnostic.caching.smart_cache import SmartCache
from ml4t.diagnostic.config.multi_signal_config import MultiSignalAnalysisConfig
from ml4t.diagnostic.config.signal_config import AnalysisSettings, SignalConfig
from ml4t.diagnostic.evaluation.signal_selector import SignalSelector
from ml4t.diagnostic.evaluation.stats import benjamini_hochberg_fdr, holm_bonferroni

# =============================================================================
# SmartCache Tests
# =============================================================================


class TestSmartCacheFingerprint:
    """Test Polars DataFrame fingerprinting."""

    def test_same_data_same_fingerprint(self) -> None:
        """Identical DataFrames should produce identical fingerprints."""
        df1 = pl.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
        df2 = pl.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})

        fp1 = SmartCache.polars_fingerprint(df1)
        fp2 = SmartCache.polars_fingerprint(df2)

        assert fp1 == fp2

    def test_different_values_different_fingerprint(self) -> None:
        """DataFrames with different values should have different fingerprints."""
        df1 = pl.DataFrame({"a": [1, 2, 3]})
        df2 = pl.DataFrame({"a": [1, 2, 4]})  # Different value

        fp1 = SmartCache.polars_fingerprint(df1)
        fp2 = SmartCache.polars_fingerprint(df2)

        assert fp1 != fp2

    def test_different_columns_different_fingerprint(self) -> None:
        """DataFrames with different columns should have different fingerprints."""
        df1 = pl.DataFrame({"a": [1, 2, 3]})
        df2 = pl.DataFrame({"b": [1, 2, 3]})  # Different column name

        fp1 = SmartCache.polars_fingerprint(df1)
        fp2 = SmartCache.polars_fingerprint(df2)

        assert fp1 != fp2

    def test_different_dtypes_different_fingerprint(self) -> None:
        """DataFrames with different dtypes should have different fingerprints."""
        df1 = pl.DataFrame({"a": [1, 2, 3]})  # int64
        df2 = pl.DataFrame({"a": [1.0, 2.0, 3.0]})  # float64

        fp1 = SmartCache.polars_fingerprint(df1)
        fp2 = SmartCache.polars_fingerprint(df2)

        assert fp1 != fp2

    def test_different_shape_different_fingerprint(self) -> None:
        """DataFrames with different shapes should have different fingerprints."""
        df1 = pl.DataFrame({"a": [1, 2, 3]})
        df2 = pl.DataFrame({"a": [1, 2, 3, 4]})

        fp1 = SmartCache.polars_fingerprint(df1)
        fp2 = SmartCache.polars_fingerprint(df2)

        assert fp1 != fp2

    def test_seed_determinism(self) -> None:
        """Same seed should produce same fingerprint."""
        df = pl.DataFrame({"a": np.random.randn(100)})

        fp1 = SmartCache.polars_fingerprint(df, seed=42)
        fp2 = SmartCache.polars_fingerprint(df, seed=42)

        assert fp1 == fp2
        # Same seed produces same fingerprint
        assert len(fp1) == 32  # MD5 hex digest length


class TestSmartCacheMakeKey:
    """Test cache key generation."""

    def test_make_key_format(self) -> None:
        """Key should have expected format."""
        cache = SmartCache()
        df = pl.DataFrame({"a": [1, 2, 3]})
        config = SignalConfig()

        key = cache.make_key("momentum", df, config)

        assert key.startswith("momentum_")
        parts = key.split("_")
        assert len(parts) == 3
        assert len(parts[1]) == 12  # DataFrame hash
        assert len(parts[2]) == 12  # Config hash

    def test_different_signal_name_different_key(self) -> None:
        """Different signal names should produce different keys."""
        cache = SmartCache()
        df = pl.DataFrame({"a": [1, 2, 3]})
        config = SignalConfig()

        key1 = cache.make_key("momentum", df, config)
        key2 = cache.make_key("value", df, config)

        assert key1 != key2

    def test_different_data_different_key(self) -> None:
        """Different data should produce different keys."""
        cache = SmartCache()
        df1 = pl.DataFrame({"a": [1, 2, 3]})
        df2 = pl.DataFrame({"a": [1, 2, 4]})
        config = SignalConfig()

        key1 = cache.make_key("momentum", df1, config)
        key2 = cache.make_key("momentum", df2, config)

        assert key1 != key2

    def test_different_config_different_key(self) -> None:
        """Different configs should produce different keys."""
        cache = SmartCache()
        df = pl.DataFrame({"a": [1, 2, 3]})
        config1 = SignalConfig(analysis=AnalysisSettings(quantiles=5))
        config2 = SignalConfig(analysis=AnalysisSettings(quantiles=10))

        key1 = cache.make_key("momentum", df, config1)
        key2 = cache.make_key("momentum", df, config2)

        assert key1 != key2


class TestSmartCacheOperations:
    """Test cache get/set/invalidate operations."""

    def test_get_miss_returns_none(self) -> None:
        """Getting non-existent key returns None."""
        cache = SmartCache()
        assert cache.get("nonexistent") is None

    def test_set_then_get(self) -> None:
        """Set then get should return the value."""
        cache = SmartCache()
        cache.set("key1", {"result": 42})

        result = cache.get("key1")
        assert result == {"result": 42}

    def test_lru_eviction(self) -> None:
        """Oldest entries should be evicted when at capacity."""
        cache = SmartCache(max_items=3)

        cache.set("a", 1)
        cache.set("b", 2)
        cache.set("c", 3)
        assert cache.size == 3

        # Adding 4th should evict oldest (a)
        cache.set("d", 4)
        assert cache.size == 3
        assert cache.get("a") is None
        assert cache.get("b") == 2

    def test_lru_updates_on_access(self) -> None:
        """Accessing an entry should move it to end (most recent)."""
        cache = SmartCache(max_items=3)

        cache.set("a", 1)
        cache.set("b", 2)
        cache.set("c", 3)

        # Access 'a' to make it most recent
        cache.get("a")

        # Add 'd' - should evict 'b' (now oldest), not 'a'
        cache.set("d", 4)
        assert cache.get("a") == 1  # Still here
        assert cache.get("b") is None  # Evicted

    def test_ttl_expiration(self) -> None:
        """Entries should expire after TTL."""
        cache = SmartCache(ttl_seconds=1)

        cache.set("key", "value")
        assert cache.get("key") == "value"

        # Wait for expiration
        time.sleep(1.1)
        assert cache.get("key") is None

    def test_invalidate_specific_key(self) -> None:
        """Invalidate should remove specific key."""
        cache = SmartCache()
        cache.set("a", 1)
        cache.set("b", 2)

        result = cache.invalidate("a")
        assert result is True
        assert cache.get("a") is None
        assert cache.get("b") == 2

    def test_invalidate_nonexistent_returns_false(self) -> None:
        """Invalidating non-existent key returns False."""
        cache = SmartCache()
        result = cache.invalidate("nonexistent")
        assert result is False

    def test_clear_removes_all(self) -> None:
        """Clear should remove all entries."""
        cache = SmartCache()
        cache.set("a", 1)
        cache.set("b", 2)

        cache.clear()
        assert cache.size == 0
        assert cache.get("a") is None

    def test_invalidate_signal(self) -> None:
        """Invalidate signal should remove all entries for that signal."""
        cache = SmartCache()
        cache.set("momentum_abc123_def456", 1)
        cache.set("momentum_xyz789_abc123", 2)
        cache.set("value_abc123_def456", 3)

        removed = cache.invalidate_signal("momentum")
        assert removed == 2
        assert cache.get("momentum_abc123_def456") is None
        assert cache.get("value_abc123_def456") == 3


class TestSmartCacheStats:
    """Test cache statistics."""

    def test_hit_rate_initial_zero(self) -> None:
        """Initial hit rate should be 0."""
        cache = SmartCache()
        assert cache.hit_rate == 0.0

    def test_hit_rate_calculation(self) -> None:
        """Hit rate should be hits / (hits + misses)."""
        cache = SmartCache()
        cache.set("a", 1)

        cache.get("a")  # Hit
        cache.get("a")  # Hit
        cache.get("b")  # Miss

        assert cache.hit_rate == pytest.approx(2 / 3)

    def test_stats_dict(self) -> None:
        """Stats should return comprehensive info."""
        cache = SmartCache(max_items=100, ttl_seconds=3600)
        cache.set("a", 1)
        cache.get("a")

        stats = cache.stats
        assert stats["hits"] == 1
        assert stats["misses"] == 0
        assert stats["size"] == 1
        assert stats["max_items"] == 100
        assert stats["ttl_seconds"] == 3600

    def test_contains_without_side_effects(self) -> None:
        """__contains__ should not update stats."""
        cache = SmartCache()
        cache.set("a", 1)

        assert "a" in cache
        assert "b" not in cache
        assert cache._hits == 0  # No hits counted
        assert cache._misses == 0  # No misses counted


# =============================================================================
# MultiSignalAnalysisConfig Tests
# =============================================================================


class TestMultiSignalAnalysisConfig:
    """Test configuration validation."""

    def test_default_values(self) -> None:
        """Default config should have expected values."""
        config = MultiSignalAnalysisConfig()

        assert config.fdr_alpha == 0.05
        assert config.fwer_alpha == 0.05
        assert config.n_jobs == -1
        assert config.cache_enabled is True
        assert config.max_signals_summary == 200

    def test_custom_values(self) -> None:
        """Custom values should be set correctly."""
        config = MultiSignalAnalysisConfig(
            fdr_alpha=0.01,
            fwer_alpha=0.01,
            n_jobs=4,
            cache_enabled=False,
        )

        assert config.fdr_alpha == 0.01
        assert config.n_jobs == 4
        assert config.cache_enabled is False

    def test_nested_signal_config(self) -> None:
        """Signal config should be nested correctly."""
        inner_config = SignalConfig(analysis=AnalysisSettings(quantiles=10, periods=(1, 5)))
        config = MultiSignalAnalysisConfig(signal_config=inner_config)

        assert config.signal_config.quantiles == 10
        assert config.signal_config.periods == (1, 5)

    def test_fdr_alpha_validation(self) -> None:
        """FDR alpha should be in valid range."""
        with pytest.raises(ValueError):
            MultiSignalAnalysisConfig(fdr_alpha=0.0)  # Too low

        with pytest.raises(ValueError):
            MultiSignalAnalysisConfig(fdr_alpha=0.6)  # Too high

    def test_selection_metric_validation(self) -> None:
        """Invalid selection metric should raise error."""
        with pytest.raises(ValueError, match="Invalid selection metric"):
            MultiSignalAnalysisConfig(default_selection_metric="invalid_metric")

    def test_valid_selection_metrics(self) -> None:
        """Valid selection metrics should be accepted."""
        for metric in ["ic_mean", "ic_ir", "ic_t_stat", "turnover_adj_ic", "quantile_spread"]:
            config = MultiSignalAnalysisConfig(default_selection_metric=metric)
            assert config.default_selection_metric == metric


# =============================================================================
# holm_bonferroni Tests
# =============================================================================


class TestHolmBonferroni:
    """Test FWER correction."""

    def test_basic_rejection(self) -> None:
        """Should reject low p-values."""
        p_values = [0.001, 0.01, 0.03, 0.08, 0.12]
        result = holm_bonferroni(p_values, alpha=0.05)

        # First two should be rejected
        # p[0]=0.001 vs 0.05/5=0.01 -> reject
        # p[1]=0.01 vs 0.05/4=0.0125 -> reject
        # p[2]=0.03 vs 0.05/3=0.0167 -> fail
        assert result["rejected"][0] is True
        assert result["rejected"][1] is True
        assert result["rejected"][2] is False
        assert result["n_rejected"] == 2

    def test_empty_p_values(self) -> None:
        """Empty input should return empty results."""
        result = holm_bonferroni([], alpha=0.05)

        assert result["rejected"] == []
        assert result["adjusted_p_values"] == []
        assert result["n_rejected"] == 0

    def test_single_p_value(self) -> None:
        """Single p-value should use standard alpha."""
        result = holm_bonferroni([0.03], alpha=0.05)
        assert result["rejected"][0] is True
        assert result["n_rejected"] == 1

        result = holm_bonferroni([0.06], alpha=0.05)
        assert result["rejected"][0] is False
        assert result["n_rejected"] == 0

    def test_all_significant(self) -> None:
        """All p-values below threshold should be rejected."""
        p_values = [0.001, 0.002, 0.003]
        result = holm_bonferroni(p_values, alpha=0.05)

        assert all(result["rejected"])
        assert result["n_rejected"] == 3

    def test_none_significant(self) -> None:
        """No p-values below threshold should be rejected."""
        p_values = [0.1, 0.2, 0.3]
        result = holm_bonferroni(p_values, alpha=0.05)

        assert not any(result["rejected"])
        assert result["n_rejected"] == 0

    def test_adjusted_p_values_bounded(self) -> None:
        """Adjusted p-values should be in [0, 1]."""
        p_values = [0.001, 0.01, 0.5, 0.9]
        result = holm_bonferroni(p_values)

        for adj_p in result["adjusted_p_values"]:
            assert 0 <= adj_p <= 1

    def test_original_order_preserved(self) -> None:
        """Results should be in original order, not sorted."""
        # Intentionally unsorted
        p_values = [0.5, 0.001, 0.1, 0.01]
        result = holm_bonferroni(p_values, alpha=0.05)

        # Position 1 (0.001) and 3 (0.01) should be rejected
        assert result["rejected"][0] is False  # 0.5
        assert result["rejected"][1] is True  # 0.001
        assert result["rejected"][2] is False  # 0.1
        assert result["rejected"][3] is True  # 0.01

    def test_fwer_more_conservative_than_fdr(self) -> None:
        """FWER should reject same or fewer hypotheses than FDR."""
        p_values = [0.001, 0.01, 0.02, 0.04, 0.05]

        fdr_result = benjamini_hochberg_fdr(p_values, alpha=0.05, return_details=True)
        fwer_result = holm_bonferroni(p_values, alpha=0.05)

        assert fwer_result["n_rejected"] <= fdr_result["n_rejected"]


# =============================================================================
# SignalSelector Tests
# =============================================================================


@pytest.fixture
def sample_summary_df() -> pl.DataFrame:
    """Create sample summary DataFrame for testing."""
    return pl.DataFrame(
        {
            "signal_name": ["sig_a", "sig_b", "sig_c", "sig_d", "sig_e"],
            "ic_mean": [0.05, 0.03, 0.08, 0.02, 0.06],
            "ic_ir": [1.5, 0.9, 2.0, 0.5, 1.8],
            "turnover_mean": [0.3, 0.1, 0.5, 0.05, 0.2],
            "fdr_significant": [True, False, True, False, True],
        }
    )


@pytest.fixture
def sample_correlation_matrix() -> pl.DataFrame:
    """Create sample correlation matrix."""
    # Create a correlation matrix where:
    # - sig_a and sig_b are highly correlated (0.9)
    # - sig_c and sig_d are highly correlated (0.85)
    # - sig_e is relatively uncorrelated with others
    corr = np.array(
        [
            [1.0, 0.9, 0.3, 0.2, 0.1],
            [0.9, 1.0, 0.25, 0.15, 0.2],
            [0.3, 0.25, 1.0, 0.85, 0.3],
            [0.2, 0.15, 0.85, 1.0, 0.25],
            [0.1, 0.2, 0.3, 0.25, 1.0],
        ]
    )
    return pl.DataFrame(
        corr,
        schema=["sig_a", "sig_b", "sig_c", "sig_d", "sig_e"],
    )


class TestSignalSelectorTopN:
    """Test top-N selection."""

    def test_select_top_n_by_ic_ir(self, sample_summary_df: pl.DataFrame) -> None:
        """Should select top N by IC IR."""
        selected = SignalSelector.select_top_n(sample_summary_df, n=3, metric="ic_ir")

        assert len(selected) == 3
        # ic_ir order: sig_c (2.0), sig_e (1.8), sig_a (1.5)
        assert selected == ["sig_c", "sig_e", "sig_a"]

    def test_select_ascending(self, sample_summary_df: pl.DataFrame) -> None:
        """Should select lowest when ascending=True."""
        selected = SignalSelector.select_top_n(
            sample_summary_df, n=2, metric="turnover_mean", ascending=True
        )

        # Lowest turnover: sig_d (0.05), sig_b (0.1)
        assert selected == ["sig_d", "sig_b"]

    def test_filter_significant(self, sample_summary_df: pl.DataFrame) -> None:
        """Should filter to significant signals."""
        selected = SignalSelector.select_top_n(
            sample_summary_df,
            n=5,
            metric="ic_ir",
            filter_significant=True,
            significance_col="fdr_significant",
        )

        # Only sig_a, sig_c, sig_e are significant
        assert len(selected) == 3
        assert all(s in ["sig_a", "sig_c", "sig_e"] for s in selected)

    def test_invalid_metric_raises(self, sample_summary_df: pl.DataFrame) -> None:
        """Invalid metric should raise error."""
        with pytest.raises(ValueError, match="not found"):
            SignalSelector.select_top_n(sample_summary_df, n=3, metric="nonexistent")


class TestSignalSelectorUncorrelated:
    """Test uncorrelated signal selection."""

    def test_selects_uncorrelated(
        self,
        sample_summary_df: pl.DataFrame,
        sample_correlation_matrix: pl.DataFrame,
    ) -> None:
        """Should select signals with low mutual correlation."""
        selected = SignalSelector.select_uncorrelated(
            sample_summary_df,
            sample_correlation_matrix,
            n=3,
            metric="ic_ir",
            max_correlation=0.5,
        )

        # Should include sig_c (best IC IR)
        assert "sig_c" in selected

        # Should NOT include both sig_a and sig_b (corr=0.9)
        # And should NOT include both sig_c and sig_d (corr=0.85)
        assert not ({"sig_a", "sig_b"}.issubset(set(selected)))
        assert not ({"sig_c", "sig_d"}.issubset(set(selected)))

    def test_respects_min_metric(
        self,
        sample_summary_df: pl.DataFrame,
        sample_correlation_matrix: pl.DataFrame,
    ) -> None:
        """Should filter by minimum metric value."""
        selected = SignalSelector.select_uncorrelated(
            sample_summary_df,
            sample_correlation_matrix,
            n=5,
            metric="ic_ir",
            min_metric_value=1.0,  # Only sig_a, sig_c, sig_e
            max_correlation=1.0,  # No correlation filter
        )

        # sig_b (0.9) and sig_d (0.5) should be excluded
        assert "sig_b" not in selected
        assert "sig_d" not in selected


class TestSignalSelectorPareto:
    """Test Pareto frontier selection."""

    def test_finds_pareto_optimal(self, sample_summary_df: pl.DataFrame) -> None:
        """Should find Pareto-optimal signals."""
        selected = SignalSelector.select_pareto_frontier(
            sample_summary_df,
            x_metric="turnover_mean",  # minimize
            y_metric="ic_ir",  # maximize
        )

        # sig_c has highest IC IR but also highest turnover
        # sig_d has lowest turnover but lowest IC IR
        # sig_e has good IC IR (1.8) with moderate turnover (0.2)
        # At least sig_d (lowest turnover) and sig_c (highest IC IR) should be on frontier
        assert "sig_c" in selected or "sig_e" in selected
        assert "sig_d" in selected

    def test_no_dominated_solutions(self, sample_summary_df: pl.DataFrame) -> None:
        """Pareto set should contain no dominated solutions."""
        selected = SignalSelector.select_pareto_frontier(
            sample_summary_df,
            x_metric="turnover_mean",
            y_metric="ic_ir",
        )

        # Extract values for selected signals
        df = sample_summary_df.filter(pl.col("signal_name").is_in(selected))
        turnover = df["turnover_mean"].to_numpy()
        ic_ir = df["ic_ir"].to_numpy()

        # Check no solution dominates another in the set
        n = len(selected)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                # i dominates j if: lower turnover AND higher ic_ir (or equal on one, better on other)
                dominates = (
                    turnover[i] <= turnover[j]
                    and ic_ir[i] >= ic_ir[j]
                    and (turnover[i] < turnover[j] or ic_ir[i] > ic_ir[j])
                )
                assert not dominates, f"{selected[i]} dominates {selected[j]}"


class TestSignalSelectorCluster:
    """Test cluster-based selection."""

    def test_selects_from_each_cluster(
        self,
        sample_summary_df: pl.DataFrame,
        sample_correlation_matrix: pl.DataFrame,
    ) -> None:
        """Should select representative from each cluster."""
        pytest.importorskip("scipy")

        selected = SignalSelector.select_by_cluster(
            sample_correlation_matrix,
            sample_summary_df,
            n_clusters=2,  # Should create ~2 clusters
            signals_per_cluster=1,
            metric="ic_ir",
        )

        # With 2 clusters, should get 2 signals
        assert len(selected) == 2

    def test_best_from_cluster(
        self,
        sample_summary_df: pl.DataFrame,
        sample_correlation_matrix: pl.DataFrame,
    ) -> None:
        """Should select best signal by metric from each cluster."""
        pytest.importorskip("scipy")

        # With 5 clusters (one per signal), should get top 5 by metric
        selected = SignalSelector.select_by_cluster(
            sample_correlation_matrix,
            sample_summary_df,
            n_clusters=5,
            signals_per_cluster=1,
            metric="ic_ir",
        )

        # Should be sorted by IC IR descending
        assert len(selected) == 5
        assert selected[0] == "sig_c"  # Highest IC IR


class TestSignalSelectorInfo:
    """Test selection info generation."""

    def test_get_selection_info(self, sample_summary_df: pl.DataFrame) -> None:
        """Should return comprehensive selection info."""
        selected = ["sig_c", "sig_e", "sig_a"]
        info = SignalSelector.get_selection_info(
            sample_summary_df,
            selected,
            method="top_n",
            n=3,
            metric="ic_ir",
        )

        assert info["method"] == "top_n"
        assert info["n_selected"] == 3
        assert info["n_total"] == 5
        assert info["signals"] == selected
        assert info["method_params"]["n"] == 3
        assert len(info["selected_summary"]) == 3
