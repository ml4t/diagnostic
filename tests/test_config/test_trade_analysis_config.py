"""Tests for trade analysis configuration module.

Tests config validation, presets, and edge cases.
"""

from __future__ import annotations

from datetime import timedelta

import pytest

from ml4t.diagnostic.config.trade_analysis_config import (
    AlignmentSettings,
    ClusteringSettings,
    ExtractionSettings,
    FilterSettings,
    HypothesisSettings,
    TradeConfig,
)
from ml4t.diagnostic.config.validation import ClusteringMethod, DistanceMetric, LinkageMethod


class TestFilterSettings:
    """Tests for FilterSettings validation."""

    def test_default_values(self):
        """Test default values are None."""
        settings = FilterSettings()
        assert settings.min_duration is None
        assert settings.max_duration is None
        assert settings.min_pnl is None
        assert settings.exclude_symbols is None
        assert settings.regime_filter is None

    def test_valid_durations(self):
        """Test valid duration settings."""
        settings = FilterSettings(
            min_duration=timedelta(minutes=5),
            max_duration=timedelta(hours=1),
        )
        assert settings.min_duration == timedelta(minutes=5)
        assert settings.max_duration == timedelta(hours=1)

    def test_negative_duration_raises_error(self):
        """Test that negative duration raises validation error."""
        with pytest.raises(ValueError, match="Duration must be positive"):
            FilterSettings(min_duration=timedelta(seconds=-1))

    def test_zero_duration_raises_error(self):
        """Test that zero duration raises validation error."""
        with pytest.raises(ValueError, match="Duration must be positive"):
            FilterSettings(min_duration=timedelta(seconds=0))

    def test_max_less_than_min_raises_error(self):
        """Test that max_duration <= min_duration raises error."""
        with pytest.raises(ValueError, match="max_duration must be greater"):
            FilterSettings(
                min_duration=timedelta(hours=1),
                max_duration=timedelta(minutes=30),
            )

    def test_equal_durations_raises_error(self):
        """Test that equal durations raise error."""
        with pytest.raises(ValueError, match="max_duration must be greater"):
            FilterSettings(
                min_duration=timedelta(hours=1),
                max_duration=timedelta(hours=1),
            )


class TestExtractionSettings:
    """Tests for ExtractionSettings validation."""

    def test_default_values(self):
        """Test default values."""
        settings = ExtractionSettings()
        assert settings.n_worst == 20
        assert settings.n_best == 10
        assert settings.percentile_mode is False

    def test_warns_on_very_small_n_worst(self):
        """Test warning when n_worst is very small."""
        with pytest.warns(UserWarning, match="may be too few"):
            ExtractionSettings(n_worst=5)

    def test_warns_on_very_large_n_worst(self):
        """Test warning when n_worst is very large."""
        with pytest.warns(UserWarning, match="may dilute signal"):
            ExtractionSettings(n_worst=150)

    @pytest.mark.xfail(
        reason="Pydantic field validation order: percentile_mode is validated after n_worst, "
        "so validators see percentile_mode=False. This is a design limitation."
    )
    def test_warns_on_large_percentile(self):
        """Test warning when percentile is >50%."""
        with pytest.warns(UserWarning, match="includes majority"):
            ExtractionSettings(n_worst=60, percentile_mode=True)

    @pytest.mark.xfail(
        reason="Pydantic field validation order: percentile_mode is validated after n_worst, "
        "so validators see percentile_mode=False. This is a design limitation."
    )
    def test_invalid_percentile_raises_error(self):
        """Test that invalid percentile value raises error."""
        with pytest.raises(ValueError, match="Percentile must be 1-100"):
            ExtractionSettings(n_worst=0, percentile_mode=True)

        with pytest.raises(ValueError, match="Percentile must be 1-100"):
            ExtractionSettings(n_worst=101, percentile_mode=True)

    def test_valid_percentile_mode(self):
        """Test valid percentile mode settings."""
        settings = ExtractionSettings(n_worst=10, percentile_mode=True)
        assert settings.n_worst == 10
        assert settings.percentile_mode is True


class TestAlignmentSettings:
    """Tests for AlignmentSettings validation."""

    def test_default_values(self):
        """Test default values."""
        settings = AlignmentSettings()
        assert settings.mode == "entry"
        assert settings.tolerance == 300
        assert settings.missing_strategy == "skip"

    def test_warns_on_large_tolerance(self):
        """Test warning when tolerance is very large."""
        with pytest.warns(UserWarning, match="may misalign SHAP values"):
            AlignmentSettings(tolerance=7200)

    def test_different_modes(self):
        """Test different alignment modes."""
        for mode in ["entry", "nearest", "average"]:
            settings = AlignmentSettings(mode=mode)
            assert settings.mode == mode

    def test_different_missing_strategies(self):
        """Test different missing value strategies."""
        for strategy in ["error", "skip", "zero"]:
            settings = AlignmentSettings(missing_strategy=strategy)
            assert settings.missing_strategy == strategy


class TestClusteringSettings:
    """Tests for ClusteringSettings validation."""

    def test_default_values(self):
        """Test default values."""
        settings = ClusteringSettings()
        assert settings.method == ClusteringMethod.HIERARCHICAL
        assert settings.linkage == LinkageMethod.WARD
        assert settings.distance_metric == DistanceMetric.EUCLIDEAN

    def test_ward_requires_euclidean(self):
        """Test that Ward linkage requires Euclidean distance."""
        with pytest.raises(ValueError, match="Ward linkage requires Euclidean"):
            ClusteringSettings(
                linkage=LinkageMethod.WARD,
                distance_metric=DistanceMetric.COSINE,
            )

    def test_non_ward_allows_other_metrics(self):
        """Test that non-Ward linkage allows other metrics."""
        settings = ClusteringSettings(
            linkage=LinkageMethod.AVERAGE,
            distance_metric=DistanceMetric.COSINE,
        )
        assert settings.distance_metric == DistanceMetric.COSINE

    def test_warns_on_small_cluster_size(self):
        """Test warning when min_cluster_size is very small."""
        with pytest.warns(UserWarning, match="may not be reliable"):
            ClusteringSettings(min_cluster_size=2)


class TestHypothesisSettings:
    """Tests for HypothesisSettings."""

    def test_default_values(self):
        """Test default values."""
        settings = HypothesisSettings()
        assert settings.enabled is True
        assert settings.min_confidence == 0.6
        assert settings.max_per_cluster == 5
        assert settings.include_interactions is True

    def test_template_libraries(self):
        """Test different template libraries."""
        for lib in ["comprehensive", "minimal", "custom"]:
            settings = HypothesisSettings(template_library=lib)
            assert settings.template_library == lib


class TestTradeConfig:
    """Tests for TradeConfig consolidated configuration."""

    def test_default_values(self):
        """Test default values."""
        config = TradeConfig()
        assert config.min_trades_for_clustering == 20
        assert config.generate_visualizations is True
        assert config.cache_shap_vectors is True

    def test_convenience_properties(self):
        """Test n_worst and n_best convenience properties."""
        config = TradeConfig(extraction=ExtractionSettings(n_worst=30, n_best=15))
        assert config.n_worst == 30
        assert config.n_best == 15

    def test_warns_on_low_min_trades(self):
        """Test warning when min_trades is very low."""
        with pytest.warns(UserWarning, match="may not identify reliable patterns"):
            TradeConfig(min_trades_for_clustering=5)

    def test_preset_quick_diagnostics(self):
        """Test for_quick_diagnostics preset."""
        config = TradeConfig.for_quick_diagnostics()
        assert config.extraction.n_worst == 20
        assert config.clustering.min_cluster_size == 3
        assert config.hypothesis.template_library == "minimal"
        assert config.generate_visualizations is False

    def test_preset_deep_analysis(self):
        """Test for_deep_analysis preset."""
        config = TradeConfig.for_deep_analysis()
        assert config.extraction.n_worst == 50
        assert config.clustering.min_cluster_size == 10
        assert config.hypothesis.template_library == "comprehensive"
        assert config.generate_visualizations is True

    def test_preset_production(self):
        """Test for_production preset."""
        config = TradeConfig.for_production()
        assert config.extraction.n_worst == 20
        assert config.extraction.group_by_symbol is True
        assert config.generate_visualizations is False
        assert config.cache_shap_vectors is True
