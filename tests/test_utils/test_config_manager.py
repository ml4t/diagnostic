"""Tests for configuration management utilities."""

from __future__ import annotations

import os
import tempfile

import pytest
import yaml

from ml4t.diagnostic.utils.config import (
    ConfigError,
    DataConfig,
    EvaluationConfigManager,
    EvaluatorConfig,
    LoggingConfig,
    QEvalConfig,
    SplitterConfig,
    VisualizationConfig,
    create_example_config,
    load_config,
)


class TestSplitterConfig:
    """Tests for SplitterConfig schema."""

    def test_valid_walk_forward_config(self):
        """Test valid walk-forward splitter configuration."""
        config = SplitterConfig(
            type="WalkForwardCV",
            params={"n_splits": 5, "test_size": 0.2},
        )

        assert config.type == "WalkForwardCV"
        assert config.params["n_splits"] == 5

    def test_valid_combinatorial_config(self):
        """Test valid combinatorial splitter configuration."""
        config = SplitterConfig(
            type="CombinatorialCV",
            params={"n_groups": 10},
        )

        assert config.type == "CombinatorialCV"

    def test_invalid_n_splits(self):
        """Test validation of n_splits range."""
        with pytest.raises(ValueError, match="n_splits must be between 2 and 50"):
            SplitterConfig(
                type="WalkForwardCV",
                params={"n_splits": 1},  # Too low
            )

    def test_invalid_test_size(self):
        """Test validation of test_size range."""
        with pytest.raises(ValueError, match="test_size must be between 0 and 1"):
            SplitterConfig(
                type="WalkForwardCV",
                params={"test_size": 1.5},  # > 1
            )

    def test_invalid_n_groups(self):
        """Test validation of n_groups range."""
        with pytest.raises(ValueError, match="n_groups must be between 2 and 20"):
            SplitterConfig(
                type="CombinatorialCV",
                params={"n_groups": 100},  # Too high
            )


class TestDataConfig:
    """Tests for DataConfig schema."""

    def test_default_values(self):
        """Test default configuration values."""
        config = DataConfig()

        assert config.label_horizon == 20
        assert config.embargo_pct == 0.01
        assert config.min_samples_per_fold == 100

    def test_custom_values(self):
        """Test custom configuration values."""
        config = DataConfig(
            label_horizon=5,
            embargo_pct=0.05,
            min_samples_per_fold=50,
        )

        assert config.label_horizon == 5
        assert config.embargo_pct == 0.05

    def test_invalid_label_horizon(self):
        """Test validation of label_horizon."""
        with pytest.raises(ValueError):
            DataConfig(label_horizon=300)  # > 252


class TestVisualizationConfig:
    """Tests for VisualizationConfig schema."""

    def test_default_values(self):
        """Test default visualization values."""
        config = VisualizationConfig()

        assert config.theme == "default"
        assert config.export_format == "html"
        assert config.include_dashboard is True

    def test_valid_themes(self):
        """Test valid theme options."""
        for theme in ["default", "dark", "light"]:
            config = VisualizationConfig(theme=theme)
            assert config.theme == theme

    def test_valid_export_formats(self):
        """Test valid export format options."""
        for fmt in ["html", "png", "pdf", "svg"]:
            config = VisualizationConfig(export_format=fmt)
            assert config.export_format == fmt


class TestLoggingConfig:
    """Tests for LoggingConfig schema."""

    def test_default_values(self):
        """Test default logging values."""
        config = LoggingConfig()

        assert config.level == "INFO"
        assert config.use_wandb is False

    def test_wandb_requires_project(self):
        """Test that W&B requires project name."""
        with pytest.raises(ValueError, match="wandb_project is required"):
            LoggingConfig(use_wandb=True)  # No project specified

    def test_valid_wandb_config(self):
        """Test valid W&B configuration."""
        config = LoggingConfig(
            use_wandb=True,
            wandb_project="test_project",
            wandb_entity="test_entity",
        )

        assert config.use_wandb is True
        assert config.wandb_project == "test_project"


class TestEvaluatorConfig:
    """Tests for EvaluatorConfig schema."""

    def test_default_values(self):
        """Test default evaluator values."""
        config = EvaluatorConfig()

        assert config.tier == 2
        assert config.confidence_level == 0.05
        assert config.bootstrap_samples == 1000

    def test_valid_tier_range(self):
        """Test valid tier values."""
        for tier in [1, 2, 3]:
            config = EvaluatorConfig(tier=tier)
            assert config.tier == tier

    def test_invalid_tier(self):
        """Test invalid tier validation."""
        with pytest.raises(ValueError):
            EvaluatorConfig(tier=4)

    def test_invalid_confidence_level(self):
        """Test invalid confidence level validation."""
        with pytest.raises(ValueError):
            EvaluatorConfig(confidence_level=1.5)


class TestQEvalConfig:
    """Tests for complete QEvalConfig schema."""

    def test_tier_1_requires_combinatorial(self):
        """Test that tier 1 requires CombinatorialCV."""
        with pytest.raises(ValueError, match="Tier 1.*CombinatorialCV"):
            QEvalConfig(
                evaluation=EvaluatorConfig(tier=1),
                splitter=SplitterConfig(type="WalkForwardCV"),
            )

    def test_valid_tier_1_config(self):
        """Test valid tier 1 configuration."""
        config = QEvalConfig(
            evaluation=EvaluatorConfig(tier=1),
            splitter=SplitterConfig(type="CombinatorialCV"),
        )

        assert config.evaluation.tier == 1

    def test_metrics_non_empty(self):
        """Test that at least one metric is required."""
        with pytest.raises(ValueError):
            QEvalConfig(
                splitter=SplitterConfig(type="WalkForwardCV"),
                metrics=[],
            )


class TestEvaluationConfigManager:
    """Tests for EvaluationConfigManager."""

    def test_init_without_file(self):
        """Test initialization without config file."""
        manager = EvaluationConfigManager()

        assert manager.config is not None
        assert manager.config.splitter.type == "WalkForwardCV"

    def test_get_nested_value(self):
        """Test getting nested configuration value."""
        manager = EvaluationConfigManager()

        tier = manager.get("evaluation.tier")
        assert tier == 2

    def test_get_default_value(self):
        """Test getting default for missing key."""
        manager = EvaluationConfigManager()

        result = manager.get("nonexistent.key", default="default_value")
        assert result == "default_value"

    def test_load_from_yaml(self):
        """Test loading configuration from YAML file."""
        yaml_content = """
evaluation:
  tier: 3
  confidence_level: 0.01

splitter:
  type: WalkForwardCV
  params:
    n_splits: 10
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()

            try:
                manager = EvaluationConfigManager(f.name)

                assert manager.config.evaluation.tier == 3
                assert manager.config.evaluation.confidence_level == 0.01
                assert manager.config.splitter.params["n_splits"] == 10
            finally:
                os.unlink(f.name)

    def test_load_missing_file(self):
        """Test error when loading missing file."""
        with pytest.raises(ConfigError, match="not found"):
            EvaluationConfigManager("/nonexistent/path.yaml")

    def test_load_invalid_yaml(self):
        """Test error for invalid YAML content."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("invalid: yaml: content: [")
            f.flush()

            try:
                with pytest.raises(ConfigError, match="Invalid YAML"):
                    EvaluationConfigManager(f.name)
            finally:
                os.unlink(f.name)

    def test_save_to_yaml(self):
        """Test saving configuration to YAML file."""
        manager = EvaluationConfigManager()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            output_path = f.name

        try:
            manager.save_to_yaml(output_path)

            # Verify file was created and contains valid YAML
            with open(output_path) as f:
                loaded = yaml.safe_load(f)

            assert "evaluation" in loaded
            assert "splitter" in loaded
        finally:
            os.unlink(output_path)

    def test_repr(self):
        """Test string representation."""
        manager = EvaluationConfigManager()

        repr_str = repr(manager)

        assert "EvaluationConfigManager" in repr_str
        assert "tier=" in repr_str

    def test_deep_merge(self):
        """Test deep merging of configurations."""
        manager = EvaluationConfigManager()

        base = {"a": {"b": 1, "c": 2}}
        override = {"a": {"b": 10}}

        result = manager._deep_merge_dicts(base, override)

        assert result["a"]["b"] == 10
        assert result["a"]["c"] == 2

    def test_validate_method(self):
        """Test validate method."""
        manager = EvaluationConfigManager()

        # Should not raise
        manager.validate()


class TestLoadConfig:
    """Tests for load_config function."""

    def test_load_with_path(self):
        """Test loading with explicit path."""
        yaml_content = """
splitter:
  type: WalkForwardCV
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()

            try:
                manager = load_config(f.name)
                assert manager.config.splitter.type == "WalkForwardCV"
            finally:
                os.unlink(f.name)

    def test_load_without_path(self):
        """Test loading without path (uses defaults)."""
        # Clear env var if set
        env_backup = os.environ.pop("QEVAL_CONFIG", None)

        try:
            manager = load_config()
            assert manager.config is not None
        finally:
            if env_backup:
                os.environ["QEVAL_CONFIG"] = env_backup


class TestCreateExampleConfig:
    """Tests for create_example_config function."""

    def test_create_example(self):
        """Test creating example configuration file."""
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            output_path = f.name

        try:
            create_example_config(output_path)

            # Verify file exists and is valid YAML
            with open(output_path) as f:
                content = yaml.safe_load(f)

            assert "evaluation" in content
            assert "splitter" in content
            assert "metrics" in content
        finally:
            os.unlink(output_path)
