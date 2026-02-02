"""Tests for splitter configuration classes."""

import tempfile
from pathlib import Path

import pytest

from ml4t.diagnostic.splitters.config import (
    CombinatorialConfig,
    WalkForwardConfig,
    SplitterConfig,
)


class TestSplitterConfig:
    """Tests for base SplitterConfig class."""

    def test_default_values(self):
        """Test that default values are set correctly."""
        config = SplitterConfig()

        assert config.n_splits == 5
        assert config.label_horizon == 0
        assert config.embargo_td is None
        assert config.align_to_sessions is False
        assert config.session_col == "session_date"
        assert config.isolate_groups is False

    def test_custom_values(self):
        """Test creating config with custom values."""
        config = SplitterConfig(
            n_splits=10,
            label_horizon=5,
            embargo_td=2,
            align_to_sessions=True,
            session_col="trading_session",
            isolate_groups=True,
        )

        assert config.n_splits == 10
        assert config.label_horizon == 5
        assert config.embargo_td == 2
        assert config.align_to_sessions is True
        assert config.session_col == "trading_session"
        assert config.isolate_groups is True

    def test_validation_n_splits_positive(self):
        """Test that n_splits must be positive."""
        with pytest.raises(ValueError, match="greater than 0"):
            SplitterConfig(n_splits=0)

        with pytest.raises(ValueError, match="greater than 0"):
            SplitterConfig(n_splits=-1)

    def test_validation_label_horizon_non_negative(self):
        """Test that label_horizon must be non-negative."""
        with pytest.raises(ValueError, match="greater than or equal to 0"):
            SplitterConfig(label_horizon=-1)

    def test_validation_embargo_td_non_negative(self):
        """Test that embargo_td must be non-negative when specified."""
        with pytest.raises(ValueError, match="greater than or equal to 0"):
            SplitterConfig(embargo_td=-1)

    def test_serialization_to_dict(self):
        """Test converting config to dictionary."""
        config = SplitterConfig(n_splits=3, label_horizon=5)
        config_dict = config.to_dict()

        assert isinstance(config_dict, dict)
        assert config_dict["n_splits"] == 3
        assert config_dict["label_horizon"] == 5

    def test_serialization_from_dict(self):
        """Test loading config from dictionary."""
        data = {"n_splits": 7, "label_horizon": 3, "embargo_td": 1}
        config = SplitterConfig.from_dict(data)

        assert config.n_splits == 7
        assert config.label_horizon == 3
        assert config.embargo_td == 1

    def test_serialization_json_round_trip(self):
        """Test JSON serialization round-trip."""
        original = SplitterConfig(
            n_splits=5,
            label_horizon=2,
            embargo_td=1,
            align_to_sessions=True,
            session_col="session",
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = Path(f.name)

        try:
            original.to_json(temp_path)
            loaded = SplitterConfig.from_json(temp_path)

            assert loaded.n_splits == original.n_splits
            assert loaded.label_horizon == original.label_horizon
            assert loaded.embargo_td == original.embargo_td
            assert loaded.align_to_sessions == original.align_to_sessions
            assert loaded.session_col == original.session_col
        finally:
            temp_path.unlink()

    def test_serialization_yaml_round_trip(self):
        """Test YAML serialization round-trip."""
        original = SplitterConfig(n_splits=10, label_horizon=5, embargo_td=2, isolate_groups=True)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            temp_path = Path(f.name)

        try:
            original.to_yaml(temp_path)
            loaded = SplitterConfig.from_yaml(temp_path)

            assert loaded.n_splits == original.n_splits
            assert loaded.label_horizon == original.label_horizon
            assert loaded.embargo_td == original.embargo_td
            assert loaded.isolate_groups == original.isolate_groups
        finally:
            temp_path.unlink()


class TestWalkForwardConfig:
    """Tests for WalkForwardConfig class."""

    def test_default_values(self):
        """Test that default values are set correctly."""
        config = WalkForwardConfig()

        assert config.n_splits == 5
        assert config.test_size is None
        assert config.train_size is None
        assert config.step_size is None
        assert config.isolate_groups is False  # Default False for walk-forward

    def test_custom_values(self):
        """Test creating config with custom values."""
        config = WalkForwardConfig(
            n_splits=10,
            test_size=100,
            train_size=500,
            step_size=50,
            label_horizon=5,
            embargo_td=2,
        )

        assert config.n_splits == 10
        assert config.test_size == 100
        assert config.train_size == 500
        assert config.step_size == 50
        assert config.label_horizon == 5
        assert config.embargo_td == 2

    def test_step_size_validation(self):
        """Test that step_size must be positive when specified."""
        with pytest.raises(ValueError, match="greater than or equal to 1"):
            WalkForwardConfig(step_size=0)

        with pytest.raises(ValueError, match="greater than or equal to 1"):
            WalkForwardConfig(step_size=-1)

    def test_time_based_size_with_sessions_validation(self):
        """Test that time-based sizes are rejected with session alignment."""
        with pytest.raises(
            ValueError,
            match="align_to_sessions=True does not support time-based size",
        ):
            WalkForwardConfig(
                align_to_sessions=True,
                test_size="4W",  # Time-based not allowed
            )

        with pytest.raises(
            ValueError,
            match="align_to_sessions=True does not support time-based size",
        ):
            WalkForwardConfig(
                align_to_sessions=True,
                train_size="12W",  # Time-based not allowed
            )

    def test_time_based_size_without_sessions_allowed(self):
        """Test that time-based sizes are allowed without session alignment."""
        config = WalkForwardConfig(
            align_to_sessions=False,
            test_size="4W",
            train_size="12W",
        )

        assert config.test_size == "4W"
        assert config.train_size == "12W"

    def test_int_size_with_sessions_allowed(self):
        """Test that integer sizes work with session alignment."""
        config = WalkForwardConfig(
            align_to_sessions=True,
            test_size=5,  # 5 sessions
            train_size=20,  # 20 sessions
        )

        assert config.test_size == 5
        assert config.train_size == 20

    def test_float_size_with_sessions_allowed(self):
        """Test that float proportions work with session alignment."""
        config = WalkForwardConfig(
            align_to_sessions=True,
            test_size=0.2,  # 20% of sessions
            train_size=0.5,  # 50% of sessions
        )

        assert config.test_size == 0.2
        assert config.train_size == 0.5

    def test_json_round_trip(self):
        """Test JSON serialization with walk-forward specific fields."""
        original = WalkForwardConfig(
            n_splits=5,
            test_size=100,
            train_size=500,
            step_size=50,
            label_horizon=5,
            embargo_td=2,
            align_to_sessions=True,
            isolate_groups=True,
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = Path(f.name)

        try:
            original.to_json(temp_path)
            loaded = WalkForwardConfig.from_json(temp_path)

            assert loaded.n_splits == original.n_splits
            assert loaded.test_size == original.test_size
            assert loaded.train_size == original.train_size
            assert loaded.step_size == original.step_size
            assert loaded.label_horizon == original.label_horizon
            assert loaded.embargo_td == original.embargo_td
            assert loaded.align_to_sessions == original.align_to_sessions
            assert loaded.isolate_groups == original.isolate_groups
        finally:
            temp_path.unlink()


class TestCombinatorialConfig:
    """Tests for CombinatorialConfig class."""

    def test_default_values(self):
        """Test that default values are set correctly."""
        config = CombinatorialConfig()

        assert config.n_groups == 8
        assert config.n_test_groups == 2
        assert config.max_combinations is None
        assert config.contiguous_test_blocks is False
        assert config.isolate_groups is True  # Default True for CPCV

    def test_custom_values(self):
        """Test creating config with custom values."""
        config = CombinatorialConfig(
            n_groups=10,
            n_test_groups=3,
            max_combinations=100,
            contiguous_test_blocks=True,
            label_horizon=5,
            embargo_td=2,
        )

        assert config.n_groups == 10
        assert config.n_test_groups == 3
        assert config.max_combinations == 100
        assert config.contiguous_test_blocks is True
        assert config.label_horizon == 5
        assert config.embargo_td == 2

    def test_n_groups_validation(self):
        """Test that n_groups must be > 1."""
        with pytest.raises(ValueError, match="greater than 1"):
            CombinatorialConfig(n_groups=1)

        with pytest.raises(ValueError, match="greater than 1"):
            CombinatorialConfig(n_groups=0)

    def test_n_test_groups_validation(self):
        """Test that n_test_groups must be positive."""
        with pytest.raises(ValueError, match="greater than 0"):
            CombinatorialConfig(n_test_groups=0)

    def test_n_test_groups_less_than_n_groups_validation(self):
        """Test that n_test_groups must be less than n_groups."""
        with pytest.raises(ValueError, match="cannot exceed"):
            CombinatorialConfig(n_groups=5, n_test_groups=5)

        with pytest.raises(ValueError, match="cannot exceed"):
            CombinatorialConfig(n_groups=5, n_test_groups=6)

    def test_max_combinations_validation(self):
        """Test that max_combinations must be positive when specified."""
        with pytest.raises(ValueError, match="greater than 0"):
            CombinatorialConfig(max_combinations=0)

        with pytest.raises(ValueError, match="greater than 0"):
            CombinatorialConfig(max_combinations=-1)

    def test_json_round_trip(self):
        """Test JSON serialization with CPCV specific fields."""
        original = CombinatorialConfig(
            n_groups=10,
            n_test_groups=3,
            max_combinations=50,
            contiguous_test_blocks=True,
            label_horizon=5,
            embargo_td=2,
            align_to_sessions=True,
            isolate_groups=False,  # Opt-out of default True
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = Path(f.name)

        try:
            original.to_json(temp_path)
            loaded = CombinatorialConfig.from_json(temp_path)

            assert loaded.n_groups == original.n_groups
            assert loaded.n_test_groups == original.n_test_groups
            assert loaded.max_combinations == original.max_combinations
            assert loaded.contiguous_test_blocks == original.contiguous_test_blocks
            assert loaded.label_horizon == original.label_horizon
            assert loaded.embargo_td == original.embargo_td
            assert loaded.align_to_sessions == original.align_to_sessions
            assert loaded.isolate_groups == original.isolate_groups
        finally:
            temp_path.unlink()


class TestConfigIntegration:
    """Integration tests for config system with splitters."""

    def test_walk_forward_with_config_object(self):
        """Test creating WalkForwardCV with explicit config object."""
        from ml4t.diagnostic.splitters import WalkForwardConfig, WalkForwardCV

        config = WalkForwardConfig(
            n_splits=5,
            test_size=100,
            train_size=500,
            label_horizon=5,
            embargo_td=2,
            align_to_sessions=True,
            session_col="session_date",
            isolate_groups=True,
        )

        cv = WalkForwardCV(config=config)

        assert cv.n_splits == 5
        assert cv.test_size == 100
        assert cv.train_size == 500
        assert cv.label_horizon == 5
        assert cv.embargo_size == 2  # Config uses embargo_td, splitter exposes as embargo_size
        assert cv.align_to_sessions is True
        assert cv.session_col == "session_date"
        assert cv.isolate_groups is True

    def test_walk_forward_with_params(self):
        """Test creating WalkForwardCV with direct parameters."""
        from ml4t.diagnostic.splitters import WalkForwardCV

        cv = WalkForwardCV(
            n_splits=5,
            test_size=100,
            train_size=500,
            label_horizon=5,
            embargo_size=2,
            align_to_sessions=True,
            session_col="session_date",
            isolate_groups=True,
        )

        # Parameters are stored in config internally
        assert cv.n_splits == 5
        assert cv.test_size == 100
        assert cv.train_size == 500
        assert cv.label_horizon == 5
        assert cv.embargo_size == 2
        assert cv.align_to_sessions is True
        assert cv.session_col == "session_date"
        assert cv.isolate_groups is True

    def test_walk_forward_rejects_mixed_config_and_params(self):
        """Test that specifying both config and params raises error."""
        from ml4t.diagnostic.splitters import WalkForwardConfig, WalkForwardCV

        config = WalkForwardConfig(n_splits=5)

        with pytest.raises(
            ValueError, match="Cannot specify both 'config' and individual parameters"
        ):
            WalkForwardCV(config=config, test_size=100)

    def test_combinatorial_with_config_object(self):
        """Test creating CombinatorialCV with explicit config object."""
        from ml4t.diagnostic.splitters import CombinatorialConfig, CombinatorialCV

        config = CombinatorialConfig(
            n_groups=10,
            n_test_groups=3,
            max_combinations=50,
            label_horizon=5,
            embargo_td=2,
            align_to_sessions=True,
            session_col="session_date",
            isolate_groups=False,
        )

        cv = CombinatorialCV(config=config)

        assert cv.n_groups == 10
        assert cv.n_test_groups == 3
        assert cv.max_combinations == 50
        assert cv.label_horizon == 5
        assert cv.embargo_size == 2
        assert cv.align_to_sessions is True
        assert cv.session_col == "session_date"
        assert cv.isolate_groups is False

    def test_combinatorial_with_params(self):
        """Test creating CombinatorialCV with direct parameters."""
        from ml4t.diagnostic.splitters import CombinatorialCV

        cv = CombinatorialCV(
            n_groups=10,
            n_test_groups=3,
            max_combinations=50,
            label_horizon=5,
            embargo_size=2,
        )

        # Parameters are stored in config internally
        assert cv.n_groups == 10
        assert cv.n_test_groups == 3
        assert cv.max_combinations == 50
        assert cv.label_horizon == 5
        assert cv.embargo_size == 2

    def test_combinatorial_rejects_mixed_config_and_params(self):
        """Test that specifying both config and params raises error."""
        from ml4t.diagnostic.splitters import CombinatorialConfig, CombinatorialCV

        config = CombinatorialConfig(n_groups=10)

        with pytest.raises(
            ValueError, match="Cannot specify both 'config' and individual parameters"
        ):
            CombinatorialCV(config=config, n_test_groups=3)

    def test_config_serialization_and_splitter_creation(self):
        """Test full workflow: create config, serialize, load, create splitter."""
        from ml4t.diagnostic.splitters import WalkForwardConfig, WalkForwardCV

        # Create and serialize config
        original_config = WalkForwardConfig(
            n_splits=5,
            test_size=100,
            label_horizon=5,
            align_to_sessions=True,
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = Path(f.name)

        try:
            # Save config
            original_config.to_json(temp_path)

            # Load config
            loaded_config = WalkForwardConfig.from_json(temp_path)

            # Create splitter from loaded config
            cv = WalkForwardCV(config=loaded_config)

            # Verify splitter has correct parameters
            assert cv.n_splits == 5
            assert cv.test_size == 100
            assert cv.label_horizon == 5
            assert cv.align_to_sessions is True
        finally:
            temp_path.unlink()
