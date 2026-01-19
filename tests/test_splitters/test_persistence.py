"""Tests for fold persistence module."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ml4t.diagnostic.splitters import (
    CombinatorialPurgedCV,
    PurgedWalkForwardConfig,
    PurgedWalkForwardCV,
    load_config,
    load_folds,
    save_config,
    save_folds,
    verify_folds,
)


class TestSaveFolds:
    """Test suite for save_folds function."""

    def test_save_folds_basic(self):
        """Test basic fold saving with numpy array."""
        X = np.arange(100).reshape(100, 1)
        cv = PurgedWalkForwardCV(n_splits=3, label_horizon=0, embargo_size=0)
        folds = list(cv.split(X))

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "folds.json"
            save_folds(folds, X, filepath)

            assert filepath.exists()
            assert filepath.stat().st_size > 0

    def test_save_folds_with_metadata(self):
        """Test saving folds with metadata."""
        X = np.arange(100).reshape(100, 1)
        cv = PurgedWalkForwardCV(n_splits=3)
        folds = list(cv.split(X))

        metadata = {
            "splitter": "PurgedWalkForwardCV",
            "n_splits": 3,
            "dataset": "test_data",
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "folds.json"
            save_folds(folds, X, filepath, metadata=metadata)

            # Load and verify metadata
            loaded_folds, loaded_metadata = load_folds(filepath)
            assert loaded_metadata == metadata

    def test_save_folds_with_timestamps_pandas(self):
        """Test saving folds with timestamps from pandas DataFrame."""
        dates = pd.date_range("2020-01-01", periods=100, freq="D", tz="UTC")
        X = pd.DataFrame({"feature": np.arange(100)}, index=dates)

        cv = PurgedWalkForwardCV(n_splits=3)
        folds = list(cv.split(X))

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "folds.json"
            save_folds(folds, X, filepath, include_timestamps=True)

            # Verify timestamps are saved
            import json

            with filepath.open() as f:
                data = json.load(f)

            assert "timestamps" in data
            assert len(data["timestamps"]) == 100
            assert "train_start" in data["folds"][0]
            assert "test_end" in data["folds"][0]

    def test_save_folds_without_timestamps(self):
        """Test saving folds without timestamp extraction."""
        dates = pd.date_range("2020-01-01", periods=100, freq="D", tz="UTC")
        X = pd.DataFrame({"feature": np.arange(100)}, index=dates)

        cv = PurgedWalkForwardCV(n_splits=3)
        folds = list(cv.split(X))

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "folds.json"
            save_folds(folds, X, filepath, include_timestamps=False)

            import json

            with filepath.open() as f:
                data = json.load(f)

            assert "timestamps" not in data
            assert "train_start" not in data["folds"][0]

    def test_save_folds_creates_directory(self):
        """Test that save_folds creates parent directories if needed."""
        X = np.arange(100).reshape(100, 1)
        cv = PurgedWalkForwardCV(n_splits=3)
        folds = list(cv.split(X))

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "nested" / "dir" / "folds.json"
            save_folds(folds, X, filepath)

            assert filepath.exists()


class TestLoadFolds:
    """Test suite for load_folds function."""

    def test_load_folds_basic(self):
        """Test basic fold loading."""
        X = np.arange(100).reshape(100, 1)
        cv = PurgedWalkForwardCV(n_splits=3, label_horizon=0, embargo_size=0)
        original_folds = list(cv.split(X))

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "folds.json"
            save_folds(original_folds, X, filepath)

            loaded_folds, metadata = load_folds(filepath)

            assert len(loaded_folds) == len(original_folds)

            for (orig_train, orig_test), (load_train, load_test) in zip(
                original_folds, loaded_folds, strict=False
            ):
                np.testing.assert_array_equal(orig_train, load_train)
                np.testing.assert_array_equal(orig_test, load_test)

    def test_load_folds_with_metadata(self):
        """Test loading folds preserves metadata."""
        X = np.arange(100).reshape(100, 1)
        cv = PurgedWalkForwardCV(n_splits=3)
        folds = list(cv.split(X))

        metadata = {"splitter": "test", "version": "1.0"}

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "folds.json"
            save_folds(folds, X, filepath, metadata=metadata)

            loaded_folds, loaded_metadata = load_folds(filepath)

            assert loaded_metadata == metadata

    def test_load_folds_file_not_found(self):
        """Test that loading non-existent file raises error."""
        with pytest.raises(FileNotFoundError, match="Fold file not found"):
            load_folds("nonexistent_file.json")

    def test_load_folds_invalid_version(self):
        """Test that invalid version raises error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "folds.json"

            # Create file with invalid version
            import json

            data = {"version": "2.0", "folds": []}
            with filepath.open("w") as f:
                json.dump(data, f)

            with pytest.raises(ValueError, match="Unsupported fold file version"):
                load_folds(filepath)


class TestSaveLoadConfig:
    """Test suite for config save/load functions."""

    def test_save_load_config_walk_forward(self):
        """Test saving and loading walk-forward config."""
        original = PurgedWalkForwardConfig(
            n_splits=5,
            test_size=100,
            label_horizon=5,
            embargo_td=2,
            align_to_sessions=True,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "config.json"
            save_config(original, filepath)

            loaded = load_config(filepath, PurgedWalkForwardConfig)

            assert loaded.n_splits == original.n_splits
            assert loaded.test_size == original.test_size
            assert loaded.label_horizon == original.label_horizon
            assert loaded.embargo_td == original.embargo_td
            assert loaded.align_to_sessions == original.align_to_sessions


class TestVerifyFolds:
    """Test suite for verify_folds function."""

    def test_verify_folds_valid(self):
        """Test verification of valid folds."""
        X = np.arange(100).reshape(100, 1)
        cv = PurgedWalkForwardCV(n_splits=3, label_horizon=0, embargo_size=0)
        folds = list(cv.split(X))

        stats = verify_folds(folds, n_samples=100)

        assert stats["valid"] is True
        assert len(stats["errors"]) == 0
        assert stats["n_folds"] == 3
        assert stats["n_samples"] == 100
        assert 0.0 < stats["coverage"] <= 1.0

    def test_verify_folds_statistics(self):
        """Test that verification computes correct statistics."""
        X = np.arange(100).reshape(100, 1)
        cv = PurgedWalkForwardCV(n_splits=3, label_horizon=0, embargo_size=0)
        folds = list(cv.split(X))

        stats = verify_folds(folds, n_samples=100)

        assert "train_sizes" in stats
        assert "test_sizes" in stats
        assert "avg_train_size" in stats
        assert "avg_test_size" in stats
        assert "std_train_size" in stats
        assert "std_test_size" in stats
        assert len(stats["train_sizes"]) == 3
        assert len(stats["test_sizes"]) == 3

    def test_verify_folds_overlap_detection(self):
        """Test detection of overlapping train/test indices."""
        # Create invalid folds with overlap
        folds = [
            (np.array([0, 1, 2, 3, 4]), np.array([3, 4, 5, 6, 7])),  # Overlap: 3, 4
        ]

        stats = verify_folds(folds, n_samples=10)

        assert stats["valid"] is False
        assert len(stats["errors"]) > 0
        assert "overlapping" in stats["errors"][0].lower()

    def test_verify_folds_out_of_range(self):
        """Test detection of out-of-range indices."""
        # Create invalid folds with indices outside valid range
        folds = [
            (np.array([0, 1, 2, 100, 101]), np.array([5, 6, 7])),  # Train indices out of range
        ]

        stats = verify_folds(folds, n_samples=10)

        assert stats["valid"] is False
        assert any("out of range" in error.lower() for error in stats["errors"])

    def test_verify_folds_coverage(self):
        """Test coverage computation."""
        X = np.arange(100).reshape(100, 1)
        cv = PurgedWalkForwardCV(n_splits=5, label_horizon=0, embargo_size=0)
        folds = list(cv.split(X))

        stats = verify_folds(folds, n_samples=100)

        # With no purging/embargo, coverage should be high
        assert stats["coverage"] > 0.9
        assert stats["train_coverage"] > 0.0
        assert stats["test_coverage"] > 0.0


class TestIntegration:
    """Integration tests for full save/load workflows."""

    def test_save_load_walk_forward_workflow(self):
        """Test complete workflow: create splitter, save folds, load and reuse."""
        # Create original data and splitter
        dates = pd.date_range("2020-01-01", periods=100, freq="D", tz="UTC")
        X = pd.DataFrame({"feature": np.arange(100)}, index=dates)

        cv = PurgedWalkForwardCV(
            n_splits=5,
            test_size=10,
            label_horizon=pd.Timedelta("2D"),
            embargo_size=pd.Timedelta("1D"),
        )

        # Generate and save folds
        original_folds = list(cv.split(X))

        with tempfile.TemporaryDirectory() as tmpdir:
            folds_path = Path(tmpdir) / "folds.json"
            save_folds(
                original_folds,
                X,
                folds_path,
                metadata={"strategy": "walk_forward", "n_splits": 5},
            )

            # Load folds and verify they match
            loaded_folds, metadata = load_folds(folds_path)

            assert metadata["strategy"] == "walk_forward"
            assert len(loaded_folds) == len(original_folds)

            # Verify fold integrity
            stats = verify_folds(loaded_folds, n_samples=100)
            assert stats["valid"] is True

    def test_save_load_combinatorial_workflow(self):
        """Test workflow with CombinatorialPurgedCV."""
        X = np.arange(200).reshape(200, 1)
        cv = CombinatorialPurgedCV(n_groups=5, n_test_groups=2, label_horizon=5, embargo_size=3)

        original_folds = list(cv.split(X))

        with tempfile.TemporaryDirectory() as tmpdir:
            folds_path = Path(tmpdir) / "cpcv_folds.json"
            save_folds(
                original_folds,
                X,
                folds_path,
                metadata={"strategy": "CPCV", "n_groups": 5},
            )

            loaded_folds, metadata = load_folds(folds_path)

            assert metadata["strategy"] == "CPCV"
            assert len(loaded_folds) == len(original_folds)

            stats = verify_folds(loaded_folds, n_samples=200)
            assert stats["valid"] is True

    def test_config_and_folds_workflow(self):
        """Test saving both config and folds for complete reproducibility."""
        X = np.arange(100).reshape(100, 1)

        # Create and save config
        config = PurgedWalkForwardConfig(
            n_splits=5,
            test_size=15,
            label_horizon=3,
            embargo_td=2,
        )

        cv = PurgedWalkForwardCV(config=config)
        folds = list(cv.split(X))

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            folds_path = Path(tmpdir) / "folds.json"

            # Save both
            save_config(config, config_path)
            save_folds(folds, X, folds_path)

            # Load both
            loaded_config = load_config(config_path, PurgedWalkForwardConfig)
            loaded_folds, _ = load_folds(folds_path)

            # Recreate splitter with loaded config
            cv_recreated = PurgedWalkForwardCV(config=loaded_config)

            # Verify parameters match
            assert cv_recreated.n_splits == cv.n_splits
            assert cv_recreated.test_size == cv.test_size
            assert cv_recreated.label_horizon == cv.label_horizon

            # Verify folds match
            assert len(loaded_folds) == len(folds)
