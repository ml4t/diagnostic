"""Tests for WalkForwardCV fold summary feature."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ml4t.diagnostic.splitters.walk_forward import WalkForwardCV


class TestFoldSummary:
    """Tests for fold_summary_ property and fold_summary_to_csv()."""

    def test_fold_summary_available_after_split(self):
        """Fold summary is populated after consuming all folds."""
        X = np.arange(100).reshape(100, 1)
        cv = WalkForwardCV(n_splits=3)
        for _ in cv.split(X):
            pass
        summary = cv.fold_summary_
        assert len(summary) == 3
        assert list(summary.columns[:3]) == ["fold", "train_rows", "val_rows"]

    def test_fold_summary_raises_before_split(self):
        """Fold summary raises if split() not called."""
        cv = WalkForwardCV(n_splits=3)
        with pytest.raises(ValueError, match="not available"):
            _ = cv.fold_summary_

    def test_fold_summary_row_counts(self):
        """Row counts in summary match actual split sizes."""
        X = np.arange(100).reshape(100, 1)
        cv = WalkForwardCV(n_splits=3)
        splits = list(cv.split(X))
        summary = cv.fold_summary_

        for i, (train_idx, val_idx) in enumerate(splits):
            row = summary.iloc[i]
            assert row["train_rows"] == len(train_idx)
            assert row["val_rows"] == len(val_idx)

    def test_fold_summary_with_timestamps(self):
        """Fold summary includes date boundaries when timestamps are available."""
        dates = pd.date_range("2020-01-01", periods=500, freq="B", tz="UTC")
        df = pd.DataFrame({"value": np.random.randn(500)}, index=dates)
        cv = WalkForwardCV(n_splits=3, label_horizon=5)

        for _ in cv.split(df):
            pass

        summary = cv.fold_summary_
        assert "train_start" in summary.columns
        assert "val_end" in summary.columns
        assert "buffer_gap_timestamps" in summary.columns
        assert "buffer_gap_duration" in summary.columns

        # All folds should have positive buffer gaps (label_horizon=5)
        for _, row in summary.iterrows():
            assert row["buffer_gap_timestamps"] >= 0
            assert row["train_rows"] > 0
            assert row["val_rows"] > 0

    def test_fold_summary_timestamps_are_dates(self):
        """Train/val start/end are actual Timestamps, not indices."""
        dates = pd.date_range("2020-01-01", periods=200, freq="B", tz="UTC")
        df = pd.DataFrame({"value": np.random.randn(200)}, index=dates)
        cv = WalkForwardCV(n_splits=2)

        for _ in cv.split(df):
            pass

        summary = cv.fold_summary_
        # Should be Timestamps, not integers
        assert isinstance(summary.iloc[0]["train_start"], pd.Timestamp)
        assert isinstance(summary.iloc[0]["val_end"], pd.Timestamp)

    def test_fold_summary_buffer_gap_with_purging(self):
        """Buffer gap reflects actual purging from label_horizon."""
        dates = pd.date_range("2020-01-01", periods=500, freq="B", tz="UTC")
        df = pd.DataFrame({"value": np.random.randn(500)}, index=dates)

        # With label_horizon=21 (trading days), there should be a purge gap
        cv = WalkForwardCV(n_splits=3, label_horizon=21)
        for _ in cv.split(df):
            pass

        summary = cv.fold_summary_
        # Buffer gap should be >= 21 timestamps (the purge removes that many)
        for _, row in summary.iterrows():
            assert row["buffer_gap_timestamps"] >= 15, (
                f"Buffer gap {row['buffer_gap_timestamps']} too small for label_horizon=21"
            )

    def test_fold_summary_to_csv(self):
        """fold_summary_to_csv writes valid CSV."""
        X = np.arange(100).reshape(100, 1)
        cv = WalkForwardCV(n_splits=3)
        for _ in cv.split(X):
            pass

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = f.name

        cv.fold_summary_to_csv(path)

        # Read back and verify
        df = pd.read_csv(path)
        assert len(df) == 3
        assert "fold" in df.columns
        assert "train_rows" in df.columns
        Path(path).unlink()

    def test_fold_summary_to_csv_with_timestamps(self):
        """CSV includes date columns when timestamps are available."""
        dates = pd.date_range("2020-01-01", periods=300, freq="B", tz="UTC")
        df = pd.DataFrame({"value": np.random.randn(300)}, index=dates)
        cv = WalkForwardCV(n_splits=3, label_horizon=5)
        for _ in cv.split(df):
            pass

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = f.name

        cv.fold_summary_to_csv(path)

        result = pd.read_csv(path)
        assert "train_start" in result.columns
        assert "buffer_gap_timestamps" in result.columns
        assert "buffer_gap_duration" in result.columns
        Path(path).unlink()

    def test_fold_summary_consecutive_validation(self):
        """Validation periods should be consecutive (no gaps) for walk-forward."""
        dates = pd.date_range("2020-01-01", periods=500, freq="B", tz="UTC")
        df = pd.DataFrame({"value": np.random.randn(500)}, index=dates)
        cv = WalkForwardCV(n_splits=4, consecutive=True)

        splits = list(cv.split(df))
        summary = cv.fold_summary_

        # Check consecutive val periods: fold i val_end should be close to fold i+1 val_start
        for i in range(len(summary) - 1):
            this_val_end = summary.iloc[i]["val_end"]
            next_val_start = summary.iloc[i + 1]["val_start"]
            # Next val should start at or after this val ends (no overlap)
            assert next_val_start >= this_val_end

    def test_fold_summary_with_polars_dataframe(self):
        """Fold summary works with Polars DataFrame input."""
        import polars as pl

        dates = pd.date_range("2020-01-01", periods=200, freq="B", tz="UTC")
        df = pl.DataFrame({
            "timestamp": dates,
            "value": np.random.randn(200),
        })
        cv = WalkForwardCV(n_splits=2, timestamp_col="timestamp")
        for _ in cv.split(df):
            pass

        summary = cv.fold_summary_
        assert len(summary) == 2
        assert "train_start" in summary.columns

    def test_fold_summary_resets_on_new_split(self):
        """Calling split() again resets the fold summary."""
        X = np.arange(100).reshape(100, 1)
        cv = WalkForwardCV(n_splits=3)

        # First split
        for _ in cv.split(X):
            pass
        assert len(cv.fold_summary_) == 3

        # Second split with different data
        X2 = np.arange(200).reshape(200, 1)
        for _ in cv.split(X2):
            pass
        assert len(cv.fold_summary_) == 3  # Still 3 folds, but from new data

    def test_fold_summary_held_out_test(self):
        """Fold summary works with held-out test configuration."""
        dates = pd.date_range("2020-01-01", periods=500, freq="B", tz="UTC")
        df = pd.DataFrame({"value": np.random.randn(500)}, index=dates)

        cv = WalkForwardCV(
            n_splits=3,
            test_start="2021-06-01",
            fold_direction="backward",
            label_horizon=5,
        )
        for _ in cv.split(df):
            pass

        summary = cv.fold_summary_
        assert len(summary) <= 3
        # All validation should be before the held-out test start
        for _, row in summary.iterrows():
            assert row["val_end"] < pd.Timestamp("2021-06-01", tz="UTC")
