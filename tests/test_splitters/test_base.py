"""Tests for the BaseSplitter abstract class."""

import numpy as np
import pandas as pd
import polars as pl
import pytest

from ml4t.diagnostic.splitters.base import BaseSplitter


class ConcreteSplitter(BaseSplitter):
    """Concrete implementation for testing the base class."""

    def __init__(self, n_splits: int = 3):
        """Initialize the splitter with specified number of splits."""
        self.n_splits = n_splits

    def split(self, X, y=None, groups=None):
        """Simple split implementation for testing."""
        n_samples = self._validate_data(X, y, groups)
        indices = np.arange(n_samples)

        # Simple non-overlapping splits
        fold_size = n_samples // self.n_splits

        for i in range(self.n_splits):
            start = i * fold_size
            end = start + fold_size if i < self.n_splits - 1 else n_samples

            test_indices = indices[start:end]
            train_indices = np.concatenate([indices[:start], indices[end:]])

            yield train_indices, test_indices

    def get_n_splits(self, X=None, y=None, groups=None):
        """Return the number of splits."""
        return self.n_splits


class TestBaseSplitter:
    """Test suite for BaseSplitter functionality."""

    def test_abstract_class_cannot_be_instantiated(self):
        """Test that BaseSplitter cannot be instantiated directly."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            BaseSplitter()

    def test_not_implemented_error(self):
        """Test that get_n_splits raises NotImplementedError if not overridden."""

        class IncompleteSplitter(BaseSplitter):
            def split(self, X, y=None, groups=None):
                yield np.array([0]), np.array([1])

        splitter = IncompleteSplitter()
        with pytest.raises(NotImplementedError, match="must implement get_n_splits"):
            splitter.get_n_splits()

    @pytest.mark.parametrize("n_samples", [10, 50, 100])
    def test_get_n_samples_numpy(self, n_samples):
        """Test _get_n_samples with numpy arrays."""
        splitter = ConcreteSplitter()
        X = np.random.rand(n_samples, 5)
        assert splitter._get_n_samples(X) == n_samples

    def test_get_n_samples_pandas(self):
        """Test _get_n_samples with pandas DataFrame."""
        splitter = ConcreteSplitter()
        X = pd.DataFrame(np.random.rand(20, 5))
        assert splitter._get_n_samples(X) == 20

    def test_get_n_samples_polars(self):
        """Test _get_n_samples with polars DataFrame."""
        splitter = ConcreteSplitter()
        X = pl.DataFrame(np.random.rand(30, 5))
        assert splitter._get_n_samples(X) == 30

    def test_get_n_samples_invalid_type(self):
        """Test _get_n_samples with invalid input type."""
        splitter = ConcreteSplitter()
        with pytest.raises(TypeError, match="must be a Polars DataFrame"):
            splitter._get_n_samples([1, 2, 3])

    def test_validate_data_consistent_lengths(self):
        """Test _validate_data with consistent lengths."""
        splitter = ConcreteSplitter()
        n_samples = 25

        X = np.random.rand(n_samples, 5)
        y = np.random.rand(n_samples)
        groups = np.random.randint(0, 3, n_samples)

        assert splitter._validate_data(X, y, groups) == n_samples

    def test_validate_data_inconsistent_y_length(self):
        """Test _validate_data raises error for inconsistent y length."""
        splitter = ConcreteSplitter()

        X = np.random.rand(20, 5)
        y = np.random.rand(15)  # Wrong length

        with pytest.raises(ValueError, match="X and y have inconsistent lengths"):
            splitter._validate_data(X, y)

    def test_validate_data_inconsistent_groups_length(self):
        """Test _validate_data raises error for inconsistent groups length."""
        splitter = ConcreteSplitter()

        X = np.random.rand(20, 5)
        groups = np.random.randint(0, 3, 10)  # Wrong length

        with pytest.raises(ValueError, match="X and groups have inconsistent lengths"):
            splitter._validate_data(X, groups=groups)

    def test_validate_data_mixed_types(self):
        """Test _validate_data with mixed DataFrame/Series types."""
        splitter = ConcreteSplitter()

        # Pandas DataFrame with Pandas Series
        X_pd = pd.DataFrame(np.random.rand(15, 5))
        y_pd = pd.Series(np.random.rand(15))
        assert splitter._validate_data(X_pd, y_pd) == 15

        # Polars DataFrame with Polars Series
        X_pl = pl.DataFrame(np.random.rand(10, 5))
        y_pl = pl.Series(np.random.rand(10))
        assert splitter._validate_data(X_pl, y_pl) == 10

    def test_concrete_splitter_basic_functionality(self):
        """Test that a concrete implementation works correctly."""
        splitter = ConcreteSplitter(n_splits=3)
        X = np.random.rand(30, 5)

        splits = list(splitter.split(X))

        # Check we get the right number of splits
        assert len(splits) == 3

        # Check all indices are used
        all_train_indices = []
        all_test_indices = []

        for train, test in splits:
            all_train_indices.extend(train)
            all_test_indices.extend(test)

            # Check no overlap between train and test
            assert len(np.intersect1d(train, test)) == 0

        # Check all indices are covered
        all_indices = np.sort(np.concatenate([all_train_indices, all_test_indices]))
        expected_indices = np.repeat(np.arange(30), 3)  # Each index appears in 3 splits
        np.testing.assert_array_equal(np.sort(all_indices), expected_indices)

    def test_repr(self):
        """Test string representation."""
        splitter = ConcreteSplitter()
        assert repr(splitter) == "ConcreteSplitter()"


class TestExtractTimestamps:
    """Tests for _extract_timestamps method - Polars/Pandas timestamp handling."""

    def test_polars_with_timestamp_col(self):
        """Test timestamp extraction from Polars DataFrame with timestamp_col."""
        splitter = ConcreteSplitter()
        dates = pd.date_range("2020-01-01", periods=10, freq="D", tz="UTC")
        X = pl.DataFrame(
            {
                "feature": np.arange(10),
                "timestamp": dates,
            }
        )

        timestamps = splitter._extract_timestamps(X, timestamp_col="timestamp")

        assert isinstance(timestamps, pd.DatetimeIndex)
        assert len(timestamps) == 10
        assert timestamps.tz is not None  # Should be timezone-aware

    def test_polars_without_timestamp_col_returns_none(self):
        """Test that Polars DataFrame without timestamp_col returns None."""
        splitter = ConcreteSplitter()
        X = pl.DataFrame({"feature": np.arange(10)})

        result = splitter._extract_timestamps(X, timestamp_col=None)
        assert result is None

    def test_polars_missing_timestamp_col_raises(self):
        """Test that missing timestamp_col raises ValueError."""
        splitter = ConcreteSplitter()
        X = pl.DataFrame({"feature": np.arange(10)})

        with pytest.raises(ValueError, match="timestamp_col='missing' not found"):
            splitter._extract_timestamps(X, timestamp_col="missing")

    def test_polars_non_datetime_col_raises(self):
        """Test that non-datetime timestamp_col raises ValueError."""
        splitter = ConcreteSplitter()
        X = pl.DataFrame(
            {
                "feature": np.arange(10),
                "timestamp": np.arange(10),  # Not datetime
            }
        )

        with pytest.raises(ValueError, match="must be datetime type"):
            splitter._extract_timestamps(X, timestamp_col="timestamp")

    def test_polars_naive_datetime_gets_utc(self):
        """Test that naive datetime gets UTC localization."""
        splitter = ConcreteSplitter()
        # Create naive datetime (no timezone)
        dates = pd.date_range("2020-01-01", periods=10, freq="D")  # No tz
        X = pl.DataFrame(
            {
                "feature": np.arange(10),
                "timestamp": dates,
            }
        )

        timestamps = splitter._extract_timestamps(X, timestamp_col="timestamp")

        assert timestamps.tz is not None
        assert str(timestamps.tz) == "UTC"

    def test_pandas_with_datetime_index(self):
        """Test timestamp extraction from Pandas DataFrame with DatetimeIndex."""
        splitter = ConcreteSplitter()
        dates = pd.date_range("2020-01-01", periods=10, freq="D", tz="UTC")
        X = pd.DataFrame({"feature": np.arange(10)}, index=dates)

        timestamps = splitter._extract_timestamps(X)

        assert isinstance(timestamps, pd.DatetimeIndex)
        assert len(timestamps) == 10

    def test_pandas_with_timestamp_col_fallback(self):
        """Test timestamp extraction from Pandas DataFrame via timestamp_col."""
        splitter = ConcreteSplitter()
        dates = pd.date_range("2020-01-01", periods=10, freq="D", tz="UTC")
        X = pd.DataFrame(
            {
                "feature": np.arange(10),
                "timestamp": dates,
            }
        )  # Integer index, not DatetimeIndex

        timestamps = splitter._extract_timestamps(X, timestamp_col="timestamp")

        assert isinstance(timestamps, pd.DatetimeIndex)
        assert len(timestamps) == 10

    def test_pandas_without_datetime_returns_none(self):
        """Test that Pandas DataFrame without datetime info returns None."""
        splitter = ConcreteSplitter()
        X = pd.DataFrame({"feature": np.arange(10)})

        result = splitter._extract_timestamps(X)
        assert result is None

    def test_numpy_returns_none(self):
        """Test that numpy array always returns None."""
        splitter = ConcreteSplitter()
        X = np.random.rand(10, 5)

        result = splitter._extract_timestamps(X)
        assert result is None

        # Even with timestamp_col specified
        result = splitter._extract_timestamps(X, timestamp_col="timestamp")
        assert result is None


class TestGetUniqueSessions:
    """Tests for _get_unique_sessions method - Polars/Pandas session handling."""

    def test_polars_unique_sessions(self):
        """Test unique session extraction from Polars DataFrame."""
        splitter = ConcreteSplitter()
        dates = pd.date_range("2020-01-01", periods=5, freq="D", tz="UTC")
        # Create multiple samples per session
        X = pl.DataFrame(
            {
                "feature": np.arange(15),
                "session": np.repeat(dates, 3),
            }
        )

        sessions = splitter._get_unique_sessions(X, session_col="session")

        assert isinstance(sessions, pl.Series)
        assert len(sessions) == 5
        # Should be sorted
        assert sessions.is_sorted()

    def test_pandas_unique_sessions(self):
        """Test unique session extraction from Pandas DataFrame."""
        splitter = ConcreteSplitter()
        dates = pd.date_range("2020-01-01", periods=5, freq="D", tz="UTC")
        X = pd.DataFrame(
            {
                "feature": np.arange(15),
                "session": np.repeat(dates, 3),
            }
        )

        sessions = splitter._get_unique_sessions(X, session_col="session")

        assert isinstance(sessions, pd.Series)
        assert len(sessions) == 5
        # Should be sorted
        assert sessions.is_monotonic_increasing

    def test_polars_sessions_preserves_appearance_order(self):
        """Test that Polars sessions are returned in order of first appearance.

        For time-series CV, data should be pre-sorted by timestamp, so appearance
        order equals chronological order. This avoids silent bugs when session IDs
        are not naturally sortable (e.g., string session IDs like "session_C").
        """
        splitter = ConcreteSplitter()
        # Create data with sessions in a specific order
        # Note: Data should be pre-sorted by time in practice
        dates = pd.to_datetime(["2020-01-03", "2020-01-01", "2020-01-02"], utc=True)
        X = pl.DataFrame(
            {
                "feature": [1, 2, 3],
                "session": dates,
            }
        )

        sessions = splitter._get_unique_sessions(X, session_col="session")

        # Should be in order of first appearance (not sorted by value)
        expected = pd.to_datetime(["2020-01-03", "2020-01-01", "2020-01-02"], utc=True)
        assert sessions.to_list() == list(expected)
