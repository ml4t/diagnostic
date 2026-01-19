"""Tests for the DataFrame adapter."""

import numpy as np
import pandas as pd
import polars as pl
import pytest

from ml4t.diagnostic.backends.adapter import DataFrameAdapter


class TestDataFrameAdapter:
    """Test suite for DataFrameAdapter functionality."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        return {
            "numpy_1d": np.array([1, 2, 3, 4, 5]),
            "numpy_2d": np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
            "pandas_df": pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}),
            "pandas_df_custom_index": pd.DataFrame(
                {"a": [1, 2, 3], "b": [4, 5, 6]},
                index=pd.Index(["x", "y", "z"], name="custom"),
            ),
            "polars_df": pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}),
        }

    def test_to_polars_from_polars(self, sample_data):
        """Test converting Polars DataFrame returns itself."""
        pl_df = sample_data["polars_df"]
        result, index = DataFrameAdapter.to_polars(pl_df)

        assert result is pl_df  # Should be the same object
        assert index is None

    def test_to_polars_from_pandas_default_index(self, sample_data):
        """Test converting Pandas DataFrame with default index."""
        pd_df = sample_data["pandas_df"]
        result, index = DataFrameAdapter.to_polars(pd_df)

        assert isinstance(result, pl.DataFrame)
        assert result.shape == pd_df.shape
        assert list(result.columns) == list(pd_df.columns)
        assert index is None  # Default index is not preserved

        # Check data is preserved
        np.testing.assert_array_equal(result.to_numpy(), pd_df.to_numpy())

    def test_to_polars_from_pandas_custom_index(self, sample_data):
        """Test converting Pandas DataFrame with custom index.

        Note: Custom index is reset to column, metadata is no longer preserved.
        """
        pd_df = sample_data["pandas_df_custom_index"]
        result, index_metadata = DataFrameAdapter.to_polars(pd_df)

        assert isinstance(result, pl.DataFrame)
        assert index_metadata is None  # No longer preserved

        # Index should be reset to a column
        assert "custom" in result.columns
        assert list(result["custom"].to_list()) == ["x", "y", "z"]

    def test_to_polars_from_numpy_1d(self, sample_data):
        """Test converting 1D numpy array."""
        arr = sample_data["numpy_1d"]
        result, index = DataFrameAdapter.to_polars(arr)

        assert isinstance(result, pl.DataFrame)
        assert result.shape == (5, 1)
        assert result.columns == ["column_0"]
        assert index is None

        np.testing.assert_array_equal(result["column_0"].to_numpy(), arr)

    def test_to_polars_from_numpy_1d_with_columns(self, sample_data):
        """Test converting 1D numpy array with custom column name."""
        arr = sample_data["numpy_1d"]
        result, index = DataFrameAdapter.to_polars(arr, columns=["values"])

        assert result.columns == ["values"]
        np.testing.assert_array_equal(result["values"].to_numpy(), arr)

    def test_to_polars_from_numpy_2d(self, sample_data):
        """Test converting 2D numpy array."""
        arr = sample_data["numpy_2d"]
        result, index = DataFrameAdapter.to_polars(arr)

        assert isinstance(result, pl.DataFrame)
        assert result.shape == arr.shape
        assert result.columns == ["column_0", "column_1", "column_2"]
        assert index is None

        np.testing.assert_array_equal(result.to_numpy(), arr)

    def test_to_polars_from_numpy_2d_with_columns(self, sample_data):
        """Test converting 2D numpy array with custom columns."""
        arr = sample_data["numpy_2d"]
        result, index = DataFrameAdapter.to_polars(arr, columns=["x", "y", "z"])

        assert result.columns == ["x", "y", "z"]
        np.testing.assert_array_equal(result.to_numpy(), arr)

    def test_to_polars_invalid_columns_1d(self, sample_data):
        """Test error when wrong number of columns for 1D array."""
        arr = sample_data["numpy_1d"]

        with pytest.raises(ValueError, match="1D array requires exactly 1 column name"):
            DataFrameAdapter.to_polars(arr, columns=["a", "b"])

    def test_to_polars_invalid_columns_2d(self, sample_data):
        """Test error when wrong number of columns for 2D array."""
        arr = sample_data["numpy_2d"]

        with pytest.raises(ValueError, match="Number of columns"):
            DataFrameAdapter.to_polars(arr, columns=["a", "b"])  # Need 3 columns

    def test_to_polars_invalid_array_dims(self):
        """Test error for arrays with more than 2 dimensions."""
        arr = np.ones((2, 3, 4))

        with pytest.raises(ValueError, match="Arrays must be 1D or 2D"):
            DataFrameAdapter.to_polars(arr)

    def test_to_polars_invalid_type(self):
        """Test error for unsupported input type."""
        with pytest.raises(TypeError, match="Data must be a Polars DataFrame"):
            DataFrameAdapter.to_polars([1, 2, 3])

    def test_to_numpy_from_various_types(self, sample_data):
        """Test converting various types to numpy."""
        # From numpy (should return same object)
        arr = sample_data["numpy_2d"]
        assert DataFrameAdapter.to_numpy(arr) is arr

        # From Polars DataFrame
        pl_df = sample_data["polars_df"]
        pl_result = DataFrameAdapter.to_numpy(pl_df)
        assert isinstance(pl_result, np.ndarray)
        assert pl_result.shape == pl_df.shape

        # From Pandas DataFrame
        pd_df = sample_data["pandas_df"]
        pd_result = DataFrameAdapter.to_numpy(pd_df)
        assert isinstance(pd_result, np.ndarray)
        assert pd_result.shape == pd_df.shape

        # From Polars Series
        pl_series = pl.Series([1, 2, 3])
        pl_series_result = DataFrameAdapter.to_numpy(pl_series)
        assert isinstance(pl_series_result, np.ndarray)
        assert pl_series_result.shape == (3,)

        # From Pandas Series
        pd_series = pd.Series([1, 2, 3])
        pd_series_result = DataFrameAdapter.to_numpy(pd_series)
        assert isinstance(pd_series_result, np.ndarray)
        assert pd_series_result.shape == (3,)

    def test_to_numpy_invalid_type(self):
        """Test error for unsupported type in to_numpy."""
        with pytest.raises(TypeError, match="Cannot convert"):
            DataFrameAdapter.to_numpy([1, 2, 3])

    def test_get_shape(self, sample_data):
        """Test getting shape from various types."""
        # 2D numpy array
        assert DataFrameAdapter.get_shape(sample_data["numpy_2d"]) == (3, 3)

        # 1D numpy array
        assert DataFrameAdapter.get_shape(sample_data["numpy_1d"]) == (5, 1)

        # Pandas DataFrame
        assert DataFrameAdapter.get_shape(sample_data["pandas_df"]) == (3, 2)

        # Polars DataFrame
        assert DataFrameAdapter.get_shape(sample_data["polars_df"]) == (3, 2)

    def test_get_shape_invalid_type(self):
        """Test error for invalid type in get_shape."""
        with pytest.raises(TypeError, match="Cannot get shape"):
            DataFrameAdapter.get_shape([1, 2, 3])
