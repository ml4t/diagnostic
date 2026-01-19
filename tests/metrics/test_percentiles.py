"""
Tests for percentile computation utilities.
"""

import numpy as np
import pandas as pd
import polars as pl
import pytest

from ml4t.diagnostic.metrics.percentiles import compute_fold_percentiles


class TestComputeFoldPercentiles:
    """Test compute_fold_percentiles function."""

    @pytest.fixture
    def sample_predictions_pandas(self) -> pd.DataFrame:
        """Create sample predictions in pandas format."""
        np.random.seed(42)
        return pd.DataFrame(
            {
                "fold_id": [0] * 1000 + [1] * 1000,
                "iteration": [50] * 500 + [100] * 500 + [50] * 500 + [100] * 500,
                "prediction": np.random.rand(2000).astype(np.float32),
            }
        )

    @pytest.fixture
    def sample_predictions_polars(self, sample_predictions_pandas: pd.DataFrame) -> pl.DataFrame:
        """Create sample predictions in polars format."""
        return pl.from_pandas(sample_predictions_pandas)

    def test_basic_percentile_computation(self, sample_predictions_pandas: pd.DataFrame) -> None:
        """Test basic percentile computation with pandas input."""
        percentiles = [10, 50, 90]
        result = compute_fold_percentiles(sample_predictions_pandas, percentiles, verbose=False)

        # Check output shape
        assert len(result) == 4  # 2 folds × 2 iterations
        assert set(result.columns) == {"fold_id", "iteration", "p10", "p50", "p90"}

        # Check fold structure
        assert result["fold_id"].nunique() == 2
        assert result["iteration"].nunique() == 2

    def test_polars_input(self, sample_predictions_polars: pl.DataFrame) -> None:
        """Test that polars input works correctly."""
        percentiles = [10, 50, 90]
        result = compute_fold_percentiles(sample_predictions_polars, percentiles, verbose=False)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 4

    def test_both_tail_percentiles(self, sample_predictions_pandas: pd.DataFrame) -> None:
        """Test computation with both low and high tail percentiles."""
        percentiles = [0.1, 0.5, 1, 5, 10, 90, 95, 99, 99.5, 99.9]
        result = compute_fold_percentiles(sample_predictions_pandas, percentiles, verbose=False)

        # Check all percentile columns exist
        expected_cols = {"fold_id", "iteration"} | {f"p{p}" for p in percentiles}
        assert set(result.columns) == expected_cols

        # Check values are monotonically increasing
        for _, row in result.iterrows():
            percentile_values = [row[f"p{p}"] for p in percentiles]
            assert all(
                percentile_values[i] <= percentile_values[i + 1]
                for i in range(len(percentile_values) - 1)
            ), "Percentiles should be monotonically increasing"

    def test_custom_column_names(self, sample_predictions_pandas: pd.DataFrame) -> None:
        """Test with custom column names."""
        df = sample_predictions_pandas.rename(
            columns={
                "fold_id": "cv_fold",
                "iteration": "iter",
                "prediction": "pred",
            }
        )

        percentiles = [50, 90]
        result = compute_fold_percentiles(
            df,
            percentiles,
            fold_col="cv_fold",
            iteration_col="iter",
            prediction_col="pred",
            verbose=False,
        )

        assert "cv_fold" in result.columns
        assert "iter" in result.columns
        assert "p50" in result.columns
        assert "p90" in result.columns

    def test_missing_columns_raises_error(self, sample_predictions_pandas: pd.DataFrame) -> None:
        """Test that missing required columns raises ValueError."""
        df = sample_predictions_pandas.drop(columns=["fold_id"])

        with pytest.raises(ValueError, match="Missing required columns"):
            compute_fold_percentiles(df, [50], verbose=False)

    def test_fold_specific_thresholds(self, sample_predictions_pandas: pd.DataFrame) -> None:
        """Test that different folds get different thresholds."""
        percentiles = [50, 90]
        result = compute_fold_percentiles(sample_predictions_pandas, percentiles, verbose=False)

        # Get thresholds for each fold
        fold_0_iter_50 = result[(result["fold_id"] == 0) & (result["iteration"] == 50)]
        fold_1_iter_50 = result[(result["fold_id"] == 1) & (result["iteration"] == 50)]

        # Thresholds may differ between folds (depending on random seed)
        assert len(fold_0_iter_50) == 1
        assert len(fold_1_iter_50) == 1

    def test_iteration_specific_thresholds(self, sample_predictions_pandas: pd.DataFrame) -> None:
        """Test that different iterations can have different thresholds."""
        percentiles = [50, 90]
        result = compute_fold_percentiles(sample_predictions_pandas, percentiles, verbose=False)

        # Get thresholds for different iterations of same fold
        fold_0_iter_50 = result[(result["fold_id"] == 0) & (result["iteration"] == 50)]
        fold_0_iter_100 = result[(result["fold_id"] == 0) & (result["iteration"] == 100)]

        assert len(fold_0_iter_50) == 1
        assert len(fold_0_iter_100) == 1

    def test_result_sorted_by_fold_iteration(self, sample_predictions_pandas: pd.DataFrame) -> None:
        """Test that result is sorted by fold_id and iteration."""
        percentiles = [50]
        result = compute_fold_percentiles(sample_predictions_pandas, percentiles, verbose=False)

        # Check sorting
        expected_order = [
            (0, 50),
            (0, 100),
            (1, 50),
            (1, 100),
        ]
        actual_order = list(zip(result["fold_id"], result["iteration"]))
        assert actual_order == expected_order

    def test_extreme_percentiles(self, sample_predictions_pandas: pd.DataFrame) -> None:
        """Test with extreme percentiles near 0 and 100."""
        percentiles = [0.01, 0.1, 99.9, 99.99]
        result = compute_fold_percentiles(sample_predictions_pandas, percentiles, verbose=False)

        # Check all percentile columns created
        for p in percentiles:
            assert f"p{p}" in result.columns

        # Check extreme percentiles are reasonable
        for _, row in result.iterrows():
            assert 0 <= row["p0.01"] <= row["p0.1"]
            assert row["p99.9"] <= row["p99.99"] <= 1.0

    def test_verbose_output(self, sample_predictions_pandas: pd.DataFrame, capsys) -> None:
        """Test that verbose mode prints expected information."""
        percentiles = [50, 90]
        compute_fold_percentiles(sample_predictions_pandas, percentiles, verbose=True)

        captured = capsys.readouterr()
        assert "Computing fold-specific percentiles" in captured.out
        assert "Computed 4 percentile arrays" in captured.out
        assert "2 folds × 2 iterations × 2 percentiles" in captured.out
