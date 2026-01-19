"""Tests for returns validation utilities."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from ml4t.diagnostic.validation.dataframe import ValidationError
from ml4t.diagnostic.validation.returns import (
    ReturnsValidator,
    validate_bounds,
    validate_returns,
)


class TestReturnsValidator:
    """Tests for ReturnsValidator class."""

    @pytest.fixture
    def normal_returns(self) -> pl.Series:
        """Create normal returns series."""
        np.random.seed(42)
        return pl.Series("returns", np.random.normal(0.001, 0.02, 100))

    @pytest.fixture
    def returns_with_nulls(self) -> pl.Series:
        """Create returns series with null values."""
        return pl.Series("returns", [0.01, None, -0.02, None, 0.03])

    def test_init_with_series(self, normal_returns: pl.Series):
        """Test initialization with Series."""
        validator = ReturnsValidator(normal_returns)
        assert len(validator.returns) == 100

    def test_init_with_dataframe(self):
        """Test initialization with DataFrame and column name."""
        df = pl.DataFrame({"returns": [0.01, 0.02, -0.01]})
        validator = ReturnsValidator(df, column="returns")
        assert len(validator.returns) == 3

    def test_init_with_dataframe_missing_column(self):
        """Test initialization with DataFrame but no column name."""
        df = pl.DataFrame({"returns": [0.01, 0.02]})

        with pytest.raises(ValueError, match="column required"):
            ReturnsValidator(df)

    def test_check_numeric_success(self, normal_returns: pl.Series):
        """Test check_numeric with numeric series."""
        validator = ReturnsValidator(normal_returns)
        result = validator.check_numeric()
        assert result is validator

    def test_check_numeric_failure(self):
        """Test check_numeric with non-numeric series."""
        returns = pl.Series("returns", ["a", "b", "c"])
        validator = ReturnsValidator(returns)

        with pytest.raises(ValidationError, match="must be numeric"):
            validator.check_numeric()

    def test_check_bounds_within_bounds(self, normal_returns: pl.Series):
        """Test check_bounds when values are within bounds."""
        validator = ReturnsValidator(normal_returns)
        result = validator.check_bounds(lower=-0.1, upper=0.1)
        assert result is validator

    def test_check_bounds_below_lower(self):
        """Test check_bounds when values below lower bound."""
        returns = pl.Series("returns", [0.01, 0.02, -0.5, 0.01])
        validator = ReturnsValidator(returns)

        with pytest.raises(ValidationError) as exc_info:
            validator.check_bounds(lower=-0.1)

        assert "below lower bound" in str(exc_info.value)
        assert exc_info.value.context["lower_bound"] == -0.1

    def test_check_bounds_above_upper(self):
        """Test check_bounds when values above upper bound."""
        returns = pl.Series("returns", [0.01, 0.02, 0.5, 0.01])
        validator = ReturnsValidator(returns)

        with pytest.raises(ValidationError) as exc_info:
            validator.check_bounds(upper=0.1)

        assert "above upper bound" in str(exc_info.value)
        assert exc_info.value.context["upper_bound"] == 0.1

    def test_check_bounds_only_lower(self, normal_returns: pl.Series):
        """Test check_bounds with only lower bound."""
        validator = ReturnsValidator(normal_returns)
        validator.check_bounds(lower=-0.1, upper=None)

    def test_check_bounds_only_upper(self, normal_returns: pl.Series):
        """Test check_bounds with only upper bound."""
        validator = ReturnsValidator(normal_returns)
        validator.check_bounds(lower=None, upper=0.1)

    def test_check_bounds_empty_after_dropna(self):
        """Test check_bounds with all null values."""
        # Must cast to Float64 to be numeric
        returns = pl.Series("returns", [None, None, None], dtype=pl.Float64)
        validator = ReturnsValidator(returns)

        # Should return early without error (after drop_nulls leaves empty)
        validator.check_bounds(lower=-0.1, upper=0.1)

    def test_check_finite_success(self, normal_returns: pl.Series):
        """Test check_finite with finite values."""
        validator = ReturnsValidator(normal_returns)
        result = validator.check_finite()
        assert result is validator

    def test_check_finite_with_inf(self):
        """Test check_finite with infinite values."""
        returns = pl.Series("returns", [0.01, float("inf"), -0.01])
        validator = ReturnsValidator(returns)

        with pytest.raises(ValidationError, match="infinite"):
            validator.check_finite()

    def test_check_finite_with_neg_inf(self):
        """Test check_finite with negative infinite values."""
        returns = pl.Series("returns", [0.01, float("-inf"), -0.01])
        validator = ReturnsValidator(returns)

        with pytest.raises(ValidationError, match="infinite"):
            validator.check_finite()

    def test_check_nulls_no_nulls(self, normal_returns: pl.Series):
        """Test check_nulls when no nulls exist."""
        validator = ReturnsValidator(normal_returns)
        result = validator.check_nulls(allow_nulls=False)
        assert result is validator

    def test_check_nulls_with_nulls_not_allowed(self, returns_with_nulls: pl.Series):
        """Test check_nulls when nulls exist and not allowed."""
        validator = ReturnsValidator(returns_with_nulls)

        with pytest.raises(ValidationError) as exc_info:
            validator.check_nulls(allow_nulls=False)

        assert "null values" in str(exc_info.value)
        assert exc_info.value.context["null_count"] == 2

    def test_check_nulls_with_nulls_allowed(self, returns_with_nulls: pl.Series):
        """Test check_nulls when nulls exist and allowed."""
        validator = ReturnsValidator(returns_with_nulls)
        result = validator.check_nulls(allow_nulls=True)
        assert result is validator

    def test_check_distribution_normal(self, normal_returns: pl.Series):
        """Test check_distribution with normal distribution."""
        validator = ReturnsValidator(normal_returns)
        result = validator.check_distribution(max_abs_skew=2.0, max_abs_kurtosis=10.0)
        assert result is validator

    def test_check_distribution_extreme_skew(self):
        """Test check_distribution with extreme skewness."""
        # Create highly skewed data
        np.random.seed(42)
        skewed_data = np.exp(np.random.normal(0, 1, 100))  # Log-normal is skewed
        returns = pl.Series("returns", skewed_data)
        validator = ReturnsValidator(returns)

        with pytest.raises(ValidationError, match="skewness"):
            validator.check_distribution(max_abs_skew=0.5)

    def test_check_distribution_extreme_kurtosis(self):
        """Test check_distribution with extreme kurtosis."""
        # Create data with extreme kurtosis using a mixture distribution
        # Most values near 0, with a few extreme outliers
        np.random.seed(42)
        normal_data = np.random.normal(0, 0.01, 90)  # Most data near 0
        outliers = np.array([10.0, -10.0, 15.0, -15.0, 20.0, -20.0, 25.0, -25.0, 30.0, -30.0])
        heavy_tail_data = np.concatenate([normal_data, outliers])
        np.random.shuffle(heavy_tail_data)
        returns = pl.Series("returns", heavy_tail_data)
        validator = ReturnsValidator(returns)

        with pytest.raises(ValidationError, match="kurtosis"):
            validator.check_distribution(max_abs_kurtosis=5.0)

    def test_check_distribution_insufficient_data(self):
        """Test check_distribution with insufficient data."""
        returns = pl.Series("returns", [0.01, 0.02, 0.03])  # Only 3 values
        validator = ReturnsValidator(returns)

        # Should return early without error (need >= 30 samples)
        result = validator.check_distribution(max_abs_skew=0.1, max_abs_kurtosis=0.1)
        assert result is validator

    def test_check_distribution_zero_std(self):
        """Test check_distribution with zero standard deviation."""
        returns = pl.Series("returns", [0.01] * 50)  # All same value
        validator = ReturnsValidator(returns)

        # Should not raise - zero std case is handled
        validator.check_distribution(max_abs_skew=0.1, max_abs_kurtosis=0.1)

    def test_chaining(self, normal_returns: pl.Series):
        """Test method chaining."""
        validator = ReturnsValidator(normal_returns)

        result = (
            validator.check_numeric()
            .check_finite()
            .check_nulls(allow_nulls=False)
            .check_bounds(lower=-0.1, upper=0.1)
            .check_distribution(max_abs_skew=2.0)
        )

        assert result is validator


class TestValidateReturns:
    """Tests for validate_returns function."""

    @pytest.fixture
    def valid_returns(self) -> pl.Series:
        """Create valid returns series."""
        return pl.Series("returns", [0.01, -0.02, 0.03, -0.01, 0.02])

    def test_valid_returns(self, valid_returns: pl.Series):
        """Test with valid returns."""
        # Should not raise
        validate_returns(valid_returns)

    def test_with_bounds(self, valid_returns: pl.Series):
        """Test with bounds parameter."""
        validate_returns(valid_returns, bounds=(-0.1, 0.1))

        with pytest.raises(ValidationError):
            validate_returns(valid_returns, bounds=(-0.01, 0.01))

    def test_nulls_not_allowed(self):
        """Test with nulls when not allowed."""
        returns = pl.Series("returns", [0.01, None, -0.02])

        with pytest.raises(ValidationError):
            validate_returns(returns, allow_nulls=False)

    def test_nulls_allowed(self):
        """Test with nulls when allowed."""
        returns = pl.Series("returns", [0.01, None, -0.02])

        # Should not raise
        validate_returns(returns, allow_nulls=True, check_finite=False)

    def test_check_finite_enabled(self):
        """Test with infinite value check enabled."""
        returns = pl.Series("returns", [0.01, float("inf"), -0.02])

        with pytest.raises(ValidationError):
            validate_returns(returns, check_finite=True)

    def test_check_finite_disabled(self):
        """Test with infinite value check disabled."""
        returns = pl.Series("returns", [0.01, float("inf"), -0.02])

        # Should not raise
        validate_returns(returns, check_finite=False)

    def test_with_dataframe(self):
        """Test with DataFrame input."""
        df = pl.DataFrame({"returns": [0.01, -0.02, 0.03]})

        # Should not raise
        validate_returns(df, column="returns")

    def test_non_numeric(self):
        """Test with non-numeric returns."""
        returns = pl.Series("returns", ["a", "b", "c"])

        with pytest.raises(ValidationError, match="numeric"):
            validate_returns(returns)


class TestValidateBounds:
    """Tests for validate_bounds function."""

    def test_within_bounds(self):
        """Test returns within bounds."""
        returns = pl.Series("returns", [0.01, -0.02, 0.03])

        # Should not raise
        validate_bounds(returns, lower=-0.1, upper=0.1)

    def test_below_lower_bound(self):
        """Test returns below lower bound."""
        returns = pl.Series("returns", [0.01, -0.2, 0.03])

        with pytest.raises(ValidationError):
            validate_bounds(returns, lower=-0.1)

    def test_above_upper_bound(self):
        """Test returns above upper bound."""
        returns = pl.Series("returns", [0.01, 0.2, 0.03])

        with pytest.raises(ValidationError):
            validate_bounds(returns, upper=0.1)

    def test_with_dataframe(self):
        """Test with DataFrame input."""
        df = pl.DataFrame({"returns": [0.01, -0.02, 0.03]})

        validate_bounds(df, column="returns", lower=-0.1, upper=0.1)

    def test_no_bounds(self):
        """Test with no bounds specified."""
        returns = pl.Series("returns", [100.0, -200.0, 300.0])

        # Should not raise when no bounds specified
        validate_bounds(returns)


class TestEdgeCases:
    """Edge case tests."""

    def test_single_value_returns(self):
        """Test with single value."""
        returns = pl.Series("returns", [0.01])

        validate_returns(returns)
        validate_bounds(returns, lower=-0.1, upper=0.1)

    def test_all_zeros(self):
        """Test with all zero returns."""
        returns = pl.Series("returns", [0.0, 0.0, 0.0])

        validate_returns(returns, bounds=(-0.1, 0.1))

    def test_large_values(self):
        """Test with large return values."""
        returns = pl.Series("returns", [1e10, -1e10, 1e10])

        # Should not raise without bounds
        validate_returns(returns)

        # Should raise with tight bounds
        with pytest.raises(ValidationError):
            validate_returns(returns, bounds=(-1.0, 1.0))

    def test_small_values(self):
        """Test with very small return values."""
        returns = pl.Series("returns", [1e-10, -1e-10, 1e-10])

        validate_returns(returns)

    def test_mixed_types_in_dataframe(self):
        """Test returns extraction from DataFrame with mixed types."""
        df = pl.DataFrame(
            {
                "date": ["2020-01-01", "2020-01-02", "2020-01-03"],
                "returns": [0.01, -0.02, 0.03],
                "volume": [1000, 2000, 3000],
            }
        )

        validate_returns(df, column="returns")

    def test_negative_skew(self):
        """Test distribution check with negative skewness."""
        # Create negatively skewed data
        np.random.seed(42)
        neg_skewed = -np.exp(np.random.normal(0, 1, 100))
        returns = pl.Series("returns", neg_skewed)
        validator = ReturnsValidator(returns)

        with pytest.raises(ValidationError, match="skewness"):
            validator.check_distribution(max_abs_skew=0.5)

    def test_bounds_count_out_of_bounds(self):
        """Test that count_out_of_bounds is reported correctly."""
        returns = pl.Series("returns", [-0.5, -0.4, 0.01, 0.02])
        validator = ReturnsValidator(returns)

        with pytest.raises(ValidationError) as exc_info:
            validator.check_bounds(lower=-0.1)

        # Two values are below -0.1
        assert exc_info.value.context["count_out_of_bounds"] == 2
