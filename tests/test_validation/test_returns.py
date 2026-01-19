"""Tests for Returns validation utilities."""

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
    def sample_returns(self):
        """Create sample returns series for testing."""
        return pl.Series("returns", [0.01, -0.02, 0.015, -0.005, 0.008])

    @pytest.fixture
    def returns_df(self):
        """Create sample returns DataFrame for testing."""
        return pl.DataFrame(
            {
                "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
                "returns": [0.01, -0.02, 0.015],
            }
        )

    @pytest.fixture
    def returns_with_nulls(self):
        """Create returns series with null values."""
        return pl.Series("returns", [0.01, None, 0.015, None, 0.008])

    @pytest.fixture
    def returns_with_inf(self):
        """Create returns series with infinite values."""
        return pl.Series("returns", [0.01, float("inf"), 0.015, float("-inf"), 0.008])

    def test_init_with_series(self, sample_returns):
        """Test initialization with Series."""
        validator = ReturnsValidator(sample_returns)
        assert len(validator.returns) == 5

    def test_init_with_dataframe(self, returns_df):
        """Test initialization with DataFrame."""
        validator = ReturnsValidator(returns_df, column="returns")
        assert len(validator.returns) == 3

    def test_init_dataframe_without_column_raises(self, returns_df):
        """Test that DataFrame without column parameter raises."""
        with pytest.raises(ValueError, match="column required"):
            ReturnsValidator(returns_df)

    def test_check_numeric_success(self, sample_returns):
        """Test check_numeric with numeric returns."""
        validator = ReturnsValidator(sample_returns)
        result = validator.check_numeric()
        assert result is validator  # Chaining works

    def test_check_numeric_failure(self):
        """Test check_numeric with non-numeric returns."""
        string_returns = pl.Series("returns", ["a", "b", "c"])
        validator = ReturnsValidator(string_returns)

        with pytest.raises(ValidationError) as exc_info:
            validator.check_numeric()

        assert "numeric" in str(exc_info.value).lower()

    def test_check_bounds_success(self, sample_returns):
        """Test check_bounds with returns within bounds."""
        validator = ReturnsValidator(sample_returns)
        result = validator.check_bounds(lower=-0.05, upper=0.05)
        assert result is validator

    def test_check_bounds_lower_violation(self):
        """Test check_bounds with returns below lower bound."""
        returns = pl.Series("returns", [0.01, -0.10, 0.015])
        validator = ReturnsValidator(returns)

        with pytest.raises(ValidationError) as exc_info:
            validator.check_bounds(lower=-0.05)

        assert "below lower bound" in str(exc_info.value).lower()
        assert exc_info.value.context["lower_bound"] == -0.05

    def test_check_bounds_upper_violation(self):
        """Test check_bounds with returns above upper bound."""
        returns = pl.Series("returns", [0.01, 0.20, 0.015])
        validator = ReturnsValidator(returns)

        with pytest.raises(ValidationError) as exc_info:
            validator.check_bounds(upper=0.10)

        assert "above upper bound" in str(exc_info.value).lower()
        assert exc_info.value.context["upper_bound"] == 0.10

    def test_check_bounds_empty_after_null_drop(self):
        """Test check_bounds with all nulls."""
        # Create a float series with all nulls (need to cast to ensure numeric type)
        returns = pl.Series("returns", [None, None, None], dtype=pl.Float64)
        validator = ReturnsValidator(returns)
        # Should not raise - no non-null values to check
        result = validator.check_bounds(lower=-1.0, upper=1.0)
        assert result is validator

    def test_check_finite_success(self, sample_returns):
        """Test check_finite with finite returns."""
        validator = ReturnsValidator(sample_returns)
        result = validator.check_finite()
        assert result is validator

    def test_check_finite_with_inf(self, returns_with_inf):
        """Test check_finite with infinite values."""
        validator = ReturnsValidator(returns_with_inf)

        with pytest.raises(ValidationError) as exc_info:
            validator.check_finite()

        assert "infinite" in str(exc_info.value).lower()
        assert exc_info.value.context["infinite_count"] == 2

    def test_check_nulls_success_no_nulls(self, sample_returns):
        """Test check_nulls with no null values."""
        validator = ReturnsValidator(sample_returns)
        result = validator.check_nulls(allow_nulls=False)
        assert result is validator

    def test_check_nulls_with_nulls_not_allowed(self, returns_with_nulls):
        """Test check_nulls when nulls exist but not allowed."""
        validator = ReturnsValidator(returns_with_nulls)

        with pytest.raises(ValidationError) as exc_info:
            validator.check_nulls(allow_nulls=False)

        assert "null" in str(exc_info.value).lower()
        assert exc_info.value.context["null_count"] == 2

    def test_check_nulls_with_nulls_allowed(self, returns_with_nulls):
        """Test check_nulls when nulls are allowed."""
        validator = ReturnsValidator(returns_with_nulls)
        result = validator.check_nulls(allow_nulls=True)
        assert result is validator

    def test_check_distribution_success(self):
        """Test check_distribution with normal-ish returns."""
        np.random.seed(42)
        returns = pl.Series("returns", np.random.randn(100) * 0.02)
        validator = ReturnsValidator(returns)

        result = validator.check_distribution(max_abs_skew=2.0, max_abs_kurtosis=10.0)
        assert result is validator

    def test_check_distribution_insufficient_data(self):
        """Test check_distribution with insufficient data."""
        returns = pl.Series("returns", [0.01, -0.02, 0.015])  # Less than 30
        validator = ReturnsValidator(returns)

        # Should not raise - insufficient data for check
        result = validator.check_distribution(max_abs_skew=0.1)
        assert result is validator

    def test_check_distribution_extreme_skew(self):
        """Test check_distribution with extreme skewness."""
        # Create highly skewed returns
        returns = pl.Series("returns", [0.01] * 50 + [1.0] * 5)
        validator = ReturnsValidator(returns)

        with pytest.raises(ValidationError) as exc_info:
            validator.check_distribution(max_abs_skew=1.0)

        assert "skewness" in str(exc_info.value).lower()

    def test_check_distribution_extreme_kurtosis(self):
        """Test check_distribution with extreme kurtosis."""
        # Create high kurtosis returns (many small, few large)
        np.random.seed(42)
        base = np.random.randn(100) * 0.01
        base[0] = 1.0  # Extreme outlier
        base[1] = -1.0  # Extreme outlier
        returns = pl.Series("returns", base)
        validator = ReturnsValidator(returns)

        with pytest.raises(ValidationError) as exc_info:
            validator.check_distribution(max_abs_kurtosis=5.0)

        assert "kurtosis" in str(exc_info.value).lower()

    def test_chaining(self, sample_returns):
        """Test method chaining."""
        validator = ReturnsValidator(sample_returns)
        result = (
            validator.check_numeric()
            .check_nulls(allow_nulls=False)
            .check_finite()
            .check_bounds(lower=-0.05, upper=0.05)
        )
        assert result is validator


class TestValidateReturns:
    """Tests for validate_returns function."""

    @pytest.fixture
    def sample_returns(self):
        """Create sample returns series."""
        return pl.Series("returns", [0.01, -0.02, 0.015, -0.005, 0.008])

    def test_basic_validation(self, sample_returns):
        """Test basic validation passes."""
        validate_returns(sample_returns)  # Should not raise

    def test_with_bounds(self, sample_returns):
        """Test validation with bounds."""
        validate_returns(sample_returns, bounds=(-0.05, 0.05))

    def test_with_bounds_violation(self):
        """Test validation with bounds violation."""
        returns = pl.Series("returns", [0.01, 0.20])
        with pytest.raises(ValidationError):
            validate_returns(returns, bounds=(-0.1, 0.1))

    def test_with_nulls_not_allowed(self):
        """Test validation with nulls not allowed."""
        returns = pl.Series("returns", [0.01, None, 0.015])
        with pytest.raises(ValidationError):
            validate_returns(returns, allow_nulls=False)

    def test_with_nulls_allowed(self):
        """Test validation with nulls allowed."""
        returns = pl.Series("returns", [0.01, None, 0.015])
        validate_returns(returns, allow_nulls=True)  # Should not raise

    def test_with_dataframe(self):
        """Test validation with DataFrame input."""
        df = pl.DataFrame({"returns": [0.01, -0.02, 0.015]})
        validate_returns(df, column="returns")

    def test_check_finite(self):
        """Test finite check."""
        returns = pl.Series("returns", [0.01, float("inf"), 0.015])
        with pytest.raises(ValidationError):
            validate_returns(returns, check_finite=True)


class TestValidateBounds:
    """Tests for validate_bounds function."""

    def test_within_bounds(self):
        """Test returns within bounds."""
        returns = pl.Series("returns", [0.01, -0.02, 0.015])
        validate_bounds(returns, lower=-0.05, upper=0.05)

    def test_lower_bound_violation(self):
        """Test lower bound violation."""
        returns = pl.Series("returns", [0.01, -0.10, 0.015])
        with pytest.raises(ValidationError):
            validate_bounds(returns, lower=-0.05)

    def test_upper_bound_violation(self):
        """Test upper bound violation."""
        returns = pl.Series("returns", [0.01, 0.15, 0.015])
        with pytest.raises(ValidationError):
            validate_bounds(returns, upper=0.10)

    def test_with_dataframe(self):
        """Test with DataFrame input."""
        df = pl.DataFrame({"returns": [0.01, -0.02, 0.015]})
        validate_bounds(df, column="returns", lower=-0.05, upper=0.05)
