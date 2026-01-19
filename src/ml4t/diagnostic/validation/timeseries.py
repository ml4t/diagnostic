"""Time series validation utilities."""

from __future__ import annotations

from typing import SupportsFloat, SupportsInt, cast

import polars as pl

from ml4t.diagnostic.validation.dataframe import ValidationError


class TimeSeriesValidator:
    """Validator for time series DataFrames.

    Examples:
        >>> validator = TimeSeriesValidator(df, index_col="date")
        >>> validator.check_sorted()
        >>> validator.check_duplicates()
        >>> validator.check_frequency()
    """

    def __init__(self, df: pl.DataFrame, index_col: str = "date"):
        """Initialize validator.

        Args:
            df: DataFrame to validate
            index_col: Name of index/date column
        """
        self.df = df
        self.index_col = index_col

    def check_index_exists(self) -> TimeSeriesValidator:
        """Check that index column exists.

        Returns:
            Self for chaining

        Raises:
            ValidationError: If index column missing
        """
        if self.index_col not in self.df.columns:
            raise ValidationError(
                f"Index column '{self.index_col}' not found",
                context={"available_columns": self.df.columns},
            )

        return self

    def check_index_type(self) -> TimeSeriesValidator:
        """Check that index is datetime/date type.

        Returns:
            Self for chaining

        Raises:
            ValidationError: If index not datetime/date
        """
        self.check_index_exists()

        dtype = self.df[self.index_col].dtype

        if not (dtype.is_temporal() or str(dtype) in ["Date", "Datetime", "Time"]):
            raise ValidationError(
                "Index column must be temporal type",
                context={
                    "column": self.index_col,
                    "actual_type": str(dtype),
                    "expected": "Date, Datetime, or Time",
                },
            )

        return self

    def check_sorted(self, ascending: bool = True) -> TimeSeriesValidator:
        """Check that time series is sorted.

        Args:
            ascending: Whether series should be ascending

        Returns:
            Self for chaining

        Raises:
            ValidationError: If not sorted
        """
        self.check_index_exists()

        index = self.df[self.index_col]

        # Check if sorted using is_sorted() method
        is_sorted = index.is_sorted() if ascending else index.is_sorted(descending=True)

        if not is_sorted:
            direction = "ascending" if ascending else "descending"
            raise ValidationError(
                f"Time series not sorted in {direction} order",
                context={"index_column": self.index_col},
            )

        return self

    def check_duplicates(self) -> TimeSeriesValidator:
        """Check for duplicate timestamps.

        Returns:
            Self for chaining

        Raises:
            ValidationError: If duplicates found
        """
        self.check_index_exists()

        duplicates = self.df[self.index_col].is_duplicated().sum()

        if duplicates > 0:
            # Get some example duplicates
            dup_values = (
                self.df.filter(pl.col(self.index_col).is_duplicated())
                .select(self.index_col)
                .unique()
                .head(5)
                .to_series()
                .to_list()
            )

            raise ValidationError(
                f"Found {duplicates} duplicate timestamps",
                context={
                    "index_column": self.index_col,
                    "duplicate_count": duplicates,
                    "examples": dup_values,
                },
            )

        return self

    def check_gaps(self, max_gap_days: int | None = None) -> TimeSeriesValidator:
        """Check for large gaps in time series.

        Args:
            max_gap_days: Maximum allowed gap in days (None = no check)

        Returns:
            Self for chaining

        Raises:
            ValidationError: If gaps exceed threshold
        """
        if max_gap_days is None:
            return self

        self.check_index_exists()

        # Calculate gaps
        gaps = self.df[self.index_col].diff().drop_nulls()

        if len(gaps) == 0:
            return self

        # Check if any gap exceeds threshold
        max_gap_raw = gaps.max()

        # Convert to days if datetime (cast to handle Polars scalar types)
        from datetime import timedelta

        if isinstance(max_gap_raw, timedelta):
            max_gap_days_actual = max_gap_raw.days
        else:
            # Assume already in days (Polars scalar type)
            max_gap_days_actual = int(cast(SupportsInt, max_gap_raw))

        if max_gap_days_actual > max_gap_days:
            raise ValidationError(
                f"Time series has gap of {max_gap_days_actual} days",
                context={
                    "max_allowed": max_gap_days,
                    "max_gap": max_gap_days_actual,
                },
            )

        return self


def validate_timeseries(
    df: pl.DataFrame,
    index_col: str = "date",
    require_sorted: bool = True,
    check_duplicates: bool = True,
    max_gap_days: int | None = None,
) -> None:
    """Validate time series DataFrame.

    Args:
        df: DataFrame to validate
        index_col: Name of index/date column
        require_sorted: Whether series must be sorted
        check_duplicates: Whether to check for duplicate timestamps
        max_gap_days: Maximum allowed gap in days (None = no check)

    Raises:
        ValidationError: If validation fails

    Examples:
        >>> validate_timeseries(
        ...     df,
        ...     index_col="date",
        ...     require_sorted=True,
        ...     check_duplicates=True,
        ...     max_gap_days=7
        ... )
    """
    validator = TimeSeriesValidator(df, index_col)

    validator.check_index_exists().check_index_type()

    if require_sorted:
        validator.check_sorted()

    if check_duplicates:
        validator.check_duplicates()

    if max_gap_days is not None:
        validator.check_gaps(max_gap_days)


def validate_index(df: pl.DataFrame, index_col: str = "date") -> None:
    """Validate time series index column.

    Args:
        df: DataFrame to validate
        index_col: Name of index column

    Raises:
        ValidationError: If index invalid
    """
    validator = TimeSeriesValidator(df, index_col)
    validator.check_index_exists().check_index_type()


def validate_frequency(
    df: pl.DataFrame,
    index_col: str = "date",
    expected_freq: str | None = None,
) -> None:
    """Validate time series frequency.

    Args:
        df: DataFrame to validate
        index_col: Name of index column
        expected_freq: Expected frequency ("daily", "weekly", "monthly")

    Raises:
        ValidationError: If frequency doesn't match

    Note:
        Basic implementation checks consistent spacing.
        Full frequency detection would require more sophisticated logic.
    """
    validator = TimeSeriesValidator(df, index_col)
    validator.check_index_exists().check_sorted()

    if expected_freq is not None:
        # Basic frequency validation - check consistent spacing
        gaps = df[index_col].diff().drop_nulls()

        if len(gaps) == 0:
            return

        # Check if gaps are consistent (within tolerance)
        # Convert Duration to microseconds for numeric comparison
        from datetime import timedelta

        gaps_us = gaps.dt.total_microseconds()
        median_gap_us = gaps_us.median()
        max_deviation_us = (gaps_us - median_gap_us).abs().max()

        # Handle None cases (shouldn't happen with valid data)
        if median_gap_us is None or max_deviation_us is None:
            return

        # Cast to float for arithmetic
        median_gap = float(cast(SupportsFloat, median_gap_us))
        max_deviation = float(cast(SupportsFloat, max_deviation_us))

        # Allow 20% deviation
        tolerance = median_gap * 0.2

        if max_deviation > tolerance:
            # Convert back to timedelta for human-readable output
            median_td = timedelta(microseconds=median_gap)
            max_dev_td = timedelta(microseconds=max_deviation)
            raise ValidationError(
                f"Inconsistent {expected_freq} frequency detected",
                context={
                    "expected": expected_freq,
                    "median_gap": str(median_td),
                    "max_deviation": str(max_dev_td),
                },
            )
