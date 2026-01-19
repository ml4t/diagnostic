"""Returns validation utilities."""

from __future__ import annotations

from typing import SupportsFloat, cast

import polars as pl

from ml4t.diagnostic.validation.dataframe import ValidationError


class ReturnsValidator:
    """Validator for returns series.

    Examples:
        >>> validator = ReturnsValidator(returns)
        >>> validator.check_numeric()
        >>> validator.check_bounds(-0.5, 0.5)
        >>> validator.check_distribution()
    """

    def __init__(self, returns: pl.Series | pl.DataFrame, column: str | None = None):
        """Initialize validator.

        Args:
            returns: Returns series or DataFrame
            column: Column name if DataFrame provided
        """
        if isinstance(returns, pl.DataFrame):
            if column is None:
                raise ValueError("column required when passing DataFrame")
            self.returns = returns[column]
        else:
            self.returns = returns

    def check_numeric(self) -> ReturnsValidator:
        """Check that returns are numeric.

        Returns:
            Self for chaining

        Raises:
            ValidationError: If not numeric
        """
        if not self.returns.dtype.is_numeric():
            raise ValidationError(
                "Returns must be numeric",
                context={"dtype": str(self.returns.dtype)},
            )

        return self

    def check_bounds(
        self, lower: float | None = None, upper: float | None = None
    ) -> ReturnsValidator:
        """Check returns fall within bounds.

        Args:
            lower: Lower bound (inclusive)
            upper: Upper bound (inclusive)

        Returns:
            Self for chaining

        Raises:
            ValidationError: If returns out of bounds
        """
        self.check_numeric()

        # Drop nulls for bounds checking
        clean_returns = self.returns.drop_nulls()

        if len(clean_returns) == 0:
            return self

        if lower is not None:
            min_val = float(cast(SupportsFloat, clean_returns.min()))
            if min_val < lower:
                out_of_bounds = (clean_returns < lower).sum()
                raise ValidationError(
                    f"Returns below lower bound: {min_val:.4f} < {lower}",
                    context={
                        "lower_bound": lower,
                        "min_value": min_val,
                        "count_out_of_bounds": out_of_bounds,
                    },
                )

        if upper is not None:
            max_val = float(cast(SupportsFloat, clean_returns.max()))
            if max_val > upper:
                out_of_bounds = (clean_returns > upper).sum()
                raise ValidationError(
                    f"Returns above upper bound: {max_val:.4f} > {upper}",
                    context={
                        "upper_bound": upper,
                        "max_value": max_val,
                        "count_out_of_bounds": out_of_bounds,
                    },
                )

        return self

    def check_finite(self) -> ReturnsValidator:
        """Check for infinite values.

        Returns:
            Self for chaining

        Raises:
            ValidationError: If infinite values found
        """
        self.check_numeric()

        # Check for inf/-inf
        is_inf = self.returns.is_infinite()
        inf_count = is_inf.sum()

        if inf_count > 0:
            raise ValidationError(
                f"Found {inf_count} infinite values",
                context={"infinite_count": inf_count},
            )

        return self

    def check_nulls(self, allow_nulls: bool = False) -> ReturnsValidator:
        """Check for null values.

        Args:
            allow_nulls: Whether nulls are allowed

        Returns:
            Self for chaining

        Raises:
            ValidationError: If nulls found when not allowed
        """
        if not allow_nulls:
            null_count = self.returns.null_count()

            if null_count > 0:
                raise ValidationError(
                    f"Found {null_count} null values",
                    context={
                        "null_count": null_count,
                        "total_count": len(self.returns),
                    },
                )

        return self

    def check_distribution(
        self,
        max_abs_skew: float | None = None,
        max_abs_kurtosis: float | None = None,
    ) -> ReturnsValidator:
        """Check distribution characteristics.

        Args:
            max_abs_skew: Maximum absolute skewness (None = no check)
            max_abs_kurtosis: Maximum absolute excess kurtosis (None = no check)

        Returns:
            Self for chaining

        Raises:
            ValidationError: If distribution extreme
        """
        self.check_numeric()

        clean_returns = self.returns.drop_nulls()

        if len(clean_returns) < 30:
            # Need sufficient data for distribution checks
            return self

        if max_abs_skew is not None:
            # Calculate skewness (simplified)
            mean = float(cast(SupportsFloat, clean_returns.mean()))
            std = float(cast(SupportsFloat, clean_returns.std()))

            if std > 0:
                skew = float(cast(SupportsFloat, ((clean_returns - mean) ** 3).mean())) / (std**3)

                if abs(skew) > max_abs_skew:
                    raise ValidationError(
                        f"Extreme skewness detected: {skew:.2f}",
                        context={
                            "skewness": skew,
                            "max_allowed": max_abs_skew,
                        },
                    )

        if max_abs_kurtosis is not None:
            # Calculate excess kurtosis (simplified)
            mean = float(cast(SupportsFloat, clean_returns.mean()))
            std = float(cast(SupportsFloat, clean_returns.std()))

            if std > 0:
                kurtosis = (
                    float(cast(SupportsFloat, ((clean_returns - mean) ** 4).mean())) / (std**4) - 3
                )

                if abs(kurtosis) > max_abs_kurtosis:
                    raise ValidationError(
                        f"Extreme kurtosis detected: {kurtosis:.2f}",
                        context={
                            "kurtosis": kurtosis,
                            "max_allowed": max_abs_kurtosis,
                        },
                    )

        return self


def validate_returns(
    returns: pl.Series | pl.DataFrame,
    column: str | None = None,
    bounds: tuple[float, float] | None = None,
    allow_nulls: bool = False,
    check_finite: bool = True,
) -> None:
    """Validate returns series.

    Args:
        returns: Returns series or DataFrame
        column: Column name if DataFrame
        bounds: (lower, upper) bounds for returns
        allow_nulls: Whether null values allowed
        check_finite: Whether to check for infinite values

    Raises:
        ValidationError: If validation fails

    Examples:
        >>> validate_returns(
        ...     returns,
        ...     bounds=(-0.5, 0.5),
        ...     allow_nulls=False,
        ...     check_finite=True
        ... )
    """
    validator = ReturnsValidator(returns, column)

    validator.check_numeric()

    if not allow_nulls:
        validator.check_nulls(allow_nulls=False)

    if check_finite:
        validator.check_finite()

    if bounds is not None:
        lower, upper = bounds
        validator.check_bounds(lower, upper)


def validate_bounds(
    returns: pl.Series | pl.DataFrame,
    column: str | None = None,
    lower: float | None = None,
    upper: float | None = None,
) -> None:
    """Validate returns fall within bounds.

    Args:
        returns: Returns series or DataFrame
        column: Column name if DataFrame
        lower: Lower bound (inclusive)
        upper: Upper bound (inclusive)

    Raises:
        ValidationError: If returns out of bounds

    Examples:
        >>> validate_bounds(returns, lower=-1.0, upper=1.0)
    """
    validator = ReturnsValidator(returns, column)
    validator.check_numeric().check_bounds(lower, upper)
