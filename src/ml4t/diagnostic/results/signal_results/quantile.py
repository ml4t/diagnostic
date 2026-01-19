"""Quantile analysis result classes for signal analysis.

This module provides result classes for storing quantile analysis outputs including
mean returns by quantile, spread statistics, and monotonicity tests.

References
----------
Lopez de Prado, M. (2018). "Advances in Financial Machine Learning"
"""

from __future__ import annotations

from typing import Any

import polars as pl
from pydantic import Field, model_validator

from ml4t.diagnostic.results.base import BaseResult
from ml4t.diagnostic.results.signal_results.validation import _normalize_period


class QuantileAnalysisResult(BaseResult):
    """Results from quantile analysis.

    Contains mean returns by quantile, spread analysis, and
    monotonicity test results.

    Examples
    --------
    >>> result = quantile_result
    >>> print(result.summary())
    >>> df = result.get_dataframe("mean_returns")
    """

    analysis_type: str = Field(default="quantile_analysis", frozen=True)

    # ==========================================================================
    # Quantile Configuration
    # ==========================================================================

    n_quantiles: int = Field(
        ...,
        description="Number of quantile bins",
    )

    quantile_labels: list[str] = Field(
        ...,
        description="Labels for each quantile (e.g., ['Q1', 'Q2', ..., 'Q5'])",
    )

    periods: list[str] = Field(
        ...,
        description="Forward return periods analyzed",
    )

    # ==========================================================================
    # Mean Returns by Quantile
    # ==========================================================================

    mean_returns: dict[str, dict[str, float]] = Field(
        ...,
        description="Mean returns: {period: {quantile: mean_return}}",
    )

    std_returns: dict[str, dict[str, float]] = Field(
        ...,
        description="Std deviation of returns: {period: {quantile: std}}",
    )

    count_by_quantile: dict[str, int] = Field(
        ...,
        description="Number of observations per quantile",
    )

    # ==========================================================================
    # Spread Analysis (Top - Bottom)
    # ==========================================================================

    spread_mean: dict[str, float] = Field(
        ...,
        description="Mean spread (top quantile - bottom quantile) per period",
    )

    spread_std: dict[str, float] = Field(
        ...,
        description="Std deviation of spread per period",
    )

    spread_t_stat: dict[str, float] = Field(
        ...,
        description="T-statistic for spread != 0",
    )

    spread_p_value: dict[str, float] = Field(
        ...,
        description="P-value for spread significance",
    )

    spread_ci_lower: dict[str, float] = Field(
        ...,
        description="Lower confidence interval for spread",
    )

    spread_ci_upper: dict[str, float] = Field(
        ...,
        description="Upper confidence interval for spread",
    )

    confidence_level: float = Field(
        default=0.95,
        description="Confidence level used for intervals",
    )

    # ==========================================================================
    # Monotonicity Test
    # ==========================================================================

    is_monotonic: dict[str, bool] = Field(
        ...,
        description="Whether returns are monotonic across quantiles per period",
    )

    monotonicity_direction: dict[str, str] = Field(
        ...,
        description="Direction of monotonicity: 'increasing', 'decreasing', or 'none'",
    )

    rank_correlation: dict[str, float] = Field(
        ...,
        description="Spearman correlation between quantile rank and mean return",
    )

    # ==========================================================================
    # Cumulative Returns (Optional)
    # ==========================================================================

    cumulative_returns: dict[str, dict[str, list[float]]] | None = Field(
        default=None,
        description="Cumulative returns by quantile: {period: {quantile: [values]}}",
    )

    cumulative_dates: list[str] | None = Field(
        default=None,
        description="Dates for cumulative returns",
    )

    # ==========================================================================
    # Validation
    # ==========================================================================

    @model_validator(mode="after")
    def _validate_keys(self) -> QuantileAnalysisResult:
        """Validate that all dicts use consistent period and quantile keys."""
        period_set = set(self.periods)
        quantile_set = set(self.quantile_labels)

        # Validate period-keyed dicts (flat dicts)
        period_dicts: list[tuple[str, dict[str, Any]]] = [
            ("spread_mean", self.spread_mean),
            ("spread_std", self.spread_std),
            ("spread_t_stat", self.spread_t_stat),
            ("spread_p_value", self.spread_p_value),
            ("spread_ci_lower", self.spread_ci_lower),
            ("spread_ci_upper", self.spread_ci_upper),
            ("is_monotonic", self.is_monotonic),
            ("monotonicity_direction", self.monotonicity_direction),
            ("rank_correlation", self.rank_correlation),
        ]
        for name, d in period_dicts:
            if set(d.keys()) != period_set:
                raise ValueError(
                    f"Key mismatch in '{name}': expected {period_set}, got {set(d.keys())}"
                )

        # Validate nested period-keyed dicts (mean_returns, std_returns)
        for name, d in [("mean_returns", self.mean_returns), ("std_returns", self.std_returns)]:
            if set(d.keys()) != period_set:
                raise ValueError(
                    f"Key mismatch in '{name}' (outer keys): expected {period_set}, got {set(d.keys())}"
                )
            for period, inner in d.items():
                if set(inner.keys()) != quantile_set:
                    raise ValueError(
                        f"Key mismatch in '{name}[{period}]': expected {quantile_set}, got {set(inner.keys())}"
                    )

        # Validate quantile-keyed dict
        if set(self.count_by_quantile.keys()) != quantile_set:
            raise ValueError(
                f"Key mismatch in 'count_by_quantile': expected {quantile_set}, got {set(self.count_by_quantile.keys())}"
            )

        # Validate n_quantiles consistency
        if self.n_quantiles != len(self.quantile_labels):
            raise ValueError(
                f"n_quantiles ({self.n_quantiles}) != len(quantile_labels) ({len(self.quantile_labels)})"
            )

        return self

    # ==========================================================================
    # Methods
    # ==========================================================================

    def get_dataframe(self, name: str | None = None) -> pl.DataFrame:
        """Get results as Polars DataFrame.

        Parameters
        ----------
        name : str | None
            DataFrame to retrieve:
            - None or "mean_returns": Mean returns by quantile x period
            - "spread": Spread statistics by period
            - "cumulative": Cumulative returns time series (if available)

        Returns
        -------
        pl.DataFrame
            Requested DataFrame
        """
        if name is None or name == "mean_returns":
            mean_return_rows: list[dict[str, Any]] = []
            for period in self.periods:
                for q in self.quantile_labels:
                    mean_return_rows.append(
                        {
                            "period": period,
                            "quantile": q,
                            "mean_return": self.mean_returns[period][q],
                            "std_return": self.std_returns[period][q],
                        }
                    )
            return pl.DataFrame(mean_return_rows)

        if name == "spread":
            return pl.DataFrame(
                {
                    "period": self.periods,
                    "spread_mean": [self.spread_mean[p] for p in self.periods],
                    "spread_std": [self.spread_std[p] for p in self.periods],
                    "spread_t_stat": [self.spread_t_stat[p] for p in self.periods],
                    "spread_p_value": [self.spread_p_value[p] for p in self.periods],
                    "spread_ci_lower": [self.spread_ci_lower[p] for p in self.periods],
                    "spread_ci_upper": [self.spread_ci_upper[p] for p in self.periods],
                    "is_monotonic": [self.is_monotonic[p] for p in self.periods],
                    "monotonicity_direction": [
                        self.monotonicity_direction[p] for p in self.periods
                    ],
                    "rank_correlation": [self.rank_correlation[p] for p in self.periods],
                }
            )

        if name == "cumulative":
            if self.cumulative_returns is None or self.cumulative_dates is None:
                raise ValueError("Cumulative returns not available")
            # Build wide DataFrame with dates and all quantile series
            rows: list[dict[str, Any]] = []
            for i, date in enumerate(self.cumulative_dates):
                row: dict[str, Any] = {"date": date}
                for period in self.periods:
                    for q in self.quantile_labels:
                        col_name = f"{period}_{q}"
                        row[col_name] = self.cumulative_returns[period][q][i]
                rows.append(row)
            return pl.DataFrame(rows)

        raise ValueError(
            f"Unknown DataFrame name: {name}. Available: 'mean_returns', 'spread', 'cumulative'"
        )

    def list_available_dataframes(self) -> list[str]:
        """List available DataFrame views."""
        dfs = ["mean_returns", "spread"]
        if self.cumulative_returns is not None:
            dfs.append("cumulative")
        return dfs

    def summary(self) -> str:
        """Get human-readable summary of quantile analysis results."""
        lines = ["=" * 60, "Quantile Analysis Summary", "=" * 60, ""]

        for period in self.periods:
            lines.append(f"Period: {period}")
            lines.append("-" * 40)
            lines.append("Quantile    Mean Return    Std")

            for q in self.quantile_labels:
                mean = self.mean_returns[period][q]
                std = self.std_returns[period][q]
                lines.append(f"  {q:<10} {mean:>10.4%}  {std:>10.4%}")

            lines.append("")
            lines.append(f"Spread (Top-Bottom): {self.spread_mean[period]:>10.4%}")
            lines.append(f"Spread t-stat:       {self.spread_t_stat[period]:>10.2f}")
            lines.append(f"Spread p-value:      {self.spread_p_value[period]:>10.4f}")
            lines.append(
                f"Monotonic:           {self.is_monotonic[period]} ({self.monotonicity_direction[period]})"
            )
            lines.append("")

        return "\n".join(lines)

    # =========================================================================
    # Convenience Accessor Methods
    # =========================================================================

    def get_quantile_returns(self, period: int | str) -> dict[str, float]:
        """Get mean returns for all quantiles at a specific period.

        Parameters
        ----------
        period : int | str
            Period as integer (21) or string ('21' or '21D').

        Returns
        -------
        dict[str, float]
            Dict mapping quantile label to mean return: {'Q1': 0.01, 'Q2': 0.02, ...}

        Examples
        --------
        >>> returns = quantile_result.get_quantile_returns(21)
        >>> for q, ret in returns.items():
        ...     print(f"{q}: {ret:.2%}")
        """
        key = _normalize_period(period)
        return self.mean_returns.get(key, {})

    def get_spread(self, period: int | str) -> tuple[float, float, float]:
        """Get spread statistics for a period.

        Parameters
        ----------
        period : int | str
            Period as integer or string.

        Returns
        -------
        tuple[float, float, float]
            Tuple of (spread_mean, spread_t_stat, spread_p_value).
            Returns (nan, nan, nan) if period not found.

        Examples
        --------
        >>> spread, t_stat, p_val = quantile_result.get_spread(21)
        >>> print(f"Spread: {spread:.2%} (t={t_stat:.2f}, p={p_val:.4f})")
        """
        key = _normalize_period(period)
        return (
            self.spread_mean.get(key, float("nan")),
            self.spread_t_stat.get(key, float("nan")),
            self.spread_p_value.get(key, float("nan")),
        )

    def get_top_quantile_return(self, period: int | str) -> float | None:
        """Get mean return for the top quantile (long side)."""
        key = _normalize_period(period)
        if key not in self.mean_returns:
            return None
        # Top quantile is the last one
        top_label = self.quantile_labels[-1]
        return self.mean_returns[key].get(top_label)

    def get_bottom_quantile_return(self, period: int | str) -> float | None:
        """Get mean return for the bottom quantile (short side)."""
        key = _normalize_period(period)
        if key not in self.mean_returns:
            return None
        # Bottom quantile is the first one
        bottom_label = self.quantile_labels[0]
        return self.mean_returns[key].get(bottom_label)

    def is_spread_significant(self, period: int | str, alpha: float = 0.05) -> bool:
        """Check if spread is statistically significant for a period.

        Parameters
        ----------
        period : int | str
            Period to check.
        alpha : float, default 0.05
            Significance level.

        Returns
        -------
        bool
            True if spread p-value < alpha.
        """
        key = _normalize_period(period)
        p_val = self.spread_p_value.get(key)
        if p_val is None:
            return False
        return p_val < alpha
