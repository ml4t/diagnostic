"""Turnover analysis result classes for signal analysis.

This module provides result classes for storing turnover analysis outputs including
quantile turnover rates, signal autocorrelation, and stability metrics.

References
----------
Lopez de Prado, M. (2018). "Advances in Financial Machine Learning"
"""

from __future__ import annotations

from typing import Any

import polars as pl
from pydantic import Field, model_validator

from ml4t.diagnostic.results.base import BaseResult


class TurnoverAnalysisResult(BaseResult):
    """Results from turnover analysis.

    Contains quantile turnover rates, signal autocorrelation,
    and stability metrics.

    Examples
    --------
    >>> result = turnover_result
    >>> print(result.summary())
    >>> df = result.get_dataframe("turnover")
    """

    analysis_type: str = Field(default="turnover_analysis", frozen=True)

    # ==========================================================================
    # Quantile Turnover
    # ==========================================================================

    quantile_turnover: dict[str, dict[str, float]] = Field(
        ...,
        description="Turnover rate by quantile and period: {period: {quantile: turnover}}",
    )

    mean_turnover: dict[str, float] = Field(
        ...,
        description="Mean turnover across all quantiles per period",
    )

    top_quantile_turnover: dict[str, float] = Field(
        ...,
        description="Turnover for top quantile (long positions)",
    )

    bottom_quantile_turnover: dict[str, float] = Field(
        ...,
        description="Turnover for bottom quantile (short positions)",
    )

    # ==========================================================================
    # Signal Autocorrelation
    # ==========================================================================

    autocorrelation: dict[str, list[float]] = Field(
        ...,
        description="Autocorrelation by lag: {period: [ac_lag1, ac_lag2, ...]}",
    )

    autocorrelation_lags: list[int] = Field(
        ...,
        description="Lag values used",
    )

    mean_autocorrelation: dict[str, float] = Field(
        ...,
        description="Mean autocorrelation (average across first 5 lags)",
    )

    # ==========================================================================
    # Stability Metrics
    # ==========================================================================

    half_life: dict[str, float | None] = Field(
        ...,
        description="Signal half-life in periods (time for AC to decay by 50%)",
    )

    # ==========================================================================
    # Validation
    # ==========================================================================

    @model_validator(mode="after")
    def _validate_keys(self) -> TurnoverAnalysisResult:
        """Validate that all period-keyed dicts share the same keys and list lengths match."""
        # Get reference period set from quantile_turnover
        period_set = set(self.quantile_turnover.keys())

        # Validate period-keyed dicts
        period_dicts: list[tuple[str, dict[str, Any]]] = [
            ("mean_turnover", self.mean_turnover),
            ("top_quantile_turnover", self.top_quantile_turnover),
            ("bottom_quantile_turnover", self.bottom_quantile_turnover),
            ("autocorrelation", self.autocorrelation),
            ("mean_autocorrelation", self.mean_autocorrelation),
            ("half_life", self.half_life),
        ]
        for name, d in period_dicts:
            if set(d.keys()) != period_set:
                raise ValueError(
                    f"Key mismatch in '{name}': expected {period_set}, got {set(d.keys())}"
                )

        # Validate autocorrelation list lengths match autocorrelation_lags
        n_lags = len(self.autocorrelation_lags)
        for period, ac_values in self.autocorrelation.items():
            if len(ac_values) != n_lags:
                raise ValueError(
                    f"Length mismatch in autocorrelation['{period}']: "
                    f"expected {n_lags} (len(autocorrelation_lags)), got {len(ac_values)}"
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
            - None or "turnover": Turnover by quantile
            - "autocorrelation": Autocorrelation by lag
            - "summary": Summary statistics

        Returns
        -------
        pl.DataFrame
            Requested DataFrame
        """
        if name is None or name == "turnover":
            periods = list(self.quantile_turnover.keys())
            if not periods:
                return pl.DataFrame()

            quantiles = list(self.quantile_turnover[periods[0]].keys())
            rows = []
            for period in periods:
                for q in quantiles:
                    rows.append(
                        {
                            "period": period,
                            "quantile": q,
                            "turnover": self.quantile_turnover[period][q],
                        }
                    )
            return pl.DataFrame(rows)

        if name == "autocorrelation":
            periods = list(self.autocorrelation.keys())
            rows = []
            for period in periods:
                for i, lag in enumerate(self.autocorrelation_lags):
                    rows.append(
                        {
                            "period": period,
                            "lag": lag,
                            "autocorrelation": self.autocorrelation[period][i],
                        }
                    )
            return pl.DataFrame(rows)

        if name == "summary":
            periods = list(self.mean_turnover.keys())
            return pl.DataFrame(
                {
                    "period": periods,
                    "mean_turnover": [self.mean_turnover[p] for p in periods],
                    "top_turnover": [self.top_quantile_turnover[p] for p in periods],
                    "bottom_turnover": [self.bottom_quantile_turnover[p] for p in periods],
                    "mean_autocorrelation": [self.mean_autocorrelation[p] for p in periods],
                    "half_life": [self.half_life[p] for p in periods],
                }
            )

        raise ValueError(
            f"Unknown DataFrame name: {name}. Available: 'turnover', 'autocorrelation', 'summary'"
        )

    def list_available_dataframes(self) -> list[str]:
        """List available DataFrame views."""
        return ["turnover", "autocorrelation", "summary"]

    def summary(self) -> str:
        """Get human-readable summary of turnover analysis results."""
        lines = ["=" * 60, "Turnover Analysis Summary", "=" * 60, ""]

        for period in self.mean_turnover:
            lines.append(f"Period: {period}")
            lines.append("-" * 40)
            lines.append(f"  Mean Turnover:        {self.mean_turnover[period]:>8.2%}")
            lines.append(f"  Top Quantile:         {self.top_quantile_turnover[period]:>8.2%}")
            lines.append(f"  Bottom Quantile:      {self.bottom_quantile_turnover[period]:>8.2%}")
            lines.append(f"  Mean Autocorrelation: {self.mean_autocorrelation[period]:>8.4f}")

            if self.half_life[period] is not None:
                lines.append(f"  Signal Half-Life:     {self.half_life[period]:>8.1f} periods")
            lines.append("")

        return "\n".join(lines)
