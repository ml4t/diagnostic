"""Profit factor analysis results for barrier outcomes.

This module provides the ProfitFactorResult class for storing profit factor
metrics (Sum(TP returns) / |Sum(SL returns)|) by signal quantile.
"""

from __future__ import annotations

import polars as pl
from pydantic import Field, model_validator

from ml4t.diagnostic.results.barrier_results.validation import _validate_quantile_dict_keys
from ml4t.diagnostic.results.base import BaseResult


class ProfitFactorResult(BaseResult):
    """Results from profit factor analysis by signal decile.

    Profit Factor = Sum(TP returns) / |Sum(SL returns)|

    A profit factor > 1 indicates net profitable trading in that decile.

    Examples
    --------
    >>> result = profit_factor_result
    >>> print(result.summary())
    >>> df = result.get_dataframe()
    """

    analysis_type: str = Field(default="barrier_profit_factor", frozen=True)

    # ==========================================================================
    # Configuration
    # ==========================================================================

    n_quantiles: int = Field(
        ...,
        description="Number of quantiles used",
    )

    quantile_labels: list[str] = Field(
        ...,
        description="Labels for each quantile (e.g., ['D1', 'D2', ..., 'D10'])",
    )

    # ==========================================================================
    # Profit Factor by Quantile
    # ==========================================================================

    profit_factor: dict[str, float] = Field(
        ...,
        description="Profit factor per quantile: Sum(TP returns) / |Sum(SL returns)|",
    )

    # ==========================================================================
    # Component Sums
    # ==========================================================================

    sum_tp_returns: dict[str, float] = Field(
        ...,
        description="Sum of returns from TP outcomes per quantile",
    )

    sum_sl_returns: dict[str, float] = Field(
        ...,
        description="Sum of returns from SL outcomes per quantile (negative values)",
    )

    sum_timeout_returns: dict[str, float] = Field(
        ...,
        description="Sum of returns from timeout outcomes per quantile",
    )

    sum_all_returns: dict[str, float] = Field(
        ...,
        description="Sum of all returns per quantile",
    )

    # ==========================================================================
    # Average Returns
    # ==========================================================================

    avg_tp_return: dict[str, float] = Field(
        ...,
        description="Average return per TP outcome per quantile",
    )

    avg_sl_return: dict[str, float] = Field(
        ...,
        description="Average return per SL outcome per quantile",
    )

    avg_return: dict[str, float] = Field(
        ...,
        description="Average return per quantile (all outcomes)",
    )

    # ==========================================================================
    # Counts
    # ==========================================================================

    count_tp: dict[str, int] = Field(
        ...,
        description="Number of TP outcomes per quantile",
    )

    count_sl: dict[str, int] = Field(
        ...,
        description="Number of SL outcomes per quantile",
    )

    count_total: dict[str, int] = Field(
        ...,
        description="Total count per quantile",
    )

    # ==========================================================================
    # Aggregates
    # ==========================================================================

    overall_profit_factor: float = Field(
        ...,
        description="Overall profit factor across all observations",
    )

    overall_sum_returns: float = Field(
        ...,
        description="Total sum of all returns",
    )

    overall_avg_return: float = Field(
        ...,
        description="Average return across all observations",
    )

    n_observations: int = Field(
        ...,
        description="Total number of observations analyzed",
    )

    # ==========================================================================
    # Monotonicity
    # ==========================================================================

    pf_monotonic: bool = Field(
        ...,
        description="Whether profit factor is monotonic across quantiles",
    )

    pf_direction: str = Field(
        ...,
        description="Direction of PF change: 'increasing', 'decreasing', or 'none'",
    )

    pf_spearman: float = Field(
        ...,
        description="Spearman correlation between quantile rank and profit factor",
    )

    # ==========================================================================
    # Validation
    # ==========================================================================

    @model_validator(mode="after")
    def _validate_quantile_keys(self) -> ProfitFactorResult:
        """Validate that all quantile-keyed dicts have consistent keys."""
        if self.n_quantiles != len(self.quantile_labels):
            raise ValueError(
                f"n_quantiles ({self.n_quantiles}) != len(quantile_labels) ({len(self.quantile_labels)})"
            )
        _validate_quantile_dict_keys(
            self.quantile_labels,
            [
                ("profit_factor", self.profit_factor),
                ("sum_tp_returns", self.sum_tp_returns),
                ("sum_sl_returns", self.sum_sl_returns),
                ("sum_timeout_returns", self.sum_timeout_returns),
                ("sum_all_returns", self.sum_all_returns),
                ("avg_tp_return", self.avg_tp_return),
                ("avg_sl_return", self.avg_sl_return),
                ("avg_return", self.avg_return),
                ("count_tp", self.count_tp),
                ("count_sl", self.count_sl),
                ("count_total", self.count_total),
            ],
        )
        return self

    def get_dataframe(self, name: str | None = None) -> pl.DataFrame:
        """Get results as Polars DataFrame.

        Parameters
        ----------
        name : str | None
            DataFrame to retrieve:
            - None or "profit_factor": Profit factor by quantile
            - "returns": Detailed returns breakdown by quantile
            - "summary": Single-row summary statistics

        Returns
        -------
        pl.DataFrame
            Requested DataFrame
        """
        if name is None or name == "profit_factor":
            return pl.DataFrame(
                {
                    "quantile": self.quantile_labels,
                    "profit_factor": [self.profit_factor[q] for q in self.quantile_labels],
                    "avg_return": [self.avg_return[q] for q in self.quantile_labels],
                    "sum_returns": [self.sum_all_returns[q] for q in self.quantile_labels],
                    "count_total": [self.count_total[q] for q in self.quantile_labels],
                }
            )

        if name == "returns":
            return pl.DataFrame(
                {
                    "quantile": self.quantile_labels,
                    "sum_tp_returns": [self.sum_tp_returns[q] for q in self.quantile_labels],
                    "sum_sl_returns": [self.sum_sl_returns[q] for q in self.quantile_labels],
                    "sum_timeout_returns": [
                        self.sum_timeout_returns[q] for q in self.quantile_labels
                    ],
                    "avg_tp_return": [self.avg_tp_return[q] for q in self.quantile_labels],
                    "avg_sl_return": [self.avg_sl_return[q] for q in self.quantile_labels],
                    "count_tp": [self.count_tp[q] for q in self.quantile_labels],
                    "count_sl": [self.count_sl[q] for q in self.quantile_labels],
                }
            )

        if name == "summary":
            return pl.DataFrame(
                {
                    "metric": [
                        "n_observations",
                        "n_quantiles",
                        "overall_profit_factor",
                        "overall_sum_returns",
                        "overall_avg_return",
                        "pf_monotonic",
                        "pf_spearman",
                    ],
                    "value": [
                        float(self.n_observations),
                        float(self.n_quantiles),
                        self.overall_profit_factor,
                        self.overall_sum_returns,
                        self.overall_avg_return,
                        float(self.pf_monotonic),
                        self.pf_spearman,
                    ],
                }
            )

        raise ValueError(
            f"Unknown DataFrame name: {name}. Available: 'profit_factor', 'returns', 'summary'"
        )

    def list_available_dataframes(self) -> list[str]:
        """List available DataFrame views."""
        return ["profit_factor", "returns", "summary"]

    def summary(self) -> str:
        """Get human-readable summary of profit factor results."""
        lines = [
            "=" * 60,
            "Barrier Profit Factor Analysis",
            "=" * 60,
            "",
            f"Observations:         {self.n_observations:>12,}",
            f"Quantiles:            {self.n_quantiles:>12}",
            "",
            "Overall Metrics:",
            f"  Profit Factor:      {self.overall_profit_factor:>12.2f}",
            f"  Sum Returns:        {self.overall_sum_returns:>12.4f}",
            f"  Avg Return:         {self.overall_avg_return:>12.4%}",
            "",
            "Monotonicity (PF vs Signal Strength):",
            f"  Monotonic:          {'Yes' if self.pf_monotonic else 'No':>12}",
            f"  Direction:          {self.pf_direction:>12}",
            f"  Spearman rho:       {self.pf_spearman:>12.4f}",
            "",
            "-" * 60,
            "Profit Factor by Quantile:",
            "-" * 60,
            f"{'Quantile':<10} {'PF':>8} {'Avg Ret':>10} {'Sum Ret':>12} {'Count':>8}",
        ]

        for q in self.quantile_labels:
            pf = self.profit_factor[q]
            avg = self.avg_return[q]
            total = self.sum_all_returns[q]
            count = self.count_total[q]
            lines.append(f"{q:<10} {pf:>8.2f} {avg:>10.4%} {total:>12.4f} {count:>8,}")

        return "\n".join(lines)
