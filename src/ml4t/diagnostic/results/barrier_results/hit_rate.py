"""Hit rate analysis results for barrier outcomes.

This module provides the HitRateResult class for storing hit rate metrics
(TP, SL, timeout) by signal quantile, including chi-square independence tests.
"""

from __future__ import annotations

import polars as pl
from pydantic import Field, model_validator

from ml4t.diagnostic.results.barrier_results.validation import _validate_quantile_dict_keys
from ml4t.diagnostic.results.base import BaseResult


class HitRateResult(BaseResult):
    """Results from hit rate analysis by signal decile.

    Contains hit rates (% TP, % SL, % timeout) for each signal quantile,
    along with chi-square test for independence between signal strength
    and barrier outcome.

    Examples
    --------
    >>> result = hit_rate_result
    >>> print(result.summary())
    >>> df = result.get_dataframe("hit_rates")
    """

    analysis_type: str = Field(default="barrier_hit_rate", frozen=True)

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
    # Hit Rates by Quantile
    # ==========================================================================

    hit_rate_tp: dict[str, float] = Field(
        ...,
        description="Take-profit hit rate per quantile: {quantile: rate}",
    )

    hit_rate_sl: dict[str, float] = Field(
        ...,
        description="Stop-loss hit rate per quantile: {quantile: rate}",
    )

    hit_rate_timeout: dict[str, float] = Field(
        ...,
        description="Timeout hit rate per quantile: {quantile: rate}",
    )

    # ==========================================================================
    # Counts
    # ==========================================================================

    count_tp: dict[str, int] = Field(
        ...,
        description="Take-profit count per quantile",
    )

    count_sl: dict[str, int] = Field(
        ...,
        description="Stop-loss count per quantile",
    )

    count_timeout: dict[str, int] = Field(
        ...,
        description="Timeout count per quantile",
    )

    count_total: dict[str, int] = Field(
        ...,
        description="Total count per quantile",
    )

    # ==========================================================================
    # Statistical Test (Chi-Square Independence)
    # ==========================================================================

    chi2_statistic: float = Field(
        ...,
        description="Chi-square statistic for independence test",
    )

    chi2_p_value: float = Field(
        ...,
        description="P-value for chi-square test",
    )

    chi2_dof: int = Field(
        ...,
        description="Degrees of freedom for chi-square test",
    )

    is_significant: bool = Field(
        ...,
        description="Whether signal quantile significantly affects outcome (p < alpha)",
    )

    significance_level: float = Field(
        ...,
        description="Significance level used for test",
    )

    # ==========================================================================
    # Aggregates
    # ==========================================================================

    overall_hit_rate_tp: float = Field(
        ...,
        description="Overall take-profit hit rate across all observations",
    )

    overall_hit_rate_sl: float = Field(
        ...,
        description="Overall stop-loss hit rate across all observations",
    )

    overall_hit_rate_timeout: float = Field(
        ...,
        description="Overall timeout hit rate across all observations",
    )

    n_observations: int = Field(
        ...,
        description="Total number of observations analyzed",
    )

    # ==========================================================================
    # Monotonicity
    # ==========================================================================

    tp_rate_monotonic: bool = Field(
        ...,
        description="Whether TP hit rate is monotonic across quantiles",
    )

    tp_rate_direction: str = Field(
        ...,
        description="Direction of TP rate change: 'increasing', 'decreasing', or 'none'",
    )

    tp_rate_spearman: float = Field(
        ...,
        description="Spearman correlation between quantile rank and TP hit rate",
    )

    # ==========================================================================
    # Validation
    # ==========================================================================

    @model_validator(mode="after")
    def _validate_quantile_keys(self) -> HitRateResult:
        """Validate that all quantile-keyed dicts have consistent keys."""
        if self.n_quantiles != len(self.quantile_labels):
            raise ValueError(
                f"n_quantiles ({self.n_quantiles}) != len(quantile_labels) ({len(self.quantile_labels)})"
            )
        _validate_quantile_dict_keys(
            self.quantile_labels,
            [
                ("hit_rate_tp", self.hit_rate_tp),
                ("hit_rate_sl", self.hit_rate_sl),
                ("hit_rate_timeout", self.hit_rate_timeout),
                ("count_tp", self.count_tp),
                ("count_sl", self.count_sl),
                ("count_timeout", self.count_timeout),
                ("count_total", self.count_total),
            ],
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
            - None or "hit_rates": Hit rates by quantile
            - "counts": Raw counts by quantile and outcome
            - "summary": Single-row summary statistics

        Returns
        -------
        pl.DataFrame
            Requested DataFrame
        """
        if name is None or name == "hit_rates":
            return pl.DataFrame(
                {
                    "quantile": self.quantile_labels,
                    "hit_rate_tp": [self.hit_rate_tp[q] for q in self.quantile_labels],
                    "hit_rate_sl": [self.hit_rate_sl[q] for q in self.quantile_labels],
                    "hit_rate_timeout": [self.hit_rate_timeout[q] for q in self.quantile_labels],
                    "count_total": [self.count_total[q] for q in self.quantile_labels],
                }
            )

        if name == "counts":
            return pl.DataFrame(
                {
                    "quantile": self.quantile_labels,
                    "count_tp": [self.count_tp[q] for q in self.quantile_labels],
                    "count_sl": [self.count_sl[q] for q in self.quantile_labels],
                    "count_timeout": [self.count_timeout[q] for q in self.quantile_labels],
                    "count_total": [self.count_total[q] for q in self.quantile_labels],
                }
            )

        if name == "summary":
            return pl.DataFrame(
                {
                    "metric": [
                        "n_observations",
                        "n_quantiles",
                        "overall_hit_rate_tp",
                        "overall_hit_rate_sl",
                        "overall_hit_rate_timeout",
                        "chi2_statistic",
                        "chi2_p_value",
                        "is_significant",
                        "tp_rate_monotonic",
                        "tp_rate_spearman",
                    ],
                    "value": [
                        float(self.n_observations),
                        float(self.n_quantiles),
                        self.overall_hit_rate_tp,
                        self.overall_hit_rate_sl,
                        self.overall_hit_rate_timeout,
                        self.chi2_statistic,
                        self.chi2_p_value,
                        float(self.is_significant),
                        float(self.tp_rate_monotonic),
                        self.tp_rate_spearman,
                    ],
                }
            )

        raise ValueError(
            f"Unknown DataFrame name: {name}. Available: 'hit_rates', 'counts', 'summary'"
        )

    def list_available_dataframes(self) -> list[str]:
        """List available DataFrame views."""
        return ["hit_rates", "counts", "summary"]

    def summary(self) -> str:
        """Get human-readable summary of hit rate results."""
        lines = [
            "=" * 60,
            "Barrier Hit Rate Analysis",
            "=" * 60,
            "",
            f"Observations:     {self.n_observations:>10,}",
            f"Quantiles:        {self.n_quantiles:>10}",
            "",
            "Overall Hit Rates:",
            f"  Take-Profit:    {self.overall_hit_rate_tp:>10.1%}",
            f"  Stop-Loss:      {self.overall_hit_rate_sl:>10.1%}",
            f"  Timeout:        {self.overall_hit_rate_timeout:>10.1%}",
            "",
            "Chi-Square Test (Signal Decile vs Outcome):",
            f"  Chi2 Statistic: {self.chi2_statistic:>10.2f}",
            f"  P-value:        {self.chi2_p_value:>10.4f}",
            f"  DoF:            {self.chi2_dof:>10}",
            f"  Significant:    {'Yes' if self.is_significant else 'No':>10} (alpha={self.significance_level})",
            "",
            "Monotonicity (TP Rate vs Signal Strength):",
            f"  Monotonic:      {'Yes' if self.tp_rate_monotonic else 'No':>10}",
            f"  Direction:      {self.tp_rate_direction:>10}",
            f"  Spearman rho:   {self.tp_rate_spearman:>10.4f}",
            "",
            "-" * 60,
            "Hit Rates by Quantile:",
            "-" * 60,
            f"{'Quantile':<10} {'TP Rate':>10} {'SL Rate':>10} {'Timeout':>10} {'Count':>8}",
        ]

        for q in self.quantile_labels:
            lines.append(
                f"{q:<10} {self.hit_rate_tp[q]:>10.1%} {self.hit_rate_sl[q]:>10.1%} "
                f"{self.hit_rate_timeout[q]:>10.1%} {self.count_total[q]:>8,}"
            )

        return "\n".join(lines)
