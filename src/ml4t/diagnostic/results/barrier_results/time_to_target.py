"""Time-to-target analysis results for barrier outcomes.

This module provides the TimeToTargetResult class for storing time-to-target
metrics (mean, median, std bars to TP/SL/timeout) by signal quantile.
"""

from __future__ import annotations

import polars as pl
from pydantic import Field, model_validator

from ml4t.diagnostic.results.barrier_results.validation import _validate_quantile_dict_keys
from ml4t.diagnostic.results.base import BaseResult


class TimeToTargetResult(BaseResult):
    """Results from time-to-target analysis by signal decile.

    Analyzes how quickly different signal quantiles reach their barrier
    outcomes (TP, SL, or timeout).

    Examples
    --------
    >>> result = time_to_target_result
    >>> print(result.summary())
    >>> df = result.get_dataframe()
    """

    analysis_type: str = Field(default="barrier_time_to_target", frozen=True)

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
    # Mean Time to Exit by Quantile and Outcome
    # ==========================================================================

    mean_bars_tp: dict[str, float] = Field(
        ...,
        description="Mean bars to TP per quantile",
    )

    mean_bars_sl: dict[str, float] = Field(
        ...,
        description="Mean bars to SL per quantile",
    )

    mean_bars_timeout: dict[str, float] = Field(
        ...,
        description="Mean bars to timeout per quantile",
    )

    mean_bars_all: dict[str, float] = Field(
        ...,
        description="Mean bars to any exit per quantile",
    )

    # ==========================================================================
    # Median Time to Exit
    # ==========================================================================

    median_bars_tp: dict[str, float] = Field(
        ...,
        description="Median bars to TP per quantile",
    )

    median_bars_sl: dict[str, float] = Field(
        ...,
        description="Median bars to SL per quantile",
    )

    median_bars_all: dict[str, float] = Field(
        ...,
        description="Median bars to any exit per quantile",
    )

    # ==========================================================================
    # Standard Deviation
    # ==========================================================================

    std_bars_tp: dict[str, float] = Field(
        ...,
        description="Std dev of bars to TP per quantile",
    )

    std_bars_sl: dict[str, float] = Field(
        ...,
        description="Std dev of bars to SL per quantile",
    )

    std_bars_all: dict[str, float] = Field(
        ...,
        description="Std dev of bars to any exit per quantile",
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

    count_timeout: dict[str, int] = Field(
        ...,
        description="Number of timeout outcomes per quantile",
    )

    # ==========================================================================
    # Overall Statistics
    # ==========================================================================

    overall_mean_bars: float = Field(
        ...,
        description="Overall mean bars to exit",
    )

    overall_median_bars: float = Field(
        ...,
        description="Overall median bars to exit",
    )

    overall_mean_bars_tp: float = Field(
        ...,
        description="Overall mean bars to TP",
    )

    overall_mean_bars_sl: float = Field(
        ...,
        description="Overall mean bars to SL",
    )

    n_observations: int = Field(
        ...,
        description="Total number of observations",
    )

    # ==========================================================================
    # Speed Analysis
    # ==========================================================================

    tp_faster_than_sl: dict[str, bool] = Field(
        ...,
        description="Whether TP is reached faster than SL on average per quantile",
    )

    speed_advantage_tp: dict[str, float] = Field(
        ...,
        description="Speed advantage of TP over SL (positive = TP faster) per quantile",
    )

    # ==========================================================================
    # Validation
    # ==========================================================================

    @model_validator(mode="after")
    def _validate_quantile_keys(self) -> TimeToTargetResult:
        """Validate that all quantile-keyed dicts have consistent keys."""
        if self.n_quantiles != len(self.quantile_labels):
            raise ValueError(
                f"n_quantiles ({self.n_quantiles}) != len(quantile_labels) ({len(self.quantile_labels)})"
            )
        _validate_quantile_dict_keys(
            self.quantile_labels,
            [
                ("mean_bars_tp", self.mean_bars_tp),
                ("mean_bars_sl", self.mean_bars_sl),
                ("mean_bars_timeout", self.mean_bars_timeout),
                ("mean_bars_all", self.mean_bars_all),
                ("median_bars_tp", self.median_bars_tp),
                ("median_bars_sl", self.median_bars_sl),
                ("median_bars_all", self.median_bars_all),
                ("std_bars_tp", self.std_bars_tp),
                ("std_bars_sl", self.std_bars_sl),
                ("std_bars_all", self.std_bars_all),
                ("count_tp", self.count_tp),
                ("count_sl", self.count_sl),
                ("count_timeout", self.count_timeout),
                ("tp_faster_than_sl", self.tp_faster_than_sl),
                ("speed_advantage_tp", self.speed_advantage_tp),
            ],
        )
        return self

    def get_dataframe(self, name: str | None = None) -> pl.DataFrame:
        """Get results as Polars DataFrame.

        Parameters
        ----------
        name : str | None
            DataFrame to retrieve:
            - None or "time_to_target": Mean times by quantile and outcome
            - "detailed": Full statistics including median and std
            - "summary": Overall statistics

        Returns
        -------
        pl.DataFrame
            Requested DataFrame
        """
        if name is None or name == "time_to_target":
            return pl.DataFrame(
                {
                    "quantile": self.quantile_labels,
                    "mean_bars_tp": [self.mean_bars_tp[q] for q in self.quantile_labels],
                    "mean_bars_sl": [self.mean_bars_sl[q] for q in self.quantile_labels],
                    "mean_bars_timeout": [self.mean_bars_timeout[q] for q in self.quantile_labels],
                    "mean_bars_all": [self.mean_bars_all[q] for q in self.quantile_labels],
                    "tp_faster": [self.tp_faster_than_sl[q] for q in self.quantile_labels],
                }
            )

        if name == "detailed":
            return pl.DataFrame(
                {
                    "quantile": self.quantile_labels,
                    "mean_bars_tp": [self.mean_bars_tp[q] for q in self.quantile_labels],
                    "median_bars_tp": [self.median_bars_tp[q] for q in self.quantile_labels],
                    "std_bars_tp": [self.std_bars_tp[q] for q in self.quantile_labels],
                    "mean_bars_sl": [self.mean_bars_sl[q] for q in self.quantile_labels],
                    "median_bars_sl": [self.median_bars_sl[q] for q in self.quantile_labels],
                    "std_bars_sl": [self.std_bars_sl[q] for q in self.quantile_labels],
                    "count_tp": [self.count_tp[q] for q in self.quantile_labels],
                    "count_sl": [self.count_sl[q] for q in self.quantile_labels],
                    "count_timeout": [self.count_timeout[q] for q in self.quantile_labels],
                }
            )

        if name == "summary":
            return pl.DataFrame(
                {
                    "metric": [
                        "n_observations",
                        "n_quantiles",
                        "overall_mean_bars",
                        "overall_median_bars",
                        "overall_mean_bars_tp",
                        "overall_mean_bars_sl",
                    ],
                    "value": [
                        float(self.n_observations),
                        float(self.n_quantiles),
                        self.overall_mean_bars,
                        self.overall_median_bars,
                        self.overall_mean_bars_tp,
                        self.overall_mean_bars_sl,
                    ],
                }
            )

        raise ValueError(
            f"Unknown DataFrame name: {name}. Available: 'time_to_target', 'detailed', 'summary'"
        )

    def list_available_dataframes(self) -> list[str]:
        """List available DataFrame views."""
        return ["time_to_target", "detailed", "summary"]

    def summary(self) -> str:
        """Get human-readable summary of time-to-target results."""
        lines = [
            "=" * 60,
            "Barrier Time-to-Target Analysis",
            "=" * 60,
            "",
            f"Observations:        {self.n_observations:>10,}",
            f"Quantiles:           {self.n_quantiles:>10}",
            "",
            "Overall Time to Exit:",
            f"  Mean Bars:         {self.overall_mean_bars:>10.1f}",
            f"  Median Bars:       {self.overall_median_bars:>10.1f}",
            f"  Mean Bars (TP):    {self.overall_mean_bars_tp:>10.1f}",
            f"  Mean Bars (SL):    {self.overall_mean_bars_sl:>10.1f}",
            "",
            "-" * 60,
            "Mean Bars to Exit by Quantile:",
            "-" * 60,
            f"{'Quantile':<10} {'TP':>8} {'SL':>8} {'Timeout':>8} {'All':>8} {'TP Faster?':>12}",
        ]

        for q in self.quantile_labels:
            tp_faster = "Yes" if self.tp_faster_than_sl[q] else "No"
            lines.append(
                f"{q:<10} {self.mean_bars_tp[q]:>8.1f} {self.mean_bars_sl[q]:>8.1f} "
                f"{self.mean_bars_timeout[q]:>8.1f} {self.mean_bars_all[q]:>8.1f} {tp_faster:>12}"
            )

        return "\n".join(lines)
