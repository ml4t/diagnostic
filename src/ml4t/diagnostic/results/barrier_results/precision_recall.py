"""Precision/recall analysis results for barrier outcomes.

This module provides the PrecisionRecallResult class for storing precision,
recall, F1 scores, and lift metrics for barrier outcomes by signal quantile.
"""

from __future__ import annotations

import polars as pl
from pydantic import Field, model_validator

from ml4t.diagnostic.results.barrier_results.validation import _validate_quantile_dict_keys
from ml4t.diagnostic.results.base import BaseResult


class PrecisionRecallResult(BaseResult):
    """Results from precision/recall analysis for barrier outcomes.

    Precision: Of signals in top quantile, what fraction hit TP?
    Recall: Of all TP outcomes, what fraction came from top quantile?

    This helps understand signal selectivity vs coverage trade-offs.

    Examples
    --------
    >>> result = precision_recall_result
    >>> print(result.summary())
    >>> df = result.get_dataframe()
    """

    analysis_type: str = Field(default="barrier_precision_recall", frozen=True)

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
    # Precision by Quantile (TP-focused)
    # ==========================================================================

    precision_tp: dict[str, float] = Field(
        ...,
        description="Precision for TP: P(TP | in quantile) = TP count / total in quantile",
    )

    # ==========================================================================
    # Recall by Quantile (TP-focused)
    # ==========================================================================

    recall_tp: dict[str, float] = Field(
        ...,
        description="Recall for TP: P(in quantile | TP) = TP in quantile / all TP",
    )

    # ==========================================================================
    # Cumulative Metrics (from top quantile down)
    # ==========================================================================

    cumulative_precision_tp: dict[str, float] = Field(
        ...,
        description="Cumulative precision: P(TP | in top k quantiles)",
    )

    cumulative_recall_tp: dict[str, float] = Field(
        ...,
        description="Cumulative recall: P(in top k quantiles | TP)",
    )

    cumulative_f1_tp: dict[str, float] = Field(
        ...,
        description="Cumulative F1 score: 2 * (precision * recall) / (precision + recall)",
    )

    # ==========================================================================
    # Lift Metrics
    # ==========================================================================

    lift_tp: dict[str, float] = Field(
        ...,
        description="Lift for TP: precision / baseline TP rate",
    )

    cumulative_lift_tp: dict[str, float] = Field(
        ...,
        description="Cumulative lift for TP",
    )

    # ==========================================================================
    # Baseline
    # ==========================================================================

    baseline_tp_rate: float = Field(
        ...,
        description="Baseline TP rate (overall TP count / total)",
    )

    total_tp_count: int = Field(
        ...,
        description="Total number of TP outcomes",
    )

    n_observations: int = Field(
        ...,
        description="Total number of observations",
    )

    # ==========================================================================
    # Best Operating Point
    # ==========================================================================

    best_f1_quantile: str = Field(
        ...,
        description="Quantile with best cumulative F1 score",
    )

    best_f1_score: float = Field(
        ...,
        description="Best cumulative F1 score achieved",
    )

    # ==========================================================================
    # Validation
    # ==========================================================================

    @model_validator(mode="after")
    def _validate_quantile_keys(self) -> PrecisionRecallResult:
        """Validate that all quantile-keyed dicts have consistent keys."""
        if self.n_quantiles != len(self.quantile_labels):
            raise ValueError(
                f"n_quantiles ({self.n_quantiles}) != len(quantile_labels) ({len(self.quantile_labels)})"
            )
        _validate_quantile_dict_keys(
            self.quantile_labels,
            [
                ("precision_tp", self.precision_tp),
                ("recall_tp", self.recall_tp),
                ("cumulative_precision_tp", self.cumulative_precision_tp),
                ("cumulative_recall_tp", self.cumulative_recall_tp),
                ("cumulative_f1_tp", self.cumulative_f1_tp),
                ("lift_tp", self.lift_tp),
                ("cumulative_lift_tp", self.cumulative_lift_tp),
            ],
        )
        return self

    def get_dataframe(self, name: str | None = None) -> pl.DataFrame:
        """Get results as Polars DataFrame.

        Parameters
        ----------
        name : str | None
            DataFrame to retrieve:
            - None or "precision_recall": Per-quantile metrics
            - "cumulative": Cumulative metrics from top down
            - "summary": Summary statistics

        Returns
        -------
        pl.DataFrame
            Requested DataFrame
        """
        if name is None or name == "precision_recall":
            return pl.DataFrame(
                {
                    "quantile": self.quantile_labels,
                    "precision_tp": [self.precision_tp[q] for q in self.quantile_labels],
                    "recall_tp": [self.recall_tp[q] for q in self.quantile_labels],
                    "lift_tp": [self.lift_tp[q] for q in self.quantile_labels],
                }
            )

        if name == "cumulative":
            return pl.DataFrame(
                {
                    "quantile": self.quantile_labels,
                    "cumulative_precision_tp": [
                        self.cumulative_precision_tp[q] for q in self.quantile_labels
                    ],
                    "cumulative_recall_tp": [
                        self.cumulative_recall_tp[q] for q in self.quantile_labels
                    ],
                    "cumulative_f1_tp": [self.cumulative_f1_tp[q] for q in self.quantile_labels],
                    "cumulative_lift_tp": [
                        self.cumulative_lift_tp[q] for q in self.quantile_labels
                    ],
                }
            )

        if name == "summary":
            return pl.DataFrame(
                {
                    "metric": [
                        "n_observations",
                        "n_quantiles",
                        "total_tp_count",
                        "baseline_tp_rate",
                        "best_f1_quantile",
                        "best_f1_score",
                    ],
                    "value": [
                        float(self.n_observations),
                        float(self.n_quantiles),
                        float(self.total_tp_count),
                        self.baseline_tp_rate,
                        self.best_f1_quantile,
                        self.best_f1_score,
                    ],
                }
            )

        raise ValueError(
            f"Unknown DataFrame name: {name}. Available: 'precision_recall', 'cumulative', 'summary'"
        )

    def list_available_dataframes(self) -> list[str]:
        """List available DataFrame views."""
        return ["precision_recall", "cumulative", "summary"]

    def summary(self) -> str:
        """Get human-readable summary of precision/recall results."""
        lines = [
            "=" * 60,
            "Barrier Precision/Recall Analysis (TP-focused)",
            "=" * 60,
            "",
            f"Observations:      {self.n_observations:>10,}",
            f"Total TP Count:    {self.total_tp_count:>10,}",
            f"Baseline TP Rate:  {self.baseline_tp_rate:>10.1%}",
            "",
            f"Best F1 Score:     {self.best_f1_score:>10.4f} (at {self.best_f1_quantile})",
            "",
            "-" * 60,
            "Per-Quantile Metrics:",
            "-" * 60,
            f"{'Quantile':<10} {'Precision':>10} {'Recall':>10} {'Lift':>8}",
        ]

        for q in self.quantile_labels:
            lines.append(
                f"{q:<10} {self.precision_tp[q]:>10.1%} {self.recall_tp[q]:>10.1%} "
                f"{self.lift_tp[q]:>8.2f}x"
            )

        lines.append("")
        lines.append("-" * 60)
        lines.append("Cumulative Metrics (from top quantile):")
        lines.append("-" * 60)
        lines.append(f"{'Quantile':<10} {'Cum Prec':>10} {'Cum Recall':>10} {'Cum F1':>10}")

        for q in self.quantile_labels:
            lines.append(
                f"{q:<10} {self.cumulative_precision_tp[q]:>10.1%} "
                f"{self.cumulative_recall_tp[q]:>10.1%} {self.cumulative_f1_tp[q]:>10.4f}"
            )

        return "\n".join(lines)
