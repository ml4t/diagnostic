"""Turnover analysis result classes for signal analysis.

This module provides result classes for storing turnover analysis outputs including
quantile turnover rates, signal autocorrelation, and stability metrics.

References
----------
Lopez de Prado, M. (2018). "Advances in Financial Machine Learning"
"""

from __future__ import annotations

import base64
from io import BytesIO
from typing import Any

import polars as pl
from pydantic import Field, model_validator

from ml4t.diagnostic.results.base import BaseResult
from ml4t.diagnostic.utils.formatting import format_finite
from ml4t.diagnostic.utils.validation import _validate_observation_keys


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

    dated_turnover_ipc_base64: str | None = Field(
        default=None, description="Owned dated turnover observations, encoded Arrow IPC/base64"
    )
    dated_autocorrelation_ipc_base64: str | None = Field(
        default=None, description="Owned full-date-rank observations, encoded Arrow IPC/base64"
    )

    @staticmethod
    def _observations(payload: str) -> pl.DataFrame:
        try:
            return pl.read_ipc(BytesIO(base64.b64decode(payload, validate=True)))
        except (ValueError, TypeError, OSError, pl.exceptions.PolarsError) as exc:
            raise ValueError("invalid dated persistence Arrow IPC/base64 payload") from exc

    @classmethod
    def from_observations(
        cls, turnover: pl.DataFrame, autocorrelation: pl.DataFrame
    ) -> TurnoverAnalysisResult:
        """Retain actual dated results, without reconstructing or duplicating horizons.

        Legacy summary maps remain empty. DataFrame views derive means only from
        defined observations, labelled by observation lag, never by return horizon.
        No half-life or sampling uncertainty is inferred from these series.
        """
        return cls(
            quantile_turnover={},
            mean_turnover={},
            top_quantile_turnover={},
            bottom_quantile_turnover={},
            autocorrelation={},
            autocorrelation_lags=autocorrelation["lag"].unique(maintain_order=True).to_list(),
            mean_autocorrelation={},
            half_life={},
            dated_turnover_ipc_base64=base64.b64encode(turnover.write_ipc(None).getvalue()).decode(
                "ascii"
            ),
            dated_autocorrelation_ipc_base64=base64.b64encode(
                autocorrelation.write_ipc(None).getvalue()
            ).decode("ascii"),
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

        if (self.dated_turnover_ipc_base64 is None) != (
            self.dated_autocorrelation_ipc_base64 is None
        ):
            raise ValueError("dated turnover and autocorrelation observations must both be present")
        if self.dated_turnover_ipc_base64 is not None:
            if any(
                [
                    self.quantile_turnover,
                    self.mean_turnover,
                    self.top_quantile_turnover,
                    self.bottom_quantile_turnover,
                    self.autocorrelation,
                    self.mean_autocorrelation,
                    self.half_life,
                ]
            ):
                raise ValueError(
                    "dated observations and legacy summary maps are distinct representations"
                )
            for payload, keys, required in [
                (
                    self.dated_turnover_ipc_base64,
                    ["date", "quantile", "lag"],
                    ["turnover", "turnover_method", "lag_unit", "clock_source"],
                ),
                (
                    self.dated_autocorrelation_ipc_base64,
                    ["date", "lag"],
                    [
                        "autocorrelation",
                        "n_obs",
                        "rank_scope",
                        "correlation_method",
                        "lag_unit",
                        "clock_source",
                    ],
                ),
            ]:
                frame = self._observations(payload)
                _validate_observation_keys(frame, keys, "dated persistence")
                if set(required).difference(frame.columns):
                    raise ValueError("dated persistence payload is missing observation columns")

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
        if self.dated_turnover_ipc_base64 is not None:
            turnover = self._observations(self.dated_turnover_ipc_base64)
            rank = self._observations(self.dated_autocorrelation_ipc_base64)
            if name == "dated_turnover":
                return turnover
            if name == "dated_autocorrelation":
                return rank
            if name is None or name == "turnover":
                return (
                    turnover.group_by("lag", "quantile")
                    .agg(pl.col("turnover").mean(), pl.col("turnover").count().alias("n_dates"))
                    .sort("lag", "quantile")
                )
            if name == "autocorrelation":
                return (
                    rank.group_by("lag")
                    .agg(
                        pl.col("autocorrelation").mean(),
                        pl.col("autocorrelation").count().alias("n_dates"),
                    )
                    .sort("lag")
                )
            if name == "summary":
                buckets = self.get_dataframe("turnover")
                means = buckets.group_by("lag").agg(
                    pl.col("turnover").mean().alias("mean_turnover"),
                    pl.col("turnover")
                    .filter(pl.col("quantile") == pl.col("quantile").max())
                    .first()
                    .alias("top_turnover"),
                    pl.col("turnover")
                    .filter(pl.col("quantile") == pl.col("quantile").min())
                    .first()
                    .alias("bottom_turnover"),
                )
                return (
                    means.join(
                        self.get_dataframe("autocorrelation").rename(
                            {"autocorrelation": "mean_autocorrelation", "n_dates": "n_rank_dates"}
                        ),
                        on="lag",
                        how="full",
                        coalesce=True,
                    )
                    .sort("lag")
                    .with_columns(
                        (pl.col("lag").cast(pl.String) + pl.lit("obs")).alias("period"),
                        pl.lit(None, dtype=pl.Float64).alias("half_life"),
                    )
                )
            raise ValueError(
                f"Unknown DataFrame name: {name}. Available: {self.list_available_dataframes()}"
            )
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
        names = ["turnover", "autocorrelation", "summary"]
        if self.dated_turnover_ipc_base64 is not None:
            names.extend(["dated_turnover", "dated_autocorrelation"])
        return names

    def summary(self) -> str:
        """Get human-readable summary of turnover analysis results."""
        lines = ["=" * 60, "Turnover Analysis Summary", "=" * 60, ""]
        if self.dated_turnover_ipc_base64 is not None:
            lines.append(
                "Entrants/current membership; full-date average ranks / Pearson; observation lags."
            )
            lines.append(str(self.get_dataframe("summary")))
            return "\n".join(lines)

        for period in self.mean_turnover:
            lines.append(f"Period: {period}")
            lines.append("-" * 40)
            mean = format_finite(self.mean_turnover[period], ".2%")
            top = format_finite(self.top_quantile_turnover[period], ".2%")
            bottom = format_finite(self.bottom_quantile_turnover[period], ".2%")
            autocorrelation = format_finite(self.mean_autocorrelation[period], ".4f")
            lines.append(f"  Mean Turnover:        {mean:>8}")
            lines.append(f"  Top Quantile:         {top:>8}")
            lines.append(f"  Bottom Quantile:      {bottom:>8}")
            lines.append(f"  Mean Autocorrelation: {autocorrelation:>8}")

            if self.half_life[period] is not None:
                half_life = format_finite(self.half_life[period], ".1f")
                lines.append(f"  Signal Half-Life:     {half_life:>8} periods")
            lines.append("")

        return "\n".join(lines)
