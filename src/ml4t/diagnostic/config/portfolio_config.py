"""Portfolio Evaluation Configuration.

This module defines configuration for portfolio performance evaluation:
- Risk/return metrics (Sharpe, Sortino, Calmar, VaR, CVaR)
- Bayesian comparison (probabilistic strategy comparison)
- Time aggregation (daily, weekly, monthly, etc.)
- Drawdown analysis (underwater curves, recovery times)

Consolidated Config:
- PortfolioConfig: Single config with all portfolio analysis settings
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field, field_validator, model_validator

from ml4t.diagnostic.config.base import BaseConfig
from ml4t.diagnostic.config.validation import (
    BayesianPriorDistribution,
    NonNegativeFloat,
    PortfolioMetric,
    PositiveInt,
    Probability,
    TimeFrequency,
)

# =============================================================================
# Settings Classes (Single-Level Nesting)
# =============================================================================


class MetricsSettings(BaseConfig):
    """Settings for risk/return metrics."""

    metrics: list[PortfolioMetric] = Field(
        default_factory=lambda: [
            PortfolioMetric.SHARPE,
            PortfolioMetric.SORTINO,
            PortfolioMetric.CALMAR,
            PortfolioMetric.MAX_DRAWDOWN,
        ],
        description="Metrics to compute",
    )
    risk_free_rate: NonNegativeFloat = Field(0.0, description="Annualized risk-free rate")
    confidence_level: Probability = Field(0.95, description="Confidence for VaR/CVaR")
    periods_per_year: PositiveInt = Field(252, description="Trading periods per year")
    downside_target: float = Field(0.0, description="Target for Sortino")
    omega_threshold: float = Field(0.0, description="Omega threshold")

    @field_validator("metrics")
    @classmethod
    def check_metrics(cls, v: list[PortfolioMetric]) -> list[PortfolioMetric]:
        """Ensure at least one metric specified."""
        if not v:
            raise ValueError("Must specify at least one metric")
        return v


class BayesianSettings(BaseConfig):
    """Settings for Bayesian strategy comparison."""

    enabled: bool = Field(False, description="Run Bayesian comparison")
    prior_distribution: BayesianPriorDistribution = Field(BayesianPriorDistribution.NORMAL)
    prior_params: dict[str, float] = Field(default_factory=lambda: {"mean": 0.0, "std": 1.0})
    n_samples: PositiveInt = Field(10000, description="MCMC samples")
    credible_interval: Probability = Field(0.95)
    compare_to_benchmark: bool = Field(False)
    benchmark_column: str | None = Field(None)

    @model_validator(mode="after")
    def validate_benchmark(self) -> BayesianSettings:
        """Validate benchmark configuration."""
        if self.compare_to_benchmark and not self.benchmark_column:
            raise ValueError("benchmark_column required when compare_to_benchmark=True")
        return self

    @model_validator(mode="after")
    def validate_prior_params(self) -> BayesianSettings:
        """Validate prior parameters match distribution."""
        required_params = {
            BayesianPriorDistribution.NORMAL: {"mean", "std"},
            BayesianPriorDistribution.STUDENT_T: {"df", "loc", "scale"},
            BayesianPriorDistribution.UNIFORM: {"low", "high"},
        }
        required = required_params[self.prior_distribution]
        provided = set(self.prior_params.keys())
        if required != provided:
            raise ValueError(f"Prior {self.prior_distribution} requires {required}, got {provided}")
        return self


class AggregationSettings(BaseConfig):
    """Settings for time aggregation analysis."""

    frequencies: list[TimeFrequency] = Field(default_factory=lambda: [TimeFrequency.DAILY])
    compute_rolling: bool = Field(False)
    rolling_windows: list[PositiveInt] = Field(default_factory=lambda: [21, 63, 252])
    min_periods: PositiveInt | None = Field(None)
    align_to_calendar: bool = Field(True)

    @field_validator("frequencies")
    @classmethod
    def check_frequencies(cls, v: list[TimeFrequency]) -> list[TimeFrequency]:
        """Ensure at least one frequency specified."""
        if not v:
            raise ValueError("Must specify at least one frequency")
        return v

    @field_validator("rolling_windows")
    @classmethod
    def check_rolling_windows(cls, v: list[int]) -> list[int]:
        """Sort rolling windows for consistency."""
        return sorted(v)


class DrawdownSettings(BaseConfig):
    """Settings for drawdown analysis."""

    enabled: bool = Field(True)
    compute_underwater_curve: bool = Field(True)
    top_n_drawdowns: PositiveInt = Field(5)
    compute_recovery_time: bool = Field(True)
    recovery_threshold: Probability = Field(1.0)


# =============================================================================
# Consolidated Config
# =============================================================================


class PortfolioConfig(BaseConfig):
    """Consolidated configuration for portfolio evaluation.

    Orchestrates portfolio performance analysis with metrics, Bayesian
    comparison, time aggregation, and drawdown analysis.

    Examples
    --------
    >>> config = PortfolioConfig(
    ...     metrics=MetricsSettings(risk_free_rate=0.02),
    ...     bayesian=BayesianSettings(enabled=True),
    ... )
    >>> config.to_yaml("portfolio_config.yaml")
    """

    metrics: MetricsSettings = Field(
        default_factory=MetricsSettings, description="Metrics settings"
    )
    bayesian: BayesianSettings = Field(
        default_factory=BayesianSettings, description="Bayesian comparison"
    )
    aggregation: AggregationSettings = Field(
        default_factory=AggregationSettings, description="Time aggregation"
    )
    drawdown: DrawdownSettings = Field(
        default_factory=DrawdownSettings, description="Drawdown analysis"
    )

    return_dataframes: bool = Field(True, description="Return as DataFrames")
    n_jobs: int = Field(-1, ge=-1, description="Parallel jobs")
    cache_enabled: bool = Field(True)
    cache_dir: Path = Field(
        default_factory=lambda: Path.home() / ".cache" / "ml4t-diagnostic" / "portfolio"
    )
    verbose: bool = Field(False)

    @classmethod
    def for_quick_analysis(cls) -> PortfolioConfig:
        """Preset for quick exploratory analysis."""
        return cls(
            metrics=MetricsSettings(metrics=[PortfolioMetric.SHARPE, PortfolioMetric.MAX_DRAWDOWN]),
            bayesian=BayesianSettings(enabled=False),
            aggregation=AggregationSettings(compute_rolling=False),
            drawdown=DrawdownSettings(compute_recovery_time=False),
        )

    @classmethod
    def for_research(cls) -> PortfolioConfig:
        """Preset for academic research."""
        return cls(
            metrics=MetricsSettings(
                metrics=[
                    PortfolioMetric.SHARPE,
                    PortfolioMetric.SORTINO,
                    PortfolioMetric.CALMAR,
                    PortfolioMetric.MAX_DRAWDOWN,
                    PortfolioMetric.VAR,
                    PortfolioMetric.CVAR,
                    PortfolioMetric.OMEGA,
                ]
            ),
            bayesian=BayesianSettings(enabled=True, n_samples=50000),
            aggregation=AggregationSettings(
                frequencies=[TimeFrequency.DAILY, TimeFrequency.WEEKLY, TimeFrequency.MONTHLY],
                compute_rolling=True,
                rolling_windows=[21, 63, 126, 252],
            ),
            drawdown=DrawdownSettings(compute_underwater_curve=True, top_n_drawdowns=10),
        )

    @classmethod
    def for_production(cls) -> PortfolioConfig:
        """Preset for production monitoring."""
        return cls(
            metrics=MetricsSettings(
                metrics=[PortfolioMetric.SHARPE, PortfolioMetric.MAX_DRAWDOWN, PortfolioMetric.VAR]
            ),
            bayesian=BayesianSettings(enabled=False),
            aggregation=AggregationSettings(
                frequencies=[TimeFrequency.DAILY], compute_rolling=True, rolling_windows=[21, 63]
            ),
            drawdown=DrawdownSettings(compute_recovery_time=False),
        )
