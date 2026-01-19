"""Result schemas for portfolio evaluation (Module D).

Portfolio-level metrics and Bayesian strategy comparison.
"""

from __future__ import annotations

from typing import Any

import polars as pl
from pydantic import Field

from ml4t.diagnostic.results.base import BaseResult


class PortfolioMetrics(BaseResult):
    """Standard portfolio performance metrics.

    Comprehensive set of risk-adjusted return metrics commonly used
    in quantitative finance.

    Attributes:
        total_return: Cumulative return over period
        annualized_return: Annualized return (CAGR)
        annualized_volatility: Annualized standard deviation of returns
        sharpe_ratio: Sharpe ratio (excess return / volatility)
        sortino_ratio: Sortino ratio (excess return / downside deviation)
        max_drawdown: Maximum peak-to-trough decline
        calmar_ratio: Return / max drawdown
        omega_ratio: Probability-weighted ratio of gains vs losses
        win_rate: Fraction of positive return periods
        avg_win: Average return on winning periods
        avg_loss: Average return on losing periods
        profit_factor: Gross profit / gross loss
        skewness: Return distribution skewness
        kurtosis: Return distribution kurtosis
    """

    analysis_type: str = "portfolio_metrics"

    # Returns
    total_return: float = Field(..., description="Cumulative return")
    annualized_return: float = Field(..., description="Annualized return (CAGR)")

    # Risk
    annualized_volatility: float = Field(..., description="Annualized standard deviation")
    max_drawdown: float = Field(..., description="Maximum drawdown (peak to trough)")

    # Risk-adjusted returns
    sharpe_ratio: float = Field(..., description="Sharpe ratio")
    sortino_ratio: float = Field(..., description="Sortino ratio")
    calmar_ratio: float = Field(..., description="Calmar ratio (return / max DD)")
    omega_ratio: float | None = Field(None, description="Omega ratio")

    # Win/loss statistics
    win_rate: float = Field(..., description="Fraction of winning periods")
    avg_win: float = Field(..., description="Average winning return")
    avg_loss: float = Field(..., description="Average losing return")
    profit_factor: float = Field(..., description="Gross profit / gross loss")

    # Distribution characteristics
    skewness: float = Field(..., description="Return skewness")
    kurtosis: float = Field(..., description="Return excess kurtosis")

    def get_dataframe(self, name: str | None = None) -> pl.DataFrame:
        """Get metrics as single-row DataFrame.

        Returns:
            DataFrame with all metrics
        """
        data = {
            "total_return": [self.total_return],
            "annualized_return": [self.annualized_return],
            "annualized_volatility": [self.annualized_volatility],
            "sharpe_ratio": [self.sharpe_ratio],
            "sortino_ratio": [self.sortino_ratio],
            "max_drawdown": [self.max_drawdown],
            "calmar_ratio": [self.calmar_ratio],
            "omega_ratio": [self.omega_ratio],
            "win_rate": [self.win_rate],
            "avg_win": [self.avg_win],
            "avg_loss": [self.avg_loss],
            "profit_factor": [self.profit_factor],
            "skewness": [self.skewness],
            "kurtosis": [self.kurtosis],
        }
        return pl.DataFrame(data)

    def summary(self) -> str:
        """Human-readable summary of portfolio metrics."""
        lines = ["Portfolio Metrics Summary", "=" * 40]
        lines.append(f"Total Return: {self.total_return:.2%}")
        lines.append(f"Annualized Return: {self.annualized_return:.2%}")
        lines.append(f"Annualized Volatility: {self.annualized_volatility:.2%}")
        lines.append(f"Sharpe Ratio: {self.sharpe_ratio:.3f}")
        lines.append(f"Sortino Ratio: {self.sortino_ratio:.3f}")
        lines.append(f"Max Drawdown: {self.max_drawdown:.2%}")
        lines.append(f"Calmar Ratio: {self.calmar_ratio:.3f}")
        lines.append("")
        lines.append(f"Win Rate: {self.win_rate:.2%}")
        lines.append(f"Profit Factor: {self.profit_factor:.2f}")
        lines.append(f"Skewness: {self.skewness:.3f}")
        lines.append(f"Kurtosis: {self.kurtosis:.3f}")
        return "\n".join(lines)


class BayesianComparisonResult(BaseResult):
    """Bayesian strategy comparison results.

    Uses Bayesian inference to compare two strategies, accounting for
    parameter uncertainty and providing probabilistic conclusions.

    Reference: Bailey & López de Prado (2012) "The Sharpe Ratio Efficient Frontier"

    Attributes:
        strategy_a_name: Name of first strategy
        strategy_b_name: Name of second strategy
        prior_sharpe_mean: Prior belief about Sharpe ratio mean
        prior_sharpe_std: Prior belief about Sharpe ratio std
        posterior_sharpe_mean: Updated belief after observing data
        posterior_sharpe_std: Updated uncertainty
        probability_a_better: P(Sharpe_A > Sharpe_B | data)
        credible_interval_95: 95% credible interval for Sharpe difference
    """

    analysis_type: str = "bayesian_comparison"

    strategy_a_name: str = Field(..., description="First strategy name")
    strategy_b_name: str = Field(..., description="Second strategy name")

    prior_sharpe_mean: float = Field(..., description="Prior Sharpe mean")
    prior_sharpe_std: float = Field(..., description="Prior Sharpe std")

    posterior_sharpe_mean: float = Field(..., description="Posterior Sharpe mean")
    posterior_sharpe_std: float = Field(..., description="Posterior Sharpe std")

    probability_a_better: float = Field(..., description="P(strategy A > strategy B | data)")
    credible_interval_95: tuple[float, float] = Field(
        ..., description="95% credible interval for Sharpe difference"
    )

    def get_dataframe(self, name: str | None = None) -> pl.DataFrame:
        """Get comparison results as single-row DataFrame.

        Returns:
            DataFrame with comparison statistics
        """
        data = {
            "strategy_a": [self.strategy_a_name],
            "strategy_b": [self.strategy_b_name],
            "prior_mean": [self.prior_sharpe_mean],
            "prior_std": [self.prior_sharpe_std],
            "posterior_mean": [self.posterior_sharpe_mean],
            "posterior_std": [self.posterior_sharpe_std],
            "prob_a_better": [self.probability_a_better],
            "ci_lower": [self.credible_interval_95[0]],
            "ci_upper": [self.credible_interval_95[1]],
        }
        return pl.DataFrame(data)

    def summary(self) -> str:
        """Human-readable summary of Bayesian comparison."""
        lines = ["Bayesian Strategy Comparison", "=" * 40]
        lines.append(f"Strategy A: {self.strategy_a_name}")
        lines.append(f"Strategy B: {self.strategy_b_name}")
        lines.append("")
        lines.append(
            f"Prior: Sharpe ~ N({self.prior_sharpe_mean:.3f}, {self.prior_sharpe_std:.3f})"
        )
        lines.append(
            f"Posterior: Sharpe ~ N({self.posterior_sharpe_mean:.3f}, {self.posterior_sharpe_std:.3f})"
        )
        lines.append("")
        lines.append(
            f"P({self.strategy_a_name} > {self.strategy_b_name}): {self.probability_a_better:.1%}"
        )
        lines.append(
            f"95% Credible Interval: [{self.credible_interval_95[0]:.3f}, {self.credible_interval_95[1]:.3f}]"
        )

        # Interpretation
        if self.probability_a_better > 0.95:
            lines.append(f"\nConclusion: Strong evidence for {self.strategy_a_name}")
        elif self.probability_a_better > 0.80:
            lines.append(f"\nConclusion: Moderate evidence for {self.strategy_a_name}")
        elif self.probability_a_better < 0.05:
            lines.append(f"\nConclusion: Strong evidence for {self.strategy_b_name}")
        elif self.probability_a_better < 0.20:
            lines.append(f"\nConclusion: Moderate evidence for {self.strategy_b_name}")
        else:
            lines.append("\nConclusion: No clear winner (inconclusive)")

        return "\n".join(lines)


class PortfolioEvaluationResult(BaseResult):
    """Complete results from Module D: Portfolio Evaluation.

    Comprehensive portfolio-level analysis including:
    - Standard performance metrics
    - Bayesian strategy comparison (if applicable)
    - Time-series metrics at different frequencies
    - Detailed drawdown analysis

    Attributes:
        metrics: Standard portfolio metrics
        bayesian_comparison: Bayesian comparison results (if comparing strategies)
        time_series_metrics: Metrics aggregated by time period
        drawdown_analysis: Detailed drawdown statistics
    """

    analysis_type: str = "portfolio_evaluation"

    metrics: PortfolioMetrics = Field(..., description="Standard portfolio metrics")

    bayesian_comparison: BayesianComparisonResult | None = Field(
        None, description="Bayesian strategy comparison"
    )

    time_series_metrics: dict[str, Any] = Field(
        default_factory=dict,
        description="Metrics by period (daily, weekly, monthly)",
    )

    drawdown_analysis: dict[str, Any] = Field(
        default_factory=dict,
        description="Detailed drawdown statistics",
    )

    def get_dataframe(self, name: str | None = None) -> pl.DataFrame:
        """Get portfolio metrics as DataFrame.

        Args:
            name: 'metrics', 'comparison', or None (default: metrics)

        Returns:
            Requested DataFrame
        """
        if name == "comparison" and self.bayesian_comparison:
            return self.bayesian_comparison.get_dataframe()
        else:
            return self.metrics.get_dataframe()

    def get_time_series_dataframe(self, frequency: str = "daily") -> pl.DataFrame:
        """Get time-series metrics as DataFrame.

        Args:
            frequency: 'daily', 'weekly', or 'monthly'

        Returns:
            DataFrame with time-series metrics
        """
        if frequency not in self.time_series_metrics:
            return pl.DataFrame()

        data = self.time_series_metrics[frequency]
        return pl.DataFrame(data)

    def summary(self) -> str:
        """Human-readable summary of portfolio evaluation."""
        lines = ["Portfolio Evaluation Summary", "=" * 40]
        lines.append("")
        lines.append(self.metrics.summary())

        if self.bayesian_comparison:
            lines.append("")
            lines.append(self.bayesian_comparison.summary())

        if self.drawdown_analysis:
            lines.append("")
            lines.append("Drawdown Analysis")
            lines.append("-" * 40)
            dd = self.drawdown_analysis
            if "max_duration_days" in dd:
                lines.append(f"Max drawdown duration: {dd['max_duration_days']} days")
            if "avg_drawdown" in dd:
                lines.append(f"Average drawdown: {dd['avg_drawdown']:.2%}")
            if "num_drawdowns" in dd:
                lines.append(f"Number of drawdowns: {dd['num_drawdowns']}")

        return "\n".join(lines)
