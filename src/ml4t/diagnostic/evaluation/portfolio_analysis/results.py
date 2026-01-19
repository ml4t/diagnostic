"""Result classes for portfolio analysis.

This module provides dataclasses for portfolio analysis results:
- PortfolioMetrics: Complete portfolio performance metrics
- RollingMetricsResult: Rolling metrics over multiple windows
- DrawdownPeriod: Individual drawdown period details
- DrawdownResult: Detailed drawdown analysis
- DistributionResult: Returns distribution analysis
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import polars as pl


@dataclass
class PortfolioMetrics:
    """Complete portfolio performance metrics.

    Comprehensive set of risk-adjusted return metrics combining
    pyfolio's perf_stats with empyrical's additional metrics.

    Attributes:
        # Returns
        total_return: Cumulative return over entire period
        annual_return: Annualized return (CAGR)
        annual_volatility: Annualized standard deviation

        # Risk-adjusted
        sharpe_ratio: Excess return / volatility
        sortino_ratio: Excess return / downside deviation
        calmar_ratio: Annual return / max drawdown
        omega_ratio: P(gain) weighted gain / P(loss) weighted loss
        tail_ratio: 95th percentile / abs(5th percentile)

        # Drawdown
        max_drawdown: Maximum peak-to-trough decline

        # Distribution
        skewness: Return distribution skewness
        kurtosis: Return distribution excess kurtosis

        # Risk
        var_95: 95% Value at Risk (daily)
        cvar_95: 95% Conditional VaR (expected shortfall)

        # Stability
        stability: R² of cumulative returns vs time

        # Win/loss
        win_rate: Fraction of positive return periods
        profit_factor: Gross profit / gross loss

        # Benchmark-relative (if benchmark provided)
        alpha: Jensen's alpha (annualized)
        beta: Market beta (CAPM)
        information_ratio: Alpha / tracking error
        up_capture: Performance in up markets vs benchmark
        down_capture: Performance in down markets vs benchmark
    """

    # Returns
    total_return: float
    annual_return: float
    annual_volatility: float

    # Risk-adjusted
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    omega_ratio: float
    tail_ratio: float

    # Drawdown
    max_drawdown: float

    # Distribution
    skewness: float
    kurtosis: float

    # Risk
    var_95: float
    cvar_95: float

    # Stability
    stability: float

    # Win/loss
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float

    # Benchmark-relative (optional)
    alpha: float | None = None
    beta: float | None = None
    information_ratio: float | None = None
    up_capture: float | None = None
    down_capture: float | None = None

    def to_dict(self) -> dict[str, float | None]:
        """Convert to dictionary."""
        return {
            "total_return": self.total_return,
            "annual_return": self.annual_return,
            "annual_volatility": self.annual_volatility,
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "calmar_ratio": self.calmar_ratio,
            "omega_ratio": self.omega_ratio,
            "tail_ratio": self.tail_ratio,
            "max_drawdown": self.max_drawdown,
            "skewness": self.skewness,
            "kurtosis": self.kurtosis,
            "var_95": self.var_95,
            "cvar_95": self.cvar_95,
            "stability": self.stability,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "avg_win": self.avg_win,
            "avg_loss": self.avg_loss,
            "alpha": self.alpha,
            "beta": self.beta,
            "information_ratio": self.information_ratio,
            "up_capture": self.up_capture,
            "down_capture": self.down_capture,
        }

    def to_dataframe(self) -> pl.DataFrame:
        """Convert to single-row DataFrame."""
        data = self.to_dict()
        # Filter out None values for cleaner output
        return pl.DataFrame({k: [v] for k, v in data.items() if v is not None})

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            "=" * 50,
            "Portfolio Performance Summary",
            "=" * 50,
            "",
            "Returns",
            "-" * 30,
            f"  Total Return:        {self.total_return:>10.2%}",
            f"  Annual Return:       {self.annual_return:>10.2%}",
            f"  Annual Volatility:   {self.annual_volatility:>10.2%}",
            "",
            "Risk-Adjusted Returns",
            "-" * 30,
            f"  Sharpe Ratio:        {self.sharpe_ratio:>10.3f}",
            f"  Sortino Ratio:       {self.sortino_ratio:>10.3f}",
            f"  Calmar Ratio:        {self.calmar_ratio:>10.3f}",
            f"  Omega Ratio:         {self.omega_ratio:>10.3f}",
            f"  Tail Ratio:          {self.tail_ratio:>10.3f}",
            "",
            "Risk Metrics",
            "-" * 30,
            f"  Max Drawdown:        {self.max_drawdown:>10.2%}",
            f"  VaR (95%):           {self.var_95:>10.2%}",
            f"  CVaR (95%):          {self.cvar_95:>10.2%}",
            "",
            "Distribution",
            "-" * 30,
            f"  Skewness:            {self.skewness:>10.3f}",
            f"  Kurtosis:            {self.kurtosis:>10.3f}",
            f"  Stability (R²):      {self.stability:>10.3f}",
            "",
            "Win/Loss",
            "-" * 30,
            f"  Win Rate:            {self.win_rate:>10.2%}",
            f"  Profit Factor:       {self.profit_factor:>10.2f}",
            f"  Avg Win:             {self.avg_win:>10.2%}",
            f"  Avg Loss:            {self.avg_loss:>10.2%}",
        ]

        if self.alpha is not None:
            lines.extend(
                [
                    "",
                    "Benchmark Comparison",
                    "-" * 30,
                    f"  Alpha (annual):      {self.alpha:>10.2%}",
                    f"  Beta:                {self.beta:>10.3f}",
                    f"  Information Ratio:   {self.information_ratio:>10.3f}",
                    f"  Up Capture:          {self.up_capture:>10.2%}",
                    f"  Down Capture:        {self.down_capture:>10.2%}",
                ]
            )

        lines.append("=" * 50)
        return "\n".join(lines)


@dataclass
class RollingMetricsResult:
    """Rolling metrics over multiple windows.

    Attributes:
        windows: List of window sizes (in periods)
        sharpe: Rolling Sharpe ratio by window
        volatility: Rolling volatility by window
        returns: Rolling returns by window
        beta: Rolling beta by window (if benchmark provided)
        dates: Date index for time series
    """

    windows: list[int]
    dates: pl.Series
    sharpe: dict[int, pl.Series] = field(default_factory=dict)
    volatility: dict[int, pl.Series] = field(default_factory=dict)
    returns: dict[int, pl.Series] = field(default_factory=dict)
    beta: dict[int, pl.Series] = field(default_factory=dict)

    def to_dataframe(self, metric: str = "sharpe") -> pl.DataFrame:
        """Convert specific metric to DataFrame with all windows."""
        metric_data = getattr(self, metric, {})
        if not metric_data:
            return pl.DataFrame()

        data = {"date": self.dates}
        for window, series in metric_data.items():
            data[f"{metric}_{window}d"] = series

        return pl.DataFrame(data)


@dataclass
class DrawdownPeriod:
    """Individual drawdown period details."""

    peak_date: Any  # datetime
    valley_date: Any  # datetime
    recovery_date: Any | None  # datetime or None if not recovered
    depth: float  # Maximum depth (negative)
    duration_days: int  # Peak to valley
    recovery_days: int | None  # Valley to recovery


@dataclass
class DrawdownResult:
    """Detailed drawdown analysis.

    Attributes:
        current_drawdown: Current drawdown level
        max_drawdown: Maximum historical drawdown
        avg_drawdown: Average of all drawdowns
        underwater_curve: Drawdown at each point in time
        top_drawdowns: List of top N drawdown periods
        max_duration_days: Longest drawdown duration
        avg_duration_days: Average drawdown duration
        num_drawdowns: Total count of drawdown periods
    """

    current_drawdown: float
    max_drawdown: float
    avg_drawdown: float
    underwater_curve: pl.Series
    top_drawdowns: list[DrawdownPeriod]
    max_duration_days: int
    avg_duration_days: float
    num_drawdowns: int
    dates: pl.Series

    def to_dataframe(self) -> pl.DataFrame:
        """Convert underwater curve to DataFrame."""
        return pl.DataFrame(
            {
                "date": self.dates,
                "drawdown": self.underwater_curve,
            }
        )

    def top_drawdowns_dataframe(self) -> pl.DataFrame:
        """Convert top drawdowns to DataFrame."""
        if not self.top_drawdowns:
            return pl.DataFrame()

        return pl.DataFrame(
            {
                "peak_date": [d.peak_date for d in self.top_drawdowns],
                "valley_date": [d.valley_date for d in self.top_drawdowns],
                "recovery_date": [d.recovery_date for d in self.top_drawdowns],
                "depth": [d.depth for d in self.top_drawdowns],
                "duration_days": [d.duration_days for d in self.top_drawdowns],
                "recovery_days": [d.recovery_days for d in self.top_drawdowns],
            }
        )


@dataclass
class DistributionResult:
    """Returns distribution analysis.

    Attributes:
        mean: Mean daily return
        std: Standard deviation
        skewness: Skewness
        kurtosis: Excess kurtosis
        jarque_bera_stat: JB test statistic
        jarque_bera_pvalue: JB test p-value
        is_normal: Whether returns are approximately normal (p > 0.05)
        var_95: 95% VaR
        var_99: 99% VaR
        cvar_95: 95% CVaR
        cvar_99: 99% CVaR
        best_day: Best single day return
        worst_day: Worst single day return
    """

    mean: float
    std: float
    skewness: float
    kurtosis: float
    jarque_bera_stat: float
    jarque_bera_pvalue: float
    is_normal: bool
    var_95: float
    var_99: float
    cvar_95: float
    cvar_99: float
    best_day: float
    worst_day: float


__all__ = [
    "PortfolioMetrics",
    "RollingMetricsResult",
    "DrawdownPeriod",
    "DrawdownResult",
    "DistributionResult",
]
