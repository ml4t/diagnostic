"""PortfolioAnalysis class - modern pyfolio replacement.

This module provides the PortfolioAnalysis class for comprehensive
portfolio tear sheet generation with:
- Polars backend (10-100x faster than pandas)
- Plotly visualizations (interactive, shareable)
- Enhanced statistics (DSR, regime analysis, Bayesian comparison)
- Pyfolio feature parity plus modern enhancements

Example:
    >>> from ml4t.diagnostic.evaluation import PortfolioAnalysis
    >>>
    >>> analysis = PortfolioAnalysis(
    ...     returns=strategy_returns,
    ...     benchmark=spy_returns,
    ... )
    >>> metrics = analysis.compute_summary_stats()
    >>> print(metrics.summary())
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Union

import numpy as np
import polars as pl
from scipy import stats

from .metrics import (
    _safe_cumprod,
    _safe_prod,
    _to_numpy,
    alpha_beta,
    annual_return,
    annual_volatility,
    calmar_ratio,
    conditional_var,
    information_ratio,
    max_drawdown,
    omega_ratio,
    sharpe_ratio,
    sortino_ratio,
    stability_of_timeseries,
    tail_ratio,
    up_down_capture,
    value_at_risk,
)
from .results import (
    DistributionResult,
    DrawdownPeriod,
    DrawdownResult,
    PortfolioMetrics,
    RollingMetricsResult,
)

if TYPE_CHECKING:
    import pandas as pd

# Type aliases - use Union for Python 3.9 compatibility
ArrayLike = Union[np.ndarray, pl.Series, "list[float]"]
DataFrameLike = Union[pl.DataFrame, "pd.DataFrame"]


class PortfolioAnalysis:
    """Modern replacement for pyfolio.create_full_tear_sheet().

    Provides comprehensive portfolio analysis with:
    - Polars backend (10-100x faster than pandas)
    - Plotly visualizations (interactive, shareable)
    - Enhanced statistics (DSR, regime analysis, Bayesian comparison)
    - Full pyfolio feature parity plus modern enhancements

    Parameters
    ----------
    returns : Series
        Daily returns of the strategy (non-cumulative).
        Accepts Polars Series, Pandas Series, or numpy array.
    benchmark : Series, optional
        Benchmark returns for alpha/beta calculation (e.g., SPY).
    positions : DataFrame, optional
        Daily position values by asset.
        Columns: [date, asset, value] or pivoted with assets as columns.
    transactions : DataFrame, optional
        Trade execution records.
        Columns: [date, asset, quantity, price, commission]
    risk_free : float, default 0.0
        Annual risk-free rate for Sharpe/Sortino calculation.
    periods_per_year : int, default 252
        Trading periods per year (252 for daily data).

    Examples
    --------
    >>> # Basic usage
    >>> analysis = PortfolioAnalysis(returns=daily_returns)
    >>> metrics = analysis.compute_summary_stats()
    >>> print(metrics.summary())

    >>> # With benchmark
    >>> analysis = PortfolioAnalysis(
    ...     returns=strategy_returns,
    ...     benchmark=spy_returns,
    ... )
    >>> metrics = analysis.compute_summary_stats()
    >>> print(f"Alpha: {metrics.alpha:.2%}")
    >>> print(f"Beta: {metrics.beta:.2f}")

    >>> # With positions and transactions
    >>> analysis = PortfolioAnalysis(
    ...     returns=strategy_returns,
    ...     positions=position_df,
    ...     transactions=trades_df,
    ... )
    >>> tear_sheet = analysis.create_tear_sheet()
    >>> tear_sheet.save_html("report.html")
    """

    def __init__(
        self,
        returns: ArrayLike | pl.Series,
        benchmark: ArrayLike | pl.Series | None = None,
        positions: DataFrameLike | None = None,
        transactions: DataFrameLike | None = None,
        dates: ArrayLike | pl.Series | None = None,
        risk_free: float = 0.0,
        periods_per_year: int = 252,
    ):
        # Convert returns to numpy
        self._returns = _to_numpy(returns)

        # Handle dates
        if dates is not None:
            if isinstance(dates, pl.Series):
                self._dates = dates
            else:
                self._dates = pl.Series("date", dates)
        else:
            # Generate synthetic dates
            self._dates = pl.Series(
                "date",
                pl.date_range(
                    pl.date(2000, 1, 1),
                    pl.date(2000, 1, 1) + pl.duration(days=len(self._returns) - 1),
                    eager=True,
                ),
            )

        # Convert benchmark if provided
        self._benchmark = _to_numpy(benchmark) if benchmark is not None else None

        # Store positions and transactions (convert to Polars if needed)
        self._positions = self._to_polars_df(positions) if positions is not None else None
        self._transactions = self._to_polars_df(transactions) if transactions is not None else None

        # Configuration
        self._risk_free = risk_free
        self._periods_per_year = periods_per_year

        # Cached results
        self._metrics_cache: PortfolioMetrics | None = None
        self._rolling_cache: dict[tuple, RollingMetricsResult] = {}
        self._drawdown_cache: DrawdownResult | None = None

    @staticmethod
    def _to_polars_df(df: DataFrameLike | None) -> pl.DataFrame | None:
        """Convert DataFrame to Polars."""
        if df is None:
            return None
        if isinstance(df, pl.DataFrame):
            return df
        # Assume pandas DataFrame
        return pl.from_pandas(df)

    @property
    def returns(self) -> np.ndarray:
        """Get returns as numpy array."""
        return self._returns

    @property
    def dates(self) -> pl.Series:
        """Get dates as Polars Series."""
        return self._dates

    @property
    def benchmark(self) -> np.ndarray | None:
        """Get benchmark returns as numpy array."""
        return self._benchmark

    @property
    def has_benchmark(self) -> bool:
        """Check if benchmark was provided."""
        return self._benchmark is not None

    @property
    def has_positions(self) -> bool:
        """Check if positions data was provided."""
        return self._positions is not None

    @property
    def has_transactions(self) -> bool:
        """Check if transactions data was provided."""
        return self._transactions is not None

    # =========================================================================
    # Core Metric Methods
    # =========================================================================

    def compute_summary_stats(self, force_recompute: bool = False) -> PortfolioMetrics:
        """Compute all standard portfolio metrics.

        This is the main method for getting performance statistics,
        equivalent to pyfolio's perf_stats output.

        Parameters
        ----------
        force_recompute : bool, default False
            Force recomputation even if cached

        Returns
        -------
        PortfolioMetrics
            Complete set of portfolio metrics

        Examples
        --------
        >>> metrics = analysis.compute_summary_stats()
        >>> print(f"Sharpe: {metrics.sharpe_ratio:.2f}")
        >>> print(f"Max Drawdown: {metrics.max_drawdown:.1%}")
        """
        if self._metrics_cache is not None and not force_recompute:
            return self._metrics_cache

        returns = self._returns
        rf = self._risk_free
        ppy = self._periods_per_year

        # Basic returns
        total_ret = float(_safe_prod(1 + returns) - 1)
        ann_ret = annual_return(returns, ppy)
        ann_vol = annual_volatility(returns, ppy)

        # Risk-adjusted
        sr = sharpe_ratio(returns, rf, ppy)
        sortino = sortino_ratio(returns, rf, ppy)
        calmar = calmar_ratio(returns, ppy)
        omega = omega_ratio(returns)
        tail = tail_ratio(returns)

        # Drawdown
        max_dd = max_drawdown(returns)

        # Distribution
        skew = float(stats.skew(returns[~np.isnan(returns)]))
        kurt = float(stats.kurtosis(returns[~np.isnan(returns)]))

        # Risk
        var95 = value_at_risk(returns, 0.95)
        cvar95 = conditional_var(returns, 0.95)

        # Stability
        stability = stability_of_timeseries(returns)

        # Win/loss
        wins = returns[returns > 0]
        losses = returns[returns < 0]

        win_rate = len(wins) / len(returns) if len(returns) > 0 else np.nan
        avg_win = float(np.mean(wins)) if len(wins) > 0 else 0.0
        avg_loss = float(np.mean(losses)) if len(losses) > 0 else 0.0

        gross_profit = float(np.sum(wins)) if len(wins) > 0 else 0.0
        gross_loss = float(abs(np.sum(losses))) if len(losses) > 0 else 0.0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf

        # Benchmark metrics
        alpha_val = beta_val = ir = up_cap = down_cap = None

        if self.has_benchmark and self._benchmark is not None:
            alpha_val, beta_val = alpha_beta(returns, self._benchmark, rf, ppy)
            ir = information_ratio(returns, self._benchmark, ppy)
            up_cap, down_cap = up_down_capture(returns, self._benchmark)

        self._metrics_cache = PortfolioMetrics(
            total_return=total_ret,
            annual_return=ann_ret,
            annual_volatility=ann_vol,
            sharpe_ratio=sr,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            omega_ratio=omega,
            tail_ratio=tail,
            max_drawdown=max_dd,
            skewness=skew,
            kurtosis=kurt,
            var_95=var95,
            cvar_95=cvar95,
            stability=stability,
            win_rate=win_rate,
            profit_factor=profit_factor,
            avg_win=avg_win,
            avg_loss=avg_loss,
            alpha=alpha_val,
            beta=beta_val,
            information_ratio=ir,
            up_capture=up_cap,
            down_capture=down_cap,
        )

        return self._metrics_cache

    def compute_rolling_metrics(
        self,
        windows: list[int] | None = None,
        metrics: list[str] | None = None,
    ) -> RollingMetricsResult:
        """Compute rolling metrics over multiple windows.

        Parameters
        ----------
        windows : list[int], optional
            Window sizes in periods. Default [21, 63, 126, 252].
        metrics : list[str], optional
            Which metrics to compute. Default ["sharpe", "volatility", "returns"].

        Returns
        -------
        RollingMetricsResult
            Rolling metrics for each window

        Examples
        --------
        >>> rolling = analysis.compute_rolling_metrics(windows=[21, 63, 252])
        >>> sharpe_df = rolling.to_dataframe("sharpe")
        """
        if windows is None:
            windows = [21, 63, 126, 252]

        if metrics is None:
            metrics = ["sharpe", "volatility", "returns"]

        cache_key = (tuple(windows), tuple(metrics))
        if cache_key in self._rolling_cache:
            return self._rolling_cache[cache_key]

        returns = self._returns
        rf = self._risk_free
        ppy = self._periods_per_year

        result = RollingMetricsResult(windows=windows, dates=self._dates)

        for window in windows:
            if "sharpe" in metrics:
                rolling_sharpe = self._rolling_sharpe(returns, window, rf, ppy)
                result.sharpe[window] = pl.Series(f"sharpe_{window}d", rolling_sharpe).fill_nan(
                    None
                )

            if "volatility" in metrics:
                rolling_vol = self._rolling_volatility(returns, window, ppy)
                result.volatility[window] = pl.Series(f"vol_{window}d", rolling_vol).fill_nan(None)

            if "returns" in metrics:
                rolling_ret = self._rolling_returns(returns, window)
                result.returns[window] = pl.Series(f"ret_{window}d", rolling_ret).fill_nan(None)

            if "beta" in metrics and self.has_benchmark and self._benchmark is not None:
                rolling_beta = self._rolling_beta(returns, self._benchmark, window)
                result.beta[window] = pl.Series(f"beta_{window}d", rolling_beta).fill_nan(None)

        self._rolling_cache[cache_key] = result
        return result

    @staticmethod
    def _rolling_sharpe(
        returns: np.ndarray,
        window: int,
        risk_free: float,
        periods_per_year: int,
    ) -> np.ndarray:
        """Compute rolling Sharpe ratio using vectorized sliding_window_view."""
        from numpy.lib.stride_tricks import sliding_window_view

        n = len(returns)
        result = np.full(n, np.nan)

        if n < window:
            return result

        daily_rf = (1 + risk_free) ** (1 / periods_per_year) - 1

        # Vectorized: create all windows at once
        windows = sliding_window_view(returns, window)
        excess = windows - daily_rf

        # Compute mean and std across each window (axis=1)
        mu = np.mean(excess, axis=1)
        sd = np.std(excess, axis=1, ddof=1)

        # Sharpe where std > 0
        sharpe = np.where(sd > 0, (mu / sd) * np.sqrt(periods_per_year), np.nan)
        result[window - 1 :] = sharpe

        return result

    @staticmethod
    def _rolling_volatility(
        returns: np.ndarray,
        window: int,
        periods_per_year: int,
    ) -> np.ndarray:
        """Compute rolling annualized volatility using vectorized sliding_window_view."""
        from numpy.lib.stride_tricks import sliding_window_view

        n = len(returns)
        result = np.full(n, np.nan)

        if n < window:
            return result

        # Vectorized: create all windows at once
        windows = sliding_window_view(returns, window)
        sd = np.std(windows, axis=1, ddof=1)
        result[window - 1 :] = sd * np.sqrt(periods_per_year)

        return result

    @staticmethod
    def _rolling_returns(
        returns: np.ndarray,
        window: int,
    ) -> np.ndarray:
        """Compute rolling cumulative returns using O(n) log1p cumsum."""
        n = len(returns)
        result = np.full(n, np.nan)

        if n < window:
            return result

        # Vectorized O(n): use log1p cumsum for compound returns
        # Requires returns > -1 (valid for typical financial returns)
        # Clip to prevent log of non-positive numbers
        safe_returns = np.clip(returns, -0.9999, None)
        log_returns = np.log1p(safe_returns)
        cumsum = np.concatenate(([0.0], np.cumsum(log_returns)))

        # Rolling sum of log returns = log(compound return + 1)
        window_log_sum = cumsum[window:] - cumsum[:-window]
        result[window - 1 :] = np.expm1(window_log_sum)

        return result

    @staticmethod
    def _rolling_beta(
        returns: np.ndarray,
        benchmark: np.ndarray,
        window: int,
    ) -> np.ndarray:
        """Compute rolling beta using vectorized sliding_window_view."""
        from numpy.lib.stride_tricks import sliding_window_view

        n = len(returns)
        result = np.full(n, np.nan)

        if n < window:
            return result

        # Vectorized: create all windows at once
        ret_windows = sliding_window_view(returns, window)
        bench_windows = sliding_window_view(benchmark, window)

        # Compute means
        ret_mean = np.mean(ret_windows, axis=1, keepdims=True)
        bench_mean = np.mean(bench_windows, axis=1, keepdims=True)

        # Deviations from mean
        ret_dev = ret_windows - ret_mean
        bench_dev = bench_windows - bench_mean

        # Covariance and variance (using ddof=1 for sample variance)
        cov = np.sum(ret_dev * bench_dev, axis=1) / (window - 1)
        var = np.sum(bench_dev * bench_dev, axis=1) / (window - 1)

        # Beta = cov / var where var > 0
        beta = np.where(var > 0, cov / var, np.nan)
        result[window - 1 :] = beta

        return result

    def compute_drawdown_analysis(
        self,
        top_n: int = 5,
        threshold: float = 0.01,
    ) -> DrawdownResult:
        """Compute detailed drawdown analysis.

        Parameters
        ----------
        top_n : int, default 5
            Number of top drawdowns to identify
        threshold : float, default 0.01
            Minimum drawdown depth to count (1%)

        Returns
        -------
        DrawdownResult
            Detailed drawdown statistics

        Examples
        --------
        >>> dd = analysis.compute_drawdown_analysis(top_n=10)
        >>> print(f"Max drawdown: {dd.max_drawdown:.1%}")
        >>> print(f"Avg duration: {dd.avg_duration_days:.0f} days")
        """
        if self._drawdown_cache is not None:
            return self._drawdown_cache

        returns = self._returns
        dates = self._dates

        # Compute cumulative returns and running max
        cum_returns = _safe_cumprod(1 + returns)
        running_max = np.maximum.accumulate(cum_returns)

        # Underwater curve
        underwater = (cum_returns - running_max) / running_max

        # Identify drawdown periods
        drawdown_periods = self._identify_drawdown_periods(underwater, dates, threshold)

        # Sort by depth and take top N
        drawdown_periods.sort(key=lambda x: x.depth)
        top_drawdowns = drawdown_periods[:top_n]

        # Statistics
        current_dd = float(underwater[-1]) if len(underwater) > 0 else 0.0
        max_dd = float(np.min(underwater))
        avg_dd = (
            float(np.mean(underwater[underwater < -threshold]))
            if np.any(underwater < -threshold)
            else 0.0
        )

        durations = [p.duration_days for p in drawdown_periods if p.duration_days > 0]
        max_duration = max(durations) if durations else 0
        avg_duration = float(np.mean(durations)) if durations else 0.0

        self._drawdown_cache = DrawdownResult(
            current_drawdown=current_dd,
            max_drawdown=max_dd,
            avg_drawdown=avg_dd,
            underwater_curve=pl.Series("drawdown", underwater),
            top_drawdowns=top_drawdowns,
            max_duration_days=max_duration,
            avg_duration_days=avg_duration,
            num_drawdowns=len(drawdown_periods),
            dates=dates,
        )

        return self._drawdown_cache

    def _identify_drawdown_periods(
        self,
        underwater: np.ndarray,
        dates: pl.Series,
        threshold: float,
    ) -> list[DrawdownPeriod]:
        """Identify individual drawdown periods."""
        periods = []

        in_drawdown = False
        peak_idx = 0
        valley_idx = 0
        valley_depth = 0.0

        for i, dd in enumerate(underwater):
            if dd < -threshold and not in_drawdown:
                # Start of drawdown
                in_drawdown = True
                peak_idx = i - 1 if i > 0 else 0
                valley_idx = i
                valley_depth = dd
            elif in_drawdown:
                if dd < valley_depth:
                    # New valley
                    valley_idx = i
                    valley_depth = dd
                elif dd >= 0:
                    # Recovery
                    period = DrawdownPeriod(
                        peak_date=dates[peak_idx],
                        valley_date=dates[valley_idx],
                        recovery_date=dates[i],
                        depth=valley_depth,
                        duration_days=valley_idx - peak_idx,
                        recovery_days=i - valley_idx,
                    )
                    periods.append(period)
                    in_drawdown = False

        # Handle ongoing drawdown
        if in_drawdown:
            period = DrawdownPeriod(
                peak_date=dates[peak_idx],
                valley_date=dates[valley_idx],
                recovery_date=None,
                depth=valley_depth,
                duration_days=valley_idx - peak_idx,
                recovery_days=None,
            )
            periods.append(period)

        return periods

    def compute_returns_distribution(self) -> DistributionResult:
        """Compute returns distribution analysis.

        Returns
        -------
        DistributionResult
            Distribution statistics and normality tests
        """
        returns = self._returns
        clean_returns = returns[~np.isnan(returns)]

        # Moments
        mean = float(np.mean(clean_returns))
        std = float(np.std(clean_returns, ddof=1))
        skew = float(stats.skew(clean_returns))
        kurt = float(stats.kurtosis(clean_returns))

        # Jarque-Bera test
        jb_stat, jb_pval = stats.jarque_bera(clean_returns)

        # VaR/CVaR
        var95 = value_at_risk(returns, 0.95)
        var99 = value_at_risk(returns, 0.99)
        cvar95 = conditional_var(returns, 0.95)
        cvar99 = conditional_var(returns, 0.99)

        return DistributionResult(
            mean=mean,
            std=std,
            skewness=skew,
            kurtosis=kurt,
            jarque_bera_stat=float(jb_stat),
            jarque_bera_pvalue=float(jb_pval),
            is_normal=jb_pval > 0.05,
            var_95=var95,
            var_99=var99,
            cvar_95=cvar95,
            cvar_99=cvar99,
            best_day=float(np.max(clean_returns)),
            worst_day=float(np.min(clean_returns)),
        )

    # =========================================================================
    # Monthly / Annual Returns
    # =========================================================================

    def compute_monthly_returns(self) -> pl.DataFrame:
        """Compute monthly returns.

        Returns
        -------
        pl.DataFrame
            Monthly returns with year and month columns
        """
        df = pl.DataFrame(
            {
                "date": self._dates,
                "return": self._returns,
            }
        )

        # Group by year-month and compound
        monthly = (
            df.with_columns(
                [
                    pl.col("date").dt.year().alias("year"),
                    pl.col("date").dt.month().alias("month"),
                ]
            )
            .group_by(["year", "month"])
            .agg((1 + pl.col("return")).product().alias("monthly_return") - 1)
            .sort(["year", "month"])
        )

        return monthly

    def compute_annual_returns(self) -> pl.DataFrame:
        """Compute annual returns.

        Returns
        -------
        pl.DataFrame
            Annual returns with year column
        """
        df = pl.DataFrame(
            {
                "date": self._dates,
                "return": self._returns,
            }
        )

        # Group by year and compound
        annual = (
            df.with_columns(
                [
                    pl.col("date").dt.year().alias("year"),
                ]
            )
            .group_by("year")
            .agg((1 + pl.col("return")).product().alias("annual_return") - 1)
            .sort("year")
        )

        return annual

    def get_monthly_returns_matrix(self) -> pl.DataFrame:
        """Get monthly returns as year x month matrix (for heatmap).

        Returns
        -------
        pl.DataFrame
            Pivoted DataFrame with years as rows, months as columns
        """
        monthly = self.compute_monthly_returns()

        # Pivot to matrix form
        return monthly.pivot(
            values="monthly_return",
            index="year",
            on="month",
        ).sort("year")

    def create_tear_sheet(
        self,
        theme: str | None = None,
        cost_info: dict[str, str] | None = None,
    ):
        """Create comprehensive portfolio tear sheet.

        Convenience method delegating to create_portfolio_dashboard().

        Parameters
        ----------
        theme : str, optional
            Plot theme ("default", "dark", "print", "presentation")
        cost_info : dict[str, str] | None
            Trading cost labels for display

        Returns
        -------
        PortfolioTearSheet
            Complete tear sheet with figures and metrics
        """
        from ml4t.diagnostic.visualization.portfolio import create_portfolio_dashboard

        return create_portfolio_dashboard(self, theme=theme, cost_info=cost_info)


__all__ = ["PortfolioAnalysis"]
