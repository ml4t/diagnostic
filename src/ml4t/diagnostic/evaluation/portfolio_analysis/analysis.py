"""PortfolioAnalysis class for portfolio diagnostics.

This module provides the PortfolioAnalysis class for comprehensive
portfolio tear sheet generation with:
- Polars-native data handling
- Plotly visualizations (interactive, shareable)
- Enhanced statistics (DSR, regime analysis, Bayesian comparison)

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

import json
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import replace
from inspect import signature
from numbers import Integral, Real
from typing import TYPE_CHECKING, Any, Literal, Union

import numpy as np
import polars as pl
from ml4t.specs import UNSET_CALENDAR, CalendarUnset, FeedSpec
from scipy import stats

from ml4t.diagnostic.metrics import sharpe_ratio, sortino_ratio
from ml4t.diagnostic.metrics.risk_adjusted import _drawdown_from_wealth

from .fifo import RoundTripResult, reconstruct_round_trips
from .frame_inputs import PortfolioInputData, normalize_portfolio_frames
from .holdings import HoldingsAnalysisResult, analyze_holdings, position_snapshot
from .holdings_sectors import SectorMapping, analyze_sector_exposures
from .inputs import (
    annual_risk_free,
    initial_valuation_time,
    normalize_observations,
    positive_integer,
)
from .metrics import (
    _safe_cumprod,
    _safe_prod,
    alpha_beta,
    annual_return,
    annual_volatility,
    calmar_ratio,
    conditional_var,
    information_ratio,
    max_drawdown,
    omega_ratio,
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
from .transactions import TransactionAnalysisResult, analyze_transactions

if TYPE_CHECKING:
    from datetime import date, datetime

    import pandas as pd

# Type aliases - use Union for Python 3.9 compatibility
ArrayLike = Union[np.ndarray, pl.Series, "pd.Series", "list[float]"]
DataFrameLike = Union[pl.DataFrame, "pd.DataFrame"]


class PortfolioAnalysis:
    """Portfolio diagnostics and reporting workflow.

    Provides comprehensive portfolio analysis with:
    - Polars-native data handling
    - Plotly visualizations (interactive, shareable)
    - Enhanced statistics (DSR, regime analysis, Bayesian comparison)

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
    dates : date-like sequence, optional
        Unique increasing dates, inferred from a pandas DatetimeIndex when present.
        Explicit dates must agree with that index. Without dates, numerical and
        rolling outputs use zero-based observation indices; calendar tables and
        complete calendar dashboards require genuine dates.
    initial_timestamp : datetime or date, optional
        Time of the initial valuation, before the first dated return. If omitted,
        a drawdown starting at initial capital has an unavailable peak timestamp.

    Notes
    -----
    Inputs are owned snapshots of nonempty, one-dimensional, finite simple returns
    no lower than -1. Missing returns are rejected, not replaced with zeros.
    Dated benchmarks are aligned to every strategy observation without filling;
    extra benchmark dates are excluded. Undated benchmarks must match in length.
    Public arrays and cached results are returned as independent copies.

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
        dates: pl.Series | np.ndarray | pd.DatetimeIndex | list[date | datetime] | None = None,
        risk_free: float = 0.0,
        periods_per_year: int = 252,
        *,
        initial_timestamp: date | datetime | None = None,
    ):
        scale = positive_integer(periods_per_year, "periods_per_year")
        annual_rf = annual_risk_free(risk_free)
        owned_returns, owned_benchmark, owned_dates = normalize_observations(
            returns, benchmark, dates
        )
        self._returns = owned_returns
        self._benchmark = owned_benchmark
        self._dates = owned_dates
        self._initial_timestamp = initial_valuation_time(initial_timestamp, owned_dates)
        self._input_data: PortfolioInputData | None = None
        self._positions = position_snapshot(positions) if positions is not None else None
        self._transactions = self._to_polars_df(transactions) if transactions is not None else None

        self._risk_free = annual_rf
        self._periods_per_year = scale

        # Cached results
        self._metrics_cache: PortfolioMetrics | None = None
        self._rolling_cache: dict[tuple, RollingMetricsResult] = {}
        self._drawdown_cache: dict[tuple[int, float], DrawdownResult] = {}

    @classmethod
    def from_frames(
        cls,
        returns: DataFrameLike | pd.Series,
        *,
        benchmark: DataFrameLike | pd.Series | None = None,
        positions: DataFrameLike | None = None,
        valuations: DataFrameLike | None = None,
        transactions: DataFrameLike | None = None,
        complete_snapshots: bool = False,
        snapshot_kind: Literal["observed", "end_of_day", "intraday"] = "observed",
        positions_format: Literal["auto", "long", "wide"] = "auto",
        feed_spec: FeedSpec | Mapping[str, Any] | None = None,
        timestamp_col: str | None = None,
        entity_col: str | None = None,
        return_col: str = "return",
        benchmark_col: str = "return",
        transaction_timestamp_col: str | None = None,
        transaction_multiplier: float = 1.0,
        calendar: str | CalendarUnset | None = UNSET_CALENDAR,
        periods_per_year: int = 252,
        risk_free: float = 0.0,
        initial_timestamp: date | datetime | None = None,
    ) -> PortfolioAnalysis:
        """Normalize dated supplied account frames without a producer dependency.

        Returns are authoritative net simple returns, not inferred from NAV.
        Pandas Series dates and benchmark labels are retained/aligned by key.
        Positions are signed cash values, long or wide; the wide cash column is
        reported cash, never an asset or inferred NAV. Missing optional records
        stay absent and incomplete holdings are not filled. Commission/slippage
        absence means unknown. Quantity is signed; an optional buy/sell side is
        reconciled before accounting. Multiplier defaults to one cash-price unit
        per quantity unit; futures must supply their actual multiplier.

        Calendar declarations are preserved, not used to resample input returns
        or infer schedule completeness. Explicit None disables schedules. Use
        transaction_timestamp_col when fills and snapshots have different keys.
        Existing constructor annualization/risk-free defaults remain unchanged.
        """
        periods_per_year = positive_integer(periods_per_year, "periods_per_year")
        risk_free = annual_risk_free(risk_free)
        inputs = normalize_portfolio_frames(
            returns,
            benchmark=benchmark,
            positions=positions,
            valuations=valuations,
            transactions=transactions,
            feed_spec=feed_spec,
            timestamp_col=timestamp_col,
            entity_col=entity_col,
            return_col=return_col,
            benchmark_col=benchmark_col,
            transaction_timestamp_col=transaction_timestamp_col,
            transaction_multiplier=transaction_multiplier,
            calendar=calendar,
            complete_snapshots=complete_snapshots,
            snapshot_kind=snapshot_kind,
            positions_format=positions_format,
        )
        observed, relative = inputs.returns, inputs.benchmark
        instance = cls(
            returns=observed["return"],
            dates=observed["date"],
            benchmark=(
                relative.to_pandas().set_index("date")["return"]
                if "date" in relative.columns
                else relative["return"]
            )
            if relative is not None
            else None,
            positions=inputs.positions,
            transactions=inputs.transactions,
            risk_free=risk_free,
            periods_per_year=periods_per_year,
            initial_timestamp=initial_timestamp,
        )
        metadata = inputs.metadata
        metadata.update(periods_per_year=periods_per_year, risk_free_annual=risk_free)
        instance._input_data = replace(inputs, _metadata=json.dumps(metadata, allow_nan=False))
        return instance

    @property
    def input_data(self) -> PortfolioInputData | None:
        """Return immutable supplied-frame provenance, or None for the array constructor."""
        return self._input_data

    def compute_transaction_analysis(
        self,
        *,
        turnover_denom: str = "beginning_nav",
        turnover_method: str = "gross",
        incremental_bps: Sequence[float] = (0.0, 10.0),
    ) -> TransactionAnalysisResult:
        """Analyze actual supplied fills, explicit turnover and extra cash-bps costs.

        Requires from_frames inputs. Named calendars use the declared FeedSpec
        timestamp semantics and aggregate to exchange sessions; calendar=None
        matches exact observation keys without calendar resampling. Missing NAV
        retains raw notional/cost dollars but cannot define a nonzero cost return.
        """
        if self._input_data is None:
            raise ValueError("transaction analysis requires owned from_frames inputs")
        return analyze_transactions(
            self._input_data,
            turnover_denom=turnover_denom,
            turnover_method=turnover_method,
            incremental_bps=incremental_bps,
        )

    def compute_round_trips(
        self,
        *,
        initial_state: Literal["flat", "supplied"] | None = None,
        opening_lots: DataFrameLike | None = None,
        opening_as_of: date | datetime | str | None = None,
        marks: DataFrameLike | None = None,
        as_of: date | datetime | str | None = None,
        synthetic_close: bool = False,
    ) -> RoundTripResult:
        """Match actual fractional FIFO lots from an explicitly complete opening book.

        Supplied opening rows require asset/quantity and opening_as_of; optional
        lot_id/entry_timestamp/entry_price/multiplier/remaining_commission retain
        known cost/time, never guessed zero basis. Supplied row order is FIFO.
        Marks require timestamp/asset/price and an explicit as_of, which also cuts
        future executions. Only quotes known at/before as_of are used, with age.
        Optional synthetic closes are separate hypothetical segments, not actual
        realizations or changes to open inventory. Supplied returns stay unchanged.
        """
        if self._input_data is None:
            raise ValueError("FIFO requires owned from_frames execution inputs")
        result = reconstruct_round_trips(
            self._input_data.transactions,
            initial_state=initial_state,
            opening_lots=opening_lots,
            opening_as_of=opening_as_of,
            marks=marks,
            as_of=as_of,
            synthetic_close=synthetic_close,
            timezone=self._input_data.metadata["feed_spec"].get("timezone"),
        )
        declarations = self._input_data.metadata
        for name in ("feed_spec", "calendar", "calendar_source", "calendar_argument"):
            if name in declarations:
                result.metadata[name] = declarations[name]
        return result

    @staticmethod
    def _to_polars_df(df: DataFrameLike | None) -> pl.DataFrame | None:
        """Convert DataFrame to Polars."""
        if df is None:
            return None
        if isinstance(df, pl.DataFrame):
            return df.clone()
        # Assume pandas DataFrame
        return pl.from_pandas(df)

    @property
    def returns(self) -> np.ndarray:
        """Get an independent copy of the strategy return snapshot."""
        return self._returns.copy()

    @property
    def dates(self) -> pl.Series:
        """Get dates or, for undated input, zero-based observation indices."""
        return self._dates.clone()

    @property
    def has_dates(self) -> bool:
        """Whether actual dates were supplied or retained from labeled returns."""
        return self._dates.dtype.is_temporal()

    def require_dates(self) -> None:
        """Reject calendar-based outputs when only observation order is known."""
        if not self.has_dates:
            raise ValueError(
                "dates are required for calendar tables and complete portfolio reports"
            )

    @property
    def benchmark(self) -> np.ndarray | None:
        """Get an independent copy of the aligned benchmark snapshot."""
        return self._benchmark.copy() if self._benchmark is not None else None

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

    def compute_holdings_analysis(
        self,
        valuations: DataFrameLike | None = None,
        *,
        complete_snapshots: bool | None = None,
        snapshot_kind: Literal["observed", "end_of_day", "intraday"] | None = None,
        sector_mapping: SectorMapping | None = None,
    ) -> HoldingsAnalysisResult:
        """Analyze complete holdings on exact supplied return observation keys.

        Positions are signed market values in long date/asset/value format or
        wide string asset columns. Pandas DatetimeIndex labels are retained.
        Valuations supply date and optional reported equity/cash, in the same
        currency as holdings. Missing NAV leaves NAV ratios unavailable; NAV is
        never inferred from cash plus holdings. Missing position marks fail.

        Explicit completeness declares absent assets zero within a snapshot;
        every observation still requires a dated snapshot. No calendar grouping,
        forward fill, intraday reconstruction or instrument valuation is done.
        Snapshot kind is caller-declared provenance, not an inferred schedule.
        The result and its tables are independent of inputs and previous calls.

        Optional sector_mapping accepts a static mapping/Series or asset/sector
        table with optional exact date keys. Unknown labels remain a null-sector
        group. Dated labels are not filled between observations. Account cash
        stays in timeline, separate from any sector literally named "cash".
        """
        self.require_dates()
        if self._positions is None:
            raise ValueError("positions are required for holdings analysis")
        metadata = self._input_data.metadata if self._input_data is not None else {}
        if complete_snapshots is None:
            complete_snapshots = metadata.get("complete_snapshots", False)
        if snapshot_kind is None:
            snapshot_kind = metadata.get("snapshot_kind", "observed")
        if valuations is None:
            valuations = self._input_data.valuations if self._input_data is not None else None
            if valuations is not None:
                from ml4t.diagnostic.utils.validation import _align_datetime_key_resolution

                valuations, keys = _align_datetime_key_resolution(
                    valuations, pl.DataFrame({"date": self._dates}), "date"
                )
                valuations = valuations.join(keys, on="date", how="semi").sort("date")
            else:
                valuations = pl.DataFrame({"date": self._dates})
        result = analyze_holdings(
            self._positions,
            valuations,
            self._dates,
            complete_snapshots=complete_snapshots,
            snapshot_kind=snapshot_kind,
        )
        if sector_mapping is not None:
            result.positions, result.sectors = analyze_sector_exposures(
                result.positions, result.timeline, sector_mapping
            )
        return result

    # =========================================================================
    # Core Metric Methods
    # =========================================================================

    def compute_summary_stats(self, force_recompute: bool = False) -> PortfolioMetrics:
        """Compute all standard portfolio metrics.

        This is the main method for return-based performance statistics.
        Match each metric's definition and input conventions before comparison.

        Profit factor uses positive and negative period-return sums, without
        annualization. No gains or losses yields NaN; gains without losses
        yields infinity. It does not use trade-lifecycle cash PnL.

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
            return deepcopy(self._metrics_cache)

        returns = self._returns
        rf = self._risk_free
        ppy = self._periods_per_year

        # Basic returns
        total_ret = float(_safe_prod(1 + returns) - 1)
        ann_ret = annual_return(returns, ppy)
        ann_vol = annual_volatility(returns, ppy) if len(returns) > 1 else np.nan

        # Risk-adjusted
        sr = sharpe_ratio(returns, rf, ppy)
        sortino = sortino_ratio(returns, rf, ppy)
        calmar = calmar_ratio(returns, ppy)
        omega = omega_ratio(returns)
        tail = tail_ratio(returns)

        # Drawdown
        max_dd = max_drawdown(returns)

        # Distribution
        has_dispersion = len(returns) > 1 and np.ptp(returns) > 0
        skew = float(stats.skew(returns)) if has_dispersion else np.nan
        kurt = float(stats.kurtosis(returns)) if has_dispersion else np.nan

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
        if gross_loss > 0:
            profit_factor = gross_profit / gross_loss
        elif gross_profit > 0:
            profit_factor = np.inf
        else:
            profit_factor = np.nan

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

        return deepcopy(self._metrics_cache)

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

        if not windows:
            raise ValueError("windows must contain positive integer observation counts")
        windows = [positive_integer(window, "windows") for window in windows]
        if not metrics or any(
            metric not in {"sharpe", "volatility", "returns", "beta"} for metric in metrics
        ):
            raise ValueError("metrics must contain sharpe, volatility, returns or beta")
        if "beta" in metrics and not self.has_benchmark:
            raise ValueError("metrics beta requires a benchmark")

        cache_key = (tuple(windows), tuple(metrics))
        if cache_key in self._rolling_cache:
            return deepcopy(self._rolling_cache[cache_key])

        returns = self._returns
        rf = self._risk_free
        ppy = self._periods_per_year

        result = RollingMetricsResult(windows=windows, dates=self._dates.clone())

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
        return deepcopy(result)

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

        if n < window or window < 2:
            return result

        daily_rf = (1 + risk_free) ** (1 / periods_per_year) - 1

        # Vectorized: create all windows at once
        windows = sliding_window_view(returns, window)
        excess = windows - daily_rf

        # Compute mean and std across each window (axis=1)
        mu = np.mean(excess, axis=1)
        sd = np.std(excess, axis=1, ddof=1)

        # Sharpe where std > 0
        sharpe = np.full(len(mu), np.nan)
        np.divide(mu, sd, out=sharpe, where=sd > 0)
        sharpe *= np.sqrt(periods_per_year)
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

        if n < window or window < 2:
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
        """Compound rolling returns in O(n), retaining exact total-loss observations."""
        n = len(returns)
        result = np.full(n, np.nan)

        if n < window:
            return result

        total_loss = returns == -1
        log_returns = np.zeros_like(returns)
        np.log1p(returns, out=log_returns, where=~total_loss)
        cumsum = np.concatenate(([0.0], np.cumsum(log_returns)))
        loss_counts = np.concatenate(([0], np.cumsum(total_loss)))
        window_log_sum = cumsum[window:] - cumsum[:-window]
        has_loss = loss_counts[window:] > loss_counts[:-window]
        compounded = np.full(len(window_log_sum), -1.0)
        np.expm1(window_log_sum, out=compounded, where=~has_loss)
        result[window - 1 :] = compounded

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

        if n < window or window < 2:
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
        beta = np.full(len(cov), np.nan)
        np.divide(cov, var, out=beta, where=var > 0)
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
            Nonnegative number of top drawdowns to select. Zero returns only the
            aggregate statistics; it does not change the total episode count.
        threshold : float, default 0.01
            Finite minimum absolute depth in [0, 1]. Episodes strictly deeper
            than this threshold are counted. Does not change the underwater curve.

        Returns
        -------
        DrawdownResult
            Detailed drawdown statistics

        Examples
        --------
        >>> dd = analysis.compute_drawdown_analysis(top_n=10)
        >>> print(f"Max drawdown: {dd.max_drawdown:.1%}")
        >>> print(f"Avg duration: {dd.avg_duration_days:.0f} observations")
        """
        if isinstance(top_n, (bool, np.bool_)) or not isinstance(top_n, Integral) or top_n < 0:
            raise ValueError("top_n must be a nonnegative integer")
        if (
            isinstance(threshold, (bool, np.bool_))
            or not isinstance(threshold, Real)
            or not np.isfinite(threshold)
            or not 0 <= threshold <= 1
        ):
            raise ValueError("threshold must be a finite real depth in [0, 1]")
        top_n, threshold = int(top_n), float(threshold)
        cache_key = (top_n, threshold)
        if cache_key in self._drawdown_cache:
            return deepcopy(self._drawdown_cache[cache_key])

        returns = self._returns
        dates = self._dates

        # The initial valuation is a high-water mark, not an extra return observation.
        cum_returns = _safe_cumprod(1 + returns)
        underwater = _drawdown_from_wealth(cum_returns)

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

        self._drawdown_cache[cache_key] = DrawdownResult(
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

        return deepcopy(self._drawdown_cache[cache_key])

    def _identify_drawdown_periods(
        self,
        underwater: np.ndarray,
        dates: pl.Series,
        threshold: float,
    ) -> list[DrawdownPeriod]:
        """Identify individual drawdown periods."""
        periods = []

        in_drawdown = False
        peak_idx = -1
        valley_idx = 0
        valley_depth = 0.0

        for i, dd in enumerate(underwater):
            if not in_drawdown and dd >= 0:
                peak_idx = i
            elif dd < -threshold and not in_drawdown:
                # Start of drawdown
                in_drawdown = True
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
                        peak_date=(self._initial_timestamp if peak_idx < 0 else dates[peak_idx]),
                        valley_date=dates[valley_idx],
                        recovery_date=dates[i],
                        depth=valley_depth,
                        duration_days=valley_idx - peak_idx,
                        recovery_days=i - valley_idx,
                    )
                    periods.append(period)
                    in_drawdown = False
                    peak_idx = i

        # Handle ongoing drawdown
        if in_drawdown:
            period = DrawdownPeriod(
                peak_date=(self._initial_timestamp if peak_idx < 0 else dates[peak_idx]),
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
        self.require_dates()
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
        self.require_dates()
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
        pivot_key = "on" if "on" in signature(monthly.pivot).parameters else "columns"

        # Pivot to matrix form
        return monthly.pivot(
            values="monthly_return",
            index="year",
            **{pivot_key: "month"},
        ).sort("year")

    def create_tear_sheet(
        self,
        theme: str | None = None,
        cost_info: dict[str, str] | None = None,
        *,
        transactions: TransactionAnalysisResult | None = None,
    ):
        """Create comprehensive portfolio tear sheet.

        Convenience method delegating to create_portfolio_dashboard().

        Parameters
        ----------
        theme : str, optional
            Plot theme ("default", "dark", "print", "presentation")
        cost_info : dict[str, str] | None
            Deprecated legacy parameter. Retained for backward compatibility.
        transactions : TransactionAnalysisResult, optional
            Explicitly prepared actual-fill turnover and incremental cost scenarios.
            Computed results are rendered without selecting financial methods again.

        Returns
        -------
        PortfolioTearSheet
            Complete tear sheet with figures and metrics
        """
        from ml4t.diagnostic.visualization.portfolio import create_portfolio_dashboard

        _ = cost_info
        holdings = None
        if (
            self._input_data is not None
            and self._input_data.positions is not None
            and self._input_data.metadata["complete_snapshots"]
        ):
            holdings = self.compute_holdings_analysis()
        return create_portfolio_dashboard(
            self, theme=theme, holdings=holdings, transactions=transactions
        )


__all__ = ["PortfolioAnalysis"]
