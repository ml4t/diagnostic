"""Trade-level analysis for backtest diagnostics and SHAP attribution.

This module provides tools for analyzing individual trades from backtests,
identifying worst/best performers, and computing trade-level statistics.

Core Components:
    - TradeMetrics: Enriched trade data with computed metrics
    - TradeAnalysis: Main analyzer for extracting worst/best trades
    - TradeStatistics: Aggregate statistics across trades
    - TradeAnalysisResult: Result schema with serialization

Integration with ml4t-diagnostics workflow:
    1. Load backtest results → Extract trades (TradeRecord instances)
    2. Analyze trades → Identify worst performers (TradeAnalysis)
    3. Compute statistics → Understand trade distribution (TradeStatistics)
    4. Feed to SHAP → Explain failures (trade_shap_diagnostics.py)

Example - Basic usage:
    >>> from ml4t.diagnostic.integration import TradeRecord
    >>> from ml4t.diagnostic.evaluation import TradeAnalysis
    >>> from datetime import datetime, timedelta
    >>>
    >>> # Create trade records from backtest
    >>> trades = [
    ...     TradeRecord(
    ...         timestamp=datetime(2024, 1, 15),
    ...         symbol="AAPL",
    ...         entry_price=150.0,
    ...         exit_price=155.0,
    ...         pnl=500.0,
    ...         duration=timedelta(days=5),
    ...         direction="long"
    ...     ),
    ...     # ... more trades
    ... ]
    >>>
    >>> # Analyze trades
    >>> analyzer = TradeAnalysis(trades)
    >>> worst = analyzer.worst_trades(n=10)
    >>> best = analyzer.best_trades(n=10)
    >>> stats = analyzer.compute_statistics()
    >>>
    >>> print(f"Win rate: {stats.win_rate:.2%}")
    >>> print(f"Average PnL: ${stats.avg_pnl:.2f}")

Example - Advanced usage with config:
    >>> from ml4t.diagnostic.config import TradeConfig, ExtractionSettings, FilterSettings
    >>>
    >>> config = TradeConfig(
    ...     extraction=ExtractionSettings(n_worst=20, n_best=10),
    ...     filter=FilterSettings(
    ...         min_duration=timedelta(hours=1),
    ...         min_pnl=-1000.0
    ...     )
    ... )
    >>>
    >>> analyzer = TradeAnalysis.from_config(trades, config)
    >>> result = analyzer.analyze()
    >>>
    >>> # Export for storage
    >>> result.to_json_string()
    >>> result.get_dataframe("worst_trades")
    >>> result.get_dataframe("statistics")

Example - Integration with SHAP diagnostics:
    >>> from ml4t.diagnostic.evaluation import TradeShapAnalyzer
    >>>
    >>> # Get worst trades
    >>> worst_trades = analyzer.worst_trades(n=20)
    >>>
    >>> # Explain with SHAP
    >>> shap_analyzer = TradeShapAnalyzer(model, features, shap_values)
    >>> patterns = shap_analyzer.explain_worst_trades(worst_trades)
    >>>
    >>> for pattern in patterns:
    ...     print(pattern.hypothesis)
    ...     print(pattern.actions)
"""

from __future__ import annotations

import heapq
from datetime import UTC, datetime, timedelta
from typing import Any, Literal, SupportsFloat, cast

import polars as pl
from pydantic import BaseModel, Field, field_validator

from ml4t.diagnostic.integration.backtest_contract import TradeRecord


class TradeMetrics(BaseModel):
    """Enriched trade data with computed metrics for analysis.

    Extends TradeRecord with additional computed fields useful for
    trade analysis, ranking, and diagnostics. Provides methods for
    DataFrame conversion and serialization.

    This class wraps TradeRecord and adds:
        - Return percentage calculation
        - Duration in hours/days for easy filtering
        - Return per day (annualized-like metric)
        - Ranking helpers

    Required Fields (from TradeRecord):
        timestamp: Trade exit timestamp
        symbol: Asset symbol
        entry_price: Average entry price
        exit_price: Average exit price
        pnl: Realized profit/loss
        duration: Time between entry and exit
        direction: Trade direction (long/short)

    Computed Fields:
        return_pct: Return as percentage of entry price
        duration_hours: Duration in hours
        duration_days: Duration in days
        pnl_per_day: PnL normalized by duration

    Example - Create from TradeRecord:
        >>> trade_record = TradeRecord(
        ...     timestamp=datetime(2024, 1, 15),
        ...     symbol="AAPL",
        ...     entry_price=150.0,
        ...     exit_price=155.0,
        ...     pnl=500.0,
        ...     duration=timedelta(days=5),
        ...     direction="long",
        ...     quantity=100
        ... )
        >>> metrics = TradeMetrics.from_trade_record(trade_record)
        >>> print(f"Return: {metrics.return_pct:.2%}")
        >>> print(f"PnL per day: ${metrics.pnl_per_day:.2f}")

    Example - Convert to DataFrame:
        >>> trades = [TradeMetrics.from_trade_record(tr) for tr in trade_records]
        >>> df = TradeMetrics.to_dataframe(trades)
        >>> print(df.select(["symbol", "pnl", "return_pct"]))
    """

    # Core fields (from TradeRecord)
    timestamp: datetime = Field(..., description="Trade exit timestamp")
    symbol: str = Field(..., min_length=1, description="Asset symbol")
    entry_price: float = Field(..., gt=0.0, description="Average entry price")
    exit_price: float = Field(..., gt=0.0, description="Average exit price")
    pnl: float = Field(..., description="Realized profit/loss")
    duration: timedelta = Field(..., description="Time between entry and exit")
    direction: Literal["long", "short"] | None = Field(None, description="Trade direction")

    # Optional fields (from TradeRecord)
    quantity: float | None = Field(None, gt=0.0, description="Position size")
    entry_timestamp: datetime | None = Field(None, description="Position entry timestamp")
    fees: float | None = Field(None, ge=0.0, description="Total transaction fees")
    slippage: float | None = Field(None, ge=0.0, description="Slippage cost")
    metadata: dict[str, Any] | None = Field(None, description="Arbitrary metadata")
    regime_info: dict[str, str] | None = Field(None, description="Market regime info")

    @field_validator("duration")
    @classmethod
    def validate_duration_positive(cls, v: timedelta) -> timedelta:
        """Ensure duration is positive."""
        if v.total_seconds() <= 0:
            raise ValueError(f"Duration must be positive, got {v}")
        return v

    @property
    def return_pct(self) -> float:
        """Return as percentage of entry price.

        Formula:
            - Long: (exit_price - entry_price) / entry_price
            - Short: (entry_price - exit_price) / entry_price
            - Unknown: absolute price change / entry_price (unsigned)

        Returns:
            Return percentage (e.g., 0.05 = 5% return)

        Example:
            >>> metrics.return_pct  # 0.0333 = 3.33%
        """
        if self.direction == "long":
            return (self.exit_price - self.entry_price) / self.entry_price
        elif self.direction == "short":
            return (self.entry_price - self.exit_price) / self.entry_price
        else:
            # Unknown direction - use absolute price change (unsigned return)
            return abs(self.exit_price - self.entry_price) / self.entry_price

    @property
    def duration_hours(self) -> float:
        """Duration in hours.

        Returns:
            Duration as float hours

        Example:
            >>> metrics.duration_hours  # 120.5
        """
        return self.duration.total_seconds() / 3600.0

    @property
    def duration_days(self) -> float:
        """Duration in days.

        Returns:
            Duration as float days

        Example:
            >>> metrics.duration_days  # 5.02
        """
        return self.duration.total_seconds() / 86400.0

    @property
    def pnl_per_day(self) -> float:
        """PnL normalized by duration in days.

        Provides a duration-adjusted performance metric. Useful for
        comparing trades of different holding periods.

        Returns:
            PnL per day (can be negative)

        Example:
            >>> metrics.pnl_per_day  # 100.0 (earned $100/day)
        """
        if self.duration_days == 0:
            return 0.0
        return self.pnl / self.duration_days

    @classmethod
    def from_trade_record(cls, trade: TradeRecord) -> TradeMetrics:
        """Create TradeMetrics from TradeRecord.

        Args:
            trade: TradeRecord instance from backtest

        Returns:
            TradeMetrics with computed fields

        Example:
            >>> metrics = TradeMetrics.from_trade_record(trade_record)
        """
        return cls(
            timestamp=trade.timestamp,
            symbol=trade.symbol,
            entry_price=trade.entry_price,
            exit_price=trade.exit_price,
            pnl=trade.pnl,
            duration=trade.duration,
            direction=trade.direction,
            quantity=trade.quantity,
            entry_timestamp=trade.entry_timestamp,
            fees=trade.fees,
            slippage=trade.slippage,
            metadata=trade.metadata,
            regime_info=trade.regime_info,
        )

    def to_dict(self) -> dict[str, Any]:
        """Export to dictionary format.

        Returns:
            Dictionary with all trade data including computed fields

        Example:
            >>> metrics.to_dict()
            {
                'timestamp': '2024-01-15T10:30:00',
                'symbol': 'AAPL',
                'pnl': 500.0,
                'return_pct': 0.0333,
                'duration_hours': 120.0,
                ...
            }
        """
        data = self.model_dump(mode="json")
        # Convert timedelta to total seconds for JSON compatibility
        if "duration" in data:
            data["duration_seconds"] = self.duration.total_seconds()
            del data["duration"]
        # Include computed properties
        data["return_pct"] = self.return_pct
        data["duration_hours"] = self.duration_hours
        data["duration_days"] = self.duration_days
        data["pnl_per_day"] = self.pnl_per_day
        return data

    @staticmethod
    def to_dataframe(trades: list[TradeMetrics]) -> pl.DataFrame:
        """Convert list of TradeMetrics to Polars DataFrame.

        Args:
            trades: List of TradeMetrics instances

        Returns:
            Polars DataFrame with all trade data and computed metrics

        Example:
            >>> df = TradeMetrics.to_dataframe(metrics_list)
            >>> print(df.select(["symbol", "pnl", "return_pct"]))
            >>> df.sort("pnl").head(10)  # Worst 10 trades
        """
        if not trades:
            # Return empty DataFrame with expected schema (must match non-empty schema)
            return pl.DataFrame(
                schema={
                    "timestamp": pl.Datetime,
                    "symbol": pl.String,
                    "entry_price": pl.Float64,
                    "exit_price": pl.Float64,
                    "pnl": pl.Float64,
                    "duration_seconds": pl.Float64,
                    "direction": pl.String,
                    "quantity": pl.Float64,
                    "entry_timestamp": pl.Datetime,
                    "fees": pl.Float64,
                    "slippage": pl.Float64,
                    "return_pct": pl.Float64,
                    "duration_hours": pl.Float64,
                    "duration_days": pl.Float64,
                    "pnl_per_day": pl.Float64,
                }
            )

        # Convert to list of dicts
        data = []
        for trade in trades:
            trade_dict = {
                "timestamp": trade.timestamp,
                "symbol": trade.symbol,
                "entry_price": trade.entry_price,
                "exit_price": trade.exit_price,
                "pnl": trade.pnl,
                "duration_seconds": trade.duration.total_seconds(),
                "direction": trade.direction,
                "quantity": trade.quantity,
                "entry_timestamp": trade.entry_timestamp,
                "fees": trade.fees,
                "slippage": trade.slippage,
                "return_pct": trade.return_pct,
                "duration_hours": trade.duration_hours,
                "duration_days": trade.duration_days,
                "pnl_per_day": trade.pnl_per_day,
            }
            data.append(trade_dict)

        return pl.DataFrame(data)


class TradeFilters(BaseModel):
    """Typed filter configuration for trade analysis.

    Provides type-safe, validated filtering options instead of raw dict[str, Any].
    All fields are optional - only specified filters are applied.

    Fields:
        symbols: List of symbols to include (None = all symbols)
        min_duration: Minimum trade duration (None = no minimum)
        min_pnl: Minimum PnL to include (None = no minimum)
        max_pnl: Maximum PnL to include (None = no maximum)
        start_date: Start of date range (None = no start bound)
        end_date: End of date range (None = no end bound)

    Example:
        >>> filters = TradeFilters(
        ...     symbols=["AAPL", "MSFT"],
        ...     min_duration=timedelta(hours=1),
        ...     min_pnl=-1000.0
        ... )
        >>> analyzer = TradeAnalysis(trades, filters=filters)
    """

    symbols: list[str] | None = Field(None, description="Symbols to include")
    min_duration: timedelta | None = Field(None, description="Minimum trade duration")
    min_pnl: float | None = Field(None, description="Minimum PnL to include")
    max_pnl: float | None = Field(None, description="Maximum PnL to include")
    start_date: datetime | None = Field(None, description="Start of date range")
    end_date: datetime | None = Field(None, description="End of date range")

    def to_dict(self) -> dict[str, Any]:
        """Convert to legacy dict format for backward compatibility."""
        result: dict[str, Any] = {}
        if self.symbols is not None:
            result["symbols"] = self.symbols
        if self.min_duration is not None:
            result["min_duration_seconds"] = self.min_duration.total_seconds()
        if self.min_pnl is not None:
            result["min_pnl"] = self.min_pnl
        if self.max_pnl is not None:
            result["max_pnl"] = self.max_pnl
        if self.start_date is not None:
            result["start_date"] = self.start_date
        if self.end_date is not None:
            result["end_date"] = self.end_date
        return result


class TradeStatistics(BaseModel):
    """Aggregate statistics across multiple trades.

    Computes summary statistics for trade analysis:
        - Win/loss metrics (win rate, profit factor)
        - PnL distribution (mean, std, quartiles, skewness)
        - Duration distribution (mean, median, quartiles)
        - Trade counts and breakdowns

    Used by TradeAnalysisResult to provide high-level performance summary.

    Fields:
        n_trades: Total number of trades
        n_winners: Number of profitable trades
        n_losers: Number of losing trades
        win_rate: Fraction of winning trades
        total_pnl: Sum of all PnL
        avg_pnl: Mean PnL per trade
        pnl_std: Standard deviation of PnL
        pnl_skewness: Skewness of PnL distribution
        pnl_kurtosis: Kurtosis of PnL distribution
        pnl_quartiles: 25th, 50th (median), 75th percentiles
        avg_winner: Average PnL of winning trades
        avg_loser: Average PnL of losing trades
        profit_factor: Gross profit / gross loss
        avg_duration_days: Average trade duration in days
        median_duration_days: Median trade duration
        duration_quartiles: Duration percentiles

    Example:
        >>> stats = TradeStatistics.compute(trades)
        >>> print(f"Win rate: {stats.win_rate:.2%}")
        >>> print(f"Avg PnL: ${stats.avg_pnl:.2f}")
        >>> print(f"Profit factor: {stats.profit_factor:.2f}")
        >>> print(stats.summary())
    """

    # Trade counts
    n_trades: int = Field(..., ge=0, description="Total number of trades")
    n_winners: int = Field(..., ge=0, description="Number of profitable trades")
    n_losers: int = Field(..., ge=0, description="Number of losing trades")

    # Win rate and PnL metrics
    win_rate: float = Field(..., ge=0.0, le=1.0, description="Fraction of winning trades")
    total_pnl: float = Field(..., description="Sum of all PnL")
    avg_pnl: float = Field(..., description="Mean PnL per trade")
    pnl_std: float = Field(..., ge=0.0, description="Standard deviation of PnL")

    # Distribution metrics
    pnl_skewness: float | None = Field(None, description="PnL distribution skewness")
    pnl_kurtosis: float | None = Field(None, description="PnL distribution kurtosis")
    pnl_quartiles: dict[str, float] = Field(..., description="PnL quartiles (q25, q50, q75)")

    # Winner/loser breakdown
    avg_winner: float | None = Field(None, description="Average PnL of winners")
    avg_loser: float | None = Field(None, description="Average PnL of losers")
    profit_factor: float | None = Field(None, description="Gross profit / gross loss")

    # Duration metrics
    avg_duration_days: float = Field(..., ge=0.0, description="Average duration in days")
    median_duration_days: float = Field(..., ge=0.0, description="Median duration in days")
    duration_quartiles: dict[str, float] = Field(
        ..., description="Duration quartiles (q25, q50, q75)"
    )

    @staticmethod
    def compute(trades: list[TradeMetrics]) -> TradeStatistics:
        """Compute statistics from list of trades.

        Args:
            trades: List of TradeMetrics instances

        Returns:
            TradeStatistics with all computed metrics

        Raises:
            ValueError: If trades list is empty

        Example:
            >>> stats = TradeStatistics.compute(metrics_list)
        """
        if not trades:
            raise ValueError("Cannot compute statistics for empty trade list")

        # Convert to DataFrame for efficient computation
        df = TradeMetrics.to_dataframe(trades)

        # Count trades
        n_trades = len(df)
        n_winners = int(df.filter(pl.col("pnl") > 0).height)
        n_losers = int(df.filter(pl.col("pnl") < 0).height)
        win_rate = n_winners / n_trades if n_trades > 0 else 0.0

        # PnL metrics
        pnl_series = df["pnl"]
        total_pnl = float(cast(SupportsFloat, pnl_series.sum()))
        avg_pnl = float(cast(SupportsFloat, pnl_series.mean()))
        pnl_std_value = pnl_series.std()
        pnl_std = float(cast(SupportsFloat, pnl_std_value)) if pnl_std_value is not None else 0.0

        # Distribution metrics (requires scipy for skewness/kurtosis)
        try:
            from scipy import stats as scipy_stats

            pnl_values = pnl_series.to_numpy()
            pnl_skewness = float(scipy_stats.skew(pnl_values))
            pnl_kurtosis = float(scipy_stats.kurtosis(pnl_values))
        except ImportError:
            pnl_skewness = None
            pnl_kurtosis = None

        # Quartiles
        pnl_q25 = float(cast(SupportsFloat, pnl_series.quantile(0.25)))
        pnl_q50 = float(cast(SupportsFloat, pnl_series.quantile(0.50)))
        pnl_q75 = float(cast(SupportsFloat, pnl_series.quantile(0.75)))
        pnl_quartiles = {"q25": pnl_q25, "q50": pnl_q50, "q75": pnl_q75}

        # Winner/loser breakdown
        winners = df.filter(pl.col("pnl") > 0)
        losers = df.filter(pl.col("pnl") < 0)

        avg_winner = (
            float(cast(SupportsFloat, winners["pnl"].mean())) if winners.height > 0 else None
        )
        avg_loser = float(cast(SupportsFloat, losers["pnl"].mean())) if losers.height > 0 else None

        # Profit factor (only defined if both winners and losers exist)
        gross_profit = float(winners["pnl"].sum()) if winners.height > 0 else 0.0
        gross_loss = abs(float(losers["pnl"].sum())) if losers.height > 0 else 0.0
        if winners.height > 0 and losers.height > 0:
            profit_factor = gross_profit / gross_loss
        else:
            profit_factor = None  # Undefined when all winners or all losers

        # Duration metrics
        duration_series = df["duration_days"]
        avg_duration_days = float(cast(SupportsFloat, duration_series.mean()))
        median_duration_days = float(cast(SupportsFloat, duration_series.median()))
        dur_q25 = float(cast(SupportsFloat, duration_series.quantile(0.25)))
        dur_q50 = float(cast(SupportsFloat, duration_series.quantile(0.50)))
        dur_q75 = float(cast(SupportsFloat, duration_series.quantile(0.75)))
        duration_quartiles = {"q25": dur_q25, "q50": dur_q50, "q75": dur_q75}

        return TradeStatistics(
            n_trades=n_trades,
            n_winners=n_winners,
            n_losers=n_losers,
            win_rate=win_rate,
            total_pnl=total_pnl,
            avg_pnl=avg_pnl,
            pnl_std=pnl_std,
            pnl_skewness=pnl_skewness,
            pnl_kurtosis=pnl_kurtosis,
            pnl_quartiles=pnl_quartiles,
            avg_winner=avg_winner,
            avg_loser=avg_loser,
            profit_factor=profit_factor,
            avg_duration_days=avg_duration_days,
            median_duration_days=median_duration_days,
            duration_quartiles=duration_quartiles,
        )

    def summary(self) -> str:
        """Generate human-readable summary of statistics.

        Returns:
            Formatted summary string

        Example:
            >>> print(stats.summary())
            Trade Statistics
            ================
            Total trades: 150
            Win rate: 62.67%
            ...
        """
        lines = ["Trade Statistics", "=" * 50]

        # Trade counts
        lines.append(f"Total trades: {self.n_trades}")
        lines.append(f"Winners: {self.n_winners} | Losers: {self.n_losers}")
        lines.append(f"Win rate: {self.win_rate:.2%}")
        lines.append("")

        # PnL summary
        lines.append("PnL Metrics")
        lines.append("-" * 50)
        lines.append(f"Total PnL: ${self.total_pnl:,.2f}")
        lines.append(f"Average PnL: ${self.avg_pnl:,.2f} ± ${self.pnl_std:,.2f}")
        if self.avg_winner is not None:
            lines.append(f"Avg winner: ${self.avg_winner:,.2f}")
        if self.avg_loser is not None:
            lines.append(f"Avg loser: ${self.avg_loser:,.2f}")
        if self.profit_factor is not None:
            lines.append(f"Profit factor: {self.profit_factor:.2f}")
        lines.append("")

        # Distribution
        lines.append("PnL Distribution")
        lines.append("-" * 50)
        lines.append(
            f"Q25: ${self.pnl_quartiles['q25']:,.2f} | "
            f"Median: ${self.pnl_quartiles['q50']:,.2f} | "
            f"Q75: ${self.pnl_quartiles['q75']:,.2f}"
        )
        if self.pnl_skewness is not None:
            lines.append(f"Skewness: {self.pnl_skewness:.3f}")
        if self.pnl_kurtosis is not None:
            lines.append(f"Kurtosis: {self.pnl_kurtosis:.3f}")
        lines.append("")

        # Duration
        lines.append("Duration Metrics")
        lines.append("-" * 50)
        lines.append(f"Average: {self.avg_duration_days:.2f} days")
        lines.append(f"Median: {self.median_duration_days:.2f} days")
        lines.append(
            f"Q25: {self.duration_quartiles['q25']:.2f} | "
            f"Q50: {self.duration_quartiles['q50']:.2f} | "
            f"Q75: {self.duration_quartiles['q75']:.2f}"
        )

        return "\n".join(lines)

    def to_dataframe(self) -> pl.DataFrame:
        """Convert statistics to DataFrame.

        Returns:
            Single-row DataFrame with all statistics

        Example:
            >>> df = stats.to_dataframe()
        """
        return pl.DataFrame(
            [
                {
                    "n_trades": self.n_trades,
                    "n_winners": self.n_winners,
                    "n_losers": self.n_losers,
                    "win_rate": self.win_rate,
                    "total_pnl": self.total_pnl,
                    "avg_pnl": self.avg_pnl,
                    "pnl_std": self.pnl_std,
                    "pnl_skewness": self.pnl_skewness,
                    "pnl_kurtosis": self.pnl_kurtosis,
                    "pnl_q25": self.pnl_quartiles["q25"],
                    "pnl_q50": self.pnl_quartiles["q50"],
                    "pnl_q75": self.pnl_quartiles["q75"],
                    "avg_winner": self.avg_winner,
                    "avg_loser": self.avg_loser,
                    "profit_factor": self.profit_factor,
                    "avg_duration_days": self.avg_duration_days,
                    "median_duration_days": self.median_duration_days,
                    "dur_q25": self.duration_quartiles["q25"],
                    "dur_q50": self.duration_quartiles["q50"],
                    "dur_q75": self.duration_quartiles["q75"],
                }
            ]
        )


class TradeAnalysis:
    """Main analyzer for extracting worst/best trades and computing statistics.

    Provides high-level API for trade analysis workflows:
        1. Load trades (TradeRecord instances from backtest)
        2. Extract worst performers → Feed to SHAP diagnostics
        3. Extract best performers → Understand success patterns
        4. Compute statistics → Aggregate performance metrics

    The analyzer supports filtering by:
        - Symbol (e.g., only analyze AAPL, MSFT)
        - Duration (e.g., trades lasting > 1 hour)
        - PnL range (e.g., exclude small trades)
        - Date range (e.g., trades in Q4 2024)

    Example - Basic usage:
        >>> analyzer = TradeAnalysis(trade_records)
        >>> worst = analyzer.worst_trades(n=20)
        >>> best = analyzer.best_trades(n=10)
        >>> stats = analyzer.compute_statistics()
        >>> print(stats.summary())

    Example - With filtering:
        >>> from ml4t.diagnostic.evaluation.trade_analysis import TradeFilters
        >>>
        >>> filters = TradeFilters(
        ...     symbols=["AAPL", "MSFT"],
        ...     min_duration=timedelta(hours=1),
        ...     min_pnl=-1000.0,
        ...     start_date=datetime(2024, 10, 1)
        ... )
        >>>
        >>> analyzer = TradeAnalysis(trade_records, filters=filters)
        >>> result = analyzer.analyze(n_worst=20, n_best=10)

    Example - Integration with config:
        >>> from ml4t.diagnostic.config import TradeConfig, ExtractionSettings
        >>>
        >>> config = TradeConfig(
        ...     extraction=ExtractionSettings(n_worst=20, n_best=10),
        ... )
        >>>
        >>> analyzer = TradeAnalysis.from_config(trade_records, config)
        >>> result = analyzer.analyze()
        >>> result.to_json_string()
    """

    def __init__(
        self,
        trades: list[TradeRecord],
        filter_config: dict[str, Any] | None = None,
        *,
        filters: TradeFilters | None = None,
    ):
        """Initialize analyzer with trades.

        Args:
            trades: List of TradeRecord instances from backtest
            filter_config: Optional filtering configuration (legacy dict format)
            filters: Optional typed TradeFilters (preferred over filter_config)

        Example:
            >>> # Using typed filters (preferred)
            >>> filters = TradeFilters(symbols=["AAPL"], min_pnl=-1000)
            >>> analyzer = TradeAnalysis(trades, filters=filters)
            >>>
            >>> # Using legacy dict format
            >>> analyzer = TradeAnalysis(trades, filter_config={"symbols": ["AAPL"]})
        """
        if not trades:
            raise ValueError("Cannot analyze empty trade list")

        # Convert to TradeMetrics
        self.trades = [TradeMetrics.from_trade_record(t) for t in trades]

        # Normalize filters to dict format (TradeFilters takes precedence)
        if filters is not None:
            filter_config = filters.to_dict()

        # Apply filters if provided
        if filter_config:
            self.trades = self._apply_filters(self.trades, filter_config)

        if not self.trades:
            raise ValueError("No trades remaining after applying filters")

    @classmethod
    def from_config(
        cls,
        trades: list[TradeRecord],
        config: Any,  # TradeConfig - avoid circular import
    ) -> TradeAnalysis:
        """Create analyzer from configuration.

        Args:
            trades: List of TradeRecord instances
            config: TradeConfig instance

        Returns:
            TradeAnalysis instance

        Example:
            >>> from ml4t.diagnostic.config import TradeConfig, ExtractionSettings
            >>> config = TradeConfig(extraction=ExtractionSettings(n_worst=20, n_best=10))
            >>> analyzer = TradeAnalysis.from_config(trades, config)
        """
        # Extract filter config if present
        filter_config = getattr(config, "filters", None)
        return cls(trades, filter_config=filter_config)

    @staticmethod
    def _apply_filters(
        trades: list[TradeMetrics],
        filters: dict[str, Any],
    ) -> list[TradeMetrics]:
        """Apply filters to trade list in a single pass.

        Args:
            trades: List of TradeMetrics
            filters: Filter criteria

        Returns:
            Filtered trade list
        """
        # Pre-extract filter values to avoid repeated dict lookups
        symbols: set[str] | None = None
        if "symbols" in filters and filters["symbols"]:
            symbols = set(filters["symbols"])

        min_dur: float | None = filters.get("min_duration_seconds")
        min_pnl: float | None = filters.get("min_pnl")
        max_pnl: float | None = filters.get("max_pnl")
        start_date: datetime | None = filters.get("start_date")
        end_date: datetime | None = filters.get("end_date")

        # Single-pass filtering
        def matches(t: TradeMetrics) -> bool:
            if symbols is not None and t.symbol not in symbols:
                return False
            if min_dur is not None and t.duration.total_seconds() < min_dur:
                return False
            if min_pnl is not None and t.pnl < min_pnl:
                return False
            if max_pnl is not None and t.pnl > max_pnl:
                return False
            if start_date is not None and t.timestamp < start_date:
                return False
            if end_date is not None and t.timestamp > end_date:
                return False
            return True

        return [t for t in trades if matches(t)]

    def worst_trades(self, n: int = 10) -> list[TradeMetrics]:
        """Extract N worst trades by PnL.

        Uses heapq.nsmallest for O(n + k log n) efficiency when k << n.

        Args:
            n: Number of worst trades to extract

        Returns:
            List of worst N trades, sorted by PnL (ascending)

        Example:
            >>> worst = analyzer.worst_trades(n=20)
            >>> for trade in worst[:5]:
            ...     print(f"{trade.symbol}: ${trade.pnl:.2f}")

        See Also
        --------
        best_trades : Extract best performing trades
        compute_statistics : Aggregate performance metrics
        analyze : Complete analysis with worst/best trades
        """
        if n <= 0:
            raise ValueError(f"n must be positive, got {n}")

        # Use heapq for O(n + k log n) instead of O(n log n) sort
        return heapq.nsmallest(n, self.trades, key=lambda t: t.pnl)

    def best_trades(self, n: int = 10) -> list[TradeMetrics]:
        """Extract N best trades by PnL.

        Uses heapq.nlargest for O(n + k log n) efficiency when k << n.

        Args:
            n: Number of best trades to extract

        Returns:
            List of best N trades, sorted by PnL (descending)

        Example:
            >>> best = analyzer.best_trades(n=10)
            >>> for trade in best[:5]:
            ...     print(f"{trade.symbol}: ${trade.pnl:.2f}")

        See Also
        --------
        worst_trades : Extract worst performing trades
        compute_statistics : Aggregate performance metrics
        analyze : Complete analysis with worst/best trades
        """
        if n <= 0:
            raise ValueError(f"n must be positive, got {n}")

        # Use heapq for O(n + k log n) instead of O(n log n) sort
        return heapq.nlargest(n, self.trades, key=lambda t: t.pnl)

    def compute_statistics(self) -> TradeStatistics:
        """Compute aggregate statistics across all trades.

        Returns:
            TradeStatistics with summary metrics

        Example:
            >>> stats = analyzer.compute_statistics()
            >>> print(f"Win rate: {stats.win_rate:.2%}")

        See Also
        --------
        TradeStatistics : Statistics result schema
        TradeStatistics.compute : Static method for statistics computation
        analyze : Complete analysis including statistics
        """
        return TradeStatistics.compute(self.trades)

    def analyze(
        self,
        n_worst: int = 10,
        n_best: int = 10,
    ) -> TradeAnalysisResult:
        """Run complete analysis and return result object.

        Args:
            n_worst: Number of worst trades to extract
            n_best: Number of best trades to extract

        Returns:
            TradeAnalysisResult with all data

        Example:
            >>> result = analyzer.analyze(n_worst=20, n_best=10)
            >>> print(result.summary())
            >>> result.to_json_string()

        See Also
        --------
        worst_trades : Extract worst trades
        best_trades : Extract best trades
        compute_statistics : Compute aggregate statistics
        TradeAnalysisResult : Result schema with serialization
        """
        return TradeAnalysisResult(
            worst_trades=self.worst_trades(n_worst),
            best_trades=self.best_trades(n_best),
            statistics=self.compute_statistics(),
            n_total_trades=len(self.trades),
        )


class TradeAnalysisResult(BaseModel):
    """Result schema for trade analysis with serialization support.

    Contains the complete output of a trade analysis:
        - Worst N trades (for SHAP diagnostics)
        - Best N trades (for success pattern analysis)
        - Aggregate statistics across all trades
        - Metadata (total trades analyzed)

    This schema extends BaseResult to provide:
        - JSON serialization via to_json_string()
        - DataFrame export via get_dataframe()
        - Human-readable summary via summary()

    Use this to store and retrieve analysis results, or to pass
    data between different stages of the diagnostics workflow.

    Fields:
        worst_trades: List of worst N trades by PnL
        best_trades: List of best N trades by PnL
        statistics: Aggregate statistics
        n_total_trades: Total trades analyzed (before worst/best filtering)
        analysis_type: Type of analysis ("trade_analysis")
        created_at: ISO timestamp of analysis creation

    Example - Basic usage:
        >>> result = analyzer.analyze(n_worst=20, n_best=10)
        >>> print(result.summary())
        >>> result.to_json_string()
        >>> df = result.get_dataframe("worst_trades")

    Example - Serialization:
        >>> # Save to file
        >>> with open("analysis_result.json", "w") as f:
        ...     f.write(result.to_json_string())
        >>>
        >>> # Load from file
        >>> with open("analysis_result.json") as f:
        ...     data = json.load(f)
        >>> result = TradeAnalysisResult(**data)

    Example - DataFrame export:
        >>> # Get worst trades as DataFrame
        >>> df_worst = result.get_dataframe("worst_trades")
        >>>
        >>> # Get statistics as DataFrame
        >>> df_stats = result.get_dataframe("statistics")
        >>>
        >>> # Get all available DataFrames
        >>> available = result.list_available_dataframes()
    """

    # Result fields
    worst_trades: list[TradeMetrics] = Field(
        ...,
        description="List of worst N trades by PnL",
    )
    best_trades: list[TradeMetrics] = Field(
        ...,
        description="List of best N trades by PnL",
    )
    statistics: TradeStatistics = Field(
        ...,
        description="Aggregate statistics across all trades",
    )
    n_total_trades: int = Field(
        ...,
        ge=1,
        description="Total number of trades analyzed",
    )

    # Metadata fields
    analysis_type: str = Field(
        default="trade_analysis",
        description="Type of analysis performed",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Analysis creation timestamp (UTC)",
    )

    def to_json_string(self, *, indent: int = 2) -> str:
        """Export to JSON string.

        Args:
            indent: Indentation level (None for compact)

        Returns:
            JSON string representation

        Example:
            >>> json_str = result.to_json_string()
            >>> with open("result.json", "w") as f:
            ...     f.write(json_str)
        """
        return self.model_dump_json(indent=indent)

    def to_dict(self) -> dict[str, Any]:
        """Export to Python dictionary.

        Returns:
            Dictionary representation

        Example:
            >>> data = result.to_dict()
            >>> data["statistics"]["win_rate"]
        """
        return self.model_dump(mode="python")

    def get_dataframe(self, name: str = "worst_trades") -> pl.DataFrame:
        """Get results as Polars DataFrame.

        Available DataFrames:
            - "worst_trades": Worst N trades with all fields
            - "best_trades": Best N trades with all fields
            - "statistics": Aggregate statistics (single row)
            - "all_trades": Combined worst + best trades

        Args:
            name: DataFrame name to retrieve

        Returns:
            Polars DataFrame with requested data

        Raises:
            ValueError: If DataFrame name not available

        Example:
            >>> df_worst = result.get_dataframe("worst_trades")
            >>> df_stats = result.get_dataframe("statistics")
        """
        if name == "worst_trades":
            return TradeMetrics.to_dataframe(self.worst_trades)
        elif name == "best_trades":
            return TradeMetrics.to_dataframe(self.best_trades)
        elif name == "statistics":
            return self.statistics.to_dataframe()
        elif name == "all_trades":
            # Combine worst and best
            all_trades = self.worst_trades + self.best_trades
            return TradeMetrics.to_dataframe(all_trades)
        else:
            available = self.list_available_dataframes()
            raise ValueError(f"DataFrame '{name}' not available. Available: {', '.join(available)}")

    def list_available_dataframes(self) -> list[str]:
        """List available DataFrame views.

        Returns:
            List of available DataFrame names

        Example:
            >>> result.list_available_dataframes()
            ['worst_trades', 'best_trades', 'statistics', 'all_trades']
        """
        return ["worst_trades", "best_trades", "statistics", "all_trades"]

    def summary(self) -> str:
        """Generate human-readable summary of analysis.

        Returns:
            Formatted summary string

        Example:
            >>> print(result.summary())
            Trade Analysis Summary
            ======================
            ...
        """
        lines = ["Trade Analysis Summary", "=" * 60]

        # Overview
        lines.append(f"Analysis timestamp: {self.created_at.isoformat()}")
        lines.append(f"Total trades analyzed: {self.n_total_trades}")
        lines.append(f"Worst trades extracted: {len(self.worst_trades)}")
        lines.append(f"Best trades extracted: {len(self.best_trades)}")
        lines.append("")

        # Statistics summary
        lines.append("Overall Statistics")
        lines.append("-" * 60)
        stats = self.statistics
        lines.append(f"Win rate: {stats.win_rate:.2%}")
        lines.append(f"Total PnL: ${stats.total_pnl:,.2f}")
        lines.append(f"Average PnL: ${stats.avg_pnl:,.2f} ± ${stats.pnl_std:,.2f}")
        if stats.profit_factor is not None:
            lines.append(f"Profit factor: {stats.profit_factor:.2f}")
        lines.append(f"Average duration: {stats.avg_duration_days:.2f} days")
        lines.append("")

        # Worst trades preview
        lines.append("Worst Trades (Top 5)")
        lines.append("-" * 60)
        for i, trade in enumerate(self.worst_trades[:5], 1):
            lines.append(
                f"{i}. {trade.symbol}: ${trade.pnl:,.2f} ({trade.return_pct:+.2%}) [{trade.duration_days:.1f}d]"
            )
        lines.append("")

        # Best trades preview
        lines.append("Best Trades (Top 5)")
        lines.append("-" * 60)
        for i, trade in enumerate(self.best_trades[:5], 1):
            lines.append(
                f"{i}. {trade.symbol}: ${trade.pnl:,.2f} ({trade.return_pct:+.2%}) [{trade.duration_days:.1f}d]"
            )

        return "\n".join(lines)

    def __repr__(self) -> str:
        """Concise representation."""
        return (
            f"TradeAnalysisResult("
            f"n_worst={len(self.worst_trades)}, "
            f"n_best={len(self.best_trades)}, "
            f"n_total={self.n_total_trades})"
        )
