"""ML4T Backtest integration contract for backtest evaluation and comparison.

This module defines the API contract between ML4T Diagnostic and ML4T Backtest for:
1. Exporting evaluation results to backtest storage
2. Comparing live vs backtest performance (Bayesian comparison)
3. Supporting paper vs live promotion workflows

Example workflow - Backtest evaluation export:
    >>> from ml4t.diagnostic.evaluation import PortfolioEvaluator
    >>> from ml4t.diagnostic.integration import EvaluationExport
    >>>
    >>> # 1. Evaluate backtest results
    >>> evaluator = PortfolioEvaluator(config)
    >>> results = evaluator.evaluate(returns_df)
    >>>
    >>> # 2. Export for ML4T Backtest storage
    >>> export = results.to_backtest_export(
    ...     strategy_id="momentum_v1",
    ...     environment="backtest"
    ... )
    >>>
    >>> # 3. Store in ML4T Backtest database
    >>> # backtest_engine.store_evaluation(export.to_dict())

Example workflow - Live vs Backtest comparison:
    >>> from ml4t.diagnostic.integration import ComparisonRequest
    >>>
    >>> # 1. Create comparison request
    >>> request = ComparisonRequest(
    ...     strategy_id="momentum_v1",
    ...     backtest_results=backtest_results.to_dict(),
    ...     live_results=live_results.to_dict(),
    ...     comparison_type="bayesian"
    ... )
    >>>
    >>> # 2. Run Bayesian comparison
    >>> from ml4t.diagnostic.evaluation import BayesianComparison
    >>> comparison = BayesianComparison.from_request(request)
    >>> result = comparison.compare()
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class EnvironmentType(str, Enum):
    """Strategy execution environment.

    - BACKTEST: Historical simulation
    - PAPER: Forward testing with simulated execution
    - LIVE: Real trading with real capital
    """

    BACKTEST = "backtest"
    PAPER = "paper"
    LIVE = "live"


class ComparisonType(str, Enum):
    """Type of performance comparison.

    - BAYESIAN: Bayesian hypothesis testing (recommended)
    - BOOTSTRAP: Bootstrap confidence intervals
    - PARAMETRIC: T-test and F-test (assumes normality)
    - CUSUM: CUSUM drift detection
    """

    BAYESIAN = "bayesian"
    BOOTSTRAP = "bootstrap"
    PARAMETRIC = "parametric"
    CUSUM = "cusum"


class TradeRecord(BaseModel):
    """Individual trade record for trade-level SHAP diagnostics.

    This schema represents a single completed trade from a backtest or live trading.
    Used by ml4t-diagnostics for trade-level analysis, SHAP attribution, and
    error pattern clustering.

    The schema supports both simple (single-leg) and complex (multi-leg) trades,
    with optional metadata for regime detection and classification.

    Required Fields:
        timestamp: Trade exit timestamp (when position was closed)
        symbol: Asset symbol (e.g., "AAPL", "BTC-USD")
        entry_price: Average entry price
        exit_price: Average exit price
        pnl: Realized profit/loss (in quote currency)
        duration: Time between entry and exit

    Optional Fields:
        direction: Trade direction (long/short)
        metadata: Arbitrary metadata (e.g., entry signals, regime info)
        regime_info: Market regime at time of trade
        quantity: Position size
        entry_timestamp: When position was opened
        fees: Total transaction fees
        slippage: Estimated or actual slippage

    Validation:
        - PnL consistency with prices (for long/short trades)
        - Duration is positive
        - Prices are positive
        - Timestamps are valid

    Example - Simple long trade:
        >>> from datetime import datetime, timedelta
        >>> trade = TradeRecord(
        ...     timestamp=datetime(2024, 1, 15, 10, 30),
        ...     symbol="AAPL",
        ...     entry_price=150.00,
        ...     exit_price=155.00,
        ...     pnl=500.00,  # (155-150) * 100 shares
        ...     duration=timedelta(days=5),
        ...     direction="long",
        ...     quantity=100
        ... )

    Example - Short trade with metadata:
        >>> trade = TradeRecord(
        ...     timestamp=datetime(2024, 2, 1, 14, 0),
        ...     symbol="BTC-USD",
        ...     entry_price=45000.0,
        ...     exit_price=44000.0,
        ...     pnl=1000.0,  # (45000-44000) * 1 BTC
        ...     duration=timedelta(hours=6),
        ...     direction="short",
        ...     quantity=1.0,
        ...     metadata={
        ...         "entry_signal": "momentum_reversal",
        ...         "volatility_regime": "high",
        ...         "market_regime": "trending_down"
        ...     },
        ...     fees=50.0,
        ...     slippage=20.0
        ... )

    Example - For SHAP diagnostics workflow:
        >>> # 1. Extract worst trades from backtest
        >>> worst_trades = [t for t in all_trades if t.pnl < threshold]
        >>>
        >>> # 2. Analyze with SHAP
        >>> from ml4t.diagnostic.evaluation import TradeShapAnalyzer
        >>> analyzer = TradeShapAnalyzer(model, features)
        >>> patterns = analyzer.explain_worst_trades(worst_trades)
        >>>
        >>> # 3. Get actionable hypotheses
        >>> for pattern in patterns:
        ...     print(pattern.hypothesis)
        ...     print(pattern.actions)
    """

    # Required fields
    timestamp: datetime = Field(
        ...,
        description="Trade exit timestamp (when position was closed)",
    )
    symbol: str = Field(
        ...,
        min_length=1,
        description="Asset symbol (e.g., 'AAPL', 'BTC-USD', 'ES_F')",
    )
    entry_price: float = Field(
        ...,
        gt=0.0,
        description="Average entry price (must be positive)",
    )
    exit_price: float = Field(
        ...,
        gt=0.0,
        description="Average exit price (must be positive)",
    )
    pnl: float = Field(
        ...,
        description="Realized profit/loss in quote currency (can be negative)",
    )
    duration: timedelta = Field(
        ...,
        description="Time between entry and exit (must be positive)",
    )

    # Optional fields
    direction: Literal["long", "short"] | None = Field(
        None,
        description="Trade direction (long=buy then sell, short=sell then buy)",
    )
    metadata: dict[str, Any] | None = Field(
        None,
        description="Arbitrary metadata (signals, regime info, stop loss triggers, etc.)",
    )
    regime_info: dict[str, str] | None = Field(
        None,
        description="Market regime at trade time (e.g., {'volatility': 'high', 'trend': 'up'})",
    )
    quantity: float | None = Field(
        None,
        gt=0.0,
        description="Position size (number of shares/contracts/coins)",
    )
    entry_timestamp: datetime | None = Field(
        None,
        description="Position entry timestamp (if available)",
    )
    fees: float | None = Field(
        None,
        ge=0.0,
        description="Total transaction fees (commissions + exchange fees)",
    )
    slippage: float | None = Field(
        None,
        ge=0.0,
        description="Estimated or actual slippage cost",
    )

    @field_validator("duration")
    @classmethod
    def validate_duration_positive(cls, v: timedelta) -> timedelta:
        """Ensure duration is positive."""
        if v.total_seconds() <= 0:
            raise ValueError(f"Duration must be positive, got {v}")
        return v

    @model_validator(mode="after")
    def validate_pnl_consistency(self) -> TradeRecord:
        """Validate PnL is consistent with prices and direction.

        For trades with known direction and quantity, verify that the PnL
        calculation matches the price difference.

        Allows for small discrepancies due to fees and slippage.
        """
        if self.direction is None or self.quantity is None:
            # Cannot validate without direction and quantity
            return self

        # Calculate expected PnL from price difference
        price_diff = self.exit_price - self.entry_price

        if self.direction == "long":
            expected_pnl = price_diff * self.quantity
        else:  # short
            expected_pnl = -price_diff * self.quantity

        # Account for fees and slippage
        total_costs = (self.fees or 0.0) + (self.slippage or 0.0)
        expected_pnl -= total_costs

        # Allow 1% tolerance for rounding and other small discrepancies
        tolerance = abs(expected_pnl) * 0.01 + 0.01  # Minimum 1 cent tolerance

        actual_diff = abs(self.pnl - expected_pnl)
        if actual_diff > tolerance:
            raise ValueError(
                f"PnL inconsistent with prices. "
                f"Expected ~{expected_pnl:.2f} (from prices), got {self.pnl:.2f}. "
                f"Difference: {actual_diff:.2f}, tolerance: {tolerance:.2f}. "
                f"Check direction, quantity, fees, or slippage."
            )

        return self

    @model_validator(mode="after")
    def validate_timestamps(self) -> TradeRecord:
        """Validate timestamp ordering if entry_timestamp provided."""
        if self.entry_timestamp is not None:
            if self.entry_timestamp >= self.timestamp:
                raise ValueError(
                    f"Entry timestamp ({self.entry_timestamp}) must be before exit timestamp ({self.timestamp})"
                )

            # Verify duration matches timestamps
            calculated_duration = self.timestamp - self.entry_timestamp
            # Allow 1 second tolerance for rounding
            if abs((calculated_duration - self.duration).total_seconds()) > 1.0:
                raise ValueError(
                    f"Duration ({self.duration}) inconsistent with timestamps. "
                    f"Calculated: {calculated_duration} from entry/exit timestamps."
                )

        return self

    def to_dict(self) -> dict[str, Any]:
        """Export to dictionary format for storage.

        Returns:
            Dictionary with all trade data, suitable for JSON serialization

        Example:
            >>> trade.to_dict()
            {
                'timestamp': '2024-01-15T10:30:00',
                'symbol': 'AAPL',
                'entry_price': 150.0,
                'exit_price': 155.0,
                'pnl': 500.0,
                'duration': 432000.0,  # seconds
                'direction': 'long',
                ...
            }
        """
        data = self.model_dump(mode="json")
        # Convert timedelta to total seconds for JSON compatibility
        if "duration" in data:
            data["duration"] = self.duration.total_seconds()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TradeRecord:
        """Create TradeRecord from dictionary.

        Args:
            data: Dictionary with trade data (from to_dict() or ML4T Backtest)

        Returns:
            TradeRecord instance

        Example:
            >>> data = {
            ...     'timestamp': '2024-01-15T10:30:00',
            ...     'symbol': 'AAPL',
            ...     'entry_price': 150.0,
            ...     'exit_price': 155.0,
            ...     'pnl': 500.0,
            ...     'duration': 432000.0  # seconds
            ... }
            >>> trade = TradeRecord.from_dict(data)
        """
        # Convert duration from seconds if needed
        if "duration" in data and isinstance(data["duration"], int | float):
            data["duration"] = timedelta(seconds=data["duration"])
        return cls(**data)


class StrategyMetadata(BaseModel):
    """Metadata about the strategy being evaluated.

    This provides context for ML4T Backtest to track evaluations across
    different versions, environments, and time periods.

    Attributes:
        strategy_id: Unique strategy identifier (e.g., "momentum_v1")
        version: Strategy version (e.g., "1.2.3")
        environment: Execution environment (backtest/paper/live)
        start_date: Evaluation period start
        end_date: Evaluation period end
        config_hash: Hash of strategy configuration for reproducibility
        description: Optional human-readable description

    Example:
        >>> metadata = StrategyMetadata(
        ...     strategy_id="momentum_rsi",
        ...     version="1.0.0",
        ...     environment=EnvironmentType.BACKTEST,
        ...     start_date=datetime(2020, 1, 1),
        ...     end_date=datetime(2023, 12, 31)
        ... )
    """

    strategy_id: str = Field(..., description="Unique strategy identifier")
    version: str | None = Field(None, description="Strategy version (semver)")
    environment: EnvironmentType = Field(..., description="Execution environment")
    start_date: datetime = Field(..., description="Evaluation period start")
    end_date: datetime = Field(..., description="Evaluation period end")
    config_hash: str | None = Field(None, description="Strategy config hash for reproducibility")
    description: str | None = Field(None, description="Human-readable description")
    tags: dict[str, str] | None = Field(
        None, description="Optional tags (e.g., {'asset_class': 'crypto'})"
    )


class EvaluationExport(BaseModel):
    """Complete evaluation results for ML4T Backtest storage.

    This is the primary export format for storing ML4T Diagnostic results in
    ML4T Backtest's database. Contains all metrics, metadata, and diagnostics.

    Attributes:
        metadata: Strategy metadata (ID, version, environment)
        metrics: Core performance metrics (Sharpe, CAGR, drawdown, etc.)
        diagnostics: Optional diagnostic results (stationarity, correlation, etc.)
        sharpe_framework: Optional enhanced Sharpe results (PSR, DSR, etc.)
        timestamp: Evaluation timestamp (UTC)
        diagnostic_version: ML4T Diagnostic library version for compatibility tracking

    Example:
        >>> export = EvaluationExport(
        ...     metadata=metadata,
        ...     metrics={
        ...         "sharpe_ratio": 1.85,
        ...         "cagr": 0.24,
        ...         "max_drawdown": -0.18
        ...     },
        ...     timestamp=datetime.utcnow()
        ... )
        >>> backtest_engine.store_evaluation(export.to_dict())
    """

    metadata: StrategyMetadata = Field(..., description="Strategy metadata")
    metrics: dict[str, float] = Field(..., description="Core performance metrics")
    diagnostics: dict[str, dict] | None = Field(None, description="Optional diagnostic results")
    sharpe_framework: dict[str, float] | None = Field(
        None, description="Enhanced Sharpe results (PSR, DSR, MinTRL)"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(tz=UTC),
        description="Evaluation timestamp (UTC)",
    )
    diagnostic_version: str | None = Field(
        None, description="ML4T Diagnostic version for compatibility"
    )

    def to_dict(self) -> dict:
        """Export to ML4T Backtest-compatible dictionary format.

        Returns dictionary suitable for JSON serialization and storage
        in ML4T Backtest's database.

        Returns:
            Dictionary with all evaluation data

        Example:
            >>> export.to_dict()
            {
                'metadata': {
                    'strategy_id': 'momentum_v1',
                    'environment': 'backtest',
                    ...
                },
                'metrics': {...},
                'timestamp': '2024-11-03T12:00:00Z'
            }
        """
        return self.model_dump(mode="json")

    def to_json(self) -> str:
        """Export to JSON string for storage.

        Returns:
            JSON string representation

        Example:
            >>> json_str = export.to_json()
            >>> # Store in database or file
            >>> with open('evaluation.json', 'w') as f:
            ...     f.write(json_str)
        """
        return self.model_dump_json(indent=2)


class ComparisonRequest(BaseModel):
    """Request for comparing performance across environments.

    Used for Bayesian comparison of live vs backtest, or paper vs backtest.
    ML4T Diagnostic uses this to determine if live performance matches expectations.

    Attributes:
        strategy_id: Strategy being compared
        backtest_export: Backtest evaluation results
        live_export: Live/paper evaluation results
        comparison_type: Type of statistical comparison
        confidence_level: Confidence level for tests (default: 0.95)
        hypothesis: Hypothesis being tested

    Example:
        >>> request = ComparisonRequest(
        ...     strategy_id="momentum_v1",
        ...     backtest_export=backtest_results,
        ...     live_export=live_results,
        ...     comparison_type=ComparisonType.BAYESIAN,
        ...     hypothesis="live >= backtest"
        ... )
    """

    strategy_id: str = Field(..., description="Strategy identifier")
    backtest_export: EvaluationExport = Field(..., description="Backtest evaluation")
    live_export: EvaluationExport = Field(..., description="Live/paper evaluation")
    comparison_type: ComparisonType = Field(
        ComparisonType.BAYESIAN, description="Type of comparison"
    )
    confidence_level: float = Field(0.95, ge=0.5, le=0.99, description="Confidence level")
    hypothesis: str | None = Field(None, description="Hypothesis (e.g., 'live >= backtest')")


class ComparisonResult(BaseModel):
    """Result of live vs backtest comparison.

    Contains statistical evidence for whether live performance matches
    backtest expectations. Used for paper-to-live promotion decisions.

    Attributes:
        strategy_id: Strategy being compared
        comparison_type: Type of comparison performed
        decision: Recommendation (PROMOTE, REJECT, UNCERTAIN)
        confidence: Confidence in decision [0.0, 1.0]
        metrics_comparison: Comparison of key metrics
        statistical_tests: Statistical test results
        bayesian_evidence: Optional Bayesian evidence (if Bayesian comparison)
        recommendation: Human-readable recommendation
        timestamp: Comparison timestamp

    Example:
        >>> result = ComparisonResult(
        ...     strategy_id="momentum_v1",
        ...     comparison_type=ComparisonType.BAYESIAN,
        ...     decision="PROMOTE",
        ...     confidence=0.92,
        ...     metrics_comparison={
        ...         "sharpe_ratio": {"backtest": 1.85, "live": 1.72, "diff": -0.13}
        ...     },
        ...     recommendation="Live performance consistent with backtest"
        ... )
    """

    strategy_id: str = Field(..., description="Strategy identifier")
    comparison_type: ComparisonType = Field(..., description="Comparison type")
    decision: str = Field(..., description="Decision (PROMOTE, REJECT, UNCERTAIN, MONITOR)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in decision")
    metrics_comparison: dict[str, dict[str, float]] = Field(
        ..., description="Comparison of metrics (backtest vs live)"
    )
    statistical_tests: dict[str, dict] = Field(..., description="Statistical test results")
    bayesian_evidence: dict[str, float] | None = Field(
        None, description="Bayesian evidence (BF, posterior prob)"
    )
    recommendation: str = Field(..., description="Human-readable recommendation")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(tz=UTC), description="Comparison timestamp"
    )
    warnings: list[str] | None = Field(None, description="Optional warnings")

    def to_dict(self) -> dict:
        """Export to dictionary format.

        Returns:
            Dictionary with comparison results

        Example:
            >>> result.to_dict()
            {
                'strategy_id': 'momentum_v1',
                'decision': 'PROMOTE',
                'confidence': 0.92,
                ...
            }
        """
        return self.model_dump(mode="json")

    def summary(self) -> str:
        """Human-readable summary of comparison.

        Returns:
            Formatted summary string

        Example:
            >>> print(result.summary())
            Strategy Comparison: momentum_v1
            ================================
            Decision: PROMOTE (confidence: 0.92)

            Metrics Comparison:
              Sharpe Ratio: 1.85 (BT) → 1.72 (Live) [Δ=-0.13]

            Recommendation: Live performance consistent with backtest
        """
        lines = [f"Strategy Comparison: {self.strategy_id}", "=" * 50]
        lines.append(f"Decision: {self.decision} (confidence: {self.confidence:.2f})")
        lines.append("")

        # Metrics comparison
        lines.append("Metrics Comparison:")
        for metric, values in self.metrics_comparison.items():
            bt = values.get("backtest", 0)
            live = values.get("live", 0)
            diff = values.get("diff", 0)
            metric_name = metric.replace("_", " ").title()
            lines.append(f"  {metric_name}: {bt:.3f} (BT) → {live:.3f} (Live) [Δ={diff:+.3f}]")

        lines.append("")
        lines.append(f"Recommendation: {self.recommendation}")

        # Warnings
        if self.warnings:
            lines.append("")
            lines.append("Warnings:")
            for warning in self.warnings:
                lines.append(f"  ⚠️  {warning}")

        return "\n".join(lines)


class PromotionWorkflow(BaseModel):
    """Paper-to-live promotion workflow configuration.

    Defines the criteria and process for promoting a strategy from
    paper trading to live trading based on evaluation results.

    Attributes:
        strategy_id: Strategy being promoted
        paper_duration_days: Minimum paper trading duration
        promotion_criteria: Required conditions for promotion
        approval_required: Whether human approval is needed
        risk_limits: Risk limits for live trading

    Example:
        >>> workflow = PromotionWorkflow(
        ...     strategy_id="momentum_v1",
        ...     paper_duration_days=30,
        ...     promotion_criteria={
        ...         "min_sharpe": 1.5,
        ...         "max_drawdown": -0.15,
        ...         "min_trades": 100,
        ...         "bayesian_confidence": 0.90
        ...     },
        ...     approval_required=True
        ... )
    """

    strategy_id: str = Field(..., description="Strategy identifier")
    paper_duration_days: int = Field(..., ge=1, description="Minimum paper trading days")
    promotion_criteria: dict[str, float] = Field(
        ..., description="Required conditions for promotion"
    )
    approval_required: bool = Field(True, description="Whether human approval needed")
    risk_limits: dict[str, float] | None = Field(None, description="Risk limits for live trading")

    def evaluate_promotion(self, comparison_result: ComparisonResult) -> bool:
        """Evaluate if promotion criteria are met.

        Args:
            comparison_result: Result of paper vs backtest comparison

        Returns:
            True if promotion criteria satisfied

        Example:
            >>> workflow.evaluate_promotion(comparison_result)
            True  # Ready for promotion
        """
        # Check decision
        if comparison_result.decision != "PROMOTE":
            return False

        # Check confidence
        min_confidence = self.promotion_criteria.get("bayesian_confidence", 0.9)
        if comparison_result.confidence < min_confidence:
            return False

        # Check metrics
        for metric, threshold in self.promotion_criteria.items():
            if metric in comparison_result.metrics_comparison:
                comparison_result.metrics_comparison[metric].get("live", 0)
                if metric.startswith("min_"):
                    metric_name = metric[4:]  # Remove 'min_' prefix
                    if metric_name in comparison_result.metrics_comparison and (
                        comparison_result.metrics_comparison[metric_name]["live"] < threshold
                    ):
                        return False
                elif metric.startswith("max_"):
                    metric_name = metric[4:]  # Remove 'max_' prefix
                    if metric_name in comparison_result.metrics_comparison and (
                        comparison_result.metrics_comparison[metric_name]["live"] > threshold
                    ):
                        return False

        return True
