"""Integration contracts for external libraries."""

from __future__ import annotations

import importlib
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from ml4t.diagnostic.integration.backtest_contract import (
    ComparisonRequest,
    ComparisonResult,
    ComparisonType,
    EnvironmentType,
    EvaluationExport,
    PromotionWorkflow,
    StrategyMetadata,
    TradeRecord,
)
from ml4t.diagnostic.integration.data_contract import (
    AnomalyType,
    DataAnomaly,
    DataQualityMetrics,
    DataQualityReport,
    DataValidationRequest,
    Severity,
)
from ml4t.diagnostic.integration.engineer_contract import (
    EngineerConfig,
    PreprocessingRecommendation,
    TransformType,
)

if TYPE_CHECKING:
    from ml4t.backtest import BacktestResult
    from ml4t.diagnostic.evaluation import PortfolioAnalysis


@lru_cache(maxsize=1)
def _load_backtest_bridge():
    try:
        return importlib.import_module("ml4t.diagnostic.integration.backtest")
    except ImportError as exc:
        if getattr(exc, "name", None) in {"ml4t.backtest", "ml4t"}:
            raise ImportError(
                "ml4t-backtest integration requires the optional 'ml4t-backtest' package. "
                "Install with: pip install 'ml4t-diagnostic[backtest]'"
            ) from exc
        raise


def compute_metrics_from_result(
    result: BacktestResult,
    calendar: str | None = None,
    confidence_intervals: bool = False,
) -> dict[str, Any]:
    """Compute diagnostic metrics from a BacktestResult."""
    return _load_backtest_bridge().compute_metrics_from_result(
        result=result,
        calendar=calendar,
        confidence_intervals=confidence_intervals,
    )


def generate_tearsheet_from_result(
    result: BacktestResult,
    template: Literal["quant_trader", "hedge_fund", "risk_manager", "full"] = "full",
    theme: Literal["default", "dark", "print", "presentation"] = "default",
    title: str | None = None,
    output_path: str | Path | None = None,
    include_statistical: bool = True,
    calendar: str | None = None,
) -> str:
    """Generate a diagnostic tearsheet from a BacktestResult."""
    return _load_backtest_bridge().generate_tearsheet_from_result(
        result=result,
        template=template,
        theme=theme,
        title=title,
        output_path=output_path,
        include_statistical=include_statistical,
        calendar=calendar,
    )


def portfolio_analysis_from_result(
    result: BacktestResult,
    calendar: str | None = None,
    benchmark: Any = None,
) -> PortfolioAnalysis:
    """Create a PortfolioAnalysis from a BacktestResult."""
    return _load_backtest_bridge().portfolio_analysis_from_result(
        result=result,
        calendar=calendar,
        benchmark=benchmark,
    )


__all__ = [
    # ml4t.data integration
    "AnomalyType",
    "DataAnomaly",
    "DataQualityMetrics",
    "DataQualityReport",
    "DataValidationRequest",
    "Severity",
    # ml4t.engineer integration
    "PreprocessingRecommendation",
    "EngineerConfig",
    "TransformType",
    # ml4t.backtest integration
    "ComparisonRequest",
    "ComparisonResult",
    "ComparisonType",
    "compute_metrics_from_result",
    "EnvironmentType",
    "EvaluationExport",
    "generate_tearsheet_from_result",
    "portfolio_analysis_from_result",
    "PromotionWorkflow",
    "StrategyMetadata",
    "TradeRecord",
]
