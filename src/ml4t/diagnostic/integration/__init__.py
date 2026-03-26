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
from ml4t.diagnostic.integration.report_metadata import BacktestReportMetadata

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


def analyze_backtest_result(
    result: BacktestResult,
    calendar: str | None = None,
    benchmark: Any = None,
    confidence_intervals: bool = False,
):
    """Build a lazy BacktestProfile from a BacktestResult."""
    return _load_backtest_bridge().analyze_backtest_result(
        result=result,
        calendar=calendar,
        benchmark=benchmark,
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
    benchmark: Any = None,
    benchmark_name: str = "Benchmark",
    report_metadata: BacktestReportMetadata | None = None,
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
        benchmark=benchmark,
        benchmark_name=benchmark_name,
        report_metadata=report_metadata,
    )


def profile_from_run_artifacts(
    backtest_dir: str | Path,
    predictions_path: str | Path | None = None,
    signals_path: str | Path | None = None,
    calendar: str | None = None,
    benchmark: Any = None,
    confidence_intervals: bool = False,
):
    """Build a BacktestProfile from case-study artifact directories."""
    return _load_backtest_bridge().profile_from_run_artifacts(
        backtest_dir=backtest_dir,
        predictions_path=predictions_path,
        signals_path=signals_path,
        calendar=calendar,
        benchmark=benchmark,
        confidence_intervals=confidence_intervals,
    )


def generate_tearsheet_from_run_artifacts(
    backtest_dir: str | Path,
    template: Literal["quant_trader", "hedge_fund", "risk_manager", "full"] = "full",
    theme: Literal["default", "dark", "print", "presentation"] = "default",
    output_path: str | Path | None = None,
    predictions_path: str | Path | None = None,
    signals_path: str | Path | None = None,
    calendar: str | None = None,
    benchmark: Any = None,
    benchmark_name: str = "Benchmark",
    report_metadata: BacktestReportMetadata | None = None,
) -> str:
    """Generate a tearsheet directly from case-study artifact directories."""
    return _load_backtest_bridge().generate_tearsheet_from_run_artifacts(
        backtest_dir=backtest_dir,
        predictions_path=predictions_path,
        signals_path=signals_path,
        template=template,
        theme=theme,
        output_path=output_path,
        calendar=calendar,
        benchmark=benchmark,
        benchmark_name=benchmark_name,
        report_metadata=report_metadata,
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
    "analyze_backtest_result",
    "compute_metrics_from_result",
    "EnvironmentType",
    "EvaluationExport",
    "generate_tearsheet_from_result",
    "generate_tearsheet_from_run_artifacts",
    "portfolio_analysis_from_result",
    "profile_from_run_artifacts",
    "PromotionWorkflow",
    "BacktestReportMetadata",
    "StrategyMetadata",
    "TradeRecord",
]
