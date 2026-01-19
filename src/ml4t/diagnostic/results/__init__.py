"""Type-safe Pydantic result schemas for all evaluation outputs.

This module provides structured result schemas for:
- Feature diagnostics (Module A)
- Cross-feature analysis (Module B)
- Feature-outcome relationships (Module C)
- Portfolio evaluation (Module D)
- Sharpe framework analysis

All results support:
- JSON serialization
- DataFrame conversion
- Human-readable summaries
- Type-safe access via Pydantic validation
"""

from ml4t.diagnostic.results.barrier_results import (
    BarrierTearSheet,
    HitRateResult,
    PrecisionRecallResult,
    ProfitFactorResult,
    TimeToTargetResult,
)
from ml4t.diagnostic.results.base import BaseResult
from ml4t.diagnostic.results.event_results import (
    AbnormalReturnResult,
    EventStudyResult,
)
from ml4t.diagnostic.results.feature_results import (
    ACFResult,
    CrossFeatureResult,
    FeatureDiagnosticsResult,
    FeatureOutcomeResult,
    ICAnalysisResult,
    StationarityTestResult,
    ThresholdAnalysisResult,
)
from ml4t.diagnostic.results.multi_signal_results import (
    ComparisonResult,
    MultiSignalSummary,
)
from ml4t.diagnostic.results.portfolio_results import (
    BayesianComparisonResult,
    PortfolioEvaluationResult,
    PortfolioMetrics,
)
from ml4t.diagnostic.results.sharpe_results import (
    DSRResult,
    FDRResult,
    MinTRLResult,
    PSRResult,
    SharpeFrameworkResult,
)
from ml4t.diagnostic.results.signal_results import (
    IRtcResult,
    QuantileAnalysisResult,
    RASICResult,
    SignalICResult,
    SignalTearSheet,
    TurnoverAnalysisResult,
)

__all__ = [
    # Base
    "BaseResult",
    # Feature diagnostics (Module A)
    "StationarityTestResult",
    "ACFResult",
    "FeatureDiagnosticsResult",
    # Cross-feature (Module B)
    "CrossFeatureResult",
    # Feature-outcome (Module C)
    "ICAnalysisResult",
    "ThresholdAnalysisResult",
    "FeatureOutcomeResult",
    # Portfolio (Module D)
    "PortfolioMetrics",
    "BayesianComparisonResult",
    "PortfolioEvaluationResult",
    # Sharpe framework
    "PSRResult",
    "MinTRLResult",
    "DSRResult",
    "FDRResult",
    "SharpeFrameworkResult",
    # Signal analysis
    "SignalICResult",
    "RASICResult",
    "QuantileAnalysisResult",
    "TurnoverAnalysisResult",
    "IRtcResult",
    "SignalTearSheet",
    # Event study
    "AbnormalReturnResult",
    "EventStudyResult",
    # Multi-signal analysis
    "MultiSignalSummary",
    "ComparisonResult",
    # Barrier analysis
    "HitRateResult",
    "ProfitFactorResult",
    "PrecisionRecallResult",
    "TimeToTargetResult",
    "BarrierTearSheet",
]
