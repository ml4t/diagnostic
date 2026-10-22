"""Signal analysis for factor/alpha evaluation.

This module provides tools for analyzing the predictive power of signals
(factors) for future returns.

Main Entry Point
----------------
analyze_signal : Compute IC, quantile returns, spread, and turnover
    for a factor signal. This is the recommended way to use this module.

Example
-------
>>> from ml4t.diagnostic.signal import analyze_signal
>>> result = analyze_signal(factor_df, prices_df)
>>> print(result.summary())
>>> result.to_json("results.json")

Building Blocks
---------------
For custom workflows, use the component functions:

- prepare_data : Join factor with prices and compute forward returns
- extract_signal_ic_series : Extract per-date IC values for one horizon
- compute_quantile_returns : Compute returns by quantile
- compute_turnover : Compute factor turnover rate
- filter_outliers : Remove cross-sectional outliers
- quantize_factor : Assign quantile labels
"""

from ml4t.diagnostic.signal._utils import (
    QuantileMethod,
    filter_outliers,
    quantize_factor,
)
from ml4t.diagnostic.signal.core import analyze_prepared_signal, analyze_signal, prepare_data
from ml4t.diagnostic.signal.event_study import SignalEventStudyResult, analyze_signal_event_study
from ml4t.diagnostic.signal.events import (
    AlignedEventResult,
    EventReturnsResult,
    align_event_paths,
    analyze_event_returns,
    analyze_signal_events,
)
from ml4t.diagnostic.signal.factor_regression import (
    FactorAlphaBetaResult,
    compute_factor_alpha_beta,
)
from ml4t.diagnostic.signal.panel import (
    MaxLossExceededError,
    PreparedSignalPanel,
    prepare_signal_panel,
)
from ml4t.diagnostic.signal.portfolio import (
    compute_factor_cumulative_returns,
    compute_factor_returns,
    compute_factor_weights,
)
from ml4t.diagnostic.signal.quantile import (
    compute_quantile_returns,
    compute_quantile_spread,
    compute_quantile_statistics,
    compute_spread,
    monotonicity_score,
)
from ml4t.diagnostic.signal.result import SignalResult
from ml4t.diagnostic.signal.returns import demean_signal_returns
from ml4t.diagnostic.signal.signal_ic import (
    compute_ic_summary,
    compute_signal_ic,
    extract_signal_ic_series,
)
from ml4t.diagnostic.signal.turnover import (
    compute_autocorrelation,
    compute_quantile_turnover,
    compute_rank_autocorrelation,
    compute_turnover,
    estimate_half_life,
)

__all__ = [
    # Main entry point
    "analyze_signal",
    "analyze_prepared_signal",
    "SignalResult",
    "AlignedEventResult",
    "EventReturnsResult",
    "analyze_event_returns",
    "SignalEventStudyResult",
    "analyze_signal_event_study",
    "align_event_paths",
    "analyze_signal_events",
    # Data preparation
    "prepare_data",
    "prepare_signal_panel",
    "PreparedSignalPanel",
    "MaxLossExceededError",
    "compute_factor_weights",
    "compute_factor_returns",
    "compute_factor_cumulative_returns",
    "compute_factor_alpha_beta",
    "FactorAlphaBetaResult",
    "filter_outliers",
    "quantize_factor",
    "QuantileMethod",
    # IC functions
    "extract_signal_ic_series",
    "compute_ic_summary",
    "compute_signal_ic",
    "demean_signal_returns",
    # Quantile functions
    "compute_quantile_returns",
    "compute_quantile_statistics",
    "compute_quantile_spread",
    "compute_spread",
    "monotonicity_score",
    # Turnover functions
    "compute_turnover",
    "compute_quantile_turnover",
    "compute_rank_autocorrelation",
    "compute_autocorrelation",
    "estimate_half_life",
]
