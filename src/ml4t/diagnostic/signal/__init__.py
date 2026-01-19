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
- compute_ic_series : Compute IC time series
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
from ml4t.diagnostic.signal.core import analyze_signal, prepare_data
from ml4t.diagnostic.signal.quantile import (
    compute_monotonicity,
    compute_quantile_returns,
    compute_spread,
)
from ml4t.diagnostic.signal.result import SignalResult
from ml4t.diagnostic.signal.signal_ic import compute_ic_series, compute_ic_summary
from ml4t.diagnostic.signal.turnover import (
    compute_autocorrelation,
    compute_turnover,
    estimate_half_life,
)

__all__ = [
    # Main entry point
    "analyze_signal",
    "SignalResult",
    # Data preparation
    "prepare_data",
    "filter_outliers",
    "quantize_factor",
    "QuantileMethod",
    # IC functions
    "compute_ic_series",
    "compute_ic_summary",
    # Quantile functions
    "compute_quantile_returns",
    "compute_spread",
    "compute_monotonicity",
    # Turnover functions
    "compute_turnover",
    "compute_autocorrelation",
    "estimate_half_life",
]
