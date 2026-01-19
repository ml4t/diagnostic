"""Dashboard data types and configuration.

Provides unified data structures for the dashboard to eliminate dict/object branching.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class DashboardConfig:
    """Configuration for the dashboard.

    Attributes
    ----------
    allow_pickle_upload : bool
        Whether to allow uploading pickle files. Disabled by default for security.
    styled : bool
        Whether to apply professional CSS styling.
    title : str
        Dashboard title.
    """

    allow_pickle_upload: bool = False  # Security: disabled by default
    styled: bool = False
    title: str = "Trade SHAP Diagnostics"


@dataclass
class DashboardBundle:
    """Unified data container for all dashboard tabs.

    This normalizes the varied input formats (dict vs object, different field names)
    into a single consistent representation that all tabs can consume.

    Attributes
    ----------
    trades_df : pd.DataFrame
        One row per trade with stable columns:
        - trade_id: str
        - entry_time: datetime
        - exit_time: datetime (optional)
        - pnl: float
        - return_pct: float (optional)
        - symbol: str (optional)
        Sorted chronologically by entry_time for time-series tests.
    returns : np.ndarray | None
        Trade returns array. Prefers return_pct if available, falls back to pnl.
    returns_label : str
        What the returns array represents: "return_pct", "pnl", or "none".
    explanations : list[dict]
        Normalized explanation dictionaries with stable keys:
        - trade_id: str
        - shap_vector: list[float]
        - top_features: list[tuple[str, float]]
        - trade_metrics: dict (optional)
    patterns_df : pd.DataFrame
        One row per error pattern with stable columns:
        - cluster_id: int
        - n_trades: int
        - description: str
        - top_features: list[tuple]
        - hypothesis: str (optional)
        - actions: list[str] (optional)
        - confidence: float (optional)
    n_trades_analyzed : int
        Total number of trades analyzed.
    n_trades_explained : int
        Number of trades successfully explained.
    n_trades_failed : int
        Number of trades that failed explanation.
    failed_trades : list[tuple[str, str]]
        List of (trade_id, reason) for failed explanations.
    config : DashboardConfig
        Dashboard configuration.
    """

    trades_df: pd.DataFrame
    returns: np.ndarray | None
    returns_label: str  # "return_pct" | "pnl" | "none"
    explanations: list[dict[str, Any]]
    patterns_df: pd.DataFrame
    n_trades_analyzed: int
    n_trades_explained: int
    n_trades_failed: int
    failed_trades: list[tuple[str, str]]
    config: DashboardConfig = field(default_factory=DashboardConfig)


@dataclass
class ReturnSummary:
    """Summary statistics for a returns series.

    Attributes
    ----------
    n_samples : int
        Number of samples.
    mean : float
        Mean return.
    std : float
        Standard deviation.
    sharpe : float
        Sharpe ratio (mean / std).
    skewness : float
        Skewness of distribution.
    kurtosis : float
        Kurtosis of distribution (not excess, 3.0 for normal).
    min_val : float
        Minimum value.
    max_val : float
        Maximum value.
    win_rate : float
        Fraction of positive returns.
    """

    n_samples: int
    mean: float
    std: float
    sharpe: float
    skewness: float
    kurtosis: float
    min_val: float
    max_val: float
    win_rate: float
