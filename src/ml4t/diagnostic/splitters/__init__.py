"""Time-series cross-validation splitters for financial data.

This module provides cross-validation methods designed specifically for
financial time-series data, addressing common issues like data leakage and
backtest overfitting.
"""

from ml4t.diagnostic.splitters.base import BaseSplitter
from ml4t.diagnostic.splitters.combinatorial import CombinatorialCV
from ml4t.diagnostic.splitters.config import (
    CombinatorialConfig,
    SplitterConfig,
    WalkForwardConfig,
)
from ml4t.diagnostic.splitters.persistence import (
    load_config,
    load_folds,
    save_config,
    save_folds,
    verify_folds,
)
from ml4t.diagnostic.splitters.walk_forward import WalkForwardCV

__all__ = [
    "BaseSplitter",
    "CombinatorialCV",
    "CombinatorialConfig",
    "WalkForwardCV",
    "WalkForwardConfig",
    "SplitterConfig",
    "load_config",
    "load_folds",
    "save_config",
    "save_folds",
    "verify_folds",
]
