"""Time-series cross-validation splitters with purging and embargo support.

This module provides advanced cross-validation methods designed specifically for
financial time-series data, addressing common issues like data leakage and
backtest overfitting.
"""

from ml4t.diagnostic.splitters.base import BaseSplitter
from ml4t.diagnostic.splitters.combinatorial import CombinatorialPurgedCV
from ml4t.diagnostic.splitters.config import (
    CombinatorialPurgedConfig,
    PurgedWalkForwardConfig,
    SplitterConfig,
)
from ml4t.diagnostic.splitters.persistence import (
    load_config,
    load_folds,
    save_config,
    save_folds,
    verify_folds,
)
from ml4t.diagnostic.splitters.walk_forward import PurgedWalkForwardCV

__all__ = [
    "BaseSplitter",
    "CombinatorialPurgedCV",
    "CombinatorialPurgedConfig",
    "PurgedWalkForwardCV",
    "PurgedWalkForwardConfig",
    "SplitterConfig",
    "load_config",
    "load_folds",
    "save_config",
    "save_folds",
    "verify_folds",
]
