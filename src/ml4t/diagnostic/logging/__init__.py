"""
ML4T Diagnostic Logging and Debugging Infrastructure

Provides structured logging with levels, progress tracking, debug mode,
and performance metrics for ML4T Diagnostic library operations.

Features:
- Structured JSON logging
- Configurable log levels (DEBUG, INFO, WARNING, ERROR)
- Progress indicators for long-running operations
- Debug mode for intermediate results
- Performance metrics tracking
- Context-aware logging
- Weights & Biases integration

Example:
    >>> from ml4t.diagnostic.logging import get_logger, set_log_level, LogLevel
    >>> logger = get_logger(__name__)
    >>> set_log_level(LogLevel.DEBUG)
    >>> logger.info("Computing Sharpe ratio", n_samples=100)
    >>> with logger.timed("sharpe_computation"):
    ...     sharpe = compute_sharpe_ratio(returns)
"""

# Structured logging
from ml4t.diagnostic.logging.logger import (
    LogLevel,
    QEvalLogger,
    configure_logging,
    get_log_level,
    get_logger,
    set_log_level,
)

# Performance metrics
from ml4t.diagnostic.logging.performance import (
    PerformanceMonitor,
    PerformanceTracker,
    get_performance_monitor,
    measure_time,
    timed,
)

# Progress tracking
from ml4t.diagnostic.logging.progress import (
    ProgressBar,
    ProgressTracker,
    progress_indicator,
    spinner,
)

# Experiment tracking (Weights & Biases)
from ml4t.diagnostic.logging.wandb import WandbLogger, log_experiment

__all__: list[str] = [
    # Logger
    "QEvalLogger",
    "get_logger",
    "set_log_level",
    "get_log_level",
    "configure_logging",
    "LogLevel",
    # Progress
    "ProgressBar",
    "progress_indicator",
    "ProgressTracker",
    "spinner",
    # Performance
    "PerformanceTracker",
    "PerformanceMonitor",
    "get_performance_monitor",
    "timed",
    "measure_time",
    # WandB
    "WandbLogger",
    "log_experiment",
]
