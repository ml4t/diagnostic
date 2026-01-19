"""
Structured Logging for ML4T Diagnostic

Provides configurable logging with levels, JSON output, and context preservation.
"""

import json
import logging
import sys
from datetime import datetime
from enum import Enum


class LogLevel(str, Enum):
    """Log level enumeration."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


class QEvalLogger:
    """
    Structured logger for ML4T Diagnostic operations.

    Provides:
    - Configurable log levels
    - JSON-structured output
    - Context preservation
    - Performance timing
    - Debug mode support

    Example:
        >>> logger = QEvalLogger("mlquant.evaluation.metrics")
        >>> logger.info("Computing metric", metric="sharpe_ratio", n_samples=100)
        >>> logger.debug("Intermediate result", value=0.5)
        >>> logger.error("Computation failed", error="division by zero")
    """

    def __init__(self, name: str, level: LogLevel = LogLevel.INFO, output_json: bool = False):
        """
        Initialize logger.

        Args:
            name: Logger name (usually module name)
            level: Minimum log level to display
            output_json: Whether to output JSON format
        """
        self.name = name
        self.level = level
        self.output_json = output_json
        self._python_logger = logging.getLogger(name)
        self._configure_python_logger()

    def _configure_python_logger(self):
        """Configure underlying Python logger."""
        # Map our levels to Python logging levels
        level_map = {
            LogLevel.DEBUG: logging.DEBUG,
            LogLevel.INFO: logging.INFO,
            LogLevel.WARNING: logging.WARNING,
            LogLevel.ERROR: logging.ERROR,
        }
        self._python_logger.setLevel(level_map[self.level])

        # Add handler if none exists
        if not self._python_logger.handlers:
            handler = logging.StreamHandler(sys.stderr)
            handler.setFormatter(
                logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            )
            self._python_logger.addHandler(handler)

    def _format_message(self, level: str, message: str, **context) -> str:
        """
        Format log message.

        Args:
            level: Log level
            message: Log message
            **context: Additional context fields

        Returns:
            Formatted message string
        """
        if self.output_json:
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "logger": self.name,
                "level": level,
                "message": message,
                **context,
            }
            return json.dumps(log_entry)
        else:
            # Human-readable format
            context_str = " ".join(f"{k}={v}" for k, v in context.items())
            if context_str:
                return f"{message} ({context_str})"
            return message

    def debug(self, message: str, **context):
        """
        Log debug message.

        Args:
            message: Debug message
            **context: Additional context
        """
        if self._should_log(LogLevel.DEBUG):
            formatted = self._format_message("DEBUG", message, **context)
            self._python_logger.debug(formatted)

    def info(self, message: str, **context):
        """
        Log info message.

        Args:
            message: Info message
            **context: Additional context
        """
        if self._should_log(LogLevel.INFO):
            formatted = self._format_message("INFO", message, **context)
            self._python_logger.info(formatted)

    def warning(self, message: str, **context):
        """
        Log warning message.

        Args:
            message: Warning message
            **context: Additional context
        """
        if self._should_log(LogLevel.WARNING):
            formatted = self._format_message("WARNING", message, **context)
            self._python_logger.warning(formatted)

    def error(self, message: str, **context):
        """
        Log error message.

        Args:
            message: Error message
            **context: Additional context
        """
        if self._should_log(LogLevel.ERROR):
            formatted = self._format_message("ERROR", message, **context)
            self._python_logger.error(formatted)

    def _should_log(self, level: LogLevel) -> bool:
        """Check if message should be logged at given level."""
        level_order = [LogLevel.DEBUG, LogLevel.INFO, LogLevel.WARNING, LogLevel.ERROR]
        return level_order.index(level) >= level_order.index(self.level)

    def timed(self, operation: str):
        """
        Context manager for timing operations.

        Args:
            operation: Operation name

        Example:
            >>> with logger.timed("compute_sharpe"):
            ...     result = compute_sharpe_ratio(returns)
        """
        from .performance import PerformanceTracker

        return PerformanceTracker(operation, logger=self)


# Global logger registry
_loggers: dict[str, QEvalLogger] = {}
_global_level: LogLevel = LogLevel.INFO
_global_json_output: bool = False


def get_logger(name: str) -> QEvalLogger:
    """
    Get or create logger for module.

    Args:
        name: Logger name (usually __name__)

    Returns:
        QEvalLogger instance

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Processing data")
    """
    if name not in _loggers:
        _loggers[name] = QEvalLogger(name, level=_global_level, output_json=_global_json_output)
    return _loggers[name]


def set_log_level(level: LogLevel):
    """
    Set global log level.

    Args:
        level: Minimum log level

    Example:
        >>> set_log_level(LogLevel.DEBUG)
    """
    global _global_level
    _global_level = level

    # Update existing loggers
    for logger in _loggers.values():
        logger.level = level
        logger._configure_python_logger()


def get_log_level() -> LogLevel:
    """
    Get current global log level.

    Returns:
        Current log level
    """
    return _global_level


def configure_logging(level: LogLevel = LogLevel.INFO, output_json: bool = False):
    """
    Configure global logging settings.

    Args:
        level: Minimum log level
        output_json: Whether to output JSON format

    Example:
        >>> configure_logging(LogLevel.DEBUG, output_json=True)
    """
    global _global_level, _global_json_output
    _global_level = level
    _global_json_output = output_json

    # Update existing loggers
    for logger in _loggers.values():
        logger.level = level
        logger.output_json = output_json
        logger._configure_python_logger()
