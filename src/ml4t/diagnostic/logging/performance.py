"""
Performance Metrics Tracking

Provides timing and performance measurement utilities.
"""

import time
from contextlib import contextmanager
from functools import wraps
from typing import Any


class PerformanceTracker:
    """
    Context manager for timing operations.

    Example:
        >>> with PerformanceTracker("sharpe_computation") as tracker:
        ...     result = compute_sharpe_ratio(returns)
        >>> print(f"Elapsed: {tracker.elapsed:.3f}s")
    """

    def __init__(self, operation: str, logger: Any | None = None):
        """
        Initialize performance tracker.

        Args:
            operation: Operation name
            logger: Optional logger to record timing
        """
        self.operation = operation
        self.logger = logger
        self.start_time: float | None = None
        self.end_time: float | None = None

    def __enter__(self):
        """Start timing."""
        self.start_time = time.time()
        if self.logger:
            self.logger.debug(f"Starting {self.operation}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop timing and log result."""
        self.end_time = time.time()

        if self.logger:
            if exc_type is None:
                self.logger.info(f"Completed {self.operation}", elapsed_seconds=self.elapsed)
            else:
                self.logger.error(
                    f"Failed {self.operation}", elapsed_seconds=self.elapsed, error=str(exc_val)
                )

    @property
    def elapsed(self) -> float:
        """
        Get elapsed time in seconds.

        Returns:
            Elapsed time (0 if not complete)
        """
        if self.start_time is None:
            return 0.0

        end = self.end_time if self.end_time else time.time()
        return end - self.start_time


def timed(func):
    """
    Decorator to time function execution.

    Args:
        func: Function to time

    Returns:
        Wrapped function that logs execution time

    Example:
        >>> @timed
        ... def compute_sharpe(returns):
        ...     return returns.mean() / returns.std()
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        try:
            result = func(*args, **kwargs)
            elapsed = time.time() - start
            print(f"{func.__name__}: {elapsed:.3f}s", flush=True)
            return result
        except Exception as e:
            elapsed = time.time() - start
            print(f"{func.__name__} failed after {elapsed:.3f}s: {e}", flush=True)
            raise

    return wrapper


class PerformanceMonitor:
    """
    Monitor and aggregate performance metrics.

    Example:
        >>> monitor = PerformanceMonitor()
        >>> with monitor.track("operation1"):
        ...     do_work()
        >>> with monitor.track("operation2"):
        ...     do_more_work()
        >>> print(monitor.summary())
    """

    def __init__(self):
        """Initialize performance monitor."""
        self.metrics: dict[str, list[float]] = {}

    def track(self, operation: str):
        """
        Track operation timing.

        Args:
            operation: Operation name

        Returns:
            Context manager for timing

        Example:
            >>> with monitor.track("compute"):
            ...     result = expensive_computation()
        """
        return _MonitoredOperation(self, operation)

    def record(self, operation: str, elapsed: float):
        """
        Record timing for operation.

        Args:
            operation: Operation name
            elapsed: Elapsed time in seconds
        """
        if operation not in self.metrics:
            self.metrics[operation] = []
        self.metrics[operation].append(elapsed)

    def summary(self) -> dict[str, dict[str, float]]:
        """
        Get summary statistics for all operations.

        Returns:
            Dict mapping operation name to statistics
                (count, total, mean, min, max)

        Example:
            >>> stats = monitor.summary()
            >>> print(stats["compute"]["mean"])
            0.523
        """
        summary = {}

        for operation, timings in self.metrics.items():
            if timings:
                summary[operation] = {
                    "count": len(timings),
                    "total": sum(timings),
                    "mean": sum(timings) / len(timings),
                    "min": min(timings),
                    "max": max(timings),
                }

        return summary

    def reset(self):
        """Clear all metrics."""
        self.metrics.clear()


class _MonitoredOperation:
    """Internal context manager for PerformanceMonitor."""

    def __init__(self, monitor: PerformanceMonitor, operation: str):
        self.monitor = monitor
        self.operation = operation
        self.start_time: float | None = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            elapsed = time.time() - self.start_time
            self.monitor.record(self.operation, elapsed)


# Global performance monitor instance
_global_monitor = PerformanceMonitor()


def get_performance_monitor() -> PerformanceMonitor:
    """
    Get global performance monitor.

    Returns:
        Global PerformanceMonitor instance

    Example:
        >>> monitor = get_performance_monitor()
        >>> with monitor.track("operation"):
        ...     do_work()
    """
    return _global_monitor


@contextmanager
def measure_time(operation: str):
    """
    Context manager to measure and print operation time.

    Args:
        operation: Operation name

    Example:
        >>> with measure_time("data_loading"):
        ...     data = load_large_dataset()
        data_loading: 2.345s
    """
    start = time.time()
    try:
        yield
    finally:
        elapsed = time.time() - start
        print(f"{operation}: {elapsed:.3f}s", flush=True)
