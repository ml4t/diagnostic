"""Tests for ml4t.diagnostic.logging module."""

import time

import pytest

from ml4t.diagnostic.logging import (
    LogLevel,
    PerformanceMonitor,
    PerformanceTracker,
    QEvalLogger,
    configure_logging,
    get_log_level,
    get_logger,
    get_performance_monitor,
    measure_time,
    set_log_level,
    timed,
)


class TestLogLevel:
    """Tests for LogLevel enum."""

    def test_log_levels_exist(self):
        """Test all expected log levels exist."""
        assert hasattr(LogLevel, "DEBUG")
        assert hasattr(LogLevel, "INFO")
        assert hasattr(LogLevel, "WARNING")
        assert hasattr(LogLevel, "ERROR")

    def test_log_level_values(self):
        """Test log level string values."""
        assert LogLevel.DEBUG.value == "DEBUG"
        assert LogLevel.INFO.value == "INFO"
        assert LogLevel.WARNING.value == "WARNING"
        assert LogLevel.ERROR.value == "ERROR"

    def test_log_level_is_string_enum(self):
        """Test that LogLevel inherits from str."""
        assert isinstance(LogLevel.DEBUG, str)
        assert LogLevel.INFO == "INFO"


class TestQEvalLogger:
    """Tests for QEvalLogger class."""

    def test_logger_creation(self):
        """Test logger can be created."""
        logger = QEvalLogger("test_module")
        assert logger.name == "test_module"
        assert logger.level == LogLevel.INFO

    def test_logger_with_custom_level(self):
        """Test logger with custom log level."""
        logger = QEvalLogger("test_module", level=LogLevel.DEBUG)
        assert logger.level == LogLevel.DEBUG

    def test_logger_with_json_output(self):
        """Test logger with JSON output enabled."""
        logger = QEvalLogger("test_module", output_json=True)
        assert logger.output_json is True

    def test_should_log_respects_level(self):
        """Test that _should_log respects level hierarchy."""
        logger = QEvalLogger("test", level=LogLevel.WARNING)

        # Should not log DEBUG or INFO
        assert logger._should_log(LogLevel.DEBUG) is False
        assert logger._should_log(LogLevel.INFO) is False

        # Should log WARNING and ERROR
        assert logger._should_log(LogLevel.WARNING) is True
        assert logger._should_log(LogLevel.ERROR) is True

    def test_format_message_plain(self):
        """Test plain text message formatting."""
        logger = QEvalLogger("test", output_json=False)

        msg = logger._format_message("INFO", "Test message")
        assert msg == "Test message"

    def test_format_message_with_context(self):
        """Test message formatting with context."""
        logger = QEvalLogger("test", output_json=False)

        msg = logger._format_message("INFO", "Test message", key1="value1", key2=42)
        assert "Test message" in msg
        assert "key1=value1" in msg
        assert "key2=42" in msg

    def test_format_message_json(self):
        """Test JSON message formatting."""
        import json

        logger = QEvalLogger("test", output_json=True)

        msg = logger._format_message("INFO", "Test message", key1="value1")
        parsed = json.loads(msg)

        assert parsed["level"] == "INFO"
        assert parsed["message"] == "Test message"
        assert parsed["key1"] == "value1"
        assert "timestamp" in parsed

    def test_debug_method(self, capfd):
        """Test debug logging method."""
        logger = QEvalLogger("test", level=LogLevel.DEBUG)
        logger.debug("Debug message", extra="data")
        # No assertion on output - just verify no exception

    def test_info_method(self, capfd):
        """Test info logging method."""
        logger = QEvalLogger("test", level=LogLevel.INFO)
        logger.info("Info message", count=5)
        # No assertion on output - just verify no exception

    def test_warning_method(self, capfd):
        """Test warning logging method."""
        logger = QEvalLogger("test", level=LogLevel.WARNING)
        logger.warning("Warning message", issue="something")
        # No assertion on output - just verify no exception

    def test_error_method(self, capfd):
        """Test error logging method."""
        logger = QEvalLogger("test", level=LogLevel.ERROR)
        logger.error("Error message", error="failed")
        # No assertion on output - just verify no exception

    def test_timed_context_manager(self):
        """Test timed context manager returns tracker."""
        logger = QEvalLogger("test", level=LogLevel.DEBUG)

        with logger.timed("test_operation") as tracker:
            time.sleep(0.01)

        assert tracker.elapsed > 0


class TestLoggerFunctions:
    """Tests for module-level logging functions."""

    def test_get_logger_creates_logger(self):
        """Test get_logger creates a new logger."""
        logger = get_logger("test_module_unique")
        assert isinstance(logger, QEvalLogger)
        assert logger.name == "test_module_unique"

    def test_get_logger_returns_same_instance(self):
        """Test get_logger returns same instance for same name."""
        logger1 = get_logger("test_shared_module")
        logger2 = get_logger("test_shared_module")
        assert logger1 is logger2

    def test_set_and_get_log_level(self):
        """Test setting and getting global log level."""
        original = get_log_level()

        set_log_level(LogLevel.DEBUG)
        assert get_log_level() == LogLevel.DEBUG

        set_log_level(LogLevel.ERROR)
        assert get_log_level() == LogLevel.ERROR

        # Restore original
        set_log_level(original)

    def test_configure_logging(self):
        """Test configure_logging function."""
        original = get_log_level()

        configure_logging(level=LogLevel.WARNING, output_json=False)
        assert get_log_level() == LogLevel.WARNING

        # Restore original
        set_log_level(original)


class TestPerformanceTracker:
    """Tests for PerformanceTracker class."""

    def test_basic_timing(self):
        """Test basic timing functionality."""
        with PerformanceTracker("test_op") as tracker:
            time.sleep(0.05)

        assert tracker.elapsed >= 0.05
        assert tracker.elapsed < 0.2

    def test_elapsed_before_complete(self):
        """Test elapsed returns current time while running."""
        tracker = PerformanceTracker("test_op")

        # Before start
        assert tracker.elapsed == 0.0

        # During execution
        tracker.__enter__()
        time.sleep(0.01)
        assert tracker.elapsed > 0

        tracker.__exit__(None, None, None)

    def test_tracker_with_logger(self):
        """Test tracker with logger integration."""
        logger = QEvalLogger("test", level=LogLevel.DEBUG)

        with PerformanceTracker("logged_op", logger=logger):
            time.sleep(0.01)

        # No exception means success

    def test_tracker_logs_error_on_exception(self):
        """Test tracker logs error when exception occurs."""
        logger = QEvalLogger("test", level=LogLevel.DEBUG)

        with pytest.raises(ValueError):
            with PerformanceTracker("failing_op", logger=logger):
                raise ValueError("Test error")


class TestPerformanceMonitor:
    """Tests for PerformanceMonitor class."""

    def test_track_single_operation(self):
        """Test tracking single operation."""
        monitor = PerformanceMonitor()

        with monitor.track("operation1"):
            time.sleep(0.01)

        stats = monitor.summary()
        assert "operation1" in stats
        assert stats["operation1"]["count"] == 1
        assert stats["operation1"]["total"] >= 0.01

    def test_track_multiple_operations(self):
        """Test tracking multiple different operations."""
        monitor = PerformanceMonitor()

        with monitor.track("op1"):
            time.sleep(0.01)

        with monitor.track("op2"):
            time.sleep(0.01)

        stats = monitor.summary()
        assert "op1" in stats
        assert "op2" in stats

    def test_track_repeated_operation(self):
        """Test tracking same operation multiple times."""
        monitor = PerformanceMonitor()

        for _ in range(3):
            with monitor.track("repeated_op"):
                time.sleep(0.01)

        stats = monitor.summary()
        assert stats["repeated_op"]["count"] == 3
        assert stats["repeated_op"]["mean"] >= 0.01

    def test_summary_statistics(self):
        """Test summary statistics calculation."""
        monitor = PerformanceMonitor()

        # Record with different times
        monitor.record("test_op", 0.1)
        monitor.record("test_op", 0.2)
        monitor.record("test_op", 0.3)

        stats = monitor.summary()
        assert stats["test_op"]["count"] == 3
        assert abs(stats["test_op"]["total"] - 0.6) < 0.001
        assert abs(stats["test_op"]["mean"] - 0.2) < 0.001
        assert abs(stats["test_op"]["min"] - 0.1) < 0.001
        assert abs(stats["test_op"]["max"] - 0.3) < 0.001

    def test_reset(self):
        """Test resetting monitor clears metrics."""
        monitor = PerformanceMonitor()

        with monitor.track("op"):
            pass

        assert len(monitor.summary()) > 0

        monitor.reset()
        assert len(monitor.summary()) == 0

    def test_empty_summary(self):
        """Test summary of empty monitor."""
        monitor = PerformanceMonitor()
        assert monitor.summary() == {}


class TestGlobalPerformanceMonitor:
    """Tests for global performance monitor."""

    def test_get_performance_monitor(self):
        """Test getting global performance monitor."""
        monitor = get_performance_monitor()
        assert isinstance(monitor, PerformanceMonitor)

    def test_global_monitor_is_singleton(self):
        """Test global monitor returns same instance."""
        monitor1 = get_performance_monitor()
        monitor2 = get_performance_monitor()
        assert monitor1 is monitor2


class TestTimedDecorator:
    """Tests for timed decorator."""

    def test_timed_decorator_returns_result(self):
        """Test timed decorator returns function result."""

        @timed
        def add(a, b):
            return a + b

        result = add(2, 3)
        assert result == 5

    def test_timed_decorator_preserves_function(self):
        """Test timed decorator preserves function metadata."""

        @timed
        def my_function():
            """My docstring."""
            pass

        assert my_function.__name__ == "my_function"
        assert my_function.__doc__ == "My docstring."

    def test_timed_decorator_handles_exception(self):
        """Test timed decorator handles exceptions."""

        @timed
        def failing_function():
            raise ValueError("Test error")

        with pytest.raises(ValueError):
            failing_function()


class TestMeasureTime:
    """Tests for measure_time context manager."""

    def test_measure_time_context(self, capfd):
        """Test measure_time context manager."""
        with measure_time("test_operation"):
            time.sleep(0.01)

        captured = capfd.readouterr()
        assert "test_operation" in captured.out

    def test_measure_time_prints_duration(self, capfd):
        """Test measure_time prints duration."""
        with measure_time("timed_op"):
            time.sleep(0.05)

        captured = capfd.readouterr()
        # Should have printed something like "timed_op: 0.05Xs"
        assert "timed_op:" in captured.out
        assert "s" in captured.out
