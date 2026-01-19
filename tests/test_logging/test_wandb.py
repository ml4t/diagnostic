"""Tests for W&B integration logging."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest


class TestWandbLoggerDisabled:
    """Tests for WandbLogger in disabled mode."""

    def test_init_disabled(self):
        """Test initialization with disabled=True."""
        from ml4t.diagnostic.logging.wandb import WandbLogger

        logger = WandbLogger(disabled=True)

        assert logger.disabled is True
        assert logger.run is None

    def test_log_config_disabled(self):
        """Test log_config returns early when disabled."""
        from ml4t.diagnostic.logging.wandb import WandbLogger

        logger = WandbLogger(disabled=True)
        # Should not raise
        logger.log_config({"key": "value"})

    def test_log_metrics_disabled(self):
        """Test log_metrics returns early when disabled."""
        from ml4t.diagnostic.logging.wandb import WandbLogger

        logger = WandbLogger(disabled=True)
        # Should not raise
        logger.log_metrics({"accuracy": 0.95}, step=1, prefix="train/")

    def test_log_fold_results_disabled(self):
        """Test log_fold_results returns early when disabled."""
        from ml4t.diagnostic.logging.wandb import WandbLogger

        logger = WandbLogger(disabled=True)
        # Should not raise
        logger.log_fold_results(
            fold_idx=0,
            train_size=100,
            test_size=25,
            metrics={"accuracy": 0.9},
        )

    def test_log_statistical_tests_disabled(self):
        """Test log_statistical_tests returns early when disabled."""
        from ml4t.diagnostic.logging.wandb import WandbLogger

        logger = WandbLogger(disabled=True)
        # Should not raise
        logger.log_statistical_tests({"adf": {"statistic": -3.5, "pvalue": 0.01}})

    def test_log_figure_disabled(self):
        """Test log_figure returns early when disabled."""
        from ml4t.diagnostic.logging.wandb import WandbLogger

        logger = WandbLogger(disabled=True)
        mock_figure = MagicMock()
        # Should not raise
        logger.log_figure(mock_figure, name="test_plot", step=1)

    def test_log_evaluation_summary_disabled(self):
        """Test log_evaluation_summary returns early when disabled."""
        from ml4t.diagnostic.logging.wandb import WandbLogger

        logger = WandbLogger(disabled=True)
        mock_result = MagicMock()
        # Should not raise
        logger.log_evaluation_summary(mock_result)

    def test_log_artifact_disabled(self):
        """Test log_artifact returns early when disabled."""
        from ml4t.diagnostic.logging.wandb import WandbLogger

        logger = WandbLogger(disabled=True)
        # Should not raise
        logger.log_artifact("/path/to/file", name="test_artifact")

    def test_finish_disabled(self):
        """Test finish returns early when disabled."""
        from ml4t.diagnostic.logging.wandb import WandbLogger

        logger = WandbLogger(disabled=True)
        # Should not raise
        logger.finish()

    def test_context_manager_disabled(self):
        """Test context manager with disabled logger."""
        from ml4t.diagnostic.logging.wandb import WandbLogger

        with WandbLogger(disabled=True) as logger:
            assert logger.disabled is True
            logger.log_metrics({"test": 1.0})
        # Should not raise on exit


class TestFlattenDict:
    """Tests for _flatten_dict static method."""

    def test_simple_dict(self):
        """Test flattening a simple dictionary."""
        from ml4t.diagnostic.logging.wandb import WandbLogger

        d = {"a": 1, "b": 2}
        result = WandbLogger._flatten_dict(d)

        assert result == {"a": 1, "b": 2}

    def test_nested_dict(self):
        """Test flattening a nested dictionary."""
        from ml4t.diagnostic.logging.wandb import WandbLogger

        d = {"outer": {"inner": 1, "another": 2}}
        result = WandbLogger._flatten_dict(d)

        assert result == {"outer/inner": 1, "outer/another": 2}

    def test_deeply_nested_dict(self):
        """Test flattening a deeply nested dictionary."""
        from ml4t.diagnostic.logging.wandb import WandbLogger

        d = {"a": {"b": {"c": 1}}}
        result = WandbLogger._flatten_dict(d)

        assert result == {"a/b/c": 1}

    def test_mixed_dict(self):
        """Test flattening with mixed nesting levels."""
        from ml4t.diagnostic.logging.wandb import WandbLogger

        d = {
            "simple": 1,
            "nested": {"value": 2},
            "deep": {"level1": {"level2": 3}},
        }
        result = WandbLogger._flatten_dict(d)

        assert result == {
            "simple": 1,
            "nested/value": 2,
            "deep/level1/level2": 3,
        }

    def test_empty_dict(self):
        """Test flattening an empty dictionary."""
        from ml4t.diagnostic.logging.wandb import WandbLogger

        result = WandbLogger._flatten_dict({})
        assert result == {}

    def test_custom_separator(self):
        """Test flattening with custom separator."""
        from ml4t.diagnostic.logging.wandb import WandbLogger

        d = {"outer": {"inner": 1}}
        result = WandbLogger._flatten_dict(d, sep=".")

        assert result == {"outer.inner": 1}

    def test_custom_parent_key(self):
        """Test flattening with parent key prefix."""
        from ml4t.diagnostic.logging.wandb import WandbLogger

        d = {"a": 1}
        result = WandbLogger._flatten_dict(d, parent_key="prefix")

        assert result == {"prefix/a": 1}


class TestLogExperiment:
    """Tests for log_experiment convenience function."""

    def test_without_wandb_installed(self):
        """Test log_experiment without wandb installed."""
        from ml4t.diagnostic.logging import wandb as wandb_module

        # Save original value
        original_has_wandb = wandb_module.HAS_WANDB
        wandb_module.HAS_WANDB = False

        try:
            from ml4t.diagnostic.logging.wandb import log_experiment

            # Create mock evaluator
            mock_evaluator = MagicMock()
            mock_result = MagicMock()
            mock_evaluator.evaluate.return_value = mock_result

            # Should run without W&B
            with pytest.warns(UserWarning, match="wandb not installed"):
                result = log_experiment(
                    mock_evaluator,
                    X=np.array([[1, 2], [3, 4]]),
                    y=np.array([0, 1]),
                    model=MagicMock(),
                )

            assert result == mock_result
            mock_evaluator.evaluate.assert_called_once()
        finally:
            wandb_module.HAS_WANDB = original_has_wandb


class TestWandbLoggerWarning:
    """Tests for wandb not installed warning."""

    def test_warning_when_wandb_not_installed(self):
        """Test warning is raised when wandb not installed and not disabled."""
        from ml4t.diagnostic.logging import wandb as wandb_module

        # Save original value
        original_has_wandb = wandb_module.HAS_WANDB
        wandb_module.HAS_WANDB = False

        try:
            from ml4t.diagnostic.logging.wandb import WandbLogger

            with pytest.warns(UserWarning, match="wandb not installed"):
                logger = WandbLogger()

            assert logger.disabled is True
        finally:
            wandb_module.HAS_WANDB = original_has_wandb

    def test_no_warning_when_explicitly_disabled(self):
        """Test no warning when explicitly disabled."""
        from ml4t.diagnostic.logging import wandb as wandb_module

        # Save original value
        original_has_wandb = wandb_module.HAS_WANDB
        wandb_module.HAS_WANDB = False

        try:
            from ml4t.diagnostic.logging.wandb import WandbLogger

            # No warning should be raised when explicitly disabled
            logger = WandbLogger(disabled=True)
            assert logger.disabled is True
        finally:
            wandb_module.HAS_WANDB = original_has_wandb


class TestHasWandbFlag:
    """Tests for HAS_WANDB module-level flag."""

    def test_has_wandb_is_boolean(self):
        """Test that HAS_WANDB is a boolean."""
        from ml4t.diagnostic.logging.wandb import HAS_WANDB

        assert isinstance(HAS_WANDB, bool)


class TestFlattenDictMore:
    """Additional tests for _flatten_dict static method."""

    def test_numeric_values(self):
        """Test that numeric values are preserved."""
        from ml4t.diagnostic.logging.wandb import WandbLogger

        d = {"int_val": 42, "float_val": 3.14, "bool_val": True}
        result = WandbLogger._flatten_dict(d)

        assert result["int_val"] == 42
        assert result["float_val"] == 3.14
        assert result["bool_val"] is True

    def test_list_values(self):
        """Test that list values are preserved (not flattened)."""
        from ml4t.diagnostic.logging.wandb import WandbLogger

        d = {"items": [1, 2, 3]}
        result = WandbLogger._flatten_dict(d)

        assert result["items"] == [1, 2, 3]

    def test_none_value(self):
        """Test that None values are preserved."""
        from ml4t.diagnostic.logging.wandb import WandbLogger

        d = {"optional": None}
        result = WandbLogger._flatten_dict(d)

        assert result["optional"] is None


class TestDisabledModeCompleteness:
    """Additional tests for disabled mode to ensure all paths are covered."""

    def test_disabled_with_all_init_params(self):
        """Test disabled mode with all init parameters provided."""
        from ml4t.diagnostic.logging.wandb import WandbLogger

        logger = WandbLogger(
            disabled=True,
            project="test",
            entity="test_entity",
            name="test_run",
            config={"param": 1},
            tags=["test"],
            notes="Test notes",
        )

        assert logger.disabled is True
        assert logger.run is None

    def test_all_log_methods_return_early(self):
        """Test that all log methods return early when disabled."""
        from ml4t.diagnostic.logging.wandb import WandbLogger

        logger = WandbLogger(disabled=True)

        # All these should complete without errors
        logger.log_config({"key": "value"})
        logger.log_metrics({"metric": 1.0})
        logger.log_fold_results(0, 100, 25, {"acc": 0.9})
        logger.log_statistical_tests({"test": {"p": 0.01}})
        logger.log_figure(MagicMock(), "name")
        logger.log_evaluation_summary(MagicMock())
        logger.log_artifact("/path", "name")
        logger.finish()

        # All operations completed successfully
        assert logger.disabled is True

    def test_context_manager_works_when_disabled(self):
        """Test that context manager works correctly when disabled."""
        from ml4t.diagnostic.logging.wandb import WandbLogger

        operations_completed = False

        with WandbLogger(disabled=True) as logger:
            logger.log_metrics({"test": 1.0})
            operations_completed = True

        assert operations_completed is True

    def test_finish_idempotent_when_disabled(self):
        """Test that finish can be called multiple times when disabled."""
        from ml4t.diagnostic.logging.wandb import WandbLogger

        logger = WandbLogger(disabled=True)
        logger.finish()
        logger.finish()
        logger.finish()

        # No error should occur
        assert logger.disabled is True


class TestLogExperimentDisabled:
    """Tests for log_experiment when W&B is not available."""

    def test_returns_evaluation_result(self):
        """Test that log_experiment returns evaluation result even without W&B."""
        from ml4t.diagnostic.logging import wandb as wandb_module
        from ml4t.diagnostic.logging.wandb import log_experiment

        original_has_wandb = wandb_module.HAS_WANDB
        wandb_module.HAS_WANDB = False

        try:
            mock_evaluator = MagicMock()
            mock_result = MagicMock()
            mock_evaluator.evaluate.return_value = mock_result

            with pytest.warns(UserWarning, match="wandb not installed"):
                result = log_experiment(
                    mock_evaluator,
                    X=np.array([[1, 2]]),
                    y=np.array([0]),
                    model=MagicMock(),
                )

            assert result == mock_result
            mock_evaluator.evaluate.assert_called_once()
        finally:
            wandb_module.HAS_WANDB = original_has_wandb

    def test_passes_kwargs_to_evaluate(self):
        """Test that kwargs are passed through to evaluate."""
        from ml4t.diagnostic.logging import wandb as wandb_module
        from ml4t.diagnostic.logging.wandb import log_experiment

        original_has_wandb = wandb_module.HAS_WANDB
        wandb_module.HAS_WANDB = False

        try:
            mock_evaluator = MagicMock()
            mock_result = MagicMock()
            mock_evaluator.evaluate.return_value = mock_result

            with pytest.warns(UserWarning, match="wandb not installed"):
                result = log_experiment(
                    mock_evaluator,
                    X=np.array([[1, 2]]),
                    y=np.array([0]),
                    model=MagicMock(),
                    custom_kwarg="value",
                )

            # Check that custom_kwarg was passed
            call_kwargs = mock_evaluator.evaluate.call_args[1]
            assert call_kwargs.get("custom_kwarg") == "value"
        finally:
            wandb_module.HAS_WANDB = original_has_wandb


class TestWandbModuleAttributes:
    """Tests for module-level attributes."""

    def test_wandb_logger_class_exists(self):
        """Test that WandbLogger class is exported."""
        from ml4t.diagnostic.logging.wandb import WandbLogger

        assert WandbLogger is not None

    def test_log_experiment_function_exists(self):
        """Test that log_experiment function is exported."""
        from ml4t.diagnostic.logging.wandb import log_experiment

        assert callable(log_experiment)

    def test_has_wandb_flag_exported(self):
        """Test that HAS_WANDB flag is exported."""
        from ml4t.diagnostic.logging.wandb import HAS_WANDB

        assert isinstance(HAS_WANDB, bool)
