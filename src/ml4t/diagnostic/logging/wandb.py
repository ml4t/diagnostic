"""Weights & Biases integration for experiment tracking.

This module provides hooks for logging ml4t-diagnostic experiments to W&B,
enabling tracking of evaluation metrics, hyperparameters, and
visualizations across experiments.
"""

import numbers
import warnings
from typing import Any, SupportsFloat, cast

import numpy as np
import pandas as pd

try:
    import wandb  # type: ignore[import-not-found,unused-ignore]

    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


class WandbLogger:
    """Logger for Weights & Biases experiment tracking.

    This class provides a unified interface for logging ml4t-diagnostic
    experiments to W&B, handling initialization, metric logging,
    and artifact management.
    """

    def __init__(
        self,
        project: str | None = None,
        entity: str | None = None,
        name: str | None = None,
        config: dict[str, Any] | None = None,
        tags: list[str] | None = None,
        notes: str | None = None,
        disabled: bool = False,
    ):
        """Initialize W&B logger.

        Parameters
        ----------
        project : str, optional
            W&B project name
        entity : str, optional
            W&B entity (team or username)
        name : str, optional
            Run name
        config : dict, optional
            Configuration dictionary to log
        tags : list[str], optional
            Tags for the run
        notes : str, optional
            Notes about the run
        disabled : bool
            If True, disables W&B logging
        """
        self.disabled = disabled or not HAS_WANDB
        self.run = None

        if self.disabled:
            if not HAS_WANDB and not disabled:
                warnings.warn(
                    "wandb not installed. Install with: pip install wandb",
                    stacklevel=2,
                )
            return

        # Initialize W&B run
        self.run = wandb.init(
            project=project or "ml4t-diagnostic",
            entity=entity,
            name=name,
            config=config,
            tags=tags or [],
            notes=notes,
            reinit=True,
        )

    def log_config(self, config: dict[str, Any]) -> None:
        """Log configuration parameters.

        Parameters
        ----------
        config : dict
            Configuration dictionary
        """
        if self.disabled or self.run is None:
            return

        # Flatten nested config for W&B
        flat_config = self._flatten_dict(config)
        wandb.config.update(flat_config)

    def log_metrics(
        self,
        metrics: dict[str, Any],
        step: int | None = None,
        prefix: str = "",
    ) -> None:
        """Log evaluation metrics.

        Parameters
        ----------
        metrics : dict
            Metrics to log
        step : int, optional
            Step number (e.g., CV fold)
        prefix : str
            Prefix for metric names
        """
        if self.disabled or self.run is None:
            return

        # Prepare metrics for logging
        log_dict = {}

        for name, value in metrics.items():
            key = f"{prefix}{name}" if prefix else name

            if isinstance(value, dict):
                # Handle nested metrics (e.g., with confidence intervals)
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, numbers.Number):
                        log_dict[f"{key}/{sub_key}"] = float(cast(SupportsFloat, sub_value))
            elif isinstance(value, numbers.Number):
                log_dict[key] = float(cast(SupportsFloat, value))
            elif isinstance(value, list | np.ndarray):
                # Log array statistics
                if len(value) > 0:
                    log_dict[f"{key}/mean"] = float(np.mean(value))
                    log_dict[f"{key}/std"] = float(np.std(value))
                    log_dict[f"{key}/min"] = float(np.min(value))
                    log_dict[f"{key}/max"] = float(np.max(value))

        if step is not None:
            log_dict["step"] = step

        wandb.log(log_dict)

    def log_fold_results(
        self,
        fold_idx: int,
        train_size: int,
        test_size: int,
        metrics: dict[str, Any],
    ) -> None:
        """Log results from a single CV fold.

        Parameters
        ----------
        fold_idx : int
            Fold index
        train_size : int
            Training set size
        test_size : int
            Test set size
        metrics : dict
            Fold metrics
        """
        if self.disabled or self.run is None:
            return

        # Add metrics with fold prefix
        self.log_metrics(metrics, step=fold_idx, prefix="fold/")

        # Log fold metadata
        wandb.log(
            {
                "fold/train_size": train_size,
                "fold/test_size": test_size,
                "fold/train_test_ratio": train_size / test_size if test_size > 0 else 0,
            },
            step=fold_idx,
        )

    def log_statistical_tests(self, tests: dict[str, Any]) -> None:
        """Log statistical test results.

        Parameters
        ----------
        tests : dict
            Statistical test results
        """
        if self.disabled or self.run is None:
            return

        log_dict = {}

        for test_name, result in tests.items():
            if isinstance(result, dict):
                for key, value in result.items():
                    if isinstance(value, numbers.Number):
                        log_dict[f"stats/{test_name}/{key}"] = float(cast(SupportsFloat, value))
                    elif key == "significant" and isinstance(value, bool):
                        log_dict[f"stats/{test_name}/{key}"] = int(value)

        wandb.log(log_dict)

    def log_figure(
        self,
        figure: Any,
        name: str,
        step: int | None = None,
    ) -> None:
        """Log a Plotly figure.

        Parameters
        ----------
        figure : plotly.graph_objects.Figure
            Figure to log
        name : str
            Figure name
        step : int, optional
            Step number
        """
        if self.disabled or self.run is None:
            return

        # Convert Plotly figure to W&B
        wandb.log({f"plots/{name}": figure}, step=step)

    def log_evaluation_summary(
        self,
        result: Any,  # EvaluationResult
        _predictions: Any | None = None,
        _returns: Any | None = None,
    ) -> None:
        """Log complete evaluation summary.

        Parameters
        ----------
        result : EvaluationResult
            Evaluation result object
        predictions : array-like, optional
            Predictions for additional logging
        returns : array-like, optional
            Returns for additional logging
        """
        if self.disabled or self.run is None:
            return

        # Log summary metrics
        summary = result.summary()

        # Log aggregate metrics
        self.log_metrics(summary["metrics"], prefix="summary/")

        # Log statistical tests
        if summary.get("statistical_tests"):
            self.log_statistical_tests(summary["statistical_tests"])

        # Log metadata
        wandb.log(
            {
                "summary/tier": result.tier,
                "summary/n_folds": summary["n_folds"],
                "summary/splitter": result.splitter_name,
            },
        )

        # Create summary table
        if result.fold_results:
            fold_data = []
            for fold in result.fold_results:
                fold_row = {"fold": fold.get("fold", 0)}
                fold_row.update(
                    {k: v for k, v in fold.items() if isinstance(v, numbers.Number)},
                )
                fold_data.append(fold_row)

            fold_table = wandb.Table(dataframe=pd.DataFrame(fold_data))
            wandb.log({"tables/fold_results": fold_table})

    def log_artifact(
        self,
        artifact_path: str,
        name: str,
        artifact_type: str = "evaluation",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Log an artifact (model, dataset, etc.).

        Parameters
        ----------
        artifact_path : str
            Path to artifact file
        name : str
            Artifact name
        artifact_type : str
            Type of artifact
        metadata : dict, optional
            Additional metadata
        """
        if self.disabled or self.run is None:
            return

        artifact = wandb.Artifact(
            name=name,
            type=artifact_type,
            metadata=metadata or {},
        )
        artifact.add_file(artifact_path)
        wandb.log_artifact(artifact)

    def finish(self) -> None:
        """Finish the W&B run."""
        if self.disabled or self.run is None:
            return

        wandb.finish()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.finish()

    @staticmethod
    def _flatten_dict(
        d: dict[str, Any],
        parent_key: str = "",
        sep: str = "/",
    ) -> dict[str, Any]:
        """Flatten nested dictionary."""
        items: list[tuple[str, Any]] = []

        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k

            if isinstance(v, dict):
                items.extend(WandbLogger._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))

        return dict(items)


def log_experiment(
    evaluator: Any,
    X: Any,
    y: Any,
    model: Any,
    project: str | None = None,
    config: dict[str, Any] | None = None,
    tags: list[str] | None = None,
    **kwargs: Any,
) -> Any:
    """Convenience function to run and log an experiment.

    Parameters
    ----------
    evaluator : ml4t-diagnostic.Evaluator
        Configured evaluator
    X : array-like
        Features
    y : array-like
        Labels
    model : estimator
        Model to evaluate
    project : str, optional
        W&B project name
    config : dict, optional
        Additional config to log
    tags : list[str], optional
        Experiment tags
    **kwargs : Any
        Additional arguments passed to evaluate()

    Returns:
    -------
    EvaluationResult
        Result with W&B logging
    """
    if not HAS_WANDB:
        warnings.warn(
            "wandb not installed. Running without logging. Install with: pip install wandb",
            stacklevel=2,
        )
        return evaluator.evaluate(X, y, model, **kwargs)

    # Initialize logger
    with WandbLogger(project=project, config=config, tags=tags) as logger:
        # Log evaluator configuration
        logger.log_config(
            {
                "evaluator": {
                    "tier": evaluator.tier,
                    "splitter": evaluator.splitter.__class__.__name__,
                    "metrics": evaluator.metrics,
                    "statistical_tests": evaluator.statistical_tests,
                    "confidence_level": evaluator.confidence_level,
                    "bootstrap_samples": evaluator.bootstrap_samples,
                },
            },
        )

        # Log model info if available
        if hasattr(model, "get_params"):
            logger.log_config({"model": model.get_params()})

        # Run evaluation
        result = evaluator.evaluate(X, y, model, **kwargs)

        # Log results
        logger.log_evaluation_summary(result)

        return result
