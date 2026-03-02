"""Main Evaluator framework implementing the Three-Tier Validation Framework.

This module provides the Evaluator class that orchestrates the complete ml4t-diagnostic
validation workflow:

- Tier 1 (Rigorous Backtesting): Full CPCV validation with statistical tests
- Tier 2 (Statistical Significance): HAC-adjusted tests and significance testing
- Tier 3 (Production Monitoring): Fast screening metrics for live systems

The Evaluator integrates with all splitters, metrics, and statistical tests to
provide a unified interface for financial ML validation.
"""

import warnings
from collections.abc import Callable
from datetime import datetime
from typing import TYPE_CHECKING, Any, Union

import numpy as np
import pandas as pd
import polars as pl
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, clone

from ml4t.diagnostic.backends.adapter import DataFrameAdapter
from ml4t.diagnostic.splitters.base import BaseSplitter
from ml4t.diagnostic.splitters.combinatorial import CombinatorialCV
from ml4t.diagnostic.splitters.walk_forward import WalkForwardCV

from .metric_registry import MetricRegistry
from .stat_registry import StatTestRegistry

if TYPE_CHECKING:
    from numpy.typing import NDArray


def _load_visualization_functions() -> tuple[Callable[..., Any], Callable[..., Any]]:
    """Load visualization functions on demand to keep viz dependencies optional."""
    try:
        from .visualization import plot_ic_heatmap, plot_quantile_returns
    except ImportError as exc:
        raise ImportError(
            "Visualization features require optional dependencies. "
            "Install with: pip install ml4t-diagnostic[viz]"
        ) from exc

    return plot_ic_heatmap, plot_quantile_returns


def _load_dashboard_function() -> Callable[..., Any]:
    """Load dashboard function on demand to keep viz dependencies optional."""
    try:
        from .dashboard import create_evaluation_dashboard
    except ImportError as exc:
        raise ImportError(
            "HTML dashboard export requires optional dependencies. "
            "Install with: pip install ml4t-diagnostic[viz]"
        ) from exc

    return create_evaluation_dashboard


def get_metric_directionality(metric_name: str) -> bool:
    """Get whether a metric should be maximized (True) or minimized (False).

    Parameters
    ----------
    metric_name : str
        Name of the metric

    Returns
    -------
    bool
        True if higher values are better, False if lower values are better
    """
    normalized = metric_name.lower().replace("-", "_").replace(" ", "_")
    return MetricRegistry.default().is_maximize(normalized)


class EvaluationResult:
    """Container for evaluation results with rich reporting capabilities."""

    def __init__(
        self,
        tier: int,
        splitter_name: str,
        metrics_results: dict[str, Any],
        statistical_tests: dict[str, Any] | None = None,
        fold_results: list[dict[str, Any]] | None = None,
        metadata: dict[str, Any] | None = None,
        oos_returns: list[np.ndarray] | None = None,
    ):
        """Initialize evaluation result.

        Parameters
        ----------
        tier : int
            Tier level (1, 2, or 3) of the evaluation
        splitter_name : str
            Name of the cross-validation method used
        metrics_results : Dict[str, Any]
            Aggregated metrics results
        statistical_tests : Optional[Dict[str, Any]]
            Statistical test results (Tier 1 & 2)
        fold_results : Optional[List[Dict[str, Any]]]
            Individual fold results for detailed analysis
        metadata : Optional[Dict[str, Any]]
            Additional metadata about the evaluation
        oos_returns : Optional[List[np.ndarray]]
            Out-of-sample strategy returns from each fold for statistical testing
        """
        self.tier = tier
        self.splitter_name = splitter_name
        self.metrics_results = metrics_results
        self.statistical_tests = statistical_tests or {}
        self.fold_results = fold_results or []
        self.metadata = metadata or {}
        self.oos_returns = oos_returns or []
        self.timestamp = datetime.now()

    def summary(self) -> dict[str, Any]:
        """Generate a summary of the evaluation results."""
        summary: dict[str, Any] = {
            "tier": self.tier,
            "splitter": self.splitter_name,
            "timestamp": self.timestamp.isoformat(),
            "n_folds": len(self.fold_results),
            "metrics": {},
            "statistical_tests": {},
        }

        # Summarize metrics
        for metric_name, value in self.metrics_results.items():
            if isinstance(value, dict) and "mean" in value:
                summary["metrics"][metric_name] = {
                    "mean": value["mean"],
                    "std": value.get("std", None),
                    "significant": value.get("significant", None),
                }
            else:
                summary["metrics"][metric_name] = value

        # Summarize statistical tests
        for test_name, result in self.statistical_tests.items():
            if isinstance(result, dict):
                summary["statistical_tests"][test_name] = {
                    "test_statistic": result.get(
                        "test_statistic",
                        result.get("dsr", None),
                    ),
                    "p_value": result.get("p_value", None),
                    "significant": result.get("p_value", 1.0) < 0.05
                    if "p_value" in result
                    else None,
                }

        return summary

    def get_oos_returns_series(self) -> np.ndarray | None:
        """Get concatenated out-of-sample returns series for statistical testing.

        Returns
        -------
        np.ndarray or None
            Concatenated strategy returns from all folds, or None if not available
        """
        if not self.oos_returns or len(self.oos_returns) == 0:
            return None

        # Filter out any NaN arrays from failed folds
        valid_returns = [returns for returns in self.oos_returns if not np.all(np.isnan(returns))]

        if not valid_returns:
            return None

        return np.concatenate(valid_returns)

    def plot(
        self,
        predictions: Any | None = None,
        returns: Any | None = None,
    ) -> Any:
        """Generate default visualization for evaluation results.

        Parameters
        ----------
        predictions : array-like, optional
            Predictions for visualization
        returns : array-like, optional
            Returns for visualization

        Returns:
        -------
        plotly.graph_objects.Figure
            Interactive visualization
        """
        plot_ic_heatmap, plot_quantile_returns = _load_visualization_functions()

        # Default plot based on available metrics
        if "ic" in self.metrics_results and predictions is not None and returns is not None:
            return plot_ic_heatmap(predictions, returns)
        if "sharpe" in self.metrics_results and returns is not None and predictions is not None:
            return plot_quantile_returns(predictions, returns)
        # Return a summary plot
        try:
            import plotly.graph_objects as go
        except ImportError as exc:
            raise ImportError(
                "Plot generation requires optional dependencies. "
                "Install with: pip install ml4t-diagnostic[viz]"
            ) from exc

        metric_names = list(self.metrics_results.keys())
        metric_values = [
            self.metrics_results[m].get("mean", 0)
            if isinstance(self.metrics_results[m], dict)
            else self.metrics_results[m]
            for m in metric_names
        ]

        fig = go.Figure(data=[go.Bar(x=metric_names, y=metric_values)])
        fig.update_layout(
            title=f"Evaluation Results - Tier {self.tier}",
            xaxis_title="Metric",
            yaxis_title="Value",
        )
        return fig

    def to_html(
        self,
        filename: str,
        predictions: Any | None = None,
        returns: Any | None = None,
        features: Any | None = None,
        title: str | None = None,
    ) -> None:
        """Generate interactive HTML dashboard.

        Parameters
        ----------
        filename : str
            Output HTML filename
        predictions : array-like, optional
            Model predictions for visualizations
        returns : array-like, optional
            Returns data for visualizations
        features : array-like, optional
            Feature data for distribution analysis
        title : str, optional
            Dashboard title

        Examples:
        --------
        >>> result.to_html("evaluation_report.html", predictions=pred_df, returns=ret_df)
        """
        create_evaluation_dashboard = _load_dashboard_function()
        create_evaluation_dashboard(
            self,
            filename,
            predictions=predictions,
            returns=returns,
            features=features,
            title=title,
        )

    def __repr__(self) -> str:
        """String representation of evaluation result."""
        summary = self.summary()
        metrics_str = ", ".join(
            [
                f"{k}: {v['mean']:.3f}"
                if isinstance(v, dict) and "mean" in v
                else f"{k}: {v:.3f}"
                if isinstance(v, int | float)
                else f"{k}: {v}"
                for k, v in summary["metrics"].items()
            ],
        )

        return (
            f"EvaluationResult(tier={self.tier}, splitter={self.splitter_name}, "
            f"n_folds={summary['n_folds']}, metrics=[{metrics_str}])"
        )


class Evaluator:
    """Main evaluator implementing the Three-Tier Validation Framework.

    The Evaluator orchestrates the complete ml4t-diagnostic validation workflow by
    integrating cross-validation splitters, performance metrics, and
    statistical tests into a unified framework.

    Three-Tier Framework:
    - Tier 3: Fast screening with basic metrics
    - Tier 2: Statistical significance testing with HAC adjustments
    - Tier 1: Rigorous backtesting with multiple testing corrections
    """

    # Backward-compatible class attributes (delegate to registries)
    @property
    def METRIC_REGISTRY(self) -> dict[str, Callable]:  # noqa: N802
        """Get metric registry (backward compatibility)."""
        registry = MetricRegistry.default()
        return {name: registry.get(name) for name in registry.list_metrics()}

    @property
    def STAT_TEST_REGISTRY(self) -> dict[str, Callable]:  # noqa: N802
        """Get stat test registry (backward compatibility)."""
        registry = StatTestRegistry.default()
        return {name: registry.get(name) for name in registry.list_tests()}

    def __init__(
        self,
        splitter: BaseSplitter | None = None,
        metrics: list[str] | None = None,
        statistical_tests: list[str] | None = None,
        tier: int | None = None,
        confidence_level: float = 0.05,
        bootstrap_samples: int = 1000,
        random_state: int | None = None,
        n_jobs: int = 1,
    ):
        """Initialize the Evaluator.

        Parameters
        ----------
        splitter : Optional[BaseSplitter], default None
            Cross-validation splitter. If None, infers from tier
        metrics : Optional[List[str]], default None
            List of metrics to compute. If None, uses tier defaults
        statistical_tests : Optional[List[str]], default None
            List of statistical tests to perform. If None, uses tier defaults
        tier : Optional[int], default None
            Tier level (1, 2, or 3). If None, infers from other parameters
        confidence_level : float, default 0.05
            Significance level for statistical tests
        bootstrap_samples : int, default 1000
            Number of bootstrap samples for confidence intervals
        random_state : Optional[int], default None
            Random seed for reproducible results
        n_jobs : int, default 1
            Number of parallel jobs for cross-validation.
            -1 means using all processors

        Examples:
        --------
        # Tier 3: Fast screening
        >>> evaluator = Evaluator(tier=3)
        >>> result = evaluator.evaluate(X, y, model)

        # Tier 1: Full rigorous evaluation
        >>> evaluator = Evaluator(
        ...     splitter=CombinatorialCV(n_groups=8),
        ...     metrics=["sharpe", "sortino", "max_drawdown"],
        ...     statistical_tests=["dsr", "whites_reality_check"],
        ...     tier=1
        ... )
        >>> result = evaluator.evaluate(X, y, model)
        """
        self.confidence_level = confidence_level
        self.bootstrap_samples = bootstrap_samples
        self.random_state = random_state
        self.n_jobs = n_jobs

        # Infer tier if not specified
        if tier is None:
            tier = self._infer_tier(splitter, metrics, statistical_tests)

        self.tier = tier
        self.splitter = splitter or self._get_default_splitter(tier)
        self.metrics = metrics or self._get_default_metrics(tier)
        self.statistical_tests = statistical_tests or self._get_default_statistical_tests(tier)

        # Validate configuration
        self._validate_configuration()

    @classmethod
    def register_metric(
        cls,
        name: str,
        func: Callable[..., float],
        maximize: bool = True,
    ) -> None:
        """Register a custom metric function.

        Parameters
        ----------
        name : str
            Name of the metric
        func : Callable
            Function that takes (predictions, actual, strategy_returns) and returns float
        maximize : bool, default True
            Whether higher values are better

        Examples
        --------
        >>> def my_metric(predictions, actual, returns):
        ...     return np.mean(predictions > 0)
        >>> Evaluator.register_metric("my_metric", my_metric)
        """
        MetricRegistry.default().register(name, func, maximize=maximize)

    @classmethod
    def register_statistical_test(
        cls,
        name: str,
        func: Callable[..., dict[str, Any]],
    ) -> None:
        """Register a custom statistical test function.

        Parameters
        ----------
        name : str
            Name of the test
        func : Callable
            Function that returns a dict with test results
        """
        StatTestRegistry.default().register(name, func)

    def _infer_tier(
        self,
        splitter: BaseSplitter | None,
        _metrics: list[str] | None,
        statistical_tests: list[str] | None,
    ) -> int:
        """Infer tier level from configuration."""
        # Tier 1 indicators: CPCV splitter or advanced statistical tests
        if isinstance(splitter, CombinatorialCV) or (
            statistical_tests
            and any(test in ["dsr", "whites_reality_check"] for test in statistical_tests)
        ):
            return 1

        # Tier 2 indicators: HAC tests or confidence intervals
        if statistical_tests and any(test in ["hac_ic", "fdr"] for test in statistical_tests):
            return 2

        # Default to Tier 3 (fast screening)
        return 3

    def _get_default_splitter(self, tier: int) -> BaseSplitter:
        """Get default splitter for tier."""
        if tier == 1:
            return CombinatorialCV(n_groups=8, n_test_groups=2)
        if tier == 2:
            return WalkForwardCV(n_splits=5)
        # tier == 3
        return WalkForwardCV(n_splits=3)

    def _get_default_metrics(self, tier: int) -> list[str]:
        """Get default metrics for tier."""
        if tier == 1:
            return ["ic", "sharpe", "sortino", "max_drawdown", "hit_rate"]
        if tier == 2:
            return ["ic", "sharpe", "hit_rate"]
        # tier == 3
        return ["ic", "hit_rate"]

    def _get_default_statistical_tests(self, tier: int) -> list[str]:
        """Get default statistical tests for tier."""
        if tier == 1:
            return ["dsr", "fdr"]
        if tier == 2:
            return ["hac_ic"]
        # tier == 3
        return []

    def _validate_configuration(self) -> None:
        """Validate evaluator configuration using Pydantic schemas."""
        from pydantic import ValidationError

        from ml4t.diagnostic.utils.config import EvaluatorConfig

        try:
            # Validate main evaluator parameters
            EvaluatorConfig(
                tier=self.tier,
                confidence_level=self.confidence_level,
                bootstrap_samples=self.bootstrap_samples,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
            )

        except ValidationError as e:
            # Convert Pydantic validation errors to clearer messages
            error_messages = []
            for error in e.errors():
                field = error["loc"][0] if error["loc"] else "unknown"
                message = error["msg"]
                error_messages.append(f"{field}: {message}")

            raise ValueError(  # noqa: B904
                f"Configuration validation failed: {'; '.join(error_messages)}",
            )

        # Validate metrics against registry
        metric_registry = MetricRegistry.default()
        invalid_metrics = [m for m in self.metrics if m not in metric_registry]
        if invalid_metrics:
            raise ValueError(
                f"Unknown metrics: {invalid_metrics}. Available: {metric_registry.list_metrics()}",
            )

        # Validate statistical tests against registry
        stat_registry = StatTestRegistry.default()
        invalid_tests = [t for t in self.statistical_tests if t not in stat_registry]
        if invalid_tests:
            raise ValueError(
                f"Unknown statistical tests: {invalid_tests}. Available: {stat_registry.list_tests()}",
            )

        # Tier-specific validations with Pydantic-style consistency checks
        if self.tier == 1 and not isinstance(self.splitter, CombinatorialCV):
            warnings.warn(
                "Tier 1 evaluation should use CombinatorialCV for maximum rigor",
                stacklevel=2,
            )

        if self.tier == 3 and len(self.statistical_tests) > 2:
            warnings.warn(
                "Tier 3 is designed for fast screening - consider limiting statistical tests",
                stacklevel=2,
            )

    def evaluate(
        self,
        x: Union[pl.DataFrame, pd.DataFrame, "NDArray[Any]"],
        y: Union[pl.Series, pd.Series, "NDArray[Any]"],
        model: BaseEstimator | Callable[..., Any],
        strategy_func: Callable[..., Any] | None = None,
        **kwargs: Any,
    ) -> EvaluationResult:
        """Evaluate a model using the configured validation framework.

        Parameters
        ----------
        x : Union[pl.DataFrame, pd.DataFrame, NDArray]
            Feature matrix
        y : Union[pl.Series, pd.Series, NDArray]
            Target values (returns)
        model : Union[BaseEstimator, Callable]
            Model to evaluate (scikit-learn compatible or callable)
        strategy_func : Optional[Callable], default None
            Function to convert predictions to returns. If None, assumes
            predictions are directly used for position sizing
        **kwargs : Any
            Additional parameters passed to splitter

        Returns:
        -------
        EvaluationResult
            Comprehensive evaluation results

        Examples:
        --------
        >>> from sklearn.ensemble import RandomForestRegressor
        >>> model = RandomForestRegressor(n_estimators=50)
        >>> evaluator = Evaluator(tier=2)
        >>> result = evaluator.evaluate(X, y, model)
        >>> print(result.summary())
        """
        # Convert inputs to consistent format
        x_array = DataFrameAdapter.to_numpy(x)
        y_array = DataFrameAdapter.to_numpy(y).flatten()

        if len(x_array) != len(y_array):
            raise ValueError("x and y must have the same number of samples")

        # Set random seed if specified
        if self.random_state is not None:
            np.random.seed(self.random_state)

        def process_fold(
            fold_idx,
            train_idx,
            test_idx,
            model,
            x_array,
            y_array,
            strategy_func,
        ):
            """Process a single fold with full process isolation."""
            try:
                x_train, x_test = x_array[train_idx], x_array[test_idx]
                y_train, y_test = y_array[train_idx], y_array[test_idx]

                if hasattr(model, "fit") and hasattr(model, "predict"):
                    # Clone to prevent shared state between parallel processes
                    model_clone = clone(model)

                    if hasattr(model_clone, "random_state") and self.random_state is not None:
                        # Deterministic but different seed per fold
                        model_clone.random_state = self.random_state + fold_idx

                    model_clone.fit(x_train, y_train)
                    predictions = model_clone.predict(x_test)
                else:
                    # Callable model (must be stateless)
                    predictions = model(x_train, y_train, x_test)

                if strategy_func is not None:
                    strategy_returns = strategy_func(predictions, y_test)
                else:
                    positions = np.sign(predictions)
                    strategy_returns = positions * y_test

                fold_metrics = {}
                metric_registry = MetricRegistry.default()
                for metric_name in self.metrics:
                    try:
                        if metric_name in metric_registry:
                            metric_func = metric_registry.get(metric_name)
                            value = metric_func(predictions, y_test, strategy_returns)

                            if metric_name == "max_drawdown" and isinstance(value, dict):
                                value = value["max_drawdown"]

                            fold_metrics[metric_name] = value
                    except Exception as e:
                        fold_metrics[metric_name] = np.nan
                        warnings.warn(
                            f"Fold {fold_idx}: Failed to calculate {metric_name}: {e}",
                            stacklevel=2,
                        )

                fold_metrics["fold"] = fold_idx
                fold_metrics["n_train"] = len(train_idx)
                fold_metrics["n_test"] = len(test_idx)

                return fold_metrics, predictions, y_test, strategy_returns

            except Exception as e:
                warnings.warn(
                    f"Fold {fold_idx} failed with error: {e}. Returning NaN results.",
                    stacklevel=2,
                )
                nan_metrics = dict.fromkeys(self.metrics, np.nan)
                nan_metrics.update(
                    {
                        "fold": fold_idx,
                        "n_train": len(train_idx),
                        "n_test": len(test_idx),
                    },
                )

                return nan_metrics, np.array([np.nan]), np.array([np.nan]), np.array([np.nan])

        splits = list(self.splitter.split(x, y, **kwargs))

        if self.n_jobs == 1:
            results = []
            for fold_idx, (train_idx, test_idx) in enumerate(splits):
                result = process_fold(
                    fold_idx, train_idx, test_idx, model, x_array, y_array, strategy_func
                )
                results.append(result)
        else:
            # Use loky backend for process isolation (prevents race conditions)
            results = Parallel(n_jobs=self.n_jobs, backend="loky")(
                delayed(process_fold)(
                    fold_idx, train_idx, test_idx, model, x_array, y_array, strategy_func
                )
                for fold_idx, (train_idx, test_idx) in enumerate(splits)
            )

        fold_results = [r[0] for r in results]
        all_predictions = [pred for r in results for pred in r[1]]
        all_actual = [actual for r in results for actual in r[2]]
        oos_returns = [r[3] for r in results]

        metrics_results = self._aggregate_metrics(fold_results)
        statistical_tests = self._perform_statistical_tests(
            fold_results,
            all_predictions,
            all_actual,
            metrics_results,
            oos_returns,
        )

        metadata = {
            "n_samples": len(x_array),
            "n_features": x_array.shape[1] if x_array.ndim > 1 else 1,
            "splitter_params": self.splitter.__dict__,
            "tier": self.tier,
            "random_state": self.random_state,
        }

        return EvaluationResult(
            tier=self.tier,
            splitter_name=self.splitter.__class__.__name__,
            metrics_results=metrics_results,
            statistical_tests=statistical_tests,
            fold_results=fold_results,
            metadata=metadata,
            oos_returns=oos_returns,
        )

    def _aggregate_metrics(self, fold_results: list[dict[str, Any]]) -> dict[str, Any]:
        """Aggregate metrics across folds."""
        aggregated = {}

        for metric_name in self.metrics:
            values = [fold.get(metric_name, np.nan) for fold in fold_results]
            valid_values = [v for v in values if not np.isnan(v)]

            if valid_values:
                aggregated[metric_name] = {
                    "mean": np.mean(valid_values),
                    "std": np.std(valid_values, ddof=1) if len(valid_values) > 1 else 0.0,
                    "min": np.min(valid_values),
                    "max": np.max(valid_values),
                    "values": valid_values,
                    "n_valid": len(valid_values),
                }

                # Add confidence interval for mean if multiple folds
                if len(valid_values) > 1:
                    se = aggregated[metric_name]["std"] / np.sqrt(len(valid_values))
                    from scipy.stats import t

                    t_val = t.ppf(1 - self.confidence_level / 2, len(valid_values) - 1)
                    margin = t_val * se

                    aggregated[metric_name]["ci_lower"] = aggregated[metric_name]["mean"] - margin
                    aggregated[metric_name]["ci_upper"] = aggregated[metric_name]["mean"] + margin
            else:
                aggregated[metric_name] = {
                    "mean": np.nan,
                    "std": np.nan,
                    "min": np.nan,
                    "max": np.nan,
                    "values": [],
                    "n_valid": 0,
                }

        return aggregated

    def _perform_statistical_tests(
        self,
        fold_results: list[dict[str, Any]],
        all_predictions: list[float],
        all_actual: list[float],
        metrics_results: dict[str, Any],
        oos_returns: list[np.ndarray],
    ) -> dict[str, Any]:
        """Perform statistical tests based on configuration."""
        statistical_results: dict[str, Any] = {}
        stat_registry = StatTestRegistry.default()

        for test_name in self.statistical_tests:
            try:
                if test_name in stat_registry:
                    test_func = stat_registry.get(test_name)

                    # Prepare test-specific arguments
                    if test_name == "dsr" and "sharpe" in metrics_results:
                        sharpe_values = metrics_results["sharpe"]["values"]
                        if sharpe_values and len(oos_returns) > 0:
                            best_sharpe = float(np.max(sharpe_values))
                            n_trials = len(fold_results)
                            # Calculate variance across trials
                            variance_trials = (
                                float(np.var(sharpe_values, ddof=1))
                                if len(sharpe_values) > 1
                                else 0.001
                            )
                            # Calculate average sample size per fold
                            n_samples = int(
                                np.mean(
                                    [len(returns) for returns in oos_returns if len(returns) > 0]
                                )
                            )
                            # Use deflated_sharpe_ratio_from_statistics with new API
                            dsr_result = test_func(
                                observed_sharpe=best_sharpe,
                                n_samples=n_samples,
                                n_trials=n_trials,
                                variance_trials=variance_trials,
                            )
                            # Convert DSRResult dataclass to dict for consistency
                            result = {
                                "dsr": dsr_result.probability,
                                "p_value": dsr_result.p_value,
                                "expected_max_sharpe": dsr_result.expected_max_sharpe,
                                "z_score": dsr_result.z_score,
                                "is_significant": dsr_result.is_significant,
                            }
                        else:
                            continue

                    elif test_name == "hac_ic" and "ic" in metrics_results:
                        result = test_func(
                            predictions=np.array(all_predictions),
                            returns=np.array(all_actual),
                            return_details=True,
                        )

                    elif test_name == "fdr":
                        # Collect p-values from other tests
                        p_values = []
                        for test_result in statistical_results.values():
                            if isinstance(test_result, dict) and "p_value" in test_result:
                                p_values.append(test_result["p_value"])

                        if p_values:
                            result = test_func(
                                p_values,
                                alpha=self.confidence_level,
                                return_details=True,
                            )
                        else:
                            continue

                    elif test_name == "whites_reality_check":
                        if len(oos_returns) > 1 and all(
                            len(returns) > 0 for returns in oos_returns
                        ):
                            # Concatenate all OOS returns into a single time series
                            # This is the correct input for White's Reality Check
                            strategy_returns_series = np.concatenate(oos_returns)

                            # Create benchmark (zero returns) of the same length
                            benchmark_returns = np.zeros(len(strategy_returns_series))

                            # Reshape for test function (expects 2D array for strategies)
                            strategy_returns_matrix = strategy_returns_series.reshape(-1, 1)

                            result = test_func(
                                returns_benchmark=benchmark_returns,
                                returns_strategies=strategy_returns_matrix,
                                bootstrap_samples=min(self.bootstrap_samples, 500),
                                random_state=self.random_state,
                            )
                        else:
                            continue
                    else:
                        # Generic test function call
                        result = test_func(
                            fold_results=fold_results,
                            predictions=all_predictions,
                            actual=all_actual,
                            metrics_results=metrics_results,
                        )

                    statistical_results[test_name] = result
                else:
                    warnings.warn(
                        f"Unknown statistical test: {test_name}",
                        stacklevel=2,
                    )
                    continue

            except Exception as e:
                warnings.warn(
                    f"Error in statistical test {test_name}: {e}",
                    stacklevel=2,
                )
                # Store error in a way that's compatible with the expected type
                error_result: dict[str, Any] = {"error": str(e)}
                statistical_results[test_name] = error_result

        return statistical_results

    def batch_evaluate(
        self,
        models: list[BaseEstimator | Callable[..., Any]],
        x: Union[pl.DataFrame, pd.DataFrame, "NDArray[Any]"],
        y: Union[pl.Series, pd.Series, "NDArray[Any]"],
        model_names: list[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, EvaluationResult]:
        """Evaluate multiple models with the same validation framework.

        Parameters
        ----------
        models : List[Union[BaseEstimator, Callable]]
            List of models to evaluate
        X : Union[pl.DataFrame, pd.DataFrame, NDArray]
            Feature matrix
        y : Union[pl.Series, pd.Series, NDArray]
            Target values
        model_names : Optional[List[str]], default None
            Names for the models. If None, uses model class names
        **kwargs : Any
            Additional parameters passed to evaluate()

        Returns:
        -------
        dict[str, EvaluationResult]
            Dictionary mapping model names to evaluation results
        """
        if model_names is None:
            model_names = [
                model.__class__.__name__ if hasattr(model, "__class__") else f"Model_{i}"
                for i, model in enumerate(models)
            ]

        if len(models) != len(model_names):
            raise ValueError("Number of models must match number of model names")

        results = {}
        for model, name in zip(models, model_names, strict=False):
            print(f"Evaluating {name}...")
            results[name] = self.evaluate(x, y, model, **kwargs)

        return results

    def compare_models(
        self,
        batch_results: dict[str, EvaluationResult],
        primary_metric: str = "sharpe",
    ) -> dict[str, Any]:
        """Compare multiple model evaluation results.

        Parameters
        ----------
        batch_results : dict[str, EvaluationResult]
            Results from batch_evaluate()
        primary_metric : str, default "sharpe"
            Primary metric for ranking models

        Returns:
        -------
        dict[str, Any]
            Comparison summary with rankings and statistical tests
        """
        if not batch_results:
            return {"error": "No results to compare"}

        # Extract primary metric values
        model_metrics = {}
        for name, result in batch_results.items():
            metric_value = result.metrics_results.get(primary_metric, {}).get(
                "mean",
                np.nan,
            )
            model_metrics[name] = metric_value

        # Rank models
        valid_models = {k: v for k, v in model_metrics.items() if not np.isnan(v)}
        if not valid_models:
            return {"error": f"No valid {primary_metric} values found"}

        # Determine sort order based on metric directionality
        maximize = get_metric_directionality(primary_metric)

        # Special handling for drawdown metrics (they're negative, closer to 0 is better)
        if "drawdown" in primary_metric.lower():
            # For drawdown, sort by absolute value (smaller absolute value is better)
            ranked_models = sorted(valid_models.items(), key=lambda x: abs(x[1]))
        else:
            # Regular sorting based on directionality
            ranked_models = sorted(valid_models.items(), key=lambda x: x[1], reverse=maximize)

        # Create comparison summary
        comparison: dict[str, Any] = {
            "primary_metric": primary_metric,
            "n_models": len(batch_results),
            "ranking": [{"model": name, primary_metric: value} for name, value in ranked_models],
            "best_model": ranked_models[0][0] if ranked_models else None,
            "model_details": {},
        }

        # Add detailed results for each model
        for name, result in batch_results.items():
            comparison["model_details"][name] = result.summary()

        return comparison
