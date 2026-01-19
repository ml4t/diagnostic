"""Metric registry for evaluation metrics with metadata.

This module provides a centralized registry for evaluation metrics,
including directionality (whether higher is better) and tier defaults.
"""

from collections.abc import Callable
from typing import Any


class MetricRegistry:
    """Registry of evaluation metrics with metadata.

    The MetricRegistry provides a centralized place to register and query
    metrics, including their computation functions, directionality (whether
    higher values are better), and tier defaults.

    Attributes
    ----------
    _metrics : dict[str, Callable]
        Mapping of metric names to computation functions
    _directionality : dict[str, bool]
        Mapping of metric names to directionality (True = higher is better)
    _tier_defaults : dict[int, list[str]]
        Default metrics for each evaluation tier

    Examples
    --------
    >>> registry = MetricRegistry()
    >>> registry.register("sharpe", sharpe_func, maximize=True, tiers=[1, 2, 3])
    >>> func = registry.get("sharpe")
    >>> registry.is_maximize("sharpe")
    True
    """

    _instance: "MetricRegistry | None" = None

    def __init__(self) -> None:
        """Initialize empty registry."""
        self._metrics: dict[str, Callable[..., Any]] = {}
        self._directionality: dict[str, bool] = {}
        self._tier_defaults: dict[int, list[str]] = {1: [], 2: [], 3: []}

    @classmethod
    def default(cls) -> "MetricRegistry":
        """Get or create the default singleton registry instance.

        Returns
        -------
        MetricRegistry
            The default registry instance with standard metrics registered
        """
        if cls._instance is None:
            cls._instance = cls()
            cls._instance._register_defaults()
        return cls._instance

    @classmethod
    def reset_default(cls) -> None:
        """Reset the default singleton instance (primarily for testing)."""
        cls._instance = None

    def register(
        self,
        name: str,
        func: Callable[..., Any],
        maximize: bool = True,
        tiers: list[int] | None = None,
    ) -> None:
        """Register a metric with the registry.

        Parameters
        ----------
        name : str
            Unique name for the metric
        func : Callable
            Function that computes the metric.
            Signature: (predictions, actual, strategy_returns) -> float
        maximize : bool, default True
            Whether higher values are better (True) or lower (False)
        tiers : list[int], optional
            Evaluation tiers where this metric is a default
        """
        self._metrics[name] = func
        self._directionality[name] = maximize
        if tiers:
            for tier in tiers:
                if tier in self._tier_defaults and name not in self._tier_defaults[tier]:
                    self._tier_defaults[tier].append(name)

    def get(self, name: str) -> Callable[..., Any]:
        """Get a metric function by name.

        Parameters
        ----------
        name : str
            Name of the metric

        Returns
        -------
        Callable
            The metric computation function

        Raises
        ------
        KeyError
            If metric name is not registered
        """
        if name not in self._metrics:
            raise KeyError(f"Unknown metric: {name}. Available: {list(self._metrics.keys())}")
        return self._metrics[name]

    def is_maximize(self, name: str) -> bool:
        """Get whether higher values are better for a metric.

        Parameters
        ----------
        name : str
            Name of the metric

        Returns
        -------
        bool
            True if higher values are better, False otherwise
        """
        if name in self._directionality:
            return self._directionality[name]
        return self._infer_directionality(name)

    def _infer_directionality(self, name: str) -> bool:
        """Infer directionality for unknown metrics based on naming conventions."""
        normalized = name.lower().replace("-", "_").replace(" ", "_")

        if any(term in normalized for term in ["drawdown", "risk", "error", "loss", "volatility"]):
            return False

        if any(
            term in normalized
            for term in ["return", "profit", "gain", "ratio", "score", "coefficient"]
        ):
            return True

        return True  # Default to higher is better

    def get_by_tier(self, tier: int) -> list[str]:
        """Get default metrics for a specific tier.

        Parameters
        ----------
        tier : int
            Evaluation tier (1, 2, or 3)

        Returns
        -------
        list[str]
            List of default metric names for the tier
        """
        return self._tier_defaults.get(tier, []).copy()

    def list_metrics(self) -> list[str]:
        """List all registered metric names.

        Returns
        -------
        list[str]
            Sorted list of metric names
        """
        return sorted(self._metrics.keys())

    def __contains__(self, name: str) -> bool:
        """Check if a metric is registered."""
        return name in self._metrics

    def _register_defaults(self) -> None:
        """Register default metrics."""
        from . import metrics

        # Core metrics with tier assignments
        self.register(
            "ic",
            lambda pred, actual, _returns: metrics.information_coefficient(pred, actual),
            maximize=True,
            tiers=[1, 2, 3],
        )
        self.register(
            "sharpe",
            lambda _pred, _actual, returns: metrics.sharpe_ratio(returns),
            maximize=True,
            tiers=[1, 2],
        )
        self.register(
            "sortino",
            lambda _pred, _actual, returns: metrics.sortino_ratio(returns),
            maximize=True,
            tiers=[1],
        )
        self.register(
            "max_drawdown",
            lambda _pred, _actual, returns: metrics.maximum_drawdown(returns),
            maximize=False,
            tiers=[1],
        )
        self.register(
            "hit_rate",
            lambda pred, actual, _returns: metrics.hit_rate(pred, actual),
            maximize=True,
            tiers=[1, 2, 3],
        )

        # Additional directionality mappings for common metric names
        self._register_directionality_defaults()

    def _register_directionality_defaults(self) -> None:
        """Register directionality for common metric names (for get_metric_directionality)."""
        # Performance metrics (higher is better)
        for name in [
            "sharpe_ratio",
            "sortino_ratio",
            "calmar",
            "calmar_ratio",
            "information_ratio",
            "omega_ratio",
            "profit_factor",
            "total_return",
            "mean_return",
            "cumulative_return",
            "annualized_return",
            "win_rate",
            "accuracy",
            "information_coefficient",
            "ic_mean",
            "spearman",
            "pearson",
            "r_squared",
            "r2",
            "t_statistic",
            "z_score",
        ]:
            self._directionality[name] = True

        # Risk metrics (lower is better)
        for name in [
            "maximum_drawdown",
            "drawdown",
            "volatility",
            "downside_deviation",
            "value_at_risk",
            "var",
            "cvar",
            "conditional_value_at_risk",
            "tracking_error",
            "beta",
            "p_value",
        ]:
            self._directionality[name] = False
