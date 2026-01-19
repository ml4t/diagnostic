"""Statistical test registry for evaluation framework.

This module provides a centralized registry for statistical tests
used in the evaluation framework, including tier defaults.
"""

from collections.abc import Callable
from typing import Any


class StatTestRegistry:
    """Registry of statistical tests for evaluation.

    The StatTestRegistry provides a centralized place to register and query
    statistical tests, including their tier defaults.

    Attributes
    ----------
    _tests : dict[str, Callable]
        Mapping of test names to test functions
    _tier_defaults : dict[int, list[str]]
        Default tests for each evaluation tier

    Examples
    --------
    >>> registry = StatTestRegistry()
    >>> registry.register("dsr", dsr_func, tiers=[1])
    >>> func = registry.get("dsr")
    """

    _instance: "StatTestRegistry | None" = None

    def __init__(self) -> None:
        """Initialize empty registry."""
        self._tests: dict[str, Callable[..., Any]] = {}
        self._tier_defaults: dict[int, list[str]] = {1: [], 2: [], 3: []}

    @classmethod
    def default(cls) -> "StatTestRegistry":
        """Get or create the default singleton registry instance.

        Returns
        -------
        StatTestRegistry
            The default registry instance with standard tests registered
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
        tiers: list[int] | None = None,
    ) -> None:
        """Register a statistical test with the registry.

        Parameters
        ----------
        name : str
            Unique name for the test
        func : Callable
            Function that performs the test.
            Should return a dict with test results
        tiers : list[int], optional
            Evaluation tiers where this test is a default
        """
        self._tests[name] = func
        if tiers:
            for tier in tiers:
                if tier in self._tier_defaults and name not in self._tier_defaults[tier]:
                    self._tier_defaults[tier].append(name)

    def get(self, name: str) -> Callable[..., Any]:
        """Get a test function by name.

        Parameters
        ----------
        name : str
            Name of the test

        Returns
        -------
        Callable
            The test function

        Raises
        ------
        KeyError
            If test name is not registered
        """
        if name not in self._tests:
            raise KeyError(f"Unknown test: {name}. Available: {list(self._tests.keys())}")
        return self._tests[name]

    def get_by_tier(self, tier: int) -> list[str]:
        """Get default tests for a specific tier.

        Parameters
        ----------
        tier : int
            Evaluation tier (1, 2, or 3)

        Returns
        -------
        list[str]
            List of default test names for the tier
        """
        return self._tier_defaults.get(tier, []).copy()

    def list_tests(self) -> list[str]:
        """List all registered test names.

        Returns
        -------
        list[str]
            Sorted list of test names
        """
        return sorted(self._tests.keys())

    def __contains__(self, name: str) -> bool:
        """Check if a test is registered."""
        return name in self._tests

    def _register_defaults(self) -> None:
        """Register default statistical tests."""
        from . import stats

        self.register("dsr", stats.deflated_sharpe_ratio_from_statistics, tiers=[1])
        self.register("hac_ic", stats.robust_ic, tiers=[2])
        self.register("fdr", stats.benjamini_hochberg_fdr, tiers=[1])
        self.register("whites_reality_check", stats.whites_reality_check, tiers=[])
