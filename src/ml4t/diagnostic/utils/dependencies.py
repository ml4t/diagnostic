"""Optional dependency checking and validation utilities.

This module provides centralized dependency checking for optional ML libraries.
It ensures clear error messages and graceful degradation when dependencies
are unavailable.

Example:
    >>> from ml4t.diagnostic.utils.dependencies import check_dependency, DEPS
    >>>
    >>> # Check if LightGBM is available
    >>> if check_dependency("lightgbm"):
    ...     import lightgbm as lgb
    ...     # Use LightGBM
    ... else:
    ...     print("LightGBM not available, using fallback")
    >>>
    >>> # Get dependency information
    >>> print(DEPS.lightgbm.install_cmd)  # pip install lightgbm
    >>> print(DEPS.lightgbm.purpose)      # Feature importance, boosting models
"""

from __future__ import annotations

import importlib
import warnings
from dataclasses import dataclass


@dataclass
class DependencyInfo:
    """Information about an optional dependency.

    Attributes:
        name: Package name (e.g., "lightgbm")
        import_name: Import name (e.g., "lightgbm" or "lgb")
        install_cmd: pip install command
        purpose: What this dependency is used for
        features: List of features requiring this dependency
        alternatives: Alternative packages that can be used instead
    """

    name: str
    import_name: str
    install_cmd: str
    purpose: str
    features: list[str]
    alternatives: list[str] | None = None

    def __post_init__(self) -> None:
        if self.alternatives is None:
            self.alternatives = []

    @property
    def is_available(self) -> bool:
        """Check if this dependency is installed."""
        try:
            importlib.import_module(self.import_name)
            return True
        except ImportError:
            return False

    def require(self, feature: str | None = None) -> None:
        """Raise ImportError with helpful message if dependency not available.

        Args:
            feature: Specific feature name requesting this dependency

        Raises:
            ImportError: If dependency is not available
        """
        if not self.is_available:
            msg = f"{self.name} is required"
            if feature:
                msg += f" for {feature}"
            msg += f". Install with: {self.install_cmd}"

            if self.alternatives:
                msg += f"\n  Alternatives: {', '.join(self.alternatives)}"

            raise ImportError(msg)

    def warn_if_missing(self, feature: str | None = None, action: str = "skipping") -> bool:
        """Warn if dependency is missing, return availability status.

        Args:
            feature: Specific feature name requesting this dependency
            action: What will happen without this dependency (e.g., "skipping", "using fallback")

        Returns:
            bool: True if available, False if missing
        """
        if not self.is_available:
            msg = f"{self.name} not available - {action}"
            if feature:
                msg += f" {feature}"
            msg += f". Install with: {self.install_cmd}"

            if self.alternatives:
                msg += f" (or use: {', '.join(self.alternatives)})"

            warnings.warn(msg, UserWarning, stacklevel=2)
            return False
        return True


class OptionalDependencies:
    """Registry of all optional dependencies with their metadata."""

    def __init__(self):
        self._deps: dict[str, DependencyInfo] = {}
        self._register_dependencies()

    def _register_dependencies(self):
        """Register all known optional dependencies."""

        # ML Libraries
        self._deps["lightgbm"] = DependencyInfo(
            name="LightGBM",
            import_name="lightgbm",
            install_cmd="pip install lightgbm",
            purpose="Feature importance (MDI, permutation), boosting models",
            features=[
                "FeatureOutcome.run_analysis (ML importance)",
                "MDI feature importance",
                "Permutation importance",
            ],
            alternatives=["xgboost", "scikit-learn RandomForest"],
        )

        self._deps["xgboost"] = DependencyInfo(
            name="XGBoost",
            import_name="xgboost",
            install_cmd="pip install xgboost",
            purpose="Domain classifier drift detection, boosting models",
            features=[
                "compute_domain_classifier_drift (XGBoost backend)",
                "Drift detection with XGBoost",
            ],
            alternatives=["lightgbm", "scikit-learn RandomForest"],
        )

        self._deps["shap"] = DependencyInfo(
            name="SHAP",
            import_name="shap",
            install_cmd="pip install shap",
            purpose="Shapley value feature importance and interactions",
            features=[
                "SHAP-based feature importance",
                "Feature interactions analysis",
                "Model interpretation",
            ],
            alternatives=["Permutation importance", "MDI importance"],
        )

        # Other optional dependencies
        self._deps["plotly"] = DependencyInfo(
            name="Plotly",
            import_name="plotly",
            install_cmd="pip install plotly",
            purpose="Interactive visualizations and dashboards",
            features=[
                "create_evaluation_dashboard",
                "Interactive plots",
                "HTML reports",
            ],
            alternatives=["matplotlib", "seaborn"],
        )

    def __getattr__(self, name: str) -> DependencyInfo:
        """Access dependencies as attributes (e.g., DEPS.lightgbm)."""
        if name in self._deps:
            return self._deps[name]
        raise AttributeError(f"Unknown dependency: {name}")

    def __getitem__(self, name: str) -> DependencyInfo:
        """Access dependencies as items (e.g., DEPS["lightgbm"])."""
        return self._deps[name]

    def get(self, name: str, default=None) -> DependencyInfo | None:
        """Get dependency info, return default if not found."""
        return self._deps.get(name, default)

    def check(self, name: str) -> bool:
        """Check if a dependency is available."""
        if name in self._deps:
            return self._deps[name].is_available
        return False

    def check_multiple(self, names: list[str]) -> dict[str, bool]:
        """Check availability of multiple dependencies.

        Args:
            names: List of dependency names to check

        Returns:
            Dict mapping dependency name to availability status
        """
        return {name: self.check(name) for name in names}

    def get_missing(self, names: list[str]) -> list[str]:
        """Get list of missing dependencies from a list.

        Args:
            names: List of dependency names to check

        Returns:
            List of missing dependency names
        """
        return [name for name in names if not self.check(name)]

    def warn_missing(self, names: list[str], feature: str | None = None) -> list[str]:
        """Warn about missing dependencies, return list of missing ones.

        Args:
            names: List of dependency names to check
            feature: Feature name using these dependencies

        Returns:
            List of missing dependency names
        """
        missing = []
        for name in names:
            if name in self._deps and not self._deps[name].is_available:
                self._deps[name].warn_if_missing(feature)
                missing.append(name)
        return missing

    def summary(self) -> str:
        """Generate summary of all dependencies and their status."""
        lines = ["Optional Dependencies Status:"]
        lines.append("=" * 60)

        for _name, info in sorted(self._deps.items()):
            status = "✓ Installed" if info.is_available else "✗ Missing"
            lines.append(f"{info.name:15} {status:15} {info.purpose}")
            if not info.is_available:
                lines.append(f"  → Install: {info.install_cmd}")

        return "\n".join(lines)


# Global instance
DEPS = OptionalDependencies()


def check_dependency(name: str) -> bool:
    """Quick check if a dependency is available.

    Args:
        name: Dependency name (e.g., "lightgbm", "shap")

    Returns:
        bool: True if available, False otherwise

    Example:
        >>> if check_dependency("lightgbm"):
        ...     import lightgbm as lgb
        ...     # Use LightGBM
    """
    return DEPS.check(name)


def require_dependency(name: str, feature: str | None = None) -> None:
    """Require a dependency, raise ImportError if missing.

    Args:
        name: Dependency name
        feature: Feature name requiring this dependency

    Raises:
        ImportError: If dependency is not available

    Example:
        >>> require_dependency("shap", "SHAP analysis")
        >>> import shap  # Safe to import now
    """
    if name in DEPS._deps:
        DEPS[name].require(feature)
    else:
        raise ImportError(f"Unknown dependency: {name}")


def warn_if_missing(name: str, feature: str | None = None, action: str = "skipping") -> bool:
    """Warn if dependency is missing, return availability status.

    Args:
        name: Dependency name
        feature: Feature name requesting this dependency
        action: What will happen without this dependency

    Returns:
        bool: True if available, False if missing

    Example:
        >>> if warn_if_missing("lightgbm", "feature importance", "using fallback"):
        ...     import lightgbm as lgb
        ...     # Use LightGBM
        ... else:
        ...     # Use fallback method
    """
    if name in DEPS._deps:
        return DEPS[name].warn_if_missing(feature, action)
    warnings.warn(f"Unknown dependency: {name}", stacklevel=2)
    return False


def get_dependency_summary() -> str:
    """Get summary of all optional dependencies and their status.

    Returns:
        str: Formatted summary of dependencies

    Example:
        >>> print(get_dependency_summary())
        Optional Dependencies Status:
        ...
    """
    return DEPS.summary()
