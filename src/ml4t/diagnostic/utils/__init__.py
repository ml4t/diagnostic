"""Utility functions for ml4t-diagnostic.

This module contains helper functions, configuration loaders, and other
utilities used throughout the library.
"""

from ml4t.diagnostic.utils.config import (
    ConfigError,
    EvaluationConfig,
    create_example_config,
    load_config,
)
from ml4t.diagnostic.utils.dependencies import (
    DEPS,
    DependencyInfo,
    OptionalDependencies,
    check_dependency,
    get_dependency_summary,
    require_dependency,
    warn_if_missing,
)
from ml4t.diagnostic.utils.sessions import (
    assign_session_dates,
    get_complete_sessions,
)

__all__: list[str] = [
    "ConfigError",
    "EvaluationConfig",
    "create_example_config",
    "load_config",
    "assign_session_dates",
    "get_complete_sessions",
    # Dependency checking
    "DEPS",
    "DependencyInfo",
    "OptionalDependencies",
    "check_dependency",
    "require_dependency",
    "warn_if_missing",
    "get_dependency_summary",
]
