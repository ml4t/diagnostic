"""Import every module shipped by the installed ml4t-diagnostic wheel."""

from __future__ import annotations

import importlib
import pkgutil

import ml4t.diagnostic

OPTIONAL_MODULE_PREFIXES = (
    "ml4t.diagnostic.visualization",
    "ml4t.diagnostic.evaluation.trade_dashboard",
    "ml4t.diagnostic.integration",
)
OPTIONAL_MODULES = {
    "ml4t.diagnostic.evaluation.dashboard",
    "ml4t.diagnostic.evaluation.diagnostic_plots",
    "ml4t.diagnostic.evaluation.report_generation",
    "ml4t.diagnostic.evaluation.themes",
    "ml4t.diagnostic.evaluation.trade_shap_dashboard",
    "ml4t.diagnostic.evaluation.visualization",
}


def _is_optional(module_name: str) -> bool:
    return module_name in OPTIONAL_MODULES or module_name.startswith(OPTIONAL_MODULE_PREFIXES)


def main() -> None:
    """Fail when any shipped module has an undeclared import dependency."""
    modules = pkgutil.walk_packages(
        ml4t.diagnostic.__path__,
        f"{ml4t.diagnostic.__name__}.",
    )
    for module in modules:
        if not _is_optional(module.name):
            importlib.import_module(module.name)


if __name__ == "__main__":
    main()
