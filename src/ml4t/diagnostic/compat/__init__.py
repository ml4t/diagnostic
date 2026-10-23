"""Explicit migration adapters; native Diagnostic defaults remain unchanged."""

from importlib import import_module
from types import ModuleType

from ml4t.diagnostic.compat import alphalens, empyrical


def __getattr__(name: str) -> ModuleType:
    if name == "pyfolio":
        module = import_module(f"{__name__}.pyfolio")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["alphalens", "empyrical", "pyfolio"]
