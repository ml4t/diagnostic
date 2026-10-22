"""Explicit Pyfolio analytical adapters without a Pyfolio runtime dependency."""

from importlib import import_module
from types import ModuleType

from . import capacity, perf_attrib, round_trips, tears, timeseries, txn, utils


def __getattr__(name: str) -> ModuleType:
    if name == "plotting":
        module = import_module(f"{__name__}.plotting")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "capacity",
    "perf_attrib",
    "plotting",
    "round_trips",
    "tears",
    "timeseries",
    "txn",
    "utils",
]
