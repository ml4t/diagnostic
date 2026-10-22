"""Explicit Pyfolio analytical adapters without a Pyfolio runtime dependency."""

from importlib import import_module
from typing import Any

from . import capacity, perf_attrib, pos, round_trips, tears, timeseries, txn, utils


def __getattr__(name: str) -> Any:
    if name == "plotting":
        module = import_module(f"{__name__}.plotting")
        globals()[name] = module
        return module
    if name in tears.__all__:
        return getattr(tears, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    *tears.__all__,
    "capacity",
    "perf_attrib",
    "plotting",
    "pos",
    "round_trips",
    "tears",
    "timeseries",
    "txn",
    "utils",
]
