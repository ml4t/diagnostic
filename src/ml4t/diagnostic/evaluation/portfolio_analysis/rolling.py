"""Shared row-window layout for optimized native metrics and migration dispatch."""

from numbers import Integral

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def _window_length(window: int) -> int:
    if isinstance(window, bool) or not isinstance(window, Integral) or window < 1:
        raise ValueError("window must be a positive integer")
    return int(window)


def _row_windows(
    values: np.ndarray, window: int, *, short: bool = False, mutable: bool = False
) -> np.ndarray:
    """Return (endpoint, row, ...) views without changing native warmup policy."""
    width = _window_length(window)
    if values.ndim == 0:
        raise ValueError("row windows require an observation axis")
    if short and len(values):
        width = min(width, len(values))
    if len(values) < width:
        raise IndexError("window exceeds the number of observations")
    windows = sliding_window_view(values, width, axis=0, writeable=mutable)
    return np.moveaxis(windows, -1, 1)
