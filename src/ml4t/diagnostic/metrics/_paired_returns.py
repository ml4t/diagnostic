"""One-dimensional supplied return alignment, without filling or calendar inference."""

from collections.abc import Sequence

import numpy as np
import pandas as pd
import polars as pl

type PairedInput = np.ndarray | pd.Series | pl.Series | Sequence[float]


def _values(value: PairedInput) -> np.ndarray:
    if isinstance(value, pd.Series):
        return value.to_numpy(dtype=float, na_value=np.nan)
    if isinstance(value, pl.Series):
        return np.asarray(value.to_numpy(), dtype=float)
    return np.asarray(value, dtype=float)


def paired_returns_1d(
    returns: PairedInput, benchmark: PairedInput
) -> tuple[np.ndarray, np.ndarray, tuple[object, ...]]:
    """Align two Series on labels; other supplied arrays keep positional rows."""
    if isinstance(returns, pd.Series) and isinstance(benchmark, pd.Series):
        aligned = pd.concat([returns, benchmark], axis=1)
        values, factor = (_values(aligned.iloc[:, index]) for index in (0, 1))
        keys = tuple(aligned.index)
    else:
        values, factor = _values(returns), _values(benchmark)
        keys = (
            tuple(returns.index)
            if isinstance(returns, pd.Series)
            else tuple(benchmark.index)
            if isinstance(benchmark, pd.Series)
            else tuple(range(len(values)))
            if values.ndim
            else ()
        )
    if values.ndim != 1 or factor.ndim != 1 or values.shape != factor.shape:
        raise ValueError("paired returns require equal-length one-dimensional values")
    return values, factor, keys
