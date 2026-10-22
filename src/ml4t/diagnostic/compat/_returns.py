"""Column and output adaptation shared by legacy return-metric signatures."""

from collections.abc import Callable, Iterator

import numpy as np
import pandas as pd

type ReturnInput = np.ndarray | pd.Series | pd.DataFrame
type MetricOutput = float | np.ndarray | pd.Series


def as_array(returns: ReturnInput) -> np.ndarray:
    if not isinstance(returns, (np.ndarray, pd.Series, pd.DataFrame)):
        raise TypeError("returns must be a Series, DataFrame or ndarray")
    values = (
        returns.to_numpy(dtype=float, na_value=np.nan)
        if isinstance(returns, (pd.Series, pd.DataFrame))
        else np.asarray(returns, dtype=float)
    )
    if values.ndim not in {1, 2}:
        raise ValueError("returns must have one or two dimensions")
    return values


def adjust_rate(returns: ReturnInput, rate: float | ReturnInput) -> ReturnInput:
    if isinstance(returns, pd.DataFrame) and isinstance(rate, (pd.Series, pd.DataFrame)):
        labels = rate.index if isinstance(rate, pd.Series) else rate.columns
        if not labels.is_unique or set(labels) != set(returns.columns):
            raise ValueError("a labeled DataFrame rate must name its return columns")
        rate = (
            rate.reindex(returns.columns)
            if isinstance(rate, pd.Series)
            else rate.reindex(columns=returns.columns)
        )
    return returns - rate


def select_benchmark_sign(
    returns: ReturnInput, factor: ReturnInput, *, positive: bool
) -> tuple[ReturnInput, ReturnInput]:
    """Retain legacy row selection versus ndarray cell selection and pandas labels."""
    values, benchmark = as_array(returns), as_array(factor)
    if isinstance(returns, pd.Series) and isinstance(factor, pd.Series):
        paired = ~(returns.isna() | factor.isna())
    elif values.ndim == benchmark.ndim == 1:
        paired = ~(np.isnan(values) | np.isnan(benchmark))
    else:
        paired = np.ones_like(benchmark, dtype=bool)
    mask = paired & (factor > 0 if positive else factor < 0)
    return returns[mask], factor[mask]


def _input_buffers(value: ReturnInput) -> Iterator[np.ndarray]:
    if isinstance(value, pd.DataFrame):
        for _, column in value.items():
            yield from _input_buffers(column)
    elif isinstance(value, pd.Series):
        array = value.array
        if isinstance(
            array, pd.arrays.FloatingArray | pd.arrays.IntegerArray | pd.arrays.BooleanArray
        ):
            # Public to_numpy/view cannot expose both nullable buffers without
            # copying. Bound this storage inspection to pandas' numeric masked
            # types and fail closed if their representation changes.
            data, mask = getattr(array, "_data", None), getattr(array, "_mask", None)
            if not isinstance(data, np.ndarray) or not isinstance(mask, np.ndarray):
                raise TypeError("cannot verify pandas masked input storage for out")
            yield data
            yield mask
        elif isinstance(value.dtype, np.dtype):
            yield value.to_numpy(copy=False)
        else:
            raise TypeError(
                "out requires NumPy or pandas numeric masked storage; "
                "convert this extension input to an owned NumPy snapshot first"
            )
    else:
        yield value


def output_array(
    returns: ReturnInput, shape: tuple[int, ...], out: np.ndarray | None, *other: object
) -> np.ndarray:
    if out is None:
        return np.empty(shape, dtype=float)
    if not isinstance(out, np.ndarray) or out.shape != shape:
        raise ValueError(f"out shape must be {shape}")
    if not np.issubdtype(out.dtype, np.floating):
        raise TypeError("out must have floating dtype")
    for value in (returns, *other):
        if isinstance(value, np.ndarray | pd.Series | pd.DataFrame) and any(
            np.shares_memory(out, buffer) for buffer in _input_buffers(value)
        ):
            raise ValueError("out must not overlap input storage")
    if not out.flags.writeable:
        raise ValueError("out must be writable")
    return out


def reduce_columns(
    returns: ReturnInput,
    function: Callable[[np.ndarray, int], float],
    *,
    out: np.ndarray | None = None,
    labeled: bool = False,
    min_length: int = 0,
    empty_scalar: bool = False,
    adjusted: ReturnInput | None = None,
    additional_inputs: tuple[object, ...] = (),
) -> MetricOutput:
    original = as_array(returns)
    values = original if adjusted is None else as_array(adjusted)
    if values.ndim != original.ndim or values.shape[1:] != original.shape[1:]:
        raise ValueError("adjusted returns must preserve the original column shape")
    if empty_scalar and len(original) == 0:
        return float("nan")
    shape = original.shape[1:]
    output = output_array(returns, shape, out, *additional_inputs)
    if len(original) < min_length:
        output[...] = np.nan
    elif original.ndim == 1:
        output[()] = function(values, 0)
    else:
        for index in range(original.shape[1]):
            output[index] = function(values[:, index], index)
    if original.ndim == 1:
        return output.item()
    if labeled and isinstance(returns, pd.DataFrame) and len(original) >= min_length:
        return pd.Series(output, index=returns.columns, copy=False)
    return output
