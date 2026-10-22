"""Legacy rolling presentation over the same qualified native-backed metric kernels."""

import warnings
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from ml4t.diagnostic.evaluation.portfolio_analysis.rolling import _row_windows, _window_length

from ._benchmark import _aligned_pair, _output
from ._returns import ReturnInput, as_array, select_benchmark_sign


def rolling_window(array: np.ndarray, length: int, mutable: bool = False) -> np.ndarray:
    """Expose shared row-window views, read-only unless explicitly requested otherwise."""
    return _row_windows(array, length, mutable=mutable)


def _finish(
    returns: ReturnInput,
    factor: ReturnInput | float,
    result: np.ndarray,
    out: np.ndarray | None,
    kwargs: dict[str, Any],
) -> Any:
    if out is not None:
        output = _output(returns, factor, result.shape, out, *kwargs.values())
        output[...] = result
        return output
    if isinstance(returns, pd.Series):
        index = returns.index[len(returns) - len(result) :]
        if result.ndim == 2:
            return pd.DataFrame(result, index=index)
        return pd.Series(result, index=index, dtype=float)
    return result


def _values(returns: ReturnInput) -> np.ndarray:
    values = as_array(returns)
    if values.ndim != 1:
        raise ValueError("rolling metrics require one series; apply per strategy column")
    return values


def _unary(name: str) -> Callable[..., Any]:
    def calculate(
        arr: ReturnInput, window: int, out: np.ndarray | None = None, **kwargs: Any
    ) -> Any:
        from . import empyrical

        width = _window_length(window)
        values = _values(arr)
        result = (
            np.asarray(
                getattr(empyrical, name)(_row_windows(values, width, short=True).T, **kwargs)
            )
            if len(values)
            else np.empty(0)
        )
        return _finish(arr, 0.0, result, out, kwargs)

    calculate.__name__ = "roll_" + name
    calculate.__doc__ = (
        f"Compute {name} on row windows, trimming warmup with one short overlong window."
    )
    return calculate


def _binary(name: str) -> Callable[..., Any]:
    def calculate(
        lhs: ReturnInput,
        rhs: ReturnInput,
        window: int,
        out: np.ndarray | None = None,
        **kwargs: Any,
    ) -> Any:
        from . import empyrical

        width = _window_length(window)
        values, factor = _values(lhs), _values(rhs)
        if len(values) != len(factor):
            raise ValueError("rolling aligned streams must have equal row counts")
        result = (
            np.asarray(
                getattr(empyrical, name)(
                    _row_windows(values, width, short=True).T,
                    _row_windows(factor, width, short=True).T,
                    **kwargs,
                )
            )
            if len(values)
            else np.empty(0)
        )
        return _finish(lhs, rhs, result, out, kwargs)

    calculate.__name__ = "roll_" + name
    calculate.__doc__ = f"Compute positional row-window {name}, with one short overlong window."
    return calculate


_UNARY = (
    "max_drawdown",
    "cagr",
    "annual_volatility",
    "sharpe_ratio",
    "sortino_ratio",
    "downside_risk",
)
_BINARY = ("excess_sharpe", "alpha_beta_aligned", "alpha", "alpha_aligned", "beta", "beta_aligned")
roll_max_drawdown = _unary("max_drawdown")
roll_cagr = _unary("cagr")
roll_annual_volatility = _unary("annual_volatility")
roll_sharpe_ratio = _unary("sharpe_ratio")
roll_sortino_ratio = _unary("sortino_ratio")
roll_downside_risk = _unary("downside_risk")
roll_excess_sharpe = _binary("excess_sharpe")
roll_alpha_beta_aligned = _binary("alpha_beta_aligned")
roll_alpha = _binary("alpha")
roll_alpha_aligned = _binary("alpha_aligned")
roll_beta = _binary("beta")
roll_beta_aligned = _binary("beta_aligned")


def roll_alpha_beta(
    returns: ReturnInput, factor_returns: ReturnInput, window: int = 10, **kwargs: Any
) -> Any:
    """Align full labeled streams before combined fits; other wrappers remain positional."""
    _window_length(window)
    aligned = _aligned_pair(returns, factor_returns)
    out = kwargs.pop("out", None)
    result = roll_alpha_beta_aligned(*aligned, window=window, **kwargs)
    if out is None:
        return result
    return _finish(returns, factor_returns, np.asarray(result), out, kwargs)


def roll(*args: ReturnInput, window: int, function: Callable[..., Any], **kwargs: Any) -> Any:
    """Apply a callback only to full observed row windows; overlong input stays empty."""
    width = _window_length(window)
    if not 1 <= len(args) <= 2:
        raise ValueError("roll requires one or two return streams")
    if len(args) == 2 and type(args[0]) is not type(args[1]):
        raise ValueError("rolling return streams must have the same type")
    if len(args) == 2 and len(args[0]) != len(args[1]):
        raise ValueError("rolling return streams must have equal row counts")
    for value in args:
        as_array(value)
    results = []
    for end in range(width, len(args[0]) + 1):
        slices = [
            value.iloc[end - width : end]
            if isinstance(value, pd.Series | pd.DataFrame)
            else value[end - width : end]
            for value in args
        ]
        results.append(function(*slices, **kwargs))
    if isinstance(args[0], pd.Series | pd.DataFrame):
        return pd.Series(results, index=args[0].index[width - 1 :], dtype=float)
    return np.asarray(results)


def up(
    returns: ReturnInput,
    factor_returns: ReturnInput,
    *,
    function: Callable[..., Any],
    **kwargs: Any,
) -> Any:
    """Forward a callback over the original strict-positive benchmark selection."""
    return function(*select_benchmark_sign(returns, factor_returns, positive=True), **kwargs)


def down(
    returns: ReturnInput,
    factor_returns: ReturnInput,
    *,
    function: Callable[..., Any],
    **kwargs: Any,
) -> Any:
    """Forward a callback over the original strict-negative benchmark selection."""
    return function(*select_benchmark_sign(returns, factor_returns, positive=False), **kwargs)


def _capture(name: str) -> Callable[..., Any]:
    def calculate(
        returns: ReturnInput, factor_returns: ReturnInput, window: int = 10, **kwargs: Any
    ) -> Any:
        from . import empyrical

        return roll(
            returns, factor_returns, window=window, function=getattr(empyrical, name), **kwargs
        )

    calculate.__name__ = "roll_" + name
    calculate.__doc__ = f"Compute {name} only for full windows, as in the legacy callback family."
    return calculate


_CAPTURE = ("up_capture", "down_capture", "up_down_capture")
roll_up_capture = _capture("up_capture")
roll_down_capture = _capture("down_capture")
roll_up_down_capture = _capture("up_down_capture")


def roll_downsize_risk(*args: Any, **kwargs: Any) -> Any:
    """Retain the historical misspelling with its deprecation notice."""
    warnings.warn(
        "roll_downsize_risk is deprecated; use roll_downside_risk", DeprecationWarning, stacklevel=2
    )
    return roll_downside_risk(*args, **kwargs)


__all__ = [
    "roll",
    "up",
    "down",
    "rolling_window",
    "roll_alpha_beta",
    "roll_downsize_risk",
    *["roll_" + name for name in (*_UNARY, *_BINARY, *_CAPTURE)],
]
