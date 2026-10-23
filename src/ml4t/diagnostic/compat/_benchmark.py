"""Legacy benchmark alignment and output projection over native statistical kernels."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ml4t.diagnostic.evaluation.portfolio_analysis import metrics as native
from ml4t.diagnostic.evaluation.portfolio_analysis.benchmark import benchmark_batting_average
from ml4t.diagnostic.metrics.risk_adjusted import periodic_sharpe_ratio

from ._returns import (
    MetricOutput,
    ReturnInput,
    adjust_rate,
    as_array,
    output_array,
    reduce_columns,
    select_benchmark_sign,
)


def _aligned_pair(
    returns: ReturnInput, factor_returns: ReturnInput
) -> tuple[ReturnInput, ReturnInput]:
    as_array(returns)
    as_array(factor_returns)
    if (
        isinstance(returns, np.ndarray)
        and isinstance(factor_returns, np.ndarray)
        and len(returns) == len(factor_returns)
    ):
        return returns, factor_returns
    frames = [
        value
        if isinstance(value, pd.Series | pd.DataFrame)
        else pd.Series(value)
        if value.ndim == 1
        else pd.DataFrame(value)
        for value in (returns, factor_returns)
    ]
    sort_dates = all(isinstance(value.index, pd.DatetimeIndex) for value in frames) and not frames[
        0
    ].index.equals(frames[1].index)
    aligned = pd.concat(frames, axis=1, sort=sort_dates)
    if aligned.shape[1] != 2:
        raise ValueError(
            "unaligned benchmark functions need two series; use aligned arrays or apply per column"
        )
    return aligned.iloc[:, 0], aligned.iloc[:, 1]


def _arrays(returns: ReturnInput, factor_returns: ReturnInput) -> tuple[np.ndarray, np.ndarray]:
    values, factor = as_array(returns), as_array(factor_returns)
    if len(values) != len(factor):
        raise ValueError("aligned returns and factor returns must have equal row counts")
    if values.ndim == 2 and factor.ndim == 1:
        factor = factor[:, None]
    elif values.ndim == 1 and factor.ndim == 2 and factor.shape[1] == 1:
        factor = factor[:, 0]
    try:
        factor = np.broadcast_to(factor, values.shape)
    except ValueError as error:
        raise ValueError(
            "factor returns must match the strategy columns or be a single benchmark column"
        ) from error
    return values, factor


def _output(
    returns: ReturnInput,
    factor: ReturnInput | float,
    shape: tuple[int, ...],
    out: np.ndarray | None,
    *other: ReturnInput | float | None,
) -> np.ndarray:
    return output_array(returns, shape, out, factor, *other)


def _betas(values: np.ndarray, factor: np.ndarray) -> np.ndarray:
    columns = values[:, None] if values.ndim == 1 else values
    benchmarks = factor[:, None] if factor.ndim == 1 else factor
    result = np.array(
        [
            native._paired_linear_regression(
                columns[:, index],
                benchmarks[:, index],
                variance_floor=1e-30,
                center_dependent=False,
            )[1]
            for index in range(columns.shape[1])
        ]
    )
    return result.reshape(values.shape[1:])


def _alphas(
    returns: ReturnInput,
    factor: ReturnInput,
    risk_free: float | ReturnInput,
    betas: np.ndarray,
    scale: float,
) -> np.ndarray:
    original = as_array(returns)
    result = np.full(original.shape[1:], np.nan)
    if len(original) < 2:
        return result
    adjusted, benchmark = _arrays(adjust_rate(returns, risk_free), adjust_rate(factor, risk_free))
    if adjusted.shape[1:] != original.shape[1:]:
        raise ValueError("periodic risk-free adjustment must preserve strategy columns")
    columns = adjusted[:, None] if adjusted.ndim == 1 else adjusted
    benchmarks = benchmark[:, None] if benchmark.ndim == 1 else benchmark
    for index in range(columns.shape[1]):
        slope = float(betas[()] if betas.ndim == 0 else betas[index])
        periodic = native._mean_regression_residual(columns[:, index], benchmarks[:, index], slope)
        result.flat[index] = native._annualize_growth(1 + periodic, 1, scale)
    return result


def alpha_beta_aligned(
    returns: ReturnInput,
    factor_returns: ReturnInput,
    risk_free: float | ReturnInput = 0.0,
    period: str = "daily",
    annualization: float | None = None,
    out: np.ndarray | None = None,
) -> np.ndarray:
    """Use positional rows, paired OLS and compounded periodic excess-return intercepts."""
    from .empyrical import annualization_factor

    scale = annualization_factor(period, annualization)
    values, factor = _arrays(returns, factor_returns)
    output = _output(returns, factor_returns, values.shape[1:] + (2,), out, risk_free)
    betas = _betas(values, factor)
    alphas = _alphas(returns, factor_returns, risk_free, betas, scale)
    staged = np.stack([alphas, betas], axis=-1)
    output[...] = staged
    return output


def alpha_beta(
    returns: ReturnInput,
    factor_returns: ReturnInput,
    risk_free: float | ReturnInput = 0.0,
    period: str = "daily",
    annualization: float | None = None,
    out: np.ndarray | None = None,
) -> np.ndarray:
    """Align labeled inputs before the same native paired regression calculation."""
    aligned = _aligned_pair(returns, factor_returns)
    if out is not None:
        _output(returns, factor_returns, as_array(aligned[0]).shape[1:] + (2,), out, risk_free)
    return alpha_beta_aligned(*aligned, risk_free, period, annualization, out)


def beta_aligned(
    returns: ReturnInput,
    factor_returns: ReturnInput,
    risk_free: float = 0.0,
    out: np.ndarray | None = None,
) -> MetricOutput:
    """Return paired column slopes; risk_free is accepted but unused, as in the reference."""
    values, factor = _arrays(returns, factor_returns)
    output = _output(returns, factor_returns, values.shape[1:], out)
    output[...] = _betas(values, factor)
    return output.item() if values.ndim == 1 else output


def beta(
    returns: ReturnInput,
    factor_returns: ReturnInput,
    risk_free: float = 0.0,
    out: np.ndarray | None = None,
) -> MetricOutput:
    """Align labeled observations; beta_aligned keeps the positional contract."""
    aligned = _aligned_pair(returns, factor_returns)
    if out is not None:
        _output(returns, factor_returns, as_array(aligned[0]).shape[1:], out)
    return beta_aligned(*aligned, risk_free, out)


def alpha_aligned(
    returns: ReturnInput,
    factor_returns: ReturnInput,
    risk_free: float | ReturnInput = 0.0,
    period: str = "daily",
    annualization: float | None = None,
    out: np.ndarray | None = None,
    _beta: float | np.ndarray | None = None,
) -> MetricOutput:
    """Compound mean residuals with the supplied or shared native fitted slope."""
    from .empyrical import annualization_factor

    scale = annualization_factor(period, annualization)
    values, factor = _arrays(returns, factor_returns)
    output = _output(returns, factor_returns, values.shape[1:], out, risk_free, _beta)
    if _beta is None:
        betas = _betas(values, factor)
    else:
        supplied = np.asarray(_beta, dtype=float)
        if values.ndim == 1 and supplied.size == 1:
            supplied = supplied.reshape(())
        betas = np.broadcast_to(supplied, values.shape[1:])
    output[...] = _alphas(returns, factor_returns, risk_free, betas, scale)
    if values.ndim == 1:
        return output.item()
    if out is None and isinstance(returns, pd.DataFrame) and len(returns) >= 2:
        return pd.Series(output, index=returns.columns, copy=False)
    return output


def alpha(
    returns: ReturnInput,
    factor_returns: ReturnInput,
    risk_free: float | ReturnInput = 0.0,
    period: str = "daily",
    annualization: float | None = None,
    out: np.ndarray | None = None,
    _beta: float | np.ndarray | None = None,
) -> MetricOutput:
    """Align before the explicitly compounded periodic alpha calculation."""
    aligned = _aligned_pair(returns, factor_returns)
    if out is not None:
        _output(returns, factor_returns, as_array(aligned[0]).shape[1:], out, risk_free, _beta)
    return alpha_aligned(*aligned, risk_free, period, annualization, out, _beta)


def excess_sharpe(
    returns: ReturnInput, factor_returns: float | ReturnInput, out: np.ndarray | None = None
) -> MetricOutput:
    """Unannualized active-return Sharpe, not the native annual information ratio."""
    _output(returns, factor_returns, as_array(returns).shape[1:], out)
    return reduce_columns(
        returns,
        lambda column, _: periodic_sharpe_ratio(column),
        adjusted=adjust_rate(returns, factor_returns),
        out=out,
        min_length=2,
    )


def _sign_alpha_beta(
    returns: ReturnInput, factor_returns: ReturnInput, positive: bool, **kwargs: Any
) -> np.ndarray:
    selected = select_benchmark_sign(returns, factor_returns, positive=positive)
    shape = as_array(selected[0]).shape[1:] + (2,)
    _output(returns, factor_returns, shape, kwargs.get("out"), kwargs.get("risk_free"))
    return alpha_beta(*selected, **kwargs)


def up_alpha_beta(returns: ReturnInput, factor_returns: ReturnInput, **kwargs: Any) -> np.ndarray:
    """Fit only strictly positive benchmark observations; zeros are excluded."""
    return _sign_alpha_beta(returns, factor_returns, True, **kwargs)


def down_alpha_beta(returns: ReturnInput, factor_returns: ReturnInput, **kwargs: Any) -> np.ndarray:
    """Fit only strictly negative benchmark observations; zeros are excluded."""
    return _sign_alpha_beta(returns, factor_returns, False, **kwargs)


def batting_average(returns: ReturnInput, factor_returns: ReturnInput) -> pd.Series:
    """Project native finite-pair batting statistics with legacy zero-in-up labels."""
    values, factor = as_array(returns), as_array(factor_returns)
    if isinstance(returns, np.ndarray) and values.ndim == 2:
        values, factor = _arrays(returns, factor_returns)
        returns, factor_returns = values.ravel(), factor.ravel()
    result = benchmark_batting_average(returns, factor_returns, up_includes_zero=True)
    return pd.Series(
        [result["overall"], result["up"], result["down"]],
        index=["batting average", "up market", "down market"],
        dtype=float,
    )
