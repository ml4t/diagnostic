"""Familiar GPD entry points over one native likelihood and fitting workflow."""

from collections.abc import Callable

import numpy as np
import pandas as pd

from ml4t.diagnostic.metrics.tail_risk import (
    _fit,
    _log_likelihood,
    _quantile,
    _shortfall,
    estimate_gpd_tail_risk,
)

from ._returns import ReturnInput, as_array


def gpd_risk_estimates(returns: ReturnInput, var_p: float = 0.01) -> np.ndarray | pd.Series:
    values = as_array(returns)
    result = estimate_gpd_tail_risk(values, tail_probability=var_p, method="empyrical_raw_tail")
    output = np.array(
        [
            result.threshold,
            result.scale,
            result.shape,
            result.value_at_risk,
            result.expected_shortfall,
        ]
    )
    if result.status != "ok":
        output[:] = np.nan
    return pd.Series(output) if isinstance(returns, pd.Series) else output


def gpd_risk_estimates_aligned(returns: ReturnInput, var_p: float = 0.01) -> np.ndarray | pd.Series:
    return gpd_risk_estimates(returns, var_p)


def gpd_var_calculator(
    threshold: float,
    scale_param: float,
    shape_param: float,
    probability: float,
    total_n: int,
    exceedance_n: int,
) -> float:
    return _quantile(threshold, scale_param, shape_param, probability, total_n, exceedance_n)


def gpd_es_calculator(
    var_estimate: float, threshold: float, scale_param: float, shape_param: float
) -> float:
    return _shortfall(var_estimate, threshold, scale_param, shape_param)


def gpd_loglikelihood_minimizer_aligned(price_data: ReturnInput) -> list[float | bool]:
    result = _fit(as_array(price_data), reference=True)
    return list(result) if result is not None else [False, False]


def gpd_loglikelihood(params: list[float] | np.ndarray, price_data: ReturnInput) -> float:
    return -gpd_loglikelihood_scale_and_shape(float(params[0]), float(params[1]), price_data)


def gpd_loglikelihood_factory(
    price_data: ReturnInput,
) -> Callable[[list[float] | np.ndarray], float]:
    return lambda params: gpd_loglikelihood(params, price_data)


def gpd_loglikelihood_scale_and_shape(scale: float, shape: float, price_data: ReturnInput) -> float:
    if shape < 0:
        return -np.inf
    return _log_likelihood(scale, shape, as_array(price_data))


def gpd_loglikelihood_scale_and_shape_factory(
    price_data: ReturnInput,
) -> Callable[[list[float] | np.ndarray], float]:
    return lambda params: -gpd_loglikelihood_scale_and_shape(params[0], params[1], price_data)


def gpd_loglikelihood_scale_only(scale: float, price_data: ReturnInput) -> float:
    return _log_likelihood(float(np.asarray(scale).item()), 0.0, as_array(price_data))


def gpd_loglikelihood_scale_only_factory(price_data: ReturnInput) -> Callable[[float], float]:
    return lambda scale: -gpd_loglikelihood_scale_only(scale, price_data)
