"""Inspectable generalized-Pareto tail fitting with explicit probability denominators."""

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
import polars as pl
from scipy import optimize, stats

type TailInput = np.ndarray | pd.Series | pl.Series | list[float]


@dataclass(frozen=True)
class GPDTailRiskResult:
    """Positive decimal loss quantiles; unsuccessful fits have undefined risk."""

    threshold: float
    scale: float
    shape: float
    value_at_risk: float
    expected_shortfall: float
    n_observations: int
    n_finite: int
    n_excluded: int
    n_losses: int
    n_exceedances: int
    probability_denominator: int
    tail_probability: float
    status: str
    method: str
    shape_lower_bound: float
    shape_at_lower_bound: bool


def _values(data: TailInput) -> np.ndarray:
    values = (
        data.to_numpy(dtype=float, na_value=np.nan)
        if isinstance(data, pd.Series)
        else np.asarray(data, dtype=float)
    )
    if values.ndim != 1:
        raise ValueError("tail fitting requires one-dimensional returns")
    return values


def _log_likelihood(scale: float, shape: float, data: np.ndarray) -> float:
    if not np.isfinite(scale) or scale <= 0 or not np.isfinite(shape):
        return -np.inf
    if np.any(~np.isfinite(data)) or np.any(data < 0):
        return -np.inf
    with np.errstate(all="ignore"):
        value = float(np.sum(stats.genpareto.logpdf(data, shape, scale=scale)))
    return value if np.isfinite(value) else -np.inf


def _fit(data: np.ndarray, *, reference: bool = False) -> tuple[float, float] | None:
    if not len(data) or np.any(~np.isfinite(data)) or np.any(data < 0):
        return None
    normalization = 1.0 if reference else float(np.max(data))
    if normalization <= 0:
        return None
    normalized = data / normalization

    def objective(params: np.ndarray) -> float:
        if reference and params[1] < 0:
            return np.inf
        return -_log_likelihood(float(params[0]), float(params[1]), normalized)

    initial = [1.0, 1.0] if reference else [float(np.mean(normalized)), 0.1]
    options = None if reference else {"maxiter": 2000, "xatol": 1e-9, "fatol": 1e-9}
    # Below -1 the density at a finite sample maximum is unbounded as the
    # distribution endpoint approaches that maximum. Do not call that a fitted MLE.
    bounds = None if reference else [(np.finfo(float).tiny, None), (-1.0, None)]
    result = optimize.minimize(
        objective, initial, method="Nelder-Mead", options=options, bounds=bounds
    )
    if not result.success or not np.isfinite(result.fun):
        return None
    scale, shape = float(result.x[0] * normalization), float(result.x[1])
    return (scale, shape) if np.isfinite(_log_likelihood(scale, shape, data)) else None


def _quantile(
    threshold: float,
    scale: float,
    shape: float,
    probability: float,
    total_n: int,
    exceedance_n: int,
) -> float:
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be positive and finite")
    if not np.isfinite(threshold) or threshold < 0 or not np.isfinite(shape):
        raise ValueError("threshold must be nonnegative and shape must be finite")
    if not 0 < probability < 1 or not 0 < exceedance_n <= total_n:
        raise ValueError("probability and sample/exceedance counts are outside their domains")
    conditional = probability * total_n / exceedance_n
    if conditional > 1:
        raise ValueError("probability is above the fitted tail fraction")
    return float(threshold + stats.genpareto.isf(conditional, shape, scale=scale))


def _shortfall(value: float, threshold: float, scale: float, shape: float) -> float:
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be positive and finite")
    if not np.isfinite(shape) or not np.isfinite(threshold) or threshold < 0:
        raise ValueError("threshold must be nonnegative and shape must be finite")
    if np.isnan(value) or value < threshold:
        raise ValueError("tail quantile must be at least the threshold")
    if shape >= 1:
        return np.inf
    if shape < 0 and value > threshold - scale / shape:
        raise ValueError("tail quantile exceeds the finite distribution endpoint")
    return float(value + (scale + shape * (value - threshold)) / (1 - shape))


def estimate_gpd_tail_risk(
    returns: TailInput,
    *,
    tail_probability: float = 0.01,
    threshold: float | None = None,
    method: Literal["peaks_over_threshold", "empyrical_raw_tail"] = "peaks_over_threshold",
) -> GPDTailRiskResult:
    """Fit one supplied portfolio without inferring a calendar or annualization.

    Native default threshold is the nonnegative 90th percentile of all finite
    losses (-returns). Fit strictly positive threshold excesses; probabilities
    use every finite observation, including gains. The named reference method
    fits raw losses at the legacy halving thresholds with a loss-only denominator.
    Native shape is constrained to at least -1 to exclude the unbounded-likelihood
    endpoint singularity. The result reports the bound and boundary fits.
    Missing/nonfinite inputs are excluded and counted. Neither insufficient data
    nor failed fits imply zero risk. Counts are not a statistical confidence claim.
    """
    values = _values(returns)
    if not np.isfinite(tail_probability) or not 0 < tail_probability < 1:
        raise ValueError("tail_probability must be between zero and one")
    if method not in ("peaks_over_threshold", "empyrical_raw_tail"):
        raise ValueError("unknown GPD fitting method")
    if threshold is not None and (not np.isfinite(threshold) or threshold < 0):
        raise ValueError("threshold must be nonnegative and finite")
    if method == "empyrical_raw_tail" and threshold is not None:
        raise ValueError("empyrical_raw_tail uses its declared halving-threshold schedule")
    finite = values[np.isfinite(values)]
    losses = -finite
    positive = losses[losses > 0]
    common = {
        "n_observations": len(values),
        "n_finite": len(finite),
        "n_excluded": len(values) - len(finite),
        "n_losses": len(positive),
        "tail_probability": tail_probability,
        "method": method,
    }
    denominator = len(finite) if method == "peaks_over_threshold" else len(positive)
    selected_threshold = (
        float(threshold)
        if threshold is not None
        else max(0.0, float(np.quantile(losses, 0.9)))
        if len(losses)
        else 0.0
    )

    def result(
        status: str, count: int, scale=np.nan, shape=np.nan, value=np.nan, shortfall=np.nan
    ) -> GPDTailRiskResult:
        return GPDTailRiskResult(
            threshold=selected_threshold,
            scale=scale,
            shape=shape,
            value_at_risk=value,
            expected_shortfall=shortfall,
            n_exceedances=count,
            probability_denominator=denominator,
            status=status,
            shape_lower_bound=-1.0 if method == "peaks_over_threshold" else 0.0,
            shape_at_lower_bound=bool(
                np.isfinite(shape)
                and abs(shape - (-1.0 if method == "peaks_over_threshold" else 0.0)) < 1e-6
            ),
            **common,
        )

    if method == "empyrical_raw_tail":
        selected_threshold = 0.2
        count = 0
        if len(positive) < 3:
            return result("insufficient_exceedances", len(positive))
        while selected_threshold > 1e-9:
            sample = positive[positive >= selected_threshold]
            count = len(sample)
            params = _fit(sample, reference=True)
            if params is not None:
                scale, shape = params
                if shape > 0 and tail_probability <= count / denominator:
                    value = _quantile(
                        selected_threshold, scale, shape, tail_probability, denominator, count
                    )
                    if value > 0:
                        return result(
                            "ok",
                            count,
                            scale,
                            shape,
                            value,
                            _shortfall(value, selected_threshold, scale, shape),
                        )
            if selected_threshold / 2 <= 1e-9:
                break
            selected_threshold /= 2
        return result("fit_failed", count)
    sample = losses[losses > selected_threshold] - selected_threshold
    count = len(sample)
    if count < 3:
        return result("insufficient_exceedances", count)
    if np.ptp(sample) == 0:
        return result("degenerate_tail", count)
    if tail_probability > count / denominator:
        return result("probability_outside_fitted_tail", count)
    params = _fit(sample)
    if params is None:
        return result("fit_failed", count)
    scale, shape = params
    value = _quantile(selected_threshold, scale, shape, tail_probability, denominator, count)
    return result(
        "ok", count, scale, shape, value, _shortfall(value, selected_threshold, scale, shape)
    )
