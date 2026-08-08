"""Strategy and backtest uncertainty helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd
import polars as pl

from ml4t.diagnostic.evaluation.stats import (
    _optimal_block_size,
    _stationary_bootstrap_indices,
    compute_min_trl,
    compute_pbo,
    compute_sharpe_variance,
    deflated_sharpe_ratio,
    whites_reality_check,
)
from ml4t.diagnostic.metrics.ic_inference import _newey_west_lag


@dataclass(frozen=True)
class BacktestUncertaintyResult:
    """Uncertainty estimates for a single strategy return series."""

    sharpe: float
    sortino: float
    annualized_return: float
    volatility: float
    max_drawdown: float
    calmar: float
    sharpe_se: float
    sharpe_ci_lower: float
    sharpe_ci_upper: float
    sortino_ci_lower: float
    sortino_ci_upper: float
    annualized_return_hac_se: float
    annualized_return_ci_lower: float
    annualized_return_ci_upper: float
    max_drawdown_ci_lower: float
    max_drawdown_ci_upper: float
    calmar_ci_lower: float
    calmar_ci_upper: float
    psr_p_value: float
    bootstrap_block_length: int
    bootstrap_samples: int
    n_observations: int

    def to_dict(self) -> dict[str, float | int]:
        """Return a flat mapping for persistence or reporting."""
        return asdict(self)


@dataclass(frozen=True)
class PairedUncertaintyResult:
    """Paired stationary-bootstrap comparison of challenger and baseline."""

    sharpe_diff: float
    sharpe_diff_ci_lower: float
    sharpe_diff_ci_upper: float
    annualized_return_diff: float
    annualized_return_diff_ci_lower: float
    annualized_return_diff_ci_upper: float
    max_drawdown_diff: float
    max_drawdown_diff_ci_lower: float
    max_drawdown_diff_ci_upper: float
    information_ratio: float
    information_ratio_ci_lower: float
    information_ratio_ci_upper: float
    probability_challenger_wins: float
    p_value: float
    bootstrap_block_length: int
    bootstrap_samples: int
    n_observations: int

    def to_dict(self) -> dict[str, float | int]:
        """Return a flat mapping for persistence or reporting."""
        return asdict(self)


@dataclass(frozen=True)
class SelectionAdjustmentResult:
    """Selection-bias diagnostics for a cohort of strategy variants."""

    leader: str
    leader_sharpe: float
    n_variants: int
    dsr: float
    dsr_p_value: float
    expected_max_sharpe: float
    min_trl_periods: float
    leader_min_trl: float
    pbo: float | None = None
    pbo_n_combinations: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a flat mapping for persistence or reporting."""
        return asdict(self)


@dataclass(frozen=True)
class RealityCheckResult:
    """White's Reality Check summary for challengers versus a benchmark."""

    p_value: float
    test_statistic: float
    best_strategy: str
    n_strategies: int

    def to_dict(self) -> dict[str, float | int | str]:
        """Return a flat mapping for persistence or reporting."""
        return asdict(self)


@dataclass(frozen=True)
class _ReturnStats:
    sharpe: float
    sortino: float
    annualized_return: float
    volatility: float
    max_drawdown: float
    calmar: float


def pick_block_length(
    returns: np.ndarray[Any, np.dtype[Any]] | pl.Series | pd.Series | list[float],
    *,
    explicit: int | None = None,
    rebalance_step: int | None = None,
    horizon: int | None = None,
) -> int:
    """Choose stationary-bootstrap block length from explicit policy or data."""
    if explicit is not None and explicit > 0:
        return int(explicit)

    floor = max(int(horizon or 1), 1)
    if rebalance_step is not None and rebalance_step > 0:
        return max(int(rebalance_step), floor)

    arr = _coerce_returns(returns)
    if arr.size < 2:
        return floor
    optimal = int(round(float(_optimal_block_size(arr))))
    return max(optimal, floor, 1)


def compute_backtest_uncertainty(
    daily_returns: np.ndarray[Any, np.dtype[Any]] | pl.Series | pd.Series | pl.DataFrame,
    *,
    periods_per_year: int = 252,
    block_length: int | None = None,
    rebalance_step: int | None = None,
    horizon: int | None = None,
    n_boot: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> BacktestUncertaintyResult:
    """Compute uncertainty estimates for one strategy return series."""
    returns = _coerce_returns(daily_returns)
    if returns.size < 4:
        raise ValueError("compute_backtest_uncertainty() requires at least 4 finite returns")

    block = pick_block_length(
        returns,
        explicit=block_length,
        rebalance_step=rebalance_step,
        horizon=horizon,
    )
    point = _sample_stats(returns, periods_per_year)
    boot = _stationary_bootstrap_metrics(
        returns,
        periods_per_year=periods_per_year,
        block_length=block,
        n_boot=n_boot,
        seed=seed,
    )
    nw_lag = _newey_west_lag(returns.size, block)
    mean_se = _newey_west_mean_se(returns, lag=nw_lag)
    annualized_return_hac_se = mean_se * periods_per_year if np.isfinite(mean_se) else float("nan")

    psr_p_value = float("nan")
    try:
        psr = deflated_sharpe_ratio(returns, periods_per_year=periods_per_year)
        psr_p_value = float(psr.p_value)
    except (ValueError, FloatingPointError, ZeroDivisionError):
        psr_p_value = float("nan")

    sharpe_lo, sharpe_hi = _percentile_ci(boot["sharpe"], alpha)
    sortino_lo, sortino_hi = _percentile_ci(boot["sortino"], alpha)
    ann_lo, ann_hi = _percentile_ci(boot["annualized_return"], alpha)
    mdd_lo, mdd_hi = _percentile_ci(boot["max_drawdown"], alpha)
    calmar_lo, calmar_hi = _percentile_ci(boot["calmar"], alpha)

    return BacktestUncertaintyResult(
        sharpe=point.sharpe,
        sortino=point.sortino,
        annualized_return=point.annualized_return,
        volatility=point.volatility,
        max_drawdown=point.max_drawdown,
        calmar=point.calmar,
        sharpe_se=_sharpe_se(returns, periods_per_year),
        sharpe_ci_lower=sharpe_lo,
        sharpe_ci_upper=sharpe_hi,
        sortino_ci_lower=sortino_lo,
        sortino_ci_upper=sortino_hi,
        annualized_return_hac_se=annualized_return_hac_se,
        annualized_return_ci_lower=ann_lo,
        annualized_return_ci_upper=ann_hi,
        max_drawdown_ci_lower=mdd_lo,
        max_drawdown_ci_upper=mdd_hi,
        calmar_ci_lower=calmar_lo,
        calmar_ci_upper=calmar_hi,
        psr_p_value=psr_p_value,
        bootstrap_block_length=block,
        bootstrap_samples=n_boot,
        n_observations=int(returns.size),
    )


def compute_paired_uncertainty(
    challenger: np.ndarray[Any, np.dtype[Any]] | pl.Series | pd.Series,
    baseline: np.ndarray[Any, np.dtype[Any]] | pl.Series | pd.Series,
    *,
    periods_per_year: int = 252,
    block_length: int | None = None,
    rebalance_step: int | None = None,
    horizon: int | None = None,
    n_boot: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> PairedUncertaintyResult:
    """Compare aligned challenger and baseline returns with paired bootstrap."""
    challenger_arr = _coerce_returns(challenger, strip_leading_zero=False)
    baseline_arr = _coerce_returns(baseline, strip_leading_zero=False)
    if challenger_arr.size != baseline_arr.size:
        raise ValueError("challenger and baseline must have the same aligned length")
    if challenger_arr.size < 4:
        raise ValueError("compute_paired_uncertainty() requires at least 4 paired returns")

    diff = challenger_arr - baseline_arr
    block = pick_block_length(
        diff,
        explicit=block_length,
        rebalance_step=rebalance_step,
        horizon=horizon,
    )
    point_c = _sample_stats(challenger_arr, periods_per_year)
    point_b = _sample_stats(baseline_arr, periods_per_year)
    sharpe_diff = point_c.sharpe - point_b.sharpe
    return_diff = point_c.annualized_return - point_b.annualized_return
    max_drawdown_diff = point_c.max_drawdown - point_b.max_drawdown
    information_ratio = _information_ratio(diff, periods_per_year)

    rng = np.random.default_rng(seed)
    sharpe_diffs = np.empty(n_boot)
    return_diffs = np.empty(n_boot)
    max_drawdown_diffs = np.empty(n_boot)
    information_ratios = np.empty(n_boot)
    wins = 0

    np_state = np.random.get_state()
    np.random.seed(int(rng.integers(0, 2**31 - 1)))
    try:
        for i in range(n_boot):
            idx = _stationary_bootstrap_indices(challenger_arr.size, float(block))
            cs = _sample_stats(challenger_arr[idx], periods_per_year)
            bs = _sample_stats(baseline_arr[idx], periods_per_year)
            sharpe_diffs[i] = cs.sharpe - bs.sharpe
            return_diffs[i] = cs.annualized_return - bs.annualized_return
            max_drawdown_diffs[i] = cs.max_drawdown - bs.max_drawdown
            information_ratios[i] = _information_ratio(
                challenger_arr[idx] - baseline_arr[idx], periods_per_year
            )
            wins += int(cs.sharpe > bs.sharpe)
    finally:
        np.random.set_state(np_state)

    sharpe_lo, sharpe_hi = _percentile_ci(sharpe_diffs, alpha)
    return_lo, return_hi = _percentile_ci(return_diffs, alpha)
    mdd_lo, mdd_hi = _percentile_ci(max_drawdown_diffs, alpha)
    ir_lo, ir_hi = _percentile_ci(information_ratios, alpha)
    centered = sharpe_diffs - np.nanmean(sharpe_diffs)
    p_value = float(np.nanmean(np.abs(centered) >= abs(sharpe_diff)))

    return PairedUncertaintyResult(
        sharpe_diff=sharpe_diff,
        sharpe_diff_ci_lower=sharpe_lo,
        sharpe_diff_ci_upper=sharpe_hi,
        annualized_return_diff=return_diff,
        annualized_return_diff_ci_lower=return_lo,
        annualized_return_diff_ci_upper=return_hi,
        max_drawdown_diff=max_drawdown_diff,
        max_drawdown_diff_ci_lower=mdd_lo,
        max_drawdown_diff_ci_upper=mdd_hi,
        information_ratio=information_ratio,
        information_ratio_ci_lower=ir_lo,
        information_ratio_ci_upper=ir_hi,
        probability_challenger_wins=float(wins) / float(n_boot),
        p_value=p_value,
        bootstrap_block_length=block,
        bootstrap_samples=n_boot,
        n_observations=int(challenger_arr.size),
    )


def compute_selection_adjustment(
    returns_by_variant: dict[str, np.ndarray[Any, np.dtype[Any]] | pl.Series | pd.Series],
    *,
    periods_per_year: int = 252,
    pbo_returns_by_fold: dict[str, list[float] | np.ndarray[Any, np.dtype[Any]]] | None = None,
) -> SelectionAdjustmentResult:
    """Compute selection-bias diagnostics for a strategy cohort."""
    arrays = {
        name: arr
        for name, values in returns_by_variant.items()
        if (arr := _coerce_returns(values)).size >= 4 and float(np.std(arr, ddof=1)) > 1e-10
    }
    if not arrays:
        raise ValueError(
            "returns_by_variant must contain at least one non-degenerate return series"
        )

    names = list(arrays)
    series = [arrays[name] for name in names]
    sharpes = {name: _sample_stats(arrays[name], periods_per_year).sharpe for name in names}
    leader = max(names, key=lambda name: sharpes[name])

    dsr = deflated_sharpe_ratio(series, periods_per_year=periods_per_year)
    mtrl = compute_min_trl(arrays[leader], periods_per_year=periods_per_year)
    pbo = _compute_pbo_from_fold_performance(pbo_returns_by_fold)

    return SelectionAdjustmentResult(
        leader=leader,
        leader_sharpe=float(sharpes[leader]),
        n_variants=len(series),
        dsr=float(dsr.deflated_sharpe),
        dsr_p_value=float(dsr.p_value),
        expected_max_sharpe=float(dsr.expected_max_sharpe),
        min_trl_periods=float(dsr.min_trl),
        leader_min_trl=float(mtrl.min_trl),
        pbo=None if pbo is None else float(pbo.pbo),
        pbo_n_combinations=None if pbo is None else int(pbo.n_combinations),
    )


def compute_reality_check(
    challengers: dict[str, np.ndarray[Any, np.dtype[Any]] | pl.Series | pd.Series],
    benchmark: np.ndarray[Any, np.dtype[Any]] | pl.Series | pd.Series,
    *,
    n_bootstrap: int = 2000,
    block_size: int | None = None,
    seed: int = 0,
) -> RealityCheckResult:
    """Run White's Reality Check for challengers against a benchmark."""
    benchmark_arr = _coerce_returns(benchmark, strip_leading_zero=False)
    names: list[str] = []
    arrays: list[np.ndarray[Any, np.dtype[Any]]] = []
    for name, values in challengers.items():
        arr = _coerce_returns(values, strip_leading_zero=False)
        if arr.size == benchmark_arr.size and float(np.std(arr, ddof=1)) > 1e-10:
            names.append(name)
            arrays.append(arr)
    if not arrays:
        raise ValueError("challengers must contain at least one aligned non-degenerate series")

    matrix = np.column_stack(arrays)
    result = whites_reality_check(
        returns_benchmark=benchmark_arr,
        returns_strategies=matrix,
        bootstrap_samples=n_bootstrap,
        block_size=block_size,
        random_state=seed,
    )
    best_idx = int(result["best_strategy_idx"])
    return RealityCheckResult(
        p_value=float(result["p_value"]),
        test_statistic=float(result["test_statistic"]),
        best_strategy=names[best_idx],
        n_strategies=len(names),
    )


def _sample_stats(returns: np.ndarray[Any, np.dtype[Any]], periods_per_year: int) -> _ReturnStats:
    if returns.size < 2:
        return _ReturnStats(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    mean = float(np.mean(returns))
    std = float(np.std(returns, ddof=1))
    sharpe = mean / std * np.sqrt(periods_per_year) if std > 0 else 0.0
    downside = np.minimum(returns, 0.0)
    downside_std = float(np.sqrt(np.mean(downside**2)))
    sortino = mean / downside_std * np.sqrt(periods_per_year) if downside_std > 0 else 0.0
    cumulative = np.cumprod(1.0 + returns)
    total_return = float(cumulative[-1] - 1.0)
    years = returns.size / periods_per_year
    base = 1.0 + total_return
    if years <= 0:
        annualized_return = 0.0
    elif base <= 0.0:
        annualized_return = -1.0
    else:
        annualized_return = float(base ** (1.0 / years) - 1.0)
    volatility = std * np.sqrt(periods_per_year)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / np.where(running_max > 0, running_max, np.nan)
    max_drawdown = float(np.nanmin(drawdown)) if np.any(np.isfinite(drawdown)) else 0.0
    calmar = annualized_return / abs(max_drawdown) if max_drawdown < 0 else 0.0
    return _ReturnStats(sharpe, sortino, annualized_return, volatility, max_drawdown, calmar)


def _stationary_bootstrap_metrics(
    returns: np.ndarray[Any, np.dtype[Any]],
    *,
    periods_per_year: int,
    block_length: int,
    n_boot: int,
    seed: int,
) -> dict[str, np.ndarray[Any, np.dtype[Any]]]:
    rng = np.random.default_rng(seed)
    metrics = {
        "sharpe": np.empty(n_boot),
        "sortino": np.empty(n_boot),
        "annualized_return": np.empty(n_boot),
        "max_drawdown": np.empty(n_boot),
        "calmar": np.empty(n_boot),
    }

    np_state = np.random.get_state()
    np.random.seed(int(rng.integers(0, 2**31 - 1)))
    try:
        for i in range(n_boot):
            sample = returns[_stationary_bootstrap_indices(returns.size, float(block_length))]
            stats = _sample_stats(sample, periods_per_year)
            metrics["sharpe"][i] = stats.sharpe
            metrics["sortino"][i] = stats.sortino
            metrics["annualized_return"][i] = stats.annualized_return
            metrics["max_drawdown"][i] = stats.max_drawdown
            metrics["calmar"][i] = stats.calmar
    finally:
        np.random.set_state(np_state)

    return metrics


def _newey_west_mean_se(returns: np.ndarray[Any, np.dtype[Any]], lag: int) -> float:
    n = returns.size
    if n < 3:
        return float("nan")
    centered = returns - np.mean(returns)
    variance = float(np.dot(centered, centered) / n)
    for h in range(1, min(lag, n - 1) + 1):
        autocov = float(np.dot(centered[:-h], centered[h:]) / n)
        weight = 1.0 - h / (lag + 1.0)
        variance += 2.0 * weight * autocov
    return float(np.sqrt(max(variance, 0.0) / n))


def _sharpe_se(returns: np.ndarray[Any, np.dtype[Any]], periods_per_year: int) -> float:
    if returns.size < 4:
        return float("nan")
    mean = float(np.mean(returns))
    std = float(np.std(returns, ddof=1))
    if std <= 0:
        return float("nan")
    native_sharpe = mean / std
    centered = returns - mean
    second = float(np.mean(centered**2))
    if second <= 0:
        return float("nan")
    skew = float(np.mean(centered**3) / second**1.5)
    kurtosis = float(np.mean(centered**4) / second**2)
    autocorrelation = float(np.corrcoef(returns[:-1], returns[1:])[0, 1])
    if not np.isfinite(autocorrelation) or abs(autocorrelation) >= 0.999:
        autocorrelation = 0.0
    variance = compute_sharpe_variance(
        sharpe=native_sharpe,
        n_samples=int(returns.size),
        skewness=skew,
        kurtosis=kurtosis,
        autocorrelation=autocorrelation,
        n_trials=1,
    )
    return float(np.sqrt(variance) * np.sqrt(periods_per_year)) if variance > 0 else float("nan")


def _percentile_ci(
    values: np.ndarray[Any, np.dtype[Any]],
    alpha: float,
) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan"), float("nan")
    return (
        float(np.percentile(finite, 100 * alpha / 2)),
        float(np.percentile(finite, 100 * (1 - alpha / 2))),
    )


def _information_ratio(returns: np.ndarray[Any, np.dtype[Any]], periods_per_year: int) -> float:
    std = float(np.std(returns, ddof=1))
    if std <= 1e-10:
        return float("nan")
    return float(np.mean(returns) / std * np.sqrt(periods_per_year))


def _coerce_returns(
    values: np.ndarray[Any, np.dtype[Any]] | pl.Series | pd.Series | pl.DataFrame | list[float],
    *,
    strip_leading_zero: bool = True,
) -> np.ndarray[Any, np.dtype[Any]]:
    if isinstance(values, pl.DataFrame):
        col = next(
            (c for c in ("daily_return", "ret", "return", "value") if c in values.columns),
            values.columns[-1],
        )
        arr = values[col].to_numpy()
    elif isinstance(values, pl.Series | pd.Series):
        arr = values.to_numpy()
    else:
        arr = np.asarray(values).flatten()
    arr = np.asarray(arr, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if strip_leading_zero and arr.size:
        nonzero = np.flatnonzero(arr != 0.0)
        if nonzero.size:
            arr = arr[nonzero[0] :]
    return arr


def _compute_pbo_from_fold_performance(
    fold_performance_by_variant: dict[str, list[float] | np.ndarray[Any, np.dtype[Any]]] | None,
):
    if not fold_performance_by_variant:
        return None

    names = list(fold_performance_by_variant)
    matrix = np.column_stack(
        [np.asarray(fold_performance_by_variant[name], dtype=np.float64) for name in names]
    )
    if matrix.ndim != 2 or matrix.shape[0] < 2 or matrix.shape[1] < 2:
        return None
    if matrix.shape[0] % 2 != 0:
        matrix = matrix[:-1]
    half = matrix.shape[0] // 2
    is_rows: list[np.ndarray[Any, np.dtype[Any]]] = []
    oos_rows: list[np.ndarray[Any, np.dtype[Any]]] = []
    all_indices = set(range(matrix.shape[0]))
    for combo in combinations(range(matrix.shape[0]), half):
        is_idx = np.array(combo)
        oos_idx = np.array(sorted(all_indices.difference(combo)))
        is_rows.append(np.nanmean(matrix[is_idx], axis=0))
        oos_rows.append(np.nanmean(matrix[oos_idx], axis=0))
    if not is_rows:
        return None
    return compute_pbo(np.vstack(is_rows), np.vstack(oos_rows))


__all__ = [
    "BacktestUncertaintyResult",
    "PairedUncertaintyResult",
    "RealityCheckResult",
    "SelectionAdjustmentResult",
    "pick_block_length",
    "compute_backtest_uncertainty",
    "compute_paired_uncertainty",
    "compute_selection_adjustment",
    "compute_reality_check",
]
