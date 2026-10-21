"""Selected Empyrical metric signatures using native Diagnostic calculations.

Risk-free/MAR arguments are periodic. Native annual-rate APIs are unchanged.
Missingness is metric-specific: CAGR counts missing rows as elapsed observations,
while deviations and means omit undefined adjusted observations per column.
Sortino preserves input column names, including the maintained reference's repair
of the labels lost by published Empyrical Reloaded 0.5.12. Calmar remains 1D;
apply it per column for a portfolio frame. This is not a full replacement module.
"""

from numbers import Real

import numpy as np
import pandas as pd
from scipy import stats as _stats

from ml4t.diagnostic.compat._benchmark import (
    alpha,
    alpha_aligned,
    alpha_beta,
    alpha_beta_aligned,
    batting_average,
    beta,
    beta_aligned,
    down_alpha_beta,
    excess_sharpe,
    up_alpha_beta,
)
from ml4t.diagnostic.compat._capture import capture, down_capture, up_capture, up_down_capture
from ml4t.diagnostic.compat._fragility import (
    beta_fragility_heuristic,
    beta_fragility_heuristic_aligned,
)
from ml4t.diagnostic.compat._holdings_attribution import compute_exposures, perf_attrib
from ml4t.diagnostic.compat._returns import (
    MetricOutput,
    ReturnInput,
    adjust_rate,
    as_array,
    output_array,
    reduce_columns,
)
from ml4t.diagnostic.compat._rolling import (
    down,
    roll,
    roll_alpha,
    roll_alpha_aligned,
    roll_alpha_beta,
    roll_alpha_beta_aligned,
    roll_annual_volatility,
    roll_beta,
    roll_beta_aligned,
    roll_cagr,
    roll_down_capture,
    roll_downside_risk,
    roll_downsize_risk,
    roll_excess_sharpe,
    roll_max_drawdown,
    roll_sharpe_ratio,
    roll_sortino_ratio,
    roll_up_capture,
    roll_up_down_capture,
    rolling_window,
    up,
)
from ml4t.diagnostic.evaluation.portfolio_analysis import metrics as _native
from ml4t.diagnostic.metrics.risk_adjusted import (
    _drawdown_from_wealth,
    _sortino_from_excess,
    periodic_downside_risk,
    periodic_sharpe_ratio,
)

DAILY, WEEKLY, MONTHLY, QUARTERLY, YEARLY = "daily", "weekly", "monthly", "quarterly", "yearly"
APPROX_BDAYS_PER_MONTH, APPROX_BDAYS_PER_YEAR = 21, 252
MONTHS_PER_YEAR, WEEKS_PER_YEAR, QTRS_PER_YEAR = 12, 52, 4
ANNUALIZATION_FACTORS = {DAILY: 252, WEEKLY: 52, MONTHLY: 12, QUARTERLY: 4, YEARLY: 1}


def simple_returns(prices: ReturnInput) -> ReturnInput:
    """Preserve pandas pct_change defaults; ndarray gaps remain undefined.

    Pandas 2 pads missing prices by default, whereas pandas 3 propagates them.
    Integer ndarray differences are computed before floating conversion.
    """
    values = as_array(prices)
    if isinstance(prices, (pd.Series, pd.DataFrame)):
        return prices.pct_change().iloc[1:]
    differences = np.diff(
        prices.astype(object) if np.issubdtype(prices.dtype, np.integer) else values, axis=0
    ).astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return differences / values[:-1]


def _restore_path(returns: ReturnInput, values: np.ndarray) -> ReturnInput:
    if isinstance(returns, pd.Series):
        return pd.Series(values, index=returns.index, copy=False)
    if isinstance(returns, pd.DataFrame):
        return pd.DataFrame(values, index=returns.index, columns=returns.columns, copy=False)
    return values


def cum_returns(
    returns: ReturnInput, starting_value: float = 0, out: np.ndarray | None = None
) -> ReturnInput:
    """Adapt native NaN-neutral compounding along each supplied return column."""
    values = as_array(returns)
    if len(values) == 0:
        return returns.copy()
    output = output_array(returns, values.shape, out)
    if values.ndim == 1:
        output[...] = _native._safe_cumprod(1 + values)
    else:
        for index in range(values.shape[1]):
            output[:, index] = _native._safe_cumprod(1 + values[:, index])
    if starting_value == 0:
        output -= 1
    else:
        output *= starting_value
    return _restore_path(returns, output) if out is None else output


def cum_returns_final(returns: ReturnInput, starting_value: float = 0) -> MetricOutput:
    """Return total gain or terminal wealth without annualizing elapsed rows."""
    result = reduce_columns(
        returns,
        lambda column, _: (
            _native._safe_prod(1 + column) - 1
            if starting_value == 0
            else _native._safe_prod(1 + column) * starting_value
        ),
        labeled=True,
        empty_scalar=True,
    )
    return np.float64(result) if isinstance(result, float) and len(returns) else result


def aggregate_returns(
    returns: pd.Series | pd.DataFrame, convert_to: str
) -> pd.Series | pd.DataFrame:
    """Compound observed civil groups, using ISO year/week and the supplied timezone."""
    if not isinstance(returns, pd.Series | pd.DataFrame) or not isinstance(
        returns.index, pd.DatetimeIndex
    ):
        raise TypeError("aggregate_returns requires pandas returns with a DatetimeIndex")
    groups = {
        WEEKLY: [lambda label: label.isocalendar()[0], lambda label: label.isocalendar()[1]],
        MONTHLY: [lambda label: label.year, lambda label: label.month],
        QUARTERLY: [lambda label: label.year, lambda label: label.quarter],
        YEARLY: [lambda label: label.year],
    }
    if convert_to not in groups:
        raise ValueError("convert_to must be weekly, monthly, quarterly or yearly")
    return returns.groupby(groups[convert_to]).apply(cum_returns_final)


def drawdown_series(returns: ReturnInput, out: np.ndarray | None = None) -> ReturnInput | float:
    """Use native initial-capital drawdowns with one output per supplied row.

    The optional buffer includes an extra initial zero anchor. Empty 1D inputs
    return NaN; empty 2D inputs return an unlabeled one-row NaN array, as in 0.7.0.
    """
    values = as_array(returns)
    output = output_array(returns, (len(values) + 1,) + values.shape[1:], out)
    if len(values) == 0:
        output[...] = np.nan
        return output.item() if values.ndim == 1 else output
    output[0] = 0
    if values.ndim == 1:
        output[1:] = _drawdown_from_wealth(_native._safe_cumprod(1 + values))
    else:
        for index in range(values.shape[1]):
            output[1:, index] = _drawdown_from_wealth(_native._safe_cumprod(1 + values[:, index]))
    return _restore_path(returns, output[1:])


def max_drawdown(returns: ReturnInput, out: np.ndarray | None = None) -> MetricOutput:
    """Adapt native negative peak-to-trough declines, including initial loss."""
    return reduce_columns(
        returns,
        lambda column, _: _native.max_drawdown(column),
        out=out,
        labeled=out is None,
        min_length=1,
    )


def annualization_factor(period: str, annualization: float | None) -> float:
    """Use explicit positive finite scale before resolving a named period."""
    if annualization is None:
        if period not in ANNUALIZATION_FACTORS:
            raise ValueError(
                f"unknown period {period!r}; choose from {tuple(ANNUALIZATION_FACTORS)}"
            )
        return float(ANNUALIZATION_FACTORS[period])
    if (
        isinstance(annualization, bool)
        or not isinstance(annualization, Real)
        or not np.isfinite(annualization)
        or annualization <= 0
    ):
        raise ValueError("annualization must be a positive finite number")
    return float(annualization)


def annual_return(
    returns: ReturnInput, period: str = DAILY, annualization: float | None = None
) -> MetricOutput:
    """Compound each column, counting every supplied row in the elapsed scale."""
    scale = annualization_factor(period, annualization)
    return reduce_columns(
        returns,
        lambda column, _: _native.annual_return(column, scale),
        labeled=True,
        empty_scalar=True,
    )


def cagr(
    returns: ReturnInput, period: str = DAILY, annualization: float | None = None
) -> MetricOutput:
    """Alias for annual_return, not a rolling cumulative-return relabeling."""
    return annual_return(returns, period, annualization)


def annual_volatility(
    returns: ReturnInput,
    period: str = DAILY,
    alpha: float = 2.0,
    annualization: float | None = None,
    out: np.ndarray | None = None,
) -> MetricOutput:
    """Scale native sample deviation by the explicit Levy exponent."""
    scale = annualization_factor(period, annualization)
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("alpha must be positive and finite")
    return reduce_columns(
        returns,
        lambda column, _: _native.annual_volatility(column, 1) * scale ** (1 / alpha),
        out=out,
        min_length=2,
    )


def calmar_ratio(
    returns: ReturnInput, period: str = DAILY, annualization: float | None = None
) -> float:
    """Calculate the one-series Calmar workflow; use DataFrame.apply for columns."""
    if as_array(returns).ndim != 1:
        raise ValueError("calmar_ratio requires one-dimensional returns; apply per column")
    value = _native.calmar_ratio(as_array(returns), annualization_factor(period, annualization))
    return float("nan") if np.isinf(value) else float(value)


def sharpe_ratio(
    returns: ReturnInput,
    risk_free: float | ReturnInput = 0,
    period: str = DAILY,
    annualization: float | None = None,
    out: np.ndarray | None = None,
) -> MetricOutput:
    """Use periodic risk-free returns and columnwise native Sharpe calculation."""
    scale = annualization_factor(period, annualization)
    return reduce_columns(
        returns,
        lambda column, _: periodic_sharpe_ratio(column, annualization_factor=scale),
        adjusted=adjust_rate(returns, risk_free),
        additional_inputs=(risk_free,),
        out=out,
        min_length=2,
    )


def downside_risk(
    returns: ReturnInput,
    required_return: float | ReturnInput = 0,
    period: str = DAILY,
    annualization: float | None = None,
    out: np.ndarray | None = None,
) -> MetricOutput:
    """Compute shared downside deviation with a periodic minimum acceptable return."""
    scale = annualization_factor(period, annualization)
    return reduce_columns(
        returns,
        lambda column, _: periodic_downside_risk(column, annualization_factor=scale),
        adjusted=adjust_rate(returns, required_return),
        additional_inputs=(required_return,),
        out=out,
        labeled=True,
        min_length=1,
    )


def sortino_ratio(
    returns: ReturnInput,
    required_return: float | ReturnInput = 0,
    period: str = DAILY,
    annualization: float | None = None,
    out: np.ndarray | None = None,
    _downside_risk: float | np.ndarray | None = None,
) -> MetricOutput:
    """Use the shared Sortino kernel with legacy raw-length and periodic-rate rules."""
    scale = annualization_factor(period, annualization)
    supplied = (
        None
        if _downside_risk is None
        else np.broadcast_to(_downside_risk, as_array(returns).shape[1:])
    )

    def calculate(column: np.ndarray, index: int) -> float:
        deviation = (
            None
            if supplied is None
            else float(supplied[()] if supplied.ndim == 0 else supplied[index]) / np.sqrt(scale)
        )
        clean = column[~np.isnan(column)]
        return _sortino_from_excess(clean, scale, deviation)

    return reduce_columns(
        returns,
        calculate,
        adjusted=adjust_rate(returns, required_return),
        additional_inputs=(required_return, _downside_risk),
        out=out,
        labeled=True,
        min_length=2,
    )


def omega_ratio(
    returns: ReturnInput,
    risk_free: float | ReturnInput = 0.0,
    required_return: float = 0.0,
    annualization: float = APPROX_BDAYS_PER_YEAR,
) -> float:
    """Use an annual hurdle, periodic risk-free returns and undefined no-loss Omega."""
    values = as_array(returns)
    scale = annualization_factor(DAILY, annualization)
    if len(values) < 2 or (scale != 1 and required_return <= -1):
        return float("nan")
    hurdle = required_return if scale == 1 else (1 + required_return) ** (1 / scale) - 1
    return float(
        _native.omega_ratio(as_array(adjust_rate(returns, risk_free)), hurdle, no_loss="nan")
    )


def tail_ratio(returns: ReturnInput) -> float:
    """Pool the actual legacy shape, using magnitudes of both empirical tails."""
    values = as_array(returns)
    if not np.any(~np.isnan(values)):
        return float("nan")
    return _native.tail_ratio(values, absolute=True)


def stability_of_timeseries(returns: ReturnInput) -> float:
    """Fit cumulative log returns on observed rows, not compounded wealth."""
    return _native.stability_of_timeseries(as_array(returns), method="log")


def value_at_risk(returns: ReturnInput, cutoff: float = 0.05) -> float:
    """Return an empirical lower-return quantile, propagating NaNs as in Empyrical."""
    values = as_array(returns)
    if values.size == 0:
        return float("nan")
    return _native.value_at_risk(values, 1 - cutoff, nan_policy="propagate")


def conditional_value_at_risk(returns: ReturnInput, cutoff: float = 0.05) -> float:
    """Omit NaNs before choosing the fixed-count worst-return tail, as in 0.7.0."""
    return _native._fixed_count_tail(as_array(returns), cutoff)


SIMPLE_STAT_FUNCS = [
    cum_returns_final,
    annual_return,
    annual_volatility,
    sharpe_ratio,
    calmar_ratio,
    stability_of_timeseries,
    max_drawdown,
    omega_ratio,
    sortino_ratio,
    _stats.skew,
    _stats.kurtosis,
    tail_ratio,
    cagr,
    value_at_risk,
    conditional_value_at_risk,
]


__all__ = [
    "compute_exposures",
    "perf_attrib",
    "beta_fragility_heuristic",
    "beta_fragility_heuristic_aligned",
    "SIMPLE_STAT_FUNCS",
    "down",
    "roll",
    "roll_alpha",
    "roll_alpha_aligned",
    "roll_alpha_beta",
    "roll_alpha_beta_aligned",
    "roll_annual_volatility",
    "roll_beta",
    "roll_beta_aligned",
    "roll_cagr",
    "roll_down_capture",
    "roll_downside_risk",
    "roll_downsize_risk",
    "roll_excess_sharpe",
    "roll_max_drawdown",
    "roll_sharpe_ratio",
    "roll_sortino_ratio",
    "roll_up_capture",
    "roll_up_down_capture",
    "rolling_window",
    "up",
    "aggregate_returns",
    "capture",
    "up_capture",
    "down_capture",
    "up_down_capture",
    "omega_ratio",
    "tail_ratio",
    "stability_of_timeseries",
    "value_at_risk",
    "conditional_value_at_risk",
    "alpha",
    "alpha_aligned",
    "alpha_beta",
    "alpha_beta_aligned",
    "beta",
    "beta_aligned",
    "excess_sharpe",
    "up_alpha_beta",
    "down_alpha_beta",
    "batting_average",
    "simple_returns",
    "cum_returns",
    "cum_returns_final",
    "drawdown_series",
    "max_drawdown",
    "annualization_factor",
    "annual_return",
    "cagr",
    "annual_volatility",
    "calmar_ratio",
    "sharpe_ratio",
    "sortino_ratio",
    "downside_risk",
    "DAILY",
    "WEEKLY",
    "MONTHLY",
    "QUARTERLY",
    "YEARLY",
    "ANNUALIZATION_FACTORS",
    "APPROX_BDAYS_PER_MONTH",
    "APPROX_BDAYS_PER_YEAR",
    "MONTHS_PER_YEAR",
    "WEEKS_PER_YEAR",
    "QTRS_PER_YEAR",
]
