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

from ml4t.diagnostic.compat._returns import (
    MetricOutput,
    ReturnInput,
    adjust_rate,
    as_array,
    reduce_columns,
)
from ml4t.diagnostic.evaluation.portfolio_analysis import metrics as _native
from ml4t.diagnostic.metrics.risk_adjusted import (
    _sortino_from_excess,
    periodic_downside_risk,
    periodic_sharpe_ratio,
)

DAILY, WEEKLY, MONTHLY, QUARTERLY, YEARLY = "daily", "weekly", "monthly", "quarterly", "yearly"
APPROX_BDAYS_PER_MONTH, APPROX_BDAYS_PER_YEAR = 21, 252
MONTHS_PER_YEAR, WEEKS_PER_YEAR, QTRS_PER_YEAR = 12, 52, 4
ANNUALIZATION_FACTORS = {DAILY: 252, WEEKLY: 52, MONTHLY: 12, QUARTERLY: 4, YEARLY: 1}


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
        out=out,
        labeled=True,
        min_length=2,
    )


__all__ = [
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
