"""Familiar geometric capture over the shared native annual-return calculation."""

from typing import Any

import numpy as np

from ._returns import MetricOutput, ReturnInput, select_benchmark_sign


def capture(
    returns: ReturnInput, factor_returns: ReturnInput, period: str = "daily"
) -> MetricOutput:
    """Divide annualized return gains, not native terminal-wealth capture ratios."""
    from .empyrical import annual_return

    strategy = annual_return(returns, period=period)
    benchmark = annual_return(factor_returns, period=period)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.divide(strategy, benchmark)


def up_capture(returns: ReturnInput, factor_returns: ReturnInput, **kwargs: Any) -> MetricOutput:
    """Use present pairs on strictly positive benchmark observations."""
    return capture(*select_benchmark_sign(returns, factor_returns, positive=True), **kwargs)


def down_capture(returns: ReturnInput, factor_returns: ReturnInput, **kwargs: Any) -> MetricOutput:
    """Use present pairs on strictly negative benchmark observations."""
    return capture(*select_benchmark_sign(returns, factor_returns, positive=False), **kwargs)


def up_down_capture(
    returns: ReturnInput, factor_returns: ReturnInput, **kwargs: Any
) -> MetricOutput:
    """Return the scalar/column up-to-down ratio, not the native pair of components."""
    up = up_capture(returns, factor_returns, **kwargs)
    down = down_capture(returns, factor_returns, **kwargs)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.divide(up, down)
