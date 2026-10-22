"""Familiar scalar fragility signatures over the native interpolation kernel."""

from ml4t.diagnostic.metrics.fragility import _beta_fragility

from ._benchmark import _aligned_pair
from ._returns import ReturnInput, as_array


def beta_fragility_heuristic(returns: ReturnInput, factor_returns: ReturnInput) -> float:
    """Align labeled pairs and retain actual reference missingness/tie conventions."""
    if len(returns) < 3 or len(factor_returns) < 3:
        return float("nan")
    aligned = _aligned_pair(returns, factor_returns)
    return _beta_fragility(*aligned, reference=True).fragility


def beta_fragility_heuristic_aligned(returns: ReturnInput, factor_returns: ReturnInput) -> float:
    """Retain the reference's actual Series-label concatenation despite its name."""
    if as_array(returns).ndim != 1 or as_array(factor_returns).ndim != 1:
        raise ValueError("aligned fragility requires one-dimensional returns")
    return beta_fragility_heuristic(returns, factor_returns)
