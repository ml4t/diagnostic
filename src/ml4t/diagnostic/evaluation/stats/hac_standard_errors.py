"""Robust standard errors for Information Coefficient estimation.

This module provides standard error estimation for rank correlation (IC)
with proper handling of temporal dependence through stationary bootstrap.

References
----------
.. [1] Politis, D.N. & Romano, J.P. (1994). "The Stationary Bootstrap."
       Journal of the American Statistical Association 89:1303-1313.

.. [2] Patton, A., Politis, D.N. & White, H. (2009). "Correction to
       Automatic Block-Length Selection for the Dependent Bootstrap."
       Econometric Reviews 28:372-375.
"""

from typing import TYPE_CHECKING, Any, Union

import numpy as np
import pandas as pd
import polars as pl

from .bootstrap import stationary_bootstrap_ic

if TYPE_CHECKING:
    from numpy.typing import NDArray


def robust_ic(
    predictions: Union[pl.Series, pd.Series, "NDArray[Any]"],
    returns: Union[pl.Series, pd.Series, "NDArray[Any]"],
    n_samples: int = 1000,
    return_details: bool = False,
) -> dict[str, float] | float:
    """Calculate Information Coefficient with robust standard errors.

    Uses stationary bootstrap [1]_ to compute standard errors that properly
    account for temporal dependence in time series data.

    The stationary bootstrap is the correct method because:
    1. Preserves temporal dependence structure
    2. No asymptotic approximations required
    3. Theoretically valid for rank correlation (Spearman IC)

    Parameters
    ----------
    predictions : Union[pl.Series, pd.Series, NDArray]
        Model predictions or scores
    returns : Union[pl.Series, pd.Series, NDArray]
        Forward returns corresponding to predictions
    n_samples : int, default 1000
        Number of bootstrap samples
    return_details : bool, default False
        Whether to return detailed statistics

    Returns
    -------
    Union[dict, float]
        If return_details=False: t-statistic (IC / bootstrap_std)
        If return_details=True: dict with 'ic', 'bootstrap_std', 't_stat',
            'p_value', 'ci_lower', 'ci_upper'

    Examples
    --------
    >>> predictions = np.random.randn(252)
    >>> returns = 0.1 * predictions + np.random.randn(252) * 0.5
    >>> result = robust_ic(predictions, returns, return_details=True)
    >>> print(f"IC: {result['ic']:.3f}, t-stat: {result['t_stat']:.3f}")

    References
    ----------
    .. [1] Politis, D.N. & Romano, J.P. (1994). "The Stationary Bootstrap."
           Journal of the American Statistical Association 89:1303-1313.
    """
    bootstrap_result = stationary_bootstrap_ic(
        predictions, returns, n_samples=n_samples, return_details=True
    )
    assert isinstance(bootstrap_result, dict)

    if not return_details:
        if bootstrap_result["bootstrap_std"] > 0:
            return bootstrap_result["ic"] / bootstrap_result["bootstrap_std"]
        return np.nan

    # Compute t-statistic
    t_stat = (
        bootstrap_result["ic"] / bootstrap_result["bootstrap_std"]
        if bootstrap_result["bootstrap_std"] > 0
        else np.nan
    )

    return {
        "ic": bootstrap_result["ic"],
        "bootstrap_std": bootstrap_result["bootstrap_std"],
        "t_stat": t_stat,
        "p_value": bootstrap_result.get("p_value", np.nan),
        "ci_lower": bootstrap_result.get("ci_lower", np.nan),
        "ci_upper": bootstrap_result.get("ci_upper", np.nan),
    }


# Keep old name as alias for now
hac_adjusted_ic = robust_ic


__all__ = [
    "robust_ic",
    "hac_adjusted_ic",  # Alias
]
