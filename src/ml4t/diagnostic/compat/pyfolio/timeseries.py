"""Pyfolio dated-event and IID cone conventions over existing native kernels.

The familiar IID stream uses RandomState, unlike the native Generator default.
Missing sampled returns are neutral factors in the existing product kernel.
Zero starting_value denotes net cumulative return, not a zero-capital account.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping, Sequence
from datetime import date, datetime

import numpy as np
import pandas as pd

from ml4t.diagnostic.evaluation.portfolio_analysis.regimes import _PERIOD_DATES
from ml4t.diagnostic.evaluation.portfolio_analysis.wealth_cones import (
    _sample_indices,
    _wealth_paths,
)


def simulate_paths(
    is_returns: pd.Series,
    num_days: int,
    starting_value: float = 1,
    num_samples: int = 1000,
    random_seed: int | None = None,
) -> np.ndarray:
    """Return raw IID draws with the familiar local seeded stream and sample shape.

    num_days counts output observations, not exchange sessions. starting_value
    has no role in raw return sampling; summarize_paths applies that convention.
    """
    if not isinstance(is_returns, pd.Series):
        raise TypeError("is_returns must be a one-dimensional pandas Series")
    indices = _sample_indices(
        len(is_returns),
        num_days,
        num_samples,
        method="iid",
        seed=random_seed,
        block_size=None,
        legacy_rng=True,
    )
    return is_returns.to_numpy(dtype=float)[indices]


def summarize_paths(
    samples: np.ndarray,
    cone_std: float | Sequence[float] = (1.0, 1.5, 2.0),
    starting_value: float = 1.0,
) -> pd.DataFrame:
    """Summarize path wealth with mean plus/minus population standard deviation.

    Numeric float columns retain input deviation order and positive/negative
    labels. These bounds are not quantiles or confidence intervals for a mean.
    Native inputs remain finite-only; this explicit legacy profile allows NaNs.
    """
    values = np.asarray(samples, dtype=float)
    if values.ndim != 2:
        raise ValueError("samples must be a two-dimensional sample-by-step array")
    wealth = _wealth_paths(values, 1.0 if starting_value == 0 else starting_value)
    if starting_value == 0:
        wealth = wealth - 1
    means, stds = wealth.mean(axis=0), wealth.std(axis=0, ddof=0)
    levels = [cone_std] if isinstance(cone_std, int | float | np.floating) else cone_std
    result = pd.DataFrame(columns=pd.Index([], dtype="float64"))
    for level in levels:
        result[float(level)] = means + stds * level
        result[float(-level)] = means - stds * level
    result.attrs.update(
        {
            "reference_profile": "Pyfolio 0.10.0",
            "definition": "mean +/- population std, not quantiles",
            "missing_returns": "neutral compounded factor",
            "starting_value_zero": "net cumulative return",
            "horizon_unit": "observations",
        }
    )
    return result


def forecast_cone_bootstrap(
    is_returns: pd.Series,
    num_days: int,
    cone_std: float | Sequence[float] = (1.0, 1.5, 2.0),
    starting_value: float = 1,
    num_samples: int = 1000,
    random_seed: int | None = None,
) -> pd.DataFrame:
    """Compose the familiar raw IID draws and explicitly defined cone bounds."""
    return summarize_paths(
        simulate_paths(is_returns, num_days, starting_value, num_samples, random_seed),
        cone_std,
        starting_value,
    )


def extract_interesting_date_ranges(
    returns: pd.Series,
    periods: Mapping[str, tuple[date | datetime | str, date | datetime | str]] | None = None,
) -> OrderedDict[str, pd.Series]:
    """Select familiar pandas label slices, retaining actual missing observations.

    Default historical endpoints are UTC instants, as in Pyfolio. Covid's open
    end is evaluated at call time rather than fixed at module import. Explicit
    periods make the selection reproducible. Unlike Pyfolio's blanket exception
    suppression, invalid boundaries raise instead of silently hiding a period.
    Native compute_stress additionally retains uncovered periods and coverage.
    """
    if not isinstance(returns, pd.Series) or not isinstance(returns.index, pd.DatetimeIndex):
        raise TypeError("returns must be a pandas Series with a DatetimeIndex")
    windows = periods
    if windows is None:
        windows = {
            name: (
                pd.Timestamp(start, tz="UTC"),
                pd.Timestamp.now(tz="UTC") if end is None else pd.Timestamp(end, tz="UTC"),
            )
            for name, (start, end) in _PERIOD_DATES.items()
        }
    result = OrderedDict()
    source = returns.copy(deep=True)
    source.index = source.index.map(pd.Timestamp)
    for name, (start, end) in windows.items():
        sample = source.loc[start:end].copy(deep=True)
        if not sample.empty:
            result[name] = sample
    return result
