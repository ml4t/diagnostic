"""Arbitrary supplied-benchmark observations and explicit batting conventions."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ml4t.diagnostic.metrics._paired_returns import paired_returns_1d

from .metrics import ArrayLike


def benchmark_batting_average(
    returns: ArrayLike | pd.Series,
    benchmark_returns: ArrayLike | pd.Series,
    *,
    up_includes_zero: bool = False,
) -> dict[str, float | int | bool]:
    """Count strictly positive active returns on finite paired observations.

    Two Series align on actual labels; arrays remain positional. Overall includes
    zero benchmark observations. Native up-market uses strictly positive returns;
    explicit up_includes_zero=True selects the Empyrical convention. Ties are
    nonwins and empty subsets are undefined. Returns and benchmark are never filled.
    """
    if not isinstance(up_includes_zero, bool):
        raise TypeError("up_includes_zero must be a boolean")
    values, benchmark, _ = paired_returns_1d(returns, benchmark_returns)
    active = values - benchmark
    paired = np.isfinite(active) & np.isfinite(benchmark)
    valid_active, valid_benchmark = active[paired], benchmark[paired]
    up = valid_benchmark >= 0 if up_includes_zero else valid_benchmark > 0
    down = valid_benchmark < 0
    wins = valid_active > 0
    result: dict[str, float | int | bool] = {
        "n_observations": len(values),
        "n_paired": int(paired.sum()),
        "n_excluded": int((~paired).sum()),
        "n_zero_benchmark": int((valid_benchmark == 0).sum()),
        "up_includes_zero": up_includes_zero,
    }
    for name, selected in (("overall", np.ones(len(wins), dtype=bool)), ("up", up), ("down", down)):
        count, won = int(selected.sum()), int(wins[selected].sum())
        result.update(
            {
                name: won / count if count else float("nan"),
                f"n_{name}": count,
                f"n_{name}_wins": won,
            }
        )
    return result
