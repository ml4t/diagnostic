"""Three-point benchmark fragility with explicit paired samples and tie ordering."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ._paired_returns import PairedInput, paired_returns_1d


@dataclass(frozen=True)
class BetaFragilityResult:
    """Endpoint interpolation gap in periodic return units, not a fitted beta or p-value."""

    fragility: float
    n_observations: int
    n_pairs: int
    n_excluded: int
    status: str
    method: str
    selected_ordinals: tuple[int, int, int] | None = None
    selected_keys: tuple[object, object, object] | None = None
    benchmark_points: tuple[float, float, float] | None = None
    strategy_points: tuple[float, float, float] | None = None
    endpoint_weights: tuple[float, float] | None = None
    constant_benchmark: bool = False


def _beta_fragility(
    returns: PairedInput, benchmark: PairedInput, *, reference: bool = False
) -> BetaFragilityResult:
    values, factor, keys = paired_returns_1d(returns, benchmark)
    paired = (
        ~(np.isnan(values) | np.isnan(factor))
        if reference
        else np.isfinite(values) & np.isfinite(factor)
    )
    ordinals = np.flatnonzero(paired)
    count = len(ordinals)
    method = "present_pairs_reference_ties" if reference else "finite_pairs_input_order"
    common = {
        "n_observations": len(values),
        "n_pairs": count,
        "n_excluded": len(values) - count,
        "method": method,
    }
    if count < 3:
        return BetaFragilityResult(float("nan"), status="insufficient_pairs", **common)
    strategy, market = values[paired], factor[paired]
    order = (
        pd.Series(market).sort_values(kind="quicksort").index.to_numpy()
        if reference
        else np.argsort(market, kind="stable")
    )
    selected = order[[0, count // 2, count - 1]]
    xs, ys = market[selected], strategy[selected]
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        span = xs[2] - xs[0]
        constant = bool(span == 0)
        coordinates = xs
        if not np.isfinite(span) and np.all(np.isfinite(xs)):
            coordinates = xs / np.max(np.abs(xs))
            span = coordinates[2] - coordinates[0]
        weights = (
            (0.5, 0.5)
            if constant
            else (
                float((coordinates[2] - coordinates[1]) / span),
                float((coordinates[1] - coordinates[0]) / span),
            )
        )
        gap = float(weights[0] * ys[0] + weights[1] * ys[2] - ys[1])
    positions = tuple(int(ordinals[index]) for index in selected)
    return BetaFragilityResult(
        gap,
        status="ok" if np.isfinite(gap) else "nonfinite_result",
        selected_ordinals=positions,
        selected_keys=tuple(keys[position] for position in positions),
        benchmark_points=tuple(float(value) for value in xs),
        strategy_points=tuple(float(value) for value in ys),
        endpoint_weights=weights,
        constant_benchmark=constant,
        **common,
    )


def beta_fragility(returns: PairedInput, benchmark_returns: PairedInput) -> BetaFragilityResult:
    """Interpolate endpoints at the upper observed middle benchmark coordinate.

    Two Series align by labels; arrays remain positional. Finite pairs only are
    used. Equal benchmark observations keep input order, so the result is not
    permutation-invariant within ties. A constant benchmark uses the declared
    half/half endpoint convention and is flagged, not described as an estimated
    beta. Counts and selected points make undefined and tied samples inspectable.
    """
    return _beta_fragility(returns, benchmark_returns)
