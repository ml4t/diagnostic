"""Calibration diagnostics for probability forecasts of binary outcomes.

A forecast is calibrated when events forecast with probability ``p`` occur with
frequency ``p``. The functions here score and decompose that property for any
probability forecast: model outputs, analyst probabilities, or prediction-market
prices read as implied probabilities.

- ``brier_score`` and ``log_loss`` are proper scoring rules.
- ``brier_decomposition`` splits the Brier score into reliability, resolution
  and uncertainty (Murphy, 1973), plus the two within-bin terms that make the
  identity exact when forecasts vary inside a bin (Stephenson et al., 2008).
- ``reliability_table`` reports observed frequency against mean forecast per
  probability bin, with a Wilson interval for independent outcomes or a
  cluster-bootstrap interval when outcomes share an event.
- ``expected_calibration_error`` and ``max_calibration_error`` summarize the
  table in one number.
- ``calibration_slope`` regresses outcomes on forecasts; a slope above one is
  the favorite-longshot bias.

All functions accept NumPy arrays, Polars or pandas Series, or sequences.
Forecasts must lie in ``[0, 1]`` and outcomes in ``{0, 1}``; missing values are
rejected rather than dropped so that a filtering decision stays with the caller.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
import polars as pl
import statsmodels.api as sm
from numpy.typing import NDArray
from scipy import sparse, stats

ArrayLike = NDArray[Any] | pl.Series | pd.Series | Sequence[float]

PREDICTION_MARKET_BIN_EDGES: tuple[float, ...] = (
    0.0,
    0.01,
    0.02,
    0.05,
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    0.9,
    0.95,
    0.98,
    0.99,
    1.0,
)
"""Default reliability bins: one-cent bins at the extremes, deciles in the middle.

Prediction-market prices cluster near 0 and 1, and that is where the
favorite-longshot bias concentrates, so equal-width deciles would hide it in
the first and last bin.
"""

_BOOTSTRAP_CHUNK_CELLS = 4_000_000


# ============================================================================
# Result types
# ============================================================================


@dataclass(frozen=True)
class BrierDecomposition:
    """Murphy decomposition of the Brier score.

    The identity
    ``brier = reliability - resolution + uncertainty + within_bin_variance -
    within_bin_covariance`` holds exactly. The two within-bin terms are zero
    when every bin holds a single forecast value, which is the default binning
    of ``brier_decomposition``.

    Attributes
    ----------
    brier
        Mean squared difference between forecast and outcome.
    reliability
        Weighted mean squared gap between each bin's mean forecast and its
        observed frequency. Lower is better; zero is perfectly calibrated.
    resolution
        Weighted variance of bin observed frequencies around the overall base
        rate. Higher is better: the forecast separates events from non-events.
    uncertainty
        ``base_rate * (1 - base_rate)``, the Brier score of always forecasting
        the base rate. It depends on the outcomes only.
    within_bin_variance
        Weighted mean squared deviation of forecasts from their bin mean.
    within_bin_covariance
        Twice the weighted mean within-bin covariance of forecasts and outcomes.
    base_rate
        Weighted mean outcome.
    n_bins
        Number of non-empty bins used.
    n
        Number of observations.
    """

    brier: float
    reliability: float
    resolution: float
    uncertainty: float
    within_bin_variance: float
    within_bin_covariance: float
    base_rate: float
    n_bins: int
    n: int

    @property
    def brier_skill_score(self) -> float:
        """``1 - brier / uncertainty``; positive beats the base-rate forecast."""
        if self.uncertainty == 0.0:
            return float("nan")
        return 1.0 - self.brier / self.uncertainty


@dataclass(frozen=True)
class CalibrationSlope:
    """Regression of outcomes on forecasts.

    ``outcome = intercept + slope * forecast`` for ``spec="linear"`` (a
    Mincer-Zarnowitz regression on probabilities), or
    ``logit P(outcome) = intercept + slope * logit(forecast)`` for
    ``spec="logit"`` (Cox's calibration slope). A calibrated forecast has
    ``slope = 1`` and ``intercept = 0``.

    ``slope > 1``: outcomes rise faster than forecasts, so low-probability
    events (longshots) happen less often than priced and high-probability
    events (favorites) more often. Longshots are overpriced and favorites
    underpriced: the favorite-longshot bias. ``slope < 1`` is the reverse, the
    pattern of an overconfident forecaster.

    Attributes
    ----------
    intercept, slope
        Point estimates.
    intercept_se, slope_se
        Sandwich standard errors with the small-sample factor
        ``G / (G - 1) * (n - 1) / (n - 2)``: cluster-robust (CR1) with ``G``
        clusters when clusters are given, heteroskedasticity-robust (HC1,
        ``G = n``) otherwise.
    intercept_ci, slope_ci
        Wald intervals at ``confidence``, using a t distribution with
        ``n_clusters - 1`` degrees of freedom under clustering and ``n - 2``
        otherwise.
    slope_p_value
        Two-sided p-value for the null ``slope = 1``.
    spec
        ``"linear"`` or ``"logit"``.
    se_type
        ``"cluster"`` or ``"HC1"``.
    confidence
        Confidence level of the intervals.
    n
        Number of observations.
    n_clusters
        Number of distinct clusters, or ``n`` without clusters.
    """

    intercept: float
    slope: float
    intercept_se: float
    slope_se: float
    intercept_ci: tuple[float, float]
    slope_ci: tuple[float, float]
    slope_p_value: float
    spec: str
    se_type: str
    confidence: float
    n: int
    n_clusters: int


# ============================================================================
# Input handling
# ============================================================================


def _to_numpy(values: ArrayLike | None) -> NDArray[Any] | None:
    if values is None:
        return None
    if isinstance(values, pl.Series | pd.Series):
        return values.to_numpy()
    return np.asarray(values)


def _prepare(
    forecasts: ArrayLike,
    outcomes: ArrayLike,
    weights: ArrayLike | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    f_raw = _to_numpy(forecasts)
    y_raw = _to_numpy(outcomes)
    assert f_raw is not None and y_raw is not None
    if f_raw.ndim != 1 or y_raw.ndim != 1:
        raise ValueError("forecasts and outcomes must be one-dimensional")
    if len(f_raw) != len(y_raw):
        raise ValueError(
            f"forecasts and outcomes must have the same length, got {len(f_raw)} and {len(y_raw)}"
        )
    if len(f_raw) == 0:
        raise ValueError("forecasts and outcomes must not be empty")

    f = f_raw.astype(np.float64)
    y = y_raw.astype(np.float64)
    if not np.all(np.isfinite(f)):
        raise ValueError("forecasts contain NaN or infinite values")
    if not np.all(np.isfinite(y)):
        raise ValueError("outcomes contain NaN or infinite values")
    if f.min() < 0.0 or f.max() > 1.0:
        raise ValueError("forecasts must lie in [0, 1]")
    if not np.all((y == 0.0) | (y == 1.0)):
        raise ValueError("outcomes must be 0 or 1")

    w_raw = _to_numpy(weights)
    if w_raw is None:
        w = np.ones_like(f)
    else:
        if w_raw.shape != f.shape:
            raise ValueError("weights must have the same length as forecasts")
        w = w_raw.astype(np.float64)
        if not np.all(np.isfinite(w)) or w.min() < 0.0:
            raise ValueError("weights must be finite and non-negative")
        if w.sum() <= 0.0:
            raise ValueError("weights must not all be zero")
    return f, y, w


def _cluster_codes(clusters: ArrayLike, n: int) -> tuple[NDArray[np.int64], int]:
    raw = _to_numpy(clusters)
    assert raw is not None
    if raw.shape != (n,):
        raise ValueError("clusters must have the same length as forecasts")
    if pd.isna(raw).any():
        raise ValueError("clusters contain missing values")
    codes, uniques = pd.factorize(raw, sort=False)
    return codes.astype(np.int64), len(uniques)


def _validate_edges(bins: Sequence[float]) -> NDArray[np.float64]:
    # Rounding snaps computed edges such as np.linspace(0, 1, 11)[3] ==
    # 0.30000000000000004 to the decimal a price of 0.3 is stored as, so a
    # price on an edge lands in the bin above it as documented.
    edges = np.round(np.asarray(bins, dtype=np.float64), 12)
    if edges.ndim != 1 or len(edges) < 2:
        raise ValueError("bins must contain at least two edges")
    if not np.all(np.diff(edges) > 0):
        raise ValueError("bins must be strictly increasing")
    if edges[0] != 0.0 or edges[-1] != 1.0:
        raise ValueError("bins must start at 0 and end at 1")
    return edges


def _bin_index(f: NDArray[np.float64], edges: NDArray[np.float64]) -> NDArray[np.int64]:
    """Assign forecasts to half-open bins ``[lo, hi)``; the last bin includes 1."""
    idx = np.searchsorted(edges, f, side="right") - 1
    return np.clip(idx, 0, len(edges) - 2).astype(np.int64)


def _z(confidence: float) -> float:
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    return float(stats.norm.ppf(0.5 + confidence / 2.0))


# ============================================================================
# Scores
# ============================================================================


def brier_score(
    forecasts: ArrayLike,
    outcomes: ArrayLike,
    *,
    weights: ArrayLike | None = None,
) -> float:
    """Mean squared error of probability forecasts against binary outcomes.

    Parameters
    ----------
    forecasts
        Probabilities in ``[0, 1]``.
    outcomes
        Realized outcomes, 0 or 1.
    weights
        Optional non-negative observation weights.

    Returns
    -------
    float
        Weighted mean of ``(forecast - outcome) ** 2``; 0 is perfect, 0.25 is
        the score of a constant 0.5 forecast.
    """
    f, y, w = _prepare(forecasts, outcomes, weights)
    return float(np.average((f - y) ** 2, weights=w))


def log_loss(
    forecasts: ArrayLike,
    outcomes: ArrayLike,
    *,
    weights: ArrayLike | None = None,
    eps: float = 1e-15,
) -> float:
    """Mean negative log-likelihood of binary outcomes under the forecasts.

    Forecasts are clipped to ``[eps, 1 - eps]`` first. Without clipping, a
    single market priced at 0 or 1 that resolves the other way makes the loss
    infinite. With the default ``eps`` such an observation contributes about
    34.5, so report how many forecasts were clipped when prices of exactly 0
    or 1 occur in the data.

    Parameters
    ----------
    forecasts
        Probabilities in ``[0, 1]``.
    outcomes
        Realized outcomes, 0 or 1.
    weights
        Optional non-negative observation weights.
    eps
        Clipping bound, in ``(0, 0.5)``.

    Returns
    -------
    float
        Weighted mean of ``-(y log p + (1 - y) log(1 - p))`` in nats.
    """
    if not 0.0 < eps < 0.5:
        raise ValueError("eps must be in (0, 0.5)")
    f, y, w = _prepare(forecasts, outcomes, weights)
    p = np.clip(f, eps, 1.0 - eps)
    losses = -(y * np.log(p) + (1.0 - y) * np.log1p(-p))
    return float(np.average(losses, weights=w))


def brier_decomposition(
    forecasts: ArrayLike,
    outcomes: ArrayLike,
    *,
    bins: Sequence[float] | None = None,
    weights: ArrayLike | None = None,
) -> BrierDecomposition:
    """Decompose the Brier score into reliability, resolution and uncertainty.

    With ``bins=None`` every distinct forecast value is its own bin, which
    gives Murphy's (1973) exact decomposition with zero within-bin terms. That
    is the natural choice for prices quoted on a tick grid (prediction markets
    quote in cents, so there are at most 101 values). For continuous model
    probabilities pass bin edges; the two within-bin terms then absorb the
    forecast variation inside each bin, keeping the identity exact
    (Stephenson, Coelho and Jolliffe, 2008).

    Parameters
    ----------
    forecasts
        Probabilities in ``[0, 1]``.
    outcomes
        Realized outcomes, 0 or 1.
    bins
        Strictly increasing bin edges from 0 to 1, or ``None`` for one bin per
        distinct forecast value.
    weights
        Optional non-negative observation weights.

    Returns
    -------
    BrierDecomposition
        Components satisfying ``brier = reliability - resolution +
        uncertainty + within_bin_variance - within_bin_covariance``.
    """
    f, y, w = _prepare(forecasts, outcomes, weights)
    if bins is None:
        _, idx = np.unique(f, return_inverse=True)
        idx = idx.astype(np.int64)
    else:
        idx = _bin_index(f, _validate_edges(bins))

    k = int(idx.max()) + 1
    total = w.sum()
    w_k = np.bincount(idx, weights=w, minlength=k)
    nonempty = w_k > 0
    f_bar = np.zeros(k)
    o_bar = np.zeros(k)
    f_bar[nonempty] = np.bincount(idx, weights=w * f, minlength=k)[nonempty] / w_k[nonempty]
    o_bar[nonempty] = np.bincount(idx, weights=w * y, minlength=k)[nonempty] / w_k[nonempty]
    base_rate = float((w * y).sum() / total)

    df = f - f_bar[idx]
    dy = y - o_bar[idx]
    return BrierDecomposition(
        brier=float((w * (f - y) ** 2).sum() / total),
        reliability=float((w_k * (f_bar - o_bar) ** 2).sum() / total),
        resolution=float((w_k * (o_bar - base_rate) ** 2).sum() / total),
        uncertainty=base_rate * (1.0 - base_rate),
        within_bin_variance=float((w * df**2).sum() / total),
        within_bin_covariance=float(2.0 * (w * df * dy).sum() / total),
        base_rate=base_rate,
        n_bins=int(nonempty.sum()),
        n=len(f),
    )


# ============================================================================
# Reliability
# ============================================================================


def _wilson(
    p_hat: NDArray[np.float64], n: NDArray[np.float64], z: float
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Vectorized Wilson score interval; ``n`` may be an effective sample size."""
    with np.errstate(invalid="ignore", divide="ignore"):
        denom = 1.0 + z**2 / n
        center = (p_hat + z**2 / (2.0 * n)) / denom
        margin = z * np.sqrt((p_hat * (1.0 - p_hat) + z**2 / (4.0 * n)) / n) / denom
    return center - margin, center + margin


def _cluster_bootstrap_counts(
    n_clusters: int,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> Iterator[NDArray[np.float64]]:
    """Yield blocks of cluster multiplicities, one row per bootstrap replicate.

    Each row counts how often each cluster is drawn when ``n_clusters`` clusters
    are sampled with replacement, so a statistic built from per-cluster sums is
    recomputed for a replicate by a matrix product with the row.
    """
    chunk = max(1, _BOOTSTRAP_CHUNK_CELLS // n_clusters)
    done = 0
    while done < n_bootstrap:
        b = min(chunk, n_bootstrap - done)
        draws = rng.integers(0, n_clusters, size=(b, n_clusters))
        offsets = (np.arange(b) * n_clusters)[:, None]
        counts = np.bincount((draws + offsets).ravel(), minlength=b * n_clusters)
        yield counts.reshape(b, n_clusters).astype(np.float64)
        done += b


def reliability_table(
    forecasts: ArrayLike,
    outcomes: ArrayLike,
    *,
    bins: Sequence[float] = PREDICTION_MARKET_BIN_EDGES,
    clusters: ArrayLike | None = None,
    weights: ArrayLike | None = None,
    confidence: float = 0.95,
    n_bootstrap: int = 1000,
    seed: int | np.random.Generator | None = 0,
) -> pl.DataFrame:
    """Observed outcome frequency against mean forecast, per probability bin.

    Bins are half-open ``[lower, upper)``; the last bin also holds forecasts
    equal to 1. Every bin appears in the output, and statistics of an empty bin
    are null.

    The interval for the observed frequency depends on the inputs:

    - no clusters, no weights: Wilson score interval (``ci_method="wilson"``);
    - weights, no clusters: Wilson interval on the Kish effective sample size
      ``(sum w)^2 / sum w^2`` (``"wilson_effective_n"``);
    - clusters: percentile interval from a cluster bootstrap that resamples
      whole clusters with replacement across all bins (``"cluster_bootstrap"``).
      Use this when outcomes are correlated within a group, for example
      several markets on one event. A Wilson interval treats them as
      independent and is too narrow. Bootstrap replicates in which a bin is
      empty are left out of that bin's interval.

    Parameters
    ----------
    forecasts
        Probabilities in ``[0, 1]``.
    outcomes
        Realized outcomes, 0 or 1.
    bins
        Strictly increasing bin edges from 0 to 1. The default,
        ``PREDICTION_MARKET_BIN_EDGES``, is finer near 0 and 1.
    clusters
        Optional cluster label per observation (any hashable dtype).
    weights
        Optional non-negative observation weights.
    confidence
        Confidence level of the interval.
    n_bootstrap
        Number of cluster-bootstrap replicates; ignored without clusters.
    seed
        Seed or Generator for the cluster bootstrap. The default makes repeated
        calls on the same data return the same interval; ``None`` draws a fresh
        stream.

    Returns
    -------
    pl.DataFrame
        One row per bin with columns ``bin_lower``, ``bin_upper``, ``n``,
        ``n_clusters``, ``weight``, ``mean_forecast``, ``observed_frequency``,
        ``difference`` (observed minus forecast; negative means the bin is
        overpriced), ``ci_lower``, ``ci_upper`` and ``ci_method``.
    """
    f, y, w = _prepare(forecasts, outcomes, weights)
    edges = _validate_edges(bins)
    z = _z(confidence)
    idx = _bin_index(f, edges)
    k = len(edges) - 1

    n_k = np.bincount(idx, minlength=k)
    w_k = np.bincount(idx, weights=w, minlength=k)
    wy_k = np.bincount(idx, weights=w * y, minlength=k)
    wf_k = np.bincount(idx, weights=w * f, minlength=k)
    nonempty = w_k > 0
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_f = np.where(nonempty, wf_k / w_k, np.nan)
        obs = np.where(nonempty, wy_k / w_k, np.nan)

    if clusters is None:
        nclu_k = n_k.copy()
        if weights is None:
            n_eff = n_k.astype(np.float64)
            method = "wilson"
        else:
            w2_k = np.bincount(idx, weights=w**2, minlength=k)
            with np.errstate(invalid="ignore", divide="ignore"):
                n_eff = np.where(w2_k > 0, w_k**2 / w2_k, 0.0)
            method = "wilson_effective_n"
        lo, hi = _wilson(obs, n_eff, z)
    else:
        if isinstance(n_bootstrap, bool) or not isinstance(n_bootstrap, int) or n_bootstrap < 2:
            raise ValueError("n_bootstrap must be an integer of at least 2")
        codes, g = _cluster_codes(clusters, len(f))
        nclu_k = np.bincount(np.unique(codes * k + idx) % k, minlength=k)
        rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
        # Per-cluster, per-bin sums; a replicate's bin sums are counts @ sums.
        s_w = sparse.csr_matrix((w, (codes, idx)), shape=(g, k))
        s_wy = sparse.csr_matrix((w * y, (codes, idx)), shape=(g, k))
        replicates = []
        for counts in _cluster_bootstrap_counts(g, n_bootstrap, rng):
            bw = np.asarray((s_w.T @ counts.T).T)
            bwy = np.asarray((s_wy.T @ counts.T).T)
            with np.errstate(invalid="ignore", divide="ignore"):
                replicates.append(np.where(bw > 0, bwy / bw, np.nan))
        boot = np.vstack(replicates)
        alpha = 1.0 - confidence
        lo = np.full(k, np.nan)
        hi = np.full(k, np.nan)
        for j in np.flatnonzero(nonempty):
            col = boot[:, j]
            col = col[~np.isnan(col)]
            if col.size:
                lo[j], hi[j] = np.quantile(col, [alpha / 2.0, 1.0 - alpha / 2.0])
        method = "cluster_bootstrap"

    def _nullable(values: NDArray[np.float64]) -> list[float | None]:
        return [float(v) if m else None for v, m in zip(values, nonempty, strict=True)]

    return pl.DataFrame(
        {
            "bin_lower": edges[:-1],
            "bin_upper": edges[1:],
            "n": n_k.astype(np.int64),
            "n_clusters": nclu_k.astype(np.int64),
            "weight": w_k,
            "mean_forecast": _nullable(mean_f),
            "observed_frequency": _nullable(obs),
            "difference": _nullable(obs - mean_f),
            "ci_lower": _nullable(lo),
            "ci_upper": _nullable(hi),
            "ci_method": [method] * k,
        },
        schema={
            "bin_lower": pl.Float64,
            "bin_upper": pl.Float64,
            "n": pl.Int64,
            "n_clusters": pl.Int64,
            "weight": pl.Float64,
            "mean_forecast": pl.Float64,
            "observed_frequency": pl.Float64,
            "difference": pl.Float64,
            "ci_lower": pl.Float64,
            "ci_upper": pl.Float64,
            "ci_method": pl.String,
        },
    )


def _bin_gaps(
    forecasts: ArrayLike,
    outcomes: ArrayLike,
    bins: Sequence[float],
    weights: ArrayLike | None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    f, y, w = _prepare(forecasts, outcomes, weights)
    edges = _validate_edges(bins)
    idx = _bin_index(f, edges)
    k = len(edges) - 1
    w_k = np.bincount(idx, weights=w, minlength=k)
    nonempty = w_k > 0
    w_k = w_k[nonempty]
    f_bar = np.bincount(idx, weights=w * f, minlength=k)[nonempty] / w_k
    o_bar = np.bincount(idx, weights=w * y, minlength=k)[nonempty] / w_k
    return np.abs(o_bar - f_bar), w_k / w_k.sum()


def expected_calibration_error(
    forecasts: ArrayLike,
    outcomes: ArrayLike,
    *,
    bins: Sequence[float] = PREDICTION_MARKET_BIN_EDGES,
    weights: ArrayLike | None = None,
) -> float:
    """Weight-averaged absolute gap between observed frequency and mean forecast.

    ``ECE = sum_k (W_k / W) * |observed_k - forecast_k|`` over non-empty bins,
    using the same bins as ``reliability_table``. The value depends on the bin
    edges, so report them with it. Sampling noise inflates ECE in thin bins:
    a perfectly calibrated forecast has positive expected ECE.

    Parameters
    ----------
    forecasts
        Probabilities in ``[0, 1]``.
    outcomes
        Realized outcomes, 0 or 1.
    bins
        Strictly increasing bin edges from 0 to 1.
    weights
        Optional non-negative observation weights.

    Returns
    -------
    float
        Expected calibration error in probability units.
    """
    gaps, shares = _bin_gaps(forecasts, outcomes, bins, weights)
    return float((gaps * shares).sum())


def max_calibration_error(
    forecasts: ArrayLike,
    outcomes: ArrayLike,
    *,
    bins: Sequence[float] = PREDICTION_MARKET_BIN_EDGES,
    weights: ArrayLike | None = None,
) -> float:
    """Largest absolute gap between observed frequency and mean forecast.

    Taken over non-empty bins. A bin with a handful of observations can
    dominate this number; check ``n`` and the interval in
    ``reliability_table`` for the bin that attains it.

    Parameters
    ----------
    forecasts
        Probabilities in ``[0, 1]``.
    outcomes
        Realized outcomes, 0 or 1.
    bins
        Strictly increasing bin edges from 0 to 1.
    weights
        Optional non-negative observation weights.

    Returns
    -------
    float
        Maximum calibration error in probability units.
    """
    gaps, _ = _bin_gaps(forecasts, outcomes, bins, weights)
    return float(gaps.max())


# ============================================================================
# Calibration slope
# ============================================================================


def calibration_slope(
    forecasts: ArrayLike,
    outcomes: ArrayLike,
    *,
    spec: Literal["linear", "logit"] = "linear",
    clusters: ArrayLike | None = None,
    weights: ArrayLike | None = None,
    confidence: float = 0.95,
    eps: float = 1e-3,
) -> CalibrationSlope:
    """Regress outcomes on forecasts to measure the favorite-longshot bias.

    ``spec="linear"`` fits ``outcome = a + b * forecast`` by (weighted) least
    squares, a Mincer-Zarnowitz regression on probabilities. ``spec="logit"``
    fits the logistic regression ``logit P(outcome) = a + b * logit(forecast)``,
    Cox's (1958) calibration slope, which respects the ``[0, 1]`` range and
    weights the tails more than the linear form does.

    Interpretation, for either specification:

    - ``b = 1`` and ``a = 0``: calibrated.
    - ``b > 1``: favorite-longshot bias. Low-priced outcomes (longshots) win
      less often than their price implies and high-priced outcomes
      (favorites) more often, so longshots are overpriced and favorites
      underpriced.
    - ``b < 1``: reverse bias; forecasts are too extreme (overconfident).

    Standard errors are cluster-robust (CR1) when ``clusters`` is given and
    HC1 otherwise. Pass clusters whenever outcomes are correlated within a
    group, such as markets that resolve on one event; ignoring that
    correlation understates the standard errors.

    Parameters
    ----------
    forecasts
        Probabilities in ``[0, 1]``.
    outcomes
        Realized outcomes, 0 or 1.
    spec
        ``"linear"`` or ``"logit"``.
    clusters
        Optional cluster label per observation.
    weights
        Optional non-negative observation weights.
    confidence
        Confidence level of the Wald intervals.
    eps
        For ``spec="logit"``, forecasts are clipped to ``[eps, 1 - eps]``
        before the logit transform. A price of 0 or 1 has an infinite logit,
        and very small ``eps`` gives such observations extreme leverage.
        Ignored for ``spec="linear"``.

    Returns
    -------
    CalibrationSlope
        Coefficients, standard errors, intervals and the p-value for
        ``slope = 1``.
    """
    if spec not in ("linear", "logit"):
        raise ValueError("spec must be 'linear' or 'logit'")
    if not 0.0 < eps < 0.5:
        raise ValueError("eps must be in (0, 0.5)")
    _z(confidence)
    f, y, w = _prepare(forecasts, outcomes, weights)

    if spec == "logit":
        p = np.clip(f, eps, 1.0 - eps)
        x = np.log(p) - np.log1p(-p)
    else:
        x = f
    if np.ptp(x) == 0.0:
        raise ValueError("forecasts are constant; the slope is not identified")
    exog = np.column_stack([np.ones_like(x), x])

    n = len(f)
    if spec == "linear":
        sw = np.sqrt(w)
        params = np.linalg.lstsq(exog * sw[:, None], y * sw, rcond=None)[0]
        mu = exog @ params
        hessian = exog.T @ (exog * w[:, None])
    else:
        glm = sm.GLM(y, exog, family=sm.families.Binomial(), var_weights=w).fit()
        params = np.asarray(glm.params, dtype=np.float64)
        mu = 1.0 / (1.0 + np.exp(-(exog @ params)))
        hessian = exog.T @ (exog * (w * mu * (1.0 - mu))[:, None])
    scores = exog * (w * (y - mu))[:, None]

    if clusters is None:
        n_groups = n
        meat_scores = scores
        se_type = "HC1"
    else:
        codes, n_groups = _cluster_codes(clusters, n)
        if n_groups < 2:
            raise ValueError("cluster-robust standard errors need at least two clusters")
        meat_scores = np.column_stack(
            [np.bincount(codes, weights=scores[:, j], minlength=n_groups) for j in range(2)]
        )
        se_type = "cluster"
    bread = np.linalg.inv(hessian)
    correction = n_groups / (n_groups - 1) * (n - 1) / (n - 2)
    cov = correction * bread @ (meat_scores.T @ meat_scores) @ bread
    bse = np.sqrt(np.diag(cov))

    t_dist = stats.t(df=n_groups - 1 if clusters is not None else n - 2)
    q = float(t_dist.ppf(0.5 + confidence / 2.0))
    ci = np.column_stack([params - q * bse, params + q * bse])
    slope_p_value = float(2.0 * t_dist.sf(abs((params[1] - 1.0) / bse[1])))
    return CalibrationSlope(
        intercept=float(params[0]),
        slope=float(params[1]),
        intercept_se=float(bse[0]),
        slope_se=float(bse[1]),
        intercept_ci=(float(ci[0, 0]), float(ci[0, 1])),
        slope_ci=(float(ci[1, 0]), float(ci[1, 1])),
        slope_p_value=slope_p_value,
        spec=spec,
        se_type=se_type,
        confidence=confidence,
        n=n,
        n_clusters=int(n_groups),
    )
