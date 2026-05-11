"""Effective number of trials for correlation-adjusted Sharpe inference.

This module implements correlation-aware estimators for the effective number of
independent strategy trials, K_eff.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike

from ml4t.diagnostic.evaluation.stats.moments import compute_return_statistics

EffectiveTrialsMethod = Literal["effective_rank", "marchenko_pastur", "clustering"]

_EIGENVALUE_TOL = 1e-12
_CLUSTERING_MAX_CANDIDATES = 40


@dataclass
class EffectiveTrialsResult:
    """Result of effective-trials estimation."""

    k_eff: float
    method: str
    eigenvalues: np.ndarray
    n_clusters: int | None = None
    mp_upper_bound: float | None = None
    variance_trials: float | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)


def _coerce_return_matrix(returns: ArrayLike) -> tuple[np.ndarray, int]:
    """Validate and clean a strategy return matrix."""
    matrix = np.asarray(returns, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("returns must be a 2D array of shape (n_periods, n_strategies)")

    n_periods, n_strategies = matrix.shape
    if n_periods < 2:
        raise ValueError("returns must contain at least two periods")
    if n_strategies < 1:
        raise ValueError("returns must contain at least one strategy")

    finite_rows = np.all(np.isfinite(matrix), axis=1)
    dropped_rows = int((~finite_rows).sum())
    clean = matrix[finite_rows]
    if clean.shape[0] < 2:
        raise ValueError("returns must contain at least two fully finite observations")

    std = np.std(clean, axis=0, ddof=1)
    if np.any(std <= _EIGENVALUE_TOL):
        raise ValueError("returns contain a constant strategy; correlation matrix is undefined")

    return clean, dropped_rows


def _effective_rank(eigenvalues: np.ndarray) -> float:
    """Compute effective rank from non-negative eigenvalues."""
    positive = eigenvalues[eigenvalues > _EIGENVALUE_TOL]
    if positive.size == 0:
        return 1.0

    weights = positive / positive.sum()
    entropy = -np.sum(weights * np.log(weights))
    return float(np.exp(entropy))


def _marchenko_pastur_upper_bound(n_periods: int, n_strategies: int) -> float:
    """Upper edge of the Marchenko-Pastur bulk for a correlation matrix."""
    aspect_ratio = n_strategies / n_periods
    return float((1.0 + np.sqrt(aspect_ratio)) ** 2)


def _marchenko_pastur_k_eff(
    eigenvalues: np.ndarray, n_periods: int
) -> tuple[int, float, list[dict[str, float]]]:
    """Iteratively count signal eigenvalues above the M-P upper bound."""
    remaining = eigenvalues.copy()
    total_signal = 0
    iterations: list[dict[str, float]] = []
    while remaining.size > 0:
        upper_bound = _marchenko_pastur_upper_bound(n_periods, remaining.size)
        signal_count = int(np.sum(remaining > upper_bound + _EIGENVALUE_TOL))
        iterations.append(
            {
                "n_remaining": float(remaining.size),
                "mp_upper_bound": upper_bound,
                "n_signal": float(signal_count),
            }
        )
        if signal_count == 0:
            return total_signal, upper_bound, iterations
        total_signal += signal_count
        remaining = remaining[signal_count:]

    return total_signal, _marchenko_pastur_upper_bound(n_periods, 1), iterations


def _minimum_variance_weights(returns: np.ndarray) -> np.ndarray:
    """Compute minimum-variance portfolio weights for a strategy cluster."""
    n_strategies = returns.shape[1]
    if n_strategies == 1:
        return np.array([1.0], dtype=float)

    cov = np.cov(returns, rowvar=False, ddof=1)
    cov = np.atleast_2d(cov)
    inv_cov = np.linalg.pinv(cov)
    ones = np.ones(n_strategies, dtype=float)
    raw = inv_cov @ ones
    denom = float(ones @ raw)
    if abs(denom) <= _EIGENVALUE_TOL:
        return np.full(n_strategies, 1.0 / n_strategies, dtype=float)
    weights = raw / denom
    if not np.all(np.isfinite(weights)):
        return np.full(n_strategies, 1.0 / n_strategies, dtype=float)
    return weights


def _cluster_trial_returns(
    returns: np.ndarray,
    labels: np.ndarray,
) -> tuple[list[np.ndarray], list[float], list[int], list[list[float]]]:
    """Aggregate strategy clusters into representative return series."""
    cluster_returns: list[np.ndarray] = []
    cluster_sharpes: list[float] = []
    cluster_sizes: list[int] = []
    cluster_weights: list[list[float]] = []

    for cluster_id in sorted(np.unique(labels)):
        cluster_slice = returns[:, labels == cluster_id]
        weights = _minimum_variance_weights(cluster_slice)
        agg_returns = cluster_slice @ weights
        sharpe, _, _, _, _ = compute_return_statistics(agg_returns)
        cluster_returns.append(agg_returns)
        cluster_sharpes.append(float(sharpe))
        cluster_sizes.append(int(cluster_slice.shape[1]))
        cluster_weights.append(weights.tolist())

    return cluster_returns, cluster_sharpes, cluster_sizes, cluster_weights


def _silhouette_t_stat(distance_matrix: np.ndarray, labels: np.ndarray) -> float:
    """Compute silhouette t-statistic for a clustering."""
    from sklearn.metrics import silhouette_samples

    unique_labels = np.unique(labels)
    if unique_labels.size < 2 or unique_labels.size >= len(labels):
        return float("-inf")

    values = silhouette_samples(distance_matrix, labels, metric="precomputed")
    if values.size < 2:
        return float("-inf")
    mean_val = float(np.mean(values))
    std_val = float(np.std(values, ddof=1))
    if std_val <= _EIGENVALUE_TOL:
        return float("inf") if mean_val > 0 else float("-inf")
    return mean_val / (std_val / np.sqrt(values.size))


def _clustering_k_eff(
    returns: np.ndarray,
    corr: np.ndarray,
) -> tuple[int, float, dict[str, Any]]:
    """Estimate K_eff by clustering strategies with silhouette selection."""
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform
    from sklearn.metrics import silhouette_score

    n_strategies = returns.shape[1]
    if n_strategies == 1:
        cluster_returns, cluster_sharpes, cluster_sizes, cluster_weights = _cluster_trial_returns(
            returns,
            np.zeros(1, dtype=int),
        )
        return (
            1,
            0.0,
            {
                "labels": [0],
                "cluster_returns": cluster_returns,
                "cluster_sharpes": cluster_sharpes,
                "cluster_sizes": cluster_sizes,
                "cluster_weights": cluster_weights,
                "silhouette_score": None,
                "silhouette_t_stat": None,
                "candidate_scores": [],
            },
        )

    if n_strategies == 2:
        labels = np.array([0, 1], dtype=int)
        cluster_returns, cluster_sharpes, cluster_sizes, cluster_weights = _cluster_trial_returns(
            returns, labels
        )
        variance = float(np.var(cluster_sharpes, ddof=1))
        return (
            2,
            variance,
            {
                "labels": labels.tolist(),
                "cluster_returns": cluster_returns,
                "cluster_sharpes": cluster_sharpes,
                "cluster_sizes": cluster_sizes,
                "cluster_weights": cluster_weights,
                "silhouette_score": None,
                "silhouette_t_stat": None,
                "candidate_scores": [],
            },
        )

    distance = np.sqrt(np.clip(0.5 * (1.0 - corr), 0.0, None))
    np.fill_diagonal(distance, 0.0)
    condensed = squareform(distance, checks=False)
    linkage_matrix = linkage(condensed, method="average")

    max_candidates = min(n_strategies - 1, _CLUSTERING_MAX_CANDIDATES)
    best_labels: np.ndarray | None = None
    best_score = float("-inf")
    candidate_scores: list[dict[str, float]] = []

    for n_clusters in range(2, max_candidates + 1):
        labels = fcluster(linkage_matrix, t=n_clusters, criterion="maxclust") - 1
        if np.unique(labels).size < 2:
            continue
        t_stat = _silhouette_t_stat(distance, labels)
        if not np.isfinite(t_stat):
            continue
        candidate_scores.append({"n_clusters": float(n_clusters), "silhouette_t": t_stat})
        if t_stat > best_score:
            best_score = t_stat
            best_labels = labels

    if best_labels is None:
        best_labels = np.arange(n_strategies, dtype=int)
        best_score = float("-inf")

    cluster_returns, cluster_sharpes, cluster_sizes, cluster_weights = _cluster_trial_returns(
        returns,
        best_labels,
    )
    variance = float(np.var(cluster_sharpes, ddof=1)) if len(cluster_sharpes) > 1 else 0.0
    silhouette = None
    if np.unique(best_labels).size >= 2 and np.unique(best_labels).size < len(best_labels):
        silhouette = float(silhouette_score(distance, best_labels, metric="precomputed"))

    return (
        int(np.unique(best_labels).size),
        variance,
        {
            "labels": best_labels.tolist(),
            "cluster_returns": cluster_returns,
            "cluster_sharpes": cluster_sharpes,
            "cluster_sizes": cluster_sizes,
            "cluster_weights": cluster_weights,
            "silhouette_score": silhouette,
            "silhouette_t_stat": None if not np.isfinite(best_score) else float(best_score),
            "candidate_scores": candidate_scores,
        },
    )


def effective_number_of_trials(
    returns: ArrayLike,
    method: EffectiveTrialsMethod = "effective_rank",
    *,
    random_state: int | None = None,
) -> EffectiveTrialsResult:
    """Estimate the effective number of trials, K_eff, from strategy returns.

    Parameters
    ----------
    returns : array-like
        Strategy return matrix of shape ``(n_periods, n_strategies)``.
    method : {"effective_rank", "marchenko_pastur", "clustering"}, default "effective_rank"
        Estimator for K_eff.
        - ``effective_rank``: eigenvalue-entropy effective rank
        - ``marchenko_pastur``: iterative denoising variant that repeatedly
          removes eigenvalues above the Marchenko-Pastur upper edge
        - ``clustering``: silhouette-selected cluster count on correlation distance
    random_state : int, optional
        Reserved for future estimators that require randomness.
    """
    del random_state  # reserved for future estimators

    clean, dropped_rows = _coerce_return_matrix(returns)
    corr = np.corrcoef(clean, rowvar=False)
    eigenvalues = np.linalg.eigvalsh(corr)
    eigenvalues = np.sort(np.clip(eigenvalues.real, 0.0, None))[::-1]

    positive = eigenvalues[eigenvalues > _EIGENVALUE_TOL]
    entropy = 0.0
    if positive.size > 0:
        weights = positive / positive.sum()
        entropy = float(-np.sum(weights * np.log(weights)))

    base_diagnostics = {
        "n_periods": int(clean.shape[0]),
        "n_strategies": int(clean.shape[1]),
        "dropped_rows": dropped_rows,
        "effective_rank_entropy": entropy,
    }

    if method == "effective_rank":
        k_eff = _effective_rank(eigenvalues)
        k_eff = float(np.clip(k_eff, 1.0, clean.shape[1]))
        return EffectiveTrialsResult(
            k_eff=k_eff,
            method=method,
            eigenvalues=eigenvalues,
            diagnostics=base_diagnostics,
        )

    if method == "marchenko_pastur":
        k_eff, upper_bound, iterations = _marchenko_pastur_k_eff(eigenvalues, clean.shape[0])
        k_eff = float(np.clip(k_eff, 1, clean.shape[1]))
        return EffectiveTrialsResult(
            k_eff=k_eff,
            method=method,
            eigenvalues=eigenvalues,
            mp_upper_bound=upper_bound,
            diagnostics={
                **base_diagnostics,
                "mp_iterations": iterations,
            },
        )

    if method == "clustering":
        k_eff, variance, clustering = _clustering_k_eff(clean, corr)
        return EffectiveTrialsResult(
            k_eff=float(np.clip(k_eff, 1, clean.shape[1])),
            method=method,
            eigenvalues=eigenvalues,
            n_clusters=int(k_eff),
            variance_trials=variance,
            diagnostics={
                **base_diagnostics,
                "cluster_labels": clustering["labels"],
                "cluster_sizes": clustering["cluster_sizes"],
                "cluster_weights": clustering["cluster_weights"],
                "cluster_sharpes": clustering["cluster_sharpes"],
                "silhouette_score": clustering["silhouette_score"],
                "silhouette_t_stat": clustering["silhouette_t_stat"],
                "candidate_scores": clustering["candidate_scores"],
            },
        )

    raise ValueError(f"Unsupported method: {method}")


__all__ = [
    "EffectiveTrialsMethod",
    "EffectiveTrialsResult",
    "effective_number_of_trials",
]
