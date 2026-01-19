"""Wasserstein distance for continuous distribution drift detection.

The Wasserstein distance (Earth Mover's Distance) measures the minimum cost
to transform one probability distribution into another.

Properties:
- True metric: non-negative, symmetric, triangle inequality
- More sensitive to small shifts than PSI
- Natural interpretation as "transport cost"
- No binning artifacts

References:
    - Villani, C. (2009). Optimal Transport: Old and New. Springer.
    - Ramdas, A., et al. (2017). On Wasserstein Two-Sample Testing.
      Entropy, 19(2), 47.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl
from scipy.stats import wasserstein_distance


@dataclass
class WassersteinResult:
    """Result of Wasserstein distance calculation.

    The Wasserstein distance (also called Earth Mover's Distance) measures the
    minimum "cost" to transform one distribution into another. It's a true metric
    and doesn't require binning, making it ideal for continuous features.

    Attributes:
        distance: Wasserstein distance value (W_p)
        p: Order of Wasserstein distance (1 or 2)
        threshold: Calibrated threshold from permutation test (if calibrated)
        p_value: Statistical significance p-value (if calibrated)
        drifted: Whether drift was detected (distance > threshold)
        n_reference: Number of samples in reference distribution
        n_test: Number of samples in test distribution
        reference_stats: Summary statistics of reference distribution
        test_stats: Summary statistics of test distribution
        threshold_calibration_config: Configuration used for threshold calibration
        interpretation: Human-readable interpretation
        computation_time: Time taken to compute (seconds)
    """

    distance: float
    p: int
    threshold: float | None
    p_value: float | None
    drifted: bool
    n_reference: int
    n_test: int
    reference_stats: dict[str, float]
    test_stats: dict[str, float]
    threshold_calibration_config: dict[str, Any] | None
    interpretation: str
    computation_time: float

    def summary(self) -> str:
        """Return formatted summary of Wasserstein distance results."""
        lines = [
            "Wasserstein Distance Drift Detection Report",
            "=" * 60,
            f"Wasserstein-{self.p} Distance: {self.distance:.6f}",
            f"Drift Detected: {'YES' if self.drifted else 'NO'}",
            "",
            "Sample Sizes:",
            f"  Reference: {self.n_reference:,}",
            f"  Test: {self.n_test:,}",
            "",
        ]

        if self.threshold is not None:
            lines.extend(
                [
                    "Threshold Calibration:",
                    f"  Threshold: {self.threshold:.6f}",
                    f"  P-value: {self.p_value:.4f}" if self.p_value else "  P-value: N/A",
                    f"  Config: {self.threshold_calibration_config}",
                    "",
                ]
            )

        lines.extend(
            [
                "Distribution Statistics:",
                "-" * 60,
                f"Reference: Mean={self.reference_stats['mean']:.4f}, "
                f"Std={self.reference_stats['std']:.4f}, "
                f"Min={self.reference_stats['min']:.4f}, "
                f"Max={self.reference_stats['max']:.4f}",
                f"Test:      Mean={self.test_stats['mean']:.4f}, "
                f"Std={self.test_stats['std']:.4f}, "
                f"Min={self.test_stats['min']:.4f}, "
                f"Max={self.test_stats['max']:.4f}",
                "",
                f"Interpretation: {self.interpretation}",
                "",
                f"Computation Time: {self.computation_time:.3f}s",
            ]
        )

        return "\n".join(lines)


def compute_wasserstein_distance(
    reference: np.ndarray | pl.Series,
    test: np.ndarray | pl.Series,
    p: int = 1,
    threshold_calibration: bool = True,
    n_permutations: int = 1000,
    alpha: float = 0.05,
    n_samples: int | None = None,
    random_state: int | None = None,
) -> WassersteinResult:
    """Compute Wasserstein distance between reference and test distributions.

    The Wasserstein distance (Earth Mover's Distance) measures the minimum cost
    to transform one probability distribution into another. Unlike PSI, it doesn't
    require binning and provides a true metric with desirable properties:
    - Metric properties: non-negative, symmetric, triangle inequality
    - More sensitive to small shifts than PSI
    - Natural interpretation as "transport cost"
    - No binning artifacts

    The p-Wasserstein distance is defined as:
        W_p(P, Q) = (∫|F_P^{-1}(u) - F_Q^{-1}(u)|^p du)^{1/p}

    For empirical distributions with sorted samples x_1 ≤ ... ≤ x_n:
        W_1(P, Q) = (1/n) Σ|x_i^P - x_i^Q|

    Threshold calibration uses a permutation test:
        H0: reference and test come from the same distribution
        H1: distributions differ

    Args:
        reference: Reference distribution (e.g., training data)
        test: Test distribution (e.g., production data)
        p: Order of Wasserstein distance (1 or 2). Default: 1
            - p=1: More robust, easier to interpret
            - p=2: More sensitive to tail differences
        threshold_calibration: Whether to calibrate threshold via permutation test
        n_permutations: Number of permutations for threshold calibration
        alpha: Significance level for threshold (default: 0.05)
        n_samples: Subsample to this many samples if provided (for large datasets)
        random_state: Random seed for reproducibility

    Returns:
        WassersteinResult with distance, threshold, p-value, and interpretation

    Raises:
        ValueError: If inputs are invalid or p not in {1, 2}

    Example:
        >>> # Detect mean shift
        >>> ref = np.random.normal(0, 1, 1000)
        >>> test = np.random.normal(0.5, 1, 1000)  # Mean shifted by 0.5
        >>> result = compute_wasserstein_distance(ref, test)
        >>> print(result.summary())
        >>>
        >>> # Detect variance shift
        >>> test_var = np.random.normal(0, 2, 1000)  # Variance doubled
        >>> result = compute_wasserstein_distance(ref, test_var)
        >>> print(f"Distance: {result.distance:.4f}, Drifted: {result.drifted}")
        >>>
        >>> # Without threshold calibration (faster)
        >>> result = compute_wasserstein_distance(
        ...     ref, test, threshold_calibration=False
        ... )
    """
    start_time = time.time()

    # Convert to numpy arrays
    if isinstance(reference, pl.Series):
        reference = reference.to_numpy()
    if isinstance(test, pl.Series):
        test = test.to_numpy()

    reference = np.asarray(reference, dtype=np.float64)
    test = np.asarray(test, dtype=np.float64)

    # Validate inputs
    if len(reference) == 0 or len(test) == 0:
        raise ValueError("Reference and test arrays must not be empty")

    if p not in [1, 2]:
        raise ValueError(f"p must be 1 or 2, got {p}")

    # Set random state
    if random_state is not None:
        np.random.seed(random_state)

    # Subsample if requested
    if n_samples is not None and len(reference) > n_samples:
        indices_ref = np.random.choice(len(reference), n_samples, replace=False)
        reference = reference[indices_ref]
    if n_samples is not None and len(test) > n_samples:
        indices_test = np.random.choice(len(test), n_samples, replace=False)
        test = test[indices_test]

    n_reference = len(reference)
    n_test = len(test)

    # Compute distribution statistics
    reference_stats = {
        "mean": float(np.mean(reference)),
        "std": float(np.std(reference)),
        "min": float(np.min(reference)),
        "max": float(np.max(reference)),
        "median": float(np.median(reference)),
        "q25": float(np.percentile(reference, 25)),
        "q75": float(np.percentile(reference, 75)),
    }

    test_stats = {
        "mean": float(np.mean(test)),
        "std": float(np.std(test)),
        "min": float(np.min(test)),
        "max": float(np.max(test)),
        "median": float(np.median(test)),
        "q25": float(np.percentile(test, 25)),
        "q75": float(np.percentile(test, 75)),
    }

    # Compute Wasserstein distance
    if p == 1:
        distance = float(wasserstein_distance(reference, test))
    else:  # p == 2
        # scipy's wasserstein_distance computes W_1
        # For W_2, we need to compute it manually
        distance = _wasserstein_2(reference, test)

    # Threshold calibration via permutation test
    threshold = None
    p_value = None
    calibration_config = None

    if threshold_calibration:
        threshold, p_value = _calibrate_wasserstein_threshold(
            reference, test, distance, n_permutations, alpha, p
        )
        calibration_config = {
            "n_permutations": n_permutations,
            "alpha": alpha,
            "method": "permutation",
        }

    # Determine drift status
    if threshold is not None:
        drifted = distance > threshold
    else:
        # Without calibration, use heuristic based on distribution statistics
        # Drift if distance > 0.5 * std of reference
        drifted = distance > 0.5 * reference_stats["std"]
        threshold = 0.5 * reference_stats["std"]

    # Generate interpretation
    if drifted:
        if p_value is not None:
            interpretation = (
                f"Distribution drift detected (W_{p}={distance:.6f} > {threshold:.6f}, "
                f"p={p_value:.4f}). The test distribution differs significantly from "
                f"the reference distribution."
            )
        else:
            interpretation = (
                f"Distribution drift detected (W_{p}={distance:.6f} > {threshold:.6f}). "
                f"The test distribution differs from the reference distribution."
            )
    else:
        if p_value is not None:
            interpretation = (
                f"No significant drift detected (W_{p}={distance:.6f} ≤ {threshold:.6f}, "
                f"p={p_value:.4f}). Distributions are consistent."
            )
        else:
            interpretation = f"No significant drift detected (W_{p}={distance:.6f} ≤ {threshold:.6f}). Distributions are consistent."

    computation_time = time.time() - start_time

    return WassersteinResult(
        distance=distance,
        p=p,
        threshold=threshold,
        p_value=p_value,
        drifted=drifted,
        n_reference=n_reference,
        n_test=n_test,
        reference_stats=reference_stats,
        test_stats=test_stats,
        threshold_calibration_config=calibration_config,
        interpretation=interpretation,
        computation_time=computation_time,
    )


def _wasserstein_2(u_values: np.ndarray, v_values: np.ndarray) -> float:
    """Compute Wasserstein-2 distance between two 1D distributions.

    W_2(P, Q) = sqrt(∫|F_P^{-1}(u) - F_Q^{-1}(u)|^2 du)

    For empirical distributions, this is computed as:
    W_2 = sqrt((1/n) Σ(x_i - y_i)^2) where x, y are sorted samples

    Args:
        u_values: First distribution samples
        v_values: Second distribution samples

    Returns:
        Wasserstein-2 distance
    """
    u_sorted = np.sort(u_values)
    v_sorted = np.sort(v_values)

    # Align to same length via CDF interpolation
    # Use linear interpolation between sorted samples
    n = min(len(u_sorted), len(v_sorted))
    u_quantiles = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(u_sorted)), u_sorted)
    v_quantiles = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(v_sorted)), v_sorted)

    # Compute L2 distance
    return float(np.sqrt(np.mean((u_quantiles - v_quantiles) ** 2)))


def _calibrate_wasserstein_threshold(
    reference: np.ndarray,
    test: np.ndarray,
    observed_distance: float,
    n_permutations: int,
    alpha: float,
    p: int,
) -> tuple[float, float]:
    """Calibrate Wasserstein distance threshold via permutation test.

    Tests the null hypothesis that reference and test come from the same
    distribution by computing the null distribution of Wasserstein distances
    under random permutations.

    H0: P_ref = P_test (no drift)
    H1: P_ref ≠ P_test (drift detected)

    Args:
        reference: Reference distribution samples
        test: Test distribution samples
        observed_distance: Observed Wasserstein distance
        n_permutations: Number of permutations
        alpha: Significance level
        p: Order of Wasserstein distance

    Returns:
        Tuple of (threshold, p_value)
            - threshold: (1-alpha) quantile of null distribution
            - p_value: Fraction of null distances >= observed
    """
    # Pool all samples
    pooled = np.concatenate([reference, test])
    n_ref = len(reference)

    # Compute null distribution
    null_distances = np.zeros(n_permutations)

    for i in range(n_permutations):
        # Random permutation
        np.random.shuffle(pooled)

        # Split into two groups
        ref_perm = pooled[:n_ref]
        test_perm = pooled[n_ref:]

        # Compute distance
        if p == 1:
            null_distances[i] = wasserstein_distance(ref_perm, test_perm)
        else:  # p == 2
            null_distances[i] = _wasserstein_2(ref_perm, test_perm)

    # Compute threshold as (1-alpha) quantile
    threshold = float(np.percentile(null_distances, (1 - alpha) * 100))

    # Compute p-value
    p_value = float(np.mean(null_distances >= observed_distance))

    return threshold, p_value
