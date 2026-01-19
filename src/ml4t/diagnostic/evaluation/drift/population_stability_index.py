"""Population Stability Index (PSI) for distribution drift detection.

PSI measures the distribution shift between a reference dataset (e.g., training)
and a test dataset (e.g., production).

PSI Interpretation:
    - PSI < 0.1: No significant change (green)
    - 0.1 ≤ PSI < 0.2: Small change, monitor (yellow)
    - PSI ≥ 0.2: Significant change, investigate (red)

References:
    - Yurdakul, B. (2018). Statistical Properties of Population Stability Index.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import polars as pl


@dataclass
class PSIResult:
    """Result of Population Stability Index calculation.

    Attributes:
        psi: Overall PSI value (sum of bin-level PSI contributions)
        bin_psi: PSI contribution per bin
        bin_edges: Bin boundaries (continuous) or category labels (categorical)
        reference_counts: Number of samples per bin in reference distribution
        test_counts: Number of samples per bin in test distribution
        reference_percents: Percentage of samples per bin in reference
        test_percents: Percentage of samples per bin in test
        n_bins: Number of bins used
        is_categorical: Whether feature is categorical
        alert_level: Alert level based on PSI thresholds
            - "green": PSI < 0.1 (no significant change)
            - "yellow": 0.1 ≤ PSI < 0.2 (small change, monitor)
            - "red": PSI ≥ 0.2 (significant change, investigate)
        interpretation: Human-readable interpretation
    """

    psi: float
    bin_psi: np.ndarray
    bin_edges: np.ndarray | list[str]
    reference_counts: np.ndarray
    test_counts: np.ndarray
    reference_percents: np.ndarray
    test_percents: np.ndarray
    n_bins: int
    is_categorical: bool
    alert_level: Literal["green", "yellow", "red"]
    interpretation: str

    def summary(self) -> str:
        """Return formatted summary of PSI results."""
        lines = [
            "Population Stability Index (PSI) Report",
            "=" * 50,
            f"PSI Value: {self.psi:.4f}",
            f"Alert Level: {self.alert_level.upper()}",
            f"Feature Type: {'Categorical' if self.is_categorical else 'Continuous'}",
            f"Number of Bins: {self.n_bins}",
            "",
            f"Interpretation: {self.interpretation}",
            "",
            "Bin-Level Analysis:",
            "-" * 50,
        ]

        # Add bin-level details
        for i in range(self.n_bins):
            if self.is_categorical:
                bin_label = self.bin_edges[i]
            else:
                if i == 0:
                    bin_label = f"(-inf, {self.bin_edges[i + 1]:.3f}]"
                elif i == self.n_bins - 1:
                    bin_label = f"({self.bin_edges[i]:.3f}, +inf)"
                else:
                    bin_label = f"({self.bin_edges[i]:.3f}, {self.bin_edges[i + 1]:.3f}]"

            lines.append(
                f"Bin {i + 1:2d} {bin_label:20s}: "
                f"Ref={self.reference_percents[i]:6.2%} "
                f"Test={self.test_percents[i]:6.2%} "
                f"PSI={self.bin_psi[i]:.4f}"
            )

        return "\n".join(lines)


def compute_psi(
    reference: np.ndarray | pl.Series,
    test: np.ndarray | pl.Series,
    n_bins: int = 10,
    is_categorical: bool = False,
    missing_category_handling: Literal["ignore", "separate", "error"] = "separate",
    psi_threshold_yellow: float = 0.1,
    psi_threshold_red: float = 0.2,
) -> PSIResult:
    """Compute Population Stability Index (PSI) between two distributions.

    PSI measures the distribution shift between a reference dataset (e.g., training)
    and a test dataset (e.g., production). It quantifies how much the distribution
    has changed.

    Formula:
        PSI = Σ (test_% - ref_%) × ln(test_% / ref_%)

    For each bin i:
        PSI_i = (P_test[i] - P_ref[i]) × ln(P_test[i] / P_ref[i])

    Args:
        reference: Reference distribution (e.g., training data)
        test: Test distribution (e.g., production data)
        n_bins: Number of quantile bins for continuous features (default: 10)
        is_categorical: Whether feature is categorical (default: False)
        missing_category_handling: How to handle categories in test not in reference:
            - "ignore": Skip missing categories (not recommended)
            - "separate": Create separate bin for missing categories (default)
            - "error": Raise error if new categories found
        psi_threshold_yellow: Threshold for yellow alert (default: 0.1)
        psi_threshold_red: Threshold for red alert (default: 0.2)

    Returns:
        PSIResult with overall PSI, bin-level contributions, and interpretation

    Raises:
        ValueError: If inputs are invalid or missing categories found with "error" handling

    Example:
        >>> # Continuous feature
        >>> ref = np.random.normal(0, 1, 1000)
        >>> test = np.random.normal(0.5, 1, 1000)  # Mean shifted
        >>> result = compute_psi(ref, test, n_bins=10)
        >>> print(result.summary())
        >>>
        >>> # Categorical feature
        >>> ref_cat = np.array(['A', 'B', 'C'] * 100)
        >>> test_cat = np.array(['A', 'A', 'B'] * 100)  # Distribution changed
        >>> result = compute_psi(ref_cat, test_cat, is_categorical=True)
        >>> print(f"PSI: {result.psi:.4f}, Alert: {result.alert_level}")
    """
    # Convert to numpy arrays
    if isinstance(reference, pl.Series):
        reference = reference.to_numpy()
    if isinstance(test, pl.Series):
        test = test.to_numpy()

    reference = np.asarray(reference)
    test = np.asarray(test)

    # Validate inputs
    if len(reference) == 0 or len(test) == 0:
        raise ValueError("Reference and test arrays must not be empty")

    # Variables with union types for both branches
    bin_labels: np.ndarray | list[str]
    bin_edges: np.ndarray | list[str]

    if not is_categorical:
        # Continuous feature: quantile binning
        bin_edges, ref_counts, test_counts = _bin_continuous(reference, test, n_bins)
        bin_labels = bin_edges  # Will be formatted in summary()
    else:
        # Categorical feature: category-based binning
        bin_labels, ref_counts, test_counts = _bin_categorical(
            reference, test, missing_category_handling
        )
        bin_edges = bin_labels
        n_bins = len(bin_labels)

    # Convert counts to percentages
    ref_percents = ref_counts / ref_counts.sum()
    test_percents = test_counts / test_counts.sum()

    # Compute PSI per bin with numerical stability
    # Add small epsilon to avoid log(0) and division by zero
    epsilon = 1e-10
    ref_percents_safe = np.maximum(ref_percents, epsilon)
    test_percents_safe = np.maximum(test_percents, epsilon)

    # PSI formula: (test% - ref%) * ln(test% / ref%)
    bin_psi = (test_percents_safe - ref_percents_safe) * np.log(
        test_percents_safe / ref_percents_safe
    )

    # Total PSI is sum of bin contributions
    psi = float(np.sum(bin_psi))

    # Determine alert level
    alert_level: Literal["green", "yellow", "red"]
    if psi < psi_threshold_yellow:
        alert_level = "green"
        interpretation = (
            f"No significant distribution change detected (PSI={psi:.4f} < {psi_threshold_yellow}). "
            "Feature distribution is stable."
        )
    elif psi < psi_threshold_red:
        alert_level = "yellow"
        interpretation = (
            f"Small distribution change detected ({psi_threshold_yellow} ≤ PSI={psi:.4f} < {psi_threshold_red}). "
            "Monitor feature closely but no immediate action required."
        )
    else:
        alert_level = "red"
        interpretation = (
            f"Significant distribution change detected (PSI={psi:.4f} ≥ {psi_threshold_red}). "
            "Investigate cause and consider model retraining."
        )

    return PSIResult(
        psi=psi,
        bin_psi=bin_psi,
        bin_edges=bin_edges,
        reference_counts=ref_counts,
        test_counts=test_counts,
        reference_percents=ref_percents,
        test_percents=test_percents,
        n_bins=n_bins,
        is_categorical=is_categorical,
        alert_level=alert_level,
        interpretation=interpretation,
    )


def _bin_continuous(
    reference: np.ndarray, test: np.ndarray, n_bins: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bin continuous features using quantiles from reference distribution.

    Uses quantile binning to ensure roughly equal-sized bins in reference distribution.
    Test distribution is binned using same bin edges.

    Args:
        reference: Reference data (used to compute quantiles)
        test: Test data (binned using reference quantiles)
        n_bins: Number of bins

    Returns:
        Tuple of (bin_edges, reference_counts, test_counts)
    """
    # Compute quantiles from reference distribution
    # Use (n_bins + 1) to get n_bins bins with n_bins + 1 edges
    quantiles = np.linspace(0, 100, n_bins + 1)
    bin_edges = np.percentile(reference, quantiles)

    # Ensure edges are unique (handle constant features)
    bin_edges = np.unique(bin_edges)

    # If all values are the same, create a single bin
    if len(bin_edges) == 1:
        return bin_edges, np.array([len(reference)]), np.array([len(test)])

    # Bin both distributions using same edges
    # Use digitize for open-interval binning
    ref_bins = np.digitize(reference, bin_edges[1:-1])
    test_bins = np.digitize(test, bin_edges[1:-1])

    # Count samples per bin
    ref_counts = np.bincount(ref_bins, minlength=len(bin_edges) - 1)
    test_counts = np.bincount(test_bins, minlength=len(bin_edges) - 1)

    return bin_edges, ref_counts, test_counts


def _bin_categorical(
    reference: np.ndarray,
    test: np.ndarray,
    missing_handling: Literal["ignore", "separate", "error"],
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Bin categorical features by category labels.

    Args:
        reference: Reference categories
        test: Test categories
        missing_handling: How to handle new categories in test

    Returns:
        Tuple of (category_labels, reference_counts, test_counts)

    Raises:
        ValueError: If new categories found and missing_handling="error"
    """
    # Get unique categories from reference
    ref_categories = sorted(set(reference))
    test_categories = set(test)

    # Check for new categories in test
    new_categories = test_categories - set(ref_categories)

    if new_categories:
        if missing_handling == "error":
            raise ValueError(
                f"New categories found in test set: {new_categories}. "
                "These categories were not present in reference distribution."
            )
        elif missing_handling == "separate":
            # Add new categories to the end
            ref_categories.extend(sorted(new_categories))
        # else "ignore": new categories will be dropped

    # Count occurrences per category
    ref_counts = np.array([np.sum(reference == cat) for cat in ref_categories])
    test_counts = np.array([np.sum(test == cat) for cat in ref_categories])

    return ref_categories, ref_counts, test_counts
