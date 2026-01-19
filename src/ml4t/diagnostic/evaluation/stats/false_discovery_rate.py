"""False Discovery Rate (FDR) and Family-Wise Error Rate (FWER) corrections.

This module implements multiple testing corrections:
- Benjamini-Hochberg FDR (1995): Controls expected proportion of false discoveries
- Holm-Bonferroni FWER (1979): Controls probability of any false discovery

These methods are essential when testing multiple hypotheses simultaneously,
which is common in quantitative finance (testing many strategies, factors, etc.).
"""

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Union

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


def benjamini_hochberg_fdr(
    p_values: Sequence[float],
    alpha: float = 0.05,
    return_details: bool = False,
) -> Union["NDArray[Any]", dict[str, Any]]:
    """Apply Benjamini-Hochberg False Discovery Rate correction.

    Controls the False Discovery Rate (FDR) - the expected proportion of false
    discoveries among the rejected hypotheses. More powerful than Bonferroni
    correction for multiple hypothesis testing.

    Based on Benjamini & Hochberg (1995): "Controlling the False Discovery Rate"

    Parameters
    ----------
    p_values : Sequence[float]
        P-values from multiple hypothesis tests
    alpha : float, default 0.05
        Target FDR level (e.g., 0.05 for 5% FDR)
    return_details : bool, default False
        Whether to return detailed information

    Returns
    -------
    Union[NDArray, dict]
        If return_details=False: Boolean array of rejected hypotheses
        If return_details=True: dict with 'rejected', 'adjusted_p_values',
                               'critical_values', 'n_rejected'

    Examples
    --------
    >>> p_values = [0.001, 0.01, 0.03, 0.08, 0.12]
    >>> rejected = benjamini_hochberg_fdr(p_values, alpha=0.05)
    >>> print(f"Rejected: {rejected}")
    Rejected: [ True  True  True False False]
    """
    p_array = np.array(p_values)
    n = len(p_array)

    if n == 0:
        if return_details:
            return {
                "rejected": np.array([], dtype=bool),
                "adjusted_p_values": np.array([]),
                "critical_values": np.array([]),
                "n_rejected": 0,
            }
        return np.array([], dtype=bool)

    # Sort p-values and keep track of original indices
    sorted_indices = np.argsort(p_array)
    sorted_p_values = p_array[sorted_indices]

    # Calculate critical values: (i/n) * alpha
    critical_values = np.arange(1, n + 1) / n * alpha

    # Find largest i such that P(i) <= (i/n) * alpha
    # Work backwards from largest p-value
    rejected_sorted = np.zeros(n, dtype=bool)

    for i in range(n - 1, -1, -1):
        if sorted_p_values[i] <= critical_values[i]:
            # Reject this and all smaller p-values
            rejected_sorted[: i + 1] = True
            break

    # Map back to original order
    rejected = np.zeros(n, dtype=bool)
    rejected[sorted_indices] = rejected_sorted

    if not return_details:
        return rejected

    # Calculate adjusted p-values (step-up method)
    adjusted_p_values = np.zeros(n)
    adjusted_p_values[sorted_indices] = np.minimum.accumulate(
        sorted_p_values[::-1] * n / np.arange(n, 0, -1),
    )[::-1]

    # Ensure adjusted p-values don't exceed 1
    adjusted_p_values = np.minimum(adjusted_p_values, 1.0)

    return {
        "rejected": rejected,
        "adjusted_p_values": adjusted_p_values,
        "critical_values": critical_values[sorted_indices],
        "n_rejected": int(np.sum(rejected)),
    }


def holm_bonferroni(
    p_values: Sequence[float],
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Holm-Bonferroni step-down procedure for FWER control.

    Controls the Family-Wise Error Rate (FWER) - the probability of making
    at least one false discovery. More powerful than Bonferroni correction
    while maintaining strong FWER control.

    Based on Holm (1979): "A Simple Sequentially Rejective Multiple Test Procedure"

    Parameters
    ----------
    p_values : Sequence[float]
        P-values from multiple hypothesis tests
    alpha : float, default 0.05
        Target FWER significance level

    Returns
    -------
    dict
        Dictionary with:
        - rejected: list[bool] - Whether each hypothesis is rejected
        - adjusted_p_values: list[float] - Holm-adjusted p-values
        - n_rejected: int - Number of rejections
        - critical_values: list[float] - Holm critical thresholds

    Notes
    -----
    The Holm procedure is a step-down method:

    1. Sort p-values ascending: p_(1) <= p_(2) <= ... <= p_(m)
    2. For p_(i), compare to alpha / (m - i + 1)
    3. Reject all hypotheses up to (and including) the last rejection
    4. Stop at first non-rejection; accept remaining hypotheses

    This is uniformly more powerful than Bonferroni while controlling FWER.

    Examples
    --------
    >>> p_values = [0.001, 0.01, 0.03, 0.08, 0.12]
    >>> result = holm_bonferroni(p_values, alpha=0.05)
    >>> print(f"Rejected: {result['rejected']}")
    Rejected: [True, True, False, False, False]
    """
    p_array = np.asarray(p_values, dtype=np.float64)
    m = len(p_array)

    if m == 0:
        return {
            "rejected": [],
            "adjusted_p_values": [],
            "n_rejected": 0,
            "critical_values": [],
        }

    # Sort p-values and track original indices
    sorted_indices = np.argsort(p_array)
    sorted_p = p_array[sorted_indices]

    # Holm critical values: alpha / (m - i + 1) for i = 0, 1, ..., m-1
    # i.e., alpha/m, alpha/(m-1), ..., alpha/1
    critical_values = alpha / (m - np.arange(m))

    # Step-down procedure: reject while p_(i) <= critical_(i)
    rejected_sorted = sorted_p <= critical_values

    # Once we fail to reject, accept all remaining
    if not rejected_sorted.all():
        first_fail = np.argmin(rejected_sorted)
        rejected_sorted[first_fail:] = False

    # Map back to original order
    rejected = np.zeros(m, dtype=bool)
    rejected[sorted_indices] = rejected_sorted

    # Compute Holm-adjusted p-values
    # adjusted_p_(i) = max_{j <= i} { (m - j + 1) * p_(j) }
    adjusted_sorted = np.maximum.accumulate(sorted_p * (m - np.arange(m)))
    adjusted_sorted = np.clip(adjusted_sorted, 0.0, 1.0)

    # Map adjusted p-values back to original order
    adjusted_p_values = np.zeros(m)
    adjusted_p_values[sorted_indices] = adjusted_sorted

    # Critical values in original order
    critical_original = np.zeros(m)
    critical_original[sorted_indices] = critical_values

    return {
        "rejected": rejected.tolist(),
        "adjusted_p_values": adjusted_p_values.tolist(),
        "n_rejected": int(rejected.sum()),
        "critical_values": critical_original.tolist(),
    }


def multiple_testing_summary(
    test_results: Sequence[dict[str, Any]],
    method: str = "benjamini_hochberg",
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Summarize results from multiple statistical tests with corrections.

    Provides a comprehensive summary of multiple hypothesis testing results
    with appropriate corrections for multiple comparisons.

    Parameters
    ----------
    test_results : Sequence[dict]
        List of test result dictionaries (each should have 'p_value' key)
    method : str, default "benjamini_hochberg"
        Multiple testing correction method
    alpha : float, default 0.05
        Significance level

    Returns
    -------
    dict
        Summary with original and corrected results

    Examples
    --------
    >>> results = [{'name': 'Strategy A', 'p_value': 0.01},
    ...           {'name': 'Strategy B', 'p_value': 0.08}]
    >>> summary = multiple_testing_summary(results)
    >>> print(f"Significant after correction: {summary['n_significant_corrected']}")
    """
    if not test_results:
        return {
            "n_tests": 0,
            "n_significant_uncorrected": 0,
            "n_significant_corrected": 0,
            "correction_method": method,
            "alpha": alpha,
        }

    # Extract p-values
    p_values = [result.get("p_value", np.nan) for result in test_results]
    valid_p_values = [p for p in p_values if not np.isnan(p)]

    if not valid_p_values:
        return {
            "n_tests": len(test_results),
            "n_significant_uncorrected": 0,
            "n_significant_corrected": 0,
            "correction_method": method,
            "alpha": alpha,
            "warning": "No valid p-values found",
        }

    # Uncorrected significance
    n_significant_uncorrected = sum(p <= alpha for p in valid_p_values)

    # Apply correction
    if method == "benjamini_hochberg":
        correction_result = benjamini_hochberg_fdr(
            valid_p_values,
            alpha=alpha,
            return_details=True,
        )
        n_significant_corrected = correction_result["n_rejected"]
        adjusted_p_values = correction_result["adjusted_p_values"]
        rejected = correction_result["rejected"]
    else:
        raise ValueError(f"Unknown correction method: {method}")

    return {
        "n_tests": len(test_results),
        "n_significant_uncorrected": n_significant_uncorrected,
        "n_significant_corrected": n_significant_corrected,
        "correction_method": method,
        "alpha": alpha,
        "adjusted_p_values": adjusted_p_values.tolist(),
        "rejected_hypotheses": rejected.tolist(),
        "uncorrected_rate": n_significant_uncorrected / len(valid_p_values),
        "corrected_rate": n_significant_corrected / len(valid_p_values),
    }


__all__ = [
    "benjamini_hochberg_fdr",
    "holm_bonferroni",
    "multiple_testing_summary",
]
