"""Monotonicity: Test monotonic relationship between feature values and outcomes.

Monotonicity is a key property for predictive features - we expect higher
(or lower) feature values to consistently correspond to higher outcomes.
"""

from typing import TYPE_CHECKING, Any, Union

import numpy as np
import pandas as pd
import polars as pl
from scipy import stats
from scipy.stats import spearmanr

if TYPE_CHECKING:
    from numpy.typing import NDArray


def compute_monotonicity(
    features: Union[pl.DataFrame, pd.DataFrame, "NDArray[Any]"],
    outcomes: Union[pl.DataFrame, pd.DataFrame, "NDArray[Any]"],
    n_quantiles: int = 5,
    feature_col: str | None = None,
    outcome_col: str | None = None,
    method: str = "spearman",
) -> dict[str, Any]:
    """Test monotonic relationship between feature values and outcomes.

    Monotonicity is a key property for predictive features - we expect higher
    (or lower) feature values to consistently correspond to higher outcomes.
    Non-monotonic relationships often indicate:
    1. Feature needs transformation (e.g., absolute value, log)
    2. Feature has regime-dependent behavior
    3. Feature is not truly predictive

    This function bins features into quantiles and checks if mean outcomes
    increase/decrease monotonically across bins.

    Parameters
    ----------
    features : Union[pl.DataFrame, pd.DataFrame, np.ndarray]
        Feature values to test
    outcomes : Union[pl.DataFrame, pd.DataFrame, np.ndarray]
        Outcome values (typically returns)
    n_quantiles : int, default 5
        Number of quantile bins (5 = quintiles, 10 = deciles)
    feature_col : str | None, default None
        Column name for features (if DataFrame)
    outcome_col : str | None, default None
        Column name for outcomes (if DataFrame)
    method : str, default "spearman"
        Correlation method: "spearman" or "pearson"

    Returns
    -------
    dict[str, Any]
        Dictionary with monotonicity analysis:
        - correlation: Spearman/Pearson correlation
        - p_value: Statistical significance of correlation
        - quantile_means: Mean outcome per quantile
        - quantile_labels: Quantile labels (Q1, Q2, ...)
        - is_monotonic: Boolean, True if strictly monotonic
        - monotonicity_score: Fraction of quantile pairs that are monotonic (0-1)
        - direction: "increasing", "decreasing", or "non-monotonic"
        - n_observations: Total observations
        - n_per_quantile: Observations per quantile

    Examples
    --------
    >>> # Test if momentum predicts returns
    >>> features = df['momentum']
    >>> outcomes = df['forward_return']
    >>> result = compute_monotonicity(features, outcomes, n_quantiles=5)
    >>>
    >>> print(f"Correlation: {result['correlation']:.3f}")
    >>> print(f"P-value: {result['p_value']:.4f}")
    >>> print(f"Monotonic: {result['is_monotonic']}")
    >>> print(f"Direction: {result['direction']}")
    >>> print(f"Quantile means: {result['quantile_means']}")
    Correlation: 0.156
    P-value: 0.0001
    Monotonic: True
    Direction: increasing
    Quantile means: [-0.002, 0.001, 0.003, 0.005, 0.008]

    Notes
    -----
    Monotonicity Score:
    - 1.0: Perfect monotonicity (all adjacent quantiles ordered correctly)
    - 0.8-1.0: Strong monotonicity (minor violations)
    - 0.6-0.8: Moderate monotonicity
    - <0.6: Weak or no monotonicity

    Common Patterns:
    - Monotonic increasing: Good positive predictor
    - Monotonic decreasing: Good negative predictor (consider sign flip)
    - U-shaped: Consider absolute value or squared feature
    - Flat: Feature not predictive

    References
    ----------
    .. [1] Kakushadze, Z., & Serur, J. A. (2018). "151 Trading Strategies."
    """
    # Extract feature and outcome arrays
    feature_vals: NDArray[Any]
    if isinstance(features, pl.DataFrame):
        if feature_col is None:
            raise ValueError("feature_col must be specified for DataFrame input")
        feature_vals = features[feature_col].to_numpy()
    elif isinstance(features, pd.DataFrame):
        if feature_col is None:
            raise ValueError("feature_col must be specified for DataFrame input")
        feature_vals = features[feature_col].to_numpy()
    else:
        feature_vals = np.asarray(features).flatten()

    outcome_vals: NDArray[Any]
    if isinstance(outcomes, pl.DataFrame):
        if outcome_col is None:
            raise ValueError("outcome_col must be specified for DataFrame input")
        outcome_vals = outcomes[outcome_col].to_numpy()
    elif isinstance(outcomes, pd.DataFrame):
        if outcome_col is None:
            raise ValueError("outcome_col must be specified for DataFrame input")
        outcome_vals = outcomes[outcome_col].to_numpy()
    else:
        outcome_vals = np.asarray(outcomes).flatten()

    # Validate inputs
    if len(feature_vals) != len(outcome_vals):
        raise ValueError(
            f"Features ({len(feature_vals)}) and outcomes ({len(outcome_vals)}) must have same length"
        )

    # Remove NaN values
    valid_mask = ~(np.isnan(feature_vals.astype(float)) | np.isnan(outcome_vals.astype(float)))
    feature_clean = feature_vals[valid_mask]
    outcome_clean = outcome_vals[valid_mask]

    n = len(feature_clean)
    if n < n_quantiles * 2:
        # Insufficient data for quantile analysis
        return {
            "correlation": np.nan,
            "p_value": np.nan,
            "quantile_means": [],
            "quantile_labels": [],
            "is_monotonic": False,
            "monotonicity_score": 0.0,
            "direction": "insufficient_data",
            "n_observations": n,
            "n_per_quantile": [],
        }

    # Compute correlation
    if method == "spearman":
        correlation, p_value = spearmanr(feature_clean, outcome_clean)
    elif method == "pearson":
        correlation, p_value = stats.pearsonr(feature_clean, outcome_clean)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'spearman' or 'pearson'.")

    # Create quantile bins
    quantile_edges = np.linspace(0, 100, n_quantiles + 1)
    quantile_bins = np.percentile(feature_clean, quantile_edges)

    # Assign observations to quantiles
    quantile_assignments = np.digitize(feature_clean, quantile_bins[1:-1])  # 0-indexed bins

    # Compute mean outcome per quantile
    quantile_means = []
    n_per_quantile = []

    for q in range(n_quantiles):
        mask = quantile_assignments == q
        if np.sum(mask) > 0:
            quantile_means.append(float(np.mean(outcome_clean[mask])))
            n_per_quantile.append(int(np.sum(mask)))
        else:
            quantile_means.append(np.nan)
            n_per_quantile.append(0)

    # Check monotonicity
    # Count how many adjacent pairs are ordered correctly
    monotonic_pairs = 0
    total_pairs = 0

    for i in range(len(quantile_means) - 1):
        if not (np.isnan(quantile_means[i]) or np.isnan(quantile_means[i + 1])):
            total_pairs += 1
            # Check if ordered (either increasing or decreasing)
            if correlation > 0:
                # Expect increasing
                if quantile_means[i + 1] > quantile_means[i]:
                    monotonic_pairs += 1
            # Expect decreasing
            elif quantile_means[i + 1] < quantile_means[i]:
                monotonic_pairs += 1

    monotonicity_score = monotonic_pairs / total_pairs if total_pairs > 0 else 0.0

    # Strict monotonicity check (all pairs ordered correctly)
    is_monotonic = monotonicity_score == 1.0

    # Determine direction
    if is_monotonic:
        direction = "increasing" if correlation > 0 else "decreasing"
    elif monotonicity_score >= 0.8:
        direction = "mostly_" + ("increasing" if correlation > 0 else "decreasing")
    else:
        direction = "non_monotonic"

    # Create quantile labels
    quantile_labels = [f"Q{i + 1}" for i in range(n_quantiles)]

    return {
        "correlation": float(correlation),
        "p_value": float(p_value),
        "quantile_means": quantile_means,
        "quantile_labels": quantile_labels,
        "is_monotonic": is_monotonic,
        "monotonicity_score": float(monotonicity_score),
        "direction": direction,
        "n_observations": n,
        "n_per_quantile": n_per_quantile,
    }
