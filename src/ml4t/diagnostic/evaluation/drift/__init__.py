"""Distribution drift detection for feature monitoring.

This module provides comprehensive drift detection with three complementary methods
and a unified analysis interface:

**Individual Methods**:
- **PSI (Population Stability Index)**: Bin-based distribution comparison
- **Wasserstein Distance**: Optimal transport metric for continuous features
- **Domain Classifier**: ML-based multivariate drift detection with feature importance

**Unified Interface**:
- **analyze_drift()**: Multi-method drift analysis with consensus-based flagging

Distribution drift is critical for ML model monitoring:
- Feature distributions change over time (concept drift)
- Model performance degrades when test distribution differs from training
- Early detection allows proactive model retraining
- Multi-method consensus increases confidence in drift detection

PSI Interpretation:
    - PSI < 0.1: No significant change (green)
    - 0.1 ≤ PSI < 0.2: Small change, monitor (yellow)
    - PSI ≥ 0.2: Significant change, investigate (red)

Wasserstein Distance Interpretation:
    - W = 0: Identical distributions
    - W > 0: Distribution drift detected
    - Larger values indicate greater drift magnitude
    - Threshold calibrated via permutation testing

Domain Classifier Interpretation:
    - AUC ≈ 0.5: No drift (random guess between reference and test)
    - AUC = 0.6: Weak drift
    - AUC = 0.7-0.8: Moderate drift
    - AUC > 0.9: Strong drift
    - Feature importance identifies which features drifted

When to Use:
    - **PSI**: Categorical features or when binning is acceptable
    - **Wasserstein**: Continuous features, more sensitive to small shifts
    - **Domain Classifier**: Multivariate drift, interaction detection
    - **analyze_drift()**: Comprehensive analysis with multiple methods
    - Model monitoring: Compare production data to training data
    - Temporal drift: Compare recent data to historical baseline
    - Segmentation drift: Compare distributions across segments

References:
    - Yurdakul, B. (2018). Statistical Properties of Population Stability Index.
      https://scholarship.richmond.edu/honors-theses/1131/
    - Webb, G. I., et al. (2016). Characterizing concept drift.
      Data Mining and Knowledge Discovery, 30(4), 964-994.
    - Villani, C. (2009). Optimal Transport: Old and New. Springer.
    - Ramdas, A., et al. (2017). On Wasserstein Two-Sample Testing and Related
      Families of Nonparametric Tests. Entropy, 19(2), 47.
    - Lopez-Paz, D., & Oquab, M. (2017). Revisiting Classifier Two-Sample Tests.
      ICLR 2017.
    - Rabanser, S., et al. (2019). Failing Loudly: An Empirical Study of Methods
      for Detecting Dataset Shift. NeurIPS 2019.

Example - Individual Methods:
    >>> import numpy as np
    >>> from ml4t.diagnostic.evaluation.drift import (
    ...     compute_psi, compute_wasserstein_distance, compute_domain_classifier_drift
    ... )
    >>>
    >>> # PSI for univariate drift
    >>> reference = np.random.normal(0, 1, 1000)
    >>> test = np.random.normal(0.5, 1, 1000)  # Mean shifted
    >>> psi_result = compute_psi(reference, test, n_bins=10)
    >>> print(f"PSI: {psi_result.psi:.4f}, Alert: {psi_result.alert_level}")
    >>>
    >>> # Wasserstein for continuous features
    >>> ws_result = compute_wasserstein_distance(reference, test)
    >>> print(f"Wasserstein: {ws_result.distance:.4f}, Drifted: {ws_result.drifted}")

Example - Unified Analysis:
    >>> import pandas as pd
    >>> from ml4t.diagnostic.evaluation.drift import analyze_drift
    >>>
    >>> # Create reference and test datasets
    >>> reference = pd.DataFrame({
    ...     'feature1': np.random.normal(0, 1, 1000),
    ...     'feature2': np.random.normal(0, 1, 1000),
    ... })
    >>> test = pd.DataFrame({
    ...     'feature1': np.random.normal(0.5, 1, 1000),  # Drifted
    ...     'feature2': np.random.normal(0, 1, 1000),    # Stable
    ... })
    >>>
    >>> # Comprehensive drift analysis with all methods
    >>> result = analyze_drift(reference, test)
    >>> print(result.summary())
    >>> print(f"Drifted features: {result.drifted_features}")
    >>>
    >>> # Get detailed results as DataFrame
    >>> df = result.to_dataframe()
    >>> print(df)
    >>>
    >>> # Use specific methods only
    >>> result = analyze_drift(reference, test, methods=['psi', 'wasserstein'])
    >>>
    >>> # Customize consensus threshold (default: 0.5)
    >>> result = analyze_drift(reference, test, consensus_threshold=0.66)
"""

# Import from submodules and re-export
from ml4t.diagnostic.evaluation.drift.analysis import (
    DriftSummaryResult,
    FeatureDriftResult,
    analyze_drift,
)
from ml4t.diagnostic.evaluation.drift.domain_classifier import (
    DomainClassifierResult,
    compute_domain_classifier_drift,
)
from ml4t.diagnostic.evaluation.drift.population_stability_index import (
    PSIResult,
    compute_psi,
)
from ml4t.diagnostic.evaluation.drift.wasserstein import (
    WassersteinResult,
    compute_wasserstein_distance,
)

__all__ = [
    # PSI
    "compute_psi",
    "PSIResult",
    # Wasserstein
    "compute_wasserstein_distance",
    "WassersteinResult",
    # Domain Classifier
    "compute_domain_classifier_drift",
    "DomainClassifierResult",
    # Unified analysis
    "analyze_drift",
    "FeatureDriftResult",
    "DriftSummaryResult",
]
