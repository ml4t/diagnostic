"""Pattern characterization with proper statistical testing.

This module provides PatternCharacterizer for characterizing error patterns
identified through clustering, with:
- Welch's t-test (doesn't assume equal variance)
- Mann-Whitney U test (non-parametric)
- Benjamini-Hochberg FDR correction for multiple testing
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy import stats

from ml4t.diagnostic.evaluation.trade_shap.models import ErrorPattern

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass
class CharacterizationConfig:
    """Configuration for pattern characterization.

    Attributes:
        alpha: Significance level for statistical tests (default: 0.05)
        top_n_features: Number of top features to include in characterization
        use_fdr_correction: Whether to apply Benjamini-Hochberg FDR correction
        min_samples_per_test: Minimum samples needed for each group in t-test
    """

    alpha: float = 0.05
    top_n_features: int = 5
    use_fdr_correction: bool = True
    min_samples_per_test: int = 3


@dataclass
class FeatureStatistics:
    """Statistical test results for a single feature.

    Attributes:
        feature_name: Name of the feature
        mean_shap: Mean SHAP value in the cluster
        mean_shap_other: Mean SHAP value in other clusters
        p_value_t: P-value from Welch's t-test
        p_value_mw: P-value from Mann-Whitney U test
        q_value_t: FDR-corrected p-value (t-test), if correction applied
        q_value_mw: FDR-corrected p-value (MW test), if correction applied
        is_significant: Whether the feature is statistically significant
    """

    feature_name: str
    mean_shap: float
    mean_shap_other: float
    p_value_t: float
    p_value_mw: float
    q_value_t: float | None = None
    q_value_mw: float | None = None
    is_significant: bool = False


def benjamini_hochberg(
    p_values: list[float], alpha: float = 0.05
) -> tuple[list[float], list[bool]]:
    """Apply Benjamini-Hochberg FDR correction to p-values.

    Args:
        p_values: List of raw p-values
        alpha: Significance level (default: 0.05)

    Returns:
        Tuple of (q_values, is_significant) where:
        - q_values: FDR-adjusted p-values (monotone)
        - is_significant: Boolean mask for significant results

    Note:
        BH procedure controls False Discovery Rate (FDR) - the expected
        proportion of false discoveries among rejected hypotheses.
        This is less conservative than Bonferroni correction.
    """
    if not p_values:
        return [], []

    n = len(p_values)
    p_array = np.asarray(p_values)

    # Sort p-values and track original order
    sorted_indices = np.argsort(p_array)
    sorted_p = p_array[sorted_indices]

    # BH adjustment: q_i = min(p_i * n / rank, 1.0)
    # Then enforce monotonicity from largest to smallest
    ranks = np.arange(1, n + 1)
    q_sorted = np.minimum(sorted_p * n / ranks, 1.0)

    # Enforce monotonicity: q[i] = min(q[i], q[i+1], ..., q[n])
    # Process from end to start
    for i in range(n - 2, -1, -1):
        q_sorted[i] = min(q_sorted[i], q_sorted[i + 1])

    # Restore original order
    q_values = np.empty(n)
    q_values[sorted_indices] = q_sorted

    # Determine significance
    is_significant = q_values < alpha

    return q_values.tolist(), is_significant.tolist()


class PatternCharacterizer:
    """Characterizes error patterns with proper statistical testing.

    Uses Welch's t-test (doesn't assume equal variance) and Mann-Whitney U test,
    with optional Benjamini-Hochberg FDR correction for multiple testing.

    Attributes:
        config: Characterization configuration
        feature_names: List of all feature names

    Example:
        >>> characterizer = PatternCharacterizer(feature_names)
        >>> pattern = characterizer.characterize_cluster(
        ...     cluster_shap=cluster_vectors,
        ...     other_shap=other_vectors,
        ...     cluster_id=0,
        ... )
        >>> print(pattern.top_features)
    """

    def __init__(
        self,
        feature_names: list[str],
        config: CharacterizationConfig | None = None,
    ) -> None:
        """Initialize characterizer.

        Args:
            feature_names: List of all feature names
            config: Characterization configuration (uses defaults if None)
        """
        self.feature_names = feature_names
        self.config = config or CharacterizationConfig()

    def characterize_cluster(
        self,
        cluster_shap: NDArray[np.floating[Any]],
        other_shap: NDArray[np.floating[Any]],
        cluster_id: int,
        centroids: NDArray[np.floating[Any]] | None = None,
    ) -> ErrorPattern:
        """Characterize a single cluster as an error pattern.

        Args:
            cluster_shap: SHAP vectors for trades in this cluster (n_cluster x n_features)
            other_shap: SHAP vectors for all other trades (n_other x n_features)
            cluster_id: Cluster identifier (0-indexed)
            centroids: Optional cluster centroids for separation score calculation

        Returns:
            ErrorPattern with statistical characterization
        """
        n_trades = cluster_shap.shape[0]
        n_features = len(self.feature_names)

        # Compute mean SHAP per feature for this cluster
        mean_shap_cluster = np.mean(cluster_shap, axis=0)
        mean_shap_other = (
            np.mean(other_shap, axis=0) if len(other_shap) > 0 else np.zeros(n_features)
        )

        # Statistical tests for each feature
        feature_stats = self._compute_feature_statistics(
            cluster_shap, other_shap, mean_shap_cluster, mean_shap_other
        )

        # Apply FDR correction if configured
        if self.config.use_fdr_correction:
            feature_stats = self._apply_fdr_correction(feature_stats)

        # Sort by absolute mean SHAP (descending)
        feature_stats.sort(key=lambda x: abs(x.mean_shap), reverse=True)

        # Take top N
        top_stats = feature_stats[: self.config.top_n_features]

        # Build top_features tuple list for ErrorPattern
        top_features = [
            (
                fs.feature_name,
                fs.mean_shap,
                fs.p_value_t,
                fs.p_value_mw,
                fs.is_significant,
            )
            for fs in top_stats
        ]

        # Generate pattern description
        description = self._generate_description(top_stats)

        # Compute separation and distinctiveness scores
        separation_score = self._compute_separation_score(mean_shap_cluster, centroids, cluster_id)
        distinctiveness = self._compute_distinctiveness(mean_shap_cluster, mean_shap_other)

        return ErrorPattern(
            cluster_id=cluster_id,
            n_trades=n_trades,
            description=description,
            top_features=top_features,
            separation_score=separation_score,
            distinctiveness=distinctiveness,
        )

    def _compute_feature_statistics(
        self,
        cluster_shap: NDArray[np.floating[Any]],
        other_shap: NDArray[np.floating[Any]],
        mean_shap_cluster: NDArray[np.floating[Any]],
        mean_shap_other: NDArray[np.floating[Any]],
    ) -> list[FeatureStatistics]:
        """Compute statistical tests for each feature.

        Uses Welch's t-test (equal_var=False) instead of standard t-test
        to handle unequal variances between groups.
        """
        results = []

        for idx, feature_name in enumerate(self.feature_names):
            cluster_values = cluster_shap[:, idx]
            other_values = other_shap[:, idx] if len(other_shap) > 0 else np.array([])

            # Skip if insufficient samples
            if (
                len(cluster_values) < self.config.min_samples_per_test
                or len(other_values) < self.config.min_samples_per_test
            ):
                results.append(
                    FeatureStatistics(
                        feature_name=feature_name,
                        mean_shap=float(mean_shap_cluster[idx]),
                        mean_shap_other=float(mean_shap_other[idx]),
                        p_value_t=1.0,
                        p_value_mw=1.0,
                        is_significant=False,
                    )
                )
                continue

            # Welch's t-test (doesn't assume equal variance)
            # This is the key fix: using equal_var=False
            try:
                t_stat, p_value_t = stats.ttest_ind(cluster_values, other_values, equal_var=False)
                p_value_t = float(p_value_t) if not np.isnan(p_value_t) else 1.0
            except Exception:
                p_value_t = 1.0

            # Mann-Whitney U test (non-parametric)
            try:
                _, p_value_mw = stats.mannwhitneyu(
                    cluster_values, other_values, alternative="two-sided"
                )
                p_value_mw = float(p_value_mw) if not np.isnan(p_value_mw) else 1.0
            except ValueError:
                # Can fail if all values are identical
                p_value_mw = 1.0

            results.append(
                FeatureStatistics(
                    feature_name=feature_name,
                    mean_shap=float(mean_shap_cluster[idx]),
                    mean_shap_other=float(mean_shap_other[idx]),
                    p_value_t=p_value_t,
                    p_value_mw=p_value_mw,
                    # Will be set after FDR correction
                    is_significant=False,
                )
            )

        return results

    def _apply_fdr_correction(
        self, feature_stats: list[FeatureStatistics]
    ) -> list[FeatureStatistics]:
        """Apply Benjamini-Hochberg FDR correction to all p-values.

        This corrects for multiple testing across all features, reducing
        false positive rate at the cost of some statistical power.
        """
        if not feature_stats:
            return feature_stats

        # Collect p-values
        p_values_t = [fs.p_value_t for fs in feature_stats]
        p_values_mw = [fs.p_value_mw for fs in feature_stats]

        # Apply BH correction
        q_values_t, sig_t = benjamini_hochberg(p_values_t, self.config.alpha)
        q_values_mw, sig_mw = benjamini_hochberg(p_values_mw, self.config.alpha)

        # Update statistics with corrected values
        corrected = []
        for i, fs in enumerate(feature_stats):
            # Significant if either test rejects after FDR correction
            is_sig = sig_t[i] or sig_mw[i]

            corrected.append(
                FeatureStatistics(
                    feature_name=fs.feature_name,
                    mean_shap=fs.mean_shap,
                    mean_shap_other=fs.mean_shap_other,
                    p_value_t=fs.p_value_t,
                    p_value_mw=fs.p_value_mw,
                    q_value_t=q_values_t[i],
                    q_value_mw=q_values_mw[i],
                    is_significant=is_sig,
                )
            )

        return corrected

    def _generate_description(self, top_stats: list[FeatureStatistics]) -> str:
        """Generate human-readable pattern description."""
        if not top_stats:
            return "Unknown pattern"

        # Filter to significant features only
        sig_features = [fs for fs in top_stats if fs.is_significant]

        # Fall back to top features if none significant
        features_to_use = sig_features[:3] if sig_features else top_stats[:2]

        components = []
        for fs in features_to_use:
            direction = "High" if fs.mean_shap > 0 else "Low"
            arrow = "↑" if fs.mean_shap > 0 else "↓"
            components.append(f"{direction} {fs.feature_name} ({arrow}{fs.mean_shap:.2f})")

        if len(components) == 1:
            return f"{components[0]} → Losses"
        return " + ".join(components) + " → Losses"

    def _compute_separation_score(
        self,
        centroid: NDArray[np.floating[Any]],
        all_centroids: NDArray[np.floating[Any]] | None,
        cluster_id: int,
    ) -> float:
        """Compute separation score (distance to nearest other cluster)."""
        if all_centroids is None or len(all_centroids) <= 1:
            return 0.0

        min_distance = float("inf")
        for i, other_centroid in enumerate(all_centroids):
            if i != cluster_id:
                distance = float(np.linalg.norm(centroid - other_centroid))
                min_distance = min(min_distance, distance)

        return min_distance if min_distance != float("inf") else 0.0

    def _compute_distinctiveness(
        self,
        cluster_centroid: NDArray[np.floating[Any]],
        other_mean: NDArray[np.floating[Any]],
    ) -> float:
        """Compute distinctiveness (ratio of max SHAP vs other clusters)."""
        max_cluster = np.max(np.abs(cluster_centroid))
        max_other = np.max(np.abs(other_mean))

        if max_other == 0:
            return float(max_cluster) if max_cluster > 0 else 1.0

        return float(max_cluster / max_other)

    def characterize_all_clusters(
        self,
        shap_vectors: NDArray[np.floating[Any]],
        cluster_labels: list[int],
        n_clusters: int,
        centroids: NDArray[np.floating[Any]] | None = None,
    ) -> list[ErrorPattern]:
        """Characterize all clusters.

        Args:
            shap_vectors: All SHAP vectors (n_samples x n_features)
            cluster_labels: Cluster assignment for each sample
            n_clusters: Total number of clusters
            centroids: Optional cluster centroids

        Returns:
            List of ErrorPattern for each cluster
        """
        labels_array = np.asarray(cluster_labels)
        patterns = []

        for cluster_id in range(n_clusters):
            mask = labels_array == cluster_id
            cluster_shap = shap_vectors[mask]
            other_shap = shap_vectors[~mask]

            pattern = self.characterize_cluster(
                cluster_shap=cluster_shap,
                other_shap=other_shap,
                cluster_id=cluster_id,
                centroids=centroids,
            )
            patterns.append(pattern)

        return patterns
