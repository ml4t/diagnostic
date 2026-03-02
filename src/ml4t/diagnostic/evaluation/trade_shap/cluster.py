"""Hierarchical clustering for trade error patterns.

Provides clustering of SHAP vectors to identify distinct error patterns,
with proper handling of small sample sizes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from ml4t.diagnostic.config import TradeClusteringSettings
from ml4t.diagnostic.evaluation.trade_shap.models import ClusteringResult

if TYPE_CHECKING:
    from numpy.typing import NDArray


DistanceMetric = Literal["euclidean", "cosine", "correlation", "cityblock"]
LinkageMethod = Literal["ward", "average", "complete", "single"]


def _to_distance_metric(value: Any) -> DistanceMetric:
    metric = str(getattr(value, "value", value))
    if metric == "manhattan":
        return "cityblock"
    return metric  # type: ignore[return-value]


def _to_linkage_method(value: Any) -> LinkageMethod:
    return str(getattr(value, "value", value))  # type: ignore[return-value]


def find_optimal_clusters(
    linkage_matrix: NDArray[np.floating[Any]],
    n_samples: int,
    min_cluster_size: int = 5,
) -> int:
    """Find optimal number of clusters using elbow method.

    Uses the acceleration of merge distances (second derivative) to find
    the "elbow" point in the dendrogram.

    Args:
        linkage_matrix: Linkage matrix from hierarchical clustering
        n_samples: Total number of samples
        min_cluster_size: Minimum samples per cluster

    Returns:
        Optimal number of clusters respecting min_cluster_size constraint

    Note:
        The key fix here is respecting min_cluster_size even when that means
        returning 1 cluster. Previously, the code would force 2 clusters even
        when there weren't enough samples to support min_cluster_size per cluster.
    """
    # Get merge distances (last column of linkage matrix)
    distances = linkage_matrix[:, 2]

    # Compute first derivative (rate of change)
    first_deriv = np.diff(distances)

    # Compute second derivative (acceleration)
    second_deriv = np.diff(first_deriv)

    # Find elbow: Maximum acceleration point
    if len(second_deriv) > 0:
        elbow_idx = int(np.argmax(second_deriv))
        # Convert index to number of clusters
        # linkage_matrix has (n_samples - 1) rows
        n_clusters = max(1, n_samples - elbow_idx - 2)
    else:
        # Fallback: sqrt(n) heuristic
        n_clusters = max(1, int(np.sqrt(n_samples)))

    # CRITICAL FIX: Respect min_cluster_size constraint
    # max_clusters is at least 1 to avoid edge case where we'd return 0
    max_clusters = max(1, n_samples // min_cluster_size)
    n_clusters = min(n_clusters, max_clusters)

    # Only force at least 2 clusters if we have room for them
    # This is the bug fix: don't force 2 if max_clusters < 2
    if max_clusters >= 2:
        n_clusters = max(2, n_clusters)

    return int(n_clusters)


def compute_cluster_sizes(
    labels: NDArray[np.intp] | list[int],
    n_clusters: int,
) -> list[int]:
    """Compute number of samples in each cluster using vectorized bincount.

    Args:
        labels: Cluster assignment for each sample (0-indexed)
        n_clusters: Total number of clusters

    Returns:
        List of cluster sizes
    """
    labels_array = np.asarray(labels, dtype=np.intp)
    counts = np.bincount(labels_array, minlength=n_clusters)
    return counts.tolist()


def compute_centroids(
    vectors: NDArray[np.floating[Any]],
    labels: NDArray[np.intp] | list[int],
    n_clusters: int,
) -> NDArray[np.floating[Any]]:
    """Compute cluster centroids (mean vector per cluster) using vectorized operations.

    Args:
        vectors: SHAP vectors of shape (n_samples, n_features)
        labels: Cluster assignment for each sample (0-indexed)
        n_clusters: Total number of clusters

    Returns:
        Centroids of shape (n_clusters, n_features)
    """
    labels_array = np.asarray(labels, dtype=np.intp)
    n_features = vectors.shape[1]

    centroids = np.zeros((n_clusters, n_features), dtype=np.float64)

    for k in range(n_clusters):
        mask = labels_array == k
        if np.any(mask):
            centroids[k] = vectors[mask].mean(axis=0)

    return centroids


class HierarchicalClusterer:
    """Hierarchical clustering for SHAP vectors.

    Provides clustering of trade SHAP vectors to identify distinct error patterns,
    with quality metrics and dendrogram support.

    Attributes:
        config: Clustering configuration

    Example:
        >>> clusterer = HierarchicalClusterer()
        >>> result = clusterer.cluster(shap_vectors, n_clusters=3)
        >>> print(f"Silhouette: {result.silhouette_score:.3f}")
    """

    def __init__(
        self,
        config: TradeClusteringSettings | None = None,
        *,
        min_trades_for_clustering: int = 10,
    ) -> None:
        """Initialize clusterer.

        Args:
            config: Trade clustering settings (uses defaults if None)
            min_trades_for_clustering: Minimum trades required to run clustering
        """
        self.config = config or TradeClusteringSettings()
        self.min_trades_for_clustering = min_trades_for_clustering

    def cluster(
        self,
        vectors: NDArray[np.floating[Any]],
        n_clusters: int | None = None,
    ) -> ClusteringResult:
        """Cluster SHAP vectors using hierarchical clustering.

        Args:
            vectors: SHAP vectors of shape (n_samples, n_features)
            n_clusters: Number of clusters (auto-determined if None)

        Returns:
            ClusteringResult with assignments, linkage matrix, and quality metrics

        Raises:
            ValueError: If insufficient samples or invalid input shape
            ImportError: If scipy is not installed
        """
        # Validate inputs
        if vectors.size == 0:
            raise ValueError("Cannot cluster empty vectors")

        if vectors.ndim != 2:
            raise ValueError(
                f"vectors must be 2D array (n_samples, n_features), got shape {vectors.shape}"
            )

        n_samples, n_features = vectors.shape

        if n_samples < self.min_trades_for_clustering:
            raise ValueError(
                f"Insufficient samples for clustering: {n_samples} < "
                f"{self.min_trades_for_clustering}"
            )

        # Import scipy
        try:
            import scipy.cluster.hierarchy as sch
            from scipy.spatial.distance import pdist
        except ImportError as e:
            raise ImportError(
                "scipy required for clustering. Install with: pip install scipy"
            ) from e

        # Compute pairwise distances
        distance_metric = _to_distance_metric(self.config.distance_metric)
        linkage_method = _to_linkage_method(self.config.linkage)
        distances = pdist(vectors, metric=distance_metric)

        # Perform hierarchical clustering
        linkage_matrix = sch.linkage(distances, method=linkage_method)

        # Determine number of clusters
        if n_clusters is None:
            n_clusters = find_optimal_clusters(
                linkage_matrix, n_samples, self.config.min_cluster_size
            )

        # Cut dendrogram to get cluster assignments
        labels = sch.fcluster(linkage_matrix, t=n_clusters, criterion="maxclust")
        # fcluster returns 1-indexed labels, convert to 0-indexed
        labels = labels - 1

        # Compute cluster metrics
        cluster_sizes = compute_cluster_sizes(labels, n_clusters)
        centroids = compute_centroids(vectors, labels, n_clusters)

        # Compute quality metrics
        silhouette = self._compute_silhouette(vectors, labels)
        davies_bouldin = self._compute_davies_bouldin(vectors, labels)
        calinski_harabasz = self._compute_calinski_harabasz(vectors, labels)

        return ClusteringResult(
            n_clusters=n_clusters,
            cluster_assignments=labels.tolist(),
            linkage_matrix=linkage_matrix,
            centroids=centroids,
            silhouette_score=silhouette,
            davies_bouldin_score=davies_bouldin,
            calinski_harabasz_score=calinski_harabasz,
            cluster_sizes=cluster_sizes,
            distance_metric=distance_metric,
            linkage_method=linkage_method,
        )

    def _compute_silhouette(
        self,
        vectors: NDArray[np.floating[Any]],
        labels: NDArray[np.intp],
    ) -> float:
        """Compute silhouette score for clustering quality.

        Returns:
            Silhouette score (-1 to 1, higher is better)
        """
        try:
            from sklearn.metrics import silhouette_score

            # Need at least 2 clusters for silhouette
            unique_labels = np.unique(labels)
            if len(unique_labels) < 2:
                return 0.0

            return float(silhouette_score(vectors, labels))
        except ImportError:
            return 0.0

    def _compute_davies_bouldin(
        self,
        vectors: NDArray[np.floating[Any]],
        labels: NDArray[np.intp],
    ) -> float | None:
        """Compute Davies-Bouldin index (lower is better)."""
        try:
            from sklearn.metrics import davies_bouldin_score

            unique_labels = np.unique(labels)
            if len(unique_labels) < 2:
                return None

            return float(davies_bouldin_score(vectors, labels))
        except ImportError:
            return None

    def _compute_calinski_harabasz(
        self,
        vectors: NDArray[np.floating[Any]],
        labels: NDArray[np.intp],
    ) -> float | None:
        """Compute Calinski-Harabasz score (higher is better)."""
        try:
            from sklearn.metrics import calinski_harabasz_score

            unique_labels = np.unique(labels)
            if len(unique_labels) < 2:
                return None

            return float(calinski_harabasz_score(vectors, labels))
        except ImportError:
            return None
