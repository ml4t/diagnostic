"""Signal selection algorithms for multi-signal comparison.

This module provides intelligent signal selection algorithms to identify
the most promising signals from a large set based on various criteria:

- **Top-N**: Select best signals by a single metric
- **Uncorrelated**: Select diverse signals with low correlation
- **Pareto Frontier**: Select non-dominated signals on two metrics
- **Cluster Representatives**: Select best signal from each correlation cluster

These algorithms help reduce a large signal universe (50-200) to a manageable
subset for detailed comparison while maximizing information value.

Examples
--------
>>> from ml4t.diagnostic.evaluation.signal_selector import SignalSelector
>>>
>>> # Select top 10 by IC IR
>>> top_signals = SignalSelector.select_top_n(summary, n=10, metric="ic_ir")
>>>
>>> # Select 5 uncorrelated signals
>>> diverse = SignalSelector.select_uncorrelated(
...     summary, correlation_matrix, n=5, max_correlation=0.5
... )
>>>
>>> # Find Pareto-optimal signals (low turnover, high IC)
>>> efficient = SignalSelector.select_pareto_frontier(
...     summary, x_metric="turnover_mean", y_metric="ic_ir"
... )
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl

if TYPE_CHECKING:
    pass


class SignalSelector:
    """Smart signal selection algorithms for comparison.

    Provides static methods for selecting subsets of signals based on
    different criteria. All methods are designed to work with summary
    DataFrames from MultiSignalAnalysis.

    Methods
    -------
    select_top_n : Select top N signals by metric
    select_uncorrelated : Select diverse, uncorrelated signals
    select_pareto_frontier : Select Pareto-optimal signals
    select_by_cluster : Select representative from each cluster
    """

    @staticmethod
    def select_top_n(
        summary_df: pl.DataFrame,
        n: int = 10,
        metric: str = "ic_ir",
        ascending: bool = False,
        filter_significant: bool = False,
        significance_col: str = "fdr_significant",
    ) -> list[str]:
        """Select top N signals by a single metric.

        Parameters
        ----------
        summary_df : pl.DataFrame
            Summary DataFrame with columns: signal_name, {metric}
        n : int, default 10
            Number of signals to select
        metric : str, default "ic_ir"
            Metric column to sort by
        ascending : bool, default False
            If True, select lowest values (e.g., for turnover)
        filter_significant : bool, default False
            If True, only consider signals that pass significance threshold
        significance_col : str, default "fdr_significant"
            Column containing significance flag

        Returns
        -------
        list[str]
            Signal names of top N signals

        Examples
        --------
        >>> # Top 10 by IC IR (highest)
        >>> top = SignalSelector.select_top_n(summary, n=10, metric="ic_ir")
        >>>
        >>> # Top 10 lowest turnover
        >>> low_turn = SignalSelector.select_top_n(
        ...     summary, n=10, metric="turnover_mean", ascending=True
        ... )
        """
        if metric not in summary_df.columns:
            raise ValueError(f"Metric '{metric}' not found. Available: {summary_df.columns}")

        df = summary_df

        # Optionally filter to significant only
        if filter_significant and significance_col in df.columns:
            df = df.filter(pl.col(significance_col))

        # Sort and take top N
        sorted_df = df.sort(metric, descending=not ascending)
        return sorted_df.head(n)["signal_name"].to_list()

    @staticmethod
    def select_uncorrelated(
        summary_df: pl.DataFrame,
        correlation_matrix: pl.DataFrame,
        n: int = 5,
        metric: str = "ic_ir",
        min_metric_value: float | None = None,
        max_correlation: float = 0.7,
    ) -> list[str]:
        """Select top N signals that are least correlated with each other.

        Uses a greedy algorithm:
        1. Filter signals with metric >= min_metric_value (if specified)
        2. Sort remaining by metric (descending)
        3. Select best signal
        4. For each remaining, select signal with lowest max correlation
           to already-selected signals, subject to max_correlation threshold
        5. Repeat until N signals selected or no more available

        Parameters
        ----------
        summary_df : pl.DataFrame
            Summary DataFrame with signal_name and metric columns
        correlation_matrix : pl.DataFrame
            Square correlation matrix with signal names as both index and columns
        n : int, default 5
            Number of signals to select
        metric : str, default "ic_ir"
            Metric to rank signals by (higher is better)
        min_metric_value : float | None, default None
            Minimum metric value to consider a signal
        max_correlation : float, default 0.7
            Maximum allowed correlation between selected signals

        Returns
        -------
        list[str]
            Signal names of selected uncorrelated signals

        Notes
        -----
        This is a greedy algorithm that may not find the globally optimal
        subset, but works well in practice and is O(n²) in the number of
        signals.

        Examples
        --------
        >>> # Select 5 diverse signals with IC > 0.02
        >>> diverse = SignalSelector.select_uncorrelated(
        ...     summary, corr_matrix, n=5,
        ...     min_metric_value=0.02, max_correlation=0.5
        ... )
        """
        # Get available signals and their metrics
        candidates = summary_df.select(["signal_name", metric])

        # Filter by minimum metric if specified
        if min_metric_value is not None:
            candidates = candidates.filter(pl.col(metric) >= min_metric_value)

        if len(candidates) == 0:
            return []

        # Sort by metric descending
        candidates = candidates.sort(metric, descending=True)
        candidate_names = candidates["signal_name"].to_list()

        # Convert correlation matrix to numpy for efficient indexing
        corr_signals = correlation_matrix.columns
        corr_numpy = correlation_matrix.to_numpy()

        # Build name-to-index mapping
        signal_to_idx = {name: i for i, name in enumerate(corr_signals)}

        # Greedy selection
        selected: list[str] = []
        remaining = set(candidate_names)

        for signal_name in candidate_names:
            if signal_name not in remaining:
                continue

            if signal_name not in signal_to_idx:
                # Signal not in correlation matrix (shouldn't happen normally)
                remaining.discard(signal_name)
                continue

            # Check correlation with already selected signals
            if len(selected) > 0:
                idx = signal_to_idx[signal_name]
                selected_idxs = [signal_to_idx[s] for s in selected]
                correlations = np.abs(corr_numpy[idx, selected_idxs])
                max_corr = np.max(correlations)

                if max_corr > max_correlation:
                    remaining.discard(signal_name)
                    continue

            # Select this signal
            selected.append(signal_name)
            remaining.discard(signal_name)

            if len(selected) >= n:
                break

        return selected

    @staticmethod
    def select_pareto_frontier(
        summary_df: pl.DataFrame,
        x_metric: str = "turnover_mean",
        y_metric: str = "ic_ir",
        minimize_x: bool = True,
        maximize_y: bool = True,
    ) -> list[str]:
        """Select signals on the Pareto frontier (efficient frontier).

        A signal is Pareto-optimal if no other signal is strictly better
        on both metrics. This finds signals that represent different
        trade-offs between the two metrics.

        Parameters
        ----------
        summary_df : pl.DataFrame
            Summary DataFrame with signal_name, x_metric, y_metric columns
        x_metric : str, default "turnover_mean"
            First metric (typically to minimize, like turnover)
        y_metric : str, default "ic_ir"
            Second metric (typically to maximize, like IC)
        minimize_x : bool, default True
            If True, lower x values are better
        maximize_y : bool, default True
            If True, higher y values are better

        Returns
        -------
        list[str]
            Signal names on the Pareto frontier, sorted by x_metric

        Notes
        -----
        The Pareto frontier helps identify signals that represent different
        trade-offs. For example, one signal might have the highest IC but
        also the highest turnover, while another has moderate IC with low
        turnover. Both are Pareto-optimal.

        Time complexity: O(n²) where n is number of signals.

        Examples
        --------
        >>> # Find signals with best IC vs turnover trade-off
        >>> frontier = SignalSelector.select_pareto_frontier(
        ...     summary, x_metric="turnover_mean", y_metric="ic_ir"
        ... )
        >>> print(f"{len(frontier)} Pareto-optimal signals")
        """
        if x_metric not in summary_df.columns or y_metric not in summary_df.columns:
            raise ValueError(
                f"Metrics not found. Required: {x_metric}, {y_metric}. "
                f"Available: {summary_df.columns}"
            )

        # Extract data
        data = summary_df.select(["signal_name", x_metric, y_metric]).to_numpy()
        names = data[:, 0].tolist()
        x_values = data[:, 1].astype(float)
        y_values = data[:, 2].astype(float)

        # Convert to "higher is better" for comparison
        if minimize_x:
            x_values = -x_values
        if not maximize_y:
            y_values = -y_values

        # Find Pareto frontier
        n = len(names)
        pareto_mask = np.ones(n, dtype=bool)

        for i in range(n):
            if not pareto_mask[i]:
                continue
            for j in range(n):
                if i == j or not pareto_mask[j]:
                    continue
                # Check if j dominates i (j better on both metrics)
                if x_values[j] >= x_values[i] and y_values[j] >= y_values[i]:
                    if x_values[j] > x_values[i] or y_values[j] > y_values[i]:
                        pareto_mask[i] = False
                        break

        # Sort by original x_metric (not negated)
        x_original = data[:, 1].astype(float)
        pareto_with_x = [(names[i], x_original[i]) for i in range(n) if pareto_mask[i]]
        pareto_with_x.sort(key=lambda x: x[1], reverse=not minimize_x)

        return [name for name, _ in pareto_with_x]

    @staticmethod
    def select_by_cluster(
        correlation_matrix: pl.DataFrame,
        summary_df: pl.DataFrame,
        n_clusters: int = 5,
        signals_per_cluster: int = 1,
        metric: str = "ic_ir",
        linkage_method: str = "ward",
    ) -> list[str]:
        """Select representative signals from each correlation cluster.

        Uses hierarchical clustering on correlation distance to group
        similar signals, then selects the best signal(s) from each cluster.

        Parameters
        ----------
        correlation_matrix : pl.DataFrame
            Square correlation matrix (signals as columns)
        summary_df : pl.DataFrame
            Summary with signal_name and metric columns
        n_clusters : int, default 5
            Number of clusters to create
        signals_per_cluster : int, default 1
            Number of signals to select from each cluster
        metric : str, default "ic_ir"
            Metric for selecting best within cluster
        linkage_method : str, default "ward"
            Hierarchical clustering linkage method

        Returns
        -------
        list[str]
            Selected signal names (one per cluster, sorted by metric)

        Notes
        -----
        This method is useful for finding truly independent signal sources.
        "100 signals = 3 unique bets" pattern can be revealed by clustering.

        Requires scipy for hierarchical clustering.

        Examples
        --------
        >>> # Select best signal from each of 5 clusters
        >>> reps = SignalSelector.select_by_cluster(
        ...     corr_matrix, summary, n_clusters=5
        ... )
        """
        try:
            from scipy.cluster.hierarchy import cut_tree, linkage
        except ImportError as err:
            raise ImportError(
                "scipy required for cluster selection. Install with: pip install scipy"
            ) from err

        # Get signal names and correlation matrix
        signal_names = correlation_matrix.columns
        corr_np = correlation_matrix.to_numpy()

        # Convert correlation to distance (1 - |correlation|)
        distance = 1 - np.abs(corr_np)
        np.fill_diagonal(distance, 0)

        # Perform hierarchical clustering
        # linkage expects condensed distance matrix
        n = len(signal_names)
        condensed = distance[np.triu_indices(n, k=1)]
        linkage_matrix = linkage(condensed, method=linkage_method)

        # Cut tree to get cluster labels
        cluster_labels = cut_tree(linkage_matrix, n_clusters=n_clusters).flatten()

        # Build cluster -> signals mapping
        clusters: dict[int, list[str]] = {i: [] for i in range(n_clusters)}
        for i, signal in enumerate(signal_names):
            clusters[cluster_labels[i]].append(signal)

        # Get metric values from summary
        metric_lookup = dict(
            zip(
                summary_df["signal_name"].to_list(),
                summary_df[metric].to_list(),
            )
        )

        # Select best signal(s) from each cluster
        selected: list[str] = []
        for cluster_id in range(n_clusters):
            cluster_signals = clusters[cluster_id]
            if not cluster_signals:
                continue

            # Sort by metric and take top signals_per_cluster
            sorted_signals = sorted(
                cluster_signals,
                key=lambda s: metric_lookup.get(s, float("-inf")),
                reverse=True,
            )
            selected.extend(sorted_signals[:signals_per_cluster])

        # Sort final list by metric
        selected.sort(
            key=lambda s: metric_lookup.get(s, float("-inf")),
            reverse=True,
        )

        return selected

    @staticmethod
    def get_selection_info(
        summary_df: pl.DataFrame,
        selected_signals: list[str],
        method: str,
        **method_params: Any,
    ) -> dict[str, Any]:
        """Get information about a signal selection for documentation.

        Parameters
        ----------
        summary_df : pl.DataFrame
            Summary DataFrame
        selected_signals : list[str]
            List of selected signal names
        method : str
            Selection method name ("top_n", "uncorrelated", "pareto", "cluster")
        **method_params : Any
            Parameters used for selection

        Returns
        -------
        dict
            Dictionary with selection metadata for reporting
        """
        # Get metrics for selected signals
        selected_data = summary_df.filter(pl.col("signal_name").is_in(selected_signals))

        return {
            "method": method,
            "n_selected": len(selected_signals),
            "n_total": len(summary_df),
            "signals": selected_signals,
            "method_params": method_params,
            "selected_summary": selected_data.to_dicts(),
        }
