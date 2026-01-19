"""Trade SHAP explanation logic.

This module provides the TradeShapExplainer class that explains individual trades
using SHAP values, with O(log n) timestamp alignment and efficient feature extraction.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from ml4t.diagnostic.evaluation.trade_shap.alignment import TimestampAligner
from ml4t.diagnostic.evaluation.trade_shap.models import (
    TradeExplainFailure,
    TradeShapExplanation,
)

if TYPE_CHECKING:
    import polars as pl
    from numpy.typing import NDArray

    from ml4t.diagnostic.evaluation.trade_analysis import TradeMetrics


class TradeShapExplainer:
    """Explains individual trades using SHAP values.

    Uses TimestampAligner for O(log n) timestamp lookup and extracts
    feature values in a single row read for efficiency.

    Returns TradeExplainFailure for expected failure cases instead of
    throwing exceptions, enabling clean batch processing.

    Attributes:
        features_df: Polars DataFrame with timestamp and feature columns
        shap_values: 2D numpy array of SHAP values (n_samples x n_features)
        feature_names: List of feature column names
        aligner: TimestampAligner for fast timestamp lookup
        top_n_features: Number of top features to include in explanation

    Example:
        >>> explainer = TradeShapExplainer(
        ...     features_df=features,
        ...     shap_values=shap_values,
        ...     feature_names=feature_names,
        ...     tolerance_seconds=60.0,
        ... )
        >>> result = explainer.explain(trade)
        >>> if isinstance(result, TradeShapExplanation):
        ...     print(result.top_features[:3])
        ... else:
        ...     print(f"Failed: {result.reason}")
    """

    def __init__(
        self,
        features_df: pl.DataFrame,
        shap_values: NDArray[np.floating[Any]],
        feature_names: list[str],
        tolerance_seconds: float = 0.0,
        top_n_features: int | None = None,
        alignment_mode: str = "entry",
        missing_value_strategy: str = "skip",
    ) -> None:
        """Initialize the explainer.

        Args:
            features_df: Polars DataFrame with 'timestamp' column and feature columns
            shap_values: SHAP values array (n_samples x n_features)
            feature_names: List of feature column names matching shap_values columns
            tolerance_seconds: Maximum seconds for nearest-match alignment (0 = exact only)
            top_n_features: Number of top features to include (None = all)
            alignment_mode: 'entry' for exact match, 'nearest' for closest within tolerance
            missing_value_strategy: How to handle alignment failures ('error', 'skip', 'zero')

        Raises:
            ValueError: If shap_values shape doesn't match features_df rows or feature_names
        """
        self.features_df = features_df
        self.shap_values = shap_values
        self.feature_names = feature_names
        self.top_n_features = top_n_features
        self.alignment_mode = alignment_mode
        self.missing_value_strategy = missing_value_strategy

        # Validate shapes
        n_rows = len(features_df)
        n_features = len(feature_names)

        if shap_values.shape[0] != n_rows:
            raise ValueError(
                f"SHAP values rows ({shap_values.shape[0]}) != features_df rows ({n_rows})"
            )
        if shap_values.shape[1] != n_features:
            raise ValueError(
                f"SHAP values columns ({shap_values.shape[1]}) != feature_names ({n_features})"
            )

        # Build aligner with appropriate tolerance
        timestamps = features_df["timestamp"].to_list()
        effective_tolerance = tolerance_seconds if alignment_mode == "nearest" else 0.0
        self.aligner = TimestampAligner.from_datetime_index(
            timestamps, tolerance_seconds=effective_tolerance
        )

        # Cache feature data as numpy for fast row extraction
        self._feature_matrix = features_df.select(feature_names).to_numpy()

    def explain(
        self,
        trade: TradeMetrics,
    ) -> TradeShapExplanation | TradeExplainFailure:
        """Explain a single trade.

        Args:
            trade: Trade to explain (must have timestamp and symbol attributes)

        Returns:
            TradeShapExplanation on success, TradeExplainFailure on expected failures
        """
        trade_id = f"{trade.symbol}_{trade.timestamp.isoformat()}"

        # Align to timestamp
        result = self.aligner.align(trade.timestamp)

        if result.index is None:
            # Handle alignment failure based on strategy
            if self.missing_value_strategy == "error":
                raise ValueError(
                    f"Cannot align SHAP values for trade {trade_id}: "
                    f"no timestamp within {self.aligner.tolerance_seconds}s "
                    f"(nearest is {result.distance_seconds:.1f}s away)"
                )
            elif self.missing_value_strategy == "zero":
                # Return zero SHAP vector
                shap_vector = np.zeros(len(self.feature_names))
                feature_values = dict.fromkeys(self.feature_names, 0.0)
                top_features = [(name, 0.0) for name in self.feature_names]
                return TradeShapExplanation(
                    trade_id=trade_id,
                    timestamp=trade.timestamp,
                    top_features=top_features,
                    feature_values=feature_values,
                    shap_vector=shap_vector,
                )
            else:  # "skip" or default
                return TradeExplainFailure(
                    trade_id=trade_id,
                    timestamp=trade.timestamp,
                    reason="alignment_missing",
                    details={
                        "alignment_mode": self.alignment_mode,
                        "tolerance_seconds": self.aligner.tolerance_seconds,
                        "distance_seconds": result.distance_seconds,
                    },
                )

        idx = result.index

        # Extract SHAP vector for this row
        shap_vector = np.asarray(self.shap_values[idx, :], dtype=np.float64)

        # Extract feature values in one row read (not per-feature loop)
        feature_row = self._feature_matrix[idx, :]
        feature_values = {
            name: float(val) for name, val in zip(self.feature_names, feature_row, strict=True)
        }

        # Get top N contributors by absolute SHAP value
        top_n = self.top_n_features if self.top_n_features is not None else len(self.feature_names)

        # Create (feature_name, shap_value) pairs and sort by |shap|
        feature_shap_pairs = list(zip(self.feature_names, shap_vector.tolist(), strict=True))
        feature_shap_pairs.sort(key=lambda x: abs(x[1]), reverse=True)
        top_features = [(name, float(val)) for name, val in feature_shap_pairs[:top_n]]

        return TradeShapExplanation(
            trade_id=trade_id,
            timestamp=trade.timestamp,
            top_features=top_features,
            feature_values=feature_values,
            shap_vector=shap_vector,
        )

    def explain_many(
        self,
        trades: list[TradeMetrics],
    ) -> tuple[list[TradeShapExplanation], list[TradeExplainFailure]]:
        """Explain multiple trades.

        Args:
            trades: List of trades to explain

        Returns:
            Tuple of (successful explanations, failures)
        """
        explanations: list[TradeShapExplanation] = []
        failures: list[TradeExplainFailure] = []

        for trade in trades:
            result = self.explain(trade)
            if isinstance(result, TradeShapExplanation):
                explanations.append(result)
            else:
                failures.append(result)

        return explanations, failures
