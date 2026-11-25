"""Trade SHAP explanation logic.

This module provides the TradeShapExplainer class that explains individual trades
using the SHAP values of the feature row the model saw when it opened the trade.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import polars as pl

from ml4t.diagnostic.evaluation.trade_shap.models import (
    TradeExplainFailure,
    TradeShapExplanation,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ml4t.diagnostic.evaluation.trade_analysis import TradeMetrics

#: Key columns of ``features_df`` that identify a row and are never model features.
KEY_COLUMNS = ("timestamp", "symbol")


def _utc_naive_ns(value: Any) -> int:
    """Return an instant as UTC nanoseconds; naive values are taken as UTC."""
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return int(ts.value)


def _timestamp_column_ns(column: pl.Series) -> NDArray[np.int64]:
    """Return a polars date/datetime column as UTC nanoseconds."""
    if column.null_count():
        raise ValueError("features_df 'timestamp' column must not contain nulls")
    if column.dtype == pl.Date:
        column = column.cast(pl.Datetime("ns"))
    elif not isinstance(column.dtype, pl.Datetime):
        raise ValueError(
            f"features_df 'timestamp' column must be Date or Datetime, got {column.dtype}"
        )
    if column.dtype.time_zone is not None:  # type: ignore[union-attr]
        column = column.dt.convert_time_zone("UTC").dt.replace_time_zone(None)
    return column.dt.cast_time_unit("ns").cast(pl.Int64).to_numpy()


class TradeShapExplainer:
    """Explains individual trades using SHAP values.

    A trade is explained by the SHAP row of the decision that opened it: the row
    for ``trade.symbol`` (when ``features_df`` has a ``symbol`` column) at the
    trade's decision time. Under the ``TradeRecord`` contract ``trade.timestamp``
    is the exit and ``trade.entry_timestamp`` the entry; when the entry is not
    recorded it is ``timestamp - duration``. The decision row is the last row
    strictly before the entry (``decision_time="before_entry"``, matching
    next-bar fills) or at or before it (``"at_or_before_entry"``, for same-bar
    fills or features stamped at the time they became known).

    Returns TradeExplainFailure for expected failure cases instead of
    throwing exceptions, enabling clean batch processing.

    Attributes:
        features_df: Polars DataFrame with timestamp, optional symbol, and feature columns
        shap_values: 2D numpy array of SHAP values (n_samples x n_features)
        feature_names: List of feature column names
        tolerance_seconds: Maximum age of the decision row in 'nearest' mode (None otherwise)
        top_n_features: Number of top features to include in explanation

    In 'entry' mode the decision row may be no older than the longest gap between
    consecutive rows of the trade's symbol, so a trade entering beyond the end of
    the feature history is reported as ``alignment_missing`` instead of being
    explained by a stale row. In 'nearest' mode ``tolerance_seconds`` sets that age.

    Example:
        >>> explainer = TradeShapExplainer(
        ...     features_df=features,  # timestamp, symbol, feature columns
        ...     shap_values=shap_values,
        ...     feature_names=feature_names,
        ... )
        >>> result = explainer.explain(trade)
        >>> if isinstance(result, TradeShapExplanation):
        ...     print(result.feature_timestamp, result.top_features[:3])
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
        decision_time: str = "before_entry",
    ) -> None:
        """Initialize the explainer.

        Args:
            features_df: Polars DataFrame with a 'timestamp' column, the feature columns
                and, for a multi-asset panel, a 'symbol' column. Without a 'symbol'
                column the timestamps must be unique (one asset).
            shap_values: SHAP values array (n_samples x n_features), row-aligned
                with features_df
            feature_names: List of feature column names matching shap_values columns
            tolerance_seconds: Maximum age of the decision row in 'nearest' mode
            top_n_features: Number of top features to include (None = all)
            alignment_mode: 'entry' accepts a decision row no older than the longest
                gap in the symbol's feature history; 'nearest' requires it to lie
                within tolerance_seconds of the entry
            missing_value_strategy: How to handle alignment failures ('error', 'skip', 'zero')
            decision_time: 'before_entry' (last row strictly before the entry) or
                'at_or_before_entry' (last row at or before the entry)

        Raises:
            ValueError: If shap_values shape doesn't match features_df rows or
                feature_names, or if a row key (symbol, timestamp) is duplicated
        """
        if decision_time not in ("before_entry", "at_or_before_entry"):
            raise ValueError(
                f"decision_time must be 'before_entry' or 'at_or_before_entry', "
                f"got {decision_time!r}"
            )
        self.features_df = features_df
        self.shap_values = shap_values
        self.feature_names = feature_names
        self.top_n_features = top_n_features
        self.alignment_mode = alignment_mode
        self.missing_value_strategy = missing_value_strategy
        self.decision_time = decision_time
        self.tolerance_seconds = float(tolerance_seconds) if alignment_mode == "nearest" else None

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

        self.has_symbol = "symbol" in features_df.columns
        self._timestamps = pd.DatetimeIndex(features_df["timestamp"].to_pandas())
        self._index = self._build_index(features_df)

        # Cache feature data as numpy for fast row extraction
        self._feature_matrix = features_df.select(feature_names).to_numpy()

    def _build_index(
        self, features_df: pl.DataFrame
    ) -> dict[str | None, tuple[NDArray[np.int64], NDArray[np.intp], int]]:
        """Map each symbol (or None) to sorted timestamps, row indices and longest gap."""
        ts_ns = _timestamp_column_ns(features_df["timestamp"])
        if self.has_symbol:
            symbols = features_df["symbol"].cast(pl.String).to_numpy()
            groups: dict[str | None, NDArray[np.intp]] = {
                str(sym): np.flatnonzero(symbols == sym) for sym in np.unique(symbols)
            }
        else:
            groups = {None: np.arange(len(ts_ns))}

        index: dict[str | None, tuple[NDArray[np.int64], NDArray[np.intp], int]] = {}
        for key, rows in groups.items():
            order = np.argsort(ts_ns[rows], kind="stable")
            rows_sorted = rows[order]
            ts_sorted = ts_ns[rows_sorted]
            if len(ts_sorted) > 1 and np.any(ts_sorted[1:] == ts_sorted[:-1]):
                if self.has_symbol:
                    raise ValueError(
                        f"features_df has duplicate timestamps for symbol {key!r}; "
                        "each (symbol, timestamp) must identify one row"
                    )
                raise ValueError(
                    "features_df has duplicate timestamps but no 'symbol' column; "
                    "add a 'symbol' column so each trade aligns to its own asset's row"
                )
            # A single row carries no spacing to bound the decision row's age with.
            max_gap = (
                int(np.diff(ts_sorted).max()) if len(ts_sorted) > 1 else np.iinfo(np.int64).max
            )
            index[key] = (ts_sorted, rows_sorted, max_gap)
        return index

    @staticmethod
    def entry_time(trade: TradeMetrics) -> datetime | None:
        """Return the trade's entry time: ``entry_timestamp``, else exit minus duration."""
        entry = getattr(trade, "entry_timestamp", None)
        if entry is not None:
            return entry
        duration = getattr(trade, "duration", None)
        if duration is not None:
            return trade.timestamp - duration
        return None

    def explain(
        self,
        trade: TradeMetrics,
    ) -> TradeShapExplanation | TradeExplainFailure:
        """Explain a single trade.

        Args:
            trade: Trade to explain (TradeMetrics or TradeRecord-like object with
                symbol, timestamp (exit) and entry_timestamp or duration)

        Returns:
            TradeShapExplanation on success, TradeExplainFailure on expected failures
        """
        trade_id = f"{trade.symbol}_{trade.timestamp.isoformat()}"
        entry = self.entry_time(trade)
        if entry is None:
            return self._missing(
                trade,
                trade_id,
                entry,
                reason="entry_time_missing",
                message="the trade has neither entry_timestamp nor duration",
                details={},
            )

        key = str(trade.symbol) if self.has_symbol else None
        series = self._index.get(key)
        if series is None:
            return self._missing(
                trade,
                trade_id,
                entry,
                reason="alignment_missing",
                message=f"symbol {trade.symbol!r} has no rows in features_df",
                details={"symbol": trade.symbol, "distance_seconds": float("inf")},
            )

        ts_sorted, rows_sorted, max_gap_ns = series
        entry_ns = _utc_naive_ns(entry)
        side = "left" if self.decision_time == "before_entry" else "right"
        pos = int(np.searchsorted(ts_sorted, entry_ns, side=side)) - 1
        if pos < 0:
            return self._missing(
                trade,
                trade_id,
                entry,
                reason="alignment_missing",
                message=f"no feature row {self.decision_time.replace('_', ' ')} {entry}",
                details={"distance_seconds": float("inf")},
            )

        age_ns = entry_ns - int(ts_sorted[pos])
        age_seconds = age_ns / 1e9
        if self.tolerance_seconds is None and age_ns > max_gap_ns:
            return self._missing(
                trade,
                trade_id,
                entry,
                reason="alignment_missing",
                message=(
                    f"the last feature row is {age_seconds:.1f}s before entry, longer than "
                    "any gap in the symbol's feature history"
                ),
                details={"distance_seconds": age_seconds, "max_gap_seconds": max_gap_ns / 1e9},
            )
        if self.tolerance_seconds is not None and age_seconds > self.tolerance_seconds:
            return self._missing(
                trade,
                trade_id,
                entry,
                reason="alignment_missing",
                message=(
                    f"the decision row is {age_seconds:.1f}s before entry, beyond the "
                    f"{self.tolerance_seconds}s tolerance"
                ),
                details={"distance_seconds": age_seconds},
            )

        idx = int(rows_sorted[pos])

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
            symbol=str(trade.symbol),
            entry_timestamp=entry,
            feature_timestamp=self._timestamps[idx],
            pnl=getattr(trade, "pnl", None),
        )

    def _missing(
        self,
        trade: TradeMetrics,
        trade_id: str,
        entry: datetime | None,
        *,
        reason: str,
        message: str,
        details: dict[str, Any],
    ) -> TradeShapExplanation | TradeExplainFailure:
        """Apply missing_value_strategy to a trade without a decision row."""
        if self.missing_value_strategy == "error":
            raise ValueError(f"Cannot align SHAP values for trade {trade_id}: {message}")
        if self.missing_value_strategy == "zero":
            shap_vector = np.zeros(len(self.feature_names))
            return TradeShapExplanation(
                trade_id=trade_id,
                timestamp=trade.timestamp,
                top_features=[(name, 0.0) for name in self.feature_names],
                feature_values=dict.fromkeys(self.feature_names, 0.0),
                shap_vector=shap_vector,
                symbol=str(trade.symbol),
                entry_timestamp=entry,
                pnl=getattr(trade, "pnl", None),
            )
        return TradeExplainFailure(
            trade_id=trade_id,
            timestamp=trade.timestamp,
            reason=reason,
            details={
                "message": message,
                "alignment_mode": self.alignment_mode,
                "decision_time": self.decision_time,
                "tolerance_seconds": self.tolerance_seconds,
                "entry_timestamp": entry,
                **details,
            },
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
