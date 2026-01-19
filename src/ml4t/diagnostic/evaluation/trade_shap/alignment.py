"""Fast timestamp alignment for trade SHAP analysis.

This module provides O(log n) timestamp lookup instead of O(n) linear scan,
using precomputed indices and binary search for nearest-match scenarios.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass(frozen=True)
class AlignmentResult:
    """Result of timestamp alignment.

    Attributes:
        index: Index into the feature DataFrame, or None if not found
        exact: Whether this was an exact match
        distance_seconds: Distance in seconds from target (0 if exact)
    """

    index: int | None
    exact: bool
    distance_seconds: float


@dataclass
class TimestampAligner:
    """Fast timestamp alignment using precomputed indices.

    Provides O(1) exact match lookup via dict and O(log n) nearest-match
    via binary search on sorted numpy datetime64 array.

    Attributes:
        timestamps_ns: Sorted numpy array of timestamps as int64 nanoseconds
        index_by_ts: Dict mapping datetime to index for O(1) exact lookup
        tolerance_seconds: Maximum allowed distance for nearest match
        _sorted_indices: Original indices corresponding to sorted timestamps

    Example:
        >>> import pandas as pd
        >>> timestamps = pd.DatetimeIndex(['2024-01-01', '2024-01-02', '2024-01-03'])
        >>> aligner = TimestampAligner.from_datetime_index(timestamps, tolerance_seconds=3600)
        >>> result = aligner.align(datetime(2024, 1, 2))
        >>> result.index
        1
        >>> result.exact
        True
    """

    timestamps_ns: NDArray[np.int64]
    index_by_ts: dict[datetime, int] = field(default_factory=dict)
    tolerance_seconds: float = 0.0
    _sorted_indices: NDArray[np.intp] = field(default_factory=lambda: np.array([], dtype=np.intp))

    @classmethod
    def from_datetime_index(
        cls,
        timestamps: NDArray | list[datetime],
        tolerance_seconds: float = 0.0,
    ) -> TimestampAligner:
        """Create aligner from datetime index or array.

        Args:
            timestamps: DatetimeIndex, numpy datetime64 array, or list of datetimes
            tolerance_seconds: Maximum allowed distance for nearest match (default: 0 = exact only)

        Returns:
            TimestampAligner ready for fast lookups

        Raises:
            ValueError: If timestamps array is empty
        """
        # Convert to numpy datetime64[ns] if needed
        ts_array = np.asarray(timestamps, dtype="datetime64[ns]")

        if len(ts_array) == 0:
            raise ValueError("Cannot create aligner from empty timestamp array")

        # Convert to int64 nanoseconds for fast comparison
        ts_ns = ts_array.astype(np.int64)

        # Get sort order (we need original indices)
        sorted_indices = np.argsort(ts_ns)
        sorted_ts_ns = ts_ns[sorted_indices]

        # Build exact-match dict using original timestamps
        # For duplicates, keep FIRST occurrence (standard behavior)
        index_by_ts: dict[datetime, int] = {}
        for i, ts in enumerate(timestamps):
            if hasattr(ts, "to_pydatetime"):
                # pandas Timestamp
                dt = ts.to_pydatetime()
            elif isinstance(ts, np.datetime64):
                # numpy datetime64
                dt = ts.astype("datetime64[us]").astype(datetime)
            else:
                dt = ts
            # Only store first occurrence of each timestamp
            if dt not in index_by_ts:
                index_by_ts[dt] = i

        return cls(
            timestamps_ns=sorted_ts_ns,
            index_by_ts=index_by_ts,
            tolerance_seconds=tolerance_seconds,
            _sorted_indices=sorted_indices,
        )

    def align(self, target: datetime) -> AlignmentResult:
        """Find index for target timestamp.

        First attempts exact match via dict lookup (O(1)).
        If no exact match and tolerance > 0, uses binary search for nearest (O(log n)).

        Args:
            target: Target timestamp to align

        Returns:
            AlignmentResult with index (or None), exact flag, and distance
        """
        # Try exact match first (O(1))
        if target in self.index_by_ts:
            return AlignmentResult(index=self.index_by_ts[target], exact=True, distance_seconds=0.0)

        # No exact match - if no tolerance, return None
        if self.tolerance_seconds <= 0:
            return AlignmentResult(index=None, exact=False, distance_seconds=float("inf"))

        # Binary search for nearest (O(log n))
        target_ns = np.datetime64(target, "ns").astype(np.int64)
        insert_pos = np.searchsorted(self.timestamps_ns, target_ns)

        # Check neighbors
        candidates = []
        if insert_pos > 0:
            candidates.append(insert_pos - 1)
        if insert_pos < len(self.timestamps_ns):
            candidates.append(insert_pos)

        if not candidates:
            return AlignmentResult(index=None, exact=False, distance_seconds=float("inf"))

        # Find closest
        best_idx = None
        best_distance_ns = float("inf")

        for sorted_idx in candidates:
            distance_ns = abs(self.timestamps_ns[sorted_idx] - target_ns)
            if distance_ns < best_distance_ns:
                best_distance_ns = distance_ns
                best_idx = sorted_idx

        # Convert to seconds and check tolerance
        distance_seconds = best_distance_ns / 1e9

        if distance_seconds <= self.tolerance_seconds:
            # Map back to original index
            original_idx = int(self._sorted_indices[best_idx])
            return AlignmentResult(
                index=original_idx,
                exact=False,
                distance_seconds=distance_seconds,
            )

        return AlignmentResult(index=None, exact=False, distance_seconds=distance_seconds)

    def align_many(self, targets: list[datetime]) -> list[AlignmentResult]:
        """Align multiple timestamps.

        Args:
            targets: List of target timestamps

        Returns:
            List of AlignmentResult for each target
        """
        return [self.align(t) for t in targets]

    def __len__(self) -> int:
        """Number of timestamps in the aligner."""
        return len(self.timestamps_ns)
