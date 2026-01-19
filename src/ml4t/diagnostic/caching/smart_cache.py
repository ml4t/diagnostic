"""Smart cache with Polars DataFrame fingerprinting.

This module provides a memory-only cache optimized for signal analysis workloads,
featuring fast and stable DataFrame fingerprinting using Polars' hash_rows().

The SmartCache is designed for exploration workflows where signals are frequently
re-analyzed with different parameters. It uses LRU eviction and optional TTL
expiration to manage memory usage.

Examples
--------
>>> from ml4t.diagnostic.caching.smart_cache import SmartCache
>>> cache = SmartCache(max_items=100, ttl_seconds=3600)
>>>
>>> # Generate cache key for a signal
>>> key = cache.make_key("momentum", signal_df, config)
>>>
>>> # Check cache
>>> result = cache.get(key)
>>> if result is None:
...     result = expensive_analysis(signal_df)
...     cache.set(key, result)

References
----------
Polars hash_rows: https://pola-rs.github.io/polars/py-polars/html/reference/dataframe.html
"""

from __future__ import annotations

import hashlib
import time
from collections import OrderedDict
from typing import TYPE_CHECKING, Any

import polars as pl

if TYPE_CHECKING:
    from ml4t.diagnostic.config.base import BaseConfig


class SmartCache:
    """Memory cache with Polars DataFrame fingerprinting.

    Provides fast, stable caching for signal analysis results using
    content-based keys generated from DataFrames and configurations.

    Features
    --------
    - **Polars fingerprinting**: Uses pl.hash_rows() for fast, stable hashing
    - **LRU eviction**: Automatically removes least recently used items
    - **TTL expiration**: Optional time-based expiration
    - **Memory-only**: No disk persistence (simpler, exploration-focused)

    Parameters
    ----------
    max_items : int, default 100
        Maximum number of items in cache. When exceeded, LRU eviction occurs.
    ttl_seconds : int | None, default 3600
        Time-to-live in seconds. None disables expiration.

    Examples
    --------
    >>> cache = SmartCache(max_items=200, ttl_seconds=None)  # No expiration
    >>>
    >>> # Cache individual signal results
    >>> for name, df in signals.items():
    ...     key = cache.make_key(name, df, config)
    ...     result = cache.get(key)
    ...     if result is None:
    ...         result = analyzer.analyze(df)
    ...         cache.set(key, result)
    """

    def __init__(self, max_items: int = 100, ttl_seconds: int | None = 3600):
        """Initialize SmartCache.

        Parameters
        ----------
        max_items : int
            Maximum cache size (LRU eviction when exceeded)
        ttl_seconds : int | None
            Time-to-live in seconds (None = no expiration)
        """
        self._cache: OrderedDict[str, tuple[Any, float]] = OrderedDict()
        self.max_items = max_items
        self.ttl_seconds = ttl_seconds
        self._hits = 0
        self._misses = 0

    @staticmethod
    def polars_fingerprint(df: pl.DataFrame, seed: int = 42) -> str:
        """Generate stable hash from Polars DataFrame.

        Uses pl.hash_rows() for fast row-wise hashing, combined with
        schema and shape information for collision resistance.

        Parameters
        ----------
        df : pl.DataFrame
            DataFrame to fingerprint
        seed : int, default 42
            Seed for hash_rows() reproducibility

        Returns
        -------
        str
            MD5 hex digest of the DataFrame content

        Notes
        -----
        The fingerprint includes:
        - Column names and dtypes (schema)
        - DataFrame shape
        - Row-wise content hash using pl.hash_rows()

        This ensures different DataFrames produce different fingerprints,
        while identical DataFrames always produce the same fingerprint.

        Examples
        --------
        >>> df1 = pl.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
        >>> fp1 = SmartCache.polars_fingerprint(df1)
        >>>
        >>> # Same data = same fingerprint
        >>> df2 = pl.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
        >>> fp2 = SmartCache.polars_fingerprint(df2)
        >>> assert fp1 == fp2
        >>>
        >>> # Different data = different fingerprint
        >>> df3 = pl.DataFrame({"a": [1, 2, 4], "b": [4.0, 5.0, 6.0]})
        >>> fp3 = SmartCache.polars_fingerprint(df3)
        >>> assert fp1 != fp3
        """
        # Build schema string for deterministic ordering
        schema_str = str([(c, str(d)) for c, d in zip(df.columns, df.dtypes)])

        # Compute row hashes using Polars' optimized function
        row_hashes = df.hash_rows(seed=seed)

        # Combine into final hash
        hasher = hashlib.md5()
        hasher.update(schema_str.encode())
        hasher.update(row_hashes.to_numpy().tobytes())
        hasher.update(f"{df.shape}".encode())

        return hasher.hexdigest()

    def make_key(
        self,
        signal_name: str,
        signal_df: pl.DataFrame,
        config: BaseConfig,
    ) -> str:
        """Generate cache key from signal name, data, and configuration.

        Parameters
        ----------
        signal_name : str
            Unique identifier for the signal
        signal_df : pl.DataFrame
            Signal data
        config : BaseConfig
            Analysis configuration

        Returns
        -------
        str
            Cache key combining signal, data fingerprint, and config hash

        Examples
        --------
        >>> key = cache.make_key("momentum_12m", momentum_df, config)
        >>> key
        'momentum_12m_a1b2c3d4e5f6_g7h8i9j0k1l2'
        """
        # DataFrame fingerprint (first 12 chars)
        df_hash = self.polars_fingerprint(signal_df)[:12]

        # Config hash (first 12 chars)
        config_hash = hashlib.md5(config.model_dump_json().encode()).hexdigest()[:12]

        return f"{signal_name}_{df_hash}_{config_hash}"

    def get(self, key: str) -> Any | None:
        """Retrieve value from cache.

        Parameters
        ----------
        key : str
            Cache key (from make_key())

        Returns
        -------
        Any | None
            Cached value, or None if not found/expired

        Notes
        -----
        Updates LRU ordering on hit. Automatically removes expired entries.
        """
        if key not in self._cache:
            self._misses += 1
            return None

        value, timestamp = self._cache[key]

        # Check TTL expiration
        if self.ttl_seconds is not None:
            age = time.time() - timestamp
            if age > self.ttl_seconds:
                del self._cache[key]
                self._misses += 1
                return None

        # Move to end (most recently used)
        self._cache.move_to_end(key)
        self._hits += 1
        return value

    def set(self, key: str, value: Any) -> None:
        """Store value in cache.

        Parameters
        ----------
        key : str
            Cache key
        value : Any
            Value to cache

        Notes
        -----
        Triggers LRU eviction if cache exceeds max_items.
        """
        # Evict oldest entries if at capacity
        while len(self._cache) >= self.max_items:
            self._cache.popitem(last=False)

        # Add/update entry
        self._cache[key] = (value, time.time())
        self._cache.move_to_end(key)

    def invalidate(self, key: str) -> bool:
        """Remove specific entry from cache.

        Parameters
        ----------
        key : str
            Cache key to invalidate

        Returns
        -------
        bool
            True if key existed and was removed, False otherwise
        """
        if key in self._cache:
            del self._cache[key]
            return True
        return False

    def clear(self) -> None:
        """Remove all entries from cache."""
        self._cache.clear()
        self._hits = 0
        self._misses = 0

    def invalidate_signal(self, signal_name: str) -> int:
        """Invalidate all cache entries for a specific signal.

        Useful when signal data has been updated and all cached
        analysis results need to be discarded.

        Parameters
        ----------
        signal_name : str
            Signal name prefix to match

        Returns
        -------
        int
            Number of entries removed
        """
        prefix = f"{signal_name}_"
        keys_to_remove = [k for k in self._cache if k.startswith(prefix)]
        for key in keys_to_remove:
            del self._cache[key]
        return len(keys_to_remove)

    @property
    def size(self) -> int:
        """Current number of items in cache."""
        return len(self._cache)

    @property
    def hit_rate(self) -> float:
        """Cache hit rate (0.0 to 1.0)."""
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    @property
    def stats(self) -> dict[str, Any]:
        """Cache statistics.

        Returns
        -------
        dict
            Dictionary with hits, misses, hit_rate, size, max_items, ttl_seconds
        """
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self.hit_rate,
            "size": self.size,
            "max_items": self.max_items,
            "ttl_seconds": self.ttl_seconds,
        }

    def __repr__(self) -> str:
        """Developer representation."""
        return (
            f"SmartCache(size={self.size}/{self.max_items}, "
            f"hit_rate={self.hit_rate:.1%}, ttl={self.ttl_seconds}s)"
        )

    def __contains__(self, key: str) -> bool:
        """Check if key exists in cache (does not update LRU or count as hit)."""
        if key not in self._cache:
            return False
        # Check expiration without modifying state
        if self.ttl_seconds is not None:
            _, timestamp = self._cache[key]
            age = time.time() - timestamp
            if age > self.ttl_seconds:
                return False
        return True

    def __len__(self) -> int:
        """Return number of items in cache."""
        return len(self._cache)
