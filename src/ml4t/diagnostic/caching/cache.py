"""Core caching implementation with pluggable backends."""

from __future__ import annotations

import hashlib
import json
import pickle
from collections import OrderedDict
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class CacheBackend(str, Enum):
    """Cache storage backend options."""

    MEMORY = "memory"
    DISK = "disk"
    DISABLED = "disabled"


class CacheConfig(BaseModel):
    """Configuration for cache behavior.

    Attributes:
        enabled: Whether caching is enabled
        backend: Storage backend to use
        ttl_seconds: Time-to-live for cache entries (None = no expiration)
        max_memory_items: Max items in memory cache (LRU eviction)
        disk_path: Path for disk cache storage
        compression: Whether to compress disk cache entries
    """

    enabled: bool = True
    backend: CacheBackend = CacheBackend.MEMORY
    ttl_seconds: int | None = Field(default=3600, description="Cache TTL in seconds")
    max_memory_items: int = Field(default=100, description="Max memory cache size")
    disk_path: Path = Field(
        default_factory=lambda: Path(".qeval_cache"),
        description="Disk cache directory",
    )
    compression: bool = Field(default=False, description="Compress disk cache")


class CacheKey:
    """Cache key with content-based hashing.

    Generates stable cache keys from arbitrary data and configuration.

    Examples:
        >>> key = CacheKey.generate(data=df, config={"alpha": 0.05})
        >>> key_str = str(key)  # "sha256:abc123..."
    """

    def __init__(self, hash_value: str, algorithm: str = "sha256"):
        """Initialize cache key.

        Args:
            hash_value: Hash digest as hex string
            algorithm: Hash algorithm used
        """
        self.hash_value = hash_value
        self.algorithm = algorithm

    @classmethod
    def generate(cls, **kwargs: Any) -> CacheKey:
        """Generate cache key from arbitrary keyword arguments.

        Args:
            **kwargs: Data to hash (must be JSON-serializable or have __hash__)

        Returns:
            CacheKey instance

        Examples:
            >>> key = CacheKey.generate(data=data_hash, config={"alpha": 0.05})
        """
        # Convert to stable JSON representation
        stable_repr = cls._to_stable_repr(kwargs)

        # Hash it
        hasher = hashlib.sha256()
        hasher.update(stable_repr.encode("utf-8"))

        return cls(hash_value=hasher.hexdigest(), algorithm="sha256")

    @staticmethod
    def _to_stable_repr(obj: Any) -> str:
        """Convert object to stable string representation.

        Args:
            obj: Object to convert

        Returns:
            Stable string representation
        """
        if isinstance(obj, dict):
            # Sort keys for stability
            items = sorted(obj.items())
            return json.dumps(items, sort_keys=True, default=str)
        elif isinstance(obj, list | tuple):
            return json.dumps(obj, default=str)
        else:
            return json.dumps(obj, default=str)

    def __str__(self) -> str:
        """String representation."""
        return f"{self.algorithm}:{self.hash_value}"

    def __repr__(self) -> str:
        """Developer representation."""
        return f"CacheKey({self.algorithm}:{self.hash_value[:12]}...)"

    def __eq__(self, other: object) -> bool:
        """Equality comparison."""
        if not isinstance(other, CacheKey):
            return False
        return self.hash_value == other.hash_value

    def __hash__(self) -> int:
        """Hash for use as dict key."""
        return hash(self.hash_value)


class CacheEntry:
    """Cache entry with metadata."""

    def __init__(self, value: Any, created_at: datetime, ttl_seconds: int | None = None):
        """Initialize cache entry.

        Args:
            value: Cached value
            created_at: Creation timestamp
            ttl_seconds: Time-to-live in seconds (None = no expiration)
        """
        self.value = value
        self.created_at = created_at
        self.ttl_seconds = ttl_seconds

    def is_expired(self) -> bool:
        """Check if entry has expired.

        Returns:
            True if expired, False otherwise
        """
        if self.ttl_seconds is None:
            return False

        now = datetime.now(UTC)
        age = (now - self.created_at).total_seconds()
        return age > self.ttl_seconds


class Cache:
    """Multi-backend cache for expensive computations.

    Supports memory and disk backends with automatic expiration and LRU eviction.

    Examples:
        >>> cache = Cache(CacheConfig(enabled=True, backend=CacheBackend.MEMORY))
        >>>
        >>> # Generate key
        >>> key = cache.generate_key(data=data_hash, config=config)
        >>>
        >>> # Get/set
        >>> result = cache.get(key)
        >>> if result is None:
        ...     result = expensive_computation()
        ...     cache.set(key, result)
    """

    def __init__(self, config: CacheConfig):
        """Initialize cache.

        Args:
            config: Cache configuration
        """
        self.config = config
        self._memory_cache: OrderedDict[CacheKey, CacheEntry] = OrderedDict()

        # Create disk cache directory if needed
        if config.backend == CacheBackend.DISK and config.enabled:
            config.disk_path.mkdir(parents=True, exist_ok=True)

    def generate_key(self, **kwargs: Any) -> CacheKey:
        """Generate cache key from data and configuration.

        Args:
            **kwargs: Data to hash

        Returns:
            Cache key

        Examples:
            >>> key = cache.generate_key(data=data_hash, config={"alpha": 0.05})
        """
        return CacheKey.generate(**kwargs)

    def get(self, key: CacheKey) -> Any | None:
        """Get value from cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found/expired
        """
        if not self.config.enabled:
            return None

        if self.config.backend == CacheBackend.MEMORY:
            return self._get_memory(key)
        elif self.config.backend == CacheBackend.DISK:
            return self._get_disk(key)
        else:
            return None

    def set(self, key: CacheKey, value: Any) -> None:
        """Store value in cache.

        Args:
            key: Cache key
            value: Value to cache
        """
        if not self.config.enabled:
            return

        if self.config.backend == CacheBackend.MEMORY:
            self._set_memory(key, value)
        elif self.config.backend == CacheBackend.DISK:
            self._set_disk(key, value)

    def invalidate(self, key: CacheKey) -> None:
        """Invalidate specific cache entry.

        Args:
            key: Cache key to invalidate
        """
        if self.config.backend == CacheBackend.MEMORY:
            self._memory_cache.pop(key, None)
        elif self.config.backend == CacheBackend.DISK:
            cache_file = self._get_disk_path(key)
            if cache_file.exists():
                cache_file.unlink()

    def clear(self) -> None:
        """Clear all cache entries."""
        if self.config.backend == CacheBackend.MEMORY:
            self._memory_cache.clear()
        elif self.config.backend == CacheBackend.DISK and self.config.disk_path.exists():
            for cache_file in self.config.disk_path.glob("*.pkl"):
                cache_file.unlink()

    def _get_memory(self, key: CacheKey) -> Any | None:
        """Get from memory cache with LRU update."""
        entry = self._memory_cache.get(key)

        if entry is None:
            return None

        # Check expiration
        if entry.is_expired():
            self._memory_cache.pop(key)
            return None

        # Move to end (LRU)
        self._memory_cache.move_to_end(key)

        return entry.value

    def _set_memory(self, key: CacheKey, value: Any) -> None:
        """Set in memory cache with LRU eviction."""
        # Check size limit
        while len(self._memory_cache) >= self.config.max_memory_items:
            # Remove oldest (first) item
            self._memory_cache.popitem(last=False)

        # Add new entry
        entry = CacheEntry(
            value=value,
            created_at=datetime.now(UTC),
            ttl_seconds=self.config.ttl_seconds,
        )
        self._memory_cache[key] = entry

    def _get_disk(self, key: CacheKey) -> Any | None:
        """Get from disk cache."""
        cache_file = self._get_disk_path(key)

        if not cache_file.exists():
            return None

        try:
            with open(cache_file, "rb") as f:
                entry = pickle.load(f)

            # Check expiration
            if entry.is_expired():
                cache_file.unlink()
                return None

            return entry.value
        except Exception:
            # Corrupted cache file - remove it
            if cache_file.exists():
                cache_file.unlink()
            return None

    def _set_disk(self, key: CacheKey, value: Any) -> None:
        """Set in disk cache."""
        cache_file = self._get_disk_path(key)

        entry = CacheEntry(
            value=value,
            created_at=datetime.now(UTC),
            ttl_seconds=self.config.ttl_seconds,
        )

        try:
            with open(cache_file, "wb") as f:
                pickle.dump(entry, f)
        except Exception:
            # Failed to cache - not critical
            pass

    def _get_disk_path(self, key: CacheKey) -> Path:
        """Get disk path for cache key."""
        return self.config.disk_path / f"{key.hash_value}.pkl"
