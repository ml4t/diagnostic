"""Core caching implementation with pluggable backends."""

from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class CacheBackend(str, Enum):
    """Cache storage backend options."""

    MEMORY = "memory"
    DISABLED = "disabled"


class CacheConfig(BaseModel):
    """Configuration for cache behavior.

    Attributes:
        enabled: Whether caching is enabled
        backend: Storage backend to use
        ttl_seconds: Time-to-live for cache entries (None = no expiration)
        max_memory_items: Max items in memory cache (LRU eviction)
    """

    enabled: bool = True
    backend: CacheBackend = CacheBackend.MEMORY
    ttl_seconds: int | None = Field(default=3600, description="Cache TTL in seconds")
    max_memory_items: int = Field(default=100, description="Max memory cache size")


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
    """In-memory cache for expensive computations.

    Supports automatic expiration, LRU eviction, and explicit disabling.

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

        if self.config.backend != CacheBackend.MEMORY:
            return None
        return self._get_memory(key)

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

    def invalidate(self, key: CacheKey) -> None:
        """Invalidate specific cache entry.

        Args:
            key: Cache key to invalidate
        """
        self._memory_cache.pop(key, None)

    def clear(self) -> None:
        """Clear all cache entries."""
        self._memory_cache.clear()

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
