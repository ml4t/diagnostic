"""Caching framework for ML4T Diagnostic computations.

Provides intelligent caching for expensive statistical computations
with support for both memory and disk caching.

Examples:
    >>> from ml4t.diagnostic.caching import Cache, CacheConfig
    >>>
    >>> # Create cache
    >>> cache = Cache(CacheConfig(enabled=True, ttl_seconds=3600))
    >>>
    >>> # Use cache decorator
    >>> @cache.cached()
    >>> def expensive_computation(data, config):
    ...     return compute_stats(data)
    >>>
    >>> # Or use cache directly
    >>> key = cache.generate_key(data, config)
    >>> result = cache.get(key)
    >>> if result is None:
    ...     result = expensive_computation(data, config)
    ...     cache.set(key, result)
"""

from ml4t.diagnostic.caching.cache import Cache, CacheBackend, CacheConfig, CacheKey
from ml4t.diagnostic.caching.decorators import cache_key, cached
from ml4t.diagnostic.caching.smart_cache import SmartCache

__all__ = [
    # Core cache
    "Cache",
    "CacheConfig",
    "CacheKey",
    "CacheBackend",
    # Smart cache for multi-signal analysis
    "SmartCache",
    # Decorators
    "cached",
    "cache_key",
]
