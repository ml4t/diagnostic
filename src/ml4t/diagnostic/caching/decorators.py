"""Decorators for automatic caching of function results."""

from __future__ import annotations

import functools
import hashlib
from collections.abc import Callable
from typing import Any, TypeVar, cast

from ml4t.diagnostic.caching.cache import Cache, CacheConfig, CacheKey

# Type variable for generic function
F = TypeVar("F", bound=Callable[..., Any])


def cache_key(*args: Any, **kwargs: Any) -> str:
    """Generate cache key from function arguments.

    Args:
        *args: Positional arguments
        **kwargs: Keyword arguments

    Returns:
        Cache key string
    """
    # Create stable representation
    key_parts = []

    # Add args
    for arg in args:
        if hasattr(arg, "__hash__"):
            try:
                key_parts.append(str(hash(arg)))
            except TypeError:
                # Not hashable - use repr
                key_parts.append(repr(arg))
        else:
            key_parts.append(repr(arg))

    # Add kwargs
    for k, v in sorted(kwargs.items()):
        if hasattr(v, "__hash__"):
            try:
                key_parts.append(f"{k}={hash(v)}")
            except TypeError:
                key_parts.append(f"{k}={repr(v)}")
        else:
            key_parts.append(f"{k}={repr(v)}")

    # Hash combined key
    combined = "|".join(key_parts)
    hasher = hashlib.sha256()
    hasher.update(combined.encode("utf-8"))

    return hasher.hexdigest()


def cached(
    cache: Cache | None = None,
    config: CacheConfig | None = None,
    key_func: Callable[..., str] | None = None,
) -> Callable[[F], F]:
    """Decorator for automatic function result caching.

    Args:
        cache: Cache instance to use (creates default if None)
        config: Cache configuration (used if cache is None)
        key_func: Custom key generation function

    Returns:
        Decorated function

    Examples:
        >>> # Use default cache
        >>> @cached()
        >>> def compute_stats(data, alpha=0.05):
        ...     return expensive_computation(data, alpha)
        >>>
        >>> # Use custom cache
        >>> cache = Cache(CacheConfig(ttl_seconds=7200))
        >>> @cached(cache=cache)
        >>> def compute_metrics(data):
        ...     return expensive_metrics(data)
        >>>
        >>> # Use custom key function
        >>> def my_key_func(data, config):
        ...     return f"{data.shape}_{config['alpha']}"
        >>>
        >>> @cached(key_func=my_key_func)
        >>> def analyze(data, config):
        ...     return analysis(data, config)
    """

    def decorator(func: F) -> F:
        # Create cache if needed - use local variable for type safety
        cache_config = config or CacheConfig()
        cache_instance: Cache = cache if cache is not None else Cache(cache_config)

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Skip cache if disabled
            if not cache_instance.config.enabled:
                return func(*args, **kwargs)

            # Generate cache key
            key_str = (
                key_func(*args, **kwargs) if key_func is not None else cache_key(*args, **kwargs)
            )

            key = CacheKey(hash_value=key_str, algorithm="sha256")

            # Check cache
            result = cache_instance.get(key)
            if result is not None:
                return result

            # Compute and cache
            result = func(*args, **kwargs)
            cache_instance.set(key, result)

            return result

        # Add cache control methods using setattr (function attributes)
        # Use setattr to bypass static type checking for dynamic attributes
        setattr(wrapper, "cache", cache_instance)  # noqa: B010
        setattr(wrapper, "cache_clear", lambda: cache_instance.clear())  # noqa: B010
        setattr(wrapper, "cache_invalidate", lambda key: cache_instance.invalidate(key))  # noqa: B010

        return cast(F, wrapper)

    return decorator
