"""Tests for caching framework."""

import time

from ml4t.diagnostic.caching import Cache, CacheBackend, CacheConfig, CacheKey, cached
from ml4t.diagnostic.caching.decorators import cache_key


class TestCacheKeyFunction:
    """Test the cache_key function from decorators module."""

    def test_cache_key_basic(self):
        """Test basic cache key generation."""
        key = cache_key(1, 2, 3)
        assert isinstance(key, str)
        assert len(key) == 64  # SHA-256 hex digest

    def test_cache_key_reproducibility(self):
        """Test that same args produce same key."""
        key1 = cache_key(1, 2, 3)
        key2 = cache_key(1, 2, 3)
        assert key1 == key2

    def test_cache_key_different_args(self):
        """Test that different args produce different keys."""
        key1 = cache_key(1, 2, 3)
        key2 = cache_key(1, 2, 4)
        assert key1 != key2

    def test_cache_key_with_kwargs(self):
        """Test cache key with keyword arguments."""
        key1 = cache_key(a=1, b=2)
        key2 = cache_key(a=1, b=2)
        key3 = cache_key(a=1, b=3)

        assert key1 == key2
        assert key1 != key3

    def test_cache_key_kwargs_order_independent(self):
        """Test that kwargs order doesn't affect key."""
        key1 = cache_key(a=1, b=2, c=3)
        key2 = cache_key(c=3, b=2, a=1)
        assert key1 == key2

    def test_cache_key_mixed_args_kwargs(self):
        """Test cache key with mixed positional and keyword args."""
        key1 = cache_key(1, 2, alpha=0.05)
        key2 = cache_key(1, 2, alpha=0.05)
        key3 = cache_key(1, 2, alpha=0.10)

        assert key1 == key2
        assert key1 != key3

    def test_cache_key_unhashable_args(self):
        """Test cache key with unhashable arguments (lists, dicts)."""
        # Lists are unhashable
        key1 = cache_key([1, 2, 3])
        key2 = cache_key([1, 2, 3])
        key3 = cache_key([1, 2, 4])

        assert key1 == key2
        assert key1 != key3

    def test_cache_key_unhashable_kwargs(self):
        """Test cache key with unhashable kwarg values."""
        key1 = cache_key(data=[1, 2, 3])
        key2 = cache_key(data=[1, 2, 3])
        key3 = cache_key(data=[1, 2, 4])

        assert key1 == key2
        assert key1 != key3

    def test_cache_key_with_dict_values(self):
        """Test cache key with dict values."""
        key1 = cache_key(config={"a": 1, "b": 2})
        key2 = cache_key(config={"a": 1, "b": 2})

        assert key1 == key2

    def test_cache_key_empty_args(self):
        """Test cache key with no arguments."""
        key = cache_key()
        assert isinstance(key, str)
        assert len(key) == 64


class TestCacheKey:
    """Test cache key generation."""

    def test_generate_key_from_dict(self):
        """Test generating key from dictionary."""
        key1 = CacheKey.generate(data="test", config={"alpha": 0.05})
        key2 = CacheKey.generate(data="test", config={"alpha": 0.05})

        assert str(key1) == str(key2)

    def test_generate_key_stability(self):
        """Test that same inputs produce same key."""
        kwargs = {"a": 1, "b": 2, "c": [1, 2, 3]}

        key1 = CacheKey.generate(**kwargs)
        key2 = CacheKey.generate(**kwargs)

        assert key1 == key2

    def test_generate_key_different_values(self):
        """Test that different inputs produce different keys."""
        key1 = CacheKey.generate(data="test1")
        key2 = CacheKey.generate(data="test2")

        assert key1 != key2

    def test_generate_key_dict_order_independent(self):
        """Test that dict key order doesn't affect hash."""
        key1 = CacheKey.generate(config={"a": 1, "b": 2})
        key2 = CacheKey.generate(config={"b": 2, "a": 1})

        assert key1 == key2

    def test_cache_key_string_repr(self):
        """Test string representation."""
        key = CacheKey(hash_value="abc123", algorithm="sha256")
        assert str(key) == "sha256:abc123"

    def test_cache_key_equality(self):
        """Test equality comparison."""
        key1 = CacheKey(hash_value="abc123")
        key2 = CacheKey(hash_value="abc123")
        key3 = CacheKey(hash_value="def456")

        assert key1 == key2
        assert key1 != key3

    def test_cache_key_hashable(self):
        """Test that CacheKey can be used as dict key."""
        key1 = CacheKey(hash_value="abc123")
        key2 = CacheKey(hash_value="abc123")

        cache_dict = {key1: "value1"}
        assert cache_dict[key2] == "value1"


class TestCacheConfig:
    """Test cache configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = CacheConfig()

        assert config.enabled is True
        assert config.backend == CacheBackend.MEMORY
        assert config.ttl_seconds == 3600
        assert config.max_memory_items == 100

    def test_custom_config(self):
        """Test custom configuration."""
        config = CacheConfig(
            enabled=True,
            backend=CacheBackend.DISK,
            ttl_seconds=7200,
            max_memory_items=50,
        )

        assert config.enabled is True
        assert config.backend == CacheBackend.DISK
        assert config.ttl_seconds == 7200
        assert config.max_memory_items == 50

    def test_disabled_cache(self):
        """Test disabled cache configuration."""
        config = CacheConfig(enabled=False)
        assert config.enabled is False


class TestMemoryCache:
    """Test memory-based caching."""

    def test_basic_get_set(self):
        """Test basic get/set operations."""
        cache = Cache(CacheConfig(backend=CacheBackend.MEMORY))

        key = cache.generate_key(data="test")
        value = {"result": 42}

        # Initially empty
        assert cache.get(key) is None

        # Set and get
        cache.set(key, value)
        assert cache.get(key) == value

    def test_cache_miss(self):
        """Test cache miss returns None."""
        cache = Cache(CacheConfig(backend=CacheBackend.MEMORY))
        key = cache.generate_key(data="nonexistent")

        assert cache.get(key) is None

    def test_lru_eviction(self):
        """Test LRU eviction when cache is full."""
        config = CacheConfig(backend=CacheBackend.MEMORY, max_memory_items=3)
        cache = Cache(config)

        # Fill cache
        key1 = cache.generate_key(data="1")
        key2 = cache.generate_key(data="2")
        key3 = cache.generate_key(data="3")

        cache.set(key1, "value1")
        cache.set(key2, "value2")
        cache.set(key3, "value3")

        # All should be present
        assert cache.get(key1) == "value1"
        assert cache.get(key2) == "value2"
        assert cache.get(key3) == "value3"

        # Add 4th item - should evict key1 (oldest)
        key4 = cache.generate_key(data="4")
        cache.set(key4, "value4")

        assert cache.get(key1) is None  # Evicted
        assert cache.get(key2) == "value2"
        assert cache.get(key3) == "value3"
        assert cache.get(key4) == "value4"

    def test_lru_access_order(self):
        """Test that accessing an item moves it to end (LRU)."""
        config = CacheConfig(backend=CacheBackend.MEMORY, max_memory_items=2)
        cache = Cache(config)

        key1 = cache.generate_key(data="1")
        key2 = cache.generate_key(data="2")

        cache.set(key1, "value1")
        cache.set(key2, "value2")

        # Access key1 (moves to end)
        _ = cache.get(key1)

        # Add key3 - should evict key2 (now oldest)
        key3 = cache.generate_key(data="3")
        cache.set(key3, "value3")

        assert cache.get(key1) == "value1"  # Still present
        assert cache.get(key2) is None  # Evicted
        assert cache.get(key3) == "value3"

    def test_ttl_expiration(self):
        """Test that entries expire after TTL."""
        config = CacheConfig(backend=CacheBackend.MEMORY, ttl_seconds=1)
        cache = Cache(config)

        key = cache.generate_key(data="test")
        cache.set(key, "value")

        # Should be present immediately
        assert cache.get(key) == "value"

        # Wait for expiration
        time.sleep(1.1)

        # Should be expired
        assert cache.get(key) is None

    def test_no_ttl_expiration(self):
        """Test that entries don't expire with None TTL."""
        config = CacheConfig(backend=CacheBackend.MEMORY, ttl_seconds=None)
        cache = Cache(config)

        key = cache.generate_key(data="test")
        cache.set(key, "value")

        # Should persist indefinitely
        time.sleep(0.5)
        assert cache.get(key) == "value"

    def test_invalidate(self):
        """Test invalidating specific cache entry."""
        cache = Cache(CacheConfig(backend=CacheBackend.MEMORY))

        key = cache.generate_key(data="test")
        cache.set(key, "value")

        assert cache.get(key) == "value"

        cache.invalidate(key)
        assert cache.get(key) is None

    def test_clear(self):
        """Test clearing all cache entries."""
        cache = Cache(CacheConfig(backend=CacheBackend.MEMORY))

        key1 = cache.generate_key(data="1")
        key2 = cache.generate_key(data="2")

        cache.set(key1, "value1")
        cache.set(key2, "value2")

        cache.clear()

        assert cache.get(key1) is None
        assert cache.get(key2) is None


class TestDiskCache:
    """Test disk-based caching."""

    def test_basic_get_set(self, tmp_path):
        """Test basic disk cache operations."""
        config = CacheConfig(backend=CacheBackend.DISK, disk_path=tmp_path / "cache")
        cache = Cache(config)

        key = cache.generate_key(data="test")
        value = {"result": 42}

        cache.set(key, value)
        assert cache.get(key) == value

    def test_persistence_across_instances(self, tmp_path):
        """Test that disk cache persists across cache instances."""
        cache_dir = tmp_path / "cache"
        config = CacheConfig(backend=CacheBackend.DISK, disk_path=cache_dir)

        # First cache instance
        cache1 = Cache(config)
        key = cache1.generate_key(data="test")
        cache1.set(key, "value1")

        # Second cache instance (same directory)
        cache2 = Cache(config)
        assert cache2.get(key) == "value1"

    def test_ttl_expiration_disk(self, tmp_path):
        """Test TTL expiration for disk cache."""
        config = CacheConfig(
            backend=CacheBackend.DISK,
            disk_path=tmp_path / "cache",
            ttl_seconds=1,
        )
        cache = Cache(config)

        key = cache.generate_key(data="test")
        cache.set(key, "value")

        # Should be present
        assert cache.get(key) == "value"

        # Wait for expiration
        time.sleep(1.1)

        # Should be expired and file deleted
        assert cache.get(key) is None

    def test_corrupted_cache_file(self, tmp_path):
        """Test handling of corrupted cache files."""
        config = CacheConfig(backend=CacheBackend.DISK, disk_path=tmp_path / "cache")
        cache = Cache(config)

        key = cache.generate_key(data="test")

        # Write corrupted cache file
        cache_file = config.disk_path / f"{key.hash_value}.pkl"
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text("corrupted data")

        # Should return None and clean up
        assert cache.get(key) is None
        assert not cache_file.exists()

    def test_clear_disk_cache(self, tmp_path):
        """Test clearing disk cache."""
        config = CacheConfig(backend=CacheBackend.DISK, disk_path=tmp_path / "cache")
        cache = Cache(config)

        key1 = cache.generate_key(data="1")
        key2 = cache.generate_key(data="2")

        cache.set(key1, "value1")
        cache.set(key2, "value2")

        cache.clear()

        assert cache.get(key1) is None
        assert cache.get(key2) is None


class TestDisabledCache:
    """Test disabled cache behavior."""

    def test_disabled_cache_no_caching(self):
        """Test that disabled cache doesn't store anything."""
        config = CacheConfig(enabled=False)
        cache = Cache(config)

        key = cache.generate_key(data="test")
        cache.set(key, "value")

        # Should not cache
        assert cache.get(key) is None


class TestCachedDecorator:
    """Test cached decorator."""

    def test_basic_caching(self):
        """Test basic function result caching."""
        call_count = 0

        @cached()
        def expensive_func(x):
            nonlocal call_count
            call_count += 1
            return x * 2

        # First call - computes
        result1 = expensive_func(5)
        assert result1 == 10
        assert call_count == 1

        # Second call - cached
        result2 = expensive_func(5)
        assert result2 == 10
        assert call_count == 1  # Not called again

        # Different args - computes
        result3 = expensive_func(10)
        assert result3 == 20
        assert call_count == 2

    def test_cached_with_custom_cache(self):
        """Test decorator with custom cache instance."""
        config = CacheConfig(backend=CacheBackend.MEMORY, ttl_seconds=7200)
        cache = Cache(config)

        call_count = 0

        @cached(cache=cache)
        def func(x):
            nonlocal call_count
            call_count += 1
            return x * 2

        func(5)
        func(5)

        assert call_count == 1

    def test_cached_disabled(self):
        """Test that disabled cache doesn't cache."""
        config = CacheConfig(enabled=False)
        cache = Cache(config)

        call_count = 0

        @cached(cache=cache)
        def func(x):
            nonlocal call_count
            call_count += 1
            return x * 2

        func(5)
        func(5)

        assert call_count == 2  # Called twice

    def test_cache_clear_method(self):
        """Test cache_clear method on decorated function."""
        call_count = 0

        @cached()
        def func(x):
            nonlocal call_count
            call_count += 1
            return x * 2

        func(5)
        assert call_count == 1

        func(5)
        assert call_count == 1  # Cached

        func.cache_clear()

        func(5)
        assert call_count == 2  # Recomputed after clear

    def test_cached_with_custom_key_func(self):
        """Test decorator with custom key function."""
        call_count = 0

        def custom_key(data, config):
            """Only use config for cache key."""
            return f"config_{config['alpha']}"

        @cached(key_func=custom_key)
        def func(data, config):
            nonlocal call_count
            call_count += 1
            return len(data) * config["alpha"]

        # Different data, same config -> same cache key
        result1 = func([1, 2, 3], {"alpha": 0.05})
        result2 = func([1, 2, 3, 4, 5], {"alpha": 0.05})  # Different data

        assert call_count == 1  # Second call used cache despite different data
        assert result2 == result1  # Returns cached result

        # Same data, different config -> different cache key
        func([1, 2, 3], {"alpha": 0.10})
        assert call_count == 2  # New computation

    def test_cached_with_config_only(self):
        """Test decorator with only config parameter."""
        call_count = 0

        config = CacheConfig(ttl_seconds=3600)

        @cached(config=config)
        def func(x):
            nonlocal call_count
            call_count += 1
            return x * 2

        func(5)
        func(5)

        assert call_count == 1  # Cached

    def test_cache_access_attribute(self):
        """Test access to cache attribute on decorated function."""
        custom_cache = Cache(CacheConfig())

        @cached(cache=custom_cache)
        def func(x):
            return x * 2

        # Cache should be accessible
        assert func.cache is custom_cache

    def test_cache_invalidate_method(self):
        """Test cache_invalidate method on decorated function."""
        call_count = 0

        @cached()
        def func(x):
            nonlocal call_count
            call_count += 1
            return x * 2

        # Call to populate cache
        func(5)
        assert call_count == 1

        # Second call uses cache
        func(5)
        assert call_count == 1

        # Invalidate a specific key (use CacheKey)
        key = CacheKey(hash_value=cache_key(5), algorithm="sha256")
        func.cache_invalidate(key)

        # Now calling should recompute
        func(5)
        assert call_count == 2

    def test_cached_preserves_function_metadata(self):
        """Test that decorator preserves function name and docstring."""

        @cached()
        def my_function(x):
            """This is the docstring."""
            return x * 2

        assert my_function.__name__ == "my_function"
        assert my_function.__doc__ == "This is the docstring."

    def test_cached_with_kwargs_only(self):
        """Test caching with kwargs-only function calls."""
        call_count = 0

        @cached()
        def func(a=None, b=None):
            nonlocal call_count
            call_count += 1
            return (a or 0) + (b or 0)

        func(a=1, b=2)
        func(a=1, b=2)
        func(b=2, a=1)  # Same args, different order

        assert call_count == 1  # All should hit same cache entry


class TestCachePerformance:
    """Test cache performance improvements."""

    def test_speedup_with_cache(self):
        """Test that cache provides speedup for repeated calls."""
        config = CacheConfig(backend=CacheBackend.MEMORY)
        cache = Cache(config)

        call_times = []

        @cached(cache=cache)
        def slow_func(x):
            """Simulate slow computation."""
            time.sleep(0.01)
            return x * 2

        # First call (uncached)
        start = time.time()
        result1 = slow_func(5)
        first_time = time.time() - start
        call_times.append(first_time)

        # Second call (cached)
        start = time.time()
        result2 = slow_func(5)
        second_time = time.time() - start
        call_times.append(second_time)

        # Cached call should be much faster
        assert result1 == result2
        assert second_time < first_time * 0.5  # At least 2x speedup


class TestIntegration:
    """Integration tests for caching framework."""

    def test_memory_and_disk_cache_independent(self, tmp_path):
        """Test that memory and disk caches are independent."""
        mem_cache = Cache(CacheConfig(backend=CacheBackend.MEMORY))
        disk_cache = Cache(CacheConfig(backend=CacheBackend.DISK, disk_path=tmp_path / "cache"))

        key = CacheKey.generate(data="test")

        mem_cache.set(key, "memory_value")
        disk_cache.set(key, "disk_value")

        assert mem_cache.get(key) == "memory_value"
        assert disk_cache.get(key) == "disk_value"


# SmartCache tests - for Polars DataFrame fingerprinting cache
from unittest.mock import MagicMock

import polars as pl

from ml4t.diagnostic.caching.smart_cache import SmartCache


class TestSmartCacheInit:
    """Tests for SmartCache initialization."""

    def test_default_init(self):
        """Test initialization with defaults."""
        cache = SmartCache()
        assert cache.max_items == 100
        assert cache.ttl_seconds == 3600
        assert cache.size == 0

    def test_custom_init(self):
        """Test initialization with custom parameters."""
        cache = SmartCache(max_items=50, ttl_seconds=7200)
        assert cache.max_items == 50
        assert cache.ttl_seconds == 7200

    def test_no_ttl_init(self):
        """Test initialization without TTL."""
        cache = SmartCache(ttl_seconds=None)
        assert cache.ttl_seconds is None


class TestPolarsFingerprint:
    """Tests for polars_fingerprint static method."""

    def test_identical_dataframes(self):
        """Test fingerprints are identical for identical DataFrames."""
        df1 = pl.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
        df2 = pl.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
        fp1 = SmartCache.polars_fingerprint(df1)
        fp2 = SmartCache.polars_fingerprint(df2)
        assert fp1 == fp2

    def test_different_data(self):
        """Test fingerprints differ for different data."""
        df1 = pl.DataFrame({"a": [1, 2, 3]})
        df2 = pl.DataFrame({"a": [1, 2, 4]})
        fp1 = SmartCache.polars_fingerprint(df1)
        fp2 = SmartCache.polars_fingerprint(df2)
        assert fp1 != fp2

    def test_different_schema(self):
        """Test fingerprints differ for different schemas."""
        df1 = pl.DataFrame({"a": [1, 2, 3]})
        df2 = pl.DataFrame({"b": [1, 2, 3]})
        fp1 = SmartCache.polars_fingerprint(df1)
        fp2 = SmartCache.polars_fingerprint(df2)
        assert fp1 != fp2


class TestSmartCacheMakeKey:
    """Tests for make_key method."""

    def test_basic_key_generation(self):
        """Test basic key generation."""
        cache = SmartCache()
        df = pl.DataFrame({"a": [1, 2, 3]})
        mock_config = MagicMock()
        mock_config.model_dump_json.return_value = '{"param": "value"}'
        key = cache.make_key("momentum", df, mock_config)
        assert "momentum" in key
        parts = key.split("_")
        assert len(parts) == 3

    def test_same_inputs_same_key(self):
        """Test same inputs produce same key."""
        cache = SmartCache()
        df = pl.DataFrame({"a": [1, 2, 3]})
        mock_config = MagicMock()
        mock_config.model_dump_json.return_value = '{"param": "value"}'
        key1 = cache.make_key("test", df, mock_config)
        key2 = cache.make_key("test", df, mock_config)
        assert key1 == key2


class TestSmartCacheOperations:
    """Tests for basic SmartCache operations."""

    def test_set_and_get(self):
        """Test basic set and get operations."""
        cache = SmartCache()
        cache.set("key1", "value1")
        result = cache.get("key1")
        assert result == "value1"

    def test_get_missing_key(self):
        """Test getting a missing key returns None."""
        cache = SmartCache()
        result = cache.get("nonexistent")
        assert result is None

    def test_eviction_when_full(self):
        """Test oldest items are evicted when cache is full."""
        cache = SmartCache(max_items=3, ttl_seconds=None)
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")
        cache.set("key4", "value4")
        assert cache.get("key1") is None
        assert cache.get("key4") == "value4"
        assert cache.size == 3

    def test_invalidate_signal(self):
        """Test invalidating all entries for a signal."""
        cache = SmartCache()
        cache.set("momentum_abc123_config1", "result1")
        cache.set("momentum_def456_config2", "result2")
        cache.set("value_abc123_config1", "result3")
        removed = cache.invalidate_signal("momentum")
        assert removed == 2
        assert "value_abc123_config1" in cache


class TestSmartCacheStatistics:
    """Tests for SmartCache statistics."""

    def test_hit_rate_calculation(self):
        """Test hit rate calculation."""
        import pytest

        cache = SmartCache(ttl_seconds=None)
        cache.set("key1", "value1")
        cache.get("key1")
        cache.get("missing1")
        cache.get("missing2")
        assert cache.hit_rate == pytest.approx(1 / 3)

    def test_stats_property(self):
        """Test stats property returns correct structure."""
        cache = SmartCache(max_items=100, ttl_seconds=3600)
        cache.set("key1", "value1")
        cache.get("key1")
        stats = cache.stats
        assert "hits" in stats
        assert "misses" in stats
        assert "size" in stats
        assert stats["hits"] == 1
