"""Caching Framework Examples

Demonstrates various caching strategies and use cases for ML4T Evaluation.

This example shows:
1. Basic caching with decorator
2. Manual cache control
3. Memory vs disk caching
4. TTL and expiration
5. LRU eviction
6. Performance benchmarking
7. Custom key functions
8. Integration patterns
"""

import time
from pathlib import Path

import polars as pl

from ml4t.diagnostic.caching import Cache, CacheBackend, CacheConfig, cached


def example_1_basic_decorator_caching():
    """Example 1: Basic caching with @cached decorator."""
    print("\n=== Example 1: Basic Decorator Caching ===")

    call_count = 0

    @cached()
    def expensive_computation(x, y):
        """Simulate expensive computation."""
        nonlocal call_count
        call_count += 1
        time.sleep(0.01)  # Simulate work
        return x + y

    # First call - computes
    print("First call...")
    result1 = expensive_computation(5, 10)
    print(f"Result: {result1}, Calls: {call_count}")

    # Second call - cached
    print("Second call (same args)...")
    result2 = expensive_computation(5, 10)
    print(f"Result: {result2}, Calls: {call_count}")  # Still 1!

    # Third call - different args
    print("Third call (different args)...")
    result3 = expensive_computation(10, 20)
    print(f"Result: {result3}, Calls: {call_count}")  # Now 2


def example_2_manual_cache_control():
    """Example 2: Manual cache get/set."""
    print("\n=== Example 2: Manual Cache Control ===")

    cache = Cache(CacheConfig(backend=CacheBackend.MEMORY))

    # Generate key
    key = cache.generate_key(data="test_data", config={"alpha": 0.05})
    print(f"Generated key: {key}")

    # Check cache
    result = cache.get(key)
    print(f"Cache get: {result}")

    if result is None:
        print("Cache miss - computing...")
        result = {"value": 42, "computed": True}
        cache.set(key, result)
        print(f"Stored in cache: {result}")

    # Get again
    result2 = cache.get(key)
    print(f"Cache hit: {result2}")


def example_3_memory_cache_with_lru():
    """Example 3: Memory cache with LRU eviction."""
    print("\n=== Example 3: Memory Cache with LRU ===")

    # Small cache for demonstration
    config = CacheConfig(backend=CacheBackend.MEMORY, max_memory_items=3)
    cache = Cache(config)

    # Fill cache
    print("Filling cache with 3 items...")
    for i in range(1, 4):
        key = cache.generate_key(data=f"item_{i}")
        cache.set(key, f"value_{i}")
        print(f"  Stored: item_{i} -> value_{i}")

    # Access item 1 (moves to end of LRU)
    key1 = cache.generate_key(data="item_1")
    _ = cache.get(key1)
    print("Accessed item_1 (now most recent)")

    # Add 4th item - should evict item_2 (oldest)
    print("Adding 4th item...")
    key4 = cache.generate_key(data="item_4")
    cache.set(key4, "value_4")

    # Check what's still cached
    for i in range(1, 5):
        key = cache.generate_key(data=f"item_{i}")
        result = cache.get(key)
        status = "✓ cached" if result else "✗ evicted"
        print(f"  item_{i}: {status}")


def example_4_disk_cache_persistence():
    """Example 4: Disk cache that persists across runs."""
    print("\n=== Example 4: Disk Cache Persistence ===")

    cache_dir = Path("example_cache")

    # First cache instance
    print("Creating first cache instance...")
    cache1 = Cache(CacheConfig(backend=CacheBackend.DISK, disk_path=cache_dir))

    key = cache1.generate_key(data="persistent_data")
    cache1.set(key, {"message": "This persists!"})
    print("Stored in disk cache")

    # Second cache instance (simulates restart)
    print("Creating second cache instance (simulating restart)...")
    cache2 = Cache(CacheConfig(backend=CacheBackend.DISK, disk_path=cache_dir))

    result = cache2.get(key)
    print(f"Retrieved from disk: {result}")

    # Cleanup
    cache2.clear()
    if cache_dir.exists():
        cache_dir.rmdir()


def example_5_ttl_expiration():
    """Example 5: Cache expiration with TTL."""
    print("\n=== Example 5: TTL Expiration ===")

    # 1 second TTL for demonstration
    config = CacheConfig(backend=CacheBackend.MEMORY, ttl_seconds=1)
    cache = Cache(config)

    key = cache.generate_key(data="expires_soon")
    cache.set(key, "temporary_value")
    print("Stored value with 1 second TTL")

    # Immediately retrieve
    result1 = cache.get(key)
    print(f"Immediately after: {result1}")

    # Wait for expiration
    print("Waiting 1.2 seconds for expiration...")
    time.sleep(1.2)

    result2 = cache.get(key)
    print(f"After expiration: {result2}")


def example_6_performance_benchmark():
    """Example 6: Performance benchmarking."""
    print("\n=== Example 6: Performance Benchmark ===")

    @cached()
    def slow_computation(n):
        """Simulate slow computation."""
        time.sleep(0.05)
        return sum(range(n))

    # Uncached call
    start = time.time()
    slow_computation(1000)
    uncached_time = time.time() - start
    print(f"Uncached: {uncached_time * 1000:.1f}ms")

    # Cached call
    start = time.time()
    slow_computation(1000)
    cached_time = time.time() - start
    print(f"Cached: {cached_time * 1000:.1f}ms")

    speedup = uncached_time / cached_time if cached_time > 0 else float("inf")
    print(f"Speedup: {speedup:.0f}x")


def example_7_custom_key_function():
    """Example 7: Custom key generation for complex types."""
    print("\n=== Example 7: Custom Key Function ===")

    def dataframe_key(df: pl.DataFrame, config: dict):
        """Generate key from DataFrame shape and config."""
        shape_str = f"{df.shape[0]}x{df.shape[1]}"
        cols_str = "_".join(df.columns)
        config_str = str(sorted(config.items()))
        return f"{shape_str}_{cols_str}_{hash(config_str)}"

    # Create sample DataFrame
    df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

    @cached(key_func=dataframe_key)
    def analyze_dataframe(df: pl.DataFrame, config: dict):
        """Analyze DataFrame with caching."""
        time.sleep(0.01)
        return {
            "shape": df.shape,
            "mean": df.select(pl.col("a").mean()).item(),
            "config": config,
        }

    # First call
    print("First analysis...")
    result1 = analyze_dataframe(df, {"method": "standard"})
    print(f"Result: {result1}")

    # Second call (cached)
    print("Second analysis (cached)...")
    result2 = analyze_dataframe(df, {"method": "standard"})
    print(f"Result: {result2}")


def example_8_cache_invalidation():
    """Example 8: Manual cache invalidation."""
    print("\n=== Example 8: Cache Invalidation ===")

    cache = Cache(CacheConfig(backend=CacheBackend.MEMORY))

    key1 = cache.generate_key(data="data1")
    key2 = cache.generate_key(data="data2")

    cache.set(key1, "value1")
    cache.set(key2, "value2")

    print(f"Cache before: key1={cache.get(key1)}, key2={cache.get(key2)}")

    # Invalidate specific key
    cache.invalidate(key1)
    print(f"After invalidate(key1): key1={cache.get(key1)}, key2={cache.get(key2)}")

    # Clear all
    cache.clear()
    print(f"After clear(): key1={cache.get(key1)}, key2={cache.get(key2)}")


def example_9_disabled_cache():
    """Example 9: Disabling cache for debugging."""
    print("\n=== Example 9: Disabled Cache ===")

    CacheConfig(enabled=True)
    config_disabled = CacheConfig(enabled=False)

    call_count = 0

    @cached(config=config_disabled)
    def func_with_disabled_cache(x):
        nonlocal call_count
        call_count += 1
        return x * 2

    # Both calls execute (no caching)
    result1 = func_with_disabled_cache(5)
    result2 = func_with_disabled_cache(5)

    print(f"Result: {result1}, {result2}")
    print(f"Calls with disabled cache: {call_count}")


def example_10_integration_pattern():
    """Example 10: Real-world integration pattern."""
    print("\n=== Example 10: Integration Pattern ===")

    # Set up persistent cache for analysis
    cache_dir = Path(".qeval_cache")
    cache = Cache(
        CacheConfig(
            backend=CacheBackend.DISK,
            disk_path=cache_dir,
            ttl_seconds=86400,  # 24 hours
        )
    )

    @cached(cache=cache)
    def run_statistical_tests(data_hash: str, test_config: dict):
        """Run expensive statistical tests with persistent caching."""
        print(f"  Computing tests for {data_hash}...")
        time.sleep(0.02)  # Simulate expensive computation

        return {
            "adf": -3.456,
            "kpss": 0.234,
            "pp": -4.123,
            "config": test_config,
        }

    # First run
    print("First run (uncached)...")
    result1 = run_statistical_tests(data_hash="abc123", test_config={"regression": "c"})
    print(f"Result: {result1}")

    # Second run (cached - instant)
    print("Second run (cached)...")
    result2 = run_statistical_tests(data_hash="abc123", test_config={"regression": "c"})
    print(f"Result: {result2}")

    # Different config (recomputes)
    print("Third run (different config)...")
    result3 = run_statistical_tests(data_hash="abc123", test_config={"regression": "ct"})
    print(f"Result: {result3}")

    # Cleanup
    cache.clear()
    if cache_dir.exists():
        cache_dir.rmdir()


def main():
    """Run all examples."""
    print("=" * 60)
    print("ML4T Evaluation Caching Framework Examples")
    print("=" * 60)

    examples = [
        example_1_basic_decorator_caching,
        example_2_manual_cache_control,
        example_3_memory_cache_with_lru,
        example_4_disk_cache_persistence,
        example_5_ttl_expiration,
        example_6_performance_benchmark,
        example_7_custom_key_function,
        example_8_cache_invalidation,
        example_9_disabled_cache,
        example_10_integration_pattern,
    ]

    for example_func in examples:
        try:
            example_func()
        except Exception as e:
            print(f"\nError in {example_func.__name__}: {e}")
            import traceback

            traceback.print_exc()

    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
