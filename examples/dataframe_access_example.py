"""Example: DataFrame Access API (FR-I4)

Demonstrates programmatic DataFrame access for ML4T Backtest integration workflows.

Examples:
1. Basic DataFrame access
2. Schema discovery
3. Multiple results aggregation
4. ML4T Backtest storage workflow
5. Export to multiple formats

Author: ML4T Evaluation Team
Date: 2025-11-03
"""

import polars as pl

from ml4t.diagnostic.results.feature_results import ACFResult, StationarityTestResult


def example_1_basic_dataframe_access():
    """Example 1: Basic DataFrame access from results."""
    print("=" * 70)
    print("Example 1: Basic DataFrame Access")
    print("=" * 70)

    # Create stationarity test result
    result = StationarityTestResult(
        feature_name="btc_returns",
        adf_statistic=-3.52,
        adf_pvalue=0.008,
        adf_is_stationary=True,
        kpss_statistic=0.18,
        kpss_pvalue=0.10,
        kpss_is_stationary=True,
        pp_statistic=-3.48,
        pp_pvalue=0.009,
        pp_is_stationary=True,
    )

    print(f"\n📊 Result: {result.feature_name}")
    print(f"   Analysis: {result.analysis_type}")
    print(f"   Created: {result.created_at}")

    # Get as DataFrame
    df = result.get_dataframe()
    print(f"\n📋 DataFrame shape: {df.shape}")
    print(f"   Columns: {df.columns}")
    print(f"\n{df}")

    # Access specific columns
    print("\n🔍 Stationarity Results:")
    print(f"   ADF: {df['adf_stationary'][0]} (p={df['adf_pvalue'][0]:.4f})")
    print(f"   KPSS: {df['kpss_stationary'][0]} (p={df['kpss_pvalue'][0]:.4f})")
    print(f"   PP: {df['pp_stationary'][0]} (p={df['pp_pvalue'][0]:.4f})")

    return result


def example_2_schema_discovery():
    """Example 2: Schema discovery before loading data."""
    print("\n" + "=" * 70)
    print("Example 2: Schema Discovery")
    print("=" * 70)

    result = StationarityTestResult(
        feature_name="eth_returns",
        adf_statistic=-2.85,
        adf_pvalue=0.05,
        adf_is_stationary=False,
    )

    # 1. Discover available DataFrames
    available = result.list_available_dataframes()
    print(f"\n📚 Available DataFrames: {available}")

    # 2. Get schema without loading data
    schema = result.get_dataframe_schema()
    print("\n📐 Schema:")
    for col, dtype in schema.items():
        print(f"   {col}: {dtype}")

    # 3. Now load DataFrame knowing structure
    df = result.get_dataframe("primary")
    print(f"\n✅ DataFrame loaded: {df.shape}")
    print(f"   Columns match schema: {set(df.columns) == set(schema.keys())}")

    return schema


def example_3_multiple_results_aggregation():
    """Example 3: Aggregate DataFrames from multiple features."""
    print("\n" + "=" * 70)
    print("Example 3: Multiple Results Aggregation")
    print("=" * 70)

    # Create results for multiple features
    features = ["btc_returns", "eth_returns", "sol_returns", "ada_returns"]
    results = []

    for i, feature in enumerate(features):
        result = StationarityTestResult(
            feature_name=feature,
            adf_statistic=-3.5 + i * 0.3,
            adf_pvalue=0.01 + i * 0.02,
            adf_is_stationary=(0.01 + i * 0.02) < 0.05,
        )
        results.append(result)

    print(f"\n📦 Created {len(results)} results")

    # Aggregate into single DataFrame
    dfs = [r.get_dataframe() for r in results]
    combined = pl.concat(dfs)

    print(f"\n🔗 Combined DataFrame: {combined.shape}")
    print(f"\n{combined.select(['feature', 'adf_pvalue', 'adf_stationary'])}")

    # Summary statistics
    stationary_count = combined["adf_stationary"].sum()
    print("\n📊 Summary:")
    print(f"   Total features: {len(combined)}")
    print(f"   Stationary: {stationary_count}")
    print(f"   Non-stationary: {len(combined) - stationary_count}")

    return combined


def example_4_qengine_storage_workflow():
    """Example 4: Complete ML4T Backtest storage workflow."""
    print("\n" + "=" * 70)
    print("Example 4: ML4T Backtest Storage Workflow")
    print("=" * 70)

    result = StationarityTestResult(
        feature_name="btc_returns",
        adf_statistic=-3.52,
        adf_pvalue=0.008,
        adf_is_stationary=True,
    )

    print("\n📝 Workflow Steps:")

    # Step 1: Discover available DataFrames
    available = result.list_available_dataframes()
    print(f"\n1️⃣  Discovered {len(available)} DataFrame(s): {available}")

    # Step 2: Get schema for storage preparation
    schema = result.get_dataframe_schema()
    print(f"\n2️⃣  Schema retrieved: {len(schema)} columns")

    # Simulate ML4T Backtest table creation
    qengine_table_def = {
        "table_name": f"{result.analysis_type}_results",
        "columns": schema,
        "primary_key": "feature",
    }
    print(f"   ML4T Backtest table: {qengine_table_def['table_name']}")

    # Step 3: Load DataFrame
    df = result.get_dataframe()
    print(f"\n3️⃣  DataFrame loaded: {df.shape}")

    # Step 4: Convert to storage format
    storage_dicts = df.to_dicts()
    print(f"\n4️⃣  Converted to dicts: {len(storage_dicts)} records")
    print(f"   Sample: {storage_dicts[0]}")

    # Simulate ML4T Backtest storage
    storage_payload = {
        "table": qengine_table_def["table_name"],
        "data": storage_dicts,
        "metadata": result.to_dict(),
        "schema": schema,
    }

    print("\n💾 Ready for ML4T Backtest:")
    print(f"   Table: {storage_payload['table']}")
    print(f"   Records: {len(storage_payload['data'])}")
    print(f"   Metadata: {len(storage_payload['metadata'])} fields")
    print("\n   Simulated: ml4t.backtest.store(storage_payload)")

    return storage_payload


def example_5_export_multiple_formats():
    """Example 5: Export to multiple formats."""
    print("\n" + "=" * 70)
    print("Example 5: Multi-Format Export")
    print("=" * 70)

    result = ACFResult(
        feature_name="btc_returns",
        acf_values=[1.0, 0.52, 0.28, 0.15, 0.08, 0.04],
        pacf_values=[1.0, 0.52, 0.05, 0.02, 0.01, 0.00],
        significant_lags_acf=[1, 2, 3],
        significant_lags_pacf=[1],
    )

    print(f"\n📊 Result: {result.feature_name} ACF/PACF")

    # Format 1: JSON (metadata + data)
    json_str = result.to_json_string()
    json_size = len(json_str)
    print(f"\n1️⃣  JSON export: {json_size:,} bytes")
    print("   Contains: metadata + data in single file")

    # Format 2: DataFrame (efficient columnar format)
    df = result.get_dataframe()
    print(f"\n2️⃣  DataFrame export: {df.shape}")
    print(f"\n{df}")

    # Format 3: Dictionary (for API responses)
    dict_export = result.to_dict()
    print(f"\n3️⃣  Dictionary export: {len(dict_export)} keys")
    print(f"   Keys: {list(dict_export.keys())[:5]}...")

    # Format 4: Parquet (compressed storage)
    # Simulate parquet export by writing to BytesIO
    from io import BytesIO

    buffer = BytesIO()
    df.write_parquet(buffer)
    parquet_bytes = len(buffer.getvalue())
    print(f"\n4️⃣  Parquet export: {parquet_bytes:,} bytes (compressed)")

    # Format 5: CSV (human-readable)
    csv_str = df.write_csv()
    csv_size = len(csv_str)
    print(f"\n5️⃣  CSV export: {csv_size:,} bytes")

    print("\n📦 Format Comparison:")
    print(f"   Parquet: {parquet_bytes:,} bytes (most efficient)")
    print(f"   CSV: {csv_size:,} bytes")
    print(f"   JSON: {json_size:,} bytes (includes metadata)")

    return {
        "json": json_str,
        "dataframe": df,
        "dict": dict_export,
        "parquet_bytes": parquet_bytes,
        "csv": csv_str,
    }


def example_6_polars_transformations():
    """Example 6: Apply Polars transformations."""
    print("\n" + "=" * 70)
    print("Example 6: Polars Transformations")
    print("=" * 70)

    # Create multiple results
    results = [
        StationarityTestResult(
            feature_name=f"feature_{i}",
            adf_statistic=-3.5 + i * 0.2,
            adf_pvalue=0.01 + i * 0.015,
            adf_is_stationary=(0.01 + i * 0.015) < 0.05,
        )
        for i in range(6)
    ]

    # Combine
    df = pl.concat([r.get_dataframe() for r in results])
    print(f"\n📊 Combined DataFrame: {df.shape}")

    # Transformation 1: Filter stationary features
    stationary = df.filter(pl.col("adf_stationary"))
    print(f"\n1️⃣  Stationary features: {len(stationary)}/{len(df)}")
    print(f"\n{stationary.select(['feature', 'adf_pvalue'])}")

    # Transformation 2: Add significance level
    df_with_sig = df.with_columns(
        pl.when(pl.col("adf_pvalue") < 0.01)
        .then(pl.lit("***"))
        .when(pl.col("adf_pvalue") < 0.05)
        .then(pl.lit("**"))
        .when(pl.col("adf_pvalue") < 0.10)
        .then(pl.lit("*"))
        .otherwise(pl.lit(""))
        .alias("significance")
    )
    print("\n2️⃣  With significance levels:")
    print(f"\n{df_with_sig.select(['feature', 'adf_pvalue', 'significance'])}")

    # Transformation 3: Summary statistics
    summary = df.select(
        [
            pl.col("adf_pvalue").mean().alias("avg_pvalue"),
            pl.col("adf_pvalue").std().alias("std_pvalue"),
            pl.col("adf_pvalue").min().alias("min_pvalue"),
            pl.col("adf_pvalue").max().alias("max_pvalue"),
            pl.col("adf_stationary").sum().alias("stationary_count"),
        ]
    )
    print("\n3️⃣  Summary Statistics:")
    print(f"\n{summary}")

    return df_with_sig


if __name__ == "__main__":
    print("DataFrame Access API - Complete Examples\n")

    # Run all examples
    result1 = example_1_basic_dataframe_access()
    schema = example_2_schema_discovery()
    combined = example_3_multiple_results_aggregation()
    storage = example_4_qengine_storage_workflow()
    formats = example_5_export_multiple_formats()
    transformed = example_6_polars_transformations()

    print("\n" + "=" * 70)
    print("✅ All examples completed successfully!")
    print("=" * 70)
    print("\n📊 Summary:")
    print("   Results created: 11+")
    print("   DataFrames loaded: 15+")
    print("   Formats demonstrated: 5 (JSON, DataFrame, Dict, Parquet, CSV)")
    print("   ML4T Backtest workflow: Complete")
    print("   Polars transformations: 3 demonstrated")
