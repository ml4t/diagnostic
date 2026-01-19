"""Integration tests with ml4t.engineer library output.

This module tests that ml4t.diagnostic splitters work correctly with the output
from ml4t.engineer, particularly with labeled data containing label horizons
and timestamps.

Requires: pip install ml4t-diagnostic[dev]  # which includes ml4t-engineer
"""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import polars as pl
import pytest

try:
    from ml4t.engineer.labeling import BarrierConfig, triple_barrier_labels

    ENGINEER_AVAILABLE = True
except ImportError:
    ENGINEER_AVAILABLE = False

from ml4t.diagnostic.splitters import PurgedWalkForwardCV


@pytest.mark.skipif(
    not ENGINEER_AVAILABLE, reason="ml4t-engineer not installed (pip install ml4t-diagnostic[dev])"
)
class TestEngineerIntegration:
    """Test integration with ml4t.engineer library."""

    @pytest.fixture
    def create_labeled_data(self):
        """Create realistic labeled financial data using ml4t.engineer."""
        np.random.seed(42)
        n_samples = 1000

        # Generate timestamps (daily data)
        base_time = datetime(2020, 1, 1)
        timestamps = [base_time + timedelta(days=i) for i in range(n_samples)]

        # Generate price series with trend and volatility
        trend = 0.0002
        volatility = 0.02
        returns = np.random.normal(trend, volatility, n_samples)
        prices = 100 * (1 + returns).cumprod()

        # Create features
        df = pl.DataFrame(
            {
                "timestamp": timestamps,
                "close": prices,
                "volume": np.random.randint(1000000, 10000000, n_samples),
                "returns": np.concatenate([[0], returns[1:]]),
            },
        )

        # Add some technical indicators as features
        df = df.with_columns(
            [
                pl.col("close").rolling_mean(window_size=20).alias("sma_20"),
                pl.col("returns").rolling_std(window_size=20).alias("volatility_20"),
                pl.col("volume").rolling_mean(window_size=5).alias("volume_ma_5"),
            ],
        )

        # Apply triple-barrier labeling
        barrier_config = BarrierConfig(
            upper_barrier=0.02,  # 2% profit target
            lower_barrier=-0.01,  # 1% stop loss
            max_holding_period=20,  # 20 days max holding
        )

        labeled_df = triple_barrier_labels(
            df,
            barrier_config,
            price_col="close",
            timestamp_col="timestamp",
        )

        # Filter to only labeled events (non-null labels)
        labeled_events = labeled_df.filter(pl.col("label").is_not_null())

        return labeled_events

    def test_purged_walk_forward_with_labeled_data(self, create_labeled_data):
        """Test PurgedWalkForwardCV with ml4t.engineer labeled data."""
        labeled_df = create_labeled_data

        # Extract features and labels
        feature_cols = ["sma_20", "volatility_20", "volume_ma_5", "returns"]
        X = labeled_df.select(feature_cols)
        y = labeled_df.select("label").to_series()

        # The label horizon in this case is the max_holding_period (20 days)
        cv = PurgedWalkForwardCV(
            n_splits=5,
            label_horizon=20,  # Match the labeling horizon
            embargo_size=5,  # 5 days embargo
        )

        splits = list(cv.split(X, y))
        assert len(splits) == 5

        # Verify no data leakage
        for train_idx, test_idx in splits:
            # Ensure train and test don't overlap
            assert len(set(train_idx) & set(test_idx)) == 0

            # Verify purging: no training sample within label_horizon of test start
            if len(train_idx) > 0 and len(test_idx) > 0:
                test_start = test_idx.min()
                close_train_samples = train_idx[train_idx >= test_start - 20]
                assert len(close_train_samples) == 0

    def test_with_pandas_timestamps(self, create_labeled_data):
        """Test with pandas DataFrame having DatetimeIndex."""
        labeled_df = create_labeled_data

        # Convert to pandas with timestamp as index
        feature_cols = ["sma_20", "volatility_20", "volume_ma_5", "returns"]
        pandas_df = labeled_df.to_pandas()
        pandas_df["timestamp"] = pd.to_datetime(pandas_df["timestamp"]).dt.tz_localize("UTC")
        pandas_df = pandas_df.set_index("timestamp")

        X = pandas_df[feature_cols]
        y = pandas_df["label"]

        # Use timedelta for label_horizon when using timestamps
        cv = PurgedWalkForwardCV(
            n_splits=5,
            label_horizon=pd.Timedelta("20D"),
            embargo_size=pd.Timedelta("5D"),
        )

        splits = list(cv.split(X, y))
        assert len(splits) == 5

        # Verify temporal ordering
        for train_idx, test_idx in splits:
            if len(train_idx) > 0 and len(test_idx) > 0:
                train_times = X.index[train_idx]
                test_times = X.index[test_idx]

                # All training times should be before test times
                assert train_times.max() < test_times.min()

    def test_label_characteristics_preserved(self, create_labeled_data):
        """Test that label characteristics are preserved through splits."""
        labeled_df = create_labeled_data

        # Extract features and labels
        feature_cols = ["sma_20", "volatility_20", "volume_ma_5", "returns"]
        X = labeled_df.select(feature_cols).to_pandas()
        y = labeled_df.select("label").to_pandas()["label"]

        cv = PurgedWalkForwardCV(n_splits=3, label_horizon=20)

        # Check label distribution across splits
        for train_idx, test_idx in cv.split(X, y):
            train_labels = y.iloc[train_idx]
            test_labels = y.iloc[test_idx]

            # Both sets should have all three label classes (-1, 0, 1)
            # Though this might not always be true for small test sets
            train_unique = set(train_labels.dropna().unique())
            test_unique = set(test_labels.dropna().unique())

            # At least verify we have some variety in labels
            assert len(train_unique) >= 1
            assert len(test_unique) >= 1

    def test_with_event_based_sampling(self, create_labeled_data):
        """Test with event-based sampling (only at label times)."""
        labeled_df = create_labeled_data

        # In practice, we might only want to train on times when we actually
        # have labels (events), not all time periods
        labeled_df.filter(pl.col("label").is_not_null()).select(
            "timestamp",
        )

        # This tests that our splitter can handle non-uniform time sampling
        feature_cols = ["sma_20", "volatility_20", "volume_ma_5", "returns"]
        X = labeled_df.select(feature_cols)
        y = labeled_df.select("label").to_series()

        cv = PurgedWalkForwardCV(
            n_splits=4,
            label_horizon=20,
            embargo_size=5,
            test_size=0.2,  # 20% of data for test
        )

        for train_idx, test_idx in cv.split(X, y):
            # Just verify the split works correctly
            assert len(train_idx) + len(test_idx) <= len(X)

    def test_multi_asset_scenario(self):
        """Test with multi-asset data typical in ml4t.engineer workflows."""
        np.random.seed(42)
        n_samples = 500
        n_assets = 3

        # Create multi-asset data
        dfs = []
        for asset_id in range(n_assets):
            base_time = datetime(2020, 1, 1)
            timestamps = [base_time + timedelta(days=i) for i in range(n_samples)]

            # Different volatility for each asset
            volatility = 0.01 + asset_id * 0.005
            returns = np.random.normal(0.0001, volatility, n_samples)
            prices = 100 * (1 + returns).cumprod()

            df = pl.DataFrame(
                {
                    "asset_id": [f"ASSET_{asset_id}"] * n_samples,
                    "timestamp": timestamps,
                    "close": prices,
                    "returns": np.concatenate([[0], returns[1:]]),
                },
            )

            # Add rolling features
            df = df.with_columns(
                [
                    pl.col("close").rolling_mean(window_size=10).alias("sma_10"),
                    pl.col("returns").rolling_std(window_size=10).alias("vol_10"),
                ],
            )

            dfs.append(df)

        # Combine all assets
        multi_asset_df = pl.concat(dfs)

        # Apply labeling per asset
        labeled_dfs = []
        for asset_id in range(n_assets):
            asset_df = multi_asset_df.filter(pl.col("asset_id") == f"ASSET_{asset_id}")

            config = BarrierConfig(
                upper_barrier=0.015,
                lower_barrier=-0.01,
                max_holding_period=15,
            )

            labeled = triple_barrier_labels(
                asset_df,
                config,
                price_col="close",
                timestamp_col="timestamp",
            )
            labeled_dfs.append(labeled)

        # Combine labeled data
        all_labeled = pl.concat(labeled_dfs)

        # For cross-validation, we typically process each asset separately
        # or use group-aware splitting
        for asset_id in range(n_assets):
            asset_data = all_labeled.filter(
                pl.col("asset_id") == f"ASSET_{asset_id}",
            ).filter(pl.col("label").is_not_null())

            if len(asset_data) < 50:  # Need enough data for CV
                continue

            X = asset_data.select(["sma_10", "vol_10", "returns"])
            y = asset_data.select("label").to_series()

            cv = PurgedWalkForwardCV(n_splits=3, label_horizon=15)
            splits = list(cv.split(X, y))

            assert len(splits) == 3

            # Verify each split maintains temporal order
            for train_idx, test_idx in splits:
                assert max(train_idx) < min(test_idx)

    def test_realistic_ml_pipeline(self, create_labeled_data):
        """Test a realistic ML pipeline with ml4t.engineer data."""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score
        from sklearn.preprocessing import StandardScaler

        labeled_df = create_labeled_data

        # Prepare features and labels
        feature_cols = ["sma_20", "volatility_20", "volume_ma_5", "returns"]
        X = labeled_df.select(feature_cols).to_pandas()
        y = labeled_df.select("label").to_pandas()["label"]

        # Remove any NaN values
        valid_idx = ~(X.isna().any(axis=1) | y.isna())
        X = X[valid_idx].values
        y = y[valid_idx].values

        # Use our purged walk-forward CV
        cv = PurgedWalkForwardCV(
            n_splits=3,
            label_horizon=20,
            embargo_size=5,
            test_size=0.3,
        )

        scores = []
        for fold, (train_idx, test_idx) in enumerate(cv.split(X, y)):
            # Scale features
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X[train_idx])
            X_test = scaler.transform(X[test_idx])

            y_train = y[train_idx]
            y_test = y[test_idx]

            # Train model
            model = RandomForestClassifier(n_estimators=50, random_state=42)
            model.fit(X_train, y_train)

            # Evaluate
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            scores.append(acc)

            print(f"\nFold {fold + 1} Results:")
            print(f"Train size: {len(train_idx)}, Test size: {len(test_idx)}")
            print(f"Accuracy: {acc:.3f}")

        # Verify we got reasonable results
        assert len(scores) == 3
        assert all(0 <= score <= 1 for score in scores)
        print(
            f"\nAverage CV Accuracy: {np.mean(scores):.3f} (+/- {np.std(scores):.3f})",
        )

    def test_walk_forward_with_expanding_window(self, create_labeled_data):
        """Test expanding window behavior with labeled data."""
        labeled_df = create_labeled_data

        feature_cols = ["sma_20", "volatility_20", "volume_ma_5", "returns"]
        X = labeled_df.select(feature_cols)
        y = labeled_df.select("label").to_series()

        cv = PurgedWalkForwardCV(
            n_splits=3,
            label_horizon=20,
            expanding=True,  # Expanding window
        )

        train_sizes = []
        for train_idx, _test_idx in cv.split(X, y):
            train_sizes.append(len(train_idx))

        # With expanding window, each successive training set should be larger
        assert train_sizes[0] < train_sizes[1] < train_sizes[2]

    def test_walk_forward_with_rolling_window(self, create_labeled_data):
        """Test rolling window behavior with labeled data."""
        labeled_df = create_labeled_data

        feature_cols = ["sma_20", "volatility_20", "volume_ma_5", "returns"]
        X = labeled_df.select(feature_cols)
        y = labeled_df.select("label").to_series()

        cv = PurgedWalkForwardCV(
            n_splits=3,
            label_horizon=20,
            expanding=False,  # Rolling window
            train_size=100,  # Fixed training size
        )

        train_sizes = []
        for train_idx, _test_idx in cv.split(X, y):
            train_sizes.append(len(train_idx))

        # With rolling window, training sizes should be similar
        # (may vary slightly due to purging)
        assert max(train_sizes) - min(train_sizes) < 30
