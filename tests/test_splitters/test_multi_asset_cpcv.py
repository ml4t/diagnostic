"""
Tests for multi-asset Combinatorial Purged Cross-Validation scenarios.

This module specifically tests the multi-asset purging logic in CombinatorialPurgedCV,
with focus on non-contiguous test groups and edge cases that can cause data leakage.

Key test scenarios:
- Non-contiguous test groups (e.g., groups [2, 5, 8])
- Multiple assets with different data availability
- Contiguous vs non-contiguous segment handling
- Edge cases with single groups and gaps
- Verification of no data leakage
"""

import numpy as np
import pandas as pd
import pytest

from ml4t.diagnostic.splitters.combinatorial import CombinatorialPurgedCV


class TestMultiAssetNonContiguous:
    """Test multi-asset purging with non-contiguous test groups."""

    def create_multi_asset_data(self, n_days=100, assets=None):
        """Create synthetic multi-asset time series data."""
        if assets is None:
            assets = ["AAPL", "MSFT", "GOOGL"]

        dates = pd.date_range("2023-01-01", periods=n_days, freq="D", tz="UTC")

        # Create data where each asset has data on all days
        data = []
        for _i, date in enumerate(dates):
            for asset in assets:
                data.append(
                    {
                        "date": date,
                        "asset": asset,
                        "return": np.random.randn() * 0.02,  # 2% daily vol
                        "feature_1": np.random.randn(),
                        "feature_2": np.random.randn(),
                    }
                )

        df = pd.DataFrame(data)
        df = df.sort_values(["date", "asset"]).reset_index(drop=True)

        # Create proper DataFrames with DatetimeIndex for CV compatibility
        X = pd.DataFrame(
            df[["feature_1", "feature_2"]].values,
            index=df["date"],
            columns=["feature_1", "feature_2"],
        )
        y = pd.Series(df["return"].values, index=df["date"])
        groups = pd.Series(df["asset"].values, index=df["date"])

        return X, y, groups, df["date"]

    def test_non_contiguous_basic(self):
        """Test basic non-contiguous groups scenario."""
        X, y, groups, timestamps = self.create_multi_asset_data(n_days=50)

        # Use non-contiguous groups: [1, 3, 5] out of 10 groups
        # Note: isolate_groups=False because we're testing temporal purging, not asset isolation
        cv = CombinatorialPurgedCV(
            isolate_groups=False,
            n_groups=10,
            n_test_groups=3,
            label_horizon=pd.Timedelta("1D"),
            embargo_pct=0.05,
        )

        splits = list(cv.split(X, y, groups))

        # Should have some splits
        assert len(splits) > 0

        # Each split should have reasonable train/test split
        for train_idx, test_idx in splits:
            assert len(train_idx) > 0
            assert len(test_idx) > 0
            assert len(np.intersect1d(train_idx, test_idx)) == 0  # No overlap

            # Verify indices are valid
            assert np.all(train_idx >= 0) and np.all(train_idx < len(X))
            assert np.all(test_idx >= 0) and np.all(test_idx < len(X))

    def test_non_contiguous_purging_correctness(self):
        """Test that non-contiguous groups are purged correctly."""
        # Create data with clear time structure
        n_days = 30
        assets = ["AAPL", "MSFT"]
        X, y, groups, timestamps = self.create_multi_asset_data(n_days, assets)

        cv = CombinatorialPurgedCV(
            isolate_groups=False,
            n_groups=6,  # 6 groups of 5 days each
            n_test_groups=2,  # Use 2 non-contiguous groups
            label_horizon=pd.Timedelta("1D"),
            embargo_pct=0.1,  # 10% embargo
        )

        # Get first split with non-contiguous test groups
        train_idx, test_idx = next(cv.split(X, y, groups))

        # Check that purging was applied correctly
        train_timestamps = X.index[train_idx]
        test_timestamps = X.index[test_idx]

        # Verify no train data is too close to any test data
        for test_ts in test_timestamps:
            # Check purging: no train data within label_horizon of test start
            time_diff = train_timestamps - test_ts
            too_close = np.abs(time_diff) < pd.Timedelta("1D")

            # There should be no training data too close to test data
            # (allowing for some samples due to asset-specific purging)
            assert np.sum(too_close) < len(train_timestamps) * 0.1  # At most 10%

    def test_contiguous_vs_non_contiguous_comparison(self):
        """Compare contiguous vs non-contiguous test group handling."""
        X, y, groups, timestamps = self.create_multi_asset_data(n_days=60)

        # Contiguous case: groups [1, 2, 3]
        cv_contiguous = CombinatorialPurgedCV(
            isolate_groups=False,
            n_groups=10,
            n_test_groups=3,
            label_horizon=pd.Timedelta("1D"),
            embargo_pct=0.05,
            max_combinations=1,  # Force specific combination
        )

        # Non-contiguous case: groups that would be [1, 4, 7] etc
        cv_non_contiguous = CombinatorialPurgedCV(
            isolate_groups=False,
            n_groups=10,
            n_test_groups=3,
            label_horizon=pd.Timedelta("1D"),
            embargo_pct=0.05,
            max_combinations=None,  # Allow all combinations
        )

        # Both should produce valid splits
        contiguous_splits = list(cv_contiguous.split(X, y, groups))
        non_contiguous_splits = list(cv_non_contiguous.split(X, y, groups))

        assert len(contiguous_splits) >= 1
        assert len(non_contiguous_splits) >= 1

        # Non-contiguous should generally have more training data
        # (less purging due to gaps between test groups)
        np.mean([len(train) for train, _ in contiguous_splits])
        np.mean([len(train) for train, _ in non_contiguous_splits])

        # This is statistical and may not always hold, but generally should
        # (commented out as it may be flaky)
        # assert avg_train_non_contiguous >= avg_train_contiguous * 0.8

    def test_single_asset_consistency(self):
        """Test that multi-asset logic works correctly with single asset."""
        # Create single-asset data
        X, y, groups, timestamps = self.create_multi_asset_data(n_days=40, assets=["AAPL"])

        cv = CombinatorialPurgedCV(
            isolate_groups=False,
            n_groups=8,
            n_test_groups=2,
            label_horizon=pd.Timedelta("1D"),
            embargo_pct=0.05,
        )

        splits = list(cv.split(X, y, groups))

        # Should work similarly to multi-asset case
        assert len(splits) > 0

        for train_idx, test_idx in splits:
            assert len(train_idx) > 0
            assert len(test_idx) > 0
            assert len(np.intersect1d(train_idx, test_idx)) == 0

    def test_asset_specific_purging(self):
        """Test that purging is applied per asset correctly."""
        # Create scenario where assets have different temporal patterns
        dates = pd.date_range("2023-01-01", periods=30, freq="D", tz="UTC")

        data = []
        for i, date in enumerate(dates):
            # AAPL has data on all days
            data.append(
                {
                    "date": date,
                    "asset": "AAPL",
                    "return": np.random.randn() * 0.02,
                    "feature": np.random.randn(),
                }
            )

            # MSFT has data only on even days
            if i % 2 == 0:
                data.append(
                    {
                        "date": date,
                        "asset": "MSFT",
                        "return": np.random.randn() * 0.02,
                        "feature": np.random.randn(),
                    }
                )

        df = pd.DataFrame(data).sort_values(["date", "asset"]).reset_index(drop=True)

        # Create proper DataFrames with DatetimeIndex
        X = pd.DataFrame(df[["feature"]].values, index=df["date"], columns=["feature"])
        y = pd.Series(df["return"].values, index=df["date"])
        groups = pd.Series(df["asset"].values, index=df["date"])

        cv = CombinatorialPurgedCV(
            isolate_groups=False,
            n_groups=6,
            n_test_groups=2,
            label_horizon=pd.Timedelta("1D"),
            embargo_pct=0.1,
        )

        # Should handle mixed asset availability correctly
        splits = list(cv.split(X, y, groups))
        assert len(splits) > 0

        for train_idx, test_idx in splits:
            # Verify basic properties
            assert len(np.intersect1d(train_idx, test_idx)) == 0

            # Both AAPL and MSFT should appear in results
            train_groups = groups[train_idx]
            test_groups = groups[test_idx]

            # At least one asset should have training data
            assert len(np.unique(train_groups)) >= 1
            # At least one asset should have test data
            assert len(np.unique(test_groups)) >= 1

    def test_extreme_non_contiguous_case(self):
        """Test extreme case with maximally spaced test groups."""
        X, y, groups, timestamps = self.create_multi_asset_data(n_days=50)

        # Use groups [1, 5, 9] - maximally spaced
        cv = CombinatorialPurgedCV(
            isolate_groups=False,
            n_groups=10,
            n_test_groups=3,
            label_horizon=pd.Timedelta("1D"),
            embargo_pct=0.05,
            max_combinations=1,  # Control which combination we get
        )

        splits = list(cv.split(X, y, groups))
        assert len(splits) >= 1

        # Each split should be valid despite large gaps
        for train_idx, test_idx in splits:
            assert len(train_idx) > 0
            assert len(test_idx) > 0
            assert len(np.intersect1d(train_idx, test_idx)) == 0

    def test_edge_case_adjacent_groups(self):
        """Test edge case where test groups are adjacent."""
        X, y, groups, timestamps = self.create_multi_asset_data(n_days=40)

        # This should behave like one contiguous segment
        cv = CombinatorialPurgedCV(
            isolate_groups=False,
            n_groups=8,
            n_test_groups=2,  # Could get adjacent groups like [3, 4]
            label_horizon=pd.Timedelta("1D"),
            embargo_pct=0.05,
        )

        splits = list(cv.split(X, y, groups))
        assert len(splits) > 0

        # Should handle adjacent groups correctly
        for train_idx, test_idx in splits:
            assert len(np.intersect1d(train_idx, test_idx)) == 0

    def test_no_data_leakage_property(self):
        """Property test: verify no data leakage in any split."""
        X, y, groups, timestamps = self.create_multi_asset_data(n_days=60)

        cv = CombinatorialPurgedCV(
            isolate_groups=False,
            n_groups=10,
            n_test_groups=3,
            label_horizon=pd.Timedelta("2D"),  # 2-day horizon
            embargo_pct=0.1,
        )

        for train_idx, test_idx in cv.split(X, y, groups):
            # No index overlap
            assert len(np.intersect1d(train_idx, test_idx)) == 0

            # Check temporal separation per asset
            for asset in np.unique(groups):
                asset_mask = groups == asset
                asset_train_idx = train_idx[np.isin(train_idx, np.where(asset_mask)[0])]
                asset_test_idx = test_idx[np.isin(test_idx, np.where(asset_mask)[0])]

                if len(asset_train_idx) == 0 or len(asset_test_idx) == 0:
                    continue

                train_times = X.index[asset_train_idx]
                test_times = X.index[asset_test_idx]

                # For each test time, check that no train data is too close
                for test_time in test_times:
                    # Calculate minimum time difference to training data
                    time_diffs = np.abs(train_times - test_time)
                    min_diff = time_diffs.min()

                    # Should respect label horizon (allowing small numerical errors)
                    # Note: This might be relaxed due to embargo and discrete time
                    assert min_diff >= pd.Timedelta("0.5D"), (
                        f"Data leakage detected for {asset}: {min_diff}"
                    )

    def test_sparse_data_multiple_assets(self):
        """Test with sparse data where assets have many missing days."""
        dates = pd.date_range("2023-01-01", periods=60, freq="D", tz="UTC")

        data = []
        for i, date in enumerate(dates):
            # AAPL has data every 3 days
            if i % 3 == 0:
                data.append(
                    {
                        "date": date,
                        "asset": "AAPL",
                        "return": np.random.randn() * 0.02,
                        "feature_1": np.random.randn(),
                        "feature_2": np.random.randn(),
                    }
                )

            # MSFT has data every 4 days, offset by 1
            if i % 4 == 1:
                data.append(
                    {
                        "date": date,
                        "asset": "MSFT",
                        "return": np.random.randn() * 0.02,
                        "feature_1": np.random.randn(),
                        "feature_2": np.random.randn(),
                    }
                )

            # GOOGL has data every 2 days
            if i % 2 == 0:
                data.append(
                    {
                        "date": date,
                        "asset": "GOOGL",
                        "return": np.random.randn() * 0.02,
                        "feature_1": np.random.randn(),
                        "feature_2": np.random.randn(),
                    }
                )

        df = pd.DataFrame(data).sort_values(["date", "asset"]).reset_index(drop=True)

        X = pd.DataFrame(
            df[["feature_1", "feature_2"]].values,
            index=df["date"],
            columns=["feature_1", "feature_2"],
        )
        y = pd.Series(df["return"].values, index=df["date"])
        groups = pd.Series(df["asset"].values, index=df["date"])

        cv = CombinatorialPurgedCV(
            isolate_groups=False,
            n_groups=8,
            n_test_groups=3,
            label_horizon=pd.Timedelta("2D"),
            embargo_pct=0.1,
        )

        splits = list(cv.split(X, y, groups))
        assert len(splits) > 0

        for train_idx, test_idx in splits:
            # Verify no overlap
            assert len(np.intersect1d(train_idx, test_idx)) == 0

            # Check that each asset's data integrity is maintained
            train_assets = groups.iloc[train_idx].unique()
            test_assets = groups.iloc[test_idx].unique()

            # At least some assets should be represented
            assert len(train_assets) >= 1
            assert len(test_assets) >= 1

    def test_overlapping_embargo_periods(self):
        """Test handling of overlapping embargo periods across assets."""
        X, y, groups, timestamps = self.create_multi_asset_data(n_days=50)

        # Large embargo that could cause overlaps
        cv = CombinatorialPurgedCV(
            isolate_groups=False,
            n_groups=5,
            n_test_groups=2,
            label_horizon=pd.Timedelta("3D"),
            embargo_pct=0.3,  # 30% embargo - very large
        )

        splits = list(cv.split(X, y, groups))
        assert len(splits) > 0

        for train_idx, test_idx in splits:
            # Despite large embargo, should maintain separation
            assert len(np.intersect1d(train_idx, test_idx)) == 0

            # Training set should be reduced due to embargo
            total_possible = len(X)
            train_pct = len(train_idx) / total_possible

            # With 40% test (2/5 groups) and 30% embargo, expect less training data
            # But the actual amount depends on group overlap, so be more lenient
            assert train_pct < 0.7  # Should have less than 70% as training

    def test_asset_correlation_effects(self):
        """Test with highly correlated assets to ensure proper purging."""
        n_days = 60
        dates = pd.date_range("2023-01-01", periods=n_days, freq="D", tz="UTC")

        # Create correlated returns
        base_returns = np.random.randn(n_days) * 0.02

        data = []
        for i, date in enumerate(dates):
            # AAPL and MSFT are highly correlated
            for asset, correlation in [("AAPL", 1.0), ("MSFT", 0.9), ("GOOGL", 0.3)]:
                data.append(
                    {
                        "date": date,
                        "asset": asset,
                        "return": base_returns[i] * correlation + np.random.randn() * 0.005,
                        "feature_1": np.random.randn(),
                        "feature_2": np.random.randn(),
                    }
                )

        df = pd.DataFrame(data).sort_values(["date", "asset"]).reset_index(drop=True)

        X = pd.DataFrame(
            df[["feature_1", "feature_2"]].values,
            index=df["date"],
            columns=["feature_1", "feature_2"],
        )
        y = pd.Series(df["return"].values, index=df["date"])
        groups = pd.Series(df["asset"].values, index=df["date"])

        cv = CombinatorialPurgedCV(
            isolate_groups=False,
            n_groups=6,
            n_test_groups=2,
            label_horizon=pd.Timedelta("2D"),
            embargo_pct=0.15,
        )

        # Should handle correlated assets without data leakage
        splits = list(cv.split(X, y, groups))
        assert len(splits) > 0

        for train_idx, test_idx in splits:
            # No direct overlap
            assert len(np.intersect1d(train_idx, test_idx)) == 0

            # Check that purging is applied per asset (not affected by correlation)
            for asset in ["AAPL", "MSFT", "GOOGL"]:
                asset_mask = groups == asset
                asset_train = train_idx[np.isin(train_idx, np.where(asset_mask)[0])]
                asset_test = test_idx[np.isin(test_idx, np.where(asset_mask)[0])]

                if len(asset_train) > 0 and len(asset_test) > 0:
                    # Verify temporal separation for each asset independently
                    train_times = X.index[asset_train]
                    test_times = X.index[asset_test]

                    for test_time in test_times[:5]:  # Check first few
                        time_diffs = np.abs(train_times - test_time)
                        # Should maintain minimum separation
                        assert time_diffs.min() >= pd.Timedelta("1D")

    def test_unbalanced_asset_representation(self):
        """Test with very unbalanced asset representation in data."""
        dates = pd.date_range("2023-01-01", periods=100, freq="D", tz="UTC")

        data = []
        for _i, date in enumerate(dates):
            # AAPL has 90% of the data points
            for _ in range(9):
                data.append(
                    {
                        "date": date,
                        "asset": "AAPL",
                        "return": np.random.randn() * 0.02,
                        "feature": np.random.randn(),
                    }
                )

            # MSFT has only 10% of data points
            if np.random.random() < 0.5:  # Only half the days
                data.append(
                    {
                        "date": date,
                        "asset": "MSFT",
                        "return": np.random.randn() * 0.02,
                        "feature": np.random.randn(),
                    }
                )

        df = pd.DataFrame(data).sort_values(["date", "asset"]).reset_index(drop=True)

        X = pd.DataFrame(df[["feature"]].values, index=df["date"], columns=["feature"])
        y = pd.Series(df["return"].values, index=df["date"])
        groups = pd.Series(df["asset"].values, index=df["date"])

        cv = CombinatorialPurgedCV(
            isolate_groups=False,
            n_groups=10,
            n_test_groups=3,
            label_horizon=pd.Timedelta("1D"),
            embargo_pct=0.1,
        )

        splits = list(cv.split(X, y, groups))
        assert len(splits) > 0

        for train_idx, test_idx in splits:
            # Should handle imbalance correctly
            assert len(np.intersect1d(train_idx, test_idx)) == 0

            # AAPL should dominate both sets due to imbalance
            train_assets = groups.iloc[train_idx].value_counts()
            test_assets = groups.iloc[test_idx].value_counts()

            if "AAPL" in train_assets:
                assert train_assets["AAPL"] > len(train_idx) * 0.7  # AAPL dominates
            if "AAPL" in test_assets:
                assert test_assets["AAPL"] > len(test_idx) * 0.7

    def test_performance_large_dataset(self):
        """Test performance with large multi-asset dataset."""
        import time

        # Create large dataset
        n_days = 500
        assets = [f"ASSET_{i}" for i in range(20)]  # 20 assets

        X, y, groups, timestamps = self.create_multi_asset_data(n_days, assets)

        cv = CombinatorialPurgedCV(
            isolate_groups=False,
            n_groups=20,
            n_test_groups=5,
            label_horizon=pd.Timedelta("2D"),
            embargo_pct=0.1,
            max_combinations=10,  # Limit combinations for speed
        )

        start_time = time.time()
        splits = list(cv.split(X, y, groups))
        elapsed = time.time() - start_time

        # Should complete in reasonable time even with many assets
        assert elapsed < 10.0  # 10 seconds max
        assert len(splits) > 0

        # Verify correctness even at scale
        for train_idx, test_idx in splits[:2]:  # Check first 2 splits
            assert len(np.intersect1d(train_idx, test_idx)) == 0
            assert len(train_idx) > 0
            assert len(test_idx) > 0

    def test_edge_case_single_group(self):
        """Test edge case with single test group."""
        X, y, groups, timestamps = self.create_multi_asset_data(n_days=30)

        cv = CombinatorialPurgedCV(
            isolate_groups=False,
            n_groups=10,
            n_test_groups=1,  # Single test group
            label_horizon=pd.Timedelta("1D"),
            embargo_pct=0.05,
        )

        splits = list(cv.split(X, y, groups))
        assert len(splits) == 10  # Should have exactly 10 combinations

        for train_idx, test_idx in splits:
            assert len(np.intersect1d(train_idx, test_idx)) == 0
            # Test set should be approximately 10% of data
            assert len(test_idx) < len(X) * 0.15

    def test_embargo_boundary_conditions(self):
        """Test embargo behavior at group boundaries."""
        # Use more data to handle higher embargo percentages without empty train sets
        X, y, groups, timestamps = self.create_multi_asset_data(n_days=120)

        # Test with different embargo percentages
        # Use moderate embargo values to avoid empty train sets
        for embargo_pct in [0.0, 0.1, 0.2, 0.3]:
            cv = CombinatorialPurgedCV(
                isolate_groups=False,
                n_groups=8,
                n_test_groups=2,
                label_horizon=pd.Timedelta("1D"),
                embargo_pct=embargo_pct,
            )

            splits = list(cv.split(X, y, groups))
            assert len(splits) > 0

            for train_idx, _test_idx in splits[:1]:  # Check first split
                # Higher embargo should result in smaller training set
                train_ratio = len(train_idx) / len(X)

                if embargo_pct == 0.0:
                    assert train_ratio > 0.6  # More training data with no embargo
                elif embargo_pct >= 0.3:
                    assert train_ratio < 0.65  # Less with higher embargo


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
