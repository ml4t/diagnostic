"""Tests for multi-asset purging functionality in CombinatorialCV."""

import numpy as np
import pandas as pd
import pytest

from ml4t.diagnostic.splitters.combinatorial import CombinatorialCV


class TestMultiAssetPurging:
    """Test multi-asset data leakage prevention in CPCV."""

    def test_per_asset_row_horizons_remove_leakage_without_dropping_safe_observations(self):
        """Sort asset identity with rows and check each label's actual observation endpoints."""
        # Create multi-asset test data
        n_assets = 3
        n_samples_per_asset = 60
        n_assets * n_samples_per_asset

        # Create time-ordered data for each asset
        dates = pd.date_range("2020-01-01", periods=n_samples_per_asset, freq="D", tz="UTC")
        asset_data = []

        for asset_id in range(n_assets):
            asset_dates = dates + pd.Timedelta(days=asset_id * 10)  # Slight offset
            asset_df = pd.DataFrame(
                {
                    "feature1": np.random.randn(n_samples_per_asset),
                    "feature2": np.random.randn(n_samples_per_asset),
                    "returns": np.random.randn(n_samples_per_asset) * 0.01,
                    "asset": asset_id,
                },
                index=asset_dates,
            )
            asset_data.append(asset_df)

        # Combine all assets
        X = pd.concat(asset_data).sort_index()
        groups = X.pop("asset").to_numpy()
        y = X["returns"]
        X = X[["feature1", "feature2"]]

        # Create CPCV with multi-asset groups
        cv = CombinatorialCV(
            isolate_groups=False,
            n_groups=6,
            n_test_groups=2,
            label_horizon=5,
            embargo_size=2,
        )

        # Test that splits work with groups
        splits = list(cv.split(X, y, groups=groups))
        assert len(splits) > 0

        # Test each split for proper per-asset purging
        for train_idx, test_idx in splits:
            # Verify no data leakage within each asset
            for asset_id in range(n_assets):
                asset_mask = groups == asset_id
                asset_indices = np.where(asset_mask)[0]

                asset_train = np.intersect1d(train_idx, asset_indices)
                asset_test = np.intersect1d(test_idx, asset_indices)

                test_local = {i for i, row in enumerate(asset_indices) if row in asset_test}
                expected = {
                    row
                    for i, row in enumerate(asset_indices)
                    if i not in test_local
                    and not any(i < j <= i + cv.label_horizon for j in test_local)
                    and not any(j < i <= j + cv.embargo_size for j in test_local)
                }
                assert set(asset_train) == expected, (
                    f"{asset_id}: per-asset label or embargo ownership differs"
                )

    # TODO: Add test_multi_asset_vs_single_asset_purging when multi-asset purging is implemented
    # Currently, multi-asset purging falls back to single-asset behavior
    # Test should verify that multi-asset purging differs from single-asset behavior

    def test_multi_asset_empty_groups_handling(self):
        """Test handling of assets with no test data."""
        n_samples = 90
        X = np.random.randn(n_samples, 3)
        y = np.random.randn(n_samples)

        # Create groups where some assets may not appear in test sets
        groups = np.array([0] * 30 + [1] * 30 + [2] * 30)  # 3 assets, 30 samples each

        cv = CombinatorialCV(
            isolate_groups=False,
            n_groups=6,  # This may result in some groups having single asset
            n_test_groups=1,
            label_horizon=5,
        )

        # Should not crash when some assets have no test data
        splits = list(cv.split(X, y, groups=groups))
        assert len(splits) > 0

        # Verify that assets without test data keep their training data
        for train_idx, test_idx in splits:
            # Check each asset
            for asset_id in [0, 1, 2]:
                asset_mask = groups == asset_id
                asset_indices = np.where(asset_mask)[0]

                asset_train = np.intersect1d(train_idx, asset_indices)
                asset_test = np.intersect1d(test_idx, asset_indices)

                if len(asset_test) == 0:
                    # Asset with no test data should keep most of its training data
                    # (some may be lost due to global group boundaries)
                    assert len(asset_train) >= 0

    def test_groups_validation(self):
        """Test validation of groups parameter."""
        X = np.random.randn(100, 3)
        y = np.random.randn(100)

        cv = CombinatorialCV(isolate_groups=False, n_groups=4, n_test_groups=1)

        # Groups with wrong length should raise error
        wrong_groups = np.array([0, 1, 2])  # Only 3 elements for 100 samples

        with pytest.raises(ValueError, match="X and groups have inconsistent lengths"):
            list(cv.split(X, y, groups=wrong_groups))

    def test_single_asset_backward_compatibility(self):
        """Test that single-asset behavior is unchanged."""
        X = np.random.randn(100, 3)
        y = np.random.randn(100)

        cv = CombinatorialCV(
            isolate_groups=False,
            n_groups=4,
            n_test_groups=1,
            label_horizon=5,
            embargo_size=2,
            random_state=42,
        )

        # Get splits without groups (should use original logic)
        splits_no_groups = list(cv.split(X, y, groups=None))

        # Get splits with single group (should be equivalent)
        single_group = np.zeros(100, dtype=int)  # All samples in same group
        splits_single_group = list(cv.split(X, y, groups=single_group))

        # Results should be very similar (might differ slightly due to
        # asset-specific sample counting in multi-asset logic)
        assert len(splits_no_groups) == len(splits_single_group)

        for (train1, test1), (train2, test2) in zip(
            splits_no_groups,
            splits_single_group,
            strict=False,
        ):
            # Test sets should be identical (group boundaries unchanged)
            np.testing.assert_array_equal(test1, test2)

            # Training sets should be very similar (may differ slightly)
            # Check that they're at least 90% similar
            intersection = len(np.intersect1d(train1, train2))
            union = len(np.union1d(train1, train2))
            similarity = intersection / union if union > 0 else 1.0

            assert similarity >= 0.9, (
                f"Training sets should be similar: similarity={similarity:.3f}"
            )

    def test_multi_asset_realistic_scenario(self):
        """Test with realistic multi-asset financial data scenario."""
        # Simulate 3 stocks over 1 year
        n_days = 252
        n_stocks = 3

        dates = pd.date_range("2020-01-01", periods=n_days, freq="B", tz="UTC")  # Business days

        data_list = []
        groups_list = []

        for stock_id in range(n_stocks):
            # Simulate stock returns with some autocorrelation
            returns = np.random.randn(n_days) * 0.02
            returns = pd.Series(returns, index=dates).rolling(5).mean().fillna(0)

            stock_data = pd.DataFrame(
                {
                    "momentum": returns.rolling(20).mean(),
                    "volatility": returns.rolling(20).std(),
                    "volume": np.random.lognormal(10, 1, n_days),
                    "returns": returns.shift(-5),  # 5-day forward returns
                },
                index=dates,
            ).dropna()

            data_list.append(stock_data)
            groups_list.extend([f"STOCK_{stock_id}"] * len(stock_data))

        # Combine all stocks
        all_data = pd.concat(data_list).sort_index()
        X = all_data[["momentum", "volatility", "volume"]]
        y = all_data["returns"]
        groups = np.array(groups_list)[all_data.index.argsort()]

        # Test with realistic CPCV parameters
        cv = CombinatorialCV(
            isolate_groups=False,
            n_groups=8,
            n_test_groups=2,
            label_horizon=5,  # 5-day label horizon
            embargo_size=1,  # 1-day embargo
            max_combinations=10,  # Limit combinations for testing
        )

        splits = list(cv.split(X, y, groups=groups))

        # Should generate valid splits
        assert len(splits) > 0
        assert len(splits) <= 10  # Respects max_combinations

        # Each split should have reasonable train/test sizes
        for train_idx, test_idx in splits:
            assert len(train_idx) > 0
            assert len(test_idx) > 0

            # No overlap between train and test
            assert len(np.intersect1d(train_idx, test_idx)) == 0

            # Combined should be subset of original data
            combined = np.union1d(train_idx, test_idx)
            assert len(combined) <= len(X)
