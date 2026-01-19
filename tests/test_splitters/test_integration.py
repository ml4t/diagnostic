"""End-to-end integration tests for CV framework enhancements.

This module tests the complete workflow combining all enhancements:
- Session-aware splitting (Phase 1)
- Group isolation (Phase 2)
- Config-first architecture (Phase 3)
- Fold persistence (Phase 4)

These tests verify that all features work together correctly in realistic scenarios.
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ml4t.diagnostic.splitters import (
    CombinatorialPurgedConfig,
    CombinatorialPurgedCV,
    PurgedWalkForwardConfig,
    PurgedWalkForwardCV,
    load_config,
    load_folds,
    save_config,
    save_folds,
    verify_folds,
)


class TestSessionAwareWorkflow:
    """Test session-aware splitting with config and persistence."""

    def create_session_data(self, n_days=100, sessions_per_day=1):
        """Create data with session identifiers."""
        dates = pd.date_range("2023-01-01", periods=n_days, freq="D", tz="UTC")

        data = []
        for date in dates:
            for session_id in range(sessions_per_day):
                data.append(
                    {
                        "timestamp": date + pd.Timedelta(hours=session_id * 8),
                        "session_date": date,
                        "feature_1": np.random.randn(),
                        "feature_2": np.random.randn(),
                        "return": np.random.randn() * 0.02,
                    }
                )

        df = pd.DataFrame(data)
        df = df.set_index("timestamp")
        return df

    def test_session_aware_config_persistence_workflow(self):
        """Test complete workflow: session-aware + config + persistence."""
        # Create data with sessions
        X = self.create_session_data(n_days=60, sessions_per_day=2)

        # Create config with session alignment
        config = PurgedWalkForwardConfig(
            n_splits=5,
            test_size=4,  # 4 sessions
            label_horizon=1,  # 1 session
            embargo_td=1,  # 1 session
            align_to_sessions=True,
            session_col="session_date",
        )

        # Create splitter from config
        cv = PurgedWalkForwardCV(config=config)

        # Generate folds
        folds = list(cv.split(X))

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)

            # Save config and folds
            config_path = tmp_path / "config.json"
            folds_path = tmp_path / "folds.json"

            save_config(config, config_path)
            save_folds(
                folds,
                X,
                folds_path,
                metadata={
                    "strategy": "session_aware_walk_forward",
                    "n_sessions": X["session_date"].nunique(),
                },
            )

            # Load config and folds
            loaded_config = load_config(config_path, PurgedWalkForwardConfig)
            loaded_folds, metadata = load_folds(folds_path)

            # Verify config round-trip
            assert loaded_config.n_splits == config.n_splits
            assert loaded_config.test_size == config.test_size
            assert loaded_config.align_to_sessions == config.align_to_sessions

            # Verify folds round-trip
            assert len(loaded_folds) == len(folds)

            # Verify metadata
            assert metadata["strategy"] == "session_aware_walk_forward"
            assert metadata["n_sessions"] == X["session_date"].nunique()

            # Recreate splitter and verify it matches
            cv_recreated = PurgedWalkForwardCV(config=loaded_config)
            assert cv_recreated.n_splits == cv.n_splits
            assert cv_recreated.align_to_sessions == cv.align_to_sessions

    def test_session_aligned_boundaries(self):
        """Verify that session alignment actually aligns to session boundaries."""
        X = self.create_session_data(n_days=50, sessions_per_day=2)

        cv = PurgedWalkForwardCV(
            n_splits=3,
            test_size=5,  # 5 sessions
            align_to_sessions=True,
            session_col="session_date",
        )

        folds = list(cv.split(X))

        for train_idx, test_idx in folds:
            # Get sessions for train and test
            train_sessions = X.iloc[train_idx]["session_date"].unique()
            test_sessions = X.iloc[test_idx]["session_date"].unique()

            # Verify no session appears in both train and test
            overlap = set(train_sessions) & set(test_sessions)
            assert len(overlap) == 0, f"Found {len(overlap)} overlapping sessions"


class TestGroupIsolationWorkflow:
    """Test group isolation with multi-asset data."""

    def create_multi_asset_data(self, n_days=100, assets=None):
        """Create multi-asset time series data."""
        if assets is None:
            assets = ["AAPL", "MSFT", "GOOGL"]

        dates = pd.date_range("2023-01-01", periods=n_days, freq="D", tz="UTC")

        data = []
        for date in dates:
            for asset in assets:
                data.append(
                    {
                        "date": date,
                        "asset": asset,
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

        return X, y, groups

    def test_group_isolation_config_persistence(self):
        """Test config persistence with CPCV (without group isolation).

        Note: isolate_groups=False is used because with multi-asset data
        interleaved by date, isolate_groups=True would remove all assets
        from training (since all assets appear in all date-based groups).
        Group isolation is meant for cases where groups represent distinct
        entities (e.g., different contracts) that are NOT interleaved in time.
        """
        # Use enough data to avoid empty train sets after purging/embargo
        X, y, groups = self.create_multi_asset_data(n_days=120)

        # Create config - use isolate_groups=False for date-interleaved multi-asset data
        config = CombinatorialPurgedConfig(
            n_groups=6,
            n_test_groups=2,
            label_horizon=pd.Timedelta("2D"),
            embargo_td=pd.Timedelta("1D"),
            isolate_groups=False,  # With date-interleaved data, isolation would be too aggressive
        )

        # Create splitter
        cv = CombinatorialPurgedCV(config=config)

        # Generate folds with group isolation
        folds = list(cv.split(X, y, groups))

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)

            # Save everything
            config_path = tmp_path / "cpcv_config.json"
            folds_path = tmp_path / "cpcv_folds.json"

            save_config(config, config_path)
            save_folds(
                folds,
                X,
                folds_path,
                metadata={"strategy": "CPCV", "n_assets": len(groups.unique())},
            )

            # Verify folds are valid
            stats = verify_folds(folds, n_samples=len(X))
            assert stats["valid"], f"Invalid folds: {stats['errors']}"

            # Load and verify
            loaded_config = load_config(config_path, CombinatorialPurgedConfig)
            loaded_folds, metadata = load_folds(folds_path)

            assert loaded_config.isolate_groups == config.isolate_groups
            assert len(loaded_folds) == len(folds)
            assert metadata["n_assets"] == len(groups.unique())

    def test_group_isolation_verification(self):
        """Verify that group isolation actually prevents group overlap.

        Note: Group isolation only works correctly when groups represent
        distinct entities that don't appear in every time period. This test
        uses non-date-interleaved data where each asset appears in distinct
        date ranges, making isolation meaningful.
        """
        # Create data where assets appear in non-overlapping time periods
        # This makes group isolation meaningful
        n_days_per_asset = 60
        assets = ["A", "B", "C"]
        data = []

        # Asset A: days 0-59, Asset B: days 60-119, Asset C: days 120-179
        # This way, isolation will work correctly since assets are time-separated
        for i, asset in enumerate(assets):
            start_date = pd.Timestamp("2023-01-01", tz="UTC") + pd.Timedelta(
                days=i * n_days_per_asset
            )
            dates = pd.date_range(start_date, periods=n_days_per_asset, freq="D", tz="UTC")
            for date in dates:
                data.append(
                    {
                        "date": date,
                        "asset": asset,
                        "return": np.random.randn() * 0.02,
                        "feature_1": np.random.randn(),
                        "feature_2": np.random.randn(),
                    }
                )

        df = pd.DataFrame(data).sort_values("date").reset_index(drop=True)
        X = df[["feature_1", "feature_2"]].copy()
        X.index = df["date"]
        y = pd.Series(df["return"].values, index=df["date"])
        groups = pd.Series(df["asset"].values, index=df["date"])

        cv = CombinatorialPurgedCV(
            n_groups=6,  # 6 groups over 180 days = ~30 days each
            n_test_groups=2,
            label_horizon=pd.Timedelta("1D"),
            isolate_groups=True,
        )

        folds = list(cv.split(X, y, groups))
        assert len(folds) > 0, "Should generate at least one fold"

        for train_idx, test_idx in folds:
            # Get groups in train and test
            train_groups = set(groups.iloc[train_idx].unique())
            test_groups = set(groups.iloc[test_idx].unique())

            # With isolate_groups=True and time-separated assets,
            # no asset should appear in both train and test
            overlap = train_groups & test_groups
            assert len(overlap) == 0, f"Found {len(overlap)} overlapping groups: {overlap}"


class TestCompleteWorkflow:
    """Test all features combined in realistic scenarios."""

    def test_session_aware_multi_asset_workflow(self):
        """Test session-aware splitting with multi-asset data and persistence."""
        # Create realistic futures data with sessions
        sessions = pd.date_range("2023-01-01", periods=50, freq="D", tz="UTC")
        contracts = ["ES", "NQ", "YM"]

        data = []
        for session in sessions:
            for contract in contracts:
                # Multiple data points per contract per session
                for _ in range(10):
                    data.append(
                        {
                            "timestamp": session + pd.Timedelta(minutes=np.random.randint(0, 1440)),
                            "session_date": session,
                            "contract": contract,
                            "return": np.random.randn() * 0.02,
                            "feature": np.random.randn(),
                        }
                    )

        df = pd.DataFrame(data).sort_values("timestamp").reset_index(drop=True)
        X = pd.DataFrame(df[["feature"]].values, index=df["timestamp"], columns=["feature"])
        X["session_date"] = df["session_date"].values
        y = pd.Series(df["return"].values, index=df["timestamp"])
        groups = pd.Series(df["contract"].values, index=df["timestamp"])

        # Create config with both session alignment and group isolation
        config = PurgedWalkForwardConfig(
            n_splits=5,
            test_size=3,  # 3 sessions
            label_horizon=1,  # 1 session
            embargo_td=1,  # 1 session
            align_to_sessions=True,
            session_col="session_date",
            isolate_groups=True,  # Also isolate contracts
        )

        cv = PurgedWalkForwardCV(config=config)
        folds = list(cv.split(X, y, groups))

        # Verify session alignment and group isolation
        for train_idx, test_idx in folds:
            # Check session alignment
            train_sessions = X.iloc[train_idx]["session_date"].unique()
            test_sessions = X.iloc[test_idx]["session_date"].unique()
            assert len(set(train_sessions) & set(test_sessions)) == 0

            # Check group isolation
            train_groups = set(groups.iloc[train_idx].unique())
            test_groups = set(groups.iloc[test_idx].unique())
            assert len(train_groups & test_groups) == 0

        # Save and verify persistence
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)

            save_config(config, tmp_path / "config.json")
            save_folds(
                folds,
                X,
                tmp_path / "folds.json",
                metadata={"strategy": "session_aware_multi_asset"},
            )

            # Verify folds
            stats = verify_folds(folds, n_samples=len(X))
            assert stats["valid"], f"Invalid folds: {stats['errors']}"

    def test_cpcv_with_all_features(self):
        """Test CPCV with sessions, config, and persistence.

        Note: isolate_groups=False is used because with multi-asset data
        interleaved by date, isolate_groups=True would remove all assets
        from training (since all assets appear in all date-based groups).
        See test_group_isolation_verification for proper group isolation testing.
        """
        # Create session-based multi-asset data
        sessions = pd.date_range("2023-01-01", periods=80, freq="D", tz="UTC")
        assets = ["ASSET_A", "ASSET_B", "ASSET_C", "ASSET_D"]

        data = []
        for session in sessions:
            for asset in assets:
                data.append(
                    {
                        "timestamp": session,
                        "session_date": session,
                        "asset": asset,
                        "return": np.random.randn() * 0.02,
                        "feature_1": np.random.randn(),
                        "feature_2": np.random.randn(),
                    }
                )

        df = pd.DataFrame(data).sort_values(["timestamp", "asset"]).reset_index(drop=True)

        X = pd.DataFrame(
            df[["feature_1", "feature_2"]].values,
            index=df["timestamp"],
            columns=["feature_1", "feature_2"],
        )
        X["session_date"] = df["session_date"].values
        y = pd.Series(df["return"].values, index=df["timestamp"])
        groups = pd.Series(df["asset"].values, index=df["timestamp"])

        # Config with session alignment (but not group isolation for date-interleaved data)
        config = CombinatorialPurgedConfig(
            n_groups=8,
            n_test_groups=2,
            label_horizon=2,  # 2 sessions
            embargo_td=1,  # 1 session
            align_to_sessions=True,
            session_col="session_date",
            isolate_groups=False,  # With date-interleaved data, isolation would be too aggressive
        )

        cv = CombinatorialPurgedCV(config=config)
        folds = list(cv.split(X, y, groups))

        # Comprehensive verification
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)

            # Save config and folds
            save_config(config, tmp_path / "config.json")
            save_folds(
                folds,
                X,
                tmp_path / "folds.json",
                metadata={
                    "strategy": "CPCV_full_featured",
                    "n_sessions": len(sessions),
                    "n_assets": len(assets),
                },
            )

            # Verify fold integrity
            stats = verify_folds(folds, n_samples=len(X))
            assert stats["valid"], f"Invalid folds: {stats['errors']}"
            assert stats["n_folds"] > 0

            # Load and verify
            loaded_config = load_config(tmp_path / "config.json", CombinatorialPurgedConfig)
            loaded_folds, metadata = load_folds(tmp_path / "folds.json")

            assert loaded_config.align_to_sessions == config.align_to_sessions
            assert loaded_config.isolate_groups == config.isolate_groups
            assert len(loaded_folds) == len(folds)

            # Verify metadata
            assert metadata["n_sessions"] == len(sessions)
            assert metadata["n_assets"] == len(assets)


class TestReproducibility:
    """Test complete reproducibility of CV workflows."""

    def test_config_serialization_reproducibility(self):
        """Verify that saving/loading config produces identical results."""
        # Create original config
        original = PurgedWalkForwardConfig(
            n_splits=5,
            test_size=20,
            train_size=80,
            label_horizon=3,
            embargo_td=2,
            align_to_sessions=True,
            session_col="session_date",
            isolate_groups=True,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "config.json"

            # Save and load
            save_config(original, filepath)
            loaded = load_config(filepath, PurgedWalkForwardConfig)

            # Verify all attributes match
            assert loaded.n_splits == original.n_splits
            assert loaded.test_size == original.test_size
            assert loaded.train_size == original.train_size
            assert loaded.label_horizon == original.label_horizon
            assert loaded.embargo_td == original.embargo_td
            assert loaded.align_to_sessions == original.align_to_sessions
            assert loaded.session_col == original.session_col
            assert loaded.isolate_groups == original.isolate_groups

            # Create splitters from both configs and verify they match
            cv_original = PurgedWalkForwardCV(config=original)
            cv_loaded = PurgedWalkForwardCV(config=loaded)

            assert cv_original.n_splits == cv_loaded.n_splits
            assert cv_original.test_size == cv_loaded.test_size
            assert cv_original.align_to_sessions == cv_loaded.align_to_sessions
            assert cv_original.isolate_groups == cv_loaded.isolate_groups

    def test_fold_persistence_reproducibility(self):
        """Verify that saved folds can be loaded and reused identically."""
        X = np.arange(200).reshape(200, 1)

        cv = PurgedWalkForwardCV(n_splits=5, test_size=30, label_horizon=5, embargo_size=3)

        original_folds = list(cv.split(X))

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "folds.json"

            # Save folds
            save_folds(original_folds, X, filepath)

            # Load folds
            loaded_folds, _ = load_folds(filepath)

            # Verify exact match
            assert len(loaded_folds) == len(original_folds)

            for (orig_train, orig_test), (load_train, load_test) in zip(
                original_folds, loaded_folds, strict=False
            ):
                np.testing.assert_array_equal(orig_train, load_train)
                np.testing.assert_array_equal(orig_test, load_test)


class TestErrorHandling:
    """Test error handling in integrated workflows."""

    def test_invalid_session_column(self):
        """Test error when session column doesn't exist."""
        dates = pd.date_range("2023-01-01", periods=50, freq="D", tz="UTC")
        X = pd.DataFrame({"feature": np.arange(50)}, index=dates)

        cv = PurgedWalkForwardCV(
            n_splits=3,
            test_size=5,
            align_to_sessions=True,
            session_col="nonexistent_column",
        )

        with pytest.raises((KeyError, ValueError)):
            list(cv.split(X))

    def test_timedelta_with_session_alignment(self):
        """Test that timedelta parameters work with session alignment."""
        dates = pd.date_range("2023-01-01", periods=50, freq="D", tz="UTC")
        X = pd.DataFrame({"feature": np.arange(50), "session_date": dates}, index=dates)

        # This should work - timedeltas are supported
        cv = PurgedWalkForwardCV(
            n_splits=3,
            test_size=5,  # Sessions
            label_horizon=pd.Timedelta("2D"),
            embargo_size=pd.Timedelta("1D"),
            align_to_sessions=True,
        )

        folds = list(cv.split(X))
        assert len(folds) == 3
