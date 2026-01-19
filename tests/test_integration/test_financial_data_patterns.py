"""Test ml4t-diagnostic with common financial data patterns.

This module tests ml4t-diagnostic splitters with data patterns typical of financial ML,
without requiring external dependencies.
"""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import polars as pl

from ml4t.diagnostic.splitters import PurgedWalkForwardCV


class TestFinancialDataPatterns:
    """Test with typical financial data patterns."""

    def test_with_daily_returns_data(self):
        """Test with daily returns data and fixed horizon labels."""
        # Create synthetic daily returns data
        np.random.seed(42)
        n_days = 252 * 3  # 3 years of daily data

        dates = pd.date_range("2021-01-01", periods=n_days, freq="B", tz="UTC")
        returns = np.random.normal(0.0001, 0.02, n_days)

        # Create features: lagged returns and rolling statistics
        df = pd.DataFrame(
            {
                "date": dates,
                "returns": returns,
            },
        )

        # Add features
        df["returns_lag1"] = df["returns"].shift(1)
        df["returns_lag2"] = df["returns"].shift(2)
        df["rolling_vol_20"] = df["returns"].rolling(20).std()
        df["rolling_mean_20"] = df["returns"].rolling(20).mean()

        # Create forward-looking labels (20-day ahead returns)
        df["label"] = df["returns"].rolling(20).sum().shift(-20)

        # Drop NaN values
        df = df.dropna()

        # Set date as index
        df = df.set_index("date")

        # Features and labels
        feature_cols = [
            "returns",
            "returns_lag1",
            "returns_lag2",
            "rolling_vol_20",
            "rolling_mean_20",
        ]
        X = df[feature_cols]
        y = df["label"]

        # Use walk-forward CV with 20-day label horizon
        cv = PurgedWalkForwardCV(
            n_splits=5,
            label_horizon=pd.Timedelta("20D"),
            embargo_size=pd.Timedelta("5D"),
            test_size=0.2,
        )

        splits = list(cv.split(X, y))
        assert len(splits) == 5

        # Verify temporal integrity
        for train_idx, test_idx in splits:
            train_dates = X.index[train_idx]
            test_dates = X.index[test_idx]

            # No overlap between train and test periods
            assert train_dates.max() < test_dates.min()

            # Check purging: last train date + 20 days <= first test date
            assert train_dates.max() + pd.Timedelta("20D") <= test_dates.min()

    def test_with_intraday_data(self):
        """Test with intraday data (5-minute bars)."""
        # Create synthetic intraday data
        np.random.seed(42)

        # 20 trading days of 5-minute bars (78 bars per day)
        bars_per_day = 78
        n_days = 20
        n_samples = bars_per_day * n_days

        # Generate timestamps
        timestamps = []
        current_date = datetime(2024, 1, 2)  # Start on a Tuesday

        for _day in range(n_days):
            # Skip weekends
            if current_date.weekday() >= 5:
                current_date += timedelta(days=2)

            # Market hours: 9:30 AM to 4:00 PM
            day_start = current_date.replace(hour=9, minute=30)

            for i in range(bars_per_day):
                timestamps.append(day_start + timedelta(minutes=5 * i))

            current_date += timedelta(days=1)

        # Generate price data
        returns = np.random.normal(0, 0.001, n_samples)  # Lower vol for intraday
        prices = 100 * (1 + returns).cumprod()

        df = pl.DataFrame(
            {
                "timestamp": timestamps,
                "close": prices,
                "returns": np.concatenate([[0], returns[1:]]),
                "volume": np.random.randint(1000, 10000, n_samples),
            },
        )

        # Add intraday features
        df = df.with_columns(
            [
                # Rolling stats over last 30 bars (2.5 hours)
                pl.col("returns").rolling_mean(window_size=30).alias("mean_30"),
                pl.col("returns").rolling_std(window_size=30).alias("vol_30"),
                # Time of day feature
                pl.col("timestamp").dt.hour().alias("hour"),
                pl.col("timestamp").dt.minute().alias("minute"),
            ],
        )

        # Create labels: next 12 bars (1 hour) forward return
        df = df.with_columns(
            pl.col("returns").rolling_sum(window_size=12).shift(-12).alias("label"),
        )

        # Remove NaN values
        df = df.filter(pl.col("mean_30").is_not_null() & pl.col("label").is_not_null())

        feature_cols = ["returns", "mean_30", "vol_30", "hour", "minute"]
        X = df.select(feature_cols)
        y = df.select("label").to_series()

        # Walk-forward CV with 12-bar (1 hour) label horizon
        cv = PurgedWalkForwardCV(
            n_splits=3,
            label_horizon=12,  # 12 bars = 1 hour
            embargo_size=6,  # 30 minutes embargo
        )

        splits = list(cv.split(X, y))
        assert len(splits) == 3

        # Verify each split
        for train_idx, test_idx in splits:
            assert len(train_idx) > 0
            assert len(test_idx) > 0
            # No training data within 12 bars of test start
            assert max(train_idx) + 12 <= min(test_idx)

    def test_with_event_driven_labels(self):
        """Test with event-driven labels (variable horizons)."""
        # Create data with events at irregular intervals
        np.random.seed(42)
        n_samples = 1000

        timestamps = pd.date_range("2022-01-01", periods=n_samples, freq="H", tz="UTC")
        prices = 100 * (1 + np.random.normal(0, 0.005, n_samples)).cumprod()

        df = pd.DataFrame(
            {
                "timestamp": timestamps,
                "price": prices,
                "returns": np.concatenate([[0], np.diff(np.log(prices))]),
            },
        )

        # Add features
        for lag in [1, 2, 3]:
            df[f"returns_lag{lag}"] = df["returns"].shift(lag)

        df["volatility"] = df["returns"].rolling(24).std()  # 24-hour vol

        # Simulate events: only label certain points
        # (e.g., when volatility exceeds threshold)
        vol_threshold = df["volatility"].quantile(0.7)
        event_mask = df["volatility"] > vol_threshold

        # For events, create labels based on next 24 hours
        df["label"] = np.nan
        df.loc[event_mask, "label"] = df.loc[event_mask].apply(
            lambda row: df.loc[
                (df.index > row.name) & (df.index <= row.name + 24),
                "returns",
            ].sum(),
            axis=1,
        )

        # Keep only labeled events
        labeled_df = df.dropna(subset=["label"])
        labeled_df = labeled_df.set_index("timestamp")

        feature_cols = [
            "returns",
            "returns_lag1",
            "returns_lag2",
            "returns_lag3",
            "volatility",
        ]
        X = labeled_df[feature_cols]
        y = labeled_df["label"]

        # With hourly data and 24-hour horizon
        cv = PurgedWalkForwardCV(
            n_splits=3,
            label_horizon=pd.Timedelta("24H"),
            embargo_size=pd.Timedelta("6H"),
        )

        splits = list(cv.split(X, y))
        assert len(splits) == 3

        print(f"\nEvent-driven data: {len(labeled_df)} events from {n_samples} samples")
        for i, (train_idx, test_idx) in enumerate(splits):
            print(f"Split {i + 1}: Train={len(train_idx)}, Test={len(test_idx)}")

    def test_with_multiple_horizons(self):
        """Test data with multiple prediction horizons."""
        # Common scenario: predicting returns at different horizons
        np.random.seed(42)
        n_samples = 500

        dates = pd.date_range("2023-01-01", periods=n_samples, freq="D", tz="UTC")
        returns = np.random.normal(0.0001, 0.015, n_samples)

        df = pd.DataFrame(
            {
                "date": dates,
                "returns": returns,
            },
        )

        # Features
        df["returns_ma_5"] = df["returns"].rolling(5).mean()
        df["returns_ma_20"] = df["returns"].rolling(20).mean()
        df["volatility"] = df["returns"].rolling(20).std()

        # Multiple horizons
        df["label_5d"] = df["returns"].rolling(5).sum().shift(-5)
        df["label_10d"] = df["returns"].rolling(10).sum().shift(-10)
        df["label_20d"] = df["returns"].rolling(20).sum().shift(-20)

        df = df.dropna().set_index("date")

        # Test with different horizons
        for horizon_days, label_col in [
            (5, "label_5d"),
            (10, "label_10d"),
            (20, "label_20d"),
        ]:
            feature_cols = ["returns", "returns_ma_5", "returns_ma_20", "volatility"]
            X = df[feature_cols]
            y = df[label_col]

            cv = PurgedWalkForwardCV(
                n_splits=3,
                label_horizon=pd.Timedelta(f"{horizon_days}D"),
                embargo_size=pd.Timedelta("2D"),
            )

            splits = list(cv.split(X, y))

            # Verify purging is correct for each horizon
            for train_idx, test_idx in splits:
                train_dates = X.index[train_idx]
                test_dates = X.index[test_idx]

                # Check purging matches the horizon
                if len(train_dates) > 0 and len(test_dates) > 0:
                    gap = (test_dates.min() - train_dates.max()).days
                    assert gap >= horizon_days

    def test_with_sparse_labels(self):
        """Test with sparse labels (many NaN values)."""
        # Simulate scenario where labels are only available occasionally
        np.random.seed(42)
        n_samples = 1000

        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2023-01-01", periods=n_samples, freq="H", tz="UTC"),
                "feature1": np.random.randn(n_samples),
                "feature2": np.random.randn(n_samples),
                "feature3": np.random.randn(n_samples),
            },
        )

        # Create sparse labels (only 20% have labels)
        label_mask = np.random.rand(n_samples) < 0.2
        df["label"] = np.nan
        df.loc[label_mask, "label"] = np.random.choice(
            [-1, 0, 1],
            size=label_mask.sum(),
        )

        # In practice, we'd filter to labeled samples for training
        labeled_df = df.dropna(subset=["label"])
        labeled_df = labeled_df.set_index("timestamp")

        X = labeled_df[["feature1", "feature2", "feature3"]]
        y = labeled_df["label"]

        cv = PurgedWalkForwardCV(
            n_splits=3,
            label_horizon=pd.Timedelta("24H"),  # 24-hour horizon
            test_size=0.3,
        )

        splits = list(cv.split(X, y))
        assert len(splits) == 3

        total_samples = sum(len(train) + len(test) for train, test in splits)
        print(f"\nSparse labels: {len(labeled_df)} labeled from {n_samples} total")
        print(f"Total samples in CV: {total_samples}")

    def test_realistic_backtest_scenario(self):
        """Test a realistic backtesting scenario with proper data handling."""
        # Create realistic financial data
        np.random.seed(42)

        # 2 years of daily data
        dates = pd.date_range("2022-01-01", "2023-12-31", freq="B", tz="UTC")
        n_samples = len(dates)

        # Generate correlated features (typical of financial data)
        market_return = np.random.normal(0.0005, 0.02, n_samples)

        df = pd.DataFrame(
            {
                "date": dates,
                "market_return": market_return,
                "sector_return": market_return * 0.8 + np.random.normal(0, 0.01, n_samples),
                "momentum": np.random.normal(0, 0.015, n_samples),
                "value": np.random.normal(0, 0.01, n_samples),
                "volatility": np.abs(np.random.normal(0.02, 0.005, n_samples)),
            },
        )

        # Add technical indicators
        df["rsi"] = 50 + np.random.normal(0, 10, n_samples)
        df["rsi"] = df["rsi"].clip(0, 100)

        # Create realistic labels (next 20-day returns)
        df["future_return"] = df["market_return"].rolling(20).sum().shift(-20)

        # Add some alpha (small predictive signal)
        df["future_return"] += 0.1 * df["momentum"].rolling(5).mean()

        df = df.dropna().set_index("date")

        # Features for prediction
        feature_cols = [
            "market_return",
            "sector_return",
            "momentum",
            "value",
            "volatility",
            "rsi",
        ]
        X = df[feature_cols]
        y = df["future_return"]

        # Realistic walk-forward setup
        cv = PurgedWalkForwardCV(
            n_splits=12,  # Monthly rebalancing
            label_horizon=pd.Timedelta("20D"),  # 20 trading days
            embargo_size=pd.Timedelta("5D"),  # 5 days embargo
            test_size=21,  # ~1 month of test data
            expanding=True,  # Expanding window (common in practice)
        )

        # Track backtest statistics
        test_periods = []
        train_sizes = []

        for _fold, (train_idx, test_idx) in enumerate(cv.split(X, y)):
            train_period = (X.index[train_idx].min(), X.index[train_idx].max())
            test_period = (X.index[test_idx].min(), X.index[test_idx].max())

            test_periods.append(test_period)
            train_sizes.append(len(train_idx))

            # Verify no lookahead bias
            assert train_period[1] < test_period[0]

            # Verify sufficient gap (purging + embargo)
            gap = (test_period[0] - train_period[1]).days
            assert gap >= 20  # At least label_horizon

        # Verify expanding window
        assert all(train_sizes[i] <= train_sizes[i + 1] for i in range(len(train_sizes) - 1))

        print("\nBacktest summary:")
        print(f"Total folds: {len(test_periods)}")
        print(f"Training set sizes: {min(train_sizes)} to {max(train_sizes)}")
        print(f"Test period coverage: {test_periods[0][0]} to {test_periods[-1][1]}")
