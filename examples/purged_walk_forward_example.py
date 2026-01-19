"""
Example: Purged Walk-Forward Cross-Validation with ML4T Engineer Output

This example demonstrates how to use ML4T Evaluation's PurgedWalkForwardCV splitter
with output from ML4T Engineer, showcasing proper data leakage prevention.
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

# Add ml4t-diagnostic source to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from ml4t.diagnostic.splitters import PurgedWalkForwardCV


def create_sample_features_data() -> pl.DataFrame:
    """Create sample data mimicking ML4T Engineer output.

    NOTE: This uses a simple synthetic data generator. For more realistic
    financial time series with proper stylized facts (fat tails, volatility
    clustering, etc.), see ml4t.engineer.synthetic module and
    qfeatures/examples/synthetic_data_example.py

    Returns a DataFrame with:
    - event_time: Timestamps for each observation
    - asset_id: Single asset for simplicity
    - features: Several feature columns
    - label: Target variable (-1, 0, 1)
    - label_time: When the label is realized
    - label_return: Actual return value
    """
    # Generate daily timestamps for 1 year
    n_samples = 252  # Trading days in a year
    base_time = datetime(2024, 1, 1, 9, 30)
    timestamps = []
    current = base_time

    for _ in range(n_samples):
        timestamps.append(current)
        # Skip weekends
        current = current + timedelta(days=1)
        while current.weekday() in [5, 6]:  # Saturday, Sunday
            current = current + timedelta(days=1)

    # Generate synthetic features
    np.random.seed(42)

    # Price-based features (increased volatility for more realistic labels)
    prices = 100 + np.cumsum(np.random.randn(n_samples) * 2.0)  # Increased volatility
    returns = np.diff(prices, prepend=prices[0]) / prices

    # Technical features
    momentum_5 = pd.Series(returns).rolling(5).mean().fillna(0).values
    momentum_20 = pd.Series(returns).rolling(20).mean().fillna(0).values
    volatility = pd.Series(returns).rolling(20).std().fillna(0.01).values

    # Microstructure features
    volume_imbalance = np.random.randn(n_samples) * 0.1
    order_flow = np.random.randn(n_samples) * 0.05

    # Labels with 20-day horizon
    label_horizon_days = 20
    future_returns = pd.Series(returns).shift(-label_horizon_days).fillna(0).values

    # Create labels: -1 (down), 0 (neutral), 1 (up)
    threshold = 0.005  # 0.5% threshold (more realistic for daily data)
    labels = np.where(
        future_returns > threshold,
        1,
        np.where(future_returns < -threshold, -1, 0),
    )

    # Label times (20 days in the future)
    label_times = [t + timedelta(days=20) for t in timestamps]

    # Create DataFrame
    df = pl.DataFrame(
        {
            "event_time": timestamps,
            "asset_id": ["SPY"] * n_samples,
            "price": prices,
            "returns": returns,
            "momentum_5": momentum_5,
            "momentum_20": momentum_20,
            "volatility": volatility,
            "volume_imbalance": volume_imbalance,
            "order_flow": order_flow,
            "label": labels,
            "label_time": label_times,
            "label_return": future_returns,
        },
    )

    return df


def demonstrate_basic_walk_forward():
    """Demonstrate basic walk-forward cross-validation."""
    print("\n" + "=" * 60)
    print("1. BASIC WALK-FORWARD CROSS-VALIDATION")
    print("=" * 60)

    # Create sample data
    df = create_sample_features_data()
    print(f"\nDataset shape: {df.shape}")
    print(f"Date range: {df['event_time'].min()} to {df['event_time'].max()}")  # type: ignore[str-bytes-safe]

    # Feature columns
    feature_cols = ["momentum_5", "momentum_20", "volatility", "volume_imbalance", "order_flow"]
    X = df.select(feature_cols).to_numpy()
    y = df["label"].to_numpy()

    # Basic walk-forward without purging
    cv = PurgedWalkForwardCV(
        n_splits=5,
        test_size=0.2,  # 20% of data for each test set
        expanding=True,  # Use expanding window
        label_horizon=0,  # No purging
        embargo_size=0,  # No embargo
    )

    print(f"\nNumber of splits: {cv.get_n_splits(X)}")
    print("\nSplit details (without purging/embargo):")
    print("-" * 40)

    for i, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        train_dates = df[train_idx.tolist()]["event_time"]
        test_dates = df[test_idx.tolist()]["event_time"]

        print(f"\nSplit {i + 1}:")
        print(f"  Train: {len(train_idx):3d} samples | {train_dates.min()} to {train_dates.max()}")
        print(f"  Test:  {len(test_idx):3d} samples | {test_dates.min()} to {test_dates.max()}")


def demonstrate_purged_walk_forward():
    """Demonstrate walk-forward with purging to prevent label leakage."""
    print("\n" + "=" * 60)
    print("2. PURGED WALK-FORWARD (PREVENTING LABEL LEAKAGE)")
    print("=" * 60)

    # Create sample data
    df = create_sample_features_data()

    # Feature columns
    feature_cols = ["momentum_5", "momentum_20", "volatility", "volume_imbalance", "order_flow"]
    X = df.select(feature_cols).to_numpy()
    y = df["label"].to_numpy()

    # Walk-forward WITH purging
    label_horizon = 20  # 20-day forward-looking labels

    cv_purged = PurgedWalkForwardCV(
        n_splits=5,
        test_size=0.2,
        expanding=True,
        label_horizon=label_horizon,  # Remove 20 days before test
        embargo_size=0,
    )

    print(f"\nLabel horizon: {label_horizon} days")
    print("This removes training samples whose labels overlap with test period")
    print("\nSplit details (WITH purging):")
    print("-" * 40)

    for i, (train_idx, test_idx) in enumerate(cv_purged.split(X, y)):
        train_dates = df[train_idx]["event_time"]
        test_dates = df[test_idx]["event_time"]

        print(f"\nSplit {i + 1}:")
        print(f"  Train: {len(train_idx):3d} samples | {train_dates.min()} to {train_dates.max()}")
        print(f"  Test:  {len(test_idx):3d} samples | {test_dates.min()} to {test_dates.max()}")

        # Verify purging: check gap between last train and first test
        if len(train_idx) > 0 and len(test_idx) > 0:
            gap_days = (
                df[int(test_idx[0])]["event_time"][0] - df[int(train_idx[-1])]["event_time"][0]
            ).days
            print(f"  Gap:   {gap_days} days between train end and test start")


def demonstrate_embargo():
    """Demonstrate embargo to prevent serial correlation leakage."""
    print("\n" + "=" * 60)
    print("3. EMBARGO (PREVENTING SERIAL CORRELATION)")
    print("=" * 60)

    # Create sample data
    df = create_sample_features_data()

    # Feature columns
    feature_cols = ["momentum_5", "momentum_20", "volatility", "volume_imbalance", "order_flow"]
    X = df.select(feature_cols).to_numpy()
    y = df["label"].to_numpy()

    # Walk-forward with BOTH purging and embargo
    cv_full = PurgedWalkForwardCV(
        n_splits=3,  # Fewer splits to see embargo effect
        test_size=0.2,
        expanding=True,
        label_horizon=20,  # Purge 20 days
        embargo_pct=0.01,  # Embargo 1% of data after each test
    )

    print("\nLabel horizon: 20 days (purging)")
    print("Embargo: 1% of total samples after each test set")
    print("\nSplit details (WITH purging AND embargo):")
    print("-" * 40)

    all_train_indices = []
    all_test_indices = []

    for i, (train_idx, test_idx) in enumerate(cv_full.split(X, y)):
        train_dates = df[train_idx]["event_time"]
        test_dates = df[test_idx]["event_time"]

        print(f"\nSplit {i + 1}:")
        print(f"  Train: {len(train_idx):3d} samples | {train_dates.min()} to {train_dates.max()}")
        print(f"  Test:  {len(test_idx):3d} samples | {test_dates.min()} to {test_dates.max()}")

        all_train_indices.append(set(train_idx))
        all_test_indices.append(set(test_idx))

    # Show which samples are never used due to purging/embargo
    all_used = set()
    for train_set, test_set in zip(all_train_indices, all_test_indices, strict=False):
        all_used.update(train_set)
        all_used.update(test_set)

    never_used = set(range(len(df))) - all_used
    print(f"\n{len(never_used)} samples never used due to purging/embargo")
    if never_used:
        never_used_dates = df[sorted(never_used)]["event_time"]
        print(f"Date ranges excluded: {never_used_dates.min()} to {never_used_dates.max()}")  # type: ignore[str-bytes-safe]


def demonstrate_rolling_window():
    """Demonstrate rolling vs expanding window."""
    print("\n" + "=" * 60)
    print("4. ROLLING VS EXPANDING WINDOW")
    print("=" * 60)

    # Create sample data
    df = create_sample_features_data()

    # Feature columns
    feature_cols = ["momentum_5", "momentum_20", "volatility", "volume_imbalance", "order_flow"]
    X = df.select(feature_cols).to_numpy()
    y = df["label"].to_numpy()

    # Expanding window (default)
    cv_expanding = PurgedWalkForwardCV(
        n_splits=4,
        test_size=0.15,
        expanding=True,
        label_horizon=10,
    )

    # Rolling window with fixed training size
    cv_rolling = PurgedWalkForwardCV(
        n_splits=4,
        test_size=0.15,
        train_size=100,  # Fixed 100 samples for training
        expanding=False,
        label_horizon=10,
    )

    print("\nEXPANDING WINDOW (training set grows):")
    print("-" * 40)
    for i, (train_idx, test_idx) in enumerate(cv_expanding.split(X, y)):
        print(f"Split {i + 1}: Train={len(train_idx):3d}, Test={len(test_idx):3d}")

    print("\nROLLING WINDOW (fixed training size):")
    print("-" * 40)
    for i, (train_idx, test_idx) in enumerate(cv_rolling.split(X, y)):
        print(f"Split {i + 1}: Train={len(train_idx):3d}, Test={len(test_idx):3d}")


def demonstrate_model_validation():
    """Demonstrate actual model validation with purged walk-forward."""
    print("\n" + "=" * 60)
    print("5. MODEL VALIDATION EXAMPLE")
    print("=" * 60)

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, classification_report

    # Create sample data
    df = create_sample_features_data()

    # Feature columns
    feature_cols = ["momentum_5", "momentum_20", "volatility", "volume_imbalance", "order_flow"]
    X = df.select(feature_cols).to_numpy()
    y = df["label"].to_numpy()

    # Set up purged walk-forward CV
    cv = PurgedWalkForwardCV(
        n_splits=3,
        test_size=0.2,
        expanding=True,
        label_horizon=20,
        embargo_pct=0.01,
    )

    # Train and evaluate model
    print("\nTraining Random Forest with Purged Walk-Forward CV")
    print("-" * 40)

    scores = []
    predictions_all = []
    actuals_all = []

    for i, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X[train_idx], y[train_idx])

        # Predict
        y_pred = model.predict(X[test_idx])
        y_true = y[test_idx]

        # Calculate accuracy
        accuracy = accuracy_score(y_true, y_pred)
        scores.append(accuracy)

        predictions_all.extend(y_pred)
        actuals_all.extend(y_true)

        print(f"Fold {i + 1}: Accuracy = {accuracy:.3f}")

    print(f"\nMean CV Accuracy: {np.mean(scores):.3f} (+/- {np.std(scores):.3f})")

    # Overall classification report
    print("\nOverall Classification Report:")
    # Get unique labels
    unique_labels = sorted(set(actuals_all) | set(predictions_all))
    label_names = {-1: "Down", 0: "Neutral", 1: "Up"}
    target_names = [label_names.get(l, str(l)) for l in unique_labels]
    print(
        classification_report(
            actuals_all,
            predictions_all,
            labels=unique_labels,
            target_names=target_names,
        ),
    )


def main():
    """Run all demonstrations."""
    print("\n" + "=" * 60)
    print("PURGED WALK-FORWARD CROSS-VALIDATION EXAMPLES")
    print("=" * 60)
    print("\nThis example demonstrates ML4T Evaluation's PurgedWalkForwardCV splitter")
    print("for time-series cross-validation with data leakage prevention.")

    # Run demonstrations
    demonstrate_basic_walk_forward()
    demonstrate_purged_walk_forward()
    demonstrate_embargo()
    demonstrate_rolling_window()
    demonstrate_model_validation()

    print("\n" + "=" * 60)
    print("KEY TAKEAWAYS:")
    print("=" * 60)
    print("""
1. PURGING prevents label leakage by removing training samples whose
   labels depend on information from the test period.

2. EMBARGO prevents serial correlation leakage by adding gaps after
   test sets to account for prediction persistence.

3. EXPANDING vs ROLLING windows control whether training sets grow
   or maintain fixed size as we walk forward.

4. These techniques are CRITICAL for realistic backtesting of
   financial ML models to avoid overfitting.
    """)


if __name__ == "__main__":
    main()
