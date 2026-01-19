"""Pytest configuration and fixtures for ML4T Diagnostic tests."""

import tempfile
from collections.abc import Generator
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from hypothesis import settings

# Configure Hypothesis for CI
settings.register_profile("ci", max_examples=50, deadline=5000)
settings.register_profile("dev", max_examples=10, deadline=2000)  # Fast local dev
settings.register_profile("debug", max_examples=1, deadline=None)

# Use dev profile by default (faster for local development)
# CI sets --hypothesis-profile=ci explicitly
import os

_profile = os.environ.get("HYPOTHESIS_PROFILE", "dev")
settings.load_profile(_profile)


@pytest.fixture
def sample_features() -> pl.DataFrame:
    """
    Generate sample feature data for validation testing.

    Returns a DataFrame with:
    - 1000 samples
    - 10 features
    - Binary labels (-1, 0, 1)
    - Timestamps
    - Asset IDs
    """
    np.random.seed(42)
    n_samples = 1000
    n_features = 10

    # Generate timestamps
    base_time = datetime(2024, 1, 1)
    timestamps = [base_time + timedelta(hours=i) for i in range(n_samples)]

    # Generate features
    features = np.random.randn(n_samples, n_features)

    # Generate labels with class imbalance
    labels = np.random.choice([-1, 0, 1], size=n_samples, p=[0.3, 0.4, 0.3])

    # Create DataFrame
    feature_cols = {f"feature_{i}": features[:, i] for i in range(n_features)}

    df = pl.DataFrame(
        {
            "timestamp": timestamps,
            "asset_id": ["ASSET_" + str(i % 10) for i in range(n_samples)],
            "label": labels,
            **feature_cols,
        }
    )

    return df


@pytest.fixture
def sample_predictions() -> pl.DataFrame:
    """Generate sample model predictions for evaluation."""
    np.random.seed(42)
    n_samples = 1000

    # Generate predictions with some correlation to true labels
    true_labels = np.random.choice([-1, 0, 1], size=n_samples)
    noise = np.random.randn(n_samples) * 0.3
    predictions = true_labels + noise

    # Add probabilities
    probabilities = 1 / (1 + np.exp(-predictions))  # Sigmoid

    return pl.DataFrame(
        {
            "y_true": true_labels,
            "y_pred": predictions,
            "y_prob": probabilities,
        }
    )


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_backtest_results(temp_dir: Path) -> Path:
    """
    Create sample backtest results for evaluation.

    Returns path to a Parquet file with backtest metrics.
    """
    # Generate returns
    np.random.seed(42)
    n_days = 252

    # Generate daily returns
    returns = np.random.randn(n_days) * 0.02  # 2% daily volatility

    # Calculate cumulative returns
    cum_returns = np.cumprod(1 + returns) - 1

    # Create DataFrame
    dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(n_days)]

    df = pl.DataFrame(
        {
            "date": dates,
            "returns": returns,
            "cumulative_returns": cum_returns,
            "portfolio_value": 100000 * (1 + cum_returns),
        }
    )

    path = temp_dir / "backtest_results.parquet"
    df.write_parquet(str(path))
    return path


@pytest.fixture
def multi_asset_features() -> pl.DataFrame:
    """Generate multi-asset feature data for cross-sectional validation."""
    np.random.seed(42)

    assets = ["AAPL", "GOOGL", "MSFT", "AMZN", "META"]
    n_days = 100
    n_features = 5

    data = []
    base_time = datetime(2024, 1, 1)

    for day in range(n_days):
        timestamp = base_time + timedelta(days=day)
        for asset in assets:
            features = np.random.randn(n_features)
            label = np.random.choice([-1, 0, 1])

            row = {
                "timestamp": timestamp,
                "asset_id": asset,
                "label": label,
                **{f"feature_{i}": features[i] for i in range(n_features)},
            }
            data.append(row)

    return pl.DataFrame(data)


@pytest.fixture
def embargo_data() -> pl.DataFrame:
    """
    Generate data for testing embargo and purging functionality.

    Returns DataFrame with overlapping label windows for testing.
    """
    np.random.seed(42)

    # Create events with known overlaps
    events = []
    base_time = datetime(2024, 1, 1)

    for i in range(100):
        event_time = base_time + timedelta(hours=i)
        # Label time is 5-10 hours in the future
        label_time = event_time + timedelta(hours=np.random.randint(5, 11))

        events.append(
            {
                "event_time": event_time,
                "label_time": label_time,
                "feature_1": np.random.randn(),
                "feature_2": np.random.randn(),
                "label": np.random.choice([-1, 0, 1]),
            }
        )

    return pl.DataFrame(events)


@pytest.fixture(autouse=True)
def reset_random_seeds():
    """Reset random seeds before each test for reproducibility."""
    np.random.seed(42)
    yield
    # Cleanup after test if needed


# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers",
        "slow: marks tests as slow (deselect with '-m \"not slow\"')",
    )
    config.addinivalue_line(
        "markers",
        "integration: marks tests as integration tests",
    )
    config.addinivalue_line(
        "markers",
        "property: marks tests as property-based tests using Hypothesis",
    )


# Patterns that indicate slow tests (actual ML model training)
# These patterns are checked against test NAME, not file name
SLOW_TEST_PATTERNS = [
    # Patterns that match truly slow test names (E2E, benchmarks)
    "benchmark",
    "e2e",
    "scale_test",
    # Removed patterns (tests complete in <15s after optimization):
    # - shap_importance, shap_interaction, xgboost, lightgbm
    # - keras, tensorflow, deep_explainer, kernel_explainer, tree_explainer
]

# Files that are known to be slow (measured sequential runtime >10s)
# Last measured: 2026-01-11
# Files removed after measurement showed they complete in <10s:
#   - test_metrics.py (11.09s/264 tests = 0.04s/test - NOT slow)
#   - test_mda_importance (3.39s), test_ml_importance_analysis (7.53s)
#   - test_h_statistic (5.83s), test_domain_classifier (7.07s)
#   - test_shap_models (7.79s), test_volatility (3.19s)
#   - test_multi_signal (3.00s), test_ras_validation (2.65s)
#   - test_report_generation (3.31s)
SLOW_TEST_FILES = [
    # Only genuinely slow files (>30s sequential)
    "test_interaction_summary",  # ~42s - interaction analysis with multiple methods
]
# NOTE: Files under 20s (test_shap_*, test_drift_*, test_signal_results) are
# acceptable for normal CI runs and no longer marked as slow.
# E2E tests (test_multi_signal_e2e, test_multi_signal_performance) have
# file-level pytestmark and don't need to be in this list.


# Pytest hooks for better test output
def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers."""
    for item in items:
        # Add integration marker to integration tests
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)

        # Add property marker to property tests
        if "property" in item.name or "hypothesis" in str(item.fspath):
            item.add_marker(pytest.mark.property)

        # Auto-mark slow tests based on name patterns
        test_name_lower = item.name.lower()
        file_name = str(item.fspath).lower()

        # Mark slow by test name pattern
        if any(pattern in test_name_lower for pattern in SLOW_TEST_PATTERNS):
            item.add_marker(pytest.mark.slow)
        # Mark slow by file name pattern
        elif any(pattern in file_name for pattern in SLOW_TEST_FILES):
            item.add_marker(pytest.mark.slow)
