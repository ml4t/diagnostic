"""Integration tests for signal analysis module.

Tests realistic end-to-end scenarios and consistency.
"""

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from ml4t.diagnostic.signal import SignalResult, analyze_signal

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def predictive_signal_data():
    """Data where higher factor predicts higher returns (positive IC expected).

    Returns two DataFrames: factor_df (date, asset, factor) and prices_df (date, asset, price).
    """
    np.random.seed(42)

    n_assets = 100
    n_dates = 60  # ~3 months of daily data

    base_date = date(2024, 1, 1)
    dates = [base_date + timedelta(days=i) for i in range(n_dates)]
    assets = [f"ASSET_{i:03d}" for i in range(n_assets)]

    # True quality signal that persists across time
    base_quality = np.random.randn(n_assets)

    factor_rows = []
    price_rows = []

    # Initialize prices
    prices = np.full(n_assets, 100.0)

    for _d_idx, d in enumerate(dates):
        # Factor = true quality + noise (observable signal)
        factor_noise = np.random.randn(n_assets) * 0.3
        factors = base_quality + factor_noise

        # Returns correlate with true quality (predictive)
        returns = base_quality * 0.002 + np.random.randn(n_assets) * 0.01
        prices = prices * (1 + returns)

        for a_idx, asset in enumerate(assets):
            factor_rows.append(
                {
                    "date": d,
                    "asset": asset,
                    "factor": factors[a_idx],
                }
            )
            price_rows.append(
                {
                    "date": d,
                    "asset": asset,
                    "price": prices[a_idx],
                }
            )

    factor_df = pl.DataFrame(factor_rows)
    prices_df = pl.DataFrame(price_rows)

    return factor_df, prices_df


@pytest.fixture
def anti_predictive_data():
    """Data where higher factor predicts LOWER returns (negative IC expected)."""
    np.random.seed(123)

    n_assets = 50
    n_dates = 30

    base_date = date(2024, 1, 1)
    dates = [base_date + timedelta(days=i) for i in range(n_dates)]

    # True quality signal
    base_quality = np.random.randn(n_assets)

    factor_rows = []
    price_rows = []
    prices = np.full(n_assets, 100.0)

    for _d_idx, d in enumerate(dates):
        factors = base_quality + np.random.randn(n_assets) * 0.2
        # Returns are INVERSELY related to quality
        returns = -base_quality * 0.003 + np.random.randn(n_assets) * 0.01
        prices = prices * (1 + returns)

        for a_idx in range(n_assets):
            factor_rows.append(
                {
                    "date": d,
                    "asset": f"A{a_idx}",
                    "factor": factors[a_idx],
                }
            )
            price_rows.append(
                {
                    "date": d,
                    "asset": f"A{a_idx}",
                    "price": prices[a_idx],
                }
            )

    factor_df = pl.DataFrame(factor_rows)
    prices_df = pl.DataFrame(price_rows)

    return factor_df, prices_df


@pytest.fixture
def random_factor_data():
    """Data with no signal (factor is pure noise, independent of returns)."""
    np.random.seed(456)

    n_assets = 50
    n_dates = 30

    base_date = date(2024, 1, 1)
    dates = [base_date + timedelta(days=i) for i in range(n_dates)]

    factor_rows = []
    price_rows = []
    prices = np.full(n_assets, 100.0)

    for _d_idx, d in enumerate(dates):
        # Factor and returns are completely independent
        factors = np.random.randn(n_assets)
        returns = np.random.randn(n_assets) * 0.01
        prices = prices * (1 + returns)

        for a_idx in range(n_assets):
            factor_rows.append(
                {
                    "date": d,
                    "asset": f"A{a_idx}",
                    "factor": factors[a_idx],
                }
            )
            price_rows.append(
                {
                    "date": d,
                    "asset": f"A{a_idx}",
                    "price": prices[a_idx],
                }
            )

    factor_df = pl.DataFrame(factor_rows)
    prices_df = pl.DataFrame(price_rows)

    return factor_df, prices_df


# =============================================================================
# Integration Tests
# =============================================================================


class TestRealisticScenarios:
    """Tests for realistic signal analysis scenarios."""

    def test_predictive_factor_positive_ic(self, predictive_signal_data):
        """Test that a predictive factor produces positive IC."""
        factor_df, prices_df = predictive_signal_data

        result = analyze_signal(
            factor=factor_df,
            prices=prices_df,
            periods=(1,),
            quantiles=5,
            compute_turnover_flag=False,
        )

        # IC should be positive for predictive factor
        ic_mean = result.ic["1D"]
        assert ic_mean > 0, f"Expected positive IC, got {ic_mean}"

        # Spread should be positive (Q5 > Q1)
        spread = result.spread["1D"]
        assert spread > 0, f"Expected positive spread, got {spread}"

    def test_anti_predictive_factor_negative_ic(self, anti_predictive_data):
        """Test that an anti-predictive factor produces negative IC."""
        factor_df, prices_df = anti_predictive_data

        result = analyze_signal(
            factor=factor_df,
            prices=prices_df,
            periods=(1,),
            quantiles=5,
            compute_turnover_flag=False,
        )

        # IC should be negative for anti-predictive factor
        ic_mean = result.ic["1D"]
        assert ic_mean < 0, f"Expected negative IC, got {ic_mean}"

        # Spread should be negative (Q5 < Q1)
        spread = result.spread["1D"]
        assert spread < 0, f"Expected negative spread, got {spread}"

    def test_random_factor_near_zero_ic(self, random_factor_data):
        """Test that a random factor produces near-zero IC."""
        factor_df, prices_df = random_factor_data

        result = analyze_signal(
            factor=factor_df,
            prices=prices_df,
            periods=(1,),
            quantiles=5,
            compute_turnover_flag=False,
        )

        # IC should be near zero for random factor
        ic_mean = result.ic["1D"]
        # Allow for random variation, but should be close to zero
        assert abs(ic_mean) < 0.3, f"Expected near-zero IC, got {ic_mean}"


class TestConsistency:
    """Tests for result consistency and reproducibility."""

    def test_deterministic_results(self, predictive_signal_data):
        """Test that same input produces same output."""
        factor_df, prices_df = predictive_signal_data

        result1 = analyze_signal(
            factor=factor_df,
            prices=prices_df,
            periods=(1,),
            quantiles=5,
            compute_turnover_flag=False,
        )

        result2 = analyze_signal(
            factor=factor_df,
            prices=prices_df,
            periods=(1,),
            quantiles=5,
            compute_turnover_flag=False,
        )

        # IC stats should be identical
        assert result1.ic["1D"] == result2.ic["1D"]
        assert result1.ic_std["1D"] == result2.ic_std["1D"]

        # Spread should be identical
        assert result1.spread["1D"] == result2.spread["1D"]

        # Quantile returns should be identical
        for q in range(1, 6):
            assert result1.quantile_returns["1D"][q] == result2.quantile_returns["1D"][q]

    def test_json_round_trip_preserves_data(self, predictive_signal_data, tmp_path):
        """Test that JSON serialization preserves all data."""
        factor_df, prices_df = predictive_signal_data

        result = analyze_signal(
            factor=factor_df,
            prices=prices_df,
            periods=(1,),
            quantiles=5,
            compute_turnover_flag=False,
        )

        # Round-trip through JSON file
        json_path = tmp_path / "result.json"
        result.to_json(str(json_path))
        restored = SignalResult.from_json(str(json_path))

        # Verify critical fields preserved
        assert restored.ic["1D"] == result.ic["1D"]
        assert restored.spread["1D"] == result.spread["1D"]
        assert restored.quantiles == result.quantiles
        assert restored.periods == result.periods


class TestLargeDataset:
    """Tests for performance with larger datasets."""

    @pytest.mark.slow
    def test_large_dataset_completes(self):
        """Test that analysis completes for a large dataset."""
        np.random.seed(789)

        n_assets = 500
        n_dates = 250  # ~1 year of trading days

        base_date = date(2024, 1, 1)

        factor_rows = []
        price_rows = []
        prices = np.full(n_assets, 100.0)

        for d_idx in range(n_dates):
            current_date = base_date + timedelta(days=d_idx)
            factors = np.random.randn(n_assets)
            returns = np.random.randn(n_assets) * 0.01
            prices = prices * (1 + returns)

            for a_idx in range(n_assets):
                factor_rows.append(
                    {
                        "date": current_date,
                        "asset": f"A{a_idx}",
                        "factor": factors[a_idx],
                    }
                )
                price_rows.append(
                    {
                        "date": current_date,
                        "asset": f"A{a_idx}",
                        "price": prices[a_idx],
                    }
                )

        factor_df = pl.DataFrame(factor_rows)
        prices_df = pl.DataFrame(price_rows)

        # Should complete without error
        result = analyze_signal(
            factor=factor_df,
            prices=prices_df,
            periods=(1,),
            quantiles=5,
            compute_turnover_flag=False,
        )

        # Basic validation
        assert result.quantiles == 5
        assert "1D" in result.ic
        assert "1D" in result.spread
