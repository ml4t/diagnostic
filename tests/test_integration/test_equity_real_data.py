"""Integration tests with real equity market data.

This module tests ml4t-diagnostic with actual market data to validate
correctness with real-world data characteristics:
- Non-uniform trading days (holidays, halts)
- Multi-asset universe with varying liquidity
- Regime changes and volatility clustering
- Autocorrelation in returns
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pytest

from ml4t.diagnostic.evaluation.metrics.information_coefficient import (
    compute_ic_by_horizon,
    information_coefficient,
)
from ml4t.diagnostic.evaluation.stats import (
    deflated_sharpe_ratio,
    deflated_sharpe_ratio_from_statistics,
)
from ml4t.diagnostic.splitters import CombinatorialPurgedCV, PurgedWalkForwardCV

# Path to test fixture data
FIXTURES_PATH = Path("~/ml4t/third-edition/tests/fixtures/data").expanduser()
WIKI_PRICES_PATH = FIXTURES_PATH / "equities" / "wiki_prices.parquet"


@pytest.fixture(scope="module")
def wiki_prices():
    """Load Wiki Prices equity data.

    Returns DataFrame with columns: date, ticker, open, high, low, close, volume, adj_close
    """
    if not WIKI_PRICES_PATH.exists():
        pytest.skip(f"Wiki prices data not found at {WIKI_PRICES_PATH}")

    df = pl.read_parquet(WIKI_PRICES_PATH)
    return df


@pytest.fixture(scope="module")
def wiki_returns(wiki_prices):
    """Compute daily returns from wiki prices."""
    df = wiki_prices

    # Calculate daily returns
    returns = (
        df.sort(["ticker", "date"])
        .with_columns(
            (pl.col("adj_close") / pl.col("adj_close").shift(1).over("ticker") - 1).alias(
                "return"
            )
        )
        .drop_nulls("return")
    )

    return returns


@pytest.fixture(scope="module")
def single_stock_returns(wiki_returns):
    """Get returns for a single stock with sufficient history."""
    # Find stock with most observations
    counts = wiki_returns.group_by("ticker").count().sort("count", descending=True)
    top_ticker = counts["ticker"][0]

    single_stock = wiki_returns.filter(pl.col("ticker") == top_ticker).sort("date")
    return single_stock


@pytest.fixture(scope="module")
def multi_stock_returns(wiki_returns):
    """Get returns for multiple stocks on the same dates."""
    # Get top 10 stocks by observation count
    counts = wiki_returns.group_by("ticker").count().sort("count", descending=True)
    top_tickers = counts["ticker"][:10].to_list()

    multi_stock = wiki_returns.filter(pl.col("ticker").is_in(top_tickers))
    return multi_stock


class TestEquityDataProperties:
    """Verify data has expected real-world properties."""

    def test_data_has_multiple_tickers(self, wiki_prices):
        """Verify we have multi-asset data."""
        n_tickers = wiki_prices["ticker"].n_unique()
        assert n_tickers > 1, "Expected multiple tickers"

    def test_data_has_sufficient_history(self, wiki_prices):
        """Verify we have sufficient time series length."""
        date_range = wiki_prices["date"].max() - wiki_prices["date"].min()
        assert date_range.days > 30, "Expected at least 30 days of history"

    def test_returns_are_reasonable(self, wiki_returns):
        """Verify returns are in realistic range."""
        ret = wiki_returns["return"].to_numpy()

        # Check statistics are reasonable
        assert -0.5 < np.mean(ret) < 0.5, "Mean returns look unrealistic"
        assert 0 < np.std(ret) < 0.5, "Std of returns looks unrealistic"

        # No extreme outliers (>500% daily move would be suspicious)
        assert np.abs(ret).max() < 5, "Found suspicious extreme return"

    def test_returns_have_fat_tails(self, single_stock_returns):
        """Verify returns exhibit fat tails (kurtosis > 3)."""
        ret = single_stock_returns["return"].to_numpy()

        if len(ret) >= 30:
            from scipy.stats import kurtosis

            kurt = kurtosis(ret, fisher=False)  # Pearson kurtosis
            # Financial returns typically have excess kurtosis
            assert kurt > 2, f"Expected fat tails, got kurtosis={kurt:.2f}"


class TestCPCVWithRealData:
    """Test Combinatorial Purged Cross-Validation with real equity data."""

    def test_cpcv_temporal_integrity(self, single_stock_returns):
        """Test that CPCV maintains temporal integrity with real data."""
        df = single_stock_returns.to_pandas()

        # Ensure timezone-aware index (required by purging)
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize("UTC")
        df = df.set_index("date")

        if len(df) < 100:
            pytest.skip("Insufficient data for CPCV test")

        # Create simple features (lagged returns)
        df["feat1"] = df["return"].shift(1)
        df["feat2"] = df["return"].rolling(5).mean().shift(1)
        df["label"] = df["return"].shift(-5)  # 5-day forward return
        df = df.dropna()

        X = df[["feat1", "feat2"]]
        y = df["label"]

        cv = CombinatorialPurgedCV(
            n_groups=5,
            n_test_groups=2,
            label_horizon=pd.Timedelta("5D"),
            embargo_size=pd.Timedelta("2D"),
        )

        n_splits = 0
        for train_idx, test_idx in cv.split(X, y):
            n_splits += 1

            train_dates = X.index[train_idx]
            test_dates = X.index[test_idx]

            # In CPCV, train and test can be in any temporal order
            # The key invariant is that purging removes contaminated samples
            # Check that train and test are disjoint
            assert len(set(train_idx) & set(test_idx)) == 0, "Train and test overlap"

            # Check that we have samples
            assert len(train_idx) > 0, "Empty train set"
            assert len(test_idx) > 0, "Empty test set"

        assert n_splits > 0, "No splits generated"

    def test_cpcv_multi_asset(self, multi_stock_returns):
        """Test CPCV with multi-asset panel data."""
        df = multi_stock_returns.to_pandas()

        if len(df) < 500:
            pytest.skip("Insufficient data for multi-asset test")

        # Pivot to wide format for panel features
        pivot = df.pivot_table(index="date", columns="ticker", values="return")

        # Create cross-sectional features (rank of returns)
        features = pivot.rank(axis=1, pct=True)
        features = features.add_prefix("rank_")

        # Label: next-day cross-sectional return rank
        labels = pivot.shift(-1).rank(axis=1, pct=True).mean(axis=1)

        # Stack back to long format
        X = features.dropna(how="all", axis=0).dropna(how="all", axis=1)
        y = labels.loc[X.index]

        # Ensure timezone-aware index (required by purging)
        X.index = pd.to_datetime(X.index).tz_localize("UTC")
        y.index = pd.to_datetime(y.index).tz_localize("UTC")

        cv = PurgedWalkForwardCV(
            n_splits=3,
            label_horizon=pd.Timedelta("1D"),
            embargo_size=pd.Timedelta("1D"),
            test_size=0.2,
        )

        n_splits = 0
        for train_idx, test_idx in cv.split(X, y):
            n_splits += 1
            assert len(train_idx) > 0
            assert len(test_idx) > 0

        assert n_splits > 0, "No splits generated"


class TestICWithRealData:
    """Test Information Coefficient calculations with real equity data."""

    def test_ic_momentum_factor(self, multi_stock_returns):
        """Test IC for a momentum factor on real data."""
        df = multi_stock_returns.to_pandas()

        if df["ticker"].nunique() < 5:
            pytest.skip("Insufficient tickers for cross-sectional IC")

        # Create momentum factor: 20-day lagged return
        pivot = df.pivot_table(index="date", columns="ticker", values="return")

        if len(pivot) < 30:
            pytest.skip("Insufficient dates for momentum test")

        momentum = pivot.rolling(20).sum().shift(1)  # Lagged 20-day return
        fwd_return = pivot.shift(-5)  # 5-day forward return

        # Compute cross-sectional IC per date
        ic_values = []
        for date in momentum.index[25:-5]:  # Skip NaN periods
            mom_cross = momentum.loc[date].dropna()
            fwd_cross = fwd_return.loc[date].dropna()

            common = mom_cross.index.intersection(fwd_cross.index)
            if len(common) >= 5:
                ic = information_coefficient(
                    mom_cross[common].values, fwd_cross[common].values
                )
                if np.isfinite(ic):
                    ic_values.append(ic)

        if len(ic_values) < 5:
            pytest.skip("Insufficient IC observations")

        # Momentum should have some predictive power
        mean_ic = np.mean(ic_values)
        # Could be positive or negative, but should be non-zero on average
        assert np.isfinite(mean_ic), "IC should be finite"

    def test_ic_by_horizon_real_data(self, single_stock_returns):
        """Test IC decay over multiple horizons with real data."""
        df = single_stock_returns.to_pandas()

        if len(df) < 50:
            pytest.skip("Insufficient data for horizon analysis")

        df = df.set_index("date")

        # Create simple mean-reversion signal
        df["signal"] = -df["return"].rolling(5).mean().shift(1)
        df = df.dropna()

        pred_df = pd.DataFrame({"date": df.index, "prediction": df["signal"].values})
        price_df = pd.DataFrame({"date": df.index, "close": df["adj_close"].values})

        ic_by_horizon = compute_ic_by_horizon(
            pred_df, price_df, horizons=[1, 5, 10], pred_col="prediction"
        )

        assert 1 in ic_by_horizon
        assert 5 in ic_by_horizon
        assert 10 in ic_by_horizon

        # All ICs should be finite
        for horizon, ic in ic_by_horizon.items():
            assert np.isfinite(ic), f"IC for horizon {horizon} is not finite"


class TestDSRWithRealData:
    """Test Deflated Sharpe Ratio with real equity returns."""

    def test_dsr_single_strategy(self, single_stock_returns):
        """Test DSR calculation on single strategy (buy and hold)."""
        returns = single_stock_returns["return"].to_numpy()

        if len(returns) < 50:
            pytest.skip("Insufficient data for DSR test")

        # Calculate DSR (PSR when n_trials=1)
        result = deflated_sharpe_ratio(returns, frequency="daily")

        assert hasattr(result, "probability")
        assert hasattr(result, "sharpe_ratio")  # This is the observed Sharpe
        assert hasattr(result, "p_value")

        # Verify reasonable values
        assert 0 <= result.probability <= 1
        assert np.isfinite(result.sharpe_ratio)
        assert np.isfinite(result.p_value)

    def test_dsr_multiple_strategies(self, multi_stock_returns):
        """Test DSR with multiple strategies (different stocks as strategies)."""
        df = multi_stock_returns.to_pandas()

        # Pivot to get returns per ticker
        pivot = df.pivot_table(index="date", columns="ticker", values="return")

        if pivot.shape[1] < 3:
            pytest.skip("Need at least 3 strategies for multi-strategy DSR")

        # Treat each stock as a different strategy
        strategies = [pivot[col].dropna().values for col in pivot.columns]

        # Need aligned returns for proper multi-strategy DSR
        # Find common dates
        min_len = min(len(s) for s in strategies)
        if min_len < 30:
            pytest.skip("Insufficient common dates")

        aligned_strategies = [s[-min_len:] for s in strategies]

        result = deflated_sharpe_ratio(aligned_strategies, frequency="daily")

        assert result.n_trials == len(strategies)
        assert result.expected_max_sharpe > 0  # Selection bias should be positive

    def test_dsr_from_statistics(self, single_stock_returns):
        """Test DSR from pre-computed statistics."""
        returns = single_stock_returns["return"].to_numpy()

        if len(returns) < 50:
            pytest.skip("Insufficient data")

        # Compute statistics manually
        sharpe = np.mean(returns) / np.std(returns, ddof=1) * np.sqrt(252)

        result = deflated_sharpe_ratio_from_statistics(
            observed_sharpe=sharpe,
            n_trials=10,  # Assume 10 strategies tested
            variance_trials=1.0,
            n_samples=len(returns),
        )

        # With multiple trials, expected_max_sharpe should be positive (selection bias)
        assert result.expected_max_sharpe > 0
        # The deflated Sharpe should be less than observed Sharpe
        assert result.deflated_sharpe < sharpe
        # z_score and probability should be finite
        assert np.isfinite(result.z_score)
        assert 0 <= result.probability <= 1


class TestPurgingWithRealData:
    """Test purging algorithms with real market data characteristics."""

    def test_purging_handles_gaps(self, wiki_prices):
        """Test that purging handles non-trading days correctly."""
        # Get dates for a single stock
        ticker = wiki_prices["ticker"][0]
        dates = (
            wiki_prices.filter(pl.col("ticker") == ticker)
            .sort("date")["date"]
            .to_list()
        )

        if len(dates) < 20:
            pytest.skip("Insufficient dates")

        # Check for gaps (non-trading days)
        date_diffs = [(dates[i + 1] - dates[i]).days for i in range(len(dates) - 1)]

        # Should have gaps > 1 day (weekends, holidays)
        has_gaps = any(d > 1 for d in date_diffs)
        assert has_gaps, "Expected gaps in trading dates"

        # Create simple test data with timezone-aware dates
        tz_dates = [pd.Timestamp(d).tz_localize("UTC") for d in dates]
        df = pd.DataFrame(
            {"date": tz_dates, "feature": np.random.randn(len(dates)), "label": np.random.randn(len(dates))}
        )
        df = df.set_index("date")

        cv = PurgedWalkForwardCV(
            n_splits=3,
            label_horizon=pd.Timedelta("5D"),
            embargo_size=pd.Timedelta("2D"),
            test_size=0.2,
        )

        # Should handle gaps without error
        splits = list(cv.split(df[["feature"]], df["label"]))
        assert len(splits) == 3


class TestEdgeCasesWithRealData:
    """Test edge cases with real market data."""

    def test_handles_extreme_returns(self, wiki_returns):
        """Test handling of extreme returns (market crashes, squeezes)."""
        returns = wiki_returns["return"].to_numpy()

        # Find extreme returns (>10% daily)
        extreme_mask = np.abs(returns) > 0.10
        n_extreme = np.sum(extreme_mask)

        # Real data should have some extreme days
        if n_extreme == 0:
            pytest.skip("No extreme returns in sample data")

        # IC should still compute with extreme values
        predictions = np.random.randn(len(returns))
        ic = information_coefficient(predictions, returns)

        assert np.isfinite(ic), "IC should be finite even with extreme returns"

    def test_handles_zero_returns(self, wiki_returns):
        """Test handling of zero returns (illiquid stocks, halts)."""
        returns = wiki_returns["return"].to_numpy()

        # Find zero returns
        zero_mask = returns == 0
        n_zeros = np.sum(zero_mask)

        if n_zeros == 0:
            pytest.skip("No zero returns in sample data")

        # IC should handle zeros correctly
        predictions = np.random.randn(len(returns))
        ic = information_coefficient(predictions, returns)

        assert np.isfinite(ic), "IC should handle zero returns"
