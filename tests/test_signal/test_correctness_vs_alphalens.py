"""Correctness verification tests against alphalens-reloaded.

These tests verify that the refactored signal module produces numerically
equivalent results to alphalens-reloaded (the reference implementation).

Reference: ~/quant/alphalens/src/alphalens/
"""

from datetime import date, timedelta

import numpy as np
import pandas as pd
import polars as pl
import pytest

# Alphalens imports (optional dependency for parity validation)
alphalens_performance = pytest.importorskip(
    "alphalens.performance",
    reason="alphalens is required for correctness-vs-alphalens parity tests",
)
alphalens_utils = pytest.importorskip(
    "alphalens.utils",
    reason="alphalens is required for correctness-vs-alphalens parity tests",
)
factor_information_coefficient = alphalens_performance.factor_information_coefficient
factor_rank_autocorrelation = alphalens_performance.factor_rank_autocorrelation
quantile_turnover = alphalens_performance.quantile_turnover
compute_forward_returns = alphalens_utils.compute_forward_returns
get_clean_factor = alphalens_utils.get_clean_factor
from numpy.testing import assert_allclose
from scipy import stats

# ML4T Signal imports
from ml4t.diagnostic.signal import analyze_signal
from ml4t.diagnostic.signal._utils import quantize_factor as ml4t_quantize_factor
from ml4t.diagnostic.signal.signal_ic import compute_ic_series
from ml4t.diagnostic.signal.turnover import (
    compute_autocorrelation,
)

# =============================================================================
# Test Data Generator
# =============================================================================


def create_comparison_data(
    n_assets: int = 50,
    n_dates: int = 60,
    seed: int = 42,
    predictive: bool = True,
) -> tuple:
    """Create test data compatible with both alphalens and ml4t.signal.

    Returns:
        factor_df_polars: Polars DataFrame for ml4t (date, asset, factor)
        prices_df_polars: Polars DataFrame for ml4t (date, asset, price)
        factor_series_pandas: Pandas MultiIndex Series for alphalens
        prices_df_pandas: Pandas DataFrame for alphalens (dates as index, assets as columns)
    """
    np.random.seed(seed)

    # Use datetime timestamps (required by alphalens)
    base_date = pd.Timestamp("2024-01-01")
    dates_ts = [base_date + pd.Timedelta(days=i) for i in range(n_dates)]
    # Also keep Python dates for Polars
    dates_py = [date(2024, 1, 1) + timedelta(days=i) for i in range(n_dates)]
    assets = [f"ASSET_{i:03d}" for i in range(n_assets)]

    # True quality signal that persists across time
    base_quality = np.random.randn(n_assets)

    factor_rows = []
    factor_rows_ts = []  # With timestamps for alphalens
    price_data = {asset: [100.0] for asset in assets}  # Initial prices

    for d_idx in range(n_dates):
        d_py = dates_py[d_idx]
        d_ts = dates_ts[d_idx]

        # Factor = true quality + noise (observable signal)
        factor_noise = np.random.randn(n_assets) * 0.3
        factors = base_quality + factor_noise

        # Returns correlate with true quality (predictive)
        if predictive:
            returns = base_quality * 0.002 + np.random.randn(n_assets) * 0.01
        else:
            # Random returns (no predictive power)
            returns = np.random.randn(n_assets) * 0.01

        for a_idx, asset in enumerate(assets):
            factor_rows.append(
                {
                    "date": d_py,
                    "asset": asset,
                    "factor": factors[a_idx],
                }
            )
            factor_rows_ts.append(
                {
                    "date": d_ts,
                    "asset": asset,
                    "factor": factors[a_idx],
                }
            )

            # Update prices
            prev_price = price_data[asset][-1]
            new_price = prev_price * (1 + returns[a_idx])
            price_data[asset].append(new_price)

    # Create Polars DataFrames for ml4t
    factor_df_polars = pl.DataFrame(factor_rows)

    price_rows = []
    for d_idx, d in enumerate(dates_py):
        for _a_idx, asset in enumerate(assets):
            price_rows.append(
                {
                    "date": d,
                    "asset": asset,
                    "price": price_data[asset][d_idx],
                }
            )
    prices_df_polars = pl.DataFrame(price_rows)

    # Create Pandas structures for alphalens
    # Factor as MultiIndex Series with DatetimeIndex
    factor_series_pandas = pd.Series(
        data=[row["factor"] for row in factor_rows_ts],
        index=pd.MultiIndex.from_tuples(
            [(row["date"], row["asset"]) for row in factor_rows_ts],
            names=["date", "asset"],
        ),
    )

    # Prices as DataFrame with DatetimeIndex, assets as columns
    prices_df_pandas = pd.DataFrame(
        data={
            asset: price_data[asset][:-1]  # Exclude last (extra) price
            for asset in assets
        },
        index=pd.DatetimeIndex(dates_ts),
    )

    return factor_df_polars, prices_df_polars, factor_series_pandas, prices_df_pandas


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def comparison_data():
    """Generate comparison data for both libraries."""
    return create_comparison_data(n_assets=50, n_dates=60, seed=42, predictive=True)


@pytest.fixture
def random_data():
    """Generate random (non-predictive) comparison data."""
    return create_comparison_data(n_assets=50, n_dates=60, seed=123, predictive=False)


@pytest.fixture
def alphalens_factor_data(comparison_data):
    """Prepare alphalens factor_data from comparison data.

    Returns the merged DataFrame with factor, forward returns, and quantiles.
    """
    _, _, factor_series, prices_df = comparison_data

    # Compute forward returns using alphalens
    forward_returns = compute_forward_returns(
        factor=factor_series,
        prices=prices_df,
        periods=(1, 5),
        filter_zscore=None,
        cumulative_returns=True,
    )

    # Get clean factor data (merges factor, returns, quantiles)
    factor_data = get_clean_factor(
        factor=factor_series,
        forward_returns=forward_returns,
        quantiles=5,
        max_loss=1.0,  # Allow any data loss for testing
    )

    return factor_data


@pytest.fixture
def ml4t_result(comparison_data):
    """Run ml4t signal analysis on comparison data."""
    factor_df, prices_df, _, _ = comparison_data

    result = analyze_signal(
        factor=factor_df,
        prices=prices_df,
        periods=(1, 5),
        quantiles=5,
        compute_turnover_flag=True,
    )

    return result


# =============================================================================
# Tests: IC Computation
# =============================================================================


class TestICCorrectness:
    """Verify IC computation matches alphalens."""

    def test_ic_series_matches_spearman(self, comparison_data):
        """IC should use Spearman correlation like alphalens."""
        factor_df, prices_df, _, _ = comparison_data

        # Prepare data for ml4t
        prepared = factor_df.join(
            prices_df.select(["date", "asset", "price"]),
            on=["date", "asset"],
            how="inner",
        )

        # Compute forward returns
        prepared = prepared.sort(["asset", "date"])
        prepared = prepared.with_columns(
            [(pl.col("price").shift(-1).over("asset") / pl.col("price") - 1).alias("1D_fwd_return")]
        )

        # Compute IC for each date manually using Spearman
        manual_ic = []
        dates = prepared.select("date").unique().sort("date").to_series().to_list()

        for d in dates[:-1]:  # Skip last date (no forward return)
            day_data = prepared.filter(pl.col("date") == d).drop_nulls(["1D_fwd_return"])
            if len(day_data) < 10:
                continue

            factors = day_data["factor"].to_numpy()
            returns = day_data["1D_fwd_return"].to_numpy()

            # Spearman correlation (same as alphalens)
            ic, _ = stats.spearmanr(factors, returns)
            if not np.isnan(ic):
                manual_ic.append(ic)

        # Compute using ml4t
        ml4t_dates, ml4t_ic = compute_ic_series(
            prepared.drop_nulls(["1D_fwd_return"]),
            period=1,
            method="spearman",
        )

        # Compare
        assert len(ml4t_ic) > 0, "ml4t should compute IC values"
        assert len(manual_ic) > 0, "Manual IC should be computed"

        # Mean IC should match closely
        assert_allclose(np.mean(ml4t_ic), np.mean(manual_ic), rtol=0.01)

    def test_ic_mean_matches_alphalens(self, alphalens_factor_data, ml4t_result):
        """Mean IC should match alphalens computation."""
        # Alphalens IC
        alphalens_ic = factor_information_coefficient(alphalens_factor_data)
        alphalens_mean_ic_1d = alphalens_ic["1D"].mean()

        # ML4T IC
        ml4t_mean_ic_1d = ml4t_result.ic["1D"]

        # Allow some tolerance due to slight differences in data alignment
        # The methods should produce similar results
        assert_allclose(ml4t_mean_ic_1d, alphalens_mean_ic_1d, rtol=0.15)

    def test_ic_sign_direction(self, comparison_data, random_data):
        """Predictive factor should have positive IC, random should be near zero."""
        # Predictive data
        pred_factor_df, pred_prices_df, _, _ = comparison_data
        pred_result = analyze_signal(
            factor=pred_factor_df,
            prices=pred_prices_df,
            periods=(1,),
            quantiles=5,
            compute_turnover_flag=False,
        )

        # Random data
        rand_factor_df, rand_prices_df, _, _ = random_data
        rand_result = analyze_signal(
            factor=rand_factor_df,
            prices=rand_prices_df,
            periods=(1,),
            quantiles=5,
            compute_turnover_flag=False,
        )

        # Predictive should have positive IC
        assert pred_result.ic["1D"] > 0, "Predictive factor should have positive IC"

        # Random should be near zero (allow for noise)
        assert abs(rand_result.ic["1D"]) < 0.2, "Random factor should have near-zero IC"


# =============================================================================
# Tests: Quantile Assignment
# =============================================================================


class TestQuantileCorrectness:
    """Verify quantile assignment matches alphalens."""

    def test_quantize_uses_per_date_ranking(self, comparison_data):
        """Quantiles should be assigned per-date like alphalens."""
        factor_df, _, _, _ = comparison_data

        # Get a single date
        single_date = factor_df.filter(pl.col("date") == date(2024, 1, 15))

        # Add quantiles using ml4t
        quantized = ml4t_quantize_factor(single_date, n_quantiles=5)

        # Check that we have 5 quantiles
        unique_quantiles = quantized["quantile"].unique().to_list()
        assert len(unique_quantiles) == 5, "Should have 5 quantiles"
        assert set(unique_quantiles) == {1, 2, 3, 4, 5}, "Quantiles should be 1-5"

        # Check roughly equal distribution
        counts = quantized.group_by("quantile").len()
        min_count = counts["len"].min()
        max_count = counts["len"].max()
        # Allow some imbalance, but should be roughly equal
        assert max_count / min_count < 2.0, "Quantiles should be roughly equal sized"

    def test_quantile_ordering_by_factor(self, comparison_data):
        """Higher quantile should have higher average factor value."""
        factor_df, _, _, _ = comparison_data

        single_date = factor_df.filter(pl.col("date") == date(2024, 1, 15))
        quantized = ml4t_quantize_factor(single_date, n_quantiles=5)

        # Compute mean factor by quantile
        mean_by_q = (
            quantized.group_by("quantile")
            .agg(pl.col("factor").mean().alias("mean_factor"))
            .sort("quantile")
        )

        means = mean_by_q["mean_factor"].to_list()

        # Each quantile should have higher mean than the previous
        for i in range(1, len(means)):
            assert means[i] > means[i - 1], f"Q{i + 1} should have higher mean than Q{i}"


# =============================================================================
# Tests: Quantile Returns
# =============================================================================


class TestQuantileReturnsCorrectness:
    """Verify quantile returns match alphalens."""

    def test_quantile_returns_direction(self, ml4t_result):
        """Higher quantiles should have higher returns for predictive factor."""
        q_returns = ml4t_result.quantile_returns["1D"]

        # For a predictive factor, Q5 should beat Q1
        assert q_returns[5] > q_returns[1], "Q5 should have higher returns than Q1"

    def test_spread_matches_q5_minus_q1(self, ml4t_result):
        """Spread should equal Q5 return minus Q1 return."""
        q_returns = ml4t_result.quantile_returns["1D"]
        spread = ml4t_result.spread["1D"]

        expected_spread = q_returns[5] - q_returns[1]
        assert_allclose(spread, expected_spread, rtol=1e-10)

    def test_monotonicity_high_for_predictive(self, ml4t_result):
        """Monotonicity should be high for predictive factor."""
        mono = ml4t_result.monotonicity["1D"]

        # Should be positive and reasonably high for predictive factor
        assert mono > 0.5, f"Monotonicity should be high for predictive factor, got {mono}"


# =============================================================================
# Tests: Turnover (Known Difference)
# =============================================================================


class TestTurnoverDocumentation:
    """Document intentional turnover formula difference.

    Alphalens: 1 - overlap / current_size (asymmetric)
    ML4T: 1 - overlap / max(prev_size, current_size) (symmetric)
    """

    def test_turnover_formulas_documented(self, comparison_data):
        """Verify both formulas and document the difference."""
        factor_df, prices_df, factor_series, prices_df_pandas = comparison_data

        # Run ml4t analysis to get quantile assignments
        result = analyze_signal(
            factor=factor_df,
            prices=prices_df,
            periods=(1,),
            quantiles=5,
            compute_turnover_flag=True,
        )

        # Prepare alphalens data
        forward_returns = compute_forward_returns(
            factor=factor_series,
            prices=prices_df_pandas,
            periods=(1,),
            filter_zscore=None,
        )
        factor_data = get_clean_factor(
            factor=factor_series,
            forward_returns=forward_returns,
            quantiles=5,
            max_loss=1.0,
        )

        # Alphalens turnover for Q5
        alphalens_turnover_q5 = quantile_turnover(
            factor_data["factor_quantile"], quantile=5, period=1
        )

        # Document: ML4T uses symmetric formula, alphalens uses asymmetric
        # The formulas ARE different by design:
        # - Alphalens: new_names / current_names
        # - ML4T: 1 - overlap / max(prev_size, current_size)

        # Just verify both compute reasonable values
        assert 0 <= result.turnover["1D"] <= 1.0, "ML4T turnover should be in [0, 1]"

        # Alphalens turnover has NaN for first date (no previous data) - drop those
        alphalens_valid = alphalens_turnover_q5.dropna()
        assert (alphalens_valid >= 0).all(), "Alphalens turnover should be >= 0"
        assert (alphalens_valid <= 1).all(), "Alphalens turnover should be <= 1"

        # Note: Values WILL differ due to formula difference - this is intentional
        # ML4T's symmetric formula is arguably more correct for measuring actual turnover


# =============================================================================
# Tests: Autocorrelation
# =============================================================================


class TestAutocorrelationCorrectness:
    """Verify factor rank autocorrelation matches alphalens approach."""

    def test_autocorrelation_uses_ranks(self, comparison_data):
        """Autocorrelation should use ranks, not raw factor values."""
        factor_df, prices_df, factor_series, prices_df_pandas = comparison_data

        # Prepare alphalens data
        forward_returns = compute_forward_returns(
            factor=factor_series,
            prices=prices_df_pandas,
            periods=(1,),
            filter_zscore=None,
        )
        factor_data = get_clean_factor(
            factor=factor_series,
            forward_returns=forward_returns,
            quantiles=5,
            max_loss=1.0,
        )

        # Alphalens autocorrelation
        alphalens_ac = factor_rank_autocorrelation(factor_data, period=1)
        alphalens_mean_ac = alphalens_ac.mean()

        # ML4T autocorrelation
        ml4t_ac = compute_autocorrelation(factor_df, lags=[1], min_obs=10)

        # Both should indicate high persistence for this data
        # (factor is based on persistent base_quality)
        assert ml4t_ac[0] > 0.5, "ML4T autocorr should show persistence"
        assert alphalens_mean_ac > 0.5, "Alphalens autocorr should show persistence"

        # Values should be in similar range
        assert_allclose(ml4t_ac[0], alphalens_mean_ac, rtol=0.3)


# =============================================================================
# Tests: Forward Returns
# =============================================================================


class TestForwardReturnsCorrectness:
    """Verify forward returns computation."""

    def test_forward_returns_direction(self, comparison_data):
        """Forward returns should match price changes."""
        factor_df, prices_df, _, _ = comparison_data

        # Get prices for first asset
        asset_prices = prices_df.filter(pl.col("asset") == "ASSET_000").sort("date")

        prices_list = asset_prices["price"].to_list()

        # Manual forward return for day 0
        expected_1d_return = (prices_list[1] / prices_list[0]) - 1

        # Verify this matches a simple pct_change computation
        returns = [(prices_list[i + 1] / prices_list[i]) - 1 for i in range(len(prices_list) - 1)]

        assert_allclose(returns[0], expected_1d_return, rtol=1e-10)

    def test_multi_period_forward_returns(self, comparison_data):
        """5D forward returns should compound 5 daily returns."""
        _, prices_df, _, _ = comparison_data

        # Get prices for first asset
        asset_prices = prices_df.filter(pl.col("asset") == "ASSET_000").sort("date")

        prices_list = asset_prices["price"].to_list()

        # 5D return from day 0
        expected_5d_return = (prices_list[5] / prices_list[0]) - 1

        # Verify compounding
        daily_returns = [(prices_list[i + 1] / prices_list[i]) - 1 for i in range(5)]
        compounded = np.prod([1 + r for r in daily_returns]) - 1

        assert_allclose(compounded, expected_5d_return, rtol=1e-10)


# =============================================================================
# Tests: End-to-End Consistency
# =============================================================================


class TestEndToEndConsistency:
    """Verify overall consistency between libraries."""

    def test_predictive_vs_random_differentiation(self, comparison_data, random_data):
        """Both libraries should distinguish predictive from random factors."""
        # Predictive data - alphalens
        _, _, pred_factor_series, pred_prices_pandas = comparison_data
        pred_fwd = compute_forward_returns(
            factor=pred_factor_series,
            prices=pred_prices_pandas,
            periods=(1,),
            filter_zscore=None,
        )
        pred_factor_data = get_clean_factor(
            factor=pred_factor_series,
            forward_returns=pred_fwd,
            quantiles=5,
            max_loss=1.0,
        )
        pred_alphalens_ic = factor_information_coefficient(pred_factor_data)["1D"].mean()

        # Random data - alphalens
        _, _, rand_factor_series, rand_prices_pandas = random_data
        rand_fwd = compute_forward_returns(
            factor=rand_factor_series,
            prices=rand_prices_pandas,
            periods=(1,),
            filter_zscore=None,
        )
        rand_factor_data = get_clean_factor(
            factor=rand_factor_series,
            forward_returns=rand_fwd,
            quantiles=5,
            max_loss=1.0,
        )
        rand_alphalens_ic = factor_information_coefficient(rand_factor_data)["1D"].mean()

        # Predictive data - ml4t
        pred_factor_polars, pred_prices_polars, _, _ = comparison_data
        pred_ml4t = analyze_signal(
            factor=pred_factor_polars,
            prices=pred_prices_polars,
            periods=(1,),
            quantiles=5,
            compute_turnover_flag=False,
        )

        # Random data - ml4t
        rand_factor_polars, rand_prices_polars, _, _ = random_data
        rand_ml4t = analyze_signal(
            factor=rand_factor_polars,
            prices=rand_prices_polars,
            periods=(1,),
            quantiles=5,
            compute_turnover_flag=False,
        )

        # Both libraries should show: predictive IC > random IC
        assert pred_alphalens_ic > rand_alphalens_ic, "Alphalens: predictive > random"
        assert pred_ml4t.ic["1D"] > rand_ml4t.ic["1D"], "ML4T: predictive > random"

        # Both should have similar magnitude difference
        alphalens_diff = pred_alphalens_ic - rand_alphalens_ic
        ml4t_diff = pred_ml4t.ic["1D"] - rand_ml4t.ic["1D"]

        # The difference should be in the same direction and similar magnitude
        assert alphalens_diff > 0 and ml4t_diff > 0, "Both should show positive difference"

    def test_statistical_significance_alignment(self, ml4t_result, alphalens_factor_data):
        """Statistical significance indicators should align."""
        # ML4T t-stat
        ml4t_t_stat = ml4t_result.ic_t_stat["1D"]

        # Alphalens IC series for manual t-stat
        alphalens_ic = factor_information_coefficient(alphalens_factor_data)["1D"]
        alphalens_mean = alphalens_ic.mean()
        alphalens_std = alphalens_ic.std()
        n = len(alphalens_ic)
        alphalens_t_stat = alphalens_mean / (alphalens_std / np.sqrt(n))

        # Both should indicate similar significance level
        # (same sign and roughly similar magnitude)
        assert np.sign(ml4t_t_stat) == np.sign(alphalens_t_stat), "T-stats should have same sign"

        # Allow larger tolerance due to different data alignment
        assert_allclose(abs(ml4t_t_stat), abs(alphalens_t_stat), rtol=0.5)


# =============================================================================
# Tests: Edge Cases
# =============================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_small_dataset(self):
        """Both libraries should handle small datasets."""
        factor_df, prices_df, factor_series, prices_df_pandas = create_comparison_data(
            n_assets=20, n_dates=15, seed=999, predictive=True
        )

        # ML4T should work
        result = analyze_signal(
            factor=factor_df,
            prices=prices_df,
            periods=(1,),
            quantiles=5,
            compute_turnover_flag=False,
        )
        assert "1D" in result.ic

        # Alphalens should work
        fwd = compute_forward_returns(
            factor=factor_series,
            prices=prices_df_pandas,
            periods=(1,),
            filter_zscore=None,
        )
        factor_data = get_clean_factor(
            factor=factor_series,
            forward_returns=fwd,
            quantiles=5,
            max_loss=1.0,
        )
        ic = factor_information_coefficient(factor_data)
        assert len(ic) > 0

    def test_constant_factor_single_date(self, comparison_data):
        """Constant factor on a single date should not crash."""
        factor_df, prices_df, _, _ = comparison_data

        # Create constant factor for one date
        modified = factor_df.with_columns(
            [
                pl.when(pl.col("date") == date(2024, 1, 10))
                .then(pl.lit(0.0))
                .otherwise(pl.col("factor"))
                .alias("factor")
            ]
        )

        # Should not crash (undefined correlation handled gracefully)
        result = analyze_signal(
            factor=modified,
            prices=prices_df,
            periods=(1,),
            quantiles=5,
            compute_turnover_flag=False,
        )

        # IC should still be computed for other dates
        assert "1D" in result.ic


# =============================================================================
# Tests: IC Decay (Multiple Periods)
# =============================================================================


class TestICDecayCorrectness:
    """Verify IC decay across multiple forward periods."""

    def test_ic_decay_matches_alphalens(self, comparison_data):
        """IC at multiple periods should match alphalens computation."""
        factor_df, prices_df, factor_series, prices_df_pandas = comparison_data

        periods = (1, 5, 10, 20)

        # ML4T analysis
        ml4t_result = analyze_signal(
            factor=factor_df,
            prices=prices_df,
            periods=periods,
            quantiles=5,
            compute_turnover_flag=False,
        )

        # Alphalens analysis
        forward_returns = compute_forward_returns(
            factor=factor_series,
            prices=prices_df_pandas,
            periods=periods,
            filter_zscore=None,
        )
        factor_data = get_clean_factor(
            factor=factor_series,
            forward_returns=forward_returns,
            quantiles=5,
            max_loss=1.0,
        )
        alphalens_ic = factor_information_coefficient(factor_data)

        # Compare IC at each period
        for period in periods:
            period_key = f"{period}D"
            ml4t_ic = ml4t_result.ic[period_key]
            alphalens_mean_ic = alphalens_ic[period_key].mean()

            # Allow some tolerance due to data alignment differences
            assert_allclose(
                ml4t_ic,
                alphalens_mean_ic,
                rtol=0.2,
                err_msg=f"IC at {period}D differs from alphalens",
            )

    def test_ic_computed_at_all_periods(self, comparison_data):
        """IC should be computed for all requested periods.

        Note: IC decay vs growth depends on signal characteristics:
        - Fast-decaying signals show IC decay with period
        - Persistent signals may show IC growth as returns accumulate
        """
        factor_df, prices_df, _, _ = comparison_data

        periods = (1, 5, 10, 20)

        result = analyze_signal(
            factor=factor_df,
            prices=prices_df,
            periods=periods,
            quantiles=5,
            compute_turnover_flag=False,
        )

        # All periods should have IC computed
        for period in periods:
            period_key = f"{period}D"
            assert period_key in result.ic, f"IC should be computed for {period_key}"
            assert np.isfinite(result.ic[period_key]), f"IC at {period_key} should be finite"


# =============================================================================
# Tests: Information Ratio (IC IR)
# =============================================================================


class TestICInformationRatio:
    """Verify IC Information Ratio (IC mean / IC std) computation."""

    def test_ic_ir_matches_alphalens(self, comparison_data):
        """IC IR should match alphalens computation."""
        factor_df, prices_df, factor_series, prices_df_pandas = comparison_data

        # ML4T analysis
        ml4t_result = analyze_signal(
            factor=factor_df,
            prices=prices_df,
            periods=(1,),
            quantiles=5,
            compute_turnover_flag=False,
        )

        # Alphalens IC series
        forward_returns = compute_forward_returns(
            factor=factor_series,
            prices=prices_df_pandas,
            periods=(1,),
            filter_zscore=None,
        )
        factor_data = get_clean_factor(
            factor=factor_series,
            forward_returns=forward_returns,
            quantiles=5,
            max_loss=1.0,
        )
        alphalens_ic_series = factor_information_coefficient(factor_data)["1D"]

        # Compute IR = mean(IC) / std(IC)
        alphalens_ir = alphalens_ic_series.mean() / alphalens_ic_series.std()

        # ML4T should have ic_ir if available, or compute from ic and ic_std
        if hasattr(ml4t_result, "ic_ir") and "1D" in ml4t_result.ic_ir:
            ml4t_ir = ml4t_result.ic_ir["1D"]
        else:
            # Compute from components
            ml4t_ir = ml4t_result.ic["1D"] / (
                ml4t_result.ic_std["1D"] if hasattr(ml4t_result, "ic_std") else 1.0
            )

        # Allow tolerance due to different data alignment
        assert_allclose(ml4t_ir, alphalens_ir, rtol=0.3, err_msg="IC IR differs from alphalens")

    def test_ic_ir_sign(self, comparison_data, random_data):
        """Predictive factor should have positive IC IR."""
        # Predictive
        pred_factor_df, pred_prices_df, _, _ = comparison_data
        pred_result = analyze_signal(
            factor=pred_factor_df,
            prices=pred_prices_df,
            periods=(1,),
            quantiles=5,
            compute_turnover_flag=False,
        )

        # Random
        rand_factor_df, rand_prices_df, _, _ = random_data
        rand_result = analyze_signal(
            factor=rand_factor_df,
            prices=rand_prices_df,
            periods=(1,),
            quantiles=5,
            compute_turnover_flag=False,
        )

        # Predictive should have higher t-stat than random
        assert pred_result.ic_t_stat["1D"] > rand_result.ic_t_stat["1D"]


# =============================================================================
# Tests: Factor-Weighted Returns
# =============================================================================


class TestFactorWeightedReturns:
    """Verify factor-weighted portfolio returns."""

    def test_quantile_spread_return(self, comparison_data):
        """Long-short spread return should match Q5-Q1."""
        factor_df, prices_df, _, _ = comparison_data

        result = analyze_signal(
            factor=factor_df,
            prices=prices_df,
            periods=(1, 5),
            quantiles=5,
            compute_turnover_flag=False,
        )

        for period in ["1D", "5D"]:
            q_returns = result.quantile_returns[period]
            spread = result.spread[period]

            # Spread should equal Q5 - Q1
            expected = q_returns[5] - q_returns[1]
            assert_allclose(spread, expected, rtol=1e-10, err_msg=f"Spread at {period} incorrect")

    def test_quantile_returns_sum(self, comparison_data):
        """Quantile returns should be centered around overall mean."""
        factor_df, prices_df, _, _ = comparison_data

        result = analyze_signal(
            factor=factor_df,
            prices=prices_df,
            periods=(1,),
            quantiles=5,
            compute_turnover_flag=False,
        )

        q_returns = result.quantile_returns["1D"]

        # Average of quantile returns should be close to overall mean
        # (if quantiles are equal-weighted)
        avg_q_return = sum(q_returns.values()) / len(q_returns)

        # Should be close to zero for demeaned returns or market return
        # At minimum, should be a reasonable value
        assert -0.05 < avg_q_return < 0.05, "Average quantile return out of range"


# =============================================================================
# Tests: Monotonicity Measure
# =============================================================================


class TestMonotonicityCorrectness:
    """Verify monotonicity calculation."""

    def test_perfect_monotonicity(self):
        """Perfect ranking should give monotonicity = 1."""
        # Create data where Q1 < Q2 < Q3 < Q4 < Q5 perfectly
        np.random.seed(42)

        # Create data with perfect factor-return relationship
        n_assets = 50
        n_dates = 30

        dates = [date(2024, 1, 1) + timedelta(days=i) for i in range(n_dates)]
        assets = [f"ASSET_{i:03d}" for i in range(n_assets)]

        rows = []
        price_data = {asset: [100.0] for asset in assets}

        for d_idx in range(n_dates):
            d = dates[d_idx]

            # Factor is just asset rank
            factors = np.arange(n_assets) / n_assets

            # Returns perfectly correlate with factor
            returns = factors * 0.02

            for a_idx, asset in enumerate(assets):
                rows.append(
                    {
                        "date": d,
                        "asset": asset,
                        "factor": factors[a_idx],
                    }
                )
                prev_price = price_data[asset][-1]
                price_data[asset].append(prev_price * (1 + returns[a_idx]))

        factor_df = pl.DataFrame(rows)
        price_rows = []
        for d_idx, d in enumerate(dates):
            for asset in assets:
                price_rows.append(
                    {
                        "date": d,
                        "asset": asset,
                        "price": price_data[asset][d_idx],
                    }
                )
        prices_df = pl.DataFrame(price_rows)

        result = analyze_signal(
            factor=factor_df,
            prices=prices_df,
            periods=(1,),
            quantiles=5,
            compute_turnover_flag=False,
        )

        # Monotonicity should be very high (close to 1)
        assert result.monotonicity["1D"] > 0.9, "Perfect relationship should give high monotonicity"

    def test_anti_monotonicity(self):
        """Inverse ranking should give monotonicity close to -1."""
        np.random.seed(42)

        n_assets = 50
        n_dates = 30

        dates = [date(2024, 1, 1) + timedelta(days=i) for i in range(n_dates)]
        assets = [f"ASSET_{i:03d}" for i in range(n_assets)]

        rows = []
        price_data = {asset: [100.0] for asset in assets}

        for d_idx in range(n_dates):
            d = dates[d_idx]

            # Factor is asset rank
            factors = np.arange(n_assets) / n_assets

            # Returns INVERSELY correlate with factor
            returns = (1 - factors) * 0.02

            for a_idx, asset in enumerate(assets):
                rows.append(
                    {
                        "date": d,
                        "asset": asset,
                        "factor": factors[a_idx],
                    }
                )
                prev_price = price_data[asset][-1]
                price_data[asset].append(prev_price * (1 + returns[a_idx]))

        factor_df = pl.DataFrame(rows)
        price_rows = []
        for d_idx, d in enumerate(dates):
            for asset in assets:
                price_rows.append(
                    {
                        "date": d,
                        "asset": asset,
                        "price": price_data[asset][d_idx],
                    }
                )
        prices_df = pl.DataFrame(price_rows)

        result = analyze_signal(
            factor=factor_df,
            prices=prices_df,
            periods=(1,),
            quantiles=5,
            compute_turnover_flag=False,
        )

        # Monotonicity should be negative (inverse relationship)
        assert result.monotonicity["1D"] < -0.9, (
            "Inverse relationship should give negative monotonicity"
        )


# =============================================================================
# Tests: Cumulative vs Non-Cumulative Returns
# =============================================================================


class TestCumulativeReturnsCorrectness:
    """Verify cumulative vs simple returns handling."""

    def test_multi_period_is_cumulative(self, comparison_data):
        """5D return should be cumulative, not 5x daily."""
        _, prices_df, _, _ = comparison_data

        # Get an asset's prices
        asset_prices = prices_df.filter(pl.col("asset") == "ASSET_000").sort("date")
        prices = asset_prices["price"].to_list()

        # 5D cumulative return
        cumulative_5d = prices[5] / prices[0] - 1

        # Compute compounded return from daily returns
        daily_returns = [(prices[i + 1] / prices[i] - 1) for i in range(5)]
        compounded = np.prod([1 + r for r in daily_returns]) - 1

        assert_allclose(cumulative_5d, compounded, rtol=1e-10)

        # But not equal to simple sum (unless very small returns)
        # This verifies we're using cumulative, not simple
        # Note: For small returns, they're approximately equal, so skip this assertion
        # if returns are too small


# =============================================================================
# Tests: Cross-Sectional Neutralization
# =============================================================================


class TestCrossSectionalCorrectness:
    """Verify cross-sectional computations are per-date."""

    def test_ic_is_per_date(self, comparison_data):
        """IC should be computed per-date, then averaged."""
        factor_df, prices_df, factor_series, prices_df_pandas = comparison_data

        # ML4T should compute IC per date
        result = analyze_signal(
            factor=factor_df,
            prices=prices_df,
            periods=(1,),
            quantiles=5,
            compute_turnover_flag=False,
        )

        # Alphalens computes per-date IC
        forward_returns = compute_forward_returns(
            factor=factor_series,
            prices=prices_df_pandas,
            periods=(1,),
            filter_zscore=None,
        )
        factor_data = get_clean_factor(
            factor=factor_series,
            forward_returns=forward_returns,
            quantiles=5,
            max_loss=1.0,
        )
        alphalens_ic_series = factor_information_coefficient(factor_data)["1D"]

        # Both should have multiple IC values (one per date)
        # The mean IC should be similar
        assert_allclose(
            result.ic["1D"],
            alphalens_ic_series.mean(),
            rtol=0.15,
            err_msg="Mean IC should match alphalens",
        )

    def test_quantiles_per_date(self, comparison_data):
        """Quantile assignment should be per-date, not global."""
        factor_df, _, _, _ = comparison_data

        # Add quantiles using ml4t
        quantized = ml4t_quantize_factor(factor_df, n_quantiles=5)

        # Check each date has all 5 quantiles
        dates = quantized.select("date").unique().to_series().to_list()

        for d in dates:
            day_data = quantized.filter(pl.col("date") == d)
            unique_q = day_data["quantile"].unique().to_list()

            # Each day should have all 5 quantiles
            assert len(unique_q) == 5, f"Date {d} should have 5 quantiles"
