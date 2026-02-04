"""Tests for signal analysis utility functions.

Tests _utils.py: compute_forward_returns, quantize_factor, filter_outliers, ensure_polars.
"""

from datetime import date

import pandas as pd
import polars as pl
import pytest

from ml4t.diagnostic.signal._utils import (
    QuantileMethod,
    compute_forward_returns,
    ensure_polars,
    filter_outliers,
    quantize_factor,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def simple_factor_data():
    """Simple factor data: 3 dates, 4 assets."""
    return pl.DataFrame(
        {
            "date": [
                date(2024, 1, 1),
                date(2024, 1, 1),
                date(2024, 1, 1),
                date(2024, 1, 1),
                date(2024, 1, 2),
                date(2024, 1, 2),
                date(2024, 1, 2),
                date(2024, 1, 2),
                date(2024, 1, 3),
                date(2024, 1, 3),
                date(2024, 1, 3),
                date(2024, 1, 3),
            ],
            "asset": ["A", "B", "C", "D"] * 3,
            "factor": [1.0, 2.0, 3.0, 4.0, 1.5, 2.5, 3.5, 4.5, 2.0, 3.0, 4.0, 5.0],
        }
    )


@pytest.fixture
def simple_price_data():
    """Simple price data: 5 dates (for forward returns), 4 assets."""
    return pl.DataFrame(
        {
            "date": [
                date(2024, 1, 1),
                date(2024, 1, 1),
                date(2024, 1, 1),
                date(2024, 1, 1),
                date(2024, 1, 2),
                date(2024, 1, 2),
                date(2024, 1, 2),
                date(2024, 1, 2),
                date(2024, 1, 3),
                date(2024, 1, 3),
                date(2024, 1, 3),
                date(2024, 1, 3),
                date(2024, 1, 4),
                date(2024, 1, 4),
                date(2024, 1, 4),
                date(2024, 1, 4),
                date(2024, 1, 5),
                date(2024, 1, 5),
                date(2024, 1, 5),
                date(2024, 1, 5),
            ],
            "asset": ["A", "B", "C", "D"] * 5,
            "price": [
                100.0,
                100.0,
                100.0,
                100.0,  # Day 1
                101.0,
                102.0,
                103.0,
                104.0,  # Day 2
                102.0,
                104.0,
                106.0,
                108.0,  # Day 3
                103.0,
                106.0,
                109.0,
                112.0,  # Day 4
                104.0,
                108.0,
                112.0,
                116.0,  # Day 5
            ],
        }
    )


# =============================================================================
# Tests: compute_forward_returns
# =============================================================================


class TestComputeForwardReturns:
    """Tests for compute_forward_returns function."""

    def test_basic(self, simple_factor_data, simple_price_data):
        """Test basic forward return computation."""
        result = compute_forward_returns(simple_factor_data, simple_price_data, periods=(1,))

        assert "1D_fwd_return" in result.columns
        # Day 1, Asset A: (101-100)/100 = 0.01
        day1_a = result.filter((pl.col("date") == date(2024, 1, 1)) & (pl.col("asset") == "A"))
        assert abs(day1_a["1D_fwd_return"][0] - 0.01) < 1e-10

    def test_multiple_periods(self, simple_factor_data, simple_price_data):
        """Test forward returns with multiple periods."""
        result = compute_forward_returns(simple_factor_data, simple_price_data, periods=(1, 2))

        assert "1D_fwd_return" in result.columns
        assert "2D_fwd_return" in result.columns

        # Day 1, Asset A: 2D return = (102-100)/100 = 0.02
        day1_a = result.filter((pl.col("date") == date(2024, 1, 1)) & (pl.col("asset") == "A"))
        assert abs(day1_a["2D_fwd_return"][0] - 0.02) < 1e-10

    def test_missing_asset_in_prices(self, simple_factor_data, simple_price_data):
        """Test when factor has asset not in prices."""
        # Add asset E to factor but not prices
        extra_row = pl.DataFrame({"date": [date(2024, 1, 1)], "asset": ["E"], "factor": [5.0]})
        factor_with_extra = pl.concat([simple_factor_data, extra_row])

        result = compute_forward_returns(factor_with_extra, simple_price_data, periods=(1,))

        # Asset E should have None for forward returns
        asset_e = result.filter(pl.col("asset") == "E")
        assert asset_e["1D_fwd_return"][0] is None

    def test_forward_return_uses_price_calendar(self, simple_factor_data):
        """Test that forward returns use price date calendar, not factor dates.

        When computing N-day forward returns, we use the Nth PRICE date forward,
        not the Nth factor date. This is the correct behavior for trading:
        "1 day forward" means the next trading day in the price calendar.
        """
        # Prices have: day 1 (idx 0), day 3 (idx 1) - skipping day 2
        prices_sparse = pl.DataFrame(
            {
                "date": [
                    date(2024, 1, 1),
                    date(2024, 1, 1),
                    date(2024, 1, 3),
                    date(2024, 1, 3),
                ],
                "asset": ["A", "B", "A", "B"],
                "price": [100.0, 100.0, 102.0, 104.0],
            }
        )

        # Factor only has day 1
        factor_day1 = simple_factor_data.filter(pl.col("date") == date(2024, 1, 1))

        result = compute_forward_returns(factor_day1, prices_sparse, periods=(1,))

        # 1D forward from day 1 uses next PRICE date (day 3), not calendar day 2
        # Return = (102 - 100) / 100 = 0.02 for asset A
        assert result.filter(pl.col("asset") == "A")["1D_fwd_return"][0] == pytest.approx(0.02)
        # Return = (104 - 100) / 100 = 0.04 for asset B
        assert result.filter(pl.col("asset") == "B")["1D_fwd_return"][0] == pytest.approx(0.04)

    def test_nan_price(self):
        """Test when current or future price is NaN."""
        # Factor has 2 dates so we can compute 1D forward returns
        factor = pl.DataFrame(
            {
                "date": [date(2024, 1, 1), date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 2)],
                "asset": ["A", "B", "A", "B"],
                "factor": [1.0, 2.0, 1.5, 2.5],
            }
        )
        prices_with_nan = pl.DataFrame(
            {
                "date": [
                    date(2024, 1, 1),
                    date(2024, 1, 1),
                    date(2024, 1, 2),
                    date(2024, 1, 2),
                ],
                "asset": ["A", "B", "A", "B"],
                "price": [float("nan"), 100.0, 101.0, 102.0],  # A has NaN on day 1
            }
        )

        result = compute_forward_returns(factor, prices_with_nan, periods=(1,))

        # Day 1, Asset A should have None (NaN current price)
        day1_a = result.filter((pl.col("date") == date(2024, 1, 1)) & (pl.col("asset") == "A"))
        assert day1_a["1D_fwd_return"][0] is None

        # Day 1, Asset B should have valid return: (102-100)/100 = 0.02
        day1_b = result.filter((pl.col("date") == date(2024, 1, 1)) & (pl.col("asset") == "B"))
        assert day1_b["1D_fwd_return"][0] is not None
        assert abs(day1_b["1D_fwd_return"][0] - 0.02) < 1e-10

    def test_end_of_series(self, simple_factor_data, simple_price_data):
        """Test forward returns use PRICE calendar, not factor calendar.

        Factor has 3 dates (1, 2, 3) but prices have 5 dates (1, 2, 3, 4, 5).
        Forward returns should use the price calendar, so all factor dates
        can compute 2D returns because price dates 4 and 5 exist.
        """
        result = compute_forward_returns(simple_factor_data, simple_price_data, periods=(2,))

        # Prices have 5 dates (1, 2, 3, 4, 5). For 2D returns:
        # - Day 1 (price idx 0): needs price idx 2 (day 3) - valid
        # - Day 2 (price idx 1): needs price idx 3 (day 4) - valid
        # - Day 3 (price idx 2): needs price idx 4 (day 5) - valid

        day1 = result.filter(pl.col("date") == date(2024, 1, 1))
        day2 = result.filter(pl.col("date") == date(2024, 1, 2))
        day3 = result.filter(pl.col("date") == date(2024, 1, 3))

        # All days should have valid 2D returns (price dates 3, 4, 5 exist)
        assert all(r is not None for r in day1["2D_fwd_return"].to_list())
        assert all(r is not None for r in day2["2D_fwd_return"].to_list())
        assert all(r is not None for r in day3["2D_fwd_return"].to_list())

        # Verify actual return values (day 1 asset A: day 3 price / day 1 price - 1)
        # Day 1 asset A: (102 - 100) / 100 = 0.02
        day1_a = result.filter((pl.col("date") == date(2024, 1, 1)) & (pl.col("asset") == "A"))
        assert day1_a["2D_fwd_return"][0] == pytest.approx(0.02)

    def test_end_of_series_insufficient_price_dates(self):
        """Test None when insufficient future PRICE dates."""
        factor = pl.DataFrame(
            {
                "date": [date(2024, 1, 3)],
                "asset": ["A"],
                "factor": [1.0],
            }
        )
        # Only 2 price dates available
        prices = pl.DataFrame(
            {
                "date": [date(2024, 1, 3), date(2024, 1, 4)],
                "asset": ["A", "A"],
                "price": [100.0, 101.0],
            }
        )

        result = compute_forward_returns(factor, prices, periods=(5,))

        # 5D forward from price idx 0 needs price idx 5, but only 2 price dates exist
        assert result["5D_fwd_return"][0] is None

    def test_single_asset(self):
        """Test with single asset."""
        factor = pl.DataFrame(
            {
                "date": [date(2024, 1, 1), date(2024, 1, 2)],
                "asset": ["A", "A"],
                "factor": [1.0, 2.0],
            }
        )
        prices = pl.DataFrame(
            {
                "date": [
                    date(2024, 1, 1),
                    date(2024, 1, 2),
                    date(2024, 1, 3),
                ],
                "asset": ["A", "A", "A"],
                "price": [100.0, 110.0, 121.0],
            }
        )

        result = compute_forward_returns(factor, prices, periods=(1,))

        assert result.height == 2
        # Day 1: (110-100)/100 = 0.10
        assert abs(result["1D_fwd_return"][0] - 0.10) < 1e-10


# =============================================================================
# Tests: quantize_factor
# =============================================================================


class TestQuantizeFactor:
    """Tests for quantize_factor function."""

    def test_quantile_method(self, simple_factor_data):
        """Test QUANTILE method (equal frequency)."""
        result = quantize_factor(simple_factor_data, n_quantiles=4, method=QuantileMethod.QUANTILE)

        assert "quantile" in result.columns
        # Each date should have assets in quantiles 1-4
        for d in result["date"].unique():
            date_data = result.filter(pl.col("date") == d)
            quantiles = set(date_data["quantile"].to_list())
            assert quantiles == {1, 2, 3, 4}

    def test_uniform_method(self, simple_factor_data):
        """Test UNIFORM method (equal width bins)."""
        result = quantize_factor(simple_factor_data, n_quantiles=4, method=QuantileMethod.UNIFORM)

        assert "quantile" in result.columns
        # UNIFORM uses equal width bins based on min/max
        # Day 1: factors 1,2,3,4, range=3, width=0.75
        # Bins: [1, 1.75), [1.75, 2.5), [2.5, 3.25), [3.25, 4]
        day1 = result.filter(pl.col("date") == date(2024, 1, 1))
        assert 1 in day1["quantile"].to_list()
        assert 4 in day1["quantile"].to_list()

    def test_uniform_all_same_value(self):
        """Test UNIFORM method when all values are identical."""
        data = pl.DataFrame(
            {
                "date": [date(2024, 1, 1)] * 4,
                "asset": ["A", "B", "C", "D"],
                "factor": [5.0, 5.0, 5.0, 5.0],  # All same
            }
        )

        result = quantize_factor(data, n_quantiles=4, method=QuantileMethod.UNIFORM)

        # With epsilon handling, all should be in quantile 1
        assert all(q == 1 for q in result["quantile"].to_list())

    def test_uniform_extreme_outlier(self):
        """Test UNIFORM method with one extreme outlier."""
        data = pl.DataFrame(
            {
                "date": [date(2024, 1, 1)] * 4,
                "asset": ["A", "B", "C", "D"],
                "factor": [1.0, 2.0, 3.0, 100.0],  # D is extreme outlier
            }
        )

        result = quantize_factor(data, n_quantiles=4, method=QuantileMethod.UNIFORM)

        # With equal width bins, A,B,C should all be in quantile 1, D in quantile 4
        assert result.filter(pl.col("asset") == "D")["quantile"][0] == 4
        # A,B,C should be in lower quantiles (likely 1)
        assert result.filter(pl.col("asset") == "A")["quantile"][0] == 1

    def test_single_asset_per_date(self):
        """Test with single asset per date."""
        data = pl.DataFrame(
            {
                "date": [date(2024, 1, 1), date(2024, 1, 2)],
                "asset": ["A", "A"],
                "factor": [1.0, 2.0],
            }
        )

        result = quantize_factor(data, n_quantiles=5, method=QuantileMethod.QUANTILE)

        # Single asset should be in quantile 1 (rank 0)
        assert all(q == 1 for q in result["quantile"].to_list())

    def test_n_quantiles_equals_n_assets(self):
        """Test when n_quantiles equals number of assets."""
        data = pl.DataFrame(
            {
                "date": [date(2024, 1, 1)] * 4,
                "asset": ["A", "B", "C", "D"],
                "factor": [1.0, 2.0, 3.0, 4.0],
            }
        )

        result = quantize_factor(data, n_quantiles=4, method=QuantileMethod.QUANTILE)

        # Should get one asset per quantile
        quantiles = sorted(result["quantile"].to_list())
        assert quantiles == [1, 2, 3, 4]

    def test_n_quantiles_greater_than_n_assets(self):
        """Test when n_quantiles > number of assets."""
        data = pl.DataFrame(
            {
                "date": [date(2024, 1, 1)] * 3,
                "asset": ["A", "B", "C"],
                "factor": [1.0, 2.0, 3.0],
            }
        )

        result = quantize_factor(data, n_quantiles=10, method=QuantileMethod.QUANTILE)

        # With 3 assets and 10 quantiles, each asset spans ~3.33 quantiles
        # All should be assigned to some quantile
        assert result.height == 3
        assert all(1 <= q <= 10 for q in result["quantile"].to_list())

    def test_ties(self):
        """Test handling of tied factor values."""
        data = pl.DataFrame(
            {
                "date": [date(2024, 1, 1)] * 4,
                "asset": ["A", "B", "C", "D"],
                "factor": [1.0, 2.0, 2.0, 3.0],  # B and C tied
            }
        )

        result = quantize_factor(data, n_quantiles=4, method=QuantileMethod.QUANTILE)

        # Should still assign all to quantiles
        assert result.height == 4
        # B and C might get same or adjacent quantiles
        b_quantile = result.filter(pl.col("asset") == "B")["quantile"][0]
        c_quantile = result.filter(pl.col("asset") == "C")["quantile"][0]
        assert abs(b_quantile - c_quantile) <= 1  # Should be close


# =============================================================================
# Tests: filter_outliers
# =============================================================================


class TestFilterOutliers:
    """Tests for filter_outliers function."""

    def test_z_threshold_zero(self, simple_factor_data):
        """Test that z_threshold=0 returns data unchanged."""
        result = filter_outliers(simple_factor_data, z_threshold=0.0)

        assert result.height == simple_factor_data.height

    def test_z_threshold_negative(self, simple_factor_data):
        """Test that negative z_threshold returns data unchanged."""
        result = filter_outliers(simple_factor_data, z_threshold=-1.0)

        assert result.height == simple_factor_data.height

    def test_constant_cross_section(self):
        """Test that constant cross-sections (std=0) are kept."""
        data = pl.DataFrame(
            {
                "date": [date(2024, 1, 1)] * 4,
                "asset": ["A", "B", "C", "D"],
                "factor": [5.0, 5.0, 5.0, 5.0],  # All same
            }
        )

        result = filter_outliers(data, z_threshold=2.0)

        # All should be kept (z_score is null when std=0)
        assert result.height == 4

    def test_all_outliers(self):
        """Test when all values exceed threshold."""
        data = pl.DataFrame(
            {
                "date": [date(2024, 1, 1)] * 3,
                "asset": ["A", "B", "C"],
                "factor": [0.0, 0.0, 100.0],  # C is extreme
            }
        )

        # With z_threshold=0.5, most will be filtered
        result = filter_outliers(data, z_threshold=0.5)

        # Some should be filtered
        assert result.height < 3

    def test_empty_dataframe(self):
        """Test with empty DataFrame."""
        data = pl.DataFrame({"date": pl.Series([], dtype=pl.Date), "asset": [], "factor": []})

        result = filter_outliers(data, z_threshold=3.0)

        assert result.height == 0

    def test_no_outliers(self, simple_factor_data):
        """Test when no values exceed threshold."""
        # Simple data has factors 1-5, std is small relative to high threshold
        result = filter_outliers(simple_factor_data, z_threshold=10.0)

        # All should be kept
        assert result.height == simple_factor_data.height

    def test_removes_extreme_outliers(self):
        """Test that extreme outliers are removed."""
        # Use more normal values so one outlier has high z-score
        # With values 1,2,3,4,5,6,7,8,9,50: mean~10, std~13, z(50)~3
        data = pl.DataFrame(
            {
                "date": [date(2024, 1, 1)] * 10,
                "asset": ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"],
                "factor": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 50.0],
            }
        )

        result = filter_outliers(data, z_threshold=2.0)

        # J (50.0) should be filtered since its z-score > 2.0
        # Mean=9.5, Std~13.5, z(50)=(50-9.5)/13.5~3.0
        assert result.height < 10
        remaining_assets = result["asset"].to_list()
        assert "J" not in remaining_assets


# =============================================================================
# Tests: ensure_polars
# =============================================================================


class TestEnsurePolars:
    """Tests for ensure_polars function."""

    def test_already_polars(self, simple_factor_data):
        """Test that Polars DataFrame returns same object."""
        result = ensure_polars(simple_factor_data)

        # Should be exact same object
        assert result is simple_factor_data

    def test_pandas_conversion(self, simple_factor_data):
        """Test conversion from pandas DataFrame."""
        pandas_df = simple_factor_data.to_pandas()

        result = ensure_polars(pandas_df)

        assert isinstance(result, pl.DataFrame)
        assert result.height == simple_factor_data.height
        assert set(result.columns) == set(simple_factor_data.columns)

    def test_preserves_dtypes(self):
        """Test that data types are preserved during conversion."""
        pandas_df = pd.DataFrame(
            {
                "int_col": [1, 2, 3],
                "float_col": [1.5, 2.5, 3.5],
                "str_col": ["a", "b", "c"],
            }
        )

        result = ensure_polars(pandas_df)

        assert result["int_col"].dtype in (pl.Int64, pl.Int32)
        assert result["float_col"].dtype == pl.Float64
        assert result["str_col"].dtype == pl.Utf8

    def test_index_handling(self):
        """Test that pandas index is handled."""
        pandas_df = pd.DataFrame({"value": [1, 2, 3]}, index=pd.Index([10, 20, 30], name="id"))

        result = ensure_polars(pandas_df)

        # Index should either become a column or be dropped
        assert "value" in result.columns
        assert result.height == 3
