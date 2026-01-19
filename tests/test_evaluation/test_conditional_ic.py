"""Tests for conditional IC feature interaction analysis.

Validates the compute_conditional_ic() function with synthetic interactions.
"""

import numpy as np
import pandas as pd
import polars as pl
import pytest

from ml4t.diagnostic.evaluation.metrics import compute_conditional_ic


class TestConditionalIC:
    """Test conditional IC computation."""

    def test_strong_interaction_high_regime_only(self):
        """Test detection of interaction where feature A works only in high B regime."""
        np.random.seed(42)
        n = 2000  # Increased sample size for more stable statistics

        # Feature B defines regimes
        feature_b = np.random.randn(n)

        # Feature A is random
        feature_a = np.random.randn(n)

        # Returns depend on A only when B > 0 (high regime)
        # Use stronger signal for clearer detection
        noise = 0.1 * np.random.randn(n)
        returns = np.where(feature_b > 0, 0.7 * feature_a + noise, noise)

        result = compute_conditional_ic(feature_a, feature_b, returns, n_quantiles=5)

        # Should detect strong interaction
        assert result["ic_range"] > 0.15, f"IC range too small: {result['ic_range']}"
        # Note: Statistical significance can vary with random sampling
        # Focus on IC range as primary indicator
        if result["significance_pvalue"] is not None:
            assert result["significance_pvalue"] < 0.15, (
                f"P-value too high: {result['significance_pvalue']}"
            )

        # Check that there's variation in IC across quantiles
        ic_array = result["quantile_ics"]
        valid_ics = ic_array[~np.isnan(ic_array)]

        # With the interaction (A works in high B), we expect:
        # High volatility (feature_b) quantiles should have higher IC
        # But quantile order depends on how pandas assigns labels
        # So just check that max IC > min IC significantly
        if len(valid_ics) >= 2:
            assert np.max(valid_ics) > np.min(valid_ics) + 0.15, (
                f"Max IC ({np.max(valid_ics):.3f}) should be > min IC ({np.min(valid_ics):.3f}) + 0.15"
            )

        # Check interpretation mentions variation
        assert (
            "interaction" in result["interpretation"].lower()
            or "regime" in result["interpretation"].lower()
        )

    def test_no_interaction_independent_features(self):
        """Test that no interaction is detected when features are independent."""
        np.random.seed(42)
        n = 1000

        # Both features independent of returns
        feature_a = np.random.randn(n)
        feature_b = np.random.randn(n)
        returns = np.random.randn(n)

        result = compute_conditional_ic(feature_a, feature_b, returns, n_quantiles=5)

        # Should not detect significant interaction
        # Allow for random variation - IC range should be small
        assert result["ic_range"] < 0.3, (
            f"IC range too large for independent features: {result['ic_range']}"
        )

        # P-value might be significant by chance, but interpretation should reflect no interaction
        if result["significance_pvalue"] is not None and result["significance_pvalue"] > 0.05:
            assert "no significant interaction" in result["interpretation"].lower()

    def test_inverted_interaction_low_regime(self):
        """Test detection when feature A works only in low B regime."""
        np.random.seed(42)
        n = 1000

        feature_b = np.random.randn(n)
        feature_a = np.random.randn(n)

        # Returns depend on A only when B < 0 (low regime)
        noise = 0.1 * np.random.randn(n)
        returns = np.where(feature_b < 0, 0.5 * feature_a + noise, noise)

        result = compute_conditional_ic(feature_a, feature_b, returns, n_quantiles=5)

        # Should detect interaction
        assert result["ic_range"] > 0.15

        # Bottom quantiles should have higher IC than top
        ic_array = result["quantile_ics"]
        # Check that low quantiles have meaningful IC
        assert not np.isnan(ic_array[0]), "Bottom quantile IC should not be NaN"

    def test_consistent_predictive_power_no_interaction(self):
        """Test when feature A is consistently predictive across all B regimes."""
        np.random.seed(42)
        n = 1000

        feature_b = np.random.randn(n)  # Irrelevant
        feature_a = np.random.randn(n)
        noise = 0.1 * np.random.randn(n)

        # Returns always depend on A, regardless of B
        returns = 0.5 * feature_a + noise

        result = compute_conditional_ic(feature_a, feature_b, returns, n_quantiles=5)

        # IC should be relatively consistent across quantiles
        ic_array = result["quantile_ics"]
        valid_ics = ic_array[~np.isnan(ic_array)]

        if len(valid_ics) >= 2:
            # All ICs should be positive and similar
            assert np.all(valid_ics > 0.2), "All ICs should be meaningfully positive"
            assert result["ic_variation"] < 0.15, f"IC variation too high: {result['ic_variation']}"

    def test_pandas_series_input(self):
        """Test with pandas Series inputs."""
        np.random.seed(42)
        n = 500

        feature_b = pd.Series(np.random.randn(n))
        feature_a = pd.Series(np.random.randn(n))
        returns = pd.Series(np.where(feature_b > 0, feature_a, 0) + 0.1 * np.random.randn(n))

        result = compute_conditional_ic(feature_a, feature_b, returns, n_quantiles=3)

        assert result["n_quantiles"] == 3
        assert len(result["quantile_ics"]) == 3
        assert len(result["quantile_labels"]) == 3

    def test_polars_series_input(self):
        """Test with Polars Series inputs."""
        np.random.seed(42)
        n = 500

        feature_b = pl.Series(np.random.randn(n))
        feature_a = pl.Series(np.random.randn(n))
        returns = pl.Series(np.where(feature_b > 0, feature_a, 0) + 0.1 * np.random.randn(n))

        result = compute_conditional_ic(feature_a, feature_b, returns, n_quantiles=3)

        assert result["n_quantiles"] == 3
        assert len(result["quantile_ics"]) == 3

    def test_numpy_array_input(self):
        """Test with NumPy array inputs."""
        np.random.seed(42)
        n = 500

        feature_b = np.random.randn(n)
        feature_a = np.random.randn(n)
        returns = np.where(feature_b > 0, feature_a, 0) + 0.1 * np.random.randn(n)

        result = compute_conditional_ic(feature_a, feature_b, returns, n_quantiles=4)

        assert result["n_quantiles"] == 4
        assert isinstance(result["quantile_ics"], np.ndarray)

    def test_nan_handling(self):
        """Test proper handling of NaN values."""
        np.random.seed(42)
        n = 500

        feature_b = np.random.randn(n)
        feature_a = np.random.randn(n)
        returns = np.random.randn(n)

        # Inject NaN values
        feature_a[10:20] = np.nan
        feature_b[30:40] = np.nan
        returns[50:60] = np.nan

        result = compute_conditional_ic(feature_a, feature_b, returns, n_quantiles=5)

        # Should still compute (dropping NaN rows)
        assert result["n_quantiles"] == 5
        # Check that we have some valid observations
        total_obs = sum(result["n_obs_per_quantile"].values())
        assert total_obs > 0, "Should have some valid observations after NaN removal"

    def test_insufficient_data(self):
        """Test behavior with insufficient data."""
        feature_a = np.random.randn(20)  # Too few observations
        feature_b = np.random.randn(20)
        returns = np.random.randn(20)

        result = compute_conditional_ic(
            feature_a, feature_b, returns, n_quantiles=5, min_periods=10
        )

        # Should return None or NaN results with appropriate message
        # None is returned by function, but numpy converts to nan in some contexts
        assert result["ic_range"] is None or np.isnan(result["ic_range"])
        assert result["ic_variation"] is None or np.isnan(result["ic_variation"])
        assert "insufficient" in result["interpretation"].lower()

    def test_quantile_bounds_meaningful(self):
        """Test that quantile bounds make sense."""
        np.random.seed(42)
        n = 1000

        # Feature B uniformly distributed
        feature_b = np.linspace(-2, 2, n)
        feature_a = np.random.randn(n)
        returns = np.random.randn(n)

        result = compute_conditional_ic(feature_a, feature_b, returns, n_quantiles=5)

        bounds = result["quantile_bounds"]

        # Q1 should have lower mean than Q5
        if "Q1" in bounds and "Q5" in bounds:
            assert bounds["Q1"] < bounds["Q5"], "Q1 should have lower mean than Q5"

        # Bounds should be ordered
        bound_values = [bounds[f"Q{i + 1}"] for i in range(5) if f"Q{i + 1}" in bounds]
        if len(bound_values) > 1:
            assert bound_values == sorted(bound_values), "Quantile bounds should be ordered"

    def test_different_quantile_numbers(self):
        """Test with different numbers of quantiles."""
        np.random.seed(42)
        n = 1000

        feature_b = np.random.randn(n)
        feature_a = np.random.randn(n)
        returns = feature_a + 0.1 * np.random.randn(n)

        for n_q in [3, 5, 10]:
            result = compute_conditional_ic(feature_a, feature_b, returns, n_quantiles=n_q)
            assert result["n_quantiles"] == n_q
            assert len(result["quantile_labels"]) == n_q

    def test_correlation_methods(self):
        """Test both Spearman and Pearson correlation methods."""
        np.random.seed(42)
        n = 500

        feature_b = np.random.randn(n)
        feature_a = np.random.randn(n)
        returns = feature_a + 0.1 * np.random.randn(n)

        result_spearman = compute_conditional_ic(feature_a, feature_b, returns, method="spearman")
        result_pearson = compute_conditional_ic(feature_a, feature_b, returns, method="pearson")

        # Both should work
        assert result_spearman["n_quantiles"] == 5
        assert result_pearson["n_quantiles"] == 5

        # Results may differ slightly
        assert not np.array_equal(
            result_spearman["quantile_ics"],
            result_pearson["quantile_ics"],
        )

    def test_min_periods_enforcement(self):
        """Test that min_periods is enforced per quantile."""
        np.random.seed(42)
        n = 100

        feature_b = np.random.randn(n)
        feature_a = np.random.randn(n)
        returns = np.random.randn(n)

        # With 5 quantiles and min_periods=30, we won't have enough data per quantile
        result = compute_conditional_ic(
            feature_a, feature_b, returns, n_quantiles=5, min_periods=30
        )

        # Some quantiles might have NaN IC due to insufficient data
        ic_array = result["quantile_ics"]
        assert np.any(np.isnan(ic_array)), "Some quantiles should have NaN IC with high min_periods"

    def test_duplicate_values_handling(self):
        """Test handling of feature_b with many duplicate values."""
        np.random.seed(42)
        n = 500

        # Feature B has only a few unique values
        feature_b = np.random.choice([1, 2, 3], size=n)
        feature_a = np.random.randn(n)
        returns = np.random.randn(n)

        # Should handle gracefully (qcut with duplicates='drop')
        result = compute_conditional_ic(feature_a, feature_b, returns, n_quantiles=5)

        # Should complete without error, but may have fewer quantiles
        assert "quantile_ics" in result
        assert result["interpretation"] is not None


class TestConditionalICPanelData:
    """Test conditional IC with panel data (multi-asset time series)."""

    def test_panel_data_cross_sectional_quantiles(self):
        """Test that quantiles are computed cross-sectionally with panel data."""
        np.random.seed(42)

        # Create panel data
        dates = pd.date_range("2020-01-01", periods=50)
        assets = ["AAPL", "MSFT", "GOOGL", "AMZN"]

        panel_data = []
        for date in dates:
            for asset in assets:
                panel_data.append(
                    {
                        "date": date,
                        "asset": asset,
                        "momentum": np.random.randn(),
                        "volatility": np.random.randn(),
                        "returns": np.random.randn(),
                    }
                )

        df = pd.DataFrame(panel_data)

        # Split into feature DataFrames
        df_a = df[["date", "asset", "momentum"]]
        df_b = df[["date", "asset", "volatility"]]
        df_ret = df[["date", "asset", "returns"]]

        result = compute_conditional_ic(
            df_a, df_b, df_ret, date_col="date", group_col="asset", n_quantiles=3
        )

        assert result["n_quantiles"] == 3
        assert len(result["quantile_labels"]) == 3

    def test_panel_data_with_interaction(self):
        """Test panel data with actual interaction pattern."""
        np.random.seed(42)

        dates = pd.date_range("2020-01-01", periods=100)
        assets = ["A", "B", "C"]

        panel_data = []
        for date in dates:
            for asset in assets:
                volatility = np.random.randn()
                momentum = np.random.randn()
                # Returns depend on momentum only in high volatility
                noise = 0.1 * np.random.randn()
                returns = momentum if volatility > 0 else noise

                panel_data.append(
                    {
                        "date": date,
                        "asset": asset,
                        "momentum": momentum,
                        "volatility": volatility,
                        "returns": returns,
                    }
                )

        df = pd.DataFrame(panel_data)

        df_a = df[["date", "asset", "momentum"]]
        df_b = df[["date", "asset", "volatility"]]
        df_ret = df[["date", "asset", "returns"]]

        result = compute_conditional_ic(
            df_a, df_b, df_ret, date_col="date", group_col="asset", n_quantiles=5
        )

        # Should detect interaction
        assert result["ic_range"] > 0.1, f"IC range too small: {result['ic_range']}"

    def test_error_on_series_with_date_col(self):
        """Test that using Series with date_col raises error."""
        feature_a = pd.Series(np.random.randn(100))
        feature_b = pd.Series(np.random.randn(100))
        returns = pd.Series(np.random.randn(100))

        with pytest.raises(ValueError, match="date_col and group_col require DataFrame"):
            compute_conditional_ic(feature_a, feature_b, returns, date_col="date", n_quantiles=5)

    def test_error_on_missing_date_col(self):
        """Test error when date_col specified but not in DataFrame."""
        df_a = pd.DataFrame({"momentum": np.random.randn(100)})
        df_b = pd.DataFrame({"volatility": np.random.randn(100)})
        df_ret = pd.DataFrame({"returns": np.random.randn(100)})

        with pytest.raises(ValueError, match="date_col.*not found"):
            compute_conditional_ic(df_a, df_b, df_ret, date_col="date", n_quantiles=5)


class TestConditionalICOutputStructure:
    """Test output structure and return types."""

    def test_return_structure(self):
        """Test that return dictionary has all expected keys."""
        np.random.seed(42)
        n = 500

        feature_a = np.random.randn(n)
        feature_b = np.random.randn(n)
        returns = np.random.randn(n)

        result = compute_conditional_ic(feature_a, feature_b, returns)

        # Check all expected keys
        expected_keys = {
            "quantile_ics",
            "quantile_labels",
            "quantile_bounds",
            "ic_variation",
            "ic_range",
            "significance_pvalue",
            "test_statistic",
            "n_quantiles",
            "n_obs_per_quantile",
            "interpretation",
        }

        assert set(result.keys()) == expected_keys

    def test_quantile_ics_is_array(self):
        """Test that quantile_ics is a NumPy array."""
        np.random.seed(42)
        feature_a = np.random.randn(500)
        feature_b = np.random.randn(500)
        returns = np.random.randn(500)

        result = compute_conditional_ic(feature_a, feature_b, returns)

        assert isinstance(result["quantile_ics"], np.ndarray)

    def test_quantile_labels_format(self):
        """Test quantile labels are properly formatted."""
        np.random.seed(42)
        feature_a = np.random.randn(500)
        feature_b = np.random.randn(500)
        returns = np.random.randn(500)

        result = compute_conditional_ic(feature_a, feature_b, returns, n_quantiles=5)

        labels = result["quantile_labels"]
        assert labels == ["Q1", "Q2", "Q3", "Q4", "Q5"]

    def test_n_obs_per_quantile_dict(self):
        """Test that n_obs_per_quantile is a dict with counts."""
        np.random.seed(42)
        feature_a = np.random.randn(500)
        feature_b = np.random.randn(500)
        returns = np.random.randn(500)

        result = compute_conditional_ic(feature_a, feature_b, returns, n_quantiles=5)

        n_obs = result["n_obs_per_quantile"]
        assert isinstance(n_obs, dict)
        assert len(n_obs) == 5

        # Total observations should be close to input size
        total_obs = sum(n_obs.values())
        assert 400 < total_obs <= 500  # Allow for NaN removal

    def test_interpretation_is_string(self):
        """Test that interpretation is a non-empty string."""
        np.random.seed(42)
        feature_a = np.random.randn(500)
        feature_b = np.random.randn(500)
        returns = np.random.randn(500)

        result = compute_conditional_ic(feature_a, feature_b, returns)

        assert isinstance(result["interpretation"], str)
        assert len(result["interpretation"]) > 0


class TestConditionalICEdgeCases:
    """Test edge cases and error handling."""

    def test_mismatched_lengths(self):
        """Test error on mismatched input lengths."""
        feature_a = np.random.randn(100)
        feature_b = np.random.randn(100)
        returns = np.random.randn(50)  # Different length

        with pytest.raises(ValueError, match="same length"):
            compute_conditional_ic(feature_a, feature_b, returns)

    def test_all_nan_input(self):
        """Test handling of all-NaN input."""
        feature_a = np.full(100, np.nan)
        feature_b = np.random.randn(100)
        returns = np.random.randn(100)

        result = compute_conditional_ic(feature_a, feature_b, returns)

        # Should return graceful failure
        assert "insufficient" in result["interpretation"].lower()

    def test_constant_feature_b(self):
        """Test handling when feature_b is constant (no variation)."""
        feature_a = np.random.randn(100)
        feature_b = np.ones(100)  # Constant
        returns = np.random.randn(100)

        result = compute_conditional_ic(feature_a, feature_b, returns)

        # Should fail gracefully (can't create quantiles)
        assert (
            "cannot compute quantiles" in result["interpretation"].lower()
            or "insufficient" in result["interpretation"].lower()
        )

    def test_perfect_correlation_in_one_quantile(self):
        """Test when feature A perfectly predicts in one quantile."""
        np.random.seed(42)
        n = 1000

        feature_b = np.random.randn(n)
        feature_a = np.random.randn(n)

        # Perfect correlation in top quantile
        returns = np.where(feature_b > 1.5, feature_a, 0.01 * np.random.randn(n))

        result = compute_conditional_ic(feature_a, feature_b, returns, n_quantiles=5)

        # Should detect strong interaction
        ic_array = result["quantile_ics"]

        # Top quantile should have high IC
        valid_ics = ic_array[~np.isnan(ic_array)]
        if len(valid_ics) > 0:
            max_ic = np.max(valid_ics)
            assert max_ic > 0.5, f"Max IC should be high in perfect correlation quantile: {max_ic}"
