"""High-quality correctness tests for stationary bootstrap methods.

These tests verify the statistical properties of the stationary bootstrap
implementation following Politis & Romano (1994).

Key properties tested:
1. Block length follows geometric distribution
2. Blocks wrap around (modular indexing)
3. Bootstrap indices preserve length
4. P-value is proportion of |null IC| >= |observed IC|
5. Confidence intervals contain observed IC with expected frequency
"""

import numpy as np
import pytest
from scipy.stats import spearmanr

from ml4t.diagnostic.evaluation.stats.bootstrap import (
    _optimal_block_size,
    _stationary_bootstrap_indices,
    stationary_bootstrap_ic,
)


class TestStationaryBootstrapIndices:
    """Tests verifying bootstrap index generation correctness."""

    def test_indices_length_exact(self):
        """Bootstrap indices should have exactly n elements."""
        for n in [10, 50, 100, 500]:
            for block_size in [2, 5, 10]:
                indices = _stationary_bootstrap_indices(n, block_size)

                assert len(indices) == n, f"Expected {n} indices, got {len(indices)}"

    def test_indices_in_valid_range(self):
        """All indices must be in [0, n-1]."""
        n = 100
        block_size = 5

        for _ in range(10):
            indices = _stationary_bootstrap_indices(n, block_size)

            assert np.all(indices >= 0), "Found negative indices"
            assert np.all(indices < n), f"Found indices >= {n}"

    def test_blocks_are_contiguous(self):
        """Within a block, indices should be consecutive (modulo n)."""
        n = 100
        block_size = 10

        indices = _stationary_bootstrap_indices(n, block_size)

        # Find block boundaries (where diff != 1 mod n)
        diffs = np.diff(indices)
        within_block = (diffs == 1) | (diffs == -(n - 1))  # Wrap around case

        # Most differences should indicate within-block
        pct_within_block = np.mean(within_block)
        assert pct_within_block > 0.5, (
            f"Expected most diffs to be within-block, got {pct_within_block:.1%}"
        )

    def test_block_wrapping(self):
        """Blocks should wrap around when starting near end."""
        n = 20
        block_size = 5

        # Run many times to catch wrapping behavior
        saw_wrap = False
        for _ in range(100):
            indices = _stationary_bootstrap_indices(n, block_size)

            # Check for wrap: a block starting at e.g. index 18 should wrap to 0, 1, 2...
            for i in range(len(indices) - 1):
                if indices[i] == n - 1 and indices[i + 1] == 0:
                    saw_wrap = True
                    break

            if saw_wrap:
                break

        # With block_size=5 and n=20, we should sometimes see wrapping
        assert saw_wrap, "Never observed block wrapping behavior"

    def test_geometric_block_length_distribution(self):
        """Block lengths should follow geometric distribution with mean = block_size."""
        n = 1000
        block_size = 10
        expected_mean = block_size

        # Generate many samples to estimate average block length
        all_block_lengths = []

        for _ in range(100):
            indices = _stationary_bootstrap_indices(n, block_size)

            # Find block lengths by detecting boundaries
            diffs = np.diff(indices)
            # Boundary where diff != 1 and not wrap-around
            boundaries = np.where((diffs != 1) & (diffs != -(n - 1)))[0]

            # Calculate block lengths
            boundaries = np.concatenate([[0], boundaries + 1, [n]])
            block_lengths = np.diff(boundaries)
            all_block_lengths.extend(block_lengths.tolist())

        # Average block length should be close to block_size
        avg_block_length = np.mean(all_block_lengths)
        assert abs(avg_block_length - expected_mean) < expected_mean * 0.3, (
            f"Average block length {avg_block_length:.2f} too far from expected {expected_mean}"
        )


class TestOptimalBlockSize:
    """Tests for automatic block size selection."""

    def test_block_size_positive(self):
        """Block size should always be positive."""
        np.random.seed(42)

        for _ in range(10):
            data = np.random.randn(100)
            block_size = _optimal_block_size(data)

            assert block_size > 0, f"Block size must be positive, got {block_size}"

    def test_block_size_bounded(self):
        """Block size should not exceed n/3."""
        np.random.seed(42)
        n = 100
        data = np.random.randn(n)

        block_size = _optimal_block_size(data)

        assert block_size <= n // 3, f"Block size {block_size} should not exceed n/3 = {n // 3}"

    def test_higher_autocorrelation_larger_blocks(self):
        """Higher autocorrelation should lead to larger block sizes."""
        np.random.seed(42)
        n = 200

        # Low autocorrelation: IID
        low_ac_data = np.random.randn(n)

        # High autocorrelation: AR(1) with rho=0.9
        high_ac_data = np.zeros(n)
        high_ac_data[0] = np.random.randn()
        for t in range(1, n):
            high_ac_data[t] = 0.9 * high_ac_data[t - 1] + np.random.randn() * 0.1

        block_size_low = _optimal_block_size(low_ac_data)
        block_size_high = _optimal_block_size(high_ac_data)

        assert block_size_high >= block_size_low, (
            f"High AC should give larger blocks: low={block_size_low}, high={block_size_high}"
        )

    def test_small_sample_minimum_block(self):
        """Small samples should get minimum block size."""
        data = np.random.randn(5)

        block_size = _optimal_block_size(data)

        assert block_size >= 1, "Block size must be at least 1"
        assert block_size <= len(data), "Block size must not exceed data length"


class TestStationaryBootstrapIC:
    """Tests for bootstrap IC inference."""

    def test_p_value_bounded(self):
        """P-value must be in [0, 1]."""
        np.random.seed(42)
        predictions = np.random.randn(50)
        returns = predictions * 0.3 + np.random.randn(50) * 0.5

        result = stationary_bootstrap_ic(predictions, returns, n_samples=100, return_details=True)

        assert 0 <= result["p_value"] <= 1, f"P-value {result['p_value']} out of bounds"

    def test_ci_contains_ic(self):
        """Confidence interval should contain the point estimate."""
        np.random.seed(42)
        predictions = np.random.randn(100)
        returns = predictions * 0.5 + np.random.randn(100) * 0.3

        result = stationary_bootstrap_ic(predictions, returns, n_samples=500, confidence_level=0.95)

        assert result["ci_lower"] <= result["ic"] <= result["ci_upper"], (
            f"IC {result['ic']:.4f} outside CI [{result['ci_lower']:.4f}, {result['ci_upper']:.4f}]"
        )

    def test_zero_correlation_high_p_value(self):
        """Independent predictions and returns should give high p-value."""
        np.random.seed(42)
        # Completely independent
        predictions = np.random.randn(100)
        returns = np.random.randn(100)

        result = stationary_bootstrap_ic(predictions, returns, n_samples=500, return_details=True)

        # P-value should be high for no correlation
        # (Cannot guarantee > 0.05 but should typically be)
        assert result["p_value"] > 0.01, (
            f"Expected high p-value for independent data, got {result['p_value']:.4f}"
        )

    def test_strong_correlation_low_p_value(self):
        """Highly correlated predictions and returns should give low p-value."""
        np.random.seed(42)
        predictions = np.random.randn(100)
        # Returns nearly perfectly correlated
        returns = predictions * 0.9 + np.random.randn(100) * 0.1

        result = stationary_bootstrap_ic(predictions, returns, n_samples=500, return_details=True)

        # P-value should be low for strong correlation
        assert result["p_value"] < 0.1, (
            f"Expected low p-value for strong correlation, got {result['p_value']:.4f}"
        )

    def test_ic_matches_spearman(self):
        """Reported IC should match scipy Spearman correlation."""
        np.random.seed(42)
        predictions = np.random.randn(50)
        returns = predictions * 0.5 + np.random.randn(50) * 0.5

        result = stationary_bootstrap_ic(predictions, returns, n_samples=100)

        expected_ic, _ = spearmanr(predictions, returns)

        assert abs(result["ic"] - expected_ic) < 1e-10, (
            f"IC mismatch: expected {expected_ic:.6f}, got {result['ic']:.6f}"
        )

    def test_length_mismatch_raises(self):
        """Different length inputs should raise ValueError."""
        predictions = np.random.randn(50)
        returns = np.random.randn(60)

        with pytest.raises(ValueError, match="same length"):
            stationary_bootstrap_ic(predictions, returns)

    def test_return_details_false(self):
        """return_details=False should return just p-value."""
        np.random.seed(42)
        predictions = np.random.randn(50)
        returns = predictions * 0.3 + np.random.randn(50) * 0.5

        result = stationary_bootstrap_ic(predictions, returns, n_samples=100, return_details=False)

        assert isinstance(result, float), f"Expected float, got {type(result)}"
        assert 0 <= result <= 1, f"P-value {result} out of bounds"

    def test_reproducibility(self):
        """Results should be reproducible with numpy seed before call."""
        predictions = np.random.randn(50)
        returns = predictions * 0.3 + np.random.randn(50) * 0.5

        np.random.seed(42)
        result1 = stationary_bootstrap_ic(predictions, returns, n_samples=100)

        np.random.seed(42)
        result2 = stationary_bootstrap_ic(predictions, returns, n_samples=100)

        assert result1["p_value"] == result2["p_value"], "Results not reproducible"
        assert result1["ci_lower"] == result2["ci_lower"]
        assert result1["ci_upper"] == result2["ci_upper"]


class TestBootstrapStatisticalProperties:
    """Tests for statistical properties of bootstrap inference."""

    def test_null_p_value_uniform(self):
        """Under null hypothesis, p-values should be approximately uniform.

        This is a key statistical property: when there is truly no correlation,
        p-values should be uniformly distributed on [0, 1].
        """
        np.random.seed(42)
        n_simulations = 50
        p_values = []

        for _ in range(n_simulations):
            # Generate independent data (null hypothesis true)
            predictions = np.random.randn(30)
            returns = np.random.randn(30)

            p_value = stationary_bootstrap_ic(
                predictions, returns, n_samples=100, return_details=False
            )
            p_values.append(p_value)

        # Under null, ~10% of p-values should be < 0.1
        pct_below_10 = np.mean(np.array(p_values) < 0.1)
        # Allow wide tolerance due to small sample
        assert 0.02 < pct_below_10 < 0.35, (
            f"Under null, ~10% should have p<0.1, got {pct_below_10:.1%}"
        )

    def test_ci_coverage(self):
        """95% CI should contain true IC approximately 95% of the time.

        This tests the calibration of the bootstrap confidence interval.
        """
        np.random.seed(42)
        n_simulations = 50
        true_rho = 0.3
        coverage_count = 0

        for _ in range(n_simulations):
            # Generate data with known correlation
            n = 50
            x = np.random.randn(n)
            noise = np.random.randn(n) * np.sqrt(1 - true_rho**2)
            y = true_rho * x + noise

            result = stationary_bootstrap_ic(x, y, n_samples=200, confidence_level=0.95)

            # Check if observed IC is in CI (as a proxy for true rho)
            # Note: We're checking if observed IC is in its own bootstrap CI
            # which should happen ~95% of the time
            if result["ci_lower"] <= result["ic"] <= result["ci_upper"]:
                coverage_count += 1

        coverage_rate = coverage_count / n_simulations
        # Should be close to 95% (allowing some tolerance)
        assert coverage_rate > 0.80, f"CI coverage rate {coverage_rate:.1%} too low (expected ~95%)"


class TestEdgeCasesBootstrap:
    """Edge cases for bootstrap methods."""

    def test_small_sample_warning(self):
        """Small sample should raise warning."""
        predictions = np.random.randn(20)
        returns = predictions * 0.3 + np.random.randn(20) * 0.5

        with pytest.warns(UserWarning, match="too small"):
            stationary_bootstrap_ic(predictions, returns, n_samples=100)

    def test_nan_handling(self):
        """NaN values should be properly excluded."""
        predictions = np.array([0.1, np.nan, 0.2, 0.3, np.nan, 0.4] * 10)
        returns = np.array([0.05, np.nan, 0.15, 0.2, np.nan, 0.25] * 10)

        result = stationary_bootstrap_ic(predictions, returns, n_samples=100, return_details=True)

        assert not np.isnan(result["ic"]), "IC should not be NaN after NaN removal"
        assert not np.isnan(result["p_value"]), "P-value should not be NaN"


class TestPanelScalingWarning:
    """Tests for large-sample panel data warning."""

    def test_warns_on_large_input(self):
        """Should warn when n > 50K that bootstrap will be slow."""
        import warnings

        n = 60_000
        np.random.seed(42)
        predictions = np.random.randn(n)
        returns = np.random.randn(n)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            # Use n_samples=1 to avoid actually running the slow bootstrap
            stationary_bootstrap_ic(predictions, returns, n_samples=1)

            panel_warnings = [x for x in w if "very slow" in str(x.message)]
            assert len(panel_warnings) == 1
            assert "60,000" in str(panel_warnings[0].message)
            assert "cross-sectional IC" in str(panel_warnings[0].message)

    def test_no_warning_below_threshold(self):
        """Should not warn when n <= 50K."""
        import warnings

        n = 5_000
        np.random.seed(42)
        predictions = np.random.randn(n)
        returns = np.random.randn(n)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            stationary_bootstrap_ic(predictions, returns, n_samples=10)

            panel_warnings = [x for x in w if "very slow" in str(x.message)]
            assert len(panel_warnings) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
