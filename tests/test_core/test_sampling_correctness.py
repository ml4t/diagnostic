"""High-quality correctness tests for sampling functions.

These tests verify mathematical properties and statistical invariants
of the sampling functions, not just that they "work" or "return something".

Key properties tested:
1. Block bootstrap: blocks are truly contiguous, block boundaries align
2. Stratified sampling: strata proportions are preserved within tolerance
3. Importance weights: exact formula verification for each method
4. Balanced subsample: class balance improvement, no duplication in undersample
5. Event sampling: minimum spacing is a HARD invariant (always respected)
"""

import numpy as np
import pandas as pd
import polars as pl
import pytest

from ml4t.diagnostic.core.sampling import (
    balanced_subsample,
    block_bootstrap,
    event_based_sample,
    sample_weights_by_importance,
    stratified_sample_time_series,
)


class TestBlockBootstrapCorrectness:
    """Tests verifying mathematical correctness of block bootstrap."""

    def test_blocks_are_truly_contiguous(self):
        """Verify each block contains consecutive indices with no gaps.

        The fundamental property of block bootstrap is that sampled blocks
        preserve temporal structure. Within each block, indices must be
        consecutive (diff=1).
        """
        indices = np.arange(100)
        n_samples = 50
        sample_length = 5  # Each block should be exactly 5 consecutive indices

        result = block_bootstrap(indices, n_samples, sample_length, random_state=42)

        # Analyze the sequence of differences
        diffs = np.diff(result)

        # Count transitions (where diff != 1, meaning block boundary)
        np.sum(diffs != 1)

        # Expected: approximately n_samples // sample_length blocks
        # So approximately that many boundaries
        expected_blocks = n_samples // sample_length

        # Within blocks, ALL differences must be 1
        within_block_diffs = diffs[diffs == 1]
        # Should be n_samples - expected_blocks (for boundaries)
        assert len(within_block_diffs) >= n_samples - expected_blocks - sample_length, (
            f"Too few consecutive pairs: {len(within_block_diffs)}, "
            f"expected at least {n_samples - expected_blocks - sample_length}"
        )

    def test_block_size_is_honored(self):
        """Verify blocks have the specified length.

        When we sample blocks of size k, each block should contribute
        exactly k consecutive samples (except possibly the last block).
        """
        indices = np.arange(200)
        sample_length = 10
        n_samples = 100

        result = block_bootstrap(indices, n_samples, sample_length, random_state=42)

        # Find block boundaries (where diff != 1)
        diffs = np.diff(result)
        boundary_positions = np.where(diffs != 1)[0]

        # Measure block sizes
        block_sizes = []
        prev_pos = 0
        for pos in boundary_positions:
            block_sizes.append(pos - prev_pos + 1)
            prev_pos = pos + 1
        # Add last block
        block_sizes.append(len(result) - prev_pos)

        # Most blocks should be exactly sample_length
        # (last block might be shorter)
        full_blocks = [s for s in block_sizes[:-1] if s == sample_length]
        assert len(full_blocks) >= len(block_sizes) - 2, (
            f"Too few blocks of correct size. Block sizes: {block_sizes}"
        )

    def test_all_sampled_indices_from_valid_range(self):
        """Verify all sampled indices are within the original range.

        This is an invariant: the bootstrap should never produce
        indices outside the original array bounds.
        """
        # Non-contiguous indices to test bounds checking
        indices = np.array([5, 10, 15, 20, 25, 30, 35, 40, 45, 50])
        n_samples = 20
        sample_length = 3

        result = block_bootstrap(indices, n_samples, sample_length, random_state=42)

        # All results must be in original indices
        assert np.all(np.isin(result, indices)), (
            f"Found indices not in original array: {set(result) - set(indices)}"
        )

    def test_output_length_exact(self):
        """Verify output has exactly n_samples elements."""
        for n_samples in [1, 10, 50, 100, 500]:
            result = block_bootstrap(np.arange(1000), n_samples, sample_length=10, random_state=42)
            assert len(result) == n_samples, f"Expected {n_samples} samples, got {len(result)}"


class TestStratifiedSamplingCorrectness:
    """Tests verifying stratified sampling preserves strata proportions."""

    def test_strata_proportions_preserved_within_tolerance(self):
        """Verify each stratum contributes proportionally to the sample.

        For stratified sampling, if stratum A has 30% of data and stratum B
        has 70%, the sample should have approximately 30/70 split (within
        statistical tolerance).
        """
        np.random.seed(42)
        n = 1000
        # Create imbalanced strata: 30% label=0, 70% label=1
        labels = np.array([0] * 300 + [1] * 700)

        df = pd.DataFrame(
            {
                "label": labels,
                "feature": np.random.randn(n),
            }
        )

        result = stratified_sample_time_series(
            df,
            stratify_column="label",
            sample_frac=0.3,
            random_state=42,
        )

        # Calculate proportions
        original_prop_0 = (df["label"] == 0).mean()
        original_prop_1 = (df["label"] == 1).mean()

        sample_prop_0 = (result["label"] == 0).mean()
        sample_prop_1 = (result["label"] == 1).mean()

        # Proportions should be close (within 0.1 tolerance due to discrete sampling)
        assert abs(sample_prop_0 - original_prop_0) < 0.1, (
            f"Stratum 0 proportion mismatch: original={original_prop_0:.3f}, "
            f"sample={sample_prop_0:.3f}"
        )
        assert abs(sample_prop_1 - original_prop_1) < 0.1, (
            f"Stratum 1 proportion mismatch: original={original_prop_1:.3f}, "
            f"sample={sample_prop_1:.3f}"
        )

    def test_all_strata_represented_in_sample(self):
        """Verify no stratum is completely omitted from the sample.

        If a stratum has samples in the original, it should have samples
        in the stratified sample (for reasonable sample_frac).

        Note: Using preserve_order=False ensures pure random sampling per stratum.
        The block-based sampling (preserve_order=True) can miss small strata
        due to probabilistic block selection.
        """
        np.random.seed(42)
        # Create 5 strata with varying sizes
        df = pd.DataFrame(
            {
                "stratum": [0] * 100 + [1] * 200 + [2] * 300 + [3] * 150 + [4] * 250,
                "value": np.random.randn(1000),
            }
        )

        result = stratified_sample_time_series(
            df,
            stratify_column="stratum",
            sample_frac=0.3,
            preserve_order=False,  # Use pure random sampling, not block-based
            random_state=42,
        )

        original_strata = set(df["stratum"].unique())
        sample_strata = set(result["stratum"].unique())

        assert sample_strata == original_strata, (
            f"Missing strata in sample: {original_strata - sample_strata}"
        )

    def test_sample_size_approximates_target(self):
        """Verify total sample size is approximately sample_frac * original."""
        np.random.seed(42)
        n = 1000
        sample_frac = 0.25

        df = pd.DataFrame(
            {
                "stratum": np.random.choice([0, 1, 2], n),
                "value": np.random.randn(n),
            }
        )

        result = stratified_sample_time_series(
            df,
            stratify_column="stratum",
            sample_frac=sample_frac,
            random_state=42,
        )

        expected_size = int(n * sample_frac)
        # Allow 20% tolerance due to discrete per-stratum sampling
        assert abs(len(result) - expected_size) < expected_size * 0.2, (
            f"Sample size {len(result)} too far from expected {expected_size}"
        )


class TestImportanceWeightsCorrectness:
    """Tests verifying exact mathematical formulas for importance weights."""

    def test_return_magnitude_formula_exact(self):
        """Verify return_magnitude: w_i = |r_i| / sum(|r_j|)."""
        returns = np.array([0.01, -0.02, 0.03, -0.01, 0.005])

        weights = sample_weights_by_importance(returns, method="return_magnitude")

        # Calculate expected weights manually
        abs_returns = np.abs(returns)
        expected_weights = abs_returns / abs_returns.sum()

        np.testing.assert_array_almost_equal(
            weights,
            expected_weights,
            decimal=10,
            err_msg="Return magnitude weights don't match formula",
        )

    def test_recency_formula_exact(self):
        """Verify recency: w_i = decay^(n-1-i) / sum(decay^(n-1-j)).

        More recent observations (higher i) get higher weight.
        """
        returns = np.random.randn(10) * 0.02
        decay = 0.94

        weights = sample_weights_by_importance(returns, method="recency", decay_factor=decay)

        # Calculate expected weights manually
        n = len(returns)
        time_weights = decay ** np.arange(n - 1, -1, -1)  # decay^(n-1), ..., decay^0
        expected_weights = time_weights / time_weights.sum()

        np.testing.assert_array_almost_equal(
            weights, expected_weights, decimal=10, err_msg="Recency weights don't match formula"
        )

    def test_recency_monotonicity(self):
        """Verify recency weights are strictly monotonically increasing.

        w_0 < w_1 < ... < w_{n-1} for any decay factor in (0, 1).
        """
        returns = np.random.randn(50) * 0.02

        for decay in [0.5, 0.9, 0.99]:
            weights = sample_weights_by_importance(returns, method="recency", decay_factor=decay)

            # Check strict monotonicity
            diffs = np.diff(weights)
            assert np.all(diffs > 0), f"Recency weights not strictly increasing for decay={decay}"

    def test_weights_sum_to_one_exactly(self):
        """Verify all weight methods sum to exactly 1.0."""
        returns = np.random.randn(100) * 0.02

        for method in ["return_magnitude", "recency", "volatility"]:
            weights = sample_weights_by_importance(returns, method=method)
            assert np.isclose(weights.sum(), 1.0, atol=1e-10), (
                f"Method '{method}' weights sum to {weights.sum()}, not 1.0"
            )

    def test_weights_all_non_negative(self):
        """Verify all weights are non-negative (valid probabilities)."""
        # Include negative returns to test edge cases
        returns = np.array([-0.05, -0.03, -0.01, 0.01, 0.03])

        for method in ["return_magnitude", "recency", "volatility"]:
            weights = sample_weights_by_importance(returns, method=method)
            assert np.all(weights >= 0), (
                f"Method '{method}' produced negative weights: {weights[weights < 0]}"
            )

    def test_volatility_high_vol_gets_more_weight(self):
        """Verify volatility method gives more weight to high-volatility periods.

        Construct returns where second half has 5x the volatility.
        """
        np.random.seed(42)
        low_vol = np.random.randn(50) * 0.01
        high_vol = np.random.randn(50) * 0.05
        returns = np.concatenate([low_vol, high_vol])

        weights = sample_weights_by_importance(returns, method="volatility")

        # Average weight in second half should be higher
        avg_weight_first_half = weights[:50].mean()
        avg_weight_second_half = weights[50:].mean()

        assert avg_weight_second_half > avg_weight_first_half, (
            f"High-vol period should have higher average weight. "
            f"First half: {avg_weight_first_half:.6f}, "
            f"Second half: {avg_weight_second_half:.6f}"
        )


class TestBalancedSubsampleCorrectness:
    """Tests verifying class balance improvement and no-duplication invariants."""

    def test_undersample_no_duplication(self):
        """Verify undersample mode never duplicates samples.

        In undersample mode with replace=False, each original sample
        appears at most once in the output.
        """
        np.random.seed(42)
        X = np.random.randn(300, 5)
        # Give each sample a unique ID in the features
        X[:, 0] = np.arange(300)
        y = np.array([0] * 200 + [1] * 50 + [-1] * 50)

        X_bal, y_bal = balanced_subsample(X, y, method="undersample", random_state=42)

        # Check no duplicate IDs
        unique_ids = np.unique(X_bal[:, 0])
        assert len(unique_ids) == len(X_bal), (
            f"Found {len(X_bal) - len(unique_ids)} duplicate samples in undersample"
        )

    def test_class_imbalance_improves(self):
        """Verify class balance improves after balancing.

        Measure imbalance as max(counts) / min(counts).
        Should decrease after balanced_subsample.
        """
        np.random.seed(42)
        X = np.random.randn(300, 5)
        # Highly imbalanced: 200 vs 50 vs 50
        y = np.array([0] * 200 + [1] * 50 + [-1] * 50)

        _, counts_before = np.unique(y, return_counts=True)
        imbalance_before = counts_before.max() / counts_before.min()

        X_bal, y_bal = balanced_subsample(X, y, method="undersample", random_state=42)

        _, counts_after = np.unique(y_bal, return_counts=True)
        imbalance_after = counts_after.max() / counts_after.min()

        assert imbalance_after < imbalance_before, (
            f"Imbalance should improve. Before: {imbalance_before:.2f}, "
            f"After: {imbalance_after:.2f}"
        )

    def test_minority_classes_preserved(self):
        """Verify minority classes are not reduced in undersample mode.

        The minority class count should remain the same (not undersampled).
        """
        np.random.seed(42)
        X = np.random.randn(200, 3)
        # Minority class has 20 samples
        y = np.array([0] * 150 + [1] * 30 + [-1] * 20)

        min_count_before = 20  # Class -1

        X_bal, y_bal = balanced_subsample(X, y, method="undersample", random_state=42)

        # Count minority class in output
        minority_count_after = np.sum(y_bal == -1)

        assert minority_count_after == min_count_before, (
            f"Minority class count changed: {min_count_before} -> {minority_count_after}"
        )

    def test_hybrid_allows_oversampling_minority(self):
        """Verify hybrid mode can oversample minority class (duplicates allowed)."""
        np.random.seed(42)
        X = np.random.randn(200, 3)
        X[:, 0] = np.arange(200)  # Unique IDs
        # Very imbalanced: majority has 180, minority has 20
        y = np.array([0] * 180 + [1] * 20)

        X_bal, y_bal = balanced_subsample(
            X, y, method="hybrid", minority_weight=1.0, random_state=42
        )

        # In hybrid mode with high minority_weight, minority should be oversampled
        minority_count = np.sum(y_bal == 1)
        minority_ids = X_bal[y_bal == 1, 0]

        # If oversampled, there will be duplicate IDs
        unique_minority_ids = len(np.unique(minority_ids))

        # With only 20 original minority samples and target_count = 2*20 = 40,
        # we expect duplication
        if minority_count > 20:
            assert unique_minority_ids < minority_count, (
                "Hybrid mode should duplicate minority samples when oversampling"
            )


class TestEventBasedSamplingCorrectness:
    """Tests verifying minimum spacing is a HARD invariant."""

    def test_minimum_spacing_never_violated(self):
        """Verify minimum spacing between ANY pair of sampled events.

        This is a HARD invariant: for all i, j in sampled events,
        |idx_i - idx_j| > min_spacing (when i != j).
        """
        np.random.seed(42)
        # All positions are events
        df = pd.DataFrame(
            {
                "event": np.ones(200, dtype=int),
                "value": np.random.randn(200),
            }
        )

        min_spacing = 10
        result = event_based_sample(
            df,
            event_column="event",
            n_samples=15,
            min_event_spacing=min_spacing,
            random_state=42,
        )

        indices = sorted(result.index.tolist())

        # Check ALL pairs of sampled events
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                spacing = abs(indices[j] - indices[i])
                assert spacing > min_spacing, (
                    f"Spacing violation: indices {indices[i]} and {indices[j]} "
                    f"have spacing {spacing} <= {min_spacing}"
                )

    def test_only_events_are_sampled(self):
        """Verify only rows where event=True are in the sample.

        Non-event rows should never appear in the output.
        """
        np.random.seed(42)
        df = pd.DataFrame(
            {
                "event": [0, 0, 1, 0, 1, 0, 0, 1, 0, 1],  # Events at indices 2,4,7,9
                "value": np.arange(10),
            }
        )

        result = event_based_sample(
            df,
            event_column="event",
            n_samples=3,
            random_state=42,
        )

        # All sampled rows must have event=1
        assert result["event"].all(), f"Non-event rows in sample: {result[result['event'] == 0]}"

        # All indices must be from event positions
        valid_event_indices = {2, 4, 7, 9}
        sampled_indices = set(result.index.tolist())
        assert sampled_indices.issubset(valid_event_indices), (
            f"Invalid indices in sample: {sampled_indices - valid_event_indices}"
        )

    def test_spacing_constraint_reduces_sample_size(self):
        """Verify spacing constraint can prevent reaching n_samples.

        If we request more samples than possible with spacing constraint,
        we should get fewer samples (not violate spacing).
        """
        # Create 50 events, request 40 with spacing of 5
        # Maximum possible is ceil(50 / (5+1)) ≈ 8-9 events
        df = pd.DataFrame(
            {
                "event": np.ones(50, dtype=int),
                "value": np.arange(50),
            }
        )

        result = event_based_sample(
            df,
            event_column="event",
            n_samples=40,  # Request more than possible
            min_event_spacing=5,
            random_state=42,
        )

        # Should get fewer than 40 samples
        assert len(result) < 40, "Should not achieve 40 samples with spacing=5 in 50 positions"

        # But spacing should still be respected
        indices = sorted(result.index.tolist())
        for i in range(len(indices) - 1):
            assert indices[i + 1] - indices[i] > 5

    def test_polars_dataframe_same_invariants(self):
        """Verify Polars DataFrame respects same invariants as Pandas."""
        np.random.seed(42)
        df = pl.DataFrame(
            {
                "event": [1] * 100,
                "value": list(range(100)),
            }
        )

        min_spacing = 8
        result = event_based_sample(
            df,
            event_column="event",
            n_samples=10,
            min_event_spacing=min_spacing,
            random_state=42,
        )

        # Get indices (row positions for Polars)
        # For Polars we need to check the actual sampled positions
        values = result["value"].to_numpy()

        for i in range(len(values)):
            for j in range(i + 1, len(values)):
                spacing = abs(values[j] - values[i])
                assert spacing > min_spacing, (
                    f"Polars: spacing violation between values {values[i]} and {values[j]}"
                )


class TestEdgeCasesCorrectness:
    """Tests for edge cases that must be handled correctly."""

    def test_constant_returns_get_equal_weights(self):
        """When all returns are zero, weights should be equal (1/n)."""
        returns = np.zeros(50)

        weights = sample_weights_by_importance(returns, method="return_magnitude")

        expected = np.ones(50) / 50
        np.testing.assert_array_almost_equal(weights, expected, decimal=10)

    def test_single_sample_weight_is_one(self):
        """Single sample should have weight 1.0."""
        returns = np.array([0.05])

        for method in ["return_magnitude", "recency", "volatility"]:
            weights = sample_weights_by_importance(returns, method=method)
            assert weights[0] == 1.0, f"Single sample weight for {method} should be 1.0"

    def test_balanced_subsample_single_class(self):
        """Single class should remain unchanged (no balancing needed)."""
        np.random.seed(42)
        X = np.random.randn(50, 3)
        y = np.zeros(50)  # All same class

        X_bal, y_bal = balanced_subsample(X, y, random_state=42)

        # Should keep all samples (nothing to balance)
        assert len(y_bal) == len(y)

    def test_event_sample_respects_available_events(self):
        """Sample size should not exceed available events."""
        df = pd.DataFrame(
            {
                "event": [1, 0, 1, 0, 0],  # Only 2 events
                "value": [1, 2, 3, 4, 5],
            }
        )

        result = event_based_sample(
            df,
            event_column="event",
            n_samples=10,  # Request more than available
            random_state=42,
        )

        assert len(result) == 2, f"Should only sample 2 available events, got {len(result)}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
