"""Tests for block bootstrap function.

This module tests the block bootstrap functionality after renaming
from sequential_bootstrap to block_bootstrap to accurately reflect
the implementation.
"""

import numpy as np
import pytest

from ml4t.diagnostic.core import block_bootstrap


class TestBlockBootstrap:
    """Test block bootstrap sampling method."""

    def test_basic_functionality(self):
        """Test basic block bootstrap functionality."""
        indices = np.arange(100)
        n_samples = 80
        sample_length = 5

        result = block_bootstrap(indices, n_samples, sample_length, random_state=42)

        # Check output shape
        assert len(result) == n_samples

        # Check all samples are valid indices
        assert np.all(np.isin(result, indices))

    def test_preserves_blocks(self):
        """Test that block structure is preserved."""
        indices = np.arange(50)
        n_samples = 20
        sample_length = 4

        result = block_bootstrap(indices, n_samples, sample_length, random_state=42)

        # Verify that we have blocks of consecutive indices
        # (allowing for block boundaries)
        differences = []
        for i in range(len(result) - 1):
            diff = result[i + 1] - result[i]
            differences.append(diff)

        # Most differences should be 1 (within blocks)
        # Some will be negative or large (between blocks)
        consecutive_count = sum(1 for d in differences if d == 1)
        assert consecutive_count > len(differences) * 0.5  # At least 50% consecutive

    def test_sample_length_none(self):
        """Test with sample_length=None (should default to 10% of data)."""
        indices = np.arange(100)
        n_samples = 50

        result = block_bootstrap(indices, n_samples, sample_length=None, random_state=42)

        assert len(result) == n_samples
        assert np.all(np.isin(result, indices))

    def test_edge_cases(self):
        """Test edge cases."""
        indices = np.arange(10)

        # Test with n_samples = 1
        result = block_bootstrap(indices, n_samples=1, sample_length=3)
        assert len(result) == 1

        # Test with sample_length >= len(indices)
        result = block_bootstrap(indices, n_samples=5, sample_length=15)
        assert len(result) == 5

        # Test with very large n_samples
        result = block_bootstrap(indices, n_samples=100, sample_length=2)
        assert len(result) == 100

    def test_random_state_reproducibility(self):
        """Test that random_state ensures reproducibility."""
        indices = np.arange(50)
        n_samples = 30
        sample_length = 5

        result1 = block_bootstrap(indices, n_samples, sample_length, random_state=42)
        result2 = block_bootstrap(indices, n_samples, sample_length, random_state=42)
        result3 = block_bootstrap(indices, n_samples, sample_length, random_state=43)

        # Same seed should produce same results
        np.testing.assert_array_equal(result1, result2)

        # Different seed should produce different results
        assert not np.array_equal(result1, result3)

    def test_invalid_inputs(self):
        """Test error handling for invalid inputs."""
        indices = np.arange(100)

        # Test n_samples <= 0
        with pytest.raises(ValueError, match="n_samples must be positive"):
            block_bootstrap(indices, n_samples=0)

        # Test empty indices
        with pytest.raises(ValueError, match="indices array cannot be empty"):
            block_bootstrap(np.array([]), n_samples=10)

        # Test invalid sample_length
        with pytest.raises(ValueError, match="sample_length must be positive"):
            block_bootstrap(indices, n_samples=10, sample_length=0)

    def test_large_dataset_performance(self):
        """Test performance with larger dataset."""
        indices = np.arange(10000)
        n_samples = 5000
        sample_length = 50

        # Should complete without errors
        result = block_bootstrap(indices, n_samples, sample_length, random_state=42)

        assert len(result) == n_samples
        assert np.all(result >= 0)
        assert np.all(result < 10000)

    def test_block_structure_validation(self):
        """Validate that the function truly implements block bootstrap."""
        # Create distinctive indices to track blocks
        indices = np.arange(1000) * 10  # 0, 10, 20, 30, ...
        n_samples = 100
        sample_length = 10

        result = block_bootstrap(indices, n_samples, sample_length, random_state=42)

        # Check that blocks are preserved
        # Within each block of sample_length, differences should be consistent
        blocks_found = 0
        i = 0
        while i < len(result) - sample_length:
            # Check if we have a complete block
            block = result[i : i + sample_length]
            differences = np.diff(block)

            # In a proper block, all differences should be the same (10 in this case)
            if len(np.unique(differences)) == 1 and differences[0] == 10:
                blocks_found += 1
                i += sample_length
            else:
                i += 1

        # Should find at least some complete blocks
        assert blocks_found > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
