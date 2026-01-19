"""Combination generation and sampling for CPCV.

This module handles the combinatorial aspects of CPCV:
- Generating C(n,k) test group combinations
- Reservoir sampling for large combination spaces
- Lazy iteration over combinations
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Iterator
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    pass


def iter_combinations(
    n_groups: int,
    n_test_groups: int,
    max_combinations: int | None = None,
    random_state: int | None = None,
) -> Iterator[tuple[int, ...]]:
    """Iterate over test group combinations, optionally sampling.

    When max_combinations is None or larger than total combinations,
    yields all C(n_groups, n_test_groups) combinations.

    When max_combinations is smaller, uses reservoir sampling to
    select a random subset without materializing all combinations.

    Parameters
    ----------
    n_groups : int
        Total number of groups to choose from.
    n_test_groups : int
        Number of groups to choose for each combination.
    max_combinations : int, optional
        Maximum number of combinations to yield.
        If None, yields all combinations.
    random_state : int, optional
        Random seed for reproducible sampling.

    Yields
    ------
    tuple[int, ...]
        Test group indices for each combination.

    Examples
    --------
    >>> list(iter_combinations(4, 2))
    [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]

    >>> list(iter_combinations(4, 2, max_combinations=3, random_state=42))
    [(0, 2), (1, 2), (2, 3)]
    """
    total_combinations = math.comb(n_groups, n_test_groups)

    if max_combinations is None or total_combinations <= max_combinations:
        # Yield all combinations directly from generator
        yield from itertools.combinations(range(n_groups), n_test_groups)
    else:
        # Use reservoir sampling for subset selection
        rng = np.random.default_rng(random_state)
        sampled = reservoir_sample_combinations(n_groups, n_test_groups, max_combinations, rng)
        yield from sampled


def reservoir_sample_combinations(
    n_groups: int,
    n_test_groups: int,
    max_combinations: int,
    rng: np.random.Generator,
) -> list[tuple[int, ...]]:
    """Sample combinations using reservoir sampling.

    Samples directly from the combinations iterator without materializing
    all C(n,k) combinations in memory. Time complexity O(C(n,k)) but
    space complexity O(max_combinations).

    Parameters
    ----------
    n_groups : int
        Total number of groups to choose from.
    n_test_groups : int
        Number of groups to choose for each combination.
    max_combinations : int
        Number of combinations to sample.
    rng : np.random.Generator
        Random number generator for reproducible sampling.

    Returns
    -------
    list[tuple[int, ...]]
        Sampled combinations, randomly selected with uniform probability.

    Notes
    -----
    Uses Algorithm R (Vitter, 1985) for reservoir sampling:
    - First k items fill the reservoir
    - Each subsequent item i replaces a random reservoir item with probability k/i
    - Result is a uniform random sample of size k
    """
    reservoir: list[tuple[int, ...]] = []

    for i, combo in enumerate(itertools.combinations(range(n_groups), n_test_groups)):
        if i < max_combinations:
            reservoir.append(combo)
        else:
            # Reservoir sampling: replace with probability max_combinations/(i+1)
            j = rng.integers(0, i + 1)
            if j < max_combinations:
                reservoir[j] = combo

    return reservoir
