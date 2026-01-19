"""Combinatorial Purged Cross-Validation for backtest overfitting detection.

This module implements Combinatorial Purged Cross-Validation (CPCV), which generates
multiple backtest paths by combining different groups of time-series data. This approach
provides a distribution of performance metrics instead of a single path, enabling robust
assessment of strategy viability and detection of backtest overfitting.

Key Concepts
------------

**Combinatorial Splits**:
    Instead of a single chronological train/test split, CPCV partitions data into N groups
    and generates all C(N,k) combinations of choosing k groups for testing. This creates
    a distribution of backtest results rather than a single path.

**Purging**:
    Removes training samples that temporally overlap with test samples within the label
    horizon. Essential for preventing information leakage when labels are forward-looking
    (e.g., future returns). Without purging, the model could train on samples that contain
    information about test set labels.

**Embargo**:
    Creates a buffer period after each test group where training samples are removed.
    Accounts for serial correlation in financial data and prevents training on samples
    that are too close in time to the test set. Can be specified as absolute time
    (embargo_size) or as a percentage of total samples (embargo_pct).

**Session Alignment**:
    Optionally aligns group boundaries to trading session boundaries rather than arbitrary
    indices. Ensures groups represent complete trading days/sessions, which is important
    for intraday strategies.

**Multi-Asset Isolation**:
    When groups parameter is provided, CPCV applies purging per asset independently.
    This prevents cross-asset information leakage and enables proper validation of
    multi-asset strategies.

Usage Example
-------------
Basic usage with purging and embargo::

    import polars as pl
    from ml4t.diagnostic.splitters import CombinatorialPurgedCV

    # Load your time-series data
    df = pl.read_parquet("features.parquet")
    X = df.select(["feature1", "feature2", "feature3"])
    y = df["target"]

    # Configure CPCV with purging for 5-day forward labels
    # and 2-day embargo to account for autocorrelation
    cv = CombinatorialPurgedCV(
        n_groups=8,           # Split into 8 time groups
        n_test_groups=2,      # Use 2 groups for testing in each combination
        label_horizon=5,      # Labels look forward 5 samples
        embargo_size=2,       # Add 2-sample buffer after test set
        max_combinations=20   # Limit to 20 combinations for efficiency
    )

    # Generate train/test splits
    for fold, (train_idx, test_idx) in enumerate(cv.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Train and evaluate your model
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        print(f"Fold {fold}: Score={score:.4f}")

Multi-asset usage with per-asset purging::

    # For multi-asset strategies, provide asset IDs as groups
    assets = df["symbol"]  # e.g., ["AAPL", "MSFT", "GOOGL", ...]

    cv = CombinatorialPurgedCV(
        n_groups=6,
        n_test_groups=2,
        label_horizon=5,
        embargo_size=2,
        isolate_groups=True  # Prevent same asset in train and test
    )

    for train_idx, test_idx in cv.split(X, groups=assets):
        # CPCV automatically applies per-asset purging
        # Each asset's data is purged independently
        pass

Session-aligned usage for intraday strategies::

    import pandas as pd

    # Data with session_date column from qdata.sessions
    df = pd.read_parquet("intraday_features.parquet")
    # df has columns: timestamp, session_date, feature1, feature2, ...

    cv = CombinatorialPurgedCV(
        n_groups=10,
        n_test_groups=2,
        label_horizon=pd.Timedelta(minutes=30),  # 30-minute forward labels
        embargo_size=pd.Timedelta(minutes=15),   # 15-minute embargo
        align_to_sessions=True,                   # Align groups to sessions
        session_col="session_date"                # Column with session IDs
    )

    for train_idx, test_idx in cv.split(df):
        # Group boundaries now align to complete trading sessions
        pass

References
----------
.. [1] Bailey, D. H., Borwein, J., López de Prado, M., & Zhu, Q. J. (2014).
       "The Probability of Backtest Overfitting." Journal of Computational Finance.

.. [2] López de Prado, M. (2018). "Advances in Financial Machine Learning."
       Wiley. Chapter 7: Cross-Validation in Finance.
"""

from __future__ import annotations

import math
from collections.abc import Generator
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd
import polars as pl

from ml4t.diagnostic.backends.adapter import DataFrameAdapter
from ml4t.diagnostic.splitters.base import BaseSplitter
from ml4t.diagnostic.splitters.config import CombinatorialPurgedConfig
from ml4t.diagnostic.splitters.cpcv import (
    apply_multi_asset_purging,
    apply_single_asset_purging,
    create_contiguous_partitions,
    create_session_partitions,
    iter_combinations,
    timestamp_window_from_indices,
    validate_contiguous_partitions,
)
from ml4t.diagnostic.splitters.group_isolation import isolate_groups_from_train

if TYPE_CHECKING:
    from numpy.typing import NDArray


class CombinatorialPurgedCV(BaseSplitter):
    """Combinatorial Purged Cross-Validation for backtest overfitting detection.

    CPCV partitions the time series into N contiguous groups and forms all combinations
    C(N,k) of choosing k groups for testing. This generates multiple backtest paths
    instead of a single chronological split, providing a robust assessment of strategy
    performance and enabling detection of backtest overfitting.

    How It Works
    ------------

    1. **Partitioning**: Divide time-series data into N contiguous groups of equal size
    2. **Combination Generation**: Generate all C(N,k) combinations of choosing k groups for testing
    3. **Purging**: For each combination, remove training samples that overlap with test labels
    4. **Embargo**: Optionally add buffer periods after test groups to account for autocorrelation
    5. **Multi-Asset Handling**: When groups are provided, apply purging independently per asset

    Purging Mechanics
    -----------------

    **Why Purge?**
        When labels are forward-looking (e.g., 5-day returns), training samples near the test
        set temporally overlap with test labels. Without purging, the model trains on information
        about test outcomes, leading to inflated performance estimates.

    **How Purging Works**:
        For each test group with range [t_start, t_end]:

        1. Remove train samples where: ``t_train > t_start - label_horizon``
        2. This ensures no training sample's label period overlaps with test samples

    **Example**::

        Test group: samples 100-119 (20 samples)
        Label horizon: 5 samples
        Purging removes: training samples 95-99
        Reason: Sample 95's label looks forward to sample 100 (first test sample)

    Embargo Mechanics
    -----------------

    **Why Embargo?**
        Financial data exhibits serial correlation - adjacent samples are not independent.
        Even with purging, training on samples immediately before the test set can leak
        information through autocorrelation.

    **How Embargo Works**:
        After purging, additionally remove a buffer of samples immediately after each test group:

        - **embargo_size**: Absolute number of samples (e.g., 10 samples)
        - **embargo_pct**: Percentage of total samples (e.g., 0.01 = 1%)

    **Example**::

        Test group: samples 100-119
        Embargo: 5 samples
        Additional removal: training samples 120-124
        Result: Creates 5-sample buffer after test group

    Multi-Asset Purging
    -------------------

    When ``groups`` parameter is provided (e.g., asset symbols), CPCV applies purging
    independently for each asset. This prevents cross-asset leakage:

    **Process**:
        1. For each asset, find its training and test samples
        2. Apply purging/embargo only to that asset's data
        3. Combine results across all assets

    **Why Important?**
        Without per-asset purging, information could leak between assets that trade at
        different times (e.g., European markets vs US markets).

    Based on Bailey et al. (2014) "The Probability of Backtest Overfitting" and
    López de Prado (2018) "Advances in Financial Machine Learning".

    Parameters
    ----------
    n_groups : int, default=8
        Number of contiguous groups to partition the time series into.

    n_test_groups : int, default=2
        Number of groups to use for testing in each combination.

    label_horizon : int or pd.Timedelta, default=0
        Forward-looking period of labels for purging calculation.

    embargo_size : int or pd.Timedelta, optional
        Size of embargo period after each test group.

    embargo_pct : float, optional
        Embargo size as percentage of total samples.

    max_combinations : int, optional
        Maximum number of combinations to generate. If None, generates all C(N,k).
        Use this to limit computational cost for large N.

    random_state : int, optional
        Random seed for combination sampling when max_combinations is set.

    align_to_sessions : bool, default=False
        If True, align group boundaries to trading session boundaries.
        Requires X to have a session column (specified by session_col parameter).

        Trading sessions should be assigned using the qdata library before cross-validation:
        - Use DataManager with exchange/calendar parameters, or
        - Use SessionAssigner.from_exchange('CME') directly

    session_col : str, default='session_date'
        Name of the column containing session identifiers.
        Only used if align_to_sessions=True.
        This column should be added by qdata.sessions.SessionAssigner

    isolate_groups : bool, default=True
        If True, prevent the same group (asset/symbol) from appearing in both
        train and test sets. This is enabled by default for CPCV as it's designed
        for multi-asset validation.

        Requires passing `groups` parameter to split() method with asset IDs.

        Note: CPCV already applies per-asset purging when groups are provided.
        This parameter provides additional group isolation guarantee.

    Attributes:
    ----------
    n_groups_ : int
        The number of groups.

    n_test_groups_ : int
        The number of test groups.

    Examples:
    --------
    >>> import numpy as np
    >>> from ml4t.diagnostic.splitters import CombinatorialPurgedCV
    >>> X = np.arange(200).reshape(200, 1)
    >>> cv = CombinatorialPurgedCV(n_groups=6, n_test_groups=2, label_horizon=5)
    >>> combinations = list(cv.split(X))
    >>> print(f"Generated {len(combinations)} combinations")
    Generated 15 combinations

    >>> # Each combination provides train/test indices
    >>> for i, (train, test) in enumerate(combinations[:3]):
    ...     print(f"Combination {i+1}: Train={len(train)}, Test={len(test)}")
    Combination 1: Train=125, Test=50
    Combination 2: Train=125, Test=50
    Combination 3: Train=125, Test=50

    Notes:
    -----
    The total number of combinations is C(n_groups, n_test_groups). For large values,
    this can become computationally expensive:
    - C(8,2) = 28 combinations
    - C(10,3) = 120 combinations
    - C(12,4) = 495 combinations

    Use max_combinations to limit computational cost for large datasets.
    """

    def __init__(
        self,
        config: CombinatorialPurgedConfig | None = None,
        *,
        n_groups: int = 8,
        n_test_groups: int = 2,
        label_horizon: int | pd.Timedelta = 0,
        embargo_size: int | pd.Timedelta | None = None,
        embargo_pct: float | None = None,
        max_combinations: int | None = None,
        random_state: int | None = None,
        align_to_sessions: bool = False,
        session_col: str = "session_date",
        timestamp_col: str | None = None,
        isolate_groups: bool = True,
    ) -> None:
        """Initialize CombinatorialPurgedCV.

        This splitter uses a config-first architecture. You can either:
        1. Pass a config object: CombinatorialPurgedCV(config=my_config)
        2. Pass individual parameters: CombinatorialPurgedCV(n_groups=8, n_test_groups=2)

        Parameters are automatically converted to a config object internally,
        ensuring a single source of truth for all validation and logic.

        Examples
        --------
        >>> # Approach 1: Direct parameters (convenient)
        >>> cv = CombinatorialPurgedCV(n_groups=10, n_test_groups=3)
        >>>
        >>> # Approach 2: Config object (for serialization/reproducibility)
        >>> from ml4t.diagnostic.splitters.config import CombinatorialPurgedConfig
        >>> config = CombinatorialPurgedConfig(n_groups=10, n_test_groups=3)
        >>> cv = CombinatorialPurgedCV(config=config)
        >>>
        >>> # Config can be serialized
        >>> config.to_json("cpcv_config.json")
        >>> loaded = CombinatorialPurgedConfig.from_json("cpcv_config.json")
        >>> cv = CombinatorialPurgedCV(config=loaded)
        """
        # Config-first: either use provided config or create from params
        if config is not None:
            # Verify no conflicting parameters when config is provided
            self._validate_no_param_conflicts(
                n_groups,
                n_test_groups,
                label_horizon,
                embargo_size,
                embargo_pct,
                max_combinations,
                random_state,
                align_to_sessions,
                session_col,
                timestamp_col,
                isolate_groups,
            )
            self.config = config
        else:
            # Create config from individual parameters
            # Note: embargo validation (mutual exclusivity) handled by config
            self.config = self._create_config_from_params(
                n_groups,
                n_test_groups,
                label_horizon,
                embargo_size,
                embargo_pct,
                max_combinations,
                random_state,
                align_to_sessions,
                session_col,
                timestamp_col,
                isolate_groups,
            )

        # Use parameter if provided, otherwise use config value
        # This allows random_state to be passed either via config or direct parameter
        self.random_state = random_state if random_state is not None else self.config.random_state

    def _validate_no_param_conflicts(
        self,
        n_groups: int,
        n_test_groups: int,
        label_horizon: int | pd.Timedelta,
        embargo_size: int | pd.Timedelta | None,
        embargo_pct: float | None,
        max_combinations: int | None,
        random_state: int | None,
        align_to_sessions: bool,
        session_col: str,
        timestamp_col: str | None,
        isolate_groups: bool,
    ) -> None:
        """Validate no conflicting parameters when config is provided."""

        def is_semantically_default(value: Any, default: Any) -> bool:
            """Check if value is semantically equal to default.

            Handles heterogeneous types:
            - pd.Timedelta(0) is semantically equal to 0
            - np.int64(0) is semantically equal to 0
            - None equals None
            """
            if value is None and default is None:
                return True
            if value is None or default is None:
                return False
            # Handle Timedelta vs int comparison for label_horizon/embargo_size
            if isinstance(value, pd.Timedelta):
                if isinstance(default, int) and default == 0:
                    return value == pd.Timedelta(0)
                return value == default
            if isinstance(default, pd.Timedelta):
                if isinstance(value, int) and value == 0:
                    return default == pd.Timedelta(0)
                return value == default
            # Handle numpy int types vs Python int
            try:
                return bool(value == default)
            except (TypeError, ValueError):
                return False

        # Check for non-default parameter values
        # Note: random_state is NOT in this list because it's now in config.
        # Users can pass random_state as a parameter to override config.random_state.
        param_checks = [
            ("n_groups", n_groups, 8),
            ("n_test_groups", n_test_groups, 2),
            ("label_horizon", label_horizon, 0),
            ("embargo_size", embargo_size, None),
            ("embargo_pct", embargo_pct, None),
            ("max_combinations", max_combinations, None),
            ("align_to_sessions", align_to_sessions, False),
            ("session_col", session_col, "session_date"),
            ("timestamp_col", timestamp_col, None),
            ("isolate_groups", isolate_groups, True),
        ]

        non_default_params = [
            name
            for name, value, default in param_checks
            if not is_semantically_default(value, default)
        ]

        if non_default_params:
            raise ValueError(
                f"Cannot specify both 'config' and individual parameters. "
                f"Got config plus: {', '.join(non_default_params)}"
            )

    def _create_config_from_params(
        self,
        n_groups: int,
        n_test_groups: int,
        label_horizon: int | pd.Timedelta,
        embargo_size: int | pd.Timedelta | None,
        embargo_pct: float | None,
        max_combinations: int | None,
        random_state: int | None,
        align_to_sessions: bool,
        session_col: str,
        timestamp_col: str | None,
        isolate_groups: bool,
    ) -> CombinatorialPurgedConfig:
        """Create config object from individual parameters."""
        return CombinatorialPurgedConfig(
            n_groups=n_groups,
            n_test_groups=n_test_groups,
            label_horizon=label_horizon,
            embargo_td=embargo_size,
            embargo_pct=embargo_pct,
            max_combinations=max_combinations,
            random_state=random_state,
            align_to_sessions=align_to_sessions,
            session_col=session_col,
            timestamp_col=timestamp_col,
            isolate_groups=isolate_groups,
        )

    # Property accessors for config values (clean API)
    @property
    def n_groups(self) -> int:
        """Number of groups to partition timeline into."""
        return self.config.n_groups

    @property
    def n_test_groups(self) -> int:
        """Number of groups per test set."""
        return self.config.n_test_groups

    @property
    def label_horizon(self) -> int | pd.Timedelta:
        """Forward-looking period of labels (int samples or Timedelta)."""
        return self.config.label_horizon

    @property
    def embargo_size(self) -> int | pd.Timedelta | None:
        """Embargo buffer size (int samples or Timedelta)."""
        return self.config.embargo_td

    @property
    def embargo_pct(self) -> float | None:
        """Embargo size as percentage of total samples."""
        return self.config.embargo_pct

    @property
    def max_combinations(self) -> int | None:
        """Maximum number of folds to generate."""
        return self.config.max_combinations

    @property
    def align_to_sessions(self) -> bool:
        """Whether to align group boundaries to sessions."""
        return self.config.align_to_sessions

    @property
    def session_col(self) -> str:
        """Column name containing session identifiers."""
        return self.config.session_col

    @property
    def timestamp_col(self) -> str | None:
        """Column name containing timestamps for time-based operations."""
        return self.config.timestamp_col

    @property
    def isolate_groups(self) -> bool:
        """Whether to prevent group overlap between train/test."""
        return self.config.isolate_groups

    def get_n_splits(
        self,
        X: pl.DataFrame | pd.DataFrame | NDArray[Any] | None = None,
        y: pl.Series | pd.Series | NDArray[Any] | None = None,
        groups: pl.Series | pd.Series | NDArray[Any] | None = None,
    ) -> int:
        """Get number of splits (combinations).

        Parameters
        ----------
        X : array-like, optional
            Always ignored, exists for compatibility.

        y : array-like, optional
            Always ignored, exists for compatibility.

        groups : array-like, optional
            Always ignored, exists for compatibility.

        Returns:
        -------
        n_splits : int
            Number of combinations that will be generated.
        """
        del X, y, groups  # Unused, for sklearn compatibility
        total_combinations = math.comb(self.n_groups, self.n_test_groups)

        if self.max_combinations is None:
            return total_combinations
        return min(self.max_combinations, total_combinations)

    def split(
        self,
        X: pl.DataFrame | pd.DataFrame | NDArray[Any],
        y: pl.Series | pd.Series | NDArray[Any] | None = None,
        groups: pl.Series | pd.Series | NDArray[Any] | None = None,
    ) -> Generator[tuple[NDArray[np.intp], NDArray[np.intp]], None, None]:
        """Generate train/test indices for combinatorial splits with purging and embargo.

        This method generates all combinations C(N,k) of train/test splits, applying
        purging and embargo to prevent information leakage. Each yielded split represents
        an independent backtest path.

        Parameters
        ----------
        X : DataFrame or ndarray of shape (n_samples, n_features)
            Training data. Must have a datetime index if using Timedelta-based
            label_horizon or embargo_size.

        y : Series or ndarray of shape (n_samples,), optional
            Target variable. Not used in splitting logic, but accepted for
            API compatibility with scikit-learn.

        groups : Series or ndarray of shape (n_samples,), optional
            Group labels for samples (e.g., asset symbols for multi-asset strategies).

            When provided:
            - Purging is applied independently per group (asset)
            - Prevents information leakage across groups
            - Essential for multi-asset portfolio validation

            Example: ``groups = df["symbol"]``  # ["AAPL", "MSFT", "GOOGL", ...]

        Yields
        ------
        train : ndarray of shape (n_train_samples,)
            Indices of training samples for this combination.
            Purging and embargo have been applied to remove:
            - Samples overlapping with test labels (purging)
            - Samples in embargo buffer after test groups (embargo)

        test : ndarray of shape (n_test_samples,)
            Indices of test samples for this combination.
            Consists of samples from the k selected test groups.

        Raises
        ------
        ValueError
            If X has incompatible shape or missing required columns
            (e.g., session_col when align_to_sessions=True).

        TypeError
            If X index is not datetime when using Timedelta parameters.

        Notes
        -----
        **Number of Combinations**:
            Generates C(n_groups, n_test_groups) combinations. For example:
            - C(8,2) = 28 combinations
            - C(10,3) = 120 combinations
            - C(12,4) = 495 combinations

            Use ``max_combinations`` parameter to limit the number of splits generated.

        **Purging Logic**:
            For each test group:
            1. Identify test sample range [t_start, t_end]
            2. Remove training samples where: t_train > t_start - label_horizon
            3. This prevents training on samples whose labels overlap with test period

        **Embargo Logic**:
            After purging, additionally remove training samples:
            - In range [t_end + 1, t_end + embargo_size]
            - This accounts for serial correlation in financial time series

        **Multi-Asset Handling**:
            When ``groups`` is provided:
            1. For each asset, find its training and test indices
            2. Apply purging/embargo independently to that asset's data
            3. Combine purged results across all assets
            4. This prevents cross-asset information leakage

        **Session Alignment**:
            When ``align_to_sessions=True``:
            - Group boundaries align to trading session boundaries
            - Ensures each group contains complete trading days/sessions
            - Requires X to have column specified by ``session_col`` parameter

        Examples
        --------
        Basic usage with purging::

            >>> import polars as pl
            >>> from ml4t.diagnostic.splitters import CombinatorialPurgedCV
            >>>
            >>> # Create sample data
            >>> n = 1000
            >>> X = pl.DataFrame({"feature1": range(n), "feature2": range(n, 2*n)})
            >>> y = pl.Series(range(n))
            >>>
            >>> # Configure CPCV
            >>> cv = CombinatorialPurgedCV(
            ...     n_groups=8,
            ...     n_test_groups=2,
            ...     label_horizon=5,
            ...     embargo_size=2
            ... )
            >>>
            >>> # Generate splits
            >>> for fold, (train_idx, test_idx) in enumerate(cv.split(X)):
            ...     print(f"Fold {fold}: Train={len(train_idx)}, Test={len(test_idx)}")
            Fold 0: Train=739, Test=250
            Fold 1: Train=739, Test=250
            ...

        Multi-asset usage::

            >>> # Multi-asset data with symbol column
            >>> symbols = pl.Series(["AAPL"] * 250 + ["MSFT"] * 250 +
            ...                      ["GOOGL"] * 250 + ["AMZN"] * 250)
            >>>
            >>> cv = CombinatorialPurgedCV(
            ...     n_groups=6,
            ...     n_test_groups=2,
            ...     label_horizon=5,
            ...     embargo_size=2,
            ...     isolate_groups=True
            ... )
            >>>
            >>> for train_idx, test_idx in cv.split(X, groups=symbols):
            ...     # Purging applied independently per asset
            ...     train_symbols = symbols[train_idx].unique()
            ...     test_symbols = symbols[test_idx].unique()

        Session-aligned usage::

            >>> import pandas as pd
            >>>
            >>> # Intraday data with session dates
            >>> df = pd.DataFrame({
            ...     "timestamp": pd.date_range("2024-01-01", periods=1000, freq="1min"),
            ...     "session_date": pd.date_range("2024-01-01", periods=1000, freq="1min").date,
            ...     "feature1": range(1000)
            ... })
            >>>
            >>> cv = CombinatorialPurgedCV(
            ...     n_groups=10,
            ...     n_test_groups=2,
            ...     label_horizon=pd.Timedelta(minutes=30),
            ...     embargo_size=pd.Timedelta(minutes=15),
            ...     align_to_sessions=True,
            ...     session_col="session_date"
            ... )
            >>>
            >>> for train_idx, test_idx in cv.split(df):
            ...     # Group boundaries aligned to session boundaries
            ...     pass

        See Also
        --------
        CombinatorialPurgedConfig : Configuration object for CPCV parameters
        apply_purging_and_embargo : Low-level purging/embargo function
        BaseSplitter : Base class for all splitters
        """
        # Validate inputs (no numpy conversion - performance optimization)
        n_samples = self._validate_inputs(X, y, groups)

        # Validate session alignment if enabled
        self._validate_session_alignment(X, self.align_to_sessions, self.session_col)

        # Extract timestamps if available (supports both Polars and pandas)
        timestamps = self._extract_timestamps(X, self.timestamp_col)

        # Create group indices or boundaries
        # For session-aligned mode, we need exact indices (not boundaries) to handle
        # non-contiguous/interleaved data correctly
        if self.align_to_sessions:
            # align_to_sessions requires X to be a DataFrame (validation enforces this)
            # Use new method that returns exact indices per group
            group_indices_list = self._create_session_group_indices(
                cast(pl.DataFrame | pd.DataFrame, X)
            )
            use_exact_indices = True
            # Also create boundaries for backward compatibility with purging logic
            group_boundaries = [
                (int(indices[0]), int(indices[-1]) + 1) if len(indices) > 0 else (0, 0)
                for indices in group_indices_list
            ]
        else:
            group_boundaries = self._create_group_boundaries(n_samples)
            group_indices_list = None
            use_exact_indices = False

        # Generate combinations with memory-efficient sampling when max_combinations is set
        # Uses reservoir sampling when needed to avoid materializing all C(n,k) combinations
        combinations = iter_combinations(
            self.n_groups,
            self.n_test_groups,
            self.max_combinations,
            self.random_state,
        )

        # Generate splits for each combination
        for test_group_indices in combinations:
            # Create test set from selected groups
            if use_exact_indices and group_indices_list is not None:
                # Use exact indices (correct for non-contiguous/interleaved data)
                test_arrays = [group_indices_list[g] for g in test_group_indices]
                test_indices_array = (
                    np.concatenate(test_arrays) if test_arrays else np.array([], dtype=np.intp)
                )
            else:
                # Use boundaries with range (only correct for contiguous data)
                test_indices: list[int] = []
                for group_idx in test_group_indices:
                    start_idx, end_idx = group_boundaries[group_idx]
                    test_indices.extend(range(start_idx, end_idx))
                test_indices_array = np.array(test_indices, dtype=np.intp)

            # Create initial training set from remaining groups
            train_group_indices_list = [
                i for i in range(self.n_groups) if i not in test_group_indices
            ]
            if use_exact_indices and group_indices_list is not None:
                # Use exact indices
                train_arrays = [group_indices_list[g] for g in train_group_indices_list]
                train_indices_array = (
                    np.concatenate(train_arrays) if train_arrays else np.array([], dtype=np.intp)
                )
            else:
                # Use boundaries with range
                train_indices: list[int] = []
                for group_idx in train_group_indices_list:
                    start_idx, end_idx = group_boundaries[group_idx]
                    train_indices.extend(range(start_idx, end_idx))
                train_indices_array = np.array(train_indices, dtype=np.intp)

            # Apply purging and embargo between test groups and training data
            clean_train_indices = self._apply_group_purging_and_embargo(
                train_indices_array,
                test_group_indices,
                group_boundaries,
                n_samples,
                timestamps,
                groups,  # Pass groups for multi-asset awareness
                group_indices_list,  # Pass exact indices for session-aligned purging
            )

            # Apply group isolation if requested
            if self.isolate_groups and groups is not None:
                clean_train_indices = isolate_groups_from_train(
                    clean_train_indices, test_indices_array, groups
                )

            # CPCV Invariant: train set must not be empty after purging
            if len(clean_train_indices) == 0:
                raise ValueError(
                    f"CPCV invariant violated: train set is empty after purging/embargo. "
                    f"Test groups: {test_group_indices}. "
                    f"Consider reducing label_horizon ({self.label_horizon}) or "
                    f"embargo_size ({self.embargo_size}) or embargo_pct ({self.embargo_pct})."
                )

            # CPCV Invariant: train and test sets must be disjoint
            overlap = np.intersect1d(clean_train_indices, test_indices_array)
            if len(overlap) > 0:
                raise ValueError(
                    f"CPCV invariant violated: train and test sets have {len(overlap)} "
                    f"overlapping indices. First few: {overlap[:5].tolist()}"
                )

            # Return sorted indices for deterministic behavior
            yield np.sort(clean_train_indices), np.sort(test_indices_array)

    def _create_group_boundaries(self, n_samples: int) -> list[tuple[int, int]]:
        """Create boundaries for contiguous groups.

        Delegates to cpcv.partitioning.create_contiguous_partitions.

        Parameters
        ----------
        n_samples : int
            Total number of samples.

        Returns:
        -------
        boundaries : list of tuple
            List of (start_idx, end_idx) for each group.

        Raises
        ------
        ValueError
            If boundaries don't satisfy CPCV invariants.
        """
        return create_contiguous_partitions(n_samples, self.n_groups)

    def _validate_group_boundaries(self, boundaries: list[tuple[int, int]], n_samples: int) -> None:
        """Validate CPCV group boundary invariants.

        Delegates to cpcv.partitioning.validate_contiguous_partitions.
        """
        validate_contiguous_partitions(boundaries, n_samples)

    def _create_session_group_indices(
        self,
        X: pl.DataFrame | pd.DataFrame,
    ) -> list[NDArray[np.intp]]:
        """Create exact index arrays per group, aligned to session boundaries.

        Delegates to cpcv.partitioning.create_session_partitions.

        Unlike _create_group_boundaries which returns (start, end) ranges suitable
        for contiguous data, this method returns EXACT index arrays for each group.
        This is critical for correct behavior with non-contiguous or interleaved data.

        Parameters
        ----------
        X : DataFrame
            Data with session column.

        Returns
        -------
        group_indices : list of np.ndarray
            List of numpy arrays containing exact row indices for each group.
        """
        return create_session_partitions(
            X, self.session_col, self.n_groups, self._session_to_indices
        )

    @staticmethod
    def _timestamp_window_from_indices(
        indices: NDArray[np.intp],
        timestamps: pd.DatetimeIndex,
    ) -> tuple[pd.Timestamp, pd.Timestamp] | None:
        """Compute timestamp window from actual indices (for session-aligned purging).

        Delegates to cpcv.windows.timestamp_window_from_indices.

        Parameters
        ----------
        indices : ndarray
            Row indices of test samples.
        timestamps : pd.DatetimeIndex
            Timestamps for all samples.

        Returns
        -------
        tuple or None
            (start_time, end_time_exclusive) if indices non-empty, None if empty.
        """
        window = timestamp_window_from_indices(indices, timestamps)
        if window is None:
            return None
        return window.start, window.end_exclusive

    def _apply_group_purging_and_embargo(
        self,
        train_indices: NDArray[np.intp],
        test_group_indices: tuple[int, ...],
        group_boundaries: list[tuple[int, int]],
        n_samples: int,
        timestamps: pd.DatetimeIndex | None,
        groups: pl.Series | pd.Series | NDArray[Any] | None = None,
        group_indices_list: list[NDArray[np.intp]] | None = None,
    ) -> NDArray[np.intp]:
        """Apply purging and embargo between test groups and training data.

        This method handles both single-asset and multi-asset scenarios.
        For multi-asset data, purging is applied per asset to prevent
        cross-asset look-ahead bias.

        Parameters
        ----------
        train_indices : ndarray
            Initial training indices.

        test_group_indices : tuple of int
            Indices of groups used for testing.

        group_boundaries : list of tuple
            Boundaries of all groups (used for non-session-aligned mode).

        n_samples : int
            Total number of samples.

        timestamps : pd.DatetimeIndex, optional
            Timestamps for the data.

        groups : array-like, optional
            Group labels for multi-asset data (e.g., asset IDs).
            If None, applies single-asset purging logic.

        group_indices_list : list of ndarray, optional
            Exact indices per group (for session-aligned mode). When provided
            along with timestamps, purging uses actual timestamp bounds instead
            of (min_idx, max_idx) boundaries.

        Returns:
        -------
        clean_indices : ndarray
            Training indices after purging and embargo.
        """
        if groups is None:
            # Single-asset case: apply global purging
            return self._apply_single_asset_purging(
                train_indices,
                test_group_indices,
                group_boundaries,
                n_samples,
                timestamps,
                group_indices_list,
            )
        # Multi-asset case: apply per-asset purging
        return self._apply_multi_asset_purging(
            train_indices,
            test_group_indices,
            group_boundaries,
            n_samples,
            timestamps,
            groups,
            group_indices_list,
        )

    def _apply_single_asset_purging(
        self,
        train_indices: NDArray[np.intp],
        test_group_indices: tuple[int, ...],
        group_boundaries: list[tuple[int, int]],
        n_samples: int,
        timestamps: pd.DatetimeIndex | None,
        group_indices_list: list[NDArray[np.intp]] | None = None,
    ) -> NDArray[np.intp]:
        """Apply purging for single-asset data.

        Delegates to cpcv.purge_engine.apply_single_asset_purging.
        """
        return apply_single_asset_purging(
            train_indices=train_indices,
            test_group_indices=test_group_indices,
            group_boundaries=group_boundaries,
            n_samples=n_samples,
            timestamps=timestamps,
            label_horizon=self.label_horizon,
            embargo_size=self.embargo_size,
            embargo_pct=self.embargo_pct,
            group_indices_list=group_indices_list,
        )

    def _apply_multi_asset_purging(
        self,
        train_indices: NDArray[np.intp],
        test_group_indices: tuple[int, ...],
        group_boundaries: list[tuple[int, int]],
        n_samples: int,
        timestamps: pd.DatetimeIndex | None,
        groups: pl.Series | pd.Series | NDArray[Any],
        group_indices_list: list[NDArray[np.intp]] | None = None,
    ) -> NDArray[np.intp]:
        """Apply purging for multi-asset data with per-asset isolation.

        Delegates to cpcv.purge_engine.apply_multi_asset_purging.
        """
        # Convert groups to numpy array for consistent indexing
        groups_array = DataFrameAdapter.to_numpy(groups).flatten()

        return apply_multi_asset_purging(
            train_indices=train_indices,
            test_group_indices=test_group_indices,
            group_boundaries=group_boundaries,
            n_samples=n_samples,
            timestamps=timestamps,
            groups_array=groups_array,
            label_horizon=self.label_horizon,
            embargo_size=self.embargo_size,
            embargo_pct=self.embargo_pct,
            group_indices_list=group_indices_list,
        )

    def _validate_inputs(
        self,
        X: pl.DataFrame | pd.DataFrame | NDArray[Any],
        y: pl.Series | pd.Series | NDArray[Any] | None = None,
        groups: pl.Series | pd.Series | NDArray[Any] | None = None,
    ) -> int:
        """Validate input shapes and return number of samples.

        Unlike the previous implementation, this does NOT convert to numpy
        for performance - just validates shapes directly.
        """
        # Use base class validation (handles all input types efficiently)
        n_samples = self._validate_data(X, y, groups)

        # Validate minimum samples per group
        min_samples_per_group = n_samples // self.n_groups
        if min_samples_per_group < 1:
            raise ValueError(
                f"Not enough samples ({n_samples}) for {self.n_groups} groups. Need at least {self.n_groups} samples.",
            )

        return n_samples
