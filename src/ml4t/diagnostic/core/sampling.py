"""Stratified and subsampling logic for financial time-series.

This module provides sampling strategies that preserve important
characteristics of financial data while reducing computational load
or balancing classes.
"""

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import polars as pl

if TYPE_CHECKING:
    from numpy.typing import NDArray


def block_bootstrap(
    indices: "NDArray[np.intp]",
    n_samples: int,
    sample_length: int | None = None,
    random_state: int | None = None,
) -> "NDArray[np.intp]":
    """Block bootstrap for time series with temporal structure.

    This method samples random blocks (contiguous sequences) of observations and includes subsequent
    observations to preserve temporal structure and label overlap patterns.
    Based on López de Prado (2018).

    Parameters
    ----------
    indices : np.ndarray
        Array of indices to sample from
    n_samples : int
        Number of bootstrap samples to generate
    sample_length : int, optional
        Length of each sequential sample. If None, uses average
        length from original data
    random_state : int, optional
        Random seed for reproducibility

    Returns:
    -------
    np.ndarray
        Bootstrap sample indices

    Raises:
    ------
    ValueError
        If n_samples <= 0, if indices is empty, or if parameters are invalid

    Examples:
    --------
    >>> indices = np.arange(100)
    >>> bootstrap_idx = block_bootstrap(indices, n_samples=80, sample_length=5)
    >>> len(bootstrap_idx)
    80
    """
    # Import here to avoid circular dependency
    from ml4t.diagnostic.core.numba_utils import block_bootstrap_numba

    # Input validation
    if n_samples <= 0:
        raise ValueError(f"n_samples must be positive, got {n_samples}")

    if len(indices) == 0:
        raise ValueError("indices array cannot be empty")

    n_indices = len(indices)

    if sample_length is None:
        # Default to 10% of data length, minimum 1
        sample_length = max(1, n_indices // 10)
    elif sample_length <= 0:
        raise ValueError(f"sample_length must be positive, got {sample_length}")

    # Set random seed
    if random_state is None:
        random_state = np.random.randint(0, 2**31 - 1)

    # Use Numba-optimized function
    return block_bootstrap_numba(indices, n_samples, sample_length, random_state)


def stratified_sample_time_series(
    data: pd.DataFrame | pl.DataFrame,
    stratify_column: str,
    sample_frac: float = 0.5,
    time_column: str | None = None,
    preserve_order: bool = True,
    random_state: int | None = None,
) -> pd.DataFrame | pl.DataFrame:
    """Stratified sampling that preserves time series properties.

    Parameters
    ----------
    data : pd.DataFrame or pl.DataFrame
        Input data to sample from
    stratify_column : str
        Column to use for stratification
    sample_frac : float
        Fraction of data to sample from each stratum
    time_column : str, optional
        Time column for maintaining temporal order
    preserve_order : bool
        Whether to preserve temporal ordering within strata
    random_state : int, optional
        Random seed for reproducibility

    Returns:
    -------
    pd.DataFrame or pl.DataFrame
        Stratified sample preserving input type

    Examples:
    --------
    >>> df = pd.DataFrame({
    ...     'time': pd.date_range('2020-01-01', periods=1000),
    ...     'label': np.random.choice([-1, 0, 1], 1000),
    ...     'feature': np.random.randn(1000)
    ... })
    >>> sampled = stratified_sample_time_series(
    ...     df, stratify_column='label', sample_frac=0.3
    ... )
    """
    rng = np.random.RandomState(random_state)

    if isinstance(data, pl.DataFrame):
        # Polars implementation
        unique_values = data[stratify_column].unique().to_list()
        sampled_dfs = []

        for value in unique_values:
            stratum_df = data.filter(pl.col(stratify_column) == value)
            n_stratum = len(stratum_df)
            n_sample = int(n_stratum * sample_frac)

            if n_sample > 0:
                if preserve_order and time_column:
                    # Sample by time blocks to preserve structure
                    block_size = max(1, n_stratum // (n_sample // 10 + 1))
                    sampled_indices: list[int] = []

                    for i in range(0, n_stratum - block_size + 1, block_size):
                        if rng.random() < sample_frac:
                            sampled_indices.extend(
                                range(i, min(i + block_size, n_stratum)),
                            )

                    sampled_stratum = stratum_df[sampled_indices[:n_sample]]
                else:
                    # Random sampling
                    sample_indices = rng.choice(n_stratum, n_sample, replace=False)
                    sampled_stratum = stratum_df[sorted(sample_indices)]

                sampled_dfs.append(sampled_stratum)

        result = pl.concat(sampled_dfs)

        if time_column and preserve_order:
            result = result.sort(time_column)

    elif isinstance(data, pd.DataFrame):
        # Pandas implementation - explicit casts to ensure proper type narrowing
        # Use completely separate variable names from Polars branch to avoid mypy redefinition errors
        data_pandas: pd.DataFrame = data
        unique_vals_pd = data_pandas[stratify_column].unique()  # Returns ndarray
        collected_dfs: list[pd.DataFrame] = []

        for val in unique_vals_pd:
            stratum: pd.DataFrame = data_pandas[data_pandas[stratify_column] == val]
            n_rows = len(stratum)
            n_to_sample = int(n_rows * sample_frac)

            if n_to_sample > 0:
                selected: pd.DataFrame
                if preserve_order:
                    # Sample contiguous blocks
                    blk_size = max(1, n_rows // (n_to_sample // 10 + 1))
                    idx_list: list[Any] = []

                    for j in range(0, n_rows - blk_size + 1, blk_size):
                        if rng.random() < sample_frac:
                            idx_list.extend(
                                stratum.index[j : j + blk_size].tolist(),
                            )

                    selected = stratum.loc[idx_list[:n_to_sample]]
                else:
                    selected = stratum.sample(
                        n=n_to_sample,
                        random_state=random_state,
                    )

                collected_dfs.append(selected)

        result_pd = pd.concat(collected_dfs)

        if time_column and preserve_order:
            result_pd = result_pd.sort_values(time_column)

        return result_pd
    else:
        raise TypeError(f"data must be pd.DataFrame or pl.DataFrame, got {type(data)}")

    return result


def sample_weights_by_importance(
    returns: "NDArray[Any]",
    method: str = "return_magnitude",
    decay_factor: float = 0.94,
) -> "NDArray[Any]":
    """Calculate sampling weights based on importance criteria.

    Parameters
    ----------
    returns : np.ndarray
        Array of returns or outcomes
    method : str
        Method for calculating importance weights:
        - 'return_magnitude': Weight by absolute return size
        - 'recency': Exponential decay weights
        - 'volatility': Weight by local volatility
    decay_factor : float
        Decay factor for recency weighting

    Returns:
    -------
    np.ndarray
        Sampling weights (sum to 1)

    Raises:
    ------
    ValueError
        If returns is empty, method is unknown, or decay_factor is invalid

    Examples:
    --------
    >>> returns = np.random.randn(100) * 0.02
    >>> weights = sample_weights_by_importance(returns, method='recency')
    >>> weights.sum()
    1.0
    """
    # Input validation
    if len(returns) == 0:
        raise ValueError("returns array cannot be empty")

    if not 0 < decay_factor < 1:
        raise ValueError(f"decay_factor must be in (0, 1), got {decay_factor}")

    valid_methods = ["return_magnitude", "recency", "volatility"]
    if method not in valid_methods:
        raise ValueError(f"method must be one of {valid_methods}, got '{method}'")

    n_samples = len(returns)

    if method == "return_magnitude":
        # Weight by absolute return magnitude
        weights = np.abs(returns)

        # Handle case where all returns are zero
        if np.sum(weights) == 0:
            weights = np.ones(n_samples)  # Equal weights if all returns are zero

        weights = weights / weights.sum()

    elif method == "recency":
        # Exponential decay weights (more recent = higher weight)
        time_weights = decay_factor ** np.arange(n_samples - 1, -1, -1)
        weights = time_weights / time_weights.sum()

    elif method == "volatility":
        # Weight by local volatility (20-period rolling std)
        if n_samples < 2:
            # Can't calculate volatility with less than 2 samples
            weights = np.ones(n_samples) / n_samples
        else:
            volatility: NDArray[Any] = (
                pd.Series(returns).rolling(20, min_periods=1).std().to_numpy()
            )

            # Handle case where volatility is all NaN or zero
            if np.all(np.isnan(volatility)) or float(np.nansum(volatility)) == 0:
                weights = np.ones(n_samples)  # Equal weights
            else:
                weights = volatility

            # Replace any remaining NaN values
            weights = np.nan_to_num(weights, nan=1.0)
            weights = weights / weights.sum()

    # Final safety check - ensure weights are valid probabilities
    weights = np.nan_to_num(weights, nan=1 / n_samples, posinf=1 / n_samples, neginf=0)

    # Ensure weights sum to 1
    weights_sum = weights.sum()
    if weights_sum <= 0:
        # Fallback to equal weights
        weights = np.ones(n_samples) / n_samples
    else:
        weights = weights / weights_sum

    return weights


def balanced_subsample(
    X: "NDArray[Any]",
    y: "NDArray[Any]",
    minority_weight: float = 1.0,
    method: str = "undersample",
    random_state: int | None = None,
) -> tuple["NDArray[Any]", "NDArray[Any]"]:
    """Balance classes through strategic subsampling.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix
    y : np.ndarray
        Labels (assumed to be -1, 0, 1 for financial ML)
    minority_weight : float
        Weight given to minority class preservation
    method : str
        Balancing method:
        - 'undersample': Undersample majority class
        - 'hybrid': Combination of under and oversampling
    random_state : int, optional
        Random seed

    Returns:
    -------
    X_balanced : np.ndarray
        Balanced feature matrix
    y_balanced : np.ndarray
        Balanced labels
    """
    rng = np.random.RandomState(random_state)

    # Get class counts
    unique_labels, counts = np.unique(y, return_counts=True)
    min_count = counts.min()
    counts.max()

    if method == "undersample":
        # Undersample to match minority class
        balanced_indices: list[int] = []

        for label in unique_labels:
            label_indices = np.where(y == label)[0]

            if len(label_indices) > min_count:
                # Undersample this class
                if label == 0:  # Neutral class in financial ML
                    # More aggressive undersampling for neutral class
                    n_sample = int(min_count * (2 - minority_weight))
                else:
                    n_sample = min_count

                sampled = rng.choice(label_indices, n_sample, replace=False)
            else:
                # Keep all minority samples
                sampled = label_indices

            balanced_indices.extend(sampled)

    elif method == "hybrid":
        # Combination approach
        balanced_indices = []
        target_count = int(min_count * (1 + minority_weight))

        for label in unique_labels:
            label_indices = np.where(y == label)[0]

            if len(label_indices) > target_count:
                # Undersample
                sampled = rng.choice(label_indices, target_count, replace=False)
            elif len(label_indices) < target_count:
                # Oversample with replacement
                sampled = rng.choice(label_indices, target_count, replace=True)
            else:
                sampled = label_indices

            balanced_indices.extend(sampled)

    else:
        raise ValueError(f"Unknown method: {method}")

    # Shuffle the indices
    balanced_arr: NDArray[np.intp] = np.array(balanced_indices, dtype=np.intp)
    rng.shuffle(balanced_arr)

    return X[balanced_arr], y[balanced_arr]


def event_based_sample(
    data: pd.DataFrame | pl.DataFrame,
    event_column: str,
    n_samples: int | None = None,
    sample_frac: float | None = None,
    min_event_spacing: int | None = None,
    random_state: int | None = None,
) -> pd.DataFrame | pl.DataFrame:
    """Sample based on events ensuring minimum spacing.

    This is useful for event-driven strategies where you want to
    sample events (like price movements) with minimum time between them.

    Parameters
    ----------
    data : pd.DataFrame or pl.DataFrame
        Input data
    event_column : str
        Column indicating events (boolean or binary)
    n_samples : int, optional
        Number of events to sample
    sample_frac : float, optional
        Fraction of events to sample
    min_event_spacing : int, optional
        Minimum spacing between sampled events
    random_state : int, optional
        Random seed

    Returns:
    -------
    pd.DataFrame or pl.DataFrame
        Sampled data containing selected events
    """
    if n_samples is None and sample_frac is None:
        raise ValueError("Either n_samples or sample_frac must be specified")

    rng = np.random.RandomState(random_state)

    if isinstance(data, pl.DataFrame):
        # Get event indices
        event_mask_pl = data[event_column].cast(bool)
        event_indices = np.where(event_mask_pl.to_numpy())[0]
    elif isinstance(data, pd.DataFrame):
        # Pandas - explicit isinstance for type narrowing
        event_mask_pd = data[event_column].astype(bool)
        event_indices = np.where(event_mask_pd.to_numpy())[0]
    else:
        raise TypeError(f"data must be pd.DataFrame or pl.DataFrame, got {type(data)}")

    if n_samples is None:
        if sample_frac is None:
            raise ValueError("Either n_samples or sample_frac must be provided")
        n_samples = int(len(event_indices) * sample_frac)

    # Sample events with spacing constraint
    sampled_events: list[int] = []
    available_indices = list(event_indices)

    while len(sampled_events) < n_samples and available_indices:
        # Sample an event
        idx = rng.choice(len(available_indices))
        event_idx = available_indices[idx]
        sampled_events.append(event_idx)

        # Remove nearby events from available pool
        if min_event_spacing is not None:
            available_indices = [
                i for i in available_indices if abs(i - event_idx) > min_event_spacing
            ]
        else:
            available_indices.pop(idx)

    # Return data at sampled event indices
    if isinstance(data, pl.DataFrame):
        return data[sorted(sampled_events)]
    return data.iloc[sorted(sampled_events)]
