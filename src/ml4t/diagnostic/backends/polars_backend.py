"""Polars-specific optimizations for DataFrame operations.

This module provides optimized implementations of common operations
when working with Polars DataFrames, leveraging Polars' lazy evaluation
and columnar operations for improved performance.
"""

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl

if TYPE_CHECKING:
    from numpy.typing import NDArray


class PolarsBackend:
    """Optimized operations for Polars DataFrames.

    This backend provides performance-optimized implementations
    of common operations used throughout ml4t-diagnostic when working with
    Polars DataFrames. Includes memory-efficient streaming methods
    for handling large datasets (10M+ samples) without memory issues.

    Key Features:
    - Vectorized rolling correlations
    - Memory-efficient streaming for large datasets
    - Adaptive chunk sizing based on available memory
    - Multi-horizon Information Coefficient calculations
    """

    @staticmethod
    def fast_rolling_correlation(
        x: pl.Series,
        y: pl.Series,
        window: int,
        min_periods: int | None = None,
    ) -> pl.Series:
        """Compute rolling correlation efficiently.

        Parameters
        ----------
        x : pl.Series
            First series
        y : pl.Series
            Second series
        window : int
            Rolling window size
        min_periods : int, optional
            Minimum number of observations required

        Returns:
        -------
        pl.Series
            Rolling correlation values
        """
        if min_periods is None:
            min_periods = window

        # Use Polars' native rolling correlation
        df = pl.DataFrame({"x": x, "y": y})

        # Compute rolling stats needed for correlation
        rolling_df = df.select(
            [
                pl.col("x").rolling_mean(window, min_samples=min_periods).alias("x_mean"),
                pl.col("y").rolling_mean(window, min_samples=min_periods).alias("y_mean"),
                (pl.col("x") * pl.col("y"))
                .rolling_mean(window, min_samples=min_periods)
                .alias("xy_mean"),
                (pl.col("x") ** 2).rolling_mean(window, min_samples=min_periods).alias("x2_mean"),
                (pl.col("y") ** 2).rolling_mean(window, min_samples=min_periods).alias("y2_mean"),
            ],
        )

        # Calculate correlation from components
        result = rolling_df.select(
            [
                (
                    (pl.col("xy_mean") - pl.col("x_mean") * pl.col("y_mean"))
                    / (
                        (
                            (pl.col("x2_mean") - pl.col("x_mean") ** 2)
                            * (pl.col("y2_mean") - pl.col("y_mean") ** 2)
                        )
                        ** 0.5
                    )
                ).alias("correlation"),
            ],
        )

        return result["correlation"]

    @staticmethod
    def fast_rolling_spearman_correlation(
        x: pl.Series,
        y: pl.Series,
        window: int,
        min_periods: int | None = None,
    ) -> pl.Series:
        """Compute rolling Spearman correlation using native Polars operations.

        This implementation calculates ranks within each rolling window to avoid
        lookahead bias, ensuring that the rank at time T only uses data up to time T.

        Parameters
        ----------
        x : pl.Series
            First series
        y : pl.Series
            Second series
        window : int
            Rolling window size
        min_periods : int, optional
            Minimum number of observations required

        Returns:
        -------
        pl.Series
            Rolling Spearman correlation values
        """
        if min_periods is None:
            min_periods = max(2, window // 2)

        # Import scipy for rank calculation
        import numpy as np
        from scipy.stats import rankdata

        # Convert to numpy for processing
        x_values = x.to_numpy()
        y_values = y.to_numpy()
        n = len(x_values)

        # Initialize result array
        result = np.full(n, np.nan)

        # Calculate rolling Spearman correlation
        for i in range(n):
            # Define window boundaries
            start_idx = max(0, i - window + 1)
            end_idx = i + 1

            # Extract window data
            x_window = x_values[start_idx:end_idx]
            y_window = y_values[start_idx:end_idx]

            # Check minimum periods
            if len(x_window) < min_periods:
                continue

            # Handle NaN values
            mask = ~(np.isnan(x_window) | np.isnan(y_window))
            if np.sum(mask) < min_periods:
                continue

            x_clean = x_window[mask]
            y_clean = y_window[mask]

            # Calculate ranks within window
            if len(x_clean) > 1:
                x_ranks = rankdata(x_clean, method="average")
                y_ranks = rankdata(y_clean, method="average")

                # Calculate correlation on ranks
                x_std = np.std(x_ranks, ddof=1)
                y_std = np.std(y_ranks, ddof=1)

                if x_std > 0 and y_std > 0:
                    # Pearson correlation on ranks = Spearman correlation
                    corr = np.corrcoef(x_ranks, y_ranks)[0, 1]
                    result[i] = corr

        return pl.Series(result)

    @staticmethod
    def fast_multi_horizon_ic(
        predictions: pl.Series,
        returns_matrix: pl.DataFrame,
        window: int,
        min_periods: int | None = None,
    ) -> pl.DataFrame:
        """Calculate rolling IC for multiple return horizons efficiently.

        This is a specialized function for IC heatmap calculations that processes
        multiple horizons in parallel. Ranks are calculated within each rolling
        window to avoid lookahead bias, ensuring that the rank at time T only
        uses data up to time T.

        Parameters
        ----------
        predictions : pl.Series
            Model predictions
        returns_matrix : pl.DataFrame
            Returns for different horizons (columns = horizons)
        window : int
            Rolling window size
        min_periods : int, optional
            Minimum periods required

        Returns:
        -------
        pl.DataFrame
            DataFrame with rolling IC for each horizon
        """
        if min_periods is None:
            min_periods = max(2, window // 2)

        import numpy as np
        from scipy.stats import rankdata

        pred_values = predictions.to_numpy()
        n = len(pred_values)

        # Initialize result matrix
        result_data = {}

        for col in returns_matrix.columns:
            ret_values = returns_matrix[col].to_numpy()
            ic_result = np.full(n, np.nan)

            # Calculate IC for each window position
            for i in range(n):
                start_idx = max(0, i - window + 1)
                end_idx = i + 1

                pred_window = pred_values[start_idx:end_idx]
                ret_window = ret_values[start_idx:end_idx]

                if len(pred_window) < min_periods:
                    continue

                # Remove NaN values
                mask = ~(np.isnan(pred_window) | np.isnan(ret_window))
                if np.sum(mask) < min_periods:
                    continue

                pred_clean = pred_window[mask]
                ret_clean = ret_window[mask]

                if len(pred_clean) > 1:
                    # Calculate ranks within this window only
                    pred_ranks = rankdata(pred_clean, method="average")
                    ret_ranks = rankdata(ret_clean, method="average")

                    # Check for constant values (all ranks identical)
                    pred_std = np.std(pred_ranks, ddof=1)
                    ret_std = np.std(ret_ranks, ddof=1)

                    if pred_std > 0 and ret_std > 0:
                        # Compute Spearman correlation (Pearson on ranks)
                        corr = np.corrcoef(pred_ranks, ret_ranks)[0, 1]
                        ic_result[i] = corr

            result_data[f"ic_{col}"] = ic_result

        # Convert to Polars DataFrame
        return pl.DataFrame(result_data)

    @staticmethod
    def _rolling_correlation_expr(
        x_col: str,
        y_col: str,
        window: int,
        min_periods: int,
    ) -> pl.Expr:
        """Create a Polars expression for rolling correlation between two columns."""
        # Rolling means
        x_mean = pl.col(x_col).rolling_mean(window, min_samples=min_periods)
        y_mean = pl.col(y_col).rolling_mean(window, min_samples=min_periods)

        # Rolling products and squares
        xy_mean = (pl.col(x_col) * pl.col(y_col)).rolling_mean(
            window,
            min_samples=min_periods,
        )
        x2_mean = (pl.col(x_col) ** 2).rolling_mean(window, min_samples=min_periods)
        y2_mean = (pl.col(y_col) ** 2).rolling_mean(window, min_samples=min_periods)

        # Compute correlation using the formula: corr = cov(x,y) / (std(x) * std(y))
        # where cov(x,y) = E[xy] - E[x]E[y] and var(x) = E[x²] - E[x]²
        numerator = xy_mean - (x_mean * y_mean)
        denominator = ((x2_mean - x_mean**2) * (y2_mean - y_mean**2)) ** 0.5

        # Handle division by zero
        correlation = pl.when(denominator > 1e-10).then(numerator / denominator).otherwise(0.0)

        return correlation

    @staticmethod
    def fast_quantile_assignment(
        data: pl.DataFrame,
        column: str,
        n_quantiles: int,
        by_group: str | None = None,
    ) -> pl.DataFrame:
        """Assign quantile labels efficiently.

        Parameters
        ----------
        data : pl.DataFrame
            Input data
        column : str
            Column to compute quantiles on
        n_quantiles : int
            Number of quantiles
        by_group : str, optional
            Column to group by before quantile assignment

        Returns:
        -------
        pl.DataFrame
            Data with quantile labels added
        """
        if by_group is not None:
            # Group-wise quantile assignment
            result = data.with_columns(
                pl.col(column)
                .qcut(n_quantiles, labels=[str(i) for i in range(1, n_quantiles + 1)])
                .over(by_group)
                .alias(f"{column}_quantile"),
            )
        else:
            # Global quantile assignment
            result = data.with_columns(
                pl.col(column)
                .qcut(n_quantiles, labels=[str(i) for i in range(1, n_quantiles + 1)])
                .alias(f"{column}_quantile"),
            )

        return result

    @staticmethod
    def fast_time_aware_split(
        data: pl.DataFrame,
        time_column: str,
        test_start: Any,
        test_end: Any,
        buffer_before: int | None = None,
        buffer_after: int | None = None,
    ) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        """Split data into train/test/buffer sets efficiently.

        Parameters
        ----------
        data : pl.DataFrame
            Input data with time column
        time_column : str
            Name of time column
        test_start : Any
            Test period start time
        test_end : Any
            Test period end time
        buffer_before : int, optional
            Purge buffer before test
        buffer_after : int, optional
            Embargo buffer after test

        Returns:
        -------
        train_df : pl.DataFrame
            Training data
        test_df : pl.DataFrame
            Test data
        buffer_df : pl.DataFrame
            Buffer zone data
        """
        # Create efficient filters
        test_mask = (pl.col(time_column) >= test_start) & (pl.col(time_column) < test_end)

        # Apply test filter
        test_df = data.filter(test_mask)

        # Create buffer masks if needed
        if buffer_before is not None:
            buffer_start = test_start - buffer_before
            before_buffer_mask = (pl.col(time_column) >= buffer_start) & (
                pl.col(time_column) < test_start
            )
        else:
            before_buffer_mask = pl.lit(False)

        if buffer_after is not None:
            buffer_end = test_end + buffer_after
            after_buffer_mask = (pl.col(time_column) >= test_end) & (
                pl.col(time_column) < buffer_end
            )
        else:
            after_buffer_mask = pl.lit(False)

        # Combine buffer masks
        buffer_mask = before_buffer_mask | after_buffer_mask
        buffer_df = data.filter(buffer_mask)

        # Train is everything not in test or buffer
        train_mask = ~(test_mask | buffer_mask)
        train_df = data.filter(train_mask)

        return train_df, test_df, buffer_df

    @staticmethod
    def fast_group_statistics(
        data: pl.DataFrame,
        group_column: str,
        value_column: str,
        statistics: list[str],
    ) -> pl.DataFrame:
        """Compute group statistics efficiently.

        Parameters
        ----------
        data : pl.DataFrame
            Input data
        group_column : str
            Column to group by
        value_column : str
            Column to compute statistics on
        statistics : list[str]
            List of statistics to compute

        Returns:
        -------
        pl.DataFrame
            Group statistics
        """
        # Map statistic names to Polars expressions
        stat_exprs = []

        for stat in statistics:
            if stat == "mean":
                stat_exprs.append(
                    pl.col(value_column).mean().alias(f"{value_column}_mean"),
                )
            elif stat == "std":
                stat_exprs.append(
                    pl.col(value_column).std().alias(f"{value_column}_std"),
                )
            elif stat == "min":
                stat_exprs.append(
                    pl.col(value_column).min().alias(f"{value_column}_min"),
                )
            elif stat == "max":
                stat_exprs.append(
                    pl.col(value_column).max().alias(f"{value_column}_max"),
                )
            elif stat == "count":
                stat_exprs.append(
                    pl.col(value_column).count().alias(f"{value_column}_count"),
                )
            elif stat == "sum":
                stat_exprs.append(
                    pl.col(value_column).sum().alias(f"{value_column}_sum"),
                )
            elif stat == "median":
                stat_exprs.append(
                    pl.col(value_column).median().alias(f"{value_column}_median"),
                )
            else:
                raise ValueError(f"Unknown statistic: {stat}")

        # Compute all statistics in one pass
        result = data.group_by(group_column).agg(stat_exprs)

        return result

    @staticmethod
    def fast_expanding_window(
        data: pl.DataFrame,
        columns: list[str],
        operation: str = "mean",
        min_periods: int = 1,
    ) -> pl.DataFrame:
        """Compute expanding window statistics efficiently.

        Parameters
        ----------
        data : pl.DataFrame
            Input data
        columns : list[str]
            Columns to compute expanding statistics on
        operation : str
            Operation to apply (mean, std, sum, etc.)
        min_periods : int
            Minimum number of observations required

        Returns:
        -------
        pl.DataFrame
            Data with expanding statistics added
        """
        result = data

        for col in columns:
            if operation == "mean":
                expr = pl.col(col).cum_sum() / pl.int_range(1, pl.len() + 1)
            elif operation == "std":
                # O(n) expanding standard deviation using Welford's online algorithm
                # Var(X) = E[X²] - E[X]² → std = sqrt(Var * n/(n-1)) with Bessel correction
                n_expr = pl.int_range(1, pl.len() + 1).cast(pl.Float64)
                cum_sum = pl.col(col).cum_sum()
                cum_sum_sq = (pl.col(col) ** 2).cum_sum()

                # Expanding mean of squares and square of mean
                mean_of_sq = cum_sum_sq / n_expr
                mean_sq = (cum_sum / n_expr) ** 2

                # Population variance (before Bessel correction)
                variance = mean_of_sq - mean_sq

                # Apply Bessel correction: sample_var = pop_var * n / (n-1)
                # Handle n=1 case (variance undefined for single observation)
                # and min_periods requirement
                expr = (
                    pl.when(n_expr >= max(min_periods, 2))
                    .then((variance * n_expr / (n_expr - 1)).sqrt())
                    .otherwise(None)
                )
            elif operation == "sum":
                expr = pl.col(col).cum_sum()
            elif operation == "min":
                expr = pl.col(col).cum_min()
            elif operation == "max":
                expr = pl.col(col).cum_max()
            else:
                raise ValueError(f"Unknown operation: {operation}")

            result = result.with_columns(expr.alias(f"{col}_expanding_{operation}"))

        return result

    @staticmethod
    def to_numpy_batch(
        data: pl.DataFrame,
        columns: list[str] | None = None,
        batch_size: int = 10000,
    ) -> "NDArray[Any]":
        """Convert DataFrame to numpy array in batches for memory efficiency.

        Parameters
        ----------
        data : pl.DataFrame
            Input DataFrame
        columns : list[str], optional
            Columns to convert (all if None)
        batch_size : int
            Batch size for conversion

        Returns:
        -------
        np.ndarray
            Numpy array
        """
        if columns is not None:
            data = data.select(columns)

        # For small data, convert directly
        if len(data) <= batch_size:
            return data.to_numpy()

        # For large data, convert in batches
        n_rows = len(data)
        n_cols = data.shape[1]
        result = np.empty((n_rows, n_cols), dtype=np.float64)

        for i in range(0, n_rows, batch_size):
            end_idx = min(i + batch_size, n_rows)
            result[i:end_idx] = data[i:end_idx].to_numpy()

        return result

    @staticmethod
    def fast_rolling_correlation_streaming(
        x: pl.Series,
        y: pl.Series,
        window: int,
        min_periods: int | None = None,
        chunk_size: int = 50000,
    ) -> pl.Series:
        """Compute rolling correlation for large datasets using streaming.

        This method processes data in chunks to manage memory usage for
        very large datasets while maintaining accuracy through proper
        overlap handling.

        Parameters
        ----------
        x : pl.Series
            First series
        y : pl.Series
            Second series
        window : int
            Rolling window size
        min_periods : int, optional
            Minimum number of observations required
        chunk_size : int, default 50000
            Size of chunks to process. Larger chunks use more memory
            but may be more efficient.

        Returns
        -------
        pl.Series
            Rolling correlation values

        Notes
        -----
        This function is designed for datasets larger than 100k samples.
        For smaller datasets, use fast_rolling_correlation() directly
        as it will be more efficient.

        Examples
        --------
        >>> x = pl.Series("x", range(200000))
        >>> y = pl.Series("y", np.random.randn(200000))
        >>> corr = PolarsBackend.fast_rolling_correlation_streaming(x, y, 100)
        """
        if min_periods is None:
            min_periods = window

        n_samples = len(x)

        # For small datasets, use the standard method
        if n_samples <= chunk_size:
            return PolarsBackend.fast_rolling_correlation(x, y, window, min_periods)

        # For very large datasets, use streaming approach
        results = []
        overlap = window - 1  # Overlap needed to maintain continuity

        for start in range(0, n_samples, chunk_size - overlap):
            end = min(start + chunk_size, n_samples)

            # Extract chunk with overlap
            x_chunk = x[start:end]
            y_chunk = y[start:end]

            # Process chunk
            chunk_result = PolarsBackend.fast_rolling_correlation(
                x_chunk,
                y_chunk,
                window,
                min_periods,
            )

            # Handle overlap: remove duplicate results from previous chunks
            if start > 0:
                # Remove the overlapping window-1 results
                chunk_result = chunk_result[overlap:]

            results.append(chunk_result)

        # Concatenate all results - results list contains Series, so concat returns Series
        concatenated = pl.concat(results)
        assert isinstance(concatenated, pl.Series), "Expected Series from concatenating Series"
        return concatenated

    @staticmethod
    def fast_multi_horizon_ic_streaming(
        predictions: pl.Series,
        returns_matrix: pl.DataFrame,
        window: int,
        min_periods: int | None = None,
        chunk_size: int = 50000,
    ) -> pl.DataFrame:
        """Calculate rolling IC for multiple horizons using streaming for large datasets.

        This is a memory-efficient version of fast_multi_horizon_ic that processes
        data in chunks to handle very large datasets without memory issues.

        Parameters
        ----------
        predictions : pl.Series
            Model predictions
        returns_matrix : pl.DataFrame
            Returns for different horizons (columns = horizons)
        window : int
            Rolling window size
        min_periods : int, optional
            Minimum periods required
        chunk_size : int, default 50000
            Size of chunks to process

        Returns
        -------
        pl.DataFrame
            DataFrame with rolling IC for each horizon

        Examples
        --------
        >>> predictions = pl.Series("pred", np.random.randn(200000))
        >>> returns = pl.DataFrame({
        ...     "1d": np.random.randn(200000),
        ...     "5d": np.random.randn(200000),
        ...     "20d": np.random.randn(200000)
        ... })
        >>> ic_matrix = PolarsBackend.fast_multi_horizon_ic_streaming(
        ...     predictions, returns, window=100
        ... )
        """
        if min_periods is None:
            min_periods = max(2, window // 2)

        n_samples = len(predictions)

        # For small datasets, use the standard method
        if n_samples <= chunk_size:
            return PolarsBackend.fast_multi_horizon_ic(
                predictions,
                returns_matrix,
                window,
                min_periods,
            )

        # For large datasets, use streaming approach
        results = []
        overlap = window - 1

        for start in range(0, n_samples, chunk_size - overlap):
            end = min(start + chunk_size, n_samples)

            # Extract chunks
            pred_chunk = predictions[start:end]
            returns_chunk = returns_matrix[start:end]

            # Process chunk
            chunk_result = PolarsBackend.fast_multi_horizon_ic(
                pred_chunk,
                returns_chunk,
                window,
                min_periods,
            )

            # Handle overlap
            if start > 0:
                chunk_result = chunk_result[overlap:]

            results.append(chunk_result)

        # Concatenate all results
        return pl.concat(results)

    @staticmethod
    def estimate_memory_usage(
        n_samples: int,
        n_features: int,
        data_type: str = "float64",
    ) -> dict[str, float]:
        """Estimate memory usage for different operations.

        Parameters
        ----------
        n_samples : int
            Number of samples
        n_features : int
            Number of features
        data_type : str, default "float64"
            Data type (float64, float32, int64, etc.)

        Returns
        -------
        dict
            Memory usage estimates in MB
        """
        # Bytes per element
        type_sizes = {"float64": 8, "float32": 4, "int64": 8, "int32": 4, "bool": 1}

        bytes_per_element = type_sizes.get(data_type, 8)

        # Basic DataFrame memory
        base_memory_mb = (n_samples * n_features * bytes_per_element) / (1024 * 1024)

        # Rolling operations typically need 2-3x memory for intermediate calculations
        rolling_memory_mb = base_memory_mb * 2.5

        # Multi-horizon IC needs additional memory for ranks and correlations
        ic_memory_mb = base_memory_mb * 3.0

        return {
            "base_dataframe_mb": base_memory_mb,
            "rolling_operations_mb": rolling_memory_mb,
            "multi_horizon_ic_mb": ic_memory_mb,
            "recommended_chunk_size": max(
                10000,
                min(100000, int(500 * 1024 * 1024 / (n_features * bytes_per_element))),
            ),
        }

    @staticmethod
    def adaptive_chunk_size(
        total_samples: int,
        n_features: int = 1,
        target_memory_mb: int = 500,
        min_chunk_size: int = 10000,
        max_chunk_size: int = 100000,
    ) -> int:
        """Calculate optimal chunk size based on available memory.

        Parameters
        ----------
        total_samples : int
            Total number of samples in dataset
        n_features : int, default 1
            Number of features (affects memory per sample)
        target_memory_mb : int, default 500
            Target memory usage in MB
        min_chunk_size : int, default 10000
            Minimum chunk size
        max_chunk_size : int, default 100000
            Maximum chunk size

        Returns
        -------
        int
            Optimal chunk size
        """
        # Estimate memory per sample (assuming float64)
        memory_per_sample = n_features * 8 * 2.5  # 2.5x factor for processing overhead

        # Calculate chunk size to fit in target memory
        target_chunk_size = int((target_memory_mb * 1024 * 1024) / memory_per_sample)

        # Apply bounds
        chunk_size = max(min_chunk_size, min(max_chunk_size, target_chunk_size))

        # Don't chunk if dataset is small
        if total_samples <= max_chunk_size:
            return total_samples

        return chunk_size

    @staticmethod
    def memory_efficient_operation(
        data: pl.DataFrame,
        operation_func: Callable[..., pl.DataFrame],
        chunk_size: int | None = None,
        overlap: int = 0,
        **kwargs,
    ) -> pl.DataFrame:
        """Apply an operation to large DataFrame in memory-efficient chunks.

        This is a generic streaming framework that can be used for any
        operation that can be applied to DataFrame chunks.

        Parameters
        ----------
        data : pl.DataFrame
            Input DataFrame
        operation_func : callable
            Function to apply to each chunk
        chunk_size : int, optional
            Chunk size (auto-calculated if None)
        overlap : int, default 0
            Number of rows to overlap between chunks
        **kwargs
            Additional arguments passed to operation_func

        Returns
        -------
        pl.DataFrame
            Result of applying operation to entire DataFrame

        Examples
        --------
        >>> def rolling_mean_op(chunk_df, window=10):
        ...     return chunk_df.select([
        ...         pl.col("value").rolling_mean(window).alias("rolling_mean")
        ...     ])
        >>>
        >>> result = PolarsBackend.memory_efficient_operation(
        ...     large_df, rolling_mean_op, overlap=9, window=10
        ... )
        """
        n_samples = len(data)

        # Auto-calculate chunk size if not provided
        if chunk_size is None:
            chunk_size = PolarsBackend.adaptive_chunk_size(n_samples, data.shape[1])

        # For small data, process directly
        if n_samples <= chunk_size:
            return operation_func(data, **kwargs)

        results = []

        for start in range(0, n_samples, chunk_size - overlap):
            end = min(start + chunk_size, n_samples)

            # Extract chunk
            chunk = data[start:end]

            # Apply operation
            chunk_result = operation_func(chunk, **kwargs)

            # Handle overlap
            if start > 0 and overlap > 0:
                chunk_result = chunk_result[overlap:]

            results.append(chunk_result)

        return pl.concat(results)
