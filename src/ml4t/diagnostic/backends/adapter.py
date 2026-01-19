"""Adapter layer for seamless DataFrame conversion between Polars and Pandas.

This module provides utilities to convert between different DataFrame representations.
The internal implementation uses Polars for performance, but the adapter ensures
compatibility with Pandas-based workflows.

Note: MultiIndex preservation/restoration has been removed as it was unused.
If you need MultiIndex support, use pandas directly or convert after receiving
the Polars DataFrame.
"""

from typing import TYPE_CHECKING, Any, Union

import numpy as np
import pandas as pd
import polars as pl

if TYPE_CHECKING:
    from numpy.typing import NDArray


class DataFrameAdapter:
    """Adapter for converting between Polars and Pandas DataFrames.

    This class handles conversions between different DataFrame representations.
    It's designed to be used internally by ml4t-diagnostic to ensure consistent
    behavior regardless of the input format.

    Methods
    -------
    to_polars(data, columns=None)
        Convert input data to Polars DataFrame.
    to_numpy(data)
        Convert any supported data type to numpy array.
    get_shape(data)
        Get the shape of the data regardless of type.
    """

    @staticmethod
    def to_polars(
        data: Union[pl.DataFrame, pd.DataFrame, "NDArray[Any]"],
        columns: list[str] | None = None,
    ) -> tuple[pl.DataFrame, None]:
        """Convert input data to Polars DataFrame.

        Parameters
        ----------
        data : polars.DataFrame, pandas.DataFrame, or numpy.ndarray
            The input data to convert.
        columns : list of str, optional
            Column names to use if data is a numpy array.

        Returns
        -------
        df : polars.DataFrame
            The data as a Polars DataFrame.
        index : None
            Always None. Kept for backward compatibility with existing code
            that unpacks the tuple return value.

        Raises
        ------
        TypeError
            If the input type is not supported.
        ValueError
            If columns are needed but not provided.

        Examples
        --------
        >>> import pandas as pd
        >>> import polars as pl
        >>> df_pd = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        >>> df_pl, _ = DataFrameAdapter.to_polars(df_pd)
        >>> isinstance(df_pl, pl.DataFrame)
        True
        """
        if isinstance(data, pl.DataFrame):
            return data, None

        if isinstance(data, pd.DataFrame):
            # Reset index to columns if it's not a default RangeIndex
            index = data.index
            if isinstance(index, pd.MultiIndex):
                # For MultiIndex, reset to columns
                df_reset = data.reset_index(drop=False)
                return pl.from_pandas(df_reset), None
            elif not isinstance(index, pd.RangeIndex) or index.start != 0 or index.step != 1:
                # Custom index - reset to column
                df_reset = data.reset_index(drop=False)
                return pl.from_pandas(df_reset), None
            else:
                # Default RangeIndex
                return pl.from_pandas(data), None

        if isinstance(data, np.ndarray):
            if data.ndim == 1:
                # 1D array, treat as single column
                if columns is None:
                    columns = ["column_0"]
                elif len(columns) != 1:
                    raise ValueError(
                        f"1D array requires exactly 1 column name, got {len(columns)}",
                    )
                return pl.DataFrame({columns[0]: data}), None

            if data.ndim == 2:
                # 2D array
                if columns is None:
                    columns = [f"column_{i}" for i in range(data.shape[1])]
                elif len(columns) != data.shape[1]:
                    raise ValueError(
                        f"Number of columns ({len(columns)}) doesn't match array shape ({data.shape[1]})",
                    )
                return pl.DataFrame(data, schema=columns), None

            raise ValueError(f"Arrays must be 1D or 2D, got {data.ndim}D")

        raise TypeError(
            f"Data must be a Polars DataFrame, Pandas DataFrame, or numpy array. "
            f"Got {type(data).__name__}",
        )

    @staticmethod
    def to_numpy(
        data: Union[pl.DataFrame, pl.Series, pd.DataFrame, pd.Series, "NDArray[Any]"],
    ) -> "NDArray[Any]":
        """Convert any supported data type to numpy array.

        Parameters
        ----------
        data : polars.DataFrame/Series, pandas.DataFrame/Series, or numpy.ndarray
            The data to convert.

        Returns
        -------
        array : numpy.ndarray
            The data as a numpy array.

        Raises
        ------
        TypeError
            If the input type is not supported.

        Examples
        --------
        >>> import polars as pl
        >>> s = pl.Series([1, 2, 3])
        >>> arr = DataFrameAdapter.to_numpy(s)
        >>> arr.tolist()
        [1, 2, 3]
        """
        if isinstance(data, np.ndarray):
            return data
        if isinstance(data, pl.DataFrame | pl.Series | pd.DataFrame | pd.Series):
            return data.to_numpy()
        raise TypeError(f"Cannot convert {type(data).__name__} to numpy array")

    @staticmethod
    def get_shape(
        data: Union[pl.DataFrame, pd.DataFrame, "NDArray[Any]"],
    ) -> tuple[int, int]:
        """Get the shape of the data regardless of type.

        Parameters
        ----------
        data : polars.DataFrame, pandas.DataFrame, or numpy.ndarray
            The data to get the shape from.

        Returns
        -------
        shape : tuple of int
            (n_rows, n_cols) for 2D data, (n_rows, 1) for 1D data.

        Raises
        ------
        TypeError
            If the input type is not supported.

        Examples
        --------
        >>> import numpy as np
        >>> arr = np.array([1, 2, 3])
        >>> DataFrameAdapter.get_shape(arr)
        (3, 1)
        """
        if isinstance(data, pl.DataFrame | pd.DataFrame):
            return data.shape
        if isinstance(data, np.ndarray):
            if data.ndim == 1:
                return (data.shape[0], 1)
            return data.shape
        raise TypeError(f"Cannot get shape of {type(data).__name__}")
