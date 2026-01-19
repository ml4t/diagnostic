"""Base class for all time-series cross-validation splitters.

This module defines the abstract base class that all ml4t-diagnostic splitters inherit from,
ensuring compatibility with scikit-learn's cross-validation framework while adding
support for time-series specific features like purging and embargo.
"""

from abc import ABC, abstractmethod
from collections.abc import Generator
from typing import TYPE_CHECKING, Any, Union, cast

import numpy as np
import pandas as pd
import polars as pl

if TYPE_CHECKING:
    from numpy.typing import NDArray


class BaseSplitter(ABC):
    """Abstract base class for all ml4t-diagnostic time-series splitters.

    This class defines the interface that all splitters must implement to ensure
    compatibility with scikit-learn's model selection tools while providing
    additional functionality for financial time-series validation.

    All splitters should support purging (removing training data that could leak
    information into test data) and embargo (adding gaps between train and test
    sets to account for serial correlation).

    Session-Aware Splitting
    -----------------------
    Splitters can optionally align fold boundaries to trading session boundaries
    by setting ``align_to_sessions=True``. This requires the data to have a
    session column (default: 'session_date') that identifies trading sessions.

    Trading sessions are atomic units that should never be split across train/test
    folds. For intraday data (e.g., CME futures with Sunday 5pm - Friday 4pm sessions),
    this prevents subtle lookahead bias from mid-session splits.

    **Integration with qdata library:**

    The session column should be added using the ``qdata`` library's session
    assignment functionality::

        from qdata import DataManager

        manager = DataManager()
        df = manager.load(symbol="BTC", exchange="CME", calendar="CME_Globex_Crypto")
        # df now has 'session_date' column automatically assigned

    Or manually using SessionAssigner::

        from ml4t.data.sessions import SessionAssigner

        assigner = SessionAssigner.from_exchange('CME')
        df_with_sessions = assigner.assign_sessions(df)

    Then use with ml4t-diagnostic splitters::

        from ml4t.diagnostic.splitters import PurgedWalkForwardCV

        cv = PurgedWalkForwardCV(
            n_splits=5,
            align_to_sessions=True,  # Align folds to session boundaries
            session_col='session_date'
        )

        for train_idx, test_idx in cv.split(df_with_sessions):
            # Fold boundaries respect session boundaries
            pass
    """

    @abstractmethod
    def split(
        self,
        X: Union[pl.DataFrame, pd.DataFrame, "NDArray[Any]"],
        y: Union[pl.Series, pd.Series, "NDArray[Any]"] | None = None,
        groups: Union[pl.Series, pd.Series, "NDArray[Any]"] | None = None,
    ) -> Generator[tuple["NDArray[np.intp]", "NDArray[np.intp]"], None, None]:
        """Generate indices to split data into training and test sets.

        Parameters
        ----------
        X : polars.DataFrame, pandas.DataFrame, or numpy.ndarray
            Training data with shape (n_samples, n_features).

        y : polars.Series, pandas.Series, numpy.ndarray, or None, default=None
            Target variable with shape (n_samples,). Always ignored but kept
            for scikit-learn compatibility.

        groups : polars.Series, pandas.Series, numpy.ndarray, or None, default=None
            Group labels for samples, used for multi-asset splitting.
            Shape (n_samples,).

        Yields:
        ------
        train : numpy.ndarray
            The training set indices for that split.

        test : numpy.ndarray
            The testing set indices for that split.

        Notes:
        -----
        The indices returned are integer positions, not labels or timestamps.
        This ensures compatibility with numpy array indexing and scikit-learn.
        """

    def get_n_splits(
        self,
        X: Union[pl.DataFrame, pd.DataFrame, "NDArray[Any]"] | None = None,
        y: Union[pl.Series, pd.Series, "NDArray[Any]"] | None = None,
        groups: Union[pl.Series, pd.Series, "NDArray[Any]"] | None = None,
    ) -> int:
        """Return the number of splitting iterations in the cross-validator.

        Parameters
        ----------
        X : polars.DataFrame, pandas.DataFrame, numpy.ndarray, or None, default=None
            Training data. Some splitters may use properties of X to determine
            the number of splits.

        y : polars.Series, pandas.Series, numpy.ndarray, or None, default=None
            Always ignored, exists for compatibility.

        groups : polars.Series, pandas.Series, numpy.ndarray, or None, default=None
            Group labels. Some splitters may use this to determine splits.

        Returns:
        -------
        n_splits : int
            The number of splitting iterations.

        Notes:
        -----
        Most splitters can determine the number of splits from their parameters
        alone, but some (like GroupKFold variants) may need to inspect the data.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement get_n_splits()",
        )

    def _get_n_samples(
        self,
        X: Union[pl.DataFrame, pd.DataFrame, "NDArray[Any]"],
    ) -> int:
        """Get the number of samples in X regardless of type.

        Parameters
        ----------
        X : polars.DataFrame, pandas.DataFrame, or numpy.ndarray
            The data to get the sample count from.

        Returns:
        -------
        n_samples : int
            The number of samples (rows) in X.
        """
        if isinstance(X, pl.DataFrame):
            return X.height
        if isinstance(X, pl.LazyFrame):
            # LazyFrame doesn't have height, need to collect first
            return X.collect().height
        if isinstance(X, pd.DataFrame):
            return len(X)
        if isinstance(X, np.ndarray):
            return int(X.shape[0])
        raise TypeError(
            f"X must be a Polars DataFrame, Pandas DataFrame, or numpy array. Got {type(X).__name__}",
        )

    def _validate_data(
        self,
        X: Union[pl.DataFrame, pd.DataFrame, "NDArray[Any]"],
        y: Union[pl.Series, pd.Series, "NDArray[Any]"] | None = None,
        groups: Union[pl.Series, pd.Series, "NDArray[Any]"] | None = None,
    ) -> int:
        """Validate input data and return the number of samples.

        Parameters
        ----------
        X : polars.DataFrame, pandas.DataFrame, or numpy.ndarray
            Training data.

        y : polars.Series, pandas.Series, numpy.ndarray, or None
            Target variable.

        groups : polars.Series, pandas.Series, numpy.ndarray, or None
            Group labels.

        Returns:
        -------
        n_samples : int
            The number of samples in the data.

        Raises:
        ------
        ValueError
            If the input data has inconsistent lengths.
        TypeError
            If the input data types are not supported.
        """
        n_samples = self._get_n_samples(X)

        # Validate y if provided
        if y is not None:
            if isinstance(y, pl.Series | pd.Series):
                n_y = len(y)
            elif isinstance(y, np.ndarray):
                n_y = y.shape[0]
            else:
                raise TypeError(
                    f"y must be a Polars Series, Pandas Series, or numpy array. Got {type(y).__name__}",
                )

            if n_y != n_samples:
                raise ValueError(
                    f"X and y have inconsistent lengths: X has {n_samples} samples, y has {n_y} samples",
                )

        # Validate groups if provided
        if groups is not None:
            if isinstance(groups, pl.Series | pd.Series):
                n_groups = len(groups)
            elif isinstance(groups, np.ndarray):
                n_groups = groups.shape[0]
            else:
                raise TypeError(
                    f"groups must be a Polars Series, Pandas Series, or numpy array. Got {type(groups).__name__}",
                )

            if n_groups != n_samples:
                raise ValueError(
                    f"X and groups have inconsistent lengths: X has {n_samples} samples, groups has {n_groups} samples",
                )

        return n_samples

    def _validate_session_alignment(
        self,
        X: Union[pl.DataFrame, pd.DataFrame, "NDArray[Any]"],
        align_to_sessions: bool,
        session_col: str,
    ) -> None:
        """Validate that session column exists if session alignment is enabled.

        Parameters
        ----------
        X : polars.DataFrame, pandas.DataFrame, or numpy.ndarray
            Training data that may contain session column.

        align_to_sessions : bool
            Whether session alignment is requested.

        session_col : str
            Name of the session column to look for.

        Raises
        ------
        ValueError
            If align_to_sessions=True but session column is missing or X is not a DataFrame.

        Notes
        -----
        This method provides helpful error messages that guide users to the qdata library
        for session assignment if the required column is missing.
        """
        if not align_to_sessions:
            return  # Skip validation if not using sessions

        # Check that X is a DataFrame (sessions require column access)
        if not hasattr(X, "columns"):
            raise ValueError(
                f"align_to_sessions=True requires X to be a DataFrame "
                f"(Polars or Pandas), got {type(X).__name__}.\n"
                f"\n"
                f"Session alignment works with tabular data that has a session "
                f"identifier column. NumPy arrays do not support column names."
            )

        # Check for session column
        columns = list(cast(Any, X.columns))
        if session_col not in columns:
            raise ValueError(
                f"align_to_sessions=True requires '{session_col}' column in X, "
                f"but it was not found.\n"
                f"\n"
                f"Available columns: {columns}\n"
                f"\n"
                f"To add session dates to your data using the qdata library:\n"
                f"\n"
                f"Option 1 - Using DataManager (recommended):\n"
                f"  from qdata import DataManager\n"
                f"  manager = DataManager()\n"
                f"  df = manager.load(\n"
                f"      symbol='BTC',\n"
                f"      exchange='CME',\n"
                f"      calendar='CME_Globex_Crypto'\n"
                f"  )\n"
                f"  # df now has '{session_col}' column automatically\n"
                f"\n"
                f"Option 2 - Using SessionAssigner directly:\n"
                f"  from ml4t.data.sessions import SessionAssigner\n"
                f"  assigner = SessionAssigner.from_exchange('CME')\n"
                f"  df_with_sessions = assigner.assign_sessions(df)\n"
                f"\n"
                f"Option 3 - If you have a different session column:\n"
                f"  cv = {self.__class__.__name__}(\n"
                f"      ...,\n"
                f"      align_to_sessions=True,\n"
                f"      session_col='your_column_name'  # Specify your column\n"
                f"  )\n"
                f"\n"
                f"Option 4 - Disable session alignment:\n"
                f"  cv = {self.__class__.__name__}(\n"
                f"      ...,\n"
                f"      align_to_sessions=False  # Use standard splitting\n"
                f"  )\n"
            )

    def _get_unique_sessions(
        self,
        X: pl.DataFrame | pd.DataFrame,
        session_col: str,
    ) -> pl.Series | pd.Series:
        """Extract unique session identifiers in order of first appearance.

        Parameters
        ----------
        X : polars.DataFrame or pandas.DataFrame
            Data containing session column.

        session_col : str
            Name of the session column.

        Returns
        -------
        sessions : polars.Series or pandas.Series
            Unique session identifiers in order of first appearance.

        Notes
        -----
        Sessions are returned in the order they first appear in the data, which
        is the correct chronological order if the data is sorted by time (as it
        should be for time-series cross-validation).

        Previously this method sorted by session ID, which is incorrect when
        session IDs are not naturally sortable in chronological order.
        """
        if isinstance(X, pl.DataFrame):
            # maintain_order=True preserves order of first appearance
            return X[session_col].unique(maintain_order=True)
        else:  # pandas DataFrame
            # drop_duplicates without sorting preserves first appearance order
            return X[session_col].drop_duplicates().reset_index(drop=True)

    def _session_to_indices(
        self,
        X: pl.DataFrame | pd.DataFrame,
        session_col: str,
    ) -> tuple[list[Any], dict[Any, "NDArray[np.intp]"]]:
        """Map each session to its row indices, preserving appearance order.

        This is the key helper for session-aligned CV. It returns EXACT indices
        per session, not (start, end) boundaries, which is critical for correct
        behavior with non-contiguous or interleaved data.

        Parameters
        ----------
        X : polars.DataFrame or pandas.DataFrame
            Data containing session column.

        session_col : str
            Name of the session column.

        Returns
        -------
        ordered_sessions : list
            Session IDs in order of first appearance.

        session_indices : dict
            Mapping from session ID to numpy array of row indices (sorted).

        Examples
        --------
        >>> # Data with interleaved assets
        >>> X = pl.DataFrame({
        ...     "session": ["A", "A", "B", "A", "B"],
        ...     "asset": ["X", "Y", "X", "X", "Y"]
        ... })
        >>> sessions, indices = splitter._session_to_indices(X, "session")
        >>> sessions
        ['A', 'B']
        >>> indices['A']
        array([0, 1, 3])  # Exact indices, NOT range(0, 3)
        >>> indices['B']
        array([2, 4])
        """
        if isinstance(X, pl.DataFrame):
            # Polars: use group_by with maintain_order=True
            # Add row indices, group by session, collect indices per group
            df_with_idx = X.with_row_index("__row_idx__")
            grouped = df_with_idx.group_by(session_col, maintain_order=True).agg(
                pl.col("__row_idx__")
            )
            ordered_sessions = grouped[session_col].to_list()
            session_indices = {
                row[session_col]: np.array(row["__row_idx__"], dtype=np.intp)
                for row in grouped.iter_rows(named=True)
            }
        else:
            # Pandas: use groupby().indices (fast, returns dict of arrays)
            grouped = X.groupby(session_col, sort=False)
            session_indices_raw = grouped.indices
            # Preserve appearance order using drop_duplicates
            ordered_sessions = X[session_col].drop_duplicates().tolist()
            session_indices = {
                session: np.array(session_indices_raw[session], dtype=np.intp)
                for session in ordered_sessions
            }

        return ordered_sessions, session_indices

    def _extract_timestamps(
        self,
        X: Union[pl.DataFrame, pd.DataFrame, "NDArray[Any]"],
        timestamp_col: str | None = None,
    ) -> pd.DatetimeIndex | None:
        """Extract timestamps from data for time-based size calculations.

        This method supports both Polars and pandas DataFrames, enabling
        time-based test_size/train_size specifications (e.g., '4W', '3M').

        Parameters
        ----------
        X : polars.DataFrame, pandas.DataFrame, or numpy.ndarray
            Input data.
        timestamp_col : str or None
            Column name containing timestamps for Polars DataFrames.
            If None, falls back to pandas DatetimeIndex (backward compatible).

        Returns
        -------
        timestamps : pandas.DatetimeIndex or None
            Timestamps as a pandas DatetimeIndex for time-based calculations.
            Returns None if timestamps cannot be extracted.

        Notes
        -----
        For Polars DataFrames:
            - Requires timestamp_col to be specified
            - Column must be datetime type
            - Converts to pandas DatetimeIndex for compatibility with time parsing

        For pandas DataFrames:
            - Uses DatetimeIndex if available
            - Falls back to timestamp_col if index is not datetime

        For numpy arrays:
            - Returns None (no timestamp information available)
        """
        # Polars DataFrame: extract from column
        if isinstance(X, pl.DataFrame):
            if timestamp_col is None:
                return None
            if timestamp_col not in X.columns:
                raise ValueError(
                    f"timestamp_col='{timestamp_col}' not found in Polars DataFrame. "
                    f"Available columns: {X.columns}"
                )
            # Convert Polars datetime column to pandas DatetimeIndex
            ts_series = X[timestamp_col].to_pandas()
            if not pd.api.types.is_datetime64_any_dtype(ts_series):
                raise ValueError(
                    f"timestamp_col='{timestamp_col}' must be datetime type, "
                    f"got {X[timestamp_col].dtype}"
                )
            idx = pd.DatetimeIndex(ts_series)
            # Ensure timezone awareness (required for purging/embargo)
            if idx.tz is None:
                idx = idx.tz_localize("UTC")
            return idx

        # pandas DataFrame: prefer index, fallback to column
        if isinstance(X, pd.DataFrame):
            if isinstance(X.index, pd.DatetimeIndex):
                return X.index
            # Fallback: try timestamp_col if specified
            if timestamp_col is not None and timestamp_col in X.columns:
                ts_series = X[timestamp_col]
                if pd.api.types.is_datetime64_any_dtype(ts_series):
                    return pd.DatetimeIndex(ts_series)
            return None

        # numpy array: no timestamp information
        return None

    def __repr__(self) -> str:
        """Return a string representation of the splitter."""
        return f"{self.__class__.__name__}()"
