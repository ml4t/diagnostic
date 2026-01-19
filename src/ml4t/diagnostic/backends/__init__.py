"""Backend adapters for DataFrame compatibility.

This module provides adapters to seamlessly work with both Polars (internal)
and Pandas (compatibility) DataFrames.
"""

from ml4t.diagnostic.backends.adapter import DataFrameAdapter
from ml4t.diagnostic.backends.polars_backend import PolarsBackend

__all__ = ["DataFrameAdapter", "PolarsBackend"]
