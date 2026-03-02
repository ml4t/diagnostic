"""Initial test to verify setup."""

from importlib.metadata import version

import ml4t.diagnostic


def test_setup_complete():
    """Test that our setup is working."""
    # Setup is complete, we can import our modules
    from ml4t.diagnostic.backends import DataFrameAdapter
    from ml4t.diagnostic.splitters import BaseSplitter

    assert ml4t.diagnostic.__version__ == version("ml4t-diagnostic")
    assert BaseSplitter is not None
    assert DataFrameAdapter is not None
