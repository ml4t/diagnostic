"""Discovering analytical adapters must not require visualization extras."""

import subprocess
import sys


def test_importing_transaction_plot_adapters_does_not_load_optional_plotly():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from ml4t.diagnostic.compat.pyfolio import plotting; "
            "assert 'plotly' not in sys.modules; assert callable(plotting.plot_turnover)",
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
