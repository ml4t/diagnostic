"""Discovering analytical adapters must not require visualization extras."""

import subprocess
import sys


def test_importing_transaction_plot_adapters_does_not_load_optional_plotly():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import importlib.abc, sys\n"
            "class WithoutVisualization(importlib.abc.MetaPathFinder):\n"
            "    def find_spec(self, fullname, path=None, target=None):\n"
            "        if fullname == 'plotly' or fullname.startswith('plotly.'):\n"
            "            raise ModuleNotFoundError(\"No module named 'plotly'\", name='plotly')\n"
            "sys.meta_path.insert(0, WithoutVisualization())\n"
            "from ml4t.diagnostic.compat.pyfolio import plotting\n"
            "assert 'plotly' not in sys.modules\n"
            "assert callable(plotting.plot_turnover)\n",
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
