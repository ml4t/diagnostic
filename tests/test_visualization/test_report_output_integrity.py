"""A real interrupted report write must not destroy the previous verified report."""

import subprocess
import sys

import pytest


def test_actual_filesystem_limit_preserves_previous_report_and_removes_partial_output(tmp_path):
    pytest.importorskip("resource", reason="POSIX file-size limit required for actual interruption")
    path = tmp_path / "report.html"
    path.write_bytes(b"previous verified combined report")
    probe = """
import errno
import resource
import signal
import sys
import plotly.graph_objects as go
from ml4t.diagnostic.visualization.report_generation import combine_figures_to_html
figure = go.Figure(go.Scatter(x=list(range(512)), y=list(range(512))))
signal.signal(signal.SIGXFSZ, signal.SIG_IGN)
resource.setrlimit(resource.RLIMIT_FSIZE, (4096, 4096))
try:
    combine_figures_to_html([figure], output_file=sys.argv[1])
except OSError as error:
    assert error.errno == errno.EFBIG, repr(error)
    print("actual write interrupted")
else:
    raise AssertionError("Real filesystem limit did not interrupt the report write")
"""
    result = subprocess.run(
        [sys.executable, "-c", probe, str(path)], capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0, result.stderr
    assert "actual write interrupted" in result.stdout
    assert path.read_bytes() == b"previous verified combined report"
    assert list(tmp_path.iterdir()) == [path]
