"""Release-blocking execution tests for public Python examples."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).parents[1]
EXAMPLES_ROOT = REPOSITORY_ROOT / "examples"
PYTHON_EXAMPLES = tuple(
    path.relative_to(EXAMPLES_ROOT)
    for path in sorted(EXAMPLES_ROOT.rglob("*.py"))
    if path.relative_to(EXAMPLES_ROOT) != Path("trade_shap_dashboard_demo.py")
)
assert PYTHON_EXAMPLES, "no public Python examples were discovered"


@pytest.mark.parametrize("relative_path", PYTHON_EXAMPLES, ids=str)
def test_public_python_example_executes(relative_path: Path, tmp_path: Path) -> None:
    """Every retained script completes in an isolated working directory."""
    copied_examples = tmp_path / "examples"
    shutil.copytree(EXAMPLES_ROOT, copied_examples)
    target = copied_examples / relative_path
    environment = {
        **os.environ,
        "MPLBACKEND": "Agg",
        "PYTHONHASHSEED": "0",
    }

    try:
        completed = subprocess.run(
            [sys.executable, str(target)],
            cwd=tmp_path,
            env=environment,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        pytest.fail(
            f"{relative_path} exceeded 120 seconds\n"
            f"stdout:\n{(exc.stdout or '')[-4_000:]}\n"
            f"stderr:\n{(exc.stderr or '')[-4_000:]}"
        )

    assert completed.returncode == 0, (
        f"{relative_path} failed with exit {completed.returncode}\n"
        f"stdout:\n{completed.stdout[-4_000:]}\n"
        f"stderr:\n{completed.stderr[-4_000:]}"
    )


def test_trade_dashboard_example_executes() -> None:
    """The Streamlit example renders through the supported runner."""
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(str(EXAMPLES_ROOT / "trade_shap_dashboard_demo.py")).run(timeout=30)

    assert not app.exception, [exception.value for exception in app.exception]
