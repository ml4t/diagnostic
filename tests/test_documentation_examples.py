"""Release-blocking execution tests for Python examples in public documentation."""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).parents[1]
DOCUMENTATION_PATHS = (
    REPOSITORY_ROOT / "README.md",
    *sorted((REPOSITORY_ROOT / "docs").rglob("*.md")),
)
PYTHON_FENCE = re.compile(r"^```python\s*\n(.*?)^```\s*$", re.MULTILINE | re.DOTALL)


def _documented_python_examples() -> tuple[tuple[str, str], ...]:
    examples = []
    for path in DOCUMENTATION_PATHS:
        blocks = PYTHON_FENCE.findall(path.read_text(encoding="utf-8"))
        if blocks:
            relative_path = path.relative_to(REPOSITORY_ROOT).as_posix()
            source = "\n\n".join(
                f"# {relative_path}, Python block {index}\n{block}"
                for index, block in enumerate(blocks, start=1)
            )
            examples.append((relative_path, source))
    return tuple(examples)


DOCUMENTED_PYTHON_EXAMPLES = _documented_python_examples()
assert DOCUMENTED_PYTHON_EXAMPLES, "no public documentation examples were discovered"


@pytest.mark.parametrize(
    ("relative_path", "source"),
    DOCUMENTED_PYTHON_EXAMPLES,
    ids=[relative_path for relative_path, _ in DOCUMENTED_PYTHON_EXAMPLES],
)
def test_documented_python_examples_execute(
    relative_path: str,
    source: str,
    tmp_path: Path,
) -> None:
    """Every page's Python blocks execute together in publication order."""
    script = tmp_path / "documented_examples.py"
    script.write_text(source, encoding="utf-8")
    environment = {
        **os.environ,
        "MPLBACKEND": "Agg",
        "PYTHONHASHSEED": "0",
        "PYTHONIOENCODING": "utf-8",
    }

    try:
        completed = subprocess.run(
            [sys.executable, str(script)],
            cwd=tmp_path,
            env=environment,
            capture_output=True,
            encoding="utf-8",
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
