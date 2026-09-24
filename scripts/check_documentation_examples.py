"""Run principal public Python examples against an installed Diagnostic wheel."""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import ml4t.diagnostic

ROOT = Path(__file__).parents[1]
PYTHON_FENCE = re.compile(r"^```python\s*\n(.*?)^```\s*$", re.MULTILINE | re.DOTALL)
PAGES = {
    "README.md": ("IC (1D):", "Q5-Q1 spread (1D):"),
    "docs/index.md": ("Probability of skill:", "Significant: False"),
    "docs/getting-started/quickstart.md": ("1-day IC: 0.3738", "Effective trials:"),
    "docs/user-guide/cross-validation.md": ("CPCV combinations: 15",),
    "docs/user-guide/statistical-tests.md": (
        "Raw trials: 4",
        "Rejected hypotheses: [True, True, True, False, False]",
    ),
    "docs/user-guide/feature-diagnostics.md": ("Health score:", "Monotonicity:"),
    "docs/user-guide/feature-selection.md": ("Selected:", "Removed:"),
    "docs/user-guide/workflows.md": ("Validation folds: 4", "DSR probability:"),
    "docs/user-guide/backtest-tearsheets.md": ("Wrote backtest_report.html",),
    "docs/user-guide/trade-analysis.md": ("[-410.0, -240.0, -130.0]",),
    "docs/user-guide/migration.md": ("One-day IC: 0.347", "Portfolio Sharpe:"),
}


def _source(page: Path) -> str:
    blocks = PYTHON_FENCE.findall(page.read_text(encoding="utf-8"))
    if not blocks:
        raise ValueError(f"No Python example in {page}")
    return "\n\n".join(blocks)


def main() -> None:
    """Check imports, assertions, and stable output without the source checkout."""
    module_path = Path(ml4t.diagnostic.__file__).resolve()
    if module_path.is_relative_to(ROOT):
        raise RuntimeError(f"Diagnostic was imported from the source checkout: {module_path}")
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment["MPLBACKEND"] = "Agg"
    environment["PYTHONHASHSEED"] = "0"
    with tempfile.TemporaryDirectory(prefix="ml4t-diagnostic-docs-") as directory:
        scratch = Path(directory)
        for relative, expected in PAGES.items():
            workdir = scratch / Path(relative).stem
            workdir.mkdir()
            completed = subprocess.run(
                [sys.executable, "-I", "-c", _source(ROOT / relative)],
                cwd=workdir,
                env=environment,
                capture_output=True,
                text=True,
                timeout=120,
                check=False,
            )
            if completed.returncode:
                raise RuntimeError(
                    f"{relative} failed with exit {completed.returncode}:\n"
                    f"{completed.stdout}\n{completed.stderr}"
                )
            for fragment in expected:
                if fragment not in completed.stdout:
                    raise AssertionError(f"{relative} did not print {fragment!r}")
            print(f"Checked {relative}: {len(expected)} expected output markers")


if __name__ == "__main__":
    main()
