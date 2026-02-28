"""Docs import contract checks for markdown Python snippets."""

from __future__ import annotations

import ast
import importlib
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOC_FILES = [ROOT / "README.md", *(ROOT / "docs").rglob("*.md")]
PYTHON_FENCE_PATTERN = re.compile(r"```python\s*\n(.*?)```", re.DOTALL)


def _iter_python_blocks(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    return [match.group(1) for match in PYTHON_FENCE_PATTERN.finditer(text)]


def test_docs_python_imports_are_resolvable() -> None:
    """All ml4t.diagnostic imports in docs snippets should resolve."""
    missing: list[str] = []

    for path in DOC_FILES:
        rel_path = path.relative_to(ROOT)
        for block_index, block in enumerate(_iter_python_blocks(path), start=1):
            try:
                tree = ast.parse(block)
            except SyntaxError:
                continue

            for node in ast.walk(tree):
                if not isinstance(node, ast.ImportFrom):
                    continue
                if node.module is None or not node.module.startswith("ml4t.diagnostic"):
                    continue

                try:
                    module = importlib.import_module(node.module)
                except Exception as exc:  # pragma: no cover - failure path only
                    missing.append(
                        f"{rel_path} [block {block_index}]: failed import '{node.module}' ({exc})"
                    )
                    continue

                for alias in node.names:
                    if alias.name == "*":
                        continue
                    if not hasattr(module, alias.name):
                        missing.append(
                            f"{rel_path} [block {block_index}]: "
                            f"'{node.module}.{alias.name}' not found"
                        )

    assert not missing, "Docs snippet import contract violations:\n" + "\n".join(missing)
