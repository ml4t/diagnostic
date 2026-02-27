"""Contract tests for preserving polars-first compute paths in hotspot modules."""

from __future__ import annotations

import io
import json
import re
import token
import tokenize
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = ROOT / "tests" / "contracts" / "polars_compute_contract.json"


def _load_contract() -> dict:
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def _strip_strings_and_comments(source: str) -> str:
    """Return source with comments/docstrings removed to reduce false positives."""
    out: list[str] = []
    tokens = tokenize.generate_tokens(io.StringIO(source).readline)
    for tok_type, tok_string, *_ in tokens:
        if tok_type in {token.STRING, tokenize.COMMENT}:
            out.append(" ")
        else:
            out.append(tok_string)
    return " ".join(out)


def test_polars_first_hotspots_do_not_reintroduce_pandas_compute_primitives() -> None:
    """Hotspot modules should avoid pandas-native compute APIs except at boundaries."""
    contract = _load_contract()
    target_files: list[str] = contract["target_files"]
    forbidden_patterns: list[str] = contract["forbidden_patterns"]
    forbidden_regexes = [re.compile(p) for p in forbidden_patterns]

    violations: list[str] = []
    for rel_path in target_files:
        path = ROOT / rel_path
        text = _strip_strings_and_comments(path.read_text(encoding="utf-8"))

        for pattern, regex in zip(forbidden_patterns, forbidden_regexes, strict=True):
            if regex.search(text):
                violations.append(f"{rel_path}: matches forbidden pandas pattern `{pattern}`")

    assert not violations, "Pandas-first compute regression detected:\n" + "\n".join(violations)
