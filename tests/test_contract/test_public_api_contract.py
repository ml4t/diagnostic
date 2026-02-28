"""Public API contract checks for the canonical api module."""

from __future__ import annotations

import importlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = ROOT / "tests" / "contracts" / "public_api_contract.json"


def _load_contract() -> dict:
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def test_public_contract_exports_exist() -> None:
    """Contract symbols must be importable from canonical modules."""
    contract = _load_contract()
    public_exports: dict[str, list[str]] = contract["public_exports"]

    for module_name, symbols in public_exports.items():
        module = importlib.import_module(module_name)
        missing = [symbol for symbol in symbols if not hasattr(module, symbol)]
        assert not missing, f"{module_name} missing contract symbols: {missing}"


def test_absent_exports_are_not_reexported() -> None:
    """Removed/stale names must not be present on canonical modules."""
    contract = _load_contract()
    absent_exports: dict[str, list[str]] = contract.get("absent_exports", {})

    for module_name, symbols in absent_exports.items():
        module = importlib.import_module(module_name)
        present = [symbol for symbol in symbols if hasattr(module, symbol)]
        assert not present, f"{module_name} unexpectedly exports stale symbols: {present}"


def test_forbidden_symbols_not_in_canonical_api_docs() -> None:
    """Canonical API files should not mention stale names."""
    contract = _load_contract()
    forbidden_symbols: list[str] = contract["forbidden_symbols"]
    pattern = re.compile(r"\b(" + "|".join(map(re.escape, forbidden_symbols)) + r")\b")

    files_to_scan = [
        ROOT / "src" / "ml4t" / "diagnostic" / "api.py",
        ROOT / "src" / "ml4t" / "diagnostic" / "__init__.py",
    ]

    violations: list[str] = []
    for path in files_to_scan:
        text = path.read_text(encoding="utf-8", errors="ignore")
        match = pattern.search(text)
        if match is not None:
            rel = path.relative_to(ROOT)
            violations.append(f"{rel}: contains '{match.group(0)}'")

    assert not violations, "Stale symbols found in canonical API files:\n" + "\n".join(violations)
