"""Contract checks for the curated evaluation API surface."""

from __future__ import annotations

import importlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = ROOT / "tests" / "contracts" / "evaluation_api_surface.json"


def _load_contract() -> dict[str, object]:
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def test_evaluation_all_matches_surface_contract() -> None:
    """`evaluation.__all__` must match the curated stable export list."""
    contract = _load_contract()
    module_name = str(contract["module"])
    expected = list(contract["stable_exports"])

    module = importlib.import_module(module_name)
    actual = list(getattr(module, "__all__", []))

    assert actual == expected, (
        "Evaluation API surface drift detected.\n"
        f"Expected ({len(expected)}): {expected}\n"
        f"Actual ({len(actual)}): {actual}"
    )


def test_evaluation_surface_symbols_exist() -> None:
    """All curated evaluation exports must resolve on the module."""
    contract = _load_contract()
    module_name = str(contract["module"])
    expected = list(contract["stable_exports"])

    module = importlib.import_module(module_name)
    missing = [name for name in expected if not hasattr(module, name)]

    assert not missing, f"{module_name} missing curated exports: {missing}"
