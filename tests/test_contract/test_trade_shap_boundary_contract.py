"""Contract checks for Trade SHAP public boundary exports."""

from __future__ import annotations

import importlib


def test_trade_shap_modules_do_not_export_internal_config_dataclasses() -> None:
    """Internal Trade SHAP config dataclasses must stay out of public boundary modules."""
    hidden = {"ClusteringConfig", "CharacterizationConfig", "HypothesisConfig"}
    module_names = [
        "ml4t.diagnostic.evaluation.trade_shap",
        "ml4t.diagnostic.evaluation.trade_shap_diagnostics",
    ]

    for module_name in module_names:
        module = importlib.import_module(module_name)
        present = sorted(name for name in hidden if hasattr(module, name))
        assert not present, f"{module_name} unexpectedly exports internal config types: {present}"
