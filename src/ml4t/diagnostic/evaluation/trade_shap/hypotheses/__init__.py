"""Hypothesis generation for trade SHAP error patterns.

This package provides template-based hypothesis generation for explaining
why trading patterns cause losses, with templates stored as YAML data.
"""

from ml4t.diagnostic.evaluation.trade_shap.hypotheses.generator import (
    HypothesisConfig,
    HypothesisGenerator,
)
from ml4t.diagnostic.evaluation.trade_shap.hypotheses.matcher import (
    Template,
    TemplateMatcher,
    load_templates,
)

__all__ = [
    "HypothesisGenerator",
    "HypothesisConfig",
    "TemplateMatcher",
    "Template",
    "load_templates",
]
