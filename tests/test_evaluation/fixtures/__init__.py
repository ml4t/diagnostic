"""Test fixtures for DSR validation.

Reference test cases from López de Prado et al. (2025).
"""

from .dsr_reference import (
    DSR_CASES,
    MINTRL_CASES,
    POWER_CASES,
    PSR_CASES,
    VARIANCE_RESCALING_FACTORS,
    DSRTestCase,
    MinTRLTestCase,
    PowerTestCase,
    PSRTestCase,
    expected_max_sharpe,
    get_summary,
    get_variance_rescaling_factor,
)

__all__ = [
    # Test case lists
    "PSR_CASES",
    "MINTRL_CASES",
    "DSR_CASES",
    "POWER_CASES",
    "VARIANCE_RESCALING_FACTORS",
    # Test case types
    "PSRTestCase",
    "DSRTestCase",
    "MinTRLTestCase",
    "PowerTestCase",
    # Helper functions
    "get_variance_rescaling_factor",
    "expected_max_sharpe",
    "get_summary",
]
