"""Reference values for the Sharpe ratio inference tests.

See dsr_reference.py for the source of each value.
"""

from .dsr_reference import (
    DSR_CASES,
    MINTRL_CASES,
    PAPER_SHARPE,
    PSR_CASES,
    VARIANCE_RESCALING_FACTORS,
    DSRTestCase,
    MinTRLTestCase,
    PSRTestCase,
)

__all__ = [
    "PAPER_SHARPE",
    # Test case lists
    "PSR_CASES",
    "MINTRL_CASES",
    "DSR_CASES",
    "VARIANCE_RESCALING_FACTORS",
    # Test case types
    "PSRTestCase",
    "DSRTestCase",
    "MinTRLTestCase",
]
