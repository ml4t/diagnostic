"""Result classes for Barrier Analysis module.

This package provides Pydantic result classes for storing and serializing
barrier analysis outputs including hit rates, profit factors, precision/recall,
and time-to-target metrics.

Triple barrier outcomes from ml4t.features:
- label: int (-1=SL hit, 0=timeout, 1=TP hit)
- label_return: float (actual return at exit)
- label_bars: int (bars from entry to exit)

References
----------
Lopez de Prado, M. (2018). "Advances in Financial Machine Learning"
    Chapter 3: Labeling (Triple Barrier Method)
"""

from __future__ import annotations

from ml4t.diagnostic.results.barrier_results.hit_rate import HitRateResult
from ml4t.diagnostic.results.barrier_results.precision_recall import PrecisionRecallResult
from ml4t.diagnostic.results.barrier_results.profit_factor import ProfitFactorResult
from ml4t.diagnostic.results.barrier_results.tearsheet import BarrierTearSheet
from ml4t.diagnostic.results.barrier_results.time_to_target import TimeToTargetResult
from ml4t.diagnostic.results.barrier_results.validation import _validate_quantile_dict_keys

__all__ = [
    # Validation helper
    "_validate_quantile_dict_keys",
    # Result classes
    "HitRateResult",
    "ProfitFactorResult",
    "PrecisionRecallResult",
    "TimeToTargetResult",
    "BarrierTearSheet",
]
