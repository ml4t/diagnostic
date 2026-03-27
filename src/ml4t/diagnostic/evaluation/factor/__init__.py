"""Factor exposure and attribution analysis.

Provides returns-based factor regression (OLS+HAC), rolling exposures,
lagged return attribution, variance-based risk decomposition, and
model validation via QLIKE/MALV.

References
----------
- Paleologo, *Elements of Quantitative Investing* (2025)
- Newey & West (1987): HAC standard errors
- Fama & French (1993, 2015): Factor models

Examples
--------
>>> from ml4t.diagnostic.evaluation.factor import (
...     FactorData, FactorAnalysis, compute_factor_model
... )
>>> factor_data = FactorData.from_dataframe(factors_df, rf_column="RF")
>>> fa = FactorAnalysis(strategy_returns, factor_data)
>>> model = fa.static_model()
>>> print(model.summary())

For Fama-French data (requires ml4t-data):

>>> factor_data = FactorData.from_fama_french(dataset="ff3")
"""

from .analysis import FactorAnalysis  # noqa: F401
from .attribution import (  # noqa: F401
    compute_maximal_attribution,
    compute_return_attribution,
)
from .data import FactorData  # noqa: F401
from .kalman import compute_kalman_betas  # noqa: F401
from .regularized import compute_regularized_model  # noqa: F401
from .results import (  # noqa: F401
    AttributionResult,
    FactorModelResult,
    FactorTimingResult,
    MaximalAttributionResult,
    ModelValidationResult,
    RiskAttributionResult,
    RollingExposureResult,
    StabilityDiagnostics,
)
from .risk import compute_risk_attribution  # noqa: F401
from .rolling_model import compute_rolling_exposures  # noqa: F401
from .static_model import compute_factor_model  # noqa: F401
from .timing import compute_factor_timing  # noqa: F401
from .validation import validate_factor_model  # noqa: F401


def load_fama_french_5factor(
    frequency: str = "daily",
    start_date: str | None = None,
    end_date: str | None = None,
) -> FactorData:
    """Load Fama-French 5-factor data ready for tearsheet use.

    Convenience wrapper around ``FactorData.from_fama_french()``.
    Requires the ``factors`` extra: ``pip install ml4t-diagnostic[factors]``.

    Parameters
    ----------
    frequency : str
        ``"daily"`` or ``"monthly"``.
    start_date, end_date : str | None
        Date range filter (``"YYYY-MM-DD"``).

    Returns
    -------
    FactorData
        Ready to pass to ``generate_backtest_tearsheet(factor_data=...)``.

    Examples
    --------
    >>> from ml4t.diagnostic.evaluation.factor import load_fama_french_5factor
    >>> factor_data = load_fama_french_5factor()
    >>> html = generate_backtest_tearsheet(result=result, factor_data=factor_data)
    """
    return FactorData.from_fama_french(
        dataset="ff5",
        frequency=frequency,
        start_date=start_date,
        end_date=end_date,
    )


__all__: list[str] = [
    # Data container
    "FactorData",
    # Convenience loaders
    "load_fama_french_5factor",
    # Orchestrator
    "FactorAnalysis",
    # Core functions
    "compute_factor_model",
    "compute_rolling_exposures",
    "compute_return_attribution",
    "compute_maximal_attribution",
    "compute_risk_attribution",
    "validate_factor_model",
    # Tier 2 functions
    "compute_kalman_betas",
    "compute_regularized_model",
    "compute_factor_timing",
    # Result types
    "FactorModelResult",
    "RollingExposureResult",
    "StabilityDiagnostics",
    "AttributionResult",
    "MaximalAttributionResult",
    "RiskAttributionResult",
    "FactorTimingResult",
    "ModelValidationResult",
]
