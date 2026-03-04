"""FactorAnalysis orchestrator providing a unified class-based API.

Wraps the functional API (compute_factor_model, compute_rolling_exposures,
etc.) into a single stateful object that caches results.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import polars as pl

from .data import FactorData
from .results import (
    AttributionResult,
    FactorModelResult,
    FactorTimingResult,
    MaximalAttributionResult,
    ModelValidationResult,
    RiskAttributionResult,
    RollingExposureResult,
)


class FactorAnalysis:
    """Unified factor analysis interface.

    Provides access to static regression, rolling exposures, return/risk
    attribution, and model validation through a single object. Results
    are cached after first computation.

    Parameters
    ----------
    returns : np.ndarray | pl.Series
        Portfolio returns (T,).
    factor_data : FactorData
        Factor return data.
    periods_per_year : int
        Annualization factor (252 for daily, 12 for monthly).

    Examples
    --------
    >>> fa = FactorAnalysis(strategy_returns, factor_data)
    >>> model = fa.static_model()
    >>> print(model.summary())
    >>> rolling = fa.rolling_model(window=63)
    >>> attr = fa.attribution()
    """

    def __init__(
        self,
        returns: np.ndarray | pl.Series,
        factor_data: FactorData,
        periods_per_year: int = 252,
    ) -> None:
        self._returns = returns
        self._factor_data = factor_data
        self._periods_per_year = periods_per_year
        self._cache: dict[str, Any] = {}

    @property
    def factor_names(self) -> list[str]:
        return self._factor_data.factor_names

    @property
    def n_periods(self) -> int:
        return self._factor_data.n_periods

    def static_model(
        self,
        *,
        method: Literal["ols"] = "ols",
        hac: bool = True,
        max_lags: int | None = None,
        confidence_level: float = 0.95,
    ) -> FactorModelResult:
        """Fit static factor model. Cached after first call."""
        key = f"static_{method}_{hac}_{max_lags}_{confidence_level}"
        if key not in self._cache:
            from .static_model import compute_factor_model

            self._cache[key] = compute_factor_model(
                self._returns,
                self._factor_data,
                method=method,
                hac=hac,
                max_lags=max_lags,
                confidence_level=confidence_level,
            )
        return self._cache[key]

    def rolling_model(
        self,
        *,
        window: int = 63,
        expanding: bool = False,
        min_periods: int | None = None,
        compute_vif: bool = False,
    ) -> RollingExposureResult:
        """Compute rolling factor exposures. Cached per window."""
        key = f"rolling_{window}_{expanding}_{min_periods}_{compute_vif}"
        if key not in self._cache:
            from .rolling_model import compute_rolling_exposures

            self._cache[key] = compute_rolling_exposures(
                self._returns,
                self._factor_data,
                window=window,
                expanding=expanding,
                min_periods=min_periods,
                compute_vif=compute_vif,
            )
        return self._cache[key]

    def attribution(
        self,
        *,
        window: int = 63,
        lag: int = 1,
        confidence_level: float = 0.95,
    ) -> AttributionResult:
        """Compute return attribution. Cached per parameters."""
        key = f"attr_{window}_{lag}_{confidence_level}"
        if key not in self._cache:
            from .attribution import compute_return_attribution

            self._cache[key] = compute_return_attribution(
                self._returns,
                self._factor_data,
                window=window,
                lag=lag,
                confidence_level=confidence_level,
            )
        return self._cache[key]

    def maximal_attribution(
        self,
        factors_of_interest: list[str],
        *,
        model_result: FactorModelResult | None = None,
    ) -> MaximalAttributionResult:
        """Compute maximal attribution (Paleologo Ch 14)."""
        from .attribution import compute_maximal_attribution

        if model_result is None:
            model_result = self.static_model()
        return compute_maximal_attribution(
            self._returns,
            self._factor_data,
            factors_of_interest,
            model_result=model_result,
        )

    def risk_attribution(
        self,
        *,
        shrinkage: Literal["none", "ledoit_wolf", "oracle"] = "ledoit_wolf",
    ) -> RiskAttributionResult:
        """Compute risk decomposition. Cached per shrinkage."""
        key = f"risk_{shrinkage}"
        if key not in self._cache:
            from .risk import compute_risk_attribution

            self._cache[key] = compute_risk_attribution(
                self._returns,
                self._factor_data,
                model_result=self.static_model(),
                shrinkage=shrinkage,
            )
        return self._cache[key]

    def validate_model(
        self,
        *,
        max_acf_lags: int = 10,
    ) -> ModelValidationResult:
        """Run model validation diagnostics (Paleologo Ch 5)."""
        key = f"validate_{max_acf_lags}"
        if key not in self._cache:
            from .static_model import _align_and_prepare
            from .validation import validate_factor_model

            model = self.static_model()
            # Use _align_and_prepare to get the exact rows used in fitting
            # (handles NaN masking and length alignment consistently)
            _, X_aligned, _ = _align_and_prepare(self._returns, self._factor_data)
            self._cache[key] = validate_factor_model(
                model, X_aligned, max_acf_lags=max_acf_lags
            )
        return self._cache[key]

    def factor_timing(self, *, window: int = 63) -> FactorTimingResult:
        """Analyze factor timing ability (Tier 2)."""
        key = f"timing_{window}"
        if key not in self._cache:
            from .timing import compute_factor_timing

            self._cache[key] = compute_factor_timing(
                self._returns, self._factor_data, window=window
            )
        return self._cache[key]

    def kalman_model(self, **kwargs: Any) -> RollingExposureResult:
        """Estimate time-varying betas via Kalman filter (Tier 2)."""
        from .kalman import compute_kalman_betas

        return compute_kalman_betas(self._returns, self._factor_data, **kwargs)

    def generate_report(
        self,
        *,
        theme: str | None = None,
        window: int = 63,
    ) -> dict[str, Any]:
        """Generate all factor analysis plots.

        Returns dict mapping plot name to plotly Figure.
        """
        from ml4t.diagnostic.visualization.factor import (
            plot_factor_betas_bar,
            plot_factor_correlation_heatmap,
            plot_residual_diagnostics,
            plot_return_attribution_waterfall,
            plot_risk_attribution_pie,
            plot_rolling_betas,
        )

        model = self.static_model()
        rolling = self.rolling_model(window=window)
        attr = self.attribution(window=window)
        risk = self.risk_attribution()

        figures = {
            "factor_betas": plot_factor_betas_bar(model, theme=theme),
            "rolling_betas": plot_rolling_betas(rolling, theme=theme),
            "attribution_waterfall": plot_return_attribution_waterfall(attr, theme=theme),
            "risk_decomposition": plot_risk_attribution_pie(risk, theme=theme),
            "residual_diagnostics": plot_residual_diagnostics(model, theme=theme),
            "factor_correlation": plot_factor_correlation_heatmap(self._factor_data, theme=theme),
        }
        return figures

    def clear_cache(self) -> None:
        """Clear all cached results."""
        self._cache.clear()
