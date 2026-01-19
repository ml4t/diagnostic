"""IC (Information Coefficient) result classes for signal analysis.

This module provides result classes for storing IC analysis outputs including
time series data, summary statistics, HAC-adjusted values, and RAS adjustments.

References
----------
Lopez de Prado, M. (2018). "Advances in Financial Machine Learning"
Paleologo, G. (2024). "Elements of Quantitative Investing"
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import polars as pl
from pydantic import Field, model_validator

from ml4t.diagnostic.results.base import BaseResult
from ml4t.diagnostic.results.signal_results.validation import (
    _normalize_period,
    _validate_dict_keys_match,
)


@dataclass
class ICStats:
    """IC statistics for a single period.

    Provides a convenient typed container for all IC metrics
    at a specific forward return period.

    Examples
    --------
    >>> stats = ic_result.get_stats(21)
    >>> if stats:
    ...     print(f"IC: {stats.mean:.4f} (t={stats.t_stat:.2f})")
    """

    mean: float
    std: float
    t_stat: float
    p_value: float
    positive_pct: float
    ir: float  # Information Ratio
    t_stat_hac: float | None = None
    p_value_hac: float | None = None
    ras_adjusted: float | None = None
    ras_significant: bool | None = None


class SignalICResult(BaseResult):
    """Results from Signal IC (Information Coefficient) analysis.

    Contains IC time series, summary statistics, t-statistics,
    and optional RAS-adjusted values for signal analysis.

    This is distinct from feature_results.ICAnalysisResult which
    handles single-feature IC analysis (Module C).

    Examples
    --------
    >>> result = signal_ic_result
    >>> print(result.summary())
    >>> df = result.get_dataframe("ic_by_date")
    """

    analysis_type: str = Field(default="signal_ic_analysis", frozen=True)

    # ==========================================================================
    # IC Time Series Data
    # ==========================================================================

    ic_by_date: dict[str, list[float]] = Field(
        ...,
        description="IC values by date for each period. Keys: period names, values: IC series",
    )

    dates: list[str] = Field(
        ...,
        description="Date strings (ISO format) corresponding to IC values",
    )

    # ==========================================================================
    # Summary Statistics
    # ==========================================================================

    ic_mean: dict[str, float] = Field(
        ...,
        description="Mean IC for each period",
    )

    ic_std: dict[str, float] = Field(
        ...,
        description="Standard deviation of IC for each period",
    )

    ic_t_stat: dict[str, float] = Field(
        ...,
        description="T-statistic for IC mean != 0",
    )

    ic_p_value: dict[str, float] = Field(
        ...,
        description="P-value for IC significance (two-tailed)",
    )

    ic_positive_pct: dict[str, float] = Field(
        ...,
        description="Percentage of periods with positive IC",
    )

    ic_ir: dict[str, float] = Field(
        ...,
        description="Information Ratio (IC_mean / IC_std)",
    )

    # ==========================================================================
    # HAC-Adjusted Statistics (Newey-West)
    # ==========================================================================

    ic_t_stat_hac: dict[str, float] | None = Field(
        default=None,
        description="HAC-adjusted t-statistic (Newey-West)",
    )

    ic_p_value_hac: dict[str, float] | None = Field(
        default=None,
        description="HAC-adjusted p-value",
    )

    hac_lags_used: int | None = Field(
        default=None,
        description="Number of lags used for HAC adjustment",
    )

    # ==========================================================================
    # RAS-Adjusted Values (Rademacher Anti-Serum)
    # ==========================================================================

    ras_adjusted_ic: dict[str, float] | None = Field(
        default=None,
        description="RAS-adjusted conservative IC lower bounds",
    )

    ras_complexity: float | None = Field(
        default=None,
        description="Rademacher complexity R^ used in adjustment",
    )

    ras_significant: dict[str, bool] | None = Field(
        default=None,
        description="Whether RAS-adjusted IC > 0 (significant after multiple testing)",
    )

    # ==========================================================================
    # Validation
    # ==========================================================================

    @model_validator(mode="after")
    def _validate_period_keys(self) -> SignalICResult:
        """Validate that all period-keyed dicts share the same keys."""
        data = self.model_dump()
        _validate_dict_keys_match(
            data,
            required_fields=[
                "ic_by_date",
                "ic_mean",
                "ic_std",
                "ic_t_stat",
                "ic_p_value",
                "ic_positive_pct",
                "ic_ir",
            ],
            optional_fields=[
                "ic_t_stat_hac",
                "ic_p_value_hac",
                "ras_adjusted_ic",
                "ras_significant",
            ],
            reference_field="ic_mean",
        )
        return self

    # ==========================================================================
    # Methods
    # ==========================================================================

    def get_dataframe(self, name: str | None = None) -> pl.DataFrame:
        """Get results as Polars DataFrame.

        Parameters
        ----------
        name : str | None
            DataFrame to retrieve:
            - None or "ic_by_date": IC time series by date
            - "summary": Summary statistics

        Returns
        -------
        pl.DataFrame
            Requested DataFrame
        """
        if name is None or name == "ic_by_date":
            # Build IC time series DataFrame
            data: dict[str, Any] = {"date": self.dates}
            for period, values in self.ic_by_date.items():
                data[f"ic_{period}"] = values
            return pl.DataFrame(data)

        if name == "summary":
            # Build summary statistics DataFrame
            periods = list(self.ic_mean.keys())
            data_summary: dict[str, Any] = {
                "period": periods,
                "ic_mean": [self.ic_mean[p] for p in periods],
                "ic_std": [self.ic_std[p] for p in periods],
                "ic_t_stat": [self.ic_t_stat[p] for p in periods],
                "ic_p_value": [self.ic_p_value[p] for p in periods],
                "ic_positive_pct": [self.ic_positive_pct[p] for p in periods],
                "ic_ir": [self.ic_ir[p] for p in periods],
            }

            if self.ras_adjusted_ic is not None and self.ras_significant is not None:
                data_summary["ras_adjusted_ic"] = [self.ras_adjusted_ic[p] for p in periods]
                data_summary["ras_significant"] = [self.ras_significant[p] for p in periods]

            return pl.DataFrame(data_summary)

        raise ValueError(f"Unknown DataFrame name: {name}. Available: 'ic_by_date', 'summary'")

    def list_available_dataframes(self) -> list[str]:
        """List available DataFrame views."""
        return ["ic_by_date", "summary"]

    def summary(self) -> str:
        """Get human-readable summary of IC analysis results."""
        lines = ["=" * 60, "IC Analysis Summary", "=" * 60, ""]

        for period in self.ic_mean:
            lines.append(f"Period: {period}")
            lines.append(f"  Mean IC:      {self.ic_mean[period]:>8.4f}")
            lines.append(f"  Std IC:       {self.ic_std[period]:>8.4f}")
            lines.append(f"  IR:           {self.ic_ir[period]:>8.4f}")
            lines.append(f"  t-stat:       {self.ic_t_stat[period]:>8.2f}")
            lines.append(f"  p-value:      {self.ic_p_value[period]:>8.4f}")
            lines.append(f"  Positive %:   {self.ic_positive_pct[period]:>8.1%}")

            if self.ras_adjusted_ic is not None and self.ras_significant is not None:
                lines.append(f"  RAS IC:       {self.ras_adjusted_ic[period]:>8.4f}")
                sig = "Y" if self.ras_significant[period] else "X"
                lines.append(f"  RAS Signif:   {sig:>8}")
            lines.append("")

        return "\n".join(lines)

    # =========================================================================
    # Convenience Accessor Methods
    # =========================================================================

    @property
    def periods(self) -> list[str]:
        """List of available periods (e.g., ['1D', '5D', '21D'])."""
        return list(self.ic_mean.keys())

    def get_ic(self, period: int | str) -> float | None:
        """Get mean IC for a period, accepting int or string keys.

        Parameters
        ----------
        period : int | str
            Period as integer (21) or string ('21' or '21D').

        Returns
        -------
        float | None
            Mean IC for the period, or None if not found.

        Examples
        --------
        >>> ic_result.get_ic(21)  # Works
        >>> ic_result.get_ic('21')  # Works
        >>> ic_result.get_ic('21D')  # Works
        """
        key = _normalize_period(period)
        return self.ic_mean.get(key)

    def get_t_stat(self, period: int | str) -> float | None:
        """Get t-statistic for a period."""
        key = _normalize_period(period)
        return self.ic_t_stat.get(key)

    def get_p_value(self, period: int | str) -> float | None:
        """Get p-value for a period."""
        key = _normalize_period(period)
        return self.ic_p_value.get(key)

    def get_ir(self, period: int | str) -> float | None:
        """Get Information Ratio (IC/std) for a period."""
        key = _normalize_period(period)
        return self.ic_ir.get(key)

    def get_stats(self, period: int | str) -> ICStats | None:
        """Get all IC statistics for a period as a typed object.

        This is the recommended way to access IC results, providing
        a clean typed interface instead of multiple dict lookups.

        Parameters
        ----------
        period : int | str
            Period as integer or string (e.g., 21, '21', '21D').

        Returns
        -------
        ICStats | None
            Typed container with all IC metrics, or None if period not found.

        Examples
        --------
        >>> stats = ic_result.get_stats(21)
        >>> if stats:
        ...     print(f"IC: {stats.mean:.4f} (t={stats.t_stat:.2f}, p={stats.p_value:.4f})")
        ...     if stats.ras_significant:
        ...         print("Significant after RAS adjustment!")
        """
        key = _normalize_period(period)
        if key not in self.ic_mean:
            return None

        return ICStats(
            mean=self.ic_mean[key],
            std=self.ic_std[key],
            t_stat=self.ic_t_stat[key],
            p_value=self.ic_p_value[key],
            positive_pct=self.ic_positive_pct[key],
            ir=self.ic_ir[key],
            t_stat_hac=self.ic_t_stat_hac.get(key) if self.ic_t_stat_hac else None,
            p_value_hac=self.ic_p_value_hac.get(key) if self.ic_p_value_hac else None,
            ras_adjusted=self.ras_adjusted_ic.get(key) if self.ras_adjusted_ic else None,
            ras_significant=self.ras_significant.get(key) if self.ras_significant else None,
        )

    def is_significant(self, period: int | str, alpha: float = 0.05, use_hac: bool = True) -> bool:
        """Check if IC is statistically significant for a period.

        Parameters
        ----------
        period : int | str
            Period to check.
        alpha : float, default 0.05
            Significance level.
        use_hac : bool, default True
            Use HAC-adjusted p-value if available.

        Returns
        -------
        bool
            True if p-value < alpha.
        """
        key = _normalize_period(period)

        # Prefer HAC-adjusted p-value if available and requested
        p_val: float | None
        if use_hac and self.ic_p_value_hac and key in self.ic_p_value_hac:
            p_val = self.ic_p_value_hac[key]
        else:
            p_val = self.ic_p_value.get(key)

        if p_val is None:
            return False
        return p_val < alpha


class RASICResult(BaseResult):
    """Results from RAS-adjusted IC analysis.

    Specialized result class for Rademacher Anti-Serum adjustments
    used in multiple testing correction.

    Examples
    --------
    >>> result = ras_ic_result
    >>> if result.any_significant:
    ...     print("Found significant signals after RAS adjustment")
    """

    analysis_type: str = Field(default="ras_ic_analysis", frozen=True)

    # ==========================================================================
    # Input Summary
    # ==========================================================================

    n_signals: int = Field(
        ...,
        description="Number of signals tested",
    )

    n_samples: int = Field(
        ...,
        description="Number of time periods used",
    )

    # ==========================================================================
    # RAS Parameters
    # ==========================================================================

    delta: float = Field(
        ...,
        description="Significance level used (1-delta = confidence)",
    )

    kappa: float = Field(
        ...,
        description="IC bound used (|IC| <= kappa)",
    )

    n_simulations: int = Field(
        ...,
        description="Monte Carlo simulations used",
    )

    # ==========================================================================
    # Results
    # ==========================================================================

    rademacher_complexity: float = Field(
        ...,
        description="Empirical Rademacher complexity R^",
    )

    massart_bound: float = Field(
        ...,
        description="Massart's theoretical upper bound sqrt(2logN/T)",
    )

    observed_ic: dict[str, float] = Field(
        ...,
        description="Observed IC for each signal",
    )

    adjusted_ic: dict[str, float] = Field(
        ...,
        description="RAS-adjusted conservative IC lower bounds",
    )

    is_significant: dict[str, bool] = Field(
        ...,
        description="Whether adjusted IC > 0 for each signal",
    )

    # ==========================================================================
    # Summary Statistics
    # ==========================================================================

    n_significant: int = Field(
        ...,
        description="Number of signals with adjusted IC > 0",
    )

    any_significant: bool = Field(
        ...,
        description="Whether any signal passed RAS test",
    )

    data_snooping_term: float = Field(
        ...,
        description="Data snooping penalty (2 * R^)",
    )

    estimation_error_term: float = Field(
        ...,
        description="Estimation error term (2*kappa*sqrt(log(2/delta)/T))",
    )

    # ==========================================================================
    # Validation
    # ==========================================================================

    @model_validator(mode="after")
    def _validate_signal_keys(self) -> RASICResult:
        """Validate that all signal-keyed dicts share the same keys."""
        data = self.model_dump()
        _validate_dict_keys_match(
            data,
            required_fields=["observed_ic", "adjusted_ic", "is_significant"],
            reference_field="observed_ic",
        )
        return self

    # ==========================================================================
    # Methods
    # ==========================================================================

    def get_dataframe(self, name: str | None = None) -> pl.DataFrame:
        """Get results as Polars DataFrame.

        Parameters
        ----------
        name : str | None
            DataFrame to retrieve:
            - None or "signals": Per-signal results
            - "summary": Summary statistics

        Returns
        -------
        pl.DataFrame
            Requested DataFrame
        """
        if name is None or name == "signals":
            signals = list(self.observed_ic.keys())
            return pl.DataFrame(
                {
                    "signal": signals,
                    "observed_ic": [self.observed_ic[s] for s in signals],
                    "adjusted_ic": [self.adjusted_ic[s] for s in signals],
                    "is_significant": [self.is_significant[s] for s in signals],
                }
            )

        if name == "summary":
            return pl.DataFrame(
                {
                    "metric": [
                        "n_signals",
                        "n_samples",
                        "rademacher_complexity",
                        "massart_bound",
                        "data_snooping_term",
                        "estimation_error_term",
                        "n_significant",
                    ],
                    "value": [
                        float(self.n_signals),
                        float(self.n_samples),
                        self.rademacher_complexity,
                        self.massart_bound,
                        self.data_snooping_term,
                        self.estimation_error_term,
                        float(self.n_significant),
                    ],
                }
            )

        raise ValueError(f"Unknown DataFrame name: {name}. Available: 'signals', 'summary'")

    def list_available_dataframes(self) -> list[str]:
        """List available DataFrame views."""
        return ["signals", "summary"]

    def summary(self) -> str:
        """Get human-readable summary of RAS IC results."""
        lines = [
            "=" * 60,
            "RAS IC Analysis Summary",
            "=" * 60,
            "",
            f"Signals Tested:       {self.n_signals:>10}",
            f"Time Periods:         {self.n_samples:>10}",
            f"Confidence Level:     {1 - self.delta:>10.1%}",
            f"IC Bound (kappa):     {self.kappa:>10.4f}",
            "",
            f"Rademacher Complexity:{self.rademacher_complexity:>10.4f}",
            f"Massart Bound:        {self.massart_bound:>10.4f}",
            f"Data Snooping Term:   {self.data_snooping_term:>10.4f}",
            f"Estimation Error:     {self.estimation_error_term:>10.4f}",
            "",
            f"Significant Signals:  {self.n_significant:>10} / {self.n_signals}",
            "",
        ]

        if self.any_significant:
            lines.append("Significant signals (RAS-adjusted IC > 0):")
            for signal, sig in self.is_significant.items():
                if sig:
                    obs = self.observed_ic[signal]
                    adj = self.adjusted_ic[signal]
                    lines.append(f"  {signal}: observed={obs:.4f}, adjusted={adj:.4f}")

        return "\n".join(lines)
