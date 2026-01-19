"""Statistical Testing Configuration.

This module provides configuration for advanced statistical testing:
- **PSR**: Probabilistic Sharpe Ratio (confidence in positive Sharpe)
- **MinTRL**: Minimum Track Record Length (required sample size)
- **DSR**: Deflated Sharpe Ratio (correction for multiple testing)
- **FDR**: False Discovery Rate control (family-wise error rate)

These methods address the critical problem of overfitting and false discoveries
in quantitative strategy research.

Consolidated Config:
- StatisticalConfig: Single config with all statistical test settings
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator

from ml4t.diagnostic.config.base import BaseConfig
from ml4t.diagnostic.config.validation import (
    FDRMethod,
    NonNegativeFloat,
    PositiveFloat,
    PositiveInt,
    Probability,
)

# =============================================================================
# Settings Classes (Single-Level Nesting)
# =============================================================================


class PSRSettings(BaseConfig):
    """Settings for Probabilistic Sharpe Ratio.

    PSR computes the probability that the true Sharpe ratio exceeds a threshold,
    accounting for higher moments (skewness, kurtosis) and estimation uncertainty.
    """

    enabled: bool = Field(True, description="Compute PSR")
    confidence_level: Probability = Field(0.95, description="Confidence level")
    target_sharpe: NonNegativeFloat = Field(0.0, description="Target SR to test against")
    adjustment_factor: PositiveFloat | Literal["auto"] = Field(
        "auto", description="Higher moment adjustment"
    )
    compute_for_thresholds: list[float] | None = Field(None, description="Multiple target values")


class MinTRLSettings(BaseConfig):
    """Settings for Minimum Track Record Length.

    MinTRL computes the minimum sample size required to be confident
    that the true Sharpe ratio exceeds a target value.
    """

    enabled: bool = Field(True, description="Compute MinTRL")
    confidence_level: Probability = Field(0.95, description="Confidence level")
    target_sharpe: NonNegativeFloat = Field(0.0, description="Target SR to detect")
    compute_for_thresholds: list[float] | None = Field(None, description="Multiple target values")


class DSRSettings(BaseConfig):
    """Settings for Deflated Sharpe Ratio.

    DSR corrects for multiple testing bias when evaluating many strategies.
    """

    enabled: bool = Field(True, description="Compute DSR")
    n_trials: PositiveInt = Field(100, description="Number of strategies tested")
    prob_zero_sharpe: Probability = Field(0.5, description="Prior probability SR=0")
    variance_inflation: PositiveFloat = Field(1.0, description="Variance inflation factor")
    expected_max_sharpe: float | Literal["auto"] = Field(
        "auto", description="Expected max SR under null"
    )

    @field_validator("n_trials")
    @classmethod
    def check_n_trials(cls, v: int) -> int:
        """Warn if n_trials is suspiciously low."""
        if v < 10:
            import warnings

            warnings.warn(
                f"n_trials={v} seems low. Include ALL strategies tested.",
                stacklevel=2,
            )
        return v


class FDRSettings(BaseConfig):
    """Settings for False Discovery Rate control.

    FDR controls the expected proportion of false discoveries among all
    rejected hypotheses.
    """

    enabled: bool = Field(True, description="Apply FDR control")
    alpha: Probability = Field(0.05, description="Family-wise error rate")
    method: FDRMethod = Field(FDRMethod.BENJAMINI_HOCHBERG, description="FDR method")
    independent_tests: bool = Field(False, description="Are tests independent?")

    @model_validator(mode="after")
    def validate_method_independence(self) -> FDRSettings:
        """Warn if using BH with correlated tests."""
        if self.method == FDRMethod.BENJAMINI_HOCHBERG and not self.independent_tests:
            import warnings

            warnings.warn(
                "Benjamini-Hochberg assumes independence. Consider BY method.",
                stacklevel=2,
            )
        return self


# =============================================================================
# Consolidated Config
# =============================================================================


class StatisticalConfig(BaseConfig):
    """Consolidated configuration for statistical testing.

    Orchestrates advanced Sharpe ratio analysis with multiple testing correction.

    Examples
    --------
    >>> config = StatisticalConfig(
    ...     psr=PSRSettings(target_sharpe=1.0),
    ...     dsr=DSRSettings(n_trials=500),
    ... )
    >>> # Or use presets
    >>> config = StatisticalConfig.for_research()
    """

    psr: PSRSettings = Field(default_factory=PSRSettings, description="PSR settings")
    mintrl: MinTRLSettings = Field(default_factory=MinTRLSettings, description="MinTRL settings")
    dsr: DSRSettings = Field(default_factory=DSRSettings, description="DSR settings")
    fdr: FDRSettings = Field(default_factory=FDRSettings, description="FDR settings")

    # Output settings
    return_dataframes: bool = Field(True, description="Return as DataFrames")
    cache_enabled: bool = Field(True, description="Enable caching")
    cache_dir: Path = Field(
        default_factory=lambda: Path.home() / ".cache" / "ml4t-diagnostic" / "sharpe",
        description="Cache directory",
    )
    verbose: bool = Field(False, description="Verbose output")

    @classmethod
    def for_quick_check(cls) -> StatisticalConfig:
        """Preset for quick overfitting check (PSR + DSR only)."""
        return cls(
            psr=PSRSettings(compute_for_thresholds=None),
            mintrl=MinTRLSettings(enabled=False),
            dsr=DSRSettings(n_trials=100),
            fdr=FDRSettings(enabled=False),
        )

    @classmethod
    def for_research(cls) -> StatisticalConfig:
        """Preset for academic research (comprehensive analysis)."""
        return cls(
            psr=PSRSettings(
                compute_for_thresholds=[0.0, 0.5, 1.0, 1.5, 2.0],
                confidence_level=0.99,
            ),
            mintrl=MinTRLSettings(compute_for_thresholds=[0.0, 0.5, 1.0]),
            dsr=DSRSettings(n_trials=500, prob_zero_sharpe=0.5),
            fdr=FDRSettings(
                method=FDRMethod.BENJAMINI_YEKUTIELI,
                alpha=0.05,
            ),
        )

    @classmethod
    def for_publication(cls) -> StatisticalConfig:
        """Preset for academic publication (very conservative)."""
        return cls(
            psr=PSRSettings(confidence_level=0.99, target_sharpe=0.5),
            mintrl=MinTRLSettings(confidence_level=0.99, target_sharpe=0.5),
            dsr=DSRSettings(
                n_trials=1000,
                prob_zero_sharpe=0.8,
                variance_inflation=1.5,
            ),
            fdr=FDRSettings(
                method=FDRMethod.BONFERRONI,
                alpha=0.01,
            ),
        )


# Rebuild models
PSRSettings.model_rebuild()
MinTRLSettings.model_rebuild()
DSRSettings.model_rebuild()
FDRSettings.model_rebuild()
StatisticalConfig.model_rebuild()
