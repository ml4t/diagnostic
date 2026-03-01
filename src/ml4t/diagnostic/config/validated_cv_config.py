"""Configuration for validated cross-validation workflows."""

from __future__ import annotations

from pydantic import Field

from ml4t.diagnostic.config.base import BaseConfig


class ValidatedCrossValidationConfig(BaseConfig):
    """Configuration for `ValidatedCrossValidation` orchestration."""

    # CV parameters
    n_groups: int = Field(default=10, ge=2, description="Number of CV groups")
    n_test_groups: int = Field(default=2, ge=1, description="Groups per test set")
    embargo_pct: float = Field(default=0.01, ge=0, le=0.2, description="Embargo fraction")
    label_horizon: int = Field(default=0, ge=0, description="Label look-ahead samples")

    # DSR parameters
    sharpe_star: float = Field(default=0.0, description="Benchmark Sharpe ratio")
    significance_level: float = Field(default=0.95, ge=0.5, le=0.999)
    annualization_factor: float = Field(default=252.0, gt=0, description="Sharpe annualization")

    # Execution
    random_state: int | None = Field(default=None)
