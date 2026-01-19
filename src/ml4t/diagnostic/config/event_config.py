"""Event Study Configuration.

This module provides configuration for event study analysis following
MacKinlay (1997) "Event Studies in Economics and Finance".

Consolidated Config:
- EventConfig: Full configuration with window settings inlined
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, field_validator

from ml4t.diagnostic.config.base import BaseConfig


class WindowSettings(BaseConfig):
    """Settings for event study windows.

    Defines the estimation window (for computing normal returns) and
    the event window (for measuring abnormal returns).
    """

    estimation_start: int = Field(
        default=-252,
        le=-1,
        description="Estimation window start relative to t=0",
    )
    estimation_end: int = Field(
        default=-20,
        le=-1,
        description="Estimation window end relative to t=0 (must be negative)",
    )
    event_start: int = Field(
        default=-5,
        description="Event window start relative to t=0",
    )
    event_end: int = Field(
        default=5,
        description="Event window end relative to t=0",
    )
    gap: int = Field(
        default=5,
        ge=0,
        description="Buffer days between estimation and event windows",
    )

    @field_validator("estimation_end")
    @classmethod
    def validate_estimation_end(cls, v: int, info) -> int:
        """Ensure estimation window is properly ordered."""
        if info.data.get("estimation_start") is not None:
            if info.data["estimation_start"] >= v:
                raise ValueError(
                    f"estimation_start ({info.data['estimation_start']}) must be < estimation_end ({v})"
                )
        return v

    @field_validator("event_end")
    @classmethod
    def validate_event_end(cls, v: int, info) -> int:
        """Ensure event window is properly ordered."""
        if info.data.get("event_start") is not None:
            if info.data["event_start"] >= v:
                raise ValueError(
                    f"event_start ({info.data['event_start']}) must be < event_end ({v})"
                )
        return v

    @property
    def estimation_window(self) -> tuple[int, int]:
        """Estimation window as tuple for backward compatibility."""
        return (self.estimation_start, self.estimation_end)

    @property
    def event_window(self) -> tuple[int, int]:
        """Event window as tuple for backward compatibility."""
        return (self.event_start, self.event_end)

    @property
    def estimation_length(self) -> int:
        """Length of estimation window in days."""
        return self.estimation_end - self.estimation_start

    @property
    def event_length(self) -> int:
        """Length of event window in days."""
        return self.event_end - self.event_start + 1


class EventConfig(BaseConfig):
    """Configuration for event study analysis.

    Configures the event study methodology including window parameters,
    abnormal return model, and statistical test.

    Attributes
    ----------
    window : WindowSettings
        Window configuration (estimation and event periods)
    model : str
        Model for computing normal/expected returns
    test : str
        Statistical test for significance
    confidence_level : float
        Confidence level for intervals
    min_estimation_obs : int
        Minimum observations in estimation window

    Examples
    --------
    >>> config = EventConfig(
    ...     window=WindowSettings(estimation_start=-252, event_end=10),
    ...     model="market_model",
    ...     test="boehmer",
    ... )
    """

    window: WindowSettings = Field(
        default_factory=WindowSettings,
        description="Window configuration",
    )
    model: Literal["market_model", "mean_adjusted", "market_adjusted"] = Field(
        default="market_model",
        description="Model for computing expected returns",
    )
    test: Literal["t_test", "boehmer", "corrado"] = Field(
        default="boehmer",
        description="Statistical test for significance",
    )
    confidence_level: float = Field(
        default=0.95,
        gt=0.0,
        lt=1.0,
        description="Confidence level for intervals",
    )
    min_estimation_obs: int = Field(
        default=100,
        ge=30,
        description="Minimum observations in estimation window",
    )

    @property
    def alpha(self) -> float:
        """Significance level (1 - confidence_level)."""
        return 1.0 - self.confidence_level
