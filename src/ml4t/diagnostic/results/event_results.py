"""Event Study Result Classes.

This module provides result containers for event study analysis,
storing abnormal returns, cumulative abnormal returns, and
statistical test results.

Classes
-------
AbnormalReturnResult
    Per-event abnormal return results
EventStudyResult
    Aggregated event study results with CAAR and statistics
"""

from __future__ import annotations

from typing import Any

import polars as pl
from pydantic import Field

from ml4t.diagnostic.results.base import BaseResult


class AbnormalReturnResult(BaseResult):
    """Per-event abnormal return results.

    Stores abnormal returns for a single event across the event window,
    including cumulative abnormal returns (CAR).

    Attributes
    ----------
    event_id : str
        Unique identifier for the event
    asset : str
        Asset/security identifier
    event_date : str
        Date of the event (ISO format)
    ar_by_day : dict[int, float]
        Abnormal returns by relative day {-5: 0.01, -4: -0.005, ...}
    car : float
        Cumulative abnormal return over event window
    estimation_alpha : float | None
        Market model alpha (if market_model used)
    estimation_beta : float | None
        Market model beta (if market_model used)
    estimation_r2 : float | None
        Market model R-squared (if market_model used)
    estimation_residual_std : float | None
        Estimation period residual std (for standardization)

    Examples
    --------
    >>> result = AbnormalReturnResult(
    ...     event_id="EVT001",
    ...     asset="AAPL",
    ...     event_date="2023-06-15",
    ...     ar_by_day={-5: 0.01, -4: 0.005, -3: -0.002, -2: 0.008, -1: 0.015,
    ...                0: 0.05, 1: 0.02, 2: -0.01, 3: 0.005, 4: 0.002, 5: -0.003},
    ...     car=0.10
    ... )
    """

    analysis_type: str = Field(default="abnormal_return", description="Result type")

    event_id: str = Field(..., description="Unique event identifier")
    asset: str = Field(..., description="Asset/security identifier")
    event_date: str = Field(..., description="Event date (ISO format)")
    ar_by_day: dict[int, float] = Field(
        ...,
        description="Abnormal returns by relative day",
    )
    car: float = Field(..., description="Cumulative abnormal return")

    # Market model parameters (optional)
    estimation_alpha: float | None = Field(
        default=None,
        description="Market model intercept (alpha)",
    )
    estimation_beta: float | None = Field(
        default=None,
        description="Market model slope (beta)",
    )
    estimation_r2: float | None = Field(
        default=None,
        description="Market model R-squared",
    )
    estimation_residual_std: float | None = Field(
        default=None,
        description="Estimation period residual standard deviation",
    )

    def get_dataframe(self, name: str | None = None) -> pl.DataFrame:
        """Get abnormal returns as DataFrame.

        Returns
        -------
        pl.DataFrame
            DataFrame with columns: relative_day, abnormal_return
        """
        return pl.DataFrame(
            {
                "relative_day": list(self.ar_by_day.keys()),
                "abnormal_return": list(self.ar_by_day.values()),
            }
        ).sort("relative_day")

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            f"Event: {self.event_id}",
            f"Asset: {self.asset}",
            f"Date: {self.event_date}",
            f"CAR: {self.car:.4f} ({self.car * 100:.2f}%)",
        ]
        if self.estimation_beta is not None:
            lines.append(f"Beta: {self.estimation_beta:.3f}")
        return "\n".join(lines)


class EventStudyResult(BaseResult):
    """Complete event study results with aggregated statistics.

    Contains average abnormal returns (AAR), cumulative average
    abnormal returns (CAAR), confidence intervals, and statistical
    test results.

    Attributes
    ----------
    aar_by_day : dict[int, float]
        Average abnormal return by relative day (cross-sectional mean)
    caar : list[float]
        Cumulative AAR time series
    caar_dates : list[int]
        Relative days corresponding to CAAR values
    caar_std : list[float]
        Standard deviation of CAAR at each point
    caar_ci_lower : list[float]
        Lower confidence interval bound
    caar_ci_upper : list[float]
        Upper confidence interval bound
    test_statistic : float
        Test statistic value (t-stat, BMP, or Corrado)
    p_value : float
        P-value for the test
    test_name : str
        Name of statistical test used
    n_events : int
        Number of events in the study
    model_name : str
        Name of model used (market_model, mean_adjusted, market_adjusted)
    event_window : tuple[int, int]
        Event window used
    confidence_level : float
        Confidence level for intervals
    individual_results : list[AbnormalReturnResult] | None
        Optional individual event results

    Examples
    --------
    >>> result = EventStudyResult(
    ...     aar_by_day={-5: 0.001, ..., 0: 0.025, ..., 5: -0.002},
    ...     caar=[0.001, 0.003, 0.008, ...],
    ...     caar_dates=[-5, -4, -3, ...],
    ...     caar_std=[0.01, 0.012, ...],
    ...     caar_ci_lower=[...],
    ...     caar_ci_upper=[...],
    ...     test_statistic=2.45,
    ...     p_value=0.014,
    ...     test_name="boehmer",
    ...     n_events=50
    ... )
    """

    analysis_type: str = Field(default="event_study", description="Result type")

    # Average abnormal returns
    aar_by_day: dict[int, float] = Field(
        ...,
        description="Average abnormal return by relative day",
    )

    # Cumulative average abnormal returns
    caar: list[float] = Field(
        ...,
        description="Cumulative AAR time series",
    )
    caar_dates: list[int] = Field(
        ...,
        description="Relative days for CAAR values",
    )
    caar_std: list[float] = Field(
        ...,
        description="Standard deviation of CAAR",
    )
    caar_ci_lower: list[float] = Field(
        ...,
        description="Lower confidence interval",
    )
    caar_ci_upper: list[float] = Field(
        ...,
        description="Upper confidence interval",
    )

    # Statistical test results
    test_statistic: float = Field(
        ...,
        description="Test statistic value",
    )
    p_value: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="P-value for significance test",
    )
    test_name: str = Field(
        ...,
        description="Name of statistical test (t_test, boehmer, corrado)",
    )

    # Metadata
    n_events: int = Field(
        ...,
        ge=1,
        description="Number of events analyzed",
    )
    model_name: str = Field(
        default="market_model",
        description="Model used for expected returns",
    )
    event_window: tuple[int, int] = Field(
        default=(-5, 5),
        description="Event window (start, end)",
    )
    confidence_level: float = Field(
        default=0.95,
        description="Confidence level for intervals",
    )

    # Optional detailed results
    individual_results: list[AbnormalReturnResult] | None = Field(
        default=None,
        description="Individual event results (optional)",
    )

    @property
    def is_significant(self) -> bool:
        """Whether CAAR is statistically significant at the configured level."""
        return self.p_value < (1 - self.confidence_level)

    @property
    def final_caar(self) -> float:
        """CAAR at the end of the event window."""
        return self.caar[-1] if self.caar else 0.0

    @property
    def event_day_aar(self) -> float:
        """AAR on the event day (t=0)."""
        return self.aar_by_day.get(0, 0.0)

    def get_dataframe(self, name: str | None = None) -> pl.DataFrame:
        """Get results as DataFrame.

        Parameters
        ----------
        name : str | None
            DataFrame name: "caar" (default), "aar", or "events"

        Returns
        -------
        pl.DataFrame
            Requested DataFrame
        """
        if name is None or name == "caar":
            return pl.DataFrame(
                {
                    "relative_day": self.caar_dates,
                    "caar": self.caar,
                    "caar_std": self.caar_std,
                    "ci_lower": self.caar_ci_lower,
                    "ci_upper": self.caar_ci_upper,
                }
            )
        elif name == "aar":
            return pl.DataFrame(
                {
                    "relative_day": list(self.aar_by_day.keys()),
                    "aar": list(self.aar_by_day.values()),
                }
            ).sort("relative_day")
        elif name == "events" and self.individual_results:
            return pl.DataFrame(
                [
                    {
                        "event_id": r.event_id,
                        "asset": r.asset,
                        "event_date": r.event_date,
                        "car": r.car,
                    }
                    for r in self.individual_results
                ]
            )
        else:
            raise ValueError(f"Unknown DataFrame name: {name}. Available: caar, aar, events")

    def list_available_dataframes(self) -> list[str]:
        """List available DataFrame views."""
        views = ["caar", "aar"]
        if self.individual_results:
            views.append("events")
        return views

    def summary(self) -> str:
        """Human-readable summary of event study results."""
        significance = "significant" if self.is_significant else "not significant"
        alpha = 1 - self.confidence_level

        lines = [
            "=" * 50,
            "EVENT STUDY RESULTS",
            "=" * 50,
            f"Events analyzed: {self.n_events}",
            f"Event window: [{self.event_window[0]}, {self.event_window[1]}]",
            f"Model: {self.model_name}",
            "",
            "CUMULATIVE AVERAGE ABNORMAL RETURN (CAAR)",
            f"  Event day AAR (t=0): {self.event_day_aar:+.4f} ({self.event_day_aar * 100:+.2f}%)",
            f"  Final CAAR: {self.final_caar:+.4f} ({self.final_caar * 100:+.2f}%)",
            f"  95% CI: [{self.caar_ci_lower[-1]:.4f}, {self.caar_ci_upper[-1]:.4f}]",
            "",
            "STATISTICAL TEST",
            f"  Test: {self.test_name}",
            f"  Test statistic: {self.test_statistic:.4f}",
            f"  P-value: {self.p_value:.4f}",
            f"  Result: {significance} at α={alpha:.2f}",
            "=" * 50,
        ]
        return "\n".join(lines)

    def to_dict(self, *, exclude_none: bool = False) -> dict[str, Any]:
        """Export to dictionary.

        Overridden to handle individual_results serialization.
        """
        data = super().to_dict(exclude_none=exclude_none)
        # Convert individual results if present
        if self.individual_results and "individual_results" in data:
            data["individual_results"] = [r.to_dict() for r in self.individual_results]
        return data
