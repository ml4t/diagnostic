"""Event Study Analysis Module.

This module implements event study methodology following MacKinlay (1997)
"Event Studies in Economics and Finance" for measuring abnormal returns
around corporate events, announcements, or other market events.

Classes
-------
EventStudyAnalysis
    Main class for conducting event studies

References
----------
MacKinlay, A.C. (1997). "Event Studies in Economics and Finance",
    Journal of Economic Literature, 35(1), 13-39.
Boehmer, E., Musumeci, J., Poulsen, A.B. (1991). "Event-study methodology
    under conditions of event-induced variance", Journal of Financial Economics.
"""

from __future__ import annotations

import warnings
from collections import Counter
from enum import StrEnum
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl
from scipy import stats

from ml4t.diagnostic.backends.adapter import DataFrameAdapter
from ml4t.diagnostic.config.event_config import EventConfig
from ml4t.diagnostic.results.event_results import AbnormalReturnResult, EventStudyResult

if TYPE_CHECKING:
    import pandas as pd


class EventRejectionReason(StrEnum):
    """Reason an event cannot enter fixed-horizon inference."""

    UNKNOWN_EVENT_DATE = "unknown event date"
    UNKNOWN_ASSET = "unknown asset"
    UNKNOWN_ASSET_AND_DATE = "unknown asset and event date"
    ESTIMATION_HISTORY = "insufficient estimation history"
    ASSET_ESTIMATION = "insufficient finite asset estimation returns"
    BENCHMARK_ESTIMATION = "insufficient finite benchmark estimation returns"
    ASSET_AND_BENCHMARK_ESTIMATION = "insufficient finite asset and benchmark estimation returns"
    ALIGNED_ESTIMATION = "insufficient aligned finite asset/benchmark estimation returns"
    EVENT_WINDOW_HISTORY = "event window extends beyond returns history"
    ASSET_EVENT_WINDOW = "incomplete or non-finite asset event window"
    BENCHMARK_EVENT_WINDOW = "incomplete or non-finite benchmark event window"
    ASSET_AND_BENCHMARK_EVENT_WINDOW = "incomplete or non-finite asset and benchmark event window"


class EventStudyAnalysis:
    """Event study analysis for measuring abnormal returns around events.

    Implements the standard event study methodology with support for:
    - Market model (CAPM-based expected returns)
    - Mean-adjusted model
    - Market-adjusted model

    And statistical tests:
    - Standard t-test
    - BMP test (Boehmer et al. 1991, robust to event-induced variance)

    Parameters
    ----------
    returns : pl.DataFrame
        Asset returns in long format with columns: [date, asset, return].
        Returns should be simple returns (not log returns).
    events : pl.DataFrame
        Events to analyze with columns: [date, asset]. Optionally
        includes [event_type, event_id] for grouping.
    benchmark : pl.DataFrame
        Market/benchmark returns with columns: [date, return].
    config : EventConfig, optional
        Configuration for the analysis. Event windows must be complete and
        finite so every CAR covers the configured horizon.

    Examples
    --------
    >>> returns_df = pl.DataFrame({
    ...     'date': [...],
    ...     'asset': [...],
    ...     'return': [...]
    ... })
    >>> events_df = pl.DataFrame({
    ...     'date': ['2023-01-15', '2023-02-20'],
    ...     'asset': ['AAPL', 'MSFT']
    ... })
    >>> benchmark_df = pl.DataFrame({
    ...     'date': [...],
    ...     'return': [...]  # Market returns
    ... })
    >>> analysis = EventStudyAnalysis(returns_df, events_df, benchmark_df)
    >>> result = analysis.run()
    >>> print(result.summary())
    """

    def __init__(
        self,
        returns: pl.DataFrame | pd.DataFrame,
        events: pl.DataFrame | pd.DataFrame,
        benchmark: pl.DataFrame | pd.DataFrame,
        config: EventConfig | None = None,
    ) -> None:
        """Initialize event study analysis."""
        self.config = config or EventConfig()

        # Convert to Polars if needed
        self._returns = self._to_polars(returns)
        self._events = self._to_polars(events)
        self._benchmark = self._to_polars(benchmark)

        # Validate inputs
        self._validate_inputs()

        # Prepare data
        self._prepare_data()

        # Cache for computed results
        self._ar_results: list[AbnormalReturnResult] | None = None
        self._aggregated_result: EventStudyResult | None = None

    def _to_polars(self, df: Any) -> pl.DataFrame:
        """Convert DataFrame to Polars if needed."""
        try:
            converted, _ = DataFrameAdapter.to_polars(df)
            return converted
        except TypeError:
            raise TypeError(f"Expected Polars or Pandas DataFrame, got {type(df)}") from None

    def _validate_inputs(self) -> None:
        """Validate input DataFrames have required columns."""
        # Check returns
        required_return_cols = {"date", "asset", "return"}
        if not required_return_cols.issubset(set(self._returns.columns)):
            raise ValueError(
                f"returns DataFrame missing columns: {required_return_cols - set(self._returns.columns)}"
            )

        # Check events
        required_event_cols = {"date", "asset"}
        if not required_event_cols.issubset(set(self._events.columns)):
            raise ValueError(
                f"events DataFrame missing columns: {required_event_cols - set(self._events.columns)}"
            )

        # Check benchmark
        required_bench_cols = {"date", "return"}
        if not required_bench_cols.issubset(set(self._benchmark.columns)):
            raise ValueError(
                f"benchmark DataFrame missing columns: {required_bench_cols - set(self._benchmark.columns)}"
            )

        # Check we have events
        if len(self._events) == 0:
            raise ValueError("No events provided")

    def _prepare_data(self) -> None:
        """Prepare data for analysis (sorting, date alignment)."""
        # Sort by date
        self._returns = self._returns.sort("date")
        self._benchmark = self._benchmark.sort("date")

        # Create date-indexed lookup for benchmark
        self._benchmark_dict: dict[Any, float] = dict(
            zip(
                self._benchmark["date"].to_list(),
                self._benchmark["return"].to_list(),
                strict=False,
            )
        )

        # Get unique dates for index mapping
        self._all_dates = sorted(self._returns["date"].unique().to_list())
        self._date_to_idx = {d: i for i, d in enumerate(self._all_dates)}
        self._return_assets = set(self._returns["asset"].unique().to_list())

        # Add event_id if not present
        if "event_id" not in self._events.columns:
            self._events = self._events.with_row_index("event_id").with_columns(
                pl.col("event_id").cast(pl.Utf8).alias("event_id")
            )

    def _get_estimation_window_data(
        self, asset: str, event_date: Any
    ) -> tuple[
        tuple[np.ndarray, np.ndarray | None] | None,
        EventRejectionReason | None,
    ]:
        """Get returns for estimation window.

        Returns
        -------
        tuple[tuple[np.ndarray, np.ndarray | None] | None, EventRejectionReason | None]
            Estimation data and no rejection reason, or no data and the specific
            reason the configured minimum could not be met.
        """
        est_start, est_end = self.config.window.estimation_window

        assert event_date in self._date_to_idx
        event_idx = self._date_to_idx[event_date]

        # Calculate estimation window indices
        start_idx = event_idx + est_start
        end_idx = event_idx + est_end

        if start_idx < 0:
            return None, EventRejectionReason.ESTIMATION_HISTORY

        # Get dates in estimation window
        est_dates = self._all_dates[start_idx : end_idx + 1]

        if len(est_dates) < self.config.min_estimation_obs:
            return None, EventRejectionReason.ESTIMATION_HISTORY

        # Get asset returns
        asset_data = self._returns.filter(
            (pl.col("asset") == asset) & (pl.col("date").is_in(est_dates))
        ).sort("date")

        asset_returns_by_date: dict[Any, float] = {}
        for row in asset_data.iter_rows(named=True):
            asset_return = row["return"]
            if asset_return is not None and np.isfinite(asset_return):
                asset_returns_by_date[row["date"]] = asset_return

        if self.config.model == "mean_adjusted":
            if len(asset_returns_by_date) < self.config.min_estimation_obs:
                return None, EventRejectionReason.ASSET_ESTIMATION
            return (np.array(list(asset_returns_by_date.values())), None), None

        benchmark_returns_by_date = {
            date: benchmark_return
            for date in est_dates
            if (benchmark_return := self._benchmark_dict.get(date)) is not None
            and np.isfinite(benchmark_return)
        }
        missing_asset = len(asset_returns_by_date) < self.config.min_estimation_obs
        missing_benchmark = len(benchmark_returns_by_date) < self.config.min_estimation_obs
        if missing_asset and missing_benchmark:
            return None, EventRejectionReason.ASSET_AND_BENCHMARK_ESTIMATION
        if missing_asset:
            return None, EventRejectionReason.ASSET_ESTIMATION
        if missing_benchmark:
            return None, EventRejectionReason.BENCHMARK_ESTIMATION

        paired_dates = [
            date
            for date in est_dates
            if date in asset_returns_by_date and date in benchmark_returns_by_date
        ]
        if len(paired_dates) < self.config.min_estimation_obs:
            return None, EventRejectionReason.ALIGNED_ESTIMATION

        return (
            (
                np.array([asset_returns_by_date[date] for date in paired_dates]),
                np.array([benchmark_returns_by_date[date] for date in paired_dates]),
            ),
            None,
        )

    def _estimate_market_model(
        self, asset_returns: np.ndarray, market_returns: np.ndarray
    ) -> tuple[float, float, float, float]:
        """Estimate market model parameters via OLS.

        AR = R - (α + β*Rm)

        Returns
        -------
        tuple[float, float, float, float]
            (alpha, beta, r_squared, residual_std)
        """
        # OLS regression: R_asset = alpha + beta * R_market + epsilon
        X = np.column_stack([np.ones(len(market_returns)), market_returns])
        y = asset_returns

        # Solve normal equations
        try:
            coeffs, residuals, _, _ = np.linalg.lstsq(X, y, rcond=None)
            alpha, beta = coeffs[0], coeffs[1]

            # Calculate R-squared
            y_pred = alpha + beta * market_returns
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

            # Residual standard deviation
            residual_std = np.std(y - y_pred, ddof=2)

            return alpha, beta, r_squared, residual_std
        except Exception:
            return 0.0, 1.0, 0.0, np.std(asset_returns)

    def _get_event_window_data(
        self, asset: str, event_date: Any
    ) -> tuple[
        dict[int, tuple[float, float | None]] | None,
        EventRejectionReason | None,
    ]:
        """Get finite returns for an event window.

        Incomplete windows are rejected so every CAR covers the same horizon.

        Returns
        -------
        tuple[dict[int, tuple[float, float | None]] | None, EventRejectionReason | None]
            Complete event-window data and no rejection reason, or no data and
            the specific incomplete input.
        """
        evt_start, evt_end = self.config.window.event_window

        assert event_date in self._date_to_idx
        event_idx = self._date_to_idx[event_date]

        required_dates: dict[int, Any] = {}
        for rel_day in range(evt_start, evt_end + 1):
            day_idx = event_idx + rel_day
            if not 0 <= day_idx < len(self._all_dates):
                return None, EventRejectionReason.EVENT_WINDOW_HISTORY
            required_dates[rel_day] = self._all_dates[day_idx]

        window_dates = list(required_dates.values())
        asset_data = self._returns.filter(
            (pl.col("asset") == asset) & (pl.col("date").is_in(window_dates))
        )
        asset_returns_by_date = {
            row["date"]: row["return"]
            for row in asset_data.iter_rows(named=True)
            if row["return"] is not None and np.isfinite(row["return"])
        }
        missing_asset = len(asset_returns_by_date) != self.config.window.event_length

        if self.config.model == "mean_adjusted":
            if missing_asset:
                return None, EventRejectionReason.ASSET_EVENT_WINDOW
            return {
                rel_day: (asset_returns_by_date[date], None)
                for rel_day, date in required_dates.items()
            }, None

        benchmark_returns_by_date = {
            date: benchmark_return
            for date in window_dates
            if (benchmark_return := self._benchmark_dict.get(date)) is not None
            and np.isfinite(benchmark_return)
        }
        missing_benchmark = len(benchmark_returns_by_date) != self.config.window.event_length
        if missing_asset and missing_benchmark:
            return None, EventRejectionReason.ASSET_AND_BENCHMARK_EVENT_WINDOW
        if missing_asset:
            return None, EventRejectionReason.ASSET_EVENT_WINDOW
        if missing_benchmark:
            return None, EventRejectionReason.BENCHMARK_EVENT_WINDOW

        return {
            rel_day: (asset_returns_by_date[date], benchmark_returns_by_date[date])
            for rel_day, date in required_dates.items()
        }, None

    def _compute_abnormal_return_single(
        self, event_row: dict[str, Any]
    ) -> tuple[AbnormalReturnResult | None, EventRejectionReason | None]:
        """Compute abnormal returns for a single event."""
        asset = event_row["asset"]
        event_date = event_row["date"]
        event_id = str(event_row.get("event_id", f"{asset}_{event_date}"))

        unknown_event_date = event_date not in self._date_to_idx
        unknown_asset = asset not in self._return_assets
        if unknown_event_date and unknown_asset:
            return None, EventRejectionReason.UNKNOWN_ASSET_AND_DATE
        if unknown_event_date:
            return None, EventRejectionReason.UNKNOWN_EVENT_DATE
        if unknown_asset:
            return None, EventRejectionReason.UNKNOWN_ASSET
        # Get estimation window data
        est_data, rejection_reason = self._get_estimation_window_data(asset, event_date)
        if est_data is None:
            if rejection_reason is None:
                raise RuntimeError("Missing rejection reason for unavailable estimation data")
            return None, rejection_reason

        asset_est_returns, market_est_returns = est_data

        # Estimate model parameters
        alpha, beta, r2, residual_std = 0.0, 1.0, 0.0, 0.0

        if self.config.model == "market_model":
            assert market_est_returns is not None
            alpha, beta, r2, residual_std = self._estimate_market_model(
                asset_est_returns, market_est_returns
            )
        elif self.config.model == "mean_adjusted":
            alpha = float(np.mean(asset_est_returns))
            beta = 0.0
            residual_std = float(np.std(asset_est_returns, ddof=1))
        elif self.config.model == "market_adjusted":
            assert market_est_returns is not None
            alpha = 0.0
            beta = 1.0
            residual_std = float(np.std(asset_est_returns - market_est_returns, ddof=1))

        # Get event window data
        event_data, rejection_reason = self._get_event_window_data(asset, event_date)
        if event_data is None:
            if rejection_reason is None:
                raise RuntimeError("Missing rejection reason for unavailable event-window data")
            return None, rejection_reason

        # Compute abnormal returns
        ar_by_day: dict[int, float] = {}
        for rel_day, (asset_ret, market_ret) in event_data.items():
            if self.config.model == "market_model":
                assert market_ret is not None
                expected_ret = alpha + beta * market_ret
            elif self.config.model == "mean_adjusted":
                expected_ret = alpha
            else:
                assert market_ret is not None
                expected_ret = market_ret

            ar_by_day[rel_day] = asset_ret - expected_ret

        # Compute CAR
        car = sum(ar_by_day.values())

        return (
            AbnormalReturnResult(
                event_id=event_id,
                asset=asset,
                event_date=str(event_date),
                ar_by_day=ar_by_day,
                car=car,
                estimation_alpha=alpha if self.config.model == "market_model" else None,
                estimation_beta=beta if self.config.model == "market_model" else None,
                estimation_r2=r2 if self.config.model == "market_model" else None,
                estimation_residual_std=residual_std,
            ),
            None,
        )

    def compute_abnormal_returns(self) -> list[AbnormalReturnResult]:
        """Compute abnormal returns for all events.

        Returns
        -------
        list[AbnormalReturnResult]
            Abnormal return results for each valid event.
        """
        if self._ar_results is not None:
            return self._ar_results

        results = []
        rejected: Counter[EventRejectionReason] = Counter()

        for row in self._events.iter_rows(named=True):
            result, reason = self._compute_abnormal_return_single(row)
            if result is not None:
                results.append(result)
            else:
                if reason is None:
                    raise RuntimeError("Rejected event did not include a rejection reason")
                rejected[reason] += 1

        n_skipped = sum(rejected.values())
        if n_skipped > 0:
            details = ", ".join(
                f"{rejected[reason]}: {reason.value}"
                for reason in EventRejectionReason
                if rejected[reason]
            )
            event_label = "event" if n_skipped == 1 else "events"
            warnings.warn(
                f"Skipped {n_skipped} {event_label} ({details})",
                stacklevel=2,
            )

        self._ar_results = results
        return results

    def aggregate(self, group_by: str | None = None) -> EventStudyResult:
        """Aggregate individual results to AAR and CAAR.

        Parameters
        ----------
        group_by : str | None
            Column to group by (e.g., 'event_type'). If None,
            aggregates all events together.

        Returns
        -------
        EventStudyResult
            Aggregated event study results.
        """
        ar_results = self.compute_abnormal_returns()

        if len(ar_results) == 0:
            raise ValueError("No valid events to aggregate")

        # Collect all relative days
        all_days = set()
        for r in ar_results:
            all_days.update(r.ar_by_day.keys())
        sorted_days = sorted(all_days)

        # Compute AAR (average AR across events for each day)
        aar_by_day: dict[int, float] = {}
        ar_matrix: dict[int, list[float]] = {d: [] for d in sorted_days}

        for r in ar_results:
            for day in sorted_days:
                if day in r.ar_by_day:
                    ar_matrix[day].append(r.ar_by_day[day])

        for day in sorted_days:
            if ar_matrix[day]:
                aar_by_day[day] = float(np.mean(ar_matrix[day]))
            else:
                aar_by_day[day] = 0.0

        # Compute CAAR and its statistics
        caar_values = []
        caar_std = []
        cumsum = 0.0

        for day in sorted_days:
            cumsum += aar_by_day[day]
            caar_values.append(cumsum)

            # Cross-sectional standard deviation at this day
            if len(ar_matrix[day]) >= 2:
                caar_std.append(float(np.std(ar_matrix[day], ddof=1)))
            else:
                caar_std.append(float("nan"))

        # Compute confidence intervals
        n_events = len(ar_results)
        z_score = stats.norm.ppf(1 - self.config.alpha / 2)

        caar_ci_lower = []
        caar_ci_upper = []
        for caar, std in zip(caar_values, caar_std, strict=False):
            se = std / np.sqrt(n_events) if n_events > 0 else 0.0
            caar_ci_lower.append(caar - z_score * se)
            caar_ci_upper.append(caar + z_score * se)

        # Run statistical test
        test_stat, p_value = self._run_statistical_test(ar_results)

        result = EventStudyResult(
            aar_by_day=aar_by_day,
            caar=caar_values,
            caar_dates=sorted_days,
            caar_std=caar_std,
            caar_ci_lower=caar_ci_lower,
            caar_ci_upper=caar_ci_upper,
            test_statistic=test_stat,
            p_value=p_value,
            test_name=self.config.test,
            n_events=n_events,
            model_name=self.config.model,
            event_window=self.config.window.event_window,
            confidence_level=self.config.confidence_level,
            individual_results=ar_results,
        )

        self._aggregated_result = result
        return result

    def _run_statistical_test(
        self,
        ar_results: list[AbnormalReturnResult],
    ) -> tuple[float, float]:
        """Run statistical significance test.

        Returns
        -------
        tuple[float, float]
            (test_statistic, p_value)
        """
        if self.config.test == "t_test":
            return self._t_test(ar_results)
        return self._bmp_test(ar_results)

    def _t_test(
        self,
        ar_results: list[AbnormalReturnResult],
    ) -> tuple[float, float]:
        """Standard parametric t-test on CAAR.

        H0: CAAR = 0
        Test statistic: t = CAAR / SE(CAAR)
        """
        # Get CARs for all events
        cars = [r.car for r in ar_results]
        n = len(cars)

        if n < 2:
            return 0.0, 1.0

        mean_car = np.mean(cars)
        std_car = np.std(cars, ddof=1)
        se_car = std_car / np.sqrt(n)

        if se_car == 0:
            return 0.0, 1.0

        t_stat = mean_car / se_car
        p_value = 2 * stats.t.sf(abs(t_stat), df=n - 1)

        return float(t_stat), float(p_value)

    def _bmp_test(self, ar_results: list[AbnormalReturnResult]) -> tuple[float, float]:
        """Boehmer, Musumeci, Poulsen (1991) test.

        Robust to event-induced variance changes by standardizing
        ARs by their estimation period volatility.

        SAR_i = AR_i / σ_i
        Test statistic: Z = (1/N) * Σ SAR_i / SE(SAR)
        """
        # Compute standardized abnormal returns
        sars = []
        for r in ar_results:
            if r.estimation_residual_std and r.estimation_residual_std > 0:
                sar = r.car / r.estimation_residual_std
            else:
                sar = r.car  # Fallback to unstandardized
            sars.append(sar)

        n = len(sars)
        if n < 2:
            return 0.0, 1.0

        mean_sar = np.mean(sars)
        std_sar = np.std(sars, ddof=1)
        se_sar = std_sar / np.sqrt(n)

        if se_sar == 0:
            return 0.0, 1.0

        z_stat = mean_sar / se_sar
        p_value = 2 * stats.norm.sf(abs(z_stat))

        return float(z_stat), float(p_value)

    def run(self) -> EventStudyResult:
        """Run complete event study analysis.

        This is the main entry point that computes abnormal returns,
        aggregates results, and runs statistical tests.

        Returns
        -------
        EventStudyResult
            Complete event study results.

        Examples
        --------
        >>> analysis = EventStudyAnalysis(returns, events, benchmark)
        >>> result = analysis.run()
        >>> print(result.summary())
        >>> if result.is_significant:
        ...     print("Significant abnormal returns detected!")
        """
        return self.aggregate()

    def create_tear_sheet(self) -> EventStudyResult:
        """Alias for run() - creates complete event study results."""
        return self.run()

    @property
    def n_events(self) -> int:
        """Number of events in the study."""
        return len(self._events)

    @property
    def n_valid_events(self) -> int:
        """Number of events with sufficient data for analysis."""
        ar_results = self.compute_abnormal_returns()
        return len(ar_results)

    @property
    def assets(self) -> list[str]:
        """List of unique assets in the events."""
        return self._events["asset"].unique().sort().to_list()

    @property
    def date_range(self) -> tuple[Any, Any]:
        """Date range of the returns data."""
        return self._all_dates[0], self._all_dates[-1]
