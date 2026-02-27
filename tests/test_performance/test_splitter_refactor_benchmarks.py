"""Lightweight benchmarks for splitter refactor value checks.

These tests compare optimized splitter paths against local legacy-style baselines.
Run with:
    pytest tests/test_performance/test_splitter_refactor_benchmarks.py -q
Optional JSON output:
    pytest tests/test_performance/test_splitter_refactor_benchmarks.py -q --benchmark-json artifacts/benchmarks/splitter-refactor.json
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from ml4t.diagnostic.splitters.calendar import TradingCalendar
from ml4t.diagnostic.splitters.config import WalkForwardConfig
from ml4t.diagnostic.splitters.walk_forward import WalkForwardCV

# Skip benchmark tests when running under xdist (timings become noisy/disabled)
if os.environ.get("PYTEST_XDIST_WORKER"):
    pytest.skip("Benchmarks not supported under xdist", allow_module_level=True)


def _median_runtime(fn, rounds: int = 5) -> float:
    """Return median wall-clock runtime over repeated runs."""
    timings = []
    for _ in range(rounds):
        start = time.perf_counter()
        fn()
        timings.append(time.perf_counter() - start)
    return float(np.median(timings))


def _legacy_get_sessions_merge_asof(
    calendar: TradingCalendar, timestamps: pd.DatetimeIndex
) -> pd.Series:
    """Legacy merge_asof-based session assignment for benchmark comparison."""
    timestamps_tz = calendar._ensure_timezone_aware(timestamps)
    start_date = timestamps_tz[0].normalize() - pd.Timedelta(days=7)
    end_date = timestamps_tz[-1].normalize() + pd.Timedelta(days=7)

    schedule = calendar.calendar.schedule(start_date=start_date, end_date=end_date)
    if schedule["market_open"].dt.tz is None:
        schedule["market_open"] = schedule["market_open"].dt.tz_localize(calendar.tz)
        schedule["market_close"] = schedule["market_close"].dt.tz_localize(calendar.tz)
    else:
        schedule["market_open"] = schedule["market_open"].dt.tz_convert(calendar.tz)
        schedule["market_close"] = schedule["market_close"].dt.tz_convert(calendar.tz)

    df_ts = pd.DataFrame({"timestamp": timestamps_tz, "original_idx": range(len(timestamps_tz))})
    df_sessions = pd.DataFrame(
        {
            "session_date": schedule.index,
            "market_open": schedule["market_open"],
            "market_close": schedule["market_close"],
        }
    ).reset_index(drop=True)

    df_ts["timestamp"] = df_ts["timestamp"].dt.as_unit("ns")
    for col in ("market_open", "market_close"):
        df_sessions[col] = df_sessions[col].dt.as_unit("ns")

    df_ts_sorted = df_ts.sort_values("timestamp")
    df_sessions_sorted = df_sessions.sort_values("market_open")
    df_merged = pd.merge_asof(
        df_ts_sorted,
        df_sessions_sorted,
        left_on="timestamp",
        right_on="market_open",
        direction="backward",
    )

    within_session = df_merged["timestamp"] < df_merged["market_close"]
    if not within_session.all():
        df_outside = df_merged[~within_session][["timestamp", "original_idx"]]
        if len(df_outside) > 0:
            df_outside_merged = pd.merge_asof(
                df_outside,
                df_sessions_sorted,
                left_on="timestamp",
                right_on="market_open",
                direction="forward",
            )
            df_merged.loc[~within_session, "session_date"] = df_outside_merged[
                "session_date"
            ].values

    return df_merged.sort_values("original_idx").set_index(timestamps)["session_date"]


def _legacy_split_by_sessions_indices(
    df: pd.DataFrame,
    *,
    session_col: str,
    n_splits: int,
    test_size_sessions: int,
    train_size_sessions: int,
    expanding: bool = True,
    gap: int = 0,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Legacy mask-based per-fold row index generation."""
    unique_sessions = df[session_col].drop_duplicates().reset_index(drop=True)
    n_sessions = len(unique_sessions)
    available_for_splits = n_sessions - test_size_sessions
    step_size = available_for_splits // n_splits
    first_test_start = test_size_sessions

    results = []
    for i in range(n_splits):
        test_start_session = first_test_start + i * step_size
        test_end_session = min(test_start_session + test_size_sessions, n_sessions)
        if expanding:
            train_start_session = 0
        else:
            train_start_session = max(0, test_start_session - gap - train_size_sessions)
        train_end_session = test_start_session - gap

        train_sessions = unique_sessions.iloc[train_start_session:train_end_session].tolist()
        test_sessions = unique_sessions.iloc[test_start_session:test_end_session].tolist()
        train_indices = np.where(df[session_col].isin(train_sessions).to_numpy())[0].astype(np.intp)
        test_indices = np.where(df[session_col].isin(test_sessions).to_numpy())[0].astype(np.intp)
        results.append((train_indices, test_indices))
    return results


@pytest.fixture
def benchmark_calendar() -> TradingCalendar:
    return TradingCalendar("NYSE")


@pytest.fixture
def benchmark_timestamps() -> pd.DatetimeIndex:
    return pd.date_range("2024-01-01", periods=120_000, freq="5min", tz="America/New_York")


@pytest.fixture
def session_panel_df() -> pd.DataFrame:
    n_sessions = 320
    n_assets = 40
    sessions = pd.date_range("2024-01-02", periods=n_sessions, freq="B", tz="UTC")
    assets = [f"A{i:03d}" for i in range(n_assets)]
    rows = []
    for sess in sessions:
        for asset in assets:
            rows.append((sess, asset, np.random.randn()))
    return pd.DataFrame(rows, columns=["session_date", "asset", "feature"])


@pytest.fixture
def large_session_panel_df() -> pd.DataFrame:
    """Larger panel shape to catch scaling regressions."""
    n_sessions = 520
    n_assets = 120
    sessions = pd.date_range("2022-01-03", periods=n_sessions, freq="B", tz="UTC")
    assets = [f"A{i:03d}" for i in range(n_assets)]
    rows = []
    for sess in sessions:
        for asset in assets:
            rows.append((sess, asset, np.random.randn()))
    return pd.DataFrame(rows, columns=["session_date", "asset", "feature"])


class TestCalendarSessionBenchmark:
    @pytest.mark.benchmark
    def test_get_sessions_correctness_and_speed(
        self, benchmark_calendar: TradingCalendar, benchmark_timestamps: pd.DatetimeIndex
    ) -> None:
        optimized = benchmark_calendar.get_sessions(benchmark_timestamps)
        legacy = _legacy_get_sessions_merge_asof(benchmark_calendar, benchmark_timestamps)
        pd.testing.assert_series_equal(optimized, legacy, check_names=False)

        optimized_time = _median_runtime(
            lambda: benchmark_calendar.get_sessions(benchmark_timestamps)
        )
        legacy_time = _median_runtime(
            lambda: _legacy_get_sessions_merge_asof(benchmark_calendar, benchmark_timestamps)
        )
        # Guard against regressions while allowing CI jitter.
        assert optimized_time <= legacy_time * 1.25


class TestWalkForwardSessionBenchmark:
    @pytest.mark.benchmark
    def test_session_split_mapping_speed(self, session_panel_df: pd.DataFrame) -> None:
        config = WalkForwardConfig(
            n_splits=6,
            test_size=20,
            train_size=120,
            align_to_sessions=True,
            session_col="session_date",
            calendar_id=None,
            filter_non_trading=False,
        )
        cv = WalkForwardCV(config=config)

        optimized_splits = list(cv.split(session_panel_df))
        legacy_splits = _legacy_split_by_sessions_indices(
            session_panel_df,
            session_col="session_date",
            n_splits=6,
            test_size_sessions=20,
            train_size_sessions=120,
            expanding=True,
            gap=0,
        )
        assert len(optimized_splits) == len(legacy_splits)
        for (opt_train, opt_test), (legacy_train, legacy_test) in zip(
            optimized_splits, legacy_splits, strict=False
        ):
            np.testing.assert_array_equal(opt_train, legacy_train)
            np.testing.assert_array_equal(opt_test, legacy_test)

        optimized_time = _median_runtime(lambda: list(cv.split(session_panel_df)))
        legacy_time = _median_runtime(
            lambda: _legacy_split_by_sessions_indices(
                session_panel_df,
                session_col="session_date",
                n_splits=6,
                test_size_sessions=20,
                train_size_sessions=120,
                expanding=True,
                gap=0,
            )
        )
        assert optimized_time <= legacy_time * 1.25

    @pytest.mark.benchmark
    def test_session_split_mapping_speed_large_panel(
        self, large_session_panel_df: pd.DataFrame
    ) -> None:
        """Validate correctness and throughput on a larger panel shape."""
        config = WalkForwardConfig(
            n_splits=8,
            test_size=25,
            train_size=220,
            align_to_sessions=True,
            session_col="session_date",
            calendar_id=None,
            filter_non_trading=False,
        )
        cv = WalkForwardCV(config=config)

        optimized_splits = list(cv.split(large_session_panel_df))
        legacy_splits = _legacy_split_by_sessions_indices(
            large_session_panel_df,
            session_col="session_date",
            n_splits=8,
            test_size_sessions=25,
            train_size_sessions=220,
            expanding=True,
            gap=0,
        )
        assert len(optimized_splits) == len(legacy_splits)
        for (opt_train, opt_test), (legacy_train, legacy_test) in zip(
            optimized_splits, legacy_splits, strict=False
        ):
            np.testing.assert_array_equal(opt_train, legacy_train)
            np.testing.assert_array_equal(opt_test, legacy_test)

        optimized_time = _median_runtime(lambda: list(cv.split(large_session_panel_df)))
        legacy_time = _median_runtime(
            lambda: _legacy_split_by_sessions_indices(
                large_session_panel_df,
                session_col="session_date",
                n_splits=8,
                test_size_sessions=25,
                train_size_sessions=220,
                expanding=True,
                gap=0,
            )
        )
        assert optimized_time <= legacy_time * 1.25
