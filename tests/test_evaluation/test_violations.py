"""Realistic violations change emitted frames only the way the world or the pipeline does.

The producing code here is a small hourly-bar pipeline: trades are aggregated into bars, and
bars feed a trailing volatility. Each perturbation is applied to what that pipeline emitted.
"""

from __future__ import annotations

import importlib
import sys
from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl
import pytest

from ml4t.diagnostic.evaluation.violations import (
    ViolationNotRealizableError,
    drop_window,
    interior_times,
    mutated_source,
    reinsert_crossers,
    retire_and_reuse,
)

HOUR = timedelta(hours=1)
START = datetime(2023, 1, 2, tzinfo=UTC)


def emitted_bars(hours: int = 300, symbols: tuple[str, ...] = ("BTC", "ETH")) -> pl.DataFrame:
    """Hourly bars as an aggregation stage would emit them: one row per symbol and hour."""
    rng = np.random.default_rng(3)
    frames = []
    for symbol in symbols:
        close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, hours)))
        frames.append(
            pl.DataFrame(
                {
                    "timestamp": [START + i * HOUR for i in range(hours)],
                    "symbol": symbol,
                    "close": close,
                }
            )
        )
    return pl.concat(frames)


def vol_by_rows(bars: pl.DataFrame, window: int = 24) -> pl.DataFrame:
    """The defective feature: a window of rows read as a window of hours."""
    return bars.sort("symbol", "timestamp").with_columns(
        pl.col("close").log().diff().over("symbol").rolling_std(window).over("symbol").alias("vol")
    )


def vol_by_clock(bars: pl.DataFrame, window: int = 24) -> pl.DataFrame:
    """The correct feature: only returns between bars one hour apart, over the last window hours."""
    out = bars.sort("symbol", "timestamp").with_columns(
        pl.when(pl.col("timestamp").diff().over("symbol") == pl.duration(hours=1))
        .then(pl.col("close").log().diff().over("symbol"))
        .alias("ret")
    )
    return out.with_columns(
        pl.col("ret")
        .rolling_std_by("timestamp", window_size=f"{window}h")
        .over("symbol")
        .alias("vol")
    ).drop("ret")


def test_gap_from_emitted_bars_separates_row_and_clock_windows():
    # The violation the row-count feature exists to be caught on: a multi-hour outage.
    bars = emitted_bars()
    t = interior_times(bars, time_col="timestamp", entity_col="symbol", entity="BTC", margin=100)[0]
    gapped = drop_window(
        bars, time_col="timestamp", start=t, end=t + 49 * HOUR, entity_col="symbol", entity="BTC"
    )
    assert gapped.height == bars.height - 49
    # Before the gap exists, the two computations agree after warm-up; after it, they do not.
    clean = vol_by_rows(bars).join(vol_by_clock(bars), on=["symbol", "timestamp"])
    warm = clean.filter(pl.col("timestamp") > START + 30 * HOUR)
    assert np.allclose(warm["vol"], warm["vol_right"], rtol=1e-9, equal_nan=True)
    broken = vol_by_rows(gapped).join(vol_by_clock(gapped), on=["symbol", "timestamp"])
    after = broken.filter(
        (pl.col("symbol") == "BTC") & pl.col("timestamp").is_between(t + 49 * HOUR, t + 70 * HOUR)
    )
    assert not np.allclose(after["vol"], after["vol_right"], rtol=1e-6, equal_nan=True)


def test_drop_window_must_leave_records_on_both_sides():
    bars = emitted_bars(hours=50)
    with pytest.raises(ViolationNotRealizableError, match="not inside"):
        drop_window(bars, time_col="timestamp", start=START, end=START + 5 * HOUR)
    with pytest.raises(ViolationNotRealizableError, match="not inside"):
        drop_window(bars, time_col="timestamp", start=START + 45 * HOUR, end=START + 60 * HOUR)


def test_drop_window_that_removes_nothing_is_refused():
    # Prevents: a perturbation that silently did nothing letting a test pass.
    bars = emitted_bars(hours=50).filter(
        ~pl.col("timestamp").is_between(START + 10 * HOUR, START + 20 * HOUR, closed="left")
    )
    with pytest.raises(ViolationNotRealizableError, match="nothing to remove"):
        drop_window(bars, time_col="timestamp", start=START + 10 * HOUR, end=START + 20 * HOUR)


def test_missing_session_across_the_panel():
    bars = emitted_bars(hours=96)
    day = START + timedelta(days=2)
    out = drop_window(bars, time_col="timestamp", start=day, end=day + timedelta(days=1))
    assert out.height == bars.height - 2 * 24
    assert out.filter(pl.col("timestamp").dt.date() == day.date()).is_empty()


def test_reused_ticker_joins_two_entities_under_one_identifier():
    bars = emitted_bars(hours=200, symbols=("OLD", "NEW", "SPY"))
    at = START + 80 * HOUR
    out = retire_and_reuse(
        bars,
        entity_col="symbol",
        time_col="timestamp",
        retired="OLD",
        successor="NEW",
        at=at,
        gap=timedelta(days=3),
    )
    old = out.filter(pl.col("symbol") == "OLD").sort("timestamp")
    assert "NEW" not in out["symbol"].to_list()
    assert out.filter(pl.col("symbol") == "SPY").equals(bars.filter(pl.col("symbol") == "SPY"))
    jump = old["timestamp"].diff().max()
    assert jump == timedelta(days=3) + HOUR
    # A return computed by identifier now spans two firms.
    first_after = old.filter(pl.col("timestamp") >= at)["close"][0]
    new_first = bars.filter(
        (pl.col("symbol") == "NEW") & (pl.col("timestamp") >= at + timedelta(days=3))
    )["close"][0]
    assert first_after == new_first


def test_reuse_needs_records_on_both_sides():
    bars = emitted_bars(hours=50, symbols=("OLD", "NEW"))
    with pytest.raises(ViolationNotRealizableError, match="need records"):
        retire_and_reuse(
            bars,
            entity_col="symbol",
            time_col="timestamp",
            retired="OLD",
            successor="NEW",
            at=START + 40 * HOUR,
            gap=timedelta(days=30),
        )


def labels(boundary: datetime, horizon: timedelta) -> tuple[pl.DataFrame, pl.DataFrame]:
    """A label stage's output before and after its holdout filter on the exit time."""
    decisions = [START + i * HOUR for i in range(100)]
    unfiltered = pl.DataFrame(
        {"decision_time": decisions, "exit_time": [d + horizon for d in decisions], "y": 0.0}
    )
    return unfiltered.filter(pl.col("exit_time") < boundary), unfiltered


def test_labels_exiting_inside_the_holdout_are_reinserted():
    boundary = START + 60 * HOUR
    filtered, unfiltered = labels(boundary, horizon=8 * HOUR)
    out = reinsert_crossers(
        filtered, unfiltered, start_col="decision_time", end_col="exit_time", boundary=boundary
    )
    crossing = out.filter((pl.col("decision_time") < boundary) & (pl.col("exit_time") >= boundary))
    assert crossing.height == 8
    assert out.height == filtered.height + 8


def test_no_crossers_is_refused():
    boundary = START + 60 * HOUR
    filtered, unfiltered = labels(boundary, horizon=timedelta(0))
    with pytest.raises(ViolationNotRealizableError, match="no emitted row"):
        reinsert_crossers(
            filtered, unfiltered, start_col="decision_time", end_col="exit_time", boundary=boundary
        )


@pytest.fixture
def producing_module(tmp_path, monkeypatch):
    module = tmp_path / "stage_features.py"
    module.write_text("WINDOW = 168\n\ndef window():\n    return WINDOW\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    yield module
    sys.modules.pop("stage_features", None)


def test_mutated_source_is_seen_inside_and_restored_after(producing_module):
    import stage_features

    assert stage_features.window() == 168
    with mutated_source(producing_module, "WINDOW = 168", "WINDOW = 167"):
        assert importlib.import_module("stage_features").window() == 167
    assert importlib.import_module("stage_features").window() == 168
    assert producing_module.read_text().startswith("WINDOW = 168")


def test_mutated_source_restores_when_the_block_raises(producing_module):
    with pytest.raises(RuntimeError), mutated_source(producing_module, "168", "0"):
        raise RuntimeError("test body failed")
    assert "WINDOW = 168" in producing_module.read_text()


def test_mutation_text_must_occur_exactly_once(producing_module):
    # Prevents: a mutation that edits nothing, or edits a different occurrence than intended.
    with pytest.raises(ViolationNotRealizableError, match="0 times"):
        with mutated_source(producing_module, "WINDOW = 24", "WINDOW = 1"):
            pass
    producing_module.write_text("A = 1\nB = 1\n")
    with pytest.raises(ViolationNotRealizableError, match="2 times"):
        with mutated_source(producing_module, "= 1", "= 2"):
            pass
