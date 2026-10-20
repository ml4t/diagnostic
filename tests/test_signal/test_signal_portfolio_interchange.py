"""Synthetic allocations expire on supplied keys, not an inferred broker clock."""

from datetime import date, datetime, timezone

import numpy as np
import pandas as pd
import polars as pl
import pytest

from ml4t.diagnostic import signal


def test_all_due_allocations_expire_and_coincident_entries_are_not_discarded():
    # The real factor pipeline can issue Saturday and Sunday allocations with
    # the same Monday business-day expiry, followed by a new Monday allocation.
    dates = [date(2025, 7, 5), date(2025, 7, 6), date(2025, 7, 7)]
    factors = pl.DataFrame(
        {
            "date": [day for day in dates for _ in range(2)],
            "asset": ["A", "B"] * 3,
            "factor": [1.0, 3.0, 3.0, 1.0, 0.0, 1.0],
            "expiry": [date(2025, 7, 7)] * 4 + [date(2025, 7, 8)] * 2,
        }
    )
    result = signal.compute_factor_positions(factors, demeaned=False, calendar=None)
    wide = result.positions.pivot(on="asset", index="date", values="weight").sort("date")
    assert wide["date"].to_list() == dates + [date(2025, 7, 8)]
    np.testing.assert_allclose(
        wide.select("A", "B").to_numpy(),
        [
            [0.25, 0.75],
            [0.5, 0.5],
            [0.0, 1.0],
            [0.0, 0.0],
        ],
    )
    assert result.timeline["active_allocations"].to_list() == [2, 4, 2, 0]
    assert result.timeline["status"].to_list() == ["invested"] * 3 + ["no_active_allocations"]
    assert result.metadata["input_contract"]["calendar"] is None
    assert result.allocations["allocation_id"].n_unique() == 6


def test_offsetting_allocations_and_zero_signal_have_distinct_owned_states():
    frame = pl.DataFrame(
        {
            "date": [1, 1, 2, 2, 4, 4],
            "asset": ["A", "B"] * 3,
            "factor": [-1.0, 1.0, 1.0, -1.0, 7.0, 7.0],
            "expiry": [4, 4, 4, 4, 5, 5],
        }
    ).with_columns(pl.col("date", "expiry").cast(pl.Date))
    result = signal.compute_factor_positions(frame)
    assert result.timeline["status"].to_list() == [
        "invested",
        "offsetting",
        "zero_signal",
        "no_active_allocations",
    ]
    assert result.positions.filter(pl.col("date") == pl.lit(2).cast(pl.Date))[
        "weight"
    ].to_list() == [0.0, 0.0]
    result.positions.replace_column(2, pl.Series("weight", [99.0] * 8))
    result.metadata["input_contract"]["calendar"] = "changed"
    restored = signal.SyntheticPositionsResult.from_json(result.to_json())
    assert restored.positions.equals(result.positions)
    assert restored.allocations.equals(result.allocations)
    assert restored.metadata["input_contract"]["calendar"] == "NYSE"


def test_nanosecond_expiry_and_stable_numeric_ids_survive_the_public_pipeline():
    start = pd.Timestamp("2025-07-07T12:00:00Z").value
    frame = pl.DataFrame(
        {
            "when": pl.Series([start, start, start + 1, start + 1], dtype=pl.Datetime("ns", "UTC")),
            "sid": [2**63 - 2, 2**63 - 1] * 2,
            "factor": [1.0, 3.0, 3.0, 1.0],
            "expiry": pl.Series([start + 1] * 2 + [start + 2] * 2, dtype=pl.Datetime("ns", "UTC")),
        }
    )
    result = signal.compute_factor_positions(
        frame,
        demeaned=False,
        feed_spec={
            "time_contract_version": 2,
            "timestamp_col": "when",
            "entity_col": "sid",
            "calendar": None,
        },
    )
    assert result.timeline["date"].cast(pl.Int64).to_list() == [start, start + 1, start + 2]
    assert result.positions["asset"].unique().sort().to_list() == [2**63 - 2, 2**63 - 1]
    np.testing.assert_allclose(result.positions["weight"], [0.25, 0.75, 0.75, 0.25, 0.0, 0.0])
    assert result.metadata["input_contract"]["calendar_source"] == "feed_spec"


@pytest.mark.parametrize("bad_expiry", [None, date(2025, 7, 7), date(2025, 7, 6)])
def test_unknown_or_nonfuture_expiry_rejects_without_mutating_input(bad_expiry):
    data = pl.DataFrame(
        {"date": [date(2025, 7, 7)], "asset": ["A"], "factor": [1.0], "expiry": [bad_expiry]},
        schema_overrides={"expiry": pl.Date},
    )
    original = data.clone()
    with pytest.raises(ValueError, match="expiry"):
        signal.compute_factor_positions(data)
    assert data.equals(original)


def test_prepared_panel_uses_actual_selected_outcome_expiry_and_filters_before_sizing():
    dates = pd.date_range("2025-07-07", periods=3, tz="UTC")
    factor = pl.DataFrame(
        {
            "date": np.repeat(dates, 4),
            "asset": ["A", "B", "C", "D"] * 3,
            "factor": [1.0, 2.0, 3.0, 4.0] * 3,
        }
    )
    outcomes = factor.select("date", "asset").with_columns(
        pl.lit("custom").alias("horizon"),
        pl.lit(0.01).alias("forward_return"),
        (pl.col("date") + pl.duration(days=2)).alias("requested_exit_time"),
    )
    panel = signal.prepare_signal_panel(factor, outcomes, horizon_col="horizon", quantiles=2)
    result = panel.positions(return_column="custom", quantiles=[2], demeaned=True)
    assert result.allocations["asset"].unique().sort().to_list() == ["C", "D"]
    np.testing.assert_allclose(result.allocations["weight"], [-0.5, 0.5] * 3)
    assert result.metadata["expiry_source"] == "requested_exit_time"
    with pytest.raises(ValueError, match="expiry|expiries"):
        signal.prepare_signal_panel(
            factor,
            factor.select("date", "asset").with_columns(pl.lit(0.01).alias("custom")),
            return_columns=["custom"],
            quantiles=2,
        ).positions(return_column="custom")
