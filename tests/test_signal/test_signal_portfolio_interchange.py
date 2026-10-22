"""Synthetic allocations expire on supplied keys, not an inferred broker clock."""

import json
from datetime import date
from pathlib import Path

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


def _daily_panel():
    dates = pd.DatetimeIndex(["2025-07-04T10:00Z", "2025-07-04T14:00Z", "2025-07-07T10:00Z"])
    factor = pd.DataFrame(
        {
            "date": dates.repeat(2),
            "asset": ["A", "B"] * 3,
            "factor": [1.0, 3.0] * 3,
        }
    )
    forward = factor[["date", "asset"]].assign(custom=[0.1, 0.1, 0.2, 0.2, 0.05, 0.05])
    panel = signal.prepare_signal_panel(factor, forward, return_columns=["custom"], quantiles=2)
    expiries = factor[["date", "asset"]].assign(
        requested_exit_time=factor["date"] + pd.Timedelta("2D")
    )
    return panel, expiries


@pytest.mark.parametrize("capital", [None, 1.0, 1000.0])
def test_daily_legacy_interchange_retains_event_sum_gaps_and_unknown_capital_marks(capital):
    panel, expiries = _daily_panel()
    result = panel.to_portfolio_inputs(
        return_column="custom",
        expiries=expiries,
        demeaned=False,
        capital=capital,
        first_return="zero",
        position_method="legacy_event_sum",
    )
    returns, positions, benchmark = result.to_pyfolio()
    assert returns.index.tolist() == list(pd.date_range("2025-07-04", periods=4, tz="UTC"))
    np.testing.assert_allclose(returns, [0.0, 0.0, 0.0, 0.05])
    assert benchmark is None
    np.testing.assert_allclose(result.returns["wealth"], [1.32, 1.32, 1.32, 1.386])
    nav = 1.0 if capital is None else capital * 1.32
    np.testing.assert_allclose(positions.iloc[0], [0.25 * nav, 0.75 * nav, 0.0])
    np.testing.assert_allclose(positions.iloc[1], [0.0, 0.0, nav])
    # Sunday contains a partial-day expiry event. Its event sum is invested,
    # although all allocations have expired by that day's final event.
    np.testing.assert_allclose(positions.iloc[2], [0.25 * nav, 0.75 * nav, 0.0])
    if capital is not None:
        assert positions.loc[pd.Timestamp("2025-07-08", tz="UTC")].isna().all()
        assert (
            result.positions.filter(pl.col("date") == pd.Timestamp("2025-07-08", tz="UTC"))[
                "value"
            ].null_count()
            == 2
        )
    else:
        np.testing.assert_allclose(positions.iloc[-1], [0.0, 0.0, 1.0])
    restored = signal.SyntheticPortfolioInputs.from_json(result.to_json())
    assert restored.positions.equals(result.positions)
    assert restored.valuations.equals(result.valuations)


def test_native_daily_defaults_keep_active_holdings_and_first_wealth_change():
    panel, expiries = _daily_panel()
    result = panel.to_portfolio_inputs(return_column="custom", expiries=expiries, demeaned=False)
    returns, holdings, _ = result.to_pyfolio()
    np.testing.assert_allclose(returns, [0.32, 0.0, 0.0, 0.05])
    np.testing.assert_allclose(holdings.iloc[1], [0.25, 0.75, 0.0])
    np.testing.assert_allclose(holdings.iloc[2], [0.0, 0.0, 1.0])
    # Annualization is selected by daily return cadence, not the two-day expiry.
    stats = result.analyze(periods_per_year=365).compute_summary_stats()
    expected = 1.386 ** (365 / 4) - 1
    assert stats.annual_return == pytest.approx(expected)
    assert result.metadata["execution_account"] is False
    assert result.metadata["transactions"] == "not_created"


@pytest.mark.parametrize("capital", [0, -1, float("inf"), float("nan"), True])
def test_invalid_capital_does_not_modify_owned_factor_or_expiry_inputs(capital):
    panel, expiries = _daily_panel()
    before = expiries.copy(deep=True)
    snapshot = panel.data
    with pytest.raises(ValueError, match="capital"):
        panel.to_portfolio_inputs(return_column="custom", expiries=expiries, capital=capital)
    pd.testing.assert_frame_equal(expiries, before)
    assert panel.data.equals(snapshot)


def test_daily_inputs_reach_existing_report_without_changing_the_annualization_scale(tmp_path):
    pytest.importorskip("plotly")
    panel, expiries = _daily_panel()
    result = panel.to_portfolio_inputs(return_column="custom", expiries=expiries, demeaned=False)
    report = result.analyze(periods_per_year=365).create_tear_sheet()
    assert report.metrics.annual_return == pytest.approx(1.386 ** (365 / 4) - 1)
    assert report.metrics.total_return == pytest.approx(0.386)
    destination = tmp_path / "synthetic-portfolio.html"
    report.save_html(destination, include_plotlyjs=False)
    assert "Portfolio Tear Sheet" in destination.read_text()
    np.testing.assert_allclose(
        report.figures["Cumulative Returns"].data[0].y, [0.32, 0.32, 0.32, 0.386]
    )


@pytest.mark.parametrize(
    "name", ["session-labels", "expected-bars", "overnight-bars", "early-close"]
)
def test_actual_producer_endpoints_feed_synthetic_holdings_without_reinterpreting_units(name):
    data = Path(__file__).parent / "data" / "outcomes"
    provenance = json.loads((data / "scheduled-provenance.json").read_text())
    factors = pl.read_ipc(data / f"{name}-factor.arrow")
    outcomes = pl.read_ipc(data / f"{name}-outcomes.arrow")
    panel = signal.prepare_signal_panel(
        factors,
        outcomes,
        horizon_col="horizon",
        feed_spec=provenance["cases"][name]["feed_spec"],
        quantiles=2,
        max_loss=0,
    )
    result = panel.positions(return_column=name)
    assert result.allocations["expiry"].dtype == outcomes["requested_exit_time"].dtype
    actual = result.allocations.select(
        pl.col("date").alias("timestamp"), "asset", pl.col("expiry").alias("requested_exit_time")
    ).sort("timestamp", "asset")
    assert actual.equals(
        outcomes.select("timestamp", "asset", "requested_exit_time").sort("timestamp", "asset")
    )
    inputs = panel.to_portfolio_inputs(return_column=name, capital=1000, benchmark_column=None)
    assert inputs.observations["n_obs"].to_list() == [4] * (factors.height // 4)
    assert inputs.positions["value"].drop_nulls().is_finite().all()


def test_finer_supplied_expiry_resolution_does_not_break_the_return_join():
    dates = pd.DatetimeIndex(["2025-07-07T10:00Z", "2025-07-07T11:00Z"])
    data = pl.DataFrame(
        {
            "date": dates.repeat(2),
            "asset": ["A", "B"] * 2,
            "factor": [1.0, 3.0] * 2,
            "custom": [0.1] * 4,
        }
    )
    expiries = data.select("date", "asset").with_columns(
        (pl.col("date").dt.cast_time_unit("ns") + pl.duration(nanoseconds=1)).alias("expiry")
    )
    positions = signal.compute_factor_positions(data, expiries=expiries, demeaned=False)
    inputs = signal.build_factor_portfolio_inputs(data, positions=positions, return_column="custom")
    assert inputs.observations["date"].dtype == pl.Datetime("ns", "UTC")
    assert inputs.returns["return"][0] == pytest.approx(0.21)
    with pytest.raises(ValueError, match="every selected allocation key"):
        signal.build_factor_portfolio_inputs(
            data.head(2), positions=positions, return_column="custom"
        )
