"""Consecutive VWAP is a named analytical convention with actual source provenance."""

import pandas as pd
import polars as pl
import pytest

from ml4t.diagnostic.evaluation import PortfolioAnalysis
from ml4t.diagnostic.evaluation.portfolio_analysis.fifo import reconstruct_round_trips
from ml4t.diagnostic.evaluation.portfolio_analysis.frame_inputs import normalize_executions


def _account(
    quantities=(10.0, 5.0, -12.0), prices=(100.0, 110.0, 120.0), fees=(10.0, 10.0, 12.0), dates=None
):
    if dates is None:
        dates = pd.to_datetime(["2025-07-07T12:00Z", "2025-07-07T13:00Z", "2025-07-08T12:00Z"])
    return PortfolioAnalysis.from_frames(
        pd.Series([0.01] * len(dates), index=dates),
        calendar=None,
        transactions=pd.DataFrame(
            {
                "timestamp": dates,
                "asset": [7] * len(dates),
                "quantity": quantities,
                "price": prices,
                "commission": fees,
                "record_id": [f"fill{index}" for index in range(len(dates))],
            }
        ),
    )


def test_vwap_partial_close_is_explicitly200_not_actual_fifo220_and_original_fills_are_retained():
    analysis = _account()
    before = analysis.input_data.transactions
    actual = analysis.compute_round_trips(initial_state="flat")
    grouped = analysis.compute_round_trips(initial_state="flat", matching="consecutive_vwap")
    assert actual.metrics["realized_gross_pnl"] == 220.0
    assert grouped.metrics["realized_gross_pnl"] == 200.0
    assert grouped.metrics["realized_net_pnl"] == 172.0
    assert grouped.open_lots["quantity"].to_list() == [3.0]
    assert grouped.open_lots["remaining_commission"].to_list() == [4.0]
    assert grouped.executions.equals(before)
    assert grouped.matching_groups["member_record_ids"].to_list() == [["fill0", "fill1"], ["fill2"]]
    assert grouped.segments["entry_record_id"].to_list() == [None]
    assert grouped.segments["exit_record_id"].to_list() == ["fill2"]
    assert grouped.completed_trades().height == 1
    assert grouped.to_trade_analysis().compute_statistics().total_pnl == 172.0
    assert grouped.metadata["processed_executions"] == 3


def test_group_price_availability_is_last_member_not_its_earlier_representative_timestamp():
    result = _account().compute_round_trips(initial_state="flat", matching="consecutive_vwap")
    assert (
        result.matching_groups["timestamp"].cast(pl.Int64).to_list()[0]
        == pd.Timestamp("2025-07-07T12:00Z").value
    )
    assert (
        result.matching_groups["available_at"].cast(pl.Int64).to_list()[0]
        == pd.Timestamp("2025-07-07T13:00Z").value
    )
    assert result.completed_trades()["entry_available_at"].cast(pl.Int64).to_list() == [
        pd.Timestamp("2025-07-07T13:00Z").value
    ]
    assert "last member" in result.to_trade_analysis().source_metadata["timestamp_convention"]


def test_as_of_cuts_raw_history_before_grouping_so_future_member_price_cannot_leak():
    result = _account().compute_round_trips(
        initial_state="flat", matching="consecutive_vwap", as_of="2025-07-07T12:30Z"
    )
    assert result.matching_groups.height == 1
    assert result.open_lots["quantity"].to_list() == [10.0]
    assert result.open_lots["entry_price"].to_list() == [100.0]
    assert result.metadata["excluded_future_executions"] == 2
    assert result.executions.height == 3


def test_eight_hour_gap_is_inclusive_between_adjacent_fills_not_a_total_group_age():
    dates = pd.to_datetime(
        ["2025-07-07T00:00Z", "2025-07-07T08:00Z", "2025-07-07T16:00Z", "2025-07-08T12:00Z"]
    )
    result = _account(
        (1.0, 1.0, 1.0, -3.0), (100.0, 110.0, 120.0, 130.0), (0.0, 0.0, 0.0, 0.0), dates
    ).compute_round_trips(initial_state="flat", matching="consecutive_vwap")
    assert result.matching_groups["n_source_executions"].to_list() == [3, 1]
    assert result.metrics["realized_gross_pnl"] == 60.0


def test_one_nanosecond_beyond_gap_splits_groups_without_losing_original_instants():
    dates = pd.to_datetime(
        [
            "2025-07-07T00:00:00.000000001Z",
            "2025-07-07T08:00:00.000000002Z",
            "2025-07-08T12:00:00.000000003Z",
        ]
    )
    result = _account(dates=dates).compute_round_trips(
        initial_state="flat", matching="consecutive_vwap"
    )
    assert result.matching_groups.height == 3
    assert (
        result.matching_groups["timestamp"].cast(pl.Int64).to_list()
        == dates.as_unit("ns").asi8.tolist()
    )
    assert result.metrics["realized_gross_pnl"] == 220.0


def test_distinct_multi_fill_closing_groups_do_not_collapse_because_actual_exit_id_is_null():
    dates = pd.to_datetime(
        [
            "2025-07-07T00:00Z",
            "2025-07-08T00:00Z",
            "2025-07-08T01:00Z",
            "2025-07-09T00:00Z",
            "2025-07-09T01:00Z",
        ]
    )
    result = _account(
        (4.0, -1.0, -1.0, -1.0, -1.0), (100.0, 110.0, 110.0, 120.0, 120.0), (0.0,) * 5, dates
    ).compute_round_trips(initial_state="flat", matching="consecutive_vwap")
    assert result.completed_trades()["exit_record_id"].to_list() == [None, None]
    assert result.completed_trades()["gross_pnl"].to_list() == [20.0, 40.0]
    assert result.to_trade_analysis().compute_statistics().n_trades == 2


def test_fractional_units_and_large_integer_price_vwap_use_exact_components_not_float_projection():
    result = _account((0.1, 0.2, -0.3), (100.0, 110.0, 120.0), (0.0, 0.0, 0.0)).compute_round_trips(
        initial_state="flat", matching="consecutive_vwap"
    )
    assert result.metrics["realized_gross_pnl"] == 4.0
    assert result.open_lots.is_empty()
    price = 2**53
    large = _account(
        (1, 2, -3), (price + 1, price + 2, price + 3), (0.0, 0.0, 0.0)
    ).compute_round_trips(initial_state="flat", matching="consecutive_vwap")
    assert large.metrics["realized_gross_pnl"] == 4.0
    assert large.completed_trades()["gross_pnl"].to_list() == [4.0]
    assert large.executions["price"].to_list() == [price + 1, price + 2, price + 3]


@pytest.mark.parametrize("max_gap", [1, "1", "-1h", pd.NaT])
def test_ambiguous_or_negative_group_gap_is_rejected_instead_of_treated_as_row_units(max_gap):
    with pytest.raises((ValueError, TypeError), match="max_gap"):
        _account().compute_round_trips(
            initial_state="flat", matching="consecutive_vwap", max_gap=max_gap
        )


def test_custom_gap_cannot_be_silently_ignored_by_default_actual_fifo():
    with pytest.raises(ValueError, match="max_gap"):
        _account().compute_round_trips(initial_state="flat", max_gap="1h")


def test_grouped_synthetic_close_keeps_group_sources_available_and_actual_totals_separate():
    result = _account().compute_round_trips(
        initial_state="flat",
        matching="consecutive_vwap",
        marks=pd.DataFrame(
            {"timestamp": pd.to_datetime(["2025-07-08T13:00Z"]), "asset": [7], "price": [130.0]}
        ),
        as_of="2025-07-08T14:00Z",
        synthetic_close=True,
    )
    assert result.metrics["realized_net_pnl"] == 172.0
    assert result.metrics["marked_net_pnl"] == 248.0
    assert result.completed_trades(include_synthetic=True)["net_pnl"].to_list() == [172.0, 76.0]
    assert result.synthetic_segments["exit_record_id"].to_list() == [None]
    assert result.synthetic_segments["exit_available_at"].cast(pl.Int64).to_list() == [
        pd.Timestamp("2025-07-08T14:00Z").value
    ]


def test_raw_execution_normalizer_reuses_account_contract_without_a_dummy_return_series():
    analysis = _account()
    raw = analysis.input_data.transactions.rename({"timestamp": "event", "asset": "sid"})
    normalized = normalize_executions(raw, timestamp_col="event", entity_col="sid")
    result = reconstruct_round_trips(normalized, initial_state="flat", matching="consecutive_vwap")
    assert result.metrics["realized_gross_pnl"] == 200.0
    assert normalized.equals(analysis.input_data.transactions)
    assert raw.columns[0] == "event"


def test_empty_typed_execution_ledger_keeps_empty_matching_groups_and_raw_identity():
    dates = pd.date_range("2025-07-07", periods=1, tz="UTC")
    fills = pl.DataFrame(
        schema={
            "timestamp": pl.Datetime("ns", "UTC"),
            "asset": pl.Int64,
            "quantity": pl.Float64,
            "price": pl.Float64,
            "record_id": pl.String,
        }
    )
    analysis = PortfolioAnalysis.from_frames(
        pd.Series([0.01], index=dates), transactions=fills, calendar=None
    )
    result = analysis.compute_round_trips(initial_state="flat", matching="consecutive_vwap")
    assert result.matching_groups.is_empty()
    assert result.executions.is_empty()
    assert result.completed_trades().is_empty()
    assert result.win_probability()["sample_state"] == "prior_only"


def test_published_pyfolio_vwap_group_uses_same_membership_cash_and_closing_sample():
    reference = pytest.importorskip("pyfolio.round_trips")
    analysis = _account()
    fills = (
        analysis.input_data.transactions.to_pandas()
        .set_index("timestamp")
        .rename(columns={"asset": "symbol", "quantity": "amount"})
    )
    fills["symbol"] = fills["symbol"].map({7: "A"})
    legacy_groups = reference._groupby_consecutive(fills)
    legacy = reference.extract_round_trips(fills)
    native = analysis.compute_round_trips(initial_state="flat", matching="consecutive_vwap")
    assert (
        native.matching_groups["quantity"].to_list()
        == legacy_groups["amount"].to_list()
        == [15.0, -12.0]
    )
    assert native.matching_groups["price"].to_list() == legacy_groups["price"].to_list()
    assert native.completed_trades().height == len(legacy) == 1
    assert native.completed_trades()["gross_pnl"].to_list() == pytest.approx(
        legacy["pnl"].to_list()
    )
    assert native.completed_trades()["gross_lot_return"].to_list() == pytest.approx(
        legacy["rt_returns"].to_list()
    )


@pytest.mark.parametrize("quantity", [1, 2**63 + 1])
def test_unsigned_external_absolute_quantities_are_signed_losslessly_before_side_reconciliation(
    quantity,
):
    dates = pd.date_range("2025-07-07", periods=2, tz="UTC")
    fills = pl.DataFrame(
        {
            "timestamp": dates,
            "asset": [7, 7],
            "quantity": pl.Series([quantity, quantity], dtype=pl.UInt64),
            "price": [100, 120],
            "side": ["buy", "sell"],
            "commission": [0.0, 0.0],
        }
    )
    before = fills.clone()
    result = PortfolioAnalysis.from_frames(
        pd.Series([0.01, 0.02], index=dates), transactions=fills, calendar=None
    ).compute_round_trips(initial_state="flat")
    assert result.metrics["realized_gross_pnl"] == float(20 * quantity)
    assert result.executions["quantity"].to_list() == [quantity, -quantity]
    assert result.segments["quantity_exact"].to_list() == [str(quantity)]
    assert fills.equals(before)
