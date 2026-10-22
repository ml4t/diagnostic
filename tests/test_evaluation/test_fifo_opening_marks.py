"""Supplied inventory and dated marks must not fabricate basis, fills or realized PnL."""

from copy import deepcopy

import pandas as pd
import polars as pl
import pytest

from ml4t.diagnostic.evaluation import PortfolioAnalysis


def _account(quantities=(10.0, 5.0, -12.0), prices=(100.0, 110.0, 120.0), fees=(10.0, 10.0, 12.0)):
    dates = pd.date_range("2025-07-07", periods=len(quantities), tz="UTC")
    return PortfolioAnalysis.from_frames(
        pd.Series([0.01] * len(dates), index=dates),
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
        calendar=None,
        transaction_timestamp_col="timestamp",
    )


def _opening(*, unknown_basis=False):
    return pd.DataFrame(
        {
            "asset": [7],
            "lot_id": ["carried0"],
            "quantity": [5.0],
            "entry_timestamp": pd.Series(
                [pd.NaT if unknown_basis else pd.Timestamp("2025-07-06T00:00Z")],
                dtype="datetime64[ns, UTC]",
            ),
            "entry_price": [None if unknown_basis else 90.0],
            "multiplier": [1.0],
            "remaining_commission": [None if unknown_basis else 5.0],
        }
    )


def _marks():
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2025-07-09T21:00Z", "2025-07-09T23:00Z"]),
            "asset": [7, 7],
            "price": [130.0, 999.0],
        }
    )


def test_carried_lots_match_before_new_fills_and_remaining_entry_fees_are_not_charged_twice():
    supplied = _opening()
    before = deepcopy(supplied)
    result = _account((5.0, -8.0), (100.0, 110.0), (5.0, 8.0)).compute_round_trips(
        initial_state="supplied", opening_lots=supplied, opening_as_of="2025-07-07T00:00Z"
    )
    assert result.segments["quantity"].to_list() == [5.0, 3.0]
    assert result.segments["gross_pnl"].to_list() == [100.0, 30.0]
    assert result.metrics["realized_gross_pnl"] == 130.0
    assert result.metrics["realized_net_pnl"] == 114.0
    assert result.segments["opening_lot_id"].to_list() == ["carried0", None]
    assert result.segments["entry_record_id"].to_list() == [None, "fill0"]
    assert result.open_lots["quantity"].to_list() == [2.0]
    assert result.open_lots["remaining_commission"].to_list() == [2.0]
    pd.testing.assert_frame_equal(supplied, before)


def test_unknown_opening_basis_and_time_do_not_become_zero_basis_or_fabricated_duration():
    result = _account((-3.0,), (110.0,), (3.0,)).compute_round_trips(
        initial_state="supplied",
        opening_lots=_opening(unknown_basis=True),
        opening_as_of="2025-07-07T00:00Z",
    )
    assert result.segments["quantity"].to_list() == [3.0]
    assert result.segments["entry_price"].to_list() == [None]
    assert result.segments["entry_timestamp"].null_count() == 1
    assert result.metrics["realized_gross_pnl"] is None
    assert result.metrics["realized_net_pnl"] is None
    assert result.open_lots["quantity"].to_list() == [2.0]


def test_known_at_mark_excludes_future_quotes_and_never_counts_open_profit_as_realized():
    prices = _marks()
    before = prices.copy(deep=True)
    result = _account().compute_round_trips(
        initial_state="flat", marks=prices, as_of="2025-07-09T22:00Z"
    )
    assert result.open_lots["mark"].to_list() == [130.0]
    assert result.open_lots["mark_age_seconds"].to_list() == [3600.0]
    assert result.open_lots["mark_source_row"].to_list() == [0]
    assert result.metrics["realized_gross_pnl"] == 220.0
    assert result.metrics["unrealized_gross_pnl"] == 60.0
    assert result.metrics["unrealized_net_pnl"] == 54.0
    assert result.synthetic_segments is None
    pd.testing.assert_frame_equal(prices, before)


def test_as_of_excludes_future_executions_but_preserves_original_full_ledger():
    analysis = _account()
    before = analysis.input_data.transactions
    prices = _marks().iloc[:1].copy()
    prices["timestamp"] = pd.to_datetime(["2025-07-08T00:00Z"])
    prices["price"] = 120.0
    result = analysis.compute_round_trips(
        initial_state="flat", marks=prices, as_of="2025-07-08T00:00Z"
    )
    assert result.segments.is_empty()
    assert result.open_lots["quantity"].to_list() == [10.0, 5.0]
    assert result.metrics["realized_gross_pnl"] == 0.0
    assert result.metrics["unrealized_gross_pnl"] == 250.0
    assert result.metrics["unrealized_net_pnl"] == 230.0
    assert result.metadata["excluded_future_executions"] == 1
    assert result.executions.equals(before)


def test_synthetic_close_is_separate_from_actual_realization_and_actual_open_inventory():
    result = _account().compute_round_trips(
        initial_state="flat", marks=_marks(), as_of="2025-07-09T22:00Z", synthetic_close=True
    )
    assert result.metrics["realized_gross_pnl"] == 220.0
    assert result.metrics["realized_net_pnl"] == 194.0
    assert result.open_lots["quantity"].to_list() == [3.0]
    assert result.synthetic_segments["gross_pnl"].to_list() == [60.0]
    assert result.synthetic_segments["net_pnl"].to_list() == [54.0]
    assert result.synthetic_segments["exit_record_id"].null_count() == 1
    assert result.synthetic_segments["is_synthetic"].to_list() == [True]
    assert result.metrics["marked_gross_pnl"] == 280.0
    assert result.metrics["marked_net_pnl"] == 248.0
    assert result.executions.height == 3


def test_marks_require_explicit_as_of_and_synthetic_close_cannot_guess_a_required_mark():
    with pytest.raises(ValueError, match="as_of"):
        _account().compute_round_trips(initial_state="flat", marks=_marks())
    with pytest.raises(ValueError, match="mark"):
        _account().compute_round_trips(
            initial_state="flat", synthetic_close=True, as_of="2025-07-09T22:00Z"
        )


def test_opening_inventory_requires_complete_state_declaration_and_cannot_enter_in_the_future():
    with pytest.raises(ValueError, match="initial_state|complete"):
        _account().compute_round_trips(opening_lots=_opening(), opening_as_of="2025-07-07T00:00Z")
    future = _opening()
    future["entry_timestamp"] = pd.to_datetime(["2025-07-08T00:00Z"])
    with pytest.raises(ValueError, match="opening|future"):
        _account().compute_round_trips(
            initial_state="supplied", opening_lots=future, opening_as_of="2025-07-07T00:00Z"
        )


def test_execution_history_in_opening_inventory_is_not_matched_again():
    supplied = _opening()
    supplied["entry_timestamp"] = pd.to_datetime(["2025-07-07T00:00Z"])
    supplied["entry_price"] = 100.0
    result = _account((5.0, -3.0), (100.0, 110.0), (5.0, 3.0)).compute_round_trips(
        initial_state="supplied", opening_lots=supplied, opening_as_of="2025-07-08T00:00Z"
    )
    assert result.segments["quantity"].to_list() == [3.0]
    assert result.metrics["realized_gross_pnl"] == 30.0
    assert result.metrics["realized_net_pnl"] == 24.0
    assert result.open_lots["quantity"].to_list() == [2.0]
    assert result.metadata["excluded_before_opening"] == 1
    assert result.executions.height == 2


@pytest.mark.parametrize(
    "column,value",
    [
        ("quantity", 0.0),
        ("quantity", float("inf")),
        ("entry_price", float("inf")),
        ("remaining_commission", float("inf")),
        ("multiplier", 0.0),
    ],
)
def test_invalid_opening_numbers_raise_before_mutating_owned_or_caller_input(column, value):
    supplied = _opening()
    supplied[column] = value
    before = supplied.copy(deep=True)
    analysis = _account()
    executions = analysis.input_data.transactions
    with pytest.raises(ValueError):
        analysis.compute_round_trips(
            initial_state="supplied", opening_lots=supplied, opening_as_of="2025-07-07T00:00Z"
        )
    pd.testing.assert_frame_equal(supplied, before)
    assert analysis.input_data.transactions.equals(executions)


def test_unavailable_quote_or_basis_keeps_unrealized_totals_unknown_and_reports_coverage():
    result = _account().compute_round_trips(initial_state="flat", as_of="2025-07-09T22:00Z")
    assert result.open_lots["mark"].to_list() == [None]
    assert result.metrics["unrealized_gross_pnl"] is None
    assert result.metrics["marked_gross_pnl"] is None
    assert result.metrics["n_marked_open_lots"] == 0
    assert result.metrics["n_known_basis_open_lots"] == 1
    assert result.metrics["n_known_unrealized_gross_lots"] == 0


def test_duplicate_quotes_and_mixed_naive_aware_instants_are_not_silently_ordered_or_localized():
    with pytest.raises(ValueError, match="duplicate"):
        _account().compute_round_trips(
            initial_state="flat", marks=pd.concat([_marks(), _marks()]), as_of="2025-07-09T22:00Z"
        )
    prices = _marks()
    prices["timestamp"] = prices["timestamp"].dt.tz_localize(None)
    with pytest.raises(ValueError, match="naive/aware"):
        _account().compute_round_trips(
            initial_state="flat", marks=prices, as_of="2025-07-09T22:00Z"
        )
    with pytest.raises(ValueError, match="naive/aware"):
        _account().compute_round_trips(initial_state="flat", as_of="2025-07-09T22:00")


def test_actual_instants_in_other_timezones_are_compared_without_changing_their_meaning():
    prices = _marks()
    prices["timestamp"] = prices["timestamp"].dt.tz_convert("America/New_York")
    result = _account().compute_round_trips(
        initial_state="flat", marks=prices, as_of="2025-07-09T18:00-04:00"
    )
    assert result.metrics["unrealized_gross_pnl"] == 60.0
    assert result.open_lots["mark_timestamp"].cast(pl.Int64).to_list() == [
        pd.Timestamp("2025-07-09T21:00Z").value
    ]


def test_flat_closed_book_can_request_empty_synthetic_table_without_fabricated_quotes():
    result = _account((1.0, -1.0), (100.0, 110.0), (0.0, 0.0)).compute_round_trips(
        initial_state="flat", synthetic_close=True, as_of="2025-07-09T22:00Z"
    )
    assert result.synthetic_segments.is_empty()
    assert result.metrics["unrealized_net_pnl"] == 0.0
    assert result.metrics["marked_net_pnl"] == 10.0


def test_fractional_opening_basis_is_not_truncated_to_integer_execution_price_dtype():
    supplied = _opening()
    supplied["entry_price"] = 90.5
    result = _account((-5,), (110,), (0.0,)).compute_round_trips(
        initial_state="supplied", opening_lots=supplied, opening_as_of="2025-07-07T00:00Z"
    )
    assert result.segments["entry_price"].to_list() == [90.5]
    assert result.metrics["realized_gross_pnl"] == 97.5


def test_mark_and_as_of_ns_are_not_truncated_when_the_mark_is_only_one_nanosecond_old():
    prices = _marks().iloc[:1].copy()
    prices["timestamp"] = pd.to_datetime(["2025-07-09T21:00:00.000000001Z"])
    result = _account().compute_round_trips(
        initial_state="flat", marks=prices, as_of="2025-07-09T21:00:00.000000002Z"
    )
    assert result.open_lots["mark_age_seconds"].to_list() == [1e-9]
    assert result.open_lots["mark_timestamp"].cast(pl.Int64).to_list() == [
        pd.Timestamp("2025-07-09T21:00:00.000000001Z").value
    ]


def test_mixed_large_integer_and_float_price_types_fail_instead_of_rounding_supplied_basis():
    supplied = _opening()
    with pytest.raises(ValueError, match="round integer prices"):
        _account((-5,), (2**53 + 1,), (0.0,)).compute_round_trips(
            initial_state="supplied", opening_lots=supplied, opening_as_of="2025-07-07T00:00Z"
        )


def test_short_opening_lot_and_rebate_mark_preserve_direction_and_remaining_fee_sign():
    supplied = _opening()
    supplied["quantity"] = -5.0
    supplied["remaining_commission"] = -5.0
    prices = _marks()
    prices["price"] = 80.0
    result = _account((2.0,), (85.0,), (2.0,)).compute_round_trips(
        initial_state="supplied",
        opening_lots=supplied,
        opening_as_of="2025-07-07T00:00Z",
        marks=prices,
        as_of="2025-07-09T22:00Z",
    )
    assert result.metrics["realized_gross_pnl"] == 10.0
    assert result.metrics["realized_net_pnl"] == 10.0
    assert result.open_lots["direction"].to_list() == ["short"]
    assert result.open_lots["remaining_commission"].to_list() == [-3.0]
    assert result.open_lots["marked_notional"].to_list() == [-240.0]
    assert result.metrics["unrealized_gross_pnl"] == 30.0
    assert result.metrics["unrealized_net_pnl"] == 33.0


def test_supplied_inventory_cannot_hide_opposing_lots_or_changing_point_multipliers():
    supplied = pd.concat([_opening(), _opening()], ignore_index=True)
    supplied.loc[1, "lot_id"] = "carried1"
    supplied.loc[1, "quantity"] = -5.0
    with pytest.raises(ValueError, match="opposing"):
        _account().compute_round_trips(
            initial_state="supplied", opening_lots=supplied, opening_as_of="2025-07-07T00:00Z"
        )
    supplied = _opening()
    supplied["multiplier"] = 50.0
    with pytest.raises(ValueError, match="multiplier cannot change"):
        _account().compute_round_trips(
            initial_state="supplied", opening_lots=supplied, opening_as_of="2025-07-07T00:00Z"
        )


def test_declared_feed_timezone_resolves_naive_event_and_mark_instants_without_utc_guessing():
    dates = pd.date_range("2025-07-07", periods=2)
    analysis = PortfolioAnalysis.from_frames(
        pd.Series([0.01, 0.02], index=dates),
        transactions=pd.DataFrame(
            {
                "timestamp": dates,
                "asset": [7, 7],
                "quantity": [1.0, -1.0],
                "price": [100.0, 110.0],
                "commission": [0.0, 0.0],
            }
        ),
        feed_spec={"timezone": "America/New_York"},
        calendar=None,
        transaction_timestamp_col="timestamp",
    )
    result = analysis.compute_round_trips(initial_state="flat", as_of="2025-07-07T20:00-04:00")
    assert result.open_lots["quantity"].to_list() == [1.0]
    assert result.open_lots["entry_timestamp"].cast(pl.Int64).to_list() == [
        pd.Timestamp("2025-07-07T00:00-04:00").value
    ]
    assert result.metadata["excluded_future_executions"] == 1


def test_marked_total_sums_exact_cash_components_before_rounding_large_offsetting_pnl():
    prices = _marks().iloc[:1].copy()
    prices["price"] = 1
    result = _account((1, -1, 1), (0, 10**16, 10**16), (0.0, 0.0, 0.0)).compute_round_trips(
        initial_state="flat", marks=prices, as_of="2025-07-09T22:00Z"
    )
    # First round trip gains10**16; open lot loses10**16-1. The combined PnL is1.
    assert result.metrics["marked_gross_pnl"] == 1.0
    assert result.metrics["marked_net_pnl"] == 1.0
