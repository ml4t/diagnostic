"""Actual fractional executions must reconcile matched lots, fees and open quantities."""

import pandas as pd
import polars as pl
import pytest

from ml4t.diagnostic.evaluation import PortfolioAnalysis


def _account(quantities, prices, commissions, *, instants=None, multipliers=None):
    dates = pd.date_range("2025-07-07", periods=len(quantities), tz="UTC")
    executions = pd.DataFrame(
        {
            "timestamp": dates if instants is None else instants,
            "asset": [7] * len(quantities),
            "quantity": quantities,
            "price": prices,
            "commission": commissions,
            "slippage": [0.0] * len(quantities),
            "record_id": [f"fill{index}" for index in range(len(quantities))],
        }
    )
    if multipliers is not None:
        executions["multiplier"] = multipliers
    return PortfolioAnalysis.from_frames(
        pd.Series([0.0] * len(dates), index=dates),
        transactions=executions,
        calendar=None,
        transaction_timestamp_col="timestamp",
    )


def test_fifo_requires_an_explicit_opening_state_instead_of_guessing_a_flat_book():
    with pytest.raises(ValueError, match="opening|initial_state"):
        _account([10.0, -10.0], [100.0, 120.0], [0.0, 0.0]).compute_round_trips()


def test_partial_fifo_realizes220_leaves3_at110_and_allocates_each_cash_fee_once():
    analysis = _account([10.0, 5.0, -12.0], [100.0, 110.0, 120.0], [10.0, 10.0, 12.0])
    before = analysis.input_data.transactions
    result = analysis.compute_round_trips(initial_state="flat")
    assert result.segments["quantity"].to_list() == [10.0, 2.0]
    assert result.segments["gross_pnl"].to_list() == [200.0, 20.0]
    assert result.segments["commission"].to_list() == [20.0, 6.0]
    assert result.metrics["realized_gross_pnl"] == 220.0
    assert result.metrics["realized_net_pnl"] == 194.0
    assert result.open_lots["quantity"].to_list() == [3.0]
    assert result.open_lots["entry_price"].to_list() == [110.0]
    assert result.open_lots["remaining_commission"].to_list() == [6.0]
    assert result.metrics["allocated_commission"] == 26.0
    assert result.metrics["remaining_commission"] == 6.0
    assert analysis.input_data.transactions.equals(before)


def test_fractional_decimal_units_close_without_integer_truncation_or_phantom_lots():
    result = _account([0.1, 0.2, -0.3], [100.0, 110.0, 120.0], [0.0, 0.0, 0.0]).compute_round_trips(
        initial_state="flat"
    )
    assert result.segments["quantity"].to_list() == [0.1, 0.2]
    assert result.segments["quantity_exact"].to_list() == ["1/10", "1/5"]
    assert result.metrics["realized_gross_pnl"] == pytest.approx(4.0)
    assert result.open_lots.is_empty()


def test_short_reversal_closes_old_lots_before_opening_the_opposite_remainder():
    result = _account(
        [-1.5, 2.0, -0.25], [100.0, 90.0, 95.0], [1.5, 2.0, 0.25]
    ).compute_round_trips(initial_state="flat")
    assert result.segments["gross_pnl"].to_list() == [15.0, 1.25]
    assert result.metrics["realized_net_pnl"] == pytest.approx(12.75)
    assert result.metrics["allocated_commission"] == 3.5
    assert result.open_lots["quantity"].to_list() == [0.25]
    assert result.open_lots["entry_price"].to_list() == [90.0]
    assert result.open_lots["remaining_commission"].to_list() == [0.25]


def test_unknown_commission_keeps_gross_pnl_but_never_fabricates_net_pnl():
    result = _account([10.0, 5.0, -12.0], [100.0, 110.0, 120.0], [None] * 3).compute_round_trips(
        initial_state="flat"
    )
    assert result.metrics["realized_gross_pnl"] == 220.0
    assert result.metrics["realized_net_pnl"] is None
    assert result.segments["net_pnl"].null_count() == 2


def test_same_instant_source_order_and_original_nanoseconds_survive_matching():
    instant = pd.Timestamp("2025-07-07T15:00:00.000000001Z")
    result = _account(
        [1.0, -1.0], [100.0, 120.0], [0.0, 0.0], instants=[instant, instant]
    ).compute_round_trips(initial_state="flat")
    assert result.metrics["realized_gross_pnl"] == 20.0
    assert result.open_lots.is_empty()
    assert result.segments["entry_timestamp"].cast(pl.Int64).to_list() == [instant.value]
    assert result.segments["exit_timestamp"].cast(pl.Int64).to_list() == [instant.value]


def test_signed_negative_contract_prices_and_point_multipliers_are_not_ignored():
    result = _account(
        [2.0, -2.0], [-100.0, -90.0], [1.0, 1.0], multipliers=[50.0, 50.0]
    ).compute_round_trips(initial_state="flat")
    assert result.metrics["realized_gross_pnl"] == 1000.0
    assert result.metrics["realized_net_pnl"] == 998.0
    assert result.segments["gross_lot_return"].to_list() == [0.1]


def test_zero_entry_price_keeps_cash_pnl_but_has_no_fabricated_finite_lot_return():
    result = _account([1.0, -1.0], [0.0, 10.0], [0.0, 0.0]).compute_round_trips(
        initial_state="flat"
    )
    assert result.metrics["realized_gross_pnl"] == 10.0
    assert result.segments["gross_lot_return"].to_list() == [None]


def test_changing_point_multiplier_is_rejected_before_mutating_owned_inputs():
    analysis = _account([1.0, -1.0], [100.0, 110.0], [0.0, 0.0], multipliers=[1.0, 50.0])
    before = analysis.input_data.transactions
    with pytest.raises(ValueError, match="multiplier cannot change"):
        analysis.compute_round_trips(initial_state="flat")
    assert analysis.input_data.transactions.equals(before)


def test_currency_overflow_raises_an_actionable_error_without_changing_the_account():
    analysis = _account([1.0e308, -1.0e308], [100.0, 200.0], [0.0, 0.0])
    before = analysis.input_data.transactions
    with pytest.raises(ValueError, match="finite Float64 account currency"):
        analysis.compute_round_trips(initial_state="flat")
    assert analysis.input_data.transactions.equals(before)


def test_zero_quantity_fees_remain_unallocated_cash_events_not_fabricated_lots():
    result = _account([0.0], [100.0], [1.0]).compute_round_trips(initial_state="flat")
    assert result.segments.is_empty() and result.open_lots.is_empty()
    assert result.metrics["realized_gross_pnl"] == 0.0
    assert result.metrics["unallocated_zero_quantity_commission"] == 1.0
    assert result.executions["commission"].to_list() == [1.0]


def test_integer_execution_prices_are_not_rounded_before_pnl_or_in_result_tables():
    price = 2**53 + 1
    result = _account([1, -1], [price, price + 1], [0.0, 0.0]).compute_round_trips(
        initial_state="flat"
    )
    assert result.metrics["realized_gross_pnl"] == 1.0
    assert result.segments["entry_price"].to_list() == [price]
    assert result.segments["exit_price"].to_list() == [price + 1]


def test_large_integer_source_record_ids_remain_exact_in_matched_segments():
    analysis = _account([1.0, -1.0], [100.0, 120.0], [0.0, 0.0])
    original = analysis.input_data
    executions = original.transactions.with_columns(pl.Series("record_id", [2**53 + 1, 2**53 + 2]))
    owned = PortfolioAnalysis.from_frames(
        original.returns,
        transactions=executions,
        calendar=None,
        transaction_timestamp_col="timestamp",
    )
    result = owned.compute_round_trips(initial_state="flat")
    assert result.segments["entry_record_id"].to_list() == [2**53 + 1]
    assert result.segments["exit_record_id"].to_list() == [2**53 + 2]


def test_published_reference_integer_fifo_matches_cash_pnl_not_segment_count():
    reference = pytest.importorskip("pyfolio.round_trips")
    analysis = _account([10.0, 5.0, -12.0], [100.0, 110.0, 120.0], [0.0, 0.0, 0.0])
    transactions = (
        analysis.input_data.transactions.to_pandas()
        .set_index("timestamp")
        .rename(columns={"asset": "symbol", "quantity": "amount"})
    )
    # The reference uses display symbols; native stable asset7 remains an integer.
    transactions["symbol"] = transactions["symbol"].map({7: "A"})
    legacy = reference.extract_round_trips(transactions)
    native = analysis.compute_round_trips(initial_state="flat")
    assert legacy["pnl"].to_list() == [220.0]
    assert native.metrics["realized_gross_pnl"] == 220.0
    assert native.segments.height == 2
    assert len(legacy) == 1
    assert legacy["rt_returns"].iloc[0] == pytest.approx(220 / 1220)


def test_fractional_reference_truncation_is_not_copied_to_native_matching():
    reference = pytest.importorskip("pyfolio.round_trips")
    analysis = _account([0.1, 0.2, -0.3], [100.0, 110.0, 120.0], [0.0, 0.0, 0.0])
    transactions = (
        analysis.input_data.transactions.to_pandas()
        .set_index("timestamp")
        .rename(columns={"asset": "symbol", "quantity": "amount"})
    )
    transactions["symbol"] = transactions["symbol"].map({7: "A"})
    with pytest.raises(KeyError, match="close_dt"):
        reference.extract_round_trips(transactions)
    native = analysis.compute_round_trips(initial_state="flat")
    assert native.metrics["realized_gross_pnl"] == 4.0
    assert native.open_lots.is_empty()


def test_integer_reference_symbol_upcast_error_does_not_change_native_asset_identity():
    reference = pytest.importorskip("pyfolio.round_trips")
    analysis = _account([10.0, 5.0, -12.0], [100.0, 110.0, 120.0], [0.0, 0.0, 0.0])
    transactions = (
        analysis.input_data.transactions.to_pandas()
        .set_index("timestamp")
        .rename(columns={"asset": "symbol", "quantity": "amount"})
    )
    with pytest.raises(TypeError, match="multiply sequence by non-int"):
        reference.extract_round_trips(transactions)
    native = analysis.compute_round_trips(initial_state="flat")
    assert native.metrics["realized_gross_pnl"] == 220.0
    assert native.segments["asset"].to_list() == [7, 7]
