"""Hypothetical closes cannot precede entries, change actual FIFO or erase signed/fractional units."""

from copy import deepcopy

import pandas as pd
import pytest
from ml4t.specs import FeedSpec

from ml4t.diagnostic.compat.pyfolio import round_trips as rt


def _book(*, quantity=(1.0,), price=(100.0,), value=110.0, day="2025-07-07", times=None):
    tx = pd.DataFrame(
        {"symbol": ["A"] * len(quantity), "amount": quantity, "price": price},
        index=pd.to_datetime([day + "T15:00Z"] * len(quantity) if times is None else times),
    )
    positions = pd.DataFrame(
        {"A": [value], "cash": [900.0]}, index=pd.to_datetime([day + "T00:00Z"])
    )
    return positions, tx


def test_daily_label_selects_session_close_not_synthetic_sale_before_real_entry():
    positions, tx = _book()
    before = [deepcopy(positions), deepcopy(tx)]
    closed = rt.add_closing_transactions(positions, tx)
    assert closed.index.to_list() == [tx.index[0], pd.Timestamp("2025-07-07T20:00Z")]
    assert closed["is_synthetic"].to_list() == [False, True]
    assert closed["amount"].to_list() == [1.0, -1]
    result = rt.extract_round_trips(closed)
    assert result["pnl"].to_list() == [10.0] and result["long"].to_list() == [True]
    assert result["duration"].to_list() == [pd.Timedelta(hours=5)]
    assert result["is_synthetic"].to_list() == [True]
    native, _ = rt._native_round_trips(closed)
    assert native.executions.height == 1 and native.completed_trades().is_empty()
    assert native.metrics["realized_gross_pnl"] == 0.0
    assert native.open_lots["quantity"].to_list() == [1.0]
    pd.testing.assert_frame_equal(positions, before[0])
    pd.testing.assert_frame_equal(tx, before[1])


def test_exchange_half_day_uses_actual_early_close_not_hardcoded_regular_hour():
    positions, tx = _book(day="2025-07-03")
    result = rt.add_closing_transactions(positions, tx)
    assert result.index[-1] == pd.Timestamp("2025-07-03T17:00Z")


def test_explicit_none_disables_calendar_inference_and_explicit_asof_supplies_actual_availability():
    positions, tx = _book()
    for options in (
        {"calendar": None},
        {"feed_spec": FeedSpec(time_contract_version=2, calendar=None)},
    ):
        with pytest.raises(ValueError, match="precedes actual executions"):
            rt.add_closing_transactions(positions, tx, **options)
        result = rt.add_closing_transactions(positions, tx, as_of="2025-07-07T21:00Z", **options)
        assert result.index[-1] == pd.Timestamp("2025-07-07T21:00Z")
        assert result.attrs["diagnostic_closing"]["calendar"] is None
    override = rt.add_closing_transactions(
        positions, tx, feed_spec=FeedSpec(time_contract_version=2, calendar=None), calendar="NYSE"
    )
    assert override.index[-1] == pd.Timestamp("2025-07-07T20:00Z")


@pytest.mark.parametrize(
    "quantity,price,value,pnl,direction",
    [
        ((-2.0,), (100.0,), -180.0, 20.0, False),
        ((0.1, 0.2), (100.0, 100.0), 33.0, 3.0, True),
        ((1.0,), (-100.0,), -90.0, 10.0, True),
        ((1.0,), (100.0,), 0.0, -100.0, True),
        ((1.0,), (0.0,), 10.0, 10.0, True),
    ],
)
def test_signed_fractional_zero_mark_and_zero_basis_closures_keep_real_cash_contracts(
    quantity, price, value, pnl, direction
):
    positions, tx = _book(quantity=quantity, price=price, value=value)
    closed = rt.add_closing_transactions(positions, tx)
    result = rt.extract_round_trips(closed)
    assert result["pnl"].to_list() == [pnl] and result["long"].to_list() == [direction]
    assert result["is_synthetic"].to_list() == [True]
    if price == (0.0,):
        assert result["rt_returns"].isna().all()


def test_actual_realization_and_hypothetical_mark_stay_separate_with_same_supplied_nav():
    positions, tx = _book(
        quantity=(2.0, -1.0),
        price=(100.0, 110.0),
        value=120.0,
        day="2025-07-08",
        times=["2025-07-07T15:00Z", "2025-07-08T15:00Z"],
    )
    closed = rt.add_closing_transactions(positions, tx)
    result = rt.extract_round_trips(closed, pd.Series([1000.0], index=positions.index))
    assert result["pnl"].to_list() == [10.0, 20.0]
    assert result["returns"].to_list() == [0.01, 0.02]
    assert result["is_synthetic"].to_list() == [False, True]
    native, _ = rt._native_round_trips(closed)
    assert native.metrics["realized_gross_pnl"] == 10.0
    assert native.completed_trades().height == 1 and native.open_lots["quantity"].to_list() == [1.0]


def test_large_net_quantity_exact_projection_cannot_leave_a_phantom_hypothetical_unit():
    positions, tx = _book(quantity=(1e16, 1.0), price=(1.0, 1.0), value=10**16 + 1)
    closed = rt.add_closing_transactions(positions, tx)
    assert closed["amount_exact"].iloc[-1] == "-10000000000000001"
    native, _ = rt._native_round_trips(closed)
    assert native.synthetic_segments["quantity_exact"].to_list() == ["10000000000000001"]
    assert rt.extract_round_trips(closed)["pnl"].to_list() == [0.0]


def test_unknown_marks_inconsistent_ending_book_and_double_projection_are_rejected():
    positions, tx = _book()
    for value in (None, float("inf")):
        unknown = positions.copy(deep=True)
        unknown["A"] = value
        with pytest.raises(ValueError, match="finite ending marked values"):
            rt.add_closing_transactions(unknown, tx)
    positions, tx = _book(quantity=(1.0, -1.0), price=(100.0, 110.0))
    with pytest.raises(ValueError, match="disagree"):
        rt.add_closing_transactions(positions, tx)
    positions, tx = _book()
    closed = rt.add_closing_transactions(positions, tx)
    with pytest.raises(ValueError, match="already synthetically closed"):
        rt.add_closing_transactions(positions, closed)
    damaged = closed.copy(deep=True)
    damaged.loc[damaged.is_synthetic, "amount"] = -2.0
    with pytest.raises(ValueError, match="exact projection"):
        rt.extract_round_trips(damaged)


def test_naive_execution_clock_needs_timezone_before_a_calendar_close_can_be_used():
    positions, tx = _book()
    positions.index = positions.index.tz_localize(None)
    tx.index = tx.index.tz_localize(None)
    with pytest.raises(ValueError, match="aware executions"):
        rt.add_closing_transactions(positions, tx)
    closed = rt.add_closing_transactions(
        positions, tx, feed_spec=FeedSpec(time_contract_version=2, timezone="UTC")
    )
    assert closed.index[-1] == pd.Timestamp("2025-07-07T20:00Z")
    assert rt.extract_round_trips(closed)["duration"].to_list() == [pd.Timedelta(hours=5)]


def test_actual_published_midnight_closing_error_and_zero_mark_omission_are_reconciled():
    reference = pytest.importorskip("pyfolio.round_trips")
    positions, tx = _book()
    old = reference.extract_round_trips(reference.add_closing_transactions(positions, tx))
    native = rt.extract_round_trips(rt.add_closing_transactions(positions, tx))
    assert old["pnl"].to_list() == native["pnl"].to_list() == [10.0]
    assert old["long"].to_list() == [False] and native["long"].to_list() == [True]
    assert old["close_dt"].iloc[0] == tx.index[0]
    assert native["close_dt"].iloc[0] == pd.Timestamp("2025-07-07T20:00Z")
    positions["A"] = 0.0
    assert len(reference.add_closing_transactions(positions, tx)) == 1
    zero = rt.extract_round_trips(rt.add_closing_transactions(positions, tx))
    assert zero["pnl"].to_list() == [-100.0] and zero["is_synthetic"].to_list() == [True]


def test_flat_terminal_book_adds_no_closing_and_keeps_real_zero_quantity_records():
    positions, tx = _book(quantity=(1.0, -1.0, 0.0), price=(100.0, 110.0, 110.0), value=0.0)
    closed = rt.add_closing_transactions(positions, tx)
    assert len(closed) == 3 and not closed.is_synthetic.any()
    assert closed["amount"].to_list() == [1.0, -1.0, 0.0]
    assert rt.extract_round_trips(closed)["pnl"].to_list() == [10.0]


def test_explicit_asof_other_timezone_is_projected_on_the_original_aware_transaction_clock():
    positions, tx = _book()
    tx.index = tx.index.tz_convert("America/New_York")
    closed = rt.add_closing_transactions(positions, tx, as_of="2025-07-07T20:00Z")
    assert isinstance(closed.index, pd.DatetimeIndex)
    assert str(closed.index.tz) == "America/New_York"
    assert closed.index[0] == tx.index[0]
    assert rt.extract_round_trips(closed)["duration"].to_list() == [pd.Timedelta(hours=5)]


def test_ambiguous_hypothetical_flags_are_rejected_instead_of_becoming_actual_rows():
    positions, tx = _book()
    closed = rt.add_closing_transactions(positions, tx)
    closed["is_synthetic"] = [None, None]
    with pytest.raises(ValueError, match="nonmissing booleans"):
        rt.extract_round_trips(closed)
