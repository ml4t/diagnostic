"""Actual cash-price replay cannot collapse fills, borrow future quotes or invent account cash."""

from copy import deepcopy

import pandas as pd
import polars as pl
import pytest
from ml4t.specs import FeedSpec

from ml4t.diagnostic.evaluation import PortfolioAnalysis


def _trips(
    *,
    quantity=(1.0, 1.0),
    price=(100.0, 100.0),
    fee=(0.5, 0.5),
    opening=True,
    matching="actual_fifo",
    marks=None,
    times=None,
    feed=None,
):
    start, fill, end = pd.to_datetime(
        ["2025-07-07T13:30Z", "2025-07-07T14:00Z", "2025-07-07T20:00Z"]
    )
    transactions = pd.DataFrame(
        {
            "timestamp": [fill] * len(quantity) if times is None else times,
            "asset": ["A"] * len(quantity),
            "quantity": quantity,
            "price": price,
            "commission": fee,
        }
    )
    analysis = PortfolioAnalysis.from_frames(
        pd.Series([0.0], index=[end.normalize()]),
        transactions=transactions,
        calendar=None,
        feed_spec=feed,
    )
    if marks is None:
        marks = pd.DataFrame(
            {"timestamp": [start, start], "asset": ["A", "B"], "price": [100.0, 100.0]}
        )
    result = analysis.compute_round_trips(
        initial_state="supplied" if opening else "flat",
        opening_lots=pd.DataFrame(
            {"asset": ["A", "B"], "quantity": [4.0, -3.0], "remaining_commission": [7.0, 9.0]}
        )
        if opening
        else None,
        opening_as_of=start if opening else None,
        marks=marks,
        as_of=end,
        matching=matching,
    )
    return analysis, result, start, end


def test_same_instant_actual_fills_preserve_each_intermediate_cash_and_quantity_state():
    _, trips, start, end = _trips()
    before = deepcopy(trips)
    result = trips.reconstruct_intraday_exposure(opening_cash=900.0, cash_model="cash_price")
    assert result.timeline["gross_exposure"].to_list() == [700.0, 800.0, 900.0, 900.0]
    assert result.timeline["cash"].to_list() == [900.0, 799.5, 699.0, 699.0]
    assert result.timeline["equity"].to_list() == [1000.0, 999.5, 999.0, 999.0]
    assert result.timeline["observation_id"].to_list() == [0, 1, 2, 3]
    assert result.timeline["event_kind"].to_list() == [
        "opening",
        "execution",
        "execution",
        "terminal",
    ]
    dates = pd.DatetimeIndex(result.timeline["date"].to_pandas())
    assert dates[0] == start and dates[-1] == end and dates[1] == dates[2]
    assert result.positions.filter(pl.col("asset") == "A")["quantity_exact"].to_list() == [
        "4",
        "5",
        "6",
        "6",
    ]
    assert result.timeline["gross_hhi"].to_list() == pytest.approx([25 / 49, 34 / 64, 5 / 9, 5 / 9])
    assert result.metrics["max_gross_exposure"] == pytest.approx(900 / 999)
    assert trips.executions.equals(before.executions) and trips.opening_lots.equals(
        before.opening_lots
    )
    assert trips.marks.equals(before.marks) and trips.metadata == before.metadata
    assert trips.completed_trades().is_empty()


@pytest.mark.parametrize(
    "opening_cash,fees,expected_cash",
    [
        (None, (0.5, 0.5), [None] * 4),
        (900.0, (None, 0.5), [900.0, None, None, None]),
        (900.0, (0.5, -0.5), [900.0, 799.5, 700.0, 700.0]),
    ],
)
def test_missing_cash_or_commission_stays_unknown_and_rebates_are_not_dropped(
    opening_cash, fees, expected_cash
):
    _, trips, _, _ = _trips(fee=fees)
    result = trips.reconstruct_intraday_exposure(opening_cash=opening_cash, cash_model="cash_price")
    assert result.timeline["cash"].to_list() == expected_cash
    assert result.timeline["gross_exposure"].to_list() == [700.0, 800.0, 900.0, 900.0]
    assert result.timeline["equity"].null_count() == expected_cash.count(None)
    assert result.coverage["n_normalized_observations"] == 4 - expected_cash.count(None)


def test_future_mark_cannot_value_opening_book_or_be_filled_backwards():
    future = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2025-07-07T20:00Z"] * 2),
            "asset": ["A", "B"],
            "price": [100.0, 100.0],
        }
    )
    _, trips, _, _ = _trips(marks=future)
    with pytest.raises(ValueError, match="known-at supplied mark"):
        trips.reconstruct_intraday_exposure(opening_cash=900.0, cash_model="cash_price")


def test_mark_only_change_and_future_quote_cutoff_keep_original_exact_quote_age():
    marks = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2025-07-07T13:30Z"] * 2 + ["2025-07-07T15:00:00.000000007Z", "2025-07-07T21:00Z"],
                format="mixed",
            ),
            "asset": ["A", "B", "A", "A"],
            "price": [100.0, 100.0, 110.0, 9999.0],
        }
    )
    _, trips, _, _ = _trips(marks=marks)
    result = trips.reconstruct_intraday_exposure(opening_cash=900.0, cash_model="cash_price")
    assert result.timeline["gross_exposure"].to_list() == [700.0, 800.0, 900.0, 960.0, 960.0]
    assert result.timeline["equity"].to_list() == [1000.0, 999.5, 999.0, 1059.0, 1059.0]
    assert result.timeline["date"].cast(pl.Int64)[3] == marks.timestamp.iloc[2].value
    a = result.positions.filter(pl.col("asset") == "A")
    assert a["mark_age_ns"][-1] == pd.Timedelta("5h").value - 7
    assert a["price"].max() == 110.0


def test_execution_price_and_commission_change_cash_but_never_replace_supplied_mark():
    _, trips, start, _ = _trips(quantity=(1.0,), price=(101.0,), fee=(1.0,), opening=False)
    result = trips.reconstruct_intraday_exposure(
        opening_cash=1000.0, opening_as_of=start, cash_model="cash_price"
    )
    assert result.timeline["cash"].to_list() == [1000.0, 898.0, 898.0]
    assert result.timeline["equity"].to_list() == [1000.0, 998.0, 998.0]
    assert result.positions["price"].to_list() == [100.0, 100.0]


def test_consecutive_vwap_analytical_profile_still_replays_original_ordered_prices():
    _, trips, start, _ = _trips(
        price=(100.0, 120.0), fee=(0.0, 0.0), opening=False, matching="consecutive_vwap"
    )
    assert trips.matching_groups.height == 1
    result = trips.reconstruct_intraday_exposure(
        opening_cash=1000.0, opening_as_of=start, cash_model="cash_price"
    )
    assert result.timeline["cash"].to_list() == [1000.0, 900.0, 780.0, 780.0]
    assert result.timeline["gross_exposure"].to_list() == [0.0, 100.0, 200.0, 200.0]


def test_fractional_flat_book_and_zero_quantity_fee_keep_exact_units_and_no_phantom_positions():
    _, trips, start, _ = _trips(
        quantity=(0.1, 0.2, -0.3, 0.0), price=(100.0,) * 4, fee=(0.0, 0.0, 0.0, 2.0), opening=False
    )
    result = trips.reconstruct_intraday_exposure(
        opening_cash=1000.0, opening_as_of=start, cash_model="cash_price"
    )
    assert result.timeline["open_positions"].to_list() == [0, 1, 1, 0, 0, 0]
    assert result.timeline["gross_exposure"].to_list() == [0.0, 10.0, 30.0, 0.0, 0.0, 0.0]
    assert result.timeline["cash"].to_list() == [1000.0, 990.0, 970.0, 1000.0, 998.0, 998.0]
    assert result.positions["quantity_exact"].to_list() == ["1/10", "3/10"]
    assert result.metrics["time_in_market"] == pytest.approx(2 / 6)


@pytest.mark.parametrize("mark,expected_gross", [(0.0, 0.0), (-10.0, 10.0)])
def test_zero_and_negative_price_holdings_counts_follow_units_not_sign_of_marked_cash(
    mark, expected_gross
):
    marks = pd.DataFrame(
        {"timestamp": pd.to_datetime(["2025-07-07T13:30Z"]), "asset": ["A"], "price": [mark]}
    )
    _, trips, start, _ = _trips(
        quantity=(1.0,), price=(mark,), fee=(0.0,), opening=False, marks=marks
    )
    result = trips.reconstruct_intraday_exposure(
        opening_cash=1000.0, opening_as_of=start, cash_model="cash_price"
    )
    assert result.timeline["open_positions"].to_list() == [0, 1, 1]
    assert result.timeline["long_positions"].to_list() == [0, 1, 1]
    assert result.timeline["short_positions"].to_list() == [0, 0, 0]
    assert result.timeline["gross_exposure"].to_list() == [0.0, expected_gross, expected_gross]


def test_cash_anchor_and_accounting_model_are_required_not_inferred_from_returns_or_lot_basis():
    _, trips, start, _ = _trips(opening=False)
    with pytest.raises(ValueError, match="explicit opening_as_of"):
        trips.reconstruct_intraday_exposure(opening_cash=1000.0, cash_model="cash_price")
    with pytest.raises(ValueError, match="cash_model"):
        trips.reconstruct_intraday_exposure(
            opening_cash=1000.0, opening_as_of=start, cash_model="margin"
        )
    _, supplied, _, _ = _trips()
    with pytest.raises(ValueError, match="must equal"):
        supplied.reconstruct_intraday_exposure(
            opening_cash=900.0, cash_model="cash_price", opening_as_of=start + pd.Timedelta("1h")
        )
    for bad in (True, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="finite"):
            supplied.reconstruct_intraday_exposure(opening_cash=bad, cash_model="cash_price")


def test_reconstruction_inherits_explicit_none_and_accepts_explicit_calendar_override():
    _, trips, _, _ = _trips(feed=FeedSpec(time_contract_version=2, calendar="CME_Equity"))
    disabled = trips.reconstruct_intraday_exposure(opening_cash=900.0, cash_model="cash_price")
    assert disabled.metadata["calendar"] is None and "session" not in disabled.timeline.columns
    scheduled = trips.reconstruct_intraday_exposure(
        opening_cash=900.0, cash_model="cash_price", calendar="NYSE"
    )
    assert scheduled.timeline["session"].n_unique() == 1


def test_before_anchor_and_future_fills_do_not_change_complete_carried_inventory_or_cash():
    times = pd.to_datetime(["2025-07-07T13:00Z", "2025-07-07T14:00Z", "2025-07-07T21:00Z"])
    _, trips, _, _ = _trips(
        quantity=(900.0, 1.0, 900.0), price=(100.0,) * 3, fee=(99.0, 0.5, 99.0), times=times
    )
    result = trips.reconstruct_intraday_exposure(opening_cash=900.0, cash_model="cash_price")
    assert result.timeline["cash"].to_list() == [900.0, 799.5, 799.5]
    assert result.timeline["gross_exposure"].to_list() == [700.0, 800.0, 800.0]
    assert trips.executions.height == 3
    assert result.timeline["execution_record_id"].drop_nulls().to_list() == [
        trips.executions["record_id"][1]
    ]


def test_large_integer_prices_and_asset_ids_keep_exact_cancellation_and_no_fake_price_rounding():
    dates = pd.to_datetime(["2025-07-07T14:00Z", "2025-07-07T15:00Z"])
    asset, price = 2**53 + 1, 2**53 + 1
    analysis = PortfolioAnalysis.from_frames(
        pd.Series([0.0], index=[dates[-1].normalize()]),
        calendar=None,
        transactions=pd.DataFrame(
            {
                "timestamp": dates,
                "asset": [asset] * 2,
                "quantity": [1, -1],
                "price": [price, price + 1],
                "commission": [0, 0],
            }
        ),
    )
    trips = analysis.compute_round_trips(
        initial_state="flat",
        as_of=dates[-1],
        marks=pd.DataFrame({"timestamp": dates[:1], "asset": [asset], "price": [price]}),
    )
    result = trips.reconstruct_intraday_exposure(
        opening_cash=1000, cash_model="cash_price", opening_as_of=dates[0]
    )
    assert result.timeline["equity"].to_list() == [1000.0, 1000.0, 1001.0]
    assert result.timeline["cash_exact"][-1] == "1001"
    assert result.positions["asset"].to_list() == [asset]
    assert result.positions.schema["asset"].is_integer()


def test_cme_overnight_calendar_retains_nanosecond_fills_and_actual_identity():
    dates = pd.to_datetime(["2025-07-06T23:30:00.000000007Z", "2025-07-07T00:30:00.000000009Z"])
    analysis = PortfolioAnalysis.from_frames(
        pd.Series([0.0], index=[dates[-1].normalize()]),
        calendar=None,
        transactions=pd.DataFrame(
            {
                "timestamp": dates[1:],
                "asset": ["A"],
                "quantity": [1.0],
                "price": [100.0],
                "commission": [0.0],
            }
        ),
    )
    trips = analysis.compute_round_trips(
        initial_state="flat",
        as_of=dates[-1],
        marks=pd.DataFrame({"timestamp": dates[:1], "asset": ["A"], "price": [100.0]}),
    )
    result = trips.reconstruct_intraday_exposure(
        opening_cash=1000.0, cash_model="cash_price", opening_as_of=dates[0], calendar="CME_Equity"
    )
    assert result.timeline["date"].cast(pl.Int64).to_list() == dates.as_unit("ns").asi8.tolist()
    assert result.timeline["session"].n_unique() == 1
    with pytest.raises(ValueError, match="sessions"):
        trips.reconstruct_intraday_exposure(
            opening_cash=1000.0, cash_model="cash_price", opening_as_of=dates[0], calendar="NYSE"
        )


def test_reconstructed_report_uses_unique_display_keys_without_altering_same_instant_timestamps(
    tmp_path,
):
    pytest.importorskip("plotly")
    from ml4t.diagnostic.visualization.portfolio import create_round_trip_dashboard

    analysis, trips, _, _ = _trips()
    result = trips.reconstruct_intraday_exposure(opening_cash=900.0, cash_model="cash_price")
    report = create_round_trip_dashboard(analysis, trips, intraday=result)
    x = report.figures["Intraday Holdings Exposure"].data[0].x
    assert len(x) == len(set(x)) == 4
    assert "observation_id=1" in x[1] and "observation_id=2" in x[2]
    exported = report.to_dict()["intraday"]["timeline"]
    assert exported[1]["date"] == exported[2]["date"]
    assert [row["cash"] for row in exported] == [900.0, 799.5, 699.0, 699.0]
    assert report.intraday.metadata["method"] == "explicit_cash_price_reconstruction"
    report.save_html(tmp_path / "reconstructed.html", include_plotlyjs=True)


@pytest.mark.parametrize("same_instant", [True, False])
def test_actual_published_pyfolio_estimator_same_instant_mean_defect_has_independent_sum_control(
    same_instant,
):
    legacy = pytest.importorskip("pyfolio.utils")
    times = pd.to_datetime(
        ["2025-07-07T14:00Z", "2025-07-07T14:00Z" if same_instant else "2025-07-07T14:01Z"]
    )
    _, trips, start, end = _trips(opening=False, fee=(0.0, 0.0), times=times)
    native = trips.reconstruct_intraday_exposure(
        opening_cash=1000.0, cash_model="cash_price", opening_as_of=start
    )
    positions = pd.DataFrame({"A": [200.0], "cash": [800.0]}, index=[end.normalize()])
    tx = pd.DataFrame({"symbol": ["A"] * 2, "amount": [1.0] * 2, "price": [100.0] * 2}, index=times)
    original = legacy.estimate_intraday(pd.Series([0.0], index=positions.index), positions, tx)
    # Independent quantity*mark oracle: two actual1unit buys at100 mean2units, not1.
    assert original["A"].iloc[0] == (100.0 if same_instant else 200.0)
    assert original["cash"].iloc[0] == (900.0 if same_instant else 800.0)
    assert native.timeline["gross_exposure"].max() == 200.0
    assert native.timeline["cash"][-1] == 800.0
    assert native.timeline["open_positions"].to_list() == [0, 1, 1, 1]
